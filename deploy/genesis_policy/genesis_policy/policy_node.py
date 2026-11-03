"""The policy node: /joint_states + /perception/bolt_pose -> Runtime at the
policy rate -> single-point JointTrajectory messages on the right arm and
right hand controllers' topics, behind the safety wrapper.

    ros2 launch genesis_policy policy.launch.py policy_dir:=/handoff/policy/stub dry_run:=true

Nothing moves until the dead-man is satisfied: `~/enable` (std_srvs/SetBool)
set to true AND a heartbeat on `~/heartbeat` (std_msgs/Empty) fresher than
heartbeat_timeout_s. Losing either, a stale joint state, a stale rod pose,
hand effort over the threshold, the measured wrist outside the fence or a
wrist step above the cap aborts: one hold trajectory at the measured pose
goes out, then nothing until the next enable.

The node goes through ros2_control only: the JTC replaces its trajectory on
each message and the hardware plugins' watchdog, slew cap and clamp stay in
the loop. Every ROS-side detail is a parameter (README has the table).
"""
import json
import math
import os

import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, String
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from . import so3
from .actor import OnnxActor
from .cell import load_workcell
from .level import N_HAND, load as load_level
from .model import RobotModel, urdf_string
from .runtime import JointStates, RodPose, Runtime, lift_dir_from_rod

TRUTHY = ('true', '1', 'yes')


def real_mode_gate(confirm):
    """Refuse to start against the real robot without the launch-line
    confirmation, the same two-key rule as stack.launch.py / real.launch.py:
    ADAM_MODE=real in .env is not consent on its own."""
    if os.environ.get('ADAM_MODE', 'sim').lower() == 'real' and not confirm:
        raise RuntimeError(
            'ADAM_MODE=real also requires confirm_real_hardware:=true on the policy launch line. '
            'A mode inherited from .env is not consent: say it on the launch line.')


def _suffix(design):
    return '' if design in ('', 'genesis', None) else f'_{design}'


def _vendored_config(name):
    """A controllers or joint-limits file vendored from adam-moveit (assets/config), or None."""
    from .model import assets_dir
    path = os.path.join(assets_dir(), 'config', name)
    return path if os.path.exists(path) else None


def controller_joints(design, controller):
    """The joint list of `controller` in the design's ros2_controllers YAML."""
    path = _vendored_config(f'ros2_controllers{_suffix(design)}.yaml')
    if path is None:
        from genesis_robot_config.builder import ros2_controllers_path
        path = ros2_controllers_path(design)
    with open(path, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    try:
        return list(raw[controller]['ros__parameters']['joints'])
    except (KeyError, TypeError) as e:
        raise RuntimeError(f'{path}: no joints for controller {controller!r}') from e


class Streamer:
    """Targets -> the controllers. One single-point JointTrajectory per tick
    on each controller's ~/joint_trajectory topic, time_from_start ahead of
    now so the JTC splines from where it is to the target.

    SEAM: a forward_command_controller for the right arm would replace
    send_arm() with one Float64MultiArray publish and need no time; it is not
    added because it claims the same position interfaces as the JTC and the
    two cannot be active together (SIM_TO_REAL.md 3.3). Keep this class the
    only place that knows the wire format.
    """

    def __init__(self, node, arm_topic, hand_topic, arm_joints, hand_joints, time_from_start_s):
        self.arm_pub = node.create_publisher(JointTrajectory, arm_topic, 10)
        self.hand_pub = node.create_publisher(JointTrajectory, hand_topic, 10)
        self.arm_joints = list(arm_joints)
        self.hand_joints = list(hand_joints)
        self.tfs = Duration(sec=int(time_from_start_s), nanosec=int((time_from_start_s % 1.0) * 1e9))
        self.sent = 0

    def _message(self, joints, targets, with_velocity):
        msg = JointTrajectory()
        msg.joint_names = joints
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in targets]
        if with_velocity:
            point.velocities = [0.0] * len(joints)
        point.time_from_start = self.tfs
        msg.points = [point]
        return msg

    def send(self, arm_targets, hand_targets):
        # the arm controller claims a velocity interface: say zero at the
        # point, as the JTC's own goals do; the hand is position only
        self.arm_pub.publish(self._message(self.arm_joints, arm_targets, True))
        self.hand_pub.publish(self._message(self.hand_joints, hand_targets, False))
        self.sent += 1


class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')
        p = self._declare()
        real_mode_gate(p['confirm_real_hardware'])
        self.dry_run = p['dry_run']
        self.state_stale_s = p['state_stale_s']
        self.perception_stale_s = p['perception_stale_s']
        self.heartbeat_timeout_s = p['heartbeat_timeout_s']
        self.hand_effort_abort = p['hand_effort_abort']
        self.fence_lo = np.array([p['fence.x_min'], p['fence.y_min'], p['fence.z_min']])
        self.fence_hi = np.array([p['fence.x_max'], p['fence.y_max'], p['fence.z_max']])
        if np.any(self.fence_hi <= self.fence_lo):
            raise RuntimeError('fence.*_max must exceed fence.*_min')
        self.fence_margin = p['fence_margin']
        self.step_cap_tolerance = p['step_cap_tolerance']
        self.world_frame = p['world_frame']
        self.tilt_roll = (math.radians(p['tilt_roll_min_deg']), math.radians(p['tilt_roll_max_deg']))

        # ---- the handoff ---------------------------------------------------------
        policy_dir = p['policy_dir']
        self.level = load_level(os.path.join(policy_dir, 'level.yaml'))
        self.actor = OnnxActor(os.path.join(policy_dir, 'actor.onnx'), obs_dim=self.level.layout.size)
        self.rate_hz = p['rate_hz'] if p['rate_hz'] > 0.0 else self.level.rate_hz
        self.dt = 1.0 / self.rate_hz

        # ---- the robot, the controllers, the cell ------------------------------------
        design = p['design']
        arm_joints = controller_joints(design, p['arm_controller'])
        hand_joints = controller_joints(design, p['hand_controller'])
        if tuple(arm_joints) != tuple(self.level.arm_joints):
            raise RuntimeError(f'{p["arm_controller"]} joints {arm_joints} differ from level.yaml arm.joints '
                               f'{list(self.level.arm_joints)}')
        if len(hand_joints) != N_HAND or set(hand_joints) != set(self.level.all_hand_joints):
            raise RuntimeError(f'{p["hand_controller"]} joints differ from the 20 level.yaml names')
        self.model = RobotModel(urdf_string(design), self.level.arm_joints, self.level.wrist_link,
                                self.level.wrist_offset)
        clip_lo, clip_hi = self._moveit_limits(design, self.level.arm_joints)
        cell = load_workcell(p['cell_file'] or None, mirror_y=p['cell_mirror_y'])
        if self.level.phase.latch.mode == 'torque' and not p['effort_scale_nm_per_a'] > 0.0:
            # the torque latch compares flexion-chain torque (N m) with the level's tau_min_nm; the hand
            # reports drive current, and a guessed torque constant would latch early or never
            raise RuntimeError('level latches on finger torque: set effort_scale_nm_per_a (the Wuji drives\' '
                               'torque constant, measured at bring-up, SIM_TO_REAL.md 3.4)')
        scene = p['safety_scene']
        if scene == 'training' and not self.level.training_boxes:
            # the stub and exports before 2026-10-14 carry no training bin: measure the real furniture instead
            self.get_logger().warn("safety_scene 'training' but level.yaml exports no safety.training_boxes: "
                                   "using workcell.json's table and bin")
            scene = 'workcell'
        if scene == 'training':
            obstacles = [cell.table, *self.level.training_boxes, *self.level.extra_boxes]
        elif scene == 'workcell':
            obstacles = cell.obstacles(self.level.extra_boxes)
        else:
            raise RuntimeError(f"safety_scene must be 'workcell' or 'training', got {p['safety_scene']!r}")
        self.runtime = Runtime(self.level, self.model, obstacles, cell.table_top,
                               self.actor, hand_joints, clip_lo, clip_hi,
                               tip_source=None,   # tactile=False levels feed zeros; the Wuji hand has no tip sensors
                               velocity_filter=p['velocity_filter'], finger_velocity=p['finger_velocity'],
                               wrist_velocity=p['wrist_velocity'], effort_scale=p['effort_scale_nm_per_a'])
        self.hand_joints = hand_joints
        self.arm_joints = arm_joints

        # ---- topics ------------------------------------------------------------------
        self.js = None
        self.js_time = -math.inf
        self.rod = None
        self.rod_time = -math.inf
        self.heartbeat_time = -math.inf
        self.enabled = False
        self.aborted = False
        self.abort_reason = ''
        self.prev_arm_targets = None
        self.last_info = {}
        best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, p['joint_states_topic'], self._on_joint_states, 10)
        self.create_subscription(PoseStamped, p['bolt_pose_topic'], self._on_bolt_pose, 10)
        self.create_subscription(Empty, '~/heartbeat', self._on_heartbeat, best_effort)
        self.create_service(SetBool, '~/enable', self._on_enable)
        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.streamer = Streamer(self, p['arm_trajectory_topic'], p['hand_trajectory_topic'],
                                 arm_joints, hand_joints, p['time_from_start_s'])
        self.timer = self.create_timer(self.dt, self._tick)
        self.status_every = max(1, int(round(self.rate_hz / 5.0)))
        self.ticks = 0
        self.get_logger().info(
            f'policy {policy_dir} ({self.level.source}): {self.level.layout.size} observations at {self.rate_hz:g} Hz, '
            f'{"DRY RUN, publishing nothing" if self.dry_run else "streaming to " + p["arm_trajectory_topic"] + " and " + p["hand_trajectory_topic"]}; '
            f'waiting for ~/enable true and ~/heartbeat')

    # ---- parameters ---------------------------------------------------------------

    def _declare(self):
        defaults = {
            'policy_dir': '/handoff/policy/stub',
            'design': 'genesis_wuji',
            'cell_file': '',
            'cell_mirror_y': False,
            'rate_hz': 0.0,
            'joint_states_topic': '/joint_states',
            'bolt_pose_topic': '/perception/bolt_pose',
            'arm_controller': 'right_arm_controller',
            'hand_controller': 'right_hand_controller',
            'arm_trajectory_topic': '/right_arm_controller/joint_trajectory',
            'hand_trajectory_topic': '/right_hand_controller/joint_trajectory',
            'time_from_start_s': 0.06,
            'dry_run': False,
            'state_stale_s': 0.5,
            'perception_stale_s': 0.5,
            'heartbeat_timeout_s': 0.5,
            'hand_effort_abort': 1.5,
            'fence.x_min': 0.25, 'fence.x_max': 0.85,
            'fence.y_min': -0.55, 'fence.y_max': 0.15,
            'fence.z_min': 0.95, 'fence.z_max': 1.45,
            'fence_margin': 0.02,
            'step_cap_tolerance': 1.5,
            'velocity_filter': 0.0,
            'finger_velocity': 'difference',
            'wrist_velocity': 'jacobian',
            'safety_scene': 'training',
            'tilt_roll_min_deg': 10.0,
            'tilt_roll_max_deg': 60.0,
            'effort_scale_nm_per_a': 0.0,
            'world_frame': 'world',
            'confirm_real_hardware': False,
        }
        out = {}
        for name, default in defaults.items():
            self.declare_parameter(name, default)
            out[name] = self.get_parameter(name).value
        return out

    @staticmethod
    def _moveit_limits(design, joints):
        """The MoveIt joint limits (the firmware values), narrower than the
        ros2_control ones for shoulderPitch; the IK output is clipped to these."""
        path = _vendored_config(f'joint_limits{_suffix(design)}.yaml')
        if path is None:
            from genesis_robot_config.builder import config_path, design_suffix
            path = config_path('moveit', f'joint_limits{design_suffix(design)}.yaml')
        with open(path, encoding='utf-8') as f:
            limits = yaml.safe_load(f)['joint_limits']
        lo, hi = [], []
        for j in joints:
            entry = limits[j]
            if not entry.get('has_position_limits', False):
                raise RuntimeError(f'{path}: {j} has no position limits')
            lo.append(float(entry['min_position']))
            hi.append(float(entry['max_position']))
        return np.array(lo), np.array(hi)

    # ---- callbacks ----------------------------------------------------------------

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_joint_states(self, msg):
        pos = dict(zip(msg.name, msg.position))
        vel = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        eff = dict(zip(msg.name, msg.effort)) if msg.effort else {}
        self.js = JointStates(pos, vel, eff)
        self.js_time = self._now()

    def _on_bolt_pose(self, msg):
        frame = msg.header.frame_id
        if frame and frame != self.world_frame:
            # the contract (genesis_cell) reports in the planning frame; a
            # pose in another frame is refused rather than silently used
            if not getattr(self, '_frame_warned', False):
                self.get_logger().error(f'{frame!r} bolt pose ignored: the policy reads poses in {self.world_frame!r}')
                self._frame_warned = True
            return
        p = msg.pose.position
        q = msg.pose.orientation
        try:
            R = so3.from_quat_xyzw((q.x, q.y, q.z, q.w))
        except ValueError:
            return
        self.rod = RodPose(np.array([p.x, p.y, p.z]), R)
        self.rod_time = self._now()

    def _on_heartbeat(self, _msg):
        self.heartbeat_time = self._now()

    def _on_enable(self, request, response):
        if request.data:
            self.enabled = True
            self.aborted = False
            self.abort_reason = ''
            self.prev_arm_targets = None
            self.runtime.end_episode()         # a fresh episode on the next tick
            response.success = True
            response.message = 'enabled; streaming starts with the next fresh heartbeat'
            self.get_logger().info(response.message)
        else:
            self._hold('disabled by the operator', error=False)
            self.enabled = False
            response.success = True
            response.message = 'disabled; one hold trajectory sent' if not self.dry_run else 'disabled'
        return response

    # ---- the wrapper ------------------------------------------------------------------

    def _hold(self, reason, error=True):
        """Publish one hold at the measured pose and stop."""
        (self.get_logger().error if error else self.get_logger().info)(f'HOLD: {reason}')
        self.aborted = error
        self.abort_reason = reason
        if self.dry_run or self.js is None:
            return
        arm = [self.js.position.get(j, 0.0) for j in self.arm_joints]
        hand = [self.js.position.get(j, 0.0) for j in self.hand_joints]
        self.streamer.send(arm, hand)

    def _checks(self, now):
        """The reasons not to step this tick, in the order they are checked."""
        if self.js is None or now - self.js_time > self.state_stale_s:
            return f'joint state stale ({now - self.js_time:.2f} s > {self.state_stale_s} s)'
        if self.rod is None or now - self.rod_time > self.perception_stale_s:
            return f'rod pose stale ({now - self.rod_time:.2f} s > {self.perception_stale_s} s)'
        if now - self.heartbeat_time > self.heartbeat_timeout_s:
            return f'no heartbeat for {now - self.heartbeat_time:.2f} s (> {self.heartbeat_timeout_s} s)'
        if self.js.effort:
            worst = max((abs(self.js.effort.get(j, 0.0)), j) for j in self.hand_joints)
            if worst[0] > self.hand_effort_abort:
                return f'hand effort {worst[0]:.2f} on {worst[1]} above {self.hand_effort_abort}'
        missing = [j for j in self.arm_joints if j not in self.js.position]
        if missing:
            return f'/joint_states lacks {missing}'
        return None

    def _tick(self):
        self.ticks += 1
        if not self.enabled or self.aborted:
            self._status()
            return
        now = self._now()
        reason = self._checks(now)
        if reason is not None:
            if self.runtime.started:
                self._hold(reason)
                self._status(force=True)
            else:
                self.get_logger().warn(f'not starting: {reason}', throttle_duration_sec=2.0)
                self._status()
            return

        lift_dir = None
        if not self.runtime.started:
            # the episode's approach / lift axis from the rod estimate at the start (a leaning rod: its up-normal)
            lift_dir = lift_dir_from_rod(self.rod, self.level.phase.roll_cap, *self.tilt_roll)
        arm, hand, info = self.runtime.step(self.js, self.rod, now, lift_dir=lift_dir)
        self.last_info = info

        # the fence: the measured wrist must be inside, the commanded target too
        wrist = info['wrist_p']
        if np.any(wrist < self.fence_lo - self.fence_margin) or np.any(wrist > self.fence_hi + self.fence_margin):
            self._hold(f'measured wrist {np.round(wrist, 3).tolist()} outside the fence')
            self._status(force=True)
            return
        if np.any(info['target_p'] < self.fence_lo) or np.any(info['target_p'] > self.fence_hi):
            self._hold(f'commanded wrist {np.round(info["target_p"], 3).tolist()} outside the fence')
            self._status(force=True)
            return
        # the step cap: the wrist may travel at most one policy step (the
        # per-axis bound times sqrt(3)) or one fast lift step per tick, measured
        # by FK between the previous commanded targets and the new ones. A
        # larger step is scaled down to the cap; if that still exceeds it (the
        # scaling is linear in joint space) the tick aborts.
        prev = self.prev_arm_targets if self.prev_arm_targets is not None else \
            np.array([self.js.position[j] for j in self.arm_joints])
        cap = max(self.level.effective_translation * math.sqrt(3.0),
                  self.level.phase.lift_height_m / max(self.level.phase.lift_steps - self.level.phase.lift_slow_steps, 1))
        moved = self._wrist_travel(prev, arm)
        if moved > cap:
            arm = prev + (np.asarray(arm) - prev) * (cap / moved)
            moved2 = self._wrist_travel(prev, arm)
            self.get_logger().warn(f'step cap: wrist step {moved * 1000:.1f} mm scaled to {moved2 * 1000:.1f} mm',
                                   throttle_duration_sec=1.0)
            if moved2 > cap * self.step_cap_tolerance:
                self._hold(f'wrist step {moved2 * 1000:.1f} mm exceeds the cap {cap * 1000:.1f} mm')
                self._status(force=True)
                return
        # the joint clamp (the runtime clips; this is the wrapper's own check)
        if np.any(arm < self.runtime.clip_lo - 1e-9) or np.any(arm > self.runtime.clip_hi + 1e-9):
            self._hold('arm target outside the joint limits')
            self._status(force=True)
            return
        self.prev_arm_targets = np.array(arm, dtype=float)

        if not self.dry_run:
            self.streamer.send(arm, hand)
        self._status()

    def _wrist_travel(self, arm_a, arm_b):
        q = self.model.configuration(self.js.position)
        q[self.model.arm_q] = arm_a
        self.model.forward(q)
        pa, _ = self.model.wrist()
        q[self.model.arm_q] = arm_b
        self.model.forward(q)
        pb, _ = self.model.wrist()
        return float(np.linalg.norm(pb - pa))

    def _status(self, force=False):
        if self.ticks % self.status_every and not force:
            return
        info = self.last_info
        msg = String()
        msg.data = json.dumps({
            'enabled': self.enabled, 'aborted': self.aborted, 'reason': self.abort_reason,
            'dry_run': self.dry_run, 'started': self.runtime.started, 'published': self.streamer.sent,
            'step': int(info.get('step', -1)), 'phase': info.get('phase', ''),
            'attempts': int(info.get('attempts', 0)), 'latched': bool(info.get('latched', False)),
            'done': bool(info.get('done', False)), 'success': bool(info.get('success', False)),
            'track_err_mm': round(float(info.get('track_err', 0.0)) * 1000.0, 2),
            'ik_failed': bool(info.get('ik_failed', False)),
            'wrist': [round(float(v), 4) for v in info.get('wrist_p', [])],
        })
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PolicyNode()
    except Exception as e:  # noqa: BLE001 - say why and fail closed
        print(f'[policy_node] refusing to start: {e}', flush=True)
        rclpy.try_shutdown()
        return 2
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
