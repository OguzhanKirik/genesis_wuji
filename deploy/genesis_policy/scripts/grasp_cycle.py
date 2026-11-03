#!/usr/bin/env python3
"""The grasp cycle: perception -> selector -> planned approach -> policy -> planned transport -> home, repeated.

    ros2 run genesis_policy grasp_cycle.py --ros-args -p level_path:=/handoff/policy/L53b/level.yaml \
        -p cycles:=5 -p drop_named_state:=tray_carry

What runs where: adam-moveit's motion skills server owns the arm between episodes (`/move_to`, `/stop`, MoveIt's
collision scene); genesis_policy's selector_node chooses the rod and publishes `/perception/bolt_pose`;
policy_node runs the learned reach-to-grasp from the pre-grasp pose; this node only sequences them:

  1. `rod_selector/select` -> wait for a choice (`rod_selector/choice` JSON, `picked` not null) and a bolt pose;
  2. pre-grasp wrist pose from the rod pose, the choice's hand yaw and level.yaml (genesis_policy.pregrasp);
  3. MoveTo(group, TARGET_POSE, tip_link = the policy's wrist link) -> OK;
  4. policy: heartbeat at 10 Hz + `policy_node/enable true`; watch `policy_node/status` until the runtime is
     done (phase 'done'), aborted, or `policy_timeout_s` passes; then `enable false` (one hold trajectory);
  5. success: MoveTo(named drop state), open the hand (a JointTrajectory to the hand controller at the level's
     open pose), MoveTo(home). Failure: Stop(group), MoveTo(home), and the cycle counts a failure;
  6. repeat `cycles` times (0: until Ctrl-C). Any exception -> Stop(all) and policy disabled.

The controller switch (trajectory controller -> forward command controller during the policy phase) is not
done here: policy_node streams single-point trajectories, as designed for the first runs. Add it when the
JTC interpolation shows in the Isaac rung.
"""
import json
import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Empty, String
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from genesis_policy import level as level_mod
from genesis_policy import pregrasp
from genesis_policy.policy_node import controller_joints

try:
    from genesis_interfaces.action import MoveTo, Stop
except ImportError as e:  # the message package comes with adam-moveit's image
    raise SystemExit(f'genesis_interfaces is not available ({e}): run inside the deploy image built on adam-moveit') from e


class CycleError(RuntimeError):
    pass


class GraspCycle(Node):
    def __init__(self):
        super().__init__('grasp_cycle')
        p = dict(level_path='/handoff/policy/stub/level.yaml', design='genesis_wuji', group='arm_R',
                 world_frame='world', home_named_state='home', drop_named_state='', standoff_m=-1.0,
                 velocity_scaling=0.2, planning_time=5.0, cycles=1, policy_timeout_s=60.0, select_timeout_s=20.0,
                 move_timeout_s=60.0, selector_node='rod_selector', policy_node='policy_node',
                 bolt_pose_topic='/perception/bolt_pose', hand_trajectory_topic='/right_hand_controller/joint_trajectory',
                 hand_open_time_s=1.0, settle_after_move_s=0.5, waist_trajectory_topic='/waist_controller/joint_trajectory',
                 waist_move_time_s=4.0, set_waist=True, move_retries=3, home_first=True)
        for k, v in p.items():
            self.declare_parameter(k, v)
        self.p = {k: self.get_parameter(k).value for k in p}
        self.level = level_mod.load(self.p['level_path'])
        cb = ReentrantCallbackGroup()
        self.choice = None
        self.bolt_pose = None
        self.status = None
        self.create_subscription(String, f"/{self.p['selector_node']}/choice", self._on_choice, 10, callback_group=cb)
        self.create_subscription(PoseStamped, self.p['bolt_pose_topic'], self._on_bolt, 10, callback_group=cb)
        self.create_subscription(String, f"/{self.p['policy_node']}/status", self._on_status, 10, callback_group=cb)
        self.heartbeat = self.create_publisher(Empty, f"/{self.p['policy_node']}/heartbeat", 10)
        self.hand_pub = self.create_publisher(JointTrajectory, self.p['hand_trajectory_topic'], 10)
        self.select_cli = self.create_client(Trigger, f"/{self.p['selector_node']}/select", callback_group=cb)
        self.enable_cli = self.create_client(SetBool, f"/{self.p['policy_node']}/enable", callback_group=cb)
        self.move_to = ActionClient(self, MoveTo, 'move_to', callback_group=cb)
        self.stop = ActionClient(self, Stop, 'stop', callback_group=cb)
        self.hand_joints = controller_joints(self.p['design'], 'right_hand_controller')
        self.waist_joints = controller_joints(self.p['design'], 'waist_controller')
        self.waist_pub = self.create_publisher(JointTrajectory, self.p['waist_trajectory_topic'], 10)
        self._beat = False
        self.create_timer(0.1, self._tick_heartbeat, callback_group=cb)
        self.get_logger().info(f"level {self.level.source}; group {self.p['group']}, wrist {self.level.wrist_link}")

    # ---- callbacks -----------------------------------------------------------------
    def _on_choice(self, msg):
        try:
            self.choice = json.loads(msg.data)
        except ValueError:
            pass

    def _on_bolt(self, msg):
        self.bolt_pose = msg

    def _on_status(self, msg):
        try:
            self.status = json.loads(msg.data)
        except ValueError:
            pass

    def _tick_heartbeat(self):
        if self._beat:
            self.heartbeat.publish(Empty())

    # ---- helpers ---------------------------------------------------------------------
    def _wait(self, cond, timeout, what):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if cond():
                return True
            time.sleep(0.05)
        raise CycleError(f'timeout waiting for {what} ({timeout:.0f} s)')

    def _call(self, client, request, timeout=5.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise CycleError(f'service {client.srv_name} is not up')
        fut = client.call_async(request)
        self._wait(fut.done, timeout, client.srv_name)
        return fut.result()

    def _action(self, client, goal, timeout, name):
        if not client.wait_for_server(timeout_sec=5.0):
            raise CycleError(f'action {name} is not up')
        send = client.send_goal_async(goal)
        self._wait(send.done, 5.0, f'{name} goal acceptance')
        handle = send.result()
        if not handle.accepted:
            raise CycleError(f'{name} goal rejected')
        res = handle.get_result_async()
        self._wait(res.done, timeout, f'{name} result')
        return res.result().result

    def _move(self, goal, name):
        """Send a MoveTo goal; an execution failure (EXEC_FAILED: a controller tripped its tolerance, in sim
        usually the bridge stalling for a frame) is retried `move_retries` times after a Stop and a pause."""
        for attempt in range(1, int(self.p['move_retries']) + 2):
            r = self._action(self.move_to, goal, float(self.p['move_timeout_s']), name)
            if r.error_code == MoveTo.Result.OK:
                return
            if r.error_code != MoveTo.Result.EXEC_FAILED or attempt > int(self.p['move_retries']):
                raise CycleError(f'{name} failed: error_code {r.error_code}')
            self.get_logger().warn(f'{name}: execution failed (attempt {attempt}); stopping and retrying')
            self.stop_group()
            time.sleep(1.5)

    def move_named(self, name):
        goal = MoveTo.Goal()
        goal.group = self.p['group']
        goal.target_type = MoveTo.Goal.TARGET_NAMED_STATE
        goal.named_state = name
        goal.velocity_scaling = float(self.p['velocity_scaling'])
        goal.planning_time = float(self.p['planning_time'])
        self._move(goal, f'move_to({name})')

    def move_pose(self, position, quat_xyzw):
        goal = MoveTo.Goal()
        goal.group = self.p['group']
        goal.target_type = MoveTo.Goal.TARGET_POSE
        goal.pose.header.frame_id = self.p['world_frame']
        goal.pose.pose.position.x, goal.pose.pose.position.y, goal.pose.pose.position.z = map(float, position)
        (goal.pose.pose.orientation.x, goal.pose.pose.orientation.y,
         goal.pose.pose.orientation.z, goal.pose.pose.orientation.w) = map(float, quat_xyzw)
        goal.tip_link = self.level.wrist_link
        goal.velocity_scaling = float(self.p['velocity_scaling'])
        goal.planning_time = float(self.p['planning_time'])
        self._move(goal, 'move_to(pre-grasp)')

    def stop_group(self, group=None):
        goal = Stop.Goal()
        goal.group = group or self.p['group']
        try:
            self._action(self.stop, goal, 10.0, 'stop')
        except CycleError as e:
            self.get_logger().error(str(e))

    def set_policy(self, on):
        self._beat = on
        r = self._call(self.enable_cli, SetBool.Request(data=bool(on)))
        if not r.success:
            raise CycleError(f'policy enable({on}) refused: {r.message}')

    def open_hand(self):
        traj = JointTrajectory()
        traj.joint_names = list(self.hand_joints)
        pt = JointTrajectoryPoint()
        open_map = dict(zip(self.level.hand_joints, self.level.hand_open))
        open_map.update(self.level.hand_held)
        pt.positions = [float(open_map.get(j, 0.0)) for j in self.hand_joints]
        pt.time_from_start.sec = int(self.p['hand_open_time_s'])
        pt.time_from_start.nanosec = int((float(self.p['hand_open_time_s']) % 1) * 1e9)
        traj.points = [pt]
        self.hand_pub.publish(traj)
        time.sleep(float(self.p['hand_open_time_s']) + 0.2)

    def set_waist(self):
        """The torso pose the policy trained with (level.yaml pregrasp.waist, adam-rl WAIST_HELD): the policy
        node never commands the waist, and the arm cannot reach the pre-grasp from an upright torso."""
        raw = self.level.raw if isinstance(getattr(self.level, 'raw', None), dict) else {}
        waist = (raw.get('pregrasp') or {}).get('waist')
        if not waist:
            self.get_logger().warn('level.yaml has no pregrasp.waist: the torso stays where it is')
            return
        traj = JointTrajectory()
        traj.joint_names = list(self.waist_joints)
        pt = JointTrajectoryPoint()
        pt.positions = [float(waist.get(j, 0.0)) for j in self.waist_joints]
        pt.time_from_start.sec = int(self.p['waist_move_time_s'])
        pt.time_from_start.nanosec = int((float(self.p['waist_move_time_s']) % 1) * 1e9)
        traj.points = [pt]
        self.waist_pub.publish(traj)
        self.get_logger().info(f'waist -> {dict(zip(traj.joint_names, pt.positions))} over {self.p["waist_move_time_s"]} s')
        time.sleep(float(self.p['waist_move_time_s']) + 0.5)

    # ---- one cycle ---------------------------------------------------------------------
    def select_rod(self):
        self.choice, self.bolt_pose = None, None
        r = self._call(self.select_cli, Trigger.Request())
        if not r.success:
            raise CycleError(f'selector refused: {r.message}')
        self._wait(lambda: self.choice is not None and (self.choice.get('picked') is not None) and self.bolt_pose is not None,
                   float(self.p['select_timeout_s']), 'a rod choice and its pose')
        return self.choice, self.bolt_pose

    def run_cycle(self, k):
        choice, bp = self.select_rod()
        q = bp.pose.orientation
        x, y, z, w = q.x, q.y, q.z, q.w
        axis = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])
        pos = np.array([bp.pose.position.x, bp.pose.position.y, bp.pose.position.z])
        standoff = float(self.p['standoff_m'])
        pg = pregrasp.pregrasp_pose(pos, axis, float(choice.get('hand_yaw_rad', 0.0)), self.level,
                                    standoff if standoff >= 0 else None)
        self.get_logger().info(f"cycle {k}: rod {choice['picked']} ({pg.kind}) at {np.round(pos, 3).tolist()}, "
                               f"pre-grasp {np.round(pg.position, 3).tolist()}")
        self.move_pose(pg.position, pg.quat_xyzw)
        time.sleep(float(self.p['settle_after_move_s']))
        self.status = None
        self.set_policy(True)
        try:
            self._wait(lambda: self.status is not None and (self.status.get('phase') == 'done' or self.status.get('aborted')),
                       float(self.p['policy_timeout_s']), 'the policy episode')
        finally:
            st = self.status or {}
            self.set_policy(False)
        if st.get('aborted'):
            raise CycleError(f"policy aborted: {st.get('reason')}")
        success = bool(st.get('success', False))
        self.get_logger().info(f"cycle {k}: episode done, success={success}, step {st.get('step')}")
        if success and self.p['drop_named_state']:
            self.move_named(self.p['drop_named_state'])
            self.open_hand()
        else:
            self.open_hand()
        self.move_named(self.p['home_named_state'])
        return success

    def run(self):
        n = int(self.p['cycles'])
        ok = fail = 0
        k = 0
        try:
            if self.p['home_first']:
                # skip in sim when the arm already rests at home: the cached zero-length home plan is aborted by
                # the Isaac bridge's phantom joint velocity (shoulderPitch reads ~0.08 rad/s at rest, tolerance 0.05)
                self.move_named(self.p['home_named_state'])
            if self.p['set_waist']:
                self.set_waist()
            while n == 0 or k < n:
                k += 1
                try:
                    if self.run_cycle(k):
                        ok += 1
                    else:
                        fail += 1
                except CycleError as e:
                    fail += 1
                    self.get_logger().error(f'cycle {k}: {e}')
                    self._beat = False
                    try:
                        self._call(self.enable_cli, SetBool.Request(data=False))
                    except CycleError:
                        pass
                    self.stop_group()
                    time.sleep(1.0)                     # the stop's hold settles before the next goal, or it is PREEMPTED
                    self.move_named(self.p['home_named_state'])
                self.get_logger().info(f'cycles {k}: {ok} picked, {fail} failed')
        finally:
            self._beat = False
            self.stop_group(Stop.Goal.ALL)
        return ok, fail


def main(args=None):
    rclpy.init(args=args)
    node = GraspCycle()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    t = threading.Thread(target=ex.spin, daemon=True)
    t.start()
    try:
        ok, fail = node.run()
        print(f'done: {ok} picked, {fail} failed')
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
