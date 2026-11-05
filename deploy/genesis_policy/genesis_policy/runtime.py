"""The sim-free policy runtime: observation -> actor -> wrist step -> IK ->
joint targets, with the scripted phase machine around it. numpy and the
robot model only; no ROS.

A transliteration of adam-rl's ReachGraspAction / observation terms
(genesis_rl/tasks/wuji_reach_grasp/env_cfg.py, 2026-10-08), one environment
at a time, with the simulator's reads replaced by what the robot reports:

    sim                                 here
    body_link_pos_w / quat_w (wrist)    FK of /joint_states (RobotModel)
    body_link_jacobian_w                pinocchio frame Jacobian, world-aligned
    body_link_lin_vel_w / ang_vel_w     finite difference of the wrist pose
    joint_pos / joint_vel (fingers)     /joint_states (velocity differenced
                                        when the hand driver reports none)
    estimated bolt pose                 /perception/bolt_pose
    contact sensors                     a pluggable TipSource, zeros by default
    the latch (force-based)             a closure-and-stillness rule, see
                                        Level.phase.latch and README

`Runtime.step(joint_states, rod_pose, t)` is deterministic given its inputs
and the actor; the node around it adds the clocks, the topics and the
safety wrapper.
"""
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from . import so3
from .level import GROUP_NAMES, N_ACTION, N_ARM, N_HAND, N_THUMB, TIP_NAMES, Box, Level
from .model import RobotModel

PHASE_GRASP, PHASE_SETTLE, PHASE_PRELIFT, PHASE_LIFT, PHASE_HOLD, PHASE_DONE = range(6)
PHASE_NAMES = ('grasp', 'settle', 'prelift', 'lift', 'hold', 'done')


@dataclass
class JointStates:
    """One /joint_states sample as dictionaries; joints the model does not
    know are ignored, joints it needs and the sample lacks read as zero."""
    position: Dict[str, float]
    velocity: Dict[str, float] = field(default_factory=dict)
    effort: Dict[str, float] = field(default_factory=dict)


@dataclass
class RodPose:
    """The estimated rod pose in world: position and rotation matrix. The
    rod's axis is the pose's z column, as in the simulator's asset."""
    position: np.ndarray
    rotation: np.ndarray


TipSource = Callable[[JointStates], "tuple[np.ndarray, np.ndarray]"]
Actor = Callable[[np.ndarray], np.ndarray]


@dataclass
class Capsule:
    """A segment p0-p1 with a radius (world): a neighbour rod."""
    p0: np.ndarray
    p1: np.ndarray
    radius: float


def clearance(centre, radius, boxes: Sequence[Box], capsules: Sequence[Capsule] = ()):
    """Smallest surface-to-surface distance from a sphere to a set of boxes
    and capsules, and the vector from the sphere surface to that closest point
    (world). Negative inside, exactly as adam-rl's _clearance computes it."""
    best = math.inf
    best_vec = np.zeros(3)
    for cap in capsules:
        seg = cap.p1 - cap.p0
        tt = min(max(float((centre - cap.p0) @ seg) / max(float(seg @ seg), 1e-9), 0.0), 1.0)
        v = cap.p0 + tt * seg - centre
        dist_c = float(np.linalg.norm(v))
        d = dist_c - cap.radius - radius
        if d < best:
            best = d
            best_vec = v / max(dist_c, 1e-9) * d
    for box in boxes:
        closest = np.minimum(np.maximum(centre, box.lo), box.hi)
        v = closest - centre
        dist_c = float(np.linalg.norm(v))
        d = dist_c - radius
        if d < best:
            best = d
            best_vec = v / max(dist_c, 1e-9) * d
    return best, best_vec


def lift_dir_from_rod(rod: RodPose, roll_cap: float, tilt_min: float, tilt_max: float):
    """The approach / lift axis for an episode, from the rod estimate (adam-rl
    reset_bolt_pose): a rod leaning between tilt_min and tilt_max (rad) gets
    the wrist rolled with it and lifts along its up-normal, cos(t) z - sin(t) u
    with u the axis's horizontal direction (raised end) and t = min(tilt,
    roll_cap); a lying or standing rod lifts along world z. The simulator
    knows which case it spawned; here the estimate's tilt decides."""
    a = np.asarray(rod.rotation, float)[:, 2]
    if a[2] < 0.0:
        a = -a                      # a cylinder fit has no preferred end: take the raised one
    tilt = math.asin(min(max(float(a[2]), 0.0), 1.0))
    h = math.hypot(float(a[0]), float(a[1]))
    if not (tilt_min <= tilt <= tilt_max) or h < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    u = np.array([a[0] / h, a[1] / h, 0.0])
    t = min(tilt, roll_cap)
    return math.cos(t) * np.array([0.0, 0.0, 1.0]) - math.sin(t) * u


class RodTracker:
    """The rod pose the runtime steps on, from a camera that loses the rod as soon as the hand covers it.

    Training's estimate (adam-rl `_est_bolt`) was the true pose plus bias, noise, up to 4 steps of latency and
    3 % dropped frames: it followed the rod under the fingers and through the lift, and the grip check, the
    retry and the success test all read its height. A cylinder fit cannot see a rod inside a hand, so:

      measured   a fit fresher than `fresh_s`, while the hand has not latched
      held       no fresh fit, not latched: the last pose, for at most `hold_s` after the last fit (or the
                 last retry); longer than that update() returns None and the node stops as on a stale pose
      attached   latched: the pose rides on the wrist with the wrist-relative pose captured when the latch
                 fired, so a lifted rod rises as the estimate did in training ...
      left       ... unless a fresh fit still finds the rod within `left_radius_m` of where it rested when
                 the latch fired: the hand closed beside it or lifted empty, the fit is reported, its height
                 does not rise, and the grip check and the retry fire as trained

    A retry (the latch cleared) goes back to measured / held, from the last 'left' fit of that latch if there
    was one (the rod is on the table there), else from the attached pose."""

    def __init__(self, fresh_s: float, hold_s: float, left_radius_m: float):
        self.fresh_s, self.hold_s, self.left_radius_m = float(fresh_s), float(hold_s), float(left_radius_m)
        self.reset()

    def reset(self):
        self.pose: Optional[RodPose] = None
        self.source = 'none'
        self.rel = None             # (position, rotation) of the rod in the wrist frame, captured at the latch
        self.rest = None            # where the rod rested when the latch fired
        self.left = None            # the last fit that found the rod still on the table during this latch
        self.hold_from = -math.inf  # the held pose's age counts from here
        self.table = None           # (time, RodPose): the latest candidate near the rest spot, from every visible rod
        self.clear_frames = 0       # candidate frames since the latch with no rod near the rest spot

    def observe_candidates(self, poses, t: float, table_radius_m: float = 0.12, table_rise_m: float = 0.02):
        """Every rod perception sees (not only the selector's choice, which can lose a nudged rod). While latched: a
        rod still at table height within `table_radius_m` of where ours rested is ours, left behind (the fingers
        can push it several cm: 47 mm seen in sim); a frame without one is evidence that it was lifted (or that the
        hand hides the spot: see the README)."""
        if self.rest is None:
            return
        def gap(p):
            q = np.asarray(p.position, float)
            return float(np.linalg.norm(q[:2] - self.rest[:2])), float(q[2] - self.rest[2])
        on_table = [p for p in poses if gap(p)[0] <= table_radius_m and gap(p)[1] < table_rise_m]
        if on_table:
            self.table = (float(t), min(on_table, key=lambda p: gap(p)[0]))
        else:
            self.clear_frames += 1

    def update(self, meas: Optional[RodPose], meas_time: float, t: float, wrist_p, wrist_R, latched: bool):
        """The pose to step on at time t, or None. `meas` is the latest fit, received at `meas_time`."""
        fresh = meas is not None and (float(t) - float(meas_time)) <= self.fresh_s
        if latched:
            if self.rel is None:
                if self.pose is None:
                    return None
                self.rest = np.array(self.pose.position, float).copy()
                self.rel = (wrist_R.T @ (self.pose.position - wrist_p), wrist_R.T @ self.pose.rotation)
                self.table, self.clear_frames = None, 0
            table = self.table[1] if self.table is not None and float(t) - self.table[0] <= self.fresh_s else None
            if fresh and float(np.linalg.norm(np.asarray(meas.position, float) - self.rest)) <= self.left_radius_m:
                table = meas
            if table is not None:
                self.pose, self.source = table, 'left'
                self.left = table
            else:
                self.pose = RodPose(wrist_p + wrist_R @ self.rel[0], wrist_R @ self.rel[1])
                self.source = 'attached'
            return self.pose
        if self.rel is not None:        # a retry: the latch cleared
            if self.left is not None:
                self.pose = self.left
            self.rel, self.rest, self.left = None, None, None
            self.table, self.clear_frames = None, 0
            self.hold_from = float(t)
        if fresh:
            self.pose, self.source = meas, 'measured'
            self.hold_from = float(meas_time)
            return self.pose
        if self.pose is None or float(t) - self.hold_from > self.hold_s:
            self.source = 'stale'
            return None
        self.source = 'held'
        return self.pose


class Runtime:
    def __init__(self, level: Level, model: RobotModel, obstacles: Sequence[Box], table_top: float,
                 actor: Actor, hand_joint_order: Sequence[str],
                 arm_clip_lo: Optional[np.ndarray] = None, arm_clip_hi: Optional[np.ndarray] = None,
                 tip_source: Optional[TipSource] = None, velocity_filter: float = 0.0,
                 finger_velocity: str = 'difference', wrist_velocity: str = 'difference',
                 effort_scale: float = 1.0):
        """
        level            the constants (level.yaml)
        model            RobotModel built for level.arm_joints / wrist
        obstacles        the boxes the safety term measures clearance to
        table_top        z of the table surface (world); the workspace clamp is relative to it
        actor            obs (float32, layout.size) -> action (8), tanh-bounded
        hand_joint_order the 20 right-hand joints in the controller's order
        arm_clip_lo/hi   the limits the IK output is clipped to (MoveIt's);
                         default: level.arm_q_limits
        tip_source       (joint_states) -> (tip_forces[3] in N, contacts[3] in {0,1})
        velocity_filter  first-order filter on the differenced wrist velocity, 0 = none
        finger_velocity  'difference' (finite difference of positions) or 'reported'
        wrist_velocity   'difference' (finite difference of the wrist FK) or 'jacobian'
                         (the full-body J(q) times every reported joint velocity, waist
                         included: what the simulator's body velocity is)
        effort_scale     N m per unit of JointStates.effort, for the torque latch
                         (1 in simulation; the hand's torque constant on the robot)
        """
        self.level = level
        self.model = model
        self.obstacles = list(obstacles)
        self.capsules = []
        self.table_top = float(table_top)
        self.actor = actor
        self.tip_source = tip_source
        self.velocity_filter = float(velocity_filter)
        if finger_velocity not in ('difference', 'reported'):
            raise ValueError("finger_velocity must be 'difference' or 'reported'")
        self.finger_velocity = finger_velocity
        if wrist_velocity not in ('difference', 'jacobian'):
            raise ValueError("wrist_velocity must be 'difference' or 'jacobian'")
        self.wrist_velocity = wrist_velocity
        self.effort_scale = float(effort_scale)
        lt = level.phase.latch
        self._chains = (tuple(level.hand_joints[:N_THUMB]), tuple(level.hand_joints[N_THUMB:N_THUMB + 3]),
                        tuple(level.hand_joints[N_THUMB + 3:]))

        order = tuple(hand_joint_order)
        if len(order) != N_HAND or set(order) != set(level.all_hand_joints):
            raise ValueError(f'hand joint order must be the {N_HAND} joints the level names; '
                             f'got {len(order)}, missing {sorted(set(level.all_hand_joints) - set(order))}, '
                             f'extra {sorted(set(order) - set(level.all_hand_joints))}')
        self.hand_order = order
        self._synergy_slot = [order.index(j) for j in level.hand_joints]
        self._held = [(order.index(j), v) for j, v in level.hand_held.items()]

        lo, hi = (level.arm_q_lo, level.arm_q_hi) if arm_clip_lo is None else (np.asarray(arm_clip_lo, float), np.asarray(arm_clip_hi, float))
        if lo.shape != (N_ARM,) or hi.shape != (N_ARM,) or np.any(hi <= lo):
            raise ValueError('arm clip limits must be 7 values with hi > lo')
        self.clip_lo = lo + level.joint_margin
        self.clip_hi = hi - level.joint_margin
        if np.any(self.clip_hi <= self.clip_lo):
            raise ValueError('joint_margin leaves no room inside the clip limits')
        # the null-space pull and arm_q_norm use the limits training saw
        self.norm_lo, self.norm_hi = level.arm_q_lo, level.arm_q_hi
        self.norm_mid = 0.5 * (self.norm_lo + self.norm_hi)
        self.norm_range = np.maximum(self.norm_hi - self.norm_lo, 1e-6)

        for name, (link, _) in level.tips.items():
            if not model.has_frame(link):
                raise ValueError(f'tip link {link!r} ({name}) is not in the URDF')
        for group, spheres in level.safety_groups.items():
            for s in spheres:
                if not model.has_frame(s.link):
                    raise ValueError(f'safety proxy link {s.link!r} ({group}) is not in the URDF')
        self._started = False

    def set_scene(self, boxes: Sequence[Box], capsules: Sequence[Capsule] = ()):
        """Replace what the safety term measures clearance to: the table and
        the bin as boxes, neighbour rods as capsules. Call before a step when
        the scene estimate changes."""
        self.obstacles = list(boxes)
        self.capsules = list(capsules)

    # ---- episode state -----------------------------------------------------------

    def reset(self, js: JointStates, rod: RodPose, t: float, lift_dir=None, target_seed=None, workspace_centre=None):
        """Start an episode at the measured state. lift_dir: this episode's
        approach / lift axis (world), default the level's (a tilted rod with
        the wrist rolled lifts along the rod's up-normal). target_seed: (p, R)
        the IK target starts from instead of the measured wrist pose (the
        simulator seeds it with the episode's analytic start pose).
        workspace_centre: the xy the workspace clamp is centred on, default the
        rod's xy plus the level's nominal wrist offset (the simulator uses the
        episode's analytic nominal wrist xy)."""
        P = self.level.phase
        q = self.model.configuration(js.position)
        self.model.forward(q)
        p, R = self.model.wrist()
        self.k = 0
        self.t_prev = float(t)
        self.p_prev, self.R_prev = p.copy(), R.copy()
        self.v_lin = np.zeros(3)
        self.v_ang = np.zeros(3)
        self.finger_prev = self._finger_q(js)
        self.raw = np.zeros(N_ACTION)
        self.closure = np.zeros(2)
        self.wrist_cmd = np.zeros(6)
        self.target_p, self.target_R = p.copy(), R.copy()
        if target_seed is not None:
            self.target_p = np.asarray(target_seed[0], float).copy()
            self.target_R = np.asarray(target_seed[1], float).copy()
        self.q_cmd = self.model.arm_of(q).copy()
        self.grasp_end = P.grasp_max_steps + P.settle_in     # adam-rl reset_rg_buffers: the settle-in is not grasp time
        self.latched = False
        self.latch_k = -1
        self.latch_streak = 0
        self.latch_closure = np.zeros(2)
        self.latch_wrist = (p.copy(), R.copy())
        self.attempts = 0
        self.was_up = False
        self.hold_streak = 0
        self.grip_fired = False
        self.pre_bolt_z = math.nan
        self.pre_tip_z = math.nan
        self._retry_now = False
        self.done = False
        self.success = False
        self.rest_z = float(rod.position[2])
        ld = self.level.phase.lift_dir if lift_dir is None else np.asarray(lift_dir, float)
        self.lift_dir = ld / max(float(np.linalg.norm(ld)), 1e-9)
        self.tau_streak = np.zeros(3, dtype=int)
        self.est_prev = np.zeros(3)
        # the previous estimates, newest first, zero at the start as the simulator's buffers are
        self.est_win = np.zeros((max(self.level.phase.latch.est_vel_window, 1), 3))
        self.centre_xy = (np.asarray(rod.position[:2], float) + self.level.nominal_wrist_offset_xy
                          if workspace_centre is None else np.asarray(workspace_centre, float)[:2].copy())
        self._started = True

    @property
    def started(self):
        return self._started

    def end_episode(self):
        """Forget the episode; the next step() starts a new one at the measured state."""
        self._started = False

    def phase(self, k=None):
        """The scripted phase at step k (default: the next step)."""
        P = self.level.phase
        k = self.k if k is None else k
        g = self.grasp_end
        if k < g:
            return PHASE_GRASP
        if k < g + P.settle_steps:
            return PHASE_SETTLE
        if k < g + P.lift_start:
            return PHASE_PRELIFT
        if k < g + P.lift_start + P.lift_steps:
            return PHASE_LIFT
        if k < g + P.scripted_steps:
            return PHASE_HOLD
        return PHASE_DONE

    # ---- pieces ---------------------------------------------------------------------

    def _finger_q(self, js):
        return np.array([js.position.get(j, 0.0) for j in self.level.hand_joints], dtype=float)

    def _tips(self):
        out = {}
        for name, (link, off) in self.level.tips.items():
            p, R = self.model.frame_pose(link)
            out[name] = p + R @ off
        return out

    def _safety(self, R_wT):
        L = self.level
        vecs, dists = [], []
        for group in GROUP_NAMES:
            best, best_vec = math.inf, np.zeros(3)
            for s in L.safety_groups[group]:
                p, R = self.model.frame_pose(s.link)
                d, v = clearance(p + R @ s.offset, s.radius, self.obstacles, self.capsules)
                if d < best:
                    best, best_vec = d, v
            vecs.append(np.clip(R_wT @ best_vec, -L.safety_vec_clamp, L.safety_vec_clamp))
            dists.append(min(max(best, L.safety_dist_clamp[0]), L.safety_dist_clamp[1]))
        return np.concatenate(vecs + [np.array(dists)])

    def _tip_dist(self, tips, rod):
        L = self.level
        axis = rod.rotation[:, 2]
        out = []
        for name in TIP_NAMES:
            v = tips[name] - rod.position
            s = min(max(float(v @ axis), -L.rod_half_len), L.rod_half_len)
            out.append(max(float(np.linalg.norm(v - s * axis)) - L.rod_radius, 0.0))
        return np.array(out)

    def observe(self, js, rod, q_arm, p_w, R_w, finger_q, finger_qd, tips):
        """The observation vector in the level's layout, float32."""
        L = self.level
        R_wT = R_w.T
        terms = {}
        for name in L.layout.names:
            if name == 'bolt_pos_rel':
                terms[name] = R_wT @ (rod.position - p_w)
            elif name == 'bolt_rot_rel':
                R_rel = R_wT @ rod.rotation
                terms[name] = np.concatenate([R_rel[:, 0], R_rel[:, 1]])
            elif name == 'wrist_lin_vel':
                terms[name] = self.v_lin
            elif name == 'wrist_ang_vel':
                terms[name] = self.v_ang
            elif name == 'finger_q':
                terms[name] = finger_q
            elif name == 'finger_qd':
                terms[name] = finger_qd
            elif name == 'last_action':
                terms[name] = self.raw
            elif name in ('tip_forces', 'contacts') and L.effort_obs:
                # adam-rl effort_obs (level 43+): the flexion-chain torque read as a force, and a touch flag
                tau = self._grip_torque(js)
                terms[name] = (np.clip(tau * L.effort_obs[0] / L.tip_force_scale, -5.0, 5.0) if name == 'tip_forces'
                               else (tau > L.effort_obs[1]).astype(float))
            elif name in ('tip_forces', 'contacts'):
                if self.tip_source is None:
                    forces, contacts = np.zeros(3), np.zeros(3)
                else:
                    forces, contacts = self.tip_source(js)
                    forces, contacts = np.asarray(forces, float), np.asarray(contacts, float)
                terms[name] = (np.clip(forces / L.tip_force_scale, -5.0, 5.0) if name == 'tip_forces'
                               else (contacts > 0.0).astype(float))
            elif name == 'ep_time':
                # adam-rl episode_time: counted from the end of the settle-in, over the budget without it
                # (level.yaml episode_max_steps includes the settle-in)
                s_in = L.phase.settle_in
                terms[name] = np.array([max(0, self.k - s_in) / (L.episode_max_steps - s_in)])
            elif name == 'arm_q_norm':
                terms[name] = np.clip(2.0 * (q_arm - self.norm_lo) / self.norm_range - 1.0, -2.0, 2.0)
            elif name == 'tip_dist':
                terms[name] = self._tip_dist(tips, rod)
            elif name == 'safety':
                terms[name] = self._safety(R_wT)
            else:  # the loader rejects unknown names; keep the failure loud
                raise KeyError(name)
        return np.concatenate([terms[n] for n in L.layout.names]).astype(np.float32)

    def _grasp_ready(self, finger_qd):
        """The hardware latch rule: both closures past closure_min and the
        fingers still, for `steps` consecutive steps."""
        lt = self.level.phase.latch
        ok = bool(np.all(self.closure >= lt.closure_min)) and float(np.max(np.abs(finger_qd))) <= lt.finger_qd_max
        self.latch_streak = self.latch_streak + 1 if ok else 0
        return self.latch_streak >= lt.steps

    def _grip_torque(self, js):
        """|effort| summed over each flexion chain: thumb, index, middle (N m)."""
        e = js.effort
        return np.array([sum(abs(e.get(j, 0.0)) for j in chain) for chain in self._chains]) * self.effort_scale

    def _torque_ready(self, js, rod, tips, k):
        """adam-rl's grasp_ready at hw_latch levels, on what the robot measures."""
        lt = self.level.phase.latch
        tau = self._grip_torque(js)
        self.tau_streak = np.where(tau > lt.tau_min_nm, self.tau_streak + 1, 0)
        ts = self.tau_streak >= lt.steps
        stable = bool(ts[0] and (ts[1] or ts[2]))
        gripping = bool(tau[0] > lt.tau_min_nm and max(tau[1], tau[2]) > lt.tau_min_nm)
        d = self._tip_dist(tips, rod)
        inside = 0.5 * (d[0] + min(d[1], d[2])) < lt.grasp_region_m
        if lt.pinch_centre_tol_m > 0.0:
            mid = 0.5 * (tips['thumb'] + 0.5 * (tips['index'] + tips['middle']))
            axis = rod.rotation[:, 2]
            v = mid - rod.position
            inside = inside and float(np.linalg.norm(v - float(v @ axis) * axis)) <= lt.pinch_centre_tol_m
        pos = np.asarray(rod.position, float)
        est_v = (pos - self.est_prev) * self.level.rate_hz
        self.est_prev = pos.copy()
        w = lt.est_vel_window
        if w > 1:
            est_v = (pos - self.est_win[w - 1]) * (self.level.rate_hz / w)
        self.est_win = np.vstack([pos[None], self.est_win[:-1]])
        settled = float(np.linalg.norm(est_v - self.v_lin)) < lt.rel_vel_max_m_s
        # why the latch does or does not fire, for the node's status (the hardware has no other view of it)
        self.latch_diag = dict(tau_nm=[round(float(v), 3) for v in tau], stable=bool(stable), gripping=bool(gripping),
                               inside=bool(inside), tip_dist_mm=[round(1000 * float(v), 1) for v in d],
                               settled=bool(settled))                      # plain bools: the node serialises this
        return stable and gripping and inside and settled

    def _update(self, k, rod, tips):
        """Lift/hold bookkeeping, the grip check and the retry decision for
        step k, before the command (adam-rl's _update, lift section)."""
        P = self.level.phase
        g = self.grasp_end
        if P.sense_est and k == 1:
            self.rest_z = float(rod.position[2])      # the estimate's rest height, read once at the start
        z = float(rod.position[2]) - self.rest_z
        up = z > P.success_lift_m
        self.hold_streak = self.hold_streak + 1 if up else 0
        if k > g:
            self.was_up = self.was_up or z > P.up_threshold_m
        if P.retry_max > 0 and self.attempts < P.retry_max:
            lift_end = k == g + P.lift_start + P.lift_steps
            in_hold = k > g + P.lift_start + P.lift_steps
            need = (lift_end and not up) or (in_hold and self.was_up and z < P.up_threshold_m)
            if P.grip_check:
                tz = float(np.mean([tips[n][2] for n in TIP_NAMES]))
                bz = float(rod.position[2])
                if P.grip_check_late:
                    n_test = P.grip_check_steps if P.free_prelift else P.pre_lift_steps
                    start = k == g + P.lift_start - n_test // 2
                else:
                    start = k == g + P.settle_steps + (P.drag_steps if P.free_prelift else 0)
                if start:
                    self.pre_bolt_z, self.pre_tip_z = bz, tz
                if k == g + P.lift_start and self.latched and math.isfinite(self.pre_tip_z):
                    tip_rise = tz - self.pre_tip_z
                    if tip_rise > P.grip_check_min_tips_m and (bz - self.pre_bolt_z) < P.grip_check_ratio * tip_rise:
                        self.grip_fired = True
                        need = True
            if need:
                self.attempts += 1
                self.grasp_end = k + P.grasp_retry_steps
                g = self.grasp_end
                self.latched = False
                self.latch_k = -1
                self.latch_streak = 0
                self.hold_streak = 0
                self.was_up = False
                self._retry_now = True
        final = k >= g + P.scripted_steps
        self.done = final
        self.success = final and self.hold_streak >= P.hold_steps

    # ---- the step ---------------------------------------------------------------------

    def step(self, js: JointStates, rod: RodPose, t: float, lift_dir=None, target_seed=None, workspace_centre=None):
        """One control step at time t. Returns (arm_targets[7], hand_targets[20], info).
        lift_dir, target_seed and workspace_centre are read only on the step that starts an episode."""
        if not self._started:
            self.reset(js, rod, t, lift_dir, target_seed, workspace_centre)
        L, P = self.level, self.level.phase
        k = self.k
        dt = float(t) - self.t_prev if k > 0 else L.dt
        dt = max(dt, 1e-4)

        q = self.model.configuration(js.position)
        self.model.forward(q)
        q_arm = self.model.arm_of(q)
        p_w, R_w = self.model.wrist()
        tips = self._tips()

        if self.wrist_velocity == 'jacobian':
            self.v_lin, self.v_ang = self.model.wrist_twist(q, js.velocity)
        elif k > 0:
            lin = (p_w - self.p_prev) / dt
            ang = so3.log(R_w @ self.R_prev.T) / dt
            a = self.velocity_filter
            self.v_lin = a * self.v_lin + (1.0 - a) * lin
            self.v_ang = a * self.v_ang + (1.0 - a) * ang
        finger_q = self._finger_q(js)
        if self.finger_velocity == 'reported':
            finger_qd = np.array([js.velocity.get(j, 0.0) for j in L.hand_joints], dtype=float)
        else:
            finger_qd = (finger_q - self.finger_prev) / dt if k > 0 else np.zeros_like(finger_q)

        # bookkeeping on this step's state, before the command
        self._update(k, rod, tips)
        retry_now, self._retry_now = self._retry_now, False
        g = self.grasp_end
        grasping = k < g
        latched_now = False
        if P.latch.mode == 'torque':
            ready = self._torque_ready(js, rod, tips, k)     # every step: the streaks and the estimate's velocity
            ready = ready and k >= max(1, P.settle_in)
        else:
            ready = grasping and not self.latched and self._grasp_ready(finger_qd)
        if grasping and not self.latched and ready:
            latched_now = True
            self.latched = True
            self.latch_k = k
            self.grasp_end = g = k
            grasping = False

        obs = self.observe(js, rod, q_arm, p_w, R_w, finger_q, finger_qd, tips)
        action = np.asarray(self.actor(obs), dtype=float).reshape(-1)
        if action.shape != (N_ACTION,):
            raise ValueError(f'actor returned {action.shape}, expected ({N_ACTION},)')
        action = np.nan_to_num(action, nan=0.0)
        a = np.clip(action, -1.0, 1.0)

        if latched_now:
            # record the measured pose and finger targets, clear the residual,
            # re-seed the IK reference and integrator at the measured state
            self.latch_wrist = (p_w.copy(), R_w.copy())
            self.latch_closure = self.closure.copy()
            self.wrist_cmd[:] = 0.0
            self.target_p, self.target_R = p_w.copy(), R_w.copy()
            self.q_cmd = q_arm.copy()
        if retry_now:
            self.closure[:] = 0.0
            self.wrist_cmd[:] = 0.0
            self.target_p, self.target_R = p_w.copy(), R_w.copy()
            self.q_cmd = q_arm.copy()

        phase = self.phase(k)
        settled = k >= P.settle_in
        t_pre = k - (g + P.settle_steps)
        live = (grasping and settled) or (P.free_prelift and self.latched and 0 <= t_pre < P.drag_steps)
        if not live:
            a[0:6] = 0.0
            self.wrist_cmd[:] = 0.0
        self.raw = action.copy()      # mdp.last_action: the actor's output as given, before the clamp and the mask

        cmd = np.concatenate([a[0:3] * L.effective_translation, a[3:6] * L.effective_rotation])
        cmd[5] *= L.yaw_rate_scale
        self.wrist_cmd += L.wrist_smoothing * (cmd - self.wrist_cmd)

        c = (a[6:8] + 1.0) / 2.0
        if grasping and settled:
            self.closure = self.closure + L.action_smoothing * (c - self.closure)

        d_pos = self.wrist_cmd[0:3].copy()
        d_rot = self.wrist_cmd[3:6].copy()
        lifting = phase == PHASE_LIFT
        t_lift = k - (g + P.lift_start)
        fast = (P.lift_height_m - P.lift_slow_steps * P.lift_slow_step_m) / (P.lift_steps - P.lift_slow_steps)
        lift_step = self.lift_dir * (P.lift_slow_step_m if t_lift < P.lift_slow_steps else fast)
        in_pre = (0 <= t_pre < P.pre_lift_steps) and self.latched
        first = t_pre < P.drag_steps
        if P.grip_check:
            n_test = P.grip_check_steps if P.free_prelift else P.pre_lift_steps
            test_step = self.lift_dir * (P.grip_test_total_m / n_test)
        else:
            test_step = self.lift_dir * P.lift_slow_step_m
        # no cover and no drag on hardware: the first steps hold still unless
        # the grip-check test lift spans the whole pre-lift
        first_step = test_step if (P.grip_check and not P.free_prelift) else np.zeros(3)
        pre_step = first_step if first else test_step
        if P.free_prelift:
            in_pre = in_pre and not first
        if lifting:
            d_pos = lift_step
        elif in_pre:
            d_pos = pre_step
        elif not live:
            d_pos = np.zeros(3)
        if not live:
            d_rot = np.zeros(3)
        if phase == PHASE_DONE:
            d_pos = np.zeros(3)
            d_rot = np.zeros(3)

        if k == g + P.settle_steps:   # the pre-lift and the lift start from the measured pose
            self.target_p, self.target_R = p_w.copy(), R_w.copy()
        self.target_p = self.target_p + d_pos
        self.target_R = so3.exp(d_rot) @ self.target_R

        # workspace clamp around the rod-relative nominal wrist xy, z above the table
        off = self.target_p[:2] - self.centre_xy
        r = float(np.linalg.norm(off))
        self.target_p[:2] = self.centre_xy + off * min(1.0, L.workspace_radius_xy / max(r, 1e-9))
        self.target_p[2] = min(max(self.target_p[2], self.table_top + L.workspace_z_min), self.table_top + L.workspace_z_max)

        # damped least squares with the mid-range null-space pull (apply_actions)
        e_pos = self.target_p - p_w
        e_rot = so3.log(self.target_R @ R_w.T)
        err = np.concatenate([e_pos, e_rot])
        J = self.model.wrist_jacobian(q)
        JJT = J @ J.T
        damped = JJT + (L.dls_damping ** 2) * np.eye(6)
        slack = 2.0 * np.minimum(q_arm - self.norm_lo, self.norm_hi - q_arm) / self.norm_range
        urgency = (1.0 - np.clip(slack, 0.0, 1.0)) ** 2
        dq_null = L.nullspace_gain * urgency * (self.norm_mid - q_arm) / self.norm_range
        rhs = err - J @ dq_null
        dq = J.T @ np.linalg.solve(damped, rhs) + dq_null
        bad = not bool(np.all(np.isfinite(dq)))
        dq = np.nan_to_num(dq, nan=0.0, posinf=0.0, neginf=0.0)
        cap = L.max_joint_velocity * L.dt
        dq = np.clip(dq, -cap, cap)
        q_des = q_arm + dq
        q_clipped = np.clip(q_des, self.clip_lo, self.clip_hi)
        self.q_cmd = q_clipped
        manip = math.sqrt(max(float(np.linalg.det(JJT)), 0.0))

        hand = np.zeros(N_HAND)
        c_t, c_f = self.closure[0], self.closure[1]
        for i, slot in enumerate(self._synergy_slot):
            c_i = c_t if i < N_THUMB else c_f
            hand[slot] = L.hand_open[i] + c_i * (L.hand_closed[i] - L.hand_open[i])
        for slot, value in self._held:
            hand[slot] = value

        info = {
            'step': k, 'phase': PHASE_NAMES[phase], 'live': bool(live), 'latched': self.latched,
            'latch': getattr(self, 'latch_diag', None),
            'reseed': bool(latched_now or retry_now),
            'latched_now': latched_now, 'retry_now': retry_now, 'attempts': self.attempts,
            'grasp_end': int(self.grasp_end), 'done': self.done, 'success': self.success,
            'grip_fired': self.grip_fired, 'closure': self.closure.copy(), 'action': a.copy(),
            'obs': obs, 'wrist_p': p_w.copy(), 'wrist_R': R_w.copy(), 'target_p': self.target_p.copy(),
            'target_R': self.target_R.copy(), 'd_pos': d_pos.copy(), 'track_err': float(np.linalg.norm(e_pos)),
            'track_err_rot': float(np.linalg.norm(e_rot)), 'clipped': int(np.sum(q_clipped != q_des)),
            'ik_failed': bool(bad or manip < L.singularity_threshold), 'manipulability': manip,
            'tips': tips, 'dt': dt,
        }

        self.k = k + 1
        self.t_prev = float(t)
        self.p_prev, self.R_prev = p_w.copy(), R_w.copy()
        self.finger_prev = finger_q
        return q_clipped.copy(), hand, info
