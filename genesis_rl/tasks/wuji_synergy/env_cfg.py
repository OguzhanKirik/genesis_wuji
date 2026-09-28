# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experiment 2: finger-only synergy grasp of one bolt with a fixed Wuji hand.

The hand alone (assets/robots/wuji_hand_right/, cut out of the Adam-U + Wuji
URDF by tools/make_wuji_hand_urdf.py) is held in a fixed, thumb-opposed
pre-grasp around one bolt on a table. No arm, no wrist control, no images, no
randomisation. The question is only whether PPO can learn the finger closure
that holds the bolt.

Episode (4 s, 30 Hz control, 120 steps), a fixed state machine driven by the
action term from `episode_length_buf`:

  phase 1  GRASP   steps   0- 59 (2 s)  policy drives two synergies
  phase 2  LIFT    steps  60- 89 (1 s)  finger targets frozen, the root
                                         prismatic joint `wrist_lift` ramps
                                         +LIFT_HEIGHT (10 cm); actions ignored
  phase 3  HOLD    steps  90-119 (1 s)  everything frozen; success is judged
                                         at the last step

Actions (2, each in [-1, 1] -> closure c in [0, 1], first-order smoothed):
  a[0] thumb synergy    r_thumb_cmc_flex, r_thumb_mcp, r_thumb_ip
  a[1] finger synergy   mcp_flex, pip, dip of each ACTIVE finger
  q_target = q_open + c * (q_closed - q_open), per joint, both ends a fixed
  fraction of that joint's flexion limit (OPEN_CURL, CLOSED_CURL). Everything
  else is held: thumb opposition (THUMB_ABD), all abduction, and the fingers
  outside the synergy at HELD_CURL.

NUM_FINGERS (env var ADAM_U_FINGERS) picks how many fingers oppose the thumb:
2 = index only, 3 = index + middle (the spec's synergy, the default), 4 = all
four, which is the whole-hand rake Experiment 1 used.

Observations (normalised by rsl-rl's running normaliser):
  finger joint positions          one per synergy joint (9 for 3 fingers)
  finger joint velocities         one per synergy joint
  previous policy action          2
  fingertip force on the bolt     3, thumb / index / middle, scaled by F_MAX.
                                  The sim reports ONE aggregate force per
                                  fingertip rather than a tactile array, so the
                                  spec's per-fingertip "maximum" and "total"
                                  force are the same number and appear once.
  fingertip contact flags         3, thumb / index / middle

Reward (every component logged by rsl-rl as Episode_Reward/<name>):
  +0.5  first STABLE thumb-bolt contact                     (once per episode)
  +0.5  first STABLE index- or middle-bolt contact          (once)
  +1.0  first simultaneous opposing stable contact          (once)
  +20   success at the last step
  -5    drop: the bolt was up (> 2 cm) after the grasp phase and came back down
  -3    first finger-table collision (> TABLE_HIT_N)        (once)
  -3    first excessive fingertip force (> F_MAX)           (once)
  -0.01 * ||a_t - a_{t-1}||^2                               (every step)
  "Stable" means the contact has persisted STABLE_STEPS control steps, so a
  one-step spike earns nothing. The bonuses are paid once rather than per step,
  so 60 steps of contact cannot outweigh the +20 terminal reward.

Success needs all of: the bolt >= SUCCESS_LIFT above its rest height for the
whole hold phase; the bolt never touching the table during that phase; no
finger-table collision and no excessive fingertip force anywhere in the
episode. Height alone is not enough.

Contacts come from five fingertip ContactSensors and one on the bolt, each
filtered against the table's collider. A fingertip's force on the BOLT is the
remainder of its net force (see _tip_forces); the bolt's own sensor gives the
bolt-table contact directly and is self-validating, since at reset the bolt
rests on the table and must report its weight.

Everything the spec called configurable is a module constant or an
action-term cfg field.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.configclass import configclass

import isaaclab.envs.mdp as mdp
from genesis_rl.tasks.wuji_reach.env_cfg import (
    ADAM_U_ASSETS,
    BOLT_LYING_ROT,
    BOLT_REST_OFFSET,
    LAYOUT,
    _static_box,
    _t,
)

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------

import os

def _hand_usd(fingers: int) -> str:
    """One asset per finger count: the lift axis is baked into the URDF (see POSE)."""
    n = 3 if fingers == 4 else fingers            # 4 reuses the 3-finger pose and asset
    return os.path.join(ADAM_U_ASSETS, "robots", "wuji_hand_right",
                        f"usd_f{n}", f"wuji_hand_right_f{n}.usd")

CONTROL_HZ = 30
PHYSICS_HZ = 120
GRASP_S, LIFT_S, HOLD_S = 2.0, 1.0, 1.0
GRASP_STEPS = int(GRASP_S * CONTROL_HZ)          # 60
LIFT_STEPS = int(LIFT_S * CONTROL_HZ)            # 30
HOLD_STEPS = int(HOLD_S * CONTROL_HZ)            # 30
EPISODE_STEPS = GRASP_STEPS + LIFT_STEPS + HOLD_STEPS
LIFT_HEIGHT = 0.10                               # m, scripted vertical lift
SUCCESS_LIFT = 0.05                              # m above rest, for the whole hold phase
UP_THRESHOLD = 0.02                              # m above rest, "lifted at all" (drop detection)
F_MAX = 40.0                                     # N, excessive fingertip force. The spec gave no number; 5 N and
                                                 # 20 N were tripped by every scripted pinch that held the bolt
                                                 # (position drives, 2 N.m at the MCP, squeeze at ~25 N)
CONTACT_N = 0.1                                  # N, a contact flag turns on above this
STABLE_STEPS = 4                                 # control steps a contact must persist to count as STABLE.
                                                 # The spec asks for 3-5 and rules out rewarding a brief spike.
BOLT_HALF_EXTENT = 0.0175                        # m, half the bolt's longest dimension (3.5 cm),
                                                 # i.e. the radius of a sphere that contains it
TABLE_HIT_N = 15.0                               # N, finger-table COLLISION. Light resting contact (the fingertips
                                                 # sit on the table in every pose that grasps) is not a collision;
                                                 # ploughing into it (20-44 N measured with the hand too low) is.
ACTION_SMOOTHING = 0.3                           # closure += alpha * (command - closure) per step

TABLE_TOP = LAYOUT["table_top"]                  # 0.95
BOLT_REST_Z = TABLE_TOP + BOLT_REST_OFFSET       # 0.961
# Hand pose: a THUMB-OPPOSED PINCH, solved rather than guessed.
#
# tools/find_pinch_pose.py walks the synergy stroke with URDF forward
# kinematics, measures where the thumb tip and the opposing fingertips actually
# converge, and then solves for the hand's ORIENTATION -- the variable that
# decides whether a pinch is possible at all. Every other pinch metric (the
# gap, the opposition angle, the fingertip-to-bolt distance) is independent of
# how the hand is rotated; only table clearance is not. So the tool places the
# wrist such that first contact happens at the bolt, then searches the sphere
# of "which way is up in the wrist frame" for a direction that keeps every
# fingertip above the table over the whole stroke.
#
# Why this replaced the first pose. Fingers-down and palm-down both fail from a
# FIXED wrist: the fingers extend ~17 cm below the wrist when open and curl up
# to ~9 cm as they close, so any wrist height that puts the closing pinch at
# the bolt buries the open fingertips up to 10 cm under the table. Experiment 1
# only worked because its arm rose while the hand closed. The winning pose is
# tilted, coming at the bolt from the side.
#
# Measured for this configuration (3 fingers, thumb cmc_abd -0.85, curls
# 0.40 -> 0.60 of the flexion limits):
#     thumb-vs-(index+middle) gap   5.2 cm open -> 1.2 cm closed
#     contact with the 2.5 cm bolt  at 38 % of the stroke, then 13 mm of squeeze
#     opposition angle              97 deg (the tips travel into each other)
#     fingertip to bolt at reset    2.9 cm to its centre = ~1.6 cm to its surface
#     lowest fingertip clearance    1.1 cm above the table, over the whole stroke
# Re-derive with:  python3 tools/find_pinch_pose.py --middle
# One solved pose per finger count, from tools/find_pinch_pose.py. Each is the
# configuration whose synergy stroke opens clear of the 2.5 cm bolt, makes first
# contact with it mid-stroke, squeezes past it, and keeps every fingertip above
# the table throughout.
#
#   lift_axis    the prismatic lift joint's axis, expressed in the HAND's frame.
#                A joint axis rotates with its parent, and the env spawns this
#                articulation rotated into the pre-grasp orientation, so a naive
#                (0, 0, 1) travels wherever the rotation points it: measured,
#                the wrist went sideways and dragged the bolt 7 cm across the
#                table while reporting a 10 cm lift. lift_axis = HAND_ROT^-1 . z
#                makes the joint travel exactly world-vertical. It is baked into
#                the URDF, so CHANGING THE FINGER COUNT NEEDS A REBUILD:
#                   python3 tools/make_wuji_hand_urdf.py --fingers N
#                   <then export_usd_wuji.py; see its docstring>
#                __post_init__ asserts the two agree rather than let it slide.
POSE = {
    3: dict(  # thumb vs index + middle -- the spec's synergy, and the default
        thumb_abd=-0.85, open_curl=0.40, closed_curl=0.60,
        lift_axis=(0.510345, -0.849959, -0.130833),
        hand_rot=(-0.6447, -0.3871, 0.0000, 0.6592),
        wrist_height=0.0592, bolt_offset=(0.0901, -0.0474), bolt_yaw_deg=-136.3,
        # gap 5.2 -> 1.2 cm, contact at 38 % of the stroke, 13 mm of squeeze,
        # opposition 97 deg, tips 2.9 cm from the bolt centre at reset,
        # lowest fingertip 1.1 cm above the table over the whole stroke
    ),
    2: dict(  # thumb vs index alone -- a true two-finger pinch
        thumb_abd=-0.65, open_curl=0.35, closed_curl=0.55,
        lift_axis=(0.158346, -0.987375, 0.004167),
        hand_rot=(-0.6967, -0.1117, 0.0000, 0.7086),
        wrist_height=0.0840, bolt_offset=(0.0509, -0.0826), bolt_yaw_deg=-159.7,
        # gap 5.2 -> 1.4 cm, contact at 38 %, 11 mm of squeeze, opposition 94.4 deg,
        # tips 2.8 cm from the bolt centre, clearance 1.2 cm
    ),
}
# 4 (the whole-hand rake Experiment 1 used) reuses the 3-finger pose: the extra
# fingers join the synergy instead of being held flat, which the pose search did
# not validate for table clearance. Treat it as a comparison, not a solved pose.
POSE[4] = POSE[3]


THUMB_JOINTS = ["r_thumb_cmc_flex", "r_thumb_mcp", "r_thumb_ip"]
# How many fingers oppose the thumb. The spec asks for index+middle (3) with
# the ring and little held; 4 is the whole-hand rake that Experiment 1 used and
# that this env first shipped with. ADAM_U_FINGERS overrides it.
#   2  thumb + index               a true two-finger pinch
#   3  thumb + index + middle      the spec's synergy
#   4  thumb + all four fingers    rake; ring/little join the synergy
NUM_FINGERS = int(os.environ.get("ADAM_U_FINGERS", "3"))
if NUM_FINGERS not in (2, 3, 4):
    raise ValueError(f"ADAM_U_FINGERS must be 2, 3 or 4, got {NUM_FINGERS}")
_ACTIVE = {2: ["index"], 3: ["index", "middle"],
           4: ["index", "middle", "ring", "pinky"]}[NUM_FINGERS]
_HELD = [f for f in ("index", "middle", "ring", "pinky") if f not in _ACTIVE]
_LONG = {"index": "index_finger", "middle": "middle_finger",
         "ring": "ring_finger", "pinky": "pinky"}


def _finger_joints(names):
    out = []
    for f in names:
        base = f"r_{_LONG[f]}"
        out += [f"{base}_mcp_flex", f"{base}_pip", f"{base}_dip"]
    return out


FINGER_JOINTS = _finger_joints(_ACTIVE)

_P = POSE[NUM_FINGERS]
LIFT_AXIS = _P["lift_axis"]
THUMB_ABD = _P["thumb_abd"]                      # thumb opposition, held fixed
OPEN_CURL = _P["open_curl"]                      # synergy ends, as a fraction of each
CLOSED_CURL = _P["closed_curl"]                  # joint's flexion limit
HAND_ROT = _P["hand_rot"]                        # (x, y, z, w)
WRIST_XY = (0.45, -0.02)
WRIST_HEIGHT = _P["wrist_height"]                # m above the table top
BOLT_XY = (WRIST_XY[0] + _P["bolt_offset"][0], WRIST_XY[1] + _P["bolt_offset"][1])
BOLT_YAW_DEG = _P["bolt_yaw_deg"]                # bolt axis across the pinch's closing direction
SYNERGY_JOINTS = THUMB_JOINTS + FINGER_JOINTS
_F = "|".join(f"r_{_LONG[f]}" for f in _ACTIVE)          # regex alternation over the active fingers
_FH = "|".join(f"r_{_LONG[f]}" for f in _HELD)
# Flexion limits, and the two synergy ends as fractions of them. Isaac Lab
# raises on a joint that matches two regex keys, so these must not overlap the
# held-finger entries in Q_FIXED.
_LIMIT = {"r_thumb_cmc_flex": 1.291, "r_thumb_mcp": 1.570, "r_thumb_ip": 1.570,
          f"({_F})_mcp_flex": 1.570, f"({_F})_pip": 2.094, f"({_F})_dip": 1.570}
Q_OPEN = {k: round(v * OPEN_CURL, 4) for k, v in _LIMIT.items()}
Q_CLOSED = {k: round(v * CLOSED_CURL, 4) for k, v in _LIMIT.items()}
# Joints the policy does not touch, held at fixed targets.
# Curl the fingers outside the synergy are held at. 0.0 (flat) is what
# tools/find_pinch_pose.py validates the table clearance against -- curling
# them brings their tips DOWN toward the table in this tilted pose, and 0.30
# put them on it.
HELD_CURL = 0.0
Q_FIXED = {
    "r_thumb_cmc_abd": THUMB_ABD,                # opposition; see the pose note above
    "r_index_finger_mcp_abd": 0.0, "r_middle_finger_mcp_abd": 0.0,
    "r_ring_finger_mcp_abd": 0.0, "r_pinky_mcp_abd": 0.0,
    **({f"({_FH})_mcp_flex": round(1.570 * HELD_CURL, 4),
        f"({_FH})_pip": round(2.094 * HELD_CURL, 4),
        f"({_FH})_dip": round(1.570 * HELD_CURL, 4)} if _HELD else {}),
}
LIFT_JOINT = "wrist_lift"
TIP_BODIES = {"thumb": "r_thumb_distal", "index": "r_index_finger_distal", "middle": "r_middle_finger_distal",
              "ring": "r_ring_finger_distal", "pinky": "r_pinky_distal"}


def _yawed_lying_rot(yaw_deg: float):
    """Bolt lying on its side (90 deg about X), then yawed about world Z. (x, y, z, w)."""
    cy, sy = math.cos(math.radians(yaw_deg) / 2), math.sin(math.radians(yaw_deg) / 2)
    qz = (0.0, 0.0, sy, cy)
    qx = BOLT_LYING_ROT
    # q = qz * qx
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qx
    return (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)


# ---------------------------------------------------------------------------
# Per-episode bookkeeping shared by the action term, rewards and evaluation
# ---------------------------------------------------------------------------


def _buffers(env) -> dict:
    """Lazily created per-env flags/streaks. Reset by `reset_synergy_buffers`."""
    b = getattr(env, "_syn", None)
    if b is None:
        n, d = env.num_envs, env.device
        z = lambda: torch.zeros(n, dtype=torch.bool, device=d)  # noqa: E731
        f = lambda k=1: torch.zeros((n, k) if k > 1 else (n,), device=d)  # noqa: E731
        b = dict(
            # this step: raw contact, the STABLE version (>= STABLE_STEPS in a
            # row, so a one-step spike earns nothing), and the forces behind them
            thumb_contact=z(), finger_contact=z(),
            thumb_stable=z(), finger_stable=z(),
            streak=f(3),                                       # thumb, index, middle contact streaks
            tip_force=f(3),                                    # thumb, index, middle bolt-contact force [N]
            paid_thumb=z(), paid_finger=z(), paid_opposing=z(),
            table_hit=z(), excess=z(), paid_table=z(), paid_excess=z(),
            bolt_on_table=z(),                                 # bolt touching the table THIS step
            verify_clean=z(),                                  # bolt never touched the table during verification
            was_up=z(), paid_drop=z(),
            hold_streak=torch.zeros(n, device=d),
            success=z(),
            lift_start=torch.zeros(n, device=d),               # wrist_lift q at reset
            # Outcome of the episode that just ENDED, written at its last step and
            # NOT cleared by the reset (which happens inside the same step()), so
            # evaluation can read it afterwards.
            last_success=z(), last_held=z(), last_table_hit=z(), last_excess=z(), last_drop=z(), last_was_up=z(),
            last_thumb_frac=torch.zeros(n, device=d), last_finger_frac=torch.zeros(n, device=d),
            last_closure=torch.zeros(n, 2, device=d), last_verify_clean=z(),
            thumb_steps=torch.zeros(n, device=d), finger_steps=torch.zeros(n, device=d),
        )
        env._syn = b
    return b


def reset_synergy_buffers(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    b = _buffers(env)
    for k, v in b.items():
        if k in ("lift_start", "_step") or k.startswith("last_"):
            continue
        v[env_ids] = 0
    robot = env.scene[asset_cfg.name]
    lift_id, _ = robot.find_joints([LIFT_JOINT])
    b["lift_start"][env_ids] = _t(robot.data.joint_pos)[env_ids, lift_id[0]]
    b["verify_clean"][env_ids] = True          # innocent until the bolt touches the table in phase 3


def hold_default_targets(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    default = _t(robot.data.default_joint_pos)[env_ids]
    robot.set_joint_position_target(default, env_ids=env_ids)
    robot.set_joint_velocity_target(torch.zeros_like(default), env_ids=env_ids)


def _tip_forces(env, name: str):
    """(bolt_force_N, table_force_N, net_force_N) for one fingertip sensor.

    The table's force comes from the PhysX filter on the table's collider.
    The bolt's is the REST of the net force (vector difference): a filter on
    the bolt's collider mesh was accepted by PhysX but reported zero while the
    fingers were visibly pushing the bolt, and a filter on its body prim
    matched two entries per env. Self-collision is off and there is nothing
    else in reach, so net - table is the bolt.
    """
    s = env.scene[f"cs_{name}"]
    fm = _t(s.data.force_matrix_w)                 # (N, 1, 1, 3): filter 0 = table
    net_v = _t(s.data.net_forces_w)[:, 0, :]
    table_v = fm[:, 0, 0, :]
    table = torch.norm(table_v, dim=1)
    bolt = torch.norm(net_v - table_v, dim=1)
    net = torch.norm(net_v, dim=1)
    return bolt, table, net


def bolt_table_clearance(env) -> torch.Tensor:
    """Gap between the bolt's lowest possible point and the table top [m].

    Geometric, not a contact sensor. A ContactSensor on the bolt was the first
    attempt and PhysX refused it: the bolt prim has TWO contact-reporting
    descendants per env (its body Xform and its collider mesh), so the sensor
    expected twice as many filter entries as the table provides. A broken
    filter reports zero force, which would make "the bolt never touched the
    table" silently always true -- the worst kind of failure for a success
    test. This cannot fail that way.

    Exact for this object: whatever its orientation, no point of the bolt is
    further from its centre than half its longest dimension, so a positive
    clearance proves it is not touching.
    """
    z = _t(env.scene["object"].data.root_pos_w)[:, 2] - env.scene.env_origins[:, 2]
    return z - BOLT_HALF_EXTENT - TABLE_TOP


def _phases(env):
    """(grasp, lift, hold) step counts for THIS env.

    Experiment 2 is 60/30/30 and these module constants are it. The staged
    pipeline (wuji_pipeline_env_cfg.py) runs a 90/30/30 episode -- the wrist
    has to travel before it can close -- and shares this bookkeeping, so the
    phase boundaries are read from the env cfg when it sets them. Absent those
    fields the values are unchanged, which is what keeps Experiment 2's
    measured runs comparable.
    """
    c = getattr(env, "cfg", None)
    g = getattr(c, "grasp_steps", GRASP_STEPS)
    # The staged pipeline lets the grasp phase END EARLY, per environment, as
    # soon as the grasp is established -- so its boundary is a (N,) tensor, not
    # a scalar. Every comparison below broadcasts either way.
    hook = getattr(env, "_grasp_boundary", None)
    if hook is not None:
        g = hook()
    return (g, getattr(c, "lift_steps", LIFT_STEPS),
            getattr(c, "hold_steps", HOLD_STEPS))


def phase_of(env) -> torch.Tensor:
    """0 grasp, 1 lift, 2 hold, from steps already taken in the episode."""
    g, l, _ = _phases(env)
    t = env.episode_length_buf
    return torch.where(t < g, 0, torch.where(t < g + l, 1, 2))


# ---------------------------------------------------------------------------
# Action term: two synergies + scripted lift
# ---------------------------------------------------------------------------


class SynergyGraspAction(ActionTerm):
    cfg: "SynergyGraspActionCfg"

    def __init__(self, cfg: "SynergyGraspActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        robot = self._asset
        self._thumb_ids, _ = robot.find_joints(cfg.thumb_joints, preserve_order=True)
        self._finger_ids, _ = robot.find_joints(cfg.finger_joints, preserve_order=True)
        self._lift_id, _ = robot.find_joints([cfg.lift_joint])
        fixed_ids, fixed_names = robot.find_joints(list(cfg.q_fixed.keys()))
        import re
        self._fixed_ids = fixed_ids
        self._fixed_q = torch.tensor(
            [next(v for k, v in cfg.q_fixed.items() if re.fullmatch(k, n)) for n in fixed_names],
            device=self.device)

        def resolve(table, ids):
            names = [robot.joint_names[i] for i in ids]
            return torch.tensor([next(v for k, v in table.items() if re.fullmatch(k, n)) for n in names],
                                device=self.device)

        self._open_t, self._closed_t = resolve(cfg.q_open, self._thumb_ids), resolve(cfg.q_closed, self._thumb_ids)
        self._open_f, self._closed_f = resolve(cfg.q_open, self._finger_ids), resolve(cfg.q_closed, self._finger_ids)
        self._raw = torch.zeros(self.num_envs, 2, device=self.device)
        self._closure = torch.zeros(self.num_envs, 2, device=self.device)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._closure

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0.0
        self._closure[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions.clamp(-1.0, 1.0)
        cmd = (self._raw + 1.0) / 2.0
        grasping = (phase_of(self._env) == 0).unsqueeze(1)
        new = self._closure + self.cfg.smoothing * (cmd - self._closure)
        # Frozen outside the grasp phase.
        self._closure[:] = torch.where(grasping, new, self._closure)

    def apply_actions(self):
        robot = self._asset
        c_t = self._closure[:, 0:1]
        c_f = self._closure[:, 1:2]
        q_t = self._open_t + c_t * (self._closed_t - self._open_t)
        q_f = self._open_f + c_f * (self._closed_f - self._open_f)
        robot.set_joint_position_target_index(target=q_t, joint_ids=self._thumb_ids)
        robot.set_joint_position_target_index(target=q_f, joint_ids=self._finger_ids)
        robot.set_joint_position_target_index(
            target=self._fixed_q.unsqueeze(0).expand(self.num_envs, -1), joint_ids=self._fixed_ids)
        # Scripted lift: ramp over the lift phase, hold afterwards.
        t = self._env.episode_length_buf.float()
        frac = ((t - GRASP_STEPS) / LIFT_STEPS).clamp(0.0, 1.0)
        lift_q = _buffers(self._env)["lift_start"] + frac * self.cfg.lift_height
        robot.set_joint_position_target_index(target=lift_q.unsqueeze(1), joint_ids=self._lift_id)


@configclass
class SynergyGraspActionCfg(ActionTermCfg):
    class_type: type = SynergyGraspAction
    asset_name: str = "robot"
    thumb_joints: list[str] = THUMB_JOINTS
    finger_joints: list[str] = FINGER_JOINTS
    q_open: dict = Q_OPEN
    q_closed: dict = Q_CLOSED
    q_fixed: dict = Q_FIXED
    lift_joint: str = LIFT_JOINT
    lift_height: float = LIFT_HEIGHT
    smoothing: float = ACTION_SMOOTHING


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def tip_forces_obs(env) -> torch.Tensor:
    """Fingertip force on the bolt, thumb / index / middle, scaled by F_MAX.

    The simulator's contact sensor reports one aggregate force per fingertip,
    which is what a real tactile pad's "total force" would reduce to; with no
    array there is no separate per-taxel maximum, so this is both.
    """
    return (_update(env)["tip_force"] / F_MAX).clamp(-5.0, 5.0)


def contact_flags(env) -> torch.Tensor:
    """Per-fingertip contact, thumb / index / middle (raw, not the stable version)."""
    b = _update(env)
    return (b["streak"] > 0).float()


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def _update(env) -> dict:
    """Per-step bookkeeping: contact flags, hit flags, hold streak, success.

    Called from every reward/observation function that reads the buffers and
    guarded by common_step_counter so it runs once per step. (It was first a
    weight-0 reward term placed first in the list; Isaac Lab SKIPS zero-weight
    terms, so it never ran and every flag stayed False.)
    """
    b = _buffers(env)
    if b.get("_step") == int(env.common_step_counter):
        return b
    b["_step"] = int(env.common_step_counter)
    g_steps, l_steps, h_steps = _phases(env)
    bolt_f = {k: _tip_forces(env, k) for k in TIP_BODIES}

    # Per-fingertip bolt-contact force, for the observation and the flags. The
    # simulator gives ONE aggregate force per fingertip, not a tactile array,
    # so the spec's "maximum force" and "total force" per fingertip are the
    # same number here; it is reported once rather than duplicated.
    b["tip_force"] = torch.stack([bolt_f[k][0] for k in ("thumb", "index", "middle")], dim=1)
    touching = b["tip_force"] > CONTACT_N                       # (N, 3) raw, this step
    b["streak"] = torch.where(touching, b["streak"] + 1, torch.zeros_like(b["streak"]))
    stable = b["streak"] >= STABLE_STEPS                        # persisted, so not a spike
    b["thumb_contact"], b["finger_contact"] = touching[:, 0], touching[:, 1] | touching[:, 2]
    b["thumb_stable"], b["finger_stable"] = stable[:, 0], stable[:, 1] | stable[:, 2]

    table_now = torch.stack([v[1] for v in bolt_f.values()], dim=1).max(dim=1).values > TABLE_HIT_N
    excess_now = torch.stack([v[2] for v in bolt_f.values()], dim=1).max(dim=1).values > F_MAX
    b["table_hit"] |= table_now
    b["excess"] |= excess_now

    z = _t(env.scene["object"].data.root_pos_w)[:, 2] - BOLT_REST_Z
    up = z > SUCCESS_LIFT
    b["hold_streak"] = torch.where(up, b["hold_streak"] + 1, torch.zeros_like(b["hold_streak"]))
    after_grasp = env.episode_length_buf > g_steps
    b["was_up"] |= after_grasp & (z > UP_THRESHOLD)

    # "The bolt does not touch the table during verification" (phase 3).
    b["bolt_on_table"] = bolt_table_clearance(env) <= 0.0
    verifying = env.episode_length_buf > g_steps + l_steps
    b["verify_clean"] &= ~(verifying & b["bolt_on_table"])

    grasping = env.episode_length_buf <= g_steps
    b["thumb_steps"] += (b["thumb_stable"] & grasping).float()
    b["finger_steps"] += (b["finger_stable"] & grasping).float()
    last = env.episode_length_buf >= env.max_episode_length
    b["success"] = (last & (b["hold_streak"] >= h_steps) & b["verify_clean"]
                    & ~b["table_hit"] & ~b["excess"])
    if bool(last.any()):
        ids = last.nonzero(as_tuple=False).squeeze(-1)
        b["last_success"][ids] = b["success"][ids]
        b["last_held"][ids] = b["hold_streak"][ids] >= h_steps          # the grasp itself, ignoring disqualifiers
        b["last_table_hit"][ids] = b["table_hit"][ids]
        b["last_excess"][ids] = b["excess"][ids]
        b["last_drop"][ids] = b["paid_drop"][ids] | (b["was_up"][ids] & (z[ids] < UP_THRESHOLD))
        b["last_was_up"][ids] = b["was_up"][ids]
        # g_steps is per-env in the staged pipeline (the grasp phase can end
        # early), so it has to be indexed by the same env ids. Episodes no
        # longer all end together either -- the structural-contact termination
        # ends them early -- so `ids` really is a subset.
        _gs = g_steps[ids] if torch.is_tensor(g_steps) else g_steps
        b["last_thumb_frac"][ids] = b["thumb_steps"][ids] / _gs
        b["last_finger_frac"][ids] = b["finger_steps"][ids] / _gs
        # Synergy closure, when there is a term to read it from. The 20-DoF
        # Experiment 3 env shares this bookkeeping but has no synergies, and
        # asking for a term by a name it does not have raises a KeyError inside
        # the reward manager, where it is hard to attribute.
        #
        # It MUST be captured here, at the episode's last step. Isaac Lab
        # resets inside the same step() that ends an episode and the action
        # term's reset() zeroes the closure, so anything reading it afterwards
        # gets 0.00 for every episode -- which looks exactly like a policy that
        # never closed its fingers, and was reported as one.
        _syn = env.action_manager._terms.get("synergies")
        if _syn is not None:
            b["last_closure"][ids] = _syn.processed_actions[ids]
        else:
            # The staged pipeline's term is named "pipeline" and carries the
            # two closures on a `closure` property alongside its 6 wrist actions.
            for _t_ in env.action_manager._terms.values():
                _c = getattr(_t_, "closure", None)
                if _c is not None:
                    b["last_closure"][ids] = _c[ids]
                    break
        b["last_verify_clean"][ids] = b["verify_clean"][ids]
    return b


def _once(b, flag_key: str, condition: torch.Tensor) -> torch.Tensor:
    pay = condition & ~b[flag_key]
    b[flag_key] |= pay
    return pay.float()


def thumb_contact_bonus(env):
    """Paid once, and only for a contact that has persisted STABLE_STEPS."""
    b = _update(env)
    return _once(b, "paid_thumb", b["thumb_stable"])


def finger_contact_bonus(env):
    """Paid once, and only for a contact that has persisted STABLE_STEPS."""
    b = _update(env)
    return _once(b, "paid_finger", b["finger_stable"])


def opposing_contact_bonus(env):
    """Both sides of the pinch stably on the bolt at the same time."""
    b = _update(env)
    return _once(b, "paid_opposing", b["thumb_stable"] & b["finger_stable"])


def success_bonus(env):
    return _update(env)["success"].float()


def drop_penalty(env):
    b = _update(env)
    z = _t(env.scene["object"].data.root_pos_w)[:, 2] - BOLT_REST_Z
    dropped = b["was_up"] & (z < UP_THRESHOLD) & (env.episode_length_buf > _phases(env)[0])
    return _once(b, "paid_drop", dropped)


def table_contact_penalty(env):
    b = _update(env)
    return _once(b, "paid_table", b["table_hit"])


def excess_force_penalty(env):
    b = _update(env)
    return _once(b, "paid_excess", b["excess"])


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


def _tip_sensor(name: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{TIP_BODIES[name]}",
        update_period=0.0,
        history_length=0,
        # PhysX filter patterns must resolve to exactly one contact entry per
        # env; the table cube's collider mesh does. See _tip_forces for why
        # the bolt is not filtered here.
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table/geometry/mesh"],
    )


@configclass
class WujiSynergySceneCfg(InteractiveSceneCfg):
    table = _static_box("Table", LAYOUT["table_center"], LAYOUT["table_size"], LAYOUT["table_color"])

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bolt",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(BOLT_XY[0], BOLT_XY[1], BOLT_REST_Z + 0.002),
            rot=_yawed_lying_rot(BOLT_YAW_DEG),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=LAYOUT["bolt_asset"],
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=1.0,
                solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=LAYOUT["bolt_mass"]),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max", restitution_combine_mode="min",
                static_friction=0.9, dynamic_friction=0.9, restitution=0.0,
            ),
        ),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_hand_usd(NUM_FINGERS),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(WRIST_XY[0], WRIST_XY[1], TABLE_TOP + WRIST_HEIGHT),
            rot=HAND_ROT,
            joint_pos={**{k: v for k, v in Q_OPEN.items()}, **Q_FIXED, LIFT_JOINT: 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["r_.*"],
                effort_limit={
                    "r_thumb_cmc_.*": 0.6, "r_.*_mcp_flex": 2.0, "r_.*_mcp_abd": 0.2,
                    "r_thumb_mcp": 0.3, "r_thumb_ip": 0.3, "r_.*_pip": 0.3, "r_.*_dip": 0.3,
                },
                velocity_limit=8.0, stiffness=3.0, damping=0.1, armature=1e-4,
            ),
            # Rigid drive: the wrist is supposed to be FIXED. At 5000 N/m the
            # fingers pressing on the table pushed the whole hand up 2.4 cm on
            # this joint (120 N), which silently changed every measured height.
            "lift": ImplicitActuatorCfg(
                joint_names_expr=[LIFT_JOINT],
                effort_limit=5000.0, velocity_limit=0.5, stiffness=2.0e5, damping=2.0e3,
            ),
        },
    )

    cs_thumb = _tip_sensor("thumb")
    cs_index = _tip_sensor("index")
    cs_middle = _tip_sensor("middle")
    cs_ring = _tip_sensor("ring")
    cs_pinky = _tip_sensor("pinky")

    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))


# ---------------------------------------------------------------------------
# MDP
# ---------------------------------------------------------------------------


@configclass
class ActionsCfg:
    synergies = SynergyGraspActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=SYNERGY_JOINTS, preserve_order=True)})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=SYNERGY_JOINTS, preserve_order=True)})
        last_action = ObsTerm(func=mdp.last_action)
        tip_forces = ObsTerm(func=tip_forces_obs)
        contacts = ObsTerm(func=contact_flags)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    thumb_contact = RewTerm(func=thumb_contact_bonus, weight=0.5)
    finger_contact = RewTerm(func=finger_contact_bonus, weight=0.5)
    opposing_contact = RewTerm(func=opposing_contact_bonus, weight=1.0)
    success = RewTerm(func=success_bonus, weight=20.0)
    drop = RewTerm(func=drop_penalty, weight=-5.0)
    table_contact = RewTerm(func=table_contact_penalty, weight=-3.0)
    excess_force = RewTerm(func=excess_force_penalty, weight=-3.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


def bolt_off_table(env):
    return _t(env.scene["object"].data.root_pos_w)[:, 2] < TABLE_TOP - 0.075


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    off_table = DoneTerm(func=bolt_off_table)


@configclass
class EventsCfg:
    # Fixed everything: reset the bolt and the hand to their defaults, no noise.
    reset_bolt = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"asset_cfg": SceneEntityCfg("object"),
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}},
    )
    reset_hand = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )
    hold_targets = EventTerm(func=hold_default_targets, mode="reset", params={})
    reset_buffers = EventTerm(func=reset_synergy_buffers, mode="reset", params={})


@configclass
class WujiSynergyGraspEnvCfg(ManagerBasedRLEnvCfg):
    scene: WujiSynergySceneCfg = WujiSynergySceneCfg(num_envs=16, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # A silent mismatch here is the difference between lifting the bolt and
        # dragging it sideways, and it is invisible in the scene.
        import numpy as _np
        from scipy.spatial.transform import Rotation as _R
        _up = _R.from_quat(_np.array(HAND_ROT)).as_matrix() @ _np.array(LIFT_AXIS)
        if abs(_up[2] - 1.0) > 1e-3:
            raise ValueError(
                f"LIFT_AXIS {LIFT_AXIS} under HAND_ROT {HAND_ROT} points at {_up.round(4)}, not world +z. "
                "Regenerate the hand URDF and USD with tools/make_wuji_hand_urdf.py --lift-axis, "
                "or re-solve the pose with tools/find_pinch_pose.py.")

        self.decimation = PHYSICS_HZ // CONTROL_HZ
        self.episode_length_s = EPISODE_STEPS / CONTROL_HZ
        self.sim.dt = 1.0 / PHYSICS_HZ
        self.sim.render_interval = self.decimation
        self.sim.device = "cuda:0"
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0,
        )
        if hasattr(self.sim, "disable_contact_processing"):
            self.sim.disable_contact_processing = False
        self.viewer.eye = (WRIST_XY[0] + 0.5, WRIST_XY[1] - 0.5, TABLE_TOP + 0.4)
        self.viewer.lookat = (BOLT_XY[0], BOLT_XY[1], TABLE_TOP + 0.03)


__all__ = ["WujiSynergyGraspEnvCfg", "_phases", "GRASP_STEPS", "LIFT_STEPS", "HOLD_STEPS", "EPISODE_STEPS",
           "BOLT_REST_Z", "SUCCESS_LIFT", "UP_THRESHOLD", "F_MAX", "TABLE_TOP", "WRIST_XY", "WRIST_HEIGHT",
           "BOLT_XY", "SYNERGY_JOINTS", "_buffers", "_update", "phase_of"]
