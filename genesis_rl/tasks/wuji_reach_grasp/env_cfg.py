# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experiment 4: safe reach-to-grasp with the Wuji hand. Design: docs/EXPERIMENT4_DESIGN.md.

Whole Adam-U, right arm through Cartesian differential IK, right hand through
two finger synergies. The wrist starts 5-10 cm ABOVE the pipeline's solved
pinch pose and must descend along world -z, correct a small pose error, pinch
the bolt, and hand over to a scripted settle / lift / hold.

Everything the pipeline (wuji_pipeline_env_cfg.py) proved is reused: the
solved pose, the IK mechanics, the synergies, the contact plumbing, the
success test and the PPO settings. What is new:

  * a standoff axis at reset, through baked arm solutions + Jacobians
  * a 74-observation interface: the pipeline's 58 + 16 safety observations
  * an explicit SETTLE phase and a latch that resets the IK reference
  * an actor mask so SETTLE / LIFT / HOLD never train the actor (MaskedPPO)
  * reward: coarse wrist reach + fine finger approach + near-object velocity
    + clearance + shield penalty, on top of the sparse contact / success terms
  * whole-body contact monitoring with a bolt-contact whitelist; any invalid
    collision ends the episode
  * per-step TensorBoard logging that survives Isaac Lab's reset
  * perception corruption at the later levels, applied to ONE estimated bolt
    pose that every geometric bolt observation is derived from

Nothing here modifies the pipeline env, its tools or its logs.

Level selection: ADAM_U_RG_LEVEL (0..8). NOT ADAM_U_LEVEL -- the pipeline
module, which this one imports, validates that variable against its own 0..3.
"""

from __future__ import annotations

import math
import dataclasses
import os
from dataclasses import dataclass, field

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab.envs.mdp as mdp
from genesis_rl.tasks.wuji_reach.env_cfg import LAYOUT, _t
from genesis_rl.tasks.wuji_pipeline import env_cfg as P
from genesis_rl.tasks.wuji_pipeline.env_cfg import (
    BOLT_HALF_LEN,
    BOLT_RADIUS,
    BOLT_XY,
    CONTROL_DT,
    CONTROL_HZ,
    FINGER_JOINTS,
    LIFT_HEIGHT,
    PHYSICS_HZ,
    Q_CLOSED,
    Q_OPEN,
    RIGHT_ARM,
    THUMB_JOINTS,
    SYNERGY_JOINTS,
    TIP_BODIES,
    TIP_OFFSETS,
    WRIST_BODY,
    WRIST_ROT,
    WRIST_XY,
    WRIST_Z_CONTACT,
    LimitsCfg,
    WujiPipelineSceneCfg,
    _body_sensor,
    _metrics,
    arm_q_normalized,
    joint_limit_penalty,
    joint_limit_termination,
    ik_failure_termination,
    wrist_ang_vel,
    wrist_lin_vel,
)
from genesis_rl.tasks.wuji_synergy.env_cfg import (
    BOLT_REST_Z,
    CONTACT_N,
    F_MAX,
    STABLE_STEPS,
    SUCCESS_LIFT,
    TABLE_HIT_N,
    TABLE_TOP,
    UP_THRESHOLD,
    _once,
    _tip_forces,
    bolt_table_clearance,
)

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

GRASP_MAX_STEPS = 120       # APPROACH + GRASP, a timeout; the latch usually ends it earlier
SETTLE_STEPS = 10
# Pre-lift (between settle and lift, every level): first DRAG_STEPS of a scripted drag along the exposed
# direction for a rod whose end is under a cover (zero motion otherwise), then GRIP_CHECK_STEPS of a
# 1 mm/step test lift. At levels with grip_check, a rod that has not risen GRIP_CHECK_RISE by the end
# is not held: the retry fires here instead of after an empty 10 cm lift.
DRAG_STEPS = 10
DRAG_STEP_M = 0.003
GRIP_CHECK_STEPS = 8
GRIP_CHECK_RATIO = 0.4              # the rod must rise at least this fraction of what the FINGERTIPS rose during the test lift
GRIP_CHECK_MIN_TIPS = 0.004         # m: the check needs the fingertips to have risen at least this much (else inconclusive)
GRIP_TEST_TOTAL_M = 0.024           # m: commanded test lift at grip-check levels (v3, level 31+): over the whole pre-lift,
                                    # or over its last GRIP_CHECK_STEPS when the first DRAG_STEPS belong to the policy
PRE_LIFT_STEPS = DRAG_STEPS + GRIP_CHECK_STEPS
LIFT_START = SETTLE_STEPS + PRE_LIFT_STEPS     # steps after grasp_end at which the lift begins
# 10 slow steps (1 mm) then 30 at 3 mm: 10 cm along the approach axis. The two env variables exist
# for A/B evaluations of the lift profile only (ADAM_U_RG_LIFT_STEPS, ADAM_U_RG_LIFT_SLOW_STEPS).
LIFT_STEPS = int(os.environ.get("ADAM_U_RG_LIFT_STEPS", "40"))
LIFT_SLOW_STEPS = int(os.environ.get("ADAM_U_RG_LIFT_SLOW_STEPS", "10"))
LIFT_SLOW_STEP_M = 0.001
HOLD_STEPS = 30
SCRIPTED_STEPS = SETTLE_STEPS + PRE_LIFT_STEPS + LIFT_STEPS + HOLD_STEPS          # 95
EPISODE_MAX_STEPS = GRASP_MAX_STEPS + SCRIPTED_STEPS             # 200 (grasp 120 + settle 10 + lift 40 + hold 30)
# Retry (level 22+): when the lift ends with the rod not up, or the rod drops in the hold, the
# hand opens, the latch clears and the grasp phase restarts from where the hand is (10 cm up,
# a natural standoff) with this many steps, up to RGLevel.retry_max times per episode.
GRASP_RETRY_STEPS = 60


def episode_max_steps(retry_max: int, settle_in: int = 0) -> int:
    """Steps the phase machine can need: the grasp budget, the scripted phases, every retry, and (the heap)
    the settle-in, during which the policy is frozen and no grasp time is spent."""
    return EPISODE_MAX_STEPS + retry_max * (GRASP_RETRY_STEPS + SCRIPTED_STEPS) + settle_in


def level_max_steps(lvl) -> int:
    return episode_max_steps(lvl.retry_max, lvl.pile.settle_in)

# ---------------------------------------------------------------------------
# Standoff buckets: arm solutions + damped pseudo-inverse Jacobians
# ---------------------------------------------------------------------------
#
# Generated by tools/solve_reach_grasp_poses.py (which reproduces the
# pipeline's PREGRASP_ARM_CONTACT and J_SHARP to 1e-5 as its self-check).
# A reset picks the bucket nearest the sampled standoff and applies only the
# residual through that bucket's Jacobian, so no residual exceeds 2.5 cm in z
# plus the level's xy / rotation error. Verified by full FK: the worst corner
# (2 cm xy, 2.5 cm z, 7.5 deg yaw, 2 deg roll and pitch) lands 2.3 mm and
# 0.65 deg off. Regenerate if the pinch pose, the held waist or WRIST_XY change.


@dataclass(frozen=True)
class StandoffBucket:
    height: float                   # m above the contact pose
    arm: dict                       # the 7 right-arm joint angles
    j_sharp: tuple                  # 7 x 6 damped pseudo-inverse, world-frame (dx, dy, dz, drx, dry, drz)
    s_min: float                    # minimum normalised arm slack at the bucket pose


STANDOFF_BUCKETS = (
    StandoffBucket(
        height=0.000,
        arm={
            "shoulderPitch_Right": -0.9142, "shoulderRoll_Right": -0.3996, "shoulderYaw_Right": +0.1797,
            "elbow_Right": -1.4277, "wristYaw_Right": +0.9023, "wristPitch_Right": -0.3292,
            "wristRoll_Right": -0.3582,
        },
        j_sharp=(
            (-3.331493, -1.038763, +1.437866, +0.026060, +0.003865, -0.052770),
            (-0.103435, +1.866381, -0.781033, +0.120792, +0.024374, -0.248876),
            (-0.716272, +1.249425, -0.142259, -0.148744, -0.027611, +0.314143),
            (+2.918086, +0.164829, -4.983457, -0.020741, +0.003640, +0.037612),
            (-0.859470, +1.444507, -1.445432, -0.666762, -0.665389, -0.044995),
            (-0.555842, +1.850988, +2.014358, -0.546271, +0.268108, -0.646786),
            (+0.917496, +0.056439, +2.936668, -0.149385, +0.935720, +0.247244),
        ),
        s_min=0.601,
    ),
    StandoffBucket(
        height=0.050,
        arm={
            "shoulderPitch_Right": -0.8511, "shoulderRoll_Right": -0.4100, "shoulderYaw_Right": +0.1414,
            "elbow_Right": -1.6649, "wristYaw_Right": +0.8569, "wristPitch_Right": -0.2454,
            "wristRoll_Right": -0.1958,
        },
        j_sharp=(
            (-3.441282, -1.221981, +0.833093, +0.023535, +0.001846, -0.040548),
            (-0.180896, +1.853625, -0.671067, +0.165749, +0.011860, -0.284917),
            (-0.734539, +1.662688, -0.233598, -0.157346, -0.009122, +0.277889),
            (+2.861413, +0.220297, -4.233018, -0.021739, +0.003775, +0.033901),
            (-0.989586, +1.801808, -1.210206, -0.705461, -0.605677, -0.166581),
            (-0.326648, +1.893684, +2.071276, -0.445770, +0.380059, -0.653368),
            (+0.992059, +0.011020, +2.669931, -0.166064, +0.853253, +0.271310),
        ),
        s_min=0.607,
    ),
    StandoffBucket(
        height=0.100,
        arm={
            "shoulderPitch_Right": -0.8214, "shoulderRoll_Right": -0.4285, "shoulderYaw_Right": +0.1143,
            "elbow_Right": -1.8665, "wristYaw_Right": +0.8097, "wristPitch_Right": -0.1473,
            "wristRoll_Right": -0.0529,
        },
        j_sharp=(
            (-3.591582, -1.446747, +0.239739, +0.019369, +0.001751, -0.026772),
            (-0.236509, +1.797076, -0.630673, +0.224236, +0.005845, -0.300508),
            (-0.755677, +2.122588, -0.352712, -0.170536, -0.002490, +0.235630),
            (+2.937040, +0.282068, -3.666079, -0.023483, +0.003403, +0.028786),
            (-1.064313, +2.137395, -1.045261, -0.709216, -0.536507, -0.271368),
            (-0.175996, +1.982897, +2.209671, -0.357321, +0.465435, -0.655129),
            (+0.996900, -0.017447, +2.556813, -0.194213, +0.778854, +0.279352),
        ),
        s_min=0.465,
    ),
)

# ---------------------------------------------------------------------------
# Curriculum levels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerceptionCfg:
    """Corruption of the policy's bolt-pose estimate. All zeros = exact."""

    bias_xy: float = 0.0            # m, per-episode constant, uniform +/-
    bias_z: float = 0.0             # m
    bias_rot: float = 0.0           # rad, per-episode constant rotation about a random axis
    noise_lo: float = 0.0           # m, per-frame isotropic position noise std, sampled per episode
    noise_hi: float = 0.0
    latency_lo: int = 0             # control steps, sampled per episode
    latency_hi: int = 0
    dropout: float = 0.0            # probability per step of holding the last valid estimate

    @property
    def active(self) -> bool:
        return any((self.bias_xy, self.bias_z, self.bias_rot, self.noise_hi, self.latency_hi, self.dropout))


@dataclass(frozen=True)
class PhysicsCfg:
    mass_scale: tuple = (1.0, 1.0)
    friction_scale: tuple = (1.0, 1.0)
    gain_scale: tuple = (1.0, 1.0)

    @property
    def active(self) -> bool:
        return self.mass_scale != (1.0, 1.0) or self.friction_scale != (1.0, 1.0) or self.gain_scale != (1.0, 1.0)


@dataclass(frozen=True)
class ClutterCfg:
    """Static (kinematic) box obstacles on the table, outside the approach corridor.

    The corridor is everything the arm and hand sweep during a nominal grasp:
    the bolt's neighbourhood (radius `r_min`) and the strip BEHIND the bolt,
    toward the wrist and the forearm, `strip_half_width` wide. Obstacles go in
    front of the bolt or well to the side. Sizes are fixed per obstacle slot
    (a PhysX cuboid cannot be resized at reset); poses are per episode.
    """

    count_lo: int = 0
    count_hi: int = 0
    fixed_xy: tuple = ()            # level 8: one fixed placement, bolt-relative (m)
    r_min: float = 0.09             # m from the bolt centre
    r_max: float = 0.14
    strip_half_width: float = 0.12  # m, the arm strip behind the bolt is excluded
    min_separation: float = 0.07    # m between obstacle centres

    @property
    def active(self) -> bool:
        return self.count_hi > 0


OBSTACLE_SIZES = ((0.05, 0.05, 0.06), (0.04, 0.06, 0.05), (0.06, 0.04, 0.08))   # m, per clutter slot
# Obstacle AABB slots: 0-2 clutter boxes (levels 8-10) / support blocks (tilted rods) / the cover
# (slot 0, lying rods); 3-6 the bin walls (level 25). Parked slots sit far away.
MAX_OBSTACLES = 7
# Capsule obstacles: segment p0-p1 with radius r, in the same clearance model. Slots 0-1 the bed rods
# (level 24, "hard": penetration is a collision); slots 2-5 the pile's neighbour rods (level 27,
# "soft": seen by the clearance observations, pushing them is allowed).
MAX_CAPSULES = 12
BED_SLOTS = (0, 1)
NEIGHBOR_SLOTS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
NEIGHBOR_R_MIN, NEIGHBOR_R_MAX = 0.04, 0.08      # m from the target's centre
DOUBLE_PICK_RISE = 0.02                         # m: a neighbour this far above its rest at the end = two rods lifted
BED_SPACING = 0.020                 # m between the two bed rods, along the target's axis
BED_RADIUS = 0.0127                 # m, the capsule radius (the rod's band; over-reads the thread by 4 mm)
BIN_INNER_HALF = float(os.environ.get("ADAM_U_RG_BIN_HALF", "0.13"))   # ADAM_U_RG_BIN_HALF: a smaller bin for edge-rod evaluations.
                                    # 0.13               # m, the virtual bin's inner half-size around BOLT_XY (a tilted rod's block reaches 7 cm out)
BIN_WALL_T = 0.01                   # m, wall thickness
BIN_WALL_H = 0.04                   # m, wall height above the table
# Support blocks for a TILTED rod (levels 20+): one end of the rod on the
# table, the rod resting on a low kinematic block. Three heights, one block
# per env (the other two parked), in the obstacle AABB slots, which are free
# at these levels because clutter is off. Axis-aligned 4 x 4 cm, so the
# analytic AABB is exact.
SUPPORT_SIZES = ((0.04, 0.04, 0.010), (0.04, 0.04, 0.017), (0.04, 0.04, 0.025))
SUPPORT_EDGE_DIST = 0.030           # m, HORIZONTAL distance from the rod's low-end contact to the block's near edge
assert len(SUPPORT_SIZES) <= MAX_OBSTACLES


# Partial cover (level 23+): a kinematic block hovering just above one end of a LYING rod, so that
# end cannot be grasped; the commanded point is drawn on the exposed part only. Uses obstacle AABB
# slot 0, which a lying rod leaves free (the support blocks of tilted rods take it otherwise).
COVER_SIZE = (0.04, 0.04, 0.03)
COVER_GAP = 0.002                   # m between the rod's top and the block's underside
COVER_CENTRE_S = 0.025              # m along the axis from the rod's centre to the block's centre
COVER_EDGE_S = COVER_CENTRE_S - COVER_SIZE[0] / 2      # 5 mm: the block starts here; the exposed part ends here


@dataclass(frozen=True)
class PileCfg:
    """Dynamic neighbour rods around the target (level 27+): lying, at random yaw, `count` of
    them (up to six) at `r` from the target's centre. `across_prob`: the first neighbour is
    dropped ACROSS a lying or tilted target instead (the pile's version of the cover)."""
    count: int = 0
    r: tuple = (NEIGHBOR_R_MIN, NEIGHBOR_R_MAX)
    across_prob: float = 0.0        # lying targets only
    prob: float = 1.0               # of episodes with the pile at all (mixed-scene levels park it otherwise)
    upper: int = 0                  # the heap (level 35+): this many of `count` are dropped onto the first layer, over the target
    settle_in: int = 0              # steps at the start of the episode with the policy frozen while the heap settles
    top_rod: bool = False           # the heap: a target buried under a dropped rod is re-labelled, the rod on top becomes the target

    @property
    def active(self) -> bool:
        return self.count > 0


@dataclass(frozen=True)
class BedCfg:
    """The target rod resting across two rods lying on the table (level 24+)."""
    prob: float = 0.0               # of LYING episodes (exclusive with the cover)

    @property
    def active(self) -> bool:
        return self.prob > 0.0


@dataclass(frozen=True)
class CoverCfg:
    """A block over one end of a lying rod."""
    prob: float = 0.0               # of LYING episodes

    @property
    def active(self) -> bool:
        return self.prob > 0.0


@dataclass(frozen=True)
class BinCfg:
    """The bin's walls. `inner` empty: the virtual bin (BIN_INNER_HALF square, BIN_WALL_H high, levels 25-49);
    else adam-moveit's real bin (workcell.json): inner x, y (m) around BOLT_XY, the walls `height` above the
    table (the bin's floor plate is the table here) and `wall` thick."""
    inner: tuple = ()
    height: float = 0.04
    wall: float = 0.01


@dataclass(frozen=True)
class FillCfg:
    """The full bin (level 50+): a settled heap from tools/make_bin_heaps.py (adam-moveit's drop of 40-60 rods)
    in `prob` of episodes. The target is an exposed rod of the heap inside the reachable window, lying, leaning
    or standing with the weights `kinds`; every other rod of the heap is a neighbour."""
    prob: float = 0.0
    file: str = "bin_heaps_full.npz"
    kinds: tuple = (0.5, 0.3, 0.2)  # lying (< FILL_LYING_DEG), leaning (up to FILL_TILT_MAX_DEG), standing (> FILL_UPRIGHT_DEG)
    # level 52+: heaps from several files, ((file, weight), ...) replacing `file` (a bin at several fill levels)
    files: tuple = ()
    # "random": an exposed reachable rod of the drawn kind (levels 50-51); "highest": the rule perception uses
    # (genesis_policy's selector): the highest exposed reachable rod, a standing one scored FILL_STAND_PENALTY lower
    pick: str = "random"

    @property
    def active(self) -> bool:
        return self.prob > 0.0


FILL_RODS = 60                      # rods in a full bin (workcell.json's 5 x 6 x 2 grid): the target and 59 neighbours
FILL_LYING_DEG = 12.0               # a heap rod's axis below this elevation is a lying target (raised, no wrist roll)
FILL_TILT_MAX_DEG = 40.0            # ... up to this, a leaning one (wrist roll; the tilt buckets reach 39.8 deg)
FILL_UPRIGHT_DEG = 75.0             # ... above this, a standing one; in between it is not a target
FILL_CYL_RADIUS = 0.01135          # m: a neighbour cylinder fits inside the mesh hull (25.4 x 22.7 mm across), so hull-settled heaps
                                    # resettle as cylinders without overlaps (12.5 mm cylinders threw rods out of 81 % of full bins)
FILL_STAND_PENALTY = 0.015         # m: "highest" picking scores a standing rod this much lower (docs/PICKABILITY_REPORT.md:
                                    # standing rods on the floor 21 %, at the top of the heap as good as lying ones)
FILL_TRIES = 8                      # (heap, mirror) draws per env and reset; the first with a fitting target is used


@dataclass(frozen=True)
class TiltCfg:
    """A tilted rod: one end on the table, the rod resting on a support block."""
    prob: float = 0.0               # of episodes (after the standing draw)

    @property
    def active(self) -> bool:
        return self.prob > 0.0
OBSTACLE_PARK = (0.0, 0.0, -0.5)    # inside the table's volume, under its top: never in reach
OBSTACLE_PEN_M = 0.004              # m, proxy penetration that counts as a collision (proxies over-read ~4 mm)


@dataclass(frozen=True)
class RGLevel:
    name: str
    standoff_lo: float = 0.0        # m above the contact pose
    standoff_hi: float = 0.0
    dxy: float = 0.0                # m, uniform +/- on x and y
    dyaw: float = 0.0               # rad, uniform +/-
    drp: float = 0.0                # rad, uniform +/- on roll and pitch
    physics: PhysicsCfg = PhysicsCfg()
    perception: PerceptionCfg = PerceptionCfg()
    clutter: ClutterCfg = ClutterCfg()
    # Pose generalisation: the BOLT itself is sampled at reset. Yaw is a delta
    # about world z from the pipeline's pose (the rod is end-to-end symmetric,
    # so 180 deg is a period); xy is uniform +/- on both axes. The pinch
    # rotates with the bolt; only arm reachability changes (see pose_buckets).
    bolt_yaw_lo: float = 0.0        # rad
    bolt_yaw_hi: float = 0.0
    bolt_xy: float = 0.0            # m
    # Outside the core yaw window the xy range shrinks to bolt_xy_edge: the
    # grid is fully reachable at +/-5 cm only inside the core.
    bolt_yaw_core: tuple = (0.0, 0.0)
    bolt_xy_edge: float = 0.0
    # Waist freed: the standoff reset draws from pose_buckets_waist.npz, whose
    # buckets carry a held waist (pitch, yaw, roll) per (bolt yaw, xy) column,
    # and the action term holds that triple for the episode. Widens the
    # reachable bolt yaw window beyond what the arm alone covers (-45..+15 deg).
    free_waist: bool = False
    # Commanded grasp point: an offset along the rod's axis, uniform +/- this
    # (m), sampled per episode. Everything the policy sees of the bolt, and
    # the nominal wrist pose, are shifted to that point, so the policy grasps
    # THERE; the physical bolt, the latch, the rewards and the success test
    # stay on the true object. Sweeping the offset at evaluation maps the
    # graspable area along the axis.
    grasp_offset: float = 0.0
    # Commanded pinch angle: the hand's yaw is offset from the rod's by a
    # sampled delta, uniform +/- this (rad), so the fingers close obliquely
    # across the rod. The policy's bolt estimate is rotated by delta about the
    # vertical, so it "aligns" to a virtual rod and pinches the real one
    # obliquely. At these levels bolt_yaw_lo/hi bound the HAND yaw (that is
    # what reachability depends on) and the rod's yaw is hand yaw - delta.
    pinch_angle: float = 0.0
    # Free grasp: no commands and no alignment. The start pose still uses the
    # sampled offset and angle (that is what puts the hand somewhere over the
    # rod at some angle), but the policy sees the TRUE rod, the reach reward
    # pulls the pinch centre to the rod's axis (any point) and asks only that
    # the approach stays from above (no yaw term), and the world-yaw action is
    # scaled by yaw_rate_scale so rotating into alignment cannot be the
    # answer. The policy is rewarded only for grasping what is under the hand.
    free_grasp: bool = False
    yaw_rate_scale: float = 1.0
    # Standing rod: with this probability the rod spawns on its end (axis
    # vertical) instead of lying. The pinch is the same horizontal pinch, at
    # any hand yaw; the commanded grasp offset becomes a HEIGHT along the
    # rod. Needs the convex-decomposition collider: the asset's 64-vertex
    # convex hull has no flat end and the rod topples on its own.
    upright_prob: float = 0.0
    bolt_collider: str = "hull"      # "hull" (the asset as shipped) or "cd" (convex decomposition)
    # Binding grasp point: the latch (and so the lift and success) only fires
    # when the pinch centre, projected on the rod's true axis, is within this
    # of the commanded offset (m). 0 = not enforced: measured at levels 15-18,
    # the policy pinched near the band whatever it was told (TRAINING_LOG).
    grasp_point_tol: float = 0.0
    pinch_centre_tol: float = 0.0    # > 0: the latch also needs the pinch midpoint within this of the rod's AXIS (a deep, centred pinch)
    free_prelift: bool = False       # the first DRAG_STEPS of the pre-lift are the POLICY's (wrist live, fingers held): slide, tilt or lift as it likes
    grip_check_late: bool = False    # the grip check compares rod and fingertip rise over the SECOND half of the test lift only (slack taken up)
    # Tilted rod, resting on a support block (levels 20+): drawn after the
    # standing draw, so lying = 1 - upright_prob - tilt.prob.
    tilt: TiltCfg = TiltCfg()
    # Wrist roll (level 21+): for a tilted rod the hand rolls about the rod's
    # horizontal normal so its approach axis is the rod's up-normal and the
    # pinch closes perpendicular to the slanted axis; the standoff and the
    # scripted lift run along that axis. Buckets from pose_buckets_tilt.npz.
    wrist_roll: bool = False
    roll_cap: float = math.radians(90)     # the most the wrist rolls (rad); the arm cannot reach every rolled pose
    # Retry (level 22+): attempts allowed after a failed lift or a drop in the hold.
    retry_max: int = 0
    retry_backoff: float = 0.0      # m: on a retry the wrist target is raised this much, so the fingers leave the heap before the policy resumes
    heap_map: bool = False          # level 36+: 8 extra observations, the free radial distance around the (estimated) rod per sector
    push_ok: bool = False           # level 37+: the clearance penalty ignores neighbour rods (pushing them aside is the way into a heap)
    # Transfer level (39): the policy and the phase machine run on what the robot can measure.
    tactile: bool = True            # False: the tip_forces and contacts observations are zero (a hand without fingertip sensors)
    hw_latch: bool = False          # the latch reads finger joint torque (motor current on the robot) and the ESTIMATED rod pose
    sense_est: bool = False         # the grip check, the retry and the drop tests read the ESTIMATED rod height, as the robot would
    arm_gains: bool = False         # the actuator-gain randomisation covers the arm joints too, not only the fingers
    # Level 40: the latch's settled test on the estimate's velocity over this many steps (1 = one-step difference,
    # which camera noise of 1-3 mm per frame turns into 0.04-0.13 m/s per axis at 30 Hz, against REL_VEL_MAX 0.1)
    est_vel_window: int = 1
    sense_est_retry: bool = False   # the retry's "not lifted at the lift's end" test reads the estimate too (39 read the true rod)
    latch_squeeze: float = 0.0      # level 41: after the latch both closures ramp up by this (capped at 1) over the settle
    effort_obs: bool = False        # level 43: the tip_forces / contacts slots carry the flexion-chain torque (motor current)
    cover_drag: bool = False        # level 47: free pre-lift levels slide a latched covered rod out from under its block
    cover_shift: bool = False       # level 48: a covered rod's estimate points at the middle of its exposed part
    # Partial cover (level 23+): a block over one end of a lying rod.
    cover: CoverCfg = CoverCfg()
    # Rod on rods (level 24+) and the bin walls (level 25+).
    bed: BedCfg = BedCfg()
    bin_walls: bool = False
    # Grip check (level 26+): a 5 mm test lift before the lift; a rod that did not rise is not held and
    # the retry fires immediately. Slide-out for covered rods is always on (it moves nothing otherwise).
    grip_check: bool = False
    # The pile (level 27+): neighbour rods around the target.
    pile: PileCfg = PileCfg()
    # The real bin and the full bin (level 50+).
    bin: BinCfg = BinCfg()
    fill: FillCfg = FillCfg()
    success_target: float = 0.95


_D = math.radians
# The held waist joints, in the order the buckets store them, and the pipeline's hold.
WAIST_JOINTS = ("waistPitch", "waistYaw", "waistRoll")
WAIST_NOMINAL = tuple(P.WAIST_HELD[j] for j in WAIST_JOINTS)
# Bolt yaw windows reachable with the waist freed, measured by
# tools/solve_reach_grasp_poses.py --grid --free-waist --min-slack 0.15 (see
# the design doc): every xy within +/-5 cm inside the core, +/-2.5 cm out to
# the edges. Beyond, the arm cannot place the pinch at any waist hold.
WAIST_WINDOW_DEG = (-60.0, 45.0)
WAIST_CORE_DEG = (-45.0, 30.0)
_PHYS5 = PhysicsCfg(mass_scale=(0.8, 1.2), friction_scale=(0.7, 1.3), gain_scale=(0.9, 1.1))
_PERC6 = PerceptionCfg(bias_xy=0.002, bias_z=0.001, bias_rot=_D(1), noise_lo=0.0005, noise_hi=0.001,
                       latency_lo=0, latency_hi=1)
_PERC7 = PerceptionCfg(bias_xy=0.005, bias_z=0.003, bias_rot=_D(3), noise_lo=0.001, noise_hi=0.002,
                       latency_lo=1, latency_hi=2, dropout=0.02)
# Transfer: a camera-grade estimate (8 mm xy / 5 mm z / 5 deg bias, 1-3 mm noise, 1-4 control steps of
# latency, 3 % dropout) and gains 0.8-1.2 on the arm and the fingers.
_PERC8 = PerceptionCfg(bias_xy=0.008, bias_z=0.005, bias_rot=_D(5), noise_lo=0.001, noise_hi=0.003,
                       latency_lo=1, latency_hi=4, dropout=0.03)
_PHYS6 = PhysicsCfg(mass_scale=(0.8, 1.2), friction_scale=(0.7, 1.3), gain_scale=(0.8, 1.2))
HIST_DEPTH = 5                                     # estimate history: latency up to HIST_DEPTH - 1 steps
# ADAM_U_RG_LEGACY_STEP=1: the pre-2026-10-13 reset, which advanced every other env's once-per-step counters twice
# on a step in which some env reset (reproduces the numbers in TRAINING_LOG.md up to level 40)
LEGACY_STEP = os.environ.get("ADAM_U_RG_LEGACY_STEP", "0") == "1"
EST_WIN_MAX = 8                                    # the latch's estimate-velocity window buffer (RGLevel.est_vel_window <= this)
TAU_CONTACT = 0.10                                 # N m: a flexion chain above this is touching (free fingers 0.03-0.05 p50-p90)
GRIP_TAU_MIN = 0.20                                # N m per finger flexion chain: the torque-based grip (hw_latch). Calibrated 2026-10-09 on the level-32 pick against the 4 N force latch (fires on ~93 % of 4 N holds, ~3 % of free fingers)

LEVELS: dict[int, RGLevel] = {
    0: RGLevel("geometry_gate"),
    1: RGLevel("fixed_5cm", standoff_lo=0.05, standoff_hi=0.05),
    2: RGLevel("small_correction_5cm", standoff_lo=0.05, standoff_hi=0.05, dxy=0.005, dyaw=_D(3)),
    3: RGLevel("approach_10cm", standoff_lo=0.075, standoff_hi=0.10, dxy=0.01, dyaw=_D(5), drp=_D(1)),
    4: RGLevel("wider_correction", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2)),
    5: RGLevel("physics", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
               physics=_PHYS5),
    6: RGLevel("perception_small", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
               physics=_PHYS5, perception=_PERC6),
    7: RGLevel("perception_moderate", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
               physics=_PHYS5, perception=_PERC7),
    # Sparse clutter, the spec's level 8 in three steps: one fixed obstacle
    # outside the corridor, then a random pose, then two to three.
    8: RGLevel("clutter_fixed", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
               physics=_PHYS5, perception=_PERC7,
               clutter=ClutterCfg(count_lo=1, count_hi=1, fixed_xy=((0.046, 0.089),))),   # 10 cm in front of the bolt
    9: RGLevel("clutter_random", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
               physics=_PHYS5, perception=_PERC7, clutter=ClutterCfg(count_lo=1, count_hi=1)),
    10: RGLevel("clutter_2_3", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, clutter=ClutterCfg(count_lo=2, count_hi=3)),
    # Pose generalisation, inside the window the held-waist arm can reach
    # (tools/solve_reach_grasp_poses.py --grid: 100 % of xy x height buckets
    # reachable for bolt yaw -45..+15 deg, s_min >= 0.14; nothing beyond +/-60).
    11: RGLevel("pose_yaw15", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(-15), bolt_yaw_hi=_D(15)),
    12: RGLevel("pose_yaw30_xy25", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(-30), bolt_yaw_hi=_D(15), bolt_xy=0.025),
    13: RGLevel("pose_window", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(-45), bolt_yaw_hi=_D(15), bolt_xy=0.05),
    # The waist (pitch, yaw, roll) is held per episode at a triple the pose
    # solver chose per (bolt yaw, xy) column; the window is what that grid
    # measured as reachable at every xy and height with slack >= 0.15.
    14: RGLevel("pose_waist", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025),
    # Level 14 plus a commanded grasp point along the rod's axis: +/-14 mm of
    # the 17.5 mm half-length, i.e. from the thick middle band out onto the
    # thinner thread near the ends. The graspable-area study sweeps this.
    15: RGLevel("grasp_point", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014),
    # Level 15 plus a commanded pinch angle, +/-30 deg off perpendicular.
    16: RGLevel("pinch_angle", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(30)),
    # Level 16's start distribution, hand-rod angle +/-45 deg, but no commands: the
    # policy grasps whatever is under the hand, with the yaw action at a quarter rate.
    17: RGLevel("free_grasp", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(45), free_grasp=True, yaw_rate_scale=0.25),
    # Level 16's commanded point and angle, and the rod standing on its end half
    # of the time (then the point is a height along the rod, the angle moot).
    18: RGLevel("upright_mix", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(30), upright_prob=0.5, bolt_collider="cd"),
    # Level 18 with the commanded point BINDING: the latch requires the pinch within 4 mm of it.
    19: RGLevel("grasp_point_bound", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(30), upright_prob=0.5, bolt_collider="cd",
                grasp_point_tol=0.004),
    # Level 19 plus the tilted rod: a third lying, a third standing, a third leaning on a block.
    20: RGLevel("tilt_mix", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(30), upright_prob=0.34, bolt_collider="cd",
                grasp_point_tol=0.004, tilt=TiltCfg(prob=0.33)),
    # Level 20 with the wrist rolled to the rod's tilt (the "20b" of the plan).
    21: RGLevel("wrist_roll", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(30), upright_prob=0.34, bolt_collider="cd",
                grasp_point_tol=0.004, tilt=TiltCfg(prob=0.33), wrist_roll=True),
    # Level 21 plus one retry: a failed lift or a drop reopens the hand and restarts the grasp phase.
    22: RGLevel("retry", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(30), upright_prob=0.34, bolt_collider="cd",
                grasp_point_tol=0.004, tilt=TiltCfg(prob=0.33), wrist_roll=True, retry_max=1),
    # Level 22 plus a block over one end of half the lying rods: the point is drawn on the exposed part.
    23: RGLevel("cover", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(30), upright_prob=0.34, bolt_collider="cd",
                grasp_point_tol=0.004, tilt=TiltCfg(prob=0.33), wrist_roll=True, retry_max=1,
                cover=CoverCfg(prob=0.5)),
    # Level 23 plus the target resting across two rods for 40 % of lying episodes (cover for 30 %).
    24: RGLevel("rod_on_rods", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(30), upright_prob=0.34, bolt_collider="cd",
                grasp_point_tol=0.004, tilt=TiltCfg(prob=0.33), wrist_roll=True, retry_max=1,
                bed=BedCfg(prob=0.4)),          # no cover: a rod pinned under a block cannot be lifted (level 23)
    # Level 24 inside a bin: four walls, 4 cm high, 10 cm from the nominal bolt position.
    25: RGLevel("bin", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(30), upright_prob=0.34, bolt_collider="cd",
                grasp_point_tol=0.004, tilt=TiltCfg(prob=0.33), wrist_roll=True, retry_max=1,
                bed=BedCfg(prob=0.4), bin_walls=True),
    # Level 25 plus the cover back (a third of lying rods), a slide-out before the lift for covered
    # rods, and the grip check that turns an empty lift into an early retry.
    26: RGLevel("slide_and_check", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(30), upright_prob=0.34, bolt_collider="cd",
                grasp_point_tol=0.004, tilt=TiltCfg(prob=0.33), wrist_roll=True, retry_max=1,
                cover=CoverCfg(prob=0.3), bed=BedCfg(prob=0.4), bin_walls=True, grip_check=False),
    # The pile: four dynamic rods around the target inside the bin; no cover or bed (the neighbours are the
    # real version of both); a standing rod is rare in a pile.
    27: RGLevel("pile", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(30), upright_prob=0.2, bolt_collider="cd",
                grasp_point_tol=0.004, tilt=TiltCfg(prob=0.3), wrist_roll=True, retry_max=1,
                bin_walls=True, grip_check=False, pile=PileCfg(count=4)),
    # Level 27 with a CENTRED pinch in the latch: the pinch midpoint within 6 mm of the rod's axis.
    # Level 27's failed lifts were shallow pinches on the upper half of the rod (fingertips 5 mm
    # higher than on the lifts that held), formed where a neighbour stopped the fingers.
    28: RGLevel("centred_pinch", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(30), upright_prob=0.2, bolt_collider="cd",
                grasp_point_tol=0.004, tilt=TiltCfg(prob=0.3), wrist_roll=True, retry_max=1,
                bin_walls=True, grip_check=False, pile=PileCfg(count=4), pinch_centre_tol=0.006),
    # Level 28 with the grasp point FREE (level 17's device in the pile): no commanded point, the
    # policy sees the true rod and picks the clear part itself; the centred pinch stays binding.
    29: RGLevel("free_pile", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(45), free_grasp=True, yaw_rate_scale=0.25,
                upright_prob=0.2, bolt_collider="cd", tilt=TiltCfg(prob=0.3), wrist_roll=True, retry_max=1,
                bin_walls=True, grip_check=False, pile=PileCfg(count=4), pinch_centre_tol=0.006),
    # Level 29 in a DENSER pile: six neighbours 3 to 7 cm away, one of them dropped across a lying or
    # tilted target in 30 % of episodes (the real cover), and two retries.
    30: RGLevel("dense_pile", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(45), free_grasp=True, yaw_rate_scale=0.25,
                upright_prob=0.2, bolt_collider="cd", tilt=TiltCfg(prob=0.3), wrist_roll=True, retry_max=2,
                bin_walls=True, grip_check=False, pile=PileCfg(count=6, r=(0.03, 0.07), across_prob=0.3),
                pinch_centre_tol=0.006),
    # Level 30 with the grip check redone (v3): a 24 mm test lift over the whole pre-lift, measured at the
    # fingertips; a latched rod that does not follow sends the episode into a retry 40 steps earlier.
    # Three retries, since a failed attempt is cheaper now.
    31: RGLevel("early_retry", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(45), free_grasp=True, yaw_rate_scale=0.25,
                upright_prob=0.2, bolt_collider="cd", tilt=TiltCfg(prob=0.3), wrist_roll=True, retry_max=3,
                bin_walls=True, grip_check=True, pile=PileCfg(count=6, r=(0.03, 0.07), across_prob=0.3),
                pinch_centre_tol=0.006),
    # Level 31 with the first 10 pre-lift steps the POLICY's: after the settle the wrist is live again with
    # the fingers held, so a rod under a neighbour can be slid, tilted or lifted before the test lift.
    32: RGLevel("free_prelift", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(45), free_grasp=True, yaw_rate_scale=0.25,
                upright_prob=0.2, bolt_collider="cd", tilt=TiltCfg(prob=0.3), wrist_roll=True, retry_max=3,
                bin_walls=True, grip_check=True, pile=PileCfg(count=6, r=(0.03, 0.07), across_prob=0.3),
                pinch_centre_tol=0.006, free_prelift=True),
    # MIXED SCENES: level 32's policy with every scene of the ladder drawn per episode inside the bin:
    # the pile in 60 % of episodes (six rods, one across a lying target in 30 %), the cover block over
    # one end of 30 % of the lying rods, the bed rods under 40 % of them, and a plain bin otherwise.
    # The level-28 pick had lost the cover/bed scene after two pile levels (82 % at level 26).
    33: RGLevel("mixed_scenes", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(45), free_grasp=True, yaw_rate_scale=0.25,
                upright_prob=0.2, bolt_collider="cd", tilt=TiltCfg(prob=0.3), wrist_roll=True, retry_max=3,
                bin_walls=True, grip_check=True, cover=CoverCfg(prob=0.3), bed=BedCfg(prob=0.4),
                pile=PileCfg(count=6, r=(0.03, 0.07), across_prob=0.3, prob=0.6),
                pinch_centre_tol=0.006, free_prelift=True),
    # Level 33 with the grip check measured over the second half of the test lift only (v4).
    34: RGLevel("late_grip_check", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(45), free_grasp=True, yaw_rate_scale=0.25,
                upright_prob=0.2, bolt_collider="cd", tilt=TiltCfg(prob=0.3), wrist_roll=True, retry_max=3,
                bin_walls=True, grip_check=True, cover=CoverCfg(prob=0.3), bed=BedCfg(prob=0.4),
                pile=PileCfg(count=6, r=(0.03, 0.07), across_prob=0.3, prob=0.6),
                pinch_centre_tol=0.006, free_prelift=True, grip_check_late=True),
    # THE HEAP: ten rods in 70 % of episodes, six on the table around the target and four dropped from
    # 2.3 cm onto them and the target; the policy is frozen for the first 25 steps while the heap settles.
    35: RGLevel("heap", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(45), free_grasp=True, yaw_rate_scale=0.25,
                upright_prob=0.2, bolt_collider="cd", tilt=TiltCfg(prob=0.3), wrist_roll=True, retry_max=3,
                bin_walls=True, grip_check=True, cover=CoverCfg(prob=0.3), bed=BedCfg(prob=0.4),
                pile=PileCfg(count=10, upper=4, r=(0.03, 0.07), across_prob=0.3, prob=0.7, settle_in=25, top_rod=True),
                pinch_centre_tol=0.006, free_prelift=True, grip_check_late=True, retry_backoff=0.03),
    # Level 35 plus the heap map: the policy sees how much room there is around the rod in eight directions
    # (HEAP_MAP_DIRS), so it can choose a free pinch instead of finding one by touch. 82 observations; a
    # level-35 checkpoint is carried over with tools/expand_checkpoint.py (zero weights on the new inputs).
    36: RGLevel("heap_map", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(45), free_grasp=True, yaw_rate_scale=0.25,
                upright_prob=0.2, bolt_collider="cd", tilt=TiltCfg(prob=0.3), wrist_roll=True, retry_max=3,
                bin_walls=True, grip_check=True, cover=CoverCfg(prob=0.3), bed=BedCfg(prob=0.4),
                pile=PileCfg(count=10, upper=4, r=(0.03, 0.07), across_prob=0.3, prob=0.7, settle_in=25, top_rod=True),
                pinch_centre_tol=0.006, free_prelift=True, grip_check_late=True, retry_backoff=0.03, heap_map=True),
    # Level 36 with the clearance penalty on hard obstacles only: in a heap the fingertips have to displace a
    # touching neighbour to get beside the rod, and the penalty (tips margin 4 mm, palm 20 mm) paid for exactly that.
    37: RGLevel("heap_push", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(45), free_grasp=True, yaw_rate_scale=0.25,
                upright_prob=0.2, bolt_collider="cd", tilt=TiltCfg(prob=0.3), wrist_roll=True, retry_max=3,
                bin_walls=True, grip_check=True, cover=CoverCfg(prob=0.3), bed=BedCfg(prob=0.4),
                pile=PileCfg(count=10, upper=4, r=(0.03, 0.07), across_prob=0.3, prob=0.7, settle_in=25, top_rod=True),
                pinch_centre_tol=0.006, free_prelift=True, grip_check_late=True, retry_backoff=0.03, heap_map=True,
                push_ok=True),
    # Level 37 with the centred-pinch latch loosened to 10 mm. The latch trace of the level-36 pick (heap on):
    # two thirds of the episodes that never latch hold the rod with >= 4 N on both sides, but the pinch midpoint
    # sits ~11 mm above the axis (the neighbours keep the tips from reaching axis height) and the 6 mm rule
    # refuses it until the budget runs out. The grip check and the retries now decide whether a shallow pinch
    # holds, which is what they were built for.
    38: RGLevel("heap_tol", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS5, perception=_PERC7, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(45), free_grasp=True, yaw_rate_scale=0.25,
                upright_prob=0.2, bolt_collider="cd", tilt=TiltCfg(prob=0.3), wrist_roll=True, retry_max=3,
                bin_walls=True, grip_check=True, cover=CoverCfg(prob=0.3), bed=BedCfg(prob=0.4),
                pile=PileCfg(count=10, upper=4, r=(0.03, 0.07), across_prob=0.3, prob=0.7, settle_in=25, top_rod=True),
                pinch_centre_tol=0.010, free_prelift=True, grip_check_late=True, retry_backoff=0.03, heap_map=True,
                push_ok=True),
    # TRANSFER: level 32's scene and policy (the last level that passes the gate under the corrected eval,
    # 95.3 %), with everything the phase machine and the policy read replaced by what the robot can measure:
    # no fingertip sensors (tactile=False), the latch on joint torque and the estimated rod (hw_latch), the
    # grip check / retry / drop on the estimated height (sense_est), a camera-grade estimate with up to 4
    # steps of latency (_PERC8), gains 0.8-1.2 on the arm and the fingers (_PHYS6, arm_gains).
    39: RGLevel("transfer", standoff_lo=0.075, standoff_hi=0.10, dxy=0.02, dyaw=_D(7.5), drp=_D(2),
                physics=_PHYS6, perception=_PERC8, bolt_yaw_lo=_D(WAIST_WINDOW_DEG[0]),
                bolt_yaw_hi=_D(WAIST_WINDOW_DEG[1]), bolt_xy=0.05, free_waist=True,
                bolt_yaw_core=(_D(WAIST_CORE_DEG[0]), _D(WAIST_CORE_DEG[1])), bolt_xy_edge=0.025,
                grasp_offset=0.014, pinch_angle=_D(45), free_grasp=True, yaw_rate_scale=0.25,
                upright_prob=0.2, bolt_collider="cd", tilt=TiltCfg(prob=0.3), wrist_roll=True, retry_max=3,
                bin_walls=True, grip_check=True, pile=PileCfg(count=6, r=(0.03, 0.07), across_prob=0.3),
                pinch_centre_tol=0.006, free_prelift=True,
                tactile=False, hw_latch=True, sense_est=True, arm_gains=True),
}
# TRANSFER, the latch on camera-grade noise: level 39 with the settled test on the estimate's velocity over 5 steps
# (one-step differences of the noisy estimate read 0.165 m/s on rods moving 0.077 m/s with the hand), the centred
# tolerance back to level 32's 10 mm (the estimate's 8 mm bias put the pinch 9.1 mm off the ESTIMATED axis while it
# was 6.4 mm off the true one), and the retry's lifted test on the estimate. tools/probe_latch.py, 2026-10-13:
# 23 % of level-39 episodes held the rod and never latched, the settled and centred tests failing on 79 / 76 % of
# their gripped grasp steps.
LEVELS[40] = dataclasses.replace(LEVELS[39], name="transfer_latch", est_vel_window=5, pinch_centre_tol=0.010,
                                 sense_est_retry=True)
assert LEVELS[40].est_vel_window <= EST_WIN_MAX
# REJECTED (2026-10-13, TRAINING_LOG.md level 40: 77.3 -> 71.1 / 65.2 % at 0.2 / 0.4). Level 40 with a squeeze after the latch: in 45 % of level-40 grip checks the rod rose 10 mm of the fingertips'
# 21 mm (tools/probe_gripcheck.py), a grip resting on the closure the latch fired at. ADAM_U_RG_SQUEEZE sweeps it.
LEVELS[41] = dataclasses.replace(LEVELS[40], name="transfer_squeeze",
                                 latch_squeeze=float(os.environ.get("ADAM_U_RG_SQUEEZE", "0.3")))
# REJECTED (2026-10-13: 88.2 vs 88.1 %, TRAINING_LOG.md level 40). Level 40 without the free pre-lift: tools/probe_gripcheck.py found the grasps that slip lose the rod during the
# policy's 10-step wrist window after the settle (pinch 8 mm off the axis at the latch, still in hand at the settle's
# end, 19.5 mm off and ~1 N at the test lift's start). Held still, the test lift runs over the whole pre-lift.
LEVELS[42] = dataclasses.replace(LEVELS[40], name="transfer_still_prelift", free_prelift=False)
# Level 40 with the hand's effort as feedback: the grasps that slip sit ~8 mm off the true axis at the latch (the
# estimate's bias; the policy aligns to the estimate) and the policy, without fingertip sensors since level 39,
# cannot feel it. The tip_forces / contacts slots (zero since 39) carry the flexion-chain torque, which the robot
# reports as motor current. Same 74-wide interface.
LEVELS[43] = dataclasses.replace(LEVELS[40], name="transfer_effort", effort_obs=True)
# Level 43 with the bed of rods and the cover block back (level 32 had them; levels 39-43 were built on level 31's
# line and lost them: the level-43 pick scored 44.8 % on the bed, 70 % of episodes disqualified by the index,
# middle and thumb tips in the support rods). Train it with ADAM_U_HELD_CURL=0.95 (ring and little in a fist: with
# rods near a bin wall the flat little finger touched the wall in 10.9 % of episodes, the fist in 0 %; the
# level-43 pick untrained with the fist 90.9 % against 91.7 %). logs/chains/chain_L44.sh sets it.
LEVELS[44] = dataclasses.replace(LEVELS[43], name="transfer_scenes", bed=BedCfg(prob=0.4), cover=CoverCfg(prob=0.3))
# Level 44 with the hard scenes more often: the level-44 pick's cover row is 66.8 % (the covered end stays pinned:
# the rod touched the table in the hold in 31.9 %, 29 % never latched) and across 82 % on level 44's scene
# (the same for the level-43 pick and for flat fingers: the scene, not the fist or the training). Cover 0.3 -> 0.5
# of lying episodes, a neighbour across the target 0.3 -> 0.5. Train with ADAM_U_HELD_CURL=0.95.
LEVELS[45] = dataclasses.replace(LEVELS[44], name="transfer_hard_scenes", cover=CoverCfg(prob=0.5),
                                 pile=dataclasses.replace(LEVELS[44].pile, across_prob=0.5))
# Level 45's scene, its own number for the fine-tune: levels 40, 44 and 45 all trained flat while the entropy rose
# (1.72 -> 3.01 / 2.48: the std inflating from the 0.3 reset). logs/chains/chain_L46.sh trains it with
# --entropy-coef 0, --reset-std 0.2, --lr-schedule adaptive.
LEVELS[46] = dataclasses.replace(LEVELS[45], name="transfer_hard_scenes_ft")
# Level 45's scene with the scripted slide-out for a covered rod: the level-44 pick's cover row was 66.8 % with the
# covered end pinned under the block (it touched the table in the hold in 31.9 %), and neither more practice (45)
# nor other training settings (46) cleared it. The drag levels 23-31 had this slide; the robot needs the block's
# side from the camera (the drag direction is the rod's axis away from it). Train with ADAM_U_HELD_CURL=0.95.
LEVELS[47] = dataclasses.replace(LEVELS[45], name="transfer_cover_drag", cover_drag=True)
# Level 45's scene with the covered rod's target moved off the block: tools/probe_latch.py on the cover scene found 19.9 %
# of episodes never gripped at all (0.4 % gripped and failed to latch); a pinch at the rod's centre, 5 mm from the
# block's edge, puts the fingers on the block. The estimate of a covered rod points 6.25 mm along the axis away from
# the block (the exposed part's middle). The robot needs the block's side from the camera. ADAM_U_HELD_CURL=0.95.
LEVELS[48] = dataclasses.replace(LEVELS[45], name="transfer_cover_shift", cover_shift=True)
# Both cover changes together, for training: the shifted target (48: 64.7 -> 67.8 % untrained on the cover row) and the
# slide-out (47: -> 66.6 %). ADAM_U_HELD_CURL=0.95; the same settings as level 45, so the two differ in the scene only.
LEVELS[49] = dataclasses.replace(LEVELS[45], name="transfer_cover", cover_shift=True, cover_drag=True)
# THE FULL BIN: adam-moveit's bin (workcell.json: inner 22 x 16 cm, walls 8 cm, 8 mm thick, centred on BOLT_XY here)
# filled with a settled heap from tools/make_bin_heaps.py in 70 % of episodes (the other 30 % keep level 44's scenes,
# inside the same bin). The target is an exposed heap rod in the reachable window (+/-5 cm, +/-2.5 cm at the yaw
# window's edges), lying / leaning / standing 0.5 / 0.3 / 0.2; the heaps settle to 29 / 42 / 29 % lying / leaning /
# standing rods. 15 frozen steps while the placed heap settles under the arm; the clearance penalty on hard obstacles
# only (level 37: in a heap the fingers have to displace a touching neighbour). Level 44's line, the fist
# (ADAM_U_HELD_CURL=0.95). 50: 15-25 rods (one loose layer), 51: 40-60 (two layers, adam-moveit's count).
_REAL_BIN = BinCfg(inner=(0.22, 0.16), height=0.08, wall=0.008)
LEVELS[50] = dataclasses.replace(LEVELS[44], name="bin_sparse", bin=_REAL_BIN, bin_walls=True, push_ok=True,
                                 fill=FillCfg(prob=0.7, file="bin_heaps_sparse.npz"),
                                 pile=dataclasses.replace(LEVELS[44].pile, settle_in=15))
LEVELS[51] = dataclasses.replace(LEVELS[50], name="bin_full", fill=FillCfg(prob=0.7, file="bin_heaps_full.npz"))
# The full bin with perception's choice of rod (docs/PICKABILITY_REPORT.md: the level-51 pick scores 66.9 % on rods 35 mm or
# more up, 17.8 % under 25 mm): the target is the highest exposed reachable rod, the rule genesis_policy's selector applies,
# in bins at three fill levels so the low rods are practised where they will be picked, when the heap has run down.
# Heap episodes 85 %. Train from the level-51 pick, ADAM_U_HELD_CURL=0.95. logs/chains/chain_L52.sh.
LEVELS[52] = dataclasses.replace(LEVELS[51], name="bin_highest",
                                 fill=FillCfg(prob=0.85, pick="highest", files=(("bin_heaps_full.npz", 0.4),
                                                                                ("bin_heaps_sparse.npz", 0.35),
                                                                                ("bin_heaps_empty.npz", 0.25))))
# Evaluation only: level 52 with 4 cm walls (the virtual bin's height, levels 25-49). Does the 8 cm wall cost the
# floor rods? The level-52 pick scores 28-35 % on half-full and near-empty bins, against 83 % for the level-44 pick
# on single rods in the 4 cm bin.
LEVELS[53] = dataclasses.replace(LEVELS[52], name="bin_highest_low_walls",
                                 bin=dataclasses.replace(LEVELS[52].bin, height=0.04))
LEVEL = int(os.environ.get("ADAM_U_RG_LEVEL", "0"))
if LEVEL not in LEVELS:
    raise ValueError(f"ADAM_U_RG_LEVEL must be one of {sorted(LEVELS)}, got {LEVEL}")
if LEVELS[LEVEL].fill.active:
    # a neighbour slot per heap rod: the capsule buffers are sized from these at the first reset
    NEIGHBOR_SLOTS = tuple(range(2, 2 + FILL_RODS - 1))
    MAX_CAPSULES = 2 + len(NEIGHBOR_SLOTS)

# ---------------------------------------------------------------------------
# Reward and latch constants
# ---------------------------------------------------------------------------

REACH_SIGMA_POS = 0.08          # m, wrist position error scale
REACH_SIGMA_ROT = 0.26          # rad, wrist orientation error scale
REACH_WEIGHT = 2.0
FINGER_SIGMA = 0.02             # m, fingertip-to-bolt-surface scale
FINGER_WEIGHT = 2.0
VEL_GATE_SIGMA = 0.04           # m, g(d_w) = 1 - tanh(d_w / 0.04)
VEL_LIN_COEF, VEL_LIN_SCALE = 0.02, 0.10      # m/s
VEL_ANG_COEF, VEL_ANG_SCALE = 0.01, 0.5       # rad/s
CLEARANCE_WEIGHT = -0.02
SHIELD_WEIGHT = -0.1

GRIP_MIN_N = 4.0                # N per side: a grip, not a touch (pipeline: latching on touch cost 26 points)
# Commanded-point shaping (binding levels only): potential 1 - tanh(|pinch - point| / sigma) on the pinch
# centre's position along the rod's true axis, grasp phase, paid as progress like the reach term. Before it,
# the policy learned the point only from the latch refusing off-point grasps.
GRASP_POINT_SIGMA = 0.01        # m
GRASP_POINT_WEIGHT = 1.0
GRASP_REGION = 0.05             # m, bolt to pinch centre
REL_VEL_MAX = 0.10              # m/s, bolt moving with the hand
STRUCT_HIT_N = 5.0              # N, palm / forearm / upper arm on the table at all
LINK_HIT_N = 15.0               # N, finger phalanges on the table
BOLT_TOUCH_N = 2.0              # N, a non-whitelisted link on the bolt (numerical-noise threshold)

# ---------------------------------------------------------------------------
# Proxy geometry: spheres attached to bodies, for clearance before contact
# ---------------------------------------------------------------------------
#
# Isaac Lab has no distance query, so clearance is computed analytically on a
# sphere model of the right arm and hand. Offsets are in each BODY frame,
# sized from the collision-mesh bounding boxes (measured 2026-09-24):
#
#   r_wrist (merged into wristRollRight)   x -0.045..0.029  y +/-0.017  z -0.079..0.007
#   wristPitchRight (knuckle)              0.037 x 0.057 x 0.038 about its origin
#   wristYawRight (forearm)                0.097 x 0.091 x 0.257, centred at z -0.095
#   elbowRight                             0.086 x 0.069 x 0.114, centred at (-0.013, -0.029, -0.031)
#   shoulderYawRight (upper arm)           0.101 x 0.098 x 0.233, centred at z -0.119
#   distal phalanges                       ~0.016 x 0.012 x 0.031, tip along -z
#
# Deliberately approximate: the clearance OBSERVATION only has to be monotone
# in the true clearance, and the margins are validated by probes, not assumed.
# tools/probe_link_geometry.py (second increment) refines these against the meshes.

_TIP = dict(TIP_OFFSETS)
_TIP["ring"] = _TIP["index"]
_TIP["pinky"] = _TIP["index"]

PROXY_GROUPS: dict[str, list[tuple[str, tuple, float]]] = {
    "tips": (
        [(TIP_BODIES[k], (_TIP[k][0], _TIP[k][1], _TIP[k][2] + 0.006), 0.006) for k in TIP_BODIES]
        + [(TIP_BODIES[k], (0.0, 0.0, _TIP[k][2] / 2), 0.007) for k in TIP_BODIES]
        + [(b, (0.0, 0.0, -0.02), 0.008) for b in ("r_thumb_middle", "r_index_finger_middle",
                                                    "r_middle_finger_middle")]
    ),
    "palm": (
        [(WRIST_BODY, (x, 0.0, z), 0.020) for x in (-0.025, 0.010) for z in (-0.010, -0.040, -0.070)]
        + [("wristPitchRight", (0.0, 0.0, 0.0), 0.030)]
    ),
    "forearm": (
        [("elbowRight", (-0.013, -0.029, -0.031), 0.045)]
        + [("wristYawRight", (0.0, 0.0, z), 0.048) for z in (-0.020, -0.095, -0.170)]
    ),
    "upperarm": (
        [("shoulderYawRight", (0.001, -0.003, z), 0.050) for z in (-0.020, -0.120, -0.220)]
        + [("shoulderRollRight", (-0.033, -0.017, -0.010), 0.065)]
    ),
}
GROUP_NAMES = ("tips", "palm", "forearm", "upperarm")
# Clearance margins m_g for the penalty and (later) the shield. Spec: tips
# 3-5 mm, palm 20 mm, forearm / upper arm 30 mm.
MARGINS = {"tips": 0.004, "palm": 0.020, "forearm": 0.030, "upperarm": 0.030}

# Contact-monitored bodies. Bolt contact is permitted ONLY for the grasp
# links (thumb, index, middle: distal, middle and proximal phalanges).
STRUCT_SENSORS = {"palm": "cs_palm", "knuckle": "cs_forearm", "forearm": "cs_forearm2",
                  "elbow": "cs_upperarm", "upperarm": "cs_upperarm2"}
PHALANX_SENSORS = ("cs_thumb_mid", "cs_index_mid", "cs_middle_mid",
                   "cs_thumb_prox", "cs_index_prox", "cs_middle_prox")
NON_GRASP_TIP_SENSORS = ("cs_ring", "cs_pinky")

# Table box, env-relative, for the analytic clearance.
_TC, _TS = LAYOUT["table_center"], LAYOUT["table_size"]
TABLE_LO = (_TC[0] - _TS[0] / 2, _TC[1] - _TS[1] / 2, _TC[2] - _TS[2] / 2)
TABLE_HI = (_TC[0] + _TS[0] / 2, _TC[1] + _TS[1] / 2, _TC[2] + _TS[2] / 2)
assert abs(TABLE_HI[2] - TABLE_TOP) < 1e-6, (TABLE_HI[2], TABLE_TOP)


# ---------------------------------------------------------------------------
# Per-env state
# ---------------------------------------------------------------------------


def _rg(env) -> dict:
    """Experiment 4's own bookkeeping. Reset by `reset_rg_buffers`; `last_*`
    survive the reset so evaluation can read the ENDED episode's outcome."""
    b = getattr(env, "_rg", None)
    if b is None:
        n, d = env.num_envs, env.device
        z = lambda: torch.zeros(n, dtype=torch.bool, device=d)  # noqa: E731
        f = lambda k=1: torch.zeros((n, k) if k > 1 else (n,), device=d)  # noqa: E731
        b = dict(
            # contact
            streak=f(3), tip_force=f(3), thumb_stable=z(), finger_stable=z(),
            paid_thumb=z(), paid_finger=z(), paid_opposing=z(),
            # disqualifiers
            table_hit=z(), excess=z(), struct_hit=z(), bolt_touch=z(), invalid=z(),
            invalid_which=torch.zeros(n, dtype=torch.long, device=d), invalid_force=f(),
            paid_excess=z(), paid_invalid=z(), paid_drop=z(),
            # lift / hold
            hold_streak=f(), was_up=z(), verify_clean=z(), success=z(), final=z(),
            attempts=f(), last_attempts=f(), last_retry_success=z(),    # retry bookkeeping (level 22+)
            _retry_step=torch.full((n,), -1.0, device=d),              # the step a retry began (progress terms re-seed)
            lift_bolt_xy=f(2), drag=f(),
            # latch / phases
            grasp_end=torch.full((n,), float(GRASP_MAX_STEPS), device=d),
            latched=z(), latch_step=torch.full((n,), -1.0, device=d),
            latch_wrist_pos=f(3), latch_wrist_quat=f(4), latch_q=f(7), latch_closure=f(2),
            # shaping potentials
            prev_reach=f(), prev_fingers=f(),
            # clearance (per group) and shield
            clear=torch.ones(n, len(GROUP_NAMES), device=d), min_clear=torch.ones(n, len(GROUP_NAMES), device=d),
            alpha=torch.ones(n, device=d), shield_steps=f(),
            # wrist kinematics
            prev_v=f(3), prev_a=f(3), acc=f(3), jerk=f(3), jerk_max=f(),
            # perception
            est_pos=f(3), est_quat=f(4), bias_pos=f(3), bias_rotvec=f(3), noise_std=f(),
            latency=torch.zeros(n, dtype=torch.long, device=d),
            hist_pos=f(HIST_DEPTH * 3).view(n, HIST_DEPTH, 3), hist_quat=f(HIST_DEPTH * 4).view(n, HIST_DEPTH, 4), hist_n=f(),
            tau_streak=f(3), _est_prev=f(3), _rest_est=f(), _est_win=f(EST_WIN_MAX * 3).view(n, EST_WIN_MAX, 3),
            # the sampled bolt pose (dx, dy, yaw) and the bolt-relative nominal wrist pose
            _bolt_sample=f(3),
            _grasp_s=f(),                # commanded grasp offset along the rod's axis this episode (m)
            _pinch_delta=f(),            # commanded pinch angle: hand yaw - rod yaw (rad)
            _upright=torch.zeros(n, dtype=torch.bool, device=d),   # the rod stands on its end this episode
            _tilted=torch.zeros(n, dtype=torch.bool, device=d),    # the rod leans on a support block
            _tilt=f(),                   # its spawn tilt from the table (rad); 0 lying, pi/2 standing
            _support_j=torch.full((n,), -1, dtype=torch.long, device=d),   # which support slot, -1 none
            _support_c=f(3),             # the support block's centre (env-relative)
            _cover_side=torch.zeros(n, dtype=torch.long, device=d),   # +1 / -1: which end of a lying rod is covered; 0 none
            _on_bed=torch.zeros(n, dtype=torch.bool, device=d),        # the target rests across two rods
            _heap_swapped=torch.zeros(n, dtype=torch.bool, device=d),  # the heap re-labelled the target (top-rod rule)
            last_heap_swapped=torch.zeros(n, dtype=torch.bool, device=d),
            _bed_c=torch.full((n, MAX_CAPSULES, 3), -10.0, device=d),  # bed rod centres (env-relative)
            _bed_axis=f(3),              # bed rods' axis direction (horizontal, perpendicular to the target)
            _cap_p0=torch.full((n, MAX_CAPSULES, 3), -10.0, device=d),
            _cap_p1=torch.full((n, MAX_CAPSULES, 3), -10.0, device=d),
            _cap_r=torch.zeros(n, MAX_CAPSULES, device=d),
            _cap_hard=torch.ones(n, MAX_CAPSULES, dtype=torch.bool, device=d),   # penetration counts as a collision
            _nb_rest_z=torch.zeros(n, len(NEIGHBOR_SLOTS), device=d),           # neighbours' spawn height (double-pick check)
            double=z(), last_double=z(),                                          # a neighbour lifted with the target
            _cover_c=f(3),               # the cover block's centre (env-relative)
            _drag_dir=f(3),              # slide-out direction for a covered rod (horizontal unit, away from the block); zero otherwise
            _pre_bolt_z=f(), _pre_wrist_z=f(),   # heights at the start of the test lift (grip check; wrist_z holds the fingertip mean since v3)
            grip_fired=z(), last_grip_fired=z(),  # the grip check sent this episode into a retry
            _fill=z(),                   # the target is a rod of a full-bin heap this episode (level 50+)
            _fill_pos=f(FILL_RODS * 3).view(n, FILL_RODS, 3), _fill_ax=f(FILL_RODS * 3).view(n, FILL_RODS, 3),   # its heap, env frame
            _fill_cnt=torch.zeros(n, dtype=torch.long, device=d), _fill_j=torch.zeros(n, dtype=torch.long, device=d),
            _rest_z=torch.full((n,), BOLT_REST_Z, device=d),       # the rod's rest height this episode (lift is measured from it); underscore: survives the buffer reset
            grasp_s_err=f(),             # |pinch centre along the rod - commanded offset| this step (m)
            prev_point=f(),              # last step's commanded-point potential (shaping)
            latch_s_err=f(),             # the same, frozen at the latch step
            last_latch_s_err=f(),        # ENDED episodes: the latch-step value (0 if never latched)
            grasp_d_err=f(),             # pinch midpoint to the rod's axis this step (m): the depth of the pinch
            latch_d_err=f(), last_latch_d_err=f(),   # the same, frozen at the latch; and for ended episodes
            _nom_dz=f(),                 # height of the grasp point above the lying rod's rest height (m)
            _lift_dir=torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, 3).clone(),   # approach / lift axis (world)
            _roll_axis=f(3),             # horizontal axis n = z x u of the rod's tilt (world); zero when unrolled
            _roll=f(),                   # the wrist roll actually applied (rad): min(tilt, roll_cap)
            lift_rel_xy=f(2),            # bolt - wrist xy at lift entry (drag is slip in the hand)
            _hand_yaw=f(),               # the hand's nominal yaw this episode (rad); == rod yaw below level 16
            _grasp_shift=f(2),           # its world xy component: where the nominal wrist moved to
            # held waist (pitch, yaw, roll) this episode; the pipeline's hold unless the level frees it
            _waist=torch.tensor(WAIST_NOMINAL, device=d).expand(n, 3).clone(),
            _nom_pos=torch.tensor([WRIST_XY[0], WRIST_XY[1], WRIST_Z_CONTACT], device=d).expand(n, 3).clone(),
            _nom_quat=torch.tensor(WRIST_ROT, device=d).expand(n, 4).clone(),
            # obstacle AABBs, env-relative, (N, MAX_OBSTACLES, 3); parked = far away
            _obs_lo=torch.full((n, MAX_OBSTACLES, 3), -10.0, device=d),
            _obs_hi=torch.full((n, MAX_OBSTACLES, 3), -10.0, device=d),
            _obs_active=torch.zeros(n, MAX_OBSTACLES, dtype=torch.bool, device=d),
            obs_clear=torch.ones(n, device=d), min_obs_clear=torch.ones(n, device=d),
            # ENDED-episode outcomes, not cleared by the reset
            last_success=z(), last_held=z(), last_invalid=z(), last_invalid_which=torch.zeros(n, dtype=torch.long, device=d),
            last_excess=z(), last_drop=z(), last_was_up=z(), last_verify_clean=z(),
            last_latched=z(), last_grasp_end=f(), last_closure=f(2), last_drag=f(),
            last_min_clear=torch.ones(n, len(GROUP_NAMES), device=d), last_shield_steps=f(),
            last_jerk_max=f(), last_thumb_frac=f(), last_finger_frac=f(), last_min_obs_clear=f(),
            thumb_steps=f(), finger_steps=f(),
        )
        env._rg = b
    return b


def _ids(env) -> dict:
    """Body / joint indices and the proxy-sphere tables, resolved ONCE."""
    c = getattr(env, "_rg_ids", None)
    if c is None:
        robot = env.scene["robot"]
        wrist, _ = robot.find_bodies(WRIST_BODY)
        arm, _ = robot.find_joints(RIGHT_ARM, preserve_order=True)
        tips = {}
        for k in ("thumb", "index", "middle"):
            ids, _ = robot.find_bodies(TIP_BODIES[k])
            tips[k] = ids[0]
        groups = {}
        for g, spheres in PROXY_GROUPS.items():
            idx, off, rad = [], [], []
            for body, o, r in spheres:
                ids, _ = robot.find_bodies(body)
                if len(ids) != 1:
                    raise ValueError(f"proxy body {body} matched {len(ids)} bodies")
                idx.append(ids[0]); off.append(o); rad.append(r)
            groups[g] = (torch.tensor(idx, device=env.device),
                         torch.tensor(off, device=env.device, dtype=torch.float),
                         torch.tensor(rad, device=env.device, dtype=torch.float))
        c = dict(wrist=wrist[0], arm=arm, tips=tips, groups=groups,
                 table_lo=torch.tensor(TABLE_LO, device=env.device),
                 table_hi=torch.tensor(TABLE_HI, device=env.device),
                 nominal_pos=torch.tensor([WRIST_XY[0], WRIST_XY[1], WRIST_Z_CONTACT], device=env.device),
                 nominal_quat=torch.tensor(WRIST_ROT, device=env.device),
                 bolt_spawn_quat=tuple(env.cfg.scene.object.init_state.rot),
                 margins=torch.tensor([MARGINS[g] for g in GROUP_NAMES], device=env.device))
        env._rg_ids = c
    return c


def _wrist_pose(env):
    c = _ids(env)
    robot = env.scene["robot"]
    pos = _t(robot.data.body_link_pos_w)[:, c["wrist"], :] - env.scene.env_origins
    quat = _t(robot.data.body_link_quat_w)[:, c["wrist"], :]
    return pos, quat


def _bolt_true(env):
    pos = _t(env.scene["object"].data.root_pos_w) - env.scene.env_origins
    return pos, _t(env.scene["object"].data.root_quat_w)


# ---------------------------------------------------------------------------
# Phases and the latch
# ---------------------------------------------------------------------------


def phase_of(env) -> torch.Tensor:
    """0 grasp, 1 settle, 2 lift, 3 hold -- PER ENV, from this step's latch state."""
    b = _rg(env)
    t = env.episode_length_buf.float()
    g = b["grasp_end"]
    return torch.where(t < g, 0, torch.where(t < g + LIFT_START, 1,
                       torch.where(t < g + LIFT_START + LIFT_STEPS, 2, 3))).long()


def fingertip_distances_true(env) -> torch.Tensor:
    """(N, 3) thumb / index / middle fingertip distance to the bolt's CAPSULE surface, ground truth."""
    return _fingertip_distances(env, *_bolt_true(env))


def _tip_positions(env) -> list[torch.Tensor]:
    """Thumb, index, middle fingertip pad positions, env-relative, (N, 3) each."""
    robot = env.scene["robot"]
    c = _ids(env)
    out = []
    for k in ("thumb", "index", "middle"):
        i = c["tips"][k]
        off = torch.tensor(TIP_OFFSETS[k], device=env.device).expand(env.num_envs, 3)
        out.append(_t(robot.data.body_link_pos_w)[:, i, :] - env.scene.env_origins
                   + math_utils.quat_apply(_t(robot.data.body_link_quat_w)[:, i, :], off))
    return out


def pinch_along_axis(env) -> torch.Tensor:
    """Where the pinch is along the TRUE rod: the thumb-vs-fingers midpoint projected on the axis (m from the centre)."""
    thumb, index, middle = _tip_positions(env)
    mid = 0.5 * (thumb + 0.5 * (index + middle))
    bp, bq = _bolt_true(env)
    axis = math_utils.quat_apply(bq, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))
    return ((mid - bp) * axis).sum(dim=1)


def pinch_centre_dist(env) -> torch.Tensor:
    """How far the pinch midpoint (thumb vs fingers) sits from the TRUE rod's axis (m). A pinch
    that closed on the upper half of the rod has the midpoint well above the axis; those slip
    during the lift (level 27 analysis: 8 mm for the lifts that failed, 5 mm for the ones that held)."""
    thumb, index, middle = _tip_positions(env)
    mid = 0.5 * (thumb + 0.5 * (index + middle))
    bp, bq = _bolt_true(env)
    axis = math_utils.quat_apply(bq, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))
    v = mid - bp
    s = (v * axis).sum(dim=1, keepdim=True)
    return torch.norm(v - s * axis, dim=1)


def _fingertip_distances(env, bolt_p, bolt_q) -> torch.Tensor:
    axis = math_utils.quat_apply(bolt_q, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))
    out = []
    for tip in _tip_positions(env):
        v = tip - bolt_p
        s = (v * axis).sum(dim=1, keepdim=True).clamp(-BOLT_HALF_LEN, BOLT_HALF_LEN)
        out.append((torch.norm(v - s * axis, dim=1) - BOLT_RADIUS).clamp(min=0.0))
    return torch.stack(out, dim=1)


def _clearance(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group clearance to the table box and the nearest-obstacle vector.

    Returns (N, G) distances and (N, G, 3) WORLD vectors from the closest point
    on the link proxy to the closest point on the obstacle. Computed once per step.
    """
    b = _rg(env)
    if b.get("_clear_step") == int(env.common_step_counter):
        return b["clear"], b["_clear_vec"]
    b["_clear_step"] = int(env.common_step_counter)
    c = _ids(env)
    _refresh_neighbor_capsules(env)
    robot = env.scene["robot"]
    pos_w = _t(robot.data.body_link_pos_w) - env.scene.env_origins.unsqueeze(1)
    quat_w = _t(robot.data.body_link_quat_w)
    n = env.num_envs
    ar = torch.arange(n, device=env.device)
    # Boxes: the table, then every obstacle slot (parked slots sit far away).
    lo = torch.cat([c["table_lo"].view(1, 1, 3).expand(n, 1, 3), b["_obs_lo"]], dim=1)   # (N, B, 3)
    hi = torch.cat([c["table_hi"].view(1, 1, 3).expand(n, 1, 3), b["_obs_hi"]], dim=1)
    B = lo.shape[1]
    # Capsules (bed rods): parked ones sit far away with radius 0.
    p0, p1, cr = b["_cap_p0"], b["_cap_p1"], b["_cap_r"]                       # (N, C, 3), (N, C)
    seg = p1 - p0
    seg_len2 = (seg * seg).sum(dim=2).clamp(min=1e-9)                          # (N, C)
    dists, vecs, obs_d, hard_d = [], [], [], []
    for g in GROUP_NAMES:
        idx, off, rad = c["groups"][g]
        p = pos_w[:, idx, :]                                                   # (N, K, 3)
        q = quat_w[:, idx, :]
        K = idx.numel()
        centre = p + math_utils.quat_apply(q.reshape(-1, 4), off.unsqueeze(0).expand(n, K, 3)
                                           .reshape(-1, 3)).view(n, K, 3)
        cc = centre.unsqueeze(2)                                               # (N, K, 1, 3)
        closest_b = torch.maximum(torch.minimum(cc, hi.unsqueeze(1)), lo.unsqueeze(1))   # (N, K, B, 3)
        # capsules: closest point on each segment to each sphere centre
        w = cc - p0.unsqueeze(1)                                               # (N, K, C, 3)
        tt = ((w * seg.unsqueeze(1)).sum(dim=3) / seg_len2.unsqueeze(1)).clamp(0.0, 1.0)   # (N, K, C)
        closest_c = p0.unsqueeze(1) + tt.unsqueeze(3) * seg.unsqueeze(1)      # (N, K, C, 3)
        closest = torch.cat([closest_b, closest_c], dim=2)                     # (N, K, B + C, 3)
        v = closest - cc
        dist_c = torch.norm(v, dim=3)                                          # (N, K, B + C) centre to obstacle axis / box
        surf = torch.cat([torch.zeros(n, 1, B, device=env.device).expand(n, K, B), cr.unsqueeze(1).expand(n, K, cr.shape[1])], dim=2)
        d = dist_c - surf - rad.view(1, K, 1)                                  # surface to surface (negative = inside)
        B_all = B + cr.shape[1]
        dflat = d.view(n, K * B_all)
        dmin, kb = dflat.min(dim=1)
        k, bb = kb // B_all, kb % B_all
        vk = v[ar, k, bb]
        dk = dist_c[ar, k, bb].unsqueeze(1).clamp(min=1e-9)
        vec = vk / dk * dmin.unsqueeze(1)                                      # from proxy surface to obstacle
        dists.append(dmin); vecs.append(vec)
        # the collision distance ignores SOFT capsules (pushing a neighbour rod is allowed); the observation keeps them
        soft = torch.cat([torch.zeros(n, B, dtype=torch.bool, device=env.device), ~b["_cap_hard"]], dim=1)   # (N, B_all)
        d_hard = torch.where(soft.unsqueeze(1), torch.full_like(d, 10.0), d)
        obs_d.append(d_hard[:, :, 1:].reshape(n, -1).min(dim=1).values if B_all > 1 else torch.ones(n, device=env.device))
        hard_d.append(d_hard.reshape(n, -1).min(dim=1).values)                  # table + hard obstacles, per group
    b["clear"] = torch.stack(dists, dim=1)
    b["clear_hard"] = torch.stack(hard_d, dim=1)                                 # for the penalty at push_ok levels
    b["_clear_vec"] = torch.stack(vecs, dim=1)
    b["min_clear"] = torch.minimum(b["min_clear"], b["clear"])
    # Obstacle-only clearance (any group), for the invalid-collision rule and the logs.
    b["_obs_clear_groups"] = torch.stack(obs_d, dim=1)                         # (N, G)
    b["obs_clear"] = b["_obs_clear_groups"].min(dim=1).values
    b["min_obs_clear"] = torch.minimum(b["min_obs_clear"], b["obs_clear"])
    return b["clear"], b["_clear_vec"]


def _force_on(env, sensor: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(table_force_N, other_force_N) for a body sensor filtered on the table.
    'other' is net minus table: with self-collision off and nothing else in
    reach, that is the bolt."""
    s = env.scene[sensor]
    fm = _t(s.data.force_matrix_w)[:, 0, 0, :]
    net = _t(s.data.net_forces_w)[:, 0, :]
    return torch.norm(fm, dim=1), torch.norm(net - fm, dim=1)


def _rerun_fresh(env, b, key, body):
    """The post-reset observations are computed in the same step as the outcomes: re-run a once-per-step body
    for the envs that just reset (b[key], set by reset_rg_buffers) and restore every other env's buffers, so no
    counter of theirs advances twice in one step. (Before 2026-10-13 the caches were dropped instead and every
    other env's streaks, hold count and estimate history advanced twice on any step in which some env reset.)"""
    fresh = b.pop(key)
    keep = ~fresh
    snap = {k: v.clone() for k, v in b.items()
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == env.num_envs}
    out = body(env)
    for k, v0 in snap.items():
        v = b.get(k)
        if torch.is_tensor(v) and v.shape == v0.shape:
            v[keep] = v0[keep]
    return out


def _update(env) -> dict:
    """Once-per-step bookkeeping: contacts, disqualifiers, lift / hold, outcomes."""
    b = _rg(env)
    if b.get("_step") == int(env.common_step_counter):
        if b.get("_fresh_upd") is not None:
            _rerun_fresh(env, b, "_fresh_upd", _update_body)
        return b
    b["_step"] = int(env.common_step_counter)
    b.pop("_fresh_upd", None)
    return _update_body(env)


def _update_body(env) -> dict:
    b = _rg(env)
    n, d = env.num_envs, env.device
    t = env.episode_length_buf
    lvl0 = LEVELS[env.cfg.actions.pipeline.level]
    if lvl0.pile.top_rod and lvl0.pile.settle_in > 1:
        _heap_top_rod(env, t == lvl0.pile.settle_in - 1)      # one step before the rest heights are recorded

    # -- fingertip contact with the bolt -------------------------------------
    tf = {k: _tip_forces(env, k) for k in TIP_BODIES}
    b["tip_force"] = torch.stack([tf[k][0] for k in ("thumb", "index", "middle")], dim=1)
    touching = b["tip_force"] > CONTACT_N
    b["streak"] = torch.where(touching, b["streak"] + 1, torch.zeros_like(b["streak"]))
    stable = b["streak"] >= STABLE_STEPS
    b["thumb_stable"], b["finger_stable"] = stable[:, 0], stable[:, 1] | stable[:, 2]
    # the robot's version of the same thing: flexion torque per finger (motor current there), streaked the same way
    tau = _grip_torque(env)
    b["tau_streak"] = torch.where(tau > GRIP_TAU_MIN, b["tau_streak"] + 1, torch.zeros_like(b["tau_streak"]))
    if lvl0.sense_est:
        ep_, _ = _est_bolt(env)
        # the estimate's rest height, read once at the start (after the settle-in: the heap moves the rod until then)
        b["_rest_est"] = torch.where(t == max(1, lvl0.pile.settle_in), ep_[:, 2], b["_rest_est"])

    # -- disqualifiers ----------------------------------------------------------
    tip_table = torch.stack([tf[k][1] for k in TIP_BODIES], dim=1).max(dim=1).values
    phal_table = torch.stack([_force_on(env, s)[0] for s in PHALANX_SENSORS], dim=1).max(dim=1).values
    table_now = (tip_table > TABLE_HIT_N) | (phal_table > LINK_HIT_N)
    struct = torch.stack([_force_on(env, s)[0] for s in STRUCT_SENSORS.values()], dim=1)
    struct_now = struct.max(dim=1)
    struct_hit = struct_now.values > STRUCT_HIT_N
    # Non-whitelisted links on the bolt: structural bodies and the ring / little tips.
    other = torch.stack([_force_on(env, s)[1] for s in STRUCT_SENSORS.values()]
                        + [tf["ring"][0], tf["pinky"][0]], dim=1)
    bolt_touch = other.max(dim=1).values > BOLT_TOUCH_N
    excess_now = torch.stack([tf[k][2] for k in TIP_BODIES], dim=1).max(dim=1).values > F_MAX

    # Obstacles: any proxy sphere more than OBSTACLE_PEN_M inside an obstacle
    # box. Analytic rather than a PhysX filter: the fingers are ALLOWED to
    # touch the bolt, so their net-minus-table force cannot say which object
    # they hit; the box geometry can. Structural links hitting an obstacle also
    # show up in `other` above and are caught either way.
    _clearance(env)
    obs_hit = b["obs_clear"] < -OBSTACLE_PEN_M
    if LEVELS[env.cfg.actions.pipeline.level].pile.active:
        # a force on a structural link cannot tell the target from a neighbour rod: in a pile, touching
        # neighbours is allowed, so the force-based "non-grasp link on bolt" rule is off here
        bolt_touch = torch.zeros_like(bolt_touch)

    # A force-detected contact on a non-grasp link while an obstacle is within
    # a centimetre of the arm is an obstacle hit: the force fires on first
    # touch, before the 4 mm proxy penetration does.
    near_obs = b["obs_clear"] < 0.01
    fresh = (table_now | struct_hit | bolt_touch | obs_hit) & ~b["invalid"]
    # which: 1 fingertip/phalanx-table, 2 structural-table, 3 bolt-touch (non-grasp link), 4 obstacle
    which = torch.where(obs_hit | (bolt_touch & near_obs), 4, torch.where(struct_hit, 2, torch.where(bolt_touch, 3, 1)))
    b["invalid_which"] = torch.where(fresh, which, b["invalid_which"])
    b["invalid_force"] = torch.where(fresh, torch.maximum(torch.maximum(tip_table, phal_table),
                                                          torch.maximum(struct_now.values, other.max(dim=1).values)),
                                     b["invalid_force"])
    b["table_hit"] |= table_now
    b["struct_hit"] |= struct_hit
    b["bolt_touch"] |= bolt_touch
    b["invalid"] |= table_now | struct_hit | bolt_touch | obs_hit
    b["excess"] |= excess_now

    # -- lift and hold -----------------------------------------------------------
    g = b["grasp_end"]
    z = _t(env.scene["object"].data.root_pos_w)[:, 2] - b["_rest_z"]
    up = z > SUCCESS_LIFT                                                # success: the TRUE rod, always
    b["hold_streak"] = torch.where(up, b["hold_streak"] + 1, torch.zeros_like(b["hold_streak"]))
    after_grasp = t.float() > g
    if lvl0.sense_est:
        z = _est_bolt(env)[0][:, 2] - b["_rest_est"]                    # the phase machine sees the estimate, as the robot would
    b["was_up"] |= after_grasp & (z > UP_THRESHOLD)
    verifying = t.float() >= g + LIFT_START + LIFT_STEPS
    b["verify_clean"] &= ~(verifying & (bolt_table_clearance(env) <= 0.0))
    grasping = t.float() < g
    b["thumb_steps"] += (b["thumb_stable"] & grasping).float()
    b["finger_steps"] += (b["finger_stable"] & grasping).float()

    # Bolt drag: the bolt's xy RELATIVE TO THE WRIST at the first lift step vs now, i.e. slip in the hand.
    # (Was the bolt's world xy travel; identical for a vertical lift, and the lift along a rolled
    # wrist's approach axis moves the bolt sideways on purpose.)
    bolt_xy = _t(env.scene["object"].data.root_pos_w)[:, :2] - env.scene.env_origins[:, :2]
    wrist_xy, _ = _wrist_pose(env)
    rel_xy = bolt_xy - wrist_xy[:, :2]
    lift_entry = t.float() == g + LIFT_START
    b["lift_bolt_xy"] = torch.where(lift_entry.unsqueeze(1), bolt_xy, b["lift_bolt_xy"])
    b["lift_rel_xy"] = torch.where(lift_entry.unsqueeze(1), rel_xy, b["lift_rel_xy"])
    b["drag"] = torch.norm(rel_xy - b["lift_rel_xy"], dim=1)

    # -- wrist kinematics ---------------------------------------------------------
    v = wrist_lin_vel(env)
    a = (v - b["prev_v"]) / CONTROL_DT
    j = (a - b["prev_a"]) / CONTROL_DT
    valid = t > 2
    b["acc"] = torch.where(valid.unsqueeze(1), a, torch.zeros_like(a))
    b["jerk"] = torch.where(valid.unsqueeze(1), j, torch.zeros_like(j))
    b["jerk_max"] = torch.maximum(b["jerk_max"], torch.norm(b["jerk"], dim=1))
    b["prev_v"], b["prev_a"] = v.clone(), b["acc"].clone()

    # -- retry (level 22+): a failed lift or a drop in the hold reopens and restarts the grasp ---
    lvl_ = LEVELS[env.cfg.actions.pipeline.level]
    if lvl_.retry_max > 0:
        lift_end = t.float() == g + LIFT_START + LIFT_STEPS
        in_hold = t.float() > g + LIFT_START + LIFT_STEPS
        up_r = (z > SUCCESS_LIFT) if lvl_.sense_est_retry else up       # z is the estimate's at sense_est levels
        need = ((lift_end & ~up_r) | (in_hold & b["was_up"] & (z < UP_THRESHOLD))) & (b["attempts"] < lvl_.retry_max) & ~b["invalid"]
        if lvl_.grip_check:
            # v3 (level 31+): the test lift is 24 mm commanded over the pre-lift (its last GRIP_CHECK_STEPS at
            # free-pre-lift levels), measured at the FINGERTIPS; a latched rod that rose less than GRIP_CHECK_RATIO
            # of what the tips rose is not in the hand. Tips that rose under GRIP_CHECK_MIN_TIPS: inconclusive, no retry.
            # (v1/v2, level 26, measured 8 mm at the wrist and fired on 40-85 % of good grasps.)
            tz = torch.stack(_tip_positions(env), dim=1)[:, :, 2].mean(dim=1)
            bz = _t(env.scene["object"].data.root_pos_w)[:, 2] - env.scene.env_origins[:, 2]
            if lvl_.sense_est:
                bz = _est_bolt(env)[0][:, 2]
            start = t.float() == g + SETTLE_STEPS + (DRAG_STEPS if lvl_.free_prelift else 0)
            if lvl_.grip_check_late:
                # v4 (level 34+): measure from the middle of the test lift, after the finger drives and the contact
                # offsets have taken up their slack (v3 fired on 12 % of lying grasps that would have lifted)
                n_test = GRIP_CHECK_STEPS if lvl_.free_prelift else PRE_LIFT_STEPS
                start = t.float() == g + LIFT_START - n_test // 2
            b["_pre_bolt_z"] = torch.where(start, bz, b["_pre_bolt_z"])
            b["_pre_wrist_z"] = torch.where(start, tz, b["_pre_wrist_z"])
            tip_rise = tz - b["_pre_wrist_z"]
            check = ((t.float() == g + LIFT_START) & b["latched"] & (tip_rise > GRIP_CHECK_MIN_TIPS)
                     & ((bz - b["_pre_bolt_z"]) < GRIP_CHECK_RATIO * tip_rise))
            check = check & (b["attempts"] < lvl_.retry_max) & ~b["invalid"]
            b["grip_fired"] |= check
            need = need | check
        if bool(need.any()):
            b["attempts"] = torch.where(need, b["attempts"] + 1, b["attempts"])
            b["_retry_step"] = torch.where(need, t.float(), b["_retry_step"])
            b["grasp_end"] = torch.where(need, t.float() + GRASP_RETRY_STEPS, b["grasp_end"])
            g = b["grasp_end"]
            b["latched"] = b["latched"] & ~need
            b["latch_step"] = torch.where(need, torch.full_like(b["latch_step"], -1.0), b["latch_step"])
            b["hold_streak"] = torch.where(need, torch.zeros_like(b["hold_streak"]), b["hold_streak"])
            b["was_up"] = b["was_up"] & ~need
            b["verify_clean"] = b["verify_clean"] | need
            b["_retry_now"] = need
    # -- the final step of the episode ------------------------------------------
    final = t.float() >= g + SCRIPTED_STEPS
    b["final"] = final
    if LEVELS[env.cfg.actions.pipeline.level].pile.active:
        nb_z = torch.stack([_t(env.scene[f"neighbor_{k}"].data.root_pos_w)[:, 2] - env.scene.env_origins[:, 2]
                            for k in range(len(NEIGHBOR_SLOTS)) if f"neighbor_{k}" in env.scene.keys()], dim=1)
        s_in = LEVELS[env.cfg.actions.pipeline.level].pile.settle_in
        if s_in > 0:
            # the heap: the rest heights are what the rods settled to, recorded when the policy is released
            landed = (t.float() == s_in).unsqueeze(1)
            b["_nb_rest_z"][:, :nb_z.shape[1]] = torch.where(landed, nb_z, b["_nb_rest_z"][:, :nb_z.shape[1]])
            if LEVELS[env.cfg.actions.pipeline.level].fill.active:
                # a heap rod settles a few mm under the arm: the lift is measured from where it came to rest
                oz = _t(env.scene["object"].data.root_pos_w)[:, 2] - env.scene.env_origins[:, 2]
                b["_rest_z"] = torch.where(landed[:, 0] & b["_fill"], oz, b["_rest_z"])
        b["double"] = ((nb_z - b["_nb_rest_z"][:, :nb_z.shape[1]]) > DOUBLE_PICK_RISE).any(dim=1)
    b["success"] = (final & (b["hold_streak"] >= HOLD_STEPS) & b["verify_clean"]
                    & ~b["invalid"] & ~b["excess"] & ~b["double"])
    if LEVELS[env.cfg.actions.pipeline.level].grasp_point_tol > 0.0 or LEVELS[env.cfg.actions.pipeline.level].pinch_centre_tol > 0.0:
        # Binding point / centred pinch: a grasp that never latched (the fingers closed off the point
        # and the grasp phase timed out) can still be lifted by the scripted phases; it is not a success.
        b["success"] &= b["latched"]
    # An invalid collision ends the episode NOW, so its outcome is recorded now too.
    ended = final | fresh
    if bool(ended.any()):
        ids = ended.nonzero(as_tuple=False).squeeze(-1)
        b["last_success"][ids] = b["success"][ids]
        b["last_held"][ids] = b["hold_streak"][ids] >= HOLD_STEPS
        b["last_invalid"][ids] = b["invalid"][ids]
        b["last_invalid_which"][ids] = b["invalid_which"][ids]
        b["last_excess"][ids] = b["excess"][ids]
        b["last_drop"][ids] = b["paid_drop"][ids] | (b["was_up"][ids] & (z[ids] < UP_THRESHOLD))
        b["last_was_up"][ids] = b["was_up"][ids]
        b["last_verify_clean"][ids] = b["verify_clean"][ids]
        b["last_latched"][ids] = b["latched"][ids]
        b["last_attempts"][ids] = b["attempts"][ids]
        b["last_double"][ids] = b["double"][ids]
        b["last_retry_success"][ids] = (b["attempts"][ids] > 0) & b["success"][ids]
        b["last_heap_swapped"][ids] = b["_heap_swapped"][ids]
        b["last_grip_fired"][ids] = b["grip_fired"][ids]
        b["last_grasp_end"][ids] = g[ids]
        b["last_drag"][ids] = torch.where(b["latched"][ids] | final[ids], b["drag"][ids], torch.zeros_like(b["drag"][ids]))
        b["last_latch_s_err"][ids] = torch.where(b["latched"][ids], b["latch_s_err"][ids], torch.zeros_like(b["latch_s_err"][ids]))
        b["last_latch_d_err"][ids] = torch.where(b["latched"][ids], b["latch_d_err"][ids], torch.zeros_like(b["latch_d_err"][ids]))
        b["last_min_clear"][ids] = b["min_clear"][ids]
        b["last_min_obs_clear"][ids] = b["min_obs_clear"][ids]
        b["last_shield_steps"][ids] = b["shield_steps"][ids]
        b["last_jerk_max"][ids] = b["jerk_max"][ids]
        gs = g[ids].clamp(min=1.0)
        b["last_thumb_frac"][ids] = b["thumb_steps"][ids] / gs
        b["last_finger_frac"][ids] = b["finger_steps"][ids] / gs
        term = env.action_manager._terms.get("pipeline")
        if term is not None:
            b["last_closure"][ids] = term.closure[ids]
    return b


def grasp_ready(env) -> torch.Tensor:
    """The latch condition (docs/EXPERIMENT4_DESIGN.md §4). At hw_latch levels the same tests read what
    the robot can measure: finger joint torque instead of fingertip force, the estimated rod instead of
    the true one, and the estimate's own motion for the settled test."""
    b = _update(env)
    lvl_ = LEVELS[env.cfg.actions.pipeline.level]
    force_ok = b["tip_force"].max(dim=1).values < F_MAX                 # kept at every level: a training constraint
    if lvl_.hw_latch:
        ts = b["tau_streak"] >= STABLE_STEPS
        stable_opposing = ts[:, 0] & (ts[:, 1] | ts[:, 2])
        tau = _grip_torque(env)
        gripping = (tau[:, 0] > GRIP_TAU_MIN) & (torch.maximum(tau[:, 1], tau[:, 2]) > GRIP_TAU_MIN)
        d = fingertip_distances_est(env)
    else:
        stable_opposing = b["thumb_stable"] & b["finger_stable"]
        gripping = ((b["tip_force"][:, 0] > GRIP_MIN_N)
                    & (torch.maximum(b["tip_force"][:, 1], b["tip_force"][:, 2]) > GRIP_MIN_N))
        d = fingertip_distances_true(env)
    inside = (0.5 * (d[:, 0] + torch.minimum(d[:, 1], d[:, 2]))) < GRASP_REGION
    # where along the rod the pinch is, against the commanded point; binding when the level says so
    b["grasp_s_err"] = (pinch_along_axis(env) - b["_grasp_s"]).abs()
    tol = LEVELS[env.cfg.actions.pipeline.level].grasp_point_tol
    if tol > 0.0:
        inside = inside & (b["grasp_s_err"] <= tol)
    # and how deep: the pinch midpoint near the axis, binding when the level says so (level 28+)
    b["grasp_d_err"] = pinch_centre_dist_est(env) if lvl_.hw_latch else pinch_centre_dist(env)
    dtol = lvl_.pinch_centre_tol
    if dtol > 0.0:
        inside = inside & (b["grasp_d_err"] <= dtol)
    if lvl_.hw_latch:
        ep_, _ = _est_bolt(env)
        est_v = (ep_[:, :3] - b["_est_prev"]) * CONTROL_HZ               # the estimate's own velocity (finite difference)
        b["_est_prev"] = ep_[:, :3].clone()
        w = lvl_.est_vel_window
        if w > 1:
            # over w steps: _est_win holds the estimates of the previous EST_WIN_MAX steps, newest first
            est_v = (ep_[:, :3] - b["_est_win"][:, w - 1]) * (CONTROL_HZ / w)
        b["_est_win"] = torch.cat([ep_[:, :3].unsqueeze(1), b["_est_win"][:, :-1]], dim=1)
        rel = est_v - wrist_lin_vel(env)
    else:
        rel = (_t(env.scene["object"].data.root_lin_vel_w) - wrist_lin_vel(env))
    settled = torch.norm(rel, dim=1) < REL_VEL_MAX
    return stable_opposing & gripping & force_ok & inside & settled & ~b["invalid"]


def update_latch(env) -> torch.Tensor:
    """Advance the latch once per step. Returns (N,) bool: envs that latched THIS step."""
    b = _rg(env)
    if b.get("_latch_step") == int(env.common_step_counter):
        return b["_latched_now"]
    b["_latch_step"] = int(env.common_step_counter)
    t = env.episode_length_buf.float()
    # never on the reset step (stale contacts from the previous episode) nor during the heap's settle-in (hand frozen open)
    t_min = max(1, LEVELS[env.cfg.actions.pipeline.level].pile.settle_in)
    now = grasp_ready(env) & ~b["latched"] & (t < b["grasp_end"]) & (t >= t_min)
    b["grasp_end"] = torch.where(now, t, b["grasp_end"])
    b["latch_step"] = torch.where(now, t, b["latch_step"])
    b["latch_s_err"] = torch.where(now, b["grasp_s_err"], b["latch_s_err"])
    b["latch_d_err"] = torch.where(now, b["grasp_d_err"], b["latch_d_err"])
    b["latched"] |= now
    b["_latched_now"] = now
    return now


# ---------------------------------------------------------------------------
# Perception: ONE estimated bolt pose that every geometric bolt observation uses
# ---------------------------------------------------------------------------


def _est_bolt(env) -> tuple[torch.Tensor, torch.Tensor]:
    """The policy's bolt pose estimate, (N, 3) env-relative position and (N, 4) quat.

    Exact at levels without perception corruption. Otherwise
    T_hat = T_true (+) eps with a per-episode constant bias, per-frame noise,
    latency (a 3-deep history) and dropout (hold the last valid estimate).
    Computed once per step; rewards, the latch and the success test never read it.
    """
    b = _rg(env)
    if b.get("_est_step") == int(env.common_step_counter):
        if b.get("_fresh_est") is not None:
            _rerun_fresh(env, b, "_fresh_est", _est_bolt_body)
        return b["est_pos"], b["est_quat"]
    b["_est_step"] = int(env.common_step_counter)
    b.pop("_fresh_est", None)
    return _est_bolt_body(env)


def _est_bolt_body(env) -> tuple[torch.Tensor, torch.Tensor]:
    b = _rg(env)
    pos, quat = _bolt_true(env)
    # Before the first physics step the simulator's pose buffers are zero, and
    # a zero quaternion turns into NaN in matrix_from_quat. Substitute the
    # bolt's spawn orientation for any quaternion that is not unit-length.
    bad = (torch.norm(quat, dim=1) < 0.5).unsqueeze(1)
    if bool(bad.any()):
        spawn = torch.tensor(_ids(env)["bolt_spawn_quat"], device=env.device).expand(env.num_envs, 4)
        quat = torch.where(bad, spawn, quat)
    perc = LEVELS[env.cfg.actions.pipeline.level].perception
    if not perc.active:
        b["est_pos"], b["est_quat"] = _to_grasp_point(env, pos, quat)
        return b["est_pos"], b["est_quat"]
    n, d = env.num_envs, env.device
    # push the true pose into the history (index 0 = newest)
    b["hist_pos"] = torch.cat([pos.unsqueeze(1), b["hist_pos"][:, :HIST_DEPTH - 1]], dim=1)
    b["hist_quat"] = torch.cat([quat.unsqueeze(1), b["hist_quat"][:, :HIST_DEPTH - 1]], dim=1)
    b["hist_n"] = (b["hist_n"] + 1).clamp(max=HIST_DEPTH)
    lag = torch.minimum(b["latency"], (b["hist_n"] - 1).clamp(min=0).long())
    ar = torch.arange(n, device=d)
    p = b["hist_pos"][ar, lag]
    q = b["hist_quat"][ar, lag]
    # constant bias
    p = p + b["bias_pos"]
    ang = torch.norm(b["bias_rotvec"], dim=1)
    axis = b["bias_rotvec"] / ang.unsqueeze(1).clamp(min=1e-9)
    q = math_utils.quat_mul(math_utils.quat_from_angle_axis(ang, axis), q)
    # per-frame noise
    p = p + torch.randn(n, 3, device=d) * b["noise_std"].unsqueeze(1)
    # dropout: hold the previous estimate
    if perc.dropout > 0:
        drop = (torch.rand(n, device=d) < perc.dropout) & (b["hist_n"] > 1)
        p = torch.where(drop.unsqueeze(1), b["est_pos"], p)
        q = torch.where(drop.unsqueeze(1), b["est_quat"], q)
    p, q = _to_grasp_point(env, p, q)
    b["est_pos"], b["est_quat"] = p, q
    return p, q


def _to_grasp_point(env, pos, quat) -> tuple[torch.Tensor, torch.Tensor]:
    """The commanded grasp: the (estimated) bolt centre moved s along the
    (estimated) axis, and the orientation yawed by the pinch angle delta. The
    policy sees this virtual bolt as 'the bolt', aligns to it, and so grasps
    the real rod at the grasp point, obliquely by delta."""
    lvl = LEVELS[env.cfg.actions.pipeline.level]
    if lvl.free_grasp:
        if lvl.cover_shift:
            # level 48+: a covered rod's estimate is moved to the middle of its EXPOSED part (the block covers
            # s >= COVER_EDGE_S on the _cover_side side), so the pinch the policy aims at clears the block
            b = _rg(env)
            side = b["_cover_side"].float()
            axis = math_utils.quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))
            s_ = side * 0.5 * (COVER_EDGE_S - BOLT_HALF_LEN)
            return pos + s_.unsqueeze(1) * axis, quat
        return pos, quat                       # no story: the policy sees the real rod
    b = _rg(env)
    axis = math_utils.quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))
    p = pos + b["_grasp_s"].unsqueeze(1) * axis
    q = math_utils.quat_mul(_zquat(b["_pinch_delta"]), quat)
    return p, q


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def bolt_pos_in_wrist(env) -> torch.Tensor:
    pos, quat = _wrist_pose(env)
    bolt, _ = _est_bolt(env)
    return math_utils.quat_apply_inverse(quat, bolt - pos)


def bolt_rot_in_wrist(env) -> torch.Tensor:
    _, quat = _wrist_pose(env)
    _, bolt_q = _est_bolt(env)
    rel = math_utils.quat_mul(math_utils.quat_inv(quat), bolt_q)
    R = math_utils.matrix_from_quat(rel)
    return torch.cat([R[:, :, 0], R[:, :, 1]], dim=1)


def fingertip_distances_est(env) -> torch.Tensor:
    return _fingertip_distances(env, *_est_bolt(env))


def _grip_torque(env) -> torch.Tensor:
    """(N, 3) |applied joint torque| summed over each finger's flexion chain: thumb, index, middle. The
    simulator's actuator torque is the stand-in for the real hand's motor current."""
    robot = env.scene["robot"]
    term = env.action_manager._terms["pipeline"]
    tau = _t(robot.data.applied_torque).abs()
    th = tau[:, term._thumb_ids].sum(dim=1)
    fi = tau[:, term._finger_ids].view(env.num_envs, 2, -1).sum(dim=2)
    return torch.stack([th, fi[:, 0], fi[:, 1]], dim=1)


def pinch_centre_dist_est(env) -> torch.Tensor:
    """pinch_centre_dist against the ESTIMATED rod (what the robot can compute)."""
    thumb, index, middle = _tip_positions(env)
    mid = 0.5 * (thumb + 0.5 * (index + middle))
    bp, bq = _est_bolt(env)
    axis = math_utils.quat_apply(bq, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))
    v = mid - bp
    s_ = (v * axis).sum(dim=1, keepdim=True)
    return torch.norm(v - s_ * axis, dim=1)


def tip_forces_obs(env) -> torch.Tensor:
    lvl = LEVELS[env.cfg.actions.pipeline.level]
    if lvl.effort_obs:
        # the flexion-chain torque as a force: GRIP_TAU_MIN reads as GRIP_MIN_N, the threshold the force latch used
        return (_grip_torque(env) * (GRIP_MIN_N / GRIP_TAU_MIN) / F_MAX).clamp(-5.0, 5.0)
    f = (_update(env)["tip_force"] / F_MAX).clamp(-5.0, 5.0)
    return f if lvl.tactile else torch.zeros_like(f)


def contact_flags(env) -> torch.Tensor:
    lvl = LEVELS[env.cfg.actions.pipeline.level]
    if lvl.effort_obs:
        return (_grip_torque(env) > TAU_CONTACT).float()
    c = (_update(env)["streak"] > 0).float()
    return c if lvl.tactile else torch.zeros_like(c)


def episode_time(env) -> torch.Tensor:
    """Episode clock in [0, 1], counted from the end of the settle-in (the heap): the policy's time budget
    is the same as at the levels it was trained on."""
    lvl = LEVELS[env.cfg.actions.pipeline.level]
    t = (env.episode_length_buf.float() - lvl.pile.settle_in).clamp(min=0.0)
    return (t / episode_max_steps(lvl.retry_max)).unsqueeze(1)


def safety_obs(env) -> torch.Tensor:
    """16: nearest-obstacle vector per group in the WRIST frame (4 x 3) + minimum clearance per group (4)."""
    _update(env)
    dist, vec = _clearance(env)
    _, wq = _wrist_pose(env)
    G = len(GROUP_NAMES)
    vw = math_utils.quat_apply_inverse(wq.unsqueeze(1).expand(-1, G, 4).reshape(-1, 4),
                                       vec.reshape(-1, 3)).view(env.num_envs, G * 3)
    return torch.cat([vw.clamp(-1.0, 1.0), dist.clamp(-0.1, 0.5)], dim=1)


HEAP_MAP_DIRS = 8                                   # sectors around the rod's axis
HEAP_MAP_RANGE = 0.045                              # m: a sector this free reads 1.0
_HEAP_MAP_PROBES = (0.005, 0.015, 0.025, 0.035, 0.045)   # m beyond the rod's surface
_HEAP_MAP_AXIAL = (-0.6, 0.0, 0.6)                  # fractions of the half-length


def heap_map_obs(env) -> torch.Tensor:
    """HEAP_MAP_DIRS: free radial distance around the ESTIMATED rod per sector, in [0, 1].

    Sector 0 is the horizontal direction u = normalize(z x axis), sector k is 2*pi*k/8 from it towards
    v = axis x u, flipped so v points up: 2 is up, 4 the other side, 6 down (the table). A sector's value
    is the largest probe distance beyond the rod's surface at which no neighbour rod (true capsules: a
    depth camera sees them), bed rod, block, wall or the table is hit, taken over three points along the
    axis, divided by HEAP_MAP_RANGE. Two opposite free sectors are a pinch site.
    """
    b = _rg(env)
    c_ = _ids(env)
    n, d = env.num_envs, env.device
    _refresh_neighbor_capsules(env)
    pos, quat = _est_bolt(env)
    ez = torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, 3)
    ax = math_utils.quat_apply(quat, ez)
    u = torch.cross(ez, ax, dim=1)
    small = torch.norm(u, dim=1, keepdim=True) < 0.2                            # a standing rod: any horizontal u
    u = torch.where(small, torch.tensor([1.0, 0.0, 0.0], device=d).expand(n, 3), u)
    u = u / torch.norm(u, dim=1, keepdim=True).clamp(min=1e-9)
    v = torch.cross(ax, u, dim=1)
    v = torch.where((v[:, 2:3] < 0.0), -v, v)
    th = torch.arange(HEAP_MAP_DIRS, device=d) * (2 * math.pi / HEAP_MAP_DIRS)
    dirs = torch.cos(th).view(1, -1, 1) * u.unsqueeze(1) + torch.sin(th).view(1, -1, 1) * v.unsqueeze(1)   # (n, D, 3)
    A = torch.tensor(_HEAP_MAP_AXIAL, device=d) * BOLT_HALF_LEN
    P = torch.tensor(_HEAP_MAP_PROBES, device=d) + BOLT_RADIUS
    pts = (pos.view(n, 1, 1, 1, 3) + A.view(1, -1, 1, 1, 1) * ax.view(n, 1, 1, 1, 3)
           + P.view(1, 1, 1, -1, 1) * dirs.view(n, 1, HEAP_MAP_DIRS, 1, 3))         # (n, A, D, P, 3)
    flat = pts.reshape(n, -1, 3)                                                 # (n, M, 3)
    # capsules: bed rods and neighbours (parked ones are far away, radius 0 where absent)
    p0, p1, cr = b["_cap_p0"], b["_cap_p1"], b["_cap_r"]
    seg = p1 - p0
    w = flat.unsqueeze(2) - p0.unsqueeze(1)                                       # (n, M, C, 3)
    tt = ((w * seg.unsqueeze(1)).sum(-1) / (seg * seg).sum(-1).clamp(min=1e-9).unsqueeze(1)).clamp(0.0, 1.0)
    closest = p0.unsqueeze(1) + tt.unsqueeze(-1) * seg.unsqueeze(1)
    hit = (torch.norm(flat.unsqueeze(2) - closest, dim=-1) < cr.unsqueeze(1) + 0.002).any(dim=2)   # (n, M)
    # boxes (blocks, walls) and the table top
    lo, hi = b["_obs_lo"], b["_obs_hi"]                                           # (n, B, 3)
    inside = ((flat.unsqueeze(2) > lo.unsqueeze(1) - 0.002) & (flat.unsqueeze(2) < hi.unsqueeze(1) + 0.002)).all(dim=-1).any(dim=2)
    hit = hit | inside | (flat[:, :, 2] < c_["table_hi"][2] + 0.002)
    hit = hit.view(n, len(_HEAP_MAP_AXIAL), HEAP_MAP_DIRS, len(_HEAP_MAP_PROBES))
    blocked = hit.cumsum(dim=3) > 0                                              # a probe behind a hit is blocked too
    free = (~blocked).float().sum(dim=3) * (_HEAP_MAP_PROBES[1] - _HEAP_MAP_PROBES[0])    # m, per axial point and sector
    return (free.min(dim=1).values / HEAP_MAP_RANGE).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def _pinch_offset_hand(env) -> torch.Tensor:
    """The pinch centre in the WRIST frame: where the rod's centre sits relative
    to the wrist at the solved contact pose (constant of the hand)."""
    c = _ids(env)
    v = c.get("pinch_offset_hand")
    if v is None:
        world = torch.tensor([BOLT_OFFSET_XY[0], BOLT_OFFSET_XY[1], BOLT_REST_Z - WRIST_Z_CONTACT], device=env.device)
        v = math_utils.quat_apply_inverse(torch.tensor(WRIST_ROT, device=env.device).unsqueeze(0), world.unsqueeze(0))[0]
        c["pinch_offset_hand"] = v
    return v


def _reach_potential(env) -> torch.Tensor:
    b = _rg(env)
    pos, quat = _wrist_pose(env)
    if LEVELS[env.cfg.actions.pipeline.level].free_grasp:
        # Free grasp: the pinch centre to the rod's AXIS SEGMENT (any point along it
        # is as good), and only the tilt of the approach, never its yaw.
        n = env.num_envs
        pinch = pos + math_utils.quat_apply(quat, _pinch_offset_hand(env).expand(n, 3))
        bp, bq = _bolt_true(env)
        axis = math_utils.quat_apply(bq, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(n, 3))
        v = pinch - bp
        s = (v * axis).sum(dim=1, keepdim=True).clamp(-(BOLT_HALF_LEN - 0.004), BOLT_HALF_LEN - 0.004)
        d_w = torch.norm(v - s * axis, dim=1)
        rel = math_utils.quat_mul(quat, math_utils.quat_inv(torch.tensor(WRIST_ROT, device=env.device).expand(n, 4)))
        up = math_utils.quat_apply(rel, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(n, 3))
        th_w = torch.acos((up * b["_lift_dir"]).sum(dim=1).clamp(-1.0, 1.0))   # tilt from the approach axis (rolled with the rod at level 21+)
    else:
        d_w = torch.norm(pos - b["_nom_pos"], dim=1)
        q_err = math_utils.quat_mul(b["_nom_quat"], math_utils.quat_inv(quat))
        th_w = torch.norm(math_utils.axis_angle_from_quat(q_err), dim=1)
    return 0.5 * (1.0 - torch.tanh(d_w / REACH_SIGMA_POS)) + 0.5 * (1.0 - torch.tanh(th_w / REACH_SIGMA_ROT)), d_w


def _finger_potential(env) -> torch.Tensor:
    d = fingertip_distances_true(env)
    d_opp = torch.minimum(d[:, 1], d[:, 2])
    return 0.5 * ((1.0 - torch.tanh(d[:, 0] / FINGER_SIGMA)) + (1.0 - torch.tanh(d_opp / FINGER_SIGMA)))


def reach_progress(env) -> torch.Tensor:
    """REACH_WEIGHT (Phi_t - Phi_{t-1}) on the wrist pose potential, grasp phase only."""
    b = _update(env)
    phi, _ = _reach_potential(env)
    # No progress on the first post-reset step: body poses are stale at reset
    # time, so the previous potential cannot be seeded there without error.
    live = (phase_of(env) == 0) & (env.episode_length_buf > 1) & (env.episode_length_buf.float() != b["_retry_step"])
    r = torch.where(live, phi - b["prev_reach"], torch.zeros_like(phi))
    b["prev_reach"] = phi
    return r


def grasp_point_progress(env) -> torch.Tensor:
    """GRASP_POINT_WEIGHT (Phi_t - Phi_{t-1}) on the pinch-to-commanded-point potential, grasp phase,
    binding levels only (zero elsewhere, so levels 0-18 are unchanged)."""
    b = _update(env)
    lvl = LEVELS[env.cfg.actions.pipeline.level]
    if lvl.grasp_point_tol <= 0.0 and lvl.pinch_centre_tol <= 0.0:
        return torch.zeros(env.num_envs, device=env.device)
    if lvl.free_grasp:
        err = pinch_centre_dist(env)                   # free grasp: only the depth of the pinch, anywhere along the rod
    else:
        err = (pinch_along_axis(env) - b["_grasp_s"]).abs()
        if lvl.pinch_centre_tol > 0.0:
            # centred-pinch levels: the full distance of the pinch midpoint to the commanded point
            err = torch.sqrt(err * err + pinch_centre_dist(env) ** 2)
    phi = 1.0 - torch.tanh(err / GRASP_POINT_SIGMA)
    live = (phase_of(env) == 0) & (env.episode_length_buf > 1) & (env.episode_length_buf.float() != b["_retry_step"])
    r = torch.where(live, phi - b["prev_point"], torch.zeros_like(phi))
    b["prev_point"] = phi
    return r


def finger_progress(env) -> torch.Tensor:
    b = _update(env)
    phi = _finger_potential(env)
    live = (phase_of(env) == 0) & (env.episode_length_buf > 1) & (env.episode_length_buf.float() != b["_retry_step"])
    r = torch.where(live, phi - b["prev_fingers"], torch.zeros_like(phi))
    b["prev_fingers"] = phi
    return r


VEL_STEP_CAP = 0.5              # per-step cap on the velocity penalty


def near_object_velocity(env) -> torch.Tensor:
    """-g(d_w) [0.02 |dv / 0.10|^2 + 0.01 |dw / 0.5|^2], grasp phase only (returned NEGATIVE).

    Paid only while NO fingertip is touching the bolt. Measured at level 0
    before this gate: a wrist that never moved and simply closed its fingers
    collected -20 -- the whole success reward -- because the impact spins the
    bolt (and rattles the wrist) at several rad/s for a few steps, and the
    relative-velocity term read that as the hand rushing the object. The term
    exists to slow the APPROACH; once the fingers are on the bolt, its motion
    is grasp dynamics. Also capped per step so a single impulse cannot dominate.
    """
    b = _update(env)
    _, d_w = _reach_potential(env)
    g = 1.0 - torch.tanh(d_w / VEL_GATE_SIGMA)
    obj = env.scene["object"]
    dv = wrist_lin_vel(env) - _t(obj.data.root_lin_vel_w)
    dw = wrist_ang_vel(env) - _t(obj.data.root_ang_vel_w)
    pen = g * (VEL_LIN_COEF * (dv / VEL_LIN_SCALE).pow(2).sum(dim=1)
               + VEL_ANG_COEF * (dw / VEL_ANG_SCALE).pow(2).sum(dim=1))
    pen = pen.clamp(max=VEL_STEP_CAP)
    untouched = (b["streak"] == 0).all(dim=1)
    live = (phase_of(env) == 0) & untouched
    return torch.where(live, -pen, torch.zeros_like(pen))


def thumb_contact_bonus(env):
    b = _update(env)
    return _once(b, "paid_thumb", b["thumb_stable"])


def finger_contact_bonus(env):
    b = _update(env)
    return _once(b, "paid_finger", b["finger_stable"])


def opposing_contact_bonus(env):
    b = _update(env)
    return _once(b, "paid_opposing", b["thumb_stable"] & b["finger_stable"])


def success_bonus(env):
    return _update(env)["success"].float()


def drop_penalty(env):
    b = _update(env)
    z = _t(env.scene["object"].data.root_pos_w)[:, 2] - b["_rest_z"]
    if LEVELS[env.cfg.actions.pipeline.level].sense_est:
        z = _est_bolt(env)[0][:, 2] - b["_rest_est"]
    dropped = b["was_up"] & (z < UP_THRESHOLD) & (env.episode_length_buf.float() > b["grasp_end"])
    return _once(b, "paid_drop", dropped)


def excess_force_penalty(env):
    b = _update(env)
    return _once(b, "paid_excess", b["excess"])


def invalid_collision_penalty(env):
    b = _update(env)
    return _once(b, "paid_invalid", b["invalid"])


def clearance_penalty(env) -> torch.Tensor:
    """sum_g max(0, (m_g - d_g) / m_g)^2  (returned POSITIVE; the weight is negative)."""
    _update(env)
    dist, _ = _clearance(env)
    if LEVELS[env.cfg.actions.pipeline.level].push_ok:
        dist = _rg(env)["clear_hard"]          # neighbour rods may be pushed: only the table and hard obstacles cost
    m = _ids(env)["margins"].unsqueeze(0)
    return (((m - dist) / m).clamp(min=0.0) ** 2).sum(dim=1)


def shield_penalty(env) -> torch.Tensor:
    """(1 - alpha)^2 (returned POSITIVE; the weight is negative). alpha = 1 until the shield lands."""
    return (1.0 - _rg(env)["alpha"]) ** 2


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def invalid_collision_termination(env) -> torch.Tensor:
    return _update(env)["invalid"]


def phase_complete(env) -> torch.Tensor:
    """The scripted hold has run its course. A TERMINATION, not a time-out, so
    rsl-rl does not bootstrap a value estimate at a state that is truly final."""
    return _update(env)["final"]


def bolt_off_table(env) -> torch.Tensor:
    return _t(env.scene["object"].data.root_pos_w)[:, 2] < TABLE_TOP - 0.075


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def reset_rg_buffers(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    b = _rg(env)
    for k, v in b.items():
        if not torch.is_tensor(v) or k.startswith("last_") or k.startswith("_") or k in ("est_pos", "est_quat"):
            continue
        v[env_ids] = 0
    # The post-reset observations are computed in the same step: the reset envs' once-per-step values are
    # recomputed for the new episode (_rerun_fresh), the other envs' left as this step made them.
    if LEGACY_STEP:
        for k in ("_est_step", "_clear_step", "_step", "_latch_step"):
            b.pop(k, None)
    else:
        mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        mask[env_ids] = True
        for k in ("_fresh_upd", "_fresh_est"):
            b[k] = mask if b.get(k) is None else (b[k] | mask)
        b.pop("_clear_step", None)          # _clearance only takes minima: re-running it is harmless
    b["verify_clean"][env_ids] = True
    b["grasp_end"][env_ids] = float(GRASP_MAX_STEPS + LEVELS[env.cfg.actions.pipeline.level].pile.settle_in)   # the settle-in is not grasp time
    b["latch_step"][env_ids] = -1.0
    b["_retry_step"][env_ids] = -1.0
    b["_heap_swapped"][env_ids] = False
    b["clear"][env_ids] = 1.0
    b["min_clear"][env_ids] = 1.0
    b["obs_clear"][env_ids] = 1.0
    b["min_obs_clear"][env_ids] = 1.0
    b["alpha"][env_ids] = 1.0
    # Perception draws for the new episode.
    perc = LEVELS[env.cfg.actions.pipeline.level].perception
    n = len(env_ids)
    d = env.device
    if perc.active:
        u = lambda lo, hi, k=1: torch.rand((n, k) if k > 1 else (n,), device=d) * (hi - lo) + lo  # noqa: E731
        b["bias_pos"][env_ids] = torch.stack([u(-perc.bias_xy, perc.bias_xy), u(-perc.bias_xy, perc.bias_xy),
                                              u(-perc.bias_z, perc.bias_z)], dim=1)
        axis = torch.randn(n, 3, device=d)
        axis = axis / torch.norm(axis, dim=1, keepdim=True).clamp(min=1e-9)
        b["bias_rotvec"][env_ids] = axis * u(-perc.bias_rot, perc.bias_rot).unsqueeze(1)
        b["noise_std"][env_ids] = u(perc.noise_lo, perc.noise_hi)
        b["latency"][env_ids] = torch.randint(perc.latency_lo, perc.latency_hi + 1, (n,), device=d)
    b["hist_n"][env_ids] = 0
    m = _metrics(env)
    for k in ("clipped", "ik_failed", "singular", "limit_terminations"):
        m[k][env_ids] = 0.0
    m["s_min"][env_ids] = 1.0
    m["arm_slack"][env_ids] = 1.0


BOLT_OFFSET_XY = (BOLT_XY[0] - WRIST_XY[0], BOLT_XY[1] - WRIST_XY[1])   # bolt - wrist, hand frame at yaw 0
POSE_BUCKETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_buckets.npz")
POSE_BUCKETS_WAIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_buckets_waist.npz")
POSE_BUCKETS_TILT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_buckets_tilt.npz")
BOLT_STAND_Z = TABLE_TOP + BOLT_HALF_LEN + 0.002        # centre height of a rod standing on its end (spawn)


def bolt_asset(collider: str) -> str:
    """The rod's USD by collider: 'hull' is the asset as shipped (a 64-vertex
    convex hull, whose end is not flat: standing, the rod tilts 4.5 deg at
    once and a third topple within 3 s); 'cd' is the same mesh with a
    32-hull convex decomposition (stands with 0.03 deg of tilt)."""
    base = LAYOUT["bolt_asset"]
    if collider == "hull":
        return base
    if collider == "cd":
        root, ext = os.path.splitext(base)
        return root + "_cd" + ext
    raise ValueError(f"unknown bolt collider {collider!r}")
_Z = None


def _zquat(psi: torch.Tensor) -> torch.Tensor:
    """Quaternion (xyzw) for a rotation of psi about world z."""
    z = torch.zeros_like(psi)
    return torch.stack([z, z, torch.sin(psi / 2), torch.cos(psi / 2)], dim=1)


def _nominal_wrist(dx, dy, psi, h, device):
    """Bolt-relative nominal wrist pose for a bolt shifted (dx, dy), yawed psi, at standoff h.

    The pinch is yaw-invariant, so the wrist target is the pipeline's wrist
    pose rotated about the bolt by psi and translated with it."""
    off = torch.tensor(BOLT_OFFSET_XY, device=device)
    c, s_ = torch.cos(psi), torch.sin(psi)
    rox = c * off[0] - s_ * off[1]
    roy = s_ * off[0] + c * off[1]
    pos = torch.stack([BOLT_XY[0] + dx - rox, BOLT_XY[1] + dy - roy, torch.full_like(dx, WRIST_Z_CONTACT) + h], dim=1)
    quat = math_utils.quat_mul(_zquat(psi), torch.tensor(WRIST_ROT, device=device).expand(psi.shape[0], 4))
    return pos, quat


def _nominal_wrist_general(g, psi, theta, nrm, h, device):
    """Wrist (pos, quat) that puts the pinch on the grasp point g (n, 3), for hand yaw psi and a wrist
    roll of -theta about the horizontal axis nrm (the rod's tilt), at standoff h along the approach
    axis. theta = 0 reduces to _nominal_wrist with the grasp point's height."""
    n = g.shape[0]
    off3 = torch.tensor([BOLT_OFFSET_XY[0], BOLT_OFFSET_XY[1], BOLT_REST_Z - WRIST_Z_CONTACT], device=device)
    qz = _zquat(psi)
    nn = nrm / torch.norm(nrm, dim=1, keepdim=True).clamp(min=1e-9)
    qt = math_utils.quat_from_angle_axis(-theta, nn)
    qw = math_utils.quat_mul(qt, qz)
    m = math_utils.quat_apply(qw, torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3))
    pos = g - math_utils.quat_apply(qw, off3.expand(n, 3)) + h.unsqueeze(1) * m
    quat = math_utils.quat_mul(qw, torch.tensor(WRIST_ROT, device=device).expand(n, 4))
    return pos, quat, m


def _quat_z_to(a: torch.Tensor) -> torch.Tensor:
    """xyzw quaternions rotating +z onto the unit vectors `a` (..., 3) with a_z >= 0 (shortest arc)."""
    ez = torch.zeros_like(a)
    ez[..., 2] = 1.0
    q = torch.cat([torch.cross(ez, a, dim=-1), (1.0 + a[..., 2:3])], dim=-1)
    return q / torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-9)


def _heaps(env, fill) -> dict:
    """The settled heaps (tools/make_bin_heaps.py) on the device, with each rod's axis (raised end up), its
    elevation and whether it is exposed: active and with no other rod resting on it (axes closer than a
    diameter plus 3 mm, the other's centre higher by half a radius) -- what a camera picking from the top sees."""
    cache = getattr(env, "_rg_heaps", None)
    if cache is not None:
        return cache
    import numpy as np
    d = env.device
    pos_l, quat_l, cnt_l, w_l = [], [], [], []
    for file, wt in (fill.files or ((fill.file, 1.0),)):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path}: run tools/make_bin_heaps.py (see its docstring for the counts)")
        z = np.load(path)
        assert z["pos"].shape[1] == FILL_RODS, f"{file}: {z['pos'].shape[1]} rods per heap, FILL_RODS is {FILL_RODS}"
        pos_l.append(z["pos"]); quat_l.append(z["quat"]); cnt_l.append(z["count"])
        w_l.append(np.full(len(z["count"]), wt / len(z["count"])))          # each FILE gets its weight, whatever its size
    pos = torch.tensor(np.concatenate(pos_l), device=d, dtype=torch.float)  # (H, R, 3), bin-centre relative, z above the table
    quat = torch.tensor(np.concatenate(quat_l), device=d, dtype=torch.float)
    cnt = torch.tensor(np.concatenate(cnt_l), device=d, dtype=torch.long)
    hw = torch.tensor(np.concatenate(w_l), device=d, dtype=torch.float)
    H, R = pos.shape[:2]
    ez = torch.tensor([0.0, 0.0, 1.0], device=d).expand(H * R, 3)
    ax = math_utils.quat_apply(quat.reshape(-1, 4), ez).view(H, R, 3)
    ax = torch.where(ax[..., 2:3] < 0, -ax, ax)
    active = torch.arange(R, device=d).unsqueeze(0) < cnt.unsqueeze(1)
    exposed = torch.zeros(H, R, dtype=torch.bool, device=d)
    for h0 in range(0, H, 128):
        p, a, act = pos[h0:h0 + 128], ax[h0:h0 + 128], active[h0:h0 + 128]
        h = p.shape[0]
        p0, p1 = p - BOLT_HALF_LEN * a, p + BOLT_HALF_LEN * a
        dist = _seg_seg_dist(p0.unsqueeze(2).expand(h, R, R, 3), p1.unsqueeze(2).expand(h, R, R, 3),
                             p0.unsqueeze(1).expand(h, R, R, 3), p1.unsqueeze(1).expand(h, R, R, 3))   # [i, j]
        on_me = (dist < 2 * BOLT_RADIUS + 0.003) & (p[:, None, :, 2] > p[:, :, None, 2] + 0.5 * BOLT_RADIUS) & act.unsqueeze(1)
        on_me &= ~torch.eye(R, dtype=torch.bool, device=d).unsqueeze(0)
        exposed[h0:h0 + 128] = act & ~on_me.any(dim=2)
    elev = torch.rad2deg(torch.asin(ax[..., 2].clamp(0.0, 1.0)))
    kind = torch.where(elev < FILL_LYING_DEG, 0, torch.where(elev <= FILL_TILT_MAX_DEG, 1, torch.where(elev >= FILL_UPRIGHT_DEG, 2, -1)))
    cache = dict(pos=pos, ax=ax, cnt=cnt, exposed=exposed, elev=torch.deg2rad(elev), kind=kind, H=H, w=hw)
    env._rg_heaps = cache
    return cache


def _fill_draw(env, env_ids, lvl, fill_fixed=None, kind_fixed=None) -> dict:
    """Full-bin episodes (level 50+): per env a heap, a mirror of it (the bin is symmetric in x and y) and an
    exposed rod of it that fits the reachable window, preferring the drawn kind (lying / leaning / standing).
    Returns the target's pose terms; the heap (env frame) goes to the buffers for reset_pile. Envs with no
    fitting rod in FILL_TRIES draws keep the level's other scenes."""
    b = _rg(env)
    hp = _heaps(env, lvl.fill)
    n, d, T, R = len(env_ids), env.device, FILL_TRIES, FILL_RODS
    p_on = lvl.fill.prob if fill_fixed is None else float(fill_fixed)
    on = torch.rand(n, device=d) < p_on
    w = torch.tensor(lvl.fill.kinds, device=d, dtype=torch.float)
    want = (torch.multinomial(w, n, replacement=True) if kind_fixed is None
            else torch.full((n,), int(kind_fixed), dtype=torch.long, device=d))
    h = torch.multinomial(hp["w"], n * T, replacement=True).view(n, T)
    sgn = torch.where(torch.rand(n, T, 2, device=d) < 0.5, -1.0, 1.0)
    mir = torch.cat([sgn, torch.ones(n, T, 1, device=d)], dim=2).unsqueeze(2)    # (n, T, 1, 3)
    P_ = hp["pos"][h] * mir                                                      # (n, T, R, 3)
    A = hp["ax"][h] * mir
    kind = hp["kind"][h]
    spawn_q = _t(env.scene["object"].data.default_root_pose)[env_ids[:1], 3:7].expand(n, 4)
    ez = torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, 3)
    a0 = math_utils.quat_apply(spawn_q, ez)
    yaw0 = torch.atan2(a0[:, 1], a0[:, 0]).view(n, 1, 1)
    wrap = lambda x, p: torch.remainder(x + p / 2, p) - p / 2  # noqa: E731
    psi_r = torch.atan2(A[..., 1], A[..., 0]) - yaw0                            # the rod's yaw in reset_bolt_pose's terms
    delta = (torch.rand(n, T, R, device=d) * 2 - 1) * lvl.pinch_angle
    psi_h = torch.where(kind == 0, wrap(psi_r + delta, math.pi), wrap(psi_r + delta, 2 * math.pi))
    psi_up = torch.rand(n, T, R, device=d) * (lvl.bolt_yaw_hi - lvl.bolt_yaw_lo) + lvl.bolt_yaw_lo
    psi_h = torch.where(kind == 2, psi_up, psi_h)
    delta = torch.where(kind == 2, torch.zeros_like(delta), delta)
    in_win = (psi_h >= lvl.bolt_yaw_lo) & (psi_h <= lvl.bolt_yaw_hi)
    lim = torch.full_like(psi_h, lvl.bolt_xy)
    if lvl.bolt_xy_edge > 0.0:
        core = (psi_h >= lvl.bolt_yaw_core[0]) & (psi_h <= lvl.bolt_yaw_core[1])
        lim = torch.where(core, lim, torch.full_like(lim, lvl.bolt_xy_edge))
    ok = hp["exposed"][h] & (kind >= 0) & in_win & (P_[..., 0].abs() <= lim) & (P_[..., 1].abs() <= lim)
    if lvl.fill.pick == "highest":
        # genesis_policy.selector.hand_yaw_for: reachable if ANY trained pinch angle puts the hand yaw in the window
        # (the core window with the core xy limit, else the whole window with the edge limit); the hand yaw is then
        # drawn uniformly over that overlap, so the robot's choice and the training's agree rod for rod
        per = torch.where(kind == 0, torch.full_like(psi_r, math.pi), torch.full_like(psi_r, 2 * math.pi))
        c0 = torch.remainder(psi_r + per / 2, per) - per / 2
        dxy = torch.maximum(P_[..., 0].abs(), P_[..., 1].abs())
        core = (lvl.bolt_yaw_core if lvl.bolt_xy_edge > 0.0 else (lvl.bolt_yaw_lo, lvl.bolt_yaw_hi))
        edge_lim = lvl.bolt_xy_edge if lvl.bolt_xy_edge > 0.0 else lvl.bolt_xy
        found = torch.zeros_like(ok)
        a_sel, b_sel, c_sel = torch.zeros_like(psi_r), torch.zeros_like(psi_r), torch.zeros_like(psi_r)
        for (wlo, whi), xy_lim in ((core, lvl.bolt_xy), ((lvl.bolt_yaw_lo, lvl.bolt_yaw_hi), edge_lim)):
            for k in range(-2, 3):
                c = c0 + k * per
                a = torch.clamp(c - lvl.pinch_angle, min=wlo)
                b_ = torch.clamp(c + lvl.pinch_angle, max=whi)
                hit = ~found & (a <= b_) & (dxy <= xy_lim) & (kind != 2)
                a_sel, b_sel, c_sel = torch.where(hit, a, a_sel), torch.where(hit, b_, b_sel), torch.where(hit, c, c_sel)
                found |= hit
            up = ~found & (kind == 2) & (dxy <= xy_lim)              # standing: any hand yaw in this window
            a_sel = torch.where(up, torch.full_like(a_sel, wlo), a_sel)
            b_sel = torch.where(up, torch.full_like(b_sel, whi), b_sel)
            found |= up
        psi_h = a_sel + torch.rand_like(a_sel) * (b_sel - a_sel)
        delta = torch.where(kind == 2, torch.zeros_like(psi_h), psi_h - c_sel)
        ok = hp["exposed"][h] & (kind >= 0) & found
    if lvl.fill.pick == "highest":
        # perception's rule: in the first (heap, mirror) draw with any candidate, its highest one (standing penalised)
        score = torch.where(ok, P_[..., 2] - FILL_STAND_PENALTY * (kind == 2).float(), torch.full_like(P_[..., 2], -1.0))
        any_t = ok.any(dim=2)                                                   # (n, T)
        on = on & any_t.any(dim=1)
        ti = any_t.float().argmax(dim=1)
        j = score[torch.arange(n, device=d), ti].argmax(dim=1)
    else:
        ok_want = ok & (kind == want.view(n, 1, 1))
        pick_from = torch.where(ok_want.flatten(1).any(dim=1, keepdim=True), ok_want.flatten(1), ok.flatten(1))
        on = on & pick_from.any(dim=1)
        flat = (torch.rand(n, T * R, device=d) * pick_from.float()).argmax(dim=1)
        ti, j = flat // R, flat % R
    ar = torch.arange(n, device=d)
    g = lambda x: x[ar, ti, j]  # noqa: E731
    cxy = torch.tensor([BOLT_XY[0], BOLT_XY[1], TABLE_TOP], device=d)
    # the whole heap, env frame, for reset_pile
    b["_fill"][env_ids] = on
    b["_fill_pos"][env_ids] = P_[ar, ti] + cxy
    b["_fill_ax"][env_ids] = A[ar, ti]
    b["_fill_cnt"][env_ids] = hp["cnt"][h[ar, ti]]
    b["_fill_j"][env_ids] = j
    return dict(on=on, kind=g(kind), psi_h=g(psi_h), delta=g(delta), pos=g(P_) + cxy, axis=g(A), elev=hp["elev"][h[ar, ti], j])


def reset_bolt_pose(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
                    grasp_offset_fixed: float | None = None, pinch_angle_fixed: float | None = None,
                    upright_fixed: bool | None = None, tilt_fixed: bool | None = None,
                    cover_fixed: bool | None = None, bed_fixed: bool | None = None,
                    fill_fixed: bool | None = None, fill_kind: int | None = None):
    """Sample the hand's yaw and the bolt's xy for the episode, the commanded
    pinch angle (rod yaw = hand yaw - delta) and grasp offset along the axis;
    place the bolt; set the nominal wrist pose (at the grasp point, at the
    hand's yaw) every later term reads. `grasp_offset_fixed` and
    `pinch_angle_fixed` pin those for every env (the evaluation sweeps)."""
    lvl = LEVELS[env.cfg.actions.pipeline.level]
    b = _rg(env)
    obj = env.scene[asset_cfg.name]
    n, d = len(env_ids), env.device
    u = lambda lo, hi: torch.rand(n, device=d) * (hi - lo) + lo  # noqa: E731
    psi_h = u(lvl.bolt_yaw_lo, lvl.bolt_yaw_hi)                    # the HAND's yaw: reachability is about the hand
    mode = torch.rand(n, device=d)
    if upright_fixed is not None:
        upright = torch.full((n,), bool(upright_fixed), dtype=torch.bool, device=d)
    else:
        upright = mode < lvl.upright_prob
    if tilt_fixed is not None:
        tilted = torch.full((n,), bool(tilt_fixed), dtype=torch.bool, device=d) & ~upright
    else:
        # the tilt band of the draw sits right after the standing band, so pinning one leaves the other's share
        tilted = ~upright & (mode >= lvl.upright_prob) & (mode < lvl.upright_prob + lvl.tilt.prob)
    if pinch_angle_fixed is not None:
        delta = torch.full_like(psi_h, float(pinch_angle_fixed))
    else:
        delta = u(-lvl.pinch_angle, lvl.pinch_angle)
    fill = _fill_draw(env, env_ids, lvl, fill_fixed, fill_kind) if lvl.fill.active else None
    f_on = fill["on"] if fill is not None else torch.zeros(n, dtype=torch.bool, device=d)
    if fill is not None:
        # a heap rod: its kind, the hand yaw and pinch angle it was found reachable with
        upright = torch.where(f_on, fill["kind"] == 2, upright)
        tilted = torch.where(f_on, fill["kind"] == 1, tilted)
        psi_h = torch.where(f_on, fill["psi_h"], psi_h)
        delta = torch.where(f_on, fill["delta"], delta)
    delta = torch.where(upright, torch.zeros_like(delta), delta)   # a standing rod has no yaw to be off from
    psi = psi_h - delta                                            # the rod's yaw
    lim = torch.full_like(psi, lvl.bolt_xy)
    if lvl.bolt_xy_edge > 0.0:
        core = (psi_h >= lvl.bolt_yaw_core[0]) & (psi_h <= lvl.bolt_yaw_core[1])
        lim = torch.where(core, lim, torch.full_like(psi, lvl.bolt_xy_edge))
    dx, dy = u(-1.0, 1.0) * lim, u(-1.0, 1.0) * lim
    if fill is not None:
        dx = torch.where(f_on, fill["pos"][:, 0] - BOLT_XY[0], dx)
        dy = torch.where(f_on, fill["pos"][:, 1] - BOLT_XY[1], dy)
    b["_bolt_sample"][env_ids] = torch.stack([dx, dy, psi], dim=1)
    b["_hand_yaw"][env_ids] = psi_h
    b["_pinch_delta"][env_ids] = delta
    b["_upright"][env_ids] = upright
    spawn = _t(obj.data.default_root_pose)[env_ids].clone()
    pos = spawn[:, 0:3] + torch.stack([dx, dy, torch.zeros_like(dx)], dim=1) + env.scene.env_origins[env_ids]
    quat = math_utils.quat_mul(_zquat(psi), spawn[:, 3:7])
    if bool(upright.any()):
        # standing: the mesh's own z (the rod's axis) stays vertical; only a yaw about it
        pos[:, 2] = torch.where(upright, BOLT_STAND_Z + env.scene.env_origins[env_ids, 2], pos[:, 2])
        quat = torch.where(upright.unsqueeze(1), _zquat(psi), quat)
    b["_tilted"][env_ids] = tilted
    b["_tilt"][env_ids] = torch.where(upright, torch.full_like(dx, math.pi / 2), torch.zeros_like(dx))
    b["_support_j"][env_ids] = -1
    b["_support_c"][env_ids] = torch.tensor(OBSTACLE_PARK, device=d)
    if bool(tilted.any()):
        # leaning: one end on the table, the rod over a support block's near edge SUPPORT_EDGE_DIST from
        # that end. Block height -> tilt by bisection on the underside height at the edge; the rod
        # CENTRE keeps the sampled xy, the block sits under the raised end. Physics settles the rest.
        j = torch.randint(0, len(SUPPORT_SIZES), (n,), device=d)
        h = torch.tensor([s[2] for s in SUPPORT_SIZES], device=d)[j]
        r = torch.full_like(h, 0.0083)                                 # thread radius, at the low end
        # The rod's underside is a line from the low-end contact point at slope tan(theta); it meets
        # the block's top edge, height h, at horizontal distance SUPPORT_EDGE_DIST: theta = atan(h / D).
        theta = torch.atan(h / SUPPORT_EDGE_DIST)                       # 18, 30, 40 deg for the three blocks
        ax0 = math_utils.quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, 3))   # the lying axis, horizontal
        nrm = torch.stack([-ax0[:, 1], ax0[:, 0], torch.zeros_like(ax0[:, 0])], dim=1)             # z x axis
        # rotate by -theta about n so the mesh's +z end is the RAISED one (the solver's convention)
        q_tilt = math_utils.quat_mul(math_utils.quat_from_angle_axis(-theta, nrm), quat)
        a = math_utils.quat_apply(q_tilt, torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, 3))
        assert bool((a[tilted, 2] >= -1e-4).all()), "tilt sign: the +z end must be the raised one"
        roll_axis = torch.where(tilted.unsqueeze(1), nrm, torch.zeros_like(nrm))
        cz = TABLE_TOP + BOLT_HALF_LEN * torch.sin(theta) + r * torch.cos(theta) + 0.002
        pos[:, 2] = torch.where(tilted, cz + env.scene.env_origins[env_ids, 2], pos[:, 2])
        quat = torch.where(tilted.unsqueeze(1), q_tilt, quat)
        uh = a[:, :2] / torch.norm(a[:, :2], dim=1, keepdim=True).clamp(min=1e-6)
        # low-end contact -> rod centre, horizontally: (L/2) cos(theta) - r sin(theta); block centre is D + half a block past the contact
        along = SUPPORT_EDGE_DIST + SUPPORT_SIZES[0][0] / 2 - (BOLT_HALF_LEN * torch.cos(theta) - r * torch.sin(theta))
        cxy = pos[:, :2] - env.scene.env_origins[env_ids, :2] + along.unsqueeze(1) * uh
        centre = torch.cat([cxy, (TABLE_TOP + h / 2).unsqueeze(1)], dim=1)
        b["_tilt"][env_ids] = torch.where(tilted, theta, b["_tilt"][env_ids])
        b["_support_j"][env_ids] = torch.where(tilted, j, b["_support_j"][env_ids])
        b["_support_c"][env_ids] = torch.where(tilted.unsqueeze(1), centre, b["_support_c"][env_ids])
    if fill is not None and bool(f_on.any()):
        # the heap rod's own pose: lying and leaning rods are the lying construction tilted by their elevation
        # about n = z x axis (the +z end raised, as for a block-supported rod), standing ones +z onto their axis
        # after the yaw; no support block (the heap holds it)
        q_lay = math_utils.quat_mul(_zquat(psi), spawn[:, 3:7])
        ax_l = math_utils.quat_apply(q_lay, torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, 3))
        nrm_f = torch.stack([-ax_l[:, 1], ax_l[:, 0], torch.zeros_like(ax_l[:, 0])], dim=1)
        # a lying rod's yaw is only known mod pi: when the construction's +z end points away from the heap rod's
        # raised end, lower it instead
        up_sign = torch.where((ax_l[:, :2] * fill["axis"][:, :2]).sum(dim=1) < 0, -1.0, 1.0)
        th = torch.where(tilted, fill["elev"], torch.where(upright, torch.zeros_like(dx), fill["elev"] * up_sign))
        q_f = math_utils.quat_mul(math_utils.quat_from_angle_axis(-th, nrm_f), q_lay)
        q_up = math_utils.quat_mul(_quat_z_to(fill["axis"]), _zquat(psi))
        q_f = torch.where(upright.unsqueeze(1), q_up, q_f)
        quat = torch.where(f_on.unsqueeze(1), q_f, quat)
        pos = torch.where(f_on.unsqueeze(1), fill["pos"] + env.scene.env_origins[env_ids], pos)
        b["_tilt"][env_ids] = torch.where(f_on, torch.where(tilted, fill["elev"], torch.where(upright, torch.full_like(dx, math.pi / 2),
                                                                                          torch.zeros_like(dx))), b["_tilt"][env_ids])
        b["_support_j"][env_ids] = torch.where(f_on, torch.full_like(b["_support_j"][env_ids], -1), b["_support_j"][env_ids])
        b["_support_c"][env_ids] = torch.where(f_on.unsqueeze(1), torch.tensor(OBSTACLE_PARK, device=d).expand(n, 3),
                                               b["_support_c"][env_ids])
        if lvl.wrist_roll and bool((f_on & tilted).any()):
            roll_axis = torch.where((f_on & tilted).unsqueeze(1), nrm_f, roll_axis)
    obj.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=1), env_ids=env_ids)
    obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=d), env_ids=env_ids)
    # the grasp point: s along the rod's axis (its z, laid on the table by the spawn quaternion, then yawed)
    # rod on rods: the target across two bed rods (lying episodes only, exclusive with the cover)
    lying_now = ~upright & ~tilted & ~f_on          # a heap rod lies on the heap: no bed, no cover
    if bed_fixed is not None:
        on_bed = torch.full((n,), bool(bed_fixed), dtype=torch.bool, device=d) & lying_now
    else:
        on_bed = lying_now & (torch.rand(n, device=d) < lvl.bed.prob)
    b["_on_bed"][env_ids] = on_bed
    if bool(on_bed.any()):
        ax_l = math_utils.quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, 3))   # target axis, horizontal
        perp = torch.stack([-ax_l[:, 1], ax_l[:, 0], torch.zeros_like(ax_l[:, 0])], dim=1)           # bed rods run this way
        # the target spawns above the bed rods' tops and settles onto them
        zt = TABLE_TOP + 2 * BED_RADIUS + BOLT_RADIUS + 0.003
        pos[:, 2] = torch.where(on_bed, zt + env.scene.env_origins[env_ids, 2], pos[:, 2])
        # the pose was already written above; write the raised one (this block runs after the first write)
        obj.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=1), env_ids=env_ids)
        cxy = pos[:, :2] - env.scene.env_origins[env_ids, :2]
        for k, sgn in enumerate((-1.0, 1.0)):
            bc = torch.cat([cxy + sgn * (BED_SPACING / 2) * ax_l[:, :2], torch.full((n, 1), TABLE_TOP + BED_RADIUS, device=d)], dim=1)
            b["_bed_c"][env_ids, k] = torch.where(on_bed.unsqueeze(1), bc, torch.tensor(OBSTACLE_PARK, device=d).expand(n, 3))
        b["_bed_axis"][env_ids] = torch.where(on_bed.unsqueeze(1), perp, torch.zeros_like(perp))
    else:
        b["_bed_c"][env_ids] = torch.tensor(OBSTACLE_PARK, device=d)
        b["_bed_axis"][env_ids] = 0.0
    # partial cover: one end of a lying rod under a block; the point is drawn on the exposed part
    lying_free = lying_now & ~on_bed
    if cover_fixed is not None:
        covered = torch.full((n,), bool(cover_fixed), dtype=torch.bool, device=d) & lying_free
    else:
        covered = lying_free & (torch.rand(n, device=d) < lvl.cover.prob)
    side = torch.where(torch.rand(n, device=d) < 0.5, torch.ones(n, dtype=torch.long, device=d), -torch.ones(n, dtype=torch.long, device=d))
    side = torch.where(covered, side, torch.zeros_like(side))
    b["_cover_side"][env_ids] = side
    if grasp_offset_fixed is not None:
        s_ = torch.full_like(dx, float(grasp_offset_fixed))
    else:
        s_ = u(-lvl.grasp_offset, lvl.grasp_offset)
        # covered: exposed part runs from -grasp_offset to +COVER_EDGE_S on the side opposite the block
        # measured towards the block (side = +1: the block covers s >= COVER_EDGE_S), then signed back
        s_exp = u(-lvl.grasp_offset, COVER_EDGE_S) * side.float()
        s_ = torch.where(covered, s_exp, s_)
    axis = math_utils.quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, 3))
    shift = s_.unsqueeze(1) * axis[:, :2]
    b["_grasp_s"][env_ids] = s_
    b["_grasp_shift"][env_ids] = shift
    cpos = pos - env.scene.env_origins[env_ids]
    cover_c = cpos + side.float().unsqueeze(1) * COVER_CENTRE_S * axis
    cover_c[:, 2] = BOLT_REST_Z + BOLT_RADIUS + COVER_GAP + COVER_SIZE[2] / 2
    b["_cover_c"][env_ids] = torch.where(covered.unsqueeze(1), cover_c, torch.tensor(OBSTACLE_PARK, device=d).expand(n, 3))
    ddir = -side.float().unsqueeze(1) * axis
    ddir[:, 2] = 0.0
    ddir = ddir / torch.norm(ddir, dim=1, keepdim=True).clamp(min=1e-6)
    b["_drag_dir"][env_ids] = torch.where(covered.unsqueeze(1), ddir, torch.zeros_like(ddir))
    # the grasp point's height: the rod's spawn centre plus the offset up its axis, above the lying rest
    # height (exactly 0 for a lying rod, so levels 0-17 are untouched)
    cz_env = pos[:, 2] - env.scene.env_origins[env_ids, 2]
    raised = upright | tilted | on_bed | f_on
    dz = torch.where(raised, cz_env + s_ * axis[:, 2] - BOLT_REST_Z, torch.zeros_like(s_))
    b["_nom_dz"][env_ids] = dz
    b["_rest_z"][env_ids] = torch.where(raised, cz_env, torch.full_like(cz_env, BOLT_REST_Z))
    if lvl.wrist_roll and bool(tilted.any()):
        # rolled wrist for tilted rods: the pinch on the grasp point, approach along the rod's up-normal
        g = torch.stack([BOLT_XY[0] + dx + shift[:, 0], BOLT_XY[1] + dy + shift[:, 1], BOLT_REST_Z + dz], dim=1)
        theta_all = torch.where(tilted, b["_tilt"][env_ids].clamp(max=lvl.roll_cap), torch.zeros_like(dx))
        nrm_all = torch.where(tilted.unsqueeze(1), roll_axis, torch.tensor([1.0, 0.0, 0.0], device=d).expand(n, 3))
        npos_r, nquat_r, m_r = _nominal_wrist_general(g, psi_h, theta_all, nrm_all, torch.zeros_like(dx), d)
        npos, nquat = _nominal_wrist(dx + shift[:, 0], dy + shift[:, 1], psi_h, dz, d)
        npos = torch.where(tilted.unsqueeze(1), npos_r, npos)
        nquat = torch.where(tilted.unsqueeze(1), nquat_r, nquat)
        b["_lift_dir"][env_ids] = torch.where(tilted.unsqueeze(1), m_r, torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, 3))
        b["_roll_axis"][env_ids] = roll_axis
        b["_roll"][env_ids] = theta_all
    else:
        npos, nquat = _nominal_wrist(dx + shift[:, 0], dy + shift[:, 1], psi_h, dz, d)
        b["_lift_dir"][env_ids] = torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, 3)
        b["_roll_axis"][env_ids] = 0.0
        b["_roll"][env_ids] = 0.0
    b["_nom_pos"][env_ids] = npos
    b["_nom_quat"][env_ids] = nquat


def _pose_buckets(env, robot) -> dict:
    """The bucket table: from pose_buckets.npz (bolt yaw x xy x height) when
    present, else the three fixed-bolt standoff buckets."""
    tabs = getattr(env, "_rg_buckets", None)
    if tabs is not None:
        return tabs
    d = env.device
    ids, _ = robot.find_joints(RIGHT_ARM, preserve_order=True)
    w_ids, w_names = robot.find_joints(list(WAIST_JOINTS), preserve_order=True)
    assert tuple(w_names) == WAIST_JOINTS, w_names
    lvl = LEVELS[env.cfg.actions.pipeline.level]
    file = POSE_BUCKETS_TILT_FILE if lvl.wrist_roll else POSE_BUCKETS_WAIST_FILE if lvl.free_waist else POSE_BUCKETS_FILE
    if (lvl.free_waist or lvl.wrist_roll) and not os.path.exists(file):
        raise FileNotFoundError(f"level {lvl.name} needs {file}; run tools/solve_reach_grasp_poses.py --grid "
                                "--free-waist [--tilts-deg 0 18.4 29.5 39.8]")
    if os.path.exists(file):
        import numpy as np
        z = np.load(file)
        ok = z["ok"].astype(bool)
        tilt_col = z["tilt"][ok] if "tilt" in z else np.zeros(int(ok.sum()))
        key = torch.tensor(np.stack([z["yaw"][ok], z["dx"][ok], z["dy"][ok], z["h"][ok], tilt_col], axis=1), device=d, dtype=torch.float)
        q = torch.tensor(z["q"][ok], device=d, dtype=torch.float)
        js = torch.tensor(z["j_sharp"][ok], device=d, dtype=torch.float)
        nominal = torch.tensor(WAIST_NOMINAL, device=d).expand(int(ok.sum()), 3)
        # (the pitch-only file stores a scalar `waist`; only the freed-waist file has a per-bucket table)
        waist = (torch.tensor(z["waist"][ok], device=d, dtype=torch.float)
                 if "waist" in z and z["waist"].ndim == 2 else nominal.clone())
    else:
        key = torch.tensor([[0.0, 0.0, 0.0, bk.height, 0.0] for bk in STANDOFF_BUCKETS], device=d)
        q = torch.tensor([[bk.arm[j] for j in RIGHT_ARM] for bk in STANDOFF_BUCKETS], device=d)
        js = torch.tensor([bk.j_sharp for bk in STANDOFF_BUCKETS], device=d)
        waist = torch.tensor(WAIST_NOMINAL, device=d).expand(key.shape[0], 3).clone()
    if not lvl.free_waist:
        waist = torch.tensor(WAIST_NOMINAL, device=d).expand(key.shape[0], 3).clone()   # levels 0-13: the pipeline's hold
    # each bucket's wrist pose, for the residual (rolled by the bucket's tilt about n = z x u(yaw))
    spawn_q = torch.tensor(tuple(env.cfg.scene.object.init_state.rot), device=d).expand(key.shape[0], 4)
    u_lying = math_utils.quat_apply(math_utils.quat_mul(_zquat(key[:, 0]), spawn_q), torch.tensor([0.0, 0.0, 1.0], device=d).expand(key.shape[0], 3))
    nrm_b = torch.stack([-u_lying[:, 1], u_lying[:, 0], torch.zeros_like(u_lying[:, 0])], dim=1)
    g_b = torch.stack([BOLT_XY[0] + key[:, 1], BOLT_XY[1] + key[:, 2], torch.full_like(key[:, 1], BOLT_REST_Z)], dim=1)
    bpos, bquat, _ = _nominal_wrist_general(g_b, key[:, 0], key[:, 4], torch.where((key[:, 4].abs() > 1e-9).unsqueeze(1), nrm_b, torch.tensor([1.0, 0.0, 0.0], device=d).expand(key.shape[0], 3)), key[:, 3], d)
    scale = torch.tensor([math.radians(15.0), 0.025, 0.025, 0.05, math.radians(10.0)], device=d)   # grid spacing, for the metric
    tabs = dict(ids=ids, w_ids=w_ids, key=key, q=q, js=js, waist=waist, bpos=bpos, bquat=bquat, scale=scale,
                n=key.shape[0])
    env._rg_buckets = tabs
    return tabs


def reset_wrist_standoff(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Start above the (bolt-relative) pinch pose by a sampled standoff, plus the level's pose error.

    Picks the nearest bucket in (bolt yaw, bolt xy, height) and applies the
    remaining pose difference through that bucket's Jacobian. Stores the
    displacement from the CONTACT pose on the env so the action term can seed
    its commanded pose analytically.
    """
    lvl = LEVELS[env.cfg.actions.pipeline.level]
    robot = env.scene[asset_cfg.name]
    b = _rg(env)
    n, d = len(env_ids), env.device

    store = getattr(env, "_rg_displacement", None)
    if store is None:
        store = torch.zeros(env.num_envs, 6, device=d)
        env._rg_displacement = store
    tabs = _pose_buckets(env, robot)

    def u(lo, hi):
        return torch.rand(n, device=d) * (hi - lo) + lo

    standoff = u(lvl.standoff_lo, lvl.standoff_hi)
    disp = torch.stack([u(-lvl.dxy, lvl.dxy), u(-lvl.dxy, lvl.dxy), standoff,
                        u(-lvl.drp, lvl.drp), u(-lvl.drp, lvl.drp), u(-lvl.dyaw, lvl.dyaw)], dim=1)
    store[env_ids] = disp

    # the wrist pose this episode starts at, in the world: the standoff runs along the approach axis
    # (vertical unless the wrist is rolled)
    m = b["_lift_dir"][env_ids]
    tgt_pos = b["_nom_pos"][env_ids] + torch.stack([disp[:, 0], disp[:, 1], torch.zeros_like(standoff)], dim=1) + standoff.unsqueeze(1) * m
    ang = torch.norm(disp[:, 3:6], dim=1)
    axis = disp[:, 3:6] / ang.unsqueeze(1).clamp(min=1e-9)
    tgt_quat = math_utils.quat_mul(math_utils.quat_from_angle_axis(ang, axis), b["_nom_quat"][env_ids])
    # nearest bucket in the (yaw, dx, dy, h) grid metric
    sample = b["_bolt_sample"][env_ids]
    gxy = sample[:, 0:2] + b["_grasp_shift"][env_ids]          # the wrist goes to the grasp point, not the bolt centre
    tilt_key = b["_roll"][env_ids]
    want = torch.stack([b["_hand_yaw"][env_ids], gxy[:, 0], gxy[:, 1], standoff + b["_nom_dz"][env_ids], tilt_key], dim=1)
    dist = (((want.unsqueeze(1) - tabs["key"].unsqueeze(0)) / tabs["scale"]) ** 2).sum(dim=2)
    bucket = dist.argmin(dim=1)
    # residual pose from the bucket's wrist pose to the target, world frame
    dpos = tgt_pos - tabs["bpos"][bucket]
    drot = math_utils.axis_angle_from_quat(math_utils.quat_mul(tgt_quat, math_utils.quat_inv(tabs["bquat"][bucket])))
    resid = torch.cat([dpos, drot], dim=1)
    q_arm = tabs["q"][bucket] + (tabs["js"][bucket] @ resid.unsqueeze(2)).squeeze(2)

    q = _t(robot.data.joint_pos)[env_ids].clone()
    q[:, tabs["ids"]] = q_arm
    # the bucket's held waist (the pipeline's hold unless the level frees it);
    # the action term holds it as the drive target for the rest of the episode
    q[:, tabs["w_ids"]] = tabs["waist"][bucket]
    b["_waist"][env_ids] = tabs["waist"][bucket]
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    # Hold the RESET configuration as the drive target until the first apply,
    # not the default (contact) pose, which is up to 12 cm away.
    robot.set_joint_position_target(q, env_ids=env_ids)
    robot.set_joint_velocity_target(torch.zeros_like(q), env_ids=env_ids)


# ---------------------------------------------------------------------------
# The action term
# ---------------------------------------------------------------------------


CORRIDOR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corridor.npz")
_CORRIDOR: dict | None = None


def _corridor():
    """The measured corridor (tools/probe_corridor.py), dilated for placement, or None.

    `free[i, j]` is True where an obstacle CENTRE may go: the swept footprint
    dilated by the largest obstacle's half-diagonal plus a 1 cm margin, so no
    part of any obstacle overlaps anything the nominal grasps sweep.
    """
    global _CORRIDOR
    if _CORRIDOR is None:
        if not os.path.exists(CORRIDOR_FILE):
            _CORRIDOR = {}
            return None
        import numpy as np
        z = np.load(CORRIDOR_FILE)
        occ = z["occ"].astype(bool)
        cell = float(z["cell"]); H = float(z["half_extent"])
        half_diag = max(math.hypot(sx, sy) / 2 for sx, sy, _ in OBSTACLE_SIZES)
        k = int(math.ceil((half_diag + 0.01) / cell))
        # dilate with a disc of radius k cells
        M = occ.shape[0]
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        disc = (xx ** 2 + yy ** 2) <= k * k
        pad = np.pad(occ, k)
        dil = np.zeros_like(occ)
        for di, dj in zip(*np.nonzero(disc)):
            dil |= pad[di:di + M, dj:dj + M]
        _CORRIDOR = dict(free=torch.tensor(~dil), cell=cell, H=H, M=M, occ_area=float(occ.sum()) * cell * cell,
                         free_area=float((~dil).sum()) * cell * cell)
    return _CORRIDOR or None


def _corridor_ok(xy: torch.Tensor, cl: ClutterCfg) -> torch.Tensor:
    """(M,) bool: bolt-relative xy positions where an obstacle centre may go.

    With the measured corridor present: inside the annulus and outside the
    dilated swept footprint. Without it: the strip heuristic (the hand and
    forearm lie behind the bolt, toward the wrist), which the measurement
    showed to be too permissive -- keep the file.
    """
    r = torch.norm(xy, dim=1)
    ok = (r >= cl.r_min) & (r <= cl.r_max)
    cor = _corridor()
    if cor is not None:
        free = cor["free"].to(xy.device)
        i = ((xy[:, 0] + cor["H"]) / cor["cell"]).long().clamp(0, cor["M"] - 1)
        j = ((xy[:, 1] + cor["H"]) / cor["cell"]).long().clamp(0, cor["M"] - 1)
        inside_grid = (xy.abs() < cor["H"]).all(dim=1)
        return ok & inside_grid & free[i, j]
    d = torch.tensor([BOLT_XY[0] - WRIST_XY[0], BOLT_XY[1] - WRIST_XY[1]], device=xy.device)
    d = d / torch.norm(d)
    along = xy @ d
    lateral = xy[:, 0] * (-d[1]) + xy[:, 1] * d[0]
    in_strip = (along < 0.0) & (lateral.abs() < cl.strip_half_width)
    return ok & ~in_strip


def reset_obstacles(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Place this episode's obstacles outside the corridor; park the unused slots.

    Rejection sampling in the annulus r_min..r_max around the bolt, excluding
    the strip behind it that the hand and forearm occupy, with a minimum
    separation between obstacles. Writes the kinematic bodies' poses and the
    analytic AABBs the clearance model reads.
    """
    cl = LEVELS[env.cfg.actions.pipeline.level].clutter
    b = _rg(env)
    n, d = len(env_ids), env.device
    sample = b["_bolt_sample"][env_ids]                                        # (n, 3): dx, dy, yaw
    bolt = torch.tensor(BOLT_XY, device=d) + sample[:, :2]                     # (n, 2), this episode's bolt
    psi = sample[:, 2]
    cpsi, spsi = torch.cos(psi), torch.sin(psi)
    rot = lambda v: torch.stack([cpsi * v[:, 0] - spsi * v[:, 1], spsi * v[:, 0] + cpsi * v[:, 1]], dim=1)  # noqa: E731
    unrot = lambda v: torch.stack([cpsi * v[:, 0] + spsi * v[:, 1], -spsi * v[:, 0] + cpsi * v[:, 1]], dim=1)  # noqa: E731
    M = len(OBSTACLE_SIZES)                                                    # clutter slots; the rest belong to other events
    sizes = torch.tensor(OBSTACLE_SIZES + ((0.01, 0.01, 0.01),) * (MAX_OBSTACLES - M), device=d)   # (MAX, 3)
    park = torch.tensor(OBSTACLE_PARK, device=d)
    count = torch.randint(cl.count_lo, cl.count_hi + 1, (n,), device=d) if cl.active else torch.zeros(n, dtype=torch.long, device=d)
    centres = park.view(1, 1, 3).expand(n, MAX_OBSTACLES, 3).clone()
    active = torch.arange(MAX_OBSTACLES, device=d).view(1, -1) < count.view(-1, 1)
    if cl.active:
        if cl.fixed_xy:
            fx = torch.tensor(cl.fixed_xy, device=d)                            # (F, 2)
            for j in range(min(M, fx.shape[0])):
                centres[:, j, :2] = bolt + rot(fx[j].view(1, 2).expand(n, 2))      # bolt-relative, yaws with it
        else:
            for j in range(M):
                need = active[:, j].clone()
                for _ in range(64):
                    if not bool(need.any()):
                        break
                    ang = torch.rand(n, device=d) * 2 * math.pi
                    rad = torch.rand(n, device=d) * (cl.r_max - cl.r_min) + cl.r_min
                    xy = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], dim=1)
                    # the corridor was measured at yaw 0: test the candidate in the bolt's frame
                    ok = _corridor_ok(unrot(xy), cl)
                    for i in range(j):
                        ok &= torch.norm(bolt + xy - centres[:, i, :2], dim=1) >= cl.min_separation
                    take = need & ok
                    centres[take, j, :2] = (bolt + xy)[take]
                    need &= ~take
                # anything still unplaced is parked (rare); count it as inactive
                active[:, j] &= ~need
        centres[:, :, 2] = torch.where(active, TABLE_TOP + sizes[:, 2].view(1, -1) / 2, park[2])
        centres[:, :, :2] = torch.where(active.unsqueeze(2), centres[:, :, :2], park[:2].view(1, 1, 2))
    half = sizes.view(1, MAX_OBSTACLES, 3) / 2
    b["_obs_lo"][env_ids] = torch.where(active.unsqueeze(2), centres - half, torch.full_like(centres, -10.0))
    b["_obs_hi"][env_ids] = torch.where(active.unsqueeze(2), centres + half, torch.full_like(centres, -10.0))
    b["_obs_active"][env_ids] = active
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=d).expand(n, 4)
    for j in range(M):
        name = f"obstacle_{j}"
        if name not in env.scene.keys():
            continue
        asset = env.scene[name]
        pos = centres[:, j, :] + env.scene.env_origins[env_ids]
        asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=1), env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=d), env_ids=env_ids)


def reset_support(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """AFTER reset_obstacles: put this episode's support block (if any) under the
    tilted rod and into its obstacle AABB slot; park the other blocks."""
    b = _rg(env)
    n, d = len(env_ids), env.device
    j_env = b["_support_j"][env_ids]
    centre = b["_support_c"][env_ids]
    park = torch.tensor(OBSTACLE_PARK, device=d)
    sizes = torch.tensor(SUPPORT_SIZES, device=d)
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=d).expand(n, 4)
    for j in range(len(SUPPORT_SIZES)):
        name = f"support_{j}"
        if name not in env.scene.keys():
            continue
        mine = j_env == j
        c = torch.where(mine.unsqueeze(1), centre, park.view(1, 3).expand(n, 3))
        half = sizes[j] / 2
        b["_obs_lo"][env_ids, j] = torch.where(mine.unsqueeze(1), c - half, torch.full_like(c, -10.0))
        b["_obs_hi"][env_ids, j] = torch.where(mine.unsqueeze(1), c + half, torch.full_like(c, -10.0))
        b["_obs_active"][env_ids, j] = mine
        asset = env.scene[name]
        asset.write_root_pose_to_sim_index(root_pose=torch.cat([c + env.scene.env_origins[env_ids], quat], dim=1), env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=d), env_ids=env_ids)


def reset_bed(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """AFTER reset_cover: the two bed rods under the target (level 24+), as kinematic copies of the
    rod lying on the table, and as capsules in the clearance model; parked otherwise."""
    b = _rg(env)
    n, d = len(env_ids), env.device
    on = b["_on_bed"][env_ids]
    axis = b["_bed_axis"][env_ids]
    park = torch.tensor(OBSTACLE_PARK, device=d)
    for k in BED_SLOTS:
        c = b["_bed_c"][env_ids, k]
        half = BOLT_HALF_LEN * axis
        b["_cap_p0"][env_ids, k] = torch.where(on.unsqueeze(1), c - half, park.view(1, 3).expand(n, 3))
        b["_cap_p1"][env_ids, k] = torch.where(on.unsqueeze(1), c + half, park.view(1, 3).expand(n, 3))
        b["_cap_r"][env_ids, k] = torch.where(on, torch.full((n,), BED_RADIUS, device=d), torch.zeros(n, device=d))
        b["_cap_hard"][env_ids, k] = True
        name = f"bed_{k}"
        if name not in env.scene.keys():
            continue
        # a cylinder (axis z by default) laid along `axis`: 90 deg about y, then yawed to the axis direction
        q_lay = torch.tensor([0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)], device=d).expand(n, 4)
        yaw = torch.atan2(axis[:, 1], axis[:, 0])
        q = math_utils.quat_mul(_zquat(yaw), q_lay)
        q = torch.where(on.unsqueeze(1), q, torch.tensor([0.0, 0.0, 0.0, 1.0], device=d).expand(n, 4))
        asset = env.scene[name]
        asset.write_root_pose_to_sim_index(root_pose=torch.cat([c + env.scene.env_origins[env_ids], q], dim=1), env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=d), env_ids=env_ids)


def _seg_seg_dist(p0, p1, q0, q1) -> torch.Tensor:
    """Closest distance between segments p0-p1 and q0-q1, batched over leading dims (Ericson 5.1.9)."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = (d1 * d1).sum(-1), (d2 * d2).sum(-1), (d2 * r).sum(-1)
    c, b = (d1 * r).sum(-1), (d1 * d2).sum(-1)
    denom = (a * e - b * b).clamp(min=1e-12)
    s_ = ((b * f - c * e) / denom).clamp(0.0, 1.0)
    t_ = (b * s_ + f) / e.clamp(min=1e-12)
    s_ = torch.where(t_ < 0, (-c / a.clamp(min=1e-12)).clamp(0.0, 1.0), torch.where(t_ > 1, ((b - c) / a.clamp(min=1e-12)).clamp(0.0, 1.0), s_))
    t_ = t_.clamp(0.0, 1.0)
    return torch.norm((p0 + s_.unsqueeze(-1) * d1) - (q0 + t_.unsqueeze(-1) * d2), dim=-1)


PILE_CANDIDATES = 12                                  # placements drawn per neighbour; the first that overlaps nothing is used
# Where an unused (dynamic) neighbour waits: on the floor beside the table, 20 cm past its edge, never
# inside a collider (a dynamic body parked inside the table is thrown out by the depenetration).
NEIGHBOR_PARK = (LAYOUT["table_center"][0], LAYOUT["table_center"][1] - LAYOUT["table_size"][1] / 2 - 0.20, BOLT_RADIUS + 0.002)


def reset_pile(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), across_fixed: bool | None = None,
               pile_fixed: bool | None = None):
    """AFTER the bed: the neighbour rods, lying at random yaw in an annulus around the target, placed
    without overlapping anything; with `across_prob` (or `across_fixed`, evaluation) the first one
    resting across a lying target. `prob` (or `pile_fixed`) draws which episodes have the pile at all."""
    lvl = LEVELS[env.cfg.actions.pipeline.level]
    b = _rg(env)
    n, d = len(env_ids), env.device
    ez = torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, 3)
    placed = []                                       # axis segments already in the scene (the target first): no overlaps
    p_pile = lvl.pile.prob if pile_fixed is None else float(pile_fixed)
    pile_on = (torch.rand(n, device=d) < p_pile) & ~b["_fill"][env_ids]      # a full-bin episode has its heap instead
    if "neighbor_0" in env.scene.keys():
        bp0, bq0 = _bolt_true(env)
        ax0 = math_utils.quat_apply(bq0[env_ids], ez)
        placed.append((bp0[env_ids] - BOLT_HALF_LEN * ax0, bp0[env_ids] + BOLT_HALF_LEN * ax0))
        for k_bed in BED_SLOTS:                       # bed rods (hard capsules; parked ones sit far away and never match)
            placed.append((b["_cap_p0"][env_ids, k_bed], b["_cap_p1"][env_ids, k_bed]))
    placed_upper = []                                 # the dropped layer: checked against itself only (it lands on the rest)
    for k, slot in enumerate(NEIGHBOR_SLOTS):
        b["_cap_hard"][env_ids, slot] = False
        upper = lvl.pile.upper > 0 and k >= lvl.pile.count - lvl.pile.upper
        if f"neighbor_{k}" not in env.scene.keys():
            b["_cap_r"][env_ids, slot] = 0.0
            b["_cap_p0"][env_ids, slot] = torch.tensor(OBSTACLE_PARK, device=d)
            b["_cap_p1"][env_ids, slot] = torch.tensor(OBSTACLE_PARK, device=d)
            continue
        if k < lvl.pile.count:
            bp, bq = _bolt_true(env)
            centre = bp[env_ids, :2]
            C = PILE_CANDIDATES
            ang = torch.rand(n, C, device=d) * 2 * math.pi
            r_lo, r_hi = (0.0, 0.06) if upper else lvl.pile.r
            rad = torch.rand(n, C, device=d) * (r_hi - r_lo) + r_lo
            # a TILTED target leans on its block and is knocked off by close neighbours: those keep the level-27 band
            t_lo, t_hi = max(r_lo, NEIGHBOR_R_MIN), max(r_hi, NEIGHBOR_R_MAX)
            rad = torch.where(b["_tilted"][env_ids].unsqueeze(1) & (not upper), torch.rand(n, C, device=d) * (t_hi - t_lo) + t_lo, rad)
            xy_c = centre.unsqueeze(1) + torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], dim=2)      # (n, C, 2)
            yaw_c = torch.rand(n, C, device=d) * 2 * math.pi
            spawn_q = _t(env.scene["object"].data.default_root_pose)[env_ids, 3:7]
            yaw0 = torch.atan2(*reversed(math_utils.quat_apply(spawn_q, ez)[:, :2].unbind(1)))
            # the candidate that overlaps neither the target nor a neighbour already placed (axis-to-axis
            # distance over one diameter plus 2 mm); failing all, the one farthest from everything
            dir_c = torch.stack([torch.cos(yaw_c + yaw0.unsqueeze(1)), torch.sin(yaw_c + yaw0.unsqueeze(1)), torch.zeros_like(yaw_c)], dim=2)
            z_spawn = BOLT_REST_Z + (2 * BOLT_RADIUS + 0.012 if upper else 0.004)     # the dropped layer starts 2.3 cm up
            zc0 = torch.full((n, C, 1), z_spawn, device=d)
            cen_c = torch.cat([xy_c, zc0], dim=2)
            p0c, p1c = cen_c - BOLT_HALF_LEN * dir_c, cen_c + BOLT_HALF_LEN * dir_c
            dmin = torch.full((n, C), 9.0, device=d)
            for (q0, q1) in (placed_upper if upper else placed):
                dmin = torch.minimum(dmin, _seg_seg_dist(p0c, p1c, q0.unsqueeze(1).expand(n, C, 3), q1.unsqueeze(1).expand(n, C, 3)))
            # ... nor a box obstacle (a tilted target's support block, the walls): a rod spawned inside the block
            # is thrown out by the depenetration and knocks the target off its block (measured: up to 3.6 cm, 50 deg)
            lo_b, hi_b = b["_obs_lo"][env_ids], b["_obs_hi"][env_ids]                 # (n, B, 3)
            for f in torch.linspace(0.0, 1.0, 9, device=d):
                pt_ = (p0c + f * (p1c - p0c)).unsqueeze(2)                                # (n, C, 1, 3)
                cl_ = torch.maximum(torch.minimum(pt_, hi_b.unsqueeze(1)), lo_b.unsqueeze(1))
                dmin = torch.minimum(dmin, (torch.norm(pt_ - cl_, dim=3) + BOLT_RADIUS).min(dim=2).values)   # box surface to axis + radius ~ clearance + R
            clear = dmin > 2 * BOLT_RADIUS + 0.002
            first = torch.where(clear.any(dim=1), clear.float().argmax(dim=1), dmin.argmax(dim=1))
            ar = torch.arange(n, device=d)
            xy = xy_c[ar, first]
            z = torch.full((n, 1), z_spawn, device=d)
            q = math_utils.quat_mul(_zquat(yaw_c[ar, first]), spawn_q)
            seg = (p0c[ar, first], p1c[ar, first])
            p_across = lvl.pile.across_prob if across_fixed is None else float(across_fixed)
            if k == 0 and p_across > 0.0:
                # across the target (LYING targets only: a standing rod has no "across", a tilted one is knocked
                # off its block): RESTING on a point of the target's middle 60 %, yawed 90 +/- 30 deg to it, with
                # its near end on the table, so nothing is dropped and the target is not knocked. The contact
                # point sits 40 % of a half-length from the neighbour's centre; the rod is pitched so the
                # near end touches the table.
                across = ((torch.rand(n, device=d) < p_across) & ~b["_upright"][env_ids] & ~b["_tilted"][env_ids]
                          & (b["_cover_side"][env_ids] == 0))          # a covered rod is pinned already; the block would eject the across rod
                axis = math_utils.quat_apply(bq[env_ids], ez)
                s = (torch.rand(n, device=d) * 1.2 - 0.6) * BOLT_HALF_LEN
                pt = bp[env_ids] + s.unsqueeze(1) * axis                           # contact point on the target's axis
                yaw_t = torch.atan2(axis[:, 1], axis[:, 0])
                yaw_n = yaw_t + 0.5 * math.pi + (torch.rand(n, device=d) * 2 - 1) * math.radians(30)
                u = torch.stack([torch.cos(yaw_n), torch.sin(yaw_n), torch.zeros_like(yaw_n)], dim=1)   # the neighbour's horizontal direction
                e = 0.4 * BOLT_HALF_LEN                                          # centre is e beyond the contact, toward the raised far end
                zc = pt[:, 2:3] + 2 * BOLT_RADIUS + 0.001                        # neighbour's axis height at the contact
                # pitch so the near end rests on the table (a target raised on bed rods needs more of it)
                theta = torch.asin(((zc[:, 0] - BOLT_REST_Z) / (BOLT_HALF_LEN - e)).clamp(0.0, 1.0))
                ct, st_ = torch.cos(theta), torch.sin(theta)
                centre = pt[:, :2] + e * ct.unsqueeze(1) * u[:, :2]
                q_flat = math_utils.quat_mul(_zquat(yaw_n - yaw0), spawn_q)
                nrm = torch.stack([-torch.sin(yaw_n), torch.cos(yaw_n), torch.zeros_like(yaw_n)], dim=1)   # z x u
                q_across = math_utils.quat_mul(math_utils.quat_from_angle_axis(-theta, nrm), q_flat)
                xy = torch.where(across.unsqueeze(1), centre, xy)
                z = torch.where(across.unsqueeze(1), zc + (e * st_).unsqueeze(1), z)
                q = torch.where(across.unsqueeze(1), q_across, q)
                far = torch.stack([ct * u[:, 0], ct * u[:, 1], st_], dim=1)
                c3 = torch.cat([xy, z], dim=1)
                seg = (torch.where(across.unsqueeze(1), c3 - BOLT_HALF_LEN * far, seg[0]), torch.where(across.unsqueeze(1), c3 + BOLT_HALF_LEN * far, seg[1]))
            (placed_upper if upper else placed).append(seg)
            use_k = pile_on
        else:                                         # past the level's pile: a full-bin heap's rod or parked
            xy, z = torch.zeros(n, 2, device=d), torch.zeros(n, 1, device=d)
            q = _t(env.scene["object"].data.default_root_pose)[env_ids, 3:7]
            use_k = torch.zeros(n, dtype=torch.bool, device=d)
        f_on = b["_fill"][env_ids]
        if lvl.fill.active and bool(f_on.any()):
            # the full bin: heap rod k, counting past the target; its own axis, a random spin about it
            ar = torch.arange(n, device=d)
            ik = k + (k >= b["_fill_j"][env_ids]).long()
            f_use = f_on & (ik < b["_fill_cnt"][env_ids])
            ikc = ik.clamp(max=FILL_RODS - 1)
            fp = b["_fill_pos"][env_ids][ar, ikc]
            fq = math_utils.quat_mul(_quat_z_to(b["_fill_ax"][env_ids][ar, ikc]), _zquat(torch.rand(n, device=d) * 2 * math.pi))
            xy = torch.where(f_use.unsqueeze(1), fp[:, :2], xy)
            z = torch.where(f_use.unsqueeze(1), fp[:, 2:3], z)
            q = torch.where(f_use.unsqueeze(1), fq, q)
            use_k = (use_k & ~f_on) | f_use
        # episodes without the pile: every neighbour parked (zero radius; rest height = park height)
        park_p = torch.tensor(NEIGHBOR_PARK, device=d).view(1, 3).expand(n, 3)
        park_xy = park_p[:, :2] + torch.tensor([0.04 * (k // 10), -0.03 * (k % 10)], device=d)     # rows of ten
        xy = torch.where(use_k.unsqueeze(1), xy, park_xy)
        z = torch.where(use_k.unsqueeze(1), z, park_p[:, 2:3])
        asset = env.scene[f"neighbor_{k}"]
        pos = torch.cat([xy, z], dim=1) + env.scene.env_origins[env_ids]
        asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos, q], dim=1), env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=d), env_ids=env_ids)
        b["_nb_rest_z"][env_ids, k] = z[:, 0]
        b["_cap_r"][env_ids, slot] = torch.where(use_k, torch.full((n,), BOLT_RADIUS, device=d), torch.zeros(n, device=d))
    b["double"][env_ids] = False


def _heap_top_rod(env, at: torch.Tensor):
    """The heap (PileCfg.top_rod): on the step `at`, a target with a neighbour resting on it is re-labelled:
    the target and the highest rod resting on it swap poses (every rod is the same body, so this only changes
    which one the policy is asked for), the way a perception system would pick the rod on top. Standing and
    tilted targets keep their label (their wrist roll and lift axis were solved at reset); a swapped target
    is a plain lying rod, so the block / bed flags are cleared."""
    if not bool(at.any()) or "neighbor_0" not in env.scene.keys():
        return
    b = _rg(env)
    n, d = env.num_envs, env.device
    _refresh_neighbor_capsules(env)
    bp, bq = _bolt_true(env)
    ez = torch.tensor([0.0, 0.0, 1.0], device=d).expand(n, 3)
    ax = math_utils.quat_apply(bq, ez)
    p0, p1 = bp - BOLT_HALF_LEN * ax, bp + BOLT_HALF_LEN * ax
    ks = [k for k in range(len(NEIGHBOR_SLOTS)) if f"neighbor_{k}" in env.scene.keys()]
    slots = [NEIGHBOR_SLOTS[k] for k in ks]
    q0, q1 = b["_cap_p0"][:, slots], b["_cap_p1"][:, slots]                       # (n, K, 3)
    K = len(ks)
    dist = _seg_seg_dist(p0.unsqueeze(1).expand(n, K, 3), p1.unsqueeze(1).expand(n, K, 3), q0, q1)
    nb_c = 0.5 * (q0 + q1)
    in_heap = torch.norm(nb_c[:, :, :2] - bp[:, :2].unsqueeze(1), dim=2) < 0.20      # parked rods sit far away
    covering = (dist < 2 * BOLT_RADIUS + 0.004) & (nb_c[:, :, 2] > bp[:, 2:3] + 0.5 * BOLT_RADIUS) & in_heap
    # the new target is the highest rod RESTING ON the old one (within a few cm of it), not the highest rod of the
    # heap: the arm's start pose was solved for the old target and the policy trained on +/-2 cm offsets
    z = torch.where(covering, nb_c[:, :, 2], torch.full_like(nb_c[:, :, 2], -1.0))
    top_z, top_k = z.max(dim=1)
    swap = at & covering.any(dim=1) & (top_z > bp[:, 2]) & ~b["_upright"] & ~b["_tilted"]
    if not bool(swap.any()):
        return
    obj = env.scene["object"]
    o_pos = _t(obj.data.root_pos_w).clone()
    o_quat = _t(obj.data.root_quat_w).clone()
    zero6 = torch.zeros(n, 6, device=d)
    for j, k in enumerate(ks):
        ids = (swap & (top_k == j)).nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            continue
        nb = env.scene[f"neighbor_{k}"]
        n_pos = _t(nb.data.root_pos_w)[ids].clone()
        n_quat = _t(nb.data.root_quat_w)[ids].clone()
        obj.write_root_pose_to_sim_index(root_pose=torch.cat([n_pos, n_quat], dim=1), env_ids=ids)
        obj.write_root_velocity_to_sim_index(root_velocity=zero6[ids], env_ids=ids)
        nb.write_root_pose_to_sim_index(root_pose=torch.cat([o_pos[ids], o_quat[ids]], dim=1), env_ids=ids)
        nb.write_root_velocity_to_sim_index(root_velocity=zero6[ids], env_ids=ids)
        b["_rest_z"][ids] = n_pos[:, 2] - env.scene.env_origins[ids, 2]       # the lift is measured from the new target's height
    b["_cover_side"] = torch.where(swap, torch.zeros_like(b["_cover_side"]), b["_cover_side"])
    b["_on_bed"] = b["_on_bed"] & ~swap
    b["_heap_swapped"] = b["_heap_swapped"] | swap


def _refresh_neighbor_capsules(env):
    """Neighbour rods move: their capsules follow their true poses every step."""
    if "neighbor_0" not in env.scene.keys():
        return
    b = _rg(env)
    n = env.num_envs
    ez = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(n, 3)
    for k, slot in enumerate(NEIGHBOR_SLOTS):
        if f"neighbor_{k}" not in env.scene.keys():
            continue
        a = env.scene[f"neighbor_{k}"]
        c = _t(a.data.root_pos_w) - env.scene.env_origins
        ax = math_utils.quat_apply(_t(a.data.root_quat_w), ez)
        b["_cap_p0"][:, slot] = c - BOLT_HALF_LEN * ax
        b["_cap_p1"][:, slot] = c + BOLT_HALF_LEN * ax


def _neighbor_cfg(k: int, collider: str) -> "RigidObjectCfg":
    """A dynamic copy of the rod: a neighbour in the pile. collider "cyl": a primitive cylinder of the rod's radius
    and length instead of the scanned mesh (the full bin: 60 mesh rods per env held 123 GB of host RAM at 512 envs)."""
    if collider == "cyl":
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Neighbor{k}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(BOLT_XY[0] - 0.06 - 0.03 * k, BOLT_XY[1] + 0.08, BOLT_REST_Z + 0.004),
                                                       rot=tuple(_yawed_lying_rot_tuple())),
            spawn=sim_utils.CylinderCfg(
                radius=FILL_CYL_RADIUS, height=2 * BOLT_HALF_LEN,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0,
                                                              solver_position_iteration_count=16, solver_velocity_iteration_count=1),
                mass_props=sim_utils.MassPropertiesCfg(mass=LAYOUT["bolt_mass"]),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.6), roughness=0.5),
                physics_material=sim_utils.RigidBodyMaterialCfg(friction_combine_mode="max", restitution_combine_mode="min",
                                                                 static_friction=0.9, dynamic_friction=0.9, restitution=0.0),
            ),
        )
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Neighbor{k}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(BOLT_XY[0] - 0.06 - 0.03 * k, BOLT_XY[1] + 0.08, BOLT_REST_Z + 0.004),
                                                   rot=tuple(_yawed_lying_rot_tuple())),
        spawn=sim_utils.UsdFileCfg(
            usd_path=bolt_asset(collider),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0,
                                                          solver_position_iteration_count=16, solver_velocity_iteration_count=1),
            mass_props=sim_utils.MassPropertiesCfg(mass=LAYOUT["bolt_mass"]),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(friction_combine_mode="max", restitution_combine_mode="min",
                                                             static_friction=0.9, dynamic_friction=0.9, restitution=0.0),
        ),
    )


def _yawed_lying_rot_tuple():
    return P.WujiPipelineSceneCfg().object.init_state.rot


def reset_walls(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The bin's four walls (level 25+): fixed around BOLT_XY, in obstacle slots 3-6."""
    if "wall_0" not in env.scene.keys():
        return
    b = _rg(env)
    n, d = len(env_ids), env.device
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=d).expand(n, 4)
    for j, (c, size) in enumerate(_bin_walls()):
        cc = torch.tensor(c, device=d).expand(n, 3)
        half = torch.tensor(size, device=d) / 2
        b["_obs_lo"][env_ids, 3 + j] = cc - half
        b["_obs_hi"][env_ids, 3 + j] = cc + half
        b["_obs_active"][env_ids, 3 + j] = True
        asset = env.scene[f"wall_{j}"]
        asset.write_root_pose_to_sim_index(root_pose=torch.cat([cc + env.scene.env_origins[env_ids], quat], dim=1), env_ids=env_ids)


def _bin_walls(level: int | None = None) -> list[tuple[tuple, tuple]]:
    """(centre, size) of the four walls, env-relative: the level's real bin (BinCfg.inner), else the virtual one."""
    cx, cy = BOLT_XY
    bin_ = LEVELS[LEVEL if level is None else level].bin
    if bin_.inner:
        hx, hy = bin_.inner[0] / 2, bin_.inner[1] / 2
        T, Z = bin_.wall, bin_.height
    else:
        hx = hy = BIN_INNER_HALF
        T, Z = BIN_WALL_T, BIN_WALL_H
    zc = TABLE_TOP + Z / 2
    return [((cx + hx + T / 2, cy, zc), (T, 2 * hy + 2 * T, Z)), ((cx - hx - T / 2, cy, zc), (T, 2 * hy + 2 * T, Z)),
            ((cx, cy + hy + T / 2, zc), (2 * hx + 2 * T, T, Z)), ((cx, cy - hy - T / 2, zc), (2 * hx + 2 * T, T, Z))]


def reset_cover(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """AFTER reset_support: the cover block over one end of a lying rod, into obstacle slot 0
    (free for a lying rod); parked otherwise."""
    if "cover_0" not in env.scene.keys():
        return
    b = _rg(env)
    n, d = len(env_ids), env.device
    covered = b["_cover_side"][env_ids] != 0
    c = b["_cover_c"][env_ids]
    half = torch.tensor(COVER_SIZE, device=d) / 2
    b["_obs_lo"][env_ids, 0] = torch.where(covered.unsqueeze(1), c - half, b["_obs_lo"][env_ids, 0])
    b["_obs_hi"][env_ids, 0] = torch.where(covered.unsqueeze(1), c + half, b["_obs_hi"][env_ids, 0])
    b["_obs_active"][env_ids, 0] = b["_obs_active"][env_ids, 0] | covered
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=d).expand(n, 4)
    asset = env.scene["cover_0"]
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([c + env.scene.env_origins[env_ids], quat], dim=1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=d), env_ids=env_ids)


def _bed_rod_cfg(k: int, collider: str) -> "RigidObjectCfg":
    """A kinematic cylinder standing in for a rod under the target: the rod's band radius and
    length, lying on the table. (A kinematic copy of the scanned asset did not collide with the
    target in Isaac Lab 3.0; a primitive does, and it matches the capsule the clearance model uses.)"""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Bed{k}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBSTACLE_PARK, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CylinderCfg(
            radius=BED_RADIUS, height=2 * BOLT_HALF_LEN,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.6), roughness=0.5),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.9, restitution=0.0),
        ),
    )


def _obstacle_cfg(j: int, size: tuple | None = None, name: str = "Obstacle",
                  color: tuple = (0.7, 0.2, 0.2)) -> "RigidObjectCfg":
    """A kinematic box: immovable, collidable, re-posable at reset."""
    sx, sy, sz = size if size is not None else OBSTACLE_SIZES[j]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}{j}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBSTACLE_PARK, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(
            size=(sx, sy, sz),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
        ),
    )


def _rg_limits() -> LimitsCfg:
    """The pipeline's limits with the translation cap doubled: a 10 cm descent
    at 1.67 mm/step is 60 of 120 steps. At 3.33 mm/step it is 30. The
    near-object velocity penalty, not the cap, slows the wrist at the bolt.
    (A function, not a subclass: configclass cannot subclass a class with
    read-only properties.)"""
    return LimitsCfg(max_translation_per_policy_step=0.004, max_wrist_linear_velocity=0.10,
                     workspace_z_max=0.35)


class ReachGraspAction(ActionTerm):
    """8 policy values -> 27 right-side targets + every held joint, with the
    Experiment 4 phase machine, latch and IK reference resets. IK mechanics are
    the pipeline's (see PipelineAction.apply_actions for the derivation)."""

    cfg: "ReachGraspActionCfg"

    def __init__(self, cfg: "ReachGraspActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        robot = self._asset
        lim = cfg.limits
        self._lim = lim
        self._level = LEVELS[cfg.level]
        self._nominal_pos = torch.tensor([WRIST_XY[0], WRIST_XY[1], WRIST_Z_CONTACT], device=self.device)
        self._nominal_quat = torch.tensor(WRIST_ROT, device=self.device)
        if not robot.is_fixed_base:
            raise ValueError("the reach-grasp env expects a fixed-base robot")
        self._arm_ids, arm_names = robot.find_joints(RIGHT_ARM, preserve_order=True)
        assert arm_names == RIGHT_ARM, arm_names
        self._waist_ids, w_names = robot.find_joints(list(WAIST_JOINTS), preserve_order=True)
        assert tuple(w_names) == WAIST_JOINTS, w_names
        self._thumb_ids, _ = robot.find_joints(THUMB_JOINTS, preserve_order=True)
        self._finger_ids, _ = robot.find_joints(FINGER_JOINTS, preserve_order=True)
        body_ids, _ = robot.find_bodies(WRIST_BODY)
        assert len(body_ids) == 1
        self._body_idx = body_ids[0]
        self._jacobi_body_idx = self._body_idx - (1 if robot.is_fixed_base else 0)
        self._jacobi_cols = [j + robot.num_base_dofs for j in self._arm_ids]
        jl = _t(robot.data.joint_pos_limits)[0, self._arm_ids, :]
        self._q_lo_hard, self._q_hi_hard = jl[:, 0].clone(), jl[:, 1].clone()
        self._q_lo = self._q_lo_hard + lim.joint_margin
        self._q_hi = self._q_hi_hard - lim.joint_margin
        self._q_mid = 0.5 * (self._q_lo_hard + self._q_hi_hard)
        self._q_range = (self._q_hi_hard - self._q_lo_hard).clamp(min=1e-6)

        def resolve(table, ids):
            names = [robot.joint_names[i] for i in ids]
            return torch.tensor([table[nm] for nm in names], device=self.device)

        self._open_t = resolve(cfg.q_open, self._thumb_ids)
        self._closed_t = resolve(cfg.q_closed, self._thumb_ids)
        self._open_f = resolve(cfg.q_open, self._finger_ids)
        self._closed_f = resolve(cfg.q_closed, self._finger_ids)

        n = self.num_envs
        self._raw = torch.zeros(n, 8, device=self.device)
        self._processed = torch.zeros(n, 8, device=self.device)
        self._closure = torch.zeros(n, 2, device=self.device)
        self._wrist_cmd = torch.zeros(n, 6, device=self.device)
        self._target_pos = torch.zeros(n, 3, device=self.device)
        self._target_quat = torch.zeros(n, 4, device=self.device)
        self._target_quat[:, 3] = 1.0
        self._base_targets: torch.Tensor | None = None
        self._nominal_xy = torch.tensor(WRIST_XY, device=self.device)
        self._q_cmd = torch.zeros(n, 7, device=self.device)
        self._q_cmd_valid = torch.zeros(n, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self) -> int:
        return 8

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    @property
    def closure(self) -> torch.Tensor:
        return self._closure

    @property
    def target_pose(self):
        return self._target_pos, self._target_quat

    @property
    def q_cmd(self):
        return self._q_cmd

    @property
    def wrist_cmd(self):
        return self._wrist_cmd

    def _ee_pose(self):
        robot = self._asset
        pos = _t(robot.data.body_link_pos_w)[:, self._body_idx, :] - self._env.scene.env_origins
        quat = _t(robot.data.body_link_quat_w)[:, self._body_idx, :]
        return pos, quat

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0.0
        self._processed[env_ids] = 0.0
        self._closure[env_ids] = 0.0
        self._wrist_cmd[env_ids] = 0.0
        # Analytic seed from the nominal pose plus this episode's displacement
        # (the body poses are not necessarily refreshed yet -- see PipelineAction.reset).
        d = getattr(self._env, "_rg_displacement", None)
        b = _rg(self._env)
        nom_pos, nom_quat = b["_nom_pos"][env_ids], b["_nom_quat"][env_ids]
        if d is None:
            self._target_pos[env_ids] = nom_pos
            self._target_quat[env_ids] = nom_quat
        else:
            dd = d[env_ids]
            m = b["_lift_dir"][env_ids]                     # the standoff runs along the approach axis
            self._target_pos[env_ids] = nom_pos + torch.stack([dd[:, 0], dd[:, 1], torch.zeros_like(dd[:, 2])], dim=1) + dd[:, 2:3] * m
            ang = torch.norm(dd[:, 3:6], dim=1)
            axis = dd[:, 3:6] / ang.unsqueeze(1).clamp(min=1e-9)
            self._target_quat[env_ids] = math_utils.quat_mul(math_utils.quat_from_angle_axis(ang, axis), nom_quat)
        self._q_cmd_valid[env_ids] = False

    def _settled(self) -> torch.Tensor:
        """False during the heap's settle-in at the start of the episode (policy frozen, hand still)."""
        s_in = self._level.pile.settle_in
        t = self._env.episode_length_buf
        return t >= s_in if s_in > 0 else torch.ones_like(t, dtype=torch.bool)

    def _live(self, grasping: torch.Tensor) -> torch.Tensor:
        """Envs whose wrist follows the policy this step: the grasp phase (after any settle-in), plus
        (free_prelift) the first DRAG_STEPS of the pre-lift of a latched rod."""
        grasping = grasping & self._settled()
        if not self._level.free_prelift:
            return grasping
        b = _rg(self._env)
        t_pre = self._env.episode_length_buf.float() - (b["grasp_end"] + SETTLE_STEPS)
        return grasping | (b["latched"] & (t_pre >= 0) & (t_pre < DRAG_STEPS))

    def process_actions(self, actions: torch.Tensor):
        env = self._env
        lim = self._lim
        b = _rg(env)
        self._raw[:] = actions.clamp(-1.0, 1.0)
        a = self._raw.clone()

        # ---- the latch: evaluate on THIS step's state, before the command -------
        now = update_latch(env)
        phase = phase_of(env)
        grasping = phase == 0
        # The actor mask for the transition (s_t, a_t): only grasp-phase actions train the actor
        # (and, at free-pre-lift levels, the policy's window after the settle: wrist live, fingers held).
        live = self._live(grasping)
        env.extras["actor_mask"] = live.clone()

        if bool(now.any()):
            cur_pos, cur_quat = self._ee_pose()
            q = _t(self._asset.data.joint_pos)[:, self._arm_ids]
            m = now.unsqueeze(1)
            # 1-2. record the measured wrist pose, arm state and finger targets
            b["latch_wrist_pos"] = torch.where(m, cur_pos, b["latch_wrist_pos"])
            b["latch_wrist_quat"] = torch.where(m, cur_quat, b["latch_wrist_quat"])
            b["latch_q"] = torch.where(m, q, b["latch_q"])
            b["latch_closure"] = torch.where(m, self._closure, b["latch_closure"])
            # 4-5. clear the residual, reset the IK reference and integrator to the measured state
            self._wrist_cmd = torch.where(m, torch.zeros_like(self._wrist_cmd), self._wrist_cmd)
            self._target_pos = torch.where(m, cur_pos, self._target_pos)
            self._target_quat = torch.where(m, cur_quat, self._target_quat)
            self._q_cmd = torch.where(m, q, self._q_cmd)

        # retry (level 22+): open the hand, clear the residual, re-seed the IK from the measured state
        rn = b.pop("_retry_now", None)
        if rn is not None and bool(rn.any()):
            cur_pos, cur_quat = self._ee_pose()
            q = _t(self._asset.data.joint_pos)[:, self._arm_ids]
            m = rn.unsqueeze(1)
            self._closure = torch.where(m, torch.zeros_like(self._closure), self._closure)
            self._wrist_cmd = torch.where(m, torch.zeros_like(self._wrist_cmd), self._wrist_cmd)
            if self._level.retry_backoff > 0.0:          # the heap: back out of the pile before trying again
                cur_pos = cur_pos.clone()
                cur_pos[:, 2] += self._level.retry_backoff
            self._target_pos = torch.where(m, cur_pos, self._target_pos)
            self._target_quat = torch.where(m, cur_quat, self._target_quat)
            self._q_cmd = torch.where(m, q, self._q_cmd)
            phase = phase_of(env)
            grasping = phase == 0
            live = self._live(grasping)
            env.extras["actor_mask"] = live.clone()

        # 3. zero the effective wrist action outside the grasp phase (and the policy's pre-lift window)
        frozen = (~live).unsqueeze(1)
        a[:, 0:6] = torch.where(frozen, torch.zeros_like(a[:, 0:6]), a[:, 0:6])
        self._processed[:] = a
        self._wrist_cmd = torch.where(frozen, torch.zeros_like(self._wrist_cmd), self._wrist_cmd)

        # Shield hook (second increment): alpha in [0, 1] scales the Cartesian command.
        alpha = b["alpha"]
        cmd = torch.cat([a[:, 0:3] * lim.effective_translation, a[:, 3:6] * lim.effective_rotation], dim=1)
        if self._level.free_grasp:
            cmd[:, 5] = cmd[:, 5] * self._level.yaw_rate_scale       # world-z rotation: alignment made slow
        cmd = cmd * alpha.unsqueeze(1)
        self._wrist_cmd += lim.wrist_smoothing * (cmd - self._wrist_cmd)

        # 6. finger closures: live in the grasp phase, HELD at the latched value afterwards
        c = (a[:, 6:8] + 1.0) / 2.0
        new = self._closure + lim.action_smoothing * (c - self._closure)
        self._closure[:] = torch.where((grasping & self._settled()).unsqueeze(1), new, self._closure)
        if self._level.latch_squeeze > 0.0:
            # squeeze: a latched grasp tightens by latch_squeeze over the settle, before the test lift, so the grip
            # does not rest on the light closure the latch fired at (the torque test fires from ~4 N)
            sq = (b["latched"] & ~grasping).unsqueeze(1)
            top = (b["latch_closure"] + self._level.latch_squeeze).clamp(max=1.0)
            up = torch.minimum(self._closure + self._level.latch_squeeze / SETTLE_STEPS, top)
            self._closure[:] = torch.where(sq, torch.maximum(self._closure, up), self._closure)

        # ---- advance the commanded wrist pose (control rate, see PipelineAction) --
        d_pos = self._wrist_cmd[:, 0:3].clone()
        d_rot = self._wrist_cmd[:, 3:6].clone()
        lifting = (phase == 2).unsqueeze(1)
        # 10 cm along the approach axis (vertical, or the rod's up-normal when the wrist is rolled):
        # LIFT_SLOW_STEPS at 1 mm to let the grip settle, then the rest at an even rate
        tt = env.episode_length_buf.float()
        t_lift = tt - (b["grasp_end"] + LIFT_START)
        fast_step = (LIFT_HEIGHT - LIFT_SLOW_STEPS * LIFT_SLOW_STEP_M) / (LIFT_STEPS - LIFT_SLOW_STEPS)
        step_len = torch.where(t_lift < LIFT_SLOW_STEPS, torch.full_like(t_lift, LIFT_SLOW_STEP_M), torch.full_like(t_lift, fast_step))
        lift_step = b["_lift_dir"] * step_len.unsqueeze(1)
        # pre-lift (latched envs only): the slide-out drag for covered rods (or the policy's window at
        # free-pre-lift levels), then the test lift: 1 mm/step, or 24 mm in all at grip-check levels
        # (over the whole pre-lift when there is no drag and no policy window)
        t_pre = tt - (b["grasp_end"] + SETTLE_STEPS)
        in_pre = ((t_pre >= 0) & (t_pre < PRE_LIFT_STEPS) & b["latched"]).unsqueeze(1)
        first = (t_pre < DRAG_STEPS).unsqueeze(1)
        if self._level.grip_check:
            n_test = GRIP_CHECK_STEPS if self._level.free_prelift else PRE_LIFT_STEPS
            test_step = b["_lift_dir"] * (GRIP_TEST_TOTAL_M / n_test)
        else:
            test_step = b["_lift_dir"] * LIFT_SLOW_STEP_M
        no_drag = (torch.norm(b["_drag_dir"], dim=1) == 0).unsqueeze(1)
        first_step = b["_drag_dir"] * DRAG_STEP_M
        if self._level.grip_check and not self._level.free_prelift:
            first_step = torch.where(no_drag, test_step, first_step)
        pre_step = torch.where(first, first_step, test_step)
        if self._level.free_prelift:
            drag_in = in_pre & first & ~no_drag if self._level.cover_drag else torch.zeros_like(in_pre)
            in_pre = in_pre & ~first                       # the first DRAG_STEPS are the policy's: d_pos stays live
        else:
            drag_in = torch.zeros_like(in_pre)
        d_pos = torch.where(lifting, lift_step, torch.where(in_pre, pre_step, torch.where(frozen, torch.zeros_like(d_pos), d_pos)))
        d_rot = torch.where(frozen, torch.zeros_like(d_rot), d_rot)
        # cover_drag (level 47+): a latched COVERED rod is slid out from under the block in the policy's window,
        # DRAG_STEPS x DRAG_STEP_M along the rod's axis away from the block, the wrist's rotation held
        d_pos = torch.where(drag_in, first_step, d_pos)
        d_rot = torch.where(drag_in, torch.zeros_like(d_rot), d_rot)
        if self._level.cover_drag:
            env.extras["actor_mask"] = env.extras["actor_mask"] & ~drag_in.squeeze(1)   # the slide is scripted: not the policy's

        # 7. the lift starts from the MEASURED wrist pose: re-seed at lift entry.
        t = env.episode_length_buf.float()
        entering_lift = (t == b["grasp_end"] + SETTLE_STEPS).unsqueeze(1)      # the pre-lift starts from the measured pose too
        if bool(entering_lift.any()):
            cur_pos, cur_quat = self._ee_pose()
            self._target_pos = torch.where(entering_lift, cur_pos, self._target_pos)
            self._target_quat = torch.where(entering_lift, cur_quat, self._target_quat)

        self._target_pos += d_pos
        ang = torch.norm(d_rot, dim=1, keepdim=True)
        axis = d_rot / ang.clamp(min=1e-9)
        dq = math_utils.quat_from_angle_axis(ang.squeeze(1), axis)
        self._target_quat[:] = math_utils.quat_mul(dq, self._target_quat)

        # Workspace clamp on the target, around this episode's (bolt-relative) nominal wrist xy.
        centre = b["_nom_pos"][:, :2]
        off = self._target_pos[:, :2] - centre
        r = torch.norm(off, dim=1, keepdim=True)
        scale = (lim.workspace_radius_xy / r.clamp(min=1e-9)).clamp(max=1.0)
        self._target_pos[:, :2] = centre + off * scale
        self._target_pos[:, 2] = self._target_pos[:, 2].clamp(TABLE_TOP + lim.workspace_z_min,
                                                              TABLE_TOP + lim.workspace_z_max)

    def apply_actions(self):
        robot = self._asset
        lim = self._lim
        m = _metrics(self._env)
        n = self.num_envs
        if self._base_targets is None:
            self._base_targets = _t(robot.data.default_joint_pos)[0:1].clone()

        cur_pos, cur_quat = self._ee_pose()
        e_pos = self._target_pos - cur_pos
        q_err = math_utils.quat_mul(self._target_quat, math_utils.quat_inv(cur_quat))
        e_rot = math_utils.axis_angle_from_quat(q_err)
        err = torch.cat([e_pos, e_rot], dim=1).unsqueeze(2)
        m["wrist_track_err"] = torch.norm(e_pos, dim=1)
        m["wrist_track_err_rot"] = torch.norm(e_rot, dim=1)

        J = _t(robot.data.body_link_jacobian_w)[:, self._jacobi_body_idx, :, :][:, :, self._jacobi_cols]
        JT = J.transpose(1, 2)
        JJT = J @ JT
        eye6 = torch.eye(6, device=self.device).expand(n, 6, 6)
        damped = JJT + (lim.dls_damping ** 2) * eye6

        q = _t(robot.data.joint_pos)[:, self._arm_ids]
        if not bool(self._q_cmd_valid.all()):
            fresh = ~self._q_cmd_valid
            self._q_cmd[fresh] = q[fresh]
            self._q_cmd_valid |= True
        slack = 2.0 * torch.minimum(q - self._q_lo_hard, self._q_hi_hard - q) / self._q_range
        urgency = (1.0 - slack.clamp(0.0, 1.0)) ** 2
        dq_null = lim.nullspace_gain * urgency * (self._q_mid - q) / self._q_range
        rhs = err - (J @ dq_null.unsqueeze(2))
        dq = (JT @ torch.linalg.solve(damped, rhs)).squeeze(2) + dq_null

        bad = ~torch.isfinite(dq).all(dim=1)
        dq = torch.nan_to_num(dq, nan=0.0, posinf=0.0, neginf=0.0)
        dq = dq.clamp(-lim.max_joint_velocity * CONTROL_DT, lim.max_joint_velocity * CONTROL_DT)
        q_des = q + dq
        q_clipped = q_des.clamp(self._q_lo, self._q_hi)
        self._q_cmd[:] = q_clipped
        m["clipped"] = (q_clipped != q_des).float().sum(dim=1)
        manip = torch.sqrt(torch.linalg.det(JJT).clamp(min=0.0))
        m["singular"] = (manip < lim.singularity_threshold).float()
        m["ik_failed"] = (bad | (manip < lim.singularity_threshold)).float()
        slack_des = 2.0 * torch.minimum(q_clipped - self._q_lo_hard, self._q_hi_hard - q_clipped) / self._q_range
        m["arm_slack"] = slack_des
        m["s_min"], m["s_argmin"] = slack_des.min(dim=1)

        targets = self._base_targets.expand(n, -1).clone()
        targets[:, self._arm_ids] = q_clipped
        targets[:, self._waist_ids] = _rg(self._env)["_waist"]      # held: the bucket's triple, the pipeline's below 14
        c_t, c_f = self._closure[:, 0:1], self._closure[:, 1:2]
        targets[:, self._thumb_ids] = self._open_t + c_t * (self._closed_t - self._open_t)
        targets[:, self._finger_ids] = self._open_f + c_f * (self._closed_f - self._open_f)
        robot.set_joint_position_target(targets)


@configclass
class ReachGraspActionCfg(ActionTermCfg):
    class_type: type = ReachGraspAction
    asset_name: str = "robot"
    limits: LimitsCfg = _rg_limits()
    level: int = LEVEL
    q_open: dict = field(default_factory=lambda: dict(Q_OPEN))
    q_closed: dict = field(default_factory=lambda: dict(Q_CLOSED))


# ---------------------------------------------------------------------------
# Scene: the pipeline's, plus sensors on the real forearm and upper arm
# ---------------------------------------------------------------------------


@configclass
class WujiReachGraspSceneCfg(WujiPipelineSceneCfg):
    # The pipeline's "forearm" sensor sits on wristPitchRight, a 4 cm knuckle;
    # the 26 cm forearm proper is wristYawRight. Likewise elbowRight vs the
    # 23 cm shoulderYawRight upper arm. Both are monitored here.
    cs_forearm2 = _body_sensor("wristYawRight")
    cs_upperarm2 = _body_sensor("shoulderYawRight")


# ---------------------------------------------------------------------------
# MDP
# ---------------------------------------------------------------------------


@configclass
class ActionsCfg:
    # Named `pipeline` so the pipeline's joint-limit penalty / terminations,
    # which read env.cfg.actions.pipeline.limits, work unchanged.
    pipeline = ReachGraspActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        bolt_pos_rel = ObsTerm(func=bolt_pos_in_wrist)                 # 3   (estimated pose)
        bolt_rot_rel = ObsTerm(func=bolt_rot_in_wrist)                 # 6   (estimated pose)
        wrist_lin_vel = ObsTerm(func=wrist_lin_vel)                    # 3
        wrist_ang_vel = ObsTerm(func=wrist_ang_vel)                    # 3
        finger_q = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=SYNERGY_JOINTS, preserve_order=True)})        # 9
        finger_qd = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=SYNERGY_JOINTS, preserve_order=True)})        # 9
        last_action = ObsTerm(func=mdp.last_action)                    # 8
        tip_forces = ObsTerm(func=tip_forces_obs)                      # 3
        contacts = ObsTerm(func=contact_flags)                         # 3
        ep_time = ObsTerm(func=episode_time)                           # 1
        arm_q_norm = ObsTerm(func=arm_q_normalized)                    # 7
        tip_dist = ObsTerm(func=fingertip_distances_est)               # 3   (estimated pose)
        safety = ObsTerm(func=safety_obs)                              # 16
                                                                       # = 74

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PolicyHeapCfg(PolicyCfg):
        """Level 36+: the 74 above plus the heap map (appended, so every earlier index is unchanged)."""
        heap_map = ObsTerm(func=heap_map_obs)                          # 8   (estimated pose, true neighbours)

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    reach = RewTerm(func=reach_progress, weight=REACH_WEIGHT)
    fingers = RewTerm(func=finger_progress, weight=FINGER_WEIGHT)
    grasp_point = RewTerm(func=grasp_point_progress, weight=GRASP_POINT_WEIGHT)   # binding levels only
    velocity = RewTerm(func=near_object_velocity, weight=1.0)          # coefficients inside, returns <= 0
    thumb_contact = RewTerm(func=thumb_contact_bonus, weight=0.5)
    finger_contact = RewTerm(func=finger_contact_bonus, weight=0.5)
    opposing_contact = RewTerm(func=opposing_contact_bonus, weight=1.0)
    success = RewTerm(func=success_bonus, weight=20.0)
    drop = RewTerm(func=drop_penalty, weight=-5.0)
    excess_force = RewTerm(func=excess_force_penalty, weight=-3.0)
    invalid_collision = RewTerm(func=invalid_collision_penalty, weight=-10.0)
    clearance = RewTerm(func=clearance_penalty, weight=CLEARANCE_WEIGHT)
    shield = RewTerm(func=shield_penalty, weight=SHIELD_WEIGHT)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    joint_limits = RewTerm(func=joint_limit_penalty, weight=-0.05)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    phase_complete = DoneTerm(func=phase_complete)
    off_table = DoneTerm(func=bolt_off_table)
    joint_limit = DoneTerm(func=joint_limit_termination)
    ik_failure = DoneTerm(func=ik_failure_termination)
    invalid_collision = DoneTerm(func=invalid_collision_termination)


@configclass
class EventsCfg:
    # FIRST: samples the bolt's yaw / xy and sets the bolt-relative nominal
    # wrist pose that the standoff reset, the reach potential, the workspace
    # clamp and the obstacle placement all read.
    reset_bolt = EventTerm(func=reset_bolt_pose, mode="reset", params={})
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )
    # AFTER reset_robot: offsets the arm from the contact pose and holds it.
    standoff = EventTerm(func=reset_wrist_standoff, mode="reset", params={})
    reset_buffers = EventTerm(func=reset_rg_buffers, mode="reset", params={})
    # AFTER reset_buffers (which clears the obstacle-clearance running minima).
    obstacles = EventTerm(func=reset_obstacles, mode="reset", params={})
    # AFTER obstacles: the tilted rod's support block takes an obstacle slot.
    support = EventTerm(func=reset_support, mode="reset", params={})
    # AFTER support: the cover block over one end of a lying rod (level 23+).
    cover = EventTerm(func=reset_cover, mode="reset", params={})
    # AFTER cover: the bed rods (level 24+) and the bin walls (level 25+).
    bed = EventTerm(func=reset_bed, mode="reset", params={})
    pile = EventTerm(func=reset_pile, mode="reset", params={})
    walls = EventTerm(func=reset_walls, mode="reset", params={})


class ReachGraspEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv plus per-step logging that survives Isaac Lab's reset.

    Isaac Lab replaces extras["log"] with an empty dict inside _reset_idx,
    which runs AFTER the action term and BEFORE the step returns. Anything
    published from an action or reward term therefore vanishes on every step
    in which any env resets -- with 2048 envs, nearly every step -- and rsl-rl's
    logger only keeps keys present in the FIRST extras of an iteration. That is
    why the pipeline's Safety/* never reached TensorBoard. Publishing here,
    after super().step(), fixes both: every key is present on every step, as a
    GPU tensor, so there is no per-step host synchronisation either.
    """

    def step(self, action):
        out = super().step(action)
        self._publish_logs()
        return out

    def _publish_logs(self):
        log = self.extras.setdefault("log", {})
        b = _rg(self)
        m = _metrics(self)
        ph = phase_of(self)
        log["Safety/min_joint_slack"] = m["s_min"].mean()
        log["Safety/min_joint_slack_worst"] = m["s_min"].min()
        log["Safety/ik_clipped_targets"] = m["clipped"].mean()
        log["Safety/ik_failures"] = m["ik_failed"].mean()
        log["Safety/wrist_tracking_error_m"] = m["wrist_track_err"].mean()
        log["Safety/invalid_collision_now"] = b["invalid"].float().mean()
        log["Safety/shield_alpha"] = b["alpha"].mean()
        log["Safety/shield_intervening"] = (b["alpha"] < 0.999).float().mean()
        for i, g in enumerate(GROUP_NAMES):
            log[f"Clearance/{g}_mean"] = b["clear"][:, i].mean()
            log[f"Clearance/{g}_min"] = b["clear"][:, i].min()
        log["Clearance/obstacle_min"] = b["obs_clear"].min()
        log["Outcome/obstacle_hit"] = (b["last_invalid"] & (b["last_invalid_which"] == 4)).float().mean()
        log["Outcome/min_clear_obstacle"] = b["last_min_obs_clear"].mean()
        log["Wrist/speed"] = torch.norm(wrist_lin_vel(self), dim=1).mean()
        log["Wrist/accel"] = torch.norm(b["acc"], dim=1).mean()
        log["Wrist/jerk"] = torch.norm(b["jerk"], dim=1).mean()
        log["Phase/grasp_frac"] = (ph == 0).float().mean()
        log["Phase/latched_frac"] = b["latched"].float().mean()
        log["Phase/grasp_end_mean"] = b["grasp_end"].mean()
        # Outcomes of the most recently ENDED episode per env: always present, so
        # rsl-rl keeps the key; a moving average over the population.
        log["Outcome/success"] = b["last_success"].float().mean()
        log["Outcome/held"] = b["last_held"].float().mean()
        log["Outcome/latch_point_err_mm"] = b["last_latch_s_err"].mean() * 1000.0
        log["Outcome/latch_centre_err_mm"] = b["last_latch_d_err"].mean() * 1000.0
        log["Outcome/retried"] = (b["last_attempts"] > 0).float().mean()
        log["Outcome/double_pick"] = b["last_double"].float().mean()
        log["Outcome/retry_success"] = b["last_retry_success"].float().mean()
        log["Outcome/grip_check_fired"] = b["last_grip_fired"].float().mean()
        log["Outcome/invalid_collision"] = b["last_invalid"].float().mean()
        log["Outcome/excess_force"] = b["last_excess"].float().mean()
        log["Outcome/drop"] = b["last_drop"].float().mean()
        log["Outcome/latched"] = b["last_latched"].float().mean()
        log["Outcome/grasp_end"] = b["last_grasp_end"].mean()
        log["Outcome/bolt_drag_mm"] = b["last_drag"].mean() * 1000.0
        log["Outcome/jerk_max"] = b["last_jerk_max"].mean()
        for i, g in enumerate(GROUP_NAMES):
            log[f"Outcome/min_clear_{g}"] = b["last_min_clear"][:, i].mean()


@configclass
class WujiReachGraspEnvCfg(ManagerBasedRLEnvCfg):
    scene: WujiReachGraspSceneCfg = WujiReachGraspSceneCfg(num_envs=16, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = PHYSICS_HZ // CONTROL_HZ
        lvl = LEVELS[self.actions.pipeline.level]
        if lvl.heap_map:
            self.observations.policy = ObservationsCfg.PolicyHeapCfg()
        if lvl.bolt_collider != "hull":
            self.scene.object.spawn.usd_path = bolt_asset(lvl.bolt_collider)
        # One step MORE than the phase machine ever needs, so `time_out` never
        # pre-empts `phase_complete` (a termination, which is not bootstrapped).
        self.episode_length_s = (level_max_steps(LEVELS[self.actions.pipeline.level]) + 1) / CONTROL_HZ
        self.sim.dt = 1.0 / PHYSICS_HZ
        if lvl.pile.active:
            # four loose rods per env overflow PhysX's default GPU patch buffer at 2048 envs (level 27's first
            # run: "Patch buffer overflow", dropped contacts, 34-44 % invalid collisions); raise the buffers
            from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
            x = 4 if lvl.fill.active else 1          # 60 rods per env in a full bin
            self.sim.physics = PhysxCfg(gpu_max_rigid_patch_count=x * 2**21, gpu_max_rigid_contact_count=x * 2**24,
                                        gpu_found_lost_pairs_capacity=x * 2**23, gpu_total_aggregate_pairs_capacity=x * 2**23)
        self.sim.render_interval = self.decimation
        self.sim.device = "cuda:0"
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0,
        )
        if hasattr(self.sim, "disable_contact_processing"):
            self.sim.disable_contact_processing = False

        lvl = LEVELS[self.actions.pipeline.level]
        if lvl.clutter.active:
            # Obstacle slots exist only at clutter levels: the interface (74 obs)
            # is the same either way, since the clearance model reads parked
            # slots as infinitely far.
            for j in range(MAX_OBSTACLES):
                setattr(self.scene, f"obstacle_{j}", _obstacle_cfg(j))
        if lvl.tilt.active:
            assert not lvl.clutter.active, "the support blocks use the obstacle AABB slots"
            for j, size in enumerate(SUPPORT_SIZES):
                setattr(self.scene, f"support_{j}", _obstacle_cfg(j, size, "Support", (0.35, 0.35, 0.4)))
        if lvl.cover.active:
            assert not lvl.clutter.active, "the cover block uses obstacle AABB slot 0"
            setattr(self.scene, "cover_0", _obstacle_cfg(0, COVER_SIZE, "Cover", (0.2, 0.3, 0.6)))
        if lvl.bed.active:
            for k in BED_SLOTS:
                setattr(self.scene, f"bed_{k}", _bed_rod_cfg(k, lvl.bolt_collider))
        if lvl.bin_walls:
            for j, (c, size) in enumerate(_bin_walls(self.actions.pipeline.level)):
                setattr(self.scene, f"wall_{j}", _obstacle_cfg(3 + j, size, "Wall", (0.5, 0.5, 0.5)))
        if lvl.pile.active:
            assert 0 < lvl.pile.count <= len(NEIGHBOR_SLOTS), f"the pile has at most {len(NEIGHBOR_SLOTS)} neighbour slots"
            n_nb = lvl.pile.count
            if lvl.fill.active:
                # the full bin: every rod of the heap but the target, as primitive cylinders (what the heaps settled
                # as; adam-moveit's convexHull of the rod is close to one); the target keeps the level's collider
                assert len(NEIGHBOR_SLOTS) >= FILL_RODS - 1, "the fill needs ADAM_U_RG_LEVEL set to the fill level at import"
                n_nb = FILL_RODS - 1
            for k in range(n_nb):
                setattr(self.scene, f"neighbor_{k}", _neighbor_cfg(k, "cyl" if lvl.fill.active else lvl.bolt_collider))
        if lvl.physics.active:
            # Physics robustness (level 5+). Applied at reset, per env.
            self.events.bolt_mass = EventTerm(
                func=mdp.randomize_rigid_body_mass, mode="reset",
                params={"asset_cfg": SceneEntityCfg("object"), "mass_distribution_params": lvl.physics.mass_scale,
                        "operation": "scale", "recompute_inertia": True})
            self.events.bolt_material = EventTerm(
                func=mdp.randomize_rigid_body_material, mode="reset",
                params={"asset_cfg": SceneEntityCfg("object"),
                        "static_friction_range": tuple(0.9 * s for s in lvl.physics.friction_scale),
                        "dynamic_friction_range": tuple(0.9 * s for s in lvl.physics.friction_scale),
                        "restitution_range": (0.0, 0.0), "num_buckets": 64})
            self.events.finger_gains = EventTerm(
                func=mdp.randomize_actuator_gains, mode="reset",
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=SYNERGY_JOINTS + (list(RIGHT_ARM) if lvl.arm_gains else [])),
                        "stiffness_distribution_params": lvl.physics.gain_scale,
                        "damping_distribution_params": lvl.physics.gain_scale, "operation": "scale"})

        self.viewer.eye = (BOLT_XY[0] + 0.45, BOLT_XY[1] - 0.45, TABLE_TOP + 0.40)
        self.viewer.lookat = (BOLT_XY[0], BOLT_XY[1], TABLE_TOP + 0.03)


__all__ = ["WujiReachGraspEnvCfg", "ReachGraspEnv", "RGLevel", "ClutterCfg", "LEVELS", "LEVEL", "STANDOFF_BUCKETS",
           "BOLT_OFFSET_XY", "_nominal_wrist", "_zquat",
           "OBSTACLE_SIZES", "MAX_OBSTACLES", "OBSTACLE_PEN_M", "_corridor_ok",
           "GRASP_MAX_STEPS", "SETTLE_STEPS", "LIFT_STEPS", "HOLD_STEPS", "EPISODE_MAX_STEPS", "LIFT_START", "PRE_LIFT_STEPS",
           "GROUP_NAMES", "MARGINS", "PROXY_GROUPS", "phase_of", "_rg", "_ids", "_update", "_est_bolt",
           "_bolt_true", "_clearance", "fingertip_distances_true", "fingertip_distances_est",
           "grasp_ready", "update_latch"]
