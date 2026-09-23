# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Staged bolt-grasping pipeline: whole Adam-U robot, right arm + right hand only.

Design: docs/GRASP_PIPELINE_DESIGN.md. One environment for every stage. The
interface is FIXED -- 55 observations, 8 actions, one network shape -- and a
stage is nothing but a mask over the 8 actions plus a set of randomisation
ranges. That is what lets a stage-k checkpoint load into stage k+1, and a
behaviour-cloned actor load into PPO, with no surgery.

Episode (5.0 s, 150 steps at 30 Hz control / 120 Hz physics):

  phase 1  GRASP   steps   0- 89 (3.0 s)  the policy drives the wrist and the
                                          two finger synergies
  phase 2  LIFT    steps  90-119 (1.0 s)  finger targets frozen; the wrist
                                          target ramps +LIFT_HEIGHT in WORLD z
                                          through the same IK; actions ignored
  phase 3  HOLD    steps 120-149 (1.0 s)  everything frozen; success is judged
                                          at the last step

Three seconds of grasp, not Experiment 2's two: the wrist has to travel before
it can close. At the conservative 2 mm/step a 5 cm descent alone is 25 steps.

Actions (8, each in [-1, 1]):

    0  wrist delta-x            }  Cartesian RESIDUALS on the wrist pose
    1  wrist delta-y            }  target, resolved to the 7 right-arm joint
    2  wrist delta-z            }  targets by damped least-squares IK with a
    3  wrist delta rotation-x   }  joint-limit-aware null-space term. The
    4  wrist delta rotation-y   }  policy NEVER commands arm joints directly.
    5  wrist delta rotation-z   }  Rotations are small axis-angle increments.
    6  thumb closure synergy       q = q_open + c * (q_closed - q_open)
    7  index/middle synergy        over the 9 driven finger joints

A stage masks actions it has not turned on: the mask zeroes the raw action AND
the scale, so an unmasked stage inherits a policy whose output on that channel
was meaningless but harmless.

Commanded joints (the right-side adapter turns 8 policy values into 27):

    right arm         7   differential IK from actions 0-5
    right hand       20   9 driven by actions 6-7, 11 held at fixed safe targets
                          ---
                         27 right-side command targets

Everything else is ACTIVELY HELD, never left to drift: waist (3, at the solved
lean), neck (2), left arm (7), left hand (20, open). The waist is deliberately
NOT in the policy and NOT in the IK chain -- see docs/GRASP_PIPELINE_DESIGN.md.

Why every joint is written every step. Joint drive targets initialise to ZERO,
and reset events write joint STATES only, so a joint the action manager never
addresses holds 0 rather than its default pose. Measured on this robot:
waistPitch with a default of 0.352 settled at 0.030. This env therefore builds
the COMPLETE (num_envs, num_joints) target tensor from the robot's defaults on
every control step and overwrites only the 27 it commands. There is no path by
which a held joint can be left at zero, and tools/test_pipeline_arch.py asserts
it. `hold_default_joint_targets` still runs on reset as a second line.

Observations (55, normalised by rsl-rl's running normaliser):

    bolt position relative to the wrist, in the wrist's frame     3
    bolt orientation relative to the wrist, 6D                    6
    wrist linear velocity                                         3
    wrist angular velocity                                        3
    driven finger joint positions                                 9
    driven finger joint velocities                                9
    previous policy action (all 8, masked entries included)       8
    thumb / index / middle fingertip force on the bolt            3
    thumb / index / middle contact flags                          3
    normalised episode time                                       1
    normalised right-arm joint positions                          7
                                                                 ---
                                                                 55

The 6D rotation rather than a quaternion because a quaternion's sign ambiguity
is something the network would have to learn around. The 7 normalised arm
positions rather than one minimum-slack scalar because a scalar says some joint
is near a limit without saying which, which end, or which Cartesian correction
moves away from it. s_min is kept as a logged metric and as the trigger for the
soft penalty and the hard safety termination, not as an observation.

Fingertip contact force stands in for the real Wuji tactile pads. PhysX gives
ONE aggregate force per fingertip, not a taxel array, so the pad's "total
force" and "maximum taxel" collapse to the same number and appear once.

Reward, success test and contact bookkeeping are Experiment 2's, imported
rather than reimplemented, plus a soft joint-limit penalty. Keeping them
identical is what makes stage 0 comparable with the 100 % already measured.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
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
from adam_u_rl.envs.adam_u_wuji_grasp_env_cfg import (
    ADAM_U_WUJI_USD,
    LAYOUT,
    ROBOT_ROOT_Z,
    _static_box,
    _t,
)
from adam_u_rl.envs.wuji_synergy_grasp_env_cfg import (
    BOLT_HALF_EXTENT,
    _once,
    BOLT_REST_Z,
    CONTACT_N,
    F_MAX,
    STABLE_STEPS,
    SUCCESS_LIFT,
    TABLE_HIT_N,
    TABLE_TOP,
    UP_THRESHOLD,
    _buffers,
    _update,
    _yawed_lying_rot,
    bolt_table_clearance,
    drop_penalty,
    excess_force_penalty,
    finger_contact_bonus,
    opposing_contact_bonus,
    success_bonus,
    table_contact_penalty,
    thumb_contact_bonus,
)

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

CONTROL_HZ = 30
PHYSICS_HZ = 120
GRASP_S, LIFT_S, HOLD_S = 3.0, 1.0, 1.0
GRASP_STEPS = int(GRASP_S * CONTROL_HZ)          # 90
LIFT_STEPS = int(LIFT_S * CONTROL_HZ)            # 30
HOLD_STEPS = int(HOLD_S * CONTROL_HZ)            # 30
EPISODE_STEPS = GRASP_STEPS + LIFT_STEPS + HOLD_STEPS       # 150
CONTROL_DT = 1.0 / CONTROL_HZ

LIFT_HEIGHT = 0.10                               # m, scripted world-vertical lift
APPROACH_HEIGHT = 0.05                           # m above contact, where a stage >= 1 starts

# ---------------------------------------------------------------------------
# The pre-grasp pose, solved against the ARM
# ---------------------------------------------------------------------------
#
# tools/solve_pipeline_pregrasp.py. Experiment 2's pinch, yawed 90 deg about
# WORLD Z. The yaw is free: rotating the whole (hand, bolt) pair about vertical
# preserves the thumb-to-finger gap, the opposition angle, the fingertip-to-bolt
# distance AND the table clearance -- clearance depends on the hand's tilt from
# vertical, not its heading -- and the bolt's world yaw simply rotates with it.
#
# Why it had to be yawed at all. find_pinch_pose.py searched hand orientations
# with NO ARM ATTACHED, so nothing in Experiment 2's pose knows the Adam-U
# arm's joint limits. Measured: that pose is reachable to 0.0 mm and 0.02 deg,
# but only by driving wristRoll_Right onto its -0.960 rad stop, everywhere in
# the reachable region -- the arm can HOLD the orientation with zero authority
# to leave it, in one of the three DoF the orientation stage exists to learn.
# Yawing 90 deg costs nothing and returns 0.80 of normalised wrist-roll slack.
#
#     pinch yaw                 90 deg about world z
#     wrist xy                  (0.50, -0.20)
#     wrist z                   1.0092 at first contact, 1.0592 at a stage-1 reset
#     position error            0.01 mm      orientation error  0.010 deg
#     minimum arm joint slack   0.607 at the start pose, 0.601 at contact
#     wristRoll_Right slack     0.796 (Experiment 2's pose: 0.000)
#
# Re-derive with:
#     python3 tools/solve_pipeline_pregrasp.py --yaw 90 --wrist-xy 0.50 -0.20 \
#         --seeds 8 --waist-max 1.36 --envelope

WRIST_XY = (0.50, -0.20)
WRIST_HEIGHT = 0.0592                            # m above the table top, at first contact
WRIST_Z_CONTACT = TABLE_TOP + WRIST_HEIGHT
WRIST_ROT = (-0.1821, -0.7296, 0.4661, 0.4661)   # (x, y, z, w) -- Isaac Lab 3.0 is xyzw
PINCH_YAW_DEG = 90.0
BOLT_XY = (0.5474, -0.1099)
BOLT_YAW_DEG = -136.3 + PINCH_YAW_DEG            # -46.3

# The held waist lean. ONE value has to serve the whole episode: the reset pose
# and the contact pose the policy descends to. Solving each height with a free
# waist picks a different lean for each and then neither is reachable. 1.05 rad
# maximises the worse of the two (start s_min 0.607, contact 0.601); below
# 0.70 rad the contact height is not reachable at all.
WAIST_HELD = {"waistPitch": 1.05, "waistRoll": 0.0, "waistYaw": 0.0}

# Arm poses for the two reset heights, same held waist. Stage 0 starts at
# contact (its question is the closure alone); every later stage starts
# APPROACH_HEIGHT above it.
PREGRASP_ARM_CONTACT = {
    "shoulderPitch_Right": -0.9142, "shoulderRoll_Right": -0.3996,
    "shoulderYaw_Right": +0.1797, "elbow_Right": -1.4277,
    "wristYaw_Right": +0.9023, "wristPitch_Right": -0.3292, "wristRoll_Right": -0.3582,
}
PREGRASP_ARM_START = {
    "shoulderPitch_Right": -0.8511, "shoulderRoll_Right": -0.4100,
    "shoulderYaw_Right": +0.1414, "elbow_Right": -1.6649,
    "wristYaw_Right": +0.8569, "wristPitch_Right": -0.2453, "wristRoll_Right": -0.1958,
}

RIGHT_ARM = ["shoulderPitch_Right", "shoulderRoll_Right", "shoulderYaw_Right",
             "elbow_Right", "wristYaw_Right", "wristPitch_Right", "wristRoll_Right"]
WRIST_BODY = "wristRollRight"        # r_wrist is joined to it by an identity fixed joint

# ---------------------------------------------------------------------------
# Finger synergies (Experiment 2's, unchanged)
# ---------------------------------------------------------------------------

THUMB_JOINTS = ["r_thumb_cmc_flex", "r_thumb_mcp", "r_thumb_ip"]
FINGER_JOINTS = ["r_index_finger_mcp_flex", "r_index_finger_pip", "r_index_finger_dip",
                 "r_middle_finger_mcp_flex", "r_middle_finger_pip", "r_middle_finger_dip"]
SYNERGY_JOINTS = THUMB_JOINTS + FINGER_JOINTS            # 9 driven
THUMB_ABD = -0.85                                        # thumb opposition, held
OPEN_CURL, CLOSED_CURL = 0.40, 0.60                      # fractions of each flexion limit
HELD_CURL = 0.0                                          # ring/little FLAT: curling them in this
                                                         # tilted pose brings their tips DOWN onto
                                                         # the table, which Experiment 2 measured
_LIMIT = {"r_thumb_cmc_flex": 1.291, "r_thumb_mcp": 1.570, "r_thumb_ip": 1.570,
          "r_index_finger_mcp_flex": 1.570, "r_index_finger_pip": 2.094, "r_index_finger_dip": 1.570,
          "r_middle_finger_mcp_flex": 1.570, "r_middle_finger_pip": 2.094, "r_middle_finger_dip": 1.570}
Q_OPEN = {k: round(v * OPEN_CURL, 4) for k, v in _LIMIT.items()}
Q_CLOSED = {k: round(v * CLOSED_CURL, 4) for k, v in _LIMIT.items()}

# The 11 right-hand joints the policy never touches, at fixed safe targets.
Q_FIXED_RIGHT = {
    "r_thumb_cmc_abd": THUMB_ABD,
    "r_index_finger_mcp_abd": 0.0, "r_middle_finger_mcp_abd": 0.0,
    "r_ring_finger_mcp_abd": 0.0, "r_pinky_mcp_abd": 0.0,
    "r_ring_finger_mcp_flex": round(1.570 * HELD_CURL, 4),
    "r_ring_finger_pip": round(2.094 * HELD_CURL, 4),
    "r_ring_finger_dip": round(1.570 * HELD_CURL, 4),
    "r_pinky_mcp_flex": round(1.570 * HELD_CURL, 4),
    "r_pinky_pip": round(2.094 * HELD_CURL, 4),
    "r_pinky_dip": round(1.570 * HELD_CURL, 4),
}
assert len(Q_FIXED_RIGHT) == 11, len(Q_FIXED_RIGHT)
assert len(SYNERGY_JOINTS) + len(Q_FIXED_RIGHT) == 20

TIP_BODIES = {"thumb": "r_thumb_distal", "index": "r_index_finger_distal",
              "middle": "r_middle_finger_distal", "ring": "r_ring_finger_distal",
              "pinky": "r_pinky_distal"}

# ---------------------------------------------------------------------------
# Limits and safety -- every one configurable
# ---------------------------------------------------------------------------


@configclass
class LimitsCfg:
    """Conservative starting values from the architecture spec.

    A note on two of them, because as given they contradict each other. At
    30 Hz, 0.002 m/step is 0.060 m/s against a 0.05 m/s cap, and 0.02 rad/step
    is 0.60 rad/s against a 0.5 rad/s cap -- so the VELOCITY caps bind first
    and the per-step numbers never apply. Rather than silently pick one, both
    are enforced and `effective_*` below reports what actually binds; raise the
    velocity caps to 0.06 / 0.6 to make the per-step limits the real ones.
    """

    max_translation_per_policy_step: float = 0.002        # m
    max_rotation_per_policy_step: float = 0.02            # rad
    max_wrist_linear_velocity: float = 0.05               # m/s
    max_wrist_angular_velocity: float = 0.5               # rad/s

    action_smoothing: float = 0.3          # first-order, on the finger closures
    wrist_smoothing: float = 0.5           # first-order, on the Cartesian residual

    # Joint-space safety on the 7 IK-controlled arm joints.
    joint_margin: float = 0.05             # rad kept clear of every URDF stop
    max_joint_velocity: float = 2.0        # rad/s, per-step joint-target travel cap
    max_command_lead: float = 0.15         # rad; anti-windup on the integrating IK loop
    slack_warn: float = 0.15               # s_i below this starts the soft penalty
    slack_terminate: float = 0.02          # s_min below this is a hard safety termination

    # Workspace box around the bolt's nominal position, in metres. A wrist
    # target outside it is clamped, which is cheaper and more predictable than
    # letting IK discover the arm's limits.
    workspace_radius_xy: float = 0.15
    workspace_z_min: float = 0.02          # m above the table top
    workspace_z_max: float = 0.35

    # Damped least squares, and the null-space gain that pulls the redundant
    # DoF toward mid-range.
    dls_damping: float = 0.05
    nullspace_gain: float = 0.30
    singularity_threshold: float = 1e-4    # on sqrt(det(J J^T))

    @property
    def effective_translation(self) -> float:
        return min(self.max_translation_per_policy_step,
                   self.max_wrist_linear_velocity * CONTROL_DT)

    @property
    def effective_rotation(self) -> float:
        return min(self.max_rotation_per_policy_step,
                   self.max_wrist_angular_velocity * CONTROL_DT)


# ---------------------------------------------------------------------------
# Stages: a mask over the 8 actions, plus randomisation
# ---------------------------------------------------------------------------

WRIST_ACTION_NAMES = ["wrist_dx", "wrist_dy", "wrist_dz", "wrist_drx", "wrist_dry", "wrist_drz"]
ACTION_NAMES = WRIST_ACTION_NAMES + ["thumb_synergy", "finger_synergy"]


@dataclass
class Stage:
    name: str
    live: tuple                      # indices into the 8-vector the policy may use
    start_above_contact: float       # m, where the wrist resets
    bolt_jitter_xy: float = 0.0      # m, uniform +/-
    bolt_jitter_yaw: float = 0.0     # rad, uniform +/-
    wrist_jitter_xy: float = 0.0     # m, uniform +/-
    wrist_jitter_rot: float = 0.0    # rad, uniform +/-
    success_target: float = 0.90     # advance when the evaluation reaches this
    provisional: bool = False        # True = not yet specified, placeholder shape only

    def mask(self, device) -> torch.Tensor:
        m = torch.zeros(8, device=device)
        m[list(self.live)] = 1.0
        return m


STAGES: dict[int, Stage] = {
    # Specified.
    0: Stage("finger_only", live=(6, 7), start_above_contact=0.0,
             success_target=0.90),
    1: Stage("vertical_approach", live=(2, 6, 7), start_above_contact=APPROACH_HEIGHT,
             success_target=0.90),
    # PROVISIONAL -- the specification that defined stages 0 and 1 was truncated
    # mid-sentence at stage 1 ("Keep: Wrist X an..."). These carry the ladder's
    # obvious shape so the machinery is testable end to end; the action masks
    # are a guess and the randomisation ranges are placeholders. Do not read a
    # result off them until they are replaced.
    # Stage 2 -- position correction. Defined here rather than inherited from
    # the specification, which was truncated at stage 1.
    #
    # The load-bearing decision is the BOLT JITTER, not the action mask.
    # Without randomisation the bolt is in the same place every episode, so the
    # optimal use of wrist_dx and wrist_dy is to leave them at zero and the
    # stage collapses back into stage 1 with two dead channels. Position
    # correction is only a task if there is a position error to correct.
    #
    # +/- 1 cm, because that is the scale the grasp actually cares about: the
    # fingertips start 1.6 cm from the bolt's surface inside a 5.2 cm open gap,
    # so a centimetre is a real miss but still inside the hand. The travel
    # budget is comfortable -- 90 grasp steps at the effective 1.67 mm gives
    # 150 mm, the descent spends 50, and +/- 1 cm of lateral needs about 14.
    #
    # Yaw stays fixed: turning the bolt is an ORIENTATION problem and belongs
    # to stage 3. Keeping translation and rotation in separate stages is what
    # makes a failure attributable to one of them.
    2: Stage("lateral_approach", live=(0, 1, 2, 6, 7), start_above_contact=APPROACH_HEIGHT,
             bolt_jitter_xy=0.01, success_target=0.90),
    3: Stage("orientation", live=(0, 1, 2, 3, 4, 5), start_above_contact=APPROACH_HEIGHT,
             bolt_jitter_xy=0.01, bolt_jitter_yaw=math.radians(15), provisional=True),
    4: Stage("randomised", live=tuple(range(8)), start_above_contact=APPROACH_HEIGHT,
             bolt_jitter_xy=0.02, bolt_jitter_yaw=math.radians(45),
             wrist_jitter_xy=0.01, wrist_jitter_rot=math.radians(5), provisional=True),
}
# Stage 3 and 4 must still drive the fingers; (0..5) alone would mask the
# synergies off and make the task unsolvable. Caught here rather than in a run.
STAGES[3] = Stage("orientation", live=(0, 1, 2, 3, 4, 5, 6, 7),
                  start_above_contact=APPROACH_HEIGHT, bolt_jitter_xy=0.01,
                  bolt_jitter_yaw=math.radians(15), provisional=True)


# ---------------------------------------------------------------------------
# Displacement curriculum
# ---------------------------------------------------------------------------
#
# The curriculum axis is HOW WRONG THE STARTING WRIST POSE IS, not which
# actions are unmasked. The bolt never moves.
#
# Why this replaced the action mask. Masking makes the task change
# discontinuously: the stage-0 policy had never moved a wrist, then three
# Cartesian channels opened at once with an inherited exploration std of 0.98,
# and contact went to zero on the very first iteration and never recovered
# (adam_u30_wuji_pipeline_s2, 0 % over 400 iterations). With displacement as
# the axis all 8 actions are live from level 0 and the task tightens smoothly.
#
# Why the wrist and not the bolt. The policy observes the bolt's pose RELATIVE
# to the wrist, so a displaced wrist and a displaced bolt are the same signal
# to the network -- but a fixed bolt keeps the pinch geometry, the table
# clearance and the solved pose all validated where they were solved. Moving
# the object is a different question (does it generalise in the world frame?)
# and deserves its own experiment.
#
# The ranges are MEASURED, by tools/probe_wrist_displacement.py, which sweeps
# displacements and reports the lowest fingertip's clearance above the table
# ACROSS THE WHOLE SYNERGY STROKE -- not just at the reset posture, because
# curling the fingers brings their tips down in this tilted pose. Undisplaced
# clearance is 1.08 cm over the stroke (1.35 cm at reset alone). The envelope
# is strongly asymmetric:
#
#     x, y     +/- 3 cm and beyond   clearance unaffected, arm slack ~0.60
#     z        -0.5 cm .. +3 cm      at -1 cm the thumb has 0.08 cm left;
#                                    at -2 cm it is 0.92 cm THROUGH the table
#     yaw      +/- 10 deg            clearance unaffected; limited by arm slack
#                                    (0.59 -> 0.52), not by geometry
#     roll     -5 .. +20 deg         at -10 deg the thumb is 0.47 cm under
#     pitch    -15 .. +5 deg         at +10 deg the middle finger has 0.02 cm
#
# Recovery is affordable: +/- 2 cm costs 12 of the 90 grasp steps and +/- 10 deg
# costs 10, leaving 70+ for the closure.


@dataclass
class Level:
    """One rung of the displacement curriculum. Level 0 is the fixed-pose
    diagnostic baseline -- no displacement at all."""

    name: str
    dxy: float = 0.0            # m, uniform +/- on x and y
    dz_lo: float = 0.0          # m, asymmetric: down is what hits the table
    dz_hi: float = 0.0
    dyaw: float = 0.0           # rad, uniform +/-
    droll_lo: float = 0.0       # rad, asymmetric
    droll_hi: float = 0.0
    dpitch_lo: float = 0.0
    dpitch_hi: float = 0.0
    success_target: float = 0.90

    @property
    def is_zero(self) -> bool:
        return not any((self.dxy, self.dz_lo, self.dz_hi, self.dyaw,
                        self.droll_lo, self.droll_hi, self.dpitch_lo, self.dpitch_hi))


_D = math.radians
LEVELS: dict[int, Level] = {
    0: Level("fixed_pose"),
    1: Level("displaced_small", dxy=0.005, dz_lo=-0.003, dz_hi=0.005,
             dyaw=_D(2.5), droll_lo=_D(-2), droll_hi=_D(2),
             dpitch_lo=_D(-2), dpitch_hi=_D(2)),
    2: Level("displaced_medium", dxy=0.010, dz_lo=-0.005, dz_hi=0.010,
             dyaw=_D(5), droll_lo=_D(-3), droll_hi=_D(3),
             dpitch_lo=_D(-3), dpitch_hi=_D(3)),
    3: Level("displaced_large", dxy=0.020, dz_lo=-0.005, dz_hi=0.020,
             dyaw=_D(10), droll_lo=_D(-5), droll_hi=_D(5),
             dpitch_lo=_D(-5), dpitch_hi=_D(5)),
}
LEVEL = int(os.environ.get("ADAM_U_LEVEL", "0"))
if LEVEL not in LEVELS:
    raise ValueError(f"ADAM_U_LEVEL must be one of {sorted(LEVELS)}, got {LEVEL}")

# Proximity shaping. sigma is set from the geometry: the fingertips start about
# 1.6 cm from the bolt's surface, so 2 cm puts the initial distance near the
# steep part of tanh where the gradient is usable.
# Damped pseudo-inverse of the right-arm Jacobian AT THE NOMINAL PRE-GRASP
# POSE, so a Cartesian displacement becomes arm joint angles in one multiply:
#
#     q = q_nominal + J_SHARP @ dx          dx = (dx, dy, dz, drx, dry, drz)
#
# Baked rather than solved at runtime because the displacements are small and
# the pose is fixed, so first order is enough and a reset needs no IK at all.
# Verified against full forward kinematics over the level-3 envelope: the worst
# corner (2 cm in x and y plus 10 deg of yaw) lands 1.42 mm and 0.52 deg from
# the commanded pose. That error does not matter -- the policy observes the
# bolt relative to the ACTUAL wrist, so the commanded displacement is only a
# sampling parameter, not something the policy ever sees.
# Regenerate if PREGRASP_ARM_CONTACT or the held waist changes.
J_SHARP = (
    (-3.331483, -1.038766, +1.437886, +0.026061, +0.003865, -0.052771),
    (-0.103434, +1.866384, -0.781030, +0.120791, +0.024376, -0.248876),
    (-0.716273, +1.249422, -0.142269, -0.148744, -0.027613, +0.314144),
    (+2.918059, +0.164837, -4.983479, -0.020740, +0.003640, +0.037611),
    (-0.859473, +1.444512, -1.445427, -0.666762, -0.665387, -0.044996),
    (-0.555831, +1.850984, +2.014353, -0.546271, +0.268106, -0.646784),
    (+0.917511, +0.056436, +2.936675, -0.149386, +0.935722, +0.247245),
)

# ---------------------------------------------------------------------------
# Table-collision coverage beyond the fingertips
# ---------------------------------------------------------------------------
#
# Until now only the five FINGERTIPS carried contact sensors, so the palm, the
# hand base, the wrist and the forearm could rest on the table completely
# undetected -- and still collect the +20 success. That is a hole in the
# success test, not just a missing metric.
#
# `r_wrist` is joined to `wristRollRight` by a fixed joint with an identity
# origin, so the USD import merges them: ONE body carries the wrist, the hand
# base and the palm. The forearm is wristPitchRight/wristYawRight, and the
# upper arm ends at elbowRight.
#
# These get a LOWER threshold than the fingers. A fingertip resting lightly on
# the table is normal -- every pose that grasps does it -- and only ploughing
# into it (20-44 N measured) is a collision. A palm or a forearm on the table
# is never correct at any force.
STRUCT_BODIES = {
    "palm": "wristRollRight",        # wrist + hand base + palm, merged at import
    "forearm": "wristPitchRight",
    "upperarm": "elbowRight",
}
# Intermediate finger links, which the fingertip sensors also miss. A finger
# lying along the table on its middle phalanx is a collision the tip never sees.
FINGER_LINK_BODIES = {
    "thumb_mid": "r_thumb_middle", "index_mid": "r_index_finger_middle",
    "middle_mid": "r_middle_finger_middle",
    "thumb_prox": "r_thumb_proximal", "index_prox": "r_index_finger_proximal",
    "middle_prox": "r_middle_finger_proximal",
}
# ---------------------------------------------------------------------------
# Early grasp completion
# ---------------------------------------------------------------------------
#
# The 90-step grasp phase is a TIMEOUT, not a waiting period. Measured on the
# level-3 policy: stable opposing contact arrives at step 15 of 90 in 100 % of
# episodes, and the wrist then travels a further 116.7 mm horizontally --
# dragging the bolt across the table for the remaining 75 steps.
#
# Nothing in the reward stopped it. The contact bonuses are paid once, at step
# 15. `approach` telescopes, so wandering earns nothing. `success` is +20
# wherever it lifts from. And `action_rate` penalises the CHANGE in action, not
# motion, so a constant command held for 75 steps costs exactly zero and
# travels 12 cm. Sustained drift was free.
#
# The fix is structural rather than a penalty: once the grasp is established,
# freeze the wrist, let the fingers settle, and lift. A wrist-motion penalty
# would have fought a symptom of an unconstrained phase length.
SETTLE_STEPS = 10         # finger-only steps after the wrist latches
GRIP_MIN_N = 4.0          # N, per side: a real grip, not merely touching
GRASP_REGION = 0.05       # m, bolt-to-pinch-centre for "inside the grasp"
REL_VEL_MAX = 0.10        # m/s, bolt must be moving with the hand, not sliding

STRUCT_HIT_N = 5.0        # N, palm/forearm/upper arm touching the table at all
LINK_HIT_N = 15.0         # N, same bar as the fingertips

PROX_SIGMA = 0.02
PROX_WEIGHT = 2.0
BOLT_RADIUS = 0.0125         # m, the rod's radius
BOLT_HALF_LEN = 0.0175       # m, half its length along its own axis

STAGE = int(os.environ.get("ADAM_U_STAGE", "0"))
if STAGE not in STAGES:
    raise ValueError(f"ADAM_U_STAGE must be one of {sorted(STAGES)}, got {STAGE}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metrics(env) -> dict:
    m = getattr(env, "_pipeline_metrics", None)
    if m is None:
        n, d = env.num_envs, env.device
        m = dict(
            s_min=torch.ones(n, device=d),
            s_argmin=torch.zeros(n, dtype=torch.long, device=d),
            arm_slack=torch.ones(n, 7, device=d),
            clipped=torch.zeros(n, device=d),          # arm targets clipped, this step
            ik_failed=torch.zeros(n, device=d),
            singular=torch.zeros(n, device=d),
            wrist_track_err=torch.zeros(n, device=d),  # m, |target - actual| position
            wrist_track_err_rot=torch.zeros(n, device=d),
            limit_terminations=torch.zeros(n, device=d),
        )
        env._pipeline_metrics = m
    return m


PUBLISH_EVERY = 20          # control steps between metric publishes


def _publish(env, force: bool = False):
    """Put the safety metrics where rsl-rl's logger will pick them up.

    THROTTLED, and this matters more than it looks. Every line below ends in
    `.item()`, which is a GPU->CPU synchronisation that stalls the pipeline.
    Called from apply_actions this ran once per PHYSICS step -- 15 syncs x 4
    substeps = 60 stalls per environment step, for numbers that are only ever
    read by TensorBoard. It is now called once per CONTROL step and only every
    PUBLISH_EVERY of those.
    """
    m = _metrics(env)
    if not force and int(env.common_step_counter) % PUBLISH_EVERY:
        return
    log = env.extras.setdefault("log", {}) if isinstance(env.extras, dict) else None
    if log is None:
        return
    log["Safety/min_joint_slack"] = m["s_min"].mean().item()
    log["Safety/min_joint_slack_worst"] = m["s_min"].min().item()
    log["Safety/ik_clipped_targets"] = m["clipped"].mean().item()
    log["Safety/ik_failures"] = m["ik_failed"].mean().item()
    log["Safety/ik_singular"] = m["singular"].mean().item()
    log["Safety/wrist_tracking_error_m"] = m["wrist_track_err"].mean().item()
    log["Safety/wrist_tracking_error_rad"] = m["wrist_track_err_rot"].mean().item()
    log["Safety/limit_terminations"] = m["limit_terminations"].mean().item()
    # Which joint is responsible, as a histogram over the 7 arm joints.
    for i, name in enumerate(RIGHT_ARM):
        log[f"Safety/slack_{name}"] = m["arm_slack"][:, i].mean().item()


def _latch(env) -> dict:
    """Per-env grasp-completion state."""
    b = _buffers(env)
    if b.get("grasp_end") is None:
        n, d = env.num_envs, env.device
        b["grasp_end"] = torch.full((n,), float(GRASP_STEPS), device=d)
        b["ready_at"] = torch.full((n,), -1.0, device=d)
        b["latched"] = torch.zeros(n, dtype=torch.bool, device=d)
        b["last_grasp_end"] = torch.full((n,), float(GRASP_STEPS), device=d)
        # Hand the per-env grasp boundary to the shared bookkeeping in
        # wuji_synergy_grasp_env_cfg._phases, which otherwise assumes a fixed
        # 90-step grasp phase for every environment.
        env._grasp_boundary = lambda: b["grasp_end"]
    return b


def grasp_ready(env) -> torch.Tensor:
    """(N,) bool: the grasp is established and the wrist has no more work.

        stable opposing contact  (thumb AND one opposing finger, >= STABLE_STEPS)
        AND no fingertip over F_MAX
        AND the bolt is inside the grasp region
        AND the bolt is moving WITH the hand, not sliding through it

    The stability requirement is what stops a one-step contact spike from
    latching the wrist -- exploration noise brushes the bolt often enough that
    a raw contact flag would fire almost immediately.
    """
    b = _update(env)
    stable_opposing = b["thumb_stable"] & b["finger_stable"]
    force_ok = b["tip_force"].max(dim=1).values < F_MAX
    # A LOWER force bound as well as the upper one. The contact flag turns on at
    # 0.1 N, which is touching, not gripping -- and latching on touch truncated
    # the closure: the level-3 policy closes gradually over 90 steps, so cutting
    # it off at step ~18 left the fingers at 0.28 instead of 0.62 and the bolt
    # fell back to the table in 28.9 % of episodes (success 96.5 % -> 70.5 %).
    gripping = ((b["tip_force"][:, 0] > GRIP_MIN_N)
                & (torch.maximum(b["tip_force"][:, 1], b["tip_force"][:, 2]) > GRIP_MIN_N))
    d = fingertip_distances(env)
    inside = (0.5 * (d[:, 0] + torch.minimum(d[:, 1], d[:, 2]))) < GRASP_REGION
    robot = env.scene["robot"]
    i = _ids(env)["wrist"]
    rel = _t(env.scene["object"].data.root_lin_vel_w) - _t(robot.data.body_link_lin_vel_w)[:, i, :]
    settled = torch.norm(rel, dim=1) < REL_VEL_MAX
    return stable_opposing & gripping & force_ok & inside & settled


def update_latch(env) -> torch.Tensor:
    """Advance the latch state once per step; returns each env's grasp-end step.

    Named `grasp_end`, not `lift_start`: the shared bookkeeping in
    wuji_synergy_grasp_env_cfg already owns a `lift_start` key holding
    Experiment 2's prismatic joint reset position.
    """
    b = _latch(env)
    if b.get("_latch_step") == int(env.common_step_counter):
        return b["grasp_end"]
    b["_latch_step"] = int(env.common_step_counter)
    t = env.episode_length_buf.float()
    ready = grasp_ready(env)
    first = ready & (b["ready_at"] < 0)
    b["ready_at"] = torch.where(first, t, b["ready_at"])
    # Latch only after the fingers have had SETTLE_STEPS to close on the bolt.
    due = (b["ready_at"] >= 0) & (t >= b["ready_at"] + SETTLE_STEPS) & ~b["latched"]
    b["grasp_end"] = torch.where(due, t, b["grasp_end"])
    b["latched"] |= due
    return b["grasp_end"]


def phase_of(env) -> torch.Tensor:
    """0 grasp, 1 lift, 2 hold -- PER ENV, because the lift can start early."""
    t = env.episode_length_buf.float()
    ls = update_latch(env)
    return torch.where(t < ls, 0, torch.where(t < ls + LIFT_STEPS, 1, 2)).long()


# ---------------------------------------------------------------------------
# The action term: 8 policy values -> 27 commanded targets, plus every held joint
# ---------------------------------------------------------------------------


class PipelineAction(ActionTerm):
    """Masked Cartesian wrist residuals + two finger synergies.

    One term rather than two so that the complete joint target tensor is built
    and written in one place, on every control step. That is the only way to
    guarantee no held joint is ever left at a zero drive target.
    """

    cfg: "PipelineActionCfg"

    def __init__(self, cfg: "PipelineActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        robot = self._asset
        lim = cfg.limits
        self._lim = lim
        self._level = LEVELS[cfg.level]
        # All 8 actions live at every level. The curriculum is the DISPLACEMENT
        # magnitude, not which channels are switched on -- masking is what made
        # stage 2 fail discontinuously.
        self._mask = torch.ones(1, 8, device=self.device)
        self._nominal_pos = torch.tensor(
            [WRIST_XY[0], WRIST_XY[1], WRIST_Z_CONTACT], device=self.device)
        self._nominal_quat = torch.tensor(WRIST_ROT, device=self.device)

        # The root must be fixed and unrotated for world-frame IK to be the
        # same as root-frame IK. Asserted rather than assumed.
        if not robot.is_fixed_base:
            raise ValueError("the pipeline env expects a fixed-base robot")

        self._arm_ids, arm_names = robot.find_joints(RIGHT_ARM, preserve_order=True)
        assert arm_names == RIGHT_ARM, arm_names
        self._thumb_ids, _ = robot.find_joints(THUMB_JOINTS, preserve_order=True)
        self._finger_ids, _ = robot.find_joints(FINGER_JOINTS, preserve_order=True)

        body_ids, _ = robot.find_bodies(WRIST_BODY)
        if len(body_ids) != 1:
            raise ValueError(f"{WRIST_BODY} matched {len(body_ids)} bodies")
        self._body_idx = body_ids[0]
        self._jacobi_body_idx = self._body_idx - (1 if robot.is_fixed_base else 0)
        self._jacobi_cols = [j + robot.num_base_dofs for j in self._arm_ids]

        # Arm joint limits, inset by the safety margin.
        jl = _t(robot.data.joint_pos_limits)[0, self._arm_ids, :]
        self._q_lo_hard, self._q_hi_hard = jl[:, 0].clone(), jl[:, 1].clone()
        self._q_lo = self._q_lo_hard + lim.joint_margin
        self._q_hi = self._q_hi_hard - lim.joint_margin
        self._q_mid = 0.5 * (self._q_lo_hard + self._q_hi_hard)
        self._q_range = (self._q_hi_hard - self._q_lo_hard).clamp(min=1e-6)

        # Synergy ends, per driven joint, in the robot's joint order.
        def resolve(table, ids):
            names = [robot.joint_names[i] for i in ids]
            return torch.tensor([table[n] for n in names], device=self.device)

        self._open_t = resolve(cfg.q_open, self._thumb_ids)
        self._closed_t = resolve(cfg.q_closed, self._thumb_ids)
        self._open_f = resolve(cfg.q_open, self._finger_ids)
        self._closed_f = resolve(cfg.q_closed, self._finger_ids)

        self._raw = torch.zeros(self.num_envs, 8, device=self.device)
        self._processed = torch.zeros(self.num_envs, 8, device=self.device)
        self._closure = torch.zeros(self.num_envs, 2, device=self.device)
        self._wrist_cmd = torch.zeros(self.num_envs, 6, device=self.device)
        # The commanded wrist pose. Residuals accumulate on the COMMAND, not on
        # the measured pose, so "wrist tracking error" is a real measurement of
        # how far the arm is behind what was asked -- and so the scripted lift
        # has a well-defined target to ramp.
        self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_quat[:, 3] = 1.0
        self._base_targets: torch.Tensor | None = None
        self._nominal_xy = torch.tensor(WRIST_XY, device=self.device)
        # Last commanded arm joint state, for logging only.
        #
        # A note on what did NOT work, so it is not retried. The loop below is
        # proportional -- q_target = q_measured + dq -- which under gravity
        # settles at a permanent offset rather than closing it: measured 7.4 mm
        # of steady-state wrist error at the stage-0 pose, 10.1 mm at stage 1.
        # The obvious fix, integrating on the previous COMMAND instead of the
        # measurement, was tried and is MEASURABLY UNSTABLE here: stage 0 went
        # from 7.4 mm to 32.4 mm and the wrist drifted upward and sideways
        # instead of settling, because the command accumulates an error the
        # measurement has not caught up to yet and overshoots. The droop is a
        # drive-stiffness problem and is fixed where it lives, in the actuator
        # gains below.
        self._q_cmd = torch.zeros(self.num_envs, 7, device=self.device)
        self._q_cmd_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # -- properties ---------------------------------------------------------

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

    # -- helpers ------------------------------------------------------------

    def _ee_pose(self):
        """Wrist body pose, world axes, origin-relative so it is env-independent."""
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
        # Seed the commanded pose ANALYTICALLY from the nominal pose plus the
        # displacement this episode was given, rather than by reading the wrist
        # out of the simulator. The reset event writes joint STATES and the body
        # poses are not necessarily refreshed before this runs, so reading them
        # here can seed the target with the previous episode's final pose --
        # which would make the IK yank the arm across the table on step 0.
        d = getattr(self._env, "_wrist_displacement", None)
        if d is None:
            self._target_pos[env_ids] = self._nominal_pos
            self._target_quat[env_ids] = self._nominal_quat
        else:
            self._target_pos[env_ids] = self._nominal_pos + d[env_ids, 0:3]
            ang = torch.norm(d[env_ids, 3:6], dim=1)
            axis = d[env_ids, 3:6] / ang.unsqueeze(1).clamp(min=1e-9)
            self._target_quat[env_ids] = math_utils.quat_mul(
                math_utils.quat_from_angle_axis(ang, axis),
                self._nominal_quat.unsqueeze(0).expand(len(self._target_quat[env_ids]), 4))
        self._q_cmd_valid[env_ids] = False        # re-seed from the measured pose

    # -- the two halves of a control step ----------------------------------

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions.clamp(-1.0, 1.0)
        # The mask zeroes the raw action AND therefore its scale: a masked
        # channel contributes exactly nothing, at any magnitude.
        masked = self._raw * self._mask
        self._processed[:] = masked

        lim = self._lim
        # Once the grasp is established the wrist is FROZEN: its six actions are
        # zeroed and the accumulated residual command is cleared, so the last
        # policy command cannot keep integrating. Fingers stay live for
        # SETTLE_STEPS, then the phase machine moves to the scripted lift.
        b = _latch(self._env)
        frozen = b["latched"].unsqueeze(1)
        masked = torch.cat([torch.where(frozen, torch.zeros_like(masked[:, 0:6]), masked[:, 0:6]),
                            masked[:, 6:8]], dim=1)
        self._processed[:] = masked
        self._wrist_cmd = torch.where(frozen, torch.zeros_like(self._wrist_cmd), self._wrist_cmd)

        # Cartesian residual, first-order smoothed, then limited by BOTH the
        # per-step cap and the velocity cap (see LimitsCfg for why both).
        cmd = torch.cat([masked[:, 0:3] * lim.effective_translation,
                         masked[:, 3:6] * lim.effective_rotation], dim=1)
        self._wrist_cmd += lim.wrist_smoothing * (cmd - self._wrist_cmd)

        # Finger closures in [0, 1], frozen outside the grasp phase.
        c = (masked[:, 6:8] + 1.0) / 2.0
        phase = phase_of(self._env)
        grasping = (phase == 0).unsqueeze(1)
        new = self._closure + lim.action_smoothing * (c - self._closure)
        self._closure[:] = torch.where(grasping, new, self._closure)

        # ---- advance the commanded wrist pose -----------------------------
        # This MUST happen here, in process_actions, and not in apply_actions.
        # Isaac Lab calls process_actions once per CONTROL step and
        # apply_actions once per PHYSICS step -- four times as often at
        # decimation 4. Integrating the residual there multiplies every
        # Cartesian rate by the decimation: measured, a commanded descent ran
        # 51 mm in the first 10 control steps against a 1.67 mm/step limit
        # (16.7 mm), and the scripted lift rose 238 mm instead of 100 mm.
        # apply_actions now only solves IK toward this target and writes it.
        d_pos = self._wrist_cmd[:, 0:3].clone()
        d_rot = self._wrist_cmd[:, 3:6].clone()
        lifting = (phase == 1).unsqueeze(1)
        holding = (phase == 2).unsqueeze(1)
        # Scripted lift: a WORLD-vertical ramp on the target. No joint axis is
        # involved, so it cannot rotate with a parent the way the cut-out
        # hand's prismatic lift joint did (that once dragged the bolt 7 cm
        # sideways while reporting a correct 10 cm rise).
        lift_step = torch.zeros_like(d_pos)
        lift_step[:, 2] = LIFT_HEIGHT / LIFT_STEPS
        d_pos = torch.where(lifting, lift_step, d_pos)
        d_rot = torch.where(lifting, torch.zeros_like(d_rot), d_rot)
        d_pos = torch.where(holding, torch.zeros_like(d_pos), d_pos)
        d_rot = torch.where(holding, torch.zeros_like(d_rot), d_rot)

        # Re-seed the target from where the wrist ACTUALLY is on the first step
        # of the lift. Without this the lift inherits whatever lateral tracking
        # error the grasp phase accumulated, and the arm pays it off by sliding
        # sideways just as it starts to rise. Measured on the level-2 policy:
        # 12.0 mm of target-to-actual error at the end of the grasp phase, then
        # 11 mm of horizontal wrist travel over the next ten steps, dragging the
        # bolt 20 mm across the table. A scripted VERTICAL lift should be
        # vertical; this makes it so, and it is a structural fix rather than a
        # reward term fighting a symptom.
        #
        # The same family as the bug Experiment 2 hit, where the prismatic lift
        # joint's axis rotated with its parent and dragged the bolt 7 cm while
        # reporting a correct 10 cm rise.
        entering_lift = (self._env.episode_length_buf.float()
                         == _latch(self._env)["lift_start"]).unsqueeze(1)
        if bool(entering_lift.any()):
            cur_pos, cur_quat = self._ee_pose()
            self._target_pos = torch.where(entering_lift, cur_pos, self._target_pos)
            self._target_quat = torch.where(entering_lift, cur_quat, self._target_quat)

        self._target_pos += d_pos
        # Small axis-angle increment, applied in the world frame.
        ang = torch.norm(d_rot, dim=1, keepdim=True)
        axis = d_rot / ang.clamp(min=1e-9)
        dq = math_utils.quat_from_angle_axis(ang.squeeze(1), axis)
        self._target_quat[:] = math_utils.quat_mul(dq, self._target_quat)

        # Workspace clamp, on the target rather than after the fact.
        centre = self._nominal_xy.unsqueeze(0)
        off = self._target_pos[:, :2] - centre
        r = torch.norm(off, dim=1, keepdim=True)
        scale = (lim.workspace_radius_xy / r.clamp(min=1e-9)).clamp(max=1.0)
        self._target_pos[:, :2] = centre + off * scale
        self._target_pos[:, 2] = self._target_pos[:, 2].clamp(
            TABLE_TOP + lim.workspace_z_min, TABLE_TOP + lim.workspace_z_max)

        _publish(self._env)          # control rate, throttled -- see _publish

    def apply_actions(self):
        robot = self._asset
        lim = self._lim
        m = _metrics(self._env)
        n = self.num_envs

        if self._base_targets is None:
            # Every joint's held target, from the robot's own defaults. Built
            # once; the 27 commanded columns are overwritten each step.
            self._base_targets = _t(robot.data.default_joint_pos)[0:1].clone()

        # ---- 1. damped least-squares IK toward the commanded pose ---------
        # The target itself is advanced in process_actions, at CONTROL rate.
        # This runs at physics rate and only tracks it.
        cur_pos, cur_quat = self._ee_pose()
        e_pos = self._target_pos - cur_pos
        q_err = math_utils.quat_mul(self._target_quat, math_utils.quat_inv(cur_quat))
        e_rot = math_utils.axis_angle_from_quat(q_err)
        err = torch.cat([e_pos, e_rot], dim=1).unsqueeze(2)              # (n, 6, 1)
        m["wrist_track_err"] = torch.norm(e_pos, dim=1)
        m["wrist_track_err_rot"] = torch.norm(e_rot, dim=1)

        J = _t(robot.data.body_link_jacobian_w)[:, self._jacobi_body_idx, :, :]
        J = J[:, :, self._jacobi_cols]                                    # (n, 6, 7)
        JT = J.transpose(1, 2)
        JJT = J @ JT
        eye6 = torch.eye(6, device=self.device).expand(n, 6, 6)
        damped = JJT + (lim.dls_damping ** 2) * eye6

        q = _t(robot.data.joint_pos)[:, self._arm_ids]
        # Seed the commanded state from the measurement on the first step after
        # a reset, then integrate.
        if not bool(self._q_cmd_valid.all()):
            fresh = ~self._q_cmd_valid
            self._q_cmd[fresh] = q[fresh]
            self._q_cmd_valid |= True
        # Null-space: move the redundant DoF toward mid-range, weighted by how
        # close each joint already is to a stop, so clearance is bought where
        # it is scarce. The policy is not left solely responsible for this.
        slack = 2.0 * torch.minimum(q - self._q_lo_hard, self._q_hi_hard - q) / self._q_range
        urgency = (1.0 - slack.clamp(0.0, 1.0)) ** 2
        dq_null = lim.nullspace_gain * urgency * (self._q_mid - q) / self._q_range
        # dq = J# e + (I - J# J) dq_null, rearranged as
        #      dq = J# (e - J dq_null) + dq_null
        # so one batched solve with a single right-hand side does the whole
        # thing, instead of forming an explicit 6x6 inverse and two 7x7 matmuls.
        rhs = err - (J @ dq_null.unsqueeze(2))
        dq = (JT @ torch.linalg.solve(damped, rhs)).squeeze(2) + dq_null

        # ---- 2. deterministic safety on the joint targets -----------------
        bad = ~torch.isfinite(dq).all(dim=1)
        dq = torch.nan_to_num(dq, nan=0.0, posinf=0.0, neginf=0.0)
        # Per-step joint travel cap (a velocity limit in joint space).
        dq = dq.clamp(-lim.max_joint_velocity * CONTROL_DT, lim.max_joint_velocity * CONTROL_DT)
        # Integrate on the COMMAND, not on the measurement (see _q_cmd).
        q_des = q + dq
        q_clipped = q_des.clamp(self._q_lo, self._q_hi)
        self._q_cmd[:] = q_clipped
        m["clipped"] = (q_clipped != q_des).float().sum(dim=1)
        # Singularity: the manipulability measure, cheap for a 6x6.
        manip = torch.sqrt(torch.linalg.det(JJT).clamp(min=0.0))
        m["singular"] = (manip < lim.singularity_threshold).float()
        m["ik_failed"] = (bad | (manip < lim.singularity_threshold)).float()

        slack_des = 2.0 * torch.minimum(q_clipped - self._q_lo_hard,
                                        self._q_hi_hard - q_clipped) / self._q_range
        m["arm_slack"] = slack_des
        m["s_min"], m["s_argmin"] = slack_des.min(dim=1)

        # ---- 3. write EVERY joint target ----------------------------------
        targets = self._base_targets.expand(n, -1).clone()
        targets[:, self._arm_ids] = q_clipped
        c_t, c_f = self._closure[:, 0:1], self._closure[:, 1:2]
        targets[:, self._thumb_ids] = self._open_t + c_t * (self._closed_t - self._open_t)
        targets[:, self._finger_ids] = self._open_f + c_f * (self._closed_f - self._open_f)
        robot.set_joint_position_target(targets)


@configclass
class PipelineActionCfg(ActionTermCfg):
    class_type: type = PipelineAction
    asset_name: str = "robot"
    limits: LimitsCfg = LimitsCfg()
    level: int = LEVEL
    q_open: dict = field(default_factory=lambda: dict(Q_OPEN))
    q_closed: dict = field(default_factory=lambda: dict(Q_CLOSED))


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def _ids(env) -> dict:
    """Body and joint indices, resolved ONCE.

    `find_bodies` / `find_joints` run a regex match over every name in the
    articulation, in Python. Called from an observation term they run on every
    environment step, for indices that never change.
    """
    c = getattr(env, "_pipeline_ids", None)
    if c is None:
        robot = env.scene["robot"]
        wrist, _ = robot.find_bodies(WRIST_BODY)
        arm, _ = robot.find_joints(RIGHT_ARM, preserve_order=True)
        jl = _t(robot.data.joint_pos_limits)[:, arm, :]
        c = dict(wrist=wrist[0], arm=arm, arm_lo=jl[..., 0], arm_hi=jl[..., 1])
        env._pipeline_ids = c
    return c


def _wrist_pose(env):
    i = _ids(env)["wrist"]
    robot = env.scene["robot"]
    pos = _t(robot.data.body_link_pos_w)[:, i, :] - env.scene.env_origins
    quat = _t(robot.data.body_link_quat_w)[:, i, :]
    return pos, quat, i


def bolt_pos_in_wrist(env) -> torch.Tensor:
    """Bolt position relative to the wrist, expressed in the WRIST's own axes (3)."""
    pos, quat, _ = _wrist_pose(env)
    bolt = _t(env.scene["object"].data.root_pos_w) - env.scene.env_origins
    return math_utils.quat_apply_inverse(quat, bolt - pos)


def bolt_rot_in_wrist(env) -> torch.Tensor:
    """Bolt orientation relative to the wrist as a 6D rotation (6).

    The first two columns of R_wrist^T R_bolt. Continuous, unlike a quaternion,
    whose sign ambiguity the network would otherwise have to learn around.
    """
    _, quat, _ = _wrist_pose(env)
    bolt_q = _t(env.scene["object"].data.root_quat_w)
    rel = math_utils.quat_mul(math_utils.quat_inv(quat), bolt_q)
    R = math_utils.matrix_from_quat(rel)
    return torch.cat([R[:, :, 0], R[:, :, 1]], dim=1)


def wrist_lin_vel(env) -> torch.Tensor:
    return _t(env.scene["robot"].data.body_link_lin_vel_w)[:, _ids(env)["wrist"], :]


def wrist_ang_vel(env) -> torch.Tensor:
    return _t(env.scene["robot"].data.body_link_ang_vel_w)[:, _ids(env)["wrist"], :]


def tip_forces_obs(env) -> torch.Tensor:
    """Thumb / index / middle fingertip force on the bolt, scaled by F_MAX (3)."""
    return (_update(env)["tip_force"] / F_MAX).clamp(-5.0, 5.0)


def contact_flags(env) -> torch.Tensor:
    return (_update(env)["streak"] > 0).float()


def episode_time(env) -> torch.Tensor:
    return (env.episode_length_buf.float() / EPISODE_STEPS).unsqueeze(1)


def arm_q_normalized(env) -> torch.Tensor:
    """The 7 right-arm joint positions, normalised to [-1, 1] over their limits.

        q_norm = 2 (q - q_min) / (q_max - q_min) - 1

    Seven values rather than one minimum-slack scalar: a scalar says a joint is
    near a limit without saying which joint, which end, or which Cartesian
    correction would move away from it.
    """
    c = _ids(env)
    q = _t(env.scene["robot"].data.joint_pos)[:, c["arm"]]
    lo, hi = c["arm_lo"], c["arm_hi"]
    return (2.0 * (q - lo) / (hi - lo).clamp(min=1e-6) - 1.0).clamp(-2.0, 2.0)


# ---------------------------------------------------------------------------
# Rewards, terminations
# ---------------------------------------------------------------------------


TIP_OFFSETS = {"thumb": (0.0, 0.0, -0.02978),
               "index": (0.0, 0.0, -0.02475),
               "middle": (0.0, 0.0, -0.02475)}


def _tip_ids(env) -> dict:
    c = getattr(env, "_pipeline_tip_ids", None)
    if c is None:
        robot = env.scene["robot"]
        c = {}
        for k in ("thumb", "index", "middle"):
            ids, _ = robot.find_bodies(TIP_BODIES[k])
            c[k] = ids[0]
        env._pipeline_tip_ids = c
    return c


def fingertip_distances(env) -> torch.Tensor:
    """(N, 3) thumb / index / middle fingertip distance to the BOLT'S SURFACE.

    The bolt is a rod, so it is modelled as a capsule -- distance to its axis
    segment minus its radius -- rather than as the bounding sphere the
    table-clearance test uses. At a 1.6 cm starting gap a bounding sphere would
    be too loose to shape anything: the sphere's radius is 1.75 cm against a
    1.25 cm real radius, so it would report contact while the tips were still
    5 mm away.

    Computed from forward kinematics and the known bolt pose, so these are
    informative BEFORE contact, which is the whole point -- the force and
    contact observations only say anything once the fingers have arrived.
    """
    robot = env.scene["robot"]
    ids = _tip_ids(env)
    bolt_p = _t(env.scene["object"].data.root_pos_w)
    bolt_q = _t(env.scene["object"].data.root_quat_w)
    axis = math_utils.quat_apply(bolt_q, torch.tensor([0.0, 0.0, 1.0], device=env.device)
                                 .expand(env.num_envs, 3))
    out = []
    for k in ("thumb", "index", "middle"):
        i = ids[k]
        off = torch.tensor(TIP_OFFSETS[k], device=env.device).expand(env.num_envs, 3)
        tip = (_t(robot.data.body_link_pos_w)[:, i, :]
               + math_utils.quat_apply(_t(robot.data.body_link_quat_w)[:, i, :], off))
        v = tip - bolt_p
        t = (v * axis).sum(dim=1, keepdim=True).clamp(-BOLT_HALF_LEN, BOLT_HALF_LEN)
        out.append((torch.norm(v - t * axis, dim=1) - BOLT_RADIUS).clamp(min=0.0))
    return torch.stack(out, dim=1)


def _potential(env) -> torch.Tensor:
    """Grasp proximity potential, in [0, 1].

        d_opposing = min(d_index, d_middle)
        Phi = 0.5 [ (1 - tanh(d_thumb/sigma)) + (1 - tanh(d_opposing/sigma)) ]
    """
    d = fingertip_distances(env)
    d_opp = torch.minimum(d[:, 1], d[:, 2])
    return 0.5 * ((1.0 - torch.tanh(d[:, 0] / PROX_SIGMA))
                  + (1.0 - torch.tanh(d_opp / PROX_SIGMA)))


def proximity_progress(env) -> torch.Tensor:
    """PROGRESS in the potential, not its value: 2 (Phi_t - Phi_{t-1}).

    This distinction is the whole design. Rewarding Phi itself pays a policy
    for LOITERING near the bolt, and over 90 grasp steps that pays far more
    than the +20 for actually succeeding -- the exact shape of failure this
    repo has already hit twice (Experiment 2's two-finger run banked the full
    contact bonus while sitting open; pipeline stage 2 then collapsed to
    holding a constant action). Progress telescopes: the total over an episode
    is 2 (Phi_end - Phi_start) regardless of path, so there is nothing to farm.
    Moving closer pays, moving away costs, standing still is exactly zero.

    Paid only during GRASP -- the lift and hold ignore the policy anyway.
    """
    b = _buffers(env)
    phi = _potential(env)
    prev = b.get("prev_phi")
    if prev is None:
        prev = phi.clone()
        b["prev_phi"] = prev
    grasping = phase_of(env) == 0
    r = torch.where(grasping, phi - prev, torch.zeros_like(phi))
    b["prev_phi"] = phi.clone()
    return r


def _table_force(env, sensor: str) -> torch.Tensor:
    """Force this body is exerting on the TABLE, from the PhysX filter [N]."""
    s = env.scene[sensor]
    return torch.norm(_t(s.data.force_matrix_w)[:, 0, 0, :], dim=1)


def structural_table_contact(env) -> torch.Tensor:
    """(N,) bool: palm, forearm or upper arm touching the table.

    Never correct at any force, so the threshold is 5 N rather than the
    fingers' 15 N. An episode that trips this cannot succeed -- see
    `_update_struct`, which is what stops a policy resting its palm on the
    table and still collecting the +20.
    """
    b = _buffers(env)
    if b.get("_struct_step") == int(env.common_step_counter):
        return b["struct_hit"]
    b["_struct_step"] = int(env.common_step_counter)
    if b.get("struct_hit") is None:
        b["struct_hit"] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        b["struct_which"] = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        b["struct_force"] = torch.zeros(env.num_envs, device=env.device)
        b["link_hit"] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # `_once` indexes this directly, so it has to exist before the first
        # reward evaluation, not just before the first reset.
        b["paid_struct"] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    names = list(STRUCT_BODIES)
    f = torch.stack([_table_force(env, f"cs_{k}") for k in names], dim=1)
    now = f.max(dim=1)
    hit = now.values > STRUCT_HIT_N
    fresh = hit & ~b["struct_hit"]
    b["struct_which"] = torch.where(fresh, now.indices, b["struct_which"])
    b["struct_force"] = torch.maximum(b["struct_force"], now.values)
    b["struct_hit"] |= hit
    # Intermediate finger links, folded into the existing finger-table flag.
    fl = torch.stack([_table_force(env, f"cs_{k}") for k in FINGER_LINK_BODIES], dim=1)
    b["link_hit"] |= fl.max(dim=1).values > LINK_HIT_N
    b["table_hit"] |= b["link_hit"]
    return b["struct_hit"]


def structural_contact_penalty(env) -> torch.Tensor:
    b = _buffers(env)
    structural_table_contact(env)
    return _once(b, "paid_struct", b["struct_hit"])


def structural_contact_termination(env) -> torch.Tensor:
    """Immediate failure. The episode ends and no terminal success is paid."""
    return structural_table_contact(env)


def joint_limit_penalty(env) -> torch.Tensor:
    """Soft, quadratic, and zero in the safe region.

    Deliberately small: a joint-limit penalty that can outweigh the +20 success
    teaches the policy to sit still in the middle of its range. This one only
    has gradient once a joint is inside `slack_warn` of a stop.
    """
    m = _metrics(env)
    warn = env.cfg.actions.pipeline.limits.slack_warn
    deficit = (warn - m["arm_slack"]).clamp(min=0.0) / warn
    return (deficit ** 2).sum(dim=1)


def joint_limit_termination(env) -> torch.Tensor:
    """Hard safety boundary. Not the only defence -- the soft penalty, the
    null-space term, the target clipping and the workspace box all act first."""
    m = _metrics(env)
    hit = m["s_min"] < env.cfg.actions.pipeline.limits.slack_terminate
    m["limit_terminations"] = hit.float()
    return hit


def ik_failure_termination(env) -> torch.Tensor:
    """A diverged or singular solve. Without this the arm can sit at a
    singularity producing NaN-free nonsense that looks like a held grasp."""
    return _metrics(env)["ik_failed"] > 0.5


def bolt_off_table(env) -> torch.Tensor:
    return _t(env.scene["object"].data.root_pos_w)[:, 2] < TABLE_TOP - 0.075


def reset_pipeline_buffers(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Experiment 2's buffer reset, minus the prismatic lift joint it read.

    There is no `wrist_lift` joint on the whole robot -- the lift is Cartesian
    here -- so `lift_start` stays at zero and is unused.
    """
    b = _buffers(env)
    for k, v in b.items():
        # Skip the once-per-step guards (`_step`, `_latch_step`, ...), which are
        # plain ints, and the buffers reset explicitly below. `last_*` is the
        # ENDED episode's outcome and must survive the reset, since Isaac Lab
        # resets inside the same step() that ends an episode.
        if not torch.is_tensor(v) or k == "lift_start" or k.startswith("last_"):
            continue
        v[env_ids] = 0
    b["verify_clean"][env_ids] = True      # innocent until the bolt touches the table
    # The proximity potential must be re-seeded, or the first progress reward
    # of a new episode is measured against the last step of the previous one.
    if b.get("prev_phi") is not None:
        b["prev_phi"][env_ids] = _potential(env)[env_ids]
    lb = _latch(env)
    # Keep the ENDED episode's lift step readable after the reset, the same way
    # the last_* outcome buffers are -- Isaac Lab resets inside the step() that
    # ends the episode, so anything read afterwards is the next episode's.
    lb["last_grasp_end"][env_ids] = lb["grasp_end"][env_ids]
    lb["grasp_end"][env_ids] = float(GRASP_STEPS)
    lb["ready_at"][env_ids] = -1.0
    lb["latched"][env_ids] = False
    for k in ("struct_hit", "link_hit", "paid_struct"):
        if b.get(k) is not None:
            b[k][env_ids] = False
    if b.get("struct_force") is not None:
        b["struct_force"][env_ids] = 0.0
    m = _metrics(env)
    for k in ("clipped", "ik_failed", "singular", "limit_terminations"):
        m[k][env_ids] = 0.0
    m["s_min"][env_ids] = 1.0
    m["arm_slack"][env_ids] = 1.0


def displace_wrist(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Start each episode with the wrist offset from the solved pinch pose.

    This is the curriculum. The bolt never moves; the WRIST starts wrong, and
    the policy's job is to find the grasping position and orientation before
    closing. A displacement is drawn per environment from this level's ranges
    and turned into arm joint angles by one multiply against the baked
    first-order map (see J_SHARP) -- no IK at reset.

    The sampled displacement is stashed on the env so the action term can seed
    its commanded pose analytically instead of reading a possibly stale wrist
    pose out of the simulator.
    """
    lvl = LEVELS[env.cfg.actions.pipeline.level]
    robot = env.scene[asset_cfg.name]
    n = len(env_ids)

    store = getattr(env, "_wrist_displacement", None)
    if store is None:
        store = torch.zeros(env.num_envs, 6, device=env.device)
        env._wrist_displacement = store
    if lvl.is_zero:
        store[env_ids] = 0.0
        return

    def u(lo, hi):
        return torch.rand(n, device=env.device) * (hi - lo) + lo

    d = torch.stack([
        u(-lvl.dxy, lvl.dxy), u(-lvl.dxy, lvl.dxy), u(lvl.dz_lo, lvl.dz_hi),
        u(lvl.droll_lo, lvl.droll_hi), u(lvl.dpitch_lo, lvl.dpitch_hi),
        u(-lvl.dyaw, lvl.dyaw),
    ], dim=1)
    store[env_ids] = d

    js = getattr(env, "_j_sharp", None)
    if js is None:
        js = torch.tensor(J_SHARP, device=env.device)
        env._j_sharp = js
    ids = getattr(env, "_arm_ids_cache", None)
    if ids is None:
        ids, _ = robot.find_joints(RIGHT_ARM, preserve_order=True)
        env._arm_ids_cache = ids

    q = _t(robot.data.joint_pos)[env_ids].clone()
    qd = torch.zeros_like(q)
    q[:, ids] = _t(robot.data.default_joint_pos)[env_ids][:, ids] + (js @ d.unsqueeze(2)).squeeze(2)
    robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)


def hold_default_joint_targets(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Second line of defence. The action term writes every joint every step,
    so this only covers the window between a reset and the first apply."""
    robot = env.scene[asset_cfg.name]
    default = _t(robot.data.default_joint_pos)[env_ids]
    robot.set_joint_position_target(default, env_ids=env_ids)
    robot.set_joint_velocity_target(torch.zeros_like(default), env_ids=env_ids)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


def _body_sensor(body: str) -> ContactSensorCfg:
    """Contact sensor on one body, filtered against the table's collider."""
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{body}",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table/geometry/mesh"],
    )


def _tip_sensor(name: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{TIP_BODIES[name]}",
        update_period=0.0,
        history_length=0,
        # A PhysX filter pattern must resolve to EXACTLY ONE contact entry per
        # environment. The table cube's collider mesh does. A broken filter
        # reports zero force silently, which for a contact reward and a success
        # test is the worst available failure mode.
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table/geometry/mesh"],
    )


def _stage_joint_pos() -> dict:
    """The nominal reset pose: waist and neck held, left side held, right hand
    at the synergy-open pose, right arm at the solved pinch pose.

    Always the CONTACT pose. Displacement is applied on top of it by the
    displace_wrist event, so level 0 -- no displacement -- is exactly the
    fixed-pose diagnostic baseline."""
    arm = PREGRASP_ARM_CONTACT
    jp = {
        **WAIST_HELD,
        "neckYaw": 0.0, "neckPitch": 0.0,
        "shoulderPitch_Left": 0.0, "shoulderRoll_Left": 0.0, "shoulderYaw_Left": 0.0,
        "elbow_Left": 0.0, "wristYaw_Left": 0.0, "wristPitch_Left": 0.0, "wristRoll_Left": 0.0,
        "l_.*": 0.0,                                  # left hand open
        **arm,
        **Q_OPEN,                                     # the 9 driven right-hand joints
        **Q_FIXED_RIGHT,                              # the 11 fixed right-hand joints
    }
    return jp


@configclass
class WujiPipelineSceneCfg(InteractiveSceneCfg):
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
            usd_path=ADAM_U_WUJI_USD,
            # TRUE here, unlike the other whole-robot envs: the fingertip
            # contact sensors are what the tactile observations, the contact
            # rewards and the excess-force check all read.
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # Off, as in every other env here. With a FIXED wrist this was
                # harmless; with a moving wrist the policy can drive the hand
                # through the torso, so the workspace box in LimitsCfg is the
                # guard. Turning it on for the arm chain is the correct fix and
                # costs solver time -- measure before switching.
                enabled_self_collisions=False,
                solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, ROBOT_ROOT_Z),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos=_stage_joint_pos(),
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Effort limits are the URDF's, not the PND env's 10 N.m. With the
            # arm reaching over the table the 10 N.m cap let the hand sag ~4 cm
            # below the kinematic solution -- and this stage starts 5 cm above
            # the bolt, so that sag would be the whole approach.
            # The RIGHT arm is stiffer than the repo's usual 300 N.m/rad, for
            # the same reason as the waist: the IK loop is proportional, so
            # whatever the drives droop under gravity stays as a permanent wrist
            # offset. Measured at 300: 7.4 mm at the stage-0 pose, 10.1 mm at
            # stage 1 (the arm is more extended), 8.5 mm of it straight down --
            # against 11 mm of fingertip-to-table clearance and a 50 mm
            # approach. Droop scales as 1/stiffness below the effort cap.
            # The LEFT arm is left soft: it is held for mass and inertia and its
            # droop costs the task nothing.
            "right_arm_actuators": ImplicitActuatorCfg(
                joint_names_expr=["shoulderPitch_Right", "shoulderRoll_Right",
                                  "shoulderYaw_Right", "elbow_Right"],
                effort_limit={"shoulder.*": 65.0, "elbow.*": 30.0},
                velocity_limit=4.0, stiffness=1500.0, damping=60.0,
            ),
            "left_arm_actuators": ImplicitActuatorCfg(
                joint_names_expr=["shoulderPitch_Left", "shoulderRoll_Left",
                                  "shoulderYaw_Left", "elbow_Left"],
                effort_limit={"shoulder.*": 65.0, "elbow.*": 30.0},
                velocity_limit=4.0, stiffness=300.0, damping=20.0,
            ),
            # STIFFER than the other envs' 300 N.m/rad, and this is not a
            # tuning preference -- it is a measured requirement.
            #
            # The waist holds a 1.05 rad (60 deg) forward lean, statically, for
            # the whole episode, and the ENTIRE reach envelope was solved at
            # that value. Measured with stiffness 300 by
            # tools/test_pipeline_arch.py: waistPitch commanded 1.05 settled at
            # 1.208 -- 0.158 rad, 9 degrees, of steady-state droop, gravity
            # pulling the leaned torso further forward. That silently invalidates
            # the solved pose and it was most of the 7.4 mm of wrist sag.
            #
            # It is stiffness-limited, not effort-limited: 300 x 0.158 = 47 N.m
            # against a 110 N.m cap. Raising stiffness is therefore the direct
            # fix, and it is free here because the waist is a static hold, never
            # a policy action -- there is no control loop to destabilise.
            "waist_actuators": ImplicitActuatorCfg(
                joint_names_expr=["waist.*"],
            #
            # Measured droop at this lean, by stiffness: 300 -> 0.158 rad,
            # 4000 -> 0.0112, 20000 -> see TRAINING_LOG. The relationship is
            # linear, as it must be for a proportional drive below its effort
            # cap. 20000 N.m/rad puts the droop at ~0.002 rad, about 1 mm at the
            # wrist, which is small against the 50 mm approach. Damping 600 is
            # near critical for the torso's inertia at that stiffness. The
            # cut-out hand's lift joint already runs at 2e5, so this is well
            # inside what this setup handles.
                effort_limit=110.0, velocity_limit=8.0, stiffness=20000.0, damping=600.0,
            ),
            "right_wrist_actuators": ImplicitActuatorCfg(
                joint_names_expr=["wristYaw_Right", "wristPitch_Right", "wristRoll_Right"],
                effort_limit=6.4, velocity_limit=4.0, stiffness=500.0, damping=20.0,
            ),
            "left_wrist_actuators": ImplicitActuatorCfg(
                joint_names_expr=["wristYaw_Left", "wristPitch_Left", "wristRoll_Left"],
                effort_limit=6.4, velocity_limit=4.0, stiffness=100.0, damping=4.0,
            ),
            "neck_actuators": ImplicitActuatorCfg(
                joint_names_expr=["neck.*"],
                effort_limit=6.4, velocity_limit=5.0, stiffness=20.0, damping=5.0,
            ),
            # The 9 DRIVEN right-hand joints keep Experiment 2's exact gains.
            # Grasp force comes out of this compliance -- a scripted full close
            # squeezes at ~25 N against the 40 N F_MAX limit -- so changing
            # stiffness here would change the contact forces and make stage 0
            # incomparable with the 100 % Experiment 2 already measured.
            "right_driven_fingers": ImplicitActuatorCfg(
                joint_names_expr=SYNERGY_JOINTS,
                effort_limit={
                    "r_thumb_cmc_flex": 0.6, "r_thumb_mcp": 0.3, "r_thumb_ip": 0.3,
                    "r_.*_mcp_flex": 2.0, "r_.*_pip": 0.3, "r_.*_dip": 0.3,
                },
                velocity_limit=8.0, stiffness=3.0, damping=0.1, armature=1e-4,
            ),
            # The 11 FIXED right-hand joints are stiffer, because they are
            # geometry, not grasping. Measured at stiffness 3.0: r_thumb_cmc_abd
            # commanded -0.85 settled at -0.900, 0.050 rad of droop -- and that
            # joint IS the thumb opposition the whole pinch is built on (the
            # gap goes 5.2 cm open to 1.2 cm closed; 0.05 rad is about 4 mm of
            # it). They touch nothing during a correct grasp, so stiffening them
            # changes no contact force.
            "right_fixed_fingers": ImplicitActuatorCfg(
                joint_names_expr=list(Q_FIXED_RIGHT.keys()),
                effort_limit={
                    "r_thumb_cmc_abd": 0.6, "r_.*_mcp_abd": 0.2,
                    "r_.*_mcp_flex": 2.0, "r_.*_pip": 0.3, "r_.*_dip": 0.3,
                },
                velocity_limit=8.0, stiffness=60.0, damping=2.0, armature=1e-4,
            ),
            "left_fingers": ImplicitActuatorCfg(
                joint_names_expr=["l_.*"],
                effort_limit={
                    "l_thumb_cmc_.*": 0.6, "l_.*_mcp_flex": 2.0, "l_.*_mcp_abd": 0.2,
                    "l_thumb_mcp": 0.3, "l_thumb_ip": 0.3,
                    "l_.*_pip": 0.3, "l_.*_dip": 0.3,
                },
                velocity_limit=8.0, stiffness=3.0, damping=0.1, armature=1e-4,
            ),
        },
    )

    cs_thumb = _tip_sensor("thumb")
    cs_index = _tip_sensor("index")
    cs_middle = _tip_sensor("middle")
    cs_ring = _tip_sensor("ring")
    cs_pinky = _tip_sensor("pinky")

    # Structural links: palm/wrist/hand base, forearm, upper arm.
    cs_palm = _body_sensor("wristRollRight")
    cs_forearm = _body_sensor("wristPitchRight")
    cs_upperarm = _body_sensor("elbowRight")
    # Intermediate finger links.
    cs_thumb_mid = _body_sensor("r_thumb_middle")
    cs_index_mid = _body_sensor("r_index_finger_middle")
    cs_middle_mid = _body_sensor("r_middle_finger_middle")
    cs_thumb_prox = _body_sensor("r_thumb_proximal")
    cs_index_prox = _body_sensor("r_index_finger_proximal")
    cs_middle_prox = _body_sensor("r_middle_finger_proximal")

    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))


# ---------------------------------------------------------------------------
# MDP
# ---------------------------------------------------------------------------


@configclass
class ActionsCfg:
    pipeline = PipelineActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        bolt_pos_rel = ObsTerm(func=bolt_pos_in_wrist)                 # 3
        bolt_rot_rel = ObsTerm(func=bolt_rot_in_wrist)                 # 6
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
        tip_dist = ObsTerm(func=fingertip_distances)                   # 3
                                                                       # = 58

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    # Experiment 2's, unchanged, so stage 0 is comparable with its 100 %.
    thumb_contact = RewTerm(func=thumb_contact_bonus, weight=0.5)
    finger_contact = RewTerm(func=finger_contact_bonus, weight=0.5)
    opposing_contact = RewTerm(func=opposing_contact_bonus, weight=1.0)
    success = RewTerm(func=success_bonus, weight=20.0)
    drop = RewTerm(func=drop_penalty, weight=-5.0)
    table_contact = RewTerm(func=table_contact_penalty, weight=-3.0)
    excess_force = RewTerm(func=excess_force_penalty, weight=-3.0)
    # Dense shaping, and the term stage 2 did not have. See proximity_progress.
    approach = RewTerm(func=proximity_progress, weight=PROX_WEIGHT)
    # -0.001, an order of magnitude below Experiment 2's -0.01. At -0.01 this
    # was the ONLY term with a gradient once the wrist could move, and the
    # stage-2 policy optimised it by holding a constant action: action_rate
    # shrank -0.25 -> -0.09 while every contact bonus stayed at exactly zero.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    # Small on purpose: see joint_limit_penalty.
    joint_limits = RewTerm(func=joint_limit_penalty, weight=-0.05)
    # Palm / forearm / upper arm on the table. Larger than the finger penalty
    # because it is never a correct configuration, only a failed one.
    structural_contact = RewTerm(func=structural_contact_penalty, weight=-5.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    off_table = DoneTerm(func=bolt_off_table)
    joint_limit = DoneTerm(func=joint_limit_termination)
    ik_failure = DoneTerm(func=ik_failure_termination)
    structural_contact = DoneTerm(func=structural_contact_termination)


@configclass
class EventsCfg:
    reset_bolt = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"asset_cfg": SceneEntityCfg("object"),
                "pose_range": {}, "velocity_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}},
    )
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )
    # AFTER reset_robot, which writes the nominal pose; this offsets from it.
    displace = EventTerm(func=displace_wrist, mode="reset", params={})
    hold_targets = EventTerm(func=hold_default_joint_targets, mode="reset", params={})
    reset_buffers = EventTerm(func=reset_pipeline_buffers, mode="reset", params={})


@configclass
class WujiPipelineEnvCfg(ManagerBasedRLEnvCfg):
    scene: WujiPipelineSceneCfg = WujiPipelineSceneCfg(num_envs=16, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Read by the shared bookkeeping in wuji_synergy_grasp_env_cfg._phases:
    # this episode is 90/30/30, Experiment 2's is 60/30/30.
    grasp_steps: int = GRASP_STEPS
    lift_steps: int = LIFT_STEPS
    hold_steps: int = HOLD_STEPS

    def __post_init__(self):
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

        # The bolt does NOT move. The curriculum displaces the wrist instead --
        # the same relative error to the policy, but the pinch geometry, the
        # table clearance and the solved pose all stay validated where they
        # were solved.
        self.events.reset_bolt.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}

        self.viewer.eye = (BOLT_XY[0] + 0.45, BOLT_XY[1] - 0.45, TABLE_TOP + 0.40)
        self.viewer.lookat = (BOLT_XY[0], BOLT_XY[1], TABLE_TOP + 0.03)


__all__ = ["WujiPipelineEnvCfg", "LimitsCfg", "Level", "LEVELS", "LEVEL",
           "Stage", "STAGES", "STAGE", "fingertip_distances", "_potential",
           "ACTION_NAMES", "RIGHT_ARM", "SYNERGY_JOINTS", "Q_FIXED_RIGHT", "WAIST_HELD",
           "PREGRASP_ARM_START", "PREGRASP_ARM_CONTACT", "WRIST_BODY", "WRIST_XY",
           "WRIST_ROT", "WRIST_Z_CONTACT", "BOLT_XY", "BOLT_YAW_DEG",
           "GRASP_STEPS", "LIFT_STEPS", "HOLD_STEPS", "EPISODE_STEPS", "LIFT_HEIGHT",
           "APPROACH_HEIGHT", "TABLE_TOP", "_metrics", "phase_of"]
