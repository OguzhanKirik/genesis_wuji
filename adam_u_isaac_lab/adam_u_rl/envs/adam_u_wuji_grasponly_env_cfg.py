# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adam-U + Wuji Hand 2, grasp-only: the hand starts over the bolt, no reach.

Same robot, table and bolt as adam_u_wuji_grasp_env_cfg.py; everything that
differs is about removing the reach phase so the policy spends its samples on
the grasp:

  start pose   torso leaned 0.35 rad forward, right arm parked so the hand
               hovers PALM DOWN over the bolt, fingers pointing forward and
               lightly pre-curled (a "C" shape). Solved from the URDF on the
               host with tools/solve_pregrasp_pose.py:
                   python3 tools/solve_pregrasp_pose.py --waist 0.7 --seeds 30 --target 0.315 -0.215 1.035
               Why palm down and why the lean, both from tools/hand_geometry.py
               and the solver: the thumb and index tips meet 8.5 cm below the
               wrist, so a fingers-down hand cannot pinch anything lying on a
               table; and with the shoulder at z~1.52 the 0.95 m table is at
               the straight arm's reach, so without a lean the palm-down wrist
               stops 5 cm too high. With the lean the pre-curled thumb/index
               tips sit ~2 cm above the table straddling BOLT_XY, gap ~5 cm.
  bolt         at BOLT_XY = (0.39, -0.21) on the bare table, +-1.5 cm and a
               random yaw at reset. (adam-moveit's 0.55 m spot is beyond a
               palm-down reach for this arm.) BOLT_XY is where the scripted
               close in tools/probe_wuji.py --scenario grasp lifts the bolt.
  actions      20 finger joints, offset from the PRE-CURLED pose: action 0 is
               the start shape, +1 is the joint's upper limit (fully flexed),
               -1 opens past flat toward hyperextension (clipped at the URDF
               limit). Arm: the same 7 joints scaled to +-0.15 rad around the
               start pose -- enough to descend onto the bolt and lift it, not
               enough to wander off. The waist is not actuated by the policy;
               its drive holds the lean.
  rewards      no coarse reach term. Fingertip proximity (fine), gated
               closure, lift bonus, height ramp, off-table penalty, and a
               small penalty for the wrist leaving its start pose.
  episode      5 s. A grasp and lift takes about two.

Requires the `single` layout (bare table): the bin lives at adam-moveit's
0.55 m spot and would be beside a bolt that is no longer in it.
"""

from __future__ import annotations

import math

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from adam_u_rl.envs.adam_u_wuji_grasp_env_cfg import (
    LAYOUT,
    RIGHT_ARM,
    RIGHT_HAND,
    RIGHT_HAND_FLEXION,
    AdamUWujiGraspEnvCfg,
    _t,
    _tip,
    compute_grasp_reward,
    compute_height_reward,
    compute_lift_bonus,
    compute_off_surface_penalty,
    compute_pinch_distance_reward,
)

# Where the bolt rests for this scenario (world xy), and the body pose that
# parks the open hand over it. Regenerate them together; see the docstring.
BOLT_XY = (0.39, -0.21)
BOLT_JITTER = 0.015
PREGRASP_ARM = {
    "shoulderPitch_Right": -0.4924,
    "shoulderRoll_Right": -0.0949,
    "shoulderYaw_Right": +0.1688,
    "elbow_Right": -0.6418,
    "wristYaw_Right": +1.1548,
    "wristPitch_Right": +0.2496,
    "wristRoll_Right": -0.8600,
    "waistPitch": +0.3525,
}
# Light pre-curl, ~30% of the way to the flexion limits (tools/hand_geometry.py
# "closing together" table, s=0.30): both tips about 6.5 cm below the wrist
# and 9-12 cm ahead of it, resting ~4 mm above the table in the start pose.
# Finger abduction stays neutral; the thumb is swung to -0.95 rad, toward the
# fingers. Measured with tools/probe_wuji.py --scenario grasp (phase B): a
# scripted close from this posture with the thumb at -0.95 lifted the bolt in
# 9 of 16 placements within +-2 cm of BOLT_XY; at -0.40 it lifted 0-1 of 16
# and at -1.17, 4 of 16. The grasp is not a fingertip pinch -- the fingers
# curl back under the bolt and trap it against the thumb and palm.
PREGRASP_FINGERS = {
    "r_thumb_cmc_flex": 0.39,
    "r_thumb_cmc_abd": -0.95,
    "r_thumb_mcp": 0.47,
    "r_thumb_ip": 0.47,
    "r_.*_mcp_flex": 0.47,
    "r_.*_mcp_abd": 0.0,
    "r_.*_pip": 0.63,
    "r_.*_dip": 0.47,
}
# Per-joint action scale = upper limit - pre-curl, so a = +1 is fully flexed.
FINGER_SCALE = {
    "r_thumb_cmc_flex": 1.291 - 0.39,
    "r_thumb_cmc_abd": 0.698 + 0.95,
    "r_thumb_mcp": 1.570 - 0.47,
    "r_thumb_ip": 1.570 - 0.47,
    "r_.*_mcp_flex": 1.570 - 0.47,
    "r_.*_mcp_abd": 0.698,
    "r_.*_pip": 2.094 - 0.63,
    "r_.*_dip": 1.570 - 0.47,
}
ARM_SCALE = 0.20


def wrist_drift_penalty(env, ee_wrist_cfg: SceneEntityCfg = SceneEntityCfg("right_wrist_frame")):
    """Squared distance of the wrist from its start pose, in the horizontal plane only.

    Vertical motion is the job (descend, lift); horizontal wandering is not.
    The start pose is read from the env's own robot default on the first call.
    """
    pos = _tip(env, ee_wrist_cfg.name) - env.scene.env_origins
    ref = getattr(env, "_wrist_start_xy", None)
    if ref is None:
        ref = pos[:, :2].clone()
        env._wrist_start_xy = ref
    return torch.sum((pos[:, :2] - ref) ** 2, dim=1)


@configclass
class GraspOnlyActionsCfg:
    right_arm = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=RIGHT_ARM, scale=ARM_SCALE, use_default_offset=True,
    )
    right_fingers = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=RIGHT_HAND, scale=FINGER_SCALE, use_default_offset=True,
    )


@configclass
class GraspOnlyRewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    reach_fine = RewTerm(func=compute_pinch_distance_reward, weight=2.0, params={"std": 0.04})
    grasp = RewTerm(
        func=compute_grasp_reward,
        weight=3.0,
        params={"std": 0.04, "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_FLEXION)},
    )
    lift_bonus = RewTerm(func=compute_lift_bonus, weight=5.0, params={"rest_z": LAYOUT["bolt_rest_z"], "min_lift": 0.02})
    # Cap 8 cm: with +-0.2 rad on the arm the scripted lift raised the wrist
    # ~4 cm and the held bolt 5-8 cm, so 15 cm was unreachable.
    object_height = RewTerm(func=compute_height_reward, weight=15.0, params={"rest_z": LAYOUT["bolt_rest_z"], "cap": 0.08})
    off_surface = RewTerm(func=compute_off_surface_penalty, weight=-2.0, params={"rest_z": LAYOUT["bolt_rest_z"]})
    wrist_drift = RewTerm(func=wrist_drift_penalty, weight=-5.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5e-4)
    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM)},
    )


@configclass
class AdamUWujiGraspOnlyEnvCfg(AdamUWujiGraspEnvCfg):
    actions: GraspOnlyActionsCfg = GraspOnlyActionsCfg()
    rewards: GraspOnlyRewardsCfg = GraspOnlyRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        if LAYOUT["bin"] is not None:
            raise ValueError("the grasp-only scenario needs ADAM_U_WORKCELL=single (bare table)")

        self.episode_length_s = 5.0

        # Torso leaned, hand parked palm-down over the bolt, fingers pre-curled.
        # The finger regexes must come AFTER the base cfg's "r_.*": 0.0 entry
        # (dict order is match order in Isaac Lab), so remove and re-add it.
        jp = self.scene.robot.init_state.joint_pos
        jp.update(PREGRASP_ARM)
        jp.pop("r_.*", None)
        jp.update(PREGRASP_FINGERS)

        # Bolt under the hand, small jitter, random yaw (body pitch == world
        # yaw for the lying bolt, see the base env).
        self.scene.object.init_state.pos = (BOLT_XY[0], BOLT_XY[1], LAYOUT["surface_z"] + 0.03)
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-BOLT_JITTER, BOLT_JITTER),
            "y": (-BOLT_JITTER, BOLT_JITTER),
            "z": (0.0, 0.0),
            "pitch": (-math.pi, math.pi),
        }
        # Smaller joint noise: +-0.1 rad on the shoulder moves the hand ~5 cm,
        # which is the whole margin the pre-grasp pose has.
        self.events.reset_robot_joints.params["position_range"] = (-0.03, 0.03)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        self.viewer.eye = (1.2, -0.9, 1.5)
        self.viewer.lookat = (BOLT_XY[0], BOLT_XY[1], 1.0)


__all__ = ["AdamUWujiGraspOnlyEnvCfg", "BOLT_XY", "PREGRASP_ARM"]
