# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experiment 3: full 20-DoF finger grasp, for imitation learning then PPO.

Same robot, scene, pre-grasp pose, episode state machine, reward and success
test as Experiment 2 (`wuji_synergy_grasp_env_cfg.py`) -- this subclasses it, so
those stay in one place. Two things change:

  actions        20, one per right-hand joint, instead of 2 synergies.
  observations   66 instead of 26, because all 20 joints are now driven and so
                 all 20 are worth reporting.

Why 20 and not 2. A Manus glove publishes exactly 20 ergonomics angles per hand
and the Wuji hand has exactly 20 joints per hand, one for one (see
docs/EXPERIMENT3_DESIGN.md section 4), so a teleoperated demonstration is
natively 20-dimensional; projecting it onto two synergies would discard 18 of
the 20 numbers the human produced. And imitation learning only earns its
complexity where exploration cannot find the reward alone -- pure PPO already
solves the 2-synergy task at 100 %, so there would be nothing to improve on.

One environment, three action sources:

  teleop   tools/record_demo.py feeds actions from the glove over UDP and
           writes trajectories. The env knows nothing about UDP.
  BC       a cloned policy, same observation and action space.
  PPO      fine-tuned from the BC weights.

Actions map to each joint's full URDF range, `a = -1` the lower limit and
`a = +1` the upper. `A_OPEN` is the action vector that reproduces the pre-grasp
pose, which is where every episode starts and what an idle glove should send.
"""

from __future__ import annotations

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from adam_u_rl.envs.adam_u_wuji_grasp_env_cfg import _t
from adam_u_rl.envs.wuji_synergy_grasp_env_cfg import (  # noqa: F401  (re-exported)
    EPISODE_STEPS,
    GRASP_STEPS,
    HELD_CURL,
    HOLD_STEPS,
    LIFT_STEPS,
    OPEN_CURL,
    THUMB_ABD,
    WujiSynergyGraspEnvCfg,
    LIFT_HEIGHT,
    LIFT_JOINT,
    _buffers,
    _update,
    contact_flags,
    phase_of,
    tip_forces_obs,
)

# ---------------------------------------------------------------------------
# The 20 right-hand joints, in the order the Wuji driver uses on the wire
# (adam-moveit's wuji_joint_map is the authority: finger index = finger_id * 4
# + joint_id, fingers thumb/index/middle/ring/pinky, joints proximal -> distal).
# Keeping that order here means a glove packet, a recorded trajectory and the
# action vector are all the same layout, so nothing has to be re-indexed.
# ---------------------------------------------------------------------------

HAND_JOINTS = [
    "r_thumb_cmc_flex", "r_thumb_cmc_abd", "r_thumb_mcp", "r_thumb_ip",
    "r_index_finger_mcp_flex", "r_index_finger_mcp_abd", "r_index_finger_pip", "r_index_finger_dip",
    "r_middle_finger_mcp_flex", "r_middle_finger_mcp_abd", "r_middle_finger_pip", "r_middle_finger_dip",
    "r_ring_finger_mcp_flex", "r_ring_finger_mcp_abd", "r_ring_finger_pip", "r_ring_finger_dip",
    "r_pinky_mcp_flex", "r_pinky_mcp_abd", "r_pinky_pip", "r_pinky_dip",
]
NUM_HAND_JOINTS = len(HAND_JOINTS)

# Flexion limits, per joint, from the URDF. Used to build the pre-grasp pose.
_FLEX_LIMIT = {"cmc_flex": 1.291, "mcp": 1.570, "ip": 1.570,
               "mcp_flex": 1.570, "pip": 2.094, "dip": 1.570}


def _pregrasp_angle(joint: str) -> float:
    """The pre-grasp angle for one joint: Experiment 2's solved pose, written out.

    Thumb and the two synergy fingers sit at OPEN_CURL of their flexion limits,
    the thumb's opposition at THUMB_ABD, ring and little at HELD_CURL, and every
    other abduction at zero. Identical geometry to Experiment 2, so the pinch
    that was solved there is the one this starts from.
    """
    if joint == "r_thumb_cmc_abd":
        return THUMB_ABD
    if joint.endswith("_abd"):
        return 0.0
    curl = HELD_CURL if ("ring" in joint or "pinky" in joint) else OPEN_CURL
    # removeprefix, not split("r_"): "r_index_finger_mcp_flex" contains "r_"
    # inside "finger" too, and splitting there yields "index_finge".
    suffix = joint.removeprefix("r_")
    for key in ("cmc_flex", "mcp_flex", "pip", "dip", "mcp", "ip"):
        if suffix.endswith(key):
            return curl * _FLEX_LIMIT[key]
    raise ValueError(f"no flexion limit for {joint}")


PREGRASP_Q = {j: round(_pregrasp_angle(j), 4) for j in HAND_JOINTS}


def action_from_angles(q: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Joint angles [rad] -> the action that commands them. Inverse of the action term."""
    return (2.0 * (q - lower) / (upper - lower).clamp(min=1e-6) - 1.0).clamp(-1.0, 1.0)


def angles_from_action(a: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """The action term's mapping, for replay and for logging targets in radians."""
    return lower + (a.clamp(-1.0, 1.0) + 1.0) / 2.0 * (upper - lower)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def hand_joint_pos(env) -> torch.Tensor:
    return mdp.joint_pos(env, SceneEntityCfg("robot", joint_names=HAND_JOINTS, preserve_order=True))


def hand_joint_vel(env) -> torch.Tensor:
    return mdp.joint_vel(env, SceneEntityCfg("robot", joint_names=HAND_JOINTS, preserve_order=True))


@configclass
class TeleopObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=HAND_JOINTS, preserve_order=True)})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=HAND_JOINTS, preserve_order=True)})
        last_action = ObsTerm(func=mdp.last_action)
        tip_forces = ObsTerm(func=tip_forces_obs)
        contacts = ObsTerm(func=contact_flags)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class TeleopFingerAction(ActionTerm):
    """20 joint targets, plus the same phase logic and scripted lift as Experiment 2.

    A plain `JointPositionToLimitsActionCfg` is NOT enough here, and the first
    version of this env used one. Two things Experiment 2's synergy action term
    was quietly doing then stopped happening, and every episode failed:

      * nothing drove the prismatic `wrist_lift` joint, so the scripted lift
        never took place and the bolt was never raised at all;
      * the finger targets were not frozen after the grasp phase, so a glove
        still streaming during the lift kept commanding the hand -- a teleop
        operator relaxing their hand, or a cyclic test signal, would reopen the
        fingers mid-lift and drop the bolt.

    Freezing the targets is part of the task definition, not an optimisation:
    the lift is what tests whether the *chosen* grasp holds, so the operator
    must not be able to correct it while it happens.

    a = -1 is each joint's lower limit, a = +1 its upper, which makes the glove
    conversion a one-liner and means no action can command a target outside the
    hardware's range.
    """

    cfg: "TeleopFingerActionCfg"

    def __init__(self, cfg: "TeleopFingerActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        robot = self._asset
        self._ids, names = robot.find_joints(cfg.joint_names, preserve_order=True)
        if list(names) != list(cfg.joint_names):
            raise ValueError(f"joint order changed under us: {names}")
        self._lift_id, _ = robot.find_joints([cfg.lift_joint])
        lim = _t(robot.data.joint_pos_limits)[0, self._ids]
        self._lower, self._upper = lim[:, 0], lim[:, 1]
        self._raw = torch.zeros(self.num_envs, len(self._ids), device=self.device)
        self._target = torch.zeros(self.num_envs, len(self._ids), device=self.device)

    @property
    def action_dim(self) -> int:
        return len(self._ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._target

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0.0
        # Start from the pre-grasp pose, so a reset with no command holds it
        # rather than snapping to the middle of every joint's range.
        self._target[env_ids] = torch.tensor(
            [PREGRASP_Q[j] for j in self.cfg.joint_names], device=self.device)

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions.clamp(-1.0, 1.0)
        q = angles_from_action(self._raw, self._lower, self._upper)
        grasping = (phase_of(self._env) == 0).unsqueeze(1)
        self._target[:] = torch.where(grasping, q, self._target)

    def apply_actions(self):
        robot = self._asset
        robot.set_joint_position_target_index(target=self._target, joint_ids=self._ids)
        t = self._env.episode_length_buf.float()
        frac = ((t - GRASP_STEPS) / LIFT_STEPS).clamp(0.0, 1.0)
        lift_q = _buffers(self._env)["lift_start"] + frac * self.cfg.lift_height
        robot.set_joint_position_target_index(target=lift_q.unsqueeze(1), joint_ids=self._lift_id)


@configclass
class TeleopFingerActionCfg(ActionTermCfg):
    class_type: type = TeleopFingerAction
    asset_name: str = "robot"
    joint_names: list[str] = HAND_JOINTS
    lift_joint: str = "wrist_lift"
    lift_height: float = LIFT_HEIGHT


@configclass
class TeleopActionsCfg:
    fingers = TeleopFingerActionCfg()


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


@configclass
class WujiTeleopGraspEnvCfg(WujiSynergyGraspEnvCfg):
    observations: TeleopObservationsCfg = TeleopObservationsCfg()
    actions: TeleopActionsCfg = TeleopActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Start from Experiment 2's solved pre-grasp pose, written out per joint
        # rather than by regex, so the recorded trajectories and the action
        # vector share one explicit joint order.
        jp = self.scene.robot.init_state.joint_pos
        for key in list(jp):
            if key.startswith("r_") or key.startswith("("):
                jp.pop(key)
        jp.update(PREGRASP_Q)


__all__ = ["WujiTeleopGraspEnvCfg", "HAND_JOINTS", "NUM_HAND_JOINTS", "PREGRASP_Q",
           "action_from_angles", "angles_from_action",
           "_update", "_buffers", "phase_of",
           "GRASP_STEPS", "LIFT_STEPS", "HOLD_STEPS", "EPISODE_STEPS"]
