# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adam-U with Wuji Hand 2: right-arm bolt grasping, Isaac Lab 3.0 / Isaac Sim 6.0.

The sibling file adam_u_grasp_env_cfg.py trains the PND (Inspire) hand to lift
a 5 cm cube. This one is the same task structure on a DIFFERENT robot and a
different workcell, both vendored from adam-moveit:

  robot      assets/robots/adam_u_wuji/   Genesis design body with Wuji Hand 2:
                                          20 independent revolute joints per
                                          hand, no coupling. 60 bodies, 59 DoF.
                                          tools/make_wuji_urdf.py, then
                                          tools/export_usd_wuji.py.
  workcell   assets/workcell/             adam-moveit's table, bin and bolt
                                          (workcell.json / workcell_single.json,
                                          rod-with-threads-0001.usdc), copied
                                          verbatim.

Frame convention is adam-moveit's, so poses transfer 1:1 from there: the robot
root (lifting_Columns_base) sits at the world origin facing +x, floor at z=0,
table top at z=0.95, waistRoll at z=1.1267. The PND env instead spawned the
robot at (0,-0.4,1.0) rotated 90 deg about Z; do not mix the two.

Layout is picked at import time by the ADAM_U_WORKCELL environment variable:

  single  (default)  one bolt on the bare table, no bin  -- workcell_single.json
  bin                the bin is present and the bolt starts inside it -- workcell.json

Both layouts put the bin/bolt at y=+0.20 in adam-moveit, in front of the LEFT
arm, because that stack grasps with the left hand. This env trains the RIGHT
arm, so WORKCELL_SIDE="right" mirrors y. Set it to "left" to keep the layout
as delivered.

Wuji-specific decisions, each measured or sourced rather than assumed:

  * Flexion is positive on every finger joint (adam-moveit FK check, and the
    SRDF's `closed` state puts every flexion joint at its upper limit). Open is
    q=0, not the lower limit -- the lower limits are hyperextension (-1.047
    rad). The grasp reward therefore measures closure as q/upper over the 15
    flexion joints, NOT (q-lower)/(upper-lower) as the PND env does; that
    formula would call an open Wuji hand 40% closed.
  * Fingertip frames: the URDF's fixed `*_tip` links merge into the distal
    links at import, so the FrameTransformers put the tip back as an offset
    along the distal link's -z (thumb 29.78 mm, fingers 24.75 mm; the URDF's
    tip joint origins).
  * Self-collision is OFF. adam-moveit measured the thumb's base housing
    nesting 2.6-4.1 mm into the forearm hulls at the home pose; with
    self-collision on, PhysX would fight that every step.
  * Physics at 120 Hz, control at 30 Hz. adam-moveit's Isaac stage diverged
    (joint angles of 1e8 rad) when a Wuji pinch pressed a bolt into the static
    table at 60 Hz and was stable at 240 Hz. 120 is the compromise for 4096
    envs; raise it if grasps blow up.
  * Finger drives: stiffness 3 N.m/rad, damping 0.1, effort limits from the
    URDF (0.2-2 N.m per joint). The vendor's published gains (0.18-0.69
    N.m/rad) are for its own hardware loop and are far too soft for a PhysX
    position drive on gram-scale links.

Every quaternion in this file is (x, y, z, w) -- Isaac Lab 3.0 order.
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.configclass import configclass

import isaaclab.envs.mdp as mdp
from adam_u_rl.envs.utils.observations import get_right_eef_pos, get_right_eef_quat

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ADAM_U_ASSETS = os.environ.get("ADAM_U_ASSETS", os.path.join(REPO_ROOT, "assets"))

# Flat USD from Isaac Lab 2.3.2's converter (tools/export_usd_wuji.py). 3.0's
# own importer nests bodies and the explicit prim paths below stop resolving.
ADAM_U_WUJI_USD = os.environ.get(
    "ADAM_U_WUJI_USD",
    os.path.join(ADAM_U_ASSETS, "robots", "adam_u_wuji", "usd", "adam_u_wuji.usd"),
)

WORKCELL_DIR = os.path.join(ADAM_U_ASSETS, "workcell")
WORKCELL_LAYOUT = os.environ.get("ADAM_U_WORKCELL", "single")
WORKCELL_SIDE = os.environ.get("WORKCELL_SIDE", "right")

# adam-moveit anchors lifting_Columns_base 0.99975 m below its 1.05 m
# base_height, so waistRoll lands at the same world height as the PND body
# there. Same number here so the table (top 0.95) is where that stack has it.
ROBOT_ROOT_Z = 1.05 - 0.99975

# The bolt's rigid body frame is at its mid-height (the vendor USD translates
# it +17.5 mm from the stem end). Lying on its side the body centre rests
# 11.0 mm above the surface -- measured with tools/probe_wuji.py (2026-09-19,
# identical across envs, bolt at rest). Re-measure after any change to the
# asset; the lift terms key off it.
BOLT_REST_OFFSET = 0.011
BOLT_LYING_ROT = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))  # 90 deg about X, (x,y,z,w)


def _load_layout() -> dict:
    """Read the adam-moveit workcell JSON and mirror it for the training arm."""
    name = {"single": "workcell_single.json", "bin": "workcell.json"}.get(WORKCELL_LAYOUT)
    if name is None:
        raise ValueError(f"ADAM_U_WORKCELL must be 'single' or 'bin', got {WORKCELL_LAYOUT!r}")
    with open(os.path.join(WORKCELL_DIR, name)) as f:
        raw = json.load(f)

    sign = -1.0 if WORKCELL_SIDE == "right" else 1.0
    table = raw["table"]
    table_top = table["center"][2] + table["size"][2] / 2.0
    bolts = raw["bolts"]
    bin_ = raw.get("bin")

    if bin_:
        cx, cy = bin_["center_xy"]
        area = tuple(bin_["inner"][:2])
        surface_z = table_top + bin_["wall"]  # the bin floor
    else:
        cx, cy = bolts.get("center_xy", table["center"][:2])
        area = tuple(bolts.get("area", [0.10, 0.10]))
        surface_z = table_top

    return {
        "file": name,
        "table_center": tuple(table["center"]),
        "table_size": tuple(table["size"]),
        "table_color": tuple(table["color"]),
        "table_top": table_top,
        "bin": None if not bin_ else {
            "center_xy": (cx, sign * cy),
            "inner": tuple(bin_["inner"]),
            "wall": bin_["wall"],
            "color": tuple(bin_["color"]),
        },
        "bolt_center_xy": (cx, sign * cy),
        "bolt_area": area,
        "bolt_asset": os.path.join(WORKCELL_DIR, bolts["asset"]),
        "bolt_mass": float(bolts.get("mass", 0.03)),
        "surface_z": surface_z,
        "bolt_rest_z": surface_z + BOLT_REST_OFFSET,
    }


LAYOUT = _load_layout()

# ---------------------------------------------------------------------------
# Joint groups
# ---------------------------------------------------------------------------

RIGHT_ARM = [
    "shoulderPitch_Right", "shoulderRoll_Right", "shoulderYaw_Right",
    "elbow_Right", "wristYaw_Right", "wristPitch_Right", "wristRoll_Right",
]
# All 20 right-hand joints. Only hand joints start with "r_"; body joints are
# camelCase (elbow_Right), so the prefix is unambiguous.
RIGHT_HAND = ["r_.*"]
# The 15 flexion joints: thumb cmc_flex/mcp/ip, each finger mcp_flex/pip/dip.
# The 5 abduction joints (r_*_abd) are excluded from the closure measure.
RIGHT_HAND_FLEXION = ["r_.*_flex", "r_.*_pip", "r_.*_dip", "r_thumb_mcp", "r_thumb_ip"]

# Fingertip offsets in the distal link frame (URDF *_tip_fixed_joint origins).
THUMB_TIP = (0.0, 0.0, -0.02978)
FINGER_TIP = (0.0, 0.0, -0.02475)

# ---------------------------------------------------------------------------
# MDP terms
# ---------------------------------------------------------------------------


def _t(x) -> torch.Tensor:
    """Isaac Lab 3.0 asset data are warp-backed ProxyArrays whose .shape is the
    warp shape (a quaternion array is (N,), not (N,4)). Torch ops unwrap them,
    but anything returned raw or indexed must be a real tensor: take .torch."""
    return x.torch if hasattr(x, "torch") else x


def _tip(env, name: str) -> torch.Tensor:
    ft: FrameTransformer = env.scene[name]
    return _t(ft.data.target_pos_w)[..., 0, :]


def compute_pinch_distance_reward(
    env, std: float,
    ee_thumb_cfg: SceneEntityCfg = SceneEntityCfg("right_thumb_frame"),
    ee_index_cfg: SceneEntityCfg = SceneEntityCfg("right_index_frame"),
):
    """1 - mean tanh(dist/std) of the thumb and index fingertips to the bolt.

    Thumb + index rather than the PND env's thumb + pinky: the Wuji hand is
    used as a thumb-index gripper in adam-moveit and a bolt is a pinch object.
    """
    obj = _t(env.scene["object"].data.root_pos_w)
    d_thumb = torch.norm(obj - _tip(env, ee_thumb_cfg.name), dim=1)
    d_index = torch.norm(obj - _tip(env, ee_index_cfg.name), dim=1)
    return 1.0 - (torch.tanh(d_thumb / std) + torch.tanh(d_index / std)) / 2.0


def compute_hand_height_reward(env, floor_z: float, ee_wrist_cfg: SceneEntityCfg = SceneEntityCfg("right_wrist_frame")):
    """Small bonus for keeping the wrist above the work surface, penalty below."""
    z = _tip(env, ee_wrist_cfg.name)[:, 2]
    return torch.clamp(z - floor_z, -1.0, 0.05)


def compute_flexion_closure(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Mean closure fraction of the flexion joints, q/upper clamped to [0,1].

    Open is q=0. The lower limit is hyperextension, so it is NOT the open
    pose, and the PND env's (q-lower)/(upper-lower) does not apply here.
    """
    robot = env.scene[asset_cfg.name]
    q = _t(robot.data.joint_pos)[:, asset_cfg.joint_ids]
    upper = _t(robot.data.joint_pos_limits)[:, asset_cfg.joint_ids, 1].clamp(min=1e-6)
    return (q / upper).clamp(0.0, 1.0).mean(dim=1)


def compute_grasp_reward(
    env, std: float, asset_cfg: SceneEntityCfg,
    ee_thumb_cfg: SceneEntityCfg = SceneEntityCfg("right_thumb_frame"),
    ee_index_cfg: SceneEntityCfg = SceneEntityCfg("right_index_frame"),
):
    """gate * closure: pay for curling the fingers only while the tips are on the bolt."""
    obj = _t(env.scene["object"].data.root_pos_w)
    dist = (torch.norm(obj - _tip(env, ee_thumb_cfg.name), dim=1)
            + torch.norm(obj - _tip(env, ee_index_cfg.name), dim=1)) / 2.0
    gate = 1.0 - torch.tanh(dist / std)
    return gate * compute_flexion_closure(env, asset_cfg)


def compute_height_reward(env, rest_z: float, cap: float = 0.20):
    """Linear ramp for lifting the bolt above its resting height, capped."""
    z = _t(env.scene["object"].data.root_pos_w)[:, 2]
    return torch.clamp(z - rest_z, 0.0, cap)


def compute_lift_bonus(env, rest_z: float, min_lift: float = 0.02):
    """Sparse step for getting the bolt off the surface at all."""
    z = _t(env.scene["object"].data.root_pos_w)[:, 2]
    return (z > rest_z + min_lift).float()


def compute_off_surface_penalty(env, rest_z: float, deadband: float = 0.01):
    """Charge for knocking the bolt below its resting height (off the table / out of the bin)."""
    z = _t(env.scene["object"].data.root_pos_w)[:, 2]
    return (rest_z - deadband - z).clamp(min=0.0, max=0.5)


def object_dropped_termination(env, height_threshold: float):
    return _t(env.scene["object"].data.root_pos_w)[:, 2] < height_threshold


def hold_default_joint_targets(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Point every joint drive at the default pose after a reset.

    Joints the policy does not actuate (waist, neck, left arm and hand) get no
    target from the action manager, and the reset events only write joint
    STATES, so their drives keep whatever target they had -- which is zero,
    not the default pose. Measured (tools/probe_wuji.py --scenario grasp):
    waistPitch default 0.352, settled at 0.030. Harmless for actuated joints,
    whose targets the action manager overwrites every step anyway.
    """
    robot = env.scene[asset_cfg.name]
    default = _t(robot.data.default_joint_pos)[env_ids]
    robot.set_joint_position_target(default, env_ids=env_ids)
    robot.set_joint_velocity_target(torch.zeros_like(default), env_ids=env_ids)


def object_to_hand(env) -> torch.Tensor:
    """Bolt position relative to the right wrist, world axes. Saves the policy the subtraction."""
    obj = _t(env.scene["object"].data.root_pos_w) - env.scene.env_origins
    return obj - get_right_eef_pos(env)


def object_quat(env) -> torch.Tensor:
    return _t(env.scene["object"].data.root_quat_w)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


def _static_box(prim: str, center, size, color) -> AssetBaseCfg:
    """A static collider, as adam-moveit's isaac_runner authors table and bin."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(center), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(
            size=tuple(size),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color), roughness=0.8),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                static_friction=0.8,
                dynamic_friction=0.7,
                restitution=0.1,
            ),
        ),
    )


@configclass
class AdamUWujiGraspSceneCfg(InteractiveSceneCfg):
    """Robot, table, bolt. The bin (if the layout has one) is added in __post_init__."""

    table = _static_box("Table", LAYOUT["table_center"], LAYOUT["table_size"], LAYOUT["table_color"])

    # The bolt. The vendor USD already carries RigidBodyAPI + MassAPI on its
    # body prim and a convex-hull collider on the 327k-face mesh; the spawner's
    # nested-apply finds them. Mass is overridden to adam-moveit's 30 g (the
    # file says 52 g at steel density).
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bolt",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(LAYOUT["bolt_center_xy"][0], LAYOUT["bolt_center_xy"][1], LAYOUT["surface_z"] + 0.03),
            rot=BOLT_LYING_ROT,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=LAYOUT["bolt_asset"],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=LAYOUT["bolt_mass"]),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
        ),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ADAM_U_WUJI_USD,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,  # see module docstring
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, ROBOT_ROOT_Z),
            rot=(0.0, 0.0, 0.0, 1.0),  # identity: robot faces +x, table ahead
            joint_pos={
                "waistRoll": 0.0, "waistPitch": 0.0, "waistYaw": 0.0,
                # Right arm: the PND env's pre-grasp pose. Same body, same
                # joint conventions, so it carries over.
                "shoulderPitch_Right": 0.3,
                "shoulderRoll_Right": -0.1,
                "shoulderYaw_Right": 0.0,
                "elbow_Right": -0.8,
                "wristYaw_Right": 0.0,
                "wristPitch_Right": -0.1,
                "wristRoll_Right": 0.0,
                # Left arm hangs.
                "shoulderPitch_Left": 0.0, "shoulderRoll_Left": 0.0, "shoulderYaw_Left": 0.0,
                "elbow_Left": 0.0, "wristYaw_Left": 0.0, "wristPitch_Left": 0.0, "wristRoll_Left": 0.0,
                "neckYaw": 0.0, "neckPitch": 0.0,
                # Both hands open (q=0 is flat on the Wuji hand).
                "r_.*": 0.0,
                "l_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Arm drives: effort limits are the URDF's (shoulder 65, elbow 30
            # N.m), not the PND env's 10 N.m. Measured with tools/probe_wuji.py
            # --scenario grasp: with the arm reaching forward over the table
            # the 10 N.m cap plus 80 N.m/rad let the hand sag ~4 cm below the
            # kinematic solution (index tip 0.952 vs 0.990 predicted). The
            # grasp-only scenario parks the hand 2-3 cm over the bolt, so that
            # sag was the whole margin.
            "arm_actuators": ImplicitActuatorCfg(
                joint_names_expr=[
                    "shoulderPitch_Right", "shoulderRoll_Right", "shoulderYaw_Right", "elbow_Right",
                    "shoulderPitch_Left", "shoulderRoll_Left", "shoulderYaw_Left", "elbow_Left",
                ],
                effort_limit={"shoulder.*": 65.0, "elbow.*": 30.0},
                velocity_limit=4.0, stiffness=300.0, damping=20.0,
            ),
            "waist_actuators": ImplicitActuatorCfg(
                joint_names_expr=["waist.*"],
                effort_limit=110.0, velocity_limit=8.0, stiffness=300.0, damping=30.0,
            ),
            "wrist_actuators": ImplicitActuatorCfg(
                joint_names_expr=["wristYaw_.*", "wristPitch_.*", "wristRoll_.*"],
                effort_limit=6.4, velocity_limit=4.0, stiffness=100.0, damping=4.0,
            ),
            "neck_actuators": ImplicitActuatorCfg(
                joint_names_expr=["neck.*"],
                effort_limit=6.4, velocity_limit=5.0, stiffness=20.0, damping=5.0,
            ),
            # Wuji finger drives. Effort limits are the URDF's, per joint type.
            "right_finger_actuators": ImplicitActuatorCfg(
                joint_names_expr=RIGHT_HAND,
                effort_limit={
                    "r_thumb_cmc_.*": 0.6,
                    "r_.*_mcp_flex": 2.0,
                    "r_.*_mcp_abd": 0.2,
                    "r_thumb_mcp": 0.3, "r_thumb_ip": 0.3,
                    "r_.*_pip": 0.3, "r_.*_dip": 0.3,
                },
                velocity_limit=8.0,
                stiffness=3.0,
                damping=0.1,
                armature=1e-4,
            ),
            "left_finger_actuators": ImplicitActuatorCfg(
                joint_names_expr=["l_.*"],
                effort_limit={
                    "l_thumb_cmc_.*": 0.6,
                    "l_.*_mcp_flex": 2.0,
                    "l_.*_mcp_abd": 0.2,
                    "l_thumb_mcp": 0.3, "l_thumb_ip": 0.3,
                    "l_.*_pip": 0.3, "l_.*_dip": 0.3,
                },
                velocity_limit=8.0,
                stiffness=3.0,
                damping=0.1,
                armature=1e-4,
            ),
        },
    )

    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# ---------------------------------------------------------------------------
# MDP
# ---------------------------------------------------------------------------


@configclass
class ActionsCfg:
    right_arm = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=RIGHT_ARM, scale=1.0, use_default_offset=True,
    )
    # All 20 right-hand joints on their full URDF range: -1 is the lower limit
    # (hyperextended), +1 the upper (fully flexed / abducted), 0 the midpoint,
    # which is a slightly curled hand -- a reasonable pre-grasp shape. Same
    # mechanism the PND env settled on after scale=0.2 left the hand unable to
    # close (see its comments).
    right_fingers = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot", joint_names=RIGHT_HAND, scale=1.0, rescale_to_limits=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        right_arm_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM)})
        right_arm_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM)})
        right_hand_pos = ObsTerm(get_right_eef_pos)
        right_hand_quat = ObsTerm(get_right_eef_quat)
        right_hand_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND)})
        right_hand_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND)})
        object_position = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_orientation = ObsTerm(func=object_quat)
        object_to_hand = ObsTerm(func=object_to_hand)
        actions = ObsTerm(func=mdp.last_action)
        robot_base_position = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # Coarse + fine reach, same two-scale reasoning as the PND env: one std
    # cannot carry both the 0.3-0.5 m approach and the last centimetres.
    reach_coarse = RewTerm(func=compute_pinch_distance_reward, weight=1.0, params={"std": 0.30})
    reach_fine = RewTerm(func=compute_pinch_distance_reward, weight=2.0, params={"std": 0.04})

    hand_height = RewTerm(func=compute_hand_height_reward, weight=5.0, params={"floor_z": LAYOUT["table_top"]})

    grasp = RewTerm(
        func=compute_grasp_reward,
        weight=2.0,
        params={"std": 0.05, "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_FLEXION)},
    )
    lift_bonus = RewTerm(func=compute_lift_bonus, weight=3.0, params={"rest_z": LAYOUT["bolt_rest_z"], "min_lift": 0.02})
    object_height = RewTerm(func=compute_height_reward, weight=10.0, params={"rest_z": LAYOUT["bolt_rest_z"]})
    off_surface = RewTerm(func=compute_off_surface_penalty, weight=-2.0, params={"rest_z": LAYOUT["bolt_rest_z"]})

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM)},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 7.5 cm below the surface: unambiguously off the table / out of the bin.
    object_dropped = DoneTerm(
        func=object_dropped_termination,
        params={"height_threshold": LAYOUT["surface_z"] - 0.075},
    )


@configclass
class EventsCfg:
    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("object"), "mass_distribution_params": (-0.01, 0.02), "operation": "add"},
    )
    # The bolt starts lying on its side (BOLT_LYING_ROT) 3 cm above the
    # surface and drops. The delta orientation is applied in the body frame;
    # after the 90 deg X rotation the body's Y axis is the world Z, so a "pitch"
    # range here is a random yaw on the table.
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {
                "x": (-LAYOUT["bolt_area"][0] / 2.0, LAYOUT["bolt_area"][0] / 2.0),
                "y": (-LAYOUT["bolt_area"][1] / 2.0, LAYOUT["bolt_area"][1] / 2.0),
                "z": (0.0, 0.0),
                "pitch": (-math.pi, math.pi),
            },
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.1, 0.1), "velocity_range": (-0.05, 0.05)},
    )
    # Must come after reset_robot_joints (dict order is execution order).
    hold_default_pose = EventTerm(func=hold_default_joint_targets, mode="reset", params={})


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


@configclass
class AdamUWujiGraspEnvCfg(ManagerBasedRLEnvCfg):
    scene: AdamUWujiGraspSceneCfg = AdamUWujiGraspSceneCfg(num_envs=16, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # 120 Hz physics, 30 Hz control (see module docstring).
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.device = "cuda:0"
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        if hasattr(self.sim, "disable_contact_processing"):
            self.sim.disable_contact_processing = False

        self.viewer.eye = (2.0, -1.5, 2.0)
        self.viewer.lookat = (0.55, -0.2, 1.0)

        # The bin: floor slab plus four walls, exactly as isaac_runner.author_workcell.
        b = LAYOUT["bin"]
        if b is not None:
            cx, cy = b["center_xy"]
            ix, iy, iz = b["inner"]
            w = b["wall"]
            top = LAYOUT["table_top"]
            col = b["color"]
            self.scene.bin_floor = _static_box("BinFloor", (cx, cy, top + w / 2), (ix + 2 * w, iy + 2 * w, w), col)
            self.scene.bin_wall_xp = _static_box("BinWallXP", (cx + (ix + w) / 2, cy, top + w + iz / 2), (w, iy + 2 * w, iz), col)
            self.scene.bin_wall_xn = _static_box("BinWallXN", (cx - (ix + w) / 2, cy, top + w + iz / 2), (w, iy + 2 * w, iz), col)
            self.scene.bin_wall_yp = _static_box("BinWallYP", (cx, cy + (iy + w) / 2, top + w + iz / 2), (ix, w, iz), col)
            self.scene.bin_wall_yn = _static_box("BinWallYN", (cx, cy - (iy + w) / 2, top + w + iz / 2), (ix, w, iz), col)

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        def frame(target_body: str, name: str, offset) -> FrameTransformerCfg:
            return FrameTransformerCfg(
                prim_path="{ENV_REGEX_NS}/Robot/waistRoll",
                debug_vis=False,
                visualizer_cfg=marker_cfg,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path=f"{{ENV_REGEX_NS}}/Robot/{target_body}",
                        name=name,
                        offset=OffsetCfg(pos=list(offset)),
                    )
                ],
            )

        self.scene.right_wrist_frame = frame("wristRollRight", "right_wrist_frame", (0.0, 0.0, 0.0))
        self.scene.right_thumb_frame = frame("r_thumb_distal", "right_thumb_frame", THUMB_TIP)
        self.scene.right_index_frame = frame("r_index_finger_distal", "right_index_frame", FINGER_TIP)


__all__ = ["AdamUWujiGraspEnvCfg", "LAYOUT"]
