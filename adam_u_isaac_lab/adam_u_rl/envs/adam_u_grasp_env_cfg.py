# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adam-U grasping env config for Isaac Lab 3.0 / Isaac Sim 6.0.

Derived mechanically from adam_u_rl/envs/adam_u_grasp_env_cfg.py so the scene,
action space, observation terms and reward terms are IDENTICAL. That is what
makes a policy trained on the 5.1 side runnable here. Only the version-specific
parts differ: the robot spawns from a pre-converted flat USD instead of the
URDF, `configclass` moved, and the unused IK imports are dropped.

Keep the two files in sync. If you change a reward or an observation here and
not there, an exported policy will silently receive a different observation
vector -- scripts/play.py checks the dimension, but not the semantics.
"""

import os
import sys
# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Repo root, derived from this file rather than the process working directory.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ADAM_U_ASSETS = os.environ.get("ADAM_U_ASSETS", os.path.join(REPO_ROOT, "assets"))

# Isaac Lab 3.0 loads a PRE-CONVERTED USD, not the URDF. 3.0's URDF importer
# nests rigid bodies under the kinematic tree, so the explicit prim paths this
# env uses for its three FrameTransformers -- {ENV_REGEX_NS}/Robot/waistRoll,
# .../R_thumb_distal, .../R_pinky_distal -- stop resolving, and every distance
# reward silently loses its frames. The USD emitted by Isaac Lab 2.3.2's
# converter is flat (all 44 bodies directly under the default prim), which both
# versions read. Generate it with tools/export_usd.py in the 5.1 image.
ADAM_U_USD = os.environ.get(
    "ADAM_U_USD", os.path.join(ADAM_U_ASSETS, "robots", "adam_u", "usd", "adam_u.usd")
)

import math
from typing import Sequence
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import FrameTransformer
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.math import quat_apply

# 导入MDP函数
import isaaclab.envs.mdp as mdp
from adam_u_rl.envs.utils.observations import *

##
# 自定义MDP函数
##

def compute_distance_reward(env, std: float, asset_cfg: SceneEntityCfg, ee_thumb_cfg: SceneEntityCfg = SceneEntityCfg("right_thumb_frame"), ee_pinky_cfg: SceneEntityCfg = SceneEntityCfg("right_pinky_frame")):
    """计算机器人手部到物体的距离奖励"""
    object_pos = env.scene["object"].data.root_pos_w

    ee_thumb: FrameTransformer = env.scene[ee_thumb_cfg.name]
    ee_thumb_t = ee_thumb.data.target_pos_w[..., 0, :]

    ee_pinky: FrameTransformer = env.scene[ee_pinky_cfg.name]
    ee_pinky_t = ee_pinky.data.target_pos_w[..., 0, :]

    # 计算距离
    object_thumb_distance = torch.norm(object_pos - ee_thumb_t, dim=1)
    object_pinky_distance = torch.norm(object_pos - ee_pinky_t, dim=1)

    return 1 - (torch.tanh(object_thumb_distance / std) + torch.tanh(object_pinky_distance / std)) / 2.0

def compute_hand_height_reward(env, ee_wrist_cfg: SceneEntityCfg = SceneEntityCfg("right_wrist_frame")):
    """计算手部高度奖励"""
    ee_frame: FrameTransformer = env.scene[ee_wrist_cfg.name]
    ee_height = ee_frame.data.target_pos_w[..., 0, 2]
    return torch.clamp((ee_height - 1.0) * 1.0, -1.0, 0.05)

def compute_height_reward(env, asset_cfg: SceneEntityCfg):
    """计算物体高度奖励"""
    # 获取物体位置
    object_pos = env.scene["object"].data.root_pos_w
    
    # 桌面高度
    table_height = 1.05
    
    # 物体高度超过桌面给予奖励
    height_above_table = object_pos[:, 2] - table_height
    
    # Capped at 0.20 m, not 0.50 m. The old cap asked for a 50 cm lift before
    # the term saturated, which put nearly all of its 5.0-point budget beyond
    # any plausible early behaviour (and possibly outside the arm's reachable
    # workspace -- unverified, see README). A 20 cm lift is already an
    # unambiguous success and makes the ramp 2.5x steeper per cm.
    return torch.clamp(height_above_table, 0.0, 0.2)

# Measured with tools/probe_grip.py: every right-hand finger joint has its
# LOWER limit at 0 rad (fully open) and closes toward its UPPER limit, so a
# joint position normalised onto [lower, upper] is a direct closure fraction.
# The sign is measured rather than assumed -- inverted, this term would pay the
# policy to splay the hand open on the cube.
def compute_grasp_reward(
    env,
    std: float,
    asset_cfg: SceneEntityCfg,
    ee_thumb_cfg: SceneEntityCfg = SceneEntityCfg("right_thumb_frame"),
    ee_pinky_cfg: SceneEntityCfg = SceneEntityCfg("right_pinky_frame"),
):
    """Pay for closing the fingers, but only while the tips are on the cube.

    gate * closure. The gate is ~1 with the fingertips at the cube and ~0 a
    hand's width away, so curling the fingers in free space earns nothing --
    the term only pays for a closure that could actually be a grip. This is the
    credit path the 12 finger DoF never had: previously they reached the reward
    only through object_height, which needs an already-completed grasp.
    """
    object_pos = env.scene["object"].data.root_pos_w

    ee_thumb: FrameTransformer = env.scene[ee_thumb_cfg.name]
    ee_thumb_t = ee_thumb.data.target_pos_w[..., 0, :]

    ee_pinky: FrameTransformer = env.scene[ee_pinky_cfg.name]
    ee_pinky_t = ee_pinky.data.target_pos_w[..., 0, :]

    dist = (
        torch.norm(object_pos - ee_thumb_t, dim=1)
        + torch.norm(object_pos - ee_pinky_t, dim=1)
    ) / 2.0
    gate = 1.0 - torch.tanh(dist / std)

    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    limits = robot.data.joint_pos_limits[:, asset_cfg.joint_ids, :]
    lower, upper = limits[..., 0], limits[..., 1]
    closure = ((q - lower) / (upper - lower).clamp(min=1e-6)).clamp(0.0, 1.0).mean(dim=1)

    return gate * closure


def compute_lift_bonus(
    env,
    asset_cfg: SceneEntityCfg,
    rest_height: float = 1.05,
    min_lift: float = 0.02,
):
    """Sparse credit for a MINIMAL successful lift.

    The height ramp alone barely paid for a real grasp: measured against the old
    budget, grasping and lifting 5 cm was worth 0.42 of 8.25 points, only ~18%
    more than hovering next to the cube doing nothing. This step is what makes
    getting the cube off the table at all worth more than standing beside it.
    """
    object_z = env.scene["object"].data.root_pos_w[:, 2]
    return (object_z > rest_height + min_lift).float()


def compute_off_table_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    rest_height: float = 1.05,
    deadband: float = 0.005,
):
    """Charge for knocking the cube below its resting height.

    Nothing used to: measured at world z 1.00, already falling off the table,
    object_height still paid 0.0035 and no term objected, while the episode ran
    on for another 0.525 m of fall before object_dropped fired at z < 0.5.
    """
    object_z = env.scene["object"].data.root_pos_w[:, 2]
    return (rest_height - deadband - object_z).clamp(min=0.0, max=0.5)


def object_dropped_termination(env, asset_cfg: SceneEntityCfg, height_threshold: float = 0.5):
    """检查物体是否掉落"""
    # 获取物体位置
    object_pos = env.scene["object"].data.root_pos_w
    
    # 如果物体高度低于阈值，认为掉落
    return object_pos[:, 2] < height_threshold

def compute_joint_vel_penalty(env, asset_cfg: SceneEntityCfg):
    """计算关节速度惩罚，鼓励平滑运动"""
    joint_vel = env.scene["robot"].data.joint_vel[:, asset_cfg.joint_ids]
    return -torch.sum(torch.square(joint_vel), dim=1)


# ---------------------------------------------------------------------------
# QUATERNION ORDER CHANGED IN ISAAC LAB 3.0.
#
#   2.3.2:  InitialStateCfg.rot = (w, x, y, z), default (1, 0, 0, 0)
#   3.0:    InitialStateCfg.rot = (x, y, z, w), default (0, 0, 0, 1)
#
# So the 2.3.2 value (0.707, 0, 0, 0.707) -- 90 deg about Z -- is read by 3.0
# as x=0.707, w=0.707, i.e. 90 deg about X, and the robot spawns tipped onto
# its side with the right hand BELOW the 1.025 m table surface. It still
# trains, just on a different physical problem: hand_height came out at
# -0.657 versus -0.037 on 2.3.2.
#
# Every rot in this file is therefore written (x, y, z, w).
# ---------------------------------------------------------------------------

##
# 场景配置
##

@configclass
class AdamUGraspSceneCfg(InteractiveSceneCfg):
    """Adam-U机器人抓取场景配置"""

    # 桌面
    table_top = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TableTop",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.0, 1.0], 
            rot=[0.0, 0.0, 0.0, 1.0]  # identity, (x,y,z,w)
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.5, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.5, 0.3), 
                metallic=0.1,
                roughness=0.8
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                static_friction=0.8,
                dynamic_friction=0.7,
                restitution=0.1,
            ),
        ),
    )

    # 桌腿
    table_leg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TableLeg",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.5], 
            rot=[0.0, 0.0, 0.0, 1.0]  # identity, (x,y,z,w)
        ),
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=1.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.3, 0.2), 
                metallic=0.0,
                roughness=0.9
            ),
        ),
    )

    # 目标物体
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.0, 1.05],  # 放在桌子上，z=0.8
            rot=[0.0, 0.0, 0.0, 1.0]  # identity, (x,y,z,w)
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),  # 5cm 立方体
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.4, 0.8), 
                metallic=0.3
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
        ),
    )

    # Adam_U 机器人配置
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ADAM_U_USD,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.4, 1.0),
            rot=(0.0, 0.0, 0.707, 0.707),  # 90 deg about Z, (x,y,z,w) -- see note above
            joint_pos={
                # 腰部关节
                "waistRoll": 0.0,
                "waistPitch": 0.0,
                "waistYaw": 0.0,
                
                # 右臂初始姿势 - 调整为抓取准备姿势
                "shoulderPitch_Right": 0.3,
                "shoulderRoll_Right": -0.1,
                "shoulderYaw_Right": 0.0,
                "elbow_Right": -0.8,
                "wristYaw_Right": 0.0,
                "wristPitch_Right": -0.1,
                "wristRoll_Right": 0.0,
                
                # 左臂保持下垂
                "shoulderPitch_Left": 0.0,
                "shoulderRoll_Left": 0.0,
                "shoulderYaw_Left": 0.0,
                # "elbow_Left": -0.5,
                "elbow_Left": 0.0,
                "wristYaw_Left": 0.0,
                "wristPitch_Left": 0.0,
                "wristRoll_Left": 0.0,
                
                # 头部和手指关节
                "neckYaw": 0.0,
                "neckPitch": 0.0,
                # 手指关节设为默认值
                ".*thumb.*": 0.0,
                ".*index.*": 0.0,
                ".*middle.*": 0.0,
                ".*ring.*": 0.0,
                ".*pinky.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            # 右臂关节执行器（主要控制对象）
            "arm_actuators": ImplicitActuatorCfg(
                joint_names_expr=[
                    "shoulderPitch_Right", 
                    "shoulderRoll_Right", 
                    "shoulderYaw_Right",
                    "elbow_Right", 
                    "shoulderPitch_Left", 
                    "shoulderRoll_Left", 
                    "shoulderYaw_Left",
                    "elbow_Left", 
                ],
                effort_limit=10.0,
                velocity_limit=2.0,
                stiffness=80.0,
                damping=10.0,
            ),
            # 其他关节执行器（保持固定或较小控制）
            "waist_actuators": ImplicitActuatorCfg(
                joint_names_expr=["waist.*"],
                effort_limit=110.0,
                velocity_limit=8.0,
                stiffness=80.0,
                damping=20.0,
            ),
            "wrist_actuators": ImplicitActuatorCfg(
                joint_names_expr=[
                    "wristYaw_Right", 
                    "wristPitch_Right", 
                    "wristRoll_Right",
                    "wristYaw_Left", 
                    "wristPitch_Left", 
                    "wristRoll_Left"
                ],
                effort_limit=10.0,
                velocity_limit=2.0,
                stiffness=80.0,
                damping=0.0,
            ),
            "neck_actuators": ImplicitActuatorCfg(
                joint_names_expr=["neck.*"],
                effort_limit=6.4,
                velocity_limit=5.0,
                stiffness=20.0,
                damping=5.0,
            ),
            "right_finger_actuators": ImplicitActuatorCfg(
                joint_names_expr=["R_thumb.*", "R_index.*", "R_middle.*", "R_ring.*", "R_pinky.*"],
                effort_limit=10.0,
                velocity_limit=5.0,
                stiffness=10.0,
                damping=2.0,
            ),
            "left_finger_actuators": ImplicitActuatorCfg(
                joint_names_expr=["L_thumb.*", "L_index.*", "L_middle.*", "L_ring.*", "L_pinky.*"],
                effort_limit=10.0,
                velocity_limit=10.0,
                stiffness=5.0,
                damping=2.0,
            ),
        },
    )

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # 灯光
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# MDP设置
##

@configclass
class ActionsCfg:
    """动作空间配置 - 只控制Adam-U右臂7个关节"""
    
    # 右臂关节位置控制
    right_arm = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[
            "shoulderPitch_Right", 
            "shoulderRoll_Right", 
            "shoulderYaw_Right",
            "elbow_Right", 
            "wristYaw_Right", 
            "wristPitch_Right", 
            "wristRoll_Right"
        ], 
        scale=1.0,  # 动作缩放因子
        use_default_offset=True,  # 使用默认偏移
    )

    # Right-hand fingers, mapped to their FULL joint range.
    #
    # This was JointPositionActionCfg(scale=0.2, use_default_offset=True), and
    # that made a grasp physically impossible. Every right-hand finger joint
    # has limits [0, 1.1 .. 1.7] rad with its default at 0 (fully open), so the
    # commandable target was default + 0.2*a, i.e. 0 .. 0.2 rad -- about 12% of
    # the closing range -- and the whole negative half of the action space
    # clipped at the lower limit and did nothing at all. Measured with
    # tools/probe_grip.py: sweeping the finger actions from -1.0 to +1.0 moved
    # the mean fingertip-to-palm distance by 0.7 mm, 0.15632 -> 0.15558. No
    # reward could have taught a grasp through that.
    #
    # rescale_to_limits maps [-1, 1] onto each joint's own soft limits, so -1 is
    # fully open and +1 fully closed for all 12 joints despite their differing
    # ranges (thumb MCP2 0.5 rad, index MCP 1.7 rad). Note this also changes the
    # neutral action: 0.0 is now mid-range, not the open default.
    right_fingers = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=[
            "R_thumb.*", "R_index.*", "R_middle.*", "R_ring.*", "R_pinky.*"
        ],
        scale=1.0,
        rescale_to_limits=True,
    )


@configclass 
class ObservationsCfg:
    """观测空间配置"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测组 - 简化观测空间"""
        
        # 机器人右臂关节位置 (7维)
        right_arm_pos = ObsTerm(
            func=mdp.joint_pos, 
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", 
                    joint_names=[
                        "shoulderPitch_Right", 
                        "shoulderRoll_Right", 
                        "shoulderYaw_Right",
                        "elbow_Right", 
                        "wristYaw_Right", 
                        "wristPitch_Right", 
                        "wristRoll_Right"
                    ]
                )
            }
        )
        
        # 机器人右臂关节速度 (7维)
        right_arm_vel = ObsTerm(
            func=mdp.joint_vel, 
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", 
                    joint_names=[
                        "shoulderPitch_Right", 
                        "shoulderRoll_Right", 
                        "shoulderYaw_Right",
                        "elbow_Right", 
                        "wristYaw_Right", 
                        "wristPitch_Right", 
                        "wristRoll_Right"
                    ]
                )
            }
        )

        right_hand_pos = ObsTerm(get_right_eef_pos)
        right_hand_quat = ObsTerm(get_right_eef_quat)

        # 机器人右手关节位置
        right_hand_joint_pos = ObsTerm(
            func=mdp.joint_pos, 
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", 
                    joint_names=[
                        "R_thumb.*", "R_index.*", "R_middle.*", "R_ring.*", "R_pinky.*"
                    ]
                )
            }
        )
        # 机器人右手关节速度
        right_hand_joint_vel = ObsTerm(
            func=mdp.joint_vel, 
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", 
                    joint_names=[
                        "R_thumb.*", "R_index.*", "R_middle.*", "R_ring.*", "R_pinky.*"
                    ]
                )
            }
        )
        
        # 物体位置 (3维)
        object_position = ObsTerm(
            func=mdp.root_pos_w, 
            params={"asset_cfg": SceneEntityCfg("object")}
        )

        actions = ObsTerm(func=mdp.last_action)
        
        # 机器人基座位置 (3维) 
        robot_base_position = ObsTerm(
            func=mdp.root_pos_w, 
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # 观测组
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    """奖励函数配置"""
    
    # 存活奖励
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    
    # 终止惩罚
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    
    # 距离奖励 - 鼓励机器人靠近物体
    #
    # Two scales, because one cannot do both jobs. The original single term used
    # std=1.0 against a 5 cm cube, which put the entire task inside the linear
    # toe of the tanh: the slope was a flat -0.02 points/cm everywhere from 0 to
    # 80 cm. Measured consequence after 300 iterations -- the policy parked
    # 7.7 cm from the cube CENTRE, i.e. 5.2 cm clear of its surface, collected
    # 92.3% of the term there, and closing that last gap was worth 0.104 of an
    # 8.25-point budget. There was no gradient sharpening on approach at all.
    #
    # Dropping std alone would have broken the other end: at the 0.23-0.34 m
    # spawn distance, std=0.05 yields ~0 reward AND ~0 slope, so there would be
    # nothing to guide the initial reach. So: a coarse term that carries the
    # long-range signal, and a fine term that only wakes up in the last ~10 cm.
    reach_coarse = RewTerm(
        func=compute_distance_reward,
        weight=1.0,
        params={"std": 0.30, "asset_cfg": SceneEntityCfg("object")}
    )

    reach_fine = RewTerm(
        func=compute_distance_reward,
        weight=2.0,
        params={"std": 0.04, "asset_cfg": SceneEntityCfg("object")}
    )

    hand_height = RewTerm(
        func=compute_hand_height_reward,
        weight=5.0,
        params={}
    )
    
    # 抓握奖励 - the credit path the fingers never had.
    # Gated on proximity, so closing the hand in free space pays nothing.
    grasp = RewTerm(
        func=compute_grasp_reward,
        weight=2.0,
        params={
            "std": 0.05,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "R_thumb.*", "R_index.*", "R_middle.*", "R_ring.*", "R_pinky.*"
                ]
            ),
        }
    )

    # 起举奖励 - sparse step for getting the cube off the table at all.
    lift_bonus = RewTerm(
        func=compute_lift_bonus,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("object"), "min_lift": 0.02}
    )

    # 高度奖励 - 鼓励抓起物体
    object_height = RewTerm(
        func=compute_height_reward,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("object")}
    )

    # 掉落惩罚 - charge for sweeping the cube off the table.
    off_table = RewTerm(
        func=compute_off_table_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("object")}
    )

    # 动作平滑奖励 - 鼓励平滑动作
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    
    # 关节速度惩罚 - 鼓励平滑运动
    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.1e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "shoulderPitch_Right", 
                    "shoulderRoll_Right", 
                    "shoulderYaw_Right",
                    "elbow_Right", 
                    "wristYaw_Right", 
                    "wristPitch_Right", 
                    "wristRoll_Right"
                ]
            )
        }
    )

@configclass
class TerminationsCfg:
    """终止条件配置"""
    
    # 时间限制
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 物体掉落终止
    #
    # 0.95 m, not 0.5 m. The table surface is at 1.025 m and the cube rests
    # centred at 1.050 m, so the old threshold let a swept-off cube fall 0.525 m
    # -- most of a second of physics -- before the episode ended, during which
    # the reach terms kept paying out. 0.95 m is 7.5 cm below the table surface:
    # unambiguously off the table, and nothing a legitimate lift ever visits.
    object_dropped = DoneTerm(
        func=object_dropped_termination,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "height_threshold": 0.95
        }
    )

@configclass
class EventsCfg:
    """事件配置"""
    
    # 启动时随机化物体质量
    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.1, 0.4),
            "operation": "add",
        },
    )
    
    # 重置时随机化物体位置 - 保持在桌子上
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {
                "x": (-0.1, 0.1),  # x方向小范围随机
                "y": (-0.1, 0.1),  # y方向小范围随机
                "z": (0.0, 0.0)   # z方向固定在桌子上
            },
            "velocity_range": {
                "x": (0.0, 0.0),  # 初始速度设为0
                "y": (0.0, 0.0),
                "z": (0.0, 0.0)
            },
        },
    )
    
    # 重置时随机化机器人右臂关节位置
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.05, 0.05),
        },
    )

##
# 环境配置
##

@configclass
class AdamUGraspEnvCfg(ManagerBasedRLEnvCfg):
    """Adam-U抓取任务环境配置"""

    # 场景设置 - 减少环境数量以便调试
    scene: AdamUGraspSceneCfg = AdamUGraspSceneCfg(num_envs=16, env_spacing=4.0)
    
    # MDP设置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    
    # RL设置
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    def __post_init__(self):
        """初始化后配置"""
        # 一般设置
        self.decimation = 2  # 控制频率降采样
        self.episode_length_s = 10.0  # 回合长度
        
        # 视角设置
        self.viewer.eye = (2.0, 2.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
        
        # 仿真设置
        self.sim.dt = 1 / 60  # 仿真时间步长
        self.sim.render_interval = self.decimation
        
        # 物理设置
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        
        # 设备设置（如果需要GPU加速）
        self.sim.device = "cuda:0"
        
        # 渲染设置
        # `disable_contact_processing` was removed from SimulationCfg in some
        # 3.0 revisions; set it only if this build still has it.
        if hasattr(self.sim, "disable_contact_processing"):
            self.sim.disable_contact_processing = False
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.right_wrist_frame = FrameTransformerCfg(
                prim_path="{ENV_REGEX_NS}/Robot/waistRoll",
                debug_vis=False,
                visualizer_cfg=marker_cfg,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}/Robot/wristRollRight",
                        name="right_wrist_frame",
                        offset=OffsetCfg(
                            pos=[0.0, 0.0, 0.0],
                        ),
                    ),
                ],
            )
        self.scene.right_thumb_frame = FrameTransformerCfg(
                prim_path="{ENV_REGEX_NS}/Robot/waistRoll",
                debug_vis=False,
                visualizer_cfg=marker_cfg,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}/Robot/R_thumb_distal",
                        name="right_thumb_frame",
                        offset=OffsetCfg(
                            pos=[0.0, 0.0, 0.0],
                        ),
                    ),
                ],
            )
        self.scene.right_pinky_frame = FrameTransformerCfg(
                prim_path="{ENV_REGEX_NS}/Robot/waistRoll",
                debug_vis=False,
                visualizer_cfg=marker_cfg,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}/Robot/R_pinky_distal",
                        name="right_pinky_frame",
                        offset=OffsetCfg(
                            pos=[0.0, 0.0, 0.0],
                        ),
                    ),
                ],
            )

# 导出配置类
__all__ = ["AdamUGraspEnvCfg"]