# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adam-U机器人抓取任务的强化学习环境配置 - 完整修复版"""

import os
import sys
# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.math import quat_apply
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.controllers.pink_ik_cfg import PinkIKControllerCfg
from pink.tasks import FrameTask

# 导入MDP函数
import isaaclab.envs.mdp as mdp
from envs.utils.observations import *

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
    
    # 使用线性奖励，限制最大值
    return torch.clamp(height_above_table, 0.0, 0.5)

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
            rot=[1.0, 0.0, 0.0, 0.0]
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
            rot=[1.0, 0.0, 0.0, 0.0]
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
            rot=[1, 0, 0, 0]
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
        spawn=sim_utils.UrdfFileCfg(
            asset_path="assets/robots/adam_u/urdf/adam_u.urdf",
            fix_base=True,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.4, 1.0),
            rot=(0.707, 0.0, 0.0, 0.707),
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

    # # full finger
    right_fingers = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "R_thumb.*", "R_index.*", "R_middle.*", "R_ring.*", "R_pinky.*"
        ],
        scale=0.2,
        use_default_offset=True,
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
    distance_to_object = RewTerm(
        func=compute_distance_reward,
        weight=2.0,
        params={"std": 1.0, "asset_cfg": SceneEntityCfg("object")}
    )

    hand_height = RewTerm(
        func=compute_hand_height_reward,
        weight=5.0,
        params={}
    )
    
    # 高度奖励 - 鼓励抓起物体
    object_height = RewTerm(
        func=compute_height_reward,
        weight=10.0,
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
    object_dropped = DoneTerm(
        func=object_dropped_termination,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "height_threshold": 0.5
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