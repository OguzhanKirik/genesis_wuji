#!/usr/bin/env python3
"""Dump the spawned robot's pose -- a fast check that the robot spawns upright.

Written to compare the 5.1 and 3.0 environments, which is how the Isaac Lab 3.0
quaternion-order change was caught: the robot spawned rotated 90 deg about X,
lying on its side with the right hand below the table, and still trained
happily on the wrong problem. Keep using it after any change to the robot's
init_state or the USD.

Expected, upright:  waistRoll world_z 1.0767, wristRollRight ~1.016

    docker compose run --rm adam_rl python /opt/tools/compare_pose.py

Prints the articulation root pose, the world height of the three bodies the
reward terms read, the joint ordering, and the first observation vector's
width -- all the things that silently differ between a URDF spawn and a USD
spawn.
"""

from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--pkg", default="adam_u_rl", help="package to load the env config from")
parser.add_argument("--num_envs", type=int, default=4)
args, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from adam_u_rl.envs.adam_u_grasp_env_cfg import AdamUGraspEnvCfg

cfg = AdamUGraspEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = "cuda:0"

gym.register(id="Cmp", entry_point="isaaclab.envs:ManagerBasedRLEnv",
             disable_env_checker=True, kwargs={"env_cfg_entry_point": AdamUGraspEnvCfg})
env = gym.make("Cmp", cfg=cfg)
obs, _ = env.reset()

u = env.unwrapped
robot = u.scene["robot"]
origins = u.scene.env_origins

print("=" * 64)
print(f"PKG {args.pkg}")
print(f"OBS_WIDTH {obs['policy'].shape[-1]}")
print(f"ACTION_DIM {u.action_manager.total_action_dim}")
print(f"NUM_BODIES {len(robot.data.body_names)}")
print(f"NUM_JOINTS {len(robot.data.joint_names)}")

root = robot.data.root_pos_w[0] - origins[0]
rq = robot.data.root_quat_w[0]
print(f"ROOT_POS {root[0]:+.5f} {root[1]:+.5f} {root[2]:+.5f}")
print(f"ROOT_QUAT {rq[0]:+.5f} {rq[1]:+.5f} {rq[2]:+.5f} {rq[3]:+.5f}")

# The bodies the reward terms actually read.
for name in ["waistRoll", "wristRollRight", "R_thumb_distal", "R_pinky_distal"]:
    if name in robot.data.body_names:
        i = robot.data.body_names.index(name)
        p = robot.data.body_pos_w[0, i] - origins[0]
        # hand_height uses the ABSOLUTE world z, not the env-relative one.
        pw = robot.data.body_pos_w[0, i]
        print(f"BODY {name:18} rel=({p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}) world_z={pw[2]:+.5f}")
    else:
        print(f"BODY {name:18} MISSING")

obj = u.scene["object"]
print(f"OBJECT_POS_W {obj.data.root_pos_w[0,0]:+.4f} "
      f"{obj.data.root_pos_w[0,1]:+.4f} {obj.data.root_pos_w[0,2]:+.5f}")

print(f"JOINT_ORDER {','.join(robot.data.joint_names)}")
print(f"BODY_ORDER {','.join(robot.data.body_names)}")
print("=" * 64)

env.close()
app.close()
