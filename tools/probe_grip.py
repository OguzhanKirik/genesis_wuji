#!/usr/bin/env python3
"""Determine, empirically, which finger action sign CLOSES the Inspire hand.

The 12 right-hand finger actions are JointPositionAction with scale=0.2 and
use_default_offset=True, so action a maps to default + 0.2*a radians. Nothing
in the URDF says which direction is "closed", and a grasp reward with the sign
inverted rewards splaying the hand open. So measure it: drive the fingers to
each extreme and compare fingertip-to-palm distance and fingertip spread.

    docker compose exec adam_rl python /opt/tools/probe_grip.py
"""

from __future__ import annotations

import sys

sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from genesis_rl.envs.adam_u_grasp_env_cfg import AdamUGraspEnvCfg

cfg = AdamUGraspEnvCfg()
cfg.scene.num_envs = 2
cfg.sim.device = "cuda:0"

gym.register(id="Grip", entry_point="isaaclab.envs:ManagerBasedRLEnv",
             disable_env_checker=True, kwargs={"env_cfg_entry_point": AdamUGraspEnvCfg})
env = gym.make("Grip", cfg=cfg)
env.reset()
u = env.unwrapped
robot = u.scene["robot"]
am = u.action_manager

fing_term = am.get_term("right_fingers")
n_arm = am.get_term("right_arm").action_dim
fing_ids = fing_term._joint_ids
if isinstance(fing_ids, slice):
    fing_ids = list(range(len(robot.data.joint_names)))[fing_ids]
fing_names = [robot.data.joint_names[i] for i in fing_ids]

print("=" * 72)
print(f"FINGER ACTION DIM {fing_term.action_dim}  "
      f"cfg={type(fing_term.cfg).__name__}  scale={fing_term.cfg.scale}  "
      f"rescale_to_limits={getattr(fing_term.cfg, 'rescale_to_limits', None)}  "
      f"use_default_offset={getattr(fing_term.cfg, 'use_default_offset', None)}")
print("\nfinger joints, default position and URDF limits:")
lim = robot.data.joint_pos_limits[0]
dflt = robot.data.default_joint_pos[0]
for i, n in zip(fing_ids, fing_names):
    print(f"  {n:24s} default={dflt[i]:+.4f}  limits=[{lim[i,0]:+.4f}, {lim[i,1]:+.4f}]")

TIPS = ["R_thumb_distal", "R_index_distal", "R_middle_distal",
        "R_ring_distal", "R_pinky_distal"]
palm_i = robot.data.body_names.index("wristRollRight")
tip_i = {t: robot.data.body_names.index(t) for t in TIPS}


def settle(action, steps=60):
    a = torch.zeros((u.num_envs, am.total_action_dim), device=u.device)
    a[:, n_arm:] = action
    for _ in range(steps):
        u.step(a)


def measure():
    palm = robot.data.body_pos_w[0, palm_i]
    tips = {t: robot.data.body_pos_w[0, tip_i[t]] for t in TIPS}
    tip_to_palm = {t: torch.norm(p - palm).item() for t, p in tips.items()}
    pts = torch.stack(list(tips.values()))
    centroid = pts.mean(dim=0)
    spread = torch.norm(pts - centroid, dim=1).mean().item()
    jp = robot.data.joint_pos[0]
    return tip_to_palm, spread, {n: jp[i].item() for i, n in zip(fing_ids, fing_names)}


rows = {}
for label, act in [("action -1.0", -1.0), ("action  0.0", 0.0), ("action +1.0", +1.0)]:
    env.reset()
    settle(act)
    rows[label] = measure()

print("\n" + "=" * 72)
print("fingertip -> palm (wristRollRight) distance, metres")
print("=" * 72)
print(f"{'tip':20}" + "".join(f"{k:>14}" for k in rows))
for t in TIPS:
    print(f"{t:20}" + "".join(f"{rows[k][0][t]:14.5f}" for k in rows))
print(f"{'MEAN':20}" + "".join(
    f"{sum(rows[k][0].values())/len(TIPS):14.5f}" for k in rows))
print(f"{'tip spread':20}" + "".join(f"{rows[k][1]:14.5f}" for k in rows))

m = {k: sum(rows[k][0].values()) / len(TIPS) for k in rows}
closes = "+1.0" if m["action +1.0"] < m["action -1.0"] else "-1.0"
print(f"\n==> fingertips are CLOSEST to the palm at action {closes}")
print(f"    so CLOSING the hand = action sign {closes[0]}"
      f"  (joint position {'ABOVE' if closes == '+1.0' else 'BELOW'} default)")

print("\n" + "=" * 72)
print("joint positions at each extreme (to confirm the drives actually track)")
print("=" * 72)
print(f"{'joint':24}" + "".join(f"{k:>14}" for k in rows))
for n in fing_names:
    print(f"{n:24}" + "".join(f"{rows[k][2][n]:+14.4f}" for k in rows))

env.close()
app.close()
