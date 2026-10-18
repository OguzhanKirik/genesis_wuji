#!/usr/bin/env python3
"""Measure the approach corridor: where the arm and hand actually go during nominal grasps.

    docker compose exec -e ADAM_U_RG_LEVEL=7 adam_rl python /opt/tools/probe_corridor.py \
        --checkpoint_path logs/rsl_rl/adam_u30_wuji_reach_grasp_L7/<run>/model_<best>.pt --episodes 512

Runs a trained policy at a clutter-free level and accumulates, on a 0.5 cm
grid around the bolt, every table-plane cell that any proxy sphere of the
right arm or hand ever covers (the sphere's xy disc, at any height). The
union over all episodes is the corridor. Obstacles at the clutter levels are
placed OUTSIDE this footprint, dilated by the largest obstacle's half-diagonal
plus a margin, so an obstacle can only be hit by an approach that leaves the
nominal envelope -- which is what the policy then has to learn to avoid.

Writes genesis_rl/tasks/wuji_reach_grasp/corridor.npz (bolt-relative grid),
which the env loads at import. The strip heuristic it replaces let the
level-7 policy hit random obstacles in 53 % of level-10 episodes.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=512)
parser.add_argument("--half_extent", type=float, default=0.25, help="m, grid half-size around the bolt")
parser.add_argument("--cell", type=float, default=0.005, help="m")
parser.add_argument("--out", type=str, default=None)
args, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

from genesis_rl.algos.rsl_rl.rsl_rl_ppo_cfg import RslRlWujiReachGraspPPORunnerCfg  # noqa: E402
from genesis_rl.envs import wuji_reach_grasp_env_cfg as G  # noqa: E402
from genesis_rl.envs.adam_u_wuji_grasp_env_cfg import _t  # noqa: E402

device = "cuda:0"
cfg = G.WujiReachGraspEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = device
lvl = G.LEVELS[cfg.actions.pipeline.level]
if lvl.clutter.active:
    print("run this at a clutter-free level (the corridor is the obstacle-free envelope)")
    raise SystemExit(1)
gym.register(id="Corridor", entry_point="genesis_rl.tasks.wuji_reach_grasp.env_cfg:ReachGraspEnv",
             disable_env_checker=True, kwargs={"env_cfg_entry_point": G.WujiReachGraspEnvCfg})
env = gym.make("Corridor", cfg=cfg)
u = env.unwrapped
ac = handle_deprecated_rsl_rl_cfg(RslRlWujiReachGraspPPORunnerCfg(), metadata.version("rsl-rl-lib"))
ac.device = device
w = RslRlVecEnvWrapper(env, clip_actions=ac.clip_actions)
runner = OnPolicyRunner(w, ac.to_dict(), log_dir=None, device=device)
runner.load(args.checkpoint_path)
policy = runner.get_inference_policy(device=device)

n = u.num_envs
c = G._ids(u)
robot = u.scene["robot"]
bolt = torch.tensor(G.BOLT_XY, device=device)
H, cell = args.half_extent, args.cell
M = int(round(2 * H / cell))
ax = (torch.arange(M, device=device) + 0.5) * cell - H            # cell centres, bolt-relative
gx, gy = torch.meshgrid(ax, ax, indexing="ij")
cells = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)     # (M*M, 2)
occ = torch.zeros(M * M, dtype=torch.bool, device=device)
hmax = torch.zeros(M * M, device=device)                           # highest sphere top over each cell

# all proxy spheres in one table
idx = torch.cat([c["groups"][g][0] for g in G.GROUP_NAMES])
off = torch.cat([c["groups"][g][1] for g in G.GROUP_NAMES])
rad = torch.cat([c["groups"][g][2] for g in G.GROUP_NAMES])
K = idx.numel()


def stamp():
    pos_w = _t(robot.data.body_link_pos_w) - u.scene.env_origins.unsqueeze(1)
    quat_w = _t(robot.data.body_link_quat_w)
    p = pos_w[:, idx, :]
    q = quat_w[:, idx, :]
    centre = p + math_utils.quat_apply(q.reshape(-1, 4), off.unsqueeze(0).expand(n, K, 3).reshape(-1, 3)).view(n, K, 3)
    xy = (centre[:, :, :2] - bolt).reshape(-1, 2)                                 # (n*K, 2)
    z_top = (centre[:, :, 2] + rad.unsqueeze(0)).reshape(-1) - G.TABLE_TOP        # sphere top above the table
    r = rad.unsqueeze(0).expand(n, K).reshape(-1)
    d2 = ((cells.unsqueeze(0) - xy.unsqueeze(1)) ** 2).sum(dim=2)                 # (n*K, M*M)
    inside = d2 <= (r ** 2).unsqueeze(1)
    occ.copy_(occ | inside.any(dim=0))
    hmax.copy_(torch.maximum(hmax, torch.where(inside, z_top.unsqueeze(1), torch.zeros_like(d2)).max(dim=0).values))


done_count, steps = 0, 0
obs = w.get_observations()
with torch.inference_mode():
    while done_count < args.episodes:
        act = policy(obs)
        obs, _, dones, _ = w.step(act)
        stamp()
        steps += 1
        done_count += int(dones.sum())

occ_np = occ.view(M, M).cpu().numpy()
h_np = hmax.view(M, M).cpu().numpy()
out = args.out or os.path.join(os.path.dirname(G.__file__), "..", "tasks", "wuji_reach_grasp", "corridor.npz")
out = os.path.abspath(out)
np.savez(out, occ=occ_np, hmax=h_np, half_extent=H, cell=cell, level=cfg.actions.pipeline.level,
         episodes=done_count, checkpoint=args.checkpoint_path)
area = occ_np.sum() * cell * cell
print(f"corridor: {done_count} episodes, {steps} steps, grid {M}x{M} at {cell*100:.1f} cm, "
      f"footprint {area*1e4:.0f} cm^2, written to {out}")
# a coarse picture, 2 cm per character, bolt at the centre, x up, y right
step = int(round(0.02 / cell))
print("  occupancy (2 cm/char, '#' swept, '.' free, 'B' bolt):")
for i in range(M - 1, -1, -step):
    row = ""
    for j in range(0, M, step):
        blk = occ_np[i:i + step, j:j + step].any()
        ib, jb = M // 2, M // 2
        row += "B" if (abs(i - ib) < step and abs(j - jb) < step) else ("#" if blk else ".")
    print("   " + row)
env.close()
app.close()
