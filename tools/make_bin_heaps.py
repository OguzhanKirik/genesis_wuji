#!/usr/bin/env python3
"""Settled full-bin heaps, dropped the way adam-moveit's isaac_runner fills its bin.

    docker compose exec -T adam_rl python /opt/tools/make_bin_heaps.py [--heaps 2048] [--num_envs 512]

adam-moveit's workcell.json bin: inner 22 x 16 x 8 cm, 40-60 rods on a 5 x 6 x 2 grid (3.4 x 2.8 cm pitch,
8 mm jitter, clamped 1.6 cm inside the walls), dropped from 6 cm over the rim plus 4 cm per layer plus up to 4 cm,
yawed index * 360 / count +/- 12 deg, tilted +/- 18 deg about x and y (the rod's axis is its local z, so they
fall end-down), with a small random linear and angular velocity (isaac_runner._scatter_one). Here: one bin per
env, the table as its floor (adam-moveit's 8 mm floor plate is left out), the mesh's 64-vertex hull
(adam-moveit's convexHull; --collider), 4 s of settling at 120 Hz. Run hull at --num_envs 256: 512 bins of mesh rods
held 123 GB of host RAM. The training env places the neighbours as cylinders at these poses.

Writes genesis_rl/tasks/wuji_reach_grasp/bin_heaps.npz: pos (H, 60, 3) relative to the bin's inner centre on the
table top, quat (H, 60, 4) xyzw, count (H,), and the bin it was made in. Rods past `count` are unused.
Heaps with a rod outside the bin or still moving are dropped.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--heaps", type=int, default=2048)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--settle_s", type=float, default=4.0)
parser.add_argument("--count", type=int, nargs=2, default=(40, 60), help="rods per bin, min max (workcell.json count_min/max)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--collider", choices=["hull", "cyl"], default="hull",
                    help="hull: the mesh's 64-vertex hull, adam-moveit's convexHull (its uneven end topples: 29 %% of rods settle "
                         "standing; ~0.25 GB host RAM per bin, 256 bins per batch); cyl: a flat-ended cylinder (64 %% standing)")
parser.add_argument("--resettle", default=None,
                    help="a heap file: place its heaps (cylinders by default with this) and settle them again instead of dropping; "
                         "hull heaps resettled as cylinders keep adam-moveit's lying/leaning/standing mix and match the training "
                         "env's cylinder neighbours (placed straight, hull poses collapse: 35 mm median target drift)")
parser.add_argument("--out", default="genesis_rl/tasks/wuji_reach_grasp/bin_heaps.npz")
args, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg  # noqa: E402
from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg  # noqa: E402

import genesis_rl.tasks.wuji_reach_grasp.env_cfg as E  # noqa: E402
from genesis_rl.tasks.wuji_pipeline import env_cfg as P  # noqa: E402
from genesis_rl.tasks.wuji_reach.env_cfg import LAYOUT, _t  # noqa: E402

# adam-moveit ws/src/genesis_isaac/assets/workcell/workcell.json
INNER = (0.22, 0.16, 0.08)
WALL = 0.008
ROWS, COLS, LAYERS = 5, 6, 2
SPACING = (0.034, 0.028)            # x along the columns, y along the rows
LAYER_H, DROP, JITTER, TILT_DEG = 0.04, 0.06, 0.008, 18.0
MARGIN = 0.016                      # isaac_runner: the spawn xy is clamped this far inside the walls
NMAX = ROWS * COLS * LAYERS
D = "cuda:0"

torch.manual_seed(args.seed)
CX, CY = E.BOLT_XY                  # the bin centre, env frame (only the relative poses are saved)
TOP = E.TABLE_TOP
RIM = TOP + INNER[2]

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
    dt=1.0 / 120.0, device=D, physics=PhysxCfg(gpu_max_rigid_patch_count=2**21, gpu_max_rigid_contact_count=2**24,
                                                 gpu_found_lost_pairs_capacity=2**23, gpu_total_aggregate_pairs_capacity=2**23)))
cfg = InteractiveSceneCfg(num_envs=args.num_envs, env_spacing=1.5)
cfg.ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
cfg.light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
cfg.table = P._static_box("Table", LAYOUT["table_center"], LAYOUT["table_size"], LAYOUT["table_color"])
ix, iy, iz = INNER
walls = [((CX + (ix + WALL) / 2, CY, TOP + iz / 2), (WALL, iy + 2 * WALL, iz)),
         ((CX - (ix + WALL) / 2, CY, TOP + iz / 2), (WALL, iy + 2 * WALL, iz)),
         ((CX, CY + (iy + WALL) / 2, TOP + iz / 2), (ix, WALL, iz)),
         ((CX, CY - (iy + WALL) / 2, TOP + iz / 2), (ix, WALL, iz))]
for j, (c, size) in enumerate(walls):
    w = E._obstacle_cfg(j, size, "Wall", (0.4, 0.4, 0.4))
    w.init_state.pos = c
    setattr(cfg, f"wall_{j}", w)
for k in range(NMAX):
    setattr(cfg, f"rod_{k}", E._neighbor_cfg(k, args.collider))
scene = InteractiveScene(cfg)
sim.reset()
N = args.num_envs
org = scene.env_origins
rods = [scene[f"rod_{k}"] for k in range(NMAX)]
park = torch.tensor(E.NEIGHBOR_PARK, device=D)


def drop():
    """isaac_runner._scatter_one for every env, vectorised. Returns the count per env."""
    count = torch.randint(args.count[0], args.count[1] + 1, (N,), device=D)
    for i, rod in enumerate(rods):
        layer, r, c = i // (ROWS * COLS), (i % (ROWS * COLS)) // COLS, i % COLS
        u = lambda lo, hi: torch.rand(N, device=D) * (hi - lo) + lo  # noqa: E731
        x = CX + (c - (COLS - 1) / 2) * SPACING[0] + u(-JITTER, JITTER)
        y = CY + (r - (ROWS - 1) / 2) * SPACING[1] + u(-JITTER, JITTER)
        x = x.clamp(CX - ix / 2 + MARGIN, CX + ix / 2 - MARGIN)
        y = y.clamp(CY - iy / 2 + MARGIN, CY + iy / 2 - MARGIN)
        z = RIM + DROP + layer * LAYER_H + u(0.0, 0.04)
        yaw = torch.deg2rad(i * 360.0 / count.float() + u(-12.0, 12.0))
        q = math_utils.quat_from_euler_xyz(torch.deg2rad(u(-TILT_DEG, TILT_DEG)), torch.deg2rad(u(-TILT_DEG, TILT_DEG)), yaw)
        vel = torch.stack([u(-0.03, 0.03), u(-0.03, 0.03), u(-0.15, 0.0),
                           torch.deg2rad(u(-120, 120)), torch.deg2rad(u(-120, 120)), torch.deg2rad(u(-240, 240))], dim=1)
        on = i < count
        pos = torch.stack([x, y, z], dim=1)
        pos = torch.where(on.unsqueeze(1), pos, park + torch.tensor([0.0, -0.03 * (i % 20), 0.0], device=D)
                          + torch.tensor([0.03 * (i // 20), 0.0, 0.0], device=D))
        q = torch.where(on.unsqueeze(1), q, torch.tensor(E._yawed_lying_rot_tuple(), device=D).expand(N, 4))
        vel = torch.where(on.unsqueeze(1), vel, torch.zeros_like(vel))
        ids = torch.arange(N, device=D)
        rod.write_root_pose_to_sim_index(root_pose=torch.cat([pos + org, q], dim=1), env_ids=ids)
        rod.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=ids)
    return count


SRC = None
if args.resettle:
    _z = np.load(args.resettle)
    SRC = dict(pos=torch.tensor(_z["pos"], device=D), quat=torch.tensor(_z["quat"], device=D), count=torch.tensor(_z["count"], device=D))
    args.heaps = min(args.heaps, SRC["count"].shape[0])
    _next = [0]


def place():
    """--resettle: the next N heaps of the source file as they settled, at rest. Envs past its end repeat heap 0."""
    i0 = _next[0]
    idx = (torch.arange(N, device=D) + i0).clamp(max=SRC["count"].shape[0] - 1)
    _next[0] += N
    count = SRC["count"][idx].long()
    ids = torch.arange(N, device=D)
    for i, rod in enumerate(rods):
        on = i < count
        pos = SRC["pos"][idx, i] + torch.tensor([CX, CY, TOP], device=D)
        pos = torch.where(on.unsqueeze(1), pos, park + torch.tensor([0.03 * (i // 20), -0.03 * (i % 20), 0.0], device=D))
        q = torch.where(on.unsqueeze(1), SRC["quat"][idx, i], torch.tensor(E._yawed_lying_rot_tuple(), device=D).expand(N, 4))
        rod.write_root_pose_to_sim_index(root_pose=torch.cat([pos + org, q], dim=1), env_ids=ids)
        rod.write_root_velocity_to_sim_index(root_velocity=torch.zeros(N, 6, device=D), env_ids=ids)
    return count


out_pos, out_quat, out_cnt = [], [], []
dt = sim.get_physics_dt()
stats = dict(lying=0, leaning=0, standing=0, rods=0, dropped_out=0, moving=0, kept=0, made=0)
while len(out_cnt) < args.heaps:
    if SRC is not None and _next[0] >= args.heaps:
        break
    count = place() if SRC is not None else drop()
    scene.write_data_to_sim()
    for _ in range(int(args.settle_s / dt)):
        sim.step(render=False)
        scene.update(dt)
    pos = torch.stack([_t(r.data.root_pos_w) for r in rods], dim=1) - org.unsqueeze(1)     # (N, 60, 3)
    quat = torch.stack([_t(r.data.root_quat_w) for r in rods], dim=1)
    vel = torch.stack([_t(r.data.root_lin_vel_w) for r in rods], dim=1)
    on = torch.arange(NMAX, device=D).unsqueeze(0) < count.unsqueeze(1)
    rel = pos - torch.tensor([CX, CY, TOP], device=D)
    inside = (rel[..., 0].abs() < ix / 2) & (rel[..., 1].abs() < iy / 2) & (rel[..., 2] > 0.0) & (rel[..., 2] < iz)
    still = torch.norm(vel, dim=2) < 0.02
    ok_out = (inside | ~on).all(dim=1)
    ok_still = (still | ~on).all(dim=1)
    ok = ok_out & ok_still
    ax = math_utils.quat_apply(quat.reshape(-1, 4), torch.tensor([0.0, 0.0, 1.0], device=D).expand(N * NMAX, 3)).view(N, NMAX, 3)
    elev = torch.rad2deg(torch.asin(ax[..., 2].abs().clamp(max=1.0)))                        # axis above horizontal
    m = on & ok.unsqueeze(1)
    stats["lying"] += int((m & (elev < 20)).sum()); stats["leaning"] += int((m & (elev >= 20) & (elev < 60)).sum())
    stats["standing"] += int((m & (elev >= 60)).sum()); stats["rods"] += int(m.sum())
    stats["dropped_out"] += int((~ok_out).sum()); stats["moving"] += int((ok_out & ~ok_still).sum())
    stats["made"] += N; stats["kept"] += int(ok.sum())
    for i in torch.nonzero(ok).flatten().tolist():
        out_pos.append(rel[i].cpu().numpy()); out_quat.append(quat[i].cpu().numpy()); out_cnt.append(int(count[i]))
    print(f"batch: kept {int(ok.sum())}/{N} (rod out of the bin {int((~ok_out).sum())}, still moving {int((ok_out & ~ok_still).sum())}); "
          f"total {len(out_cnt)}", flush=True)

H = min(len(out_cnt), args.heaps)
np.savez_compressed(args.out, pos=np.stack(out_pos[:H]).astype(np.float32), quat=np.stack(out_quat[:H]).astype(np.float32),
                    count=np.array(out_cnt[:H], dtype=np.int32), inner=np.array(INNER), wall=WALL)
R = max(stats["rods"], 1)
print(f"\nwrote {args.out}: {H} heaps ({stats['kept']}/{stats['made']} kept)")
print(f"settled rods: lying (< 20 deg) {100 * stats['lying'] / R:.1f} %, leaning (20-60) {100 * stats['leaning'] / R:.1f} %, "
      f"standing (> 60) {100 * stats['standing'] / R:.1f} %")
app.close()
