"""The full-bin scene (levels 50+) before any training: what the reset builds and how far it moves while it settles.

    docker compose exec -T -e ADAM_U_RG_LEVEL=51 adam_rl python /opt/tools/probe_fill.py [--num_envs 256] [--steps 40]

Holds the arm still (zero actions; the settle-in freezes the policy anyway) and reports the share of heap episodes,
the target kinds, the rods in the bin, the target's drift between the reset and the end of the settle-in (the
start pose was solved for the reset pose), the rods that left the bin, and the step time.
"""
import argparse
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=40)
from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import genesis_rl.tasks.wuji_reach_grasp.env_cfg as E  # noqa: E402

dev = "cuda:0"
lvl = E.LEVELS[E.LEVEL]
assert lvl.fill.active, f"level {E.LEVEL} has no full bin"
cfg = E.WujiReachGraspEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.sim.device = dev
gym.register(id="ProbeFill", entry_point="genesis_rl.envs.wuji_reach_grasp_env_cfg:ReachGraspEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": E.WujiReachGraspEnvCfg})
env = gym.make("ProbeFill", cfg=cfg); u = env.unwrapped
b = E._rg(u)
N = args.num_envs
obs, _ = env.reset()
obj = u.scene["object"]
org = u.scene.env_origins
p0 = E._t(obj.data.root_pos_w).clone() - org
fill = b["_fill"].clone()
kind = torch.where(b["_upright"], 2, torch.where(b["_tilted"], 1, 0))
nb = [u.scene[f"neighbor_{k}"] for k in range(len(E.NEIGHBOR_SLOTS)) if f"neighbor_{k}" in u.scene.keys()]
in_bin = lambda p: ((p[..., 0] - E.BOLT_XY[0]).abs() < lvl.bin.inner[0] / 2) & ((p[..., 1] - E.BOLT_XY[1]).abs() < lvl.bin.inner[1] / 2)  # noqa: E731
nbp0 = torch.stack([E._t(a.data.root_pos_w) for a in nb], dim=1) - org.unsqueeze(1)
rods0 = in_bin(nbp0).sum(dim=1)
act = torch.zeros(N, env.action_space.shape[-1], device=dev)
s_in = lvl.pile.settle_in
p_settled = None
t0 = time.time()
with torch.inference_mode():
    for t in range(args.steps):
        obs, *_ = env.step(act)
        if t == s_in - 1:
            p_settled = E._t(obj.data.root_pos_w).clone() - org
dt = (time.time() - t0) / args.steps
nbp1 = torch.stack([E._t(a.data.root_pos_w) for a in nb], dim=1) - org.unsqueeze(1)
f = fill.float()
print(f"\nlevel {E.LEVEL} ({lvl.name}), {N} envs, heaps from {lvl.fill.files or lvl.fill.file}, pick {lvl.fill.pick}")
print(f"  heap episodes {100 * f.mean():.1f} % (level: {100 * lvl.fill.prob:.0f} %)")
for k, name in enumerate(("lying", "leaning", "standing")):
    m = fill & (kind == k)
    print(f"    target {name:9s} {100 * m.float().sum() / f.sum().clamp(min=1):5.1f} %   mean height above the table "
          f"{1000 * (p0[m, 2] - E.TABLE_TOP).mean():5.1f} mm" if bool(m.any()) else f"    target {name:9s}   0.0 %")
print(f"  neighbour rods in the bin: mean {rods0[fill].float().mean():.1f}, min {int(rods0[fill].min())}, max {int(rods0[fill].max())}")
if p_settled is not None:
    dr = torch.norm(p_settled - p0, dim=1)
    print(f"  target drift over the {s_in}-step settle-in (heap episodes): median {1000 * dr[fill].median():.1f} mm, "
          f"p90 {1000 * dr[fill].quantile(0.9):.1f} mm, > 10 mm {100 * (dr[fill] > 0.01).float().mean():.1f} %")
    print(f"                                       (other episodes): median {1000 * dr[~fill].median():.1f} mm" if bool((~fill).any()) else "")
left = (in_bin(nbp0) & ~in_bin(nbp1)).sum(dim=1)
print(f"  neighbours that left the bin in {args.steps} steps: {left[fill].float().mean():.2f} per heap episode")
print(f"  {1000 * dt:.0f} ms per control step at {N} envs ({N / dt:.0f} env-steps/s)")
app.close()
