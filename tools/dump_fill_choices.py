"""The full bin's target choices, for checking genesis_policy's selector against them (levels with FillCfg.pick "highest").

    docker compose exec -T -e ADAM_U_RG_LEVEL=52 adam_rl python /opt/tools/dump_fill_choices.py [--num_envs 256] [--resets 4]

Per heap episode: every rod of the placed heap (world = env frame, the target included at its index), its axis, the
heap's rod count and the index the env chose. adam-moveit: test_selector.py's parity test runs the selector on the
same rods (GENESIS_POLICY_FILL=<npz> GENESIS_POLICY_LEVEL=<level.yaml>) and expects the same index.
"""
import argparse
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--resets", type=int, default=4)
parser.add_argument("--out", default="logs/fill_choices.npz")
from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import genesis_rl.tasks.wuji_reach_grasp.env_cfg as E  # noqa: E402

lvl = E.LEVELS[E.LEVEL]
assert lvl.fill.active and lvl.fill.pick == "highest", f"level {E.LEVEL} does not pick the highest rod"
cfg = E.WujiReachGraspEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.sim.device = "cuda:0"
cfg.events.reset_bolt.params["fill_fixed"] = True
gym.register(id="DumpFill", entry_point="genesis_rl.envs.wuji_reach_grasp_env_cfg:ReachGraspEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": E.WujiReachGraspEnvCfg})
env = gym.make("DumpFill", cfg=cfg); u = env.unwrapped
b = E._rg(u)
ids = torch.arange(args.num_envs, device="cuda:0")
P, A, C, J = [], [], [], []
for r in range(args.resets):
    E.reset_bolt_pose(u, ids, fill_fixed=True)            # the draw alone: it fills the heap buffers
    on = b["_fill"]
    P.append(b["_fill_pos"][on].cpu().numpy()); A.append(b["_fill_ax"][on].cpu().numpy())
    C.append(b["_fill_cnt"][on].cpu().numpy()); J.append(b["_fill_j"][on].cpu().numpy())
np.savez(args.out, pos=np.concatenate(P), axis=np.concatenate(A), count=np.concatenate(C), chosen=np.concatenate(J), level=E.LEVEL)
print(f"wrote {args.out}: {sum(len(c) for c in C)} heaps")
app.close()
