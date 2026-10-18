"""Which heap rods the policy picks: per full-bin episode, the target rod's situation when the policy is released
(the end of the settle-in) and the episode's outcome.

    docker compose exec -T -e ADAM_U_HELD_CURL=0.95 -e ADAM_U_RG_LEVEL=51 adam_rl python /opt/tools/probe_pickable.py \
        --checkpoint_path <ck> [--num_envs 512] [--episodes 2048] [--out logs/pickable_L51.npz]

Features (true poses): kind (0 lying, 1 leaning, 2 standing), axis elevation, centre height above the table, the free
radial distance around the rod in 8 sectors (heap_map_obs on the true pose: sector 0 horizontal across the axis, 2 up,
4 the other side, 6 down), neighbours touching it (axis gap under 3 mm), neighbours within 6 cm, distance to the nearest
wall, position in the bin, hand yaw and pinch angle. Outcomes: success, latched, obstacle hit, excess force, drop,
double pick, retries. Heap episodes only (--fill 1).
"""
import argparse
import sys
from importlib import metadata

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--episodes", type=int, default=2048)
parser.add_argument("--out", default="logs/pickable.npz")
from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from genesis_rl.algos.rsl_rl.rsl_rl_ppo_cfg import RslRlWujiReachGraspPPORunnerCfg  # noqa: E402
import genesis_rl.tasks.wuji_reach_grasp.env_cfg as E  # noqa: E402

dev = "cuda:0"
lvl = E.LEVELS[E.LEVEL]
assert lvl.fill.active, f"level {E.LEVEL} has no full bin"
cfg = E.WujiReachGraspEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.sim.device = dev
cfg.events.reset_bolt.params["fill_fixed"] = True
gym.register(id="ProbePick", entry_point="genesis_rl.envs.wuji_reach_grasp_env_cfg:ReachGraspEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": E.WujiReachGraspEnvCfg})
env = gym.make("ProbePick", cfg=cfg); u = env.unwrapped
agent_cfg = RslRlWujiReachGraspPPORunnerCfg(); agent_cfg.device = dev
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(wenv, agent_cfg.to_dict(), log_dir=None, device=dev); runner.load(args.checkpoint_path)
policy = runner.get_inference_policy(device=dev)

N = args.num_envs
b = E._rg(u)
s_in = lvl.pile.settle_in
hx, hy = lvl.bin.inner[0] / 2, lvl.bin.inner[1] / 2
FEAT = ["kind", "elev_deg", "height_mm", *[f"free{k}_mm" for k in range(E.HEAP_MAP_DIRS)], "touching", "near6cm",
        "wall_mm", "dx_mm", "dy_mm", "hand_yaw_deg", "pinch_deg", "count"]
OUTC = ["success", "latched", "obstacle", "excess", "drop", "double", "attempts", "invalid"]


def features() -> torch.Tensor:
    """(N, len(FEAT)) for every env, from the true poses now."""
    est = E._est_bolt
    E._est_bolt = lambda env_: E._bolt_true(env_)          # the map on the TRUE pose
    try:
        free = E.heap_map_obs(u) * E.HEAP_MAP_RANGE * 1000.0
    finally:
        E._est_bolt = est
    bp, bq = E._bolt_true(u)
    ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(N, 3)
    ax = math_utils.quat_apply(bq, ez)
    elev = torch.rad2deg(torch.asin(ax[:, 2].abs().clamp(max=1.0)))
    kind = torch.where(b["_upright"], 2, torch.where(b["_tilted"], 1, 0)).float()
    slots = list(E.NEIGHBOR_SLOTS)
    q0, q1, cr = b["_cap_p0"][:, slots], b["_cap_p1"][:, slots], b["_cap_r"][:, slots]
    K = len(slots)
    p0, p1 = bp - E.BOLT_HALF_LEN * ax, bp + E.BOLT_HALF_LEN * ax
    dseg = E._seg_seg_dist(p0.unsqueeze(1).expand(N, K, 3), p1.unsqueeze(1).expand(N, K, 3), q0, q1)
    on = cr > 0
    touching = (on & (dseg < E.BOLT_RADIUS + cr + 0.003)).sum(dim=1).float()
    near = (on & (torch.norm(0.5 * (q0 + q1) - bp.unsqueeze(1), dim=2) < 0.06)).sum(dim=1).float()
    rx, ry = bp[:, 0] - E.BOLT_XY[0], bp[:, 1] - E.BOLT_XY[1]
    wall = torch.minimum(hx - rx.abs(), hy - ry.abs())
    cnt = b["_fill_cnt"].float()
    cols = [kind, elev, 1000 * (bp[:, 2] - E.TABLE_TOP), *free.unbind(1), touching, near, 1000 * wall,
            1000 * rx, 1000 * ry, torch.rad2deg(b["_hand_yaw"]), torch.rad2deg(b["_pinch_delta"]), cnt]
    return torch.stack(cols, dim=1)


feat_buf = torch.zeros(N, len(FEAT), device=dev)
has = torch.zeros(N, dtype=torch.bool, device=dev)
warm = torch.zeros(N, dtype=torch.bool, device=dev)
rows_f, rows_o = [], []
obs = wenv.get_observations()
with torch.inference_mode():
    while len(rows_f) < args.episodes:
        obs, _, dones, _ = wenv.step(policy(obs))
        done = dones.bool()
        end = done & warm & has & b["_fill"]
        # (the reset inside step() already redrew the heap for `done` envs: the outcome is in last_*, the features in feat_buf)
        if bool(end.any()):
            o = torch.stack([b["last_success"].float(), b["last_latched"].float(),
                             (b["last_invalid"] & (b["last_invalid_which"] == 4)).float(), b["last_excess"].float(),
                             b["last_drop"].float(), b["last_double"].float(), b["last_attempts"].float(),
                             b["last_invalid"].float()], dim=1)
            rows_f.append(feat_buf[end].cpu().numpy()); rows_o.append(o[end].cpu().numpy())
        has &= ~done
        warm |= done
        t = u.episode_length_buf
        now = (t == s_in) & ~done
        if bool(now.any()):
            f = features()
            feat_buf = torch.where(now.unsqueeze(1), f, feat_buf)
            has |= now
        n_done = sum(r.shape[0] for r in rows_f)
        if n_done and n_done % 256 < int(end.sum()):
            print(f"{n_done} episodes", flush=True)
        if n_done >= args.episodes:
            break

F = np.concatenate(rows_f)[: args.episodes]
O = np.concatenate(rows_o)[: args.episodes]
np.savez(args.out, F=F, O=O, feat=np.array(FEAT), outc=np.array(OUTC), checkpoint=args.checkpoint_path, level=E.LEVEL)
print(f"wrote {args.out}: {F.shape[0]} heap episodes, success {100 * O[:, 0].mean():.1f} %")
app.close()
