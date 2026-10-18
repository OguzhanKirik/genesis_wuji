"""Which link causes the obstacle collisions that disqualify episodes.

    python /opt/tools/probe_collisions.py --checkpoint_path <ck> [--num_envs 128] [--episodes 256] [--bed 1]

Per step it recomputes every proxy sphere's surface distance to the non-table obstacles the obstacle test
reads (boxes after the table, hard capsules; G._clearance's obs_d), and when an env's episode is first
disqualified by an obstacle hit (invalid_which == 4) it records the link of the deepest sphere. Also counts,
per link, the env-steps with that link more than OBSTACLE_PEN_M inside an obstacle. Run with ADAM_U_RG_LEVEL set.
"""
import argparse
import sys
from collections import Counter
from importlib import metadata

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=256)
parser.add_argument("--bed", type=int, choices=[0, 1], default=None)
parser.add_argument("--cover", type=int, choices=[0, 1], default=None)
parser.add_argument("--across", type=int, choices=[0, 1], default=None)
from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from genesis_rl.algos.rsl_rl.rsl_rl_ppo_cfg import RslRlWujiReachGraspPPORunnerCfg  # noqa: E402
import genesis_rl.tasks.wuji_reach_grasp.env_cfg as E  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402

dev = "cuda:0"
cfg = E.WujiReachGraspEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.sim.device = dev
if args.bed is not None:
    cfg.events.reset_bolt.params["bed_fixed"] = bool(args.bed)
if args.across is not None:
    cfg.events.pile.params["across_fixed"] = bool(args.across)
if args.cover is not None:
    cfg.events.reset_bolt.params["cover_fixed"] = bool(args.cover)
gym.register(id="ProbeCol", entry_point="genesis_rl.envs.wuji_reach_grasp_env_cfg:ReachGraspEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": E.WujiReachGraspEnvCfg})
env = gym.make("ProbeCol", cfg=cfg); u = env.unwrapped
agent_cfg = RslRlWujiReachGraspPPORunnerCfg(); agent_cfg.device = dev
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(wenv, agent_cfg.to_dict(), log_dir=None, device=dev); runner.load(args.checkpoint_path)
policy = runner.get_inference_policy(device=dev)

N = args.num_envs
b = E._rg(u)
robot = u.scene["robot"]
# the proxy spheres, flattened over the groups, with their link names
sph_link, sph_idx, sph_off, sph_rad = [], [], [], []
for g in E.GROUP_NAMES:
    for link, off, rad in E.PROXY_GROUPS[g]:
        sph_link.append(f"{g}:{link}")
        sph_idx.append(robot.body_names.index(link)); sph_off.append(off); sph_rad.append(rad)
sph_idx = torch.tensor(sph_idx, device=dev); sph_off = torch.tensor(sph_off, device=dev); sph_rad = torch.tensor(sph_rad, device=dev)
K = len(sph_link)


def sphere_obstacle_dist():
    """(N, K) surface distance of every proxy sphere to the nearest non-table hard obstacle."""
    pos = E._t(robot.data.body_link_pos_w)[:, sph_idx, :] - u.scene.env_origins.unsqueeze(1)
    quat = E._t(robot.data.body_link_quat_w)[:, sph_idx, :]
    centre = pos + math_utils.quat_apply(quat.reshape(-1, 4), sph_off.unsqueeze(0).expand(N, K, 3).reshape(-1, 3)).view(N, K, 3)
    cc = centre.unsqueeze(2)
    lo, hi = b["_obs_lo"], b["_obs_hi"]                                         # obstacle slots (the table is not among them)
    db = torch.norm(torch.maximum(torch.minimum(cc, hi.unsqueeze(1)), lo.unsqueeze(1)) - cc, dim=3)   # (N, K, B)
    p0, p1, cr = b["_cap_p0"], b["_cap_p1"], b["_cap_r"]
    seg = p1 - p0
    w = cc - p0.unsqueeze(1)
    tt = ((w * seg.unsqueeze(1)).sum(dim=3) / (seg * seg).sum(dim=2).clamp(min=1e-9).unsqueeze(1)).clamp(0.0, 1.0)
    dc = torch.norm(p0.unsqueeze(1) + tt.unsqueeze(3) * seg.unsqueeze(1) - cc, dim=3) - cr.unsqueeze(1)
    dc = torch.where(b["_cap_hard"].unsqueeze(1), dc, torch.full_like(dc, 10.0))    # soft capsules (pushable) excluded
    d = torch.cat([db, dc], dim=2).min(dim=2).values - sph_rad.view(1, K)
    return d


# An obstacle hit terminates the episode and the env resets inside the same step(): take the sphere distances
# inside the termination check, before the reset, and the outcome from last_invalid* after it.
_term = E.invalid_collision_termination
snap = {}


def term_probe(env_, *a, **kw):
    snap["d"] = sphere_obstacle_dist()
    return _term(env_, *a, **kw)


E.invalid_collision_termination = term_probe
cfg_terms = u.termination_manager._term_cfgs[u.termination_manager._term_names.index("invalid_collision")]
cfg_terms.func = term_probe

first_hit = Counter()
ep_min = torch.full((N, K), 10.0, device=dev)       # this episode's closest approach per sphere
link_names = sorted({l.split(":")[1] for l in sph_link})
sph_of = {ln: [k for k, l in enumerate(sph_link) if l.split(":")[1] == ln] for ln in link_names}
closest = {ln: [] for ln in link_names}             # per ended episode, the link's closest approach (m)
pen_steps = Counter()
episodes = hits = 0
prev_invalid = torch.zeros(N, dtype=torch.bool, device=dev)
warm = torch.zeros(N, dtype=torch.bool, device=dev)
obs = wenv.get_observations()
with torch.inference_mode():
    while episodes < args.episodes:
        act = policy(obs)
        obs, _, dones, _ = wenv.step(act)
        d = snap["d"]
        deep = d < -E.OBSTACLE_PEN_M
        for k in torch.nonzero(deep.any(dim=0)).flatten().tolist():
            pen_steps[sph_link[k]] += int(deep[:, k].sum())
        ep_min = torch.minimum(ep_min, d)
        done = dones.bool()
        for i in torch.nonzero(done & warm).flatten().tolist():
            for ln, ks in sph_of.items():
                closest[ln].append(float(ep_min[i, ks].min()))
        ep_min[done] = 10.0
        fresh = done & warm & b["last_invalid"] & (b["last_invalid_which"] == 4)
        for i in torch.nonzero(fresh).flatten().tolist():
            first_hit[sph_link[int(d[i].argmin())]] += 1
            hits += 1
        episodes += int((done & warm).sum())
        warm |= done

print(f"\nlevel {E.LEVEL}, {args.checkpoint_path}, bed={args.bed}: {episodes} episodes, {hits} obstacle disqualifications "
      f"({100 * hits / max(episodes, 1):.1f} %)")
print("  the deepest sphere at the disqualifying step, by link:")
for k, v in first_hit.most_common(12):
    print(f"    {k:40s} {v:4d}  ({100 * v / max(hits, 1):4.1f} %)")
print(f"  env-steps with a sphere > {1000 * E.OBSTACLE_PEN_M:.0f} mm inside an obstacle, by link:")
for k, v in pen_steps.most_common(10):
    print(f"    {k:40s} {v:6d}")
print("  closest approach to an obstacle per episode, by link (surface to surface; < 0 touching, < -4 mm a hit):")
print(f"    {'link':28s} {'median':>8s} {'p5':>8s} {'touch':>7s} {'hit':>6s}")
rows = []
for ln, v in closest.items():
    t = torch.tensor(v)
    rows.append((float(t.quantile(0.05)), ln, float(t.median()), float((t < 0).float().mean()), float((t < -E.OBSTACLE_PEN_M).float().mean())))
for p5, ln, med, touch, hit in sorted(rows):
    print(f"    {ln:28s} {1000 * med:6.1f}mm {1000 * p5:6.1f}mm {100 * touch:6.1f}% {100 * hit:5.1f}%")
app.close()
