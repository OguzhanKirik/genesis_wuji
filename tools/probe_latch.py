"""Which latch test blocks the grasps that hold the rod (hw_latch levels).

    python /opt/tools/probe_latch.py --checkpoint_path <ck> [--num_envs 128] [--episodes 256]

Wraps grasp_ready: on every grasp-phase step of every env it records the latch's tests (torque
streak, gripping, inside, centred and settled on the ESTIMATE as the latch reads them, and the same
two on the TRUE rod for comparison). At each episode's end it classifies the episode: latched,
or never latched but held the rod through the hold (a lost success), or neither. For the lost
ones it reports, over the grasp steps where the torque grip held, how often each test failed and
how often it was the ONLY failing test.

Run with ADAM_U_RG_LEVEL set (39).
"""
import argparse
import sys
from importlib import metadata

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=256)
parser.add_argument("--cover", type=int, choices=[0, 1], default=None)
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

lvl = E.LEVELS[E.LEVEL]
assert lvl.hw_latch, "probe_latch reads the hw_latch tests"
dev = "cuda:0"
cfg = E.WujiReachGraspEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.sim.device = dev
if args.cover is not None:
    cfg.events.reset_bolt.params["cover_fixed"] = bool(args.cover)
gym.register(id="ProbeLatch", entry_point="genesis_rl.envs.wuji_reach_grasp_env_cfg:ReachGraspEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": E.WujiReachGraspEnvCfg})
env = gym.make("ProbeLatch", cfg=cfg); u = env.unwrapped
agent_cfg = RslRlWujiReachGraspPPORunnerCfg(); agent_cfg.device = dev
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(wenv, agent_cfg.to_dict(), log_dir=None, device=dev); runner.load(args.checkpoint_path)
policy = runner.get_inference_policy(device=dev)

N = args.num_envs
TESTS = ("stable", "gripping", "inside", "centred", "settled", "force_ok", "valid")
# per env, this episode: grasp steps with the grip held, failures per test, sole failures per test
acc = {k: torch.zeros(N, device=dev) for k in ("grip_steps",) + tuple(f"fail_{t}" for t in TESTS) + tuple(f"only_{t}" for t in TESTS)}
extra = {k: torch.zeros(N, device=dev) for k in ("centred_true_ok", "settled_true_ok", "d_est_sum", "d_true_sum", "v_est_sum", "v_true_sum")}
_orig = E.grasp_ready


def grasp_ready_probe(env_):
    b = E._update(env_)
    t = env_.episode_length_buf.float()
    grasping = (t < b["grasp_end"]) & ~b["latched"] & (t >= 1)
    ts = b["tau_streak"] >= E.STABLE_STEPS
    stable = ts[:, 0] & (ts[:, 1] | ts[:, 2])
    tau = E._grip_torque(env_)
    gripping = (tau[:, 0] > E.GRIP_TAU_MIN) & (torch.maximum(tau[:, 1], tau[:, 2]) > E.GRIP_TAU_MIN)
    d = E.fingertip_distances_est(env_)
    inside = (0.5 * (d[:, 0] + torch.minimum(d[:, 1], d[:, 2]))) < E.GRASP_REGION
    d_est = E.pinch_centre_dist_est(env_)
    d_true = E.pinch_centre_dist(env_)
    centred = d_est <= lvl.pinch_centre_tol
    ep_, _ = E._est_bolt(env_)
    w = lvl.est_vel_window
    prev = b["_est_win"][:, w - 1] if w > 1 else b["_est_prev"]          # as grasp_ready reads it, before it updates
    v_est = torch.norm((ep_[:, :3] - prev) * (E.CONTROL_HZ / w) - E.wrist_lin_vel(env_), dim=1)
    v_true = torch.norm(E._t(env_.scene["object"].data.root_lin_vel_w) - E.wrist_lin_vel(env_), dim=1)
    settled = v_est < E.REL_VEL_MAX
    force_ok = b["tip_force"].max(dim=1).values < E.F_MAX
    valid = ~b["invalid"]
    ok = dict(stable=stable, gripping=gripping, inside=inside, centred=centred, settled=settled, force_ok=force_ok, valid=valid)
    m = grasping & gripping                                   # the steps where a latch was plausible
    acc["grip_steps"] += m.float()
    nfail = sum((~v).float() for v in ok.values())
    for k, v in ok.items():
        acc[f"fail_{k}"] += (m & ~v).float()
        acc[f"only_{k}"] += (m & ~v & (nfail == 1)).float()
    extra["centred_true_ok"] += (m & (d_true <= lvl.pinch_centre_tol)).float()
    extra["settled_true_ok"] += (m & (v_true < E.REL_VEL_MAX)).float()
    extra["d_est_sum"] += torch.where(m, d_est, torch.zeros_like(d_est)); extra["d_true_sum"] += torch.where(m, d_true, torch.zeros_like(d_true))
    extra["v_est_sum"] += torch.where(m, v_est, torch.zeros_like(v_est)); extra["v_true_sum"] += torch.where(m, v_true, torch.zeros_like(v_true))
    return _orig(env_)


E.grasp_ready = grasp_ready_probe

cls = {"latched": 0, "lost": 0, "other": 0}
tot = {k: 0.0 for k in list(acc) + list(extra)}
obs = wenv.get_observations(); done_n = 0; warm = torch.zeros(N, dtype=torch.bool, device=dev)
b = E._rg(u)
with torch.inference_mode():
    while done_n < args.episodes:
        act = policy(obs)
        # the episode's outcome is in last_* / the buffers before the reset inside step(): snapshot now
        latched_pre = b["latched"].clone(); hold_pre = b["hold_streak"].clone()
        obs, _, dones, _ = wenv.step(act)
        d = dones.bool()
        if bool(d.any()):
            for i in torch.nonzero(d).flatten().tolist():
                if warm[i]:
                    held = bool(hold_pre[i] >= E.HOLD_STEPS - 1)
                    c = "latched" if bool(latched_pre[i]) else ("lost" if held else "other")
                    cls[c] += 1; done_n += 1
                    if c == "lost":
                        for k in tot:
                            tot[k] += float((acc if k in acc else extra)[k][i])
                warm[i] = True
                for v in list(acc.values()) + list(extra.values()):
                    v[i] = 0.0

n = sum(cls.values())
print(f"\nlevel {E.LEVEL} '{lvl.name}', {args.checkpoint_path}, {n} episodes")
print(f"  latched {100 * cls['latched'] / n:.1f} %   never latched but held the rod (lost) {100 * cls['lost'] / n:.1f} %   neither {100 * cls['other'] / n:.1f} %")
g = max(tot["grip_steps"], 1.0)
print(f"  lost episodes: {tot['grip_steps']:.0f} grasp steps with the torque grip held; per test, failing / the only failing test:")
for t in TESTS:
    print(f"    {t:9s} {100 * tot[f'fail_{t}'] / g:5.1f} %   only {100 * tot[f'only_{t}'] / g:5.1f} %")
print(f"  on those steps: pinch to ESTIMATED axis mean {1000 * tot['d_est_sum'] / g:.1f} mm, to TRUE axis {1000 * tot['d_true_sum'] / g:.1f} mm "
      f"(tol {1000 * lvl.pinch_centre_tol:.0f}); centred on the true rod {100 * tot['centred_true_ok'] / g:.1f} %")
print(f"  relative speed, estimate {tot['v_est_sum'] / g:.3f} m/s, true {tot['v_true_sum'] / g:.3f} m/s (max {E.REL_VEL_MAX}); "
      f"settled on the true rod {100 * tot['settled_true_ok'] / g:.1f} %")
app.close()
