"""How often the grip check (sense_est levels) sends a GOOD grasp into a retry.

    python /opt/tools/probe_gripcheck.py --checkpoint_path <ck> [--num_envs 128] [--checks 512]

At every grip check (the end of a latched pre-lift's test lift, v3 timing) it records the fingertips'
rise, the ESTIMATED rod's rise (what the check reads at sense_est levels) and the TRUE rod's rise,
and whether the check fired. A check that fired while the true rod rose at least GRIP_CHECK_RATIO of
the tips is a false retry: the rod was in the hand. Run with ADAM_U_RG_LEVEL set.
"""
import argparse
import sys
from importlib import metadata

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--checks", type=int, default=512)
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
assert lvl.grip_check and lvl.free_prelift and not lvl.grip_check_late, "written for the v3 check at free-pre-lift levels"
dev = "cuda:0"
cfg = E.WujiReachGraspEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.sim.device = dev
gym.register(id="ProbeGrip", entry_point="genesis_rl.envs.wuji_reach_grasp_env_cfg:ReachGraspEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": E.WujiReachGraspEnvCfg})
env = gym.make("ProbeGrip", cfg=cfg); u = env.unwrapped
agent_cfg = RslRlWujiReachGraspPPORunnerCfg(); agent_cfg.device = dev
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(wenv, agent_cfg.to_dict(), log_dir=None, device=dev); runner.load(args.checkpoint_path)
policy = runner.get_inference_policy(device=dev)

N = args.num_envs
b = E._rg(u)
start_off = E.SETTLE_STEPS + E.DRAG_STEPS          # the check's start, after the grasp end
check_off = E.LIFT_START
pre = {k: torch.zeros(N, device=dev) for k in ("tips", "est", "true", "depth", "ct", "cf", "fth", "ffi")}
# depth and the weaker side's tip force at the latch and at the settle's end, the rod in hand or not
stage = {k: torch.zeros(N, device=dev) for k in ("d_latch", "f_latch", "d_settle", "f_settle")}
check_at = torch.full((N,), -1.0, device=dev)
rows = []                                           # (tips rise, est rise, true rise, fired, tilt, depth, c_t, c_f, f_thumb, f_finger)
term = u.action_manager._terms["pipeline"]
obs = wenv.get_observations()
with torch.inference_mode():
    while len(rows) < args.checks:
        act = policy(obs)
        att0 = b["attempts"].clone(); latched0 = b["latched"].clone(); g0 = b["grasp_end"].clone()
        obs, _, dones, _ = wenv.step(act)
        # the phase machine (_update) runs after the step's physics and the length increment, on that state
        live = ~dones.bool()
        t = u.episode_length_buf.float()
        tz = torch.stack(E._tip_positions(u), dim=1)[:, :, 2].mean(dim=1)
        true_z = E._t(u.scene["object"].data.root_pos_w)[:, 2] - u.scene.env_origins[:, 2]
        est_z = E._est_bolt(u)[0][:, 2]
        tf_ = b["tip_force"]
        dep = E.pinch_centre_dist(u)
        fmin = torch.minimum(tf_[:, 0], torch.maximum(tf_[:, 1], tf_[:, 2]))
        just = live & b["latched"] & ~latched0                     # latched in this step
        stage["d_latch"] = torch.where(just, dep, stage["d_latch"]); stage["f_latch"] = torch.where(just, fmin, stage["f_latch"])
        st_end = live & latched0 & (t == g0 + E.SETTLE_STEPS)
        stage["d_settle"] = torch.where(st_end, dep, stage["d_settle"]); stage["f_settle"] = torch.where(st_end, fmin, stage["f_settle"])
        start = live & latched0 & (t == g0 + start_off)
        for k, v in (("tips", tz), ("est", est_z), ("true", true_z), ("depth", E.pinch_centre_dist(u)),
                     ("ct", term._closure[:, 0]), ("cf", term._closure[:, 1]), ("fth", tf_[:, 0]),
                     ("ffi", torch.maximum(tf_[:, 1], tf_[:, 2]))):
            pre[k] = torch.where(start, v, pre[k])     # at the start of the test lift, the rod still in hand
        check_at = torch.where(start, g0 + check_off, check_at)
        at = live & (t == check_at)
        if bool(at.any()):
            fired = b["attempts"] > att0
            for i in torch.nonzero(at).flatten().tolist():
                rows.append((float(tz[i] - pre["tips"][i]), float(est_z[i] - pre["est"][i]),
                             float(true_z[i] - pre["true"][i]), bool(fired[i]), float(b["_tilt"][i]),
                             float(pre["depth"][i]), float(pre["ct"][i]), float(pre["cf"][i]), float(pre["fth"][i]),
                             float(pre["ffi"][i]), float(stage["d_latch"][i]), float(stage["f_latch"][i]),
                             float(stage["d_settle"][i]), float(stage["f_settle"][i]), float(dep[i] * 0 + pre["depth"][i]),
                             float(torch.minimum(pre["fth"][i], pre["ffi"][i]))))
            check_at = torch.where(at, torch.full_like(check_at, -1.0), check_at)
        check_at = torch.where(dones.bool(), torch.full_like(check_at, -1.0), check_at)

R = torch.tensor(rows)
tips, est, true, fired = R[:, 0], R[:, 1], R[:, 2], R[:, 3].bool()
valid = tips > E.GRIP_CHECK_MIN_TIPS
good = valid & (true >= E.GRIP_CHECK_RATIO * tips)            # the rod followed the tips
n = len(rows)
print(f"\nlevel {E.LEVEL} '{lvl.name}', {args.checkpoint_path}: {n} grip checks")
print(f"  fired {100 * fired.float().mean():.1f} %;  the true rod followed the tips {100 * good.float().mean():.1f} %")
print(f"  FALSE retries (fired, true rod followed) {100 * (fired & good).float().mean():.1f} % of checks "
      f"= {100 * (fired & good).sum() / max(int(fired.sum()), 1):.0f} % of the firings")
print(f"  missed slips (not fired, true rod did not follow) {100 * (~fired & valid & ~good).float().mean():.1f} % of checks")
print(f"  rise over the test lift, mean: tips {1000 * tips.mean():.1f} mm, true rod {1000 * true.mean():.1f} mm, "
      f"estimate {1000 * est.mean():.1f} mm; |estimate - true| mean {1000 * (est - true).abs().mean():.1f} mm, "
      f"p95 {1000 * (est - true).abs().quantile(0.95):.1f} mm")
tilt, depth, c_t, c_f, f_th, f_fi = (R[:, j] for j in range(4, 10))
slip = valid & ~good
print("  (depth, closures and tip forces at the START of the test lift)")
print("  by scene (tilt at spawn):          checks   slipped   depth hold/slip (mm)   closure t,f hold / slip      tip force th,fi hold / slip (N)")
for name, m in (("lying", tilt < 0.05), ("leaning", (tilt >= 0.05) & (tilt < 1.2)), ("upright", tilt >= 1.2)):
    if int(m.sum()) == 0:
        continue
    h, sl = m & good, m & slip
    f = lambda x, mm: float(x[mm].mean()) if int(mm.sum()) else float("nan")  # noqa: E731
    print(f"    {name:9s} {int(m.sum()):5d}    {100 * f(slip.float(), m):5.1f} %     {1000 * f(depth, h):4.1f} / {1000 * f(depth, sl):4.1f}"
          f"          {f(c_t, h):.2f},{f(c_f, h):.2f} / {f(c_t, sl):.2f},{f(c_f, sl):.2f}"
          f"          {f(f_th, h):4.1f},{f(f_fi, h):4.1f} / {f(f_th, sl):4.1f},{f(f_fi, sl):4.1f}")
dl, fl, ds, fs, dst, fst = (R[:, j] for j in range(10, 16))
print("  along the grasp, mean depth (mm) / weaker side's tip force (N):     latch        settle end    test-lift start")
for name, m in (("held", valid & good), ("slipped", slip)):
    print(f"    {name:8s}  {1000 * dl[m].mean():5.1f} / {fl[m].mean():4.1f}     {1000 * ds[m].mean():5.1f} / {fs[m].mean():4.1f}     "
          f"{1000 * dst[m].mean():5.1f} / {fst[m].mean():4.1f}")
for lo_, hi_ in ((0, 3e-3), (3e-3, 6e-3), (6e-3, 1e-2), (1e-2, 1.0)):
    m = valid & (depth >= lo_) & (depth < hi_)
    if int(m.sum()):
        print(f"  pinch depth {1000 * lo_:4.0f}-{min(1000 * hi_, 99):3.0f} mm: {int(m.sum()):4d} checks, slipped {100 * float(slip[m].float().mean()):5.1f} %")
app.close()
