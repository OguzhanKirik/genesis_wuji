#!/usr/bin/env python3
"""Reward audit: prove the shaping cannot be farmed, BEFORE spending GPU on PPO.

    docker compose exec adam_rl python /opt/tools/audit_reward.py
    ADAM_U_LEVEL=1 docker compose exec ... python /opt/tools/audit_reward.py --episodes 128

Runs four fixed policies and prints every reward component separately:

    zero        a[:] = 0            fingers stay half-open, wrist holds still
    half        constant half closure
    random      uniform in [-1, 1] every step
    scripted    both synergies ramp 0 -> 1 over the grasp phase; this is the
                policy that should succeed

and then checks the properties that a shaping term has to have. This exists
because this repo has twice shipped a reward that looked fine and had no usable
gradient:

  * Experiment 2's two-finger run banked the full +2.0 of contact bonuses in
    EVERY episode while its mean action sat fully open -- exploration noise
    alone brushed the bolt often enough. A reward an untrained policy already
    collects in full provides no gradient.
  * Pipeline stage 2 then collapsed the other way: with the wrist free and
    nothing pulling the hand toward the bolt, contacts were exactly zero and
    the only term left with a slope was the action-rate penalty, which is
    minimised by holding a constant action.

Both are the same disease -- no usable slope between where the policy starts
and where success lives -- and both would have been caught here.
"""

from __future__ import annotations

import argparse
import math
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=128)
args, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from genesis_rl.envs import wuji_pipeline_env_cfg as P  # noqa: E402
from genesis_rl.envs import wuji_synergy_grasp_env_cfg as C  # noqa: E402

cfg = P.WujiPipelineEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = "cuda:0"
env = ManagerBasedRLEnv(cfg=cfg)
lvl = P.LEVELS[cfg.actions.pipeline.level]
n = env.num_envs
dev = env.device
TERMS = list(env.reward_manager.active_terms)


def run_components(name, action_fn):
    """One episode's worth of per-term sums, averaged over envs.

    Components come from the reward manager's OWN accumulator, not from
    re-calling the term functions. Most of these terms are paid once per
    episode and latch a flag when they pay (`_once`), so calling them a second
    time returns zero -- which is how the first version of this audit reported
    0.000 for every component including `success`, while the success RATE was
    75.8 %.
    """
    env.reset()
    ep_ret = torch.zeros(n, device=dev)
    acc = torch.zeros(n, len(TERMS), device=dev)
    for t in range(P.EPISODE_STEPS):
        _, rew, _, _, _ = env.step(action_fn(t))
        ep_ret += rew
        # Accumulate every step. The manager's own _episode_sums cannot be read
        # afterwards: Isaac Lab resets inside the same step() that ends the
        # episode and the reset zeroes them, which is how an earlier version of
        # this audit reported 0.000 for every component -- including `success`
        # -- while the success rate was 75.8 %.
        acc += env.reward_manager._step_reward
    b = C._buffers(env)
    # _step_reward is value/dt, i.e. the term's NATURAL size, so a +20 success
    # reads as 20.0 here rather than as 20*dt. Everything the audit compares is
    # in those units, including the +20 the thresholds are written against.
    sums = {k: float(acc[:, i].mean()) for i, k in enumerate(TERMS)}
    return (name, sums, float(ep_ret.mean()) / env.step_dt,
            float(b["last_success"].float().mean()), float(b["last_held"].float().mean()))


def a_open(t):
    """Fingers fully OPEN. a = -1 maps to closure 0, so this is the real
    do-nothing policy -- a = 0 is already half-closed, which is why the first
    version of this audit had two identical baselines."""
    a = torch.zeros(n, 8, device=dev)
    a[:, 6:8] = -1.0
    return a


def a_half(t):
    return torch.zeros(n, 8, device=dev)        # a = 0 -> closure 0.5


def a_random(t):
    return torch.rand(n, 8, device=dev) * 2.0 - 1.0


def a_scripted(t):
    a = torch.zeros(n, 8, device=dev)
    frac = min(1.0, t / max(1, P.GRASP_STEPS - 10))
    a[:, 6:8] = -1.0 + 2.0 * frac        # closure 0 -> 1 over the grasp phase
    return a


print(f"\nlevel {cfg.actions.pipeline.level} '{lvl.name}'   "
      f"xy +/-{lvl.dxy*100:.1f} cm, z {lvl.dz_lo*100:+.1f}/{lvl.dz_hi*100:+.1f} cm")
print(f"episodes per policy: {args.episodes} (num_envs {n})\n")

rows = []
for name, fn in (("open", a_open), ("half", a_half), ("random", a_random),
                 ("scripted", a_scripted)):
    reps = [run_components(name, fn) for _ in range(max(1, args.episodes // n))]
    comp = {k: sum(r[1][k] for r in reps) / len(reps) for k in TERMS}
    ret = sum(r[2] for r in reps) / len(reps)
    su = sum(r[3] for r in reps) / len(reps)
    he = sum(r[4] for r in reps) / len(reps)
    rows.append((name, comp, ret, su, he))
    print(f"  {name:9s} ran {len(reps) * n} episodes")

w = max(len(t) for t in TERMS) + 2
print("\n" + "=" * 78)
print("PER-EPISODE REWARD COMPONENTS")
print("=" * 78)
print(f"  {'term':{w}s}" + "".join(f"{r[0]:>12s}" for r in rows))
for k in TERMS:
    print(f"  {k:{w}s}" + "".join(f"{r[1][k]:12.3f}" for r in rows))
print(f"  {'TOTAL RETURN':{w}s}" + "".join(f"{r[2]:12.3f}" for r in rows))
print(f"  {'success rate':{w}s}" + "".join(f"{r[3]*100:11.1f}%" for r in rows))
print(f"  {'held rate':{w}s}" + "".join(f"{r[4]*100:11.1f}%" for r in rows))

by = {r[0]: r for r in rows}
APP = "approach"
checks = []


def check(desc, ok, detail=""):
    checks.append((desc, ok, detail))
    print(f"\n  {'PASS' if ok else 'FAIL'}  {desc}" + (f"   {detail}" if detail else ""))


print("\n" + "=" * 78)
print("AUDIT")
print("=" * 78)

z = by["open"][1].get(APP, 0.0)
check("an open-fingered do-nothing policy accumulates no approach reward",
      abs(z) < 0.15, f"open collected {z:+.3f}")

# The property that matters is NOT that a constant action collects less
# approach reward than the scripted grasp -- it may legitimately collect a
# little more, because the potential is about where the fingertips END UP and
# a half-closed hand can sit marginally nearer the bolt surface than a fully
# squeezed one. What matters is that the approach reward is BOUNDED and cannot
# be collected twice.
#
# Progress telescopes: whatever path it takes, a policy's total approach reward
# is 2(Phi_end - Phi_start) <= 2(1 - Phi_start). With sigma = 2 cm and the tips
# starting 1.6 cm out, Phi_start ~ 0.34, so the ceiling is ~1.33 -- against a
# +20 success. There is nothing to farm.
sc = by["scripted"][1].get(APP, 0.0)
phi0 = 1.0 - math.tanh(0.016 / P.PROX_SIGMA)
bound = P.PROX_WEIGHT * (1.0 - phi0)
worst = max(abs(r[1].get(APP, 0.0)) for r in rows)
check("approach reward is bounded by the potential and cannot be farmed",
      worst <= bound * 1.15,
      f"largest collected {worst:+.3f}, telescoping bound {bound:.3f}")
check("approach shaping stays small against the +20 success reward",
      worst < 0.10 * 20.0, f"largest {worst:+.3f} vs 20.0")

rn = by["random"][1].get(APP, 0.0)
check("random actions do not collect most of the shaping", abs(rn) < 0.8 * max(1e-6, abs(sc)),
      f"random {rn:+.3f} vs scripted {sc:+.3f}")

check("the scripted grasp returns substantially more than the others",
      by["scripted"][2] > max(by[k][2] for k in ("open", "half", "random")) + 5.0,
      f"scripted {by['scripted'][2]:.2f} vs best other "
      f"{max(by[k][2] for k in ('open','half','random')):.2f}")

pens = sum(by["scripted"][1].get(k, 0.0) for k in
           ("drop", "table_contact", "excess_force", "action_rate", "joint_limits"))
check("safety penalties do not outweigh success on a good grasp",
      abs(pens) < 0.25 * 20.0, f"penalties {pens:+.3f} against +20 success")

bon = sum(by["scripted"][1].get(k, 0.0) for k in
          ("thumb_contact", "finger_contact", "opposing_contact"))
check("contact bonuses stay small relative to the +20 success reward",
      bon <= 2.5, f"bonuses {bon:+.3f}")

# The scripted full-close is the NO-CORRECTION baseline: it closes the fingers
# and never moves the wrist. At level 0 it must succeed essentially always --
# that is the check that the pose, the collision geometry, the friction and the
# controller gains all permit a grasp at all, and PPO should not start without
# it. At any displaced level it is EXPECTED to fail sometimes, and how often is
# the number the experiment is about: if closing alone still worked, wrist
# control would have nothing to learn. Measured: 100 % at level 0, 72.7 % at
# level 1 (+/- 0.5 cm, +/- 2.5 deg).
if lvl.is_zero:
    check("the scripted closure succeeds reliably at zero displacement",
          by["scripted"][3] > 0.95, f"{by['scripted'][3]*100:.1f}% success")
else:
    print(f"\n  BASELINE  scripted full-close, no wrist correction: "
          f"{by['scripted'][3]*100:.1f}% success at this level.")
    print("            A trained policy has to beat this to show the wrist is earning its place.")

bad = [c for c in checks if not c[1]]
print("\n" + "=" * 78)
print(f"  {len(checks) - len(bad)} of {len(checks)} audit checks passed")
if bad:
    print("  DO NOT START PPO TRAINING until these pass:")
    for d, _, det in bad:
        print(f"    - {d}   ({det})")
print("=" * 78)
env.close()
app.close()
raise SystemExit(1 if bad else 0)
