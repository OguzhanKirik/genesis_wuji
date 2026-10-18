#!/usr/bin/env python3
"""Experiment 4 geometry gate, scripted baseline and reward audit. No PPO.

    docker compose exec -e ADAM_U_RG_LEVEL=0 adam_rl python /opt/tools/audit_reach_grasp.py --episodes 128
    docker compose exec -e ADAM_U_RG_LEVEL=3 adam_rl python /opt/tools/audit_reach_grasp.py

Runs five fixed policies for whole episodes and prints every reward component
separately (from the reward manager's own per-step accumulator, in the terms'
natural units, so a +20 success reads 20.0):

    open        wrist still, fingers fully open           the do-nothing policy
    closed      wrist still, fingers fully closed at once
    random      uniform in [-1, 1] every step
    scripted    descend straight down to the contact height, then ramp both
                synergies closed -- the NO-CORRECTION baseline every level's
                trained policy has to beat
    unsafe      drive the wrist down and sideways as hard as it can and never
                close -- must trip the invalid-collision termination

Then the checks. At level 0 (no standoff, no error) this is the GEOMETRY GATE
from docs/EXPERIMENT4_DESIGN.md §7: scripted success >= 95 %, zero invalid
collisions, valid joint slack, bolt drag < 5 mm. Training does not start
until it passes. At every level the reward audit checks that nothing can be
farmed, that the unsafe policy is punished and stopped, and that the scripted
grasp out-earns the others.
"""

from __future__ import annotations

import argparse
import math
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--skip", nargs="*", default=[], help="policies to skip, e.g. --skip random unsafe")
parser.add_argument("--bolt-mass", type=float, default=None, help="override the bolt's mass (kg)")
parser.add_argument("--finger-scale", type=float, default=None,
                    help="scale the driven fingers' stiffness, damping and effort limits (what-if for a stronger hand)")
args, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import torch  # noqa: E402

from genesis_rl.envs import wuji_reach_grasp_env_cfg as G  # noqa: E402
from genesis_rl.envs.wuji_pipeline_env_cfg import _metrics  # noqa: E402

cfg = G.WujiReachGraspEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = "cuda:0"
if args.bolt_mass is not None:
    cfg.scene.object.spawn.mass_props.mass = args.bolt_mass
    print(f"MASS OVERRIDE: bolt {args.bolt_mass:g} kg")
if args.finger_scale is not None:
    act = cfg.scene.robot.actuators["right_driven_fingers"]
    act.stiffness = act.stiffness * args.finger_scale
    act.damping = act.damping * args.finger_scale
    act.effort_limit = {k: v * args.finger_scale for k, v in act.effort_limit.items()}
    print(f"FINGER OVERRIDE: driven-finger stiffness / damping / effort x{args.finger_scale:g}")
env = G.ReachGraspEnv(cfg=cfg)
lvl = G.LEVELS[cfg.actions.pipeline.level]
n, dev = env.num_envs, env.device
TERMS = list(env.reward_manager.active_terms)
lim = cfg.actions.pipeline.limits


def wrist_z():
    pos, _ = G._wrist_pose(env)
    return pos[:, 2]


# ---- the five policies -----------------------------------------------------

def a_open(t):
    a = torch.zeros(n, 8, device=dev)
    a[:, 6:8] = -1.0
    return a


def a_closed(t):
    a = torch.zeros(n, 8, device=dev)
    a[:, 6:8] = 1.0
    return a


def a_random(t):
    return torch.rand(n, 8, device=dev) * 2.0 - 1.0


class Scripted:
    """Straight down to the contact height, then close over 40 steps. Per env."""

    def __init__(self):
        self.reached_at = torch.full((n,), -1.0, device=dev)

    def reset(self):
        self.reached_at[:] = -1.0

    def __call__(self, t):
        a = torch.zeros(n, 8, device=dev)
        a[:, 6:8] = -1.0
        z = wrist_z()
        above = z - G.WRIST_Z_CONTACT
        arrived = above < 0.003
        self.reached_at = torch.where(arrived & (self.reached_at < 0), torch.full_like(self.reached_at, float(t)),
                                      self.reached_at)
        # descend at full speed until arrived; a proportional tail avoids overshoot
        dz = -(above / lim.effective_translation).clamp(0.0, 1.0)
        a[:, 2] = torch.where(arrived, torch.zeros_like(dz), dz)
        frac = ((t - self.reached_at) / 40.0).clamp(0.0, 1.0)
        c = -1.0 + 2.0 * frac
        a[:, 6:8] = torch.where((self.reached_at >= 0).unsqueeze(1), c.unsqueeze(1).expand(n, 2), a[:, 6:8])
        return a


def a_unsafe(t):
    a = torch.zeros(n, 8, device=dev)
    a[:, 0] = 1.0        # sideways, hard
    a[:, 2] = -1.0       # down, hard
    a[:, 6:8] = -1.0
    return a


scripted = Scripted()


def run(name, fn, episodes):
    """Whole episodes; per-term sums averaged over episodes, plus outcomes."""
    acc = torch.zeros(len(TERMS), device=dev)
    ret = 0.0
    done_count = 0
    out = dict(success=0, held=0, invalid=0, inv_table=0, inv_struct=0, inv_bolt=0, inv_obs=0, excess=0, drop=0,
               latched=0, drag_sum=0.0, drag_max=0.0, slack_worst=1.0, clear=torch.ones(len(G.GROUP_NAMES), device=dev),
               grasp_end=0.0)
    env.reset()
    if hasattr(fn, "reset"):
        fn.reset()
    t = 0
    # per-env accumulators over the CURRENT episode
    ep_acc = torch.zeros(n, len(TERMS), device=dev)
    ep_ret = torch.zeros(n, device=dev)
    quota = max(1, math.ceil(episodes / n))           # every env contributes exactly quota episodes (no length bias)
    ep_count = torch.zeros(n, dtype=torch.long, device=dev)
    while bool((ep_count < quota).any()):
        _, rew, term, trunc, _ = env.step(fn(t))
        t += 1
        ep_acc += env.reward_manager._step_reward
        ep_ret += rew
        m = _metrics(env)
        out["slack_worst"] = min(out["slack_worst"], float(m["s_min"].min()))
        done = (term | trunc) & (ep_count < quota)
        if done.any():
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            ep_count[ids] += 1
            b = G._rg(env)
            k = len(ids)
            done_count += k
            acc += ep_acc[ids].sum(dim=0)
            ret += float(ep_ret[ids].sum())
            ep_acc[ids] = 0.0
            ep_ret[ids] = 0.0
            out["success"] += int(b["last_success"][ids].sum())
            out["held"] += int(b["last_held"][ids].sum())
            inv = b["last_invalid"][ids]
            out["invalid"] += int(inv.sum())
            w = b["last_invalid_which"][ids]
            out["inv_table"] += int((inv & (w == 1)).sum())
            out["inv_struct"] += int((inv & (w == 2)).sum())
            out["inv_bolt"] += int((inv & (w == 3)).sum())
            out["inv_obs"] += int((inv & (w == 4)).sum())
            out["excess"] += int(b["last_excess"][ids].sum())
            out["drop"] += int(b["last_drop"][ids].sum())
            out["latched"] += int(b["last_latched"][ids].sum())
            out["drag_sum"] += float(b["last_drag"][ids].sum())
            out["drag_max"] = max(out["drag_max"], float(b["last_drag"][ids].max()))
            out["clear"] = torch.minimum(out["clear"], b["last_min_clear"][ids].min(dim=0).values)
            out["grasp_end"] += float(b["last_grasp_end"][ids].sum())
            if hasattr(fn, "reset"):
                fn.reached_at[ids] = -1.0
    E = max(1, done_count)
    comps = {k: float(acc[i]) / E for i, k in enumerate(TERMS)}
    # _step_reward is value / dt: natural units. `ret` is the dt-scaled env reward.
    return dict(name=name, comps=comps, ret=ret / E / env.step_dt, E=E, **{k: v for k, v in out.items()})


print(f"\nlevel {cfg.actions.pipeline.level} '{lvl.name}'   standoff {lvl.standoff_lo*100:.1f}-{lvl.standoff_hi*100:.1f} cm, "
      f"xy +/-{lvl.dxy*100:.1f} cm, yaw +/-{math.degrees(lvl.dyaw):.1f} deg, roll/pitch +/-{math.degrees(lvl.drp):.1f} deg")
print(f"episodes per policy: {args.episodes} (num_envs {n}); per-step translation cap {lim.effective_translation*1000:.2f} mm\n")

rows = []
for name, fn in (("open", a_open), ("closed", a_closed), ("random", a_random), ("scripted", scripted),
                 ("unsafe", a_unsafe)):
    if name in args.skip:
        continue
    r = run(name, fn, args.episodes)
    rows.append(r)
    print(f"  {name:9s} ran {r['E']} episodes")

w = max(len(t) for t in TERMS) + 2
print("\n" + "=" * 92)
print("PER-EPISODE REWARD COMPONENTS (natural units)")
print("=" * 92)
print(f"  {'term':{w}s}" + "".join(f"{r['name']:>12s}" for r in rows))
for k in TERMS:
    print(f"  {k:{w}s}" + "".join(f"{r['comps'][k]:12.3f}" for r in rows))
print(f"  {'TOTAL RETURN':{w}s}" + "".join(f"{r['ret']:12.3f}" for r in rows))
for key, label, pct in (("success", "success rate", True), ("held", "held rate", True), ("latched", "latched", True),
                        ("invalid", "invalid collision", True), ("inv_table", "  finger-table", True),
                        ("inv_struct", "  structural-table", True), ("inv_bolt", "  non-grasp link on bolt/obstacle", True), ("inv_obs", "  obstacle", True),
                        ("excess", "excess force", True), ("drop", "drop", True)):
    print(f"  {label:{w}s}" + "".join(f"{r[key] / r['E'] * 100:11.1f}%" for r in rows))
print(f"  {'grasp end (mean step)':{w}s}" + "".join(f"{r['grasp_end'] / r['E']:12.1f}" for r in rows))
print(f"  {'bolt drag mean / max mm':{w}s}" + "".join(f"{r['drag_sum'] / r['E'] * 1000:6.1f}/{r['drag_max'] * 1000:5.1f}" for r in rows))
print(f"  {'worst joint slack':{w}s}" + "".join(f"{r['slack_worst']:12.3f}" for r in rows))
for i, g in enumerate(G.GROUP_NAMES):
    print(f"  {'min clearance ' + g + ' cm':{w}s}" + "".join(f"{float(r['clear'][i]) * 100:12.2f}" for r in rows))

by = {r["name"]: r for r in rows}
checks = []


def check(desc, ok, detail=""):
    checks.append((desc, ok, detail))
    print(f"\n  {'PASS' if ok else 'FAIL'}  {desc}" + (f"   {detail}" if detail else ""))


print("\n" + "=" * 92)
print("AUDIT")
print("=" * 92)
if "open" in by:
    sh = by["open"]["comps"].get("reach", 0.0) + by["open"]["comps"].get("fingers", 0.0)
    check("the do-nothing policy collects no shaping", abs(sh) < 0.15, f"open collected {sh:+.3f}")
if "scripted" in by:
    sc = by["scripted"]
    shaping = sc["comps"].get("reach", 0.0) + sc["comps"].get("fingers", 0.0)
    check("shaping stays small against the +20 success", shaping < 0.25 * 20.0, f"scripted shaping {shaping:+.3f}")
    bon = sum(sc["comps"].get(k, 0.0) for k in ("thumb_contact", "finger_contact", "opposing_contact"))
    check("contact bonuses stay small relative to success", bon <= 2.5, f"bonuses {bon:+.3f}")
    pens = sum(sc["comps"].get(k, 0.0) for k in ("velocity", "clearance", "shield", "action_rate", "joint_limits"))
    check("regularisers do not outweigh success on a good grasp", abs(pens) < 0.25 * 20.0, f"{pens:+.3f}")
    # At level 0 there is no standoff, so an instant full close IS a grasp and
    # earns what the scripted ramp earns (it just drags the bolt more); the
    # comparison against "closed" only means something once the hand has to
    # descend first.
    rivals = ("open", "random") if cfg.actions.pipeline.level == 0 else ("open", "closed", "random")
    others = [by[k]["ret"] for k in rivals if k in by]
    if others:
        check(f"the scripted grasp returns substantially more than {' / '.join(rivals)}",
              sc["ret"] > max(others) + 5.0, f"scripted {sc['ret']:.2f} vs best other {max(others):.2f}")
if "random" in by and "scripted" in by:
    rn = by["random"]["comps"].get("reach", 0.0) + by["random"]["comps"].get("fingers", 0.0)
    sc_sh = by["scripted"]["comps"].get("reach", 0.0) + by["scripted"]["comps"].get("fingers", 0.0)
    # Signed: random flailing that LOSES shaping is fine; collecting most of it is not.
    check("random actions do not collect most of the shaping", rn < 0.8 * max(1e-6, sc_sh),
          f"random {rn:+.3f} vs scripted {sc_sh:+.3f}")
if "unsafe" in by:
    u = by["unsafe"]
    check("the unsafe policy is caught by the invalid-collision termination in every episode",
          u["invalid"] == u["E"], f"{u['invalid']}/{u['E']} episodes")
    check("the unsafe policy earns less than doing nothing",
          u["ret"] < by["open"]["ret"] - 5.0 if "open" in by else u["ret"] < -5.0,
          f"unsafe {u['ret']:.2f}" + (f" vs open {by['open']['ret']:.2f}" if "open" in by else ""))
    check("clearance penalty fires on the unsafe policy", u["comps"].get("clearance", 0.0) < -1e-3,
          f"{u['comps'].get('clearance', 0.0):+.4f}")

if "scripted" in by:
    sc = by["scripted"]
    if cfg.actions.pipeline.level == 0:
        print("\n  GEOMETRY GATE (level 0)")
        check("scripted lift-and-hold success >= 95 %", sc["success"] / sc["E"] >= 0.95,
              f"{sc['success'] / sc['E'] * 100:.1f}%")
        check("zero invalid collisions on the scripted grasp", sc["invalid"] == 0, f"{sc['invalid']}")
        check("joint slack stays above the termination threshold",
              sc["slack_worst"] > lim.slack_terminate, f"worst {sc['slack_worst']:.3f}")
        check("horizontal bolt drag during the lift < 5 mm", sc["drag_max"] < 0.005,
              f"mean {sc['drag_sum'] / sc['E'] * 1000:.1f} mm, max {sc['drag_max'] * 1000:.1f} mm")
        check("the latch fires on the scripted grasp", sc["latched"] / sc["E"] > 0.95,
              f"{sc['latched'] / sc['E'] * 100:.1f}% latched, mean grasp end {sc['grasp_end'] / sc['E']:.1f}")
    else:
        print(f"\n  BASELINE  scripted straight-down grasp, no correction: {sc['success'] / sc['E'] * 100:.1f}% "
              f"success at this level. A trained policy has to beat this.")

bad = [c for c in checks if not c[1]]
print("\n" + "=" * 92)
print(f"  {len(checks) - len(bad)} of {len(checks)} audit checks passed")
if bad:
    print("  DO NOT START PPO TRAINING until these pass:")
    for d, _, det in bad:
        print(f"    - {d}   ({det})")
print("=" * 92)
env.close()
app.close()
raise SystemExit(1 if bad else 0)
