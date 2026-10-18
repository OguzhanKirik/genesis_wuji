#!/usr/bin/env python3
"""Evaluate an Experiment 4 checkpoint per EPISODE, against the advancement gate.

    docker compose exec -e ADAM_U_RG_LEVEL=<n> adam_rl python /opt/tools/eval_reach_grasp.py \
        --checkpoint_path logs/rsl_rl/adam_u30_wuji_reach_grasp_L<n>/<run>/model_<k>.pt \
        --num_envs 256 --episodes 512

    # watch it
    docker compose exec -e ADAM_U_RG_LEVEL=<n> adam_rl python /opt/tools/eval_reach_grasp.py --viz kit \
        --num_envs 16 --episodes 16 --real-time --speed 0.25 --start-delay 15 --checkpoint_path ...

The level MUST be passed with `docker compose exec -e`; prefixing it on the
host does not reach the container and the eval silently runs level 0. The
level is printed -- check it.

Reports (docs/EXPERIMENT4_DESIGN.md §9): grasp success, collision-free
success, failure types, latch statistics, minimum clearance per link group,
shield intervention rate, fingertip force and joint-slack extremes, wrist
jerk, bolt drag, and the advancement checklist. Outcomes come from the env's
`last_*` buffers, written at each episode's final step and NOT cleared by the
reset (Isaac Lab resets inside the step() that ends an episode).
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import math
import sys
import time

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=512)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--start-delay", type=float, default=0.0)
parser.add_argument("--bolt-yaw-deg", type=float, default=0.0,
                    help="generalisation probe: randomise the BOLT's yaw at reset, uniform +/- this (deg). "
                         "The policy was trained at one bolt pose; this measures how far it stretches.")
parser.add_argument("--bolt-mass", type=float, default=None,
                    help="override the bolt's mass (kg) for this evaluation; the level's mass randomisation "
                         "then scales THIS value")
parser.add_argument("--grasp-offset-mm", type=float, default=None,
                    help="pin the commanded grasp point at this offset along the rod's axis (mm) for every "
                         "episode; the graspable-area sweep runs this over a range of values")
parser.add_argument("--pinch-angle-deg", type=float, default=None,
                    help="pin the commanded pinch angle (hand yaw minus rod yaw, deg) for every episode")
parser.add_argument("--bolt-collider", choices=["hull", "cd"], default=None,
                    help="override the rod's collision shape: the asset's 64-vertex convex hull, or the "
                         "convex-decomposition variant (levels with a standing rod use 'cd')")
parser.add_argument("--upright", type=int, choices=[0, 1], default=None,
                    help="pin the rod's pose: 1 standing on its end, 0 lying (levels 18+ sample it)")
parser.add_argument("--tilt", type=int, choices=[0, 1], default=None,
                    help="pin the rod leaning on a support block (1) or not (0) (levels 20+ sample it)")
parser.add_argument("--pile", type=int, choices=[0, 1], default=None,
                    help="pile levels: 1 puts the pile in every episode, 0 parks it (default: the level's probability)")
parser.add_argument("--across", type=int, choices=[0, 1], default=None,
                    help="pile levels: 1 pins a neighbour across every lying/tilted target, 0 none (default: the level's probability)")
parser.add_argument("--cover", type=int, choices=[0, 1], default=None,
                    help="pin the cover block over one end of every LYING rod (1) or none (0) (level 23+)")
parser.add_argument("--bed", type=int, choices=[0, 1], default=None,
                    help="pin the target resting across two bed rods (1) or on the table (0) for LYING episodes (level 24+)")
parser.add_argument("--fill", type=int, choices=[0, 1], default=None,
                    help="full-bin levels (50+): 1 a heap in every episode, 0 none (default: the level's probability)")
parser.add_argument("--fill-kind", type=int, choices=[0, 1, 2], default=None,
                    help="full-bin levels: prefer a lying (0), leaning (1) or standing (2) heap rod as the target")
parser.add_argument("--bolt-xy-cm", type=float, default=0.0,
                    help="generalisation probe: randomise the bolt's x and y at reset, uniform +/- this (cm)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

from genesis_rl.algos.rsl_rl.rsl_rl_ppo_cfg import RslRlWujiReachGraspPPORunnerCfg  # noqa: E402
from genesis_rl.envs import wuji_reach_grasp_env_cfg as G  # noqa: E402
from genesis_rl.envs.wuji_pipeline_env_cfg import _metrics  # noqa: E402

device = "cuda:0"
cfg = G.WujiReachGraspEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = device
lvl = G.LEVELS[cfg.actions.pipeline.level]
if args.bolt_mass is not None:
    cfg.scene.object.spawn.mass_props.mass = args.bolt_mass
if args.grasp_offset_mm is not None:
    cfg.events.reset_bolt.params["grasp_offset_fixed"] = args.grasp_offset_mm / 1000.0
if args.pinch_angle_deg is not None:
    cfg.events.reset_bolt.params["pinch_angle_fixed"] = math.radians(args.pinch_angle_deg)
if args.upright is not None:
    cfg.events.reset_bolt.params["upright_fixed"] = bool(args.upright)
if args.tilt is not None:
    cfg.events.reset_bolt.params["tilt_fixed"] = bool(args.tilt)
if args.across is not None:
    cfg.events.pile.params["across_fixed"] = bool(args.across)
if args.pile is not None:
    cfg.events.pile.params["pile_fixed"] = bool(args.pile)
    if args.tilt == 1 and args.upright is None:
        cfg.events.reset_bolt.params["upright_fixed"] = False
if args.cover is not None:
    cfg.events.reset_bolt.params["cover_fixed"] = bool(args.cover)
if args.bed is not None:
    cfg.events.reset_bolt.params["bed_fixed"] = bool(args.bed)
if args.fill is not None:
    cfg.events.reset_bolt.params["fill_fixed"] = bool(args.fill)
if args.fill_kind is not None:
    cfg.events.reset_bolt.params["fill_kind"] = args.fill_kind
if args.bolt_collider is not None:
    cfg.scene.object.spawn.usd_path = G.bolt_asset(args.bolt_collider)
if args.bolt_yaw_deg or args.bolt_xy_cm:
    d, a = args.bolt_xy_cm / 100.0, math.radians(args.bolt_yaw_deg)
    cfg.events.reset_bolt.params["pose_range"] = {"x": (-d, d), "y": (-d, d), "z": (0.0, 0.0), "yaw": (-a, a)}

gym.register(id="EvalReachGrasp", entry_point="genesis_rl.envs.wuji_reach_grasp_env_cfg:ReachGraspEnv",
             disable_env_checker=True, kwargs={"env_cfg_entry_point": G.WujiReachGraspEnvCfg})
env = gym.make("EvalReachGrasp", cfg=cfg)
u = env.unwrapped

agent_cfg = RslRlWujiReachGraspPPORunnerCfg()
agent_cfg.device = device
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(wenv, agent_cfg.to_dict(), log_dir=None, device=device)
runner.load(args.checkpoint_path)
policy = runner.get_inference_policy(device=device)

n = u.num_envs
G_ = len(G.GROUP_NAMES)
tot = dict(retried=0, retry_success=0, grip=0, relabel=0, double=0, episodes=0, success=0, held=0, invalid=0, inv_table=0, inv_struct=0, inv_bolt=0, inv_obs=0, excess=0, drop=0,
           lifted=0, dirty=0, latched=0, grasp_end=0.0, drag_sum=0.0, drag_max=0.0, jerk_sum=0.0, jerk_max=0.0,
           shield_steps=0.0, thumb=0.0, finger=0.0, clo_t=0.0, clo_f=0.0,
           clear_min=torch.ones(G_, device=device), clear_sum=torch.zeros(G_, device=device))
drags: list[float] = []          # per-episode, for the percentile
serrs: list[float] = []          # |pinch along the rod - commanded point| at the latch, latched episodes
derrs: list[float] = []          # pinch midpoint to the rod axis at the latch, latched episodes
steps = 0
policy_steps = 0
shield_policy_steps = 0
s_min_worst, s_min_sum = 1.0, 0.0
limit_terms, ik_fail_steps, clip_steps = 0, 0, 0
track_sum, track_worst = 0.0, 0.0
tip_force_max = 0.0

if args.start_delay > 0:
    wenv.reset()
    t_end = time.time() + args.start_delay
    while time.time() < t_end:
        u.sim.render()
        time.sleep(0.05)

dt = u.step_dt
obs = wenv.get_observations()
with torch.inference_mode():
    # Every env contributes exactly `quota` episodes. Stopping at the first N completions instead counted the
    # fastest-finishing episodes and left the long failing ones (three retries) in flight uncounted: on the
    # heap that read 8-11 points too high (2026-10-09).
    quota = max(1, math.ceil(args.episodes / n))
    ep_count = torch.zeros(n, dtype=torch.long, device=device)
    while bool((ep_count < quota).any()):
        t0 = time.time()
        act = policy(obs)
        obs, rew, dones, extras = wenv.step(act)
        if args.real_time:
            left = dt / args.speed - (time.time() - t0)
            if left > 0:
                time.sleep(left)
        steps += 1
        m = _metrics(u)
        b = G._rg(u)
        s_min_worst = min(s_min_worst, float(m["s_min"].min()))
        s_min_sum += float(m["s_min"].mean())
        ik_fail_steps += int((m["ik_failed"] > 0.5).sum())
        clip_steps += int((m["clipped"] > 0).sum())
        limit_terms += int(u.termination_manager.get_term("joint_limit").sum())
        track_sum += float(m["wrist_track_err"].mean())
        track_worst = max(track_worst, float(m["wrist_track_err"].max()))
        tip_force_max = max(tip_force_max, float(b["tip_force"].max()))
        grasp = (G.phase_of(u) == 0)
        policy_steps += int(grasp.sum())
        shield_policy_steps += int((grasp & (b["alpha"] < 0.999)).sum())
        done = dones.bool()
        done = done & (ep_count < quota)
        if done.any():
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            ep_count[ids] += 1
            k = len(ids)
            tot["episodes"] += k
            tot["success"] += int(b["last_success"][ids].sum())
            tot["held"] += int(b["last_held"][ids].sum())
            inv = b["last_invalid"][ids]
            w = b["last_invalid_which"][ids]
            tot["invalid"] += int(inv.sum())
            tot["inv_table"] += int((inv & (w == 1)).sum())
            tot["inv_struct"] += int((inv & (w == 2)).sum())
            tot["inv_bolt"] += int((inv & (w == 3)).sum())
            tot["inv_obs"] += int((inv & (w == 4)).sum())
            tot["excess"] += int(b["last_excess"][ids].sum())
            tot["drop"] += int(b["last_drop"][ids].sum())
            tot["lifted"] += int(b["last_was_up"][ids].sum())
            tot["dirty"] += int((~b["last_verify_clean"][ids]).sum())
            tot["latched"] += int(b["last_latched"][ids].sum())
            tot["grasp_end"] += float(b["last_grasp_end"][ids].sum())
            tot["drag_sum"] += float(b["last_drag"][ids].sum())
            tot["drag_max"] = max(tot["drag_max"], float(b["last_drag"][ids].max()))
            drags.extend(b["last_drag"][ids].tolist())
            serrs.extend(b["last_latch_s_err"][ids][b["last_latched"][ids] > 0].tolist())
            derrs.extend(b["last_latch_d_err"][ids][b["last_latched"][ids] > 0].tolist())
            tot["retried"] += int((b["last_attempts"][ids] > 0).sum())
            tot["double"] += int(b["last_double"][ids].sum())
            tot["retry_success"] += int(b["last_retry_success"][ids].sum())
            tot["grip"] += int(b["last_grip_fired"][ids].sum())
            tot["relabel"] += int(b["last_heap_swapped"][ids].sum()) if "last_heap_swapped" in b else 0
            tot["jerk_sum"] += float(b["last_jerk_max"][ids].sum())
            tot["jerk_max"] = max(tot["jerk_max"], float(b["last_jerk_max"][ids].max()))
            tot["shield_steps"] += float(b["last_shield_steps"][ids].sum())
            tot["thumb"] += float(b["last_thumb_frac"][ids].sum())
            tot["finger"] += float(b["last_finger_frac"][ids].sum())
            tot["clo_t"] += float(b["last_closure"][ids, 0].sum())
            tot["clo_f"] += float(b["last_closure"][ids, 1].sum())
            tot["clear_min"] = torch.minimum(tot["clear_min"], b["last_min_clear"][ids].min(dim=0).values)
            tot["clear_sum"] += b["last_min_clear"][ids].sum(dim=0)

E = max(1, tot["episodes"])
S = max(1, steps)
pct = lambda k: tot[k] / E * 100.0  # noqa: E731
print("=" * 74)
print(f"checkpoint {args.checkpoint_path}")
print(f"level {cfg.actions.pipeline.level} '{lvl.name}'   standoff {lvl.standoff_lo*100:.1f}-{lvl.standoff_hi*100:.1f} cm, "
      f"xy +/-{lvl.dxy*100:.1f} cm, yaw +/-{math.degrees(lvl.dyaw):.1f} deg, roll/pitch +/-{math.degrees(lvl.drp):.1f} deg, "
      f"perception {'ON' if lvl.perception.active else 'off'}")
print(f"episodes {tot['episodes']}   (num_envs {n}, policy mean action)")
if args.bolt_mass is not None:
    print(f"MASS PROBE: bolt mass {args.bolt_mass:g} kg (trained at 0.03 kg, x0.8-1.2 from level 5)")
if args.grasp_offset_mm is not None:
    print(f"GRASP POINT: pinned at {args.grasp_offset_mm:+g} mm along the rod's axis "
          f"(rod half-length {G.BOLT_HALF_LEN*1000:.1f} mm; level trains +/-{lvl.grasp_offset*1000:.0f} mm)")
elif lvl.grasp_offset > 0:
    print(f"GRASP POINT: sampled +/-{lvl.grasp_offset*1000:.0f} mm along the rod's axis")
if args.pinch_angle_deg is not None:
    print(f"PINCH ANGLE: pinned at {args.pinch_angle_deg:+g} deg off perpendicular "
          f"(level trains +/-{math.degrees(lvl.pinch_angle):.0f} deg)")
elif lvl.pinch_angle > 0:
    print(f"PINCH ANGLE: sampled +/-{math.degrees(lvl.pinch_angle):.0f} deg off perpendicular")
if args.upright is not None or args.tilt is not None or lvl.upright_prob > 0 or lvl.tilt.active:
    pose = ("pinned tilted (leaning on a block)" if args.tilt == 1 else "pinned standing" if args.upright == 1
            else "pinned lying" if (args.upright == 0 and args.tilt in (None, 0)) or args.tilt == 0 and args.upright is None and lvl.upright_prob == 0
            else f"standing {lvl.upright_prob:g}, tilted {lvl.tilt.prob:g}, lying {max(0.0, 1 - lvl.upright_prob - lvl.tilt.prob):.2f}")
    print(f"ROD POSE: {pose}; collider {args.bolt_collider or lvl.bolt_collider}")
if lvl.pile.active:
    print(f"PILE: {lvl.pile.count} neighbours at {lvl.pile.r[0]*100:g}-{lvl.pile.r[1]*100:g} cm in "
          f"{'every episode' if args.pile == 1 else 'no episode' if args.pile == 0 else format(lvl.pile.prob, 'g') + ' of episodes'}; one across the target: "
          f"{'pinned' if args.across == 1 else 'none' if args.across == 0 else format(lvl.pile.across_prob, 'g')}")
if lvl.free_grasp:
    print(f"FREE GRASP: no commands, the policy sees the true rod; reach reward to the rod's axis, no yaw term; "
          f"world-yaw action x{lvl.yaw_rate_scale:g}. Pinned offset/angle above set only the START pose.")
if args.bolt_yaw_deg or args.bolt_xy_cm:
    print(f"GENERALISATION PROBE: bolt yaw +/-{args.bolt_yaw_deg:g} deg, xy +/-{args.bolt_xy_cm:g} cm at reset "
          f"(the policy never trained on a moved bolt)")
print("\n  GRASP")
print(f"    lift-and-hold success            {pct('success'):5.1f} %   (the gate number)")
print(f"    held >= {G.SUCCESS_LIFT*100:.0f} cm through the hold     {pct('held'):5.1f} %   (ignoring disqualifiers)")
print(f"    collision-free success           {(tot['success']) / E * 100:5.1f} %   (success already requires it)")
print(f"    lifted at all                    {pct('lifted'):5.1f} %")
print(f"    latched (early grasp end)        {pct('latched'):5.1f} %   mean grasp end step {tot['grasp_end'] / E:5.1f}")
print(f"    bolt-drop rate                   {pct('drop'):5.1f} %")
if lvl.retry_max > 0:
    print(f"    retried                          {pct('retried'):5.1f} %   of which succeeded {tot['retry_success'] / max(1, tot['retried']) * 100:5.1f} %"
          f"   (success without a retry {(tot['success'] - tot['retry_success']) / E * 100:5.1f} %)")
    if lvl.grip_check:
        print(f"    grip check sent into a retry     {pct('grip'):5.1f} %   (v3: 24 mm test lift, fingertips vs rod)")
if lvl.pile.active:
    print(f"    double picks (a neighbour lifted)  {pct('double'):5.1f} %")
    if tot["relabel"]:
        print(f"    heap re-labelled the target (top-rod rule)  {pct('relabel'):5.1f} %")
if serrs:
    se = sorted(serrs)
    print(f"    pinch vs commanded point         mean {sum(se) / len(se) * 1000:5.1f} mm   p95 {se[int(0.95 * (len(se) - 1))] * 1000:5.1f} mm   "
          f"max {se[-1] * 1000:5.1f} mm   ({'binding, tol ' + format(lvl.grasp_point_tol * 1000, '.0f') + ' mm' if lvl.grasp_point_tol > 0 else 'NOT binding at this level'})")
if derrs:
    de = sorted(derrs)
    print(f"    pinch midpoint vs rod axis       mean {sum(de) / len(de) * 1000:5.1f} mm   p95 {de[int(0.95 * (len(de) - 1))] * 1000:5.1f} mm   "
          f"max {de[-1] * 1000:5.1f} mm   ({'binding, tol ' + format(lvl.pinch_centre_tol * 1000, '.0f') + ' mm' if lvl.pinch_centre_tol > 0 else 'NOT binding at this level'})")
print(f"    bolt touched table in hold       {pct('dirty'):5.1f} %")
print(f"    thumb / opposing contact         {tot['thumb'] / E * 100:5.1f} % / {tot['finger'] / E * 100:5.1f} % of grasp steps")
print(f"    final synergy (thumb, fingers)   ({tot['clo_t'] / E:.2f}, {tot['clo_f'] / E:.2f})")
print("\n  SAFETY")
print(f"    invalid collisions               {pct('invalid'):5.1f} %   finger-table {pct('inv_table'):.1f} %, "
      f"structural {pct('inv_struct'):.1f} %, non-grasp link on bolt/obstacle {pct('inv_bolt'):.1f} %, obstacle {pct('inv_obs'):.1f} %")
print(f"    excessive force (> {G.F_MAX:.0f} N)        {pct('excess'):5.1f} %   peak fingertip force {tip_force_max:5.1f} N")
print(f"    shield interventions             {shield_policy_steps / max(1, policy_steps) * 100:5.2f} % of policy steps")
for i, g in enumerate(G.GROUP_NAMES):
    print(f"    min clearance {g:9s}          mean {float(tot['clear_sum'][i]) / E * 100:5.2f} cm   worst {float(tot['clear_min'][i]) * 100:6.2f} cm")
drag_p95 = sorted(drags)[int(0.95 * (len(drags) - 1))] if drags else 0.0
print(f"    bolt drag during lift            mean {tot['drag_sum'] / E * 1000:5.1f} mm   p95 {drag_p95 * 1000:5.1f} mm   max {tot['drag_max'] * 1000:5.1f} mm")
print(f"    wrist jerk (episode max)         mean {tot['jerk_sum'] / E:7.1f}   max {tot['jerk_max']:7.1f} m/s^3")
print("\n  ARM AND IK")
print(f"    minimum arm joint slack          worst {s_min_worst:5.3f}   mean {s_min_sum / S:5.3f}")
print(f"    joint-limit terminations         {limit_terms}")
print(f"    env-steps with an IK failure     {ik_fail_steps}")
print(f"    env-steps with a clipped target  {clip_steps}")
print(f"    wrist tracking error             mean {track_sum / S * 1000:5.2f} mm   worst {track_worst * 1000:5.2f} mm")
print("\n  ADVANCEMENT GATE (docs/EXPERIMENT4_DESIGN.md §9)")
gate = [
    ("lift-and-hold success >= 95 %", pct("success") >= 95.0),
    ("invalid collisions == 0", tot["invalid"] == 0),
    ("shield interventions < 1 % of policy steps", shield_policy_steps / max(1, policy_steps) < 0.01),
    ("joint-limit terminations == 0", limit_terms == 0),
    ("excessive-force failures < 1 %", pct("excess") < 1.0),
    # 95th percentile, not the maximum: the max of 512 draws from a
    # distribution with mean 2.6 mm and p95 3.7 mm flipped between 4.7 and
    # 5.2 mm across identical evaluations (level 2, 2026-09-24), while the
    # wrist itself moved 0.7 mm. The max is still printed above.
    ("bolt drag during lift < 5 mm (95th percentile)", drag_p95 < 0.005),
]
for desc, ok in gate:
    print(f"    {'PASS' if ok else 'FAIL'}  {desc}")
print("    ----  beats the scripted baseline: compare with tools/audit_reach_grasp.py at this level")
print("=" * 74)
env.close()
app.close()
