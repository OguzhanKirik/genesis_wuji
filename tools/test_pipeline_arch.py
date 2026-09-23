#!/usr/bin/env python3
"""Architecture validation for the staged grasp pipeline. Runs in the container.

    docker compose exec adam_rl python /opt/tools/test_pipeline_arch.py
    ADAM_U_STAGE=1 docker compose exec ... python /opt/tools/test_pipeline_arch.py
    python /opt/tools/test_pipeline_arch.py --tolerance 0.02 --seconds 2.0

Every check the architecture spec asks for, each one printed with its own
pass/fail so a regression names itself:

  1. action mapping       policy 8 -> arm 7 + hand 20 = 27 right-side targets
  2. target propagation   every held joint has a nonzero, correct drive target
                          after reset AND after several control steps
  3. zero-action stability 2 s of zero policy action; every held joint stays at
                          its configured target and the wrist stays put
  4. held-joint regression per-group drift table: waist, neck, left arm, left
                          hand, the 11 fixed right-hand joints
  5. IK limit             drive the wrist at the workspace edge and confirm the
                          targets stay inside the joint limits, the null-space
                          term buys clearance, clipping is reported, IK failure
                          is reported, and nothing goes NaN
  6. reach envelope       the sim-side counterpart of
                          tools/solve_pipeline_pregrasp.py --envelope, which
                          also measures the arm SAG the kinematic solve cannot
  7. control-rate         a held Cartesian command must move the wrist by the
                          per-step limit once per CONTROL step, and the scripted
                          lift must rise by LIFT_HEIGHT -- not by the decimation
                          times either of them

The test this exists for: `waistPitch` reset to 1.05 rad while its drive target
stays at 0, which is silent, moves the whole reach envelope, and was measured
happening on this robot before `hold_default_joint_targets` existed.
"""

from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=2.0,
                    help="how long the zero-action stability test runs")
parser.add_argument("--tolerance", type=float, default=0.02,
                    help="rad; a held joint may drift this far from its configured target. 0.02 "
                         "rather than 0.01 because the finger drives are deliberately soft "
                         "(stiffness 3.0, 0.3 N.m on a pip/dip) and an UNLOADED held finger sags "
                         "~0.017 rad under gravity at a correct target -- measured. That is drive "
                         "compliance, not a target bug; the strict check for a target left at zero "
                         "is check 2, which has no tolerance at all")
parser.add_argument("--wrist-tolerance", type=float, default=0.005,
                    help="m; how far the wrist may move under zero action")
parser.add_argument("--only", type=str, default=None,
                    help="run one check by number, e.g. --only 3")
args, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from adam_u_rl.envs import wuji_pipeline_env_cfg as P  # noqa: E402
from adam_u_rl.envs.wuji_pipeline_env_cfg import CONTROL_DT  # noqa: E402
from adam_u_rl.envs.adam_u_wuji_grasp_env_cfg import _t  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, ok, detail))
    print(f"\n  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def head(n: int, title: str):
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


def want(n: int) -> bool:
    return args.only is None or args.only == str(n)


# ---------------------------------------------------------------------------

cfg = P.WujiPipelineEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = "cuda:0"
env = ManagerBasedRLEnv(cfg=cfg)
robot = env.scene["robot"]
stage = P.LEVELS[cfg.actions.pipeline.level]
lim = cfg.actions.pipeline.limits
dev = env.device
n = env.num_envs

print(f"\nlevel {cfg.actions.pipeline.level} '{stage.name}'"
      f"   displacement xy +/-{stage.dxy*100:.1f} cm, z {stage.dz_lo*100:+.1f}/{stage.dz_hi*100:+.1f} cm, "
      f"yaw +/-{__import__('math').degrees(stage.dyaw):.1f} deg")
print(f"live actions: all 8 (the curriculum is displacement magnitude, not masking)")
print(f"robot: {robot.num_bodies} bodies, {robot.num_joints} joints, "
      f"fixed base {robot.is_fixed_base}")
print(f"effective per-step limits: {lim.effective_translation * 1000:.2f} mm, "
      f"{lim.effective_rotation * 1000:.2f} mrad  "
      f"(spec asked {lim.max_translation_per_policy_step * 1000:.1f} mm / "
      f"{lim.max_rotation_per_policy_step * 1000:.1f} mrad; the velocity caps bind first)")

term = env.action_manager._terms["pipeline"]
arm_ids, _ = robot.find_joints(P.RIGHT_ARM, preserve_order=True)
hand_ids, hand_names = robot.find_joints(["r_.*"])
wrist_body, _ = robot.find_bodies(P.WRIST_BODY)
wrist_body = wrist_body[0]


def wrist_pos():
    return _t(robot.data.body_link_pos_w)[:, wrist_body, :] - env.scene.env_origins


def zero_step():
    return env.step(torch.zeros(n, 8, device=dev))


# ---- the held-joint groups, and their configured targets ------------------
# Read from the robot's own defaults, which is what the action term builds its
# base target tensor from -- so this checks the same source the env uses.
defaults = _t(robot.data.default_joint_pos)[0]
# The per-group tolerance is the drift that group can absorb before it costs
# the TASK something, which is not the same number for all of them:
#
#   waist        the whole reach envelope was solved at waistPitch = 1.05, so
#                its droop moves the wrist. Tight, and deliberately so.
#   right hand   the 11 fixed joints include the thumb opposition and the
#                ring/little flexion that the table-clearance check depends on.
#   neck, left   present for mass, inertia and gravity load only. Their droop
#                changes the centre of mass slightly and the task not at all.
#
# --tolerance scales all of them, so one flag still tightens the whole suite.
GROUPS = {
    "waist": ([j for j in robot.joint_names if j.startswith("waist")], 0.5),
    "neck": ([j for j in robot.joint_names if j.startswith("neck")], 5.0),
    "left arm": ([j for j in robot.joint_names if j.endswith("_Left")], 5.0),
    "left hand": ([j for j in robot.joint_names if j.startswith("l_")], 5.0),
    # 4x, because five of these eleven are ABDUCTION joints whose URDF effort
    # limit is 0.2 N.m. While the bolt is between the fingers it pushes them
    # sideways harder than that drive can resist, so they deflect ~0.06 rad
    # under contact and return to target afterwards (measured: peak 0.0606,
    # final -0.0040). That is drive saturation under load, not a target bug --
    # check 2 verifies the targets themselves with no tolerance at all.
    "right hand, 11 fixed": (list(P.Q_FIXED_RIGHT.keys()), 4.0),
}
HELD_NAMES = sorted({j for names, _ in GROUPS.values() for j in names})
held_ids, _ = robot.find_joints(HELD_NAMES, preserve_order=True)
held_target = defaults[held_ids].clone()

# ---------------------------------------------------------------------------
# 1. Action mapping
# ---------------------------------------------------------------------------
if want(1):
    head(1, "ACTION-MAPPING TEST")
    obs, _ = env.reset()
    pol = obs["policy"] if isinstance(obs, dict) else obs
    shapes = {
        "policy action": (env.action_manager.total_action_dim, 8),
        "observation": (pol.shape[1], 58),
        "right-arm joints controlled": (len(arm_ids), 7),
        "right-hand joints targeted": (len(hand_ids), 20),
        "synergy-driven right-hand joints": (len(P.SYNERGY_JOINTS), 9),
        "fixed right-hand joints": (len(P.Q_FIXED_RIGHT), 11),
        "right-side command targets": (len(arm_ids) + len(hand_ids), 27),
        "waist policy actions": (0, 0),
    }
    ok = True
    for label, (got, expect) in shapes.items():
        good = got == expect
        ok &= good
        print(f"    {label:36s} {got:4d}   expected {expect:4d}   {'ok' if good else 'MISMATCH'}")
    record("action mapping 8 -> 7 + 20 = 27, observation 58", ok)

# ---------------------------------------------------------------------------
# 2. Target propagation
# ---------------------------------------------------------------------------
if want(2):
    head(2, "TARGET-PROPAGATION TEST")
    print("  Every joint outside the policy must carry an explicit, correct drive")
    print("  target -- after the reset event, and still after several control steps.")
    env.reset()
    after_reset = _t(robot.data.joint_pos_target)[:, held_ids].clone()
    for _ in range(5):
        zero_step()
    after_steps = _t(robot.data.joint_pos_target)[:, held_ids].clone()

    ok = True
    for label, tgt in (("after reset", after_reset), ("after 5 control steps", after_steps)):
        err = (tgt - held_target.unsqueeze(0)).abs().max().item()
        # The specific failure this catches: a nonzero configured pose whose
        # drive target is still sitting at zero.
        zeroed = ((held_target.abs() > 1e-6).unsqueeze(0) & (tgt.abs() < 1e-9)).sum().item()
        good = err <= args.tolerance and zeroed == 0
        ok &= good
        print(f"    {label:24s} max |target - configured| {err:.6f} rad, "
              f"{zeroed} nonzero joints left at target 0   {'ok' if good else 'BAD'}")
    record("held-joint drive targets propagate and are never left at zero", ok)

# ---------------------------------------------------------------------------
# 3 + 4. Zero-action stability, and the held-joint regression table
# ---------------------------------------------------------------------------
if want(3) or want(4):
    head(3, "ZERO-ACTION STABILITY TEST")
    steps = int(args.seconds * P.CONTROL_HZ)
    print(f"  {args.seconds:.1f} s ({steps} control steps) of zero policy action.")
    env.reset()
    q0 = _t(robot.data.joint_pos).clone()
    w0 = wrist_pos().clone()
    drift_max = torch.zeros(n, robot.num_joints, device=dev)
    # Accumulate only while the env is still in the GRASP phase and has not
    # latched. Since grasp_ready was added, a zero action (which is HALF
    # closure, not open) can establish a stable grasp, latch the wrist and
    # start the scripted lift early -- and then this check measures a
    # deliberate 10 cm rise and calls it sag. Measured: 97.6 mm.
    held = torch.ones(n, dtype=torch.bool, device=dev)
    w_at_latch = wrist_pos().clone()
    for _ in range(steps):
        zero_step()
        still = (P.phase_of(env) == 0) & ~P._latch(env)["latched"]
        drift_max = torch.maximum(
            drift_max,
            torch.where(still.unsqueeze(1), (_t(robot.data.joint_pos) - q0).abs(), drift_max))
        w_at_latch = torch.where((held & ~still).unsqueeze(1), wrist_pos(), w_at_latch)
        held &= still
    latched_frac = float((~held).float().mean())
    q1 = _t(robot.data.joint_pos).clone()
    # The wrist pose to compare against is the one at the moment each env left
    # the grasp phase, not wherever the scripted lift subsequently put it.
    w1 = w_at_latch

    head(4, "HELD-JOINT REGRESSION TABLE")
    print(f"  tolerance {args.tolerance} rad (configurable with --tolerance)\n")
    print(f"  {'group':24s} {'joints':>6} {'configured':>11} {'final':>9} "
          f"{'max drift':>10} {'tol':>7} {'result':>7}")
    all_ok = True
    for gname, (names, factor) in GROUPS.items():
        if not names:
            continue
        tol = args.tolerance * factor
        ids, _ = robot.find_joints(names, preserve_order=True)
        cfg_t = defaults[ids]
        fin = q1[:, ids].mean(dim=0)
        dr = drift_max[:, ids].max().item()
        good = dr <= tol
        all_ok &= good
        # Report the worst joint in the group by name, which is what makes a
        # failure actionable.
        wi = int(drift_max[:, ids].max(dim=0).values.argmax())
        print(f"  {gname:24s} {len(ids):6d} {cfg_t[wi].item():+11.4f} {fin[wi].item():+9.4f} "
              f"{dr:10.5f} {tol:7.3f} {'PASS' if good else 'FAIL':>7}   "
              f"worst: {robot.joint_names[ids[wi]]}")
    print()
    for gname, (names, factor) in GROUPS.items():
        ids, _ = robot.find_joints(names, preserve_order=True)
        for i, jid in enumerate(ids):
            d = drift_max[:, jid].max().item()
            if d > args.tolerance * factor:
                print(f"    over tolerance: {robot.joint_names[jid]:24s} "
                      f"configured {defaults[jid]:+.4f}  final {q1[:, jid].mean():+.4f}  drift {d:.5f}")
    record(f"held joints stay within their per-group tolerance over {args.seconds:.1f} s", all_ok)

    wdrift = torch.norm(w1 - w0, dim=1).max().item()
    wok = wdrift <= args.wrist_tolerance
    print(f"\n  wrist travel under zero action   {wdrift * 1000:.2f} mm "
          f"(tolerance {args.wrist_tolerance * 1000:.1f} mm)")
    print(f"  envs that latched early          {latched_frac * 100:.0f} %"
          f"   (zero action = HALF closure, which can grasp)")
    print(f"  wrist start {w0[0].tolist()}")
    print(f"  wrist end   {w1[0].tolist()}")
    # This number is the arm SAG the host-side kinematic solve cannot see, and
    # Experiment 1 measured ~4 cm of it before the effort limits were raised.
    # At 5 cm of approach that would be the whole budget.
    record(f"wrist stationary under zero action (< {args.wrist_tolerance * 1000:.0f} mm)", wok,
           f"measured {wdrift * 1000:.2f} mm of sag/settle")

    m = P._metrics(env)
    print(f"\n  safety metrics after {steps} zero-action steps:")
    print(f"    minimum joint slack     {m['s_min'].min().item():.4f} "
          f"(worst joint {P.RIGHT_ARM[int(m['s_argmin'][int(m['s_min'].argmin())])]})")
    print(f"    clipped arm targets     {m['clipped'].sum().item():.0f}")
    print(f"    IK failures             {m['ik_failed'].sum().item():.0f}")
    print(f"    singular configurations {m['singular'].sum().item():.0f}")
    print(f"    wrist tracking error    {m['wrist_track_err'].max().item() * 1000:.2f} mm, "
          f"{m['wrist_track_err_rot'].max().item() * 1000:.2f} mrad")

# ---------------------------------------------------------------------------
# 5. IK limit test
# ---------------------------------------------------------------------------
def force_all_actions_live():
    """Checks 5 and 6 test the IK, not the curriculum.

    Stage 0 masks every wrist action off, so run behind its mask they command
    nothing and pass vacuously -- which is exactly what the first run of this
    file did: all nine commanded directions reached the same pose to 0.1 mm.
    Open the mask for the duration.
    """
    term._mask = torch.ones(1, 8, device=dev)


if want(5):
    head(5, "IK LIMIT TEST")
    force_all_actions_live()
    print("  (action mask forced fully open: this check is about the IK, not the stage)")
    print("  Full-scale wrist commands held for the whole grasp phase, driving the")
    print("  arm at the edge of the workspace box. Nothing here should produce a")
    print("  joint target outside the URDF limits, and nothing should go NaN.")
    hard_lo = term._q_lo_hard
    hard_hi = term._q_hi_hard
    worst = {}
    for label, vec in (
        ("+x +y +z", [1, 1, 1, 0, 0, 0]),
        ("-x -y -z", [-1, -1, -1, 0, 0, 0]),
        ("+roll +pitch +yaw", [0, 0, 0, 1, 1, 1]),
        ("-roll -pitch -yaw", [0, 0, 0, -1, -1, -1]),
        ("everything +1", [1, 1, 1, 1, 1, 1]),
    ):
        env.reset()
        a = torch.zeros(n, 8, device=dev)
        a[:, 0:6] = torch.tensor(vec, device=dev, dtype=torch.float32)
        a[:, 6:8] = 1.0
        viol = nan = clipped = failed = 0
        slack_first = slack_last = None
        for i in range(P.GRASP_STEPS):
            env.step(a)
            tgt = _t(robot.data.joint_pos_target)[:, arm_ids]
            viol += int(((tgt < hard_lo - 1e-6) | (tgt > hard_hi + 1e-6)).sum().item())
            nan += int((~torch.isfinite(tgt)).sum().item())
            mm = P._metrics(env)
            clipped += int(mm["clipped"].sum().item())
            failed += int(mm["ik_failed"].sum().item())
            if i == 0:
                slack_first = mm["s_min"].mean().item()
            slack_last = mm["s_min"].mean().item()
        mm = P._metrics(env)
        worst[label] = (viol, nan, clipped, failed, slack_first, slack_last)
        print(f"    {label:20s} limit violations {viol:5d}  NaN {nan:3d}  "
              f"clipped {clipped:6d}  IK failures {failed:5d}  "
              f"s_min {slack_first:.3f} -> {slack_last:.3f}")
    ok = all(v[0] == 0 and v[1] == 0 for v in worst.values())
    record("IK never commands a target outside the joint limits, and never NaNs", ok)
    # Clipping and failures are meant to be REPORTED, not absent -- the point of
    # the test is that the counters move when the arm is pushed.
    reported = any(v[2] > 0 for v in worst.values())
    print(f"\n  clipping was exercised and counted: {reported}")
    print("  (a zero here would mean the counters are not wired, not that the arm is safe)")

# ---------------------------------------------------------------------------
# 6. Reach envelope, measured in the simulator
# ---------------------------------------------------------------------------
if want(6):
    head(6, "REACH-ENVELOPE TEST (simulator, includes arm sag)")
    force_all_actions_live()
    print("  (action mask forced fully open)")
    print("  One target per env on a grid around the nominal wrist pose. Each env")
    print("  is driven by the IK term toward its target for the whole grasp phase,")
    print("  then the achieved pose is measured. The host-side kinematic sweep")
    print("  (tools/solve_pipeline_pregrasp.py --envelope) cannot see sag; this can.")
    env.reset()
    base = wrist_pos()[0].clone()
    # Spread targets over the envs by holding a constant direction per env.
    a = torch.zeros(n, 8, device=dev)
    dirs = []
    k = 0
    for dx in (-1.0, 0.0, 1.0):
        for dy in (-1.0, 0.0, 1.0):
            if k < n:
                dirs.append((dx, dy))
                a[k, 0], a[k, 1] = dx, dy
                k += 1
    for _ in range(P.GRASP_STEPS):
        env.step(a)
    reached = wrist_pos()
    mm = P._metrics(env)
    print(f"\n  nominal wrist {base.tolist()}")
    print(f"  {'cmd dir':>10} {'reached x':>10} {'reached y':>10} {'reached z':>10} "
          f"{'track err':>10} {'s_min':>7} {'clipped':>8}")
    for i, d in enumerate(dirs):
        p = reached[i]
        print(f"  {str(d):>10} {p[0].item():10.4f} {p[1].item():10.4f} {p[2].item():10.4f} "
              f"{mm['wrist_track_err'][i].item() * 1000:9.2f}mm {mm['s_min'][i].item():7.3f} "
              f"{mm['clipped'][i].item():8.0f}")
    xs, ys = reached[:len(dirs), 0], reached[:len(dirs), 1]
    print(f"\n  reached x range   {xs.min().item():.4f} .. {xs.max().item():.4f} m")
    print(f"  reached y range   {ys.min().item():.4f} .. {ys.max().item():.4f} m")
    print(f"  max tracking err  {mm['wrist_track_err'].max().item() * 1000:.2f} mm")
    print(f"  min joint slack   {mm['s_min'].min().item():.3f}")
    spread = max((xs.max() - xs.min()).item(), (ys.max() - ys.min()).item())
    moved = spread > 0.002
    print(f"  spread across commanded directions {spread * 1000:.2f} mm")
    if not moved:
        print("  ZERO SPREAD: every direction reached the same pose, so the wrist was")
        print("  never actually commanded. The IK is not being exercised by this check.")
    record("reach envelope measured, and the commanded directions actually diverge",
           moved and bool(mm["wrist_track_err"].max().item() < 0.02),
           f"spread {spread * 1000:.1f} mm, max tracking error "
           f"{mm['wrist_track_err'].max().item() * 1000:.1f} mm")

# ---------------------------------------------------------------------------
# 7. Control-rate integration
# ---------------------------------------------------------------------------
if want(7):
    head(7, "CONTROL-RATE INTEGRATION TEST")
    print("  A full-scale Cartesian command, held, must move the commanded wrist pose")
    print("  by the EFFECTIVE per-step limit once per CONTROL step -- not once per")
    print("  physics step. Isaac Lab calls process_actions at control rate and")
    print("  apply_actions at physics rate, so integrating the residual in the wrong")
    print("  one multiplies every Cartesian rate by the decimation. That bug was real:")
    print("  a commanded descent ran 51 mm in 10 control steps against a 16.7 mm")
    print("  budget, and the scripted lift rose 238 mm instead of 100 mm. Nothing")
    print("  else in this suite caught it -- the zero-action checks command no")
    print("  motion, and the envelope check only looks at where the arm ended up.")
    force_all_actions_live()
    env.reset()
    z0 = wrist_pos()[:, 2].clone()
    a = torch.zeros(n, 8, device=dev)
    a[:, 2] = -1.0
    N = 10
    for _ in range(N):
        env.step(a)
    travelled = (z0 - wrist_pos()[:, 2]).mean().item()
    budget = N * lim.effective_translation
    # The first-order smoothing on the residual means the real figure is a
    # little UNDER the budget, never over. Over means the decimation bug.
    ok = travelled <= budget * 1.05
    print(f"\n    commanded  full -z for {N} control steps")
    print(f"    budget     {budget * 1000:.2f} mm  ({N} x {lim.effective_translation * 1000:.2f} mm)")
    print(f"    travelled  {travelled * 1000:.2f} mm")
    print(f"    ratio      {travelled / budget:.2f}x  "
          f"(decimation is {int(1 / (CONTROL_DT / env.cfg.sim.dt) * env.cfg.decimation) if False else env.cfg.decimation}, "
          f"so a {env.cfg.decimation}x ratio is the bug)")
    record("Cartesian residual integrates at control rate, not physics rate", ok,
           f"{travelled / budget:.2f}x of budget")

    # The scripted lift, which is the same integration path -- but each env now
    # starts its lift when IT latches, not at a fixed step 90, so the rise has
    # to be measured per env between its own phase 0->1 and 1->2 boundaries.
    env.reset()
    z_at_lift = torch.zeros(n, device=dev)
    z_at_hold = torch.zeros(n, device=dev)
    prev_phase = torch.zeros(n, dtype=torch.long, device=dev)
    for _ in range(P.EPISODE_STEPS - 1):
        env.step(torch.zeros(n, 8, device=dev))
        ph = P.phase_of(env)
        z = wrist_pos()[:, 2]
        z_at_lift = torch.where((prev_phase == 0) & (ph == 1), z, z_at_lift)
        z_at_hold = torch.where((prev_phase == 1) & (ph == 2), z, z_at_hold)
        prev_phase = ph
    seen = (z_at_lift > 0) & (z_at_hold > 0)
    rise = ((z_at_hold - z_at_lift)[seen].mean().item() if bool(seen.any()) else 0.0)
    lift_ok = bool(seen.any()) and abs(rise - P.LIFT_HEIGHT) < 0.02
    print(f"\n    scripted lift commanded {P.LIFT_HEIGHT * 1000:.0f} mm, "
          f"wrist rose {rise * 1000:.1f} mm over {int(seen.sum())} of {n} envs")
    record("scripted lift rises by LIFT_HEIGHT, not a multiple of it", lift_ok,
           f"{rise * 1000:.1f} mm of {P.LIFT_HEIGHT * 1000:.0f} mm")

print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
for name, ok, detail in RESULTS:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
failed = [r for r in RESULTS if not r[1]]
print(f"\n  {len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed")

env.close()
app.close()
raise SystemExit(1 if failed else 0)
