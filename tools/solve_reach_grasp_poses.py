#!/usr/bin/env python3
"""Arm poses and reset Jacobians for Experiment 4's standoff heights.

Host-side, numpy + scipy, no Isaac.

Experiment 4 (docs/EXPERIMENT4_DESIGN.md) keeps the pipeline's solved pinch
pose and starts the wrist ABOVE it, along world -z, by a sampled standoff. The
reset needs an arm configuration for every sampled standoff plus a small
xy / yaw / roll / pitch error. The pipeline's trick -- one baked first-order
Jacobian at the contact pose -- was validated to 1.4 mm at 2 cm and would not
hold at 10 cm, so this tool bakes one arm solution AND one damped
pseudo-inverse Jacobian per standoff "bucket". At reset the env picks the
bucket nearest the sampled standoff and applies only the residual through
that bucket's Jacobian, which keeps every residual inside the validated range.

For each bucket height it prints, ready to paste into
genesis_rl/envs/wuji_reach_grasp_env_cfg.py:

    the 7 right-arm joint angles (waistPitch held at the pipeline's 1.05)
    the position / orientation residual of the solve and the minimum slack
    the 7x6 damped pseudo-inverse Jacobian J_SHARP

and it CHECKS ITSELF against the pipeline: the bucket at 0 cm must reproduce
PREGRASP_ARM_CONTACT and its Jacobian must reproduce the pipeline's J_SHARP,
otherwise the method is not the one those numbers came from.

    python3 tools/solve_reach_grasp_poses.py
    python3 tools/solve_reach_grasp_poses.py --heights 0 0.05 0.10 --verify
"""

from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


pp = _load("solve_pipeline_pregrasp")

# The pipeline's solved pose (wuji_pipeline_env_cfg.py), copied rather than
# imported: that module needs Isaac Lab.
WRIST_XY = (0.50, -0.20)
WRIST_Z_CONTACT = 0.95 + 0.0592
WRIST_ROT = R.from_quat([-0.1821, -0.7296, 0.4661, 0.4661])
WAIST_HELD = 1.05
ARM = pp.ARM
PIPELINE_CONTACT = [-0.9142, -0.3996, +0.1797, -1.4277, +0.9023, -0.3292, -0.3582]
PIPELINE_J_SHARP = np.array([
    (-3.331483, -1.038766, +1.437886, +0.026061, +0.003865, -0.052771),
    (-0.103434, +1.866384, -0.781030, +0.120791, +0.024376, -0.248876),
    (-0.716273, +1.249422, -0.142269, -0.148744, -0.027613, +0.314144),
    (+2.918059, +0.164837, -4.983479, -0.020740, +0.003640, +0.037611),
    (-0.859473, +1.444512, -1.445427, -0.666762, -0.665387, -0.044996),
    (-0.555831, +1.850984, +2.014353, -0.546271, +0.268106, -0.646784),
    (+0.917511, +0.056436, +2.936675, -0.149386, +0.935722, +0.247245),
])


def jacobian(arm, q_arm, eps=1e-6, held=(WAIST_HELD,)):
    """(6, 7) world-frame geometric Jacobian of the wrist body, by central differences.

    Rows 0-2: position. Rows 3-5: rotation as a small world-frame rotation
    vector, i.e. the same convention the env's IK uses (axis-angle error
    applied in the world frame). `held` are the values of the arm's extra
    (pinned) variables after the 7 arm joints: the waist pitch, and the waist
    yaw when it is part of the chain."""
    x0 = np.append(q_arm, held)
    T0 = arm.fk(x0)
    R0 = R.from_matrix(T0[:3, :3])
    J = np.zeros((6, 7))
    for i in range(7):
        xp, xm = x0.copy(), x0.copy()
        xp[i] += eps
        xm[i] -= eps
        Tp, Tm = arm.fk(xp), arm.fk(xm)
        J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * eps)
        # world-frame rotation vector taking R0 -> R(x)
        wp = (R.from_matrix(Tp[:3, :3]) * R0.inv()).as_rotvec()
        wm = (R.from_matrix(Tm[:3, :3]) * R0.inv()).as_rotvec()
        J[3:, i] = (wp - wm) / (2 * eps)
    return J


def damped_pinv(J, lam):
    return J.T @ np.linalg.inv(J @ J.T + (lam ** 2) * np.eye(6))


# The bolt sits at a fixed offset from the wrist in the HAND's frame, so it
# yaws with the hand (solve_pipeline_pregrasp.bolt_xy). Bolt xy - wrist xy at
# the pipeline's yaw:
BOLT_OFFSET_XY = np.array([0.5474 - WRIST_XY[0], -0.1099 - WRIST_XY[1]])


# The waist, when freed: its three joints (chain: waistRoll, waistPitch,
# waistYaw, torso) are held at ONE triple per (bolt yaw, xy) column for the
# whole episode, chosen from this scan by the best worst-height arm slack,
# nearest-to-nominal first so a column the nominal hold serves keeps it.
# Joint limits: pitch -0.663..1.361, yaw +/-0.829, roll +/-0.279 rad.
WAIST_NOMINAL = (WAIST_HELD, 0.0, 0.0)                      # (pitch, yaw, roll)
WAIST_JOINTS = ("waistPitch", "waistYaw", "waistRoll")
WAIST_PITCH_SCAN = (1.05, 0.9, 1.2)
WAIST_YAW_SCAN = (0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75)
WAIST_ROLL_SCAN = (0.0, 0.2, -0.2)
WAIST_GOOD = 0.25              # stop scanning once the column's worst slack reaches this


def waist_scan():
    combos = [(p, y, r) for p in WAIST_PITCH_SCAN for y in WAIST_YAW_SCAN for r in WAIST_ROLL_SCAN]
    key = lambda c: abs(c[0] - WAIST_HELD) / 0.15 + abs(c[1]) / 0.25 + abs(c[2]) / 0.2  # noqa: E731
    return sorted(combos, key=key)


# Arm-link proxy spheres (body, offset in the body frame, radius) and the table
# box, for the solver's own clearance check: a hold that drops the elbow or
# upper arm into the table is not a reachable bucket, whatever its slack.
# KEEP IN SYNC with PROXY_GROUPS["forearm"], ["upperarm"] and the table in
# genesis_rl/tasks/wuji_reach_grasp/env_cfg.py; tools/test_reach_grasp.py
# checks that they match.
ARM_PROXIES = (
    ("elbowRight", (-0.013, -0.029, -0.031), 0.045),
    ("wristYawRight", (0.0, 0.0, -0.020), 0.048),
    ("wristYawRight", (0.0, 0.0, -0.095), 0.048),
    ("wristYawRight", (0.0, 0.0, -0.170), 0.048),
    ("shoulderYawRight", (0.001, -0.003, -0.020), 0.050),
    ("shoulderYawRight", (0.001, -0.003, -0.120), 0.050),
    ("shoulderYawRight", (0.001, -0.003, -0.220), 0.050),
    ("shoulderRollRight", (-0.033, -0.017, -0.010), 0.065),
)
TABLE_LO = (0.30, -0.40, 0.0)          # assets/workcell/workcell_single.json
TABLE_HI = (0.80, 0.40, 0.95)
# The least arm-link clearance a bucket may have. The standoff buckets (5 and
# 10 cm) are where episodes START, plus up to 2 cm of sampled reset error, so
# they need more; the contact bucket only has to exist for the descent.
ARM_CLEAR_MIN_CONTACT = 0.01           # m, h < 2.5 cm
ARM_CLEAR_MIN_STANDOFF = 0.025         # m, h >= 2.5 cm


def arm_clear_min(h: float) -> float:
    return ARM_CLEAR_MIN_CONTACT if h < 0.025 else ARM_CLEAR_MIN_STANDOFF


def arm_clearance(arm, x) -> float:
    """Least clearance of the forearm / upper-arm proxy spheres to the table box, metres (negative = inside)."""
    lo, hi = np.array(TABLE_LO), np.array(TABLE_HI)
    q = dict(zip(arm.vars, x))
    worst = np.inf
    for body, off, r in ARM_PROXIES:
        sub = arm.path[:next(i for i, (n, j) in enumerate(arm.path) if j["child"] == body) + 1]
        T = arm.m.fk(sub, q)
        c = T[:3, 3] + T[:3, :3] @ np.array(off)
        d = np.maximum(np.maximum(lo - c, 0.0), c - hi)
        if (d > 0).any():
            clear = np.linalg.norm(d) - r
        else:
            clear = -(np.minimum(c - lo, hi - c).min() + r)
        worst = min(worst, clear)
    return float(worst)


def _make_arm(free_waist: bool):
    """The arm chain with waistPitch pinned; with the waist freed, all three
    waist joints are variables (pinned per scan value by the caller)."""
    arm = pp.Arm(pp._load_solver(), waist_max=1.36)
    arm.lo[-1], arm.hi[-1] = WAIST_HELD - 1e-6, WAIST_HELD + 1e-6
    if free_waist:
        arm.vars = ARM + list(WAIST_JOINTS)
        arm.lo = np.concatenate([arm.lo[:7], [WAIST_HELD - 1e-6, -1e-6, -1e-6]])
        arm.hi = np.concatenate([arm.hi[:7], [WAIST_HELD + 1e-6, 1e-6, 1e-6]])
        arm.ready = np.concatenate([arm.ready[:7], WAIST_NOMINAL])
    return arm


def _pin_waist(arm, w):
    arm.lo[7:] = np.array(w) - 1e-6
    arm.hi[7:] = np.array(w) + 1e-6
    arm.ready[7:] = w


def _solve_pose(arm, pos, rot, seed_q, held, seeds):
    """One wrist pose: least squares from the warm start, the multi-seed solve as a fallback."""
    from scipy.optimize import least_squares
    x0 = np.append(seed_q, held)

    def resid(x):
        T = arm.fk(x)
        ori = (rot.inv() * R.from_matrix(T[:3, :3])).as_rotvec()
        return np.concatenate([T[:3, 3] - pos, ori * 0.15, 0.002 * (x - arm.ready)])
    sol = least_squares(resid, np.clip(x0, arm.lo + 1e-7, arm.hi - 1e-7), bounds=(arm.lo, arm.hi))
    r = arm.report(sol.x, pos, rot)
    if r["pos_err"] > 0.003 or np.degrees(r["ori_err"]) > 1.5:
        r2 = arm.solve(pos, rot, seeds=seeds)
        if r2["pos_err"] < r["pos_err"]:
            r = r2
    r["arm_clear"] = arm_clearance(arm, r["q"])
    r["ok"] = r["pos_err"] <= 0.003 and np.degrees(r["ori_err"]) <= 1.5 and r["s_min"] > 0.05
    return r


# The rod's axis direction at hand yaw 0, from the workcell spawn: lying = 90 deg about X (mesh z -> -y),
# then yawed BOLT_YAW_DEG about Z. Its +end is the one the tilted-rod reset raises; the wrist roll for a
# tilt theta is a rotation of -theta about n = z x u (horizontal), which turns the hand's approach axis
# from vertical to the rod's up-normal. tools/test_reach_grasp.py checks the env agrees with this vector.
BOLT_YAW_DEG = -136.3
U0 = np.array([np.cos(np.radians(BOLT_YAW_DEG)), np.sin(np.radians(BOLT_YAW_DEG)), 0.0])   # measured in the env: (-0.723, -0.691)
BOLT_REST_Z = 0.961                 # the lying rod's centre height (TABLE_TOP + 0.011)
OFF3 = np.array([BOLT_OFFSET_XY[0], BOLT_OFFSET_XY[1], BOLT_REST_Z - WRIST_Z_CONTACT])   # rod centre - wrist, hand yaw 0, contact


def wrist_pose_for(psi, tilt, g, h):
    """Wrist (pos, R) that puts the pinch on the grasp point g (world) for a rod at hand yaw psi tilted
    `tilt` from horizontal, at standoff h along the approach axis."""
    Rz = R.from_rotvec([0, 0, psi])
    u = Rz.apply(U0)
    n = np.cross([0, 0, 1.0], u)
    Rt = R.from_rotvec(-tilt * n / np.linalg.norm(n)) if abs(tilt) > 1e-9 else R.identity()
    Rw = Rt * Rz
    m = Rw.apply([0, 0, 1.0])
    pos = g - Rw.apply(OFF3) + h * m
    return pos, Rw * WRIST_ROT


def _solve_yaw(psi, offs, heights, lam, seeds, free_waist, min_slack, tilts=(0.0,)):
    """All (xy, height, tilt) buckets of one bolt yaw. Returns rows of
    (psi, dx, dy, h, q, J_sharp, s_min, pos_err, ori_err_deg, ok, waist(3), arm_clear, tilt)."""
    arm = _make_arm(free_waist)
    q0 = np.array(PIPELINE_CONTACT)
    Rz = R.from_rotvec([0, 0, psi])
    rows, warm = [], {}
    scan = waist_scan() if free_waist else (None,)
    for tilt in tilts:
      for dx in offs:
        for dy in offs:
            bolt = np.array([0.5474 + dx, -0.1099 + dy])
            wxy = bolt - Rz.as_matrix()[:2, :2] @ BOLT_OFFSET_XY
            g = np.array([bolt[0], bolt[1], BOLT_REST_Z])
            step = offs[1] - offs[0] if len(offs) > 1 else 0.0
            seed = warm.get((round(dx, 4), round(dy, 4)),
                            warm.get((round(dx - step, 4), round(dy, 4)), q0))
            best = None                      # (score, waist, per-height reports)
            for w in scan:
                held = (WAIST_HELD,) if w is None else tuple(w)
                if w is not None:
                    _pin_waist(arm, w)
                reps, s = [], []
                for h in heights:
                    pos, rot = wrist_pose_for(psi, tilt, g, h)
                    r = _solve_pose(arm, pos, rot, seed, held, seeds)
                    r["ok"] = r["ok"] and r["s_min"] >= min_slack and r["arm_clear"] >= arm_clear_min(h)
                    reps.append(r)
                    s.append(r["s_min"] if r["ok"] else -1.0)
                    if not r["ok"]:
                        break            # this hold cannot serve the column
                score = min(s) if len(reps) == len(heights) else -1.0
                if best is None or score > best[0] + 1e-6:
                    best = (score, w, reps)
                if score >= WAIST_GOOD:
                    break
            score, w, reps = best
            held = (WAIST_HELD,) if w is None else tuple(w)
            waist = WAIST_NOMINAL if w is None else tuple(w)
            for h, r in zip(heights, reps):
                q = np.array(r["q"][:7])
                ok = bool(r["ok"]) and score > 0.0
                if ok:
                    warm[(round(dx, 4), round(dy, 4))] = q
                Js = damped_pinv(jacobian(arm, q, held=held), lam) if ok else np.zeros((7, 6))
                rows.append((psi, dx, dy, h, q, Js, r["s_min"], r["pos_err"], np.degrees(r["ori_err"]), ok, waist,
                             r["arm_clear"], tilt))
            for i in range(len(reps), len(heights)):    # heights the chosen hold never reached
                rows.append((psi, dx, dy, heights[i], q0, np.zeros((7, 6)), -1.0, 1.0, 180.0, False, waist, -1.0, tilt))
    return rows


def solve_grid(args) -> int:
    """Arm solutions + Jacobians over bolt yaw x bolt xy x standoff height.

    For a bolt yawed by psi about world z (relative to the pipeline's pose)
    and shifted by (dx, dy), the pinch is the same pinch rotated by psi: the
    wrist target is bolt_xy - R(psi) @ BOLT_OFFSET_XY at the same height, with
    orientation R(psi) * WRIST_ROT. Every pinch metric and the table clearance
    are invariant to psi; only arm reachability changes, which is what the
    per-bucket s_min reports.

    With --free-waist the three waist joints (chain: waistRoll, waistPitch,
    waistYaw, torso) are held variables, chosen per (yaw, xy) column from
    waist_scan() by the best worst-height arm slack, and written to the bucket
    file as `waist` (pitch, yaw, roll). The env holds them for the episode
    exactly as it holds the pitch at 1.05 today. One triple per column,
    because the arm descends through every height of the column in one
    episode. --min-slack sets the slack a bucket needs to count as reachable.
    """
    import time
    from multiprocessing import Pool
    lam = args.damping if args.damping is not None else 0.05
    yaws = np.radians(np.arange(args.yaw_deg[0], args.yaw_deg[1] + 1e-9, args.yaw_deg[2]))
    offs = np.arange(args.xy_cm[0], args.xy_cm[1] + 1e-9, args.xy_cm[2]) / 100.0
    heights = np.array(args.heights)
    t0 = time.time()
    total = len(yaws) * len(offs) ** 2 * len(heights) * len(args.tilts_deg)
    tilts = tuple(np.radians(args.tilts_deg))
    jobs = [(float(psi), offs, heights, lam, args.seeds, args.free_waist, args.min_slack, tilts) for psi in yaws]
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        per_yaw = pool.starmap(_solve_yaw, jobs)
    rows = [r for rs in per_yaw for r in rs]
    print(f"  {len(rows)}/{total} buckets in {time.time() - t0:.0f} s, reachable {sum(1 for r in rows if r[9])}")
    default = os.path.join(HERE, "..", "genesis_rl", "tasks", "wuji_reach_grasp",
                           "pose_buckets_tilt.npz" if len(args.tilts_deg) > 1 or args.tilts_deg[0] != 0.0
                           else "pose_buckets_waist.npz" if args.free_waist else "pose_buckets.npz")
    out = os.path.abspath(args.out or default)
    np.savez(out,
             yaw=np.array([r[0] for r in rows]), dx=np.array([r[1] for r in rows]), dy=np.array([r[2] for r in rows]),
             h=np.array([r[3] for r in rows]), q=np.array([r[4] for r in rows]), j_sharp=np.array([r[5] for r in rows]),
             s_min=np.array([r[6] for r in rows]), pos_err=np.array([r[7] for r in rows]),
             ori_err_deg=np.array([r[8] for r in rows]), ok=np.array([r[9] for r in rows]),
             waist=np.array([r[10] for r in rows]), arm_clear=np.array([r[11] for r in rows]),
             tilt=np.array([r[12] for r in rows]),
             free_waist=bool(args.free_waist), min_slack=args.min_slack,
             bolt_offset_xy=BOLT_OFFSET_XY, wrist_rot=WRIST_ROT.as_quat(), waist_pitch_held=WAIST_HELD)
    ok = np.array([r[9] for r in rows])
    print(f"\nwrote {out}: {len(rows)} buckets, {ok.sum()} reachable")
    # reachability by yaw, as a table: fraction of xy x height buckets reachable, worst s_min, waist holds used
    print(f"{'yaw deg':>8} {'reachable':>10} {'min s_min':>10} {'min clear':>10}   waist holds (pitch, yaw, roll)")
    for psi in yaws:
      for tilt in tilts:
        sel = [r for r in rows if abs(r[0] - psi) < 1e-9 and abs(r[12] - tilt) < 1e-9]
        okk = [r for r in sel if r[9]]
        ws = sorted(set(tuple(round(v, 2) for v in r[10]) for r in okk))
        if len(tilts) > 1:
            print(f"  tilt {np.degrees(tilt):5.1f}:", end="")
        print(f"{np.degrees(psi):8.1f} {len(okk)/len(sel)*100:9.0f}% "
              f"{min((r[6] for r in okk), default=float('nan')):10.3f} "
              f"{min((r[11] for r in okk), default=float('nan')):10.3f}   {ws}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heights", type=float, nargs="+", default=[0.0, 0.05, 0.10],
                    help="standoffs above the contact pose, metres")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--damping", type=float, default=None,
                    help="DLS damping for J_SHARP; default: the value that reproduces the "
                         "pipeline's baked matrix, searched over a small grid")
    ap.add_argument("--verify", action="store_true", help="also verify the first-order reset "
                    "over the level ranges by full FK")
    ap.add_argument("--grid", action="store_true",
                    help="pose-generalisation buckets: solve the arm over a grid of BOLT yaw and xy "
                         "offsets at every height and write pose_buckets.npz beside the env")
    ap.add_argument("--yaw-deg", type=float, nargs=3, default=[-90.0, 90.0, 15.0], metavar=("LO", "HI", "STEP"))
    ap.add_argument("--xy-cm", type=float, nargs=3, default=[-5.0, 5.0, 2.5], metavar=("LO", "HI", "STEP"))
    ap.add_argument("--free-waist", action="store_true",
                    help="with --grid: also choose a held waist (pitch, yaw, roll) per (bolt yaw, xy) "
                         "column; writes pose_buckets_waist.npz instead")
    ap.add_argument("--tilts-deg", type=float, nargs="+", default=[0.0],
                    help="with --grid: rod tilts from horizontal to solve the ROLLED wrist for (level 21+); "
                         "more than one, or a non-zero one, writes pose_buckets_tilt.npz")
    ap.add_argument("--min-slack", type=float, default=0.05,
                    help="with --grid: arm slack a bucket needs to count as reachable")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if args.grid:
        return solve_grid(args)

    arm = pp.Arm(pp._load_solver(), waist_max=1.36)
    arm.lo[-1], arm.hi[-1] = WAIST_HELD - 1e-6, WAIST_HELD + 1e-6

    # ---- 0. calibrate: which damping produced the pipeline's J_SHARP? --------
    q0 = np.array(PIPELINE_CONTACT)
    J0 = jacobian(arm, q0)
    if args.damping is None:
        best = None
        for lam in (0.0, 0.01, 0.02, 0.05, 0.1):
            err = np.abs(damped_pinv(J0, lam) - PIPELINE_J_SHARP).max()
            if best is None or err < best[1]:
                best = (lam, err)
        lam, err = best
        print(f"damping that reproduces the pipeline's J_SHARP: {lam:g}  (max |dJ| {err:.2e})")
    else:
        lam = args.damping
        err = np.abs(damped_pinv(J0, lam) - PIPELINE_J_SHARP).max()
        print(f"damping {lam:g}: max |dJ| vs the pipeline's J_SHARP {err:.2e}")
    if err > 1e-3:
        print("WARNING: this is NOT how the pipeline's matrix was produced; check the convention.")

    # ---- 1. one arm solution + Jacobian per standoff bucket ------------------
    print("\nSolving the arm at each standoff (waistPitch held at "
          f"{WAIST_HELD}, wrist xy {WRIST_XY}, orientation = the pipeline's pinch).\n")
    results = []
    for h in args.heights:
        pos = np.array([WRIST_XY[0], WRIST_XY[1], WRIST_Z_CONTACT + h])
        if abs(h) < 1e-9:
            # Seed from the pipeline's pose so the bucket at 0 IS the pipeline's pose.
            r = arm.report(np.append(q0, WAIST_HELD), pos, WRIST_ROT)
        else:
            r = arm.solve(pos, WRIST_ROT, seeds=args.seeds)
        q = np.array(r["q"][:7])
        Js = damped_pinv(jacobian(arm, q), lam)
        results.append((h, q, r, Js))
        print(f"  standoff {h*100:5.1f} cm   pos err {r['pos_err']*1000:.2f} mm   "
              f"ori err {np.degrees(r['ori_err']):.3f} deg   s_min {r['s_min']:.3f} ({r['worst']})   "
              f"wristRoll slack {r['wrist_roll_slack']:.3f}")

    # The bucket at 0 must be the pipeline's contact pose.
    dq = np.abs(results[0][1] - q0).max() if abs(args.heights[0]) < 1e-9 else float("nan")
    print(f"\n  bucket 0 vs PREGRASP_ARM_CONTACT: max |dq| = {dq:.2e} rad")

    # ---- 2. paste block --------------------------------------------------------
    print("\n" + "=" * 78 + "\nPASTE INTO wuji_reach_grasp_env_cfg.py\n" + "=" * 78)
    print("# Arm solutions and damped pseudo-inverse Jacobians, one per standoff bucket.")
    print("# Generated by tools/solve_reach_grasp_poses.py; regenerate if the pinch pose,")
    print("# the held waist or the wrist xy changes.")
    print("STANDOFF_BUCKETS = (")
    for h, q, r, Js in results:
        print(f"    StandoffBucket(")
        print(f"        height={h:.3f},")
        print("        arm={")
        for n, v in zip(ARM, q):
            print(f'            "{n}": {v:+.4f},')
        print("        },")
        print("        j_sharp=(")
        for row in Js:
            print("            (" + ", ".join(f"{v:+.6f}" for v in row) + "),")
        print("        ),")
        print(f"        s_min={r['s_min']:.3f},")
        print("    ),")
    print(")")

    # ---- 3. verify the first-order reset by full FK --------------------------
    if args.verify:
        print("\n" + "=" * 78 + "\nFIRST-ORDER RESET vs FULL FK\n" + "=" * 78)
        rng = np.random.default_rng(0)
        # Worst residuals the env will ever apply through one bucket: half the
        # bucket spacing in z, plus the widest level's xy / rotation error.
        spacing = min(np.diff(sorted(args.heights))) if len(args.heights) > 1 else 0.05
        cases = [
            ("+xy 2 cm", np.array([0.02, 0.02, 0.0, 0, 0, 0])),
            (f"+z {spacing/2*100:.1f} cm", np.array([0, 0, spacing / 2, 0, 0, 0])),
            (f"-z {spacing/2*100:.1f} cm", np.array([0, 0, -spacing / 2, 0, 0, 0])),
            ("yaw 7.5 deg", np.array([0, 0, 0, 0, 0, np.radians(7.5)])),
            ("roll/pitch 2 deg", np.array([0, 0, 0, np.radians(2), np.radians(2), 0])),
            ("corner", np.array([0.02, 0.02, spacing / 2, np.radians(2), np.radians(2), np.radians(7.5)])),
        ]
        for h, q, r, Js in results:
            print(f"\n  bucket {h*100:.1f} cm")
            for name, d in cases:
                qd = q + Js @ d
                T = arm.fk(np.append(qd, WAIST_HELD))
                want_p = np.array([WRIST_XY[0], WRIST_XY[1], WRIST_Z_CONTACT + h]) + d[:3]
                want_R = R.from_rotvec(d[3:]) * WRIST_ROT
                ep = np.linalg.norm(T[:3, 3] - want_p) * 1000
                er = np.degrees(np.linalg.norm((want_R.inv() * R.from_matrix(T[:3, :3])).as_rotvec()))
                s = arm.slack(np.append(qd, WAIST_HELD)).min()
                print(f"    {name:18s} pos err {ep:6.2f} mm   ori err {er:5.2f} deg   s_min {s:.3f}")
            _ = rng
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
