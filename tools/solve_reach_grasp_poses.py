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


def jacobian(arm, q_arm, eps=1e-6):
    """(6, 7) world-frame geometric Jacobian of the wrist body, by central differences.

    Rows 0-2: position. Rows 3-5: rotation as a small world-frame rotation
    vector, i.e. the same convention the env's IK uses (axis-angle error
    applied in the world frame)."""
    x0 = np.append(q_arm, WAIST_HELD)
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
    args = ap.parse_args()

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
