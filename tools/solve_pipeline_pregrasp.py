#!/usr/bin/env python3
"""Solve the staged pipeline's pre-grasp pose against the ARM, not just the hand.

Host-side, numpy + scipy, no Isaac.

tools/find_pinch_pose.py solved Experiment 2's thumb-opposed pinch for a hand
with no arm attached: it searched the sphere of hand orientations for one that
keeps every fingertip above the table through the synergy stroke, and nothing
in it knows about the Adam-U arm's joint limits. Measured, that pose is
reachable to 0.0 mm and 0.02 deg -- but only by driving wristRoll_Right onto
its -0.960 rad stop, everywhere in the reachable region. The arm can HOLD the
orientation and has no authority to leave it, in one of the three DoF the
pipeline's orientation stage exists to learn.

The fix, and why it is free: yawing the whole (hand, bolt) pair about WORLD Z
preserves every pinch metric -- the thumb-to-finger gap, the opposition angle,
the fingertip-to-bolt distance -- and preserves the table clearance too, since
clearance depends on the hand's tilt from vertical and not on its heading. The
bolt's world yaw simply rotates with it, and that was always a free parameter.
So there is a one-parameter family of equally good pinches, and only one member
of it was ever checked against the arm.

This tool searches that family jointly with the wrist's XY placement:

    for each world-z yaw of the pinch
      for each wrist xy
        solve the 7 right-arm joints (waistPitch as an 8th variable, HELD at
        the solution, never a policy action) for the wrist pose
        score: position error, orientation error, per-joint slack

and reports the pose with the most margin, plus the reach envelope around it.

    python3 tools/solve_pipeline_pregrasp.py                  # solve and report
    python3 tools/solve_pipeline_pregrasp.py --envelope       # + reach sweep
    python3 tools/solve_pipeline_pregrasp.py --yaw 90         # force a yaw

Everything it prints is meant to be pasted into
genesis_rl/envs/wuji_pipeline_env_cfg.py, and the --envelope report is the
"reach-envelope test" the architecture spec asks for.
"""

from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_solver():
    """Reuse solve_pregrasp_pose's URDF parser and FK rather than fork them."""
    spec = importlib.util.spec_from_file_location(
        "solve_pregrasp_pose", os.path.join(HERE, "solve_pregrasp_pose.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Experiment 2's solved three-finger pinch, from POSE[3] in
# genesis_rl/envs/wuji_synergy_grasp_env_cfg.py. r_wrist is joined to
# wristRollRight by a fixed joint with an IDENTITY origin, so HAND_ROT is the
# orientation of the arm's wrist body directly -- no offset to compose.
HAND_ROT = (-0.6447, -0.3871, 0.0000, 0.6592)   # (x, y, z, w)
WRIST_HEIGHT = 0.0592        # m above the table top, at first contact
BOLT_OFFSET = (0.0901, -0.0474)                 # bolt xy - wrist xy, at yaw 0
BOLT_YAW_DEG = -136.3
TABLE_TOP = 0.95
APPROACH = 0.05              # m, the pipeline starts this far above contact

ARM = ["shoulderPitch_Right", "shoulderRoll_Right", "shoulderYaw_Right",
       "elbow_Right", "wristYaw_Right", "wristPitch_Right", "wristRoll_Right"]
WAIST = "waistPitch"
WAIST_MAX = 1.0              # rad of forward lean we are willing to hold


class Arm:
    """The right-arm chain, and a bounded least-squares solve for a wrist pose."""

    def __init__(self, m, waist_max=WAIST_MAX):
        self.m = m
        self.joints = m.parse(m.URDF)
        self.path, base = m.chain_to(self.joints, "wristRollRight")
        assert base == "lifting_Columns_base", base
        self.vars = ARM + [WAIST]
        self.lo = np.array([self.joints[n]["lo"] for n in self.vars])
        self.hi = np.array([self.joints[n]["hi"] for n in self.vars])
        self.lo[-1], self.hi[-1] = 0.0, waist_max       # forward lean only
        self.ready = np.append(m.READY, 0.3)
        # Slack is reported against the ARM's own limits; the waist is held, so
        # its distance to a limit is not the policy's problem.
        self.arm_lo = self.lo[:7]
        self.arm_hi = self.hi[:7]

    def fk(self, x):
        return self.m.fk(self.path, dict(zip(self.vars, x)))

    def solve(self, pos, R_target, seeds=12, ori_weight=0.15):
        def resid(x):
            T = self.fk(x)
            ori = (R_target.inv() * R.from_matrix(T[:3, :3])).as_rotvec()
            return np.concatenate([T[:3, 3] - pos, ori * ori_weight,
                                   0.002 * (x - self.ready)])

        best = None
        for s in range(seeds):
            # A pinned variable (the held waist) has a near-zero window, so the
            # usual 0.02 inset would put x0 outside its own bounds.
            pad = np.minimum(0.02, (self.hi - self.lo) / 4.0)
            x0 = np.clip(self.ready + np.random.default_rng(s).normal(0, 0.7, len(self.ready)),
                         self.lo + pad, self.hi - pad)
            sol = least_squares(resid, x0, bounds=(self.lo, self.hi))
            if best is None or sol.cost < best.cost:
                best = sol
        return self.report(best.x, pos, R_target)

    def slack(self, x):
        """Normalised per-joint slack, the architecture spec's s_i, over the 7 arm joints.

            s_i = 2 * min(q - q_min, q_max - q) / (q_max - q_min)

        1.0 is dead centre of the range, 0.0 is on a stop.
        """
        q = x[:7]
        return 2.0 * np.minimum(q - self.arm_lo, self.arm_hi - q) / (self.arm_hi - self.arm_lo)

    def report(self, x, pos, R_target):
        T = self.fk(x)
        s = self.slack(x)
        return dict(
            q=x,
            pos_err=float(np.linalg.norm(T[:3, 3] - pos)),
            ori_err=float(np.linalg.norm((R_target.inv() * R.from_matrix(T[:3, :3])).as_rotvec())),
            slack=s,
            s_min=float(s.min()),
            worst=ARM[int(np.argmin(s))],
            wrist_roll=float(x[6]),
            wrist_roll_slack=float(s[6]),
            waist=float(x[7]),
        )


def yawed(yaw_deg: float) -> R:
    """Experiment 2's pinch, rotated about world z. Every pinch metric survives."""
    return R.from_rotvec([0, 0, np.radians(yaw_deg)]) * R.from_quat(np.array(HAND_ROT))


def bolt_xy(wrist_xy, yaw_deg: float):
    """The bolt sits at a fixed offset in the HAND's frame, so it yaws with the hand."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    dx, dy = BOLT_OFFSET
    return (wrist_xy[0] + c * dx - s * dy, wrist_xy[1] + s * dx + c * dy)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaw", type=float, default=None,
                    help="force this world-z yaw (deg) instead of searching")
    ap.add_argument("--yaw-step", type=float, default=15.0)
    ap.add_argument("--wrist-xy", type=float, nargs=2, default=None,
                    help="force the wrist xy instead of searching")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--envelope", action="store_true",
                    help="also sweep the reach envelope around the chosen pose")
    ap.add_argument("--waist-max", type=float, default=WAIST_MAX)
    ap.add_argument("--pos-tol", type=float, default=0.002, help="m, a solve counts as reached")
    ap.add_argument("--waist-grid", type=int, default=19,
                    help="how many held-waist values to try. The waist is HELD at ONE value for "
                         "the whole episode, so the arm must reach the start pose AND the contact "
                         "pose with the same lean; solving each height with a free waist silently "
                         "picks two different leans and neither pose is then valid")
    ap.add_argument("--ori-tol", type=float, default=1.0, help="deg, a solve counts as reached")
    args = ap.parse_args()

    arm = Arm(_load_solver(), waist_max=args.waist_max)
    z_contact = TABLE_TOP + WRIST_HEIGHT
    z_start = z_contact + APPROACH

    # ---- 1. choose the pinch yaw and the wrist xy ---------------------------
    xys = ([tuple(args.wrist_xy)] if args.wrist_xy else
           [(x, y) for x in np.arange(0.30, 0.51, 0.05) for y in np.arange(-0.35, -0.09, 0.05)])
    yaws = ([args.yaw] if args.yaw is not None else
            list(np.arange(0.0, 360.0, args.yaw_step)))

    print("Searching the pinch family (world-z yaw) x wrist placement, at the CONTACT height.")
    print("Scored on the arm's own limits; the pinch geometry is identical for every yaw.\n")
    print(f"{'yaw':>6} {'wrist xy':>16} {'pos mm':>7} {'ori deg':>8} {'s_min':>6} "
          f"{'worst joint':>20} {'roll slack':>10} {'waist':>7}")
    best = None
    for yaw in yaws:
        Rt = yawed(yaw)
        for xy in xys:
            r = arm.solve(np.array([xy[0], xy[1], z_contact]), Rt, seeds=args.seeds)
            ok = r["pos_err"] <= args.pos_tol and np.degrees(r["ori_err"]) <= args.ori_tol
            if not ok:
                continue
            if best is None or r["s_min"] > best[0]["s_min"]:
                best = (r, yaw, xy)
        if best is not None and (args.yaw is not None or yaw % 45 == 0 or best[1] == yaw):
            r, by, bxy = best
            print(f"{yaw:6.0f} {str(tuple(round(v,3) for v in bxy)):>16} {r['pos_err']*1000:7.2f} "
                  f"{np.degrees(r['ori_err']):8.3f} {r['s_min']:6.3f} {r['worst']:>20} "
                  f"{r['wrist_roll_slack']:10.3f} {r['waist']:+7.3f}   <- best so far (yaw {by:.0f})")

    if best is None:
        print("\nNo pose met the tolerances. Widen --waist-max or the xy grid.")
        return 1

    r, yaw, xy = best
    b_xy = bolt_xy(xy, yaw)
    print(f"\n{'='*78}\nCHOSEN PRE-GRASP POSE\n{'='*78}")
    print(f"  pinch yaw about world z   {yaw:.1f} deg   (Experiment 2 used 0)")
    print(f"  wrist xy                  ({xy[0]:.4f}, {xy[1]:.4f})")
    print(f"  wrist z at contact        {z_contact:.4f}   ({WRIST_HEIGHT:.4f} above the table)")
    print(f"  wrist z at episode start  {z_start:.4f}   ({APPROACH*100:.0f} cm above contact)")
    q = yawed(yaw).as_quat()
    print(f"  wrist orientation (x,y,z,w) ({q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}, {q[3]:+.4f})")
    print(f"  bolt xy                   ({b_xy[0]:.4f}, {b_xy[1]:.4f})")
    print(f"  bolt yaw                  {BOLT_YAW_DEG + yaw:.1f} deg")
    print(f"  position error            {r['pos_err']*1000:.2f} mm")
    print(f"  orientation error         {np.degrees(r['ori_err']):.3f} deg")
    print(f"  minimum arm joint slack   {r['s_min']:.3f}  (worst: {r['worst']})")
    print(f"  wristRoll_Right           {r['wrist_roll']:+.4f} rad, slack {r['wrist_roll_slack']:.3f}")
    print(f"  waistPitch (HELD)         {r['waist']:+.4f} rad")

    # ---- 1b. pin the held waist -------------------------------------------
    # The waist is held, not actuated, so ONE lean has to serve the whole
    # episode: the reset pose 5 cm up, and the contact pose the policy descends
    # to. Solving each height with a free waist picks a different lean for each,
    # and then neither pose is reachable in the env. Score every candidate lean
    # by the WORSE of the two poses.
    print("\n" + "=" * 78)
    print("PINNING THE HELD WAIST  (one lean must serve start AND contact)")
    print("=" * 78)
    print(f"{'waist':>7} {'start s_min':>12} {'contact s_min':>14} {'worse':>7} "
          f"{'start err':>11} {'contact err':>13}")
    Rt = yawed(yaw)
    cand = None
    for w in np.linspace(0.3, min(1.2, arm.hi[-1]), args.waist_grid):
        arm.lo[-1], arm.hi[-1] = w - 1e-6, w + 1e-6   # scipy needs lo < hi
        a = arm.solve(np.array([xy[0], xy[1], z_start]), Rt, seeds=args.seeds)
        b = arm.solve(np.array([xy[0], xy[1], z_contact]), Rt, seeds=args.seeds)
        okay = (a["pos_err"] <= args.pos_tol and b["pos_err"] <= args.pos_tol
                and np.degrees(a["ori_err"]) <= args.ori_tol
                and np.degrees(b["ori_err"]) <= args.ori_tol)
        worse = min(a["s_min"], b["s_min"]) if okay else -1.0
        flag = "" if okay else "   <- unreachable"
        print(f"{w:7.3f} {a['s_min']:12.3f} {b['s_min']:14.3f} {worse:7.3f} "
              f"{a['pos_err']*1000:9.2f}mm {b['pos_err']*1000:11.2f}mm{flag}")
        if okay and (cand is None or worse > cand[0]):
            cand = (worse, w, a, b)
    if cand is None:
        print("\nNo single held lean reaches both heights. Try a different wrist xy.")
        return 1
    _, waist, rs, rc = cand
    arm.lo[-1], arm.hi[-1] = waist - 1e-6, waist + 1e-6
    print(f"\n  held waistPitch = {waist:+.4f} rad   "
          f"(start s_min {rs['s_min']:.3f}, contact s_min {rc['s_min']:.3f})")
    print(f"  the 5 cm descent costs {rc['s_min'] - rs['s_min']:+.3f} of minimum joint slack")
    print(f"\nPREGRASP_ARM (reset pose, {APPROACH*100:.0f} cm above contact) -- paste into the env cfg:")
    print("PREGRASP_ARM = {")
    for n, v in zip(arm.vars, rs["q"]):
        tag = "    # HELD, not a policy action" if n == WAIST else ""
        print(f'    "{n}": {v:+.4f},{tag}')
    print("}")
    print(f"  start-pose position error {rs['pos_err']*1000:.2f} mm, orientation "
          f"{np.degrees(rs['ori_err']):.3f} deg, s_min {rs['s_min']:.3f} ({rs['worst']})")
    print("  per-joint slack at the start pose:")
    for n, v, s in zip(ARM, rs["q"][:7], rs["slack"]):
        bar = "#" * int(round(s * 30))
        print(f"    {n:22s} {v:+.4f}  s={s:.3f} {bar}")

    # ---- 2. the reach envelope ---------------------------------------------
    if args.envelope:
        print(f"\n{'='*78}\nREACH-ENVELOPE TEST  (pinch yaw {yaw:.0f} deg, contact height)\n{'='*78}")
        print(f"{'':>8}" + "".join(f"{y:+7.2f}" for y in np.arange(-0.40, 0.01, 0.05)))
        rows = []
        for x in np.arange(0.20, 0.61, 0.05):
            cells, line = [], f"x={x:5.2f} "
            for y in np.arange(-0.40, 0.01, 0.05):
                rr = arm.solve(np.array([x, y, z_contact]), yawed(yaw), seeds=4)
                reached = rr["pos_err"] <= args.pos_tol and np.degrees(rr["ori_err"]) <= args.ori_tol
                cells.append((x, y, rr, reached))
                line += f"{rr['s_min']:7.2f}" if reached else "      ."
            print(line)
            rows += cells
        good = [c for c in rows if c[3]]
        print("\n  '.' = not reachable at this orientation; numbers are minimum arm joint slack.")
        if good:
            xs = sorted({c[0] for c in good}); ys = sorted({c[1] for c in good})
            worst = min(good, key=lambda c: c[2]["s_min"])
            bestc = max(good, key=lambda c: c[2]["s_min"])
            print(f"  reachable x range         {min(xs):.2f} .. {max(xs):.2f} m")
            print(f"  reachable y range         {min(ys):.2f} .. {max(ys):.2f} m")
            print(f"  cells reached             {len(good)} of {len(rows)}")
            print(f"  max IK residual           {max(c[2]['pos_err'] for c in good)*1000:.2f} mm, "
                  f"{max(np.degrees(c[2]['ori_err']) for c in good):.3f} deg")
            print(f"  best  slack               {bestc[2]['s_min']:.3f} at "
                  f"({bestc[0]:.2f},{bestc[1]:+.2f}), worst joint {bestc[2]['worst']}")
            print(f"  worst slack               {worst[2]['s_min']:.3f} at "
                  f"({worst[0]:.2f},{worst[1]:+.2f}), worst joint {worst[2]['worst']}")
            print(f"  wrist-roll margin         {min(c[2]['wrist_roll_slack'] for c in good):.3f} .. "
                  f"{max(c[2]['wrist_roll_slack'] for c in good):.3f}")
            print(f"  waist target range        {min(c[2]['waist'] for c in good):+.3f} .. "
                  f"{max(c[2]['waist'] for c in good):+.3f} rad")
        bad = [(c[0], c[1]) for c in rows if not c[3]]
        if bad:
            print(f"  failure locations         {len(bad)} cells, e.g. "
                  f"{', '.join(f'({x:.2f},{y:+.2f})' for x, y in bad[:6])}"
                  f"{' ...' if len(bad) > 6 else ''}")
        print("\n  Kinematics only. It says nothing about arm SAG under the drives, which")
        print("  Experiment 1 measured at ~4 cm before the effort limits were raised to the")
        print("  URDF's. At 5 cm of approach that is the whole budget, so the start pose is")
        print("  not trusted until tools/test_pipeline_arch.py measures it in the simulator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
