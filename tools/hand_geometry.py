#!/usr/bin/env python3
"""Where are the Wuji fingertips, in the wrist frame, for a given finger posture?

Host-side URDF forward kinematics (reuses tools/solve_pregrasp_pose.py). Used
to pick the pre-grasp finger curl for the grasp-only env: the thumb/index
pinch point has to sit a few centimetres below the palm, directly under it,
so a short descent plus a close catches a bolt on the table.

    python3 tools/hand_geometry.py                    # a few candidate postures
    python3 tools/hand_geometry.py --flex 0.5 0.8 0.3 --thumb 0.4 -0.3 0.4 0.4

Frame: wristRollRight. -z is the finger direction at q=0 (down when the hand
hangs), +y is the thumb side.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solve_pregrasp_pose as kin  # noqa: E402

FINGERS = ["index_finger", "middle_finger", "ring_finger", "pinky"]


def posture(flex, thumb, abd=0.0):
    """flex = (mcp_flex, pip, dip) for the four fingers; thumb = (cmc_flex, cmc_abd, mcp, ip)."""
    q = {}
    for f in FINGERS:
        q[f"r_{f}_mcp_flex"], q[f"r_{f}_pip"], q[f"r_{f}_dip"] = flex
        q[f"r_{f}_mcp_abd"] = abd
    q["r_thumb_cmc_flex"], q["r_thumb_cmc_abd"], q["r_thumb_mcp"], q["r_thumb_ip"] = thumb
    return q


def tips_in_wrist(joints, q):
    wrist_path, _ = kin.chain_to(joints, "wristRollRight")
    Tw = kin.fk(wrist_path, q)
    out = {}
    for name, link in [("thumb", "r_thumb_tip"), ("index", "r_index_finger_tip"),
                       ("middle", "r_middle_finger_tip"), ("pinky", "r_pinky_tip")]:
        path, _ = kin.chain_to(joints, link)
        T = kin.fk(path, q)
        out[name] = (np.linalg.inv(Tw) @ T)[:3, 3]
    return out


def report(joints, label, q):
    t = tips_in_wrist(joints, q)
    mid = (t["thumb"] + t["index"]) / 2
    gap = np.linalg.norm(t["thumb"] - t["index"])
    print(f"{label}")
    for k in ("thumb", "index", "middle", "pinky"):
        print(f"    {k:7s} tip  x={t[k][0]:+.3f} y={t[k][1]:+.3f} z={t[k][2]:+.3f}")
    print(f"    thumb-index gap {gap * 100:.1f} cm   pinch midpoint x={mid[0]:+.3f} y={mid[1]:+.3f} z={mid[2]:+.3f}")
    return t, mid, gap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flex", type=float, nargs=3, default=None, metavar=("MCP", "PIP", "DIP"))
    ap.add_argument("--thumb", type=float, nargs=4, default=None, metavar=("CMC_FLEX", "CMC_ABD", "MCP", "IP"))
    ap.add_argument("--abd", type=float, default=0.0)
    args = ap.parse_args()
    joints = kin.parse(kin.URDF)

    if args.flex is not None or args.thumb is not None:
        flex = tuple(args.flex) if args.flex else (0.0, 0.0, 0.0)
        thumb = tuple(args.thumb) if args.thumb else (0.0, 0.0, 0.0, 0.0)
        report(joints, f"posture flex={flex} thumb={thumb} abd={args.abd}", posture(flex, thumb, args.abd))
        return 0

    report(joints, "OPEN  (q = 0)", posture((0, 0, 0), (0, 0, 0, 0)))
    report(joints, "CLOSED (every flexion joint at its upper limit)",
           posture((1.570, 2.094, 1.570), (1.291, 0.0, 1.570, 1.570)))
    print()
    print("candidate pre-grasp curls:")
    for flex, thumb in [
        ((0.5, 0.8, 0.3), (0.4, -0.3, 0.4, 0.3)),
        ((0.7, 1.0, 0.4), (0.5, -0.5, 0.5, 0.4)),
        ((0.9, 1.2, 0.5), (0.6, -0.6, 0.6, 0.5)),
        ((0.8, 1.4, 0.6), (0.7, -0.8, 0.7, 0.6)),
    ]:
        report(joints, f"  flex={flex} thumb={thumb}", posture(flex, thumb))
    print()
    print("thumb vs index tip while closing together (fraction of the way to the limits):")
    for s in (0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        q = posture((1.570 * s, 2.094 * s, 1.570 * s), (1.291 * s, -0.4, 1.570 * s, 1.570 * s))
        t = tips_in_wrist(joints, q)
        gap = np.linalg.norm(t["thumb"] - t["index"])
        print(f"    s={s:.2f}  gap {gap * 100:5.1f} cm   thumb z {t['thumb'][2]:+.3f}  index z {t['index'][2]:+.3f}  "
              f"thumb xy ({t['thumb'][0]:+.3f},{t['thumb'][1]:+.3f})  index xy ({t['index'][0]:+.3f},{t['index'][1]:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
