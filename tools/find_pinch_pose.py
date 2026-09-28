#!/usr/bin/env python3
"""Find a thumb-and-index (or thumb vs index+middle) PINCH pre-grasp for the Wuji hand.

Host-side URDF forward kinematics, no Isaac. Answers the question the spec
insists on -- "do not simply place the palm above the bolt without checking the
fingertip geometry" -- and it answers it for the hand's ORIENTATION too, which
is the variable that decides whether a pinch is possible at all.

Method. For a candidate (thumb abduction, open curl, closed curl) walk the
synergy stroke q = q_open + s*(q_closed - q_open), s in [0, 1], and track every
fingertip in the WRIST frame. Then:

  * gap, c*      the thumb-vs-opposing gap at the start and end of the stroke,
                 and where in it the gap first crosses the bolt's 2.5 cm width.
                 THAT crossing is where the bolt goes: it is where the fingers
                 first close onto it. (Placing the bolt at the tips' closest
                 approach instead puts it where they end up after squeezing
                 through it, which is a different and wrong pose.) A usable
                 stroke opens wider than the bolt and closes past it, and the
                 remainder is the squeeze.
  * opposition   angle between the thumb's and the opposing finger's travel to
                 that point. ~180 deg is a true opposing pinch; near 0 means
                 both sweep the same way and the bolt is pushed, not held.
  * start_dist   open fingertip to the bolt. The spec asks for 1-2 cm to its
                 surface, so 2.3-3.3 cm to its centre.

All three are independent of how the hand is rotated. Only table clearance is
not, so the orientation is solved for separately: with the wrist placed so the
pinch point lands at the bolt's centre height, a tip at wrist-frame p has world
height (p - P) . u + bolt_r, where u is world-up expressed in the wrist frame.
A pose is reachable iff some unit u keeps every tip, over the whole stroke,
above the table:

        max over u of  min over tips p of  (p - P) . u   >=  clearance - bolt_r

which this scans over a sphere of u. The winning u is turned back into the
hand's quaternion, with the spin about vertical chosen so the pinch closes
across the bolt's axis.

    python3 tools/find_pinch_pose.py                      # rank candidates
    python3 tools/find_pinch_pose.py --middle             # thumb vs index+middle
    python3 tools/find_pinch_pose.py --show 0.40 0.60 -0.85   # the committed 3-finger pose

--show prints the POSE entry to paste into the env config, and the command to
rebuild the hand asset with the matching lift axis.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solve_pregrasp_pose as kin  # noqa: E402

LIM = {"thumb_cmc_flex": 1.291, "thumb_mcp": 1.570, "thumb_ip": 1.570,
       "mcp_flex": 1.570, "pip": 2.094, "dip": 1.570}
BOLT_R = 0.0125            # bolt half-width = the height of its centre at rest
BOLT_D = 0.025             # bolt width: the tips CONTACT it when their gap crosses this
CLEARANCE = 0.004          # m, how far every fingertip must stay above the table
FINGERS = ("index_finger", "middle_finger", "ring_finger", "pinky")
TIP_NAMES = ("thumb",) + FINGERS


def posture(thumb_c: float, finger_c: float, thumb_abd: float, held=()):
    q = {"r_thumb_cmc_abd": thumb_abd,
         "r_thumb_cmc_flex": LIM["thumb_cmc_flex"] * thumb_c,
         "r_thumb_mcp": LIM["thumb_mcp"] * thumb_c,
         "r_thumb_ip": LIM["thumb_ip"] * thumb_c}
    for f in FINGERS:
        c = held.get(f, finger_c) if isinstance(held, dict) else (0.0 if f in held else finger_c)
        q[f"r_{f}_mcp_flex"] = LIM["mcp_flex"] * c
        q[f"r_{f}_pip"] = LIM["pip"] * c
        q[f"r_{f}_dip"] = LIM["dip"] * c
        q[f"r_{f}_mcp_abd"] = 0.0
    return q


def tips(joints, q, names=TIP_NAMES):
    wrist_path, _ = kin.chain_to(joints, "wristRollRight")
    Tw = np.linalg.inv(kin.fk(wrist_path, q))
    out = {}
    for n in names:
        link = "r_thumb_tip" if n == "thumb" else f"r_{n}_tip"
        path, _ = kin.chain_to(joints, link)
        out[n] = (Tw @ kin.fk(path, q))[:3, 3]
    return out


def _sphere(n=1200):
    """Roughly uniform unit vectors (Fibonacci sphere)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


_U = _sphere()


def evaluate(joints, open_c, closed_c, thumb_abd, held=(), steps=21, use_middle=False):
    track = [tips(joints, posture(c, c, thumb_abd, held))
             for c in np.linspace(open_c, closed_c, steps)]
    thumb = np.array([t["thumb"] for t in track])
    index = np.array([t["index_finger"] for t in track])
    middle = np.array([t["middle_finger"] for t in track])
    opp = (index + middle) / 2 if use_middle else index

    gaps = np.linalg.norm(thumb - opp, axis=1)
    # The bolt goes where the tips FIRST CLOSE ONTO IT -- where their gap
    # crosses the bolt's width -- not where they would eventually meet. Placing
    # it at the closest approach instead puts the bolt where the fingers end up
    # after squeezing through it, which is a different (and wrong) pose.
    closing = np.nonzero(gaps <= BOLT_D)[0]
    usable = gaps[0] > BOLT_D and len(closing) > 0
    k = int(closing[0]) if usable else int(gaps.argmin())
    P = (thumb[k] + opp[k]) / 2

    d_th, d_op = thumb[k] - thumb[0], opp[k] - opp[0]
    if min(np.linalg.norm(d_th), np.linalg.norm(d_op)) > 1e-6:
        cos = float(np.dot(d_th, d_op) / (np.linalg.norm(d_th) * np.linalg.norm(d_op)))
        approach = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    else:
        approach = float("nan")

    # Every tip over the whole stroke, relative to the pinch point.
    pts = np.array([[t[n] for n in TIP_NAMES] for t in track]).reshape(-1, 3) - P
    proj = pts @ _U.T                        # (points, directions)
    worst = proj.min(axis=0)                 # lowest tip for each candidate up-vector
    b = int(worst.argmax())
    u, clear = _U[b], float(worst[b])        # best up-vector and its margin
    ok = clear >= CLEARANCE - BOLT_R

    open_tips = track[0]
    return dict(gap=float(gaps[k]), gap_open=float(gaps[0]), gap_closed=float(gaps[-1]),
                squeeze=float(BOLT_D - gaps[-1]), usable=bool(usable),
                c_star=float(k / (steps - 1)), P=P, approach=approach,
                u=u, clear=clear, reachable=ok and usable, wrist_h=float(-P @ u + BOLT_R),
                d_thumb0=float(np.linalg.norm(open_tips["thumb"] - P)),
                d_index0=float(np.linalg.norm(open_tips["index_finger"] - P)),
                thumb_track=thumb, open_tips=open_tips, track=track)


def quat_from_up(u, spin_deg=0.0):
    """A hand rotation whose wrist-frame vector u points to world +z.

    The spin about vertical is free; spin_deg rotates the hand about world z.
    Returns (x, y, z, w), Isaac Lab 3.0 order.
    """
    from scipy.spatial.transform import Rotation as R
    u = u / np.linalg.norm(u)
    # any rotation taking u -> +z
    v = np.array([0.0, 0.0, 1.0])
    axis = np.cross(u, v)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        Rm = np.eye(3) if u[2] > 0 else R.from_euler("x", 180, degrees=True).as_matrix()
    else:
        ang = np.arctan2(s, float(np.dot(u, v)))
        Rm = R.from_rotvec(axis / s * ang).as_matrix()
    Rm = R.from_euler("z", spin_deg, degrees=True).as_matrix() @ Rm
    return R.from_matrix(Rm).as_quat()      # x, y, z, w


def fmt(r, label=""):
    why = ""
    if not r["usable"]:
        why = "   [already touching at open]" if r["gap_open"] <= BOLT_D else "   [never closes onto the bolt]"
    elif not r["reachable"]:
        why = "   [tips would hit the table]"
    return (f"{label}gap {r['gap_open'] * 100:4.1f}->{r['gap_closed'] * 100:4.1f} cm "
            f"(contact at {r['c_star'] * 100:3.0f}% of the stroke, squeeze {r['squeeze'] * 1000:3.0f} mm)  "
            f"opp {r['approach']:5.1f} deg  wrist_h {r['wrist_h'] * 100:5.1f} cm  "
            f"clear {(r['clear'] + BOLT_R) * 100:+5.1f} cm  "
            f"start thumb/index {r['d_thumb0'] * 100:4.1f}/{r['d_index0'] * 100:4.1f} cm{why}")


def _active(args):
    """Fingers opposing the thumb, for labelling the POSE entry."""
    return ["index", "middle"] if args.middle else ["index"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", type=float, nargs=3, metavar=("OPEN", "CLOSED", "THUMB_ABD"))
    ap.add_argument("--middle", action="store_true", help="oppose the thumb with index+middle (three-finger pinch)")
    ap.add_argument("--hold", type=float, default=0.0,
                    help="curl to hold the ring and little fingers at (0 = flat open)")
    ap.add_argument("--max-start", type=float, default=0.035, help="max open fingertip-to-bolt distance (m)")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    joints = kin.parse(kin.URDF)
    held = {"ring_finger": args.hold, "pinky": args.hold}

    if args.show:
        o, c, a = args.show
        r = evaluate(joints, o, c, a, held, steps=41, use_middle=args.middle)
        print(fmt(r))
        from scipy.spatial.transform import Rotation as R
        q = quat_from_up(r["u"])
        Rm = R.from_quat(q).as_matrix()
        off = Rm @ r["P"]
        k = int(round(r["c_star"] * 40))
        d = Rm @ (r["thumb_track"][k] - r["thumb_track"][0]); d[2] = 0.0
        yaw = float(np.degrees(np.arctan2(d[1], d[0]))) if np.linalg.norm(d) > 1e-9 else 0.0
        print(f"  pinch point (wrist frame) {r['P'].round(4)}, world-up in that frame {r['u'].round(4)}")
        print("\n  paste into POSE in genesis_rl/envs/wuji_synergy_grasp_env_cfg.py:\n")
        print(f"    {len(_active(args)) + 1}: dict(")
        print(f"        thumb_abd={a:.2f}, open_curl={o:.2f}, closed_curl={c:.2f},")
        print(f"        lift_axis=({r['u'][0]:.6f}, {r['u'][1]:.6f}, {r['u'][2]:.6f}),")
        print(f"        hand_rot=({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}),")
        print(f"        wrist_height={r['wrist_h']:.4f}, bolt_offset=({off[0]:.4f}, {off[1]:.4f}), "
              f"bolt_yaw_deg={yaw:.1f},")
        print("    ),")
        print("\n  then rebuild the hand asset:")
        print(f"    python3 tools/make_wuji_hand_urdf.py --fingers {len(_active(args)) + 1} "
              f"--lift-axis {r['u'][0]:.6f} {r['u'][1]:.6f} {r['u'][2]:.6f}")
        print()
        for s in (0.0, 0.25, 0.5, 0.75, 1.0):
            tc = o + s * (c - o)
            t = tips(joints, posture(tc, tc, a, held))
            h = {n: (t[n] - r["P"]) @ r["u"] + BOLT_R for n in TIP_NAMES}
            print(f"  s={s:.2f} c={tc:.2f}  gap {np.linalg.norm(t['thumb'] - t['index_finger']) * 100:5.1f} cm   "
                  f"heights above table [cm] " + " ".join(f"{n[:5]}={h[n] * 100:+5.1f}" for n in TIP_NAMES))
        return 0

    rows = []
    for abd, o, c in itertools.product(
            np.arange(-1.45, 0.66, 0.1), np.arange(0.0, 0.71, 0.05), np.arange(0.2, 1.01, 0.05)):
        if c <= o + 0.15:
            continue
        r = evaluate(joints, float(o), float(c), float(abd), held, steps=17, use_middle=args.middle)
        if (r["reachable"] and r["squeeze"] >= 0.006 and r["approach"] > 60
                and 0.033 <= r["gap_open"] <= 0.055
                and r["d_index0"] < args.max_start and r["d_thumb0"] < args.max_start
                and r["d_index0"] > 0.008 and 0.02 < r["wrist_h"] < 0.20):
            rows.append((abd, o, c, r))
    # Prefer a firm squeeze, well opposed, contacting around mid-stroke.
    rows.sort(key=lambda x: -(x[3]["squeeze"] * 3 + x[3]["approach"] / 180 * 0.02
                              - abs(x[3]["c_star"] - 0.45) * 0.02))
    print(f"{len(rows)} reachable pinches "
          f"({'thumb vs index+middle' if args.middle else 'thumb vs index'}, ring/pinky held at {args.hold})\n")
    for abd, o, c, r in rows[:args.top]:
        print(fmt(r, f"abd={abd:+.2f} open={o:.2f} closed={c:.2f}  "))
    if not rows:
        print("none reachable. Try --middle, a different --hold, or a larger --max-start.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
