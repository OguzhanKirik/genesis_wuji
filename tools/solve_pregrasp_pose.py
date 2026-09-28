#!/usr/bin/env python3
"""Solve the right-arm joint angles that put the Wuji hand above the bolt.

Host-side, numpy + scipy, no Isaac. Reads the vendored URDF, builds the chain
lifting_Columns_base -> ... -> wristRollRight (waist at 0) and solves a small
least-squares IK for a wrist pose whose -z axis (the finger direction at q=0)
points straight down over a target point, staying inside joint limits and
close to adam-moveit's `ready` posture.

    python3 tools/solve_pregrasp_pose.py                          # palm down over the bolt
    python3 tools/solve_pregrasp_pose.py --orient fingers_down --frame index --target 0.39 -0.215 0.99

Prints the joint_pos dict to paste into the grasp-only env cfg, plus the FK
check of where the wrist and the thumb/index fingertips land. The env spawns
the robot root at z = 1.05 - 0.99975; that is folded in here so the numbers
are world coordinates.
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "..", "assets", "robots", "adam_u_wuji", "urdf", "adam_u_wuji.urdf")
ROOT_Z = 1.05 - 0.99975

ARM = ["shoulderPitch_Right", "shoulderRoll_Right", "shoulderYaw_Right",
       "elbow_Right", "wristYaw_Right", "wristPitch_Right", "wristRoll_Right"]
WAIST = "waistPitch"  # optional 8th variable: lean the torso forward (+) to reach a low table
# adam-moveit's right_arm `ready` state: a bent, forward pose the solver is
# pulled toward so it does not pick a contorted branch.
READY = np.array([0.0, 0.06, 0.07, -1.45, 0.0, 0.0, 0.0])

# Fingertip positions in the wristRollRight frame at q=0 (hand open, flat):
# composed from the URDF's r_wrist -> mcp -> ... -> tip chain. Used only for
# the report, so the solver's target can be chosen by where the TIPS land.
THUMB_TIP_OFFSET = None  # filled by FK below
INDEX_TIP_OFFSET = None


def parse(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joints = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        rpy = np.array([float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()])
        ax = j.find("axis")
        axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3)
        lim = j.find("limit")
        lo = float(lim.get("lower")) if lim is not None and lim.get("lower") else -np.inf
        hi = float(lim.get("upper")) if lim is not None and lim.get("upper") else np.inf
        joints[j.get("name")] = dict(
            type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            xyz=xyz, rpy=rpy, axis=axis, lo=lo, hi=hi,
        )
    return joints


def chain_to(joints, link):
    by_child = {j["child"]: (n, j) for n, j in joints.items()}
    path = []
    while link in by_child:
        n, j = by_child[link]
        path.append((n, j))
        link = j["parent"]
    return list(reversed(path)), link


def fk(path, q: dict):
    T = np.eye(4)
    T[2, 3] = ROOT_Z
    for n, j in path:
        A = np.eye(4)
        A[:3, :3] = R.from_euler("xyz", j["rpy"]).as_matrix()
        A[:3, 3] = j["xyz"]
        T = T @ A
        if j["type"] == "revolute":
            ang = q.get(n, 0.0)
            Rj = np.eye(4)
            Rj[:3, :3] = R.from_rotvec(j["axis"] / np.linalg.norm(j["axis"]) * ang).as_matrix()
            T = T @ Rj
    return T


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=URDF)
    ap.add_argument("--target", type=float, nargs=3, default=[0.315, -0.215, 1.035],
                    help="world xyz for the frame named by --frame")
    ap.add_argument("--frame", choices=["wrist", "index", "thumb"], default="wrist",
                    help="which frame --target positions: the wrist origin or a fingertip at q=0")
    ap.add_argument("--orient", choices=["fingers_down", "palm_down", "quat"], default="palm_down",
                    help="fingers_down: wrist -z (finger direction) points at the floor. "
                         "palm_down: wrist +y (the side the fingers curl toward) points at the floor "
                         "and the fingers are horizontal, yaw free -- a top-down pinch. "
                         "quat: match --quat exactly; this is what the staged pipeline needs, because "
                         "its pre-grasp orientation is the tilted one tools/find_pinch_pose.py solved "
                         "and neither canned orientation can express it")
    ap.add_argument("--quat", type=float, nargs=4, metavar=("X", "Y", "Z", "W"), default=None,
                    help="--orient quat: the target wristRollRight orientation (x,y,z,w). r_wrist is "
                         "attached to wristRollRight by a fixed joint with an identity origin, so the "
                         "HAND_ROT the synergy env spawns the cut-out hand with IS this quaternion")
    ap.add_argument("--reg", type=float, default=0.005, help="pull toward the READY posture")
    ap.add_argument("--finger-yaw", type=float, default=None,
                    help="palm_down only: force the finger direction to this yaw (deg from +x); default free")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--waist", type=float, default=None, metavar="MAX_RAD",
                    help="also solve waistPitch in [0, MAX_RAD] (forward lean). The shoulder sits at "
                         "z~1.52 and the 0.95 table is at the straight arm's reach; a lean fixes that")
    ap.add_argument("--ori-weight", type=float, default=0.3,
                    help="weight of the orientation residual against position (metres)")
    ap.add_argument("--margin", type=float, default=0.10,
                    help="keep every joint this far (rad) inside its URDF limits, so the reset "
                         "noise and the drive's sag do not pin a joint on its stop")
    args = ap.parse_args()

    joints = parse(args.urdf)
    wrist_path, base = chain_to(joints, "wristRollRight")
    thumb_path, _ = chain_to(joints, "r_thumb_tip")
    index_path, _ = chain_to(joints, "r_index_finger_tip")
    assert base == "lifting_Columns_base", base
    names = [n for n, j in wrist_path if j["type"] == "revolute"]
    assert names[3:] == ARM, names
    lo = np.array([joints[n]["lo"] for n in ARM]) + args.margin
    hi = np.array([joints[n]["hi"] for n in ARM]) - args.margin
    target = np.array(args.target)
    if args.orient == "quat":
        if args.quat is None:
            ap.error("--orient quat needs --quat X Y Z W")
        R_target = R.from_quat(np.array(args.quat))

    use_waist = args.waist is not None
    var_names = ARM + ([WAIST] if use_waist else [])
    if use_waist:
        lo = np.append(lo, 0.0)
        hi = np.append(hi, args.waist)
    ready = np.append(READY, 0.0) if use_waist else READY

    def qdict(x):
        return {n: v for n, v in zip(var_names, x)}

    frame_path = {"wrist": wrist_path, "index": index_path, "thumb": thumb_path}[args.frame]

    def resid(x):
        q = qdict(x)
        T = fk(wrist_path, q)
        pos_err = fk(frame_path, q)[:3, 3] - target
        if args.orient == "quat":
            # Full orientation error as a rotation vector, in radians, weighted
            # into the same least-squares as the position error in metres.
            ori_err = (R_target.inv() * R.from_matrix(T[:3, :3])).as_rotvec() * args.ori_weight
        elif args.orient == "fingers_down":
            # wrist -z must be world -z: its x,y components vanish
            down = -T[:3, 2]
            ori_err = np.array([down[0], down[1]]) * 0.3
        else:
            palm = T[:3, 1]           # wrist +y -> world -z
            fingers = -T[:3, 2]       # wrist -z horizontal; its yaw is free (the bolt's yaw is random)
            if args.finger_yaw is None:
                fdir_err = [fingers[2] * args.ori_weight]
            else:
                yaw = np.radians(args.finger_yaw)
                fdir_err = (fingers - np.array([np.cos(yaw), np.sin(yaw), 0.0])) * args.ori_weight
            ori_err = np.concatenate([(palm - np.array([0, 0, -1.0])) * args.ori_weight, fdir_err])
        reg = args.reg * (x - ready)
        if use_waist:
            reg[-1] = 0.02 * x[-1]  # prefer the smallest lean that works
        return np.concatenate([pos_err, ori_err, reg])

    best = None
    for seed in range(args.seeds):
        x0 = np.clip(ready + np.random.default_rng(seed).normal(0, 0.8, len(ready)), lo + 0.05, hi - 0.05)
        sol = least_squares(resid, x0, bounds=(lo, hi))
        if best is None or sol.cost < best.cost:
            best = sol
    x = best.x
    T = fk(wrist_path, qdict(x))
    Tt = fk(thumb_path, qdict(x))
    Ti = fk(index_path, qdict(x))
    down = -T[:3, 2]
    palm = T[:3, 1]
    print("solution (rad):")
    for n, v in zip(var_names, x):
        print(f'    "{n}": {v:+.4f},')
    Tf = fk(frame_path, qdict(x))
    print(f"\n{args.frame} frame @ q=0    {Tf[:3, 3].round(4)}   target {target}   |err| {np.linalg.norm(Tf[:3, 3] - target) * 1000:.1f} mm")
    print(f"wrist origin        {T[:3, 3].round(4)}")
    print(f"finger dir (wrist -z) {down.round(3)}   palm normal (wrist +y) {palm.round(3)}")
    if args.orient == "quat":
        err = (R_target.inv() * R.from_matrix(T[:3, :3])).as_rotvec()
        print(f"orientation error   {np.degrees(np.linalg.norm(err)):.2f} deg   "
              f"achieved quat (x,y,z,w) {R.from_matrix(T[:3, :3]).as_quat().round(4)}")
    if args.orient == "palm_down":
        yaw = np.degrees(np.arctan2(down[1], down[0]))
        print(f"palm tilt from vertical {np.degrees(np.arccos(np.clip(-palm[2], -1, 1))):.1f} deg, "
              f"fingers pitched {np.degrees(np.arcsin(np.clip(-down[2], -1, 1))):.1f} deg below horizontal, "
              f"pointing {yaw:+.0f} deg from +x")
    print(f"thumb tip (q=0)     {Tt[:3, 3].round(4)}")
    print(f"index tip (q=0)     {Ti[:3, 3].round(4)}")
    print(f"palm x-axis         {T[:3, 0].round(3)}  palm y-axis {T[:3, 1].round(3)}")
    print(f"joint margin to URDF limits (rad): min {np.min(np.minimum(x - lo, hi - x)) + args.margin:.3f}")
    print(f"residual cost {best.cost:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
