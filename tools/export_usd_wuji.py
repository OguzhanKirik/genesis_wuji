#!/usr/bin/env python3
"""Convert the Adam-U + Wuji URDF to a FLAT USD for the Isaac Lab 3.0 env.

Same job as tools/export_usd.py, for the Wuji-hand body. It must run under
Isaac Lab **2.3.2** (Isaac Sim 5.1): 3.0's URDF importer nests rigid bodies
under the kinematic tree and the env's explicit FrameTransformer prim paths
({ENV_REGEX_NS}/Robot/<body>) stop resolving. 2.3.2's converter emits all
bodies as direct children of the default prim. This repo no longer builds a
2.3.2 image, but adam-wbc's `adam-lite-wbc:isaaclab-2.3.2` is one:

    docker run --rm --gpus all \\
        -v $PWD/adam_u_isaac_lab:/workspace -v $PWD/tools:/opt/tools:ro \\
        -v adam-wbc_isaac-cache-kit:/opt/venv/lib/python3.11/site-packages/isaacsim/kit/cache \\
        -v adam-wbc_isaac-cache-ov:/root/.cache/ov -v adam-wbc_isaac-cache-nvidia:/root/.nv \\
        --entrypoint python adam-lite-wbc:isaaclab-2.3.2 /opt/tools/export_usd_wuji.py

Expected result: 60 rigid bodies at one prim depth. The URDF has 84 links and
24 fixed joints (two frozen column joints, two wrist-to-palm joints, ten
fingertip frames, ten tip sensor frames); merge_fixed_joints folds each fixed
child into its parent, so the palm mesh lands on wristRoll{Left,Right} and the
tips on the distal links. 59 revolute joints remain.

`--entrypoint python` skips that image's entrypoint, which otherwise links
adam-lite model assets into /workspace/adam_lite_lab/ (a stray directory here).

Idempotent -- pass --force to reconvert.

The hand-only robot for Experiment 2 uses the same tool:

    ... /opt/tools/export_usd_wuji.py --urdf /workspace/assets/robots/wuji_hand_right/urdf/wuji_hand_right_f3.urdf \
        --out-dir /workspace/assets/robots/wuji_hand_right/usd_f3 --name wuji_hand_right_f3 \
        --expect-bodies 22 --expect-joints 21 --needed r_wrist r_thumb_distal r_index_finger_distal r_middle_finger_distal

  One asset per finger count (f2, f3): each solved pose bakes its own prismatic
  lift axis into the URDF, so the modes cannot share a file. Generate the URDF
  with tools/make_wuji_hand_urdf.py --fingers N first.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = "/workspace"
DEFAULT_URDF = os.path.join(REPO_ROOT, "assets", "robots", "adam_u_wuji", "urdf", "adam_u_wuji.urdf")
DEFAULT_OUT = os.path.join(REPO_ROOT, "assets", "robots", "adam_u_wuji", "usd")
EXPECT_BODIES = 60
EXPECT_JOINTS = 59
# Bodies the env addresses by explicit prim path.
NEEDED = ["waistRoll", "wristRollRight", "r_thumb_distal", "r_index_finger_distal",
          "r_middle_finger_distal", "r_pinky_distal"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--name", default="adam_u_wuji", help="USD file stem")
    ap.add_argument("--expect-bodies", type=int, default=EXPECT_BODIES)
    ap.add_argument("--expect-joints", type=int, default=EXPECT_JOINTS)
    ap.add_argument("--needed", nargs="*", default=NEEDED, help="body names the env addresses by prim path")
    args = ap.parse_args()

    if not os.path.isfile(args.urdf):
        print(f"error: URDF not found: {args.urdf}", file=sys.stderr)
        return 1

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    cfg = UrdfConverterCfg(
        asset_path=args.urdf,
        usd_dir=args.out_dir,
        usd_file_name=f"{args.name}.usd",
        fix_base=True,
        force_usd_conversion=args.force,
        merge_fixed_joints=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
        ),
    )
    print(f"[export_usd_wuji] converting {args.urdf}")
    converter = UrdfConverter(cfg)
    print(f"[export_usd_wuji] wrote {converter.usd_path}")

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(converter.usd_path)
    bodies = [p.GetPath().pathString for p in stage.Traverse() if UsdPhysics.RigidBodyAPI(p)]
    joints = [p.GetPath().pathString for p in stage.Traverse()
              if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)]
    depths = {len(b.strip("/").split("/")) for b in bodies}
    names = {b.rsplit("/", 1)[-1] for b in bodies}
    print(f"[export_usd_wuji] {len(bodies)} rigid bodies, {len(joints)} joints, prim depths {sorted(depths)}")
    ok = True
    if len(depths) != 1:
        print("[export_usd_wuji] WARNING: hierarchy is NOT flat", file=sys.stderr)
        ok = False
    if len(bodies) != args.expect_bodies:
        print(f"[export_usd_wuji] WARNING: expected {args.expect_bodies} bodies, got {len(bodies)}", file=sys.stderr)
        ok = False
    if len(joints) != args.expect_joints:
        print(f"[export_usd_wuji] WARNING: expected {args.expect_joints} joints, got {len(joints)}", file=sys.stderr)
        ok = False
    missing = [n for n in args.needed if n not in names]
    if missing:
        print(f"[export_usd_wuji] WARNING: bodies the env addresses are missing: {missing}", file=sys.stderr)
        ok = False
    print("[export_usd_wuji] bodies:", " ".join(sorted(names)))
    print(f"[export_usd_wuji] {'OK' if ok else 'FAILED'}")
    app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
