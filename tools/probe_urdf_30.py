#!/usr/bin/env python3
"""Can Isaac Lab 3.0 import the Adam-U URDF usably, on its own?

This decides whether the 6.0 environment can stand alone. adam-wbc found that
3.0's URDF importer nests rigid bodies under the kinematic tree, which broke a
contact-sensor pattern of the form "{ENV_REGEX_NS}/Robot/.*". This env does not
use that pattern -- it addresses three bodies by explicit prim path for its
FrameTransformers -- so the nesting may or may not actually matter here.

If 3.0 can import the URDF directly, the 6.0 image needs nothing from the 5.1
image and the flat-USD export step disappears. If it cannot, the USD has to be
committed (or generated once on 5.1).

Run inside the 6.0 image:

    docker compose run --rm adam_rl30 python /opt/tools/probe_urdf_30.py
"""

from __future__ import annotations

import os
import sys

URDF = "/workspace/assets/robots/adam_u/urdf/adam_u.urdf"

# The three body prims the env's FrameTransformers reference by explicit path.
NEEDED = ["waistRoll", "wristRollRight", "R_thumb_distal", "R_pinky_distal"]


def main() -> int:
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app

    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdPhysics

    out_dir = "/tmp/probe_urdf_30"
    print(f"[probe] importing {URDF} with Isaac Lab 3.0's converter")

    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] RESULT=NO_CONVERTER {type(exc).__name__}: {exc}")
        app.close()
        return 1

    try:
        cfg = UrdfConverterCfg(
            asset_path=URDF,
            usd_dir=out_dir,
            usd_file_name="probe.usd",
            fix_base=True,
            force_usd_conversion=True,
        )
        conv = UrdfConverter(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] RESULT=CONVERT_FAILED {type(exc).__name__}: {exc}")
        app.close()
        return 1

    stage = Usd.Stage.Open(conv.usd_path)
    bodies = [p.GetPath().pathString for p in stage.Traverse() if UsdPhysics.RigidBodyAPI(p)]
    depths = sorted({len(b.strip("/").split("/")) for b in bodies})
    leaves = {b.rsplit("/", 1)[-1]: b for b in bodies}

    print(f"[probe] rigid bodies: {len(bodies)}")
    print(f"[probe] prim path depths: {depths}  ({'FLAT' if len(depths) == 1 else 'NESTED'})")
    print("[probe] sample paths:")
    for b in bodies[:5]:
        print(f"          {b}")

    missing = [n for n in NEEDED if n not in leaves]
    print(f"[probe] bodies the env addresses by name: "
          f"{'ALL PRESENT' if not missing else f'MISSING {missing}'}")
    for n in NEEDED:
        if n in leaves:
            depth = len(leaves[n].strip('/').split('/'))
            print(f"          {n:18} -> {leaves[n]}  (depth {depth})")

    # The env writes "{ENV_REGEX_NS}/Robot/<name>". That only resolves if the
    # body is a DIRECT child of the robot root, i.e. depth 2 in this stage.
    direct = [n for n in NEEDED if n in leaves
              and len(leaves[n].strip("/").split("/")) == 2]
    ok = len(bodies) == 44 and len(direct) == len(NEEDED)
    print(f"[probe] RESULT={'USABLE' if ok else 'NOT_USABLE'}"
          f"  (44 bodies: {len(bodies) == 44}, "
          f"all needed prims at depth 2: {len(direct)}/{len(NEEDED)})")
    app.close()
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
