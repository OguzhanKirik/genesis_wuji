#!/usr/bin/env python3
"""Convert the Adam-U URDF to a FLAT USD, for the Isaac Lab 3.0 viewer.

Run this inside the **2.3.2 training image**, not the 6.0 one:

    docker compose run --rm adam_rl python /opt/tools/export_usd.py

Why it exists
-------------
Isaac Lab 3.0's URDF importer nests rigid bodies underneath the kinematic
tree. `adam_u_grasp_env_cfg.py` addresses bodies by explicit prim path --
`{ENV_REGEX_NS}/Robot/waistRoll`, `.../R_thumb_distal`, `.../R_pinky_distal`
for the three FrameTransformers -- and those paths no longer resolve against a
nested hierarchy, so the frame sensors (and therefore every distance reward)
break.

Isaac Lab 2.3.2's converter emits a **flat** hierarchy: all 44 rigid bodies as
direct children of the default prim. Both versions read that happily. So the
5.1 image produces the USD and the 6.0 image consumes it.

The output is gitignored deliberately: at ~109 MB it is fully reproducible
from the URDF plus the sanitized meshes, so committing it would bloat every
clone for no information gain. adam-wbc checked its (39 MB) equivalent in;
this model is nearly three times the size.

Idempotent -- Isaac Lab hashes the inputs and skips the conversion if the
existing output is current. Pass --force to reconvert anyway.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = "/workspace"
DEFAULT_URDF = os.path.join(REPO_ROOT, "assets", "robots", "adam_u", "urdf", "adam_u.urdf")
DEFAULT_OUT = os.path.join(REPO_ROOT, "assets", "robots", "adam_u", "usd")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="reconvert even if up to date")
    args = ap.parse_args()

    if not os.path.isfile(args.urdf):
        print(f"error: URDF not found: {args.urdf}", file=sys.stderr)
        return 1

    # Isaac Sim must be running before the converters import.
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    # These must match adam_u_grasp_env_cfg.py's spawn config, or the 6.0 env
    # gets a robot with different joint drives and collision properties than
    # the one the policy was trained against.
    cfg = UrdfConverterCfg(
        asset_path=args.urdf,
        usd_dir=args.out_dir,
        usd_file_name="adam_u.usd",
        fix_base=True,
        force_usd_conversion=args.force,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
        ),
    )

    print(f"[export_usd] converting {args.urdf}")
    print(f"[export_usd]         -> {args.out_dir}/adam_u.usd")
    converter = UrdfConverter(cfg)
    print(f"[export_usd] wrote {converter.usd_path}")

    # Verify the hierarchy really is flat, since that is the entire point.
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(converter.usd_path)
    bodies = [p.GetPath().pathString for p in stage.Traverse()
              if UsdPhysics.RigidBodyAPI(p)]
    depths = {len(b.strip("/").split("/")) for b in bodies}
    print(f"[export_usd] {len(bodies)} rigid bodies, prim path depths {sorted(depths)}")
    if len(depths) != 1:
        print("[export_usd] WARNING: hierarchy is NOT flat; the 3.0 env's "
              "FrameTransformer prim paths will not resolve.", file=sys.stderr)
        app.close()
        return 1
    if len(bodies) != 44:
        print(f"[export_usd] WARNING: expected 44 rigid bodies, got {len(bodies)}. "
              "Are the Collada meshes sanitized? See tools/sanitize_dae.py.",
              file=sys.stderr)
        app.close()
        return 1

    print("[export_usd] OK: flat hierarchy, 44 rigid bodies")
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
