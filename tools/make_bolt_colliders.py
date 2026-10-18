#!/usr/bin/env python3
"""Write the rod asset's collider variants next to it (run inside the container: needs pxr).

    docker compose exec adam_rl python /opt/tools/make_bolt_colliders.py

The asset as shipped carries `physics:approximation = convexHull`; PhysX cooks
the 327k-face mesh to a 64-vertex hull whose end faces are not flat, so a rod
stood on its end tilts 4.5 deg at once and a third topple within 3 s untouched.
`_cd`: a 32-hull convex decomposition (V-HACD), stands with 0.03 deg of tilt.
"""
import os
from pxr import Sdf, Usd, UsdGeom

SRC = os.environ.get("BOLT_SRC", "/workspace/assets/workcell/rod-with-threads-0001/rod-with-threads-0001.usdc")
VARIANTS = {
    "cd": ("convexDecomposition", {
        "physxConvexDecompositionCollision:maxConvexHulls": (Sdf.ValueTypeNames.Int, 32),
        "physxConvexDecompositionCollision:hullVertexLimit": (Sdf.ValueTypeNames.Int, 64),
        "physxConvexDecompositionCollision:voxelResolution": (Sdf.ValueTypeNames.Int, 500000),
        "physxConvexDecompositionCollision:errorPercentage": (Sdf.ValueTypeNames.Float, 0.5),
    }),
}
for tag, (approx, extra) in VARIANTS.items():
    stage = Usd.Stage.Open(SRC)
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            prim.GetAttribute("physics:approximation").Set(approx)
            for name, (typ, val) in extra.items():
                prim.CreateAttribute(name, typ).Set(val)
    root, ext = os.path.splitext(SRC)
    out = f"{root}_{tag}{ext}"
    stage.GetRootLayer().Export(out)
    print("wrote", out)
