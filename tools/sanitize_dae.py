#!/usr/bin/env python3
"""Make Collada (.dae) meshes safe to import as USD.

USD prim names accept only [A-Za-z0-9_]. The PNDbotics Adam Lite meshes shipped
in GMR carry ids with hyphens ("pelvis-mesh") and CJK characters
("灰色-material"), plus one hyphenated *node* name ("vn100-0001", the VectorNav
IMU housing). Isaac Sim's URDF importer turns node and material names into prim
names, so it silently creates a null prim and yields an articulation with ZERO
rigid bodies. The failure surfaces much later and misleadingly as:

    ValueError: No contact sensors added to the prim: '.../Robot'.
                This means that no rigid bodies are present under this prim.

This script rewrites every declared id (and every reference to it) to a legal
form, in place. Idempotent: running twice is a no-op.

    python tools/sanitize_dae.py <mesh_dir> [--dry-run]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

# Transliterations for the material names in the Adam Lite meshes. Anything not
# listed falls back to a "u<hex>" escape, which is legal if ugly.
TRANSLITERATE = {
    "灰色": "gray",
    "黑色": "black",
    "白色": "white",
    "红色": "red",
    "蓝色": "blue",
    "绿色": "green",
    "黄色": "yellow",
    "银色": "silver",
    "橙色": "orange",
    "紫色": "purple",
}


def sanitize(name: str) -> str:
    """Return a USD-legal version of a Collada id."""
    for src, dst in TRANSLITERATE.items():
        name = name.replace(src, dst)
    name = "".join(c if c.isascii() else "u%04x" % ord(c) for c in name)
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if name and name[0].isdigit():
        name = "_" + name
    return name


def process(path: pathlib.Path, dry_run: bool = False) -> int:
    text = path.read_text(encoding="utf-8")
    declared = set(re.findall(r'\bid="([^"]+)"', text))
    remap = {old: sanitize(old) for old in declared if sanitize(old) != old}
    if not remap:
        return 0

    # Longest ids first: "pelvis-mesh-positions" must be rewritten before
    # "pelvis-mesh", or the shorter replacement corrupts the longer id.
    for old in sorted(remap, key=len, reverse=True):
        new = remap[old]
        text = text.replace(f'"{old}"', f'"{new}"')      # id= and name= attrs
        text = text.replace(f'"#{old}"', f'"#{new}"')    # source=, url=, target=
        text = text.replace(f">{old}<", f">{new}<")      # <init_from> element text

    if not dry_run:
        path.write_text(text, encoding="utf-8")
    return len(remap)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh_dir", type=pathlib.Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    if not args.mesh_dir.is_dir():
        print(f"error: not a directory: {args.mesh_dir}", file=sys.stderr)
        return 1

    files = sorted(args.mesh_dir.glob("*.dae"))
    if not files:
        print(f"error: no .dae files in {args.mesh_dir}", file=sys.stderr)
        return 1

    touched = total = 0
    for f in files:
        n = process(f, args.dry_run)
        if n:
            touched += 1
            total += n
            print(f"  {f.name}: {n} ids")

    verb = "would sanitize" if args.dry_run else "sanitized"
    print(f"{verb} {total} ids across {touched}/{len(files)} files")

    # A hyphen in the *filename* is fine for USD, but keep it tidy and make the
    # URDF reference match what patch_urdf.py expects.
    for f in files:
        if "-" in f.name:
            target = f.with_name(f.name.replace("-", "_"))
            if args.dry_run:
                print(f"  would rename {f.name} -> {target.name}")
            else:
                f.rename(target)
                print(f"  renamed {f.name} -> {target.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
