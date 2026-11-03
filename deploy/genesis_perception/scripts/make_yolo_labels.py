#!/usr/bin/env python3
"""tools/render_rod_dataset.py output -> an ultralytics YOLO-seg dataset (host, needs opencv-python).

    python deploy/genesis_perception/scripts/make_yolo_labels.py datasets/rods_L52 datasets/rods_L52_yolo --val 0.1

Writes images/{train,val}/*.png, labels/{train,val}/*.txt (one line per rod: `0 x1 y1 x2 y2 ...` normalised
polygon) and data.yaml. Rods under `--min-pixels` are dropped (a sliver between two others is not a target).
Train, in the segmenter's own venv (one env per model):

    yolo segment train data=datasets/rods_L52_yolo/data.yaml model=yolo11n-seg.pt imgsz=640 epochs=60
"""
import argparse
import os
import random
import shutil

import cv2
import numpy as np


def polygons(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2) for c in cs if len(c) >= 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--min-pixels", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(a.src, "inst")) if f.endswith(".png"))
    random.Random(a.seed).shuffle(names)
    n_val = int(round(a.val * len(names)))
    split = {n: ("val" if i < n_val else "train") for i, n in enumerate(names)}
    for s in ("train", "val"):
        os.makedirs(os.path.join(a.dst, "images", s), exist_ok=True)
        os.makedirs(os.path.join(a.dst, "labels", s), exist_ok=True)
    kept = 0
    for n in names:
        inst = cv2.imread(os.path.join(a.src, "inst", n + ".png"), cv2.IMREAD_UNCHANGED)
        h, w = inst.shape
        lines = []
        for k in range(1, int(inst.max()) + 1):
            m = inst == k
            if m.sum() < a.min_pixels:
                continue
            for poly in polygons(m):
                xy = (poly / np.array([w, h])).clip(0, 1).reshape(-1)
                lines.append("0 " + " ".join(f"{v:.5f}" for v in xy))
        s = split[n]
        shutil.copyfile(os.path.join(a.src, "rgb", n + ".png"), os.path.join(a.dst, "images", s, n + ".png"))
        with open(os.path.join(a.dst, "labels", s, n + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        kept += len(lines)
    with open(os.path.join(a.dst, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write(f"path: {os.path.abspath(a.dst)}\ntrain: images/train\nval: images/val\nnames:\n  0: rod\n")
    print(f"{len(names)} frames ({n_val} val), {kept} rod polygons -> {a.dst}/data.yaml")


if __name__ == "__main__":
    main()
