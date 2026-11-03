"""Rod instance masks from a depth (and optionally colour) image.

Two segmenters behind one call, `segment(rgb, depth_m, K, T_world_cam) -> list[np.ndarray(bool, HxW)]`:

- `DepthClusterSegmenter`: classical. Keeps the pixels whose world point is inside the bin's xy box and above
  its floor by more than `floor_margin`, then takes 8-connected components of at least `min_pixels`. Enough for
  the table rung and for rods that do not touch; two touching rods come out as one mask, which the cylinder fit
  then rejects or fits badly. No model, no GPU.
- `YoloSegmenter`: an ultralytics YOLO segmentation model (the one trained on Replicator renders of the heap
  scene, tools/render_rod_dataset.py). Imported lazily so the package works without ultralytics installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .rod_fit import backproject, transform_points


@dataclass(frozen=True)
class BinRegion:
    """Where rods can be, in the world frame: an xy box and the floor height (bin floor or table top)."""
    xy_min: np.ndarray
    xy_max: np.ndarray
    floor_z: float
    floor_margin: float = 0.006   # m above the floor before a point counts as rod (RealSense noise at 0.6 m ~ 2-3 mm)
    ceiling: float = 0.25         # m above the floor; anything higher is the arm or the wall top

    def contains(self, pts_world: np.ndarray) -> np.ndarray:
        xy = pts_world[:, :2]
        z = pts_world[:, 2]
        return ((xy >= self.xy_min).all(axis=1) & (xy <= self.xy_max).all(axis=1)
                & (z > self.floor_z + self.floor_margin) & (z < self.floor_z + self.ceiling))


class DepthClusterSegmenter:
    def __init__(self, region: BinRegion, min_pixels: int = 60, max_pixels: int = 20000):
        self.region = region
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def segment(self, rgb: Optional[np.ndarray], depth_m: np.ndarray, K: np.ndarray,
                T_world_cam: np.ndarray) -> List[np.ndarray]:
        from scipy import ndimage
        h, w = depth_m.shape
        valid = np.isfinite(depth_m) & (depth_m > 0)
        pts = backproject(depth_m, K)
        inside = self.region.contains(transform_points(T_world_cam, pts))
        keep = np.zeros((h, w), bool)
        keep[valid] = inside
        labels, n = ndimage.label(keep, structure=np.ones((3, 3), int))
        masks = []
        for k in range(1, n + 1):
            m = labels == k
            c = int(m.sum())
            if self.min_pixels <= c <= self.max_pixels:
                masks.append(m)
        return masks


class YoloSegmenter:
    def __init__(self, model_path: str, conf: float = 0.5, device: Optional[str] = None, imgsz: int = 640):
        from ultralytics import YOLO   # optional dependency
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        self.imgsz = imgsz

    def segment(self, rgb: Optional[np.ndarray], depth_m: np.ndarray, K: np.ndarray,
                T_world_cam: np.ndarray) -> List[np.ndarray]:
        if rgb is None:
            raise ValueError('YoloSegmenter needs the colour image')
        h, w = depth_m.shape
        res = self.model.predict(rgb, conf=self.conf, device=self.device, imgsz=self.imgsz, verbose=False)[0]
        if res.masks is None:
            return []
        masks = []
        for m in res.masks.data.cpu().numpy():
            if m.shape != (h, w):
                m = _resize_nearest(m, h, w)
            masks.append(m > 0.5)
        return masks


def _resize_nearest(img: np.ndarray, h: int, w: int) -> np.ndarray:
    ys = (np.arange(h) * img.shape[0] / h).astype(int)
    xs = (np.arange(w) * img.shape[1] / w).astype(int)
    return img[ys][:, xs]
