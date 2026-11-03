import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from synth import K, fill_holes, render   # noqa: E402

from genesis_perception.rod_fit import ROD_HALF_LEN, ROD_RADIUS  # noqa: E402
from genesis_perception.segment import BinRegion, DepthClusterSegmenter  # noqa: E402


def _camera_looking_down(height=0.6):
    """T_world_cam for an optical frame (z forward, y down) looking straight down at the world origin."""
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)   # cam x = world x, cam z = world -z
    T[:3, 3] = [0, 0, height]
    return T


def test_two_separated_rods_on_a_floor():
    T = _camera_looking_down()
    Tinv = np.linalg.inv(T)
    # rods lying on the floor z = 0 at two world positions; floor as a big flat 'rod' is simpler as a plane:
    world_rods = [(np.array([-0.05, 0.0, ROD_RADIUS]), [1, 0, 0]), (np.array([0.06, 0.03, ROD_RADIUS]), [0, 1, 0])]
    cam_rods = []
    for c, a in world_rods:
        cc = Tinv[:3, :3] @ c + Tinv[:3, 3]
        aa = Tinv[:3, :3] @ np.asarray(a, float)
        cam_rods.append((cc, aa, ROD_RADIUS, ROD_HALF_LEN))
    depth, ids = render(cam_rods)
    depth, ids = fill_holes(depth, ids)
    floor = np.isnan(depth)
    depth[floor] = 0.6                                   # the floor plane at camera distance 0.6 m
    region = BinRegion(np.array([-0.2, -0.2]), np.array([0.2, 0.2]), floor_z=0.0, floor_margin=0.006)
    masks = DepthClusterSegmenter(region, min_pixels=30).segment(None, depth, K, T)
    assert len(masks) == 2
    for m in masks:
        assert m.sum() >= 30
    assert not (masks[0] & masks[1]).any()
