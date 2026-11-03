import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from synth import K, fill_holes, render   # noqa: E402

from genesis_perception.rod_fit import ROD_HALF_LEN, ROD_RADIUS, backproject, fit_cylinder, normals_from_depth, pose_quat_xyzw  # noqa: E402


def _fit_rod(centre, axis, noise_m=0.0, extra=()):
    depth, ids = render([(centre, axis, ROD_RADIUS, ROD_HALF_LEN)] + list(extra), noise_m=noise_m)
    depth, ids = fill_holes(depth, ids)
    mask = ids == 1
    pts = backproject(depth, K, mask)
    nrm = normals_from_depth(depth, K)[mask & np.isfinite(depth)]
    return fit_cylinder(pts, nrm, rng=np.random.default_rng(0))


def _axis_err_deg(a, b):
    return math.degrees(math.acos(min(1.0, abs(float(np.dot(a, b))))))


@pytest.mark.parametrize('axis, axis_tol_deg', [([1, 0, 0], 1.0), ([0, 1, 0], 1.0), ([0.7, 0.7, 0.1], 1.0),
                                                 ([0.3, -0.2, 0.93], 5.0)])
def test_clean_rod_at_60cm(axis, axis_tol_deg):
    """A rod seen side-on fits to better than a degree. The last case is a rod pointing almost at the camera
    (21 deg off the view direction: a standing rod under a down-looking head camera), where only a sliver of the
    side is visible and the axis is worth about 4 deg; the selector treats standing rods separately anyway."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    centre = np.array([0.03, -0.02, 0.60])
    fit = _fit_rod(centre, axis)
    assert fit.ok and fit.inlier_frac > 0.8, fit
    assert np.linalg.norm(fit.centre - centre) < 0.002, fit
    assert _axis_err_deg(fit.axis, axis) < axis_tol_deg, fit
    assert fit.residual_m < 0.001


def test_noisy_rod_inside_budget():
    axis = np.array([0.8, 0.6, 0.0])
    centre = np.array([-0.05, 0.04, 0.65])
    fit = _fit_rod(centre, axis, noise_m=0.0015)       # RealSense-grade depth noise at 0.65 m
    assert fit.ok and fit.inlier_frac > 0.5, fit
    assert np.linalg.norm(fit.centre - centre) < 0.004, fit
    assert _axis_err_deg(fit.axis, axis) < 3.0, fit


def test_outliers_from_a_neighbour_in_the_mask():
    """The mask bleeds into a touching neighbour: RANSAC must still fit the first rod."""
    axis = np.array([1.0, 0.0, 0.0])
    centre = np.array([0.0, 0.0, 0.60])
    neighbour = (centre + np.array([0.0, 2 * ROD_RADIUS + 0.001, 0.0]), axis, ROD_RADIUS, ROD_HALF_LEN)
    depth, ids = render([(centre, axis, ROD_RADIUS, ROD_HALF_LEN), neighbour])
    depth, ids = fill_holes(depth, ids)
    mask = ids > 0                                      # both rods in one mask
    pts = backproject(depth, K, mask)
    nrm = normals_from_depth(depth, K)[mask & np.isfinite(depth)]
    fit = fit_cylinder(pts, nrm, rng=np.random.default_rng(0))
    assert fit.ok
    d = min(np.linalg.norm(fit.centre - centre), np.linalg.norm(fit.centre - neighbour[0]))
    assert d < 0.003, fit                                # it fit one of the two, not a blend
    assert _axis_err_deg(fit.axis, axis) < 1.5
    assert 0.35 < fit.inlier_frac < 0.75                 # about half the points belong to the other rod


def test_hidden_end_is_reported():
    axis = np.array([1.0, 0.0, 0.0])
    centre = np.array([0.0, 0.0, 0.60])
    depth, ids = render([(centre, axis, ROD_RADIUS, ROD_HALF_LEN)])
    depth, ids = fill_holes(depth, ids)
    u_centre = int(K[0, 2])
    mask = (ids == 1)
    mask[:, u_centre:] = False                           # the +x half is occluded
    pts = backproject(depth, K, mask)
    nrm = normals_from_depth(depth, K)[mask & np.isfinite(depth)]
    fit = fit_cylinder(pts, nrm, rng=np.random.default_rng(0))
    assert fit.ok and not fit.ends_visible
    assert fit.visible_len < ROD_HALF_LEN + 0.004
    assert _axis_err_deg(fit.axis, axis) < 1.5
    assert abs(fit.centre[0] - (centre[0] - ROD_HALF_LEN / 2)) < 0.003   # midpoint of the visible half


def test_too_few_points_is_not_a_fit():
    fit = fit_cylinder(np.zeros((5, 3)), np.zeros((5, 3)))
    assert not fit.ok and fit.n_inliers == 0


def test_pose_quat_raised_end_up_and_x_horizontal():
    q, z = pose_quat_xyzw([0.0, 0.6, -0.8], up=[0, 0, 1])
    assert z[2] > 0
    x, y, zz, w = q
    R = np.array([[1 - 2 * (y * y + zz * zz), 2 * (x * y - zz * w), 2 * (x * zz + y * w)],
                  [2 * (x * y + zz * w), 1 - 2 * (x * x + zz * zz), 2 * (y * zz - x * w)],
                  [2 * (x * zz - y * w), 2 * (y * zz + x * w), 1 - 2 * (x * x + y * y)]])
    assert np.allclose(R[:, 2], z, atol=1e-9)
    assert abs(R[2, 0]) < 1e-9                           # local x horizontal
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    q2, z2 = pose_quat_xyzw([0, 0, 1.0])                 # standing rod: no singularity
    assert np.isfinite(q2).all() and z2[2] > 0


@pytest.mark.parametrize('axis', [[1, 0, 0], [0, 1, 0], [0.7, 0.7, 0.0]])
def test_coarse_head_camera_resolution(axis):
    """The sim head camera: 87 deg HFOV at 640 px (fx 337) from 0.73 m, so the rod is ~12 px wide and ~16 px
    long. Position stays inside the 5 mm budget; the axis is worth a few degrees at this resolution."""
    from synth import render as _render
    Kc = np.array([[337.2, 0, 320.0], [0, 337.2, 240.0], [0, 0, 1.0]])
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    centre = np.array([0.02, -0.03, 0.73])
    depth, ids = _render([(centre, axis, ROD_RADIUS, ROD_HALF_LEN)], K=Kc)
    depth, ids = fill_holes(depth, ids)
    mask = ids == 1
    pts = backproject(depth, Kc, mask)
    nrm = normals_from_depth(depth, Kc)[mask & np.isfinite(depth)]
    fit = fit_cylinder(pts, nrm, rng=np.random.default_rng(0))
    assert fit.ok, fit
    assert np.linalg.norm(fit.centre - centre) < 0.005, fit
    assert _axis_err_deg(fit.axis, axis) < 5.0, fit
