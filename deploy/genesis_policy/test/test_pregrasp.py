import math

import numpy as np
import pytest

from genesis_policy import pregrasp as P


def _R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_lying_rod_yaw_zero_matches_the_nominal_wrist(level):
    g = np.array([0.5474, -0.1099, 0.961])
    pg = P.pregrasp_pose(g, [1.0, 0.0, 0.0], 0.0, level, standoff=0.08)
    assert pg.kind == 'lying' and pg.theta == 0.0
    # xy: the level's nominal wrist offset (WRIST_XY - BOLT_XY); z: contact height plus the standoff
    assert np.allclose(pg.position[:2], g[:2] + level.nominal_wrist_offset_xy, atol=1e-6)
    off_z = level.raw['pregrasp']['bolt_offset_xyz'][2]
    assert abs(pg.position[2] - (g[2] - off_z + 0.08)) < 1e-9
    w0 = np.asarray(level.raw['pregrasp']['wrist_rot_xyzw'])
    assert np.allclose(pg.quat_xyzw, w0 / np.linalg.norm(w0), atol=1e-6)   # the constant is rounded to 4 decimals
    assert np.allclose(pg.approach, [0, 0, 1])


def test_yawed_hand_rotates_the_offset_about_the_rod(level):
    g = np.array([0.5, -0.1, 0.961])
    psi = math.radians(90)
    pg = P.pregrasp_pose(g, [1.0, 0.0, 0.0], psi, level, standoff=0.05)
    off = np.asarray(level.raw['pregrasp']['bolt_offset_xyz'][:2])
    Rz = np.array([[0, -1], [1, 0]])
    assert np.allclose(pg.position[:2], g[:2] - Rz @ off, atol=1e-9)
    R = _R(pg.quat_xyzw)
    w0 = np.asarray(level.raw['pregrasp']['wrist_rot_xyzw'])
    R0 = _R(w0 / np.linalg.norm(w0))
    assert np.allclose(R, _R(P.quat_from_angle_axis(psi, [0, 0, 1])) @ R0, atol=1e-6)


def test_leaning_rod_rolls_the_approach_onto_the_up_normal(level):
    a = np.array([math.cos(math.radians(30)), 0.0, math.sin(math.radians(30))])
    pg = P.pregrasp_pose([0.5, -0.1, 0.98], a, 0.0, level, standoff=0.06)
    assert pg.kind == 'leaning' and abs(pg.theta - math.radians(30)) < 1e-9
    assert abs(np.dot(pg.approach, a)) < 1e-9                 # approach perpendicular to the rod
    assert abs(pg.approach[2] - math.cos(math.radians(30))) < 1e-9
    assert pg.approach[0] < 0                                 # tilted against the raised end


def test_axis_sign_does_not_matter(level):
    p1 = P.pregrasp_pose([0.5, -0.1, 0.961], [0.7, 0.7, 0.0], 0.3, level)
    p2 = P.pregrasp_pose([0.5, -0.1, 0.961], [-0.7, -0.7, 0.0], 0.3, level)
    assert np.allclose(p1.position, p2.position) and np.allclose(p1.quat_xyzw, p2.quat_xyzw)


def test_standing_rod_is_refused(level):
    with pytest.raises(P.PregraspError):
        P.pregrasp_pose([0.5, -0.1, 0.98], [0.0, 0.0, 1.0], 0.0, level)
