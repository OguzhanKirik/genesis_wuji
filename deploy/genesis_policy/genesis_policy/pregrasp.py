"""The pre-grasp wrist pose the policy's episode starts from, for the planner to deliver.

Reproduces adam-rl's env_cfg._nominal_wrist_general for one rod: the simulator resets the wrist at

    position    = g - R(qw) bolt_offset_xyz + standoff * R(qw) z
    orientation = qw * wrist_rot
    qw          = roll(-theta about n) * yaw(psi)

with g the grasp point (the rod's centre; the commanded grasp offset along the axis is zero on the robot), psi
the hand yaw (the selector's `Choice.hand_yaw`, a yaw the policy trained with for that rod), theta the rod's
elevation for a leaning rod (zero for a lying one, capped by the level's roll cap) and n = z x axis the
horizontal axis of that tilt, so the approach axis R(qw) z is the rod's up-normal. The constants come from
level.yaml's `pregrasp` section (tools/export_policy.py, 2026-11-03 or later). Quaternions are xyzw.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LYING, LEANING, STANDING = 'lying', 'leaning', 'standing'


class PregraspError(ValueError):
    pass


@dataclass(frozen=True)
class Pregrasp:
    position: np.ndarray       # (3,) world
    quat_xyzw: np.ndarray      # (4,) world
    approach: np.ndarray       # (3,) unit, the direction the policy will descend along (negated)
    kind: str
    theta: float               # rad, the wrist roll applied


def quat_mul(a, b):
    """Hamilton product, xyzw: rotation b first, then a."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def quat_from_angle_axis(angle, axis):
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = axis / n
    return np.array([*(axis * math.sin(angle / 2)), math.cos(angle / 2)])


def quat_apply(q, v):
    x, y, z, w = q
    qv = np.array([x, y, z])
    t = 2 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def pregrasp_pose(rod_position, rod_axis, hand_yaw, level, standoff=None) -> Pregrasp:
    """World pose of the policy's wrist frame (level.wrist_link) at the start of an episode on this rod."""
    raw = level.raw if isinstance(getattr(level, 'raw', None), dict) else {}
    pg = raw.get('pregrasp')
    if not isinstance(pg, dict):
        raise PregraspError(f'{level.source}: level.yaml has no pregrasp section; re-export it with adam-rl '
                            'tools/export_policy.py (2026-11-03 or later)')
    ws = raw.get('workspace') or {}
    wrist_rot = np.asarray(pg['wrist_rot_xyzw'], float)
    off3 = np.asarray(pg['bolt_offset_xyz'], float)
    lo, hi = (float(v) for v in pg['standoff_m'])
    h = float(standoff) if standoff is not None else (lo + hi) / 2
    roll_cap = float(getattr(level, 'roll_cap', math.pi / 2)) if bool(pg.get('wrist_roll', False)) else 0.0

    a = np.asarray(rod_axis, float)
    a = a / np.linalg.norm(a)
    if a[2] < 0:
        a = -a
    elev = math.asin(min(1.0, max(-1.0, float(a[2]))))
    lying_max = math.radians(float(ws.get('lying_max_deg', 12.0)))
    leaning_max = math.radians(float(ws.get('leaning_max_deg', 40.0)))
    if elev < lying_max:
        kind, theta = LYING, 0.0
    elif elev < leaning_max:
        kind, theta = LEANING, min(elev, roll_cap)
    else:
        raise PregraspError(f'rod elevation {math.degrees(elev):.1f} deg: standing rods have no pre-grasp pose yet')

    qz = quat_from_angle_axis(float(hand_yaw), [0.0, 0.0, 1.0])
    n = np.cross([0.0, 0.0, 1.0], a)
    qt = quat_from_angle_axis(-theta, n) if theta > 0 else np.array([0.0, 0.0, 0.0, 1.0])
    qw = quat_mul(qt, qz)
    m = quat_apply(qw, np.array([0.0, 0.0, 1.0]))
    g = np.asarray(rod_position, float)
    pos = g - quat_apply(qw, off3) + h * m
    quat = quat_mul(qw, wrist_rot)
    quat = quat / np.linalg.norm(quat)
    return Pregrasp(position=pos, quat_xyzw=quat, approach=m, kind=kind, theta=theta)
