"""Rotation helpers on plain numpy: matrices in, matrices out.

Quaternions appear only at the edges (ROS PoseStamped is x, y, z, w; so is
Isaac Lab 3.0). Everything inside the runtime is a 3x3 matrix, so the
observation terms that compare orientations (bolt_rot_rel takes two columns
of a relative rotation matrix) cannot pick up a convention error.
"""
import math

import numpy as np


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp(rotvec):
    """Rotation matrix of an axis-angle vector (Rodrigues)."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3) + skew(rotvec)
    k = np.asarray(rotvec, dtype=float) / angle
    K = skew(k)
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def log(R):
    """Axis-angle vector of a rotation matrix, angle in [0, pi]."""
    R = np.asarray(R, dtype=float)
    cos = (np.trace(R) - 1.0) / 2.0
    cos = min(1.0, max(-1.0, cos))
    angle = math.acos(cos)
    if angle < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    if angle > math.pi - 1e-6:
        # Near pi the off-diagonal formula loses the axis; take it from the
        # symmetric part instead.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        # fix the signs from the off-diagonal terms
        i = int(np.argmax(axis))
        for j in range(3):
            if j != i and A[i, j] < 0.0:
                axis[j] = -axis[j]
        n = np.linalg.norm(axis)
        return axis / n * angle if n > 0.0 else np.zeros(3)
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * (angle / (2.0 * math.sin(angle)))


def rpy(roll, pitch, yaw):
    """URDF convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def from_quat_xyzw(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        raise ValueError('zero quaternion')
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def to_quat_xyzw(R):
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def transform(xyz, R):
    """4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T
