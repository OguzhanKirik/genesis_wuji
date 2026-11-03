"""Rod pose from a depth image: back-projection, a RANSAC cylinder fit with the rod's known radius, and the pose.

Pure numpy, no ROS. The rod of adam-rl's reach-to-grasp task is a cylinder of radius 12.5 mm and length 35 mm
(genesis_rl/tasks/wuji_pipeline/env_cfg.py: BOLT_RADIUS, BOLT_HALF_LEN); its axis is the object's local z. A
depth camera sees at most the upper half of its surface, so the fit leans on the known radius: two surface
points with normals fix an axis (the axis is perpendicular to both normals and a radius behind each point), the
inliers are the points a radius from that axis, and the final axis is polished on the inliers.

    pts = backproject(depth_m, K, mask)                  # (N, 3) camera frame
    nrm = normals_from_depth(depth_m, K)[mask]           # (N, 3) outward (towards the camera)
    fit = fit_cylinder(pts, nrm, radius=0.0125, half_len=0.0175)
    fit.centre, fit.axis, fit.inlier_frac, fit.visible_len, fit.ends_visible

`pose_quat_xyzw(axis, up)` turns an axis into the rod's orientation with the convention the selector expects:
local z along the axis with its raised end up, local x horizontal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

ROD_RADIUS = 0.0125     # m
ROD_HALF_LEN = 0.0175   # m


@dataclass(frozen=True)
class CylinderFit:
    centre: np.ndarray      # (3,) midpoint of the visible extent on the axis, camera frame (m)
    axis: np.ndarray        # (3,) unit, sign arbitrary
    inlier_frac: float      # inliers / points given to the fit
    n_inliers: int
    residual_m: float       # RMS of |distance to axis - radius| over the inliers
    visible_len: float      # extent of the inliers along the axis (m)
    ends_visible: bool      # visible_len covers the rod's length within `end_margin`

    @property
    def ok(self) -> bool:
        return self.n_inliers > 0


def backproject(depth_m: np.ndarray, K: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Pixels with a valid depth (> 0, finite, and inside `mask`) as (N, 3) camera-frame points (x right, y down,
    z forward, the optical convention of sensor_msgs/CameraInfo)."""
    h, w = depth_m.shape
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if mask is not None:
        valid &= mask.astype(bool)
    v, u = np.nonzero(valid)
    z = depth_m[v, u]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)


def normals_from_depth(depth_m: np.ndarray, K: np.ndarray, smooth: int = 2) -> np.ndarray:
    """(H, W, 3) unit surface normals pointing towards the camera, from the image-space derivatives of the
    back-projected points. NaN where the depth or a neighbour is invalid. Cheap (no neighbour search) and good
    enough for a 25 mm cylinder at 0.5-0.8 m with a 640-wide depth image once the depth is smoothed: `smooth`
    is the half-width of a NaN-aware box filter applied first (2 = 5x5), which a RealSense's 1-2 mm of depth
    noise needs; 0 disables it."""
    h, w = depth_m.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.arange(w)[None, :]
    v = np.arange(h)[:, None]
    z = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m, np.nan)
    if smooth > 0:
        z = _nan_mean_filter(z, smooth)
    P = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)
    du = np.full_like(P, np.nan)
    dv = np.full_like(P, np.nan)
    du[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / 2
    dv[1:-1, :] = (P[2:, :] - P[:-2, :]) / 2
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.where(norm > 0, norm, np.nan)
    # orient towards the camera: n . (-P) > 0
    flip = np.sum(n * P, axis=-1) > 0
    n[flip] *= -1
    return n


def _nan_mean_filter(z: np.ndarray, half: int) -> np.ndarray:
    """Box mean over (2*half+1)^2 ignoring NaN; NaN where the centre pixel is NaN (no filling across edges)."""
    from scipy import ndimage
    size = 2 * half + 1
    valid = np.isfinite(z)
    num = ndimage.uniform_filter(np.where(valid, z, 0.0), size=size, mode='constant')
    den = ndimage.uniform_filter(valid.astype(float), size=size, mode='constant')
    out = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    out[~valid] = np.nan
    return out


def _axis_from_pair(p1, n1, p2, n2, radius):
    """Axis (point, direction) from two oriented surface points of a cylinder of known radius, or None."""
    d = np.cross(n1, n2)
    nd = np.linalg.norm(d)
    if nd < 0.2:  # nearly parallel normals: the pair sits on one generator line, the axis is unconstrained
        return None
    d /= nd
    c1 = p1 - radius * n1
    c2 = p2 - radius * n2
    # both c1 and c2 should lie on the axis; take their mean after removing the along-axis difference
    mid = (c1 + c2) / 2
    off = (c2 - c1) - np.dot(c2 - c1, d) * d
    if np.linalg.norm(off) > 0.5 * radius:
        return None
    return mid, d


def _pca_init(pts: np.ndarray, radius: float):
    """(point on axis, direction) from the point cloud alone: direction = first principal component, point =
    the surface mean pushed one radius away from the camera (the optical centre is the origin). None when the
    cloud has no elongation to speak of."""
    if len(pts) < 10:
        return None
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    w, v = np.linalg.eigh(cov)
    d = v[:, np.argmax(w)]
    view = mean / max(np.linalg.norm(mean), 1e-9)
    view -= np.dot(view, d) * d                 # push perpendicular to the axis
    n = np.linalg.norm(view)
    if n < 1e-6:
        return None
    return mean + radius * view / n, d / np.linalg.norm(d)


def _dist_to_line(pts, c, d):
    v = pts - c
    return np.linalg.norm(v - np.outer(v @ d, d), axis=1)


def _polish(pts, nrm, c, d, radius, iters=50):
    """Least squares on (c, d) minimising sum (distance to axis - radius)^2. c moves only across d and d is
    parametrised by two angles about a basis normal to it, so the problem has four parameters."""
    from scipy.optimize import least_squares
    d = d / np.linalg.norm(d)
    e1 = np.cross(d, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 0.3:
        e1 = np.cross(d, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)

    def unpack(x):
        cc = c + x[0] * e1 + x[1] * e2
        dd = d + x[2] * e1 + x[3] * e2
        return cc, dd / np.linalg.norm(dd)

    def resid(x):
        cc, dd = unpack(x)
        return _dist_to_line(pts, cc, dd) - radius

    try:
        sol = least_squares(resid, np.zeros(4), method='lm', max_nfev=iters * 5)
        return unpack(sol.x)
    except Exception:
        return c, d


def fit_cylinder(pts: np.ndarray, nrm: np.ndarray, radius: float = ROD_RADIUS, half_len: float = ROD_HALF_LEN,
                 inlier_tol: float = 0.002, n_iter: int = 200, end_margin: float = 0.006,
                 rng: Optional[np.random.Generator] = None) -> CylinderFit:
    """RANSAC cylinder of known `radius` through oriented points. Points with a NaN normal are ignored for sampling
    but still scored. `inlier_tol` is the allowed |distance to axis - radius| (2 mm: a RealSense's depth noise at
    0.6 m plus the normal error). Returns the best model polished on its inliers; `n_inliers == 0` when nothing fit."""
    rng = rng or np.random.default_rng(0)
    pts = np.asarray(pts, float)
    nrm = np.asarray(nrm, float)
    empty = CylinderFit(np.full(3, np.nan), np.full(3, np.nan), 0.0, 0, np.nan, 0.0, False)
    if len(pts) < 20:
        return empty
    has_n = np.isfinite(nrm).all(axis=1)
    idx = np.nonzero(has_n)[0]
    if len(idx) < 10:
        return empty
    best, best_n = None, 0
    for _ in range(n_iter):
        i, j = rng.choice(idx, 2, replace=False)
        m = _axis_from_pair(pts[i], nrm[i], pts[j], nrm[j], radius)
        if m is None:
            continue
        c, d = m
        inl = np.abs(_dist_to_line(pts, c, d) - radius) < inlier_tol
        n = int(inl.sum())
        if n > best_n:
            best, best_n = (c, d, inl), n
            if n > 0.9 * len(pts):
                break
    # A second candidate that needs no normals: the points' principal axis, the axis line a radius behind the
    # surface mean (away from the camera). At a coarse resolution (a dozen pixels across the rod) the normals
    # are noise and this initialisation, polished on the points alone, is the one that converges.
    pca = _pca_init(pts, radius)
    if pca is not None:
        c0, d0 = _polish(pts, nrm, pca[0], pca[1], radius)
        inl0 = np.abs(_dist_to_line(pts, c0, d0) - radius) < inlier_tol
        if best is None or inl0.sum() > best_n:
            best, best_n = (c0, d0, inl0), int(inl0.sum())
    if best is None or best_n < 10:
        return empty
    c, d, inl = best
    # polish in rounds: the RANSAC axis comes from two noisy normals, so its inlier set is biased; re-select
    # the inliers after each polish (a looser band first) until the set stops changing.
    for tol in (2 * inlier_tol, inlier_tol, inlier_tol):
        if inl.sum() < 10:
            return empty
        c, d = _polish(pts[inl], nrm[inl], c, d, radius)
        new_inl = np.abs(_dist_to_line(pts, c, d) - radius) < tol
        if np.array_equal(new_inl, inl):
            break
        inl = new_inl
    if inl.sum() < 10:
        return empty
    P = pts[inl]
    s = (P - c) @ d
    s_lo, s_hi = np.percentile(s, [1, 99])     # robust extent: a few stray pixels must not stretch the rod
    centre = c + d * (s_lo + s_hi) / 2
    res = np.sqrt(np.mean((_dist_to_line(P, c, d) - radius) ** 2))
    vis = float(s_hi - s_lo)
    return CylinderFit(centre=centre, axis=d, inlier_frac=float(inl.mean()), n_inliers=int(inl.sum()),
                       residual_m=float(res), visible_len=vis, ends_visible=vis >= 2 * half_len - end_margin)


def pose_quat_xyzw(axis: np.ndarray, up: np.ndarray = (0.0, 0.0, 1.0)) -> tuple:
    """Orientation (xyzw) of a rod whose local z is `axis`: z = axis with its `up` component non-negative (the
    raised end up, as genesis_policy.selector.rod_axes assumes), x horizontal (perpendicular to `up` and z), y = z x x.
    A standing rod (axis along `up`) takes x along the world x projected off the axis. Returns (quat, z) so the
    caller can read the sign-fixed axis back."""
    z = np.asarray(axis, float)
    z = z / np.linalg.norm(z)
    up = np.asarray(up, float)
    if np.dot(z, up) < 0:
        z = -z
    x = np.cross(up, z)
    if np.linalg.norm(x) < 0.1:
        x = np.array([1.0, 0.0, 0.0]) - z * z[0]
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    return _quat_from_matrix(R), z


def _quat_from_matrix(R: np.ndarray) -> tuple:
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return tuple(q / np.linalg.norm(q))


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to (N, 3) points."""
    return pts @ T[:3, :3].T + T[:3, 3]


def fit_cylinders_sequential(pts: np.ndarray, nrm: np.ndarray, radius: float = ROD_RADIUS, half_len: float = ROD_HALF_LEN,
                             inlier_tol: float = 0.002, n_iter: int = 200, min_inliers: int = 40, max_rods: int = 40,
                             max_len_margin: float = 0.012, rng: Optional[np.random.Generator] = None) -> list:
    """Several rods in one point set (a heap the segmenter could not split): fit the best cylinder, drop its
    points (the inliers and everything inside its radius), fit again, until fewer than `min_inliers` points
    remain, `max_rods` is reached, or a fit fails. A fit longer than the rod (`2 * half_len + max_len_margin`:
    collinear neighbours) is dropped and its points removed anyway so the search moves on. Returns the kept
    fits, best first. Each fit's `inlier_frac` is relative to the points still in play at its turn."""
    rng = rng or np.random.default_rng(0)
    pts = np.asarray(pts, float)
    nrm = np.asarray(nrm, float)
    keep = np.ones(len(pts), bool)
    out = []
    for _ in range(max_rods):
        if keep.sum() < max(min_inliers, 20):
            break
        fit = fit_cylinder(pts[keep], nrm[keep], radius=radius, half_len=half_len, inlier_tol=inlier_tol,
                           n_iter=n_iter, rng=rng)
        if not fit.ok or fit.n_inliers < min_inliers:
            break
        d = _dist_to_line(pts, fit.centre, fit.axis)
        s = (pts - fit.centre) @ fit.axis
        along = np.abs(s) <= fit.visible_len / 2 + 0.004
        taken = ((np.abs(d - radius) < inlier_tol) | (d < radius)) & along
        if fit.visible_len <= 2 * half_len + max_len_margin:
            out.append(fit)
        if taken.sum() == 0:
            break
        keep &= ~taken
    return out

