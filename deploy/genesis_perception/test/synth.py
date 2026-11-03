"""Synthetic depth images of rods for the tests: a z-buffer render of cylinder surfaces with a pinhole camera."""
import numpy as np

K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
H, W = 480, 640


def cylinder_points(centre, axis, radius, half_len, n=40000, rng=None):
    rng = rng or np.random.default_rng(1)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    e1 = np.cross(axis, [1.0, 0, 0])
    if np.linalg.norm(e1) < 0.3:
        e1 = np.cross(axis, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    s = rng.uniform(-half_len, half_len, n)
    th = rng.uniform(0, 2 * np.pi, n)
    nrm = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
    pts = np.asarray(centre) + s[:, None] * axis + radius * nrm
    return pts, nrm


def render(rods, noise_m=0.0, rng=None, K=K, h=H, w=W):
    """rods: list of (centre, axis, radius, half_len) in the camera frame. Returns (depth_m, instance_ids)."""
    rng = rng or np.random.default_rng(2)
    depth = np.full((h, w), np.inf)
    ids = np.zeros((h, w), int)
    for k, (c, a, r, hl) in enumerate(rods, start=1):
        pts, nrm = cylinder_points(c, a, r, hl, rng=rng)
        front = np.sum(nrm * -pts, axis=1) > 0        # faces the camera
        pts = pts[front]
        z = pts[:, 2]
        u = np.round(K[0, 0] * pts[:, 0] / z + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * pts[:, 1] / z + K[1, 2]).astype(int)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        for uu, vv, zz in zip(u[ok], v[ok], z[ok]):
            if zz < depth[vv, uu]:
                depth[vv, uu] = zz
                ids[vv, uu] = k
    depth[~np.isfinite(depth)] = np.nan
    if noise_m > 0:
        depth = depth + rng.normal(0, noise_m, depth.shape)
    return depth, ids


def fill_holes(depth, ids, iters=2):
    """Close the one-pixel gaps of the point-splat render so image-space normals are defined (a real depth map
    is dense)."""
    from scipy import ndimage
    d = depth.copy()
    i = ids.copy()
    for _ in range(iters):
        nan = np.isnan(d)
        filled = ndimage.generic_filter(np.where(nan, np.inf, d), np.min, size=3, mode='constant', cval=np.inf)
        grow = nan & np.isfinite(filled)
        d[grow] = filled[grow]
        i_filled = ndimage.maximum_filter(i, size=3)
        i[grow] = i_filled[grow]
    return d, i
