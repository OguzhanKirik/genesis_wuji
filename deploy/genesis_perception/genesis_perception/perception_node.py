#!/usr/bin/env python3
"""Rod candidates from the head camera: depth + camera_info -> masks -> cylinder fits -> /perception/rod_candidates.

Replaces genesis_isaac's rod_candidates.py stand-in and keeps its contract: one geometry_msgs/PoseArray in the
world frame, one pose per rod, the rod's local z along its axis with the raised end up. The selector
(genesis_policy.selector) picks the rod and latches on it; this node is stateless and publishes what it sees
every frame, so an occluded rod simply drops out of the list.

Works on the Isaac head camera (32FC1 depth in metres) and on a RealSense (16UC1 depth in millimetres); the
topics are parameters. Needs TF from the camera's optical frame to `world_frame` at the image stamp.

A second topic, /perception/rod_fit_stats (std_msgs/String, JSON), carries the per-candidate fit statistics
(inlier fraction, residual, visible length, whether both ends are visible) for the accuracy test and for tuning.
"""
import json
import time

import numpy as np
import rclpy
import rclpy.time
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
import tf2_ros

from .rod_fit import ROD_HALF_LEN, ROD_RADIUS, backproject, fit_cylinder, fit_cylinders_sequential, normals_from_depth, pose_quat_xyzw, transform_points
from .segment import BinRegion, DepthClusterSegmenter, YoloSegmenter


def image_to_depth_m(msg: Image) -> np.ndarray:
    """sensor_msgs/Image -> float32 metres (NaN where invalid), without cv_bridge."""
    h, w = msg.height, msg.width
    if msg.encoding in ('32FC1',):
        d = np.frombuffer(msg.data, dtype=np.float32).reshape(h, msg.step // 4)[:, :w].astype(np.float32)
    elif msg.encoding in ('16UC1', 'mono16'):
        d = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, msg.step // 2)[:, :w].astype(np.float32) / 1000.0
    else:
        raise ValueError(f'unsupported depth encoding {msg.encoding}')
    d = d.copy()
    d[d <= 0] = np.nan
    return d


def image_to_rgb(msg: Image) -> np.ndarray:
    h, w = msg.height, msg.width
    if msg.encoding in ('rgb8', 'bgr8'):
        a = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, msg.step)[:, :3 * w].reshape(h, w, 3)
        return a[:, :, ::-1] if msg.encoding == 'bgr8' else a
    if msg.encoding == 'rgba8':
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, msg.step)[:, :4 * w].reshape(h, w, 4)[:, :, :3]
    raise ValueError(f'unsupported colour encoding {msg.encoding}')


def tf_to_matrix(t) -> np.ndarray:
    q = t.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
    return T


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('rod_perception')
        p = dict(depth_topic='/head_camera/depth/image_raw', info_topic='/head_camera/camera_info',
                 rgb_topic='', camera_frame='', world_frame='world', candidates_topic='/perception/rod_candidates',
                 stats_topic='/perception/rod_fit_stats', segmenter='depth', yolo_model='', yolo_conf=0.5,
                 rod_radius_m=ROD_RADIUS, rod_half_len_m=ROD_HALF_LEN,
                 bin_xy_min=[-0.2, -0.5], bin_xy_max=[0.2, -0.1], bin_floor_z=0.95, floor_margin_m=0.006,
                 min_pixels=60, min_inlier_frac=0.6, max_residual_m=0.0015, inlier_tol_m=0.002, ransac_iters=200,
                 split_above_pixels=600, split_min_inliers=40, split_max_rods=40,
                 max_rate_hz=15.0, tf_timeout_s=0.05)
        for k, v in p.items():
            self.declare_parameter(k, v)
        self.p = {k: self.get_parameter(k).value for k in p}
        self.K = None
        self.rgb = None
        self.last = 0.0
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        region = BinRegion(np.asarray(self.p['bin_xy_min'], float), np.asarray(self.p['bin_xy_max'], float),
                           float(self.p['bin_floor_z']), float(self.p['floor_margin_m']))
        if self.p['segmenter'] == 'yolo':
            self.seg = YoloSegmenter(self.p['yolo_model'], conf=float(self.p['yolo_conf']))
        elif self.p['segmenter'] == 'depth':
            self.seg = DepthClusterSegmenter(region, min_pixels=int(self.p['min_pixels']))
        else:
            raise ValueError(f"segmenter must be 'depth' or 'yolo', not {self.p['segmenter']!r}")
        sensor_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(CameraInfo, self.p['info_topic'], self._info, sensor_qos)
        self.create_subscription(Image, self.p['depth_topic'], self._depth, sensor_qos)
        if self.p['rgb_topic']:
            self.create_subscription(Image, self.p['rgb_topic'], self._rgb, sensor_qos)
        self.pub = self.create_publisher(PoseArray, self.p['candidates_topic'], 10)
        self.pub_stats = self.create_publisher(String, self.p['stats_topic'], 10)
        self.rng = np.random.default_rng(0)
        self.get_logger().info(f"{self.p['depth_topic']} -> {self.p['candidates_topic']} ({self.p['segmenter']} segmenter, "
                               f"rod r={1000 * self.p['rod_radius_m']:.1f} mm)")

    def _info(self, msg: CameraInfo):
        self.K = np.array(msg.k, float).reshape(3, 3)
        if not self.p['camera_frame']:
            self.p['camera_frame'] = msg.header.frame_id

    def _rgb(self, msg: Image):
        self.rgb = msg

    def _depth(self, msg: Image):
        now = time.monotonic()
        if self.K is None or now - self.last < 1.0 / float(self.p['max_rate_hz']):
            return
        self.last = now
        frame = self.p['camera_frame'] or msg.header.frame_id
        try:
            t = self.tf_buffer.lookup_transform(self.p['world_frame'], frame, msg.header.stamp,
                                                timeout=rclpy.duration.Duration(seconds=float(self.p['tf_timeout_s'])))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
            # the image stamp is usually a few ms ahead of the newest transform; the camera does not move
            # within a frame, so the latest transform is the right one (a moving head would need the wait)
            try:
                t = self.tf_buffer.lookup_transform(self.p['world_frame'], frame, rclpy.time.Time())
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
                self.get_logger().warn(f'no TF {self.p["world_frame"]} <- {frame}: {e}', throttle_duration_sec=5.0)
                return
        T = tf_to_matrix(t)
        depth = image_to_depth_m(msg)
        rgb = image_to_rgb(self.rgb) if (self.rgb is not None and self.p['segmenter'] == 'yolo') else None
        masks = self.seg.segment(rgb, depth, self.K, T)
        normals = normals_from_depth(depth, self.K)
        out = PoseArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.p['world_frame']
        stats = []
        up_cam = T[:3, :3].T @ np.array([0.0, 0.0, 1.0])     # world up seen in the camera frame
        half_len = float(self.p['rod_half_len_m'])
        for m in masks:
            pts = backproject(depth, self.K, m)
            nrm = normals[m & np.isfinite(depth)]
            kw = dict(radius=float(self.p['rod_radius_m']), half_len=half_len, inlier_tol=float(self.p['inlier_tol_m']),
                      n_iter=int(self.p['ransac_iters']), rng=self.rng)
            if m.sum() > int(self.p['split_above_pixels']):
                # a blob of touching rods: sequential fits, judged by inlier count rather than fraction
                fits = fit_cylinders_sequential(pts, nrm, min_inliers=int(self.p['split_min_inliers']),
                                                max_rods=int(self.p['split_max_rods']), **kw)
                judge = lambda f: f.n_inliers >= int(self.p['split_min_inliers'])  # noqa: E731
            else:
                fits = [fit_cylinder(pts, nrm, **kw)]
                judge = lambda f: f.inlier_frac >= float(self.p['min_inlier_frac'])  # noqa: E731
            for fit in fits:
                rec = dict(pixels=int(m.sum()), ok=bool(fit.ok), inlier_frac=round(fit.inlier_frac, 3), n_inliers=int(fit.n_inliers),
                           residual_mm=None if not fit.ok else round(1000 * fit.residual_m, 2),
                           visible_len_mm=round(1000 * fit.visible_len, 1), ends_visible=bool(fit.ends_visible), accepted=False)
                too_long = fit.visible_len > 2 * half_len + 0.012
                if fit.ok and judge(fit) and fit.residual_m <= float(self.p['max_residual_m']) and not too_long:
                    q, _ = pose_quat_xyzw(fit.axis, up_cam)
                    centre_w = transform_points(T, fit.centre[None])[0]
                    R = T[:3, :3] @ _quat_to_matrix(q)
                    qw = _matrix_to_quat(R)
                    pose = Pose()
                    pose.position.x, pose.position.y, pose.position.z = map(float, centre_w)
                    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, qw)
                    out.poses.append(pose)
                    rec['accepted'] = True
                    rec['centre_world'] = [round(float(v), 4) for v in centre_w]
                stats.append(rec)
        self.pub.publish(out)
        self.pub_stats.publish(String(data=json.dumps(dict(stamp=msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec,
                                                           masks=len(masks), accepted=len(out.poses), fits=stats))))


def _quat_to_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _matrix_to_quat(R):
    from .rod_fit import _quat_from_matrix
    return _quat_from_matrix(R)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
