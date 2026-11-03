#!/usr/bin/env python3
"""Where does the truth land in the depth image, and where does the depth cloud put the rod? A one-shot probe
for frame-convention mistakes between the depth image + camera_info, TF, and Isaac's point cloud."""
import struct
import time

import numpy as np
import rclpy
import rclpy.time
import tf2_ros
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

from genesis_perception.perception_node import image_to_depth_m, tf_to_matrix
from genesis_perception.rod_fit import backproject, transform_points


def main():
    rclpy.init()
    n = Node('probe_frame')
    got = {}
    n.create_subscription(Image, '/head_camera/depth/image_raw', lambda m: got.setdefault('depth', m), qos_profile_sensor_data)
    n.create_subscription(CameraInfo, '/head_camera/camera_info', lambda m: got.setdefault('info', m), qos_profile_sensor_data)
    n.create_subscription(PointCloud2, '/head_camera/depth/points', lambda m: got.setdefault('cloud', m), qos_profile_sensor_data)
    n.create_subscription(PoseArray, '/perception/rod_truth', lambda m: got.setdefault('truth', m), 10)
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, n)
    t0 = time.time()
    while len(got) < 4 and time.time() - t0 < 15:
        rclpy.spin_once(n, timeout_sec=0.2)
    print('have', sorted(got))
    depth = image_to_depth_m(got['depth'])
    K = np.array(got['info'].k).reshape(3, 3)
    print('depth frame', got['depth'].header.frame_id, '| info frame', got['info'].header.frame_id,
          '| cloud frame', got['cloud'].header.frame_id, '| truth frame', got['truth'].header.frame_id)
    t0 = time.time()
    while not buf.can_transform('world', got['depth'].header.frame_id, rclpy.time.Time()) and time.time() - t0 < 15:
        rclpy.spin_once(n, timeout_sec=0.2)
    T = tf_to_matrix(buf.lookup_transform('world', got['depth'].header.frame_id, rclpy.time.Time()))
    print('T_world_cam translation', np.round(T[:3, 3], 4))
    tp = got['truth'].poses[0].position
    truth = np.array([tp.x, tp.y, tp.z])
    print('truth world', np.round(truth, 4))
    # truth into the image
    Tinv = np.linalg.inv(T)
    pc = Tinv[:3, :3] @ truth + Tinv[:3, 3]
    u = K[0, 0] * pc[0] / pc[2] + K[0, 2]
    v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
    print(f'truth in camera {np.round(pc, 4)} -> pixel ({u:.1f}, {v:.1f}), depth there {depth[int(round(v)), int(round(u))] if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1] else "out"}')
    # the depth image: points above the table
    pts = backproject(depth, K)
    pw = transform_points(T, pts)
    above = pw[(pw[:, 2] > 0.956) & (pw[:, 2] < 1.1)]
    print('depth-image points above the table:', len(above), 'centroid', np.round(above.mean(axis=0), 4) if len(above) else '-')
    # Isaac's cloud
    c = got['cloud']
    step = c.point_step
    off = {f.name: f.offset for f in c.fields}
    data = np.frombuffer(c.data, dtype=np.uint8).reshape(-1, step)
    xyz = np.stack([data[:, off[k]:off[k] + 4].copy().view(np.float32).ravel() for k in ('x', 'y', 'z')], axis=1)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    Tc = tf_to_matrix(buf.lookup_transform('world', c.header.frame_id, rclpy.time.Time()))
    cw = transform_points(Tc, xyz)
    cabove = cw[(cw[:, 2] > 0.956) & (cw[:, 2] < 1.1)]
    print('isaac cloud points above the table:', len(cabove), 'centroid', np.round(cabove.mean(axis=0), 4) if len(cabove) else '-')
    # the same cloud, read as if it were (x right, y down, z forward) vs the image's convention
    table = pw[(pw[:, 2] > 0.940) & (pw[:, 2] < 0.956)]
    print('table-height points: x', np.round(np.percentile(table[:, 0], [0.5, 99.5]), 3), 'y', np.round(np.percentile(table[:, 1], [0.5, 99.5]), 3), '(layout: x 0.30..0.80, y -0.40..0.40)')
    q = got['truth'].poses[0].orientation
    x, y, z, w = q.x, q.y, q.z, q.w
    ax = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])
    print('truth axis', np.round(ax, 3), '| centre if origin is an end: +', np.round(truth + 0.0175 * ax, 4), ' -', np.round(truth - 0.0175 * ax, 4))
    if len(above) > 10:
        c = above - above.mean(axis=0)
        wv, vv = np.linalg.eigh(np.cov(c.T))
        print('cloud principal axis', np.round(vv[:, np.argmax(wv)], 3))
    print('cloud sample (cam frame)', np.round(xyz[len(xyz) // 2], 3), '| depth-image sample (cam frame)', np.round(pts[len(pts) // 2], 3))
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
