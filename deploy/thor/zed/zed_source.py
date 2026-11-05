#!/usr/bin/env python3
"""The ZED camera as the perception node's depth source: depth image + camera info (+ colour) and a fixed TF.

Publishes what genesis_perception's rod_perception subscribes to, on the RealSense-style topic names the Thor's
config/perception.yaml already uses, so the perception node needs no change:

    /camera/camera/aligned_depth_to_color/image_raw     sensor_msgs/Image 32FC1, metres, NaN where invalid
    /camera/camera/aligned_depth_to_color/camera_info   sensor_msgs/CameraInfo of the left camera (depth is aligned to it)
    /camera/camera/color/image_raw                      sensor_msgs/Image rgb8 (the left image), for the YOLO segmenter
    TF world -> camera_frame (static), from the mount_xyz / mount_rpy parameters: the bench mount for now, the
    hand-eye calibration on the robot later (then publish it from the robot's TF tree instead and set publish_tf false)

The depth is the ZED SDK's (stereo, on the GPU) in the IMAGE coordinate frame: x right, y down, z forward, the
optical convention the pinhole back-projection expects. Parameters: resolution (HD720 | HD1080 | VGA), fps,
depth_mode (NEURAL_LIGHT | NEURAL | ULTRA | QUALITY | PERFORMANCE), depth_min_m, rate_hz, serial (0 = any).
"""
import math

import numpy as np
import pyzed.sl as sl
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster


def _quat_from_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


class ZedSource(Node):
    def __init__(self):
        super().__init__('zed_source')
        p = dict(depth_topic='/camera/camera/aligned_depth_to_color/image_raw',
                 info_topic='/camera/camera/aligned_depth_to_color/camera_info',
                 rgb_topic='/camera/camera/color/image_raw', camera_frame='zed_left_optical', world_frame='world',
                 publish_tf=True, mount_xyz=[0.0, 0.0, 0.7], mount_rpy=[math.pi, 0.0, 0.0],
                 resolution='HD720', fps=30, depth_mode='NEURAL_LIGHT', depth_min_m=0.2, depth_max_m=3.0,
                 rate_hz=15.0, serial=0, publish_rgb=True)
        for k, v in p.items():
            self.declare_parameter(k, v)
        self.p = {k: self.get_parameter(k).value for k in p}
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, self.p['resolution'])
        init.camera_fps = int(self.p['fps'])
        init.depth_mode = getattr(sl.DEPTH_MODE, self.p['depth_mode'])
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        init.depth_minimum_distance = float(self.p['depth_min_m'])
        init.depth_maximum_distance = float(self.p['depth_max_m'])
        if int(self.p['serial']):
            init.set_from_serial_number(int(self.p['serial']))
        self.cam = sl.Camera()
        err = self.cam.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise SystemExit(f'ZED open failed: {err}')
        info = self.cam.get_camera_information()
        cal = info.camera_configuration.calibration_parameters.left_cam
        res = info.camera_configuration.resolution
        self.get_logger().info(f'ZED {info.camera_model} S/N {info.serial_number}: {res.width}x{res.height} @ {self.p["fps"]} fps, '
                               f'depth {self.p["depth_mode"]}, fx {cal.fx:.1f} fy {cal.fy:.1f} cx {cal.cx:.1f} cy {cal.cy:.1f}')
        self.info = CameraInfo()
        self.info.header.frame_id = self.p['camera_frame']
        self.info.width, self.info.height = int(res.width), int(res.height)
        self.info.distortion_model = 'plumb_bob'
        self.info.d = [0.0] * 5                      # rectified images: no distortion left
        self.info.k = [cal.fx, 0.0, cal.cx, 0.0, cal.fy, cal.cy, 0.0, 0.0, 1.0]
        self.info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.info.p = [cal.fx, 0.0, cal.cx, 0.0, 0.0, cal.fy, cal.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.pub_depth = self.create_publisher(Image, self.p['depth_topic'], qos)
        self.pub_info = self.create_publisher(CameraInfo, self.p['info_topic'], qos)
        self.pub_rgb = self.create_publisher(Image, self.p['rgb_topic'], qos) if self.p['publish_rgb'] else None
        if self.p['publish_tf']:
            self.tf_pub = StaticTransformBroadcaster(self)
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.p['world_frame']
            t.child_frame_id = self.p['camera_frame']
            x, y, z = (float(v) for v in self.p['mount_xyz'])
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = x, y, z
            qx, qy, qz, qw = _quat_from_rpy(*(float(v) for v in self.p['mount_rpy']))
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = qx, qy, qz, qw
            self.tf_pub.sendTransform(t)
            self.get_logger().info(f'static TF {self.p["world_frame"]} -> {self.p["camera_frame"]} at {self.p["mount_xyz"]} rpy {self.p["mount_rpy"]}')
        self.depth_mat = sl.Mat()
        self.rgb_mat = sl.Mat()
        self.runtime = sl.RuntimeParameters()
        self.create_timer(1.0 / float(self.p['rate_hz']), self._tick)

    def _tick(self):
        if self.cam.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            self.get_logger().warn('grab failed', throttle_duration_sec=5.0)
            return
        stamp = self.get_clock().now().to_msg()
        self.cam.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)
        d = np.ascontiguousarray(self.depth_mat.get_data(), dtype=np.float32)   # NaN / inf where invalid
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.p['camera_frame']
        msg.height, msg.width = d.shape[0], d.shape[1]
        msg.encoding = '32FC1'
        msg.is_bigendian = 0
        msg.step = msg.width * 4
        msg.data = d.tobytes()
        self.info.header.stamp = stamp
        self.pub_info.publish(self.info)
        self.pub_depth.publish(msg)
        if self.pub_rgb is not None:
            self.cam.retrieve_image(self.rgb_mat, sl.VIEW.LEFT)
            bgra = self.rgb_mat.get_data()
            rgb = np.ascontiguousarray(bgra[:, :, 2::-1])
            m = Image()
            m.header.stamp = stamp
            m.header.frame_id = self.p['camera_frame']
            m.height, m.width = rgb.shape[0], rgb.shape[1]
            m.encoding = 'rgb8'
            m.step = m.width * 3
            m.data = rgb.tobytes()
            self.pub_rgb.publish(m)


def main():
    rclpy.init()
    node = ZedSource()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cam.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
