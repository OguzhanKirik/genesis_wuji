#!/usr/bin/env python3
"""The sim accuracy gate: this node's candidates against Isaac's true rod poses.

Run genesis_isaac's rod_candidates.py with zero noise on a second topic (the truth), and this perception node on
the head camera; this script matches each estimate to the nearest truth and prints position and axis error
statistics over `--seconds`. Pass when the medians sit inside the policy's perception budget (5 mm, 3 deg) and
the recall of exposed rods is high.

    ros2 run genesis_isaac rod_candidates.py --ros-args -p topic:=/perception/rod_truth -p noise_xy_m:=0.0 \
        -p noise_z_m:=0.0 -p noise_rot_deg:=0.0
    ros2 run genesis_perception compare_candidates --seconds 20
"""
import argparse
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node


def _axes(poses):
    out = []
    for p in poses:
        x, y, z, w = p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
        out.append([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])   # R[:, 2]
    return np.asarray(out, float).reshape(-1, 3)


def _pos(poses):
    return np.asarray([[p.position.x, p.position.y, p.position.z] for p in poses], float).reshape(-1, 3)


class Compare(Node):
    def __init__(self, est_topic, truth_topic, match_m):
        super().__init__('compare_candidates')
        self.truth = None
        self.match_m = match_m
        self.pos_err, self.ax_err, self.n_est, self.n_truth, self.frames = [], [], 0, 0, 0
        self.create_subscription(PoseArray, truth_topic, self._truth, 10)
        self.create_subscription(PoseArray, est_topic, self._est, 10)

    def _truth(self, msg):
        self.truth = msg

    def _est(self, msg):
        if self.truth is None:
            return
        tp, ta = _pos(self.truth.poses), _axes(self.truth.poses)
        ep, ea = _pos(msg.poses), _axes(msg.poses)
        self.frames += 1
        self.n_truth += len(tp)
        self.n_est += len(ep)
        for i in range(len(ep)):
            if len(tp) == 0:
                break
            d = np.linalg.norm(tp - ep[i], axis=1)
            j = int(np.argmin(d))
            if d[j] <= self.match_m:
                self.pos_err.append(d[j])
                c = abs(float(np.clip(np.dot(ta[j], ea[i]), -1, 1)))   # end-to-end symmetric: sign-free
                self.ax_err.append(math.degrees(math.acos(c)))

    def report(self):
        if not self.pos_err:
            print(f'no matches in {self.frames} frames (estimates {self.n_est}, truth rods {self.n_truth})')
            return
        pe, ae = 1000 * np.asarray(self.pos_err), np.asarray(self.ax_err)
        print(f'frames {self.frames}  estimates {self.n_est}  truth rods {self.n_truth}  matched {len(pe)} '
              f'({100 * len(pe) / max(self.n_est, 1):.0f} % of estimates)')
        print(f'position error mm: median {np.median(pe):.1f}  p90 {np.percentile(pe, 90):.1f}  max {pe.max():.1f}')
        print(f'axis error deg:    median {np.median(ae):.2f}  p90 {np.percentile(ae, 90):.2f}  max {ae.max():.2f}')
        print('budget: 5 mm / 3 deg bias (adam-rl docs/SIM_TO_REAL.md, _PERC7)')


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--est', default='/perception/rod_candidates')
    ap.add_argument('--truth', default='/perception/rod_truth')
    ap.add_argument('--seconds', type=float, default=20.0)
    ap.add_argument('--match-m', type=float, default=0.02)
    opts, ros_args = ap.parse_known_args(args)
    rclpy.init(args=ros_args)
    node = Compare(opts.est, opts.truth, opts.match_m)
    end = node.get_clock().now().nanoseconds + int(opts.seconds * 1e9)
    try:
        while rclpy.ok() and node.get_clock().now().nanoseconds < end:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.report()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
