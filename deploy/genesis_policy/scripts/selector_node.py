#!/usr/bin/env python3
"""The rod selector node: every rod perception sees in, the one the policy should pick out.

    ros2 run genesis_policy selector_node.py --ros-args -p level_path:=<handoff>/level.yaml

Subscribes `candidates_topic` (geometry_msgs/PoseArray in `world_frame`: every visible rod, its axis the
orientation's z). On `~/select` (std_srvs/Trigger), or when no rod is chosen yet and `auto_select` is on, it runs
genesis_policy.selector (the highest exposed reachable rod; adam-rl level 52 trains on the same choice) and from then
on publishes that rod's latest pose on `bolt_pose_topic` (geometry_msgs/PoseStamped, what policy_node reads),
following it from frame to frame as the nearest candidate within `track_radius_m`. The choice is kept until the next
`~/select`: the policy's episode needs one target. Nothing is chosen while the heap moves (every rod within
`settle_tol_m` of where it was `settle_window_s` ago). If the chosen rod is lost, publishing stops (the policy holds on
a stale pose), `~/choice` says so, and nothing is chosen automatically again until `~/select`. `~/choice` (std_msgs/String, JSON): the pick, its kind, height, a reachable hand
yaw, and per candidate why it was passed over.
"""
import collections
import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from genesis_policy import level as level_mod
from genesis_policy import selector as S


class SelectorNode(Node):
    def __init__(self):
        super().__init__('rod_selector')
        p = {
            'level_path': '',
            'candidates_topic': '/perception/rod_candidates',
            'bolt_pose_topic': '/perception/bolt_pose',
            'world_frame': 'world',
            'track_radius_m': 0.02,
            'auto_select': True,
            'settle_window_s': 1.0,
            'settle_tol_m': 0.015,          # above the camera noise (2 mm per axis, two frames, 60 rods)
        }
        for k, v in p.items():
            self.declare_parameter(k, v)
        p = {k: self.get_parameter(k).value for k in p}
        if not p['level_path']:
            raise SystemExit('selector_node: set level_path to the policy handoff level.yaml')
        self.ws = S.Workspace.from_level(level_mod.load(p['level_path']))
        self.world_frame = p['world_frame']
        self.track_radius = float(p['track_radius_m'])
        self.auto = bool(p['auto_select'])
        self.settle_window = float(p['settle_window_s'])
        self.settle_tol = float(p['settle_tol_m'])
        self.history = collections.deque()     # (receive time, positions) over the settle window
        self.lost = False                      # the chosen rod vanished: no automatic re-select until ~/select
        self.latest = None                     # (stamp, positions (N, 3), quats (N, 4))
        self.chosen = None                     # the chosen rod's last position (3,)
        self.pub = self.create_publisher(PoseStamped, p['bolt_pose_topic'], 10)
        self.diag = self.create_publisher(String, '~/choice', 10)
        self.create_subscription(PoseArray, p['candidates_topic'], self._on_candidates, 10)
        self.create_service(Trigger, '~/select', self._on_select)
        self.get_logger().info(f'selector: {p["candidates_topic"]} -> {p["bolt_pose_topic"]}, '
                               f'workspace from {p["level_path"]}')

    def _report(self, payload):
        self.diag.publish(String(data=json.dumps(payload)))

    def settled(self):
        """Every rod within settle_tol_m of a rod in the frame settle_window_s ago, and as many rods: the heap is still
        (rods falling after a drop, or a pile sliding, would hand the policy a target that is about to move)."""
        if not self.history or self.history[-1][0] - self.history[0][0] < 0.8 * self.settle_window:
            return False
        old, new = self.history[0][1], self.history[-1][1]
        if len(old) != len(new) or not len(new):
            return False
        d = np.linalg.norm(new[:, None, :] - old[None, :, :], axis=2).min(axis=1)
        return bool(d.max() <= self.settle_tol)

    def _select(self):
        if self.latest is None:
            return False, 'no candidates received yet'
        if not self.settled():
            return False, f'the heap is still moving (needs {self.settle_window:.1f} s within {1000 * self.settle_tol:.0f} mm)'
        _, pos, quat = self.latest
        c = S.select(pos, quat, self.ws)
        if c.index < 0:
            self.chosen = None
            self._report(dict(picked=None, reasons=list(c.reasons)))
            return False, f'no pickable rod among {len(pos)}' + (f': {", ".join(sorted(set(c.reasons)))}' if c.reasons else '')
        self.chosen = pos[c.index].copy()
        self.lost = False
        self._report(dict(picked=int(c.index), kind=S.KIND_NAMES[c.kind], height_m=round(c.height, 4),
                          hand_yaw_rad=round(c.hand_yaw, 4), reasons=list(c.reasons)))
        return True, f'rod {c.index} of {len(pos)}: {S.KIND_NAMES[c.kind]}, {1000 * c.height:.0f} mm up'

    def _on_select(self, _req, res):
        res.success, res.message = self._select()
        self.get_logger().info(f'select: {res.message}')
        return res

    def _on_candidates(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.world_frame:
            self.get_logger().error(f'candidates in {msg.header.frame_id!r} ignored: expected {self.world_frame!r}',
                                    throttle_duration_sec=5.0)
            return
        pos = np.array([[q.position.x, q.position.y, q.position.z] for q in msg.poses]).reshape(-1, 3)
        quat = np.array([[q.orientation.x, q.orientation.y, q.orientation.z, q.orientation.w] for q in msg.poses]).reshape(-1, 4)
        self.latest = (msg.header.stamp, pos, quat)
        now = time.monotonic()
        self.history.append((now, pos))
        while self.history and now - self.history[0][0] > self.settle_window:
            self.history.popleft()
        if self.chosen is None:
            if self.auto and not self.lost and len(pos) and self.settled():
                ok, why = self._select()
                self.get_logger().info(f'auto select: {why}')
            if self.chosen is None:
                return
        # follow the chosen rod: the nearest candidate, if it is close enough to be the same rod
        d = np.linalg.norm(pos - self.chosen, axis=1) if len(pos) else np.array([])
        if not len(d) or d.min() > self.track_radius:
            why = f'nearest candidate {1000 * d.min():.0f} mm away' if len(d) else 'no candidates'
            self.get_logger().warn(f'chosen rod lost ({why}); call ~/select to choose again')
            self._report(dict(picked=None, lost=why))
            self.chosen, self.lost = None, True
            return
        i = int(d.argmin())
        self.chosen = pos[i].copy()
        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.world_frame
        out.pose = msg.poses[i]
        self.pub.publish(out)


def main():
    rclpy.init()
    node = SelectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
