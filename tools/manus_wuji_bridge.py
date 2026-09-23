#!/usr/bin/env python3
"""Manus glove -> Wuji hand -> Isaac Lab. The teleop bridge for Experiment 3.

Subscribes to the raw glove topic, maps the 20 ergonomics angles onto the 20
Wuji joints, and sends them to the recorder over UDP (tools/glove_link.py).

Why a new node rather than manus_teleop's existing retarget. That one collapses
the glove to 6 scalars -- four finger flexions plus thumb flex and rotation --
discards all abduction, PIP and DIP, and emits *PND* hand joint names, which do
not intersect the Wuji joint list at all. Experiment 3 wants all 20, and the
glove already provides exactly 20.

The correspondence is one to one, but note the thumb: Manus names the thumb's
three flexion joints MCP/PIP/DIP where Wuji names the same three CMC/MCP/IP.
Mapping by name alone silently misdrives the thumb, which is the finger this
grasp depends on most.

    # on the host (ROS 2 Jazzy, manus_ws built there)
    source /home/revel/manus_ws/install/setup.bash
    python3 tools/manus_wuji_bridge.py --side right

    # or inside the manus container, with this directory mounted
    docker compose run --rm -v /home/revel/adam-rl/tools:/opt/rl_tools:ro manus \\
        python3 /opt/rl_tools/manus_wuji_bridge.py --side right

    # no ROS, no glove: check the mapping and the packets
    python3 tools/manus_wuji_bridge.py --self-test

Calibration. Each ergonomic is normalised from its own open/closed degrees onto
[0, 1] and then onto the joint's range. `manus_teleop/config/hand_retarget.yaml`
holds per-side values but only for the 6 channels that node uses, so the rest
fall back to DEFAULT_RANGE_DEG until you record them. Write a YAML of the form

    right:
      IndexMCPStretch: {open: -9.4, closed: 85.2}
      ...

and pass it with --calib.

Abduction is NOT teleoperated by default. The pinch geometry that makes this
grasp work was solved for a specific thumb opposition, and a miscalibrated
spread channel splays the fingers and breaks it. `--abduction` turns it on.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glove_link as G  # noqa: E402

# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

# Wuji joint -> Manus ergonomics type string. The thumb rows are the trap: the
# two naming schemes are offset by one, and they are the same three joints.
ERGO_FOR_JOINT = {
    "r_thumb_cmc_flex": "ThumbMCPStretch",
    "r_thumb_cmc_abd": "ThumbMCPSpread",
    "r_thumb_mcp": "ThumbPIPStretch",
    "r_thumb_ip": "ThumbDIPStretch",
}
for _f, _F in (("index_finger", "Index"), ("middle_finger", "Middle"),
               ("ring_finger", "Ring"), ("pinky", "Pinky")):
    ERGO_FOR_JOINT[f"r_{_f}_mcp_flex"] = f"{_F}MCPStretch"
    ERGO_FOR_JOINT[f"r_{_f}_mcp_abd"] = f"{_F}Spread"
    ERGO_FOR_JOINT[f"r_{_f}_pip"] = f"{_F}PIPStretch"
    ERGO_FOR_JOINT[f"r_{_f}_dip"] = f"{_F}DIPStretch"

assert set(ERGO_FOR_JOINT) == set(G.HAND_JOINTS), "mapping does not cover every Wuji joint"

# Joint limits, from assets/robots/adam_u_wuji/urdf.
JOINT_RANGE = {
    "r_thumb_cmc_flex": (-1.187, 1.291), "r_thumb_cmc_abd": (-1.484, 0.698),
    "r_thumb_mcp": (-1.047, 1.570), "r_thumb_ip": (-1.047, 1.570),
}
for _f in ("index_finger", "middle_finger", "ring_finger", "pinky"):
    JOINT_RANGE[f"r_{_f}_mcp_flex"] = (-1.047, 1.570)
    JOINT_RANGE[f"r_{_f}_mcp_abd"] = (-0.698, 0.698)
    JOINT_RANGE[f"r_{_f}_pip"] = (-1.047, 2.094)
    JOINT_RANGE[f"r_{_f}_dip"] = (-1.047, 1.570)

# Fallback calibration, degrees, until a real one is recorded. Manus reports
# stretch negative when flexed, so "open" is the larger number.
DEFAULT_RANGE_DEG = {"stretch": (0.0, 90.0), "spread": (-20.0, 20.0)}

# What each joint is driven to at f = 0 and f = 1. Flexion runs from flat to the
# joint's limit; abduction spans its whole range.
def _f_endpoints(joint: str) -> tuple[float, float]:
    lo, hi = JOINT_RANGE[joint]
    return (lo, hi) if joint.endswith("_abd") else (0.0, hi)


def retarget(ergo_deg: dict[str, float], calib: dict, use_abduction: bool,
             pregrasp: list[float]) -> list[float]:
    """20 ergonomics in degrees -> 20 Wuji joint targets in radians, in wire order."""
    out = []
    for i, joint in enumerate(G.HAND_JOINTS):
        name = ERGO_FOR_JOINT[joint]
        if joint.endswith("_abd") and not use_abduction:
            out.append(pregrasp[i])          # hold the solved pinch geometry
            continue
        deg = ergo_deg.get(name)
        if deg is None:
            out.append(pregrasp[i])          # channel missing this frame
            continue
        kind = "spread" if name.endswith("Spread") else "stretch"
        c = calib.get(name, {})
        o = float(c.get("open", DEFAULT_RANGE_DEG[kind][0]))
        cl = float(c.get("closed", DEFAULT_RANGE_DEG[kind][1]))
        f = 0.0 if abs(cl - o) < 1e-6 else (deg - o) / (cl - o)
        f = min(1.0, max(0.0, f))
        a, b = _f_endpoints(joint)
        out.append(a + f * (b - a))
    return out


MANUS_CALIB_YAML = "/home/revel/manus_ws/src/manus_teleop/config/hand_retarget.yaml"


def load_calib(path: str | None, side: str) -> dict:
    """Per-ergonomic open/closed degrees.

    Understands two formats. A plain one, written for this bridge:

        right:
          IndexMCPStretch: {open: -9.4, closed: 85.2}

    and manus_teleop's own `hand_retarget.yaml`, which is what is actually
    calibrated today. That file holds only the SIX channels its retarget uses --
    the four finger MCP stretches plus thumb flex and rotation -- so the eight
    PIP and DIP channels are derived from the same finger's measured MCP range.
    That is an assumption, not a measurement: the joints are on the same finger
    and the glove reports them in the same units and sign convention, so the
    range is a far better estimate than a generic default, but a proper
    20-channel calibration should replace it. The four finger Spread channels
    stay uncalibrated, which is harmless while --abduction is off.
    """
    if not path:
        return {}
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    if "/**" in data:                       # manus_teleop's format
        p = data["/**"]["ros__parameters"]
        out: dict[str, dict] = {}

        def put(ergo, o, c):
            if o is not None and c is not None:
                out[ergo] = {"open": float(o), "closed": float(c)}

        for finger, prefix in (("Index", "index"), ("Middle", "middle"),
                               ("Ring", "ring"), ("Pinky", "pinky")):
            o = p.get(f"{side}_{prefix}_open_deg")
            c = p.get(f"{side}_{prefix}_closed_deg")
            put(f"{finger}MCPStretch", o, c)
            put(f"{finger}PIPStretch", o, c)      # derived, see the docstring
            put(f"{finger}DIPStretch", o, c)      # derived
        to = p.get(f"{side}_thumb_flex_open_deg")
        tc = p.get(f"{side}_thumb_flex_closed_deg")
        put("ThumbMCPStretch", to, tc)
        put("ThumbPIPStretch", to, tc)            # derived
        put("ThumbDIPStretch", to, tc)            # derived
        put("ThumbMCPSpread", p.get(f"{side}_thumb_rot_open_deg"),
            p.get(f"{side}_thumb_rot_closed_deg"))
        return out

    return data.get(side, data)


# ---------------------------------------------------------------------------
# Self-test: exercise the mapping and the wire with no ROS and no glove
# ---------------------------------------------------------------------------


def self_test(use_abduction: bool, calib_path: str | None = None, side: str = "right") -> int:
    calib = load_calib(calib_path, side) if calib_path and os.path.exists(calib_path) else {}
    print(f"calibration: {calib_path if calib else 'none, using defaults'}"
          f"{f' -> {len(calib)} channels for the {side} glove' if calib else ''}")
    for n in sorted(calib):
        print(f"  {n:20s} open {calib[n]['open']:+7.1f} deg   closed {calib[n]['closed']:+7.1f} deg")
    print()
    print("mapping, Wuji joint <- Manus ergonomic:")
    for j in G.HAND_JOINTS:
        print(f"  {j:26s} <- {ERGO_FOR_JOINT[j]}")
    print(f"\n{len(ERGO_FOR_JOINT)} joints covered, "
          f"{len(set(ERGO_FOR_JOINT.values()))} distinct ergonomics channels")

    flat = {n: (0.0 if n.endswith("Spread") else 0.0) for n in ERGO_FOR_JOINT.values()}
    fist = {n: (0.0 if n.endswith("Spread") else 90.0) for n in ERGO_FOR_JOINT.values()}
    # Drive each channel to its own calibrated open and closed value.
    def at(frac):
        out = {}
        for n in ERGO_FOR_JOINT.values():
            kind = "spread" if n.endswith("Spread") else "stretch"
            c = calib.get(n, {})
            o = float(c.get("open", DEFAULT_RANGE_DEG[kind][0]))
            cl = float(c.get("closed", DEFAULT_RANGE_DEG[kind][1]))
            out[n] = o + frac * (cl - o)
        return out

    flat, fist = at(0.0), at(1.0)
    for label, ergo in (("hand open (each channel at its calibrated open)", flat),
                        ("hand closed (each channel at its calibrated closed)", fist)):
        q = retarget(ergo, calib, use_abduction, G.PREGRASP_Q)
        print(f"\n{label}:")
        for j, v in list(zip(G.HAND_JOINTS, q))[:8]:
            lo, hi = JOINT_RANGE[j]
            print(f"  {j:26s} {v:+.3f} rad   (limits {lo:+.2f}..{hi:+.2f})")
        print("  ...")
        bad = [j for j, v in zip(G.HAND_JOINTS, q)
               if not (JOINT_RANGE[j][0] - 1e-6 <= v <= JOINT_RANGE[j][1] + 1e-6)]
        print(f"  outside limits: {bad or 'none'}")
        pkt = G.unpack(G.pack(q, 0))
        assert pkt and max(abs(a - b) for a, b in zip(pkt["q"], q)) < 1e-5
    print("\npacks and unpacks cleanly; mapping covers all 20 joints and stays in limits")
    return 0


# ---------------------------------------------------------------------------
# The ROS node
# ---------------------------------------------------------------------------


def run_node(args) -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSPresetProfiles
    from manus_ros2_msgs.msg import ManusGlove

    calib = load_calib(args.calib, args.side)
    tx = G.GloveSender((args.host, args.port))
    pregrasp = list(G.PREGRASP_Q)

    class Bridge(Node):
        def __init__(self):
            super().__init__("manus_wuji_bridge")
            self.latest: dict[str, float] | None = None
            self.last_rx = 0.0
            self.sent = 0
            self.frames = 0
            for topic in args.topics:
                self.create_subscription(ManusGlove, topic, self.on_glove,
                                         QoSPresetProfiles.SENSOR_DATA.value)
            self.create_timer(1.0 / args.rate, self.on_tick)
            self.get_logger().info(
                f"bridging {args.side} glove from {args.topics} -> "
                f"udp {args.host}:{args.port} at {args.rate:g} Hz, "
                f"abduction {'ON' if args.abduction else 'held at the pre-grasp'}, "
                f"{len(calib)} calibrated channels")

        def on_glove(self, msg):
            if msg.side.lower() != args.side:
                return
            self.latest = {e.type: float(e.value) for e in msg.ergonomics}
            self.last_rx = time.monotonic()
            self.frames += 1

        def on_tick(self):
            if self.latest is None:
                return
            fresh = (time.monotonic() - self.last_rx) < args.stale
            q = retarget(self.latest, calib, args.abduction, pregrasp) if fresh else pregrasp
            flags = G.FLAG_ENGAGED if (fresh and args.engaged) else 0
            if args.publish:
                tx.send(q, flags)
            self.sent += 1
            if self.sent % int(args.rate * 5) == 0:
                self.get_logger().info(
                    f"{self.frames} glove frames, {self.sent} packets, "
                    f"{'fresh' if fresh else 'STALE -> holding pre-grasp'}, "
                    f"thumb_cmc_flex={q[0]:+.2f} index_mcp_flex={q[4]:+.2f} rad")

    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        tx.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="check the mapping with no ROS and no glove")
    ap.add_argument("--side", default="right", choices=("right", "left"))
    ap.add_argument("--topics", nargs="+", default=["/manus_glove_0", "/manus_glove_1"])
    ap.add_argument("--calib", default=MANUS_CALIB_YAML,
                    help="calibration YAML; defaults to manus_teleop's own hand_retarget.yaml")
    ap.add_argument("--abduction", action="store_true",
                    help="teleoperate the spread joints too (off by default: it breaks the solved pinch)")
    ap.add_argument("--rate", type=float, default=30.0, help="send rate, matching the env's control rate")
    ap.add_argument("--stale", type=float, default=0.25, help="seconds before falling back to the pre-grasp")
    ap.add_argument("--engaged", action="store_true", default=True)
    ap.add_argument("--no-publish", dest="publish", action="store_false", default=True,
                    help="log instead of sending, to check the retarget safely")
    ap.add_argument("--host", default=G.DEFAULT_ADDR[0])
    ap.add_argument("--port", type=int, default=G.DEFAULT_ADDR[1])
    args = ap.parse_args()
    if args.self_test:
        return self_test(args.abduction, args.calib, args.side)
    return run_node(args)


if __name__ == "__main__":
    raise SystemExit(main())
