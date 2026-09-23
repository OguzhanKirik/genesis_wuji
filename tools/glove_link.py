#!/usr/bin/env python3
"""The Manus glove -> Isaac Lab wire protocol, and both ends of it.

Why UDP and not ROS. The teleop stack is ROS 2 Humble on Python 3.10; the Isaac
Lab container is Python 3.12 with no rclpy, no zmq and no pandas, and putting
ROS into it would mean fighting two Python versions and a large dependency tree
for the sake of one 20-float message. Both containers already run with
`network_mode: host`, so a UDP datagram on localhost crosses between them with
nothing but the standard library. adam-moveit reaches the same conclusion for
the same reason and passes its Isaac requests through a file.

Datagram, little-endian:

    magic    4s   b"WUJI"
    version  B    PROTOCOL_VERSION
    flags    B    bit0 engaged, bit1 reset requested, bit2 mark success
    n        H    number of joints that follow (20)
    t_glove  d    sender's monotonic clock, seconds
    seq      I    incrementing, for loss and ordering
    q        n*f  joint targets in RADIANS, in HAND_JOINTS order

Radians rather than a normalised action so the packet is self-describing and a
wrong joint limit on one side cannot silently rescale the other's motion. The
receiver converts.

Latest-wins: the receiver drains the socket every tick and keeps the newest
packet, so a slow consumer falls behind in freshness, never in a queue.

    python3 tools/glove_link.py --listen              # print what arrives
    python3 tools/glove_link.py --fake close          # send a scripted close
"""

from __future__ import annotations

import argparse
import socket
import struct
import time

PROTOCOL_VERSION = 1
DEFAULT_ADDR = ("127.0.0.1", 47921)
MAGIC = b"WUJI"
_HEADER = "<4sBBHdI"
_HEADER_SIZE = struct.calcsize(_HEADER)

FLAG_ENGAGED = 1 << 0        # the operator is driving; otherwise hold the pre-grasp
FLAG_RESET = 1 << 1          # start a new episode
FLAG_MARK_SUCCESS = 1 << 2   # operator asserts this attempt was good

# Must match adam_u_rl/envs/wuji_teleop_grasp_env_cfg.HAND_JOINTS. Duplicated
# rather than imported because the sender runs in the ROS container, which
# cannot import Isaac Lab. verify_joint_order() checks the two agree.
HAND_JOINTS = [
    "r_thumb_cmc_flex", "r_thumb_cmc_abd", "r_thumb_mcp", "r_thumb_ip",
    "r_index_finger_mcp_flex", "r_index_finger_mcp_abd", "r_index_finger_pip", "r_index_finger_dip",
    "r_middle_finger_mcp_flex", "r_middle_finger_mcp_abd", "r_middle_finger_pip", "r_middle_finger_dip",
    "r_ring_finger_mcp_flex", "r_ring_finger_mcp_abd", "r_ring_finger_pip", "r_ring_finger_dip",
    "r_pinky_mcp_flex", "r_pinky_mcp_abd", "r_pinky_pip", "r_pinky_dip",
]
N_JOINTS = len(HAND_JOINTS)


def verify_joint_order(other: list[str]) -> None:
    """Fail loudly if the env's joint order and the wire order have diverged.

    A silent mismatch here drives the wrong finger from the wrong glove channel,
    which looks like a badly calibrated glove rather than a bug.
    """
    if list(other) != HAND_JOINTS:
        raise ValueError(
            "glove wire joint order does not match the environment's.\n"
            f"  wire: {HAND_JOINTS}\n  env : {list(other)}")


def pack(q, seq: int, flags: int = FLAG_ENGAGED, t_glove: float | None = None) -> bytes:
    if len(q) != N_JOINTS:
        raise ValueError(f"expected {N_JOINTS} joint targets, got {len(q)}")
    head = struct.pack(_HEADER, MAGIC, PROTOCOL_VERSION, flags & 0xFF, N_JOINTS,
                       time.monotonic() if t_glove is None else t_glove, seq & 0xFFFFFFFF)
    return head + struct.pack(f"<{N_JOINTS}f", *[float(v) for v in q])


def unpack(data: bytes) -> dict | None:
    """Returns None for anything that is not a well-formed packet of our version."""
    if len(data) < _HEADER_SIZE:
        return None
    magic, ver, flags, n, t_glove, seq = struct.unpack(_HEADER, data[:_HEADER_SIZE])
    if magic != MAGIC or ver != PROTOCOL_VERSION or n != N_JOINTS:
        return None
    if len(data) < _HEADER_SIZE + 4 * n:
        return None
    q = struct.unpack(f"<{n}f", data[_HEADER_SIZE:_HEADER_SIZE + 4 * n])
    return dict(q=q, seq=seq, t_glove=t_glove, flags=flags,
                engaged=bool(flags & FLAG_ENGAGED),
                reset=bool(flags & FLAG_RESET),
                mark_success=bool(flags & FLAG_MARK_SUCCESS))


class GloveReceiver:
    """Non-blocking latest-wins receiver.

    `poll()` drains everything queued and returns the newest valid packet, or
    None if nothing arrived. `stale` says whether the last packet is older than
    `timeout`, which is what the recorder uses to fall back to holding the
    pre-grasp pose instead of freezing on a stale command.
    """

    def __init__(self, addr=DEFAULT_ADDR, timeout: float = 0.25):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(addr)
        self.sock.setblocking(False)
        self.timeout = timeout
        self.last: dict | None = None
        self.last_t: float = -1e9
        self.received = 0
        self.dropped = 0          # gaps in the sequence number
        self.senders_seen = 1     # incremented when the sequence restarts
        self._prev_seq: int | None = None

    def poll(self) -> dict | None:
        newest = None
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            except OSError:
                break
            pkt = unpack(data)
            if pkt is not None:
                newest = pkt
                self.received += 1
                if self._prev_seq is not None:
                    gap = (pkt["seq"] - self._prev_seq) & 0xFFFFFFFF
                    if 1 < gap < 100:
                        self.dropped += gap - 1
                    elif gap == 0 or gap >= 100:
                        # A restarted or second sender resets the sequence; that
                        # is not packet loss, and counting it as such once
                        # reported "43078 dropped" out of 646 received.
                        self.senders_seen += 1
                self._prev_seq = pkt["seq"]
        if newest is not None:
            self.last, self.last_t = newest, time.monotonic()
        return newest

    @property
    def stale(self) -> bool:
        return (time.monotonic() - self.last_t) > self.timeout

    def close(self):
        self.sock.close()


class GloveSender:
    def __init__(self, addr=DEFAULT_ADDR):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = addr
        self.seq = 0

    def send(self, q, flags: int = FLAG_ENGAGED, t_glove: float | None = None) -> None:
        self.sock.sendto(pack(q, self.seq, flags, t_glove), self.addr)
        self.seq += 1

    def close(self):
        self.sock.close()


# ---------------------------------------------------------------------------
# Standalone: listen, or send a scripted hand so the pipeline can be tested
# before any hardware is connected.
# ---------------------------------------------------------------------------

# The pre-grasp pose, duplicated from the env for the same reason as the joint
# order. tools/record_demo.py checks it against the env's on startup.
PREGRASP_Q = [
    0.5164, -0.85, 0.628, 0.628,          # thumb: cmc_flex, cmc_abd, mcp, ip
    0.628, 0.0, 0.8376, 0.628,            # index
    0.628, 0.0, 0.8376, 0.628,            # middle
    0.0, 0.0, 0.0, 0.0,                   # ring   (held flat)
    0.0, 0.0, 0.0, 0.0,                   # little (held flat)
]
# The CLOSED end of Experiment 2's synergy: 60 % of each flexion limit, not
# 100 %. That is the grasp measured to hold the bolt (16/16 scripted successes).
# Driving to the hard limits instead curls the fingers straight through the
# bolt -- which is exactly what happened the first time this was tested, and is
# the difference between the 20-DoF action space and Experiment 2's narrow
# synergy band. A human's glove produces a natural grasp, not the limits.
CLOSED_Q = [
    0.7746, -0.85, 0.9420, 0.9420,        # thumb
    0.9420, 0.0, 1.2564, 0.9420,          # index
    0.9420, 0.0, 1.2564, 0.9420,          # middle
    0.0, 0.0, 0.0, 0.0,                   # ring   (held flat)
    0.0, 0.0, 0.0, 0.0,                   # little (held flat)
]


def _fake(kind: str, rate: float, seconds: float, addr, period: float) -> int:
    """A stand-in for the glove, so the whole path can be tested with no hardware.

    `close` repeats an open -> closed ramp every `period` seconds, which should
    be the episode length: the recorder's episodes free-run, so a single ramp
    spread over a long send window leaves most episodes barely closed.
    """
    tx = GloveSender(addr)
    n = int(seconds * rate)
    print(f"[fake-glove] sending {kind} to {addr[0]}:{addr[1]} at {rate:g} Hz for {seconds:g} s"
          + (f", repeating every {period:g} s" if kind == "close" else ""))
    t0 = time.monotonic()
    try:
        for _ in range(n):
            if kind == "open":
                q = PREGRASP_Q
            elif kind == "close":
                phase = ((time.monotonic() - t0) % period) / period
                frac = min(1.0, phase / 0.2)        # close over the first 20 % of the period
                q = [a + frac * (b - a) for a, b in zip(PREGRASP_Q, CLOSED_Q)]
            else:
                raise SystemExit(f"unknown --fake {kind!r}; use open or close")
            tx.send(q)
            time.sleep(1.0 / rate)
    finally:
        tx.close()
    print(f"[fake-glove] sent {tx.seq} packets")
    return 0


def _listen(addr, seconds: float) -> int:
    rx = GloveReceiver(addr)
    print(f"[listen] bound to {addr[0]}:{addr[1]}, {seconds:g} s")
    end = time.monotonic() + seconds
    last_print = 0.0
    try:
        while time.monotonic() < end:
            pkt = rx.poll()
            now = time.monotonic()
            if pkt and now - last_print > 0.5:
                last_print = now
                q = " ".join(f"{v:+.2f}" for v in pkt["q"][:8])
                print(f"  seq={pkt['seq']:6d} engaged={pkt['engaged']} "
                      f"q[0:8]={q} ...  received={rx.received} dropped={rx.dropped}")
            time.sleep(0.005)
    finally:
        rx.close()
    print(f"[listen] {rx.received} packets, {rx.dropped} dropped, {rx.senders_seen} sender(s)")
    return 0 if rx.received else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", action="store_true", help="print arriving packets")
    ap.add_argument("--fake", choices=("open", "close"), help="send a scripted hand instead")
    ap.add_argument("--host", default=DEFAULT_ADDR[0])
    ap.add_argument("--port", type=int, default=DEFAULT_ADDR[1])
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--period", type=float, default=4.0,
                    help="--fake close: repeat the ramp every this many seconds (the episode length)")
    args = ap.parse_args()
    addr = (args.host, args.port)
    if args.fake:
        return _fake(args.fake, args.rate, args.seconds, addr, args.period)
    if args.listen:
        return _listen(addr, args.seconds)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
