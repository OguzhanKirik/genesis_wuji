#!/usr/bin/env python3
"""Record teleoperated grasp demonstrations in the Experiment 3 environment.

Drives the 20-DoF Wuji hand from glove packets arriving over UDP (see
tools/glove_link.py) and writes one file per episode. The episode is exactly
Experiment 2's: 2 s of operator-controlled grasping, then a scripted 10 cm lift
and a 1 s hold, so every demonstration is labelled success or failure by the
same test the policy is later evaluated with.

The environment knows nothing about UDP. This script reads the socket, converts
radians to the action the env expects, and steps it -- so the same env is used
unchanged for teleoperation, behaviour cloning and PPO.

    # inside the Isaac Lab container
    python /opt/tools/record_demo.py --viz kit --episodes 5

    # elsewhere, with the real glove:  the manus_ws bridge node
    # or, to test the whole path with no hardware:
    python3 tools/glove_link.py --fake close --rate 30 --seconds 60

If no packet has arrived for --timeout seconds, or the operator is not engaged,
the hand holds the pre-grasp pose. That is a deliberate choice: freezing on the
last stale command would quietly record a grasp the operator was not making.

Output, one NPZ + one CSV per episode, under --out:
  t, joint_pos, joint_vel, action, joint_target, tip_force, contact,
  bolt_pose, phase, plus scalars success / episode_id and a meta block with the
  joint order, the env constants and the git SHA.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import uuid

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", default="/workspace/demos", help="directory for the trajectories")
parser.add_argument("--episodes", type=int, default=1, help="stop after this many episodes")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=47921)
parser.add_argument("--timeout", type=float, default=0.25,
                    help="seconds without a packet before falling back to the pre-grasp pose")
parser.add_argument("--keep-failures", action="store_true",
                    help="write unsuccessful episodes too (they are labelled either way)")
parser.add_argument("--real-time", action="store_true", default=True,
                    help="pace to wall clock; teleoperation is meaningless without it")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glove_link as G  # noqa: E402

from adam_u_rl.envs import wuji_teleop_grasp_env_cfg as C  # noqa: E402
from adam_u_rl.envs.adam_u_wuji_grasp_env_cfg import _t  # noqa: E402

# The wire order and the env's joint order must be the same list, or the glove
# drives the wrong fingers and it looks like bad calibration.
G.verify_joint_order(C.HAND_JOINTS)

cfg = C.WujiTeleopGraspEnvCfg()
cfg.scene.num_envs = 1
cfg.sim.device = "cuda:0"
gym.register(id="TeleopRecord", entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": C.WujiTeleopGraspEnvCfg})
env = gym.make("TeleopRecord", cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
bolt = u.scene["object"]
dev = u.device

ids, names = robot.find_joints(C.HAND_JOINTS, preserve_order=True)
assert list(names) == C.HAND_JOINTS, names
limits = _t(robot.data.joint_pos_limits)[0, ids]          # (20, 2)
lower, upper = limits[:, 0], limits[:, 1]

pregrasp_q = torch.tensor([C.PREGRASP_Q[j] for j in C.HAND_JOINTS], device=dev)
pregrasp_a = C.action_from_angles(pregrasp_q, lower, upper)
# The sender's copy of the pre-grasp pose must agree, or an idle glove would
# command a different hand shape than the env resets to.
wire_pregrasp = torch.tensor(G.PREGRASP_Q, device=dev)
if float((wire_pregrasp - pregrasp_q).abs().max()) > 1e-3:
    print(f"[record] WARNING: glove_link.PREGRASP_Q differs from the env's by up to "
          f"{float((wire_pregrasp - pregrasp_q).abs().max()):.4f} rad", file=sys.stderr)

os.makedirs(args.out, exist_ok=True)
rx = G.GloveReceiver((args.host, args.port), timeout=args.timeout)
print(f"[record] listening on {args.host}:{args.port}, writing to {args.out}")
print(f"[record] episode = {C.GRASP_STEPS} operator steps, then a scripted "
      f"{C.LIFT_STEPS}-step lift and {C.HOLD_STEPS}-step hold; "
      f"{u.max_episode_length} steps total at {1.0 / float(u.step_dt):.0f} Hz")


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", "/workspace", "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


META = dict(
    joint_names=C.HAND_JOINTS,
    joint_lower=[float(v) for v in lower.tolist()],
    joint_upper=[float(v) for v in upper.tolist()],
    pregrasp_q=[float(v) for v in pregrasp_q.tolist()],
    control_hz=1.0 / float(u.step_dt),
    episode_steps=int(u.max_episode_length),
    protocol_version=G.PROTOCOL_VERSION,
    git_sha=git_sha(),
    recorded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
)


class Trajectory:
    """Accumulates one episode, then writes NPZ + CSV."""

    def __init__(self):
        self.rows = []
        self.id = uuid.uuid4().hex[:12]
        self.engaged_steps = 0
        self.stale_steps = 0

    def add(self, t, a, q_target, engaged, stale):
        jp = _t(robot.data.joint_pos)[0, ids]
        jv = _t(robot.data.joint_vel)[0, ids]
        b = C._update(u)
        self.rows.append(dict(
            t=t,
            joint_pos=jp.cpu().numpy().copy(),
            joint_vel=jv.cpu().numpy().copy(),
            action=a.cpu().numpy().copy(),
            joint_target=q_target.cpu().numpy().copy(),
            tip_force=b["tip_force"][0].cpu().numpy().copy(),
            contact=(b["streak"][0] > 0).cpu().numpy().copy(),
            bolt_pose=(_t(bolt.data.root_pose_w)[0] - torch.cat(
                [u.scene.env_origins[0], torch.zeros(4, device=dev)])).cpu().numpy().copy(),
            phase=int(C.phase_of(u)[0]),
        ))
        self.engaged_steps += int(engaged)
        self.stale_steps += int(stale)

    def write(self, out_dir, success: bool) -> str:
        n = len(self.rows)
        arrays = {k: np.stack([r[k] for r in self.rows]) for k in
                  ("joint_pos", "joint_vel", "action", "joint_target", "tip_force", "contact", "bolt_pose")}
        arrays["t"] = np.array([r["t"] for r in self.rows], dtype=np.float32)
        arrays["phase"] = np.array([r["phase"] for r in self.rows], dtype=np.int8)
        meta = dict(META, episode_id=self.id, success=bool(success), steps=n,
                    engaged_steps=self.engaged_steps, stale_steps=self.stale_steps)
        base = os.path.join(out_dir, f"demo_{self.id}")
        np.savez_compressed(f"{base}.npz", meta=json.dumps(meta), success=np.array(success), **arrays)
        with open(f"{base}.csv", "w", newline="") as f:
            w = csv.writer(f)
            cols = (["t", "phase"]
                    + [f"q_{j}" for j in C.HAND_JOINTS]
                    + [f"a_{j}" for j in C.HAND_JOINTS]
                    + [f"tgt_{j}" for j in C.HAND_JOINTS]
                    + ["f_thumb", "f_index", "f_middle", "c_thumb", "c_index", "c_middle"])
            w.writerow(cols)
            for i in range(n):
                w.writerow([f"{arrays['t'][i]:.4f}", int(arrays["phase"][i])]
                           + [f"{v:.5f}" for v in arrays["joint_pos"][i]]
                           + [f"{v:.5f}" for v in arrays["action"][i]]
                           + [f"{v:.5f}" for v in arrays["joint_target"][i]]
                           + [f"{v:.4f}" for v in arrays["tip_force"][i]]
                           + [int(v) for v in arrays["contact"][i]])
        return base


def main() -> int:
    written = kept = 0
    obs, _ = env.reset()
    traj = Trajectory()
    t0 = time.monotonic()
    dt = float(u.step_dt)
    action = pregrasp_a.clone().unsqueeze(0)

    while written < args.episodes and app.is_running():
        step_start = time.monotonic()
        pkt = rx.poll()
        stale = rx.stale
        engaged = bool(pkt and pkt["engaged"]) and not stale

        if engaged:
            q = torch.tensor(pkt["q"], device=dev, dtype=torch.float32)
            action = C.action_from_angles(q, lower, upper).unsqueeze(0)
        elif stale or (pkt and not pkt["engaged"]):
            action = pregrasp_a.clone().unsqueeze(0)

        q_target = C.angles_from_action(action[0], lower, upper)
        traj.add(time.monotonic() - t0, action[0], q_target, engaged, stale)

        obs, rew, term, trunc, extras = env.step(action)

        if bool((term | trunc).any()):
            b = C._update(u)
            success = bool(b["last_success"][0])
            operator = traj.engaged_steps > 0
            if not operator:
                print(f"[record] episode {traj.id}: no operator input, discarded")
            elif success or args.keep_failures:
                base = traj.write(args.out, success)
                kept += 1
                print(f"[record] episode {traj.id}: {'SUCCESS' if success else 'failure'}, "
                      f"{len(traj.rows)} steps, {traj.engaged_steps} engaged, "
                      f"{traj.stale_steps} stale -> {os.path.basename(base)}.npz")
            else:
                print(f"[record] episode {traj.id}: failure, discarded "
                      f"(pass --keep-failures to write it)")
            written += 1
            traj = Trajectory()
            t0 = time.monotonic()
            action = pregrasp_a.clone().unsqueeze(0)

        if args.real_time:
            left = dt - (time.monotonic() - step_start)
            if left > 0:
                time.sleep(left)

    print(f"[record] {written} episodes seen, {kept} written to {args.out}")
    print(f"[record] glove packets: {rx.received} received, {rx.dropped} dropped, "
          f"{rx.senders_seen} sender(s)")
    rx.close()
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
