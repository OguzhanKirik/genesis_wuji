#!/usr/bin/env python3
"""Validate the reward terms by driving the env to scripted states.

Every term is checked where its behaviour matters, not just at spawn:

  * grasp      must pay ~0 for a closed hand FAR from the cube (the proximity
               gate), and clearly more for a closed hand AT the cube.
  * lift_bonus must be 0 at the cube's resting height and 1 once it is lifted
               past min_lift.
  * off_table  must be 0 at rest and rise as the cube falls.
  * reach_*    the coarse term must still carry gradient at spawn distance,
               which is what a single small std would have destroyed.

Also prints the reward budget so a change's effect on the total is visible.

    docker compose exec adam_rl python /opt/tools/probe_reward.py
"""

from __future__ import annotations

import math
import sys

sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from genesis_rl.envs.adam_u_grasp_env_cfg import AdamUGraspEnvCfg

cfg = AdamUGraspEnvCfg()
cfg.scene.num_envs = 2
cfg.sim.device = "cuda:0"

gym.register(id="Probe", entry_point="isaaclab.envs:ManagerBasedRLEnv",
             disable_env_checker=True, kwargs={"env_cfg_entry_point": AdamUGraspEnvCfg})
env = gym.make("Probe", cfg=cfg)
env.reset()
u = env.unwrapped
rm = u.reward_manager
am = u.action_manager
obj = u.scene["object"]
n_arm = am.get_term("right_arm").action_dim

# ep-norm factor: step_dt * episode_steps / episode_length_s, which is 1.0 here,
# so an episode-normalised value is just mean(raw) * weight.
EPN = u.step_dt * u.max_episode_length / u.cfg.episode_length_s
print("=" * 74)
print(f"STEP_DT {u.step_dt:.5f}  EPISODE_STEPS {u.max_episode_length}  EPN_FACTOR {EPN:.4f}")


def dropped_now():
    """Evaluate object_dropped directly.

    termination_manager.get_term() returns the value stored at the last step,
    and these checks teleport the cube without stepping, so that buffer is
    stale -- it reported True at every height, including 1.30 m.
    """
    c = u.cfg.terminations.object_dropped
    return bool(c.func(u, **c.params)[0].item())


def raw_terms():
    out = {}
    for name in rm.active_terms:
        c = rm.get_term_cfg(name)
        out[name] = (c.func(u, **c.params)[0].item(), c.weight)
    return out


def act(fingers=None, arm=0.0):
    a = torch.zeros((u.num_envs, am.total_action_dim), device=u.device)
    a[:, :n_arm] = arm
    if fingers is not None:
        a[:, n_arm:] = fingers
    return a


def settle(a, steps=40):
    for _ in range(steps):
        u.step(a)


def put_cube(z, dxy=(0.0, 0.0)):
    st = obj.data.default_root_state.clone()
    st[:, :3] += u.scene.env_origins
    st[:, 0] += dxy[0]
    st[:, 1] += dxy[1]
    st[:, 2] = z
    obj.write_root_pose_to_sim(st[:, :7])
    obj.write_root_velocity_to_sim(torch.zeros_like(st[:, 7:]))


def tip_dist():
    o = obj.data.root_pos_w[0]
    t = u.scene["right_thumb_frame"].data.target_pos_w[0, 0, :]
    p = u.scene["right_pinky_frame"].data.target_pos_w[0, 0, :]
    return ((torch.norm(o - t) + torch.norm(o - p)) / 2).item()


# --------------------------------------------------------------- grasp gate
print("\n" + "=" * 74)
print("grasp: proximity gate x closure. Must NOT pay for closing in free space.")
print("=" * 74)
print(f"{'state':34} {'tip dist m':>11} {'grasp epn':>11} {'reach_fine':>11}")
for label, fing in [("hand OPEN,   far from cube", -1.0),
                    ("hand CLOSED, far from cube", +1.0)]:
    env.reset()
    settle(act(fingers=fing))
    t = raw_terms()
    print(f"{label:34} {tip_dist():11.4f} "
          f"{t['grasp'][0]*t['grasp'][1]*EPN:11.4f} "
          f"{t['reach_fine'][0]*t['reach_fine'][1]*EPN:11.4f}")

# Bring the cube TO the hand instead of solving IK, then re-measure.
for label, fing in [("hand OPEN,   cube at fingertips", -1.0),
                    ("hand CLOSED, cube at fingertips", +1.0)]:
    env.reset()
    settle(act(fingers=fing), steps=40)
    thumb = u.scene["right_thumb_frame"].data.target_pos_w[0, 0, :]
    pinky = u.scene["right_pinky_frame"].data.target_pos_w[0, 0, :]
    mid = (thumb + pinky) / 2.0
    st = obj.data.default_root_state.clone()
    st[:, :3] = mid.unsqueeze(0).expand(u.num_envs, 3)
    obj.write_root_pose_to_sim(st[:, :7])
    obj.write_root_velocity_to_sim(torch.zeros_like(st[:, 7:]))
    # Deliberately NOT stepping: the cube is interpenetrating a closed fist and
    # one PhysX step launches it clear before it can be measured. gate*closure
    # is pure kinematics, so the teleported pose is the state to evaluate.
    obj.update(0.0)
    t = raw_terms()
    print(f"{label:34} {tip_dist():11.4f} "
          f"{t['grasp'][0]*t['grasp'][1]*EPN:11.4f} "
          f"{t['reach_fine'][0]*t['reach_fine'][1]*EPN:11.4f}")

# ----------------------------------------------------- lift / height / drop
print("\n" + "=" * 74)
print("cube height sweep: lift_bonus, object_height, off_table, termination")
print("=" * 74)
print(f"{'cube z':>8} {'lift_bonus':>12} {'object_height':>14} "
      f"{'off_table':>11} {'dropped?':>10}")
for z in [1.050, 1.060, 1.075, 1.100, 1.200, 1.250, 1.300, 1.020, 0.990, 0.940]:
    env.reset()
    put_cube(z)
    t = raw_terms()
    dropped = dropped_now()
    print(f"{z:8.3f} {t['lift_bonus'][0]*t['lift_bonus'][1]*EPN:12.4f} "
          f"{t['object_height'][0]*t['object_height'][1]*EPN:14.4f} "
          f"{t['off_table'][0]*t['off_table'][1]*EPN:11.4f} {str(dropped):>10}")

# ---------------------------------------------------------- reach at range
print("\n" + "=" * 74)
print("reach terms vs distance -- coarse must keep gradient at spawn range")
print("=" * 74)
print(f"{'d (m)':>8} {'coarse epn':>12} {'fine epn':>10} {'sum':>8} "
      f"{'d(sum)/dcm':>12} {'old std=1.0':>12}")
prev = None
for d in [0.30, 0.23, 0.15, 0.10, 0.077, 0.05, 0.03, 0.025, 0.01]:
    c = (1 - math.tanh(d / 0.30)) * 1.0
    f = (1 - math.tanh(d / 0.04)) * 2.0
    old = (1 - math.tanh(d / 1.0)) * 2.0
    slope = "" if prev is None else f"{(c+f-prev[1])/((d-prev[0])*100):+12.5f}"
    print(f"{d:8.3f} {c:12.4f} {f:10.4f} {c+f:8.4f} {slope:>12} {old:12.4f}")
    prev = (d, c + f)

# ------------------------------------------------------------------ budget
print("\n" + "=" * 74)
print("REWARD BUDGET (episode-normalised maxima)")
print("=" * 74)
budget = {
    "alive":         (1.0,   1.0),
    "reach_coarse":  (1.0,   1.0),   # ~0.92 realistic at contact
    "reach_fine":    (1.0,   2.0),   # ~0.45 realistic (cube half-width 2.5 cm)
    "hand_height":   (0.05,  5.0),
    "grasp":         (1.0,   2.0),
    "lift_bonus":    (1.0,   3.0),
    "object_height": (0.2,  10.0),
}
tot = 0.0
print(f"{'term':18} {'max raw':>10} {'weight':>8} {'max epn':>10}")
for k, (mx, w) in budget.items():
    print(f"{k:18} {mx:10.3f} {w:8.2f} {mx*w:10.3f}")
    tot += mx * w
print(f"{'TOTAL':18} {'':>10} {'':>8} {tot:10.3f}")
print("\nfor reference, the OLD budget totalled 8.250 and the 300-iteration")
print("policy scored 3.105 of it (37.6%) without ever touching the cube.")

env.close()
app.close()
