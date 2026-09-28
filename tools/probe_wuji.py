#!/usr/bin/env python3
"""Spawn a Wuji grasp env and measure what the reward terms assume.

Run inside the 6.0 image after any change to the robot USD, the workcell
layout or either Wuji env cfg:

    docker compose run --rm adam_rl python /opt/tools/probe_wuji.py                    # reach scenario
    docker compose run --rm adam_rl python /opt/tools/probe_wuji.py --scenario grasp   # grasp-only scenario
    docker compose run --rm -e ADAM_U_WORKCELL=bin adam_rl python /opt/tools/probe_wuji.py

It prints, in order:
  1. counts: obs width, action dim, bodies, joints -- and the right-hand joint
     order with limits, which is the action order the policy learns
  2. pose: the waist/arm joints actually reached vs their defaults (a drive
     that is not holding shows up here), world heights of waistRoll, the wrist
     and fingertip frames relative to the settled bolt
  3. grip sign: fingers driven to action +1 then -1, tip-to-palm distances and
     the flexion closure the grasp reward computes -- +1 must be CLOSED
  4. (grasp scenario) phase A: a flexion-only close on nothing with one thumb
     abduction per env, reporting the thumb/index closest approach --
     diagnostic only, the grasp is not a fingertip pinch; phase B: the bolt on
     a +-2 cm grid around BOLT_XY, close, lift -- how many placements a naive
     scripted close picks up. 5/16 from the configured pose on 2026-09-19.
  5. every reward term at the settled state

Everything runs inside one 5 s / 10 s episode; a time-out reset in the middle
of a sweep respawns the bolt and reads as a phantom +19 mm rise.
"""

from __future__ import annotations

import argparse
import sys

_ap = argparse.ArgumentParser()
_ap.add_argument("--scenario", choices=["reach", "grasp"], default="reach",
                 help="reach: adam_u_wuji_grasp_env_cfg. grasp: adam_u_wuji_grasponly_env_cfg, "
                      "which also runs a scripted close-and-lift attempt.")
_args, _ = _ap.parse_known_args()
sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from genesis_rl.envs.adam_u_wuji_grasp_env_cfg import LAYOUT, _t  # noqa: E402

if _args.scenario == "grasp":
    from genesis_rl.envs.adam_u_wuji_grasponly_env_cfg import AdamUWujiGraspOnlyEnvCfg as AdamUWujiGraspEnvCfg  # noqa: E402
    from genesis_rl.envs.adam_u_wuji_grasponly_env_cfg import BOLT_XY  # noqa: E402
else:
    from genesis_rl.envs.adam_u_wuji_grasp_env_cfg import AdamUWujiGraspEnvCfg  # noqa: E402
    BOLT_XY = LAYOUT["bolt_center_xy"]

cfg = AdamUWujiGraspEnvCfg()
cfg.scene.num_envs = 16 if _args.scenario == "grasp" else 4
cfg.sim.device = "cuda:0"

gym.register(id="ProbeWuji", entry_point="isaaclab.envs:ManagerBasedRLEnv",
             disable_env_checker=True, kwargs={"env_cfg_entry_point": AdamUWujiGraspEnvCfg})
env = gym.make("ProbeWuji", cfg=cfg)
obs, _ = env.reset()
u = env.unwrapped
robot = u.scene["robot"]
bolt = u.scene["object"]
am = u.action_manager
origins = u.scene.env_origins

W = 72
print("=" * W)
print(f"SCENARIO {_args.scenario}  LAYOUT {LAYOUT['file']}  table_top={LAYOUT['table_top']:.3f}  "
      f"surface_z={LAYOUT['surface_z']:.3f}  bolt_center_xy={BOLT_XY}  bin={'yes' if LAYOUT['bin'] else 'no'}")
print(f"OBS_WIDTH {obs['policy'].shape[-1]}")
print(f"ACTION_DIM {am.total_action_dim}  (arm {am.get_term('right_arm').action_dim} + fingers {am.get_term('right_fingers').action_dim})")
print(f"NUM_BODIES {len(robot.data.body_names)}")
print(f"NUM_JOINTS {len(robot.data.joint_names)}")

fing = am.get_term("right_fingers")
n_arm = am.get_term("right_arm").action_dim
fing_ids = fing._joint_ids
if isinstance(fing_ids, slice):
    fing_ids = list(range(len(robot.data.joint_names)))[fing_ids]
fing_names = [robot.data.joint_names[i] for i in fing_ids]
lim = _t(robot.data.joint_pos_limits)[0]
dflt = _t(robot.data.default_joint_pos)[0]
print("\nright-hand action order (joint, default, limits):")
for k, (i, n) in enumerate(zip(fing_ids, fing_names)):
    print(f"  a[{n_arm + k:2d}] {n:26s} default={dflt[i]:+.3f}  [{lim[i, 0]:+.3f}, {lim[i, 1]:+.3f}]")

# --- pose --------------------------------------------------------------------
zero = torch.zeros((u.num_envs, am.total_action_dim), device=u.device)


def settle(action, steps):
    a = zero.clone()
    a[:, n_arm:] = action
    for _ in range(steps):
        u.step(a)


def body_z(name):
    return _t(robot.data.body_pos_w)[0, robot.data.body_names.index(name), 2].item()


def frame_pos(name):
    return _t(u.scene[name].data.target_pos_w)[0, 0]


OPEN = -1.0 if _args.scenario == "reach" else 0.0  # action that means "open hand"
settle(OPEN, 90)  # 3 s: bolt drops and stops, hand open
print("\n" + "=" * W)
root = _t(robot.data.root_pos_w)[0] - origins[0]
print(f"robot root (env frame)      x={root[0]:+.4f} y={root[1]:+.4f} z={root[2]:+.4f}   expected z={cfg.scene.robot.init_state.pos[2]:.4f}")
print(f"waistRoll world z           {body_z('waistRoll'):.4f}   expected ~1.1267 (adam-moveit)")
jn = robot.data.joint_names
qq = _t(robot.data.joint_pos)[0]
qd = _t(robot.data.default_joint_pos)[0]
print("arm/waist joints: measured vs default (rad):")
for name in ["waistPitch", "shoulderPitch_Right", "shoulderRoll_Right", "shoulderYaw_Right", "elbow_Right",
             "wristYaw_Right", "wristPitch_Right", "wristRoll_Right"]:
    i = jn.index(name)
    print(f"  {name:22s} q={qq[i]:+.3f}  default={qd[i]:+.3f}  diff={qq[i] - qd[i]:+.3f}")
print(f"wristRollRight world z      {body_z('wristRollRight'):.4f}")
bp = _t(bolt.data.root_pos_w)[0] - origins[0]
for fname in ("right_wrist_frame", "right_thumb_frame", "right_index_frame"):
    p = frame_pos(fname) - origins[0]
    rel = p - bp
    print(f"{fname:26s}  x={p[0]:+.4f} y={p[1]:+.4f} z={p[2]:+.4f}   rel. to bolt dx={rel[0]:+.3f} dy={rel[1]:+.3f} dz={rel[2]:+.3f}")
bq = _t(bolt.data.root_quat_w)[0]
print(f"bolt (env frame)            x={bp[0]:+.4f} y={bp[1]:+.4f} z={bp[2]:+.4f}  quat(xyzw)={[round(v, 3) for v in bq.tolist()]}")
print(f"bolt z above surface        {bp[2] - LAYOUT['surface_z']:+.4f}   cfg BOLT_REST_OFFSET={LAYOUT['bolt_rest_z'] - LAYOUT['surface_z']:.4f}")
rest_all = _t(bolt.data.root_pos_w)[:, 2] - LAYOUT["surface_z"]
print(f"bolt rest offset, all envs  min={rest_all.min():.4f} max={rest_all.max():.4f}")
print(f"bolt lin vel                {torch.norm(_t(bolt.data.root_lin_vel_w)[0]).item():.4f} m/s (should be ~0)")

# --- grip sign -----------------------------------------------------------------
TIPS = ["r_thumb_distal", "r_index_finger_distal", "r_middle_finger_distal",
        "r_ring_finger_distal", "r_pinky_distal"]
palm_i = robot.data.body_names.index("wristRollRight")
tip_i = {t: robot.data.body_names.index(t) for t in TIPS}

flex_re = ["_flex", "_pip", "_dip", "thumb_mcp", "thumb_ip"]
flex_ids = [i for i, n in zip(fing_ids, fing_names) if any(s in n for s in flex_re)]


def measure(label):
    palm = _t(robot.data.body_pos_w)[0, palm_i]
    d = {t: torch.norm(_t(robot.data.body_pos_w)[0, tip_i[t]] - palm).item() for t in TIPS}
    q = _t(robot.data.joint_pos)[0]
    up = _t(robot.data.joint_pos_limits)[0, :, 1]
    closure = torch.stack([(q[i] / up[i]).clamp(0, 1) for i in flex_ids]).mean().item()
    mean_d = sum(d.values()) / len(d)
    print(f"  {label:10s} mean tip-to-palm {mean_d:.4f} m   thumb {d[TIPS[0]]:.4f}  index {d[TIPS[1]]:.4f}  "
          f"closure(q/upper over {len(flex_ids)} flexion joints) {closure:.3f}")
    print(f"             flexion q: " + " ".join(f"{q[i]:+.2f}" for i in flex_ids))
    return mean_d, closure


print("\n" + "=" * W)
print("grip sign (fingers only; arm held at default):")
# Fresh episode: the sections above used 90 of the 300 steps, and a time-out
# reset in the middle of this sweep would read as "the fingers went back to
# zero" (it did, once). 4 x 45 steps = 180 < 300.
env.reset()
settle(0.0, 45); d0, c0 = measure("a=0")
settle(+1.0, 45); dp, cp = measure("a=+1")
settle(-1.0, 45); dm, cm = measure("a=-1")
settle(+1.0, 45); dp2, cp2 = measure("a=+1 again")
verdict = "CLOSES" if dp < dm and cp > cm and cp2 > 0.9 else "DOES NOT CLOSE -- check sign"
print(f"  => action +1 {verdict}: tip-to-palm {dm:.4f} -> {dp:.4f} m, closure {cm:.2f} -> {cp:.2f} (again {cp2:.2f})")

# --- scripted grasp attempt (grasp scenario) --------------------------------------
if _args.scenario == "grasp":
    import math as _m
    from isaaclab.utils.math import quat_mul
    n = u.num_envs
    a = zero.clone()

    def tip(name):
        return _t(u.scene[name].data.target_pos_w)[:, 0]

    # Finger action vector for a CLOSE: flexion joints to +1, finger abduction
    # held (0 = pre-grasp), thumb abduction set per env. Driving every joint to
    # +1 also abducts the thumb to +0.698 -- away from the fingers -- and the
    # thumb and index then never meet (measured: closest approach 43 mm, at
    # the START of the close).
    abd_cols = [k for k, nme in enumerate(fing_names) if nme.endswith("_abd")]
    thumb_abd_col = fing_names.index("r_thumb_cmc_abd")
    finger_abd_cols = [k for k in abd_cols if k != thumb_abd_col]

    def close_action(thumb_abd_a):
        f = torch.ones((n, len(fing_names)), device=u.device)
        f[:, finger_abd_cols] = 0.0
        f[:, thumb_abd_col] = thumb_abd_a
        return f

    # Phase A: close on nothing and record where the thumb and index tips come
    # closest. Two thumb-opposition settings, half the envs each.
    print("\n" + "=" * W)
    print("phase A (diagnostic): close on nothing, one thumb-abduction value per env; where do the")
    print("thumb and index tips come closest? (the grasp is NOT a tip pinch, so this is informational)")
    env.reset()
    settle(OPEN, 30)
    pose = _t(bolt.data.root_pose_w).clone()
    pose[:, 1] += 0.5                       # bolt out of the way
    bolt.write_root_pose_to_sim(pose)
    bolt.write_root_velocity_to_sim(torch.zeros((n, 6), device=u.device))
    settle(OPEN, 10)
    # One thumb-abduction value per env, spanning the joint's range. The
    # action is offset from the pre-grasp -0.40 rad with scale 1.098, so
    # a = (q - (-0.40)) / 1.098.
    abd_q = torch.linspace(-1.45, 0.65, n, device=u.device)
    thumb_abd = (abd_q + 0.40) / 1.098
    f_close = close_action(thumb_abd)
    th, ix = [], []
    for _ in range(30):
        a[:, :n_arm] = 0.0
        a[:, n_arm:] = f_close
        u.step(a)
        th.append(tip("right_thumb_frame").clone()); ix.append(tip("right_index_frame").clone())
    th = torch.stack(th); ix = torch.stack(ix)                 # (T, N, 3)
    gap = torch.norm(th - ix, dim=2)                            # (T, N)
    k = gap.argmin(dim=0)
    P = (th[k, torch.arange(n)] + ix[k, torch.arange(n)]) / 2   # (N, 3) world
    P_env = P - origins
    th0 = th[0, 0] - origins[0]; ix0 = ix[0, 0] - origins[0]
    gmin = gap.min(dim=0).values
    print(f"  open tips (env0): thumb {[round(v, 3) for v in th0.tolist()]}  index {[round(v, 3) for v in ix0.tolist()]}")
    print("  thumb cmc_abd [rad] -> closest thumb-index approach during a flexion close:")
    thumb_end = th[-1] - origins; index_end = ix[-1] - origins
    for i in range(n):
        print(f"    abd={abd_q[i]:+.2f}  gap {gmin[i] * 1000:5.1f} mm at step {int(k[i]):2d}   pinch pt x={P_env[i, 0]:.3f} y={P_env[i, 1]:.3f} z={P_env[i, 2]:.3f}"
              f"   closed thumb tip z={thumb_end[i, 2]:.3f} index tip z={index_end[i, 2]:.3f}")
    ib = int(gmin.argmin())
    thumb_abd_best = thumb_abd[ib].item()
    P_best = P_env[ib]
    print(f"  => best opposition at cmc_abd={abd_q[ib]:+.2f} rad (action {thumb_abd_best:+.2f}): gap {gmin[ib] * 1000:.1f} mm, "
          f"pinch point (env frame) {[round(v, 4) for v in P_best.tolist()]}   (table top {LAYOUT['table_top']:.3f}, bolt rests at {LAYOUT['bolt_rest_z']:.3f})")

    # Phase B: bolts on a grid around the CONFIGURED BOLT_XY, on the table;
    # close with the env's own default abduction (action 0); lift.
    print("\n" + "=" * W)
    print(f"phase B: bolt on the table around BOLT_XY={BOLT_XY} (dx, dy grid +-2 cm, rows alternate yaw 0/90);")
    print("close (flexion +1, abduction held at the env default), then raise the arm (shoulder + elbow).")
    env.reset()
    settle(OPEN, 30)
    side = int(round(n ** 0.5))
    ds = torch.linspace(-0.02, 0.02, side, device=u.device)
    grid = torch.cartesian_prod(ds, ds)[:n]
    idx = torch.arange(n, device=u.device)
    yaw90 = (idx // side) % 2 == 1
    pose = _t(bolt.data.root_pose_w).clone()
    pose[:, 0] = origins[:, 0] + BOLT_XY[0] + grid[:, 0]
    pose[:, 1] = origins[:, 1] + BOLT_XY[1] + grid[:, 1]
    pose[:, 2] = LAYOUT["bolt_rest_z"] + 0.002
    q_lie = torch.tensor([_m.sqrt(0.5), 0.0, 0.0, _m.sqrt(0.5)], device=u.device).expand(n, 4)
    q_yaw = torch.tensor([0.0, 0.0, _m.sqrt(0.5), _m.sqrt(0.5)], device=u.device).expand(n, 4)
    pose[:, 3:7] = torch.where(yaw90.unsqueeze(1), quat_mul(q_yaw, q_lie), q_lie)
    bolt.write_root_pose_to_sim(pose)
    bolt.write_root_velocity_to_sim(torch.zeros((n, 6), device=u.device))
    settle(OPEN, 10)
    z0 = _t(bolt.data.root_pos_w)[:, 2].clone(); xy0 = _t(bolt.data.root_pos_w)[:, :2].clone()
    f_close = close_action(torch.zeros(n, device=u.device))
    min_d = torch.full((n,), 9.0, device=u.device)
    for _ in range(30):
        a[:, :n_arm] = 0.0
        a[:, n_arm:] = f_close
        u.step(a)
        bp = _t(bolt.data.root_pos_w)
        d = torch.minimum(torch.norm(tip("right_thumb_frame") - bp, dim=1), torch.norm(tip("right_index_frame") - bp, dim=1))
        min_d = torch.minimum(min_d, d)
    z1 = _t(bolt.data.root_pos_w)[:, 2].clone(); xy1 = _t(bolt.data.root_pos_w)[:, :2].clone()
    lift = torch.zeros((n, n_arm), device=u.device)
    lift[:, 0] = -1.0; lift[:, 3] = -1.0
    for _ in range(34):
        a[:, :n_arm] = lift
        a[:, n_arm:] = f_close
        u.step(a)
    z2 = _t(bolt.data.root_pos_w)[:, 2].clone()
    wz = tip("right_wrist_frame")[:, 2]
    lifted = (z2 - z0) > 0.02
    print(f"  wrist z after lift {wz.mean():.4f}  (open was ~1.02)")
    print(f"  bolts lifted >2 cm: {int(lifted.sum())}/{n}")
    print("  case                       rise[mm]  pushed_xy[mm]  min tip-bolt[mm]")
    for i in range(n):
        print(f"    dx={grid[i, 0]:+.3f} dy={grid[i, 1]:+.3f} yaw={'90' if yaw90[i] else ' 0'}   "
              f"{(z2[i] - z0[i]) * 1000:+7.1f}   {torch.norm(xy1[i] - xy0[i]).item() * 1000:7.1f}   {min_d[i] * 1000:7.1f}   {'LIFTED' if lifted[i] else ''}")
    print("  => " + ("A SCRIPTED CLOSE LIFTS THE BOLT in the marked cases from the configured start pose"
                     if lifted.any() else
                     "no case lifted at BOLT_XY; the finger posture, gains or friction need work"))

# --- reward terms ---------------------------------------------------------------
print("\n" + "=" * W)
print("reward terms at the settled state (open hand, arm at default), mean over envs:")
env.reset()
settle(OPEN, 60)
rm = u.reward_manager
for name in rm.active_terms:
    tc = rm.get_term_cfg(name)
    val = tc.func(u, **tc.params)
    print(f"  {name:18s} raw={val.float().mean().item():+.4f}  weight={tc.weight:+.4g}")

print("=" * W)
env.close()
app.close()
