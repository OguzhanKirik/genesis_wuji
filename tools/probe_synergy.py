#!/usr/bin/env python3
"""Check the Experiment 2 scene and find the hand pose with a scripted synergy grasp.

    docker compose run --rm adam_rl python /opt/tools/probe_synergy.py
    docker compose run --rm adam_rl python /opt/tools/probe_synergy.py --sweep

Without --sweep: spawns the env, prints counts, the fingertip positions
relative to the bolt at the open posture, the contact-sensor shapes, then
runs ONE scripted episode (both synergies ramp 0 -> 1 over the grasp phase,
then the env's own lift and hold) and prints the per-step phase, closure,
contacts, bolt height and the reward components -- the state machine end to end.

With --sweep: one hand pose per env (num_envs = 16): the hand's height above
the table (via the wrist_lift start position) x the bolt's forward offset from
the wrist, each running the same scripted grasp. Reports which cells succeed
under the env's own success criterion. Use it to set WRIST_HEIGHT / BOLT_XY.
"""

from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--sweep", action="store_true")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--closure", type=float, nargs=2, default=[1.0, 1.0],
                    help="final thumb / finger synergy commands for the scripted grasp (0..1)")
parser.add_argument("--trace-tips", action="store_true",
                    help="move the bolt away and print the five fingertip positions during the scripted close")
parser.add_argument("--wrist-height", type=float, default=None, help="override WRIST_HEIGHT (m above the table)")
parser.add_argument("--orient", choices=["config", "palm_down", "fingers_down"], default="config",
                    help="hand root orientation: config = env's HAND_ROT; fingers_down = URDF identity "
                         "(fingers along -z, palm normal +y, thumb on +x)")
parser.add_argument("--open", type=float, default=None,
                    help="override the OPEN curl as a fraction of the flexion limits for all synergy joints (e.g. 0.4)")
parser.add_argument("--thumb-abd", type=float, default=None, help="override the fixed thumb cmc_abd (rad)")
parser.add_argument("--bolt-rel", type=float, nargs=2, default=None, metavar=("DX", "DY"),
                    help="place the bolt at wrist_xy + (DX, DY) instead of BOLT_XY")
parser.add_argument("--bolt-yaw", type=float, default=None, help="override BOLT_YAW_DEG (0: axis along y, 90: along x)")
parser.add_argument("--closure-range", type=float, nargs=2, default=[0.0, 1.0],
                    help="--sweep-closure: min and max closure level to sweep")
parser.add_argument("--sweep-closure", action="store_true",
                    help="--sweep over closure levels instead of poses: rows = thumb closure, cols = finger closure")
parser.add_argument("--sweep-dy", type=float, nargs=2, default=None,
                    help="--sweep: sweep bolt y offsets (min max) instead of x offsets")
parser.add_argument("--bolt-dx", type=float, default=0.0, help="shift the bolt in x from BOLT_XY")
parser.add_argument("--bolt-dy", type=float, default=0.0, help="shift the bolt in y from BOLT_XY")
parser.add_argument("--sweep-heights", type=float, nargs=2, default=[0.0, 0.03],
                    help="--sweep: wrist_lift start offsets (>= 0) added to the wrist height, min max")
parser.add_argument("--sweep-dx", type=float, nargs=2, default=[-0.03, 0.03],
                    help="--sweep: bolt x offsets, min max")
args, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from genesis_rl.envs import wuji_synergy_grasp_env_cfg as C  # noqa: E402
from genesis_rl.envs.adam_u_wuji_grasp_env_cfg import _t  # noqa: E402

cfg = C.WujiSynergyGraspEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = "cuda:0"
import re as _re  # noqa: E402
if args.wrist_height is not None:
    cfg.scene.robot.init_state.pos = (C.WRIST_XY[0], C.WRIST_XY[1], C.TABLE_TOP + args.wrist_height)
if args.orient == "palm_down":
    cfg.scene.robot.init_state.rot = (0.5, 0.5, -0.5, -0.5)
elif args.orient == "fingers_down":
    cfg.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
if args.open is not None:
    q_open = {k: round(v * args.open, 4) for k, v in cfg.actions.synergies.q_closed.items()}
    cfg.actions.synergies.q_open = q_open
    jp = cfg.scene.robot.init_state.joint_pos
    for k in list(jp.keys()):
        if k in q_open:
            jp[k] = q_open[k]
if args.thumb_abd is not None:
    cfg.actions.synergies.q_fixed["r_thumb_cmc_abd"] = args.thumb_abd
    cfg.scene.robot.init_state.joint_pos["r_thumb_cmc_abd"] = args.thumb_abd
bx, by, bz = cfg.scene.object.init_state.pos
if args.bolt_rel is not None:
    bx, by = C.WRIST_XY[0] + args.bolt_rel[0], C.WRIST_XY[1] + args.bolt_rel[1]
cfg.scene.object.init_state.pos = (bx + args.bolt_dx, by + args.bolt_dy, bz)
if args.bolt_yaw is not None:
    cfg.scene.object.init_state.rot = C._yawed_lying_rot(args.bolt_yaw)
WRIST_H = args.wrist_height if args.wrist_height is not None else C.WRIST_HEIGHT
print(f"overrides: orient={args.orient} open={args.open} thumb_abd={args.thumb_abd} bolt_rel={args.bolt_rel} bolt_yaw={args.bolt_yaw} wrist_h={WRIST_H}")
gym.register(id="ProbeSyn", entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": C.WujiSynergyGraspEnvCfg})
env = gym.make("ProbeSyn", cfg=cfg)
obs, _ = env.reset()
u = env.unwrapped
robot, bolt = u.scene["robot"], u.scene["object"]
origins = u.scene.env_origins
n = u.num_envs
W = 78

print("=" * W)
print(f"OBS_WIDTH {obs['policy'].shape[-1]}  ACTION_DIM {u.action_manager.total_action_dim}  "
      f"BODIES {len(robot.data.body_names)}  JOINTS {len(robot.data.joint_names)}  "
      f"episode {u.max_episode_length} steps  phases {C.GRASP_STEPS}/{C.LIFT_STEPS}/{C.HOLD_STEPS}")
print("joints:", " ".join(robot.data.joint_names))
cs = u.scene["cs_thumb"]
print(f"contact sensor: force_matrix_w {tuple(_t(cs.data.force_matrix_w).shape)}  net_forces_w {tuple(_t(cs.data.net_forces_w).shape)}")
from pxr import Usd, UsdPhysics  # noqa: E402
stage = u.sim.stage if hasattr(u.sim, "stage") else None
if stage is None:
    import isaaclab.sim as _sim
    stage = _sim.utils.get_current_stage() if hasattr(_sim.utils, "get_current_stage") else None
if stage is not None:
    for root in ("/World/envs/env_0/Bolt", "/World/envs/env_0/Table"):
        print(f"prims under {root}:")
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
            apis = [a for a in prim.GetAppliedSchemas() if "Physics" in a]
            print(f"   {prim.GetPath()}  [{prim.GetTypeName()}] {apis}")


def body_pos(name):
    return _t(robot.data.body_pos_w)[:, robot.data.body_names.index(name)] - origins


def report_pose(label):
    bp = _t(bolt.data.root_pos_w) - origins
    wr = body_pos("r_wrist")
    print(f"{label}")
    print(f"  wrist  x={wr[0, 0]:+.3f} y={wr[0, 1]:+.3f} z={wr[0, 2]:+.3f}  ({wr[0, 2] - C.TABLE_TOP:+.3f} above table)")
    print(f"  bolt   x={bp[0, 0]:+.3f} y={bp[0, 1]:+.3f} z={bp[0, 2]:+.3f}  ({bp[0, 2] - C.BOLT_REST_Z:+.4f} vs rest)")
    for k, body in C.TIP_BODIES.items():
        p = body_pos(body)[0]
        d = p - bp[0]
        print(f"  {k:6s} distal x={p[0]:+.3f} y={p[1]:+.3f} z={p[2]:+.3f}   rel. bolt dx={d[0]:+.3f} dy={d[1]:+.3f} dz={d[2]:+.3f}  "
              f"({p[2] - C.TABLE_TOP:+.3f} above table)")


# settle the open pose
a = torch.zeros((n, 2), device=u.device)
a[:] = -1.0
for _ in range(15):
    u.step(a)
report_pose("\nopen posture after 0.5 s (action -1 = closure 0):")
# Self-check: the bolt is resting on the table, so its contact sensor must read
# its weight. A zero here means the PhysX filter stopped matching and every
# "bolt did not touch the table" test would silently pass.
print(f"  bolt-table clearance at rest {C.bolt_table_clearance(u).mean() * 1000:+.1f} mm   "
      f"(negative at rest is expected: the bolt IS on the table)")
_tf = C._update(u)["tip_force"][0]
print(f"  fingertip bolt-force obs at rest (thumb/index/middle) "
      f"{_tf[0]:.2f}/{_tf[1]:.2f}/{_tf[2]:.2f} N")

if args.sweep:
    # one cell per env: height offsets via the lift joint start position, bolt forward offsets
    side = int(round(n ** 0.5))
    if args.sweep_closure:
        heights = torch.zeros(side, device=u.device)
        forwards = torch.zeros(side, device=u.device)
        levels = torch.linspace(*args.closure_range, side, device=u.device)
        clo_grid = torch.cartesian_prod(levels, levels)[:n]        # (thumb, fingers) per env
    else:
        heights = torch.linspace(*args.sweep_heights, side, device=u.device)   # added to the wrist height, via wrist_lift
        forwards = torch.linspace(*(args.sweep_dy if args.sweep_dy is not None else args.sweep_dx), side, device=u.device)
    sweep_axis = 1 if args.sweep_dy is not None else 0
    grid = torch.cartesian_prod(heights, forwards)[:n]
    env.reset()
    lift_id, _ = robot.find_joints([C.LIFT_JOINT])
    q = _t(robot.data.joint_pos).clone()
    q[:, lift_id[0]] = grid[:, 0].clamp(min=0.0)
    # heights below 0 cannot be reached with the prismatic joint (lower limit 0); lower the bolt instead? no --
    # drop those cells to the joint's floor and say so.
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    pose = _t(bolt.data.root_pose_w).clone()
    pose[:, sweep_axis] += grid[:, 1]
    bolt.write_root_pose_to_sim(pose)
    bolt.write_root_velocity_to_sim(torch.zeros((n, 6), device=u.device))
    # the buffers' lift_start must see the new start height
    C._buffers(u)["lift_start"][:] = q[:, lift_id[0]]
    u.episode_length_buf[:] = 0
    for _ in range(15):
        u.step(a)
    report_pose("\nsweep: env0 after re-placement:")

if args.trace_tips:
    pose = _t(bolt.data.root_pose_w).clone()
    pose[:, 0] += 0.20          # park it further along the table (x 0.30-0.80), not off its side
    bolt.write_root_pose_to_sim(pose)
    bolt.write_root_velocity_to_sim(torch.zeros((n, 6), device=u.device))
    print("\nfingertip trace during the scripted close (env 0), relative to the wrist origin; h = height above the table:")
    print("   thumb/index/middle/ring/pinky as (dx, dy, h)")

# scripted grasp: ramp both synergies to the target over the grasp phase
target = torch.tensor(args.closure, device=u.device) * 2.0 - 1.0
if args.sweep and args.sweep_closure:
    target = clo_grid * 2.0 - 1.0                                   # (n, 2) per-env targets
z0 = _t(bolt.data.root_pos_w)[:, 2].clone()
rows = []
_first_table = {}
rm = u.reward_manager
comp = {name: torch.zeros(n, device=u.device) for name in rm.active_terms}
steps_done = int(u.episode_length_buf[0])
for k in range(u.max_episode_length - steps_done):
    t = int(u.episode_length_buf[0])
    frac = min(1.0, t / (C.GRASP_STEPS * 0.6))
    a[:] = -1.0 + frac * (target + 1.0)          # broadcasts for a (2,) or (n, 2) target
    obs, rew, term, trunc, extras = u.step(a)
    for name in rm.active_terms:
        comp[name] += rm._step_reward[:, rm.active_terms.index(name)] if hasattr(rm, "_step_reward") else 0.0
    b = C._buffers(u)
    z = _t(bolt.data.root_pos_w)[:, 2] - C.BOLT_REST_Z
    if not args.sweep:
        for _k in C.TIP_BODIES:
            _tf = C._tip_forces(u, _k)[1]
            if bool((_tf > C.TABLE_HIT_N).any()) and _k not in _first_table:
                _first_table[_k] = (t, float(_tf.max()), int((_tf > C.TABLE_HIT_N).sum()))
    if args.trace_tips and t <= C.GRASP_STEPS and t % 5 == 0:
        wr = body_pos("r_wrist")[0]
        cl = u.action_manager.get_term("synergies").processed_actions[0]
        parts = []
        for kk, body in C.TIP_BODIES.items():
            pp = body_pos(body)[0]
            parts.append(f"({pp[0] - wr[0]:+.3f},{pp[1] - wr[1]:+.3f},{pp[2] - C.TABLE_TOP:+.3f})")
        nets = [round(C._tip_forces(u, kk)[2][0].item(), 1) for kk in C.TIP_BODIES]
        filt = [round(C._tip_forces(u, kk)[1][0].item(), 1) for kk in C.TIP_BODIES]
        rows.append(f"  t={t:3d} clo=({cl[0]:.2f},{cl[1]:.2f}) " + " ".join(parts) + f"  netF={nets} tableF={filt}")
    if not args.sweep and not args.trace_tips and (t % 10 == 0 or t in (C.GRASP_STEPS - 1, C.GRASP_STEPS + C.LIFT_STEPS - 1, u.max_episode_length - 1)):
        cl = u.action_manager.get_term("synergies").processed_actions[0]
        lift_q = _t(robot.data.joint_pos)[0, robot.find_joints([C.LIFT_JOINT])[0][0]]
        fb = {kk: C._tip_forces(u, kk) for kk in ("thumb", "index", "middle")}
        bxy = _t(bolt.data.root_pos_w)[0, :2] - origins[0, :2]
        paid = "".join(k[5] for k in ("paid_thumb", "paid_finger", "paid_opposing", "paid_table", "paid_excess", "paid_drop") if b[k][0])
        rows.append(f"  t={t:3d} ph={int(C.phase_of(u)[0])} paid=[{paid:6s}] clo=({cl[0]:.2f},{cl[1]:.2f}) lift={lift_q:.3f} "
                    f"bolt dz={z[0]:+.3f} xy=({bxy[0]:.3f},{bxy[1]:.3f}) "
                    f"bolt-filtered F thumb/idx/mid={fb['thumb'][0][0]:.1f}/{fb['index'][0][0]:.1f}/{fb['middle'][0][0]:.1f}N "
                    f"table-filtered max={max(fb[kk][1][0].item() for kk in fb):.1f}N "
                    f"NET thumb/idx/mid={fb['thumb'][2][0]:.1f}/{fb['index'][2][0]:.1f}/{fb['middle'][2][0]:.1f}N "
                    f"streak={int(b['hold_streak'][0])} r={rew[0]:+.2f}")
    if bool((term | trunc).any()):
        break

if not args.sweep:
    if not args.trace_tips:
        print("\nscripted episode, env 0 (both synergies ramp to", args.closure, "over the first 60 % of the grasp phase):")
    print("\n".join(rows))

if not args.sweep:
    print("\nfirst finger-table contact above TABLE_HIT_N, per finger:")
    print("  " + ("none" if not _first_table else
                  "; ".join(f"{k} at t={v[0]} ({v[1]:.1f} N, {v[2]} envs)" for k, v in _first_table.items())))

b = C._buffers(u)
# the reset inside the last step() cleared the per-episode flags; the last_* copies survive it
print("\n" + "=" * W)
if not args.sweep:
    print("reward components summed over the episode, env 0:")
    for name in rm.active_terms:
        print(f"  {name:18s} {comp[name][0]:+8.3f}")
if args.sweep:
    if args.sweep_closure:
        print("sweep result: rows = THUMB closure, cols = FINGER closure (scripted, held for the grasp phase); S = success")
        row_labels = [f"th={v:.2f}" for v in levels.tolist()]
        col_header = "            " + " ".join(f"fi={v:.2f}" for v in levels.tolist())
    else:
        print(f"sweep result: rows = wrist height above table, cols = bolt {'y' if sweep_axis else 'x'} offset from nominal; S = success")
        row_labels = [f"h={WRIST_H + max(0.0, h):.3f}" for h in heights.tolist()]
        col_header = "               " + " ".join(f"{f:+.3f}" for f in forwards.tolist())
    succ = b["last_success"]
    print(col_header)
    for i in range(side):
        cells = ["   S  " if (i * side + j) < n and succ[i * side + j] else "   .  " for j in range(side)]
        print(f"  {row_labels[i]:8s} " + " ".join(cells))
    for label, key, mark in (("lifted at all (>2 cm after the grasp phase)", "last_was_up", "L"),
                             ("HELD >= 5 cm for the whole hold phase (success without the collision/force rules)", "last_held", "H"),
                             ("finger-table contact during the episode", "last_table_hit", "T"),
                             ("fingertip force above F_MAX", "last_excess", "F")):
        flags = b[key]
        print(f"  {label}, same grid:")
        for i in range(side):
            cells = [f"   {mark}  " if (i * side + j) < n and flags[i * side + j] else "   .  " for j in range(side)]
            print(f"  {row_labels[i]:8s} " + " ".join(cells))
print(f"episode outcome: success {int(b['last_success'].sum())}/{n}   held {int(b['last_held'].sum())}   table_hit {int(b['last_table_hit'].sum())}   "
      f"excess_force {int(b['last_excess'].sum())}   lifted(>2cm) {int(b['last_was_up'].sum())}   dropped {int(b['last_drop'].sum())}   "
      f"thumb contact {b['last_thumb_frac'].mean() * 100:.0f} % of grasp steps, finger {b['last_finger_frac'].mean() * 100:.0f} %")
print("=" * W)
env.close()
app.close()
