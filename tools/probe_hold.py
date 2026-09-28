#!/usr/bin/env python3
"""Can this hand physically HOLD the cube? A scripted-grasp feasibility test.

Independent of RL. If a hand-placed, hand-closed grasp cannot hold the cube
against gravity, then no reward function and no amount of training will produce
a lift, and the blocker is the actuator spec rather than the MDP.

The right hand spawns beside the table, not over it, so a cube teleported to the
fingertips is already in free space -- no arm motion or IK is needed. Place the
cube at the pinch point with the hand open, close the fingers, then hold and
watch whether the cube stays or falls.

    docker compose exec adam_rl python /opt/tools/probe_hold.py
"""

from __future__ import annotations

import sys

sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

from genesis_rl.envs.adam_u_grasp_env_cfg import AdamUGraspEnvCfg

cfg = AdamUGraspEnvCfg()
cfg.scene.num_envs = 2
cfg.sim.device = "cuda:0"
# Determinism for a physics check: no cube jitter, no joint jitter.
cfg.events.reset_object_position.params["pose_range"] = {
    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
# The fingertips sit at z 0.84-0.99, BELOW the 0.95 m object_dropped threshold.
# Left enabled, placing the cube at the pinch point terminates the episode on
# the very next step, the env auto-resets, and the cube reappears on the table
# at z 1.0500 -- which reads as a rock-solid "HELD" while nothing was ever
# grasped. Disable the termination for this physics check.
# -1000 so it can NEVER fire. At 0.05 the cube still fell out of the hand,
# reached the floor inside the close loop, terminated, and reset_object_position
# put it back on the table at 1.0500 -- masking the fall a second time.
cfg.terminations.object_dropped.params["height_threshold"] = -1000.0

gym.register(id="Hold", entry_point="isaaclab.envs:ManagerBasedRLEnv",
             disable_env_checker=True, kwargs={"env_cfg_entry_point": AdamUGraspEnvCfg})
env = gym.make("Hold", cfg=cfg)
env.reset()
u = env.unwrapped
robot = u.scene["robot"]
obj = u.scene["object"]
am = u.action_manager
n_arm = am.get_term("right_arm").action_dim

TIPS = ["R_thumb_distal", "R_index_distal", "R_middle_distal",
        "R_ring_distal", "R_pinky_distal"]
tip_i = {t: robot.data.body_names.index(t) for t in TIPS}
palm_i = robot.data.body_names.index("wristRollRight")

mass = float(torch.as_tensor(obj.root_view.get_masses()).flatten()[0].item())
print("=" * 74)
print(f"cube mass after startup randomisation: {mass:.4f} kg")
print(f"cube size 0.05 m, static friction 0.9, table surface z 1.025")
print(f"finger effort_limit 10.0 Nm, stiffness 10.0/5.0, damping 2.0  (UNVERIFIED,")
print(f"  README flags these as certainly wrong -- fingers as strong as a shoulder)")


def act(fing, arm=0.0):
    a = torch.zeros((u.num_envs, am.total_action_dim), device=u.device)
    a[:, :n_arm] = arm
    a[:, n_arm:] = fing
    return a


def tips_now():
    return {t: robot.data.body_pos_w[0, tip_i[t]].clone() for t in TIPS}


def place(pos):
    st = obj.data.default_root_state.clone()
    st[:, :3] = pos.unsqueeze(0).expand(u.num_envs, 3)
    obj.write_root_pose_to_sim(st[:, :7])
    obj.write_root_velocity_to_sim(torch.zeros_like(st[:, 7:]))


def candidates():
    t = tips_now()
    palm = robot.data.body_pos_w[0, palm_i].clone()
    allc = torch.stack(list(t.values())).mean(dim=0)
    return {
        "all-fingertip centroid":  allc,
        "thumb-middle midpoint":   (t["R_thumb_distal"] + t["R_middle_distal"]) / 2,
        "thumb-index midpoint":    (t["R_thumb_distal"] + t["R_index_distal"]) / 2,
        "centroid, 1cm to palm":   allc + 0.35 * (palm - allc) / torch.norm(palm - allc) * 0.03,
    }


print("\n" + "=" * 74)
print("STEP 1 -- does a 5 cm cube even fit in this hand's span?")
print("=" * 74)
span = {}
for label, a in [("open (-1)", -1.0), ("mid (0)", 0.0), ("closed (+1)", +1.0)]:
    env.reset()
    for _ in range(60):
        u.step(act(a))
    t = tips_now()
    row = {f.replace("R_", "").replace("_distal", ""):
           torch.norm(t["R_thumb_distal"] - t[f]).item()
           for f in TIPS if f != "R_thumb_distal"}
    span[label] = row
    print(f"  thumb-to-finger tip distance, {label:12} " +
          "  ".join(f"{k} {v:.4f}" for k, v in row.items()))
max_span = max(max(r.values()) for r in span.values())
print(f"\n  widest thumb-to-fingertip opening: {max_span:.4f} m   cube edge: 0.050 m")
fits = max_span > 0.05
print(f"  a 0.05 m cube {'FITS' if fits else 'DOES NOT FIT'} between thumb and fingers")

print("\n" + "=" * 74)
print("STEP 2 -- place cube at the pinch point, close, trace z every step")
print("=" * 74)
results = {}
for label in ["thumb-middle midpoint", "thumb-index midpoint",
              "all-fingertip centroid", "centroid, 1cm to palm"]:
    env.reset()
    for _ in range(40):
        u.step(act(-1.0))
    pos = candidates()[label]
    z_target = pos[2].item()
    place(pos)
    obj.update(0.0)
    z_place = obj.data.root_pos_w[0, 2].item()
    trace = [z_place]
    for _ in range(150):
        u.step(act(+1.0))
        trace.append(obj.data.root_pos_w[0, 2].item())
    z_end = trace[-1]
    z_min = min(trace)
    drop = z_place - z_end
    # A cube that stayed within 2 cm of where it was placed was held. A cube
    # that reappears at 1.0500 was reset; with the termination disabled that
    # cannot happen, and the trace would show the fall regardless.
    held = drop < 0.02 and abs(z_end - 1.05) > 1e-4
    results[label] = (z_place, z_end, drop, held)
    print(f"{label:26} placed {z_place:.4f}  end {z_end:.4f}  min {z_min:.4f}  "
          f"fell {drop:+.4f}  {'HELD' if held else 'fell'}")
    print(f"{'':26} z trace: " +
          " ".join(f"{trace[i]:.3f}" for i in [0, 5, 10, 20, 40, 80, 150]))

print("\n" + "=" * 74)
print("STEP 3 -- grip strength, isolated from closing speed")
print("=" * 74)
print("Step 2 is unfair: the cube is in free fall from the moment it is placed,")
print("so it asks the hand to CATCH a falling cube. A real grasp closes around a")
print("cube supported by the table. So: hold the cube kinematically while the")
print("fingers close, then release it and see whether the grip keeps it.")
print()
grip = {}
for label in ["thumb-middle midpoint", "thumb-index midpoint",
              "centroid, 1cm to palm"]:
    for close_steps in [30, 60, 120]:
        env.reset()
        for _ in range(40):
            u.step(act(-1.0))
        pos = candidates()[label].clone()
        # Freeze the cube at the pinch point while the fingers close around it.
        for _ in range(close_steps):
            place(pos)
            u.step(act(+1.0))
        place(pos)
        obj.update(0.0)
        z_release = obj.data.root_pos_w[0, 2].item()
        # Release: stop writing the pose, keep the fingers commanded closed.
        trace = []
        for _ in range(120):
            u.step(act(+1.0))
            trace.append(obj.data.root_pos_w[0, 2].item())
        z_end = trace[-1]
        drop = z_release - z_end
        held = drop < 0.02
        grip[(label, close_steps)] = held
        print(f"  {label:24} close {close_steps:>3} steps  "
              f"release z {z_release:.4f} -> {z_end:.4f}  fell {drop:+.4f}  "
              f"{'HELD' if held else 'fell'}")
        print(f"  {'':24} z trace: " +
              " ".join(f"{trace[i]:.3f}" for i in [0, 5, 15, 30, 60, 119]))

any_grip = any(grip.values())
print()
print(f"  grip held in {sum(grip.values())}/{len(grip)} configurations")

print("\n" + "=" * 74)
if any_grip:
    print("VERDICT: the hand CAN hold the cube once closed around it. Grip")
    print("         strength is NOT the blocker; the hand simply cannot catch a")
    print("         cube already in free fall. A grasp of a table-supported cube")
    print("         is physically reachable, so the remaining gap is the MDP --")
    print("         exploration, and the termination that punishes approaching.")
elif False:
    print("VERDICT: the hand CAN hold the cube. A grasp-and-lift is physically")
    print("         reachable, so the remaining gap is the MDP -- exploration and")
    print("         the termination that punishes approaching the cube.")
else:
    print("VERDICT: the hand CANNOT hold the cube in ANY tested placement.")
    print("         No reward function fixes this. The blocker is the actuator")
    print("         spec -- finger stiffness/damping/effort limits -- and those")
    print("         are exactly the numbers the README lists as unverified.")
    print("         Get them from PNDbotics before spending GPU time on RL.")
print("=" * 74)

env.close()
app.close()
