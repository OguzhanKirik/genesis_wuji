"""Export a reach-grasp checkpoint for the robot: actor.onnx, level.yaml and (optionally) episode.npz.

    python /opt/tools/export_policy.py --checkpoint_path <ck> --out handoff/policy/<name>            (ONNX + yaml, no sim)
    python /opt/tools/export_policy.py --checkpoint_path <ck> --out handoff/policy/<name> --record 1 (+ one recorded episode)

actor.onnx   input  obs    float32 [1, W]   (W = 74, or 82 at heap-map levels), raw observations
             output action float32 [1, 8]   the distribution mean; the runtime clamps it to [-1, 1]
             The observation normaliser ((obs - mean) / (std + eps)) is folded in.
level.yaml   every constant the sim-free runtime needs (control rate, action scales and smoothing, the synergy
             endpoints, the phase-machine thresholds, the torque latch, the observation layout); with --record
             also the 20 hand-joint base targets in right_hand_controller order and the wrist frame name.
episode.npz  per step: joint states, estimated rod pose, observation, action, joint targets (for the parity test).

Run with ADAM_U_RG_LEVEL set to the level the checkpoint was trained on.
"""
import argparse
import json
import math
import os
import sys

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--out", required=True, help="output directory")
parser.add_argument("--record", type=int, default=0, help="consecutive episodes of env 0 to record for the parity test")
parser.add_argument("--num_envs", type=int, default=4)
# The env module imports Isaac at import time, so the app is launched even when nothing is simulated.
from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app

os.makedirs(args.out, exist_ok=True)

# ---------------------------------------------------------------- actor.onnx
ck = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
sd = ck["actor_state_dict"]
W = sd["mlp.0.weight"].shape[1]
layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.nn.Parameter(sd["obs_normalizer._mean"].clone().float(), requires_grad=False)
        self.std = torch.nn.Parameter(sd["obs_normalizer._std"].clone().float(), requires_grad=False)
        self.eps = 1e-2                                   # rsl_rl EmpiricalNormalization default
        mods = []
        for i, li in enumerate(layers):
            lin = torch.nn.Linear(sd[f"mlp.{li}.weight"].shape[1], sd[f"mlp.{li}.weight"].shape[0])
            lin.weight.data = sd[f"mlp.{li}.weight"].clone().float()
            lin.bias.data = sd[f"mlp.{li}.bias"].clone().float()
            mods.append(lin)
            if i < len(layers) - 1:
                mods.append(torch.nn.ELU())
        self.mlp = torch.nn.Sequential(*mods)

    def forward(self, obs):
        return self.mlp((obs - self.mean) / (self.std + self.eps))


actor = Actor().eval()
dummy = torch.zeros(1, W)
onnx_path = os.path.join(args.out, "actor.onnx")
torch.onnx.export(actor, dummy, onnx_path, input_names=["obs"], output_names=["action"],
                  dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17, dynamo=False)
with torch.no_grad():
    ref = actor(torch.randn(3, W))
print(f"actor.onnx: obs [{W}] -> action [{ref.shape[1]}], layers {[sd[f'mlp.{l}.weight'].shape[0] for l in layers]}, iter {ck.get('iter')}")

# ---------------------------------------------------------------- level.yaml
from genesis_rl.envs import wuji_reach_grasp_env_cfg as G  # noqa: E402
from genesis_rl.tasks.wuji_pipeline import env_cfg as P  # noqa: E402


def C(name):
    """A constant from the reach-grasp module, or from the pipeline module it imports from."""
    return getattr(G, name) if hasattr(G, name) else getattr(P, name)



lvl_id = int(os.environ.get("ADAM_U_RG_LEVEL", "0"))
lvl = G.LEVELS[lvl_id]
lim = G._rg_limits()
obs_layout = [("bolt_pos_rel", 3), ("bolt_rot_rel", 6), ("wrist_lin_vel", 3), ("wrist_ang_vel", 3), ("finger_q", 9), ("finger_qd", 9),
              ("last_action", 8), ("tip_forces", 3), ("contacts", 3), ("ep_time", 1), ("arm_q_norm", 7), ("tip_dist", 3), ("safety", 16)]
if lvl.heap_map:
    obs_layout.append(("heap_map", G.HEAP_MAP_DIRS))
assert sum(n for _, n in obs_layout) == W, (sum(n for _, n in obs_layout), W)
assert lvl.free_grasp, "the runtime has no commanded grasp offset / pinch angle: export free-grasp levels only"
assert lvl.grasp_point_tol == 0.0, "the runtime cannot bind the latch to the commanded grasp point"

# ---------------------------------------------------------------- the simulator (joint limits, base targets, the recording)
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from importlib import metadata  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from genesis_rl.algos.rsl_rl.rsl_rl_ppo_cfg import RslRlWujiReachGraspPPORunnerCfg  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402

dev = "cuda:0"
cfg = G.WujiReachGraspEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.sim.device = dev
gym.register(id="ExportRG", entry_point="genesis_rl.envs.wuji_reach_grasp_env_cfg:ReachGraspEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": G.WujiReachGraspEnvCfg})
env = gym.make("ExportRG", cfg=cfg); u = env.unwrapped
robot = u.scene["robot"]
names = list(robot.joint_names)
arm_ids = [names.index(n) for n in C('RIGHT_ARM')]
jl = G._t(robot.data.joint_pos_limits)[0, arm_ids, :].cpu().numpy()

wxy, bxy = C('WRIST_XY'), C('BOLT_XY')
sph = lambda g: [[link, [float(x) for x in off], float(r)] for link, off, r in G.PROXY_GROUPS[g]]
synergy = list(C('THUMB_JOINTS')) + list(C('FINGER_JOINTS'))
# the runtime's schema (genesis_policy/level.py, schema 1); keys it does not read are carried for the record
# the lying rod's axis yaw at the spawn pose: reset_bolt_pose's rod yaw psi is measured from it
_q0 = G._t(u.scene["object"].data.default_root_pose)[:1, 3:7]
_a0 = math_utils.quat_apply(_q0, torch.tensor([[0.0, 0.0, 1.0]], device=_q0.device))[0]
YAW0 = float(math.atan2(float(_a0[1]), float(_a0[0])))

level = dict(
    schema=1,
    source=f"adam-rl export: {args.checkpoint_path} (iteration {int(ck.get('iter', -1))}), level {lvl_id} '{lvl.name}'",
    control=dict(rate_hz=C('CONTROL_HZ')),
    observation=dict(layout=[dict(name=n, size=k) for n, k in obs_layout],
                     episode_max_steps=int(G.episode_max_steps(lvl.retry_max, lvl.pile.settle_in)),
                     tip_force_scale=float(C('F_MAX')),
                     arm_q_limits=dict(lo=[float(x) for x in jl[:, 0]], hi=[float(x) for x in jl[:, 1]]),
                     tactile=bool(lvl.tactile),
                     **({"effort_obs": dict(force_per_nm=float(C('GRIP_MIN_N') / G.GRIP_TAU_MIN), contact_nm=float(G.TAU_CONTACT))}
                        if lvl.effort_obs else {})),
    action=dict(effective_translation=float(lim.effective_translation), effective_rotation=float(lim.effective_rotation),
                wrist_smoothing=float(lim.wrist_smoothing), action_smoothing=float(lim.action_smoothing),
                yaw_rate_scale=float(lvl.yaw_rate_scale if lvl.free_grasp else 1.0)),
    ik=dict(dls_damping=float(lim.dls_damping), nullspace_gain=float(lim.nullspace_gain), joint_margin=float(lim.joint_margin),
            max_joint_velocity=float(lim.max_joint_velocity), singularity_threshold=float(lim.singularity_threshold),
            workspace=dict(radius_xy=float(lim.workspace_radius_xy), z_min=float(lim.workspace_z_min), z_max=float(lim.workspace_z_max),
                           nominal_wrist_offset_xy=[round(wxy[0] - bxy[0], 6), round(wxy[1] - bxy[1], 6)])),
    wrist=dict(link=C('WRIST_BODY'), offset=dict(xyz=[0.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0])),
    arm=dict(joints=list(C('RIGHT_ARM'))),
    hand=dict(thumb_joints=list(C('THUMB_JOINTS')), finger_joints=list(C('FINGER_JOINTS')),
              open={n: float(C('Q_OPEN')[n]) for n in synergy}, closed={n: float(C('Q_CLOSED')[n]) for n in synergy},
              held={n: float(v) for n, v in C('Q_FIXED_RIGHT').items()},
              tips={k: dict(link=C('TIP_BODIES')[k], offset=[float(x) for x in C('TIP_OFFSETS')[k]]) for k in ("thumb", "index", "middle")}),
    phase=dict(settle_in=int(lvl.pile.settle_in), grasp_max_steps=int(G.GRASP_MAX_STEPS), grasp_retry_steps=int(G.GRASP_RETRY_STEPS),
               settle_steps=int(G.SETTLE_STEPS), drag_steps=int(G.DRAG_STEPS), free_prelift=bool(lvl.free_prelift),
               grip_check=bool(lvl.grip_check), grip_check_steps=int(G.GRIP_CHECK_STEPS), grip_test_total_m=float(G.GRIP_TEST_TOTAL_M),
               grip_check_ratio=float(G.GRIP_CHECK_RATIO), grip_check_min_tips_m=float(G.GRIP_CHECK_MIN_TIPS),
               grip_check_late=bool(lvl.grip_check_late), lift_steps=int(G.LIFT_STEPS), lift_slow_steps=int(G.LIFT_SLOW_STEPS),
               lift_slow_step_m=float(G.LIFT_SLOW_STEP_M), lift_height_m=float(C('LIFT_HEIGHT')), hold_steps=int(G.HOLD_STEPS),
               retry_max=int(lvl.retry_max), retry_backoff_m=float(lvl.retry_backoff), success_lift_m=float(C('SUCCESS_LIFT')),
               up_threshold_m=float(C('UP_THRESHOLD')), sense_est=bool(lvl.sense_est),
               roll_cap=float(lvl.roll_cap) if lvl.wrist_roll else 0.0,   # 0: never roll (lift along z)
               # untilted rods; a tilted rod's approach axis comes per episode (Runtime.reset(lift_dir=...))
               lift_dir=[0.0, 0.0, 1.0],
               latch=(dict(mode="torque", steps=int(C('STABLE_STEPS')), tau_min_nm=float(G.GRIP_TAU_MIN),
                           grasp_region_m=float(C('GRASP_REGION')), pinch_centre_tol_m=float(lvl.pinch_centre_tol),
                           rel_vel_max_m_s=float(C('REL_VEL_MAX')), est_vel_window=int(lvl.est_vel_window))
                      if lvl.hw_latch else
                      dict(mode="closure", closure_min=0.8, steps=int(C('STABLE_STEPS')), finger_qd_max=0.5))),
    safety=dict(groups={g: sph(g) for g in G.GROUP_NAMES}, vec_clamp=1.0, dist_clamp=[-0.1, 0.5], extra_boxes=[],
                # the bin the policy trained against (world = env frame): four walls around BOLT_XY, not workcell.json's
                training_boxes=([dict(lo=[float(c_ - s_ / 2) for c_, s_ in zip(c, sz)], hi=[float(c_ + s_ / 2) for c_, s_ in zip(c, sz)])
                                 for c, sz in G._bin_walls()] if lvl.bin_walls else [])),
    rod=dict(radius=float(C('BOLT_RADIUS')), half_len=float(C('BOLT_HALF_LEN'))),
    # not read by the runtime: the pre-grasp wrist pose the episode starts from (genesis_policy.pregrasp reproduces
    # env_cfg._nominal_wrist_general): wrist = g - R(qw) bolt_offset_xyz + standoff * R(qw) z, orientation qw * wrist_rot,
    # qw = roll(-tilt about z x axis) * yaw(hand_yaw); the planner (MoveTo) delivers the wrist there, then the policy runs.
    pregrasp=dict(wrist_rot_xyzw=[float(v) for v in C('WRIST_ROT')],
                  bolt_offset_xyz=[float(G.BOLT_OFFSET_XY[0]), float(G.BOLT_OFFSET_XY[1]), float(G.BOLT_REST_Z - C('WRIST_Z_CONTACT'))],
                  standoff_m=[float(lvl.standoff_lo), float(lvl.standoff_hi)],
                  wrist_roll=bool(lvl.wrist_roll),
                  # the torso pose the episode holds (WAIST_HELD); the orchestrator sets it before the pre-grasp
                  waist={j: float(v) for j, v in G.WAIST_HELD.items()}),
    # not read by the runtime: where the policy can pick (genesis_policy.selector reads it). A rod centre within
    # xy_core_m of rod_xy_centre on x and y with the hand yaw inside hand_yaw_core_rad, or within xy_edge_m with the
    # hand yaw anywhere in hand_yaw_rad; hand yaw = rod yaw + pinch, |pinch| <= pinch_angle_rad, the rod yaw measured
    # from yaw0_rad (mod pi for a lying rod, mod 2 pi for a leaning one, whose raised end sets the direction).
    workspace=dict(rod_xy_centre=[float(v) for v in G.BOLT_XY], xy_core_m=float(lvl.bolt_xy),
                   xy_edge_m=float(lvl.bolt_xy_edge or lvl.bolt_xy),
                   hand_yaw_rad=[float(lvl.bolt_yaw_lo), float(lvl.bolt_yaw_hi)],
                   hand_yaw_core_rad=([float(v) for v in lvl.bolt_yaw_core] if lvl.bolt_xy_edge > 0.0
                                      else [float(lvl.bolt_yaw_lo), float(lvl.bolt_yaw_hi)]),
                   pinch_angle_rad=float(lvl.pinch_angle), yaw0_rad=YAW0,
                   lying_max_deg=float(G.FILL_LYING_DEG), leaning_max_deg=float(G.FILL_TILT_MAX_DEG),
                   standing_min_deg=float(G.FILL_UPRIGHT_DEG), stand_penalty_m=float(G.FILL_STAND_PENALTY),
                   resting_gap_m=0.003, table_top_m=float(C('TABLE_TOP')),
                   bin_inner_m=[float(v) for v in lvl.bin.inner] if lvl.bin.inner else [],
                   bin_wall_height_m=float(lvl.bin.height) if lvl.bin.inner else 0.0),
    # not read by the runtime: what the policy was trained against
    training=dict(perception=dict(bias_xy_m=lvl.perception.bias_xy, bias_z_m=lvl.perception.bias_z, bias_rot_rad=lvl.perception.bias_rot,
                                  noise_m=[lvl.perception.noise_lo, lvl.perception.noise_hi],
                                  latency_steps=[lvl.perception.latency_lo, lvl.perception.latency_hi], dropout=lvl.perception.dropout),
                  hw_latch=bool(lvl.hw_latch), sense_est=bool(lvl.sense_est), tactile=bool(lvl.tactile),
                  table_top_m=float(C('TABLE_TOP')), robot_root_z_m=float(C('ROBOT_ROOT_Z'))),
)
if not lvl.hw_latch:
    print("WARNING: level", lvl_id, "latches on fingertip force; the runtime's closure latch is a different rule")

# ---------------------------------------------------------------- episode.npz: env 0's second episode, in the runtime's frames
if args.record > 0:
    agent_cfg = RslRlWujiReachGraspPPORunnerCfg(); agent_cfg.device = dev
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wenv, agent_cfg.to_dict(), log_dir=None, device=dev); runner.load(args.checkpoint_path)
    policy = runner.get_inference_policy(device=dev)
    term = u.action_manager._terms["pipeline"]
    b = G._rg(u)
    hand_order = [n for n in names if n.startswith("r_")]      # the articulation's order, named; the runtime maps by name
    hand_ids = [names.index(n) for n in hand_order]
    dt = 1.0 / C('CONTROL_HZ')

    # The recording is in the env frame, which is adam-moveit's planning frame `world`: the floor at z 0 and
    # lifting_Columns_base at (0, 0, ROBOT_ROOT_Z), axis-aligned, as genesis.urdf.xacro's world joint puts it.
    rp0 = (G._t(robot.data.root_pos_w)[0] - u.scene.env_origins[0]).cpu().numpy()
    rq0 = G._t(robot.data.root_quat_w)[0].cpu().numpy()
    assert np.allclose(rp0, [0.0, 0.0, C('ROBOT_ROOT_Z')], atol=1e-5) and np.allclose(np.abs(rq0), [0, 0, 0, 1], atol=1e-6), \
        f"robot root at {rp0} / {rq0}, not where adam-moveit's world joint puts it"

    rec = {k: [] for k in ("time", "step", "joint_pos", "joint_vel", "joint_effort", "rod_pos", "rod_quat_xyzw", "obs", "action",
                           "arm_targets", "hand_targets", "arm_targets_last", "phase", "done", "lift_dir",
                           "box_lo", "box_hi", "cap_p0", "cap_p1", "cap_r", "episode",
                           "sim_tau_streak", "sim_grip_torque", "sim_latched")}
    # The action term re-solves the IK on every physics substep (decimation 4) from the measured state; a
    # runtime that solves once per control step reproduces the FIRST substep's targets, so capture those.
    first = {"tgt": None}
    _apply = term.apply_actions

    def apply_and_capture():
        _apply()
        if first["tgt"] is None:
            first["tgt"] = G._t(robot.data.joint_pos_target)[0].cpu().numpy().copy()
    term.apply_actions = apply_and_capture
    c_ids = G._ids(u)
    seeds = {"target_seed_pos": [], "target_seed_quat_xyzw": [], "workspace_centre": []}
    obs = wenv.get_observations(); dones_seen = 0; k = 0
    with torch.inference_mode():
        while dones_seen < 1 + args.record:          # episode 1 warms up the buffers (zero quats at the first reset)
            act = policy(obs)
            if dones_seen >= 1:
                ep, eq = G._est_bolt(u)
                p, q = ep[0].cpu().numpy(), eq[0].cpu().numpy()     # Isaac Lab 3.0 quats are (x, y, z, w) already
                ld = b["_lift_dir"][0]
                rec["time"].append(k * dt); rec["step"].append(int(u.episode_length_buf[0])); k += 1
                rec["episode"].append(dones_seen - 1)
                rec["obs"].append(obs["policy"][0].cpu().numpy()); rec["action"].append(act[0].cpu().numpy())
                rec["joint_pos"].append(G._t(robot.data.joint_pos)[0].cpu().numpy())
                rec["joint_vel"].append(G._t(robot.data.joint_vel)[0].cpu().numpy())
                rec["joint_effort"].append(G._t(robot.data.applied_torque)[0].cpu().numpy())
                rec["rod_pos"].append(p); rec["rod_quat_xyzw"].append(q); rec["phase"].append(int(G.phase_of(u)[0]))
                rec["lift_dir"].append(ld.cpu().numpy())
                if k == 1:   # the IK target the action term seeded at the reset (the analytic start pose), before this step moves it
                    seeds["target_seed_pos"].append(term._target_pos[0].cpu().numpy().copy())
                    seeds["target_seed_quat_xyzw"].append(term._target_quat[0].cpu().numpy().copy())
                    seeds["workspace_centre"].append(b["_nom_pos"][0].cpu().numpy().copy())    # the workspace clamp's centre
                # the scene the safety observation measured this step (G._clearance's inputs)
                rp = np.zeros(3)
                rec["box_lo"].append(np.concatenate([c_ids["table_lo"].view(1, 3).cpu().numpy(), b["_obs_lo"][0].cpu().numpy()]) - rp)
                rec["box_hi"].append(np.concatenate([c_ids["table_hi"].view(1, 3).cpu().numpy(), b["_obs_hi"][0].cpu().numpy()]) - rp)
                rec["cap_p0"].append(b["_cap_p0"][0].cpu().numpy() - rp); rec["cap_p1"].append(b["_cap_p1"][0].cpu().numpy() - rp)
                rec["cap_r"].append(b["_cap_r"][0].cpu().numpy())
                # the phase machine's own latch inputs, for diagnosing a parity failure (not read by the test)
                rec["sim_tau_streak"].append(b["tau_streak"][0].cpu().numpy().copy())
                rec["sim_grip_torque"].append(G._grip_torque(u)[0].cpu().numpy().copy())
                rec["sim_latched"].append(bool(b["latched"][0]))
            first["tgt"] = None
            obs, rew, dones, extras = wenv.step(act)
            if dones_seen >= 1:
                tgt = G._t(robot.data.joint_pos_target)[0].cpu().numpy()
                rec["arm_targets"].append(first["tgt"][arm_ids]); rec["hand_targets"].append(first["tgt"][hand_ids])
                rec["arm_targets_last"].append(tgt[arm_ids])
                rec["done"].append(bool(dones[0]))
            if bool(dones[0]):
                dones_seen += 1; k = 0
    base = getattr(term, "_base_targets", None)
    if base is not None:
        base = G._t(base).squeeze(0)
        level["hand"]["base_targets"] = {n: float(base[names.index(n)]) for n in hand_order}
    np.savez(os.path.join(args.out, "episode.npz"), joint_names=np.array(names), hand_target_names=np.array(hand_order),
             **{k_: np.array(v) for k_, v in seeds.items()},
             arm_joint_names=np.array(list(C('RIGHT_ARM'))), **{k_: np.array(v) for k_, v in rec.items()})
    print(f"episode.npz: {len(rec['time'])} steps, {args.record} episode(s) of env 0, phases {sorted(set(rec['phase']))}")

# ---------------------------------------------------------------- MODEL.md: the architecture and the robot <-> policy contract
from genesis_rl.algos.rsl_rl.rsl_rl_ppo_cfg import RslRlWujiReachGraspPPORunnerCfg as _RunnerCfg  # noqa: E402
_rc = _RunnerCfg()
_alg = _rc.algorithm
_hidden = [sd[f"mlp.{l}.weight"].shape[0] for l in layers[:-1]]
_std = torch.exp(sd["distribution.std_param"]) if "distribution.std_param" in sd and float(sd["distribution.std_param"].min()) < 0 \
    else sd.get("distribution.std_param", torch.zeros(8))
_TERM_DOC = {
    "bolt_pos_rel": "rod position (camera estimate) minus wrist position, in the wrist frame (m)",
    "bolt_rot_rel": "rod orientation in the wrist frame, first two columns of R_wrist^T R_rod (6D)",
    "wrist_lin_vel": "wrist linear velocity, world frame (m/s): full-body J(q) x joint velocities",
    "wrist_ang_vel": "wrist angular velocity, world frame (rad/s)",
    "finger_q": "the 9 synergy joints (thumb 3, index 3, middle 3), rad",
    "finger_qd": "their velocities, rad/s",
    "last_action": "the previous step's actor output as given (before clamp and phase mask)",
    "tip_forces": ("flexion-chain abs. torque (thumb, index, middle) x %.0f N/Nm / %.0f N: the hand's drive current x torque constant"
                   % (C('GRIP_MIN_N') / G.GRIP_TAU_MIN, C('F_MAX'))) if lvl.effort_obs else
                  ("zeros (no fingertip sensors)" if not lvl.tactile else "fingertip force / F_MAX"),
    "contacts": ("flexion-chain abs. torque > %.2f N m, per finger (0/1)" % G.TAU_CONTACT) if lvl.effort_obs else
                ("zeros (no fingertip sensors)" if not lvl.tactile else "fingertip contact flags"),
    "ep_time": "step / episode_max_steps (%d)" % G.episode_max_steps(lvl.retry_max, lvl.pile.settle_in),
    "arm_q_norm": "the 7 right-arm joints normalised to [-1, 1] over the training limits",
    "tip_dist": "fingertip pad to the rod's surface (thumb, index, middle), m",
    "safety": "per proxy group (tips, palm, forearm, upper arm): the vector to the nearest obstacle in the wrist frame (4 x 3), then the 4 clearances (m)",
    "heap_map": "free radial distance around the rod per sector",
}
_rows = []
_off = 0
for _n, _k in obs_layout:
    _rows.append(f"| {_off}-{_off + _k - 1} | `{_n}` | {_k} | {_TERM_DOC.get(_n, '')} |")
    _off += _k
md = f"""# {os.path.basename(os.path.normpath(args.out))}: model card

Written by adam-rl `tools/export_policy.py` from `{args.checkpoint_path}` (iteration {int(ck.get('iter', -1))}),
level {lvl_id} `{lvl.name}`. Read with `level.yaml` (every constant the runtime needs) and `actor.onnx`.

## Architecture

| | |
|---|---|
| network | MLP, input {W} -> {' -> '.join(str(h) for h in _hidden)} -> {sd[f'mlp.{layers[-1]}.weight'].shape[0]}, ELU between layers, linear output |
| parameters | {sum(v.numel() for k_, v in sd.items() if k_.startswith('mlp.'))} |
| input normalisation | folded into `actor.onnx`: (obs - mean) / (std + 0.01), running statistics from training |
| output | the Gaussian policy's MEAN; the runtime clamps it to [-1, 1]. The training std ({', '.join(f'{float(x):.3f}' for x in _std)}) is not used on the robot |
| ONNX | input `obs` float32 [batch, {W}], output `action` float32 [batch, 8], opset 17 |
| critic | same shape to a scalar value, training only, not exported |
| training | PPO (rsl-rl), {_rc.num_steps_per_env} steps per env per update, {_alg.num_learning_epochs} epochs x {_alg.num_mini_batches} minibatches, lr {_alg.learning_rate:g} ({_alg.schedule}), gamma {_alg.gamma}, lambda {_alg.lam}, clip {_alg.clip_param}, entropy {_alg.entropy_coef} (training-run overrides in TRAINING_LOG.md) |

## Inputs: the observation, {W} floats, in this order

| index | term | size | on the robot |
|---|---|---|---|
""" + "\n".join(_rows) + f"""

## Outputs: the action, 8 floats in [-1, 1], every {1000 / C('CONTROL_HZ'):.1f} ms ({C('CONTROL_HZ')} Hz)

| index | meaning | scale |
|---|---|---|
| 0-2 | wrist translation step, world frame | x {float(lim.effective_translation) * 1000:.2f} mm, smoothed {lim.wrist_smoothing} |
| 3-5 | wrist rotation step, world frame (axis-angle) | x {float(lim.effective_rotation) * 1000:.1f} mrad (z x {lvl.yaw_rate_scale if lvl.free_grasp else 1.0}), smoothed {lim.wrist_smoothing} |
| 6 | thumb closure target | c = (a + 1) / 2 in [0, 1]: q = q_open + c (q_closed - q_open) on 3 joints, smoothed {lim.action_smoothing} |
| 7 | index + middle closure target | the same on 6 joints |

The policy only drives the grasp phase (and, at free-pre-lift levels, a {G.DRAG_STEPS}-step window after the settle).
The rest is the runtime's phase machine: the latch ({'torque' if lvl.hw_latch else 'force/closure'}: flexion-chain torque
> {G.GRIP_TAU_MIN} N m for {C('STABLE_STEPS')} steps, fingertips within {C('GRASP_REGION') * 100:.0f} cm of the rod, pinch within
{lvl.pinch_centre_tol * 1000:.0f} mm of its axis, the rod moving with the hand over {lvl.est_vel_window} steps), settle {G.SETTLE_STEPS} steps,
grip-check test lift, a {C('LIFT_HEIGHT') * 100:.0f} cm lift over {G.LIFT_STEPS} steps, a {G.HOLD_STEPS}-step hold, up to {lvl.retry_max} retries.
The wrist step goes through damped least-squares IK to the 7 arm joints; the 9 synergy joints follow the closures;
the other 11 hand joints are held (ring and little {'in a fist' if C('Q_FIXED_RIGHT')['r_pinky_pip'] > 1.0 else 'flat'}).

## What the robot must provide

- `/joint_states` for the body and the right hand: positions, velocities, and the hand's effort
  {'(drive current; the node needs effort_scale_nm_per_a, the torque constant)' if (lvl.hw_latch or lvl.effort_obs) else ''}.
- The rod pose estimate at {C('CONTROL_HZ')} Hz in the planning frame `world`; trained against bias up to
  {lvl.perception.bias_xy * 1000:.0f} mm / {math.degrees(lvl.perception.bias_rot):.0f} deg, noise {lvl.perception.noise_lo * 1000:.0f}-{lvl.perception.noise_hi * 1000:.0f} mm,
  latency {lvl.perception.latency_lo}-{lvl.perception.latency_hi} steps, dropout {lvl.perception.dropout * 100:.0f} %.
- The arm at the start pose (inside the workspace: {float(lim.workspace_z_min) * 100:.0f}-{float(lim.workspace_z_max) * 100:.0f} cm above the table,
  within {float(lim.workspace_radius_xy) * 100:.0f} cm of the rod-relative nominal wrist xy).
"""
with open(os.path.join(args.out, "MODEL.md"), "w") as f:
    f.write(md)
print(f"MODEL.md written to {args.out}")

try:
    import yaml
    with open(os.path.join(args.out, "level.yaml"), "w") as f:
        yaml.safe_dump(level, f, sort_keys=False)
except ImportError:
    with open(os.path.join(args.out, "level.yaml"), "w") as f:
        f.write("# JSON is valid YAML\n" + json.dumps(level, indent=2))
print(f"level.yaml written to {args.out}")
app.close()
