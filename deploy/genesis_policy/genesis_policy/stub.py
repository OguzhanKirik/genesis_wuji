"""A stand-in handoff: a zero-action actor.onnx of the right shape and a
level.yaml whose numbers come from adam-rl's source as read on 2026-10-08,
with `# FROM adam-rl` on every value the export must confirm or replace.

    ros2 run genesis_policy make_stub_policy.py --out /handoff/policy/stub

The runtime runs end to end on this; the wrist holds still, the hand goes to
half closure, the phase machine times out of the grasp and lifts.
"""
import os

import numpy as np

from .level import N_ACTION, load

LEVEL_TEMPLATE = """\
# genesis_policy level file. Schema documented in genesis_policy/README.md.
# STUB written by make_stub_policy.py. Every number below was read from
# adam-rl (genesis_rl/tasks/wuji_pipeline/env_cfg.py and
# wuji_reach_grasp/env_cfg.py as of 2026-10-08); a line marked FROM adam-rl is
# level-dependent or a hardware substitute and must come from the export.
schema: 1
source: "stub: adam-rl constants as read on 2026-10-08, no trained level"

control:
  rate_hz: 30                              # CONTROL_HZ

observation:
  # The policy's input, in order. The sum must equal the actor's input width
  # ({obs_dim} here). The task brief says 61; these 13 terms add up to 74 as
  # ObservationsCfg.PolicyCfg lists them. FROM adam-rl: the exported layout.
  layout:
    - {{name: bolt_pos_rel, size: 3}}
    - {{name: bolt_rot_rel, size: 6}}
    - {{name: wrist_lin_vel, size: 3}}
    - {{name: wrist_ang_vel, size: 3}}
    - {{name: finger_q, size: 9}}
    - {{name: finger_qd, size: 9}}
    - {{name: last_action, size: 8}}
    - {{name: tip_forces, size: 3}}
    - {{name: contacts, size: 3}}
    - {{name: ep_time, size: 1}}
    - {{name: arm_q_norm, size: 7}}
    - {{name: tip_dist, size: 3}}
    - {{name: safety, size: 16}}
  # ep_time = step / episode_max_steps, with episode_max_steps(retry_max) =
  # GRASP_MAX_STEPS + SCRIPTED_STEPS + retry_max * (GRASP_RETRY_STEPS + SCRIPTED_STEPS).
  episode_max_steps: 692                   # FROM adam-rl: 120 + 98 + 3 * (60 + 98) for retry_max 3
  tip_force_scale: 40.0                    # F_MAX, N
  # arm_q_norm and the IK null-space pull use the limits training saw: the
  # URDF's (Isaac's joint_pos_limits), WIDER than MoveIt's for shoulderPitch.
  arm_q_limits:                            # FROM adam-rl: robot.data.joint_pos_limits of RIGHT_ARM
    lo: [-3.613, -2.793, -2.583, -2.496, -2.670, -0.960, -0.960]
    hi: [2.042, 0.628, 2.583, 0.209, 2.670, 0.960, 0.960]

action:
  # LimitsCfg.effective_translation / effective_rotation at _rg_limits():
  # min(0.004, 0.10 / 30) and min(0.02, 0.5 / 30), metres and radians per step.
  effective_translation: 0.0033333333
  effective_rotation: 0.0166666667
  wrist_smoothing: 0.5                     # first-order, on the Cartesian residual
  action_smoothing: 0.3                    # first-order, on the finger closures
  yaw_rate_scale: 1.0                      # FROM adam-rl: RGLevel.yaw_rate_scale (0.25 at free-grasp levels)

ik:
  dls_damping: 0.05
  nullspace_gain: 0.30
  joint_margin: 0.05                       # rad inside every clip limit
  max_joint_velocity: 2.0                  # rad/s, per-step joint travel cap
  singularity_threshold: 0.0001            # on sqrt(det(J J^T))
  workspace:
    radius_xy: 0.15                        # m, around the rod-relative nominal wrist xy
    z_min: 0.02                            # m above the table top
    z_max: 0.35                            # _rg_limits()
    nominal_wrist_offset_xy: [-0.0474, -0.0901]   # WRIST_XY - BOLT_XY

wrist:
  link: wristRollRight                     # WRIST_BODY; r_wrist is an identity fixed joint off it
  offset:                                  # the policy's wrist frame in the link frame
    xyz: [0.0, 0.0, 0.0]
    rpy: [0.0, 0.0, 0.0]

arm:
  joints:                                  # RIGHT_ARM = right_arm_controller order
    - shoulderPitch_Right
    - shoulderRoll_Right
    - shoulderYaw_Right
    - elbow_Right
    - wristYaw_Right
    - wristPitch_Right
    - wristRoll_Right

hand:
  thumb_joints: [r_thumb_cmc_flex, r_thumb_mcp, r_thumb_ip]                  # THUMB_JOINTS, closure 0
  finger_joints: [r_index_finger_mcp_flex, r_index_finger_pip, r_index_finger_dip,
                  r_middle_finger_mcp_flex, r_middle_finger_pip, r_middle_finger_dip]   # FINGER_JOINTS, closure 1
  open:                                    # Q_OPEN = 0.40 x flexion limit
    r_thumb_cmc_flex: 0.5164
    r_thumb_mcp: 0.628
    r_thumb_ip: 0.628
    r_index_finger_mcp_flex: 0.628
    r_index_finger_pip: 0.8376
    r_index_finger_dip: 0.628
    r_middle_finger_mcp_flex: 0.628
    r_middle_finger_pip: 0.8376
    r_middle_finger_dip: 0.628
  closed:                                  # Q_CLOSED = 0.60 x flexion limit
    r_thumb_cmc_flex: 0.7746
    r_thumb_mcp: 0.942
    r_thumb_ip: 0.942
    r_index_finger_mcp_flex: 0.942
    r_index_finger_pip: 1.2564
    r_index_finger_dip: 0.942
    r_middle_finger_mcp_flex: 0.942
    r_middle_finger_pip: 1.2564
    r_middle_finger_dip: 0.942
  held:                                    # Q_FIXED_RIGHT: THUMB_ABD, abductions 0, ring and pinky at HELD_CURL 0
    r_thumb_cmc_abd: -0.85
    r_index_finger_mcp_abd: 0.0
    r_middle_finger_mcp_abd: 0.0
    r_ring_finger_mcp_abd: 0.0
    r_pinky_mcp_abd: 0.0
    r_ring_finger_mcp_flex: 0.0
    r_ring_finger_pip: 0.0
    r_ring_finger_dip: 0.0
    r_pinky_mcp_flex: 0.0
    r_pinky_pip: 0.0
    r_pinky_dip: 0.0
  tips:                                    # TIP_BODIES + TIP_OFFSETS: the fingertip pads
    thumb: {{link: r_thumb_distal, offset: [0.0, 0.0, -0.02978]}}
    index: {{link: r_index_finger_distal, offset: [0.0, 0.0, -0.02475]}}
    middle: {{link: r_middle_finger_distal, offset: [0.0, 0.0, -0.02475]}}

phase:
  settle_in: 0                             # FROM adam-rl: PileCfg.settle_in (25 at level 35); policy frozen at episode start
  grasp_max_steps: 120                     # GRASP_MAX_STEPS
  grasp_retry_steps: 60                    # GRASP_RETRY_STEPS
  settle_steps: 10                         # SETTLE_STEPS
  drag_steps: 10                           # DRAG_STEPS (the policy's window when free_prelift)
  free_prelift: true                       # FROM adam-rl: RGLevel.free_prelift of the exported level
  grip_check: true                         # FROM adam-rl: RGLevel.grip_check
  grip_check_steps: 8                      # GRIP_CHECK_STEPS
  grip_test_total_m: 0.024                 # GRIP_TEST_TOTAL_M
  grip_check_ratio: 0.4                    # GRIP_CHECK_RATIO
  grip_check_min_tips_m: 0.004             # GRIP_CHECK_MIN_TIPS
  grip_check_late: false                   # FROM adam-rl: RGLevel.grip_check_late (true from level 34)
  lift_steps: 40                           # LIFT_STEPS
  lift_slow_steps: 10                      # LIFT_SLOW_STEPS
  lift_slow_step_m: 0.001                  # LIFT_SLOW_STEP_M
  lift_height_m: 0.10                      # LIFT_HEIGHT
  hold_steps: 30                           # HOLD_STEPS
  retry_max: 3                             # FROM adam-rl: RGLevel.retry_max
  success_lift_m: 0.05                     # SUCCESS_LIFT, rod rise that counts as up
  up_threshold_m: 0.02                     # UP_THRESHOLD, rod rise that counts as lifted at all
  lift_dir: [0.0, 0.0, 1.0]                # FROM adam-rl: _lift_dir (the rod's up-normal when the wrist is rolled)
  # The hardware latch (the simulator's grasp_ready reads contact forces):
  # both closures past closure_min and the synergy joints slower than
  # finger_qd_max for `steps` consecutive steps end the grasp phase.
  latch:                                   # FROM adam-rl: a rule the transfer level can be trained against
    closure_min: 0.8
    steps: 4                               # STABLE_STEPS
    finger_qd_max: 0.5                     # rad/s

safety:
  # PROXY_GROUPS: [link, offset in the link frame, radius], per group.
  groups:
    tips:
      - [r_thumb_distal, [0.0, 0.0, -0.02378], 0.006]
      - [r_index_finger_distal, [0.0, 0.0, -0.01875], 0.006]
      - [r_middle_finger_distal, [0.0, 0.0, -0.01875], 0.006]
      - [r_ring_finger_distal, [0.0, 0.0, -0.01875], 0.006]
      - [r_pinky_distal, [0.0, 0.0, -0.01875], 0.006]
      - [r_thumb_distal, [0.0, 0.0, -0.01489], 0.007]
      - [r_index_finger_distal, [0.0, 0.0, -0.012375], 0.007]
      - [r_middle_finger_distal, [0.0, 0.0, -0.012375], 0.007]
      - [r_ring_finger_distal, [0.0, 0.0, -0.012375], 0.007]
      - [r_pinky_distal, [0.0, 0.0, -0.012375], 0.007]
      - [r_thumb_middle, [0.0, 0.0, -0.02], 0.008]
      - [r_index_finger_middle, [0.0, 0.0, -0.02], 0.008]
      - [r_middle_finger_middle, [0.0, 0.0, -0.02], 0.008]
    palm:
      - [wristRollRight, [-0.025, 0.0, -0.010], 0.020]
      - [wristRollRight, [-0.025, 0.0, -0.040], 0.020]
      - [wristRollRight, [-0.025, 0.0, -0.070], 0.020]
      - [wristRollRight, [0.010, 0.0, -0.010], 0.020]
      - [wristRollRight, [0.010, 0.0, -0.040], 0.020]
      - [wristRollRight, [0.010, 0.0, -0.070], 0.020]
      - [wristPitchRight, [0.0, 0.0, 0.0], 0.030]
    forearm:
      - [elbowRight, [-0.013, -0.029, -0.031], 0.045]
      - [wristYawRight, [0.0, 0.0, -0.020], 0.048]
      - [wristYawRight, [0.0, 0.0, -0.095], 0.048]
      - [wristYawRight, [0.0, 0.0, -0.170], 0.048]
    upperarm:
      - [shoulderYawRight, [0.001, -0.003, -0.020], 0.050]
      - [shoulderYawRight, [0.001, -0.003, -0.120], 0.050]
      - [shoulderYawRight, [0.001, -0.003, -0.220], 0.050]
      - [shoulderRollRight, [-0.033, -0.017, -0.010], 0.065]
  vec_clamp: 1.0                           # safety_obs clamps the wrist-frame vectors to [-1, 1]
  dist_clamp: [-0.1, 0.5]                  # and the clearances to [-0.1, 0.5]
  # Boxes besides the table and the bin from the workcell file ({{lo: [x,y,z], hi: [x,y,z]}} in world).
  extra_boxes: []

rod:
  radius: 0.0125                           # BOLT_RADIUS
  half_len: 0.0175                         # BOLT_HALF_LEN
pregrasp:                                  # env_cfg._nominal_wrist_general constants (genesis_policy.pregrasp)
  wrist_rot_xyzw: [-0.1821, -0.7296, 0.4661, 0.4661]   # WRIST_ROT
  bolt_offset_xyz: [0.0474, 0.0901, -0.0482]           # BOLT_OFFSET_XY, BOLT_REST_Z - WRIST_Z_CONTACT
  standoff_m: [0.05, 0.10]
  wrist_roll: true
"""


def write_actor(path, obs_dim):
    """A zero-weight MLP: action = tanh(obs @ 0 + 0) = 0. Needs the onnx package."""
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as e:
        raise RuntimeError('the onnx package is needed to write the stub actor (pip install onnx)') from e
    w = numpy_helper.from_array(np.zeros((obs_dim, N_ACTION), dtype=np.float32), name='W')
    b = numpy_helper.from_array(np.zeros((N_ACTION,), dtype=np.float32), name='b')
    graph = helper.make_graph(
        [helper.make_node('MatMul', ['obs', 'W'], ['h']),
         helper.make_node('Add', ['h', 'b'], ['pre']),
         helper.make_node('Tanh', ['pre'], ['action'])],
        'stub_actor',
        [helper.make_tensor_value_info('obs', TensorProto.FLOAT, [1, obs_dim])],
        [helper.make_tensor_value_info('action', TensorProto.FLOAT, [1, N_ACTION])],
        initializer=[w, b])
    model = helper.make_model(graph, producer_name='genesis_policy.stub',
                              opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def write_level(path):
    level_text = LEVEL_TEMPLATE.format(obs_dim='(see layout)')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(level_text)
    return load(path)


def write_stub(out_dir, with_actor=True):
    """Write level.yaml (and actor.onnx) into out_dir; returns the Level."""
    os.makedirs(out_dir, exist_ok=True)
    level = write_level(os.path.join(out_dir, 'level.yaml'))
    if with_actor:
        write_actor(os.path.join(out_dir, 'actor.onnx'), level.layout.size)
    return level
