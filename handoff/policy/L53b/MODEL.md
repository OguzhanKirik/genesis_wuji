# L53b: model card

Written by adam-rl `tools/export_policy.py` from `logs/rsl_rl/adam_u30_wuji_reach_grasp_L53/2026-10-18_21-05-22/model_21550.pt` (iteration 21550),
level 53 `bin_highest_low_walls`. Read with `level.yaml` (every constant the runtime needs) and `actor.onnx`.

## Architecture

| | |
|---|---|
| network | MLP, input 74 -> 256 -> 256 -> 128 -> 8, ELU between layers, linear output |
| parameters | 118920 |
| input normalisation | folded into `actor.onnx`: (obs - mean) / (std + 0.01), running statistics from training |
| output | the Gaussian policy's MEAN; the runtime clamps it to [-1, 1]. The training std (0.149, 0.123, 0.146, 0.133, 0.156, 0.152, 0.149, 0.124) is not used on the robot |
| ONNX | input `obs` float32 [batch, 74], output `action` float32 [batch, 8], opset 17 |
| critic | same shape to a scalar value, training only, not exported |
| training | PPO (rsl-rl), 24 steps per env per update, 5 epochs x 4 minibatches, lr 0.0003 (fixed), gamma 0.99, lambda 0.95, clip 0.2, entropy 0.003 (training-run overrides in TRAINING_LOG.md) |

## Inputs: the observation, 74 floats, in this order

| index | term | size | on the robot |
|---|---|---|---|
| 0-2 | `bolt_pos_rel` | 3 | rod position (camera estimate) minus wrist position, in the wrist frame (m) |
| 3-8 | `bolt_rot_rel` | 6 | rod orientation in the wrist frame, first two columns of R_wrist^T R_rod (6D) |
| 9-11 | `wrist_lin_vel` | 3 | wrist linear velocity, world frame (m/s): full-body J(q) x joint velocities |
| 12-14 | `wrist_ang_vel` | 3 | wrist angular velocity, world frame (rad/s) |
| 15-23 | `finger_q` | 9 | the 9 synergy joints (thumb 3, index 3, middle 3), rad |
| 24-32 | `finger_qd` | 9 | their velocities, rad/s |
| 33-40 | `last_action` | 8 | the previous step's actor output as given (before clamp and phase mask) |
| 41-43 | `tip_forces` | 3 | flexion-chain abs. torque (thumb, index, middle) x 20 N/Nm / 40 N: the hand's drive current x torque constant |
| 44-46 | `contacts` | 3 | flexion-chain abs. torque > 0.10 N m, per finger (0/1) |
| 47-47 | `ep_time` | 1 | step / episode_max_steps (707) |
| 48-54 | `arm_q_norm` | 7 | the 7 right-arm joints normalised to [-1, 1] over the training limits |
| 55-57 | `tip_dist` | 3 | fingertip pad to the rod's surface (thumb, index, middle), m |
| 58-73 | `safety` | 16 | per proxy group (tips, palm, forearm, upper arm): the vector to the nearest obstacle in the wrist frame (4 x 3), then the 4 clearances (m) |

## Outputs: the action, 8 floats in [-1, 1], every 33.3 ms (30 Hz)

| index | meaning | scale |
|---|---|---|
| 0-2 | wrist translation step, world frame | x 3.33 mm, smoothed 0.5 |
| 3-5 | wrist rotation step, world frame (axis-angle) | x 16.7 mrad (z x 0.25), smoothed 0.5 |
| 6 | thumb closure target | c = (a + 1) / 2 in [0, 1]: q = q_open + c (q_closed - q_open) on 3 joints, smoothed 0.3 |
| 7 | index + middle closure target | the same on 6 joints |

The policy only drives the grasp phase (and, at free-pre-lift levels, a 10-step window after the settle).
The rest is the runtime's phase machine: the latch (torque: flexion-chain torque
> 0.2 N m for 4 steps, fingertips within 5 cm of the rod, pinch within
10 mm of its axis, the rod moving with the hand over 5 steps), settle 10 steps,
grip-check test lift, a 10 cm lift over 40 steps, a 30-step hold, up to 3 retries.
The wrist step goes through damped least-squares IK to the 7 arm joints; the 9 synergy joints follow the closures;
the other 11 hand joints are held (ring and little flat).

## What the robot must provide

- `/joint_states` for the body and the right hand: positions, velocities, and the hand's effort
  (drive current; the node needs effort_scale_nm_per_a, the torque constant).
- The rod pose estimate at 30 Hz in the planning frame `world`; trained against bias up to
  8 mm / 5 deg, noise 1-3 mm,
  latency 1-4 steps, dropout 3 %.
- The arm at the start pose (inside the workspace: 2-35 cm above the table,
  within 15 cm of the rod-relative nominal wrist xy).
