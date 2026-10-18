# Sim-to-real plan for the reach-to-grasp policy

Written 2026-10-08, after the Experiment 4 ladder reached level 33 (pile, free
grasp, retries). This is the plan for running that policy on the real ADAM
upper body with the Wuji Hand 2, built on the `~/adam-moveit` workspace. It
is a plan, not a record: nothing below has been built or run yet. Status of
the ladder itself is in `TRAINING_LOG.md` and `EXPERIMENT4_STEPS.md`.

## 1. What the policy needs

The policy (`genesis_rl/tasks/wuji_reach_grasp/env_cfg.py`) runs at
`CONTROL_HZ = 30` (physics 120 Hz, decimation 4). Per step it emits 8 numbers:

- 6: a wrist Cartesian step (3 translation, 3 rotation), scaled by
  `effective_translation` / `effective_rotation`, low-pass filtered by
  `wrist_smoothing`, and turned into 7 right-arm joint targets by the
  Jacobian IK in `ReachGraspAction` (line ~2399). The waist is held at the
  level's triple; the arm targets are clipped to the joint limits.
- 2: thumb and finger closures in [0, 1], low-pass filtered by
  `action_smoothing`, mapped linearly between the open and closed synergy
  poses onto 9 hand joints: `THUMB_JOINTS` (cmc_flex, mcp, ip) and
  `FINGER_JOINTS` (index and middle mcp_flex, pip, dip). Thumb abduction,
  all mcp_abd joints, ring and pinky are held (`THUMB_ABD`, `HELD_CURL`
  in `wuji_pipeline/env_cfg.py`).

Around the policy sits a scripted phase machine (settle, 24 mm test lift,
grip check at the fingertips, up to three retries, lift). On hardware the
phase machine runs in the same process as the policy; it is not learned.

Observations (`ObservationsCfg.PolicyCfg`, line ~2744), 74 values, and
where each one comes from on hardware:

| Term | Size | Hardware source | Exists in adam-moveit? |
|---|---|---|---|
| `bolt_pos_rel`, `bolt_rot_rel` | 3 + 6 | camera + rod pose estimator, in the wrist frame | no; only the `/perception/bolt_pose` topic contract in `genesis_cell/config/cell.yaml` |
| `wrist_lin_vel`, `wrist_ang_vel` | 3 + 3 | finite difference of arm FK at 30 Hz, filtered | computable |
| `finger_q`, `finger_qd` | 9 + 9 | `/hand_right/joint_states` (velocity must be differenced; the driver publishes position and current only) | plugin written, never run on the hand |
| `last_action` | 8 | the node itself | trivial |
| `tip_forces` | 3 | fingertip normal force sensors | unknown whether the mounted hand has them |
| `contacts` | 3 | same | same |
| `ep_time` | 1 | the node itself | trivial |
| `arm_q_norm` | 7 | ros2_control joint states | yes |
| `tip_dist` | 3 | fingertip FK + estimated rod pose | computable |
| `safety` | 16 | nearest-obstacle vectors and clearance per group from FK + the known cell geometry | computable from `cell.yaml` (table, bin) |

The perception randomisation the policy trained with (`_PERC7`: bias 5 mm
xy / 3 mm z / 3°, noise 1-2 mm, latency 1-2 steps, dropout 2 %) is the
accuracy budget for the real estimator. The physics randomisation
(`_PHYS5`: mass 0.8-1.2, friction 0.7-1.3, gains 0.9-1.1) is the budget for
the real rod and the real hand.

## 2. What the adam-moveit workspace provides

Survey of `/home/revel/adam-moveit` on 2026-10-08.

**Body (arm + waist).** `genesis_driver/GenesisSystem`
(`ws/src/genesis_driver/src/genesis_system.cpp`) is a ros2_control hardware
interface that publishes `pnd_adam/LowCmd` on `/lowcmd` and reads
`/lowstate`. `controller_manager` runs at 400 Hz. It seeds commands from the
measured pose, refuses to activate without a LowState, withholds commands
when state is older than 100 ms, caps each cycle's step at 0.2 rad, and
clamps to the ros2_control limits. `MotorCmd.tau` is hard-coded 0, so there
is no gravity feedforward and the arms droop about 0.04 rad. The plugin has
run in Isaac and on the loopback; the docs still list the first real-robot
execution as not done.

Controllers (`ws/src/genesis_robot_config/config/ros2_control/ros2_controllers_genesis_wuji.yaml`),
all `JointTrajectoryController`:

- `right_arm_controller`: `shoulderPitch_Right, shoulderRoll_Right,
  shoulderYaw_Right, elbow_Right, wristYaw_Right, wristPitch_Right,
  wristRoll_Right` (position + velocity command).
- `waist_controller`: `waistRoll, waistPitch, waistYaw`.
- `right_hand_controller`: the 20 Wuji joints, position only, in exactly
  the order this repo's `tools/manus_wuji_bridge.py` uses.

There is no streaming controller (no forward_command_controller, no
moveit_servo). The only way in today is `FollowJointTrajectory` or the
controllers' `~/joint_trajectory` topic.

**Wuji hand.** `genesis_wuji_hand/WujiHandSystem`
(`ws/src/genesis_wuji_hand/src/wuji_hand_system.cpp`) publishes
`/hand_right/joint_commands` (always all 20 joints, radians) and subscribes
`/hand_right/joint_states` (position rad, effort A). The vendor driver
`wujihandros2` (USB, state at 1000 Hz) is not installed in the image, and
neither compose file passes USB devices. The design doc
(`docs/genesis_wuji_design.md`) lists as not done: the hull probe, the
Isaac end-to-end run, and checking every finger's sign and zero on the real
hand. Fingertip contact sensing is listed as UNKNOWN.

**Perception.** None on the real robot. Cameras exist only in Isaac
(`ws/src/genesis_isaac/scripts/isaac_runner.py`). No hand-eye or head-camera
calibration exists; `genesis_metrology/config/mounts.yaml` is all
placeholders. The table and bin collision objects are published only by the
sim scene script.

**Safety.** MoveIt velocity/acceleration scaling (0.1 default, 0.3 in the
skill servers), stale-state refusal, the driver protections above, planning
scene collision checks, the arm wall, a `/stop` action, and real-mode launch
gates (confirmation, NUC ping, single `/lowcmd` publisher) in
`ws/src/genesis_bringup/launch/stack.launch.py`. No software e-stop, no
force or torque limit anywhere. The ros2_control clamp for
`shoulderPitch_Right` (min -3.61) is wider than the firmware limit MoveIt
uses (-1.047), so only the planner enforces the firmware value.

**Containers.** The Isaac container (this repo) has no ROS; the MoveIt
container has no Isaac. `tools/glove_link.py` already bridges the two over
UDP for glove teleop. Real mode uses Fast DDS on the robot domain
(`scripts/container_entrypoint.sh`); domain 9 belongs to adam-project and
must not be used.

## 3. Plan

**Where the code lives.** No separate sim-to-real container, and no copies of
adam-moveit packages. The body driver, the Wuji plugin, the controllers, the
URDF and the launch gates are the robot's wire contract and must stay in one
place; a copy would fork them and bypass the single-publisher gate on
`/lowcmd`. The policy node is a normal ROS 2 package inside the adam-moveit
workspace, built into the same image (one extra pip dependency,
onnxruntime). adam-rl trains and exports; the artefact handed over is an ONNX
actor plus a YAML of level constants, via the `./handoff` mount that
`docker-compose.robot.yml` already has. Perception is the one candidate for
its own container: `genesis_cell` already defines it as an external
`PoseStamped` source on `/perception/bolt_pose`, so a camera container with
its own SDK and model dependencies fits without touching the control stack.


Steps 1-2 are in this repo, 3-6 in adam-moveit. Steps 1, 2 and 4 are
independent and can run in parallel.

### 3.1 Sim-free runtime (adam-rl)

New module `genesis_rl/deploy/runtime.py` that reproduces `ReachGraspAction`
and the observation assembly without Isaac:

- arm FK and Jacobian for `right_upper_body` from the URDF (pinocchio),
  matching the IK in the action term, including the joint-limit clip;
- the phase machine (settle, test lift, grip check, retries, lift) with
  the same thresholds as the level config;
- the closure to 20-joint hand command, in `right_hand_controller` order,
  held joints filled from `THUMB_ABD` / `HELD_CURL`;
- the 74-value observation from joint states, an estimated rod pose and
  the cell geometry (`safety`, `tip_dist`).

New script `tools/export_policy.py`: load an rsl_rl checkpoint, export the
actor to ONNX with the observation normaliser folded in.

New test `tools/test_runtime_parity.py`: record one Isaac episode (joint
states, estimated rod pose, policy outputs, joint targets) and replay it
through the runtime; assert identical joint targets to float tolerance.
This test is the deliverable of the step. Without it every later failure
on hardware is ambiguous.

### 3.2 Transfer level (adam-rl)

One more curriculum level, trained before export, whose observation set is
what hardware can provide:

- `tip_forces` and `contacts` either dropped, or replaced by a proxy from
  the hand driver's effort (motor current) field; decide after step 3.4
  shows what the hand actually publishes;
- grip check driven by the estimated rod pose (perception) instead of
  ground truth, so the retry logic works with the same signal on hardware;
- wider latency (up to 4 steps) and gain randomisation, lower force cap,
  and the collision gate held at 0.

No new scene variants. The pick from this level is the one that gets
exported.

### 3.3 Policy node (adam-moveit)

New package `ws/src/genesis_policy` with one node:

- subscribes `/joint_states` (body + hand) and `/perception/bolt_pose`;
- runs the runtime from 3.1 at 30 Hz with the ONNX actor (onnxruntime,
  CPU is enough for the MLP);
- streams targets. First version: publish single-point
  `JointTrajectory` messages to `/right_arm_controller/joint_trajectory`
  and `/right_hand_controller/joint_trajectory` with time_from_start of
  about 60 ms, every policy tick. The JTC replaces its trajectory on each
  message and the driver's watchdog, slew cap and clamp stay in the loop.
  If the spline interpolation visibly lags the policy, add a
  `forward_command_controller` for the right arm and switch to it with
  `controller_manager` only while the policy is active (it claims the same
  position interfaces as the JTC, so the two cannot be active together);
- safety wrapper: a Cartesian fence around the bin, a per-tick step cap
  equal to the action term's `effective_translation`, clamping to the MoveIt
  joint limits (`config/moveit/joint_limits_genesis_wuji.yaml`, not the
  ros2_control ones), an abort on hand current or joint-limit proximity, a
  dead-man key, and cancel through the existing `/stop` action.

Template for sending a trajectory: `scripts/verify_hardware.py --execute`
(stage 6).

### 3.4 Hand bring-up (adam-moveit, bench)

- install `ros-humble-wujihand` in the image; add the USB device and udev
  rule to `docker-compose.yml` and `docker-compose.robot.yml`;
- run the vendor sign and zero check per finger against the URDF (home,
  open, closed), as `docs/genesis_wuji_design.md` lists;
- verify the sim's open and closed closure endpoints give the same
  fingertip gap on the real hand (`tools/manus_wuji_bridge.py` already maps
  the joint order; the synergy endpoints are in `wuji_pipeline/env_cfg.py`);
- record what `/hand_right/joint_states` effort looks like during a pinch
  on a rod, for step 3.2.

### 3.5 Perception (adam-moveit)

Staged, so the first grasps do not wait for it:

1. no camera: one rod at a measured pose on the table, published as a
   constant `/perception/bolt_pose`;
2. a fixed RealSense over the bin, hand-eye calibrated (nothing exists;
   `genesis_metrology` has the ChArUco tooling that can be reused),
   publishing the rod pose from a depth segmentation and cylinder fit;
   target accuracy 5 mm / 3° (the `_PERC7` bias budget), 30 Hz, latency
   under 2 policy steps;
3. the same estimator on a rod in the bin, then on the top rod of a pile.

The estimated pose must be expressed in the wrist frame by the policy
node, using the arm FK, exactly as `bolt_pos_in_wrist` does in sim.

### 3.6 Test ladder (adam-moveit)

Each rung reuses the node from 3.3 unchanged.

1. **Loopback.** Node against `ws/src/genesis_driver/scripts/loopback_backend.py`:
   joint order, rates, topic plumbing.
2. **Isaac through ROS.** The MoveIt workspace's Isaac backend with
   `--hands wuji`, so the policy runs through the real 400 Hz control stack
   in sim. Compare success against `tools/eval_reach_grasp.py` on the same
   scene. This is the only place the JTC interpolation and the
   command-to-state latency can be measured before hardware.
3. **Arm only, hand open, no object.** First replay a recorded sim
   trajectory, then the policy with closure forced to zero and the fence on.
4. **Rod on the table at a known pose.** First full grasps, one rod,
   retries enabled.
5. **Camera, then bin, then pile.** Only the pose source changes.

## 4. Known blockers outside the policy

Recorded in adam-moveit's `docs/open_items.md`, `docs/notion_page.md`,
`todo.md`:

- the body driver has never executed against the real robot;
- no gravity feedforward (`tau = 0`), about 0.04 rad droop;
- the PND `vel_limit: 120` has no stated unit; if motor-side, the joint
  speed limit is about 2.4 rad/s, a third of the URDF value;
- `LowCmd` has no timestamp, so command-to-feedback latency cannot be
  measured on the real robot;
- Isaac stalls of about 250 ms every 38 s trip the watchdog in the
  Isaac-through-ROS rung;
- the lifting column is held by the mock system; the head camera transform
  is a placeholder.

All of these affect rung 3 and are independent of the policy work.

## 5. The transfer level, as done (2026-10-09) — the procedure to repeat or improve

Decision: stop adding scene complexity and transfer the last policy that passes the gate.
Under the corrected evaluation (per-env quota, 2048 episodes; see TRAINING_LOG.md,
"Evaluation stopping rule") that is **level 32, `model_13700.pt`** at 95.3 %. Levels 33–38
(mixed scenes, the heap) are parked with their traces. The ROS side (section 3.3 onward)
comes later; this section is the simulator side.

**Step 1 — fix the yardstick first.** Re-score the candidate with the quota rule before
anything else; every earlier number was inflated by the stop-at-N-completions rule.

**Step 2 — list what the policy and the phase machine read that the robot cannot measure**,
and replace each with its measurable counterpart behind a level flag (all in
`genesis_rl/tasks/wuji_reach_grasp/env_cfg.py`, levels 0–38 unchanged):

| simulator signal | used by | replacement on the robot | flag |
|---|---|---|---|
| fingertip force sensors (`tip_forces`, `contacts` observations) | the policy | none on the Wuji hand (open question): the two observations read zero, the interface stays 74 wide | `tactile=False` |
| fingertip force ≥ 4 N on both sides, contact streaks | the latch | finger joint torque per flexion chain (`_grip_torque`, the simulator's `applied_torque`), which is motor current on the robot; streaked the same way; threshold `GRIP_TAU_MIN` calibrated against 4 N | `hw_latch=True` |
| true rod pose in the latch's inside / centred tests | the latch | the estimated pose (`fingertip_distances_est`, `pinch_centre_dist_est`) | `hw_latch=True` |
| true rod velocity (settled test) | the latch | the estimate's finite-difference velocity | `hw_latch=True` |
| true rod height (grip check, the retry's "lifted" test, the drop test) | the phase machine | the estimated height against the estimate's own rest height read at the first step | `sense_est=True` |
| success (held ≥ 5 cm through the hold) | the evaluation only | stays on the true rod: it is the yardstick, not something the robot computes | — |

**Step 3 — widen the randomisation to what the robot will show**: `_PERC8` (bias 8 mm xy / 5 mm
z / 5°, noise 1–3 mm, latency 1–4 control steps, dropout 3 %; the estimate history is 5 deep,
`HIST_DEPTH`) and `_PHYS6` (gains 0.8–1.2 on the arm as well as the fingers, `arm_gains`).

**Step 4 — calibrate the torque threshold** (`scratchpad/calib_torque.py`): run the level-32
pick, log fingertip force against flexion torque per finger during the grasp phase, and set
`GRIP_TAU_MIN` where the torque test fires on most 4 N holds and on few free fingers. Record
the numbers in TRAINING_LOG.md under level 39.

**Step 5 — train level 39** from `model_13700.pt`: `bash logs/chains/chain_L39.sh` (tests, audit,
the pick untrained at 39, 600 iterations at 2048 envs with `--reset-std 0.3`, the last three
checkpoints evaluated with the quota rule at 2048 episodes, the pick, the scene rows). The
untrained number says how much the signal swap alone costs; the pick says what training wins
back.

**Step 6 — export**: `tools/export_policy.py` writes `actor.onnx` (74 → 8, normaliser folded in),
`level.yaml` (every constant the runtime needs, including `GRIP_TAU_MIN`, the synergy endpoints
and the phase thresholds) and `episode.npz` (one recorded episode for the parity test). These
go to adam-moveit's `handoff/policy/<name>/` (section 3.3).

**To improve it later**: if the hand turns out to have fingertip sensors, drop `tactile=False`
and keep the rest; if the real motor current is noisier than the simulator's torque, raise
`STABLE_STEPS` for the torque streak rather than the threshold; if the camera's latency is
measured above 4 steps, raise `HIST_DEPTH` and `latency_hi` together.

## 6. Parity, as done (2026-10-13)

Step 3.1 lives in adam-moveit, not here: `ws/src/genesis_policy` (runtime, model, level loader, node;
uncommitted there). Its `test/test_parity.py` now passes against this repo's recording: the level-32 pick
exported at level 39 (`tools/export_policy.py --record 8`), 8 consecutive episodes of env 0, 2002 steps,
covering first-try latches, one to four retries and a time-out. Actions match to 1e-4, every observation
term to 1e-3 (the wrist velocities to 5e-3), arm and hand targets to 1e-4.

**The export changed** (`tools/export_policy.py`): `level.yaml` is now the runtime's schema 1 with the
values read from the env (it was a different, unread schema); `episode.npz` is recorded in adam-moveit's
planning frame `world` (the env frame: lifting_Columns_base at z 0.05025, as the xacro's world joint puts
it), skips the first episode (zero quats at the first reset), and carries the effort, the per-episode IK
seed, workspace centre and lift direction, the per-step obstacle scene, and the FIRST physics substep's
joint targets.

**What the replay found** — each fixed on the runtime side unless marked:

| difference | effect | now |
|---|---|---|
| Isaac Lab 3.0 quaternions are (x, y, z, w); the old recorder reordered them as if (w, x, y, z) | rod rotation scrambled | recorder writes them as they are |
| `last_action` is `mdp.last_action`: the actor output before the clamp and the phase mask | the runtime fed back the clipped, masked action | runtime feeds back the raw output |
| the wrist velocities are PhysX body velocities; the waist moves under its PD hold | finite-differenced FK was ~55 % off; arm-only J qdot also wrong | `wrist_velocity: jacobian`, full-body J times every reported joint velocity (node default) |
| the latch at level 39 is the torque latch | the runtime had only the closure rule | `latch.mode: torque` ported from `grasp_ready`; node refuses to start without `effort_scale_nm_per_a` |
| the safety scene is per episode (bin around the rod, cover blocks, neighbour rods as capsules) | the runtime measured a fixed bin from `workcell.json` (and, mirrored, the wrong one) | `Runtime.set_scene(boxes, capsules)`; **the node still uses the fixed table and bin** |
| `_lift_dir` is per episode (a tilted rod with the wrist rolled lifts along its up-normal) | runtime lifted along world z | `Runtime.reset(lift_dir=)`; **the node does not supply it** |
| the IK target is seeded at the reset with the analytic start pose; the workspace clamp is centred on the analytic nominal wrist xy | a few mm / ~17 mrad start offset; a different clamp edge | `target_seed`, `workspace_centre` for the replay; the robot keeps the measured pose and the rod-relative centre |
| the action term re-solves the IK on every physics substep (120 Hz) from the measured state | the runtime solves once per policy step | **not reproducible**; the test compares against the first substep. On hardware the node could run the IK faster than the policy |
| sense_est: the retry's "not lifted at the lift's end" test (`lift_end & ~up`) reads `up` from the TRUE rod height (computed before the estimate replaces `z`) | one phase-machine decision at level 39 still uses ground truth | **simulator, not fixed** (left alone while level 39 trains; the runtime uses the estimate, as the robot must) |

**Added the same evening** (adam-moveit, 41 tests):
- `lift_dir_from_rod`: the node computes the episode's lift axis from the rod estimate at the start (a
  rod leaning 10-60 deg: its up-normal, capped at `phase.roll_cap`, now exported; otherwise world z).
  Against the 8 recorded episodes it classifies all eight right (5 leaning, 1 upright, 2 lying) and
  matches the simulator's axis to 0.4-4.2 deg, inside the 5 deg perception bias.
- **The training bin is not the real cell's bin.** adam-rl's virtual bin is 26 x 26 cm inside, 4 cm
  walls, centred on the nominal rod (0.547, -0.110); `workcell.json`'s is 22 x 16 cm, 8 cm walls, at
  (0.55, -0.20) mirrored. The table matches. The export now carries the training bin
  (`safety.training_boxes`) and the node has `safety_scene: workcell | training` (default workcell: the
  real furniture). Either the real bin is built to the training size and place, or a level is trained
  with the real bin; until then the safety observation on the robot is out of distribution near the walls.

**Still needed before the robot** (adam-moveit): the Wuji drives' torque constant (A to N m) for
`effort_scale_nm_per_a`, measured at bring-up (3.4); neighbour rods for the safety term in a pile (perception), and the bin
decision above; the image rebuilt with `pin` / `onnxruntime` (the current `adam-moveit:latest`
has neither); `colcon build` and the launch_testing suite.

The untrained level-32 pick scores **51.1 %** at level 39 (the signal swap alone, against 95.3 % at level
32); the trained pick's numbers go in TRAINING_LOG.md under level 39.

## 7. Grasping on robot-measurable signals (2026-10-13, night)

Lift-and-hold success, quota rule, 2048 episodes (details in TRAINING_LOG.md, levels 39-43):

| policy | level | success |
|---|---|---|
| level-32 pick, untrained | 39 (transfer) | 51.1 % |
| level-32 pick, untrained | 40 (latch on a 5-step estimate velocity, 10 mm centring) | 77.3 % |
| level-39 pick (`model_14899`) | 39 | 74.9 % |
| level-39 pick, untrained | 40 | 88.1 % |
| level-40 pick (`model_16098`) | 40 | 89.5 % (87.9 % with the once-per-step fix) |
| level-40 pick, untrained | 43 (finger torque as touch feedback) | 82.1 % (fixed sim) |
| level-43 pick (`model_17297`) | 43 | 91.7 % (fixed sim; no heap 94.0 %) |
| level-44 pick (`model_18400`), ring and little in a fist | 44 (+ bed and cover) | 83.2 % (bed 91.9 %, cover 67.8 %, across 82 %) |
| level-45 training (cover and across at 0.5) | 45 | in progress |

What limited it, in order: the latch reading the camera estimate (noise broke the settled test, the 8 mm bias the
6 mm centring); then shallow pinches (8 mm off the true axis at the latch, the size of the estimate's bias) that let
the rod go before the test lift. Rejected: a squeeze after the latch (worse), the pre-lift held still (no change).
Level 43 gives the policy the hand's effort (motor current on the robot) in the dead tip-sensor slots, so it can
feel an off-centre pinch. A simulator bug (once-per-step counters advanced twice on reset steps) made earlier
numbers 1-2 points optimistic; fixed.

The robot runtime (adam-moveit genesis_policy) mirrors levels 40 and 43 (estimate-velocity window, effort
observation); parity passes on recordings from the fixed simulator, including a 64-env one. The current robot
candidates: for single rods in the box the level-43 pick (`logs/export/L43_17297/`); for rods on rods and
mixed heaps the level-44 pick (`logs/export/L44_18400/`, exported with `ADAM_U_HELD_CURL=0.95` so `hand.held`
carries the fist: ring and little at 1.49 / 1.99 / 1.49 rad, inside the URDF's limits). Both pass parity. Neither is
yet copied into adam-moveit's root-owned `handoff/policy/`.

**Ring and little in a fist** (2026-10-14): near a box wall the flat little finger touched the wall in 10.9 % of
episodes (rods 3-13 cm from a wall in a 16 x 16 cm bin), the fist in 0 %, at the same grasp success. A partial curl
puts the fingertips into the table; the fist does not. Levels 44+ train with it; the export must run with
`ADAM_U_HELD_CURL=0.95` too.


## 8. Bench preparation (2026-10-15)

- `handoff/policy/L43_17297/` and `handoff/policy/L44_18400/` copied into adam-moveit (actor.onnx, level.yaml,
  episode.npz).
- `adam-moveit:latest` rebuilt with `pin` 4.1.0, `onnxruntime` 1.31.0, `onnx`. The pip install pulled **numpy
  2.4.6**: the workspace had to be rebuilt with `--cmake-clean-cache` (numpy 2 moved its headers to
  `numpy/_core/include`; `pnd_adam`'s message generation failed on the stale cache). 17 packages build.
- `genesis_policy` in the image: 45 tests pass, parity against the handoff export with the real xacro URDF.
- **Test-ladder rung 1 passes** (`test_policy_node` on the loopback with the Wuji body, isolated domain): real mode
  refuses without confirmation, dry run publishes nothing, the stream reaches both controllers at 30 Hz.
- Fixed on the way: `safety_scene: training` (the default) falls back to the workcell with a warning when level.yaml
  has no training bin (the stub; it had made the node refuse to start).
- Rung 1 with the real policy waits for the hand's torque constant: L43/L44 latch on torque and the node refuses to
  start without `effort_scale_nm_per_a` (to measure at bring-up, 3.4).
