# genesis_policy

Runs the learned reach-to-grasp policy (right arm + Wuji Hand 2, trained in
`~/adam-rl` with Isaac Lab) on the Adam-U through ros2_control. Section 3.3
of `~/adam-rl/docs/SIM_TO_REAL.md`.

```
/joint_states ──┐                                   ┌─> /right_arm_controller/joint_trajectory
/perception/    ├─> Runtime (runtime.py) ─> safety ─┤
  bolt_pose ────┘   obs ─> ONNX actor ─> wrist step  └─> /right_hand_controller/joint_trajectory
                    ─> DLS IK ─> 7 arm + 20 hand
                    targets, phase machine
```

Nothing here trains. adam-rl exports an actor (`actor.onnx`) and a file of
level constants (`level.yaml`) into `handoff/policy/<name>/`; this package
reproduces the simulator's action term and observation terms around that
actor without the simulator, and one node streams the result at the policy
rate. The waist is not commanded: it stays where the stack left it.

| Piece | File | What |
|---|---|---|
| level loader | `genesis_policy/level.py` | `level.yaml` -> frozen `Level`, every key validated (schema below) |
| model | `genesis_policy/model.py` | pinocchio model from the design's xacro: FK, the wrist Jacobian (world-aligned, shifted to the policy's wrist point), URDF limits |
| cell | `genesis_policy/cell.py` | table and bin boxes from `genesis_isaac/assets/workcell/workcell.json` |
| runtime | `genesis_policy/runtime.py` | observation assembly, actor call, wrist step, workspace clamp, damped least-squares IK with the mid-range null-space pull, the phase machine (grasp, settle, pre-lift with the grip check, lift, hold, retries), hand synergies -> 20 targets. `step(joint_states, rod_pose, t) -> (arm[7], hand[20], info)`, deterministic |
| actor | `genesis_policy/actor.py` | `OnnxActor` (onnxruntime, CPU), `ZeroActor`, `ScriptedActor` |
| node | `genesis_policy/policy_node.py` | the ROS node: topics, the dead-man, the safety wrapper, the streamer |
| stub | `scripts/make_stub_policy.py` | writes a zero-action `actor.onnx` and a `level.yaml` of adam-rl's constants |
| launch | `launch/policy.launch.py`, `config/defaults.yaml` | |

## Running

Build inside the container (`colcon build --packages-up-to genesis_policy`,
the image carries `pin`, `onnxruntime` and `onnx` from the Dockerfile). Then,
against a running stack with the Wuji body
(`genesis_isaac sim.launch.py design:=genesis_wuji`, which puts the Isaac
bridge in `--hands wuji` mode, or the real robot):

```bash
ros2 run genesis_policy make_stub_policy.py --out /handoff/policy/stub     # once, until adam-rl hands over
ros2 launch genesis_policy policy.launch.py policy_dir:=/handoff/policy/stub dry_run:=true
ros2 topic echo /policy_node/status                                          # JSON: phase, step, aborted, reason
```

The node does nothing until the dead-man is satisfied: the enable service
AND a live heartbeat. Hold both while it runs:

```bash
ros2 topic pub -r 10 /policy_node/heartbeat std_msgs/msg/Empty &
ros2 service call /policy_node/enable std_srvs/srv/SetBool "{data: true}"
```

Drop `dry_run:=true` to stream. Killing the heartbeat publisher, or
`{data: false}`, sends one hold trajectory at the measured pose and stops;
`{data: true}` again starts a fresh episode at the measured state.

On the real robot the container is in real mode (`ADAM_MODE=real`), and the
node refuses to start unless the launch line says
`confirm_real_hardware:=true`, the same two-key rule as `real.launch.py`.
The arm must already be inside the policy's workspace (0.02 to 0.35 m above
the table top, within 15 cm of the rod-relative nominal wrist xy) and the
fence when the episode starts: bring it there with `/move_to` first. A
target outside the workspace is clamped, a step above the cap is scaled, a
measured wrist outside the fence aborts.

### What the node enforces before every publish

| Check | Parameter | On failure |
|---|---|---|
| joint state fresher than | `state_stale_s` (0.5, the `genesis_motion_skills` convention) | hold, stop |
| rod pose fresher than | `perception_stale_s` (0.5) | hold, stop |
| rod pose, once running | the tracker (below): a fit fresher than `perception_stale_s`, else the last pose for `rod_hold_s` (10.0), after the latch a pose riding on the wrist | hold, stop when the hold runs out before the latch |
| heartbeat fresher than | `heartbeat_timeout_s` (0.5) | hold, stop |
| enable service said true | `~/enable` | not started / hold |
| every right-hand effort below | `hand_effort_abort` (A, the wujihandros2 driver's filtered current) | hold, stop |
| measured wrist inside the fence box (+ `fence_margin`) | `fence.*` | hold, stop |
| commanded wrist inside the fence | `fence.*` | hold, stop |
| wrist travel per tick (FK of consecutive commands) at most max(`effective_translation` x sqrt 3, the fast lift step) | from `level.yaml` | scaled to the cap; hold if still above `step_cap_tolerance` x cap |
| arm targets inside MoveIt's joint limits minus `ik.joint_margin` | `genesis_robot_config/config/moveit/joint_limits_genesis_wuji.yaml` | hold, stop (the runtime clips; this is the wrapper's own check) |

"hold, stop" is one single-point trajectory at the measured positions on
both controllers, then no more messages until the next enable. In
`dry_run` nothing is ever published, including the hold.

The stream is single-point `JointTrajectory` messages on each controller's
`~/joint_trajectory` topic every tick, `time_from_start` =
`time_from_start_s` (60 ms), all joints of each controller present
(`allow_partial_joints_goal` is false), zero velocity at the point on the
arm. The JTC replaces its trajectory on each message; the hardware plugins'
watchdog, slew cap and clamp stay in the loop. `Streamer` in
`policy_node.py` is the one place that knows this wire format; the comment
marked SEAM is where a `forward_command_controller` would go if the JTC
interpolation visibly lags (not added: it claims the same interfaces as the
JTC and the two cannot be active together).

## Parameters

`config/defaults.yaml`; the launch file overlays `policy_dir`, `design`,
`dry_run`, `rate_hz`, `cell_file`, `confirm_real_hardware`, and
`params_file:=` adds a YAML on top.

| Parameter | Default | Meaning |
|---|---|---|
| `policy_dir` | `/handoff/policy/stub` | directory with `actor.onnx` and `level.yaml` |
| `design` | `genesis_wuji` | body; the URDF (xacro) and the controllers' joint lists are read for it |
| `cell_file` | `""` | workcell JSON (table, bin); empty = genesis_isaac's `workcell.json` |
| `cell_mirror_y` | false | mirror the bin to y < 0, as adam-rl trained the right arm (see open questions) |
| `rate_hz` | 0 | policy rate; 0 = `level.yaml control.rate_hz` |
| `joint_states_topic` | `/joint_states` | body + hands, from the joint state broadcaster |
| `bolt_pose_topic` | `/perception/bolt_pose` | `geometry_msgs/PoseStamped` in `world_frame` (the `genesis_cell` contract); other frames are refused |
| `arm_controller`, `hand_controller` | `right_arm_controller`, `right_hand_controller` | whose joint lists are read from `ros2_controllers_genesis_wuji.yaml` |
| `arm_trajectory_topic`, `hand_trajectory_topic` | `/<controller>/joint_trajectory` | where the stream goes |
| `time_from_start_s` | 0.06 | the single point's time ahead of now |
| `dry_run` | false | compute, log and publish `~/status` only |
| `state_stale_s`, `perception_stale_s`, `heartbeat_timeout_s` | 0.5 | the refusals above |
| `rod_hold_s` | 10.0 | how long the last fit is held without a fresh one before the latch (the fingers hide the rod for seconds while closing: 3 s aborted 2 of 3 sim episodes); after it the pose rides on the wrist |
| `rod_left_radius_m` | 0.03 | a fresh fit this close to where the rod rested at the latch means the hand lifted empty: the fit is reported instead of the wrist-attached pose |
| `hand_effort_abort` | 1.5 | A |
| `fence.{x,y,z}_{min,max}` | a box around the right-side bin | world frame, metres |
| `fence_margin` | 0.02 | how far outside the fence the measured wrist may sit |
| `step_cap_tolerance` | 1.5 | abort when a scaled step still exceeds this multiple of the cap |
| `velocity_filter` | 0 | first-order filter on the differenced wrist velocities (0 = none) |
| `finger_velocity` | `difference` | `difference` or `reported` (the hand driver reports none) |
| `world_frame` | `world` | the planning frame; FK, the fence and the rod pose are in it |
| `confirm_real_hardware` | false | must be true when `ADAM_MODE=real` |

Topics and services the node owns: `~/heartbeat` (`std_msgs/Empty`, in),
`~/enable` (`std_srvs/SetBool`), `~/status` (`std_msgs/String`, JSON at 5 Hz:
enabled, aborted, reason, step, phase, attempts, latched, done, success,
tracking error, wrist position, messages published).

## Handoff schema (`handoff/policy/<name>/`)

The `./handoff` directory is mounted at `/handoff` by
`docker-compose.robot.yml`; `docker-compose.yml` mounts `./handoff/policy`
at `/handoff/policy` next to the adam-grasp handoff. Three files per policy:

| File | Contract |
|---|---|
| `actor.onnx` | input `obs` float32 `[1, N]`, output `action` float32 `[1, 8]`, tanh-bounded, the observation normaliser folded in. N must equal the layout's sum; the node refuses otherwise |
| `level.yaml` | every constant below. The runtime reads nothing else; a missing, mistyped or inconsistent key refuses to start, naming the key |
| `episode.npz` | one recorded simulator episode for `test/test_parity.py` (keys below); optional until adam-rl records it |
| `MODEL.md` | the model card, generated by the export: the network (MLP 74 -> 256 -> 256 -> 128 -> 8, ELU, normaliser folded into the ONNX, the mean action out, ~119k parameters), the PPO settings it was trained with, the observation table index by index with what the robot provides for each, the action table with its scales, the phase machine around the policy, and what the robot must supply (joint states with effort, the rod estimate and the noise it was trained against, the start pose) |

`make_stub_policy.py` writes an example `level.yaml` with every value as
read from adam-rl's source on 2026-10-08; the lines marked `FROM adam-rl`
are level-dependent or hardware substitutes and must come from the export.

```yaml
schema: 1                      # this runtime's schema version
source: "<free text: adam-rl commit, level, checkpoint>"
control:
  rate_hz: 30                  # policy rate (CONTROL_HZ)
observation:
  layout:                      # the actor's input, in order; names from the table below, fixed sizes
    - {name: bolt_pos_rel, size: 3}
    - ...
  episode_max_steps: 692       # ep_time = step / episode_max_steps
  tip_force_scale: 40.0        # tip_forces = force / this, clipped to [-5, 5] (F_MAX)
  arm_q_limits: {lo: [7], hi: [7]}   # the limits arm_q_norm and the IK null-space pull were trained with (the URDF's)
action:
  effective_translation: 0.00333     # m per step per axis, the per-axis bound of the wrist step
  effective_rotation: 0.01667        # rad per step per axis
  wrist_smoothing: 0.5               # residual += s * (cmd - residual)
  action_smoothing: 0.3              # closure += s * (c - closure)
  yaw_rate_scale: 1.0                # scales the world-z rotation step
ik:
  dls_damping: 0.05                  # lambda in J^T (J J^T + lambda^2 I)^-1
  nullspace_gain: 0.30               # mid-range pull, urgency-weighted
  joint_margin: 0.05                 # rad kept clear of every clip limit
  max_joint_velocity: 2.0            # rad/s, per-step joint travel cap (times 1/rate)
  singularity_threshold: 1.0e-4      # on sqrt(det(J J^T)), reported in info
  workspace:
    radius_xy: 0.15                  # m around centre = rod xy at episode start + nominal_wrist_offset_xy
    z_min: 0.02                      # m above the table top (from the cell file)
    z_max: 0.35
    nominal_wrist_offset_xy: [-0.0474, -0.0901]
wrist:
  link: wristRollRight               # the policy's wrist frame = this link ...
  offset: {xyz: [0, 0, 0], rpy: [0, 0, 0]}   # ... times this fixed transform
arm:
  joints: [7 names]                  # must equal right_arm_controller's list
hand:
  thumb_joints: [3 names]            # closure 0 drives these, in observation order
  finger_joints: [6 names]           # closure 1 drives these
  open: {joint: rad}                 # 9 entries, the synergy at closure 0
  closed: {joint: rad}               # 9 entries, at closure 1
  held: {joint: rad}                 # the other 11 right_hand_controller joints, fixed
  tips: {thumb: {link, offset}, index: ..., middle: ...}   # the fingertip pads (tip_dist, the grip check)
phase:                               # steps are policy steps
  settle_in, grasp_max_steps, grasp_retry_steps, settle_steps, drag_steps,
  free_prelift, grip_check, grip_check_steps, grip_test_total_m, grip_check_ratio,
  grip_check_min_tips_m, grip_check_late, lift_steps, lift_slow_steps, lift_slow_step_m,
  lift_height_m, hold_steps, retry_max, success_lift_m, up_threshold_m, lift_dir: [3],
  sense_est                          # optional, default false: the rest height is the estimate's at step 1
  latch:                             # mode closure (default) or torque
    mode: closure                    # {steps, closure_min, finger_qd_max}: the stub's rule
    mode: torque                     # adam-rl hw_latch (level 39+): {steps, tau_min_nm, grasp_region_m,
                                     #   pinch_centre_tol_m, rel_vel_max_m_s}; needs effort_scale_nm_per_a
safety:
  groups: {tips: [[link, [x, y, z], radius], ...], palm: ..., forearm: ..., upperarm: ...}
  vec_clamp: 1.0                     # the wrist-frame vectors are clipped to +-this
  dist_clamp: [-0.1, 0.5]            # the clearances to this range
  extra_boxes: []                    # [{lo: [3], hi: [3]}] in world, besides the table and the bin
rod:
  radius: 0.0125
  half_len: 0.0175
```

Observation terms and sizes (`genesis_policy.level.TERM_SIZES`):

| Term | Size | Source on the robot |
|---|---|---|
| `bolt_pos_rel` | 3 | rod position minus wrist position, in the wrist frame |
| `bolt_rot_rel` | 6 | first two columns of R_wrist^T R_rod |
| `wrist_lin_vel`, `wrist_ang_vel` | 3 + 3 | `wrist_velocity: jacobian` (default): the full-body J(q) times every reported joint velocity, world frame, the simulator's body velocity; `difference`: FK difference, half a step late |
| `finger_q`, `finger_qd` | 9 + 9 | the synergy joints from `/joint_states`; velocity differenced by default |
| `last_action` | 8 | the previous step's actor output as given, before the clamp and the phase mask (Isaac Lab's `mdp.last_action`) |
| `tip_forces`, `contacts` | 3 + 3 | a pluggable `tip_source`; **zeros** until one exists (TODO in `policy_node.py`: the hand's effort field) |
| `ep_time` | 1 | step / `episode_max_steps` |
| `arm_q_norm` | 7 | 2 (q - lo) / (hi - lo) - 1 over `arm_q_limits`, clipped to [-2, 2] |
| `tip_dist` | 3 | fingertip pad to the rod's axis segment, minus the radius, floored at 0 |
| `safety` | 16 | per group the vector from the nearest proxy sphere to the nearest obstacle (boxes, and capsules for neighbour rods via `Runtime.set_scene`), in the wrist frame (4 x 3), then the four clearances |

`episode.npz` (adam-rl `tools/export_policy.py --record N`): N consecutive
episodes of one simulated env, in the planning frame `world`. Per step:
`episode`, `time` (s from the episode start), `joint_pos`, `joint_vel`,
`joint_effort` (the simulator's applied torque, N m) over `joint_names`,
`rod_pos`, `rod_quat_xyzw` (the estimate the policy saw), `lift_dir`,
`box_lo`/`box_hi` (table first) and `cap_p0`/`cap_p1`/`cap_r` (neighbour
rods), `obs`, `action`, `arm_targets` and `hand_targets` over
`hand_target_names` (the first physics substep's), `phase`. Per episode:
`target_seed_pos`, `target_seed_quat_xyzw`, `workspace_centre`.

## The phase machine on hardware

Mirrors `ReachGraspAction` step for step; `g` is the grasp-end step
(`grasp_max_steps` until the latch fires, then the latch step).

| Steps | Phase | Wrist | Hand |
|---|---|---|---|
| `k < settle_in` | grasp, frozen | still | still |
| `k < g` | grasp | the policy's step, smoothed | the policy's closures, smoothed |
| `g .. g+settle_steps` | settle | still (re-seeded at the measured pose at the end) | held at the latched closure |
| then `drag_steps` | pre-lift, policy window if `free_prelift` | the policy's (fingers held) | held |
| then `grip_check_steps` | pre-lift, test lift (latched only) | `grip_test_total_m` along `lift_dir` | held |
| `g+lift_start .. +lift_steps` | lift | `lift_slow_steps` x `lift_slow_step_m`, then an even rate to `lift_height_m` | held |
| then `hold_steps` | hold | still | held |
| after | done | still | held |

The grip check at the end of the test lift: if the fingertips rose more than
`grip_check_min_tips_m` and the rod rose less than `grip_check_ratio` of
that, the rod is not in the hand. That, a lift that ends with the rod not
`success_lift_m` up, or a drop in the hold (`up_threshold_m`) triggers a
retry while `attempts < retry_max`: the hand opens, the residual clears, the
IK re-seeds at the measured state and the grasp phase restarts with
`grasp_retry_steps`. The rod's rest height is its z at the start of the
episode.

The latch ends the grasp phase. `latch.mode: torque` is adam-rl's
`grasp_ready` at its hw_latch levels: the |effort| summed over the thumb's
and over the index's or the middle's flexion chain above `tau_min_nm`, each
for `steps` consecutive steps (streaks run every step), the fingertips
inside `grasp_region_m` of the estimated rod, the pinch midpoint within
`pinch_centre_tol_m` of its axis, and the estimate's own finite-difference
velocity within `rel_vel_max_m_s` of the wrist's; never on step 0 or during
`settle_in`. The effort is the hand driver's current times
`effort_scale_nm_per_a`. `latch.mode: closure` (the stub) is a rule on what
the runtime knows without effort: both closures at or above
`latch.closure_min` and every synergy joint slower than
`latch.finger_qd_max` for `latch.steps` consecutive steps.

## Choosing the rod: `selector_node.py`

Perception reports every visible rod; the policy takes one. `genesis_policy.selector` chooses it with the rule
adam-rl's level 52 trains on (adam-rl `docs/PICKABILITY_REPORT.md`: the level-51 policy picks 66.9 % of rods 35 mm
or more above the bin floor, 17.8 % of those under 25 mm): drop rods another rod rests on, rods between leaning and
standing, and rods outside the trained window (level.yaml `workspace`, exported since 2026-10-16); take the highest
centre, a standing rod counted 15 mm lower.

```
/perception/rod_candidates (PoseArray, world) -> selector_node -> /perception/bolt_pose (PoseStamped) -> policy_node
```

`~/select` (Trigger) chooses; with `auto_select` (default on) the first choice is automatic. Nothing is chosen while
the heap moves (every rod within `settle_tol_m` 15 mm of where it was `settle_window_s` 1 s ago). The chosen rod is
then followed as the nearest candidate within `track_radius_m` (2 cm) and published at the candidates' rate; if it is
lost, publishing stops and nothing is chosen again until `~/select`, so the target never switches during a pick.
`~/choice` (String, JSON) carries the pick, its kind, height, a reachable hand yaw, and why each other rod was passed
over. In sim, `genesis_isaac rod_candidates.py` publishes Isaac's rod poses as candidates with 2 mm / 2 deg noise
(the bin must be on the right hand's side: `workcell_layout:=workcell_bin_right.json`). Real perception replaces it
with the same topic and type.

## Tests

| Test | Where | Needs | Proves |
|---|---|---|---|
| `test_level.py` | this package, pytest | nothing | the stub parses; 15 malformed files are refused naming the key; a subset layout works |
| `test_model.py` | pytest | xacro (or `GENESIS_POLICY_URDF`), pinocchio | wrist and fingertip FK equal an independent numpy walk of the URDF; the Jacobian equals finite differences, with and without a wrist offset; the right-arm limits are the URDF's; joint names map to the right slots |
| `test_runtime.py` | pytest | the above | observation layout and values; hand targets land in the controller order; the IK follows translation and rotation commands and converges; the clip uses MoveIt's limits; the workspace clamp binds; the phase machine times out, lifts 10 cm and retries; the latch, a passing grip check and a successful hold; a failing grip check and the reopen; determinism; the tip source; the rod tracker (measured, held, stale, attached to the wrist, left on the table, the retry) |
| `test_parity.py` | pytest | `episode.npz` (the xacro URDF, with the world joint) | adam-rl's recorded actions (1e-4), observations (1e-3; the velocities 5e-3, PhysX's link velocities agree with J qdot to ~0.1 %; the safety vectors may flip on 0.5 % of steps where two obstacles tie) and first-substep joint targets (1e-4) are reproduced, episode by episode with the recorded seed, workspace centre, lift direction and scene (skips with the reason until the fixture exists) |
| `test_selector.py` | pytest | nothing (parity: adam-rl `dump_fill_choices.py` + a level.yaml with `workspace`) | the highest exposed reachable rod wins; a covered rod, a steep one and one outside the window are passed over with the reason; standing rods count 15 mm lower; the yaw window's core and edge limits; a leaning rod's direction; parity: the rod adam-rl's level 52 chose, heap for heap (1024 / 1024 on 2026-10-16) |
| `test_policy_node.py` | `genesis_integration_tests`, launch_testing | the loopback stack with the Wuji body (`test/wuji_hand_loopback.py` stands in for the hand driver) | real mode refuses without confirmation; `dry_run` steps and publishes nothing; the stream reaches both controllers at 30 Hz with their joint lists, the hand follows the targets through the plugins, the scripted lift raises the wrist, a lost heartbeat holds and stops |

```bash
colcon test --packages-select genesis_policy                       # unit
ADAM_TEST_DOMAIN_ID=42 scripts/ci/run_integration.sh               # integration, with the others
```

FK is checked against an independent URDF walk rather than MoveIt's
`/compute_fk` because the unit tests run without a stack; on the stack the
integration suite reads the wrist from TF (robot_state_publisher, the same
URDF) and the lift assertion holds only if the runtime's FK agrees with it.

## Open questions for adam-rl

Written down rather than guessed. Each is a `FROM adam-rl` line in the
stub `level.yaml` or a parameter here.

1. **Observation width.** The brief says 61; the 13 terms it lists
   (`ObservationsCfg.PolicyCfg`) add up to 74. The layout is data in
   `level.yaml` and the node checks it against the actor, so either works;
   the export decides. If the transfer level (3.2) drops `tip_forces` and
   `contacts`, the layout shrinks to 68.
2. **The latch.** The simulator's `grasp_ready` reads fingertip contact
   forces. Until the hand's effort field is characterised (3.4), the
   runtime ends the grasp phase on the closure-and-stillness rule above.
   The transfer level should train against the same rule, or the export
   should say what the latch is.
3. **`arm_q_norm` limits.** Training normalised with Isaac's joint limits,
   the URDF's (shoulderPitch down to -3.61). The IK output is clipped to
   MoveIt's firmware value (-1.047) as required; the observation and the
   null-space pull keep the training limits (`observation.arm_q_limits`),
   so the policy sees the numbers it was trained on. Confirm, or export the
   limits the policy should normalise with.
4. **Which side is the bin.** `workcell.json` puts the bin at y = +0.20 in
   front of the left arm; adam-rl mirrors it to y = -0.20 for the right arm
   (`WORKCELL_SIDE`). On the real cell the bin is wherever it is placed:
   `cell_mirror_y` picks the side for the safety boxes, the rod pose comes
   from perception either way.
5. **`lift_dir`.** The simulator lifts along the rod's up-normal when the
   wrist is rolled (`_lift_dir`); the stub lifts along world z.
6. **Level constants** marked `FROM adam-rl`: `settle_in`, `free_prelift`,
   `grip_check`, `grip_check_late`, `retry_max`, `yaw_rate_scale`,
   `episode_max_steps`.
7. **The waist.** Training held the waist at the level's triple (the
   pose-bucket's). The node does not command the waist; whoever starts the
   episode must put it there with `/move_to` or `/go_ready` first. The
   triple should be in the export so the operator knows what it is.
8. **Obstacles.** The brief named `genesis_cell/config/cell.yaml` for the
   table and bin; that file holds the bolt task's offsets and topics, the
   furniture is in `genesis_isaac/assets/workcell/workcell.json`, which
   adam-rl also reads. The runtime reads the JSON (`cell_file`); extra
   boxes go in `level.yaml safety.extra_boxes`.
9. **The handoff mount on the bench.** `docker-compose.yml` already mounts
   the adam-grasp directory (`~/adam-handoff`) at `/handoff`, so the policy
   directory is mounted inside it (`./handoff/policy` -> `/handoff/policy`)
   rather than replacing it. The repo's `handoff/` is root-owned; compose
   creates `handoff/policy` on the next `docker compose up -d`.
10. **`onnx`** is a third pip dependency beyond `pin` and `onnxruntime`,
    only so `make_stub_policy.py` can write the stub graph; onnxruntime
    does not write models. Drop it once the real artefacts exist, or keep
    it for the stub.

## Answers from adam-rl (2026-10-13)

1. The export's layout is 74 wide: `tip_forces` and `contacts` stay in it and
   read zero at `tactile: false` levels.
2. Level 39 trains against the torque latch above; the export says so
   (`latch.mode: torque`). The closure rule stays for the stub.
3. `arm_q_limits` are Isaac's joint limits (the URDF's), exported.
5. `lift_dir` is per episode in the simulator: world z, or the rod's
   up-normal when a tilted rod gets a rolled wrist. `Runtime.reset(lift_dir=)`
   takes it; on the robot it has to come from the rod estimate. Not wired in
   the node yet.
6. The level constants come from the export.
8. The simulator's safety scene is per episode (the bin placed around the
   rod, cover blocks, the neighbour rods as capsules). `Runtime.set_scene`
   takes it; the node still uses `workcell.json`'s fixed table and bin.

Two differences the parity test does not cover, because the runtime is
seeded with the simulator's values: the IK target starts from the episode's
analytic start pose in the simulator (`target_seed`, the measured pose by
default here), and the workspace clamp is centred on the analytic nominal
wrist xy (`workspace_centre`, the rod plus the nominal offset by default).
And one the runtime cannot reproduce: the simulator re-solves the IK on every
physics substep (120 Hz) from the measured state; the runtime, once per
policy step. The parity test compares against the first substep.

## Status (2026-10-13)

Parity passes against adam-rl's level-39 recording (the level-32 pick, 8
episodes, 2002 steps: first-try latches, one to four retries, a time-out).
39 tests on the host. Not yet: the image rebuilt with the pip dependencies,
`colcon build`, the launch_testing suite on the loopback, the run against
the Isaac backend, the hand's torque constant. Nothing has run on the real
robot.

## Status (2026-10-08)

Written and unit-tested on the host against a xacro-generated Wuji URDF
(38 tests, parity skipped). Not yet: the image rebuilt with the pip
dependencies, `colcon build`, the launch_testing suite on the loopback, the
run against the Isaac backend (`sim.launch.py design:=genesis_wuji`).
Nothing has run on the real robot.
