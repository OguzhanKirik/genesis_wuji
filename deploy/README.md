# deploy

The grasp application that runs the reach-to-grasp policy on the robot: everything the robot-side container
needs that is not the robot stack itself. adam-moveit stays the robot runtime (drivers, controllers, collision
scene, MoveTo and Stop skills) in its own container; this directory becomes the second container on the Jetson
Thor and talks to adam-moveit only over ROS 2 topics, services and actions.

| Package | What | State |
|---|---|---|
| `genesis_perception` | rod candidates from the head depth camera: segmentation, a RANSAC cylinder fit with the rod's known radius, `/perception/rod_candidates` (PoseArray, world frame). Replaces genesis_isaac's `rod_candidates.py` stand-in with the same contract. `compare_candidates` is the sim accuracy gate against Isaac's true poses. | written 2026-11-03, 10 host tests pass, not yet run against the Isaac head camera |
| `genesis_policy` | the policy runtime, policy_node, selector_node (from adam-moveit's `rod-selector` branch, commit 250f85c) plus `pregrasp.py` (the episode's start pose from the rod pose) and `scripts/grasp_cycle.py` (the orchestrator). adam-moveit's files it needs are vendored under `assets/` by `scripts/vendor_from_adam_moveit.sh`. | moved 2026-11-03; 33 host tests pass (parity skipped without a recording); `assets/genesis_wuji.urdf` still to generate (needs xacro: run the vendor script in the adam-moveit container) |
| `tools/render_rod_dataset.py` (repo root) | labelled head-camera frames of the heap scenes for the YOLO segmenter; `genesis_perception/scripts/make_yolo_labels.py` converts them | written, not yet run |
| `Dockerfile`, `docker-compose.yml` | the application image on adam-moveit's image (ROS, genesis_interfaces, pinocchio, onnxruntime come from there; x86 and arm64) and the two-container compose | written, not yet built |

## The grasp cycle

```
perception_node ──PoseArray──> selector_node ──PoseStamped──> policy_node ──JointTrajectory──> adam-moveit controllers
                                    ^   choice                    ^  enable/heartbeat/status
                                    └──── grasp_cycle.py ─────────┘── MoveTo / Stop actions ──> adam-moveit motion skills
```

1. `grasp_cycle` asks the selector for a rod; the selector picks the highest exposed reachable one (level 52's rule).
2. `pregrasp.py` turns the rod pose and the chosen hand yaw into the wrist pose the sim episode starts from
   (`level.yaml: pregrasp`, exported by `tools/export_policy.py`), and MoveTo plans the arm there.
3. The policy runs the approach, grasp and lift; `grasp_cycle` watches its status.
4. MoveTo carries the rod to the drop state, the hand opens, MoveTo returns home. A failure stops the group and homes.

Launch lines (inside the app container, with adam-moveit's stack up on the same domain):

```bash
ros2 launch genesis_perception perception.launch.py                       # or genesis_isaac rod_candidates.py in sim
ros2 run genesis_policy selector_node.py --ros-args -p level_path:=/handoff/policy/L53b/level.yaml
ros2 launch genesis_policy policy.launch.py policy_dir:=/handoff/policy/L53b
ros2 run genesis_policy grasp_cycle.py --ros-args -p level_path:=/handoff/policy/L53b/level.yaml -p cycles:=5
```

Not done yet: the controller switch to a forward command controller during the policy phase (policy_node streams
single-point trajectories to the JTC, as designed for the first runs), a pre-grasp pose for standing rods, the
YOLO model, the arm64 image build.

## Perception

```
head camera depth + camera_info ──> masks (depth clusters | YOLO-seg) ──> cylinder fit per mask ──> PoseArray
                                       TF world <- camera optical frame (hand-eye calibration, assumed done)
```

- The rod is a cylinder, radius 12.5 mm, length 35 mm, axis = local z (adam-rl `BOLT_RADIUS`, `BOLT_HALF_LEN`).
  The fit uses the radius as a constraint, so the top half of the surface is enough.
- Output orientation: z along the axis with the raised end up, x horizontal, as `genesis_policy.selector` expects.
- Stateless: an occluded rod drops out of the list; the selector latches on its choice and the policy node holds
  the last pose, which is what the policy trained with (`_PERC7`: 2 % dropout, 1-2 steps latency).
- `segmenter: depth` (clusters above the bin floor inside its xy box) for the table rung and isolated rods;
  `segmenter: yolo` with a model trained on Replicator renders of the heap scene for the bin and the pile.
- Known limit: a rod pointing within ~25 deg of the camera's view direction (a standing rod under the
  down-looking head camera) shows only a sliver of its side and its axis comes out ~4 deg off on clean data.
  The selector scores standing rods separately; if they matter, fit the end-face disc or use FoundationPose.
- Budget to pass before the hardware ladder: 5 mm / 3 deg bias, 1-2 mm noise (`compare_candidates` in sim,
  a marker-tracked rod on the bench).

Host tests (numpy + scipy only): `cd deploy/genesis_perception && python -m pytest test -q`.
