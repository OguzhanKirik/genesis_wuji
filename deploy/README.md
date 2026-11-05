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

- The rod is NOT the 12.5 mm cylinder of adam-rl's collision proxy (`BOLT_RADIUS`): the asset is a threaded rod
  35 mm long, thread radius ~8 mm, with a hex nut 11 mm thick at its middle (22.0 mm across the flats, 25.4 across
  the corners). The known-radius cylinder fit still finds the position (~3 mm) but its axis comes out 15-28 deg off.
  A rod lying on a surface therefore gets its pose from `rod_fit.refine_pose`, which fits that real shape
  (`rod_top_height`): 0.6-1.1 mm and 0.3-1.6 deg against the simulator's truth (2026-11-04, `compare_candidates`,
  4 drops). Leaning and standing rods keep the cylinder fit and are not validated yet. `refine_axis:=false` turns
  the shape fit off; it costs the node about half its rate (7 -> 3-4 Hz on the bench).
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

## The sim rung on the bench (what was run on 2026-11-03)

adam-moveit on `rod-selector` (needs 8ad47ab: lifting-column joint limits for the Wuji design, else the motion
skills server dies on the first MoveTo; e700f44: `workcell_single_rg.json`, one rod at the trained window centre),
its container on its own ROS domain (`SIM_ROS_DOMAIN_ID=42 ADAM_MODE=sim docker compose up -d adam_moveit`), then:

```bash
docker exec adam_moveit bash -lc 'ros2 launch genesis_cell sim.launch.py design:=genesis_wuji workcell_layout:=workcell_single_rg.json headless:=true rviz:=false'
docker build -f deploy/Dockerfile --build-arg BASE_IMAGE=adam-moveit:latest -t genesis-rl-deploy:bench .
docker run -d --name genesis_rl_app --network host --ipc host --gpus all -e ROS_DOMAIN_ID=42 \
    -v ~/adam-moveit/ws:/root/ws:ro -v $PWD/handoff:/handoff:ro genesis-rl-deploy:bench sleep infinity
E=/usr/local/bin/genesis_rl_entrypoint.sh          # docker exec bypasses the entrypoint: source the overlay through it
docker exec -d adam_moveit bash -lc 'ros2 run genesis_isaac rod_candidates.py --ros-args -p topic:=/perception/rod_truth -p noise_xy_m:=0.0 -p noise_z_m:=0.0 -p noise_rot_deg:=0.0'
docker exec -d genesis_rl_app $E bash -c 'ros2 run genesis_perception perception_node --ros-args -p bin_xy_min:="[0.35, -0.45]" -p bin_xy_max:="[0.75, 0.05]" -p bin_floor_z:=0.95'
docker exec -d genesis_rl_app $E bash -c 'ros2 run genesis_policy selector_node.py --ros-args -p level_path:=/handoff/policy/L53b/level.yaml'
printf "policy_node:\n  ros__parameters:\n    effort_scale_nm_per_a: 1.0\n" > /tmp/policy_sim.yaml   # the bridge passes torque through
# The camera loses the rod once the hand covers it; policy_node tracks it (runtime.RodTracker: fit -> held for
# rod_hold_s -> riding on the wrist after the latch, unless a fresh fit shows the rod left on the table). No
# perception_stale_s override is needed for that.
docker exec -d genesis_rl_app $E bash -c 'ros2 launch genesis_policy policy.launch.py policy_dir:=/handoff/policy/L53b params_file:=/tmp/policy_sim.yaml'
docker exec genesis_rl_app $E bash -c 'ros2 run genesis_policy grasp_cycle.py --ros-args -p level_path:=/handoff/policy/L53b/level.yaml -p cycles:=1'
```

Verified: the depth back-projection equals Isaac's own point cloud and the table edges land on the layout
(camera, intrinsics and TF agree); parity with the level-53b recording; MoveTo and Stop through the motion
skills server; the selector accepts the rod at the window centre.

Known: `/tmp/genesis_bolt_poses.json` (the truth stand-in) reports the rod prim's origin, which the asset puts at
one END of the rod, and even corrected it sits a few cm from the rendered rod, so `compare_candidates` is not a
valid gate until that is fixed on the adam-moveit side. A heap at the head camera's resolution (a rod is ~12 px
wide) defeats the depth-cluster segmenter even with sequential fits: the bin needs the YOLO segmenter.


## CI images (`.github/workflows/ci.yml`)

Tests on every push and pull request; images on tags `v*` and on manual dispatch, pushed to
`ghcr.io/<owner>/genesis-rl-deploy:<tag>-<arch>` plus the multi-arch manifest `<tag>` (arm64 when the repository
variable `ARM64_RUNNER` names a runner). The adam-moveit base comes from one of two places, decided per run:

| what the repository holds | base image | needs |
|---|---|---|
| secret `ADAM_MOVEIT_DEPLOY_KEY` (a read-only deploy key on adam-moveit) + variable `ADAM_MOVEIT_REF` (a branch, a tag, or a FULL 40-character SHA; a short SHA is taken for a branch pattern and the checkout fails) | adam-moveit checked out at that ref and its `deploy` target built on the runner (`BASE_IMAGE=ubuntu:24.04`, the same recipe as adam-moveit's release and the Thor) | nothing else; ~20-30 min per architecture (set 2026-11-03, ref e700f44) |
| secret `GHCR_PULL_TOKEN` (read:packages), or the genesis_rl repository granted read access on the adam-moveit package | `ghcr.io/<owner>/adam-moveit:<ADAM_IMAGE_TAG>` pulled | that access; adam-moveit's package is private and scoped to its own repository, so the plain `GITHUB_TOKEN` gets 403 |

Pulling the published genesis-rl-deploy image onto the Thor still needs one `docker login ghcr.io` there with a
token that has `read:packages`; until then `deploy/thor` runs the images built natively on the Thor
(`~/build`, tags `e700f44` / `local` in its `.env`).
