# deploy/thor

What the Jetson Thor runs: two containers from the GitHub registry, pulled with `pull.sh`, driven with `run.sh`.
Copy this directory to the Thor (`rsync -a deploy/thor/ thor4:~/genesis_cell/`), `cp .env.example .env`, set the
tags and the robot's ROS domain, `docker login ghcr.io` once with a token that has `read:packages`, then:

```bash
./pull.sh             # both images
./run.sh robot        # adam-moveit stack (real.launch.py with the two-key confirm)
./run.sh chain        # perception + selector + policy node
./run.sh cycle 5      # five grasp cycles
./stop.sh
```

| File | What |
|---|---|
| `docker-compose.yml` | the two services; images and tags from `.env`; host network, one ROS domain, `/dev` for the hands and the camera |
| `.env.example` | tags, `ROS_DOMAIN_ID`, `ADAM_MODE`, `POLICY_NAME` |
| `pull.sh` | checks the registry login, `docker compose pull` |
| `run.sh` | robot / app / perception / chain / cycle, each a `docker exec` into the right container with the overlay sourced |
| `config/perception.yaml` | the head RealSense topics and the bin box in the world frame (measure at bring-up) |
| `config/policy.yaml` | `effort_scale_nm_per_a`, the hand's torque constant (0 until measured; the policy node refuses to start) |
| `policy/` | REQUIRED with CI-built images: `<name>/actor.onnx + level.yaml (+ MODEL.md)` from adam-rl `tools/export_policy.py`; the CI image ships an empty `/handoff/policy` because checkpoints are not in git, and compose mounts this directory over it |

Images: `ghcr.io/revel-robotics/adam-moveit:<tag>-arm64` from adam-moveit's release workflow and
`ghcr.io/revel-robotics/genesis-rl-deploy:<tag>-arm64` from this repo's `.github/workflows/ci.yml` (on a `v*` tag or
a manual run; arm64 needs the repository variable `ARM64_RUNNER`). The Thor never builds: `pull.sh` only pulls.
`./pull.sh app` pulls the application image alone, for as long as the robot image is a native build the registry
does not have (adam-moveit's arm64 release must include 8ad47ab, the lifting-column joint limits, or the first
MoveTo kills the motion skills server).

Before the first real grasp, in this order: the head camera's hand-eye calibration, the bin box in
`config/perception.yaml`, the torque constant in `config/policy.yaml`, then the hardware ladder of
adam-rl `docs/SIM_TO_REAL.md` 3.6 (arm only, one rod at a known pose, camera, bin).

## State on Thor-4 (2026-11-03, evening)

| container | image | from |
|---|---|---|
| `adam_moveit` | `ghcr.io/revel-robotics/adam-moveit:e700f44-arm64` | built natively on the Thor from adam-moveit `rod-selector` e700f44 (`~/build/adam-moveit`, deploy target, `BASE_IMAGE=ubuntu:24.04`); has the lifting-column limits fix |
| `genesis_rl_app` | `ghcr.io/revel-robotics/genesis-rl-deploy:sha-f457e2c-arm64` | CI run 41709196759 (deploy-thor f457e2c), base adam-moveit e700f44 built from source in CI; pulled with `./pull.sh app` |

`.env`: `ADAM_IMAGE_TAG=e700f44 APP_TAG=sha-f457e2c ARCH_SUFFIX=-arm64 ROS_DOMAIN_ID=1 ADAM_MODE=real POLICY_NAME=L53b`.
`policy/L53b/` holds the level-53b export (actor.onnx, level.yaml, MODEL.md). `config/perception.yaml` has the
RealSense topics and the bin box of the sim layout (re-measure at bring-up); `config/policy.yaml` has
`effort_scale_nm_per_a: 0.0`, so the policy node refuses to start until the hand's torque constant is measured.

Verified inside the CI image on the Thor: aarch64, the genesis packages, pinocchio 4.1 / onnxruntime 1.31 / scipy,
54 + 13 tests pass, the L53b actor runs (74 observations -> 8 actions), the level file loads with its `pregrasp`
section. Both containers are up and idle; the real robot stack has not been launched and nothing has moved.

## Tomorrow morning

```bash
ssh thor4
cd ~/genesis_cell
docker compose ps                                   # both containers Up; else: docker compose up -d
./run.sh robot                                      # real.launch.py with confirm_real_hardware; needs the robot PC on the wire
tail -f logs/robot/stack.log                        # wait for the controllers to activate, Ctrl-C
docker exec adam_moveit bash -lc 'ros2 topic hz /lowstate'   # the wire is alive
```

That is the arm-only rung (adam-rl `docs/SIM_TO_REAL.md` 3.6, step 3): the stack, MoveTo to named states, hand
open, no rod, no policy. Before `./run.sh chain` and `./run.sh cycle`: measure the hand's torque constant and set
`effort_scale_nm_per_a` in `config/policy.yaml`; plug the head RealSense into the Thor, calibrate it, and set the
bin box in `config/perception.yaml`; put one rod at a measured pose for the first grasps. The policy launch keeps
`confirm_real_hardware:=false` unless `CONFIRM_REAL_HARDWARE=true` is exported before `./run.sh chain`.

To move to a newer CI image: set `APP_TAG` in `.env`, `./pull.sh app`, `docker compose up -d app`. To rebuild the
robot image after an adam-moveit change: see `~/build` (or adam-moveit's release workflow once its arm64 image
carries 8ad47ab, then `ADAM_IMAGE_TAG=<tag>` and `./pull.sh robot`).

