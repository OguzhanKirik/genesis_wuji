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
| `policy/` | optional: a newer `<name>/actor.onnx + level.yaml` than the one in the image |

Images: `ghcr.io/revel-robotics/adam-moveit:<tag>-arm64` from adam-moveit's release workflow and
`ghcr.io/revel-robotics/genesis-rl-deploy:<tag>-arm64` from this repo's `.github/workflows/ci.yml` (on a `v*` tag or
a manual run; arm64 needs the repository variable `ARM64_RUNNER`). The Thor never builds: `pull.sh` only pulls.

Before the first real grasp, in this order: the head camera's hand-eye calibration, the bin box in
`config/perception.yaml`, the torque constant in `config/policy.yaml`, then the hardware ladder of
adam-rl `docs/SIM_TO_REAL.md` 3.6 (arm only, one rod at a known pose, camera, bin).
