# Genesis RL

Train and test reinforcement-learning policies for the **Adam-U** robot in
Isaac Lab. The Python package is `genesis_rl`. Policies are PPO from
[rsl-rl](https://github.com/leggedrobotics/rsl_rl) 5.0.1 (`rsl-rl-lib`).

A task is one Isaac Lab environment, one PPO config, and one log directory.
`train.py` and `play.py` both take `--task`. Adding a task does not require a
new training script.

Past runs, reward equations, and the long command list are in
[`docs/EXPERIMENT4_STEPS.md`](docs/EXPERIMENT4_STEPS.md) (Experiment 4 step by step),
[`docs/GRASP_AREA_REPORT.md`](docs/GRASP_AREA_REPORT.md) (the graspable-area study, levels 15–17) and
[`docs/`](docs/EXPERIMENTS_SUMMARY.md). This page is how to train and how to
watch a policy.

## Tasks

| `--task` | what it trains |
|---|---|
| `pnd_grasp` | Adam-U with Inspire hands, lift a cube |
| `wuji_reach` | Adam-U with Wuji Hand 2, reach a bolt and grasp it |
| `wuji_grasp` | same robot, hand already over the bolt |
| `wuji_synergy` | right hand only, two finger synergies, scripted lift |
| `wuji_pipeline` | whole robot, wrist through IK plus synergies |
| `wuji_reach_grasp` | standoff approach, then a scripted lift |

`--task` also accepts the gym id, for example `Isaac-WujiSynergyGrasp30`.
The older `--robot` and `--scenario` flags select the same tasks.

## Environment

One image trains and renders: Isaac Sim **6.0** / Isaac Lab **3.0**.

| | |
|---|---|
| Python | 3.12 |
| torch | 2.10.0+cu128 |
| Isaac Sim | 6.0.0.1 |
| Isaac Lab | 3.0, commit `28a37ce` (beta) |
| rsl-rl-lib | 5.0.1 |

Isaac Lab 3.0 is beta. A later commit can break this image; the pin above is
the one verified here.

Robots spawn from a flat USD under `assets/`. Those files are in git.
`assets/robots/adam_u/usd/configuration/adam_u_base.usd` is 104 MiB, over
GitHub's 100 MiB file limit, so it is stored with Git LFS. After a clone, run
`git lfs pull` if that file is still a pointer.

## Start

From the repo root, on the host:

```bash
docker compose build          # first time; the image export is slow, not hung
xhost +local:root             # once per login, only if you want a window
docker compose up -d
docker compose exec adam_rl bash
```

The shell is now inside the container at `/workspace`, which is this repo.
Run the commands below there. `exit` returns to the host.

One command from the host, without staying in the container:

```bash
docker compose run --rm adam_rl python genesis_rl/scripts/train.py --task wuji_synergy --num_envs 1024
```

## Train

Headless, which is what a real run should use:

```bash
python genesis_rl/scripts/train.py --task wuji_synergy --num_envs 1024
```

A short smoke test:

```bash
python genesis_rl/scripts/train.py --task wuji_synergy --num_envs 16 --max_iterations 3
```

Watch learning in a window. Keep the env count small; drawing thousands of
robots is slow.

```bash
python genesis_rl/scripts/train.py --task wuji_synergy --viz kit --num_envs 16
```

| `--viz` | |
|---|---|
| omitted | headless |
| `kit` | RTX viewport |
| `newton` | Newton viewer |
| `rerun` | no RTX |
| `viser` | browser, including from another machine |

Checkpoints are written every 50 iterations to
`logs/rsl_rl/<experiment>/<timestamp>/`. That directory is gitignored and
stays on the machine that trained.

```bash
tensorboard --logdir logs/rsl_rl
```

Resume:

```bash
python genesis_rl/scripts/train.py --task wuji_reach_grasp --resume \
    --checkpoint_path logs/rsl_rl/adam_u30_wuji_reach_grasp_L0/<run>/model_N.pt
```

Some tasks read an environment variable that changes the scene or the log
folder:

| task | variable | effect |
|---|---|---|
| `wuji_synergy` | `ADAM_U_FINGERS` | `2`, `3` (default), or `4` |
| `wuji_pipeline` | `ADAM_U_LEVEL`, `ADAM_U_STAGE` | curriculum level and stage |
| `wuji_reach_grasp` | `ADAM_U_RG_LEVEL` | level `0`–`8` |
| `wuji_reach`, `wuji_grasp` | `ADAM_U_WORKCELL` | `single` (default) or `bin` |

## Test a policy

`play.py` rolls a checkpoint out in Isaac Lab. Pass the same `--task` used
for training.

```bash
python genesis_rl/scripts/play.py --task wuji_synergy --viz kit \
    --num_envs 4 --real-time \
    --checkpoint_path logs/rsl_rl/adam_u30_wuji_synergy_f3/<run>/model_N.pt
```

`--real-time` plays at wall clock. `--speed 0.25` slows it down.
`--checkpoint_path` loads a checkpoint from this image. `--policy` loads a
TorchScript export instead.

Per-episode success numbers for a finished run are in
[`docs/COMMANDS.md`](docs/COMMANDS.md).

## Add a task

1. Create `genesis_rl/tasks/<name>/env_cfg.py` with the Isaac Lab env config.
2. Add a PPO runner config in `genesis_rl/algos/rsl_rl/rsl_rl_ppo_cfg.py`,
   or reuse one that matches the network you want.
3. Register it in `genesis_rl/tasks/__init__.py` with `_add(...)`: gym id,
   short name, env class, PPO class, and experiment directory.

Then:

```bash
python genesis_rl/scripts/train.py --task <name> --num_envs 16 --max_iterations 3
```

## Layout

```
genesis_rl/          tasks, PPO configs, train.py, play.py
assets/              robot URDF, meshes, USD, and the workcell
pyproject.toml       package name genesis-rl
Dockerfile           Isaac Sim 6.0 / Isaac Lab 3.0
docker-compose.yml   service adam_rl, repo mounted at /workspace
tools/               asset builders and scene probes, at /opt/tools in the container
docs/                experiment notes and the full command list
logs/                checkpoints (gitignored)
```

## Notes from earlier training

[`docs/EXPERIMENTS_SUMMARY.md`](docs/EXPERIMENTS_SUMMARY.md) is the short
history. [`docs/TRAINING_LOG.md`](docs/TRAINING_LOG.md) has the curves and the
commands that produced them. [`docs/ALGORITHM.md`](docs/ALGORITHM.md) is the
synergy grasp, end to end. [`docs/COMMANDS.md`](docs/COMMANDS.md) is every
probe, eval, and asset rebuild.
