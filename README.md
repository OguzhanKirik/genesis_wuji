# Adam-U RL

Reinforcement learning for the PNDbotics **Adam-U** (44-link, 43-DoF humanoid
with two 11-DoF Inspire hands) in Isaac Lab.

The task is **right-arm grasping**: lift a 5 cm cube off a table. This is
manipulation, not locomotion — the sibling repo `adam-wbc` covers whole-body
locomotion for Adam Lite, and this repo copies its Docker structure.

Upstream is [pndbotics/adam_u_isaac_lab](https://github.com/pndbotics/adam_u_isaac_lab),
vendored at `bc84036` under `adam_u_isaac_lab/`. Upstream calls itself "a
minimal example ... simplified for clarity and should be re-designed for
production research" — take that at face value; see
[What upstream gives you](#what-upstream-gives-you-and-what-it-does-not).

## Layout

```
Dockerfile                  Isaac Lab 2.3.2 / Isaac Sim 5.1.0 training image
docker-compose.yml          single service `adam_rl`, GPU + caches + bind mounts
docker/
  requirements.lock.txt     280 pins, frozen from a verified environment
  entrypoint.sh             CUDA preflight + mesh repair on the bind mount
tools/sanitize_dae.py       makes the Collada meshes legal for USD import
adam_u_isaac_lab/           vendored upstream (see git history for the delta)
  adam_u_rl/
    envs/adam_u_grasp_env_cfg.py    scene, actions, observations, rewards
    envs/utils/observations.py      end-effector pose observation terms
    algos/rsl_rl/rsl_rl_ppo_cfg.py  PPO hyperparameters
    scripts/train.py, play.py       entry points
  assets/robots/adam_u/     URDF + 33 DAE + 14 STL meshes (sanitized)
  logs/                     checkpoints and tensorboard runs (gitignored)
  .cache/usd/               URDF->USD conversion cache (gitignored)
```

The repo is standalone: nothing outside it is needed to build or run, and the
git history is three commits — pristine upstream, the mesh fix, then the port.
`git log -p` is the record of everything changed and why.

## Environment

| | |
|---|---|
| GPU | RTX PRO 6000 Blackwell, 96 GB, sm_120 |
| Python | 3.11 (Ubuntu 24.04 ships only 3.12) |
| torch | 2.7.0+cu128 — includes sm_120 |
| Isaac Sim | 5.1.0 · Isaac Lab 2.3.2 · rsl-rl-lib 3.0.1 |

Upstream targets Isaac Sim 5.0. This repo runs 5.1 because that is the stack
already verified on this card in `adam-wbc`, and because Isaac Sim 5.0's torch
pin predates sm_120. Moving up one minor version is what the
[port](#the-port-to-isaac-lab-232) below is about.

## Quick start

```bash
docker compose build            # ~20 min, ~25 GB image
docker compose up -d
docker compose exec adam_rl bash

# 30 s sanity check
python adam_u_rl/scripts/train.py --headless --num_envs 16 --max_iterations 3

# real run
python adam_u_rl/scripts/train.py --headless --num_envs 4096
```

Or one-shot, without leaving a container behind:

```bash
docker compose run --rm adam_rl \
    python adam_u_rl/scripts/train.py --headless --num_envs 4096
```

Checkpoints land on the host in `adam_u_isaac_lab/logs/rsl_rl/adam_u_grasp/<timestamp>/`,
every 50 iterations. Watch them with:

```bash
tensorboard --logdir adam_u_isaac_lab/logs/rsl_rl/adam_u_grasp
```

Replay a checkpoint (headless — see the renderer note below):

```bash
python adam_u_rl/scripts/play.py --headless --num_envs 4 \
    --checkpoint_path logs/rsl_rl/adam_u_grasp/<run>/model_1499.pt
```

`play.py` also writes TorchScript and ONNX exports next to the checkpoint, in
`exported/`.

## The port to Isaac Lab 2.3.2

Upstream runs on Isaac Sim 5.0 / Isaac Lab 2.2 / rsl-rl 2.x. Six changes were
needed for 5.1 / 2.3.2 / rsl-rl 3.0.1. All are in the third commit.

**`obs_groups` is now required.** `RslRlBaseRunnerCfg.obs_groups` is `MISSING`
by default, and `OnPolicyRunner.__init__` reads `cfg["obs_groups"]` before the
first step. Left unset, training dies in `resolve_obs_groups()`. This env
publishes only a `policy` group, so the critic shares it:
`{"policy": ["policy"], "critic": ["policy"]}`.

**`actor_obs_normalization` / `critic_obs_normalization` are new and also
`MISSING`.** Both are set to `True`, which is more than a compile fix: this
observation vector mixes object positions (~1 m) with joint velocities (~1e-2),
so an unnormalized critic is badly conditioned.

**`get_observations()` returns a `TensorDict`, not `(obs, extras)`.**
`play.py`'s `obs, _ = env.get_observations()` silently unpacked dictionary
*keys* instead of failing cleanly.

**The empirical normalizer moved from the runner onto the policy.**
`runner.obs_normalizer` no longer exists in rsl-rl 3.x; it is
`runner.alg.policy.actor_obs_normalizer`.

**`play.py --video` could never be turned off.** It was declared
`action="store_true", default=True`, and video forces `--enable_cameras`, i.e.
the RTX renderer — which crashes on this GPU (below). Default is now off, with
`--no-video` for scripts.

**`envs/utils/__init__,py`** — a comma, not a dot. The directory only ever
worked as an implicit namespace package, so the file's contents were dead code
importing an `actions` module that does not exist in the repo. Renamed and
trimmed to the one import that resolves.

Two more changes are conveniences rather than fixes: the URDF path and the log
directory are now derived from the repo layout instead of the process working
directory (upstream only worked when launched from the repo root), and the
URDF→USD conversion is cached — see below.

## Gotchas that cost real time

**The Collada meshes have USD-illegal ids.** All 33 `.dae` files, 280 ids:
dots in node names (`L_middle_distal.001`), hyphens in mesh ids
(`waistRoll-mesh`), and CJK material names (`灰色-material`, `黑色-effect`).
USD prim names allow only `[A-Za-z0-9_]`, and Isaac Sim's URDF importer turns
node and material names into prim names — so it produces a null prim and an
articulation with **zero rigid bodies**, surfacing much later as a confusing

```
ValueError: No contact sensors added to the prim: '.../Robot'.
            This means that no rigid bodies are present under this prim.
```

Fixed in the second commit; `tools/sanitize_dae.py` is idempotent, so re-run it
if the meshes are ever re-imported. The entrypoint re-checks the bind mount on
every start, because a `git checkout -- .` brings the bad ids straight back.
Adam Lite needed the identical fix in `adam-wbc`.

**Isaac Sim 5.1 cannot render on this GPU.** Any GUI or camera run dies ~2.4 s
in at `librtx.scenedb.plugin.so!carbOnPluginStartup`. Verified in `adam-wbc`
both native and in Docker; not an X11 or Docker problem, and not fixed by
forcing the NVIDIA Vulkan ICD. The RTX PRO 6000 Blackwell (sm_120) is newer
than Isaac Sim 5.1's renderer; Isaac Sim **6.0** renders on the same card.

Headless training is entirely unaffected — PhysX and CUDA never initialize RTX.
But it means **judge policies by metrics, not by looking at them**: track
`Episode_Reward/distance_to_object`, `/hand_height` and `/object_height` in
tensorboard. `object_height` rising off zero is the only real evidence of a
grasp. Visualizing would mean Isaac Sim 6.0, hence the beta Isaac Lab 3.0,
hence porting the env config — the same two-environment split `adam-wbc`
ended up with.

**The URDF is re-imported on every launch unless you cache it.** With
`usd_dir` unset, Isaac Lab's converter writes to a fresh random `/tmp`
directory each run, so a 44-link, 47-mesh robot is converted from scratch every
time. Now pinned to `.cache/usd/adam_u/`, which makes it hash-cached.

**`R_middle_proximal` is misnamed in the URDF.** The link is called
`R_index_distalR_middle_proximal` (and so is its mesh file) — evidently a
string-concatenation slip upstream, since the left side is plain
`L_middle_proximal`. It is self-consistent and USD-legal, so it loads and
trains fine; just do not expect the name to match the left hand's. Joint names
(`R_middle_MCP_joint`) are unaffected, so the `R_middle.*` regexes still work.

**`pink` must be importable even though nothing uses it.**
`adam_u_grasp_env_cfg.py` imports `pink.tasks.FrameTask` and Isaac Lab's
`PinkInverseKinematicsActionCfg` at module scope, but the action space is plain
joint-position control — the IK imports are dead. Rather than delete them (IK
actions are a plausible next step for grasping), the lock carries the whole
Pink stack: `pin`, `pin-pink`, `qpsolvers`, `daqp`, `quadprog` and nine
`cmeel-*` binary wheels, pinned via `uv pip compile` against the numpy/scipy
already in the lock.

**`shm_size: 16gb` is required.** PhysX fails oddly on Docker's 64 MB default,
and the errors never mention shared memory.

## What upstream gives you, and what it does not

Working, and worth keeping:

- URDF loads, 43 joints and 44 links import, actuator groups are sensible.
- Frame transformers on the right wrist, thumb tip and pinky tip.
- A shaped reward that is genuinely three-stage: reach (tanh distance to thumb
  and pinky), raise the hand, then lift the cube.

Simplified in ways that will limit results:

1. **The observation space omits the object's orientation** and any
   hand-relative object vector — the policy sees `object_position` and
   `right_hand_pos` in *world* frame and must learn the subtraction.
   `envs/utils/observations.py` already defines `object_obs()`, which returns
   exactly the hand-relative deltas, and nothing calls it.
2. **`reset_robot_joints` randomizes every joint**, not just the right arm,
   including the 22 finger joints and the neck. `position_range` is ±0.1 rad.
3. **Nothing rewards closing the fingers**, and nothing penalizes knocking the
   cube off the table — only `object_dropped` at z < 0.5, which is *below the
   table top* at 1.05 m, so the episode continues after a sweep-off until the
   cube falls to the floor.
4. **`distance_to_object` uses the thumb and pinky tips only.** A policy can
   maximize it by straddling the cube without ever gripping.
5. **`num_steps_per_env = 32` with `episode_length_s = 10.0`** at 30 Hz control
   (`dt = 1/60`, `decimation = 2`) means 300 steps per episode against 32-step
   rollouts — nearly ten updates per episode, so early value estimates are
   heavily bootstrapped.
6. **`wrist_actuators` have `damping=0.0`.** The other groups use 10–20.
   Undamped position drives on a 7-DoF arm invite oscillation; suspect this
   first if the wrist chatters.

None of this blocks training — it is what to change once a baseline exists.

## Unverified inputs — get these from PNDbotics

1. **Actuator gains.** `stiffness=80, damping=10` on the arm is a round number,
   not a measured value. Adam Lite's gains in `adam-wbc` were derived from
   rotor inertia via `Kp = J·ω²`; the equivalent numbers for Adam-U's arm and
   for the Inspire hands are unknown.
2. **Effort limits.** The arm and wrist are both set to 10 Nm and the fingers to
   10 Nm — the fingers being as strong as the shoulder is certainly wrong.
3. **Authoritative joint limits**, and whether the URDF's agree with the MJCF.
   The URDF and MJCF disagreed for Adam Lite; assume the same until checked.
4. **The `lifting_Columns` base.** It is `fix_base=True` at z = 1.0 with a
   0.707/0.707 quaternion (90° about x). Whether that matches the real torso
   height and the table geometry has not been checked against hardware.

## Native install (optional)

Docker is the day-to-day path. For a native venv, mirror the image:

```bash
uv venv --python 3.11 .venv && . .venv/bin/activate
uv pip install --no-deps "setuptools<81" wheel
uv pip install --no-deps --no-build-isolation \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    -r docker/requirements.lock.txt
git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git
for p in isaaclab isaaclab_assets isaaclab_mimic isaaclab_rl isaaclab_tasks; do
    uv pip install --no-deps -e IsaacLab/source/$p
done

export OMNI_KIT_ACCEPT_EULA=YES
python adam_u_isaac_lab/adam_u_rl/scripts/train.py --headless --num_envs 16 --max_iterations 3
```

Add packages with the lock as a constraint, or you will silently downgrade torch:

```bash
uv pip install --constraint docker/requirements.lock.txt <pkg>
```

## Tools

```bash
python tools/sanitize_dae.py <mesh_dir> [--dry-run]   # USD-legal Collada ids
```

Idempotent. Re-run it if the model is ever re-fetched from upstream.
