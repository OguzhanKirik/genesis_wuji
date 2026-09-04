# Adam-U RL

Reinforcement learning for the PNDbotics **Adam-U** (44 links, 43 DoF: a torso,
two 7-DoF arms and two 12-DoF Inspire hands on a `lifting_Columns` base) in
Isaac Lab.

The task is **right-arm grasping**: lift a 5 cm cube off a table. This is
manipulation, not locomotion — the sibling repo `adam-wbc` covers whole-body
locomotion for Adam Lite, and this repo started from its Docker structure.

Upstream is [pndbotics/adam_u_isaac_lab](https://github.com/pndbotics/adam_u_isaac_lab),
vendored at `bc84036` under `adam_u_isaac_lab/`. Upstream calls itself "a
minimal example ... simplified for clarity and should be re-designed for
production research" — take that at face value; see
[What upstream gives you](#what-upstream-gives-you-and-what-it-does-not).

## One environment

Isaac Sim **6.0** / Isaac Lab **3.0**, which both trains and renders.

| | |
|---|---|
| GPU | RTX PRO 6000 Blackwell, 96 GB, sm_120 |
| Python | 3.12 |
| torch | 2.10.0+cu128 — includes sm_120 |
| Isaac Sim | 6.0.0.1 |
| Isaac Lab | 3.0, commit `28a37ce` (**beta**) |
| rsl-rl-lib | 5.0.1 |
| numpy | 2.5.2 |

Isaac Sim **5.1 cannot render on this card** — it segfaults at 2.64 s in
`librtx.scenedb.plugin.so!carbOnPluginStartup`, reproduced here and in
`adam-wbc`. 6.0 renders on the same card (`app ready` in ~9 s). Since 5.1 could
never show you anything, 6.0 is the only version that does both jobs, so it is
the only one kept. The repo briefly carried a second 5.1 / Isaac Lab 2.3.2
training image; it is in git history, and two things it left behind still
matter — the [committed USD](#the-usd-is-committed-on-purpose) and the
[quaternion fix](#gotchas-that-cost-real-time).

Isaac Lab 3.0 is beta and its release notes warn of continuing breaking
changes. Expect this to need repair on update; this repo's git history is a
worked example of what that looks like.

## Layout

```
Dockerfile                  Isaac Sim 6.0 / Isaac Lab 3.0
docker-compose.yml          one service, `adam_rl`
docker/
  requirements.lock.txt     258 pins, frozen from a verified environment
  entrypoint.sh             CUDA, USD and X11 preflight
tools/
  sanitize_dae.py           makes the Collada meshes legal for USD import
  compare_pose.py           checks the robot spawns upright  <- run after any
                            change to init_state or the USD
  export_usd.py             URDF -> flat USD (needs the 2.3.2 image; history)
  probe_urdf_30.py          shows why 3.0 cannot import this URDF directly
adam_u_isaac_lab/
  adam_u_rl/
    envs/adam_u_grasp_env_cfg.py    scene, actions, observations, rewards
    envs/utils/observations.py      end-effector pose observation terms
    algos/rsl_rl/rsl_rl_ppo_cfg.py  PPO hyperparameters (rsl-rl 5.x schema)
    scripts/train.py, play.py       entry points
  assets/robots/adam_u/
    urdf/, meshes/          URDF + 33 DAE + 14 STL (meshes sanitized)
    usd/                    committed flat USD -- what actually gets spawned
  logs/                     checkpoints and tensorboard runs (gitignored)
```

The repo is standalone: nothing outside it is needed to build or run.
`git log -p` is the record of every change and why.

## Quick start

```bash
cd /home/revel/adam-rl
docker compose build            # ~20 min; the final image export is slow, not hung
xhost +local:root               # once per login, for the GUI

docker compose up -d
docker compose exec adam_rl bash
```

Then, inside:

```bash
# 30 s sanity check
python adam_u_rl/scripts/train.py --num_envs 16 --max_iterations 3

# real run, headless
python adam_u_rl/scripts/train.py --num_envs 4096

# train with the RTX viewport open
python adam_u_rl/scripts/train.py --viz kit --num_envs 16

# watch a trained policy
python adam_u_rl/scripts/play.py --viz kit --num_envs 4 --real-time \
    --checkpoint_path logs/rsl_rl/adam_u30_grasp/<run>/model_1499.pt

# confirm the robot spawns upright
python /opt/tools/compare_pose.py
```

Or one-shot, without leaving a container behind:

```bash
docker compose run --rm adam_rl python adam_u_rl/scripts/train.py --num_envs 4096
```

Checkpoints land on the host in
`adam_u_isaac_lab/logs/rsl_rl/adam_u30_grasp/<timestamp>/`, every 50 iterations.

```bash
tensorboard --logdir adam_u_isaac_lab/logs/rsl_rl
```

`--viz` takes a CSV of backends; omit it for headless.

| `--viz` | notes |
|---|---|
| *(omitted)* | headless — use this for real training runs |
| `kit` | RTX viewport. Verified working on this card |
| `newton` | Newton's own viewer |
| `rerun` | bypasses RTX entirely |
| `viser` | browser-based, so it also works from another machine |

`rerun` and `viser` are pinned deliberately: both skip RTX, so they are the
fallback if `kit` ever regresses the way 5.1 did.

Render 16 envs, not 4096. Drawing thousands of robots is slow and unreadable —
train big headless, watch small.

## Verified

On this machine:

```
robot spawns upright   waistRoll world_z 1.07670, wristRollRight 1.01262
                       (matches the 2.3.2 reference to within 5 mm)
articulation           44 rigid bodies, 43 joints
observations           70,  actions 19  (7 arm + 12 right-hand fingers)
GUI                    --viz kit, "app ready" in 9.1 s, trains with viewport open
training               reward 1.21 -> 2.34 -> 2.91 -> 3.65 -> 4.15 over 5 iters
```

Iteration 0 gives **1.21 on both** Isaac Lab 3.0 and the retired 2.3.2 setup —
same random policy, same physics — which is the evidence that the port is
faithful and not merely running.

A 300-iteration run on the retired 2.3.2 environment reached mean reward 30.81
at 28,487 steps/s (19.7 M timesteps), with `distance_to_object` 1.85 of ~2.0
and `object_height` **0.0139 of 5.0**. That is the headline result: the policy
learns to reach the cube and then stops. See point 3 below.

## Gotchas that cost real time

**Quaternion order changed in Isaac Lab 3.0.** `InitialStateCfg.rot` is
`(x, y, z, w)` in 3.0 and `(w, x, y, z)` in 2.3.2:

```
2.3.2   rot = (w, x, y, z)   default (1, 0, 0, 0)
3.0     rot = (x, y, z, w)   default (0, 0, 0, 1)
```

Upstream's `rot=(0.707, 0, 0, 0.707)` is 90° about **Z** under 2.3.2 but 90°
about **X** under 3.0 — so the robot spawns lying on its side, right hand
*below* the 1.025 m table. It trains perfectly happily on the wrong physical
problem: `hand_height` came out at **−0.657 instead of −0.037**, a 20× error,
and total reward went *negative and decreasing* (−0.29 → −0.90) where it should
climb. Nothing errors. `rot=[1,0,0,0]` on the table and cube is likewise a 180°
flip about X, not identity — harmless only because both are symmetric.

Every `rot` in the env config is now written `(x, y, z, w)` with a comment.
**Run `tools/compare_pose.py` after touching `init_state` or the USD** — it
prints the world height of the bodies the rewards read, which is how this was
caught.

**The Collada meshes have USD-illegal ids.** All 33 `.dae` files, 280 ids: dots
in node names (`L_middle_distal.001`), hyphens in mesh ids (`waistRoll-mesh`),
and CJK material names (`灰色-material`, `黑色-effect`). USD prim names allow
only `[A-Za-z0-9_]`, and the URDF importer turns node and material names into
prim names — so it produces a null prim and an articulation with **zero rigid
bodies**, surfacing much later as a confusing

```
ValueError: No contact sensors added to the prim: '.../Robot'.
            This means that no rigid bodies are present under this prim.
```

Fixed in the second commit; `tools/sanitize_dae.py` is idempotent. Adam Lite
needed the identical fix in `adam-wbc`.

### The USD is committed on purpose

Isaac Lab 3.0 **cannot import this URDF usably**. `tools/probe_urdf_30.py`
measures it: all 44 bodies import, but nested at prim depths 3–17, and the env
addresses three of them by explicit path (`{ENV_REGEX_NS}/Robot/waistRoll`,
`.../R_thumb_distal`, `.../R_pinky_distal`) for its FrameTransformers. Those
stop resolving and every distance reward silently loses its frames.

Isaac Lab 2.3.2's converter emits a **flat** hierarchy — 44 bodies as direct
children of the default prim — which 3.0 reads happily. So that converter's
output is committed under `assets/robots/adam_u/usd/` (~104 MB) and is what
gets spawned. `tools/export_usd.py` regenerates it and verifies the result is
flat with 44 bodies, but it needs the Isaac Sim 5.1 / Isaac Lab 2.3.2 image,
which this repo no longer builds — recover that `Dockerfile` from git history
if the URDF ever changes.

**`handle_deprecated_rsl_rl_cfg()` is not optional.** `RslRlMLPModelCfg` still
carries the pre-5.0 `stochastic` and `init_noise_std` fields, both `MISSING`.
Left in, they reach the model as kwargs and rsl-rl 5.0.1 dies with
`TypeError: MLPModel.__init__() got an unexpected keyword argument 'stochastic'`.
Isaac Lab's own train script calls the helper; a hand-written one must too.
Both scripts here do.

**`obs_groups` is required** on `RslRlBaseRunnerCfg` and `MISSING` by default;
`OnPolicyRunner` reads it before the first step. This env publishes only a
`policy` group, so the critic shares it.

**The image export is slow, not hung.** `docker compose build` finishes all
build stages and then sits at `exporting layers` for many minutes writing a
~48 GB image. It looks wedged and is not — measure `docker system df` growth
before concluding otherwise.

**6.0's first launch looks hung too.** It spends minutes building the shader
cache. The named cache volumes in `docker-compose.yml` mean it happens once.

**`shm_size: 16gb` is required.** PhysX fails oddly on Docker's 64 MB default,
and the errors never mention shared memory.

**`R_middle_proximal` is misnamed in the URDF.** The link is
`R_index_distalR_middle_proximal` — a string-concatenation slip upstream, since
the left side is plain `L_middle_proximal`. It is self-consistent and
USD-legal, so it loads and trains fine. Joint names (`R_middle_MCP_joint`) are
unaffected, so the `R_middle.*` regexes still work.

**Old checkpoints do not load.** The two `model_0.pt` files upstream ships in
`logs/` are `iter: 0` — untrained — and were written by rsl-rl 2.x. Anything
from the retired 2.3.2 environment is also incompatible: rsl-rl 4.0 split the
`policy` model config into separate `actor` and `critic`. `play.py` accepts a
TorchScript export via `--policy` as a version-independent path, but note that
`get_right_eef_quat` returns `body_quat_w`, whose component order *also*
changed — so 4 of the 70 inputs would be permuted. Retrain rather than
transfer.

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
   including the 22 finger joints and the neck, at ±0.1 rad.
3. **Nothing rewards closing the fingers.** This is the big one — measured, not
   theorised: after 300 iterations `distance_to_object` reached 1.85 of ~2.0
   while `object_height` sat at 0.0139 of 5.0. The policy reaches the cube and
   stops, because reaching is rewarded and gripping is not. Fix this before any
   long run.
4. **Nothing penalizes knocking the cube off the table** — only
   `object_dropped` at z < 0.5, far below the 1.025 m table surface, so the
   episode runs on after a sweep-off until the cube reaches the floor.
5. **`distance_to_object` uses the thumb and pinky tips only.** A policy can
   maximize it by straddling the cube without ever gripping.
6. **`num_steps_per_env = 32` with `episode_length_s = 10.0`** at 30 Hz control
   (`dt = 1/60`, `decimation = 2`) means 300 steps per episode against 32-step
   rollouts — nearly ten updates per episode, so early value estimates are
   heavily bootstrapped.
7. **`wrist_actuators` have `damping=0.0`** where other groups use 10–20.
   Undamped position drives on a 7-DoF arm invite oscillation; suspect this
   first if the wrist chatters.

## Unverified inputs — get these from PNDbotics

1. **Actuator gains.** `stiffness=80, damping=10` on the arm is a round number,
   not a measured value. Adam Lite's gains in `adam-wbc` were derived from
   rotor inertia via `Kp = J·ω²`; the equivalent numbers for Adam-U's arms and
   the Inspire hands are unknown.
2. **Effort limits.** Arm and wrist are both 10 Nm and the fingers 10 Nm — the
   fingers being as strong as the shoulder is certainly wrong.
3. **Authoritative joint limits**, and whether URDF and MJCF agree. They
   disagreed for Adam Lite; assume the same until checked.
4. **The `lifting_Columns` base.** `fix_base=True` at z = 1.0, rotated 90°
   about Z. Whether that matches the real torso height and table geometry has
   not been checked against hardware.

## Notes on the filesystem

The container runs as root, so `logs/` comes back root-owned on the host (same
as adam-wbc's `logs/sonic_eval/`). Delete such files from inside a container
rather than reaching for `sudo`.
