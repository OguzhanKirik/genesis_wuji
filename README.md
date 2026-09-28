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

**What to do next**, and the state of every experiment, is in
[`docs/TODO.md`](docs/TODO.md).

**Every command** — start, test, train, evaluate, watch, rebuild assets — is
collected in [`docs/COMMANDS.md`](docs/COMMANDS.md).

**How the learning itself works** — the policy's observations, the synergy
action mapping, the episode state machine, the exact reward equation, how
success is judged, and how to read the training logs — is documented separately
in [`docs/ALGORITHM.md`](docs/ALGORITHM.md).

## Layout

```
docs/TODO.md                state of every experiment and what to do next
docs/TRAINING_LOG.md        run-by-run numbers, curves, and commands
docs/ALGORITHM.md           how the RL algorithm works, end to end
docs/COMMANDS.md            every command: test, train, evaluate, watch
Dockerfile                  Isaac Sim 6.0 / Isaac Lab 3.0
docker-compose.yml          one service, `adam_rl`
docker/
  requirements.lock.txt     258 pins, frozen from a verified environment
  entrypoint.sh             CUDA, USD and X11 preflight
tools/                      diagnostics and asset builders, mounted at /opt/tools
adam_u_isaac_lab/           the installable package (pyproject.toml); container /workspace
  genesis_rl/
    tasks/                  one folder per training task, registered in tasks/__init__.py
      pnd_grasp/            Inspire hands, cube            --task pnd_grasp
      wuji_reach/           Wuji hand, arm must reach      --task wuji_reach
      wuji_grasp/           hand starts over the bolt      --task wuji_grasp
      wuji_synergy/         hand only, two synergies       --task wuji_synergy
      wuji_pipeline/        IK wrist + synergies           --task wuji_pipeline
      wuji_reach_grasp/     standoff approach + lift       --task wuji_reach_grasp
      wuji_teleop/          20-DoF glove env, not registered (no runner config yet)
    mdp/observations.py     shared end-effector observation terms
    algos/rsl_rl/           PPO runner configs and MaskedPPO
    scripts/train.py, play.py   --task, or the older --robot/--scenario flags
    envs/                   old import paths; they re-export the task modules
    paths.py                checkout root, assets/, logs/ — not derived from file depth
  assets/robots/adam_u/
    urdf/, meshes/          URDF + 33 DAE + 14 STL (meshes sanitized)
    usd/                    committed flat USD -- what actually gets spawned
  assets/robots/adam_u_wuji/
    urdf/, meshes/          generated URDF + 61 OBJ visuals + 52 STL hulls
    usd/                    committed flat USD (13 MB), 60 bodies
  assets/robots/wuji_hand_right/
    urdf/, usd/             the right hand alone (Experiment 2); meshes shared
  assets/workcell/          adam-moveit's table, bin and bolt: workcell.json,
                            workcell_single.json, rod-with-threads-0001/
  logs/                     checkpoints and tensorboard runs (gitignored)
```

The repo is standalone: nothing outside it is needed to build or run.
`git log -p` is the record of every change and why.

`genesis_rl` is the reusable piece. A new training run is a folder under
`genesis_rl/tasks/` and one entry in `genesis_rl/tasks/__init__.py` (gym id,
environment class, PPO config, experiment directory). `train.py` and `play.py`
look the task up; they do not grow a new branch. The old
`from genesis_rl.envs.<file> import ...` paths still work, so the tools under
`tools/` did not have to move. Robot meshes and USD stay in `assets/`: they
are data, and the container already expects them at `/workspace/assets`.

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
python genesis_rl/scripts/train.py --num_envs 16 --max_iterations 3

# real run, headless
python genesis_rl/scripts/train.py --num_envs 4096

# train with the RTX viewport open
python genesis_rl/scripts/train.py --viz kit --num_envs 16

# watch a trained policy
python genesis_rl/scripts/play.py --viz kit --num_envs 4 --real-time \
    --checkpoint_path logs/rsl_rl/adam_u30_grasp/<run>/model_1499.pt

# confirm the robot spawns upright
python /opt/tools/compare_pose.py
```

Or one-shot, without leaving a container behind:

```bash
docker compose run --rm adam_rl python genesis_rl/scripts/train.py --num_envs 4096
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
frame targets          all 3 FrameTransformers resolve (waistRoll, thumb, pinky)
throughput             22,300 steps/s at 4096 envs, 5.4 GB of 96 GB VRAM
                       -> 1500 iterations in ~2.5 h
finger travel          1.8 cm fingertip-to-palm across the action range
                       (0.7 mm before the action fix)
```

The port itself was verified against the retired 2.3.2 environment by matching
iteration-0 reward — **1.21 on both**, same random policy, same physics, which
is the evidence that it is faithful and not merely running. That comparison is
no longer reproducible: the reward rebuild changed both the terms and the finger
action semantics, so iteration 0 now starts near 1.40 on a different budget.

A 300-iteration run on the retired 2.3.2 environment reached mean reward 30.81
at 28,487 steps/s (19.7 M timesteps), with `distance_to_object` 1.85 of ~2.0 and
`object_height` **0.0139 of 5.0** — the policy reached toward the cube and
stopped. That run is what diagnosed the reward, and the two causes it exposed
are the [finger action range](#gotchas-that-cost-real-time) and the reach
scaling; both are written up under [the reward rebuild](#the-reward-rebuild).

There is no post-fix training result here yet. Add one when a run finishes
rather than assuming the fix worked — the failure mode this repo has already
demonstrated twice is an environment that trains happily on the wrong
problem.

## Gotchas that cost real time

**The hand could not close, and no reward could have fixed it.** Upstream drove
the 12 right-hand finger joints with
`JointPositionActionCfg(scale=0.2, use_default_offset=True)`. Every finger joint
has limits `[0, 1.1 .. 1.7]` rad with its default at **0 = fully open**, so the
commandable target was `default + 0.2*a` — i.e. **0 to 0.2 rad, about 12% of the
closing range** — and the entire negative half of the action space clipped at
the lower limit and did nothing at all. Measured with `tools/probe_grip.py`:
sweeping the finger actions from −1.0 to +1.0 moved the mean fingertip-to-palm
distance by **0.7 mm** (0.15632 → 0.15558 m). A grasp was not merely
under-rewarded, it was kinematically unreachable.

Now `JointPositionToLimitsActionCfg(rescale_to_limits=True)`, which maps
`[-1, 1]` onto each joint's own soft limits, so −1 is fully open and +1 fully
closed across all 12 joints despite their differing ranges (thumb MCP2 0.5 rad,
index MCP 1.7 rad). Re-measured: joints now reach 1.32–1.60 rad and fingertip
travel is **1.8 cm**, 25× the old range. Note this changes the neutral action —
0.0 is now mid-range, not the open default.

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
`policy` model config into separate `actor` and `critic`. As of the reward fix,
anything trained before it is incompatible for a second and more basic reason —
the finger action semantics changed, so the same 19 numbers mean something
different at the actuators. `play.py` accepts a
TorchScript export via `--policy` as a version-independent path, but note that
`get_right_eef_quat` returns `body_quat_w`, whose component order *also*
changed — so 4 of the 70 inputs would be permuted. Retrain rather than
transfer.

## The Wuji hands variant

Adam-U is getting **Wuji Hand 2** instead of the PND/Inspire hands. adam-moveit
already carries that robot (`design:=genesis_wuji`: the Genesis_SIM_test_003
delivery, de-flattened, hull collisions; see its `docs/genesis_wuji_design.md`)
and the workcell the bolt-picking task uses. This repo vendors both, so the
second robot lives here beside the first:

|  | PND (`--robot pnd`, default) | Wuji (`--robot wuji`) |
|---|---|---|
| env cfg | `adam_u_grasp_env_cfg.py` | `adam_u_wuji_grasp_env_cfg.py` |
| hands | 12 joints each, 6 driven + 6 mimic | **20 independent revolute joints each**, no coupling |
| bodies / DoF | 44 / 43 | 60 / 59 (column frozen, fixed links merged) |
| object | 5 cm cube, 0.25 kg | adam-moveit's bolt (`rod-with-threads-0001`, 30 g) |
| table | 0.6 × 0.5 box, top at 1.025 | adam-moveit's 0.50 × 0.80 × 0.95 table, **top at 0.95**, optional bin |
| frame | robot at (0, −0.4, 1.0), 90° about Z | **adam-moveit's**: root at the origin facing +x, waistRoll at z 1.1267 |
| obs / actions | 70 / 19 (7 arm + 12 fingers) | 101 / 27 (7 arm + 20 fingers) |
| experiment dir | `logs/rsl_rl/adam_u30_grasp` | `logs/rsl_rl/adam_u30_wuji_grasp` |

```bash
python genesis_rl/scripts/train.py --robot wuji --num_envs 4096          # bolt on the bare table
ADAM_U_WORKCELL=bin python genesis_rl/scripts/train.py --robot wuji --num_envs 4096   # bolt inside the bin
python genesis_rl/scripts/play.py --robot wuji --viz kit --num_envs 4 --real-time \
    --checkpoint_path logs/rsl_rl/adam_u30_wuji_grasp/<run>/model_N.pt
python /opt/tools/probe_wuji.py                                          # measure before you trust it
```

`ADAM_U_WORKCELL` picks `workcell_single.json` (default: one bolt on the
table, no bin) or `workcell.json` (the bin is authored — floor slab and four
8 mm walls, exactly as adam-moveit's `isaac_runner.author_workcell` — and the
bolt starts inside it). Both files are adam-moveit's, byte for byte
(`assets/workcell/SOURCE.md` has the hashes). One deliberate change: adam-moveit
puts the bin at y = +0.20, in front of the **left** arm, because that stack
grasps left-handed. This env trains the right arm, so `WORKCELL_SIDE=right`
(default) mirrors y to −0.20. Set `WORKCELL_SIDE=left` for the layout as
delivered.

### Regenerating the Wuji robot

Two steps, two containers, both idempotent:

```bash
# 1. host: adam-moveit's xacro -> standalone URDF + the 171 meshes it references
python3 tools/make_wuji_urdf.py            # --check verifies instead of writing

# 2. Isaac Lab 2.3.2 (adam-wbc's image): URDF -> flat USD
docker run --rm --gpus all \
    -v $PWD/adam_u_isaac_lab:/workspace -v $PWD/tools:/opt/tools:ro \
    -v adam-wbc_isaac-cache-kit:/opt/venv/lib/python3.11/site-packages/isaacsim/kit/cache \
    -v adam-wbc_isaac-cache-ov:/root/.cache/ov -v adam-wbc_isaac-cache-nvidia:/root/.nv \
    --entrypoint python adam-lite-wbc:isaaclab-2.3.2 /opt/tools/export_usd_wuji.py --force
```

Step 1 changes exactly three things about adam-moveit's file and records them
in the URDF header: `package://` mesh URIs become paths relative to the URDF,
the two prismatic lifting-column joints become fixed at q=0 (nothing drives the
column; adam-moveit's own Isaac stage freezes them after the torso slid down
the column under gravity), and the xacro fragment gets a `<robot>` root. The
header also pins the adam-moveit commit and the xacro's sha256. Three column
OBJs name an `.mtl` the delivery never shipped, so the column renders grey.

Step 2 is the same [flat-USD requirement](#the-usd-is-committed-on-purpose) as
the PND robot, and the 2.3.2 image is still on this machine as
`adam-lite-wbc:isaaclab-2.3.2`. The converter merges the 24 fixed joints:
palms into `wristRoll{Left,Right}`, the ten fingertip and ten tip-sensor links
into the distal links, the column pieces into `lifting_Columns_base`. So the
fingertips are not bodies; the env puts them back as FrameTransformer offsets
along the distal link's −z (thumb 29.78 mm, fingers 24.75 mm — the URDF's tip
joint origins).

### What was measured (2026-09-19, `tools/probe_wuji.py`)

| | measured | note |
|---|---|---|
| bodies / joints / obs / actions | 60 / 59 / 101 / 27 | both layouts |
| robot root z | 0.0502 | `1.05 − 0.99975`, adam-moveit's anchor |
| waistRoll world z | 1.1267 | matches adam-moveit; the robot spawns upright |
| wrist at the init pose | (0.03, −0.28, 1.019) | hand hangs beside the torso, fingers down, clear of the table (front edge x 0.30) |
| bolt settled height above surface | +0.0110 m, identical in every env | `BOLT_REST_OFFSET`; lying on its side, body frame at mid-height |
| finger action +1 | **closes**: mean tip-to-palm 0.092 → 0.056 m, closure 0 → 1.00 | measured twice inside one episode |
| finger action −1 | hyperextends to the lower limits (−1.05 rad) | closure metric reads 0, as intended |
| reward terms at rest | reach_coarse 0.06–0.07, hand_height +0.05, everything else 0 | no term pays for doing nothing except `alive` |
| 1024 envs, headless | 1.18 s / iteration after the first, 32 s scene start, ~1 GB GPU | the 327k-face bolt is instanced; not a cost |
| `train.py --robot wuji`, 16 envs × 3 iterations | completes, checkpoints land in `adam_u30_wuji_grasp/` | both layouts |

### Decisions specific to the Wuji hand

- **Flexion is positive and open is q = 0.** The lower limits (−1.047 rad) are
  hyperextension, not "open". The PND env's closure formula
  `(q − lower)/(upper − lower)` would call an open Wuji hand 40 % closed, so
  this env uses `q / upper` over the 15 flexion joints and ignores the 5
  abduction joints. adam-moveit's FK check and its SRDF `closed` state agree
  on the sign; the probe confirms it.
- **Pinch, not power grasp.** The reach and grasp gates use the thumb and index
  fingertips (the PND env used thumb and pinky on a cube). adam-moveit treats
  the Wuji hand as a thumb–index gripper, and a bolt is a pinch object.
- **Finger actions map to the full joint range** (`JointPositionToLimitsActionCfg`,
  `rescale_to_limits`), like the PND env after its fix. Action 0 is therefore
  the mid-range, a slightly curled hand, not the open default.
- **Self-collision is off.** adam-moveit measured the thumb's base housing
  nesting 2.6–4.1 mm into the forearm hulls at the home pose. With
  self-collision on PhysX would fight that every step.
- **Physics 120 Hz, control 30 Hz** (decimation 4). adam-moveit's Isaac stage
  diverged to joint angles of 1e8 rad when a Wuji pinch pressed a bolt into the
  static table at 60 Hz and was stable at 240. Raise `sim.dt` resolution first
  if grasps blow up.
- **Finger drives: stiffness 3 N·m/rad, damping 0.1, armature 1e-4, effort
  limits from the URDF** (0.2–2 N·m by joint type, as a dict per regex). The
  vendor's published 0.18–0.69 N·m/rad are for its own hardware loop and are
  far too soft for a PhysX position drive on gram-scale links.
- **Isaac Lab 3.0 asset data are warp `ProxyArray`s, not tensors.** Their
  `.shape` is the warp shape — a quaternion array reports `(N,)` — so a custom
  observation term that returns one raw makes the observation manager fail with
  `stack expects each tensor to be equal size`. Torch ops unwrap them; anything
  returned or indexed goes through `.torch` (the `_t()` helper in the env).

### The grasp-only scenario (`--robot wuji --scenario grasp`)

Skips the reach: the torso leans 0.35 rad forward and the right arm starts
with the hand **palm down over the bolt**, fingers pointing forward and lightly
pre-curled, so the policy spends its samples on closing and lifting. Same
robot, table and bolt; `adam_u_wuji_grasponly_env_cfg.py` subclasses the
reach env and changes only the start pose, the bolt spot, the actions, the
rewards and the episode length (5 s). PPO is unchanged (`rsl_rl_ppo_cfg.py`).

```bash
python genesis_rl/scripts/train.py --robot wuji --scenario grasp --num_envs 4096
python genesis_rl/scripts/play.py  --robot wuji --scenario grasp --viz kit --num_envs 4 --real-time \
    --checkpoint_path logs/rsl_rl/adam_u30_wuji_grasponly/<run>/model_N.pt
python /opt/tools/probe_wuji.py --scenario grasp
```

| | value | how it was fixed |
|---|---|---|
| bolt | `BOLT_XY = (0.39, −0.21)`, ±1.5 cm, random yaw | where a scripted close from the start pose lifts it (below) |
| start pose | `PREGRASP_ARM` (7 arm joints + `waistPitch` 0.35) | `tools/solve_pregrasp_pose.py --waist 0.7 --seeds 30 --target 0.315 -0.215 1.035`: least-squares IK on the URDF, palm normal down, fingers horizontal, 0.1 rad inside every joint limit |
| fingers at start | `PREGRASP_FINGERS`: flexion ≈30 % of the limits, thumb swung to −0.95 rad | `tools/hand_geometry.py` for the curl; the thumb value from the probe's phase-B sweep (9/16 lifts at −0.95, 0–1/16 at −0.40, 4/16 at −1.17) |
| finger actions | offset from the pre-curl, per-joint scale = limit − pre-curl | so action 0 is the start shape and +1 fully flexed |
| arm actions | 7 joints, ±0.2 rad around the start pose | enough to descend and lift (~4 cm wrist rise measured), not to wander |
| rewards | fine reach, gated closure, lift bonus, height ramp capped at 8 cm, off-table, wrist drift, action rate | the coarse reach term is gone; 8 cm is what the arm range can lift |

Three things had to be found out before this pose existed, all with the host
tools and confirmed in the simulator:

- **A fingers-down hand cannot pinch a bolt off a table.** The thumb and index
  tips meet 8.5 cm below the wrist (`hand_geometry.py`), so with the tips at
  table height the pinch point is 8 cm in the air. Palm down, fingers forward
  puts the closing motion where the bolt is.
- **Palm down at 0.95 m needs the waist.** The shoulder is at z ≈ 1.52 and the
  straight arm just reaches the table, so every palm-down solve stopped ~5 cm
  high with the wrist roll on its stop. A 0.35 rad forward lean fixes it.
- **Unactuated joints were not holding their defaults.** Isaac Lab 3.0
  initialises joint drive targets to zero, and the reset events only write
  joint *states*, so the leaned waist fell back to 0.03 rad and the hand sat
  10 cm high. `hold_default_joint_targets` (a reset event in the base Wuji env)
  writes the default pose as the drive target. Measured before/after:
  waistPitch 0.030 → 0.410 against a 0.352 default.

Two more measurements that shaped it: the arm drives now use the URDF effort
limits (65/30/6.4 N·m) with stiffness 300, because the PND env's 10 N·m cap let
the outstretched hand sag 4 cm; and the grasp this hand makes is **not a
fingertip pinch** — during a close the thumb and index tips never get closer
than ~3 cm, the fingers curl back under the bolt and trap it against the thumb
and palm. That is why the bolt spot is where a scripted close *lifts*, not
where the tips meet. The probe's phase B measures it: from the configured
start pose a naive close-then-lift (no arm adjustment) lifted **5 of 16**
placements on a ±2 cm grid, so the policy has a basin to find and ±0.2 rad of
arm to find it with.

**First training run (2026-09-19, 4096 envs × 600 iterations, ~35 min):** the
policy learned the grasp. Per-episode evaluation over 512 episodes
(`tools/eval_grasponly.py`): the bolt is lifted in 100 %, still held when the
5 s episode ends in 99.4 %, held for the whole last second in 95.5 %; dropped
0.6 %, knocked off 0.2 %; mean peak lift 8.4 cm. The breakthrough came between
iterations 300 and 400. Full curve and settings in `docs/TRAINING_LOG.md`.

### Experiment 2: finger-only synergy grasp (`--robot wuji --scenario synergy`)

One Wuji hand, fixed wrist, one bolt, two closure synergies, tactile
observations, a scripted 10 cm lift and a held-for-1-s success. Built as
`wuji_synergy_grasp_env_cfg.py` on a **hand-only articulation**
(`assets/robots/wuji_hand_right/`, cut out of the full URDF by
`tools/make_wuji_hand_urdf.py`; a prismatic `wrist_lift` root joint is the
scripted lift). Full design, measurements and results in `docs/TRAINING_LOG.md`.

```bash
python genesis_rl/scripts/train.py --robot wuji --scenario synergy --num_envs 1024 --max_iterations 500
python /opt/tools/eval_synergy.py --checkpoint_path logs/rsl_rl/adam_u30_wuji_synergy/<run>/model_499.pt
# the same, but with the Isaac Lab window open and slowed down so you can watch
python /opt/tools/eval_synergy.py --viz kit --num_envs 16 --episodes 16 \
    --real-time --speed 0.25 --start-delay 15 --checkpoint_path <ckpt>
python /opt/tools/probe_synergy.py                  # scripted close from the configured pose
python3 tools/find_pinch_pose.py --middle           # host-side: re-solve the pre-grasp pose
```

`ADAM_U_FINGERS` picks how many fingers oppose the thumb: 2 (index), 3
(index + middle, the default) or 4 (the whole-hand rake).

**The pre-grasp pose is solved, not guessed.** `find_pinch_pose.py` walks the
synergy stroke with URDF forward kinematics and measures where the fingertips
actually converge. The gap, the opposition angle and the fingertip-to-bolt
distance do not depend on how the hand is *rotated*; only table clearance does.
So the tool places the wrist so first contact with the 2.5 cm bolt happens at
the bolt, then searches for an orientation that keeps every fingertip above the
table across the whole stroke. That search is what makes a pinch possible at
all: fingers-down and palm-down both fail from a fixed wrist, because the
fingers hang ~17 cm below the wrist when open and curl up to ~9 cm as they
close, so any wrist height that puts the closing pinch at the bolt buries the
open fingertips under the table. Experiment 1 only worked because its arm rose
while the hand closed.

Two defects it uncovered, both invisible in the rendered scene. The prismatic
lift axis is expressed in the hand's own frame and rotates with it, so once the
hand was tilted into the pinch pose a nominal vertical lift travelled sideways
and dragged the bolt 7 cm across the table while reporting a 10 cm rise; the
axis is now baked into the URDF and the env asserts it maps to world +z. And
the fingers outside the synergy were curled, which in this tilted pose puts
their tips *down* onto the table; they are held flat.

A scripted full close from the solved pose achieves **16/16 stable grasps**
under the spec's rules before PPO is involved.

**Results (1024 envs × 500 iterations, ~6 min each).** Three fingers reaches
**100 % stable grasps** over 512 evaluation episodes: no drops, no table
collisions, no force violations, the bolt clear of the table throughout
verification. Two fingers reaches **0 %**. The scripted sweep shows a
two-finger grasp does exist, but at one closure combination out of sixteen
against four for three fingers, and PPO never lands in it — the contact
bonuses are collected in full by exploration noise while the policy's mean
stays fully open, so they carry no gradient, and +20 for success is never
reached by chance. Details and the three ways out are in `docs/TRAINING_LOG.md`.

### Not yet done for the Wuji variant

- **Reach scenario: no real training run yet.** Only the smoke runs above.
  The reward weights are carried over from the PND cube task unchanged, and
  the rest heights and thresholds were re-derived for the bolt; expect to
  retune once a run exists. The grasp-only scenario's first 4096-env run is
  recorded in its section.
- **The bin layout is untested as a task.** The bin's inner width is 16 cm and
  its walls 8 cm tall; whether the hand can reach a bolt inside without the
  walls in the way is not known. Start with `single`.
- **Body inertials are placeholders** in the delivery (torso ixx = mass/6);
  hand links have real tensors. adam-moveit substitutes PNDbotics' tensors for
  the 19 body links and this URDF inherits that. The three column links keep
  1 kg isotropic placeholders; they are fixed, so it does not matter here.
- **`compare_pose.py` and the PND probes do not apply** to this robot; use
  `probe_wuji.py`.

## What upstream gives you, and what it does not

Working, and worth keeping:

- URDF loads, 43 joints and 44 links import, actuator groups are sensible.
- Frame transformers on the right wrist, thumb tip and pinky tip.
- A shaped reward that is genuinely three-stage: reach (tanh distance to thumb
  and pinky), raise the hand, then lift the cube.

Fixed here — see [the reward rebuild](#the-reward-rebuild):

1. ~~**Nothing rewards closing the fingers.**~~ The deeper cause was the finger
   action range above; a `grasp` term now pays for closure gated on proximity.
2. ~~**Nothing penalizes knocking the cube off the table.**~~ `off_table` term
   plus an `object_dropped` threshold raised from 0.5 m to 0.95 m.
3. ~~**`distance_to_object` uses std = 1.0 against a 5 cm cube.**~~ Replaced by
   a two-scale reach; the straddle exploit is now countered by `grasp`, though
   the reach terms still read only the thumb and pinky tips.

Still open:

1. **The observation space omits the object's orientation** and any
   hand-relative object vector — the policy sees `object_position` and
   `right_hand_pos` in *world* frame and must learn the subtraction.
   `envs/utils/observations.py` already defines `object_obs()`, which returns
   exactly the hand-relative deltas, and nothing calls it. Left alone on
   purpose: it changes the observation width from 70, so it wants its own run
   to attribute the effect.
2. **`reset_robot_joints` randomizes every joint**, not just the right arm,
   including the 22 finger joints and the neck, at ±0.1 rad. Now slightly worse
   than it looks: finger defaults sit *on* their lower limit, so half of that
   ±0.1 rad clips away.
3. **`num_steps_per_env = 32` with `episode_length_s = 10.0`** at 30 Hz control
   (`dt = 1/60`, `decimation = 2`) means 300 steps per episode against 32-step
   rollouts — nearly ten updates per episode, so early value estimates are
   heavily bootstrapped.
4. **`wrist_actuators` have `damping=0.0`** where other groups use 10–20.
   Undamped position drives on a 7-DoF arm invite oscillation; suspect this
   first if the wrist chatters.

### The reward rebuild

The 300-iteration run reached mean reward 30.81 while never touching the cube.
Reading the tensorboard scalars back out, it scored **3.105 of the 8.250-point
episode-normalised budget (37.6%)** — and three of the four positive terms were
saturated, with the fourth (`object_height`, 61% of the budget) paying 0.3%.

The reach term was the reason. At `std = 1.0` against a 5 cm cube the whole task
sat in the linear toe of the tanh, so the slope was a flat **−0.02 points/cm
everywhere from 0 to 80 cm**. The policy parked **7.7 cm from the cube centre**
— 5.2 cm clear of its surface — banked 92.3% of the term there, and closing that
last gap was worth 0.104 points. Simply lowering `std` would have broken the
other end: at the 0.23–0.34 m spawn distance, `std = 0.05` gives ~0 reward *and*
~0 slope.

| term | before | after |
|---|---|---|
| reach | one term, `std=1.0`, weight 2.0 | `reach_coarse` `std=0.30` w 1.0 + `reach_fine` `std=0.04` w 2.0 |
| grasp | *(none)* | `grasp` = proximity gate × finger closure, w 2.0 |
| lift | `object_height` ramp only, cap 0.5 m | `lift_bonus` sparse step at +2 cm, w 3.0 |
| height | cap 0.5 m (a 50 cm lift), w 10.0 | cap 0.2 m, w 10.0 |
| sweep-off | *(none)* | `off_table` w −2.0, drop threshold 0.5 → 0.95 m |

Budget goes 8.25 → 11.25, and the share reachable without touching the cube
drops from 37.6% to about 20%. `tools/probe_reward.py` validates each term at
scripted states — the numbers that matter: `grasp` pays **0.0000** for a closed
hand in free space (the gate works) versus **0.5038** closed at the cube and
**0.0042** open at the cube, and `reach_fine`'s slope at the cube surface is
**0.355 points/cm** against the old 0.020.

The closure sign is measured, not assumed. All finger joints open at their lower
limit and close toward the upper, so normalising position onto `[lower, upper]`
is a closure fraction; inverted, the term would have paid to splay the hand open
on the cube.

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
