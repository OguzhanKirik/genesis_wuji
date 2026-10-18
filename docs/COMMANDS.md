# Command reference

## Host or container?

Almost everything runs **inside** the container. Getting there is two host
commands, and the second one drops you into a shell inside it:

```bash
# on the HOST
cd /home/revel/adam-rl
xhost +local:root            # only needed for the GUI, once per login
docker compose up -d         # start the container
docker compose exec adam_rl bash    # <- this puts you INSIDE
```

Your prompt then changes to `root@<machine>:/workspace#`. **That means you are
already in.** Do not run `docker compose` again from there — Docker is not
installed inside the container and you will get `bash: docker: command not
found`. Just run the tools directly. `exit` takes you back to the host.

`groups: cannot find name for group ID 992` on entry is harmless and can be
ignored.

Inside, the repo is `/workspace`, the tools are `/opt/tools`, and checkpoint
paths are relative to `/workspace` (so `logs/rsl_rl/...`).

Below, commands under a **HOST** heading run outside; everything else runs at
the `/workspace#` prompt. To run one container command from the host without
leaving a shell behind, prefix it with `docker compose run --rm adam_rl`.

The trained checkpoints referenced below:

| scenario | checkpoint |
|---|---|
| Experiment 2, 3 fingers (100 % success) | `logs/rsl_rl/adam_u30_wuji_synergy/2026-09-19_12-08-53/model_499.pt` |
| Experiment 2, 2 fingers (0 % success) | `logs/rsl_rl/adam_u30_wuji_synergy_f2/2026-09-19_12-19-30/model_499.pt` |
| Experiment 1, grasp-only (99.6 %) | `logs/rsl_rl/adam_u30_wuji_grasponly/2026-09-19_07-47-30/model_599.pt` |

---

## 1. Check it all still works (no training, ~2 min)

Run these after pulling, after changing the environment, or to convince
yourself nothing is broken. Each spawns the scene, exercises it and prints
measurements.

```bash
# Experiment 2: scene, contact plumbing, and a scripted grasp from the
# configured pose. Expect "success 16/16".
python /opt/tools/probe_synergy.py

# the same for the two-finger configuration. Expect success 0/16, lifted 11/16
ADAM_U_FINGERS=2 python /opt/tools/probe_synergy.py

# Experiment 1: spawn heights, grip sign, bolt rest height, reward terms
python /opt/tools/probe_wuji.py --scenario grasp
python /opt/tools/probe_wuji.py                 # the reach scenario

# the PND-hand robot spawns upright (the original environment)
python /opt/tools/compare_pose.py
```

**HOST** side, no container and no GPU needed (plain Python with numpy/scipy):

```bash
python3 tools/find_pinch_pose.py --middle          # re-solve the pre-grasp pose
python3 tools/hand_geometry.py                     # fingertip positions per posture
python3 tools/make_wuji_urdf.py --check            # vendored robot is current
python3 tools/make_wuji_hand_urdf.py --fingers 3 --check
python3 tools/make_wuji_hand_urdf.py --fingers 2 --check
```

---

## 2. Evaluate a trained policy

Per-episode success rates, not per-step averages. This is the number that
answers "does it grasp the bolt".

```bash
# Experiment 2, three fingers -- expect 100.0 % stable-grasp success
python /opt/tools/eval_synergy.py \
    --checkpoint_path logs/rsl_rl/adam_u30_wuji_synergy/2026-09-19_12-08-53/model_499.pt \
    --num_envs 256 --episodes 512

# Experiment 2, two fingers -- expect 0.0 %
ADAM_U_FINGERS=2 python /opt/tools/eval_synergy.py \
    --checkpoint_path logs/rsl_rl/adam_u30_wuji_synergy_f2/2026-09-19_12-19-30/model_499.pt \
    --num_envs 256 --episodes 512

# Experiment 1, grasp-only -- expect 99.4 % held
python /opt/tools/eval_grasponly.py \
    --checkpoint_path logs/rsl_rl/adam_u30_wuji_grasponly/2026-09-19_07-47-30/model_599.pt \
    --num_envs 256 --episodes 512
```

### Evaluate with the Isaac Lab window open

Add `--viz kit` and Isaac Lab launches and renders while it evaluates. You get
the same metrics at the end, and you can watch the grasps happen. Add
`--real-time --speed 0.25` to slow them down and `--start-delay` so the window
is up before the first one, since a grasp takes about a second at full speed.

```bash
python /opt/tools/eval_synergy.py --viz kit \
    --num_envs 16 --episodes 16 --real-time --speed 0.25 --start-delay 15 \
    --checkpoint_path logs/rsl_rl/adam_u30_wuji_synergy/2026-09-19_12-08-53/model_499.pt
```

This works straight from the container prompt: the `adam_rl` service already
has `DISPLAY` set and the X socket mounted. Only `xhost +local:root` on the
host is needed first, once per login.

Keep `--num_envs` small. At 256 it renders thousands of prims and is
unwatchable; 16 is a good balance and 4 lets you study a single hand. Use the
headless form above for the statistics, which wants many more episodes.

`--viz` also takes `rerun` or `viser`, which bypass the RTX renderer; `viser`
serves to a browser, so it works from another machine.

### Record videos instead

Rendered offscreen, no display needed:

```bash
python /opt/tools/eval_synergy.py \
    --checkpoint_path logs/rsl_rl/adam_u30_wuji_synergy/2026-09-19_12-08-53/model_499.pt \
    --num_envs 4 --episodes 8 --video
# -> <checkpoint dir>/videos/*.mp4
```

---

## 3. Watch a policy in the viewport

**HOST** commands — these start their own container so the display passes
through. Run them from `/home/revel/adam-rl`, not from inside a container.
Needs `xhost +local:root` first.
Use `--speed` below 1 for slow motion and `--start-delay` so the window is open
before anything happens — a grasp takes about a second at full speed.

```bash
docker compose run --rm -e DISPLAY=:1 adam_rl \
  python genesis_rl/scripts/play.py --robot wuji --scenario synergy \
    --viz kit --num_envs 4 --real-time --speed 0.25 --start-delay 25 --steps 600 \
    --checkpoint_path logs/rsl_rl/adam_u30_wuji_synergy/2026-09-19_12-08-53/model_499.pt

# Experiment 1 (the whole robot, arm and hand)
docker compose run --rm -e DISPLAY=:1 adam_rl \
  python genesis_rl/scripts/play.py --robot wuji --scenario grasp \
    --viz kit --num_envs 4 --real-time --speed 0.3 --start-delay 20 --steps 900 \
    --checkpoint_path logs/rsl_rl/adam_u30_wuji_grasponly/2026-09-19_07-47-30/model_599.pt
```

`--viz` also takes `rerun` or `viser`, which bypass the RTX renderer; `viser`
serves to a browser, so it works from another machine. Omit `--viz` for headless.

---

## 4. Train

```bash
# Experiment 2, three fingers -- ~6 min, solves by iteration ~400
python genesis_rl/scripts/train.py --robot wuji --scenario synergy \
    --num_envs 1024 --max_iterations 500

# two fingers (needs ADAM_U_FINGERS on the container, not just the shell)
ADAM_U_FINGERS=2 python genesis_rl/scripts/train.py --robot wuji --scenario synergy \
    --num_envs 1024 --max_iterations 500

# Experiment 1, grasp-only -- ~35 min
python genesis_rl/scripts/train.py --robot wuji --scenario grasp \
    --num_envs 4096 --max_iterations 600

# the reach scenario, and the bin layout variant
python genesis_rl/scripts/train.py --robot wuji --num_envs 4096
ADAM_U_WORKCELL=bin python genesis_rl/scripts/train.py --robot wuji --num_envs 4096

# the original PND-hand cube task
python genesis_rl/scripts/train.py --num_envs 4096
```

From the **HOST** with `run --rm`, pass the variables through explicitly:

```bash
docker compose run --rm -e ADAM_U_FINGERS=2 adam_rl \
  python genesis_rl/scripts/train.py --robot wuji --scenario synergy --num_envs 1024
```

Quick smoke test of any of the above, about 30 seconds:

```bash
python genesis_rl/scripts/train.py --robot wuji --scenario synergy \
    --num_envs 16 --max_iterations 3
```

Watch the curve while it runs, from the **HOST**:

```bash
tensorboard --logdir logs/rsl_rl
```

Remember the log scaling: `Episode_Reward/success` is divided by the episode
length, so multiply by **120** to read it as a reward. 0.1667 means 100 %
success. See `ALGORITHM.md` section 8.

---

## 5. Diagnostics when something is wrong

```bash
# which closure levels actually hold the bolt (scripted, no policy)
python /opt/tools/probe_synergy.py --sweep --sweep-closure --closure-range 0.3 1.0

# which hand heights and bolt offsets work
python /opt/tools/probe_synergy.py --sweep --sweep-heights 0.0 0.015 --sweep-dy -0.0075 0.0075

# fingertip paths through a close, with the bolt moved out of the way
python /opt/tools/probe_synergy.py --trace-tips

# try a different pose without editing the config
python /opt/tools/probe_synergy.py --orient fingers_down --open 0.40 \
    --thumb-abd -0.4 --wrist-height 0.12 --bolt-rel 0.027 0.047

# Experiment 1: does a scripted close lift the bolt from the configured pose?
python /opt/tools/probe_wuji.py --scenario grasp
```

---

## 6. Rebuild the robot assets

**HOST** commands. Only needed if the upstream model changes, or if you change
a solved pose. Two steps, two different containers.

```bash
# 1. vendor the URDF and meshes from adam-moveit
python3 tools/make_wuji_urdf.py                    # full Adam-U + Wuji hands
python3 tools/make_wuji_hand_urdf.py --fingers 3   # the right hand alone

# 2. Isaac Lab 2.3.2 (adam-wbc's image): URDF -> flat USD.
#    Isaac Lab 3.0's own importer nests rigid bodies and breaks the
#    environment's prim paths, so the conversion must happen on 2.3.2.
docker run --rm --gpus all \
    -v $PWD:/workspace -v $PWD/tools:/opt/tools:ro \
    -v adam-wbc_isaac-cache-kit:/opt/venv/lib/python3.11/site-packages/isaacsim/kit/cache \
    -v adam-wbc_isaac-cache-ov:/root/.cache/ov -v adam-wbc_isaac-cache-nvidia:/root/.nv \
    -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    --entrypoint python adam-lite-wbc:isaaclab-2.3.2 /opt/tools/export_usd_wuji.py --force \
    --urdf /workspace/assets/robots/wuji_hand_right/urdf/wuji_hand_right_f3.urdf \
    --out-dir /workspace/assets/robots/wuji_hand_right/usd_f3 --name wuji_hand_right_f3 \
    --expect-bodies 22 --expect-joints 21 \
    --needed r_wrist r_thumb_distal r_index_finger_distal r_middle_finger_distal

# the files come back owned by root
docker run --rm -v $PWD:/workspace --entrypoint bash \
    adam-u-rl:isaaclab-3.0 -c "chown -R $(id -u):$(id -g) /workspace/assets"
```

Changing the finger count changes the solved pose, which changes the lift axis
baked into the URDF, so both files must be rebuilt together. The environment
refuses to start if they disagree.

---

## 6b. Experiment 4: safe reach-to-grasp (`--scenario reach_grasp`)

Design: [`EXPERIMENT4_DESIGN.md`](EXPERIMENT4_DESIGN.md). The level is
`ADAM_U_RG_LEVEL` (0 gate, 1-4 standoff and pose error, 5 physics, 6-7
perception, 8-10 clutter, 11-13 bolt pose), and it MUST be passed with `docker compose exec -e`.

```bash
# HOST: arm poses and reset Jacobians for the standoff buckets (paste block)
python3 tools/solve_reach_grasp_poses.py --heights 0 0.05 0.10 --verify

# regression tests (43 checks at level 0; perception checks need level 6 or 7)
docker compose exec -e ADAM_U_RG_LEVEL=0 adam_rl python /opt/tools/test_reach_grasp.py
docker compose exec -e ADAM_U_RG_LEVEL=6 adam_rl python /opt/tools/test_reach_grasp.py --only 8

# geometry gate (level 0) / scripted baseline (any level) / reward audit
docker compose exec -e ADAM_U_RG_LEVEL=0 adam_rl python /opt/tools/audit_reach_grasp.py --episodes 128
docker compose exec -e ADAM_U_RG_LEVEL=3 adam_rl python /opt/tools/audit_reach_grasp.py

# train a level; resume the next from the previous level's BEST checkpoint
docker compose exec -e ADAM_U_RG_LEVEL=1 adam_rl python genesis_rl/scripts/train.py \
    --robot wuji --scenario reach_grasp --num_envs 2048 --max_iterations 400
docker compose exec -e ADAM_U_RG_LEVEL=2 adam_rl python genesis_rl/scripts/train.py \
    --robot wuji --scenario reach_grasp --num_envs 2048 --max_iterations 400 \
    --resume --checkpoint_path logs/rsl_rl/adam_u30_wuji_reach_grasp_L1/<run>/model_<best>.pt --reset-std 0.3

# evaluate against the advancement gate (512 episodes, mean action)
docker compose exec -e ADAM_U_RG_LEVEL=1 adam_rl python /opt/tools/eval_reach_grasp.py \
    --checkpoint_path logs/rsl_rl/adam_u30_wuji_reach_grasp_L1/<run>/model_<best>.pt --num_envs 256 --episodes 512

# level 14+: the waist buckets (once, on the host; ~5 min on 48 cores)
python3 tools/solve_reach_grasp_poses.py --grid --free-waist --min-slack 0.15
# level 21+: the same grid with the wrist rolled to the three block tilts (~20 min)
python3 tools/solve_reach_grasp_poses.py --grid --free-waist --min-slack 0.15 --tilts-deg 0 18.4 29.5 39.8

# level 15: graspable-area sweep along the rod's axis (17 offsets x 512 episodes, ~25 min)
tools/sweep_grasp_offset.sh 15 logs/rsl_rl/adam_u30_wuji_reach_grasp_L15/<run>/model_<best>.pt 512
# level 16: offset x pinch-angle map (PINCH pins the angle; OUT=<dir> resumes a sweep)
for a in -45 -30 -15 0 15 30 45; do PINCH=$a tools/sweep_grasp_offset.sh 16 <checkpoint> 512 -12 -8 -4 0 4 8 12; done
# level 18: the standing rod's height map (UPRIGHT=1), and the lying map again on the new collider (UPRIGHT=0)
UPRIGHT=1 tools/sweep_grasp_offset.sh 18 <checkpoint> 512 -14 -10 -6 -2 2 6 10 14
# one offset only
docker compose exec -e ADAM_U_RG_LEVEL=15 adam_rl python /opt/tools/eval_reach_grasp.py \
    --checkpoint_path logs/rsl_rl/adam_u30_wuji_reach_grasp_L15/<run>/model_<best>.pt --grasp-offset-mm 12

# watch it
docker compose exec -e ADAM_U_RG_LEVEL=1 adam_rl python genesis_rl/scripts/play.py --robot wuji \
    --scenario reach_grasp --viz kit --num_envs 4 --real-time --speed 0.25 --start-delay 15 \
    --checkpoint_path logs/rsl_rl/adam_u30_wuji_reach_grasp_L1/<run>/model_<best>.pt
```

TensorBoard for these runs carries `Safety/*`, `Clearance/*`, `Wrist/*`,
`Phase/*` and `Outcome/*` alongside `Episode_Reward/*`; the env publishes
them after Isaac Lab's reset, which is why the pipeline's never appeared.

---

## 7. Environment variables

| variable | values | effect |
|---|---|---|
| `ADAM_U_FINGERS` | `2`, `3` (default), `4` | how many fingers oppose the thumb in Experiment 2 |
| `ADAM_U_LEVEL` | `0`..`3` | displacement level of the staged pipeline |
| `ADAM_U_RG_LEVEL` | `0`..`35` | curriculum level of Experiment 4 (`--task wuji_reach_grasp`) |
| `ADAM_U_WORKCELL` | `single` (default), `bin` | bare table, or adam-moveit's bin, for Experiments 1 and the reach scenario |
| `WORKCELL_SIDE` | `right` (default), `left` | mirror the workcell for the training arm |
| `ADAM_U_ASSETS` | path | override the asset root |
| `ADAM_U_WUJI_USD` | path | override the full-robot USD |

Pass them with `-e NAME=value` on `docker compose run`, or export them inside
an `exec` shell.
