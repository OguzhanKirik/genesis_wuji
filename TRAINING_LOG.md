# Training log

How the algorithm itself works — observations, actions, reward, success test,
reading these logs — is in [`docs/ALGORITHM.md`](docs/ALGORITHM.md).

One entry per training run worth remembering: what was trained, on which env
config, with which command, and what came out. Smoke runs (a handful of
iterations to see that the code runs) are listed once per config, not per run.
Checkpoints live under `adam_u_isaac_lab/logs/rsl_rl/<experiment>/<timestamp>/`
(gitignored); TensorBoard: `tensorboard --logdir adam_u_isaac_lab/logs/rsl_rl`.

Conventions: "mean reward" and the per-term values are rsl-rl's console
numbers, i.e. mean per-step reward over the last iteration. The lift bonus
has weight 5 and is a 0/1 step, so `lift_bonus / 5` is the fraction of
timesteps with the bolt more than 2 cm off the table.

---

## Experiment 1 — Wuji Hand 2, grasp-only, full joint control

**Date** 2026-09-19 · **Run** `adam_u30_wuji_grasponly/2026-09-19_07-47-30` ·
**Status** complete, 600 iterations, ~35 min

| | |
|---|---|
| env | `adam_u_rl/envs/adam_u_wuji_grasponly_env_cfg.py` (`AdamUWujiGraspOnlyEnvCfg`) |
| robot | Adam-U + Wuji Hand 2, `assets/robots/adam_u_wuji/usd/adam_u_wuji.usd` (60 bodies, 59 DoF) |
| scene | adam-moveit table (top 0.95 m), one bolt `rod-with-threads-0001` (30 g) at (0.39, −0.21) ± 1.5 cm, random yaw, no bin |
| start | torso leaned 0.35 rad, right hand palm-down over the bolt, fingers pre-curled ~30 %, thumb cmc_abd −0.95 (`PREGRASP_ARM`, `PREGRASP_FINGERS`) |
| actions | 27: 7 arm joints ±0.2 rad around the start pose, 20 right-hand joints offset from the pre-curl (+1 = fully flexed) |
| observations | 101: arm q/qd (7+7), wrist pos/quat (3+4), hand q/qd (20+20), bolt pos/quat (3+4), bolt−wrist (3), last action (27), base pos (3) |
| reward | alive 0.5, terminating −2, reach_fine 2 (thumb+index tip to bolt, std 4 cm), grasp 3 (proximity-gated flexion closure), lift_bonus 5 (bolt > 2 cm above rest), object_height 15 (ramp, cap 8 cm), off_surface −2, wrist_drift −5, action_rate −5e-4, joint_vel −1e-5 |
| episode | 5 s, 30 Hz control, 120 Hz physics; ends on time-out or bolt 7.5 cm below the table |
| algorithm | PPO, rsl-rl 5.0.1, `rsl_rl_ppo_cfg.py`: actor/critic MLP 512-256-128 elu, obs normalisation, lr 1e-3 adaptive (KL 0.01), γ 0.99, λ 0.95, clip 0.2, entropy 0.005, 32 steps/env, 8 epochs, 4 minibatches |
| command | `python adam_u_rl/scripts/train.py --robot wuji --scenario grasp --num_envs 4096 --max_iterations 600` |
| hardware | RTX PRO 6000 Blackwell; 2.5–3.7 s / iteration at 4096 envs, ~35 min |

**Pre-flight** (`tools/probe_wuji.py --scenario grasp`): spawn heights as
predicted by the URDF kinematics; finger action +1 closes; bolt rests 11.0 mm
above the table; a scripted close-and-lift from the start pose lifted the bolt
in 5 of 16 placements on a ±2 cm grid.

**Learning curve** (mean per-step reward terms):

| iteration | mean reward | reach_fine | grasp | lift_bonus | object_height | ≈ % steps lifted |
|---|---|---|---|---|---|---|
| 0 | 0.67 | 0.015 | 0.005 | 0.014 | 0.006 | 0 |
| 100 | 2.83 | 0.089 | 0.028 | 0.041 | 0.018 | 1 |
| 200 | 4.30 | 0.237 | 0.126 | 0.102 | 0.037 | 2 |
| 300 | 7.53 | 0.502 | 0.297 | 0.380 | 0.092 | 8 |
| 400 | 27.79 | 0.797 | 0.501 | 3.284 | 0.440 | 66 |
| 500 | 35.44 | 0.932 | 0.707 | 4.311 | 0.752 | 86 |
| 550 | 36.87 | 0.967 | 0.779 | 4.374 | 0.839 | 87 |
| 599 | 36.50 | 0.989 | 0.811 | 4.345 | 0.908 | 87 |

**Result** The policy grasps and lifts. At iteration 599 the bolt is more
than 2 cm off the table in 87 % of all timesteps (lift_bonus 4.35 / 5) and the
mean lift height is about 6 cm (object_height 0.91 / 15, cap 8 cm). Episodes
run to the 5 s time-out (mean length 147 of 150 steps); the off-table penalty
is essentially zero, so the bolt is not being knocked off. Fingertip proximity
(reach_fine 0.99 of 1) and gated closure (grasp 0.81 of 1) are near saturation.
Plateau from iteration ~450. No NaN, no instability at 120 Hz physics.

**Per-episode evaluation** (`tools/eval_grasponly.py`, 512 episodes, 256
envs, policy mean action, same bolt jitter and yaw randomisation as training):

| criterion | rate |
|---|---|
| lifted (bolt > 2 cm above rest at any point) | 100.0 % |
| held (still > 2 cm up when the 5 s episode ends) | 99.4 % |
| held for the whole last second | 95.5 % |
| lifted then dropped | 0.6 % |
| knocked off the table | 0.2 % |
| mean peak lift height | 8.4 cm |
| mean final flexion closure (q / upper) | 0.59 |

So yes: the policy grasps the bolt. The 4 % gap between "held at the end"
and "held for the whole last second" is the bolt shifting in the grip and
dipping below +2 cm briefly near the end, not drops. Not tested: holds longer
than the 5 s episode, and a scripted lift beyond the arm's own range.

Playback:

    python adam_u_rl/scripts/play.py --robot wuji --scenario grasp --viz kit --num_envs 4 --real-time \
        --checkpoint_path logs/rsl_rl/adam_u30_wuji_grasponly/2026-09-19_07-47-30/model_599.pt

**Notes**

- The breakthrough came between iterations 300 and 400: the fraction of
  lifted timesteps went from 8 % to 66 % and the mean lift height from under
  1 cm to about 3 cm (object_height / 15). By 500 it was 86 % and 5 cm.
- **Watched in the viewport (slow motion): the grasp looks odd.** The bolt
  starts between the pre-curled fingers, the fingers curl under it and pin it
  against the thumb and palm, and the arm lifts. It is a hold, not a
  recognisable pinch, and there is no approach to speak of because the
  scenario spawns the bolt inside the hand's reach with 1.5 cm of jitter.
  That is the motivation for Experiment 2: a fixed wrist at a checked
  pre-grasp distance, contact-based rewards and a held-after-lift success.
- The wrist-drift penalty settled at about −0.035 per step, i.e. the wrist
  ends up ~8 cm from its start in the horizontal plane (sqrt(0.035/5)); the
  policy uses most of its ±0.2 rad arm range.
- Reward weights were carried from the reach scenario and not tuned. The
  height cap (8 cm) is what the ±0.2 rad arm range can lift.
- Fixes that had to land before this run could work are in the README
  ("The grasp-only scenario"): URDF effort limits on the arm drives, the
  waist lean, and `hold_default_joint_targets` for unactuated joints.

---

## Smoke runs (code-path checks only, not training)

| date | env | command | result |
|---|---|---|---|
| 2026-09-19 | reach scenario, single layout | `train.py --robot wuji --num_envs 16 --max_iterations 3` | runs, checkpoints written |
| 2026-09-19 | reach scenario, bin layout | `ADAM_U_WORKCELL=bin … --robot wuji --num_envs 16 --max_iterations 3` | runs |
| 2026-09-19 | reach scenario, 1024 envs | `--robot wuji --num_envs 1024 --max_iterations 6` | 1.18 s / iteration, no NaN |
| 2026-09-19 | reach scenario, viewport | `--robot wuji --viz kit --num_envs 16 --max_iterations 20` | renders and trains, exit 0 |
| 2026-09-19 | grasp-only, 64 envs | `--robot wuji --scenario grasp --num_envs 64 --max_iterations 5` | runs |

## Experiment 2 — Wuji hand only, fixed wrist, two finger synergies

**Date** 2026-09-19 · **Runs** `adam_u30_wuji_synergy/2026-09-19_09-06-47` (v1, four-finger rake)
and a v2 run on the redesigned pinch · **Status** v1 complete; v2 training

### v2 — the spec implemented literally (thumb-opposed pinch)

v1 met the spec's shape but deviated in five places. v2 closes all of them. The
substantive one was the finger count: v1 drove all four fingers because with
index+middle only, a fixed-wrist close dragged the bolt across the table. That
turned out to be a consequence of the *pose*, not of the hand — the pose had
been tuned for a rake — and re-solving it gives a real pinch.

| gap in v1 | v2 |
|---|---|
| only 2 merged contact flags | per-fingertip force (thumb/index/middle, scaled by `F_MAX`) **and** 3 separate contact flags; obs 22 → 26 |
| contact paid on its first step | paid only after `STABLE_STEPS` (4) consecutive steps, so a spike earns nothing |
| contact bonuses +1/+1/+2 | the spec's **+0.5/+0.5/+1.0** |
| no verification-phase table test | success also needs the bolt clear of the table for the whole hold phase |
| all four fingers in the synergy | **3 fingers** (thumb vs index+middle), ring and little held flat. `ADAM_U_FINGERS` selects 2, 3 or 4 |

**The pose, solved rather than guessed** (`tools/find_pinch_pose.py`). The tool
walks the synergy stroke with URDF forward kinematics and measures where the
thumb and opposing fingertips actually converge. The key realisation is that
the gap, the opposition angle and the fingertip-to-bolt distance are all
independent of how the hand is *rotated*; only table clearance is not. So the
tool places the wrist so first contact with the 2.5 cm bolt happens at the
bolt, then searches the sphere of "which way is up in the hand frame" for a
direction that keeps every fingertip above the table over the whole stroke.

That search is what made a pinch possible. Fingers-down and palm-down both fail
from a fixed wrist: the fingers hang ~17 cm below the wrist when open and curl
up to ~9 cm as they close, so any wrist height putting the closing pinch at the
bolt buries the open fingertips up to 10 cm *under* the table. Experiment 1 only
worked because its arm rose while the hand closed. The winning pose is tilted,
coming at the bolt from the side. Measured for it:

| | |
|---|---|
| thumb vs index+middle gap | 5.2 cm open → 1.2 cm closed |
| contact with the bolt | at 38 % of the stroke, then 13 mm of squeeze |
| opposition angle | 97° (the tips travel into each other) |
| fingertip to bolt at reset | 2.9 cm to its centre ≈ 1.6 cm to its surface (spec: 1–2 cm) |
| lowest fingertip clearance | 1.1 cm above the table, over the whole stroke |

**Two defects this uncovered, both invisible in the scene:**

- **The scripted lift travelled sideways.** The prismatic lift joint's axis is
  expressed in the hand's own frame, and a joint axis rotates with its parent,
  so once the hand was rotated into the pinch pose a nominal `(0, 0, 1)` lift
  went off-vertical: the wrist moved sideways and *dragged the bolt 7 cm across
  the table* while reporting a 10 cm lift. The axis is now baked into the URDF
  as `HAND_ROT⁻¹ · world_up` (`make_wuji_hand_urdf.py --lift-axis`), and the env
  asserts the product is world +z at startup.
- **The held fingers were on the table.** Ring and little were curled 0.30
  while the pose was validated with them flat; curling brings their tips *down*
  in this tilted pose. Every episode logged a finger-table collision until they
  were flattened.

**Scripted baseline** (`tools/probe_synergy.py --closure 1.0 1.0`, full close,
no policy): **16/16 stable-grasp successes** under the spec's rules — lifted,
held through verification, no table contact, no excessive force. Stable thumb
contact on 65 % of grasp-phase steps, opposing fingers 68 %. So the task is
well posed before PPO sees it, which is the point of a feasibility experiment.

**Runs** (1024 envs, 500 iterations, ~6 min each, PPO settings exactly as the spec):

| | 3 fingers (thumb vs index+middle) | 2 fingers (thumb vs index) |
|---|---|---|
| run | `adam_u30_wuji_synergy/2026-09-19_12-08-53` | `adam_u30_wuji_synergy_f2/2026-09-19_12-19-30` |
| observations | 26 | 20 |
| scripted baseline (full close) | **16/16** stable grasps | 0/16 (lifts 11, drops all 11) |
| scripted closure cells that succeed | 4 of 16 (the whole high-closure corner) | **1 of 16** (thumb 0.77, fingers 0.77) |
| success at iteration 100 / 200 / 300 / 400 / 500 | 0 % / 16 % / 35 % / 94 % / 99 % | 0 % throughout |
| **evaluation, 512 episodes** | **100.0 %** stable grasps | **0.0 %** |
| drop / table-collision / excess-force | 0 % / 0 % / 0 % | 0 % / 0 % / 0 % |
| bolt touched the table during verification | 0 % | 100 % (never lifted) |
| thumb / opposing contact, grasp phase | 90 % / 95 % of steps | 8 % / 10 % |
| final synergy (thumb, fingers) | (1.00, 1.00) | (0.00, 0.00) |

Videos of the 3-finger policy: `logs/rsl_rl/adam_u30_wuji_synergy/2026-09-19_12-08-53/videos/`.

**Result: three fingers, yes; two fingers, no — and the reason is exploration,
not physics.** The scripted sweep proves a two-finger grasp exists, but at
exactly one closure combination out of sixteen, against four for three fingers.
PPO never landed in it.

The failure mode is worth recording because the reward looks fine until you
read the two numbers together. Throughout the two-finger run the policy
collected the full +2.0 of contact bonuses in *every* episode while its
evaluated mean action sat at closure (0.00, 0.00) — fully open, never
deliberately touching the bolt. The bonuses were being earned by the
exploration noise: over 60 grasp-phase steps, a policy whose mean is "open"
still brushes the bolt often enough to trip a 4-step stable contact. A reward
that a random policy collects in full carries no gradient, so the only term
left with any slope was the −0.01 action-rate penalty, which rewards holding
*any* constant action. With +20 success unreachable by chance, the mean drifted
to a constant and stayed.

Three ways out, none tried yet: shape the grasp with a dense term (grip force,
or bolt height during the lift) so there is a slope toward closing; widen the
basin with a pose that has more squeeze margin; or start from the 3-finger
policy as a curriculum. Raising the entropy coefficient alone would not help —
the problem is that the *mean* has no gradient, not that exploration is too
narrow.

### v1 — four-finger rake (superseded)

Kept for the record: 500 iterations at 1024 envs, ~7 min, solved by iteration
20, **99.6 % stable grasps** over 512 evaluation episodes with no drops, no
table collisions and no force violations, at a partial closure of (0.59, 0.53).
Learning curve, settings and the v1 pre-flight sweeps are in this file's git
history. Videos: `logs/rsl_rl/adam_u30_wuji_synergy/2026-09-19_09-06-47/videos/`.

## Planned

- Experiment 2 v2 evaluation, and a two-finger (`ADAM_U_FINGERS=2`) comparison.
- Experiment 2 with bolt/hand pose randomisation (±5 mm, ±1 cm, ±2 cm) to
  measure the basin the learned closure covers.

---

# Staged pipeline — whole robot, IK wrist, displacement curriculum

Design: [`docs/GRASP_PIPELINE_DESIGN.md`](docs/GRASP_PIPELINE_DESIGN.md).
Environment: `adam_u_rl/envs/wuji_pipeline_env_cfg.py`.

One environment for every rung. **58 observations and 8 actions are fixed
throughout**, so a checkpoint carries from one rung to the next with no
surgery. The robot is the whole Adam-U; the policy commands six Cartesian
wrist residuals through damped least-squares IK plus two finger synergies,
and never commands an arm joint directly.

Episode: 90 grasp steps (3 s) + 30 scripted lift + 30 hold, 150 steps at 30 Hz.

## Level by level

### What every level shares

**Goal.** Pinch the 30 g bolt (`rod-with-threads-0001`) between the thumb and
index/middle finger of the right Wuji hand, and keep it at least 5 cm off the
table through a scripted lift and hold, without jamming fingers into the table
or crushing the bolt.

**Setup.** The whole Adam-U on a fixed base. The bolt is always in the same
place; what changes between levels is how far the **wrist** starts from the
solved pinch pose (`tools/solve_pipeline_pregrasp.py`, yawed 90° so
`wristRoll_Right` is off its stop). Displacement is applied at reset through
the baked Jacobian `J_SHARP`.

**Observations (58).** Bolt position (3) and orientation (6) in the wrist
frame, wrist linear and angular velocity (3 + 3), the 9 driven finger joint
positions and velocities (9 + 9), last action (8), fingertip forces (3),
contact flags (3), episode time (1), normalised arm joint positions (7),
fingertip-to-bolt distances (3).

**Actions (8).** Six Cartesian wrist residuals (dx, dy, dz, droll, dpitch,
dyaw), smoothed and rate-limited, turned into arm joint targets by damped
least-squares IK; two finger synergies (thumb, fingers), mapped from [−1, 1]
to closure 0–1 between the open and closed hand poses. The policy never
commands an arm joint directly.

**Episode.**

1. **GRASP**, up to 90 steps: the policy acts.
2. **LIFT**, 30 steps: scripted 10 cm world-vertical rise, finger closure frozen.
3. **HOLD**, 30 steps: scripted, checks the bolt stays up.

**Reward.** The same terms at every level; the full table is in the Reward
section below.

- **The goal:** `success` +20, paid on the last step.
- **Shaping on the way:** the three one-off contact bonuses (+0.5, +0.5, +1)
  and `approach`, which pays progress in the fingertip-to-bolt potential Φ.
- **Disqualifiers:** drop −5, finger-table hit −3, over-force −3,
  structural contact −5 (which also ends the episode).
- **Regularisers:** `joint_limits` −0.05 and `action_rate` −0.001.

**Success test.** Checked on the last step: the bolt was ≥ 5 cm above rest for
the whole hold, never touched the table after the lift, no fingertip hit the
table above 15 N, and no fingertip exceeded 40 N.

**Algorithm.** PPO (rsl-rl 5.0.1), actor and critic MLP `[256, 256, 128]` elu,
observation normalisation, lr 3e-4, γ 0.99, λ 0.95, clip 0.2, 24 steps/env,
2048 envs (~49k samples/iteration), init std 0.3
(`RslRlWujiPipelinePPORunnerCfg`). Training command, per level:

    docker compose exec -e ADAM_U_LEVEL=<n> adam_rl python adam_u_rl/scripts/train.py \
        --robot wuji --scenario pipeline --num_envs 2048 \
        [--resume --checkpoint_path <previous level's best> --reset-std <std>]

The exact per-run commands were not recorded. This is the form `train.py`
takes. Runs land in `logs/rsl_rl/adam_u30_wuji_pipeline_L<n>/`.

**The baseline to beat.** `tools/audit_reward.py`'s scripted policy: close
both synergies fully and never move the wrist. If the policy does not beat it,
the wrist control is not earning its place.

### Level 0 — fixed pose (the fingers-only diagnostic)

- **Goal.** Prove that the hand, contact sensing, success test and scripted
  lift work end to end on the whole robot, with nothing for the wrist to fix.
- **Starting pose.** Exactly the solved pinch pose: the fingertips start
  about 1.6 cm from the bolt surface.
- **What was trained.** Run `adam_u30_wuji_pipeline_s0/2026-09-21_17-05-57`,
  iterations 0–999, from scratch. This predates the displacement curriculum:
  it was trained as **stage 0 of the earlier action-mask curriculum**. Only
  the two finger synergies were live; the six wrist channels were zeroed.
- **Reward.** Experiment 2's sparse terms: contact bonuses plus +20 success.
  `approach` did not exist yet, and `action_rate` was still −0.01.
- **Result.**
  - Scripted close: 100 %. Trained policy: 100 % (512 episodes).
  - At level 0 closing the hand is enough; the policy only had to learn not
    to break that.
- **What went wrong next.** Resuming this checkpoint into the mask curriculum's
  stage 2 (wrist x/y/z unmasked, bolt jittered ±1 cm) gave **0 % over 400
  iterations**. It had never moved a wrist, and its inherited std of 0.98
  walked the hand off the bolt. It then learned to hold still to minimise
  `action_rate`. This failure is why the curriculum was rebuilt around
  displacement, with all 8 actions live from the start, and why `approach`
  was added.
- **Status today.** The checkpoint no longer loads: it has 55 observations
  and the env now has 58.

### Level 1 — small displacement (±0.5 cm, ±2.5°)

- **Goal.** Learn that the wrist can and should correct a small pose error
  before closing.
- **Starting displacement.**
  - Position: x, y ±0.5 cm; z −0.3 to +0.5 cm.
  - Rotation: yaw ±2.5°, roll and pitch ±2°.
- **What was trained.** Run `adam_u30_wuji_pipeline_L1/2026-09-22_07-17-37`,
  iterations 0–399, **from scratch** (not from level 0). All 8 actions live.
  Best checkpoint `model_399.pt`.
- **Reward.** The current table, including `approach` (+2 × ΔΦ) and
  `action_rate` −0.001.
- **Result.**
  - Scripted close: 72.7 %. Trained policy: **99.4 %** (+27 points).
  - Even this small an error already breaks the blind close in a quarter of
    episodes; the policy recovers nearly all of them.
- **Open issue.** One unexplained 844 mm wrist-tracking outlier. It did not
  recur at levels 2 or 3.

### Level 2 — medium displacement (±1 cm, ±5°)

- **Goal.** Double the error; test that the curriculum transfers smoothly.
- **Starting displacement.**
  - Position: x, y ±1 cm; z −0.5 to +1 cm.
  - Rotation: yaw ±5°, roll and pitch ±3°.
- **What was trained.** Run `adam_u30_wuji_pipeline_L2/2026-09-22_09-51-48`,
  resumed from level 1's `model_399`, iterations 400–798. Best `model_650.pt`.
  Success started at **81 % within ten iterations** of the resume, against
  0 % for the mask curriculum. That transfer is the evidence that the
  displacement axis works.
- **Reward.** Unchanged from level 1.
- **Result.**
  - Scripted close: 39.1 %. Trained policy: **99.8 %** (+61 points).
  - Late-run decline: best 99.7 % at iteration 669, final 97.1 %.
- **Defects found on this policy.**
  - **Lift drag.** The scripted lift started from the commanded target, which
    was 12 mm away from the actual wrist. It dragged the bolt 19.9 mm across
    the table. Fixed by re-seeding the lift from the measured wrist pose:
    drag 1.2 mm, success unchanged.
  - **Exploration noise ratchet.** With `entropy_coef = 0.01`, the action std
    grew 0.98 → 1.15 → 2.25 over levels 0–2. In the last 148 iterations,
    success fell 99.8 → 96.1 % and finger-table collisions rose 0.2 → 3.9 %.
    Fixed by entropy 0.003 and `--reset-std` on resume.

### Level 3 — large displacement (±2 cm, ±10°)

- **Goal.** The widest error the table clearance allows
  (`tools/probe_wrist_displacement.py`).
- **Starting displacement.**
  - Position: x, y ±2 cm; z −0.5 to +2 cm.
  - Rotation: yaw ±10°, roll and pitch ±5°.
- **What was trained.** Run `adam_u30_wuji_pipeline_L3/2026-09-22_10-46-13`,
  resumed from level 2's `model_650`, iterations 650–1049. First run with
  entropy 0.003 and the std reset. Best `model_850.pt`.
- **Training health vs level 2.**
  - Final std: 0.77 vs 2.25.
  - Clipped IK targets: 10,096 vs 24,949.
  - Mean minimum arm joint slack: 0.324 vs 0.216.
- **Reward.** Unchanged.
- **Result.**
  - Scripted close: 14.1 %. Trained policy: **96.5 %** (+82 points).
  - Late-run decline: best 98.4 % at iteration 859 in training, final 93.9 %.
- **What limits it.** Every remaining failure is a finger-table collision;
  drops and over-force are 0 %. At −5° roll the lowest fingertip has 0.31 cm
  of clearance during the closure, against 1.08 cm undisplaced. This is
  geometry, not learning: narrow roll and pitch, or re-solve the pre-grasp
  pose for more clearance.
- **Status today.**
  - 76.4 % on the current code, because of the early-grasp latch added after
    this run.
  - 0 % when started at zero displacement; the cause is not yet found.
  - See the re-evaluation section below.

## Results

| level | wrist displacement at reset | scripted baseline | trained policy | gap |
|---|---|---|---|---|
| 0 | none | 100 % | 100 % | — |
| 1 | ±0.5 cm, ±2.5° | 72.7 % | 99.4 % | +27 |
| 2 | ±1 cm, ±5° | 39.1 % | **99.8 %** | +61 |
| 3 | ±2 cm, ±10° | 14.1 % | **96.5 %** | +82 |

512 evaluation episodes each, deterministic (mean) action,
`tools/eval_pipeline.py`. The **scripted baseline** is a full finger close with
no wrist correction at all — the number the policy has to beat for wrist
control to be earning its place. It is produced by `tools/audit_reward.py`,
which also gates training.

Checkpoints (best, NOT final — see "late-run degradation"):

| level | run | checkpoint |
|---|---|---|
| 0 | `adam_u30_wuji_pipeline_s0/2026-09-21_17-05-57` | `model_999.pt` |
| 1 | `adam_u30_wuji_pipeline_L1/2026-09-22_07-17-37` | `model_399.pt` |
| 2 | `adam_u30_wuji_pipeline_L2/2026-09-22_09-51-48` | `model_650.pt` |
| 3 | `adam_u30_wuji_pipeline_L3/2026-09-22_10-46-13` | `model_850.pt` |

Level 1 trained from scratch; 2 and 3 resumed from the previous level's best.

## Re-evaluation on the current code (2026-09-23) — the table above is stale

The numbers above were measured on the code as it stood when each checkpoint
was trained. `wuji_pipeline_env_cfg.py` was edited afterwards (last change
2026-09-22 17:26), most importantly adding the early-grasp latch
(`grasp_ready` / `update_latch`: once contact is stable and gripping, freeze
the wrist, settle 10 steps, lift early). None of these checkpoints trained
with it. Re-run with `tools/eval_pipeline.py`, 256 episodes, mean action:

| checkpoint | eval level | logged | now |
|---|---|---|---|
| level 3 `model_850` | 3 | 96.5 % | **76.4 %** |
| level 3 `model_850` | 2 | — | 60.0 % |
| level 3 `model_850` | 1 | — | 58.0 % |
| level 3 `model_850` | 0 | — | **0.0 %** — fingers never close, synergies (0.00, 0.00) |
| level 2 `model_650` | 2 | 99.8 % | 94.5 % |
| level 0 `s0/model_999` | 0 | 100 % | does not load: 55 observations vs the env's 58 |

- The level-3 policy **fails when the wrist starts close to the nominal pose**.
  It is not the `is_zero` shortcut in `displace_wrist`: a level-0 override
  with a 1 µm displacement also gives 0 %. Root cause not yet found.
- Two later runs are not documented above: `pipeline_L0/2026-09-22_15-00-23`
  (resumed at 650, ran to 1049) and `pipeline_L3/2026-09-22_16-24-54`
  (850 → 854, looks aborted).
- **`ADAM_U_LEVEL` must be passed with `docker compose exec -e`.** Prefixing
  it to `docker compose` on the host does not reach the container, and the
  eval silently runs level 0. Check the `level ...` line it prints.

To get trustworthy numbers again: retrain the ladder on the current env, or
revert the latch change and re-evaluate.

## The curriculum axis is displacement, not action masking

The first attempt used the action mask as the curriculum: stage 0 drove only
the fingers, stage 2 also unmasked wrist x/y/z, with the bolt jittered ±1 cm.
**It returned 0 % over 400 iterations** (`adam_u30_wuji_pipeline_s2`), and the
curve says exactly why:

```
  iter    reward   succ%   thumb   oppos   act_rate    std
   999    -0.175    0.0%   0.000   0.000    -0.0234   0.98   <- resumed from stage 0
  1074   -17.213    0.0%   0.057   0.003    -3.3745   1.20
  1398    -0.451    0.0%   0.000   0.000    -0.0915   0.70
```

Contact was **already zero at the resumed checkpoint**, before any learning:
the stage-0 policy had never moved a wrist, three Cartesian channels opened at
once, and its inherited exploration std of 0.98 walked the hand off the bolt
immediately. From ~1124 the policy then *learned to hold still*, because the
action-rate penalty was the only term left with a gradient.

Two lessons, both now designed around:

- **A mask curriculum changes the task discontinuously.** Displacement
  magnitude does not: all 8 actions are live from level 0 and the task tightens
  smoothly. Measured directly — resuming level 1 → level 2 started at **81 %**
  within ten iterations, where stage 0 → stage 2 started at 0 % and never
  recovered.
- **Experiment 2's reward only works with a fixed wrist.** It is entirely
  sparse (contact bonuses plus terminal success). That was fine when the
  fingertips were guaranteed to start 1.6 cm from the bolt, but with the wrist
  free nothing pulls the hand toward the object, and random exploration
  essentially never trips a stable contact.

## The wrist moves, not the bolt

The bolt is fixed at every level. The policy observes the bolt's pose
*relative to the wrist*, so a displaced wrist and a displaced bolt are the same
signal to the network — but a fixed bolt keeps the pinch geometry, the table
clearance and the solved pose validated where they were solved.

Displacement ranges are **measured**, by `tools/probe_wrist_displacement.py`,
which sweeps them against the lowest fingertip's clearance above the table
**across the whole closure stroke**. Checking only the reset posture reports
1.35 cm; the stroke reports 1.08 cm, and that difference flipped −1 cm in z
from safe to failing. The envelope is strongly asymmetric:

| axis | safe to | what stops it |
|---|---|---|
| x, y | ±3 cm+ | nothing; clearance unaffected |
| z | −0.5 / +3 cm | at −2 cm the thumb is 0.92 cm **through** the table |
| yaw | ±10° | arm joint slack (0.59 → 0.52), not geometry |
| roll | −5 / +20° | at −10° the thumb is 0.47 cm under |
| pitch | −15 / +5° | at +10° the middle finger has 0.02 cm |

Displacement is applied at reset by one multiply against a baked first-order
Jacobian (`J_SHARP`) — no IK at reset. Verified against full FK: the worst
level-3 corner lands 1.42 mm and 0.52° from the commanded pose, which does not
matter because the policy observes the *actual* wrist.

The pre-grasp pose itself had to be re-solved against the arm
(`tools/solve_pipeline_pregrasp.py`). Experiment 2's pinch was found with no
arm attached, and it pins `wristRoll_Right` on its −0.960 rad stop everywhere
in the reachable region — the arm can hold that orientation with zero authority
to leave it, in one of the DoF the experiment exists to learn. Yawing the whole
pinch 90° about world z costs nothing (every pinch metric and the table
clearance are invariant to it) and returns 0.80 of normalised wrist-roll slack.

## Reward

Every term, as in `RewardsCfg` in `wuji_pipeline_env_cfg.py`. "Once" means
paid on the first step the condition holds and never again in that episode;
contact counts as stable after 4 consecutive steps above 0.1 N.

| term | weight | paid when |
|---|---|---|
| `thumb_contact` | +0.5 | once: thumb stably on the bolt |
| `finger_contact` | +0.5 | once: index or middle stably on the bolt |
| `opposing_contact` | +1.0 | once: thumb and a finger stably on it together |
| `success` | **+20** | last step: bolt ≥ 5 cm up for the whole hold, never on the table after the lift, no finger-table hit, no tip force > 40 N |
| `approach` | +2 | every GRASP step: 2 (Φₜ − Φₜ₋₁), see below |
| `drop` | −5 | once: bolt was lifted, then fell below 2 cm |
| `table_contact` | −3 | once: a fingertip hits the table > 15 N |
| `excess_force` | −3 | once: a fingertip exceeds 40 N |
| `structural_contact` | −5 | once: palm, forearm or upper arm touches the table (also terminates) |
| `joint_limits` | −0.05 | quadratic, zero until an arm joint is within `slack_warn` of a stop |
| `action_rate` | −0.001 | squared change in action between steps |

Early terminations: bolt off the table, arm joint past its hard limit, IK
failure, structural contact. The +20 dominates; the contact bonuses and
`approach` shape the way to it, the penalties disqualify grasps that only
work by pushing into the table or crushing the bolt.

### How the reward is computed

Every control step (30 Hz), Isaac Lab's reward manager evaluates each term,
multiplies it by its weight **and by the step length dt = 1/30 s**, and sums:

    r_t = Σ_k  w_k · f_k(s_t) · dt

So the +20 success bonus adds 20/30 ≈ 0.67 to the return, not 20. The dt
scales every term equally, so the *ratios* between terms, which are what
the policy optimises, are exactly the weights in the table. All numbers
below are in weight units, before the dt factor. rsl-rl's
`Episode_Reward/<term>` curves are also printed after the dt scaling.

Every term reads one shared per-step bookkeeping pass (`_update` in
`wuji_synergy_grasp_env_cfg.py`). It runs once per step, guarded by the step
counter, and computes the contact streaks, the hit flags, the hold streak and
the success flag. It is called from the terms rather than registered as a
zero-weight term, because Isaac Lab 3.0 silently skips zero-weight terms.

"Once" terms use `_once`: a per-env flag records that the term has paid, and
it pays 1.0 only on the first step its condition is true. The flags are
cleared at reset.

### Where the forces come from

- **Fingertips.** There is one `ContactSensor` per fingertip, filtered against
  the table's collider mesh. The sensor reports the **net** force and the
  **table** force. The force on the **bolt** is their vector difference:
  `|F_net − F_table|`. A filter on the bolt itself reported zero while the
  fingers were visibly pushing it; self-collision is off, so net minus table
  is the bolt.
- **Contact.** A fingertip is "touching" when its bolt force exceeds 0.1 N,
  and "stable" once it has touched for **4 consecutive control steps**
  (133 ms). A one-step brush earns nothing. "Thumb stable" is the thumb tip;
  "finger stable" is the index **or** middle tip.
- **Structure.** Separate sensors sit on the palm (`wristRollRight`, which
  includes the merged hand base), the forearm (`wristPitchRight`), the upper
  arm (`elbowRight`), and the proximal and middle phalanges of the thumb,
  index and middle fingers. The fingertip sensors miss a finger lying along
  the table on its middle phalanx; these catch it.
- **Bolt on the table.** This is geometric (`bolt_table_clearance`, the gap
  from the bolt's lowest possible point to the table top), not a contact
  sensor. A broken contact filter reports zero force, which would make "the
  bolt never touched the table" silently always true.

### Term by term

**`success`, +20, last step only.** Pays 1 on step 150 (the time-out) when
**all** of these hold:

1. the bolt's centre was more than **5 cm above its rest height** for at least
   the last 30 consecutive steps (the whole HOLD phase);
2. the bolt never touched the table during HOLD (geometric clearance > 0);
3. no finger-table hit anywhere in the episode (the `table_contact` flag
   below);
4. no fingertip force over 40 N anywhere in the episode (the `excess_force`
   flag below).

An episode that ends early (structural contact, joint limit, IK failure, bolt
off the table) never reaches step 150, so it can never collect it. This is
the only term that says "the task was done". Everything else either points
toward it or disqualifies it.

**`thumb_contact` +0.5, `finger_contact` +0.5, `opposing_contact` +1.0,
once each.**

- Paid on the first step the thumb becomes stable, the first step index or
  middle becomes stable, and the first step both are stable at once.
- The opposing bonus is double because a pinch needs both sides; either
  finger alone can only push the bolt.
- They are one-off because a per-step contact bonus summed over the 90-step
  grasp would outweigh the +20. That is the failure Experiment 2's two-finger
  run hit: it banked contact reward while sitting open.
- Together they are worth at most 2.0, a tenth of success. They mark the
  route to the goal and cannot replace it.
- They are not phase-gated, but in practice contact is made during GRASP.

**`approach`, +2 × (Φₜ − Φₜ₋₁), GRASP phase only.** Dense shaping from the
fingertips' distance to the bolt:

    Φ = ½ [ (1 − tanh(d_thumb / 2 cm)) + (1 − tanh(min(d_index, d_middle) / 2 cm)) ]

- **The distances.** Each d is from a fingertip (a known offset from the
  fingertip body, by forward kinematics) to the bolt's **capsule** surface:
  distance to its axis segment minus the 1.25 cm radius. These are available
  before contact, which the force observations are not.
- **The range of Φ.** Φ is 1 with both sides touching and 0 far away. At the
  undisplaced start (tips ~1.6 cm out), Φ ≈ 0.34.
- **Change, not value.** The term pays the *change* in Φ, so over an episode
  it sums to 2 (Φ_end − Φ_start), whatever path the hand takes.
  - Closing from the level-0 start is worth about 2 × (1 − 0.34) ≈ 1.3.
  - Moving closer pays, moving away costs the same back, standing still pays
    exactly 0, and nothing can be farmed.
  - Paying Φ itself would reward hovering next to the bolt for 90 steps,
    which beats the +20.
- **Why it was added.** It is the term that made the wrist trainable. With
  the wrist free, the sparse contact bonuses alone gave no signal toward the
  bolt (mask curriculum stage 2: 0 %).
- **Why σ = 2 cm.** It puts the starting 1.6 cm gap on the steep part of
  tanh, where the gradient is usable.
- **Reset.** The previous-step Φ is re-seeded at every reset. Otherwise the
  first step of an episode would be measured against the previous episode's
  last step.

**`drop`, −5, once.**

- **Trigger.** The bolt was more than 2 cm above rest at some point after the
  grasp phase, and is now back below 2 cm.
- **Why it is large.** Dropping a lifted bolt is the most direct failure of
  the task. It is weighted above the table and force penalties because a
  policy should not learn that a risky lift costs nothing when it fails.

**`table_contact`, −3, once.** Set the first time any fingertip pushes the
table with more than **15 N**, or any proximal or middle finger phalanx does.

- **Why 15 N.** Light resting contact is normal: the fingertips sit on the
  table in every pose that grasps this low bolt. Ploughing into it, measured
  at 20–44 N with the hand too low, is not.
- **Effect on success.** It also disqualifies success, so a table hit really
  costs −3 plus the lost +20.
- **Where it bites.** At level 3 this is the whole remaining failure rate.

**`excess_force`, −3, once.** Set the first time any fingertip's **net**
force exceeds **40 N**.

- It also disqualifies success.
- **Why 40 N.** The spec gave no number. 5 N and 20 N were tripped by every
  scripted pinch that held the bolt: the position drives squeeze at ~25 N.

**`structural_contact`, −5, once, and ends the episode.** Triggers when the
palm, forearm or upper arm pushes the table with more than **5 N**.

- **Why 5 N, not 15 N.** A finger touching the table can be part of a good
  grasp; the palm or forearm touching it never is.
- **Why it terminates.** Without the termination, a policy could rest its
  palm on the table, pay −5 once, and still collect +20. Terminating makes it
  cost the −5, the +20, and whatever else was left in the episode.

**`joint_limits`, −0.05 × Σᵢ ((0.15 − sᵢ) / 0.15)², every step.**

- **What sᵢ is.** The normalised slack of arm joint *i*:
  sᵢ = 2 · min(q − q_lo, q_hi − q) / (q_hi − q_lo). It is 1 at mid-range
  and 0 at a hard stop, computed on the commanded (clipped) joint target.
- **When it applies.** It is zero while every joint has at least 15 % of its
  half-range left, and grows quadratically inside that band.
- **Why it is small.** A limit penalty that could outweigh +20 teaches the
  policy to park mid-range and do nothing. Its job is only to add gradient
  near the stops.
- **Other defences.** The hard boundary is the separate termination at
  s_min < 0.02. The IK also clips targets 0.05 rad inside every URDF stop.

**`action_rate`, −0.001 × ‖aₜ − aₜ₋₁‖², every step.**

- **What it measures.** The sum of squared changes over all 8 actions. The
  worst case, every channel flipping from −1 to +1, costs −0.032 per step.
- **What it does.** It smooths the commands. It does not penalise holding a
  constant command, so it cannot stop steady drift; the early-grasp latch
  deals with that structurally.
- **Why it is small.** It was −0.01 in Experiment 2. At that weight it was
  the only term with a gradient once the wrist could move, and the stage-2
  policy optimised it by holding a constant action.

### What an episode is worth

| outcome | terms collected | total (weight units) |
|---|---|---|
| clean success from the level-0 start | 0.5 + 0.5 + 1 + 20 + ≈1.3 approach | **≈ 23.3** |
| pinched and lifted, then dropped | 2 + 1.3 − 5 | ≈ −1.7 |
| lifted cleanly, but a finger hit the table | 2 + 1.3 − 3 (success void) | ≈ 0.3 |
| palm on the table mid-grasp | whatever contact was made − 5, episode ends | ≤ −3 |
| never touched the bolt, held still | 0 | 0 |

At the larger displacements the wrist starts further away, so Φ_start is
lower and `approach` is worth somewhat more, up to 2 in total. Success is
still ~85 % of the best possible return. Every other term is too small to
change which behaviour is best; they only change how quickly the policy
finds it.

### Checking the reward before training

`tools/audit_reward.py` runs four fixed policies over 100+ episodes: always
open, half closure, random, and the scripted full close. It prints every term
and checks that none of them collects shaping reward by accident, for
example `approach` paying a random policy or contact bonuses paying one that
never grips. It passed 8/8 at level 0. Run it after any reward change.

Relative to Experiment 2, the one term that made the difference is:

    approach          2 (Phi_t - Phi_{t-1})     during GRASP only
    Phi = 0.5 [ (1 - tanh(d_thumb/sigma)) + (1 - tanh(d_opposing/sigma)) ]
    d_opposing = min(d_index, d_middle),  sigma = 2 cm

**Progress in the potential, never its value.** Rewarding Phi itself pays a
policy for loitering near the bolt, and over 90 steps that outweighs the +20
for succeeding. Progress telescopes — the episode total is
`2 (Phi_end - Phi_start)` regardless of path — so there is nothing to farm.
Distances are to the bolt's **capsule** surface; a bounding sphere (1.75 cm
against the real 1.25 cm radius) would report contact with the tips still 5 mm
out.

`action_rate` is **−0.001**, an order of magnitude below Experiment 2's −0.01.
At −0.01 it was the only term with a gradient once the wrist could move, and
stage 2's policy optimised it by holding a constant action.

## Two defects found and fixed during these runs

**The scripted lift slid sideways.** Measured on the level-2 policy: at the end
of the grasp phase the arm was **12.0 mm behind its commanded target**
laterally (a third of steps had their IK target clipped). The lift then held
the lateral target still while ramping z, so the arm paid that error off by
sliding — 11 mm of horizontal wrist travel, **dragging the bolt 19.9 mm across
the table** while rising. The fix is structural, not a reward: re-seed the
target's position and orientation from where the wrist *actually is* on the
first step of the lift.

| during LIFT+HOLD | before | after |
|---|---|---|
| wrist horizontal travel | 11.1 mm | **1.8 mm** |
| bolt dragged | 19.9 mm | **1.2 mm** |

Success was unchanged at 99.8 % and both remaining failure modes went to zero
(drops 0.2 → 0.0 %, table collisions 0.2 → 0.0 %). Same family as Experiment
2's lift-axis bug, which dragged the bolt 7 cm while reporting a correct rise.

**Exploration noise ratcheted up the curriculum.** With the spec's
`entropy_coef = 0.01` the action std inflated within every run *and* was
inherited by the next: **0.98 → 1.15 → 2.25** over levels 0–2. At level 2 it
stopped being cosmetic — the last 148 iterations took success from 99.8 % down
to 96.1 % and finger-table collisions from 0.2 % to 3.9 %, because wider noise
on the Cartesian channels drives the fingers into the table. Evaluation uses
the mean action, so this hides in the success curve until it does not.

Fixed by `entropy_coef = 0.003` and a `--reset-std` flag on resume: a
curriculum should inherit the policy, not its exploration noise. Level 3 ran at
double level 2's displacement with markedly healthier arm behaviour:

| | level 2 | level 3 |
|---|---|---|
| final action std | 2.25 | **0.77** |
| clipped IK targets | 24,949 | **10,096** |
| min arm joint slack (mean) | 0.216 | **0.324** |

## Late-run degradation, and which checkpoint to use

Every level so far peaks and then declines. Level 2 best 99.7 % at iteration
669, final 97.1 %; level 3 best 98.4 % at 859, final 93.9 %. The decline is
specifically finger-table collisions. **Use the best checkpoint, not the last**
— the table above lists them.

## What limits level 3, and it is not learning

The remaining 2.9 % failure at level 3 is **entirely** finger-table collisions:
drops 0.0 %, excess force 0.0 %. Without that disqualifier the grasp itself
succeeds ~99.4 %. The clearance sweep predicted it before training — at level
3's −5° of roll the lowest fingertip has **0.31 cm** across the closure stroke
against 1.08 cm undisplaced. The hand is running out of room under the table.
More training will not help; narrowing roll/pitch or re-solving the pre-grasp
pose for clearance headroom would.

## Known gaps

- **There is no reaching.** Every level starts at the contact pose with a small
  pose error. The original stage-1 design had a 5 cm standoff and a real
  descent; the displacement rewrite dropped it. Standoff would have to come
  back as its own curriculum axis.
- **LIFT and HOLD still train the actor.** 60 of every 150 transitions feed the
  PPO actor loss with actions the environment discards. Fixing it needs
  `step()` overridden so the episode is 90 steps from rsl-rl's view.
- **The arm still works near its limits.** Minimum joint slack 0.052 against a
  0.02 hard-termination threshold, even after the entropy fix.
- **`Safety/*` metrics never reach TensorBoard.** `extras["log"]` is not
  forwarded by the rsl-rl wrapper; `eval_pipeline.py` reports them instead.
- One unexplained **844 mm** wrist tracking-error outlier at level 1. It did not
  recur at levels 2 or 3 (worst 115 mm) and has no root cause.

