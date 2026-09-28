# Training log

How the algorithm itself works — observations, actions, reward, success test,
reading these logs — is in [`ALGORITHM.md`](ALGORITHM.md).

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

# Overview — every experiment designed, and what came of it

State as of 2026-09-23. All experiments grasp the same object: the 30 g bolt
`rod-with-threads-0001` (a 2.5 cm-diameter, 3.5 cm-long rod) on the
adam-moveit table, with the right Wuji Hand 2 on Adam-U.

## The experiments

| # | question it answers | what the policy controls | actions | status | headline result |
|---|---|---|---|---|---|
| **1** | Can PPO grasp and lift at all, with every joint free? | 7 arm joints + 20 hand joints, direct joint targets | 27 | trained | **99.4 %** still holding at the end of the episode; the grasp is a "scoop", not a pinch |
| **2** | With the wrist fixed at a solved pose, can two finger synergies make a real pinch? | thumb synergy + fingers synergy | 2 | trained | **100 %** with 3 fingers; **0 %** with 2 fingers (exploration failure) |
| **3** | Do glove demonstrations (BC, then PPO) make the 20-DoF hand learnable? | all 20 hand joints, wrist fixed | 20 | **designed, not trained** | none yet; blocked on the Manus glove not being detected |
| **Pipeline, mask curriculum** | Can the wrist be added stage by stage by switching action channels on? | 6 wrist residuals through IK + 2 synergies, some masked | 8 | trained, abandoned | stage 0 **100 %**, stage 2 **0 %**, which motivated the rebuild |
| **Pipeline, displacement curriculum** | Can the policy correct a wrong starting wrist pose, level by level? | 6 wrist residuals through IK + 2 synergies, all live | 8 | trained, **needs retraining** | L1 99.4 %, L2 99.8 %, L3 96.5 % as logged; **76–95 % on today's code** |
| **4** | Can the policy approach from 5–10 cm above along −z, correct its pose, pinch, and never touch the table, with a masked actor loss and whole-body contact monitoring? | 6 wrist residuals through IK + 2 synergies, all live | 8 | **levels 1–7 trained** (2026-09-24) | **≥ 99.8 %** at every level, 0 invalid collisions, against baselines falling from 100 to 32.8 %; shield and clutter (level 8) pending |

How they build on each other:

1. **Experiment 1 → 2.** Experiment 1 worked, but it wasn't a pinch.
   Experiment 2 fixes the wrist at a solved pinch pose and makes the reward
   contact-based.
2. **Experiment 2 → pipeline.** Experiment 2 only works with a fixed wrist.
   The pipeline puts the wrist back under policy control through IK:
   - The mask curriculum failed.
   - The displacement curriculum replaced it.
3. **Experiment 2 → Experiment 3.** Experiment 3 takes Experiment 2's task to
   full 20-DoF finger control. Its 2-finger failure is the evidence that
   exploration alone doesn't find hard grasps, which is the reason for using
   demonstrations.

## Reward functions compared

Weights as configured. Isaac Lab multiplies every term by dt = 1/30 s, which
scales all terms equally (see "How the reward is computed" in the pipeline
section). "Once" means paid on the first qualifying step of an episode only.

| term | Exp 1 | Exp 2 | Exp 3 (planned) | Pipeline |
|---|---|---|---|---|
| alive, per step | +0.5 | — | — | — |
| terminated early | −2 | — | — | — |
| fingertip proximity | `reach_fine` +2, per step: 1 − mean tanh(d/4 cm), thumb + index to the bolt centre | — | — | `approach` +2 × ΔΦ, grasp phase only, capsule surface, σ 2 cm |
| closing near the bolt | `grasp` +3, per step: proximity gate × flexion closure | — | — | — |
| stable thumb contact | — | +0.5 once | = Exp 2 | +0.5 once |
| stable opposing-finger contact | — | +0.5 once | = Exp 2 | +0.5 once |
| both at once | — | +1 once | = Exp 2 | +1 once |
| bolt off the table | `lift_bonus` +5 per step above 2 cm; `object_height` +15 × height (cap 8 cm) | — | — | — |
| **success** (held ≥ 5 cm through the hold, bolt clear of the table, no collision, no over-force) | — (no success test) | **+20** at the last step | = Exp 2 | **+20** at the last step |
| drop after lift | — | −5 once | = Exp 2 | −5 once |
| finger-table collision > 15 N | — | −3 once | = Exp 2 | −3 once (tips and phalanges) |
| fingertip force > 40 N | — | −3 once | = Exp 2 | −3 once |
| palm/forearm/upper arm on the table | — | — | — | −5 once, and the episode ends |
| bolt knocked below the table | `off_surface` −2 × depth | — | — | — (episode ends) |
| wrist drift | −5 × horizontal distance² | — (wrist fixed) | — (wrist fixed) | — (the latch freezes the wrist) |
| arm near a joint limit | — | — | — | −0.05 quadratic |
| action rate | −5e-4 | −0.01 | = Exp 2 | −0.001 |
| arm joint velocity | −1e-5 | — | — | — |

The reward changed in three stages:

1. **Experiment 1 is dense.** It pays every step for being close, closing and
   being high. It has no notion of success, so "held" is only measured
   afterwards by `eval_grasponly.py`.
2. **Experiment 2 is sparse.** A +20 terminal success dominates, the one-off
   contact bonuses mark the route to it, and one-off penalties disqualify bad
   grasps. The per-step proximity terms are gone, because they paid for
   hovering near the bolt.
3. **The pipeline is Experiment 2's reward plus two additions:**
   - one dense term, `approach`, that can't be farmed, because it pays only
     progress toward the bolt;
   - structural and joint-limit safety terms, needed now that the arm moves.

   It also lowers `action_rate` to −0.001, because at −0.01 holding still was
   the policy's best option.

## Runs not written up anywhere else

Success here is TensorBoard's `Episode_Reward/success` converted to a
fraction of episodes. This is a training-time number, measured with
exploration noise on, and is not an evaluation.

| run | iterations | what it appears to be |
|---|---|---|
| `pipeline_L0/2026-09-22_15-00-23` | 650 → 1049 | resumed from a `model_650` (level 2's or level 3's, not recorded) and trained at level 0, probably on the code with the early latch; training success ~99 % |
| `pipeline_L0/2026-09-22_13-09-38` | 650 → 714 | a short attempt at the same; ~78 % |
| `pipeline_L0/2026-09-22_13-18-58` | 650 → 656 | aborted after a few iterations |
| `pipeline_L3/2026-09-22_16-24-54` | 850 → 854 | aborted; mean episode length 90 instead of 150, i.e. the early latch was active — the start of retraining level 3 on the new code |
| `pipeline_s0` 16-29, 16-37, 16-44; `pipeline_s1/2026-09-21_16-30-06`; `grasponly/2026-09-21_15-46-58`; `synergy_f3/2026-09-21_15-46-22` | 2–89 | smoke tests |
| `adam_u_grasp`, `adam_u30_grasp` | up to 752 | the older PND-hand grasp work (September 2025 and 2026-09-04); `adam_u30_grasp/2026-09-04_14-41-15` ran 752 iterations and never lifted (`lift_bonus` ≈ 0) |

---

## Experiment 1 — Wuji Hand 2, grasp-only, full joint control

**Date** 2026-09-19 · **Run** `adam_u30_wuji_grasponly/2026-09-19_07-47-30` ·
**Status** complete, 600 iterations, ~35 min

| | |
|---|---|
| env | `genesis_rl/envs/adam_u_wuji_grasponly_env_cfg.py` (`AdamUWujiGraspOnlyEnvCfg`) |
| robot | Adam-U + Wuji Hand 2, `assets/robots/adam_u_wuji/usd/adam_u_wuji.usd` (60 bodies, 59 DoF) |
| scene | adam-moveit table (top 0.95 m), one bolt `rod-with-threads-0001` (30 g) at (0.39, −0.21) ± 1.5 cm, random yaw, no bin |
| start | torso leaned 0.35 rad, right hand palm-down over the bolt, fingers pre-curled ~30 %, thumb cmc_abd −0.95 (`PREGRASP_ARM`, `PREGRASP_FINGERS`) |
| actions | 27: 7 arm joints ±0.2 rad around the start pose, 20 right-hand joints offset from the pre-curl (+1 = fully flexed) |
| observations | 101: arm q/qd (7+7), wrist pos/quat (3+4), hand q/qd (20+20), bolt pos/quat (3+4), bolt−wrist (3), last action (27), base pos (3) |
| reward | alive 0.5, terminating −2, reach_fine 2 (thumb+index tip to bolt, std 4 cm), grasp 3 (proximity-gated flexion closure), lift_bonus 5 (bolt > 2 cm above rest), object_height 15 (ramp, cap 8 cm), off_surface −2, wrist_drift −5, action_rate −5e-4, joint_vel −1e-5 |
| episode | 5 s, 30 Hz control, 120 Hz physics; ends on time-out or bolt 7.5 cm below the table |
| algorithm | PPO, rsl-rl 5.0.1, `rsl_rl_ppo_cfg.py`: actor/critic MLP 512-256-128 elu, obs normalisation, lr 1e-3 adaptive (KL 0.01), γ 0.99, λ 0.95, clip 0.2, entropy 0.005, 32 steps/env, 8 epochs, 4 minibatches |
| command | `python genesis_rl/scripts/train.py --robot wuji --scenario grasp --num_envs 4096 --max_iterations 600` |
| hardware | RTX PRO 6000 Blackwell; 2.5–3.7 s / iteration at 4096 envs, ~35 min |

**Goal.** The first feasibility check on the Wuji hand. With the arm and all
20 finger joints free, does PPO find any way to get the bolt off the table
and keep it there? The bolt spawns between the pre-curled fingers, so there
is no reaching.

**Reward in detail.** Every term is dense: paid every step, scaled by dt like
all Isaac Lab terms. Distances are to the bolt's **centre**, not its surface.

| term | weight | per-step value | purpose |
|---|---|---|---|
| `alive` | +0.5 | 1 | small constant, keeps episodes worth continuing |
| `terminating` | −2 | 1 on the step an episode ends early | ending early (bolt 7.5 cm below the table) costs |
| `reach_fine` | +2 | 1 − ½ [tanh(d_thumb / 4 cm) + tanh(d_index / 4 cm)] | pulls the thumb and index tips to the bolt; 1 at the bolt, ~0 at 10 cm |
| `grasp` | +3 | gate × closure. gate = 1 − tanh(mean(d_thumb, d_index) / 4 cm); closure = mean flexion q / q_upper over the flexion joints | pays for curling the fingers **only when the tips are at the bolt**; curling in free air earns nothing |
| `lift_bonus` | +5 | 1 if the bolt is > 2 cm above rest | step reward for getting it off the table at all, so lifting beats hovering |
| `object_height` | +15 | clamp(z − z_rest, 0, 8 cm), in metres, so at most 1.2 per step | ramp for lifting higher; capped at 8 cm, which is what the ±0.2 rad arm range can lift |
| `off_surface` | −2 | clamp(z_rest − 1 cm − z, 0, 0.5) | charges for pushing the bolt below its resting height |
| `wrist_drift` | −5 | horizontal distance² of the wrist from its start | vertical motion is the job; horizontal wandering is not |
| `action_rate` | −5e-4 | ‖aₜ − aₜ₋₁‖² over 27 actions | smoothness |
| `joint_vel` | −1e-5 | Σ q̇² over the 7 arm joints | smoothness |

The best possible per-step value is 0.5 + 2 + 3 + 5 + 1.2 ≈ 11.7, reached with
the tips on the bolt, the fingers fully closed and the bolt 8 cm up. The lift
terms (6.2) are worth more than proximity and closing (5), so the policy is
pushed past "touch it" to "lift it". There is no success term: whether the
bolt was *held* is measured only afterwards, by `tools/eval_grasponly.py`.
The weights were carried over from the reach scenario and never tuned.

**What the result means.** The policy lifts the bolt in 100 % of episodes and
is still holding it at the end in 99.4 %. But it does so by curling the
fingers under the bolt and pinning it against the thumb and palm: a scoop,
not a pinch. The dense proximity and closure terms paid for *any* closed hand
near the bolt, and nothing in the reward asked for opposing contact.
Experiment 2 exists to fix that.

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

    python genesis_rl/scripts/play.py --robot wuji --scenario grasp --viz kit --num_envs 4 --real-time \
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

**Date** 2026-09-19 · **Runs** `adam_u30_wuji_synergy/2026-09-19_09-06-47` (v1, four-finger rake),
`adam_u30_wuji_synergy/2026-09-19_12-08-53` (v2, 3 fingers),
`adam_u30_wuji_synergy_f2/2026-09-19_12-19-30` (v2, 2 fingers) · **Status**
complete: v1 99.6 %, v2 3 fingers **100 %**, v2 2 fingers **0 %**

| | |
|---|---|
| env | `genesis_rl/envs/wuji_synergy_grasp_env_cfg.py`; `ADAM_U_FINGERS` = 2, 3 (default) or 4 |
| robot | the Wuji hand alone (`assets/robots/wuji_hand_right/`), wrist fixed at a solved pinch pose, with a prismatic lift joint for the scripted lift |
| goal | a thumb-opposed pinch that holds the bolt through a lift, not just any closed hand |
| actions | 2: thumb synergy and fingers synergy, each from −1 (open pose) to +1 (closed pose), smoothed by α = 0.3 per step. Ring and little are held flat |
| observations | 26 (3 fingers): 9 synergy joint positions + 9 velocities, last action (2), fingertip force on the bolt ÷ 40 N (3), contact flags (3) |
| episode | 120 steps at 30 Hz: **GRASP** 60 (2 s, policy acts), **LIFT** 30 (scripted 10 cm), **HOLD** 30 (scripted, verification) |
| algorithm | PPO, `RslRlWujiSynergyPPORunnerCfg`: actor/critic MLP 128-128 elu, init std 0.5, lr 3e-4 fixed, entropy 0.01, γ 0.99, λ 0.95, clip 0.2, 24 steps/env, 5 epochs, 4 minibatches |
| command | `docker compose exec [-e ADAM_U_FINGERS=2] adam_rl python genesis_rl/scripts/train.py --robot wuji --scenario synergy --num_envs 1024` (500 iterations, ~6 min) |

**Reward in detail.** It is sparse: nothing pays per step except the small
action-rate cost. The same bookkeeping, force measurement and success test
carry over unchanged to the pipeline, whose Reward section has the full
mechanics: how fingertip-on-bolt force is measured, what "stable" means, and
the dt scaling.

| term | weight | paid when |
|---|---|---|
| `thumb_contact` | +0.5 | once: thumb tip on the bolt > 0.1 N for 4 consecutive steps |
| `finger_contact` | +0.5 | once: index or middle tip the same |
| `opposing_contact` | +1.0 | once: both stable at the same time |
| `success` | **+20** | step 120: bolt > 5 cm above rest for all 30 hold steps, never touching the table during the hold, no finger-table hit > 15 N and no fingertip force > 40 N anywhere in the episode |
| `drop` | −5 | once: the bolt was > 2 cm up after the grasp phase, then came back below 2 cm |
| `table_contact` | −3 | once: any fingertip pushes the table > 15 N (also voids success) |
| `excess_force` | −3 | once: any fingertip's net force > 40 N (also voids success) |
| `action_rate` | −0.01 | every step: ‖aₜ − aₜ₋₁‖² over the 2 actions |

Why it is shaped this way:

- **Contact bonuses are paid once, and only after 4 steps.** Paid per step,
  60 grasp steps of contact would outweigh the +20. Paid on the first touch,
  exploration noise brushing the bolt would collect them.
- **Success is the only term that says the task was done**, and it is 10× all
  the contact bonuses together. The penalties both cost points and void the
  success, so a grasp that works only by ploughing into the table or crushing
  the bolt scores worse than not grasping at all.
- **A clean success is worth 22.0. A pinch that lifts, then drops is −3.0.
  Never touching the bolt is 0.**

**The weakness this reward has, measured.** The 2-finger run exposed it: a
reward a random policy collects in full carries no gradient. Exploration
noise tripped all three contact bonuses in every episode while the policy's
mean stayed fully open. With +20 out of reach by chance, the only slope left
was `action_rate`, which rewards holding any constant action. The pipeline's
`approach` term and its −0.001 action rate are the fixes for exactly this.

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

## Still open for Experiment 2

- Bolt/hand pose randomisation (±5 mm, ±1 cm, ±2 cm), to measure the basin the
  learned closure covers. The pipeline's displacement curriculum answers a
  version of this with the wrist free.
- Three untried ways to make 2 fingers work:
  - a dense grasp term;
  - a pose with more squeeze margin;
  - starting from the 3-finger policy.

---

## Experiment 3 — imitation learning, then reinforcement learning (designed, not trained)

**Status** Environment, glove link, retarget bridge and recorder are built.
They were validated end to end with a **synthetic** glove: 5/5 episodes
recorded and labelled SUCCESS, no packet loss. **No demonstrations recorded,
no training.** Blocked on the real Manus glove: the SDK detects no glove, so
`/manus_glove_0` has zero publishers (`TODO.md` §2a).
Full design: [`EXPERIMENT3_DESIGN.md`](EXPERIMENT3_DESIGN.md).

**Goal.** Show that human demonstrations make a hard action space learnable.
Pure PPO already solves Experiment 2's 2-synergy task at 100 %, so
demonstrations would prove nothing there. Experiment 3 controls **all 20**
finger joints, where exploration is expected to fail. The Manus glove's 20
ergonomic angles map one to one onto the Wuji hand's 20 joints.

| | |
|---|---|
| env | `genesis_rl/envs/wuji_teleop_grasp_env_cfg.py` |
| robot | the Wuji hand alone, wrist fixed at Experiment 2's pinch pose |
| actions | 20: one per joint, q_target = q_lower + (a + 1)/2 · (q_upper − q_lower), so −1 is the joint's lower limit and +1 its upper; `A_OPEN` is the action vector that reproduces the pre-grasp pose. (The design doc maps −1 to the pre-grasp pose instead; the built env uses the full range.) |
| observations | 66: 20 joint positions + 20 velocities, last action (20), fingertip force on the bolt (3), contact flags (3) |
| episode | the same as Experiment 2: GRASP 60 steps, then the scripted LIFT 30 and HOLD 30 |
| reward and success | **identical to Experiment 2** (table above), so every stage is scored by the same test |

**Planned stages.**

1. **Record.** The operator wears the glove, and the simulated hand mirrors
   their fingers from the pre-grasp pose. The data path is:
   glove → `manus_ros2` → `tools/manus_wuji_bridge.py` → UDP →
   `tools/record_demo.py`.
   - Each demonstration is labelled success or failure automatically, by
     Experiment 2's test.
   - The target is 20–50 successful ones. Record one and replay it first.
2. **Data tooling.** `tools/demo_io.py` validates, resamples, clamps and
   splits by trajectory. `tools/replay_demo.py` checks that a recording still
   succeeds when replayed open-loop. Neither is written yet.
3. **Behaviour cloning.** Regression from observation to the **commanded**
   action over the grasp phase of the successful demonstrations.
   - Loss: Gaussian NLL, so the learned std carries into PPO.
   - Network: the same as the PPO actor.
   - The observation normaliser is frozen into the checkpoint.
   - Not written yet: rsl-rl 5.0.1 has no auxiliary-loss hook.
4. **PPO fine-tuning.** Load the BC actor; the critic starts fresh. Use lr
   ~1e-4 and start from the BC std. The reward is Experiment 2's.
   - Fallback: 5 per-finger synergies instead of 20 joints.

**What would make it conclusive.** Four policies on the same 512-episode
protocol:

1. PPO-only at 20 DoF, from scratch;
2. BC only;
3. BC, then PPO;
4. Experiment 2's 3-finger policy as the 100 % reference.

Also: how many demonstrations are needed, with BC on 5, 10, 20 and 50.

The experiment succeeds if **PPO-only fails and BC + PPO succeeds**. If
PPO-only also succeeds, the task didn't need demonstrations. That is a real
result, but a different one.

**The question to answer first.** Can a human produce a working grasp through
the glove from this pose? The fingertips start 1.6 cm from the bolt with a
5.2 cm thumb-to-finger gap. One recorded demonstration answers it.

---

# Staged pipeline — whole robot, IK wrist, displacement curriculum

Design: [`GRASP_PIPELINE_DESIGN.md`](GRASP_PIPELINE_DESIGN.md).
Environment: `genesis_rl/envs/wuji_pipeline_env_cfg.py`.

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

    docker compose exec -e ADAM_U_LEVEL=<n> adam_rl python genesis_rl/scripts/train.py \
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



---

# Experiment 4 — safe reach-to-grasp (built 2026-09-24, not yet trained)

Design: [`EXPERIMENT4_DESIGN.md`](EXPERIMENT4_DESIGN.md).
Environment: `genesis_rl/envs/wuji_reach_grasp_env_cfg.py`. Level:
`ADAM_U_RG_LEVEL`. Runs will land in `logs/rsl_rl/adam_u30_wuji_reach_grasp_L<n>/`.

The pipeline's task with a standoff: the wrist starts 5–10 cm above the solved
pinch pose (which is kept; "from above" is the wrist trajectory, not a
palm-down hand) plus a pose error, descends along −z, pinches, and hands over
to a scripted settle / lift / hold. 74 observations (the pipeline's 58 + 16
proxy-model clearances), 8 actions, episode up to 120 policy steps + 70
scripted. Only the policy steps train the actor (`MaskedPPO`). Any invalid
collision (fingertip or phalanx on the table above 15 N, palm / forearm /
upper arm on the table above 5 N, a non-grasp link on the bolt above 2 N)
pays −10 and ends the episode.

## First increment, what was checked

**Regression tests** (`tools/test_reach_grasp.py`): 43 of 43 at level 0; at
level 3 the same minus the checks that assume an exact start; 5 of 5
perception checks at level 6 (estimate ≠ truth by 1.1 mm mean, bias constant
within an episode and redrawn at reset, `bolt_pos_rel` derived from the
estimate). The latch clears the residual, re-seeds the IK target and
integrator from the measured state, holds the fingers through SETTLE, and the
lift rises 9.5 cm with 0.8 mm of lateral travel at level 0. Masked
transitions contribute exactly zero actor gradient (their advantages scaled
by 1000 leave the gradient unchanged to 0.0).

**Geometry gate, level 0** (`tools/audit_reach_grasp.py`, 128 episodes,
scripted close with no wrist motion): success 100 %, invalid collisions 0,
worst joint slack 0.478, bolt drag 2.0 mm mean / 2.1 mm max, latch at step
26. Passed. An *instant* full close also succeeds 100 % but drags the bolt
6.3 mm mean / 8.8 mm max, over the 5 mm gate: closing speed matters to drag.

**Scripted straight-down baseline** (descend at full speed, then ramp closed,
no correction): level 1 (5 cm, no error) **100 %**, latch at step 41, drag
3.4 / 4.7 mm. Level 2 (5 cm, ±0.5 cm, ±3° yaw): **73.6 %**, latched 81 %,
drag 4.5 mm mean / 31 mm max. Level 3 (7.5–10 cm, ±1 cm, ±5° yaw, ±1° roll/pitch):
**65.6 %**, latched 82 %, drops 5.5 %, drag 6.1 mm mean / 59 mm max (the
misses drag the bolt). Level 4 (7.5–10 cm, ±2 cm, ±7.5° yaw, ±2° roll/pitch):
**49.3 %**, latched 58 %, drag 7.6 mm mean / 62 mm max. Level 5 (level 4 +
physics randomisation): **41.2 %**. Level 6 (+ small perception error, which
the scripted policy ignores, so this measures the physics draw): **34.6 %**.
Level 7: **32.8 %**. The baseline falls with every level, as it should: 100,
100, 73.6, 65.6, 49.3, 41.2, 34.6, 32.8 %.

**Reward audit** (open, closed, random, scripted, unsafe): the do-nothing
policy collects no shaping; shaping totals +0.6 (level 0) to +2.4 (level 1)
against +20; random collects less shaping than it loses; the unsafe policy
(drive down and sideways, never close) is terminated by the invalid-collision
rule in 128 of 128 episodes and returns −11 to −19 against −0.6 for doing
nothing.

**One reward changed after measurement.** The near-object velocity term,
applied literally, charged a wrist that never moved and just closed its
fingers −20 per episode: finger impact spins the bolt at several rad/s for a
few steps and the hand-minus-bolt relative velocity read that as rushing the
object. It is now paid only while no fingertip touches the bolt and capped at
0.5 per step; the scripted grasp then pays −0.02.

**Proxy clearances at the contact pose:** tips 0.7 cm (the mesh probe says
1.1 cm, so the spheres over-approximate by ~4 mm), palm 2.5 cm, forearm
2.0 cm, upper arm 11.1 cm. The forearm proxy is inside its 3 cm margin at the
goal pose, costing −0.09 per episode; the mesh-accurate probe in the second
increment decides whether the proxy or the margin moves.

## Level 1 — fixed 5 cm approach (trained 2026-09-24)

**Run** `adam_u30_wuji_reach_grasp_L1/2026-09-24_10-51-25`, iterations 0–399
from scratch, 2048 envs, ~24 min (3.4 s per iteration). **Best checkpoint
`model_250.pt`.** Command:

    docker compose exec -e ADAM_U_RG_LEVEL=1 adam_rl python genesis_rl/scripts/train.py \
        --robot wuji --scenario reach_grasp --num_envs 2048 --max_iterations 400

Training curve (population mean of the last ended episode, exploration on):

| iteration | success | invalid collisions | bolt drag | grasp end | action std |
|---|---|---|---|---|---|
| 50 | 70.5 % | 1.45 % | 5.4 mm | 75 | 0.28 |
| 100 | 95.8 % | 0 | 5.0 mm | 30 | 0.27 |
| 200 | 100 % | 0 | 2.7 mm | 22 | 0.18 |
| 250 | 99.8 % | 0 | 2.3 mm | 21 | 0.17 |
| 399 | 100 % | 0 | 4.6 mm | 21 | 0.13 |

Collisions never exceeded 1.5 % and were zero from iteration 100. The actor
mask fraction was 0.65: a third of every rollout was scripted steps that did
not train the actor.

**Evaluation** (`tools/eval_reach_grasp.py`, 512 episodes, mean action):

| | `model_250` | `model_399` |
|---|---|---|
| lift-and-hold success | **100.0 %** | 100.0 % |
| invalid collisions | 0 | 0 |
| latched, mean grasp end | 100 %, step 20.8 | 100 %, step 20.4 |
| bolt drag, mean / max | **2.3 / 2.6 mm** | 4.5 / **6.2 mm** |
| peak fingertip force | 9.9 N | — |
| min clearance tips / palm / forearm | 0.31 / 2.5 / 1.8 cm | 0.32 / — / — |
| worst joint slack | 0.494 | 0.491 |
| gate | **passes all six** | fails drag |

**Which checkpoint, and why.** Success is saturated from iteration 200 on,
but bolt drag during the lift climbs again over the last 100 iterations as
the policy speeds up (grasp end 21 → 20 steps, closure less symmetric: thumb
0.57 vs 0.71). The final checkpoint drags 6.2 mm at worst, over the 5 mm
gate; iteration 250 drags 2.6 mm. Level 2 resumes from `model_250`. Same
lesson as the pipeline: the last checkpoint is not the best one.

The scripted no-correction baseline at level 1 is also 100 %, so this level
matches rather than beats it. It exists to learn the descent and the closure
with the wrist live; wrist correction is first asked for at level 2.

## Level 2 — small correction from 5 cm (trained 2026-09-24)

**Run** `adam_u30_wuji_reach_grasp_L2/2026-09-24_11-15-29`, iterations
250–649, resumed from level 1's `model_250` with `--reset-std 0.3`. **Best
checkpoint `model_649.pt`.** Baseline to beat: 73.6 %.

**Transfer.** 95.5 % success 25 iterations after the resume (82 % after ten),
collisions zero throughout. Same shape as the pipeline's level 1 → 2 resume.

| iteration | success | invalid collisions | bolt drag | grasp end | action std |
|---|---|---|---|---|---|
| 275 | 95.5 % | 0 | 2.4 mm | 25 | 0.30 |
| 350 | 99.6 % | 0 | 4.1 mm | 16 | 0.23 |
| 450 | 100 % | 0 | 2.3 mm | 14 | 0.18 |
| 550 | 100 % | 0 | 1.8 mm | 14 | 0.14 |
| 649 | 99.9 % | 0 | 2.6 mm | 13 | 0.12 |

**Evaluation** (512 episodes, mean action): `model_649` 100.0 % success, 0
invalid collisions, latched 100 % at step 15.0, peak fingertip force under
the limit, worst joint slack 0.443, bolt drag 2.6 mm mean / 5.2 mm max.
`model_550`: 100 %, drag 4.6 mm mean / 6.3 mm max. Beats the 73.6 % baseline
by 26 points.

**The drag gate, and what the drag is.** Both checkpoints fail a gate written
as "maximum drag < 5 mm". Measured on `model_649` over another 512 episodes:
the WRIST moves 0.7 mm laterally during the whole lift, in every episode; the
bolt moves 2.5 mm mean, 3.6 mm p95, 4.7 mm max, and the two are uncorrelated
(r = −0.10). So the lift controller is clean and the drag is the bolt
settling in the pinch as it leaves the table, a grasp-quality effect of a few
millimetres whose 512-sample maximum lands on either side of 5 mm by chance.
The gate is now the 95th percentile (3.7 mm here), with the maximum still
reported. Level 3 resumes from `model_649`.

## Level 3 — 7.5–10 cm approach (trained 2026-09-24)

**Run** `adam_u30_wuji_reach_grasp_L3/2026-09-24_11-40-15`, iterations
650–1048, resumed from level 2's `model_649` with `--reset-std 0.3`. **Best
checkpoint `model_1000.pt`.** Baseline to beat: 65.6 %. This is the first
level where the scripted straight-down grasp fails often, so it is the first
real test of the wrist correction.

**Transfer.** Slower than level 1 → 2, as the standoff doubled and roll /
pitch error appeared: 8.6 % five iterations after the resume with 4.7 %
collisions, 75.4 % and zero collisions at 25, 99.8 % at 50. The only place in
the ladder so far where collisions exceeded 1 %.

| iteration | success | invalid collisions | bolt drag | grasp end | action std |
|---|---|---|---|---|---|
| 675 | 75.4 % | 0 | 5.7 mm | 24 | 0.30 |
| 700 | 99.8 % | 0 | 2.6 mm | 22 | 0.27 |
| 850 | 99.8 % | 0 | 2.2 mm | 23 | 0.19 |
| 1000 | 99.8 % | 0 | 1.9 mm | 26 | 0.18 |
| 1048 | 99.9 % | 0 | 2.0 mm | 26 | 0.17 |

No late degradation; the grasp end drifts from 22 to 26 steps as the policy
spends a little longer correcting before it closes.

**Evaluation** (512 episodes, mean action):

| | `model_1000` | `model_1048` |
|---|---|---|
| lift-and-hold success | **100.0 %** | 100.0 % |
| invalid collisions | 0 | 0 |
| latched, mean grasp end | 100 %, step 27.6 | 100 %, step 28.1 |
| bolt drag mean / p95 / max | **2.1 / 3.4 / 4.4 mm** | 2.4 / 3.8 / 4.8 mm |
| min fingertip clearance, worst | 0.75 cm | 0.69 cm |
| worst joint slack | 0.451 | 0.452 |
| gate | passes all six | passes all six |

Beats the 65.6 % baseline by 34 points. Level 4 resumes from `model_1000`.

## Level 4 — wider wrist correction (trained 2026-09-24)

**Run** `adam_u30_wuji_reach_grasp_L4/2026-09-24_12-03-15`, iterations
1000–1399, resumed from level 3's `model_1000` with `--reset-std 0.3`. **Best
checkpoint `model_1250.pt`.** Baseline to beat: 49.3 %.

**Transfer.** The level-3 policy already covered most of this range: 45.7 %
five iterations after the resume, 97.8 % at ten, 99.3 % at 25, collisions
1.1 % → 0 over the same span.

| iteration | success | invalid collisions | bolt drag | grasp end | action std |
|---|---|---|---|---|---|
| 1050 | 99.9 % | 0 | 2.5 mm | 26 | 0.27 |
| 1200 | 100 % | 0 | 1.7 mm | 26 | 0.22 |
| 1250 | 100 % | 0 | 1.6 mm | 26 | 0.21 |
| 1399 | 100 % | 0 | 1.8 mm | 27 | 0.22 |

**Evaluation** (512 episodes, mean action):

| | `model_1250` | `model_1399` |
|---|---|---|
| lift-and-hold success | **100.0 %** | 100.0 % |
| invalid collisions | 0 | 0 |
| latched, mean grasp end | 100 %, step 28.9 | 100 %, step 28.9 |
| bolt drag mean / p95 / max | **1.7 / 2.9 / 4.4 mm** | 1.8 / 3.3 / 6.8 mm |
| min fingertip clearance, worst | 0.32 cm | 0.32 cm |
| worst joint slack | 0.432 | 0.431 |
| gate | passes all six | passes all six |

Beats the 49.3 % baseline by 51 points. **The geometric ladder (levels 1–4)
is complete**, trained in one afternoon, four runs of 400 iterations each,
every level at 100 % with zero invalid collisions on 512-episode evaluation.

| level | baseline | trained (best checkpoint) | resumed from |
|---|---|---|---|
| 1 | 100 % | 100 % (`L1/…/model_250`) | scratch |
| 2 | 73.6 % | 100 % (`L2/…/model_649`) | L1 best |
| 3 | 65.6 % | 100 % (`L3/…/model_1000`) | L2 best |
| 4 | 49.3 % | 100 % (`L4/…/model_1250`) | L3 best |

Worst fingertip clearance over all evaluations is 0.32 cm (level 4), against
a 0.4 cm margin on a proxy that over-approximates by ~0.4 cm; no fingertip
ever pressed the table above 15 N. Wrist jerk peaks at 110–125 m/s³ on the
latch step; not visibly oscillatory, so no jerk term yet.

## Level 5 — physics robustness: pre-flight (2026-09-24)

The mass / material / actuator-gain events (Isaac Lab's own
`randomize_rigid_body_mass`, `randomize_rigid_body_material`,
`randomize_actuator_gains`) were wired but untested. Measured over 64 envs
after a reset: bolt mass 0.024–0.036 kg (0.8–1.2 × 0.030), static friction
0.63–1.15 (0.7–1.3 × 0.9), thumb CMC stiffness 2.71–3.29 (0.9–1.1 × 3.0),
every env a different draw, redrawn at the next reset. Regression suite at
level 5: 40 of 40. Scripted no-correction baseline: **41.2 %**, latched 57 %,
drag 7.1 mm mean / 47 mm max. Resumes from level 4's `model_1250`.

## Level 5 — physics robustness (trained 2026-09-24)

**Run** `adam_u30_wuji_reach_grasp_L5/2026-09-24_12-28-38`, iterations
1250–1649, resumed from level 4's `model_1250` with `--reset-std 0.3`. **Best
checkpoint `model_1649.pt`.** Baseline to beat: 41.2 %.

**Transfer.** 93.6 % ten iterations after the resume, 98.2 % at 25, with
collisions ≤ 0.25 % and drops falling 1.4 → 0.05 %.

| iteration | success | invalid collisions | bolt drag | grasp end | action std |
|---|---|---|---|---|---|
| 1300 | 99.6 % | 0.04 % | 2.3 mm | 27 | 0.31 |
| 1450 | 99.6 % | 0.05 % | 1.8 mm | 27 | 0.29 |
| 1500 | 99.7 % | 0 | 1.7 mm | 27 | 0.28 |
| 1649 | 99.9 % | 0.05 % | 1.9 mm | 28 | 0.29 |

The action std stays near 0.28 instead of shrinking to 0.13 as at levels
1–4: with mass and friction drawn per episode there is no single closing
speed to converge on. Excess force stayed at zero, so the ±10 % gain draw
never pushed a fingertip over 40 N (peak 15 N).

**Evaluation** (512 episodes, mean action):

| | `model_1500` | `model_1649` |
|---|---|---|
| lift-and-hold success | 100.0 % | **100.0 %** |
| invalid collisions | 0 | 0 |
| latched, mean grasp end | 100 %, step 29.5 | 100 %, step 29.4 |
| bolt drag mean / p95 / max | 1.1 / 2.8 / 11.4 mm | 1.6 / **2.5 / 9.0 mm** |
| final synergy (thumb, fingers) | (0.81, 0.57) | (0.84, 0.69) |
| peak fingertip force | 15.3 N | 13.3 N |
| worst joint slack | 0.429 | 0.429 |
| gate | passes all six | passes all six |

The drag tail is longer than at the geometric levels (max 9–11 mm against
4–5 mm): low-friction draws let the bolt settle further in the pinch. The
95th percentile is unaffected. `model_1649` has the smaller tail and the
firmer closure and is what level 6 resumes from. Beats the 41.2 % baseline
by 59 points.

## Level 6 — small perception error (trained 2026-09-24)

**Run** `adam_u30_wuji_reach_grasp_L6/2026-09-24_12-53-47`, iterations
1650–2048, resumed from level 5's `model_1649` with `--reset-std 0.3`. **Best
checkpoint `model_1900.pt`.** Baseline to beat: 34.6 %. First level with a
corrupted bolt estimate: per-episode bias ±2 mm xy, ±1 mm z, ±1°; frame noise
0.5–1 mm; latency 0–1 step. Rewards, latch and success use the true pose.

**Transfer.** 99.7 % ten iterations after the resume. The run was flat from
then on: the level-5 policy already tolerated this much error, because the
closure is driven by contact once the fingers arrive and a 2 mm bias is
inside the 5.2 cm open gap.

| iteration | success | invalid collisions | bolt drag | grasp end | action std |
|---|---|---|---|---|---|
| 1700 | 99.8 % | 0 | 2.1 mm | 28 | 0.31 |
| 1900 | 99.9 % | 0 | 2.4 mm | 28 | 0.29 |
| 2048 | 99.8 % | 0 | 2.3 mm | 28 | 0.31 |

**Evaluation** (512 episodes, mean action):

| | `model_1900` | `model_2048` |
|---|---|---|
| lift-and-hold success | **99.8 %** | 99.8 % |
| invalid collisions | 0 | 0 |
| latched, mean grasp end | 100 %, step 29.0 | 100 %, step 29.6 |
| bolt drag mean / p95 / max | **1.6 / 3.1 / 9.4 mm** | 2.1 / 3.8 / 14.3 mm |
| peak fingertip force | 14.9 N | 15.2 N |
| worst joint slack | — | 0.408 |
| gate | passes all six | passes all six |

The one failure in 512 is a lift that did not stay 5 cm up for the whole
hold: no drop, no collision, no over-force. Beats the 34.6 % baseline by 65
points. Level 7 resumes from `model_1900`.

## Level 7 — moderate perception error (trained 2026-09-24)

**Run** `adam_u30_wuji_reach_grasp_L7/2026-09-24_13-18-02`, iterations
1900–2299, resumed from level 6's `model_1900` with `--reset-std 0.3`. **Best
checkpoint `model_2299.pt`.** Baseline to beat: 32.8 %. Bias ±5 mm xy, ±3 mm
z, ±3°; frame noise 1–2 mm; latency 1–2 steps; 2 % dropout holding the last
valid estimate.

**Transfer.** 98.4 % ten iterations after the resume, 99 % at 25, collisions
zero throughout. Action std settles at 0.33, the highest of any level: with
latency and dropout in the observation, sharper actions do not pay.

| iteration | success | invalid collisions | bolt drag | grasp end | action std |
|---|---|---|---|---|---|
| 1950 | 99.4 % | 0 | 3.1 mm | 28 | 0.31 |
| 2100 | 99.6 % | 0 | 2.5 mm | 28 | 0.33 |
| 2250 | 99.9 % | 0 | 2.4 mm | 27 | 0.33 |
| 2299 | 99.7 % | 0 | 2.8 mm | 27 | 0.33 |

**Evaluation** (512 episodes, mean action):

| | `model_2250` | `model_2299` | `model_2250` **at level 4** (exact perception) |
|---|---|---|---|
| lift-and-hold success | 100.0 % | **100.0 %** | 100.0 % |
| invalid collisions | 0 | 0 | 0 |
| latched, mean grasp end | 100 %, step 28.9 | 100 %, step 29.0 | 100 %, step 28.0 |
| bolt drag mean / p95 / max | 1.7 / 2.8 / 12.7 mm | **1.7 / 2.4 / 11.4 mm** | 1.7 / 2.1 / 9.0 mm |
| final synergy (thumb, fingers) | (0.79, 0.76) | (0.77, 0.74) | — |
| gate | passes all six | passes all six | passes all six |

Beats the 32.8 % baseline by 67 points, and loses nothing on the clean task.

## The ladder through level 7, in one table (2026-09-24)

Seven levels, seven runs of 400 iterations at 2048 envs, about 25 minutes
each, one afternoon end to end. Each level resumed from the previous level's
best checkpoint with the std reset to 0.3. Every level evaluates at ≥ 99.8 %
lift-and-hold success with **zero** invalid collisions over 512 deterministic
episodes and passes all six advancement gates.

| level | what it adds | baseline | trained | best checkpoint |
|---|---|---|---|---|
| 1 | 5 cm standoff | 100 % | 100 % | `L1/2026-09-24_10-51-25/model_250` |
| 2 | ±0.5 cm, ±3° yaw | 73.6 % | 100 % | `L2/2026-09-24_11-15-29/model_649` |
| 3 | 7.5–10 cm, ±1 cm, ±5°, ±1° | 65.6 % | 100 % | `L3/2026-09-24_11-40-15/model_1000` |
| 4 | ±2 cm, ±7.5°, ±2° | 49.3 % | 100 % | `L4/2026-09-24_12-03-15/model_1250` |
| 5 | mass, friction, gain draws | 41.2 % | 100 % | `L5/2026-09-24_12-28-38/model_1649` |
| 6 | bias 2 mm / 1°, noise, latency ≤ 1 | 34.6 % | 99.8 % | `L6/2026-09-24_12-53-47/model_1900` |
| 7 | bias 5 mm / 3°, latency 1–2, dropout | 32.8 % | 100 % | `L7/2026-09-24_13-18-02/model_2299` |

Paths under `logs/rsl_rl/adam_u30_wuji_reach_grasp_`. What the numbers say:

- **The wrist correction is real.** From level 3 on the scripted straight-down
  grasp fails a third to two thirds of the time and the policy does not.
- **Nothing ever touched the table.** Across 14 evaluations of 512 episodes,
  invalid collisions were zero. Worst fingertip proxy clearance 0.32 cm.
- **The last checkpoint is not the best.** At levels 1 and 2 the final
  checkpoint dragged the bolt more than an earlier one; from level 5 on the
  drag tail lengthened with low-friction draws (max 9–14 mm) while the p95
  stayed under 3 mm. Pick by drag, gate on the p95.
- **Perception error was free** at these magnitudes: transfer into levels 6
  and 7 was ≥ 98 % within ten iterations, and the level-7 policy is 100 % on
  the clean task.
- **Untested by design:** the safety shield (α = 1 everywhere, so the
  "interventions < 1 %" gate is vacuous), clutter, and anything outside the
  ±2 cm / ±7.5° envelope. Those are the second increment.

## Deferred to the second increment

The swept-path safety shield with the α search (α is 1 everywhere now, so
the shield penalty is 0), clutter (level 8 raises `NotImplementedError`),
mesh-accurate proxy radii, PhysX self-collision.
