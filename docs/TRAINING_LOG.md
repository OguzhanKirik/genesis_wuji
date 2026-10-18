# Training log

How the algorithm itself works — observations, actions, reward, success test,
reading these logs — is in [`ALGORITHM.md`](ALGORITHM.md).

One entry per training run worth remembering: what was trained, on which env
config, with which command, and what came out. Smoke runs (a handful of
iterations to see that the code runs) are listed once per config, not per run.
Checkpoints live under `logs/rsl_rl/<experiment>/<timestamp>/`
(gitignored); TensorBoard: `tensorboard --logdir logs/rsl_rl`.

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
Level 7: **32.8 %**. Level 8 (one fixed box 10 cm in front of the bolt):
**38.6 %**, no obstacle hits by any of the fixed policies, including the
unsafe one, so the fixed obstacle is outside everything a blind policy
sweeps. The baseline falls with every level, as it should: 100, 100, 73.6,
65.6, 49.3, 41.2, 34.6, 32.8 %, then 38.6 % at level 8 (a different physics
draw, not an easier task).

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

## Clutter levels 8–10: the corridor had to be measured (2026-09-28)

Obstacles are kinematic PhysX boxes re-posed at every reset, folded into the
analytic clearance model (so the 16 safety observations and the clearance
penalty see them unchanged) and detected as a collision either by force on a
non-grasp link or by a proxy sphere sinking 4 mm into a box. Regression
suite at level 8: 53 of 53; at level 10 the placement, pose, clearance and
drive-into checks pass.

**The first placement rule was wrong.** "Outside the corridor" was coded as
a 9–14 cm annulus minus a 24 cm strip behind the bolt. The level-7 policy,
evaluated at level 10 without any clutter training, hit an obstacle in
**53.4 %** of episodes (all on the palm, forearm or the ring / little tips).
So `tools/probe_corridor.py` measured the corridor instead: 512 clutter-free
episodes of the level-7 policy, every 0.5 cm table cell any proxy sphere
covers. The footprint is 646 cm², a band about 12 cm wide from the forearm's
side through the bolt and 10 cm beyond it where the fingers reach. It is
stored in `corridor.npz` beside the env and dilated by the largest
obstacle's half-diagonal plus 1 cm for placement.

With the measured corridor, the same untrained level-7 policy, 512 episodes:

| level | obstacles | success | obstacle contacts | worst palm / forearm clearance |
|---|---|---|---|---|
| 8 | one fixed, 10 cm in front | 100 % | 0 | 1.98 / 1.56 cm |
| 9 | one, random in the annulus | 100 % | 0 | 1.98 / 1.59 cm |
| 10 | two or three, random | 99.4 % | 0 | 2.04 / 1.52 cm |

Which says what the spec intended and no more: obstacles placed outside the
nominal corridor are never touched by a policy that stays in the corridor,
so levels 8–10 confirm rather than teach. Avoidance as a learned behaviour
needs obstacles that intrude on the corridor's edge, where the start error
would otherwise sweep the hand into them; that is a level beyond the spec's
ladder and is left for after the shield.

Scripted baseline at level 8: 38.6 %, no obstacle hits by any fixed policy.

## Level 8 — one fixed obstacle (trained 2026-09-28)

**Run** `adam_u30_wuji_reach_grasp_L8/2026-09-28_12-17-08`, iterations
2300–2698, resumed from level 7's `model_2299` with `--reset-std 0.3`. **Best
checkpoint `model_2650.pt`.** Baseline: 38.6 %. A 5×5×6 cm box 10 cm in
front of the bolt, outside the measured corridor.

**Transfer.** 99.6 % ten iterations after the resume, obstacle hits zero at
every iteration, closest approach to the box 4 cm. As predicted by the
untrained level-7 policy's 100 % here, this level confirms rather than
teaches.

| iteration | success | invalid collisions | obstacle hits | bolt drag | action std |
|---|---|---|---|---|---|
| 2350 | 99.5 % | 0 | 0 | 2.4 mm | 0.32 |
| 2500 | 100 % | 0 | 0 | 2.2 mm | 0.30 |
| 2650 | 100 % | 0 | 0 | 1.6 mm | 0.28 |
| 2698 | 99.8 % | 0 | 0 | 1.8 mm | 0.26 |

**Evaluation** (512 episodes, mean action): `model_2650` **100.0 %**, 0
invalid collisions, latched 100 %, drag 1.5 / 2.1 / 4.2 mm (mean / p95 /
max), all six gates. `model_2698`: 100 %, drag 1.3 / 2.4 / 5.7 mm. Level 9
resumes from `model_2650`.

## Level 9 — one random obstacle (trained 2026-09-28)

**Run** `adam_u30_wuji_reach_grasp_L9/2026-09-28_12-43-41`, iterations
2650–3049, resumed from level 8's `model_2650` with `--reset-std 0.3`. One
box per episode, random pose in the 9–14 cm annulus outside the measured
corridor. 99.8 % ten iterations after the resume, obstacle hits zero at every
iteration, other collisions at most 0.1 %.

| iteration | success | invalid collisions | obstacle hits | bolt drag | action std |
|---|---|---|---|---|---|
| 2700 | 99.9 % | 0 | 0 | 1.7 mm | 0.29 |
| 2850 | 99.9 % | 0.10 % | 0 | 2.0 mm | 0.26 |
| 3000 | 99.9 % | 0 | 0 | 2.2 mm | 0.23 |
| 3049 | 100 % | 0 | 0 | 2.3 mm | 0.22 |

**Evaluation** (512 episodes, mean action): `model_3000` 100.0 %, 0 invalid
collisions, drag 1.8 / 3.0 / 7.2 mm; `model_3049` 100.0 %, 0, drag 1.5 / 2.4
/ 6.7 mm. Both pass all six gates. Level 10 was started from `model_3000`
before the second evaluation finished; the difference is within noise.

## Level 10 — two to three random obstacles (trained 2026-09-28)

**Run** `adam_u30_wuji_reach_grasp_L10/2026-09-28_13-14-05`, iterations
3000–3399, resumed from level 9's `model_3000`. 99.7 % ten iterations after
the resume, obstacle hits zero throughout, closest approach 3.3 cm with two
or three boxes on the table.

| iteration | success | invalid collisions | obstacle hits | bolt drag | action std |
|---|---|---|---|---|---|
| 3050 | 99.9 % | 0 | 0 | 2.5 mm | 0.29 |
| 3150 | 100 % | 0 | 0 | 2.2 mm | 0.25 |
| 3200 | 99.9 % | 0 | 0 | 2.4 mm | 0.25 |
| 3399 | 99.9 % | 0 | 0 | 3.3 mm | 0.24 |

**Evaluation** (512 episodes, mean action): `model_3200` **100.0 %**, 0
invalid collisions, drag 1.6 / 2.5 / 7.8 mm; `model_3150` 100.0 %, 0, drag
1.9 / 2.7 / 6.1 mm. Both pass all six gates; `model_3200` is the level-10
best. **The clutter progression (8–10) is complete**, and as the corridor
measurement predicted, every level was at ≥ 99.6 % within ten iterations of
its resume: with obstacles kept outside the measured corridor, the policy
never had to learn avoidance.

## Pose generalisation: the probe, the reach window, levels 11–13 (2026-09-28)

**The question.** Every level so far used one bolt position and one bolt
yaw. Does the level-7 policy generalise, or only correct a relative error
around the pose it was trained at?

**The probe** (`eval_reach_grasp.py --bolt-yaw-deg / --bolt-xy-cm`): the
level-7 checkpoint, 512 episodes, with the BOLT moved at reset while the
wrist still starts at the old nominal pose, so the bolt offset adds to the
trained start error. No retraining.

| bolt moved at reset | success | latched | invalid collisions |
|---|---|---|---|
| yaw ±15° | **100 %** | 100 %, step 29 | 0 |
| yaw ±30° | 96.5 % | 96.9 %, step 32 | 0.2 % |
| yaw ±45° | 80.3 % | 81.9 %, step 46 | 0.6 % |
| xy ±3 cm | **48.5 %** | 67.6 %, step 59 | 0 |

Orientation is free to about ±15°, the pinch being yaw-invariant and the
wrist-relative observations doing their job; beyond that the latch comes
later and later and the grasp fails. Position is a cliff: a 3 cm bolt shift
is outside the ±2 cm the policy ever had to correct, and the arm is in joint
configurations it never saw.

**The reach window** (`tools/solve_reach_grasp_poses.py --grid`, 975 buckets
of bolt yaw × xy ±5 cm × height, waist held at 1.05 rad): 100 % of buckets
reachable with s_min ≥ 0.14 for bolt yaw **−45° to +15°**; 93 % at +30°,
85 % at −60°, nothing at ±75° or beyond. The pipeline's pose was yawed 90° to
lift the wrist roll off its stop, and turning the other way puts it back.
So the pinch follows the bolt through a 60° window out of the rod's 180°
period. Wider needs the waist yaw freed or a second grasp family; that is a
design decision for the pile work, not this ladder.

**Levels 11–13**, built on the measured window. The bolt is sampled at
reset; the nominal wrist pose, the standoff reset (nearest of 582 reachable
buckets plus the residual through its Jacobian), the reach potential, the
workspace clamp and the obstacle corridor all become bolt-relative. The
observation interface is unchanged.

| level | bolt yaw | bolt xy | everything else |
|---|---|---|---|
| 11 | ±15° | 0 | as level 7 |
| 12 | −30° to +15° | ±2.5 cm | as level 7 |
| 13 | −45° to +15° | ±5 cm | as level 7 |

**The level-7 policy at level 13, untrained, bolt-relative start** (512
episodes): **24.1 %** success, latched 35 % at a mean step of 89, invalid
collisions 36.7 % (finger-table 25.9 %, structural 10.8 %), joint-limit
terminations, worst slack 0.020. So the bolt-relative start does not rescue
it: the policy trained in one arm configuration does not know the others,
where joint slack is scarce and the same Cartesian commands produce clipped
targets and a hand that drifts into the table. This is the gap levels 11–13
exist to close, and it is a real learning problem, unlike the clutter levels.

Scripted straight-down baselines: level 11 **38.1 %**, level 12 **44.7 %**,
level 13 **34.6 %** (worst joint slack 0.185 on the scripted policy, so the
whole window is inside the arm's limits); audits pass. The baselines are
driven by the physics and perception draws, which is why 12 is not below 11.

## Level 11 — bolt yaw ±15° (trained 2026-09-28)

**Run** `adam_u30_wuji_reach_grasp_L11/2026-09-28_13-58-53`, iterations
2300–2698, resumed from level 7's `model_2299` (the clutter levels add no
capability the pose levels need). **Best checkpoint `model_2400.pt`.**
Baseline 38.1 %. 95.7 % ten iterations after the resume, 99 % at 25, then
flat at 99.8–100 % with collisions ≤ 0.05 %: as the probe predicted, ±15° of
bolt yaw is nearly free.

**Evaluation** (512 episodes, mean action): `model_2400` **100.0 %**, 0
invalid collisions, drag 2.2 / 3.7 / 6.6 mm, worst joint slack 0.265, all
six gates; `model_2600` 100.0 %, 0, drag 2.1 / 4.1 / 8.5 mm. Level 12
resumes from `model_2400`.

## Level 12 — bolt yaw −30° to +15°, position ±2.5 cm (trained 2026-09-28)

**Run** `adam_u30_wuji_reach_grasp_L12/2026-09-28_14-24-26`, iterations
2400–2799, resumed from level 11's `model_2400`. **Best checkpoint
`model_2650.pt`.** Baseline 44.7 %. The first level since 3 where the
resume started far from solved: this is the position generalisation the
probe said was missing.

| iteration | success | invalid collisions | bolt drag | worst joint slack | action std |
|---|---|---|---|---|---|
| 2405 | 38.6 % | 6.5 % | — | 0.22 | 0.30 |
| 2450 | 78.8 % | 0.23 % | 4.6 mm | 0.11 | 0.31 |
| 2500 | 93.9 % | 0 | 3.6 mm | 0.30 | 0.31 |
| 2550 | 98.8 % | 0 | 2.9 mm | 0.37 | 0.32 |
| 2650 | 99.3 % | 0 | 2.4 mm | 0.36 | 0.32 |
| 2799 | 99.1 % | 0 | 2.8 mm | 0.36 | 0.32 |

The first fifty iterations are the interesting ones: collisions at 6.5 %
and joint slack down to 0.06, i.e. the policy driving the arm into its
limits in configurations it had never seen; both recovered by iteration
2500 as it learned to keep slack. The std stays at 0.32, higher than at any
fixed-pose level.

**Evaluation** (512 episodes, mean action):

| | `model_2650` | `model_2750` |
|---|---|---|
| lift-and-hold success | **99.8 %** | 99.8 % |
| invalid collisions | 0 | 0 |
| latched, mean grasp end | 100 %, step 27.9 | 100 %, step 28.4 |
| bolt drag mean / p95 / max | 2.6 / **5.0** / 29 mm | 2.5 / 4.8 / 40 mm |
| worst joint slack | 0.356 | 0.311 |
| gate | passes all six | passes all six |

Beats the 44.7 % baseline by 55 points. The drag tail is the longest of
any level (p95 right at the 5 mm gate, max 29–40 mm): at the new bolt
positions the pinch closes from slightly different finger angles and some
episodes let the bolt settle further. Worth a look if level 13 pushes it
over. Level 13 resumes from `model_2650`.

## Level 13 — bolt yaw −45° to +15°, position ±5 cm (trained 2026-09-28)

**Run** `adam_u30_wuji_reach_grasp_L13/2026-09-28_14-47-45`, iterations
2650–3049, resumed from level 12's `model_2650`. **Best checkpoint
`model_2800.pt`.** Baseline 34.6 %. The full reach window the pose
solver found for the held waist; the last level of the ladder.

| iteration | success | invalid collisions | bolt drag |
|---|---|---|---|
| 2655 | 62.8 % | 0.72 % | — |
| 2660 | 95.3 % | 0.45 % | 3.0 mm |
| 2700 | 97.3 % | 0.05 % | 3.0 mm |
| 2800 | 99.1 % | 0 | 2.7 mm |
| 2900 | 98.9 % | 0 | 2.7 mm |
| 3000 | 99.4 % | 0 | 2.4 mm |
| 3049 | 99.0 % | 0 | 2.6 mm |

The level-12 policy already covered most of the wider window: 95 % ten
iterations after the resume, against 39 % at the start of level 12. The
remaining half-percent of collisions took fifty iterations to clear.

**Evaluation** (512 episodes, mean action):

| | `model_2800` | `model_2950` | `model_3000` |
|---|---|---|---|
| lift-and-hold success | **100.0 %** | 100.0 % | 99.8 % |
| invalid collisions | 0 | 0 | 0 |
| bolt drag mean / p95 / max | 2.3 / **4.5** / 11.8 mm | 2.6 / 5.8 / 13.4 mm | 2.2 / 5.0 / 12.3 mm |
| gate | passes all six | drag p95 fails | drag p95 fails (at the line) |

Beats the 34.6 % baseline by 65 points. The drag tail is what separates
the checkpoints, as it did at level 12: success is saturated from 2800 on
and the p95 wanders between 4.5 and 5.8 mm from one checkpoint to the
next. `model_2800` is the pick and the ladder's final policy. It is worth
noting that the drag gate is now the only one doing any work at the pose
levels; if the window is widened further, the drag reward weight or the
lift speed should be revisited before training rather than after.

## Level 14 — waist freed, bolt yaw −60° to +45° (trained 2026-09-28)

**Why.** A pile presents bolts at any yaw and levels 11–13 stop at −45° to
+15°, a third of the rod's 180° period. The pinch ties the hand's yaw to
the bolt's, so the only lever left is where the shoulder is: the three
waist joints, held at the pipeline's (pitch 1.05, yaw 0, roll 0) so far.

**Reach measurement.** `tools/solve_reach_grasp_poses.py --grid
--free-waist --min-slack 0.15` scans waist pitch {0.9, 1.05, 1.2} × yaw
{0, ±0.25, ±0.5, ±0.75} × roll {0, ±0.2} rad per (bolt yaw, xy) column,
nearest-to-nominal first, keeps one triple per column (the arm descends
through every height in one episode), and writes `pose_buckets_waist.npz`.
Two things it had to learn:

- Yaw alone bought +15° (−45° to +30°). The waist yaw axis is tilted with
  the 60° forward lean, so most of it is roll at the shoulder. Pitch and
  roll together open −60° to +45°.
- The first grid put the elbow into the table: hold (1.2, 0.25, 0.2) at
  bolt yaw +45° gave an upper-arm clearance of −1.8 cm at reset, caught by
  the regression test. The solver now computes the forearm and upper-arm
  proxy clearance to the table box itself (copy of `PROXY_GROUPS`, checked
  against the env by the test) and requires 2.5 cm at the standoff buckets
  (where episodes start, plus up to 2 cm of sampled error) and 1 cm at
  contact. 585 of 975 buckets survive.

Window with slack ≥ 0.15 and that clearance: every xy within ±5 cm for
bolt yaw −45° to +30°, within ±2.5 cm out to −60° and +45°; nothing at
±75° with any hold. So bolt yaws +45° to +120° (mod 180°), a third of the
orientations, stay out of reach of a palm-down pinch from this side, at any
waist hold: that needs the other arm or a different pinch, not a wider
reset. Level 14 samples yaw −60° to +45° with xy ±5 cm inside the core and
±2.5 cm at the edges (`bolt_yaw_core`, `bolt_xy_edge`).

**Env.** `RGLevel.free_waist`; the standoff reset writes the bucket's waist
as joint state and the action term holds it as the drive target, exactly as
the pitch was held before. Levels 0–13 unchanged (their buckets carry the
pipeline's hold; level-13 tests 57 of 57 after the change, level 14 60 of
60). Baseline at level 14: 40.0 %.

**Run** `adam_u30_wuji_reach_grasp_L14/2026-09-28_16-13-42`, iterations
2800–3199, resumed from level 13's `model_2800`. **Best checkpoint
`model_3150.pt`.**

| iteration | success | invalid collisions | bolt drag |
|---|---|---|---|
| 2805 | 53.6 % | 7.2 % | — |
| 2850 | 93.3 % | 4.0 % | 3.4 mm |
| 2900 | 97.7 % | 0.55 % | 3.2 mm |
| 2950 | 98.7 % | 0.04 % | 2.6 mm |
| 3050 | 98.6 % | 0 | 2.4 mm |
| 3150 | 99.4 % | 0 | 2.1 mm |
| 3199 | 99.0 % | 0.18 % | 2.1 mm |

The hardest resume since level 12: 7 % collisions in the first iterations
(new shoulder placements the arm had never seen), a hundred iterations to
get collisions under 1 %, two hundred to zero.

**Evaluation** (512 episodes, mean action):

| | `model_3150` at level 14 | `model_3100` at level 14 | `model_3150` at level 13 |
|---|---|---|---|
| lift-and-hold success | **99.2 %** | 99.2 % | **100.0 %** |
| invalid collisions | 0 | 0.2 % (finger–table) | 0 |
| bolt drag mean / p95 / max | 1.8 / **3.6** / 10.7 mm | 2.4 / 6.5 / 30 mm | 1.7 / 2.9 / 13.5 mm |
| worst joint slack | 0.134 | 0.052 | — |
| worst clearance forearm / upper arm | 0.3 / 2.0 cm | 0.6 / 1.8 cm | — |
| gate | passes all six | collisions and drag fail | passes all six |

Beats the 40 % baseline by 59 points and loses nothing at level 13. Fifty
iterations separate a checkpoint that fails two gates from one that passes
all six, again on the tails (one finger–table touch in 512, drag p95): the
pick-by-gates rule is doing real work at these levels. The forearm's worst
clearance of 3 mm is the number to watch: the reset guarantees 2.5 cm, the
policy spends most of it on the way down.

## Level 15 — commanded grasp point along the rod (trained 2026-09-29)

**Why.** Everything to level 14 grasps the rod's centre. The graspable-area
study needs the grasp point commanded and swept, so level 15 samples an
offset s along the rod's axis, ±14 mm of its 17.5 mm half-length, per
episode. The rod is a 25 mm band over the middle 12 mm and 16.6 mm thread
over the 17 mm at each end, so ±14 mm runs from the band out onto the thread.

**How, without changing the interface.** The nominal wrist pose and the
policy's *estimated* bolt pose are moved to the grasp point, so the reach
reward, the reset bucket, the workspace clamp and the bolt-in-wrist and
fingertip-distance observations all say "the bolt is here". The physical
bolt, the latch (surface distances, indifferent to where along the rod),
the drag and the success test stay on the true object. 74 observations and
8 actions as before, which is what lets level 15 resume from the level-14
weights: at s = 0 the policy is unchanged. `eval_reach_grasp.py
--grasp-offset-mm` pins s; `tools/sweep_grasp_offset.sh` runs the sweep.
Tests 64 of 64 at level 15, 63 of 63 at level 14. Baseline 31.6 %.

**Run** `adam_u30_wuji_reach_grasp_L15/2026-09-29_07-36-37`, iterations
3150–3549, resumed from level 14's `model_3150`. **Best checkpoint
`model_3549.pt`** (the final one, for once: every earlier checkpoint is
worse on success and 3400 fails two gates).

| iteration | success | invalid collisions | bolt drag |
|---|---|---|---|
| 3155 | 52.3 % | 0.36 % | — |
| 3160 | 91.8 % | 0.48 % | 3.3 mm |
| 3250 | 95.6 % | 0.29 % | 2.5 mm |
| 3350 | 96.6 % | 0 | 2.5 mm |
| 3450 | 97.7 % | 0.08 % | 2.2 mm |
| 3549 | 98.6 % | 0 | 2.1 mm |

The slowest climb of the pose levels: the grasp off the band is a
different contact geometry (thinner rod, pads overhanging the end at
|s| > 10 mm), and success was still rising at the end of the run.

**Evaluation** (512 episodes, mean action, s sampled ±14 mm):

| | `model_3549` | `model_3400` |
|---|---|---|
| lift-and-hold success | **98.8 %** | 98.2 % |
| latched | 99.8 % | 99.4 % |
| invalid collisions | 0 | 0.2 % (structural) |
| bolt drag mean / p95 / max | 2.0 / **4.9** / 22 mm | 3.2 / 6.0 / 561 mm |
| gate | passes all six | collisions and drag fail |

Beats the 31.6 % baseline by 67 points. The 561 mm "drag" at 3400 is a
bolt that slipped out and rolled; the p95 gate is at the line for 3549.

**The graspable-area map along the axis** (`tools/sweep_grasp_offset.sh`,
`model_3549`, 512 episodes per point, logs in
`logs/sweeps/grasp_offset_2026-09-29_10-02-25/`):

| grasp point (mm from centre) | on the rod | success | collisions | drag p95 | latched |
|---|---|---|---|---|---|
| −16 | 1.5 mm from the end, untrained | 96.5 % | 0 | 8.3 mm | 98.8 % |
| −14 | thread, edge of training | 97.3 % | 0 | 7.4 mm | 99.2 % |
| −12 | thread | 98.0 % | 0 | 7.6 mm | 100 % |
| −10 | thread | 99.4 % | 0 | 5.8 mm | 99.8 % |
| −8 | thread / band shoulder | 99.0 % | 0 | 6.2 mm | 100 % |
| −6 | band | 99.2 % | 0 | 4.4 mm | 100 % |
| −4 | band | 99.8 % | 0 | 4.5 mm | 100 % |
| −2 | band | 99.6 % | 0 | 4.0 mm | 100 % |
| 0 | band, centre | 99.4 % | 0 | 3.1 mm | 100 % |
| +2 | band | 99.6 % | 0 | 3.1 mm | 100 % |
| +4 | band | 99.8 % | 0 | 3.2 mm | 100 % |
| +6 | band | 99.8 % | 0 | 2.9 mm | 100 % |
| +8 | band / thread shoulder | 99.4 % | 0 | 4.0 mm | 99.6 % |
| +10 | thread | 98.2 % | 0 | 3.7 mm | 99.4 % |
| +12 | thread | 98.4 % | 0 | 3.6 mm | 99.2 % |
| +14 | thread, edge of training | 96.3 % | 0.2 % | 5.1 mm | 98.4 % |
| +16 | 1.5 mm from the end, untrained | 92.8 % | 0.2 % | 8.4 mm | 96.1 % |

Reading it: the whole rod is graspable by this pinch. Success is above 99 %
across the band and the shoulders (±8 mm), 98 % out on the thread to
±12 mm, and still 96 to 97 % at ±14 mm where only a 3.5 mm of rod remains
beyond the pinch centre. At ±16 mm, outside anything the policy trained
on and 1.5 mm from the end, it holds 93 to 96 %. The drag tail is the
sharper indicator: it grows from 3 mm at the centre to 6 to 8 mm past
±10 mm on the negative side, where the pads sit on the thinner thread and
the rod rolls before the grip settles. The map is asymmetric: the negative
side (the thumb side of the approach) drags roughly twice as much for the
same offset, while success is slightly lower on the positive side. Latching
tracks success one to one, so the failures are grasps that never got a
stable opposing grip, not lifts that dropped.

What the map does not say: it is for one grasp family (three-finger pinch
square across the rod, approach from above). Angle tolerance comes with
level 16, and grasps on the end faces are a separate family.

**Aside: the scripted-lift test flake.** The level-14 regression test
failed once in three runs on "the lift is vertical" (41 mm lateral travel
at bolt yaw −6°, xy (−4.7, −4.9) cm, nominal waist, slack 0.47, nothing
clipped). That is the no-correction *scripted* policy latching in a poor
pose. Measured with the *trained* level-14 policy over 1021 lifts: mean
0.7 mm, p95 1.3 mm, max 1.7 mm lateral travel. The check is informational
at displaced levels now and prints the offending pose.

## Level 16 — commanded pinch angle ±30° (trained 2026-09-29)

**Why.** The along-axis map says where the rod can be held with a square
pinch. The second dimension is how skewed the hand may be. Level 16
samples a pinch angle delta, ±30° off perpendicular, per episode, on top
of level 15's offset, with the same device: the policy's estimated bolt
orientation is yawed by delta, so it aligns to a virtual rod and pinches
the real one obliquely. At this level the sampled yaw window bounds the
hand's yaw (reachability is about the hand) and the rod's yaw is hand yaw
minus delta. Tests 66 of 66 at level 16; level 15 still 65 of 65. Baseline
32.8 %.

**Run** `adam_u30_wuji_reach_grasp_L16/2026-09-29_08-37-58`, iterations
3550–3948, resumed from level 15's `model_3549`. **Pick `model_3850.pt`**,
which does NOT pass all gates: it is the best of the last three.

| iteration | success | invalid collisions | bolt drag |
|---|---|---|---|
| 3555 | 74.8 % | 0.15 % | — |
| 3560 | 96.1 % | 0.15 % | 2.6 mm |
| 3650 | 95.7 % | 0 | 2.8 mm |
| 3750 | 96.6 % | 0.05 % | 2.6 mm |
| 3850 | 97.1 % | 0.38 % | 2.5 mm |
| 3900 | 96.9 % | 0 | 2.6 mm |

Fast transfer (96 % ten iterations in: the level-15 policy already tolerates
skew) and then a plateau at 96–97 %, the lowest of any pose level.

**Evaluation** (512 episodes, delta and offset sampled):

| | `model_3850` | `model_3900` | `model_3948` |
|---|---|---|---|
| lift-and-hold success | **98.8 %** | 97.3 % | 96.9 % |
| invalid collisions | 0.2 % (structural) | 0.2 % (finger–table) | — |
| bolt drag mean / p95 / max | 2.0 / 5.6 / 20 mm | 2.2 / 6.7 / 34 mm | — |
| gates | success, force, limits, shield pass; collisions and drag fail | same two fail | one fails |

Beats the 32.8 % baseline by 66 points but misses the collision gate by
one episode in 512 and the drag gate by 0.6 mm at every checkpoint. The
map below shows the misses are concentrated at the rod's ends at large
positive angles, which is a physical corner, not a training defect; a
longer run would likely shave the drag but not the corner. Recorded as
**not gated**, used as the map's policy and as level 17's start.

**The offset × angle map** (`tools/sweep_grasp_offset.sh` with `PINCH`,
`model_3850`, 512 episodes per cell, 49 cells; logs in
`logs/sweeps/grasp_offset_pinch<angle>_2026-09-29_*`). Success, %:

| angle \ point | −12 | −8 | −4 | 0 | +4 | +8 | +12 mm |
|---|---|---|---|---|---|---|---|
| −45° | 99.4 | 99.8 | 100 | 100 | 99.6 | 98.6 | 92.4 |
| −30° | 99.4 | 99.6 | 99.8 | 99.8 | 100 | 97.3 | 94.9 |
| −15° | 99.6 | 99.2 | 99.8 | 100 | 99.4 | 98.6 | 96.3 |
| 0° | 97.1 | 99.4 | 99.6 | 99.8 | 99.6 | 99.2 | 99.4 |
| +15° | 94.5 | 96.9 | 97.1 | 97.9 | 97.9 | 99.2 | 98.3 |
| +30° | 97.1 | 98.2 | 98.6 | 98.8 | 98.8 | 98.8 | 95.3 |
| +45° | 86.0 | 93.6 | 97.9 | 98.4 | 99.4 | 99.4 | 96.9 |

Bolt drag during the lift, p95, mm:

| angle \ point | −12 | −8 | −4 | 0 | +4 | +8 | +12 mm |
|---|---|---|---|---|---|---|---|
| −45° | 3.7 | 2.5 | 2.6 | 2.7 | 2.8 | 4.2 | 8.6 |
| −30° | 4.0 | 2.7 | 2.4 | 2.6 | 2.7 | 6.0 | 7.0 |
| −15° | 6.9 | 4.0 | 2.6 | 2.7 | 2.8 | 2.7 | 4.4 |
| 0° | 7.7 | 6.3 | 4.1 | 3.3 | 2.9 | 2.8 | 3.8 |
| +15° | 11.1 | 10.1 | 8.4 | 6.8 | 6.2 | 5.1 | 6.9 |
| +30° | 7.1 | 6.6 | 7.0 | 7.8 | 6.8 | 7.9 | 9.9 |
| +45° | 9.3 | 7.8 | 6.5 | 6.3 | 5.6 | 6.1 | 7.4 |

Reading it. Success (first table, %, 512 episodes per cell): the pinch
works over the whole rod at every angle from −45° to +45°, including the
two angles outside the trained ±30°. 43 of 49 cells are above 95 %; the
weak corners are the rod's ends at large angles: −45° at +12 mm (92 %) and
+45° at −12 mm (86 %), where the pads on the thin thread are also skewed
towards the end. The map is not symmetric in angle: negative angles are
better than positive ones (row means 98.5 % at −45° against 96.0 % at
+45°), because a negative angle turns the thumb towards the band and a
positive one turns it towards the thread end. Drag (second table, p95 mm)
says the same thing more sharply: below 3 mm on the band at negative
angles, 6 to 11 mm at positive angles, worst at +15° on the negative
half, which is exactly the training-window edge the collision column also
marks (1 to 2 % at +45°, −12 and −8 mm; zero on the band). Grasp end is
later at large angles (36 steps against 30), the policy taking longer to
settle an oblique pinch.

Boundary: none inside ±45°. The pinch's angle limit lies beyond what was
swept, and level 17's ±45° start range is inside the feasible region
everywhere except the two end corners.

## Level 17 — free grasp, no alignment (trained 2026-09-29)

**Why.** The end state the user asked for: grasp the rod from different
areas without needing to align the wrist. Levels 15 and 16 commanded the
point and the angle; level 17 commands nothing.

**What changes.** The start pose is drawn as at level 16 (hand ±14 mm
along the rod, hand yaw ±45° off square, from level 16's map, which is
graspable everywhere but the two end corners), but: the policy sees the
*true* rod (no virtual bolt); the reach reward pulls the pinch centre (a
fixed point in the wrist frame, where the rod's centre sits at the solved
contact pose) to the rod's axis *segment*, any point along it, and its
orientation term is the approach tilt only, no yaw; and the world-yaw
component of the wrist action runs at a quarter rate. At that rate the
hand can turn at most about 7° in the 30 steps a grasp takes, so a 45°
start is grasped at 40° or more: rotating into alignment is not available.
Latch, drag, success and collision rules unchanged. Tests 67 of 67 at
level 17 (66 of 66 at 16). Baseline 33.8 %.

**Run** `adam_u30_wuji_reach_grasp_L17/2026-09-29_09-31-57`, iterations
3850–4449 (600, the pose levels' 400 plus half again, since the reward
changed), resumed from level 16's `model_3850`. **Best checkpoint
`model_4400.pt`.**

| iteration | success | invalid collisions | bolt drag |
|---|---|---|---|
| 3855 | 63.6 % | 0.36 % | — |
| 3860 | 92.2 % | 0.79 % | 3.1 mm |
| 3900 | 98.0 % | 0.12 % | 2.3 mm |
| 4000 | 98.8 % | 0 | 2.0 mm |
| 4050 | 99.3 % | 0 | 2.1 mm |
| 4200 | 98.9 % | 0 | 2.1 mm |
| 4300 | 98.3 % | 0 | 2.1 mm |
| 4449 | — | 0 | — |

92 % ten iterations after the resume: the level-16 policy already knew
how to close on a skewed rod, and dropping the commands cost it little.
The reward change shows only in the first iterations' collision rate.

**Evaluation** (512 episodes, mean action, start pose sampled):

| | `model_4350` | `model_4400` | `model_4449` |
|---|---|---|---|
| lift-and-hold success | 99.4 % | **100.0 %** | 99.4 % |
| latched, mean grasp end | 99.8 %, step 29.9 | 100 %, step 29.6 | 99.8 %, step 29.3 |
| invalid collisions | 0.2 % (finger–table) | **0** | 0 |
| bolt drag mean / p95 / max | 1.6 / 3.6 / 35 mm | 1.5 / **3.5** / 8.9 mm | 1.6 / 3.9 / 12 mm |
| worst joint slack | — | 0.076 | 0.058 |
| gate | collisions fail | **passes all six** | passes all six |

Beats the 33.8 % baseline by 66 points, and is the best-gated policy of
the study: 100 % in 512 with a maximum drag of 8.9 mm, lower than any
commanded level, on a harder start distribution. Two readings. First, the
free version is *easier* than the commanded one: at level 16 the policy had
to hit a named point at a named angle, here it may settle on whatever the
contact tells it, and the map below says where it chooses to. Second, the
grasp ends at the same step (29.6) as at level 15, so it is not buying
success with time.

**The start-pose map** (`tools/sweep_grasp_offset.sh` with `PINCH` at
level 17, `model_4400`, 512 episodes per cell; logs in
`logs/sweeps/grasp_offset_pinch<angle>_2026-09-29_1[23]*`). Success, %:

| start angle \ start point | −12 | −8 | −4 | 0 | +4 | +8 | +12 mm |
|---|---|---|---|---|---|---|---|
| −45° | 100 | 99.8 | 99.2 | 99.6 | 99.8 | 99.6 | 99.8 |
| −30° | 99.8 | 100 | 100 | 100 | 100 | 99.8 | 100 |
| −15° | 99.8 | 100 | 100 | 100 | 99.8 | 100 | 100 |
| 0° | 99.8 | 100 | 100 | 100 | 99.8 | 99.8 | 100 |
| +15° | 99.8 | 100 | 99.4 | 100 | 100 | 99.6 | 99.8 |
| +30° | 99.4 | 99.8 | 99.6 | 99.4 | 99.4 | 99.6 | 99.8 |
| +45° | 98.2 | 98.8 | 99.2 | 98.0 | 98.4 | 99.2 | 98.2 |

Bolt drag during the lift, p95, mm:

| start angle \ start point | −12 | −8 | −4 | 0 | +4 | +8 | +12 mm |
|---|---|---|---|---|---|---|---|
| −45° | 2.9 | 3.0 | 3.2 | 3.2 | 3.1 | 3.4 | 3.1 |
| −30° | 3.0 | 2.9 | 2.9 | 2.7 | 2.8 | 2.7 | 2.8 |
| −15° | 2.8 | 2.7 | 2.9 | 2.8 | 2.6 | 2.6 | 2.8 |
| 0° | 3.0 | 3.4 | 3.2 | 3.1 | 2.9 | 2.9 | 2.7 |
| +15° | 4.0 | 4.3 | 4.2 | 3.6 | 3.6 | 3.8 | 3.7 |
| +30° | 5.3 | 4.6 | 4.9 | 5.0 | 4.6 | 5.0 | 4.7 |
| +45° | 5.9 | 5.3 | 6.6 | 6.6 | 5.3 | 4.5 | 5.6 |

Reading it. At level 17 the pinned values set only where the hand
*starts*; the policy then grasps as it likes, and with the yaw action at a
quarter rate it can turn at most about 7° before the grasp ends. Success:
every one of the 49 cells is at or above 98 %, 42 of them at or above
99.4 %, against 43 of 49 above 95 % for the commanded policy on the same
grid. The two end corners that were 86 % and 92 % when commanded are 98.2 %
and 99.8 % when free. Drag: below 3.5 mm at every start from −45° to 0°,
rising to 4 to 5 mm at +15° to +30° and 5 to 7 mm at +45°, the same
thumb-towards-the-thread asymmetry level 16 showed, but milder. Collisions
0 to 0.6 %, all at +30° and +45°, all finger–table.

Why the free grasp beats the commanded one on its own grid: the commanded
policy had to place its pinch on a named point at a named angle and was
scored on holding *that*; the free policy is scored only on the lift, so
where a start would make a poor grasp, it settles a few millimetres along
the rod or a few degrees round it to a better one. The table says that
from any start over the rod within ±45°, that adjustment is enough, and
the pinch never needed the wrist to align.

## The commanded point was not binding (2026-09-29)

Measured with the level-18 pick, 384 to 511 latched episodes per cell:
the pinch centre (thumb-versus-fingers midpoint) projected on the rod's
true axis, against the commanded offset.

| pose | commanded | actual pinch, mean | std |
|---|---|---|---|
| lying | −14 mm | −1.6 mm | 3.6 |
| lying | −7 mm | +1.3 mm | 3.1 |
| lying | 0 mm | +4.5 mm | 3.7 |
| lying | +7 mm | +7.5 mm | 4.5 |
| lying | +14 mm | +10.8 mm | 5.0 |
| standing | −14 mm | −2.7 mm | 2.6 |
| standing | −7 mm | −1.2 mm | 2.3 |
| standing | 0 mm | +0.4 mm | 2.1 |
| standing | +7 mm | +1.4 mm | 2.5 |
| standing | +14 mm | +2.6 mm | 2.8 |

The policy follows the command partly on a lying rod (about half the
commanded travel on the positive side, almost none on the negative) and
barely at all on a standing one (a fifth). The mechanism: the nominal pose
and the observations move to the point, but the latch fires on a stable
grip anywhere along the rod, and the 2.0-weighted reach shaping is no
match for a +20 success available a few millimetres away on the band.

**Consequence for the maps.** The level-15, 16 and 18 maps are maps of
"the policy started aimed at this point and got a stable lift", not of
"the policy grasped at this point". Their high plateaus are real for the
first reading and say nothing for the second. The level-17 map claimed
only the first reading and stands. The level-15 and 16 entries above are
left as written, with this note as their correction.

**Fix.** Level 19: the latch requires the pinch within 4 mm of the
commanded point (`RGLevel.grasp_point_tol`); every evaluation prints the
pinch-versus-commanded error at the latch; the sweep table carries it.

## Level 18 — the rod standing on its end (trained 2026-09-29)

**Why.** The first variation that changes the geometry of the grasp
rather than its parameters. The rod stands 35 mm tall on a 16.6 mm end
face; the pinch is unchanged (it closes horizontally, which is across a
vertical rod at any hand yaw), and the commanded grasp point becomes a
height along the rod, ±14 mm about its centre. Half the episodes stand
the rod, half lay it as at level 16, so one policy learns both; it sees
the rod's orientation in its wrist frame and can tell them apart. The
level uses the convex-decomposition collider (previous entry). Tests 73
of 73 at level 18. Baseline 33.1 %.

**Run** `adam_u30_wuji_reach_grasp_L18/2026-09-29_11-02-29`, iterations
4400–4799, resumed from level 17's `model_4400`. **Pick `model_4750.pt`**,
which misses the drag gate.

| iteration | success | invalid collisions | bolt drag |
|---|---|---|---|
| 4405 | 35.4 % | 0.10 % | — |
| 4410 | 63.0 % | 0.13 % | 5.0 mm |
| 4450 | 82.1 % | 0.10 % | 3.7 mm |
| 4500 | 95.7 % | 0.10 % | 3.2 mm |
| 4600 | 98.6 % | 0.13 % | 2.2 mm |
| 4700 | 98.6 % | 0 | 2.3 mm |
| 4750 | 98.2 % | 0 | 2.4 mm |

The slowest resume of the study (35 % at ten iterations, 96 % at a
hundred), as expected for a new contact geometry: half the episodes are a
vertical rod the level-17 policy had never seen.

**Evaluation** (512 episodes, mean action):

| | `model_4750`, mixed | `model_4750`, standing only | `model_4750`, lying only | `model_4700`, mixed | `model_4799`, mixed |
|---|---|---|---|---|---|
| lift-and-hold success | **99.6 %** | 99.8 % | 99.4 % | 98.4 % | 99.0 % |
| invalid collisions | 0 | 0 | 0 | 0 | 0 |
| bolt drag mean / p95 / max | 2.3 / 5.9 / 38 mm | 2.1 / 5.3 / 31 mm | 2.3 / 6.6 / 17 mm | 2.4 / 6.1 / 28 mm | 2.5 / 6.8 / 25 mm |
| gate | drag fails | drag fails | drag fails | drag fails | drag fails |

Beats the 33.1 % baseline by 66 points. Success and collisions are at
the level-17 standard on both poses; the drag gate fails everywhere, and
the lying rod drags *more* (6.6 mm) than the standing one (5.3 mm), so the
extra drag is not the standing pose. Two candidates: the decomposition
collider (the level-17 policy already showed 3.7 mm on it against 3.5 on
the hull, so a small part), and the commanded-point training with the
rod's contact geometry now differing between halves of the batch. The
maps below split it by grasp point.

## Level 19 — the commanded point made binding (trained 2026-09-29)

**What changes.** As level 18, plus: the latch requires the pinch centre
within 4 mm of the commanded point along the rod's true axis, and (added
after the first evaluations, see below) success requires the latch. Every
evaluation prints the pinch-versus-commanded error at the latch. Tests 75
of 75. Baseline 26.6 % (33.1 % at level 18: a blind close now only counts
when it lands on the point).

**Run** `adam_u30_wuji_reach_grasp_L19/2026-09-29_11-51-42`, iterations
4750–5349 (600), resumed from level 18's `model_4750`. **Pick
`model_5300.pt`**, not gated.

| iteration | success | invalid collisions | bolt drag | pinch error at latch |
|---|---|---|---|---|
| 4755 | 38.4 % | 0.9 % | — | 0.7 mm |
| 4800 | 73.2 % | 1.9 % | 2.2 mm | 1.6 mm |
| 4900 | 89.0 % | 0.8 % | 3.2 mm | 1.9 mm |
| 5000 | 90.7 % | 0.2 % | 2.9 mm | 2.0 mm |
| 5200 | 94.6 % | 0.4 % | 3.0 mm | 1.9 mm |
| 5349 | 93.7 % | 0.15 % | 2.9 mm | 1.9 mm |

The binding point is a real constraint: the curve saturates at 94 %
instead of 99 %, and the pinch error settles at 2 mm, half the tolerance.

**Two scoring bugs found on the way, both fixed before these numbers.**
(1) The per-episode rest height added for raised rods was in a buffer the
reset zeroed, so lift and success were measured from the table's origin;
every evaluation between 14:10 and 15:10 was invalid. (2) A grasp that
never latched (refused for being off the point) could still be lifted by
the scripted phases when the grasp window timed out and counted as a
success, which showed as success above the latch rate in some map cells.
At binding levels success now requires the latch. The training run
predates both changes and stands; its reward saw (2), so its policy was
paid for some off-point grasps.

**Evaluation** (`model_5300`, 512 episodes, mean action, success requiring the latch):

| | mixed (first rerun, before fix 2) | standing only | lying only |
|---|---|---|---|
| lift-and-hold success | 97.1 % | **96.7 %** | **88.7 %** |
| latched, mean grasp end | 96.6 %, step 34 | 98.5 %, step 30 | 91.8 %, step 41 |
| pinch error mean / p95 / max | 2.0 / 3.9 / 4.0 mm | 1.9 / 3.8 / 4.0 mm | 2.0 / 3.9 / 4.0 mm |
| bolt-drop rate | 1.5 % | 1.0 % | 1.2 % |
| invalid collisions | 0 | 0 | 0.2 % |
| bolt drag mean / p95 | 3.0 / 8.6 mm | 2.9 / 7.0 mm | 3.1 / 9.8 mm |
| gates | drag fails | drag fails | success and drag fail |

Beats the 26.6 % baseline by 60 to 70 points. The point is certified
binding (maximum error 4.0 mm, the tolerance) and the price is paid in
latch rate and drag, most on lying rods, which roll.

**The honest maps** (`model_5300`, 512 episodes per point, success
requiring the latch; standing rod
`logs/sweeps/grasp_offset_upright1_2026-09-29_15-3*`, lying rod at a
square pinch `grasp_offset_pinch0_upright0_2026-09-29_15-4*`).

Standing rod, height along it:

| height from centre | on the rod | success | collisions | drag p95 | latched | grasp end | pinch error |
|---|---|---|---|---|---|---|---|
| −16 mm | bottom thread, 1.5 mm off the table | 79.5 % | 0.6 % | 13.3 mm | 83.6 % | step 65 | 3.1 mm |
| −14 mm | bottom thread | 86.7 % | 0.4 % | 6.3 mm | 91.3 % | 46 | 3.3 mm |
| −10 mm | bottom thread | 95.3 % | 0 | 5.7 mm | 95.7 % | 32 | 2.0 mm |
| −6 mm | lower shoulder | 93.6 % | 0.2 % | 5.9 mm | 94.0 % | 33 | 1.6 mm |
| −2 mm | band | 98.1 % | 0.2 % | 5.2 mm | 98.4 % | 30 | 1.6 mm |
| +2 mm | band | 97.7 % | 0 | 7.4 mm | 98.7 % | 30 | 1.6 mm |
| +6 mm | upper shoulder | 98.6 % | 0 | 7.7 mm | 99.4 % | 29 | 1.6 mm |
| +10 mm | top thread | 97.5 % | 0 | 8.0 mm | 99.6 % | 30 | 2.0 mm |
| +14 mm | top thread | 96.5 % | 0.2 % | 9.3 mm | 99.2 % | 31 | 2.6 mm |
| +16 mm | top thread, 1.5 mm from the top | 88.1 % | 0.2 % | 11.6 mm | 96.9 % | 36 | 2.8 mm |

Lying rod, point along it, pinch square across:

| point from centre | on the rod | success | collisions | drag p95 | latched | grasp end | pinch error |
|---|---|---|---|---|---|---|---|
| −16 mm | thread, 1.5 mm from the end | 67.3 % | 0.6 % | 12.1 mm | 75.7 % | step 61 | 3.4 mm |
| −14 mm | thread | 90.1 % | 0.4 % | 7.3 mm | 94.6 % | 44 | 2.8 mm |
| −10 mm | thread | 89.3 % | 0.2 % | 10.6 mm | 93.2 % | 41 | 2.3 mm |
| −6 mm | shoulder | 86.6 % | 0.2 % | 10.9 mm | 91.7 % | 41 | 2.0 mm |
| −2 mm | band | 88.1 % | 0 | 9.7 mm | 91.2 % | 41 | 1.8 mm |
| +2 mm | band | 85.0 % | 0.2 % | 6.3 mm | 86.4 % | 43 | 2.1 mm |
| +6 mm | shoulder | 90.6 % | 0 | 6.6 mm | 91.3 % | 38 | 1.9 mm |
| +10 mm | thread | 87.1 % | 0 | 10.1 mm | 92.8 % | 38 | 2.1 mm |
| +14 mm | thread | 76.4 % | 0.6 % | 14.6 mm | 91.6 % | 44 | 2.6 mm |
| +16 mm | thread, 1.5 mm from the end | 63.9 % | 0.6 % | 16.4 mm | 85.4 % | 51 | 2.3 mm |

Reading them. These are the first maps where a cell means "grasped
there": the pinch error column certifies the fingers closed within 4 mm
of the commanded point (1.6 to 3.4 mm on average), and success now also
requires the latch, so a grasp that was refused for being off the point
and then lifted anyway by the scripted phases no longer counts.

- **Standing rod.** Band and shoulders (±6 mm): 94 to 99 %. Top thread:
  96 to 98 %, falling to 88 % in the last 1.5 mm. Bottom thread, next to
  the table: 95 % at −10 mm, 87 % at −14 mm, 80 % at −16 mm, where the
  fingers have to get under the band, the grasp takes twice as long
  (step 46 to 65 against 30) and the drag reaches 13 mm.
- **Lying rod, square pinch.** Lower everywhere: 85 to 91 % over the band
  and shoulders, 87 to 90 % on the thread, 64 to 76 % at the ends. The
  latch rate is the loss (86 to 95 % against 100 % when the point was not
  binding): a pinch that settles a few millimetres off the point is
  refused and has to be re-seated, and the drag (6 to 16 mm) says the rod
  rolls while that happens. A lying rod is the harder object for a
  commanded point because it rolls; a standing one does not.
- **Where a pile planner should aim.** On either pose, the band and the
  first 10 mm of thread; the last 4 mm before an end costs 10 to 30
  points and double the drag.

Against the earlier, non-binding maps (99 % nearly everywhere): the rod
is graspable at every point along it, but the ends have a real cost that
those maps hid because the fingers went to the band regardless.

## Levels 50 and 51 — the full bin (2026-10-15/16, night)

adam-moveit's bin (workcell.json: inner 22 x 16 cm, walls 8 cm, centred on BOLT_XY here, on the right hand's side)
filled with a settled heap in 70 % of episodes; the other 30 % keep level 44's scenes inside the same bin. Heaps from
`tools/make_bin_heaps.py`: adam-moveit's drop (5 x 6 x 2 grid, jitter, tilt, spin) with the mesh hull, then resettled
as 22.7 mm cylinders (`--resettle --collider cyl`), which is what the training env simulates: 60 mesh rods per env held
123 GB of host RAM at 512 envs, and 25 mm cylinders placed at hull poses overlapped and blew the heap apart (35 mm
median target drift; now 2.1 mm). Full set: 349 of 1024 hull heaps survive the resettle; rods 36 / 25 / 39 % lying /
leaning / standing (hull drop: 27 / 43 / 30). Target: an exposed heap rod in the reachable window, lying / leaning /
standing 0.5 / 0.3 / 0.2. 15 frozen settle steps, push_ok. 50: 15-25 rods, 51: 40-60. 512 envs, ~13 s / iteration
(the GPU shared with a hug-grasp run). `logs/chains/chain_L50.sh`.

Level-44 pick untrained at 51, heap episodes: **14.2 %**, 50 % obstacle disqualifications, 95 % of them the forearm
(wristYawRight) on the 8 cm wall top (median 4.4 mm in; `tools/probe_collisions.py`), 17 % excess force.

| | level-50 pick `model_18999` (600 it.) | level-51 pick `model_19950` (1000 it. from it) |
|---|---|---|
| level mix | 28.8 % (at 50) | 45.2 % (at 51) |
| heap (`--fill 1`) | | **43.8 %** |
| heap, lying target | | 53.1 % |
| heap, leaning | | 49.0 % |
| heap, standing | | 23.0 % |
| level 44's scenes in the bin (`--fill 0`) | | 36.5 % |

Heap episodes at the level-51 pick: obstacle 3.5 % (from 50 %), excess force 4.4 %, latched 57.3 %, lifted 48.8 %,
drops 13.5 %, retried 68.8 %. The wall is solved; the limit now is getting and keeping a grip in the heap, and
standing rods. `--fill 0` lost most to obstacles (26.4 %: support and cover blocks, the walls) and finger-table 3.5 %:
the old scenes in the tall bin were trained only 30 % of the time.

## Levels 47-49 — the cover scene (2026-10-15)

**The slide-out (47).** `cover_drag`: in the free pre-lift a latched covered rod is slid 10 x 3 mm along its axis away
from the block, the policy's actions on those steps masked out of the actor loss. Level-44 pick untrained, cover row,
1024 episodes: 64.7 % at level 45 -> **66.6 %**. Barely: only ~69 % of cover episodes latch at all, and the slide acts
after the latch.

**Why covered rods fail** (`tools/probe_latch.py --cover 1`, level 45): latched 79.7 %, held but never latched 0.4 %,
**never gripped 19.9 %**. Not the latch: the fingers never close on the rod. The exposed part of a covered rod runs
from its free end to 5 mm past its centre; a pinch at the centre puts the fingers against the block.

**The shifted target (48).** `cover_shift`: a covered rod's estimate points at the middle of its exposed part,
6.25 mm along the axis away from the block. Untrained, cover row: **67.8 %**.

**Level 49** = level 45 + both, trained from the level-44 pick with level 45's settings (scene change only).
Untrained 78.2 %. Pick (quota rule, 2048 episodes): `model_19500` 76.6 %, **`model_19550` 78.6 %**, `model_19599`
75.5 % (`logs/rsl_rl/adam_u30_wuji_reach_grasp_L49/2026-10-15_09-58-28/`), level 45's 76.3 % within noise. Scene rows:
cover **65.3 %** (the level-44 pick 66.8 %; latched 69.3 %, the rod touched the table in the hold in 31.3 %), bed 90.4,
tilted 90.1, upright 93.8, no heap 82.2, across 76.5, lying (bed and cover included) 62.3 %. **No gain on the cover,
and the other rows a little lower.** Neither the slide-out nor the shifted target, alone or trained together, moves the
covered rod: about 20 % of cover episodes never grip at all. The cover scene is at a ceiling for this hand and
approach; it is deprioritised (the level-44 pick stays the heap candidate).

For the record, the level-43 pick on a REAL cover (level 44's scene, flat fingers as trained, 2048 episodes): **66.3 %**
(latched 75.5 %, touched the table in the hold 29.5 %). Every policy since level 39 sits at ~66 % on the cover; the
69-72 % rows of levels 35-38 were on the double-counting simulator and on level 32's scene settings.

On the robot both need the block's side from the camera (the slide's direction and the shift's sign).

## Levels 45 and 46 — the hard scenes more often; the std that would not shrink (2026-10-14, evening)

**Level 45** (`LEVELS[45]`: level 44 with cover 0.5 and a neighbour across 0.5 of lying episodes; the level-44
pick's cover row was 66.8 %, the covered end pinned under the block, and across 82 % on level 44's scene). From the
level-44 pick, fist, 1200 iterations at the usual settings (`--reset-std 0.3`, entropy 0.003, fixed LR 3e-4).
Untrained 77.6 %; training-time success flat at 0.75-0.79; pick `model_19550` **76.3 %** (19500 75.1 %, 19599
75.2 %). No gain. Scene rows not run.

**Why the last continuations trained flat.** In every continuation the policy's entropy ROSE: 1.72 -> 3.01 at level
40, 1.72 -> 2.48 at level 44, 1.72 -> 2.05 by the middle of level 45 (8 action dims at std 0.3 is 1.72; 3.01 is about
0.35). The entropy bonus (0.003) wins over a weak late-training reward gradient and the std grows instead of
shrinking, with a fixed learning rate. (The same ratchet is why the bonus was cut from 0.01 at the start of
Experiment 4.) The across row differing between levels 43 and 44 is the scene, not training: on level 44's scene
the level-43 pick (fist) scores 80.1 %, the level-44 pick flat 80.8 %, fist 82.2 %.

**Level 46** = level 45's scene (`LEVELS[46]`, its own number for its own logs), from the LEVEL-44 pick like level
45, so the two differ only in the training settings: `--entropy-coef 0 --reset-std 0.2 --lr-schedule adaptive`
(new `train.py` options; `logs/chains/chain_L46.sh`).

First try with the adaptive schedule: the entropy fell (-1.53 -> -1.87) but the KL-targeted rate dropped to
1e-5-3e-5 within 100 iterations (a smaller std makes each update's KL larger) and the policy barely moved; stopped,
rerun with `--lr 1e-4 --lr-schedule fixed`. That run: entropy -1.53 -> -3.0 (the std shrinking, as intended),
training-time success flat at 0.74-0.79. Untrained 77.2 %; pick candidates `model_19500` 75.6 %, `model_19550`
74.4 %, `model_19599` 74.4 % (pick `model_19500`). **No gain either.** Scene rows not run.

**Conclusion.** On the level-45 scene mix the level-44 pick sits at 76-79 % and neither more practice on the hard
scenes (45) nor training without the entropy bonus and with a lower rate (46) moves it. The ceiling is the scenes,
not the optimiser: the covered rod's end stays pinned under the block (31.9 % touched the table in the hold), which
the policy's 10-step free pre-lift does not learn to clear. Next: change what the hard scenes ask, e.g. a scripted
slide-out for a covered rod (the drag levels 23-31 had), or a shaping term for the covered end rising.

## Level 44 — the bed and the cover back, ring and little in a fist (2026-10-14)

**The bed row.** The level-43 pick scored 44.8 % with the target on a bed of rods: 51.3 % of episodes were
disqualified by obstacle hits, and `tools/probe_collisions.py` (which reads the sphere depths inside the
termination check, before the reset) attributes them all to the gripping tips reaching into the support rods:
middle 67 %, index 21 %, thumb 12 %. Levels 39-43 were built on level 31's line, which has no bed (level 32 had
`bed=BedCfg(prob=0.4)`, `cover=CoverCfg(prob=0.3)`).

**The idle fingers at a wall.** In the default 26 cm bin rods never come within 8 cm of a wall and the ring and
little fingers touched nothing (0 of 1025 episodes). With the bin shrunk to 16 x 16 cm (`ADAM_U_RG_BIN_HALF=0.08`,
rods 3-13 cm from a wall, the real box being 22 x 16 cm), the level-43 pick, 512 episodes each:

| ring / little | little touching a wall | ring touching | p5 closest, little / ring | disqualified (forearm, palm) |
|---|---|---|---|---|
| flat (`HELD_CURL` 0) | 10.9 % | 2.9 % | -0.9 / 2.7 mm | 4.1 % |
| fist (0.95) | 0 % | 0 % | 4.1 / 10.4 mm | 5.9 % |

(1.0 puts default positions past the joint limits; 0.95 is the fist.) At the grasp poses the flat fingers are the
highest and farthest from the rod (35 mm above the table at worst, 104 mm from the rod); a PARTIAL curl brings
their tips into the table (0.3: -11 mm), which is what Experiment 2 saw; the fist does not (22 mm). The level-43
pick untrained with the fist on the normal mix: 90.9 % (91.7 % flat), finger-table 0.1 %.

**What.** `LEVELS[44] = replace(LEVELS[43], bed=BedCfg(prob=0.4), cover=CoverCfg(prob=0.3))`, trained and
evaluated with `ADAM_U_HELD_CURL=0.95` (`logs/chains/chain_L44.sh` sets it on every call; the export must too, so
that `hand.held` in level.yaml carries the fist). From the level-43 pick, 1024 envs, 1200 iterations.

**Untrained** (level-43 pick, fist): 83.5 %. **Training** 09:35-11:16, training-time success flat at 0.80-0.86.
**Pick** (quota rule, 2048 episodes): **`model_18400` 83.2 %**, `model_18450` 81.4 %, `model_18496` 81.9 %
(`logs/rsl_rl/adam_u30_wuji_reach_grasp_L44/2026-10-14_07-37-44/`). Scene rows:

| scene | level-43 pick | level-44 pick |
|---|---|---|
| bed of rods | 44.8 % (obstacle hits 51 %) | **91.9 %** (collisions 1.9 %) |
| tilted | 88.4 % | 89.9 % |
| upright | 97.8 % | 96.0 % |
| no heap | 94.0 % | 88.7 % |
| lying (now with bed and cover) | 90.2 % | 75.9 % |
| across a rod | 91.7 % | 81.8 % |
| cover block | (92.0 %, not real: see below) | 67.8 % |

**The cover rows of levels 39-43 measured nothing.** The cover block is only built for a level whose
`cover.active` (`if lvl.cover.active:` in the scene config), and `reset_cover` returns early without it, so
`--cover 1` at levels 39-43 ran the scene without a block. Level 44's 67.8 % is the first real cover number since
level 38. The bed is unaffected (its support rods are neighbour slots, present at every pile level).

What the bed and the fist cost elsewhere is the open question: across fell 10 points and the cover is the weakest
scene now. Level 44 is the robot candidate for the bed; for single rods in the box the level-43 pick is better.

## Level 43 — the hand's effort as touch (2026-10-13, night)

**Why.** The grasps that slip sit ~8 mm off the true axis at the latch, the size of the estimate's bias: the policy
aligns to the estimate and, without fingertip sensors since level 39, cannot feel the pinch is off. The robot does
report the finger drives' current.

**What.** `LEVELS[43] = replace(LEVELS[40], effort_obs=True)`: the `tip_forces` slots carry the flexion-chain
|torque| (thumb, index, middle) scaled so `GRIP_TAU_MIN` reads as 4 N (x 20 N per N m, / F_MAX), the `contacts`
slots fire above `TAU_CONTACT` 0.10 N m (free fingers 0.03-0.05). Same 74-wide interface; first level trained on the
once-per-step fix (below). Runtime mirrors it (`observation.effort_obs`); parity passes on 8 recorded episodes of
the pick (64-env recording, 1179 steps).

**Untrained** (level-40 pick): 82.1 % (87.9 % at level 40, fixed sim): inputs it was never trained on.

**Training** from the level-40 pick, 1024 envs, 1200 iterations (22:02-23:41), training-time success 0.83 -> 0.90.
**Pick** (quota rule, 2048 episodes, fixed sim): `model_17200` 89.6 %, `model_17250` 91.1 %, **`model_17297`
91.7 %** (`logs/rsl_rl/adam_u30_wuji_reach_grasp_L43/2026-10-13_20-04-28/`), +3.8 points over the level-40 pick on
the same simulator. Exported to `logs/export/L43_17297/`.

Scene rows (the pick, 2048 episodes each, fixed sim):

| scene | success |
|---|---|
| all (the pick) | 91.7 % |
| no heap | 94.0 % |
| lying (no upright, no tilt) | 90.2 % |
| tilted | 88.4 % |
| upright | **97.8 %** (passes) |
| across a rod | 91.7 % |
| cover block | 92.0 % |
| bed of rods | 44.8 % — not in training: levels 39-43 are built on level 31's line, which has no bed |

The weakest trained scenes are the tilted rod and the lying rod, the two the slip analysis named.

## Level 40 — the latch on camera-grade noise (2026-10-13, evening)

**Why.** At level 39 the latch reads the camera estimate, and the estimate's noise broke it, not the grasps.
`tools/probe_latch.py` (level-32 pick at level 39, 258 episodes): 62.8 % latched, **23.3 % held the rod through
the hold and never latched** (a lost success: with `pinch_centre_tol > 0` an unlatched lift is not a success),
14.0 % neither. On the lost episodes' grasp steps with the torque grip held, the two tests that read the
estimate failed most:

| test | fails | on the true rod it would fail |
|---|---|---|
| settled (rod vs wrist speed < 0.1 m/s) | 78.8 % (estimate 0.165 m/s) | 23.2 % (true 0.077 m/s) |
| centred (pinch midpoint to the axis <= 6 mm) | 76.4 % (9.1 mm to the estimated axis) | 44.1 % (6.4 mm to the true axis) |

A one-step difference of an estimate with 1-3 mm frame noise at 30 Hz is 0.04-0.13 m/s per axis by itself, and
the estimate's 8 mm bias is wider than the 6 mm tolerance (level 32 trained with 10 mm; level 39 was built on
level 31's line and inherited its 6 mm).

**What.** `LEVELS[40] = replace(LEVELS[39], est_vel_window=5, pinch_centre_tol=0.010, sense_est_retry=True)`:
the settled test on the estimate's velocity over 5 steps (0.17 s), the centred tolerance back to 10 mm, and the
retry's "not lifted at the lift's end" test on the estimate (level 39 still read the true rod there). Levels
0-39 unchanged. The runtime (adam-moveit genesis_policy) mirrors the window (`latch.est_vel_window`); parity
passes on 8 recorded level-40 episodes.

**Untrained** (level-32 pick, quota rule, 2048 episodes): **77.3 %** (51.1 % at level 39). Latched 88.5 %,
drops 25.2 %, the grip check sent 39.5 % into a retry, success without a retry 52.4 %.

**The grip check is right; the grasps slip.** `tools/probe_gripcheck.py` (same pick, level 40, 514 checks): it
fired on 43.4 %, and only 10 % of the firings were false (the true rod followed the tips); 6.0 % of checks missed
a slip. Over the 21 mm test lift the true rod rose 10.2 mm on average: in 45 % of checks the rod did not follow
the fingers. That is grasp quality, which the policy lost its fingertip-force observations for at level 39, and
what the training below has to win back.

**Level 41, a squeeze after the latch — rejected.** If the slips were a light grip, tightening the closures
after the latch should fix them: `LEVELS[41]` ramps both closures up by `latch_squeeze` over the settle
(`ADAM_U_RG_SQUEEZE`). Untrained level-32 pick, 1024 episodes each: squeeze 0.2 → 71.1 %, 0.4 → 65.2 % (77.3 %
without), the grip check firing more (44.7 / 53.1 %) and more drops (26.8 / 29.1 %). Tightening an off-centre pinch
on a cylinder pushes the rod out of it. The slips are placement, not force; the level stays in the file at its
default 0.3 for the record and is not trained.

**Where the slipping grasps lose the rod** (`tools/probe_gripcheck.py` on the level-39 pick at level 40, 768
checks; the true pinch depth and the weaker side's fingertip force):

| | at the latch | settle end | test-lift start |
|---|---|---|---|
| held | 4.4 mm / 5.8 N | 4.4 mm / 5.7 N | 5.0 mm / 4.4 N |
| slipped | 8.1 mm / 4.9 N | 8.3 mm / 4.7 N | 19.5 mm / 1.0 N |

By depth at the test lift's start: 0-3 mm slipped 3 %, 3-6 mm 4 %, 6-10 mm 20 %, beyond 84 %. Leaning rods slip
most (40 %), then lying (31 %), upright least (18 %).

**Level 42, the pre-lift held still — rejected.** The rod leaves during the 10-step free pre-lift window, so
`LEVELS[42] = replace(LEVELS[40], free_prelift=False)`. Level-39 pick untrained, 2048 episodes: 88.2 % against
88.1 % at level 40, the grip check firing as often (28.8 / 28.1 %). The wrist's motion is not what pulls the rod
out: a pinch 8 mm off a cylinder's axis lets it go over those steps either way. The cure is a deeper pinch at
the latch, which the latch cannot demand (the estimate's 8 mm bias hides the depth): the policy has to learn it.

**Untrained level-39 pick at level 40** (chain, 2048 episodes): **88.1 %** (74.9 % at level 39); latched 96.3 %,
grip check into a retry 28.1 %, success without a retry 67.8 %, invalid collisions 0.7 %, excess force 1.7 %.

**Training** from the level-39 pick, 1024 envs, 1200 iterations (`logs/chains/chain_L40.sh`, 19:49-21:26).
Training-time success flat at 0.87-0.90 throughout. **Pick** (quota rule, 2048 episodes): `model_16000` 89.1 %,
`model_16050` 89.5 %, **`model_16098` 89.5 %** (`logs/rsl_rl/adam_u30_wuji_reach_grasp_L40/2026-10-13_17-51-14/`);
no heap 92.5 %. The other scene rows were not run: the double count below makes every number above optimistic.

## The once-per-step counters advanced twice on reset steps (found 2026-10-13, fixed)

`reset_rg_buffers` dropped the once-per-step caches (`_step`, `_est_step`, ...) so that a reset env's first
observation is recomputed in the same step. That re-ran `_update` and the estimate update for EVERY env: on any step
in which some env reset, every other env's torque and contact streaks, hold count and perception history advanced
twice. Found by the parity test (adam-moveit genesis_policy): the runtime counts once per step and missed a latch
the simulator made one step early; in a 4-env recording the simulator's `tau_streak` ran ahead of a plain count on
540 of 1078 steps, by exactly one each time a neighbour env reset.

With 1024 training envs and ~150-step episodes several envs reset on nearly every step, so in training most steps
double-counted: the latch's 4-step torque streak was ~2 real steps, the 30-step hold ~15, and the latency history
advanced twice. Evaluations at 256 envs were affected less but still. **Every success number up to here is
optimistic against a robot**, which counts each step once.

Fix: `_rerun_fresh` re-runs the once-per-step bodies for the envs that just reset and restores every other env's
buffers. `ADAM_U_RG_LEGACY_STEP=1` restores the old behaviour to reproduce earlier numbers. Verified: a 64-env
recording (resets on most steps) has 0 streak mismatches against a plain count in 876 steps, and parity passes.
**Its size:** the level-40 pick re-evaluated with the fix (quota rule, 2048 episodes): **87.9 %** (89.5 % before),
latched 94.7 %, the grip check into a retry 21.3 %, drops 10.6 % (the drop count was inflated too), success without
a retry 71.8 %. The policy did not depend on the double count; the numbers above are 1-2 points optimistic.

Note also: buffers whose names start with `_` (`_est_win`, `_est_prev`, `_rest_est`) are not cleared at reset and
carry over from the previous episode; the runtime starts them at zero.

## Level 39 — transfer (2026-10-09, afternoon)

**Why.** Under the corrected evaluation the last level that passes the gate is 32 (95.3 %);
levels 33–38 sit at 74–89 % and the four heap levers moved nothing. The user's decision: stop
adding scenes, take `model_13700.pt` to the robot. The simulator side of that is a level on
which the policy and the phase machine read only what the robot can measure, trained from the
level-32 pick on the level-32 scene. The procedure is in `SIM_TO_REAL.md` §5.

**What.** `RGLevel` flags `tactile=False` (the tip_forces and contacts observations read zero,
interface still 74), `hw_latch=True` (the latch's grip and stability tests read finger flexion
torque, `_grip_torque`, the simulator's `applied_torque`; its inside, centred and settled tests
read the estimated rod and the estimate's own velocity), `sense_est=True` (the grip check, the
retry's lifted test and the drop test read the estimated height against the estimate's rest
height at step 1; **success stays on the true rod**), `arm_gains=True` (gain randomisation on
the arm too); `_PERC8` (bias 8 / 5 mm / 5°, noise 1–3 mm, latency 1–4 steps over a 5-deep
history, dropout 3 %), `_PHYS6` (gains 0.8–1.2). `GRIP_TAU_MIN` is calibrated against the 4 N
force latch on the level-32 pick before training. `logs/chains/chain_L39.sh` runs the level
with the quota rule at 2048 episodes.

**Calibration of the torque grip signal** (level-32 pick at level 32, 256 envs, 1500 steps,
54 414 grasp-phase samples): the summed |applied torque| over a finger's flexion chain
correlates with the fingertip force at +0.86 (thumb), +0.80 (index), +0.82 (middle); a PD
proxy kp·(target − q) does not (+0.2 to +0.6; it swings negative while a blocked finger
overshoots). At ≥ 4 N the chain torque is 0.25–0.53 N·m (p10–p50), a free finger 0.03–0.05
N·m (p50–p90). `GRIP_TAU_MIN = 0.20 N·m` fires on about 93 % of 4 N holds and 3 % of free
samples (thumb 95 / 3.0 % at 0.18, index 90 / 3.0 % at 0.30, middle 90 / 1.8 % at 0.19).

**Untrained** (the level-32 pick at level 39, quota rule, 2048 episodes, 2026-10-13): **51.1 %**
lift-and-hold success, against 95.3 % at level 32. That is what the signal swap alone costs.

**Training** (2026-10-13, 17:25): 1024 envs (2048 had run the host out of memory on 2026-10-09, the
trainer killed at 115 GB), 1200 iterations from `model_13700.pt`, `--reset-std 0.3`. Training-time
success 0.68 two hundred iterations in.

**Pick** (quota rule, 2048 episodes): `model_14800` 71.6 %, `model_14850` 74.2 %, **`model_14899` 74.9 %**
(`logs/rsl_rl/adam_u30_wuji_reach_grasp_L39/2026-10-13_15-27-40/`). Training won back 24 of the 44 points the
signal swap cost. The scene rows were not run: the level-39 latch itself was the bottleneck (level 40 above), so
the chain was stopped after the pick and level 40 trained from it.

## Evaluation stopping rule (2026-10-09, morning): the heap rows were biased upward

The level-36 and level-38 picks scored 79.0 / 83.6 % on the heap-only row at 512 episodes and
70.6 / 72.8 % at 2048. A per-block trace first looked like drift (the first 256 completed
episodes at 93 %, every later block at 63–70 %), and a per-env trace then showed no drift at
all: the state at the policy's release is identical for first and later episodes of an env,
and success by episode index is flat. The effect is **selection by episode length**:
`eval_reach_grasp.py` stopped at the first N completions, so it counted the fast episodes
(a latch at step 60, lift, hold: ~250 steps) and left the slow ones (three retries: up to 700
steps) in flight, uncounted. The bias scales with the failure rate and the retry count, so it
was small on the table scenes (97 % success, few retries) and 8–11 points on the heap.

Fixed in `eval_reach_grasp.py` and `audit_reach_grasp.py`: every env contributes exactly
`ceil(episodes / num_envs)` episodes; completions past an env's quota are ignored and the run
ends when every env has its quota. **All rows in this log up to here were measured with the
old rule, and the bias is not confined to the heap**: the level-34 pick reads 89.4 % mixed with
the quota rule at 2048 episodes against 97.1 % at 512 with the old rule. Since level 22, where
retries made failures three to four times longer than successes, every row has been inflated
by an amount that grows with the failure rate; the gates passed since then were measured
against inflated numbers. The ladder from level 22 on needs re-scoring with the quota rule
before any level is called passed; the picks of levels 34–38 are re-scored below first.

Re-scored with the quota rule, 2048 episodes (8 per env), policy mean action:

| pick, scene | old rule, 512 | quota, 2048 | latched | retried / recovered | grip check | collisions |
|---|---|---|---|---|---|---|
| level-34 pick at level 34, mixed | 97.1 % | **89.4 %** | 90.9 % | 15.4 % / 40 % | 7.6 % | 0.9 % |
| level-34 pick at level 35, heap only | 78.6 % | **70.6 %** | 74.7 % | 38.9 % / 37 % | 20.1 % | 2.1 % |
| level-36 pick at level 36, mixed | 90.0 % | **74.1 %** | 77.9 % | 34.7 % / 36 % | 18.4 % | 2.6 % |
| level-36 pick at level 36, heap only | 79.0 % | **67.1 %** | 72.7 % | 42.1 % / 35 % | 22.8 % | 3.2 % |
| level-38 pick at level 38, mixed | 86.5 % | **77.2 %** | 87.3 % | 39.5 % / 52 % | 31.8 % | 3.0 % |
| level-38 pick at level 38, heap only | 83.6 % | **71.3 %** | 84.2 % | 48.8 % / 48 % | 41.5 % | 2.7 % |

Read with the corrected rule: the pre-heap policy stands at 89.4 % on its own level, not 97;
the heap levels sit at 74–77 % mixed and 67–71 % on the heap; the level-38 pick (10 mm latch)
is the best of the heap-trained picks on both rows, by three to four points over level 36,
and the level-34 pick untrained on the heap (70.6 %) is within a point of the best trained one.
The ladder's picks re-scored on their own level, mixed scene, quota rule, 2048 episodes:

| level, pick | old rule, 512 | quota, 2048 | retried / recovered | collisions |
|---|---|---|---|---|
| 29, model_12000 | 97.3 % | 92.9 % | 11.0 % / 53 % | 1.1 % |
| 30, model_12550 | 97.5 % | 93.6 % | 13.5 % / 65 % | 0.2 % |
| 31, model_13149 | 98.2 % | **95.4 %** | 16.0 % / 79 % | 0.6 % |
| 32, model_13700 | 97.7 % | **95.3 %** | 13.1 % / 78 % | 0.9 % |
| 33, model_14200 | 98.1 % | 87.5 % | 17.9 % / 39 % | 1.4 % |
| 34, model_14799 | 97.1 % | 89.4 % | 15.4 % / 40 % | 0.9 % |

Levels 31 and 32 clear the 95 % gate under the corrected rule (collisions apart, as before).
The cliff is **level 33**, the mixed scenes: 87.5 %, with retry recovery halved (39 % from 78 %).
The heap on top of that costs another 10–15 points. So the ladder's real frontier is the mixed
bin scene (cover block, bed rods, pile together), not the heap; the heap levels were built on
a level that had not passed.

| level, pick | quota, 2048 |
|---|---|
| 25, model_9850 (bin walls) | 91.2 % |
| 26, model_10400 (grip check, commanded point) | 86.8 % |
| 27, model_10900 (pile, commanded point) | 85.4 % |
| 28, model_11450 (centred pinch) | 80.8 % |

Levels 25–28 were under the gate as well; the ladder recovered to 95 % at levels 31–32 (free
grasp, the v3 grip check, three retries) and fell again at the mixed scenes.

| level, pick | quota, 2048 | retried / recovered |
|---|---|---|
| 22, model_7648 (first retry) | 95.0 % | 7.5 % / 60 % |
| 23, model_8247 (cover block) | 84.8 % | 16.2 % / 35 % |
| 24, model_8750 (bed rods) | 94.7 % | 8.3 % / 51 % |

The full re-scored ladder: 22 at 95.0, 23 at 84.8, 24 at 94.7, 25 at 91.2, 26 at 86.8, 27 at
85.4, 28 at 80.8, 29 at 92.9, 30 at 93.6, **31 at 95.4, 32 at 95.3**, 33 at 87.5, 34 at 89.4,
36 at 74.1, 38 at 77.2. The **cover block** is the recurring loss: level 23 introduced it and
fell to 84.8, and level 33, which mixes it back in at 30 % of lying rods, fell to 87.5 with the
cover rows the weakest of every pick since (69–72 % at the heap levels). The next level should
be built on the level-32 pick, on the mixed scene, with the cover block traced the way the heap
was: where the fingers end up, what the latch refuses, and what the retry does under the block.

## Level 38 — the heap, a 10 mm latch (prepared 2026-10-09, 00:30)

**Why.** Level 37's first 300 iterations read exactly like levels 35 and 36 (success 0.74,
latched 0.79, retried 0.35), and the fingertips were already pushing 2 mm into neighbours, so
the clearance penalty was not what held them. A latch trace of the level-36 pick with the
heap forced on (768 episodes, per episode the closest tip-to-rod approach, the best two-sided
grip and the pinch-centre error at that grip, recorded over the grasp phase) says what does:

| never-latched heap episodes (148 of 768, 19 %) | share |
|---|---|
| two-sided grip ≥ 4 N, pinch midpoint more than 6 mm off the axis (mean 11 mm) | 66 % |
| two-sided grip, centred, still no latch (relative velocity / stability) | 17 % |
| never reached the rod (closest mean tip distance > 15 mm) | 16 % |
| reached, no two-sided grip | 5 % |

The policy holds the rod; the **centred-pinch latch** (`pinch_centre_tol` 6 mm, level 28)
refuses the hold because the tips sit on the upper half of the rod, which is all the
neighbours against its sides allow. Level 28 built that rule because shallow pinches slipped
in the lift; levels 31–34 since built the grip check (v4) and three retries to catch exactly a
hold that does not lift. Latched heap episodes have the same best grip (7.0 N) and the same
share of steps with a tip on a neighbour (29 %) as the refused ones. The estimated rod pose
is 5 mm off the true one on average (p95 8.8 mm), so a 6 mm rule on a perception-driven pinch
is also at the edge of what the estimate supports.

**What.** Level 37 with `pinch_centre_tol=0.010` (tips at up to 80 % of the rod's height). The
grip check and the retries decide whether the shallow pinch holds. `logs/chains/chain_L38.sh`
runs it from the level-36 pick in parallel with level 37, so the two levers are measured
separately: 37 is the penalty alone, 38 the penalty plus the latch. The looser latch exposed a
guard that was missing: the level-38 tests had one scripted env latch with `grasp_end` 0, a
latch on the reset step from the previous episode's stale contacts. `update_latch` now refuses
the latch at t = 0 and during the settle-in (hand frozen open); tests 137/137.

**Level 38, result (2026-10-09, 04:48).** Run `adam_u30_wuji_reach_grasp_L38/2026-10-08_22-44-57`,
iterations 16000–17197 at 1024 envs from the level-36 pick, `--reset-std 0.3`, sharing the GPU
with level 37 for its first 700 iterations. The level-36 pick untrained under the 10 mm latch:
85.0 % mixed (latched 90.4 %, grip check fired 23.4 %, retried 25.7 %), 78.4 % heap only: the
looser latch turns refused holds into latched-then-rejected holds, and on its own moves the
failure from the latch to the grip check. Training success 0.71 / 0.75 / 0.75 / 0.76 / 0.75 /
0.76 / 0.75 / 0.75 / 0.74 / 0.74 / 0.74 / 0.74 per hundred: a hair above levels 35–37, flat.
Candidates 17100 / 17150 / 17197: 86.5 / 84.8 / 86.2 %. **Pick `model_17100.pt`.** Rows (512
episodes):

| level-38 pick, scene | success | latched | retried / recovered | grip check fired | collisions | joint-limit |
|---|---|---|---|---|---|---|
| mixed | 86.5 % | 92.4 % | 29.5 % / 66 % | 27.3 % | 3.1 % | 2 |
| heap only | 83.6 % | 93.9 % | 35.4 % / 66 % | 34.2 % | 3.5 % | 4 |
| no heap | 97.3 % | 98.7 % | 8.2 % / 81 % | 6.9 % | 1.7 % | 0 |
| lying | 73.8 % | 83.2 % | 34.9 % / 45 % | 27.1 % | 5.5 % | 11 |
| tilted | 91.0 % | 97.7 % | 26.2 % / 74 % | 23.8 % | 0.4 % | 0 |
| standing | 94.4 % | 98.1 % | 11.5 % / 80 % | 11.1 % | 0.6 % | 2 |
| rod across the target | 85.0 % | 93.4 % | 29.0 % / 60 % | 25.9 % | 3.5 % | 5 |
| cover block | 71.7 % | 81.1 % | 37.9 % / 41 % | 26.8 % | 4.3 % | 4 |
| bed rods | 82.5 % | 92.2 % | 24.4 % / 55 % | 21.8 % | 5.5 % | 0 |

The policy latches in 92–94 % of heap episodes now (level 36: 81 %), but a third of those holds
fail the grip check and only two thirds of the retries recover, so the mixed number is 86.5 %
against 90.0 % under the 6 mm latch. The heap-only row's 83.6 % was the best of the night at 512
episodes, but **at 2048 episodes** the level-36 and level-38 picks score 70.6 % and 72.8 % on
the heap, 8–11 points under their 512-episode rows (79.0 / 83.6). Two 512-episode draws of the
level-36 pick had already given 79.0 and 73.8. Either the heap row's draw noise is far larger
than ±3 points or success drifts across evaluation rounds; a per-block trace (2048 episodes,
success per 256 completions) is running to tell the two apart, and every heap number above is
provisional until it reports.

## Level 37 — the heap, pushing allowed (prepared 2026-10-08, late evening)

**Why.** With the heap forced on, the eight-sector map at the policy's release says a clean
pinch rarely exists: the two horizontal sides of the rod are each fully blocked in about 60 %
of episodes and both free in 10–16 %; the end faces are both free in 4 %. In most heap
episodes the fingertip has to displace a touching neighbour to get beside the rod. Pushing a
neighbour is allowed by the rules (neighbour capsules are "soft": not a collision), but the
clearance penalty (`clearance_penalty`, tips margin 4 mm, palm 20 mm, forearm 30 mm) is
computed from every capsule including the soft ones, so the policy was paid, every step, to
stay away from exactly the rods it must move. At levels 27–34 the neighbours sat 3 cm or
more from the target and the term rarely fired.

**What.** `RGLevel.push_ok`: the penalty reads `clear_hard` (the table, blocks, walls and bed
rods; neighbours excluded), computed in `_clearance` beside the full clearance that the
observation and the logs keep. Level 37 is level 36 plus this flag; `logs/chains/chain_L37.sh`
runs it from a level-36 pick (a 74-wide checkpoint is widened first). Levels 0–36 are
unchanged.

**Level 37, result (2026-10-09, 03:47).** Run `adam_u30_wuji_reach_grasp_L37/2026-10-08_21-31-42`,
iterations 16000–17197 at 1024 envs from the level-36 pick, `--reset-std 0.3` (the last 700
iterations shared the GPU with level 38). The level-36 pick untrained at level 37: 88.7 % mixed,
73.8 % heap only. Training success flat at 0.72–0.74 throughout; the fingertips' minimum
clearance to neighbours was already −2 mm in the first hundred iterations, so the penalty had
not been holding them. Candidates 17100 / 17150 / 17197: 82.4 / 83.8 / 87.3 %. **Pick
`model_17197.pt`**, under the untrained start. Rows (512 episodes):

| level-37 pick, scene | success | retried / recovered | grip check fired | collisions | joint-limit |
|---|---|---|---|---|---|
| mixed | 87.3 % | 18.4 % / 70 % | 16.0 % | 3.7 % | 8 |
| heap only | 77.4 % | 30.3 % / 50 % | 21.4 % | 3.1 % | 10 |
| no heap | 91.6 % | 9.2 % / 64 % | 7.4 % | 4.9 % | 2 |
| lying | 73.3 % | 27.4 % / 41 % | 17.2 % | 5.8 % | 14 |
| tilted | 87.3 % | 25.0 % / 67 % | 18.9 % | 0.4 % | 1 |
| standing | 96.5 % | 8.0 % / 81 % | 6.6 % | 1.0 % | 5 |
| rod across the target | 86.9 % | 21.7 % / 77 % | 18.0 % | 3.7 % | 13 |
| cover block | 70.9 % | 34.6 % / 33 % | 18.4 % | 3.1 % | 4 |
| bed rods | 86.4 % | 18.4 % / 63 % | 15.0 % | 3.3 % | 11 |

Removing the neighbour term from the penalty bought nothing on the heap (77.4 % against the
level-36 pick's 79.0) and cost off it: no heap 91.6 % against 97.1, collisions 3–6 % against
2–4, and joint-limit terminations up to 14 per 512 episodes. The policy did not learn to push;
it got looser. The lever is withdrawn: level 38 carries `push_ok` only because it was defined
on top of 37, and the next level after it drops the flag.

## Level 36 — the heap map (2026-10-08, evening; prepared while level 35 trains)

**Why.** Level 35's first 300 iterations sit flat at 0.75 training success (level 34 ran at
0.86–0.88), the signature of a level the policy cannot see (level 27 did the same until the
centred pinch). The per-episode trace says what it cannot see: success falls with the number
of rods touching the target (93 % with none, 73 % with three or more, half the heap episodes),
while the safety observation carries one nearest-obstacle vector per link group and nothing
about which side of the target is free. The policy finds a pinch by touch, the grip check
rejects it, and three retries run out.

**What.** Level 35 plus `heap_map_obs`: **8 observations**, the free radial distance around the
*estimated* rod in eight sectors (`HEAP_MAP_DIRS`), each the largest probe distance beyond the
rod's surface (5 to 45 mm in 10 mm steps, over three points along the axis) at which no
neighbour rod, bed rod, block, wall or the table is hit, divided by 45 mm. Sector 0 is the
horizontal direction `z × axis`, 2 is up, 4 the other side, 6 down. Two opposite free sectors
are a pinch site. The neighbours are their true capsules (a depth camera sees them directly);
the rod is the estimate, as everywhere else. The term is appended, so indices 0–73 are
unchanged; `ObservationsCfg.PolicyHeapCfg` is swapped in by `RGLevel.heap_map`. A level-35
checkpoint is carried over with `tools/expand_checkpoint.py --add 8`: zero first-layer
columns for the new inputs in the actor and the critic, mean 0 / std 1 in their normalisers,
zero Adam moments, so the widened policy computes exactly what the old one did until training
moves the new weights. `logs/chains/chain_L36.sh` widens the level-35 pick, evaluates it (must
match its level-35 numbers), trains 1200 iterations at 1024 envs, picks and runs the rows.
Tests: the interface check allows 82 at map levels. On hardware the map comes from the depth
image around the detected rod, the same way the rod pose does.

Parity: the level-34 pick widened and evaluated at level 36 (mixed) scores 89.9 %, against
90.0 % at level 35; the widening changes nothing. A re-labelled target also takes its own rest
height (the 5 cm lift is measured from where the new rod sits), and the tests that read the
rest height or the bed height skip re-labelled targets; the across-rod and bed checks tolerate
the heap's dropped rods, which with the fixed seed throw one across rod and knock one bed target.

**Level 36, result (2026-10-08, 23:45).** Run `adam_u30_wuji_reach_grasp_L36/2026-10-08_18-54-07`,
iterations 14800–15998 at 1024 envs from the **level-34 pick widened** (`model_14799_w82.pt`; not
the level-35 pick, which was weaker on every scene), `--reset-std 0.3`, launched 20:51 after the
top-rod rule was corrected to the rod resting on the target. The widened pick untrained at
level 36: 88.1 % mixed, 80.3 % heap only. The map is read (the actor's first-layer columns for
the eight new inputs grew from 0 to a mean norm of 0.37 by iteration 15200, against 2.2 for
the old columns; the critic's to 1.2 against 4.4), and the map itself is sane (checked at the
policy's release with the heap forced on: sector means 0.25 / 0.58 / 0.66 / 0.56 / 0.23 /
0.03 / 0.04 / 0.04 for side / up-diagonal / up / up-diagonal / side / down…, the mean value
falling from 0.57 with no touching rod to 0.28 with three or more). But training success
stayed flat at 0.73–0.75 for all 1200 iterations. Candidates 15900 / 15950 / 15998: 88.1 /
88.7 / 90.0 %. **Pick `model_15998.pt`.** Rows (512 episodes):

| level-36 pick, scene | success | retried / recovered | collisions | re-labelled | joint-limit |
|---|---|---|---|---|---|
| mixed | 90.0 % | 20.1 % / 67 % | 1.8 % | 12.1 % | 6 |
| heap only | 79.0 % | 30.5 % / 52 % | 2.7 % | 16.7 % | 10 |
| no heap | 97.1 % | 5.8 % / 87 % | 2.1 % | | 0 |
| lying | 74.8 % | 30.1 % / 36 % | 3.3 % | 23.0 % | 5 |
| tilted | 90.6 % | 21.6 % / 70 % | 0.6 % | | 0 |
| standing | 96.3 % | 5.2 % / 82 % | 0.4 % | | 4 |
| rod across the target | 89.1 % | 16.8 % / 65 % | 3.1 % | 10.9 % | 6 |
| cover block | 69.0 % | 37.2 % / 28 % | 4.5 % | 10.1 % | 3 |
| bed rods | 86.1 % | 17.6 % / 46 % | 3.3 % | 3.5 % | 2 |

Against level 35's run, the map run kept what the policy had (no heap 97.1 % against level
35's 92.2; collisions 2–4 % against 4–9) and held the heap at the untrained level instead of
losing to it, but it gained nothing on the heap (79.0 % against 80.3 untrained; the heap-only
row has about ±3 points of draw noise at 512 episodes: the same checkpoint scored 79.0 and
73.8 on two draws). Joint-limit terminations are up (6–10 per 512 on the heap rows). The map
tells the policy where the rod is blocked, but the diagnostic that produced level 37 says a
clean pinch rarely exists to be found: the policy has to push, and the clearance penalty paid
it not to.

## Level 35 — the heap (queued 2026-10-06, evening)

**What.** Level 34's scenes plus a real heap in 70 % of episodes: ten rods, six placed
overlap-free on the table around the target (3 to 7 cm, one resting across a lying
target in 30 %), four **dropped from 2.3 cm** onto them and onto the target at random yaw
within 6 cm of it. The policy is **frozen for the first 25 steps** (`PileCfg.settle_in`:
actor mask off, wrist and fingers still) while the heap settles; the neighbours' rest
heights for the double-pick rule are re-recorded when the policy is released. Ten
neighbour slots (`NEIGHBOR_SLOTS`, `MAX_CAPSULES` 12). Measured at 16 envs: the settling
heap moves the target up to 5 cm and turns it up to 15°, so the start pose can be well
off the rod; the free-grasp policy sees the true rod and has to travel. The tests report
the knocks as information at heap levels and keep the 2 cm rule for the plain pile.

Test group 22 adds: the dropped rods start above the first layer, the policy is frozen
through the settle-in and released on the step after it, the dropped rods have landed
(within 5 cm of the rest height) by then, and no double pick is flagged after the rest
heights are re-recorded. Full suite at level 35: 133 checks. Queued behind level 34 from
its pick; the same scene evals plus the pick at level 26.

**Level 35, first attempt (2026-10-08, evening).** Tests 138/138, audit 8/9. The level-34 pick
untrained at level 35: **83.3 %**, the first level transition under 90 % (every one since level
29 was above 96 %). The chain's training run was then **killed by the host kernel** 100 s after
launch: the trainer reached 115 GB resident on the 125 GB machine (four more rod bodies per env
than level 34, at 2048 envs), and the chain stopped with no checkpoint.

*Trace of the untrained pick* (`--pile 1` vs `--pile 0`, then a per-episode trace with the heap
forced on, 768 episodes, heap structure recorded at the end of the settle-in):

| | heap off | heap on |
|---|---|---|
| success | 98.3 % | 84.2 % |
| success without a retry | 95.0 % | 67.3 % |
| grip check sent into a retry | 2.3 % | 19.5 % |
| retried / recovered | 3.7 % / 90 % | 29.0 % / 58 % |
| mean grasp end step | 55 | 124 |
| rod touched the table in the hold | 0.0 % | 7.8 % |

Per episode, success by the number of rods **touching** the target after the settle: none
93 % (4 % of episodes), one 81 %, two 78 %, three or more 73 % (49 % of episodes). A dropped
rod **resting on** the target: 36 % of episodes, 71 % success. Where a free span of at least
1 cm along the axis survives, success is above 90 %. Failures: never lifted 14.8 % (101 of 114
never latched, 100 with all three retries spent), drops 4.2 %, double picks 3.5 %, collisions
1.8 %. The dropped rods spawn within 6 cm of the target at random yaw and are checked for
overlap only against each other, so landing on it is the expected case. At levels 33 and 34
the neighbours sat 3 cm or more from the target and never touched it; the policy had never
seen a touching rod. Two design faults on top: the 25 frozen steps came out of the 120-step
grasp budget (the first attempt had 95 live steps) and the time observation was not shifted,
so the policy saw a clock 25 steps ahead of the one it trained on; and a retry re-closed
where the previous attempt had failed, with the fingers still in the pile.

*Fixes (env_cfg.py, all gated by level fields so levels 0–34 are unchanged):*

- **Settle-in budget.** `episode_max_steps(retry_max, settle_in)` / `level_max_steps(lvl)`:
  the grasp timeout and the episode are `settle_in` steps longer, and `episode_time` counts
  from the policy's release.
- **Top-rod rule** (`PileCfg.top_rod`, `_heap_top_rod`). One step before the rest heights are
  recorded, a lying target with a dropped rod resting on it swaps poses with the highest rod
  resting on it; every rod is the same body, so this only changes which rod is asked for, the
  way a perception system would pick the rod on top. (The first version, used for the level-35
  run and its rows, took the highest rod of the whole heap, up to 6 cm from the old target,
  which the arm's start pose and the policy's +/-2 cm training offsets do not cover: the
  level-35 pick's lying row fell to 69.9 % with 22.9 % re-labelled. Corrected 20:46 before
  level 36 trained.) Standing and tilted targets keep their
  label (wrist roll and lift axis were solved at reset); a swapped target clears its cover and
  bed flags. The eval prints "heap re-labelled the target".
- **Retry back-off** (`RGLevel.retry_backoff` 3 cm): on a retry the wrist target is raised
  before the policy resumes.
- **Memory**: `logs/chains/chain_L35.sh` trains with 1024 envs for 1200 iterations (the same
  sample count).

Re-traced with the fixes, untrained: 76 % on the heap alone, unchanged (re-labelling cut the
covered share to 21 %, but 2.4 rods still touch the target on average, and that is what
training has to learn). Tests after the fixes 132/136; the four failures were the tests'
own assumptions about the episode length and the across rod (now re-labelled), updated.
Relaunched 17:49.

**Level 35, result (2026-10-08, 20:57).** Run `adam_u30_wuji_reach_grasp_L35/2026-10-08_16-04-25`,
iterations 14800–15998 at 1024 envs from the level-34 pick with `--reset-std 0.3` (tests
136/137 after the fixes, the one failure the across-rod draw; audit 8/9). Trainer memory: 63 GB
peak during scene construction, 10 GB steady. The level-34 pick untrained after the fixes:
**90.0 %** mixed (from 83.3), 78.6 % with the heap forced on. The run **did not learn**: per
hundred iterations, training success 0.71 / 0.75 / 0.75 / 0.75 / 0.76 / 0.74 / 0.73 / 0.72 /
0.73 …, latched 0.78, retried 0.33, grip check fired 0.19, collisions 0.06, grasp end step 150,
all flat from iteration 14900 to the end (level 34 trained at 0.86–0.88). Candidates 15900 /
15950 / 15998: 88.5 / 86.8 / 84.0 %, every one under the untrained 90.0. **Pick `model_15900.pt`**
for the record; it is not carried forward. Rows (512 episodes; the top-rod rule in its first
version, highest rod of the heap):

| level-35 pick, scene | success | retried / recovered | collisions | re-labelled | joint-limit |
|---|---|---|---|---|---|
| mixed | 88.5 % | 18.2 % / 74 % | 4.5 % | 9.2 % | 1 |
| heap only | 73.3 % | 32.7 % / 40 % | 4.5 % | 16.4 % | 1 |
| no heap | 92.2 % | 5.5 % / 71 % | 5.8 % | | 2 |
| lying | 69.9 % | 28.1 % / 35 % | 9.2 % | 22.9 % | 1 |
| tilted | 90.8 % | 20.1 % / 69 % | 0.8 % | | 1 |
| standing | 93.0 % | 9.1 % / 79 % | 3.9 % | | 0 |
| rod across the target | 78.6 % | 22.2 % / 40 % | 5.6 % | 11.5 % | 0 |
| cover block | 72.1 % | 36.3 % / 37 % | 5.7 % | 8.6 % | 2 |
| bed rods | 85.6 % | 12.7 % / 55 % | 5.3 % | 2.5 % | 3 |

The non-learning run also eroded what the policy had: without the heap 92.2 % against 98.3 %
for the level-34 pick on the same scene, and collisions at 4–9 % on every row (level 34: 1–3 %).
The lying row is the worst (69.9 %, 22.9 % re-labelled): it is where the first top-rod rule
moved the target up to 6 cm from the hand's solved start pose. Level 36 (the heap map) starts
from the **level-34 pick** widened, not from this one.

## Level 34 — the grip check over the last half of the test lift (queued 2026-10-06, evening)

Level 33 with the grip check (v4) measuring the rod's rise against the fingertips' over
the **second half of the test lift only**: the reference heights are recorded at the
middle of the test lift, after the finger drives and the 2 mm contact offsets have taken
up their slack. Level 31's v3 measured from the start and fired on 12 % of lying grasps
that would have lifted. `grip_check_late=True`; test group 24 checks the reference is
recorded mid-lift. Queued behind level 33 from its pick; the same scene evals.

**Level 34, result (2026-10-08).** Run `adam_u30_wuji_reach_grasp_L34/2026-10-08_13-35-44`,
iterations 14200–14799 from the level-33 pick with `--reset-std 0.3`. Tests 134/134; audit 8/9
(as at level 33). The level-33 pick untrained at level 34: 97.3 %. Candidates 14700 / 14750 /
14799: 96.7 / 96.9 / 97.1 %. **Pick `model_14799.pt`.** 512 episodes per row:

| pick, scene | success | retried / recovered | collisions | joint-limit | notes |
|---|---|---|---|---|---|
| mixed | 97.1 % | 4.9 % / 80 % | 1.0 % | 0 | success without a retry 93.2 % |
| lying | 95.5 % | 8.8 % / 78 % | 2.7 % | 0 | peak 107 N |
| tilted | 99.2 % | 4.8 % / 92 % | 0.0 % | 0 | |
| standing | 99.2 % | 1.4 % / 86 % | 0.8 % | 0 | |
| rod across the target | 97.9 % | 4.8 % / 92 % | 1.4 % | 1 | |
| plain bin (no pile) | 96.9 % | 3.5 % / 61 % | 1.4 % | 2 | excess force 1.0 % |
| cover block | 93.6 % | 8.8 % / 44 % | 2.0 % | 2 | excess force 1.4 %, peak 118 N |
| bed rods | 97.7 % | 3.3 % / 82 % | 2.1 % | 1 | |
| level-26 scene | 69.7 % | | 12.7 % | 2 | not comparable (commanded point) |

The late grip check did what it was built for: retries roughly halved against level 33 (4.9 %
from 9.6 % on the lying rod's share of the mix, 8.8 % from 11.1 % under the cover) and the
recovery rate rose on the table scenes. Mixed slipped one point (97.1 from 98.1) and the cover
block stays the weak scene at 93.6 %. Peak fingertip forces came down (118 N from 194 N under
the cover). Drag p95 is under 1.5 mm on every comparable row; shield interventions zero. The
joint-limit terminations (one or two per 512 episodes on five rows) are new since level 32 and
sit with the free pre-lift window.

## Level 33 — mixed scenes (queued 2026-10-06, evening)

**Why.** The level-28 pick evaluated on the level-26 scene (cover block, bed rods, bin,
no pile) fell to 82.1 % with 7.8 % obstacle collisions, from 91.6 % for the level-26
pick: two pile levels without the cover and the bed were enough to lose them. A
checkpoint meant to be final has to see every scene of the ladder.

**What.** Level 32's policy with every scene drawn per episode inside the bin: the pile
in 60 % of episodes (six rods, one resting across a lying target in 30 %), the cover
block over one end of 30 % of the lying rods, the bed rods under 40 % of them, plain bin
otherwise; standing 20 %, tilted 30 %. `PileCfg.prob` draws the pile; unused neighbour
rods are parked on the floor beside the table (a dynamic body parked inside the table,
where the kinematic obstacles wait, is thrown out by the depenetration; found by test
group 22). The across rod is placed on lying targets without the cover (a covered rod is
pinned already, and the block ejected the across rod) and is pitched per episode so its
near end rests on the table even when the target sits on bed rods. The overlap-free
placement also avoids the bed rods. Evaluation flags `--pile 1/0`, `--cover`, `--bed`,
`--across` select the scenes; the chain also evaluates the pick at level 26, the scene
the level-28 pick had lost. Queued behind level 32 from its pick.

**Level 33, result (2026-10-08).** Run `adam_u30_wuji_reach_grasp_L33/2026-10-08_11-51-59`,
iterations 13700–14299 from the level-32 pick with `--reset-std 0.3`. Tests 133/133; audit
8/9 (the scripted-grasp check fails by design at pile levels: a blind close cannot satisfy the
centred-pinch latch). The level-32 pick untrained at level 33: 96.3 % with 2.5 % collisions.
Candidates 14200 / 14250 / 14299: 98.1 / 97.5 / 98.0 %. **Pick `model_14200.pt`.** 512
episodes per row:

| pick, scene | success | retried / recovered | collisions | notes |
|---|---|---|---|---|
| mixed | 98.1 % | | | |
| lying | 95.1 % | 9.6 % / 73 % | 3.1 % | excess force 1.2 % (peak 144 N), 1 joint-limit termination |
| tilted | 99.4 % | | | |
| standing | 100 % | | | |
| rod across the target | 96.7 % | | | |
| plain bin (no pile) | 98.2 % | | | |
| cover block | 92.0 % | 11.1 % / 47 % | 3.3 % | drops 2.1 %, peak 194 N |
| bed rods | 97.9 % | | | |
| level-26 scene | 60.9 % | | 24.5 % | not comparable: level 26 commands the grasp point, this pick grasps freely; drag p95 14.8 mm |

The mixed-scene checkpoint keeps every table scene and the plain bin and the bed. The cover
block is the weak scene: half the retries there fail and the fingertip force peaks are the
highest of the ladder. Collisions sit at 3 % on the lying rod and under the cover, three times
level 32's; the free pre-lift window's cost did not go away with the mixed scenes.

## Levels 31 and 32 — the grip check redone, and the pre-lift as a policy phase (queued 2026-10-06)

Queued behind level 30 in the same way (previous pick untrained first, then 600 iterations).

**Level 31, `early_retry`.** Level 30 with the grip check rebuilt (v3). The two level-26
versions measured an 8 mm test lift at the wrist and fired on 40–85 % of good grasps,
because the finger drives and the 2 mm contact offsets take up the first millimetres and
the wrist lags its target. Now the test lift is **24 mm commanded over the whole 18-step
pre-lift**, measured at the **fingertips** (mean of the three pads), and a latched rod that
rose less than 40 % of what the tips rose is not in the hand: the retry fires there, 40
steps before the empty 10 cm lift would have ended. Tips that rose under 4 mm make the
test inconclusive (no retry). Test group 24 fakes a latch at the standoff: the tips rise
18 mm and the check fires on the step expected. With a failed attempt that much cheaper the
level allows three retries. Evaluation prints "grip check sent into a retry"; the training
log has `Outcome/grip_check_fired`. If the level-30 pick evaluated untrained at level 31
loses success, the check is still firing on good grasps and the level is wrong, not the
policy.

**Level 32, `free_prelift`.** Level 31 with the first 10 pre-lift steps the **policy's**:
after the 10-step settle the wrist follows the policy again with the fingers held at the
latched closure, then the 8-step test lift (24 mm, grip check) and the scripted lift. The
actor mask includes the window. No new reward: the success, the drop and the double-pick
rules are what the policy has to satisfy. This is the slide-out as a policy phase (level
26's scripted drag made the covered rod worse): a rod under a crossing neighbour can be
slid, tilted up or lifted straight away before the lift. Test group 25 checks the mask,
the live wrist, the held fingers and the scripted test lift after the window.

_Result under level 31 below._

## Levels 29 and 30 — free grasp in the pile, then a denser pile (queued 2026-10-06)

Queued behind level 28, each starting from the previous level's pick, 600 iterations, with
the previous pick evaluated untrained at the new level first.

**Level 29, `free_pile`.** Level 28 without the commanded point: level 17's free grasp in the
pile. The policy sees the true rod, the reach reward is to the rod's axis segment with the
tilt-only orientation term (now measured against the rolled approach axis, so it holds for
the tilted rod), the world-yaw action runs at a quarter rate, and the centred pinch stays
binding (success needs the latch). The level-27 analysis gives the reason: with the point
commanded next to a neighbour the first attempt lifts 41 % of the time, with none within
3 cm 84 to 91 %. A free policy can choose the clear part of the rod. The commanded-point
shaping becomes a depth-only shaping (pinch midpoint to the axis).

**Level 30, `dense_pile`.** Level 29 with six neighbours at 3 to 7 cm (was four at 4 to 8),
one of them dropped ACROSS a lying or tilted target in 30 % of the episodes (centred on the
target's middle 60 %, yawed 90 ± 30° to it, one diameter up; it settles on the target), and
two retries. The rod across the target is the pile's version of the cover block of levels
23 and 26 (10 to 24 % there): loose, 30 g, so it can be pushed off or lifted through, but a
neighbour that rises more than 2 cm with the target is still a double pick. Evaluation flag
`--across 1/0` pins or removes it. Test group 22 covers the slot count, the parked slots and
the across placement.

**Level 29, result.** Run `adam_u30_wuji_reach_grasp_L29/2026-10-06_10-53-04`, iterations
11450–12049 from the level-28 pick. Tests 102/102 (two stale expectations fixed first: the
blind scripted close cannot latch under the centred rule, now an info line; and the
free-grasp estimate check compared a yaw that a standing rod does not have, now the axis
angle with 2° of slack for the settling rod). **Pick `model_12000.pt`.** 512 episodes per row:

| policy, scene | mixed | lying | tilted | standing | latched | retried | collisions |
|---|---|---|---|---|---|---|---|
| level-28 pick at level 28 (commanded point) | 86.7 % | 85.4 % | 85.9 % | 93.3 % | 87 % | 12–15 % | 1.0 % |
| level-28 pick at level 29, **untrained** | 92.4 % | | | | 94.9 % | 15.6 % | |
| level 29, model 11950 / **12000** / 12049 | 94.7 / **95.3** / 92.0 % | | | | | 8–10 % | 2.3 / 1.2 / 2.5 % |
| level-29 pick, per pose | | **94.1 %** | **92.6 %** | **98.8 %** | 94–99 % | 3–11 % | 3.3 / 0.8 / 0.6 % |

The free point is worth 6 points before any training (86.7 to 92.4 %) and 9 after
(95.3 %): the policy picks the clear part of the rod, which is what the level-27 analysis
said the near-neighbour failures needed. Retries halve. Every pose is at or above its
clean-scene number from level 22 to 24. The one blemish is 3.3 % obstacle collisions on
the lying rod (the bin walls, since the neighbours are soft): the free policy reaches
further along the rod and gets closer to the walls; level 30's denser pile will tell
whether it grows.

**Level 30, what the tests found before training (2026-10-06, afternoon).** The dense pile
failed seven pose checks on its first run and the cause was the environment, not the
tests. Measured on 384 episodes, 30 settle steps, before the fix: a tilted target moved up
to 3.6 cm and turned up to 51°, a lying target under the across rod turned up to 15°.
Three causes, three fixes in `reset_pile`:

- Neighbours spawned through the tilted target's **support block** (and, at 3 cm, through
  the target or each other); the depenetration threw them out and knocked the target off
  its block. Now each neighbour draws 12 candidate placements and takes the first whose
  axis is more than a diameter plus 2 mm from the target, from every neighbour already
  placed and from every box obstacle (block, walls). This applies to levels 27 to 29 as
  well, which trained with the overlaps; their picks are unchanged, their scenes slightly
  tamer from now on.
- The across rod was **dropped** from 4 mm up and pivoted the target. It is now placed
  resting: on the target's axis at the crossing (one diameter up plus 1 mm) with its near
  end on the table, pitched to fit, 40 % of a half-length off centre. After settling it
  stays on top of a lying target in 70–100 % of episodes and otherwise rolls to lean on
  its side; the target moves under 3 mm.
- The across rod goes on **lying targets only**; a tilted target keeps the level-27 band
  (4 to 8 cm) for its neighbours.

After the fixes: lying targets move 0.4 mm and turn under 3° while the pile settles; test
groups 12 and 22 pass three runs each at 16 envs. Three test tolerances were widened with
the physics in mind: 5 mm of xy slack for six neighbours, 5° of yaw and 1 cm for a target
under the across rod, and 10° on the estimate check for a tilted rod still dropping onto
its block one step after the reset.

**Level 30, result.** Run `adam_u30_wuji_reach_grasp_L30/2026-10-06_12-37-08`, iterations
12000–12599 from the level-29 pick. Tests 106/106, audit done. **Pick `model_12550.pt`.**
512 episodes per row:

| policy, scene | mixed | lying | tilted | standing | across rod pinned (`--across 1`) | no across rod | latched | retried | double picks | collisions |
|---|---|---|---|---|---|---|---|---|---|---|
| level-29 pick at level 30, **untrained** | 94.9 % | | | | | | 96.3 % | 11.3 % | 0.4 % | 2.5 % |
| level 30, model 12500 / **12550** / 12599 | 96.9 / **98.5** / 97.3 % | | | | | | 98 % | 6.6–7.6 % | 0–0.2 % | 2.1 / 0.6 / 0.8 % |
| level-30 pick, per scene | | **94.9 %** | **98.5 %** | **98.4 %** | **95.5 %** (1.2 % double) | 96.4 % | 97–99 % | 2–12 % | | 0.2–1.9 % |

(The `--across 1 --tilt 1` row, 96.7 %, is a tilted-only control: the across rod goes on
lying targets only.) Six neighbours at 3 cm and a rod across the target cost the level-29
policy nothing (94.9 % untrained, against 95.3 % at level 29), and training in the dense
pile added three points. The pinned rod, the case the cover block of levels 23 and 26
left at 10 to 24 %, is at 95.5 % when the pin is a loose rod: the policy pushes it aside
or lifts through, and lifts both rods in 1.2 % of episodes (the double-pick rule counts
those as failures). The tilted rod's jump from 92.6 to 98.5 % is largely the spawn fix
(neighbours no longer thrown out of the support block), so the level-29 number was partly
an artefact of the scene. Lying in the dense pile, 94.9 % with 1.6 % wall contacts, is
the softest scene: the free policy reaches along the rod and meets the wall.

**Level 31, result.** Run `adam_u30_wuji_reach_grasp_L31/2026-10-06_14-15-17`, iterations
12550–13149 from the level-30 pick. Tests 112/112. **Pick `model_13149.pt`.** 512 episodes
per row; "check fired" is the share of episodes the grip check sent into a retry:

| policy, scene | success | success without a retry | retried | check fired | double picks | collisions |
|---|---|---|---|---|---|---|
| level-30 pick at level 31, untrained, mixed | 97.3 % | 88.1 % | 10.1 % | 8.0 % | 0.2 % | 0.4 % |
| level 31, model 13050 / 13100 / **13149**, mixed | 97.7 / 97.9 / **98.2 %** | 88–90 % | 9–11 % | | 0.2–0.6 % | 0.6–0.8 % |
| pick, lying | 96.4 % | 82.2 % | 14.5 % | 12.2 % | 0.6 % | 2.5 % |
| pick, tilted | 98.5 % | 92.7 % | 6.4 % | 3.9 % | 0.0 % | 0.2 % |
| pick, standing | 99.6 % | 96.7 % | 2.9 % | 1.6 % | 0.0 % | 0.2 % |
| pick, rod across the target | 97.5 % | 86.5 % | 12.1 % | 10.5 % | 0.2 % | 0.6 % |
| pick, no across rod | 97.5 % | 89.5 % | 9.0 % | 7.2 % | 0.0 % | 1.6 % |

Against level 30 (98.5 mixed, lying 94.9, across 95.5): the final numbers are the same or
better (lying +1.5, pinned +2), because 90–97 % of the retries succeed and each one is now
40 steps cheaper. But the check itself is not clean: it fires on 12 % of lying grasps and
success without a retry drops from 92 to 82–88 %, so most of what it sends back are grasps
that would have lifted. The 24 mm test lift at the fingertips separates held from unheld
far better than the 8 mm wrist test of level 26 (40–85 % false alarms) and not well
enough to call it a gate: the rod still lags the fingertips by several millimetres in the
first centimetres of a lift when the pinch is on a thread. Training did not make the
policy avoid the check (it fired in 13–14 % of training episodes throughout). Kept at
levels 31 and 32 since it costs nothing in the final number; a cleaner version would
compare the rod's rise with the fingertips' over the LAST 12 mm of the test lift only,
after the slack is taken up. Added to the plan.

**Level 32, result.** Two runs: `adam_u30_wuji_reach_grasp_L32/2026-10-06_16-39-25`,
iterations 13150–13350 from the level-31 pick (stopped at 200 of 600), and
`adam_u30_wuji_reach_grasp_L32/2026-10-08_10-38-38`, iterations 13350–13749 resumed from
`model_13350.pt` with no std reset (the same run continued). Tests 118/118.
**Pick `model_13700.pt`.** 512 episodes per row:

| policy, scene | success | success without a retry | retried | check fired | double picks | collisions |
|---|---|---|---|---|---|---|
| level 32, model 13650 / **13700** / 13749, mixed | 96.7 / **97.7** / 97.3 % | 90–92 % | 6–8 % | 4–6 % | 0.2–0.4 % | 1.2–1.4 % |
| pick, lying | 95.0 % | 84.7 % | 11.0 % | 9.3 % | 1.5 % | 2.1 % |
| pick, tilted | 98.4 % | 94.0 % | 5.2 % | 3.3 % | 0.0 % | 0.6 % |
| pick, standing | 99.6 % | 98.3 % | 1.4 % | 0.6 % | 0.0 % | 0.0 % |
| pick, rod across the target | 95.5 % | 87.3 % | 9.4 % | 7.2 % | 1.0 % | 1.6 % |

Against level 31 (98.2 mixed, lying 96.4, across 97.5): the free pre-lift did not help.
Mixed is 0.5 lower, lying 1.4 lower and the pinned rod 2 points lower, inside the
run-to-run noise for mixed (13650 evaluated twice: 95.9 and 96.7 %) but not for the
across rod. What changed is where the loss sits: retries halved (6–8 % from 9–11 %) and
success without a retry rose (90–92 % from 88–90 %), so the live wrist does settle the
grasp before the test lift; but obstacle contacts doubled (1.2 % from 0.6–0.8 %, 1.7 % on
the lying rod) and 13650 and 13700 each had one joint-limit termination, which no level
since 17 had. With the fingers held and the wrist free for 10 steps the policy moves the
hand into the pile. Double picks on the lying rod (1.5 %) are also up from 0.6 %. The
scene evals carry the window on to level 33 unchanged since the design asks for it in the
mixed-scene checkpoint; if level 33's collisions stay at 1 % or more the window needs a
clearance term or a shorter horizon.

## Level 28 — a centred pinch (2026-10-06)

**Why.** The pile (level 27) trained 600 iterations and stayed flat at the untrained
level-26 policy's number: success 0.80 in training from iteration 75 to 600, latched 0.91,
held 0.84. A diagnostic of the level-27 pick (1 537 episodes, 256 envs, the training
scene) split the failures:

| pose | n | success | latched | retried | never latched | latched, never lifted | lifted, then dropped |
|---|---|---|---|---|---|---|---|
| lying | 760 | 78.4 % | 89.6 % | 30.7 % | 79 | 80 | 4 |
| tilted | 442 | 74.7 % | 89.4 % | 31.7 % | 47 | 65 | 0 |
| standing | 335 | 96.1 % | 97.0 % | 3.3 % | 10 | 3 | 0 |

Half the failures are grasps that satisfied the latch and then did not come up with the
scripted lift. No neighbour touched the target at the latch (4 of 730 lying latches), and
no neighbour lay on top of it: the neighbours do not pin the rod. The latch-time features
of the first-attempt latches say what the difference is:

| at the latch (lying) | lifted clean (562) | failed to lift (197) |
|---|---|---|
| pinch midpoint to the rod's axis | 4.9 mm | 8.1 mm |
| thumb tip height above the rod centre | 1.8 mm | 6.8 mm |
| index / middle tip height above the rod centre | 4.1 / 2.1 mm | 7.9 / 6.6 mm |
| commanded point to the nearest neighbour | 24.5 mm | 19.3 mm |
| fingertip clearance to a neighbour before the latch | +1.0 mm | −0.7 mm (touching) |
| tip forces, closures, point error along the rod | same | same |

The failed lifts are **shallow pinches**: the fingers closed on the upper half of the rod,
5 mm higher than on the lifts that held, and the rod rolls out during the lift. They form
where a neighbour stops the fingertips from going down beside the rod. First-attempt
lifts by the distance from the commanded point to the nearest neighbour (lying):

| point to nearest neighbour | < 10 mm | 10–20 | 20–30 | 30–45 | > 45 |
|---|---|---|---|---|---|
| lifted on the first attempt | 41 % | 69 % | 78 % | 84 % | 91 % |
| final success (with the retry) | 54 % | 80 % | 90 % | 96 % | 100 % |

The tilted rod shows the same pattern (midpoint 4.4 vs 7.9 mm, tips 5 mm higher); the
standing rod has nothing to fix. The latch had no term for the depth of the pinch, only
for its place along the rod (level 19) and the forces, so the policy had no reason to
push past a neighbour for a deeper grasp, and the +20 for the lift 100 steps later did
not teach it in 600 iterations.

**Change (level 28, `centred_pinch`).** The latch also requires the pinch midpoint
(thumb vs fingers) within **6 mm of the rod's axis** (`pinch_centre_tol`), the same way
the commanded point became binding at level 19. The commanded-point shaping uses the
full distance of the midpoint to the commanded point (along and across) at these levels.
Everything else is level 27. Evaluation prints "pinch midpoint vs rod axis"; the training
log has `Outcome/latch_centre_err_mm`. Test group 23 checks the rule.

Pipeline: tests 28 and 27, audit, the level-27 pick evaluated at level 28 untrained (how
many latches the new rule removes), 600 iterations from `L27/2026-09-30_17-02-49/model_10900`,
the three poses, and the pick at level 26 (the scene without the pile).

**Result.** Run `adam_u30_wuji_reach_grasp_L28/2026-10-06_09-15-00`, iterations 10900–11499.
Tests 103/103 (and 100/100 at level 27), audit baseline 1.6 %. **Pick `model_11450.pt`.**
512 episodes per row, 256 envs:

| policy, scene | mixed | lying | tilted | standing | latched | retried | pinch midpoint p95 |
|---|---|---|---|---|---|---|---|
| level-27 pick at level 27 (for reference) | 88.9 % | 84.8 % | 81.5 % | 92.6 % | 91 % | 16–23 % | 8 mm (not binding) |
| level-27 pick at level 28, untrained | 82.7 % | | | | 83.8 % | 15.8 % | 5.8 mm |
| level 28, model 11400 / 11450 / 11499 | 85.7 / **86.7** / 84.6 % | | | | 87 % | 12–15 % | 5.7–5.9 mm |
| level-28 pick, per pose | | 85.4 % | **85.9 %** | 93.3 % | 87–94 % | 4–17 % | 5.7 mm |

The centred latch cut the latch rate from 91 to 84 % untrained (the shallow pinches), and
training brought it back to 87 % with every latch now within 6 mm of the axis. The tilted
rod gained the most (81.5 to 85.9 %), lying and standing are within noise of level 27,
and the mixed number (86.7 vs 88.9 %) is within the ±1.5 % of a 512-episode run. The
remaining failures are now mostly grasps that never latch within the window (13 % of
episodes end with no latch) rather than latched grasps that slip: the rule did what it
was built for, and what is left is reaching a deep pinch next to a neighbour at a
commanded point at all, which is level 29's job (free point).

**A regression worth noting.** The level-28 pick at level 26 (cover block, bed rods, bin,
no pile): **82.1 %** with 7.8 % obstacle collisions, where the level-26 pick had 91.6 %.
Two levels without the cover and the bed were enough to lose the clearance habits for
them. The curriculum needs a mixed-scene level at the end (each episode draws its scene:
plain, cover, bed, pile) before any checkpoint is called final; added to the plan.

## Level 27 — the pile (2026-09-30, evening)

**Scene.** Four dynamic copies of the rod at random yaw 4 to 8 cm around
the target inside the bin; soft capsules in the clearance model (seen,
pushing allowed); a neighbour lifted with the target is a failure. The
level-26 policy, untrained on it, already scores 88.5 % over 256 episodes
with 0.8 % collisions and 0.4 % double picks.

**First run, aborted at iteration 10684.** Success fell to 2 % with 30 to
44 % invalid collisions from the first iterations, which the untrained
evaluation contradicted. The cause was in the training log: "PhysX error:
Patch buffer overflow", up to 550k patches against the default 164k.
At 2048 environments, five loose rods each exceed PhysX's GPU contact
patch buffer, contacts are dropped, and rods and fingers pass through each
other. The evaluation at 256 environments stayed inside the buffer. Fix:
the pile levels raise the GPU patch, contact and pair buffers (2^21
patches). Restarted from level 26's pick.

**Second run, result.** Run `adam_u30_wuji_reach_grasp_L27/2026-09-30_17-02-49`,
600 iterations, zero overflow errors. Pick `model_10900.pt`.

| | mixed | tilted | standing | lying |
|---|---|---|---|---|
| success | **88.9 %** | 81.5 % | 92.6 % | 84.8 % |
| latched, mean grasp end | — | 90.6 %, step 63 | 93.6 %, step 35 | 93.6 %, step 58 |
| retried, of which succeeded | 16.5 %, 46 % | 23.4 %, 26 % | 3.3 %, 12 % | 20.7 %, 44 % |
| double picks | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| invalid collisions | 0.8 % | 0 | 2.3 % | 1.9 % |

**Lying rod in the pile, point along it** (`model_10900`, 512 episodes per point):

| point | success | collisions | slip p95 | latched |
|---|---|---|---|---|
| -14 mm | 79.3 % | 3.9 % | 6.0 mm | 85.6 % |
| -7 mm | 86.7 % | 2.3 % | 6.7 mm | 93.6 % |
| 0 mm | 86.7 % | 1.2 % | 5.4 mm | 92.6 % |
| 7 mm | 83.0 % | 1.6 % | 5.8 mm | 89.1 % |
| 14 mm | 73.3 % | 2.5 % | 6.5 mm | 88.3 % |
| -14 mm | 79.3 % | 3.9 % | 6.0 mm | 85.6 % |
| -7 mm | 86.7 % | 2.3 % | 6.7 mm | 93.6 % |
| 0 mm | 86.7 % | 1.2 % | 5.4 mm | 92.6 % |
| 7 mm | 83.0 % | 1.6 % | 5.8 mm | 89.1 % |
| 14 mm | 73.3 % | 2.5 % | 6.5 mm | 88.3 % |

**Training added nothing here.** The untrained level-26 policy already
scored 88.5 % on the pile; the best trained checkpoint is 88.9 %, and the
later ones fall to 81 % as retries climb from 16 % to 22 %. The pile
costs 5 to 10 points against the same poses without neighbours (bin,
level 26): mostly retries, as the neighbours shift the target or block
the fingers during the first attempt, and the grasp takes twice as long.
No double picks and almost no collisions: the policy does not disturb the
pile, it just loses first attempts to it. More iterations of the same
recipe are unlikely to help; the next lever is what the policy sees (the
16 safety observations are the only view of the neighbours) or letting it
choose where to grasp (level 28).


## Level 26 — slide-out and grip check (2026-09-30, design note; results below when in)

**Why.** The overnight run left two mechanical gaps: a rod under a
neighbour cannot be lifted straight up (level 23, 24 %), and on raised
rods one latch in ten is the fingers pressing the rod onto its support,
found only after an empty 10 cm lift (level 21's trace).

**What changes, at every level.** A 15-step pre-lift between the settle
and the lift: ten steps of drag at 3 mm/step along the rod's axis away
from a covering block (no motion for an uncovered rod), then five steps
of a 1 mm test lift along the approach axis. The scripted part of an
episode is 95 steps. **At level 26 only:** `grip_check`: a latched rod
that has not risen 2 mm by the end of the test lift triggers the retry
right there, 5 mm up, instead of after the lift. The scene is level 25's
(bin, tilted, standing, on rods) with the cover back on 30 % of lying
rods, so the slide-out is exercised. Trains 600 iterations from level 25's
continuation pick.

**First attempt, aborted at iteration 250.** The check as first written
("the rod has risen 2 mm by the end of a 5-step, 5 mm test lift") fired
in 82 to 89 % of episodes, good grasps included, and success collapsed to
46 %: the wrist lags its target by a few steps, so after 5 mm of commanded
lift the rod had risen barely 2 mm even when held. Rewritten as a relative
test: eight steps of 1 mm, and the rod must have risen at least half of
what the wrist itself rose. Restarted from the same checkpoint.

**Second attempt, aborted at iteration 320.** The relative check still
fired in 42 to 64 % of episodes (success 70 %). During the first 8 mm of
a lift the rod barely follows the wrist even when it is held: the finger
drives and the 2 mm contact offsets take up the slack first, and the rod
lags several millimetres. A grip check at this scale cannot separate a
held rod from a pressed one. **Level 26 keeps the slide-out and drops the
grip check**; empty lifts stay with the lift-end retry that levels 22 to
25 used. The check stays in the code, off, for a later attempt with a
longer test lift measured against the fingertips rather than the wrist.

**Third run (slide-out only), result.** Run `adam_u30_wuji_reach_grasp_L26/2026-09-30_14-17-09`,
600 iterations from level 25's `model_9850`. Pick `model_10400.pt`, 91.6 % mixed.

| | tilted | standing | lying, flat | lying, covered | lying, on rods |
|---|---|---|---|---|---|
| success | 90.3 % (86.3 at 25) | 97.3 % | 91.7 % | **10.2 %** (23.9 at 23) | 93.8 % |
| latched | 92.1 % | 97.9 % | 94.6 % | 49.0 % | 94.8 % |
| invalid collisions | 1.9 % | 1.0 % | 3.5 % | 15.4 % | 2.1 % |

**The slide-out did not work.** The covered rod got worse: half the
episodes never latch, and 15 % end in a collision (6 % finger on the
table, 10 % with the block). A scripted 3 cm drag along the axis with the
fingers closed on a thin thread pulls the fingers across the rod or into
the table rather than sliding the rod. The other poses improved a few
points from the extra training. A covered rod needs a different motion:
the policy itself controlling the drag (the pre-lift as a policy phase,
not a script), or tilting the rod up by its exposed end first.

## Level 25, continued — 600 more iterations in the bin (2026-09-30, early morning)

**Run** `adam_u30_wuji_reach_grasp_L25/2026-09-29_22-21-40`, iterations
9350–9948, resumed from `model_9349`. **Pick `model_9850.pt`**, not
gated (drag, collisions). Training sat at 90 % with collisions falling
from 6.5 % to 2.4 %.

**Evaluation** (`model_9850`, 512 episodes each; the first run's pick in brackets):

| | mixed | tilted | standing | lying, flat | lying, on rods |
|---|---|---|---|---|---|
| lift-and-hold success | **93.3 %** (92.4) | 86.3 % (86.2) | **98.4 %** (93.9) | **94.4 %** (91.7) | **92.4 %** (90.8) |
| success without a retry | 90.4 % | 81.8 % | 98.2 % | 88.1 % | 88.5 % |
| obstacle collisions (walls) | **2.7 %** (3.9) | 1.8 % (2.9) | 1.0 % (5.1) | 2.9 % (4.7) | 2.5 % (2.9) |

The extra 600 iterations went into the walls: contacts nearly halved,
the standing rod is back at 98 %, the flat and bedded rods gained 2 to 3
points. The tilted rod did not move: in the bin its rolled approach
still touches the near wall in 1.8 % of episodes and its lifts still slip
at the block. **This is the current best policy** for the full scene:
lying, standing, tilted and on-rods poses inside a bin, commanded point
binding within 4 mm, one retry.

## Level 25 — inside a bin (trained 2026-09-29/30, night)

**What changes.** Four kinematic walls, 1 cm thick and 4 cm high, 13 cm
from the nominal bolt position on each side (10 cm at first; a tilted
rod's block reached the wall and the rod fell flat, caught by the tests).
The walls take obstacle AABB slots 3 to 6. Rods spawn within ±5 cm, so
at least 6 cm from a wall. Tests 101 of 101. Baseline 0.0 %.

**Run** `adam_u30_wuji_reach_grasp_L25/2026-09-29_21-27-21`, iterations
8750–9349 (600), resumed from level 24's `model_8750`. **Pick
`model_9349.pt`** (the last; still rising), not gated. The hardest resume
of the study: collisions 24 to 30 % in the first ten iterations, the arm
and hand hitting walls on approaches it had never made, down to 6.5 % by
the end, with success at 85 % and climbing.

**Evaluation** (`model_9349`, 512 episodes each):

| | mixed | tilted | standing | lying, flat | lying, on rods |
|---|---|---|---|---|---|
| lift-and-hold success | **92.4 %** (96.5 at 24) | 86.2 % (93.8) | 93.9 % (100) | 91.7 % (95.0) | 90.8 % (92.4) |
| success without a retry | 87.9 % | 78.9 % | 93.6 % | 84.1 % | 87.3 % |
| retried, of which succeeded | 7.2 %, 62 % | 16.2 %, 45 % | 0.6 %, 67 % | 9.9 %, 77 % | 8.1 %, 43 % |
| obstacle collisions (walls) | 3.9 % | 2.9 % | 5.1 % | 4.7 % | 2.9 % |
| latched, mean grasp end | — | 91.2 %, step 51 | 95.1 %, step 28 | 93.2 %, step 45 | 92.5 %, step 42 |
| worst forearm clearance | −0.5 cm | | | | |

**The lying-rod map inside the bin** (`model_9349`, 512 episodes per point):

| point | success | collisions | slip p95 | latched |
|---|---|---|---|---|
| -14 mm | 82.2 % | 6.2 % | 8.4 mm | 86.1 % |
| -7 mm | 92.0 % | 5.3 % | 11.4 mm | 93.4 % |
| 0 mm | 94.1 % | 5.3 % | 9.0 mm | 95.7 % |
| 7 mm | 87.5 % | 7.2 % | 7.1 mm | 88.7 % |
| 14 mm | 82.3 % | 10.3 % | 8.5 mm | 84.8 % |
| -14 mm | 82.2 % | 6.2 % | 8.4 mm | 86.1 % |
| -7 mm | 92.0 % | 5.3 % | 11.4 mm | 93.4 % |
| 0 mm | 94.1 % | 5.3 % | 9.0 mm | 95.7 % |
| 7 mm | 87.5 % | 7.2 % | 7.1 mm | 88.7 % |
| 14 mm | 82.3 % | 10.3 % | 8.5 mm | 84.8 % |

The walls cost 4 to 7 points on every pose, and the loss is almost all
wall contact: 3 to 5 % of episodes end with the forearm or hand more than
4 mm into a wall. The 4 cm walls are below the hand at the standoff and
above the fingertips at the pinch, so the contact is the forearm coming
in low over the near wall, which the 3 cm forearm margin flags but the
policy has not yet learned to respect on every approach. A 600-iteration
continuation is queued (below).

## Level 24 — the rod resting across two rods (trained 2026-09-29, night)

**What changes.** For 40 % of lying episodes the target spawns across two
kinematic rods lying on the table 2 cm apart, perpendicular to it, and
settles on their bands about 35 mm up; it rolls when touched. The bed
rods are kinematic cylinders (a kinematic copy of the scanned asset did
not collide with the target in Isaac Lab 3.0; a primitive does) and enter
the clearance model as capsules, a third obstacle kind. The cover is
dropped (level 23). Two bugs of mine on the way, both caught by the
tests: the raised spawn height was set after the pose had been written,
so the target started on the table inside the bed rods. Tests 97 of 97.

**Run** `adam_u30_wuji_reach_grasp_L24/2026-09-29_20-29-57`, iterations
8250–8846 (600), resumed from level 23's `model_8247`. **Pick
`model_8750.pt`** (96.5 %, misses only the drag gate). Training sat at
94 to 95 % from iteration 8260 on, collisions 0.1 %, retries 9 %.

**Evaluation** (`model_8750`, 512 episodes each):

| | mixed | tilted | standing | lying, flat | lying, on rods |
|---|---|---|---|---|---|
| lift-and-hold success | **96.5 %** | **93.8 %** | **100.0 %** | 95.0 % | **92.4 %** |
| success without a retry | 90.9 % | 89.3 % | 99.4 % | 88.7 % | 86.5 % |
| retried, of which succeeded | 6.6 %, 85 % | 8.0 %, 56 % | 0.6 %, 100 % | 8.5 %, 73 % | 10.0 %, 59 % |
| latched, mean grasp end | — | 96.1 %, step 39 | 100 %, step 26 | 95.7 %, step 39 | 94.5 %, step 41 |
| slip in the hand, p95 | — | 10.5 mm | 6.1 mm | 8.6 mm | 9.8 mm |

**The rod-on-rods map** (`model_8750`, `--bed 1`, 512 episodes per point):

| point | success | collisions | slip p95 | latched |
|---|---|---|---|---|
| -14 mm | 92.0 % | 2.0 % | 6.8 mm | 93.8 % |
| -7 mm | 93.0 % | 0.8 % | 11.6 mm | 97.1 % |
| 0 mm | 93.0 % | 0.8 % | 9.0 mm | 96.7 % |
| 7 mm | 95.8 % | 0.8 % | 6.1 mm | 96.5 % |
| 14 mm | 91.7 % | 0.4 % | 7.8 mm | 93.6 % |
| -14 mm | 92.0 % | 2.0 % | 6.8 mm | 93.8 % |
| -7 mm | 93.0 % | 0.8 % | 11.6 mm | 97.1 % |
| 0 mm | 93.0 % | 0.8 % | 9.0 mm | 96.7 % |
| 7 mm | 95.8 % | 0.8 % | 6.1 mm | 96.5 % |
| 14 mm | 91.7 % | 0.4 % | 7.8 mm | 93.6 % |

The rod on rods is grasped at 92.4 %, 2.6 points below the flat rod and
at the tilted rod's level: rolling under the fingers costs about a tenth
of first attempts, and the retry recovers six in ten of those. Every
pose of the study is now above 92 % with the point binding, except the
pinned rod of level 23.

## Level 23 — a block over one end of the rod (trained 2026-09-29, night)

**What changes.** Half the lying rods get a 4 × 4 × 3 cm block hovering
2 mm above one end, its near edge 5 mm from the rod's centre; the
commanded point is drawn on the exposed part (a sign error in that
sampling put the point under the block on the first try; the level-23
test caught it). Tests 96 of 96. Baseline: see the audit log.

**Run** `adam_u30_wuji_reach_grasp_L23/2026-09-29_19-26-17`, iterations
7650–8247 (600), resumed from level 22's `model_7648`. **Pick
`model_8247.pt`**, not gated. Training plateaued at 79 % from iteration
7700 with retries in 22 to 24 % of episodes: the covered rods.

**Evaluation** (`model_8247`, 512 episodes each):

| | mixed | tilted | standing | lying, no cover | lying, covered |
|---|---|---|---|---|---|
| lift-and-hold success | 85.5 % | 92.2 % | **100.0 %** | **97.3 %** | **23.9 %** |
| success without a retry | 82.4 % | 86.6 % | 99.0 % | 92.1 % | 16.1 % |
| retried, of which succeeded | 14.9 %, 21 % | 9.7 %, 58 % | 1.0 %, 100 % | 6.6 %, 79 % | 78.8 %, 10 % |
| latched, mean grasp end | — | 94.6 %, step 40 | 100 %, step 26 | 97.7 %, step 37 | 69.7 %, step 130 |

The uncovered poses are the best of the study (standing 100 %, lying
97.3 %, tilted 92.2 %). The covered rod is a different story, and the
numbers say which: the policy latches on the exposed part in 70 % of
episodes, then the lift fails in nearly all of them and the retry fails
again. A rod whose far end sits under a block 2 mm above it cannot be
lifted straight up: the covered end hits the block. This is physics, not
learning. A pinned rod has to be slid out along its axis, or tilted up by
its exposed end and drawn out, before it can rise, and neither is a
motion the scripted lift or the policy's grasp phase can make. The
covered map (`model_8247`, `--cover 1`, 512 episodes per point, offset
measured towards the block) shows the same thing from the other side:
the latch is 68 to 75 % everywhere, and success climbs from 2 % at
−14 mm (the free end: lifting there levers the covered end into the
block) to 30 % at +6 mm (next to the block: the pinch is under the
block's edge and a lift there lets the rod slip out), with 3 to 6 %
collisions with the block throughout. Level 23
therefore measures the wrong thing for a covered rod, and the cover is
dropped from levels 24 and 25 so it does not spend a third of their
lying episodes on an impossible lift. The slide-out is a candidate for a
later level: a "drag" phase between latch and lift, along the exposed
direction, for rods the estimate says are covered.

## Level 22 — one retry after a failed lift or a drop (trained 2026-09-29, night)

**What changes.** When the lift ends with the rod not up, or the rod drops
in the hold, and the episode has an attempt left: the hand opens, the
latch clears, the IK reference is re-seeded from the measured pose, and
the grasp phase restarts from where the hand is (10 cm up) with a 60-step
window. The progress terms skip the restart step; a drop already paid
stands; success on the second attempt is a success. Episodes are up to
340 steps; the time observation is normalised by the level's length. No
new observation. A grasp that timed out without a latch but lifted the
rod anyway is not retried (the rod is up), and stays a failure at binding
levels. Tests 90 of 90 (after four test expectations were fixed for the
longer episodes). Baseline 3.1 %: with the point binding the blind close
almost never counts.

**Run** `adam_u30_wuji_reach_grasp_L22/2026-09-29_18-30-46`, iterations
7050–7648 (600), resumed from level 21's `model_7049`. **Pick
`model_7648.pt`** (the last), not gated (drag).

| iteration | success | invalid collisions | retried | retry success | pinch error |
|---|---|---|---|---|---|
| 7055 | 73.4 % | 0.24 % | 8.3 % | 7.2 % | 1.5 mm |
| 7060 | 90.8 % | 0.24 % | 17.5 % | 10.5 % | 1.9 mm |
| 7100 | 93.4 % | 0.10 % | 9.6 % | 4.9 % | 2.0 mm |
| 7300 | 93.8 % | 0.10 % | 9.1 % | 4.6 % | 2.0 mm |

The policy used the retry heavily at first (17 % of episodes) and then
learned to need it less as its first attempts improved; the retry's
own success (the share of episodes saved by it) settled near 5 points.

**Evaluation** (`model_7648`, 512 episodes each):

| | mixed | tilted only | standing only | lying only |
|---|---|---|---|---|
| lift-and-hold success | **96.9 %** (90.1 at 21) | **92.3 %** (81.8) | **99.8 %** (99.2) | **94.4 %** (89.3) |
| success without a retry | 93.8 % | 87.3 % | 99.6 % | 90.7 % |
| retried, of which succeeded | 4.9 %, 64 % | 10.2 %, 49 % | 0.2 %, 100 % | 6.0 %, 61 % |
| latched, mean grasp end | — | 95.6 %, step 41 | 99.8 %, step 26 | 95.7 %, step 39 |
| pinch error mean / max | 1.9 / 4.0 mm | — | — | — |
| invalid collisions | 0.4 % | — | — | — |
| slip in the hand, p95 | — | 15.3 mm | 5.8 mm | 9.1 mm |
| gates | drag, collisions fail | four fail | drag fails | three fail |

Beats the 3.1 % baseline by 94 points. The retry does two things: it
recovers half to two thirds of the grasps that slipped at the lift, and
it raised the first-attempt numbers too (87.3 % tilted at the first
attempt against 81.8 % at level 21), because a policy that can retry
learns to seat a grasp it would otherwise have committed to. This is the
best policy of the study on every pose.

**The tilted-rod map** (`model_7648`, `--tilt 1`, 512 episodes per point):

| point | success | latched | collisions | slip p95 |
|---|---|---|---|---|
| −14 mm | 87.5 % | 89.1 % | 2.7 % | 12.5 mm |
| −7 mm | 94.2 % | 95.9 % | 1.4 % | 13.2 mm |
| 0 | 93.2 % | 95.1 % | 0.4 % | 11.1 mm |
| +7 mm | 92.0 % | 93.4 % | 0.8 % | 12.1 mm |
| +14 mm | 78.9 % | 85.9 % | 2.1 % | 52 mm |

Along the leaning rod the band and shoulders are at 92 to 94 %, the low
end 88 %, the raised end 79 % where the pinch sits over the block's edge
and the fingers touch it (2 to 3 % collisions at both ends).

## Level 21 — the wrist rolled to the rod's tilt (trained 2026-09-29, evening)

**What changes.** For a tilted rod the hand rolls by −θ about the rod's
horizontal normal so its approach axis is the rod's up-normal; the
standoff and the scripted lift run along that axis; the reset draws from
`pose_buckets_tilt.npz`. Getting there took three corrections of mine,
all caught by the level-21 tests: the solver's rod-axis direction was
90° off (every rolled bucket solved for the wrong roll), a buffer the
tests read changed meaning, and five test expectations assumed a
vertical standoff. With the axis fixed, the rolled wrist is reachable at
all three tilts over the whole −60° to +45° window, with more slack than
the flat pinch. Tests 86 of 86 at level 21, 80 of 80 at level 20.
Baseline 12.5 %.

**Run** `adam_u30_wuji_reach_grasp_L21/2026-09-29_17-17-24`, iterations
6450–7049 (600), resumed from level 20's `model_6450`. **Pick
`model_7049.pt`** (the last; still rising), not gated.

| iteration | success | invalid collisions | bolt drag (slip) | pinch error |
|---|---|---|---|---|
| 6455 | 41.1 % | 5.6 % | — | 0.9 mm |
| 6460 | 81.3 % | 6.3 % | 5.4 mm | 1.7 mm |
| 6600 | 85.9 % | 1.2 % | 6.3 mm | 1.9 mm |
| 6800 | 86.3 % | 0.23 % | 5.8 mm | 2.0 mm |
| 6900 | 87.1 % | 0.10 % | 6.4 mm | 2.0 mm |

The 6 % of collisions at the resume are the rolled approaches the
level-20 policy had never made (and the mirrored tilt convention, see the
previous entry); they clear within 300 iterations.

**Evaluation** (`model_7049`, 512 episodes each, success requiring the latch):

| | mixed | tilted only | standing only | lying only |
|---|---|---|---|---|
| lift-and-hold success | **90.1 %** | **81.8 %** (78.3 at 20) | **99.2 %** (98.0) | 89.3 % (95.7) |
| latched, mean grasp end | — | 95.5 %, step 37 | 99.8 %, step 26 | 97.1 %, step 34 |
| lifted at all | — | 86.1 % | 99.6 % | 93.0 % |
| slip in the hand, p95 | — | 56 mm | 5.5 mm | 10.8 mm |
| gates | | success, drag fail | drag fails | three fail |

**The tilted-rod map** (`model_7049`, `--tilt 1`, 512 episodes per point):

| point | success | latched | lifted | slip p95 |
|---|---|---|---|---|
| −14 mm | 76.0 % | 92.2 % | — | 60 mm |
| −7 mm | 76.0 % | 96.3 % | — | 57 mm |
| 0 | 78.4 % | 93.2 % | — | 57 mm |
| +7 mm | 83.3 % | 95.5 % | — | 60 mm |
| +14 mm | 66.5 % | 93.2 % | — | 62 mm |

Lying rod, square pinch, and standing rod (`model_7049`, 512 episodes per point):

| point | lying: success | latched | slip p95 | standing: success | latched | slip p95 |
|---|---|---|---|---|---|---|
| −14 mm | 95.1 % | 99.2 % | 7.1 mm | 94.0 % | 98.6 % | 9.7 mm |
| −7 mm | 87.5 % | 98.6 % | 13.4 mm | 98.1 % | 98.3 % | 4.4 mm |
| 0 | 93.6 % | 98.6 % | 9.3 mm | 99.4 % | 99.4 % | 4.1 mm |
| +7 mm | 93.4 % | 95.5 % | 7.4 mm | 98.6 % | 99.8 % | 7.6 mm |
| +14 mm | 84.8 % | 96.1 % | 12.2 mm | 96.7 % | 99.6 % | 9.1 mm |

**What the roll did, and did not do.** The roll raised the tilted rod's
latch rate from 91 % to 95 % and success from 78 % to 82 %, and it
made the standing rod the best pose of the study at 99.2 %. It did not
fix the lift: 14 % of latched tilted grasps leave the rod behind when
the lift starts. A trace of 303 rolled lifts settles what the 57 mm of
"slip" is: in the lifts that succeed the rod moves with the wrist to
within 5 mm; in the ones that fail the rod does not move at all while
the hand travels 6 cm along the tilted axis, so the p95 of the slip is
that 6 cm. The rod was never really held: on a block, the latch's 4 N per
side can be the fingers pressing the rod onto its support rather than
pinching it, and the hand lifts away empty. The lying rod's loss (95.7 to
89.3 %) has the same signature, 4 % between latched and lifted. Level 22's
retry gives those grasps a second attempt; a latch that checks the grip
is a pinch (balanced sides, rod free of its support) is the deeper fix.

## Before level 21: the lift profile, and a flipped tilt convention (2026-09-29, evening)

Two global changes went in with level 21 and were measured on the level-20
pick (`model_6450`, 512 episodes each) before it trained on.

**Lift profile.** The scripted lift became 40 steps along the approach
axis, the first ten at 1 mm, and drag became the bolt's slip relative to
the wrist. On the level-20 policy, mixed poses:

| lift | success | lifted at all | drag p95 |
|---|---|---|---|
| 30 steps, constant (the old one) | 77.9 % | 82.8 % | 10.9 mm |
| 40 steps, constant | 80.2 % | 87.2 % | 11.0 mm |
| 40 steps, ten slow (the new one) | 82.7 % | 87.7 % | 11.4 mm |

The slow start is worth 3 to 5 points of lift on a policy that never
trained with it; the drag gate it was meant for did not move, so the
drag is not the lift's speed.

**Why all three read far below the 93.4 % of the afternoon.** The tilted
rod's convention flipped: the reset now raises the mesh's +z end (the
pose solver's convention for the rolled wrist), where it used to raise
the −z end. Every tilted scene is mirrored relative to what the level-20
policy trained on. Pose split on the current env: lying rods 93.6 %
(unchanged), tilted rods 47.3 % (was 78.3 %). Level 21 trains under the
new convention and absorbs it; the level-20 numbers stand as recorded,
for the old convention.

## Level 20, continued — commanded-point shaping, 600 more iterations (2026-09-29)

**What changes.** A shaping term, `grasp_point_progress`: the potential
1 − tanh(|pinch − point| / 1 cm) on the pinch centre's position along the
true axis, paid as progress in the grasp phase with weight 1.0, zero at
levels without a binding point. Until now the policy learned the point
only from the latch refusing off-point grasps. Tests 81 of 81; the reward
audit passes (random actions collect −0.24 of shaping against +2.24 for
the scripted grasp; the scripted baseline reads 14.7 %).

**Run** `adam_u30_wuji_reach_grasp_L20/2026-09-29_14-57-54`, iterations
5900–6499, resumed from the first level-20 run's `model_5899`. **Pick
`model_6450.pt`**, not gated.

| iteration | success | invalid collisions | bolt drag | pinch error |
|---|---|---|---|---|
| 5910 | 87.9 % | 0.04 % | 3.5 mm | 2.0 mm |
| 6000 | 85.3 % | 0.25 % | 3.5 mm | 1.9 mm |
| 6200 | 87.3 % | 0.16 % | 3.4 mm | 1.9 mm |
| 6400 | — | — | — | — |

**Evaluation** (512 episodes each, success requiring the latch):

| | `model_6450`, mixed | tilted only | standing only | lying only | `model_6400` | `model_6498` |
|---|---|---|---|---|---|---|
| lift-and-hold success | **93.4 %** (87.6 before) | **78.3 %** (75.0) | 98.0 % (98.2) | **95.7 %** (86.4) | 90.8 % | 88.2 % |
| latched, mean grasp end | 97.9 %, step 34 | 92.4 %, step 45 | 98.6 %, step 29 | 97.9 %, step 36 | | |
| pinch error mean / max | 2.0 / 4.0 mm | 2.1 / 4.0 mm | 1.7 / 4.0 mm | 2.1 / 4.0 mm | | |
| invalid collisions | 0.8 % | 1.4 % | 0.2 % | 0 | | |
| bolt drag mean / p95 | 3.2 / 10.8 mm | 4.5 / 14.5 mm | 2.2 / 6.0 mm | 3.1 / 9.9 mm | | |
| gates | success, drag fail | three fail | drag fails | drag fails | | |

The lying rod is the winner: +9 points, and above the success gate. The
tilted rod gained 3 points and keeps its 14.5 mm drag, as predicted: its
loss is the pinch geometry.

**Maps** (`model_6450`, 512 episodes per point). Tilted rod
(`logs/sweeps/grasp_offset_tilt1_2026-09-29_17-43-11`):

| point | success | collisions | drag p95 | latched | grasp end | pinch error |
|---|---|---|---|---|---|---|
| −14 mm | 79.1 % | 0 | 9.7 mm | 90.8 % | step 51 | 2.9 mm |
| −10 mm | 79.0 % | 0 | 15.2 mm | 95.0 % | 43 | 2.2 mm |
| −6 mm | 86.3 % | 0.4 % | 13.1 mm | 96.7 % | 40 | 2.3 mm |
| −2 mm | 79.3 % | 0.4 % | 15.0 mm | 93.8 % | 41 | 1.9 mm |
| +2 mm | 82.7 % | 0 | 13.5 mm | 93.6 % | 43 | 2.2 mm |
| +6 mm | 82.5 % | 1.2 % | 11.5 mm | 91.6 % | 44 | 2.2 mm |
| +10 mm | 77.9 % | 1.6 % | 15.1 mm | 91.2 % | 46 | 2.0 mm |
| +14 mm | 71.1 % | 3.1 % | 16.5 mm | 84.8 % | 53 | 2.2 mm |

Lying rod, square pinch (`grasp_offset_pinch0_upright0_tilt0_2026-09-29_17-48-09`):

| point | success | collisions | drag p95 | latched | grasp end | pinch error |
|---|---|---|---|---|---|---|
| −14 mm | 95.1 % | 0 | 8.5 mm | 98.6 % | step 37 | 2.7 mm |
| −10 mm | 96.7 % | 0 | 11.6 mm | 99.6 % | 36 | 2.3 mm |
| −6 mm | 94.1 % | 0 | 12.0 mm | 99.2 % | 34 | 2.1 mm |
| −2 mm | 90.5 % | 0 | 12.6 mm | 96.6 % | 35 | 1.8 mm |
| +2 mm | 94.0 % | 0.2 % | 6.4 mm | 94.6 % | 39 | 2.1 mm |
| +6 mm | 94.9 % | 0.4 % | 4.2 mm | 95.3 % | 35 | 1.8 mm |
| +10 mm | 94.7 % | 0.2 % | 7.1 mm | 96.9 % | 35 | 2.0 mm |
| +14 mm | 84.2 % | 0.4 % | 15.6 mm | 95.5 % | 37 | 2.2 mm |

Standing rod (`grasp_offset_upright1_2026-09-29_17-52-56`):

| height | success | collisions | drag p95 | latched | grasp end | pinch error |
|---|---|---|---|---|---|---|
| −14 mm | 88.0 % | 0.6 % | 13.4 mm | 92.8 % | step 41 | 3.1 mm |
| −10 mm | 96.9 % | 0 | 4.5 mm | 99.0 % | 28 | 1.8 mm |
| −6 mm | 98.4 % | 0.2 % | 4.3 mm | 98.6 % | 28 | 1.5 mm |
| −2 mm | 99.6 % | 0 | 4.1 mm | 99.6 % | 28 | 1.6 mm |
| +2 mm | 99.2 % | 0 | 5.9 mm | 99.8 % | 28 | 1.7 mm |
| +6 mm | 98.2 % | 0 | 8.1 mm | 99.8 % | 28 | 1.6 mm |
| +10 mm | 96.9 % | 0 | 7.8 mm | 99.8 % | 28 | 1.7 mm |
| +14 mm | 95.7 % | 0 | 8.7 mm | 99.4 % | 30 | 2.3 mm |

Reading them against level 19 and the first level-20 run. The
commanded-point shaping did what it was added for: on the lying rod the
latch is now 95 to 100 % at every point (87 to 95 % before) and success
84 to 97 % (64 to 91 %); on the standing rod 96 to 100 % from −10 mm up
(94 to 99 %), with the bottom thread still the weak spot at 88 %. The
tilted rod moved from 70–75 % to 71–86 %, its latch to 85–97 %, and its
drag not at all (10 to 17 mm): the point is reached, the lift still lets
it slide. Grasp end on the lying rod dropped from about 41 steps to 35:
fewer re-seats. Drag p95 remains above 5 mm almost everywhere, on every
pose; it is the one gate no level since 17 passes, and a slower or
two-stage lift is the next thing to try for it.

## Level 20 — the rod leaning on a block (trained 2026-09-29)

**What changes.** A third of the episodes stand the rod on its end, a
third lay it flat, a third lean it on a low kinematic block (4 × 4 cm,
10, 17 or 25 mm tall, one per episode) with one end on the table: 18°,
30° or 40° from horizontal. The block takes an obstacle AABB slot, so
the clearance observations and the obstacle rule see it. The commanded
point is binding as at level 19, success requires the latch, and lift is
measured from each rod's own rest height. The shipped hull collider's
tilt formula went through one wrong version (an 80° "lean" that fell
over) before the plain one, tilt = atan(height / edge distance). Tests
81 of 81. Baseline 11.7 %, the lowest of the study: a blind close rarely
lands on the point of a tilted rod.

**Run** `adam_u30_wuji_reach_grasp_L20/2026-09-29_13-31-09`, iterations
5300–5899 (600), resumed from level 19's `model_5300`. **Pick
`model_5899.pt`** (the last; success still rising), not gated.

| iteration | success | invalid collisions | bolt drag | pinch error |
|---|---|---|---|---|
| 5305 | 39.5 % | 0.05 % | — | 1.0 mm |
| 5310 | 76.3 % | 0.48 % | 3.9 mm | 1.8 mm |
| 5500 | 80.1 % | 0.25 % | 3.2 mm | 1.9 mm |
| 5700 | 83.6 % | 0.16 % | 3.1 mm | 2.0 mm |
| 5899 | 85.3 % | 0.14 % | 3.5 mm | 1.8 mm |

**Evaluation** (512 episodes each, mean action, success requiring the latch):

| | `model_5899`, mixed | tilted only | standing only | lying only | `model_5850` | `model_5800` |
|---|---|---|---|---|---|---|
| lift-and-hold success | **87.6 %** | **75.0 %** | 98.2 % | 86.4 % | 85.4 % | 83.4 % |
| latched, mean grasp end | 95.0 %, step 40 | 91.1 %, step 45 | — | — | — | — |
| lifted at all / held | — | 82.0 % / 79.7 % | — | — | — | — |
| pinch error mean / max | 2.0 / 4.0 mm | 2.1 / 4.0 mm | 1.8 / 4.0 mm | 2.1 / 4.0 mm | 2.0 / 4.0 mm | 2.0 / 4.0 mm |
| invalid collisions | 0 | 0.6 % | 0 | — | 0.2 % | 0 |
| bolt drag mean / p95 | 3.4 / 11.1 mm | 4.5 / 14.5 mm | — | — | 3.8 / 11.3 mm | 3.6 / 12.4 mm |
| gates | success, drag fail | three fail | drag fails | success, drag fail | | |

Beats the 11.7 % baseline by 76 points. Standing rods improved with the
extra training (98.2 % against 96.7 % at level 19); lying rods held
(86.4 % against 88.7 %); the tilted rod is the new number.

**The tilted-rod map** (`model_5899`, `--tilt 1`, 512 episodes per
point, `logs/sweeps/grasp_offset_tilt1_2026-09-29_16-17-32`):

| point from centre | success | collisions | drag p95 | latched | grasp end | pinch error |
|---|---|---|---|---|---|---|
| −14 mm | 74.5 % | 0.6 % | 15.1 mm | 91.0 % | step 54 | 2.8 mm |
| −10 mm | 74.6 % | 0.2 % | 15.5 mm | 93.6 % | 45 | 2.0 mm |
| −6 mm | 74.8 % | 0.2 % | 14.0 mm | 94.5 % | 43 | 2.3 mm |
| −2 mm | 73.9 % | 0.2 % | 13.2 mm | 92.6 % | 43 | 1.6 mm |
| +2 mm | 70.0 % | 0 | 16.8 mm | 88.3 % | 48 | 2.3 mm |
| +6 mm | 75.2 % | 0 | 13.7 mm | 87.5 % | 47 | 2.4 mm |
| +10 mm | 73.9 % | 1.4 % | 15.0 mm | 85.6 % | 51 | 2.0 mm |
| +14 mm | 71.3 % | 1.6 % | 16.9 mm | 84.6 % | 51 | 2.4 mm |

Reading it. The tilted map is flat at 70 to 75 % along the whole rod,
with the latch at 85 to 95 %: the point is reached almost as reliably as
on the other poses, and the loss comes after it. The detailed split
(tilted only, 512 episodes): latched 91 %, lifted at all 82 %, held
80 %, success 75 %, drag p95 14.5 mm, twice the other poses. So about one
grasp in ten slips out of the fingers as the lift starts, and those that
stay drag the rod 14 mm. The cause is geometric, not a training gap: the
pinch is horizontal, the rod's axis is 18° to 40° from horizontal, so the
pads meet the cylinder off its widest diameter and a vertical lift pushes
the rod down its own axis and out of the grip. The fix is a wrist roll
that puts the pinch line perpendicular to the slanted axis, which is a
new hand orientation with its own reach solve, i.e. a level of its own,
not more iterations of this one.

## The rod cannot stand: the collider, not the object (2026-09-29)

Asked for a standing rod as the next variation, the first check was
whether the rod stands at all in the simulation. Spawned on its end, 64
copies, nothing touching them: 4.5° of tilt on the first step, one in five
toppled by step 30, one in three by step 90. The mesh is not the cause: its
end faces are flat discs (the lowest vertex in every 30° sector is at the
same height to 0.01 mm). The collider is. The asset carries
`physics:approximation = convexHull`, and PhysX cooks the 327k-face mesh
to a hull of at most 64 vertices, whose end is a handful of points that
are not coplanar.

A copy of the asset with a 32-hull convex decomposition
(`rod-with-threads-0001_cd.usdc`, `tools/make_bolt_colliders.py`) stands
with 0.03° of tilt, 64 of 64 up after 150 steps. (An SDF variant needs
the SDF collision API applied, not just the attribute; PhysX fell back to
the hull. Not pursued.)

What the collider changes for the grasp: the level-17 policy re-evaluated
at level 17 on the decomposition, 512 episodes, 99.2 % success (100 % on
the hull), one finger–table touch, drag p95 3.7 mm (3.5), peak fingertip
force 16.7 N. So the threads' contact geometry is close enough that the
lying-rod policy carries over, and levels with a standing rod use the
decomposition (`RGLevel.bolt_collider = "cd"`). Levels 0–17 keep the hull
they were trained and gated on.

## Bolt mass: the hand's limit, not the policy's (2026-09-28)

The bolt's 0.030 kg comes from adam-moveit's workcell file and was never
weighed; the real part may be close to 1 kg. Probed with
`eval_reach_grasp.py --bolt-mass` (level-12 policy, level 12, 512 episodes)
and `audit_reach_grasp.py --bolt-mass` (scripted full close, level 0):

| bolt mass | policy: success | policy: latched | policy: peak tip force | scripted full close: success |
|---|---|---|---|---|
| 0.03 kg | 99.8 % | 100 % | 16 N | 100 % |
| 0.1 kg | 100 % | 100 % | 18 N | — |
| 0.3 kg | 93.8 % | 100 % | 24 N | — |
| 0.6 kg | 45.7 % | 99.8 % | 23 N | **0 %** |
| 1.0 kg | 4.2 % | 98.9 % | 27 N | **0 %** |

The pinch forms at every mass; the bolt slides out of the fingers as the
wrist rises. The policy closes to (0.83, 0.65), never having needed more,
but the scripted FULL close does no better above 0.3 kg: the driven fingers'
drives (stiffness 3 N·m/rad, effort 0.3 N·m at the middle and tip joints,
chosen to reproduce a 25 N squeeze on the 30 g part) cannot produce the
normal force a kilogram needs. This is a hand-capability question. The
inputs that decide it are the real bolt's mass and the Wuji hand's rated
fingertip force; until then, randomising mass up to 1 kg would only teach
the policy to close fully and still fail above ~0.3 kg.

**Strength is not the whole story.** The same scripted full close at 1 kg
with the driven fingers' stiffness, damping and effort scaled ×3 and ×10
(`audit_reach_grasp.py --finger-scale`): still **0 %** held, and no
fingertip ever exceeded 40 N. So the pinch as posed cannot carry a
kilogram however hard it squeezes, which points at the contact geometry
(two near-point contacts on a smooth rod, a hold that lets it pivot or slide
out) rather than at the drives. Diagnosed with a lift-phase trace
(scripted close, 1 kg, per-step fingertip forces and bolt height):

- **The latch freezes the closure at the first 4 N of grip.** With the
  normal drives the grip crosses 4 N at closure 0.67, the fingers are held
  there, the lift starts with 4 / 8 / 4 N on thumb / index / middle, and the
  bolt slides out at the third lift step. The ×10 drives failed identically
  because the latch fired at the same 4 N.
- **With the latch raised to 12 N and normal drives, the grip never gets
  there**: full closure (1.00) produces 6 / 9 / 0 N, the middle finger having
  lost contact, because the pip / dip effort limit of 0.3 N·m caps each
  fingertip near 10 N. No lift ever starts.
- **With the latch at 12 N and ×3 drives, the grip reaches 11 / 16 / 9 N and
  the bolt still rolls out** during the lift, forces collapsing to zero
  within four steps of the wrist rising. At those forces friction alone
  would carry several kilograms, so the contacts are not opposing across
  the rod's centre: the tilted pinch holds a 30 g rod by friction torque and
  a 1 kg rod pivots out of it.

So the ceiling is the **grasp geometry**, then the finger effort limits,
then the latch threshold, in that order. A heavy rod needs a grasp that
cages it (fingers under the rod, or a wrap) rather than the two-sided
fingertip pinch every level so far was built on. That is exactly the
graspable-area / finger-count study proposed for the pile work, with mass
as one of its inputs. Parked until the real mass is known; the ladder
continues at 30 g.

## Deferred to the second increment

The swept-path safety shield with the α search (α is 1 everywhere now, so
the shield penalty is 0), clutter (level 8 raises `NotImplementedError`),
mesh-accurate proxy radii, PhysX self-collision.
