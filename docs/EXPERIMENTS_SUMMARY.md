# Experiments so far, and what to carry into the next one

State as of 2026-09-24. This is the short version of
[`TRAINING_LOG.md`](TRAINING_LOG.md), written so the next experiment (a
top-down grasp with a standoff, see the last section) can be designed against
what has already been tried. Run-by-run numbers, learning curves and commands
stay in the log; the algorithm mechanics are in [`ALGORITHM.md`](ALGORITHM.md).

Everything below grasps the same object: the 30 g bolt `rod-with-threads-0001`
(2.5 cm diameter, 3.5 cm long, lying on its side) on the adam-moveit table
(top at 0.95 m), with the right Wuji Hand 2. Nothing else is on the table.

---

## 1. The experiments at a glance

| # | question | robot | policy controls | actions | reward style | result |
|---|---|---|---|---|---|---|
| 1 | Can PPO grasp and lift at all? | whole Adam-U | 7 arm joints + 20 hand joints, direct targets | 27 | dense (proximity, closure, height) | 99.4 % held, but a scoop, not a pinch |
| 2 | With the wrist fixed at a solved pinch pose, can two finger synergies make a real pinch? | hand only, fixed wrist | thumb synergy + fingers synergy | 2 | sparse (one-off contact bonuses + terminal success) | 100 % with 3 fingers, 0 % with 2 |
| 3 | Do glove demonstrations make all 20 finger joints learnable? | hand only, fixed wrist | 20 hand joints | 20 | same as Exp 2 | designed and built, never trained (glove not detected) |
| Pipeline A, mask curriculum | Can the wrist be added by switching action channels on stage by stage? | whole Adam-U | 6 wrist residuals via IK + 2 synergies, some channels masked | 8 | Exp 2 + nothing dense | stage 0 100 %, stage 2 0 %, abandoned |
| Pipeline B, displacement curriculum | Can the policy correct a wrong starting wrist pose, level by level? | whole Adam-U | 6 wrist residuals via IK + 2 synergies, all live | 8 | Exp 2 + dense `approach` + safety | L1 99.4 %, L2 99.8 %, L3 96.5 % as logged; 76 % on today's code |

The thread: 1 proved lifting is learnable but paid for the wrong grasp; 2
fixed the grasp by fixing the wrist and making the reward contact-based; the
pipeline put the wrist back under control. Experiment 3 branched off 2 and is
parked.

---

## 2. Experiment 1: grasp-only, every joint free

**Goal.** First feasibility check on the Wuji hand. Does PPO find any way to
get the bolt off the table and keep it there?

**Setup.**
- Whole Adam-U, torso leaned 0.35 rad, right hand palm-down over the bolt,
  fingers pre-curled about 30 %. The bolt spawns between the fingers with
  ±1.5 cm jitter and random yaw. No reaching.
- Actions (27): 7 arm joints as ±0.2 rad offsets around the start pose, 20
  hand joints as offsets from the pre-curl.
- Observations (101): arm and hand joint positions and velocities, wrist pose,
  bolt pose, bolt relative to wrist, last action, base position.
- Episode 5 s at 30 Hz, ends on time-out or the bolt falling far below the
  table. No scripted lift, no success test.
- PPO, MLP 512-256-128, lr 1e-3 adaptive, 4096 envs, 600 iterations, ~35 min.

**Reward.** Everything dense, paid every step, distances to the bolt centre.

| term | weight | pays for |
|---|---|---|
| alive | +0.5 | every step |
| terminating | −2 | ending early |
| reach_fine | +2 | thumb and index tips near the bolt (tanh, 4 cm) |
| grasp | +3 | finger flexion, gated by tip proximity |
| lift_bonus | +5 | bolt more than 2 cm above rest |
| object_height | +15 × height | bolt height, capped at 8 cm |
| off_surface | −2 × depth | pushing the bolt below rest |
| wrist_drift | −5 × d² | horizontal wrist travel from start |
| action_rate, joint_vel | −5e-4, −1e-5 | smoothness |

**Result.** 100 % lifted, 99.4 % still held at the end, mean peak 8.4 cm.

**Lesson.** Dense proximity and closure terms pay for any closed hand near
the bolt. The policy curled the fingers under the bolt and pinned it to the
palm. Nothing asked for opposing contact, so nothing produced a pinch. A
"held" criterion measured only in evaluation is not a training signal.

---

## 3. Experiment 2: hand only, fixed wrist, two synergies

**Goal.** A thumb-opposed pinch that survives a lift, not just any closed hand.

**Setup.**
- The Wuji hand alone on a fixed wrist, with a prismatic joint for a scripted
  vertical lift. The wrist pose is **solved**, not guessed
  (`tools/find_pinch_pose.py`): first contact happens at the bolt, and the
  hand's tilt is searched so every fingertip stays above the table across the
  whole closing stroke. Palm-down and fingers-down both failed that search
  from a fixed wrist; the open fingers hang about 17 cm below the wrist. The
  winning pose comes at the bolt from the side, tilted.
- Actions (2): thumb synergy and fingers synergy, each −1 (open) to +1
  (closed), smoothed. Ring and little held flat. `ADAM_U_FINGERS` picks 2, 3
  or 4 fingers.
- Observations (26): 9 driven joint positions and velocities, last action,
  fingertip force on the bolt (3), contact flags (3).
- Episode 120 steps: GRASP 60 (policy acts), LIFT 30 (scripted 10 cm, fingers
  frozen), HOLD 30 (scripted verification).
- PPO, MLP 128-128, init std 0.5, lr 3e-4, entropy 0.01, 1024 envs, 500
  iterations, ~6 min.

**Reward.** Sparse. Only action rate is paid per step.

| term | weight | paid when |
|---|---|---|
| thumb_contact | +0.5 | once, thumb tip on the bolt above 0.1 N for 4 consecutive steps |
| finger_contact | +0.5 | once, index or middle tip the same |
| opposing_contact | +1 | once, both at the same time |
| success | **+20** | last step: bolt above 5 cm for the whole hold, clear of the table, no finger-table hit above 15 N, no fingertip above 40 N |
| drop | −5 | once, bolt was up after the grasp and came back down |
| table_contact | −3 | once, fingertip pushes the table above 15 N (also voids success) |
| excess_force | −3 | once, fingertip above 40 N (also voids success) |
| action_rate | −0.01 | every step |

A clean success is worth about 22, a lift-then-drop about −3, never touching
the bolt exactly 0.

**Result.** 3 fingers: 100 % stable grasps over 512 episodes, solved by
iteration 400. 2 fingers: 0 %.

**Lessons.**
- Solving the pose against the table before training is what made a pinch
  exist at all. The scripted full close succeeded 16/16 before PPO saw it.
- The two-finger failure is exploration, not physics. Exploration noise
  collected the full +2 of contact bonuses every episode while the policy mean
  stayed fully open. A bonus a random policy collects in full carries no
  gradient on the mean, and with +20 out of reach by chance the only slope
  left was action rate, which rewards holding still. Watch for this whenever a
  sparse bonus is cheap to trip by accident.
- Two lift-axis bugs dragged the bolt sideways while reporting a correct rise.
  Always measure the bolt's horizontal travel during the lift.

---

## 4. Experiment 3: demonstrations, then PPO (not trained)

**Goal.** Show that human demonstrations make a hard action space learnable.
Controls all 20 finger joints from Experiment 2's fixed wrist pose, where pure
exploration is expected to fail.

**Setup.** Same hand, same wrist pose, same episode and same reward as
Experiment 2. 20 actions, 66 observations. Data path glove → ROS 2 →
`tools/manus_wuji_bridge.py` → UDP → `tools/record_demo.py`; validated end to
end with a synthetic glove. Behaviour cloning and the BC+PPO loss are not
written yet.

**Status.** Blocked on the Manus glove SDK detecting no glove. Independent of
the pipeline work, so it can be resumed later without affecting the next
experiment.

---

## 5. Pipeline A: whole robot, action-mask curriculum (abandoned)

**Goal.** Put the wrist back under policy control, one group of action
channels at a time.

**Setup.** Whole Adam-U on a fixed base. Actions (8): six Cartesian wrist
residuals turned into arm joint targets by damped least-squares IK, plus the
two finger synergies. Stage 0 masked all six wrist channels; stage 2 unmasked
wrist x, y, z and jittered the bolt ±1 cm. Reward was Experiment 2's, action
rate still −0.01.

**Result.** Stage 0: 100 %. Stage 2, resumed from stage 0: **0 % over 400
iterations**.

**Lessons.**
- A mask curriculum changes the task discontinuously. The stage-0 policy had
  never moved a wrist; three channels opened at once with an inherited
  exploration std of 0.98, and the hand walked off the bolt immediately.
- Experiment 2's reward only works with a fixed wrist. With the wrist free
  nothing pulls the hand toward the bolt, so contact never happens by chance,
  and the policy learns to hold still to minimise action rate.

---

## 6. Pipeline B: whole robot, displacement curriculum (the last experiment)

**Goal.** Learn to correct a wrong starting wrist pose before pinching, with
the error grown level by level. All 8 actions live from the start.

**Setup.**
- Whole Adam-U, fixed base, waist held at a solved lean. The bolt is always in
  the same place. The wrist starts at Experiment 2's pinch pose, re-solved
  against the arm and yawed 90° about vertical so the wrist roll joint is off
  its stop (`tools/solve_pipeline_pregrasp.py`). Fingertips start about 1.6 cm
  from the bolt surface.
- At reset the wrist is displaced from that pose by a random offset, applied
  through a baked first-order Jacobian. This is the curriculum axis:

  | level | xy | z | yaw | roll, pitch |
  |---|---|---|---|---|
  | 0 | 0 | 0 | 0 | 0 |
  | 1 | ±0.5 cm | −0.3 to +0.5 cm | ±2.5° | ±2° |
  | 2 | ±1 cm | −0.5 to +1 cm | ±5° | ±3° |
  | 3 | ±2 cm | −0.5 to +2 cm | ±10° | ±5° |

  The ranges were measured against the lowest fingertip's table clearance
  across the whole closing stroke (`tools/probe_wrist_displacement.py`). The
  envelope is asymmetric: z down and roll are what run into the table.
- Actions (8): dx, dy, dz, droll, dpitch, dyaw wrist residuals, smoothed and
  rate-limited at 0.2 cm per step, through IK; thumb and fingers synergies.
- Observations (58): bolt position and orientation in the wrist frame, wrist
  velocities, 9 driven finger joint positions and velocities, last action,
  fingertip forces, contact flags, episode time, normalised arm joints,
  fingertip-to-bolt distances.
- Episode 150 steps at 30 Hz: GRASP up to 90, LIFT 30 scripted, HOLD 30
  scripted. An early-grasp latch (added after these runs) freezes the wrist
  once a grip is stable, settles 10 steps and lifts early.
- Structural contact sensors on the palm/wrist/hand-base body, forearm and
  upper arm, plus the finger phalanges, all filtered against the table.
- PPO, MLP 256-256-128, init std 0.3, lr 3e-4 fixed, entropy 0.003, 2048 envs,
  24 steps/env. Level 1 from scratch, levels 2 and 3 resumed from the previous
  best with the exploration std reset.

**Reward.** Experiment 2's sparse scheme, plus one dense term that cannot be
farmed and the safety terms the moving arm needs.

| term | weight | paid when |
|---|---|---|
| success | **+20** | last step, Experiment 2's test plus no structural contact |
| thumb_contact, finger_contact, opposing_contact | +0.5, +0.5, +1 | once each, as in Experiment 2 |
| approach | +2 × ΔΦ | per step in the grasp phase, progress in a fingertip-to-bolt-surface potential (2 cm scale). Telescoping: hovering earns nothing net |
| drop | −5 | once |
| table_contact | −3 | once, fingertip or phalanx on the table above 15 N |
| excess_force | −3 | once, fingertip above 40 N |
| structural_contact | −5 | once, palm/wrist/forearm/upper arm on the table above 5 N, **and the episode ends** |
| joint_limits | −0.05 | per step, quadratic near any arm joint limit |
| action_rate | −0.001 | per step (lowered from −0.01, which made holding still the best option) |

**Results** (512 episodes, mean action, best checkpoint of each level).

| level | scripted full close, no wrist motion | trained policy |
|---|---|---|
| 1 | 72.7 % | 99.4 % |
| 2 | 39.1 % | 99.8 % |
| 3 | 14.1 % | 96.5 % |

Resuming level 1 into level 2 started at 81 % within ten iterations, against
0 % for the mask curriculum. That transfer is the evidence the displacement
axis works.

**Lessons.**
- The scripted baseline is the number to beat. If the policy does not beat a
  blind close, wrist control is not earning its place.
- Exploration std ratchets across a curriculum unless reset on resume.
  Entropy 0.01 inflated std 0.98 → 2.25 over three levels and drove fingers
  into the table. Entropy 0.003 plus `--reset-std` fixed it.
- The scripted lift must start from where the wrist actually is, not from
  the commanded target, or it drags the bolt.
- Every level peaks then declines, always through finger-table collisions.
  Pick the best checkpoint, not the last.
- Level 3's remaining failures are geometry: at −5° roll the lowest fingertip
  has 0.31 cm of clearance during closure. More training will not fix it.
- Action rate penalises change, not motion. A constant command drifted the
  wrist 12 cm after the grasp was already stable. The latch fixes this
  structurally.

**Open problems that the next experiment inherits.**
- The logged numbers predate the latch. On today's code the level 3
  checkpoint gets 76 % at level 3 and **0 % at level 0**, fingers never
  closing. Root cause unknown. Any new run trains on the latched env from
  scratch, so this needs a fresh eye but does not block.
- There is no reaching. Every level starts at the contact pose.
- The wrist penalty is a 5 N threshold against the table only. It gives no
  gradient before contact, and it does not see the bolt or any other object.
- LIFT and HOLD steps still feed the actor loss with actions the env ignores.
- Safety metrics do not reach TensorBoard; only `tools/eval_pipeline.py`
  reports them.

---

## 7. What carries over into any new setup

**Keep.**
- The 58/8 interface, so checkpoints load across levels.
- Sparse terminal success at +20 with one-off contact bonuses, disqualifying
  one-off penalties, and exactly one dense telescoping shaping term.
- Solving the pre-grasp pose against the arm and the table before training,
  and measuring the displacement envelope against the whole closing stroke.
- The scripted-baseline gate, the std reset on resume, entropy 0.003.
- Best-checkpoint selection and per-episode evaluation with the mean action.

**Rules learned the hard way.**
- Never pay per step for proximity or closure. Never pay a one-off bonus that
  exploration noise trips by itself.
- A curriculum must change the task continuously. Change magnitudes, not
  which actions exist.
- Measure bolt drag during the lift, and fingertip clearance during closure,
  before training.
- Pass `ADAM_U_LEVEL` with `docker compose exec -e`, or the eval silently runs
  level 0.

---

## 8. The next experiment: top-down grasp with a standoff

What was asked for: approach the bolt **from above**, start further away and
with larger pose error, and make sure the wrist never touches the table or
anything else.

**What changes.**
1. **A new pre-grasp pose.** The current pinch comes from the side because a
   fixed wrist could not hold a top-down pose with the fingers clear of the
   table. With the arm descending under policy control that constraint is
   gone, but the pose has to be solved again with `tools/solve_pipeline_pregrasp.py`
   under a new orientation constraint (approach axis near world −z), checked
   for arm reachability and wrist-roll slack, and its clearance envelope
   re-measured. The reset Jacobian and arm start poses are re-baked from it.
   Old checkpoints load but were trained for a different orientation, so this
   is **a new training run from level 1**.
2. **A standoff axis.** Start 5 cm above the contact pose (the constant and
   arm solution already exist in the env, unused), later 10 cm. At 0.2 cm per
   step a 5 cm descent costs 25 of the 90 grasp steps; 10 cm needs a higher
   per-step cap or a longer grasp phase. Use the solved start pose as the
   reset nominal rather than the first-order Jacobian, which was only
   validated to 2 cm.
3. **Wider displacement.** ±3 cm xy and ±15° yaw are outside the current
   envelope only because roll and pitch run into the table; a top-down pose
   changes that envelope, so re-measure before picking ranges.
4. **A dense wrist-clearance penalty.** Keep the −5 and terminate on contact,
   and add a per-step quadratic penalty when the lowest point of the
   palm/wrist body is under a margin (2 cm proposed) above the table. Add the
   bolt to the palm sensor's filter so pushing the bolt with the wrist is
   caught. Other objects do not exist in the scene yet; adding them is a
   later level, and each needs its own filter entry.
5. **Approach shaping from further away.** The 2 cm scale of `approach` has
   little gradient at 5 cm. Widen it or add a coarse potential, or the policy
   re-learns to hold still.

**Proposed ladder.**

| level | pose | standoff | xy | yaw | roll, pitch |
|---|---|---|---|---|---|
| T1 | top-down | 0 | ±1 cm | ±5° | measured |
| T2 | top-down | 5 cm | ±2 cm | ±10° | measured |
| T3 | top-down | 10 cm | ±3 cm | ±15° | measured |
| T4 | top-down | 10 cm | as T3 | as T3 | plus clutter objects |

T1 first, from scratch, to confirm the new pose grasps at all with a scripted
close before any wrist learning is asked for. Each level resumes from the
previous best with the std reset, and is gated by beating the scripted
baseline.

**To decide before building.**
- Confirm "from above" means a palm-down or fingers-down pinch, not a higher
  start on the current sideways pose.
- Whether clutter objects are part of this experiment or the one after.
- Whether to spend time on the level-0 0 % mystery first, or let the fresh
  from-scratch runs on the latched env show whether it recurs.
