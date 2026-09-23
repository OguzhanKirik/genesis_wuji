# Experiment 3 — imitation learning then reinforcement learning

**Status: design, for review. Nothing implemented yet.**

Two-stage pipeline for grasping one bolt with the Wuji hand: behaviour cloning
from teleoperated demonstrations, then PPO fine-tuning against the same reward
Experiment 2 uses. The wrist stays fixed; the policy controls only the fingers.

This document is the "show the design before implementing" step. Sections 1–3
are the decisions that need agreement, section 9 lists what I cannot answer
without you.

---

## 1. The decision that shapes everything: action dimensionality

**Recommendation: full 20-DoF right-hand control, not the 2 synergies of
Experiment 2.**

Three reasons, in order of weight.

**It is the only version where imitation learning earns its place.** Pure PPO
already solves the two-synergy task at 100 %. Adding demonstrations there would
demonstrate nothing. Behaviour cloning is worth its complexity when the action
space is too large for exploration to find the reward on its own — and we have
that failure measured: the two-finger pinch scored 0 % because the contact
bonus was collected by exploration noise and the +20 success was never reached
by chance (`TRAINING_LOG.md`, Experiment 2).

**The glove and the hand match one for one.** The Manus glove publishes exactly
20 ergonomics angles per hand, and the Wuji hand has exactly 20 joints per hand,
and they correspond structurally (section 4). Projecting that onto two synergies
throws away 18 of the 20 numbers the human produced. A glove demonstration
compressed to two scalars is not really a demonstration.

**It is what the spec's data format implies.** The trajectory schema records
"commanded finger targets", not a synergy scalar.

The cost is honest: 20-DoF PPO fine-tuning may still be hard even from a BC
initialisation. Section 8 keeps a fallback.

---

## 2. Pipeline

```
   Manus glove  ──120 Hz──>  manus_ws (ROS 2 container)
                                  │  retarget 20 ergonomics -> 20 Wuji joint targets
                                  │  UDP datagram, localhost
                                  ▼
   Isaac Lab container:  wuji_teleop_record_env  ──>  trajectories/*.npz + *.csv
                                                            │
                                    ┌───────────────────────┤
                                    ▼                       ▼
                             replay & validate        BC training
                                                            │  actor weights + obs normaliser
                                                            ▼
                                                   PPO fine-tune (Exp 2 reward)
                                                            │
                                                            ▼
                                                     evaluation, 3 ways
```

**Stage A — teleoperate and record.** The human wears the glove, the simulated
hand mirrors their fingers from the fixed pre-grasp pose, they close on the
bolt. The scripted lift and hold run exactly as in Experiment 2, so each
demonstration is automatically labelled success or failure by the same test.

**Stage B — data tooling.** Load, validate, resample, clamp, normalise, drop
invalid, replay, split.

**Stage C — behaviour cloning.** Supervised regression from observation to
commanded action over the successful demonstrations.

**Stage D — PPO fine-tuning.** Initialise the actor from BC, keep the observation
normaliser, train with Experiment 2's reward at a reduced learning rate and
exploration noise.

---

## 3. What is fixed, and what is not

Unchanged from Experiment 2, so the comparison is clean: the hand-only
articulation, the fixed pre-grasp pose and wrist, the fixed bolt pose, the
table, the 60/30/30-step episode, the reward, and the success test.

Changed: the action space (section 5), and the addition of a teleop input path.

---

## 4. Manus → Wuji retargeting

The glove publishes `manus_ros2_msgs/ManusGlove` on `/manus_glove_N` at 120 Hz,
carrying 20 `ergonomics` entries **in degrees**, with `side` = Left/Right.
Stretch is negative when flexed.

The correspondence with the Wuji hand is one to one:

| Wuji joint (per finger) | Manus ergonomic |
|---|---|
| `*_mcp_flex` | `<Finger>MCPStretch` |
| `*_mcp_abd` | `<Finger>Spread` |
| `*_pip` | `<Finger>PIPStretch` |
| `*_dip` | `<Finger>DIPStretch` |
| `thumb_cmc_flex` | `ThumbMCPStretch` |
| `thumb_cmc_abd` | `ThumbMCPSpread` |
| `thumb_mcp` | `ThumbPIPStretch` |
| `thumb_ip` | `ThumbDIPStretch` |

**Naming trap, worth stating loudly.** The thumb rows are shifted by one name:
Manus calls the thumb's three flexion joints MCP/PIP/DIP, Wuji calls them
CMC/MCP/IP. They are the same three physical joints. A mapping written by
matching names alone will silently misdrive the thumb, which is the one finger
this grasp depends on most.

**The existing retarget path is not reusable.** `manus_teleop/hand_retarget.py`
collapses the glove to 6 scalars (four finger flexions plus thumb flex and
rotation), discards all abduction, PIP and DIP, and emits *PND* hand joint names.
Experiment 3 needs a new node reading the raw `ergonomics` field.

**Calibration.** `hand_retarget.yaml` holds per-side open/closed degrees, but
only for the 6 sources currently used. All 20 need it. `hand_calibrate.py` is
the pattern to extend: record an open-flat pose and a tight-fist pose, average
2 s of each, and store per-ergonomic min and max. The retarget is then linear
from the calibrated range onto the Wuji joint's `[0, upper]` limit, clamped.
Left and right gloves report flexion with opposite signs, so calibration is
per side; we only need the right.

---

## 5. Observations and actions

**Observation — identical to Experiment 2 and used by both stages**, so a BC
policy and a PPO policy are interchangeable:

| block | size | contents |
|---|---|---|
| finger joint positions | 20 | all right-hand joints, radians |
| finger joint velocities | 20 | radians per second |
| previous action | 20 | |
| fingertip force on the bolt | 3 | thumb, index, middle, scaled by `F_MAX` |
| fingertip contact flags | 3 | thumb, index, middle |
| **total** | **66** | |

(Experiment 2's is 26 because only the 9 synergy joints are reported. Here all
20 are controlled, so all 20 are observed.)

**Action — 20 normalised joint targets**, one per right-hand joint:

```
q_target[j] = q_open[j] + (a[j] + 1)/2 * (q_limit_upper[j] - q_open[j])
```

with the same per-step smoothing and the same PD actuators. `a = -1` is the
pre-grasp pose, `a = +1` is the joint's limit. A demonstration frame's action is
the *commanded* target inverted through the same equation — never the measured
joint position, as the spec insists.

---

## 6. Demonstration data

One file per trajectory, NPZ for loading and CSV for inspection. The Isaac
container has numpy but no pandas, so the tooling uses numpy plus the standard
library `csv`.

| field | shape | notes |
|---|---|---|
| `t` | (T,) | seconds from episode start |
| `episode_id` | scalar | UUID, also the filename |
| `joint_pos`, `joint_vel` | (T, 20) | measured |
| `action` | (T, 20) | **commanded**, normalised — the BC target |
| `joint_target` | (T, 20) | the same in radians, for replay |
| `tip_force` | (T, 3) | thumb, index, middle, newtons |
| `contact` | (T, 3) | boolean |
| `wrist_pose` | (7,) | fixed, recorded anyway |
| `bolt_pose` | (T, 7) | |
| `phase` | (T,) | 0 grasp, 1 lift, 2 hold |
| `success` | scalar | the Experiment 2 success test |
| `meta` | dict | glove calibration, joint names in order, env constants, git SHA |

**Target: 20–50 successful trajectories**, with one recorded first to validate
the loader and the replay before collecting in bulk.

Only the grasp phase (steps 0–59) is used for BC; the lift and hold are scripted
and contain no decisions.

### Tools

`tools/demo_io.py` — load NPZ/CSV, validate that `meta.joint_names` matches the
environment's joint order exactly, resample to 30 Hz, clamp to limits, drop
invalid trajectories with a reason, and split **by trajectory** into train and
validation. Never split timesteps from one trajectory across both.

`tools/replay_demo.py` — play a recorded trajectory back through the environment
as open-loop joint targets and report whether it still succeeds. This is the
test that the recording is faithful: if a demonstration that succeeded live
fails on replay, the recording or the retarget is wrong, and no amount of BC
will fix it.

---

## 7. Behaviour cloning

Regression from observation to commanded action over the grasp phase of
successful trajectories.

- Loss: Gaussian negative log-likelihood, so the learned standard deviation
  carries into PPO as a sensible starting exploration. Mean-squared error is the
  simpler fallback but leaves the std arbitrary.
- Network: identical to the PPO actor — the point is that the weights transfer.
  For 66 inputs and 20 outputs, 256 → 256 rather than Experiment 2's 128 → 128.
- Observation normaliser: fitted on the demonstration set and **frozen into the
  checkpoint**. If PPO starts with a different normaliser the policy's inputs
  shift under it and the BC initialisation is destroyed.
- Validation: held-out trajectories, reporting action MSE and, more usefully,
  the success rate of rolling the BC policy out in the environment.

Output is a checkpoint shaped like an rsl-rl one, so stage D loads it directly.

---

## 8. PPO fine-tuning

Same reward, same success test, same episode as Experiment 2.

- Load the BC actor with `runner.load(bc_ckpt, strict=False)`; rsl-rl supports
  partial loading. The critic starts from scratch — there is nothing to clone it
  from — so expect the first iterations to be noisy while the value function
  catches up.
- Reduce the learning rate to about 1e-4 and start the exploration standard
  deviation at the BC-learned value, typically well under Experiment 2's 0.5.
  A large initial std walks the policy off the demonstrated manifold immediately
  and throws away the initialisation.
- Watch for the known trap from Experiment 2: if the contact bonuses are already
  earned in full by the BC policy, they contribute no gradient, and only the
  success term is doing work. That is fine here, but it means the success rate
  is the only metric worth watching.

**Fallback if 20-DoF fine-tuning does not converge.** Reduce the action space to
five per-finger synergies (thumb, index, middle, ring, little), which keeps the
glove's per-finger information but cuts the dimension by four. The demonstration
data does not need re-recording: the projection happens in the loader.

---

## 9. Evaluation, and what would make this experiment conclusive

Four policies on the same 512-episode protocol as Experiment 2:

| policy | what it tells us |
|---|---|
| PPO only, 20-DoF, from scratch | the baseline IL is supposed to beat |
| BC only | how good the demonstrations are |
| BC then PPO | the headline |
| Experiment 2's 3-finger synergy policy | the 100 % reference |

Plus the number of demonstrations needed, by training BC on 5, 10, 20 and 50
trajectories.

The experiment is conclusive if **PPO-only at 20 DoF fails and BC+PPO
succeeds**. If PPO-only at 20 DoF also reaches 100 %, the experiment has shown
that this task does not need demonstrations, which is a real result but a
different one, and the honest response is to move the comparison to a harder
task rather than claim a benefit that is not there.

---

## 10. Open questions

1. **The spec I was given is truncated** mid-sentence in the observation
   section. Anything after it — BC hyperparameters, the fine-tuning schedule,
   success criteria — I am inferring.
2. **Which hand.** The asset is the right Wuji hand. Confirm you will
   teleoperate with the right glove.
3. **Can you reach the grasp through the glove?** The pre-grasp pose puts the
   fingertips 1.6 cm from the bolt with a 5.2 cm thumb-to-finger gap. The
   demonstration is only useful if a human closing their own hand produces a
   grasp that works there. Worth testing with one trajectory before collecting
   20–50.
4. **Full 20-ergonomic calibration** does not exist yet and needs a session.
5. **Where should the new teleop node live** — a fourth package in `manus_ws`,
   or a script in this repo that subscribes over DDS? `manus_ws` is cleaner but
   couples the two repos.

---

## 11. Build order

1. Extend calibration to all 20 ergonomics; verify the retarget on the real hand
   or in RViz.
2. UDP bridge plus the recording environment. Record **one** trajectory.
3. `demo_io.py` and `replay_demo.py`; prove the single trajectory replays and
   still succeeds.
4. Only then collect 20–50.
5. BC training and rollout evaluation.
6. PPO fine-tuning and the four-way comparison.

Steps 1–3 are where this experiment will actually succeed or fail. The learning
is the easy part.
