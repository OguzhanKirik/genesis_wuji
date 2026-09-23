# How the grasping algorithm works

This is the reference for the reinforcement learning itself: what the policy
sees, what it controls, how it is paid, and how the training loop turns that
into a grasp. It covers **Experiment 2**, the finger-only synergy grasp
(`--robot wuji --scenario synergy`), which is the cleanest statement of the
method; the last section says how the other two scenarios differ.

For the environment, the robot assets and the measured results, see
[`../README.md`](../README.md) and [`../TRAINING_LOG.md`](../TRAINING_LOG.md).

---

## 1. The short version

A Wuji hand is held in a fixed pre-grasp pose around one bolt on a table. For
two seconds a neural-network policy controls **two numbers** — how far the
thumb closes and how far the fingers close. Then the policy is switched off, a
scripted controller raises the wrist 10 cm, and the episode ends by checking
whether the bolt is still held. A grasp that survives that check pays +20; a
few small bonuses pay for touching the bolt on the way there.

PPO runs 1024 copies of that episode in parallel, collects the outcomes, and
adjusts the network so that the actions which led to successful lifts become
more likely. Nothing is demonstrated or scripted about the *grasp* itself —
only the lift that tests it.

---

## 2. The learning algorithm: PPO

Proximal Policy Optimization, via [rsl-rl](https://github.com/leggedrobotics/rsl_rl)
5.0.1. It is **on-policy**: every update uses data the current policy just
generated, and that data is then discarded.

### The loop

One *iteration* is:

1. **Collect.** Run all 1024 environments for 24 control steps, recording at
   each step the observation, the action sampled from the policy, the reward,
   and the critic's value estimate. That is 24,576 transitions.
2. **Estimate how good each action was.** Compute returns and advantages with
   Generalised Advantage Estimation (λ = 0.95, γ = 0.99). The advantage is
   "how much better did this action turn out than the critic expected?" It is
   then normalised to zero mean and unit variance across the whole batch.
3. **Update.** Five passes over the batch, each split into four minibatches.
   Each minibatch takes a gradient step on the clipped surrogate objective
   (clip 0.2), the value loss (coefficient 1.0) and an entropy bonus
   (coefficient 0.01), with gradients clipped to norm 1.0.
4. Repeat, 500 times.

The clipping is what makes it *proximal*: the policy is not allowed to move far
from the one that collected the data, because that data would no longer describe
it. `desired_kl = 0.01` is set but the schedule is `fixed`, so the learning rate
stays at 3e-4 throughout.

### The networks

Two separate multilayer perceptrons, both 128 → 128 with ELU activations:

- **Actor.** Observation → the *mean* of a Gaussian over the 2 actions. The
  standard deviation is a learned parameter, initialised to 0.5, not a function
  of the observation. During training actions are sampled from that Gaussian;
  during evaluation the mean is used directly.
- **Critic.** Observation → one number, the predicted return. It is never used
  to act; it exists only to reduce the variance of the advantage estimate.

Both normalise their inputs with a running mean and variance, which matters
because the observation mixes radians, radians per second and newtons.

Configured in `adam_u_rl/algos/rsl_rl/rsl_rl_ppo_cfg.py`
(`RslRlWujiSynergyPPORunnerCfg`).

---

## 3. What the policy sees

The observation is a single concatenated vector, 26 numbers for the
three-finger configuration:

| block | size | contents |
|---|---|---|
| finger joint positions | 9 | the three joints of each synergy finger, radians |
| finger joint velocities | 9 | radians per second |
| previous action | 2 | what the policy asked for last step |
| fingertip force on the bolt | 3 | thumb, index, middle, newtons ÷ `F_MAX` |
| fingertip contact flags | 3 | thumb, index, middle, 1 if force > 0.1 N |

With two fingers the joint blocks shrink to 6 each and the vector is 20 long.
The force and flag blocks always cover thumb, index and middle so the
observation layout does not depend on which fingers are driven.

**On the force channels.** The spec this was built to asks for a per-fingertip
*maximum* and *total* force. The simulator's contact sensor reports one
aggregate force per fingertip rather than a tactile array, so those two
quantities are the same number here and it appears once. On hardware with a real
tactile pad they would differ and both would belong in the vector.

**How the bolt force is obtained.** Each fingertip has a `ContactSensor`
filtered against the *table's* collider. The force on the *bolt* is the
remainder: `‖net_force − table_force‖`. A filter aimed at the bolt itself was
tried first and PhysX either matched nothing or matched two entries per
environment; the failure mode is a silent zero, which is the worst possible
outcome for a contact reward. Self-collision is off and nothing else is within
reach, so the subtraction is exact. See `_tip_forces`.

Nothing in the observation tells the policy where the bolt is. It does not need
to know: the bolt never moves.

---

## 4. What the policy controls

Two numbers in `[-1, 1]`. They are **synergies** — one scalar drives several
joints together — rather than individual joint targets, which reduces the
search space from 9 dimensions to 2.

```
a[0]  thumb synergy    -> r_thumb_cmc_flex, r_thumb_mcp, r_thumb_ip
a[1]  finger synergy   -> mcp_flex, pip, dip of each active finger
```

### From action to joint target

```
raw      = clamp(a, -1, 1)
command  = (raw + 1) / 2                        # -> [0, 1], 0 = open, 1 = closed
closure += 0.3 * (command - closure)            # first-order smoothing, per step
q_target = q_open + closure * (q_closed - q_open)     # per joint
```

`q_open` and `q_closed` are each joint's flexion limit scaled by `OPEN_CURL`
and `CLOSED_CURL` — for three fingers, 40 % and 60 % of the limit. So the
policy does not command "40 % to 60 % of a fully closed fist"; it commands a
position inside a deliberately narrow band that was chosen so the stroke opens
clear of the 2.5 cm bolt and closes 13 mm past contact with it.

The smoothing coefficient of 0.3 means the closure reaches a new command in
about ten control steps. It exists so a noisy action sequence does not become a
juddering finger.

Every joint the policy does not drive is held at a fixed target every step:
the thumb's opposition (`THUMB_ABD`), all abduction joints, and the fingers
outside the synergy, held flat.

### The low-level controller

Joint targets go to Isaac Lab's implicit PD actuators, which run inside the
physics solver at 120 Hz:

| group | stiffness | damping | effort limit |
|---|---|---|---|
| fingers | 3 N·m/rad | 0.1 | 0.2–2 N·m, per joint type, from the URDF |
| wrist lift | 2·10⁵ N/m | 2·10³ | 5000 N |

The lift drive is deliberately rigid: the wrist is meant to be *fixed*, and at a
softer 5·10³ the fingers pressing on the table pushed the whole hand up 2.4 cm
on its own joint, silently changing every height in the experiment.

---

## 5. The episode: a three-phase state machine

120 control steps at 30 Hz, four seconds. The phase is a pure function of
`episode_length_buf`, so every environment is always in a known phase and
nothing has to be tracked.

| phase | steps | duration | what happens |
|---|---|---|---|
| 1 GRASP | 0–59 | 2 s | The policy drives both synergies. The wrist does not move. |
| 2 LIFT | 60–89 | 1 s | Finger targets **frozen** at their last value; policy actions are ignored. The prismatic `wrist_lift` joint ramps 10 cm. |
| 3 HOLD | 90–119 | 1 s | Everything frozen. Success is judged on the final step. |

Freezing the fingers during the lift is what makes the test meaningful: the
policy cannot react to the bolt slipping, so the *configuration it chose* has to
be good enough on its own.

**A defect worth knowing about.** The lift joint's axis is expressed in the
hand's own frame, and a joint axis rotates with its parent. The environment
spawns this articulation rotated into its pre-grasp orientation, so a nominal
`(0, 0, 1)` lift travelled wherever that rotation pointed it — measured, the
wrist moved sideways and *dragged the bolt 7 cm across the table* while
faithfully reporting a 10 cm lift. The axis is now baked into the URDF as
`HAND_ROT⁻¹ · ẑ`, and `__post_init__` refuses to build the environment if the
two disagree.

---

## 6. The reward

Let `t` be the control step, `1{·}` an indicator, and "stable" mean a contact
that has persisted `STABLE_STEPS = 4` consecutive steps.

```
r(t) =  0.5 · 1{first stable thumb–bolt contact}
      + 0.5 · 1{first stable index-or-middle–bolt contact}
      + 1.0 · 1{first simultaneous stable opposing contact}
      + 20  · 1{t is the last step and the grasp succeeded}
      -  5  · 1{first drop}
      -  3  · 1{first finger–table collision, > 15 N}
      -  3  · 1{first excessive fingertip force, > 40 N}
      - 0.01 · ‖a(t) − a(t−1)‖²
```

Every term except the last is paid **at most once per episode**. That is the
load-bearing detail. Paid per step, the contact bonuses would be worth up to
2.0 × 60 = 120 against a +20 success, and the optimal policy would be to hold a
finger against the bolt for two seconds and never lift it.

"Stable" is the second guard: a single-step brush earns nothing, so the policy
cannot farm the bonus by flickering in and out of contact.

A *drop* means the bolt was more than 2 cm up after the grasp phase and then
came back below it — it can only fire once something was actually lifted.

### Why the thresholds are thresholds

The collision and force limits are 15 N and 40 N, not zero. Every fixed-wrist
grasp this hand can make rests its fingertips lightly on the table and squeezes
the bolt at roughly 25 N, so a zero-tolerance rule would score every physically
achievable grasp as a failure. The numbers separate *resting on* the table from
*ploughing into* it (20–44 N was measured when the hand was too low). This is a
deliberate departure from the specification and is flagged as one.

---

## 7. How success is judged

An episode succeeds only if **all** of these hold at the final step:

1. the bolt has been at least 5 cm above its resting height for the whole of
   phase 3 (a 30-step streak, so a bolt that slips at any point fails);
2. the bolt never touched the table during phase 3;
3. no fingertip collided with the table anywhere in the episode;
4. no fingertip force exceeded 40 N anywhere in the episode.

Height alone is explicitly not enough — condition 1 is a *streak*, not an
instantaneous test, so flicking the bolt upward cannot pass.

Condition 2 is geometric, not a contact sensor: the bolt's centre must clear
the table by more than half the bolt's longest dimension. A contact sensor was
the first implementation and PhysX rejected the filter, and a rejected filter
reports zero force — which would have made "never touched the table" silently
always true. The geometric test cannot fail that way.

The outcome of each finished episode is copied into `last_*` buffers *before*
the reset. Isaac Lab resets an environment inside the same `step()` call that
ends it, so anything read afterwards belongs to the next episode. This cost a
full evaluation cycle to discover: the first evaluation script reported 0 %
success for a policy that was actually succeeding 99 % of the time.

---

## 8. Reading the training logs

rsl-rl reports each reward term as

```
Episode_Reward/<term> = (sum over the episode of weight · value · dt) / episode_length_s
```

with `dt = 1/30` and `episode_length_s = 4`. So for a term paid **once** per
episode, multiply the logged number by **120** to recover the reward in the
units of the table above:

| logged `Episode_Reward/success` | means |
|---|---|
| 0.1667 | +20 every episode → 100 % success |
| 0.0833 | 50 % success |
| 0.0000 | never |

This trips people up. A logged 0.16 looks tiny and is in fact a solved task.

---

## 9. Why the pre-grasp pose had to be solved

The policy controls the fingers only. If the hand is not already in a pose from
which *some* finger closure grasps the bolt, no amount of training will help.
Establishing that this pose exists is a geometry problem, and it is the part of
the work that took the longest.

`tools/find_pinch_pose.py` walks the synergy stroke with URDF forward
kinematics and measures where the fingertips actually converge. The insight
that makes it tractable: the thumb-to-finger gap, the opposition angle and the
fingertip-to-bolt distance do **not** depend on how the hand is rotated. Only
table clearance does. So the tool

1. places the wrist so that first contact with the 2.5 cm bolt happens *at* the
   bolt, then
2. searches the sphere of "which direction is world-up, expressed in the hand's
   frame" for one that keeps every fingertip above the table across the whole
   stroke.

That second step is what made a fixed-wrist pinch possible at all. Both obvious
orientations fail: the fingers hang about 17 cm below the wrist when open and
curl up to about 9 cm as they close, so any wrist height that puts the closing
pinch at the bolt buries the open fingertips up to 10 cm *under* the table.
Experiment 1 only worked because its arm rose while the hand closed.

The solved three-finger pose gives a gap of 5.2 cm open closing to 1.2 cm, first
contact at 38 % of the stroke, 13 mm of squeeze beyond it, an opposition angle
of 97°, fingertips 1.6 cm from the bolt's surface at reset, and 1.1 cm of table
clearance throughout. A scripted full close from it succeeds 16 times out of 16
*before PPO is involved* — which is the point. A feasibility experiment whose
scripted baseline fails is measuring the wrong thing.

---

## 10. What this does and does not demonstrate

**Does.** That PPO learns the finger configuration which holds this bolt, from
scratch, with a two-dimensional action and no demonstrations. Three fingers
reaches 100 % stable grasps over 512 evaluation episodes.

**Does not.** Anything about generality. The bolt pose, the hand pose and the
initial finger posture are all fixed, so the learned policy is close to a
constant — the evaluated three-finger policy simply commands full closure on
both synergies. The value of the run is that the whole pipeline is verified end
to end, not that the policy is interesting.

The two-finger result is the more instructive one, and it is a **reward-design**
failure rather than a physics one. A two-finger grasp exists — the scripted
sweep finds it — but at one closure combination out of sixteen, against four for
three fingers. Across the entire two-finger run the policy banked the full +2.0
of contact bonuses in every episode while its evaluated mean action sat at fully
open. The bonuses were being collected by the exploration noise: over 60 steps,
a policy whose mean is "open" still brushes the bolt often enough to trip a
4-step stable contact. **A reward that a random policy already collects in full
provides no gradient.** With +20 unreachable by chance, the only term left with
a slope was the −0.01 action-rate penalty, which rewards holding any constant
action, and the mean drifted to a constant and stayed.

The general lesson: when checking whether shaping rewards will help, ask what
an *untrained* policy already earns. If it earns all of it, the shaping is
decoration.

---

## 11. Commands

The full set, including asset rebuilds and diagnostics, is in
[`COMMANDS.md`](COMMANDS.md). The essentials:

```bash
docker compose up -d && docker compose exec adam_rl bash

# train (ADAM_U_FINGERS picks 2, 3 or 4 fingers; 3 is the default)
python adam_u_rl/scripts/train.py --robot wuji --scenario synergy \
    --num_envs 1024 --max_iterations 500

# evaluate: success, drop, collision, force, contact fractions, final synergies
python /opt/tools/eval_synergy.py \
    --checkpoint_path logs/rsl_rl/adam_u30_wuji_synergy/<run>/model_499.pt \
    --num_envs 256 --episodes 512

# the same, with the Isaac Lab viewport open and slowed down
python /opt/tools/eval_synergy.py --viz kit --num_envs 16 --episodes 16 \
    --real-time --speed 0.25 --start-delay 15 --checkpoint_path <ckpt>

# the same, recording videos offscreen
python /opt/tools/eval_synergy.py --checkpoint_path <ckpt> \
    --num_envs 4 --episodes 8 --video

# watch it, slowed down, with a pause so the window is up before the close
python adam_u_rl/scripts/play.py --robot wuji --scenario synergy --viz kit \
    --num_envs 4 --real-time --speed 0.25 --start-delay 25 --checkpoint_path <ckpt>

# scripted baseline and diagnostics, no policy involved
python /opt/tools/probe_synergy.py                      # one scripted episode, traced
python /opt/tools/probe_synergy.py --sweep --sweep-closure   # which closures hold
python /opt/tools/probe_synergy.py --trace-tips         # fingertip paths during a close
```

Host side, no simulator needed:

```bash
python3 tools/find_pinch_pose.py --middle          # re-solve the pre-grasp pose
python3 tools/hand_geometry.py                     # fingertip positions for a posture
python3 tools/make_wuji_hand_urdf.py --fingers 3   # rebuild the hand asset
```

Changing the finger count changes the solved pose, which changes the baked lift
axis, so it needs a URDF and USD rebuild. The environment asserts the two agree.

---

## 12. Where each piece lives

| piece | file |
|---|---|
| scene, observations, reward, success, state machine | `adam_u_rl/envs/wuji_synergy_grasp_env_cfg.py` |
| synergy action term and the scripted lift | same file, `SynergyGraspAction` |
| PPO hyperparameters | `adam_u_rl/algos/rsl_rl/rsl_rl_ppo_cfg.py` |
| training entry point | `adam_u_rl/scripts/train.py` |
| playback | `adam_u_rl/scripts/play.py` |
| per-episode evaluation and videos | `tools/eval_synergy.py` |
| scene checks and scripted sweeps | `tools/probe_synergy.py` |
| pre-grasp pose solver | `tools/find_pinch_pose.py` |
| hand asset generation | `tools/make_wuji_hand_urdf.py`, `tools/export_usd_wuji.py` |

---

## 13. The other two scenarios

Same algorithm, different task definition. Both use the full Adam-U robot
rather than the hand alone.

**`--scenario grasp`** (Experiment 1). The arm starts with the hand hovering
over the bolt and the policy controls 7 arm joints plus all 20 right-hand
joints — 27 actions, 101 observations. The reward is dense: fingertip proximity,
a proximity-gated closure term, a lift bonus and a height ramp. It reaches 99.4 %
of episodes ending with the bolt held. Because the arm is free to rise while the
fingers close, it does not need a solved pinch pose, and the grasp it learns is a
four-finger rake rather than a pinch.

**`--scenario reach`**. The arm starts beside the torso and must reach the bolt
first. Same action space, plus a coarse reach term to carry the long-range
signal. Smoke-tested only.

The progression is deliberate: `reach` is the full problem, `grasp` removes the
approach, and `synergy` removes the arm entirely so that the grasp itself is the
only thing being learned.
