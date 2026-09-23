# Staged grasp pipeline — design

**Status: partly superseded by what was measured. Read this with
[`TRAINING_LOG.md`](../TRAINING_LOG.md), which records the runs and the three
things this document got wrong.**

Still correct: one environment, one fixed interface, the whole robot, Cartesian
wrist residuals through differential IK, the episode structure, the reward's
contact and success terms, and the safety layer.

Changed by measurement:

1. **The curriculum axis is wrist DISPLACEMENT, not the action mask.** The
   mask ladder (stages 0-4 below) was built and it failed: stage 2 returned
   0 % over 400 iterations because unmasking three Cartesian channels at once
   changed the task discontinuously. All 8 actions are now live at every
   level and only the displacement magnitude grows.
2. **The reward needed a dense approach term.** Experiment 2's sparse reward,
   inherited here wholesale, only works with a fixed wrist. A potential-progress
   term `2(Phi_t - Phi_{t-1})` was added, with an audit that gates training.
3. **The observation is 58, not 55** — three fingertip-to-bolt distances were
   added, informative before contact.

The stage table in section 10 is kept as the record of what was tried.

One bolt, one Wuji hand on the Adam-U arm, one table, parallel Isaac Lab
environments. The policy learns the approach and the grasp; a scripted
controller does the lift. Complexity is raised by *unmasking actions and
turning on randomisation*, never by changing the interface.

Explicitly out of scope, and not designed for: multiple bolts, piles or
clutter, camera or point-cloud observations, object detection, VLM planning,
ROS 2 deployment. The target bolt pose is given to the policy directly.

---

## 1. The one thing that makes this different from what is in the repo

Experiments 1–3 (`TRAINING_LOG.md`) each had their own action space, their own
observation width and their own robot asset, so no checkpoint or measurement
crossed between them. This pipeline fixes **48 observations and 8 actions for
every stage**, and a stage is nothing but

  - a mask over the 8 actions, with the masked entries' scale held at zero, and
  - a set of randomisation ranges.

The network architecture does not change between stages, so a stage-`k`
checkpoint loads into stage `k+1` without surgery, and a BC policy loads into
PPO without surgery. That property is the reason for the fixed interface and
it constrains every decision below.

---

## 2. Episode

```
reset → PPO grasping 3.0 s (90 steps) → scripted lift 1.0 s (30) → verification 1.0 s (30) → reset
```

30 Hz policy, 120 Hz physics, 150 steps, 5.0 s. The grasp phase is 3 s rather
than Experiment 2's 2 s because the wrist now has to travel before it can
close (section 6 does the arithmetic).

The hand starts **~5 cm above the bolt** in a solved pre-grasp orientation.
Finger targets freeze at the end of the grasp phase; the lift and the
verification ignore the policy's actions entirely, which is what makes the lift
a test of the grasp the policy chose rather than something it can recover from.

Success, unchanged from Experiment 2 so the numbers stay comparable: the bolt
at least `SUCCESS_LIFT` above its rest height for the whole verification phase,
the bolt never touching the table during that phase, and no finger–table
collision and no excessive fingertip force anywhere in the episode.

---

## 3. Observation — 48

| block | dim | source |
|---|---|---|
| bolt position relative to the wrist | 3 | wrist frame, rotated into the wrist's own axes |
| bolt orientation relative to the wrist, 6D | 6 | first two columns of `R_wrist^T R_bolt` |
| wrist linear velocity | 3 | body link velocity |
| wrist angular velocity | 3 | body link velocity |
| finger joint positions | 9 | the 9 driven synergy joints (thumb ×3, index ×3, middle ×3) |
| finger joint velocities | 9 | same joints, same order |
| previous policy action | 8 | the full 8, including masked entries (always 0) |
| thumb force | 1 | fingertip contact force on the bolt / `F_MAX` |
| index force | 1 | " |
| middle force | 1 | " |
| per-finger contact flags | 3 | thumb / index / middle, raw |
| normalised episode time | 1 | `episode_length_buf / 150` |

All normalised by rsl-rl's running observation normaliser, which is fit on the
demonstrations for BC and **frozen into the checkpoint** so PPO does not shift
the inputs underneath the initialisation.

Two notes that are decisions, not description:

**6D rotation, not quaternion.** A quaternion has a sign ambiguity that a
network has to learn around; the 6D representation is continuous. This is the
reason the spec asks for it and it is worth keeping.

**Contact force is the tactile stand-in.** PhysX reports one aggregate force
per fingertip, not a taxel array, so the real Wuji pad's "total force" and
"maximum taxel" collapse to the same number and appear once. That is honest as
long as it is written down; a per-taxel policy trained here would not transfer.

`N = 9` follows from thumb + index + middle being the driven joints. Ring and
little are held at fixed safe positions (`HELD_CURL`, flat — curling them
brings their tips *down* toward the table in this tilted pose, which is how
Experiment 2 logged a finger–table collision every episode). If a stage ever
needs them, `N` changes and so does the observation width — which would break
the fixed-interface property, so it is a deliberate fork, not a tweak.

---

## 4. Action — 8

| # | action | stage it turns on |
|---|---|---|
| 0 | wrist Δx | 2 (pending, see §10) |
| 1 | wrist Δy | 2 (pending) |
| 2 | wrist Δz | 1 |
| 3 | wrist Δrot-x | 3 (pending) |
| 4 | wrist Δrot-y | 3 (pending) |
| 5 | wrist Δrot-z | 3 (pending) |
| 6 | thumb closure synergy | 0 |
| 7 | index/middle closure synergy | 0 |

Actions 0–5 are **Cartesian residuals**: a small position delta and a small
axis-angle rotation delta applied to the current wrist target each policy step,
resolved to arm joint targets by
`isaaclab.envs.mdp.actions.DifferentialInverseKinematicsActionCfg` (verified
present in the `adam-u-rl:isaaclab-3.0` image; it reads
`body_link_jacobian_w` and is batched on the GPU, so it runs at 2048 envs).

**PPO never commands arm joint positions.** Low-level control is the implicit
PD drives already configured on the arm (stiffness 300, damping 20, the URDF's
effort limits) and on the fingers (stiffness 3, damping 0.1).

Pink IK was considered and rejected for training: `pink_task_space_actions.py`
instantiates one controller per environment and solves in numpy on the CPU
inside a per-env Python loop. It is a teleoperation tool, not a 2048-env
training tool.

Actions 6–7 are the synergies, unchanged from Experiment 2:

```
q_target = q_open + closure × (q_closed − q_open)
```

per joint, with `q_open`/`q_closed` fixed fractions of each joint's flexion
limit, first-order smoothed. Finger control stays position control.

**Masking.** A masked action is forced to zero *and* its scale is zero, so an
unmasked stage inherits a policy whose output on that channel was trained to be
meaningless but harmless. The mask is applied inside the action term, after the
network, so the policy always emits 8 numbers and the BC data always carries 8.

---

## 5. Limits and safety

All configurable, as the spec requires; these are the starting values.

```yaml
max_translation_per_policy_step: 0.002   # m
max_rotation_per_policy_step:    0.02    # rad
max_wrist_linear_velocity:       0.05    # m/s
max_wrist_angular_velocity:      0.5     # rad/s
```

Also applied: arm and finger joint limits (URDF), a workspace box around the
bolt, first-order action smoothing, fingertip force limit `F_MAX`, table
collision termination, and an emergency termination for unsafe configurations
(IK solve divergence, joint limit saturation held for several steps, wrist
outside the workspace box).

**Two of these four numbers contradict each other and one must move.** At
30 Hz, `0.002 m/step` is `0.06 m/s`, which the `0.05 m/s` velocity cap forbids;
`0.02 rad/step` is `0.6 rad/s` against a `0.5 rad/s` cap. As written the
velocity caps bind first and the per-step limits never do, which means the
per-step numbers are decoration and the effective limits are 0.00167 m and
0.0167 rad per step. Pick one: raise the velocity caps to 0.06 / 0.6, or lower
the per-step deltas. This matters because of the budget in §6.

---

## 6. Why the grasp phase is 3 s

Starting 5 cm above the bolt at 0.002 m per step, a pure vertical descent needs
**25 steps** — and 30 steps at the 0.05 m/s cap. A 2 s (60-step) grasp phase
would spend nearly half of itself descending before a single closure command
mattered. At 3 s (90 steps) the descent is a quarter of the phase and the
remaining 65 steps are enough for the 4-step stable-contact window plus the
squeeze. The translation budget over a full phase is 18 cm, so lateral
correction in later stages is affordable but not generous.

If stage 2 or 3 turns out to be limited by travel rather than by learning, the
fix is the per-step limit, not the phase length — a longer episode changes the
reward scale and breaks comparability with everything already measured.

---

## 7. Reward

Experiment 2's structure, unchanged, because it is measured and because
changing it would make the stage-0 result incomparable with the 100 % already
in `TRAINING_LOG.md`:

| term | weight | paid |
|---|---|---|
| first stable thumb–bolt contact | +0.5 | once |
| first stable index/middle–bolt contact | +0.5 | once |
| first simultaneous opposing stable contact | +1.0 | once |
| success | +20 | at the last step |
| drop | −5 | once |
| first finger–table collision | −3 | once |
| first excessive fingertip force | −3 | once |
| action rate | −0.01 | every step |

"Stable" means ≥4 consecutive control steps, so a one-step spike earns nothing.

**The trap this must not fall into.** Experiment 2's two-finger run collected
the full +2.0 of contact bonuses in every episode while its mean action sat
fully open, because exploration noise alone brushes the bolt often enough to
trip the stable-contact window. A reward an untrained policy already collects
in full carries no gradient. Once the wrist moves, the hand can be far from the
bolt, so those bonuses stop being free — but any *new* dense shaping added for
the wrist (an approach-distance term, an orientation-alignment term) has to be
checked against the same test before it is trusted: **measure what an untrained
policy collects, and if it is most of the term, the term is not shaping
anything.**

---

## 8. Demonstrations, BC, and PPO fine-tuning

Recorded per timestep: timestamp, finger joint positions, finger joint
commands, wrist position and orientation, wrist Cartesian command, tactile /
force, contact flags, bolt pose, episode success. Resampled to 30 Hz,
converted into the same 8-D action space PPO uses, split **by trajectory** into
train and validation — never timesteps from one trajectory across both.

BC: MLP `[256, 256, 128]`, elu, lr 3e-4, batch 256, Gaussian negative
log-likelihood so the learned standard deviation carries into PPO. Evaluated by
**closed-loop rollout in the environment**, not by offline action error alone.

PPO fine-tuning: actor initialised from BC, critic from scratch, with an
auxiliary BC loss annealed to zero:

```
actor_loss = PPO_loss + bc_coefficient × BC_loss     1.0 → 0.0
```

```yaml
num_envs: 2048            gamma: 0.99          entropy_coefficient: 0.01
learning_rate: 0.0003     gae_lambda: 0.95     clip_range: 0.2
```

The actor/critic MLP is `[256, 256, 128]` elu **in every stage**, PPO-only runs
included, so that a PPO-only baseline and a BC+PPO run differ in initialisation
and nothing else.

Two implementation facts, neither of which is free:

- rsl-rl 5.0.1's `PPO` has no auxiliary-loss hook. The BC term needs a
  subclass that keeps a demonstration batch and adds the term in
  `update()`. That is the one piece of the algorithm side that is not
  configuration.
- The repo's existing runner configs are `[512,256,128]` (Experiment 1) and
  `[128,128]` (Experiment 2). Neither matches, so this pipeline gets its own
  runner config rather than editing those and invalidating their logged runs.

---

## 9. What has to be built, and what already exists

| piece | state |
|---|---|
| synergy action term, `q_open + c·(q_closed − q_open)` | exists, `wuji_synergy_grasp_env_cfg.py` |
| contact bookkeeping, stable-contact windows, success test | exists, same file |
| geometric bolt–table clearance test | exists, same file |
| solved pinch geometry (wrist↔bolt transform at contact) | exists, `tools/find_pinch_pose.py` |
| full-body Adam-U + Wuji articulation | exists, `adam_u_wuji_grasp_env_cfg.py` |
| **fingertip contact sensors on the full-body robot** | **missing** — that asset spawns with `activate_contact_sensors=False` |
| **differential IK wrist action + masking + limits** | **missing** |
| **bolt-relative pose and wrist-velocity observations** | **missing** — Experiment 2 was blind to the bolt because its pose was fixed |
| **Cartesian scripted lift** | **missing** — replaces the prismatic `wrist_lift` joint, which only exists in the cut-out hand URDF |
| **stage registry, masks, randomisation ranges** | **missing** |
| BC trainer, PPO-with-aux-BC-loss, demo I/O | missing (Experiment 3's §2d–2f, never started) |

Four structural consequences of moving to the full body:

1. **The prismatic lift joint is gone.** The lift becomes a world-frame
   Cartesian +z ramp through the IK term. This retires the worst trap in the
   repo — a joint axis rotating with its parent, which once moved the wrist
   sideways and dragged the bolt 7 cm while reporting a correct 10 cm rise —
   because a world-frame Cartesian target cannot rotate with anything. A new
   equivalent trap replaces it: **a failed or saturated IK solve looks exactly
   like a held grasp**, so the lift must assert that the wrist actually
   travelled, not just that it was commanded to.
2. **Self-collisions are off** (`enabled_self_collisions=False`). Harmless
   with a fixed wrist; with a free wrist the policy can drive the hand through
   the torso and be rewarded for it. The workspace box is the cheap guard;
   enabling self-collision on the arm chain is the correct one, and costs
   solver time.
3. **Contact filters must resolve to exactly one collider entry per
   environment.** A broken filter reports zero force *silently*, which for a
   contact reward and a success test is the worst available failure mode. The
   table's collider mesh resolves; the bolt's prim does not (two
   contact-reporting descendants), which is why fingertip force on the bolt is
   computed as net − table.
4. **The pre-grasp pose has to be re-solved against the arm, and this is the
   one prerequisite that is not bookkeeping.** `find_pinch_pose.py` searched
   the sphere of hand orientations with *no arm attached*, so nothing in
   Experiment 2's pose knows about the arm's joint limits. Measured (section
   9a): the pose it picked is reachable to 0.0 mm and 0.02°, but only by
   driving `wristRoll_Right` onto its −0.960 rad stop in every solution,
   across the whole reachable XY region. The arm can hold that orientation and
   has no authority to leave it — in the one DoF stage 3 exists to learn.

   The fix is measured and cheap. Yawing the whole (hand, bolt) pair about
   world z preserves every pinch metric *and* the table clearance — clearance
   depends on the hand's tilt from vertical, not its heading, and the bolt's
   world yaw is a free parameter that rotates with it. Sweeping that yaw:

   | pinch yaw | pos err | ori err | `wristRoll_Right` | its slack |
   |---|---|---|---|---|
   | 0° (Experiment 2's) | 0.0 mm | 0.02° | −0.960 | **0.000 — on the stop** |
   | 45° | 0.0 mm | 0.01° | −0.840 | 0.120 |
   | 90° | 0.0 mm | 0.01° | −0.566 | 0.394 |
   | 105° | 0.0 mm | 0.02° | −0.477 | **0.483** |
   | 225°–345° | 0.4–11.4 mm | 0.3–9.1° | −0.960 | 0.000 |

   So the pipeline's pre-grasp pose is Experiment 2's pinch yawed ~90–105°
   about world z, and the tool that picks it must score arm slack, not only
   table clearance. Stage 1's job is then to close a 5 cm gap with Δz, from a
   pose the arm can actually move away from.

   Two further measurements from the same sweep, both useful:
   - **Position reachability is wide**: wrist x 0.25–0.55 m, y −0.40 to
     −0.10 m all solve to 0.0 mm at the contact height. The XY randomisation
     stages have room.
   - **`waistPitch` runs to 0.86–0.99 rad** at the contact height. The URDF
     limit is 1.361, so there is headroom, but the torso lean is doing real
     work and it is held, not actuated. If a randomisation stage pushes the
     bolt further away, the waist is the first thing that will need to become
     an action — and that breaks the fixed 8-D interface, so it is a fork to
     decide deliberately rather than discover.

---

## 10. Stages

Stage 0 and stage 1 are specified. **Stages 2 onward are not yet filled in —
the specification they came from was truncated mid-sentence at stage 1
("Keep: Wrist X an…").** The table below records what is known; the remaining
rows are a placeholder shape, not a decision.

| stage | unmasked actions | bolt / wrist pose | randomisation | advance at |
|---|---|---|---|---|
| **0 — finger only** | 6, 7 | fixed, hand in the solved pre-grasp pose close to the bolt | none | 90 % success |
| **1 — vertical approach** | 2, 6, 7 | hand ~5 cm above the bolt; X and Y fixed *(sentence truncated)* | none | — |
| 2 — lateral *(pending)* | 0, 1, 2, 6, 7 | ? | ? | ? |
| 3 — orientation *(pending)* | all 8 | ? | ? | ? |
| 4 — randomisation *(pending)* | all 8 | ? | physics / pose randomisation | ? |

Stage 0 is very close to Experiment 2's 3-finger run, which already evaluates
at **100 % over 512 episodes** — but not identical: the interface is 48/8
instead of 26/2, the grasp phase is 3 s instead of 2 s, and the robot is the
full body with an IK-driven wrist held still instead of a cut-out hand on a
prismatic joint. So stage 0 is a re-run, not a copy, and it is cheap (~6 min at
1024 envs). Its value is a baseline *in the interface every later stage uses*,
and a check that the IK-held wrist is as rigid as the old fixed one.

---

## 9a. Reachability, as measured

Host-side, from the vendored URDF, with `tools/solve_pregrasp_pose.py --orient
quat` (the arbitrary-orientation mode added for this — neither canned
orientation can express a tilted pinch). Least-squares over the 7 arm joints
plus `waistPitch`, seeded 10–20 times, bounded by the URDF limits.

Reproduce:

    python3 tools/solve_pregrasp_pose.py --orient quat \
        --quat -0.6447 -0.3871 0.0 0.6592 --target 0.40 -0.20 1.0092 \
        --waist 1.0 --seeds 30 --ori-weight 1.0

The result that matters is in section 9 item 4: the orientation is reachable
everywhere useful, and everywhere it is reachable the wrist roll is on its
stop, until the pinch is yawed about world z.

A caveat on reading this: it is pure kinematics. It says nothing about the
arm's *sag* under the drives, which Experiment 1 measured at ~4 cm with a
10 N·m effort cap and fixed by raising the caps to the URDF's. At 5 cm above
the bolt, sag of that order is the whole approach. It has to be re-measured in
the simulator before the pre-grasp pose is trusted.

---

## 11. Open questions

1. **Stages 2–4** — the rest of the truncated ladder: which actions, which
   randomisation, which success threshold advances each stage. This is the
   only thing blocking the stage registry; everything else in sections 3–8 is
   stage-independent and can be built now.
2. **The limit contradiction in §5** — raise the velocity caps or lower the
   per-step deltas.
3. **Physics randomisation, concretely** — friction, mass, bolt pose, wrist
   start pose, actuator gains? Ranges? The repo currently has *no*
   randomisation anywhere, so this is the axis with the least evidence behind
   it.
4. **Demonstrations** — the glove supplies 20 finger angles and no wrist pose;
   Manus trackers are the agreed 6-DoF source, but the tracker topic, its
   frame and its calibration against the glove frame are unverified, and the
   glove publisher is currently reporting zero messages (`TODO.md` §2a).
5. **Self-collision** — turn it on for the arm chain, or rely on the workspace
   box? Measure the solver cost before deciding.
6. **The waist** — held at ~0.9 rad of lean, as section 9 item 4 measures. It
   stays a held drive for as long as the fixed 8-D interface holds.
