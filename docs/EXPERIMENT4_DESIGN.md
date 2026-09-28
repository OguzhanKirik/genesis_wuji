# Experiment 4: Safe Reach-to-Grasp with the Wuji Hand

State as of 2026-09-24. Builds on the displacement-curriculum pipeline
(`TRAINING_LOG.md`, "Staged pipeline"; `EXPERIMENTS_SUMMARY.md` for the
short history). Nothing in the earlier experiments is modified: Experiment 4
is a new environment, a new PPO config, new tools and a new log directory.

| | |
|---|---|
| env | `genesis_rl/envs/wuji_reach_grasp_env_cfg.py` (`WujiReachGraspEnvCfg`, `ReachGraspEnv`) |
| PPO | `RslRlWujiReachGraspPPORunnerCfg`, algorithm `genesis_rl/algos/rsl_rl/masked_ppo.py` |
| level | `ADAM_U_RG_LEVEL=<0..8>` (not `ADAM_U_LEVEL`, which the pipeline env validates against its own 0..3) |
| train | `docker compose exec -e ADAM_U_RG_LEVEL=<n> adam_rl python genesis_rl/scripts/train.py --robot wuji --scenario reach_grasp --num_envs 2048 [--resume --checkpoint_path ... --reset-std 0.3]` |
| runs | `logs/rsl_rl/adam_u30_wuji_reach_grasp_L<n>/` |
| gate and audit | `tools/audit_reach_grasp.py` |
| eval | `tools/eval_reach_grasp.py` |
| tests | `tools/test_reach_grasp.py` |
| poses | `tools/solve_reach_grasp_poses.py` (host side) |

---

## 1. Objective

The whole Adam-U is present. The policy drives the seven right-arm joints
through Cartesian differential IK and the right Wuji hand through two finger
synergies. It must:

1. start 5 to 10 cm above the fixed bolt,
2. approach roughly along world −z,
3. make small wrist position and orientation corrections,
4. close into a stable opposing pinch,
5. avoid collisions between any robot link and the table, clutter and
   non-target objects,
6. hand over to a scripted controller that lifts the bolt 10 cm and holds it.

The bolt pose is given exactly until the perception levels. "Approach from
above" describes the **wrist trajectory**, not a palm-down hand: the hand keeps
the pipeline's solved, tilted pinch orientation, because that is the
orientation the geometry solver proved closes without hitting the table
(`tools/find_pinch_pose.py`, `tools/solve_pipeline_pregrasp.py`). Palm-down and
fingers-down both failed that test.

## 2. What is reused from the pipeline

- The solved pinch pose: wrist at (0.50, −0.20), 5.92 cm above the table at
  contact, yawed 90° so the wrist roll joint is off its stop; waist held at
  1.05 rad.
- The IK action term's mechanics: damped least squares, joint-limit-aware
  null-space, per-step and velocity caps, first-order smoothing, workspace box,
  target clipping, every held joint written every step.
- The synergy mapping (thumb: 3 joints, fingers: 6 joints, closure 0.40 to
  0.60 of each flexion limit), the fingertip contact sensors and the
  bolt-force derivation (net minus table), the stable-contact rule (4 steps
  above 0.1 N), the success test, the once-per-episode bookkeeping.
- The PPO settings: 256-256-128 elu, obs normalisation, lr 3e-4 fixed, init
  std 0.3, entropy 0.003, 2048 envs, 24 steps per env, std reset on resume.
- The scripted-baseline gate and the best-checkpoint rule.

## 3. Interface (fixed at every level)

**Actions (8)**, all live at every level, no masking:

| index | meaning |
|---|---|
| 0–2 | wrist Cartesian x, y, z residual |
| 3–5 | wrist roll, pitch, yaw residual |
| 6 | thumb closure synergy |
| 7 | index/middle closure synergy |

Six wrist actions become seven arm joint targets through damped
least-squares IK with a joint-limit-aware null-space term, Cartesian and
angular velocity caps, smoothing and rate limiting, a workspace box, and an IK
reference reset at every phase transition. The waist, neck, left arm, left
hand and the 11 unused right-hand joints are written every step.

Velocity caps are raised from the pipeline's values so a 10 cm descent does
not eat half the phase:

| | pipeline | Experiment 4 |
|---|---|---|
| translation per step | 1.67 mm (0.05 m/s) | 3.33 mm (0.10 m/s) |
| rotation per step | 0.0167 rad (0.5 rad/s) | unchanged |

A 10 cm descent costs 30 of 120 steps. The near-object velocity penalty
(section 6) is what slows the wrist near the bolt.

**Observations (74)** = the pipeline's 58 + 16 safety observations:

| block | dim |
|---|---|
| bolt position in the wrist frame (estimated pose at perception levels) | 3 |
| bolt orientation in the wrist frame, 6D (estimated) | 6 |
| wrist linear and angular velocity | 6 |
| driven finger joint positions and velocities | 18 |
| last action | 8 |
| fingertip forces on the bolt (÷ 40 N) | 3 |
| contact flags | 3 |
| episode time | 1 |
| normalised arm joint positions | 7 |
| fingertip-to-bolt distances (estimated) | 3 |
| nearest-obstacle vector, distal fingers, wrist frame | 3 |
| nearest-obstacle vector, palm / hand base / wrist | 3 |
| nearest-obstacle vector, forearm | 3 |
| nearest-obstacle vector, upper arm | 3 |
| minimum clearance per group | 4 |

Clearances come from an analytic proxy model of the right arm and hand
(spheres attached to bodies, sized from the collision meshes) against the
table box, and later against clutter primitives. Isaac Lab has no distance
queries, so this is the only way to know a distance before contact. The
proxies are approximate on purpose and the margins are validated by probes,
not assumed. Measured at the level-0 contact pose (2026-09-24): tips 0.7 cm
(the mesh probe says 1.1 cm, so the tip spheres over-approximate by ~4 mm),
palm 2.5 cm, forearm 2.0 cm, upper arm 11.1 cm. The forearm proxy sits inside
its 3 cm margin at the grasp pose itself, which costs −0.09 per episode; the
mesh-accurate probe in the second increment decides whether the proxy or the
margin moves.

## 4. Episode

30 Hz control, 120 Hz physics.

| phase | steps | who acts |
|---|---|---|
| APPROACH + GRASP | up to 120 | the policy |
| SETTLE | 10 | scripted: finger targets held, wrist frozen |
| LIFT | 30 | scripted: wrist target rises 10 cm in world z |
| HOLD | 30 | scripted: everything held, success is judged at the end |

Maximum 190 steps. An episode ends at grasp end + 70 steps, or earlier on an
invalid collision, a joint limit, an IK failure or the bolt leaving the table.

Only APPROACH + GRASP transitions contribute to the PPO actor loss. The env
publishes `extras["actor_mask"]` every step and `MaskedPPO` weights the
surrogate and entropy terms by it. The critic is trained on every step. This
is what the pipeline's known gap ("LIFT and HOLD still train the actor")
asked for.

**Grasp latch.** The grasp ends early when, for the same step:

- thumb and index/middle contact are both stable (4 consecutive steps above
  0.1 N),
- each side grips with at least 4 N (the pipeline measured that latching on a
  0.1 N touch cut closure short and dropped success from 96.5 to 70.5 %),
- no fingertip is above 40 N,
- the bolt is inside the grasp region and moving with the hand (relative
  speed below 0.10 m/s),
- no invalid collision has occurred.

On the latch step the env records the measured wrist pose and arm joints,
freezes the finger targets, zeroes the wrist action and the accumulated
Cartesian residual, resets the IK reference and integrator to the measured
state, holds for 10 settle steps, then starts the lift **from the measured
wrist pose**, never from an old commanded target. The pipeline's lift
re-seeding line compared the step counter against a buffer that is always
zero in that env, so the re-seed fired at step 0 instead of at the lift;
Experiment 4 keys it on its own phase machine and the tests check it.

## 5. Whole-body safety shield

Reward alone is not the collision defence. Before each policy command is
applied the shield

1. forms the candidate Cartesian target,
2. solves IK for the candidate arm joints,
3. checks joint limits,
4. checks robot-environment and robot self-collision on the proxy model,
5. checks a swept path between the current and candidate configurations,
6. if unsafe, scales the Cartesian command by α ∈ [0, 1],
7. if no non-zero α is safe, holds the previous safe target.

Monitored: every right-arm and right-hand link against the table and scene
objects, the rest of the robot against the environment, robot self-collision,
palm / wrist / hand base / forearm / upper arm contact, and any robot-bolt
contact outside the whitelist (thumb, index and middle links only).

Proposed margins, to be validated by geometry probes: distal fingertips vs
table 3 to 5 mm; palm, wrist and hand base 20 mm; forearm and upper arm 30 mm;
held body parts 20 to 30 mm.

Any real invalid collision above the noise threshold ends the episode. Logged
per step: α, the rejected action, the closest link pair and distance, the
contact force and body names.

**Status.** The proxy geometry, the contact monitoring, the invalid-collision
termination, the clearance observations and the α hook are in the first
increment. The swept-path check and the α search are the second increment;
until then α is 1 everywhere and the shield penalty is zero. Self-collision is
**not** checked until PhysX self-collision is enabled and measured: the sim
runs with it off, and a shield that forbids what the sim allows would fight
the policy for nothing.

## 6. Reward

During APPROACH + GRASP:

r = r_reach + r_fingers + r_velocity + r_contact + r_safety + r_regularisation

Isaac Lab multiplies every term by dt = 1/30 s. "Once" means paid on the first
qualifying step of an episode only.

| term | weight | definition |
|---|---|---|
| `reach` | 2.0 × ΔΦ | Φ_reach = ½(1 − tanh(d_w / 0.08)) + ½(1 − tanh(θ_w / 0.26)); d_w wrist position error and θ_w geodesic orientation error from the solved grasp pose. Progress, not value: standing still pays 0. The 8 cm scale gives a usable gradient from 10 cm |
| `fingers` | 2.0 × ΔΦ | Φ_fingers = ½[(1 − tanh(d_thumb / 0.02)) + (1 − tanh(d_opp / 0.02))], distances to the bolt capsule surface, d_opp = min(index, middle). Progress |
| `velocity` | −1 × | g(d_w) [0.02 ‖(v_hand − v_bolt) / 0.10‖² + 0.01 ‖(ω_hand − ω_bolt) / 0.5‖²], g = 1 − tanh(d_w / 0.04), **paid only while no fingertip touches the bolt**, capped at 0.5 per step. Slows the approach, not the grasp (see the note below) |
| `thumb_contact`, `finger_contact`, `opposing_contact` | +0.5, +0.5, +1.0 | once, stable contact |
| `success` | +20 | final step: bolt ≥ 5 cm up through the whole hold, clear of the table, no invalid collision, no over-force |
| `drop` | −5 | once |
| `excess_force` | −3 | once, fingertip above 40 N, voids success |
| `invalid_collision` | −10 | once, and the episode ends, voids success |
| `clearance` | −0.02 × Σ c_g | c_g = max(0, (m_g − d_g) / m_g)² per link group |
| `shield` | −0.1 (1 − α)² | shield intervention |
| `action_rate` | −0.001 | ‖a_t − a_{t−1}‖² |
| `joint_limits` | −0.05 | quadratic below 0.15 normalised slack; hard termination below 0.02 |

**Deviation from the spec, measured.** Applied literally, the velocity term
charged a wrist that never moved and simply closed its fingers −20 per
episode, the whole success reward: the impact spins the bolt and rattles the
wrist at several rad/s for a few steps, and the relative-velocity term read
that as the hand rushing the object. With the contact gate and the per-step
cap the scripted grasp pays −0.1 and the instant close −0.3 (level-0 audit,
2026-09-24).

No jerk penalty initially. Wrist velocity, acceleration and jerk are logged;
a configurable jerk term is added later only if successful policies are
visibly oscillatory.

## 7. Curriculum

One env, one 74/8 interface. Only reset ranges, physics and observation
corruption change between levels.

| level | standoff | xy error | yaw | roll, pitch | physics | perception | clutter |
|---|---|---|---|---|---|---|---|
| 0 gate | 0 | 0 | 0 | 0 | — | exact | — |
| 1 | 5 cm | 0 | 0 | 0 | — | exact | — |
| 2 | 5 cm | ±0.5 cm | ±3° | 0 | — | exact | — |
| 3 | 7.5–10 cm | ±1 cm | ±5° | ±1° | — | exact | — |
| 4 | 7.5–10 cm | ±2 cm | ±7.5° | ±2° | — | exact | — |
| 5 | as 4 | as 4 | as 4 | as 4 | mass 0.8–1.2, friction 0.7–1.3, gains 0.9–1.1 | exact | — |
| 6 | as 5 | | | | as 5 | bias xy ±2 mm, z ±1 mm, rot ±1°; noise 0.5–1 mm; latency 0–1 | — |
| 7 | as 5 | | | | as 5 | bias xy ±5 mm, z ±3 mm, rot ±3°; noise 1–2 mm; latency 1–2; dropout ≤ 2 % | — |
| 8 | as 7 | | | | as 7 | as 7 | 1 fixed obstacle outside the corridor, then randomised, then 2–3 |

**Reset.** The wrist starts at the solved pose plus a sampled standoff along
world z plus the level's xy / yaw / roll / pitch error. Arm joints come from
one of three baked solutions (0, 5, 10 cm above contact, waist held at 1.05)
plus the residual through that bucket's damped pseudo-inverse Jacobian. The
first-order reset was checked by full forward kinematics: worst corner
(2 cm xy, 2.5 cm z, 7.5° yaw, 2° roll and pitch) lands 2.3 mm and 0.65° off,
which the policy never sees because it observes the actual wrist.

| bucket | minimum arm slack |
|---|---|
| 0 cm | 0.601 |
| 5 cm | 0.607 |
| 10 cm | 0.465 |

**Perception levels.** The physical bolt is untouched. The policy's geometric
bolt observations (relative pose and fingertip distances) come from one
estimated pose T̂ = T ⊕ ε with a per-episode constant bias, per-frame noise,
latency and dropout (last valid estimate). Contact flags, forces and
proprioception stay physical. Rewards, the latch, collision checks and the
success test use ground truth. The policy never sees the error itself.

**Geometry gate (level 0, no PPO).** At the contact pose, scripted closure and
scripted lift over at least 100 episodes must give ≥ 95 % lift-and-hold, zero
structural or table collisions, valid joint slack, and < 5 mm horizontal bolt
drag during the lift. Training does not start until this passes.

## 8. PPO

`RslRlWujiReachGraspPPORunnerCfg`: 256-256-128 elu actor and critic,
observation normalisation, lr 3e-4 fixed, init std 0.3, entropy 0.003,
γ 0.99, λ 0.95, clip 0.2, 5 epochs, 4 minibatches, 24 steps per env, 2048
envs, save every 50. Algorithm `MaskedPPO`, which is rsl-rl 5.0.1's PPO with
the actor loss weighted by the env's actor mask.

Each level resumes from the previous level's **best** checkpoint with
`--reset-std 0.3`. The final checkpoint is never selected automatically.

## 9. Evaluation and advancement

`tools/eval_reach_grasp.py`, deterministic mean action, 512 episodes. Advance
only when all hold:

- lift-and-hold success ≥ 95 %,
- invalid physical collisions = 0,
- shield interventions on < 1 % of policy steps,
- joint-limit terminations = 0,
- excessive-force failures < 1 %,
- horizontal bolt drag during the lift < 5 mm at the 95th percentile (the
  maximum is reported too; over 512 episodes it is sampling noise around the
  gate, see the level-2 entry in `TRAINING_LOG.md`),
- the policy beats the scripted no-correction baseline.

Reported separately: grasp success, collision-free success, contact type,
minimum clearance per link group, shield intervention rate, fingertip forces,
joint-limit slack, wrist velocity / acceleration / jerk, bolt drag, and at
perception levels success as a function of bias.

## 10. Tests and audits before training

1. Scripted grasp baseline for every level (`audit_reach_grasp.py`).
2. Reward audit with stationary-open, stationary-closed, random, scripted and
   unsafe policies (`audit_reach_grasp.py`).
3. Collision tests for every monitored link group (`test_reach_grasp.py`,
   second increment for the shield).
4. Swept-collision test: safe endpoints, unsafe interpolation (second increment).
5. Grasp-latch and IK-reset regression tests (`test_reach_grasp.py`).
6. LIFT and HOLD produce zero actor-loss contribution (`test_reach_grasp.py`,
   CPU test of `MaskedPPO`).
7. Perception levels expose no ground-truth bolt position (`test_reach_grasp.py`).
8. Constant perception bias stays constant within an episode (`test_reach_grasp.py`).
9. TensorBoard logging of every reward and safety component: the env
   publishes per-step tensors into `extras["log"]` **after** Isaac Lab's
   reset has cleared it, which is why the pipeline's `Safety/*` never
   appeared.

## 11. Files

Added:

- `genesis_rl/envs/wuji_reach_grasp_env_cfg.py`
- `genesis_rl/algos/rsl_rl/masked_ppo.py`
- `tools/solve_reach_grasp_poses.py`, `tools/audit_reach_grasp.py`,
  `tools/eval_reach_grasp.py`, `tools/test_reach_grasp.py`
- `EXPERIMENT4_DESIGN.md` (this file)

Modified (additive only):

- `genesis_rl/algos/rsl_rl/rsl_rl_ppo_cfg.py`: `RslRlWujiReachGraspPPORunnerCfg`
- `genesis_rl/scripts/train.py`, `genesis_rl/scripts/play.py`: `--scenario reach_grasp`

## 12. Increments

1. Env with standoff levels 0 to 7, new reward terms, latch with IK reset,
   masked PPO, per-step logging, proxy clearances against the table, gate,
   audit and tests. **This increment.**
2. Swept-path shield with the α search, clutter primitives in the proxy
   model, collision and swept tests, level 8.
3. Train levels 1 to 4, then 5 to 7, then 8. **Levels 1 to 7 trained
   2026-09-24, all ≥ 99.8 % with zero invalid collisions; see
   `TRAINING_LOG.md`, "The ladder through level 7".** Level 8 waits for
   increment 2.
