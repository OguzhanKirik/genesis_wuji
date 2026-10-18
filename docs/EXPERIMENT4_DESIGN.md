# Experiment 4: Safe Reach-to-Grasp with the Wuji Hand

State as of 2026-09-24. Builds on the displacement-curriculum pipeline
(`TRAINING_LOG.md`, "Staged pipeline"; `EXPERIMENTS_SUMMARY.md` for the
short history). Nothing in the earlier experiments is modified: Experiment 4
is a new environment, a new PPO config, new tools and a new log directory.

| | |
|---|---|
| env | `genesis_rl/envs/wuji_reach_grasp_env_cfg.py` (`WujiReachGraspEnvCfg`, `ReachGraspEnv`) |
| PPO | `RslRlWujiReachGraspPPORunnerCfg`, algorithm `genesis_rl/algos/rsl_rl/masked_ppo.py` |
| level | `ADAM_U_RG_LEVEL=<0..27>` (not `ADAM_U_LEVEL`, which the pipeline env validates against its own 0..3) |
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
**Re-scored 2026-10-09.** Every status below was measured with an evaluation that stopped at the first N completed episodes and so under-counted the long failing ones (see TRAINING_LOG.md, "Evaluation stopping rule"). With a per-env quota at 2048 episodes the picks score: 22 95.0, 23 84.8, 24 94.7, 25 91.2, 26 86.8, 27 85.4, 28 80.8, 29 92.9, 30 93.6, 31 95.4, 32 95.3, 33 87.5, 34 89.4, 36 74.1, 38 77.2 %. Only levels 22, 31 and 32 clear the success gate; the mixed bin scene (level 33) is the frontier and the cover block its weakest part.


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
| 8 | as 7 | | | | as 7 | as 7 | 1 fixed box 10 cm in front of the bolt |
| 9 | as 7 | | | | as 7 | as 7 | 1 box, random pose in the 9–14 cm annulus outside the arm strip |
| 10 | as 7 | | | | as 7 | as 7 | 2–3 boxes, random, ≥ 7 cm apart |
| 11 | as 7, **bolt yaw ±15°** | | | | as 7 | as 7 | — |
| 12 | as 7, **bolt yaw −30° to +15°, bolt xy ±2.5 cm** | | | | as 7 | as 7 | — |
| 13 | as 7, **bolt yaw −45° to +15°, bolt xy ±5 cm** | | | | as 7 | as 7 | — |
| 14 | as 7, **bolt yaw −60° to +45°, waist freed**; xy ±5 cm inside −45° to +30°, ±2.5 cm at the edges | | | | as 7 | as 7 | — |
| 15 | as 14, **commanded grasp point ±14 mm along the rod's axis** | | | | as 7 | as 7 | — |
| 16 | as 15, **commanded pinch angle ±30° off perpendicular** | | | | as 7 | as 7 | — |
| 17 | **free grasp**: start as 16 with ±45°, no commands, true rod observed, reach to the axis, yaw action ×0.25 | | | | as 7 | as 7 | — |
| 18 | as 16, **rod standing on its end half the time** (grasp point = height along it), convex-decomposition collider | | | | as 7 | as 7 | — |
| 19 | as 18, **commanded point binding**: the latch needs the pinch within 4 mm of it | | | | as 7 | as 7 | — |
| 20 | as 19, **rod leaning on a support block a third of the time** (lying / standing / tilted, a third each) | | | | as 7 | as 7 | support block |
| 21 | as 20, **wrist rolled to the rod's tilt**, standoff and lift along the approach axis, slow first centimetre of lift | | | | as 7 | as 7 | support block |
| 22 | as 21, **one retry**: a failed lift or a drop reopens the hand and restarts the grasp phase | | | | as 7 | as 7 | support block |
| 23 | as 22, **a block over one end of half the lying rods**; the point drawn on the exposed part | | | | as 7 | as 7 | support, cover |
| 24 | as 22 (no cover: a pinned rod cannot be lifted), **the target across two rods** for 40 % of lying episodes | | | | as 7 | as 7 | support, bed rods |
| 25 | as 24, **inside a bin**: four walls 4 cm high, 13 cm from the nominal position | | | | as 7 | as 7 | + walls |
| 26 | as 25, cover back (30 % of lying rods), **slide-out before the lift for covered rods** (grip check built, found unreliable, off) | | | | as 7 | as 7 | + cover |
| 27 | **the pile**: four dynamic rods 4 to 8 cm around the target in the bin (no cover or bed); a lifted neighbour is a failure | | | | as 7 | as 7 | walls, 4 rods |
| 28 | as 27, the latch also needs a **centred pinch**: the pinch midpoint within 6 mm of the rod's axis (level 27's failed lifts were shallow pinches next to a neighbour) | | | | as 7 | as 7 | walls, 4 rods |
| 29 | as 28 with the grasp point **free** (level 17's device): no command, the policy picks the clear part of the rod; the centred pinch stays binding | | | | as 7 | as 7 | walls, 4 rods |
| 30 | as 29, **dense pile**: six neighbours 3 to 7 cm away placed without overlaps, one resting across a lying target in 30 % of episodes, two retries | | | | as 7 | as 7 | walls, 6 rods |
| 31 | as 30, **grip check v3**: 24 mm test lift over the pre-lift measured at the fingertips, a rod that does not follow is retried there; three retries | | | | as 7 | as 7 | walls, 6 rods |
| 32 | as 31, **free pre-lift**: the first 10 pre-lift steps are the policy's (wrist live, fingers held) before the test lift | | | | as 7 | as 7 | walls, 6 rods |
| 33 | as 32, **mixed scenes**: per episode the pile (60 %), the cover block (30 % of lying rods), the bed rods (40 %) or a plain bin | | | | as 7 | as 7 | walls, 6 rods, cover, bed |
| 34 | as 33, **late grip check**: rod vs fingertip rise over the second half of the test lift only | | | | as 7 | as 7 | as 33 |
| 35 | as 34, **the heap**: ten rods in 70 % of episodes, four dropped onto the first layer and the target; 25-step settle-in with the policy frozen | | | | as 7 | as 7 | walls, 10 rods, cover, bed |

**Clutter (levels 8–10).** Obstacles are kinematic PhysX boxes (5×5×6,
4×6×5 and 6×4×8 cm), immovable and collidable, re-posed at every reset and
parked inside the table's volume when unused, so the 74-observation
interface is identical at every level. Placement excludes the *approach
corridor*, which is **measured, not assumed**: `tools/probe_corridor.py` runs
the level-7 policy for 512 clutter-free episodes and records every 0.5 cm
table cell that any proxy sphere of the arm or hand ever covers. The union
(646 cm², a 12 cm wide band from the forearm's side through the bolt and
about 10 cm beyond it, where the fingers reach) is stored in
`corridor.npz` next to the env, dilated by the largest obstacle's
half-diagonal plus 1 cm, and obstacle centres are drawn from the 9–14 cm
annulus outside it. A strip heuristic was tried first and let the level-7
policy hit random obstacles in 53 % of level-10 episodes; the measured
footprint replaces it. The obstacles enter the analytic clearance model as extra
boxes, so the 16 safety observations and the clearance penalty see them
without any change. An obstacle collision is detected analytically, any proxy
sphere more than 4 mm inside a box, because the fingers are allowed to touch
the bolt and their contact force cannot tell one object from another; it is
invalid-collision kind 4, −10 and the episode ends.

**Pose generalisation (levels 11–13).** Levels 0–10 use one bolt pose. From
level 11 the bolt's yaw and xy are sampled at reset and everything nominal
follows the bolt: the pinch is yaw-invariant, so the wrist target is the
solved pose rotated about the bolt and translated with it. The standoff
reset draws from `pose_buckets.npz`, 582 reachable arm solutions over bolt
yaw × xy ±5 cm × three heights (`tools/solve_reach_grasp_poses.py --grid`),
nearest bucket plus the residual through its Jacobian. Reachability with
the waist held at 1.05 rad limits bolt yaw to −45° to +15°; the ranges stop
there. The clutter corridor is measured in the bolt's frame and rotates with
it.

**Freed waist (level 14).** Levels 0–13 hold the waist at the pipeline's
lean (pitch 1.05 rad, yaw 0, roll 0). Level 14 lets the pose solver choose
the waist too: `solve_reach_grasp_poses.py --grid --free-waist --min-slack
0.15` scans pitch {0.9, 1.05, 1.2} × yaw {0, ±0.25, ±0.5, ±0.75} × roll
{0, ±0.2} rad per (bolt yaw, xy) column, nearest to the nominal hold first,
and keeps the first triple whose worst-height arm slack reaches 0.25 (else
the best found). One triple per column, because the arm descends through
every height in one episode; the standoff reset writes it as joint state and
the action term holds it as the drive target, exactly as the pitch was held
before. The result is `pose_buckets_waist.npz` (621 of 975 buckets with
slack ≥ 0.15). Measured window: every xy within ±5 cm reachable for bolt yaw
−45° to +30°, within ±2.5 cm out to −60° and +45°, nothing at ±75° with any
hold. The waist yaw axis is tilted with the 60° lean, so yaw alone bought
only +15°; the pitch and roll do the rest. Bolt yaws from +45° to +120° (mod
180°) stay out of reach of a palm-down pinch from this side: a third of the
orientations, which needs the other arm or a different pinch, not a wider
reset.

**Commanded grasp point (level 15).** The rod is 35 mm long: a 25 mm
thick band in the middle 12 mm, 16.6 mm thread over the 17 mm at each end.
Levels 0–14 grasp its centre. Level 15 samples an offset s along the axis,
±14 mm, per episode, and moves everything the policy is told about the
bolt to that point: the nominal wrist pose (so the reach reward, the reset
bucket and the workspace clamp follow), and the estimated bolt pose that
feeds the bolt-in-wrist and fingertip-distance observations. The physical
bolt, the latch (surface distances, indifferent to where along the rod),
the drag and the success test stay on the true object. The interface stays
74/8: the policy sees a bolt whose centre is where it should pinch, which
is what lets it start from the level-14 weights. Evaluation pins s
(`eval_reach_grasp.py --grasp-offset-mm`) and sweeps it; the success rate
per offset is the graspable area along the axis. Pinch direction other
than across the axis is a second grasp family, not covered here.

**Commanded pinch angle (level 16).** The same device, one axis further:
the policy's estimated bolt orientation is yawed about the vertical by a
sampled delta, ±30°, so the policy aligns to a virtual rod and closes on
the real one obliquely. At this level the sampled yaw window bounds the
*hand's* yaw, since the arm's reach is about the hand, and the rod's yaw is
hand yaw minus delta; the reset bucket is looked up by hand yaw. Latch,
rewards and success stay on the true rod. `--pinch-angle-deg` pins delta;
offset × angle sweeps give the two-dimensional graspable-area map.

**Free grasp (level 17).** The commands come out. The start pose is still
drawn as at level 16 (hand somewhere over the rod, ±14 mm along it, hand
yaw ±45° off square), but the policy now sees the *true* rod, the reach
reward pulls the pinch centre (a fixed point in the wrist frame, where the
rod's centre sits at the solved contact pose) to the rod's *axis segment*,
any point along it, and asks only that the approach stays from above: a
tilt term, no yaw term. The world-yaw component of the wrist action runs
at a quarter rate, so rotating into alignment within the grasp window is
not available as the easy answer. Latch, drag, success and collision rules
are unchanged. The policy is rewarded for grasping whatever is under the
hand, wherever along the rod and at whatever angle the start gave it. The
±45° comes from level 16's angle map and is adjusted to what that map shows
as physically graspable.

**Standing rod (level 18).** Half the episodes spawn the rod on its end,
axis vertical, centre 17.5 mm above the table, at a random yaw. The pinch
is unchanged: it closes horizontally, which is across a vertical rod at
any hand yaw, so the hand yaw is drawn from the reach window and the pinch
angle is moot. The commanded grasp offset becomes a height along the rod,
±14 mm about its centre, and the nominal wrist pose rises with it (the
reset bucket is looked up at the raised standoff). The rod as shipped
cannot stand: PhysX cooks its 327k-face mesh to a 64-vertex convex hull
whose end is not flat, so a standing rod tilts 4.5° at once and a third
topple within 3 s untouched. A copy of the asset with a 32-hull convex
decomposition (`rod-with-threads-0001_cd.usdc`, made by
`tools/make_bolt_colliders.py`) stands with 0.03° of tilt, and levels with
a standing rod use it (`RGLevel.bolt_collider`); the level-17 policy was
re-evaluated on it to measure what the collider changes for the lying
grasp. A palm on top of the rod is a non-grasp-link contact and ends the
episode, so low grasps on a standing rod are where the map is expected to
fall off.

**Binding grasp point (level 19).** Measured at level 18 (and it applies
to 15 and 16): the policy pinched near the band whatever point it was
commanded, because the latch fires on a stable grip anywhere on the rod
and the shaping towards the point is weak against the +20 success. From
level 19 the latch requires the pinch centre, the thumb-versus-fingers
midpoint projected on the true axis, within `grasp_point_tol` (4 mm) of
the commanded offset; no latch, no lift, no success. Every evaluation now
prints the pinch-versus-commanded error at the latch, and the sweep table
carries it, so a map's cell means "grasped there" only when that column
is inside the tolerance.

**Commanded-point shaping (level 20, continuation).** At binding levels
a potential 1 − tanh(|pinch − point| / 1 cm) on the pinch centre's position
along the true axis is paid as progress in the grasp phase, weight 1.0,
beside the reach and finger terms. Zero at levels without a binding point.
Added because the policy had learned the point only from the latch
refusing off-point grasps.

**Tilted rod (level 20).** One end on the table, the rod resting on a
low kinematic block (4 × 4 cm, 10, 17 or 25 mm tall, one per episode, the
others parked) whose near edge is 30 mm along the rod from the low end.
The block's height fixes the spawn tilt by a bisection on the rod's
underside height at the edge: about 17°, 32° and 52°; physics settles the
rest. The block takes one of the obstacle AABB slots (free at these
levels, clutter being off), so the clearance observations and the
obstacle-collision rule see it; it is axis-aligned so the AABB is exact.
The horizontal pinch is across a slanted axis at any tilt, so the hand's
yaw is the rod's plus the pinch angle as before; the grasp point's height
follows the axis and the reset bucket is looked up at the raised standoff.
Lift and drop are measured from the rod's spawn rest height (`rest_z`),
which for a lying rod is the old constant, so levels 0–17 are unchanged.

**Wrist roll (level 21).** For a tilted rod the level-20 maps showed
the point reached but the lift lost: a horizontal pinch, lifted
vertically, pivots the rod about its end on the table (14 mm of drag)
and one grasp in ten slides out. At level 21 the hand rolls by −θ about
the rod's horizontal normal n = z × u, so its approach axis is the rod's
up-normal and the pinch closes square to the slanted axis; the standoff
and the scripted lift run along that axis, so the rod translates off its
block instead of pivoting. The reset draws from `pose_buckets_tilt.npz`
(the waist grid solved at tilts 0°, 18.4°, 29.5° and 39.8°, keyed by
tilt as a fifth coordinate). Two global changes came with it, applied at
every level: the lift is 40 steps, the first ten at 1 mm to let the grip
settle; and drag is measured as the bolt's slip relative to the wrist
during the lift, which for a vertical lift is what it was, and for a
tilted lift excludes the sideways travel that the lift itself commands.

**Retry (level 22).** When the scripted lift ends with the rod not up, or
the rod drops during the hold, and the episode has attempts left
(`retry_max`, 1 at level 22): the hand opens, the latch clears, the IK
reference is re-seeded from the measured pose, and the grasp phase
restarts from where the hand is, 10 cm up, with a 60-step window. The
progress terms skip the restart step so the jump in potential is not
charged; the drop penalty already paid stands; success on the second
attempt is a success. The episode is 140 steps longer, and the
time observation is normalised by the level's own length. No new
observation: the policy sees the hand high with open fingers late in the
episode, a state it never met before, and learns the retry from there.

**Partial cover (level 23).** Half the lying rods get a 4 × 4 × 3 cm
kinematic block hovering 2 mm above one end, its near edge 5 mm from the
rod's centre, so that end cannot be reached. The block takes obstacle AABB
slot 0, free for a lying rod, so the clearance observations and the
obstacle-collision rule see it. The commanded point is drawn on the
exposed part, from −14 mm to +5 mm measured towards the block; a pinned
point under the block at evaluation is an impossible grasp and reads as
such. This is the first level where a grasp planner's choice matters: the
policy is told a reachable point, and the map says how close to the cover
it can still grasp.

**Rod on rods (level 24).** For 40 % of lying episodes the target spawns
across two kinematic copies of the rod lying on the table, 2 cm apart
along the target's axis and perpendicular to it, and settles on their
bands about 35 mm up. It rolls when touched. The bed rods enter the
clearance model as capsules (segment plus the band radius), a third
obstacle kind beside the table box and the AABB slots, so the 16 safety
observations and the collision rule see them; a fingertip more than 4 mm
into a bed rod ends the episode as an obstacle collision.

**Bin (level 25).** Four kinematic walls, 1 cm thick and 4 cm high,
10 cm from the nominal bolt position on each side, in obstacle slots 3
to 6. Rods spawn within ±5 cm, so at least 3 cm from a wall; the arm's
approach and the hand's yaw are what the walls constrain.

**Pre-lift: slide-out and grip check (level 26).** Between the settle
and the lift, every level now has a 15-step pre-lift: ten steps of a
scripted drag at 3 mm/step along the rod's axis away from a covering
block (zero motion for an uncovered rod), then eight steps of a 1 mm test
lift along the approach axis. At levels with `grip_check`, a latched rod
that rose less than half of what the wrist rose during the test lift (the
wrist lags its target, so the test is relative) is not in the hand
(the level-21 trace: the fingers pressing the rod onto its support), and
the retry fires there, 5 mm up, instead of after an empty 10 cm lift. The
drag answers level 23: a rod whose end is under a neighbour is pulled out
3 cm before it is lifted. The scripted part of an episode is 98 steps.

**The pile (level 27).** Four dynamic copies of the rod are laid at
random yaw 4 to 8 cm from the target inside the bin, and settle while the
hand descends. They enter the clearance model as *soft* capsules that
follow their true poses every step: the 16 safety observations see them,
but pushing them is allowed, so neither the capsule-penetration rule nor
the force-based "non-grasp link on bolt" rule fires on them (the latter
cannot tell the target from a neighbour and is off at pile levels). A
neighbour more than 2 cm above its rest at the end of the hold is a
double pick and fails the episode. The cover and the bed rods are off:
the neighbours are the real version of both.

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


### Planned: levels 20 to 25 (2026-09-29)

Every level keeps level 19's binding grasp point, so its map is honest,
and the pile levels alternate a commanded version (for the map) with a
free version (for the behaviour), as levels 16 and 17 did.

| level | variation | needs | effort |
|---|---|---|---|
| 20 | tilted rod, leaning on a block | DONE: 75 % with the horizontal pinch; the slip at lift is the pinch geometry | — |
| 21 | wrist roll matching the tilt (was "20b") | DONE: tilted 81.8 %; the loss is the latch firing on a rod pressed to its support | — |
| 22 | retry after a slip or a miss | DONE: every pose ≥ 92 % | — |
| 23 | partial cover | DONE as a negative result: a pinned rod cannot be lifted (23.9 %); needs a slide-out phase | — |
| 24 | rod on rods | DONE: 92.4 % | — |
| 25 | rod in the bin | DONE: 92.4 % mixed, 3 to 5 % wall contacts; continuation queued | — |
| 24 | the real bolt | its geometry and mass; new pinch solve, yaw window re-measured, the head as a new grasp area; retrain from level 5 | 2–3 days once the file is here |
| 25 | grasp families | end-face pinch, two-finger hand; a new solved pose and a retrain from level 1 each | 2 days each |

Order (renumbered 2026-09-29 evening): 21, 22, 23, 24, 25 in sequence; the real bolt is the
scanned rod, so the former "real bolt" level is dropped; grasp families stay parallel work.

| 26 | slide-out + grip check | DONE: grip check unreliable (off); scripted slide-out failed on covered rods (10 %); other poses 90–97 % | — |
| 27 | a pile: four dynamic rods around the target | DONE: 88.9 % (no gain over the untrained level-26 policy); 0 double picks | — |
| 28 | centred pinch | DONE (2026-10-06): 86.7 % mixed, tilted 85.9 % (from 81.5), lying 85.4, standing 93.3; every latch within 6 mm of the axis; the failures left are grasps that never latch. The pick loses the cover/bed scene of level 26 (82.1 %, 7.8 % collisions) | — |
| 29 | free grasp in the pile | DONE (2026-10-06): 95.3 % mixed (from 86.7), lying 94.1, tilted 92.6, standing 98.8; 92.4 % already untrained. The policy picks the clear part of the rod. 3.3 % wall contacts on the lying rod | — |
| 30 | denser pile | DONE (2026-10-06): 98.5 % mixed, lying 94.9, tilted 98.5, standing 98.4, rod across the target 95.5 (1.2 % double picks). Spawn overlaps (through the support block and the target) found and fixed, also for levels 27–29 | — |
| 31 | grip check, redone | DONE (2026-10-06): 98.2 % mixed, lying 96.4, tilted 98.5, standing 99.6, rod across 97.5. The fingertip test lift fires on 2–12 % of episodes, mostly on grasps that would have lifted; the retries make it harmless, not a clean gate | — |
| 32 | the pre-lift as a policy phase | QUEUED behind 31 (2026-10-06): the first 10 pre-lift steps are the policy's with the fingers held, for the rod under a crossing neighbour (the scripted slide-out of level 26 failed: 10 %) | — |
| 33 | mixed scenes | QUEUED behind 32 (2026-10-06): each episode draws its scene (plain, cover, bed, pile) in the bin; the level-28 pick had lost the cover/bed scene (82 % at level 26, 7.8 % collisions); the chain evaluates the pick at level 26 | — |
| 34 | grip check, last-12-mm version | QUEUED behind 33 (2026-10-06): the reference heights are recorded at the middle of the test lift (v3 fired on 12 % of lying grasps that would have lifted) | — |
| 35 | the heap | DONE, NOT PASSED (2026-10-08): pick 15900 at 88.5 % mixed, 73.3 % on the heap; 1200 iterations flat, the policy cannot see which side of the rod is free (see level 36). ten rods, four dropped onto the first layer and the target, 25-step settle-in before the policy moves (extra time, not grasp time); a lying target buried under a dropped rod is re-labelled, the rod on top becomes the target (`PileCfg.top_rod`); a retry raises the wrist 3 cm first (`retry_backoff`); 1024 envs (2048 ran the host out of RAM) | — |
| 36 | the heap map | DONE, NOT PASSED (2026-10-08): pick 15998 at 90.0 % mixed, 79.0 % on the heap, flat training; the map is read but a clean pinch rarely exists (see level 37). level 35 plus 8 observations, the free radial distance around the estimated rod per sector; a level-35 checkpoint is widened with `tools/expand_checkpoint.py` | — |
| 37 | the heap, pushing allowed | DONE, NOT PASSED (2026-10-09): pick 17197 at 87.3 % mixed, 77.4 % heap; dropping the neighbour clearance penalty made the policy looser off the heap (no heap 91.6 % from 97.1) and no better on it; lever withdrawn | — |
| 38 | the heap, 10 mm latch | DONE, NOT PASSED (2026-10-09): pick 17100 at 86.5 % mixed; latches up to 92–94 % but a third fail the grip check; heap 83.6 % at 512 episodes, 72.8 % at 2048 (eval variance or drift under investigation). level 37 with `pinch_centre_tol` 10 mm; the latch trace shows the refused heap grasps are shallow holds the grip check can judge | — |
| 39 | transfer | IN PREPARATION (2026-10-09): level 32's scene from its pick, with the observation set, the latch and the phase machine on robot-measurable signals (`tactile`, `hw_latch`, `sense_est`, `arm_gains`, `_PERC8`, `_PHYS6`); see SIM_TO_REAL.md §5 | — |
| 36+ | more complexity | a smaller bin, a second object type, two-handed clearing | later |
| — | the drag gate | no level since 17 passes 5 mm at p95; find what moves the rod in the hand during the lift | half a day |
| — | grasp families | two-finger hand, end-face pinch; each a retrain from level 1 | 2 days each |

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
