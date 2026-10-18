# Experiment 4, step by step

A plain walkthrough of what was done, in order, with the command, the
criterion and the outcome of each step. The detailed numbers, curves and
checkpoint tables are in [`TRAINING_LOG.md`](TRAINING_LOG.md); the design in
[`EXPERIMENT4_DESIGN.md`](EXPERIMENT4_DESIGN.md); every command in
[`COMMANDS.md`](COMMANDS.md) §6b. All commands run from the repo root on the
host. `<n>` is the level, `<run>` the timestamped run directory, `<k>` the
checkpoint iteration.

## The loop that every level follows

Each curriculum level is the same five moves. Everything below is this loop,
repeated.

| move | command | what it answers |
|---|---|---|
| 1. baseline | `docker compose exec -e ADAM_U_RG_LEVEL=<n> adam_rl python /opt/tools/audit_reach_grasp.py` | how well a blind scripted grasp does at this level; the number to beat |
| 2. train | `docker compose exec -e ADAM_U_RG_LEVEL=<n> adam_rl python genesis_rl/scripts/train.py --task wuji_reach_grasp --num_envs 2048 --max_iterations 400 --resume --checkpoint_path <previous best> --reset-std 0.3` | 400 PPO iterations, ~25 min, resumed from the previous level's best |
| 3. pick | read `Outcome/success` and `Outcome/bolt_drag_mm` per saved checkpoint (every 50 iterations) | which checkpoint to evaluate; never the last one by default |
| 4. evaluate | `docker compose exec -e ADAM_U_RG_LEVEL=<n> adam_rl python /opt/tools/eval_reach_grasp.py --checkpoint_path logs/rsl_rl/adam_u30_wuji_reach_grasp_L<n>/<run>/model_<k>.pt --num_envs 256 --episodes 512` | 512 deterministic episodes against the six gates |
| 5. advance | all six gates pass and the policy beats the baseline | the chosen checkpoint becomes `<previous best>` for level n+1 |

The six gates: success ≥ 95 %, invalid collisions = 0, shield interventions
< 1 %, joint-limit terminations = 0, excessive force < 1 %, bolt drag < 5 mm
at the 95th percentile.

---

## Step 0. Build and verify the environment (2026-09-24, morning)

- Wrote the env, the masked PPO, the tools and the tests. Nothing in the
  earlier experiments was modified.
- `tools/solve_reach_grasp_poses.py` produced the arm poses and reset
  Jacobians for 0, 5 and 10 cm standoff. Self-check: reproduces the
  pipeline's pose and Jacobian to 1e-5.
- `tools/test_reach_grasp.py` at level 0: 43 of 43 checks. Among them: the
  latch resets the IK target from the measured pose, fingers are held through
  settle, the lift is vertical to 0.8 mm, masked transitions contribute
  exactly zero actor gradient.
- **Geometry gate**, level 0, `tools/audit_reach_grasp.py`: scripted close
  100 % success, 0 collisions, drag 2.0 mm. Passed. Training allowed.
- One reward changed after measurement: the near-object velocity penalty is
  paid only before fingertip contact, because the literal version charged a
  motionless closing hand −20.

## Step 1. Level 1, 5 cm standoff, no error

- Baseline 100 % (no error to correct; this level teaches the descent).
- Trained from scratch. 95.8 % at iteration 100, 100 % from 200.
- Pick: the final checkpoint dragged the bolt 6.2 mm at worst, over the gate,
  because the policy kept speeding up after success saturated. Iteration 250
  chosen: 100 %, drag 2.3 mm mean.
- Lesson: pick by drag, not by iteration count.

## Step 2. Level 2, ±0.5 cm and ±3° start error

- Baseline 73.6 %. First level that needs wrist correction.
- Resumed from L1 best. 82 % ten iterations after the resume, 100 % by 425.
- Pick: iteration 649, 100 %, drag 2.6 mm mean.
- Lesson: the maximum drag over 512 episodes flips around 5 mm by chance
  while the wrist moves 0.7 mm; the gate became the 95th percentile.

## Step 3. Level 3, 7.5 to 10 cm standoff, ±1 cm, ±5°, ±1° roll and pitch

- Baseline 65.6 %. The first real test of wrist correction.
- Resumed from L2 best. Slowest transfer: 8.6 % at five iterations with 4.7 %
  collisions, 75 % at 25, 99.8 % at 50. The only time collisions exceeded 1 %.
- Pick: iteration 1000, 100 %, 0 collisions, drag 2.1 mm.

## Step 4. Level 4, ±2 cm, ±7.5°, ±2°

- Baseline 49.3 %.
- Resumed from L3 best. 97.8 % within ten iterations; the L3 policy already
  covered most of the range.
- Pick: iteration 1250, 100 %, drag 1.7 mm. The geometric ladder is complete.

## Step 5. Level 5, physics randomisation

- Pre-flight: verified the mass, friction and gain draws actually vary per
  env and are redrawn at reset (0.024 to 0.036 kg, 0.63 to 1.15 friction,
  2.7 to 3.3 stiffness). Tests 40 of 40 at level 5.
- Baseline 41.2 %.
- Resumed from L4 best. 93.6 % at ten iterations, 98.2 % at 25. The action
  std stopped shrinking at 0.28: with random friction there is no single best
  closing speed.
- Pick: iteration 1649, 100 %, peak fingertip force 13 N against the 40 N
  limit. Drag tail longer than before (max 9 mm, p95 2.5 mm).

## Step 6. Level 6, small perception error

- Bolt estimate corrupted: 2 mm bias, 1 mm noise, up to one step of latency.
  Rewards and the success test keep the true pose.
- Baseline 34.6 %.
- Resumed from L5 best. 99.7 % at ten iterations; flat from there. A 2 mm
  bias is inside what contact-driven closure absorbs.
- Pick: iteration 1900, 99.8 %. The one failure in 512 held the bolt below
  5 cm; no drop, no collision.

## Step 7. Level 7, moderate perception error

- 5 mm bias, 3°, 1 to 2 mm noise, one to two steps latency, 2 % dropout.
- Baseline 32.8 %.
- Resumed from L6 best. 98.4 % at ten iterations, 99 % at 25. Std settled at
  0.33, the highest of any level.
- Pick: iteration 2299, 100 %, and also 100 % when evaluated back at level 4
  with exact perception: the perception training cost nothing on the clean
  task.

At this point levels 1 to 7 were done in one afternoon: every level ≥ 99.8 %
with zero invalid collisions over 512 episodes.

## Step 8. Repackaging (2026-09-28)

- The repo became the `genesis_rl` package with a `--task` interface;
  Experiment 4 was committed with it. Regression suite rerun on the new
  layout: 42 of 42. Checkpoints untouched.

## Step 9. Clutter machinery and the corridor measurement

- Added kinematic box obstacles, parked when unused, placed at reset; folded
  them into the clearance model; obstacle contact ends the episode.
- First placement rule, an annulus minus a strip behind the bolt, was
  **wrong**: the untrained L7 policy hit obstacles in 53 % of level-10
  episodes.
- `tools/probe_corridor.py` measured where the arm and hand actually go over
  512 clutter-free episodes: a 646 cm² footprint, stored as `corridor.npz`
  beside the env. Placement now excludes it.
- With the measured corridor the untrained L7 policy: level 8 100 %, level 9
  100 %, level 10 99.4 %, zero obstacle contacts. Conclusion: obstacles
  outside the nominal corridor are never touched, so the clutter levels
  confirm rather than teach.

## Step 10. Level 8, one fixed obstacle

- Baseline 38.6 %, no obstacle hits by any fixed policy.
- Resumed from L7 best. 99.6 % at ten iterations, obstacle hits zero
  throughout, closest approach 4 cm.
- Pick: iteration 2650, 100 %, drag 1.5 mm mean, 4.2 mm max.

## Step 11. Level 9, one random obstacle

- Resumed from L8 best. 99.8 % at ten iterations, zero obstacle hits.
- Pick: iteration 3000, 100 %, all gates. (3049 also passes.)

## Step 12. Level 10, two to three random obstacles

- Resumed from L9. 99.7 % at ten iterations, zero obstacle hits.
- Pick: iteration 3200, 100 %, all gates. Clutter progression complete; it
  confirmed rather than taught, because obstacles stayed outside the
  measured corridor.

## Step 13. Generalisation probe: does the policy handle other bolt poses?

- Level-7 checkpoint, bolt moved at reset, no retraining: yaw ±15° 100 %,
  ±30° 96.5 %, ±45° 80 %, position ±3 cm **48.5 %**. Position is the cliff.
- Arm reach over bolt yaw × position (`solve_reach_grasp_poses.py --grid`):
  with the waist held, only bolt yaws from −45° to +15° are reachable
  everywhere. 582 reachable buckets stored as `pose_buckets.npz`.

## Step 14. Levels 11 to 13, bolt pose sampled at reset

- The bolt's yaw and position are sampled; the nominal wrist pose, the reset,
  the reach reward, the workspace clamp and the obstacle corridor follow it.
  Tests: 52 of 52 at level 7 (nothing regressed), 53 at level 13.
- Untrained L7 policy at level 13 with the bolt-relative start: 24 % success,
  37 % collisions. A real learning problem, unlike clutter.
- Level 11 (yaw ±15°): resumed from L7. Pick iteration 2400, 100 %.
- Level 12 (yaw −30° to +15°, ±2.5 cm): resumed from L11, started at 38.6 %
  with 6.5 % collisions and joint slack down to 0.06, solved by iteration
  2500. Pick 2650, 99.8 %, drag p95 exactly 5.0 mm.
- Level 13 (yaw −45° to +15°, ±5 cm): resumed from L12, 95 % within ten
  iterations. Pick 2800, 100 %, 0 collisions, drag p95 4.5 mm. Checkpoints
  2950 and 3000 fail only the drag gate (5.8 and 5.0 mm).
- The ladder as designed is complete: 13 levels, every pick ≥ 99.8 % with
  zero invalid collisions over 512 episodes.

## Step 15. Level 14, the waist freed: wider bolt yaw (2026-09-28)

- Why: a pile presents bolts at any yaw, and −45° to +15° covers a third of
  the 180° period. The pinch ties the hand's yaw to the bolt's, so the only
  lever is where the shoulder is: the three waist joints.
- `tools/solve_reach_grasp_poses.py --grid --free-waist --min-slack 0.15`
  scans waist pitch × yaw × roll per (bolt yaw, xy) column and keeps one
  held triple per column. Yaw alone (its axis is tilted with the 60° lean)
  bought +15°; with pitch and roll the window is −60° to +45°, ±5 cm inside
  −45° to +30° and ±2.5 cm at the edges. ±75° is unreachable with any hold.
- Env: the standoff reset writes the bucket's waist as joint state and the
  action term holds it as the drive target, exactly as the pitch was held
  before. Levels 0 to 13 are unchanged (their buckets carry the pipeline's
  hold; the level-13 tests rerun to confirm).
- The first grid put the elbow into the table at one hold (upper-arm
  clearance −1.8 cm at reset, caught by the tests); the solver now checks
  the arm proxies against the table itself. 585 of 975 buckets survive.
- Baseline 40.0 %. Trained from the level-13 pick: 7 % collisions at the
  resume, zero from iteration 3050. Pick 3150: 99.2 %, 0 collisions, drag
  p95 3.6 mm, all six gates; and 100 % back at level 13. Checkpoint 3100
  fails two gates (one finger–table touch, drag p95 6.5 mm).
- Window now −60° to +45° of bolt yaw, 105° of the 180° period (was 60°).

## Step 16. Level 15, the grasp point commanded (2026-09-29)

- Why: the graspable-area study needs the grasp point controlled and swept.
- The offset along the rod's axis, ±14 mm, is sampled per episode. The
  nominal wrist pose and the policy's bolt estimate move to that point, so
  the policy grasps there; the physical bolt, latch, drag and success test
  stay on the true object. Interface unchanged, resume from level 14 works.
- Baseline 31.6 %. Trained from the level-14 pick. Pick 3549: 98.8 %,
  0 collisions, drag p95 4.9 mm, all six gates.
- `tools/sweep_grasp_offset.sh` pins the offset and evaluates 512 episodes
  at each of 17 values from −16 to +16 mm. Result: ≥ 99 % across the band
  (±8 mm), 98 % on the thread to ±12 mm, 96 to 97 % at ±14 mm, 93 to 96 % at
  ±16 mm (untrained, 1.5 mm from the end). Drag grows from 3 mm at the
  centre to 6 to 8 mm past −10 mm. Full table in `TRAINING_LOG.md`.

## Step 17. Level 16, the pinch angle commanded (2026-09-29)

- Same device as level 15, one axis further: the estimated rod is yawed by
  a sampled ±30°, so the policy pinches the real rod obliquely. The sampled
  yaw window now bounds the hand's yaw; the rod's is hand yaw minus delta.
- Baseline 32.8 %. Trained from level 15; plateau at 97 %. Pick 3850:
  98.8 %, but one collision in 512 and drag p95 5.6 mm, so **not gated**.
- The 49-cell map (offset −12..+12 mm × angle −45..+45°): 43 cells above
  95 %; weak corners are the rod's ends at large angles (86 % at +45°,
  −12 mm). No angle limit inside ±45°.

## Step 18. Level 17, free grasp without alignment (2026-09-29)

- Commands removed: true rod observed, reach reward to the rod's axis with
  no yaw term, wrist-yaw action at a quarter rate (≤ 7° of turn per grasp).
- Baseline 33.8 %. Trained 600 iterations from level 16. Pick 4400:
  **100 %**, 0 collisions, drag p95 3.5 mm, all six gates.
- Its map: every one of 49 start cells ≥ 98 %, 42 ≥ 99.4 %. The free
  policy beats the commanded one on the same grid.
- Report: `GRASP_AREA_REPORT.md`.

## Step 19. Level 18, the rod standing on its end (2026-09-29)

- The rod could not stand: PhysX cooks the mesh to a 64-vertex hull with
  no flat end. A convex-decomposition copy of the asset
  (`tools/make_bolt_colliders.py`) stands with 0.03° of tilt; levels with
  a standing rod use it. The level-17 policy loses 0.8 points on it.
- Half the episodes stand the rod; the commanded point becomes a height.
  Pick 4750: 99.6 % mixed, 99.8 % standing, zero collisions, drag gate
  missed. But the fingers closed within 3 mm of the band whatever height
  they were told: the point was not binding (measured; see below).

## Step 20. Level 19, the commanded point made binding (2026-09-29)

- The latch now requires the pinch within 4 mm of the commanded point,
  and success requires the latch. Every evaluation prints the measured
  pinch error. Two scoring bugs of mine were found and fixed on the way
  (a zeroed rest-height buffer; success without a latch).
- Pick 5300: 96.7 % standing, 88.7 % lying, pinch error 2 mm mean, 4 mm
  max. The honest maps: standing rod 94 to 99 % on the band, 80 to 88 %
  at the ends; lying rod 85 to 91 % on the band, 64 to 76 % at the ends.
  The earlier level-15, 16 and 18 maps are superseded by these.

## Step 21. Level 20, the rod leaning on a block (2026-09-29)

- A third of the episodes lean the rod on a 10, 17 or 25 mm block: 18°,
  30° or 40° from horizontal. Baseline 11.7 %.
- Pick 5899: 87.6 % mixed; 98.2 % standing, 86.4 % lying, **75.0 %
  tilted**. The tilted map is flat at 70 to 75 % along the rod with the
  latch at 85 to 95 %: the point is reached, but one grasp in ten slips
  out as the lift starts and the rest drag 14 mm. A horizontal pinch on a
  slanted rod pushes it down its own axis. The fix is a wrist roll that
  matches the tilt, a new hand orientation and its own level.

## Step 22. Level 20 continued: point shaping, 600 more iterations (2026-09-29)

- A shaping term on the pinch's distance to the commanded point, binding
  levels only. Pick 6450: 93.4 % mixed (87.6 before); lying 95.7 % (86.4),
  standing 98.0 %, tilted 78.3 % (75.0). The lying rod's latch rate went
  from 92 % to 98 %; the tilted rod's drag did not move.
- Current picks: standing and lying rods at a commanded point are at 96 to
  98 %; the tilted rod needs the wrist roll (level 20b); the drag gate
  needs a slower lift.

## Step 23. Levels 21 to 24, overnight (2026-09-29/30)

Run unattended in one queue: build, test, baseline, 600 iterations from the
previous pick, three-checkpoint evaluation, per-pose split, one map.

- **Level 21, wrist rolled to the rod's tilt.** The pose solver had the
  rod's axis 90° off (every rolled bucket wrong); fixed, the rolled wrist
  is reachable at all three tilts across the window. Pick 7049: tilted
  81.8 % (78.3 before), standing 99.2 %, lying 89.3 %. A trace of 303
  rolled lifts showed the remaining loss: in 14 % of tilted grasps the
  latch fires on the fingers pressing the rod onto its block, and the hand
  lifts away empty. Two global changes came with it: a 40-step lift with a
  slow first centimetre, and drag measured as slip in the hand. A flipped
  tilt convention (which end is raised) mirrored the level-20 policy's
  tilted scenes; measured and logged.
- **Level 22, one retry.** A failed lift or a drop reopens the hand and
  restarts the grasp from 10 cm up. Pick 7648: mixed 96.9 %, standing
  99.8 %, lying 94.4 %, tilted 92.3 %; the retry recovers half to two
  thirds of slipped lifts and improves first attempts too. The best
  policy of the study on every pose.
- **Level 23, a block over one end.** Uncovered poses at their best
  (standing 100 %, lying 97.3 %); the covered rod at 23.9 %, because a rod
  whose far end sits under a block cannot be lifted straight up. The cover
  needs a slide-out motion, not a grasp; dropped from 24 and 25.
- **Level 24, the rod across two rods.** Kinematic cylinders under the
  target, capsules in the clearance model; a spawn-height bug found by the
  tests. Pick 8750: mixed 96.5 %, rod on rods 92.4 %, tilted 93.8 %,
  standing 100 %, flat 95.0 %. Map flat at 92 to 96 % along the rod.
- **Level 25, the bin.** Four walls 13 cm from the nominal position.
  Pick 9349: mixed 92.4 %, standing 93.9 %, lying 91.7 %, on rods 90.8 %,
  tilted 86.2 %, with 3 to 5 % wall contacts by the forearm. A 600-iteration
  continuation (pick 9850) halved the wall contacts: mixed 93.3 %, standing
  98.4 %, lying 94.4 %, on rods 92.4 %, tilted 86.3 %. Current best policy.

## Step 24. Level 26, slide-out and grip check (2026-09-30)

- The grip check fired on 40 to 85 % of good grasps in two versions (the
  held rod lags the wrist in the first millimetres of a lift); turned off.
- The scripted slide-out made the covered rod worse, 10 % against 24 %:
  the drag pulls the fingers across the thread or into the table. Other
  poses gained: tilted 90.3 %, standing 97.3 %, on rods 93.8 %.
- Level 27, the pile, trains next from this pick. The queue after it is in
  the design doc's plan table (levels 28 to 30 and the open items).

## Step 25. Level 27, the pile (2026-09-30)

- Four loose rods around the target in the bin. The first run broke on a
  PhysX GPU buffer overflow at 2048 environments (dropped contacts, 44 %
  collisions); fixed by raising the buffers at pile levels.
- Pick 10900: 88.9 % mixed, tilted 81.5 %, standing 92.6 %, lying 84.8 %,
  no double picks. Same as the untrained level-26 policy: training did not
  help. The neighbours cost first attempts, not safety.

## Step 26. Level 28, the centred pinch (2026-10-06)

- Analysis of level 27: half of the pile failures are latched grasps that
  do not lift. They are shallow pinches on the upper half of the rod
  (fingertips 5 mm higher than on the lifts that held), formed where a
  neighbour stops the fingers; first-attempt lifts go from 41 % with a
  neighbour under 10 mm from the commanded point to 91 % with none within
  45 mm. Neighbours never pinned the target.
- Level 28 adds the depth of the pinch to the latch: the pinch midpoint
  within 6 mm of the rod's axis, binding, with the shaping on the full
  distance to the commanded point. Trained from the level-27 pick.
- Pick 11450: 86.7 % mixed, tilted 85.9 % (from 81.5), lying 85.4 %,
  standing 93.3 %, every latch within 6 mm of the axis. The remaining
  failures are grasps that never latch, not latched grasps that slip.
  At level 26 (cover, bed, no pile) the pick drops to 82.1 % with 7.8 %
  obstacle collisions: the scenes of levels 23 to 26 need to come back
  as a mixed-scene level before a final checkpoint.

## Step 27. Level 29, free grasp in the pile (queued 2026-10-06)

- Level 28 without the commanded point: the policy picks where along the
  rod to pinch, the centred pinch stays binding. The reach term's tilt is
  now against the rolled approach axis, so free grasp works on the tilted
  rod too.
- Pick 12000: 95.3 % mixed (level 28: 86.7), lying 94.1, tilted 92.6,
  standing 98.8, retries halved. Already 92.4 % untrained: the free point
  removes the near-neighbour failures. 3.3 % wall contacts on the lying
  rod to watch.

## Step 28. Level 30, the dense pile (queued 2026-10-06)

- Six neighbours at 3 to 7 cm, one resting across the target in 30 % of
  the lying episodes (the pile's cover), two retries. Eval `--across 1`
  pins it.
- The first test run exposed spawn overlaps (neighbours through the
  support block and the target) that had knocked the target since level
  27; fixed with overlap-free candidate placement, across rod placed
  resting, lying targets only.
- Pick 12550: 98.5 % mixed, lying 94.9, tilted 98.5, standing 98.4,
  pinned by a rod across it 95.5 (1.2 % double picks). Untrained the
  level-29 pick already had 94.9 %.

## Step 29. Level 31, the grip check redone (queued 2026-10-06)

- A 24 mm test lift over the whole pre-lift, measured at the fingertips;
  a latched rod that rose less than 40 % of the tips is retried there,
  40 steps earlier than at the lift's end. Three retries.
- Pick 13149: 98.2 % mixed, lying 96.4, tilted 98.5, standing 99.6,
  rod across 97.5. The check fires on 2 to 12 % of episodes and most of
  those grasps would have lifted; harmless because the retries succeed,
  but not a clean gate yet.

## Step 30. Level 32, the pre-lift as a policy phase (2026-10-06, finished 2026-10-08)

- The first 10 pre-lift steps are the policy's (wrist live, fingers held);
  then the test lift and the scripted lift. The policy can slide or tilt a
  rod out from under a neighbour itself.
- The run stopped at 200 of 600 iterations on 2026-10-06 and was resumed
  from `model_13350.pt` on 2026-10-08 without a std reset.
- Pick 13700: 97.7 % mixed (level 31: 98.2), lying 95.0, tilted 98.4,
  standing 99.6, rod across 95.5. Retries halved and success without a
  retry rose to 90–92 %, but obstacle contacts doubled to 1.2 % and two
  candidates each had a joint-limit termination: the free window moves
  the hand into the pile. No gain over level 31; carried into level 33 as
  designed, to be revisited if collisions stay at 1 %.

## Step 31. Level 33, mixed scenes (2026-10-08)

- Every scene of the ladder drawn per episode in the bin: pile 60 %,
  cover 30 % of lying rods, bed 40 %, plain otherwise. Built so a final
  checkpoint keeps the cover/bed scene the pile levels had lost.
- Unused pile rods park on the floor beside the table; the across rod
  stays off covered targets.
- Pick 14200: 98.1 % mixed, lying 95.1, tilted 99.4, standing 100, across
  96.7, plain bin 98.2, cover 92.0, bed 97.9. The cover block is the weak
  scene (half its retries fail, peak force 194 N); collisions 3 % on the
  lying rod and under the cover. The level-26 row (60.9 %) is not
  comparable: level 26 commands the grasp point.

## Step 32. Level 34, the late grip check (2026-10-08)

- The grip check compares rod and fingertip rise over the second half of
  the test lift only, after the slack is taken up; v3 fired on 12 % of
  good lying grasps.
- Pick 14799: 97.1 % mixed, lying 95.5, tilted 99.2, standing 99.2, across
  97.9, plain bin 96.9, cover 93.6, bed 97.7. Retries roughly halved
  against level 33 and recovery rose; peak forces down (118 N from 194 N
  under the cover). One or two joint-limit terminations per 512 episodes
  on five rows.

## Step 33. Level 35, the heap (2026-10-08, in progress)

- Ten rods in 70 % of episodes, four dropped from 2.3 cm onto the first
  layer and the target; the policy is frozen for 25 steps while the heap
  settles. The heap moves the target up to 5 cm: the policy has to find
  the rod from a start pose that no longer fits it.
- First attempt: the level-34 pick untrained scored 83.3 % (84.2 % with
  the heap forced on, 98.3 % without), and the 2048-env trainer was
  OOM-killed by the host at 115 GB. Per-episode trace: success falls with
  the number of rods touching the target (93 % with none, 73 % with three
  or more, which is half the heap episodes); the policy had never seen a
  touching rod, since levels 33–34 placed neighbours 3 cm or more away.
- Fixes: the settle-in no longer eats the grasp budget and the time
  observation starts at release; the top-rod rule re-labels a lying target
  buried under a dropped rod (the rod on top becomes the target, as a
  perception system would choose); a retry raises the wrist 3 cm before
  the policy resumes; training at 1024 envs for 1200 iterations
  (`logs/chains/chain_L35.sh`). Relaunched 17:49.
- Result: 1200 iterations flat at 0.72–0.76 training success; candidates
  88.5 / 86.8 / 84.0 % mixed, all under the untrained 90.0 %. Pick 15900 for
  the record (heap only 73.3, no heap 92.2, lying 69.9, tilted 90.8,
  standing 93.0, across 78.6, cover 72.1, bed 85.6; collisions 4–9 %). Not
  carried forward: level 36 starts from the level-34 pick.

## Step 34. Level 36, the heap map (prepared 2026-10-08)

- Level 35's training success sat flat at 0.75 for its first 300
  iterations; the trace says the policy cannot tell which side of the
  target is free. Eight appended observations give the free radial
  distance around the estimated rod per sector (`heap_map_obs`).
- `tools/expand_checkpoint.py` widens a 74-input checkpoint to 82 with
  zero weights, so the level-35 pick starts level 36 computing exactly
  what it did. `logs/chains/chain_L36.sh` runs the level from that pick.
- Result: run from the level-34 pick widened; the map is read (its weights
  grow) but training stayed flat at 0.73–0.75. Pick 15998: 90.0 % mixed
  (= untrained), heap only 79.0 (untrained 80.3), no heap 97.1, lying 74.8,
  tilted 90.6, standing 96.3, across 89.1, cover 69.0, bed 86.1. Kept what
  level 35 had lost, gained nothing on the heap. Carried into level 37.

## Step 35. Level 37, the heap with pushing allowed (prepared 2026-10-08)

- The map shows a clean pinch exists in only 10–16 % of heap episodes: the
  fingertips have to push a touching neighbour aside. The clearance penalty
  counted neighbour rods; `push_ok` restricts it to the table and hard
  obstacles. `logs/chains/chain_L37.sh` runs it from a level-36 pick.
- Result: flat at 0.72–0.74; pick 17197 at 87.3 % mixed (untrained 88.7),
  heap 77.4, no heap 91.6 (level 36: 97.1), collisions 3–6 %. The policy
  got looser, not better at pushing. Lever withdrawn.

## Step 36. Level 38, the heap with a 10 mm latch (prepared 2026-10-09)

- A latch trace of the level-36 pick: two thirds of the heap episodes that
  never latch already hold the rod with ≥ 4 N on both sides, 11 mm above
  the axis; the 6 mm centred-pinch rule refuses them. Level 38 loosens it
  to 10 mm and lets the grip check and the retries judge the hold.
  `logs/chains/chain_L38.sh`, run from the level-36 pick beside level 37.
- Result: flat at 0.74–0.76; pick 17100 at 86.5 % mixed (6 mm latch: 90.0).
  Latches rose to 92–94 % of heap episodes but a third fail the grip check.
  Heap only 83.6 % at 512 episodes, 72.8 % at 2048 (level 36: 79.0 / 70.6):
  the heap row either has very large draw noise or drifts across rounds;
  being checked.

## Step 37. Level 39, the transfer level (2026-10-09)

- Decision after the corrected evaluation: stop adding scenes, transfer the
  level-32 pick (`model_13700.pt`, 95.3 %). Level 39 is the level-32 scene
  with the policy and the phase machine on robot-measurable signals: no
  fingertip sensors, a torque-based latch on the estimated rod, the grip
  check and the drop test on the estimated height, camera-grade perception
  noise with up to 4 steps of latency, gains 0.8–1.2 on arm and fingers.
  Procedure in `SIM_TO_REAL.md` §5; `logs/chains/chain_L39.sh`.

## What is not done

- The swept-path safety shield with the α search. α is 1 everywhere, so the
  "shield interventions" gate is vacuous and collision avoidance so far comes
  from the reward, the termination rule and the workspace box. Zero
  collisions across 17 evaluations, but the spec asked for a shield.
- Mesh-accurate proxy radii; the forearm proxy sits inside its 3 cm margin at
  the grasp pose.
- Obstacles that intrude on the corridor's edge, which is what would make
  avoidance a learned behaviour rather than a placement guarantee.
- A latch that checks the grip is a true pinch (balanced sides, rod free
  of its support): 5 to 10 % of latched grasps on raised rods still lift
  away empty; the retry recovers most, a better latch would avoid them.
- A slide-out motion for a rod pinned under a neighbour (level 23's
  finding): a drag phase between latch and lift.
- The drag gate: no level since 17 passes 5 mm at p95; the slow lift did
  not move it.
- Bolt yaws from +45° to +120° (mod 180°), a third of the orientations: out
  of reach of a palm-down pinch from this side at any waist hold. Needs the
  left arm or a different pinch, not a wider reset.
- PhysX self-collision.

## Where things are

| what | where |
|---|---|
| best checkpoints | `logs/rsl_rl/adam_u30_wuji_reach_grasp_L<n>/<run>/model_<k>.pt`, listed per level in `TRAINING_LOG.md` |
| graspable-area report | `GRASP_AREA_REPORT.md` |
| TensorBoard | `tensorboard --logdir logs/rsl_rl`; `Outcome/*`, `Clearance/*`, `Safety/*` per run |
| watch a policy | `docker compose exec -e ADAM_U_RG_LEVEL=<n> adam_rl python /opt/tools/eval_reach_grasp.py --viz kit --num_envs 4 --episodes 20 --real-time --speed 0.25 --start-delay 15 --checkpoint_path ...` |
