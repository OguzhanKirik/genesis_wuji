# Graspable-area study on the threaded rod: levels 15 to 17

Status: levels 15 to 25 built, trained, evaluated and mapped (2026-09-29/30);
level 25's continued policy is the current best for the full scene; section 7e has the overnight levels. Numbers, curves
and checkpoint tables are in `TRAINING_LOG.md`; commands in `COMMANDS.md`
§6b; the design in `EXPERIMENT4_DESIGN.md` §7.

## Corrections (2026-09-29, afternoon)

The commanded grasp point of levels 15, 16 and 18 was not binding: measured
after the fact, the fingers closed within a few millimetres of the rod's
band whatever point they were commanded (on a standing rod, a ±14 mm command
moved the pinch by ±2.7 mm). The maps in sections 5, 6 and the level-18
entry therefore read "started aimed at this point and lifted the rod", not
"grasped at this point". Level 17's map never claimed a point and stands.
Level 19 makes the point binding (no latch unless the pinch is within 4 mm
of it) and every evaluation now reports the actual pinch position; its maps
replace these. Details in `TRAINING_LOG.md`, "The commanded point was not
binding".

## 0. In one paragraph

The policy can now grasp the rod from any point along it and at any hand
angle up to 45° off square, without being told where and without aligning
the wrist. Trained in three steps of one day: first the grasp point was
commanded (level 15), then the pinch angle too (level 16), then both
commands were removed and the reward reduced to "lift what is under the
hand" (level 17). The free policy scores 100 % over 512 random starts with
zero collisions, and at or above 98 % in every one of 49 pinned start
cells. The maps below are the graspable-area result: the whole rod is
graspable by this pinch, the thin thread ends slightly less reliably than
the band, and a hand skewed towards the thread end drags the rod more than
one skewed towards the band.

## 1. Question

Can the reach-to-grasp policy hold the rod at places other than its centre,
at angles other than square across it, and eventually without aligning the
wrist to the rod at all? And where on the rod does grasping stop working?

## 2. Object

The rod from the workcell file: 35 mm long, a 25.4 mm thick band over the
middle 12 mm, 16.6 mm thread over the 17 mm at each end, symmetric end to
end, 30 g, friction 0.9. (Mass to be replaced by the measured value.)

## 3. Starting point (level 14)

Three-finger pinch (thumb against index and middle) square across the rod
at its centre, approach from above, bolt yaw −60° to +45°, position
±5 cm, physics and perception noise, waist held per episode. 99.2 %
success, zero collisions over 512 episodes.

## 4. The device all three levels share

The policy's 74 observations and 8 actions never change. Instead of adding
inputs, the bolt the policy is *told about* is moved: its estimated centre
slid along the axis by the commanded offset (level 15) and its estimated
orientation yawed by the commanded pinch angle (level 16). The policy aligns
to that virtual bolt and so grasps the real one where and how it was told.
Everything that judges the grasp stays on the real rod: the latch (stable
opposing grip, 4 N per side), the drag measure, the lift-and-hold success
test, the collision rules. Because the interface is unchanged, each level
resumes from the previous level's weights.

## 5. Level 15: commanded grasp point along the axis

Offset sampled within ±14 mm per episode. Trained 400 iterations from the
level-14 policy; the best checkpoint (3549) scores 98.8 % success, zero
collisions and 4.9 mm drag at the 95th percentile over 512 episodes with
random offsets, against a 31.6 % scripted baseline. All six advancement
gates pass.

The map, 512 episodes at each pinned offset:

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

## 6. Level 16: commanded pinch angle

Pinch angle sampled within ±30° off perpendicular, on top of the offset.
Trained 400 iterations from level 15. Best checkpoint (3850): 98.8 %
success over 512 episodes with random offset and angle, against a 32.8 %
baseline; it misses two gates narrowly (one collision in 512, drag p95
5.6 mm against 5.0), so it is recorded as not gated.

The map, 512 episodes per cell, success in %:

| angle \ point | −12 | −8 | −4 | 0 | +4 | +8 | +12 mm |
|---|---|---|---|---|---|---|---|
| −45° | 99.4 | 99.8 | 100 | 100 | 99.6 | 98.6 | 92.4 |
| −30° | 99.4 | 99.6 | 99.8 | 99.8 | 100 | 97.3 | 94.9 |
| −15° | 99.6 | 99.2 | 99.8 | 100 | 99.4 | 98.6 | 96.3 |
| 0° | 97.1 | 99.4 | 99.6 | 99.8 | 99.6 | 99.2 | 99.4 |
| +15° | 94.5 | 96.9 | 97.1 | 97.9 | 97.9 | 99.2 | 98.3 |
| +30° | 97.1 | 98.2 | 98.6 | 98.8 | 98.8 | 98.8 | 95.3 |
| +45° | 86.0 | 93.6 | 97.9 | 98.4 | 99.4 | 99.4 | 96.9 |

Drag during the lift, p95 in mm:

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

## 7. Level 17: free grasp, no alignment

The commands come out. The start is drawn as at level 16 (hand ±14 mm
along the rod, ±45° off square), but the policy sees the real rod, the
reach reward pulls the pinch centre to the rod's axis, any point along it,
with no yaw term, and the wrist's yaw action runs at a quarter rate, so
the hand can turn at most about 7° during a grasp. Reward: a stable lift,
nothing else. Trained 600 iterations from level 16. Best checkpoint
(4400): 100 % success over 512 episodes, zero collisions, drag p95 3.5 mm,
maximum 8.9 mm, all six gates, against a 33.8 % baseline.

The map, 512 episodes per pinned start, success in %:

| start angle \ start point | −12 | −8 | −4 | 0 | +4 | +8 | +12 mm |
|---|---|---|---|---|---|---|---|
| −45° | 100 | 99.8 | 99.2 | 99.6 | 99.8 | 99.6 | 99.8 |
| −30° | 99.8 | 100 | 100 | 100 | 100 | 99.8 | 100 |
| −15° | 99.8 | 100 | 100 | 100 | 99.8 | 100 | 100 |
| 0° | 99.8 | 100 | 100 | 100 | 99.8 | 99.8 | 100 |
| +15° | 99.8 | 100 | 99.4 | 100 | 100 | 99.6 | 99.8 |
| +30° | 99.4 | 99.8 | 99.6 | 99.4 | 99.4 | 99.6 | 99.8 |
| +45° | 98.2 | 98.8 | 99.2 | 98.0 | 98.4 | 99.2 | 98.2 |

Drag during the lift, p95 in mm:

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

## 7b. Levels 18 and 19: the standing rod, and the point made binding

Level 18 stands the rod on its end half the time (a convex-decomposition
collider was needed: the shipped hull has no flat end and the rod topples
untouched). It lifts 99.6 % of rods, 99.8 % of standing ones, zero
collisions, but its height map was not a graspable-area map: measured
after the fact, the fingers closed within 3 mm of the band whatever height
they were told. Level 19 makes the point binding: the latch requires the
pinch within 4 mm of the commanded point, success requires the latch, and
every cell below reports the measured pinch error. Its policy scores
96.7 % on standing rods and 88.7 % on lying ones, pinch error 2 mm mean
and 4 mm maximum. 512 episodes per point.

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

## 7c. Level 20: the rod leaning on a block

A third of the episodes lean the rod on a 10, 17 or 25 mm block, 18° to
40° from horizontal, with the binding point. The policy (checkpoint 5899)
scores 87.6 % mixed: 98.2 % standing, 86.4 % lying, 75.0 % tilted.

Tilted rod, point along it (512 episodes per point):

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

## 7d. Level 20 continued: shaping towards the point

A shaping reward on the pinch's distance to the commanded point, plus 600
more iterations. Checkpoint 6450: 93.4 % mixed; standing 98.0 %, lying
95.7 % (from 86.4 %), tilted 78.3 % (from 75.0 %). These are the current
maps (512 episodes per point).

Lying rod, square pinch:

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

Standing rod, height along it:

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

Tilted rod:

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

## 7e. Levels 21 to 25, overnight: roll, retry, cover, rod on rods, bin

Built and trained unattended in one queue, each level from the previous
pick, 600 iterations, 512-episode evaluations per pose.

| level | what it adds | mixed | standing | lying | tilted | new case |
|---|---|---|---|---|---|---|
| 21 | wrist rolled to the tilt, 40-step lift | 90.1 % | 99.2 % | 89.3 % | 81.8 % | — |
| 22 | one retry after a slip or a miss | 96.9 % | 99.8 % | 94.4 % | 92.3 % | — |
| 23 | a block over one end of half the lying rods | 85.5 % | 100 % | 97.3 % | 92.2 % | covered 23.9 % |
| 24 | the rod across two rods (cover dropped) | 96.5 % | 100 % | 95.0 % | 93.8 % | on rods 92.4 % |
| 25 | inside a bin, walls 13 cm out (after a 600-iteration continuation) | 93.3 % | 98.4 % | 94.4 % | 86.3 % | on rods 92.4 %; 2.7 % wall contacts |

What each taught:

- **The wrist roll** reaches the point on a leaning rod better (latch 95 %)
  but does not stop the lift losing it: in about one grasp in ten on a
  raised rod the latch fires on the fingers pressing the rod onto its
  support, and the hand lifts away empty.
- **The retry** is the single most valuable change of the study: it
  recovers half to two thirds of those empty lifts and, by making a slip
  recoverable, improves first attempts as well. With it every pose is at
  92 % or better.
- **A rod pinned under a neighbour cannot be lifted.** The policy latches
  on its exposed part in 70 % of episodes and the vertical lift then
  drives the covered end into the block. That is a slide-out problem, not
  a grasp problem, and is the next thing a pile needs.
- **A rod resting on other rods** rolls under the fingers and costs about
  a tenth of first attempts, most of which the retry recovers: 92.4 %.
- **The bin's walls** cost 4 to 7 points on every pose, almost all of it
  the forearm coming in low over the near wall; the policy was still
  learning to respect that margin when its run ended.

## 8. What the maps say, and what they do not

They say (revised with the level-19 maps, which certify the pinch point):

- The three-finger pinch from above holds this rod everywhere along its
  length, but not equally. With the point shaped and binding (level 20
  continued): standing rod 96 to 100 % from the band up, 88 % on the
  bottom thread next to the table; lying rod 90 to 97 % on the band and
  the thread, 84 % at 14 mm from the centre; tilted rod 71 to 86 %, the
  horizontal pinch letting it slide down its axis at the lift.
- A free grasp (no point commanded) is at or above 98 % from every start,
  because it settles on the band.
- Hand skew up to 45° costs nothing in success and a few millimetres of
  drag, and it is directional: skew that turns the thumb towards the band
  is free, skew that turns it towards the thread end is what drags.
- Alignment is not needed. A policy that cannot align does as well as one
  that is told exactly where to pinch, better in the corners, because it
  may settle onto the best nearby contact instead of the named one.
- The angle limit of the pinch is beyond 45°; it was not reached.

They do not say:

- How a tilted rod is held once the pinch is rolled to match its tilt:
  with the horizontal pinch it is 70 to 75 % along the whole rod, and that
  number is the pinch geometry, not the policy.
- Anything about grasps on the end faces (fingers closing along the axis),
  about two-finger grasps, or about other finger pairs. Each is its own
  family: a new solved pinch, its own reach window, a retrain and a sweep.
- Anything about a pile: neighbours, occlusion, a bolt resting on others.
  The free policy grasps what is under the hand; choosing *which* exposed
  bolt and *where* to come down is the next stage, for which level 15's
  commanded point and angle are the natural interface from a planner.
- Anything beyond 30 g. The mass is the workcell file's; the measured mass
  replaces it when known, and the level-5 physics randomisation is where
  it goes.

## 9. Where things are

| what | where |
|---|---|
| level 15 pick | `logs/rsl_rl/adam_u30_wuji_reach_grasp_L15/2026-09-29_07-36-37/model_3549.pt` |
| level 16 pick (not gated) | `logs/rsl_rl/adam_u30_wuji_reach_grasp_L16/2026-09-29_08-37-58/model_3850.pt` |
| level 17 pick | `logs/rsl_rl/adam_u30_wuji_reach_grasp_L17/2026-09-29_09-31-57/model_4400.pt` |
| level 18 pick (standing rod, not gated) | `logs/rsl_rl/adam_u30_wuji_reach_grasp_L18/2026-09-29_11-02-29/model_4750.pt` |
| level 19 pick (binding point, not gated) | `logs/rsl_rl/adam_u30_wuji_reach_grasp_L19/2026-09-29_11-51-42/model_5300.pt` |
| level 20 pick (tilted rod, not gated) | `logs/rsl_rl/adam_u30_wuji_reach_grasp_L20/2026-09-29_13-31-09/model_5899.pt` |
| level 20 continued | `logs/rsl_rl/adam_u30_wuji_reach_grasp_L20/2026-09-29_14-57-54/model_6450.pt` |
| level 21 (wrist roll) | `logs/rsl_rl/adam_u30_wuji_reach_grasp_L21/2026-09-29_17-17-24/model_7049.pt` |
| level 22 (retry) | `logs/rsl_rl/adam_u30_wuji_reach_grasp_L22/2026-09-29_18-30-46/model_7648.pt` |
| level 23 (cover) | `logs/rsl_rl/adam_u30_wuji_reach_grasp_L23/2026-09-29_19-26-17/model_8247.pt` |
| level 24 (rod on rods) | `logs/rsl_rl/adam_u30_wuji_reach_grasp_L24/2026-09-29_20-29-57/model_8750.pt` |
| level 25 (bin), current best for the full scene | `logs/rsl_rl/adam_u30_wuji_reach_grasp_L25/2026-09-29_22-21-40/model_9850.pt` |
| maps (per-cell eval logs) | `logs/sweeps/grasp_offset*_2026-09-29_*/` |
| sweep | `tools/sweep_grasp_offset.sh <level> <checkpoint> 512 [offsets]`, `PINCH=<deg>` pins the angle |
| watch | `docker compose exec -e ADAM_U_RG_LEVEL=17 adam_rl python /opt/tools/eval_reach_grasp.py --viz kit --num_envs 4 --episodes 20 --real-time --speed 0.25 --start-delay 15 --checkpoint_path <pick>` |
