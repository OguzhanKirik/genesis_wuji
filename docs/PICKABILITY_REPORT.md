# Full-bin pickability report

Which rods in a full bin the reach-to-grasp policy picks reliably, and which it doesn't (2026-10-16).

**Finding:** how high the rod sits in the heap decides the pick more than anything else measured. Rods on top of the
heap (centre ≥ 35 mm above the bin floor) succeed in **66.9 %** of picks, rods low in the heap (< 25 mm) in
**17.8 %**. If perception offers the highest exposed rod, success rises from 47.5 % to about 66.9 %
with no new training.

## Setup

| | |
|---|---|
| Policy | level 51 (`bin_full`) pick, `logs/rsl_rl/adam_u30_wuji_reach_grasp_L51/2026-10-15_20-34-38/model_19950.pt` (fist: `ADAM_U_HELD_CURL=0.95`) |
| Scene | adam-moveit's bin (`workcell.json`: inner 22 × 16 cm, 8 cm walls), centred on the right hand's reach at (0.547, −0.110) |
| Heap | 40–60 rods dropped the way adam-moveit's `isaac_runner` does (5 × 6 × 2 grid, jitter, tilt, spin), settled with the mesh hull, resettled as 22.7 mm cylinders (`tools/make_bin_heaps.py`, `bin_heaps_full.npz`) |
| Target | an exposed rod (nothing resting on it) inside the reachable window (±5 cm, ±2.5 cm at the yaw window's edges); lying / leaning / standing drawn 0.5 / 0.3 / 0.2 |
| Episodes | 3072, heap in every episode (`--fill 1`), policy mean action |
| Measured | the target's situation when the policy starts (after the 15-step settle), true poses (`tools/probe_pickable.py`, `logs/pickable_L51.npz`) |
| Success | lifted and held to the end of the hold, no invalid collision, no excess force, no second rod lifted |

Heights are the rod centre above the bin floor (the table in the sim): a rod lying on the floor sits at ~12 mm,
one standing on the floor at 17.5 mm, one lying on the first layer near 35 mm.

## By pose

| Pose | Share of targets | Success | 95 % interval | Got a grip | Dropped at some point |
|---|---|---|---|---|---|
| lying | 51 % | **56.1 %** | ±2.4 | 66 % | 16 % |
| leaning | 26 % | **50.5 %** | ±3.5 | 62 % | 18 % |
| standing | 23 % | **25.0 %** | ±3.2 | 51 % | 10 % |
| **all** | 100 % | **47.5 %** | ±1.8 | 61 % | 15 % |

"Dropped at some point" includes drops a retry recovered. Standing rods are the weak pose, mostly because they stand on the floor wedged between neighbours (see the next table).

## By height in the heap

| Centre height (mm) | Lying | Leaning | Standing | All |
|---|---|---|---|---|
| < 20 | 3.6 % (n=83) | 8.7 % (n=115) | 21.2 % (n=650) | 17.8 % (n=848) |
| 20–25 | 9.5 % (n=21) | 25.0 % (n=20) | – | 17.1 % (n=41) |
| 25–30 | 33.7 % (n=95) | 39.4 % (n=71) | – | 36.1 % (n=166) |
| 30–35 | 55.6 % (n=694) | 52.7 % (n=205) | n=6 | 55.0 % (n=905) |
| 35–40 | 62.0 % (n=129) | 60.8 % (n=186) | 62.1 % (n=29) | 61.3 % (n=344) |
| 40–45 | 62.7 % (n=177) | 68.9 % (n=122) | n=12 | 65.9 % (n=311) |
| ≥ 45 | 71.5 % (n=382) | 73.1 % (n=67) | n=8 | 71.8 % (n=457) |

Cells with fewer than 15 episodes show only the count. Every pose improves steadily with height. A standing rod on
top of the heap (35–40 mm, 29 episodes) does as well as a lying one.

## By room around the rod (lying and leaning)

Mean free distance from the rod's surface to the nearest neighbour, wall or floor, over 8 directions around its axis.

| Mean free distance | Lying + leaning | Episodes |
|---|---|---|
| 0–5 mm | 31.2 % | 128 |
| 5–10 mm | 36.2 % | 334 |
| 10–15 mm | 51.9 % | 740 |
| 15–20 mm | 64.0 % | 1063 |
| ≥ 20 mm | 57.8 % | 102 |

Room helps mostly where height doesn't: among rods ≥ 35 mm up, requiring ≥ 15 mm of room raises success only from
66.9 % to 68.8 %.

## How picks end

Each episode counted once, in this order: hit an obstacle, never got a grip, dropped, other failure.

| Pose | Success | No grip | Dropped | Hit obstacle | Other failure |
|---|---|---|---|---|---|
| lying | 56 % | 31 % | 3 % | 4 % | 6 % |
| leaning | 51 % | 38 % | 4 % | 1 % | 7 % |
| standing | 25 % | 46 % | 3 % | 3 % | 23 % |

Getting a grip is the main failure for every pose. Second is a grip that latched but still failed the lift or the
hold ("other": the grip check, a rod that never rose, a rod that slipped low); for standing rods it is 23 %. Drops
that end an episode are rare (3–4 %): most drops are caught by a retry.

## Smaller effects

| Rods in the bin | Success | Episodes |
|---|---|---|
| 40–44 | 41.4 % | 1532 |
| 45–49 | 49.7 % | 861 |
| 50–54 | 55.8 % | 471 |
| 55–60 | 64.4 % | 208 |

Fuller bins do better because more targets rest high on the second layer.

| Rod centre to nearest wall | Success | Episodes |
|---|---|---|
| < 40 mm | 44.1 % | 589 |
| 40–50 mm | 43.6 % | 528 |
| 50–60 mm | 46.6 % | 633 |
| 60–70 mm | 50.4 % | 760 |
| ≥ 70 mm | 52.0 % | 562 |

No clear effect: the number of neighbours touching the rod, the rod's tilt (within each pose), the pinch angle.

## Picking rules for perception

What the policy would score if perception only offered rods passing a rule, and how many of the exposed targets pass it.

| Rule for perception | Targets passing | Success | Episodes |
|---|---|---|---|
| Any exposed rod (as trained) | 100 % | **47.5 %** | 3072 |
| Not standing | 77 % | **54.2 %** | 2367 |
| Centre ≥ 35 mm up | 36 % | **66.9 %** | 1112 |
| Centre ≥ 35 mm up, mean free ≥ 15 mm | 21 % | **68.8 %** | 653 |
| Centre ≥ 40 mm up, mean free ≥ 15 mm | 16 % | **71.3 %** | 487 |
| Centre ≥ 45 mm up | 15 % | **71.8 %** | 457 |
| Centre < 25 mm up (avoid) | 29 % | **17.8 %** | 889 |

## Recommendation

1. **Perception: pick the highest exposed rod.** A depth camera gets this directly. Skip rods whose centre is under
   25 mm while higher ones exist; prefer lying or leaning rods to standing ones.
2. **Training (level 52):** draw targets the way perception will (height-weighted), and spend the remaining
   training on the grip, the main failure even on top of the heap.
3. **Bins that empty out:** as the heap shrinks, the remaining rods sit low and success falls toward the 18–20 % of
   the low rods. A shake or tilt of the bin, or a person topping it up, keeps targets high.

## Limits

- The neighbour rods are cylinders, slightly slimmer than the real rod (22.7 mm vs the mesh's 25.4 × 22.7 mm), and
  the box holds more standing rods than adam-moveit's sim (36 / 25 / 39 % lying / leaning / standing vs 27 / 43 / 30).
- Only the reachable middle of the bin was sampled; rods along the walls weren't targets.
- One target per episode, so "targets passing" is a share of the exposed rods drawn, not a count of rods per bin:
  a bin may have no rod passing a strict rule.
- Simulation only. The real camera's estimate, the hand's torque constant and the real rods' friction are not in
  these numbers.

Reproduce:

```
docker compose exec -T -e ADAM_U_HELD_CURL=0.95 -e ADAM_U_RG_LEVEL=51 adam_rl python /opt/tools/probe_pickable.py \
    --checkpoint_path logs/rsl_rl/adam_u30_wuji_reach_grasp_L51/2026-10-15_20-34-38/model_19950.pt \
    --num_envs 256 --episodes 3072 --out logs/pickable_L51.npz --headless
```
