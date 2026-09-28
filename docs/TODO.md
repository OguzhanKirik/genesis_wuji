# What to do in this repo

State as of 2026-09-19. Read [`../README.md`](../README.md) for the environment,
[`ALGORITHM.md`](ALGORITHM.md) for how the learning works,
[`COMMANDS.md`](COMMANDS.md) for every command, and
[`TRAINING_LOG.md`](TRAINING_LOG.md) for what has actually been run and measured.

---

## Where things stand

| | status |
|---|---|
| **Experiment 1** — Adam-U + Wuji, arm hovering over the bolt, 27 actions | trained, **99.4 %** of episodes end with the bolt held |
| **Experiment 2** — hand only, fixed wrist, 2 synergies, 3 fingers | trained, **100 %** stable grasps over 512 episodes |
| **Experiment 2** — same, 2 fingers | trained, **0 %**; the failure is understood (see below) |
| **Experiment 3** — imitation then reinforcement, 20 DoF | env, wire protocol, teleop bridge and recorder built and validated end to end **with a synthetic glove**; no learning yet |
| Reach scenario (arm starts beside the torso) | smoke-tested only, never trained |
| **Experiment 4** — safe reach-to-grasp, standoff + safety, masked PPO | **levels 1–7 trained 2026-09-24, ≥ 99.8 % each, zero collisions**; next: the swept-path shield, mesh-accurate proxies, clutter level 8 (`EXPERIMENT4_DESIGN.md` §12) |

---

## 1. Commit the work

**Nothing here is committed.** 31 paths are modified or untracked, including
~77 MB of generated robot assets, four environments, fifteen tools, four
documents and four trained runs.

Decide before committing:

- **The USD assets.** `assets/robots/adam_u_wuji/` is 59 MB and
  `assets/workcell/` is 17 MB. Both are reproducible from `adam-moveit` plus
  the tools here, but the PND robot's USD was committed deliberately (see the
  README) because regenerating it needs an image this repo no longer builds.
  The same argument applies to these. Recommend committing them.
- **`adam_u_isaac_lab/demos/`** held scratch recordings from the recorder check.
  Those files are deleted. The directory stays gitignored so a new recording
  is not committed by accident.
- **`logs/`** is already gitignored, so the four trained runs are local only.
  If the 100 % checkpoint matters, copy it somewhere durable.

---

## 2. Experiment 3 — finish the pipeline

Built and working: the environment (`wuji_teleop_grasp_env_cfg.py`), the UDP
protocol (`tools/glove_link.py`), the Manus retarget bridge
(`tools/manus_wuji_bridge.py`) and the recorder (`tools/record_demo.py`).
Validated end to end against a synthetic glove: five out of five episodes
recorded and labelled SUCCESS, no packet loss.

### 2a. Unblock the hardware — this is the only real blocker

The glove publisher has been reporting `left: 0 glove msgs/5s  right: 0` for
hours and `/manus_glove_0` has **zero publishers**, because that node creates a
topic per *detected* glove. The SDK is seeing nothing.

```bash
lsusb | grep -i 0483                        # the Manus dongle
cd /home/revel/manus_ws && docker compose restart manus
docker logs -f manus_ros2 | grep -i glove   # want a nonzero rate
```

ROS_DOMAIN_ID is **97** and RMW is **cyclonedds**, both from `manus_ws/.env`.
The host has ROS 2 Jazzy but no CycloneDDS, so the bridge must run **inside**
the `manus_ros2` container. It is already copied to `/opt/rl_tools` there,
but that is lost on `docker compose down`; make it permanent by adding a volume
to `manus_ws/docker-compose.yml`:

```yaml
- /home/revel/adam-rl/tools:/opt/rl_tools:ro
```

### 2b. Collect 20–50 demonstrations

Record **one** first and check it before collecting in bulk.

```bash
# terminal 1 — the bridge
docker exec -it manus_ros2 bash -lc \
  'source /ws/install/setup.bash && python3 /opt/rl_tools/manus_wuji_bridge.py \
     --side right --calib /opt/rl_tools/hand_retarget.yaml'

# terminal 2 — the recorder, with the viewport so you can see what you are grasping
cd /home/revel/adam-rl && docker compose exec adam_rl \
  python /opt/tools/record_demo.py --viz kit --episodes 25 --out /workspace/demos
```

Only the first 2 s of each 4 s episode is yours; the lift and hold ignore your
input, because the lift is what tests whether the grasp you chose holds. So the
grasp must be complete by the two-second mark.

**The open question this answers**, and it is the one that decides the
experiment: *can a human produce a working grasp through the glove from this
pre-grasp pose?* The fingertips start 1.6 cm from the bolt's surface with a
5.2 cm thumb-to-finger gap. If the answer is no, the pose needs re-solving with
`tools/find_pinch_pose.py` before any data is worth collecting.

### 2c. Calibrate the remaining glove channels

`hand_retarget.yaml` measures six channels. The bridge derives the eight PIP and
DIP channels from the same finger's MCP range — an assumption, not a
measurement. Extend `manus_teleop/hand_calibrate.py` to record all 20, or at
least the 15 flexion channels. The four finger Spread channels do not matter
while `--abduction` stays off.

### 2d. Build the data tooling — not started

Per `EXPERIMENT3_DESIGN.md` section 6:

- `tools/demo_io.py` — load NPZ/CSV, validate the joint order against the env,
  resample to 30 Hz, clamp to limits, drop invalid trajectories with a reason,
  split **by trajectory** into train and validation. Never split timesteps from
  one trajectory across both.
- `tools/replay_demo.py` — play a recorded trajectory back as open-loop joint
  targets and check it still succeeds. **This is the test that the recording is
  faithful.** If a demonstration that succeeded live fails on replay, the
  recording or the retarget is wrong and no amount of cloning will fix it.

### 2e. Behaviour cloning — not started

Regression from observation to *commanded* action over the grasp phase of
successful trajectories. Gaussian negative log-likelihood so the learned
standard deviation carries into PPO. Same network as the PPO actor, 256 → 256
for 66 inputs and 20 outputs. Fit the observation normaliser on the
demonstrations and **freeze it into the checkpoint** — if PPO starts with a
different normaliser the inputs shift underneath the policy and the
initialisation is destroyed. Validate by rolling the BC policy out in the
environment, not just by action error.

### 2f. PPO fine-tuning — not started

`runner.load(bc_ckpt, strict=False)`; rsl-rl supports partial loading. The
critic starts from scratch, so expect noisy early iterations. Drop the learning
rate to about 1e-4 and start the exploration standard deviation at the BC value,
well under Experiment 2's 0.5 — a large initial std walks straight off the
demonstrated manifold and throws the initialisation away.

### 2g. The comparison that makes it conclusive

Four policies on the same 512-episode protocol: PPO-only at 20 DoF, BC-only,
BC then PPO, and Experiment 2's 3-finger policy as the 100 % reference. Plus a
sweep of how many demonstrations BC needs: 5, 10, 20, 50.

It is conclusive if **PPO-only at 20 DoF fails and BC+PPO succeeds**. If
PPO-only also reaches 100 %, the honest conclusion is that this task does not
need demonstrations, and the comparison should move to a harder task rather
than claim a benefit that is not there.

---

## 3. Open threads from Experiments 1 and 2

- **The 2-finger pinch, 0 %.** A two-finger grasp exists — the scripted sweep
  finds it — but at one closure combination out of sixteen, against four for
  three fingers. PPO never lands in it. The cause is a reward-design trap worth
  remembering: the policy banked the full +2.0 of contact bonuses in every
  episode while its mean action sat fully open, because exploration noise alone
  brushes the bolt often enough to trip a 4-step stable contact. **A reward an
  untrained policy already collects in full provides no gradient.** Three ways
  out, none tried: a dense shaping term (grip force, or bolt height during the
  lift), a pose with more squeeze margin, or a curriculum from the 3-finger
  policy. Raising entropy would not help — the problem is a flat mean, not
  narrow exploration.
- **No randomisation anywhere.** Every experiment uses one fixed bolt pose and
  one fixed hand pose, so the learned policies are close to constants. Adding
  ±5 mm, ±1 cm and ±2 cm of bolt jitter and measuring where success falls off
  would say how wide the learned grasp actually is. This is the cheapest next
  result in the repo.
- **The bin layout is untested as a task.** The bin's inner width is 16 cm and
  its walls 8 cm tall; whether the hand can reach a bolt inside without fouling
  them is unknown.
- **Experiment 1's reward weights were never tuned** — they were carried over
  from the PND cube task and worked anyway.
- **The reach scenario has never been trained.**

---

## 4. Known traps, so they are not rediscovered

Each of these cost real time and is documented where it bites.

- **Isaac Lab skips zero-weight reward terms.** A weight-0 bookkeeping term
  never runs.
- **Contact sensor filters must resolve to one collider entry per environment.**
  A broken filter reports zero force silently, which is the worst failure mode
  for a contact reward or a success test.
- **Asset data are warp `ProxyArray`s, not tensors.** Their `.shape` is the warp
  shape, so a quaternion reports `(N,)`. Take `.torch` before returning or
  indexing.
- **Joint drive targets initialise to zero**, and reset events write joint
  *states* only, so an unactuated joint holds 0 rather than its default pose.
- **A prismatic joint's axis rotates with its parent.** A nominal vertical lift
  on a rotated hand travelled sideways and dragged the bolt 7 cm while
  reporting a correct 10 cm rise.
- **Overlapping joint regexes raise.** A joint matching two dict keys is an
  error, not a precedence rule.
- **Isaac Lab resets an environment inside the same `step()` that ends it**, so
  anything read afterwards belongs to the next episode. Episode outcomes are
  copied into `last_*` buffers before the reset for exactly this reason. This
  one produced a confident 0 % success report for a policy that was succeeding
  99 % of the time.
- **`Episode_Reward/x` is divided by the episode length.** Multiply by 120 to
  read a once-per-episode reward. A logged 0.16 is a solved task.

---

## 5. Longer term

- **Pose randomisation, then wrist control**, in that order. The spec for both
  Experiments 2 and 3 says not to add wrist learning until the fixed-bolt policy
  is reliable, and it now is for three fingers.
- **Transfer to the real hand.** The 20-joint order here is verified identical
  to adam-moveit's `right_hand_controller`, so a policy trained here emits
  commands in the layout the real Wuji driver expects. What is missing is the
  bridge in the other direction and any treatment of the sim-to-real gap.
- **adam-moveit's Wuji path is scaffolding.** The plugin, joint map, controllers
  and SRDF exist, but its own design note says nothing downstream knows these
  hands and the command path is unbuilt. If a trained policy is ever to drive
  the real hand through that stack, that work is a prerequisite.
