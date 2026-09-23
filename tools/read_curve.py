#!/usr/bin/env python3
"""Print a training run's learning curve from its TensorBoard events.

    docker compose exec adam_rl python /opt/tools/read_curve.py \
        logs/rsl_rl/adam_u30_wuji_pipeline_L1/<run>

Columns are rescaled to be readable per EPISODE, not per step: rsl-rl divides
every Episode_Reward/x by the episode length, so a once-per-episode bonus of
weight w appears here as w/150. The `succ%` column is therefore
`Episode_Reward/success * 150 / 20`, i.e. the real success RATE, and the
contact columns are the fraction of episodes that ever earned each bonus.

Read the contact columns together with succ%. A run whose contacts sit at
0.000 has a policy that never touches the object, and more training will not
help -- the only term left with a gradient is the action-rate penalty, which
is minimised by holding still. That is what pipeline stage 2 did for 400
iterations.
"""

import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0})
ea.Reload()


def s(t):
    try:
        return {e.step: e.value for e in ea.Scalars(t)}
    except Exception:
        return {}


su = s("Episode_Reward/success"); rw = s("Train/mean_reward")
th = s("Episode_Reward/thumb_contact"); op = s("Episode_Reward/opposing_contact")
ap = s("Episode_Reward/approach"); dr = s("Episode_Reward/drop")
ar = s("Episode_Reward/action_rate"); st = s("Policy/mean_std")
tc = s("Episode_Reward/table_contact"); ef = s("Episode_Reward/excess_force")
E = 150
print("%6s %9s %7s %8s %8s %9s %7s %7s %6s" %
      ("iter", "reward", "succ%", "thumb", "oppos", "approach", "drop%", "table%", "std"))
ks = sorted(su)
for k in ks:
    if (k - ks[0]) % 25 and k not in (ks[0], ks[-1]):
        continue
    print("%6d %9.3f %6.1f%% %8.3f %8.3f %9.3f %6.1f%% %6.1f%% %6.2f" % (
        k, rw.get(k, 0), su[k] * E / 20.0 * 100, th.get(k, 0) * E / 0.5,
        op.get(k, 0) * E / 1.0, ap.get(k, 0) * E / 2.0,
        dr.get(k, 0) * E / -5.0 * 100, tc.get(k, 0) * E / -3.0 * 100, st.get(k, 0)))
b = max(su, key=lambda k: su[k])
print("\nbest  %.1f%% at iteration %d" % (su[b] * E / 20.0 * 100, b))
print("final %.1f%% at iteration %d" % (su[max(su)] * E / 20.0 * 100, max(su)))
