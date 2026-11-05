"""Widen a reach-grasp checkpoint's critic for ADAM_U_RG_PRIV_CRITIC=1.

The critic's first layer and its observation normaliser grow by K inputs (the env_cfg CriticPrivCfg group, appended
after the policy group). The new weight columns, and their Adam moments, are zero and the new normaliser entries are
mean 0 / var 1, so the widened critic gives exactly the old values until training moves it. The actor is untouched.

    python tools/widen_critic.py IN.pt OUT.pt [--extra 16]
"""
import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--extra", type=int, default=16)
a = ap.parse_args()

ck = torch.load(a.src, map_location="cpu", weights_only=False)
c = ck["critic_state_dict"]
w = c["mlp.0.weight"]
h, n = w.shape
k = a.extra
c["mlp.0.weight"] = torch.cat([w, torch.zeros(h, k, dtype=w.dtype)], dim=1)
for name, fill in (("obs_normalizer._mean", 0.0), ("obs_normalizer._var", 1.0), ("obs_normalizer._std", 1.0)):
    t = c[name]
    c[name] = torch.cat([t, torch.full((t.shape[0], k), fill, dtype=t.dtype)], dim=1)

# Adam moments: the optimizer holds chain(actor, critic) parameters, so the critic's first layer is the LAST state
# entry shaped like it (the actor's first layer has the same shape and comes earlier).
st = ck["optimizer_state_dict"]["state"]
hits = [i for i in sorted(st) if tuple(st[i]["exp_avg"].shape) == (h, n)]
assert len(hits) == 2, hits
i = hits[-1]
for m in ("exp_avg", "exp_avg_sq"):
    t = st[i][m]
    st[i][m] = torch.cat([t, torch.zeros(h, k, dtype=t.dtype)], dim=1)
torch.save(ck, a.dst)
print(f"critic input {n} -> {n + k}; optimizer state {i} widened; wrote {a.dst}")
