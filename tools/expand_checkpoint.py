"""Carry an rsl_rl checkpoint over to a wider observation vector.

New observations are APPENDED to the vector (the reach-grasp levels keep every earlier index), so the
actor's and the critic's first layer get zero columns for them, the observation normalisers get
mean 0 / var 1 / std 1 for them, and the Adam moments of the first-layer weights get zero columns.
With zero columns the expanded policy computes exactly what the old one did until training moves
the new weights.

    python tools/expand_checkpoint.py --src <old.pt> --dst <new.pt> --add 8
"""
import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--dst", required=True)
parser.add_argument("--add", type=int, required=True, help="observations appended to the vector")
parser.add_argument("--old-width", type=int, default=None, help="the current width (default: read from the actor)")
args = parser.parse_args()

ck = torch.load(args.src, map_location="cpu", weights_only=False)
old = args.old_width or ck["actor_state_dict"]["mlp.0.weight"].shape[1]
new = old + args.add
print(f"{args.src}: observation width {old} -> {new}")


def widen_cols(t: torch.Tensor) -> torch.Tensor:
    return torch.cat([t, torch.zeros(t.shape[0], args.add, dtype=t.dtype)], dim=1)


changed = []
for part in ("actor_state_dict", "critic_state_dict"):
    sd = ck[part]
    for k, v in list(sd.items()):
        if not torch.is_tensor(v) or v.dim() == 0:
            continue
        if k == "mlp.0.weight":
            assert v.shape[1] == old, (k, v.shape)
            sd[k] = widen_cols(v); changed.append(f"{part}.{k}")
        elif k.startswith("obs_normalizer.") and v.shape[-1] == old:
            fill = 0.0 if k.endswith("_mean") else 1.0
            sd[k] = torch.cat([v, torch.full((*v.shape[:-1], args.add), fill, dtype=v.dtype)], dim=-1)
            changed.append(f"{part}.{k}")
        elif v.dim() >= 1 and v.shape[-1] == old:
            raise SystemExit(f"unexpected tensor with the old width: {part}.{k} {tuple(v.shape)}")

opt = ck.get("optimizer_state_dict")
if opt is not None:
    for idx, st in opt["state"].items():
        for k, v in list(st.items()):
            if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == old:
                st[k] = widen_cols(v); changed.append(f"optimizer.state[{idx}].{k}")

print("widened:", ", ".join(changed))
torch.save(ck, args.dst)
print(f"wrote {args.dst} (iter {ck.get('iter')})")
