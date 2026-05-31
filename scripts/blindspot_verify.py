"""Is the residual label inconsistency a true SigLIP blind spot, or just loose neighbours?

label_audit averages over k=20 neighbours, mixing "SigLIP thinks these are near-identical"
with "loosely similar". To tell whether the ~3% inconsistency is a representational blind
spot, look at each image's SINGLE nearest neighbour (highest cosine) and bucket the label
gap by that cosine:

  - high-cosine buckets still show big label gaps  -> TRUE blind spot: SigLIP cannot see the
    difference you see; relabelling can't fix it (only a better backbone can).
  - big gaps only in low-cosine buckets            -> not a blind spot: neighbours were just
    loose; finer labels / tighter neighbours can converge.

The gap MAGNITUDE is the signal: a cosine~0.98 pair you scored 4 apart is the strongest
evidence; a cosine~0.65 pair is not.

    uv run python scripts/blindspot_verify.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify whether residual label inconsistency is a SigLIP blind spot.")
    ap.add_argument("--manifest", default="data/manifest.parquet")
    ap.add_argument("--top", type=int, default=20, help="extreme near-identical-but-far-scored pairs to list")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(args.manifest, columns=["post_id", "personal_score", "embedding"])
    emb = F.normalize(torch.tensor(np.stack(df["embedding"].to_numpy()), dtype=torch.float32, device=device), dim=1)
    scores = torch.tensor(df["personal_score"].to_numpy(), dtype=torch.float32, device=device)
    n = emb.shape[0]

    nn_cos = torch.empty(n, device=device)
    nn_idx = torch.empty(n, dtype=torch.long, device=device)
    bs = 2048
    for start in range(0, n, bs):
        end = min(start + bs, n)
        sims = emb[start:end] @ emb.T
        sims[torch.arange(end - start, device=device), torch.arange(start, end, device=device)] = -1e9
        vals, idx = sims.max(dim=1)  # single nearest neighbour
        nn_cos[start:end] = vals
        nn_idx[start:end] = idx

    own = scores.cpu().numpy()
    nn_score = scores[nn_idx].cpu().numpy()
    cos = nn_cos.cpu().numpy()
    gap = np.abs(own - nn_score)

    print(f"images={n}   (gap = |your score - nearest-neighbour's score|)\n")
    print(f"{'nn cosine bucket':<18}{'n':>8}{'mean gap':>10}{'gap>=2':>9}{'gap>=3':>9}")
    print("-" * 54)
    buckets = [(0.70, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 0.98), (0.98, 1.01)]
    for lo, hi in buckets:
        m = (cos >= lo) & (cos < hi)
        c = int(m.sum())
        if not c:
            continue
        g = gap[m]
        print(f"[{lo:.2f},{hi:.2f}){'':<6}{c:>8}{g.mean():>10.2f}{(g >= 2).mean():>8.1%}{(g >= 3).mean():>8.1%}")

    # strongest blind-spot evidence: near-identical pairs you scored far apart
    df = df.drop(columns=["embedding"])
    pid = df["post_id"].to_numpy()
    strong = np.where((cos >= 0.97) & (gap >= 3))[0]
    order = strong[np.argsort(-cos[strong])]
    print(f"\n=== near-identical (cosine>=0.97) but scored >=3 apart: {len(strong)} pairs (top {args.top}) ===")
    print(f"{'post_id':>10}{'you':>4}  <->  {'nn_post_id':>10}{'nn':>4}{'cosine':>9}")
    for i in order[:args.top]:
        j = int(nn_idx[i])
        print(f"{int(pid[i]):>10}{int(own[i]):>4}  <->  {int(pid[j]):>10}{int(nn_score[i]):>4}{cos[i]:>9.4f}")


if __name__ == "__main__":
    main()
