"""Find unreliable labels by visual-neighbour disagreement (no model training needed).

For every image, looks at its K nearest neighbours in the frozen SigLIP embedding space and
compares your label to the neighbours' labels. A score that deviates sharply from what you
gave to the most visually-similar images is a prime relabel candidate. Unlike the misclass
probe this does NOT depend on the trained head (which can itself be skewed by noisy labels);
it reads label (in)consistency straight off the embedding geometry.

Two lists, both with post_id for cross-checking in pictoria:
  - suspicious HIGH: you rated high, near-identical images you rated much lower
  - suspicious LOW : you rated low, near-identical images you rated much higher

    uv run python scripts/label_audit.py [--k 20 --top 30 --dev 1.5]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit label reliability via embedding nearest neighbours.")
    ap.add_argument("--manifest", default="data/manifest.parquet")
    ap.add_argument("--k", type=int, default=20, help="neighbours per image")
    ap.add_argument("--top", type=int, default=30, help="rows to print per direction")
    ap.add_argument("--dev", type=float, default=1.5, help="|label - neighbour_mean| threshold to count as inconsistent")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(args.manifest, columns=["post_id", "personal_score", "split", "embedding"])
    emb = F.normalize(torch.tensor(np.stack(df["embedding"].to_numpy()), dtype=torch.float32, device=device), dim=1)
    scores = torch.tensor(df["personal_score"].to_numpy(), dtype=torch.float32, device=device)
    n = emb.shape[0]

    # batched cosine top-k (embeddings L2-normalised -> cosine = dot), excluding self
    neigh_mean = torch.empty(n, device=device)
    bs = 2048
    for start in range(0, n, bs):
        end = min(start + bs, n)
        sims = emb[start:end] @ emb.T  # [b, n]
        idx = torch.arange(start, end, device=device)
        sims[torch.arange(end - start, device=device), idx] = -1e9  # drop self
        top_idx = sims.topk(args.k, dim=1).indices
        neigh_mean[start:end] = scores[top_idx].mean(dim=1)

    df["neigh_mean"] = neigh_mean.cpu().numpy()
    df["dev"] = df["personal_score"] - df["neigh_mean"]  # +: you rated higher than neighbours
    df = df.drop(columns=["embedding"])

    n_incon = int((df["dev"].abs() >= args.dev).sum())
    print(f"images={n}  k={args.k}  |dev|>={args.dev}: {n_incon} ({n_incon / n:.1%}) labels inconsistent with their visual neighbours\n")

    hi = df[(df["personal_score"] >= 4) & (df["dev"] >= args.dev)].sort_values("dev", ascending=False)
    print(f"=== suspicious HIGH labels — you rated high, neighbours much lower (top {args.top}) ===")
    print(f"{'post_id':>10} {'you':>4} {'neigh_mean':>11} {'split':>6}")
    for _, r in hi.head(args.top).iterrows():
        print(f"{int(r.post_id):>10} {int(r.personal_score):>4} {r.neigh_mean:>11.2f} {r.split:>6}")

    lo = df[(df["personal_score"] <= 2) & (df["dev"] <= -args.dev)].sort_values("dev")
    print(f"\n=== suspicious LOW labels — you rated low, neighbours much higher (top {args.top}) ===")
    print(f"{'post_id':>10} {'you':>4} {'neigh_mean':>11} {'split':>6}")
    for _, r in lo.head(args.top).iterrows():
        print(f"{int(r.post_id):>10} {int(r.personal_score):>4} {r.neigh_mean:>11.2f} {r.split:>6}")


if __name__ == "__main__":
    main()
