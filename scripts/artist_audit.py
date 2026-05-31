"""Which artists do you score most inconsistently vs their visual neighbours?

Aggregates the per-image kNN deviation (see label_audit.py) up to the ARTIST level
(danbooru tag group_id=3). For each artist with enough images it reports your average
label vs the average label of those images' visual neighbours:

  - mean_dev << 0  -> you systematically UNDER-rate this artist (good art you mark low;
                      the komone_ushio pattern) -> prime relabel-up candidates
  - mean_dev >> 0  -> you systematically OVER-rate this artist (you mark high, near-
                      identical images you marked lower) -> a private-favourite bias

    uv run python scripts/artist_audit.py [--k 20 --min-n 5 --top 15]
"""

from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

DB = r"E:/pictoria/server/illustration/images/.pictoria/pictoria.sqlite"


def main() -> None:
    ap = argparse.ArgumentParser(description="Find artists you label inconsistently vs visual neighbours.")
    ap.add_argument("--manifest", default="data/manifest.parquet")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--min-n", type=int, default=5, help="min rated images per artist to report")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(args.manifest, columns=["post_id", "personal_score", "embedding"])
    emb = F.normalize(torch.tensor(np.stack(df["embedding"].to_numpy()), dtype=torch.float32, device=device), dim=1)
    scores = torch.tensor(df["personal_score"].to_numpy(), dtype=torch.float32, device=device)
    n = emb.shape[0]

    neigh_mean = torch.empty(n, device=device)
    bs = 2048
    for start in range(0, n, bs):
        end = min(start + bs, n)
        sims = emb[start:end] @ emb.T
        sims[torch.arange(end - start, device=device), torch.arange(start, end, device=device)] = -1e9
        neigh_mean[start:end] = scores[sims.topk(args.k, dim=1).indices].mean(dim=1)

    df = df.drop(columns=["embedding"])
    df["neigh_mean"] = neigh_mean.cpu().numpy()
    df["dev"] = df["personal_score"] - df["neigh_mean"]

    # map post_id -> artist tags (group_id=3)
    con = sqlite3.connect(DB)
    q = "SELECT pt.post_id, pt.tag_name FROM post_has_tag pt JOIN tags t ON t.name=pt.tag_name WHERE t.group_id=3"
    art = pd.DataFrame(con.execute(q).fetchall(), columns=["post_id", "artist"])
    con.close()

    merged = art.merge(df[["post_id", "personal_score", "neigh_mean", "dev"]], on="post_id", how="inner")
    g = merged.groupby("artist").agg(n=("dev", "size"), your_avg=("personal_score", "mean"),
                                     neigh_avg=("neigh_mean", "mean"), mean_dev=("dev", "mean"))
    g = g[g["n"] >= args.min_n]

    def show(title: str, rows: pd.DataFrame) -> None:
        print(f"\n=== {title} ===")
        print(f"{'artist':<28}{'n':>5}{'your_avg':>9}{'neigh_avg':>10}{'mean_dev':>9}")
        for artist, r in rows.iterrows():
            print(f"{artist:<28}{int(r.n):>5}{r.your_avg:>9.2f}{r.neigh_avg:>10.2f}{r.mean_dev:>+9.2f}")

    print(f"artists with >= {args.min_n} rated images: {len(g)}")
    show(f"UNDER-rated artists (you score LOW, neighbours HIGHER) - top {args.top}", g.sort_values("mean_dev").head(args.top))
    show(f"OVER-rated artists (you score HIGH, neighbours LOWER) - top {args.top}", g.sort_values("mean_dev", ascending=False).head(args.top))


if __name__ == "__main__":
    main()
