"""Where do the 1~5 grades really sit? Print category-anchor estimates from every source.

The rating scale is ordinal, not interval: the rater's own retest shows 3 and 4 are about
half as far apart as their neighbours. This prints anchor estimates side by side so they
can be sanity-checked against each other (and against gut feeling) before training with
``score_anchors: auto``:

  - kNN pseudo-retest over the manifest (what "auto" computes at train start)
  - a real retest sheet, if given: a CSV with post_id,new_score (intra_rater output),
    compared against the manifest labels — the gold standard once enough events exist

    uv run python scripts/estimate_anchors.py [--retest data/intra_rater_filled_*.csv]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from silva_train.anchors import anchors_from_confusion, anchors_from_neighbours


def _print_anchors(label: str, anchors: list[float]) -> None:
    gaps = " ".join(f"{anchors[i + 1] - anchors[i]:.2f}" for i in range(4))
    print(f"  {label:<28} {[round(a, 2) for a in anchors]}   gaps: {gaps}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate non-uniform score anchors from available evidence.")
    ap.add_argument("--manifest", default="data/manifest.parquet")
    ap.add_argument("--k", type=int, default=20, help="neighbours for the pseudo-retest")
    ap.add_argument("--retest", nargs="*", default=[], help="filled re-rating sheet(s): post_id,new_score CSV")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(args.manifest, columns=["post_id", "personal_score", "embedding"])
    emb = torch.tensor(np.stack(df["embedding"].to_numpy()), dtype=torch.float32, device=device)
    scores = torch.tensor(df["personal_score"].to_numpy(), dtype=torch.float32, device=device)

    print(f"manifest rows={len(df)}  k={args.k}  device={device}\n")
    _print_anchors("kNN pseudo-retest (auto)", anchors_from_neighbours(emb, scores, k=args.k))

    for sheet_path in args.retest:
        sheet = pd.read_csv(sheet_path)
        merged = sheet.merge(df[["post_id", "personal_score"]], on="post_id")
        merged = merged[pd.to_numeric(merged["new_score"], errors="coerce").notna()]
        conf = torch.zeros(5, 5)
        for _, r in merged.iterrows():
            conf[int(r.personal_score) - 1, int(r.new_score) - 1] += 1
        _print_anchors(f"retest {sheet_path} (n={len(merged)})", anchors_from_confusion(conf))


if __name__ == "__main__":
    main()
