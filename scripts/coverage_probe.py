"""Rank probe images by how SPARSELY the training set covers them (label-expansion finder).

New-distribution robustness comes from coverage, not regularisation: the model is unstable
exactly where it has never seen a labelled neighbour. This script takes a probe parquet
(any parquet with an ``embedding`` column — e.g. ``danbooru_probe.py --save`` output, which
also carries post_id/tag) and ranks each probe by its mean cosine distance to the k nearest
training embeddings. The sparsest probes are where one new label buys the most robustness:
label those first.

Probes beyond the manifest's own in-domain ceiling (leave-one-out p99 distance) are flagged
``OOD?`` — sparse-but-anime is an expansion candidate, beyond-the-ceiling may be a domain
mismatch instead.

    uv run python scripts/coverage_probe.py --probe data/danbooru_probe.parquet [--k 8 --top 30 --csv data/expand_queue.csv]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from silva_train.coverage import calibrate_threshold, knn_distance_to_reference


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank probe images by training-set coverage sparsity.")
    ap.add_argument("--probe", required=True, help="parquet with an 'embedding' column (danbooru_probe --save output works)")
    ap.add_argument("--manifest", default="data/manifest.parquet")
    ap.add_argument("--k", type=int, default=8, help="nearest training rows to average")
    ap.add_argument("--quantile", type=float, default=0.99, help="in-domain ceiling quantile")
    ap.add_argument("--top", type=int, default=30, help="sparsest rows to print")
    ap.add_argument("--csv", default=None, help="optional CSV to dump the full ranking")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_df = pd.read_parquet(args.manifest, columns=["embedding"])
    ref = torch.tensor(np.stack(ref_df["embedding"].to_numpy()), dtype=torch.float32, device=device)
    probe = pd.read_parquet(args.probe)
    query = torch.tensor(np.stack(probe["embedding"].to_numpy()), dtype=torch.float32, device=device)

    dist = knn_distance_to_reference(query, ref, k=args.k).cpu().numpy()
    ceiling = calibrate_threshold(ref, k=args.k, quantile=args.quantile)

    probe = probe.drop(columns=["embedding"])
    probe["coverage_dist"] = dist
    probe["beyond_ceiling"] = probe["coverage_dist"] > ceiling
    probe = probe.sort_values("coverage_dist", ascending=False)

    q = np.percentile(dist, [10, 50, 90, 99])
    print(f"probes={len(probe)}  train_rows={len(ref_df)}  k={args.k}  device={device}")
    print(f"probe distance p10/p50/p90/p99: {q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f} / {q[3]:.4f}")
    print(f"in-domain ceiling (manifest LOO p{args.quantile * 100:.0f}): {ceiling:.4f}  -> {probe['beyond_ceiling'].sum()} probes beyond it")

    id_col = "post_id" if "post_id" in probe.columns else None
    tag_col = "tag" if "tag" in probe.columns else None
    print(f"\n=== sparsest probes - label these first (top {args.top}) ===")
    for _, r in probe.head(args.top).iterrows():
        ident = f"post_id={int(r[id_col]):<10}" if id_col else ""
        tag = f" tag={r[tag_col]}" if tag_col else ""
        flag = "  OOD?" if r["beyond_ceiling"] else ""
        print(f"  {ident} dist={r['coverage_dist']:.4f}{tag}{flag}")

    if args.csv:
        probe.to_csv(args.csv, index=False)
        print(f"\nsaved {len(probe)} ranked rows -> {args.csv}")


if __name__ == "__main__":
    main()
