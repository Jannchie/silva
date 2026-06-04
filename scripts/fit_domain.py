"""Fit the domain-reference artifact: the OOD gate shipped next to the published head.

Photos, 3D renders and memes are "not applicable" to a personal anime-illustration
scorer, not "low score" — but the head will happily emit a confident number for them.
This script subsamples the training embeddings, calibrates the in-domain distance
ceiling (leave-one-out kNN, see silva_train.coverage), and writes a small
``domain_reference.safetensors``. Consumers (the HF Space demo) compare an input's kNN
distance against the ceiling and warn instead of trusting the score.

    uv run python scripts/fit_domain.py [--k 8 --quantile 0.99 --max-rows 4096]
    # then publish it next to the model weights:
    uv run python scripts/fit_domain.py --push Jannchie/silva-aesthetic
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from silva_train.coverage import DomainReference

ARTIFACT_NAME = "domain_reference.safetensors"


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit + save the domain-reference OOD gate artifact.")
    ap.add_argument("--manifest", nargs="+", default=["data/manifest.parquet"], help="real labelled manifest(s); leave synthetic anchors out")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--quantile", type=float, default=0.99)
    ap.add_argument("--max-rows", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=f"outputs/{ARTIFACT_NAME}")
    ap.add_argument("--push", default=None, metavar="REPO_ID", help="also upload to this HF model repo")
    args = ap.parse_args()

    df = pd.concat([pd.read_parquet(m, columns=["embedding"]) for m in args.manifest], ignore_index=True)
    emb = torch.tensor(np.stack(df["embedding"].to_numpy()), dtype=torch.float32)

    ref = DomainReference.fit(emb, k=args.k, quantile=args.quantile, max_rows=args.max_rows, seed=args.seed)
    ref.save(args.out)

    # sanity: the full manifest queried against the shipped subsample should pass ~always
    in_rate = ref.in_domain(emb).float().mean()
    print(f"reference rows={ref.embeddings.shape[0]}  k={ref.k}  threshold={ref.threshold:.4f}")
    print(f"in-domain rate of the full manifest ({len(df)} rows) vs shipped reference: {in_rate:.1%}")
    print(f"saved -> {args.out}")

    if args.push:
        from huggingface_hub import upload_file

        upload_file(
            path_or_fileobj=args.out,
            path_in_repo=ARTIFACT_NAME,
            repo_id=args.push,
            repo_type="model",
            commit_message=f"Add domain reference (k={args.k}, p{args.quantile * 100:.0f} ceiling)",
        )
        print(f"pushed {ARTIFACT_NAME} -> https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
