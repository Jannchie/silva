"""Rank relabel candidates by OUT-OF-FOLD disagreement (memorisation-free mislabel signal).

misclass_probe reads the in-fold gap: the production head can memorise the train set, so a
noisy label it managed to fit shows no gap at all. This audit predicts every row with a
model that NEVER saw its label (k-fold cross-validation over content-keyed folds) — the
prediction is what the rest of your data implies the row should score. The biggest
|OOF - label| gaps are the rows where your label most conflicts with your own learnable
taste: relabel those first, every minute of relabelling lands where it pays the most.

Folds are keyed by embedding content (same mechanism as splits), so the queue stays
stable across manifest re-exports.

    uv run python scripts/oof_audit.py [--folds 5 --epochs 40 --min-gap 1.0 --csv data/oof_queue.csv]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from silva_train.data.manifest import assign_folds, embedding_key
from silva_train.oof import make_fit_head, oof_predictions


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank relabel candidates by out-of-fold disagreement.")
    ap.add_argument("--manifest", default="data/manifest.parquet")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-gap", type=float, default=1.0, help="|oof - label| threshold for the queue")
    ap.add_argument("--top", type=int, default=30, help="rows to print per direction")
    ap.add_argument("--csv", default=None, help="optional CSV to dump the full relabel queue")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(args.manifest, columns=["post_id", "personal_score", "split", "embedding"])
    emb = torch.tensor(np.stack(df["embedding"].to_numpy()), dtype=torch.float32, device=device)
    scores = torch.tensor(df["personal_score"].to_numpy(), dtype=torch.float32, device=device)
    folds = assign_folds([embedding_key(e) for e in df["embedding"]], n_folds=args.folds, seed=args.seed)

    print(f"images={len(df)}  folds={args.folds}  epochs={args.epochs}  device={device}")
    fit = make_fit_head(epochs=args.epochs, seed=args.seed, device=device)
    preds = oof_predictions(emb, scores, folds, fit)

    df = df.drop(columns=["embedding"])
    df["oof"] = preds.numpy()
    df["gap"] = df["oof"] - df["personal_score"]
    df["absgap"] = df["gap"].abs()

    print(f"\n=== |oof - label| >= {args.min_gap} by split ===")
    for s in ("train", "val", "test"):
        d = df[df.split == s]
        bad = d[d.absgap >= args.min_gap]
        print(f"  {s:<5} n={len(d):<6} queue={len(bad):<5} ({len(bad) / max(1, len(d)):.1%})")

    fmt = "  post_id={:<10} you={} oof={:.2f} split={}"
    print(f"\n=== relabel queue - you rated LOW, the rest of your data says HIGH (top {args.top}) ===")
    for _, r in df[df.gap > 0].sort_values("gap", ascending=False).head(args.top).iterrows():
        print(fmt.format(int(r.post_id), int(r.personal_score), r.oof, r.split))

    print(f"\n=== relabel queue - you rated HIGH, the rest of your data says LOW (top {args.top}) ===")
    for _, r in df[df.gap < 0].sort_values("gap").head(args.top).iterrows():
        print(fmt.format(int(r.post_id), int(r.personal_score), r.oof, r.split))

    if args.csv:
        queue = df[df.absgap >= args.min_gap].sort_values("absgap", ascending=False)
        queue.to_csv(args.csv, index=False)
        print(f"\nsaved {len(queue)} queue rows -> {args.csv}")


if __name__ == "__main__":
    main()
