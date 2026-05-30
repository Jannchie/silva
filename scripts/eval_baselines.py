"""Baseline: how well do the external scorers (waifu / aesthetic) rank images by YOUR
taste, compared to the trained head?

Kept entirely OUT of the training manifest so the training contract stays pure. This
script reads the val/test ``post_id``s from the manifest, pulls the external scorers
straight from the pictoria SQLite (ordinary tables — no sqlite-vec needed), aligns on
``post_id``, and reports metrics vs ``personal_score``. Only scale-free metrics
(Spearman, Pearson, top-k) are comparable: the external scorers live on different
numeric ranges, so MAE/RMSE/QWK (which assume the 1~5 scale) are not meaningful here.
"""

from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

from silva.metrics import compute_metrics

DEFAULT_DB = "E:/pictoria/server/illustration/images/.pictoria/pictoria.sqlite"


def main() -> None:
    parser = argparse.ArgumentParser(description="External scorers vs personal score (ranking baseline)")
    parser.add_argument("--manifest", default="data/manifest.parquet")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--split", default="val")
    parser.add_argument("--wandb", action="store_true", help="log each baseline as a wandb run")
    args = parser.parse_args()

    df = pd.read_parquet(args.manifest)
    df = df[df["split"] == args.split][["post_id", "personal_score"]]

    con = sqlite3.connect(args.db)
    waifu = dict(con.execute("SELECT post_id, score FROM post_waifu_scores"))
    aesthetic = dict(con.execute("SELECT post_id, score FROM post_aesthetic_scores WHERE scorer = 'siglip-v2-5'"))
    con.close()

    df["waifu_score"] = df["post_id"].map(waifu)
    df["aesthetic_score"] = df["post_id"].map(aesthetic)

    for col in ("waifu_score", "aesthetic_score"):
        sub = df[[col, "personal_score"]].dropna()
        metrics = compute_metrics(sub[col].to_numpy(), sub["personal_score"].to_numpy())
        line = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[{args.split}] {col} vs personal (n={len(sub)}): {line}")
        if args.wandb:
            import wandb

            run = wandb.init(project="silva", name=f"{col}-{args.split}", reinit=True)
            run.log({f"{args.split}/{k}": v for k, v in metrics.items()})
            run.finish()


if __name__ == "__main__":
    main()
