"""Can a frozen-SigLIP probe separate komone_ushio's good art from low-rated art?

Decides whether the painter-style signal SILVA gets wrong is even PRESENT in the frozen
embedding. Trains a logistic probe with cross-validation on two questions:

  1) komone good (you>=4) vs low-rated (you<=2): high AUC => the good-vs-bad direction for
     this style EXISTS in the embedding; the main head just under-weights it -> reweighting
     / class-balancing can learn it and generalise. Low AUC => SigLIP is blind here -> only
     backbone fine-tuning (v2) can fix it.
  2) komone good vs OTHER good (you>=4 non-komone): high AUC => the style is a recognisable
     cluster -> retrieval features (similarity-to-known-komone) are viable.

CV generalises across held-out komone images, so a high AUC means the probe transfers to the
painter's other images, not that it memorised these.

    uv run --with scikit-learn python scripts/separability_probe.py
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

DB = r"E:/pictoria/server/illustration/images/.pictoria/pictoria.sqlite"


def auc(pos: np.ndarray, neg: np.ndarray, seed: int = 42) -> tuple[float, int, int]:
    n = min(len(pos), len(neg) if neg.ndim == 2 else len(neg))
    rng = np.random.default_rng(seed)
    p = pos[rng.permutation(len(pos))[:n]]
    q = neg[rng.permutation(len(neg))[:n]]
    x = np.vstack([p, q])
    y = np.r_[np.ones(n), np.zeros(n)]
    scores = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), x, y, cv=5, scoring="roc_auc")
    return float(scores.mean()), n, n


def main() -> None:
    df = pd.read_parquet("data/manifest.parquet", columns=["post_id", "personal_score", "embedding"])
    con = sqlite3.connect(DB)
    komone_ids = {r for (r,) in con.execute("SELECT post_id FROM post_has_tag WHERE tag_name='komone_ushio'")}
    con.close()

    df["komone"] = df["post_id"].isin(komone_ids)
    emb = np.stack(df["embedding"].to_numpy()).astype(np.float32)

    komone_good = emb[(df["komone"] & (df["personal_score"] >= 4)).to_numpy()]
    low = emb[(df["personal_score"] <= 2).to_numpy()]
    other_good = emb[(~df["komone"] & (df["personal_score"] >= 4)).to_numpy()]

    print(f"komone total in manifest: {int(df['komone'].sum())}")
    print(f"komone good(>=4): {len(komone_good)}   low(<=2): {len(low)}   other good(>=4): {len(other_good)}\n")

    if len(komone_good) < 15:
        print("too few komone good images for a stable probe; results indicative only\n")

    a1, n1, _ = auc(komone_good, low)
    print(f"[Q1] komone-good vs low-rated   5-fold AUC = {a1:.3f}   (n={n1}/class)")
    print("     -> >0.85: good-vs-bad direction EXISTS in frozen embedding (reweighting can fix)")
    print("     -> ~0.5-0.7: SigLIP largely blind here (needs backbone fine-tune)\n")

    a2, n2, _ = auc(komone_good, other_good)
    print(f"[Q2] komone-good vs other-good  5-fold AUC = {a2:.3f}   (n={n2}/class)")
    print("     -> high: the painter style is a recognisable cluster (retrieval features viable)")


if __name__ == "__main__":
    main()
