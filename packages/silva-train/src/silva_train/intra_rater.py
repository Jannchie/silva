"""Intra-rater reliability: how consistent is the single human rater with themselves?

Every label comes from one person, so test-retest agreement on a blind re-rating IS the
noise ceiling of the whole pipeline: a model cannot agree with the rater more than the
rater agrees with themselves. By attenuation, the expected ceiling on model-vs-label
Spearman is ``sqrt(reliability)`` — if blind re-rating yields 0.85, no amount of training
pushes test Spearman meaningfully past ~0.92, which tells you whether the next lever is
modelling or relabelling.

Two pieces, consumed by ``scripts/intra_rater.py``:

  - :func:`sample_for_rerating` — deterministic score-proportional sample, shuffled so the
    rating sheet leaks no ordering hint.
  - :func:`agreement_report` — test-retest metrics plus the implied model ceiling.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from silva_train.metrics import mae, quadratic_weighted_kappa, spearman

if TYPE_CHECKING:
    import pandas as pd


def sample_for_rerating(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Draw ``n`` rows stratified proportionally by ``personal_score``, shuffled.

    Proportional allocation (largest remainder) mirrors the deployment distribution, so
    the reliability measured on the sample transfers to the test-set metrics. The final
    shuffle removes any score grouping a blind rater could pick up on.
    """
    if n > len(df):
        raise ValueError(f"n={n} exceeds the {len(df)} available rows")
    rng = np.random.default_rng(seed)

    counts = df["personal_score"].value_counts().sort_index()
    quota = counts / len(df) * n
    alloc = {int(s): int(q) for s, q in zip(quota.index, np.floor(quota), strict=True)}
    leftovers = quota - np.floor(quota)
    for s in leftovers.sort_values(ascending=False).index:
        if sum(alloc.values()) >= n:
            break
        alloc[int(s)] += 1

    picked: list[np.ndarray] = []
    for score, k in alloc.items():
        idx = df.index[df["personal_score"] == score].to_numpy()
        picked.append(rng.choice(idx, size=k, replace=False))
    order = rng.permutation(np.concatenate(picked))
    return df.loc[order].reset_index(drop=True)


def agreement_report(old: list[float] | np.ndarray, new: list[float] | np.ndarray) -> dict[str, float]:
    """Test-retest agreement metrics + the attenuation-implied model ceiling.

    ``ceiling_spearman = sqrt(max(0, spearman))``: the highest model-vs-label correlation
    a perfectly faithful model could reach against labels this noisy.
    """
    o = np.asarray(old, dtype=np.float64)
    nw = np.asarray(new, dtype=np.float64)
    reliability = spearman(o, nw)
    return {
        "spearman": reliability,
        "qwk": quadratic_weighted_kappa(o, nw),
        "mae": mae(o, nw),
        "exact": float((o == nw).mean()),
        "ceiling_spearman": math.sqrt(reliability) if reliability > 0 else 0.0,
    }
