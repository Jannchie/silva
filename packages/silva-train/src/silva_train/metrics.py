"""Validation metrics for the personal aesthetic scorer (computed in 1~5 label space)."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

ArrayLike = "list[float] | np.ndarray | object"


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def mae(preds: ArrayLike, targets: ArrayLike) -> float:
    return float(np.mean(np.abs(_to_numpy(preds) - _to_numpy(targets))))


def rmse(preds: ArrayLike, targets: ArrayLike) -> float:
    diff = _to_numpy(preds) - _to_numpy(targets)
    return float(np.sqrt(np.mean(diff**2)))


def pearson(preds: ArrayLike, targets: ArrayLike) -> float:
    return float(stats.pearsonr(_to_numpy(preds), _to_numpy(targets)).statistic)


def spearman(preds: ArrayLike, targets: ArrayLike) -> float:
    return float(stats.spearmanr(_to_numpy(preds), _to_numpy(targets)).statistic)


def quadratic_weighted_kappa(
    preds: ArrayLike,
    targets: ArrayLike,
    min_rating: int = 1,
    max_rating: int = 5,
) -> float:
    p = np.clip(np.rint(_to_numpy(preds)).astype(int), min_rating, max_rating)
    t = np.clip(np.rint(_to_numpy(targets)).astype(int), min_rating, max_rating)
    n = max_rating - min_rating + 1

    observed = np.zeros((n, n), dtype=np.float64)
    for true_r, pred_r in zip(t, p, strict=True):
        observed[true_r - min_rating, pred_r - min_rating] += 1

    idx = np.arange(n)
    weights = (idx[:, None] - idx[None, :]) ** 2 / (n - 1) ** 2

    hist_true = observed.sum(axis=1)
    hist_pred = observed.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / observed.sum()

    denom = float((weights * expected).sum())
    if denom == 0:
        return 1.0
    return 1.0 - float((weights * observed).sum()) / denom


def top_k_precision(preds: ArrayLike, targets: ArrayLike, frac: float) -> float:
    """Fraction of the top-`frac` predicted items that are also in the top-`frac` by true score."""
    p = _to_numpy(preds)
    t = _to_numpy(targets)
    k = max(1, round(frac * len(p)))
    top_pred = set(np.argsort(p)[-k:].tolist())
    top_true = set(np.argsort(t)[-k:].tolist())
    return len(top_pred & top_true) / k


def is_improvement(current: float, best: float) -> bool:
    """True if ``current`` is a valid (non-NaN) metric strictly better than ``best``.

    Guards early-stopping against degenerate evals (e.g. constant predictions make
    Spearman NaN), which must not count as an improvement.
    """
    return not math.isnan(current) and current > best


def compute_metrics(
    preds: ArrayLike,
    targets: ArrayLike,
    min_rating: int = 1,
    max_rating: int = 5,
) -> dict[str, float]:
    return {
        "mae": mae(preds, targets),
        "rmse": rmse(preds, targets),
        "pearson": pearson(preds, targets),
        "spearman": spearman(preds, targets),
        "qwk": quadratic_weighted_kappa(preds, targets, min_rating, max_rating),
        "top_1pct": top_k_precision(preds, targets, frac=0.01),
        "top_5pct": top_k_precision(preds, targets, frac=0.05),
    }
