"""Validation metrics for the personal aesthetic scorer (computed in 1~5 label space)."""

from __future__ import annotations

import math
from dataclasses import dataclass

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


def big_gap_rate(preds: ArrayLike, targets: ArrayLike, threshold: float = 2.0) -> float:
    """Fraction of predictions off by >= ``threshold`` rating points — the large-gap
    blunders (you=4 / model=1) that the QWK loss targets. Lower is better.

    Uses the continuous prediction (not rounded) vs the integer label, matching the
    ``biggap`` column the sweep script watches against Spearman.
    """
    return float(np.mean(np.abs(_to_numpy(preds) - _to_numpy(targets)) >= threshold))


def is_improvement(current: float, best: float) -> bool:
    """True if ``current`` is a valid (non-NaN) metric strictly better than ``best``.

    Guards early-stopping against degenerate evals (e.g. constant predictions make
    Spearman NaN), which must not count as an improvement.
    """
    return not math.isnan(current) and current > best


_TOP_K_FRACS: dict[str, float] = {"top_1pct": 0.01, "top_5pct": 0.05}


def _continuous_metrics(
    p: np.ndarray,
    t: np.ndarray,
    min_rating: int = 1,
    max_rating: int = 5,
) -> dict[str, float]:
    return {
        "mae": mae(p, t),
        "rmse": rmse(p, t),
        "pearson": pearson(p, t),
        "spearman": spearman(p, t),
        "qwk": quadratic_weighted_kappa(p, t, min_rating, max_rating),
    }


def compute_metrics(
    preds: ArrayLike,
    targets: ArrayLike,
    min_rating: int = 1,
    max_rating: int = 5,
) -> dict[str, float]:
    p, t = _to_numpy(preds), _to_numpy(targets)
    m = _continuous_metrics(p, t, min_rating, max_rating)
    for name, frac in _TOP_K_FRACS.items():
        m[name] = top_k_precision(p, t, frac=frac)
    m["biggap"] = big_gap_rate(p, t)
    return m


@dataclass(frozen=True, slots=True)
class MetricCI:
    value: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.value:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


def _wilson_ci(hits: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials == 0:
        return 0.0, 1.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = hits / trials
    denom = 1 + z**2 / trials
    centre = (p_hat + z**2 / (2 * trials)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denom
    # clamp to [0,1] and pin around p_hat so float residue at the p_hat=0/1 edges
    # can't push the bound past the point estimate (lo <= p_hat <= hi always holds)
    lo = min(p_hat, max(0.0, centre - margin))
    hi = max(p_hat, min(1.0, centre + margin))
    return lo, hi


def bootstrap_ci(
    preds: ArrayLike,
    targets: ArrayLike,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    min_rating: int = 1,
    max_rating: int = 5,
) -> dict[str, MetricCI]:
    """Point estimates + confidence intervals for all metrics.

    Continuous metrics (mae, rmse, pearson, spearman, qwk) use BCa bootstrap.
    Top-k precision uses a Wilson score interval (binomial CI on the hit count),
    because bootstrap resampling with replacement creates duplicates that
    systematically distort rank-based top-k sets.
    """
    p = _to_numpy(preds)
    t = _to_numpy(targets)
    n = len(p)
    rng = np.random.default_rng(seed)

    point = compute_metrics(p, t, min_rating, max_rating)
    cont_keys = list(_continuous_metrics(p, t, min_rating, max_rating))

    boot = np.empty((n_resamples, len(cont_keys)))
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        m = _continuous_metrics(p[idx], t[idx], min_rating, max_rating)
        for j, k in enumerate(cont_keys):
            boot[b, j] = m[k]

    jack = np.empty((n, len(cont_keys)))
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[i] = False
        m = _continuous_metrics(p[mask], t[mask], min_rating, max_rating)
        for j, k in enumerate(cont_keys):
            jack[i, j] = m[k]
        mask[i] = True

    alpha = (1 - confidence) / 2
    z_alpha = stats.norm.ppf(alpha)
    z_1alpha = stats.norm.ppf(1 - alpha)

    result: dict[str, MetricCI] = {}
    for j, k in enumerate(cont_keys):
        theta = point[k]
        b_col = boot[:, j]

        z0 = stats.norm.ppf(np.clip(np.mean(b_col <= theta), 1e-10, 1 - 1e-10))

        jk = jack[:, j]
        diff = jk.mean() - jk
        denom = 6 * (np.sum(diff**2)) ** 1.5
        acc = float(np.sum(diff**3) / denom) if denom > 0 else 0.0

        def _adj(z_a: float) -> float:
            numer = z0 + z_a
            return float(stats.norm.cdf(z0 + numer / (1 - acc * numer)))

        lo = float(np.nanpercentile(b_col, 100 * np.clip(_adj(z_alpha), 0, 1)))
        hi = float(np.nanpercentile(b_col, 100 * np.clip(_adj(z_1alpha), 0, 1)))
        result[k] = MetricCI(value=theta, lo=lo, hi=hi)

    for name, frac in _TOP_K_FRACS.items():
        k = max(1, round(frac * n))
        hits = round(point[name] * k)
        lo, hi = _wilson_ci(hits, k, confidence)
        result[name] = MetricCI(value=point[name], lo=lo, hi=hi)

    # biggap is a proportion over all n samples -> Wilson, same as top-k
    big_lo, big_hi = _wilson_ci(round(point["biggap"] * n), n, confidence)
    result["biggap"] = MetricCI(value=point["biggap"], lo=big_lo, hi=big_hi)

    return result
