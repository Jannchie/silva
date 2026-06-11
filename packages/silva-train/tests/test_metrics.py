import math

import numpy as np

from silva_train.metrics import (
    MetricCI,
    _wilson_ci,
    big_gap_rate,
    bootstrap_ci,
    compute_metrics,
    is_improvement,
    mae,
    pearson,
    quadratic_weighted_kappa,
    rmse,
    spearman,
    top_k_precision,
)


def test_mae_simple():
    assert mae([1, 2, 3], [1, 2, 4]) == 1 / 3


def test_rmse_simple():
    assert rmse([1, 2, 3], [1, 2, 4]) == math.sqrt(1 / 3)


def test_pearson_perfect_positive():
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == 1.0


def test_spearman_perfect_positive():
    # Monotonic but non-linear -> Spearman 1.0
    assert spearman([1, 2, 3, 4], [1, 4, 9, 16]) == 1.0


def test_spearman_perfect_negative():
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_qwk_perfect_agreement():
    assert quadratic_weighted_kappa([1, 3, 5], [1, 3, 5], min_rating=1, max_rating=5) == 1.0


def test_qwk_chance_agreement_is_zero():
    # O == E for this layout -> kappa 0.0 (hand-computed)
    assert quadratic_weighted_kappa([1, 2, 1, 2], [1, 1, 2, 2], min_rating=1, max_rating=2) == 0.0


def _qwk_loop_reference(preds, targets, min_rating=1, max_rating=5):
    """Pre-vectorisation reference: fills the confusion matrix with a Python loop."""
    p = np.clip(np.rint(np.asarray(preds, dtype=np.float64)).astype(int), min_rating, max_rating)
    t = np.clip(np.rint(np.asarray(targets, dtype=np.float64)).astype(int), min_rating, max_rating)
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


def test_qwk_matches_loop_reference_on_random_data():
    rng = np.random.default_rng(7)
    for _ in range(20):
        preds = rng.integers(1, 6, size=200)
        targets = rng.integers(1, 6, size=200)
        assert quadratic_weighted_kappa(preds, targets) == _qwk_loop_reference(preds, targets)


def test_top_k_precision_identical_ranking():
    preds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert top_k_precision(preds, targets, frac=0.2) == 1.0


def test_top_k_precision_reversed_ranking():
    preds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    targets = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert top_k_precision(preds, targets, frac=0.2) == 0.0


def test_big_gap_rate_no_gaps():
    assert big_gap_rate([1, 2, 3, 4], [1, 2, 3, 4]) == 0.0


def test_big_gap_rate_counts_only_gaps_at_or_above_threshold():
    # diffs: 0, 1, 2, 3 -> two of four are >= 2
    assert big_gap_rate([1, 2, 1, 1], [1, 3, 3, 4]) == 0.5


def test_big_gap_rate_uses_continuous_prediction():
    # 1.6 vs 4 is a 2.4 gap (would be a 2.0 gap if rounded to 2) -> still counts
    assert big_gap_rate([1.6], [4]) == 1.0


def test_compute_metrics_keys():
    m = compute_metrics([1, 2, 3, 4], [1, 2, 3, 4])
    assert {"mae", "rmse", "pearson", "spearman", "qwk", "top_1pct", "top_5pct", "biggap"} <= set(m)
    assert m["spearman"] == 1.0
    assert m["biggap"] == 0.0


def test_is_improvement_better():
    assert is_improvement(0.5, 0.4) is True


def test_is_improvement_worse():
    assert is_improvement(0.4, 0.5) is False


def test_is_improvement_equal_is_not_improvement():
    assert is_improvement(0.5, 0.5) is False


def test_is_improvement_nan_is_never_improvement():
    assert is_improvement(float("nan"), -math.inf) is False


# --- bootstrap_ci ---


def test_bootstrap_ci_returns_all_metric_keys():
    rng = np.random.default_rng(0)
    p = rng.uniform(1, 5, size=200)
    t = np.clip(np.rint(p + rng.normal(0, 0.5, size=200)), 1, 5)
    result = bootstrap_ci(p, t, n_resamples=50)
    assert set(result) == set(compute_metrics(p, t))


def test_bootstrap_ci_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    p = rng.uniform(1, 5, size=200)
    t = np.clip(np.rint(p + rng.normal(0, 0.5, size=200)), 1, 5)
    result = bootstrap_ci(p, t, n_resamples=200)
    for k, m in result.items():
        assert m.lo <= m.value <= m.hi, f"{k}: {m.lo} <= {m.value} <= {m.hi}"


def test_bootstrap_ci_perfect_predictions_tight_interval():
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
    result = bootstrap_ci(p, t, n_resamples=200)
    assert result["mae"].value == 0.0
    assert result["mae"].hi < 0.01


def test_bootstrap_ci_deterministic_with_seed():
    p = np.array([1.5, 2.3, 3.7, 4.1, 2.9] * 10)
    t = np.array([2, 2, 4, 4, 3] * 10)
    a = bootstrap_ci(p, t, seed=123)
    b = bootstrap_ci(p, t, seed=123)
    for k in a:
        assert a[k].lo == b[k].lo
        assert a[k].hi == b[k].hi


def test_metric_ci_str_format():
    m = MetricCI(value=0.738, lo=0.715, hi=0.762)
    assert str(m) == "0.7380 [0.7150, 0.7620]"


def test_bootstrap_ci_top_k_uses_wilson_interval():
    """top_k CIs should always contain the point estimate (Wilson, not bootstrap)."""
    rng = np.random.default_rng(2)
    p = rng.uniform(1, 5, size=200)
    t = np.clip(np.rint(p + rng.normal(0, 0.5, size=200)), 1, 5)
    result = bootstrap_ci(p, t, n_resamples=50)
    for name in ("top_1pct", "top_5pct"):
        m = result[name]
        assert m.lo <= m.value <= m.hi, f"{name}: {m.lo} <= {m.value} <= {m.hi}"


def test_wilson_ci_contains_proportion():
    lo, hi = _wilson_ci(7, 20)
    assert lo <= 7 / 20 <= hi


def test_wilson_ci_perfect_hit():
    lo, hi = _wilson_ci(50, 50)
    assert lo > 0.9
    assert hi == 1.0


def test_wilson_ci_zero_hits():
    lo, hi = _wilson_ci(0, 50)
    assert lo < 1e-10
    assert hi < 0.1
