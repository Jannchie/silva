import math

from silva.metrics import (
    compute_metrics,
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


def test_top_k_precision_identical_ranking():
    preds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert top_k_precision(preds, targets, frac=0.2) == 1.0


def test_top_k_precision_reversed_ranking():
    preds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    targets = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert top_k_precision(preds, targets, frac=0.2) == 0.0


def test_compute_metrics_keys():
    m = compute_metrics([1, 2, 3, 4], [1, 2, 3, 4])
    assert {"mae", "rmse", "pearson", "spearman", "qwk", "top_1pct", "top_5pct"} <= set(m)
    assert m["spearman"] == 1.0
