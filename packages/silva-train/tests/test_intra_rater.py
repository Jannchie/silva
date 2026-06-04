import math

import pandas as pd
import pytest

from silva_train.intra_rater import agreement_report, sample_for_rerating


def _df(n: int = 400) -> pd.DataFrame:
    # skewed score distribution, like the real manifest
    scores = ([1] * (n // 10) + [2] * (n // 5) + [3] * (n // 2))
    scores += [4] * (n - len(scores) - n // 10) + [5] * (n // 10)
    return pd.DataFrame({"post_id": range(n), "personal_score": scores})


def test_sample_returns_exactly_n_rows():
    out = sample_for_rerating(_df(), n=100, seed=42)
    assert len(out) == 100


def test_sample_is_deterministic():
    a = sample_for_rerating(_df(), n=80, seed=7)
    b = sample_for_rerating(_df(), n=80, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_sample_has_no_duplicate_rows():
    out = sample_for_rerating(_df(), n=120, seed=1)
    assert out["post_id"].is_unique


def test_sample_is_score_proportional():
    df = _df(1000)
    out = sample_for_rerating(df, n=200, seed=3)
    src = df["personal_score"].value_counts(normalize=True)
    got = out["personal_score"].value_counts(normalize=True)
    for s in src.index:
        assert abs(got.get(s, 0.0) - src[s]) < 0.05


def test_sample_is_shuffled_not_grouped_by_score():
    out = sample_for_rerating(_df(), n=100, seed=42)
    assert not out["personal_score"].is_monotonic_increasing
    assert not out["personal_score"].is_monotonic_decreasing


def test_sample_rejects_n_larger_than_dataset():
    with pytest.raises(ValueError, match="n"):
        sample_for_rerating(_df(50), n=100, seed=0)


def test_agreement_report_perfect_agreement():
    r = agreement_report([1, 2, 3, 4, 5, 3, 2], [1, 2, 3, 4, 5, 3, 2])
    assert r["spearman"] == pytest.approx(1.0)
    assert r["exact"] == pytest.approx(1.0)
    assert r["mae"] == pytest.approx(0.0)
    assert r["ceiling_spearman"] == pytest.approx(1.0)


def test_agreement_report_exact_fraction_and_mae():
    r = agreement_report([1, 2, 3, 4], [1, 2, 3, 2])
    assert r["exact"] == pytest.approx(0.75)
    assert r["mae"] == pytest.approx(0.5)


def test_ceiling_is_sqrt_of_reliability():
    old = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    new = [1, 3, 2, 4, 5, 2, 1, 3, 5, 4]
    r = agreement_report(old, new)
    assert r["ceiling_spearman"] == pytest.approx(math.sqrt(r["spearman"]))


def test_ceiling_clamped_at_zero_for_negative_reliability():
    r = agreement_report([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert r["ceiling_spearman"] == 0.0
