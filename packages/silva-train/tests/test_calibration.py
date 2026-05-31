import numpy as np

from silva_train.calibration import histogram_specify


def test_histogram_specify_matches_target_distribution():
    # the whole point: the fraction of outputs landing in each 1/L band equals target_fracs.
    rng = np.random.default_rng(0)
    values = rng.normal(size=20000)
    target = [0.07, 0.26, 0.44, 0.20, 0.03]
    s = histogram_specify(values, target)
    L = len(target)
    seg = np.clip((s * L).astype(int), 0, L - 1)
    fracs = [(seg == k).mean() for k in range(L)]
    assert np.allclose(fracs, target, atol=0.01)


def test_histogram_specify_is_monotone_in_values():
    # rank-preserving: sorting by input must sort the output (ordering untouched).
    rng = np.random.default_rng(1)
    values = rng.normal(size=500)
    s = histogram_specify(values, [0.2, 0.2, 0.2, 0.2, 0.2])
    order = np.argsort(values)
    assert np.all(np.diff(s[order]) >= 0)


def test_histogram_specify_stays_in_unit_range():
    values = np.array([-5.0, 0.0, 5.0, 2.0, 1.0])
    s = histogram_specify(values, [0.07, 0.26, 0.44, 0.20, 0.03])
    assert s.min() >= 0.0
    assert s.max() <= 1.0


def test_histogram_specify_smooth_is_monotone_and_bounded():
    # smooth=True replaces the piecewise-linear band map with a monotone cubic CDF;
    # it must still preserve order and stay in [0, 1].
    rng = np.random.default_rng(3)
    values = rng.normal(size=3000)
    s = histogram_specify(values, [7, 26, 44, 20, 3], smooth=True)
    assert s.min() >= 0.0
    assert s.max() <= 1.0
    order = np.argsort(values)
    assert np.all(np.diff(s[order]) >= -1e-9)


def test_histogram_specify_smooth_roughly_tracks_target():
    # smoothing blurs the hard band fractions but must keep the shape: very few in the
    # worst band, few in the best, bulk in the middle.
    rng = np.random.default_rng(4)
    values = rng.normal(size=10000)
    s = histogram_specify(values, [1, 11, 39, 33, 16], smooth=True)
    assert (s < 0.2).mean() < 0.10
    assert (s > 0.8).mean() < 0.30


def test_histogram_specify_normalises_unnormalised_target():
    # target given as percentages (sum=100) must behave like the normalised version.
    rng = np.random.default_rng(2)
    values = rng.normal(size=5000)
    a = histogram_specify(values, [7, 26, 44, 20, 3])
    b = histogram_specify(values, [0.07, 0.26, 0.44, 0.20, 0.03])
    assert np.allclose(a, b)
