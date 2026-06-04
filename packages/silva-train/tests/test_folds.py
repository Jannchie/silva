from collections import Counter

from silva_train.data.manifest import assign_folds, assign_splits


def test_fold_values_are_in_range():
    folds = assign_folds([str(i) for i in range(100)], n_folds=5, seed=42)
    assert set(folds) <= set(range(5))


def test_same_key_gets_same_fold():
    keys = ["a", "b", "a", "c", "b", "a", "c"]
    folds = assign_folds(keys, n_folds=3, seed=0)
    seen: dict[str, int] = {}
    for k, f in zip(keys, folds, strict=True):
        if k in seen:
            assert seen[k] == f
        else:
            seen[k] = f


def test_deterministic_with_seed():
    keys = [str(i) for i in range(100)]
    assert assign_folds(keys, n_folds=5, seed=42) == assign_folds(keys, n_folds=5, seed=42)


def test_folds_approximately_balanced():
    counts = Counter(assign_folds([str(i) for i in range(1000)], n_folds=5, seed=42))
    assert set(counts) == set(range(5))
    for f in range(5):
        assert abs(counts[f] / 1000 - 0.2) < 0.04


def test_existing_key_keeps_fold_when_new_keys_added():
    """Incremental relabel/re-export must not move a row's fold: OOF predictions stay
    comparable across manifest updates, same property assign_splits guarantees."""
    before = assign_folds([str(i) for i in range(500)], n_folds=5, seed=42)
    after = assign_folds([str(i) for i in range(1000)], n_folds=5, seed=42)
    assert after[:500] == before


def test_seed_reshuffles_folds():
    keys = [str(i) for i in range(200)]
    assert assign_folds(keys, n_folds=5, seed=1) != assign_folds(keys, n_folds=5, seed=2)


def test_folds_balanced_within_a_split_at_same_seed():
    """Folds must be independent of splits even at the SAME seed.

    The OOF audit folds the train-split rows with the default seed — the same seed
    assign_splits used. If both read the same hash point, train rows occupy only the
    [0, 0.85) band and the last fold collapses to the 0.80~0.85 sliver.
    """
    keys = [str(i) for i in range(5000)]
    splits = assign_splits(keys, seed=42)
    train_keys = [k for k, s in zip(keys, splits, strict=True) if s == "train"]
    counts = Counter(assign_folds(train_keys, n_folds=5, seed=42))
    n = len(train_keys)
    for f in range(5):
        assert abs(counts[f] / n - 0.2) < 0.04, f"fold {f} unbalanced within train split: {counts}"
