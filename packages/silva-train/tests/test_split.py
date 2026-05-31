from collections import Counter

from silva_train.data.manifest import assign_splits


def test_same_path_gets_same_split():
    paths = ["a", "b", "a", "c", "b", "a", "c"]
    splits = assign_splits(paths, ratios=(0.5, 0.25, 0.25), seed=0)
    seen: dict[str, str] = {}
    for p, s in zip(paths, splits, strict=True):
        if p in seen:
            assert seen[p] == s
        else:
            seen[p] = s


def test_deterministic_with_seed():
    paths = [str(i) for i in range(100)]
    assert assign_splits(paths, seed=42) == assign_splits(paths, seed=42)


def test_ratios_approximately_respected():
    paths = [str(i) for i in range(1000)]
    counts = Counter(assign_splits(paths, ratios=(0.85, 0.10, 0.05), seed=42))
    assert set(counts) == {"train", "val", "test"}
    assert abs(counts["train"] / 1000 - 0.85) < 0.03
    assert abs(counts["val"] / 1000 - 0.10) < 0.03
    assert abs(counts["test"] / 1000 - 0.05) < 0.03


def test_existing_key_keeps_split_when_new_keys_added():
    """Incremental update: growing the key set must NOT move an existing key's split.

    This is the property the old permutation-based split lacked — adding rows reshuffled
    everyone, leaking past test rows into train. A per-key deterministic hash fixes it.
    """
    before = assign_splits([str(i) for i in range(500)], seed=42)
    before_map = {str(i): s for i, s in enumerate(before)}
    after = assign_splits([str(i) for i in range(1000)], seed=42)
    for i in range(500):
        assert after[i] == before_map[str(i)], f"key {i} moved split when the set grew"


def test_existing_assignments_are_carried_over():
    """An explicit existing map pins keys to their prior split regardless of the hash."""
    keys = [str(i) for i in range(50)]
    existing = {"0": "test", "1": "test", "2": "val"}
    by_key = dict(zip(keys, assign_splits(keys, seed=42, existing=existing), strict=True))
    assert by_key["0"] == "test"
    assert by_key["1"] == "test"
    assert by_key["2"] == "val"
