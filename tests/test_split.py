from collections import Counter

from silva.data.export_manifest import assign_splits


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
    assert abs(counts["train"] / 1000 - 0.85) < 0.02
    assert abs(counts["val"] / 1000 - 0.10) < 0.02
    assert abs(counts["test"] / 1000 - 0.05) < 0.02
