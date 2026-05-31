import pandas as pd
import pytest

from silva_train.data.manifest import build_manifest, diff_manifests, merge_manifests, validate_manifest, write_manifest


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "embedding": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
            "personal_score": [1, 3, 5],
            "split": ["train", "val", "test"],
        }
    )


def test_valid_manifest_passes():
    df = _valid_df()
    assert validate_manifest(df) is df


def test_missing_score_column_raises():
    df = _valid_df().drop(columns=["personal_score"])
    with pytest.raises(ValueError, match="personal_score"):
        validate_manifest(df)


def test_missing_embedding_column_raises():
    df = _valid_df().drop(columns=["embedding"])
    with pytest.raises(ValueError, match="embedding"):
        validate_manifest(df)


def test_score_out_of_range_raises():
    df = _valid_df()
    df.loc[0, "personal_score"] = 6
    with pytest.raises(ValueError, match="personal_score"):
        validate_manifest(df)


def test_non_integer_score_raises():
    df = _valid_df()
    df["personal_score"] = df["personal_score"].astype(float)
    df.loc[0, "personal_score"] = 3.5
    with pytest.raises(ValueError, match="integer"):
        validate_manifest(df)


def test_unknown_split_label_raises():
    df = _valid_df()
    df.loc[0, "split"] = "holdout"
    with pytest.raises(ValueError, match="split"):
        validate_manifest(df)


def test_null_embedding_raises():
    df = _valid_df()
    df.at[0, "embedding"] = None
    with pytest.raises(ValueError, match="embedding"):
        validate_manifest(df)


def test_inconsistent_embedding_dim_raises():
    df = _valid_df()
    df.at[0, "embedding"] = [0.1, 0.2]  # wrong length
    with pytest.raises(ValueError, match="dimension"):
        validate_manifest(df)


def test_write_manifest_roundtrips(tmp_path):
    df = _valid_df()
    out = write_manifest(df, tmp_path / "m.parquet")
    assert out.exists()
    validate_manifest(pd.read_parquet(out))


def test_write_manifest_rejects_invalid_without_writing(tmp_path):
    df = _valid_df().drop(columns=["split"])
    target = tmp_path / "m.parquet"
    with pytest.raises(ValueError, match="split"):
        write_manifest(df, target)
    assert not target.exists()


def test_build_manifest_produces_valid_contract():
    embeddings = [[0.01 * i, 0.02 * i, 0.03 * i, 0.04 * i] for i in range(20)]
    scores = [(i % 5) + 1 for i in range(20)]
    post_ids = list(range(20))
    df = build_manifest(embeddings, scores, post_ids=post_ids, seed=0)
    assert len(df) == 20
    assert {"embedding", "personal_score", "split", "post_id"} <= set(df.columns)
    assert set(df["split"]) <= {"train", "val", "test"}
    validate_manifest(df)  # must satisfy the training contract


def test_build_manifest_post_id_is_optional():
    # post_id is provenance-only: a manifest without it is still valid and trainable
    embeddings = [[0.01 * i, 0.02 * i] for i in range(10)]
    scores = [(i % 5) + 1 for i in range(10)]
    df = build_manifest(embeddings, scores, seed=0)
    assert "post_id" not in df.columns
    validate_manifest(df)


def test_split_is_keyed_by_embedding_content_not_id():
    # identical embeddings must land in the same split regardless of (or absence of) post_id
    embeddings = [[0.0, 0.0], [0.0, 0.0], [1.0, 2.0], [1.0, 2.0]]
    scores = [3, 4, 5, 2]
    df = build_manifest(embeddings, scores, post_ids=[10, 11, 12, 13], seed=0)
    assert df["split"].iloc[0] == df["split"].iloc[1]  # same embedding -> same split
    assert df["split"].iloc[2] == df["split"].iloc[3]


def test_split_is_stable_when_id_changes():
    # relabel loop: same image (embedding) keeps its split even if its post_id differs
    emb = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(50)]
    a = build_manifest(emb, [3] * 50, post_ids=list(range(50)), seed=7)
    b = build_manifest(emb, [3] * 50, post_ids=list(range(100, 150)), seed=7)
    assert list(a["split"]) == list(b["split"])


def test_diff_manifests_reports_added_removed_rescored():
    e = {k: [float(i + 1), 0.0] for i, k in enumerate("abcd")}  # distinct embedding per logical image
    old = build_manifest([e["a"], e["b"], e["c"]], [3, 4, 5], post_ids=[1, 2, 3], seed=0)
    new = build_manifest([e["b"], e["c"], e["d"]], [4, 1, 2], post_ids=[2, 3, 4], seed=0)  # drop a/#1, add d/#4, rescore c/#3: 5->1
    d = diff_manifests(old, new)
    assert d["added_ids"] == [4]
    assert d["removed_ids"] == [1]
    assert d["rescored"] == [(3, 5, 1)]
    assert (d["n_added"], d["n_removed"], d["n_rescored"]) == (1, 1, 1)


def test_merge_manifests_concatenates_valid_frames():
    a = build_manifest([[float(i), 0.0] for i in range(6)], [(i % 5) + 1 for i in range(6)], post_ids=list(range(6)), seed=0)
    b = build_manifest([[0.0, float(i)] for i in range(4)], [(i % 5) + 1 for i in range(4)], post_ids=list(range(100, 104)), seed=0)
    merged = merge_manifests([a, b])
    assert len(merged) == 10
    validate_manifest(merged)


def test_merge_manifests_inconsistent_dim_raises():
    a = build_manifest([[1.0, 2.0]], [3], seed=0)
    b = build_manifest([[1.0, 2.0, 3.0]], [4], seed=0)
    with pytest.raises(ValueError, match="dimension"):
        merge_manifests([a, b])


def test_merge_manifests_drops_post_id_unless_all_present():
    a = build_manifest([[1.0, 2.0]], [3], post_ids=[1], seed=0)
    b = build_manifest([[3.0, 4.0]], [4], seed=0)  # no post_id
    merged = merge_manifests([a, b])
    assert "post_id" not in merged.columns  # mixed presence -> drop to avoid NaN ids
    validate_manifest(merged)


def test_merge_manifests_keeps_post_id_when_all_present():
    a = build_manifest([[1.0, 2.0]], [3], post_ids=[1], seed=0)
    b = build_manifest([[3.0, 4.0]], [4], post_ids=[2], seed=0)
    merged = merge_manifests([a, b])
    assert set(merged["post_id"]) == {1, 2}
