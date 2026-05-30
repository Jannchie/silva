import pandas as pd
import pytest

from silva_train.data.manifest import build_manifest, validate_manifest, write_manifest


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
    post_ids = list(range(20))
    embeddings = [[0.01 * i, 0.02 * i, 0.03 * i, 0.04 * i] for i in range(20)]
    scores = [(i % 5) + 1 for i in range(20)]
    df = build_manifest(post_ids, embeddings, scores, seed=0)
    assert len(df) == 20
    assert {"embedding", "personal_score", "split", "post_id"} <= set(df.columns)
    assert set(df["split"]) <= {"train", "val", "test"}
    validate_manifest(df)  # must satisfy the training contract


def test_build_manifest_dedups_split_by_post_id():
    # same post_id must never straddle splits
    post_ids = [1, 1, 2, 2, 3, 3]
    embeddings = [[0.0, 0.0]] * 6
    scores = [3] * 6
    df = build_manifest(post_ids, embeddings, scores, seed=0)
    by_id = df.groupby("post_id")["split"].nunique()
    assert (by_id == 1).all()
