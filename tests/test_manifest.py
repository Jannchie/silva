import pandas as pd
import pytest

from silva.data.manifest import validate_manifest, write_manifest


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_path": ["a.png", "b.png", "c.png"],
            "personal_score": [1, 3, 5],
            "split": ["train", "val", "test"],
        }
    )


def test_valid_manifest_passes():
    df = _valid_df()
    assert validate_manifest(df) is df


def test_missing_required_column_raises():
    df = _valid_df().drop(columns=["personal_score"])
    with pytest.raises(ValueError, match="personal_score"):
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


def test_null_image_path_raises():
    df = _valid_df()
    df.loc[0, "image_path"] = None
    with pytest.raises(ValueError, match="image_path"):
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
