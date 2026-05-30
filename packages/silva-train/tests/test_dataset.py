import pandas as pd
import torch

from silva_train.data.dataset import AestheticDataset


def _manifest(tmp_path, rows) -> str:
    p = tmp_path / "m.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return str(p)


def test_returns_embedding_and_score(tmp_path):
    m = _manifest(
        tmp_path,
        [{"embedding": [0.1, 0.2, 0.3, 0.4], "personal_score": 4, "split": "train"}],
    )
    sample = AestheticDataset(m, "train")[0]
    assert sample["score"] == 4
    assert sample["embedding"].shape == (4,)
    assert sample["embedding"].dtype == torch.float32


def test_filters_by_split(tmp_path):
    m = _manifest(
        tmp_path,
        [
            {"embedding": [0.1, 0.2, 0.3, 0.4], "personal_score": 2, "split": "train"},
            {"embedding": [0.5, 0.6, 0.7, 0.8], "personal_score": 5, "split": "val"},
        ],
    )
    train = AestheticDataset(m, "train")
    assert len(train) == 1
    assert train[0]["score"] == 2


def test_len_counts_only_split_rows(tmp_path):
    m = _manifest(
        tmp_path,
        [
            {"embedding": [0.0, 0.0, 0.0, 0.0], "personal_score": 1, "split": "train"},
            {"embedding": [0.0, 0.0, 0.0, 0.0], "personal_score": 3, "split": "train"},
            {"embedding": [0.0, 0.0, 0.0, 0.0], "personal_score": 5, "split": "test"},
        ],
    )
    assert len(AestheticDataset(m, "train")) == 2
