import pandas as pd
import pytest
import torch

from silva_train.data.pairs import PairDataset


def _pairs(tmp_path, rows) -> str:
    p = tmp_path / "pairs.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return str(p)


def test_loads_and_prestacks(tmp_path):
    m = _pairs(
        tmp_path,
        [
            {"embedding_a": [0.1, 0.2, 0.3], "embedding_b": [0.4, 0.5, 0.6], "target": 1},
            {"embedding_a": [0.7, 0.8, 0.9], "embedding_b": [0.0, 0.1, 0.2], "target": -1},
            {"embedding_a": [0.3, 0.3, 0.3], "embedding_b": [0.3, 0.3, 0.3], "target": 0},
        ],
    )
    ds = PairDataset(m)
    assert len(ds) == 3
    assert ds.emb_a.shape == (3, 3)
    assert ds.emb_b.shape == (3, 3)
    assert ds.emb_a.dtype == torch.float32
    assert ds.emb_b.dtype == torch.float32
    assert ds.targets.dtype == torch.long
    assert ds.targets.tolist() == [1, -1, 0]


def test_rejects_out_of_range_target(tmp_path):
    m = _pairs(
        tmp_path,
        [{"embedding_a": [0.1, 0.2], "embedding_b": [0.3, 0.4], "target": 2}],
    )
    with pytest.raises(ValueError, match="target"):
        PairDataset(m)


def test_sample_returns_expected_shapes(tmp_path):
    m = _pairs(
        tmp_path,
        [{"embedding_a": [float(i), float(i)], "embedding_b": [0.0, 0.0], "target": 1} for i in range(10)],
    )
    ds = PairDataset(m)
    gen = torch.Generator().manual_seed(0)
    a, b, t = ds.sample(4, gen)
    assert a.shape == (4, 2)
    assert b.shape == (4, 2)
    assert t.shape == (4,)
    assert t.dtype == torch.long


def test_sample_caps_at_dataset_size(tmp_path):
    m = _pairs(
        tmp_path,
        [{"embedding_a": [1.0], "embedding_b": [0.0], "target": 1} for _ in range(3)],
    )
    ds = PairDataset(m)
    gen = torch.Generator().manual_seed(0)
    a, _, t = ds.sample(10, gen)
    assert a.shape[0] == 3
    assert t.shape[0] == 3
