import pandas as pd
import pytest
import torch
from PIL import Image

from silva.data.dataset import AestheticDataset


def _fake_processor(images, return_tensors):  # noqa: ARG001
    return {"pixel_values": [torch.zeros(3, 4, 4)]}


def _valid_png(path) -> None:
    Image.new("RGB", (8, 8), (123, 222, 64)).save(path)


def _corrupt(path) -> None:
    path.write_bytes(b"not an image")


def _manifest(tmp_path, rows) -> str:
    p = tmp_path / "m.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return str(p)


def test_returns_sample_for_readable_image(tmp_path):
    img = tmp_path / "a.png"
    _valid_png(img)
    m = _manifest(tmp_path, [{"image_path": str(img), "personal_score": 4, "split": "train"}])
    sample = AestheticDataset(m, "train", _fake_processor)[0]
    assert sample["score"] == 4
    assert sample["pixel_values"].shape == (3, 4, 4)


def test_skips_corrupt_and_returns_next(tmp_path):
    bad = tmp_path / "bad.png"
    _corrupt(bad)
    good = tmp_path / "good.png"
    _valid_png(good)
    m = _manifest(
        tmp_path,
        [
            {"image_path": str(bad), "personal_score": 1, "split": "train"},
            {"image_path": str(good), "personal_score": 5, "split": "train"},
        ],
    )
    assert AestheticDataset(m, "train", _fake_processor)[0]["score"] == 5


def test_raises_when_all_unreadable(tmp_path):
    b1 = tmp_path / "b1.png"
    _corrupt(b1)
    b2 = tmp_path / "b2.png"
    _corrupt(b2)
    m = _manifest(
        tmp_path,
        [
            {"image_path": str(b1), "personal_score": 2, "split": "train"},
            {"image_path": str(b2), "personal_score": 3, "split": "train"},
        ],
    )
    ds = AestheticDataset(m, "train", _fake_processor)
    # Must be a clean, bounded error (not RecursionError from infinite self-calls).
    with pytest.raises(RuntimeError, match="readable"):
        _ = ds[0]
