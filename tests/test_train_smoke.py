"""Opt-in end-to-end smoke test for the training closed loop.

Skipped by default because it loads the real SigLIP2 backbone. Enable with:

    SILVA_RUN_INTEGRATION=1 uv run pytest tests/test_train_smoke.py
"""

import os

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

pytestmark = pytest.mark.skipif(
    os.environ.get("SILVA_RUN_INTEGRATION") != "1",
    reason="integration smoke test; set SILVA_RUN_INTEGRATION=1 (requires SigLIP2 weights)",
)


def _make_image(path: str) -> None:
    arr = np.random.default_rng().integers(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_training_closed_loop(tmp_path):
    from silva.train import train

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    rows = []
    for i in range(12):
        p = img_dir / f"{i}.png"
        _make_image(str(p))
        rows.append({"image_path": str(p), "personal_score": (i % 5) + 1, "split": "train" if i < 8 else "val"})
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)

    out_dir = tmp_path / "out"
    cfg = {
        "model": {"model_id": "google/siglip2-so400m-patch14-384", "dropout": 0.1},
        "data": {"manifest_path": str(manifest), "num_workers": 0},
        "train": {
            "freeze_backbone": True,
            "batch_size": 2,
            "grad_accum": 1,
            "epochs": 1,
            "lr_head": 1e-3,
            "weight_decay": 0.01,
            "warmup_ratio": 0.0,
            "max_grad_norm": 1.0,
            "smooth_l1_weight": 0.2,
            "eval_every": 1,
            "early_stop_metric": "spearman",
            "early_stop_patience": 1,
            "seed": 0,
            "output_dir": str(out_dir),
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    metrics = train(str(cfg_path))

    assert (out_dir / "best.pt").exists()
    assert "spearman" in metrics
