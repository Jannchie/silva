"""End-to-end smoke test of the training closed loop on toy embeddings.

Runs by default: since the training library consumes precomputed embeddings (no
SigLIP backbone, no weight download), the whole train loop is exercisable on CPU.
"""

import numpy as np
import pandas as pd
import yaml

from silva_train.data.manifest import write_manifest


def test_training_closed_loop(tmp_path):
    from silva_train.train import train

    rng = np.random.default_rng(0)
    rows = []
    for i in range(24):
        score = (i % 5) + 1
        emb = rng.standard_normal(16)
        emb[0] += score * 3.0  # put a learnable signal in one dim (survives LayerNorm)
        rows.append(
            {
                "post_id": i,
                "embedding": emb.tolist(),
                "personal_score": score,
                "split": "train" if i < 18 else "val",
            }
        )
    manifest = tmp_path / "manifest.parquet"
    write_manifest(pd.DataFrame(rows), manifest)

    out_dir = tmp_path / "out"
    cfg = {
        "model": {"embedding_dim": 16, "dropout": 0.0},
        "data": {"manifest_path": str(manifest), "num_workers": 0},
        "train": {
            "batch_size": 4,
            "grad_accum": 1,
            "epochs": 3,
            "lr_head": 1e-2,
            "weight_decay": 0.01,
            "warmup_ratio": 0.0,
            "max_grad_norm": 1.0,
            "use_pos_weight": True,
            "mixed_precision": "no",
            "eval_every": 1,
            "early_stop_metric": "spearman",
            "early_stop_patience": 3,
            "seed": 0,
            "output_dir": str(out_dir),
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    metrics = train(str(cfg_path))

    assert (out_dir / "best.safetensors").exists()
    assert (out_dir / "best.json").exists()
    assert "spearman" in metrics
