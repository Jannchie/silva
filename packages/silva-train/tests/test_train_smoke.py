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
            "report_to": "none",  # this loop tests training, not tracking (default is pandm)
            "output_dir": str(out_dir),
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    metrics = train(str(cfg_path))

    assert (out_dir / "best.safetensors").exists()
    assert (out_dir / "best.json").exists()
    assert "spearman" in metrics


def test_training_closed_loop_with_anchored_qwk(tmp_path):
    # exercises the anchors path: explicit list + qwk term, and the "auto" estimator
    from silva_train.train import train

    rng = np.random.default_rng(0)
    rows = []
    for i in range(24):
        score = (i % 5) + 1
        emb = rng.standard_normal(16)
        emb[0] += score * 3.0
        rows.append({"post_id": i, "embedding": emb.tolist(), "personal_score": score, "split": "train" if i < 18 else "val"})
    manifest = tmp_path / "manifest.parquet"
    write_manifest(pd.DataFrame(rows), manifest)

    base_train = {
        "batch_size": 4,
        "epochs": 2,
        "lr_head": 1e-2,
        "warmup_ratio": 0.0,
        "use_pos_weight": True,
        "mixed_precision": "no",
        "eval_every": 1,
        "early_stop_patience": 3,
        "seed": 0,
        "qwk_weight": 1.0,
        "report_to": "none",  # this loop tests anchors+QWK, not tracking (default is pandm)
    }
    for name, anchors in (("explicit", [1.0, 2.0, 3.0, 3.5, 4.5]), ("auto", "auto")):
        out_dir = tmp_path / f"out_{name}"
        cfg = {
            "model": {"embedding_dim": 16, "dropout": 0.0},
            "data": {"manifest_path": str(manifest), "num_workers": 0},
            "train": {**base_train, "score_anchors": anchors, "output_dir": str(out_dir)},
        }
        cfg_path = tmp_path / f"cfg_{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        # guard against pydantic silently dropping an unknown key (the config must OWN this field)
        from silva_train.config import Config

        assert Config.from_yaml(cfg_path).train.score_anchors == anchors, name

        metrics = train(str(cfg_path))

        assert (out_dir / "best.safetensors").exists(), name
        assert "spearman" in metrics, name


def test_training_closed_loop_reports_to_pandm(tmp_path, monkeypatch):
    # the accelerate -> PandmTracker seam: report_to "pandm" must flow real metrics into a pandm run
    import json
    import sqlite3

    from silva_train.train import train

    rng = np.random.default_rng(0)
    rows = []
    for i in range(24):
        score = (i % 5) + 1
        emb = rng.standard_normal(16)
        emb[0] += score * 3.0
        rows.append({"post_id": i, "embedding": emb.tolist(), "personal_score": score, "split": "train" if i < 18 else "val"})
    manifest = tmp_path / "manifest.parquet"
    write_manifest(pd.DataFrame(rows), manifest)

    monkeypatch.chdir(tmp_path)  # pandm writes its db under ./.pandm relative to cwd
    cfg = {
        "model": {"embedding_dim": 16, "dropout": 0.0},
        "data": {"manifest_path": str(manifest), "num_workers": 0},
        "train": {
            "batch_size": 4,
            "epochs": 2,
            "lr_head": 1e-2,
            "warmup_ratio": 0.0,
            "use_pos_weight": True,
            "mixed_precision": "no",
            "eval_every": 1,
            "early_stop_patience": 3,
            "seed": 0,
            "output_dir": str(tmp_path / "out"),
            "report_to": "pandm",
            "project_name": "silva-smoke",
            "run_name": "e2e",
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    train(str(cfg_path))

    db = tmp_path / ".pandm" / "pandm.db"
    assert db.exists()
    con = sqlite3.connect(db)
    try:
        keys = {k for (k,) in con.execute("SELECT DISTINCT key FROM metrics")}
        runs = list(con.execute("SELECT project, name, status FROM runs"))
        (config_json,) = con.execute("SELECT config FROM runs").fetchone()
        (summary_json,) = con.execute("SELECT summary FROM runs").fetchone()
    finally:
        con.close()
    config = json.loads(config_json)
    summary = json.loads(summary_json)
    assert {"train/loss", "train/lr"} <= keys  # per-step training metrics logged
    assert any(k.startswith("val/") for k in keys)  # eval metrics logged
    assert {"train/spearman", "gap/spearman", "val/biggap"} <= keys  # train-val gap + biggap monitoring
    # pos_weight is a run-start derived constant -> recorded as config, never a single-point metric
    assert not any(k.startswith("pos_weight/") for k in keys)
    assert {"pos_weight/>1", "pos_weight/>2", "pos_weight/>3", "pos_weight/>4"} <= set(config)
    # the chosen checkpoint's self-consistent row is pinned as the terminal summary (not on the curves)
    assert "best/epoch" in summary and "best/spearman" in summary
    assert not any(k.startswith("best/") for k in keys)
    assert ("silva-smoke", "e2e", "finished") in runs  # run named via build_log_with + finished cleanly


def test_training_closed_loop_with_ema(tmp_path):
    # exercises the EMA path: update each step, swap shadow weights in for eval + save
    from silva_train.train import train

    rng = np.random.default_rng(0)
    rows = []
    for i in range(24):
        score = (i % 5) + 1
        emb = rng.standard_normal(16)
        emb[0] += score * 3.0
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
            "epochs": 3,
            "lr_head": 1e-2,
            "warmup_ratio": 0.0,
            "use_pos_weight": True,
            "mixed_precision": "no",
            "eval_every": 1,
            "early_stop_patience": 3,
            "seed": 0,
            "ema_decay": 0.99,
            "report_to": "none",  # this loop tests EMA, not tracking (default is pandm)
            "output_dir": str(out_dir),
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    metrics = train(str(cfg_path))

    assert (out_dir / "best.safetensors").exists()
    assert "spearman" in metrics
