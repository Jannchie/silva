import pandas as pd
import torch

from silva.models.aesthetic import EmbeddingAestheticModel
from silva_train.checkpoint import save_checkpoint
from silva_train.evaluate import evaluate


def _manifest(tmp_path, dim, scores_splits) -> str:
    rows = [{"embedding": torch.randn(dim).tolist(), "personal_score": s, "split": sp} for s, sp in scores_splits]
    p = tmp_path / "m.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return str(p)


def test_evaluate_takes_slim_data_signature(tmp_path):
    # evaluate rebuilds the model from the checkpoint itself (load_model); the caller
    # passes only data args, never the architecture params again.
    model = EmbeddingAestheticModel(embedding_dim=4, hidden_dims=[8]).eval()
    save_checkpoint(tmp_path, model.state_dict(), {"model": {"embedding_dim": 4, "hidden_dims": [8]}}, {})
    manifest = _manifest(tmp_path, 4, [(4, "val"), (2, "val"), (5, "val"), (1, "val")])

    metrics = evaluate(tmp_path, manifest, "val", num_workers=0)

    assert "spearman" in metrics
    assert "mae" in metrics
