"""Evaluate a trained checkpoint on a manifest split (personal validation metrics)."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from silva.models.aesthetic import EmbeddingAestheticModel
from silva.scoring import ordinal_score_from_logits
from silva_train.checkpoint import load_checkpoint
from silva_train.config import Config
from silva_train.data.dataset import AestheticDataset
from silva_train.metrics import compute_metrics


@torch.no_grad()
def evaluate(
    checkpoint: str,
    manifest_path: str | list[str],
    split: str,
    embedding_dim: int,
    dropout: float = 0.1,
    hidden_dims: list[int] | None = None,
    batch_size: int = 256,
    num_workers: int = 4,
) -> dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state, _config, _metrics = load_checkpoint(checkpoint)

    model = EmbeddingAestheticModel(embedding_dim=embedding_dim, dropout=dropout, hidden_dims=hidden_dims)
    model.load_state_dict(state, strict=False)  # calibration buffers are optional (filled at publish time), absent in training checkpoints
    model.to(device).eval()

    loader = DataLoader(AestheticDataset(manifest_path, split), batch_size=batch_size, num_workers=num_workers)

    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        out = model(batch["embedding"].to(device))
        preds.append(ordinal_score_from_logits(out["logits"]).float().cpu())
        targets.append(batch["score"].float())
    return compute_metrics(torch.cat(preds), torch.cat(targets))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SILVA checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/v1_stage1_head.yaml")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    metrics = evaluate(
        args.checkpoint,
        cfg.data.manifest_path,
        args.split,
        cfg.model.embedding_dim,
        cfg.model.dropout,
        cfg.model.hidden_dims,
        cfg.train.batch_size,
        cfg.data.num_workers,
    )
    print(f"[{args.split}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
