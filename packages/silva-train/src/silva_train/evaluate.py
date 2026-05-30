"""Evaluate a trained checkpoint on a manifest split (personal validation metrics)."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from silva.models.aesthetic import EmbeddingAestheticModel
from silva_train.config import Config
from silva_train.data.dataset import AestheticDataset
from silva_train.metrics import compute_metrics


@torch.no_grad()
def evaluate(
    checkpoint: str,
    manifest_path: str,
    split: str,
    embedding_dim: int,
    dropout: float = 0.1,
    hidden_dims: list[int] | None = None,
    batch_size: int = 256,
    num_workers: int = 4,
) -> dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    model = EmbeddingAestheticModel(embedding_dim=embedding_dim, dropout=dropout, hidden_dims=hidden_dims)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    loader = DataLoader(AestheticDataset(manifest_path, split), batch_size=batch_size, num_workers=num_workers)

    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        out = model(batch["embedding"].to(device))
        preds.append(out["ordinal_score"].float().cpu())
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
