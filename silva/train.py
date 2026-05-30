"""v1 Stage-1 training: accelerate loop training the ordinal head on precomputed embeddings."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from silva.config import Config
from silva.data.dataset import AestheticDataset
from silva.losses import compute_pos_weight, silva_loss
from silva.metrics import compute_metrics, is_improvement
from silva.models.aesthetic import EmbeddingAestheticModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("silva.train")


def cosine_warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Linear warmup then cosine decay to zero — torch-native, no transformers dependency."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def run_eval(model: torch.nn.Module, loader: DataLoader, accelerator: Accelerator) -> dict[str, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        out = model(batch["embedding"])
        preds.append(accelerator.gather_for_metrics(out["ordinal_score"]).float().cpu())
        targets.append(accelerator.gather_for_metrics(batch["score"]).float().cpu())
    return compute_metrics(torch.cat(preds), torch.cat(targets))


def train(config_path: str) -> dict[str, float]:
    cfg = Config.from_yaml(config_path)
    set_seed(cfg.train.seed)
    accelerator = Accelerator(mixed_precision=cfg.train.mixed_precision, gradient_accumulation_steps=cfg.train.grad_accum)

    train_ds = AestheticDataset(cfg.data.manifest_path, "train")
    val_ds = AestheticDataset(cfg.data.manifest_path, "val")
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.data.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    model = EmbeddingAestheticModel(embedding_dim=cfg.model.embedding_dim, dropout=cfg.model.dropout)
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr_head, weight_decay=cfg.train.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / cfg.train.grad_accum)
    total_steps = steps_per_epoch * cfg.train.epochs
    scheduler = cosine_warmup_scheduler(optimizer, round(total_steps * cfg.train.warmup_ratio), total_steps)

    pos_weight = None
    if cfg.train.use_pos_weight:
        train_scores = torch.tensor(train_ds.rows["personal_score"].to_numpy())
        pos_weight = compute_pos_weight(train_scores).to(accelerator.device)
        accelerator.print(f"pos_weight (per threshold >1..>4): {pos_weight.tolist()}")

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -math.inf
    best_metrics: dict[str, float] = {}
    patience = 0

    for epoch in range(cfg.train.epochs):
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                out = model(batch["embedding"])
                loss = silva_loss(out["logits"], batch["score"], pos_weight=pos_weight)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad()

        if (epoch + 1) % cfg.train.eval_every != 0:
            continue

        metrics = run_eval(model, val_loader, accelerator)
        accelerator.print(f"[epoch {epoch + 1}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        current = metrics[cfg.train.early_stop_metric]
        if is_improvement(current, best_metric):
            best_metric = current
            best_metrics = metrics
            patience = 0
            if accelerator.is_main_process:
                torch.save(
                    {"model": accelerator.unwrap_model(model).state_dict(), "config": cfg.model_dump(), "metrics": metrics},
                    output_dir / "best.pt",
                )
                accelerator.print(f"  saved best.pt ({cfg.train.early_stop_metric}={current:.4f})")
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                accelerator.print(f"early stopping at epoch {epoch + 1}")
                break

    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SILVA v1 (Stage 1)")
    parser.add_argument("--config", default="configs/v1_stage1_head.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
