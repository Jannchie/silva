"""v1 Stage-1 training: accelerate loop, frozen SigLIP2 backbone + ordinal head."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, get_cosine_schedule_with_warmup

from silva.config import Config
from silva.data.dataset import AestheticDataset
from silva.losses import silva_loss
from silva.metrics import compute_metrics, is_improvement
from silva.models.siglip_aesthetic import SigLIP2AestheticModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("silva.train")


@torch.no_grad()
def run_eval(model: torch.nn.Module, loader: DataLoader, accelerator: Accelerator) -> dict[str, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        out = model(batch["pixel_values"])
        preds.append(accelerator.gather_for_metrics(out["ordinal_score"]).float().cpu())
        targets.append(accelerator.gather_for_metrics(batch["score"]).float().cpu())
    return compute_metrics(torch.cat(preds), torch.cat(targets))


def _set_train_mode(model: torch.nn.Module, *, freeze_backbone: bool) -> None:
    model.train()
    if freeze_backbone:
        # keep the frozen backbone in eval mode (no dropout / running-stat updates)
        model.vision.eval()


def train(config_path: str) -> dict[str, float]:
    cfg = Config.from_yaml(config_path)
    set_seed(cfg.train.seed)
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=cfg.train.grad_accum)

    processor = AutoImageProcessor.from_pretrained(cfg.model.model_id)
    train_ds = AestheticDataset(cfg.data.manifest_path, "train", processor)
    val_ds = AestheticDataset(cfg.data.manifest_path, "val", processor)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.data.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    model = SigLIP2AestheticModel(cfg.model.model_id, dropout=cfg.model.dropout, freeze_backbone=cfg.train.freeze_backbone)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.train.lr_head, weight_decay=cfg.train.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / cfg.train.grad_accum)
    total_steps = steps_per_epoch * cfg.train.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, round(total_steps * cfg.train.warmup_ratio), total_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    unwrapped = accelerator.unwrap_model(model)

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -math.inf
    best_metrics: dict[str, float] = {}
    patience = 0

    for epoch in range(cfg.train.epochs):
        _set_train_mode(unwrapped, freeze_backbone=cfg.train.freeze_backbone)
        for batch in train_loader:
            with accelerator.accumulate(model):
                out = model(batch["pixel_values"])
                loss = silva_loss(out["logits"], batch["score"], cfg.train.smooth_l1_weight)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, cfg.train.max_grad_norm)
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
