"""v1 Stage-1 training: accelerate loop training the ordinal head on precomputed embeddings."""

from __future__ import annotations

import argparse
import logging
import math
from contextlib import nullcontext
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import DataLoader, Subset

from silva.models.aesthetic import EmbeddingAestheticModel
from silva.scoring import ordinal_score_from_logits
from silva_train.anchors import anchors_from_neighbours
from silva_train.checkpoint import save_checkpoint
from silva_train.config import Config
from silva_train.data.dataset import AestheticDataset
from silva_train.ema import EmaShadow
from silva_train.losses import compute_pos_weight, silva_loss
from silva_train.metrics import bootstrap_ci, compute_metrics, is_improvement
from silva_train.tracking import build_log_with

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
def run_eval(
    model: torch.nn.Module, loader: DataLoader, accelerator: Accelerator
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        out = model(batch["embedding"])
        preds.append(accelerator.gather_for_metrics(ordinal_score_from_logits(out["logits"])).float().cpu())
        targets.append(accelerator.gather_for_metrics(batch["score"]).float().cpu())
    p, t = torch.cat(preds), torch.cat(targets)
    return compute_metrics(p, t), p, t


def _pin_best_summary(
    accelerator: Accelerator, preds: torch.Tensor, targets: torch.Tensor, epoch: int
) -> None:
    """Pin the chosen checkpoint's self-consistent metric row + bootstrap CIs as the
    run's terminal verdict (pandm summary). Unlike the per-key stats max/min on the
    curves — which mix values from different epochs — this is the one model actually
    saved. Flat keys (best/spearman, best/spearman_lo/_hi) match pandm's scalar store;
    non-finite CIs (tiny val sets) are dropped. Best-effort: never breaks training."""
    try:
        run = accelerator.get_tracker("pandm", unwrap=True)
    except Exception:  # no pandm tracker (report_to != pandm) -> nothing to pin
        return
    if not hasattr(run, "summary"):
        return
    finite = {
        key: v
        for k, m in bootstrap_ci(preds, targets).items()
        for key, v in ((f"best/{k}", m.value), (f"best/{k}_lo", m.lo), (f"best/{k}_hi", m.hi))
        if math.isfinite(v)
    }
    run.summary({"best/epoch": float(epoch), **finite})


def train(config_path: str) -> dict[str, float]:
    cfg = Config.from_yaml(config_path)
    set_seed(cfg.train.seed)
    log_with = build_log_with(cfg.train.report_to, cfg.train.project_name, cfg.train.run_name)
    accelerator = Accelerator(
        mixed_precision=cfg.train.mixed_precision,
        gradient_accumulation_steps=cfg.train.grad_accum,
        log_with=log_with,
    )

    train_ds = AestheticDataset(cfg.data.manifest_path, "train")
    val_ds = AestheticDataset(cfg.data.manifest_path, "val")
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.data.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    # A fixed train-split subset, evaluated like val each round, so the train-val gap
    # (train_spearman - val_spearman etc.) is a live curve — the generalization signal
    # the sweeps optimise. Subset is seeded so the gap is comparable across epochs/runs.
    n_train_eval = min(cfg.train.train_eval_samples or len(val_ds), len(train_ds))
    train_eval_idx = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(cfg.train.seed))[:n_train_eval].tolist()
    train_eval_loader = DataLoader(Subset(train_ds, train_eval_idx), batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    model = EmbeddingAestheticModel(
        embedding_dim=cfg.model.embedding_dim, dropout=cfg.model.dropout, hidden_dims=cfg.model.hidden_dims,
        n_residual_blocks=cfg.model.n_residual_blocks,
    )
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr_head, weight_decay=cfg.train.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / cfg.train.grad_accum)
    total_steps = steps_per_epoch * cfg.train.epochs
    if cfg.train.cosine_restarts > 0:
        t_0 = total_steps // (cfg.train.cosine_restarts + 1)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, t_0), T_mult=1)
    else:
        scheduler = cosine_warmup_scheduler(optimizer, round(total_steps * cfg.train.warmup_ratio), total_steps)

    pos_weight = None
    if cfg.train.use_pos_weight:
        pos_weight = compute_pos_weight(train_ds.scores).to(accelerator.device)
        accelerator.print(f"pos_weight (per threshold >1..>4): {pos_weight.tolist()}")

    # Non-uniform category anchors for QWK: "auto" re-estimates the rater's perceptual
    # spacing from the train split each run (kNN pseudo-retest), so the spacing tracks
    # the labels as relabel rounds move them.
    anchors = None
    if cfg.train.score_anchors == "auto":
        emb_t = train_ds.embeddings.to(accelerator.device)
        sc_t = train_ds.scores.float().to(accelerator.device)
        anchors = torch.tensor(anchors_from_neighbours(emb_t, sc_t))
        del emb_t, sc_t
        accelerator.print(f"score_anchors (auto, kNN pseudo-retest): {[round(float(a), 3) for a in anchors]}")
    elif cfg.train.score_anchors is not None:
        anchors = torch.tensor([float(a) for a in cfg.train.score_anchors])

    model, optimizer, train_loader, val_loader, train_eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, train_eval_loader, scheduler
    )

    # EMA: shadow weights, initialized AFTER prepare() so tensors are on the right device
    ema = EmaShadow(accelerator.unwrap_model(model), cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None

    if log_with:
        flat_config = {f"{section}.{k}": v for section, values in cfg.model_dump().items() for k, v in values.items()}
        if pos_weight is not None:
            # derived per-threshold balancing weights are a run-start constant, not a measurement:
            # record them as config (a static snapshot) instead of a single-point metric stranded on the curves
            flat_config.update({f"pos_weight/>{i + 1}": w for i, w in enumerate(pos_weight.tolist())})
        accelerator.init_trackers(cfg.train.project_name, config=flat_config)

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -math.inf
    best_metrics: dict[str, float] = {}
    best_epoch = 0
    best_val: tuple[torch.Tensor, torch.Tensor] | None = None  # (preds, targets) of the chosen checkpoint, for the terminal summary
    patience = 0

    global_step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                emb = batch["embedding"]
                score = batch["score"].float()
                if cfg.train.mixup_alpha > 0:
                    lam = torch.distributions.Beta(cfg.train.mixup_alpha, cfg.train.mixup_alpha).sample().to(emb.device)
                    perm = torch.randperm(emb.size(0), device=emb.device)
                    emb = lam * emb + (1 - lam) * emb[perm]
                    score = lam * score + (1 - lam) * score[perm]
                out = model(emb)
                loss = silva_loss(
                    out["logits"],
                    score,
                    pos_weight=pos_weight,
                    ranking_weight=cfg.train.ranking_weight,
                    soft_spearman_weight=cfg.train.soft_spearman_weight,
                    qwk_weight=cfg.train.qwk_weight,
                    label_smoothing=cfg.train.label_smoothing,
                    loss_truncation=cfg.train.loss_truncation,
                    anchors=anchors,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                    global_step += 1
                    if ema is not None:
                        ema.update(accelerator.unwrap_model(model))
                    if log_with:
                        accelerator.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]}, step=global_step)
                optimizer.zero_grad()

        if (epoch + 1) % cfg.train.eval_every != 0:
            continue

        # Evaluate (and checkpoint) on EMA weights when enabled; live weights restored on block exit.
        raw_model = accelerator.unwrap_model(model)
        with ema.swapped(raw_model) if ema is not None else nullcontext():
            metrics, val_preds, val_targets = run_eval(model, val_loader, accelerator)
            train_metrics, _, _ = run_eval(model, train_eval_loader, accelerator)
            gap = {k: train_metrics[k] - metrics[k] for k in metrics}
            accelerator.print(
                f"[epoch {epoch + 1}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                + f"  | gap spearman={gap['spearman']:.4f} qwk={gap['qwk']:.4f}"
            )
            if log_with:
                accelerator.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)
                accelerator.log({f"train/{k}": v for k, v in train_metrics.items()}, step=global_step)
                accelerator.log({f"gap/{k}": v for k, v in gap.items()}, step=global_step)

            current = metrics[cfg.train.early_stop_metric]
            if is_improvement(current, best_metric):
                best_metric = current
                best_metrics = metrics
                best_epoch = epoch + 1
                best_val = (val_preds, val_targets)
                patience = 0
                if accelerator.is_main_process:
                    save_checkpoint(output_dir, raw_model.state_dict(), cfg.model_dump(), metrics)
                    accelerator.print(f"  saved best.safetensors ({cfg.train.early_stop_metric}={current:.4f})")
            else:
                patience += 1

        if patience >= cfg.train.early_stop_patience:
            accelerator.print(f"early stopping at epoch {epoch + 1}")
            break

    if log_with:
        if best_val is not None and accelerator.is_main_process:
            _pin_best_summary(accelerator, best_val[0], best_val[1], best_epoch)
        accelerator.end_training()
    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SILVA v1 (Stage 1)")
    parser.add_argument("--config", default="configs/v1_stage1_head.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
