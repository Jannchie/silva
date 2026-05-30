"""Throwaway experiment harness: sweep head architectures x loss functions.

NOT production code (TDD-exempt prototype). Preloads every split's embeddings to
the GPU once, then trains the head-only model many times to rank ideas cheaply.
The winning recipe is re-confirmed via the real ``silva.train`` path and baked
into ``configs/`` with proper tests before it counts.

Usage:  uv run python scripts/sweep.py [--seeds 42] [--epochs 24]
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from silva_train.losses import (
    compute_pos_weight,
    listwise_loss,
    ordinal_loss,
    pairwise_ranking_loss,
    soft_spearman_loss,
)
from silva_train.metrics import compute_metrics
from silva.models.aesthetic import EmbeddingAestheticModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MANIFEST = "data/manifest.parquet"


def load_split(df: pd.DataFrame, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    rows = df[df["split"] == split]
    x = torch.tensor(np.stack(rows["embedding"].to_numpy()), dtype=torch.float32, device=DEVICE)
    y = torch.tensor(rows["personal_score"].to_numpy(), dtype=torch.long, device=DEVICE)
    return x, y


def lr_factor(step: int, warmup: int, total: int, kind: str = "cosine") -> float:
    """LR multiplier in [0, 1] for several schedules (warmup is linear for all)."""
    if kind == "onecycle":  # long warmup to peak (30%) then cosine down to ~0
        warmup = round(0.3 * total)
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    if kind == "constant":
        return 1.0
    decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    if kind == "cosine_floor":  # decay to 10% of peak instead of 0
        return 0.1 + 0.9 * decay
    return decay  # "cosine" and "onecycle"


def make_loss(ranking: float = 0.0, soft_sp: float = 0.0, listwise: float = 0.0):
    def loss_fn(logits, scores, pos_weight):
        loss = ordinal_loss(logits, scores, pos_weight=pos_weight)
        if ranking:
            loss = loss + ranking * pairwise_ranking_loss(logits, scores)
        if soft_sp:
            loss = loss + soft_sp * soft_spearman_loss(logits, scores)
        if listwise:
            loss = loss + listwise * listwise_loss(logits, scores)
        return loss

    return loss_fn


def train_once(data, *, hidden_dims, loss_fn, dropout, seed, epochs, lr, batch_size, sched="cosine", weight_decay=0.01, noise_std=0.0):  # noqa: PLR0913
    x_tr, y_tr, x_val, y_val = data["train"][0], data["train"][1], data["val"][0], data["val"][1]
    torch.manual_seed(seed)
    model = EmbeddingAestheticModel(embedding_dim=x_tr.shape[1], dropout=dropout, hidden_dims=hidden_dims).to(DEVICE)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    feat_std = x_tr.std()  # scale Gaussian input noise relative to feature spread
    pos_weight = compute_pos_weight(y_tr).to(DEVICE)

    n = x_tr.shape[0]
    steps_per_epoch = n // batch_size
    total = steps_per_epoch * epochs
    warmup = round(total * 0.03)
    step = 0
    best_val = -math.inf
    best_state = None

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for b in range(steps_per_epoch):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            xb = x_tr[idx]
            if noise_std > 0:  # embedding-space augmentation (no images to augment)
                xb = xb + noise_std * feat_std * torch.randn_like(xb)
            out = model(xb)
            loss = loss_fn(out["logits"], y_tr[idx], pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for g in opt.param_groups:
                g["lr"] = lr * lr_factor(step, warmup, total, sched)
            opt.step()
            step += 1

        model.eval()
        with torch.no_grad():
            sp = compute_metrics(model(x_val)["ordinal_score"], y_val)["spearman"]
        if not math.isnan(sp) and sp > best_val:
            best_val = sp
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(data["test"][0])["ordinal_score"]
        train_sp = compute_metrics(model(x_tr)["ordinal_score"], y_tr)["spearman"]
    test_metrics = compute_metrics(test_pred, data["test"][1])
    return best_val, test_metrics, test_pred.float().cpu(), train_sp


# Round 6: fix the winning arch+loss+lr; attack the train(0.95)->val(0.72) overfit gap
# with stronger regularisation + embedding-space augmentation (no images to augment).
_ARCH = dict(hidden_dims=[1024, 512, 256], loss_fn=make_loss(ranking=1.0, soft_sp=0.5), lr=1.5e-3, sched="cosine", epochs=40)
EXPERIMENTS = [
    ("baseline d0.1 wd0.01", dict(**_ARCH, dropout=0.1, weight_decay=0.01)),
    ("dropout0.3", dict(**_ARCH, dropout=0.3, weight_decay=0.01)),
    ("dropout0.5", dict(**_ARCH, dropout=0.5, weight_decay=0.01)),
    ("wd0.05", dict(**_ARCH, dropout=0.1, weight_decay=0.05)),
    ("wd0.1", dict(**_ARCH, dropout=0.1, weight_decay=0.1)),
    ("dropout0.3 wd0.05", dict(**_ARCH, dropout=0.3, weight_decay=0.05)),
    ("noise0.1", dict(**_ARCH, dropout=0.1, weight_decay=0.01, noise_std=0.1)),
    ("noise0.3", dict(**_ARCH, dropout=0.1, weight_decay=0.01, noise_std=0.3)),
    ("dropout0.3 noise0.2", dict(**_ARCH, dropout=0.3, weight_decay=0.01, noise_std=0.2)),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--only", type=str, default=None, help="substring filter on experiment name")
    args = parser.parse_args()

    df = pd.read_parquet(MANIFEST, columns=["split", "personal_score", "embedding"])
    data = {s: load_split(df, s) for s in ("train", "val", "test")}
    print(f"device={DEVICE} train={data['train'][0].shape} seeds={args.seeds} epochs={args.epochs}\n")

    print(f"{'experiment':<34} {'train':>6} {'val_sp':>7} {'test_sp':>8} {'gap':>6} {'top5%':>6} {'mae':>6}")
    print("-" * 84)
    for name, spec in EXPERIMENTS:
        if args.only and args.only not in name:
            continue
        spec = dict(spec)
        spec.setdefault("dropout", 0.1)
        vals, tests, preds, trains = [], [], [], []
        for seed in args.seeds:
            v, t, p, tr = train_once(
                data, hidden_dims=spec["hidden_dims"], loss_fn=spec["loss_fn"], dropout=spec["dropout"],
                seed=seed, epochs=spec.get("epochs", args.epochs), lr=spec.get("lr", args.lr),
                batch_size=args.batch_size, sched=spec.get("sched", "cosine"),
                weight_decay=spec.get("weight_decay", 0.01), noise_std=spec.get("noise_std", 0.0),
            )
            vals.append(v)
            tests.append(t)
            preds.append(p)
            trains.append(tr)
        trn = float(np.mean(trains))
        vm = float(np.mean(vals))
        tsp = float(np.mean([t["spearman"] for t in tests]))
        t5 = float(np.mean([t["top_5pct"] for t in tests]))
        mae = float(np.mean([t["mae"] for t in tests]))
        std = f"+-{np.std([t['spearman'] for t in tests]):.3f}" if len(args.seeds) > 1 else ""
        print(f"{name:<34} {trn:>6.3f} {vm:>7.4f} {tsp:>8.4f}{std:<6} {trn - vm:>6.3f} {t5:>6.3f} {mae:>6.3f}")
        if len(args.seeds) > 1:
            ens = compute_metrics(torch.stack(preds).mean(dim=0), data["test"][1].cpu())
            tag = f"  └ ensemble x{len(args.seeds)}"
            print(f"{tag:<34} {'':>7} {ens['spearman']:>8.4f}{'':<6} {ens['pearson']:>9.4f} {ens['top_5pct']:>6.3f} {ens['mae']:>6.3f}")


if __name__ == "__main__":
    main()
