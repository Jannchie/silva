"""Out-of-fold predictions — the memorisation-free disagreement signal for label audits.

``misclass_probe`` reads the in-fold gap: the production head has the capacity to memorise
the train set, so a noisy label it *did* fit produces no gap at all. Out-of-fold flips
that: each row is predicted by a model trained on every fold but its own, so the
prediction is what the rest of the data implies the row *should* score. A large
``|OOF - label|`` therefore means the label conflicts with the learnable pattern — the
strongest single mislabel signal available without a second rater.

Two seams:

  - :func:`oof_predictions` — the fold loop itself, generic over ``fit_fn`` so tests can
    inject a fake trainer and assert no row is ever predicted by a model that saw it.
  - :func:`make_fit_head` — the default ``fit_fn``: the production stage-1 head + loss mix
    (defaults mirror ``configs/v1_stage1_head.yaml``) in a plain in-memory loop, no
    accelerate/wandb harness.

Fold assignment is the caller's job (``assign_folds`` over embedding content keys), so
folds stay stable across manifest re-exports just like splits do.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from silva.models.aesthetic import EmbeddingAestheticModel
from silva.scoring import ordinal_score_from_logits
from silva_train.losses import compute_pos_weight, silva_loss

if TYPE_CHECKING:
    from collections.abc import Sequence

PredictFn = Callable[[torch.Tensor], torch.Tensor]
FitFn = Callable[[torch.Tensor, torch.Tensor], PredictFn]


def oof_predictions(
    embeddings: torch.Tensor,
    scores: torch.Tensor,
    folds: Sequence[int],
    fit_fn: FitFn,
) -> torch.Tensor:
    """Predict every row with a model fitted on the other folds only.

    ``fit_fn(train_emb, train_scores) -> predict`` is called once per distinct fold with
    that fold held out; ``predict`` is applied to the held-out rows. Returns ``[N]``
    continuous scores on CPU. Raises if some fold holds every row (no complement to
    train on).
    """
    folds_t = torch.as_tensor(list(folds), dtype=torch.long)
    preds = torch.empty(len(folds_t), dtype=torch.float32)
    for fold in folds_t.unique().tolist():
        held = folds_t == fold
        train = ~held
        if not train.any():
            raise ValueError(f"fold {fold} contains every row; nothing left to train on")
        predict = fit_fn(embeddings[train], scores[train])
        preds[held] = predict(embeddings[held]).detach().float().cpu()
    return preds


def make_fit_head(
    *,
    hidden_dims: Sequence[int] = (1024, 512, 256),
    n_residual_blocks: int = 0,
    dropout: float = 0.3,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1.5e-3,
    weight_decay: float = 0.05,
    ranking_weight: float = 1.0,
    soft_spearman_weight: float = 0.5,
    qwk_weight: float = 1.0,
    label_smoothing: float = 0.2,
    loss_truncation: float = 0.0,
    use_pos_weight: bool = True,
    seed: int = 42,
    device: str | None = None,
) -> FitFn:
    """Build the default ``fit_fn``: stage-1 head + production loss mix, in-memory loop.

    Deterministic for a given seed (model init, batch order and dropout masks are all
    seeded per ``fit`` call). ``device`` defaults to the training embeddings' device.
    """

    def fit(train_emb: torch.Tensor, train_scores: torch.Tensor) -> PredictFn:
        dev = torch.device(device) if device else train_emb.device
        torch.manual_seed(seed)  # model init + dropout masks
        gen = torch.Generator().manual_seed(seed)  # batch order, kept off the global stream
        model = EmbeddingAestheticModel(
            embedding_dim=train_emb.shape[1],
            dropout=dropout,
            hidden_dims=list(hidden_dims),
            n_residual_blocks=n_residual_blocks,
        ).to(dev)
        emb = train_emb.detach().float().to(dev)
        target = train_scores.detach().float().to(dev)
        pos_weight = compute_pos_weight(target).to(dev) if use_pos_weight else None

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = max(1, math.ceil(len(emb) / batch_size)) * epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(emb), generator=gen)
            for start in range(0, len(emb), batch_size):
                idx = perm[start : start + batch_size]
                if len(idx) < 2:  # ranking/spearman terms need a pair
                    continue
                out = model(emb[idx])
                loss = silva_loss(
                    out["logits"],
                    target[idx],
                    pos_weight=pos_weight,
                    ranking_weight=ranking_weight,
                    soft_spearman_weight=soft_spearman_weight,
                    qwk_weight=qwk_weight,
                    label_smoothing=label_smoothing,
                    loss_truncation=loss_truncation,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
        model.eval()

        @torch.no_grad()
        def predict(query: torch.Tensor) -> torch.Tensor:
            return ordinal_score_from_logits(model(query.float().to(dev))["logits"]).float().cpu()

        return predict

    return fit
