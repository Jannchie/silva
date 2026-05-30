"""Ordinal regression losses and score reconstruction for the personal aesthetic head.

v1 uses only the personal (single-source) path. Multi-task hooks for external
scorers (calibration, auxiliary heads, disagreement weighting) are deferred to v2.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from silva.scoring import NUM_THRESHOLDS, ordinal_score_from_logits


def make_ordinal_targets(scores: torch.Tensor) -> torch.Tensor:
    """Map integer scores in {1..5} to binary threshold targets of shape ``[B, 4]``.

    Example: ``5 -> [1,1,1,1]``, ``3 -> [1,1,0,0]``, ``1 -> [0,0,0,0]``.
    """
    thresholds = torch.arange(1, 1 + NUM_THRESHOLDS, device=scores.device)
    return (scores.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def ordinal_loss(
    logits: torch.Tensor,
    scores: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    targets = make_ordinal_targets(scores).to(logits.dtype)
    if pos_weight is not None:
        pos_weight = pos_weight.to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction=reduction)


def compute_pos_weight(scores: torch.Tensor) -> torch.Tensor:
    """Per-threshold ``pos_weight`` (#neg / #pos) for ordinal BCE, from the train split.

    Balances each "score > k" threshold independently so the rare tails (few 1s, few
    5s) are not drowned out by the bulk. Pass the result to :func:`ordinal_loss` /
    :func:`silva_loss`. Compute on the train split ONLY — never val/test (leakage).
    """
    scores = scores.reshape(-1)
    thresholds = torch.arange(1, 1 + NUM_THRESHOLDS, device=scores.device)
    pos = (scores.unsqueeze(1) > thresholds.unsqueeze(0)).sum(dim=0).float()
    neg = scores.shape[0] - pos
    return neg / pos


def pairwise_ranking_loss(logits: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """RankNet-style pairwise logistic loss — directly optimises ranking (Spearman).

    For every ordered pair (i, j) with ``score_i > score_j``, pushes i's continuous
    ``ordinal_score`` above j's. This aligns training with the Spearman selection
    metric, which ordinal BCE only optimises indirectly. Returns a graph-preserving
    zero when the batch has no ordered pairs (all scores equal).
    """
    rank = ordinal_score_from_logits(logits)
    diff_rank = rank.unsqueeze(1) - rank.unsqueeze(0)
    diff_score = scores.unsqueeze(1) - scores.unsqueeze(0)
    mask = diff_score > 0
    if not mask.any():
        return logits.sum() * 0.0
    ordered = diff_rank[mask]
    return F.binary_cross_entropy_with_logits(ordered, torch.ones_like(ordered))


def soft_spearman_loss(logits: torch.Tensor, scores: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """Differentiable Spearman surrogate: ``1 - corr(soft_rank(pred), target)``.

    Spearman is non-differentiable because ranking is a hard sort. We relax the
    rank of each item to ``sum_j sigmoid((pred_i - pred_j) / temp)`` — a smooth count
    of how many items it outranks — then maximise its Pearson correlation with the
    targets. Unlike :func:`pairwise_ranking_loss` (a per-pair AUC/Kendall surrogate),
    this optimises the *global* rank correlation that the model is selected on.
    Returns a graph-preserving zero when targets have no variance (no ranking signal).
    """
    pred = ordinal_score_from_logits(logits)
    diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    soft_rank = torch.sigmoid(diff / temp).sum(dim=1)
    target = scores.to(soft_rank.dtype)

    a = soft_rank - soft_rank.mean()
    b = target - target.mean()
    denom = a.norm() * b.norm()
    if denom < 1e-8:
        return logits.sum() * 0.0
    return 1.0 - (a * b).sum() / denom


def listwise_loss(logits: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """ListNet top-one listwise loss: cross-entropy between the softmax score lists.

    Treats the batch as one ranked list and matches the predicted top-one probability
    distribution ``softmax(pred)`` to the target distribution ``softmax(target)``. A
    listwise view complements the pairwise/global surrogates above. Targets are taken
    as the raw 1~5 labels (a fixed monotone target distribution over the batch).
    """
    pred = ordinal_score_from_logits(logits)
    target_p = torch.softmax(scores.to(pred.dtype), dim=0)
    return -(target_p * torch.log_softmax(pred, dim=0)).sum()


def silva_loss(
    logits: torch.Tensor,
    scores: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    ranking_weight: float = 0.0,
    soft_spearman_weight: float = 0.0,
) -> torch.Tensor:
    """Personal loss: ordinal BCE + optional ``pos_weight`` + ranking + soft-Spearman.

    No SmoothL1 regression: the personal 1~5 scale is deliberately non-equidistant
    (1=very bad … 5=very nice), so pulling toward equidistant integer labels would
    fight the ordinal head's self-adjusting thresholds. ``pos_weight`` (see
    :func:`compute_pos_weight`) balances per-threshold imbalance. Two ranking terms
    sharpen the Spearman objective the model is selected on: ``ranking_weight`` adds
    the pairwise :func:`pairwise_ranking_loss`, and ``soft_spearman_weight`` adds the
    global :func:`soft_spearman_loss` (also markedly improves score calibration /
    MAE). The tuned recipe is ``ranking_weight=1.0, soft_spearman_weight=0.5``.
    See design §3.3 / §5.
    """
    loss = ordinal_loss(logits, scores, pos_weight=pos_weight)
    if ranking_weight > 0:
        loss = loss + ranking_weight * pairwise_ranking_loss(logits, scores)
    if soft_spearman_weight > 0:
        loss = loss + soft_spearman_weight * soft_spearman_loss(logits, scores)
    return loss
