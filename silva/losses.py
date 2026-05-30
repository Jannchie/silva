"""Ordinal regression losses and score reconstruction for the personal aesthetic head.

v1 uses only the personal (single-source) path. Multi-task hooks for external
scorers (calibration, auxiliary heads, disagreement weighting) are deferred to v2.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# 5 ordinal levels (scores 1..5) -> 4 binary "score > k" thresholds.
NUM_THRESHOLDS = 4


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


def unit_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Canonical output in ``[0, 1]``: the mean of the 4 threshold probabilities."""
    return torch.sigmoid(logits).mean(dim=-1)


def ordinal_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Label-space score in ``[1, 5]`` used for the regression term and readable metrics."""
    return 1.0 + torch.sigmoid(logits).sum(dim=-1)


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


def silva_loss(
    logits: torch.Tensor,
    scores: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    ranking_weight: float = 0.0,
) -> torch.Tensor:
    """Personal loss: ordinal BCE + optional ``pos_weight`` + optional ranking term.

    No SmoothL1 regression: the personal 1~5 scale is deliberately non-equidistant
    (1=very bad … 5=very nice), so pulling toward equidistant integer labels would
    fight the ordinal head's self-adjusting thresholds. ``pos_weight`` (see
    :func:`compute_pos_weight`) balances per-threshold imbalance; ``ranking_weight``
    adds :func:`pairwise_ranking_loss` to directly optimise the Spearman objective.
    See design §3.3 / §5.
    """
    loss = ordinal_loss(logits, scores, pos_weight=pos_weight)
    if ranking_weight > 0:
        loss = loss + ranking_weight * pairwise_ranking_loss(logits, scores)
    return loss
