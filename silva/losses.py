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


def silva_loss(
    logits: torch.Tensor, scores: torch.Tensor, pos_weight: torch.Tensor | None = None
) -> torch.Tensor:
    """v1 personal loss: pure ordinal BCE, with optional per-threshold ``pos_weight``.

    No regression term: the personal 1~5 scale is deliberately non-equidistant
    (1=very bad, 2=bad, 3=ok-but-not-good, 4=nice, 5=very nice), so a SmoothL1
    pull toward equidistant integer labels would fight the ordinal head's
    self-adjusting thresholds. Model selection is by Spearman (ranking), which a
    regression term does not help. ``pos_weight`` (see :func:`compute_pos_weight`)
    balances the per-threshold class imbalance. See design §3.3 / §5 for rationale.
    """
    return ordinal_loss(logits, scores, pos_weight=pos_weight)
