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


def ordinal_loss(logits: torch.Tensor, scores: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    targets = make_ordinal_targets(scores).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)


def unit_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Canonical output in ``[0, 1]``: the mean of the 4 threshold probabilities."""
    return torch.sigmoid(logits).mean(dim=-1)


def ordinal_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Label-space score in ``[1, 5]`` used for the regression term and readable metrics."""
    return 1.0 + torch.sigmoid(logits).sum(dim=-1)


def silva_loss(logits: torch.Tensor, scores: torch.Tensor, smooth_l1_weight: float = 0.2) -> torch.Tensor:
    """v1 personal loss: ordinal BCE + ``smooth_l1_weight`` * SmoothL1 in 1~5 space."""
    loss_ord = ordinal_loss(logits, scores)
    pred = ordinal_score_from_logits(logits)
    loss_reg = F.smooth_l1_loss(pred, scores.float())
    return loss_ord + smooth_l1_weight * loss_reg
