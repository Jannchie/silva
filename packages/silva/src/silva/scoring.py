"""Score reconstruction from ordinal threshold logits.

Shared by the model's forward pass (inference) and by the training losses, so
published weights and the training loop agree on exactly what a logit means. No
``torch.nn``, no loss functions — just the logit -> score maps. The loss
functions that consume these live in ``silva_train.losses``.
"""

from __future__ import annotations

import torch

# 5 ordinal levels (scores 1..5) -> 4 binary "score > k" thresholds.
NUM_THRESHOLDS = 4


def unit_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Canonical output in ``[0, 1]``: the mean of the threshold probabilities."""
    return torch.sigmoid(logits).mean(dim=-1)


def ordinal_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Label-space score in ``[1, 5]``: ``1 + sum`` of the threshold probabilities."""
    return 1.0 + torch.sigmoid(logits).sum(dim=-1)
