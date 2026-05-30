"""Ordinal regression head with learnable, strictly-monotone thresholds."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from silva.scoring import NUM_THRESHOLDS


class OrdinalHead(nn.Module):
    """Maps a pooled feature to ``num_thresholds`` ordinal logits ``latent - threshold_k``.

    Thresholds are parameterised as ``base + cumsum(softplus(raw_deltas))`` so that
    ``threshold_1 < threshold_2 < ... < threshold_K`` holds for any parameter values
    (softplus is strictly positive, so the cumulative sum is strictly increasing).
    """

    def __init__(self, hidden_size: int, num_thresholds: int = NUM_THRESHOLDS) -> None:
        super().__init__()
        self.latent = nn.Linear(hidden_size, 1)
        self.base_threshold = nn.Parameter(torch.zeros(()))
        self.raw_deltas = nn.Parameter(torch.zeros(num_thresholds))

    def get_thresholds(self) -> torch.Tensor:
        return self.base_threshold + torch.cumsum(F.softplus(self.raw_deltas), dim=0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        latent = self.latent(features)  # [B, 1]
        return latent - self.get_thresholds().view(1, -1)  # [B, num_thresholds]
