"""Personal ordinal aesthetic head on top of precomputed embeddings.

No SigLIP backbone here: v1 freezes the backbone, so embeddings are precomputed
upstream (by a script) and the training library only learns the head. This keeps
the training library free of ``transformers`` and lets the model be unit-tested
without loading any pretrained weights.

``hidden_dims`` optionally inserts an MLP trunk before the ordinal head: a single
``Linear`` (``hidden_dims=[]``) is a linear probe, which caps how much non-linear
aesthetic signal it can pull from the frozen embedding; a small MLP lifts that cap.
"""

from __future__ import annotations

import torch
from torch import nn

from silva.losses import ordinal_score_from_logits, unit_score_from_logits
from silva.models.ordinal_head import OrdinalHead


class EmbeddingAestheticModel(nn.Module):
    """``embedding[D] -> LayerNorm -> Dropout -> [MLP trunk] -> ordinal head``.

    forward returns:
      - ``logits``        : ordinal threshold logits ``[B, 4]``
      - ``score``         : canonical output in ``[0, 1]`` (mean threshold prob)
      - ``ordinal_score`` : label-space score in ``[1, 5]`` (readable metrics)
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.1, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        hidden_dims = hidden_dims or []
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        trunk: list[nn.Module] = []
        in_dim = embedding_dim
        for h in hidden_dims:
            trunk += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        self.trunk = nn.Sequential(*trunk)
        self.head = OrdinalHead(in_dim)

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.norm(embedding.float())
        x = self.dropout(x)
        x = self.trunk(x)
        logits = self.head(x)
        return {
            "logits": logits,
            "score": unit_score_from_logits(logits),
            "ordinal_score": ordinal_score_from_logits(logits),
        }
