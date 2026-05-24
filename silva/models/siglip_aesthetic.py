"""SigLIP2 vision backbone + personal ordinal aesthetic head."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from transformers import AutoModel

from silva.losses import ordinal_score_from_logits, unit_score_from_logits
from silva.models.ordinal_head import OrdinalHead


class SigLIP2AestheticModel(nn.Module):
    """Image -> SigLIP2 pooled feature -> ordinal head.

    forward returns:
      - ``logits``        : ordinal threshold logits ``[B, 4]``
      - ``score``         : canonical output in ``[0, 1]`` (mean threshold prob)
      - ``ordinal_score`` : label-space score in ``[1, 5]`` (for regression term / metrics)

    ``aux_scorers`` reserves auxiliary external-scorer heads for v2; empty in v1.
    """

    def __init__(
        self,
        model_id: str = "google/siglip2-so400m-patch14-384",
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        aux_scorers: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        # AutoModel resolves the correct architecture from the checkpoint config.
        # (The fixed-resolution siglip2 checkpoints load as SigLIP-v1 vision towers.)
        # Keep only the vision tower and drop the unused text tower to save memory.
        backbone = AutoModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.vision = backbone.vision_model
        del backbone
        hidden_size = self.vision.config.hidden_size

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = OrdinalHead(hidden_size)
        # v2 stub: not constructed in v1 (aux_scorers is empty).
        self.aux_heads = nn.ModuleDict({name: nn.Linear(hidden_size, 1) for name in aux_scorers})

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.vision.parameters():
            param.requires_grad_(False)
        self.vision.eval()

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        pixel_values = pixel_values.to(self.vision.dtype)
        backbone_trainable = any(p.requires_grad for p in self.vision.parameters())
        ctx = nullcontext() if backbone_trainable else torch.no_grad()
        with ctx:
            pooled = self.vision(pixel_values=pixel_values).pooler_output

        x = self.norm(pooled.float())
        x = self.dropout(x)
        logits = self.head(x)
        return {
            "logits": logits,
            "score": unit_score_from_logits(logits),
            "ordinal_score": ordinal_score_from_logits(logits),
        }
