"""Personal ordinal aesthetic head on top of precomputed embeddings.

No SigLIP backbone here: v1 freezes the backbone, so embeddings are precomputed
upstream (by a script) and the training library only learns the head.

``EmbeddingAestheticModel`` IS the published model: an ordinal head plus
``PyTorchModelHubMixin``, so it round-trips through ``from_pretrained`` /
``push_to_hub`` (the constructor args land in ``config.json``, the weights +
calibration buffers in ``model.safetensors``). The Hub repo carries ONLY the head —
the frozen SigLIP2 backbone is upstream (see the model card).

It optionally carries a baked **calibration LUT** (``set_calibration``): a monotone
``latent -> score`` table that lets a single-image forward emit the SAME calibrated
score the library writer produces in batch (see ``silva_train.calibration``). Without
it, ``calibrated_score`` falls back to the raw ``score``.
"""

from __future__ import annotations

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from silva.models.ordinal_head import OrdinalHead
from silva.scoring import unit_score_from_logits

REPO_URL = "https://github.com/Jannchie/silva"
N_CAL_KNOTS = 512  # size of the baked calibration lookup table


class EmbeddingAestheticModel(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url=REPO_URL,
    pipeline_tag="image-classification",
    license="mit",
    tags=["aesthetic", "siglip2", "ordinal-regression", "image-scoring"],
):
    """``embedding[D] -> LayerNorm -> Dropout -> [MLP trunk] -> ordinal head``.

    forward returns:
      - ``logits``           : ordinal threshold logits ``[B, 4]``
      - ``score``            : raw aesthetic score in ``[0, 1]`` (mean threshold prob)
      - ``calibrated_score`` : score remapped through the baked calibration LUT, or
                               ``score`` when no calibration is set

    The constructor signature is captured by the Hub mixin and stored in ``config.json``;
    keep it JSON-serialisable (``embedding_dim``, ``dropout``, ``hidden_dims``).
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

        # baked calibration LUT (monotone latent -> calibrated score); zeros until set_calibration
        self.register_buffer("cal_lat_knots", torch.zeros(N_CAL_KNOTS))
        self.register_buffer("cal_score_knots", torch.zeros(N_CAL_KNOTS))
        self.register_buffer("cal_fitted", torch.zeros((), dtype=torch.bool))

    def set_calibration(self, lat_knots: torch.Tensor, score_knots: torch.Tensor) -> None:
        """Bake a ``(lat_knots, score_knots)`` lookup table (both length ``N_CAL_KNOTS``, ascending)."""
        lk = torch.as_tensor(lat_knots, dtype=torch.float32)
        sk = torch.as_tensor(score_knots, dtype=torch.float32)
        if lk.shape != (N_CAL_KNOTS,) or sk.shape != (N_CAL_KNOTS,):
            msg = f"calibration knots must both be length {N_CAL_KNOTS}, got {tuple(lk.shape)} / {tuple(sk.shape)}"
            raise ValueError(msg)
        self.cal_lat_knots.copy_(lk)
        self.cal_score_knots.copy_(sk)
        self.cal_fitted.fill_(value=True)

    def _calibrate(self, latent: torch.Tensor) -> torch.Tensor:
        # monotone linear interpolation over the baked (lat_knots -> score_knots) table,
        # clamped to the endpoints outside the trained latent range
        xp, fp = self.cal_lat_knots, self.cal_score_knots
        idx = torch.searchsorted(xp, latent).clamp(1, N_CAL_KNOTS - 1)
        x0, x1 = xp[idx - 1], xp[idx]
        y0, y1 = fp[idx - 1], fp[idx]
        t = ((latent - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0.0, 1.0)
        return y0 + t * (y1 - y0)

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.norm(embedding.float())
        x = self.dropout(x)
        x = self.trunk(x)
        logits = self.head(x)
        score = unit_score_from_logits(logits)
        if bool(self.cal_fitted):
            latent = self.head.latent(x).squeeze(-1)
            calibrated = self._calibrate(latent)
        else:
            calibrated = score
        return {"logits": logits, "score": score, "calibrated_score": calibrated}
