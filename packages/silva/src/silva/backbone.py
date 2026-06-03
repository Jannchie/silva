"""Image -> 1152-d SigLIP2 embedding, matching how the training vectors were made.

This is the single gatekeeper for train/inference consistency. It pins the
backbone to ``patch14-384`` and takes the raw pooled feature (``pooler_output``) —
exactly the path verified against the stored training embeddings at cosine 0.9998
(see ``scripts/verify_embedding.py``). A different backbone / patch size / pooling
produces vectors the published head was never trained on.

Needs the ``[backbone]`` extra (``transformers`` + ``pillow``). The heavy import is
deferred to ``Embedder.__init__`` so this module — and the CLI entry point — import
cleanly in a core-only install and fails with a clear message only when actually run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

BACKBONE = "google/siglip2-so400m-patch14-384"

if TYPE_CHECKING:
    from PIL.Image import Image


class Embedder:
    """Loads the frozen SigLIP2 backbone once and turns images into embeddings."""

    def __init__(self, device: str | None = None) -> None:
        try:
            from transformers import AutoModel, AutoProcessor  # noqa: PLC0415
        except ImportError as e:
            msg = 'silva image scoring needs the backbone extra: pip install "silva-scorer[backbone]"'
            raise ImportError(msg) from e
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(BACKBONE)
        self.model = AutoModel.from_pretrained(BACKBONE).to(self.device).eval()

    @torch.no_grad()
    def embed(self, image: Image) -> torch.Tensor:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(pixel_values=inputs.pixel_values)
        # newer transformers wraps the result; the pooled [1, 1152] vector is .pooler_output
        return (feats.pooler_output if hasattr(feats, "pooler_output") else feats).float()
