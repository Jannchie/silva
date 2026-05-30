"""End-to-end facade: image(s) -> aesthetic score(s) in ``[0, 1]``.

``AestheticScorer`` ties the published head to the SigLIP2 backbone behind one
``score`` call, so callers never touch embeddings, processors, or tensors:

    from silva import AestheticScorer

    scorer = AestheticScorer.from_pretrained("Jannchie/silva-aesthetic")
    scorer.score("a.png")             # -> 0.73
    scorer.score(["a.png", "b.png"])  # -> [0.73, 0.41]

The backbone (``transformers`` + ``pillow``, the ``[backbone]`` extra) is loaded
lazily on the first ``score`` call, so importing this module — and constructing a
scorer from an already-loaded head — needs only the core install.
"""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING

from silva.backbone import Embedder, score_images
from silva.hub import HubAestheticModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL.Image import Image
    from torch import nn

    ImageInput = str | PathLike[str] | Image


class AestheticScorer:
    """Scores images for personal aesthetic appeal via a published head + SigLIP2 backbone."""

    def __init__(self, head: nn.Module, *, device: str | None = None) -> None:
        self.head = head.eval()
        self._device = device
        self._embedder: Embedder | None = None

    @classmethod
    def from_pretrained(cls, repo_id: str, *, device: str | None = None) -> AestheticScorer:
        """Load the published head from the Hugging Face Hub (e.g. ``"Jannchie/silva-aesthetic"``)."""
        return cls(HubAestheticModel.from_pretrained(repo_id), device=device)

    @property
    def embedder(self) -> Embedder:
        """The SigLIP2 backbone, loaded on first use and pinned to the head's device."""
        if self._embedder is None:
            self._embedder = Embedder(device=self._device)
            self.head.to(self._embedder.device)
        return self._embedder

    def score(self, images: ImageInput | Sequence[ImageInput]) -> float | list[float]:
        """Score one image (path or ``PIL.Image``) or a list of them.

        A single image returns a ``float``; a list/tuple returns a ``list[float]``.
        """
        batch = isinstance(images, (list, tuple))
        items = list(images) if batch else [images]
        scores = score_images([self._load(item) for item in items], self.head, self.embedder)
        return scores if batch else scores[0]

    @staticmethod
    def _load(image: ImageInput) -> Image:
        if isinstance(image, (str, PathLike)):
            from PIL import Image as PILImage  # noqa: PLC0415 — only needed to open paths

            return PILImage.open(image)
        return image
