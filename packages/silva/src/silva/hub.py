"""Hugging Face Hub integration for the personal aesthetic head.

Lives in the ``silva`` inference library; ``huggingface-hub`` is a core dependency,
so no extra is needed. Kept separate from :mod:`silva.models.aesthetic` so the model
definition itself carries no Hub coupling.

``HubAestheticModel`` is :class:`~silva.models.aesthetic.EmbeddingAestheticModel`
plus :class:`~huggingface_hub.PyTorchModelHubMixin`, which serialises the model to
``safetensors`` and persists the constructor arguments to ``config.json`` so the
head round-trips through ``from_pretrained`` / ``push_to_hub`` with no extra code.

The Hub repo carries ONLY the head — the frozen SigLIP2 backbone that produces the
1152-d input embedding is upstream and not part of these weights. See the model card.
"""

from __future__ import annotations

from huggingface_hub import PyTorchModelHubMixin

from silva.models.aesthetic import EmbeddingAestheticModel

REPO_URL = "https://github.com/Jannchie/silva"


class HubAestheticModel(
    EmbeddingAestheticModel,
    PyTorchModelHubMixin,
    repo_url=REPO_URL,
    pipeline_tag="image-classification",
    license="mit",
    tags=["aesthetic", "siglip2", "ordinal-regression", "image-scoring"],
):
    """``EmbeddingAestheticModel`` that can ``push_to_hub`` / ``from_pretrained``.

    The constructor signature is captured by the mixin and stored verbatim in
    ``config.json``; keep it JSON-serialisable (``embedding_dim``, ``dropout``,
    ``hidden_dims``).
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.1, hidden_dims: list[int] | None = None) -> None:
        super().__init__(embedding_dim=embedding_dim, dropout=dropout, hidden_dims=hidden_dims)
