import torch

from silva.backbone import BACKBONE, score_images
from silva.models.aesthetic import EmbeddingAestheticModel


def test_backbone_is_pinned_to_patch14():
    # Must match pictoria ai/siglip_embed.py — patch14, NOT patch16.
    assert BACKBONE == "google/siglip2-so400m-patch14-384"


def test_score_images_runs_head_on_embedder_output():
    head = EmbeddingAestheticModel(embedding_dim=16).eval()

    class DummyEmbedder:
        def embed(self, image):
            return torch.randn(1, 16)

    results = score_images(["fake-image"], head, DummyEmbedder())
    assert len(results) == 1
    assert isinstance(results[0], float)
    assert 0.0 <= results[0] <= 1.0


def test_cli_main_is_importable_without_backbone_extra():
    # Importing the CLI entry point must not require transformers/pillow.
    from silva.cli import main

    assert callable(main)
