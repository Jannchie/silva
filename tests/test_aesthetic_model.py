import torch

from silva.models.aesthetic import EmbeddingAestheticModel


def test_forward_shapes():
    model = EmbeddingAestheticModel(embedding_dim=16)
    out = model(torch.randn(3, 16))
    assert out["logits"].shape == (3, 4)
    assert out["score"].shape == (3,)
    assert out["ordinal_score"].shape == (3,)


def test_score_in_unit_range_and_ordinal_in_1_5():
    model = EmbeddingAestheticModel(embedding_dim=8)
    out = model(torch.randn(10, 8))
    assert torch.all((out["score"] >= 0) & (out["score"] <= 1))
    assert torch.all((out["ordinal_score"] >= 1) & (out["ordinal_score"] <= 5))


def test_no_backbone_no_transformers_dependency():
    # The model must be constructible with no pretrained weights / no network.
    model = EmbeddingAestheticModel(embedding_dim=4)
    assert not hasattr(model, "vision")
