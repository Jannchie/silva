import torch

from silva.models.aesthetic import EmbeddingAestheticModel


def test_forward_returns_only_logits_and_score():
    model = EmbeddingAestheticModel(embedding_dim=16)
    out = model(torch.randn(3, 16))
    assert set(out) == {"logits", "score"}  # ordinal_score dropped: it was just 1 + 4*score
    assert out["logits"].shape == (3, 4)
    assert out["score"].shape == (3,)


def test_score_in_unit_range():
    model = EmbeddingAestheticModel(embedding_dim=8)
    out = model(torch.randn(10, 8))
    assert torch.all((out["score"] >= 0) & (out["score"] <= 1))


def test_no_backbone_no_transformers_dependency():
    # The model must be constructible with no pretrained weights / no network.
    model = EmbeddingAestheticModel(embedding_dim=4)
    assert not hasattr(model, "vision")


def test_mlp_head_adds_capacity_and_keeps_output_shape():
    linear = EmbeddingAestheticModel(embedding_dim=16, hidden_dims=[])
    mlp = EmbeddingAestheticModel(embedding_dim=16, hidden_dims=[32, 16])
    assert sum(p.numel() for p in mlp.parameters()) > sum(p.numel() for p in linear.parameters())
    out = mlp(torch.randn(4, 16))
    assert out["logits"].shape == (4, 4)
    assert out["score"].shape == (4,)


def test_empty_hidden_dims_is_pure_linear_probe():
    # hidden_dims=[] must reproduce the original LayerNorm+Linear head exactly (no trunk).
    model = EmbeddingAestheticModel(embedding_dim=8, hidden_dims=[])
    assert not any(isinstance(m, torch.nn.GELU) for m in model.modules())
