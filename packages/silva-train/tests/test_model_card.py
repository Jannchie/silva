from silva_train.config import ModelConfig
from silva_train.model_card import render_model_card


def test_render_model_card_reads_architecture_from_model_config():
    cfg = ModelConfig(embedding_dim=1152, hidden_dims=[512], n_residual_blocks=6)
    card = render_model_card("user/silva-aesthetic", "google/siglip2-so400m", cfg, {"spearman": 0.773})

    assert "512" in card
    assert "6× residual block" in card
    assert "0.7730" in card  # metric formatted into the card


def test_render_model_card_calls_a_bare_trunk_a_linear_probe():
    cfg = ModelConfig(embedding_dim=1152, hidden_dims=[], n_residual_blocks=0)
    card = render_model_card("u/s", "backbone", cfg, {})

    assert "linear probe" in card
