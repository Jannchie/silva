import torch

from silva.models.aesthetic import EmbeddingAestheticModel
from silva_train.checkpoint import load_checkpoint, load_model, save_checkpoint


def test_roundtrips_weights_config_and_metrics(tmp_path):
    state = {"head.weight": torch.randn(4, 8), "head.bias": torch.zeros(4)}
    config = {"model": {"embedding_dim": 8, "hidden_dims": [16]}, "train": {"seed": 0}}
    metrics = {"spearman": 0.71, "mae": 0.42}

    save_checkpoint(tmp_path, state, config, metrics)

    # weights as safetensors, metadata as a plain JSON sidecar — no pickle.
    assert (tmp_path / "best.safetensors").exists()
    assert (tmp_path / "best.json").exists()
    assert not (tmp_path / "best.pt").exists()

    loaded_state, loaded_config, loaded_metrics = load_checkpoint(tmp_path)
    assert set(loaded_state) == set(state)
    assert torch.equal(loaded_state["head.weight"], state["head.weight"])
    assert loaded_config == config
    assert loaded_metrics == metrics


def test_load_accepts_the_weights_file_directly(tmp_path):
    save_checkpoint(tmp_path, {"w": torch.ones(3)}, {"model": {}}, {"spearman": 1.0})

    state, _config, metrics = load_checkpoint(tmp_path / "best.safetensors")
    assert torch.equal(state["w"], torch.ones(3))
    assert metrics == {"spearman": 1.0}


def test_load_model_rebuilds_residual_architecture(tmp_path):
    # the architecture-defining params (incl. n_residual_blocks) must all be threaded
    # through; a rebuild that drops n_residual_blocks loses the trunk.*.fc1/fc2 weights.
    model = EmbeddingAestheticModel(embedding_dim=16, dropout=0.1, hidden_dims=[32], n_residual_blocks=2).eval()
    config = {"model": {"embedding_dim": 16, "dropout": 0.1, "hidden_dims": [32], "n_residual_blocks": 2}, "train": {}}
    save_checkpoint(tmp_path, model.state_dict(), config, {"spearman": 0.5})

    loaded = load_model(tmp_path)

    assert set(loaded.state_dict()) == set(model.state_dict())  # residual-block keys survive
    x = torch.randn(4, 16)
    assert torch.allclose(loaded(x)["logits"], model(x)["logits"], atol=1e-6)


def test_load_model_returns_eval_mode_model(tmp_path):
    model = EmbeddingAestheticModel(embedding_dim=8, hidden_dims=[16]).eval()
    save_checkpoint(tmp_path, model.state_dict(), {"model": {"embedding_dim": 8, "hidden_dims": [16]}}, {})

    assert load_model(tmp_path).training is False
