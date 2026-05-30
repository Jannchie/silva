import torch

from silva.hub import HubAestheticModel


def test_save_pretrained_writes_safetensors_and_config(tmp_path):
    model = HubAestheticModel(embedding_dim=16, hidden_dims=[32])
    model.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "config.json").exists()


def test_from_pretrained_round_trips_weights(tmp_path):
    model = HubAestheticModel(embedding_dim=16, hidden_dims=[32]).eval()
    model.save_pretrained(tmp_path)
    loaded = HubAestheticModel.from_pretrained(tmp_path).eval()

    x = torch.randn(4, 16)
    assert torch.allclose(model(x)["score"], loaded(x)["score"], atol=1e-6)


def test_config_persists_constructor_args(tmp_path):
    HubAestheticModel(embedding_dim=16, dropout=0.2, hidden_dims=[8]).save_pretrained(tmp_path)
    loaded = HubAestheticModel.from_pretrained(tmp_path)
    # the LayerNorm width reflects embedding_dim recovered from config.json
    assert loaded.norm.normalized_shape == (16,)
