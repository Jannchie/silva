import torch

from silva.models.aesthetic import N_CAL_KNOTS, EmbeddingAestheticModel


def test_save_pretrained_writes_safetensors_and_config(tmp_path):
    model = EmbeddingAestheticModel(embedding_dim=16, hidden_dims=[32])
    model.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "config.json").exists()


def test_from_pretrained_round_trips_weights(tmp_path):
    model = EmbeddingAestheticModel(embedding_dim=16, hidden_dims=[32]).eval()
    model.save_pretrained(tmp_path)
    loaded = EmbeddingAestheticModel.from_pretrained(tmp_path).eval()

    x = torch.randn(4, 16)
    with torch.no_grad():
        assert torch.allclose(model(x)["score"], loaded(x)["score"], atol=1e-6)


def test_config_persists_constructor_args(tmp_path):
    EmbeddingAestheticModel(embedding_dim=16, dropout=0.2, hidden_dims=[8]).save_pretrained(tmp_path)
    loaded = EmbeddingAestheticModel.from_pretrained(tmp_path)
    # the LayerNorm width reflects embedding_dim recovered from config.json
    assert loaded.norm.normalized_shape == (16,)


def test_residual_blocks_round_trip_through_pretrained(tmp_path):
    # the published SDK path must reconstruct a deep residual head from config.json alone:
    # n_residual_blocks is captured by the Hub mixin, so from_pretrained rebuilds the same
    # architecture and the residual-block weights survive the safetensors round-trip.
    model = EmbeddingAestheticModel(embedding_dim=16, hidden_dims=[32], n_residual_blocks=3).eval()
    model.save_pretrained(tmp_path)
    loaded = EmbeddingAestheticModel.from_pretrained(tmp_path).eval()

    x = torch.randn(4, 16)
    with torch.no_grad():
        assert torch.allclose(model(x)["score"], loaded(x)["score"], atol=1e-6)


def test_calibrated_score_falls_back_to_raw_when_unfitted():
    model = EmbeddingAestheticModel(embedding_dim=16).eval()
    x = torch.randn(4, 16)
    with torch.no_grad():
        out = model(x)
    # no calibration baked -> calibrated_score is exactly the raw score
    assert torch.allclose(out["calibrated_score"], out["score"])


def test_set_calibration_applies_and_round_trips(tmp_path):
    model = EmbeddingAestheticModel(embedding_dim=16).eval()
    lat = torch.linspace(-5, 5, N_CAL_KNOTS)
    sco = torch.linspace(0, 1, N_CAL_KNOTS)
    model.set_calibration(lat, sco)
    x = torch.randn(8, 16)
    with torch.no_grad():
        out = model(x)
    assert out["calibrated_score"].min() >= 0.0
    assert out["calibrated_score"].max() <= 1.0
    # the baked LUT must survive save/load (it lives in safetensors buffers)
    model.save_pretrained(tmp_path)
    loaded = EmbeddingAestheticModel.from_pretrained(tmp_path).eval()
    assert bool(loaded.cal_fitted)
    with torch.no_grad():
        assert torch.allclose(loaded(x)["calibrated_score"], out["calibrated_score"], atol=1e-5)
