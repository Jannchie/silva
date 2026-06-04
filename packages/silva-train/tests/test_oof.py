import pytest
import torch

from silva_train.metrics import spearman
from silva_train.oof import make_fit_head, oof_predictions


def _identity_rows(n: int, d: int) -> torch.Tensor:
    """Rows unique by content so a predictor can recognise membership by value."""
    emb = torch.zeros(n, d)
    emb[torch.arange(n), torch.arange(n) % d] = torch.arange(1, n + 1).float()
    return emb


def _membership_fit(train_emb: torch.Tensor, _train_scores: torch.Tensor):
    """Fake fit: predicts 1.0 for any query row it was trained on, else 0.0."""

    def predict(query: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(len(query))
        for i, q in enumerate(query):
            out[i] = float((train_emb == q).all(dim=1).any())
        return out

    return predict


def test_every_row_gets_a_finite_prediction():
    emb = _identity_rows(30, 8)
    scores = torch.arange(30).float() % 5 + 1
    folds = [i % 3 for i in range(30)]

    preds = oof_predictions(emb, scores, folds, _membership_fit)

    assert preds.shape == (30,)
    assert torch.isfinite(preds).all()


def test_prediction_never_comes_from_a_model_that_saw_the_row():
    emb = _identity_rows(30, 8)
    scores = torch.arange(30).float() % 5 + 1
    folds = [i % 3 for i in range(30)]

    preds = oof_predictions(emb, scores, folds, _membership_fit)

    # membership predictor outputs 1.0 iff the model trained on that row -> all must be 0
    assert torch.equal(preds, torch.zeros(30))


def test_single_fold_has_no_training_complement():
    emb = _identity_rows(10, 4)
    scores = torch.ones(10)

    with pytest.raises(ValueError, match="fold"):
        oof_predictions(emb, scores, [0] * 10, _membership_fit)


def test_make_fit_head_recovers_synthetic_signal_out_of_fold():
    """The default head must learn a real signal: OOF predictions (never trained on the
    row) should rank a noiseless synthetic latent far above chance."""
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(600, 16, generator=g)
    w = torch.randn(16, generator=g)
    latent = emb @ w
    # bucket the latent into 1..5 by quintile -> a perfectly learnable ordinal target
    scores = torch.bucketize(latent, torch.quantile(latent, torch.tensor([0.2, 0.4, 0.6, 0.8]))).float() + 1
    folds = [i % 3 for i in range(600)]

    fit = make_fit_head(hidden_dims=[32], epochs=30, batch_size=128, lr=3e-3)
    preds = oof_predictions(emb, scores, folds, fit)

    assert spearman(preds, scores) > 0.6


def test_make_fit_head_is_deterministic():
    g = torch.Generator().manual_seed(1)
    emb = torch.randn(120, 8, generator=g)
    scores = torch.arange(120).float() % 5 + 1
    folds = [i % 2 for i in range(120)]

    fit = make_fit_head(hidden_dims=[16], epochs=3, batch_size=64)
    a = oof_predictions(emb, scores, folds, fit)
    b = oof_predictions(emb, scores, folds, fit)

    assert torch.equal(a, b)
