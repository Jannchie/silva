import pytest
import torch

from silva.scoring import ordinal_score_from_logits, unit_score_from_logits


def test_unit_score_is_zero_based_mean_of_threshold_probs():
    # logits 0 -> sigmoid 0.5 -> mean = 0.5
    assert unit_score_from_logits(torch.zeros(1, 4)).item() == pytest.approx(0.5)


def test_unit_score_bounds():
    assert unit_score_from_logits(torch.full((1, 4), 20.0)).item() == pytest.approx(1.0, abs=1e-6)
    assert unit_score_from_logits(torch.full((1, 4), -20.0)).item() == pytest.approx(0.0, abs=1e-6)


def test_ordinal_score_is_unit_rescaled_to_1_5():
    # logits 0 -> sum sigmoid = 2 -> 1 + 2 = 3.0
    assert ordinal_score_from_logits(torch.zeros(1, 4)).item() == pytest.approx(3.0)
