import math

import pytest
import torch

from silva.losses import (
    make_ordinal_targets,
    ordinal_loss,
    ordinal_score_from_logits,
    silva_loss,
    unit_score_from_logits,
)


def test_make_ordinal_targets_maps_each_score():
    scores = torch.tensor([1, 2, 3, 4, 5])
    expected = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(make_ordinal_targets(scores), expected)


def test_ordinal_loss_at_zero_logits_is_log2():
    # BCE at logit 0 is -log(0.5) = log(2) for any target.
    logits = torch.zeros(3, 4)
    scores = torch.tensor([1, 3, 5])
    assert ordinal_loss(logits, scores).item() == pytest.approx(math.log(2))


def test_unit_score_is_zero_based_mean_of_threshold_probs():
    # logits 0 -> sigmoid 0.5 -> mean = 0.5
    assert unit_score_from_logits(torch.zeros(1, 4)).item() == pytest.approx(0.5)


def test_unit_score_bounds():
    assert unit_score_from_logits(torch.full((1, 4), 20.0)).item() == pytest.approx(1.0, abs=1e-6)
    assert unit_score_from_logits(torch.full((1, 4), -20.0)).item() == pytest.approx(0.0, abs=1e-6)


def test_ordinal_score_is_unit_rescaled_to_1_5():
    # logits 0 -> sum sigmoid = 2 -> 1 + 2 = 3.0
    assert ordinal_score_from_logits(torch.zeros(1, 4)).item() == pytest.approx(3.0)


def test_silva_loss_pure_ordinal_when_regression_is_zero():
    # logits 0 -> ordinal_score 3.0; target 3 -> SmoothL1 = 0 -> total = log(2)
    loss = silva_loss(torch.zeros(2, 4), torch.tensor([3, 3]), smooth_l1_weight=0.2)
    assert loss.item() == pytest.approx(math.log(2))


def test_silva_loss_adds_weighted_regression():
    # ordinal_score 3.0 vs target 5 -> SmoothL1(diff=2, beta=1) = 1.5; total = log2 + 0.2*1.5
    loss = silva_loss(torch.zeros(2, 4), torch.tensor([5, 5]), smooth_l1_weight=0.2)
    assert loss.item() == pytest.approx(math.log(2) + 0.3)
