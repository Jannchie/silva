import math

import pytest
import torch

from silva.losses import (
    compute_pos_weight,
    listwise_loss,
    make_ordinal_targets,
    ordinal_loss,
    ordinal_score_from_logits,
    pairwise_ranking_loss,
    silva_loss,
    soft_spearman_loss,
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


def test_silva_loss_is_pure_ordinal_bce():
    # v1 loss is pure ordinal BCE — no regression term pulling predictions toward
    # equidistant 1~5 labels (personal scores are deliberately non-equidistant).
    logits = torch.zeros(2, 4)
    scores = torch.tensor([5, 5])
    assert silva_loss(logits, scores).item() == pytest.approx(ordinal_loss(logits, scores).item())
    # logits 0, any score -> BCE only = log(2); the old SmoothL1 term would have added 0.3.
    assert silva_loss(logits, scores).item() == pytest.approx(math.log(2))


def test_compute_pos_weight_balances_each_threshold():
    # one of each score 1..5:
    #   >1: pos={2,3,4,5}=4 neg={1}=1     -> 0.25
    #   >2: pos={3,4,5}=3   neg={1,2}=2   -> 2/3
    #   >3: pos={4,5}=2     neg={1,2,3}=3 -> 1.5
    #   >4: pos={5}=1       neg={1..4}=4  -> 4.0
    pw = compute_pos_weight(torch.tensor([1, 2, 3, 4, 5]))
    assert torch.allclose(pw, torch.tensor([0.25, 2 / 3, 1.5, 4.0]))


def test_ordinal_loss_pos_weight_none_equals_unweighted():
    logits, scores = torch.zeros(2, 4), torch.tensor([3, 3])
    ones = torch.ones(4)
    assert ordinal_loss(logits, scores, pos_weight=ones).item() == pytest.approx(
        ordinal_loss(logits, scores).item()
    )


def test_ordinal_loss_pos_weight_changes_loss():
    logits, scores = torch.zeros(2, 4), torch.tensor([3, 3])
    weighted = ordinal_loss(logits, scores, pos_weight=torch.full((4,), 2.0)).item()
    assert weighted != pytest.approx(ordinal_loss(logits, scores).item())


def test_pairwise_ranking_loss_lower_when_ordered():
    scores = torch.tensor([1, 3, 5])
    ordered = torch.tensor([[-9.0, -9, -9, -9], [0.0, 0, 0, 0], [9.0, 9, 9, 9]])  # ordinal ~1,3,5
    reversed_ = torch.flip(ordered, dims=[0])  # ordinal ~5,3,1 vs scores 1,3,5
    assert pairwise_ranking_loss(ordered, scores).item() < pairwise_ranking_loss(reversed_, scores).item()


def test_pairwise_ranking_loss_zero_when_all_scores_equal():
    # no ordered pairs -> no ranking signal -> exactly zero (and must keep the graph)
    logits = torch.randn(3, 4, requires_grad=True)
    loss = pairwise_ranking_loss(logits, torch.tensor([3, 3, 3]))
    assert loss.item() == pytest.approx(0.0)
    loss.backward()  # must not error


def test_soft_spearman_loss_lower_when_ordered():
    scores = torch.tensor([1, 2, 3, 4, 5])
    ordered = torch.tensor(  # ordinal scores roughly increase with the targets
        [[-9.0, -9, -9, -9], [-3.0, -3, -3, -3], [0.0, 0, 0, 0], [3.0, 3, 3, 3], [9.0, 9, 9, 9]]
    )
    reversed_ = torch.flip(ordered, dims=[0])  # ordinal scores decrease -> anti-correlated
    assert soft_spearman_loss(ordered, scores).item() < soft_spearman_loss(reversed_, scores).item()


def test_soft_spearman_loss_near_zero_for_perfect_ranking():
    # monotone increasing predictions vs increasing targets -> correlation ~1 -> loss ~0
    scores = torch.tensor([1, 2, 3, 4, 5])
    perfect = torch.tensor([[-12.0] * 4, [-6.0] * 4, [0.0] * 4, [6.0] * 4, [12.0] * 4])
    assert soft_spearman_loss(perfect, scores).item() == pytest.approx(0.0, abs=0.1)


def test_soft_spearman_loss_zero_when_all_scores_equal():
    # zero target variance -> correlation undefined -> graph-preserving zero
    logits = torch.randn(4, 4, requires_grad=True)
    loss = soft_spearman_loss(logits, torch.tensor([3, 3, 3, 3]))
    assert loss.item() == pytest.approx(0.0)
    loss.backward()  # must not error


def test_listwise_loss_lower_when_ordered():
    scores = torch.tensor([1, 2, 3, 4, 5])
    ordered = torch.tensor(
        [[-9.0, -9, -9, -9], [-3.0, -3, -3, -3], [0.0, 0, 0, 0], [3.0, 3, 3, 3], [9.0, 9, 9, 9]]
    )
    reversed_ = torch.flip(ordered, dims=[0])
    assert listwise_loss(ordered, scores).item() < listwise_loss(reversed_, scores).item()


def test_listwise_loss_keeps_graph():
    logits = torch.randn(5, 4, requires_grad=True)
    loss = listwise_loss(logits, torch.tensor([1, 2, 3, 4, 5]))
    loss.backward()  # must not error
    assert logits.grad is not None


def test_silva_loss_combines_ranking_and_soft_spearman_terms():
    # the winning recipe: ordinal BCE + 1.0*ranking + 0.5*soft_spearman
    logits = torch.randn(8, 4)
    scores = torch.tensor([1, 2, 3, 4, 5, 1, 3, 5])
    combined = silva_loss(logits, scores, ranking_weight=1.0, soft_spearman_weight=0.5)
    expected = (
        ordinal_loss(logits, scores)
        + 1.0 * pairwise_ranking_loss(logits, scores)
        + 0.5 * soft_spearman_loss(logits, scores)
    )
    assert combined.item() == pytest.approx(expected.item(), abs=1e-5)


def test_silva_loss_soft_spearman_weight_zero_is_noop():
    logits = torch.randn(6, 4)
    scores = torch.tensor([1, 2, 3, 4, 5, 2])
    base = silva_loss(logits, scores, ranking_weight=1.0)
    assert silva_loss(logits, scores, ranking_weight=1.0, soft_spearman_weight=0.0).item() == pytest.approx(base.item())
