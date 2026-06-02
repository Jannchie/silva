"""Ordinal regression losses and score reconstruction for the personal aesthetic head.

v1 uses only the personal (single-source) path. Multi-task hooks for external
scorers (calibration, auxiliary heads, disagreement weighting) are deferred to v2.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from silva.scoring import NUM_THRESHOLDS, ordinal_score_from_logits


def make_ordinal_targets(scores: torch.Tensor) -> torch.Tensor:
    """Map scores to ordinal threshold targets of shape ``[B, 4]``.

    Integer: ``5 -> [1,1,1,1]``, ``3 -> [1,1,0,0]``, ``1 -> [0,0,0,0]``.
    Continuous (mixup): ``2.7 -> [1.0, 0.7, 0.0, 0.0]`` — linearly interpolated.
    """
    thresholds = torch.arange(1, 1 + NUM_THRESHOLDS, device=scores.device, dtype=scores.dtype)
    return (scores.unsqueeze(1) - thresholds.unsqueeze(0)).clamp(0, 1)


def ordinal_loss(
    logits: torch.Tensor,
    scores: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    targets = make_ordinal_targets(scores).to(logits.dtype)
    if label_smoothing > 0:
        # Soften {0, 1} -> {eps, 1-eps}: the optimum logit becomes the FINITE value
        # ln((1-eps)/eps) instead of +-inf, so the head stops pushing latent to +-40
        # and sigmoid never saturates (this is what crushes the 0~1 tail pile-up).
        targets = targets * (1 - 2 * label_smoothing) + label_smoothing
    if pos_weight is not None:
        pos_weight = pos_weight.to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction=reduction)


def compute_pos_weight(scores: torch.Tensor) -> torch.Tensor:
    """Per-threshold ``pos_weight`` (#neg / #pos) for ordinal BCE, from the train split.

    Balances each "score > k" threshold independently so the rare tails (few 1s, few
    5s) are not drowned out by the bulk. Pass the result to :func:`ordinal_loss` /
    :func:`silva_loss`. Compute on the train split ONLY — never val/test (leakage).
    """
    scores = scores.reshape(-1)
    thresholds = torch.arange(1, 1 + NUM_THRESHOLDS, device=scores.device)
    pos = (scores.unsqueeze(1) > thresholds.unsqueeze(0)).sum(dim=0).float()
    neg = scores.shape[0] - pos
    return neg / pos


def pairwise_ranking_loss(logits: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """RankNet-style pairwise logistic loss — directly optimises ranking (Spearman).

    For every ordered pair (i, j) with ``score_i > score_j``, pushes i's continuous
    ``ordinal_score`` above j's. This aligns training with the Spearman selection
    metric, which ordinal BCE only optimises indirectly. Returns a graph-preserving
    zero when the batch has no ordered pairs (all scores equal).
    """
    rank = ordinal_score_from_logits(logits)
    diff_rank = rank.unsqueeze(1) - rank.unsqueeze(0)
    diff_score = scores.unsqueeze(1) - scores.unsqueeze(0)
    mask = diff_score > 0
    if not mask.any():
        return logits.sum() * 0.0
    ordered = diff_rank[mask]
    return F.binary_cross_entropy_with_logits(ordered, torch.ones_like(ordered))


def soft_spearman_loss(logits: torch.Tensor, scores: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """Differentiable Spearman surrogate: ``1 - corr(soft_rank(pred), target)``.

    Spearman is non-differentiable because ranking is a hard sort. We relax the
    rank of each item to ``sum_j sigmoid((pred_i - pred_j) / temp)`` — a smooth count
    of how many items it outranks — then maximise its Pearson correlation with the
    targets. Unlike :func:`pairwise_ranking_loss` (a per-pair AUC/Kendall surrogate),
    this optimises the *global* rank correlation that the model is selected on.
    Returns a graph-preserving zero when targets have no variance (no ranking signal).
    """
    pred = ordinal_score_from_logits(logits)
    diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    soft_rank = torch.sigmoid(diff / temp).sum(dim=1)
    target = scores.to(soft_rank.dtype)

    a = soft_rank - soft_rank.mean()
    b = target - target.mean()
    denom = a.norm() * b.norm()
    if denom < 1e-8:
        return logits.sum() * 0.0
    return 1.0 - (a * b).sum() / denom


def listwise_loss(logits: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """ListNet top-one listwise loss: cross-entropy between the softmax score lists.

    Treats the batch as one ranked list and matches the predicted top-one probability
    distribution ``softmax(pred)`` to the target distribution ``softmax(target)``. A
    listwise view complements the pairwise/global surrogates above. Targets are taken
    as the raw 1~5 labels (a fixed monotone target distribution over the batch).
    """
    pred = ordinal_score_from_logits(logits)
    target_p = torch.softmax(scores.to(pred.dtype), dim=0)
    return -(target_p * torch.log_softmax(pred, dim=0)).sum()


def ordinal_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Ordinal threshold logits ``[B, 4]`` -> per-class probabilities ``[B, 5]``.

    ``P(y>k) = sigmoid(logit_k)`` is monotone-decreasing for the head's
    ``latent - increasing thresholds`` logits, so ``P(y=1)=1-P(y>1)``,
    ``P(y=k)=P(y>k-1)-P(y>k)``, ``P(y=5)=P(y>4)`` form a valid distribution.
    """
    cum = torch.sigmoid(logits)  # P(y > k), k = 1..4
    p_first = 1 - cum[:, 0:1]
    p_mid = cum[:, :-1] - cum[:, 1:]
    p_last = cum[:, NUM_THRESHOLDS - 1 : NUM_THRESHOLDS]
    return torch.cat([p_first, p_mid, p_last], dim=1).clamp_min(1e-6)


def qwk_loss(logits: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Differentiable quadratic-weighted-kappa loss (``1 - kappa``) over the batch.

    Weights each prediction error by the SQUARE of the rating gap, so a 4-off mistake
    costs ~16x a 1-off one. Unlike ordinal BCE (linear in the gap) and the ranking terms
    (gap-blind), this directly suppresses large-gap blunders (you=4 / model=1).
    """
    n_classes = NUM_THRESHOLDS + 1
    probs = ordinal_to_probs(logits)  # [B, 5]
    b = probs.shape[0]
    ratings = torch.arange(1, 1 + n_classes, device=logits.device, dtype=probs.dtype)
    weights = (ratings.view(-1, 1) - ratings.view(1, -1)) ** 2 / (n_classes - 1) ** 2
    s = (scores.float() - 1).clamp(0, n_classes - 1 - 1e-6)
    lo = s.long()
    hi = (lo + 1).clamp(max=n_classes - 1)
    w = (s - lo.float()).unsqueeze(1)
    onehot = torch.zeros(b, n_classes, device=logits.device, dtype=probs.dtype)
    onehot.scatter_(1, lo.unsqueeze(1), 1 - w)
    onehot.scatter_add_(1, hi.unsqueeze(1), w)  # [B, 5] — soft for mixup, exact for integers
    observed = probs.t() @ onehot  # soft confusion [5, 5]
    expected = torch.outer(probs.sum(0), onehot.sum(0)) / b
    return (weights * observed).sum() / ((weights * expected).sum() + 1e-8)


def silva_loss(
    logits: torch.Tensor,
    scores: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    ranking_weight: float = 0.0,
    soft_spearman_weight: float = 0.0,
    qwk_weight: float = 0.0,
    label_smoothing: float = 0.0,
    loss_truncation: float = 0.0,
) -> torch.Tensor:
    """Personal loss: ordinal BCE + optional ``pos_weight`` + ranking + soft-Spearman + QWK.

    ``loss_truncation`` (0..1) drops the top-k% highest per-sample ordinal losses before
    averaging — those are likely mislabelled samples that would pull the model astray.
    """
    if loss_truncation > 0:
        per_sample = ordinal_loss(logits, scores, pos_weight=pos_weight, label_smoothing=label_smoothing, reduction="none")
        per_sample = per_sample.mean(dim=1)  # [B, 4] -> [B]
        keep = int(len(per_sample) * (1 - loss_truncation))
        if keep > 0:
            _, idx = per_sample.topk(keep, largest=False)
            loss = per_sample[idx].mean()
        else:
            loss = per_sample.mean()
    else:
        loss = ordinal_loss(logits, scores, pos_weight=pos_weight, label_smoothing=label_smoothing)
    if ranking_weight > 0:
        loss = loss + ranking_weight * pairwise_ranking_loss(logits, scores)
    if soft_spearman_weight > 0:
        loss = loss + soft_spearman_weight * soft_spearman_loss(logits, scores)
    if qwk_weight > 0:
        loss = loss + qwk_weight * qwk_loss(logits, scores)
    return loss
