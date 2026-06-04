"""Estimate non-uniform category anchors: where do the 1~5 grades REALLY sit?

The rating scale is ordinal, not interval — the rater's own retest data shows the 3/4
boundary is about half as wide as its neighbours (3<->4 swaps ~29% vs ~17%). Training
losses that price errors by ``(i - j)^2`` therefore over-penalise confusions across the
fuzzy boundary and manufacture artifacts (the bimodal-latent canyon at the 3/4 cut).
These estimators turn observed confusability into latent anchor positions for
:func:`silva_train.losses.qwk_loss`.

Thurstone's law of comparative judgment, adjacent pairs only: with equal-noise Gaussian
readings, a swap rate ``p`` between neighbouring grades implies a latent distance
``Phi^-1(1 - p)``. Two data sources:

  - :func:`anchors_from_confusion` — a real retest confusion matrix (blind re-rating
    sessions; the gold standard, improves as rating events accumulate).
  - :func:`anchors_from_neighbours` — a pseudo-retest from embedding geometry: each
    image's "second rating" is its kNN mean label. The absolute swap level is inflated
    (kNN noise != rater noise) but the RELATIVE spacing survives the normalisation, and
    it works on the full manifest today.

Anchors are normalised to span exactly ``[1, 5]`` so they stay readable next to raw
labels and the QWK weight normalisation is unchanged.
"""

from __future__ import annotations

import torch

from silva_train.neighbours import neighbour_score_mean

N_CLASSES = 5
P_FLOOR = 1e-4  # swap rate below this is treated as "cleanly separated"
P_CEIL = 0.45  # swap rate above this is "barely distinguishable" — distance floored, never zero


def anchors_from_confusion(confusion: torch.Tensor) -> list[float]:
    """Retest confusion matrix ``[5, 5]`` (rows = first rating, cols = second) -> anchors.

    Only adjacent-pair swaps carry distance information here; a pair with no support
    (e.g. the library has no 1s) falls back to the mean of the observed distances.
    """
    conf = confusion.double()
    n_per_class = conf.sum(dim=1)

    distances: list[float | None] = []
    for i in range(N_CLASSES - 1):
        if n_per_class[i] == 0 or n_per_class[i + 1] == 0:
            distances.append(None)  # a side with no rows has an UNDEFINED swap rate, not zero
            continue
        swaps = conf[i, i + 1] + conf[i + 1, i]
        p = float((swaps / (n_per_class[i] + n_per_class[i + 1])).clamp(P_FLOOR, P_CEIL))
        normal = torch.distributions.Normal(0.0, 1.0)
        distances.append(float(normal.icdf(torch.tensor(1.0 - p))))

    observed = [d for d in distances if d is not None]
    if not observed:
        return [1.0, 2.0, 3.0, 4.0, 5.0]
    fallback = sum(observed) / len(observed)
    gaps = torch.tensor([fallback if d is None else d for d in distances])

    anchors = torch.cat([torch.zeros(1), gaps.cumsum(0)])
    anchors = 1.0 + 4.0 * anchors / anchors[-1]  # span exactly [1, 5]
    return [float(a) for a in anchors]


def anchors_from_neighbours(
    embeddings: torch.Tensor,
    scores: torch.Tensor,
    k: int = 20,
    batch_size: int = 2048,
) -> list[float]:
    """Pseudo-retest anchors from embedding geometry (no second human rating needed).

    Each row's second "rating" is the rounded mean label of its ``k`` nearest neighbours
    in embedding space; grades whose populations interleave in that space swap often and
    land close together.
    """
    k = min(k, embeddings.shape[0] - 1)  # tiny datasets: can't have more neighbours than rows
    pseudo = neighbour_score_mean(embeddings, scores.float(), k, batch_size).round().clamp(1, N_CLASSES)
    confusion = torch.zeros(N_CLASSES, N_CLASSES)
    first = scores.long().clamp(1, N_CLASSES) - 1
    second = pseudo.long() - 1
    for i, j in zip(first.tolist(), second.cpu().tolist(), strict=True):
        confusion[i, j] += 1
    return anchors_from_confusion(confusion)
