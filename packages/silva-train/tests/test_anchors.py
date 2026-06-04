import torch

from silva_train.anchors import anchors_from_confusion, anchors_from_neighbours


def _confusion_with_swaps(swaps: dict[tuple[int, int], float], n_per_class: int = 1000) -> torch.Tensor:
    """5x5 retest confusion: diagonal mass, with the given symmetric adjacent swap rates."""
    c = torch.zeros(5, 5)
    for i in range(5):
        c[i, i] = n_per_class
    for (a, b), rate in swaps.items():
        moved = rate * n_per_class
        c[a - 1, b - 1] += moved
        c[b - 1, a - 1] += moved
        c[a - 1, a - 1] -= moved
        c[b - 1, b - 1] -= moved
    return c


def test_uniform_swaps_give_equidistant_anchors():
    conf = _confusion_with_swaps({(1, 2): 0.1, (2, 3): 0.1, (3, 4): 0.1, (4, 5): 0.1})
    a = anchors_from_confusion(conf)
    gaps = [a[i + 1] - a[i] for i in range(4)]
    assert max(gaps) - min(gaps) < 1e-6


def test_high_swap_pair_lands_closer():
    conf = _confusion_with_swaps({(1, 2): 0.1, (2, 3): 0.1, (3, 4): 0.3, (4, 5): 0.1})
    a = anchors_from_confusion(conf)
    assert a[3] - a[2] < a[2] - a[1]  # 3-4 gap shrinks
    assert a[3] - a[2] < a[4] - a[3]


def test_anchors_are_monotone_and_span_one_to_five():
    conf = _confusion_with_swaps({(1, 2): 0.05, (2, 3): 0.17, (3, 4): 0.29, (4, 5): 0.18})
    a = anchors_from_confusion(conf)
    assert a[0] == 1.0
    assert a[-1] == 5.0
    assert all(a[i] < a[i + 1] for i in range(4))


def test_pair_with_no_data_falls_back_to_mean_distance():
    # no 1-rated rows at all: the 1<->2 swap rate is undefined, not zero
    conf = _confusion_with_swaps({(2, 3): 0.1, (3, 4): 0.1, (4, 5): 0.1})
    conf[0, :] = 0.0
    a = anchors_from_confusion(conf)
    assert all(a[i] < a[i + 1] for i in range(4))
    gaps = [a[i + 1] - a[i] for i in range(4)]
    assert abs(gaps[0] - gaps[1]) < 1e-6  # fallback gap = mean of observed gaps (all equal here)


def test_indistinguishable_pair_keeps_positive_distance():
    conf = _confusion_with_swaps({(1, 2): 0.1, (2, 3): 0.1, (3, 4): 0.49, (4, 5): 0.1})
    a = anchors_from_confusion(conf)
    assert a[3] - a[2] > 0  # floored, never collapses or inverts


def test_anchors_from_neighbours_clamps_k_on_tiny_datasets():
    g = torch.Generator().manual_seed(1)
    emb = torch.randn(8, 4, generator=g)
    scores = torch.tensor([1.0, 2, 3, 4, 5, 3, 4, 2])

    a = anchors_from_neighbours(emb, scores, k=20)  # k > n-1 must not crash

    assert len(a) == 5
    assert all(a[i] < a[i + 1] for i in range(4))


def test_anchors_from_neighbours_reads_embedding_overlap():
    # classes 3 and 4 share one tight region; every other class is far apart
    g = torch.Generator().manual_seed(0)
    centres = {1: 0.0, 2: 10.0, 3: 20.0, 4: 20.6, 5: 30.0}  # 3 and 4 nearly coincide
    emb, scores = [], []
    for cls, centre in centres.items():
        pts = centre + 0.5 * torch.randn(200, 8, generator=g)
        emb.append(pts)
        scores += [cls] * 200
    emb = torch.cat(emb)
    scores = torch.tensor(scores, dtype=torch.float32)

    a = anchors_from_neighbours(emb, scores, k=10)

    gaps = [a[i + 1] - a[i] for i in range(4)]
    assert min(gaps) == gaps[2]  # the 3-4 gap is the smallest
    assert all(g > 0 for g in gaps)
