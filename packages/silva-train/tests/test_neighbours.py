import torch

from silva_train.neighbours import nearest_neighbours, neighbour_score_mean


def test_nearest_neighbours_excludes_self_and_returns_idx_and_cosine():
    # near-duplicate pairs; each row's single nearest (self excluded) is its pair-mate
    emb = torch.tensor([[1.0, 0.0], [1.0, 0.02], [0.0, 1.0], [0.02, 1.0]])

    idx, cos = nearest_neighbours(emb, k=1)

    assert idx[:, 0].tolist() == [1, 0, 3, 2]
    assert torch.all(cos[:, 0] > 0.9)  # pair-mates are near-identical -> high cosine
    assert torch.all(cos <= 1.0 + 1e-5)


def test_nearest_neighbours_orders_by_descending_cosine():
    emb = torch.tensor([[1.0, 0.0], [1.0, 0.01], [1.0, 0.5], [0.0, 1.0]])

    idx, cos = nearest_neighbours(emb, k=2)

    assert idx[0].tolist() == [1, 2]  # row1 (closest) then row2; row3 (orthogonal) excluded
    assert cos[0, 0] >= cos[0, 1]


def test_neighbour_score_mean_excludes_self_and_takes_nearest():
    # two near-duplicate pairs; with k=1 each row's nearest (excluding itself) is its pair-mate
    emb = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
    scores = torch.tensor([5.0, 1.0, 4.0, 2.0])

    out = neighbour_score_mean(emb, scores, k=1)

    assert torch.allclose(out, torch.tensor([1.0, 5.0, 2.0, 4.0]))


def test_neighbour_score_mean_averages_k_nearest_and_excludes_far_rows():
    emb = torch.tensor([[1.0, 0.0], [1.0, 0.01], [1.0, 0.02], [0.0, 1.0]])
    scores = torch.tensor([0.0, 2.0, 4.0, 100.0])

    out = neighbour_score_mean(emb, scores, k=2)

    assert torch.allclose(out[0], torch.tensor(3.0))  # mean(2, 4); the far row (100) is not a neighbour
