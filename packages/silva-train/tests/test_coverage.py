import torch

from silva_train.coverage import DomainReference, calibrate_threshold, knn_distance_to_reference


def _cluster(n: int, d: int, seed: int, spread: float = 0.05) -> torch.Tensor:
    """A tight cluster around one direction — a stand-in for in-domain embeddings."""
    g = torch.Generator().manual_seed(seed)
    centre = torch.zeros(d)
    centre[0] = 1.0
    return centre + spread * torch.randn(n, d, generator=g)


def test_query_identical_to_reference_row_has_zero_distance():
    ref = torch.eye(4)
    query = torch.eye(4)[:1]

    dist = knn_distance_to_reference(query, ref, k=1)

    assert torch.allclose(dist, torch.zeros(1), atol=1e-5)


def test_orthogonal_query_has_distance_one():
    ref = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.01, 0.0]])
    query = torch.tensor([[0.0, 0.0, 1.0]])

    dist = knn_distance_to_reference(query, ref, k=2)

    assert torch.allclose(dist, torch.ones(1), atol=1e-2)


def test_distance_averages_over_k_nearest_only():
    # two near rows (cos~1) and one orthogonal: k=2 must ignore the orthogonal row
    ref = torch.tensor([[1.0, 0.0], [1.0, 0.02], [0.0, 1.0]])
    query = torch.tensor([[1.0, 0.01]])

    near = knn_distance_to_reference(query, ref, k=2)
    with_far = knn_distance_to_reference(query, ref, k=3)

    assert near < 0.01
    assert with_far > near  # pulling in the orthogonal row must raise the mean distance


def test_blocked_computation_matches_single_block():
    g = torch.Generator().manual_seed(0)
    ref = torch.randn(50, 8, generator=g)
    query = torch.randn(23, 8, generator=g)

    a = knn_distance_to_reference(query, ref, k=5, batch_size=7)
    b = knn_distance_to_reference(query, ref, k=5, batch_size=1000)

    assert torch.allclose(a, b, atol=1e-6)


def test_calibrate_threshold_is_small_for_tight_cluster_and_monotonic_in_quantile():
    ref = _cluster(200, 16, seed=1)

    t50 = calibrate_threshold(ref, k=4, quantile=0.5)
    t99 = calibrate_threshold(ref, k=4, quantile=0.99)

    assert 0.0 < t50 <= t99 < 0.5


def test_domain_reference_separates_in_domain_from_far_queries():
    ref = DomainReference.fit(_cluster(300, 16, seed=2), k=4, quantile=0.99)
    in_dom = _cluster(50, 16, seed=3)
    far = torch.zeros(5, 16)
    far[:, 1] = 1.0  # orthogonal to the cluster axis

    assert ref.in_domain(in_dom).float().mean() > 0.9
    assert not ref.in_domain(far).any()


def test_fit_subsamples_reference_to_max_rows():
    ref = DomainReference.fit(_cluster(500, 8, seed=4), k=2, max_rows=64)

    assert ref.embeddings.shape[0] == 64


def test_save_load_roundtrip_preserves_verdicts(tmp_path):
    ref = DomainReference.fit(_cluster(200, 8, seed=5), k=3, quantile=0.95)
    query = torch.cat([_cluster(10, 8, seed=6), torch.full((3, 8), -1.0)])
    path = tmp_path / "domain.safetensors"

    ref.save(path)
    loaded = DomainReference.load(path)

    assert loaded.k == ref.k
    assert abs(loaded.threshold - ref.threshold) < 1e-6
    assert torch.equal(loaded.in_domain(query), ref.in_domain(query))
