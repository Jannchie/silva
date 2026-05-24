import torch

from silva.models.ordinal_head import OrdinalHead


def test_thresholds_strictly_increasing_at_init():
    head = OrdinalHead(hidden_size=8)
    thr = head.get_thresholds()
    assert thr.shape == (4,)
    assert torch.all(thr[1:] > thr[:-1])


def test_thresholds_strictly_increasing_after_perturbation():
    head = OrdinalHead(hidden_size=8)
    with torch.no_grad():
        head.raw_deltas.copy_(torch.randn(4) * 5)
    thr = head.get_thresholds()
    assert torch.all(thr[1:] > thr[:-1])


def test_forward_output_shape():
    head = OrdinalHead(hidden_size=8)
    logits = head(torch.randn(5, 8))
    assert logits.shape == (5, 4)
