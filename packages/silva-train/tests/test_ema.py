import pytest
import torch
from torch import nn

from silva_train.ema import EmaShadow


def _model() -> nn.Linear:
    m = nn.Linear(3, 1)
    with torch.no_grad():
        m.weight.fill_(1.0)
        m.bias.fill_(0.0)
    return m


def test_update_is_decayed_average_of_live_weights():
    m = _model()
    ema = EmaShadow(m, decay=0.9)
    with torch.no_grad():
        m.weight.fill_(2.0)  # live weight moves

    ema.update(m)

    assert torch.allclose(ema.shadow["weight"], torch.full((1, 3), 1.1))  # 0.9*1.0 + 0.1*2.0


def test_swapped_loads_shadow_then_restores_live():
    m = _model()
    ema = EmaShadow(m, decay=0.9)
    with torch.no_grad():
        m.weight.fill_(2.0)
    ema.update(m)  # shadow=1.1, live=2.0

    with ema.swapped(m):
        assert torch.allclose(m.weight, torch.full((1, 3), 1.1))  # shadow swapped in for eval

    assert torch.allclose(m.weight, torch.full((1, 3), 2.0))  # live restored after


def test_swapped_restores_live_even_on_exception():
    m = _model()
    ema = EmaShadow(m, decay=0.9)
    with torch.no_grad():
        m.weight.fill_(2.0)
    ema.update(m)

    msg = "eval blew up"
    with pytest.raises(RuntimeError), ema.swapped(m):
        raise RuntimeError(msg)

    assert torch.allclose(m.weight, torch.full((1, 3), 2.0))  # restored despite the exception
