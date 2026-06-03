"""Exponential moving average of model weights, as one module.

The training loop keeps a shadow copy of the weights, decayed toward the live weights each
step; for eval (and for the saved checkpoint) it swaps the shadow in and the live weights
back out afterwards. Folding init / update / swap into one module keeps that three-part
state machine — and the swap-back, which must run even if eval raises — in one place.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

    from torch import nn


class EmaShadow:
    """Shadow weights tracking ``model`` by exponential moving average with ``decay``."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        """Decay the shadow one step toward ``model``'s current weights."""
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)

    @contextmanager
    def swapped(self, model: nn.Module) -> Generator[None]:
        """Load the shadow weights into ``model`` for the block, restoring the live weights on exit.

        The live weights are restored even if the block raises.
        """
        live = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(live[k].dtype) for k, v in self.shadow.items()})
        try:
            yield
        finally:
            model.load_state_dict(live)
