"""Checkpoint I/O for the training pipeline: weights + metadata, no pickle.

The best checkpoint is two files in the run directory:

  - ``best.safetensors`` — the head's ``state_dict`` (tensors only)
  - ``best.json``        — ``{"config": ..., "metrics": ...}`` (the run config + val metrics)

This replaces a single ``torch.save({...}).pt`` so loading never needs
``torch.load(weights_only=False)`` (arbitrary-code pickle execution). The published
Hub artifact is already ``model.safetensors`` via :class:`~silva.models.aesthetic.EmbeddingAestheticModel`;
this brings the local checkpoint to the same footing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from safetensors.torch import load_file, save_file

from silva.models.aesthetic import EmbeddingAestheticModel
from silva_train.config import ModelConfig

if TYPE_CHECKING:
    import torch

WEIGHTS_NAME = "best.safetensors"
META_NAME = "best.json"


def save_checkpoint(out_dir: str | Path, state_dict: dict[str, torch.Tensor], config: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Write ``best.safetensors`` (weights) and ``best.json`` (config + metrics) into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, out_dir / WEIGHTS_NAME)
    (out_dir / META_NAME).write_text(json.dumps({"config": config, "metrics": metrics}, indent=2), encoding="utf-8")


def load_checkpoint(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    """Load ``(state_dict, config, metrics)`` from a run directory or its ``best.safetensors`` file."""
    path = Path(path)
    weights = path / WEIGHTS_NAME if path.is_dir() else path
    meta = weights.with_name(META_NAME)
    info = json.loads(meta.read_text(encoding="utf-8"))
    return load_file(str(weights)), info["config"], info["metrics"]


def load_model(path: str | Path) -> EmbeddingAestheticModel:
    """Rebuild the trained head from a checkpoint, in eval mode, on CPU.

    The single seam for "checkpoint -> ready model": it threads EVERY architecture
    param from the stored config — including ``n_residual_blocks`` — into the constructor,
    so a rebuild can't silently drop the residual trunk (and load ``trunk.*`` weights into
    a model that has no slots for them). Callers move the result to their device.

    ``strict=False`` tolerates pre-calibration checkpoints whose ``state_dict`` predates the
    calibration buffers; with the params applied in full there are no unexpected keys.
    """
    state, config, _metrics = load_checkpoint(path)
    mc = ModelConfig(**config["model"])
    model = EmbeddingAestheticModel(
        embedding_dim=mc.embedding_dim,
        dropout=mc.dropout,
        hidden_dims=mc.hidden_dims,
        n_residual_blocks=mc.n_residual_blocks,
    )
    model.load_state_dict(state, strict=False)
    return model.eval()
