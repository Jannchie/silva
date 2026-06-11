"""Explicit-preference pair dataset for the margin-ranking loss.

Where :class:`~silva_train.data.dataset.AestheticDataset` carries the absolute 1~5
labels, this carries the human-judged COMPARISONS: each row is two embeddings plus a
``target`` of ``+1`` (a preferred), ``-1`` (b preferred) or ``0`` (tie). It is the only
signal that can teach same-bucket (boundary) ordering the absolute labels can't express.
Turning pictoria's ``pairwise_annotations`` into this parquet is a script's job
(see ``scripts/export_pairs.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

_VALID_TARGETS = (-1, 0, 1)


class PairDataset:
    """Pre-stacked preference pairs: ``self.emb_a``, ``self.emb_b`` ``[P, D]`` and ``self.targets`` ``[P]``.

    Required parquet columns: ``embedding_a``, ``embedding_b`` (equal-length float vectors)
    and ``target`` (int in {-1, 0, +1}). Unlike the absolute dataset this is not a torch
    ``Dataset`` (the train loop samples mini-batches directly via :meth:`sample`).
    """

    def __init__(self, manifest_path: str) -> None:
        df = pd.read_parquet(manifest_path)
        for col in ("embedding_a", "embedding_b", "target"):
            if col not in df.columns:
                raise ValueError(f"pair manifest missing required column: {col!r}")
        targets = df["target"].to_numpy()
        if not np.all(np.isin(targets, _VALID_TARGETS)):
            raise ValueError(f"pair target must be one of {_VALID_TARGETS}; got values {sorted(set(targets.tolist()))}")
        self.emb_a = torch.from_numpy(np.stack(df["embedding_a"].to_numpy()).copy()).float()
        self.emb_b = torch.from_numpy(np.stack(df["embedding_b"].to_numpy()).copy()).float()
        self.targets = torch.as_tensor(targets.copy(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def sample(self, n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A random mini-batch ``(emb_a, emb_b, targets)`` of ``min(n, len)`` pairs, sampled WITHOUT replacement.

        Indices are drawn on CPU (the ``generator`` is a CPU generator) and used to index the
        stacked tensors wherever they live, so the returned slices follow the dataset's device.
        """
        k = min(n, len(self.targets))
        idx = torch.randperm(len(self.targets), generator=generator)[:k].to(self.targets.device)
        return self.emb_a[idx], self.emb_b[idx], self.targets[idx]
