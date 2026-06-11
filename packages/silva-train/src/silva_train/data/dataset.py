"""Dataset that reads the columnar manifest and yields precomputed embeddings.

No images, no SigLIP backbone, no image processor: the training library consumes
embeddings directly. Turning images (or a DB) into embeddings is a script's job
(see ``scripts/export_manifest.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from silva_train.data.manifest import merge_manifests

if TYPE_CHECKING:
    from collections.abc import Sequence


class AestheticDataset(Dataset):
    """Yields ``{"embedding": Tensor[D], "score": int}`` for a given split.

    ``manifest_path`` may be a single parquet or a list of them: multiple manifests are
    merged on the fly (a plain concat — splits are content-keyed, so cross-source rows
    never straddle a split). This is how training ingests several sources at once.
    """

    def __init__(self, manifest_path: str | Sequence[str], split: str) -> None:
        paths = [manifest_path] if isinstance(manifest_path, str) else list(manifest_path)
        df = merge_manifests([pd.read_parquet(p) for p in paths])
        self.rows = df[df["split"] == split].reset_index(drop=True)
        # Stack the whole split into resident tensors once: np.stack on a (frozen) parquet
        # column gives a read-only/non-contiguous view, so .copy() before from_numpy.
        self.embeddings = torch.from_numpy(np.stack(self.rows["embedding"].to_numpy()).copy()).float()
        self.scores = torch.as_tensor(self.rows["personal_score"].to_numpy().copy(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        return {"embedding": self.embeddings[idx], "score": int(self.scores[idx])}
