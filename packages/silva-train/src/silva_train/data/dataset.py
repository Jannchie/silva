"""Dataset that reads the columnar manifest and yields precomputed embeddings.

No images, no SigLIP backbone, no image processor: the training library consumes
embeddings directly. Turning images (or a DB) into embeddings is a script's job
(see ``scripts/export_manifest.py``).
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from silva_train.data.manifest import validate_manifest


class AestheticDataset(Dataset):
    """Yields ``{"embedding": Tensor[D], "score": int}`` for a given split."""

    def __init__(self, manifest_path: str, split: str) -> None:
        df = validate_manifest(pd.read_parquet(manifest_path))
        self.rows = df[df["split"] == split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        row = self.rows.iloc[idx]
        embedding = torch.tensor(row["embedding"], dtype=torch.float32)
        return {"embedding": embedding, "score": int(row["personal_score"])}
