"""Dataset that reads the parquet manifest and loads local images for SigLIP2."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from silva.data.manifest import validate_manifest

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class AestheticDataset(Dataset):
    """Yields ``{"pixel_values": Tensor[3,H,W], "score": int}`` for a given split.

    ``processor`` is a HuggingFace image processor (callable returning ``pixel_values``).
    Unreadable images are skipped by advancing to the next index.
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
        processor: Callable,
    ) -> None:
        df = validate_manifest(pd.read_parquet(manifest_path))
        self.rows = df[df["split"] == split].reset_index(drop=True)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        row = self.rows.iloc[idx]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except (OSError, UnidentifiedImageError):
            logger.warning("Skipping unreadable image: %s", row["image_path"])
            return self.__getitem__((idx + 1) % len(self))
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": pixel_values, "score": int(row["personal_score"])}
