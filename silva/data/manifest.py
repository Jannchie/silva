"""The training manifest contract — the columnar parquet shape that training consumes.

The training library is deliberately blind to where embeddings come from: it only
requires that each row carries a fixed-dimension feature vector, a 1~5 score, and a
split label. Turning an external source (a SQLite DB, a CSV, a scrape, a merge) into
this shape is the job of a script (see ``scripts/export_manifest.py``), never of the
training library. Use :func:`assign_splits` for leakage-free splits and
:func:`write_manifest` (which validates first) to persist it.

Schema
------
| column            | type           | required | notes                                |
|-------------------|----------------|----------|--------------------------------------|
| ``embedding``     | list<float>[D] | yes      | fixed-dimension feature vector       |
| ``personal_score``| int (1..5)     | yes      | your rating                          |
| ``split``         | str            | yes      | one of ``train`` / ``val`` / ``test``|
| ``post_id``       | int            | no       | provenance / split-dedup key         |
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

REQUIRED_COLUMNS = ("embedding", "personal_score", "split")
SPLIT_NAMES = ("train", "val", "test")
MIN_SCORE = 1
MAX_SCORE = 5
DEFAULT_RATIOS = (0.85, 0.10, 0.05)


def assign_splits(
    keys: Sequence[Hashable],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> list[str]:
    """Assign a split label per row, keyed by ``keys`` so the same key never leaks across splits."""
    unique = sorted(set(keys))
    n = len(unique)
    perm = np.random.default_rng(seed).permutation(n)
    n_train = round(ratios[0] * n)
    n_val = round(ratios[1] * n)

    label_of: dict[Hashable, str] = {}
    for rank, idx in enumerate(perm):
        if rank < n_train:
            label = "train"
        elif rank < n_train + n_val:
            label = "val"
        else:
            label = "test"
        label_of[unique[idx]] = label

    return [label_of[k] for k in keys]


def build_manifest(
    post_ids: Sequence[Hashable],
    embeddings: Sequence[Sequence[float]],
    scores: Sequence[int],
    seed: int = 42,
) -> pd.DataFrame:
    """Shape parallel ``(post_id, embedding, score)`` columns into a split-assigned manifest.

    ``post_id`` is the leakage-free split key. Validate/persist via :func:`write_manifest`.
    """
    df = pd.DataFrame(
        {
            "post_id": list(post_ids),
            "embedding": [list(e) for e in embeddings],
            "personal_score": [int(s) for s in scores],
        }
    )
    df["split"] = assign_splits(df["post_id"].tolist(), seed=seed)
    return df


def validate_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Assert ``df`` matches the manifest contract; return it unchanged on success."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing required column(s): {missing}")

    embedding = df["embedding"]
    if embedding.apply(lambda e: e is None).any():
        msg = "manifest has null embedding values"
        raise ValueError(msg)
    dims = {len(e) for e in embedding}
    if len(dims) != 1:
        raise ValueError(f"embedding dimension is inconsistent across rows: {sorted(dims)}")

    scores = df["personal_score"].to_numpy()
    if not np.all(np.isfinite(scores)) or not np.all(np.mod(scores, 1) == 0):
        msg = "personal_score must be integer-valued"
        raise ValueError(msg)
    if scores.min() < MIN_SCORE or scores.max() > MAX_SCORE:
        raise ValueError(f"personal_score must be within [{MIN_SCORE}, {MAX_SCORE}]")

    unknown = set(df["split"].unique()) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"manifest has unknown split label(s): {sorted(unknown)}")

    return df


def write_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    """Validate against the contract, then write the manifest parquet."""
    validate_manifest(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
