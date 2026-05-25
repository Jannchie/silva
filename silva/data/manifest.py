"""The training manifest contract — the parquet shape that training consumes.

Producing the manifest is intentionally open: any data source (a database, a CSV,
a scrape, or a merge of several) just needs to emit a parquet with this schema.
Use :func:`assign_splits` to add leakage-free splits and :func:`write_manifest`
(which validates first) to persist it.

Schema
------
| column           | type        | required | notes                                  |
|------------------|-------------|----------|----------------------------------------|
| ``image_path``   | str         | yes      | local path to the image                |
| ``personal_score``| int (1..5) | yes      | your rating                            |
| ``split``        | str         | yes      | one of ``train`` / ``val`` / ``test``  |
| ``scorer_a``     | float       | no       | external scorer A (stored for v2)      |
| ``scorer_b``     | float       | no       | external scorer B (stored for v2)      |
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

REQUIRED_COLUMNS = ("image_path", "personal_score", "split")
OPTIONAL_COLUMNS = ("scorer_a", "scorer_b")
SPLIT_NAMES = ("train", "val", "test")
MIN_SCORE = 1
MAX_SCORE = 5
DEFAULT_RATIOS = (0.85, 0.10, 0.05)


def assign_splits(
    paths: list[str],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> list[str]:
    """Assign a split label to each row, keyed by unique ``image_path`` (no leakage)."""
    unique = sorted(set(paths))
    n = len(unique)
    perm = np.random.default_rng(seed).permutation(n)
    n_train = round(ratios[0] * n)
    n_val = round(ratios[1] * n)

    label_of: dict[str, str] = {}
    for rank, idx in enumerate(perm):
        if rank < n_train:
            label = "train"
        elif rank < n_train + n_val:
            label = "val"
        else:
            label = "test"
        label_of[unique[idx]] = label

    return [label_of[p] for p in paths]


def validate_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Assert ``df`` matches the manifest contract; return it unchanged on success."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing required column(s): {missing}")

    if df["image_path"].isna().any():
        raise ValueError("manifest has null image_path values")

    scores = df["personal_score"].to_numpy()
    if not np.all(np.isfinite(scores)) or not np.all(np.mod(scores, 1) == 0):
        raise ValueError("personal_score must be integer-valued")
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
