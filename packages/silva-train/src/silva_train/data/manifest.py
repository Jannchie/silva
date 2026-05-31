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

import hashlib
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


def _split_point(key: Hashable, seed: int) -> float:
    """Map a key to a stable point in ``[0, 1)`` via a seeded hash (process-independent)."""
    digest = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def assign_splits(
    keys: Sequence[Hashable],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
    existing: dict[Hashable, str] | None = None,
) -> list[str]:
    """Assign a split label per row, keyed by ``keys`` so the same key never leaks across splits.

    Each key is bucketed by a deterministic seeded hash into ``[0, 1)``, so its split is
    independent of every other key: adding or removing rows never moves an existing key
    across splits, which keeps incremental dataset updates leakage-free. ``existing`` pins
    keys to a previously-assigned split (overriding the hash); unknown keys are hashed.
    Because buckets are per-key, the realised ratios only *approximate* ``ratios`` (exact
    counts would require global knowledge, which is what breaks incremental stability).
    """
    existing = existing or {}
    t_train = ratios[0]
    t_val = ratios[0] + ratios[1]
    labels: list[str] = []
    for key in keys:
        if key in existing:
            labels.append(existing[key])
            continue
        u = _split_point(key, seed)
        labels.append("train" if u < t_train else "val" if u < t_val else "test")
    return labels


def build_manifest(
    post_ids: Sequence[Hashable],
    embeddings: Sequence[Sequence[float]],
    scores: Sequence[int],
    seed: int = 42,
    existing: dict[Hashable, str] | None = None,
) -> pd.DataFrame:
    """Shape parallel ``(post_id, embedding, score)`` columns into a split-assigned manifest.

    ``post_id`` is the leakage-free split key. Pass ``existing`` (a ``post_id -> split`` map
    from a previous manifest) to pin known posts to their prior split on an incremental
    rebuild; new posts are hashed in. Validate/persist via :func:`write_manifest`.
    """
    df = pd.DataFrame(
        {
            "post_id": list(post_ids),
            "embedding": [list(e) for e in embeddings],
            "personal_score": [int(s) for s in scores],
        }
    )
    df["split"] = assign_splits(df["post_id"].tolist(), seed=seed, existing=existing)
    return df


def diff_manifests(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Summarise what changed between two manifests, keyed by ``post_id``.

    Returns added / removed / rescored post lists plus the new split and score
    distributions — the payload an update report prints. Both frames need ``post_id``.
    """
    old_score = {int(k): int(v) for k, v in zip(old["post_id"], old["personal_score"], strict=True)}
    new_score = {int(k): int(v) for k, v in zip(new["post_id"], new["personal_score"], strict=True)}
    old_ids, new_ids = set(old_score), set(new_score)
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    rescored = [(pid, old_score[pid], new_score[pid]) for pid in sorted(old_ids & new_ids) if old_score[pid] != new_score[pid]]
    return {
        "n_added": len(added),
        "added_ids": added,
        "n_removed": len(removed),
        "removed_ids": removed,
        "n_rescored": len(rescored),
        "rescored": rescored,
        "split_counts": {k: int(v) for k, v in new["split"].value_counts().items()},
        "score_counts": {int(k): int(v) for k, v in new["personal_score"].value_counts().sort_index().items()},
    }


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
