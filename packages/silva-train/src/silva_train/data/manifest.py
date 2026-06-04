"""The training manifest contract — the columnar parquet shape that training consumes.

The training library is deliberately blind to where embeddings come from: it only
requires that each row carries a fixed-dimension feature vector, a 1~5 score, and a
split label. Turning an external source (a SQLite DB, a CSV, a scrape, a merge) into
this shape is the job of a script (see ``scripts/export_manifest.py``), never of the
training library. Use :func:`assign_splits` for leakage-free splits and
:func:`write_manifest` (which validates first) to persist it.

Splits are keyed by the *embedding content itself* (a hash of its float32 bytes),
not by ``post_id``: the same image always lands in the same split, so incremental
re-exports and relabels never leak rows across splits — and no external id is needed
for it. ``post_id`` is now purely an optional provenance label (trace a row back to
its source image); nothing in splitting, merging, or training depends on it.

Schema
------
| column            | type           | required | notes                                |
|-------------------|----------------|----------|--------------------------------------|
| ``embedding``     | list<float>[D] | yes      | fixed-dimension feature vector; split key |
| ``personal_score``| int (1..5)     | yes      | your rating                          |
| ``split``         | str            | yes      | one of ``train`` / ``val`` / ``test``|
| ``post_id``       | int            | no       | optional provenance label (unused by logic) |
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


def embedding_key(embedding: Sequence[float]) -> str:
    """A stable content id for an embedding: hash of its canonical float32 bytes.

    Casting to float32 first makes the key invariant to the parquet round-trip
    (``float32 -> tolist() -> float64 -> read back``): a float64 that originated as
    float32 is exactly representable, so it narrows back to the identical bytes. Two
    rows with the same image therefore share a key — and thus a split — with no id.
    """
    raw = np.asarray(embedding, dtype=np.float32).tobytes()
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def assign_splits(
    keys: Sequence[Hashable],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> list[str]:
    """Assign a split label per row, keyed by ``keys`` so the same key never leaks across splits.

    Each key is bucketed by a deterministic seeded hash into ``[0, 1)``, so its split is
    independent of every other key: adding or removing rows never moves an existing key
    across splits, which keeps incremental dataset updates leakage-free. Because buckets
    are per-key, the realised ratios only *approximate* ``ratios`` (exact counts would
    require global knowledge, which is what breaks incremental stability).
    """
    t_train = ratios[0]
    t_val = ratios[0] + ratios[1]
    labels: list[str] = []
    for key in keys:
        u = _split_point(key, seed)
        labels.append("train" if u < t_train else "val" if u < t_val else "test")
    return labels


def assign_folds(
    keys: Sequence[Hashable],
    n_folds: int,
    seed: int = 42,
) -> list[int]:
    """Assign a CV fold per row, keyed like :func:`assign_splits` so folds never drift.

    Each key maps independently to ``[0, n_folds)`` via the same deterministic seeded
    hash: re-exports and relabels keep every existing row in its fold, so out-of-fold
    predictions stay comparable across manifest updates. Balance is approximate for the
    same reason split ratios are (per-key hashing has no global view).

    The key domain is salted (``fold:``) so folds are independent of splits even at the
    same seed — otherwise the train split occupies one band of the hash line and the
    last fold collapses to a sliver of it.
    """
    return [int(_split_point(f"fold:{key}", seed) * n_folds) for key in keys]


def build_manifest(
    embeddings: Sequence[Sequence[float]],
    scores: Sequence[int],
    post_ids: Sequence[Hashable] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Shape parallel ``(embedding, score[, post_id])`` columns into a split-assigned manifest.

    Splits are keyed by each row's embedding *content* (:func:`embedding_key`), so the same
    image always lands in the same split — incremental re-exports and relabels stay
    leakage-free with no prior-split map and no id. ``post_ids`` is optional provenance: when
    given it's carried as a ``post_id`` column, otherwise the manifest simply omits it.
    Validate/persist via :func:`write_manifest`.
    """
    data = {
        "embedding": [list(e) for e in embeddings],
        "personal_score": [int(s) for s in scores],
    }
    if post_ids is not None:
        data["post_id"] = list(post_ids)
    df = pd.DataFrame(data)
    df["split"] = assign_splits([embedding_key(e) for e in df["embedding"]], seed=seed)
    return df


def merge_manifests(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate several manifests into one, validating the contract and shared dim.

    Each frame keeps its own ``split`` labels: because splits are content-keyed, the same
    image carries the same split everywhere, so a plain concat can never straddle splits
    (cross-source images simply don't collide). The optional ``post_id`` column is kept only
    when *every* frame has it, otherwise dropped to avoid NaN ids from mixed provenance.
    """
    frames = [validate_manifest(f) for f in frames]
    dims = {len(e) for f in frames for e in f["embedding"]}
    if len(dims) > 1:
        raise ValueError(f"embedding dimension differs across manifests: {sorted(dims)}")
    merged = pd.concat(frames, ignore_index=True)
    if not all("post_id" in f.columns for f in frames) and "post_id" in merged.columns:
        merged = merged.drop(columns=["post_id"])
    return validate_manifest(merged)


def diff_manifests(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Summarise what changed between two manifests, keyed by embedding content.

    Identity is the embedding (:func:`embedding_key`), so a relabel — same image, new score
    — reads as *rescored* rather than add+remove. The reported ids use ``post_id`` when the
    frames carry it (friendly for cross-referencing in pictoria), else the content hash.
    Returns added / removed / rescored lists plus the new split and score distributions.
    """

    def index(df: pd.DataFrame) -> dict[str, tuple[Hashable, int]]:
        has_id = "post_id" in df.columns
        out: dict[str, tuple[Hashable, int]] = {}
        for i, (emb, score) in enumerate(zip(df["embedding"], df["personal_score"], strict=True)):
            ident = int(df["post_id"].iloc[i]) if has_id else None
            out[embedding_key(emb)] = (ident, int(score))
        return out

    old_idx, new_idx = index(old), index(new)
    old_keys, new_keys = set(old_idx), set(new_idx)
    ident = lambda idx, k: idx[k][0] if idx[k][0] is not None else k  # noqa: E731
    added = sorted(ident(new_idx, k) for k in new_keys - old_keys)
    removed = sorted(ident(old_idx, k) for k in old_keys - new_keys)
    rescored = sorted(
        (ident(new_idx, k), old_idx[k][1], new_idx[k][1]) for k in old_keys & new_keys if old_idx[k][1] != new_idx[k][1]
    )
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
