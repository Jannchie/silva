"""Adapter: turn the pictoria SQLite library into a SILVA training manifest.

This is NOT part of the training library — it is the dirty work of integrating one
external data source. It reads precomputed SigLIP2 embeddings from the pictoria DB
(``post_vectors_siglip2``) joined with your personal 1~5 scores (``posts.score``,
filtered to ``score > 0``), and emits the columnar parquet manifest the training
library consumes (see ``silva.data.manifest``). The training library knows nothing
about this; to use a different source, write a different adapter that calls
``build_manifest`` / ``write_manifest``.

Reading the vec0 embedding table needs the sqlite-vec extension:

    uv sync --extra export
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from silva_train.data.manifest import build_manifest, diff_manifests, write_manifest

if TYPE_CHECKING:
    from collections.abc import Iterator

# pictoria keeps its DB at .../images/.pictoria/pictoria.sqlite (WSL view of E:\pictoria)
DEFAULT_DB = "/mnt/e/pictoria/server/illustration/images/.pictoria/pictoria.sqlite"

QUERY = """
SELECT p.id, v.embedding, p.score
FROM posts p
JOIN post_vectors_siglip2 v ON v.post_id = p.id
WHERE p.score > 0
"""

QUERY_PER_SCORE = """
SELECT p.id, v.embedding, p.score
FROM posts p
JOIN post_vectors_siglip2 v ON v.post_id = p.id
WHERE p.score = ?
ORDER BY p.id
LIMIT ?
"""


def fetch_records(db_path: str, per_score_limit: int | None = None) -> Iterator[tuple[int, list[float], int]]:
    """Yield ``(post_id, embedding, score)`` from the pictoria SQLite DB (score>0 only).

    ``per_score_limit`` caps rows per score (1..5) — handy for a small balanced
    debug set to exercise the training loop on real embeddings.
    """
    import sqlite_vec  # lazy: only the adapter needs the vec0 extension

    con = sqlite3.connect(db_path)
    try:
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)
        queries = (
            [(QUERY, ())]
            if per_score_limit is None
            else [(QUERY_PER_SCORE, (s, per_score_limit)) for s in range(1, 6)]
        )
        for sql, params in queries:
            for post_id, embedding_blob, score in con.execute(sql, params):
                embedding = np.frombuffer(embedding_blob, dtype=np.float32).tolist()
                yield post_id, embedding, score
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a SILVA manifest from the pictoria SQLite DB")
    parser.add_argument("--db", default=DEFAULT_DB, help="path to pictoria.sqlite")
    parser.add_argument("--output", default="data/manifest.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-score-limit", type=int, default=None, help="cap rows per score (1..5) for a small balanced debug set")
    parser.add_argument("--previous", default=None, help="prior manifest to diff against; defaults to --output if it already exists")
    args = parser.parse_args()

    # splits are content-keyed (by embedding), so a re-export is leakage-free on its own —
    # no need to carry over a prior split map. We still read the previous manifest, but only
    # to diff against it and report what this refresh changed.
    prev_path = Path(args.previous) if args.previous else Path(args.output)
    old_df = pd.read_parquet(prev_path) if prev_path.exists() else None

    post_ids: list[int] = []
    embeddings: list[list[float]] = []
    scores: list[int] = []
    for post_id, embedding, score in fetch_records(args.db, args.per_score_limit):
        post_ids.append(post_id)
        embeddings.append(embedding)
        scores.append(score)

    df = build_manifest(embeddings, scores, post_ids=post_ids, seed=args.seed)
    out = write_manifest(df, args.output)
    print(f"Wrote {len(df)} rows to {out}")
    print(df["split"].value_counts().to_string())
    print(df["personal_score"].value_counts().sort_index().to_string())

    if old_df is not None:
        d = diff_manifests(old_df, df)
        print(f"\n=== update vs {prev_path.name} ===")
        print(f"added: {d['n_added']}  removed: {d['n_removed']}  rescored: {d['n_rescored']}  (splits content-keyed: stable across re-export)")
        if d["n_rescored"]:
            preview = ", ".join(f"#{p}:{a}->{b}" for p, a, b in d["rescored"][:10])
            print(f"  rescored e.g. {preview}{' ...' if d['n_rescored'] > 10 else ''}")
        if d["n_added"]:
            print(f"  added ids e.g. {d['added_ids'][:10]}{' ...' if d['n_added'] > 10 else ''}")


if __name__ == "__main__":
    main()
