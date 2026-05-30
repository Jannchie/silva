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
from typing import TYPE_CHECKING

import numpy as np

from silva.data.manifest import build_manifest, write_manifest

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


def fetch_records(db_path: str) -> Iterator[tuple[int, list[float], int]]:
    """Yield ``(post_id, embedding, score)`` from the pictoria SQLite DB (score>0 only)."""
    import sqlite_vec  # lazy: only the adapter needs the vec0 extension

    con = sqlite3.connect(db_path)
    try:
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)
        for post_id, embedding_blob, score in con.execute(QUERY):
            embedding = np.frombuffer(embedding_blob, dtype=np.float32).tolist()
            yield post_id, embedding, score
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a SILVA manifest from the pictoria SQLite DB")
    parser.add_argument("--db", default=DEFAULT_DB, help="path to pictoria.sqlite")
    parser.add_argument("--output", default="data/manifest.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    post_ids: list[int] = []
    embeddings: list[list[float]] = []
    scores: list[int] = []
    for post_id, embedding, score in fetch_records(args.db):
        post_ids.append(post_id)
        embeddings.append(embedding)
        scores.append(score)

    df = build_manifest(post_ids, embeddings, scores, seed=args.seed)
    out = write_manifest(df, args.output)
    print(f"Wrote {len(df)} rows to {out}")
    print(df["split"].value_counts().to_string())
    print(df["personal_score"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
