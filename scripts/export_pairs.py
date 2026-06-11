"""Export your pictoria pairwise comparisons to the PairDataset parquet schema.

Reads ``pairwise_annotations`` (post_a/post_b/winner/dimension) joined to the SigLIP2
vectors in ``post_vectors_siglip2``, and writes one row per decided pair with columns
``embedding_a`` / ``embedding_b`` (the two sides' vectors) and ``target`` (winner mapped
``a -> +1``, ``b -> -1``, ``tie -> 0``). This is the explicit-preference signal the
production training loop consumes when ``train.pairwise_weight > 0`` and
``data.pair_manifest_path`` is set (the margin-ranking term). Needs the export extra
for the sqlite_vec loader:

    uv run --extra export python scripts/export_pairs.py [--dimension overall] [--out data/pairs.parquet]
"""

from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd

DEFAULT_DB = r"E:/pictoria/server/illustration/images/.pictoria/pictoria.sqlite"

_WINNER_TO_TARGET = {"a": 1, "b": -1, "tie": 0}


def load_pairs(db: str, dimension: str) -> list[dict]:
    """Decided pairs for ``dimension`` as PairDataset rows (embedding_a, embedding_b, target)."""
    import sqlite_vec

    con = sqlite3.connect(db)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    emb = {r[0]: np.frombuffer(r[1], dtype=np.float32) for r in con.execute("SELECT post_id, embedding FROM post_vectors_siglip2")}
    raw = con.execute(
        "SELECT post_a, post_b, winner FROM pairwise_annotations WHERE dimension = ? AND winner IN ('a', 'b', 'tie')",
        (dimension,),
    ).fetchall()
    con.close()
    rows: list[dict] = []
    for a, b, w in raw:
        if a in emb and b in emb:
            rows.append({"embedding_a": emb[a].tolist(), "embedding_b": emb[b].tolist(), "target": _WINNER_TO_TARGET[w]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Export pictoria pairwise comparisons to a PairDataset parquet.")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--dimension", default="overall")
    ap.add_argument("--out", default="data/pairs.parquet")
    args = ap.parse_args()

    rows = load_pairs(args.db, args.dimension)
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    n_tie = int((df["target"] == 0).sum()) if len(df) else 0
    print(f"wrote {len(df)} pairs (a/b={len(df) - n_tie}, tie={n_tie}) for dimension={args.dimension!r} -> {args.out}")


if __name__ == "__main__":
    main()
