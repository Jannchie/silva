"""Score the whole pictoria library with the trained SILVA head and upsert into the DB.

Reads every SigLIP2 embedding from ``post_vectors_siglip2``, runs the trained head to a
0-1 personal-aesthetic score, and writes it back to ``post_aesthetic_scores`` under
``scorer='silva'`` (the name + 0-1 scale already used there). Default is a DRY RUN that
only prints the score distribution; ``--write`` commits it (DELETE+INSERT in one
transaction, so the silva rows are replaced atomically).

    uv run --extra export python scripts/score_pictoria.py            # dry run
    uv run --extra export python scripts/score_pictoria.py --write    # commit to DB
"""

from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import torch

from silva.models.aesthetic import EmbeddingAestheticModel
from silva_train.checkpoint import load_checkpoint

DEFAULT_DB = r"E:/pictoria/server/illustration/images/.pictoria/pictoria.sqlite"
SCORER = "silva"


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the pictoria library with the SILVA head.")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--checkpoint", default="outputs/v1_stage1_head")
    ap.add_argument("--write", action="store_true", help="commit to DB (default: dry run)")
    ap.add_argument("--batch", type=int, default=8192)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state, config, _ = load_checkpoint(args.checkpoint)
    mc = config["model"]
    model = EmbeddingAestheticModel(embedding_dim=mc["embedding_dim"], dropout=mc.get("dropout", 0.1), hidden_dims=mc.get("hidden_dims", []))
    model.load_state_dict(state)
    model.to(device).eval()

    import sqlite_vec  # vec0 extension needed to read post_vectors_siglip2

    con = sqlite3.connect(args.db)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)

    rows = con.execute("SELECT post_id, embedding FROM post_vectors_siglip2").fetchall()
    post_ids = np.array([r[0] for r in rows], dtype=np.int64)
    embs = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    print(f"embeddings in library: {len(post_ids)}  dim={embs.shape[1]}")

    scores = np.empty(len(post_ids), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(post_ids), args.batch):
            xb = torch.tensor(embs[i:i + args.batch], device=device)
            scores[i:i + args.batch] = model(xb)["score"].float().cpu().numpy()

    qs = [0, 10, 25, 50, 75, 90, 100]
    print(f"\nnew silva scores [0-1]:  mean={scores.mean():.3f}")
    print("  " + "  ".join(f"p{q}={np.percentile(scores, q):.3f}" for q in qs))

    existing = dict(con.execute("SELECT post_id, score FROM post_aesthetic_scores WHERE scorer=?", (SCORER,)).fetchall())
    new_ids = set(post_ids.tolist()) - set(existing)
    print(f"existing '{SCORER}' rows: {len(existing)}  ->  would update {len(post_ids) - len(new_ids)}, add {len(new_ids)}")

    if args.write:
        with con:  # single transaction: replace all silva rows atomically
            con.execute("DELETE FROM post_aesthetic_scores WHERE scorer=?", (SCORER,))
            con.executemany(
                "INSERT INTO post_aesthetic_scores(post_id, scorer, score) VALUES(?, ?, ?)",
                [(int(p), SCORER, float(s)) for p, s in zip(post_ids, scores)],
            )
        print(f"\nWROTE {len(post_ids)} '{SCORER}' scores to {args.db}")
    else:
        print("\ndry run - nothing written. pass --write to commit.")
    con.close()


if __name__ == "__main__":
    main()
