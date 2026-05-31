"""Score the whole pictoria library with the trained SILVA head and upsert into the DB.

Reads every SigLIP2 embedding from ``post_vectors_siglip2``, runs the trained head to its
aesthetic *latent*, then HISTOGRAM-SPECIFIES the library onto a target band distribution
(``--target``, default ``train`` = your 1~5 label distribution: few worst 1s, peak at 3,
many good 4-5s — band 0 is the worst, band 4 the best; or pass comma-separated band
fractions low->high) and writes the resulting 0~1 score under ``scorer='silva'``.

The remap is strictly rank-preserving (selection/order is untouched); it only fixes the
*distribution shape* so it matches your rating system instead of the head's raw, bimodal
latent. This is a write-time, library-wide calibration — the published HF model still emits
the raw per-image score. Default is a DRY RUN; ``--write`` commits (DELETE+INSERT in one txn).

    uv run --extra export python scripts/score_pictoria.py            # dry run
    uv run --extra export python scripts/score_pictoria.py --write    # commit to DB
"""

from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import torch

from silva.models.aesthetic import EmbeddingAestheticModel
from silva_train.calibration import histogram_specify
from silva_train.checkpoint import load_checkpoint

DEFAULT_DB = r"E:/pictoria/server/illustration/images/.pictoria/pictoria.sqlite"
SCORER = "silva"
DEFAULT_TARGET = "train"  # match your 1~5 label distribution: few worst (1s), peak at 3, many good (4-5s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the pictoria library with the SILVA head.")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--checkpoint", default="outputs/v1_stage1_head")
    ap.add_argument("--write", action="store_true", help="commit to DB (default: dry run)")
    ap.add_argument("--target", default=DEFAULT_TARGET, help="'train' = use the manifest's 1~5 score distribution; else comma-separated band fractions")
    ap.add_argument("--manifest", default="data/manifest.parquet", help="manifest to derive --target train from")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--no-smooth", dest="smooth", action="store_false", help="hard 5-band steps instead of the smooth PCHIP CDF (default: smooth)")
    args = ap.parse_args()
    if args.target == "train":
        import pandas as pd

        ps = pd.read_parquet(args.manifest, columns=["personal_score"])["personal_score"]
        target = [float((ps == k).sum()) for k in range(1, 6)]
    else:
        target = [float(x) for x in args.target.split(",")]

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

    # raw aesthetic latent (pre-sigmoid, monotone) — the ranking signal we calibrate
    latent = np.empty(len(post_ids), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(post_ids), args.batch):
            xb = torch.tensor(embs[i:i + args.batch], device=device).float()
            feat = model.trunk(model.norm(xb))
            latent[i:i + args.batch] = model.head.latent(feat).squeeze(-1).float().cpu().numpy()

    # histogram-specify the library onto the target band distribution (rank-preserving)
    scores = histogram_specify(latent, target, smooth=args.smooth).astype(np.float32)

    levels = len(target)
    seg = np.clip((scores * levels).astype(int), 0, levels - 1)
    tnorm = np.array(target) / sum(target)
    print(f"\ntarget fracs (worst->best): {[round(float(f), 3) for f in tnorm]}")
    print("actual fracs (worst->best): " + "  ".join(f"{(seg == k).mean():.3f}" for k in range(levels)))
    qs = [0, 10, 25, 50, 75, 90, 100]
    print("score percentiles: " + "  ".join(f"p{q}={np.percentile(scores, q):.3f}" for q in qs))

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
