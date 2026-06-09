"""How well does the current pointwise head agree with your pairwise labels? (pairwise held-out)

Reads the pairwise comparisons you labelled in pictoria, scores both images of each pair with
the current head, and checks three things:

- DIRECTIONAL ACCURACY (a/b pairs): how often the model's higher score picks your winner,
  with a Wilson CI. Above 0.5 means the pointwise head already carries the overall preference.
- ACCURACY BY MODEL CONFIDENCE |d-score|: bucketed by how far apart the model scores the pair.
  The signal lives in the LOW-d (boundary) bucket — if accuracy there is ~0.5 the model is
  guessing on close pairs, which is exactly what absolute labels can't teach and pairwise can.
  High-d pairs the model already gets right; relabelling those teaches nothing.
- TIE CALIBRATION: on pairs you called a tie, the model's |d-score| should be SMALLER than on
  a/b pairs. If it isn't, the head pulls apart images you judged equal — a tie-loss target.

This is the pairwise-side held-out scorecard; run it each relabel round (sibling of
misclass_probe / oof_audit). Reads embeddings from pictoria, so needs the export extra:

    uv run --extra export python scripts/pairwise_eval.py [--dimension overall] [--buckets 3]
"""

from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import torch

from silva.scoring import ordinal_score_from_logits
from silva_train.metrics import _wilson_ci

DEFAULT_DB = r"E:/pictoria/server/illustration/images/.pictoria/pictoria.sqlite"
DEFAULT_CKPT = "outputs/v1_stage1_head"  # dir -> reads best.safetensors via load_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the current head against your pictoria pairwise labels.")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--dimension", default="overall", help="which pairwise dimension to evaluate")
    ap.add_argument("--buckets", type=int, default=3, help="confidence buckets (quantiles of |Δscore|)")
    args = ap.parse_args()

    from silva_train.checkpoint import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint).to(device)

    import sqlite_vec  # vec0 extension needed to read post_vectors_siglip2

    con = sqlite3.connect(args.db)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)

    emb = {r[0]: np.frombuffer(r[1], dtype=np.float32) for r in con.execute("SELECT post_id, embedding FROM post_vectors_siglip2")}
    pairs = con.execute(
        "SELECT post_a, post_b, winner FROM pairwise_annotations WHERE dimension = ? AND winner IN ('a', 'b', 'tie')",
        (args.dimension,),
    ).fetchall()
    con.close()

    missing = sum(1 for a, b, _ in pairs if a not in emb or b not in emb)
    pairs = [(a, b, w) for a, b, w in pairs if a in emb and b in emb]
    ab = [(a, b, w) for a, b, w in pairs if w in ("a", "b")]
    tie = [(a, b) for a, b, w in pairs if w == "tie"]
    print(f"dimension={args.dimension!r}  a/b pairs={len(ab)}  tie pairs={len(tie)}  (dropped {missing} pairs missing an embedding)")
    if not ab:
        print("no usable a/b pairs — nothing to evaluate")
        return

    ids = sorted({p for a, b, _ in pairs for p in (a, b)})
    x = torch.tensor(np.stack([emb[i] for i in ids]), dtype=torch.float32, device=device)
    with torch.no_grad():
        score = ordinal_score_from_logits(model(x)["logits"]).float().cpu().numpy()  # continuous 1..5
    s = dict(zip(ids, score, strict=True))

    d_ab = np.array([s[a] - s[b] for a, b, _ in ab])
    a_wins = np.array([w == "a" for _, _, w in ab])
    correct = int(((d_ab > 0) == a_wins).sum())
    acc = correct / len(ab)
    lo, hi = _wilson_ci(correct, len(ab))
    print("\n== directional accuracy (model's higher score picks your winner) ==")
    print(f"  {acc:.3f}  [{lo:.3f}, {hi:.3f}]   (n={len(ab)}, chance=0.5)")

    absd = np.abs(d_ab)
    qs = np.quantile(absd, np.linspace(0, 1, args.buckets + 1))
    print(f"\n== accuracy by model confidence |d-score| ({args.buckets} quantile buckets) ==")
    for i in range(args.buckets):
        upper_closed = i == args.buckets - 1
        m = (absd >= qs[i]) & (absd <= qs[i + 1] if upper_closed else absd < qs[i + 1])
        if not m.any():
            continue
        c = int(((d_ab[m] > 0) == a_wins[m]).sum())
        b_lo, b_hi = _wilson_ci(c, int(m.sum()))
        print(f"  |d-score| in [{qs[i]:.2f}, {qs[i + 1]:.2f}]  n={int(m.sum()):<4} acc={c / m.sum():.3f} [{b_lo:.3f}, {b_hi:.3f}]")

    print("\n== |d-score| distribution: tie vs a/b (tie should be smaller if calibrated) ==")
    print(f"  a/b pairs: median={np.median(absd):.3f}  mean={absd.mean():.3f}")
    if tie:
        d_tie = np.abs([s[a] - s[b] for a, b in tie])
        print(f"  tie pairs: median={np.median(d_tie):.3f}  mean={d_tie.mean():.3f}")
    else:
        print("  tie pairs: none")


if __name__ == "__main__":
    main()
