"""Does adding your pictoria pairwise labels to training actually help? (cheap probe)

The pointwise head is at the intra-rater ceiling on ABSOLUTE metrics, but it only
guesses on boundary (same-bucket) pairs — exactly what pairwise labels can teach and
absolute labels can't. This script tests whether injecting a margin-ranking loss over
your ~945 pictoria pairs improves the head's *directional* accuracy, without leaking.

Design — leave-PAIRS-out OOF (not leave-images-out):
  - The absolute supervision (manifest personal_score) is IDENTICAL for baseline and
    treatment — every image's absolute label is always in training. The only held-out
    experimental variable is each pair's WINNER relation.
  - baseline = absolute-only head. Since the absolute set never changes, it's the same
    head across folds: train ONCE, score every pair. (This is the ~0.66 we already saw.)
  - treatment = absolute + margin_pairwise_loss on the TRAIN folds' pairs. Trained once
    per fold with that fold's pairs held out; the held-out pairs are scored and the
    out-of-fold predictions aggregate to a verdict over all 945 pairs (CI ~+-0.03).
  - Pairs are bucketed by the BASELINE head's |dscore| so we can read whether treatment
    lifts accuracy specifically on the low-d boundary pairs (where baseline ~= 0.5).

Caveat: an image can appear in a held-out pair AND a train-fold pair (median 1 pair/
image, max 3), so image-level pairwise info isn't perfectly isolated — this measures
"does pairwise supervision generalise to new pairs", which is the question we care about.

    uv run --extra export python scripts/pairwise_oof_experiment.py \
        [--pairwise-weight 1.0] [--margin 0.5] [--epochs 40] [--folds 5]
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sqlite3

import numpy as np
import pandas as pd
import torch

from silva.models.aesthetic import EmbeddingAestheticModel
from silva.scoring import ordinal_score_from_logits
from silva_train.data.manifest import merge_manifests
from silva_train.losses import compute_pos_weight, margin_pairwise_loss, silva_loss
from silva_train.metrics import _wilson_ci, compute_metrics

DEFAULT_DB = r"E:/pictoria/server/illustration/images/.pictoria/pictoria.sqlite"
DEFAULT_MANIFESTS = ["data/manifest.parquet", "data/bad.parquet"]


def pair_fold(a: int, b: int, n_folds: int, salt: str = "pairfold-v1") -> int:
    """Hash an unordered pair to a fold, salted so folds don't correlate with splits."""
    key = f"{salt}:{min(a, b)}-{max(a, b)}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % n_folds


def load_absolute(manifests: list[str], split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
    df = merge_manifests([pd.read_parquet(p) for p in manifests])
    df = df[df["split"] == split].reset_index(drop=True)
    emb = torch.tensor(np.stack(df["embedding"].to_numpy()), dtype=torch.float32)
    score = torch.tensor(df["personal_score"].to_numpy(), dtype=torch.float32)
    return emb, score


@torch.no_grad()
def abs_test_metrics(model: EmbeddingAestheticModel, test_emb: torch.Tensor, test_score: torch.Tensor, device: torch.device) -> dict[str, float]:
    """Absolute held-out metrics — a guardrail that the pairwise term doesn't wreck pointwise quality."""
    preds = ordinal_score_from_logits(model(test_emb.to(device))["logits"]).float().cpu()
    return compute_metrics(preds, test_score)


def load_pairs(db: str, dimension: str) -> tuple[list[tuple[int, int, int]], dict[int, np.ndarray]]:
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
    tmap = {"a": 1, "b": -1, "tie": 0}
    pairs = [(a, b, tmap[w]) for a, b, w in raw if a in emb and b in emb]
    return pairs, emb


def fit_head(
    abs_emb: torch.Tensor,
    abs_score: torch.Tensor,
    *,
    pair_a: torch.Tensor | None = None,
    pair_b: torch.Tensor | None = None,
    pair_t: torch.Tensor | None = None,
    pairwise_weight: float = 0.0,
    margin: float = 0.5,
    epochs: int = 40,
    batch_size: int = 256,
    pair_batch: int = 256,
    lr: float = 1.5e-3,
    weight_decay: float = 0.05,
    seed: int = 42,
    device: str | None = None,
) -> EmbeddingAestheticModel:
    """Production stage-1 head + loss mix, plus an optional pairwise margin term."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = EmbeddingAestheticModel(embedding_dim=abs_emb.shape[1], dropout=0.3, hidden_dims=[1024, 512, 256], n_residual_blocks=0).to(dev)
    emb = abs_emb.to(dev)
    target = abs_score.to(dev)
    pos_weight = compute_pos_weight(target).to(dev)
    use_pair = pairwise_weight > 0 and pair_a is not None and len(pair_a) > 0
    if use_pair:
        pa, pb, pt = pair_a.to(dev), pair_b.to(dev), pair_t.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, math.ceil(len(emb) / batch_size)) * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(emb), generator=gen)
        for start in range(0, len(emb), batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) < 2:
                continue
            out = model(emb[idx])
            loss = silva_loss(out["logits"], target[idx], pos_weight=pos_weight, ranking_weight=1.0, soft_spearman_weight=0.5, qwk_weight=1.0, label_smoothing=0.2)
            if use_pair:
                pidx = torch.randint(0, len(pa), (min(pair_batch, len(pa)),), generator=gen)
                la = model(pa[pidx])["logits"]
                lb = model(pb[pidx])["logits"]
                loss = loss + pairwise_weight * margin_pairwise_loss(la, lb, pt[pidx], margin=margin)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model


@torch.no_grad()
def score_posts(model: EmbeddingAestheticModel, ids: list[int], emb: dict[int, np.ndarray], device: torch.device) -> dict[int, float]:
    x = torch.tensor(np.stack([emb[i] for i in ids]), dtype=torch.float32, device=device)
    s = ordinal_score_from_logits(model(x)["logits"]).float().cpu().numpy()
    return dict(zip(ids, s, strict=True))


def report(name: str, pairs: list[tuple[int, int, int]], dscore: np.ndarray, base_absd: np.ndarray | None, buckets: int) -> tuple[np.ndarray, float, list[float]]:
    """Print directional accuracy overall + by |dscore| bucket.

    Returns (per-pair |dscore|, overall accuracy, per-bucket accuracy) so the caller
    can tabulate a weight sweep; bucket 0 is the lowest-|dscore| (hardest) boundary pairs.
    """
    decisive = np.array([t != 0 for _, _, t in pairs])
    a_wins = np.array([t == 1 for _, _, t in pairs])
    absd = np.abs(dscore)
    d, aw = dscore[decisive], a_wins[decisive]
    correct = int(((d > 0) == aw).sum())
    n = len(d)
    lo, hi = _wilson_ci(correct, n)
    overall = correct / n
    print(f"\n== {name}: directional accuracy ==")
    print(f"  {overall:.3f}  [{lo:.3f}, {hi:.3f}]   (n={n})")
    # bucket by the BASELINE head's confidence so both models share edges
    edges_src = base_absd if base_absd is not None else absd
    qs = np.quantile(edges_src[decisive], np.linspace(0, 1, buckets + 1))
    print(f"  by |dscore| bucket (edges from {'baseline' if base_absd is not None else 'self'}):")
    bucket_accs: list[float] = []
    src = base_absd[decisive] if base_absd is not None else absd[decisive]
    for i in range(buckets):
        up = i == buckets - 1
        m = (src >= qs[i]) & (src <= qs[i + 1] if up else src < qs[i + 1])
        if m.any():
            c = int(((d[m] > 0) == aw[m]).sum())
            bucket_accs.append(c / m.sum())
            print(f"    [{qs[i]:.3f}, {qs[i + 1]:.3f}]  n={int(m.sum()):<4} acc={c / m.sum():.3f}")
        else:
            bucket_accs.append(float("nan"))
    tie = ~decisive
    if tie.any():
        print(f"  tie |dscore|: median={np.median(absd[tie]):.3f}  vs a/b median={np.median(absd[decisive]):.3f}")
    return absd, overall, bucket_accs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--manifests", nargs="+", default=DEFAULT_MANIFESTS)
    ap.add_argument("--dimension", default="overall")
    ap.add_argument("--pairwise-weight", type=float, nargs="+", default=[1.0], help="one or more weights to sweep (baseline trained once, reused)")
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--buckets", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = args.pairwise_weight
    print(f"device={device}  epochs={args.epochs}  pairwise_weight(s)={weights}  margin={args.margin}  folds={args.folds}")

    abs_emb, abs_score = load_absolute(args.manifests)
    test_emb, test_score = load_absolute(args.manifests, "test")
    pairs, pemb = load_pairs(args.db, args.dimension)
    print(f"absolute train rows={len(abs_emb)}  test rows={len(test_emb)}  pairs={len(pairs)} (a/b={sum(1 for *_, t in pairs if t)}, tie={sum(1 for *_, t in pairs if not t)})")

    all_ids = sorted({p for a, b, _ in pairs for p in (a, b)})
    folds = np.array([pair_fold(a, b, args.folds) for a, b, _ in pairs])

    # ---- baseline: absolute-only head, trained ONCE, reused across every weight ----
    print("\n[baseline] training absolute-only head ...")
    base_model = fit_head(abs_emb, abs_score, pairwise_weight=0.0, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, device=str(device))
    base_s = score_posts(base_model, all_ids, pemb, device)
    base_dscore = np.array([base_s[a] - base_s[b] for a, b, _ in pairs])
    base_absd, base_acc, base_buckets = report("BASELINE (absolute only)", pairs, base_dscore, None, args.buckets)
    base_abs = abs_test_metrics(base_model, test_emb, test_score, device)

    # variant -> (overall dir acc, hardest-bucket acc, absolute test metrics)
    summary: list[tuple[str, float, float, dict[str, float]]] = [("baseline", base_acc, base_buckets[0], base_abs)]

    # ---- treatment: absolute + pairwise, leave-pairs-out OOF, swept over weights ----
    for w in weights:
        oof_dscore = np.full(len(pairs), np.nan)
        probe_model = None  # a full-pair (fold-0 train) head, only for the absolute guardrail
        for f in range(args.folds):
            held = folds == f
            train_idx = np.where(~held)[0]
            pa = torch.tensor(np.stack([pemb[pairs[i][0]] for i in train_idx]), dtype=torch.float32)
            pb = torch.tensor(np.stack([pemb[pairs[i][1]] for i in train_idx]), dtype=torch.float32)
            pt = torch.tensor([float(pairs[i][2]) for i in train_idx], dtype=torch.float32)
            print(f"[w={w}] fold {f}: train_pairs={len(train_idx)} held={int(held.sum())} ...")
            m = fit_head(abs_emb, abs_score, pair_a=pa, pair_b=pb, pair_t=pt, pairwise_weight=w, margin=args.margin, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, device=str(device))
            if probe_model is None:
                probe_model = m
            held_ids = sorted({p for i in np.where(held)[0] for p in (pairs[i][0], pairs[i][1])})
            s = score_posts(m, held_ids, pemb, device)
            for i in np.where(held)[0]:
                a, b, _ = pairs[i]
                oof_dscore[i] = s[a] - s[b]
        _, acc, buckets = report(f"TREATMENT (absolute + pairwise w={w})", pairs, oof_dscore, base_absd, args.buckets)
        abs_m = abs_test_metrics(probe_model, test_emb, test_score, device)
        summary.append((f"w={w}", acc, buckets[0], abs_m))

    # ---- sweep summary: directional accuracy vs absolute guardrail ----
    print("\n== SWEEP SUMMARY ==")
    print(f"  {'variant':<10}{'dir_acc':>9}{'hardest':>9}   |  {'abs_mae':>8}{'abs_spr':>9}{'abs_qwk':>9}")
    for name, acc, hard, mm in summary:
        print(f"  {name:<10}{acc:>9.3f}{hard:>9.3f}   |  {mm['mae']:>8.4f}{mm['spearman']:>9.4f}{mm['qwk']:>9.4f}")
    print("  (hardest = lowest-|dscore| boundary bucket; abs_* = absolute test split guardrail: higher spr/qwk + lower mae = unharmed)")


if __name__ == "__main__":
    main()
