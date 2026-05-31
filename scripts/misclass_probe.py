"""Where does the trained head most disagree with your labels? (mislabel / hard-case finder)

Predicts every manifest row with the current head and surfaces the biggest label-vs-prediction
gaps. TRAIN-split rows with a huge gap are prime suspects for LABEL NOISE: the head has the
capacity to memorise the train set, so if it still can't fit a row, that row's score likely
conflicts with many visually-similar rows (a slip when you rated it). Cross-check the printed
post_ids in pictoria to decide: relabel the image, or accept the model is wrong.

    uv run python scripts/misclass_probe.py [--min-gap 2.0] [--top 25]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from silva.models.aesthetic import EmbeddingAestheticModel
from silva.scoring import ordinal_score_from_logits

DEFAULT_CKPT = "outputs/v1_stage1_head"  # dir -> reads best.safetensors via load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description="Find images where the head most disagrees with your labels.")
    ap.add_argument("--manifest", default="data/manifest.parquet")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--min-gap", type=float, default=2.0, help="|pred-true| threshold for 'badly misscored'")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    from silva_train.checkpoint import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state, config, _ = load_checkpoint(args.checkpoint)
    mc = config["model"]
    model = EmbeddingAestheticModel(embedding_dim=mc["embedding_dim"], dropout=mc.get("dropout", 0.1), hidden_dims=mc.get("hidden_dims", []))
    model.load_state_dict(state)
    model.to(device).eval()

    df = pd.read_parquet(args.manifest, columns=["post_id", "personal_score", "split", "embedding"])
    x = torch.tensor(np.stack(df["embedding"].to_numpy()), dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = ordinal_score_from_logits(model(x)["logits"]).float().cpu().numpy()  # continuous 1..5
    df = df.drop(columns=["embedding"])
    df["pred"] = pred
    df["gap"] = df["pred"] - df["personal_score"]
    df["absgap"] = df["gap"].abs()
    df["pred_r"] = np.clip(np.round(df["pred"]), 1, 5).astype(int)

    print("=== confusion (rows=your label, cols=model rounded) - ALL splits ===")
    print(pd.crosstab(df["personal_score"], df["pred_r"], margins=True).to_string())

    print(f"\n=== |pred - your label| >= {args.min_gap} by split ===")
    for s in ("train", "val", "test"):
        d = df[df.split == s]
        bad = d[d.absgap >= args.min_gap]
        over = int((d.gap >= args.min_gap).sum())    # you rated LOW, model says HIGH
        under = int((d.gap <= -args.min_gap).sum())  # you rated HIGH, model says LOW
        print(f"  {s:<5} n={len(d):<6} bad={len(bad):<5} ({len(bad) / max(1, len(d)):.1%})  you-low/model-high={over}  you-high/model-low={under}")

    print(f"\n=== TRAIN label-noise suspects - you rated LOW but model says HIGH (gap >= +{args.min_gap}) ===")
    sus = df[(df.split == "train") & (df.gap >= args.min_gap)].sort_values("gap", ascending=False)
    for _, r in sus.head(args.top).iterrows():
        print(f"  post_id={int(r.post_id):<10} you={int(r.personal_score)} model={r.pred:.2f}")

    print(f"\n=== TRAIN label-noise suspects - you rated HIGH but model says LOW (gap <= -{args.min_gap}) ===")
    sus2 = df[(df.split == "train") & (df.gap <= -args.min_gap)].sort_values("gap")
    for _, r in sus2.head(args.top).iterrows():
        print(f"  post_id={int(r.post_id):<10} you={int(r.personal_score)} model={r.pred:.2f}")


if __name__ == "__main__":
    main()
