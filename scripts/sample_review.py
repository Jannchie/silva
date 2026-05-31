"""Sample images where the SILVA head most disagrees with your manual score, for visual review.

Uses the trained head as a SPOTLIGHT: predicts a continuous 1-5 for every scored post, then
picks the biggest manual-vs-model gaps in both directions (you-high/model-low = suspected
over-rated; you-low/model-high = suspected under-rated), plus random controls. Writes small
thumbnails into a temp dir named ``you{manual}_m{model}_{id}.jpg`` so a human (or Claude's
vision) can eyeball whether the MANUAL label is off.

    uv run --extra export python scripts/sample_review.py --out review_out --per 6
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from silva.models.aesthetic import EmbeddingAestheticModel
from silva.scoring import ordinal_score_from_logits
from silva_train.checkpoint import load_checkpoint

DEFAULT_DB = r"E:/pictoria/server/illustration/images/.pictoria/pictoria.sqlite"
IMG_BASE = r"E:/pictoria/server/illustration/images"


def resolve_path(file_path: str, file_name: str, ext: str) -> str:
    return os.path.normpath(os.path.join(IMG_BASE, file_path, f"{file_name}.{ext}"))


def thumb(src: str, dst: str, max_side: int) -> bool:
    try:
        im = Image.open(src).convert("RGB")
    except Exception as e:  # noqa: BLE001
        print(f"  skip {src}: {e}")
        return False
    im.thumbnail((max_side, max_side))
    im.save(dst, "JPEG", quality=82)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--manifest", default="data/manifest.parquet")
    ap.add_argument("--checkpoint", default="outputs/v1_stage1_head")
    ap.add_argument("--out", default="review_out")
    ap.add_argument("--per", type=int, default=6, help="images per category")
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state, config, _ = load_checkpoint(args.checkpoint)
    mc = config["model"]
    model = EmbeddingAestheticModel(embedding_dim=mc["embedding_dim"], dropout=mc.get("dropout", 0.1), hidden_dims=mc.get("hidden_dims", []))
    model.load_state_dict(state)
    model.to(device).eval()

    df = pd.read_parquet(args.manifest, columns=["post_id", "personal_score", "split", "embedding"])
    x = torch.tensor(np.stack(df["embedding"].to_numpy()), dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = ordinal_score_from_logits(model(x)["logits"]).float().cpu().numpy()
    df = df.drop(columns=["embedding"])
    df["pred"] = pred
    df["gap"] = df["pred"] - df["personal_score"]

    rng = np.random.default_rng(args.seed)
    cats: list[tuple[str, pd.DataFrame]] = []
    # suspected OVER-rated: you rated high, model says low
    cats.append(("over_you5", df[df.personal_score == 5].nsmallest(args.per, "pred")))
    cats.append(("over_you4", df[df.personal_score == 4].nsmallest(args.per, "pred")))
    # suspected UNDER-rated: you rated low, model says high
    cats.append(("under_you1", df[df.personal_score == 1].nlargest(args.per, "pred")))
    cats.append(("under_you2", df[df.personal_score == 2].nlargest(args.per, "pred")))
    # controls: random near-agreement at each band
    for s in (2, 3, 4):
        band = df[(df.personal_score == s) & (df.gap.abs() < 0.4)]
        if len(band):
            cats.append((f"ctrl_you{s}", band.iloc[rng.choice(len(band), min(args.per // 2 or 1, len(band)), replace=False)]))

    # map post_id -> file path
    con = sqlite3.connect(args.db)
    meta = {pid: (fp, fn, ext) for pid, fp, fn, ext in con.execute("SELECT id, file_path, file_name, extension FROM posts WHERE score>0")}
    con.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()

    rows = []
    for cat, sub in cats:
        for _, r in sub.iterrows():
            pid = int(r.post_id)
            if pid not in meta:
                continue
            src = resolve_path(*meta[pid])
            dst = str(out / f"{cat}__you{int(r.personal_score)}_m{r.pred:.1f}_{pid}.jpg")
            if thumb(src, dst, args.max_side):
                rows.append((cat, pid, int(r.personal_score), round(float(r.pred), 2), r.split, Path(dst).name))

    rep = pd.DataFrame(rows, columns=["cat", "post_id", "you", "model", "split", "thumb"])
    rep.to_csv(out / "index.csv", index=False)
    print(rep.to_string())
    print(f"\n{len(rep)} thumbnails -> {out}/")


if __name__ == "__main__":
    main()
