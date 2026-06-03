"""Does the frozen SigLIP2 embedding (and the trained head) perceive low-level degradation?

Synthetic-degradation probe. Takes a batch of decent images, applies several kinds of
low-level corruption at increasing strength (jpeg / gaussian noise / blur / pixelate),
recomputes the L2-normalised SigLIP2 embedding for each, and reports:

  - cosine(original_emb, degraded_emb)  -> how far the embedding actually moves
  - head score on original vs degraded  -> whether the trained head reads it as worse

If cosine stays ~0.99 and the score barely drops, the pooled feature is largely BLIND to
that corruption, so feeding more synthetically-degraded "score=1" anchors would teach the
head nothing — low-quality data has to be semantic (bad composition/subject), not pixel-level.

    uv run --with transformers --with pillow python scripts/degrade_probe.py
"""

from __future__ import annotations

import argparse
import io
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from transformers import AutoModel, AutoProcessor

from silva_train.checkpoint import load_model

BACKBONE = "google/siglip2-so400m-patch14-384"
DEFAULT_DB = r"E:/pictoria/server/illustration/images/.pictoria/pictoria.sqlite"
DEFAULT_IMAGES = r"E:/pictoria/server/illustration/images"
DEFAULT_CKPT = "outputs/v1_stage1_head"


# --- degradations: name -> list of (label, fn(Image) -> Image) at increasing strength ---
def _jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + np.random.normal(0, sigma, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _blur(img: Image.Image, r: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=r))


def _pixelate(img: Image.Image, factor: int) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, w // factor), max(1, h // factor)), Image.BILINEAR)
    return small.resize((w, h), Image.NEAREST)


DEGRADATIONS = {
    "jpeg":     [(f"q{q}", lambda im, q=q: _jpeg(im, q)) for q in (30, 10, 3)],
    "noise":    [(f"s{s}", lambda im, s=s: _noise(im, s)) for s in (10, 30, 80)],
    "blur":     [(f"r{r}", lambda im, r=r: _blur(im, r)) for r in (2, 5, 10)],
    "pixelate": [(f"x{f}", lambda im, f=f: _pixelate(im, f)) for f in (4, 8, 16)],
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe SigLIP/head sensitivity to low-level degradation.")
    ap.add_argument("--manifest", default="data/manifest.parquet")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--images-root", default=DEFAULT_IMAGES)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--n", type=int, default=24, help="number of decent (score>=4) images to probe")
    ap.add_argument("--min-score", type=int, default=4)
    args = ap.parse_args()

    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # decent images to corrupt (degrading already-good images shows the cleanest signal)
    df = pd.read_parquet(args.manifest, columns=["post_id", "personal_score"])
    sample = df[df["personal_score"] >= args.min_score].sample(n=args.n, random_state=42)

    # backbone + processor
    proc = AutoProcessor.from_pretrained(BACKBONE)
    backbone = AutoModel.from_pretrained(BACKBONE).to(device).eval()

    head = load_model(args.checkpoint).to(device)

    con = sqlite3.connect(args.db)
    root = Path(args.images_root)

    @torch.no_grad()
    def embed(img: Image.Image) -> torch.Tensor:
        px = proc(images=img, return_tensors="pt").to(device)
        out = backbone.get_image_features(pixel_values=px.pixel_values)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        return torch.nn.functional.normalize(feats.float(), p=2, dim=-1)[0]  # [1152]

    @torch.no_grad()
    def score(emb: torch.Tensor) -> float:
        return float(head(emb.unsqueeze(0))["score"][0])

    # accumulate per (kind, level): cosines, score deltas, degraded scores
    agg: dict[tuple[str, str], dict[str, list[float]]] = {}
    base_scores: list[float] = []
    loaded = 0
    for _, r in sample.iterrows():
        pid = int(r["post_id"])
        row = con.execute("SELECT file_path, file_name, extension FROM posts WHERE id=?", (pid,)).fetchone()
        if row is None:
            continue
        fp, fn, ext = row
        path = root / fp / f"{fn}.{ext}"
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        emb0 = embed(img)
        s0 = score(emb0)
        base_scores.append(s0)
        loaded += 1
        for kind, levels in DEGRADATIONS.items():
            for label, fn_d in levels:
                emb_d = embed(fn_d(img))
                cos = float(torch.dot(emb0, emb_d))  # both L2-normalised
                sd = score(emb_d)
                k = (kind, label)
                a = agg.setdefault(k, {"cos": [], "ds": [], "sd": []})
                a["cos"].append(cos)
                a["ds"].append(sd - s0)
                a["sd"].append(sd)
    con.close()

    print(f"\ndevice={device}  images probed={loaded}  mean original score={np.mean(base_scores):.3f}\n")
    print(f"{'degradation':<16} {'cosine':>8} {'deg_score':>10} {'d_score':>9}   interpretation")
    print("-" * 74)
    for kind, levels in DEGRADATIONS.items():
        for label, _ in levels:
            a = agg.get((kind, label))
            if not a:
                continue
            cos = float(np.mean(a["cos"]))
            sd = float(np.mean(a["sd"]))
            ds = float(np.mean(a["ds"]))
            verdict = "BLIND" if (cos > 0.97 and abs(ds) < 0.03) else ("weak" if cos > 0.9 else "sees it")
            print(f"{kind + ' ' + label:<16} {cos:>8.4f} {sd:>10.3f} {ds:>+9.3f}   {verdict}")


if __name__ == "__main__":
    main()
