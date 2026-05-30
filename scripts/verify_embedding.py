"""Pre-publish check: do manifest embeddings match the backbone's get_image_features?

Recomputes embeddings for a few posts straight from their source images via
``google/siglip2-so400m-patch16-384`` and compares them to the vectors stored in
the manifest. If cosine ~1.0 and the norms match, the model card's usage recipe is
correct; if not, the published head would score downloaders' embeddings wrongly.

Needs the source images + a backbone (not training-lib dependencies):

    uv run --with transformers --with pillow python scripts/verify_embedding.py
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

BACKBONE = "google/siglip2-so400m-patch14-384"  # MUST match pictoria ai/siglip_embed.py (patch14, not patch16)
DEFAULT_DB = r"E:/pictoria/server/illustration/images/.pictoria/pictoria.sqlite"
DEFAULT_IMAGES = r"E:/pictoria/server/illustration/images"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify manifest embeddings match the SigLIP2 backbone.")
    ap.add_argument("--manifest", default="data/manifest.parquet")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--images-root", default=DEFAULT_IMAGES)
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest)
    samples = pd.concat([df[df["personal_score"] == s].head(1) for s in sorted(df["personal_score"].unique())])

    con = sqlite3.connect(args.db)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(BACKBONE)
    model = AutoModel.from_pretrained(BACKBONE).to(device).eval()
    root = Path(args.images_root)

    print(f"{'post_id':>8} {'score':>5} {'dim':>5} {'cosine':>8} {'norm_stored':>12} {'norm_recomp':>12}")
    cosines = []
    for _, r in samples.iterrows():
        pid = int(r["post_id"])
        stored = np.asarray(r["embedding"], dtype=np.float32)
        fp, fn, ext = con.execute("SELECT file_path, file_name, extension FROM posts WHERE id=?", (pid,)).fetchone()
        img = Image.open(root / fp / f"{fn}.{ext}").convert("RGB")
        inputs = proc(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            # Exactly as pictoria ai/siglip_embed.py: get_image_features(pixel_values=...) -> (N, 1152), then L2-normalise.
            out = model.get_image_features(pixel_values=inputs.pixel_values)
            feats = out.pooler_output if hasattr(out, "pooler_output") else out
            feats = torch.nn.functional.normalize(feats.float(), p=2, dim=-1)
        emb = feats[0].cpu().numpy()
        if emb.shape != stored.shape:
            print(f"{pid:>8} {int(r['personal_score']):>5}  SHAPE MISMATCH: stored={stored.shape} feats={tuple(feats.shape)}")
            continue
        c = cosine(stored, emb)
        cosines.append(c)
        print(f"{pid:>8} {int(r['personal_score']):>5} {emb.shape[0]:>5} {c:>8.4f} {np.linalg.norm(stored):>12.3f} {np.linalg.norm(emb):>12.3f}")
    con.close()

    mean_cos = float(np.mean(cosines))
    print(f"\nmean cosine = {mean_cos:.4f}")
    if mean_cos > 0.999:
        print("OK: stored embeddings ARE get_image_features output. Model card usage is correct.")
    elif mean_cos > 0.95:
        print("CLOSE: same direction but check norms — a normalization step may be needed in the card.")
    else:
        print("MISMATCH: stored embeddings are NOT this backbone/processor's get_image_features. Do NOT publish as-is.")


if __name__ == "__main__":
    main()
