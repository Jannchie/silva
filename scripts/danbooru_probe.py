"""Do real danbooru 'structurally broken' images land at the low end of the trained head?

Small-batch feasibility probe BEFORE injecting anything. Pulls posts tagged with
structural-failure tags (bad_anatomy etc. — the kind SigLIP can actually encode, unlike
pixel-level lowres/jpeg which degrade_probe.py showed it is blind to), downloads them,
recomputes the SAME L2-normalised SigLIP2 embedding the manifest uses, scores them with the
current head, and compares their score distribution against the manifest's own low-rated
images (test split, leakage-free).

Verdict we want: danbooru-broken scores cluster as low as — ideally lower than — the real
score<=2 images. If they don't, SigLIP doesn't read these as low quality either, and
injecting them as score=1 would just add noise.

    uv run --with transformers --with pillow --with requests python scripts/danbooru_probe.py
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from silva.models.aesthetic import EmbeddingAestheticModel

BACKBONE = "google/siglip2-so400m-patch14-384"
DEFAULT_CKPT = "outputs/v1_stage1_head/best.pt"
API = "https://danbooru.donmai.us/posts.json"
UA = "silva-aesthetic-probe/0.1 (research; contact jannchie@gmail.com)"
# structural / semantic failure tags — SigLIP-visible, unlike pixel-quality tags
STRUCTURAL_TAGS = ["bad_anatomy", "bad_proportions", "bad_hands", "poorly_drawn", "extra_digits"]


def fetch_posts(tag: str, rating: str, limit: int) -> list[dict]:
    # anonymous danbooru allows 2 tags per search: one structural tag + one rating metatag
    q = tag if not rating else f"{tag} rating:{rating}"
    r = requests.get(API, params={"tags": q, "limit": limit}, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe whether danbooru-broken images score low under the head.")
    ap.add_argument("--manifest", default="data/manifest.parquet")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--tags", nargs="+", default=STRUCTURAL_TAGS)
    ap.add_argument("--per-tag", type=int, default=30)
    ap.add_argument("--rating", default="g", help="danbooru rating filter (g/s/q/e); '' to disable")
    ap.add_argument("--save", default=None, help="optional parquet to dump {post_id,score,tag,embedding} for later injection")
    ap.add_argument("--save-images", default=None, help="dir to dump downloaded images named '{score}_{tag}_{id}.jpg' for human inspection")
    args = ap.parse_args()

    img_dir = None
    if args.save_images:
        img_dir = Path(args.save_images)
        img_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(BACKBONE)
    backbone = AutoModel.from_pretrained(BACKBONE).to(device).eval()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = ck["config"]["model"]
    head = EmbeddingAestheticModel(embedding_dim=mc["embedding_dim"], dropout=mc.get("dropout", 0.1), hidden_dims=mc.get("hidden_dims", []))
    head.load_state_dict(ck["model"])
    head.to(device).eval()

    @torch.no_grad()
    def embed(img: Image.Image) -> np.ndarray:
        px = proc(images=img, return_tensors="pt").to(device)
        out = backbone.get_image_features(pixel_values=px.pixel_values)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        return torch.nn.functional.normalize(feats.float(), p=2, dim=-1)[0].cpu().numpy()

    @torch.no_grad()
    def score_emb(emb: np.ndarray) -> float:
        t = torch.tensor(emb, device=device).unsqueeze(0)
        return float(head(t)["score"][0])

    # --- reference: how the head scores REAL low-rated manifest images (test split, no leakage) ---
    dfm = pd.read_parquet(args.manifest, columns=["split", "personal_score", "embedding"])
    ref = {}
    for s in (1, 2, 3, 4, 5):
        rows = dfm[(dfm["split"] == "test") & (dfm["personal_score"] == s)]
        if len(rows):
            embs = torch.tensor(np.stack(rows["embedding"].to_numpy()), dtype=torch.float32, device=device)
            with torch.no_grad():
                ref[s] = head(embs)["score"].cpu().numpy()
    print("\n=== reference: head score on REAL manifest images (test split) ===")
    print(f"{'true label':<12} {'n':>5} {'mean':>7} {'median':>7} {'p90':>7}")
    for s, sc in ref.items():
        print(f"score={s:<6} {len(sc):>5} {sc.mean():>7.3f} {np.median(sc):>7.3f} {np.percentile(sc, 90):>7.3f}")

    # --- probe: download danbooru broken images, embed, score ---
    print(f"\n=== danbooru structural-failure images (rating:{args.rating or 'any'}) ===")
    rows_out = []
    sess = requests.Session()
    for tag in args.tags:
        try:
            posts = fetch_posts(tag, args.rating, args.per_tag)
        except Exception as e:  # noqa: BLE001
            print(f"  {tag:<18} API error: {e}")
            continue
        scores = []
        for p in posts:
            url = p.get("large_file_url") or p.get("file_url")
            if not url or p.get("file_ext") not in ("jpg", "jpeg", "png", "webp"):
                continue
            try:
                ir = sess.get(url, headers={"User-Agent": UA}, timeout=30)
                ir.raise_for_status()
                img = Image.open(io.BytesIO(ir.content)).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            emb = embed(img)
            sc = score_emb(emb)
            scores.append(sc)
            rows_out.append({"post_id": p["id"], "score": sc, "tag": tag, "embedding": emb.tolist()})
            if img_dir is not None:  # score-prefixed filename -> sort by name = sort by head score
                img.save(img_dir / f"{sc:.3f}_{tag}_{p['id']}.jpg", quality=92)
            time.sleep(0.3)  # be polite to danbooru
        if scores:
            a = np.array(scores)
            lowfrac = float((a < 0.4).mean())
            print(f"  {tag:<18} n={len(a):<3} mean={a.mean():.3f} median={np.median(a):.3f} p90={np.percentile(a, 90):.3f}  frac(<0.4)={lowfrac:.2f}")

    if rows_out:
        allsc = np.array([r["score"] for r in rows_out])
        print(f"\n  ALL danbooru-broken  n={len(allsc)} mean={allsc.mean():.3f} median={np.median(allsc):.3f}  frac(<0.4)={(allsc < 0.4).mean():.2f}")
        ref1 = ref.get(1)
        if ref1 is not None:
            print(f"  (real score=1 test images: mean={ref1.mean():.3f} median={np.median(ref1):.3f})")
            print("\n  VERDICT:", "danbooru-broken DO score low -> injection plausible"
                  if np.median(allsc) <= np.median(ref1) + 0.05
                  else "danbooru-broken DON'T score lower than real lows -> SigLIP barely reads them as bad")
        if args.save:
            pd.DataFrame(rows_out).to_parquet(args.save)
            print(f"\n  saved {len(rows_out)} rows -> {args.save}")


if __name__ == "__main__":
    main()
