"""Render the Hugging Face model card (README.md) for the published head.

Kept as a template function so :mod:`scripts.push_to_hub` can inject the real
metrics from the checkpoint instead of hand-maintaining numbers in two places.
"""

from __future__ import annotations

from typing import Any


def _fmt(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def render_model_card(repo_id: str, backbone: str, model_cfg: dict[str, Any], metrics: dict[str, Any]) -> str:
    hidden = model_cfg.get("hidden_dims", []) or "linear probe"
    return f"""---
library_name: silva
pipeline_tag: image-classification
license: mit
base_model: {backbone}
base_model_relation: adapter
tags:
  - aesthetic
  - siglip2
  - ordinal-regression
  - image-scoring
metrics:
  - spearmanr
  - pearsonr
  - mae
model-index:
  - name: {repo_id.rsplit("/", 1)[-1]}
    results:
      - task:
          type: image-classification
          name: Personal Aesthetic Scoring
        metrics:
          - type: spearmanr
            value: {_fmt(metrics, "spearman")}
          - type: pearsonr
            value: {_fmt(metrics, "pearson")}
          - type: mae
            value: {_fmt(metrics, "mae")}
---

# SILVA — Personal Aesthetic Head

**SigLIP-based Illustration Visual Aesthetic** scorer: a small ordinal-regression
head trained on personal 1–5 ratings. It outputs a continuous aesthetic score.

> ⚠️ **This repo contains ONLY the head (~7 MB), not a full image model.** The input
> is a **1152-d [`{backbone}`]("https://huggingface.co/{backbone}") image embedding**, not
> an image. You must run the frozen SigLIP2 backbone yourself to produce that embedding —
> the backbone is *not* included in these weights.

## Architecture

`embedding[1152] → LayerNorm → Dropout → MLP trunk {hidden} → ordinal head (4 thresholds)`

Outputs:
- `score` ∈ **[0, 1]** — canonical output (mean of 4 threshold probabilities).
- `ordinal_score` ∈ **[1, 5]** — label-space score for readable comparison with raw ratings.

## Usage

```bash
pip install "git+{_REPO_URL}" transformers pillow
# (the head class lives in the GitHub repo; weights live here on the Hub)
```

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from silva.hub import HubAestheticModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Frozen backbone (NOT in this repo) -> 1152-d embedding
proc = AutoProcessor.from_pretrained("{backbone}")
backbone = AutoModel.from_pretrained("{backbone}").to(device).eval()

# 2. The published head
head = HubAestheticModel.from_pretrained("{repo_id}").to(device).eval()

img = Image.open("your_image.jpg").convert("RGB")
inputs = proc(images=img, return_tensors="pt").to(device)
with torch.no_grad():
    emb = backbone.get_image_features(**inputs)   # [1, 1152]
    out = head(emb)

print("score [0-1]:", out["score"].item())
print("score [1-5]:", out["ordinal_score"].item())
```

> ⚠️ **Embedding must match training.** These weights were trained on embeddings as
> stored upstream. `get_image_features` returns the **un-normalised** pooled feature.
> If your training embeddings were L2-normalised, add
> `emb = torch.nn.functional.normalize(emb, dim=-1)` before calling the head — otherwise
> the scores will be wrong. Verify against the pipeline that produced your training data.

## Evaluation (held-out test split)

| metric | value |
|---|---|
| Spearman ρ | {_fmt(metrics, "spearman")} |
| Pearson r | {_fmt(metrics, "pearson")} |
| MAE (1–5) | {_fmt(metrics, "mae")} |
| RMSE (1–5) | {_fmt(metrics, "rmse")} |
| QWK | {_fmt(metrics, "qwk")} |
| Top-1% precision | {_fmt(metrics, "top_1pct")} |
| Top-5% precision | {_fmt(metrics, "top_5pct")} |

## Training data & intended use

Trained on **one person's subjective 1–5 aesthetic ratings** of illustrations. It
models *that individual's* taste, **not** any universal notion of quality, and will
not transfer to other people's preferences. The rating labels and source images are
**not** released (personal preference data + image copyright). This is a personal
research artifact — use it as such.

## Limitations

- Captures a single user's taste; not a general aesthetic predictor.
- Hard-wired to the `{backbone}` embedding space; other backbones / poolings won't work.
- Frozen backbone, head-only — caps how much non-linear signal it can extract.

Source code: {_REPO_URL}
"""


_REPO_URL = "https://github.com/Jannchie/silva"
