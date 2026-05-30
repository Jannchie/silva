"""Render the Hugging Face model card (README.md) for the published head.

Kept as a template function so :mod:`scripts.push_to_hub` can inject the real
metrics from the checkpoint instead of hand-maintaining numbers in two places.
"""

from __future__ import annotations

from typing import Any

REPO_URL = "https://github.com/Jannchie/silva"


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

Scores an illustration 1–5 by **one specific person's** taste — not a universal quality
model, so it won't match anyone else's preferences.

**Only the head ships here (~7 MB), not an image model.** Feed it a 1152-d `{backbone}`
image embedding (you run that backbone yourself) and it returns a score.

## Quickstart

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from silva.hub import HubAestheticModel  # pip install "git+{REPO_URL}"

dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoProcessor.from_pretrained("{backbone}")
backbone = AutoModel.from_pretrained("{backbone}").to(dev).eval()
head = HubAestheticModel.from_pretrained("{repo_id}").to(dev).eval()

img = Image.open("your_image.jpg").convert("RGB")
px = proc(images=img, return_tensors="pt").to(dev).pixel_values
with torch.no_grad():
    feats = backbone.get_image_features(pixel_values=px)
    emb = feats.pooler_output if hasattr(feats, "pooler_output") else feats  # [1, 1152]
    out = head(emb)
print(out["score"].item(), out["ordinal_score"].item())  # [0,1] and [1,5]
```

Backbone must be exactly `{backbone}` (**patch14**) with the raw `pooler_output` — verified
against the training embeddings at cosine 0.9998. Anything else scores wrong.

## Scores (held-out test split)

| Spearman | Pearson | MAE (1–5) | Top-5% |
|---|---|---|---|
| {_fmt(metrics, "spearman")} | {_fmt(metrics, "pearson")} | {_fmt(metrics, "mae")} | {_fmt(metrics, "top_5pct")} |

Architecture: `embedding[1152] → LayerNorm → MLP {hidden} → ordinal head`. Trained on one
person's private 1–5 ratings; labels and images not released. [Source]({REPO_URL})
"""
