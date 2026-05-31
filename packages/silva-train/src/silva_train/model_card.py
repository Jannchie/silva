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

Scores an illustration by **one specific person's** taste — not a universal quality
model, so it won't match anyone else's preferences. Output is a single number in
`[0, 1]`; higher means more to this person's liking.

**Only the head ships here (~7 MB), not an image model.** It runs on top of the frozen
`{backbone}` backbone, which `silva[backbone]` installs and loads for you.

## Quickstart

```python
# pip install "silva-scorer[backbone] @ git+{REPO_URL}"
from silva import SilvaScorer

scorer = SilvaScorer.from_pretrained("{repo_id}")
print(scorer.score("your_image.jpg"))     # 0.73
print(scorer.score(["a.jpg", "b.jpg"]))    # [0.73, 0.41]
```

Already have `{backbone}` embeddings? Skip the backbone and score them directly:

```python
# pip install "silva-scorer @ git+{REPO_URL}"
from silva import EmbeddingAestheticModel

head = EmbeddingAestheticModel.from_pretrained("{repo_id}").eval()
score = head(embedding)["calibrated_score"]  # calibrated to the label distribution; ["score"] for raw. embedding: [B, 1152] pooler_output
```

## Scores (held-out test split)

| Spearman | Pearson | MAE (1–5) | Top-5% |
|---|---|---|---|
| {_fmt(metrics, "spearman")} | {_fmt(metrics, "pearson")} | {_fmt(metrics, "mae")} | {_fmt(metrics, "top_5pct")} |

Architecture: `embedding[1152] → LayerNorm → MLP {hidden} → ordinal head`. Trained on one
person's private 1–5 ratings; labels and images not released. [Source]({REPO_URL})
"""
