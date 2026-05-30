# SILVA monorepo split + Hugging Face publishing — design

Date: 2026-05-31
Status: approved (brainstorming), ready for implementation plan

## Goal

Publish the personal aesthetic-scoring model so others can use it, while keeping
the training pipeline private. This requires separating today's single `silva/`
package into three distinct products with clean boundaries:

1. **Training scripts** — private, not published.
2. **Library (+ CLI)** — public, `pip install silva`.
3. **Model weights** — public, on the Hugging Face Hub.

The driving constraint: the library others call and the scripts that train the
model are different concepts and must not share a dependency footprint. A
downloader who only wants `embedding → score` should not pull `transformers`,
`accelerate`, `wandb`, or `scipy`.

## Non-goals

- Releasing training labels or source images (personal-preference data + image
  copyright). The HF repo is the head weights only.
- Publishing `silva-train` to PyPI.
- Retraining or changing the model architecture. This is purely packaging +
  publishing.

## 1. Repository layout — monorepo (uv workspace), `src/` layout

```
silva/                              # repo root = uv workspace
├── pyproject.toml                  # [tool.uv.workspace] members = ["packages/*"]
├── packages/
│   ├── silva/                      # (1) public library — published to PyPI
│   │   ├── pyproject.toml          # name = "silva"
│   │   └── src/silva/
│   │       ├── models/
│   │       │   ├── aesthetic.py    # EmbeddingAestheticModel (model definition, single source)
│   │       │   └── ordinal_head.py # OrdinalHead
│   │       ├── scoring.py          # score-reconstruction funcs (moved out of losses.py)
│   │       ├── hub.py              # HubAestheticModel (PyTorchModelHubMixin)
│   │       ├── backbone.py         # [backbone] extra: image → embedding (transformers)
│   │       └── cli.py              # [backbone] extra: `silva score img.jpg`
│   │
│   └── silva-train/                # (2) training package — NOT published, internal
│       ├── pyproject.toml          # name = "silva-train", depends on silva
│       └── src/silva_train/
│           ├── losses.py           # training losses only
│           ├── model_card.py       # render_model_card (publish-time only)
│           ├── config.py           # pydantic Config
│           ├── data/ metrics.py evaluate.py train.py
│           └── ...
└── scripts/
    ├── push_to_hub.py              # publish: silva.hub + silva_train.evaluate + silva_train.model_card
    └── verify_embedding.py         # pre-publish self-check (cosine vs backbone)
```

**Dependency direction (one-way, invariant):** `silva-train` → depends on →
`silva`. `silva` never depends on `silva-train`. The model definition lives in
exactly one place (`silva`); training imports it.

**Three products:**

| Product | Physical location | Who gets it |
|---|---|---|
| (1) Training scripts | `packages/silva-train/` | private (not published) |
| (2) Library + CLI | `packages/silva/` | public, `pip install silva` |
| (3) Model weights | HF repo (3 files) | public, `from_pretrained` |

Decision: `src/` layout (standard, avoids tests importing the source tree by
accident); full physical move into `packages/` (cleanest long-term, accepts the
git path-history churn).

## 2. `losses.py` split

`losses.py` today mixes two responsibilities. Inference needs **score
reconstruction** (`model.forward` uses it → must ship in the library). Training
needs the **loss functions** (stay in the training package).

| Symbol | Class | Destination |
|---|---|---|
| `unit_score_from_logits` | score reconstruction | `silva/scoring.py` (library) |
| `ordinal_score_from_logits` | score reconstruction | `silva/scoring.py` (library) |
| `NUM_THRESHOLDS = 4` | shared constant | `silva/scoring.py` (library); training imports it |
| `make_ordinal_targets` | training | `silva_train/losses.py` |
| `ordinal_loss` / `compute_pos_weight` | training | `silva_train/losses.py` |
| `pairwise_ranking_loss` / `soft_spearman_loss` / `listwise_loss` / `silva_loss` | training | `silva_train/losses.py` |

Knock-on edits:
- `aesthetic.py` import changes `from silva.losses import ...` → `from silva.scoring import ...`.
- `silva_train/losses.py` adds `from silva.scoring import ordinal_score_from_logits, NUM_THRESHOLDS`
  so training losses reuse the library's reconstruction functions — training and
  inference stay single-sourced.

Result: the library contains zero loss code; `transformers` / `scipy` / `wandb`
never enter the core dependency set.

## 3. Library layering + dependency boundaries

`packages/silva/pyproject.toml`:

```toml
[project]
name = "silva"
dependencies = [          # core: embedding → score, nothing more
    "torch",
    "huggingface-hub",    # PyTorchModelHubMixin / from_pretrained
]

[project.optional-dependencies]
backbone = [              # end-to-end image → score + CLI
    "transformers",
    "pillow",
]

[project.scripts]
silva = "silva.cli:main"  # entry point registered; only runs if [backbone] is installed
```

**Three usage modes:**

| Scenario | Install | Entry point |
|---|---|---|
| Already have embeddings, just score | `pip install silva` | `HubAestheticModel.from_pretrained(...)` → `head(emb)` |
| Have images, want end-to-end | `pip install "silva[backbone]"` | `from silva.backbone import score_image` or CLI |
| Command line | `pip install "silva[backbone]"` | `silva score img.jpg` |

**Robustness:** `cli.py` / `backbone.py` must not break a core-only `import
silva`. `backbone.py` imports `transformers` at module top and, on
`ImportError`, raises a clear `pip install "silva[backbone]"` message. The entry
point is registered unconditionally but only functions with the extra installed.

## 4. Embedding consistency (RESOLVED)

The head is head-only; its 1152-d input must be produced exactly as the training
embeddings were (pictoria `ai/siglip_embed.py`). Verified by
`scripts/verify_embedding.py` (recompute from source images, compare to stored
manifest vectors):

| Item | Conclusion |
|---|---|
| Backbone identity | **`google/siglip2-so400m-patch14-384`** (patch14, NOT patch16). Pinned in `push_to_hub.py:28` and `model_card.py:101`. |
| Pooling | take `pooler_output` (newer `transformers` wraps `get_image_features`). |
| Normalisation | non-issue. Training embeddings are un-normalised (norm ≈ 15), but the head's input `LayerNorm` absorbs vector scale, so L2-normalising is harmless but unnecessary. |
| Measured match | recompute vs stored **cosine 0.9998**. |

`backbone.py` is the single gatekeeper for train/inference consistency. What it
guards is **backbone identity (patch14) + pooling (pooler_output)** — not
normalisation. It reuses `verify_embedding.py`'s verified path.

Note: these correctness fixes currently live as **uncommitted working-tree
changes** (`scripts/push_to_hub.py`, `silva/model_card.py`) plus the untracked
`scripts/verify_embedding.py`. The restructure must carry them along (or commit
them first) so the move does not drop them.

## 5. Publishing flow

`scripts/push_to_hub.py` orchestrates (not auto-run — needs the user's HF token
+ namespace):

```
best.pt ──┐
          ├─ silva_train.evaluate(test split) ──→ metrics
          │                                          │
          ├─ silva.hub.HubAestheticModel ←load_state_dict
          │        │                                 │
          │        ├─ push_to_hub() → model.safetensors + config.json
          │        │
          └─ silva_train.model_card.render(metrics) → README.md ─upload_file→
                                                                    HF repo (3 files)
```

Pre-publish: run `verify_embedding.py` (confirm cosine ≈ 1; already 0.9998), then
`--dry-run` to inspect local artifacts, then push for real.

File placement:

| File | Destination | Reason |
|---|---|---|
| `hub.py` (`HubAestheticModel`) | `packages/silva/src/silva/hub.py` (library) | downloaders' `from_pretrained` needs it |
| `model_card.py` (`render_model_card`) | `packages/silva-train/` (training) | publish-time only, embeds training metrics |
| `push_to_hub.py` | root `scripts/` (unchanged) | spans both packages |
| `verify_embedding.py` | root `scripts/` (unchanged) | pre-publish self-check |

## Risks / open items

- Git path-history churn from the physical move (accepted).
- Carrying the uncommitted correctness fixes through the move (call-out above).
- `transformers` `get_image_features` return shape across versions — handled by
  the `pooler_output` fallback already in the card and verify script.
