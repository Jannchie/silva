# SILVA

**S**igLIP-based **I**llustration **V**isual **A**esthetic Scorer — learns *one person's*
taste from their own 1–5 ratings and scores any illustration on a continuous scale.

It is not a universal quality model. You feed it your ratings, it fits an ordinal-regression
head on top of frozen `google/siglip2-so400m-patch14-384` embeddings, and the score reflects
*your* preferences — nobody else's. The backbone stays frozen, so a run only trains a ~7 MB
head over precomputed embeddings: minutes on a GPU, and it runs on CPU too.

- **Input**: 1152-d SigLIP2-SO400M-384 embeddings, computed upstream by an adapter — so the
  training library has no `transformers` dependency.
- **Head**: learnable monotone ordinal thresholds, with optional per-threshold class balancing.
- **Output**: a single `score ∈ [0, 1]` (fraction of quality bars cleared). Training
  metrics (MAE / RMSE / Spearman / QWK) are computed in the 1–5 label space.

## Packages

This is a uv workspace of two packages:

| Package | Install | Role |
|---|---|---|
| [`silva`](packages/silva) | `pip install silva-scorer` | Inference library + `silva` CLI. Published to PyPI (imports as `silva`). |
| [`silva-train`](packages/silva-train) | workspace-only | Training, evaluation, manifest tooling. Private. |

Already have a trained head on the Hub? You only need `silva` — see its
[README](packages/silva#readme).

## Setup (development)

```bash
uv sync                 # GPU-ready: torch is pinned to a CUDA build (cu132)
uv sync --extra export  # + sqlite-vec, to read embeddings from the pictoria adapter
```

`torch` is pinned to cu132, so `uv sync` is GPU-ready out of the box (NVIDIA driver must
support CUDA ≥ 13.2). Training also runs on CPU.

## End-to-end workflow

**1. Build a manifest** — a columnar parquet matching the contract in `silva_train.data.manifest`:

| column | type | required | notes |
|---|---|---|---|
| `embedding` | list<float>[1152] | yes | SigLIP2 image embedding |
| `personal_score` | int 1..5 | yes | your rating |
| `split` | `train`/`val`/`test` | yes | content-keyed by embedding (leakage-free, id-free) |
| `post_id` | int | no | optional provenance label (unused by splitting/merging) |

An adapter for the pictoria SQLite DB ships in `scripts/`:

```bash
uv run python scripts/export_manifest.py \
    --db /path/to/pictoria.sqlite \
    --output data/manifest.parquet
```

Any other source works the same way — call `build_manifest` / `write_manifest` to emit the
columns above. `validate_manifest` enforces the contract on load.

**2. Train** the ordinal head:

```bash
uv run accelerate launch -m silva_train.train --config configs/v1_stage1_head.yaml
```

Saves the best checkpoint (by Spearman) to `outputs/v1_stage1_head/` as `best.safetensors` + `best.json`.

**3. Evaluate** on a split:

```bash
uv run python -m silva_train.evaluate --checkpoint outputs/v1_stage1_head --split val
```

Reports MAE, RMSE, Pearson, Spearman, QWK, Top-1% / Top-5% precision.

**4. Score images** once a head is published to the Hub:

```bash
pip install "silva-scorer[backbone]"
silva image.jpg
```

See [`packages/silva`](packages/silva#readme) for the Python API.

## Tests

```bash
uv run pytest
```

Covers pure logic (ordinal targets, threshold monotonicity, score reconstruction, all metrics,
leakage-free splits, manifest contract) plus a CPU-only end-to-end training smoke test.

## Scope (v1)

Personal score only. External AI scorers, LoRA / full fine-tune, distribution head, and serving
are deferred. Design spec:
[`docs/superpowers/specs/2026-05-24-silva-design.md`](docs/superpowers/specs/2026-05-24-silva-design.md).
