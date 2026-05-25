# SILVA

**S**igLIP-based **I**llustration **V**isual **A**esthetic Scorer — a personal aesthetic
scorer trained on your own 1~5 ratings. Trained as ordinal regression; outputs a
continuous score.

- **Backbone**: `google/siglip2-so400m-patch14-384` (frozen in v1).
- **Head**: ordinal head with learnable monotone thresholds.
- **Canonical output**: `score ∈ [0, 1]` (mean of the 4 threshold probabilities,
  i.e. "fraction of quality bars cleared"). Rescale to any range with
  `lo + (hi - lo) * score`. Labels, training, and `MAE/RMSE` stay in `1~5` space.

Design spec: [`docs/superpowers/specs/2026-05-24-silva-design.md`](docs/superpowers/specs/2026-05-24-silva-design.md).

## Setup

```bash
uv sync
```

`torch` is pinned to a CUDA build (cu132) in `pyproject.toml`, so `uv sync` is
GPU-ready out of the box (NVIDIA driver must support CUDA ≥ 13.2). The Postgres
example producer needs an optional extra:

```bash
uv sync --extra postgres
cp .env.example .env   # set DATABASE_URL
```

## Workflow

1. **Produce a manifest** — a parquet matching the contract in `silva.data.manifest`.
   The training pipeline depends only on this shape; the data source is up to you.

   | column | type | required | notes |
   |---|---|---|---|
   | `image_path` | str | yes | local image path |
   | `personal_score` | int 1..5 | yes | your rating |
   | `split` | `train`/`val`/`test` | yes | use `assign_splits` for leakage-free splits |
   | `scorer_a`, `scorer_b` | float | no | external scorers, stored for v2 |

   Add splits with `assign_splits`, then validate + write with `write_manifest`.
   A Postgres source is provided as one example (optional extra):

   ```bash
   uv sync --extra postgres
   uv run python scripts/export_manifest.py \
       --table my_table --image-col image_path --score-col my_score \
       --output data/manifest.parquet
   ```

   Any other source works the same way — emit a parquet with the columns above;
   `validate_manifest` (also run on dataset load) enforces the contract.

2. **Train** (Stage 1: frozen backbone + ordinal head):

   ```bash
   uv run accelerate launch -m silva.train --config configs/v1_stage1_head.yaml
   ```

   Saves the best checkpoint (by Spearman) to `outputs/v1_stage1_head/best.pt`.

3. **Evaluate** a checkpoint on a split:

   ```bash
   uv run python -m silva.evaluate --checkpoint outputs/v1_stage1_head/best.pt --split val
   ```

   Reports MAE, RMSE, Pearson, Spearman, QWK, Top-1%/Top-5% precision.

## Tests

```bash
uv run pytest
```

Covers the pure logic: ordinal target conversion, threshold monotonicity, score
reconstruction, all metrics, and leakage-free split assignment.

## Scope (v1)

Personal score only. External AI scorers, LoRA / full fine-tune, distribution head,
and serving are deferred — extension points are reserved in code (`aux_heads`, the
multi-task loss hook, and the exported `scorer_a/b` columns).
