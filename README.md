# SILVA

**S**igLIP-based **I**llustration **V**isual **A**esthetic Scorer — a personal aesthetic
scorer trained on your own 1~5 ratings. Trained as ordinal regression; outputs a
continuous score.

- **Input**: precomputed **SigLIP2-SO400M-384 embeddings** (1152-d). v1 freezes the
  backbone, so embeddings are computed upstream by an adapter — the training library
  itself has no backbone and no `transformers` dependency.
- **Head**: ordinal head with learnable monotone thresholds, plus optional automatic
  per-threshold class balancing (`pos_weight`).
- **Canonical output**: `score ∈ [0, 1]` (mean of the 4 threshold probabilities,
  i.e. "fraction of quality bars cleared"). Rescale to any range with
  `lo + (hi - lo) * score`. Labels, training, and `MAE/RMSE` stay in `1~5` space.

Design spec: [`docs/superpowers/specs/2026-05-24-silva-design.md`](docs/superpowers/specs/2026-05-24-silva-design.md).

## Setup

```bash
uv sync
```

`torch` is pinned to a CUDA build (cu132) in `pyproject.toml`, so `uv sync` is
GPU-ready out of the box (NVIDIA driver must support CUDA ≥ 13.2). Training also
runs on CPU (`mixed_precision: no`). The pictoria SQLite adapter needs an optional
extra (sqlite-vec) to read the embedding table:

```bash
uv sync --extra export
```

## Workflow

1. **Produce a manifest** — a columnar parquet matching the contract in
   `silva.data.manifest`. The training library depends only on this shape and is
   blind to where embeddings come from.

   | column | type | required | notes |
   |---|---|---|---|
   | `embedding` | list<float>[D] | yes | fixed-dimension feature vector (v1: 1152) |
   | `personal_score` | int 1..5 | yes | your rating |
   | `split` | `train`/`val`/`test` | yes | use `assign_splits` for leakage-free splits |
   | `post_id` | int | no | provenance / split-dedup key |

   An adapter for the pictoria SQLite library is provided. It reads precomputed
   SigLIP2 embeddings (`post_vectors_siglip2`) joined with your scores
   (`posts.score`, filtered to `score > 0`):

   ```bash
   uv run python scripts/export_manifest.py \
       --db /mnt/e/pictoria/server/illustration/images/.pictoria/pictoria.sqlite \
       --output data/manifest.parquet
   ```

   Any other source works the same way — call `build_manifest` / `write_manifest`
   to emit the columns above; `validate_manifest` (also run on dataset load)
   enforces the contract.

2. **Train** the ordinal head on the embeddings:

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

Covers pure logic (ordinal targets, `pos_weight`, threshold monotonicity, score
reconstruction, all metrics, leakage-free splits, manifest contract) plus an
end-to-end training smoke test on toy embeddings (runs by default, CPU-only).

## Scope (v1)

Personal score only. External AI scorers, LoRA / full fine-tune, distribution head,
and serving are deferred — the multi-task loss hook is reserved in code, and the
external scorers in the DB (`post_aesthetic_scores`, `post_waifu_scores`) can be
added as manifest columns + aux heads when needed.
