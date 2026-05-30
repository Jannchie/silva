# silva-train

Training, evaluation, and manifest tooling for [SILVA](https://github.com/Jannchie/silva) — fits
the ordinal-regression head that the [`silva`](../silva) inference package serves. Private: it is
not published to PyPI and lives only in the workspace.

It depends on `silva` for the model definition and scoring, and adds everything inference does
not need: the manifest contract, the dataset loader, the loss functions, metrics, the training
loop (via `accelerate`), and the Hub model-card renderer.

## Install

Part of the uv workspace — run from the repo root:

```bash
uv sync                 # core training stack (torch, accelerate, pandas, scipy, ...)
uv sync --extra export  # + sqlite-vec, for the pictoria manifest adapter
uv sync --extra wandb   # + wandb, for experiment tracking
```

## Manifest contract

Training reads a columnar parquet defined by `silva_train.data.manifest`:

| column | type | required | notes |
|---|---|---|---|
| `embedding` | list<float>[1152] | yes | SigLIP2-SO400M-384 image embedding |
| `personal_score` | int 1..5 | yes | your rating |
| `split` | `train`/`val`/`test` | yes | `assign_splits` gives leakage-free splits |
| `post_id` | int | no | provenance / dedup key |

Build one from any source with `build_manifest` / `write_manifest`; `validate_manifest` enforces
the contract on load. An adapter for the pictoria SQLite DB ships in `scripts/export_manifest.py`.

## Train

```bash
uv run accelerate launch -m silva_train.train --config configs/v1_stage1_head.yaml
```

Configs live in `configs/` — `v1_stage1_head.yaml` for production, `sample_cpu.yaml` for a quick
CPU run. The loss combines ordinal cross-entropy with optional pairwise-ranking and soft-Spearman
terms, plus per-threshold class balancing. The best checkpoint (by Spearman) lands in
`outputs/<run>/best.pt` as `{model, config, metrics}`.

## Evaluate

```bash
uv run python -m silva_train.evaluate --checkpoint outputs/v1_stage1_head/best.pt --split val
```

Reports MAE, RMSE, Pearson, Spearman, and quadratic weighted kappa in 1–5 label space.

## Publish

```bash
uv run python scripts/push_to_hub.py \
    --checkpoint outputs/v1_stage1_head/best.pt \
    --repo-id Jannchie/silva-aesthetic
```

Rebuilds the head as `silva.hub.HubAestheticModel`, re-evaluates on the test split, and pushes
`model.safetensors` + `config.json` + a rendered model card (`silva_train.model_card`) so `silva`
users can `from_pretrained` it. Add `--dry-run` to write the repo files to `hub_export/` without
uploading.
