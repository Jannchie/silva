#!/usr/bin/env bash
# One-shot relabel iteration: re-export manifest from pictoria, retrain, report error drift.
#
# Run this after relabeling a batch in pictoria. It carries over all existing splits (no
# leakage), retrains the head, and prints the test metrics + misclassification types so you
# can watch how the error TYPES (komone_ushio etc.) shift round over round.
#
#   bash scripts/refresh.sh [DB_PATH]
set -euo pipefail
export WANDB_MODE=disabled
export PYTHONIOENCODING=utf-8
DB="${1:-E:/pictoria/server/illustration/images/.pictoria/pictoria.sqlite}"

echo "### 1/4  re-export manifest (carry splits + diff) ###"
uv run --extra export python scripts/export_manifest.py --db "$DB" --output data/manifest.parquet

echo; echo "### 2/4  retrain head ###"
uv run python -m silva_train.train --config configs/v1_stage1_head.yaml | tail -4

echo; echo "### 3/4  test metrics ###"
uv run python -m silva_train.evaluate --checkpoint outputs/v1_stage1_head --split test

echo; echo "### 4/4  misclassification + tag enrichment ###"
uv run python scripts/misclass_probe.py --min-gap 2.0 --top 10
echo
uv run python scripts/suspect_tags.py | head -22
