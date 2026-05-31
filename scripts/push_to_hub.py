"""Publish the trained aesthetic head to the Hugging Face Hub.

Loads a ``best.safetensors`` + ``best.json`` checkpoint, rebuilds it as
:class:`~silva.models.aesthetic.EmbeddingAestheticModel`, BAKES a calibration LUT from
the manifest (so the published model's ``calibrated_score`` matches what the pictoria
library writer produces in batch), and pushes ``model.safetensors`` + ``config.json`` +
a generated ``README.md`` model card. Only the head ships — the frozen SigLIP2 backbone
is upstream (see the model card).

Runs in the workspace venv (``uv sync --all-packages``).

Usage:
    hf auth login                # once, or pass HF_TOKEN in the environment
    uv run python scripts/push_to_hub.py \
        --checkpoint outputs/v1_stage1_head \
        --repo-id <user>/silva-aesthetic
    # dry run — write the repo files locally without uploading:
    uv run python scripts/push_to_hub.py --repo-id <user>/silva --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from silva.models.aesthetic import N_CAL_KNOTS, EmbeddingAestheticModel
from silva_train.calibration import build_calibration_lut
from silva_train.checkpoint import load_checkpoint
from silva_train.model_card import render_model_card

BACKBONE = "google/siglip2-so400m-patch14-384"  # must match the embeddings in the manifest (pictoria ai/siglip_embed.py)


def fit_calibration(model: EmbeddingAestheticModel, manifests: list[str]) -> list[float]:
    """Bake the calibration LUT into ``model`` from the manifest latents + label distribution.

    The published per-image model then emits the same calibrated score the batch library
    writer does: target = the (merged) manifests' 1~5 label frequencies, source = the model's
    latents. Accepts several parquets so calibration matches a multi-source training set.
    """
    import pandas as pd

    df = pd.concat([pd.read_parquet(m, columns=["personal_score", "embedding"]) for m in manifests], ignore_index=True)
    x = torch.tensor(np.stack(df["embedding"].to_numpy()), dtype=torch.float32)
    with torch.no_grad():
        feat = model.trunk(model.norm(x))
        latent = model.head.latent(feat).squeeze(-1).numpy()
    target = [float((df["personal_score"] == k).sum()) for k in range(1, 6)]
    lat_knots, score_knots = build_calibration_lut(latent, target, n_knots=N_CAL_KNOTS)
    model.set_calibration(lat_knots, score_knots)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Push the SILVA aesthetic head to the Hugging Face Hub.")
    parser.add_argument("--checkpoint", default="outputs/v1_stage1_head", help="run dir (or its best.safetensors)")
    parser.add_argument("--repo-id", required=True, help="target repo, e.g. <user>/silva-aesthetic")
    parser.add_argument("--manifest", nargs="+", default=None, help="manifest(s) for eval + calibration (default: from the checkpoint config)")
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    parser.add_argument("--dry-run", action="store_true", help="write repo files to ./hub_export/ instead of uploading")
    parser.add_argument("--commit-message", default="Upload SILVA aesthetic head")
    args = parser.parse_args()

    state, config, metrics = load_checkpoint(args.checkpoint)
    model_cfg = config["model"]

    model = EmbeddingAestheticModel(
        embedding_dim=model_cfg["embedding_dim"],
        dropout=model_cfg.get("dropout", 0.1),
        hidden_dims=model_cfg.get("hidden_dims", []),
    )
    model.load_state_dict(state, strict=False)  # calibration buffers default to zeros; filled by fit_calibration
    model.eval()

    # The checkpoint's stored metrics are the *val* best; the card reports the held-out
    # *test* split, so re-evaluate on test when the manifest is available — and bake the
    # calibration LUT from the same manifest(s). manifest_path may be a single parquet or a
    # list (multi-source training); normalise to a list so eval/calibration see the same data.
    manifests = args.manifest or config["data"]["manifest_path"]
    manifests = [manifests] if isinstance(manifests, str) else list(manifests)
    if all(Path(m).exists() for m in manifests):
        from silva_train.evaluate import evaluate

        metrics = evaluate(
            args.checkpoint, manifests, "test",
            model_cfg["embedding_dim"], model_cfg.get("dropout", 0.1), model_cfg.get("hidden_dims", []),
        )
        print(f"evaluated on test split of {manifests}: spearman={metrics.get('spearman'):.4f}")
        target = fit_calibration(model, manifests)
        tnorm = [round(t / sum(target), 3) for t in target]
        print(f"baked calibration LUT ({N_CAL_KNOTS} knots) to label distribution {tnorm}")
    else:
        missing = [m for m in manifests if not Path(m).exists()]
        print(f"WARNING: manifest(s) {missing} not found — VAL metrics on the card, and NO calibration baked (model emits raw score).")

    model.eval()
    card = render_model_card(repo_id=args.repo_id, backbone=BACKBONE, model_cfg=model_cfg, metrics=metrics)

    if args.dry_run:
        out = Path("hub_export")
        model.save_pretrained(out)
        (out / "README.md").write_text(card, encoding="utf-8")
        print(f"dry run -> wrote {out}/ : {[p.name for p in out.iterdir()]}")
        print(json.dumps(metrics, indent=2, default=str))
        return

    model.push_to_hub(args.repo_id, private=args.private, commit_message=args.commit_message)
    # README is not part of save_pretrained — upload it as the model card.
    from huggingface_hub import upload_file

    tmp = Path("hub_export_readme.md")
    tmp.write_text(card, encoding="utf-8")
    upload_file(path_or_fileobj=str(tmp), path_in_repo="README.md", repo_id=args.repo_id, repo_type="model", commit_message="Add model card")
    tmp.unlink()
    print(f"pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
