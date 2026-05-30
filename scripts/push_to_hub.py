"""Publish the trained aesthetic head to the Hugging Face Hub.

Loads a ``best.pt`` checkpoint (``{model, config, metrics}``), rebuilds the head as
:class:`~silva.hub.HubAestheticModel`, and pushes ``model.safetensors`` +
``config.json`` + a generated ``README.md`` model card to a model repo. Only the
head ships — the frozen SigLIP2 backbone is upstream (see the model card).

Usage:
    huggingface-cli login        # once, or pass HF_TOKEN in the environment
    uv run --extra hub python scripts/push_to_hub.py \
        --checkpoint outputs/v1_stage1_head/best.pt \
        --repo-id <user>/silva-aesthetic
    # dry run — write the repo files locally without uploading:
    uv run --extra hub python scripts/push_to_hub.py --repo-id <user>/silva --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from silva.hub import HubAestheticModel
from silva.model_card import render_model_card

BACKBONE = "google/siglip2-so400m-patch16-384"


def main() -> None:
    parser = argparse.ArgumentParser(description="Push the SILVA aesthetic head to the Hugging Face Hub.")
    parser.add_argument("--checkpoint", default="outputs/v1_stage1_head/best.pt")
    parser.add_argument("--repo-id", required=True, help="target repo, e.g. <user>/silva-aesthetic")
    parser.add_argument("--manifest", default=None, help="manifest for test-split eval (default: the one in the checkpoint config)")
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    parser.add_argument("--dry-run", action="store_true", help="write repo files to ./hub_export/ instead of uploading")
    parser.add_argument("--commit-message", default="Upload SILVA aesthetic head")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = ckpt["config"]["model"]

    # The checkpoint's stored metrics are the *val* best; the card reports the held-out
    # *test* split, so re-evaluate on test when the manifest is available locally.
    metrics = ckpt.get("metrics", {})
    manifest = args.manifest or ckpt["config"]["data"]["manifest_path"]
    if Path(manifest).exists():
        from silva.evaluate import evaluate

        metrics = evaluate(
            args.checkpoint, manifest, "test",
            model_cfg["embedding_dim"], model_cfg.get("dropout", 0.1), model_cfg.get("hidden_dims", []),
        )
        print(f"evaluated on test split of {manifest}: spearman={metrics.get('spearman'):.4f}")
    else:
        print(f"WARNING: manifest {manifest} not found — card will show the checkpoint's VAL metrics, not test.")

    model = HubAestheticModel(
        embedding_dim=model_cfg["embedding_dim"],
        dropout=model_cfg.get("dropout", 0.1),
        hidden_dims=model_cfg.get("hidden_dims", []),
    )
    model.load_state_dict(ckpt["model"])
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
