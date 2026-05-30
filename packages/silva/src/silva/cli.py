"""`silva score IMG [IMG ...]` — end-to-end image aesthetic scoring.

Requires the ``[backbone]`` extra. The transformers/pillow imports live inside
``main`` so the registered entry point imports cleanly in a core-only install and
fails with a clear message only when actually run.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="silva", description="Score images for personal aesthetic appeal.")
    parser.add_argument("command", choices=["score"], help="only 'score' is supported")
    parser.add_argument("images", nargs="+", help="image file paths")
    parser.add_argument("--repo-id", default="<user>/silva-aesthetic", help="Hugging Face repo of the published head")
    parser.add_argument("--device", default=None, help="torch device override (default: auto)")
    args = parser.parse_args()

    from PIL import Image  # noqa: PLC0415

    from silva.backbone import Embedder, score_images  # noqa: PLC0415
    from silva.hub import HubAestheticModel  # noqa: PLC0415

    head = HubAestheticModel.from_pretrained(args.repo_id)
    embedder = Embedder(device=args.device)
    images = [Image.open(path) for path in args.images]
    for path, res in zip(args.images, score_images(images, head, embedder), strict=True):
        print(f"{path}\tscore={res['score']:.4f}\tordinal={res['ordinal_score']:.4f}")


if __name__ == "__main__":
    main()
