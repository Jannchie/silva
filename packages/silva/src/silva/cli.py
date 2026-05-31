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

    from silva.scorer import SilvaScorer  # noqa: PLC0415

    scorer = SilvaScorer.from_pretrained(args.repo_id, device=args.device)
    for path, score in zip(args.images, scorer.score(args.images), strict=True):
        print(f"{path}\tscore={score:.4f}")


if __name__ == "__main__":
    main()
