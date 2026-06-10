"""`silva IMG [IMG ...]` — end-to-end image aesthetic scoring.

Requires the ``[backbone]`` extra. The transformers/pillow imports live inside
``main`` so the registered entry point imports cleanly in a core-only install and
fails with a clear message only when actually run.
"""

from __future__ import annotations

import argparse

DEFAULT_REPO_ID = "Jannchie/silva-aesthetic"


def main() -> None:
    parser = argparse.ArgumentParser(prog="silva", description="Score images for personal aesthetic appeal.")
    parser.add_argument("images", nargs="+", help="image file paths")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"Hugging Face repo of the published head (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--device", default=None, help="torch device override (default: auto)")
    args = parser.parse_args()

    from silva.scorer import SilvaScorer  # noqa: PLC0415

    scorer = SilvaScorer.from_pretrained(args.repo_id, device=args.device)
    scores = scorer.score(args.images)
    # One image -> bare score (pipe-friendly); many -> "score\tpath" rows.
    for path, score in zip(args.images, scores, strict=True):
        print(f"{score:.4f}" if len(args.images) == 1 else f"{score:.4f}\t{path}")


if __name__ == "__main__":
    main()
