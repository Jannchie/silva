"""CLI: export the training manifest (parquet) from Postgres.

DB credentials come from .env (DATABASE_URL). Table / column names are passed as
flags — this is the only place the DB schema is referenced.
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from silva.data.export_manifest import export_manifest


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Export SILVA training manifest from Postgres")
    parser.add_argument("--table", required=True, help="source table name")
    parser.add_argument("--image-col", required=True, help="column with local image path")
    parser.add_argument("--score-col", required=True, help="column with your 1~5 score")
    parser.add_argument("--scorer-a-col", default=None, help="optional external scorer A column (stored for v2)")
    parser.add_argument("--scorer-b-col", default=None, help="optional external scorer B column (stored for v2)")
    parser.add_argument("--output", default="data/manifest.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    database_url = os.environ["DATABASE_URL"]
    df = export_manifest(
        database_url=database_url,
        table=args.table,
        image_path_col=args.image_col,
        personal_score_col=args.score_col,
        scorer_a_col=args.scorer_a_col,
        scorer_b_col=args.scorer_b_col,
        output_path=args.output,
        seed=args.seed,
    )
    print(f"Wrote {len(df)} rows to {args.output}")
    print(df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
