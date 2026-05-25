"""Example manifest producer: Postgres -> parquet.

This is ONE example data source. The training pipeline depends only on the manifest
contract (`silva.data.manifest`), so you can produce the parquet from anything —
a CSV, a scrape, or a merge of several sources — as long as `validate_manifest`
passes. See README for the schema.

Requires the optional 'postgres' extra:

    uv sync --extra postgres
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from silva.data.manifest import assign_splits, write_manifest


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Example: export a SILVA manifest from Postgres")
    parser.add_argument("--table", required=True, help="source table name")
    parser.add_argument("--image-col", required=True, help="column with local image path")
    parser.add_argument("--score-col", required=True, help="column with your 1~5 score")
    parser.add_argument("--scorer-a-col", default=None, help="optional external scorer A column (stored for v2)")
    parser.add_argument("--scorer-b-col", default=None, help="optional external scorer B column (stored for v2)")
    parser.add_argument("--output", default="data/manifest.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import pandas as pd
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise SystemExit("Postgres export needs the optional extra: uv sync --extra postgres") from exc

    select = [f"{args.image_col} AS image_path", f"{args.score_col} AS personal_score"]
    if args.scorer_a_col:
        select.append(f"{args.scorer_a_col} AS scorer_a")
    if args.scorer_b_col:
        select.append(f"{args.scorer_b_col} AS scorer_b")
    query = f"SELECT {', '.join(select)} FROM {args.table} WHERE {args.score_col} IS NOT NULL"  # noqa: S608

    df = pd.read_sql(query, create_engine(os.environ["DATABASE_URL"]))
    df = df.dropna(subset=["image_path", "personal_score"])
    df["personal_score"] = df["personal_score"].astype("int64")
    df["split"] = assign_splits(df["image_path"].tolist(), seed=args.seed)

    out = write_manifest(df, args.output)
    print(f"Wrote {len(df)} rows to {out}")
    print(df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
