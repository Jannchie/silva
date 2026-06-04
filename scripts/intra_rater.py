"""Measure YOUR test-retest reliability — the noise ceiling of the whole pipeline.

The model cannot agree with you more than you agree with yourself. This script samples a
blind re-rating sheet from the manifest, you fill in fresh 1~5 scores (without peeking at
the old ones), and the report converts your self-agreement into the implied model ceiling
(``sqrt(reliability)`` by attenuation). If the ceiling sits close to the current test
Spearman, the next lever is relabelling/data — more modelling cannot help.

  1. uv run python scripts/intra_rater.py sample --n 150          -> data/intra_rater_sheet.csv
  2. open each post_id in pictoria, fill the new_score column (do NOT look at old scores)
  3. uv run python scripts/intra_rater.py report                  -> reliability + ceiling
"""

from __future__ import annotations

import argparse

import pandas as pd

from silva_train.intra_rater import agreement_report, sample_for_rerating

DEFAULT_SHEET = "data/intra_rater_sheet.csv"


def cmd_sample(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.manifest, columns=["post_id", "personal_score"])
    sheet = sample_for_rerating(df, n=args.n, seed=args.seed)
    # blind sheet: post_id only — the old score deliberately stays behind in the manifest
    out = sheet[["post_id"]].copy()
    out["new_score"] = ""
    out.to_csv(args.sheet, index=False)
    print(f"saved {len(out)} rows -> {args.sheet}")
    print("fill new_score (1~5) for each post_id WITHOUT looking at your old scores,")
    print(f"then run: uv run python scripts/intra_rater.py report --sheet {args.sheet}")


def cmd_report(args: argparse.Namespace) -> None:
    sheet = pd.read_csv(args.sheet)
    rated = sheet[pd.to_numeric(sheet["new_score"], errors="coerce").notna()].copy()
    if rated.empty:
        print(f"no filled new_score rows in {args.sheet} - nothing to report yet")
        return
    rated["new_score"] = rated["new_score"].astype(int)

    manifest = pd.read_parquet(args.manifest, columns=["post_id", "personal_score"])
    merged = rated.merge(manifest, on="post_id", how="inner")
    if len(merged) < len(rated):
        print(f"warning: {len(rated) - len(merged)} sheet rows have no manifest match (relabelled away?)")

    r = agreement_report(merged["personal_score"], merged["new_score"])
    print(f"rated={len(merged)}/{len(sheet)}")
    print(f"  test-retest spearman : {r['spearman']:.4f}")
    print(f"  test-retest qwk      : {r['qwk']:.4f}")
    print(f"  exact agreement      : {r['exact']:.1%}")
    print(f"  mae                  : {r['mae']:.3f}")
    print(f"\n  implied model ceiling (sqrt reliability): spearman ~{r['ceiling_spearman']:.3f}")
    print("  -> if current test spearman is already near this ceiling, invest in relabelling, not modelling")


def main() -> None:
    ap = argparse.ArgumentParser(description="Blind re-rating sheet + intra-rater reliability report.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("sample", help="export a blind re-rating sheet")
    sp.add_argument("--manifest", default="data/manifest.parquet")
    sp.add_argument("--n", type=int, default=150)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--sheet", default=DEFAULT_SHEET)
    sp.set_defaults(fn=cmd_sample)

    rp = sub.add_parser("report", help="score the filled sheet against the manifest")
    rp.add_argument("--manifest", default="data/manifest.parquet")
    rp.add_argument("--sheet", default=DEFAULT_SHEET)
    rp.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
