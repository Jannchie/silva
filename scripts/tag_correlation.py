"""Point-biserial correlation of every pictoria tag with the SILVA score, exported to CSV.

For a binary indicator (tag present / absent) vs a continuous score, the Pearson r reduces
to the point-biserial closed form  r = (mean_tag - mean_all) * sqrt(p / (1 - p)) / std,
where p is the tag's prevalence. Two coefficients are reported per tag:

  - r_raw:    correlation with the silva score itself (marginal association)
  - r_within: correlation with the within-artist residual (artist-deconfounded; the
              residual mean is 0 by construction, see tag_preference.py for rationale)

Same eligible set as tag_preference.py: single-artist images whose artist has enough works.
Artist tags (group 3) are excluded by default — they ARE the confound being removed.

    uv run python scripts/tag_correlation.py
    uv run python scripts/tag_correlation.py --groups 1 --min-tag-n 200 --out outputs/general_tags.csv
"""

from __future__ import annotations

import argparse
import math
import sqlite3

import pandas as pd

DEFAULT_DB = r"E:/pictoria/server/illustration/images/.pictoria/pictoria.sqlite"
DEFAULT_OUT = "outputs/tag_correlation.csv"
SCORER = "silva"
ARTIST_GROUP_ID = 3
DEFAULT_GROUPS = [1, 2, 4, 5]  # general, character, meta, copyright


def load_eligible(con: sqlite3.Connection, min_artist_posts: int) -> pd.DataFrame:
    """Scored images with an unambiguous, well-sampled artist + within-artist residual."""
    scores = pd.read_sql_query(
        "SELECT post_id, score FROM post_aesthetic_scores WHERE scorer=?", con, params=(SCORER,)
    ).set_index("post_id")
    artists = pd.read_sql_query(
        "SELECT pt.post_id, pt.tag_name AS artist FROM post_has_tag pt "
        "JOIN tags t ON t.name = pt.tag_name WHERE t.group_id = ?",
        con, params=(ARTIST_GROUP_ID,),
    )
    counts = artists.groupby("post_id").size()  # keep images with EXACTLY one artist tag
    single = counts[counts == 1].index
    artists = artists[artists.post_id.isin(single)].set_index("post_id")["artist"]

    df = scores.join(artists.rename("artist"), how="inner").dropna(subset=["artist"])
    ac = df.groupby("artist").size()
    df = df[df.artist.isin(ac[ac >= min_artist_posts].index)]
    df["resid"] = df.score - df.groupby("artist").score.transform("mean")
    return df


def correlate_group(con: sqlite3.Connection, df: pd.DataFrame, group_id: int, min_tag_n: int) -> pd.DataFrame:
    """Point-biserial r of each tag in the group against score and within-artist residual."""
    gname = con.execute("SELECT name FROM tag_groups WHERE id=?", (group_id,)).fetchone()[0]
    tagmap = pd.read_sql_query(
        "SELECT pt.post_id, pt.tag_name FROM post_has_tag pt JOIN tags t ON t.name = pt.tag_name WHERE t.group_id = ?",
        con, params=(group_id,),
    )
    tagged = tagmap.join(df, on="post_id", how="inner")

    n_all, mean_all = len(df), float(df.score.mean())
    std_score, std_resid = float(df.score.std(ddof=0)), float(df.resid.std(ddof=0))

    agg = tagged.groupby("tag_name").agg(
        n=("score", "size"), n_artists=("artist", "nunique"),
        mean_score=("score", "mean"), mean_resid=("resid", "mean"),
    )
    agg = agg[agg.n >= min_tag_n].reset_index().rename(columns={"tag_name": "tag"})

    p = agg.n / n_all
    odds = (p / (1 - p)).map(math.sqrt)
    agg["diff_raw"] = agg.mean_score - mean_all
    agg["diff_within"] = agg.mean_resid  # residual global mean is 0 by construction
    agg["r_raw"] = agg.diff_raw * odds / std_score
    agg["r_within"] = agg.diff_within * odds / std_resid
    agg["group"] = gname
    return agg[["tag", "group", "n", "n_artists", "r_raw", "r_within", "diff_raw", "diff_within"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export point-biserial tag/score correlations to CSV.")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--groups", type=int, nargs="*", default=DEFAULT_GROUPS, help="tag group ids (default: all but artist)")
    ap.add_argument("--min-artist-posts", type=int, default=5, help="drop artists with fewer works (within-artist mean is noise)")
    ap.add_argument("--min-tag-n", type=int, default=100, help="drop tags with fewer scored images in the eligible set")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    df = load_eligible(con, args.min_artist_posts)
    print(f"eligible images: {len(df)}   artists: {df.artist.nunique()}   mean: {df.score.mean():.4f}")

    out = pd.concat([correlate_group(con, df, gid, args.min_tag_n) for gid in args.groups])
    con.close()

    out = out.sort_values("r_raw", ascending=False).round(4)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} tags -> {args.out}\n")

    show = out[["tag", "group", "n", "r_raw", "r_within"]]
    print("=== strongest positive (top 15 by r_raw) ===")
    print(show.head(15).to_string(index=False))
    print("\n=== strongest negative (bottom 15 by r_raw) ===")
    print(show.tail(15).to_string(index=False))


if __name__ == "__main__":
    main()
