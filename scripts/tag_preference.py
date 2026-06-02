"""Tag-level aesthetic preference of the SILVA library scores, de-confounded by artist.

Marginal tag means conflate a tag with the artist who tends to use it (hair colour <->
character <-> character popularity <-> artist skill <-> image quality). A raw "streaked
hair scores high" may just be "skilled artists draw streaked hair".

This reports, per tag, BOTH:
  - raw_diff:    mean(silva | tag) - global mean         (the marginal effect)
  - within_diff: mean(silva - that-image's-artist-mean | tag)   (artist-deconfounded)

within_diff strips each artist's skill baseline: an effect that SURVIVES the residual is
intrinsic to the tag; one that COLLAPSES to ~0 was an artist artefact. Only images with
exactly one artist tag (group_id=3) and artists with >= --min-artist-posts works are used,
so the within-artist mean is meaningful.

    uv run python scripts/tag_preference.py
    uv run python scripts/tag_preference.py --min-artist-posts 10 --min-tag-n 200
"""

from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

DEFAULT_DB = r"E:/pictoria/server/illustration/images/.pictoria/pictoria.sqlite"
SCORER = "silva"
ARTIST_GROUP_ID = 3

DIMENSIONS: dict[str, list[str]] = {
    "HAIR COLOR": [
        "blonde_hair", "brown_hair", "black_hair", "blue_hair", "purple_hair", "pink_hair",
        "red_hair", "white_hair", "silver_hair", "grey_hair", "green_hair", "orange_hair",
        "aqua_hair", "light_brown_hair", "multicolored_hair", "two-tone_hair",
        "gradient_hair", "streaked_hair",
    ],
    "FRAMING / SHOT": [
        "portrait", "close-up", "upper_body", "cowboy_shot", "full_body", "lower_body",
        "feet_out_of_frame", "wide_shot",
    ],
    "CAMERA ANGLE": [
        "from_above", "from_below", "from_side", "from_behind", "dutch_angle", "straight-on",
    ],
    "EYE COLOR": [
        "blue_eyes", "brown_eyes", "red_eyes", "green_eyes", "purple_eyes", "yellow_eyes",
        "pink_eyes", "aqua_eyes", "orange_eyes", "grey_eyes", "heterochromia",
    ],
}


def _verdict(raw: float, within: float) -> str:
    """How much of the marginal tag effect survives artist de-confounding."""
    if abs(raw) < 0.008:
        return "~ (flat)"
    ratio = within / raw  # same sign & magnitude -> intrinsic; ~0 -> artist artefact
    if ratio < 0.25:
        return "ARTIST (collapses)"
    if ratio < 0.6:
        return "partial"
    if ratio > 1.25:
        return "intrinsic (amplified)"
    return "intrinsic"


def load_eligible(con: sqlite3.Connection, min_artist_posts: int) -> tuple[pd.DataFrame, float]:
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
    df["artist_mean"] = df.groupby("artist").score.transform("mean")
    df["resid"] = df.score - df.artist_mean
    return df, float(df.score.mean())


def analyze_group(con: sqlite3.Connection, df: pd.DataFrame, sub_global: float, group_id: int, min_tag_n: int, show: int) -> None:
    """De-confounded preference over a whole tag group (character / copyright / ...).

    Auto-selects the group's tags with >= min_tag_n eligible images, ranks by within-artist
    effect, and prints the top/bottom `show`. n_art = distinct artists (low -> within is noisy:
    the tag barely varies across artists, so de-confounding can't separate it from one studio).
    """
    gname = con.execute("SELECT name FROM tag_groups WHERE id=?", (group_id,)).fetchone()[0]
    tagmap = pd.read_sql_query(
        "SELECT pt.post_id, pt.tag_name FROM post_has_tag pt JOIN tags t ON t.name = pt.tag_name WHERE t.group_id = ?",
        con, params=(group_id,),
    )
    tagmap = tagmap[tagmap.post_id.isin(df.index)]
    rows = []
    for tag, g in tagmap.groupby("tag_name"):
        sub = df.loc[df.index.intersection(g["post_id"].to_numpy())]
        if len(sub) < min_tag_n:
            continue
        rows.append((tag, len(sub), sub.artist.nunique(),
                     float(sub.score.mean() - sub_global), float(sub.resid.mean())))
    rows.sort(key=lambda r: r[4], reverse=True)
    print(f"=== {gname.upper()} (group {group_id}): {len(rows)} tags with >={min_tag_n} imgs, by within-artist effect ===")
    print(f"  {'tag':<40} {'n':>6} {'n_art':>5}  {'raw':>7}  {'within':>7}   verdict")

    def line(r: tuple) -> str:
        tag, n, na, raw, within = r
        return f"  {tag[:40]:<40} {n:>6} {na:>5}  {raw:+.3f}  {within:+.3f}   {_verdict(raw, within)}"

    if len(rows) <= 2 * show:  # few enough to list in full without top/bottom overlap
        for r in rows:
            print(line(r))
    else:
        print(f"  -- most preferred (top {show}) --")
        for r in rows[:show]:
            print(line(r))
        print(f"  -- most penalised (bottom {show}) --")
        for r in rows[-show:]:
            print(line(r))
    print()


def analyze_named(con: sqlite3.Connection, df: pd.DataFrame, sub_global: float, min_tag_n: int) -> None:
    """De-confounded preference over the hand-picked semantic DIMENSIONS (hair/eye/framing/angle)."""
    all_tags = sorted({t for tags in DIMENSIONS.values() for t in tags})
    placeholders = ",".join("?" * len(all_tags))  # bound params, not values — safe
    tagmap = pd.read_sql_query(
        f"SELECT post_id, tag_name FROM post_has_tag WHERE tag_name IN ({placeholders})",  # noqa: S608
        con, params=all_tags,
    )
    tag_to_posts = {t: g["post_id"].to_numpy() for t, g in tagmap.groupby("tag_name")}
    for dim, tags in DIMENSIONS.items():
        rows = []
        for t in tags:
            ids = tag_to_posts.get(t)
            if ids is None:
                continue
            sub = df.loc[df.index.intersection(ids)]
            if len(sub) < min_tag_n:
                continue
            raw = float(sub.score.mean() - sub_global)
            within = float(sub.resid.mean())
            rows.append((t, len(sub), raw, within, _verdict(raw, within)))
        rows.sort(key=lambda r: r[3], reverse=True)  # sort by within-artist effect
        print(f"=== {dim}  (sorted by within-artist effect) ===")
        print(f"  {'tag':<20} {'n':>6}  {'raw':>7}  {'within':>7}   verdict")
        for t, n, raw, within, v in rows:
            print(f"  {t:<20} {n:>6}  {raw:+.3f}  {within:+.3f}   {v}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Artist-deconfounded tag preference of SILVA scores.")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--min-artist-posts", type=int, default=5, help="drop artists with fewer works (within-artist mean is noise)")
    ap.add_argument("--min-tag-n", type=int, default=100, help="drop tags with fewer scored images in the eligible set")
    ap.add_argument("--group-id", type=int, nargs="*", default=None, help="analyze whole tag group(s), e.g. 2=character 5=copyright (default: named dims)")
    ap.add_argument("--show", type=int, default=20, help="top/bottom N to print per group in --group-id mode")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    df, sub_global = load_eligible(con, args.min_artist_posts)
    print(f"eligible images: {len(df)}  (single-artist, artist with >={args.min_artist_posts} works)")
    print(f"artists: {df.artist.nunique()}   sub-global silva mean: {sub_global:.4f}\n")

    if args.group_id:
        for gid in args.group_id:
            analyze_group(con, df, sub_global, gid, args.min_tag_n, args.show)
    else:
        analyze_named(con, df, sub_global, args.min_tag_n)
    con.close()


if __name__ == "__main__":
    main()
