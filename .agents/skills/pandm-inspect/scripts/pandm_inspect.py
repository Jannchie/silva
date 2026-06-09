#!/usr/bin/env python3
"""Machine-readable (JSON) views into a pandm data directory.

The pandm CLI (`pandm ls` / `pandm show`) prints Rich tables for humans; this
prints clean JSON for a program/LLM to parse. It reads the same SQLite store
through pandm's own LocalStore, so status (running->crashed) and per-metric
summaries match the dashboard exactly.

    python pandm_inspect.py runs                    # all runs, newest first
    python pandm_inspect.py runs --project mnist    # filter by project
    python pandm_inspect.py show <run_id>           # config, summary, metric keys, media
    python pandm_inspect.py series <run_id>         # full series for every metric
    python pandm_inspect.py series <run_id> -k train/loss -k val/loss
    python pandm_inspect.py compare <id1> <id2> …   # config + summary side by side
    python pandm_inspect.py media <run_id>          # logged images with absolute paths

Pass --dir to target a specific store (default: ./.pandm or $PANDM_DIR).
"""

from __future__ import annotations

import argparse
import json
import sys

try:
    from pandm.storage import LocalStore, resolve_dir
except ImportError:
    sys.exit("pandm is not installed in this environment — run `pip install pandm` first.")


def _emit(obj: object) -> None:
    json.dump(obj, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def _require(store: "LocalStore", run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        sys.exit(f"run {run_id} not found in {store.root}")
    return run


def cmd_runs(store: "LocalStore", args: argparse.Namespace) -> None:
    _emit(
        [
            {
                "id": r["id"],
                "name": r["name"],
                "project": r["project"],
                "status": r["status"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "finished_at": r["finished_at"],
                "progress": r["progress"],
                "progress_total": r["progress_total"],
                "config": r["config"],
                "summary": r["summary"],  # latest value per metric key
            }
            for r in store.list_runs(args.project)
        ]
    )


def cmd_show(store: "LocalStore", args: argparse.Namespace) -> None:
    run = _require(store, args.run_id)
    run["metric_keys"] = store.metric_keys(args.run_id)  # [{key, points, last_step}]
    run["media"] = store.list_media(args.run_id)
    _emit(run)


def cmd_series(store: "LocalStore", args: argparse.Namespace) -> None:
    _require(store, args.run_id)
    keys = args.key or [k["key"] for k in store.metric_keys(args.run_id)]
    _emit({k: store.metric_series(args.run_id, k, max_points=args.max_points) for k in keys})


def cmd_compare(store: "LocalStore", args: argparse.Namespace) -> None:
    runs = [_require(store, rid) for rid in args.run_ids]
    config_keys = sorted({k for r in runs for k in r["config"]})
    metric_keys = sorted({k for r in runs for k in r["summary"]})
    _emit(
        {
            "runs": [
                {"id": r["id"], "name": r["name"], "project": r["project"], "status": r["status"]}
                for r in runs
            ],
            # each row: a config/metric key -> one value per run, in the same order as "runs"
            "config": {k: [r["config"].get(k) for r in runs] for k in config_keys},
            "summary": {k: [r["summary"].get(k) for r in runs] for k in metric_keys},
        }
    )


def cmd_media(store: "LocalStore", args: argparse.Namespace) -> None:
    run = _require(store, args.run_id)
    items = store.list_media(args.run_id)
    for m in items:
        path = store.media_path(args.run_id, m["filename"])
        m["path"] = str(path) if path else None
    _emit({"run_id": run["id"], "name": run["name"], "media": items})


def main() -> None:
    # --dir lives on a shared parent so it is accepted in any position after the
    # subcommand too (e.g. `runs --dir X`), not only before it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dir", help="pandm data directory (default: ./.pandm or $PANDM_DIR)")

    parser = argparse.ArgumentParser(description="JSON views into a pandm data directory.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("runs", parents=[common], help="list runs as JSON (newest first)")
    p.add_argument("--project", "-P", help="filter to one project")
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("show", parents=[common], help="one run's config, summary, metric keys, media")
    p.add_argument("run_id")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("series", parents=[common], help="full metric series for a run")
    p.add_argument("run_id")
    p.add_argument("--key", "-k", action="append", help="metric key (repeatable; default: all keys)")
    p.add_argument("--max-points", type=int, default=2**31, help="downsample target (default: no downsampling)")
    p.set_defaults(func=cmd_series)

    p = sub.add_parser("compare", parents=[common], help="config + summary across several runs")
    p.add_argument("run_ids", nargs="+")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("media", parents=[common], help="logged images with absolute file paths")
    p.add_argument("run_id")
    p.set_defaults(func=cmd_media)

    args = parser.parse_args()
    store = LocalStore(resolve_dir(args.dir))
    args.func(store, args)


if __name__ == "__main__":
    main()
