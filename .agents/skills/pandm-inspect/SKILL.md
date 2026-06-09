---
name: pandm-inspect
description: Query and analyze machine-learning experiments tracked by pandm — list runs, read a run's config/summary/metrics, compare runs to find the best hyperparameters, read full metric series, and locate logged images. Use when the user asks about past pandm experiments, wants to compare runs, pick a winner, inspect a metric over time, or analyze results stored in a `.pandm/` directory.
---

# Inspecting pandm experiments

pandm stores every run as plain SQLite + PNG under a `.pandm/` directory
(default `./.pandm`, overridable with `--dir` or `$PANDM_DIR`):

```
.pandm/
  pandm.db          # runs, metrics, media metadata (SQLite, WAL mode)
  media/<run_id>/   # the actual PNG files
```

There are three ways in, cheapest first. Prefer the **JSON helper** when an
answer needs the data parsed; fall back to the **CLI** for a quick human look,
and to **raw SQL** for aggregations the helper doesn't cover.

## 1. Structured JSON — `scripts/pandm_inspect.py` (preferred)

The built-in CLI prints Rich tables (with color codes) meant for human eyes.
This helper prints clean JSON from the same store, so values are parseable.

```sh
python scripts/pandm_inspect.py runs                   # all runs, newest first
python scripts/pandm_inspect.py runs --project mnist   # filter by project
python scripts/pandm_inspect.py show <run_id>          # config, summary, metric keys, media
python scripts/pandm_inspect.py series <run_id>        # every metric's full series
python scripts/pandm_inspect.py series <run_id> -k train/loss -k val/loss
python scripts/pandm_inspect.py compare <id1> <id2> …  # config + summary side by side
python scripts/pandm_inspect.py media <run_id>         # logged images with absolute paths
```

Add `--dir /path/to/.pandm` to any command to point at a non-default store.
`runs` / `show` / `compare` carry each run's `summary` (the latest value per
metric) — that's what you compare to pick a winner without reading full series.
For images, `media` returns absolute file paths you can then open/Read directly.

## 2. CLI — quick human-readable look

```sh
pandm ls                          # table of runs
pandm show <run_id>               # config, summary, metric keys
pandm export <run_id> --json      # full series as JSON (one run, metrics only)
pandm export <run_id> -k train/loss > loss.csv
```

## 3. Raw SQL — custom aggregations

```sh
sqlite3 .pandm/pandm.db "SELECT id, name, project, status FROM runs ORDER BY created_at DESC"
```

Schema essentials:

- `runs(id, project, name, status, config /*JSON*/, created_at, updated_at,
  finished_at, progress, progress_total, progress_ts)`
- `metrics(run_id, key, step, value, ts)` — one row per logged scalar
- `media(run_id, key, step, filename, caption, ts)` — file is `media/<run_id>/<filename>`

"Best value of `val/acc`" across runs, straight from SQL:

```sql
SELECT run_id, MAX(value) AS best FROM metrics
WHERE key = 'val/acc' GROUP BY run_id ORDER BY best DESC;
```

## Semantics to keep in mind

- **`summary[key]`** is the *latest* logged value for that key (max step), not
  the best — take a max/min over `series` or SQL when you want the extremum.
- **`status`** is computed on read: a `running` run whose heartbeat has been
  quiet for >60 s is reported as `crashed` (self-heals if the process resumes).
  So a run can flip to `crashed` between two reads — don't cache it.
- **`progress` / `progress_total`** drive the dashboard ETA; either may be null.
- The DB is in WAL mode, so reading while a training run is live is safe and
  returns committed rows.
