---
name: pandm-track
description: Record machine-learning experiment metrics, images, hyperparameters, and training progress to pandm — a local-first, account-free wandb/tensorboard alternative that writes to a `.pandm/` SQLite + PNG store. Use when instrumenting a training or evaluation loop, logging scalar metrics or images from a Python script, reporting an ETA, or saving a run's config so it can later be compared in the pandm dashboard.
---

# Recording experiments with pandm

pandm tracks ML runs locally: `pandm.init()` starts a run, `run.log()` writes
scalar metrics, `run.log_image()` writes images. Everything lands in `.pandm/`
next to the script (plain SQLite + PNG, no account, no daemon). The same code
reports to a shared server when one env var is set — see *Modes* below.

Requires the package: `pip install pandm` (import name is `pandm`).

## Minimal loop

```python
import pandm

run = pandm.init(
    project="mnist",
    name="baseline",                       # optional; a random name is generated if omitted
    config={"lr": 1e-3, "batch_size": 64}, # hyperparameters — anything JSON-able
    total_steps=1000,                      # optional; lets the dashboard show an ETA
)
for step in range(1000):
    loss, acc = train_step()
    run.log({"train/loss": loss, "train/acc": acc}, step=step)
    if step % 100 == 0:
        run.log_image("samples", sample_grid, step=step, caption=f"step {step}")
run.finish()
```

Prefer the context manager — it finishes the run (and marks it `crashed` on an
exception) even if the loop raises:

```python
with pandm.init(project="mnist", config={"lr": 1e-3}) as run:
    run.log({"loss": 0.5})
```

## API

| Call | Purpose |
|---|---|
| `pandm.init(project="default", name=None, config=None, *, total_steps=None, directory=None, remote=None, api_key=None)` | Start a run; returns a `Run` (also a context manager). |
| `run.log(metrics: dict, step=None)` | Log scalar metrics. `step` defaults to an internal per-run counter. |
| `run.log_image(key, image, step=None, caption=None)` | Log one image. `step` defaults to the latest metric step. |
| `run.set_progress(current, total=None)` | Report progress in a custom unit (epochs, samples) for the ETA. |
| `run.finish(status="finished")` | End the run. Also runs automatically at process exit. |

Module-level `pandm.log(...)`, `pandm.log_image(...)`, `pandm.set_progress(...)`,
and `pandm.finish(...)` act on the most recently started run — convenient when
passing the `run` object around is awkward.

## Behaviour that matters

- **`step` is optional.** Omit it and pandm uses a per-run counter that advances
  by one per `log()` call. If you pass `step`, keep it monotonic per key — the
  dashboard plots against it.
- **Group keys with `/`.** `train/loss`, `val/loss`, `lr` — the dashboard groups
  by the prefix before the slash.
- **NaN / Inf are dropped silently.** Guard or sanitize values you actually need;
  a metric that is sometimes non-finite will have gaps, not errors.
- **Images** accept a PIL `Image`, a numpy/torch array (HWC *or* CHW is detected;
  float arrays in `[0,1]` are auto-scaled to `[0,255]`), a file path, or raw PNG
  bytes. Don't pre-convert tensors — pass them through.
- **Run status** is `running` → `finished` or `crashed`. Uncaught exceptions
  (via `sys.excepthook`) and hard kills (`kill -9`, OOM — detected when the 15 s
  heartbeat goes quiet for 60 s) become `crashed`. Always reach `finish()` /
  exit the `with` block on the happy path so it lands as `finished`.
- **ETA:** pass `total_steps=` and progress follows your `log(step=...)`
  automatically; for other units call `run.set_progress(current, total)`.

## Modes (the training code never changes)

| Goal | How |
|---|---|
| Local only (default) | nothing — writes to `./.pandm`. |
| Different data dir | `PANDM_DIR=/path` env var, or `init(directory=...)`. |
| Local **and** a shared server | `pandm login <url>` once per machine → `init()` dual-writes and syncs in the background, backfilling anything logged offline. |
| Remote only (no local copy) | `PANDM_REMOTE=<url>` + `PANDM_API_KEY=...`, or `init(remote=..., api_key=...)`. |
| Force local even when signed in | `init(remote=False)` or `PANDM_NO_SYNC=1`. |

## Hugging Face Accelerate

Accelerate only resolves strings for its built-in trackers, so pass an instance:

```python
from accelerate import Accelerator
from pandm.integrations.accelerate import PandmTracker

accelerator = Accelerator(log_with=PandmTracker(project="mnist", name="baseline"))
accelerator.init_trackers("mnist", config={"lr": 1e-3})
accelerator.log({"loss": 0.42}, step=10)            # -> run.log
accelerator.end_training()                          # -> run.finish
# images: accelerator.get_tracker("pandm", unwrap=True).log_image("samples", img, step=step)
```

## Verify it landed

```sh
pandm ls                 # the new run shows up here
pandm show <run_id>      # config + per-metric last value
pandm ui                 # http://127.0.0.1:7878 — live charts
```

A full working example lives in the repo at `examples/train_demo.py`. To read
runs back programmatically, use the **pandm-inspect** skill.
