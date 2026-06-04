"""Generate a single-file, keyboard-driven review page for fast relabelling in pictoria.

pictoria has no "browse this list of ids" view, but its API has everything a relabel
sprint needs: images by id and a score endpoint (CORS is open). This script turns a queue
CSV (``oof_audit.py --csv`` / ``coverage_probe.py --csv`` output, or the intra-rater
sheet) into one local HTML file: big image, press 1~5 to score, auto-advance. Two modes:

  - default (relabel): shows the audit context (your old score, OOF, split) and PUTs each
    rating straight into pictoria (``PUT /posts/{id}/score``). Undo re-PUTs the old score.
  - ``--blind`` (intra-rater): hides every context column, writes NOTHING to pictoria;
    ratings accumulate in the page and export as a ``post_id,new_score`` CSV that
    ``intra_rater.py report`` consumes directly.

Progress persists in localStorage (keyed by queue content), so the sprint survives a
browser restart. Requires the pictoria backend running (default :4777).

    uv run python scripts/review_page.py --csv data/oof_queue.csv --limit 1100
    uv run python scripts/review_page.py --csv data/intra_rater_sheet.csv --blind --out blind.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CONTEXT_COLUMNS = ("personal_score", "oof", "gap", "split")  # shown (and used for undo) in relabel mode


def build_rows(df: pd.DataFrame, blind: bool) -> list[dict]:
    """Queue rows for the page: post_id always; context columns only when not blind."""
    rows = []
    for _, r in df.iterrows():
        row: dict = {"id": int(r["post_id"])}
        if not blind:
            for col in CONTEXT_COLUMNS:
                if col in df.columns and pd.notna(r[col]):
                    row[col] = round(float(r[col]), 2) if isinstance(r[col], float) else r[col]
        rows.append(row)
    return rows


def render_html(rows: list[dict], api: str, front: str, blind: bool, title: str) -> str:
    payload = json.dumps(rows, ensure_ascii=False)
    mode = "blind" if blind else "relabel"
    return (
        HTML_TEMPLATE.replace("__TITLE__", title)
        .replace("__MODE__", mode)
        .replace("__API__", api.rstrip("/"))
        .replace("__FRONT__", front.rstrip("/"))
        .replace("__ROWS__", payload)
    )


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,340;9..144,500&family=IBM+Plex+Mono:wght@400;500&display=swap');
  :root {
    --bg: #131211;          /* darkroom charcoal */
    --panel: #1b1916;
    --ink: #e8e2d6;
    --faint: #756d5f;
    --line: rgba(232, 226, 214, 0.09);
    --amber: #d9a050;       /* safelight */
    --green: #9fb87a;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    background: var(--bg);
    color: var(--ink);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    overflow: hidden;
    user-select: none;
  }
  /* hairline progress, top of screen */
  #bar { position: fixed; top: 0; left: 0; height: 2px; background: var(--amber); width: 0%; transition: width 0.25s ease; z-index: 5; }

  #stage {
    position: absolute; inset: 0 0 64px 0;
    display: flex; align-items: center; justify-content: center;
    padding: 28px;
  }
  #stage img {
    max-width: 100%; max-height: 100%;
    box-shadow: 0 18px 60px rgba(0, 0, 0, 0.55);
    background: var(--panel);
  }
  /* stamped score feedback */
  #stamp {
    position: absolute; inset: 0 0 64px 0;
    display: flex; align-items: center; justify-content: center;
    pointer-events: none;
    font-family: 'Fraunces', serif; font-weight: 340;
    font-size: 9rem; color: var(--amber);
    opacity: 0; transform: scale(0.92);
  }
  #stamp.show { animation: stamp 0.55s ease both; }
  @keyframes stamp {
    0% { opacity: 0; transform: scale(1.25); }
    25% { opacity: 0.95; transform: scale(1); }
    100% { opacity: 0; transform: scale(0.92); }
  }

  /* bottom chrome */
  #chrome {
    position: fixed; left: 0; right: 0; bottom: 0; height: 64px;
    display: flex; align-items: center; gap: 28px;
    padding: 0 24px;
    border-top: 1px solid var(--line);
    background: color-mix(in srgb, var(--bg) 88%, black);
    font-variant-numeric: tabular-nums;
  }
  .label { color: var(--faint); font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase; }
  .ctx b { color: var(--amber); font-weight: 500; }
  #pos { min-width: 110px; }
  #keys { margin-left: auto; color: var(--faint); font-size: 11px; }
  #keys b { color: var(--ink); font-weight: 500; }
  #toast {
    position: fixed; bottom: 84px; left: 50%; transform: translateX(-50%) translateY(8px);
    padding: 8px 16px; border: 1px solid var(--line); background: var(--panel);
    color: var(--ink); opacity: 0; transition: all 0.25s ease; z-index: 9;
  }
  #toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
  #toast.err { border-color: #b3543f; color: #e2937f; }
  #export {
    font: inherit; color: var(--bg); background: var(--amber);
    border: 0; padding: 7px 16px; cursor: pointer; letter-spacing: 0.06em;
  }
  #export:hover { filter: brightness(1.1); }
  #done {
    position: absolute; inset: 0 0 64px 0; display: none;
    align-items: center; justify-content: center; flex-direction: column; gap: 14px;
    font-family: 'Fraunces', serif; font-size: 2.2rem; color: var(--ink);
  }
  #done .sub { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--faint); }
  .scored { color: var(--green); }
</style>
</head>
<body>
<div id="bar"></div>
<div id="stage"><img id="img" alt=""></div>
<div id="stamp"></div>
<div id="done"><div>queue clear</div><div class="sub" id="doneSub"></div></div>
<div id="chrome">
  <span id="pos"></span>
  <span class="ctx" id="ctx"></span>
  <span id="state"></span>
  <span id="keys"><b>1–5</b> score &nbsp; <b>←→</b> move &nbsp; <b>u</b> undo &nbsp; <b>o</b> open</span>
</div>
<div id="toast"></div>
<script>
const MODE = "__MODE__";            // "relabel" | "blind"
const API = "__API__";
const FRONT = "__FRONT__";
const ROWS = __ROWS__;

const KEY = "silva-review-" + MODE + "-" + ROWS.length + "-" + (ROWS[0]?.id ?? 0) + "-" + (ROWS[ROWS.length-1]?.id ?? 0);
const saved = JSON.parse(localStorage.getItem(KEY) || "{}");
let ratings = saved.ratings || {};   // id -> score
let idx = saved.idx ?? 0;

const $ = (s) => document.querySelector(s);
const img = $("#img");

function persist() { localStorage.setItem(KEY, JSON.stringify({ ratings, idx })); }
function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "show" + (err ? " err" : "");
  clearTimeout(t._h);
  t._h = setTimeout(() => (t.className = ""), 1600);
}
function stamp(n) {
  const s = $("#stamp");
  s.textContent = n;
  s.className = "";
  void s.offsetWidth;            // restart animation
  s.className = "show";
}
function preload(i) {
  for (let d = 1; d <= 2; d++) {
    const r = ROWS[i + d];
    if (r) new Image().src = API + "/images/original/id/" + r.id;
  }
}
function render() {
  const total = ROWS.length, nScored = Object.keys(ratings).length;
  $("#bar").style.width = (100 * nScored / total) + "%";
  if (idx >= total) {
    $("#stage").style.display = "none";
    $("#done").style.display = "flex";
    $("#doneSub").textContent = nScored + " / " + total + " scored" + (MODE === "blind" ? " - export below" : "");
    $("#pos").textContent = total + " / " + total;
    $("#ctx").textContent = ""; $("#state").textContent = "";
    return;
  }
  $("#stage").style.display = "flex";
  $("#done").style.display = "none";
  const row = ROWS[idx];
  img.src = API + "/images/original/id/" + row.id;
  $("#pos").innerHTML = `<span class="label">queue</span> ${idx + 1} / ${total}`;
  let ctx = "";
  if (MODE === "relabel") {
    if (row.personal_score != null) ctx += `you <b>${row.personal_score}</b>`;
    if (row.oof != null) ctx += ` &nbsp;oof <b>${row.oof}</b>`;
    if (row.split) ctx += ` &nbsp;<span class="label">${row.split}</span>`;
  }
  $("#ctx").innerHTML = ctx;
  $("#state").innerHTML = ratings[row.id] != null ? `<span class="scored">scored ${ratings[row.id]}</span>` : "";
  preload(idx);
}
async function putScore(id, score) {
  const res = await fetch(API + "/posts/" + id + "/score", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ score }),
  });
  if (!res.ok) throw new Error("HTTP " + res.status);
}
async function rate(n) {
  const row = ROWS[idx];
  if (!row) return;
  if (MODE === "relabel") {
    try { await putScore(row.id, n); }
    catch (e) { toast("write failed: " + e.message + " - is pictoria running?", true); return; }
  }
  ratings[row.id] = n;
  stamp(n);
  idx++;
  persist();
  render();
}
async function undo() {
  const i = idx > 0 ? idx - 1 : idx;
  const row = ROWS[i];
  if (!row || ratings[row.id] == null) { toast("nothing to undo"); return; }
  if (MODE === "relabel") {
    const back = row.personal_score;
    if (back != null) {
      try { await putScore(row.id, back); }
      catch (e) { toast("undo write failed: " + e.message, true); return; }
    }
  }
  delete ratings[row.id];
  idx = i;
  persist();
  render();
  toast("undone");
}
function exportCsv() {
  const lines = ["post_id,new_score"];
  for (const r of ROWS) lines.push(r.id + "," + (ratings[r.id] ?? ""));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\\n")], { type: "text/csv" }));
  a.download = "intra_rater_filled.csv";
  a.click();
  toast("exported " + Object.keys(ratings).length + " ratings");
}
if (MODE === "blind") {
  const btn = document.createElement("button");
  btn.id = "export";
  btn.textContent = "export csv";
  btn.onclick = exportCsv;
  $("#keys").before(btn);
  window.addEventListener("beforeunload", (e) => {
    if (Object.keys(ratings).length) e.preventDefault();
  });
}
document.addEventListener("keydown", (e) => {
  if (e.repeat) return;
  if (e.key >= "1" && e.key <= "5") rate(Number(e.key));
  else if (e.key === "ArrowRight") { idx = Math.min(idx + 1, ROWS.length); persist(); render(); }
  else if (e.key === "ArrowLeft") { idx = Math.max(idx - 1, 0); persist(); render(); }
  else if (e.key === "u") undo();
  else if (e.key === "o") window.open(FRONT + "/post/" + ROWS[Math.min(idx, ROWS.length - 1)].id, "_blank");
});
render();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a keyboard-driven relabel/blind-rating page from a queue CSV.")
    ap.add_argument("--csv", required=True, help="queue CSV with a post_id column (oof_audit/coverage_probe/intra_rater output)")
    ap.add_argument("--out", default="review.html")
    ap.add_argument("--limit", type=int, default=None, help="cap the queue to the first N rows")
    ap.add_argument("--blind", action="store_true", help="hide context and write nothing; export a post_id,new_score CSV instead")
    ap.add_argument("--api", default="http://localhost:4777/v2", help="pictoria backend (incl. the /v2 mount)")
    ap.add_argument("--front", default="http://localhost:4778", help="pictoria web UI (the 'o' shortcut)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "post_id" not in df.columns:
        raise SystemExit(f"{args.csv} has no post_id column")
    if args.limit:
        df = df.head(args.limit)

    rows = build_rows(df, blind=args.blind)
    title = ("SILVA blind re-rate" if args.blind else "SILVA relabel") + f" - {Path(args.csv).name}"
    Path(args.out).write_text(render_html(rows, args.api, args.front, args.blind, title), encoding="utf-8")
    mode = "blind (no writes, export csv)" if args.blind else f"relabel (writes to {args.api})"
    print(f"{len(rows)} rows -> {args.out}  [{mode}]")
    print("open it in a browser; keys: 1-5 score, arrows move, u undo, o open in pictoria")


if __name__ == "__main__":
    main()
