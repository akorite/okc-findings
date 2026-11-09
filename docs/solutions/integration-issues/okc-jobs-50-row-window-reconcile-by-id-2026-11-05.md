---
title: "`okc jobs` is a 50-row window, not a history: reconcile judge jobs by id"
date: 2026-11-05
category: integration-issues
module: "tools/reconcile_all.py"
problem_type: integration_issue
component: tooling
severity: high
symptoms:
  - "Watcher scripts reported the judge queue as stalled or empty for hours"
  - "`okc jobs --json` returned the same ~50 recent dispatches every tick, so no completion ever surfaced"
  - "The session concluded nothing had landed all day while 72 of its own layer jobs had already finished"
  - "`artifacts/bench/ingest_watch.md` stopped accumulating landing rows while the judge kept scoring"
root_cause: wrong_api
resolution_type: tooling_addition
related_components:
  - "development_workflow"
tags:
  - "okc-cli"
  - "job-reconciliation"
  - "silent-truncation"
  - "api-pagination"
  - "judge-queue"
  - "observability-gap"
  - "id-based-polling"
  - "dedup"
---

# `okc jobs` is a 50-row window, not a history: reconcile judge jobs by id

## Problem

The long-lived loop watched the OKC judge through `okc jobs --json` and diffed that list against a previous
snapshot. `okc jobs` returns only the **50 most recent** dispatches. Once ~50 component-board checks fired
inside one hour, every earlier job fell out of the list and both watchers went blind: finished results were
not delayed, they were **unread**. Hours went into reasoning about a congested judge queue while 72 of our own
layer jobs had already completed and were only ever recovered by writing a new tool.

## Symptoms

- `tools/queue_watch.py` and `tools/ingest_watch.sh` both reported the queue as stalled or empty for hours
  (`docs/PROGRESS.md:1698-1701`).
- `okc jobs --json` returned the same ~50 newest dispatches on every tick, so the watcher's state-change
  branch never fired (`tools/queue_watch.py:59,78-89`).
- `artifacts/bench/ingest_watch.md` stopped accumulating landing rows while the judge kept scoring
  (`docs/PROGRESS.md:1700-1702`).
- The working conclusion was "0 jobs landed all day", later contradicted by 72 completed layer jobs
  (`docs/PROGRESS.md:1701`).

## What Didn't Work

- **Diffing a shrinking list.** `queue_watch.py` rebuilt `seen = {j["job_id"]: j for j in jobs}` every 120 s
  from the 50-item response (`tools/queue_watch.py:89,29`). When a job aged out of the window and reappeared
  already finished, the `prev is None` branch explicitly skipped it — the code comment is literal:
  `pass  # finished before watch started; snapshot silently` (`tools/queue_watch.py:72-77`). A completion that
  happened between ticks while the id sat outside the window was unloggable by construction, and the status
  cache refreshed only ids still present in the current window (`:91-99`).
- **Blaming the judge.** The public queue endpoint did show a real global backlog — 636 with "only 1 of 18
  screen checks even running" at 00:00 UTC (`docs/PROGRESS.md:1729`). That congestion was real *and
  irrelevant*: our results had already finished and were sitting unread.
- **Assuming the snapshot was history.** `artifacts/bench/jobs_snapshot_0604.json` holds exactly **50** rows
  (verified: `len(...) == 50`). The truncation had been persisted into the snapshot, so the "prior state" the
  watcher diffed against was itself a window. A snapshot whose row count equals the documented cap is
  truncated, not complete.
- **Overriding `dedup_guard` on a same-mode theory.** The guard flagged `attn/plan.py` and `bf16/geo.py` as
  canonical twins of `draw_plan_C.py` / `draw_geo_C.py`. The session argued the twins had only ever been
  *checked*, that judge dedup is same-mode, and dispatched anyway. Both returned `duplicate`
  (`artifacts/bench/ingest_watch.md:192-193`, jobs `292f40e558954632` and `d00bcc67142d4454` at 04:50:49Z).
  Cost: 2 serial slots, 0 runs (`docs/PROGRESS.md:1748-1753`). The push-through happened because
  `tools/submit_guarded.sh:10` already passes `--allow-banner-clone`, which downgrades the canonical check to
  `warn` and lets the wrapper's `grep -q "^BLOCK"` pass it — not because the guard was wrong.

## Solution

`tools/reconcile_all.py` ignores `okc jobs` entirely and reconciles **by id**. Every id ever seen is unioned
from three monotone sources, then each unknown or still-active id is polled individually.

```python
# tools/reconcile_all.py — verbatim excerpts, elisions marked with "..."
def load_known_ids() -> dict[str, str]:
    """Return {job_id: relative_file_path} from ~/.config/okc/jobs.json + snapshots + ledger."""
    id_to_file: dict[str, str] = {}
    cfg_jobs = Path(os.path.expanduser("~/.config/okc/jobs.json"))         # :23
    if cfg_jobs.exists():
        ...
    for snap in ("artifacts/bench/jobs_snapshot_0604.json",                # :33
                 "artifacts/bench/jobs_snapshot_latest.json"):
        ...
    if LEDGER_PATH.exists():
        for m in re.finditer(r"\b([0-9a-f]{16})\b", LEDGER_PATH.read_text(errors="ignore")):
            id_to_file.setdefault(m.group(1), "?")                          # :41-42
    return id_to_file


def fetch_one(jid: str) -> tuple[str, dict]:
    out = subprocess.check_output(["okc", "status", jid, "--json"],        # :50
                                   stderr=subprocess.DEVNULL, text=True, timeout=15)
    return jid, json.loads(out)


def main() -> int:
    ...
    to_query = [jid for jid in sorted(id_to_file)                           # :70-74
                if jid not in results or results[jid].get("status") in ("queued", "running")]

    with ThreadPoolExecutor(max_workers=8) as ex:                           # :78
        for jid, d in ex.map(fetch_one, to_query):                          # :79
            if d.get("status") != "err" and d.get("board"):
                prev_status = (results.get(jid) or {}).get("status")
                results[jid] = d
                (STATUS_DIR / f"{jid}.json").write_text(json.dumps(d))      # :83
                if d.get("status") not in ("queued", "running") \
                        and prev_status in ("queued", "running", None):      # :84-85
                    newly_finished.append((jid, d))

    RECON_PATH.write_text(json.dumps(results, indent=2))                    # :87
```

A first transition out of `queued`/`running` — or a first-ever sighting of an already-finished job — appends
exactly one row to both ledgers, so landings are written once and never re-logged:

```python
    if newly_finished:
        with INGEST_PATH.open("a") as ing, LEDGER_PATH.open("a") as led:     # :97
            ...
            ing.write(f"- {line}\n")                                        # :110
            led.write(f"| landed | {now_s} | ... | {jid} | {st} pass={passed} | {sp_s} | {us_s} | - |\n")
```

The watcher was then reduced to a timer around that script (`tools/ingest_watch.sh:11`, every 90 s until the
deadline at `:9`) — the file no longer mentions `okc jobs` at all:

```bash
.venv/bin/python tools/reconcile_all.py >> /tmp/reconcile_all.log 2>&1
```

Verified outcome: `artifacts/bench/all_jobs_reconciled.json` finished at **384 jobs** — 345 `done`,
37 `duplicate`, 2 `failed` — including **132 `muse_decode_layer` jobs** (81 `check`: 79 done / 1 dup /
1 failed; 51 `run`: 40 done / 10 dup / 1 failed), with one cached status file per id under
`artifacts/bench/job_statuses/`. Those recovered means, not the displayed draws, then drove every later
dispatch decision.

## Why This Works

- **`okc status <id>` is id-addressed, so it has no window.** Only the list endpoint truncates; a per-id query
  answers for a job however old it is (`tools/reconcile_all.py:47-55`).
- **The id set only grows.** It is the union of the CLI's own dispatch record (`~/.config/okc/jobs.json`,
  `:23`), the dated snapshots (`:33` — even the truncated 50-row one still contributes its 50 ids), and a
  16-hex-id regex over the human ledger (`:41-42`). No id can age out.
- **Re-querying is scoped and idempotent.** Only unknown or `queued`/`running` ids are fetched (`:70-74`);
  finished results come from the existing map, so the file is both cache and record, and a failed fetch never
  overwrites a good prior result (`:80-82`).
- **Landing rows are edge-triggered.** The `prev_status in ("queued","running",None)` guard (`:84-85`) means
  a first transition is the only thing that writes to a ledger — precisely the moment the old watcher dropped.

## Prevention

1. **Never treat a truncated list as a history.** A response whose length equals the documented cap is a
   window, not a census (`docs/PROGRESS.md:1744-1747`). This is how easy it is to persist the mistake:
   ```bash
   .venv/bin/python -c 'import json;print(len(json.load(open("artifacts/bench/jobs_snapshot_0604.json"))))'
   # 50  <- exactly the cap: truncated
   ```
2. **Persist ids monotonically, at dispatch time.** Append every `okc submit|check` job id to a local
   append-only file; never re-derive the id set by paging a list endpoint. `reconcile_all.py` already keeps
   three backstops (`:23`, `:33`, `:41-42`) — add to them, never rebuild them from `okc jobs`.
3. **Reconcile by id before saying anything about the queue.** One command makes the health signal explicit
   (`tools/reconcile_all.py:94`):
   ```bash
   .venv/bin/python tools/reconcile_all.py
   # [12:00:03Z] queried=184 newly_finished=31 active=9
   ```
   Non-zero `newly_finished` while a watcher claims "stalled" means the watcher is wrong, not the judge.
4. **Retire anything still diffing `okc jobs`.** `tools/queue_watch.py:59,89` still rebuilds `seen` from the
   50-row list — the exact code path that lost these results, and it still runs on a 120 s timer. The
   reconciled map is the source of truth now; a watcher that ignores `all_jobs_reconciled.json` will
   reproduce this bug.
5. **Never override `dedup_guard`'s block.** A body match is a run-block, *including* a run submitted against
   a payload that was only ever `check`ed (`docs/PROGRESS.md:1750-1752`;
   `artifacts/bench/ingest_watch.md:192-193`). If the twin is genuinely a different kernel, change the bytes
   — that is also what the scoring rules require. Two live footguns make the override tempting and both
   should be treated as blocks:
   - `tools/submit_guarded.sh:10` passes `--allow-banner-clone` on every guarded dispatch, so the canonical
     check prints `ok … warn:` and the wrapper's `grep -q "^BLOCK"` lets it through;
   - `tools/dedup_guard.py:163-169` still *returns* on a byte-exact resend of a check-mode job with a warning,
     and its comment encodes the same-mode theory that failed here ("Server dedup is same-mode only").

## Related Issues

- `docs/PROGRESS.md:1698-1706` — the instrument-repair narrative that first recorded this failure.
- `docs/PROGRESS.md:1742-1753` — "Two traps for whoever picks this up next" (this 50-row window, and the dedup
  cross-mode trap). That file's LIVE STATE block is rewritten at every milestone, so this document is the
  durable copy of the rule while that section is the incident record.
- `tools/queue_watch.py` — still on the truncated-list design; the known stale instrument (see Prevention #4).
- `tools/dedup_guard.py:163-169` — latent same-mode assumption in the byte-exact branch (Prevention #5).
- `tools/submit_guarded.sh:10` — downgrades the canonical check to a warning on every guarded submit.
- Known residual risk in the fix itself: `reconcile_all.py` silently drops errored `okc status` fetches
  (`:80`) without counting them, and runs 8 threads with no pacing against a CLI that is documented at
  ~6 calls/min elsewhere (`tools/queue_watch.py:31`), so a rate-limited burst under-reports until it re-runs.
