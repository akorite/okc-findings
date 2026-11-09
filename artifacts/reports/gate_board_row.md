# The submit gate's baseline: board row vs check row

Date: 2026-11-02. Owner: GateFix. Files: `src/okc_loop/auto_submit.py`, `ledger.py`,
`orchestrator.py`, `profiler_parser.py`, `tests/test_autosubmit_board_row.py`.

## 1. The audit's two claims

**Claim A — "the gate compares the candidate to the incumbent blob's check row instead of the
submit that produced the board row." CONFIRMED, with one correction.**

* Correction: the *geomean* half was already the board row at refusal time. The logged decision
  (event 1793624716.988584) reads `gain: candidate=2.6497 required=2.6167 ... incumbent=2.6089`,
  i.e. `incumbent_from_ledger` (HEAD `auto_submit.py:374-385`) read `personal_best`, and that row
  held 2.6088834 — the submit's number, itself a hand repair (no `personal_best` event exists for
  `4cfc5cca`; the `score` row of kind `submit`, id 29, was inserted outside the ledger API).
* Confirmed: nothing in the code ever wrote that row. `record_upload` (HEAD
  `orchestrator.py:914-918`) passes `geomean=candidate.geomean`, and `candidate` is built from the
  **check** report (HEAD `orchestrator.py:882-893`), so a normally-uploaded incumbent's
  `personal_best` is its *check* geomean (2.6667 here). `grep record_score` shows only `"check"` is
  ever recorded, so no submit row and no submit per-load rows exist.
* Confirmed, and this is what actually refused the candidate: `_per_load_dominance` re-queried
  `ledger.latest_load_timings(incumbent.sha256)` (HEAD `auto_submit.py:282`), which returns whatever
  rows the sha has — the check's. Against the check's six loads the candidate wins **2/6**; against
  the submit's six loads (the run the judge scored) it wins **5/6**:

  | load | candidate (check) | incumbent (submit, 2.6089) | incumbent (check, 2.6667) |
  |---|---|---|---|
  | decode 1 | 3.6612 | 3.6055 ✔ | 3.5338 ✔ |
  | decode 16 | 2.5393 | 2.5359 ✔ | 2.6333 ✘ |
  | decode 32 | 2.4281 | 2.4269 ✔ | 2.5412 ✘ |
  | verify 1 | 3.2884 | 3.0378 ✔ | 3.0940 ✔ |
  | verify 16 | 2.1759 | 2.1654 ✔ | 2.2184 ✘ |
  | verify 32 | 2.1425 | 2.1602 ✘ | 2.2159 ✘ |

  Geomean of the submit row = 2.6088834 (exactly the board row); of the check row = 2.6667289.
  So the 5-of-6 escape hatch was unreachable purely because the baseline's per-load rows came from
  the wrong run. The interval was wrong the same way: `personal_best.ci` = [2.5971, 2.8240] (the
  check's) while the submit's own `summary.speedup_ci` = [2.5698, 2.6480].

**Claim B — "`load_regression` compares against a constant instead of the incumbent." CONFIRMED.**
HEAD `auto_submit.py:236` was `if load and ratio < (1.0 / MAX_LOAD_REGRESSION)`: a fixed 0.952
against the *reference* baseline. It held a load 4% down against an incumbent sitting at 3.0x
(nothing wrong with the blob) and passed a load 40% down against one at 1.2x.

## 2. What changed

* `auto_submit.record_board_row` (new): stores the submit run's own score row (`kind='submit'`,
  its geomean + its CI) and its six per-load rows, tagged with the submit job id.
* `auto_submit.record_upload(submit=...)`: prefers the submit payload's own measurement over the
  check's for `personal_best`; falls back to the caller's (check) numbers when the payload carries
  no measurement. `orchestrator.evaluate_and_upload` passes the judge's payload; the cycle now also
  writes one `submit` score row per upload (previously zero, which is why `pareto.uploads` was 0
  and the attention board row had to be hand-inserted).
* `orchestrator._adopt_reconciled` records the same board row when an ambiguous submit is adopted.
* `auto_submit.incumbent_from_ledger`: `personal_best` → best `submit` row → best `check` row
  (`exclude_sha` keeps a candidate from being measured against its own check) → the static floor.
  Per-load rows come from the job that produced the board row (`ledger.latest_load_timings(sha,
  job_id=...)`), falling back to the sha's newest rows when the incumbent has only check rows —
  documented in the docstrings.
* The check-only fallback is a **bar**, not a progression: `required = max(best_check, floor)`, no
  `MIN_GAIN` and no interval/dominance test. `MIN_GAIN` is the progression over a *scored* row; a
  sibling candidate's unsubmitted check is not one (and the loop checks two candidates per cycle,
  so tying them against each other refused every first upload — 4 existing tests caught it).
* `evaluate`'s `load_regression` now bounds each load at `incumbent_ratio / 1.05`
  (`_worst_load`), with 1.0x only for loads the incumbent never measured. `Candidate.worst_load`
  is gone (it had no callers).
* `profiler_parser`: the payloads the judge actually returns (and that `artifacts/cas/profiles/`
  holds) put timings under `benchmarks` and the aggregate under `summary`, not under `result` —
  `parse_run` returned an all-zero report for them, which is also why `_adopt_reconciled` silently
  no-opped on the attention submit. `_benchmark_records` accepts `benchmarks`; `speedup`,
  `speedup_ci`, `score_us`, `passed` fall back to `summary`; a suite reported as
  `tests`/`tests_passed` counts is spelled as pass/fail records so `nearby_shapes` reads one shape.
* Ledger backfill (data, from the saved judge payload `artifacts/cas/profiles/4cfc5cca.<job>.json`):
  the incumbent's six submit load rows were recorded under job `3063f7fb1ec2402c`, and its
  `personal_best` interval was replaced with the submit's own `ci()` = [2.5403, 2.7724]. No `okc`
  command was run.

## 3. The verdict for `okc-loop submit 55fd129a`, evaluated against the live ledger

```
ALLOW muse_attention_decode 55fd129a8a12: geomean 2.6497 beats 2.6167 with CI [2.5767, 2.8195]
  tier: tier0=PASS tier1=PASS_SM90A_WRAPPER capability=hopper_sm90a_specialized
  fresh_check: job=09b60f3ae8bf4c2b geomean=2.649666543474289
  gain: candidate=2.6497 required=2.6167 floor=1.05 baseline=2.6089
  ci_separation: candidate_lo=2.5767 incumbent_hi=2.7724 per_load=5/6 loads improved
  load_regression: worst=verify 32 ratio=2.1425 floor=2.0573
  nearby_shapes: verified=True
  reserve: bucket=open
```

(`nearby_shapes` needs the `summary.tests` fallback: this payload carries `tests: 16,
tests_passed: 16` and no per-test records, so without it the gate would stop one predicate later.)

No predicate was weakened: `ci_separation` is untouched apart from which six numbers it pairs,
`assert_automated` is untouched, and the 5% bound is tighter for every incumbent above 1.05x.

## 4. Tests

`tests/test_autosubmit_board_row.py` (9 tests) uses the exact ledger numbers: the candidate's check
2.6497 with its six loads, the incumbent's board row 2.6089 with the six loads its submit measured,
and both counterfactuals — no board row (`gain` refuses at 2.6667) and a board row whose per-load
rows are the check's (`ci_separation` refuses at 2/6). Negative case: +1.5% with 3/6 loads → HOLD
`ci_separation`; a load 20% down on the incumbent → HOLD `load_regression`.

`PYTHONPATH=src .venv/bin/python -m pytest tests/ -x -q` → **696 passed, 10 skipped**.

## 5. Open risk

`latest_load_timings(sha)` without a `job_id` (used by `pareto`) now sees submit rows too and lets
them win, so `_composite` mixes the submit run's per-load `cand_us` into the assembled step. That is
arguably the board-faithful number, but it is a silent behaviour change for the composite display.
