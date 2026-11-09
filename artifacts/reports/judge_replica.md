# Offline judge replica (P0.1)

`tools/judge_replica.py` is `tools/bench_bundle.py`'s payload plus the three conventions the bundle
deliberately omits. `tools/replica_selftest.py` proves, on CPU, the two of them that can be proven
without a GPU.

```
python tools/judge_replica.py <candidate.py> [muse_decode_layer] --out artifacts/replica \
    [--router-targets targets.json] [--nearby] [--gpu H200] [--no-subtract]
python artifacts/replica/okc_replica_<candidate>.py      # on the H200 box
python tools/replica_selftest.py                        # CPU, no torch, exits 0
```

The generator writes an ordinary `colab_runner.write_harness` script and then substitutes five
anchored edits into it. If `src/okc_loop/colab_harness.py` changes shape, generation fails with the
offending anchor rather than silently producing an unpatched bench.

## What it adds

1. **Router-seed load search.** Before `prepare()` and before any candidate is verified, the harness
   calls the reference's *own* `weights_fields(weights, cases, weights_seed)` on the payload's case
   rows, sets the returned `router_seed`, and every later `generate_weights(seed, router_seed=...)`
   uses it. Nothing is re-derived: the draw order (`seed + i * 0x9E3779B97F4A7C15`), `LOAD_TOL`
   (0.015), `LOAD_ATTEMPTS` (5000), the per-load weighted mean and the "draw whose worst load is
   closest" fallback all come from `problems/muse_decode_layer/reference.py`. `judge_replica`
   re-exports the two constants parsed from that file (no torch needed) and the selftest asserts
   they are exactly 0.015 / 5000.

   The judge's per-case `router_load` / `router_distinct` live in its private `task.yml`, so they
   are an input, not a guess:

   ```json
   {"decode 1":  {"router_load": 260.5, "router_distinct": 21.0},
    "verify 32": {"router_load": 330.0, "router_distinct": 31.0}}
   ```

   Keys are load names; each entry is attached to every scored case of that load and flows into
   `generate_input` (accepted, ignored) and `speed_of_light_ns` (used, as the judge uses it).
   **Without `--router-targets` no search runs** — `weights_fields` returns `{}` for a case table
   with no targets, and that run is not a replica of the scored run's routing.

2. **Fresh inputs per repetition.** `Graphs.timed` takes a `refresh` callback and calls it before
   every repetition, immediately before the cold-L2 flush. Each call advances an LCG seed and
   writes a new `generate_input` draw into the captured buffers' own addresses. The launch order is
   `refresh → flush → record → replay`, so the new values are in HBM, L2 is cold, and the copy is
   outside the timed interval. A cached answer cannot ride one input pool through a run.

3. **The flat per-replay subtraction.** `replica_setup` captures two graphs — one holding one
   1-element `add_` launch, one holding two — times both with the same 30-repetition cold-L2
   ritual, and computes `one - (two - one)` (`per_replay_overhead`). Since `one = G + K` and
   `two = G + 2K`, that recovers `G`, the graph-launch/event cost, and cancels the kernel's own
   time regardless of how long the "empty" kernel takes. `G` is subtracted from every reported
   per-shape time (`round(max(0, graph_us - subtract_us), 4)`), so the per-load and total sums use
   subtracted values. `--no-subtract` (on the generator, or appended to the run command) reports
   raw medians. `result["replica"]` carries `{router_seed, router_targets, one_replay_us,
   two_replay_us, subtract_us, subtract, fresh_inputs}` and is copied onto every candidate entry so
   a parsed `ColabResult` keeps it; each case carries both `graph_us` and subtracted `replay_us`.

## Acceptance gate (from `docs/PLAN-2026-11-03-h200-test-plan.md` §3 P0.1)

A replica number is trustworthy only when both of these hold, measured on a matching H200 node:

- **A/A:** two runs of two byte-identical copies of the candidate produce per-shape times whose
  paired-ratio 95% interval **contains 0** (i.e. spans 1.0x). A/A is a property of the judge's
  harness, so its interval can only be read off a check pair; the replica's contribution is the
  per-shape medians that feed the same comparison, and two bundles on the same node must agree
  within ±2% per shape before any A/B between different candidates is believed.
- **Reproduction:** the replica's per-shape `replay_us` for the 12 scored shapes must match the
  judge's own 12 per-shape microseconds within **±2%** on the same node, same hour. Until that
  check is run and passes, every replica number is relative-only.

**Status: not yet run on a GPU.** This file and the harness were written and self-tested on CPU;
no H200 session has executed them. Treat the gate as open.

```
$ python tools/replica_selftest.py
PASS constants: LOAD_TOL=0.015 LOAD_ATTEMPTS=5000 read from reference.py
PASS case rows: _load/_weight/_seed restored, router targets carried, no metadata leaks
PASS convergence: seed 4354685564957106459 hits every load (worst 0.0000% <= 1.5%); identical to reference.weights_fields
PASS no targets: search returns {} instead of inventing a target (judge's own behaviour)
PASS fallback: no draw is within 1.5%, closest of 40 chosen (worst 65.6667%)
PASS subtraction: overhead = one - (two - one) recovers the per-call time for one/two replays (4.1 us graph cost, 0/0.9/2/17.5/28.3 us kernels)
PASS harness glue: 5 anchors land on colab_harness; 12 real layer cases reach the search, seed 2177342782468427829 applied, subtract_us 3.0, --no-subtract disables it
replica selftest: 7/7 passed
```

The convergence test executes the reference's `weights_fields` **source, verbatim**, against tiny
synthetic generators (and a small stand-in for the handful of tensor operations it touches), so it
fails if the replica ever stops delegating the search to the reference; its independent `worst_error`
then re-checks the chosen seed against the targets. The subtraction test feeds synthetic `one`/`two`
medians and asserts `one - overhead` is the per-call time. The glue test generates a real harness
from a stub candidate, imports it, and runs the injected `replica_setup` with the empty-graph pair
faked — it fails if either `colab_harness.py` or the injection drifts, and it pins that all 12
scored shapes reach the search with their targets.

## What this replica still cannot produce

- **The SGLang baseline.** No baseline, no paired ratios, no per-load speedup, no geomean, no board
  score. Output is **absolute microseconds per shape** and, between two candidates on the same
  node and hour, **A/B ratios**. That is enough for the T1 kill criterion ("is the bare-ring
  skeleton already ≥26 µs?") and for choosing between two builds; it is not a leaderboard estimate.
- **The judge's routing.** Only with `--router-targets`. The targets are the judge's private
  `task.yml` values; a replica run without them measures a different amount of expert work, which
  is exactly the confound (`distinct` experts per T=1 load) the search exists to remove.
- **Per-input `local_start`.** The judge computes each input's busiest 32-expert range
  (`case_fields`) and passes it in; the harness leaves `local_start` at 0, so the *checked* and
  *charged* routing is the range-0 view. Timing of the MoE bytes is unaffected (it is one routed
  expert either way), the routing tolerance checks and `sol_fraction` can be.
- **The judge's exact repetition count.** The judge's harness uses `MAX_REPEATS = 20`; the replica
  times 30 repetitions per shape and takes the absolute median. It does not implement the judge's
  order-swapped paired median, which needs the baseline to pair against.
- **The judge's subtraction constant.** 4.1 µs is the plan's number for H200; the replica measures
  `G` on the node at run time and reports it, which is strictly better than assuming — but it is a
  measurement of the empty-kernel pair, not the judge's own constant.
- **`prepare()` and weight generation are timed differently.** The router search generates the
  weights once more than a bench run does (search weights, then seeded weights). It is untimed and
  outside `prepare_s`, but it does add wall-clock and peak-memory churn to a run.

## Operator checklist before trusting a number

1. `python tools/replica_selftest.py` exits 0.
2. A/A: two copies of the same file, two bundles, same ten minutes → within ±2% per shape.
3. `--router-targets` supplied, and `result["replica"]["router_targets"] == 12` in the output.
4. `result["replica"]["subtract"] == true` and `subtract_us` near the expected ~4 µs; a wildly
   different `subtract_us` (or `one_replay_us` ≈ `two_replay_us`) means the empty-kernel pair was
   mis-captured and the shape numbers are unadjusted.
5. Only then compare candidates, and only within the same node and session.
