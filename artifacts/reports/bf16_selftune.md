# bf16_selftune: per-cell kernel swap on `muse_bf16_proj`

Candidate: `src/okc_loop/vendored_work/bf16/bf16_selftune.py` (`# okc: problem=muse_bf16_proj`), a
byte-for-byte copy of `geo.py` (sha256 `be1b0c8d482e0a8180ba79e2f379861fbd5e24206b0ff87e8c6506b684280001`,
653,734 B) plus one additive pass. Board context: our row 1.5891 (#74/1672), geo measured 1.5964,
#1 1.6796; 48 ranks per +1%, 20 free checks.

## 1. Did `geo.py` already dispatch per cell? Yes -- so this is a coverage pass, not a table

`geo.py` ships (a) `_TABLE`, 30 explicit per-(layer, m) family picks for every scored cell, (b) a
per-cell geometry retime at 20 cells (m = 16/32/128/256) that changed 0 of 20, and (c) per-cell v6
routes for the 10 unscored m's 2..384. The cell table, the measured µs and the CTAs are in
`artifacts/reports/bf16_cells.md`.

What is missing is not granularity but **which alternatives get timed at a scored cell**:
`_v6_routes` skips every `_TABLE` key (`if "%s:%d" % (layer, m) in _TABLE: continue`), and
`_geo_routes` builds its own list (`base` / `sk2` / `gemm90` / `ag`). So the other four ways to run
a cell that the *same file* already composes -- the next-bucket `pin`, chenwainuo's `rq` plan of a
larger m (its TMA zero-fill serves the real m), `tT` (cuBLAS on the K-major weight) and
`extra_plan`'s cuBLASLt pin built for this m -- have never been timed at a scored m. `tT` is the one
with a recorded per-cell win elsewhere (`bf16_mechanism_scan.md`: "only unpooler m128 -> torchT").

## 2. What was added (additive, no new kernels)

`_st_routes(state, weights)`, run in `prepare()` after `_geo_routes`, over all 30 scored cells:

* **incumbent** = `state["routes"].get((layer, m), ("base",))`, i.e. geo's *exact* choice after both
  passes -- so a cell the tuner does not win keeps geo's route bit for bit, and `kernel_fn` needs no
  change (`routes` is consulted before `_TABLE` already).
* **candidates** = `_v6_cands(layer, m, state)` (def / next-bucket pin / `nbol` / `rq` of a larger m /
  `ag` via `extra_plan` / `tT`) plus, at m in `GEO_M` only, the geometry candidates `_geo_routes`
  uses (`sk2` pick + `SK2_ALT` twin, co-resident `GEO_GEMM` grids, `ag` when a pin exists). 134
  route-timings over the 30 cells; the per-cell sets are the last column of `bf16_cells.md`.
* **measurement** = `_v6_bench(..., ST_ROUNDS = 3 * V6_REPS = 42)`: cold-L2 CUDA-graph replays,
  alternating order every rep, median of 42 -- the same instrument v6/geo use, so the numbers are
  comparable to the `GEO|` lines.
* **threshold**: a candidate is adopted only if `t < geo - max(1.5% * geo, 1.5 us)` **and** it passes
  `_v6_ok` (shape/dtype, rel L2 and probe rows, on normal *and* outlier inputs, at that exact m).
  The 1.5 µs floor is not cosmetic: at 67-102 µs per load, 1.5 µs is 1.5-2.2% of a load = 0.24-0.37%
  of the board geomean (12-18 ranks), so anything under it is not worth the risk.
* **proof**: one `[btune] cell=<layer,m> geo=<us> winner=<name> <us> gain=<pct>` line per cell, so a
  check log shows what happened at every cell -- including `winner=geo gain=0.00%` where geo's pick
  held.

### Guard (never worse than geo)

1. Per cell `try/except`; the whole pass in `prepare()` in `try/except BaseException`: a cell or the
   pass that fails leaves `state["routes"]` exactly as geo left it.
2. Any candidate whose launch/capture/validation throws is dropped by `_v6_bench`/`_v6_ok` (they
   already return `None`/`False` on exception); if the incumbent's own timing fails (`d` falsy) the
   cell is skipped.
3. First-launch guard, matching geo's own practice (it excluded m = 1/8 sk2 because "a first launch
   that wedges costs a whole run"): `sk2`/`gemm`/`ag` candidates are only offered at m in `GEO_M`,
   i.e. routes geo has already launched; and `def` is suppressed for the router at m <= 32, where
   chenwainuo's path is `run3`, which `prepare()` never launches at a scored m.
4. `ST_BUDGET_S = 30.0` wall-clock cap (v5 prepare ~72 s + v6 60 + geo 25; judge limit 300 s), and
   `ST_ON` as the single off switch.
5. Candidates the incumbent already runs are dropped (`ag` where `_TABLE` is `ag`, `nbol` where it is
   `nb`) so a library plan is not timed against itself.

## 3. Expected gain

Cannot be measured on this workstation (no GPU; running `okc`/`colab` is out of bounds for this
ticket), so the expectation is stated as the ticket asks: **geomean of the per-cell max over geo's
picks** --

$$\text{gain}=\Big(\prod_{\text{load }L}\frac{\sum_{c\in L}t^{\text{geo}}_{c}}{\sum_{c\in L}t^{\text{best}}_{c}}\Big)^{1/6}-1,\qquad
t^{\text{best}}_c=\min\big(t^{\text{geo}}_c,\ \min_{\text{cand}}t^{\text{cand}}_c\big)$$

with `t` the `[btune]` medians. Three readings of that formula:

| case | board gain | ranks |
|---|---|---|
| prior (mechanism scan's per-cell-swap EV: +0.3-0.8%) | +0.3 to +0.8% | 14-38 |
| one 1.5 µs swap on one cell of one load | +0.24 to +0.37% | 12-18 |
| ceiling of the whole axis (all 20 retimed cells take exactly the 1.5 µs floor) | ~+7% | -- |

Best single bet, from the recorded tournament evidence, is **unpooler@128 -> `tT`** (`bfp_geo` has
geo's own pick there at 22.21 µs, the slowest dense cell of its load); second is **o@256 -> `nbol`**
(nb's ol_projection, never timed against the cuBLASLt pin at 12.77 µs). The dense m = 16/32 `rq`
cross-m plans are the long shot; the geometry candidates are re-timed only as the exhaustiveness
control (geo already showed them losing by 10-60%).

## 4. Gates run here

* `ast.parse` -- OK, 659,869 B, first line `# okc: problem=muse_bf16_proj`.
* `PYTHONPATH=src .venv/bin/python -m okc_loop.cli lint` -- **83 findings, the identical normalised
  set to `geo.py`** (the pre-existing `time.time` / `os.environ` / `threading` sandbox errors):
  delta = none.
* `tools/cu_build.py bf16_selftune.py` -- `no embedded CUDA source found`, the same output as
  `geo.py` modulo the path (both exit 1 on that no-source path: this file generates its CUDA inside
  the factory functions). No candidate adds a kernel or an instantiation, so the Colab/A100
  `bfp__get_ext` gate compiles exactly geo's translation unit; the offline enumeration of the 30
  cells' candidate sets was run against the real `_TABLE`/`_v6_cands` (`134 route-timings`, no
  duplicate tags, router `def` correctly suppressed).

## 5. Kill criterion

Spend at most the 20 free checks. Kill the candidate and keep `geo.py` if any of these holds:

1. **The check log's 30 `[btune]` lines are all `winner=geo`** -- then the kernel axis is as closed
   as the geometry axis (0/20) and nothing here can pay.
2. **The only winners are aliases** (`winner=def` at a cell whose `_TABLE` pick is already
   `chenwainuo`): the 1.5 µs floor was crossed by noise, and the route change buys nothing.
3. **The row does not beat geo's 1.5964** beyond the run's own +-1.5% band (geo's 1.5891 vs 1.5964
   spread is that band) after the swaps -- then the swaps are inside the noise and the board row
   should stay geo's.
4. Any `[btune] ... failed` line that repeats at the same cell in two runs (an unstable candidate
   route is not worth a submit), or any cell where a winner's `_v6_ok` passes here but the judge's
   `verify 16` load gets slower (the judge's inputs differ; the guard cannot catch that, only the
   check can).
