# `klr_plan`: prepare-time re-tune of the attention launch plan, per scored shape

Candidate: `src/okc_loop/vendored_work/layer/klr_plan.py` — a copy of the shipped `klr_qd2.py` (board `82c098ae`,
ATT7 `-DOKC_QDIRECT=1 -DOKC_PK2=1`, measured 2.2536). Only host-side code changed: no CUDA source, no define, no
build name. The sweep runs in `prepare()` after `_attn_selftest`, so one judge check measures every candidate plan on
the real H200.

Why this retune is sound where the per-T MoE self-tune was not: the plan is a function of the shape alone —
`_plan7(B, pairs, L, window, max_pages, P, device)` — never of routing. Prepare knows every input of it.

## 1. What the launcher picks today (the twelve scored shapes)

Scored rows are the problem's own table: `B in {1,16,32} x L in {1,8} x {sliding, full}`, ctx 80000, page 1, tp8
(`NUM_Q_HEADS=16`, `NUM_KV_HEADS=2` → `pairs=1`). `_launch7` derives `coop` from the grid when the plan does not
set it; the table below shows the resolved plan (`_TUNED7` applied), the tile count at ctx 80000, the CTA count
(`n_splits*B*pairs`, or `n_splits` when flat) and the grid's share of the 132-SM, 2-CTA-per-SM capacity.

| B | L | layer | n_splits | nc | coop | tiles | CTAs | grid | candidates timed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | sliding | 32 | 1 | 0 | 32 | 32 | 12% | 32/1/0 32/1/1 32/8/0 32/2/0 16/1/0 24/1/0 |
| 16 | 1 | sliding | 16 | 1 | 0 | 32 | 256 | 97% | 16/1/0 16/1/1 16/8/0 16/2/0 8/1/0 12/1/0 20/1/0 24/1/0 |
| 32 | 1 | sliding | 8 | 1 | 0 | 32 | 256 | 97% | 8/1/0 8/1/1 8/8/0 8/2/0 4/1/0 6/1/0 10/1/0 12/1/0 |
| 1 | 8 | sliding | 33 | 1 | 1 | 33 | 33 | 12% | 33/1/1 33/1/0 33/8/1 33/2/1 16/1/1 25/1/1 |
| 16 | 8 | sliding | 8 | 1 | 1 | 33 | 128 | 48% | 8/1/1 8/1/0 8/8/1 8/2/1 4/1/1 6/1/1 10/1/1 12/1/1 |
| 32 | 8 | sliding | 4 | 1 | 1 | 33 | 128 | 48% | 4/1/1 4/1/0 4/4/1 4/2/1 2/1/1 3/1/1 5/1/1 6/1/1 |
| 1 | 1 | full | 252 | 8 | 1 | 1250 | 252 | 95% | 252/8/1 252/8/0 252/1/1 252/4/1 126/8/1 189/8/1 315/8/0 378/8/0 |
| 16 | 1 | full | 264 | 1 | 0 | 1250 | 264 | 100% | 264/1/0 264/1/1 264/8/0 264/2/0 132/1/0 198/1/0 330/1/0 396/1/0 |
| 32 | 1 | full | 264 | 1 | 0 | 1250 | 264 | 100% | 264/1/0 264/1/1 264/8/0 264/2/0 132/1/0 198/1/0 330/1/0 396/1/0 |
| 1 | 8 | full | 252 | 8 | 1 | 1250 | 252 | 95% | 252/8/1 252/8/0 252/1/1 252/4/1 126/8/1 189/8/1 315/8/0 378/8/0 |
| 16 | 8 | full | 264 | 1 | 1 | 1250 | 264 | 100% | 264/1/1 264/1/0 264/8/1 264/2/1 132/1/1 198/1/1 330/1/0 396/1/0 |
| 32 | 8 | full | 264 | 1 | 1 | 1250 | 264 | 100% | 264/1/1 264/1/0 264/8/1 264/2/1 132/1/1 198/1/1 330/1/0 396/1/0 |

### Flags (where the shipped numbers are klr-build artifacts or marginal)

* **b1 sliding, L1 and L8 — 12% of capacity, but tile-bound, not plan-bound.** A sliding layer reads
  `window + L - 1` keys → 32/33 tiles, and the kernel puts at most one tile per split (`OKC_TCHUNK` switches to the
  whole-tile instance once `n_splits*64 >= kmax`). 32/33 splits *is* the parallelism limit; more splits are empty
  splits that still pay the finalize. Only `coop` (0↔1) and `nc` are sweepable here, which is what the candidate
  set does. The two `_TUNED7` entries are on other shapes, so these two were never swept on the judge.
* **b16/b32 sliding L1 — 256 CTAs = 1.94 per SM, not a whole multiple of 132.** 124 SMs get two CTAs and 8 get one,
  so the tail is set by the 2-CTA SMs. `n = 8/12/20/24` (128/192/320/384 CTAs) are timed against it.
* **b16/b32 sliding L8 — 128 CTAs (48%), the SM-multiple value** (`132//16`, `264//32`); b32's 4 splits is the file's
  own Oct-26 sweep winner, b16's 8 splits is the "cooperative finalize at 8 splits" note. Swept again only ±.
* **b1 full, L1 and L8 — 252/8/1, already the file's own tuned entry** (Oct 26 14:44 plan sweep, GPU-1). Re-measured
  only as the baseline plus its two knobs; a pick needs to beat a plan that was already swept on the judge.
* **b16/b32 full — `nc=1` and `coop=0/1` are the file's *derived defaults*, never swept on the judge.** The single
  strongest knob in the file's own history is `nc` (b1 q1 full: 1→8 copies, 12.30 → 12.00 us) and the flat coop
  gate only arms for `L == 8`; both are timed here for the L1 shapes for the first time.

## 2. Candidate set, and what is dropped as illegal

Per shape, at most 8 triples, shipped first, each a *full* `(n_splits, nc, coop)` (an override that named only
`n_splits` would let `_launch7` re-derive `nc` and silently drop the tuned value):

1. the shipped triple (the baseline; the same launch the layer would run);
2. the `coop` flip (cooperative finalize ↔ last-CTA finalize), only when the grid can be resident;
3. two workspace-copy counts (`nc = 8, 2` when the shipped is 1, else `1, 4`);
4. `n_splits` x 0.5 / 0.75 / 1.25 / 1.5;
5. the CTA count that lands exactly on 1 and 2 whole SM multiples (`132/gridb`, `264/gridb`).

Dropped, with the reason:

* **`cs`/`ncl` clusters** — every scored shape's `_plan7` result has `cs=0`; the cluster path needs sm_90 non-portable
  cluster sizes and the file's own notes measured it slower for all of these (and the residency guard at
  `klr:3557` would clear `coop` on a cluster grid anyway). Not a knob the sweep may invent.
* **`n_splits` past the last tile** — clamped to `tiles` (never below the shipped value, so the baseline is exact).
* **`coop=1` on a grid that cannot be resident** — the launcher clears it (`q.coop` guard, `klr:3557` / `OKC_COOPGUARD`),
  so that candidate would only re-measure the shipped plan and burn a slot.
* **`nc > n_splits`** — clamped; extra copies of a workspace no split writes are pure allocation.
* **`PDL_MAX_T` 32 vs 128** — not swept. `PDL` is 128 today and `T <= 128` covers every scored shape except
  b32 x L8 (T=256), so the only shape it would move is b16 x L8. PDL's win is *overlap with the preceding kernel*
  and an isolated cold-L2 replay cannot price it (the file's `PIN_PICK` exists for exactly this reason). Changing it
  blind would be a guess, not a measurement.
* **`w0` (first-wave tile weight)** — flat-only, and `_launch7` derives it from `coop`/`L`; the shipped values
  (1024 / 1135) are the file's re-swept ones. Left alone to keep the sweep at 8 slots.

## 3. The sweep

* Inputs: `_plan_case7` — the `_att_shape_inputs` idiom from the sibling attention file: random *finite* e4m3 K/V
  codes (an all-zero cache leaves every score equal and exercises no finalize path) and exactly the pages the scored
  positions need. Sliding uses ctx = window (2048), which gives the same `min(window+L-1, max_pages)` tile count the
  scored ctx 80000 gives; full uses ctx 80000.
* Timing: `_plan_time7` — one CUDA graph per candidate, cold L2 by a 256 MB read sweep before every replay, the
  replay order alternating each round (so no candidate owns the first slot), 16 rounds, one sync at the end.
* Decision: argmin of the median, kept only if it beats the shipped plan by `max(3%, 2 us)` **and** wins ≥ 75% of
  the 16 rounds **and** reproduces the shipped plan's rows within 1e-3 rel-L2. The shipped plan is candidate 0 in the
  same round-robin, so a clock drift has to move both.
* Row equivalence: the launcher writes `out` as `__nv_bfloat16*` only, so a different split count changes the
  reduction tree and moves the last bit of a few elements — the honest scale is the bf16 ulp, and a real plan bug
  (dropped keys, an unheld grid barrier) moves the rows by >0.1. 1e-6 rel-L2 is the fp32-accumulator scale and is not
  observable from the launcher; the sibling attention file can use it only because its variant keeps the per-row order
  identical. The measured value is printed per pick.
* Application: `_PLAN7_PICK[(B, L, sliding, pairs)] = (raw n_splits, n_splits, nc, coop)`; `_launch7` applies an entry
  only when `_plan7`'s own `n_splits` (before any override) still equals the value the sweep saw. Any other context
  (a correctness test at ctx 3000, say, where `n_splits = min(tiles, ...)` shrinks) keeps the shipped plan, and every
  failure path — a capture error, a sweep exception — leaves the shipped plan standing.

Printed per shape (one line, 12 total) plus a total:

```
[plan] B16 L1 full    shipped=264/1/0 best=264/8/1 med=[91.4 91.4 89.9 92.0 91.3 91.4 92.1 91.4] cand=[...] pick=264/8/1 wins=15/16 rel=3.1e-05
[plan] prepare +1.7s, 1/12 shapes re-planned
```

`shipped=` is the triple the layer runs today, `best=` the fastest measured candidate (even when rejected), `med=`
all candidate medians in µs in `cand=` order, `pick=` the accepted triple or `shipped`.

## 4. Prepare cost

No compiles. 12 shapes x ≤8 candidates x 16 replays = ≤1536 graph replays + 96 graph captures; each replay pays a
256 MB L2 read sweep (~0.09 ms) plus the kernel (6 µs sliding … ~200 µs b32 full). Estimated **≈1–2 s** of the
~140 s layer prepare (judge cap 300 s); the sweep prints its own `[plan] prepare +Ns`.

## 5. Gates run here (no GPU on this workstation)

* `py_compile` + `ast.parse`: clean.
* `okc_loop.cli lint` delta vs `klr_qd2.py`: **30 errors both, identical list** (all inherited: `threading`/`ctypes`/
  `os.environ`/`globals`/`time.time` sandbox hits and the 140 KB line — the vendored graft's, not this change's). One
  extra *warning* of a class that already fires repeatedly (FP8 e4m3 division saturation), from the copied
  `float8_e4m3fn` case input.
* `tools/cu_build.py --variant MID` and `--variant MAIN`: **identical FAIL sets** to the base (9 and 16 entries, all
  pre-existing include/`int64_t` failures in the offline tool's stubs). `ATT7_CUDA_SRC` compiles `ok` in both — with
  the tool's own caveat that it does not compile ATT7's bodies below the `<torch/` cut, which this change does not
  touch.
* `tests/test_attn_plan_sweep.py` (new, pure — no torch needed): shipped-first, ≤8, every candidate legal
  (`1 <= n_splits <= max(tiles, shipped)`, `1 <= nc <= n_splits`, `coop=1` only on a resident grid), tile-bound
  shapes never oversubscribe, and both knobs are always offered. 4/4 pass.

## 6. What this does not do

* It does not re-time the *layer*; only the attention launch. A plan that wins isolated can lose in-chain (PDL
  overlap, L2 pressure from the qkv GEMM) — the in-chain risk is the same one the file's `PIN_PICK` notes for the
  dense tuner. Mitigation: the acceptance bar is 3% / 2 µs, and the sweep prints every median so a losing pick is
  visible in the judge log.
* It cannot verify anything on this workstation (no torch, no GPU): the numbers above come from `_plan7`'s arithmetic
  replayed in pure Python, and the sweep's own timing only exists on the judge.
