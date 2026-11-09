# st2 calibration — noise-sized gates for the two build tuners

New candidates (old first-attempt files left untouched as the record):

| file | base | axis |
|---|---|---|
| `src/okc_loop/vendored_work/moe/moe_st2.py` | CAS `6402f5f3…b3e5.py` | per-bs build (`TINY`/`TINY2`/`MID`/`MAIN`) |
| `src/okc_loop/vendored_work/layer/layer_st2.py` | `klr_fci5.py` | per-T MoE build (`tiny`/`tiny2`/`mid`/`b4`/`main`) |

The old gate was a constant `max(1.5 %, 1.5 µs)` over 3 rounds. At the in-session paired
σ_d ≈ 0.56 % that is ≈ 4.6 SE; at the judge's per-shape σ (1.5–2 %) it is 15–20 SE. It
cannot fire on a real effect (`research_09_autotune.md` §1.1–1.3).

## Gate now implemented (both files)

Rounds run as interleaved pairs on the same GPU, weights and input, with a 256 MiB flush
before every timed launch. For each candidate, `d_r = t_cand,r - t_inc,r` (µs);
after every pair:

* `σ = sd(d_r) / mean(t_inc)`  (relative paired sd, sample sd, n−1)
* `n_req = max(6, ceil(15.7 · (σ / 0.01)²))` — resolve δ = 1 % at 95 % confidence; 15.7 =
  (1.96 + 0.8416)² is the conservative independent-arms coefficient; 6 is the smallest n
  at which an exact sign test can reach p < 0.05 (`research_09` §1.2).
* stop when `n >= n_req` or the added-prepare cap is spent, whichever first.
* accept only if `gain > 1.96 · σ_d/√n` (gain = mean(incumbent) − mean(candidate)) **and**
  `wins >=` the one-sided 95 % binomial sign-test threshold for n.
* a gate pass then needs (a) the existing exact-output probe (`max|d| == 0`) on the tuning
  input and (b) a repeat of the same gate and exactness probe on a fresh held-out draw
  (`seed 7919 + bs/T`, never the tuning seed `917 + bs/T`). Only then is the winner bound.
* any exception leaves the shipped dispatch byte-for-byte as today; non-scored shapes keep
  the hardcoded thresholds.

Every decision prints one line with n, σ, the margin actually used, wins/n and the final
decision, plus the held-out stats when it ran:

```
[stune] bs=… inc=… cand=… n=… sigma=…% margin=…us gain=…% wins=…/… decision=swap|keep(cap|gate|holdout|exact) [hn=… hsigma=… hmargin=… hgain=… hwins=…]
[ltune] T=…   … same fields …
```

## What σ we expect to measure

Planning number **σ_d ≈ 0.6 %** (range 0.3–0.8 %). This is the in-session paired figure
(±0.2 % at 30+ reps, `docs/PLAN-2026-11-03-h200-test-plan.md:18`, derived σ_d ≈ 0.56 % in
`research_09` §1.2), not the judge's 0.63 %/run or the 1.5–2 % per-shape check-to-check
figures, which include input redraws and session drift. An in-process A/B on one GPU, one
weight set, one input and a flushed L2 cancels both; if it does not, the tuner measures its
own larger σ and buys more rounds automatically.

Resulting n for δ = 1 % (formula above, floor 6):

| measured σ_d | n for δ=1 % |
|---|---|
| 0.3 % | 6 (floor) |
| 0.6 % | 6 (5.65 → 6) |
| 0.8 % | 11 |
| 1.0 % | 16 |
| 2.0 % | 63 |
| 4.0 % | 252 |
| ≥ 6.3 % | cap-bound → no swap |

## Caps

| file | added prepare cap | why |
|---|---|---|
| `moe_st2.py` | **150 s** | standalone prepare; judge gives 300 s total |
| `layer_st2.py` | **120 s** | the layer's own prepare is ~140 s, so 120 s keeps 260 s < 300 s |

Charged as measured GPU µs + 100 µs per timed call (`SELFTUNE_SLOT_US` / `_LT_SLOT_US`).
Wall-clock APIs are banned by tier-0 (`time.monotonic` is a sandbox error), so the cap uses
the same CUDA-event accounting the old tuner used; it is a prepare-time proxy. Measured
calls at H200 speeds are ~0.02–0.2 ms (`artifacts/reports/moe_selftune.md` "Time budget"),
so a full tuning sweep is well under 1 s and the cap will not bind in practice.

## Kill criterion

If the cap is reached before `n >= n_req`, the shape gets **no swap** (`keep(cap)`) and
today's dispatch stands. The axis is closed if this happens on every scored shape:

* pessimistic bound 10 ms/call (≈50–500× the measured 0.02–0.2 ms), 4–5 calls per round;
* one shape's cap budget is `cap / (4..5 × 10 ms)` = 2400–3750 rounds; shared across the
  four shapes that have ≥2 candidates it is **≥600 rounds per shape**;
* `n(δ=1 %) = 600` at **σ ≈ 6.2 %**, so a measured σ_d above ≈ 6.3 % means the cap binds
  before significance on every shape — no candidate can ever bind, and the axis is closed
  for real this time rather than re-run. The expected σ (0.3–0.8 %) is ≈8–20× below that
  threshold.

## Evidence (run on this tree)

* `okc_loop.cli lint`: `moe_st2.py` **PASS** (0 errors; the 9 pre-existing precision
  warnings are identical to the CAS base). `layer_st2.py` 30 errors, **identical
  normalised finding set** to `klr_fci5.py` (same 13 error/warning messages, line shifts
  only) — zero new findings.
* `tools/cu_build.py`: `moe_st2.py` 10/10 ok (MAIN, TINY, TINY2, MID, MAIN fallback; CUDA
  and BW sources). `layer_st2.py` compiles `MOE_CUDA_SRC` (4 tuples) and `ATT7_CUDA_SRC`
  (6), and re-compiles all four `MOE_CUDA_SRC` tuples with `-D OKC_B4FIN=1 -D
  OKC_PRMTSC=1` (the `b4` build plus the PRMT suffix the tuner can serve). The other 54
  probe failures are byte-identical to `klr_fci5.py` (the tool's `cudaGetLastError` router
  stub limitation, recorded in the base run).
* The extracted `_paired` / `_binom_min_wins` / `_rounds_needed` functions reproduce the
  table above (e.g. n = [6, 16, 63] for σ = [0.6, 1, 2] %), and a stubbed harness (no GPU)
  exercised all five exits: `swap`, `keep(gate)` on noise, `keep(holdout)` when the fresh
  draw reverses, `keep(exact)` on a non-zero `max|d|`, and `keep(cap)` when the budget
  expires before the minimum rounds.
