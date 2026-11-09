# research_09 — autotuning & experiment methodology for the last 46 h

Scope: what published practice and our own measurements say about (1) accepting a kernel when
σ/run ≈ 0.6–2 %, (2) proving a candidate emitted different code before timing it, (3) sizing a
tuning set when the judge draws fresh inputs, (4) tuning under cold-L2, graph-replay timing.
Every claim carries a repo path, a doc URL, or venue+title.

## TL;DR — top 3 by usefulness for the two remaining days

1. **Replace every fixed acceptance margin with one derived from σ and n.** Our three tuners use
   three unrelated gates (`moe_selftune`: 1.5 %/1.5 µs over 3 rounds,
   `artifacts/reports/moe_selftune.md:59-66`; `attn_gap1._DUAL_*`: 3 %/2 µs over 8 rounds with a
   75 % round-win rule, `src/okc_loop/vendored_work/attn/attn_gap1.py:3367-3375`;
   `bf16_selftune`: 1.5 %/1.5 µs over a 42-rep median, `artifacts/reports/bf16_selftune.md:36-40`).
   A constant 3 % margin at n = 8 with σ_d ≈ 0.56 % is 15 SE — it has ~0 power for a real 2 % win;
   a constant 1.5 % margin over bf16's 42-round median at a noisy 4 % cell is ~2 SE (accepts a null
   ~2.5 % of the time) and ~20 SE at a 0.4 % cell (accepts nothing). Set
   `t = z·σ_d/√n` (n from §1.2), and size n from the effect you need to see.
2. **Emitted-code gate before any timing.** Cubin hashing is deterministic (measured here: same PTX
   + flags → identical sha256; one changed instruction → different hash), and the machinery is
   already in the tree (`tools/cu_build.py:98-104` assembles `sm_90a` with `ptxas -v`;
   `src/okc_loop/colab_harness.py:595-625` already compiles with `-Xptxas=-v` and parses spills).
   Keep the `.cubin` and hash it; equal hash ⇒ not a candidate (§2).
3. **Tune over fresh input draws, with a held-out confirmation split.** The judge redraws inputs
   before every repetition (`artifacts/reports/judge_replica.md:34-41`), so a tuner that picks on
   one seed pays a selection bias of ≈ σ_input·√(2 ln k) (≈1.7σ for 4 candidates; winner's-curse
   mechanics, Thaler, *Anomalies: The Winner's Curse*, JEP 1988). Rounds cost ~0.1 ms; builds cost
   seconds. Spend the budget on ≥8 draws × ≥16 paired rounds, not on one seed × 3 rounds (§3).

## 0. Our own noise, as measured (use these numbers, not the judge's ±1.5 %)

| quantity | value | source |
|---|---|---|
| σ per judge run, board speedup | ≈ 0.63 % | `docs/PLAN-2026-11-03-h200-test-plan.md:18,76-79` (keell0renz, 460 experiments) |
| check-to-check per-shape | ±1.5–1.9 % (up to ±2 %) | same; `artifacts/reports/const_sweep_fci5.md:111-117` |
| geomean noise, one check | ±0.3–0.5 % | `artifacts/reports/const_sweep_fci5.md:111-117` |
| A/A drift in one session | +0.9 % (T=8) / +1.9 % (T=1) | `artifacts/reports/g8_chain.md:25` |
| same script, two sessions | absolute µs moved 40 % | `artifacts/reports/g8_chain.md:25` |
| per-cell A/A | up to 3.9 % at a bf16 cell | `artifacts/reports/bfp_geo.md:148,199` |
| in-session paired A/B at 30+ reps | ±0.2 % | `docs/PLAN-2026-11-03-h200-test-plan.md:18` |
| our ledger CI | percentile bootstrap on paired ratios, widened to ±1.5 % | `src/okc_loop/scoring.py:157-192` |

The judge's own A/B is 5 order-shuffled paired rounds with a 10 000-resample bootstrap (seed
20261030) reporting a one-sided 95 % lower bound — reverse-engineered and recorded in
`artifacts/reports/autoresearch_repo_digest.md` §4 (`benchmarks/gpu/judge.py:98-103`,
`timing_stats.py:5-16`). **5 rounds resolves (80 % power) MDE = 1.25·σ_d: ≈1.1 % at the board's
σ_d ≈ 0.9 %, but ≈2.6 % on a per-shape σ_d ≈ 2.1 %** — so the judge cannot certify a 0.5 % win and
on individual shapes it cannot certify a 2 % one either. That is the whole argument for deciding
local and shipping only ≥2 % ideas.

## 1. Acceptance gates that survive 0.6–2 % noise

### 1.1 What the tooling actually does (so we do not copy a non-gate)

- **Triton** benches each config **once** with `do_bench(..., quantiles=(0.5,0.2,0.8))`, then
  `min(timings, key=timings.get)`; the key is the list `[p50,p20,p80]`, so a tie on p50 is broken by
  p20 then p80, and an exact tie by dict insertion order (first declared config wins). There is no
  re-run, no significance test, and with `cache_results` the *decision* is cached to disk keyed on
  triton key + backend hash + fn cache key + env + tuning key + configs, so a re-run reuses the old
  winner without re-measuring (`python/triton/runtime/autotuner.py`, `_bench`, `run`,
  `check_disk_cache`; fetched from `raw.githubusercontent.com/triton-lang/triton/main/`).
- **CUTLASS profiler** reports all kernels' GFLOPS/s and lets the user sort
  (`--sort-results-flops-per-sec`, `--enable-best-kernel-for-fixed-shape`); samples are controlled by
  `--warmup-iterations` (default 10), `--profiling-iterations`, `--min-iterations`,
  `--profiling-duration`, `--sleep-duration`. No significance test on the winner; correctness is a
  separate bit-level gate (`--verification-enabled`, `--epsilon` default 0 = bitwise)
  (https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html).
- **Hoefler & Belli, SC'15, "Scientific Benchmarking of Parallel Computing Systems"** is the usable
  methodology: report a CI for every non-deterministic number; choose n from the CI width
  (`n = (s·t(n−1,α/2)/(e·x̄))²`, §4.2.2); **n > 5 measurements are needed to assess CIs
  nonparametrically**; time single events (not blocks) so ranks/CIs are computable; timer overhead
  < 5 % of the interval and precision ~10× finer; exclude the first/warm-up measurement; state the
  cache state (cold vs warm) explicitly. That is exactly our regime, and it is the citation the
  report should use when justifying n.
- **Triton `do_bench`** already does cold-cache: a cache-sized buffer is cleared before every timed
  call, one event pair per call, median/mean of `n_repeat` (`python/triton/testing.py`, `do_bench`).
  So does CUTLASS (`--llc-capacity` cycles input tensors to force capacity misses,
  `--workspace-count` avoids cache-resident workspaces; profiler docs above).

Conclusion: industry autotuners pick a point-estimate min and leave the statistics to the user.
Our gate must be ours.

### 1.2 How many paired rounds for a d % delta

Paired, two-sided 95 %, power 1−β:

```
n_pairs = (z_{1-α/2} + z_{1-β})² · σ_d² / δ²        (z .975 = 1.96, z .8 = 0.8416 → coefficient 7.85)
```

σ_d is the sd of the **paired log-ratio per round**. With independent arms σ_d = √2·σ, giving the
conservative `n = 15.7·(σ/δ)²`; if drift cancels within a round (our interleaved design) σ_d ≈ σ and
`n = 7.85·(σ/δ)²`. For a CI-only criterion (no power requirement) use `n = (1.96·σ_d/δ)²`, about
half. Nonparametric floor: **no exact sign/Wilcoxon test can reach p < 0.05 below 6 pairs**
(2/2⁶ = 3.1 % two-sided is the smallest attainable).

σ = per-run (per-arm) relative sd; conservative `n = 15.7(σ/δ)²`; parentheses = drift-cancelled
`7.85(σ/δ)²`, rounded up, floored at 6:

| σ \ δ | 0.5 % | 1 % | 2 % |
|---|---|---|---|
| **0.6 %** | **23** (12) | **6** (3→6) | **6** (1→6) |
| **1.0 %** | **63** (32) | **16** (8) | **6** (2→6) |
| **2.0 %** | **251** (126) | **63** (32) | **16** (8) |

Worked against our numbers: σ = 0.63 % (board) → δ = 1 % needs **7** pairs (matches the plan's
independently derived "6–8 runs per arm", `docs/PLAN-2026-11-03-h200-test-plan.md:18,79`); δ = 0.5 %
needs **25**; δ = 2 % needs 3, use 6. At the per-shape σ = 1.9 %, δ = 1 % needs **57** pairs — i.e.
per-shape sub-2 % calls are only decidable in-session, where σ_d ≈ 0.56 % (the ±0.2 % at 30 reps
implies σ_d ≈ 0.56 %): then δ = 0.5 % needs **10**, δ = 1 % needs **4→6**. **8–16 interleaved rounds
is the right in-prepare budget; 3 is not** (`moe_selftune` and `layer_build_tune` both use 3).

### 1.3 Setting the margin so a false accept is rare

The gate should be an equivalence-style test with a threshold that *tracks the standard error*, not
a constant percentage:

```
accept iff median(log-ratio) ≤ −t,  t = z_{1−α'} · σ_d/√n      (one-sided; use the paired ratio's own σ_d)
```

and cross-check with the round-win count. Numbers for `t` (one-sided 95 %, z = 1.645):

| σ_d | n=8 | n=32 |
|---|---|---|
| 0.5 % | 0.29 % | 0.15 % |
| 1 % | 0.58 % | 0.29 % |
| 2 % | 1.16 % | 0.58 % |
| 4 % | 2.33 % | 1.16 % |

A fixed 1.5 % margin corresponds to σ_d ≈ 2.6 % at n = 8 — right for a noisy cell, ~5× too loose
for a quiet one, and ~20 SE at bf16's 42-rep median when that cell is quiet (σ_d ≈ 0.4 % → median
SE ≈ 0.08 %), where it accepts a null essentially never but also cannot see the 1–1.5 % wins those
cells actually offer. A fixed 3 % margin is unreachable at any of our in-session σ (SE 0.2 % at
n = 8): `_DUAL_MARGIN` will essentially never fire on a real effect, which is consistent with the
"gain=0.00 %" outcomes in the selftune reports.

Multiplicity: we compare many variants per session. With m = 20 comparisons at α = 0.05 each, the
family-wise false-positive rate is 1 − 0.95²⁰ ≈ 64 %. Either Bonferroni (α' = 0.0025, z = 2.807 →
`t` × 1.7, and n × 1.7 for the same power) or stop-and-look-safe intervals. **Sequential testing is
not optional if we peek**: p-values and CIs are invalid under continuous monitoring; use
always-valid p-values/CIs (Johari, Pekelis, Walsh, "Always Valid Inference: Bringing Sequential
Analysis to A/B Testing", arXiv:1512.04922) — their whole point is that "try it as data arrives and
stop when it looks good" is exactly what we do with 10 checks/day.

Sign test / Wilcoxon detail: the round-win rule (≥75 %) is a sign test and is **weak**: 6/8 wins has
one-sided p = 0.145 (not evidence); exact 5 % needs 8/8 at n = 8, 6/6 at n = 6, 9/10 at n = 10, and
75 % needs n ≥ 20 (15/20, one-sided p = 0.021). Use it only as a "clock drift did not do this"
cross-check, as `attn_gap1.py:3370` comments, never as the primary gate. Wilcoxon signed-rank is the
better nonparametric primary (uses magnitudes); exact two-sided p = 0.008 at n = 8 with all ranks
positive.

### 1.4 What this implies for the gate we should ship

- per-load gate (judge scores loads, weight 46/16 per `artifacts/reports/autoresearch_repo_digest.md`
  §4): `t` from the cell's own σ_d and n; n = 16 interleaved rounds for sub-1 % calls, 8 for 1–2 %.
- board-relevant aggregation: the judge gates on the **point estimate**, reports the CI separately
  (`autoresearch_repo_digest.md` §4, "theirs is the more honest framing"). We should do the same:
  report CI, require the point estimate to clear the judge's own resolution (≥2 % on a load that is
  ≥40 % of the step) before spending a check.
- never accept on a single seed, never accept a 0.5 % effect from a 5-round judge A/B — it cannot
  exist in that data.

## 2. The emitted-code gate

Why: a source permutation that compiles to identical code is not a new candidate
(`artifacts/reports/autoresearch_repo_digest.md` row 24, quoting `OfficialJudge.md:37`), and "fewer
registers / fewer instructions is a hypothesis, not a speedup" (row 16, `OfficialJudge.md:35`).
Timing an identical build only measures noise, and at σ = 1–2 % it manufactures false wins.

What is cheap and sufficient for a `load_inline` submission:

1. **Cubin hash.** `nvcc -arch=sm_90a -cubin -Xptxas=-v -o k.cubin k.cu`, then sha256. Measured
   here with the repo's own `ptxas` (`.venv/.../cuda_nvcc/bin/ptxas`): identical PTX + flags →
   byte-identical cubin (`0c3c2b6c…` twice); changing one immediate (`add.s32 … ,1` → `,2`) →
   different cubin (`bcb5c4b1…`). So the hash is a faithful "did the machine code change" bit *within
   one toolkit version* (hash equality across different CUDA versions proves nothing).
2. **Resource diff.** The same `-Xptxas -v` line already parsed at
   `src/okc_loop/colab_harness.py:595-625` and `src/okc_loop/profiler_parser.py:469-483` gives
   registers, spill bytes, smem. CUTLASS's equivalent is the per-kernel CSV row plus static
   `SharedStorageSize`; Triton's is `CompiledKernel.metadata` (`shared`, `num_warps`) plus
   `asm["cubin"]` / `asm["sass"]` (`python/triton/compiler/compiler.py`, `CompiledKernel`), and its
   cache key is the source+options+env hash, i.e. Triton already treats "same key = same binary" as
   an invariant.
3. **SASS delta (only when 1–2 disagree).** `cuobjdump -sass k.cubin | …` for instruction counts,
   `cuobjdump -res-usage` for the resource table
   (https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html, §2.1.1 and `-res-usage`). Rule:
   identical SASS text ⇒ reject as non-candidate; different SASS but identical instruction count and
   schedule ⇒ treat as a scheduling gamble, not an improvement, and demand a bigger δ before timing.
4. **Correctness is a separate, bit-level gate**, already present and should stay: our selftune
   exactness probe `max|d| == 0` (`artifacts/reports/moe_selftune.md:66-68`) is the same convention
   as CUTLASS `--epsilon=0` (bitwise default) and TVM's `check_correctness`; Triton has no default
   correctness check at all.

Cost/placement: a standalone `ptxas` run is ~0.1–1 s (`tools/cu_build.py` does it offline with no
GPU), so it belongs **before** `prepare()` (in the offline pre-check) and, for the exact shipped
build, in `prepare()` where the nvcc log is already captured. Only limit: `compile_sm90a` compiles
non-template bodies, so a templated `__global__` (the MoE) reports no registers
(`tools/cu_build.py:12-16`); for those, hash the cubin extracted from the real `.so`
(`cuobjdump -xelf all`, binary-utilities doc) rather than the standalone probe. This also retires the
"nvcc ~5 % codegen jitter" folklore (`docs/DEVIN_HANDOFF.md:100`): identical source+flags is
bit-reproducible (measured above), so a 5 % spread is measurement/clock, not codegen.

## 3. Tuning-set sizing when the judge redraws inputs

Facts: the judge refreshes inputs **before every repetition** (`artifacts/reports/judge_replica.md`
§"Fresh inputs per repetition", `judge.py:73-94` in the digest), and the replica's launch order is
`refresh → flush → record → replay`. `prepare()` has 300 s and is untimed
(`docs/HANDOFF-2026-11-03.md:16-18`), and the router load is the judge's private number
(`judge_replica.md` "What this replica still cannot produce").

Implications, in order of importance:

1. **One seed is a biased estimator, not a sample.** With k candidates measured on the same draw the
   winner's measured gain is inflated by ≈ σ_in·E[max of k standard normals] ≈ σ_in·√(2 ln k)
   (1.67σ for k = 4, 2.04σ for k = 8; order-statistics/winner's-curse result, Thaler JEP 1988). For
   a 1 % decision you need σ_in ≲ 0.3 %, i.e. ~11 averaged draws if input-driven spread is 1 %. The
   MoE in particular changes *work* with the draw (distinct experts per load,
   `judge_replica.md` router section), so this is not hypothetical.
2. **Split the draws.** Tune on draws 1..m−2, confirm on the last 2 (or 1/3 held out). The candidate
   must win on the held-out median of paired ratios before `prepare()` may select it. This is the
   cheapest defence against the exact failure we keep recording ("lost twice because synthetic
   routing is not the judge's", `docs/HANDOFF-2026-11-03.md:163`).
3. **Budget the currencies correctly.** A round = 256 MiB flush (≈53 µs at 4.8 TB/s) + a ~30–100 µs
   kernel + a device-side input refresh ⇒ ~0.1–0.2 ms. 5 builds × 6 shapes × 16 rounds × 8 draws ≈
   4 k rounds ≈ 0.6 s. **Rounds are free; builds are not** (each `load_inline` is seconds to tens of
   seconds inside the 300 s; the existing tuners cap builds, not rounds: `moe_selftune.md:75-80`,
   `bf16_selftune.md:48`). So spend on n and draws, never on a third build.
4. **Tune the shapes the judge scores, on the input law the judge uses.** `_gen_input` in
   `moe_selftune` is already bit-for-bit the task generator (`moe_selftune.md:61-64`); keep that,
   add draw variation, and do not tune near-shape branches on scored-input statistics.
5. **Selection is per-shape, so the per-shape σ applies**, not the geomean's 0.3–0.5 %
   (`const_sweep_fci5.md:111-117`). That is why per-shape gates need n ≈ 16–25 while a
   load-weighted decision needs n ≈ 7.

## 4. Tuning in a cold-L2, graph-replay-only regime

What the regime is, from the sources: judge = 5 order-shuffled paired rounds of a block of N replays,
one event pair around the block, block mean, `~4 µs` graph overhead subtracted (digest §4). Ours =
per-replay event pairs, median of 30, 256 MiB flush before every replay, and a *measured* empty-graph
subtraction (`judge_replica.md` items 2–3: `one − (two − one)`).

Rules that follow:

- **Instantiate once, replay many.** CUDA graphs pay setup/upload at instantiation and replay with
  "very little overhead" (CUDA Programming Guide §4.2, "definition, instantiation, execution";
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html). The first
  replay is still slower; Hoefler & Belli's "exclude the first measurement" is the citation, and the
  harness already does 5 warmups (`colab_harness.py:186,220-235` per the digest).
- **Subtract or block-average; do not trust a single replay.** `cudaEventElapsedTime` resolves
  ~0.5 µs (CUDA Runtime API, `cudaEventElapsedTime`,
  https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html) — 0.5–1.7 % of a 30–100 µs
  kernel, and the ~4 µs graph launch is 13 % of a 30 µs layer. The judge block-averages N; Triton's
  `do_bench_cudagraph` unrolls `n_repeat` calls into one graph, times the whole replay and divides,
  with 10 retries (`python/triton/testing.py`); our replica subtracts the empty-graph cost. Pick one
  and say which — the digest records that judge-style block means and our per-replay medians differ
  and must not be compared across.
- **"Cold L2" is true only for the first kernel in a replay.** A graph launches its nodes
  back-to-back, so kernel 2..n inherit kernel 1's L2 state. Per-kernel tuning with a flush before
  each kernel is a different regime from in-graph reuse. For a single-launch layer this is moot; for
  any multi-kernel candidate, tune the *whole graph*, as the judge does. Triton's answer when
  per-kernel attribution is needed: `do_bench_cudagraph_proton` puts Proton scopes *inside* the graph
  and clears the cache per iteration, which "exclude[s] cache flushes from the measurement"
  (`python/triton/testing.py`). CUTLASS's answer is `--llc-capacity` plus `--use-cuda-graphs`.
- **No host sync means no CPU timers and no per-launch gaps**, so the ranking measured with
  sync-per-kernel does not transfer: the repo has measured +0.9 %/+1.9 % A/A drift within a session
  and a 40 % absolute shift across sessions (`g8_chain.md:25`). Ranking transfer requires the same
  ritual: same graph, same flush, same warmups, same order swap.
- **Timer discipline.** Hoefler & Belli: overhead < 5 % of the interval, precision 10× finer; if
  either fails, time a block and divide (which is what the judge does). Also: report the cache state
  and the flush size — ours is 256 MiB, the judge's is 64 MiB (digest §4), which changes how much of
  a 28 MB expert set survives into the next replay.

## 5. Concrete, minimal change list (no new abstractions)

- `moe_selftune`/`layer_build_tune`: raise rounds 3 → 16, keep the exactness probe, replace the fixed
  `max(1.5 %, 1.5 µs)` with `max(1.645·σ_d/√n, 1.5 µs)` where σ_d is measured from the same rounds,
  and keep a ≤2 % "one loud check" override (existing behaviour).
- `attn_gap1`: `_DUAL_MARGIN 3 %` and `_PLAN_MARGIN 3 %` will not fire at n = 8/16
  (`attn_gap1.py:3203,3368`); replace with the same σ-derived threshold, keep `_DUAL_WIN` only as a
  sanity cross-check and note 6/8 is p = 0.145.
- `tools/cu_build.py`: keep the `.cubin` (`assemble()` currently discards it, lines 98-104) and print
  `sha256`; wire `compile_sm90a`'s log hash through the replica so an unchanged build is rejected
  before a check.
- Replica: run the A/A pair the plan already specifies (`judge_replica.md` §"Acceptance gate") on the
  borrowed node; until it passes, every number is relative-only — that is also the only way to learn
  whether the in-session σ_d is 0.56 % or 2 %.

## Sources

- `python/triton/runtime/autotuner.py` — `_bench` (quantiles 0.5/0.2/0.8), `min(timings, key=…)`,
  disk-cached tuning decision. https://raw.githubusercontent.com/triton-lang/triton/main/python/triton/runtime/autotuner.py
- `python/triton/testing.py` — `do_bench` (per-run events, cache clear before each call),
  `do_bench_cudagraph` (unroll → one replay, `/n_repeat`, 10 retries), `do_bench_cudagraph_proton`.
  https://raw.githubusercontent.com/triton-lang/triton/main/python/triton/testing.py
- `python/triton/compiler/compiler.py` — `CompiledKernel`, `asm["cubin"]`/`asm["sass"]`, cache-key hash.
- CUTLASS Profiler docs — `--warmup-iterations` (10), `--profiling-iterations`, `--min-iterations`,
  `--use-cuda-graphs`, `--llc-capacity`, `--verification-enabled`, `--epsilon` (0 = bitwise).
  https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html
- Hoefler & Belli, SC'15, "Scientific Benchmarking of Parallel Computing Systems" — CIs, n from CI
  width, n > 5 for nonparametric CIs, single-event timing, warm-up, cold vs warm cache.
  https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf
- Johari, Pekelis, Walsh, "Always Valid Inference…", arXiv:1512.04922 — sequential monitoring.
- Thaler, "Anomalies: The Winner's Curse", JEP 1988 — selection bias; extreme-value factor √(2 ln k)
  (David & Nagaraja, *Order Statistics*).
- CUDA Programming Guide §4.2 CUDA Graphs — definition / instantiation / execution, overhead paid at
  instantiation. https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- CUDA Runtime API, Event Management — `cudaEventElapsedTime` "resolution of around 0.5 microseconds".
  https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
- CUDA Binary Utilities — `cuobjdump -sass`, `-res-usage`, `-xelf`, `--function`.
  https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
- Repo: `docs/PLAN-2026-11-03-h200-test-plan.md:18,76-79,222`; `src/okc_loop/scoring.py:157-192`;
  `src/okc_loop/colab_harness.py:595-625`; `src/okc_loop/profiler_parser.py:469-483`;
  `tools/cu_build.py:98-104`; `src/okc_loop/vendored_work/attn/attn_gap1.py:3203,3367-3375`;
  `artifacts/reports/{moe_selftune,bf16_selftune,layer_build_tune,attn_selftune,judge_replica,
  const_sweep_fci5,g8_chain,bfp_geo,autoresearch_repo_digest}.md`.
- Measured here: `ptxas -arch=sm_90a -v` on identical PTX twice → identical sha256; one immediate
  changed → different sha256 (the `.venv` wheel `nvidia-cuda-nvcc-cu12`, no GPU needed).
