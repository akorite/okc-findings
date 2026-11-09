# OKC kernel loop — measured findings

Findings, post-mortems, and runbooks from the **Muse Spark Open Kernel Challenge** kernel track
(2026-10-22 → 2026-11-05 07:59 UTC), produced by the automated `okc_loop` harness. Code stays in
the private harness repo; this repo is the evidence: what was measured, what was falsified, and
what the leaderboard said at the close.

## Final standing (verified against the live boards 2026-11-05 07:16 UTC, 44 min before close)

| board | score | value | rank at close |
|---|---|---|---|
| `muse_decode_layer` (**= overall rank**) | **2.2837x** | 9,905.1 µs | **#143 / ~1295** |
| `muse_bf16_proj` | 1.5986x | 3,022.0 µs | #68 |
| `muse_prefill_layer` | 1.9171x | 202.08 ms | #109 |
| `muse_attention_decode` | 2.6677x | 1,411.8 µs | #164 |
| `muse_moe_w4a16` | 2.409x | 5,583.4 µs | #171 |

Four boards improved on the final day; the layer row was frozen at 0/20 runs while the field kept
improving, so its rank decayed #139 → #143 with an unchanged score. The displayed 2.2837 is a
+1.33% lucky draw of a file whose mean is 10,038.8 µs (σ measured at 1.17%, three code-identical
submissions); the organizer's post-challenge reruns use **medians**, so the honest expectation is
≈2.254x. Better-mean files were submitted to give that rerun a better file to judge.

## The rules that shaped everything

1. **`baseline.py` is the thing being timed, not `submission.py`.** The starter is ~10x slower by
   construction (expands NVFP4 weights per call, Python attention loop, plain `x @ w.T`). A
   cached-weight starter variant beat the starter **26x on L4** and scored **0.0927x on H200** —
   10.8x *slower* than baseline. Beating the starter is evidence of nothing.
2. **Fork `baseline.py`; never rewrite a hand-tuned kernel.** Marlin / FA3 / cuBLAS stay, and only
   the measured wrapper around them is replaceable.

## Measured ceilings (why some boards cannot reach their line)

Profiles of `baseline.py` at ~1.00x, each share measured, not guessed:

| board | protected kernel(s) | share | ceiling if all wrapper were free |
|---|---|---|---|
| `muse_attention_decode` | FA3 | 82.5% | 1.348x |
| `muse_moe_w4a16` | Marlin | 87.5% | 1.253x |
| `muse_decode_layer` | Marlin / cuBLAS / FA3 | 49.9% / 21.3% / 14.9% | 2.650x |
| `muse_bf16_proj` | cuBLAS | 96.7% | 2.043x |

Only the dense projections have a board line below the wrapper ceiling — the algorithm (cuBLAS)
itself has to be replaced there, which is why it was the top priority and the one board where a
from-scratch Split-K GEMV was justified.

## Findings that cost the most to learn

- **Generated kernels lose to implementation quality, not algorithms.** Four model-generated MoE
  candidates failed their first build on Triton API errors; the two repaired ones ran **40x slower
  than a plain-torch control** (naive tiles, nibble decode per output element).
- **The judge's noise floor is 1.17% (σ)**, measured on byte-different / code-identical probes
  (9,905.1 / 10,125.3 / 10,085.9 µs). A 4-draw mean resolves ~1.15%; a single draw does not
  resolve anything. `okc check` on already-submitted bytes returns `duplicate` and does not re-time.
- **Daily caps reset at 00:00 UTC** (20 scored runs + 40 checks per problem per day; 10 submits).
  A spent quota is rank decay: the layer was frozen at 0/20 while competitors kept moving.
- **Never allocate large tensors inside CUDA-graph capture.** A 536 MB RoPE table built during
  capture killed an entire `muse_attention_decode` seed line; compute `inv_freq` from `positions`.
- **A graph capture returns the output tensor without executing it** — `replay()` before checking.
- **Triton needs a real file on disk**: `@triton.jit` calls `inspect.getsourcelines`, so candidates
  must be written with a matching linecache entry; exec-from-string fails at import.
- **Sliding-window gathers can start below position 0** — clamp the index only together with a
  `t >= 0` mask, or a short context lets clamped copies through (a real correctness bug).
- **`page_size` and `kv_scale` are the knobs a fast candidate folds away and then fails tests on**
  (1 scored / 64 tested; 1.0 scored / 1/64 tested).
- **Colab L4 is a correctness gate, not a speed ranker**: no sglang / FA3 / Marlin, so ratios are
  never against baseline. T4 is banned outright (no FP8, half the BF16).
- **The mean is the deliverable.** Final positions are medians of organizer reruns, so a lucky
  draw is not a result and a dispatcher that ignores means is a lottery ticket, not a strategy.

## What is here

| path | contents |
|---|---|
| `docs/PROGRESS.md` | the authoritative live state: every lever, every falsification, the final block |
| `docs/HANDOFF-2026-11-04-FINAL.md` | final-day constraints, noise instrument, blockers, dispatch set |
| `docs/plans/` | the harness plan, the H200 on-device autotuner plan, the final-day runbook |
| `docs/reports/` | dated engineering / second-opinion / status reports |
| `docs/HANDOFF-2026-11-03.md`, `DEVIN_HANDOFF.md`, `RESEARCH_HANDOFF.md` | mid-run carry-over notes |
| `docs/PROGRESS-ARCHIVE-2026-10-30-31.md` | the earlier phase, kept for mechanism history |
| `artifacts/reports/` | 101 measured note files: knob tables, A/B campaigns, kernel intel |
| `artifacts/bench/` | final-day campaign plan, dispatch queue, reconciled job ledger |

## Harness invariants worth reusing

- Tier-0 static lint is free and local; `sm_90a` build with `-Xptxas=-v` requires **0 spills**;
  the L4 answers exactly three yes/no questions (build, tolerances, graph replay).
- **The remote H200 `okc check` is the only performance oracle** — spend it only on a candidate
  whose algorithm is already believed to beat baseline, and compute its ceiling first.
- Submit gate: first upload needs ≥1.15x, then ≥1.5% over personal best; never bypass it.
- Nothing automated may prompt a human (`okc login`, rule acceptance, `okc withdraw` stay manual;
  the loop exits 2 = EXIT_HUMAN for a supervisor).

`SUPER_EXPERIMENTS_KEY` in the original handoff is redacted in this copy.
