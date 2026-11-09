# Prior art: trxphxle/okc-2026-muse-kernels (read 2026-11-03)

Source: `github.com/trxphxle/okc-2026-muse-kernels`, README + `experiments/attn-overlap/HANDOFF.md`.
Their final: layer **2.1756 (#191)**, MoE 2.3249 (#260), attention 2.5673 (#241), bf16 1.4423 (#404),
prefill 1.4774 (#386). All below our rows on every board — but their *ceiling tests* are what we lack.

## Ceiling (deletion) tests — upper bounds someone actually measured
| Idealization | Measured upper bound |
|---|---|
| **Perfect finish-time balancing (heaviest-expert-first)** | **+1.1–1.3% expert board** |
| **Remove re-reads across 64-token jobs of the same expert (multicast / L2 reuse)** | **~1.5% expert board; V32 duplicate weight bytes 8.6% avg (0% at D32)** |
| Delete nearly all FP4 unpack + half the scale multiplies | +0.5–0.9% expert board |
| Exact KV gather-read in place of full attention | +0.63% decode |
| Every deletable dense-projection epilogue + consumer cost | +1.0% decode at D16/D32 (+2.75% over all four affected loads) |
| Pre-warmed code/constants for every D1/V1 kernel | +0.16% decode |
| Overlap recovery attention<->GEMM | 0 (verify attention never co-runs with a GEMM in this chain) |

## Negative results we had not measured
| Change | Result | Cause |
|---|---|---|
| Deeper expert pipelines (4–6 ring slots, half-K stages, batched WGMMA commits) | **−1 to −3% each** | "weight pipeline saturates at 3 slots; **stalls come from unit boundaries, not depth**" |
| Router GEMM + top-k fusion | −4.2% | removed existing inter-kernel overlap |
| Pre-expert RMSNorm fusion | −2 to −4% | same |
| ~10.9-bit lossless BF16 codec (measured entropy 10.46 bits) | decode 6–11% slower | byte savings don't cover decode cost ⇒ **our 12-bit is optimal** |
| Hopper inline memory compression on dense weights | ~0 | weights behave like random draws |
| VMM-backed granularity-aligned expert arenas | +0.1–0.2% | already 2 MiB aligned |
| Wider tiles / two resident CTAs / register donation | slower | 576 threads × 96 regs, ~226 KiB smem/CTA |

## Engineering findings that constrain us
1. The layer is a serial PDL chain with **<1.1 us of idle gaps at T <= 8**; deleting a whole glue launch
   saves only **1.2–2.0 us**. ⇒ our T2 "aux-kernel fusion" (guessed 8–14 us) is bounded at ~2 us.
2. **Achieved DRAM bandwidth depends on transfer size**: ~4.6 TB/s at 2 GiB, ~4.2 at 128 MiB, **~3 TB/s for
   D1's ~64 MB expert read**. ⇒ the standalone MoE's 22 us at bs=1 is at the finite-transfer floor, and our
   in-layer 29.4 us has ~7 us of slack — consistent with our own T1 estimate, now independently measured.
3. Expert-stage bytes equal the planned stream including per-job re-reads, at 53–75% of peak, no spills.
4. Per-stage board leaders are MoE ~15%, attention ~21%, dense ~6.5% ahead of their base, weighted to the
   decode leader's margin — "a different expert-kernel design, not one missing trick".
5. Judge noise ±1.5–2% run-to-run plus ~1% between clusters; paired local A/B (A/A within ±0.3%) was
   required to resolve anything under ~2%.

## What this changes for us
- Our `moe/wm_t1.py` (`OKC_WMAJ`, deepen the tiny-build ring 3→5) falls inside their measured **−1 to −3%**
  class. Keep it off; do not spend a check on it.
- The two mechanisms with a *measured* ceiling we have never tested: **heaviest-expert-first unit order**
  (+1.1–1.3%) and **eliminating the 8.6% duplicate expert reads at V32** (~1.5%). On the MoE board, where a
  percent buys ~101 ranks, those are worth ~120–250 ranks.
