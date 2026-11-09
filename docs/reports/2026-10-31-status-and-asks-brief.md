# Status brief and asks: `muse_bf16_proj` and `muse_decode_layer`

Written 2026-10-31 for a stronger model with H200 kernel experience. Everything below is a number
returned by the judge's own `okc check`, by the leaderboard API, or by the judge's own kernel
profile of our submission. Job ids are given so any claim can be re-read with
`okc status <job_id> --json`. A deeper bf16_proj-specific earlier brief sits beside this one
(`2026-10-31-second-opinion-brief.md`).

---

## 1. Scoreboard

| board | our row | rank | `#50` | what `#50` requires of us |
| --- | --- | --- | --- | --- |
| `muse_bf16_proj` | **1.0970x** / 4403.7 us | **1387** of 1504 | 1.5718x | +43% |
| `muse_decode_layer` | **1.1177x** measured, no upload yet (floor 1.15) | — | 2.2252x | +99% |
| overall | — | 1654 | — | — |

`okc submit` is done through `auto_submit.evaluate()`: six predicates, first upload >= the floor,
each later upload >= 1.5% over the personal best, no single load below 0.952x, plus CI separation
and nearby-shape verification.

## 2. Why this is so hard: both boards are defined by kernels we are winning 0% against

**`muse_bf16_proj`.** The baseline *is* cuBLAS (`_bf16_gemm_dispatch_impl`, 96.7% of weighted GPU
time, job `0281313b02ef46c2`). Solving the judge's formula against the 30 measured cells says the
four wide layers must be **1.544x faster than cuBLAS on every cell** to reach 1.5718x, because
cuBLAS already reads 48-67% of HBM peak. Three hand-written GEMMs were built and measured:

| design | judge result | why it lost |
| --- | --- | --- |
| CUDA-core Split-K GEMV (staged, 4 loads in flight, `__ldcs`) | **0.8086x** whole problem | instruction-bound: `qkv_gate` m=8 = 69.34 us vs cuBLAS 13.87 |
| `wmma` 16x16x16 + `cp.async`, dynamic smem, BK 32-256, 2-8 stages | cuBLAS parity on the L4, 1.0x | cuBLAS is itself at 0.44-0.70 of L4 peak |
| Hopper TMA + `wgmma`, transposed operands `m64n8/16/32k16` (weight on the m64 axis, tokens on n, matching `nvjet_sm90_tst_128x8_...`), last-CTA lock reduction, descriptors prebuilt in `prepare()`, 0 spills / 58 regs | **0.8669x** whole problem; correct, and **slower than cuBLAS on 15 of the 16 cells it touched** (its only win is `router@m=1`: 7.87 vs 8.75 us) | not established, and this is question 1 below |

**`muse_decode_layer`.** The objective's premise was a "2.650x wrapper-only ceiling". I reproduced
that figure exactly — and only when FA3 is scaled to zero, i.e. it counted FA3 as improvable. Under
the objective's own invariant (keep `moe_wna16_marlin_gemm`, `flash_attn_with_kvcache` and cuBLAS
cells intact) the ceiling is much lower. Measured from **our own submission's profile**
(job `c75f1c58c8734f24`), per cell: take the time the three protected kernels consume and ask what
the score would be if every other kernel *and* every launch gap vanished:

| load | cand_us | protected_us | removable_us |
| --- | --- | --- | --- |
| decode 1 sliding / full | 142.0 / 152.3 | 103.9 / 119.2 | 38.1 / 33.1 |
| decode 16 sliding / full | 218.1 / 394.3 | 189.0 / 368.7 | 29.1 / 25.6 |
| decode 32 sliding / full | 250.6 / 605.4 | 247.2 / 697.2 | 3.4 / — |
| verify 1 sliding / full | 194.1 / 197.5 | 159.2 / 172.1 | 34.9 / 25.4 |
| verify 16 sliding / full | 531.2 / 712.4 | 537.3 / 633.9 | — / 78.5 |
| verify 32 sliding / full | 745.3 / 1099.7 | 693.7 / 1049.8 | 51.6 / 49.9 |

> **Ceiling with cuBLAS, FA3 and Marlin protected: 1.239x. Target: 2.2252x.**
> We are at 1.1177, so ~11% of legal headroom remains; the target is **1.8x beyond** the ceiling.

Why the three cannot be beaten by us, with evidence:

* **Marlin** is weight streaming at 1.45-1.8 TB/s of a 4.8 TB/s HBM (804 MB re-read per call at
  `verify 32`), and its bytes are irreducible. Route compaction — the objective's Phase 2D — is
  already implemented inside it: `moe_wna16_marlin.cuh:121` skips `expert_ids == -1` blocks,
  `marlin_template.h:386-393` sets `parallel = num_valid_blocks`, `activation.py:114-117` skips dead
  rows, and the finalize has a `HAS_IDS` guard. Its workspace cannot be shrunk either: the grid is
  `sms * blocks_per_sm` from occupancy and it writes `locks[blockIdx.x]` into that buffer.
* **FA3** is invoked correctly already (`page_table`, `cache_seqlens`, `causal=True`,
  `window_size=(window-1, 0)`, FP8 descale), and it is 34-61% of the four `full` cells.
* **cuBLAS** beat three hand-written GEMMs, as above.

To reach 2.2252 the arithmetic needs **FA3 at <= 0.50x** (that alone reaches 2.223 with a perfect
wrapper), or the dense cuBLAS cells replaced. Note the size of the FA3 opportunity that *is* there:
at `decode 32 full`, FA3 costs 371-501 us while the KV the window requires is on the order of 25 MB,
i.e. ~5 us at peak — FA3 is tens of times off the floor at T=32 because it is a large-batch kernel
serving 32 queries. That is the plan's U6 hypothesis, and it is the only lever that lifts the
ceiling.

## 3. What is working (do not discard)

* `muse_decode_layer` **1.1177x**, job `c75f1c58c8734f24`, all tests pass. Per cell: `verify 1`
  **1.339x**, `decode 1` **1.204x**, `decode 16` 1.180x, `decode 32` 1.044x, `verify 16` 1.022x,
  `verify 32` 1.013x. The gains land exactly where the launch chain dominated.
  From: attention prologue fused 5 launches -> 1 (qk-norm + RoPE + both FP8 casts + KV scatter,
  bit-exact against a torch oracle of `FusedQKNorm`, 13/13 cases); post-expert finalize chain
  14 -> 1 for `T < 32` (verified by the judge's oracle at T = 1, 8, 16, 32, 128, 256); router as
  one BF16 GEMM on the stacked `[hi; lo]` at every T; and the `ffn_boundary` + `F.pad` tail fused
  4 launches -> 2 for `T <= 8` (residual columns 1024..8191 bit-identical, `hidden` bit-identical
  in 9 of 11 cases, 1 element of 8192 within one bf16 ulp in the other two). That last one measures
  **+0.7%**, not the +3% it was scoped at.
* `muse_bf16_proj` **1.0970x on the board** from one change: the router as a single BF16 GEMM on
  the stacked `[hi; lo]` (1.18x at m=1, 2.87x at m=8, 1.59/1.57/1.38/1.41 at m=16/32/128/256).
* The harness itself, which had four defects that blocked autonomous work: `MILESTONE_FLOOR`
  carried the board line (8 checks, 0 uploads); `overall_rank()` scanned for a `you` flag the
  payload does not contain (so the termination check could never fire); `lint` printed PASS without
  recording the tier-0 verdict (so `check` refused quota for evidence lint already had); and a
  crashed caller left `inflight=1`, wedging every later OKC call.

## 4. The handicap that explains most of the lost time

**We have no Hopper GPU to iterate on.** Every `sm_90a` kernel is written blind:

* Colab refuses H100/A100 on this account (`Backend rejected accelerator 'H100'`);
* the L4 cannot execute `sm_90a` at all, and its bandwidth per SM is 5 GB/s against the H200's 36,
  so it cannot rank a compute-bound kernel either;
* therefore each `sm_90a` correction costs a judge round trip (~80-260 s) against a budget of
  20 checks/day/problem.

Concretely, the first Hopper kernel needed four round trips for two bugs that a dev node finds in
minutes: torch's `load_inline` emits its own `-gencode` set, so `-arch=sm_90a` alone still generated
`cuda.compute_90.ptx` and ptxas rejected `wgmma` there (fix:
`--generate-code=arch=compute_90a,code=sm_90a`); and the split-K epilogue branched on
`out == nullptr`, passing that null pointer to the reducer. A third round trip went on the kernel's
own probe whose reference had a shape bug. None of those three were algorithmic.

## 5. What I need, in order of expected value

1. **A Hopper dev node for 1-2 hours a day** (H100 PCIe is enough; `sm_90a` is what matters).
   With `ncu` and a compiler that runs, the TMA/`wgmma` kernel could be profiled instead of
   inferred: I would expect to settle within an hour whether it can beat cuBLAS, and if not, why
   (the candidates are: 1 CTA/SM starves the memory pipe; the activation re-read at m >= 32; the
   serial last-CTA reduction; or the 64-row `wgmma` tile being wrong for m <= 8 even transposed).
   Without it, each hypothesis costs a judge check and a paragraph of inference.
2. **Permission to replace FA3 for the small-T `full` loads** (`decode 16`, `decode 32`,
   `verify 16`, `verify 32`). This is the only lever that lifts the decode ceiling above 1.239:
   FA3 is 34-61% of those cells and is tens of times off the KV-bandwidth floor at T=32. The plan
   already scoped it as U6 and expected 2.1-2.4x on the attention problem; here it would be a
   paged FP8 GQA kernel for T <= 256 with the same window/causal/GQA semantics, FA3 kept for the
   sliding loads where it is cheap.
3. **A production-grade dense GEMM to test the bf16_proj hypothesis** — not another from-scratch
   kernel. If a CUTLASS/CuTe `sm90` `KernelTmaWarpSpecializedCooperative` build with a 128x128
   tile lands above 85% of peak on these shapes (M <= 256, N <= 8192, K <= 8192), then bf16_proj's
   board line is reachable and the work is integration. If it lands where my 64x64 kernel did
   (24-25% of peak), the board line is out of reach and the effort should stop being spent there.
   This is a one-day experiment with a Hopper node and a pure guess without one.
4. **More Colab CU** (currently 338 of 340 left) is *not* the binding constraint; the H200 check
   quota is 20/day/problem and we have used 3 today, 1 on decode. Cheaper than compute: another
   pair of eyes on question 1 below.

## 6. Questions

1. **Is 85-95% of HBM peak reachable on `M <= 256` skinny BF16 GEMMs on H200 under a per-replay
   256 MB L2 flush?** If yes, with which tile/stage/warp-specialisation, and what is the single
   most likely reason a 64x64 TMA+`wgmma` design with 6 stages, `n8/n16/n32` token tiling and
   >= 128 CTAs measured **slower than cuBLAS on 15 of 16 cells** (e.g. `decode 32` `qkv_gate`
   26.71 us against cuBLAS 14.30)? My candidate explanations, in order: the serial last-CTA
   reduction sits on the critical path at these sizes; one CTA/SM with 96 KB of smem cannot keep
   the pipe full where cuBLAS uses two CTAs and split-K with a separate reduce; or the activation
   re-read at m >= 16 (`n/64 * m * k * 2`) co-bottlenecks with the weight stream.
2. **What is `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` doing that a hand kernel cannot?**
   Its name says a 128-row weight tile, 8 tokens, K=64 slices, 12 stages, 4x1 clusters, split-K.
   At m=1 it costs 10.5 us for 37.7 MB of weight plus 1.7 us of `splitKreduce` — 3.1 TB/s. My
   best was 15.24 us on the same cell.
3. **Is the decode_layer ceiling of 1.239 right?** If you disagree, the disagreement is in the
   per-cell table in section 2, and re-deriving it from the profile takes two minutes.
4. **Is there a legitimate way to lower FA3's cost at T <= 256 without replacing it** — e.g. a
   different `scheduler_metadata`, varlen packing of the 32 queries, or avoiding the
   `.expand()`-ed FP8 descale tensors, which are non-contiguous views?
5. **Should we stop?** If both board lines require beating kernels at which NVIDIA's own libraries
   are near the hardware limit, then the honest expected value of more checks on these two problems
   is zero and the effort belongs on a board whose line sits under its own arithmetic.

## 7. Reproduction

```bash
# board position and the cutoff we are chasing
PYTHONPATH=src .venv/bin/python -c "
from okc_loop import ledger as L, api_client as A
c = A.OkcClient(L.Ledger('artifacts'))
print(c.board_rank('muse_bf16_proj'), c.cutoffs('muse_bf16_proj', ranks=(50,)))"

# any check's full per-cell truth, including the judge's kernel profile of our own submission
okc status <job_id> --json

# the four candidates that matter, in the order they were measured
src/okc_loop/candidates/bf16_proj/kernel_v5.py           # incumbent 1.0908x (board 1.0970)
src/okc_loop/candidates/bf16_proj/kernel_wide_tma.py     # TMA+wgmma transposed, 0.8669x, correct
src/okc_loop/candidates/decode_layer/composite.py        # 1.1177x + the ffn_tail fusion
src/okc_loop/candidates/decode_layer/post_attn_fused.py  # post-attention fusion, in verification
docs/PROGRESS.md                                          # the checkpoint, all tables
```

Quota: 3 of 20 checks used today (1 on `muse_decode_layer`), 1 of 10 submits on `muse_bf16_proj`,
0 on `muse_decode_layer`. Deadline 2026-11-04 23:59 PT.
