# Second-opinion brief: why `muse_bf16_proj` stalls at 1.09x

Written 2026-10-31 for an external reviewer with H200 kernel experience. Everything marked
**measured** is a number returned by the judge's own `okc check` or by our L4 harness; anything
marked **[INFERENCE]** is my reasoning from those numbers. Reproduction commands are at the end.

---

## 1. The task and the exact scoring

Muse Spark Open Kernel Challenge, problem `muse_bf16_proj`: dense projections for Muse Spark Open
on one H200. The judge times our `kernel_fn(data, state)` against SGLang's published reference.

* 30 scored cells = 6 loads x 5 layers. Layers and weights (bf16 unless noted):
  `qkv_gate` 2304x8192, `o` 8192x1024, `pooler` 4096x8192, `unpooler` 8192x4096 (only rows 0-1023
  at m<=8, i.e. n=1024), `router` 256x8192 in **fp32**.
* Loads and token counts: `decode 1` (m=1), `decode 16` (16), `decode 32` (32), `verify 1` (8),
  `verify 16` (128), `verify 32` (256). Every cell carries weight 62.
* Aggregation (verified bit-exact against the judge: replaying the baseline's own medians through
  our implementation reproduces the reported `1.005957` exactly):
  per cell `ratio_i = baseline_ns_i / median_ns_i`; per load `T_load = sum(weight_i * ns_i)/1000`
  us; **problem score = geometric mean over the 6 loads of `T_ref_load / T_cand_load`**. The
  reported microsecond value is `4830.8 / score`.
* `baseline.py` for the four wide layers is `_bf16_gemm_dispatch_impl` (cuBLAS/cublasLt). The
  router is `F.linear(x.float(), w)` in fp32 at m<=8 and two bf16 GEMMs on a `hi`/`lo` split above.
* Tolerances: `REL_L2_TOL=1e-2` relative L2, `PROBE_TOL=1e-3` on 1-in-8 power-of-two probe rows
  (which requires bit-exact weight retention, so no compression or approximation of the weight is
  possible), `ROUTER_TOL=1e-4` on the router (bf16 weights alone fail it; the `hi`/`lo` pair is
  required).
* Measurement protocol, and this matters: each shape is captured in a CUDA graph and replayed
  ~60 times; **a 256 MB buffer is zeroed before every timed replay** to flush L2, and the median
  replay is the timed value. So the weight is re-read from HBM every iteration.

## 2. The target

Live `muse_bf16_proj` board (1484 rows, read 2026-10-30 18:2x UTC): **`#50` = 1.5687x**,
`#1` = 1.6498x. The cutoff drifts up ~0.002/hour. The top fifty sit inside a 5% band.

Our best: **1.0908x** (job `9dbc5a8301024dfa`). Gap to `#50`: 44%.

## 3. Measured per-cell baseline, and where the baseline sits relative to the floor

The judge also reports a speed-of-light floor per cell, `sol_us`, computed from
`hbm_bytes_per_s = 4.8e12` and `bf16_dense_flops_per_s = 989e12`. `q_ref = ref_us / sol_us` is how
far the baseline itself is from that floor. From job `9dbc5a8301024dfa`:

| load | layer | m | ref_us | q_ref | load | layer | m | ref_us | q_ref |
|---|---|---|---|---|---|---|---|---|---|
| decode 1 | qkv_gate | 1 | 14.21 | 1.81 | verify 1 | qkv_gate | 8 | 14.27 | 1.81 |
| decode 1 | o | 1 | 6.82 | 1.95 | verify 1 | o | 8 | 6.46 | 1.83 |
| decode 1 | pooler | 1 | 21.34 | 1.53 | verify 1 | pooler | 8 | 21.23 | 1.51 |
| decode 1 | unpooler | 1 | 7.10 | 4.06 | verify 1 | unpooler | 8 | 6.86 | 3.89 |
| decode 1 | router | 1 | 10.37 | 5.92 | verify 1 | router | 8 | 26.24 | 14.77 |
| decode 16 | qkv_gate | 16 | 14.11 | 1.78 | verify 16 | qkv_gate | 128 | 17.38 | 2.06 |
| decode 16 | o | 16 | 7.10 | 2.00 | verify 16 | o | 128 | 7.26 | 1.82 |
| decode 16 | pooler | 16 | 21.70 | 1.54 | verify 16 | pooler | 128 | 24.98 | 1.71 |
| decode 16 | unpooler | 16 | 18.82 | 1.34 | verify 16 | unpooler | 128 | 19.95 | 1.36 |
| decode 16 | router | 16 | 14.78 | 8.19 | verify 16 | router | 128 | 16.48 | 7.45 |
| decode 32 | qkv_gate | 32 | 14.37 | 1.80 | verify 32 | qkv_gate | 256 | 19.49 | 1.99 |
| decode 32 | o | 32 | 6.78 | 1.87 | verify 32 | o | 256 | 9.25 | 2.07 |
| decode 32 | pooler | 32 | 21.81 | 1.54 | verify 32 | pooler | 256 | 30.90 | 1.78 |
| decode 32 | unpooler | 32 | 18.67 | 1.32 | verify 32 | unpooler | 256 | 24.08 | 1.39 |
| decode 32 | router | 32 | 15.62 | 8.38 | verify 32 | router | 256 | 17.66 | 6.60 |

Reading the `q_ref` columns: cuBLAS is at **48-67% of HBM peak** on the wide cells, and the router
(the baseline's 2-3 launches on 8.4 MB) is 5.9x-14.8x off floor.

The router is 20% of the weighted baseline time. The four wide layers are 80%, and their floor is
pure bandwidth (the floor's tensor-core FLOPs term never binds at these shapes).

## 4. The arithmetic of the gap

Solving the judge's formula against the 30 measured cells (`score` as a function of a uniform
multiplier on the four wide layers, with the router as measured):

| wide layers, relative to cuBLAS | score |
|---|---|
| 1.00x (parity) | 1.095 |
| 0.90x (10% faster) | 1.198 |
| 0.80x | 1.321 |
| 0.70x | 1.474 |
| 0.65x | 1.564 |
| 0.60x | 1.667 |

**We need the four wide layers 1.544x faster than cuBLAS on every cell, with the router as
measured.** Since cuBLAS achieves 48-67% of peak, that means a kernel at **83-100% of HBM peak**,
i.e. 85-95% in practice, under a protocol that flushes 256 MB of L2 before every timed replay.

Two consequences worth challenging:
* Beating cuBLAS by 1.5x is not "write a good GEMM". At these skinny shapes (M <= 256, N <= 8192,
  K <= 8192) cuBLAS is already near what a single-pass streaming kernel can do; the only headroom I
  could identify is cuBLAS's own `splitKreduce` launch (1.5-1.8 us per cell) and its generic tactic
  choice. cuBLAS's own profile, from the judge, per cell at m=1:
  `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` 10.5 us + `cublasLt::splitKreduce_kernel` 1.7 us;
  at m=16 `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` 8.0 + 1.7.
* The top fifty teams are therefore at 85-95% of HBM peak on all four layers, or they have
  something structural I have not found. That is the first thing I would want a second opinion on.

## 5. Everything tried, with the measurement that killed it

Five candidates were built and measured on the judge (`okc check`, 20/day quota, 8 used):

| # | candidate | judge score | what it did | why it lost |
|---|---|---|---|---|
| 1 | `2fecf61d8c9c` | **0.8086** | hand CUDA-core Split-K GEMV for wide m<=8 and the router at m<=8; cuBLAS above | `qkv_gate` m=8: **69.34 us vs cuBLAS 13.87** (5x slower); m=1: 18.58 vs 13.81. Scalar `fmaf` + bf16-convert is ~20 instructions per 32 B; one H200 SM issues ~3.6 G instr/s and has 36 GB/s, so the path tops out near 5-6 GB/s per SM |
| 2 | `cda20a06ee72` | **1.0908** | cuBLAS for every wide cell; router = one bf16 `torch.mm` on the stacked `[hi; lo]` plus an fp32 half-add | it is the baseline plus a router win. This is the current incumbent |
| 3 | `1cf95849abd9` | 1.0972 | same as #2, with the Hopper wide GEMM added | the Hopper extension silently failed to build (section 5c), so this run *is* #2; 1.0972 vs 1.0908 is run-to-run noise on the same effective code. Its value was diagnostic: the judge's own kernel profile showed cuBLAS running, which is how the silent fallback was found |
| 4 | `973...`/`1d34...`/`442...` | 0.0 (4 diagnostics) | build/self-check diagnostics for the Hopper kernel | see section 6 |
| 5 | `2cecd1086d0e` | **0.7977** | Hopper TMA + `wgmma` GEMM for all four wide layers, plus a fused split-K `wmma` router kernel (`kernel_router_tc.py`) instead of `torch.mm` | **0.48-0.96x on every wide cell**. The fused router also lost on H200 where `torch.mm` wins: 12.96 vs 8.75 us at m=1, 12.88 vs 9.30 at m=32, 20.68 vs 11.95 at m=128, 30.19 vs 12.49 at m=256 (its only wins are 1.17-1.18x at m=16/32). Correct: the judge's full test suite passed, 58 registers, 0 spills, >=128 CTAs per cell |

### The three wide-layer kernel designs in detail

All three aimed at the same thing: read the weight exactly once from HBM, keep the activation tile
in shared memory, one launch per cell.

**(a) CUDA-core Split-K GEMV** (`kernel_v5.py`, retired). Warp-per-weight-row, 128-bit `float4`
loads, `bf16x2` pairs accumulated into fp32 registers, 4 loads in flight per lane, staged
activation tile in shared memory, `__ldcs` (evict-first) for the weight. Measured 61-69% of the
**L4's** peak bandwidth — which is why it looked right — and then lost 5x on the judge. This is the
single most important methodological finding: **the L4 offers 5 GB/s per SM (300/60) and the H200
36 GB/s per SM (4800/132), so an L4 measurement of a compute-heavy kernel says nothing about the
judge.** It also produced two real bugs worth recording: an early `return` before `__syncthreads()`
(UB for the exited warp) and a recovery stride of `32*8*4 = 1024` elements against a 512-element
staged tile, which left lanes 16-31 outside the k loop entirely.

**(b) `wmma` + `cp.async`** (`kernel_wide_tc.py`). `nvcuda::wmma` 16x16x16, `cp.async` staged,
dynamic shared memory, padded shared stride (`BK+8`) to keep `ldmatrix` off bank conflicts, BK
swept 32/64/128/256, stages 2-8, warps 4/8/16, BM bands 16/32/64/128/256. Best L4 result 0.34-0.60
of L4 peak across the m bands, against a **cuBLAS control measured in the same run at 0.44-0.70**.
So on the L4 it is at parity with cuBLAS, not ahead. Things that did *not* help: 2 blocks/SM
(0.47-0.49), 8 or 16 warps (0.41-0.51), BN=32 at m=256 (0.31), K splits of 1/4/8 (+-1%), reading
activation fragments straight from L2 (0.27-0.31).

**(c) Hopper TMA + `wgmma`** (`kernel_wide_tma.py`). This is the canonical Hopper design and the
most informative failure. 64x64x64 tiles, 6-stage `mbarrier` pipeline in 96 KB dynamic shared
memory, one producer warp issuing `cp.async.bulk.tensor.2d` against `CUtensorMap`s built per call
(`cuTensorMapEncodeTiled` via `cudaGetDriverEntryPoint`), `SWIZZLE_128B`, and one consumer warpgroup
issuing four `wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16` per stage with
`wgmma.commit_group`/`wgmma.wait_group 1`; descriptors LBO=16 B, SBO=1024 B, swizzle mode 1; grid
`(n/64, ceil(m/64), ks)` with split-K + an fp32 reducer where the grid does not fill the 132 SMs.
0 spill stores / 0 spill loads / 58 registers at `sm_90a`. Every scored cell >= 128 CTAs.

It took four judge round trips to get running, each of which is a finding of its own:

1. **It silently fell back and looked finished.** First check: 1.0972x, every wide cell at exactly
   1.00x. The tell was the judge's own kernel profile: `nvjet_*_splitK` + `splitKreduce` — the
   extension had never built. Lesson applied: `prepare()` now raises on an sm_90 device instead of
   falling back.
2. **`-arch=sm_90a` is not enough.** torch's `load_inline` also emits its own `-gencode` set for the
   detected device, producing an extra `cuda.compute_90.ptx` target, and ptxas rejects `wgmma` there:
   `Instruction 'wgmma.mma_async with floating point types' not supported on .target 'sm_90'`.
   Fix: `--generate-code=arch=compute_90a,code=sm_90a`, which suppresses torch's pass. Reproduced
   byte-for-byte on the L4 by pinning `TORCH_CUDA_ARCH_LIST=9.0+PTX`.
3. **The judge truncates a prepare error at ~2 KB**, and nvcc prints command lines first, so the
   message was unreadable until `_BUILD_ERROR` kept the *last* 1400 characters.
4. **Split-K wrote through a null pointer**: the epilogue branched on `out == nullptr` as the
   "partials" selector and then handed that same null to the reducer. After that fix the classifier
   (which compares a 64x64x64 GEMM against `torch.mm` and names a transpose / half-k / row-half /
   block-permutation match) reported `base`, `pipe` and `real` levels clean over 128 stages and
   64 CTAs — the descriptors, swizzle, mbarrier discipline, wgmma layout and epilogue are all
   correct — and the full judge test suite passed.

And it is still **1.3-2x slower than cuBLAS on every wide cell** (0.48-0.96). That is the result I
would most like a second opinion on, because the technique is right and the tuning is not obviously
wrong: 6 stages of 16 KB is ~2.6 us of in-flight bytes against a ~0.8 us TMA latency, 1 CTA of
96 KB leaves room for a second CTA per SM, and every cell fills the machine.

## 6. Ruled out, with the reason

* **L2 residency / warm-cache tricks.** Our harness (`GraphRunner.timed`) zeroes a 256 MB buffer
  before every timed replay, and the judge's numbers behave the same way (cuBLAS at 48-67% of peak
  is what a cold weight stream looks like). If the judge's flush were much smaller, both the
  baseline and any candidate would be faster and the *ratio* would be unchanged - so this cannot be
  the source of the top-50's 1.5x, but it does mean the "floor" is not physically reachable and
  every candidate's `q` has a hard lower bound of roughly 1.2-1.3 on the small cells.
* **Reducing bytes.** The floor charges weight + activation + output; the weight must be read whole
  and used bit-exact (`PROBE_TOL=1e-3` on one-hot rows). No compression, no fp8, no approximation.
  The unpooler's 1024-row slice at m<=8 is already exploited.
* **Fewer launches.** Worth 1.5-1.8 us of cuBLAS's `splitKreduce` per cell, which is 10-20% of the
  small cells - real, and captured by the router win, but not 1.5x.
* **A different decomposition of the same bytes.** The activation re-read is the structural trap:
  a 64-column N tile means `n/64` blocks each read the activation, which at m=256 is 268 MB against
  a 67 MB weight. Mitigations tried: bigger BM (register-infeasible), persistent blocks, K splits
  (no effect on the L4), reading A from L2 (much worse).
* **Hopper-only instructions are allowed** (our verifier classifies a source with `wgmma` as
  `hopper_sm90a_specialized` and compiles it for `sm_90a` with a spill check), so this was not a
  sandbox limitation. The judge image is H200 / driver 580.159.03, torch under `/opt/sglang`.

## 7. What I would ask a stronger model

1. **Is 85-95% of HBM peak actually achievable on these shapes under a per-replay 256 MB L2 flush,
   and if so with what kernel?** Specifically M ∈ {1,8,16,32,128,256}, N ∈ {1024,2304,4096,8192},
   K ∈ {1024,4096,8192}, bf16 in/out, fp32 accumulate, weight read once. My three designs reached
   0.2x, 1.0x and 0.5-0.9x of cuBLAS. If a canonical CUTLASS `sm90` dense GEMM with
   `KernelTmaWarpSpecializedCooperative` and a 128x128 tile would beat cuBLAS here, the interesting
   part is *why* (tile shape? 2 CTAs/SM? the `ks` split?), because my 64x64 TMA design did not.
2. **Is the top of this board consistent with a per-cell `q` of ~1.1, or is there a structural trick
   in the measurement I have missed?** `#1` = 1.6498x against a floor that no kernel I wrote can
   approach. If the intended solution is not "beat cuBLAS on the wide layers", what is it?
3. **Is my reading of the flush right?** If the judge flushes L2 *once* rather than per replay, a
   kernel whose weight is L2-resident (16.8 MB `o`, 37.7 MB `qkv_gate` both fit in 50 MB) would beat
   the HBM floor outright - and that would explain the board's top. I have no way to see the judge's
   harness; my evidence is only that cuBLAS measures 48-67% of peak, which is cold-stream behaviour.
4. **Is `muse_decode_layer` the better instrument?** Our own plan's arithmetic says yes: `#50` =
   2.2252 against a wrapper-only ceiling of 2.650 (`marlin_moe` 49.9%, cuBLAS 21.3%, FA3 14.9% of
   weighted time), i.e. reachable in principle with 89% of the non-Marlin time removed, whereas
   `muse_attention_decode` (FA3 82.5%, ceiling 1.348 against `#50` 2.677) and `muse_moe_w4a16`
   (Marlin 87.5%, ceiling 1.253 against `#50` 2.420) are arithmetically impossible by wrapper work.
5. **Any reason to distrust the router win?** One bf16 `torch.mm` on the stacked `[hi; lo]` (512
   rows) plus `both[:, :n] + both[:, n:]` gives 1.18x/2.87x/1.59x/1.57x/1.38x/1.41x at
   m=1/8/16/32/128/256, and it passes the fp32 `ROUTER_TOL=1e-4` probe-row test. It looks free, so
   if it is not, I would like to know before it goes on a board.

## 8. Reproduction

```bash
# the board, free and quota-free
PYTHONPATH=src .venv/bin/python -c "
from okc_loop import ledger as L, api_client as A
c = A.OkcClient(L.Ledger('artifacts'))
print(c.cutoffs('muse_bf16_proj', ranks=(1,50)))"

# any check's full per-cell truth, including the judge's kernel profile for our own submission
okc status <job_id> --json

# one priced check of a candidate (Costs 1 of 20/day; the ledger enforces tier-0 + tier-1 first)
PYTHONPATH=src .venv/bin/python -m okc_loop.cli lint  <sha|path>
PYTHONPATH=src .venv/bin/python -m okc_loop.cli colab-batch <sha|path> --problem-of muse_bf16_proj
PYTHONPATH=src .venv/bin/python -m okc_loop.cli check <sha|path>

# the candidates, in the order they were measured
src/okc_loop/candidates/bf16_proj/kernel_v5.py        # incumbent, 1.0908x
src/okc_loop/candidates/bf16_proj/kernel_router_tc.py # fused router, +0.006, worse at m>=128
src/okc_loop/candidates/bf16_proj/kernel_wide_tc.py   # wmma+cp.async wide GEMM, cuBLAS parity
src/okc_loop/candidates/bf16_proj/kernel_wide_tma.py  # TMA+wgmma Hopper GEMM, 0.7977x overall
docs/PROGRESS.md                                      # the full checkpoint, per-cell tables included
```

Quota state: 8 of 20 checks/day used, 0 of 10 submits. The first-upload gate for this problem is
`auto_submit.MILESTONE_FLOOR["muse_bf16_proj"] = 1.56`, not the 1.15 that `AGENTS.md` states, so
nothing can be uploaded below the board line and no board attempt has been made.
