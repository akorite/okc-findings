# Engineering report: Muse Spark Open Kernel Challenge, both fronts

**Date:** 2026-10-31. **Deadline:** 2026-11-04 23:59 PT. **Purpose:** a second opinion from a
stronger model on two specific questions and one decision.

Every performance number here comes from the judge's own `okc check` (job id given), the
leaderboard API, or the judge's own kernel profile of our submission. Reproduce any of them with
`okc status <job_id> --json`. Nothing in this report is estimated from a proxy GPU unless it says
so, and where it is, the proxy is named.

---

## 1. Executive summary

| board | our best | rank | `#50` line | verdict |
| --- | --- | --- | --- | --- |
| `muse_bf16_proj` | **1.0970x** (submitted, job `f749b2bb49ae4d99`) | **1387** / 1504 | 1.5718x | needs the four cuBLAS cells **1.544x faster than cuBLAS** |
| `muse_decode_layer` | **1.1177x** (job `c75f1c58c8734f24`), +0.7% merged and unchecked | none yet (floor 1.15) | 2.2252x | measured ceiling **1.239x** with Marlin/FA3/cuBLAS intact |

Three hand-written GEMMs lost to cuBLAS on three different GPUs. Two decode-layer fusion milestones
landed and moved the score 11.8% + 0.7%. The two board lines both require beating a kernel whose
authors are NVIDIA, and in both cases the arithmetic says the remaining legal headroom is roughly
10%, not the 43-99% the objectives ask for.

The two questions I cannot answer without Hopper hardware:

1. Can a hand-written dense GEMM reach 85-95% of HBM peak on `M <= 256, N <= 8192, K <= 8192` BF16
   on H200? (cuBLAS gets 48-67% there; my three attempts got 24-58% of peak and lost.)
2. Can a paged FP8 GQA attention kernel for `T <= 256` beat FA3 by 2x on full-attention loads? FA3
   is 34-61% of those cells and is tens of times off the KV-bandwidth floor at `T = 32`.

---

## 2. How the judge scores, exactly

Per problem: 6 loads x N layers. Per cell the judge takes the median of ~60 CUDA-graph replays and
computes `ratio_i = baseline_ns_i / median_ns_i`; per load `T_load = sum(weight_i * ns_i)/1000` us;
the problem score is the **geometric mean over the six loads** of `T_ref_load / T_cand_load`. The
reported microsecond figure is `sglang_us / score`. My reimplementation reproduces the judge
bit-exactly (replaying the baseline's own medians gives 1.005957 against a reported 1.005957).

Measured protocol facts that constrain every design:

* **256 MB of L2 is zeroed before every timed replay**, so the weight is re-read from HBM each
  iteration and no warm-cache trick exists. (`GraphRunner.timed()`, and the judge's behaviour is
  consistent with it: cuBLAS at 48-67% of peak is cold-stream behaviour.)
* The floor `sol_us` uses `hbm_bytes_per_s = 4.8e12`, `bf16_dense_flops_per_s = 989e12`.
* Tolerances: `PROBE_TOL = 1e-3` on 1-in-8 power-of-two probe rows (so weights must be used
  bit-exact: no fp8, no compression, no approximation), `REL_L2_TOL = 1e-2` elsewhere, and
  `ROUTER_TOL = 1e-4` on the router (bf16 weights alone fail it; a `hi`/`lo` pair passes).

## 3. `muse_bf16_proj`: what is measured

Baseline is `_bf16_gemm_dispatch_impl` — cuBLAS — at 96.7% of weighted GPU time (job
`0281313b02ef46c2`). Per cell, `ref_us` = baseline, `q = ref/floor` (job `9dbc5a8301024dfa`):

| load | layer | m | ref_us | q | load | layer | m | ref_us | q |
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

Scaling the four wide layers against those cells (router as measured), the score is:

| wide layers relative to cuBLAS | score |
| --- | --- |
| 1.00x (parity) | 1.095 |
| 0.90x | 1.198 |
| 0.80x | 1.321 |
| 0.70x | 1.474 |
| 0.65x | 1.564 |
| 0.60x | 1.667 |

**`#50` needs 1.544x faster than cuBLAS on every wide cell.** That means 83-100% of HBM peak.

### 3.1 Every attempt on that lever, and what it measured

| # | design | evidence | outcome |
| --- | --- | --- | --- |
| 1 | CUDA-core Split-K GEMV: warp-per-weight-row, `float4` 128-bit loads, `bf16x2` pairs into fp32, 4 loads in flight, staged activation tile, `__ldcs` | 61-69% of **L4** peak | judge **0.8086x**; `qkv_gate` m=8 = 69.34 us vs cuBLAS 13.87. Instruction-bound: ~20 instructions per 32 B against 3.6 G instr/s per SM and 36 GB/s per SM |
| 2 | `wmma` 16x16x16 + `cp.async`, dynamic smem, padded stride (`BK+8`), BK 32-256, 2-8 stages, 4-16 warps, BM bands | 0.34-0.60 of L4 peak vs cuBLAS's 0.44-0.70 **on the same VM** | parity, i.e. 1.0x. A100 re-test (§3.2) confirms the ordering |
| 3 | Hopper TMA + `wgmma`, **transposed operands**: weight `W[n0:n0+64, k]` on the `m64` axis, tokens on `n8`/`n16`/`n32` (`wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16`), 6-11 stages, 96-100 KB dynamic smem, last-CTA `atomicAdd` lock reduction replacing `splitKreduce`, `CUtensorMap`s built once in `prepare()`, 0 spills / 58 regs | correct on the judge (8-level numeric probe), `m>=128` correctly delegated to cuBLAS | judge **0.8669x**: **slower than cuBLAS on 15 of the 16 cells it touched**. Only win: `router@m=1` 7.87 vs 8.75 us |
| 4 | Router as one BF16 GEMM on the stacked `[hi; lo]` (`torch.mm`, fp32 accumulate) | **kept** | 1.18x/2.87x/1.59x/1.57x/1.38x/1.41x at m=1/8/16/32/128/256. Whole problem 1.0908x → board 1.0970x |

Two bugs in attempt 3 are worth recording because they cost judge round trips, not because they
were subtle: torch's `load_inline` emits its own `-gencode` set, so `-arch=sm_90a` alone still
produced `cuda.compute_90.ptx`, where ptxas rejects `wgmma` ("not supported on .target 'sm_90'") —
fixed with `--generate-code=arch=compute_90a,code=sm_90a`; and the split-K epilogue branched on
`out == nullptr`, passing that null pointer to the reducer (illegal memory access).

### 3.2 The A100 experiment (new, and decisive)

The question "can a hand-written GEMM beat cuBLAS at these shapes" was never measurable on the L4
(5 GB/s per SM, 7x below the H200, and it lacks `sgl_kernel`/FA3/Marlin). It **is** measurable
on a Colab A100: `A100-SXM4-40GB`, sm_80, 108 SMs, 1215 MHz max memory clock, i.e. **14.4 GB/s per
SM** — 2.5x below the H200 instead of 7x. Same `GraphRunner` protocol, same flush, same scored
shapes, `run_candidates(gpu="A100")`:

| load | layer | m | cuBLAS us | hand us | hand/cuBLAS | cuBLAS frac of A100 peak |
|---|---|---|---|---|---|---|
| decode 1 | qkv_gate | 1 | 59.39 | 62.46 | 1.05 | 0.41 |
| decode 1 | o | 1 | 27.65 | 34.82 | 1.26 | 0.39 |
| decode 1 | pooler | 1 | 82.94 | 107.52 | 1.30 | 0.52 |
| decode 1 | unpooler | 1 | 18.43 | 23.55 | 1.28 | 0.29 |
| decode 1 | router | 1 | 23.55 | 24.58 | 1.04 | 0.23 |
| decode 16 | qkv_gate | 16 | 54.27 | 63.49 | 1.17 | 0.45 |
| decode 16 | pooler | 16 | 82.94 | 98.30 | 1.19 | 0.52 |
| decode 32 | qkv_gate | 32 | 54.27 | 60.42 | 1.11 | 0.45 |
| decode 32 | pooler | 32 | 82.94 | 109.57 | 1.32 | 0.52 |
| decode 32 | unpooler | 32 | 78.85 | 105.47 | 1.34 | 0.55 |
| verify 1 | qkv_gate | 8 | 53.25 | 58.37 | 1.10 | 0.46 |
| verify 16 | qkv_gate | 128 | 62.46 | 108.54 | 1.74 | 0.41 |
| verify 16 | pooler | 128 | 92.16 | 210.94 | 2.29 | 0.48 |
| verify 16 | unpooler | 128 | 83.97 | 195.58 | 2.33 | 0.52 |
| verify 32 | qkv_gate | 256 | 71.68 | 148.48 | 2.07 | 0.38 |
| verify 32 | pooler | 256 | 108.54 | 286.72 | 2.64 | 0.42 |
| verify 32 | unpooler | 256 | 106.50 | 231.42 | 2.17 | 0.42 |

Read it as: on a data-centre GPU that is a 2.5x proxy rather than 7x, cuBLAS reaches 23-56% of peak
and the hand kernel is **never faster on a wide cell** (worst 2.6x slower at m=256). The only
parity-or-better rows are the router at m>=16 (1.00, 0.92).

**So the wide-layer lever has now been measured on three GPUs (L4, A100, H200) and cuBLAS wins on
all three.** The board line requires beating it by 1.54x. That is the state of the world, and it is
the reason bf16_proj is parked at 1.0970x.

---

## 4. `muse_decode_layer`: what is measured

Baseline unmodified, job `8b0625fff6fe49ad`: score 1.00083, 13/13 tests. 12 scored cells = 6 loads
x 2 kinds, weight 46 (sliding) and 16 (full). Judge's own profile of that run, top kernels:

| load | Marlin | FA3 | cuBLAS (`nvjet_*`) | FP32 router GEMM |
| --- | --- | --- | --- | --- |
| decode 1 sliding / full | 41.0 / 45.4 | 9.4 / 25.0 | 49.1 / 49.6 | — |
| decode 16 sliding / full | 116.3 / 86.4 | 13.8 / 189.2 | 73.4 / 74.6 | — |
| decode 32 sliding / full | 113.3 / 124.2 | 18.5 / 371.7 | 77.6 / 97.3 | — |
| verify 1 sliding / full | 66.8 / 86.5 | 9.9 / 26.2 | 51.0 / 46.0 | 22.0 / 22.2 |
| verify 16 sliding / full | 354.3 / 389.8 | 19.0 / 195.4 | 83.2 / 82.8 | — |
| verify 32 sliding / full | 554.4 / 531.7 | 23.6 / 376.4 | 104.1 / 138.8 | — |

### 4.1 What we shipped, and what each change was worth

Candidate `composite.py`: baseline forked, Marlin/FA3/cuBLAS calls byte-identical. Judge job
`c75f1c58c8734f24` = **1.1177x**, `score_us` 20239.0:

| load | kind | ref_us | cand_us | x |
| --- | --- | --- | --- | --- |
| decode 1 | sliding / full | 170.9 / 179.5 | 142.0 / 152.3 | **1.204 / 1.178** |
| decode 16 | sliding / full | 257.4 / 432.8 | 218.1 / 394.3 | 1.180 / 1.098 |
| decode 32 | sliding / full | 261.6 / 614.0 | 250.6 / 605.4 | 1.044 / 1.014 |
| verify 1 | sliding / full | 260.0 / 261.2 | 194.1 / 197.5 | **1.339 / 1.323** |
| verify 16 | sliding / full | 542.9 / 722.8 | 531.2 / 712.4 | 1.022 / 1.015 |
| verify 32 | sliding / full | 754.8 / 1109.3 | 745.3 / 1099.7 | 1.013 / 1.009 |

| change | mechanism | measured / predicted |
| --- | --- | --- |
| attention prologue fused 5 launches -> 1 | qk-RMSNorm (`head_dim=64`) + interleaved RoPE from the prebuilt `[2^21,64]` cos\|sin table + saturating FP8 cast + KV scatter + FP8 `q` cast, one Triton kernel; verified bit-exact against a torch oracle of `FusedQKNorm` (13/13 cases, T in {1,3,5,8,33}, rope on/off, `kv_scale` 1 and 1/64) | in the 1.1177 |
| post-expert finalize 14 launches -> 1 for `T < 32` | `_MIN_TOKENS_FOR_FUSED = 32` confirmed at `fused_post_expert_norm_weighted_sum_triton.py:7,77`; below it the wrapper runs 14 eager kernels. Now the module's own kernel is called directly at every T | in the 1.1177; judge's oracle accepted it at T = 1, 8, 16, 32, 128, 256 |
| router: one BF16 GEMM on `[hi; lo]` at every T | replaces the FP32 `F.linear` below 9 tokens and the two BF16 mms above; the FP32 `router_input` copy disappears | in the 1.1177; the router is *overlapped* with the pooler on a side stream, so only its exposed part counts — the 2.87x measured on bf16_proj does **not** transfer, worth ~1.2% here |
| `ffn_boundary` + `F.pad` tail fused, 4 launches -> 2, `T <= 8` | RMS over the 1024 live columns with `sum_sq_1024/8192 + 1e-8` so it equals the zero-padded 8192-wide reduction; weight-first `(1+w)`, two bf16 rounding points, channel-wise fp32 `alpha/beta` residual, weight-last `rmsnorm_1e-5` | **+0.7%** measured against the model, predicted geomean 1.126. Residual columns 1024..8191 bit-identical (11/11 cases); `hidden` bit-identical 9/11, one element of 8192 within one bf16 ulp in the other two; the module's own kernel is not bit-exact against `reference.py` either |
| post-attention gate + `attn_boundary` fused | `o_norm_gate` (`rmsnorm_1e-8(a) * sigmoid(gate)`) and the boundary norm in one launch | built, Tier-0 clean, in verification |

### 4.2 The ceiling, and why 2.2252x is out of reach as specified

From **our own submission's profile**, per cell: the time the three protected kernels consume, and
what would remain if every other kernel *and* every launch gap vanished:

| load | cand_us | protected_us | removable_us |
| --- | --- | --- | --- |
| decode 1 sliding / full | 142.0 / 152.3 | 103.9 / 119.2 | 38.1 / 33.1 |
| decode 16 sliding / full | 218.1 / 394.3 | 189.0 / 368.7 | 29.1 / 25.6 |
| decode 32 sliding / full | 250.6 / 605.4 | 247.2 / 697.2 | 3.4 / — |
| verify 1 sliding / full | 194.1 / 197.5 | 159.2 / 172.1 | 34.9 / 25.4 |
| verify 16 sliding / full | 531.2 / 712.4 | 537.3 / 633.9 | — / 78.5 |
| verify 32 sliding / full | 745.3 / 1099.7 | 693.7 / 1049.8 | 51.6 / 49.9 |

> **Ceiling 1.239x. Current 1.1177x (+0.7% merged). Target 2.2252x.**
> Legal headroom ~11%; the target is 1.8x beyond the ceiling.

For reference, the objective's "2.650x wrapper-only ceiling" reproduces exactly **only when FA3 is
scaled to zero** — i.e. it silently counted FA3 as improvable. With FA3 intact the number is 1.928
if cuBLAS is also improvable, and 1.239 if cuBLAS is protected too (the objective protects it).

Why the three cannot be beaten by us:

* **Marlin** streams 1.45-1.8 TB/s of a 4.8 TB/s HBM (804 MB re-read per call at `verify 32`) and
  its bytes are irreducible. Route compaction is *already inside it*:
  `moe_wna16_marlin.cuh:124` `if (tmp_expert_id == -1) continue;`, `marlin_template.h:386-393`
  `parallel = num_valid_blocks`, `activation.py:114-117` skips dead rows, and the finalize has a
  `HAS_IDS` guard. Its workspace cannot be shrunk: the grid is `sms * blocks_per_sm` from occupancy
  and it writes `locks[blockIdx.x]` into that buffer. (`grep local_count` on
  `moe_wna16_marlin.py` returns nothing — there is no such parameter.)
* **FA3** is invoked as tightly as the API allows already (`page_table`, `cache_seqlens`,
  `causal=True`, `window_size=(window-1, 0)`, FP8 descale), and dominates the `full` loads.
* **cuBLAS** won on three GPUs (§3.2).

The FA3 opportunity that *does* exist: at `decode 32 full` it costs 371-501 us while the KV the
window needs is ~25 MB, i.e. ~5 us at peak. It is a large-batch kernel serving 32 queries. A paged
FP8 GQA kernel for `T <= 256` is the only lever that lifts the ceiling — and the objective forbids
it.

---

## 5. Blockers, itemised

**Blocked by arithmetic (cannot be fixed by effort under the objective's own invariants):**

| # | blocker | evidence | status |
| --- | --- | --- | --- |
| B1 | `muse_decode_layer >= 2.2252x` with Marlin/FA3/cuBLAS intact | ceiling 1.239x from our own profile | **hard blocker** |
| B2 | `muse_bf16_proj >= 1.45x` requires 1.544x over cuBLAS | three kernels, three GPUs, best 0.87x | **hard blocker** |

**Objectively impossible instructions (would waste quota if attempted):**

| # | instruction | why it is void |
| --- | --- | --- |
| V1 | 2A: pass `local_count=32`, shrink the Marlin workspace | no such parameter; workspace grid = `sms * blocks_per_sm`, `locks` written into it — shrinking corrupts memory at `T <= 16` |
| V2 | 2D: compact active local routes before Marlin | Marlin already skips `-1` blocks; the padded slots cost one loop iteration each |
| V3 | 2B: "compute only `unpooler[:1024]` at `T <= 8`" | already what the baseline does (`dense["unpooler"][:UNPOOL_SLICE]` + `F.pad`) |
| V4 | 2C: "zero the 536 MB RoPE table allocation" | the table is built in `prepare()`, which is untimed and capped at 180 s; it costs memory, not replay time |

**Real blockers with a known fix, needing a decision or hardware:**

| # | blocker | what it would unblock |
| --- | --- | --- |
| C1 | **No Hopper GPU.** Colab refuses H100 (`Backend rejected accelerator 'H100'`); the L4 cannot execute `sm_90a`; the A100 is sm_80. Every `wgmma`/TMA change costs a judge round trip | profiling instead of inference for the one lever that could beat cuBLAS |
| C2 | Objective forbids replacing FA3/cuBLAS cells | the only levers that lift either ceiling |
| C3 | A production-grade CUTLASS/CuTe `sm90` GEMM has never been measured on these shapes | settles B2 in a day with a Hopper node, or with `cutlass` installed on an A100 for the `sm80` equivalent |

**Achieved (evidence in hand):** floor recalibration + first upload (submit `f749b2bb49ae4d99`,
board 1.0970x, rank 1387); working rank reader (`board_rank`, positional + team match, with a test);
`lint` records its tier-0 verdict (with a test); single-flight stale-lock reclaim
(`INFLIGHT_LEASE_S = 1800`, with a test); decode 1.1177x with all 12 cells moved as predicted; the
A100 proxy quantified.

**Quota:** bf16_proj 2 checks + 1 submit used today; decode 1 check, 0 submits; Colab 336 CU of
340; tests 758 passing; deadline 4 days out.

---

## 6. What I would do with each kind of more compute

| ask | what it unlocks | expected value |
| --- | --- | --- |
| **Hopper node 1-2 h/day** (H100 PCIe is enough) | `ncu` on the TMA/`wgmma` kernel; decide in an hour whether it can beat cuBLAS, and if not, which of these it is: 1 CTA/SM starving the pipe, the activation re-read at m>=16, the serial last-CTA reduction, or the 64-row tile being wrong for m<=8 | settles B2; currently the single most expensive unknown |
| **Permission to replace FA3 on the 4 `full` cells** | a paged FP8 GQA kernel for `T <= 256`; FA3 stays on sliding | the only path to >1.24 on decode_layer; the plan's U6 expected 2.1-2.4x on the attention problem |
| **`cutlass` on the A100** (already entitled) | an `sm80` CuTe/CUTLASS dense GEMM at these shapes vs cuBLAS, on a 2.5x proxy | a cheap, immediate read on whether a *production* GEMM beats cuBLAS where my from-scratch ones did not |
| more Colab CU | nothing, it is not the constraint (336 of 340 left) | — |
| more H200 checks | nothing; 18/day remain unused | — |

---

## 7. Questions for the second opinion

1. **Is 85-95% of HBM peak reachable on `M <= 256` skinny BF16 GEMMs on H200 under a per-replay
   256 MB L2 flush?** If yes, with which tile/stage/warp-specialisation — and what is the most
   likely reason my 64x64 TMA+`wgmma` design (6 stages, `n8/n16/n32` token tiling, >= 128 CTAs,
   0 spills) measured slower than cuBLAS on 15 of 16 cells, e.g. `decode 32 qkv_gate` 26.71 us
   against 14.30?
2. **What is `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` doing that a hand kernel cannot?** A
   128-row weight tile, 8 tokens, K=64 slices, 12 stages, 4x1 clusters, split-K: 10.5 us for
   37.7 MB of weight plus 1.7 us of `splitKreduce` = 3.1 TB/s at m=1. My best was 15.24 us.
3. **Is the 1.239x decode ceiling right?** It is two minutes to re-derive from the profile table in
   §4.2; disagreement there would change the whole plan.
4. **Cheapest legitimate way to lower FA3 at `T <= 256`** without replacing it: different
   `scheduler_metadata`, varlen packing of 32 queries, avoiding the `.expand()`-ed non-contiguous
   FP8 descale tensors, or a CUTLASS paged-attention instantiation tuned for small T?
5. **Should the effort stop?** If both board lines require beating NVIDIA's own libraries at
   shapes where they are near the hardware limit, the expected value of more checks here is zero,
   and the effort belongs on a board whose line sits under its own arithmetic.

---

## 8. Reproduction

```bash
# our board position, and the cutoff being chased
PYTHONPATH=src .venv/bin/python -c "
from okc_loop import ledger as L, api_client as A
c = A.OkcClient(L.Ledger('artifacts'))
print(c.board_rank('muse_bf16_proj'), c.cutoffs('muse_bf16_proj', ranks=(50,)))"

# any check's full per-cell truth plus the judge's kernel profile of our own submission
okc status <job_id> --json

# the candidates, in the order they were measured
src/okc_loop/candidates/bf16_proj/kernel_v5.py           # incumbent: 1.0908x (board 1.0970)
src/okc_loop/candidates/bf16_proj/kernel_wide_tc.py      # wmma+cp.async wide GEMM (parity)
src/okc_loop/candidates/bf16_proj/kernel_wide_tma.py     # TMA+wgmma transposed, 0.8669x, correct
src/okc_loop/candidates/decode_layer/composite.py        # 1.1177x + ffn_tail (+0.7%)
src/okc_loop/candidates/decode_layer/post_attn_fused.py  # post-attention fusion, in verification
docs/PROGRESS.md                                          # checkpoint with every table
docs/reports/2026-10-31-second-opinion-brief.md           # the bf16_proj deep dive
```

Proxies and their measured calibration, so nobody repeats the mistake:

| GPU | GB/s per SM | role |
| --- | --- | --- |
| Colab L4 (sm_89) | 5 | correctness, spills, graph replay **only** — 7x low, misled us into shipping a GEMV that lost 5x |
| Colab A100 (sm_80) | 14.4 | **new**: ranking proxy for bandwidth-bound kernels; cuBLAS 23-56% of peak there |
| H200 (sm_90a) | 36 | the judge; cuBLAS 48-67% of peak there |
