# research_02 — general kernel-repo sweep + analyzers/tooling (2026-11-03)

Scope delivered: the operator's list (CUTLASS examples, cuda-samples-class Hopper examples, MatX/cccl class,
attention-gym/flash-attention/FA3-Hopper, Triton tutorials, Triton kernel collections, mit-han-lab, KernelBench,
gpu-mode, HF kernels, SGLang/vLLM/TRT-LLM kernels, FlashInfer, DeepGEMM, MoE repos, TileLang/TVM, competition repos,
Chinese-ecosystem kernels) **plus** ncu/nsys parsers, Nsight explainers, SASS/instrumentation, roofline/occupancy
calculators, benchmark harnesses. Filtered for one thing only: **what moves our four boards on one H200 (sm_90)**.
Anything sm_100-only is recorded as dead and not priced.

Method: 16 repos cloned locally with `--depth 1 --filter=blob:none` (sparse where the repo is large), 4 more read via
API/HTML (PRs, papers, one search sweep); 8 further repos were surfaced by search and are cited but **not inspected**
(marked). No `okc`, no `colab`, no GPU. Every row below carries a path inside one of those trees plus the URL, so it is
re-findable; our-side facts are cited to reports already in `artifacts/reports/`.

## Repo inventory (what "inspected" means here)

| # | repo | how | used for |
|---|---|---|---|
| 1 | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) `examples/`, `include/cutlass/gemm/collective/` | cloned, sparse | F1 mixed-input `rs` design, F4 grouped, F2 low-latency GQA |
| 2 | [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) `cpp/tensorrt_llm/kernels/` | cloned, sparse | F1 MoE (min-latency, W4A16 dispatchers), F3 GEMV |
| 3 | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) `include/flashinfer/attention/` | cloned, sparse | F2 scheduler/cascade/pod |
| 4 | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) `hopper/` | cloned, sparse | F2 combine/bulk-reduce |
| 5 | [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | cloned, full | F1/F4 kernel inventory, F2 MQA |
| 6 | [novitalabs/chord](https://github.com/novitalabs/chord) | cloned, full | **F1 W4A16 MoE on H200 (design.md + performance.md)** |
| 7 | [vllm-project/humming](https://github.com/vllm-project/humming) | cloned, full | F1 `indexed` W4A16 MoE (chord's upstream) |
| 8 | [cloudflareresearch/unweight-kernels](https://github.com/cloudflareresearch/unweight-kernels) + [report](https://research.cloudflare.com/papers/unweight-2026.pdf) | cloned, full + PDF | **F3 lossless BF16 compression matmul** |
| 9 | [triton-lang/triton](https://github.com/triton-lang/triton) `python/tutorials/`, `python/triton_kernels/` | cloned, sparse | F1/F4 block-scaled matmul, PDL, persistent matmul |
| 10 | [mit-han-lab/ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) | cloned, full | **analyzers: ncu parse + stall attribution + playbook** |
| 11 | [Enigmatisms/tachyon](https://github.com/Enigmatisms/tachyon) | cloned, full | analyzers: rule engine, profile diff, MCP |
| 12 | [LancerLab/kebab](https://github.com/LancerLab/kebab) | cloned, full | microbench rig: raw WGMMA / mma.sync / 2D TMA copy cost |
| 13 | [gpu-mode/discord-cluster-manager](https://github.com/gpu-mode/discord-cluster-manager) | cloned, full | harness family (KernelBot runners) — cross-ref only |
| 14 | [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench) | cloned, full | methodology only (levels 1-4, no Hopper mechanism) |
| 15 | [osayamenja/FlashMoE](https://github.com/osayamenja/FlashMoE) | cloned, full | single-kernel distributed MoE = megakernel class |
| 16 | [sgl-project/sglang](https://github.com/sgl-project/sglang) | cloned, sparse — **branch has no `sgl-kernel/`** | — (see #18) |
| 17 | [PR sgl-project/sglang#7772](https://github.com/sgl-project/sglang/pull/7772) | read (merged) | **F1 W4A8 Hopper MoE, ByteDance co-authored** |
| 18 | [Docs: nsight-python extraction](https://docs.nvidia.com/nsight-python/extraction.html), [KuangjuX/ncu-cli](https://github.com/KuangjuX/ncu-cli), [OptimAI-Lab/CudaForge](https://github.com/OptimAI-Lab/CudaForge/blob/main/run_ncu.py) | search hits, **not inspected** | cited only |
| 19 | [HPC4AI/HopperGEMM](https://github.com/HPC4AI/HopperGEMM), [microsoft/MMA-Sim](https://github.com/microsoft/MMA-Sim), [loostrum/wgmma_interface](https://github.com/loostrum/wgmma_interface) | search hits, **not inspected** | cited only |
| 20 | [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples), [NVIDIA/MatX](https://github.com/NVIDIA/MatX), [NVIDIA/cccl](https://github.com/NVIDIA/cccl), [pytorch-labs/attention-gym](https://github.com/pytorch-labs/attention-gym), [huggingface/kernels](https://github.com/huggingface/kernels), [TileLang](https://github.com/tile-ai/tilelang), [mit-han-lab KernelWiki](https://github.com/mit-han-lab/kernel-wiki) | **not inspected this pass** | explicitly out of the result set — nothing known from them at this granularity; noted so the next agent does not re-run |

Family codes: **F1** = NVFP4 W4A16 MoE decode (bs 1-32, 2-phase w13→w2, 32 local experts × 28.31 MB); **F2** = paged FP8
decode attention (hd 64, GQA 8:1, window 2048, page_size 1); **F3** = BF16 GEMV/GEMM m 1..32 (weight-BW bound, our 12-bit
lossless codec); **F4** = grouped NVFP4 prefill GEMM (math-bound, n=256 token tiles).

---

## Table A — kernel repos: mechanism → family → do we already? → verdict

| item (path + URL) | mechanism (as the source states it) | family | already ours? | verdict, priced |
|---|---|---|---|---|
| [chord `docs/design.md`](https://github.com/novitalabs/chord/blob/main/docs/design.md) "Indexed kernel / Swap-AB decode" | weight on `mma`-M, tokens on `mma`-N, "so a few-token block is one `m16n8k16` and block-M 8 is legal"; H200 EP8 decode = `(8\|16\|24, 256, 64)` swap-AB, one-pass, **3 CTAs/SM** (down block-M 8: 4) | F1 | **no** — our small bands are `nw8+mqmma.sync` with wgmma-class tiles, block-M 16 fixed (`moe_struct.md` §0, `moe_w4a16.py` `UNIT_BYTES 36864`, 12288 rows/exp) | **yes — the only item in the sweep that prices a different design.** Their own like-for-like measurement: "decode is **4–29% faster** through routed_m 888 at 48 experts" vs *their* wgmma grouped path at the same shape/bytes/GPU. Our d1/v1/d16 are all inside the sub-18.75-rows/expert band where they always pick swap-AB. Midpoint 15% on 3 of 6 loads ⇒ `1.15^(3/6)` = **+7.2% MoE board** (range +2% at 4%, +13.6% at 29%). ≈ 730 ranks at 101 ranks/%. Cost: a *second* decode kernel for the three small bands (not a knob); packed layout already serves both per their note "one pack serves every schedule". |
| same file, "H200 decode: `(8\|16\|24,256,64)` … **3 CTAs/SM**"; plus "Residency: small masked tiles (`block_m <= 24`) run two CTAs per SM — a persistent grid of `num_sms*2`, halved smem ring … **which hides the single TMA producer's latency**" | multi-CTA/SM residency for the small tile bands, paid for with a halved smem ring | F1 | **no** — 1 CTA/SM by construction (226 KB WGS / ~200 KB skinny; `moe_struct.md` §2) | **cheap-ish, unpriced by its author.** This directly contradicts our own §2 verdict ("2 CTAs/SM = 0%", `moe_struct.md`) — but our §2 reasoned from MLP-sufficiency, and chord ships it *for the TMA-producer latency*, not for MLP. Nearest legal build is the one `moe_struct.md` §2\#3 already computed: drop the dead `xb` (200→132 KB) + `OKC_NS=2` ⇒ 98 KB/CTA ⇒ 2 CTAs/SM. Our own model then says d1 ≤ +15% ⇒ **+0.75% board**, chord's *mechanism* says more. Verdict: **the best cheap check on the board** (≤1 file, host-only grid change), expected 0…+4%. |
| same file, "**WGMMA `wait<1>`** … one warp-K iteration of WGMMAs per commit, one group in flight over the next load and dequant: **3-6% on gate/up, 1-5% on down**" | commit-group width: exactly one group in flight while the *next* stage is loaded+dequantized | F1 | **no** — we flush every `OKC_WGSFCI=3` (layer) / `3` (cas) commits (`moe_struct.md` §3, `muse_decode_layer.py:5599`, `:6057`) | **yes, ~1 line.** 3 of 6 loads carry the WGS build ⇒ +3.5% × 3/6 ≈ **+1.75% MoE board** (their range ⇒ +0.5…+3%). Free to A/B; no new code path. |
| same file, "Semi-static token tiles. Tile `j=0` is unconditional, only `j >= 1` is guarded: **−7% cycles**, 186 vs 216 µs against fully dynamic" | per-job guard hoisting for the first unit | F1 | partly — our `nS = njS·RB_A − dyn_pool` path guards every unit (`moe_struct.md` §1) | **yes, small**: our job count at the timed shapes is 1 per expert (`FT_T=64` vs ~14 tokens/expert), so the guard count is tiny ⇒ 0%. Only `dist="skewed"` sees it. |
| same file, "Fused dequant `lop3 → hsub2 → hmul2`; subtract-then-scale keeps `w-8` exact, single rounding" | int4→half2 dequant with zero-point folded into a SIMD2 subtract, scale in `hmul2` | F1 | **yes, cheaper form** — `dq4` + `SCD/SCDK` fold the block scale into the bf16 **exponent field** by integer add (`muse_decode_layer.py:4170`, `4377-4384`), no mul at all | 0%. Their 3-instr/8-value is not better than our PRMT+IADD/8-value, and our tolerance budget is already spent. **Do not chase.** |
| [cutlass `include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp) + [`examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_bf16_gemm.cu`](https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_bf16_gemm.cu) | the canonical W4A16 arrangement: narrow type passes through the **register file** (`rs`), scale multiply in the mainloop, epilogue after an implicit swap; `TileShape <128,128,64>`; offline nibble reorder `[0,2,4,6,1,3,5,7]` so 16 K-elements a thread needs are contiguous | F1 | **yes** — ours is this design (`dq4` in registers → `wgmma.rs`, A = weight, B = activation smem ring) | 0%. The offline-shuffle idea would need our own weight copy; we already read 3×`LDS.128` per 16 output words, i.e. at or past their "single 64-bit load". Nothing left. |
| [`examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm.cu`](https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm.cu) | **int4 × fp8** GEMM: dequant applies the scale into **e4m3** (not bf16) with an 8-entry pre-scaled LUT (`Array<ElementScale,8>`, `UseScaleLookupTable` at line 275); encoding pre-unified offline | F1 (design variant) | **no** | **candidate design, not a knob**: fp8 activations halve the B-side smem ring (96→48 KB) and halve wgmma issue per weight byte (K32 vs K16). The LUT itself is dead for us (per-16 scales ⇒ 16 B of LUT per 1 B of scale = +60 KB/unit). Priced only as "if the stall profile says wgmma-issue- or LDS-limited, this is the lever": **[INFERENCE] 5-15%**, needs its own check. Shipping precedent: [#7772](https://github.com/sgl-project/sglang/pull/7772). |
| [PR #7772 (merged, ByteDance co-author)](https://github.com/sgl-project/sglang/pull/7772) `sgl-kernel/csrc/cutlass_extensions/.../sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp` | W4A8 **grouped** MoE on SM90 for DeepSeek-R1-W4AFP8, `GROUP_SIZE 128`, plus `chunk_size`/`reload_factor` so the scale stage is only re-fetched every *n* K-tiles | F1, F4 | no (A8; we are A16) | **reference implementation for the W4A8 variant.** We cannot use `reload_factor`: our blocks are 16-wide, so a K16 tile consumes exactly one scale per row — nothing to amortize. Also confirms this is a *production* Hopper design, not a paper. |
| [cutlass `examples/93_blackwell_low_latency_gqa/readme.md`](https://github.com/NVIDIA/cutlass/blob/main/examples/93_blackwell_low_latency_gqa/readme.md) + `tgv_gqa_paged.cuh` | low-batch GQA: **one cluster `1×1×MAX_SPLITS` per (batch, kv head)**, flash-decoding splits, **cluster reduction of fmax/fsum/O via DSMEM** (`store_shared_remote_f32`, `get_dsmem_tensor`, `NUM_REDUCTION_CTA`), 7 warps = 1 DMA_Q + 1 DMA_KV + 1 MMA + **4 EPILOG**; paged mode with a Fisher-Yates page table, sliding window, dH 64/128, bf16/fp8 KV, CUDA-graph safe, "persistent schedule: unsupported" | F2 | **no** for DSMEM; yes for splits+paged | **partial.** Its *distribution* is the same as ours (kvL/MAX_SPLITS ⇒ ~2-4 tiles/CTA at window 2048), so it does **not** fix our fill-bound cells — that confirms the persistent-one-wave plan is the right one (`attn_gap.md` §3, +12-14%). What we do lack is **DSMEM replacement of the global finalize**: bounded by our own glue measurement `1.2-2.0 µs` per deleted launch (`prior_art_trxphxle.md` §1.1) ⇒ on 7 `!FLAT` cells ⇒ **+2-4% attention board**, bundling with the STAGES=4 candidate. |
| [flashinfer `include/flashinfer/attention/hopper/tile_scheduler.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/hopper/tile_scheduler.cuh) | `BatchPrefillPersistentTileScheduler` (one-wave persistent grid, per-tile work items) vs plain `BatchPrefillTileScheduler` | F2 | **no for windowed** — our `_plan` never emits a flat plan for `window != 0` (`attn_gap.md` §1, §5) | **yes, and it is the same lever as `wst_ct14b`**: confirms persistent-one-wave is the library-standard answer at Hopper, not an exotic. +12-14% attention board (already specced, `attn_gap.md`). |
| [flashinfer `attention/cascade.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/cascade.cuh) + [FA3 `hopper/copy_sm90_bulk_reduce.hpp`](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/copy_sm90_bulk_reduce.hpp) + `flash_fwd_combine_kernel.h` | merging two attention states (`merge_state` of index sets A∪B), and `cp.reduce.async.bulk` for the combine without register round-trip | F2 | **partly** — our MoE already uses `red.release.gpu.global.add`; attention's finalize reads partials | 0…1 µs. The merge is not our critical path (fill is); `cp.reduce.async.bulk` would at best replace the finalize's read-modify-write. Fold into the DSMEM row, do not price separately. |
| [DeepGEMM `csrc/jit_kernels/impls/`](https://github.com/deepseek-ai/DeepGEMM/tree/main/csrc/jit_kernels/impls) (`sm90_*`, `sm100_*_mega_moe`, `sm100_fp8_fp4_gemm_1d1d`) | SM90 tree is BF16/FP8/MQA-logits **only** — the SM90 W4A16 GEMM that chord vendors is *not* in the current tree; `mega_moe` (single fused EP dispatch+GEMM+combine kernel) is **sm100-only** | F1, F4 | — | **0%.** The mega-MoE is closed for us anyway (`RESEARCH_ROSTER.md`), and it is sm100. SM90 MQA-logits (`sm90_fp8_mqa_logits.hpp`) is a QK-only MQA scorer — not F2 (we need PV too). |
| [TensorRT-LLM `kernels/moe/cutlass/moe_kernels.cu`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/moe/cutlass/moe_kernels.cu) `min_latency_mode` | min-latency MoE: `num_moe_inputs = num_experts_per_node * num_rows` — every token against **every** local expert, router weight zeroed where unselected; kills the sort/gather | F1 | no | **0%, actively worse.** At d1 the judge touches 1.93 experts; computing 32 multiplies weight bytes ~16×. Our gather-based path is already the sparse-correct one. Useful only as a confirmation that "no gather" needs all-experts-active, which happens at prefill, and prefill is math-bound. |
| [TRT-LLM `kernels/weightOnlyBatchedGemv/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv) `kernelDispatcherBf16Int4GroupwiseColumnMajorInterleavedForHopperTrue.cu`, `cudaCoreGemmNVFP4.cu` | small-batch W4A16 by **CUDA cores** (FFMA, no tensor cores) with an **offline interleaved weight layout specialised for Hopper**, group-wise scales, fp16/bf16 × int4/int8 dispatchers | F1, F3 | no | **0% for F1**: our d1 is decode+stream bound, and FFMA at 402 MFLOP/call is ~12 µs of 24.9 µs duty (`[INFERENCE]` from 132 SM × 128 FMA × 1.98 GHz) — strictly worse than free wgmma math. **0% for F3**: F3 has no quantization at all. Record it as the reference "no tensor cores at bs 1" datapoint. |
| [`triton/python/tutorials/10-block-scaled-matmul.py`](https://github.com/triton-lang/triton/blob/main/python/tutorials/10-block-scaled-matmul.py) | NVFP4/MXFP block-scaled matmul via `tl.dot_scaled`, plus the scale-factor **packed layout 1×/128-row-swizzle** documentation | F1, F4 | no such instruction on sm_90 | **0% (dead by ISA).** `tcgen05.mma` with block scales is sm_100a/PTX 8.7. Valuable only as format ground truth: the scale layout description matches our per-16 block arrangement. |
| [`triton/python/tutorials/09-persistent-matmul.py`](https://github.com/triton-lang/triton/blob/main/python/tutorials/09-persistent-matmul.py), [`11-programmatic-dependent-launch.py`](https://github.com/triton-lang/triton/blob/main/python/tutorials/11-programmatic-dependent-launch.py) | persistent grid with TMA descriptors; PDL from Triton | F4, all | **yes** (persistent grouped kernel; PDL chain — `prior_art_trxphxle.md` §1.1 "<1.1 µs idle at T≤8") | 0%. No mechanism we lack. |
| [chord `docs/performance.md` "H200 grouped prefill (`contiguous`)"](https://github.com/novitalabs/chord/blob/main/docs/performance.md) + [design.md "Grouped kernel (SM90)"](https://github.com/novitalabs/chord/blob/main/docs/design.md) | best public H200 W4A16 grouped prefill: tiles `~96…176 × 256 × 64`, **6 stages** ("deepest ring that keeps the CTAs resident, worth **1-5%**"), stream-K **only when the last wave is <90% full**, BLOCK_M ≈ 1.3× expected rows/expert, TMA weight loads from a producer warpgroup, consumer word = its RS-WGMMA A fragment, lane-permuted scales give rows r and r+8 as one BF16 pair | F4 | mostly — our prefill MoE is a persistent grouped kernel with n=256 token tiles, and 128 measured slower (`prefill_budget.md` §2) | **their numbers are ours**: 603-643 TFLOPS at 128-512 rows/E (of 989 BF16 peak) vs our MoE at 63% of the judge's math floor at 16K. **No design delta exists to steal.** Only two cheap knobs: (a) **stage depth to 6** — "worth 1-5%" *of the MoE* — MoE is 38-40% of the prefill load ⇒ **+0.4-2% board**; (b) stream-K on the K tail — they gate it on "last wave <90% full", and our 16K grids are full ⇒ **0%**. Note they also pay the 3-6× weight re-read we pay (`prefill_budget.md` §2.1) and still hit 643 TFLOPS ⇒ our re-read is *not* the 63% gap. |
| [cutlass `examples/92_blackwell_moe_gemm/`](https://github.com/NVIDIA/cutlass/tree/main/examples/92_blackwell_moe_gemm), [`68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling`](https://github.com/NVIDIA/cutlass/tree/main/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling) | grouped MoE GEMM (incl. `fp4_regular`/`fp4_grouped`/`rcgrouped`) and the Hopper grouped **blockwise-scaled FP8** GEMM | F4 | n/a | **0%.** The FP4 grouped examples are all sm100 blockscaled. The Hopper one is FP8×FP8 with 1×128 scales and attaches the scale stage with its own TMA — our NVFP4 scales are 4× denser along K and are already handled in-register (`SCD`). |
| [cutlass `examples/63_hopper_gemm_with_weight_prefetch/README.md`](https://github.com/NVIDIA/cutlass/blob/main/examples/63_hopper_gemm_with_weight_prefetch/README.md) | "TMA loads use non-default cache hints: **A (weights) `EvictFirst`, B (activation) `EvictLast`**"; L2 prefetch of A during the previous kernel's tail via `griddepcontrol`; split-DMA warps for A and B | F1 | **yes, exactly** — `muse_moe_w4a16.py:335-344`: "L2 policies (CUTLASS CacheHintSm90): weights stream through once (evict first); inter and ybuf are re-read later (evict last)" with `.L2::cache_hint` + `0x14F0…` descriptor | **0%.** Already shipped, and independently justified by the same doc. The L2-prefetch half is bounded by our PDL gaps (<1.1 µs at T≤8, `prior_art_trxphxle.md`). |
| [`examples/57_hopper_grouped_gemm`](https://github.com/NVIDIA/cutlass/tree/main/examples/57_hopper_grouped_gemm), [`69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_bf16_grouped_gemm.cu`](https://github.com/NVIDIA/cutlass/blob/main/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_bf16_grouped_gemm.cu) | grouped W4A16: `TileShape <128,16,K>` (n=16!), scale-only and scale+zero layouts, TMA vs shuffled layouts | F1, F4 | partly (we use 16-row units) | **0%.** CUTLASS's own n=16 comes from the mixed-dtype collective's constraints, not from a decode insight; and `TileShapeK` comment "use a small tile size, register pressure for RS GEMM is high" is the tradeoff we already live in (576×96 regs). |
| [FlashMoE (NeurIPS'25)](https://github.com/osayamenja/FlashMoE) | "fully fused distributed MoE … single kernel, eliminates kernel boundaries, overlaps communication and compute; CUDA-graph compatible" | F1 | closed axis (megakernel) | **0%.** Same class as our closed megakernel axis; and its win is EP communication, which we do not have (single GPU). |
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) (levels 1-4, EVAL.md) | problem suite + `compile`/`correctness`/`performance` triage | — | — | **0% mechanism.** Methodology only; cross-ref to `research_05`/`research_09`. Nothing here targets H200 shapes. |
| [gpu-mode/discord-cluster-manager](https://github.com/gpu-mode/discord-cluster-manager) (`kernelbot/`, `libkernelbot/`, `runners/`) | the KernelBot judge: task runner, per-GPU runners, leaderboard DB | — (harness) | our judge replica already models this (`judge_replica.md`, `research_10`) | **0% new.** Read-only cross-ref; do not duplicate `research_10`'s work. |

## Table B — analyzers / tooling

| item (path + URL) | mechanism | family it serves | already ours? | verdict, priced |
|---|---|---|---|---|
| [mit-han-lab/ncu-report-skill `SKILL.md`](https://github.com/mit-han-lab/ncu-report-skill/blob/main/SKILL.md) + `helpers/analyze_reports.py`, `helpers/extract_stall_hotspots.py`, `helpers/plot_timeline.py` + `reference/05-analysis-dimensions.md`, `reference/06-diagnosis-playbook.md` | one `ncu --set full --import-source yes` collection → (a) key metrics + NCU's own **rule results with `Est. Speedup: X%`**; (b) **per-source-line stall attribution incl. inlined CUTLASS/CuTe via `--binary`**; (c) PM-sampling timeline of `sm__throughput` vs `dram__throughput`; then a pattern→cause→fix playbook (E "latency-bound: `smsp__pcsamp_warps_issue_stalled_long_scoreboard/sample_count > 0.40`", H "`short_scoreboard` on shared-memory load lines" = bank conflicts, I "`barrier` > 20% of samples", M "sawtooth sm/dram timeline = pipeline bubbles", O "few warps, fixed-latency stalls = serial dependency chain") | **F1 above all**, F2/F4 | **no** — we have CUPTI per-kernel times and `DBG(0..7)` phase stamps (`moe_struct.md` §0), i.e. *when*, not *why* | **the highest-value transferable item after the chord schedule, because it decides which of the two remaining hypotheses is true.** Our MoE deficit is 45% of peak spent **per 36 KB stage** (`moe_struct.md`), and the two candidate explanations are (i) LDS/`short_scoreboard` + bank conflicts in the ring reads and (ii) `wgmma` dependency (`wait`) / issue density. Per-line stalls distinguish them in one collection; the eight `DBG` timestamps cannot. Cost ≈ 1 profile run + 3 scripts; no kernel edit. Expected: **0% by itself, but it gates a +7-14% (F1) decision and a 1-line `wait<1>` (+1.75%)**. Playbook also supplies the magnitude conventions for ranking. |
| [Enigmatisms/tachyon](https://github.com/Enigmatisms/tachyon) `src/tachyon/analyzers/{roofline,memory,occupancy,warp_stall,instruction,launch}.py` | rule analyzers (roofline / memory / occupancy / warp-stall / instruction / launch — "7 built-in analyzers" in the README), NCU↔source↔SASS three-way mapping, and **`tachyon diff before.ncu-rep after.ncu-rep`** | F1, F3 (per-cell A/B) | no | **yes for the A/B half.** Our binding constraint is that nothing under ~2% resolves without a paired local A/B (`prior_art_trxphxle.md` §5: ±1.5-2% judge noise, A/A ±0.3%). `tachyon diff` on two `.ncu-rep` files of the same kernel is a *counter-level* paired diff — a stronger control than a wall-clock replay pair. Expected: no % directly; reduces the cost of every marginal check. |
| [LancerLab/kebab `kebab/lib/microbench/`](https://github.com/LancerLab/kebab/tree/main/kebab/lib/microbench): `mma_wgmma.cu`, `mma_mma_sync.cu`, `copy_gmem_to_smem_2d_tma_{cute,ptx,cde}.cu`, `hgemm_wgmma.cu`; docs `WGMMA_ATOM_SELECTION.md`, `TMA_IMPLEMENTATION_SUMMARY.md` | ready-made probes for (a) raw `wgmma` vs `mma.sync` issue/latency, (b) 2-D TMA gmem→smem copy cost at a chosen box size, (c) WGMMA atom selection notes | F1 (the "what does one 36 KB stage cost" question) | no | **yes — cheap and decisive.** It answers, off-line and without touching our kernel: how long a 36 KB TMA box takes to land, and what `m16n8k16` costs vs `wgmma m64n16k16` per weight byte. `kebab` is H800-PCIe (slower HBM) so use it for *shape* of the answer, not absolute µs. Expected: replaces guessing with a number for the chord swap-AB candidate's feasibility. |
| [nsight-python extraction docs](https://docs.nvidia.com/nsight-python/extraction.html), [KuangjuX/ncu-cli](https://github.com/KuangjuX/ncu-cli), [OptimAI-Lab/CudaForge `run_ncu.py`](https://github.com/OptimAI-Lab/CudaForge/blob/main/run_ncu.py) | programmatic report extraction; CLI wrappers; an ncu-in-a-loop driver | F1 harness | no | **not inspected**; superseded by the two rows above. Listed to close the class. |
| [microsoft/MMA-Sim `src/mmasim_kernels/mmasim_kernels/nv_ptx/hopper.py`](https://github.com/microsoft/MMA-Sim/blob/main/src/mmasim_kernels/mmasim_kernels/nv_ptx/hopper.py) | simulator of Hopper MMA instruction behaviour | F1 | no | **not inspected**; could price wgmma issue patterns analytically, but `kebab` measures the real thing. Keep as fallback only. |
| [`ncu --set full` sections: SourceCounters + PmSampling; `--clock-control boost` vs `base` (1.7× swing on newer parts)` (from SKILL.md) | collection discipline: clock control, per-PC sampling, embedded source | all | no — our replays are graph-based and clock-uncontrolled | **risk note, not a mechanism.** Any ncu conclusion must be re-anchored to a graph-replay A/B, because ncu locks clocks and serialises replays. Do not compare an ncu µs to a judge µs. |

---

## Family-by-family verdicts

**F1 — NVFP4 W4A16 MoE decode.** *Not* "nothing transferable: this is where the sweep pays.* CUTLASS's Hopper
mixed-input `rs` collective is the exact arrangement we already run (narrow type in registers, weight=A, activation
smem=B), and the L2 evict-policy hint is already shipped (`muse_moe_w4a16.py:335`), so the *dequant* and *cache* axes are
closed. What is open is **schedule shape**:
`novitalabs/chord` ships a production H200 W4A16 MoE decode that runs the weight on the `mma`-M axis with block-M **8** (legal
only because of the swap) at **3-4 CTAs/SM**, and their like-for-like measurement puts that **4-29% ahead of their own
wgmma/grouped schedule** at exactly our rows-per-expert band (`docs/design.md`). Everything else in the sweep for F1 is
either sm100 (NVFP4 MMA, DeepGEMM mega-MoE, CUTLASS 92), already ours (cache hints, `rs` dequant), worse (TRT-LLM
min-latency all-experts expansion, CUDA-core GEMV), or unpriceable at our scale (the W4A8 variant). The 16% is *not*
explained by stream rate. Comparing at a *constant* transfer size, chord at 8 tokens/expert reads 792 MB and runs 3.28 TB/s
≈75% of what 792 MB should give, while our v1 (179 MB, 10.1 routes/expert) runs 2.64 TB/s ≈63% of what 179 MB should give
(`[INFERENCE]` off the size→BW curve in `prior_art_trxphxle.md` §2) — i.e. **~15-20% of their advantage survives the
transfer-size correction**, and the rest of their headline GB/s is the larger read. That ~15-20% residual is the size of
the top team's lead, and it is explained by **which instruction consumes the weight bytes and how many CTAs are streaming
at once**, not by the stream itself. Routes-per-touched-expert at the scored loads (`moe_struct.md` §0): d1 4.1, v1 10.1,
d16 14.4, d32 21.3, v16 59.5, v32 110.7 — so the sub-18.75 swap-AB band is exactly the three skinny/WS loads d1, v1, d16.

**F2 — paged FP8 decode attention.** Nothing in the sweep beats the plan already identified: the only published
Hopper-class low-batch GQA kernel (CUTLASS `93_blackwell_low_latency_gqa`) distributes keys exactly as we do
(kvL/MAX_SPLITS ⇒ 2-4 tiles/CTA) and therefore shares our fill-boundedness; flashinfer confirms persistent one-wave is the
standard answer. The one genuinely new, modest item is **cluster/DSMEM in-kernel reduction** replacing the global
finalize: ≤1.2-2.0 µs per affected cell, +2-4% board, bundle-only. Everything else (page-table striping, deeper pipes) was
already priced in `attn_gap.md`.

**F3 — BF16 GEMV/GEMM, lossless codec.** **Nothing transferable, and the sweep is now a *closing* measurement.** The only
published lossless-BF16 codec GEMM on Hopper (`cloudflareresearch/unweight-kernels`, [report](https://research.cloudflare.com/papers/unweight-2026.pdf))
is, by design, our 12-bit codec (8-bit sign+mantissa verbatim + 4-bit palette index = 12 bits; Huffman ~32%/68% at
storage time, palette ~25%/75% for the GEMM) and it **loses to plain cuBLAS at every batch size on H100**: 86.1 → 109.1 µs
at bs 1, 87.7 → 108.2 at 64, 93.8 → 149.6 at 256, 213.1 → 458.6 at 1024 (Table 11), end-to-end −41% at bs 1 to −30% at
bs 1024 (Table 9). Their own diagnosis is the same as ours: reconstruction cost is additive on a machine with no spare
bandwidth or cycles, and the persistent grid reconstructs the same weight tile redundantly. Their conclusion for bs<32 is
literally ours: "a simpler ALU-based decode+matmul kernel — without warpgroup specialization, TMA, or WGMMA — would likely
perform better". **Our 12-bit decision stands; do not re-open it.** Two reusable sub-facts: verbatim-row level routing
(u64 bitmask + `popcount`) beats per-symbol escapes; and their explored negative A.2 (two-level fast LUT) regressed 10%,
which matches our intuition that extra decode-path branching on a bandwidth-bound kernel never pays.

**F4 — grouped NVFP4 prefill, math-bound.** **Nothing transferable for the MoE.** The best public H200 W4A16 grouped
prefill sits at 603-643 TFLOPS of 989 (chord, 128-512 rows/E) — i.e. the same 63% of roofline we measure at T=16384 — and it
pays the same multiple weight re-read we pay. Our n=256 token tile is at their BLOCK_M ceiling (`block-M <= 176` for
prefill in their table; 96-176 in practice). Only two knobs survive: **stage depth 6** (their measured 1-5% *of the MoE* ⇒
+0.4-2% board) and stream-K gated on an under-full last wave (0% for us at 16K). CUTLASS's grouped FP4 examples are sm100
blockscaled and dead by ISA.

---

## Ranked top-3 (transferable mechanisms, by expected board gain)

1. **F1 — chord's swap-AB `mma.sync` decode schedule (block-M 8, 3-4 CTAs/SM) for the three sub-18.75-rows/expert
   loads.** Source: [`chord/docs/design.md`](https://github.com/novitalabs/chord/blob/main/docs/design.md) "Indexed kernel"
   + tuning table. Effect: **+10-25% on d1/v1/d16 ⇒ +5-12% MoE board** (their range 4-29% ⇒ +2-14%), ≈ 500-1200 ranks at
   101 ranks/%. This is the only candidate in the sweep that is a *different design* rather than a knob, and it is the
   only one whose magnitude was measured on our GPU at our rows-per-expert. Gate it behind the ncu per-line stall profile
   so the wgmma-vs-`mma.sync` choice is made on a stall class, not a guess.
2. **F1 — the cheap pair, both ~1 line: `wgmma.wait<1>` (commit-group width) + 2 CTAs/SM for the skinny bands.**
   Sources: [design.md "WGMMA wait<1"](https://github.com/novitalabs/chord/blob/main/docs/design.md) (3-6% gate/up, 1-5%
   down) and [design.md "Residency"](https://github.com/novitalabs/chord/blob/main/docs/design.md) ("hides the single TMA
   producer's latency"); our legality path for 2 CTAs/SM already computed in `moe_struct.md` §2\#3 (drop dead `xb`, `OKC_NS=2`
   ⇒ 98 KB). Effect: **+1.75% (wait<1>, 3 of 6 loads) + 0…+4% (residency) ⇒ +2-6% MoE board**, two independent checks, both
   reversible, neither needs a new kernel.
3. **F2 — cluster + DSMEM in-kernel reduction of split partials on the `!FLAT` cells.** Sources:
   [`93_blackwell_low_latency_gqa`](https://github.com/NVIDIA/cutlass/blob/main/examples/93_blackwell_low_latency_gqa/readme.md)
   (`store_shared_remote_f32`, `NUM_REDUCTION_CTA`, 4 EPILOG warps) and [`cascade.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/cascade.cuh).
   Effect: bounded at **1.2-2.0 µs/cell** by our own glue measurement ⇒ **+2-4% attention board**, stacks with the
   STAGES=4 candidate (+12-14%) and with the persistent-one-wave windowed plan (the real lever, unchanged).

Runner-up, not ranked because its magnitude is unpublished: **F4 stage depth → 6** (+0.4-2% prefill board).

The enabling measurement (not a mechanism, but it is the operator's second ask and it gates #1): **`ncu --set full
--import-source yes --clock-control boost` + `mit-han-lab/ncu-report-skill` `extract_stall_hotspots.py --binary` on
`moe_w4a16`, plus `tachyon diff` for anything under 2%.** One collection attributes the 45%-of-peak per-stage deficit to a
named stall class; nothing cheaper exists, and our `DBG(0..7)` timestamps cannot distinguish LDS from wgmma-dependency.

## Risks / non-obvious negatives the next agent needs

* **chord's GB/s table needs a transfer-size correction before it is compared to us.** Every chord decode row reads a
  constant ~0.79 GB (48 experts × 16.5 MB gate_up, all touched at any batch), i.e. 3-14× more than our loads (55-523 MB).
  Normalising with the size→bandwidth curve we already measured (`prior_art_trxphxle.md` §2: ~4.6 TB/s at 2 GiB, ~4.2 at
  128 MiB, ~3.0 at 64 MB): our v1 (179 MB, 10.1 routes/expert) runs 2.64 TB/s ≈ **63%** of what 179 MB should give, while
  chord at 8 tokens/expert (792 MB) runs 3.28 TB/s ≈ **[INFERENCE] 75%** of what 792 MB should give. So roughly
  **15-20% of real work-distribution difference survives the correction** — the same order as the top team's 16% lead.
  Do not quote "chord is 50% faster": the defendable claim is their within-design **4-29% at decode** plus this ~15-20%.
* **Their kernel is block-scaled group-32 int4 while ours is NVFP4 per-16.** The swap-AB *schedule* transfers; their
  scale layout does not. Their `reload_factor` has no analogue for us (§Table A).
* **ncu locks clocks and serialises replays.** Any ncu number must be re-anchored to a graph-replay A/B before it is allowed
  to change a decision (`SKILL.md` §"Where the GPU is": `base` vs `boost` is a 1.7× swing on newer parts).
* **The two `n=16`-tile items are not decode insights.** CUTLASS 69's `TileShape <128,16,TileShapeK>` is forced by the
  mixed-dtype collective's scale/alignment rules, and the "register pressure … is high (it uses a lot of register space)"
  note is from 55's own "Optimizing suggestions" (`55_hopper_int4_fp8_gemm.cu:85-88`) — that is the constraint we already
  live in (576 threads × 96 regs, ~226 KiB smem), not a new placement.
* **Do not re-open the lossless-codec axis (F3).** A published Hopper implementation of exactly our 12-bit scheme loses to
  cuBLAS at every batch size; our 6-11% measurement on 10.9 bits is the same physics, from the other side.
