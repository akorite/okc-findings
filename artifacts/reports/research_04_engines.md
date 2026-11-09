# research_04_engines — serving-engine kernel source, read directly

Clones (sparse, blobless, read at these commits). Every `path:line` below resolves to
`https://github.com/<owner>/<repo>/blob/<sha>/<path>#L<line>` at these SHAs (owners: `sgl-project`,
`vllm-project`, `NVIDIA`, `flashinfer-ai`, `deepseek-ai`, `NVIDIA`):

| repo | ref | commit |
|---|---|---|
| sglang | `codex/muse-spark-open-sglang-miles-20261102` (closest reachable Muse-Spark lineage; the `dev-muse-spark-open` name does not exist upstream) | `d5f16a38d5deee34e2b5113396c24617e737abe6` |
| vllm | main | `ffdf26702aec559f9e2fbf93ed4c74bd783836e7` |
| TensorRT-LLM | main | `a649e41d2cc28944e3b76bc553dbd6fd52dbe195` |
| flashinfer | main | `08d74a47163138273af02f034a500449b9a2a6d0` |
| DeepGEMM | main | `5081f8513af57e61696de4f4f04b0e707ffb503e` |
| CUTLASS | main | `e13f5cd4142c43efc37c2f6339923eb4a57afb99` |

Architecture caveat that governs every "actionable here" verdict: the FlashInfer `cake_*` and CUTLASS `93_*` families are **sm100a/sm103a** (tcgen05/TMEM/CLC). We run H200 (sm90). Their *scheduling* is portable (thread-block clusters, DSMEM, TMA, atomics, persistent grids all exist on sm90); their *math* path is not. Every trick below is graded on the scheduling axis.

Two facts from the checkpoints, worth stating because they are load-bearing later:

* Muse Spark Open is 62 layers, hidden 8192, 256 experts top-8, expert width 4096, **128 q / 16 kv heads, head_dim 64** (per TP8 rank: 16 q / 2 kv), sliding 2048 on 46 layers, 16 full layers, and the **full layers are NoPE** (`layer_rope_theta[i] == 0` → `use_rope=False`). Sources: `vllm/recipes/models/meta-models/Muse-Spark-Open-816B-A42B.yaml` (guide text, "drops rotary position (`layer_rope_theta: 0`)"); `sglang/python/sglang/srt/models/muse_spark.py:348` (`self.use_rope = config.layer_rope_theta[layer_idx] > 0`). Our roster says "RoPE theta 5e5" — if our plan applies RoPE on all 62 layers, 16 of them are doing work the reference model does not. Verify against the harness checkpoint before acting; flagged as a correctness-shaped finding, not a perf trick.
* Baseline MoE on H200 (the kernel the harness times) is Marlin W4A16 + a *separate* finalize that does post-expert RMSNorm and the top-k sum: "Every expert backend stops after GEMM2 and hands its rows to the finalizer." — `vllm/vllm/models/muse_spark/nvidia/muse_moe_runner.py:5`.

---

## 1. SGLang — the Muse Spark lineage

### 1.1 What wraps `flash_attn_with_kvcache`, concretely

The wrapper itself is thin (`sglang/python/sglang/kernels/ops/attention/flash_attention_v3.py:100-175` — it forwards `scheduler_metadata`, `num_splits=0`, `pack_gqa`, `sm_margin`, `out=` straight to the cubin). The kernels around it are the cost:

1. **Fused Q/K norm + RoPE + FP8 KV store, one kernel** — `sglang/python/sglang/kernels/ops/attention/muse_spark_qk_norm_rope_store.py:1-15`. Docstring: fuses per-head RMSNorm of the strided Q and K slices of a QKV projection, RoPE, and the K/V store into the paged pool; `USE_PDL` does `gdc_wait()` / `gdc_launch_dependents()` so the *next* kernel can start before this one retires. This replaces what would otherwise be 3–4 launches (norm×2, rope, `set_kv_buffer`).
2. **Q cast to the KV dtype + descale expansion**, host-side only — `flashattention_backend.py:1408-1436` (`prepare_paged_mha_query`). No kernel.
3. **KV store fallback** — `flashattention_backend.py:2103` (`set_kv_buffer`), used when the fused prologue's buffer handshake is not available (`get_fused_kv_write_buffers`, `:1441`).
4. **Attention metadata, precomputed once per forward and reused by all 62 layers** — `flashattention_backend.py:409-437` (`_compute_scheduler_metadata`), consumed at `:2273-2312`. FA3's split schedule normally costs one `prepare_varlen_num_blocks` *per layer*; SGLang removes 62× of it. It is explicitly disabled for SWA layers (`:2284` comment: "scheduler_metadata is only valid for non-SWA, non-cascade decode") — exactly our 46:16 split.
5. **Split-KV combine** — FA3 launches a *separate* combine kernel (`sglang/python/sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_combine.py:1-30`, `tile_m=8`, `k_block_size=64`, `log_max_splits=4`, `stages=4`, `use_pdl`). This is the extra launch that XQA and TGV both delete (§3, §5.1).
6. **Zero-KV fixup** — `sglang/python/sglang/kernels/jit/csrc/attention/fixup_zero_kv.cuh:1-8`: a post-pass that forces `out=0, lse=-inf` for sequences with `kv_len==0` (ragged/TRT-LLM zero-length rows). If our kernel handles `kv_len==0` natively this kernel does not exist for us.

**Verdict:** the reference stack still pays (a) one fused prologue per layer, (b) one combine launch per layer when split, (c) a zero-length fixup for contracts it does not own. Fusing (1) is already the right shape; the actionable item is (b)+(c): *do not add* a combine or fixup launch — fold both into the attention kernel (see §3.3). Estimate: 1 launch ≈ 2–3 µs GPU-side, ~0.5–1 µs when PDL-overlapped [INFERENCE]; ×62 layers ×2 (combine+fixup) ≈ 60–300 µs per decode step if they are on the critical path. Our cells are 1.07–1.67 TB/s on 2–8 tiles/CTA, i.e. launch-bound territory, so this is not noise.

### 1.2 `moe_wna16_marlin_gemm` — what it is, what can be restructured

Launch: `sglang/python/sglang/kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh:780-781`

```
int blocks = sms * exec_cfg.blocks_per_sm;
kernel<<<blocks, num_threads, max_shared_mem, stream>>>(...)
```

i.e. **a persistent grid sized to SMs × occupancy**, not to the number of MoE row blocks. Occupancy comes from a register/shared-memory budget (`:587-593`: `allow_count = min(reg_limit, smem_limit)`, capped at 4 when `thread_m_blocks==1`, further capped by an estimate `prob_n/thread_n * prob_m * top_k * 4`). Work is then striped statically inside the kernel: `marlin_template.h:360-373` explains the stripe map with a 3×3 / 5-SM example, and every CTA walks `slice_row/slice_col_par` from `blockIdx.x` (`:490-500`). Cross-CTA split-K partials meet through a global lock whose initial value is `1 - slice_count` (`:676-692`) and float atomic adds (`:1906-1925`).

Two source-level facts that a submission can use:

* **SGLang added a hand-tuned H200 special case** (`moe_wna16_marlin.cuh:509-518`), for exactly our regime — `prob_m==1`, FP4 weights, group 32:
  ```
  return exec_config_t{2, thread_config_t{128, 64, 128}};   // 2 CTAs/SM, thread_k=128, thread_n=64
  ```
  with the comment "six sparse expert rows need a narrower N tile and fewer persistent blocks than the occupancy-only heuristic." That is a citation that the general occupancy heuristic over-provisions CTAs at M=1 and the fix is *fewer, narrower* CTAs — the opposite of the attention fix.
* **M=1 alignment is a single warp, not two kernels** — `sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py:302-318`: when `M==1 and topk<=32` it calls `moe_align_single_token` ("top-k ids are distinct, so alignment is a single-warp sort instead of the align + count_and_sort kernel pair"). Same in vLLM fused: `vllm/csrc/moe/fused_globalize_align_block_size.cu:1-20` fuses *globalize + align + count_and_sort* into one launch (zero+sentinel → shared histogram → block-0 prefix → scatter), with a 1-block non-cooperative form for small decode and a cooperative grid otherwise.

**Verdict (MoE, non-duplicating R2):** at M=1/top-8 the whole Marlin chain is two persistent-kernel GEMMs plus an align kernel plus an activation plus a finalize that reads `top_k × hidden` fp32 rows per token. The align launch and the finalize's per-route RMSNorm are the parts a fused-epilogue submission deletes; the persistent-grid/stripe part is already what we do better. `moe_align_single_token`-class fusion is worth ~1–2 µs/layer [INFERENCE]; the finalize is the bigger one (it reads/writes 8×8192 fp32 per token per layer in the baseline). No new proposal here because deep MoE rings / fused-slice MoE are closed; recorded as evidence that the baseline pays these.

### 1.3 Two small-batch wins that are *not* MoE rings

* **Router GEMV instead of cuBLAS split-K** — `sglang/python/sglang/kernels/ops/gemm/muse_spark_router_gemv_cutedsl.py:1-12`:
  > "cuBLAS runs this as a split-K GEMV plus a separate reduction kernel on the router side stream, which then finishes after the main-stream expert input and stalls the MoE. Here one CTA owns one (T=1) or two weight rows over the whole K range: each thread issues all of its 16-byte weight loads before the first FMA … There is no split-K workspace or second launch, and the summation order is fixed."
  `:21-27`: 512 threads, `_experts_per_cta = 1 if tokens == 1 else 2` ("one expert per CTA at T=1 (more CTAs in flight)"). 256 experts ⇒ 256 CTAs ≈ 2 waves on H200.
  **Actionable:** if our decode path computes router logits with any library GEMM/split-K at M=1, replace it with one-CTA-per-expert GEMV on the *same* stream. Win = the removed reduction launch plus the side-stream stall: 3–8 µs/layer at M=1 [INFERENCE], i.e. potentially 0.2–0.5 ms/step across 62 layers. This is a new axis (router, not experts) and is closer to R2's remit; flagged for it rather than claimed here.
* **Gate + post-attention norm fused** — `sglang/python/sglang/srt/models/muse_spark.py:433-440`: "The output norm and the sigmoid gate are one fused kernel", and the QKV projection carries the o-gate (`QKVOGateParallelLinear`). Reference attention prologue/epilogue is already 1+1 launches, so there is no free launch to win there.

---

## 2. vLLM — the graph-safety constraint that decides the whole design

### 2.1 FA3 split scheduling is not CUDA-graph-safe, so serving stacks pin it

`vllm/vllm/v1/attention/backends/flash_attn.py:855-865`:

```
max_num_splits = 0  # 0 means use FA3's heuristics, not CG compatible
...
# we only set num_splits when using cuda graphs.
max_num_splits = self.max_num_splits
```

and `:707` "No upper bound on the number of splits", with `max_num_splits` configured per-model (`flash_attn_max_num_splits_for_cuda_graph`). `:1409-1420` then forces `num_splits = attn_metadata.max_num_splits`, and `:1413-1420` comment: hd256 "requires page-aligned lengths, exact-width block tables, and no SplitKV".

**Why this matters at our scale:** FA3's split decision reads sequence lengths on the host. Under CUDA graphs it must be frozen. So *any* dynamic split/occupancy decision we make has to be computed **on device inside the launch** (or from device-resident metadata) — which is exactly what the sm100 cake kernels and XQA do, and why they can respond to a 8–17 MB KV step while FA3 cannot.

### 2.2 Mixed sliding/full layers disable AOT scheduling outright

`flash_attn.py:844-853`:

```
if self.aot_sliding_window is None:
    ...
    sliding_window_configs = _get_sliding_window_configs(self.vllm_config)
    if len(sliding_window_configs) == 1: ...
    elif len(sliding_window_configs) > 1:
        self.aot_schedule = False
```

Our model is the `> 1` case by construction (46 sliding : 16 full, plus NoPE/full). The reference stack therefore cannot pre-schedule across layers; SGLang only gets away with it by computing one non-SWA schedule and skipping SWA layers. **Actionable:** if our per-layer plan is recomputed per layer at 62 layers, hoist it (one plan for the 46 sliding cells, one for the 16 full cells) — SGLang's `:409-437` pattern. Estimate: metadata build is tens of ns when graph-captured but a few µs when not; ×62 = single-digit µs to ~0.15 ms [INFERENCE]. Low risk, low ceiling.

### 2.3 The rest of the vLLM MoE chain (baseline shape)

* Score/route in one kernel: `vllm/csrc/moe/muse_spark_routing.cu:1-20` (adapted from SGLang; one warp per token, 8 experts per lane via 128-bit loads, `redux.sync.max` on sm10x, `ex2.approx` + `div.full.f32` to match Triton numerics). `vllm/vllm/models/muse_spark/common/moe.py:21-30` documents the exact semantics: sigmoid → +bias for *selection only* → top-k → gather unbiased scores → divide by `(sum+eps)` → `route_scale`.
* Experts stop after GEMM2; finalize does the norm: `muse_moe_runner.py:5`, `:160-175` ("Pre-norm TP all-reduce → per-route RMSNorm → fp32 weighted reduce"). Decode threshold constants: `:149-157` fused all-reduce up to **256 tokens (TP4) / 128 (multi-node TP8)** — i.e. the reference explicitly treats ≤256 tokens as the fused, latency-bound regime we are in.
* Blackwell-only fused MoE the baseline *also* has: `vllm/vllm/models/muse_spark/nvidia/mega_moe.py:1-20` uses DeepGEMM `fp4_fp4_mega_moe` / `bf16_mega_moe` with `post_expert_norm_gamma` in one launch, selected above `_TRTLLM_TOKEN_THRESHOLD_TEP4 = 128` / `TEP8 = 256` (`:41-42`). On H200 this path does not exist; our Marlin chain is the H200 baseline.

**Verdict:** the baseline's own authors consider "one launch for the whole MoE including norm" the right answer for decode; on H200 they fall back to two GEMMs + finalize. Our roster has closed megakernel/fused-slice MoE, so this is context, not a proposal — but it explains why MoE rank 246 is hard: the reference is already 2 kernels when it can be.

---

## 3. TensorRT-LLM XQA — the closest thing to a small-batch decode champion

Source lives in FlashInfer's tree (`flashinfer/csrc/xqa/*`, the same kernel TRT-LLM JITs; `trtllm/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/` is only the JIT engine).

### 3.1 Grid and the CTAs-per-request formula

`flashinfer/csrc/xqa/mha.cu:1593-1601`:

```
bool const isMultiBlock = allowMultiBlockMode && (gridDim.x != 1);
uint32_t const nbSubSeqPerSeq = allowMultiBlockMode ? gridDim.x : 1;
...
// gridDim: x - K/V sequence-dim split; y - number of K or V heads per token; z - requests
```

So the grid is `(KV-split, kv_heads, batch)` — one CTA per (request, KV head, KV chunk). Split size, Hopper path (`mha_sm90.cu:2960-2971`):

```
float const factor = 0.25f;
return min(max(1, round(prop.multiProcessorCount * 3 / (batchSize * nbKHeads) * factor)),
           divUp(maxSeqLen, gemm0CtaTileNbTokens));
```

i.e. `0.75 × SM_count / (batch × kv_heads)`, capped by pages-per-sequence. The `3` is the assumed resident-CTAs-per-SM; `0.25` is a deliberate under-subscription so the merge does not dominate. The Blackwell path (`mha.cu:3016-3031`) is `min(max(1, SM_count/(batch*nbKHeads)), divUp(maxSeqLen, ctaTile.x))` — full subscription, no factor.

TRT-LLM's own gate (`trtllm/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunnerUtils.h:405-430`, `decoderXQAConstants.h:27-31`):

```
multiBlockCount = historyLength / kMinHistoryTokensPerBlock;      // 128 tokens per block
multiBlockCount = min(multiBlockCount, round(sqrt(multiBlockCount * 8)));  // merge cost cap
waveCount = batchSize * numKVHeads * multiBlockCount / numSMs;
if (waveCount / kTargetWaveFactor > 1) multiBlockCount /= adjFactor;     // kTargetWaveFactor = 8
```

and admission (`decoderXQARunner.cpp:154-176`): use XQA only if `numKVHeads*batch*multiBlockCount * kEnableMinBlockFactor >= numSMs` with `kEnableMinBlockFactor = 4.0` — i.e. **at least ~¼ of the SMs busy, target ~8 waves, per-sequence splits capped at √(8·blocks)** because "the final reduction may dominate". Workspace cap `getXqaMaxNumSubSeq = 6000` entries ≈ 12 MB.

**This is the actionable runway formula.** Our sliding cells at 2/4/4/8 tiles/CTA are exactly the "1 CTA per request, and the request is short" case XQA refuses to launch with. Two levers, both in XQA's source: (a) split KV until `batch × kv_heads × splits ≈ 8 × SMs`, (b) never let `splits` grow faster than √(8·blocks) — the merge, not the QK, becomes the bottleneck.

### 3.2 What XQA does that FA3 does not

1. **Merge in the last CTA, on a self-resetting semaphore, with no combine launch.** `mha.cu:2756-2762`:
   ```
   atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;   // +lastOld, lastOld = nbSubSeqPerSeq-1
   mbsmem.isLastCta = (old == lastOld);
   ```
   the winner reads every sub-sequence's row-max/row-sum and output from global scratch with `ldgsts` (`:2813-2900`, two-deep ring, `loadBufAsync`) and writes the final O. FA3 instead launches `flash_fwd_combine` as a second kernel and needs a workspace reset between calls.
2. **Fixed warp roster with a dedicated IO warp group.** `mha_sm90.cu:93-105`: `gemm0WarpGrpTileNbTokens = 64`, `nbIOWarps = 4`, math warps + IO warps, `dimCta = {gmmaWarpsPerGrp, 1, 3}` (three warp groups: 2 math + load). CTA tile = 128 KV tokens per math warp group, `nbKBuf = 2`, `nbVBuf = 2` (`:142-163`) — **double-buffered, i.e. ~32 KB in flight per CTA, the same budget we have.** XQA's runway therefore comes from CTAs-per-request, not from a deeper ring. That is the single most useful calibration in this report.
3. **Q stays in registers/smem for the whole KV walk** (`persistentQ = paddedInputHeadBytes * ctaTile.y <= 16 KB`, `mha.cu:129-131`) — no Q reload per KV tile.
4. **4-bit KV cache** behind `ENABLE_4BIT_KV_CACHE` (`mha.h:65-171`, `mha.cu` `grainBytesGmemCache = grainBytes / ElemsPerContainer`, smem-padded to 128 B grains). See §6.4.
5. **PDL**: `ENABLE_PDL` / `preExit(); acqBulk();` in the kernel (`mha.cu:1665-1666`) and `enable_pdl` in the host launch (`makeLaunchConfig(..., enable_pdl)`); TRT-LLM's MoE side does the same with `cudaLaunchAttributeProgrammaticStreamSerialization` (`trtllm/cpp/tensorrt_llm/kernels/moe/trtllmGen/DevKernel.h:40-56`).

**Verdict (runway, primary):** adopt XQA's *occupancy policy* rather than its math: compute `splits = clamp(round(0.75·SMs/(batch·kv_heads)), 1, ceil(kv_len/tile))` per (request, kv-head), keep the per-CTA ring where it is, and merge in the last CTA with an acq_rel counter. Expected: sliding cells 1.07–1.67 TB/s → 2.5–3.4 TB/s if the machine is actually filled (the flat cells prove the same ring can sustain 3.07–3.65 TB/s when CTAs are occupied), i.e. **up to −40% of the attention cell; 30–50% of that is the realistic capture ⇒ −12% to −20% of the attention cell** [INFERENCE]. Cost is one in-kernel merge and one counter; no new launch.

---

## 4. FlashInfer — the most explicit "short grid" machinery I found

### 4.1 Persistent + cluster launch control + CGA splits, with a device-side scheduler warp

`flashinfer/csrc/cake_fmha/cuda/decode_balanced_hd64/default.cu` (hd64 — our head_dim — BF16, 2973 lines). Header constants `:40-41, 98-104`:

```
NUM_Q_PIPE_STAGES 2 | NUM_KV_PIPE_STAGES 8 | NUM_WORK_PIPE_STAGES 4
SMEM_SMEM_KV_STAGE_BYTES 16384 | NUM_KV_STAGES 6 | THREADS 512
BLOCK_N 128 | HEAD_DIM 64 | TILE_Q 16 | PAGE_SIZE 16 | SMEM_TOTAL 159616
```

So: **512 threads (16 warps), 156 KB smem, 6×16 KB KV stages = 96 KB of KV in flight per CTA** — 3× our 32 KB — and one warp is a *scheduler*.

Scheduler warp (`:2022-2045`): every CTA computes, on device, from `seq_lens_kv`:

```
total_pairs      = ceil-over-128 of (seqlen - (q_len-1)) summed over batch
ideal_pairs      = ceil(total_work / num_ctas)
balance_k        = ceil(ideal_pairs / 64)
```

then turns a flat ticket into `(batch, q_row, kv_head, block_begin, block_end)` with per-request chunks sized to ~`balance_k` pairs (`:2464-2700`). Tickets come from one grid-wide counter (`:2467-2472`):

```
asm("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;")   // queue_counters[0] += 1
ticket_lane0 = old + num_ctas;
```

**So the KV of a long request is dynamically partitioned into equal 128-key chunks across CTAs, and the partition is recomputed from device data every launch — no host read, hence graph-safe.** Completion is counted the same way (`:2720-2723`: `queue_counters[1] += num_ctas-1`, and the CTA that sees `old == num_ctas-1` resets `queue_counters[0] = 0`), and the split-merge counter is reset by the merging CTA (`:1321-1335` increments `tile_counters[counter_idx]`, `:1469` zeroes it). **No host-side memset of any counter ⇒ CUDA-graph replay with zero host work.** Every queue/counter is self-cleaning.

`flashinfer/flashinfer/cake_fmha.py:300-345` states the neighbouring designs:

* aligned hd64 q1: "one persistent CTA per (request, KV head) whose 8-row query tile holds the KV head's eight query heads" — grid `(1, kv_heads, batch)`, CLC hands cancelled CTAs' coordinates to the persistent loop (`cake_fmha.py:1182-1215`).
* aligned hd256: "the cluster programs split each tile's KV sequence over a cluster of 2, 4 or 8 CTAs and merge the partials through distributed shared memory" — `("cga", 2/4/8)` (`:334-340`, `:1186-1216`).
* request-ordered decode (`flashinfer/flashinfer/cake_fmha_request_ordered.py:15-30, 85-110`): host selects a plan from immutable metadata (`grid`, `workspace_parts`, `total_tiles`) *before* graph capture; the device `request_order` int32 tensor is updated in place between replays, with `request_order_case: "identity" | "length_desc"`. Manifest (`csrc/cake_fmha/request_ordered_paged_decode/cake_fmha_request_ordered_paged_decode_manifest.json`) pins exact routes like `grid:[1,4,8], kv_lens:[897..1793], workspace_parts:4` — i.e. **length-descending order plus a 4-way split for the long ones**.

And a cost model with timings (`cake_fmha.py:944-984`): one-stage (write stage-0 as soon as its PV is in) vs the two-stage persistent form — "in those launches the early store ran 2.5% slower; a launch of one tile per CTA has no next tile (… ran 0.5–0.9 µs faster)"; split vs whole tile — "the one-wave program ran 1.7–3.8% faster than the two-way split; over 1700 keys and more the split ran faster". **Small-KV regimes lose to splitting; long-KV regimes win. Our sliding window is 2048 keys — right at that crossover.**

**Verdict:** three separable pieces, all sm90-legal:
(a) **device-side chunking of KV** from `seq_lens` at launch time (no host read) — the graph-safe form of XQA's split formula;
(b) **one CTA per (request, kv-head) persistent, with CLC or an atomic ticket** so the grid never goes idle when batch × kv_heads ≪ SMs;
(c) **6 stages / 96 KB of KV in flight** instead of 2 / 32 KB — a 3× deeper ring at the same tile size.
Cost: one scheduler warp (of 16) and self-cleaning counters. Expected on our sliding cells: (b)+(c) together are the strongest single lever in this report; (c) alone raises bytes-in-flight 3× and turns a 2–8-iteration pipeline from prologue/epilogue-bound into steady-state. **Estimate −20% to −35% of the sliding cells' time ⇒ −17% to −30% of the attention cell** [INFERENCE], with the caveat that 156 KB smem/CTA costs occupancy (1 CTA/SM) — the reference pays it deliberately and offsets it with 16 warps.

### 4.2 Single-kernel MoE with sentinel handoff (monomoe, BS ≤ 8)

`flashinfer/csrc/fused_moe/monomoe/src/moe.cuh:1-100`. "Top-K MoE kernel — split-phase TMA+WGMMA path for BS <= 8 … five-phase pipeline (routing+prefetch / quantize / up-proj / down-proj / writeback)." Notably:

* per-block private launch counter stored in scratch gives **scale-buffer parity** for the cross-phase handoff (`:100-112`): `flip = spec->launch_flip[blockIdx.x]; scale_parity = flip & 1;` — so consecutive launches reuse the same buffers without a host memset and without a `__threadfence` between phases;
* the *other* parity's sentinel state is zero-refilled **during** this launch "off the critical path — nobody reads those words this launch; kernel completion publishes them before the next launch starts" (`:80-105`);
* one warp handles routing while the rest quantize, and a dedicated prefetch warp issues the routing-window BF16 TMA before routing completes (`:137-190`).

This is the cleanest published answer to "how do you fuse MoE phases without host-visible state and still be CUDA-graph replayable". Our roster closed megakernel, so this is recorded as mechanism, not a proposal — **except** the parity/sentinel discipline, which is a reusable way to keep any multi-phase in-kernel handoff graph-safe (same idea as cake's self-resetting counters).

### 4.3 Warp-decode: a *generated* family of decode MoE schedules

`flashinfer/csrc/fused_moe/warp_decode/cake_warp_decode_contract.cuh` — this is the most interesting non-obvious find for MoE shape-selection, and it is per-token-count:

* `kPackedWorkfeedCtas = 152` (`:36`) — the persistent workfeed pool is one CTA per SM (152 for sm103; ours would be 132).
* `Schedule` (`:114-131`) is `(route layout, route packer, FC1 schedule, FC2 schedule, finalize threads/unroll, workfeed_ctas)`; `Fc1Schedule` enum includes `kPersistentDeviceWorkfeed`, `kPersistentEarlySfbDeviceWorkfeed`, `kPersistentMetadataPublication`, `kFc1Static`…; `Fc2Schedule` includes `kRouteParallelK256`, `kRouteParallelK512DeviceWorkfeed`, `kRouteParallelK512MmaU2DeviceWorkfeed`, `kImmutableWeightPrefill`.
* `SelectSm103aSchedule` (`:255-400`) pins a *different* schedule per geometry **and per token count**, e.g. H4096 I1536 E128 K8: `tokens 6..32` → fused-route-pack FC1 + K512 MMA-U2 device workfeed, workfeed 144 CTAs except `tokens ∈ {6,7,12,13,15,16,17,19,20,21,23} → 0` (device SM count). H4096 I2048 E256 **K6** has `kSwiGLUParameterized` and `kSiTU` variants.
* Comment `:147-152`: "one 1-D grid of `min(grid_m * route tiles, workfeed CTAs)` CTAs, each starting at the tile of its block index, instead of the `grid_m × (workfeed CTAs / grid_m)` pool, with a **five-deep work ring** and every route tile's metadata read once into a 512-row shared-memory table."
* Comment `:86-89`: "the packed route tables are derived inside the persistent FC1 prologue; the graph has no `route_pack` launch and FC1/FC2 re-arm their own workfeeds."

**Verdict:** two takeaways for us, both structural rather than ring-deepening: (i) at 1–32 tokens the correct *number of CTAs* for a MoE workfeed is **≈ SM count, not ≈ 4×SM** (the Marlin heuristic's 4× is for larger M); (ii) route-table construction belongs *inside* the first GEMM launch, not in a separate align kernel. Estimates: workfeed CTA count is worth a few % on its own; deleting the route/align launch is ~1–2 µs/layer [INFERENCE]. This is R2's territory; handed over, not claimed.

### 4.4 Device-side LPT planning (Blackwell FMHA)

`flashinfer/include/flashinfer/attention/blackwell/plan.cuh:67-140`: a single-CTA kernel that walks all `(head, batch, qo_tile)` work items, greedily assigns each to the least-loaded of `num_buckets` virtual queues with cost `kv_len` (causal-adjusted), writes `work_indptr/qo_tile_indices/head_indices/batch_indices`, and runs with `enable_pdl=true`. That is an **LPT (longest-processing-time-first) bin pack computed on device inside the launch stream**: better than block-cyclic assignment when per-tile cost varies 10× (our sliding-vs-full and page-table-ragged cases).

**Verdict:** if a single launch ever mixes tile costs that differ by >2× (e.g. one launch covering both sliding and full cells, or ragged batch), replace `blockIdx → tile` linear mapping with either this device plan or FA's LPT scheduler (`sglang/…/cute/tile_scheduler.py:445-470`, which additionally swizzles over a 50 MB L2 budget). Low cost, moderate win on tail latency: 2–5% of the attention cell [INFERENCE]. Not a new axis beyond "scheduling", but it is a concrete code pattern with source.

### 4.5 Packed NVFP4 KV cache (moonshot support, shipped)

`flashinfer/csrc/fp4_kv_quantization.cu:26-29` (`NVFP4_BLOCK_SIZE = 16`, "linear (non-swizzled) block scale layout") and `flashinfer/csrc/cake_fmha_decode_nvfp4/cake_fmha_decode_nvfp4_bcfe92828a999ce45ff4_kernel.cu:327` — the decode kernel takes **five** TMA descriptors (`Qt, Kp, Vp, Ksf, Vsf`), and dequantizes 4 E2M1 values against one E4M3 scale with a single instruction (`:313-321`: `mul.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4`). API surface: `flashinfer/attention/_core.py:1183-1190` (`kv_cache_sf` required "when the wrapper was constructed with a packed fp4 `kv_data_type`").

At head_dim 64: FP8 row = 64 B; NVFP4 row = 32 B + 4 scale bytes = **36 B, 1.78× narrower**. In a 16 KB stage that is 1.78× more keys per stage at the same in-flight budget. See §6.4 for the gate risk.

---

## 5. DeepGEMM and CUTLASS — small-M rules and the cluster attention design

### 5.1 DeepGEMM (Hopper path is real; MegaMoE is Blackwell)

* **Small-M tile rules** — `deepgemm/csrc/jit_kernels/heuristics/sm90.hpp:20-32`: `block_m_candidates = {64,128}`, **append 16 when `m ≤ 16`, 32 when `m ≤ 32`** ("smaller block M can avoid TMA L2 OOB bound"). At M=1 the reference does not use a 64-row tile.
* **Stage floor for small matrices** — same file `:96-110`: "To hide TMA latency, the stage count should be at least 3; **for small matrices, at least 4**" and rejects layouts with `num_stages < 3 || (block_m*block_n < 128*192 && num_stages < 4)`. Our compile-time `(STAGES-1)×16 KB = 32 KB` is a **2-stage** pipeline; DeepGEMM's own rule forbids that configuration for small tiles. Independent corroboration of §4.1(c).
* **Expert-block sizing model** — `deepgemm/csrc/jit_kernels/heuristics/mega_moe.hpp:96-120`: "Every M block costs roughly 128 extra rows (its tasks reload the weight tiles), so use the fewest blocks per expert… `block_m = num_expected_tokens <= 10 ? 16 : 32`"; the F4/2-CTA cluster + 2 epilogue warpgroups are fixed. At our M=1·top8 over 32 local experts, expected tokens/expert = 0.25 ⇒ **block_m = 16**, i.e. one 16-row tile per routed expert — matches SGLang's H200 special case.
* MegaMoE itself (`sm100_fp8_fp4_mega_moe.hpp`, `deep_gemm/mega/__init__.py`) is sm100-only, multi-rank symmetric-memory, and fuses L1 GEMM + SwiGLU + L2 GEMM store with `post_expert_norm_gamma` folded in. Not available on H200; the baseline uses it only on Blackwell.

### 5.2 CUTLASS `93_blackwell_low_latency_gqa` — the reference answer to "short grid"

`cutlass/examples/93_blackwell_low_latency_gqa/readme.md`:

* "Each **cluster (of size 1x1xMAX_SPLITS)** works on a single kv head in a batch. It divides the kvL evenly into each CTA in the cluster in a **flash decoding style**. And the final reduction is performed by `NUM_REDUCTION_CTA` number of CTAs in the cluster in parallel."
* Worked example: 1 batch, 2 KV heads, 4 Q heads, 6×128 keys, 4-CTA cluster ⇒ 2/2/1/1 tiles per CTA, reduction by 2 CTAs via DSMEM (`store_shared_remote_f32`, `common.cuh`).
* Warp roster: **"7 warps in total, 1 for DMA_Q, 1 for DMA_KV, 1 for MMA, 4 for EPILOG (softmax + cluster reduction)"** — deeper specialisation than FA3's producer/consumer split.
* Supported: dH 64/128, bf16/fp8 non-block-scaled KV, arbitrary seqlen/batch, CUDA graphs, configurable splits, paged KV (`--mode 1`, layout `(num_pages, 2, Page_Size, kvH, dH)` with a shuffled page table "to stress non-contiguous mappings"), attention sink, **sliding window**. Unsupported: persistent schedule, MTP.
* Scheduler: `tgv_gqa.cuh:148-180` — a static per-CTA KV range `[kvL_idx_start, kvL_idx_end)`; no dynamic queue. So the split is chosen host-side, per launch, and the cluster does the merge.

**Verdict:** this is the cleanest cluster form of exactly what our sliding cells need: 4–8 CTAs per (request, kv-head), even split of KV, DSMEM merge, no global partials, no combine launch. On H200 clusters of 2/4/8 are supported and DSMEM works. **Expected: sliding cells reach the flat cells' bandwidth class (2.5–3.5 TB/s) at 4-CTA clusters; realistic capture −25% to −40% of the sliding cells ⇒ −22% to −35% of the attention cell** [INFERENCE]. Risk: cluster launch forbids the CTA-independent preemption that persistent schedulers rely on, and TGV explicitly does *not* support a persistent schedule — so this is an alternative to §4.1(b), not additive.

### 5.3 Hopper mixed-input grouped GEMM (our NVFP4-on-sm90 path)

`cutlass/examples/69_hopper_mixed_dtype_grouped_gemm/README.md`:

> "Currently, the Mixed-input Grouped GEMM only supports **row-wise scaling**, and group-wise scaling for identical problem shapes across all groups. Please contact us if zero-points or **block-wise scaling** are needed."

That is a hard fact for us: on Hopper there is no CUTLASS block-scaled (NVFP4) mixed-input grouped GEMM. Our NVFP4 MoE on sm90 either dequantizes to bf16 before the MMA (Marlin's approach) or writes its own block-scale path. Any plan that assumes a CUTLASS sm90 NVFP4 W4A16 grouped GEMM exists is wrong.
`examples/91_fp4_gemv/91_fp4_gemv.cu` shows the other end: a dedicated `gemv_blockscaled` (M=1) path with `Sm1xxBlockScaledOutputConfig`, i.e. CUTLASS itself ships a FP4 **GEMV** for decode.

---

## 6. Runway — the consolidated answer

**The question:** how do these engines keep enough requests in flight when a decode step is only 8–17 MB of KV?

Four mechanisms, in descending order of measured support:

| mechanism | source | per-CTA bytes in flight | CTAs per (request, kv-head) |
|---|---|---|---|
| **Sequence-partitioned KV, host formula** | XQA `mha_sm90.cu:2960-2971`; TRT-LLM `decoderXQARunnerUtils.h:405-430` | 2 KV buffers × 128 tokens ≈ 32 KB (`nbKBuf=2`) — **same as ours** | `0.75·SMs/(batch·kv_heads)`, capped √(8·blocks), target 8 waves |
| **Cluster split + DSMEM merge** | CUTLASS `93…/readme.md`; cake CGA2/4/8 `cake_fmha.py:334-340` | unchanged per CTA | 2/4/8 CTAs in one cluster, one merge |
| **Persistent grid + device work queue** | cake `decode_balanced_hd64` `:2022-2045,2464-2472`; FA `tile_scheduler.py` LPT; TRT-LLM `DevKernel.cu:658` | **6×16 KB = 96 KB** (`NUM_KV_STAGES 6`, smem 156 KB) | grid = SMs (or 144/152), tiles pulled by atomic ticket |
| **Deeper ring, no split** | FA3/flat cells (151 tiles/CTA) | ours: 32 KB | 1 |

Read the table carefully — **XQA, the reference decode kernel, runs the same 32 KB per CTA as we do.** Its 3–10× advantage at small batch comes entirely from *CTAs per request*, not from a deeper pipeline. cake goes further and does both (6 stages **and** a persistent ticket). DeepGEMM states the floor independently: ≥3 stages, ≥4 for small tiles.

Concrete recipe for our sliding cells (all four are compatible; pick two):
1. **Dynamic, device-side split** from `seq_lens_kv` at launch, XQA's formula (`0.75·SMs/(batch·kv_heads)`), capped by `ceil(kv_len/tile)` and by √(8·blocks) — violates FA3's host-side heuristic, so it must be computed in-kernel (cake's scheduler warp is a working example).
2. **Merge in the last CTA on an acq_rel counter, self-resetting** (XQA `:2756-2762`, cake `:1321-1335`+`:1469`) — deletes FA3's combine launch and the workspace reset, and stays graph-replayable.
3. **If the split is not enough, deepen the ring to 4–6 stages** (`NUM_KV_STAGES 6`, smem 156 KB) — this is the lever XQA chose *not* to pull, so it is additive; smem cost is occupancy.
4. **If clusters are cheap on H200**, 4-CTA cluster + DSMEM merge (TGV) instead of 1+2, accepting loss of persistence (TGV's own limitation).

Not recommended, already closed or contradictory: L2 persistence windows (FA's `size_l2 = 50 MB` swizzle is the same idea), megakernel (monomoe/MegaMoE shapes), fused-slice MoE, deeper MoE rings.

---

## 7. Moonshots — qualitatively different from FA3 + Marlin

Ranked by (novelty × gate-permissibility):

### 7.1 Delete both split launches and both counter resets (high confidence, not really a moonshot)
XQA's last-CTA merge + cake's self-cleaning counters together mean a split decode can be **one launch with no host-visible state**. FA3 needs two (attention + combine) and a workspace reset; TRT-LLM's MoE needs a device-side memset of counters. This is the "algorithmic shortcut the correctness gate permits" with zero numerics change: the merge is the same online-softmax combine, just moved. Evidence: `xqa/mha.cu:2756-2762,2799-2900`; `cake …/default.cu:1321-1335,1469,2720-2723`. Expected: 1 launch + 1 round trip per split decode ≈ 2–4 µs/layer ⇒ **−0.1 to −0.25 ms/step at 62 layers** [INFERENCE].

### 7.2 A different KV layout: packed NVFP4 KV with hardware block-scaled dequant (real, gate-risky)
`csrc/cake_fmha_decode_nvfp4` consumes `Ksf/Vsf` as first-class TMA operands and dequantizes with `mul.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4`. At head_dim 64 that is 36 B/row vs our 64 B — **1.78× more keys per 16 KB stage**, and the dequant is one instruction per four values. Gate risk: the roster already closed "BF16 codecs other than 12-bit", implying KV precision is tightly scored; 4-bit KV with per-16 block scales is a much larger perturbation than a 12-bit codec. Probability of passing an accuracy gate: low. But the *mechanism* (scale tensors as separate TMA descriptors so the KV ring never widens) is worth copying even for an FP8 KV if we ever add per-block scales.

### 7.3 Fuse the attention read into the MoE dispatch (nothing found in any repo)
I searched SGLang/vLLM/TRT-LLM/FlashInfer for a fused attention-output → MoE-input kernel (`pre_attention`, `qkv_quantize_pack`, `fused_norm_combine`, `moe_finalize_allreduce`). What exists is *adjacent* fusion only:
* norm+quant before attention: `csrc/cake_minimax_h3_nvfp4_pre_attention`, `csrc/cake_minimax_h3_qkv_quantize_pack`, `vllm/models/muse_spark/nvidia/ops/prepare_trtllm_nvfp4.py` (`norm_quant_trtllm_nvfp4`, PDL-enabled);
* attention-prologue fusion: `vllm/csrc/fused_muse_spark_qknorm_rope_kv_insert_kernel.cu:1-22` (norm+RoPE+KV insert, "a group of `head_dim*sizeof(scalar_t)/16` lanes owns one head (8 lanes for bf16 head_dim=64)");
* MoE epilogue fusion: `csrc/cake_moe_finalize_allreduce_fusion`, `cake_fused_norm_combine`, `cake_tp_moe_finalize_rmsnorm_backend`.
No kernel consumes the attention output directly as MoE input, in any of the five trees. So that moonshot is unexplored, but the pieces it would need (route pack inside FC1, finalize inside the expert kernel) are exactly what the closed axes forbade. Recorded, not proposed.

### 7.4 Replacing the router GEMM with a one-CTA-per-expert GEMV (real, cheap, new axis)
§1.3: the reference measured that cuBLAS' router split-K + separate reduce "finishes after the main-stream expert input and stalls the MoE", and that at T=1 one CTA per expert over the whole K is faster and deterministic. This is not a megakernel and not an expert-ring change; it is removing a library call and a sync from the critical path at M=1. **Highest confidence-per-risk item in this report**, but it belongs to R2 (min-latency MoE mode) — flagged there.

---

## 8. Ranked top 3 by expected value

**#1 — Device-side dynamic KV split + last-CTA merge on self-resetting counters (attention).**
Evidence: `flashinfer/csrc/xqa/mha_sm90.cu:2966-2977` (`0.75·SMs/(batch·kv_heads)`, 3 resident CTAs assumed) and `mha.cu:3016-3031`; admission/target-wave policy `trtllm/…/decoderXQARunnerUtils.h:405-430` + `decoderXQAConstants.h:27-31` (`128` tokens/block, `√(8·blocks)` cap, 8 waves, `kEnableMinBlockFactor=4`); merge without a combine launch `xqa/mha.cu:2756-2762,2799-2900`; graph-safe counters `flashinfer/csrc/cake_fmha/cuda/decode_balanced_hd64/default.cu:2464-2472,2720-2723,1469`; the whole thing made legal under CUDA graphs by `vllm/v1/attention/backends/flash_attn.py:855-865` ("0 means use FA3's heuristics, not CG compatible"). Calibration that this is *grid*, not depth: XQA runs `nbKBuf = 2`, i.e. the same ~32 KB we cap at (`mha_sm90.cu:142`).
Expected: sliding cells 1.07–1.67 → 2.5–3.4 TB/s ⇒ **−12% to −20% of the attention cell (2.6590)**, i.e. the single largest addressable line here [INFERENCE].

**#2 — 4-CTA cluster split with DSMEM merge for the sliding cells (attention, alternative to #1).**
Evidence: `cutlass/examples/93_blackwell_low_latency_gqa/readme.md` (cluster `1×1×MAX_SPLITS` per (batch, kv-head), even KV split, `NUM_REDUCTION_CTA` DSMEM merge, 7-warp roster, paged KV + sliding window supported); `tgv_gqa.cuh:148-180` (per-CTA KV range); FlashInfer's same design at hd256 (`flashinfer/flashinfer/cake_fmha.py:334-340,1186-1216`). Caveat in the source: "Unsupported features: **Persistent schedule**" — so it replaces #1's persistence rather than stacking with it.
Expected: same ceiling as #1, −10% to −18% of the attention cell; slightly better merge (`DSMEM`, no global partials), slightly worse tail (no work stealing) [INFERENCE].

**#3 — 4–6-stage KV ring instead of 2 stages (attention, additive to #1/#2).**
Evidence: `cake …/decode_balanced_hd64/default.cu:87-96` (`NUM_KV_STAGES 6`, 16 KB/stage, 156 KB smem); `deepgemm/csrc/jit_kernels/heuristics/sm90.hpp:96-110` ("at least 3; for small matrices, at least 4" — and it *rejects* configurations below that). This directly attacks our compile-time `(STAGES-1)×16 KB = 32 KB`.
Expected: 3× bytes in flight; on a 2–8-iteration pipeline that is the difference between prologue-bound and steady-state. **−10% to −20% of the sliding cells ⇒ −7% to −15% of the attention cell**; smem cost drops occupancy to 1 CTA/SM, which is why the reference pairs it with 16 warps [INFERENCE].

Adjacent, handed to R2 rather than ranked here: router GEMV (§1.3/§7.4), MoE workfeed CTA count ≈ SMs (§4.3), route-pack inside FC1 (§4.3), sentinel parity for multi-phase handoffs (§4.2).

---

## 9. Checked and found nothing (or found it closed)

* **Cross-layer / cross-task kernel fusion for attention**: no engine batches two layers' attention into one launch. vLLM's AOT schedule is per-layer metadata only.
* **An attention kernel that consumes a fused MoE**: none in any of the five trees (§7.3).
* **Persistent schedule in the cluster-GQA reference**: explicitly unsupported (`93…/readme.md`).
* **CUTLASS block-scaled (NVFP4) mixed-input grouped GEMM for sm90**: does not exist ("row-wise scaling" only, `69…/README.md`).
* **DeepGEMM MegaMoE on Hopper**: sm100-only files (`sm100_*_mega_moe.cuh`, `sm100_*_mega_gate.cuh`); the sm90 family is plain GEMM (`sm90_bf16_gemm.cuh`, `sm90_fp8_gemm_1d1d/1d2d`).
* **TRT-LLM XQA kernel source inside the TRT-LLM tree**: only the JIT engine (`decoderXQAImplJIT/*.cpp`, `copy_cu.py`); the kernel itself is read from FlashInfer's `csrc/xqa/` above.
* **TRT-LLM `moePlugin` source**: `cpp/tensorrt_llm/plugins/` is not in the tree at this commit; the monolith is `kernels/moe/trtllmGen` (routing + `DevKernel`) and its generated GEMM comes from `trtllmGenKernels/batchedGemm` (not vendored here).

## 10. Discrepancy to verify before acting

The checkpoint config says the 16 full layers use `layer_rope_theta = 0`, i.e. **NoPE** (`sglang/…/muse_spark.py:348`, `vllm/recipes/…/Muse-Spark-Open-816B-A42B.yaml` guide; `vllm/csrc/fused_muse_spark_qknorm_rope_kv_insert_kernel.cu:1-10` "GPT-J interleaved RoPE (skipped on NoPE layers, `cos_sin_cache == None`)"). Our roster records "RoPE theta 5e5" for the model. If our plan applies RoPE uniformly across 62 layers, 16 layers pay a rotation the reference does not — small, but it is a correctness-shaped difference: confirm which layers the harness checkpoint actually tags before optimising the rope path.
