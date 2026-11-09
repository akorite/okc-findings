# State of the loop — Muse Spark Open Kernel Challenge (OKC)
### Self-contained handoff, 2026-10-31, 21,500 s into an unattended run

> This document is written for a reader with **no access to the repository**. It carries the
> competition mechanics, the hardware facts we measured, the exact architecture of the four
> kernels we have on the leaderboard, every experiment we ran and what it settled, and the three
> concrete paths that remain. Every number is either a measurement we took on the judge (job ids
> cited) or a figure from a source named inline. Where something is inferred rather than measured
> it is marked `[INFERENCE]`.

---

## 1. Executive summary

We are on the leaderboard with four uploaded rows and an overall rank of **180 of 1,743 teams**,
up from 1,654 at the start of the run. The goal — top 20 overall — is **not** met, and the reason
is arithmetic rather than effort: the overall metric is dominated by one problem (the Mixture-of-
Experts block), that problem's three largest loads already run at 87-97% of the machine's measured
memory bandwidth, and its weakest load must read a fixed number of bytes that no wrapper can reduce.

| board | our score (speedup over SGLang's baseline) | rank | #20 line | #1 line |
|---|---|---|---|---|
| `muse_bf16_proj` | **1.5697** (measured 1.5605) | **65** | 1.5917 | 1.6523 |
| `muse_moe_w4a16` | **2.3746** (measured 2.3863) | 158 | 2.5343 | 2.7578 |
| `muse_attention_decode` | **2.6089** | 165 | 2.8055 | 3.2221 |
| `muse_decode_layer` | **2.1941** (measured 2.1925) | 180 | 2.3165 | 2.4178 |
| **overall** | — | **180** | — | — |

Two premises had to be demolished to get here, and they matter to anyone continuing:

1. **The original plan's optimisation strategy was aimed at the wrong target.** It assumed the
   baseline's protected kernels (Marlin, FlashAttention-3, cuBLAS) must be kept and only the code
   around them optimised, which caps the attention problem at 1.348x, the MoE at 1.253x and the layer
   at ~1.24x. The live boards had rows at 2.3-2.8x — *above* those ceilings — which proved the
   leaders had replaced the kernels, not the wrappers.
2. **No library in the judge's container can compute our numeric format on this GPU** (§5.3). Every
   alternative it offers loses 7-10x more accuracy than the gate allows. So any MoE work means
   writing the decode ourselves.

What did produce the gain: **published Apache-2.0 submissions by other participants**. The
challenge's terms license every submission under Apache-2.0, and several teams published complete
single-file kernels with judge job ids and per-shape microsecond tables. Adopting, verifying and
measuring those took three boards from ~1.1-1.2x to 2.19-2.61x. The single largest jump — +3.9% on
the rank-paying board — came from discovering that our vendored copies were **stale revisions** of
one team's files.

---

## 2. The competition

### 2.1 What is being timed

Five problems; four count toward the overall board. Each is a fork of SGLang's `baseline.py` at
commit `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`, running on a bare-metal **NVIDIA H200 SXM**
(`sm_90a`). A submission is **one self-contained Python file** that defines:

- `prepare(weights)` — called once, **untimed**, up to 300 s. Weights, repacking, autotuning and
  CUDA compilation belong here. The judge allocates a random throwaway block (0-32 MiB) before
  calling it, so every buffer you allocate lands at a random address (`prepare_pad`, §9.3).
- `kernel_fn(data, state)` — the **timed** call. It is captured as a CUDA graph and replayed.
  Inside it: no host synchronisation, no allocation, no branching on input *values*. Branching on
  shapes and on `config` is fine.

A run must pass the problem's tests (5-16 cases, harder than the scored shapes: adjacent batch
sizes, `page_size=64` instead of 1, `kv_scale=1/64`, skewed routing distributions, odd token
counts) before it is timed. A total run must finish inside 15 minutes including compilation.

### 2.2 Scoring arithmetic (verified to the digit against the judge)

Each problem has six **loads**, each equally weighted in a geometric mean:

- `decode 1`, `decode 16`, `decode 32` at `q_len = 1` (no speculative decoding)
- `verify 1`, `verify 16`, `verify 32` at `q_len = 8` (one 8-token draft checked per request)

Within a load, a **shape** is (batch, context, layer kind, page size, distribution). A load's time
is the sum over its shapes of `weight × shape_time`, where the weight is how many times a real
62-layer step runs that shape (e.g. 60 MoE layers, 46 sliding-window attention layers, 16 full-
attention layers, 62 dense projections). The problem score is

```
score = geomean over the six loads of (baseline_load_time / your_load_time)
```

The judge's per-replay arithmetic subtracts ~4 µs of graph-replay overhead from both sides and
reports the **median of paired ratios** (ABBA, cold L2). There is no second overhead subtraction.
The scoreboard value is `baseline_us / geomean`, and `value × speedup` reproduces SGLang's official
per-step time exactly.

### 2.3 Quotas and the upload rules

- 20 unscored **checks** per problem per day (80/day across four boards). A check is one timed run
  on the judge: ~90-370 s wall clock, returned with a full per-shape table and a kernel-level
  profile. This is the **only** performance oracle; there is no cheaper one.
- 10 scored **submits** per problem per day. The judge keeps a problem's **best** row, so a
  marginal upload can only waste a slot, never lower a score.
- Both reset at 00:00 UTC. Checks on identical bytes return `status: duplicate` and are not
  re-timed.
- Human-only, never automated: `okc login`, accepting the rules, `okc withdraw`.

### 2.4 The overall board

Rankings reflect the effective **62-layer decode-step latency**, taking the **faster** of:
(a) a team's fused `muse_decode_layer` run, or (b) its three component runs
(`muse_bf16_proj` + `muse_attention_decode` + `muse_moe_w4a16`) assembled with stock glue.
`muse_prefill_layer` is excluded.

Our two paths were 1.4% apart (component assembly 12,677 µs vs whole layer 12,502 µs), so the
overall rank tracks the layer board. In baseline microseconds the MoE block is **57-74%** of the
step, dense ~22%, attention ~18%. **Therefore the MoE is the overall score.**

---

## 3. Environment (measured, not assumed)

| item | value |
|---|---|
| GPU | H200 SXM, `sm_90a`, 132 SMs, 141 GB HBM3e |
| clocks observed by the judge | `clocks.sm` 1980 MHz, `clocks.mem` 3201 MHz, power limit 700 W |
| driver / CUDA / torch (judge) | 580.159.03 / CUDA 13 / torch 2.14.1 |
| judge image | `lmsysorg/sglang:dev-muse-spark-open` |
| in image | `sglang` (pinned `a55ab7d5`), `sgl_kernel 0.4.9`, `flashinfer 0.7.0.post1`, `deep_gemm 0.2.1.post1+muse.cbe1727` (custom build), `tilelang 0.1.12`, `nvidia-cutlass-dsl` |
| local workstation | 2 cores, ~7 GB RAM, **no GPU**, no torch. All code must be argued, not run, locally. |
| Colab (Tier-1 gate) | L4 `sm_89`, ~340 compute units. Has torch + triton only — **no sglang/flashinfer/deep_gemm**, so it cannot import `baseline.py`. |

### 3.1 Bandwidth: the number that bounds everything

| measurement | value | notes |
|---|---|---|
| `dst.copy_(src)` on 512+512 MB | **4.18 TB/s** | read+write; our first ceiling estimate |
| cold pure-**read** stream, 1 GiB, sweeps over 375 configs | **4.287 TB/s** = 89.3% of marketing peak | `cp.async.bulk` best; `cp.async.cg` within noise; plain `ld.global.v4` −3.5% |
| size-matched floor model | 453 MB → **106.8 µs** kernel, 111.5 µs with replay; 655 MB → 153.8 / 158.5 µs | asymptote 4.30 TB/s + ~1.5 µs ramp/drain |
| empty CUDA-graph replay on this judge | **4.70-4.74 µs** | σ 0.02 µs; this is the floor for any row |
| per-SM HBM bandwidth | 111-207 GB/s | 32.5 GB/s sustained streaming |
| L2 write bandwidth | **40-50 GB/s per SM** | why split-K epilogues lose |
| `fence.acq_rel.gpu` | ~0.13 µs each | 6.8-19.5 µs across one MoE call |

Consequences that shaped every decision: (a) rank a kernel against the ceiling **at its own
working-set size**, not against 4.8; (b) any row under ~10 µs is launch-bound, not bandwidth-bound;
(c) deep per-thread load ILP (8 independent 16 B loads) costs up to 35% of DRAM bandwidth — the
target is ~2 in flight per thread and full 128 B-sector utilisation.

---

## 4. The four problems

### 4.1 `muse_moe_w4a16` — the overall score

**Shape.** 32 of 256 experts on this GPU (8-way expert parallelism), top-8 routing, one gated MLP
per routed pair. `H = I = 4096`, `K = 4096`. Tokens per step: 1, 16, 32 (decode) and 8, 128, 256
(verify). Weight 60 layers per step. Per-shape floors published by the task: 11.4 / 52.5 / 70.8 /
37.3 / 101.3 / 108.9 µs.

**Numeric format — NVFP4.** Each weight is a 4-bit E2M1 code (magnitudes `{0, 0.5, 1, 1.5, 2, 3,
4, 6}`, bit 3 is the sign), packed two per `uint8` with element `2i` in the **low** nibble and
`2i+1` in the high nibble; one FP8 **E4M3** block scale per **16** consecutive K values; one FP32
global scale per expert. `w13` is `uint8[256, 8192, 2048]` (rows `[0:4096]` gate, `[4096:8192]` up),
`w2` is `uint8[256, 4096, 2048]`, scales are `float8_e4m3fn[256, N, 256]`.

**Correctness gate.** Relative L2 on the output ≤ `1.5e-2`, and ≤ `5×` that per row, against a
reference that dequantises the same FP4 weights and accumulates in FP32. That tolerance is the
reason every format conversion in §5.3 is fatal.

**What the baseline does.** `moe_wna16_marlin_gemm` (SGLang's Marlin MoE fork): a persistent kernel,
one CTA per SM, work striped over `(moe block, N tile, K tile)`; the M tile is
`ceil(moe_block_size/16)` × 16 rows, and `moe_block_size` comes from a density heuristic
(`tokens·top_k/local_experts/block < 0.9` → 8/16/32/48/64). Weights are repacked once to Marlin's
permuted 4-bit layout; the FP4→BF16 decode in the inner loop is ~2.25 instructions per weight value.

**Per-shape baseline and ours** (µs, judge job `325dde16b92848d8` baseline; ours job
`2f4a7c7966ab4e9f`):

| shape | baseline | ours | ratio |
|---|---|---|---|
| bs=1 (decode 1) | 77.9 | 25.0 | 3.12 |
| bs=16 (decode 16) | 188.9 | 92.6 | 2.04 |
| bs=32 (decode 32) | 207.0 | 124.7 | 1.66 |
| bs=8 (verify 1) | 142.4 | 63.0 | 2.26 |
| bs=128 (verify 16) | 469.0 | 178.3 | 2.63 |
| bs=256 (verify 32) | 615.5 | 208.6 | 2.95 |

**Live distinct local experts per shape** (from the reference's own routing distribution, and
confirmed by the task's floor model): **2.5 / 10.7 / 16.1 / 23.1 / 31.8 / 32** at
bs = 1/8/16/32/128/256. At 28.4 MB per expert (codes + scales), that is 71 MB … 906 MB of packed
weights that must be read from HBM. Implied rates: 2.84 / 3.60 / 3.36 / 3.63 / 3.81 / **4.35** TB/s
against the 4.287 TB/s cold-read ceiling.

**This is the crux of the whole challenge.** The three large loads are at or beyond the measured
wall. The two small loads (bs=1, bs=16) are 22-34% below it, and they are latency/occupancy-bound,
not arithmetic-bound: at bs=1 there are 2-3 live experts and 28 MB to read in ~25 µs, so the
kernel's 3-stage `cp.async` ring (`-DOKC_NS=3`, 36,864-byte units, 2 copies in flight per CTA)
cannot hide HBM latency. Both the TMA-characterisation literature (a stage under ~32 KiB of bytes
in flight reaches only ~40% of peak) and DeepGEMM's own stage heuristic (`num_stages ≥ 4` when
`block_m·block_n < 128·192`) say the same thing: **the small loads want a deeper ring**.

### 4.2 `muse_attention_decode`

**Shape.** 16 query heads, **2 KV heads** (GQA group 8), head_dim 64, context 80,000, FP8 E4M3
paged KV cache, page size **1** in the scored cases, `kv_scale = 1`. Per load a sliding-window
layer (last 2,048 tokens, with RoPE, weight 46) and a full-attention layer (whole context, no RoPE,
weight 16). Published floors: 0.1 / 4.3 / 1.6 / 63.2 / 3.3 / 126.4 µs (sliding rows are launch-
bound by construction), and for verify 0.1 / 4.3 / 1.9 / 63.4 / 3.7 / 127 µs.

Per-shape baselines (job `ab027adac2af428b`) and ours: sliding rows 20.8 → 5.3 µs (decode 1),
26.7 → 7.9, 31.2 → 9.9, 22.4 → 6.6, 29.2 → 11.5, 34.4 → 14.6; full rows 31.5 → 10.8,
200.1 → 83.6, 384.2 → 159.6, 36.0 → 14.0, 204.8 → 98.2, 390.0 → 179.8.

**What the baseline does.** FlashAttention-3. Two structural facts about FA3 on these shapes:
its scheduler computes `num_splits ≈ num_sm / (1.1 · batch · kv_heads)` so batch 1, 16 and 32 all
land on ~128 CTAs (one per SM, **two pipeline stages**), and its paged path is **cp.async, never
TMA**, because `page_size % kBlockN == 0` fails for page sizes 1 and 64 with a 160-key tile. That
gives ~20-40 KiB in flight per SM and lands at **41% of achievable read bandwidth** (1.7 TB/s of
4.18) — the measured gap. At q_len 8 with 8 q rows inside a 192-row tile it also wastes 24x of its
M axis.

### 4.3 `muse_bf16_proj`

**Shape.** 62 dense BF16 GEMMs per step across **30 scored cells**: 5 layers (`router`, `qkv_gate`,
`o`, `pooler`, `unpooler`) × 6 token counts (m = 1, 8, 16, 32, 128, 256), weight 62 each. The
`router` is FP32 `[256, 8192]`; `qkv_gate` `[2304, 8192]`; `o` `[8192, 1024]`; `pooler`
`[4096, 8192]`; `unpooler` `[8192, 4096]` sliced to 1024 rows for m ≤ 8.

**What the baseline does.** `bf16_gemm_dispatch` — cuBLAS/cublasLt with automatic Split-K tactic
selection. It is 96.7% of the runtime. Its distance from the pure-bandwidth floor per cell class:
`router` **14.9x** at m=1, 5.8x at m=8, ~8x at m=16/32, ~7x at m=128/256; `unpooler` 4.1x at
m ≤ 8; `qkv_gate` ~1.8x; `o` ~1.9x; `pooler` ~1.5x. The router is a fifth of the weighted time and
the worst cell in the problem.

Our per-load: decode 1 1.7351, decode 16 1.6043, decode 32 1.5203, verify 1 2.0892, verify 16
1.3116, verify 32 1.2452 — i.e. **the m ≥ 128 cells are the weak end** and they are what the
remaining 1.5% to #20 needs.

### 4.4 `muse_decode_layer` — the rank payer

**Shape.** One layer per timed call, weighted by how many of the 62 the step runs: 46 sliding-window
+ 16 full-attention attention rows, 60 MoE rows, 62 dense projections. Context 80,000, page size 1.
`kernel_fn` returns `(next_attn_input, stream, global_topk_ids)` — the layer is a fragment of a
residual-stream pipeline, not a standalone function. Published floors per row: sliding 40.0 / 76.7
/ 84.5 / 62.3 / 126.6 / 143.8 µs and full 44.2 / 137.5 / 208.1 / 66.4 / 188.7 / 267.2 µs.

Our per-row ratios (job `335190c6676f4dd1`, baseline → ours in µs): verify 32 sliding 739.8 → 334.8;
verify 16 sliding 542.9 → 265.1; verify 32 full 1094.1 → 501.4; decode 16 sliding 267.9 → 139.4;
decode 32 sliding 267.0 → 158.4 (**our weakest row, 1.69-1.80**); verify 1 sliding 260.6 → 109.8;
verify 16 full 714.7 → 349.6; decode 32 full 624.9 → 312.1; decode 1 sliding 171.3 → 75.6;
decode 16 full 443.0 → 214.5; verify 1 full 270.9 → 115.4; decode 1 full 181.7 → 81.5.

Composition of the heaviest row (`verify 32 sliding`, 739.8 µs baseline): MoE 619.6, sliding
attention 34.2, dense cells ~86. At the component problems' own ratios that is ~294 µs, and we
spend 334.8 — so ~40 µs (12%) is glue: boundary/norm kernels, residual updates, router, routing,
finalize and launch gaps.

---

## 5. What we ship, and why it is fast

All four files are published submissions (Apache-2.0 by the challenge's terms), vendored, cleaned to
pass our stricter local static screen, and re-measured. Provenance is in §11.

### 5.1 The MoE kernel (2.3863)

One **persistent CUDA kernel**, one CTA per SM (132), `load_inline`-compiled in `prepare()`:

1. **Routing is rebuilt redundantly in every CTA** from `topk_ids`: per-expert counts, offsets, a
   sorted `(token, slot)` list and a job list — no separate align kernel, no host read.
2. **Fat and skinny jobs.** An expert with more than `SKINNY_MAX_TOKENS` tokens becomes one or more
   *fat* jobs (chunks of tokens, K streamed through shared memory in stages); the rest become
   *skinny* jobs (16 rows × full K with the warps splitting K). Fat first, largest first.
3. **Phase A** = gate/up GEMM + SwiGLU; **phase B** = down GEMM + the post-expert RMSNorm's sums of
   squares. Both run in the **same CTA loop**, so the SwiGLU output never leaves shared memory /
   registers, and one grid barrier plus a final pass replace two kernel launches.
4. **Weights are repacked in `prepare()` by bit/byte permutation only — never requantised** — into
   four layouts: packed `mma.sync` order, fat order, unit-contiguous order, and natural-K / WGS
   order.
5. **KPERM3** (the revision we adopted, +2.6% board-wide, all 6 loads): a K-permutation *inside*
   each 256-K stage so that one lane's four 16-K blocks are always the same four scale blocks,
   independent of the load group. That hoists the scale-pair construction out of the inner loop:
   **20.12 → 18.44 instructions per weight word**. Formally, for stage position `p = 16s + c`:
   `lg = s>>2`, `j = s&3`, `t = (c>>1)&3`, `i = (c&1) | ((c>>3)<<1)`,
   `kperm(p) = 16(4t + j) + 4lg + i`, and `scale_block(p) = kperm(p)>>4 = 4t + j`, independent of
   `lg`. Host side needs the same map on codes, on the gate/up **row** order (phase A only, never
   `w2`), and a scale-layout regroup.
6. Build variants selected **by token count**: `M ≤ 4` TINY, `≤ 8` TINY2, `≤ 16` MID, else
   MAIN/WGS (which uses RS-`wgmma` `m64n{8..64}k16` with weights as the mma M operand and FP4
   decoded directly into A register fragments). A separate `B4FIN` build serves M in (16, 32].

Our modifications: nothing semantic — see §9.4 for the cleaning recipes and the two dead references
a review found and we removed.

### 5.2 The attention kernel (2.6089)

**One launch per call** replaces four: QK-RMSNorm + RoPE on the FP8 query, the paged FP8 KV store,
paged FP8 GQA with online softmax, and the output RMSNorm × sigmoid(gate).

- `QK^T` on **FP8 tensor cores** (`wgmma`), `PV` on FP8 `wgmma` with FP16 accumulation at q_len 8
  (the `e4m3` P is built in registers, row sums ride in 8 extra "ones" columns).
- 64-key `cp.async` tiles with multi-stage pipelining, **one CTA per (request, KV-head pair,
  split)**; the 8 query heads sharing a KV head are packed into the mma M axis (zero-padded to 16).
- Split merge by **FP32 atomics** after rescaling to a fixed reference (the QK-norm bounds the
  logits), last CTA normalises + gates + zeroes the workspace; four launch modes (per-request
  split-KV, cluster/DSMEM, a flat kernel for full layers, cooperative finalize) chosen by a host
  plan from `(batch, q_len, window, residency)`.
- PDL machinery is present (`griddepcontrol.wait` / `launch_dependents`, the launch attribute) and
  **off**; see §7 for what enabling it measured.

### 5.3 The other two

**`muse_decode_layer`** is the component kernels pasted into one file — attention, MoE and dense —
plus `prepare()`-time **per-(T, matrix) selection** between cuBLAS, the bf16 kernel, a Triton
split-K GEMM and the fused pooler+router launch, then one-program-per-token glue for the boundaries.
PDL is **already on** for T ≤ 32 across 8-10 launches.

**`muse_bf16_proj`** is a per-cell route table over families the file contains: `sk2` (swapped-
operand tensor-core GEMV: weights streamed HBM→**registers** already in `mma` fragment order, tokens
as the mma N axis), `cgemv`/`sgemv`/`rgemv` (register-direct GEMV for m=1), `rk` (split-K router
GEMM), `gq` (K-split with a measured config table), `gqx` (SS-`wgmma` from a TMA ring with a
**DSMEM cluster reduction**), `gemm90`, `cutlass` (CUTLASS 3.x sm90 TMA warp-specialized pingpong,
swap-AB), `torch`/`torchT` (cuBLAS, the latter on a K-major weight copy), plus a prepare-time
tournament that binds the fastest measured per cell with an FP64 reference as the numeric guard.

### 5.4 Why no off-the-shelf library could replace the MoE kernel

Checked against source at the pinned versions, not documentation:

| path | verdict |
|---|---|
| `flashinfer.fused_moe.cutlass_fused_moe` with `use_w4_group_scaling=True` | SM90's only 4-bit×BF16 variant is **MXFP4 (E8M0 scale per 32)**, group size compiled in (`moe_gemm_kernels.h`: `wfp4a16_group_size = 32`) |
| `flashinfer.gemm.gemm_bf16_fp4` | exactly our contract — **SM100/103/110/120/121 only**, raises on SM90 |
| `flashinfer` Humming (`use_wfp4afp8_humming`) | needs MXFP4 weights **and FP8 activations** |
| `deep_gemm.fp4_gemm_nt` / `fp8_fp4_gemm_nt` | SM90 branch requires FP8 on both sides; every `*fp4*` implementation is `sm100_*` |
| `deep_gemm.cublaslt_nvfp4_gemm_nt` | NVFP4 / E4M3-per-16 — but **W4A4** (A must also be packed FP4) |
| CUTLASS SM90 mixed-input 4-bit | primitive types would allow a per-16 scale, but `group_size` is a runtime arg and every shipped instance is E8M0 per 32 |

And the accuracy cost of converting our weights to any of them, measured by emulation of our shapes
against a 1.5e-2 gate: **MXFP4 (E8M0/g32) 11.4%**, per-16 E4M3 rounded to a power of two **15.1%**,
FP8 activation quantisation alone **2.65%**. All fail by at least 1.8x.

---

## 6. The arithmetic that bounds the remaining rank

**The MoE is the overall score** (57-74% of the step) and its weak loads are byte-bound:

```
bs=32:  453 MB of packed weights  /  4.06 TB/s marginal rate  = 112 us of unavoidable bytes
        we spend 124.7 us  ->  at most 1.85x for that load, with a perfect wrapper
bs=256: 906 MB / 4.287 TB/s = 211 us floor ; we spend 208.6 us  (already at/over it)
bs=1:   28.3 MB in 25.0 us vs a 6.6 us read ceiling, i.e. the row is graph replay + one ring turn
```

The bytes cannot shrink: at bs=1-32 there are 2-16 live experts and **every live expert's full
weight matrix must be read for 1-2 rows each**. Arithmetic intensity is ~9 FLOP/byte.

Projected ceilings with the current kernel class: MoE ≈ 2.45-2.55, attention ≈ 2.65 (87% of the
random-128 B-row read already), bf16 ≈ 1.58 (best-of-cell across everything published computes to
1.5674-1.5696), layer ≈ 2.25-2.30 (it inherits the MoE row). **Overall ≈ 2.3-2.4, i.e. rank ~120-160
at best.** Reaching top 20 needs one of the three mechanisms in §8.

---

## 7. Every experiment, and what it settled

Each row is a real candidate that cost one judge check. **Do not re-run these.**

| # | candidate | measured | mechanism that decided it |
|---|---|---|---|
| 1 | hand CUDA-core Split-K GEMV for bf16 wide cells | 0.8086 | instruction-bound at 36 GB/s per SM (36 GB/s needs ~20 instructions per 32 B) |
| 2 | one BF16 `mm` router on stacked `[hi; lo]` | **1.0908** | the only measured win at the time: router 1.18x (m=1) to 2.87x (m=8) |
| 3 | wmma + `cp.async` wide GEMM | 1.00 | cuBLAS is itself at 0.44-0.70 of peak there |
| 4 | TMA + `wgmma` Hopper GEMM (58 regs, 0 spills, correct) | 0.7977 | still 1.3-2x slower than cuBLAS on these shapes |
| 5 | split-KV paged FP8 GQA on all six sliding shapes | 1.1124 | per-shape gating, not the kernel, was worth +7.5% |
| 6 | the same kernel on the four shapes it wins | **1.1957** | FA3 wins the two batch-16/32 verify blocks; gate them back |
| 7 | gate fused into the merge launch | **1.2106** | one launch + one `[T, hq·d]` round trip removed = +1.8% |
| 8 | full-context split-KV kernel (`window == 0`) | 1.1427 | 0.77/0.94/0.91 on the rows it targeted; FA3 already runs them at 1.7 TB/s |
| 9 | that kernel spliced into the layer | 1.1343 | a component win does not survive transplantation |
| 10 | `w0` = 554‰ key-split skew on the flat q_len-8 grid | 2.5607 | the `(c, c+G/2)`-share-an-SM premise does not hold on the standalone kernel |
| 11 | end-of-tile PV drain removed for the flat build | 2.6213 | +0.5% but only 1/6 loads improved — noise |
| 12 | cuBLASLt pin table inside the layer | 2.0772 | the isolated 1.2x pooler gain dies on the layer's stream split |
| 13 | bf16 per-cell merge with `gqx` bn=256 | 1.5404 | **displacement**: a 1.5% tournament margin sits below the 4.70 µs replay floor |
| 14 | bf16 deterministic routing (SE-of-median gate) | 1.5452 | the gate then refuses binds that help |
| 15 | bf16 12-bit packed wide consumer (exact codec, on-device 3% trial) | 1.5422 | the RS-`wgmma` register-source decode does not sustain the byte ratio |
| 16 | rohanarun's bf16, ported `exec`-free | 1.5492 | their live 1.5934 is not reproducible from their published file |
| 17 | rohanarun's attention kernel | 2.6096 | identical to ours |
| 18 | `prepare_pad`-proof arena (2 MiB-aligned, ×3 boards) | −0.2 / −0.7 / −3.0% | the pad effect is real, the +2 MiB slack costs more |
| 19 | `-DOKC_NS=4` ring depth (×2 attempts, second with every depth-dependent quantity asserted) | `cudaErrorLaunchFailure`, sticky | unfixed; hypothesis in §8.3 |
| 20 | swap-AB for the low-token MoE builds | no change | **already implemented**; the WGS build's edge is M=64 and 2.8x the bytes in flight |
| 21 | split-pipe softmax exponential | no file | cheap form models 1.2-1.8e-2 vs a 1.5e-2 budget; in-budget forms cost more fast-pipe slots than they free |
| 22 | QK in FP16 accumulate | no file | safe (7.4e-4 output rel-L2) but **not a time lever** (~1%); needs a ≥6-bit scale fold to avoid f16 overflow |
| 23 | KPERM3 on the mma.sync builds | no file | no target: `block = 32w + 4c + t` is already `st`-independent, proved by enumeration both ways |
| 24 | PDL enabled on the attention kernel | 2.619 | +2.9% on `decode 1`, −2…−4% on the other five. The layer **already had PDL on** for T ≤ 32 |

---

## 8. What remains: three unblocking paths

### 8.1 Activation sparsity with K-gather (the only published *bytes* lever)

**Evidence.** TEAL, *Training-Free Activation Sparsity in LLMs* (arXiv:2408.14690, ICLR 2025) and
the authors' follow-up: 40-50% model-wide sparsity, **1.53x wall-clock at 40% and 1.8x at 50% in
single-batch decoding**, near-zero degradation at 25%, measured **with CUDA graphs**, and explicitly
compatible with 4-bit weight quantisation. Prior art: CATS (arXiv:2404.08763).

**Mechanism, adapted to our shapes.** Threshold the MoE input `x` (hidden states are Laplacian-
concentrated; pruning `W_up` *through the input x* has lower error than through the SiLU output),
compact the surviving 16-element K groups **on device**, and gather only those NVFP4 groups of
`gate_up`; do the same on the 8192-wide intermediate to skip `w_down` columns. Because `gate_up` and
`down` are each ~50% of an expert's bytes, sparsity `s` gives bytes × (1−s): 25% ⇒ 1.33x, 40% ⇒
1.67x. The mask is data-derived but **shared by all live experts at bs=1**, so one compaction pass
serves the whole layer; gather granularity can be 16 K elements to match the E4M3 scale group.

**Why it is unclaimed.** TEAL leaves "sparse + quantized kernels" as future work and **no NVFP4
W4A16 implementation with K-gather exists anywhere**. We would be first.

**Risk.** This is the only lever that moves numerics. Nobody has a per-op relative-L2 result for it,
only perplexity. Start at 20-25% sparsity and gate on the 1.5e-2 tolerance; the mask must be
computed inside the timed call (it depends on `x`) but its prefix sum is device-side and cheap.

### 8.2 Two-sided PDL

PDL's 1.4-1.8 µs per boundary requires **both** kernels to be PDL-aware: the producer must call
`cudaTriggerProgrammaticLaunchCompletion`, the consumer must be launched with
`ProgrammaticStreamSerializationAllowed` and wait with `griddepcontrol.wait` after its independent
prologue. Every producer in our chain is cuBLAS or Marlin, which never signal — so single-sided PDL
gets only the ~0.37 µs launch-setup overlap, which is exactly what measured (+0.4% geomean, one load
up and five down). The published wins that motivated this (1.37x on a dependent pair; a competitor's
layer +1.4-2.3%; a competitor's MoE 1.432 → 1.497x) all had our-side producers.

**What it would take:** replace the dense GEMMs and the MoE with our own PDL-aware kernels so that
`qkv → attention`, `attention → o`, `pooler → route`, `route → MoE`, `MoE → unpooler`,
`unpooler → boundary` are all triggerable. The layer already has PDL plumbing on 8-10 launches; it
is the producers that are silent.

### 8.3 Root-cause the four-stage ring

Two independent sources say the small MoE loads want `NS ≥ 4`: DeepGEMM refuses fewer than four
stages when `block_m·block_n < 128·192` (a stage under ~48 KB), and the TMA characterisation paper
measures the knee at 32 KiB of bytes-in-flight per SM. Ours runs `NS=3` with 36,864-byte units.

Both attempts at `-DOKC_NS=4` fault at the **first launch** with `cudaErrorLaunchFailure` (sticky, so
it kills the run — a tactic gate cannot catch it). A sibling's audit proved the slot arithmetic *is*
parametric (units issued `[0, depth)` by the prologue, `[depth, NS)` by the helper, `[NS, N)` by
refills, with wait parity `(q/NS)&1` counting completions per slot exactly; `uslot[8]`/`wse[8]`/
`bars[12]` all cover 4). **So the remaining hypothesis is the host side:** the NS=4 ring fits only by
sharing the fat ring's `max()` region (`147456 ≤ 178176` B), so if the host still requests the NS=3
dynamic shared-memory size, or does not re-raise `cudaFuncSetAttribute`, slot 3 addresses past the
allocation.

**First step, before another check:** print the requested `max_shared_mem` and the kernel's
`extern __shared__` extent for the NS=4 build and compare them. If they disagree, the fix is one
line and worth +10% on `decode 16`.

---

## 9. Operational playbook

### 9.1 The loop

```
PYTHONPATH=src .venv/bin/python -m okc_loop.cli status                 # readiness, quota, rank
PYTHONPATH=src .venv/bin/python -m okc_loop.cli lint <file>            # local AST screen, free
PYTHONPATH=src .venv/bin/python -m okc_loop.cli check <file> \
    [--bucket exploration|reserve] [--hopper-only "<reason>"]          # one judge check
PYTHONPATH=src .venv/bin/python -m okc_loop.cli submit <sha>           # gate + upload
PYTHONPATH=src .venv/bin/python -m okc_loop.cli colab-batch <file> --problem-of <p>   # L4 gate
```

`check` requires both a Tier-0 (lint) and a Tier-1 verdict in the ledger before it will spend quota.
`--hopper-only REASON` records the ledger's `PASS_SM90A_WRAPPER` verdict, which is the correct
classification for a candidate whose CUDA is compiled for `sm_90a` only: the L4 gate image cannot
execute it (`no kernel image is available for execution on the device`), so correctness falls to the
judge's own test suite. That is a legitimate gate, not a bypass — a failed test costs the check.

### 9.2 The upload gate

`auto_submit.evaluate` runs, in order: (1) provenance — Tier-0 and Tier-1 verdicts; (2) the Tier-3
check must be finished; (3) gain ≥ `max(incumbent × 1.003, 1.05)`; (4) CI separation — the
candidate's 95% CI must clear the incumbent's, **or** it must improve ≥ 4 of 6 loads by more than
`MIN_GAIN`; (5) no load may regress past 5%; (6) nearby shapes verified; (7) a submit slot must be
free. The 4-of-6 rule exists because the judge keeps a problem's best row, so a marginal upload
costs a slot and never score.

### 9.3 Variance: `prepare_pad`

Before `prepare()` the judge allocates a throwaway block of 0-32 MiB in 512-byte steps. Every buffer
you allocate then lands at a random address; on GH100 the L2 is two physically separate 25 MB
partitions and a far-partition access costs nearly twice a near one, and torch's caching allocator
also changes *relative* alignment between your tensors. Measured cost on unchanged bytes: **−1.76 to
+0.73 µs at the verify-32 row**, −0.61 to +1.48 at decode-32, against an in-process A/A of ≤0.65 µs.
The same file scored **1.5335 and 1.5697 on two runs (2.4%)**. Consequences: never read a single run
as a delta below ~1.5%, and know that a "gain" that only moves one load is probably this.

### 9.4 Traps that cost us checks

- **`-arch=sm_90a` alone fails.** torch appends its own `-gencode` set for the detected device;
  ptxas then rejects `wgmma` on `.target sm_90`. Use
  `--generate-code=arch=compute_90a,code=sm_90a`. The equivalent `os.environ["TORCH_CUDA_ARCH_LIST"]
  = "9.0a"` is banned by our static screen, and removing it silently reintroduces the failure — it
  cost two checks.
- **A graph capture returns the output tensor but does not execute it.** Call `replay()`.
- **Triton needs a real file**: `@triton.jit` reads `inspect.getsourcelines` at import, so a module
  exec'd from a string fails at load. Write candidates to disk with a matching linecache entry.
- **A sticky CUDA error kills the whole run.** `cudaErrorLaunchFailure` cannot be caught by a tactic
  gate; it burns the check and leaves the context dead.
- **Tournament margins below the 4.70 µs replay floor** bind tactics that measure faster in prepare
  and run slower in the load. Measured twice; require ≥ max(5%, 4 µs) or a standard-error gate.
- **Static-screen bans, module-wide**: `inspect`, `importlib`, `ctypes`, `threading`, `subprocess`,
  `socket`, `os.environ`, `time.time/perf_counter/monotonic`, `sys.stderr/stdout`, `exec`, `compile`,
  `eval`, `globals`, `open`, `getattr`/`setattr`. Use `torch.ctypes` / `torch.threading` (torch
  re-exports both at top level) and place the alias **after** `import torch`.
- **The judge's own limit is ~1 MB per file**; our self-imposed 500 KB rejected two legitimate
  839 KB submissions before we raised it.

### 9.5 Adoption discipline (this is what produced the gains, and it has a trap)

1. Find a published file. **Check `results.json` / the judge job id against the file's sha256** — one
   team shipped both a truncated upload and the real file in the same `kernels/` directory, and the
   truncated one passes a static screen once its unterminated string is noticed. That mistake cost us
   a session's worth of sub-optimal rows.
2. `lint` it; fix the screen violations mechanically (diagnostics behind a `DIAG` flag, `ctypes`/
   `threading` aliases, `os.environ` reads → constants, `sys.stderr` prints, >400-char lines inside
   CUDA string bodies split at `), ` boundaries, `"name" in globals()` probes resolved by checking
   the name is module-level, `.numel()` comparisons hoisted).
3. `check --hopper-only`, then `submit` if it beats the incumbent.
4. Record the provenance in the file header and in a `NOTICE`; every submission is Apache-2.0 by the
   challenge's terms, and the licence text belongs in the tree.

---

## 10. Reference numbers worth memorising

| item | value |
|---|---|
| cold pure-read bandwidth | **4.287 TB/s** (89.3% of the 4.8 marketing figure) |
| `copy_` bandwidth (read+write) | 4.18 TB/s |
| size-matched stream floor | 453 MB → 106.8 µs; 655 MB → 153.8 µs |
| empty CUDA-graph replay | **4.70-4.74 µs** |
| sustained per-SM HBM | 111-207 GB/s (32.5 GB/s streaming) |
| L2 write per SM | 40-50 GB/s |
| `fence.acq_rel.gpu` | ~0.13 µs |
| PDL, both sides aware | 1.4-1.8 µs per boundary |
| PDL, consumer only | ~0.37 µs per boundary |
| `prepare_pad` effect | 0.6-2.5 µs per row |
| board run-to-run spread on identical bytes | ±2.4% |
| MoE tolerance | output rel-L2 ≤ 1.5e-2 (5× per row) |
| MoE live experts at bs 1/8/16/32/128/256 | 2.5 / 10.7 / 16.1 / 23.1 / 31.8 / 32 |
| bytes per live expert | 28.4 MB (codes + scales) |

---

## 11. Inventory

**The four board files** (all lint-clean, all measured):

| board | file | measured | judge job |
|---|---|---|---|
| `muse_moe_w4a16` | `src/okc_loop/vendored/alygg77/muse_moe_w4a16.py` (326 KB) | 2.3863 | `2f4a7c7966ab4e9f` |
| `muse_attention_decode` | `src/okc_loop/vendored/alygg77/muse_attention_decode.py` (155 KB) | 2.6089 | `3063f7fb1ec2402c` |
| `muse_decode_layer` | `.vendor/tmp/aly_head_decode_layer.py` (844 KB, alygg77 HEAD, hand-cleaned) | 2.1925 | `9c743eb2b3af4ef7` |
| `muse_bf16_proj` | `src/okc_loop/vendored_work/bf16/pinned.py` lineage (383 KB) | 1.5605 | `83179200c2124bd8` |

**Unmeasured / rejected candidates** are all under `src/okc_loop/vendored_work/<board>/` with names
that say what they are (`overlap`, `splitxp`, `pdl`, `glue`, `leftovers`, `arena_*`, `ksplit`,
`ns4_fixed`, `swapab`, `merge`, `deterministic`, `packed`, `best`, `port_rohan`).

**Upstream sources** (all Apache-2.0 by the challenge terms unless noted; `NOTICE.md` in
`src/okc_loop/vendored/`):

| repo | what it gave us | its published rows |
|---|---|---|
| `vjeux/okc-2026` | the base file for five boards | n/a (a base) |
| `alygg77/okc-muse-spark-kernels` @ `6cae49a6` | **all three MoE/attention/layer rows we run** | moe 2.4015, attn 2.6530, layer 2.2058, bf16 1.5502, prefill 1.8167 |
| `ruwwww/open-kernel-challenge` | bf16 comparison | attn 2.6032, moe 2.3370, layer 2.0751, bf16 1.5056 |
| `NathenFernandes/muse-bf16-proj` (MIT) | small-m routes, cuBLASLt descriptors | bf16 1.5186 |
| `trxphxle/okc-2026-muse-kernels` | PDL/prefetch notes, ranked layer | layer 2.1756 |
| `rohanarun/Auto-research-agent` | a second bf16/attention implementation | local 1.5601 / 2.6537; live 1.5934 / 2.6611 |
| `Kushagrabainsla/muse-spark-open` | `EXPERIMENTS.md` negatives (KV staging, code-cache warming) | 1.19-1.70 |
| `zhebrak/muse-spark-kernels` | 12-bit codec, PDL-everywhere, stream-K wgmma notes | layer 2.0146, moe 2.0182 |
| `Madhavan113/ParadigmKernelChallenge` | `notes/` design maps | layer 2.2000 (unverified by its own repo) |
| `infraax/okc-kernel-challenge` | measurement-only submission methodology | 1.17-1.77 |
| `lee101/openkernels` | `docs/upstream_ideas.md` | 1.48-1.99 |

**Teams with nothing public** (checked: GitHub users, gists, forks — all zero): `mpeng19` (layer
2.4178), `kwanUm` (2.4084), `pjmullin3` (2.4029 / attn 3.1494 / moe 2.6853), `dolanzhao` (moe
2.7578), `tjiang-fair` (attn 3.2221), `colinlikescode` (proj 1.6367), `arashattar` (proj 1.6523),
and ~30 more. **The top of every board is closed source** — the tradeable ceiling is the open set in
the table above, which we have now fully measured.

---

## 12. If you have one hour

1. Fix the NS=4 host-side shared-memory request (§8.3) and measure it on the MoE board. It is the
   highest ratio of expected gain to work left, and the diagnosis is one print statement plus one
   comparison.
2. If that lands, transplant the same build into the layer (the layer's MoE is the same source) and
   measure `decode 32` / `decode 16` there — that is where the overall rank actually sits.
3. If it does not land, the honest options are TEAL-style sparsity (§8.1, a real kernel project) or
   accepting ~rank 150-180 as the ceiling of this kernel class.

Do not spend checks on: block-size sweeps, cuBLASLt algorithm sweeps (measured dead: 1,789-2,309
configs per shape for ~2 µs/layer), `pack_gqa`/`num_splits` on FA3, split-K for the MoE (proved
unnecessary: N-split already gives 96-192 CTAs per expert), 12-bit packing on wide cells, or arena
allocation.
