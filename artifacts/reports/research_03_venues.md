# research_03_venues — non-arXiv venues + vendor/industry material (2026-11-03)

Scope as assigned: MLSys / ASPLOS / ISCA / HPCA / MICRO / SC / PPoPP / CGO / PACT / NSDI / OSDI 2024–26,
GTC/Hot Chips talks, Colfax / Red Hat / PyTorch / DeepSeek / vLLM engineering posts, and the NVIDIA PTX ISA,
CUTLASS and Transformer Engine docs. arXiv is cited only where the same work has a proceedings version.
Filter: what changes a decision on **one H200 (sm_90, 132 SMs, 4.8 TB/s, 50 MB L2)** for the four families

* **F1** = NVFP4 W4A16 MoE decode, ≤19 rows/expert, 32 local experts × 28.31 MB, D_HIDDEN 8192 / D_EXPERT 4096, top-8
* **F2** = paged FP8 decode attention, head_dim 64, GQA 8:1 (2 kv heads/rank), window 2048, 46 sliding : 16 full
* **F3** = dense BF16 projections (incl. the bf16 board and the 12-bit codec question)

Every claim below carries a venue/vendor + title + URL. §6 lists, per venue, where the search returned nothing.

---

## 0. Headline

Three findings that should change what we do next, in order of confidence:

1. **The hardware has no E2M1 path on sm_90a, at all.** `cvt.rn.satfinite.e2m1x2.f32`, `cvt.rs.*.e2m1x4.f32`,
   `add/sub/mul/fma` with `.e2m1x4`/`.e2m1p4x4` are **sm_100a/103a/107a/120a only** in PTX ISA 9.4, and NVIDIA's own
   Transformer Engine doc says NVFP4 is a *Blackwell* datatype. Any plan that names the hardware `cvt` is dead. (§3.1–3.2)
2. **Published Hopper practice at tiny M is *not* "abandon wgmma" — it is "swap the operands".** Three independent
   industrial implementations put the *weights* on wgmma's M=64 side and the *tokens* on N, whose granularity is 8,
   and dequantize into register A fragments (RS mode) so the weights never re-enter SMEM. That is the same
   structural fix our block-M-8 `mma.sync` lead is reaching for, but it keeps the wgmma math path. (§1.4–1.6)
3. **12-bit lossless palette+escape is at 96 % of its own roofline and has been independently replicated twice.**
   NyoomFloat12 measures 1.28× against a 1.33× theoretical maximum; a live vLLM RFC describes the identical format
   (including the "+0 escape") and measures 1.23–1.27×. There is ~4 % left at 12 bits. (§4)

---

## 1. Axis 1 — MoE decode at sub-19 rows per expert

### 1.1 The structural cause: wgmma's M is hard-fixed at 64

* PTX ISA 9.4, *Asynchronous Warpgroup-Level Matrix Shape* — every `wgmma.mma_async` shape is `.m64nNk16`;
  N is the flexible axis. https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shape
* Colfax Research, *CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on NVIDIA Hopper GPUs* (2024-08),
  states it without hedging: "`M` is always 64, `N` is a multiple of 8 from 8 to 256, and for 16-bit operand
  datatype, `K` is 16"; and "`bM` should be a multiple of 64".
  https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/
  Same page, the two facts that decide the swap: "**operand `B` must always be stored in shared memory**. In
  contrast, operand `A` can be located in either SMEM or register memory (RMEM)".

At 19 rows/expert a `wgmma` M=64 tile spends 70 % of its rows on padding before any dequant cost is counted.
That is the mechanism behind the 4–29 % gap in our lead. It is not a tuning artefact.

**Quantified padding tax at our size** (19 rows/expert max, F1):

| CTA/MMA tile M | padded rows | waste |
|---|---|---|
| 8 (our `mma.sync` lead) | 24 | +26 % |
| 16 (`mma.sync` m16n8k16, FlashInfer's stated TC minimum) | 32 | +68 % |
| 64 (one wgmma M tile) | 64 | **+237 %** |

This is corroborated as a *scheme* by MLSys 2026 below: the padding overhead "depends not on skew per se, but on
the interaction between the token distribution and the kernel's padding mechanism."

### 1.2 FlashDecoding++ — MLSys 2024 (flat GEMM, the "pad to 8" argument)

Hong et al., *FlashDecoding++: Faster LLM Inference with Asynchronization, Flat GEMM Optimization, and Heuristics*,
Proceedings of the 7th MLSys Conference, Santa Clara, 2024.
https://proceedings.mlsys.org/paper_files/paper/2024/file/5321b1dabcd2be188d796c21b733e8c7-Paper-Conference.pdf

The closest published proceedings statement of our lead's premise, for dense flat GEMM:

* "previous designs ... pad the matrix with zeros to perform GEMMs of larger sizes (*e.g.*, 64), leading to over
  50 % computation under-utilization"; their fix is "**only pad the matrix size to 8** rather than 64".
* The two-regime split by N: "**for the smaller N, the flat GEMM is parallelism-bounded**" (B_N ends up tied to the
  SM count — 108 on their A100), "for the larger N, the flat GEMM becomes memory-bounded" (double buffering,
  up to 52 %).
* The heuristic dataflow: only four [N,K] shapes exist per model; sweep M to find two inflection points per shape
  (M₁ where flat-GEMM beats GEMV, M₂ where CUTLASS beats flat-GEMM) and store a lookup table. Their measured points:
  at M=1 the tensor-core path achieves only **82.15 %** of a CUDA-core FastGEMV; at M=4 the CUDA-core path achieves
  only **49.75 %** of the tensor-core path. So the sign of the crossover flips inside M∈[1,4].

Transferable to us two ways: the *8-row* padding granularity, and the *method* (per-shape M sweep for inflection
points, not a single global choice). Their N-parallelism-bounded / memory-bounded dichotomy is also the correct
frame for §2.

### 1.3 Demystifying the MoE Serving Tax — MLSys 2026

Patel, Lee, Iwasaki, Krishnamurthy (UW + Meta), *Demystifying the Mixture of Experts Serving Tax*,
Proceedings of the 9th MLSys Conference, Bellevue, 2026.
https://proceedings.mlsys.org/paper_files/paper/2026/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf

Measured on 8×A100 (Mixtral, Qwen2-MoE) and 8×B200 (DeepSeek-V3), vLLM + DeepGEMM/DeepEP. Points that bear on F1:

* **Decode tax is bell-shaped**: "as low as 1.05×" at batch 1, peaking at "2.08× for Mixtral and 2.57× for Qwen"
  at batch 32, ~3× for DeepSeek at 128. "Under common operating points, the MoE decode latency closely matches, or
  even exceeds, that of the DensePA models" (DensePA = a dense FFN with the *total* parameter count). So at our
  operating point the MoE layer is expected to cost what a 14.5 GB-per-layer dense GEMM costs. Weight amplification,
  not FLOPs, is the whole bill.
* **Two production padding schemes, opposite worst cases**: "Blockwise padding independently rounds each expert's
  tokens to a block boundary (used by FusedMoE and DeepGEMM prefill). Max padding pads all experts to a single shared
  dimension equal to the maximum per-expert token count (used by DeepGEMM decode)." Blockwise is worst under *spread*
  routing, max padding worst under *concentrated* routing.
* **Skew helps decode**: "skewed routing can improve decode performance by activating fewer overall experts, since
  the savings from mitigating the dominant weight amplification tax and reducing the padding overheads outweigh the
  minor costs of imbalance." Table 1: "Non-uniform routing can activate fewer experts — Decode (**beneficial**)".
* Ancillary kernels (router, top-k, align, local sum) are "typically under 5 % for Mixtral and under 8 % for Qwen" of
  the MoE block; the *align* kernel is largest at small batch. (Our fused routing is on the closed list; nothing here
  reopens it.)

Implication for us: at ≤19 rows/expert we are in the *max-padding / small-batch* corner, which is the corner the paper
says is dominated by fixed cost per activated expert and is least sensitive to token imbalance. It also predicts our
cost scales with the number of *activated* local experts, not with rows — consistent with "heaviest-expert-first"
already being present.

### 1.4 DeepGEMM PR #473 — SM90 fused FP8×MXFP4 MegaMoE (the closest published analogue to F1)

deepseek-ai/DeepGEMM, PR #473, *Add optimized fused FP8 x MXFP4 MegaMoE for SM90*, opened 2026-10-14.
https://github.com/deepseek-ai/DeepGEMM/pull/473  (derived from #440, MXFP4 decoder from #411 / Humming)

Hardware: **H20-3e, 78 SMs, CUDA 13** — sm_90, so the scheduling transfers to H200, but see the SM-count caveat in
§2.4. Shape: hidden 4096 / intermediate 2048 / **32 experts per GPU** / top-k 6 (Flash) — i.e. *our* per-rank expert
count. Two mechanisms:

* **Small M — "tiled register decoding + RS WGMMA"**: "Pack each 256-row x K128 weight stage into a contiguous
  16 KiB tile and **decode directly into register A fragments**, avoiding expanded-weight shared-memory traffic and
  associated synchronization." Then "Extend #411's swapAB with **N8/16/24/32/40/48 buckets**, selected double
  buffering, and direct SwiGLU/quantization epilogues."
* **Optional single-token split-K**, and specialized small-M buckets.

Measured, EP8 Flash (µs, lower better): M=1 66.8 → 46.4 (**1.44×**); M=2 53.0 (1.27×); M=4 69.1 (1.34×);
M=8 78.8 (1.35×); M=16 **117.0** (1.36×); M=32 151.5 (1.34×); M=128 206.3 (1.34×); M=2048 410.7 (1.11×).
Single-GPU Flash: M=1 66.2 vs FP8 baseline 79.0 → the MXFP4 path **beats the FP8 path** at M=1 and stays ahead
through M=128 (247.6 vs 387.1).

Two things to take: **19 rows/expert lands inside their N8/N24 buckets**, so this is not an extrapolation; and the
gain is *largest exactly where we live* (1.27–1.44× for M≤16, decaying to 1.06–1.11× by M=4096).

### 1.5 Machete — Red Hat / Neural Magic (the swap argument, spelled out; and the counter-argument)

L. Wilkinson, *Introducing Machete, a mixed-input GEMM kernel optimized for NVIDIA Hopper GPUs*, Red Hat Developer,
2024-10-14. https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel

* The swap, stated as the reason it exists: "A key challenge with 'wgmma' is that for a matrix multiplication
  C = AB, only A and C can be sourced from registers, while B must be sourced from shared memory. Since we upconvert
  the 4-bit weights to 16-bit floating point values in registers, we can avoid restoring them into shared memory by
  computing **Y^T = W^T * X^T instead of Y = XW**. This ensures the weights are 'A' (left side input-operand) with
  respect to the 'wgmma' instructions, allowing them to be sourced directly from registers."
* The sub-byte operand load problem and its fix: "this instruction [ldmatrix] isn't available for 4-bit types ...
  we would naively need to resort to performing four 8-bit shared memory loads per tensor core operation." Hence
  **weight pre-shuffling**: interleave four tiles so one **128-bit** shared load replaces four 8-bit loads; upconvert
  by "shifting the nibbles into the lower four bits of their destination registers using simple bit shifts and
  masking operations and then expanding in-place" (`vllm_numeric_conversion.cuh`).
* **The counter-argument to the pure-mma.sync remedy**: "For Marlin, the poor performance on Hopper architecture
  primarily stems from the use of outdated 'mma' tensor core operations. To achieve peak FLOPs on NVIDIA Hopper, the
  new 'wgmma' instructions must be utilized. Using only 'mma' instructions results in a loss of approximately
  **37 % of peak compute throughput**." Machete's whole contribution is replacing an mma-based kernel (Marlin) with
  a wgmma-based one on Hopper, and it lists "*Optimizing low batch size performance (sub-32)*" as **future work** —
  i.e. they did not solve sub-32 with mma.sync; they solved it with swapAB + wgmma.

Regime note that reconciles the two: the 37 % applies to compute-bound shapes. F1 at ≤19 rows/expert is
memory-bound (28.31 MB of weights per expert streamed against ~19×4096 f16 activations), so raw MMA issue rate is
not the binding constraint — the *M=64 padding* is. Both statements can be true at once, and the swap resolves both.

### 1.6 FlashInfer — MLSys 2025 (the TC row minimum, stated in proceedings)

Ye et al., *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving*, MLSys 2025.
https://proceedings.mlsys.org/paper_files/paper/2025/file/dbf02b21d77409a2db30e56866a8ab3a-Paper-Conference.pdf

"**For query tile size 1, we use CUDA Cores template since tensor core instruction *m* (minimum rows) is 16**, and
use Tensor Cores for other query tile sizes. For FA3 we provides row tile sizes that are **multiples of 64**,
aligning with Hopper's WGMMA requirements." This is the proceedings statement of the two granularities in the table
in §1.1: 16 for warp-level MMA, 64 for wgmma. Below 16 rows, published practice abandons the tensor core entirely
(beats it, per FlashDecoding++ §1.2 at M=1).

### 1.7 What this means for the live lead (verdict)

**Corroborated**: the diagnosis. At 19 rows/expert a wgmma M=64 tile is 70 % padding; every source above agrees the
useful granularity is 8–16, and one (MLSys 2024) states the number 8 explicitly.

**Contradicted**: the remedy. **No non-arXiv proceedings, vendor doc or engineering post found in this pass claims
`mma.sync` beats `wgmma` at small M on Hopper.** The three published industrial answers to "4-bit weights, few rows,
sm_90" are all *swapAB + wgmma with the weights on the M side and register-resident A* — DeepGEMM #473 (MXFP4,
measured 1.27–1.44× at M≤128), Machete/CUTLASS w4a16 (int4), and FlashInfer's XQA sm_90 `SWAP_AB` macro (§2.3).
Our 4–29 % is therefore most likely either (a) measured against a build where the swap path was not available, or
(b) real but a *partial* recovery — in which case a swapAB RS-wgmma at N∈{8,16,24,32} would add on top of it, not
compete with it. This is the highest-value thing to test, and it is cheap to state as a test: hold everything fixed,
swap A/B and switch to RS-mode wgmma with N bucketed to the expert's row count, versus the block-M-8 `mma.sync`.

---

## 2. Axis 2 — short-grid paged attention (8–17 MB total KV)

### 2.1 The split←CTA-count relation, from proceedings

FlashInfer, MLSys 2025, **Algorithm 1** ("FlashInfer's balanced scheduling algorithm"),
https://proceedings.mlsys.org/paper_files/paper/2025/file/dbf02b21d77409a2db30e56866a8ab3a-Paper-Conference.pdf

```
cost(l_q, l_kv) = α·l_q + β·l_kv                                  (α, β hyperparameters)
L_kv ← ceil( Σ_i ceil(l_qo(i)/T_q) · l_kv(i)  /  #CTA )           (max KV chunk size)
split each query tile's KV into chunks ≤ L_kv; sort chunks desc; greedy min-heap assign to CTAs
```

Direct answers to the three sub-questions:

* **KV-split counts** are *derived from the CTA count*, not the reverse. Total work is divided by `#CTA` and that
  quotient is the chunk cap. So on a **short grid the chunks get bigger and the grid stays short** — the algorithm
  cannot manufacture parallelism it does not have.
* **Merge protocol**: partial outputs + LSE go to a user workspace; a separate **contraction** stage applies the
  associative ⊕ operator (FlashInfer §2.2, Eq. 1–2: `LSE(I∪J) = LSE(I) ⊕ LSE(J)`). Deliberately **not** Stream-K's
  atomics — "because LLM serving requires deterministic outputs, we did not incorporate atomic aggregation ... The
  scheduling algorithm generates deterministic aggregation order." The contraction and the attention stage are
  **merged into one persistent kernel** ("eliminating intra-kernel overhead"), with a fixed grid size and fixed
  workspace offsets so it is CUDAGraph-capturable. The **plan is CPU-side, per generation step, cacheable and reused
  across all layers** (62 for us).
* Chunk assignment is **greedy LPT** (longest-first into the least-loaded CTA), not fixed-split.

Reported effect: on H100 decode, bandwidth utilization up to 73 % across constant / uniform / skewed length
distributions, beating FlashAttention in the uniform and skewed cases "because of our load-balanced dynamic
scheduler" and because "FlashAttention use suboptimal tile size for decoding."

### 2.2 LeanAttention — MLSys 2025 (the wave-quantization fix, with numbers)

Sanovar, Bharadwaj, St. Amant, Rühle, Rajmohan (Microsoft), *LeanAttention: Hardware-Aware Scalable Attention
Mechanism for the Decode-Phase of Transformers*, MLSys 2025.
https://proceedings.mlsys.org/paper_files/paper/2025/file/16ec6494e9b5a4138de7238761d715b4-Paper-Conference.pdf

The paper whose subject *is* our failure mode:

* "at any given point in time, the number of CTAs in flight on the GPU is directly proportional to the number of
  query tiles, and, therefore, to the query length." For a 128-head model at decode, "an 8 GPU A100 system that has
  864 compute cores" is severely under-utilized. **Our case is worse**: 2 kv heads per rank × small batch, so query
  tiles are ~2.
* Fixed-split (FlashDecoding, FlashInfer) "often results in partially full waves ... suffer from quantization
  inefficiencies"; "increasing the number of splits could help occupy the GPU better, [but] it would result in
  reduction overheads that scale with the split factor and would allocate minimal work to each SM."
* **The recipe, with the constants we need**: stream-K over KV. `LeanTile` is "the smallest optimal granularity":
  "a tile size granularity of **256** and 128 tokens along the N_k dimension to be the most optimal for a
  **head size of 64** and 128 respectively for FP16→32 problems" (measured on A100). We are head_dim 64 → **256**.
* `TilesPerCTA = (BatchSize × NumHeads × ContextLen) / (TileSize × NumSMs × MaxCTAsPerSM)`, and the grid is fixed at
  `G = NumSMs × MaxCTAsPerSM` (their A100 example: 108 × 2 = 216, two CTAs co-resident per SM by SMEM). The grid size
  is a *fixed* constant; only the tile→CTA mapping changes with length.
* **Merge protocol**: softmax re-scaling as an associative reduction (proof given for unequal block sizes). The CTA
  that computes an output tile's *first* LeanTile is the **"host block"**; every other contributing CTA does
  `StorePartials(O,m,l)` then `Signal(flags[g])`; the host `Wait`s on each and folds in with
  `m_new = max(...)`, `l = e^{...}l_cta + e^{...}l`, `O = e^{...}O_cta + e^{...}O`; the finishing CTA writes
  `O = diag(l)^{-1}O` and `L = m + log(l)` directly from registers with no reduction at all. **Single kernel launch**
  — the separate combine launch that FlashAttention-3/SGLang incur is deleted.
  Per-partial storage is negligible at decode ("the output tensors are of dimensions 1× head dim, where head dim is
  typically in the range of 64 to 256").
* Generality: "generalizes to FlashAttention-2 when the number of output tiles is equal to grid size, and generalizes
  to FlashDecoding when grid size is an even multiple of number of output tiles."
* Result: **1.73× average** over FlashDecoding, up to **2.18×** at 256 k context.

### 2.3 The empirical failure mode, measured on sm_90 — FlashInfer issue #5998

flashinfer-ai/flashinfer issue #5998, *[Performance] XQA sm90 decode (head_dim 256, fp8 KV, page 16), batch 10 /
124K ctx: 18% of HBM peak, 66-block grid, 12.5% warp occupancy — 65% of the decode step* (opened 2026-10-03,
6 comments). https://github.com/flashinfer-ai/flashinfer/issues/5998

Not a paper, but it is the only public CUPTI-level diagnostic of an sm_90 decode kernel at low CTA count, and it is
pinned to a shipped build (vLLM 0.30.0 / flashinfer 0.6.18.post1, CUDA 13, 2×H20, 78 SMs). Findings:

* `kernel_mha` runs at **721 GB/s = 18.0 % of 4.0 TB/s**, against `deep_gemm` FP8 GEMM at **1,299 GB/s = 32.5 %**
  *in the same step on the same device*. Fitted marginal KV read bandwidth: **≈758 GB/s**.
* The launch is `grid (split, KV heads, batch) = (3, 2, 11) = 66 blocks for 78 SMs`, block `(128,1,2)` = 256 threads,
  **129.6 KiB SMEM**, 248 registers → **1 block/SM, ~12.5 % warp occupancy**. 124 K ctx / 3 splits ⇒ each block
  serially covers ~41 K tokens of KV.
* The split is a *fixed heuristic on the SM count*: maintainer reply — "`launchMHAFlashInfer` ... uses block
  `(128, 1, 2)` and grid `(split, KV heads, batch)`. Its split heuristic gives `floor(78 / (11 x 2)) = 3`."
  `XQA_NB_SUB_SEQ=12/24` does nothing (the launcher never reads it). `multiBlockMinNbTilesPerCta` still carries a
  `// @fixme: need tuning` in `mha_sm90.cu`.
* **page_size is not a lever**: grids, block, register count and SMEM are "byte-identical for page 16/32/64", and
  per-call duration moves 1552/1574/1588 µs (±1.2 %). (Their own follow-up correction: XQA silently caps page_size
  at 64 on sm_90, so their `--block-size` A/B was null by construction.)
* **The Hopper-specific FP8 launcher is 2.13× faster per call than the generic one**: `(1,2,32)` block `(128,1,3)`
  at 753.9 µs vs `(3,2,11)` block `(128,1,2)` at 1574 µs. With spec-decode off, *every* call takes the fast path;
  with MTP on, the multi-token verification pass forces the generic launcher.
* XQA > FA3: "swapping in `FLASH_ATTN` (FA3) made the *whole step* 1.8× slower (82.7 → 45.6 ms/step), i.e. XQA is
  already the better of the two available paths." Triton `unified_attention` is 3.2× slower again.
* The sm_90 kernel's swap is a compile-time macro: `#ifdef SPEC_Q_SEQ_LEN ... #define SWAP_AB 1 #else #define SWAP_AB
  (!SPEC_DEC) #endif` (`csrc/xqa/mha_sm90.cu:44-51`), with `static_assert(specDecQLen * headGrpSize <= 32)`. So the
  sm_90 XQA fast path is itself a **swapAB kernel gated on a ≤32-element token-side tile** — the same 8-granularity
  argument as §1.4.

**Why this is our case.** Our sliding cells run 1.07–1.67 TB/s against 3.65 TB/s for our own flat cells. The public
analogue is a 1-CTA/SM, 129.6 KiB-SMEM, long-serial-chain kernel sitting at 18 % of peak. Two consequences:
(a) with **50 MB of L2** on H100/H200 (Hot Chips 34, P. Choquette, *Hopper H100 Tensor Core GPU*, slide "Larger 50MB
L2", http://hc34.hotchips.org/assets/program/conference/day1/GPU%20HPC/HC2022.NVIDIA.Choquette.vfinal01.pdf) an
8–17 MB working set is **L2-resident**, so 1.07–1.67 TB/s is not an HBM-capacity effect — it is an
occupancy/latency effect; (b) 3.65 TB/s is *also* below the 4.8 TB/s HBM3e peak of the H200
(https://www.nvidia.com/en-us/data-center/h200/, "141GB HBM3e at 4.8 TB/s"), so the flat path is not at roofline
either — it is merely much closer.

### 2.4 The SM-count portability trap

Both published heuristics are keyed to `NumSMs`: XQA's `floor(SMs/(batch × kv_heads))` (78 on H20) and
LeanAttention's `G = NumSMs × MaxCTAsPerSM` (108 on A100, and they also benchmark H100's 132). **H200 has 132 SMs.**
Any constant lifted from the H20 or A100 papers must be rescaled, and any *our-side* measurement of a split factor
must be re-derived against 132. This is the single easiest way to import a paper's number and get a wrong answer.

### 2.5 MLSys 2026 attention scan — nothing on short-grid decode scheduling

FlexiCache (temporal stability of heads → KV selection), OPKV (recallable sparsity plugin for paged KV), Kitty
(2-bit KV quantization with channel-wise boost), MAC-Attention (match-amend-complete), and *A Pragmatic Exploration
of Prefill-Decode Disaggregation* are all either KV-**selection/quantization** or serving-phase scheduling. None
addresses CTA residency or merge at a fixed small KV budget.

---

## 3. Axis 3 — NVFP4/E2M1 on Hopper (`sm_90a`, not Blackwell)

### 3.1 PTX ISA 9.4 — the instructions do not exist on sm_90

Primary source, read directly: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

| instruction / type | architectures | PTX version |
|---|---|---|
| `cvt.rn.satfinite{.relu}.{e2m1x2/e2m3x2/e3m2x2/ue8m0x2}.f32` | **sm_100a, sm_101a(→110a), sm_120a**; sm_100f/110f/120f families from 8.8 | 8.6 |
| `cvt.rn{.relu}.f16x2.{e2m1x2/e2m3x2/e3m2x2}` | same set (100a/110a/120a, 100f+) | 8.6 |
| `cvt.rs{.e2m1x4/.e4m3x4/.e5m2x4/.e3m2x4/.e2m3x4}.f32` (+ `.rs` rounding) | **sm_100a, sm_103a, sm_107a** | 8.7 |
| `add/sub/mul/fma` with `.e2m1x4`, **`.e2m1p4x4`**, `.ue8m0x4`, … | **sm_100a, sm_103a** only | 9.4 |
| `cvt` with `.e4m3x2/.e5m2x2` (contrast) | **sm_89 or higher** | 7.8 |

So: `cvt.rn.satfinite.e2m1x2.f32` — the conversion a dequant sequence would want — **is unavailable on sm_90a**,
as is every packed-FP4 arithmetic form including `mul.satfinite.e2m1p4x4`. FP8 conversion *is* available on Hopper;
FP4 is not. There is no `.target sm_90a` variant to reach for.

### 3.2 Transformer Engine — vendor statement that NVFP4 is a Blackwell datatype

NVIDIA Transformer Engine 2.9.0, *Using FP8 and FP4 with Transformer Engine*:
https://archive.docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.9/user-guide/examples/fp8_primer.html

> "H100 GPU introduced support for a new datatype, FP8 ... **Blackwell added support for NVFP4 and MXFP8 datatypes**."
> "NVIDIA **Blackwell** introduced support for an even smaller, 4-bit format called NVFP4. The values are represented
> there in E2M1 format, able to represent values of magnitude up to +/-6."

Same page gives the exact NVFP4 block contract we have to reproduce in software: **one E4M3 scale per 16 elements**
(MXFP8 is one E8M0 per 32), plus "an additional single per-tensor FP32 scaling factor to avoid overflows". That is a
three-level scale chain: per-tensor fp32 → per-16-block E4M3 → E2M1 value.

CUTLASS confirms the same split by example numbering: all NVFP4 GEMM examples are Blackwell —
`72a_blackwell_nvfp4_bf16_gemm.cu`, `72b_blackwell_nvfp4_nvfp4_gemm.cu`, `79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu`
(SM100 / SM120); there is no sm_90 NVFP4 GEMM example.
https://github.com/NVIDIA/cutlass/tree/main/examples/72_blackwell_narrow_precision_gemm

### 3.3 The published sm_90 dequant sequences (there are exactly two, and they agree)

**A. Dequantize into register A fragments, then RS-mode wgmma** — DeepGEMM #473:
"Pack each 256-row x K128 weight stage into a contiguous **16 KiB** tile and decode directly into **register A
fragments**, avoiding expanded-weight shared-memory traffic and associated synchronization." Because the weights are
the A operand of a swapped problem, they never have to be written back to SMEM. Their MXFP4 scale chain is
"preserve[d] ... for bitwise repeatability" across split-M tiles.

**B. Pre-shuffle + shift/mask expand into fp16 registers** — Machete. Sub-byte operands have no `ldmatrix`, so
pre-shuffle four tiles in *global* memory (free — it happens at load time) so that a single **128-bit** shared load
delivers four MMA tiles' worth; then "shifting the nibbles into the lower four bits of their destination registers
using simple bit shifts and masking operations and then expanding in-place". Code: `vllm_numeric_conversion.cuh`,
`cutlass_extensions/`.

Both are *software* decode. Neither uses, or can use, a hardware E2M1 converter on sm_90.

### 3.4 Scale-application placement (the actionable part)

Because the sm_90 path must materialize values in registers anyway, the E4M3-per-16-block scale should be folded
**into the dequantized A fragment before the MMA** rather than applied post-MMA — at ≤19 rows the accumulator tile is
tiny and the post-MMA rescale is pure epilogue cost, while the fold is free inside the register decode. Evidence
that this is the right split: DeepGEMM #473's small-M path decodes to registers and keeps the scale chain explicit;
their SM90 API also requires **FP32 scaling factors** whereas SM100 requires packed UE8M0
(DeepGEMM README, "the data format for the scaling factor of SM90 and SM100 is different: SM90 requires scaling
factors in FP32 ... SM100 requires packed UE8M0") — i.e. on Hopper the scale is a runtime float multiply, not an
instruction field. https://github.com/deepseek-ai/DeepGEMM#interfaces

One caution: with a per-16 E4M3 block scale and our 4096-deep K, a K128 stage spans **8 block scales**; folding them
into A is 8 multiplies per 128 values per 19 rows — negligible — but it means the A fragment cannot be a single
contiguous 16 KiB copy of packed nibbles; it must be *decoded* per block. That is exactly why #473 says "decode
directly into register fragments".

### 3.5 Where axis 3 returns nothing

No GTC session, Hot Chips talk, CUTLASS example, Colfax post, or engineering blog found in this pass implements
**NVFP4** (E4M3-per-16) *compute* on sm_90. The nearest things are MXFP4 (E8M0-per-32, DeepGEMM #473) and int4
w4a16 (Machete), and both translate on load. One third-party blog claims vLLM 0.32 enables "NVFP4 KV cache" on
"SM8x (Ampere…)" — that is a *storage* format with dequant on read, not FP4 compute, and the claim is unverified
against vLLM release notes: https://www.spheron.network/blog/vllm-nvfp4-kv-cache-in-0-32-more-concurrent-requests-per-gpu
(low authority; recorded, not relied on).

---

## 4. Axis 4 — lossless weight re-encoding at 10–12 bits

### 4.1 NyoomFloat12 — ICLR 2026 (SPOT workshop): 12 bits, and its own roofline

Liberman, Zhang, Fu, *NyoomFloat12: Lossless 12-bit Weight Compression for Post-Training Inference*,
ICLR 2026 Workshop on Scaling Post-training for LLMs (SPOT), poster.
https://iclr.cc/virtual/2026/10013421 · OpenReview https://openreview.net/forum?id=yYFj1E1mSZ

* Mechanism: "By analyzing per-bit entropy in trained weight exponents, we find that **the upper exponent nibble is
  nearly constant (0111)**; dropping it yields a fixed-length format with **13-operation SIMT-friendly decode**."
* Exactness: "escape groups are zeroed during bulk decode and overwritten with the stored originals, producing
  **bitwise-exact BF16 output**."
* **The number that matters**: "fusing decode into GEMV achieves up to **1.28×** speedup, **approaching the 1.33×
  theoretical maximum**." 1.28/1.33 = **96 % of the byte-reduction roofline**, in the memory-bound regime.
* Capacity side: 25 % weight-footprint reduction doubles max batch on Qwen3-32B/SGLang → 64 rollouts 1.5× faster.

### 4.2 vLLM RFC #63626 — independent convergence on the identical format

vllm-project/vllm issue #63626, *[RFC]: Lossless 12-bit storage for BF16 dense weights to speed up memory-bound
decode*, opened 2026-10-26. https://github.com/vllm-project/vllm/issues/63626

Same design, derived separately, with the engineering details our scheme probably already has:

* 8-bit `sm` = sign + 7 mantissa bits unchanged; **4-bit exponent index** = exponent minus a per-tensor base over a
  15-wide window; **"Index 15 is an escape that decodes to +0, because padded rows are all zeros"**; out-of-window
  values (0.02–0.11 %) stored **row-wise as CSR exceptions** and added back exactly. 12 bits = 75 % of BF16.
* Guard: "A tensor stays in BF16 if any 32-row block holds more than 2048 exceptions. This keeps the exception path
  from turning serial." Eligibility: ≥15M elements and K a multiple of 256.
* **The failure they hit — directly relevant to a palette+escape scheme**: "our first version encoded zeros as
  exceptions. The 12 zero padding rows of one fused matrix then put **122,880 exceptions into a single row block,
  and decode ran about 20× slower**." The +0 escape, vectorized exception handling, per-block guard and size filter
  all came from that.
* Measured (GB10, 273 GB/s, sm_121, M=4 GEMV): 1.24× (linear-attn in_proj 16384×2560), 1.27× (LM head
  248320×2560), 1.24× (out_proj), 1.23× (attention qkv). End-to-end +11 % decode steps/s.
* **Exactness caveat worth carrying**: "The stored weights are bit-exact. The decode GEMV results differ from the
  BF16 path only in summation order: they are within about 1 BF16 ulp of an fp32 reference." Lossless *storage* is
  not bit-exact *output* unless accumulation order is preserved. If our bf16 board's contract is bit-exactness
  against a reference, an in-register decode that changes summation order is a risk independent of the codec.
* Explicitly positions itself against DFloat11: "trades about one bit per value for a fixed-length layout that the
  GEMV can decode in registers."

### 4.3 What is left below 12 bits, and what it costs

* **DFloat11** — Zhang et al., *70 % Size, 100 % Accuracy: Lossless LLM Compression for Efficient GPU Inference via
  Dynamic-Length Float*, **NeurIPS 2025** (proceedings).
  https://proceedings.neurips.cc/paper_files/paper/2025/file/8e8b4449c80879215ff2ee47b6dd01a6-Paper-Conference.pdf
  70 % of BF16 ≈ **11.2 bits/value**, via variable-length exponent entropy coding. But it "decompresses to BF16 on
  the GPU" as a **separate pass** — it does not fuse into the GEMV, which is precisely the axis on which
  NyoomFloat12/RFC chose 12 bits. Lower bytes, higher kernel overhead.
* **The floor**: the exponent's own entropy limit is ~2.67 bits (Yang et al., *To Compress or Not? … Exponent
  Concentration*, arXiv:2510.02676 — cited only as a bound; the fixed sign+mantissa 8 bits are not compressible).
  With sign+mantissa fixed at 8, the entire remaining design space at 12 bits is 4 bits for window-index + escape;
  getting to 11 needs a **variable-length** exponent code, which is exactly the DFloat11 trade.

### 4.4 Verdict on axis 4

**Nothing beats a 12-bit palette + escape scheme at 12 bits** — because 12 bits *is* the scheme, and it is already at
96 % of its roofline, independently replicated by two groups (one academic workshop, one live vLLM RFC with the same
"+0 escape" trick). The 1.33× ceiling is arithmetic: 12/16. **Do not spend more on the 12-bit axis.** The only
strictly-better ratio is ~11.2 bits via DFloat11's variable-length exponent, and it costs a decompression pass; that
trade is worth it only if the dense path is bandwidth-bound end-to-end *and* has spare compute, which on a decode
step with ≤19 rows/expert it may actually have — but it is a different project, not a tweak.

Not found: any SC / PPoPP / ISCA / MICRO / HPCA material on lossless sub-16-bit weight re-encoding. (Cloudflare's
Unweight was already covered in `research_02`.)

---

## 5. Ranked by expected value

### #1 — F1 (MoE): swapAB + RS-mode wgmma with N bucketed to 8, dequantizing NVFP4 straight into register A

Evidence: DeepGEMM #473, measured on sm_90/78-SM at our expert count and top-k — 1.44× at M=1, 1.36× at M=16,
1.34× at M=32–128 vs their own MXFP4 baseline, and *beating the FP8 baseline* at M=1 (66.2 vs 79.0 µs), with 19
rows/expert inside their N8/N24 buckets. Mechanism spelled out by Machete (wgmma B must be SMEM, A may be RMEM →
compute Yᵀ=WᵀXᵀ so weights are A). Same swap already shipped in FlashInfer's sm_90 XQA (`SWAP_AB`, gated ≤32).
Contradicts the mma.sync *remedy* while corroborating its *diagnosis*: no source found claims mma.sync beats wgmma at
small M; Machete puts pure-mma at −37 % of peak and lists sub-32 as unsolved-in-mma. **Why #1**: MoE is 101 ranks
per +1 %, we sit at 2.3746, and every source says the win is concentrated at M≤32 — the largest, best-measured,
most directly transferable gain in this pass.

### #2 — F2 (attention): LeanTile = 256 @ head_dim 64, stream-K tile→CTA mapping with the host-block/flag merge

Evidence: LeanAttention (MLSys 2025) gives the constants for *our* head_dim (256 tokens at head size 64), the mapping
formula `TilesPerCTA = (Batch × Heads × CtxLen)/(TileSize × NumSMs × MaxCTAsPerSM)` with a **fixed** grid of
`NumSMs × MaxCTAsPerSM`, the associative softmax-rescaling merge with an explicit proof for *unequal* block sizes,
the host-block/non-host-CTA `StorePartials`+`Signal`+`Wait` protocol, and one kernel launch instead of two; 1.73×
average over FlashDecoding, 2.18× at 256 k. Corroborated on the split/merge side by FlashInfer Algorithm 1
(split derived from `#CTA`, greedy-LPT assignment, deterministic contraction for CUDAGraph). Diagnosis pinned by
FlashInfer #5998: an sm_90 decode kernel at 66 blocks/78 SMs, 1 CTA/SM, 129.6 KiB SMEM, 12.5 % warp occupancy,
18 % of HBM peak, with page_size provably not a lever (±1.2 % across 16/32/64) and the Hopper-specific launcher
2.13× faster per call than the generic one. **Why #2**: it attacks the 2.2–3.4× sliding-vs-flat gap directly, and
unlike #1 it needs no new math — only a scheduling change. Second because it must be re-derived against 132 SMs
(§2.4) rather than 108, so one more measurement round is unavoidable.

### #3 — F1 (MoE): fold the per-16 E4M3 block scale into the register-resident A fragment instead of the epilogue

Evidence: PTX ISA 9.4 shows there is no hardware E2M1 or block-scale path on sm_90 (§3.1) and Transformer Engine
confirms NVFP4 the datatype is Blackwell's (§3.2), so the dequant is necessarily software and the scale is
necessarily a runtime multiply — DeepGEMM's own API contrast makes this explicit (SM90 takes **FP32** scaling
factors, SM100 takes packed UE8M0). Given a register decode is already required, pre-MMA folding costs 8 multiplies
per 128 values per row; post-MMA rescaling costs an epilogue pass over the accumulator. This is a small, certain,
independent increment that rides on #1 rather than competing with it. **Why #3, not higher**: it is a refinement of
#1's implementation, not a separate mechanism.

### Deliberately not ranked

* **12-bit lossless codec (F3)** — evidence says stop (§4.4): already at 96 % of the 1.33× roofline, replicated twice.
  Expected value ≈ 0, and any exploration spend is negative.
* **Split-K / more KV splits** — corroborated by three sources (FlashInfer Alg. 1, LeanAttention §3.3, vLLM RFC
  #63626 "the GEMV splits along K when a shape gives too few row blocks"), but 4-CTA K-splits are on our closed list.
  Recorded as *confirmation that the closed axis was right*, not as a proposal.
* **Dynamic per-CTA device-side TMA descriptors for expert weights** (PyTorch, *Accelerating MoE's with a Triton
  Persistent Cache-Aware Grouped GEMM Kernel*, 2025-08-19,
  https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/) — a real
  mechanism (`tl.extra.cuda.experimental_device_tensormap_create2d` + `fence.proxy` per routed expert, plus grouped
  launch ordering for +60 % L2 hit / 1.33×). But the post is explicit that it targets "training and prefill workloads
  ... the matrix multiplication problem sizes are large", uses `grid = NUM_SMS` = 132, and defers MXFP4 to Blackwell.
  **Returns nothing for tiny-M decode**; research_01 already records host-built descriptors in a competitor.

---

## 6. Per-target ledger (what each venue returned)

| target | result |
|---|---|
| **MLSys 2024** | FlashDecoding++ — flat GEMM, pad-to-8, M-inflection heuristics. **Directly usable** (§1.2) |
| **MLSys 2025** | FlashInfer (Alg. 1 split/merge) + LeanAttention (LeanTile=256, stream-K, host-block merge). **The two best axis-2 sources** (§2.1–2.2). Also QMoE, Comet, Context Parallelism, NEO, MiLo — not applicable |
| **MLSys 2026** | *Demystifying the MoE Serving Tax* (weight amplification, two padding schemes, skew-helps-decode) (§1.3). Attention papers are all KV-selection/quantization — **nothing on short-grid decode scheduling** |
| **ASPLOS 2025/26** | POD-Attention (hybrid-batch prefill/decode overlap), vAttention — both about *batching/KV management*, **nothing on CTA residency at a fixed small KV budget** |
| **ISCA 2025** | WindServe (phase-disaggregated serving, stream-based dynamic scheduling). Serving-side only; **nothing at kernel granularity** |
| **HPCA / MICRO 2024–25** | **Nothing found** for MoE-at-tiny-M or short-grid decode. No proceedings item met the filter |
| **SC 2024/25** | HyTiS, *Hybrid Tile Scheduling for GPU GEMM with Enhanced Wave Utilization and Cache Locality*, SC'25 pp. 1604–1618, DOI **10.1145/3712285.3759771** — title is exactly our axis, but ACM returned HTTP 403 on both `/abs` and `/full`; **abstract not readable, cannot be priced**. KAMI (within-GPU communication-avoiding GEMM) — not applicable |
| **PPoPP 2025** | Only *POSTER: Triangle Counting on Tensor Cores* surfaced; **nothing applicable** |
| **CGO / PACT 2024–26** | **Nothing found** |
| **NSDI / OSDI** | Phase-level serving scheduling only (disaggregation, batching); **nothing at kernel granularity** — out of scope for these three axes |
| **ICLR 2026 (workshop)** | NyoomFloat12 — the axis-4 answer (§4.1) |
| **NeurIPS 2025** | DFloat11 — 11.2 bits, separate decompress pass (§4.3) |
| **Hot Chips 34** | Hopper deck: 80 GB HBM3, **50 MB L2**, TMA section. Used for the L2-residency argument (§2.3) |
| **GTC** | No GTC talk found that adds anything beyond the PTX/CUTLASS/Colfax material already cited. **Nothing new** |
| **Colfax** | WGMMA tutorial — "M is always 64", "bM a multiple of 64", B-must-be-SMEM (§1.1). Also their stream-K/persistent-kernel post, cited by both PyTorch and FlashInfer as the canonical wave-quantization reference |
| **PTX ISA 9.4 / CUTLASS / Transformer Engine docs** | The decisive axis-3 evidence (§3.1–3.2) |
| **DeepGEMM (non-arXiv)** | PR #473 — the single most transferable F1 result (§1.4) |
| **Chinese-ecosystem** | DeepSeek/DeepGEMM is the only one that published sm_90-specific FP4 code. No non-arXiv Moonshot / Zhipu / Alibaba-Pai / ByteDance-Seed post found adding to these axes; the ByteDance W4A8 Hopper MoE work was already captured as sglang PR #7772 in `research_02` |

### Reproducibility notes

* All PTX architecture gates above were read from the live 9.4 document, not from memory; the grep-able strings are
  `is supported on the following architectures:` immediately after each instruction's *Target ISA Notes*.
* DeepGEMM #473's measurements are on **H20 (78 SMs)**, not H200 (132 SMs). Scale every CTA-count heuristic.
* FlashInfer #5998 is a user-filed issue with maintainer cross-examination in-thread; two of the reporter's own
  readings were corrected mid-thread (the effective page_size is 64, not 16; and the `--block-size` sweep was null by
  construction). The occupancy and bandwidth figures survive the corrections; the *split-factor* question is still
  open upstream.
* HyTiS (SC'25) is the one identified item I could not read. If someone has ACM access, it is the only proceedings
  paper whose title promises wave-utilization work on exactly our tile-count problem.
