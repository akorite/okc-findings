# fd_r2_leaders — reverse-the-leaders, mechanism deltas vs `vendored_work/` (2026-11-04)

Per entry: user, live score (okc leaderboard --json snapshot ~06:15 UTC), inferred kernel family, whether mechanism already in `src/okc_loop/vendored_work/` (have/partial/no/unk). Zero top-10 team publishes leader code — all "family" labels are inference.

## muse_decode_layer (ours 2.2837 #111; top-10 = 2.4656–2.5401)

|#|user|speedup|family|have?|
|---|---|---|---|---|
|1|gauravjain14|2.5401|composite: near-floor WGS MoE + FP8 paged attn + tuned dense + minimal glue (MoE −8–13%, attn −21%, dense −6.5%)|partial|
|2|AlexCuadron|2.5226|same composite (OpenHands lineage)|partial|
|3|dolanzhao|2.5179|composite + profiling-driven per-stage tuning|partial|
|4|caoshiyi|2.5021|agent-evolved search (EAT/eat-kernels)|partial|
|5|gogowhy|2.4972|composite; private lineage|unk|
|6|tonybolivar|2.4854|decode grinder (81 att)|partial|
|7|tjiang-fair|2.4844|shared-cohort composite|unk|
|8|lxe|2.4708|hand-CUDA decode|partial|
|9|Charbel199|2.4688|imported design, ≤11 att|unk|
|10|mpeng19|2.4656|ThunderKittens composite|partial|

Missing: (a) MoE inside layer −8–13% (expert-sync schedule order = fd_kernel_sources.md lever 1); (b) attn cell −21% (128B whole-row FP8 gathers ~3.87 TB/s vs ours 3.4); (c) swept cuBLASLt dense plans; (d) deeper glue elimination.

## muse_moe_w4a16 (ours 2.3806 #260; top-10 = 2.6613–2.7642)

Top-10 all one family: zero-DRAM-intermediate NVFP4 WGS grouped GEMV/GEMM + fused router + per-shape tiles + PDL + FMA-pipe dequant. Missing: **expert-synchronous unit→CTA interleave** (DeepGEMM `scheduler/gemm.cuh:186` vs our contiguous `wgs_range` `moe_dup.py:2365-2369`) +0.3–1.7%. Rest is per-load stream-rate tuning (we ~84–88% of 4.35 TB/s floor vs ~97%).

## muse_attention_decode (ours 2.6590 #149; top-10 = 2.9969–3.3389)

Family: FP8 paged kernel at random-gather ceiling ~4 TB/s + µs-latency sliding flat + one-L2-round-trip split merge. Missing: (a) 3.4→4.0 TB/s head-packed whole-row loads; (b) sliding constant 5.0→2–3 µs (FINWARM shipped, rest is launch-chain/page-table restructure); (c) global-chunk equal-work boundaries on sliding batched cells (FlashMLA idiom, FLAT exists `dual.py:2602` unused for sliding).

## muse_bf16_proj (ours 1.5891 #80; top-10 = 1.6460–1.6940)

Family: per-shape ~1.5 µs C12-lossless/FP32 floors + 12-bit compressed weights beating nvjet at m128. Have: C12 codec ✅, `_OKC_LT_PINS` pattern ✅ (verify-32 only). Missing: non-heuristic cuBLASLt pins spanning all 30 cells incl. m128 compressed-domain consumer (partial).

## muse_prefill_layer (ours 1.9087 #99; top-10 = 2.0195–2.0754)

Family: turn-MoE near floor + dense ≥800 TF + attention ~1.4× FA3 via `ex2.approx.f16x2` softmax + epilogue-folded glue. Missing: **`ex2.approx.f16x2` softmax** — only mechanism named by source no vendored file ships.

## Delta list (ranked)
1. Expert-synchronous schedule order (moe + decode layer): DeepGEMM unit→CTA interleave +0.3–1.7% moe.
2. Full-shape attn bandwidth 3.4→~4 TB/s (head-packed 128B whole-row).
3. `ex2.approx.f16x2` softmax (prefill) ~1.4× FA3 ceiling — missing entirely.
4. Per-cell cuBLASLt pin coverage (bf16 + decode dense), all scored cells vs verify-32 only.
5. Sliding-shape latency floor 5→2–3 µs.
6. Global-chunk equal-work split boundaries on sliding.

All top-10 = same vjeux/keell0renz family we vendored; leaders differ in execution depth, not mechanism.
