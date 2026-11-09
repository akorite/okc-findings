# Ten-agent web research roster — 2026-11-03

Constraint: the objective caps concurrency at 3 subagents, so these run in waves of 3 (never more than
3 alive). Each agent writes its own report to `artifacts/reports/` and returns <=3 lines. Every claim must
carry a URL. "Nothing new" is an acceptable and useful conclusion.

| # | slug | scope | report |
|---|---|---|---|
| 1 | gh-code | GitHub **code** search for the OKC problem tags and every repo containing `muse_decode_layer`/`muse_moe_w4a16`/`moe_wna16_marlin`/`flash_attn_with_kvcache` content beyond the 11 teams we know | `research_01_gh_code.md` |
| 2 | analyzers | GPU **analyzer/tooling** repos **and the general kernel-repo sweep the operator asked for** (CUTLASS, flash-attention forks, Triton kernels collections, Hopper example repos, `ATED`-style, `Kitchen`-style, KernelBench/KernelWiki, `awesome-cuda` lists, Chinese-ecosystem kernel repos, GTC sample repos): ncu/nsys parsers, Nsight workflows, KernelBench/KernelWiki/TritonBench/gpu-mode tooling, `cuda-best-practices`, SASS-viewer utilities | `research_02_analyzers.md` |
| 3 | venues | **Non-arXiv papers**: MLSys / ASPLOS / ISCA / HPCA / MICRO / SC / PPoPP / CGO 2024-26 on Hopper TMA, wgmma, warp specialization, persistent kernels, MoE decode | `research_03_venues.md` |
| 4 | engines | **Serving-engine internals**: SGLang, vLLM, TensorRT-LLM, FlashInfer source for the three baseline kernels Muse Spark times (`moe_wna16_marlin_gemm`, `flash_attn_with_kvcache`, `bf16_gemm_dispatch`) | `research_04_engines.md` |
| 5 | leaderboards | **GPU-MODE / KernelBot / LeetGPU / Popcorn / MLSys26 FlashInfer** top submissions + any public writeups or Discord lecture notes | `research_05_leaderboards.md` |
| 6 | microarch | **Hopper microarchitecture, non-arXiv**: L2 partitioning/uGPU locality, HBM3e finite-transfer effects, clock/memory-controller behaviour (GTC, Hot Chips, Colfax, Modulus, Dissecting papers as presented) | `research_06_microarch.md` |
| 7 | nvfp4 | **NVFP4/E2M1 specifics**: dequant sequences, block-scale layouts, CUTLASS NVFP4 docs+code, `cvt`-based decode, scale-compression schemes | `research_07_nvfp4.md` |
| 8 | xqa | **Paged decode attention on Hopper**: FlashInfer, FlashMLA, TRT-LLM XQA, chunked-decode kernels — what an XQA-style kernel does that FA3 does not, at head_dim 64, GQA 8:1, window 2048 | `research_08_xqa.md` |
| 9 | autotune | **Kernel autotuning methodology**: Triton autotune, CUTLASS profiler, KernelBench-style search, "ceiling-first" practice, A/A controls, noise handling | `research_09_autotune.md` |
| 10 | harness | **Judge/timing-harness reverse engineering**: CUDA-graph capture semantics, L2-flush effects on instruction fetch, event timing, clock locking, cross-node robustness | `research_10_harness.md` |

Standing context for every agent: one H200, `muse_decode_layer` 2.28370 (#84/1221, the overall rank, top-20
~2.49), MoE 2.3746 (#246/1346, **101 ranks per +1%**), attention 2.6590 (#126/1289), bf16 1.5891
(#74/1672), prefill 1.9087 (#85/831). Architecture: head_dim **64**, 16 q / 2 kv heads (GQA 8:1),
sliding window **2048**, RoPE theta 5e5, paged FP8 KV with page_size 1 scored, 62 layers weighted
**46 sliding : 16 full**, 256-expert NVFP4 MoE with 32 local experts (~28.31 MB each), D_HIDDEN 8192,
D_EXPERT 4096, top-8.
Closed axes (do not re-propose): deeper MoE rings, megakernel, fused-slice MoE, L2 persistence windows,
speculative expert prefetch, dual-lane wgmma, per-unit barriers, 4-CTA K-splits, BF16 codecs other than
12-bit, i-cache pre-warm (+0.16% for us), TMA multicast (not expressible), dense epilogue deletion (+1%),
heaviest-expert-first (already present).
