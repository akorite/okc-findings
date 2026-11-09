# fd_r3_kernel_sources — one surviving mechanism per source (2026-11-04)

Lane R3 refresh of `fd_kernel_sources.md`. DeepGEMM levers 1+4 now BUILT (`moe_isync*.py`, `moe_sstage*.py`, checks 0e583f85/60f3621a pending); `ks_xqa.py` built. #1s to beat: layer 2.5401, moe 2.7642, attn 3.3389, bf16 1.6940, prefill 2.0754. Ours: 2.2837/2.3806/2.6590/1.5891/1.9087.

## Lever table (5 surviving)

| src | mechanism | source cite | board | lever shape | est delta |
|---|---|---|---|---|---|
| MonoMoE | Static-count register-resident warp softmax/argmax (compile-time Count regs, no LDL) + tie-break-by-lowest-index argmax | tmpbuild/engines/flashinfer/csrc/fused_moe/monomoe/src/moe_routing.cuh:24-60 | decode-layer (route glue; our Triton serial argmax still runs at small T) | dispatch swap inside routev4 glue | +0.05–0.1% layer |
| DeepGEMM | Wave-aware job→SM rotation: rank same-wave entries by size, rotate/reverse across SMs | tmpbuild/engines/deepgemm/deep_gemm/include/deep_gemm/scheduler/sm100_sparse_mqa_logits_metadata.cuh:75-121 | moe | host/job-order permutation of stream-K job list inside route() | +0.1–0.3% moe |
| TRT-LLM XQA | KV-tile striping across splits: idxKTile = idxKTileInit + idxIter*nbSubSeq | tmpbuild/engines/flashinfer/csrc/xqa/mha_sm90.cu:716-719,805 vs ours contiguous chunk attn_flatwin_v2.py:2554-2563 | attention | flag on split tile loop (5-line remap), fold into next A/B | ~0 |
| FlashMLA | Equal-payload packing w/ per-request fixed charge INSIDE balance (FLAT equal-work + SEGKF idiom, run on full cells only) | FlashMLA smxx/decode/get_decoding_sched_meta.cu:64-104 vs attn_flatwin_v2.py:2649-2692 vs sliding n_splits :2539-2570 | attention (sliding batched) | dispatch: route sliding cells to FLAT plan, reuse nact merge | +0.03–0.1% attn |
| LeanAttention | Plan-free global stream-K: tile index space global over B×H×Ctx, grid = SMs×CTAsPerSM, tiles cross request boundaries, ONE launch | artifacts/reports/research_03_venues.md:233-243,467-472 vs !FLAT per-request split attn_flatwin_v2.py:2539 | attention (sliding) | kernel-plan swap (= K6 flat-for-sliding); merge becomes per-tile host-block | +1.9% attn on sliding cells — deep change, fallback only |
| vLLM Marlin | RESIDUE ONLY — dominated. Adaptive block_size_m bound M*topk/E/block<0.9 = only untried trick | tmpbuild/engines/vllm/vllm/model_executor/layers/fused_moe/marlin_moe.py:346-348 | moe | — | ~0, do not build |

## Per-source notes
1. MonoMoE — sentinel-in-data phase handoff, launch-parity counters, routing ∥ prefetch all already ours. Only router microkernel survives.
2. DeepGEMM #473 — headline levers LANDED: expert-synchronous interleave → moe_isync*.py; kStaticStages → moe_sstage*.py. TMA multicast dead (no cluster launch). Reversed-remainder subsumed. Wave-rotation is the residue.
3. XQA — occupancy factor arm in flight (check 01b835e2); 4-warp merge ring dominated. Striping is only residue.
4. FlashMLA — combine kernel dominated by in-kernel merge; sm100 umma doesn't port.
5. LeanAttention — grid formula adjudicated (= _plan @ XQA_CTAS_PER_SM=2); the plan-free global tile space is what we don't ship on sliding.
6. Marlin — locks protocol = conflict-avoidance not balance; align blocks subsumed by in-CTA route(); dequant-in-m-loop is mma.sync-era (ours is wgmma+folded dequant).

## Where this leaves the day
Two levers worth a slot before close: **FlashMLA→sliding-FLAT** (dispatch-only if _plan exposes flat for sliding; else LeanAttention deep port — check attn_xqa_on verdict first, same lines) and **DeepGEMM wave-rotation** (host reorder, safe to stack on moe_isync if check lands green). MonoMoE router microkernel = reserve wave-3 glue tweak; XQA striping folds into next A/B free; Marlin nothing.
