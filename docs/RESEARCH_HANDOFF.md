# RESEARCH_HANDOFF — OKC kernel climb (shared file for every fresh attempt)

Read `docs/PROGRESS.md` first (authoritative checkpoint), then this file, then your lane's
spec. This file is the *map*: what every new attempt must know before it spends anything.

## Current live board (2026-11-02 ~10:20 UTC)

| board | ours | #50 line | gap |
| --- | --- | --- | --- |
| muse_decode_layer | 10.31 ms / 2.194 | 2.303 | +4.97% |
| muse_bf16_proj | 1.570 | 1.589 | +1.2% |
| muse_attention_decode | 2.609 | 2.721 | +4.3% |
| muse_moe_w4a16 | 2.375 | 2.470 | +4.0% |

Overall rank = layer rank (#282). Deadline 2026-11-04 23:59 PT. Daily caps: 10 checks + 5
submits per board (halved since Oct 31).

## Incumbents (board files, CAS)

- layer: `artifacts/cas/90fc75e8eaf8839667812358f0313cecbbdbb28f0c69c0bde5d981a9c444a3a5.py`
- moe:   `artifacts/cas/6402f5f354428aa180681499e6e9ebd3554e40feafa569cbe4b1cc37b5d2b3e5.py`
- bf16:  `artifacts/cas/4ab25a7deaef51e2e7f5948bab7a49bc39fcb5d4208cbb37e4ffd2045e349631.py`
- attn:  `artifacts/cas/4cfc5cca0b8eafe0759db7a29ca22b508b7f1873130dc23d3c9bfaaebfe09682.py`

## Open levers (ranked by ceiling)

1. `OKC_SYNC0` — per-warp slot release in skinny_ring_ws; spec `vendored_work/sprint2/NEXT.md` §1.
   Ceiling −9.4% MoE / ≈the whole layer gap. Build: `vendored_work/moe/moe_sync0.py`,
   layer port `vendored_work/layer/layer_sync0.py`.
2. `OKC_SC6` — 6-bit lossless scale indices; spec NEXT.md §2. −1.5…−2% MoE.
   Build: `vendored_work/moe/moe_sc6.py`, combined `moe_sync0_sc6.py`.
3. `layer_autotune_dense.py` — per-shape build tournament, check `b49ac78067874b9d` in flight.
4. bf16 +1.2% to #50: no known mechanism; incumbent already tournaments Lt tactics/gqx.
   Only real new kernels or lucky variance.

## Closed — do not re-open (all measured, see PROGRESS.md)

ring depth (any form), activation sparsity, SwiGLU gate<0, accumulator precision, L2/multicast/
clusters/DSMEM, stream-K, cuBLASLt pins, batch sweeps, grid resize, pack_gqa/num_splits, deep
attn STAGES, dense ring, prepare_pad arena, PDL enablement, variance re-submits, rohanarun
bf16/attn ports, WGMMA/TMA custom dense GEMM (0.45–0.87x), Huffman BF16 codec (decoder 2.2–3x
too slow). Vendored rival tree `.vendor/okc-2026-muse-kernels/` confirmed no portable winners:
their best layer 2.1756 < our 2.1925.

## Rules for any attempt

- `okc problems` = authoritative quota. `okc-loop check` returns at dispatch; poll `okc status
  <job> --wait`. Never spend a check on expected move <3% (repeatability ~3.6%/shape).
- lint (`okc_loop.cli lint`) + `tools/cu_build.py` first; `cu_build`'s `_ROUTER_FP32_CUDA`
  `cudaGetLastError` failure is a known false positive (incumbent fails identically).
- New defines via `extra_cuda_cflags` ONLY — never into `*_DEFINES` tuples (module-name trap).
- sm_90a kernels → the H200 judge is the ONLY timing oracle. Colab (L4/A100) = correctness only,
  and only for paths without FP8/TMA (A100 has neither).
- Submits go through `auto_submit.evaluate()` — never bypassed.
