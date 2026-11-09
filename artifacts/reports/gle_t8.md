# GlueFuse Stage 1 (GLE at T=8): gate passes in Python, path is unrunnable — M = 2T rows

Board file: `src/okc_loop/vendored/keell0renz/muse_decode_layer.py` (sha 82c098ae, 16345 lines).
Verdict: **no file produced.** `GLE_T = {1: True, 8: True}` + `GLE_RC = {1: 8, 8: 8}` alone makes the T=8
layer raise. All Python gate keys do pass; the blocker is a hard 8-row ceiling in both FP32 router
kernels, and design A feeds them **2 rows per token** (u = alpha*s*pre, v = beta*P*pre).

## 1. Patch ordering is fine — the b1 launch really would disappear

- `_sl_tile_after_attention` (:13285) calls `_sl_cluster_module().b1(...)` at :13301.
- Its wrapper `_after_attention` (:13352) is installed *before* the GLE wrapper (`_gle_prior_after =
  _after_attention` at :14158, GLE `_after_attention` at :14159). No later `def _after_attention` exists
  (grep: only :12520 and :14159), and `_gle_prior_after` is what GLE falls back to.
- So enabling GLE at T=8 skips `_sl_tile_after_attention` for T=8 and b1 stops being launched for
  `T in {1, 8}` (b1's only remaining callers are T=1, already preempted by GLE_T[1], and T=8).

## 2. Every Python gate key passes at T=8 (= `verify` T=8, agent-attention path)

`_after_attention` gate (:14160-14171), T=8:

| key | value | source |
| --- | --- | --- |
| `GLE_T[8]` / `GLE_RC[8]` | the two edits | :13653, :13657 |
| `gate is None` | yes — `kernel_fn` returns `_after_attention(..., None, T)` on the agent-attention path (:11490); q_len 1 always, q_len 8 on full layers (:11471) | :11490 |
| `_pdl(8)` | True (`PDL=True`, `PDL_MAX_T=128`) | :97-98 |
| `'graft' in state`, `compose_graft_enabled[8]` | True (already required by the live T=8 sl path) | :12737 |
| `T in SL_TILE_CONFIG`, `T in SL_CLUSTER_CONFIG` | (4,4,True) / (16,1024) | :13284, :13078 |
| `ppick` | `bfpf` (or `bfp`) — set for T in (1,8,16,32) | :12702-12705 |
| `opick == "bfp"` | set for T in (1,8,16,32) | :12703 |
| `moe == "agent"` | `DEFAULT_MOE="agent"` + TUNE_MARGIN guard; judge image has no Marlin bank | :108, :11186-11188 |
| `state["bfp"]`, `router_fp32` [256,8192] fp32, `router_fp32_ext` | all present | :11207, :13629-13631 |
| `g_preffn` bf16/contig/HIDDEN | present | :11217 |
| `_gle_o_choice` | `choice[("o",8)]=="sk2"` and `bfp__sk2_name("o",8)=='o8h_r88'`==`bfp_SK2_PICK[("o",8)]` | :14099-14104, :9528, :9541 |

`_gle_o_choice` at T=8 is *identical* to T=1 (both `tokr = 8` -> `o8h_r88`), and GLE_T[1] is live in the
board file, so this gate is already exercised on the board. `_gle_source()`'s assert
`(layer, R, NWR*WSK, CL) == ("o", 32, 4, 1)` holds for `o8h_r88` = (8192, 32, 4, 1, 4, 1, 2, 1).

## 3. The blocker: the 2T-row router input

`_gle_after_attention` (:14107-14146) allocates `uv = (2*T, 8192)` bf16 and `uvf = (2*T, 8192)` fp32; the
o epilogue writes rows `tt` (u) and `M + tt` (v) (:13857-13860 `put(uu, tt); put(vv, M + tt)`), and
`_gle_route` reads `LOGITS + t*256` and `LOGITS + (T + t)*256` (:13882 region, `q`/`r1` from `part`
layout `(cta*T + t)*8`). For T=8 that is **M = 16 rows**, and both router kernels cap at 8:

- fused pooler+FP32 router (GLE_FRT=True, the live branch): `gle_pr` host check
  `TORCH_CHECK(M >= 1 && M <= 8 && x.size(0) == M && x32.size(0) == M && ...)` at :13944-13945, dispatch
  `if (tokr == 8 && N == 8192 && K == 1024 && M <= 8)` at :13920, and the device side is 8-row by
  construction: `static_assert(... SST >= 32768)` (:13794, 8 router rows + M*x rows per 32 KB slot),
  `xa[u] = (th*4 + u < M)` with `th = tid>>7` in {0,1} (:13809), `afi[8]`/`af[32]` and `t < 8` loops
  (:13882, ilv path), `warp = router row` mapping.
- split path (GLE_FRT=False): BF16 pooler at 16 rows is fine (generic sk2, `(pooler,16)` exists), but the
  FP32 router GEMV is `router_fp32_launch` -> `if (m < 1 || m > 8) return -1;` (:13588), whose kernel has
  `acc[8]`, `t < 8` loops, `partial[8][8]`, out-writer `warp < m`, 256 threads (:13551-13584).
  `_router_fp32_gemv` raises `RuntimeError('FP32 router launch failed: -1')` on that.

So at T=8 either branch throws before a launch; `GLE_FRT` is a module constant, so no flag combination
avoids it (and running the 8-row router twice on rows u, v doubles the 8 MiB router-weight read,
~1.9 us at the 4.287 TB/s cold wall, plus 2 launches -> erases the gain).

Upstream agrees: every snapshot in the repo has `GLE_T = {1: True}   # T = 8 keeps 0389's b1` (:13653),
and `.vendor/okc-2026-muse-kernels/experiments/experts-v32/precision_glue.py:202` states the raw defaults
"enable T=1 only". The GLE path has only ever run at 2T <= 8, i.e. T <= 4.

## 4. What Stage 1 actually needs (new kernel work, hence this report)

Minimal: raise the FP32 router GEMV to m <= 16 — `acc[16]`, `#pragma unroll for t < 16 if (t < m)`,
`partial[8][16]` (512 B smem), out-writer `for (r = warp; r < m; r += 8)` (2 rounds, 256 threads kept).
Weight-bound at 8 MiB, so the extra 8 rows of FMAs are free. Verifiable standalone against a torch FP32
reference on random data — no model weights, no okc submission.
Optional, to keep one launch: extend `sk2f`'s FRT geometry to M=16 (slot 8*2048 + M*2048 = 49152 B ->
SST static_assert, `af[32]`->`af[64]`, `th*4+u` -> 16 tokens, ilv `afi[8]`->`afi[16]`, dispatch M<=16).
Then the two `GLE_T`/`GLE_RC` lines go live and b1 disappears at T=8, with `GLE_RC[8]=8` already
consistent (8 carry programs x 8 tokens x 128 cols = 8192, grid `(T, PARTS+1+nch) = (8, 13)`).

## 5. Loose end worth confirming

If the scored load name `verify 1` means **T=1**, there is nothing to win here: `GLE_T[1]=True` is already
in the board file, every gate key passes at T=1 too, so b1 is already gone at T=1 — the audit's T=8
framing only holds if the load's token count is 8.
