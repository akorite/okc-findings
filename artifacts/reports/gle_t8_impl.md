# GLE at T=8: implemented on the split pooler + an m <= 16 FP32 router

Candidate: `src/okc_loop/vendored_work/layer/klr_gle.py`, copied from the board candidate
`src/okc_loop/vendored/keell0renz/muse_decode_layer.py` (sha 82c098ae). Edits are 4 hunks, all inside the
GLE region (+ `_ROUTER_FP32_CUDA`); nothing outside `:13550-14125` changed.

## The three 8-row caps, and which one this change lifts

`gle_o` is called with `M = T` (8), so o's epilogue and its `M <= 8` checks are untouched. The two rows
per token (`uv`, `uvf`, `2T = 16` rows) go to the pooler and the router:

| cap | where | what m = 16 needs | this change |
| --- | --- | --- | --- |
| fused `gle_pr` / `sk2f` | `TORCH_CHECK(M <= 8)`, dispatch `M <= 8`, device `th*4+u` with `th = tid>>7` (2 h values), `af[32]`, `afi[8]`, `mbar_expect_tx(8*2048 + M*2048)` -> 48 KB slots vs the `SST >= 32768` static_assert | a new slot geometry, `af[64]`, 4 h values, second reduction round | **not taken** — the fused kernel is only entered when `2*T <= 8` now, so it never sees M = 16 |
| BF16 pooler at 16 rows | none: `bfp__sk2_name("pooler", 16) == "pooler"`, generic sk2 | nothing | taken as-is: `_bfp_gemm(state, "pool", uv, 2*T, ...)` |
| FP32 router GEMV at 16 rows | `router_fp32_launch` `m > 8 -> -1`; `acc[8]`, `t < 8`, `partial[8][8]` (256 B), out-writer `warp < m` (8 warps = 8 rows) | `acc[16]`, `t < 16`, `partial[8][16]` (512 B), `for (r = warp; r < m; r += 8)` | done |

Why the split pooler and not `sk2f` M=16: the fused kernel's router ring needs `8*2048 + M*2048` bytes per
slot (48 KB at M=16) against the 32 KB `SST` static_assert and the shared-memory budget it already spends on
pooler stages, i.e. a slot-geometry redesign of a generated kernel. The T=8 split is the laziest complete
path and costs one extra launch versus the fused one — but T=8 *already* runs the split pooler + GEMV today
(`_pool_router_fp32`'s `T <= 8` branch), so GLE adds no launch: it removes `_sl_cluster_module().b1(...)`
(`:13301`) and its cta cluster sync, and reuses o's epilogue for the boundary vectors.

## Router numerics: per-row identical, not just close

The kernel body is unchanged per row — same `k = tid*4; k += 1024` loop order, same 4 FMA per `float4` in
the same order, same warp butterfly (d = 16..1), same 8-partial shared-memory reduction (d = 4,2,1, lanes
8..31 zero). `m` only selects which of the 16 rows run (`if (t < m)`) and how many out-writer rounds execute
(`for (r = warp; r < m; r += 8)`), so rows 0..7 compute the same instruction sequence at m = 8 and m = 16 and
rows 8..15 are new rows, not a different reduction.

Check (numpy transcription of the kernel, exact float32 op order, 4 experts x 8192 x random data):

```
rows 0..7 bit-identical m=8 vs m=16: True
rel err vs float64 ref  m=8: 8.4e-07   m=16: 5.7e-06
```

i.e. the m = 16 path is bit-identical to the m <= 8 path on the rows that already existed, and the new rows
carry the same FP32 GEMV error class as the old ones (both ~1e-6 rel, far below the ~1e-3 that could move a
top-8 pick: the router feeds `sigmoid(logit/32) + bias` and the pick is decided by gaps >> 1e-6).

## The other gates (all left alone, so a miss still falls back)

`GLE_T = {1: True, 8: True}`, `GLE_RC = {1: 8, 8: 8}`; `_after_attention`'s gate, `_pool_router_fp32`,
`gle_pr`, `gle_o` and every `SL_*_CONFIG`, `GLE_FRT`, `GLE_ILV` value are unchanged. T=1 keeps the fused
`gle_pr` (M = 2) exactly as before. Knobs that the T=8 call now exercises, all already present:
`SL_TILE_CONFIG[8] = (4, 4, True)`, `SL_ROUTE_CONFIG[8] = (4, 0, 0)`, `SL_CLUSTER_CONFIG[8] = (16, 1024)`,
route grid `(8, 4 + 1 + 8) = (8, 13)`, carry programs `8 x 8 tokens x 128 columns = 8192`.

Also re-checked against the source, since T = 8 first exercises them: `_gle_carry` for `TT > 1` uses
`fma(alpha, s, beta*bf16(r1*P))` on columns < 4096 and `fma(beta, bn, alpha*s)` on columns >= 4096, which is
exactly `_SL_GLUE_CUDA`'s b1 branch `if(SLICED || (PRELOADED_BASE && col < 4096))` (`:11796-11804`), and the
tail's `col >= 1024` mask in the o epilogue matches b1's `cluster_squares_pair(ns, masked, ...)` plus
`_sl_cluster_finish`'s `(tail + sum(head^2))/8192` split.

## Gates run

- `ast`: parses.
- `lint`: full-output diff versus the board candidate is one pre-existing `sandbox: time.time` line number
  (14346 -> 14350). **Delta 0 errors, 0 warnings**; the same 30 tier-0 errors.
- `tools/cu_build.py klr_gle.py --variant MID`: identical result set to the board candidate (ok: MOE, ATT7;
  the 9 FAILs are the pre-existing environment ones — `tvm/ffi/...`, `nv/target`, and the two nvrtc host-decl
  artifacts). The router source is one of those two, so it was compiled device-only (host launcher stripped,
  `#include <cuda_runtime.h>` dropped, as it is for the board file): **ptxas ok, 48 registers (was 40), 512 B
  smem (was 256), 0 spills**.

## Not verified here (no GPU, no torch on this workstation)

- The T=8 GLE launch sequence itself (`gle_o -> pooler(16) -> router(16) -> _gle_route`) cannot be executed
  offline; it is the T=1 GLE sequence with 16 rows instead of 2 and the b1-carry arithmetic that is already
  live for T=16 in `_sl_tile_after_attention`. Nothing in it is a new kernel.
- `verify-1`'s token count: every snapshot notes the T=8 gate exists because "T = 8 keeps 0389's b1"; if the
  scored load is really T=1 then b1 is already gone there and this change buys nothing. Worth one check.

Expected: the b1 launch + cluster sync at T=8 disappears, +0.9-1.25% of the layer score per the GlueFuse
audit; below the 1.5% check threshold on its own, so it belongs in a batch.
