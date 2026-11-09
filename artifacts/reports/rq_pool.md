# rq_pool: the pooler+router cell at T=128/256 through the 12-bit pack, without un-fusing the router

Candidate: `src/okc_loop/vendored_work/layer/klr_pool.py` (16,7xx lines, **998,682 B**), base `klr_qd5.py`
(998,629 B, row 2.24736 family). The 998,682 figure matters: the file was 1,371 B under the lint's
`1000000`-byte single-file budget, so the feature had to fit inside that and 1.3 KB of decoration was
freed (below).

## 1. What the route actually is at v16 / v32 (T = 128 / 256)

T is not the batch: the graft chain's own comment states the mapping (`klr:11709`) — **T=1 decode 1,
T=8 verify 1, T=128 verify 16, T=256 verify 32**, and the decoder loads are T=16/32 ("decode 16/32").
Live chain at T in {128, 256}: `_after_attention` → `_graft_after_attention` (`klr:12504`), because
`compose_graft_enabled` defaults to `T in (1, 8, 128, 256)`:

* `_pool_router_fp32(h, hr, T, state, ppick)` (`klr:13529`) → no `bfpf`/`bfp` pick at these T → `dense(h,
  st["pr"], T, state, "pr")` → `_GRAFT_LT_PICKS[("pr", T)]`, **one cuBLASLt launch** over
  `pr = pooler(4096) ++ router_hi(256) ++ router_lo(256)`, bf16, `fp32` out `[T, 4608]` = `allp`
  (`nvjet_sm90_tss_144x64_64x8_1x1_v_bz_TNN`, 31.2 µs at v32 in the judge's own profile).
* `_graft_route[(T, 2)](allp, allp, allp, ..., SPLIT=False, PACKED=1)`: kind 0 normalises
  `allp[t, 0:4096]`, kind 1 reads `logit = allp[t,4096+e] + allp[t,4352+e]`, i.e. **the consumer sums the
  hi and lo row blocks in fp32** — that is the router's FP32 emulation, and the reason `RQ_USE` excluded
  the fused 4608-row matrix (`rq`'s epilogue is bf16-only; a bf16 out would round `hi` and make `lo`
  worthless, silently demoting 256 logits to bf16 precision and able to flip a top-8 near-tie).
* `_sl_tile_route` / `_sl_cluster_after_attention` / `_gle_after_attention` do **not** run at these T —
  they are gated on `SL_TILE_CONFIG = {1, 8}` / `SL_CLUSTER_CONFIG = {1, 8}` / `GLE_T = {1}`. The one
  route node that exists at T>32 already has the two-pointer form: `SPLIT=True` takes `POOLED` `[T,4096]`
  (dtype-agnostic, so bf16 is legal there) and `LOGITS` `[T,256]` fp32 separately — the `S == -1` branch,
  reached whenever the cell returns two tensors.

## 2. Shape chosen, and the graph-node count of each

| option | nodes in the region | verdict |
|---|---|---|
| (a) decode the pack inside the fused kernel's mainloop | 2 (fused, route) | `sk2f` is `m <= 32` *by construction* (`TOKR <= 32`, `MT <= 2`, `sk2` ring sized 200 KB). At m = 128/256 it is a new CUDA mainloop — unverifiable offline in this container (no `nv/target`, no `tvm_ffi`), so not shippable on 1 check. |
| (b) the pooler's packed GEMM called from the fused kernel's epilogue | 2 | the "launch region" that matters is the captured graph: a second GEMM is a second node regardless of who issues it, and it still needs the new kernel. Strictly worse than (c). |
| (c) **chosen**: pack only the 4096-row pooler; router stays fp32 in bf16_proj's own kernel | **3** (rq pool, router, route) vs 2 | +1 node, −16.8 MB, no new kernel, no new format, every probe already in `RQPROD`. |

(c) in detail: `pooled = _rq_gemm(state, "pool", h, T)` → bf16 `[T,4096]` from the 12-bit pack, and
`logits = _bfp_gemm(state, "router", h, T, 256, 8192)` → fp32 `[T,256]` (bf16_proj's `wide`/`gemm90`
router, already baked for m=128 and 256 by `bfp_WIDE_CONFIGS[("router",128,1/2)]`, `bfp_GEMM_CONFIGS
[("router",128/256)]`; both are fp32-out with BF16 hi + lo and fp32 accumulation, `bfp__selftest` holds
them to 2e-5). The cell then returns `(pooled, logits), -1`, which `_graft_after_attention`'s existing
`S == -1` branch hands to `_graft_route(SPLIT=True)`. Router bytes: 512 rows x 8192 x 2 = 8.39 MB
(unchanged — the same hi+lo pair that `pr` holds), pooler: 67.11 → 50.33 MB.

## 3. The diff (nothing else changes)

1. `RQLAYERS += ("pooler",)` (last: a bake-budget overrun then drops only this layer).
2. `_RQ_NAMES["pooler"] = "pool"`; `RQ_USE += {("pool",128), ("pool",256)}` (the only application gate).
3. `RQBAKE_L[("pooler",16)] = [("pooler",32)] = []` — the pooler is never packed below m=32 (sk2f already
   reads the 12-bit stream there), so those probes would be pure prepare time.
4. `_pool_router_fp32`: one guarded block before the `ppick == "bfp"` fallback
   (`if T > 32 and state.get("rq_pool", {}).get(T) and state.get("bfp") is not None`), falling through to
   the incumbent if `_rq_gemm` returns `None`.
5. New `_rqpool_T` + final `prepare()` override: the prepare-time self-check and A/B (§5).
6. Byte budget: 13 decorative `# ----…` divider lines (96 chars, all at column 0, verified outside every
   multi-line string by `tokenize`) and the `_make_chen` mechanism history (pw11–pw31, now pointing at
   `artifacts/reports/bf16_mechanism_scan.md`) removed — comments only, so the file lands at 998,682 B
   instead of 1,003,666 B (which is a **new** lint error: `1003666 bytes exceeds the 1000000-byte budget`).

No new CUDA source, no new `rq` instantiation: the pooler's probe names are all in `RQPROD`
(`d2s5hze`, `d2s5hzwP`, `s4s5hzw`, `s4s5hzeEH`, `d2s6hze`, `n256s5m1hzmEH`, `n256s5m1hzm`), `RQKS["pooler"]
= 4` gives a one-wave grid (4096/128 x 4 = 128 CTAs), `RQTOL[4] = 6e-3`, and `RQPICK128[("pooler",128)] =
"d2s5hzwP"` was already in the force list.

## 4. Byte accounting per shape

| stream | bf16 today | packed now | delta |
|---|---|---|---|
| pooler 4096x8192 | 67.11 MB | 50.33 MB (`32 x 128 k64 x 8 tiles x 1536 B`) | −16.78 |
| escape/patch tables | — | +~0.8 MB | +0.8 |
| router hi+lo 512x8192 | 8.39 MB | 8.39 MB (unchanged, now its own launch) | 0 |
| **cell** | **75.50 MB, 1 launch** | **58.7 MB, 2 launches** | **−16.0 MB net** |

Identical at T=128 and T=256 (the stream is m-independent). VRAM cost of the new pack: +50.33 MB (+~0.8 MB
tables), one `_c12` over 33.5 M weights (~1–3 s) + a bit-exact `_rqunpack` compare, which `_rqbake` refuses
to keep unless it passes.

## 5. Prepare-time self-check (the file's own idiom, no new machinery)

`_rqpool_T(st, T, gen)` runs per T, after `_rqpool_T`'s own prerequisites exist and after the rq bake:

1. synthetic `data`/`o_in` exactly as `_bnd2_T`/`_selftune` build them, `local_start` moved to the busiest
   32-expert range so the MoE is never timed idle;
2. **parity**: `ref = _after_attention(inc)` vs `got = _after_attention(pk)` with a one-key `rq_pool`
   shadow dict, requiring `torch.equal(ids)`, rel-L2 ≤ 1e-5 on `out` and `s2` (the residual stream's own
   budget, same constant as `_fuse_s1_check`/`_bnd2_choice`); the `pk` run must therefore reach the route
   as `S == -1` — if the pack silently did not land, the two runs are the same path and the margin below
   rejects it anyway;
3. **timing**: `_bnd2_ab` (cold-L2 flush, alternating order, CUDA-graph replays, per-round win count) on the
   *cell only* — `_pool_router_fp32(h, None, T, inc|pk, ppick)` — not on the whole chain: 4 % of the chain
   at v32 is ~11 µs, larger than the cell's entire excess, so a chain-level 4 % bar could never pass. The
   cell is ~26–31 µs, where `max(4 %, 3 µs)` = 3 µs is the honest band;
4. `on = same and ts_pk <= ts_inc - max(0.04 * ts_inc, 3.0) and wins_pk >= 24/32`, printed as
   `rq_pool T{T} fused=… packed=… us wins=[…] eq=… -> on/off`. Any exception disables that T, and every
   failure path (no pack, no bfp, no margin, no parity) leaves `state["rq_pool"] = {}` — i.e. **the default
   is byte-identical to klr_qd5.py**.

## 6. Expected µs

The cell is 31.2 µs at v32 (2.34 TB/s) and 26.6 µs at v16 (2.84 TB/s); −16.8 MB of stream is −7.2 / −5.9 µs
at those rates, −3…−6 µs once the second launch's bubble and the pooler kernel's own rate (the `rq` pair at
T=128 runs 78.6 MB in 31.2 µs = 2.52 TB/s; un's m=128 analogue, same 128-CTA one-wave grid at 50.3 MB,
bakes at 21.4 µs cold-L2) are priced in. Both are inside the audit's −4…6 µs band for +0.28 % expected /
+0.56 % ceiling of the layer; the A/B decides per T, so the worst case is 0 and the v32 cell (the largest
single excess in the layer) is where it should win.

## 7. Gates

* `ast.parse` — OK (also `tokenize`-verified that the deleted divider lines were outside string literals).
* `okc_loop.cli lint` — 30 tier-0 errors, **the identical set** as `klr_qd5.py` after line-number
  normalization (same constructs, same counts, no `portability` size error; 998,682 B < 1 MB).
* `tools/cu_build.py --variant MID` and `--variant MAIN` — **identical `ok`/`FAIL` sets** to the base file
  (MID 2 ok / 9 FAIL, MAIN 5 ok / 12 FAIL; the FAILs are this container's missing `nv/target`,
  `cuda/std/type_traits`, `tvm/ffi`, `cudaGetLastError`).
* Not verifiable offline (no judge access, `rq` compiles only inside a real prepare): the pooler's pack
  exactness and esc2 cap (`BAKE rq pooler pack not exact` / `esc2 slots max=` lines), the probe timings, and
  the final on/off. Every one of those fails safe to the incumbent.
