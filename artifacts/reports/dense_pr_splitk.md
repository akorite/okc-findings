# The pooler+router cell (4608 x 8192): tile arithmetic, and the K-split candidates the bake now decides

File under test: `src/okc_loop/vendored_work/layer/klr_qd2.py` (board 82c098ae + ATT7; 2.2536).
Candidate: `artifacts/cas/e5b72fa5d381095c335545630890e5c6378a4242773eed529b852bdca3ed73cb.py`
(same content as `artifacts/work/klr_pr.py`). Kernel sources are unchanged: this is a
config/geometry change plus three lines of bake wiring.

## 0. The cell

`st["pr"] = cat([pooler 4096, router_hi 256, router_lo 256], 0)` → **[4608, 8192] bf16 = 75.5 MB**. The route
kernel then adds columns 4096+e and 4352+e, so the cell's output contract is FP32 `[T, 4608]` (S = 1).

## 1. The 36-tile arithmetic: confirmed, for the file's own kernel — and it was never offered

`gemm90` (`bfp_GEMM_SRC` 7860-8032) is oriented weight-rows-by-tokens: `BM = 64*NWG` **output columns** per CTA,
`BN` = the *token* tile (the wgmma `n`), grid = `(N / (64*NWG), splits)`, one 64-k step per wgmma, `STAGES` of
`BM*128 + BN*128` bytes. So for the pr matrix:

| NWG | column tile | N-tiles = 4608/(64 NWG) | CTAs = tiles x splits |
|---|---|---|---|
| 2 | 128 | **36** | 36 x splits |
| 1 | 64 | 72 | 72 x splits |

**36 confirmed.** Real tile: 128 output columns x `BN` tokens x 64 k, 2 consumer warpgroups + 1 producer warp
(384 threads), `__launch_bounds__(384, 1)`.

The cliff was not the arithmetic, it was the candidate set: `bfp_GEMM_CONFIGS` had entries for all five dense
layers and **no `("pr", .)` at all**. The only bfp option for this cell was `ppick="bfp"` = two separate launches
(pooler 4096 + router 256) — the author's measured +10..12 us. So at v16/v32 the cell ran the cuBLASLt nvjet plan
`('tn',1,fp32)` / `('tn',2,fp32)`: 64-column x 144-token CTAs = **72 CTAs at T=128** (0.55 waves) and 144 at
T=256 — and 2.34 TB/s at T=256 even with 1.09 waves, i.e. filling the machine alone does not close it.

## 2. T <= 32 (the fused `sk2f` launch): nothing fits — refuted, with the constraint that forbids it

`bfp__sk2_geom`, the config `'pooler'` (R = NWR*NT*8 = 32 rows/CTA, NT=4, NWR=1, WSK=8, UPW=1, UPS=8),
grid = 4096/32 = **128 CTAs**, `__launch_bounds__((8+1+NESC)*32, 1)` → 1 CTA/SM, 4 of 132 SMs idle:

| tokr | warps | CTAs | stage (SST) | stages | ring |
|---|---|---|---|---|---|
| 8 | 13 | 128 | 32 KB | 6 | 192 KB |
| 16 | 13 | 128 | 40 KB | 5 | 200 KB |
| 32 | 12 | 128 | 56 KB | 3 | 168 KB |

Three static asserts close every lever:
* grid must be **exactly 128**: the fused router half indexes `rt = cta>>2` over its 32 tiles (256 rows / 8) and
  `rq = cta&3` over quarters of K, so R = 32, i.e. the pooler row tile is pinned. R=16 (256 CTAs = 2/SM, the
  `pooler8h_r72` geometry) covers only half the router tiles.
* `static_assert(!FUSE || (COMP && CL == 1 && UPW == 1 && UPS == 8 && WBYR <= WBY))` → UPS = WSK*UPW = 8 fixed
  (the router's 4 stages are 8 k-units each) and NT >= 3. UPS = 4 — the one change that would give 5 stages at
  TOKR = 32 (stage 40 KB) — is rejected by that assert and would need the router half reworked.
* smem: 4 stages at TOKR=32 need 4 x 57344 = 229,376 B, and the ring can only be `232448 - 18848` (alignment pad,
  barriers, corrections, escape table, zero row) = 213,600 B. So **3 stages is the ceiling**, and it is the whole
  of the d32 gap (same bytes at T=16 run 3.21 TB/s with 5 stages, 2.77 with 3).

`pooler32_r144` (NT=2, NWR=2) has the same R and the same 57 KB stage → no gain; that is why the d32 side is
left alone.

## 3. Candidates added (T >= 128), every gate stated

`bfp_GEMM_CONFIGS[("pr", 128)] = (2, 3, 6)`, `[("pr", 256)] = (2, 1, 4)` (defaults), plus `bfp_GEMM_ALT`:

| m | bn | nwg | splits | stages | tiles | CTAs | waves@132 | stage | smem | split-K traffic |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 128 | 2 | **3** | 6 | 36 | 108 | 0.82 | 32 KB | 193 KB | 6.75 MB |
| 128 | 128 | 2 | 4 | 6 | 36 | 144 | 1.09 | 32 KB | 193 KB | 9.0 MB |
| 128 | 128 | 2 | 2 | 6 | 36 | 72 | 0.55 | 32 KB | 193 KB | 4.5 MB |
| 128 | 128 | 1 | 1 | 8 | 72 | 72 | 0.55 | 24 KB | 193 KB | 0 |
| 128 | 128 | 1 | 2 | 8 | 72 | 144 | 1.09 | 24 KB | 193 KB | 4.5 MB |
| 256 | 256 | 2 | **1** | 4 | 36 | 36 | 0.27 | 48 KB | 193 KB | 0 |
| 256 | 256 | 2 | 2 | 4 | 36 | 72 | 0.55 | 48 KB | 193 KB | 9.0 MB |
| 256 | 256 | 1 | 1 | 4 | 72 | 72 | 0.55 | 40 KB | 161 KB | 0 |
| 256 | 256 | 1 | 2 | 4 | 72 | 144 | 1.09 | 40 KB | 161 KB | 9.0 MB |

Gates that shaped the set:
* `BN` must be >= M (gemm90 has one token tile, no token-block grid) → 128 at m=128, 256 at m=256.
* smem = `1024 + STAGES*(64*nwg + BN)*128 + 16*STAGES + 64` <= 232,448: 8 stages at nwg=1/bn=128 (197.8 KB) and
  nwg=1/bn=256 (165 KB) are the deepest that fit; 4 stages at nwg=2/bn=256 is the limit (197.8 KB).
* **residency is 1 CTA/SM always** (`__launch_bounds__(., 1)` and 161-198 KB of smem), so the Chord 4-CTA/SM
  lever is not available in this kernel; "higher residency" was dropped, not omitted.
* split-K cost is not free: the partial buffer is the whole `[M tokens x 64 nwg columns]` FP32 accumulator per
  CTA — 32 KB/CTA at bn=128, 131 KB/CTA at bn=256. At T=256 a 2-way split therefore adds 18.9 MB of DRAM traffic
  to a 75.5 MB cell; the ws max over all configs stays under the existing 16.8 MB, so no workspace grows.
* `assert` in `bfp__gemm_src` now pins one instantiation per `(nwg, bn, splits, out_f32, a_s)` with one stage
  count (the dispatch cannot see `stages`; a silent mismatch would have been a shorter ring, not a wrong answer).

## 4. The bake decides (nothing is forced)

* `bfp_prepare` builds `state["raw"]["pr"]` = cat(pooler, hi, lo) — byte-identical to `st["pr"]` — plus
  `mid_cnt`; `bfp__route("pr", m) = "gemm"` only for m >= 128.
* `bfp__selftest` runs it at m = 1/8/16/32/33/128/200/256 against an FP64 reference. `pr` is compared like
  `router` (no bf16 cast, tol 2e-5) because it emits FP32; a single failing config disables the whole pr gemm
  route (`bfp__route_key` is per (layer, bn)), i.e. failure = today's plan.
* `bfp__candidates("pr", 128/256)` now returns the default + every `bfp_GEMM_ALT` entry (4 alts at 128, 3 at 256)
  + cuBLAS; `bfp__autotune` times them as cold-L2 graph replays (24 reps, interleaved) and stores the winner.
* `prepare` prints, per cell and m, what it measured:
  `TUNE pr      m=128 gemm(2, 2, 6):..us gemm(2, 4, 6):..us gemm(1, 1, 8):..us gemm(1, 2, 8):..us gemm:..us torch:..us`
  (also `pooler` and `router`, the other members of the cell).
* `_tune` gained a fifth arm `bfpp` (one launch over [4608, 8192], FP32 out) that must first pass
  `_bfp_pr_ok`: the picked config's output vs `x.double() @ st["pr"].double().t()` within 1e-4 (and finite) —
  this also rejects a cuBLAS-bf16 pick, whose out would break the FP32 consumer. It then must beat the best
  other arm by **2 %** (`MARGIN_RQ128`'s margin) or the Lt plan stands.
* `prepare` no longer forces `mm_pick[(T,"pr")] = True` at 128/256; it keeps `"bfpp"` when the bake chose it and
  `True` otherwise. qkv/o/un stay forced (unchanged).
* Consumer: `_pool_router_fp32` returns `(fp32 [T,4608], 1)` for `bfpp` — the same S=1 contract the Lt plan
  gives, so `_graft_route(SPLIT=False)` and the non-graft `_route` are untouched. `_bfpp` is never offered below
  T=128 (the fused sk2 claim runs there).

## 5. Price

| cell | bytes | time | rate | floor @4.287 | excess | row | excess share |
|---|---|---|---|---|---|---|---|
| v32 (T=256) | 75.5 MB | 32.2 us | 2.34 TB/s | 17.6 us | **14.6-14.9 us** | 350.5 us | 4.2 % |
| v16 (T=128) | 75.5 MB | 26.6 us | 2.84 TB/s | 17.6 us | **+9.0 us** | ~261 us | 3.4 % |
| d32 (T=32) | 58.7 MB | 21.2 us | 2.77 TB/s | 13.7 us | +7.5 us | 144.9 us | 5.2 % |
| d16 (T=16) | 58.7 MB | 18.3 us | 3.21 TB/s | 13.7 us | +4.6 us | — | — |

At **3.2 TB/s** the cell is 23.6 us at both verify sizes: **-8.6 us at v32** (-2.45 % of the row, +0.41 % of a
six-T geometric-mean score) and **-3.0 us at v16** (-1.15 % of the row, +0.19 %) → **~+0.6 %, 2.2536 -> ~2.267**
if a 3.2 TB/s geometry exists among the candidates. If none wins by 2 % the cell stays on the plan, so the
downside is the prepare time of the bake (~40 ms of timing, ~0.5 s for the two FP64 reference products).

d16/d32 are untouched: the fused launch has no addable candidate (section 2).

## 6. Gates

* `ast.parse`: ok.
* `okc_loop.cli lint`: same 12 FAILs as the base, the only delta a line-number shift of the pre-existing
  140,927-char line.
* `tools/cu_build.py --variant MID|MAIN`: 9 FAILs each, **identical set** to the base (`GRAFT_*`, `SL_GLUE`,
  `ROUTER_FP32`). As warned, that gate does not compile the bfp template kernels, so `gemm90`'s new dispatch is
  not compiled by it; it was checked by generating the source (`bfp__gemm_src(True)`, 21 lines, all conditions
  `nwg/bn/splits/out_f32/as`) and by reading it against `launch_gemm90`'s signature.
