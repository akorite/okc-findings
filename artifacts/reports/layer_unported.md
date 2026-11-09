# Unported mechanisms in the upstream files our layer was patched from

**+0.4 % of the layer score** — the largest mechanism present upstream and absent from `klr_fci5.py` is the
m == 1 GEMV family (`cgemv` / `sgemv` / `rgemv`, vjeux `muse_bf16_proj.py:2267-2500, 3173, 3415-3418, 3615-3696`),
worth **-1.5 … -1.9 µs on the decode-1 step** (73.5 µs) = 2.1-2.6 % of that load = **+0.35-0.43 % of the board**
(ceiling +0.82 % if the router and pooler could also move, which needs the fused `pr` launch split — that split costs
more than it saves). **Nothing clears 1.5 %: nothing was built; `layer_stack1.py` was not created.**

Arithmetic: the layer board is the equal-weight geomean over six loads (`src/okc_loop/scoring.py:98-107`), so
`Δscore/score = (Δt_load/t_load)/6`. All per-load µs below are one layer step from the judge-side CUPTI profile
(`artifacts/reports/kernarc_digest.md:29-34`): d1s **73.5**, v1s 109.9, d32s 154.1, v32s 305.6, byte floor 44.6.

---

## 1. Three-way diff: `klr_fci5.py` vs `alygg77/muse_decode_layer.py` vs `vjeux_okc2026/muse_decode_layer.py`

Ours is a **line-level superset of both layer files**; neither layer file carries a mechanism we lack.

| upstream file | what was checked | result |
|---|---|---|
| `vendored/vjeux_okc2026/muse_decode_layer.py` | defs / Triton kernels / `__global__` names, then a diff of the glue half (`:6539-11195` vs ours `:6863-16345`) and of the MoE source | ours ⊇ theirs. The 61 lines only they have are older forms of things we carry in a newer shape: `_wgs_copy` (ours has the SC6 argument), `_bfp_pdl`/`bfp_SK2_DELAY` (ours folded into `bfp__sk2_launch`), the pre-v2 `_boundary1`, older `bfp__boundary1` call shapes. No mechanism. |
| `vendored/alygg77/muse_decode_layer.py` | same, plus a full diff of `ATT7_CUDA_SRC` (`:112-3694`) against ours (`:122-3714`) | ours ⊇ theirs: **3 hunks only** — ours *adds* the `pf`/`pf_chunk` L2-prefetch parameters (pf4) and hoists `pdl_launch_dependents()` from the end of two branches to directly after `pdl_wait()` (strictly earlier release). Nothing dropped. **Coverage limit:** their vendored file is truncated at `:6061`, mid-`MOE_CUDA_SRC` — the MoE half of *their layer* cannot be audited; it does not matter, because their component MoE file is complete and identical to vjeux's (below). |
| `vendored/keell0renz/muse_decode_layer.py` | `diff` vs `klr_fci5.py` | 2 lines: `OKC_QDIRECT`/`OKC_PK2` and `OKC_WGSFCI 4→5`. Same lineage, not an independent upstream. |

Macro-level sweep (all `#define` in the nine upstream files vs ours): the only upstream macros our layer does not
define are `OKC_NACC` (default 2 in the component; ours passes `-DOKC_NACC=4` in its tiny/tiny2/mid builds — a tuned
knob, not a mechanism), `RD` (an alias for `__device__ __forceinline__`), and `RK_QS 6` (the `rk` route, see §2.3).

## 2. Reverse direction: component **board** files vs our layer's inlined copies

This is where the only real unported mechanisms are — all in the dense family.

| # | mechanism | upstream path:line | what it does | effect on the layer (µs/load) |
|---|---|---|---|---|
| 1 | `cgemv` (m == 1, 12-bit) | `vjeux/muse_bf16_proj.py:2301-2427`, `3173` (`CG_CONFIGS`), `3415-3418`, `3615-3696` | dedicated scalar GEMV at m = 1: reads the same lossless 12-bit planes (`L` low bytes + `C` 4-bit codes), decodes in registers, FMA, fixed-order reduction, one escape warp per CTA. No TMA, no smem ring, no wgmma, no m-dimension waste | **-0.58 qkv**, **-0.45 o**, **-0.87 pooler** (measured, same harness as the layer's own autotune: `bf16/profiles/geo.c8c64c72c95a4ae9.log:49-51` `cgemv=12.48 sk2=13.06`, `8.77/9.22`, `17.34/18.21`). `alygg77` `:3202` re-measures pooler 2x512 as **-0.34** (2x256 ties) |
| 2 | `sgemv` (m == 1, bf16) | `vjeux/...:2434-2484` | bf16 GEMV for a K = 4096 slice; the board's unpooler-m1 fallback | -0.32 ex-unpooler vs sk2 (`geo...log:52` `cgemv=7.1 sgemv=7.49 sk2=7.81`); dominated by `cgemv` at every measured shape |
| 3 | `rgemv` (m == 1, fp32 router) | `vjeux/...:2238-2300` | router GEMV reading the fp32 weights directly (8.4 MB, no hi+lo split) | **-1.00** on an isolated router launch (`geo...log:53` `rgemv=7.42 sk2=8.42`) — **not reachable in our layer**: at T ≤ 32 the router is inside the fused `bfp_fused_pooler_router` launch, and splitting it back out costs +5.5 µs at d1 (`kernarc_digest.md:61`) |
| 4 | `gqx` (m 33..256, big tiles) | `vjeux/...:2007-2230`, `USE_GQX` | TMA + wgmma cluster GEMM, 128/256-row tiles, 3-way K split (108 CTAs at qkv m=128) | ≈ 0. Our T = 128/256 cells already run `rq`/`Lt` plans that beat its board numbers (`profiles/klr_fci5.….log:18-32`: qkv m=128 14.81, m=256 21.45, unpooler m=128 19.55); the board's own retime changed **0 of 20** scored cells (`bfp_geo.md`; `bf16_cells.md:16`) |
| 5 | `rk` (router split-K GEMM, m 33..256) | `vjeux/...:2724-2790`, `3492`, `3643`, `RK_QS 6` | router GEMM with K split, order-dependent fp32 adds | ≈ 0. Our router is inside the fused `pr` cell at T > 32, and the router branch **ends 6.0 µs (d16) / 8.6 µs (d32) before** the pooler GEMM ends — it is off the critical path (`kernarc_digest.md:52`) |
| 6 | `cutlass` pingpong GEMM | `vjeux/...:2859` (`_build_cutlass`), `3758`; `alygg77/...:2798-2904` | CUTLASS pingpong 128x64, pooler m 129..256 only, built in a background thread during prepare | **negative.** Board: beats cuBLAS only at pooler m=256 (**31.3 vs 34.5 µs**, `alygg77/...:2795`); our in-layer T=256 `pr` is **25.5 µs** for the same 75.5 MB (`dense_rate.md`). Also +14 µs at m=129, +4 µs at m=200 (partial tiles) |
| 7 | `sk2_balance_perm` (12-bit row deal) | `vjeux/...:3226-3262`; used at `3305-3325` under `SK2_BALANCE_LAYERS = {"qkv_gate"}` | sorts 4-row groups by escape count and snake-deals them to sk2 row blocks so every CTA carries ~the same escape work (escape warps set the CTA tail) | **-0.5 µs qkv at m = 8/16 only** (their own measurement, `:3227`); `o`/`unpooler` at m=32 *lost*. 2 of 6 loads → **+0.06 % board**. Below the 2.2 % σ |
| 8 | `_ensure_skinny_stream` | `vjeux/...:3350` | runs the skinny route on its own CUDA stream | this is the fork/junction lever, already closed by measurement in-layer: each join back into the PDL chain costs ≥ 2 µs and needs > 3 µs of hidden serial gap (`kernarc_digest.md:49`) |
| 9 | `dd_DECODEDIET` / `_NDD_cpssm` (pooler m=16) | `alygg77/...:5968-6010` (zlib blob), `6144`, `6596-6644` | a pooled "cps" kernel route wired for `('pooler', 16)` only, source compressed | ≈ 0: the cell it targets is already at **3.21 TB/s = 96 %** of the 3.35 TB/s peak in our layer (`dense_rate.md`, T=16 `pr` 58.7 MB in 18.3 µs), so the ceiling is 0.8 µs of a 128 µs step; no measurement of the route exists and its source is not auditable |

**Attention (`alygg77`/`vjeux` `muse_attention_decode.py`): zero.** Their two component CUDA sources are
**byte-identical** (2,790 lines each, `diff` empty); ours is that source **plus** the `pf` L2-prefetch parameters and the
hoisted `pdl_launch_dependents()` (§1).

**MoE (`alygg77`/`vjeux` `muse_moe_w4a16.py`): zero.** Ours is a superset: SC6 6-bit scale slabs
(`UNIT_SC = OKC_SC6S ? 3072 : 4096`, `WGS_PIECE = OKC_SC6W ? 8960 : 9216`), `OKC_WGSFINSS`, `OKC_WGSFCI 5`,
`wgs_prefix`/`wgs_stage_at`/`wgs_cta_of` cost-weighted stream-K, and the NTT 1..8 segment switch all present. What the
diff shows on their side is the older 16-bit-scale layout (`SCDK(..., k16)` vs ours `k4`) and older accumulator
overlap lists — i.e. our file is the later revision of the *same* kernel.

---

## 3. Rejected, and why

* **#1-3 (the GEMV family) — the only live item, rejected on size, not on shape.** Porting qkv + o + unpooler to
  `cgemv` at T = 1 is directly applicable (all three shapes have a `cgemv` instantiation: K 8192/1024/4096) and is
  worth **-1.5 … -1.9 µs** on d1s = **+0.35-0.43 % of the board**. Even a hypothetical zero-cost fused
  pooler+router `cgemv`+`rgemv` (which needs a new kernel: the two cells are one launch today) tops out at
  **-3.6 µs = +0.82 %**. Both are under the 1.5 % bar and inside the board's 2.2 % σ.
* **#4, #5, #6**: the cells they target are already served by kernels that measure *faster* in-layer (rq deliveries,
  the fused Lt `pr` plan at 2.96 TB/s); porting them is a regression, not a gain.
* **#7**: measured at -0.5 µs on two loads by its author; +0.06 % board.
* **#8**: closed axis by prior in-layer measurement (junction cost).
* **#9**: unmeasured, unauditable source, and its target cell is at 96 % of peak.
* **Closed-axis list**: nothing in the unported set touches a closed axis (`forder` is the MoE ordering and is present;
  slightly-overlapping-but-distinct: #7 is *dense* escape-work balance, not expert ordering — I priced it rather than
  dismissing it).

## 4. Chosen item and flag

**None.** Nothing above 1.5 %, so no file was created and no flag exists. `klr_fci5.py` is untouched; `lint` and
`tools/cu_build.py` are unchanged by construction (no new artefact to check).

If a future run wants this axis anyway, the shape is: copy `klr_fci5.py` → `layer/layer_stack1.py`, add the
`cgemv`/`sgemv` kernels + `cg_compress` + `CG_CONFIGS`, dispatch at `T == 1` for `qkv`/`o`/`un` behind
`OKC_M1GEMV = False` (default off, incumbent `sk2` untouched), leave `pr` fused. Expected **+0.4 %**, i.e. it needs
**≥ 3 runs averaged** to be distinguishable from A/A noise at 2.2 % σ — which is why it is not worth one of the 5
remaining runs.

## 5. Kill criterion

Kill the whole "unported upstream mechanism" axis unless a *new* upstream file appears. Concretely: the audit shows
`klr_fci5.py` is a line-level superset of every alygg77 / vjeux / keell0renz file on disk, and the whole remaining
delta across all nine files is **-3.6 µs on one load = +0.82 % of the board at the absolute ceiling**. Re-open only if
(a) an upstream layer/component file arrives that is *not* already absorbed, or (b) someone finds a dense cell in the
T ≤ 32 path running below 2.5 TB/s (the current floor is 3.21 TB/s on the heaviest cell), or (c) the board's σ drops
below 0.8 % — then #1 becomes resolvable.

## 6. Method and coverage limits

* Symbol-level set diffs (top-level defs, nested defs, `@triton.jit` kernels, `__global__`/`__device__` names) for all
  nine upstream files vs `klr_fci5.py`, prefix-normalised (`bfp_`/`bfp__`) to avoid naming-only mismatches.
* Textual diffs of the three CUDA sources (`ATT7_CUDA_SRC`, `MOE_CUDA_SRC`) and of the Python glue half, plus a full
  `#define` value comparison — this is what proves "superset" rather than "same names".
* Effect estimates are **measured deltas from the upstream files' own logs** where they exist (`bf16/profiles/geo.*.log`,
  `SK2_BALANCE_LAYERS` and `v137` comments), and in-layer profile µs (`dense_rate.md`, `kernarc_digest.md`,
  `profiles/klr_fci5.9743b6e71fbd42fd.log`) otherwise; no GPU was run for this report.
* **Not audited**: the tail of `alygg77/muse_decode_layer.py` (truncated at `:6061`, mid-MoE) and the body of the
  `_NDD_` zlib blobs. Their component files cover the same kernels and were audited in full.
* **[INFERENCE]** where flagged: the board's m=1 candidate times include a ~4 µs graph-replay fixed cost, so I use only
  their *differences* (the fixed cost cancels); the d1s step time is alygg77's file, our file's decode-1 row is within
  0.5 % of it (`const_sweep_fci5.md:105`).
