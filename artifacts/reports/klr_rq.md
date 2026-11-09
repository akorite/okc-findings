# klr_rq: the packed `rq` dense path at T=16/32 and at the verify loads

Candidate: `src/okc_loop/vendored_work/layer/klr_rq.py` (16368 lines). Base: the board layer
`src/okc_loop/vendored/keell0renz/muse_decode_layer.py` sha `82c098ae` (16345 lines), row **2.24736**.
Line 1 (`# okc: problem=muse_decode_layer`) and the provenance header are untouched; `okc lint`
still identifies the file as `tier0 muse_decode_layer`.

## 1. What the pack is, and whether it is a real 12-bit lossless codec

**Yes — 12 bits/weight, bit-exact, and it is the same size class DenseGap is asking for (slightly denser).**

* `_rqpack(w)` (`klr:15917`) → `_pack_nat` → `_c12(fr, 1)` (`klr:14435`): 8 mantissa bits of each bf16
  are replaced by a 3-bit window select off a per-tensor base `B` (the `pw10 c12` scheme); the low 8 bits
  are stored raw. Layout: `lay = (N/128, K/64, 8 tiles, 1536 B)` = **1536 B per 128 rows × 8 K-weights**
  = **12 bits/weight**, i.e. **12288 B per (128 N-rows, k64) vs bf16's 16384 B = 0.75×**.
* Weight positions that fall outside the 8-wide window go to exception lists: `esc` (per `(ks,nb,warp)`,
  sentinel-terminated, `p.ecap` stride) and, for the FL-bit-20 cursor variants, `esc2`
  (per `(warp,lane)`, `CAPT=4` sub-slots). Tables add ~0.8 MB per layer (≤1.5 % of the packed bytes).
* **Lossless, verified per layer at prepare time**: `_rqbake` refuses the pack unless
  `torch.equal(_rqunpack(*pk, n, K), w.view(torch.int16).int() & 0xffff)` — the round trip must reproduce
  the bf16 source **bit for bit** (`klr:16079`). So the only numerics left in the cell are wgmma fp32
  accumulation order and the BF16 *output* store.
* **Kernel numerics gated separately**: `_rqval` (`klr:16010`) runs a full one-hot sweep
  (`eye(K)*8`, every K column group) plus `randn` and 1 %-outlier draws, and requires rel-L2 ≤ `RQTOL[KS]`
  **and** max-row rel ≤ 5× that, against an fp64/fp32 reference of the bf16 source
  (`tol` = 5e-3 for KS=2 cells, 8e-3 for qkv's KS=7). A failure drops the pick; the cell keeps the
  incumbent. `_pickrule` then needs the variant to be **≥2 % faster** than the proven reference
  (`MARGIN_RQ128`), else the incumbent stands.
* The pack is **built per layer, once, for all m** — `pk` is reused for every baked m
  (`st["rq_pk"]`), so adding m values costs no VRAM. The *application* is what `RQ_USE` gates
  (`klr:16211`, `if (_RQ_NAMES[lay], m) in RQ_USE`), not the packing.

## 2. Byte accounting per cell at m=128/256 (one layer call)

| cell | shape (N×K) | today's path @256 | bf16 bytes | packed bytes (W + tables) | ratio | kernel that runs it |
|---|---|---|---|---|---|---|
| qkv | 2304×8192 | **already rq** (board) | 37.75 MB | 28.31 + ~0.80 = **29.1 MB** | 0.77 | `rq` `n256s5m1hzmEH` / `n256s5m1hzm` |
| un  | 8192×4096 | bf16 `torch.mm`/graft-Lt @256 (**new**) | 67.11 MB | 50.33 + 0.79 = **51.1 MB** | 0.76 | `rq` `n256s5m1hzmEH` / `n256s5m1hzm` |
| o   | 8192×1024 | fp32 `torch.mm` @128/256 (**new**) | 16.78 MB | 12.58 + 0.79 = **13.4 MB** | 0.80 | `rq` `n256*` @256, `s4s5hzw*` @128 |
| pooler+router (`pr`) | 4608×8192 | fp32 `torch.mm` (fused) | 75.50 MB | **not packable — see §4** | — | — |
| **five cells** | | | **197.14 MB** | 121.64 MB packable → **93.6 MB** | 0.77 | |

197.14 MB is DenseGap's 197.1 MB exactly (qkv 37.75 + un 67.11 + o 16.78 + pr 75.50).

**Delta this file actually moves** (qkv@256 and un@128 are already packed on the board):

| load | new bytes saved | of the load | at the dense cells' measured marginal rate |
|---|---|---|---|
| v32 (T=256) | un 16.0 + o 3.4 = **19.4 MB** | 207.8 µs | 6.5–9.2 µs → +0.52…+0.74 % |
| v16 (T=128) | o 3.4 = **3.4 MB** | 175.2 µs | 1.1–1.6 µs → +0.11…+0.15 % |
| d16 (T=16) | qkv 8.7 + un 16.0 + o 3.4 = **28.1 MB** | 92.3 µs | 9.4–13.4 µs → 0.182 %/µs → +1.7…+2.4 % |
| d32 (T=32) | same **28.1 MB** | 124.0 µs | 9.4–13.4 µs → 0.135 %/µs → +1.3…+1.8 % |

Marginal rate 2.1–3.0 TB/s is the band implied by "dense cells sit at 1.5–1.6× their 4.8 TB/s byte
floor" and by DenseGap's own −38 MB ≈ −18 µs. **Expected total: ≈ +2.3…+2.6 % of the layer score**
(row 2.24736 → ≈2.30–2.31). The per-cell truth is printed by the bake (`BAKE <layer> m=<m> rq:<v> <us>
(ratio … vs r6 …)` and `PICK …`), so the judge log self-reports whether each cell landed.

## 3. Tiling analysis (what breaks when m is not 128/256)

* **The weight pack is N-blocked, not M-blocked.** A CTA owns `NB = N/128` (one 12-bit pack block,
  12288 B/k64) — or 64 W rows when `FL & 16` (HALF), where `grid.x = 2*NB` and the escape sub-list
  index becomes `4*hh + (warp&3)`. So `N % 128 == 0` is the only N constraint (`_rqpack` refuses
  otherwise); qkv 2304, un/o 8192, pr 4608 all pass.
* **K is split across `grid.y = KS`**, `s0 = ks*KC/KS`, `KCS = (ks+1)*KC/KS - s0` (`klr:15264`):
  boundary split, *not* required to divide (qkv already runs KS=7 over KC=128). Host gate
  `p.KC/KS >= GS` (`GS` = stages per consumer) and `KS <= 64`. The one-wave rule in the source is
  `N/128 × KS <= 132`.
* **M is never tiled.** `NT` (template param) is the *x-tile* row count, and the vector is
  `grid = (NB·(HALF?2:1), KS)`, `blockDim = 384` (or 256 under `FL & 1024`). The only m gate is
  `if (p.m < 1 || p.m > NT) return -4;` and the maps are built with `box_outer = NT` against an
  **extent of `p.m`**: `rq_map` comments "out-of-range rows read as 0 and are not written", so
  `m < NT` is legal by construction (TMA zero-fills the A rows above m; the wgmma is `m64nNk16`, so a
  112-row zero tail is just wasted A). That is why **no new instantiation is needed for T=16/32** —
  the existing NT=128 variants accept any `m ∈ [1,128]`, and NT=256 accepts 256.
* `MS` (`RQMS`) = 2 only for the `m2*` variants (two 64-row token halves), and
  `static_assert(MS == 1 || (GS == MS && !HALF))`; `HALF` (FL bit 4) = 64 W rows, one math warpgroup
  (`NW=4, CW=64`), `grid.x = 2NB`; `H2` (bit 10) = 256-thread CTA, needs HALF; `F2` (bit 11) = 384
  threads, full 128 rows. None of the compiled variants uses them, so nothing to satisfy.
* **FL bit meanings** used by the probe set: 0 `HINT` (L2 cache hint on the bulk), 1 `ZI` (in-kernel
  out zeroing + `cnt` handshake), 2 = plain store instead of `cp.reduce.add` (bit 3 = no store at all,
  `klr:15700-15702`), 4 `HALF`, 5 `EARLY`, 6 `MRG` (one empty barrier per slot), 10 `H2`,
  11 `F2`, 12 `DILP`, 13 `WPF`, 14 `FX2` (fixed-iteration escape walk), 15 `WK` (patch-table walker,
  needs `PRB==0 && MS==1` + `ST*8*PTW*4` B smem), 16 `EH` (two token halves, **out map box = NT/2**;
  `static_assert` admits only NT=128/ST5-8/selfzero and the one NT=256 shape `FL==65603`),
  20 cursor + `esc2`, 22 `VOTE`, 23 `w` cursor, 25 `CURP` (patch 8 hi-byte regs).
* **What blocks what**: nothing structural for `m<NT`. The four real gates are all Python, and they
  are exactly the four this file changes: `_rqbake`'s hard-coded `for m in (128, 256)`, `RQBAKE`
  having no key for a new m, `dense()`'s `plan … if plans and T > 32` (which made `rq` unreachable
  below 128 even with a pick), and `mm_pick[(T,"qkv")] = "bfp"` + `T in (128, 256)` in `kernel_fn`.
* **Grid/occupancy consequence of packing** (smem, registers): nil. The same `RqC<>` instantiations
  run, `SMEM = 1024 + ST*(XB+WB) + 24*ST + 16` is unchanged (it is a function of NT/ST, not of the
  weight encoding — the W tile is 12288 B *packed* per k64 slot either way, vs 16384 B of bf16 per
  slot for the incumbent kernels), and registers/`setmaxnreg` are untouched. Cost is VRAM:
  **+13.4 MB** for the new `o` pack (`un`/`qkv` packs already existed on the board).
* **Why not the `t16*/t32*/h16*/h32*` shapes**: they *are* the better small-m tiles (NT=16/32 ⇒ x tile
  4 KB instead of 16 KB per slot) and they are already written in `RQV`/`RQCUR`, but each is a whole
  extra instantiation of this kernel and `_rqstart` compiles one set (`RQPROD`) inside a 280 s prepare
  budget that already guards `−45 s`. Adding 2–4 instantiation is the follow-up only if the NT=128
  pick wins the margin but lands short of the byte model.

## 4. The fused pooler+router is *not* a `RQ_USE` away — it is fp32-out

`rq`'s epilogue writes **BF16 only**: `rqrun` rejects any `out.scalar_type() != kBFloat16`,
`rq_map` builds a `CU_TENSOR_MAP_DATA_TYPE_BFLOAT16` map, and the epilogue is
`stmatrix.trans → SW128 slab → TMA reduce-add` into that bf16 map. That is not a size limit, it is a
*dtype* limit, and the pooler+router cell depends on the other dtype:

* `state["pr"]` is `pooler(bf16) ++ router_hi(bf16) ++ router_lo(bf16)` (`klr:11207`) — a deliberate
  **fp32 emulation of the router**: the consumer adds the `hi` and `lo` column blocks in fp32, so the
  logits come out at ~fp32 precision. The fused `bfp_fused_pooler_router` preserves it explicitly
  ("pooler bf16, router **fp32** … BF16 hi + lo with FP32 accumulation", `klr:9752`).
* A bf16 *output* rounds the `hi` block to 2⁻⁹ and makes `lo` worthless, i.e. it would silently demote
  the router to plain-bf16 logits and can flip a top-8 near-tie. Not a `_rqval` verdict — the GEMM
  would still pass — so the bake gate cannot cover it.
* Two legal ways, priced:
  **(A) FP32-out epilogue in `rq`** (f32 slab + a `cp.reduce.async.bulk.tensor` FLOAT map, plus
  `rqrun`/`rq_map` dtype plumbing and a new `RQV` FL bit): a kernel change, not mechanical, and
  unmeasurable offline. Byte win would be the full −18.9 MB on 75.5 MB.
  **(B) split the fused launch**: pack the pooler's **4096** rows (`RQKS["pooler"] = 4`,
  `RQBAKE_L[("pooler",128/256)]` and `RQCUR`'s cursor variants for this layer **already exist** — only
  `RQLAYERS`, `cs["pooler_raw"]`, `_RQ_NAMES` and `RQ_USE` are missing) and run the 512-row router
  hi+lo as its own `torch.mm(..., out_dtype=float32)`. Bytes: 75.50 → 50.33+8.39 = **58.7 MB
  (−16.8 MB, −22 %)**; cost: **one extra launch**. Net ≈ −4…−6 µs per verify load, i.e. **+0.3…+0.5 %**
  of the layer — around the 0.5–2 % judge noise, and it needs a new branch in `_pool_router_fp32` whose
  2-launch cost I cannot measure offline. Not shipped; the exact edit is
  `_pool_router_fp32` → `if T > 32 and state["chen_rq"].get(("pool", T)): pooled = rq; logits =
  torch.mm(h, state["router_fp32"].t(), out_dtype=torch.float32)` + `RQLAYERS += ("pooler",)` +
  `RQ_USE += {('pool',128), ('pool',256)}`.

## 5. What this file changes (complete diff, 13 hunks)

1. `_rqbake` m loops `(128, 256)` → `(128, 256, 16, 32)`; `_bl` = the union of all four m's probe
   lists; `_pickrule` margin `MARGIN_RQS = 0.83` for m ≤ 32 (1.20× over `r6` ⇒ ≥11 % over the real
   incumbent, since bf16_proj is 4–9 % ahead of `r6` at T ≤ 32 per `BFP_MAX_T`'s note).
2. `RQBAKE[16]`, `RQBAKE[32]` probe lists; `RQBAKE_L[("o",128)]`, `[("o",256)]`, `[("unpooler",256)]`.
   Every probe name is already in `RQPROD` (checked statically), so **no new CUDA instantiation**.
3. `RQLAYERS = ("qkv_gate","unpooler","o")`, `_RQ_NAMES["o"]="o"`, `cs["o_raw"] = weights["o"]`
   (`+13.4 MB` pack, one `_c12` over 8.4 M weights ≈ 0.3–1 s, one `_rqunpack` compare).
4. `RQ_USE` = shipped set `{(un,128),(qkv,256),(qkv,128)}` + `(un,256),(o,128),(o,256)` +
   `(un,16),(un,32),(qkv,16),(qkv,32),(o,16),(o,32)`. The `RQ_USE` filter is the *only* application
   gate, so a variant that fails `_rqval`/the margin/margin-reset simply never appears here.
5. `dense()` tries `_rq_gemm` first, for every T (it used to require a `graft_lt` plan, i.e. T > 32).
6. `kernel_fn`: the `qpick == "bfp"` branch tries `rq` for qkv before `_bfp_gemm`; the FA3 fallback's
   `dense(x, state["qkv"], T)` gets the same one-line rq attempt (identical when there is no pick).
7. Everything else — weights, MoE, attention, glue, the kernel source — byte-identical.

Prepare-time cost: +1 pack (`o`, ~0.3–1 s, 13.4 MB), 2 extra m values × 3 layers × ≤6 probes
(one-hot `_rqval` + a 7-pair graph timing each) ≈ 0.3–0.6 s, 2 extra `r6` baselines per layer.
Well inside the 280 s budget, and the new m's are baked **after** 128/256 so a budget overrun drops
only them (`PREPARE_BUDGET_S − 30` guard sits inside the candidate loop).

## 6. Gates

* `ast.parse` — OK; `py_compile` — OK.
* `okc lint` — the board file's own 30 tier-0 errors, unchanged in count and text (normalized diff is
  empty except for the shifted line numbers; both files report `30 error(s)`). No new construct.
* `tools/cu_build.py --variant MID` and `--variant MAIN` — **identical to the board file**: the same 9
  extracted sources, all FAIL in this venv for a missing libcu++ header (`cooperative_groups/…/info.h`
  → `nv/target`), which the unmodified board file hits identically. The `rq` CUDA source is not among
  the extracted ones (it is `load_inline`d at prepare), and this file adds no CUDA source.
* **Not verified offline** (no judge access, and `_rqbake` only runs inside a real prepare): the
  actual per-cell `rq` vs `r6` ratios at m=16/32/128/256 and the pack's `esc2` cap per layer. The bake
  prints one line per candidate plus `PICK`/`esc2 slots max=…` per layer, so the first judge run
  reports every cell's outcome and falls back per cell on any failure.
