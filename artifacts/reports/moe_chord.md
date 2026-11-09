# novitalabs/chord — small-batch W4A16 path, extracted

Source: `github.com/novitalabs/chord` @ `main` (cloned 2026-11-03, `/tmp/chord`), Apache-2.0.
All line references are to that tree. `docs/performance.md` is where the percentages come from.

Two kernels live in the repo. Only one is the lead:

| kernel | file | who it beats |
|---|---|---|
| **indexed** (humming-derived) | `chord_kernels/operator/{tuning.py,include/humming/**}` | Humming's indexed path — this is the swap-AB `mma.sync` decode path |
| grouped masked/contiguous (DeepGEMM-derived) | `include/deep_gemm/impls/sm90_w4a16_gemm.cuh` | DeepGEMM; `_ctas_per_sm` -> 2 for `block_m <= 24` |

## 1. Selection: when the small-batch path is chosen

`tuning.py`
```
_H200_DECODE_MAX_TOKENS_PER_EXPERT = 18.75          # :45   <-- the number in our brief
def _table_for(capability, n, k, experts, m):       # :493
    if m / experts < _H200_DECODE_MAX_TOKENS_PER_EXPERT: return "h200_decode_ep8"
    return "h200_prefill_ep8"                       # else wgmma
```
The band below **18.75 routed rows per expert** is a *different compiled tile family*, not a knob on
the wgmma one. The decode table (`_Table("h200_decode_ep8", "mma", True, 8)`, `tuning.py:37`) is
`layout="mma"`, `swap_ab=True`, fallback `block_m=8`.

`_select_indexed_kernel_config`, `h200_decode_ep8` branch (`tuning.py:399-421`):
```python
tokens_per_expert = m / max(experts,1) / 0.9        # m = routed rows on this GPU
bm8_max = 5.5 if k >= 4096 else 6                   # gate/up goes to bm16 later than down
block_m = 8 if tokens_per_expert <= bm8_max else (16 if tokens_per_expert <= 13 else 24)
num_ctas_per_sm = 4 if k < 4096 and block_m == 8 else 3
return _indexed_config(prof, block_m=block_m, block_n=256,
                       num_ctas_per_sm=num_ctas_per_sm, swap_ab=True)
```
`_indexed_config` (`tuning.py:48-69`) then fixes `block_k=64`, `num_stages=4`,
`warp_shape=(block_m, 64, 64)`. Layout selector is `warp_n` (`api.py:163`):
`"wgmma" if warp_shape[1] == 32 else "mma"` — i.e. **the mma path is the one with warp-N 64**, and
`api.py:150-156` rejects `swap_ab=False` for any `block_m % 16 == 8` tile: *"MMA indexed configs with
block_m=8 must set swap_ab=True"* (there is no m8-shaped non-swapped mma tile).

## 2. The extraction table

| property | chord, `h200_decode_ep8` | evidence |
|---|---|---|
| instruction | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | `config/mma.py:76` `_MmaOp._generate_ptx` |
| operand A | **weights**, dequantized to bf16 in registers, tiled by `n_out` on mma-M: `kNumARowTiles = WarpShape::N / 16` | `include/humming/mma/wmma.cuh:30-38, 61-72` |
| operand B | **activations** (tokens), bf16, on mma-N: `kNumBColTiles = WarpShape::M / 8` | same |
| accumulator | `regs_c[kNumARowTiles][kNumBColTiles]`, n_out-major, "mirror of WGMMA" | `wmma.cuh:38-40` |
| 4-bit -> bf16 | int4 with zero-point 8, no LUT: `lop3` builds bf16 pairs `128+w`, then one `__hsub2` with `136` and one `__hmul2` with the group scale; `regs[kAFragPerm[i]]`, `kAFragPerm={0,2,1,3}` to land in A-frag slots | `arith/mainloop_arith.cuh:74-101` |
| per-16 scales | one bf16×2 word per (16-K group), pre-loaded two iterations ahead into `bs[2][kNumScaleWords]`, read with `__low2bfloat162/__high2bfloat162` (single `mov.b32{lo,hi}`, deliberately not through the array base: SHF.R.U64 costs 1-2% on decode) | `arith/mainloop_arith.cuh:12-45, 50-57` |
| tokens -> M when M is 8 | CTA tile `block_m` = token tile (8/16/24); per-warp token count is `block_m/8` n-tiles; at `block_m=8` **one** n-tile | `wmma.cuh:34`, `tuning.py:399-421` |
| padding guard | semi-static: `num_token_tiles` = populated 8-token tiles of this m-block, counted in the kernel from the row-index sentinel; `run()` executes `j=0` unconditionally and skips `j>=1 && j>=jmax` so 1-tile blocks stay branch-free | `kernel/humming.cuh:90-101`, `wmma.cuh:76-102` |
| n_out tile | `block_n=256` (full EP8 gate/up N=4096 in 16 CTAs, down 7168 in 28) | `tuning.py:416-419` |
| threads/CTA | `block_m*block_n*block_k / prod(warp_shape) * 32` = **128** (4 warps) for `8x256x64` | `api.py:195-197` |
| warp tile | `(8, 64, 64)`: 64 n_out rows x 8 tokens x 64 K | `tuning.py:61-62` |
| regs/warp (8x256x64) | A 2x4x1x4=32, B 2x1x1x2=4, C 4x1x4=16, qb 2x4=8 -> 60 u32 + addressing | counted from `wmma.cuh:41-45` |
| K stage | `block_k=64`, cp.async ring, `num_stages=4` | `tuning.py:48-55, 399-421` |
| smem/CTA | `stage = bm*bk*2 + bn*bk/2 + (bk/32)*bn*2` = 1024+8192+1024 = **10,240 B**; ring 4 st = 40 KB, unioned with the `bm*bn*2`=4 KB epilogue buffer | `tuning.py:270-281` |
| CTAs/SM | **3** (4 for the short-K down at bm8); `__launch_bounds__(threads, num_ctas_per_sm)` | `tuning.py:410-421`, `kernel/humming.cuh:24` |
| residency budget | `num_stages * (smem + 1024) <= 233472`; deepest ring of {6,5,4} that fits | `tuning.py:26-29, 284-292` |
| stream-K | on for decode gated/up and down at bm8/16; FP32 partials, one rounding | `tuning.py:540-560`, `docs/performance.md` |
| w13 -> SwiGLU -> w2 | **not fused**: chord is a GEMM library. gate/up and down are two calls per projection with the activation between them in global memory; the epilogue only rescales and writes | `ops.py`, `epilogue/*`, no `silu` anywhere in the tree |
| activation tile | A (x) is *not* the operand-`A`; the swap means x is TMA'd as B and read s2r by `LoaderB`, weights dequantized in place in `regs_a` per k-step | `memory/s2r_pipeline.cuh:44-56`, `s2r_loader/loader_a.cuh` (`load_swap_act`) |

## 3. Side by side with our kernel (`artifacts/cas/6402f5f3...py`, the current best check)

Our M <= 16 bands are TINY (M<=4, `OKC_NW=8`), TINY2 (M<=8) and MID (M<=16). All three are
**skinny-only** builds: `route()` emits no fat job (`-DOKC_SPLIT=1`, `route()` line ~545:
`nfj = 0; nsj = ceil(tot/SK_T)`), so every unit is a *skinny unit*:

| property | chord h200_decode_ep8 | our TINY/TINY2/MID (M <= 16) |
|---|---|---|
| instruction | mma.sync m16n8k16 swap-AB | **same**: `mma_bf16(...)` from `dq4()` registers, `x0[2c]` B-frag (`skinny_ring`, lines 870-886) |
| A operand | weights, 16 n_out rows per mma-M tile | **same**: `dq4(ww[st]) -> r0..r3` = 16 rows x k16 into A |
| B operand | tokens, 8 per mma-N | **same**: one 8-token tile (`acc`), a second (`acc2`) only when a job has > 8 tokens |
| n_out tile | 256 rows/CTA | **16 rows/unit** (`UNIT_CODES = 32768 = 16*4096/2`) |
| token tile | 8 / 16 / 24 by tokens-per-expert | 8 (`OKC_SKT=8`), jobs split to <= 8 |
| K ownership | warp owns all K of its 64 rows | **8 warps split K** of the same 16 rows, 512 K each, then a cross-warp reduce (`S.red[i&1][warp][lane]`, `for ww in 0..NW`) |
| threads | 128 (4 warps x 64 rows) | 256 + 32 helper warp = 288 (`NTL`) |
| weight ring | cp.async 4 x 10 KB | TMA 3 x 36 KB (`OKC_NS=3`, `UNIT_BYTES=36864`) |
| CTAs/SM | 3-4 | **1** (smem request = `max(NS*UNIT, NSF*UNIT + 2*FT_T*XP_FT*2)` = 178,176 B + `Smem`) |
| epilogue reduce | per-warp accumulators, no cross-warp sum | cross-warp partials + helper warp |
| m64 wgmma | only in the *prefill* table | only in MAIN/WGS (M >= 17), `wgmma.mma_async...m64n{8..64}k16` |

**Which axis is m64?** In our kernel the warpgroup's m64 is the **n_out (weight-row) axis**:
`wg_rs` issues `m64nNk16` with `A = dequantized weights (natural k order) from registers, B = x tile
from smem` (line 1248 comment) and WGS accumulates "its 64 rows over K" (line 1817). Every one of
those 64 rows is a real weight row. So the premise "at T=8 an expert's tile is 8/64 = 12.5% dense and
87.5% of the MMA work is padding" does not hold for our kernel: the padding axis is the mma-N token
axis, which is 8-wide on both sides, and our T <= 16 path never issues a wgmma at all.

## 4. What is actually left to take, and what it is worth

`docs/performance.md`, H200 EP8 decode (chord vs its upstream Humming, same routing draw):

| tokens/GPU | gate_up | down | sum |
|---|---|---|---|
| 20 | 1.02 | 1.03 | 1.03 |
| 30 | 1.04 | 1.11 | 1.07 |
| 40 | 1.06 | 1.10 | 1.07 |
| 50 | 1.03 | 1.06 | 1.04 |

masked/grouped decode (the DeepGEMM-style path), per tokens/expert: 8 -> 1.05, 16 -> 1.10,
32 -> 1.04, 64 -> 1.15 (gate_up+down). So the honest transferable range is **+3..+10%**, not 4-29%;
29% appears nowhere in their own tables.

And the sentence that decides what to port (`docs/performance.md:167-169`):

> "The 8 tokens per expert row was closed by running **two resident CTAs per SM, not by a tile**:
> none of 91 swept layouts beats the heuristic's `16x256x128`."

Combined with `_ctas_per_sm` (`grouped/heuristics.py:258-283`: 2 CTAs for `block_m <= 24`,
*loses* 7-15% below one wave, *loses* once the ring drops under 3 stages), the ordering of the levers
for a sub-18.75-rows-per-expert decode is:

1. swap-AB mma.sync small tile — **already ours** for M <= 16 (see the table above). This is the part
   of chord's 4-29% that our row has already banked; our MAIN/WGS build is where a wgmma->mma.sync
   swap is still unexplored (M >= 17, out of this ticket's band).
2. **2+ resident CTAs/SM** — the lever their doc says closed the 8-token row, and the one our
   M <= 16 builds cannot currently reach: the smem request includes the fat ring
   (`NSF*UNIT + 2*FT_T*XP_FT*2 = 178,176 B`) although a skinny-only build never runs a fat unit.
3. `wgmma.wait_group<1>` commit width (`OKC_WGSGS`), 3-6% on gate/up per the sweep — WGS-only, so
   M >= 17, not this band.

## 5. Tile / CTA math for the build that this ticket adds

`-DOKC_M8=1` sizes the shared memory of a skinny-only build to the ring it actually uses and takes
the residency floor off the kernel, which are the two things that keep the incumbent at one CTA:
```cuda
#if OKC_M8
  return base + (size_t)OKC_NS * okc::UNIT_BYTES;      // no fat ring, no fat x staging
#else
  ... existing max(xa, xb) ...
#endif
// and, on moe_fused:
#if OKC_M8
__global__ void __launch_bounds__(NTL, 2) moe_fused(Params p) {   // ptxas: <= 113 regs/thread at 288 threads
#else
__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {
#endif
```
(`__launch_bounds__`'s second argument is a floor, not a ceiling — `(NTL, 1)` lets ptxas use up to 227
registers and that alone would refuse a second CTA. chord sets the same pair:
`__launch_bounds__(kNumThreads, kNumCtasPerSm)`, `kernel/humming.cuh:24`.)
With `-DOKC_NS=2` (the build's other define), `sizeof(okc::Smem)` measured at **36,576 B** (nvrtc probe,
base rounded to 36,608) and the edited `okc_smem_bytes` compiled and run under g++ with the real
`OKC_NS`/`UNIT_BYTES`:

| | incumbent MID | M8 (`NS=2`, `OKC_M8=1`) |
|---|---|---|
| dynamic smem | 214,784 B (209.75 KB) | **110,336 B** (107.75 KB) |
| CTAs/SM | 1 | **2** (2 x 110,336 = 220,672 <= 233,472, the H200 budget; 3 would be 331,008) |
| grid at bs<=16 | 132 | 264 |
| threads/SM | 288 | 576 |
| 36 KB units in flight per SM | 3 | 2 x 2 = 4 |
| work per unit | unchanged: same `mma_bf16`, same `dq4`, same 8-warp K split, same epilogue | unchanged |

`NS=2` is forced rather than tuned: with `NS=3` the build is 147,200 B and two CTAs no longer fit,
so a deeper ring and a second resident CTA are mutually exclusive at a 36,864 B slot.

Nothing in the fragment, packing, decode, epilogue or routing code changes, so the arithmetic per
output element is the same expression tree: the only differences are which CTA owns which unit and a
shallower ring. Order across CTAs is not preserved (`atomicAdd(&ss_cur()[...])` in the phase-B
epilogue, line ~933), so the probe gate compares with a tight relative tolerance, not `torch.equal`:
`_M8_TOL = 1e-5` relative Frobenius, which is ~5 orders of magnitude tighter than any tile/mapping
error would produce and ~3 orders looser than the FP32 reordering.

## 6. The flag, the expected %, the kill criterion

* flag: `M8 = False` in `src/okc_loop/vendored_work/moe/moe_mma8.py`. Off -> `prepare()` compiles
  exactly the tuples the base file compiles and `kernel_fn` is byte-identical to the base's variant
  ladder. On -> `prepare()` adds one `load_inline` (`M8_DEFINES`) and runs `_m8_tune()`.
* binding: per token count only. `_m8_tune` times the incumbent and the M8 build at the 2-CTA grid on
  a judge-shaped probe for every `M in 1..16`, and binds `state["m8"][M] = (defines, grid)` only when
  `rel_err <= 1e-5` **and** `t_us < base_us - max(0.03*base_us, 4.0 us)`. Any cell that raises, any
  shape whose probe fails, and every `M > 16` keeps the incumbent build. Shapes 31/32/33, 63/64/65,
  127/128/129 never enter this code path (M > MID_MAX_M), so they are untouched by construction.
  (`M8_GRID_MUL` is a single value, not a sweep: 3 CTAs/SM cannot be resident at a 36,864 B slot
  ring — see the table above.)
* expected: +2..5% on the three <= 16-token cells (decode 1, verify 1, decode 16), i.e. ~+1..2.5% on
  the row total (~100-250 ranks at 101 ranks/+1%), extrapolated from chord's own
  "two resident CTAs closed the 8-token row" and our 3 -> 4 units in flight per SM. **Unverified:**
  no GPU run was permitted for this ticket, so the sign of the effect at 2 slots of ring depth
  (deeper ring is worth real percent in `swapab.py`'s NS4/NS5 measurements, and NS=2 is shallower
  than the incumbent's 3) is an open question that only `_m8_tune` can answer.
* kill criterion: if no cell in 1..16 binds (i.e. the M8 build never beats the incumbent by the
  margin while staying exact), the candidate is dead — delete `M8_DEFINES`, `_m8_tune` and the
  `#if OKC_M8` block. A second kill: if any cell binds but the scored row moves < +0.5%, the lever is
  below the noise floor and should not be resubmitted.

## 7. What is deliberately not ported

* **int4 zp=8 `lop3` decode** (`arith/mainloop_arith.cuh:74-101`): their weights are plain int4 with
  a fixed bias; ours are NVFP4 (e4m3 per-16 block scales, `dq4()` + `bmul2`). Same cost class, no
  gain, and requantizing is forbidden.
* **their epilogue / stream-K reduction**: ours is fused (SwiGLU in phase A, norm stats and the
  `out` reduction in the final pass) and already does FP32 partials.
* **`block_n=256` per CTA with 4 warps x 64 rows**: this is the one real structural difference left
  (chord owns 256 rows/CTA, we own 16/unit and split K). Porting it means a 128-thread CTA, a
  4-row-block warp compute, a k-sliced ring at `block_k=64`, and a new epilogue mapping — a new
  kernel body in a file whose only test is a GPU. A wrong row/tile mapping there is silent numeric
  corruption, which the brief ranks worse than no candidate, so it is specified here and not
  attempted.

## 8. What was verified here, and what is not

Verified on this workstation (no GPU, as this ticket's instructions require):

| check | result |
|---|---|
| `okc_loop.cli lint` on the new file | PASS, 1/1 clean (base file also PASS) |
| `cu_build.py` all shipped tuples, flag off | 10/10 compile (5 `CUDA_SRC` + 5 `BW_SRC`) |
| `cu_build.py -D OKC_M8=1 [-D OKC_NS=2]`, all tuples | 10/10 compile |
| the launch-bounds pair, both sources, flag on and off | compiles (`__launch_bounds__(NTL, 2)` only under `OKC_M8`) |
| the real M8 source (`ALT_SRCS["helper0"]`, MID flags + `OKC_M8=1 OKC_NS=2`) through nvrtc | compiles (identical PTX size: `moe_fused` is a template body, so ptxas applies the bound only at instantiation) |
| `sizeof(okc::Smem)` | 36,576 B (nvrtc device probe) |
| the edited `okc_smem_bytes` compiled and run under g++ | 214,784 B incumbent / 110,336 B M8 -> 1 vs 2 CTAs/SM |

**Not verified (needs the GPU):** whether two resident CTAs with a two-slot ring is faster than one
CTA with three slots at 4-16 tokens. That is the entire expected gain, and `_m8_tune` measures it on
the real routing before binding anything; nothing binds if it is slower. The candidate cannot ship
incorrect numbers because no fragment, decode, packing, epilogue or routing code differs from the
incumbent — only the ring depth, the shared-memory request and the CTA count.

A note on "bit-exact": no re-tiling of this kernel can be bit-exact against the incumbent, because
phase B accumulates each route's sum of squares with a global `atomicAdd` (line ~933 of the source),
so the order of CTAs is visible in the last bits of `ss` and hence of the output scale. The M8 build
changes nothing but the CTA count, i.e. it is as exact as the change allows; `_M8_TOL = 1e-5` is the
gate, five orders of magnitude below an O(1) tile/mapping error and three above the reordering.
