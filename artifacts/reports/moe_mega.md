# mega: one-check re-tune of the MoE's compile-time knobs

File: `src/okc_loop/vendored_work/layer/klr_mega.py` (= `klr_qd3.py` + one appended block, nothing above
it modified). The vendored constants were swept on keell0renz's own build/state, so they are re-swept
here on the judge, not trusted.

## Mechanics (confirmed, not assumed)

* `_ma_ext(defines)` (6912): `defines` = the band's tuple + `_MA_SFX` (+ `-DOKC_SC6W=1` when `_MA_SC6W`,
  + `-DOKC_DQ7=1`), compiled with `load_inline(name=f"glue_moe_{source_tag}_sm{arch}", ...)` where
  `source_tag = md5(MOE_CPP_SRC + src + repr((defines, _MA_MAIN_NS)))[:16]`. The module name is a hash of
  the define tuple, so **any** new define gives a distinct module/PyInit automatically -- no tag
  collision is possible, and `_MA_EXTS` caches on the same tuple.
* `_moe(M, state)` (13515) picks the band from M -- tiny `<=4`, tiny2 `<=8`, mid `<=16`, b4 `16<M<=32`
  (only when `_main_defines(sfx) == MOE_DEFINES`), else main = `_main_defines(sfx)` -- then
  `state["ma_ext"][key]`, building on first use. The later override (16389) returns
  `state["ma_ext"][state["moe_force"][M]]` when that key exists, i.e. **the per-shape pick is a string
  key in `moe_force`, resolved at launch**, so a variant only has to be registered in `ma_ext`.
* The scored token counts 1/8/16/32/128/256 therefore run tiny / tiny2 / mid / b4 / main / main; the
  three swept knobs are WGS-only, so they can only move b4 (T=32) and main (T>=128).

## Variants (6 compiles: 3 defines x 2 bands, one define off the incumbent)

| band | variants | base define |
|---|---|---|
| b4 | `b4_fci3`, `b4_fci5`, `b4_nx2` | `MOE_DEFINES + -DOKC_B4FIN=1` |
| main | `main_fci3`, `main_fci5`, `main_nx2` | `_main_defines(sfx)` |

Legal by construction: `OKC_WGSFCI` 3 and 5 are both inside the stage's 8 commit groups
(`ci = lg * (4/OKC_WGSGS) + h`, GS=2) so the deferred `wgs_flush` still runs; `OKC_WGSNX=2` satisfies
`static_assert(BX_E + WGS_NX <= 24)` (8+2) and shrinks smem, never grows it.

Out of the 6-compile budget, and why: `OKC_NS=4` needs one build **each** for tiny/tiny2/mid (3 more)
and was already measured worse on the standalone MoE; the `OKC_WGSU/OKC_WGSUB` pair needs 2 more
(0.9x = 144/126, 1.1x = 176/154, both legal -- that static_assert only wants 1..255); `FT_T` is not a
define swap: it needs `WGS_NT = FT_T/8` re-derived through `WGS_XS = WGS_NT * 4096`, `WGS_FIXW`,
`JMAX`/`EMAX` and the `wgs_cost` n-tile table (`OKC_WGSA`/`OKC_WGSB`) rescaled -- ring geometry, one
check of its own.

## Tournament (prepare time)

After the parent prepare fixed `_MA_SFX`/`_MA_SC6W` (so the variant modules read the same weight
packing as production), the six builds start together on daemon threads; the tournament runs only if
all are joined (a background `cuModuleLoad` must not overlap a graph capture). Per scored T whose band
has variants: the incumbent band build and each variant are timed through `_after_attention` with the
file's own cold-L2 idiom (`_selftune_time`: 256 MB L2 flush, CUDA-graph replay, alternating order,
12 rounds). Gate: rel-L2 <= 5e-3 on the float outputs, integer outputs (ids) exactly equal, median
<= incumbent - max(4%, 3 us), and >= 75% round wins. Prints one line per (T, variant):

    [mega] T=256 band=main base    41.2 us | main_fci3    40.1 us (-2.7%) eq=1 win=0.67 gate=no

plus `[mega] T=.. pick=..`, `[mega] N/6 variant builds in Xs (prepare at Ys)` and `[mega] tune done at
Zs`. Every failure path falls back to the incumbent band; one shape's exception never costs the others.

## Gates run offline

* `ast.parse`: OK.
* `okc_loop.cli lint klr_mega.py`: **byte-identical output to the base file** (same 30 tier0 errors at
  the same lines; the appended block adds none).
* `tools/cu_build.py klr_mega.py --variant MAIN -D <each of the 3 defines>`: `MOE_CUDA_SRC` compiles ok
  in all 7 define sets for each variant (identical to baseline) and the FAIL set is unchanged --
  `_GRAFT_*`/`_ROUTER_FP32`/`_SL_GLUE` + the 5 `bfp_*` sources, all failing on headers this container
  lacks (`nv/target`, `tvm/ffi/...`, `cudaGetLastError`). So this confirms syntax and every
  `static_assert` for the variant values, **not** registers: that build is a template body, and nvcc
  host codegen is not available here, so the real compile time is unknown offline.
* Logic smoke test with stubbed torch: gate/pick/print/dict-merge paths and the per-T exception
  isolation exercised; it caught one real bug (`dict(m, **{int_key: v})` -> `{**m, T: v}`).

## What the printed table tells us

Per shape, whether klr's FCI 4 (their sweep, our feature mix), their 3-slot x ring, are still the right
values here -- and the medians quantify the win. A `pick` line means that shape now runs the variant
build; `gate=no` with `eq=1` means the define is irrelevant at that shape (no regression, no gain);
`eq=0` means the variant changes numbers and it is refused.
