# klr_pf — slot-free L2 bulk prefetch ported into the board layer file's MoE

File: `src/okc_loop/vendored_work/layer/klr_pf.py` = `src/okc_loop/vendored/keell0renz/muse_decode_layer.py`
(sha `82c098ae`, 16345 lines) + this patch, nothing else. The vendored board file itself is untouched
(`git status` clean for `src/okc_loop/vendored/`), so the sha every other note cites stays valid; the
derived copy is the deliverable, as with `klr_qd.py` / `klr_l1.py`.
Reference patch: `artifacts/cas/moe_pf.py` (standalone MoE, same four hunks).

## What was added

1. `ring_pf<PB>(p, S, v)` + `#ifndef OKC_PF / #define OKC_PF 0` immediately after `ring_issue_all`
   in `MOE_CUDA_SRC` (line 4434) and, byte-identical, in the `_MA_ALT_SRCS["moe_helper0"]` copy.
   `cp.async.bulk.prefetch.L2.global [src], UNIT_BYTES` — no slot, no mbarrier, pure hint.
2. Call sites, both copies (line numbers are `klr_pf.py` in `MOE_CUDA_SRC`):
   * (a) skinny consumer, inside `if (tid == 0) { ... }` after the refill: `4651`
     (`if (i + OKC_NS + OKC_PF < nU) ring_pf<PB>(p, S, v + OKC_NS + OKC_PF);`)
   * (b) the `ring_issue_all` refill site: `4821` (`if (threadIdx.x == 0 && ...)`)
   * (c) `ws_issue`, slot-free lookahead bounded by `qp < nA + (S.wsr[3] - b0)`, A/B pointer picked
     by `qp < nA`: `5454`
   * (d) WGS producer right after `const unsigned char* src = base + ...`: `5751`, prefetching stage
     `gs + OKC_PF` when `< g1`.
3. Host: `PF_DEPTH = 4` next to the `MOE_*_MAX_M` knobs (`6928`) and
   `-DOKC_PF={PF_DEPTH if major >= 9 else 0}` injected in `_ma_ext`'s `extra_cuda_cflags` (`6981`),
   exactly like `-DOKC_NS=`. No `*_DEFINES` tuple got `OKC_PF`, so the load_inline module name (which
   `_ma_ext` hashes from `MOE_CPP_SRC + src + defines`) and the `_MA_EXTS` keying are unchanged.
   `-DOKC_PF=` in a defines tuple would still win (the injection is `[] if ... startswith`-guarded).

The helper0 copy is klr's original text (`OKC_WGSFINSS`/`TIDX`, no SC6/MFIN) and differs from
`MOE_CUDA_SRC` in two spots this patch touches: it guards with `TIDX == 0` and its WGS producer stages
are `UNIT_BYTES`, not `WGS_STG`. Both handled (own guard token, own stage size). Because that copy is a
140 KB single-line Python literal, it is kept **verbatim** and the four hunks are applied at import by
`_pf4()` (the `_df_patch` idiom already in the file), with `count(anchor) != 1 -> AssertionError` so a
future text change cannot silently leave the MID/TINY builds unprefetched. `_pf4` runs before
`@moe_helper0` is added to `MOE_MID_DEFINES`/`MOE_TINY*_DEFINES` and before `_df_patch` derives
`@moe_helper0df` (the string the MID build actually compiles), so the patch reaches the df copy too.

## Gates (all offline; no `okc` command run)

| gate | result |
| --- | --- |
| `ast.parse` | ok |
| `okc_loop.cli lint klr_pf.py` | **30 errors, unchanged** vs the board file (same 12 shown + 72 more; they are line-shifted only: `threading`, `os.environ`, `globals`, `ctypes`, `time.time`, the 140927-char helper0 line, the `_glue_bind_context` graph branch). No new class of error. |
| `cu_build --variant MID/MAIN` on `MOE_CUDA_SRC` | all 6 MID-selected / 6 MAIN-selected define sets `ok`, each with `-D OKC_PF=4` and without (i.e. `OKC_PF=0` default). Includes the `OKC_PRMTSC=1` and `MAIN_FALLBACK` sets. |
| `cu_build.to_ptx` on the decoded `moe_helper0` | MID, TINY, TINY2 define sets x {`OKC_PF=0`, `OKC_PF=4`}: all COMPILED. Same for `moe_helper0df` (`_df_patch` applied, `OKC_DF=1`) — that is the string the MID build really compiles. |
| prefetch reaches both copies | both: 4 `ring_pf<...>(p, S, ...)` call sites, 1 `ring_pf` definition, 3 `#if OKC_PF > 0` guards, 1 inline asm inside `ring_pf` + 1 inline in the WGS producer; every `ring_pf` occurrence lies before the `#include <torch/` cut (compiled region). |

Other `ok`/`FAIL` lines in the `cu_build` sweep are the tool's known limits for the fragment sources
(`bfp_*`, `_GRAFT_*`, `_SL_GLUE_CUDA`, `_ROUTER_FP32_CUDA`: `nv/target`, `cudaGetLastError`); identical on
the pristine board file.

## Does the PTX contain the instruction?

The MoE `__global__` is `template <bool BIG> moe_fused`, instantiated by the host wrapper **below** the
`#include <torch/extension.h` cut, so a device-only compile emits no entry at all (cu_build's documented
limit; nvrtc here drops kernels whenever `<stdint.h>` is in the TU, which is why an appended probe kernel
is not emitted either). So the shipped PTX cannot show the instruction, and the gate proves it directly
instead: the file's own `ring_pf` text, extracted verbatim into a 20-line stub TU, compiles to

```
cp.async.bulk.prefetch.L2.global [%rd1], %r2;      # OKC_PF=4, ring_pf<false>
cp.async.bulk.prefetch.L2.global [%rd2], %r2;      # OKC_PF=4, ring_pf<true>
```
with `OKC_PF=0` emitting **0** instructions, and `ptxas -arch=sm_90a` assembles the PTX fine. The WGS
producer's own inline asm (`sp2` / `WGS_STG`) likewise emits 1 instruction and assembles.

## Not checked

Nothing measured on a GPU: PF_DEPTH = 4 is the standalone MoE's tuned value (its measured band was
3.5-3.8 TB/s against the 4.287 TB/s cold-read wall), not a layer measurement. If it helps, the open knob
is `PF_DEPTH` (2/4/8) via the same `-DOKC_PF=`; `PF_DEPTH = 0` restores the unpatched build bit-for-bit.
