# kitchen_sink: every measured non-negative layer lever in one build

Candidate: `src/okc_loop/vendored_work/layer/kitchen_sink.py` (999,963 B; line 1
`# okc: problem=muse_decode_layer`; byte-identical to `layer_stack2.py` except the header and the four
`#define` lines in §2).

Base: `layer_stack2.py` = SC6S + MOEDUP + vjeux `m==1` GEMV (`MOEDUP=True`, `SC6_SKINNY=True`,
`GEMV_M1=True`, `XRES_L2=False`, confirmed at `:6930-6938`). Nothing else was carried over: `XRES_L2`
stays off, `OKC_PVW 0`, `OKC_NP8` untouched, and `layer_st2`'s prepare-time tuner is not in this file
(`layer_stack2` never had it).

## 1. Flag table

Every row is a lever that was either measured non-negative or is a documented win elsewhere. "µs" is
against the board's control mean **10,038.8 µs** (three code-identical draws of the row file,
SD 1.17%).

| # | flag | value here | lever | evidence | kind | Δ µs | risk |
|---|---|---|---|---|---|---|---|
| 1 | `SC6_SKINNY` → `-DOKC_SC6S=1` (Python, already on) | True | 6-bit NVFP4 block scales on the three skinny MoE builds (4,096→3,072 B per block) | `layer_moe_bytes.md` §A: −2.78% of every byte the skinny loads read, **+0.68% predicted**; isolated run `f41d1b56` read **+1.21% vs the A/A quiet level** but **−0.99% vs the row** | derived/measured, unresolved sign | −121.5 (at +1.21%) | probe-gated: `_sc6s_probe` must round-trip the 3,072-byte planes exactly and match the kernel bit-for-bit, else the lever clears itself and the 4 KB copies come back |
| 2 | `MOEDUP` → `-DOKC_MOEDUP=1` (Python, already on) | True | L2 evict-last hint on the first copy of a duplicated expert's tiles (WGS builds only) | `layer_moedup.md`: identical mechanism measured **+0.27%** on the standalone MoE board; layer bound **≈+0.5%** (1.44 duplicated streams / 40 MB) | derived | −50.2 (at +0.5%) | hints only, no numerics; only T≥128 has any duplication (1.000 below), so it is a no-op on the d1 cells |
| 3 | `GEMV_M1` → `-DOKC_GEMV_M1` (Python, already on) | True | vjeux's 12-bit scalar GEMV for the `m==1` qkv / o / unpooler | `layer_gemv_m1.md`: upstream per-kernel replay −0.58 / −0.45 / −0.71 µs = **−1.74 µs on d1 = +0.39%** | derived (upstream measured) | −39.2 | m>1 and misshapen matrices keep sk2; the kernel routes by shape |
| 4 | `OKC_WGDEC` | **1 → 0** | q_len 8 producer: all consumer warpgroups sync as one block per stage instead of per KV head | `sweep_b3_wgdec0.py`: **10,010.0 µs = −0.29%** vs control | **measured, 1 draw** | −28.8 | 0.25σ of an 1.17% SD: indistinguishable from zero on its own; correctness exercised by the layer judge in the sweep |
| 5 | `OKC_TMAPG8` | **0 → 1** | q_len 8 flat kernel fetches page indices by TMA (one bulk copy per 64-key tile, PD=3 ahead, PS=4 slots) instead of 32 `__ldg` per pass | `sweep_b1_tmapg8.py`: **10,043.9 µs = +0.05%** (neutral here) but documented **≈+0.7%** for the q8 flat kernel elsewhere | measured (neutral) + documented | −5.0 (at +0.05%) | extra smem + TMA barrier state in the flat kernel; the q1 path already runs this mechanism |
| 6 | `OKC_QP16` | **1 → 0** | q_len 8 QDIRECT q pass: eight threads per row (8 dims, 3 shuffles, 8-byte e4m3 store) instead of four (16 dims, 2 shuffles, 16-byte store) | `sweep_b6_qp16_0.py`: **9,984.6 µs = −0.54%** — best of the sweep | **measured, 1 draw** | −54.2 | 0.46σ at one draw; a pure instruction-count/occupancy trade in the q pass, no numeric change |
| 7 | `OKC_FINWARM` | **0 → 1** | dry call of the `__noinline__` last-CTA finalize (`fin_nc1`, `dry=1`, every store predicated) during the prologue, so the judge's cold-L2 replay does not pay the i-cache fetch on the once-per-CTA path | `attn_fill1.py` (the whole delta of that file is this flip plus a default-off `OKC_BULKRED`): **measured +0.16%** on the layer board; `attn_fill.md` §5.1 prices the cold-code row at 1.1–2.0 µs | measured | −16.1 | zero numeric risk (dry call, predicated); guard is `!FLAT && cs<=1 && !coop && nc==1`, so it fires only on the cells where that path is taken |

Excluded, as briefed: `OKC_PVW 1` (10,330.0 µs, +2.9%), `OKC_NP8 256` (failed correctness), `OKC_WS8 0`
(10,060.0, +0.21%), `XRES_L2` (measured harmful), `layer_st2`'s tuner (declined every swap, cost prepare
time), `OKC_BULKRED` (attn_fill1 ships it default-off; not measured).

## 2. The diff (only intended lines)

```
$ diff layer_stack2.py kitchen_sink.py
2c2,3
< # layer_stack2: layer_sc6s_dup (SC6S + MOEDUP, XRES off) + vjeux's m == 1 cgemv for qkv / o / unpooler at T = 1.
---
> # kitchen_sink: layer_stack2 (SC6S + MOEDUP + GEMV_M1) + WGDEC 0 + TMAPG8 1 + QP16 0 + FINWARM 1.
> # Each addition is one #define (its own switch): OKC_WGDEC, OKC_TMAPG8, OKC_QP16, OKC_FINWARM below.
151c152
< #define OKC_QP16 1      // q_len 8 QDIRECT q pass: four threads per row (16 dims each)
---
> #define OKC_QP16 0      // q_len 8 QDIRECT q pass: eight threads per row (8 dims each; OKC_QP16 0 measured -0.54%)
158c159
< #define OKC_TMAPG8 0    // ... for the q_len 8 flat kernel too (measured ~+0.7%)
---
> #define OKC_TMAPG8 1    // ... for the q_len 8 flat kernel too (measured ~+0.7%; layer board neutral)
162c163
< #define OKC_WGDEC 1     // q_len 8 producer mode: the two consumer warpgroups sync separately (one per head)
---
> #define OKC_WGDEC 0     // q_len 8 producer mode: all warpgroups sync as one block per stage (0 measured -0.29%)
198c199
< #define OKC_FINWARM 0   // dry call of the last-CTA finalize during the prologue (instruction cache warm-up)
---
> #define OKC_FINWARM 1   // dry call of the last-CTA finalize during the prologue (instruction cache warm-up; measured +0.16%)
```

Each is its own `#define`, so any one can be reverted by flipping it back (and `-D OKC_*=` overrides are
how the sweeps varied them). The code bodies of all four are **byte-identical** to the file each came from:
`diff` of the `OKC_QP16` region vs `sweep_b6_qp16_0.py`, the two `OKC_WGDEC` sites (`:2156`, `:2279`) vs
`sweep_b3_wgdec0.py`, and the `TMAPG8` sites (`:1226`, `:3431`) vs `sweep_b1_tmapg8.py` were all clean; the
`FINWARM` body is the same three lines as `attn_fill1.py:1854-1856`. Nothing else from `attn_fill1.py`
(the `OKC_BULKRED` helpers and staging branches) was ported.

## 3. Verification

**Lint** (`PYTHONPATH=src .venv/bin/python -m okc_loop.cli lint <file> --limit 500`):

```
layer_stack2.py : FAIL tier0 muse_decode_layer: 30 error(s)   0/1 clean
kitchen_sink.py : FAIL tier0 muse_decode_layer: 30 error(s)   0/1 clean
```

Finding **multiset identical**: normalizing `line N:` in both outputs, `diff` is empty except the file
name and the char-count self-reference inside the long-line finding:

```
11c11
<   [error] portability: line N: line 12952 is 140927 chars (limit 400)
>   [error] portability: line N: line 12953 is 140927 chars (limit 400)
```

(The 30 findings — `threading`/`ctypes`/`os.environ`/`time.time`/`globals`/`locals`, the 140,927-char line,
and the `_glue_bind_context` value branch — are all pre-existing in `layer_stack2.py`, unchanged here.)

**Build** (`.venv/bin/python tools/cu_build.py <file>`), 70 entries, status compared after substituting the
module name in the emitted tag:

```
STATUS IDENTICAL   (16 ok, 54 FAIL — the same 54 in both files)
ok   ATT7_CUDA_SRC      ×6 tuples   <- the four new flags live here; this is the source they touch
ok   MOE_CUDA_SRC       ×4 tuples
ok   GEMV_M1_SRC        ×6 tuples
FAIL bfp_TMA_SRC / bfp_GEMM_SRC / bfp_SK2_SRC / bfp_WIDE_SRC / bfp_GQ_SRC /
     _GRAFT_GLUE_CUDA / _GRAFT_B2_CUDA / _SL_GLUE_CUDA / _ROUTER_FP32_CUDA   ×6 each
```

The 54 FAILs are the base's own: those host-side sources call `cudaGetLastError()` without the CUDA runtime
include, so `nvrtc` rejects them identically in `layer_stack2.py`. All six `ATT7_CUDA_SRC` tuples (the
attention source carrying `OKC_WGDEC`/`OKC_TMAPG8`/`OKC_QP16`/`OKC_FINWARM`) compile `ok` in both files, and
both compile to *template body* (no kernel emitted) because the kernel is instantiated by the host wrapper —
the status is what is being compared, as briefed.

**Stacked defines** (`.venv/bin/python tools/cu_build.py <file> -D OKC_MOEDUP=1`): all 70 entries, status
identical between base and candidate (`54 FAIL`, the same pre-existing set); every `MOE_CUDA_SRC` and
`ATT7_CUDA_SRC` entry `ok`, so `-DOKC_MOEDUP=1` on top of the four new defines compiles.

Not run (not available/not permitted here): any GPU run, `okc check`, `colab`, `judge`.

## 4. Expected value — a bet on additivity, not a certainty

Arithmetic sum if the levers add, applied to the 10,038.8 µs control mean:

| component | Δ | source |
|---|---|---|
| SC6S | +1.21% | single run vs the A/A quiet level (sign unresolved in the dataset) |
| MOEDUP | +0.50% | derived bound; +0.27% measured on the identical mechanism |
| GEMV_M1 | +0.39% | derived from upstream's measured per-kernel replays |
| WGDEC 0 | +0.29% | measured, 1 draw |
| QP16 0 | +0.54% | measured, 1 draw |
| TMAPG8 1 | +0.05% | measured, 1 draw (neutral) |
| FINWARM 1 | +0.16% | measured |
| **sum** | **+3.14%** | **≈ 10,038.8 × 0.9686 = 9,723.5 µs ≈ 2.2710** |

That number would clear #100 (9,886.0) and #50 (9,649.4 it only approaches), i.e. it is not a
top-20 claim: at +3.14% it lands around 2.27, still short of the 2.3900 top-20 bar that costs +4.65%.

**The honest caveat the whole exercise turns on:** our own per-lever decomposition has already been
refuted once on this board. `layer_sc6s_dup.py` (SC6S + MOEDUP) was predicted at **+3.7%** over the quiet
level and measured **+0.19%** (`docs/PROGRESS.md`, 2026-11-03 ~12:50 UTC); eight candidate files spanning
1.3% are explained entirely by σ≈2.2% noise. Each of the three new sweep readings (QP16 −0.54%, WGDEC
−0.29%, TMAPG8 +0.05%) is within 1σ of zero at one draw each, and two of the three existing levers are
derived, not layer-measured. So:

* if additive → **≈9,724 µs (+3.1%)**;
* if the stack behaves like the last one measured → **≈+0.2–0.8%, ≈9,960–10,020 µs**, i.e. the same band
  as the control;
* the only justification for building it is that it costs no run to *build* and the incumbent's own
  methodology is "small and stacked": every included lever is non-negative on its own evidence, so the
  stack's sign is wrong only if the levers anti-correlate — which the one direct stack measurement
  (SC6S+MOEDUP, −1.0% when combined) says is possible.

Recommendation: submit it as one of the queued draws; at ~±1.1% per 4-draw mean, a 3-draw mean around
9,720 would be the first signal on this board that survives the instrument.
