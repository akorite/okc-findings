# Class-(a) constant sweep off klr_fci5.py (muse_decode_layer)

Base: `src/okc_loop/vendored_work/layer/klr_fci5.py` (979,654 B, sha ee74c9e4…, check 2.27296 /
job a407ad5e63ae4e2d). All variants are byte-copies with **exactly one** `#define` line changed.

## 0. What the board actually measures (from the base's own profile)

12 scored shapes = 6 loads x {sliding (w=46), full (w=16)}; per load the time is one whole 62-layer
pass, so a kernel's reach is `46*sliding + 16*full`. Per-load GPU-time shares computed from
`profiles/klr_fci5.a407ad5e63ae4e2d.txt` (kernel families: `moe_fused`, `attn_kernel`, `sk2_kernel`
+`nvjet`+`rq` = dense GEMM, rest = glue):

| load | moe_fused | attention | dense GEMM | glue |
|---|---|---|---|---|
| decode 1  | 31.5% |  9.5% | 52.5% | 6.4% |
| decode 16 | 44.6% | 17.5% | 31.2% | 6.8% |
| decode 32 | 40.3% | 25.5% | 28.6% | 5.6% |
| verify 1  | 51.4% |  7.5% | 34.9% | 6.2% |
| verify 16 | 57.0% | 13.7% | 12.9% | 16.4% |
| verify 32 | 54.3% | 16.9% | 18.9% | 9.9% |

Two consequences that drive the ranking:
* MoE is the largest single block everywhere, but it is **weight-bandwidth saturated** (T=16 73.6 us,
  T=32 77.1 us, T=256 184 us; the T=1 -> T=16 marginal cost is 3.3 us/token, T=16 -> T=32 is
  0.2 us/token). Only ~3% swings are available from scheduling bits there (measured: WGSFCI 5->6
  -1.5%, WGSFCI 5->4 -1.0%, WGSNX 3->2 -4.6%).
* Attention only shows up in the **full** shape (80k ctx); it is the q_len-1 flat kernel for
  decode 16/32 (25.5% of decode 32) and the q_len-8 flat kernel for verify 16/32 (16.9% of
  verify 32), and it is the only block with real headroom (full-layer attn 171 us against a
  ~110 us KV-read bound at decode 32).

## 1. Class-(a) list: `#define`s whose comment cites a measurement made on another build

| # | constant | value | provenance in the base | consumed at | scored shapes it can move |
|---|---|---|---|---|---|
| 1 | `OKC_NP8` | 128 | "q_len 8: 4 producer warps (its consumers need the registers)" — a register trade-off fixed on the MoE/attn agent's build; the sibling `OKC_NP` was swept 256->128 ("128 since v111: f16 -0.7%, f32 -0.4% vs 256") and then **never extended downward** | `Cfg::NP`, `BASEREGS`/`CREGS` | verify 16, verify 32 (q_len-8 flat kernel, WS8) |
| 2 | `OKC_ST1F` | 3 | "3 measured best: b16 full 93 -> 91.5 us vs 6" — their b16, their register mix (no QDIRECT/PK2/dual build) | `Cfg::STAGES` (L==1, F) | decode 16, decode 32 (q_len-1 flat kernel) |
| 3 | `OKC_NP` | 128 | "128 since v111: f16 -0.7%, f32 -0.4% vs 256" — same truncation, q_len 1 | `Cfg::NP` | decode 16, decode 32 |
| 4 | `OKC_ST8F` | 4 | "pipeline stages of the q_len 8 flat kernel" (no sweep recorded here at all) | `Cfg::STAGES` (L==8, F) | verify 16, verify 32 |
| 5 | `OKC_SEGK8` | 600 | "(b16 107.4 -> 103, b32 187.4 -> 184)" — fitted at b16/b32 on their build | flat partition (`MAXSEG`/segments) | verify 16, verify 32 |
| 6 | `OKC_WGSREORD` | 1 | "phase A starts with the trailing partial unit" — introduced on their stream-K fit | `wgs_split` | decode 32, verify 16, verify 32 |
| 7 | `OKC_STN` | 3 | "per-request kernels: pipeline stages" (unmeasured) | `Cfg::STAGES` (!F) | decode 1, verify 1 |
| 8 | `OKC_GATHB` | 8 | "one x row at a time was one dependent HBM round trip per row … 1.3-7 us at bs256" — their grid | `wgs_gather` | (see exclusion E3) |
| 9 | `-DOKC_UWARP` / `-DOKC_WSBLK` / `-DOKC_L2H` / `-DOKC_CPF` | their small-M builds | `MOE_TINY/TINY2/MID_DEFINES` are the MoE agent's own builds ("their small-M build", "their mid-M build") | MoE T<=16 kernels | decode 1, verify 1, decode 16 |

## 2. Exclusions

* **In flight (peers):** `OKC_WGSFCI` (fci6/fci7 measured: 2.23759 / -1.5%), `OKC_WGSNX` (nx4),
  `OKC_WGSGS` (gs3), `OKC_WGSDFIN` (dfin3/dfin0), `OKC_WGSU`/`OKC_WGSUB` (cost11), `OKC_WGSA`/`OKC_WGSB`
  (same cost table), route PACK (`klr_best`), `klr_pool`.
* **E1 — cannot reach a scored shape (inert):** `OKC_BIGM` (only moves the fat/skinny M=32..128
  threshold), `OKC_DYNK`/`OKC_FCOSTB`/`OKC_FT_T`/`OKC_FCOST*` (fat-job queue: `SKDYN = OKC_DYN &&
  (!OKC_TINY || OKC_TDYN) …` is 0 in every TINY build and the WGS build "has no fat queue";
  `MAIN_FALLBACK_DEFINES`' DYNK=7/COSTB=35 are therefore both inert here), `OKC_SKT` (blocked by
  `static_assert(!OKC_WS || OKC_SKT == 8)` in all three small-M builds), `OKC_NW` on the MAIN build
  (WGS needs `NT = 4 warpgroups = 512` = `NW 16`).
* **E2 — falsified on our build / cited as slower:** `OKC_WGSX` (2.1693 today), `OKC_L2H` at T=1
  (header line 2: "judge d1 25.7-26.6 us with them vs 24.5-25.8 without"), `OKC_W0L1` ("measured
  +3..5 us"), `OKC_W0SPREAD` ("8 sequential polls"), `OKC_SENSE2`, `OKC_RAWTHR`, `OKC_TURN`,
  `OKC_INEW`, `OKC_STSM`, `OKC_EXPPOLY`, `OKC_PTR`, `OKC_F2`, `OKC_S2` (all "measured slower").
* **E3 — inert at our grid:** `OKC_GATHB`. The WGS grid is `nsm` (132 CTAs, smem-bound, one CTA/SM),
  so at T=128/256 a CTA gathers `ceil(R/132)` = 1..2 rows per batch: 8 and 16 rows behave identically.
  Their "bs256" fit ran a different grid.
* **E4 — probe/debug/test hooks:** `OKC_X*`, `OKC_PCHKTEST`, `OKC_W0FORCEBAD`, `OKC_*DBG`,
  `OKC_PROF*`, `OKC_MRGDBG`, `OKC_NOLOAD`.

## 3. Ranked (reach x share x plausible range)

| rank | variant | file | reach | why |
|---|---|---|---|---|
| 1 | `OKC_NP8 64` | `klr_c_np8_64.py` | verify 16/32, attn 13.7/16.9% of load | extends a documented monotone sweep (256->128 won on the twin knob) into the kernel that owns the biggest full-layer attention block; fewer barrier arrivals per stage, `CREGS` 104 -> 112 |
| 2 | `OKC_ST1F 4` | `klr_c_st1f_4.py` | decode 16/32, attn 17.5/25.5% of load | one more 8 KB K/V stage (48 -> 64 KB, still 2 CTAs/SM) for the 80k-ctx stream; the 3 was fitted with their register mix, not ours |
| 3 | `OKC_NP 64` | `klr_c_np_64.py` | decode 16/32 | same mechanism as rank 1 on the q_len-1 kernel (held back: correlated with rank 1) |
| 4 | `OKC_ST8F 5` | `klr_c_st8f_5.py` | verify 16/32 | deeper q_len-8 pipeline; 80 KB of stages still fits 2 CTAs/SM (higher smem risk) |
| 5 | `OKC_SEGK8 700` | `klr_c_segk8_700.py` | verify 16/32 | **falsified while this sweep ran**: peer AttnFlatKnobs measured 720/480 at 2.2411 (< base) — 600 is already past its optimum, the live direction is *down* (`klr_segk_lo.py` 500/340, `klr_segk_min.py` 200/100, their gate) — so this file stays emitted but is not a bet |
| 6 | `OKC_WGSREORD 0` | `klr_c_wgsreord_0.py` | decode 32, verify 16/32 (MoE 40-57%) | largest block, but a symmetric coin flip (REORD=1 was their win) — kept as the MoE fallback |

Measured (quota-sparing): ranks 1 and 2. Ranks 3-6 are emitted, `ast.parse`-clean and ready.

Register arithmetic for rank 1 (`Cfg`, NT=256, MINB=2): `BASEREGS = 65536/(2*NTX)` rounded to 8;
NP8 128 -> `NTX 384, BASEREGS 80, CREGS 104`; NP8 64 -> `NTX 320, BASEREGS 96, CREGS 112`.
The producer-pass assert (`64 % (NP/8) == 0`) holds at 64 (8 rows each), so no code path is lost.
`OKC_NW` was ruled out structurally, not by taste: the WGS path asserts `NT * 8 == HD` (NT=512 = 4
consumer warpgroups), so `NW 16` is a build requirement.

## 4. Gates

All six files: `ast.parse` ok; size <= 979,696 B (limit 1,000,000); line 1 unchanged. sm_90a compile
gate on Colab A100 (`tools/colab_gate.py` + `colab run`), 89-91 s wall each:
`bad = ["_graft_b2_module", "_graft_glue_module", "_sl_cluster_module"]` — exactly the three
`tvm_ffi` glue modules Colab cannot build; `_att7`, `_moe`, `_bf16_proj` and the Triton/extension
glue all built for sm_90a in every variant.

## 5. Results (judge checks, base = 2.27296 / 9951.9 us, job a407ad5e63ae4e2d)

| variant | job | score | verdict |
|---|---|---|---|
| `klr_c_np8_64.py` | e743bf39bbbc435e | none (crash) | **falsified**: `CUDA error: unspecified launch failure` in the attention self-test / `_check_guards()` at `prepare()`, before any timing — the q_len-8 flat producer path needs its 128 lanes (4 warps); 64 is not a supported depth, so the `CREGS 104 -> 112` trade cannot be bought this way |
| `klr_c_st1f_4.py` | e6b9fcdee7dc4bbd | 2.26484 (passed=True) | **-0.36%** vs base: the load it actually touches went decode 16 -0.71% time (better) and decode 32 +1.44% (worse) |

Per-load deltas of `ST1F=4` against base (`ratio` then cand_us):

| load | base ratio | new ratio | d(ratio) | cand_us |
|---|---|---|---|---|
| decode 1  | 2.5604 | 2.5720 | +0.45% | 4252.2 -> 4219.4 (-0.77%) |
| decode 16 | 2.1312 | 2.1385 | +0.34% | 8898.8 -> 8835.2 (-0.71%) |
| decode 32 | 1.9716 | 1.9442 | -1.39% | 11146.4 -> 11307.3 (+1.44%) |
| verify 1  | 2.4961 | 2.4579 | -1.53% | 6697.2 -> 6748.9 (+0.77%) |
| verify 16 | 2.2322 | 2.1825 | -2.23% | 15916.7 -> 16376.2 (+2.89%) |
| verify 32 | 2.3003 | 2.3528 | +2.28% | 21774.5 -> 21831.5 (+0.26%) |
| geomean | 2.27296 | 2.26484 | **-0.36%** | |

### Noise calibration (the sweep's own control)

`ST1F` only exists in the q_len-1 **flat** kernel (decode 16/32). decode 1 (batch 1, per-request) and
all three verify loads (q_len 8) cannot see it, yet they moved -1.53% / -2.23% / +2.28%. So a single
check's per-load noise on this board is **±2%** and its geomean noise is **±0.3-0.5%** — an order larger
than the ±0.2% implied by the ledger's duplicate-sha repeats. Any future one-constant attention bet
needs either a much bigger effect or a paired repeat; MoE knobs (3 loads x 40-57% of time) are the
only class-(a) items whose plausible range clears that floor.

### What this leaves live (all emitted, ast + sm_90a gated, unspent)

`klr_c_np_64.py` (NP 64, same mechanism as the crashed NP8 - expect the same structural failure, keep
only as a 2-minute confirmation), `klr_c_st8f_5.py` (q_len-8 stages 4->5), `klr_c_segk8_700.py`
(falsified upward by AttnFlatKnobs; the live direction is down: their `klr_segk_lo.py` 500/340),
`klr_c_wgsreord_0.py` (the only MoE-side free bit, 3 loads x 40-57%).

