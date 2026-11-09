# moe_klrbands.py — keell0renz's in-layer MoE build knobs on the standalone MoE board

Candidate: `src/okc_loop/vendored_work/moe/moe_klrbands.py`
Base: `artifacts/cas/6402f5f3…b5d2b3e5.py` (standalone MoE incumbent, score 2.3863)
Compared against: `src/okc_loop/vendored/keell0renz/muse_decode_layer.py:3880-6862` (MoE_CUDA_SRC) and its host
(`:6870-7210`, `:10210-10235`, `:11235-11260`).

Method: extracted both `CUDA_SRC` strings and diffed them (49 hunks, 2698 vs 2982 lines), plus the module-level
define tuples and `prepare()`/`_ext` host code.

## 1. Knob table

| knob | ours (standalone MoE) | layer | who is newer |
|---|---|---|---|
| `TINY` tuple (M<=4) | NW=8 TINY=1 EPI4=0 FAST1=2 L2H=0 NACC=4 UWARP=0 WS=1 WSB=1 WSBL=2 WSBLK=2 | identical (11 flags; **neither** has `FINPF`) | same |
| `TINY2` (M<=8) | same + WSBLK=1 FINPF=1 | identical | same |
| `MID` (M<=16) | NW=8 TINY=1 SPLIT=1 CPF=0 EPI4=0 FAST1=2 NACC=4 UWARP=0 WS=1 WSB=1 WSBL=2 WSBLK=2 FINPF=1 | identical | same |
| `MAIN` (M>=17) | NW=16 FAST1=0 EPI4=0 WGS=1 | identical | same |
| `MAIN_FALLBACK_DEFINES` | CPF=0 CPFF=0 DYNK=7 EARLYN=2 EPI4=0 FCOSTB=35 FCOSTBB=36 FINCOL=0 | same + `"-DOKC_FPF=5"` | layer — **inert**: no line of either kernel source reads `OKC_FPF` (`grep -c` = 0), so it changes nothing |
| `b4` band | absent: 17..32 runs plain `MAIN` | `16 < M <= 32` → `MOE_DEFINES + -DOKC_B4FIN=1`, gated on `_main_defines(sfx) == MOE_DEFINES` (PRMT decode available) | layer |
| `OKC_B4FIN` | absent from source | default 0; on in the b4 build. Column-parallel final pass, `Params::sspart`, `B4_EARLY_Y=1` | layer (large code, not a knob) |
| `OKC_MFIN` (+`MFP=15`, `MFQ=47`, `MFS=0`, `MF_B4L=0`) | absent | default **1** (WGS only): per-expert completion counters + launch-parity `mfl` region instead of the grid barrier | layer (large code, not a knob) |
| `OKC_SC6S` | absent | define 0 + `UNIT_SC = 3072`, `SCDK`→k4, `sc_load`; `_sc6_unit` written but **never set by prepare()** — and the fat-unit scale section is not packed, so a TINY/MID build with it would fault | layer; ports as `OKC_SC6S` (kept off) |
| `OKC_SC6W` | absent | define 0, auto `-DOKC_SC6W=1` on WGS builds when every scale byte ∈ 0x40..0x7F **and** the PRMT decode is on; `WGS_PIECE 8960`, `WGS_STG 35840`, `sgs[4]` decode, `_sc6_wgs` | layer, live |
| `OKC_WGSFCI` | 3 | 4 (st2/E374) | layer; already ported by peer `moe/moe_cepi.py` |
| `OKC_WGSU`/`WGSUB` | absent (no CEPI table) | 160/140 + `wgs_unitc`, prefix/`stage_at`/`cta_of` cost tables, warp-parallel `wgs_ranges` | layer; peer `moe_cepi.py` ports the *board's* 80/70 (not the layer's) |
| `OKC_YBF` | absent | `#if !OKC_B4FIN → 1`: WGS ybuf in bf16 | layer; peer `moe_ybfl.py` |
| `.ns4` / `NS`, `NSF` | `MAIN_NS=3`, `NSF=3`, `NS=1` below sm90 | `_MA_MAIN_NS=3`, `NSF=3`, same rule | same |
| `OKC_FNST` | not passed | `-DOKC_FNST=4` (sm90) | layer-only string — **inert** (no kernel line reads it) |
| `DQ7`, `PRMTSC`, `WGS_NS=3`, `WGSNX=3`, `WGSGS=2`, `WGSDFIN=2`, `WGSREORD=1`, `GATHB=8`, `WGSPR=1`, `WGSA/WGSB` tables, `WGMIN=9`, `WSNR=4` | — | identical | same |

## 2. What was adopted (flags behind `#ifndef`, default build byte-identical)

`OKC_SC6W` (host `SC6_WGS = True`), `OKC_SC6S` (host `SC6_SK = False`), plus the `WGS_STG` stage stride it needs.

Device (`CUDA_SRC` only — `ALT_SRCS["helper0"]` is byte-identical to the incumbent, as upstream leaves it):
* `#ifndef OKC_SC6S/OKC_SC6W` + `UNIT_SC = OKC_SC6S ? 3072 : 4096`.
* `SCDK` SC6S override (k4 + 0x6000), `#define SCK ((OKC_SC6S ? 4u : 16u))`, `sc_load()` (3 planes → 4 words).
* 3 `k16 = 16u + p.zero` → `SCK + p.zero` (skinny dyn, skinny static, `skinny_ring_ws`); 3 `uint4 sc = *(slot + UNIT_CODES + warp*512 + lane*16)` → `sc_load(slot, warp, lane)` (same 3 functions + `fat_compute`).
* `WGS_PIECE = OKC_SC6W ? 8960 : 9216`; `#define WGS_STG ((OKC_SC6W) ? 35840 : 36864)` (macro, so the default build emits no extra `.global` symbol). WGS producer: src stride, `expect_tx` and `cp.async.bulk` size `UNIT_BYTES` → `WGS_STG`; the smem slot keeps `UNIT_BYTES` (35840 ⊂ 36864, pieces at +0/+8960/+17920/+26880).
* `wgs_seg`: `k16c = (OKC_SC6W ? 4u : 16u) + p.zero`; `#if OKC_SC6W` builds the 4 scale-pair words `sgs[4]` from the 768-byte plane W/H block (`pc + 8192 + tt*4`, `pc + 8192 + 512 + tt*2`) and `sgp = sgs[js]`, `#else` keeps the `uint2 scs` + PRMT/IMAD path verbatim.

Host:
* `_sc6_ok` (bytes ∈ 0x40..0x7F), `_sc6_unit` (4096→3072), `_sc6_wgs` (1024→768) — **AST-identical to the layer's**.
* `_unit_order(codes, scales, sc6=False)`, `_wgs_copy(codes, sf, rows, sc6=False)`.
* `prepare()`: `_SC6W = SC6_WGS and prmt and _wgs_on() and _sc6_ok(w13_sf, w2_sf)` computed *before* packing (their gate: SC6W rides the WGS build, which their kernel only ships with the PRMT decode); `_SC6SK` likewise with an assert that forbids the skinny packer next to any fat-capable build.
* `_ext()`: appends `-DOKC_SC6W=1` only to tuples that already contain `-DOKC_WGS=1` (so `MAIN_FALLBACK` — no WGS — can never see it: the layer's fallback escape is implicit), `-DOKC_SC6S=1` only for non-`ALT_SRCS` builds; both respect `-DOKC_SC6OFF=1` for an A/B twin. The define tuple itself is untouched.

## 3. Gates (offline, this workstation; no okc/GPU/judge)

* `PYTHONPATH=src .venv/bin/python -m okc_loop.cli lint` → **PASS** (1/1 clean).
* `tools/cu_build.py` → **PASS**, all 10 builds (CUDA_SRC + BW_SRC × MAIN/MAIN_FALLBACK/TINY/TINY2/MID), also with `-D OKC_SC6W=1` and with `-D OKC_SC6S=1`; ptxas assembles every one (registers/spills not reported: the `__global__` is a template instantiated below the `#include <torch/` cut — the documented cu_build limit).
* **Default build is byte-identical**: nvrtc PTX of `CUDA_SRC` for all 5 tuples × {no flag, `OKC_PRMTSC=1`, `OKC_NS=4`} is character-identical to the incumbent's (15/15). Only the SC6W build differs, as intended.
* `ALT_SRCS["helper0"]` string: identical to the incumbent's (the 3 accidental in-string edits during the port were reverted).
* SC6 layout check (pure-python model of host pack → device decode, all 6-bit values 0..63): the WGS path reproduces the incumbent `SCD` value `(byte<<4)+0x5C00` for **1024/1024** (thread, scale) slots and the skinny `sc_load` path **256/256** — the repack is exact and the two sides agree on the unit stride.

## 4. Left off, and why

| knob | why not enabled |
|---|---|
| `OKC_MFIN` (+MFP/MFQ/MFS/B4L) | not a build flag: it rewrites the final pass, `Params`, scratch layout and the launch-parity protocol (~150 lines across `run_phase`, `wgs_epi`, `wgs_flush`) and interacts with `OKC_YBF`. A wrong guess here faults or hangs, and nothing offline can validate it. Needs its own candidate + a GPU A/B. |
| `OKC_B4FIN` / the 17..32 band | same class: column-parallel final pass + `sspart` scratch; a separate candidate (it also forces `YBF=0`). |
| `-DOKC_FPF=5` (MAIN_FALLBACK) | dead define on both sides — no kernel line reads `OKC_FPF`; adding the string would only churn the compiled-module key. |
| `-DOKC_FNST=4` | dead string in the layer too (no `OKC_FNST` in its source). |
| `OKC_SC6S` | upstream never sets it and the fat-unit scale section stays 4096 B, so a TINY/TINY2/MID build would read a 3072-B section as 4096 B and fault. Ported (device + `_sc6_unit` + `_unit_order`) but `SC6_SK = False`, with an assert that keeps it away from fat builds. |
| `OKC_WGSFCI=4`, `OKC_WGSU/WGSUB`, `OKC_YBF` | already owned by peers (`moe_cepi.py`, `moe_ybfl.py`); combining them with SC6W is the main agent's call. Note the layer's CEPI costs are 160/140, the board's are 80/70 — `moe_cepi.py` ported the board's. |

## 5. Risks

* SC6W only touches the **WGS** build, i.e. only helps when M >= 17; if the board's timed M is <= 16 the flag is inert (but consistent: the WGS copies are only read by WGS builds, and the gate keeps the packing and the define in lockstep).
* SC6W changes the WGS weight array's *global* stride (36864 → 35840). Every consumer was audited: `WGS_STG` in the producer, `WGS_PIECE` for the per-warpgroup piece, `UNIT_BYTES` kept for the smem slots and `xring`. A missed site would be an OOB read, not a wrong number.
* `wgs_seg` holds 4 extra `uint32` (the `sgs` words) across the stage loop; the WGS kernel is register-bound (96 registers at 544 threads). SC6W also drops the per-stage `uint2` load and 4 PRMT+IMAD per stage, so it may net out, but only a GPU run can tell.
* No `okc check`/`submit`/`login` and no GPU run were performed.
