# sweep_b — the q_len-8 / verify attention path and the page-index machinery

Base: `src/okc_loop/vendored_work/layer/klr_fci5.py` (979,654 B, row 2.28370 / 9,905.1 µs, #88/1227,
profile job `9743b6e71fbd42fd`). Seven candidates, each a byte-copy + one banner line (line 2) + the
intended `#define` line(s). Board σ ≈ 1.17%, so a 4-draw mean resolves ~1.15%; single draw ~2.3%.

## 0. What these knobs can reach (from the row's own profile, not from the old share table)

Per-shape GPU time of the q8 kernels in the scored verify shapes:

| scored shape | q8 kernel | µs | share of that shape |
|---|---|---|---|
| verify 1, sliding | `attn_kernel<8,false,15>` | 6.6 | 5.4% |
| verify 1, full | `attn_kernel<8,false,11>` | 14.2 | 10.9% |
| verify 16, sliding | `attn_kernel<8,false,3>` | 11.7 | 4.8% |
| verify 16, full | `attn_kernel<8,true,0>` (**flat**) | 102.0 | **31.4%** |
| verify 32, sliding | `attn_kernel<8,false,3>` | 15.1 | 5.0% |
| verify 32, full | `attn_kernel<8,true,0>` (**flat**) | 181.7 | **37.6%** |

Load-level leverage (share of a whole scored load, both shapes weighted by their times):

| load (T) | flat q8 kernel | per-request q8 kernels |
|---|---|---|
| verify 8 | 0 | 8.2% |
| verify 128 | **20.3%** | — |
| verify 256 | **24.8%** | — |

Score leverage (score = geomean of the 12 shape ratios):
`d ln score = -(δ_kernel/12) · Σ share` → **0.0575 per unit δ** for a flat-q8 change, **0.0136** for a
per-request-q8 change. So to move the row by the campaign's 1.15% resolution a *flat*-q8 knob must be
worth **~20%** at kernel level, and a per-request one **~85%**. Two consequences kept in the ranking:

* no single-flip q8 knob in this family is mechanically capable of clearing S2's 1.15% on its own;
* the only candidates worth a draw are the ones whose *structural* range plausibly reaches ±10% on that
  kernel (B5, B4), and the only way they pay is a 0.3-0.7% row move — i.e. this family supplies lottery
  tickets, not the campaign winner. That is stated up front so the checks are spent knowingly.

## 1. Candidates

`δ_k` = expected change of the q8 kernel's own time (negative = faster). Load numbers are *time* deltas
(`+` = slower). All expectations are **[ESTIMATE]** unless the source cites a measurement.

| # | file | defines | mechanism | verify 8 (T=8) | verify 128 (T=128) | verify 256 (T=256) | verification |
|---|---|---|---|---|---|---|---|
| B1 | `sweep_b1_tmapg8.py` | `OKC_TMAPG8 0→1` | page indices by TMA for the q8 flat kernel: one `cp.async.bulk` per 64-key tile (PS=4 slots, PD=3 ahead, producer `bar.sync 2`) instead of `PK` per-thread `__ldg` page loads per pass; same code the q1 flat kernel already runs | 0 (per-request kernels never take TMAPG) | ±0.14% [ESTIMATE] (δ_k ±0.7% × 20.3%) | ±0.17% [ESTIMATE] | lint identical, build identical, diff = 3 lines |
| B2 | `sweep_b2_np8_256.py` | `OKC_NP8 128→256` | q8 flat producer warpgroup 4 → 8 warps (32 rows each, 2 passes/tile instead of 4); compiler-checked cost: `NP 256, NTX 512, BASEREGS 80→64, CREGS 104→96`, registers/SM exactly 65536×2 CTAs — the consumers pay for the producers | 0 | +0.10% [ESTIMATE] (prior: the twin knob `OKC_NP` 256 was 0.7%/0.4% *worse* than 128 on the q1 kernel) | +0.12% [ESTIMATE] | lint identical, build identical, diff = 3 lines |
| B3 | `sweep_b3_wgdec0.py` | `OKC_WGDEC 1→0` | q8 consumers sync as one block per stage (`csync<WS,NT>`) instead of per KV head (`wgsync(wgi)` + per-head release): removes the head-level overlap the q8 flat kernel was built for (line 2163 now always takes the `csync` path, per-head release at line 2300 becomes unreachable) | 0 (WS=0 there, `csync` either way) | +0.20% [ESTIMATE] (δ_k +1%) | +0.25% [ESTIMATE] | lint identical, build identical, diff = 3 lines |
| B4 | `sweep_b4_pvw1.py` | `OKC_PVW 0→1` | PV becomes an async wgmma over a double-buffered f16 `V16` (`PV8ON→0`): no e4m3 V^T transposes, no ones-rows, no e4m3 P pack, `S.kvq` 18 KB → 32 KB; PV operand bytes double and wgmma throughput halves | −0.25…+0.66% [ESTIMATE] (δ_k −3…+8% × 8.2%) | −0.6…+1.6% [ESTIMATE] | −0.7…+2.0% [ESTIMATE] | lint identical, build identical, diff = 3 lines |
| B5 | `sweep_b5_ws8_0.py` | `OKC_WS8 1→0` | q8 flat kernel drops the producer warpgroup entirely (v10 flat structure): `WS 0, NP 0, NTX 384→256, MINB 2, BASEREGS 128, CREGS 104→128`; consumers issue K/V + page indices themselves, no mbarrier handover, no TMA page path | 0 | −1.0…+2.4% [ESTIMATE] (δ_k −5…+12%) | −1.2…+3.0% [ESTIMATE] | lint identical, build identical, diff = 3 lines |
| B6 | `sweep_b6_qp16_0.py` | `OKC_QP16 1→0` | q8 QDIRECT q pass back to 8 threads/row (8 dims each, 3 shuffle levels, 8 B e4m3 store) instead of 4 threads/row (16 dims, 2 shuffles, one 16 B store) — the knob's own comment claims the 4-thread form halves the passes | +0.08…+0.25% [ESTIMATE] (δ_k +1…3% on the 8.2% per-request share) | 0 | 0 | lint identical, build identical, diff = 3 lines |
| B7 | `sweep_b7_qss_off.py` | `OKC_QSS 1→0` + `OKC_PV8FLAT 1→0` | **pair**: q8 loses the FP8/QSS complex — no 8 KB e4m3 q block (`q8s` 8192→16 B), QK^T back to RS wgmma, q8 flat takes f16 PV from `V16` (`PV8` folds to 0 when `!FLAT\|\|PV8FLAT`), and per-request q8 falls out of `QDIRECT` (it requires QSS) | +0.2…+0.6% [ESTIMATE] (QDIRECT loss) | +0.8…+2.0% [ESTIMATE] (δ_k +4…10%) | +1.0…+2.5% [ESTIMATE] | lint identical, build identical, diff = 5 lines |

Nothing here changes a scored non-verify shape: `OKC_PV8`/`OKC_Q8RAW`/`OKC_QDIRECT` are all gated on
`L == 8`, and the q_len-1 decode loads never enter these paths.

## 2. Ranked for the campaign

Ordered by (size of the plausible row move) × (probability the sign is favourable). Only the first two
have a credible path past 0.3%; none is expected to clear 1.15% alone.

| rank | file | why this rank |
|---|---|---|
| 1 | `sweep_b5_ws8_0.py` | biggest structural swing available (producer warpgroup on/off for q8) **and** the only knob in the family that *gives* the q8 flat consumers registers (`CREGS 104→128`) — the one resource they are documented to be short of ("needs OKC_QSS: 104-register consumers"). If the producers are not earning their warp slot at verify 128/256, this is worth ~0.3-0.7% on the row. Downside risk is symmetric and large — do not submit without a mean |
| 2 | `sweep_b4_pvw1.py` | second-biggest structural swing: the whole PV path is swapped, so the answer also tells us whether the e4m3 P pack/transposes or the FP8 tensor throughput dominates PV. If PV8's packing cost is real, δ_k −3…−5% is plausible |
| 3 | `sweep_b2_np8_256.py` | explicitly requested; a real capacity change (copy issue halves, consumer registers drop 8) but the documented prior from the q1 twin points the wrong way (256 was 0.7%/0.4% worse). Cheap 1-draw ticket with an informative negative |
| 4 | `sweep_b1_tmapg8.py` | explicitly requested, cheapest possible candidate, but **the source contradicts itself on the sign**: the define line says "measured ~+0.7%" while the use site (line 1228) says "the q_len 8 flat kernel lost ~0.7%". Either way the row move is ~0.05%, i.e. one fifth of the noise. Take it as a free information draw, not as a bet |
| 5 | `sweep_b7_qss_off.py` | the one **pair** swap; almost certainly a regression (it is the configuration FP8 PV/QSS was introduced to replace), but it is the only direct measurement of what the FP8 PV complex is worth on the q8 flat kernel — worth having as the family's negative control |
| 6 | `sweep_b3_wgdec0.py` | pure sync-structure control; expected worse, reach only the flat kernel. Useful to prove WGDEC is live and correctly valued |
| 7 | `sweep_b6_qp16_0.py` | verify-8 only, 0.0136 score coefficient — the smallest reach in the set. Include only if verify 1 has spare checks |

## 3. New combinations vs single flips

* **New combinations:** B7 (`QSS` + `PV8FLAT` off together — the pre-FP8/QSS configuration, never emitted
  as a variant) and, in effect, B5 (turning `WS8` off is what makes `WGDEC`, `NP8` and the TMA page path
  moot for q8; the triple has never been tested as "no producer warpgroup for q8 at all"). B2 is a single
  flip, but of a value (`256`) that was only ever measured on the *q_len-1* twin (`OKC_NP`), never on NP8.
* **Single flips:** B1, B3, B4, B6.
* Not emitted, with reason: `OKC_PREGS`/`OKC_WSMINB` are second-order variants of B2's register split
  (with NP8=256 the split is already exactly saturated: 256·32 + 256·96 = 32768 per CTA, two CTAs = the
  whole file), and `OKC_SETMAXNREG 0` is not q8-isolated (it changes the q_len-1 kernel and the MoE builds
  with it), so it belongs to another family.

## 4. Verification

All seven files: line 1 `# okc: problem=muse_decode_layer` unchanged; line 2 = the one-line banner naming
the change and its mechanism; `ast.parse` ok; 979,874-979,935 B (limit 1,000,000).

**Lint multiset identical to the base** — `uv run ruff check --no-cache --output-format concise <file>`,
normalised to `(count, rule, message)` and compared by md5: base and all seven give the same 53-line
multiset, `5acfe506`. No new warning, none removed.

**`tools/cu_build.py` status identical to the base** — base and all seven: `ok 10`, `FAIL 54`, exit 1,
and the per-source status lines compare equal after canonicalisation (the 54 failures are the base's own
pre-existing `bfp_*`, `_GRAFT_*`, `_ROUTER_FP32`, `_SL_GLUE` ones; none is a candidate's).

**Read this before trusting that gate.** cu_build's attention check is *vacuous*: it strips the source at
`#include <torch/`, and `ATT7_CUDA_SRC` **starts** with `#include <torch/extension.h>` on its line 2, so
the string it compiles is 1 byte long — the "compiled, no kernel emitted (template body)" line for
`ATT7_CUDA_SRC` means nothing was compiled. Since every knob here lives in that string, I added two
compiler-evaluated checks outside the repo (`/tmp/cfg_probe.py`, `/tmp/attn_inst.py`):

1. **`Cfg` constants, evaluated by nvrtc** on the real build's define set
   (`OKC_QDIRECT=1 OKC_PK2=1 OKC_W0ENABLE=1 OKC_SM90=1`) for `Cfg<8,true>` / `Cfg<1,true>`
   (`Show<N>`-instantiation trick, values read out of the diagnostic). Base: `WS 1, NP 128, NTX 384,
   MINB 2, BASEREGS 80, PREGS 32, CREGS 104, STAGES 4`; B2: `NP 256, NTX 512, BASEREGS 64, CREGS 96`;
   B5: `WS 0, NP 0, NTX 256, CREGS 128`. `Cfg<1,true>` is unchanged in all seven (the q1 kernel is
   untouched, as intended). The producer-coverage `static_assert` (line 289) holds in every variant:
   `256 % 8 == 0, 64 % 32 == 0` for B2, vacuous (`!WS`) for B5.
2. A full device-only instantiation of `attn_kernel<8,true,0>` / `<8,false,1>` / `<8,false,5>` was
   attempted to get registers and spills. It does not complete offline: the extracted source needs both
   the torch includes (for the host `launch` template that shares the string) and the helper prelude the
   real TU gets from elsewhere, and nvrtc rejects the `tfor`/`csync` overload sets in that partial
   translation unit. **Registers/spills of the q8 kernel are therefore unverified for all seven
   candidates** — the numbers quoted above are `Cfg`'s launch-time budget, not ptxas's allocation. The
   only two `static_assert`s in the attention source (lines 289, 1224) are untouched by every candidate.

**Diffs against the base** (only the intended lines; `1a2` is the banner):

```
=== sweep_b1_tmapg8
1a2
> # sweep_b1: OKC_TMAPG8 0->1 -- ... (banner)
157c158
< #define OKC_TMAPG8 0    // ... for the q_len 8 flat kernel too (measured ~+0.7%)
---
> #define OKC_TMAPG8 1    // ... for the q_len 8 flat kernel too (measured ~+0.7%)
=== sweep_b2_np8_256
1a2
> # sweep_b2: ... (banner)
137c138
< #define OKC_NP8 128     // q_len 8: 4 producer warps (its consumers need the registers)
---
> #define OKC_NP8 256     // q_len 8: 8 producer warps (sweep_b2: NP8 128 -> 256; its consumers need the registers)
=== sweep_b3_wgdec0
1a2
> # sweep_b3: ... (banner)
161c162
< #define OKC_WGDEC 1     // q_len 8 producer mode: the two consumer warpgroups sync separately (one per head)
---
> #define OKC_WGDEC 0     // q_len 8 producer mode: the two consumer warpgroups sync separately (one per head)
=== sweep_b4_pvw1
1a2
> # sweep_b4: ... (banner)
160c161
< #define OKC_PVW 0       // q_len 8 on sm_90: PV as an asynchronous wgmma from a double-buffered f16 V tile
---
> #define OKC_PVW 1       // q_len 8 on sm_90: PV as an asynchronous wgmma from a double-buffered f16 V tile
=== sweep_b5_ws8_0
1a2
> # sweep_b5: ... (banner)
136c137
< #define OKC_WS8 1       // producer warps for the q_len 8 flat kernel too
---
> #define OKC_WS8 0       // producer warps for the q_len 8 flat kernel too
=== sweep_b6_qp16_0
1a2
> # sweep_b6: ... (banner)
150c151
< #define OKC_QP16 1      // q_len 8 QDIRECT q pass: four threads per row (16 dims each)
---
> #define OKC_QP16 0      // q_len 8 QDIRECT q pass: four threads per row (16 dims each)
=== sweep_b7_qss_off
1a2
> # sweep_b7: ... (banner)
168,169c169,170
< #define OKC_PV8FLAT 1   // FP8 PV in the flat (B > 1 full) kernel too (needs OKC_QSS: 104-register consumers)
< #define OKC_QSS 1       // with FP8 PV: q in shared memory (QK^T as an SS wgmma), 8 fewer live registers
---
> #define OKC_PV8FLAT 0   // FP8 PV in the flat (B > 1 full) kernel too (needs OKC_QSS: 104-register consumers)
> #define OKC_QSS 0       // with FP8 PV: q in shared memory (QK^T as an SS wgmma), 8 fewer live registers
```

Diff line counts: 3 / 3 / 3 / 3 / 3 / 3 / 5 (two of those lines are the define and its value; B7 has two
defines). No other byte differs.

## 5. What to do with them

If S1 spends one draw per sweep file, B1/B2/B3/B6 will read as noise (±2.3% per draw against a ~0.1%
effect) — they are information, not bets, and a negative is still a result because four of them carry
measurements from a *different* build that our file has been inheriting unchecked. Only B5 and B4 have a
row-level ceiling above 0.3%, and both need a 4-draw mean before anything is submitted.
