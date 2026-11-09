# sweep_a — attention finalize + PDL constants (first isolated screen)

Base row: `src/okc_loop/vendored_work/layer/klr_fci5.py` (2.28370, #88/1227). Noise SD 1.17%, 4-draw screen ≈1.15%.
Every variant below is a byte copy of the base with **one coherent change** and a line-2 comment; `diff` evidence at the bottom.

Reachability: every constant below is a literal `#define` inside `ATT7_CUDA_SRC` / a module-level Python value in this file,
compiled by `load_inline` with only `-DOKC_QDIRECT / -DOKC_PK2 / -DOKC_W0ENABLE / -DOKC_SM90` (line 3736). None of the
targets is overridden by a host `-D` flag or by `_ma_ext` / `_moe_build` define tuples, and each name occurs exactly once
in the file, so every variant is *reachable* from Python. `PDL_MAX_T` is Python-side and gates 43 `_pdl(T)` sites layer-wide.

## Liveness per token count (read from `_plan7` / `_TUNED7` / the `mode1` expression)

| T (scored shapes) | plan | COOP | MERGE2 | MERGE3 | SC1 | SC8 | PINDIV |
|---|---|---|---|---|---|---|---|
| 1 (b1 q1, full) | `_TUNED7 (1,1,False,1)`: nc=8, coop=1, 252 splits | ✓ | ✓ | – | – | ✓ (nc==8) | ✓ |
| 8 (b1 q8, sliding) | reqs<16: nc=1, coop=1 | ✓ | ✓ | – | ✓ | – | ✓ |
| 8 (b1 q8, full) | non-flat one-request: nc=min(n,8)=8, coop=1 | ✓ | ✓ | – | – | ✓ | ✓ |
| 16/32 (q1 sliding) | reqs<24: nc=1, no coop | – | ✓ | – | ✓ | – | – |
| 128/256 (b16/b32 q8, sliding) | reqs≥16: nc=1, coop=1 | ✓ | ✓ | – | ✓ | – | ✓ |
| 128/256 (q8 full, multi-req) | flat, L==8, n_splits·pairs ≤ 2·SMs → coop=1 | ✓ | – | ✓ | – | – | – |
| ≥24 requests (q1 sliding) | nc=1, no coop | – | ✓ | – | ✓ | – | – |

`mode1 = !((FLAT && L==8 && M_COOP) && !(!FLAT && M_COOP))`: MERGE2 covers every per-request shape, MERGE3 exactly the flat-coop q8
shapes. So `MERGE2+MERGE3` is the only single change here that is live on all six token counts.

## Variants

| file | changed | mechanism | expected effect (µs/load or %) | lint | build |
|---|---|---|---|---|---|
| `sweep_a2.py` | `OKC_MERGE2 0`, `OKC_MERGE3 0` | merge release: per-thread `__threadfence()` before `bar.sync` instead of thread 0's one `atom.acq_rel` after it | sign open; ±0.3–0.7 µs per attention launch (2–4% of a 15 µs launch) [ESTIMATE]; live at T=1…256 | 30 errors / identical multiset | identical per entry (10 ok / 54 fail) |
| `sweep_a3.py` | `PDL_MAX_T 128 → 64` | PDL (griddepcontrol) only for T≤64; T=128 launches plain | −0.5…−1.5% at T=128 if the documented overlap is real, 0 elsewhere; conclusive either way [ESTIMATE] | identical | identical |
| `sweep_a1.py` | `OKC_COOP 0` | single-request grids use the last-CTA finalize (one CTA's rows) instead of a grid barrier + per-CTA slices | known-positive knob re-measured under the new floor: in-file coop wins are 12.30→12.00 µs at T=1 (+2.5% if reverted) and 13.0→10.2 µs at T=8 b1 q8; the q8 b16/b32 wins (1.6% / 3.0%) predate parity barriers, inlined finalizes and the cheap merges, so they may have shrunk [ESTIMATE] | identical | identical |
| `sweep_a4.py` | `OKC_SC8 0` | eight-copy finalize falls back to the rolled 4-copy loop: two dependent L2 round trips instead of 17 loads held in registers | −0.1…−0.3 µs at T=1 / T=8 b1 q8 (finalize is ~1 µs of a 12 µs launch) [ESTIMATE]; upside only if the 72 live registers of the nc==8 burst are costing occupancy/spills — check the variant's reg count if it wins | identical | identical |

### Why each might matter at our token counts

- **a2 MERGE2/MERGE3.** The merge release sits on the critical path of every launch, between the reds and the finalize
  barrier. The base pays one `atom.acq_rel` L2 RMW per CTA to release; the alternative pays a local `membar.gl` per thread
  (no L2 RMW, but every thread). At T=1 the grid is 252 CTAs of one 16-row tile each, so per-CTA merge volume is tiny and
  the atomic is nearly pure latency; at T=128/256 many-request q1 shapes there are only ~2 splits per request, same story.
  Where CTA counts are low and row counts high (q8 b16/b32) the reds already dominate and both forms should tie.
  Bidirectional: if a win appears it is the cheapest kind (a few instructions, no new state).
- **a3 PDL_MAX_T.** PDL is a layer-wide launch property (MoE `okc_set_pdl`, `sk2_set_pdl`, attention `use_pdl` bit 16,
  Triton `USE_PDL`). The header records "PDL at T=128" inside a winning bundle and "~1% loss at T=256", i.e. 128 was never
  isolated. Dropping PDL at 128 turns the bundle claim into a number; PDL only helps if the next kernel's prologue is long
  enough to hide, and the MoE at T=128 is the largest prologue in the layer, so the documented win is plausible.
- **a1 COOP.** The coop finalize is chosen only when the whole grid is resident (`grid ≤ occ·SMs`). Its alternative is a
  single CTA reading and zeroing the workspace for the whole request, which the file's own notes put 0.3–2.8 µs behind at
  b1 shapes. The interesting question is not T=1 (settled) but the q8 b16/b32 shapes, where the win was 1.6–3.0% before the
  merge/barrier rewrites; if it is now flat, the coop path is dead weight worth deleting.
- **a4 SC8.** Fires only at nc==8, i.e. the two single-request full-layer shapes (T=1 b1 q1, T=8 b1 q8). The straight-line
  burst issues 8×2 float4 + 8 loads in one round trip; the register cost lands at the very end of a kernel that is already
  register-heavy. Two dependent round trips is the price of reverting. Small in absolute terms — include only if the
  campaign has spare checks.

## Ranked order for the campaign

1. **`sweep_a2.py`** — MERGE2+MERGE3 0. Only change live at T=1…256, cheapest possible win, and the one constant whose
   isolated effect is genuinely unknown in both directions.
2. **`sweep_a3.py`** — `PDL_MAX_T 64`. Decisive about a documented but unmeasured 128-token win; a null result frees the
   PDL accounting, a loss closes the topic in one draw set.
3. **`sweep_a1.py`** — `OKC_COOP 0`. Highest measurement value per check after a2/a3 (three shape families, one control
   draw each); prior is negative at T=1/T=8 b1 and open at the q8 b16/b32 shapes.
4. **`sweep_a4.py`** — `OKC_SC8 0`. Narrow (T=1, T=8 b1 full-layer) and likely below the 1.15% screen; run last, or skip
   in favour of a5 below if the day's checks are short.

Substitutes if any of the above turns out to be flat/unreachable: `OKC_PINDIV 0` (live on every coop shape, division
scheduling) and `OKC_ONENEW 0` + `OKC_ONEPRED 0` (ONE-build prologue, live on every per-request off32/page-1 shape).
Both are one-line copies of the same recipe; neither was emitted because both priors are clearly negative.

## Verification

```
$ PYTHONPATH=src .venv/bin/python -m okc_loop.cli lint <file>
FAIL tier0 muse_decode_layer: 30 error(s)        # same count, same 84-finding multiset as the base
$ PYTHONPATH=src .venv/bin/python tools/cu_build.py <file>
10 ok / 54 fail per entry — status line-for-line identical to the base (differences only in the module's file-name prefix)
```

The only lint diff versus the base is the `line N` offsets shifting by +1 (the inserted line-2 comment); with line numbers
normalised the outputs are byte-identical for all four files.

## Diffs

```diff
===== sweep_a1 =====
1a2
> # sweep_a1: OKC_COOP 0 -- single-request coop grids use the last-CTA finalize: one CTA's rows instead of a grid barrier + per-CTA slice divisions.
159c160
< #define OKC_COOP 1      // cooperative finalize for single-request grids (0: always the last-CTA finalize)
---
> #define OKC_COOP 0      // cooperative finalize for single-request grids (0: always the last-CTA finalize)

===== sweep_a2 =====
1a2
> # sweep_a2: OKC_MERGE2/OKC_MERGE3 0 -- the merge releases with per-thread gpu_fence() instead of thread 0's single acq_rel atomic after bar.sync.
145,146c146,147
< #define OKC_MERGE2 1    // last-CTA merges: one acq_rel atomic after the CTA barrier instead of per-thread fences
< #define OKC_MERGE3 1    // cooperative merges: one acq_rel barrier arrive after the CTA barrier instead of per-thread fences
---
> #define OKC_MERGE2 0    // last-CTA merges: one acq_rel atomic after the CTA barrier instead of per-thread fences
> #define OKC_MERGE3 0    // cooperative merges: one acq_rel barrier arrive after the CTA barrier instead of per-thread fences

===== sweep_a3 =====
1a2
> # sweep_a3: PDL_MAX_T 64 -- programmatic dependent launch only for T <= 64; T = 128 falls back to plain launches (isolates the documented 128 PDL overlap).
98c99
< PDL_MAX_T = 128  # a launch-latency win at decode sizes, a ~1% loss at T = 256
---
> PDL_MAX_T = 64  # a launch-latency win at decode sizes, a ~1% loss at T = 256

===== sweep_a4 =====
1a2
> # sweep_a4: OKC_SC8 0 -- the eight-copy coop finalize uses the rolled 4-copy loop (17 loads held in registers swapped for two dependent L2 round trips).
152c153
< #define OKC_SC8 1       // coop finalize: the eight workspace copies' loads in one round trip
---
> #define OKC_SC8 0       // coop finalize: the eight workspace copies' loads in one round trip
```
