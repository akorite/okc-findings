# Dense bake pick margins on `klr_fci5.py` — no relaxation is justified

Base: `src/okc_loop/vendored_work/layer/klr_fci5.py` (979,654 B), measured 2.2730 (board-matched
`profiles/klr_fci5.9743b6e71fbd42fd.log`, speedup 2.2837). Logs read: all 17 decode-layer
`src/okc_loop/vendored_work/layer/profiles/*.log` (24 h).

## 1. Knob inventory — which cells and shapes each can change

| knob | value | decides | cells it can change | used on the scored path? |
|---|---|---|---|---|
| `MARGIN_RQ128` | 0.98 (bar = 1/0.98 = **1.0204**) | `_pickrule` at L16148 | every baked `(layer,m)` | only the 3 in `RQ_USE` (L16209): `('qkv',128) ('un',128) ('qkv',256)` |
| `MARGIN_RQS` | **not present in fci5** (`klr_bnd2`-lineage only, L14486 there) | small-m bake | m=16/32 | no: fci5 bakes only `for m in (128, 256)` (L16064) and `RQ_USE` has no m≤32 |
| `RQTOL[KS]` | 4e-3 … 8e-3 (judge 1e-2) | `_rqval` (L16104) rejects a candidate *before* timing | any candidate | no rejection to recover — see §4 |
| `RWIN128` | 1.01 | `_refsel` picks the group's reference | changes the *reference*, not the bar | inert on every logged run (§4) |
| `bfp__autotune` | `reps=24`, **strict argmin** (L10057-10105) | `bfp_TUNE_M=(1,16,32,8,128,256)` dense chain | m≤32 / m=128 / m=256 GEMM choice | live, but there is no margin to relax |
| `TUNE_MARGIN` | 0.95 | `_tune` moe pick (L11186) | Marlin vs agent per T | inert: `st["bank"]=None` unconditionally (L11270) → `cands==["agent"]` → branch dead |
| `PREPARE_BUDGET_S` | 280 | bake budget | which candidates get timed | no `skipped-budget` line in any of the 17 logs |

Scored T values (from `profiles/klr_fci5.…txt`): decode 1/16/32 and verify 1/16/32 with `q_len=8`,
i.e. **T = 1, 8, 16, 32, 128, 256**. `dense()` takes the rq path only for `T > 32` (L12388), so the
rq cells that matter are exactly T=128 and T=256.

## 2. Current picks per (cell, m), from today's judge logs

`profiles/klr_fci5.9743b6e71fbd42fd.log` (the 2.2730 run):

```
BAKE qkv_gate m=128 rq:d2s5hze 15.30 (ratio 1.3766 vs r6 20.61)
PICK qkv_gate m=128 rq:d2s5hzwP 14.81 (ref rq:d2s5hze 15.30)
PICK qkv_gate m=128 rq:d2s5hzwP (forced: cursor + short apply, resident -0.26/-0.13/-0.32 us, bake ratio 1.0331)
BAKE qkv_gate m=256 rq:n256s5m1hzmEH 21.45 (ratio 1.1022 vs r6 23.14)
BAKE qkv_gate m=256 rq:n256s5m1hzm 22.00 (ratio 0.9834 vs rq:n256s5m1hzmEH 21.45)
PICK qkv_gate m=256 rq:n256s5m1hzmEH 21.45 (ref rq:n256s5m1hzmEH 21.45)
BAKE unpooler m=128 rq:d2s5hze 20.39 (ratio 1.1851 vs r6 23.71)
PICK unpooler m=128 rq:s4s5hzw 19.55 (ref rq:d2s5hze 20.39)
PICK unpooler m=128 rq:d2s5hzwP (forced: ... bake ratio 1.0337)
PICK unpooler m=256 r6 27.65 (ref r6 27.65)
```

| cell | bake winner | final `chen_rq` | governed by |
|---|---|---|---|
| (qkv_gate, 128) T=128 | `d2s5hzwP` (1.0331) | `d2s5hzwP` | `RQPICK128` force → margin irrelevant |
| (unpooler, 128) T=128 | `s4s5hzw` (1.0430) or `d2s5hzwP`, run-dependent | `d2s5hzwP` | `RQPICK128` force → margin irrelevant |
| (qkv_gate, 256) T=256 | `n256s5m1hzmEH` (1.1022) | `n256s5m1hzmEH` | `MARGIN_RQ128` — clears the bar by 5.4× |
| (unpooler, 256) | none baked (`RQBAKE_L` has no entry) | not in `RQ_USE`; falls back to graft/`_vj_dense` | no candidate exists |
| (pooler, 128/256) | listed in `RQBAKE_L` but `RQLAYERS` excludes `pooler` | never baked, never used | dead config |

`RQPICK128` covers m=128 only, and `RQ_USE` is `{('un',128),('qkv',128),('qkv',256)}` — so **two of the
three used cells are decided by the force table, not by a margin**.

## 3. Every candidate the margins actually rejected (all 17 logs)

Only three rejections exist, and all three are inert:

1. `profiles/klr_merged3.43b76d9dc2eb4854.log:54` —
   `BAKE qkv_gate m=128 rq:d2s5hzwP[126c] 15.88 (ratio 1.0155 vs rq:d2s5hze 15.26)` at bar 1.0204 →
   `:60 PICK qkv_gate m=128 rq:d2s5hze 15.26`, i.e. the 1.55% candidate lost. **The very next line
   reinstates it**: `PICK qkv_gate m=128 rq:d2s5hzwP (forced: ...)`. Final state identical to a
   relaxed bar ⇒ zero effect.
2. `profiles/klr_bundle2.…log` — `qkv_gate m=32` best cand `s4s5hzw` at 1.0211 vs a 1.2048 bar
   (`MARGIN_RQS=0.83` in that lineage) → `PICK qkv_gate m=32 r6`.
3. `profiles/klr_qd3.…log` — `qkv_gate m=32` best cand `d2s5hze` at 1.0285 → `PICK qkv_gate m=32 r6`.

(2) and (3) are m=32, which `RQ_USE` never uses; at m=32 the dense chain runs `sk2`
(`TUNE qkv_gate m=32 sk2:14.30us/96c torch:17.60us/lib`, 4 µs faster than the 17.9-18.5 µs rq
candidates those logs timed). Relaxing `MARGIN_RQS` to 1.0 there would still pick `r6` (all
candidates were *below* 1.0 in `klr_bnd2`: 0.9474 / 0.9136 / 0.9888 / 0.9886).

The ticket's exemplar — "a candidate at 1.15 rejected by a 1.20 bar" — does not occur in any log.

## 4. Why relaxing is not merely inert but mildly harmful at the one live cell

`MARGIN_RQ128` governs only `(qkv_gate, 256)` (T=256, verify 32). Its rival `n256s5m1hzm` was timed
against `n256s5m1hzmEH` on the same box in 17 runs:

```
0.9815  1.0000  1.0044  0.9711  0.9945  0.9834  1.0010  1.0014  1.0112  0.9789  0.9983  1.0128  1.0065  0.9962  0.9928  0.9973  0.9935
```

median 0.997: the rival is ~0.3-2.5% **slower** and its "wins" are noise inside ±1.3%. Absolute µs
from the logs: `21.70 vs 21.51`, `21.52 vs 21.06`, `21.44 vs 20.60`, `21.46 vs 21.13`.
- `0.99` (bar 1.0101) flips 5 of 17 runs to the slower variant (ratios 1.0000/1.0010/1.0014/1.0044/1.0065).
- `1.00` (bar 1.0) flips 6 of 17 (adds 1.0112 and 1.0128; a 1.0000 ratio is not `> 1.0`).

Cost if flipped: +0.17…0.52 µs on the 21 µs GEMM ⇒ +0.06…0.17% on the two verify-32 cases
(307.8 / 471.2 µs) ⇒ **≤0.0006 on the row** — noise, signed the wrong way.

`RQTOL`: no `INVALID` line in any of the 17 logs (only three `ERROR launch -14` from the `@4/@8`
occupancy probes, not tolerance failures), so moving `RQTOL[7] 8e-3 → 1e-2` would unlock nothing.
`RWIN128`: in every fci5-lineage log (fci5 ×2, fci6, best, glue ×2, nx2, merged3) each proven pick
qualified as its own reference and `PICK … (ref r6 …)` never appears for a baked cell, so 1.01 is not
deciding anything on our build. The older `bnd2`/`bundle*`/`qd*` logs do fall back to an `r6`
reference, but every one of those proven picks read 0.56-0.99 of r6 (genuinely slower), not a
1%-window miss.

## 5. Verdict

- **No `klr_margin.py` emitted and no judge check spent.** The rule was "do not relax a margin
  without a log line"; no log line exists, and the arithmetic above says a relaxation is 0.0000 on
  the row with a 25-50% chance of a −0.0006 tail.
- The dense bake is not a margin problem: it is a *coverage* problem. Of the 4 `(layer, m)` groups
  `_rqbake` walks (`RQLAYERS` = qkv_gate, unpooler × m=128, 256), `(unpooler, 256)` has no candidate
  at all and is filtered out of `RQ_USE` anyway; the `(pooler, 128)`/`(pooler, 256)` entries in
  `RQBAKE_L` are dead because `RQLAYERS` excludes `pooler`; and the two used m=128 cells are
  hand-forced past their own bake. If someone wants the dense bake to pay on the board, the levers
  are `RQ_USE`/`RQBAKE_L` coverage, not the bars.
- Falsification, if Main insists on measuring: `MARGIN_RQ128 = 0.99` is the only "one step" with any
  backing at all (`klr_merged3:54`, a 1.0155 candidate under a 1.0204 bar) and it is provably a
  no-op because `RQPICK128` re-selects that same variant on the next line.

## 6. Hand-off (not this ticket): the live pick variance is `esc2` capacity, not a margin

`profiles/klr_fci7.a44d8fad04124598.log:28` (01:14, newest run):
`BAKE rq unpooler esc2 slots max=5 OVERFLOW` → `esc2 is None` → L16094 drops every `RQCUR`
candidate, so the log shows only `d2s5hze / s4s5hzeEH / s4s5hze / d2s6hze` and the `RQPICK128` force
for `(unpooler, 128)` cannot fire at all: `:34 PICK unpooler m=128 rq:d2s5hze 21.33`. In runs that
pack (`max=3 ok`) the same cell ships `d2s5hzwP` at 20.58-20.74 µs.

- Frequency: `esc2 slots max` over 19 unpooler bakes = 16×3, 2×4, 1×5 ⇒ ~5% overflow, 16% at the CAPT=4
  boundary.
- CAPT is baked into the CUDA gather: L15273 `E4 = *(const uint4*)(p.esc2 + … * 32 + lane) * 4)`
  (hardcoded stride 4, one uint4 load), so raising it to 5-6 needs a kernel change + bit-identity
  evidence and its own gate.
- Worth ~0.75 µs on the T=128 unpooler ⇒ 0.23-0.32% on the two verify-16 cases ⇒ **≈+0.001 on the
  row, 5% of runs only**.

