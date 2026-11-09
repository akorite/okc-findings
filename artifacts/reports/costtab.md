# WGS stream-K cost tables (`OKC_WGSA` / `OKC_WGSB`) — what they are and what changing them does

Files: `klr_costtab.py` (bytes-model linear table), `klr_costtfl.py` (flat table). Base
`klr_fci5.py` (`ee74c9e4…`, judge 2.2730 / per-load d1 2.5604 d16 2.1312 d32 1.9716 v1 2.4961 v16 2.2322 v32 2.3003).

## 1. How the table enters the balance

`wgs_cost(pb, cnt)` (`L5592`) indexes the 8-entry table by **n-tile count**
`ntt = (cnt + 7) >> 3`, `cnt = S.jC[j]` = the job's token count ⇒ entry `i` is `ntt = i+1`, i.e. 8·(i+1) n8
columns of the stage's weight slab. `ntt` saturates at 8 (`FT_T = 64`, so `WGS_NT = 8`).

One stage = 36864 B of weights (the whole 256-row slab is copied whatever `ntt` is: only `ntt` of its 8 n8 tiles
feed the `wgmma`) **plus `npad * 512 = ntt * 4096` B of x**, refetched per stage (`L5757` `bytes = npad * 512u`).
So the byte count per stage is `36864 + 4096·ntt`.

The table is used in three places, all of them the same cost function:

* `wgs_prefix`/`wgs_ranges` (`L6125`, `L6154`): per job, phase A is 32 units of 16 stages, phase B 16 units of 16
  stages; the prefix accumulates `(16·cs(ntt) + U) << 5` (A) / `<< 4` (B), i.e. one *unit* costs `16·cs + U`.
* `wgs_stage_at` (`L6136`): turns a cost position into a stage — binary search the job prefix, then
  `un = rem / (16cs+U)`, `st = (rem − U) / cs`. So a CTA's range is the set of stages whose cost lies in
  `[tot·c/G, tot·(c+1)/G)`.
* `wgs_cta_of` (`L6187`): the inverse, used to find the CTA that owns a split unit's first stage.

A CTA whose range holds larger-`cs` stages gets **fewer stages** — that is the whole lever. In this build every
unit of every job goes through this split (`L6458` calls `wgs_phase` over `S.wgr`; `fat_stream`/`run_phase` are
the non-WGS builds), and the ranges are contiguous in the job order (fat jobs first, then skinny), so a CTA sits
on ≈one job and the table's `cs(ntt+1)/cs(ntt)` is priced almost raw. A table that *overstates* the ratio starves
the big-`ntt` CTAs and makes the small-`ntt` CTAs the phase end.

Which loads can even see it (band selection `_moe`, `L13512`): T=1 → tiny, T=8 → tiny2, T=16 → mid (**no
`OKC_WGS`**), T=17..32 → b4 = WGS, T=128/256 → main = WGS. So `decode 32`, `verify 16`, `verify 32`.
Per local expert the token count is `Poisson(T·8/256)` (32 of 256 experts, topk 8): mean 1.0 / 4.0 / 8.0, so
`ntt = 1` for ~all jobs at T=32 and T=128 (P(cnt>8) = 1e-6 / 0.021) and for 56 % of the jobs at T=256 (41 % are
`ntt = 2`). **Only entry 0 is exercised by decode 32 / verify 16, entries 0–1 by verify 32; entries 2–7 are dead
on this mix** — which is also why the tuner's note ("bs32 +0.45, bs128 +0.94, bs256 +0.17 %", fitted at bs
96..256 where `ntt` reached 4–8) does not transfer to our shapes.

## 2. The two tables disagree, and the bytes say where the truth is

| ntt | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| shipped A | 112 | 135 | 140 | 158 | 173 | 185 | 200 | 207 |
| shipped B | 100 | 104 | 115 | 123 | 132 | 144 | 156 | 174 |
| bytes model, A (anchor 112) | 112 | 123 | 134 | 146 | 157 | 168 | 179 | 190 |
| bytes model, B (anchor 100) | 100 | 110 | 120 | 130 | 140 | 150 | 160 | 170 |
| flat (`klr_costtfl.py`) | 112 | 112 | … | | | | | 112 |

`112·(36864+4096·n)/40960 = 112 + 11.2·(n−1)`; the shipped A and B describe the *same* byte stream and disagree by
16 % on the live bucket (A(2)/A(1) = 1.205, B(2)/B(1) = 1.04 vs bytes 1.100) — a noisy per-CTA fit.

The file's own measured numbers settle it: one phase-B unit = **18.5 µs at d32/v16 and 19.4 µs at v32**
(`aly_moe_diff.md:106`), i.e. 1.156 → 1.2125 µs per stage. v32's job mix is 56 % `ntt=1` + 44 % `ntt=2`:

* bytes 1.100 → 1.156·(0.56 + 0.44·1.100) = **1.207 µs** (+0.5 % of the measurement);
* flat 1.000 → 1.156 µs (−4.7 %);
* shipped A 1.205 → 1.260 µs (+3.9 %); shipped B 1.04 → 1.176 µs (−3.0 %).

So the truth is the byte ratio, and the shipped A over- and B under-states it — exactly what `klr_costtab.py`
reconciles (both tables get the one byte shape, anchored at the shipped `ntt = 1` value so the four `ntt = 1`
loads' allocation is untouched).

## 3. Predicted size (`artifacts/reports/costtab_model.py`, G = 132)

Rebuilds `wgs_prefix`/`wgs_stage_at` exactly and prices the resulting per-CTA ranges under the byte truth:

| load | jobs (ntt hist) | phase A max-work, shipped → costtab | phase B |
|---|---|---|---|
| decode 32 | 22 × ntt 1 | identical (1.0000) | identical |
| verify 16 | 31 × ntt 1 | identical (1.0000) | identical |
| verify 32 | 18 × ntt 1, 14 × ntt 2 | stages 121–128 → 118–129, imbalance +3.6 % → +1.3 % (**2.3 % shorter**) | +1 % longer (rounding) |

Net ≈ **1.2 % of the WGS time on verify 32 only** ⇒ ~0.12 % of the board — small, and the pattern differs from the
WGSU/WGSUB sweep's (a scalar change moves all three WGS loads uniformly; this moves only verify 32, and only its
`ntt = 2` CTAs).

## 4. Gates

* `ast.parse` ok both files; 980,496 / 980,092 B; line 1 = `# okc: problem=muse_decode_layer`.
* `python tools/colab_gate.py … --out artifacts/bench/gate_costtab.py` + `colab run --gpu A100`: 85.1 s,
  `_moe`, `_moe(32)`, `_moe_tiny`, `_moe_tiny2`, `_moe_mid`, `_att7`, `bfp__get_ext` all `ok: true`; only the
  three documented-ungateable tvm_ffi glue modules fail.
* `static_assert(WGS_CA[0] > 0 && WGS_CA[7] < 256 && …)` holds (max 190).

## 5. Measured (`klr_costtab.py`, job `3edcd41d4421463f`, geomean **2.2386**, gate: *gain* failed vs 2.2474)

| load | ratio | vs board row |
|---|---|---|
| decode 1 | 2.5140 | −1.81 % |
| decode 16 | 2.1085 | +0.18 % |
| decode 32 | 1.9091 | −0.66 % |
| verify 1 | 2.4947 | +0.67 % |
| verify 16 | 2.1804 | −1.24 % |
| verify 32 | 2.2861 | +0.49 % |

**No gain.** Read against the same base's other recent runs, the only load the change can reach does not move either
(`verify 32` ratio across four runs of this base: 2.3003 / 2.2799 / 2.3020 / 2.2861 — the candidate is −0.3 % off
their mean), and the three non-WGS loads moved ≤1.8 % with no mechanism (tiny/mid builds do not even contain the
table), i.e. the run's noise floor is ~±1.5 %. This run's *reference* times were also 2.7–4.0 % slower on v16/v32, so
the absolute `cand_us` deltas are not ours. Note the base's 2.2730 was itself a lucky run (`decode 32` 1.9716 vs
1.888–1.897 in two other runs, `verify 16` 15914 vs 16514–16646): its honest standing is ≈2.245, which is what the
board row 2.2474 says.

**Verdict: the cost-table lane is closed, and its ceiling was ~0.1 % of the board anyway.** The model above says at
most 2.3 % of one phase on one load is available (2–3 % of `verify 32`'s WGS time), which is 10× below the judge's
run-to-run band — unmeasurable by construction. The shipped `OKC_WGSA`/`OKC_WGSB` are not detectable as a mistake at
our noise floor, so the "wrong table costs 12.5 % of the stream-K balance" bound does not bind: measured imbalance in
the ranges is +0.5…+3.6 %, not 12.5 %. `klr_costtfl.py` (flat) is predicted worse than both under every truth
hypothesis tested (−2 % phase A) and was **not** spent a check on.

