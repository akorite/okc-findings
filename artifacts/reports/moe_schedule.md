# MoE unit order and duplicate expert reads (prior-art mechanisms A / B)

Base: `artifacts/cas/6402f5f3…d2b3e5.py` (board 2.3746). Candidate with the arm:
`src/okc_loop/vendored_work/moe/moe_dup.py` (line cites below are that file; the base is identical
modulo +20 lines). Inputs: `problems/muse_moe_w4a16/reference.py`, `artifacts/reports/moe_bytes.md`,
`artifacts/reports/prior_art_trxphxle.md`, `artifacts/reports/costtab.md`.

**Verdict: Arm A (heaviest-expert-first) is already in the kernel — the cost table *is* the
finish-time balance. Arm B (duplicate expert reads) is missing: 5.47% of the fetched bytes at bs=256,
1.53% at bs=128, 2.8% at bs=16, 0% at bs=32/8/1. Arm B is what was implemented, behind
`OKC_MOEDUP` (default 0).**

---

## 1a. How units are ordered within a launch — current vs Arm A

| order decision | what the kernel does now | Arm A would do | verdict |
|---|---|---|---|
| fat-unit queue (`!OKC_WGS`) | `S.forder` = **LPT: largest n-tile count first**, stable by job index (`route`, :592-603) | identical | **already heaviest-first** |
| which CTA carries which work | `skinny_split` (:1805): stream-K cost split with the `OKC_FCOST*` fat/skinny ratios, then a run-time atomic tail pool | identical | already cost-balanced |
| **WGS units (bs ≥ 32: the three big scored shapes)** | per-CTA contiguous stage range from the cost prefix: `wgs_prefix` builds `S.wcp[j] = Σ 512·wgs_cost(pb, n-tiles_j)`, `wgs_range` hands CTA c the interval `[c·T/G, (c+1)·T/G)` of that cost (:2339-2365) — i.e. **every CTA carries exactly T/132 of the modelled cost** | sort units by predicted cost, heaviest first | **no-op**: the makespan of an equal-cost partition cannot be changed by the order inside a range |
| first unit of a range | phase A runs its *trailing partial unit first* (`wgs_split`, :1978) so the split unit's epilogue is not exposed at the phase end; phase B keeps range order because its first stages must wait for phase A's units in that order | identical | already the reorder that matters |
| within a unit | 16 K-stages ascending through the 3-slot ring | identical | — |

The cost source Arm A would use is the one already there (`OKC_WGSA/WGSB`, :1860-1861) and it
already orders *and sizes*: `wgs_cost` sizes the per-CTA ranges, so finish-time balancing is
collected analytically rather than by issue order. Prior art's +1.1–1.3% came from adding
finish-time balancing to a kernel that claimed units dynamically without cost weighting; ours has
had it since the stream-K split, and `costtab.md` shows exactly what is left in it (cost-table
retune: bs32 +0.45%, bs128 +0.94%, bs256 +0.17%) — that residual is the cost model's error, not an
ordering effect. **Do not spend a check on Arm A.**

## 1b. Does an expert serving several jobs re-read its weight tiles? Yes — and how much

A job streams the expert's whole matrix: 768 units × 36,864 B = **28,311,552 B per expert per job**
(w13 18.87 MB + w2 9.44 MB). Jobs are built per expert (`route`, :564-575):

```
nfj = tot > SK_T ? (tot + FT_T - 1)/FT_T : 0   // OKC_FT_T = 64 (:250) -> ceil(tot/64) jobs
nsj = 0 < tot <= SK_T ? 1 : 0                  // SK_T = 8  (:348)  -> one job
```

so a job re-reads only when `tot_e > 64` (MAIN/WGS, `WGS_NT = FT_T/8` = 8 n-tiles, :1844) or
`tot_e > 8` (MID's `OKC_SPLIT`). `POPULARITY = 0.03` adds a Student-t(2 dof) draw to the router
scores (`reference.py:65-68`) and the `busiest` relabel makes experts 0-31 the hottest 32 of 256, so
at bs=256 a couple of experts get 65-117 local routes while the mean is ~8 — the tail, not the mean,
causes the split.

Replica (numpy, 1200-4000 seeds/shape, same draw as `generate_input`; cross-checked against
`moe_bytes.md`'s independent replica — agreement ≤1% on every row):

| shape | bs | build (split cap) | distinct experts | jobs | dup = Σjobs/Σdistinct | duplicate bytes | share of fetched |
|---|---|---|---|---|---|---|---|
| decode 1 | 1 | TINY (-) | 2.55 | 2.55 | 1.000 | 0 | 0% |
| verify 1 | 8 | TINY2 (-) | 8.44 | 8.44 | 1.000 | 0 | 0% |
| decode 16 | 16 | MID (8) | 11.92 | 12.27 | **1.029** | **9.6 MB** | 2.77% of 347.0 MB |
| decode 32 | 32 | MAIN/WGS (64) | 15.76 | 15.76 | 1.000 | 0 | 0% |
| verify 16 | **128** | MAIN/WGS (64) | 22.58 | 22.93 | **1.015** | **9.9 MB** | 1.53% of 648.6 MB |
| verify 32 | **256** | MAIN/WGS (64) | 24.75 | 26.18 | **1.058** | **40.4 MB** | 5.47% of 738.3 MB |

(`moe_bytes.md`: 1.000 / 1.000 / 1.027 / 1.000 / 1.015 / 1.060 — same answer.)

At bs=256 that is 1.43 extra full expert streams per launch (71% of launches have ≥1 expert over 64
routes, 0.35 extra at bs=128 — 33% of launches).

**Why the re-read misses L2 today.** The copies of one expert are 512 stage-slots apart in phase A's
stream (1024 in the flat job order). All 132 CTAs sweep their own cost interval in parallel, so a
stage is read at `τ·frac(x/Δ)` with `Δ = T/132`; two positions `D` apart are read
`(D mod Δ)/Δ·τ` apart. For a duplicated pair at bs=256, `D = 512·c_A` is 5.08 CTA-ranges, whose
residual is ~0.3Δ ≈ **~45 µs** (phase A takes ~139 µs) — and 738 MB streams through a 50 MB L2 per
call, i.e. a line's life is ~14 µs. On top of that every weight copy is tagged
`evict_first` (`0x12F0…`, :304-318), which discards it on purpose. So the second job's identical
28.31 MB is a guaranteed DRAM read.

## 2. The arm implemented: Arm B, L2-retention variant (`OKC_MOEDUP`)

Both alternatives were checked against the code first:

* **Multicast is not expressible here.** `okc_run` launches a plain grid, 1 CTA/SM, no cluster
  dimension (`cudaLaunchKernelEx` with only the PDL attribute, :2902-2906); the weight copies are
  1D `cp.async.bulk` of 36,864 B (:707, :2000) and `multicast::cluster` exists only for
  `cp.async.bulk.tensor` (needs a tensor map per weight array). Worse, the stream-K partition gives
  every CTA a *distinct* slice of the stage space, so two CTAs never want the same tile at the same
  time anyway — multicast would need the schedule rebuilt first.
* **Bigger jobs (one job per expert, up to 128 tokens) would delete the duplicate outright and are
  register- and smem-blocked.** 128 tokens = 16 n-tiles ⇒ `d[16][4]` = 64 f32/thread on top of the 64
  non-accumulator registers = 128, vs 113 available (65536 regs / 576 threads, `d[8][4]` today = 32,
  measured 96 used); and `WGS_XS = WGS_NT·4096` would add 48 KB to the x ring while the whole launch
  sits at 231,120 B of the 232,448 B opt-in limit (`okc_smem_bytes` = Smem 22,224 + 3·36,864 +
  3·32,768), i.e. ~1.3 KB spare. Raising the cap to 96 tokens (12 n-tiles, 20 MB of the 40.4 MB)
  needs `WGS_NX` 3→2 to pay for the x ring (2·49,152 = today's 98,304 B), which trades a
  measured-tuned ring depth for 0.5% — not shipped.

What is left is the ticket's fallback: **keep the first copy's tiles hot for the second job.** The
arm tags only the fetch of a job whose next job is the same expert (they are consecutive by
construction) with `EVICT_LAST` (`0x14F0…`) instead of `EVICT_FIRST`; all other weight fetches stay
evict-first, so the streaming traffic evicts around the kept set instead of through it. Two build
paths carry every scored shape: `ring_issue` (skinny ring: bs=1/8/16) and `wgs_wproducer` (bs≥32).

```c
// moe_dup.py:301-319
#ifndef OKC_MOEDUP
#define OKC_MOEDUP 0
#endif
#if OKC_MOEDUP
#define L2H_DUP_ARG(S_, j_) , "l"(((j_) + 1 < (S_).njF + (S_).njS && (S_).jE[(j_) + 1] == (S_).jE[j_]) \
                                  ? 0x14F0000000000000ull : 0x12F0000000000000ull)
#else
#define L2H_DUP_ARG(S_, j_) L2H_ARG          // shipped: unchanged operand
#endif
```
used at :707 (`ring_issue`, from `j = njF + v/rbs`) and :2000 (`wgs_wproducer`, from `j = u >> ubl`).
Not tagged: `fat_issue` (:1149) and the FAST1 prologue (:2525) — the fat queue exists only in the
MAIN *fallback* build (PRMT scale decode unavailable, not the scored path) and the FAST1 prologue
only in the M≤8 builds, where no expert is ever split.
A cache hint cannot change results; with the flag off the operand is the shipped `0x12F0…`, so the
compiled code is identical (`cu_build.py` passes on MAIN/MID/TINY/TINY2 with and without the flag;
a `#error` probe under `#if OKC_MOEDUP` confirms `-D OKC_MOEDUP=1` really reaches the compiler).

**Enable:** Python `MOEDUP = True` in the candidate (:190, appends `-DOKC_MOEDUP=1` to every build)
or `-DOKC_MOEDUP=1` directly. `lint`: PASS (same as base, 0 findings).

## 3. Expected gain, with the prior-art bound

Upper bound = every duplicate byte served from L2 instead of DRAM, at each shape's measured rate
(`yours` µs and TB/s from `moe_bytes.md` §4; board weight 1/6 per shape):

| shape | duplicate bytes | µs saved | shape | board |
|---|---|---|---|---|
| verify 32 (bs=256) | 40.4 MB @ 3.55 TB/s | 11.4 µs of 208.0 | 5.5% | **+0.94%** |
| verify 16 (bs=128) | 9.9 MB @ 3.75 TB/s | 2.6 µs of 172.8 | 1.5% | **+0.26%** |
| decode 16 (bs=16) | 9.6 MB @ 3.80 TB/s | 2.5 µs of 91.3 | 2.8% | **+0.47%** |
| bs=32/8/1 | 0 | 0 | — | 0 |
| **all six** | 59.9 MB | | | **+1.67% ⇒ ~168 ranks** (101 ranks / +1%) |

Prior art measured this mechanism's ceiling as **~1.5% expert board at 8.6% duplicate bytes** (their
V32); our duplication is 5.47% at bs=256 plus 1.5% at bs=128 plus a second source at bs=16, so
1.2–1.7% is the same number scaled to our byte fractions — consistent, and ~120–170 ranks.

Honest expectation below the bound: the hint is a priority, not a reservation, and the keep set is
28 MB per duplicated pair (one pair at bs=256 fits the 50 MB L2; the 29% of launches with two pairs
request 56 MB and will thrash among themselves). Landing half of the bound = **+0.8% board ≈ 80
ranks**. If it lands at ≈0, the residual lever is the reuse *distance* (the pair's copies are ~45 µs
apart because 5.08 CTA-ranges is not an integer: interleaving the copies' units in the stage space
would make the re-read ~1 µs apart) — a stream-K range rewrite, only worth doing if this arm shows
the policy is honoured.

## 4. Kill criterion

Paired local A/B on `judge_replica` (A/A within ±0.3%); the judge's own spread is ±1.5–2%, so:

* **Kill** if the six-shape geomean gain is **< +0.3%** — under the noise floor, not worth a check.
* **Kill** (bug, not noise) if bs=1, 8 or 32 move more than ±0.3%: those shapes have zero duplicate
  jobs, so `OKC_MOEDUP` cannot fire there and any movement is measurement or a build-tag mistake.
* **Kill** if bs=256 *regresses* while bs=128/16 improve: that is the 2-pair L2 thrash case, and the
  fix (cap the keep set, i.e. tag only one expert per launch) is a different arm.
* Keep and iterate (unit interleave) only if bs=256 lands in +0.2…+0.6%, i.e. the policy works but
  the reuse distance is the limit.
