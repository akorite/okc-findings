# attn_fill — what the attention constant is made of, what the literature removed, and the one thing left

Base row: `src/okc_loop/vendored_work/layer/klr_fci5.py`, judge **2.28370 / 9,905.1 µs** (job
`9743b6e71fbd42fd`), profile `profiles/klr_fci5.9743b6e71fbd42fd.txt`. Candidate:
`src/okc_loop/vendored_work/layer/attn_fill1.py` (copy of the base; `OKC_FINWARM` flipped to 1,
new `OKC_BULKRED` default 0). Nothing here was run on a GPU: no judge, no `colab`, no `okc check`.
Every µs below is either **measured** (from that profile or a report cited by path:line), **derived**
from those numbers (marked `[DER]`), or a **cycle estimate from the source** (marked `[EST]`, model in §2).
Clock for all cycle↔µs conversions: **1.76 GHz** (0.568 ns/cycle), the same 1.7–1.8 GHz band the file's own
arithmetic uses (`muse_arch_levers.md` §"A3").

---

## 0. Bottom line

1. **The constant is 5.0–5.3 µs, not 6.3.** `muse_arch_levers.md` §A1 fits `t = fixed + S/R` through **two**
   points (B=16 and B=32) at *identical grid size* (256 CTAs, 2/SM), so a per-CTA cost and a per-byte cost are
   collinear in that fit — and it reports the symptom: a marginal of **4.85 TB/s**, above the machine's 4.2–4.45
   ceiling (`research_06_microarch.md`), which cannot be physical for HBM. Adding the B=1 cells, which run the
   *same* kernel with a 32-CTA grid, breaks the collinearity: three-point fit over D1S/D16S/D32S gives
   **5.0 µs + 0.29 µs/MB (3.4 TB/s)**, i.e. no contradiction, at the cost of a *smaller* constant. The 6.3 is
   an upper envelope that borrows ~1 µs of streaming into the intercept.
2. **Of that ~5.0 µs, ~2.0 is code and ~3.0 is latency that no code change removes** (two DRAM round trips for
   page index + K/V, one L2 round trip for the cross-CTA reduction, one for the read-back, plus the warp-issue
   chain in between). §2 prices all of it.
3. **The merge is already at its atomic minimum and both of its alternatives are measured closed on this file**:
   clusters/DSMEM lose on our grids (`_plan7:3821-3828`, `attn_cores.md` §3), the cooperative finalize loses for
   q_len 1 (`_plan7:3817-3819`). The key-half exchange (`:2646-2677`) already removed the 2× duplication, so a
   q1 CTA issues **256 `red.v4` = exactly 4 KB / 16 B** — one atomic per output word, the floor.
4. **The largest addressable row is cold instruction fetch of the once-per-CTA path** (the judge flushes L2
   before every replay). Our file warms exactly one function (`fin_nc1`) and ships that warm *off*
   (`:197 #define OKC_FINWARM 0`) even though it measured **+0.16%**, and only for the three cells where
   `!coop && nc == 1` holds. That family is the recommendation.
5. **The brief's "+1.91% for 3 µs" is not reachable**, and this is a property of the decomposition, not of
   effort: 3 µs is 60% of the whole 5.0 µs constant, and §2 shows only ~1.2–2.2 µs of it is code-shaped at all
   (a merge floor of ~1.1 µs that is 1 atomic/word plus 2 round trips, a prep of ~0.6, cold fetch of 1–2).
   Realistic capture from everything in §5: **+0.22 to +0.58%** (their bands sum), i.e. band **+0.2 to
   +0.9%** if the cold-code family is attacked systematically rather than one function at a time.

---

## 1. What the per-request path is, in the source (the shape of the decomposition)

`attn_kernel<L,false,MODE>` → `segment()` (`:1500-2990`). Grid `(n_splits, B)` (`:3087`), `chunk` from
`n_splits` (`:3101`, `OKC_MAGICDIV` at `:3109`). `_plan7` (`:3790-3860`) for the scored cells:

| cell | template | plan | CTAs | tiles/CTA | read |
|---|---|---|---|---|---|
| D1S | `<1,false,13>` (MODE\|4 = `ONE`) | 32 splits, nc 1, no coop | 32 | 1 | 0.52 MB |
| V1S | `<8,false,15>` | 33 splits, nc 1, **coop** | 33 | 1 | 0.52 MB |
| D16S | `<1,false,1>` | 16/req, nc 1 | 256 | 2 | 7.8 MB |
| V16S | `<8,false,3>` | 8/req, nc 1, coop | 128 | 4 | 7.8 MB |
| D32S | `<1,false,1>` | 8/req, nc 1 | 256 | 4 | 15.5 MB |
| V32S | `<8,false,3>` | 4/req, nc 1, coop | 128 | 8 | 15.5 MB |
| D1F | `<1,false,11>` | 252 splits, nc 8, coop | 252 | 5 | 18.9 MB |
| V1F | `<8,false,11>` | 252 splits, nc 8, coop | 252 | 5 | 18.9 MB |

(`muse_arch_levers.md` §A1 template list + `_plan7`; `pairs = Hkv/2 = 1` on this shard, `reference.py:31`.)

The whole cell is one launch. The three things a CTA does exactly once are: the **prologue** (`:1512-1845`),
the **merge/departure** (`:2645-2790`), and — for 1/n_splits CTAs — the **finalize** (`fin_nc1` `:1079-1135`,
`sum_copies` `:1001-1077`). Everything else is the tile loop, which is priced by the marginal term.

---

## 2. Cycle decomposition (`[EST]` unless the row says measured)

Model: instructions counted from the cited region of `klr_fci5.py`; latency per dependent L2 round trip
**600 cycles**, per DRAM round trip **900 cycles**; issue width 1 instr/cycle/lane on the once-path (128 threads
= 4 warps of the 2-CTA/SM build, `Cfg::NT = 128` at `:272`); a `bar.sync` **40 cycles**; `__sincosf` **≤40**.

| # | item | where | cycles | µs | addressable? |
|---|---|---|---|---|---|
| P0 | **cold instruction fetch** of the taken path (every scored replay runs on a flushed L2 — the same 256 MB flush our own `_plan_time7` uses, `attn_plan.md`) | `segment<1,false,13>` body ≈ 8–12 KB of SASS for the taken branches | 2 000–3 500 | **1.13–1.99** | **yes** — the only large row that is |
| P1 | `griddepcontrol.wait` for the predecessor (the qkv GEMM, `PDL = True`, `PDL_MAX_T = 128`, `_pdl`, `:11381`) | `:1532` | 0–600 | 0–0.34 | no: the primary fires `launch_dependents` early (`attn_cores.md` §1-2: the qkv GEMM's producer lane streams its pre-wait stages, then triggers) so only the tail lands on us; `pdlkv.md` §4 measured ±0 for moving work across it |
| P2 | page-index staging, 2×256 B, *above* the wait | `:1521-1526` | 0 exposed | 0 | no — already hidden (`pdlkv.md` §"premise") |
| P3 | q rows (128 × 16 B cp.async) + new k/v (32) + `inv_freq`/scales/`cache_loc` + 16 gate-row `prefetch.global.L2` | `:1537-1562` | 40–60 instr/lane | 0.02–0.03 | no |
| P4 | RoPE table: `L·32` = 32 `sincos` + Cody-Waite 3-term reduction, one warp, then `csync` | `:1564-1584` | 150 exposed (ifreq RTT overlapped by P3) | 0.09 | no (0.05 at most: make each lane compute its own 4 pairs and drop the barrier) |
| P5 | q load + RMS + RoPE + pack (1 pass for L=1) | `:1718-1745` | 300–400 | 0.17–0.23 | partly — the *identical* q work is redone by every split of a request, but they are parallel, so only one copy is on the wall |
| P6 | new k/v norm + RoPE + e4m3 + the cache patch pass (32 lanes live) | `:1760-1832` | 300–500 | 0.17–0.28 | no (needed once per request, done by one split) |
| P7 | first tile: `TILE/RPP` = 4 rows/lane × 2 (k,v) `cp.async` + `cp_wait<1>` + barrier | `:1583-1626`, `:1893` | 300 exposed | 0.17 | no — 16 KB in flight is irreducible; `OKC_PDLKV` variants A/B/C measured −0.9/−1.7/−0.08% trying to move it (`pdlkv.md` §4) |
| P8 | QK wgmma (4 k-steps) + online softmax + PV (8 n-steps) + drains, **per tile** | `:1886-2660` | 800–1 200 /tile | 0.45–0.68 /tile | no (this is the marginal term's compute) |
| P9 | merge: 256 `red.global.add.v4.f32` (4/lane from 2 warps — the key-half exchange at `:2646-2677` already dropped the duplicate half) + 16 `red_f32` row sums, `bar.sync`, `atom.acq_rel` ticket, spin | `:2719-2848` | 1 300–1 600 | **0.74–0.91** | **no** — 1 atomic per 16 B output word is the floor; the two round trips (ticket, and the read-back in P10) are the reduction itself |
| P10 | finalize (`fin_nc1`): 3 dependent L2 reads (partials, row sums, gate rows) + normalise + RMS·sigmoid + store + workspace zero | `:1079-1135`, `:2905` | 900–1 100 on **one** CTA per request | **0.51–0.63** | no for q1 (coop measured worse, `:3817-3819`); already removed for the q8 cells (coop, 13.0→10.2 µs, `:3828-3830`) |
| P11 | straggler skew across splits/requests (the cell ends on the slowest) | — | 500–900 | 0.28–0.51 | no |
| P12 | **borrowed** pooler L2 prefetch: 4 MB `cp.async.bulk.prefetch.L2`, `T ∈ {1,8}` only | `:3079-3086` + host `_PF7` `:11347` | 4 MB / 4.2 TB/s | **0.95** at D1S/V1S/D1F/V1F | not ours — it is a deliberate loan to the *next* kernel (measured a win); listed because it is charged to the attention cell and must not be mistaken for attention's own cost |
| P13 | 256-CTA contention — **B≥16 cells only** | — | **unattributed, 2 000–5 000** | **1.1–2.8** | unknown — the honest hole; §6 narrows it to cp.async backpressure |

Checks: D1S = P0+P1+P3..P11+P12 ≈ 1.5+0.1+0.2+0.09+0.2+0.25+0.17+0.6+0.8+0.57+0.4+0.95 ≈ **5.4 vs measured 5.3**
(the P0 band's low end with P12 at 0.6). D16S = same chain minus P12 plus 2 tiles of P8 plus P13 ≈
1.5+0.1+0.2+0.09+0.2+0.25+0.17+1.2+0.8+0.57+0.4+P13 = 5.5+P13 = **7.9 measured ⇒ P13 ≈ 2.4**.
D32S: 5.5+1.2 more of P8 = 6.7+P13 = 9.5 ⇒ P13 ≈ 2.8. V16S/V32S: +0.5 (P5/P6 are 4× larger at L=8) ⇒ 11.7/15.1
with P13 ≈ 2.4/3.1. **The whole B≥16 residual lands in P13 and it is the same ~2.5 µs in all four cells** —
i.e. it is grid-shaped (a per-CTA or per-wave cost), not byte-shaped. Three candidate causes, and **the file's
own tuning record eliminates the first**: (i) *L2 atomic same-address serialisation* — `_TUNED7:3774-3776` records
that for the 250-split D1F grid `nc = 1/2/4` costs **+26/+16/+4%**, i.e. ~3 µs of pure same-address add
serialisation at depth 250; at the sliding cells' depth 8–32 the same sweep is flat (`attn_plan.md`: `nc`
candidates 0/16 picks, medians within 1%), so this term is **bounded at <0.5 µs for D16S/D32S**. (ii) *cp.async
queue backpressure* — 3 stages × 16 KB × 256 CTAs = 12 MB of in-flight copies against a ~50 MB L2 and 4.2 TB/s of
HBM, i.e. ~3 µs of drain at the wall: the leading candidate. (iii) the `bar.sync` + ticket chain repeated per
CTA (1300–1600 cycles each, but they overlap across CTAs, so this can only contribute its tail).

### Why this differs from `pdlkv.md` §1's "8.42 µs" table
That table charges launch+dispatch (2 µs) and the page-index fetch (0.7) to the *cell*. Both sit above
`pdl_wait()` (`:1521-1532`) and `profile` kernel durations start at the first CTA, so neither is in the 5.3:
`pdlkv.md`'s own three variants (A/B/C) moved exactly that work and measured −0.87/−1.65/−0.08%. What remains
of its table is P7+P8+P9+P10+P11 = 3.5–4.6 µs, which is this report's P7–P11. **P0 (cold i-fetch) and P12 (the
prefetch loan) are the two rows nobody had counted**, and they are 2.1–2.9 µs of the 5.3 at B=1.

---

## 3. How the people who solved it removed it

### 3.1 The per-CTA prologue: dedicated IO warps, not a shorter prologue
**XQA** (TensorRT-LLM's/JIT'd from FlashInfer's tree — [`csrc/xqa/mha.cu`](https://github.com/flashinfer-ai/flashinfer/blob/08d74a47163138273af02f034a500449b9a2a6d0/csrc/xqa/mha.cu),
`mha_sm90.cu`) does not shrink the prologue: it gives it to warps that are not on the math's critical path —
`nbIOWarps = 4`, `gemm0WarpGrpTileNbTokens = 64`, `dimCta = {gmmaWarpsPerGrp, 1, 3}` (three warp groups, 2
math + 1 load; `research_04_engines.md` §3.2, read at commit `08d74a47`). Its ring is `nbKBuf = 2`
(`:142-163`) — **the same ~32 KB in flight we run** — so XQA's small-batch advantage is CTAs-per-request, not
depth. Its merge is *our* merge: **last-CTA on an acq_rel counter, self-resetting** (`mha.cu:2756-2762,
2799-2900`); XQA never reduces across a cluster and never launches a second kernel for it.
**Consequence for us:** the prologue-delegation trick needs ≥8 warps of consumers to hide 4 warps of producer;
our `!WS` build has 4 warps total (`Cfg::NT = 128`, `:272`), so the producer would *be* the math. This is why
`WS` is flat-only (`:275`) and why the per-request path is not missing an obvious XQA structure.

### 3.2 FlashInfer: the split/merge structure is chosen per shape, and short shapes get no split
[FlashInfer's attention API](https://docs.flashinfer.ai/api/attention.html) exposes single-decode
(`single_decode_with_kv_cache`, one CTA per request, no workspace), batch decode with a separate
`merge_states` kernel, and XQA (`xqa_batch_decode_with_kv_cache`) — the merge is *either* a second kernel
(FA2 path, `merge_states`; also [`cascade.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/cascade.cuh))
*or* in-kernel. [The paper](https://arxiv.org/pdf/2501.01005) §decode makes the same point our `_plan7` does:
the split only pays once `keys/CTA` exceeds the fill. Our sliding cells are 1–8 tiles/CTA, i.e. the regime
where the literature's answer is "do not split" — and we cannot follow it, because 32 tiles at 1 CTA each is
524 KB on one SM (≈9 µs at the per-SM gather rate `[DER]`).

### 3.3 The one published instruction that removes the merge's drain: `cp.reduce.async.bulk`
FA3's [`hopper/copy_sm90_bulk_reduce.hpp`](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/copy_sm90_bulk_reduce.hpp)
and FlashInfer's cascade combine reduce a split's partial with **one TMA bulk reduction from shared memory**
(`cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32`) instead of per-lane `red` — the same instruction
`attn/dual.py:243-249` ships in our own tree (`bulk_red_add_f32` / `bulk_commit` / `bulk_wait_all`), and the
one thing `klr_fci5.py` does **not** have (grep: 0 occurrences). `research_02_analyzers.md:60` already priced
it once as "0…1 µs, the merge is not our critical path (fill is)" — §2 agrees with the *number* and disagrees
with the *reason*: the merge is 0.74–0.91 µs of a 5.0 µs constant, i.e. 15–18%, so it is neither the critical
path nor ignorable.

### 3.4 The competitor who actually did it: A13/A14 (the single-launch fused prep+attention+merge)
[`Kushagrabainsla/muse-spark-open`](https://github.com/Kushagrabainsla/muse-spark-open), vendored at
`.vendor/muse-spark-open/muse_attention_decode/EXPERIMENTS.md:188-232`. Their sequence, with their own phase
clocks (cycles, at bs1/16/32 after A14): **prep 4200/4800/4400, loop 4800/7350/14450, push ~650, barrier
wait ~1500, merge 2700/1900/1700.**

* **A13's finding, which is the single most valuable line in the whole literature for us:** *"the merge's
  ~6600 cycles were instruction fetch. Run twice without the L2 flush: merge 6928 → 2298 cycles, prep 3514 →
  2553, whole kernel 10.8 → 7.8 µs. The judge flushes L2 before every replay, so code that runs once per CTA
  (prologue, epilogue) is fetched from DRAM on every call. Straight-line (unrolled) merge code was worse
  (8758): bytes matter, not branches."* Their fix — a `__noinline__` dry call of the merge from the idle
  producer warp during the last copy — took the kernel 10.8 → 8.9 µs and vslide1 13.0 → 10.2.
* **A14**: partials *pushed* into the merging CTA's shared memory (`st.shared::cluster`, no wait) + a per-lane
  merge from local memory: slide-1 8.9 → 8.5, 16: 12.1 → 11.4, 32: 16.6 → 15.8 µs. **That is a DSMEM cluster
  win of 0.4–0.8 µs** — on *their* kernel, whose merge baseline was 2700 cycles because it was warm-but-big.
* A15 (a separate CUDA merge kernel with PDL) measured **no gain**: 28.4 vs 27.6, 119.8 vs 115.4, 223 vs 214 µs.
  So "one launch, in-kernel merge" > "two launches, PDL" for this shape class — which is what we ship.
* Their prologue is 4.2–4.8 kcycles = **2.4–2.7 µs**, and the pre-A13 note gives the reason the brief mentions
  `sincosf`: *"prep ~5500 (8 sincosf per thread)"* before they tabled RoPE to one `(token, pair)` entry. Ours is
  P4+P5+P6 = 0.4–0.6 µs, because we tabulate 32 angles on one warp instead of 8 per thread.

### 3.5 Hiding the prologue behind the previous kernel
PDL (`cudaLaunchAttributeProgrammaticStreamSerialization`, `:3606-3610`; the CUDA programming guide's
[programmatic dependent launch](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization)
section) is the only published mechanism that overlaps a *prologue* with the *previous kernel*, and we already
run it on every attention launch (`_pdl(T)`, `:11381`): the page-index group is issued above
`griddepcontrol.wait` (`:1521-1532`) and `pdl_launch_dependents` (`:1533`) releases the o-projection. The three
prologue reorderings this enables were measured at −0.9/−1.7/−0.08% (`pdlkv.md` §4). **`cp.async.bulk.prefetch`
is the same idea for data, and our kernel already uses it — on behalf of the *pooler*, 4 MB, `:3079-3086`
(P12).** Warp-level pipelining (a producer one tile ahead) is what our `STAGES` ring is; deepening it is
contraindicated (`OKC_STN 3`, `attn_gap.md` §4: "deeper pipelines (3–4 stages) were within noise or worse" on
the publisher's long-context shapes, and our own judge check measured STAGES 3→4 negative).
**Persistent CTAs that skip the prologue after the first call are impossible here**: every layer is its own
launch (62 per step) with its own page table, grid and workspace, and the CUDA-graph replay flushes L2 between
calls — so "the first call" never happens for a given launch.

### 3.6 The one structural axis nobody has crossed (and why it does not help us)
`attn_gap.md` §2/§4 identifies the untried axis: route the **windowed** shapes through the flat/persistent
partition (Heidari et al., *Paged Flash-Decoding at Memory Speed*, `https://arxiv.org/html/2610.34907v1` via
`attn_gap.md`, which reports 86% of H100 peak with striped page lookup + producer-ahead + a one-wave persistent
grid). Our layer file's `WS` gate is `if not window and 1 < B <= 512` (`:3795`), so sliding can never reach it.
**But the file's own fit forbids it:** the `WS` build's fixed cost is **9.20 µs** against the `!WS` build's
6.30 (`muse_arch_levers.md` §A1) — a one-wave producer/consumer grid costs 2.9 µs more per launch before it
reads a byte, and sliding cells are 0.5–16.8 MB. Predicted: D16S 7.9 → ≈11 µs. Closed, by the numbers we
already have, without running anything.

---

## 4. Pricing the merge substitution (the brief's item 3)

What the merge costs today, per CTA: **256 `red.v4`** (2 warps × 32 lanes × 4; the key-half exchange at
`:2646-2677` already halved it, `OKC_NOXCH1` = "measured slower", `:201`) **+ 16 `red_f32`** row sums + one
`bar.sync` + one `atom.acq_rel` ticket + (1 CTA in n_splits) the `fin_nc1` read-back. 256 × 16 B = 4 KB =
exactly the partial: **one atomic per output word, the arithmetic floor.**

| substitution | what it removes | what it cannot | measured verdict on *this* file |
|---|---|---|---|
| **DSMEM cluster push + per-lane merge** (A14) | the global workspace entirely: partials land in the merging CTA's smem, the merge reads local memory | nothing else — the cluster barrier still costs ~1500 cycles (A13's clock) | **closed**: clusters measured slower on exactly these grids (`_plan7:3811` "clusters measured slower here"; `:3821-3828` "plain split-KV … beats clusters here"; `attn_cores.md` §3). Occupancy is not the blocker — `MINB = 2` (`:278`) admits 2 CTAs/SM and a cs=2 cluster is legal at 2/SM — the blocker is that an empty cluster CTA still runs the cluster epilogue and cluster placement restricts the wave. A14's 0.4–0.8 µs was won on a kernel whose merge was 2700 cycles; ours is 1300–1600 |
| **one TMA bulk reduction from smem** (`cp.reduce.async.bulk`) | 256 lane atomics → 1 op; the red-drain before the release becomes one `cp.async.bulk.wait_group 0`; `ws_l` can fold into a padded 72-float row | the L2's own add work (same 4 KB per CTA into the same words) and the ticket + read-back round trips | **unmeasured; written** (`OKC_BULKRED`, §5.2). Honest band from A13/A14: the merge phase is 1300–1600 cycles ≈ 0.74–0.91 µs, of which the issue+drain half is ~40% ⇒ **0.1–0.35 µs** |
| **separate merge kernel with PDL** (FA3/FlashInfer `merge_states`) | — | adds a launch | **measured no gain** in our own tree and in A15 |
| **cooperative finalize** (every CTA finalizes a slice) | the one-CTA tail (P10) | the ticket round trip | **already on for q8** (13.0 → 10.2 µs, `:3828-3830`), **measured worse for q1** (13.0–13.1 vs 12.6–12.9, `:3817-3819`) |
| **per-warp `red.release` instead of `bar.sync` + one ticket** | the CTA-wide barrier before the release | needs a returning atom anywhere (last-arrival test), so the ticket stays | same change measured on the MoE (`klr_unitend_fences.md` §2, `OKC_WGSFR1`): "+0.234% … noise" |

**Verdict on item 3 of the brief: the DSMEM substitution is worth 0 and is measured closed; the bulk-reduce form
is worth 0.1–0.35 µs, not 1.5–3, because the 2× atomic duplication it would have removed is already gone and
what is left (one atomic per word, two round trips) is the reduction itself.**

---

## 5. Ranked recommendation

### 5.1 #1 — ship the cold-code warm and extend it to the rest of the once-per-CTA path
**The flip alone touches 3 of the 8 cells (D1S .180, D16S .088, D32S .070 — the only cells whose plan has
`coop = 0` and `nc = 1`, both required by the guard) ⇒ 0.25 µs × 0.338 = +0.08%. Extending the same dry call to
the cooperative slice finalize and to the merge block reaches the other five (D1F .063, V1F .040, V1S .114,
V16S .048, V32S .035) ⇒ 0.3–0.6 µs on all eight = +0.19 to +0.38%, family band +0.15 to +0.50%.**

* The measured instance is already in the file and is switched **off**: `:197 #define OKC_FINWARM 0`, whose
  own note says it is a dry call of the last-CTA finalize during the prologue, measured **+0.16%** on this
  board. `attn_fill1.py` ships it **on**. Zero numerics risk: the dry call passes `dry = 1` and every store is
  predicated (`fin_nc1:1127-1129`).
* Why this is the biggest available row rather than a rounding error: P0 is 1.1–2.0 µs of the 5.3, it is the
  one row the competitor's phase clocks isolated as *pure i-cache* (their whole-kernel delta from the L2 flush
  alone was **3.0 µs**, and 3.3 of their merge's 6.6 kcycles), and our file warms **one** of four cold blocks.
  The other three: the merge block (`:2719-2779`, ~1.5 KB of SASS), the cooperative slice finalize
  (`:2846-2886`, ~1.5 KB), and nothing at all on the flat path. Extending the same `__noinline__` + `dry`
  pattern to those two is the mechanical follow-up; the coop one is the only one that reaches V1S/V16S/V32S
  (coop cells are excluded by `!p.coop` at `:1843`, precisely because the coop finalize is inline).
* Risk: none numeric; the cost is 1–3 KB of extra code in the cold region and one extra `csync` per launch.
  The failure mode of over-warming is a *smaller* win, not a wrong answer.

### 5.2 #2 — `OKC_BULKRED`: one TMA bulk reduction per CTA instead of 256 lane atomics
**0.1–0.35 µs/cell on the three q1 non-WS cells whose plan has `nc = 1` and `cs = 0` (D1S .180, D16S .088,
D32S .070 — D1F is `nc = 8, coop = 1` and is outside the flag's guard, §7) ⇒ +0.03 to +0.10%.** Written,
default off.

### 5.3 #3 — move the borrowed pooler prefetch off the attention prologue
**0.2–0.4 µs on the four B=1 cells (D1S .180, V1S .114, D1F .063, V1F .040) ⇒ +0.08 to +0.16%.** 4 MB
prefetched into L2 at `:3079-3086` *before* the page-index copies, against a cell that reads 0.5 MB (D1S) — the
KV gather queues behind its own loan. Issuing the same 4 MB *after* the first tile's `wgmma` keeps 3–4 µs of
slack for the pooler and puts attention's 524 KB first. Needs an A/B on both the attention cell **and** the
pooler GEMM (the loan exists because it was measured a win; this only changes its *position*, and the failure
mode is a pooler regression).

**These three stack and are disjoint: +0.22 to +0.58% (band +0.2 to +0.9%).** That is the honest number for
the whole fixed-fill programme, and it is 1/3 of the brief's target because 3 µs is 60% of the constant.

---

## 6. What the decomposition cannot say, and the one run that would

P13 (~2.4–2.8 µs, all four B≥16 cells) is unattributed. The instrument is already in the file and is compiled
out: **`PROF_INIT` / `PROF_MARK(k)` / `PROF_DUMP`, `:500-504`, enabled with `-DOKC_PROF=1`**, plus
`OKC_FPRO` (`:232`, prologue stamps 40–46 in `p.dbg` at stride 64) and `OKC_MRGDBG` (`:144`, merge marks 6/7).
One build of `attn_fill1.py` with `-DOKC_PROF=1 -DOKC_SEGDBG=1 -DOKC_FPRO=1 -DOKC_MRGDBG=1`, replayed cold-L2
at D16S and D32S, separates: tile-0 wait, per-tile issue vs math, the red phase, the ticket, the finalize, and
the prologue. That single profile decides the one open question: if P13 is the cp.async drain (the leading candidate,
§2), the answer is a *joint* `STAGES × n_splits` cut — 12 MB in flight at D16S is 3 µs of HBM time for 7.8 MB of
data, so halving the stages at fixed grid would drop in-flight to 6 MB and the drain to ~1.5 µs, at a cost the
file has never measured *at this grid* (its `STAGES` verdict is from the 1–8-tile/CTA cells and from the
publisher's long-context shapes, not from a 256-CTA sliding wave). If instead it is the ticket chain, the answer
is the per-warp release. **Do not spend a check before that profile exists**: every candidate for P13 in the
file's history was priced wrong until a profile was read.

---

## 7. Change shipped

`src/okc_loop/vendored_work/layer/attn_fill1.py` (982,256 B, line 1 unchanged, `ast.parse` ok, **ruff delta vs
`klr_fci5.py`: none**, 180 findings in each, identical message multiset after stripping line numbers):

* `:197` `OKC_FINWARM 0 → 1` — the measured +0.16%, sitting off.
* `:198-199` new `OKC_BULKRED 0`; `:481-489` the three helpers (`bulk_red_add_f32`, `bulk_commit`,
  `bulk_wait_all`, `fence_proxy_async_global`) — byte-for-byte the PTX `attn/dual.py:243-249` ships.
* `:2730-2735` the flag: `BREDC = OKC_BULKRED && L == 1 && !FLAT && !OKC_NOXCH1 && !CONT`, `bred = BREDC && p.nc == 1`.
* `:2750-2756` when `bred`, the q1 partial is *staged* into the free V stages (rows 0–15 are unique because the
  key-half exchange left warps 0/1 as the only reducers) instead of `red_v4`'d.
* `:2780-2789` `fence.proxy.async` → `csync` → thread 0: one `bulk_red_add_f32` of `L*16*64*4` = 4096 B →
  `commit` → `wait_group 0` → `fence.proxy.async.global`, which is the ordering A14/A15 and FA3's combine use
  before the CTA's release.
* With both flags at their shipped values (`FINWARM 1`, `BULKRED 0`) the kernel issues exactly the base's
  instruction sequence plus the base's own dry finalize call; with `OKC_BULKRED 0` the staging branch is folded
  away at compile time.
* Not run: any build. `tools/cu_build.py` is vacuous for this file (`pdlkv.md` §3: it compiles the empty string
  before `#include <torch/`), and the real gate is the sm_90a compile of `_att7` (89.7 s on the colab gate per
  `pdlkv.md` §3) plus one `okc check`. **`cp.reduce.async.bulk…add.f32` is sm_90+ only**; this file cannot build
  for sm_80 anyway (`segment()` calls `pdl_wait()` unconditionally and ptxas rejects `.wait` below sm_90,
  reproduced in `pdlkv.md` §2), so the instruction adds no new architecture constraint.
