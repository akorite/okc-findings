# `okc::moe_fused` — structural scan (target: ≥7% of the MoE board)

Scope: only the four requested angles + the ground truth they need. GEMM/inner-loop rewrites excluded
by ticket. Files: **B** = `artifacts/cas/6402f5f3…b5d2b3e5.py` (the scored MoE board candidate, 2.3746);
**L** = `src/okc_loop/vendored/keell0renz/muse_decode_layer.py` (newer lineage). All line refs verified.

---

## 0. Ground truth: the judge's own floor, per load

`problems/muse_moe_w4a16/reference.py:116-137` is the judge's speed-of-light model (the "floor" column
in `problems/muse_moe_w4a16/TASK.md`):

```
touched      = max(1, 0.75 * router_distinct)
per_expert   = 3 * 4096 * 4096                              # 50,331,648 weight elements
weight_bytes = touched * per_expert * (0.5 + 1/16)          # 0.5625 B/elem = 28.31 MB per expert
floor        = max((weight_bytes + act_bytes)/4.8e12, flops/989e12)
```

So the floor **is** "read the touched experts' NVFP4 weights once at 4.8 TB/s" (the math floor never
binds: 26 µs at bs=256 vs 108.9 µs). Inverting it against `TASK.md`'s floor column gives the *actual*
number of local experts the judge's router touches (busiest rank, ep8):

| load | bs | floor µs | floor bytes | touched experts | our median µs | our achieved BW | % of 4.8 TB/s | ours/floor |
|---|---|---|---|---|---|---|---|---|
| decode 1 | 1 | 11.4 | 54.7 MB | 1.93 | 24.93 | **2.20 TB/s** | 46% | **2.19×** |
| verify 1 | 8 | 37.3 | 179 MB | 6.32 | 67.82 | 2.64 | 55% | 1.82× |
| decode 16 | 16 | 52.5 | 252 MB | 8.90 | 91.28 | 2.76 | 58% | 1.74× |
| decode 32 | 32 | 70.8 | 340 MB | 12.0 | 125.3 | 2.71 | 57% | 1.77× |
| verify 16 | 128 | 101.3 | 486 MB | 17.2 | 172.8 | 2.81 | 59% | 1.71× |
| verify 32 | 256 | 108.9 | 523 MB | 18.5 | 208.0 | 2.51 | 52% | 1.91× |

(times = `median_ns` of job `a857807a067a4ad9`; score 2.3746 = geometric mean of the six `baseline/ours`
ratios, verified numerically. Board max at the floor = 4.39; top team 2.764 = 63% of floor.)

Two consequences that decide everything below:

1. **The deficit is per byte, not per shape.** 46–59% of 4.8 TB/s on all six loads, across 1.9 → 18.5
   touched experts, three different build bands (tiny `nw8+WS`, tiny2, mid, WGS `nw16+wgmma`) and both
   ring designs. A structural scheduling or sync cost would not be this flat; a per-32 KB-stage cost is.
2. **The memory pipeline is already 8× oversubscribed.** Bytes in flight needed to saturate 4.8 TB/s at
   ~0.7 µs loaded HBM latency = 3.4 MB total = **25 KB/SM**. The kernel holds, per CTA:
   WGS `WGS_NS(3)×36864 + WGS_NX(3)×32768 = 204 KB` (B:1823-1827, one CTA/SM); skinny
   `OKC_NS(3)×36864 = 108 KB`. That is 4–8× the requirement, and the author's own dead ends confirm the
   corollary: ring depth (`OKC_NS=4`, “ring depth OKC_NS”), warp-parallel prologue, cluster GEMV and
   speculative prefetch are *all* in the memory-parallelism family and all measured to nothing.

Bit-exactness of the byte account: one unit = 16 weight rows × 4096 K at 4-bit = `UNIT_CODES 32768` +
`UNIT_SC 4096` (`B:657`) = `UNIT_BYTES 36864`; 12288 rows/expert ⇒ 768 units ⇒ 28.31 MB/expert, which is
exactly the judge's `per_expert × 0.5625`. We move the floor's bytes and no more: **52–59% of peak is
spent per stage, not on extra traffic.**

---

## 1. Cross-CTA expert-weight sharing (cluster + TMA multicast) — **0%**

**Does the job assignment ever give two CTAs the same expert's bytes?** No, never. Two different
statements must be separated:

* Two CTAs *do* work on the same expert. Units of one expert are spread over the whole grid
  (`B:1574`/`L:5298` `nS = S.njS * RB_A − dyn_pool(...)`, split by `muldiv_g(nS, blockIdx.x, G)`;
  WGS uses the cost-weighted stream-K ranges `L:6157-6200`). This CTA does rows 16·v, that CTA rows
  16·(v+1).
* Those bytes are **disjoint**. Every copy address is `wAu + (jE[j]*rbs + rb) * UNIT_BYTES`
  (`B:687-688`, `L:4419-4420`), and `(jE, rb)` is a bijection onto the unit list; the dynamic
  pool tail is claimed with `atomicAdd` per unit (`B:1574`). No byte of weight is fetched twice.

A cluster (`cudaLaunchKernelEx` + cluster dims) with `cp.async.bulk…multicast` only helps when two CTAs
in the cluster read the *same* address. The only shared reads in the kernel are x/inter/ybuf
(`p.inter`, `p.xA/p.xB`, `p.ybuf`) — an order of magnitude smaller than the weights and already
L2-resident: at verify 32, `act_bytes = 2·bs·HD·2 = 4 MB` vs 523 MB of weights (**0.8%**).
Ceiling if multicast/x-sharing were free: **<1%** of a load ⇒ **≤0.15% of the board**. Also note the
launch path already knows how to do clusters (`L:3589-3614`) and deliberately does not for MoE.

*Cost*: cluster launch + multicast masks; smem unchanged. *Failure mode*: cluster co-scheduling forces
SMs to launch in pairs, which conflicts with the 1-CTA/SM 226 KB smem footprint (a cluster of 2 needs 2
SMs that each host 226 KB — fine — but any odd CTA count or preemption stalls the whole cluster).
**Verdict: dead, 0%.**

One real duplication *class* exists and is worth guarding: an expert with more than `FT_T` = 64 tokens
forms `ceil(tot/64)` jobs (`B:551`, `L:4254`) and **each job re-streams that expert's whole
weight slice**. At the timed shapes the judge's router peaks at ~14 tokens/expert (bs 256, 18.5 touched),
so 1 job — no duplication today. `dist="skewed"` (a correctness-only case) can exceed 64 and pay it
twice. Raising `FT_T` is not available: `WGS_XS = (FT_T/8)·4096` (`B:1827`) is allocated 3×, so 64→128
adds 96 KB of smem that does not exist. Note it, don't chase it.

## 2. Occupancy vs bytes: a 2-CTA/SM build at T ≥ 128 — **0%, and strictly worse**

Budget (sm_90: 232,448 B dynamic/CTA, 233,472 B/SM ⇒ 2 CTAs need **≤113.5 KB** each):

| build | struct `sizeof(Smem)` | weight ring | x ring | total | CTAs/SM |
|---|---|---|---|---|---|
| WGS (main, bs≥64) | ≈22.2 KB | 3×36,864 = 108 KB | 3×32,768 = 96 KB | **≈226 KB** | 1 (2.3 KB spare) |
| skinny (tiny/tiny2/mid) | ≈24.3 KB | 3×36,864 = 108 KB | 0 | ≈132 KB… | 1 |

`B:2802-2811` / `L:6747-6757`: the non-WGS branch returns `base + max(xa, xb)` where
`xb = NSF*UNIT_BYTES + 2*FT_T*XP_FT*2 = 110,592 + 67,584 = 178,176` — the **fat-path x ring**, which the
tiny/tiny2/mid builds can never enter (tiny: ≤4 tokens/expert < `SK_T` 8; mid: `OKC_SPLIT=1` forces
`nfj = 0`, `L:4247-4254`). So the three small-M loads pay **+67.6 KB of dead smem** and request ≈200 KB.

To get 2 CTAs/SM at ≤113.5 KB you must cut ring bytes to ≤91 KB:
* WGS: `NS=1, NX=1` (68 KB) — depth 1, no prefetch at all;
* or half-size stages (128 K ⇒ 18.4 KB weights + 16 KB x) with `NS=2, NX=2` = 91.7 KB.

Either way **bytes-in-flight per SM falls from 204 KB to 136–144 KB**, i.e. from 8× to 5.5× the 25 KB
needed. Since §0 shows we are not MLP-limited (and `OKC_NS=4` measured nothing), the expected gain is
**0**, and the doubled per-stage handshake count (32 stages/unit instead of 16) is a straight loss.
For the skinny builds, dropping the dead `xb` alone (200 → 132 KB) **still does not reach 2 CTAs/SM**;
paired with `OKC_NS=2` it lands at 98 KB / 2 CTAs and lifts skinny MLP from 108 to 144 KB/SM — a
±0–10% shot on d1 only, unverifiable without a check run.
**Verdict: the 1-CTA/SM design is not the problem; the ring is already oversized. 0%.**

## 3. Per-unit epilogue / `S.red` reduction on the critical path

* **WGS build (bs ≥ 64, 4 of 6 loads — note the build bands: `MOE_TINY_MAX_M=4`, `TINY2=8`, `MID=16`,
  `B4=32`, main ≥64 in the layer, so d1/v1/d16 use `nw8+mqmma.sync` and d32/v16/v32 use WGS `wgmma`):
  the epilogue is OFF the critical path**, by three mechanisms:
  1. deferred by commit group: `OKC_WGSFCI` (`B:1859` = 3, `L:5599` = 4) with `wgs_flush` only at
     `L:6057-6059` — i.e. the epilogue of a unit waits until the wgmma of the *next* stage is in flight;
  2. executed by the warpgroup that owns the unit, not by a shared helper, after its `wgmma.commit`;
  3. the partial-sum hand-off is a fire-and-forget `red.release.gpu.global.add` to `p.mfl + MF_CE`
     (`L:5890-5892`, `L:5948-5951`) that the *phase-B producer* consumes for another job
     (`L:5759` `while (ld_acquire(mf_da + j*16+st) < 8) __nanosleep(32)`), never the unit's own producer.
* **Skinny/WS build (d1, v1, d16): it IS on the critical path, but it has already been swept.**
  Per unit the compute warps: `mbar_wait(bars+sl)` → MMA → `mbar_wait(S.wsre+bb)` → write
  `S.red[bb][warp][lane]` (float4) → **`CTA_SYNC()`** (`bar.sync 1` over NT) → thread 0 refills the ring
  slot and `mbar_arrive1(S.wsrf+bb)` (`L:5369-5376`); the helper warp then sums NW float4s and runs the
  epilogue (`L:5470-5520`). Both disambiguation points are *compile-time tuned flags* in the same file:
  `OKC_WSB=1` = “keep the CTA barrier, thread 0 hands over” vs `OKC_WSB=0` = per-warp `mbarrier` arrivals
  on `wse`/`wsrf` with the helper doing the refill (`L:5443-5445`, `L:5457`). The builds ship WSB=1,
  so the no-barrier variant was measured and lost. `OKC_WSBL/WSBLK` likewise give the last 2 units to
  warps 0/1 to keep them off the helper queue (`L:5381`, `L:5483`), and `SYNC0` (removing the barrier)
  was already measured at −1.85% (dead-end list).
  **Verdict: nothing left here in either direction — the skinny handshake is a two-way tuned choice and
  the WGS epilogue is already amortised. Expected gain ≤0.**

## 4. Routing / compaction / cost tables — **already the current design, ~0%**

* **Compaction**: the routing prologue builds exactly a dense, expert-sorted (expert, token) list in
  smem: `S.ent[q]`/`S.pos[i]` (route → compact index) and per-job `jE/jS/jC` + an LPT `forder`
  (`B:520-560`; `L:4232-4300`, `S.eid/ent/pos` at `B:407-409` (= `L:4110-4112`)), using `__match_any_sync`
  per 32 routes. Dynamic job formation vs a prebuilt list changes nothing: the list *is* prebuilt, in
  the kernel, before the first weight copy.
* **Cost table**: it already exists in both lineages — `WGS_CA[8]/WGS_CB[8]` indexed by token count,
  packed into a 64-bit lookup `wgs_pack8` (`B:1848-1851`), and in the layer the cepi linear model
  `OKC_WGSU=160 / OKC_WGSUB=140` (`L:6122-6140`) with prefix sums `wcp/wcpB` (`L:6129-6140`). The
  acceptance criterion of 12.5% cost accuracy applies to a *host-side* table; a host table is
  impossible here because `S.jC[]` (tokens per expert) is only known after the in-kernel routing pass,
  and computing it is O(nj ≤ 88) done by 32 lanes in the x-producer warp **while the consumers gather x**
  (`wgs_ranges` called at `L:5744`, before `WS_JOIN`), i.e. already off the critical path.
* The only prologue cost that scales with M is the `__match_any_sync` pass over M·8 routes (2048
  iterations/32 lanes ≈ 3 µs at bs 256 = 1.4% of that load; ~0 at bs 1).
  **Verdict: 0%.**

---

## Top-3, ranked by expected % of the MoE board

All three are *below* the 7% bar; see the verdict for why nothing in-source clears it.

### 1. 6-bit block scales in the skinny builds (`OKC_SC6S`) — **≈ +1.4%**
* Site: `B:657` `UNIT_SC = 4096` (hard-coded); the layer already parameterised it —
  `L:4363-4369` `OKC_SC6S ? 3072 : 4096` with `SCK`/`SCD` making the decode an opaque IMAD
  (`L:4381-4390`, `L:4479`), and the WGS analogue `OKC_SC6W` **ships and is measured**
  (`L:6913` auto-adds `-DOKC_SC6W=1`; `WGS_PIECE`/`WGS_STG` 8960/35840 at `L:5550-5551`).
* Change: `-DOKC_SC6S=1` for tiny/tiny2/mid + a 6-bit scale repack of the unit-contiguous copies
  `wAu/wBu` in `prepare()` (same transform `harness/make_sc6.py` already does for `wAs/wBs`).
* Mechanism: −1024 B per 36,864 B unit = **−2.78% of all bytes read** on d1, v1, d16 (the three loads
  that still carry 8-bit scales). §0 says time ∝ bytes on these loads (2.20/2.64/2.76 TB/s), so
  −2.78% bytes ≈ −2.7% time on half the loads ⇒ `ln(1.027)·3/6` = **+1.35% board**.
* Cost: none at run time (same unit stride minus 1 KB); prepare-time repack of 3 load bands' scales.
* Failure mode: the SC6 test (`_sc6_ok`) needs every scale byte in 0x40..0x7F — if a weight draw has a
  byte outside that window the flag must fall back per-band, else the output is wrong (silent, tolerance
  is 1.5%). Also `static_assert(OKC_NSF ≤ 7 && OKC_NS ≤ 5)` and the `uslot`/barrier layout assume the
  36 KB stride is a constant (`L:4039`), so the skinny `ring` arithmetic must take `UNIT_BYTES`, which
  it already does.

### 2. Drop the dead fat-path x ring from the skinny builds' smem request — **≈ 0% alone, enables #3**
* Site: `B:2802-2811` / `L:6747-6757` — `xb` (178,176 B) always wins the `max()`, but the fat path is
  unreachable when `OKC_TINY && OKC_SPLIT` or `M ≤ TINY_MAX` (tiny: ≤4 tokens/expert; mid: `SPLIT=1`
  ⇒ `nfj=0`, `L:4244-4252`).
* Change: `#if OKC_TINY` → use `xa` only in `okc_smem_bytes()`. 200 KB → 132 KB.
* Mechanism/cost: pure host-side; frees 68 KB of smem, which by itself still leaves 1 CTA/SM, so the
  measured effect should be **0** (a 132 KB request vs 200 KB changes nothing when 1 CTA/SM either way).
* Failure mode: if a future data path forms a fat job in a `TINY` build, `xbuf` overlaps `S` → silent
  corruption. Needs a `static_assert` that the compiled band cannot produce `njF > 0`.

### 3. Skinny `OKC_NS=2` + #2 ⇒ 2 CTAs/SM at M ≤ 16 — **0 … +1.6% (d1 only), unproven**
* Site: `OKC_NS` (`L:6935` host default = `_MA_MAIN_NS` = 3 for the skinny bands too), `dyn_pool`
  (`B:1574`), `uslot[8]`/`bars` layout (`L:4039`).
* Mechanism: 98 KB/CTA ⇒ 2 resident CTAs/SM; bytes in flight 2×2×36,864 = 144 KB/SM vs 108 KB today.
  Only the loads that are latency-bound rather than bandwidth-bound can gain, and by §0 none is
  MLP-bound — best case d1 1.15× ⇒ **+0.75% board** (`ln 1.15/6`).
* Cost: ring depth 3 → 2 (halves per-CTA prefetch) and `grid` must go to `2·nsm` or the second resident
  CTA never launches (today `grid = state["nsm"]`, `L:14146/16296`). For WGS builds `grid ≤ 159` is a
  hard constraint (`WGS_SLOTS = 160`, `B:1830` “≥ grid + 1”), so this must stay a skinny-band change.
* Failure mode: `dyn_pool(nS, G) = G·OKC_DYNK` (`L:4037`) grows with G and can swallow the whole static
  range at small M (nS = 512 for one expert, 264 CTAs × 4 = 1056 > 512) — the pool would take everything
  and the per-CTA `dync[]` crediting (`L:4108`) becomes the only schedule. Must clamp.

---

## Verdict: no ≥7% structural change exists in-source

**The number that proves it:** every load runs at **2.20–2.81 TB/s of the judge's own 4.8 TB/s floor
(46–59%)**, uniformly from 1.9 to 18.5 touched experts and across both ring/mma designs, while the
weight ring already holds **204 KB/SM against the ~25 KB/SM needed to saturate HBM** (8×), and the
author's measured dead ends are exactly the memory-parallelism family (`OKC_NS=4`, ring depth,
warp-parallel prologue, cluster GEMV, speculative prefetch — all ≈0). A 7% board gain needs a **uniform
6.5% cut on all six loads**; the four angles price at 0 / 0 / 0 / 0, and the three best in-source changes
outside the stage loop sum to **≈ +2%** (bytes −2.8% on half the loads, plus a d1-only occupancy shot).
The missing 45% of peak is spent *per 32–36 KB stage*, i.e. inside the code the ticket excludes
(dequant `dq4`/`SCD` density and wgmma issue, `B:2200-2300`, `L:5967-6080`).

Two facts the main agent should carry forward:
* Reaching the top team's 63%-of-floor needs +20–25% of achieved bandwidth on all six loads at once —
  that is a per-stage cost problem, and it is the same knob the MoE board's top team must have turned.
* d1 is the only load with a *second* deficit class: 2.20 TB/s vs v16's 2.81, i.e. ~13.5 µs of d1's
  24.9 µs is above its own floor share. It is attributable **without a check run** by the in-file
  instrumentation that already exists: `DBG(0..7)` + `p.dbg` (`B:2860` `p.dbg = scratch[3]`, `L:4158`; `DBG(2)/(3)` bracket phases A/B at `B:2605-2613`
  (= `L:6464-6466`), `DBG(4)` the grid barrier at `B:2646`). Reading those eight
  timestamps is the cheapest next measurement on the board; until then no d1-specific scheduling change
  can be ranked above a hope.
