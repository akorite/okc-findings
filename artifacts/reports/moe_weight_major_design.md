# MoE small-M weight-major path — design note (2026-11-03)

Candidate: `src/okc_loop/vendored_work/moe/wm_t1.py` (`# okc: problem=muse_moe_w4a16`), a copy of the
MoE incumbent `artifacts/cas/6402f5f3…b5d2b3e5.py` (2.3863). One new compile-time flag, `OKC_WMAJ`
(default 0 in the CUDA source; the file enables it for `M <= 8` only, `WMAJ = True` is the one-line
kill switch). Nothing was measured on a GPU; the orchestrator spends the judge checks.

## 1. TL;DR

* The incumbent's small-M path **already is** MonoMoE's structure, all four pieces of it (weights on
  the mma **M** axis, the whole token tile on **N**, one persistent 132-CTA launch, per-expert
  readiness flags inside the operator). There is no token-major waste to remove: a "weight-major
  rewrite" has nothing left to re-partition at these shapes.
* What was missing is **residency**: a CTA owns a 9–15-unit (T=1) / 46–83-unit (T=8) slice of the
  routed experts' weight tiles but only holds 3 × 36 KB of it in smem, 2 units of lookahead.
  `OKC_WMAJ` deepens the ring to the smem budget (5 slots) and issues the whole slice at launch.
* Expected gain is **bounded and small**: at T=1 the shapes move 57–113 MB (mean 71 MB) and 22 µs
  implies **3.2 TB/s**, i.e. 91 % of the file author's own 3.5 TB/s bare-ring reference. The whole
  remaining headroom at bs=1 is ≈ 8 %, realistically ≤ 5 %. The brief's "22 µs at 1.0–1.9 TB/s"
  premise assumed 1–2 local experts; the routing distribution actually delivers **2–4** (mean 2.5),
  which is where the 2× difference comes from (§2).

## 2. Partition map of the incumbent at the two shapes (measured from the source, not guessed)

### Units

| quantity | value |
|---|---|
| unit | one 16-row block of an expert's weight rows × full K (4096) |
| unit bytes | 32 768 codes + 4 096 scales = **36 864 B**, contiguous in `w[10]`/`w[11]` (unit order) |
| phase A (`w13`, 8 192 rows) | `RB_A = 512` units/expert = 18.0 MiB |
| phase B (`w2`, 4 096 rows) | `RB_B = 256` units/expert = 9.0 MiB |
| expert | 768 units, 27.0 MiB (= 28.31 MB decimal) |

`wm_t1.py:686` (`UNIT_BYTES`), `:362` (`RB_A/RB_B`), packing `_unit_order()`.

**MonoMoE's "tokens on N" is already here**: the weight word is the mma **A** operand and x is **B**,
so `mma.sync.m16n8k16` computes 16 *weight rows* × 8 *tokens*; the epilogue maps lane `gg` to weight
row `rb*16 + gg` and `gg+8`, lane `tt` to token `2tt`/`2tt+1` (`skinny_ring_ws`, `wm_t1.py:1634-1734`;
`ws_helper`, `:1743`). `SK_T = 8` tokens per job is exactly the mma's N width, so a full job fills N.

### CTAs (grid = `multi_processor_count` = 132 on H200, one CTA per SM, one launch, `WMAJ` does not change it)

Routed local work per shape, from the task's own input distribution (`_gen_input`, top-8 of 256 with the
0.03 noise, then the busiest-group relabel), 40 seeds:

| shape | local routes | local experts (jobs) | tokens/expert |
|---|---|---|---|
| bs = 1 | 2–4 (mean 2.5) | 2–4 | always 1 (`cnt = 1`, max over seeds) |
| bs = 8 | 10–17 (mean 12) | 8–14 (mean 10.2) | 1–3 |

No fat jobs at either shape (`cnt <= SK_T = 8`), so `njS` = number of routed local experts and
`nS = njS × RB_phase`. `run_phase` (`:2400`) then flat-splits that list over the grid:
`v0 = floor(nS·c/G)`, `v1 = floor(nS·(c+1)/G)` (`skinny_split`, `:2413`). Because `RB_B = RB_A/2` and G is
the same, **phase B's slice of a CTA is the same fraction of the expert set as its phase A slice** — w13
and w2 tiles of the same expert already land on the same CTA; there is no second partition to fix.

| shape | total bytes | units/CTA phase A | units/CTA phase B | units/CTA total | bytes/CTA | CTAs with work |
|---|---|---|---|---|---|---|
| T=1, 2 experts (min) | 56.6 MB | 7–8 | 3–4 | 10–12 | 360–432 KB | 132/132 |
| T=1, 2.5 (mean) | 70.8 MB | 9–10 | 4–5 | 13–15 | 468–540 KB | 132/132 |
| T=1, 4 experts (max) | 113.2 MB | 15–16 | 7–8 | 22–24 | 792–864 KB | 132/132 |
| T=8, 8 experts (min) | 226.5 MB | 31–32 | 15–16 | 46–48 | 1.62–1.69 MB | 132/132 |
| T=8, 10.2 (mean) | 288.8 MB | 39–40 | 19–20 | 58–60 | 2.04–2.11 MB | 132/132 |
| T=8, 14 experts (max) | 396.4 MB | 54–55 | 27–28 | 81–83 | 2.85–2.92 MB | 132/132 |

All 132 CTAs are occupied at both shapes (the smallest possible work set, one local expert = 512 units,
is still 3.9 × the grid), so no CTA idles and there is no tail CTA to reclaim.

### Per-unit cost model (why the bytes are what matters)

Per 36 864 B unit: 256 `mma.m16n8k16` (8 warps × 32) = 524 288 MACs = **128 cycles** of tensor work on
an H200 SM (4 096 MAC/cycle) ≈ 0.08 µs, versus ≈ 1.1 µs of fair-share DRAM time at 4.2 TB/s / 132.
Compute is ~7 % of a unit; the rest is the fetch, one `mbarrier` wait, one `CTA_SYNC`, and the helper
warp's 8-warp partial-sum reduce + epilogue. That ratio is why the small-M path is a memory problem,
and why re-partitioning cannot buy much: at `cnt = 1` the mma wastes 7 of 8 N columns, but replacing
that waste only saves ≤ 0.07 µs/unit.

## 3. What the incumbent already has, versus MonoMoE (arXiv 2609.04244)

| MonoMoE piece | here |
|---|---|
| CTAs partitioned over expert-weight tiles | yes: one 16-row tile of a routed expert per unit, flat split over 132 CTAs (`skinny_split`) |
| whole token tile on the tensor-core N | yes: weights = A (M = 16 rows), tokens = B (N = 8) |
| one persistent launch, readiness sentinels inside the operator | yes: `doneA[j]` per-expert counters, `mbarrier` slot ring |
| weight tile residency / memory-level parallelism | **no** — 3 smem slots (108 KB of a 227 KB budget) |
| readiness flags instead of a CTA-wide barrier per unit | the incumbent keeps one `CTA_SYNC` per unit; SYNC0 already tested its removal (**falsified, 2.342 vs 2.3863**) |

## 4. What `OKC_WMAJ` changes (2 hunks, both guarded, no math)

1. **Ring depth 3 → 5** (`#if OKC_WMAJ && OKC_TINY`, `wm_t1.py:312`). The smem budget was the only
   reason for 3: `sizeof(Smem)` = 36 576 B (measured, §6) + 5 × 36 864 = **220 928 B** of the
   H200's 232 448 B. Margin 11 520 B. A `static_assert` on both smem layouts (ring and fat x stages,
   `:690`) makes an over-budget configuration a build failure, not a silent launch failure — the
   launcher discards `cudaFuncSetAttribute`'s error code.
2. **the launch prologue issues the whole slice** instead of `OKC_FAST1 = 2` units: `n = min(WM_PRE,
   v1 - v0)` (`:2532`) with `WM_PRE = OKC_NS` under the flag and `OKC_FAST1` otherwise. The FAST1 warp
   already knows the CTA's flat range from the touched-expert mask before routing runs, so this is the
   same loop with a bigger bound; the helper's top-up loop then starts empty.

Effect: outstanding lookahead per CTA goes from 2 units (72 KB) to 4 (144 KB); grid-wide in flight
9.7 MB → 19.5 MB against the ~3.4 MB needed to cover an 0.8 µs HBM latency at 4.2 TB/s. No occupancy
cost: 145 KB (the incumbent's footprint) already pins the CTA to 1/SM, so 221 KB changes nothing.

Nothing else changes: same TMA `cp.async.bulk` ring, same `dq4`/`SCDK` E2M1+scale decode
(PRMT/IMAD path when `prepare()` proves the scales are normal e4m3, LUT otherwise — untouched), same
mma, same `S.red` cross-warp reduce, same epilogue stores, same `doneA` hand-off, same grid barrier,
same final pass, same packing (`prepare()` byte-for-byte identical).

### Why this is not SYNC0

SYNC0 replaced the per-unit `CTA_SYNC` with `wse`/`wsre` mbarrier hand-shakes (a *synchronisation*
change) and lost 1.85 %. `OKC_WMAJ` keeps every barrier exactly where the incumbent has it and changes
only *how many slots exist and when they are filled* — a memory-level-parallelism change. If both were
enabled they would compose, not overlap.

## 5. Correctness argument

* **Six shapes.** `kernel_fn` picks `TINY_SEL` for `M ≤ 4`, `TINY2_SEL` for `5 ≤ M ≤ 8`, and
  `MID_DEFINES` / `_main_defines()` / `MAIN_FALLBACK_DEFINES` untouched for 9..16 / ≥17. The flag's
  define is only in the two small-M tuples, and the source-side hunk is additionally gated on
  `OKC_TINY`, so MID (which compiles the `helper0` source, no `OKC_WMAJ` text at all), MAIN and the
  fallback compile the incumbent's code. `WMAJ = False` restores the incumbent's tuples exactly.
* **Partition unchanged** → every unit is still issued exactly once, to the same CTA, in the same
  order; only the slot count and the issue point move. The ring invariant `slot = q % NS`,
  `parity = (q/NS) & 1` is NS-generic in both the producer (`ws_issue`, `:1737`) and the consumer
  (`skinny_ring_ws`, `:1635`); `mbar_init(wse + k)` and `mbar_init(bars + k)` already loop to
  `OKC_NS`; `S.uslot[8]`, `S.wse[8]`, `S.bars[12]` (0–4 skinny, 5–11 fat, `static_assert(OKC_NS ≤ 5)`)
  all cover NS = 5; `SKDYN = 0` in the tiny builds so there is no run-time pool to re-derive.
* **No double issue**: `S.preissued` is written by the FAST1 warp before the helper's `WS_JOIN` and
  the helper issues from `S.preissued` to `OKC_NS`, so 5 pre-issued units means an empty top-up.
* **Bounded**: `n = min(WM_PRE, v1 - v0)` never runs past the CTA's slice, and the expert lookup
  `__fns(m, 0, (v >> RB_A_LOG) + 1)` is valid for any `v < popc(m) × RB_A` (it already ran with
  `n = 2` and is used identically for larger `n`).
* **No new accumulation order**: the mma/partial-sum/epilogue path is byte-identical, so the numerics
  are those of the incumbent.
* **Precondition, stated**: the flag assumes a 227 KB-smem device (H200). `_ext`'s small-smem fallback
  cannot strip a source-side `#undef/#define`, so `OKC_WMAJ` must not be turned on for a dev GPU —
  hence `WMAJ` lives next to the define tuples, not inside `_ext`.

## 6. Verification performed here (no GPU, no `okc`)

* `okc_loop.cli lint` → `PASS` (1/1 clean) on `wm_t1.py`.
* `tools/cu_build.py` (nvrtc → ptxas, sm_90a) on all five `CUDA_SRC` variants, with and without
  `-DOKC_WMAJ=1`: **ok** every time.
* Direct nvrtc compiles of TINY, TINY2, MID, MAIN with the flag on and off: no diagnostics;
  `static_assert(OKC_NS == 5)` holds only under `OKC_WMAJ=1`, so the flag is the sole control.
* `sizeof(okc::Smem)` bisected through a live `static_assert` = **36 576 B** → WMAJ total
  **220 928 B ≤ 232 448 B** (margin 11 520 B); both new asserts hold.
* Default path inert by construction: every hunk is inside `#if OKC_WMAJ` (asserts) or uses
  `WM_PRE`/`OKC_NS`, which reduce to the incumbent's values (`OKC_FAST1`, `-DOKC_NS=3`) when the flag
  is 0.
* Not run: `okc check` / `okc submit`, judge, GPU.

## 7. Expected gain, and the piece that blocks a bigger one

Benchmark: 22 µs standalone at bs=1 (the file author's own number). With the real routing fan-out that
is 70.8 MB → **3.22 TB/s**, 91 % of the author's 3.5 TB/s ring reference for the same kind of read.
So:

* T=1 ceiling from better residency: 70.8 MB / 3.5 TB/s = 20.2 µs ⇒ **≤ +8 %**, realistic **0–5 %**.
* T=8: 289 MB/CTA-summed, 58–60 units per CTA, and the ring is 97 % of the bytes' fair-share time —
  the same order of headroom, pursued for the same reason (more of the slice resident).

The blocker for a genuinely different structure at these shapes is **smem, not the partition**:

* A warpgmma (MonoMoE-proper) consumer needs 64-row A tiles, i.e. a 64-row × full-K w13 tile =
  128 KB, and x as an smem **B** operand (8 tokens × 4096 K = 64 KB) — together 192 KB before any
  double buffering, against a 227 KB budget that must also hold `Smem` (36 KB) and the fat x stages.
  The existing WGS (RS-wgmma) build cannot be reused as-is either: its unit is 256 rows, i.e. 32 units
  per expert for phase A, fewer than the 132 CTAs — at T=1 most of the grid would be idle and the rest
  stream-K, which is why the WGS build only starts at M ≥ 17.
* Shipping that blind (new `_wgs_copy`-style 64-row natural-k packing + a new wgmma consumer + x
  staging) without a GPU or a judge check is a silent-numeric-corruption risk for a ≤ 8 % prize, so it
  was not attempted; it would also add a second 27 MiB weight copy per expert.

The other axes in this lineage are recorded as already spent: CPF/CPFF (cross-phase prefill),
WS/WSB/WSBL/WSBLK (helper warp + tail), FINPF, WPAR, CEPI/FCI4, YBFL, SC6, SYNC0. `OKC_WMAJ` is the
one that was not: residency of the CTA's own weight-tile slice.
