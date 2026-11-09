# klr in-layer MoE: the unit-end fence/release path (WGS consumer)

Scope: `src/okc_loop/vendored/keell0renz/muse_decode_layer.py` (`MOE_CUDA_SRC`, sha 82c098ae) lines
5560-6120, the WGS consumer / epilogue / finalizer. Compared line-for-line against the published
alygg77 2.4015 kernel (`kernels/muse_moe_w4a16.py`, sha b3307089…, fetched, 3601 lines) — the two
files are the same kernel family and `wgs_flush`/`wgs_seg`/`wgs_phase` are **byte-identical** except
for the MFIN block (see §3).

Shape constants that turn counts into per-CTA numbers: `HD = ID = 4096`, `NW = 16`, `NT = 512`,
4 consumer warpgroups (`wg = warp >> 2`), `UNIT = 16 weight rows x 4096 K = 36,864 B`,
`RB_A = 512` blocks (phase A = 32 units/expert), `RB_B = 256` blocks (phase B = 16 units/expert) ⇒
**48 units/expert**. Grid ≈ 132 CTAs (`WGS_SLOTS 160 ≥ grid + 1`). d32 = 12 touched experts ⇒ 576
units ⇒ **4.4 units/CTA (2.9 A + 1.45 B)**; v16/v32 = 17.2/18.5 experts ⇒ 6.4-6.7 units/CTA.

## 1. Enumeration — every fence / membar / bar.sync / release on the unit-end path

| # | line | instruction | scope | executes per unit, per CTA | per warpgroup |
|---|------|-------------|-------|---------------------------|---------------|
| 1 | 6040 `wg_fence()` | `wgmma.fence.sync.aligned` | warpgroup; **register** order, no memory order | 512 (16 stages x 8 commit groups x 4 wg) | 128 |
| 2 | 6049/6050 | `wgmma.commit_group` + `wgmma.wait_group 1` | warpgroup | 512 | 128 |
| 3 | 6018 | `mbar_arrive1(BW_E + ws)` (weight slot free) | CTA smem | 64 | 16 |
| 4 | 6053/6069 | `mbar_arrive1(BX_E + prev)` (x slot free) | CTA smem | 64 | 16 |
| 5 | 6068 | `wgmma.wait_group 0` (drain at unit end) | warpgroup | 4 | 1 |
| 6 | 5889 | `bar.sync (3+wg), 128` (order this wg's 4 warps before its release) | CTA named barrier | 4 (one per wg) | 1 |
| 7 | **5900** | **`red.release.gpu.global.add.s32` — the doneA release** (`mf_da + j*16 + (ub>>1)`) | GPU | **4** (one per wg) | 1 |
| 8 | **5892** | **`red.release.gpu.global.add.s32` — the MFIN phase-B expert counter** (`MF_CE + gen*32 + jE[j]`), kind 4 | GPU | **4**, phase B only (**ours only**) | 1 |
| 9 | 6075 | `red.global.add.v4.f32` x NTT (split partial sums) | GPU, relaxed | split units only (≤2/CTA/phase) | NTT |
| 10 | 5916/5921 | `bar.sync` + `atom.acq_rel.gpu.global.add.s32` (non-DF "am I last") | CTA / GPU | split unit, phase A (DF bit0=0) | 1+1 |
| 11 | 5912 | `ld.acquire.gpu` spin on the fixup counter | GPU | split unit, phase B (DF bit1=1) | 1 |
| 12 | **5929** | **`fence.acq_rel.gpu`** — executed by **all 128 threads** of the warpgroup, before the fixup read-back | GPU | split finisher, ≤2-3/CTA/replay | 1 |
| 13a | **5951** | `red.release.gpu.global.add.s32` after the split finisher's epilogue, `mf_da`-style expert counter (`MF_CE + gen*32 + jE[j]`, `OKC_MFP & 2`) | GPU | split finisher only, phase B — **ours only** (their 2151 jumps straight to the read-back) | 1 |
| 13b | 5956 | `red.release.gpu.global.add.s32` after the split finisher's epilogue (`!PB`) — the split-unit doneA release | GPU | split finisher only, phase A (both files: their 2172) | 1 |
| 14 | 5814, 6309/6324/6334/6380, 5731 | gather release, mbarrier-init fences, routing barrier | — | once per launch / per job | — |

`wgs_wait` (5982/5984) is `mbarrier.try_wait.parity` — a wait, not a fence. `st_keep_*` (4000-4004) is
a plain `st.global.L2::cache_hint` with an evict-last hint — no fence per store.

**The number that matters.** Per unit per CTA: **8** ordering ops in phase A (rows 6+7: 4 `bar.sync`
+ 4 `red.release.gpu`) and **8** more in phase B (rows 6+8) = **16 per unit**, i.e. **35 per CTA per
replay at d32** (4.4 units) and 51-54 at v16/v32. alygg77: **8 per unit, phase A only = 23 per CTA**
(their 2118/2122; their phase B sets no `pd`, so it executes **zero** — verified: the published file
has no `OKC_MFIN`, no `mfl`, no per-expert counter). Ours is therefore **1.5x their count per replay
and 2.0x per unit**, and the whole excess is row 8.

Everything else is equal: the 8 `wgmma.fence` per stage (both use `OKC_WGSGS 2`), the 32 mbarrier
arrivals, the `fence.acq_rel.gpu` on the split read-back, the `atom.acq_rel`, the `red.add.v4` — all
present, same frequency, same lines. Nothing on the unit-end path is executed *more often* than in
alygg77's kernel except row 8 (the phase-B whole-unit counter, +8 ops/unit, and it is the only item
with a real count) and row 13a (+1 per split-unit finisher, ≤1 per CTA per replay; their 2151 goes
straight from the `fence.acq_rel.gpu` to the read-back).

## 2. What is hoistable, and the PTX rule it needs

* **Row 7/8 four releases → one per CTA (done; `OKC_WGSFR1`, default off).** The four warpgroups
  cannot be merged for free: each release only orders *its own thread's* prior stores, so a single
  releaser needs a CTA-scope rendezvous that orders the other warpgroups' epilogue stores first.
  Rule: `bar.arrive`/`bar.sync` with the same count form the CTA-scope synchronizes-with edge, and
  `bar.sync` is a `fence.sc` at `.cta` for the arriving threads (PTX ISA `bar`); causality order then
  composes that edge with wg 0's `red.release.gpu` (the memory model's scope axiom — `.cta` is a
  subset of `.gpu`), so the one release publishes all 512 threads' stores. This is the exact argument
  the file already relies on for the split read-back (§4) and the same `bar.arrive`+`bar.sync` idiom
  it already uses at 5731 / 2543 (`bar.sync 7, NT+32` vs `bar.arrive 7, NT+32`).
  The release itself cannot be *dropped*: the consumer side is `ld.acquire` (5756/5759), and a
  relaxed write plus an acquire read does not order anything — release on the writer is required.
  Cost of the rendezvous ≈ 0: the four warpgroups already march in lock-step over the same weight/x
  rings (same slot, same parity, ring depth 3), so their drift is already bounded by a couple of
  stages; the three non-leaders only `bar.arrive` and continue.

* **NOT hoistable — row 7 to once per phase.** The counter is per (job, phase-B stage): the phase-B x
  producer waits per stage (5759 `while (ld_acquire(mf_da + j*16 + st) < 8)`), which is the whole
  point of MFIN (`OKC_MFQ & 4` = no grid barrier): phase B starts on stage 0 while other CTAs are
  still in phase A. Coarsening it to per-job = serialising A→B.
* **NOT hoistable — row 8 to once per CTA per phase.** The consumer (6529/6587) waits per token per
  *expert* for `>= 64 * njobs` = 16 units x 4 wg. Per CTA that is 1.45 units per expert at d32 and
  2.24 at v32, so per-expert and per-unit are the same event; a single grid counter would instead
  make the final pass wait for the slowest CTA's phase-B end, i.e. trade the count for the whole
  phase-B skew. The only real reduction is the 4→1 merge of §2.
* **NOT removable — row 1.** `wgmma.fence` is required after each write of the A registers and before
  the consuming wgmma (PTX `wgmma` rules); 8 per stage = one per commit group is the minimum at
  `GS 2`. `GS 4` halves it but trips C7512 (ptxas serialises every wgmma at 96 registers) — that is
  why the knob is 2, and alygg77's is 2 as well.
* **Removable but already refuted — row 12.** `fence.acq_rel.gpu` by all 128 threads is redundant
  after `atom.acq_rel` (5921) or `ld_acquire` (5912) + the `bar.sync` at 5924, by the same causality
  rule; `__ldcg` bypasses L1 so no stale read. alygg77 measured exactly this deletion: **+0.234% on
  the MoE board, no row beat A/A** — i.e. noise. Not worth a check.

## 3. The one real difference from alygg77, and why it is a deliberate trade

Row 8 exists because klr replaced the MoE's grid barrier with per-expert completion counters. Their
file still pays the grid barrier (`<our 6495>: __threadfence + atomicAdd + spin`, once per CTA) and
no per-unit release; ours pays 4 releases + 4 barriers per phase-B unit and no grid barrier. At d32
that is +11.6 ordering ops per CTA (of which ~+2 are exposed: the four releases of one unit overlap
across warpgroups, so the per-unit exposure is ~one drain, not four) against the grid barrier's
A→B serialisation. `OKC_WGSFR1` cuts the excess from +11.6 to +2.9 ops/CTA without giving up the
overlap. If the board still wants the other direction, the lever is `-DOKC_MFIN=0` (both code paths
exist in-source) — but that is a knob A/B, not a fence change, and `moe_klrbands.md` already flags it
as needing a GPU A/B.

## 4. Implemented change

`src/okc_loop/vendored_work/layer/klr_fence.py` (copy of the vendored board file, sha 82c098ae; the
only edits are the three hunks below, 33 lines added, no other line touched):

* new `OKC_WGSFR1` (default **0** = the board behaviour byte-for-byte);
* `wgs_rel4(c, tid, wg)`: wg 0 does `bar.sync 8, NT` then one thread does `red.release.gpu … +4`;
  wg 1-3 do `bar.arrive 8, NT` and continue. Barrier 8 is the only free id (0 `__syncthreads`,
  1 `CTA_SYNC`, 2 `WS_JOIN`, 3-6 the four warpgroups, 7 routing); 9 of 16 used.
* `wgs_flush`: the wg-local `bar.sync (3+wg),128` is skipped for kinds 1/4 (the CTA barrier covers
  it); kind 4 and kind 1 call `wgs_rel4` instead of four per-warpgroup `red.release … +1`. The `+4`
  keeps every consumer target unchanged (8 = 2 units x 4 wg; 64 = 16 units x 4 wg).

Result at d32 per CTA per replay: **35 → 8.7** ordering ops (4.0x), **2.6x fewer than alygg77's 23**;
fences per unit per CTA **16 → 4** (1 `bar.sync` + 1 `red.release` per unit, the three `bar.arrive`
do not wait).

### Gates (offline)

* `okc_loop.cli lint`: **delta 0** — identical output to the vendored file modulo line numbers
  (same 30 pre-existing tier0 errors, all in the harness/python surface, none in this region).
* `tools/cu_build.py --variant MID` and `--variant MAIN`: **result set identical to the vendored
  baseline** (diff of ok/FAIL lines modulo filename is empty); the 6 `MOE_CUDA_SRC` builds (incl. the
  WGS band `OKC_NW=16 OKC_WGS=1`) and the MID band all pass, with the flag off **and** with
  `-D OKC_WGSFR1=1`. `bfp_*`/`_SL_GLUE`/`_ROUTER_FP32` failures are pre-existing on the baseline
  (missing `nv/target`, torch host symbols).
* ptxas accepts the exact new instructions: `barrier.sync 8, 512`, `barrier.arrive 8, 512`,
  `red.release.gpu.global.add.s32 [..], 4` on `sm_90a`, 9 barriers used (limit 16).
* Note: cu_build compiles this kernel to a *template body* (empty PTX, see its docstring), so it is a
  syntax/semantic gate only — it cannot report registers, spills or verify the barrier rendezvous.

### Risk (why it ships off)

The change adds a CTA-wide rendezvous where the board has none, so its correctness rests on `pd.kind`
being CTA-uniform at each flush: true by construction (`slot`, `s0/s1`, `PB` and the `OKC_WGSDFIN`
bits are all CTA-uniform, `pd.kind` is assigned only from them, and the flush call sites are
`ci == OKC_WGSFCI` of the same stage loop). A non-uniform `pd.kind` would hang the leader warpgroup
instead of mis-computing. Needs one GPU A/B (a check or the MoE harness) before it can go on: it
buys ≤1.5% of the MoE board on the count alone, which is under the ~6% per-shape judge noise, so it
is only shippable inside a cluster of wins.

## 5. Verdict on the frame of the ticket

"Unit-end stall" is a **latency**, not a count, problem. alygg77's own breakdown (doneA release
0.9 us + epilogue 0.4-0.6 + gap 0.3 per warpgroup per unit) is ~one drain per unit, and four
warpgroups' drains overlap: the exposure per unit is ~1.8 us, so the whole 1-7 us band at d32
(4.4 units/CTA ⇒ 10.5-21.7 us) is that latency times the units/CTA. Our count is only 1.5x theirs,
so even deleting every release we have would buy ~1% of the MoE — **below the noise floor**. The
0.9 us item cannot be removed by counting: it is the release-atomic's drain and PTX requires a
release on the writer. The remaining structural lever is the *depth* per unit (the wgmma drain at
6068 plus the deferred epilogue), not the fence inventory.
