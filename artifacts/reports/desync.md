# Per-CTA unit rotation of the WGS stage stream (`OKC_WGSROT`)

Board file: `src/okc_loop/vendored_work/moe/moe_desync.py` (copy of `muse_moe_w4a16.py`, md5
`1fac0cbd…` + the diff below). Flag `OKC_WGSROT`, **default 0**; `-D OKC_WGSROT=1` to measure.
Gate: `artifacts/reports/desync_gate.py`.

## 1. The invariants a rotation must preserve (klr:5690-6190, standalone 1975-2360)

The stage order is defined **once**, by `wgs_split(pb,g0,g1)` returning the seam `sp`, and read by three
places that must agree exactly: `wgs_wproducer` (weight ring), `wgs_xproducer` (x ring) and `wgs_phase`
(consumer, which also re-derives `sp` from `S.wgr` at its seam). `seq`/`n` are the global ring cursor and are
continuous across the two phases.

1. **Every stage of `[g0,g1)` issued exactly once, by both producers and the consumer** — the walk is
   `[sp,g1)` then `[g0,sp)`; the ring is a 3-slot FIFO with `seq % WGS_NS` on both ends, so a different
   *order* is legal only if producer and consumer use the identical permutation.
2. **A unit's 16 stages stay contiguous and ascending.** `wgs_seg` accumulates one unit into one register
   `d[WGS_NT][4]` with 16 chained `wgmma.m64nNk16` (K = 256 each); the FP32 accumulation order over K *is*
   the stage order. Reordering stages inside a unit changes rounding (that is why the "rotate the 16 stages
   by `cta%16`" variant is not free).
3. **Segments never straddle a unit.** `wgs_seg` takes `(u,s0,s1)`; the seam must fall on a unit boundary or
   at `g0`, else `s1 = min(16, s0+(ge-gs))` silently truncates a unit and `s0 != 0, s1 != 16` makes a bogus
   fix-up slot.
4. **Split-unit ownership is a function of the ranges, not of the order.** `slot = (s1-s0==16) ? -1 :
   (s0==0 ? blockIdx.x : wgs_cta_of(u<<4))`, and the DFIN finisher is `PB ? s0==0 : s1==16`. Both read
   `s0/s1`, i.e. the cost-weighted ranges, which the rotation does not touch. Only *when* a contributor runs
   changes, and nothing blocks cross-CTA, so no deadlock — but a finisher can wait longer.
5. **Retire-then-release per stage.** `BW_E` is arrived at with the stage's last code word in registers, `BX_E`
   one commit group after the stage's last `wgmma` retired, and the deferred `WgsPend` (doneA release, fix-up
   count, expert counter) is flushed exactly once at commit group `OKC_WGSFCI` of the next segment or at
   phase end. All of these are per-*segment* and carry their identity in `pd`, so they are order-independent;
   the fix-up buffer slot is per-CTA and per-phase, never reused within a phase.
6. Phase B (`pb == 1`) additionally waits per stage on `mf_da[j*16+st]` (phase-A units of the same job), so
   the rotation is applied to **phase A only** — the same scope `OKC_WGSREORD` already uses.

## 2. Is the rotation arithmetically free? Yes, proved, not sampled

`float d[WGS_NT][4]` is declared **inside** `wgs_seg` and its first use is `WGSMMA<NTT>::mma0`, i.e. the
`wgmma` accumulate=0 form: every segment zeroes its own accumulator and accumulates only its own stages. A
*unit-major* permutation therefore leaves

- each unit's K order (ascending stages, contiguous — invariant 2), and
- every cross-unit relation (there is none: units are independent outputs; the only cross-unit state is the
  **order-independent** `red.global.add` fix-up of a split unit and the atomics that count it),

untouched. **No per-element accumulation order changes anywhere in the kernel.** The only difference is the
order in which the independent per-unit segments are issued.

## 3. Implementation: one function, no restructuring

The existing walk is *already* a one-seam cyclic rotation with `sp = ue` (the trailing unit's start,
`OKC_WGSREORD`). So the rotation is a change of `sp` only — no new loop, no second seam, and all three call
sites inherit it:

```c
#if OKC_WGSROT
  const int b0 = ((g0 + 15) >> 4) << 4;   // smallest unit boundary in [g0, g1)
  if (pb == 0 && ue > g0) return b0 + ((int)((unsigned)blockIdx.x % (unsigned)(((ue - b0) >> 4) + 1)) << 4);
  return g0;                              // a range inside one unit has no boundary to rotate to
#else
  … existing ue/g0 …
#endif
```

Sequence becomes `[sp,g1)` then `[g0,sp)`: a cyclic rotation of `[g0,g1)` whose seam is a unit boundary, so
invariants 1-3 hold by construction and per-CTA `sp` never leaves the CTA. Cost: one 32-bit `%` per phase per
call site (producer, x producer, consumer seam) — off the critical path; the consumer's hot stage loop is
byte-identical.

Gate `artifacts/reports/desync_gate.py <board> [--rot0]`: rebuilds `wgs_prefix`/`wgs_stage_at`/`wgs_range`/
`wgs_split`/the walk from the file's own `OKC_WGSA`/`OKC_WGSB` and asserts, for 60 synthetic routings ×
G ∈ {1,2,7,32,96,132}: exact permutation, per-unit contiguity + ascending, unit-aligned seam.
`ok rotations=360` both modes.

## 4. Price: ~1 µs at d32 — the mechanism is real but the peak does not move

The rotation is a **permutation of the per-CTA end *times*, not a change of their multiset** whenever the
units it permutes cost the same: with equal-cost units `{a,16,16,b}` the end timeline is the same whatever the
order. It only desyncs where a CTA's range straddles a job boundary with a different `ntt`.

Counted on the file's own cost tables (stage = 1.12 µs at the d32 phase-A ceiling, 36 KB):

| d32 routing | units/CTA | candidate starts/CTA | distinct 1-µs end bins, base → rot | mean CTAs per occupied bin | **peak bin** |
|---|---|---|---|---|---|
| 8 jobs, `jC=[32,24,19,12,8,5,3,1]` | 2.0-3.0 | 1/2/3 = 20/100/12 | 63 → **89** | 6.0 → **4.25** (−29 %) | 61 → 62 |
| 8 × 32 tokens (uniform cost) | 2.0 | 2 (124 CTAs) | 18 → **32** | 20.9 → **11.75** (−44 %) | 128 → 128 |
| 12 jobs | 3.0-4.0 | 2/3/4 = 15/98/15 | 51 → **80** | 9.9 → **6.33** (−36 %) | 98 → 98 |

So the bulk coincidence drops ~30-45 %, the **peak is unchanged**: the worst spikes are the *first* end (the
trailing partial unit, 0.13-2.8 µs, which is where `ue` already points) and the *phase end*, where the
cost-equalized ranges make every CTA finish together. Whole-unit rotation cannot touch either — the head spike
would need the partial unit moved *off* the front (undoing `OKC_WGSREORD`, which exists for the opposite
reason) and the tail is the phase boundary, not a unit end.

Priced off the phase-A-wave component of the d32 loss budget (3-6 µs of the 23): −30-45 % of it = **0.9-1.7 µs,
point estimate ~1.2 µs**, i.e. 0.65 % of the 186 µs row ⇒ **~0.1 % of the score** (one of six loads in the
geomean). alygg77's independent "dispersing unit ends" measured **~1 µs at d32**, below their own A/A band
(0.7-3.1 µs), on the same mechanism class; the coupon on this lane is not larger than theirs.

**Recommendation: do not spend a check on `OKC_WGSROT` alone.** 0.1 % of the score with a peak invariant is
inside judge noise, and the file's own history (`NEGATIVE_RESULTS` §2, `kernarc_digest`:216) says everything
that shortens one end without lengthening the tail has already lost twice. It is only worth riding along with
a change that already moves the row.

## Gates

- `okc_loop.cli lint src/okc_loop/vendored_work/moe/moe_desync.py` → `PASS 1/1 clean`.
- `tools/cu_build.py … --variant MAIN` → `ok` on both `CUDA_SRC` and both `BW_SRC`, with `OKC_WGSROT=0` (default)
  and with `-D OKC_WGSROT=1`.
- `artifacts/reports/desync_gate.py` → invariants 1-3 hold for both modes.

Skipped: no GPU and no `nvidia-smi` on this workstation, so the µs figure is the coincidence model above
cross-checked against alygg77's measurement, not a run. Not implemented: the 3-segment "partial unit first,
rest rotated" variant (needs the consumer's seam state machine rewritten for an unmeasurable ~1 µs).
