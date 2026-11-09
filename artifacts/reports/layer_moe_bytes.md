# MoE byte levers — 6-bit skinny scales (`OKC_SC6S`) and L2-resident x refill (`OKC_XRES`)

File: `src/okc_loop/vendored_work/layer/moe_bytes.py` (copy of `klr_fci5.py`; the base is untouched).
Lint multiset identical to the base (30 errors, findings shift only); `tools/cu_build.py` status identical to the
base for every source and both flags compile (`-DOKC_SC6S=1` on the mid/tiny tuples, `-DOKC_XRES=1` on the WGS
tuple). Everything below is either cited from `muse_arch_levers.md` / the profile it cites, or marked
`[DERIVED]` where I recomputed it from those numbers.

Both levers sit behind their own define, each `#ifndef`-defaulting to **0**, and each behind a Python switch
(`SC6_SKINNY`, `XRES_L2`) that `_ma_ext()` turns into the define per build. With both switches `False` the file
compiles and packs exactly as `klr_fci5.py` did. Each lever has an `prepare()`-time probe that compares the enabled
and disabled builds and reverts the lever unless the difference stays inside those same builds' run-to-run floor
(`0` where the epilogue is deterministic); each prints its verdict (`[sc6s] …`, `[xres] …`).

---

## Lever A — `OKC_SC6S`: 6-bit NVFP4 block scales on the three skinny MoE builds — **+0.68% layer**

### What the bytes are

A skinny unit is one 16-row block × the whole K, contiguous, one `cp.async.bulk`:

| | codes | scales | unit |
|---|---|---|---|
| shipped (`UNIT_SC = 4096`) | 32,768 B | 4,096 B | **36,864 B** |
| `OKC_SC6S=1` (`UNIT_SC = 3072`) | 32,768 B | **3,072 B** | **34,816 B** |

`-DOKC_SC6S=1` therefore removes **1,024 B of every 36,864 B = 2.78% of every byte the three skinny loads read**.
The three WGS loads (d32, v16, v32) already ship this via `OKC_SC6W`; d1, v1 and d16 were the only loads still
carrying the 4,096-byte payload.

### Byte account per load `[DERIVED from muse_arch_levers.md §2.1 + moe_struct rate table]`

MoE bytes are the profile's own per-load counts; saved = 2.78% of them; µs = saved ÷ the rate that load's MoE
ran at (2.20/1.86/2.27/2.32/2.55/2.48 TB/s); board % = µs × that shape's sensitivity.

| load | MoE fetched | saved | µs | board %/µs | board % |
|---|---|---|---|---|---|
| D1S | 53.1 MB | 1.48 MB | 0.67 | 0.1799 | 0.12 |
| D1F | 54.6 MB | 1.52 MB | 0.82 | 0.0626 | 0.05 |
| V1S | 160.0 MB | 4.45 MB | 1.96 | 0.1137 | 0.22 |
| V1F | 161.5 MB | 4.49 MB | 1.94 | 0.0395 | 0.08 |
| D16S | 161.8 MB | 4.50 MB | 1.76 | 0.0883 | 0.16 |
| D16F | 158.0 MB | 4.39 MB | 1.77 | 0.0307 | 0.05 |
| **total** | 749.0 MB | **17.8 MB** | 8.9 | | **+0.68** |

Touched is unchanged: the codes are untouched and the scales are the same 6-bit values in a different container, so
this is the only lever in the layer that goes *below* the judge's `0.5 + 1/16` B/element floor model (4.5 → 4.375).

### Guard logic (all four must hold; any failure = the 4,096-byte layout everywhere)

1. `_MA_SFX` non-empty — the PRMT scale decode is in use. The LUT decode (`#if !OKC_PRMTSC`, `:4590-4592`)
   indexes the raw e4m3 bytes and has no 6-bit form; `_sc6_ok` implies the PRMT window, the check is kept explicit.
2. `_sc6_ok(w13_sf, w2_sf)` (`:7091`) — every scale byte is in `0x40..0x7F`, so `byte - 0x40` is a lossless 6-bit
   index. A byte outside that window decodes a block off by ≥2× (wrong output, not a small error), which is why
   this is a hard gate rather than a tolerance.
3. The build is a **skinny** one: `"-DOKC_TINY=1" in defines and "-DOKC_NW=16" not in defines` (`:6935-6940`).
   That is exactly TINY / TINY2 / MID, and it is the exact predicate because those are the only builds that
   - read the unit-order copies `p.wAu / p.wBu` (`ring_issue`, `:4420`) whose stride is `UNIT_BYTES`;
   - never take the fat path, which also indexes with `UNIT_BYTES` (`fat_issue`, `:4860`) but whose weight copies
     are `_fat_order`-shaped 36,864-byte stages: TINY/TINY2 have `nfj = 0` because `tot <= SK_T = 8` at M ≤ 8, MID
     has `nfj = 0` forced by `OKC_SPLIT` (`:4249-4251`), and the build `__trap()`s if a fat unit ever appears
     (`:6223`);
   - use `NW = 8`, so the `#if OKC_NW == 16` half-slice scale read (`:4575`) — which has no 6-bit form — is not
     compiled.
4. **Probe** (`_sc6s_probe`, `:7177`): (a) the byte round-trip, every block of w13's units:
   `_sc6_unit` → `_sc6_unit_back` (the inverse of the kernel's `sc_load`, `:4389`) must reproduce the 4,096-byte
   layout exactly, `max|d| == 0`; (b) the kernel itself: the **disabled** MID build (M = 16) and TINY build (M = 4)
   are run on the task's own routing distribution on the same buffers *before* the repack, then the **enabled**
   builds on the 6-bit buffers, two runs each, and the outputs compared as `max|d| <= floor` where `floor` is those
   same builds' run-to-run difference (`_mb_ab`, `:7168`) — the skinny epilogue sums a route's K strips into the row
   sum with `atomicAdd` (`ss_reduce_add`, `:4352`), so an exact 0 is not assertable when the hardware orders those
   adds differently run to run, while a wrong block scale is off by >=2x, four orders above that floor. On a shape
   whose epilogue is deterministic the floor is 0 and the requirement is exactly `max|d| == 0`. Any exception
   (compile, launch, smem) counts as a failure.

On failure the plain unit copies are re-packed, `_MA_SC6S` is cleared, and every cached MoE build is dropped
(`_clear_moe_builds`) so nothing keeps the 6-bit decode.

### Consistency of the decode path (`:4369`, `:4387`, `:4389-4399`, `:4479/:4693/:5326`)

Everything that must agree derives from the single `UNIT_SC`: `UNIT_CODES`/`UNIT_SC`/`UNIT_BYTES`, the bulk-copy
size and `expect_tx` (`ring_issue`), every `sl * UNIT_BYTES` ring address, `okc_smem_bytes()`'s
`OKC_NS * UNIT_BYTES`, and `SCK = OKC_SC6S ? 4 : 16` used as the scale decode's opaque multiply (`k16` at
`:4479/:4693/:5326`) — 4 is the 6-bit re-bias (`(0x40 + v) * 16 + 0x5C00 == 4v + 0x6000`, `SCDK` at `:4384`).
`sc_load` under `#if OKC_SC6S` reads three 1,024-byte planes at `UNIT_CODES + 1024w + 128·warp + 4·lane` instead of
one `uint4` per lane at `UNIT_CODES + 512·warp + 16·lane`; the Python packer writes exactly that layout
(`_sc6_unit`: plane `w` byte `n = 4·v[4w+n] | ((v[12+n] >> 2w) & 3)`, which is what `sc_load`'s
`r.x/r.y/r.z = planes & 0xFC` and its recombined `r.w` invert). Unit size 34,816 B is a 16-byte multiple, as the
bulk copy requires.

### Kill criterion

Kill if any of the three loads' MoE fetched bytes does not fall by 2.78% (±0.5% measurement noise on the per-kernel
row), or if the layer does not gain **+0.4%** (the lever is predicted at +0.68%, so 60% of it must show).
The guard and the probe are the automatic kill: a scale window outside `0x40..0x7F`, a non-skinny build picking up
the flag, or one mismatching output byte reverts the whole lever to the shipped layout within `prepare()`.

---

## Lever B — `OKC_XRES`: stop re-fetching the job's x for every unit — **+1.18% layer**

### Where the refetch is

`wgs_xproducer` (`:5731-5784`) issues one bulk copy per **stage** into a 3-slot ring (`WGS_NX = OKC_WGSNX = 3`):

```
bytes = npad * 512 ; src = jb + st * npad * 256      // stage st of any unit of the job, npad = ceil(cnt/8)*8
```

`st = gs & 15` and the source depends only on `st` — not on the unit. A job walks 32 units (phase A) + 16 units
(phase B), each with stages 0..15, so the **same 16 x blocks are copied 32× and 16× respectively**: 768 copies of a
16-block set that is 16 blocks large. With `WGS_NX = 3` a block is evicted 13 stages before it is wanted again, so
13 of every 14 of those copies go to the memory system instead of hitting a resident copy. That is the
"`npad·512` x refetch per stage" in `moe_struct.md`, and it is the only excess byte in the layer.

### Byte account at v32 (T = 256) `[DERIVED from the lever report + moe_struct]`

| | fetched | touched | ratio |
|---|---|---|---|
| weights (judge floor, ~19.1 experts × 28.3 MB) | 540 MB | 540 MB | 1.00 |
| x, per-load demand 19 jobs × 768 stages × 8 KB (`npad = 16`) | 120 MB | 2.4 MB unique (19 × 128 KB) | 50× |
| **total, measured** | **622 MB** | 542 MB | **1.152** |

So **82 MB of the 622 MB is the x refetch**, and only 2.4 MB of it is unavoidable. Keeping the job's x resident in
L2 for the duration of the job turns the other ~79.6 MB into L2 hits: fetched 622 → ~542 MB (**−12.8%** of the MoE's
bytes at v32, ratio 1.152 → 1.005). At the profile's 2.39–2.54 TB/s that is −24.1 µs on V32S and −25.7 µs on V32F,
which reproduces the lever report's −24.3 / −26.0 µs and its **+0.86 + 0.32 = +1.18%** board.

### Why L2 and not shared memory (the implementation decision)

The 16-block residency cannot be done in smem: a v32 job has `npad = 16`, so its block set is 16 × 8 KB = 128 KB,
and the shipped WGS build already spends 208,896 B of the 232,448 B of dynamic smem (`WGS_NS·UNIT_BYTES = 3 ×
36,864` weights + `WGS_NX·WGS_XS = 3 × 32,768` x slots, plus `sizeof(Smem)` ≈ 4.7 KB) — 13.8 KB free. The x slot
stride is `WGS_XS = FT_T/8 · 4096 = 32,768 B` (sized for a 64-token job), so even 16 slots would need 512 KB. The
lever report reached the same conclusion ("115 KB does not fit alongside a 108 KB weight ring in 226 KB"). Handing
over the third weight slot does not close the gap either.

So the residency is spent in L2, which is where the 82 MB is actually lost: the live set is one job's blocks per
CTA and all CTAs on a job share the same lines, so the pin is ~2.4 MB at v32 (≈16 MB if every expert of the 32 is
touched), against a 50 MB L2 whose weight stream is already marked `evict_first` (`L2H_ARG`, `:3999`).

### The change (`:5547-5549`, `:5742-5748`, `:5773-5779`)

```c
#ifndef OKC_XRES
#define OKC_XRES 0
#endif
...
uint64_t xpol;                                              // once per launch
asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(xpol));
...
// the x copy only:
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [dst], [src], bytes, [bar], xpol;
```

Nothing else changes: the same destination and size, same mbarrier protocol, same slot rotation, same
`WGS_NX`, same smem size, same weight copies. The `#else` branch is the shipped instruction verbatim.

### Bit-exactness and the probe

A cache policy cannot change a value, so the guard is structural — but the probe (`_xres_probe`, `:7208`) still
runs the WGS main build at **M = 256** (the shape the lever touches) with `-DOKC_XRES=1` and `-DOKC_XRES=0` on the
same `st["ma_w"]` and the same input, two runs each, and requires `max|d| <= floor` (those builds' own
run-to-run difference, `_mb_ab`; 0 on a deterministic epilogue); an exception (compile, launch, smem) also fails it. On failure `_MA_XRES` is cleared, `st["ma_ext"]` and every cached build are dropped, and the shipped refill is
used. The disabled twin is compiled in `prepare()`'s parallel nvcc batch, not on the critical path
(`:11372-11374`).

### Kill criterion

Kill if the v32 MoE fetched/touched ratio does not fall from **1.15 to ≤1.03**, or if the V32S cell does not drop
**≥10 µs** (predicted −24 µs, so 40% must show). Also kill if any *other* shape regresses: the pin holds ~2–16 MB
of L2 away from the kernels that follow the MoE launch (attention's KV gather, the dense chain), so D32F and V16F
are the regression checks — if either loses >2 µs the pin is taken with a `createpolicy.fractional` fraction < 1.0
or the lever is dropped, and the third weight ring slot stays as is either way (the lever does not spend it).

---

## Summary

| lever | flag | expected | measured basis | automatic kill |
|---|---|---|---|---|
| A | `-DOKC_SC6S` (`SC6_SKINNY`) | **+0.68%** | −2.78% of the MoE bytes of d1/v1/d16 (17.8 MB of 749 MB) | `_sc6_ok` window + byte round-trip (`== 0`) + MID(M=16)/TINY(M=4) kernel A/B, `max|d| <= floor` |
| B | `-DOKC_XRES` (`XRES_L2`) | **+1.18%** (1.2%) | 622 → ~542 MB fetched at v32 (82 MB of x refetch, 2.4 MB unavoidable) | WGS main build at M=256, `max|d| <= floor`; ratio ≥1.03 or a neighbour shape's regression kills it |

Untouched, as required: the ATT7/attention path, the dense/sk2 path, the PDL predicate, the MoE build thresholds,
the WGS ring depths, the barrier protocol and the fat/unit/WGS weight copies other than the skinny unit scales.
