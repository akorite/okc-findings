# OCC2: can the MoE run 2 CTAs per SM?

**Yes — for the skinny bands (M <= 16) only, and the arithmetic is not close.** The blocker was never
the SM's shared memory: it was `okc_smem_bytes()` charging every TINY/MID build for the **fat** ring
(178,176 B) that `OKC_TINY` compiles out. Drop that term and the ring to 2 slots and a CTA asks for
**110,336 B**, i.e. 2 CTAs = 220,672 B of the 231,424 B two CTAs may hold (slack 5,376 B/CTA).
Registers are the binding constraint and are met by capping at **112** (measured: 112 regs,
24 B spill stores / 8 B loads for MID, 0 / 0 for TINY; the default path is unchanged at 149 / 145).

`src/okc_loop/vendored_work/moe/moe_occ2.py`, flag `OCC2 = False`, defines `+OKC_OCC2=1 +OKC_NS=2`,
grid `2 x nsm` for M <= MID_MAX_M. Gates: `okc_loop.cli lint` PASS; `tools/cu_build.py` 12/12 ok,
exit 0.

Sources: `artifacts/cas/6402f5f3…d2b3e5.py` (the 2.3746 MoE board file). All numbers below are from
that file, `artifacts/reports/occ2_probe.py` (nvrtc 12.9 + ptxas sm_90a, explicit instantiation of
`moe_fused<false>/<true>` so ptxas reports the real `__global__`), and `ptxas -v`.

---

## 1. The shared-memory budget

### 1.1 What the kernel actually queries

Nothing. There is no `cudaFuncGetAttributes`, no `cudaDeviceProp.sharedMemPerMultiprocessor`, no
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` anywhere in the file (grep: 0 hits). The only
shared-memory API it calls is

```c
cudaFuncSetAttribute(okc::moe_fused<false/true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
```

with `sm = okc_smem_bytes()` (`:2862`, `:4104`). So the per-SM figure is the hardware constant, not
something the code reads:

| fact | bytes | note |
|---|---|---|
| `sharedMemPerMultiprocessor` (sm_90) | **233,472** (228 KiB) | the whole budget |
| `sharedMemPerBlockOptin` (sm_90) | 232,448 (227 KiB) | 1 KiB less: driver reserve |
| 2 CTAs may hold, in total | **231,424** | 233,472 − 2 x 1 KiB (reserve is per resident CTA) |
| **per CTA for 2 CTAs/SM** | **115,712** | 231,424 / 2 (116,736 if the reserve were per SM) |

### 1.2 `okc_smem_bytes()` = base + max(xa, xb)

```
base = ceil128(sizeof(okc::Smem))                       // the routing/job tables + barriers
xa   = OKC_NS  * UNIT_BYTES                             // skinny ring (+ 2nd-n-tile stages when SKT > 8)
xb   = NSF * UNIT_BYTES + 2 * FT_T * XP_FT * 2          // fat ring + its two x stages
```
`UNIT_BYTES = 32,768 codes + 4,096 scales = 36,864` (`:657`); `FT_T = 64`, `XP_FT = KS + 8 = 264`
(`:243,344`); `NSF = 3`, `OKC_NS = 3` on sm_90 (`:3091-3105`). WGS builds take a different branch
(`base + WGS_NS*UNIT_BYTES + WGS_NX*WGS_XS`).

### 1.3 Per-CTA itemisation, skinny bands (NW=8, OKC_TINY=1, OKC_WS=1)

`sizeof(Smem)` measured by forcing the compiler to print it (`ShowSize<sizeof(Smem)>`):
**36,576** (base 36,608 after the 128 B align).

| field | bytes | who needs it |
|---|---|---|
| `lut[256]` | 1,024 | scale LUT (only when `!OKC_PRMTSC`) |
| `red[WSNR=4][NW=8][32][RED_W=4]` floats | 16,384 | cross-warp partials, compute warps -> helper warp |
| `wse[8] wsrf[8] wsre[8]` (uint64) | 192 | WS ring partial/ready/free barriers |
| `wsr[4]` | 16 | this CTA's phase A/B unit ranges |
| `bars[12]` (uint64) | 96 | 0..4 skinny ring, 5..11 fat ring (fat: dead here) |
| `seq, fseq, preissued, preissuedB` | 16 | ring cursors |
| `fcnt[32]` | 128 | FAST1 per-expert entry counts |
| `xbuf_ptr` | 8 | ring base |
| `cnt[64] + off[64]` | 512 | per-expert token counts/offsets |
| `wc[NW=8][32]` | 1,024 | per-warp unit claim |
| `jE/jS/jC/forder[JMAX=88]` | 1,408 | job table |
| `njF, njS, gen(+pad), ssp` | 24 | job counts, ss generation, ss pointer |
| `uslot[8]` | 32 | ring slot -> unit id |
| `dync[JMAX=88]` | 352 | phase-A pool units per job |
| `eid[EMAX=3072]` (char) | 3,072 | entry -> expert |
| `ent[EMAX]` (short) | 6,144 | sorted entries |
| `pos[EMAX]` (short) | 6,144 | entry -> pair position |
| **struct** | **36,576** | (measured; 36,572 + 4 pad) |
| **base** | **36,608** | ceil to 128 |
| `xa` at `NS=3` | 110,592 | 3 x 36,864 |
| `xb` at `NSF=3` | **178,176** | 3 x 36,864 + 2 x 64 x 264 x 2 = 110,592 + 67,584 |

`okc_smem_bytes()` returns `base + max(xa, xb)` = 36,608 + 178,176 = **214,784** for TINY/TINY2/MID —
and `xb` is dead weight: `run_phase` (`:2371`) is

```c
#if OKC_TINY
  if (nF > 0) __trap();   // a job never has more than 8 tokens: there are no fat units
#else
  if (nF > 0) fat_stream...        // the only reader of slots NSF..NSF+1 and of xs0 = ring + NSF*UNIT_BYTES
#endif
```

So the 178,176-byte fat ring and its two x stages are allocated in every skinny build and never
touched. **That is 104,448 B of the 214,784** (and the reason "2 CTAs never fits" was true of the
*current* number, not of the kernel).

MAIN/WGS for comparison (NW=16): `sizeof(Smem) = 22,224 -> base 22,272`, `WGS_NS=3 x 36,864 =
110,592`, `WGS_NX=3 x WGS_XS(8 x 4,096 = 32,768) = 98,304` -> **231,168** (matches SassAudit).

### 1.4 Current per-CTA totals

| build | threads | regs | smem B | 2 CTAs? |
|---|---|---|---|---|
| MAIN/WGS (`M >= 17`) | 576 | 96 | 231,168 | no — 2 x 231,168 > 231,424 |
| TINY (`M <= 4`) | 288 | 145 | 214,784 | no — 429,568 |
| TINY2 (`M <= 8`) | 288 | 145 | 214,784 | no |
| MID (`M <= 16`) | 288 | 149 | 214,784 | no |
| **OCC2 (this variant)** | 288 | **112** | **110,336** | **yes — 220,672, slack 10,752 B** |

## 2. What to shrink, and what is safe

| candidate | saving | safe? |
|---|---|---|
| **drop `xb` for `OKC_TINY`** (fat ring + x stages) | **178,176** | yes: `OKC_TINY` traps on a fat unit, the code is compiled out. This is a correctness fix to `okc_smem_bytes()`, not a tuning knob. Gives 36,608 + 110,592 = 147,200 at `NS=3`. Still 31,488 too big. |
| **`OKC_NS` 3 -> 2** | 36,864 | yes, generic: every ring index is `q % OKC_NS`, parity `(q / OKC_NS) & 1`, prologue/refill loops are `for (i < OKC_NS)`. Barriers 0..1 of `bars[12]`, `uslot[8]` sized for NS <= 5. **110,336 -> fits.** |
| `OKC_NS` 3 -> 1 | 73,728 | correct but ring depth 1: no unit prefetch at all (73,472 total). Emergency option only. |
| `OKC_NSF` 3 -> 2 or 1 | 36,864 / 73,728 | inert for TINY builds (`xb` is already 0); for the fat builds it changes the stage count of the K loop (`ks + NSF` refill). Not needed. |
| `FT_T` 64 -> 32 | 33,792 of `xb` | **not safe to touch**: it moves `JMAX`, the fat job split, `WGS_NT = FT_T/8`, `WGS_XS`, `WGS_FIXW`, and the WGS cost tables. Irrelevant to the skinny bands (`xb = 0`) — do not trade that risk for a saving we already have. |
| `red` `WSNR` 4 -> 2 | 8,192 | legal (`static_assert(WSBLK <= WSNR)` keeps it; WSBLK=2 in MID, 1 in TINY2) but it halves the partial-buffer pipeline for the helper warp's epilogues. Only buys slack we do not need (would give 102,144). |
| `EMAX` tables (15,360 in `eid/ent/pos`) | scales with `CHUNK_TOKENS` | not needed; changes the host scratch contract. |

Minimum change that reaches 2 CTAs: **`xb = 0` under `OKC_TINY` + `OKC_NS = 2`**. Nothing else.

## 3. The partition and the grid

`grid` enters the kernel only as `gridDim.x`, and the host passes `state["grid"] =
multi_processor_count` (132 on H200) to `okc_run` -> `<<<grid, NTL, sm, st>>>`. Doubling it is correct
by construction — every split goes through `gridDim.x`:

| site | code | grid-generic? |
|---|---|---|
| `prepare` `:3336` | `"grid": get_device_properties(dev).multi_processor_count` | the only producer of the grid; `kernel_fn` passes `state["grid"] * 2` for the skinny bands |
| `early_issue` `:1576` | `muldiv_g(nS, blockIdx.x, gridDim.x, rG)` | yes |
| `route`/`skinny_split` (phase split) | `muldiv_g(nS, c, G, rG)`, `muldiv_g(nF, 1, G, rG)` | yes |
| `run_phase` `:2371` | `nSs = nS - dyn_pool(nS, G)`; `skinny_split<BIG,PB>(nSs, nF, G, c, v0, v1)` | yes |
| `dyn_pool` `:344` | `G * OKC_DYNK` units claimed atomically | yes by design; `SKDYN = 0` whenever `OKC_TINY` (`OKC_TDYN 0`), so the pool is empty and the static split covers all units |
| `WS` ranges `:2574` | `S.wsr[0..3] = muldiv_g(nA/nB, c, G, rG)` | yes |
| WGS ranges / cost tables | `muldiv_g(R, blockIdx.x, gridDim.x, rG)`, `wgs_cta_of(S, PB, u << 4, gridDim.x)` | yes, but see the slot table below |
| grid barrier `:2631` | `while (ld_acquire(bc) < (int)gridDim.x)`, bump when `old == gridDim.x - 1` | yes — any grid, **provided all CTAs are resident** |
| flag reset `:2645` | `for (i = i0; i < JMAX; i += G * NT)` | yes |
| final pass `:2662` | `for (tk = blockIdx.x; tk < M; tk += gridDim.x)`; `itp = blockIdx.x; hasp = itp < M * FSEGP`, `FSEGP = HD/(NT*4) = 4` | yes: at M <= 16 only 64 of 264 CTAs have items; the rest idle to the barrier |

The one hard grid assumption in the file:

```c
constexpr int WGS_SLOTS = 160;   // fixup slots per phase (>= grid + 1)      :1830 (and host :120)
const int slot = (s1 - s0 == 16) ? -1 : (s0 == 0 ? (int)blockIdx.x : wgs_cta_of(S, PB, u << 4, gridDim.x));
```

The fixup slot **is a CTA id**, so the WGS build needs `grid <= 159`. At 2 x 132 = 264 it overflows
the stream-K fixup area (and raising it to 265 costs 2 x 105 x 4 x 4,096 x 4 B ≈ 13.8 MB of host
scratch). MAIN also cannot do 2 CTAs on registers (§4), so **OCC2 is restricted to M <= MID_MAX_M
and MAIN keeps `grid = #SMs`** — the launch site computes `grid = state["grid"] * 2 if occ2 else 1`
with `occ2 = OCC2 and M <= MID_MAX_M`, so TINY/TINY2/MID double and MAIN never does.

Second-order check: `muldiv_g` divides without an integer divide and is exact while `a * c < 2^24`.
At M <= 16, `nS = njS * RB_A` (njS ~ 32-40 jobs x 512 16-row blocks ~ 16k-20k units), so
`nS * grid = 20,480 x 264 = 5.4 M < 2^24`. Fine (at 132 it was 2.7 M).

## 4. Registers

65536 registers per SM, allocated in 8-register granules per warp.

| shape | threads/SM | regs/thread for 2 CTAs | what the build uses | verdict |
|---|---|---|---|---|
| MAIN geometry, 2 CTAs | 2 x 576 = 1152 | **56** (32768/18 warps/32) | 96 | **impossible**: 96 x 1152 = 110,592 > 65,536 |
| MID geometry, 2 CTAs | 2 x 288 = 576 | **112** (32768/9 warps/32 = 113.7) | 149 (MID), 145 (TINY/TINY2) | 37-39 over, but ptxas reaches 112 cheaply (below) |
| 256-thread (no helper warp), 2 CTAs | 512 | 128 | — | **not buildable in this file**: the pruning removed the non-WS skinny path (`-DOKC_WS=0` fails: "class Smem has no member wse/wsrf/wsre"), so 288 threads is the only skinny geometry that exists |

So the answer to "which existing build is closest" is **MID at 288 threads**: same thread count as
the 2-CTA shape, 37 registers over, and a cap of 112 costs almost nothing:

| build | cap | regs | spill stores | spill loads | stack |
|---|---|---|---|---|---|
| MID (today) | none | 149 | 0 | 0 | 0 |
| **MID, `__launch_bounds__(288, 2)` — shipped** | ptxas picks | **96** | **116 B** | **56 B** | 40 B |
| MID, `__maxnreg__(112)` | 112 | 112 | 24 B | 8 B | 8 B |
| MID, `__maxnreg__(104)` | 104 | 104 | 24 B | 8 B | 8 B |
| TINY, `__launch_bounds__(288, 2)` — shipped | ptxas picks | 96 | 112 B | 48 B | 32 B |
| TINY, `__maxnreg__(112)` | 112 | 112 | 0 | 0 | 0 |

The variant ships the `launch_bounds` form because it is the only one that is safe in *every* band:
`__maxnreg__` cannot be combined with `__launch_bounds__`, and a bare `__maxnreg__(255)` would drop
MAIN's launchable-allocation guarantee (576 threads x 113 regs is the last legal step, so ptxas
freeing itself to 255 could produce a kernel that cannot launch at all). `launch_bounds(.,2)` caps at
exactly 112 for the 2-CTA case and at 224/112 for the 1-CTA cases, which is what the current
`launch_bounds(NTL,1)` already implies. If the first check is ambiguous, the cheap follow-up is
`__maxnreg__(112)` on a skinny-only source: it cuts the spill 5x (24/8 vs 116/56).

Where the spill is: replacing one half of the 64-register `uint4 x0[2*NCH]` x-operand array
(`load_x_regs`, `:700`) with constants removes *all* spills at the same cap, so the spilled values
are the register-staged x, which is reloaded per unit. At 112 regs that is 2 words per unit per
thread = 288 x 8 B = 2.3 KB per 36 KB unit (+6%, L1-resident: 8 B x 288 = 2.3 KB/CTA of stack
against the ~28 KB of L1 left after a 228 KB carveout). At the shipped 96 regs it is 14 words =
15.75 KB/unit (+43% of the unit's L1 read traffic, still L1-resident at 40 B x 288 = 11.5 KB/CTA),
i.e. ~14 extra LSU instructions in a loop that carries 32 HMMA and 8 LDG.128 per unit.

## 5. The variant

`src/okc_loop/vendored_work/moe/moe_occ2.py` (copy of the board file), `OCC2 = False` by default:

1. `#define OKC_OCC2 0` default in the raw source (the ALT `helper0` chunk gets it from the host).
2. `__maxnreg__(OKC_OCC2 ? 112 : 255)` (was `__launch_bounds__(NTL, 1)`; tested, but see §4 —
   **shipped as `__launch_bounds__(NTL, OKC_OCC2 ? 2 : 1)`**, which is safe in every band).
3. `size_t xb = OKC_TINY ? 0 : ...` in `okc_smem_bytes()` — the only byte that had to move.
4. `OCC2_DEFINES = ("-DOKC_OCC2=1", "-DOKC_NS=2")`, `_skinny_defines(d)` appends it for the
   TINY/TINY2/MID tuples when `OCC2`, and `_ext` passes `-DOKC_OCC2=<flag>` to every tuple (the ALT
   source has no default).
5. `kernel_fn`: `occ2 = OCC2 and M <= MID_MAX_M`; `grid = state["grid"] * (2 if occ2 else 1)`.

Gate results:

```
okc_loop.cli lint moe_occ2.py                       PASS (1/1 clean)
tools/cu_build.py moe_occ2.py                       12/12 ok, exit 0
  incl. CUDA_SRC [OKC_OCC2=1 OKC_NS=2 OKC_NSF=3]    compiles
probe helper0 + MID_DEFINES + OKC_OCC2=1 + NS=2     96 regs, 116 B stores / 56 B loads, 40 B stack
probe CUDA_SRC + TINY_DEFINES + OKC_OCC2=1 + NS=2   96 regs, 112 B / 48 B, 32 B stack
probe helper0 + MID_DEFINES + OKC_OCC2=0            149 regs, 0 spills  (default path untouched)
sizeof(okc::Smem) (TINY/OCC2)                       36,576 -> base 36,608
okc_smem_bytes()                                    36,608 + 2 x 36,864 = 110,336
```

## 6. What it costs, and what one check would settle

- **Peak math: unchanged.** Same 8 compute warps per CTA doing the same mma work; 2 CTAs = 16 compute
  warps + 2 helper warps per SM (18 warps) instead of 9. Nothing in the skinny loop is partitioned
  differently, so the per-CTA unit count simply halves (at M <= 16, `nS = njS x 512` ~ 16k units, i.e.
  ~124 per CTA at 132 CTAs and ~62 at 264).
- **Ring depth 3 -> 2** is the one real structural cost: per CTA only one 36 KB unit is in flight
  ahead of compute instead of two. Per *SM* the lookahead actually rises (2 CTAs x 2 slots =
  4 x 36,864 = 147,456 B vs 3 x 36,864 = 110,592 B), so the SM-level pipeline is deeper; a single
  CTA's cover of a TMA latency drops from ~2 units to ~1 (~1.2 us of compute per unit).
- **Spill:** +6% L1 traffic (2 words/unit/thread), 8 B stack per thread.
- **Final pass** at 264 CTAs: only `M * 4 = 64` CTAs have items at M = 16; the rest wait at the
  barrier. ~1/50 of the work, no correctness issue.

The defect this targets is a 1-7 us per-unit-end CTA stall (10.5-21.7 us per CTA at d32) with the
landed rate at 1.6-3.4 TB/s against 3.87-4.05 for the pure stream. With 2 CTAs/SM a stalled CTA's
unit-end epilogue can be covered by its neighbour's stream. The uncertainty a single check resolves
is whether the two co-resident CTAs *drift* far enough apart (they start together and the partition
gives each the same unit count, so their unit ends could coincide) — everything else (bytes, regs,
partition, ring legality) is settled above.

**Risk if it is tried:** the kernel's grid barrier needs every CTA resident. 2 CTAs/SM is guaranteed
by construction (112 regs <= 112 budget, 110,336 B <= 115,712 B), so a single wave of 264 CTAs fits
132 SMs; if a future edit broke either bound the kernel would hang rather than score slowly. Keep
`OCC2 = False` in any submission that is not this experiment.

## 7. Reproduce

```
PYTHONPATH=src .venv/bin/python -m okc_loop.cli lint src/okc_loop/vendored_work/moe/moe_occ2.py
TMPDIR=$PWD PYTHONPATH=src .venv/bin/python tools/cu_build.py src/okc_loop/vendored_work/moe/moe_occ2.py
TMPDIR=$PWD PYTHONPATH=src .venv/bin/python artifacts/reports/occ2_probe.py \
    src/okc_loop/vendored_work/moe/moe_occ2.py helper0 MID_DEFINES -DOKC_OCC2=1 -DOKC_NS=2
#   add --sizeof for sizeof(Smem); add -DOKC_LAUNCHMIN=2 to see the launch_bounds(.,2) 96-reg variant
```
