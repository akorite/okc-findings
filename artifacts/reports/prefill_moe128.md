# `muse_prefill_layer` — the n=128 MoE tile arm: built, and priced out

Candidate: `src/okc_loop/vendored_work/prefill/prefill_moe128.py` (`prefill_gap1.py` + the arm).
Arm off by default. **Verdict: do not run it.** The brief's `−3…−7%` does not survive; the arm is
`+5.8% … +24.2%` on the MoE at `T = 16384` and exactly `0%` on the other five shapes. The brief's
own "30% gap to wgmma time" number is what kills it: that 30% is per (item, k-block) and does not
shrink when the item does.

## 1. What the arm actually is (and what it is not)

The kernel already had the n=128 tile: `consumer_item<128>` is dispatched today
(`moe_fused_persistent`, `case 128:`), because every expert's *remainder* tile is 128 rows wide
whenever its route count mod 256 lands in 65…128. So the arm adds **no kernel, no instantiation,
no register pressure**. It is a `plan_warp` change: full tiles of `MOE_TT` rows instead of 256,
with the plan's flat-row decomposition (`route pos → (tile, row-in-tile)`) shifted by
`MOE_TT_SH = 7` and masked by `MOE_TT_MSK = 127`.

Diff (4 seams, all parameterized by one constant):

| seam | change |
|---|---|
| `plan_warp` | `full = c >> MOE_TT_SH`, `rows = full*MOE_TT + nr + nb`, full tiles `make_int4(e, r0 + MOE_TT*i, MOE_TT, MOE_TT)` |
| size classes | `MOE_NBIG = MOE_TT/32 − 2` big classes (`MOE_TT … 96`), small classes at `cbase[MOE_NBIG]`/`[MOE_NBIG+1]`; `MOE_TT − 32k` lands on 64/32 at `k = MOE_NBIG, MOE_NBIG+1` |
| `moe_gather_kernel<PLAN>` | PLAN-block row scan, pad-fill base, and the token scatter's `tl = ps >> MOE_TT_SH`, `lr = ps & MOE_TT_MSK` |
| `rem_tile` | caps at `MOE_TT` (the fine widths 160/192/224 can never appear at 128: `r = c & 127 < 128`) |

At `MOE_TT = 256` every replaced expression is the literal it replaced (`MOE_NBIG = 6`,
`MOE_TT_SH = 8`, `MOE_TT_MSK = 255`), so the shipped path is unchanged by construction. Verified,
not assumed: a transliteration of `plan_warp` over **300 random count vectors × 3 orders × 2 fine
settings × 2 widths** reproduces the shipped tile list *exactly* at 256, and at 128 every expert's
rows are covered exactly once, contiguously, with no duplicate or overlapping tile slot
(`/tmp/plan_check.py`).

## 2. Register / occupancy table (real ptxas numbers, sm_90a)

`tools/cu_build.py` cannot report registers for this file (its `_CUDA_SRC` kernels are templates
instantiated from the host glue, which the tool strips; and nvrtc 12.9 under `-default-device`
drops every entry when `<stdint.h>` is included — both pre-existing, both true of `prefill_gap1.py`
too). The numbers below come from the same nvrtc + ptxas the tool drives, with the two template
instantiations forced explicitly:

| build | registers | spills | stack |
|---|---|---|---|
| shipped (`case 128` + `default: 256`, both inlined) | **168** | 0 | 8 B |
| `-DMOE_TT=128` | **168** | 0 | 8 B |
| N=256 path only (128 path elided) | 168 | 8 B stores / 24 B loads | 16 B |
| N=128 path only (256 path elided) | 168 | 0 | 8 B |

Occupancy is identical in every build and is set by the launch shape, not by the tile:
`__launch_bounds__(384, 1)` → 384 × 168 = 64,512 of 65,536 registers, 227 KB dynamic smem ⇒
**1 CTA/SM**. `setmaxnreg` only *moves* that allocation: producer (128 threads) `dec 40`,
consumers (256 threads) `inc 232` → 128·40 + 256·232 = 64,512, the same budget.

Live set per path, for the record: N=256 → `GROUP4` false, `NACC = 1`, `acc[128]` (N/2 live) +
`afrag[2][4]` = 8 ⇒ ~136 registers; N=128 → `GROUP4` true, `NACC = 2`, `acc + acc2` = 2×64 = 128 +
`afrag[4·AFB][4]` = 32 ⇒ ~160. The 128 tile costs **+24 registers of live state and zero spills**.

> The brief's premise — "the tile family is register-exhausted at `setmaxnreg 232`, which is why 256
> was chosen" — is not what the compiler says. 232 is the *consumer* share of a fixed 64,512 and the
> 128 path already ships inside it, spilling nothing.

## 3. Independent check: is the item set by wgmma throughput, or by barrier/issue?

Re-derived from the shape and the rate, not from the brief:

```
one item = one (token tile, n-tile): M = 128 weight rows, N = tile width, K = 4096
FLOPs    = 2 · N · 128 · 4096                     = 268.4 MFLOP at N = 256
per-SM   = 989.4 TFLOP/s / 132 SMs               = 7.495 TFLOP/s   (H200 BF16 dense, published)
tau(256) = 268.4e6 / 7.495e12                     = 35.81 µs   <- matches the brief's 35.8 exactly
tau(128) = 17.91 µs,  tau(64) = 8.95,  tau(32) = 4.48
```

So 35.8 µs *is* the pure tensor-core time of one full-width item on one SM, and the measured
51.4 µs is **70% of that floor** — the wgmma issue stream is not the problem, the feeding of it is.
The per-item overhead `O = measured − tau` falls out of the incumbent's own profile at three tile
widths, because the incumbent *already runs three widths*:

| shape | tile width | items | MoE µs | µs / item | tau | **O = µs/item − tau** | tensor share |
|---|---|---|---|---|---|---|---|
| turn1 (T=753, 32 rows/expert) | 32 | 32×96 | 239.3 | 10.28 | 4.48 | **5.80** | 44% |
| turn4 (T=3012, 127 rows/expert) | 128 | 32×96 | 751.4 | 32.22 | 17.91 | **14.31** | 56% |
| first (T=16384, ~691 rows/expert) | 256 | 96–128×96 | 3587.5 | 38–51 | 35.81 | **16.1** | 63–70% |

One overhead ladder `O(32)=5.80, O(128)=14.31, O(256)=16.1` reproduces **all three** measured
whole-kernel MoE times to ≤0.3% (239.2 / 749.8 / 3575 µs vs 239.3 / 751.4 / 3587.5) with the plan's
own padding model and the fitted `1.35·T` route count. `O` is therefore *measured*, and it is
nearly flat between 128 and 256: it is the NVFP4→BF16 A-fragment dequantisation
(`convert8`, 256 calls per thread per item, the same number at every width) plus the 32/64
per-stage barrier waits — all per (item, k-block), none of it per token.

**The arm's arithmetic, exactly.** Per 256 tokens:

```
N=256:  1 item   tau 35.81 + O(256)
N=128:  2 items  2 · (tau 17.91 + O(128)) = 35.81 + 2·O(128)
delta   = (2·O(128) − O(256)) / (35.81 + O(256))
```

Break-even needs `O(256) = 2·O(128) = 28.6 µs`, i.e. the 256 tile would have to waste **28.6 µs**
per item against the measured **16.1** — the premise is inverted. With `O(256) = 16.1`:
`(28.6 − 16.1)/51.9` = **+24%**. Even under the most generous reading (`O(256) = 25 µs`, which the
`1.35·T` accounting cannot support), it is **+5.8%**. Any decomposition in between is a loss; the
only way to break even is for the *entire* 16 µs overrun to be proportional to N, and then the
delta is exactly 0, not negative.

Independent, composition-free confirmation: bytes read per padded row are
`8192 (X tile) + 294,912/W_tokens = 9344 B at N=256` vs `10,496 B at N=128` → **+12.3% data per
row**, with the weight stream doubling from 2.7 GB to 5.4 GB over the same 3.6 ms. A kernel that
already moves that stream at 6–8 TB/s aggregate has nothing to gain from asking for more of it.

## 4. Per-shape delta

`tiles/expert` is from the plan's own accounting at the fitted `1.35·T` routes; the five small
shapes produce a **bit-identical plan** at both widths (32- and 127-row remainders do not touch the
full-tile width), so their delta is exactly zero, not "small".

| load | shape | T | MoE µs | MoE share | tiles/expert 256→128 | Δ MoE | Δ shape |
|---|---|---|---|---|---|---|---|
| first | ctx0, full | 16384 | 3587.5 | 39.4% | 4 → 6 | **+5.8 … +24.2%** | +2.3 … +9.5% |
| first | ctx32K, sliding | 16384 | 3750.0 | 42.1% | 4 → 6 | **+5.8 … +24.2%** | +2.4 … +10.2% |
| first | ctx32K, full | 16384 | 3421.0 | 31.5% | 4 → 6 | **+5.8 … +24.2%** | +1.8 … +7.6% |
| first | ctx65K, full | 16384 | 3488.9 | 27.4% | 4 → 6 | **+5.8 … +24.2%** | +1.6 … +6.6% |
| turn 1 | sliding | 753 | 239.3 | 44.5% | 1 → 1 | 0 | 0 |
| turn 1 | full | 753 | 241.9 | 31.4% | 1 → 1 | 0 | 0 |
| turn 4 | sliding | 3012 | 751.4 | 42.7% | 1 → 1 | 0 | 0 |
| turn 4 | full | 3012 | 732.9 | 28.3% | 1 → 1 | 0 | 0 |

Load-level, at `1.35·T` and the `cases.py` weights: the **first-turn load** (the only load that
contains any T=16384 shape — 186 of 186 of its weight) goes **+2.3% … +9.5%**; the score's geometric
mean moves **−0.7% … −3.0%**. The brief priced the arm at `−400…−900 µs at T=16384`; the measured
overhead ladder prices it at **+280…+1290 µs**, same order of magnitude, opposite sign.

Where the brief's number went wrong: it assumed the 30% per-item overrun is *per-stage barrier and
issue* and would therefore shrink with the stage. The stage count per item does halve (32 instead of
64, `MOE_KPS128 = 2`), and the item count doubles, so the barrier total is unchanged; but the
dominant term — the A-fragment dequantisation, 256 `convert8` calls per thread per item — has no N
in it at all and is simply paid twice. The one place the incumbent does beat it, `O(32) = 5.8 µs`,
is the one place the accumulators are deep enough to hide it (`NACC = 4` at N ≤ 32, `NACC = 2` at
128, `NACC = 1` above).

## 5. Enable flag, and the kill criterion

Built and off. Two equivalent switches, both one token:

| to | do |
|---|---|
| host (`_build`) | `MOE_TILE128 = True` (line 141) prepends `#define MOE_TT 128` to `_CUDA_SRC` |
| offline / `-D` | `.venv/bin/python tools/cu_build.py <file> -D MOE_TT=128` (guard `static_assert(MOE_TT == 256 \|\| MOE_TT == 128)` fires on anything else) |

Status of the acceptance checks:

* lint `okc_loop.cli lint --limit 500`: **62 findings, identical set to `prefill_gap1.py`** (3 errors,
  59 warnings; the one textual difference is the line number inside the pre-existing "line 3327 is
  1686 chars" message, shifted by the 4 added lines). Zero new findings.
* `tools/cu_build.py`: `ok _CUDA_SRC` for sm_90a, both with and without `-D MOE_TT=128`. The shipped
  file needed one enablement fix to get there at all — `#ifndef INFINITY` → `__int_as_float(0x7f800000)`,
  because the offline check is nvrtc and its include chain does not bring the libc macro in (nvcc's
  does; the guard is inert under the judge's build). That fix is in this candidate, not in
  `prefill_gap1.py`, and is the only non-arm source change.
* numerics: the arm is *not* bit-identical to the shipped path even for the same rows (`NACC = 2`
  at N=128 alternates k-blocks between two accumulator sets before summing), so the gate is the
  judge's own `REL_L2_TOL` check, not an equality check. It should pass with margin — this is the
  arithmetic the incumbent already runs for its 128-row remainder tiles.
* not validated: anything requiring a GPU. No device was available; the plan, the register counts
  and the cost model are all offline.

Kill criterion, one check, `first ctx0` only (T=16384 is the shape the arm exists for; the other
seven are 0% by construction):

| measured MoE span at `first ctx0` | action |
|---|---|
| ≥ 3900 µs (model −7%) | **kill the arm**, delete the flag, keep `prefill_gap1.py` as the candidate |
| 3480–3900 µs | inconclusive (model says 3790–4460): one more check, then kill unless the second run is ≤3480 |
| ≤ 3480 µs (−3% vs 3587.5) | keep, and only then spend a submit on it |

Belt and braces: run it against a *closed* problem first — the arm must reproduce
`first ctx0` within tolerance on the same weights before any timing is believed.

## 6. What the same measurement says *does* pay

`tau` is the floor and `O` is the tax, so the money is in **tokens per dequantised weight block**,
and the arm moves the wrong way. The wgmma `m64n256k16` caps one item at N = 256, so the only way to
buy more tokens per weight fetch is to stop rebuilding the A fragment in registers per item:
dequantise A into **shared memory once** and run two 256-token tiles against it with the ss-form
wgmma. That removes one `O(256) = 16.1 µs` per 512 tokens, i.e. `2·51.9 → 87.7 µs`, **−15% on the
MoE** — the exact mirror image of the arm's loss, and worth ~+2% on the score through the first-turn
load. It is a much larger change (a new A layout, an ss-form MMA path, a second X staging slot) and
it is where a follow-up arm should go.

Same-source corollary, free: `O` is flat from 128 to 256 while the tensor share runs 56% → 70%, so
the plan's **remainder policy is already the right one** (128 + 64 beats one 256 for 129…192 rows,
as the code comment claims) and there is no cheaper remainder fix hiding behind it.
