# dense_rate: the five serialized dense cells at T = 16/32/128/256

Candidate `src/okc_loop/vendored_work/layer/klr_drate.py` (989,496 B), base `klr_prsk.py` (qd2 + the `pr`
fused-GEMM alternatives). Nothing in the base was removed or replaced: every existing config is still the
incumbent and every new geometry is only a *candidate* that the file's own prepare-time bake must win.

## 1. Where the time goes (the judge's own profile, 82c098ae…7cd40fdd, per-kernel rows)

T is not the batch: `decode 16` = T 16, `decode 32` = T 32, `verify 16` = T 128, `verify 32` = T 256.
Bytes are the bytes the kernel actually streams for one cell (12-bit packed where the cell reads the packed
stream, BF16 where it reads the raw matrix); wall = bytes / 4.3 TB/s.

### T = 16 (`decode 16 sliding`, step 128.33 µs, weight 46; profile total 143.2 µs)

| cell | kernel | grid (CTAs) | MB | µs | TB/s | wall | excess |
|---|---|---|---|---|---|---|---|
| qkv_gate | `sk2<1,16,3,1,4,2,5>` | **96** | 28.3 | 10.7 | 2.64 | 6.6 | **4.1** |
| o | `sk2<1,16,4,1,4,1,4>` | 256 | 12.6 | 5.9 | 2.14 | 2.9 | 3.0 |
| pooler+router | `sk2<1,16,4,1,8,1,5,FUSE>` | 128 | 58.7 | 18.3 | 3.21 | 13.7 | 4.7 |
| unpooler | `sk2<1,16,4,1,4,1,5>` | 256 | 50.3 | 15.0 | 3.35 | 11.7 | 3.3 |
| **chain** | | | **149.9** | **49.9** | **3.00** | **34.9** | **15.0** |

### T = 32 (`decode 32 sliding`, step 144.94 µs; profile total 150.0 µs, GPU busy 145.2)

| cell | kernel | grid (CTAs) | MB | µs | TB/s | wall | excess |
|---|---|---|---|---|---|---|---|
| qkv_gate | `sk2<2,32,3,1,4,2,4>` | **96** | 28.3 | 11.7 | 2.42 | 6.6 | **5.1** |
| o | `sk2<2,32,2,2,2,2,3>` | 256 | 12.6 | 6.4 | 1.97 | 2.9 | 3.5 |
| pooler+router | `sk2<2,32,4,1,8,1,3,FUSE>` | 128 | 58.7 | 21.2 | 2.77 | 13.7 | **7.6** |
| unpooler | `sk2<2,32,4,2,2,2,3>` | **128** | 50.3 | 17.0 | 2.96 | 11.7 | 5.3 |
| **chain** | | | **149.9** | **56.3** | **2.66** | **34.9** | **21.4** |

`decode 32 full` runs the identical kernels and shapes 10-13 µs (20-25 %) slower cell for cell
(`sk2<2,32,2,2,2,2,3>` 6.4 → 13.1 µs for `o`): that case's step is 304 µs, so it is contention in the
bigger-KV build, not a different geometry.

### T = 128 (`verify 16 sliding`, step 240.32 µs; profile total 251.7 µs)

| cell | kernel | MB | µs | TB/s | wall | excess | TFLOP/s |
|---|---|---|---|---|---|---|---|
| qkv_gate | `rq<128,5,…,1>` (12-bit codec) | 28.3 | 15.6 | 1.81 | 6.6 | **9.0** | 310 |
| o | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT` (cuBLASLt plan) | 16.8 | 8.6 | 1.95 | 3.9 | 4.7 | 250 |
| pr | `nvjet_sm90_tss_80x64_64x11_4x2_h_bz_coopB_TNN` (Lt plan) | 75.5 | 26.6 | 2.84 | 17.6 | 9.0 | 363 |
| unpooler | `rq<128,5,…,1>` | 50.3 | 15.6 | 3.22 | 11.7 | 3.9 | 551 |
| **chain** | | **170.9** | **66.4** | **2.57** | **39.7** | **26.7** | |

The `rq` row is 2 calls / 31.2 µs in the profile (qkv + un); the byte split above is the only assignment
consistent with `RQ_USE = {('un',128), ('qkv',128), ('qkv',256)}` (**[INFERENCE]** on which nvjet is o and
which is pr; both are cuBLASLt and both are ~2-2.8 TB/s either way).

### T = 256 (`verify 32 sliding`, step 307.88 µs; profile total 309.2 µs)

| cell | kernel | MB | µs | TB/s | wall | excess | TFLOP/s |
|---|---|---|---|---|---|---|---|
| qkv_gate | `rq<256,5,1,0,1,0,65603,1>` | 28.3 | 18.1 | 1.56 | 6.6 | **11.5** | 534 |
| o | `nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNN` | 16.8 | 9.5 | 1.77 | 3.9 | 5.6 | 452 |
| pr | `nvjet_sm90_tss_144x64_64x8_1x1_v_bz_TNN` | 75.5 | 25.5 | 2.96 | 17.6 | 7.9 | 758 |
| unpooler | `nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNT` | 67.1 | 32.5 | 2.06 | 15.6 | **16.9** | 529 |
| **chain** | | **187.7** | **85.6** | **2.19** | **43.7** | **41.9** | |

## 2. Attribution of the excess

1. **Grid ramp — the real, fixable cause at T ≤ 32.** The `sk2` kernel owns `N/R` weight rows per CTA and
   nothing else, so its CTA count is `N/R`: qkv **96** (73 % of 132 SMs), unpooler 128, pooler 128, o 256.
   A 96-CTA kernel cannot exceed 3.14 TB/s whatever it does; it measures 2.42, i.e. 77 % of its own ceiling.
   `pr` cannot be raised: `static_assert(!FUSE || (COMP && CL == 1 && UPW == 1 && UPS == 8))` plus the router
   piece map `rt = cta >> 2` over 32 tiles pins the fused grid at exactly 128. That is why `pr`'s 7.6 µs is
   the one excess this ticket does not touch.
2. **Tail.** o's 256 CTAs is 1.94 waves (the second wave is 124 CTAs, then 8 SMs idle); un/pr at one 97 %
   wave end with every CTA's last stage issued with no successor to hide it — and the ring is only 3-5 stages
   deep at these shapes (`_r144` → 3 × 40 KB for the unpooler), so the drain is not prefetched.
3. **Launch/PDL gap.** `bfp_SK2_PDL = 0` at these shapes, so each of the 5 cells starts only after its
   producer's grid retires. The profile's own `GPU busy 145.2 / total 150.0` at T = 32 puts 4.8 µs of
   inter-kernel gap across 7 launches (~0.7 µs each) — and each cell additionally pays its own cold HBM ramp
   (~0.6-1 µs) because its first byte depends on the previous cell's last.
4. **At T ≥ 128 the cells leave the bandwidth regime.** FLOP/byte = T, so the ridge point is ~230 FLOP/byte
   (990 TFLOP/s ÷ 4.3 TB/s): T = 128 is still memory-bound (128 < 230) but T = 256 is compute-bound. The
   534/452/758/529 TFLOP/s measured at T = 256 are 46-77 % of peak, so the 41.9 µs of "excess" there is MMA
   efficiency on a skinny-M GEMM, not ramp — a byte-based remedy (the 12-bit pack) provably cannot collect it.
   The 2.06 TB/s that made `un@256` look like the worst cell is a compute-bound GEMM wearing a byte metric.

## 3. What changed (candidates only)

1. **`bfp_SK2_CONFIGS` + `bfp_SK2_ALT`** — three alternatives, each instantiated next to the incumbent by
   `bfp__sk2_src` (which now emits `PICK ∪ ALT`; previously only the pick was compiled, so no alternative
   could ever have been timed):
   * `qkv32c2` — same rows/CTA, `CL = 2` K-split: 96 → **192 CTAs**, half of K each, *shares the incumbent's
     packed stream* (the stream key ignores `CL`, so this costs no VRAM and no repack). Used at tokr 16 and 32.
   * `un32c2_r144` — same, 128 → **256 CTAs** at tokr 32 (the multi-rank peer-sum path already ships: the
     router config runs `CL = 3` every decode step).
   * `o16h_r100` — 16 rows/CTA instead of 32: 256 → **512 CTAs** at tokr 16/32 (`K = 1024` is only four
     stages, so o's gap is request-in-flight, not ring depth: CUTLASS #1176's "requests-in-flight" diagnosis).
2. **`bfp__sk2_bake`** (new, called from `bfp_sk2_prepare` behind a `try`) — times the pick and its
   alternatives cold-L2 (64 MB read sweep before every replay, alternated order, CUDA-graph replays, 24 reps,
   median), holds an alternative to a **1e-3** output-parity check against the pick (the layer's own budget is
   8.4e-3 of a 0.15 bound), and swaps it in only at **≤ 0.97 × the incumbent and ≥ 0.3 µs**. Prints one line
   per (cell, tokr): `SK2 unpooler t32 unpooler32_r144:18.42us un32c2_r144:17.10us -> un32c2_r144`. Any
   failure (capture, launch, parity) leaves the proven static picks untouched.
3. **`bfp_GEMM_ALT`** — the sibling's alternative-geometry machinery for `pr` extended to the two cells that
   are still on the cuBLASLt plan at m ≥ 128: `o` and `unpooler` at bn 128/256 get `splits = 1` (a one-wave
   64-CTA grid with no cross-CTA reduction) and `nwg = 1` (half-width column tiles, twice the CTAs at the same
   split) as candidates. These need exactly **one** new `gemm90` instantiation: every other geometry in the
   list is already compiled for `pr` (stage counts are harmonised at 6 for bn 128 / 4 for bn 256, which the
   dispatch key requires). The existing `bfp__autotune` then times them per `(layer, m)` from a cold L2, the
   existing `bfp__selftest` holds each to 5e-3 against FP64 (a bad geometry is disabled, never shipped), and
   the existing `_tune` still has to beat cuBLASLt at the chain level for the pick to ship.
4. **`TUNE` print widened** from `pr/pooler/router` to every dense cell at m ≥ 16, so one check now shows the
   candidate times and the cliff shape per cell.
5. `bfp__run_sk2(..., nm=)` — the config name is now a parameter (both the base and the later `_DE` override),
   so the bake can time a specific geometry instead of the pick.

Not touched, deliberately:
* **`rq` for `o@128`/`o@256`/`un@256`** — the file's own sweep already measured this and lost:
  `RQBAKE = {128: [], 256: []}  # wf_jsweep: o m=128 (s4s5hz 13.06 vs r6 11.17) and all m=256 (n256* 30.8-31.9 …)`,
  and `RQLAYERS = ("qkv_gate","unpooler")` never bakes `o`. At T = 256 those cells are compute-bound (§2.4), so
  the 25 % byte saving could not have paid anyway.
* **Sharing one launch between `pr` and `router` at m ≥ 128** — that is exactly what the `bfpp` arm inherited
  from `klr_prsk.py` already does (one FP32 `[T, 4608]` GEMM over pooler ‖ router-hi ‖ router-lo, exactness-
  gated at 1e-4 against FP64 and then A/B'd against the Lt plan at a 2 % margin). Duplicating it here would
  give the same cell a second, competing candidate.

## 4. Price

The score is a weight ratio, so the relative gain of saving Δ on both cases of one load is `62·Δ / Σ w·t`
(Σ w·t = 69,595 µs from the profile's own benchmarks): **0.089 % per µs**, 0.066 % for a sliding case alone.

| shape pair | addressable excess (qkv + o + un) | at ⅓ captured | at ½ captured |
|---|---|---|---|
| T = 16 | 10.4 µs | 0.31 % | 0.46 % |
| T = 32 | 13.9 µs | 0.41 % | 0.62 % |
| T = 128 | 8.6 µs (qkv is `rq`, not addressable) | 0.26 % | 0.38 % |
| T = 256 | 22.5 µs | 0.67 % | 1.00 % |
| **all four** | **55.4 µs** | **+1.6 %** | **+2.4 %** |

Expected: **+0.5 … +1.6 %** (the sk2 alternatives and the `o`/`un` GEMM geometries each land on a subset of
the cells, and the bake may reject all of them — the worst case is byte-identical to the base).

## 5. Gates

* `ast.parse` — OK, 989,496 B (1 MB lint budget: 10.5 KB headroom).
* `okc_loop.cli lint` — 30 tier-0 errors, the **identical set** to `klr_prsk.py`/`klr_qd2.py` after
  line-number normalization (only the `portability` message's embedded line number moved).
* `tools/cu_build.py --variant MID` / `--variant MAIN` — **identical ok/FAIL sets** to `klr_prsk.py`
  (MID 2 ok / 9 FAIL, MAIN 5 ok / 12 FAIL; the FAILs are this container's missing `nv/target`,
  `cuda/std/type_traits`, `tvm/ffi`, `cudaGetLastError`).
* **Colab sm_90a compile gate** (`tools/colab_gate.py` → `colab run --gpu A100`): every buildable extension
  OK — `bfp__get_ext` 57.1 s (compiles the 4 new `sk2` instantiations and the 1 new `gemm90` instantiation;
  the same gate on `klr_prsk.py` gives 49.4 s, so the additions cost **+7.7 s of nvcc**, against a judge
  prepare of 140 s in a 280 s budget), `_att7` 90.0, `_moe` 90.2, `_moe(32)`, `_moe_tiny` 75.7,
  `_moe_tiny2` 75.5, `_moe_mid` 75.5 s wall.
  The 3 `bad` entries are the known tvm_ffi-only modules (`_graft_glue_module`, `_graft_b2_module`,
  `_sl_cluster_module`), unavailable on Colab and present on the judge image — identical to the base file.

Unverifiable offline (no GPU in this container): the baked timings, whether an alternative actually wins, and
the `SK2`/`TUNE` lines. Every one of those fails safe to the incumbent.
---

# Addendum: the rq split-K candidate set, and the CTA counts Main asked for

## A1. The premise check: what grid does `rq` actually launch?

`rq<<<dim3(NB * (HALF ? 2 : 1), KS), ...>>>` (`klr_drate:15994`), `NB = N / 128`, one CTA per SM
(`__launch_bounds__(384, (FL & 3072) ? 2 : 1)` — occupancy 2 only for the H2/F2 variants, which nothing
ships). The shipped variant for the dense cells is `d2s5hze` (`NT=128, ST=5, GS=1, FL=0x3023`): **no HALF,
no H2/F2**, so the grid is exactly `N/128 x KS`.

The judge's profile confirms the instantiation: `rq<128, 5, 1, 0, 0, 0, 42999939, 1>` = `NT=128, ST=5, GS=1,
PRB=0, WM=0, INS=0, FL=0x2902083, MS=1` (bits 0,1,7,13,20,23,25 — no bit 4 HALF, no bit 10/11).

So, for the cells Main named (the number in brackets is what the bake now prints next to each candidate):

| layer | N | incumbent KS | incumbent CTAs | new KS candidates |
|---|---|---|---|---|
| qkv_gate | 2304 | 7 | **126** (18 blocks x 7) | `@4` → 72, `@8` → 144 |
| unpooler | 8192 | 2 | **128** (64 x 2) | `@1` → 64, `@4` → 256 |
| o | 8192 | (never baked) | — | not added, see §4 |

qkv@128 is at **126 / 132 SMs = 95 %**, not 36 CTAs / 27 %: the 36 in that sum is `N/128` without the KS
factor. So the parallelism cliff Main diagnosed for `sk2` (which really did launch 96 CTAs) is *not* what
qkv@128 is losing its 9 µs to. The grid is already one wave; **raising KS cannot add SMs.** What KS still
buys is the other side of the same trade, so it is now a candidate — see §2.

## A2. The split-K reduction: what it costs, and why a high KS is not free

`rq` does **not** use a workspace, a partials tensor, a counter or a last-arriver finisher. The epilogue is a
hardware TMA reduce:

```
cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [to, {cb + 64*g, tok0}], [slab]
```

— each CTA's fp32/warpgroup accumulator tile is `stmatrix`'d into shared and then **added atomically into the
BF16 output tile by the TMA engine** (`klr_drate:15835, 15921`). The only handshake is the `ZI` bit: one warp
zeroes the output tile and sets a shared flag, and the issuer does a bounded spin on it before the first
reduce (trap on timeout), so the first red.add never lands on stale bytes. Nothing is paid at CTA granularity
and no extra launch exists to be captured in the graph.

The cost is therefore pure **output-tile read-modify-write traffic**: `KS x m x N x 2 B` (bf16) on top of the
one zeroing pass over the same tile (`m x N x 2 B`):

| cell | tile | KS | RMW bytes | ≈ µs at 4.3 TB/s | share of the cell |
|---|---|---|---|---|---|
| qkv@128 | 128 x 2304 | 7 | 4.13 MB (+0.59 zero) | 1.1 | **7 %** of 15.6 µs |
| qkv@128 | " | 4 | 2.36 MB | 0.6 | 4 % |
| qkv@128 | " | 8 | 4.72 MB | 1.2 | 8 % |
| un@128 | 128 x 8192 | 2 | 4.19 MB (+2.10) | 1.5 | 9 % of 15.6 µs |
| un@128 | " | 1 | 2.10 MB | 0.7 | 4 % |

So the reduce is ~1-2 µs and it scales with KS: KS=8 buys 18 more CTAs (144) and pays ~0.1 MB more RMW and a
1.14-wave tail; KS=4 gives up 54 CTAs and halves the RMW. Neither dominates a priori, which is exactly what a
bake is for — and this is now considered for qkv@128/256 (KS 4, 8) and un@128 (KS 1, 4), with the existing
`_rqval` exactness gate (`torch.equal` on one-hot rows + rel-L2 ≤ `RQTOL[ks]`, per-KS) and the existing
`MARGIN_RQ128 = 0.98` rule. **Every probe prints its own CTA count**, e.g.
`BAKE qkv_gate m=128 rq:d2s5hze@4[72c] 14.87 (ratio …)`, so the log shows occupancy and time together.

Implementation: a probe name may carry `@KS` (`d2s5hze@4`). KS is a launch argument (it is `gridDim.y`, and
`s0 = ks*KC/KS / KCS` are runtime), so **a KS candidate compiles nothing new** — only its own escape table,
which the bake already builds per KS (`ESC[_kk]`). `_ksof`/`_tag` (`klr_drate:14794`) decode the name and the
grid; `_try` now splits the name, passes `RQTOL[ksv]`, and `names.index(base)` picks the instantiation.

## A3. `pr` at T=128/256: CTA counts of the candidates the bake already times

`pr`'s incumbent is cuBLASLt (`nvjet_sm90_tss_*`, 2.84 TB/s at T=128, 2.96 at T=256) — a library grid, shown
as `lib`. The custom candidates from `bfp_GEMM_ALT` (all considered by `bfp__candidates` at m ≥ 128, all
timed by `bfp__autotune`) at `N = 4608`:

| block | geometry (nwg, splits) | CTAs |
|---|---|---|
| T=128 | (2, 3) default | 36 x 3 = **108** |
| T=128 | (2, 2) / (2, 4) / (1, 1) / (1, 2) | 72 / **144** / 72 / **144** |
| T=256 | (2, 1) default | 36 x 1 = **36** |
| T=256 | (2, 2) / (1, 1) / (1, 2) | 72 / 72 / **144** |

So at T=256 the *default* gemm90 pr candidate is a 36-CTA grid (27 % of the machine) and the sibling's
`pr256w1s2` (144 CTAs) is the entry that fixes it — it was already in the candidate list; what was missing was
that nobody could see the occupancy in the log. The `TUNE` line now prints it:
`TUNE pr m=256 gemm(2,1,4):25.5us/36c gemm(1,2,4):…us/144c torch:…us/lib -> …`.

The same print now covers the other dense cells: `o`/`unpooler` at bn 128/256 are 128 CTAs at the default
`(2,2)` split, 64 at `(2,1)`, 256 at `(1,2)`; `sk2` is `N/R x CL` (qkv 96 → 192 with the CL=2 alt, o 256 →
512, un@32 128 → 256).

## A4. What is deliberately still not a candidate

* **`rq` for `o`** (m 128/256): the file's own sweep measured it and lost —
  `RQBAKE = {128: [], 256: []}  # wf_jsweep: o m=128 (s4s5hz 13.06 vs r6 11.17)` — and `RQLAYERS` never packs
  `o`. Adding the layer would cost a 12.6 MB pack + a full one-hot validation to re-measure a 1.9 µs loss.
* **`rq` for `un@256`**: `RQBAKE[256] = []`, and T=256 is compute-bound (534 TFLOP/s of ~990 at that cell ⇒
  FLOP/byte 256 > the 230 ridge), so a 25 % byte saving cannot pay.
* **qkv@128's real remaining lever** is 2 CTAs/SM (FL bit 10/11 = H2/F2). The kernel's own comment says F2 caps
  registers at 84 while the math warps want 104 — i.e. the occupancy variant needs its own decode flavour, a
  new instantiation that cannot be validated without a Hopper GPU. Not shipped blind; the KS bake above is the
  part that is free.

## A5. Gates (re-run after this change)

* `ast.parse` — OK; 991,109 B (1 MB budget, 8.9 KB headroom).
* `okc_loop.cli lint` — the **identical** 30-error set to `klr_prsk.py` (only the embedded line number in the
  `portability` message moved).
* `tools/cu_build.py --variant MID` / `MAIN` — **identical** ok/FAIL sets to `klr_prsk.py` (MID 2 ok / 9
  FAIL, MAIN 7 ok / 9 FAIL).
* **Colab sm_90a gate re-run** (`colab run --gpu A100`, 88.8 s wall): all buildable extensions OK —
  `bfp__get_ext` 55.0 s, `_att7` 88.8, `_moe` 88.3, `_moe(32)`, `_moe_tiny` 75.5, `_moe_tiny2` 75.2,
  `_moe_mid` 74.7 s; the 3 `bad` are the known tvm_ffi-only modules (`_graft_glue_module`, `_graft_b2_module`,
  `_sl_cluster_module`), identical to the base. **No CUDA source changed** by this addendum (KS is a launch
  argument, the probes reuse `RQPROD` instantiations), so the compile time is unchanged within nvcc noise
  (55.0 / 57.1 s against 49.4 s for the base file, whose 4 sk2 + 1 gemm90 instantiations are the real delta).

## A6. Revised price

The KS candidates only touch qkv@128/256 and un@128; the reduce analysis caps their upside at the ramp/ROI
above, and the bake may reject all of them. That leaves the expected band from the first report unchanged
(**+0.5 … +1.6 %**, ceiling +2.4 %) with a small upward bias at T=128 (qkv's 9 µs excess is now probed from
three grid shapes instead of one). If `rq`'s 42 % rate at qkv@128 turns out to be the reduce rather than the
ramp, KS=4 (~0.6 µs of RMW saved, 54 CTAs given up) is the shape that would show it.
