# bf16 cells: what `geo.py` dispatches on, per scored cell

Sources: `problems/muse_bf16_proj/reference.py` + `TASK.md` (30 scored shapes, 6 loads of 5 cells,
shape weight 62, floors), `src/okc_loop/vendored_work/bf16/geo.py` (`_TABLE`, `_geo_routes`,
`_v6_cands`, `_v6_routes`), and the measured `GEO|` lines in `artifacts/reports/bfp_geo.md`
(judge H200, cold-L2 CUDA-graph replays, median of 14). µs are graph-replay µs, i.e. they include
the ~4 µs replay fixed cost the judge subtracts from both sides; only their ratios are meaningful.

## Answer: yes, `geo.py` already dispatches per cell -- three layers of it

1. **`_TABLE`: 30 explicit per-(layer, m) kernel-family picks**, one entry per scored cell, measured
   from judge profiles (team keell0renz). This is not a global pick and not a `m`-only pick: the
   same `m` routes to different families per layer (`o:128` -> nb, `qkv_gate:128` -> chenwainuo).
2. **`_geo_routes`, per-cell geometry retime at 20 of the 30 cells** (m = 16/32/128/256 x 5 layers):
   times `base` + sk2-half-rows (m <= 32) + gemm90 nwg/split grids (m > 64, non-router) + the
   cuBLASLt pin when one exists, adopts at `< 0.97 x base` + `_v6_ok`. **Result: 0 of 20 changed.**
3. **`_v6_routes`, per-cell routes for the 10 unscored m's 2..384** (pins, rq plans, `tT` cuBLAS on
   the K-major weight, `extra_plan` cuBLASLt): it skips every `_TABLE` key, so it never touches a
   scored cell.

So the table is per cell and measured. The gap is **coverage of the candidate set at the scored
cells**, not the granularity: `_geo_routes` builds its own list (`base`/`sk2`/`gemm`/`ag`), so the
four other ways to run a cell that the *same file* already carries -- next-bucket `pin`, `rq` plan of
a larger m, `tT` (cuBLAS on the K-major weight), `extra_plan` cuBLASLt -- have never been timed at a
scored m. That is what `bf16_selftune.py` adds.

## The 30 cells: family, `geo`'s final route, measured time, floor

`route` = what `geo.py` serves after both passes (`_TABLE` pick unless the tuner won, and none did).
µs from the `GEO|` lines; `-` = never retimed (m = 1/8 are outside `GEO_M`). Floor from `TASK.md`.

| layer | m | `_TABLE` | route / pick | CTAs | geo µs | floor µs | candidates the selftune adds |
|---|---|---|---|---|---|---|---|
| qkv_gate | 1 | vjeux | vjeux GEMV | lib | - | 7.9 | `def` |
| o | 1 | vjeux | vjeux GEMV | lib | - | 3.5 | `def` |
| pooler | 1 | vjeux | vjeux GEMV | lib | - | 14.0 | `def` |
| unpooler | 1 | vjeux | vjeux sgemv slice (n 1024) | lib | - | 1.7 | `def` |
| router | 1 | vjeux | vjeux rgemv | lib | - | 1.8 | (none: `run3` first launch) |
| qkv_gate | 8 | chenwainuo | r6 | r6 | - | 7.9 | `def` (= same) |
| o | 8 | chenwainuo | r6 | r6 | - | 3.5 | `def` (= same) |
| pooler | 8 | chenwainuo | r6 | r6 | - | 14.0 | `def` (= same) |
| unpooler | 8 | chenwainuo | r6 (n 1024 slice) | r6 | - | 1.8 | `def` (= same) |
| router | 8 | nb | routertail small, 16-CTA cluster | 16c | - | 1.8 | (none) |
| qkv_gate | 16 | vjeux | sk2 | 96c | 13.54 | 7.9 | def, rq32/128/256, tT, sk2 x2 |
| o | 16 | chenwainuo | r6 | r6 | 9.50 | 3.6 | def(=), rq32/128/256, tT, sk2 x2 |
| pooler | 16 | chenwainuo | r6 | r6 | 17.92 | 14.1 | def(=), rq32/128/256, tT, sk2 x2 |
| unpooler | 16 | vjeux | sk2 | 256c | 17.92 | 14.1 | def, rq32/128/256, tT, sk2 x2 |
| router | 16 | nb | routertail small | 16c | 8.38 | 1.8 | rq32/128/256, sk2 x2 |
| qkv_gate | 32 | chenwainuo | rq (KS 7) | 126c | 13.86 | 8.0 | def(=), rq128/256, ag, tT, sk2 x2 |
| o | 32 | vjeux | sk2 | 256c | 9.76 | 3.6 | def, nbol, rq128/256, ag, tT, sk2 x2 |
| pooler | 32 | vjeux | sk2 | 128c | 19.14 | 14.1 | def, rq128/256, ag, tT, sk2 x2 |
| unpooler | 32 | vjeux | sk2 | 128c | 19.01 | 14.1 | def, rq128/256, ag, tT, sk2 x2 |
| router | 32 | nb | routertail small | 16c | 9.15 | 1.9 | rq128/256, sk2 x2 |
| qkv_gate | 128 | chenwainuo | rq (KS 7) | 126c | 15.97 | 8.4 | def(=), rq256, ag, tT, gemm x5 |
| o | 128 | nb | ol_projection | lib | 10.82 | 4.0 | def, rq256, ag, tT, gemm x3 |
| pooler | 128 | chenwainuo | rq (KS 4) | 128c | 20.96 | 14.6 | def(=), rq256, ag(pin), tT, gemm x5 |
| unpooler | 128 | chenwainuo | rq (KS 2) | 128c | 22.21 | 14.6 | def(=), rq256, ag, tT, gemm x3 |
| router | 128 | vjeux | rk | lib | 9.79 | 2.2 | def, rq256 |
| qkv_gate | 256 | ag | cuBLASLt pin | lib | 21.66 | 9.8 | def, tT, gemm x5 |
| o | 256 | ag | cuBLASLt pin | lib | 12.77 | 4.5 | def, nbol, tT, gemm x3 |
| pooler | 256 | ag | cuBLASLt pin | lib | 28.35 | 17.4 | def, tT, gemm x5 |
| unpooler | 256 | ag | cuBLASLt pin | lib | 27.90 | 17.4 | def, tT, gemm x3 |
| router | 256 | vjeux | rk/wide | lib | 11.07 | 2.7 | def |

`def` = chenwainuo's own `kernel_fn` (= the rq pick where its prepare baked one, else r6); `(=` means
it calls the same kernel as the incumbent. `ag(pin)` = the pooler@128 cuBLASLt pin; `nbol` = nb's
ol_projection, valid at any m with a guarded store. `sk2 x2` = the pick's config + its `SK2_ALT`
half-rows twin; `gemm xN` = the co-resident `GEO_GEMM` grids only.

## The load arithmetic (why a per-cell µs matters so little, and where it matters most)

Each load sums its 5 cells x weight 62, and the score is the geomean over the 6 loads. From the
measured cells: m=16 load = 67.26 µs, m=32 = 70.92, m=128 = 79.75, m=256 = 101.75 (sums of the `geo
µs` column). One cell 1.5 µs faster is therefore:

| load | sum µs | load gain | board gain (^(1/6)) |
|---|---|---|---|
| decode 16 | 67.26 | +2.23% | +0.37% |
| decode 32 | 70.92 | +2.11% | +0.35% |
| verify 16 | 79.75 | +1.88% | +0.31% |
| verify 32 | 101.75 | +1.47% | +0.24% |

48 ranks per +1% => **one adopted 1.5 µs swap on one cell = 12-18 ranks**. That is the price of the
1.5 µs floor in the selftune's threshold, and it is also why the axis is worth a check.

## Not covered / not known

* **m = 1 and m = 8 have no measured geo-cell time at all** (`GEO_M = (16, 32, 128, 256)`): the only
  per-cell numbers for those 10 cells are the incumbent profile in `bf16_mechanism_scan.md` (wall µs:
  m1 qkv/o/pooler/unpooler/router = 9.05/5.10/13.55/3.75/3.78). The selftune covers them with `def`
  only, because no other route is legal there without a first launch.
* **Variants in other bf16 files that geo does not carry** (rohan's `_ALT_ROUTES` cells pooler16 /
  unpooler16 / router32, `port_rohan.py:4355`, and `V9..V18_SELECTED_ROUTES`): they are rohan-family
  kernels, so adopting them means new kernels, which the ticket forbids. Every *mechanism* in those
  files is already in geo per `bf16_mechanism_scan.md` ("Unadopted mechanisms >1.2%: none found").
* The `CL = 2` cluster split is the one candidate class `bfp_geo.md` could not settle; it is still
  unmeasured and still out of scope here (first launch at every scored cell).
