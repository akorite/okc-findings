# bfp_geo: does a bigger grid pay on the five dense cells? (muse_bf16_proj as the layer's oracle)

Candidate `src/okc_loop/vendored_work/bf16/geo.py`, sha256 `be1b0c8d482e0a8180ba79e2f379861fbd5e24206b0ff87e8c6506b684280001`
(653,734 B), base `src/okc_loop/vendored/keell0renz/muse_bf16_proj.py` (643,105 B, sha256 `ba42c825…`,
the file the ticket's 1.5891 row was cut from). Nothing in the base was removed or replaced: `_TABLE`
still routes every cell the new geometry does not win, and the new candidates are only timed at
prepare() behind the v6 rule (cold-L2 graph replays, `V6_GAIN = 0.97`, then `_v6_ok`'s rel-L2 /
probe-row gate before a pick is taken).

*Why this board:* it times the same five matrices as the dense chain (qkv_gate [8192,2304],
o [1024,8192], pooler [4096,4096], unpooler [8192,4096], router [256,4096] FP32) at the same
m = 1/8/16/32/128/256, one cell per load. A cell's own ramp is therefore measured without the layer's
serialization: a cleaner probe of geometry, a weaker probe of the chain.

**Verdict: no geometry won. All 20 probed cells kept their `_TABLE` route** -- the incumbent grids are
already at or past the optimum, and every slack direction (more CTAs, half the rows per CTA, an nwg
sweep, the cuBLASLt pin on a second cell) measured slower.

## 1. The file's own tuners, their candidate sets, and the CTAs each pick launches

Two tuners already exist and both are kept:

1. **`_autotune`** (vjeux: the `m <= 32` cells and the router). `TUNE_M = (1,16,32,8,128,256)`,
   24 cold-L2 graph replays per candidate, keeps the minimum per `(layer, m)` in `state["choice"]`.
   Candidate set `_candidates`: the static route (`sk2` / `skinny` / `wide` / `torch`), `TMA_ALT`,
   `gemm` (router only), `gq` / `gqx` (33..256), `rk` (router), `rgemv` / `sgemv` / `cgemv` (m = 1),
   `torchT`, `cutlass`, `torch`.
2. **chenwainuo's own bakes** (`_rqbake` at m = 128/256, `_rqsbake` at m = 16/32): probe lists of up
   to 6 timed paths per `(layer, m)`, each carrying a K-split (`t32s6hzw-k7` = variant + KS), gated by
   `MARGIN_RQ128 = 0.98` / `MARGIN_RQ32 = 0.97` against the proven pick and by `_rqval` (one-hot rows
   bit-exact). `RQKS = {qkv 7, pooler 4, un 2, o 2}` sets the grid.

Grids: sk2 = `N/R x CL` (R = NWR·NT·8), rq = `N/128 x KS`, gemm90 = `N/(64 nwg) x splits`, `lib` =
library grid (unobservable).

| cell | m=8 | m=16 | m=32 | m=128 | m=256 |
|---|---|---|---|---|---|
| qkv_gate | r6 | **sk2 96** | r6 | **rq 126** (18x7) | cuBLASLt `lib` |
| o | r6 | r6 | sk2 256 | nb `lib` | cuBLASLt `lib` |
| pooler | r6 | r6 | **sk2 128** | **rq 128** (32x4) | cuBLASLt `lib` |
| unpooler | r6 (slice, n=1024) | sk2 256 | **sk2 128** | **rq 128** (64x2) | cuBLASLt `lib` |
| router | nb 16-CTA cluster | nb | nb | vjeux `rk`/`wide` | vjeux `rk` |

**Below ~132 CTAs:** qkv at **every** m (96 at 16, 126 at 128 -- its sk2 grid is `2304/24 = 96` by
construction), pooler at 32 and 128 (128), unpooler at 32 and 128 (128), and the router's 16-CTA
`routertail` cluster at 8/16/32. Everything else is a 256-CTA single wave or a library grid. The
layer-side work saw the same numbers from the other end (`sk2<1,16,…>` = 96 CTAs at T = 16,
`rq<128,5,…>` = 126 at T = 128).

## 2. What was added (additive)

1. **`SK2_ALT`, 5 sk2 configs on half the rows per CTA** (`qkv16m2_r200`, `qkv32m2_r208`,
   `pooler32m2_r144`, `un32m2_r144`, `o32m2_r100`): the pick's own K/ring shape at R = 16 or 32 instead
   of 24/32/64, i.e. **qkv 96 -> 144, pooler 128 -> 256, unpooler 128 -> 256, o 256 -> 512** (the o
   move is the layer's own `o16h_r100` "16 rows/CTA instead of 32", and doubles as the
   already-full control). `_sk2_src` now emits `PICK ∪ ALT`.
   *Why not the layer's `CL = 2` cluster split:* `sk2_kernel`'s `CL > 1` path is exercised in this
   snapshot only by the router config (`CL = 3`, timed at m = 1); every CL = 2 config would be a
   first launch at its cell, and a first launch that wedges costs a whole timed run. The row-block
   probe has no cross-CTA handshake at all, so a wrong pick can only be slow or inexact -- both of
   which the gate catches. (The layer's own bake covers CL = 2 on its quota.)
2. **`GEO_GEMM`: gemm90 re-timed at `(nwg, splits)` in `{(1,1),(1,2),(1,4),(2,1),(2,2),(2,4)}`** on
   the cell's own bn, **filtered to grids that are co-resident** (`(N/(64 nwg))·splits <= nsm`).
   That filter is not cosmetic: `gemm90_kernel` reduces its K splits with an unbounded in-kernel spin
   on a per-row-block counter, and the table's own comment states the invariant
   (`grid = (N/(64 nwg), splits) <= 132 (co-resident)`). A 256- or 512-CTA probe deadlocks the device
   -- **that is exactly what the first attempt did** (`okc check 3d888c0e68af4dfe` stalled ~8 min in
   prepare with 144-CTA `gemm(1,4)` probes; see §5). `splits` is a launch argument, so only `nwg = 1`
   compiles: **2 new instantiations** (`<1,128,6>`, `<1,256,4>`). `_run_gemm(geo=)` and the
   `mid_ws_shared` sizing were extended to match (`ws` need = `(N/(64 nwg))·splits·(bn/8)·nwg·128`
   floats, the launcher's own index).
3. **`_geo_routes`** (runs after the v6 route build): per scored `(cell, m)` it times `base` + the
   geometry candidates + `ag` where a cuBLASLt pin exists, prints one `GEO|` line per cell with every
   candidate's time and CTA count, and adopts a winner only at `< 0.97 x base` that also passes
   `_v6_ok` (plus a 3x-rep paired re-time when the margin is under 10 %). The pick lands in
   `state["routes"]`, which `kernel_fn` now consults **before** `_TABLE`; a cell the tuner does not win
   keeps its `_TABLE` route and any exception is caught per cell.
4. **`bfp__get_ext`**, a module-level compile hook for `tools/colab_gate.py`. The dense builder lived
   only inside the vjeux factory, so the gate had **nothing to compile** for this file
   (`bfp__get_ext` came back "absent"); now the gate builds the whole dense translation unit on an
   A100, which is where the 5 new sk2 and 2 new gemm90 instantiations get checked.

Deliberately not touched: m = 1 (scalar GEMV -- no CTA geometry to move), m = 8 (its sk2 configs are
not routed anywhere in this snapshot, so every launch there is a first launch; 16/32 already answer
the small-m ramp), the router (1.5 % of the layer score, and its 16-CTA cluster is a deliberately
different kernel), and the CL = 2 cluster path (see §2.1).

## 3. Gates

* `ast.parse` -- OK, 653,734 B; line 1 is `# okc: problem=muse_bf16_proj`.
* `okc_loop.cli lint` -- the **identical 83-error set** to the vendored file after normalising line
  numbers (the pre-existing `time.time` / `os.environ` / `threading` sandbox errors): **delta = none**.
* `tools/cu_build.py --variant MAIN` / `MID` -- both the base and the candidate report
  `no embedded CUDA source found`: this file generates its CUDA inside the factory, so that gate is a
  no-op for it (identical result, no delta). The Colab gate is the real compile check.
* **Colab sm_90a compile gate** (`colab run --gpu A100`, 57.9 s wall incl. VM): `bfp__get_ext`
  **ok in 35.6 s** -- that build is the whole dense TU (skinny + gemm90 + wide + sk2 + gq + gqx +
  rgemv + rk), i.e. all 20 `launch_sk2` instantiations (5 new) and both new `launch_gemm90<1,…>`.
  The only `bad` entry is `_moe(32)`, an artefact of the gate's fixed name list (this problem has no
  `_moe`); every other name is `absent`, exactly as on the base. (Re-run after the §2.2 fix; the
  first version's 7 CL = 2 instantiations also compiled, 37.6 s, but were dropped.)

## 4. The check: `okc check` job c8c64c72c95a4ae9 -- **passed, 1.5964x / 3026.0 µs, 9/9 tests**

Board row at ingest time: **1.5605 (3095.7 µs)**, sha `4ab25a7deaef` -- not the 1.5891 the ticket
quoted, so the board moved under us. Against that row every load improved
(`loads improved 6/6`): decode 1 1.7351 → 1.7452 (+0.58 %), decode 16 1.6043 → 1.6122 (+0.49 %),
decode 32 1.5203 → 1.5361 (+1.04 %), verify 1 2.0892 → 2.0985 (+0.44 %), **verify 16 1.3116 → 1.4414
(+9.90 %)**, verify 32 1.2452 → 1.2664 (+1.70 %). Against the ticket's 1.5891 / 3040.0 µs the same
measurement is **+0.43 %, inside the run's own ±1.5 % band**.

The +9.90 % at verify 16 is *not* ours: the tuner adopted nothing, so every scored cell runs the
route it ran in the base file, and the base snapshot (`ba42c825`) simply differs from the current
board row at that cell. Read the check as *behaviour-identical at the scored cells*; its value is the
oracle below, not a new row. `okc submit` was not run (the ticket reserves that for the lead);
`ingest_check`'s gate refuses only on Tier-0 (`tier0=FAIL`, our lint being stricter than the judge's
screen -- identical to the base file), not on speed.

### The measured geometry, every candidate at every probed cell

Times are µs, cold-L2 CUDA-graph replays, median of 14, on the judge's H200 (132 SMs). `[Nc]` is the
candidate's launch grid.

```
GEO|qkv_gate|16|base 13.54[96c] sk2@qkv16t_r200 13.44[96c] sk2@qkv16m2_r200 18.21[144c] -> none
GEO|o|16|base 9.50[r6] sk2@o16t_r68 9.57[256c] -> none
GEO|pooler|16|base 17.92[r6] sk2@pooler16t_r192 19.65[128c] -> none
GEO|unpooler|16|base 17.92[256c] sk2@unpooler16t_r72 18.62[256c] -> none
GEO|router|16|base 8.38[lib] sk2@router 8.83[96c] -> none
GEO|qkv_gate|32|base 13.86[126c] sk2@qkv32_r208 14.94[96c] sk2@qkv32m2_r208 19.74[144c] -> none
GEO|o|32|base 9.76[256c] sk2@o32h_r100 9.73[256c] sk2@o32m2_r100 12.19[512c] -> none
GEO|pooler|32|base 19.14[128c] sk2@pooler32_r144 19.07[128c] sk2@pooler32m2_r144 28.29[256c] -> none
GEO|unpooler|32|base 19.01[128c] sk2@unpooler32_r144 19.01[128c] sk2@un32m2_r144 20.38[256c] -> none
GEO|router|32|base 9.15[lib] sk2@router 9.15[96c] -> none
GEO|qkv_gate|128|base 15.97[126c] gemm(1,1) 50.50[36c] gemm(1,2) 33.89[72c] gemm(2,1) 61.98[18c] gemm(2,2) 40.06[36c] gemm(2,4) 24.77[72c] -> none
GEO|o|128|base 10.82[lib] gemm(1,1) 12.80[128c] gemm(2,1) 14.40[64c] gemm(2,2) 18.50[128c] -> none
GEO|pooler|128|base 20.96[128c] gemm(1,1) 50.62[64c] gemm(1,2) 34.98[128c] gemm(2,1) 61.57[32c] gemm(2,2) 40.70[64c] gemm(2,4) 29.38[128c] ag 24.51[lib] -> none
GEO|unpooler|128|base 22.21[128c] gemm(1,1) 29.54[128c] gemm(2,1) 35.52[64c] gemm(2,2) 31.10[128c] -> none
GEO|router|128|base 9.79[lib] -> none
GEO|qkv_gate|256|base 21.66[lib] gemm(1,1) 50.53[36c] gemm(1,2) 39.26[72c] gemm(2,1) 86.40[18c] gemm(2,2) 57.50[36c] gemm(2,4) 34.37[72c] -> none
GEO|o|256|base 12.77[lib] gemm(1,1) 14.56[128c] gemm(2,1) 19.65[64c] gemm(2,2) 26.34[128c] -> none
GEO|pooler|256|base 28.35[lib] gemm(1,1) 50.40[64c] gemm(1,2) 41.60[128c] gemm(2,1) 84.45[32c] gemm(2,2) 57.60[64c] gemm(2,4) 36.58[128c] -> none
GEO|unpooler|256|base 27.90[lib] gemm(1,1) 32.42[128c] gemm(2,1) 48.10[64c] gemm(2,2) 41.60[128c] -> none
GEO|router|256|base 11.07[lib] -> none
GEO|DONE|20 cells timed, 0 routes, 0.2 s
```

Noise floor from the A/A pairs (the same kernel timed twice in one line): qkv@16 13.54 vs 13.44
(0.7 %), unpooler@32 19.01 vs 19.01, o@32 9.76 vs 9.73, pooler@32 19.14 vs 19.07, unpooler@16 17.92 vs
18.62 (3.9 % at that cell). So a 3-5 % win is at the edge of readability here and nothing came close.

### What moved where

* **Raising CTAs past ~128 costs, monotonically.** Seconds per cell above a wave: qkv@16 96 -> 144
  CTAs = **+35 %** (13.44 -> 18.21); qkv@32 96 -> 144 = +32 %; pooler@32 128 -> 256 = **+48 %**
  (19.07 -> 28.29); un@32 128 -> 256 = +7 % (19.01 -> 20.38); o@32 256 -> 512 = **+25 %** (9.73 ->
  12.19). The 512-CTA control is the clearest: doubling a grid that is already two waves straight
  doubles the per-CTA cold-start of the weight ring, and nothing hides it.
* **Within one kernel the ramp is real and saturates near a wave.** gemm90 at qkv@128:
  18c 61.98 -> 36c 40.06 -> 72c 24.77 (each doubling -35 / -38 %); pooler@256: 32c 84.45 -> 64c 57.60
  -> 128c 36.58; unpooler@256: 64c 48.10 -> 128c 41.60. It is starved *below* ~64 CTAs and flat above.
* **96-CTA qkv is not occupancy-starved, it is per-CTA-startup-bound.** Halving its rows per CTA
  doubles the CTAs and keeps the same total weight bytes, yet loses 35 %: the fixed cost is each
  CTA's ring fill (the layer report's "grid ramp" is that latency, not an idle SM count).
* **The cuBLASLt pin is already the right kernel at m = 256.** Every custom gemm90 geometry loses to
  it (qkv 34.37 vs 21.66, o 14.56 vs 12.77, pooler 36.58 vs 28.35, unpooler 32.42 vs 27.90) and at
  m = 128 the rq picks beat every gemm90 geometry (qkv 15.97 vs 24.77 best, pooler 20.96 vs 29.38,
  unpooler 22.21 vs 29.54). `ag` on a cell it was not routed to (pooler@128, 24.51) loses to rq too.
* **`CL = 2` was not measured.** That is the one candidate class of the ticket's list this run could
  not settle: its first launch anywhere would be at these cells, and the cost of a wedge is a whole
  timed run. Given the measured slope above (~+35 % per doubling past one wave), a 2× cluster split
  has to buy its extra CTAs from the same per-CTA startup budget, so the prior is now against it in
  this file as well -- but it stays a measurement.

## 5. One-line expectation per cell (what the next candidate should assume)

| cell | expectation |
|---|---|
| qkv@1 | GEMV; no geometry to move (1.609x vs SGLang already). |
| qkv@8 | 96-CTA sk2 is not routable here (chenwainuo's r6 wins today); do not spend a probe on more CTAs. |
| qkv@16 | 96 CTAs is the optimum, not a limit: +48 CTAs cost 35 %. Fix the ring's start latency, not the grid. |
| qkv@32 | rq at 126 CTAs (13.86) already beats sk2 at 96 (14.94) and at 144 (19.74); geometry is dead here. |
| qkv@128 | rq 126c (15.97) > every gemm90 grid 18..72c (61.98..24.77); the ramp inside gemm90 is 18 -> 64 CTAs, above which it is flat. |
| qkv@256 | cuBLASLt 21.66 beats the best custom 72-CTA grid 34.37 by 37 %; a custom kernel needs a better tile, not more CTAs. |
| o@1 | GEMV (1.334x). |
| o@8 | r6 at 9.2 µs; sk2 256c is 9.57 -- a wash, not a target. |
| o@16 | r6 9.50 vs sk2 256c 9.57: r6 stays; no CTA lever. |
| o@32 | sk2 256c is the optimum; 512c costs 25 %. |
| o@128 | nb (`lib`) 10.82 beats every gemm90 grid (12.80..18.50); do not swap the plan for a custom grid. |
| o@256 | cuBLASLt 12.77 vs best custom 14.56 -- keep the pin. |
| pooler@1 | GEMV (1.632x). |
| pooler@8 | r6; sk2 (256c) not entered. |
| pooler@16 | r6 17.92 beats sk2 128c 19.65 by 9.7 % -- the only cell where an sk2 cross-check was clearly wrong. |
| pooler@32 | 128 CTAs is the optimum (rq 19.14 ~ sk2 19.07); 256c costs 48 %. |
| pooler@128 | rq 128c 20.96 > gemm90 29.38 > ag pin 24.51; keep rq, the pin is 17 % worse here. |
| pooler@256 | cuBLASLt 28.35 vs best custom 36.58 -- keep the pin. |
| unpooler@1 | GEMV/sgemv slice (2.021x). |
| unpooler@8 | the 1024-row slice; not probed (first launch). |
| unpooler@16 | 256c sk2 18.62 vs base 17.92 (A/A noise 3.9 % at this cell) -- r6 holds. |
| unpooler@32 | 128c is the optimum; 256c +7 %. |
| unpooler@128 | rq 128c 22.21 vs gemm90 29.54 -- keep rq. |
| unpooler@256 | cuBLASLt 27.90 vs best custom 32.42 -- keep the pin. |
| router@* | not a geometry target (rk/wide/16-CTA cluster already tuned; sk2 96c ties at 32, loses at 16). |

Bottom line for the layer side (where the same five cells are 32 % of the score): **the dense cells are
not CTA-starved above ~one wave and no amount of extra concurrency pays at these shapes**; the
addressable loss is per-CTA ring start (`qkv` at 96 CTAs is 35 % slower the moment you halve its rows,
i.e. the cost is startup, not idle SMs) and MMA efficiency at m = 256 (custom grids lose 14-37 % to
nvjet there). The one lever this oracle could not settle is the `CL = 2` cluster split, whose prior
the measured slope now puts firmly on the wrong side.

## 6. Failure record (for the next attempt)

`okc check 3d888c0e68af4dfe` (the first version, sha `02f1ad5f…` series) passed all gates and then
stalled ~8 min inside prepare: the geo build launched `gemm(1,4)` / `(2,4)` probes of 144-512 CTAs,
and `gemm90_kernel`'s split reduction is an unbounded spin that needs the whole grid co-resident
(the table's own `<= 132` comment). Fixed by filtering every gemm probe through
`(N/(64 nwg))·splits <= nsm` (`state["vjeux"]["nsm"]`), which is now in the file. That run also
showed the geo lines and the v6 lines are only printed at the end of prepare, so a stall there is
invisible in the log -- one more reason the clamp is not optional.
