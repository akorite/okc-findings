# The T<=8 boundary/route group: profiled on the A100 — verdict: closed, no rewrite

Ticket: census + rewrite of the T<=8 graft chain. Base file
`src/okc_loop/vendored_work/layer/klr_merged2.py` (999,545 B, judge cap 1,000,000) is **untouched**:
net byte change **0**, headroom still 455 B. Nothing was shipped because nothing in the group is
dependency-bound in a removable way — the numbers below are the evidence, and §2 corrects which
kernels the T<=8 step actually launches.

## 0. Method (and its limits)

`artifacts/bench/g8_census.py` (launch table), `g8_decomp.py` (dependency split), `g8_ident.py`
(bit-identity matrix). Each kernel is **extracted by AST from the candidate itself** and launched with
the candidate's own tensors/grid/num_warps; one CUDA graph holds K=32 consecutive launches of one
kernel, 40 rounds, 64 MB L2 flush before every replay, variants interleaved, median reported.

Two corrections this channel forces:

* **PDL cannot compile on the A100.** `griddepcontrol.wait/launch_dependents` need `.target sm_90`; on
  the A100 (sm_80) ptxas refuses (`artifacts/bench/g8_census_a100.log`). Every variant here therefore
  runs with the two PDL hints stripped *identically*, so A/B is honest but absolute µs include a launch
  latency the judge's PDL-normalised step hides. **Only ratios within one session transfer.**
* **The harness launch floor is 1.63-1.82 µs** (a `nop` at the same grid/num_warps, same graph, same
  flush). It is T-independent and grid-independent over 2..104 CTAs. Every µs below is reported both
  raw and as `device = ship − nop`.
* A/A drift inside one session is **+0.9 % (T=8) / +1.9 % (T=1)** — but the *same script's* absolute µs
  moved **40 %** between two sessions (`_sl_tile_route` T=1: 4.96 µs vs 3.42 µs). Cross-session
  absolutes are not used anywhere in this report.
* A100-40GB: 108 SMs, 1.41 TB/s, per-SM stream share 18.3 GB/s (the figure the earlier group census
  calibrated). Judge H200: 132 SMs @ 4.8 TB/s, i.e. a 2.0x larger per-SM share (36.4 GB/s).
* **These µs are a ranking instrument, not the step's µs.** The ticket's 4.4 µs (d1s) / 5.7 (v1s) are
  H200 *in-step* figures with PDL engaged; the A100 table below carries a 1.66 µs launch floor the step
  hides, and the same kernel measured 4.96 vs 3.42 µs in two different A100 sessions. Sanity check that
  the two scales agree: the byte floors here are 38-58 % of the per-SM wall and latency-exposed at 4-8
  warps, so a 2x-wider per-SM share buys ~1.0-1.3x (not 2x) -> 2.2-4.0 µs of A100 floor is ~1.5-3 µs
  of H200 time for a 2-3 kernel chain, consistent with 4.4 µs in-step.

## 1. Which kernels the T<=8 step actually runs (the ticket's list is partly from another chain)

Wrapper order is `_after_attention` @14591 (b1x) -> @14471 (GLE) -> @13658 (sl_tile) -> @13453
(sl_cluster) -> @12817 (graft) -> base. Result:

| T | chain | kernels |
| --- | --- | --- |
| 1 | GLE (`GLE_T={1:True}`, `GLE_RC={1:8}`, `GLE_FRT=True`) | `gle_o` (CUDA) -> `gle_pr` (CUDA, fused pool+FP32 router) -> **`_gle_route` `(1,1+1+8)`** -> MoE -> unpool -> **`_sl_cluster_finish` `(1,32)`** |
| 8 | `_sl_tile_after_attention` (`SL_TILE_CONFIG[8]=(4,4,True)`) | `b1` = `glue_cluster_boundary` (CUDA/cluster) -> sk2 pooler + FP32 router GEMV (CUDA) -> **`_sl_tile_route` `(8,5)`** -> MoE -> unpool -> **`_sl_cluster_finish` `(8,8)`** |

`_graft_row_boundary` / `_graft_row_boundary_pre` are **not** in this chain: `_graft_boundary` @12724
launches them only for `16 < T <= 32` and `T > 32` (they are also Gluon + 1 CTA/row + two chained rms,
i.e. exactly the `_boundary1`/`_boundary2` class the ticket already falsified). `glue_cluster_boundary`
is CUDA (`cg::cluster_group`, `__launch_bounds__(128)`) — not runnable and not touchable.

So the T<=8 Triton surface is three kernels, and at each scored shape only two of them run.

## 2. Census (A100, one session, interleaved, cold L2; SPLIT=True = sliding layer)

`nop` = launch floor at the same grid; `floor` = a hand-written mirror moving the identical bytes over
the identical grid/warps with every reduction removed; `dep = ship − floor`. Bytes are from the launch
arguments (`g_preexp`/`pre2`/`post2` are bf16, `bias`/`alpha*`/`beta*`/`carry`/`g_preexp*` as shipped).

| kernel | T | grid | nw | ship µs | nop | floor | **dep** | bytes (all CTAs) | worst CTA | µs/KB/CTA | % of per-SM wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `_sl_tile_route` | 1 | (1,2) | 4 | 4.96 | 1.63 | 3.90 | **1.06** | 25.7 K | 24.3 K | 2.27/24.3 | **58 %** |
| `_sl_tile_route` | 2 | (2,2) | 4 | 4.93 | 1.63 | 4.10 | **0.83** | 51 K | 24.3 K | 2.47/24.3 | 55 % |
| `_sl_tile_route` | 4 | (4,5) | 4 | 5.02 | 1.66 | 3.62 | **1.41** | 234 K | 14.3 K | 1.96/14.3 | 41 % |
| `_sl_tile_route` | 8 | (8,5) | 4 | 4.99 | 1.66 | 3.58 | **1.41** | 468 K | 14.3 K | 1.92/14.3 | 41 % |
| `_sl_cluster_finish` | 1 | (1,32) | 8 | 4.67 | 1.70 | 3.94 | **0.74** | 492 K | 15.4 K | 2.24/15.4 | 38 % |
| `_sl_cluster_finish` | 2 | (2,32) | 8 | 4.77 | 1.76 | 4.19 | **0.58** | 984 K | 15.4 K | 2.43/15.4 | 35 % |
| `_sl_cluster_finish` | 4 | (4,8) | 16 | 4.90 | 1.66 | 4.19 | **0.70** | 787 K | 24.5 K | 2.53/24.5 | 53 % |
| `_sl_cluster_finish` | 8 | (8,8) | 16 | 5.15 | 1.73 | 4.29 | **0.86** | 1.57 M | 24.5 K | 2.56/24.5 | 52 % |
| `_gle_route` | 1 | (1,10) | 4 | 5.95 | 1.66 | 5.70 | **0.26** | 355 K | 32 K | 4.04/32 | 43 % |
| `_gle_route` | 8 | (8,13) | 4 | 13.70 | 1.82 | 7.26 | **6.43** | 7.5 M | 73 K | 5.44/73 | **74 %** |

The `SPLIT=False` (full-layer, 4608-wide `ALL`) rows are within 0.4 µs of their `SPLIT=True`
counterparts at every T (dep 0.54-1.44 vs 0.67-1.41), so the layer kind does not change any verdict.

Chain, as one graph of K consecutive pairs (T=1: `_gle_route`+finish, T=8: `_sl_tile_route`+finish):

| T | route | finish | sum | as one chain | overlap |
| --- | --- | --- | --- | --- | --- |
| 1 | 5.73 | 4.45 | 10.18 | **10.18** | 0 (fully serial) |
| 8 | 4.70 | 4.80 | 9.50 | **9.38** | 0.13 |

i.e. with PDL stripped the two launches do not overlap at all; in the shipped step PDL is what buys the
gap between 9.4 µs of device time and the 4.4/5.7 µs the group is measured at in-step.

Marginal cost of one **dependent** whole-CTA reduction, same harness (`redchain`, grid 1, nw 8):

| width | 8 | 256 | 1024 | 4096 |
| --- | --- | --- | --- | --- |
| µs/level | 0.128-0.160 | 0.416-0.448 | 0.416-0.480 | 0.480-0.544 |

Two levels of anything is ~1 µs; that is the ceiling on every dependency win below.

## 3. Where the dependency cost inside `_sl_tile_route` sits — it is one sort network

`artifacts/bench/g8_decomp.py`, same kernel text, one term deleted per variant, interleaved:

| T | parts | ship | +1 extra chained 4096-wide level | delete `tl.topk` | delete `wv/sum(wv)` | floor (all three) | dep |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 3.424 | +0.032 | **−0.640** | −0.032 | 2.752 | 0.672 |
| 2 | 1 | 3.488 | +0.032 | **−0.608** | −0.064 | 2.848 | 0.640 |
| 4 | 4 | 3.520 | +0.064 | **−0.832** | −0.032 | 2.688 | 0.832 |
| 8 | 4 | 3.552 | +0.032 | **−0.768** | −0.032 | 2.784 | 0.768 |

* The row-norm reduction (`inv = rsqrt(sum(pooled*pooled)/4096)`) costs **0.03-0.06 µs**, not 0.48: it
  overlaps its own load completely, exactly as the earlier `redcost` found for a single block reduction.
* `tl.topk(packed, 8)` over the 256-wide ordered-bits packing is **90-100 % of the whole dependency
  cost**. It is ~1.5-2 reduction levels, i.e. Triton's top-k is already a near-minimal network — a
  full bitonic sort of 256 would be 8 levels (3.2 µs) and is not what it emits.
* Alternatives measured in the same session: `tl.sort(packed, descending=True)`+gather (the file's
  MODE 1) **+1.0 µs**; a uint32 top-k (would drop the `255-e` tie-break, so illegal) only **−0.13 µs**.
  A hand-rolled 5-stage bitonic top-8 could in principle reach ~−0.3 µs, i.e. below the in-session A/A
  noise, for ~600 B of a 455 B budget. Not taken.

So the packing trick the ticket proposes as the rewrite ("`tl.topk` over an ordered-bits packing") is
**already what this chain ships** — `SL_ROUTE_CONFIG = {1:(8,0,0), 8:(4,0,0)}` selects MODE 0 at both
scored shapes. There is no second copy of the sibling's win to collect here.

## 4. The other two: irreducible chains, and one dead removable chain

* `_sl_cluster_finish` (`dep` 0.58-0.86) is `inv = rsqrt(sum(f*f)/8192+eps)` -> `n = bf16(f*inv)` ->
  `head = carry + beta*n` -> `invnext = rsqrt((tail+sum(head*head))/8192+eps)`: **two dependent
  1024-wide levels**, 0.42 each, partly overlapped. The bf16 rounding of `n` forbids the algebraic
  expansion that would make `sum(head*head)` independent (the same identity the earlier census already
  rejected for `_boundary2`, measured worse there). Irreducible. Its tiling is also already optimal:
  traffic = (8192/BC)×10.1 K redundant prefix + 12 B/column, minimised at BC=256 per token (32 CTAs,
  15.4 K/CTA); BC=1024 would cut total traffic 2.5× but raise per-CTA bytes to 24.5 K and make it
  *slower* per the table above, and BC=8192 would be 108 K/CTA.
* `_gle_route` at T=1 (`dep` 0.26) is one 256×8-wide sum plus lane extractions — already minimal, and
  the kernel is 43 % of the per-SM wall, i.e. byte/latency-bound, not dependency-bound.
* `_gle_route` at **TT=8 is the only genuinely removable dependent chain in the group: 6.43 µs**, from
  `for tok in tl.static_range(TT): r1t = rsqrt(tl.sum(load(PART+(cta*TT+tok)*8), 0)/8192)` — 8
  barrier-separated 256-wide reductions per carry program, serialised, in every one of the 8+ carry
  programs. The fix is the standard one (one 2-D `[256, TT]` load + one axis-0 reduction for all TT
  tokens, then a TT-wide lane extraction; the values are the same 256, but the 1-D -> 2-D reduction
  re-associates, so it needs its own bit-identity proof before it could ever ship).
  **It is dead code**: `GLE_T = {1: True}` and re-enabling GLE at T=8 is a measured loss the ticket
  forbids. This census *explains* that loss quantitatively — GLE@T=8 pays 13.7 µs for the route alone
  against 5.0 µs for `_sl_tile_route` — and the note is left here for whoever revisits the gate: fix the
  TT loop *before* re-measuring `GLE_T[8]`, otherwise the gate is measuring this loop, not GLE.

## 5. Bit-identity (the shipped forms, over the scored matrix)

`artifacts/bench/g8_ident.py`, A100: `_sl_tile_route` MODE 0 (shipped pack+`tl.topk`) vs MODE 2 (the
base file's 8× argmax + masked-sum loop), over **T ∈ {1..8} × SPLIT ∈ {True,False} × LS ∈ {0,160,224}
× input scale ∈ {1, 1/64}** (96 cells; LS stands in for page_size, the scale for kv_scale — these
kernels never see the page table, only `local_start` and the residual stream they are handed):

| cells | ids `torch.equal` | lids `torch.equal` | wts rel-L2 | xm rel-L2 | verdict |
| --- | --- | --- | --- | --- | --- |
| 96/96 | yes | yes | **0.0** | **0.0** | identical |

`_sl_cluster_finish` re-launched into fresh buffers at T ∈ {1,2,4,8}, both configs: `s2` and `out`
`torch.equal` (bit-equal), grid/nw as shipped. There is no variant to compare it against, so this is
only the A/A control — it does confirm the kernel is deterministic under the exact shipped launch.

## 6. Byte cap

Net change **0 B**; file remains 999,545 B against 1,000,000 → **455 B free**. If a future merge needs
headroom, the cheapest displacement is the trailing comments on the `#define` run at lines 123-225
(10,717 B of comment text, zero code) — trim ~2 KB there and no `#define` value changes.

## 7. Price

| item | Δ µs/step | where | Δ % of load | Δ score (1/6 log weight) |
| --- | --- | --- | --- | --- |
| shipped | **0** | — | 0 | **0** |
| ceiling if *every* dependency cost in the live group vanished (physically impossible: a sort network and a norm-of-norm chain) | −1.00 (T=1: 0.26 route + 0.74 finish) / −2.27 (T=8: 1.41 route + 0.86 finish) | d1s 67.8 µs / v1s ~110 µs | 1.5 % / 2.1 % | **+0.24 % / +0.34 %** |
| reachable part of that ceiling (only the top-k's excess over a minimal network, ≤0.3 µs, under the 0.9-1.9 % A/A noise) | ≤0.3 | T=8 only | ≤0.3 % | **≤+0.05 %** |

For scale: the whole T<=8 group is 4.4 µs of a 67.8 µs step (6.5 %) and 5.7 of ~110 (5.2 %), so
eliminating it *entirely* would move the score by ≤0.6 % — the row needs **+5.8 %** (2.24934 ->
2.37932). This group cannot close that gap and has no removable dependency left in the live path.

## 8. Gates

No candidate produced, so no lint/compile delta applies; `ast.parse` of the base is unaffected (file
untouched). Oracles executed: `g8_census_run.py`, `g8_decomp_run.py`, `g8_ident_run.py` (A100, logs in
`artifacts/bench/g8_{census,decomp,ident}_a100.log`).
