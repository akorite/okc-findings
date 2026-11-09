# glue: the boundary+route group, why it costs what it costs, and the one rewrite that pays

File: `src/okc_loop/vendored_work/layer/klr_glue.py` — a copy of `klr_qd2.py` (board `82c098ae` +
ATT7 QDIRECT/PK2, measured 2.2536) with one kernel changed (`_route_split`, `:10833`) and its launch
site (`:11642`). Flag `GLUE_PACK` (`:16406`). Everything else in the file is byte-identical.

The group is three Triton launches on the base chain — `_boundary1[(T,)]`, `_route_split[(T,)]`,
`_boundary2[(T,)]` — and at T=1/8 the graft chain replaces them with `glue_cluster_boundary`,
`_sl_graft_route[(T,2)]` and `_sl_cluster_finish`.

All numbers below are **measured**, on a free Colab A100-40GB, by
`artifacts/bench/{glue_lat,route_top8,redcost,glue_selfcheck}.py` (each: one CUDA graph holding K=32
consecutive launches of the same kernel, 30-60 rounds, a 64 MB L2 flush before every replay, per-launch
time = elapsed/K, variants interleaved so A/A drift shows up as a nonzero delta). Only ordinal A/B is
transferable: A100 is 108 SMs @1.41 TB/s @1.41 GHz, the judge H200 132 @4.8 @1.98. Two sessions of the
same script differed by up to 20 % in absolute µs, so no cross-session comparison is used anywhere here.

## 0. The cost of a block reduction (the number everything else follows from)

`artifacts/bench/redcost.py`, grid (1,), num_warps=8, A100:

| kernel | µs | what it shows |
| --- | --- | --- |
| `nop` (grid 1, one store) | **1.44** | this harness's per-launch floor |
| one 4096-load + store, no reduce | 2.336 | a 16 KB load costs 0.90 |
| **1** `tl.sum` over that block | 2.304 | **a single block reduction is free** (fully overlapped with its load) |
| 2 chained `tl.sum` | 2.784 | +0.48 |
| 4 chained | 3.744 | +0.48 each |
| 8 chained | 5.696 | +0.48 each |
| 8 chained, 256-wide (the router's row) | 4.672 | **+0.35 each** |
| **8 independent** `tl.sum` of one operand | 2.336 | **+0.00 — independent reductions overlap completely** |

So a *dependent* whole-CTA reduction is a **0.35 us (256-wide) / 0.48 us (4096-wide) serialisation**, and
reduction depth — not bytes — is what a Triton glue kernel is actually paying for.

## 1. Census, as the file launches it on each scored shape

Bytes/token are from the launch arguments (S=1 at every scored shape: `dense()` returns `torch.mm` above
T=32 and `_bfp_gemm` at or below, so the o/unpooler partial count is 1; NU = 4096 at T<=8, else 8192).
µs are A100, interleaved, cold-L2. `floor` = the same kernel with the two row reductions deleted and the
identical loads/stores/grid/num_warps — i.e. the bandwidth-only floor at that geometry.

| kernel | grid | thr | read/token | write/token | µs T=1 | T=8 | T=16 | T=32 | T=128 | T=256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `_boundary1` `:10768` | `(T,)` | 512 (nw16) | P 16 K + stream 32 K + gates 32 K = 80 K | s1 32 K + h 16 K = 48 K → **128 K** | 8.06 | 8.32 | 8.48 | 6.69 | 12.51 | 18.37 |
| `_route_split` `:10833` | `(T,)` | 256 (nw8) | pooled 8 K + logits 1 K + bias .5 K + gpre 8 K = 17.5 K | xm 8 K + ids/wts .1 K → **25.6 K** | 9.06 | 9.22 | 9.28 | 7.30 | 8.86 | 10.66 |
| `_boundary2` `:10895` | `(T,)` | 512 (nw16) | f 8/16 K + s1 32 K + gates 48 K = 88/96 K | s2 32 K + out 16 K = 48 K → **136/144 K** | 8.38 | 8.54 | 8.96 | 7.23 | 11.94 | 16.74 |
| **group** | | | | | **25.5** | **26.1** | **26.7** | **21.2** | **33.3** | **45.8** |
| `floor` b1 (no reductions) | `(T,)` | 256 | same | same | 6.88 | 7.01 | 7.17 | 5.70 | 9.60 | 13.54 |
| `floor` b2 (no reductions) | `(T,)` | 256 | same | same | 7.04 | 7.17 | 7.39 | 5.95 | 10.02 | 14.02 |

Achieved bandwidth, and the wall it is being judged against (A100: 13.1 GB/s of DRAM share per SM,
1.41 TB/s total; L2 40 MB, so the gate vectors and, at T>=32, `stream` are L2-hot — the same state the
in-step chain sees, because each kernel's input was written by the launch immediately before it):

| kernel | GB/s at T=1 | T=8 | T=16 | T=32 | T=128 | T=256 | % of what T CTAs can pull |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `_boundary1` | 16.3 (1 SM) | 146 | 293 | 627 | 1.28 T | 1.83 T | **93-100 %** at every T |
| `_boundary2` | 15.9 | 152 | 313 | 723 | 1.39 T | 1.94 T | **96-100 %** |
| `_route_split` | **2.8** (1 SM) | 28 | 57 | 112 | 443 | 615 | **15-44 %** |
| floor b1 | 18.6 (1 SM) | 146 | 293 | 736 | 1.75 T | 2.48 T | — (this *is* the wall) |

Arithmetic intensity is ~0.05 flop/B for both boundary kernels (one row-rms and one gate fma per element)
and ~0.02 for the route, so none of them can be compute-bound; the question is only which wall.

**Classification.**

| kernel | class | evidence |
| --- | --- | --- |
| `_boundary1`, `_boundary2`, T <= 32 | **SM-count-bound**, each CTA already at its *per-SM* wall | grid is T of 132 SMs (12-24 % of the machine); the reduction-free floor moves the same bytes in the same time over 1, 8, 16, 32 CTAs at exactly 18.3±0.4 GB/s per CTA and is 85 % of the kernel; the two dependent rms are the other 15-22 % |
| `_boundary1`, `_boundary2`, T >= 128 | **at the DRAM wall** | b1 T=256 1.83 TB/s (130 % of DRAM, L2-assisted), floor 13.5 of 18.4 us |
| `_route_split`, every T | **latency-bound** | 2.8 GB/s at T=1 against 18.6 available on that one SM; 615 GB/s at T=256 against 2.48 TB/s available to 108 SMs; 84 % of the kernel is 16 *dependent* block reductions |

## 2. The worst kernel, and its limiter in numbers

By µs×share the largest is **`_boundary1`** — 7.4 of 45.8 µs at T=256 (16 % of the step's group), 3.0 of
21.2 at T=32 — but it has almost nothing to give: at T<=32 it is one CTA per token moving 128 KB, and the
CTA measured 627 GB/s at T=32 = 19.6 GB/s per SM against the 18.3 GB/s/SM the reduction-free floor gets;
at T>=128 it is inside 5 % of DRAM. Adding CTAs needs a cross-CTA row sum, which is a second launch or
atomics (falsified, §3c), and the two rms are inherently dependent (`sum(a*a)` -> `s1` -> `sum(s1*s1)`).

The worst kernel **with headroom** is **`_route_split`**, and this is the ticket's question answered
exactly: one CTA per token; each CTA does `for k in static_range(8)` of

```
idx = tl.argmax(sel, 0)                            # reduce #2k   (256-wide, dependent)
val = tl.sum(tl.where(e == idx, sc, 0.0), 0)       # reduce #2k+1 (256-wide, dependent)
sel = tl.where(e == idx, float("-inf"), sel)       # next round cannot start until both retire
```

that is **16 dependent 256-wide block reductions at 0.35 us = 5.6 us, inside a kernel that only moves
25.6 KB**. Measured ship-minus-(same kernel with the value reductions deleted) = 1.52/1.50/1.50/1.15 us for
the 8 even-numbered ones; ship-minus-`tl.topk` form = 5.5 us for all 16. The bandwidth is not there to be
saved: at T=1 the kernel is at 15 % of one SM, at T=256 at 44 % of the device.

## 3. Rewrite candidates

| # | candidate | measured | risk |
| --- | --- | --- | --- |
| **a** | **pack + `tl.topk`**: `sel` -> monotone ordered uint32 -> `(ordered<<32) \| (255-e)` -> `tl.topk(.., 8)`, then `tl.gather(sc, ids, 0)`. The tie-break by `255-e` reproduces argmax's first-index rule exactly. Same trick the file already ships inside `_sl_tile_route` (MODE 0), so `tl.topk` is known-good on the judge's Triton. | **9.06 -> 4.96** at T=1, 10.66 -> 6.62 at T=256 (`route_top8`, A100) | none observed: bit-identical, and it strictly removes work |
| **b** | **+ (T,2) program split**: (t,0) pre-expert norm + store, (t,1) top-8 — the structure `_sl_tile_route`/`_graft_route` already use | **4.96 -> 3.78** at T=1, 6.62 -> 6.40 at T=256 (nw8) | a second CTA per token; at T=256 the grid becomes 512 |
| **c** | more CTAs for `_boundary1`/`_boundary2` (a cross-CTA row sum: atomics or a 2-launch partials+apply) | **does not pay**: the reduction-free floor at the same grid is already 85 % of the kernel, and with NP CTAs each still reading the whole row the trade is exactly 0 (T=32 np2 6.75 vs 6.69), -0.48 at T=8 np4, -0.80 at T=1 np8 | needs `torch.zeros` + atomics or an extra launch, and it is the shipped-and-lost `gluev2` in an optimised form. **Left out.** |
| **d** | one dependent reduction instead of two, from the three moments of `s1 = a1*stv + b1*r*a` (the identity `_bnd2_apply` already trusts) | **worse**: 8.19 -> 9.50 at T=1, 18.11 -> 22.27 at T=256 | the 3 extra independent reductions are free but the 6 extra 8192-wide multiplies are not. **Left out.** |
| **e** | `num_warps` retune of the boundary kernels | 4/8/16/32 swept: w8 is 0.1-0.2 us ahead of the shipped w16 at T<=32, a tie above — inside the 20 % session drift | already owned by `compose_boundary_warps`. **Left out.** |
| — | merging two group kernels that read the same tensor | nothing to merge: `_boundary1` -> `_route_split` share no input, and `_boundary2`'s only shared input (`s1`) is a 32 KB/token store/load round trip that costs less than recomputing it | falsified class, see `fuse_boundary.md` |

## 4. What was built

`_route_split` gains `PACK: tl.constexpr`; `PACK=True` launches as `(T, 2)` with program 0 = the
pre-expert norm and the store (both `PERMUTED` layouts) and program 1 = (a) + its own pre-wait loads of
`BIAS`/`LS`, so the PDL early-trigger keeps its meaning. The shipped body stays verbatim under
`PACK=False` and is the fallback. Host side, one line at the only launch site (`:11642`):

```python
gpk = state.get("glue_pack", {}).get(T, False)
_route_split[(T, 2 if gpk else 1)](..., PACK=gpk, num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
```

`state["glue_pack"]` comes from `_glue_pack_selfcheck` (`:16409`), called once per prepare at `:12783` —
**before** the compose sweep, so the sweep's base variant is the route the run will actually use. Per T in
`TUNED_T`, both `PERMUTED` layouts: shipped vs `PACK`, requiring `torch.equal` on `ids` and `lids` and
rel-L2 <= 1e-5 on `wts` and `xm` (the bound the file already uses for a path that only reassociates). Any
mismatch, any raise (e.g. no `tl.topk`) leaves that T on the shipped loop; `GLUE_PACK = False` forces the
shipped loop everywhere.

`_route` (the fused-4608 pooler variant) is deliberately untouched: it has the same loop, but
`prepare` pins `mm_pick[(T,"pr")] = "bfpf"` at every scored shape, and that pooler returns `S == -1`, i.e.
`_route_split` is the kernel that runs.

## 5. The oracle: both variants, same script, interleaved

`artifacts/bench/glue_selfcheck_run.py` extracts `_route_split`'s **actual source text** from
`klr_glue.py` by AST, writes it to a module and runs it, so these are the shipped kernel and the shipped
rewrite, not a transcription.

Correctness, **A100, all nine tuned T, both store layouts**: `ids`/`lids` `torch.equal`, `wts` and `xm`
rel-L2 **0.0** everywhere.

| T | shipped loop | PACK (this file) | delta | ratio |
| --- | --- | --- | --- | --- |
| 1 | 9.344 | 3.808 | **-5.536** | 2.45x |
| 8 | 9.440 | 3.840 | **-5.600** | 2.46x |
| 16 | 9.504 | 3.936 | **-5.568** | 2.41x |
| 32 | 9.600 | 4.032 | **-5.568** | 2.38x |
| 128 | 11.392 | 5.856 | **-5.536** | 1.95x |
| 256 | 13.920 | 8.128 | **-5.792** | 1.71x |

Second device, L4 (58 SMs, sm_89), same script: 5.344 -> 2.207 at T=1 (2.42x), 15.712 -> 8.128 at T=256
(1.93x), same exactness. Two different SM counts and two different architectures give the same ratio,
which is what a latency fix should look like.

## 6. Price

The delta is pure reduction latency, so it scales with SM clock, not bandwidth: x1.41/1.98 = 0.71 for the
judge. Per step that is **-3.9 to -4.1 us wherever the base chain's route runs**.

| load | step µs | group µs | route share | Δ µs/step | Δ % of that load |
| --- | --- | --- | --- | --- | --- |
| d1s | 67.8 | 4.4 | graft covers it (`SL_CLUSTER_CONFIG[1]`) | **0** | 0 % |
| verify 1 (T=8) | - | - | graft covers it | **0** | 0 % |
| d16s / d32s | 186 | 10.9 | the loop route runs — a 10.9 us group cannot be built from a 2 us route plus 8.5 MB at 32 CTAs (max 2.2 TB/s = 3.9 us), so this row *is* the loop | **-4.0** | **2.2 %** |
| verify 32 (T=256) | 350.5 | 21.1 | ~6 of 21.1 (the boundaries are at 70 % of DRAM there, 70 MB) | **-4.1** | **1.2 %** |

Score = geomean of the six speedups, so a uniform fraction f off every load is +f in the score. Two of six
loads are already covered by the graft chain; the other four take it:

* if the graft chain is **off** at T=16/32/128/256 (which the 10.9 us d32s group implies): **+0.5 to +1.1 %
  score**, 2.2536 -> **2.265-2.278**;
* if the compose sweep later decides the graft chain (it will now see a 4 us cheaper base, so it should
  decide *less* often): **0** — the graft chain's `_sl_tile_route` already ships this exact packing, and
  this change is then a pure tie.

The second row is the honest downside: not a regression, an add-nothing. There is no configuration in
which PACK makes a step slower — it removes 15 of the 16 dependent reductions from the same launch.

## 7. Gates

* `ast` — parses.
* `PYTHONPATH=src .venv/bin/python -m okc_loop.cli lint klr_glue.py`: same **30 Tier-0 errors as
  `klr_qd2.py`**, and the full-output diff is only six line-number shifts (+46) from the ~50 added lines.
  No new error class.
* sm_90a compile gate: `artifacts/bench/gate_glue.py`, `colab run --gpu L4`, 87.6 s wall.
  `_att7`, `_moe`, `_moe_tiny`, `_moe_tiny2`, `_moe_mid`, `bfp__get_ext`, `_moe(32)` **ok**. The three
  `ok=False` are `_graft_glue_module`, `_graft_b2_module`, `_sl_cluster_module`, each
  `ModuleNotFoundError: tvm_ffi` — the limitation `tools/colab_gate.py` documents ("Colab lacks tvm_ffi;
  the judge image has it"), identical to the base file's gate. The module imported and every CUDA
  extension compiled for sm_90a.
* Executed the rewritten kernel on hardware: §5.
