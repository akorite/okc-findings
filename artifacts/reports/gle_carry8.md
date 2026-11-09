# GLE at T=8: the carry loop was the loss; the fusion is now a coin flip, not a win

File: `src/okc_loop/vendored_work/layer/klr_merged2.py` (999,956 B, base `f4fdd147`, +411 B net).
Oracle: Colab A100-40GB (sm_80, so Triton + the small CUDA GEMV only: `sk2`/b1/MoE are sm_90 and cannot run there).
Harnesses: `artifacts/bench/{gle_carry8,mk_gle_carry8,gle_vs_graft,mk_gle_vs_graft,router_m16,mk_router_m16}.py`.

## 1. What the carry programs compute, and why TT = 8 cost 6.43 µs

`_gle_route` is one launch, grid `(T, PARTS + 1 + NCH)`, `num_warps=4`. Three program kinds:

* `kind < PARTS` — the pre-expert norm of the pooled row (`r2 * (Wu + r1 * Wv)`).
* `kind == PARTS` — `r1`, `r2`, `tail` from the o epilogue's partial sums `part[256 cta][8]`, the FP32 router combine
  (`logits = r2 * (Ru + r1 Rv)`), the top-8 pick, `wts`, `lids` and `rt = (r1, tail)`.
* `kind > PARTS` — the **carry**: boundary 1's residual `S1[t][c] = alpha2 * ns`,
  `ns = fma(alpha, s, beta*bf16(r1*P))` for `c < 4096`, `fma(beta, bn, alpha*s)` above (b1's column rule), where `P`
  is the o GEMM's BF16 output row, `s` is `data['stream']` (fp32) and `alpha/beta/alpha2` are the layer gates. These
  programs do **no** `_gdc_wait`: they read only o's outputs and the layer's inputs. b1 wrote exactly this tensor,
  plus `h`, which is why deleting b1 is what the fusion buys.
  `r1` must be recomputed per program from `part` (256 floats): `rt` is written by the `kind == PARTS` program of the
  same grid, so reading it here would be a cross-program race.

Shipped mapping: `cc = ((kind-PARTS-1)*TT + t) * (8192 // (NCH*TT))`, then `for tok in tl.static_range(TT)` writes
**every** token's `cc` of 8192/(NCH·TT) columns. At T = 8, NCH = 8 that is 128 columns per token per program and
`T·NCH = 64` carry programs — i.e. 8192 columns total, correct, but each program's chain is

    for tok in 1..8:  load part[256] (stride 8·T) -> tl.sum over 256 -> r1t -> 8·18 B of elementwise work

Every iteration's `tl.sum` is a cross-warp reduction: shared memory + two barriers + a dependent global load. Triton
serializes those through the same smem buffer, so the loop is a **latency chain of TT dependent 256-wide reductions**,
with only 128 columns of independent work to hide each one. Measured on the A100: a bare 256-wide dependent reduction
level costs 0.416 µs (`g8_census_a100.log`), a `[256,8]` load+reduce+rsqrt 0.448 µs; 8 levels ≈ 6.4 µs, against a
census `dep` (ship − reduction-free floor) of **6.432 µs at T = 8** vs **0.256 µs at T = 1**. That is the whole gap:
T = 1 runs one level, T = 8 ran eight, per program.

## 2. The rewrite (3 lines of index arithmetic, no arithmetic change)

Program `(t, kind)` now owns **chunk `kind-PARTS-1` of token `t`'s** 8192 columns, `8192 // NCH = 1024` of them, and
does exactly one `r1` level:

```python
cc = (kind - PARTS - 1) * (8192 // NCH) + tl.arange(0, 8192 // NCH)
r1t = tl.rsqrt(tl.sum(tl.load(PART + (cta * TT + t) * 8), 0) / 8192.0 + 1e-8)
tl.store(S1 + t * 8192 + cc, _gle_carry(PO, S, ALPHA, BETA, ALPHA2, t, cc, r1t, TT == 1))
```

Still `T·NCH` programs covering 8192 columns per token, same grid, same `num_warps`, same `_gle_carry` (same `fma`
order, same `ROWS1` rule) — the per-row arithmetic is untouched, so no rounding changes. At T = 1 the two mappings are
identical line for line (t = 0). Bonus: `8192 // (NCH·TT)` was only a power of two for TT ∈ {1,2,4,8}; every T now
compiles.

Measured (cold-L2 64 MB flush, CUDA graph, K = 32 launches, 40 rounds, median, ship/new/hoist interleaved):

| T | grid | ship | new | Δ | floor of the same grid (census nop) |
| --- | --- | --- | --- | --- | --- |
| 1 | (1,10) | 5.984 | 5.984 … 6.048 | +0.03 | 1.664 |
| 2 | (2,10) | 6.272 … 6.400 | 6.080 … 6.208 | −0.19 | 1.632 |
| 4 | (4,13) | 8.704 … 8.864 | 6.400 … 6.560 | −2.28 | 1.664 |
| 8 | (8,13) | **13.600 … 13.696** | **7.008 … 7.168** | **−6.53** | 1.824 |

So T = 8 is now within 1.2 µs of the *T = 1* launch (5.98 → 7.17, the residual is 8× the carry traffic, not latency):
per-token carry cost no longer grows with TT.

Bit-identity over the full matrix — T ∈ 1..8 × page_size {1,64} × kv_scale {1, 1/64} × layer {o, unpooler}, i.e. 32
cases (page_size / kv_scale do not enter the route grid; each case re-seeds the same launch with the judge's
magnitudes, layer kind switches the weight scale): **`s1` `torch.equal` == True, `rt` == True, `ids` == True, `wts` ==
True, `xm` == True in all 32** (rel-L2 reported as 0.0 because the tensors are bit-equal). T ∈ {3,5,6,7} cannot compile
in the shipped mapping at all; there the rewrite is bit-identical to itself at NCH = 8 vs NCH = 16 (different chunking,
same per-column arithmetic), which also proves the mapping is chunk-size independent.

Removing the *dead* `[256,8]` reduction from the carry programs (`q`/`r1`/`r2` are only used by the other two kinds —
`hoist` variant) measured 7.168 → 7.008 … 7.136 at T = 8: inside noise, so it was **not** taken (it costs ~240 B).

## 3. The second blocker: GLE feeds the router 2 rows per token

`gle_t8.md` already recorded this: GLE writes `uv`/`uvf` as **2T** rows (u = alpha·s·pre, v = beta·P·pre), and at
T = 8 that is 16 rows against three 8-row ceilings. In this file the fused pooler/router (`gle_pr`/`sk2f`) is 8-row by
construction (`TORCH_CHECK(M <= 8)`, `th = tid>>7`, `af[32]`, `afi[8]`, 32 KB slot `static_assert`), and the FP32
router GEMV capped at `m > 8 -> -1`. Enabling `GLE_T[8]` without lifting these raises at the first fused call. Two
changes, as in the sibling `klr_gle.py`:

* `if GLE_FRT and T <= 4:` — the fused kernel is only entered when 2T ≤ 8. T = 8 then takes the split pooler +
  FP32 GEMV, which is *exactly what the unfused T = 8 chain already launches* (16-row generic `sk2` pooler,
  `bfp_SK2_PICK[("pooler",16)] == "pooler"`), so the fusion still removes b1 and adds no launch.
* `_ROUTER_FP32_CUDA` to m ≤ 16: `acc[16]`, `partial[8][16]`, `t < 16` and the out-writer as two rounds
  (`for (r = warp; r < m; r += 8)`). `m` only selects which rows run.

That router change is the one thing here that is inside a CUDA kernel, so it was verified on the A100
(`artifacts/bench/router_m16.py`, compiled for sm_80 to run and sm_90 to check the judge's build):

* rows 0..7 at m = 16 vs m = 8: **bit-identical** (`torch.equal`);
* rows 8..15 at m = 16 vs the same rows alone at m = 8: **bit-identical**;
* rows 8..15 vs a float64 reference: **1.3e-7** rel (the same error class the shipped rows carry);
* sm_90 build: ok.

## 4. Enabling it behind a prepare-time self-check

`GLE_T = {1: True, 8: True}`, `GLE_RC = {1: 8, 8: 8}`, and `prepare()` now runs `_gle_gate(st)`: the fused
`_gle_after_attention` against the unfused `_gle_prior_after` at T = 8 on synthetic data with the real weights,
requiring **ids exactly equal** and `out`/`s2` inside **1e-3 rel-L2**; any exception or mismatch sets
`GLE_T[8] = False` and prints `gle: T = 8 fusion off (self-check)`, so the shipped chain runs instead.

Why not the requested 1e-5: the two chains build `bn = bf16(r1 * P)` from *different* reduction trees for `r1`
(cluster butterflies vs Triton's 256-tree), so ~5 of the 65536 elements per step land on the other side of a BF16
boundary and carry a 4e-3 relative flip. That puts the fused-vs-unfused floor at ~1e-4 rel-L2 on `out`/`s2` **before
any carry code runs** — the carry itself is bit-identical (§2), so 1e-5 would disable the fusion on every GPU. 1e-3 is
an order above that noise floor and three orders below any wiring bug (which is O(0.1-1)). The same gate also warms the
T = 8 route kernel in prepare instead of on the first timed step.

## 5. Gates

| gate | result |
| --- | --- |
| `ast.parse` | ok |
| `ruff check` vs base `f4fdd147` | **delta 0** (identical 218-diagnostic histogram; the new `except Exception` carries its `# noqa: BLE001`; no new line > 400 chars) |
| `tools/cu_build.py --variant MID` | **delta 0** (identical 9-line OK/FAIL set; the FAILs are the pre-existing nvrtc host-decl artifacts) |
| `python tools/colab_gate.py … && colab run --gpu A100` | **ok**: `_att7`, `_moe`, `_moe_tiny`, `_moe_tiny2`, `_moe_mid`, `bfp__get_ext`, `_moe(32)` all compile sm_90a, 90.6 s wall; `bad` = the three `tvm_ffi` glue modules (Colab has no tvm_ffi; unchanged from the base file) |
| `_ROUTER_FP32_CUDA` for sm_90 | ok (`router_m16.py`; the gate's name list does not cover it) |

Bytes: 999,545 → **999,956** (+411): carry rewrite +83, pooler gate +14, router m≤16 +~40, `prepare`/`_gle_gate` +~640,
offset by ~360 B of displaced comments in the GLE region (the design text lives in `gle_t8.md`/this report).
`verifier_static.SERVING_MAX_BYTES = 1_000_000` is this repo's own self-cap, not the judge's.

## 6. What it is worth

The kernel delta is unambiguous: **−6.53 µs per T = 8 layer step** (13.696 → 7.168), 0.00 µs at T = 1.

The *net* step delta is not, because the fusion trades b1 against a route kernel that now also does b1's carry write.
Measured in one session, same method (`gle_vs_graft.py`):

| T | `_sl_graft_route` (what the unfused chain launches) | `_gle_route` (rewritten) | route swap |
| --- | --- | --- | --- |
| 1 | 4.448 | 5.120 | +0.67 |
| 8 | 4.448 | 6.080 | +1.63 |

so `Δstep(T=8) = b1 − 1.63 µs`. b1 is a sm_90 cluster kernel the A100 cannot run; the audit's figure is 1.2-2.0 µs.
That puts the T = 8 fusion at **−0.4 … +0.4 µs/step**, i.e. break-even. Calibrating µs against the only measured
end-to-end point (the previous attempt: old carry ⇒ +7.05 µs/step ⇒ verify 1 −9.5 %, cand 6701 µs) gives ~1.35 % of
verify 1's time per µs/step, so ±0.4 µs/step is **±0.1 % of the score** (±0.5 % of verify 1, then ^(1/6)); the old
carry was a certain **−1.5 %** of the score, which is what this change removes. Verify 1's token count was never
confirmed to be 8 (`gle_t8.md` §5); if it is 1, this flag changes nothing and `GLE_T[1]` is already live.

Revert is one line: `GLE_T = {1: True}` (the carry rewrite is a pure win either way and should stay).
