# fuse_boundary: absorbing a per-row boundary/route launch into a neighbour's epilogue

File: `src/okc_loop/vendored_work/layer/klr_fuse.py` (copy of the board file, sha `82c098ae` base).
Flag: `OKC_FUSE_S1 = False` (`:16380`). Self-check fills `state["fuse_s1"][T]` at prepare time.

## 1. The chain, with what actually runs at d32

`kernel_fn` -> `_after_attention` (base body at :11549; the live name is rebound by the graft/gle/sl wrappers,
`_vj_after_attention = _after_attention` at :12517). At **T = 32** the dispatch outcome is:

| # | launch | grid | state at T = 32 |
| --- | --- | --- | --- |
| 1 | qkv | `_bfp_gemm` (sk2, BF16 out) | `mm_pick[(32,"qkv")] = "bfp"` forced in `prepare` :12698 |
| 2 | attention | `_decode_attention7` fused | `att_ok` |
| 3 | `o` | `_bfp_gemm` -> `(1,32,8192)` bf16, `S = 1` | `mm_pick[(32,"o")] = "bfp"` |
| 4 | **`_boundary1[(32,)]`** :10765 | 32 CTAs, 8192 cols each | reads P(=o) bf16 0.5 MB + STREAM fp32 1 MB, writes s1 fp32 1 MB + h bf16 0.5 MB = **3.0 MB** |
| 5 | pooler+router | `bfp_fused_pooler_router` (ppick `"bfpf"`, :12698) | returns `(pooled, logits), -1` |
| 6 | **`_route_split[(32,)]`** :10833 | 32 CTAs, 4096 + 256 cols | reads pooled bf16 0.26 MB + logits fp32 0.03 MB = **0.3 MB**, writes ids/lids/wts + xm bf16 0.26 MB |
| 7 | MoE | `_moe(32)` agent kernel | black box, 72.2 us |
| 8 | unpooler | `_bfp_gemm` -> bf16 `(1,32,8192)` | `mm_pick[(32,"un")] = "bfp"` |
| 9 | **`_boundary2[(32,)]`** :10895 | 32 CTAs, 8192 cols each | reads f bf16 0.5 + s1 fp32 1.0, writes s2 fp32 1.0 + out bf16 0.5 = **3.0 MB** |

Three boundary/route launches, 6.3 MB, and each is capped at **32 CTAs on 132 SMs**. That is the 10.9 us: an
effective ~0.85 TB/s, i.e. 24% of the SM count and 20-25% of the 4.3 TB/s wall. **The cost is occupancy, not
bytes and not launch count** — deleting a launch outright (1.2-2.0 us) is worth less than widening one.

Ordering facts that decide the fusion space:
* 4 produces `h`, which is 5's **A operand**; 5 produces `pooled/logits`, which is 6's only input. 3->4->5->6 is a
  strict serial chain, all with PDL early-trigger + `gdc_wait` (so no stream junction is available to exploit).
* `s1` (4's fp32 output) is consumed **only** by 9, i.e. after the MoE and the unpooler. Its production is
  therefore order-free with respect to 5..8.
* `gluev2` (parallel 256-CTA boundaries, `_gemm_o`+`_bnd1_apply`, `_route_a`+`_route_b`, `_bnd2_sums`+`_bnd2_apply`,
  :10601-10760) is **dead**: `for v2 in (False,)` at :11186, comment "Parallel boundary search lost every
  complete-chain sweep".

## 2. Why every epilogue pair is blocked

A whole-row reduction (4, 6, 9) can only be absorbed by a producer that owns a whole row. All three producers are
N-tiled GEMMs, and folding the activation into their A-operand load path is bandwidth-prohibitive because A is
re-read `N/tile` times:

| pair | verdict |
| --- | --- |
| 3 (`o`) -> 4 | o's o-grid is (128 tiles x 1); boundary1 needs the full 8192 row => grid sync. Only available as cluster/atomic (see below). o is also `_bfp_gemm` (12-bit lossless, measured 4-9% better than any Triton o), so a Triton o+boundary1 would pay **+4.2 MB of weights (~1.05 us)**. |
| 4 -> 5 | h is the pooler's A operand: A is re-read ~17x (4352/256). Materialising h costs 0.5 MB/pass; folding `h = f(a, stream)` into A costs 1.5 MB/pass => **+8.5 MB (~2 us)**. Blocked. |
| 5 -> 6 | `_route`'s top-8 needs all 256 logits => 2-CTA junction. Measured **0.1-0.4 us**; already killed. |
| 6 -> 7 | xm is the MoE's input, per-token row re-read per expert slot. Blocked. |
| 7 -> 9 | unpooler GEMM in between. |
| 8 -> 9 | unpooler A operand is `m` (0.26 MB) re-read 8192/BN times; folding f is the same trap as 4->5. Blocked. |

The only pair where **both sides are whole-row kernels** is 4 and 6. They are not adjacent, but 4's `s1` half is
order-free (consumer is 9), so it can move into 6's launch. That is the one implementable absorption.

## 3. What was built

`_route_split` gains `NS1` (`:10838`): grid `(T, 1+NS1)`, programs `j < NS1` write `s1` columns
`[j*8192/NS1, ...)` from `a1*stream + b1*(a*rstd(a))` with `rstd(a)` handed over in `RSCAL[t]`; `_boundary1` gains
`SAVE_S1` / `RSCAL` (`:10766`) and skips its `s1` store when fused. The arithmetic is **bit-identical** to
`_boundary1` (same operands, same association `b1 * (a * r)`, same `rsqrt(sum(a*a)/8192 + 1e-8)` computed in the
same kernel), so the self-check bound is the file's own `1e-5` on `out`/`s2` plus exact `ids`, not the problem's
1.5e-2. Both branches carry the file's PDL idiom (`gdc_launch_dependents` then `gdc_wait`); `gdc_wait` returning
means the pooler completed, which means *its* wait returned, which means `_boundary1` completed — the chain
property `_boundary2`'s own comment (`:10899`) already relies on.

Gate: `T > 8`, o-GEMM `S == 1`, `ppick in ("bfp","bfpf")` and `bfp` present (so the pooler provably returns
`S == -1`, i.e. `_route_split` is the kernel that runs) — see `:11599`. The self-check skips any T whose
`compose_graft_enabled` is set (that chain replaces these kernels).

## 4. Ranking by (us/row x probability)

| candidate | us/row | p | why |
| --- | --- | --- | --- |
| widen `_boundary1`/`_boundary2` to 4-8 CTAs/token, cross-CTA row sum | **2.5** | 0.5 | 3 MB at 3 TB/s instead of 0.85 TB/s. Not a launch deletion; already shipped in two forms (`gluev2` lost, `_boundary_kernel` Gluon cluster in the graft chain, swept at T=32 by `_policy_variant` with a 0.4 us guard that discards sub-0.4 us wins). **Not mine.** |
| **s1 -> route** (this change) | **0.6-0.8** | 0.4 | move the 1 MB `s1` store off the pre-pooler launch onto a 160-CTA launch; pays 1.5 MB of re-reads. Unique feasible epilogue absorption. |
| top-8 -> pooler epilogue | 0.1-0.4 | - | measured and killed |
| `_boundary2` -> unpooler epilogue | <=0 | 0.05 | `_bfp_gemm` is a proven component; a Triton replacement costs +weights |
| o -> boundary1 (cluster) | ~0 | 0.1 | replaces `_bfp_gemm("o")`, +4.2 MB weights |

**Expected 0.6-0.8 us/row = 0.3-0.4% of the 186 us d32 row** (~1.3-1.7% of the 10.9 us boundary/route group).
Not 3.1% of the score: the group is occupancy-bound, and the occupancy fix already exists under the graft flag.
The honest read of this item is that it is worth at most ~0.4% unless the graft chain is re-armed at T=32.

## 5. Gates

* `ast.parse`: ok.
* lint: **delta 0**. 31 tier-0 errors first (banned `os.environ` for the flag), 30 after switching the flag to a
  plain module constant — identical to the board file's 30, modulo one line-number shift of the pre-existing
  140927-char line.
* `cu_build --variant MID` and `--variant MAIN`: **identical PASS/FAIL sets** to the board file (all 9 CUDA
  sources fail identically in this container: `nv/target`, `cuda/std/type_traits`, `tvm/ffi/container/tensor.h`
  and the `cudaGetLastError` host shim are all missing from the wheels; the baseline fails the same way).
  `cu_build` compiles only the `*_CUDA_SRC` strings, so it does not cover the two Triton kernels this change
  edits, and torch is not installed in this container, so the prepare-time self-check could not be exercised
  here. First real evidence must come from a run that reaches `prepare`.
