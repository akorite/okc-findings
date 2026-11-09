# bnd2: make the boundary/route occupancy paths reachable and self-verifying

File: `src/okc_loop/vendored_work/layer/klr_bnd2.py` = `klr_bundle2.py` + one appended block (`:16497-16604`).
No new kernel: the change selects between chains the file already ships.

## 1. Re-arm points

### `gluev2` -- `state["gluev2"]`, the parallel boundary/route chain
* Read at `_after_attention`:11558, consumed at three sites -- `_gemm_o` + `_bnd1_apply` (o projection with
  per-token partial sums + a 256-CTA `(T, 4)` boundary, :11561-:11571), `_route_a` + `_route_b`
  (`(T, 4)` route, :11623-:11630), `_bnd2_sums` + `_bnd2_apply` (`(T, 8)` boundary, :11650-:11656).
* Picked in `_tune`: `st["gluev2"][T] = bestv2` (:11189) out of a loop whose candidate list is
  **`for v2 in (False,)`** (:11165), i.e. the A/B is compiled out and `bestv2` is always `False`. The only
  surviving record is that loop's comment -- *"Parallel boundary search lost every complete-chain sweep"* --
  **no per-T measurement survives**, so we cannot say at which T it lost or by how much.
* Structural reason it lost, from the code: the v2 branch **ignores `mm_pick[(T, "o")]`** and always runs its own
  Triton `_gemm_o`. At T <= 32 `prepare` pins `mm_pick[(T, "o")] = "bfp"` (:12698, and again for the graft's
  `T <= 16` at :12557), and bf16_proj's 12-bit lossless o GEMM is measured 4-9% ahead of the Triton split-K. So
  v2 was A/B'd against a strictly better o projection and had to win the boundary back out of that deficit. That
  is a real reason for it to lose, and exactly the kind of thing that can flip when the rest of the chain changes.

### the graft chain -- `_graft_after_attention` / `_graft_boundary`
* Dispatcher gate: `:12524` `if "graft" in state and T in _GRAFT_GLUE_T and
  state.get("compose_graft_enabled", {}).get(T, T in (1, 8, 128, 256))`; `_GRAFT_GLUE_T = (1,8,16,32,128,256)`.
* `_graft_boundary` (:12409) picks the width by shape, not by a flag:
  | shape | kernel | width |
  | --- | --- | --- |
  | `T <= 16` and bf16 | `_graft_glue_module().b1` / `.b2c` / `.b2s` = `glue_cluster_boundary` | **8-CTA cluster** per row |
  | `16 < T <= 32`, width 8192 | `_graft_row_boundary_pre[(T,)]` | 1 CTA/row, **32 warps** |
  | `T > 32`, width 8192 | `_graft_row_boundary[(T,)]` | 1 CTA/row, 32 warps (16 above T > 192) |
  and the route becomes `_sl_tile_route[(T, 2)]` (program `(t,0)` pre-expert norm, `(t,1)` top-8, `PRELOAD=True`)
  instead of the base `_route_split[(T,)]`.
* Knobs: `state["compose_graft_enabled"][T]` (bool) and `state["compose_boundary_warps"][T]` (warp count for the
  *base* `_boundary1`/`_boundary2` fallbacks, :11580/:11640). `st["graft"]` (the gate/weight tensors) is built at
  :12572, before every sweep that runs later.
* **The 0.4 us guard**: the compose sweep at :12747-:12761 evaluates `modes = [(orig,16,()), (True,16,()),
  (False,16,()), (False,8,()), (False,4,())]` per T with `_compose_times(..., reps=32)` and then
  ```python
  best = min(range(len(ts)), key=ts.__getitem__)
  if ts[best] >= ts[0] - .4: best = 0        # :12758
  ```
  So graft **is** reachable at T=32 today, but a real win smaller than 0.4 us is discarded, and the rule has no
  output check and no win-rate check -- a 0.5 us median fluke on 32 interleaved rounds is enough to switch the
  chain for the whole run. That is the actual defect: not that the path is unreachable, but that the decision is
  taken on a median with a 0.4 us noise floor and no correctness gate.

## 2. What bnd2 does instead

Per T, three variants of the **same state dict**, differing only in the two knobs:

| variant | knobs |
| --- | --- |
| A (incumbent) | whatever `_tune` + the compose sweep + the gle/sl pins left |
| B1 graft | `compose_graft_enabled[T] = True` |
| B2 parallel | `gluev2[T] = True` |

and one `bnd2` print per T:
`bnd2 T32 base=… graft=… par=… us wins=[…] -> base|graft|par`.

### Timing idiom (identical to `_compose_times`, :12668)
1. synthetic `data["stream"]` and `o_in`, `torch.Generator(manual_seed(2718281))`;
2. route the synthetic batch to its **busiest EP range** (`local_start = 32*argmax(bincount(ids//32))`) so no
   variant benchmarks idle experts -- same line as the compose sweep;
3. one warmup call per variant, `torch.cuda.synchronize()`;
4. `torch.cuda.CUDAGraph()` capture per variant (so launch overhead is the graph's, not the host's);
5. 32 rounds, each round replaying all three in alternating order with a 256 MB `torch.sum(flush)` before every
   replay (cold L2), one `cuda.Event`-pair per replay, a single `synchronize()` at the end.

The only addition to `_compose_times` is that the round structure is kept: per-round times are grouped by round
index, so a variant's **round-win count** (`argmin` within the round) is available alongside its median.

### Decision rule (`_bnd2_choice`)
Take variant `k = argmin(median)` only if **all** of:
1. `k != 0`;
2. **correctness**: `torch.equal(ids_k, ids_A)` and rel-L2(`out`) <= 1e-5 and rel-L2(`s2`) <= 1e-5 -- the file's own
   bound for a path that only reassociates the residual stream (the boundary kernels write an FP32 residual and a
   BF16 normalized activation, so 1e-5 on both is the right bar, not the problem's 1.5e-2);
3. **margin**: `median_k <= median_A - max(0.04 * median_A, 3.0)` us;
4. **stability**: `wins_k >= 0.75 * 32` rounds.

Otherwise the incumbent stays. On acceptance only the corresponding knob is written (`compose_graft_enabled[T]`
or `gluev2[T]`); nothing else in the state is touched.

### Never fails a run
`prepare` wraps the whole sweep in `try/except` and each T in its own `try/except`, printing `[bnd2] FAILED:` /
`bnd2 T{n} skipped:` and leaving the incumbent chain in place -- the same idiom as bundle2's own
`_selftune_prior_prepare` wrapper. `BND2_PICK = False` disables it entirely.

## 3. Gates

* `ast.parse`: ok.
* lint delta vs `klr_bundle2.py`: **0** -- both files report the same 30 tier-0 errors, line for line (the block is
  appended after the last offending line, which is why even the line numbers match).
* `cu_build --variant MID` and `--variant MAIN`: **set difference vs bundle2: none** -- the same 9 `*_CUDA_SRC`
  verdicts in both. All 9 fail identically in this container (missing `nv/target`, `cuda/std/type_traits`,
  `tvm/ffi/container/tensor.h`, the `cudaGetLastError` host shim, and nvrtc frontend rejections in the bfp
  sources that hit the baseline too), so the tool discriminates nothing here; it also does not compile Triton,
  which is why the block adds no kernel and only re-launches existing ones.

## 4. Expected effect

Nothing by construction -- the block can only pick a chain that is >= 4% (>= 3 us) faster on a cold-L2 median,
wins 75% of rounds and reproduces the incumbent bit-for-bit within 1e-5. If the occupancy paths are still behind,
the six printed lines say so per T and the run is unchanged; the value of the change is that the two occupancy
chains stop being decided by a 0.4 us median with no correctness check, and that one judge check now produces the
per-T numbers (`compose_selection`'s cousin) that the earlier "lost every sweep" note never kept.
