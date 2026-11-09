# klr_l2p: a persisting-L2 carve-out — **FALSIFIED, do not develop** (see §4)

> **Verdict: dead.** The premise this was built on — "the 62 layer rows of a replay re-read the same
> weights, so a pinned slice stops being HBM traffic 61 times over" — is wrong for this problem. The timed
> call is **one layer step** (`problems/muse_decode_layer/TASK.md`: "Everything one GPU does for one
> layer"), the 46/16 figures are *accounting weights* on that single-step time (they are the board assembly
> model in `src/okc_loop/scoring.py:196-231`), and the judge flushes L2 with a read larger than L2 before
> **every** replay. So there is no cross-step reuse for a carve-out to capture: expected gain **0**, and the
> carve-out would still take 16-32 MB of the 50 MB L2 away from the streaming side — a small loss.
> Sections 1-3 and 5-8 are kept as the record of the mechanism and the gates; §4 has the falsification and
> §4b the question it exposes (is there *intra*-step reuse worth pinning? answer: not in the weights).

Candidate: `src/okc_loop/vendored_work/layer/klr_l2p.py` (16,446 lines). Base: `klr_qd2.py` (board
`82c098ae` + ATT7 QDIRECT/PK2, 16,345 lines, measured **2.2536**). Line 1 and the provenance header are
untouched, so `okc lint` still sees `tier0 muse_decode_layer`.

**Switch is off by default**: `L2P_MB = 0` (`klr_l2p:16394`). `32` enables the whole thing. Nothing
else changes; the print below says which state the judge ran in.

## 1. Route: the PTX policy, not the stream window. Why, in CUDA's own words

* A stream's access-policy window is **host-side stream state**. CUDA Programming Guide §4.13.2, "L2
  Policy for Persisting Accesses": *"The code example below shows how to set an L2 persisting access
  window using a CUDA Stream … `cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow,
  &stream_attribute)` … When a kernel subsequently executes in CUDA `stream`, memory accesses within the
  global memory extent `[ptr..ptr+num_bytes)` are more likely to persist."* It is a property of *that
  stream's later kernel launches*, and the judge captures our `kernel_fn` and replays the graph on **its
  own** stream, at a time when no host call is allowed.
* That a stream attribute does **not** travel inside a captured graph is exactly why NVIDIA documents a
  second, node-level API for the same effect — §4.13.2 again: *"L2 persistence can also be set for a CUDA
  Graph Kernel Node … `cudaGraphKernelNodeSetAttribute(node,
  cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute)`"*. Two different mechanisms means the
  stream one is not baked into the graph; and patching the judge's graph nodes is not available to us
  (the graph does not exist until the judge builds it, and host work inside the timed region is banned).
* So the hint must live **in the kernel's code**: `createpolicy.fractional.L2::evict_last.b64` +
  `.L2::cache_hint` on the weight `cp.async.bulk`. That is per-instruction PTX, it is inside the SASS of
  the node, and it travels with the graph. This file already relies on it (`bfp_MBAR_HELPERS`
  `bulk_g2s` uses `createpolicy.fractional.L2::evict_first` for every SK2/wide weight copy).
* The **carve-out** is a different kind of state and *does* survive: `cudaLimitPersistingL2CacheSize` is a
  context limit (§4.13.8: *"queried using `cudaDeviceGetLimit` and set using `cudaDeviceSetLimit` as a
  `cudaLimit`. The maximum value for setting this limit is `cudaDeviceProp::persistingL2CacheMaxSize`"*),
  set once in `prepare` and in force for every replay afterwards. §4.13.1 gives the guarantee the whole
  idea rests on: *"Persisting accesses have prioritized use of this set-aside portion of L2 cache, whereas
  normal or streaming accesses to global memory can only utilize this portion of L2 when it is unused by
  persisting accesses."* — i.e. the MoE's `evict_first` ~900 MB/row cannot evict the pinned lines, and
  when nothing is pinned the carved-out bytes are not wasted.
* Two documented ways this can be a no-op, both detected and printed at prepare: §4.13.1 *"When the GPU
  is configured in Multi-Instance GPU (MIG) mode, the L2 cache set-aside functionality is disabled"*, and
  *"When using the Multi-Process Service (MPS), the L2 cache set-aside size cannot be changed by
  `cudaDeviceSetLimit`"* (only at MPS-server start, via
  `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT`). The judge's `persistingL2CacheMaxSize` is
  read with `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108)` at prepare and
  printed, because it is the only machine that matters and it is not a constant (the guide's own idiom is
  `min(l2CacheSize * 0.75, persistingL2CacheMaxSize)`).

## 2. Pointer map: what a row actually reads, and what is pinned

Every dense cell at T ≤ 32 runs an `sk2` launch on a **stage-major copy** of the 12-bit `bfp_COMPRESS`
stream (`bfp_sk2_compress` → `bfp_sk2_stream`), not on the raw weight: `[stage][CTA][warp][unit][tile]
[768 B]`, i.e. `(n_rows/8) * (K/64) * TU` bytes, `TU = 768` (COMP) or `2048` (router hi|lo). These are
`torch.Tensor`s built once in `bfp_sk2_prepare` (`klr_l2p:9724`, called from the single prepare-time state
build at `:9891`), stored in `state["sk2"][name] = (ci, stream, tab, pmax, b4, n_rows)`, and **never
written again** — contiguous and pointer-stable for the whole run.

| tensor | stream bytes | reads per step | fits a 24–40 MB carve-out |
| --- | --- | --- | --- |
| `qkv_gate` (2304 × 8192) | **28,311,552 (27 MiB)** | one per row, T ≤ 32 | **yes — pinned** (`qkv8_r176` / `qkv16_r176` / `qkv32_r208` all share one stream: same `lkey`) |
| `o` (8192 × 1024) | 12,582,912 | one per row | yes, but half the bytes |
| `pooler` (4096 × 8192) | 50,331,648 (48 MiB) | one per row | no |
| `unpooler` (8192 × 4096) | 50,331,648 (48 MiB) | one per row | no |
| `router` (256 × 8192, hi+lo) | 8,388,608 | one per row | yes, but 3.4× fewer bytes |

Total 149.9 MB (143 MiB) of dense stream per row — the ticket's 112–150 MB. Pinning **qkv_gate** takes
the largest chunk that fits under the 32 MB knob, and it is the same tensor for all three T ≤ 32 configs,
so one pointer covers decode 1/16/32 and verify 1.

Exact range to pin: **base = that stream's `data_ptr()`, offset 0, length 28,311,552 B** — the whole
buffer, because the kernel reads it as `Ws + cta * WBY + (gs0 + st) * wstride` over every stage, so all
of it is touched every row. The base address is whatever the caching allocator returns for that tensor
(it is `Ws` as passed to `ext.sk2(...)`, printed at prepare), which is why it is handed to
`sk2_set_l2p()` by value instead of being hard-coded, and why the tag is a pointer comparison rather
than a fixed address.

Not covered: **verify 16 / verify 32 (T = 128/256)** dispatch `qkv` to the `rq` pack instead
(`RQ_USE = {('un',128),('qkv',256),('qkv',128)}`), whose kernel is the closed `_CHEN` extension with no
source we control — there is no policy hook there. So the pin pays on 4 of the 6 scored loads.

## 3. Implementation (all behind the switch, all in the file's own idioms)

* Carve-out: `_l2p_carveout()` (`:16400`) — `cuCtxGetDevice` → `cuDeviceGetAttribute(108)` for the device
  max → `cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 6, min(L2P_MB << 20, max))` →
  `cuCtxGetLimit` to report what actually took effect. ctypes via `_C.CDLL("libcuda.so.1")`, the idiom the
  file already uses for its context bind. Any failure (no libcuda, MIG, MPS) returns `None` and the run
  continues with hints only.
* Tag: `bulk_g2s_pol(dst, src, bytes, bar, pol)` (`:7323`) next to `bulk_g2s`; in `sk2_kernel`
  (`:8123`) `wpol = l2_pol_ef(); if (l2p) wpol = l2_pol_el();` and the two weight copies
  (`:8158` prologue, `:8209` ring) use it. **The router piece and every other `bulk_g2s` stay
  `evict_first`, and the MoE is untouched** (`bfp_MBAR_HELPERS` only *gains* a function).
* The per-launch "is this the pinned stream" test is a **pointer compare in the host launcher**:
  `static const uint8_t* g_l2p_w` (`:8047`) + `void sk2_set_l2p(int64_t)` (`:8048`), and in `launch_sk2`
  (`:8622`) `const int pdl = g_sk2_pdl | (wsp == g_l2p_w ? (1 << 20) : 0)`. Bit 20 of the existing `pdl`
  already-there int carries it, so **no kernel signature changes** — which is what keeps the two text
  surgeries in this file that touch `sk2_kernel` (`_de_source`, `_gle_source`) working. Bit 20 is masked
  everywhere `pdl` is read (`pdl &= 0xFFFFF` in the kernel; `(pdl & 0xFFFFF)` at the two host tests), so
  the PDL/PCX behaviour is bit-identical to today. The plain `<<<>>>` path now passes `pdl` instead of a
  literal `0` — identical while bit 20 is clear, which is the default (`g_l2p_w == nullptr`).
* `sk2_set_l2p` is exported from both sk2 modules (main `bfp__get_ext` `:9398` and the narrow decode one
  `_de_ext` `:13404`), and the appended `prepare` wrapper (`:16427`) calls it once with the pinned
  stream's `data_ptr()`.
* Print at prepare (only channel we have for judge-side facts):
  `[l2p] carve-out <set> B of max <max> B; pinned qkv8_r176 28311552 B contig True at 0x…` — or
  `[l2p] off (L2P_MB=0)…`, or the two no-op warnings from §1.
* Numerics: unchanged by construction. `evict_last` is a retention hint; `createpolicy` produces a policy
  register, not arithmetic; the copied bytes, the decode, the escape corrections and the reduction order
  are untouched.

## 4. FALSIFIED: the premise is wrong — there is no cross-step reuse in this problem

What the priced claim assumed: a replay runs the layer 62 times, each re-reading the same weights, so a
pinned 27 MiB stops being HBM traffic 61 times over (28,311,552 B x 61 / 4.3 TB/s = 401.6 us per replay =
6.48 us per row). Both halves of that are false here:

1. **One replay is one layer step, not 62.** `problems/muse_decode_layer/TASK.md`: *"Everything one GPU
   does for **one layer** of Muse Spark Open under TP8/EP8"*, and *"A shape's weight is how many times one
   step runs it. A load's time is the sum of its shapes' times, each times its weight."* The 46 sliding /
   16 full multiplier is how the **overall board** assembles per-step times into a per-model figure
   (`src/okc_loop/scoring.py:196-231`); it is an accounting weight, not 46 replays of our graph. Inside one
   replay each weight byte is read exactly once. That module says it itself: *"Every argument is a load
   time, i.e. already ``sum(weight_i * t_i)`` over that problem's benchmark shapes. Those weights are the
   layer counts themselves (62 dense, 60 MoE, and 46/16 for sliding/full attention), so the three load
   times add directly: nothing here multiplies by a layer count a second time."*
2. **The judge flushes L2 before every replay.** TASK.md, *"How a run is timed"*: *"Every timed replay
   starts with a cold L2: the judge reads a buffer larger than L2 just before it."* So even two consecutive
   replays of the same shape cannot share a line: the pin would find an empty cache every time and would
   re-fill it from HBM — the same traffic as today — while permanently reserving 16-32 MB of L2 that the
   streaming side (`evict_first`/normal) can only use "when it is unused by persisting accesses"
   (guide §4.13.1). Expected gain 0, small expected loss. Nothing to measure, nothing to ship.

What a step actually moves (bytes, from the shapes; `HIDDEN 8192`, `D_EXPERT 4096`, `N_EXPERTS 256`,
32 local experts, top-8, KV FP8 with `NUM_KV_HEADS 2` x `HEAD_DIM 64` x K,V = 256 B/token, sliding window
2048 tokens):

| shape | dense 12-bit | MoE 4-bit | KV | activations | per step |
| --- | --- | --- | --- | --- | --- |
| decode 1, sliding | 149.9 MB | <=8 routed experts: <=201 MB (task's own skew: ~4-6 distinct, ~100-150) | 0.5 MB | 0.06 MB | **~250-350 MB** |
| decode 1, full | 149.9 | same | 20.5 MB | 0.06 | **~270-370 MB** |
| decode 16, sliding | 149.9 | all 32 local: 805 MB | 8.4 | 0.5 | **~965 MB** |
| decode 32, full | 149.9 | 805 | 655 | 1.0 | **~1.6 GB** |
| verify 32, full | 149.9 | 805 | 655 | 16 | **~1.63 GB** |

(`dense`: qkv 28.31 + o 12.58 + pooler 50.33 + unpooler 50.33 + router 8.39 MB. `MoE`: (2x4096x4096 +
4096x4096) weights = 25.17 MB per expert at 4 bits, x32 local, plus ~1.5 % scales. The floor-derived lower
bounds agree in magnitude: verify 32 full floors at 267.2 us x 4.8 TB/s = 1.28 GB.) That is one pass over
250 MB - 1.6 GB per replay, all of it cold by protocol; the earlier "112-150 MB dense + ~900 MB MoE" figure
was the same arithmetic and it is a *single* pass, which is exactly why nothing can be amortized.

## 4b. The question this exposes: what is read twice *inside* one step, and can L2 serve it?

All five dense cells, the attention stream and both MoE sweeps are single-pass. The only repeat reads in a
step are activation-side, and they are small:

| range read twice within a step | bytes (T = 1 / 256) | read by | L2 can serve it? |
| --- | --- | --- | --- |
| the 5 dense weight ranges (qkv 28.3, o 12.6, pooler 50.3, unpooler 50.3, router 8.4 MB) | 149.9 MB | each CTA a disjoint row-block (`wsrc = Ws + cta*WBY`), once per step | **No reuse at all.** Tagging any of it serves nothing. |
| `x` tiles inside each GEMM (activation side) | 16 KB / 4.2 MB | every CTA of that GEMM (qkv 96, o 256, unpooler 128 CTAs) -> up to ~400 MB of traffic at T=256 | **Yes, and already done**: the TMA x loads carry `createpolicy.fractional.L2::evict_last` (`l2_pol_el()`), and 4.2 MB needs no carve-out out of 50 MB. |
| attention KV | 0.5 MB / 655 MB (distinct) | each request's pages once; the 8 q-heads sharing a KV head (16 q / 2 kv) consume the same tile from smem inside one CTA, not through L2; FA3's tile scheduler gives different CTAs disjoint KV | **No.** The 8x head reuse never reaches L2, and the distinct stream is read once. |
| router reads the pooler output, then the MoE's phase 1 reads it again | 8 KB / 2.1 MB | pooler -> router -> MoE first phase | **Yes, trivially** (normal LRU, 2 MB); a carve-out could only take space from it. |
| boundary kernels re-read the residual stream `s` | 32 KB / 8.4 MB (FP32 [T,8192]) | ~4-5 kernels per step | **Yes, 8.4 MB working set**, already partly tagged `evict_last` (`st_keep_f32`); no carve-out needed. |
| MoE `w13` / `w2` sweeps | 16.8 + 8.4 MB per routed expert | disjoint ranges, one pass per routed expert (expert-blocked CTAs) | **No** for the weights. The only cross-phase datum is `inter` (SwiGLU out, 8 KB / 2.1 MB) which `w2`'s phase reads after phase 1 writes it — resident, and already covered by the existing `L2H` stores. |

**Answer: no. There is no weight range in a single step that is read twice, so there is nothing for a
persisting carve-out to retain.** The only intra-step reuse is 0.5-8.4 MB of activations plus the `x`-tile
fan-out, all of which fits inside 50 MB with normal LRU and much of which the file *already* tags
`evict_last` (that is what `L2H` / `l2_pol_el` / `st_keep_*` are for, and they are the right size for a
hint, not for a reserve). A 16-32 MB set-aside is therefore pure cost on this problem.

## 5. Gates

* `ast.parse`: ok (409 top-level nodes).
* `okc lint` vs `klr_qd2.py`: **zero delta** — 12 errors in both, identical set, only the reported line
  numbers move (the added lines shift them); the long-line/`ctypes`/`globals` findings are the base's.
  Nothing new from the appended block (its `globals()` use was removed in favour of the plain `_DE_EXT`
  name).
* `tools/cu_build.py`: `MID` and `MAIN` both give the **same 9 FAILs** as the base, all of them
  pre-existing container artifacts (`int64_t`/`at::Tensor`/`tvm`/`nv/target` unresolved because the tool
  does not stitch the cpp header, and it does not compile the template kernels — `bfp_SK2_SRC` dies at its
  host wrapper `int64_t` in both files, so my device-code edits are *not* covered by this gate). One new
  benign line: `warning #550-D: variable "g_l2p_w" was set but never used`, an artifact of that truncated
  compile (the reader `launch_sk2` is past the failure point).
* Because cu_build cannot see the edited device code, the two asm forms were compiled on their own through
  the same nvrtc/ptxas (sm_90a): `createpolicy.fractional.L2::evict_first/evict_last.b64` both appear as
  `%rd12`, both `cp.async.bulk … .L2::cache_hint` copies emit, `ptxas` accepts them, 4 registers, 0 spills.
  (Note for the next user of `tools/cu_build.py`: any source with `#include <stdint.h>` renders an *empty*
  PTX under its option set — that is the tool, not the source.)

## 6. What to run on the judge — **moot, kept for the record**

Enable with `L2P_MB = 32` and check the `[l2p]` line; then A/B `klr_qd2.py` (off) against it on the
paired loads. If the carve-out itself looks costly (it shrinks the L2 the MoE streams through from 50 MB
to ~18 MB), `L2P_MB = 28` is the tightest setting that still covers the 28,311,552 B stream exactly —
worth one A/B point, since nothing else in the file trades that resource.

Expectation to falsify: decode 1/16/32 and verify 1 each drop by ~6.5 µs per row
(≈ 6.5/row-time, so 3–16 %), verify 16/32 unchanged. If `carve-out` prints 0 the box is MIG/MPS and the
experiment says nothing.

## 7. Self-gating (`klr_l2p32.py`, L2P_MB = 32) — **moot, kept for the record**

The pin is a **runtime** compare (`wsp == g_l2p_w` in the host launcher) and the carve-out is context state,
so one build can be timed both ways. `klr_l2p32.py` measures, decides, and then leaves the context and the
tag in the winning state — and if anything fails, it releases both, so the run is exactly today's behaviour.

**Why the arms cannot interleave per round.** Two facts force a block structure:

1. A captured graph carries the tag or not *for its whole life*: `launch_sk2` folds bit 20 into the `pdl`
   kernel argument at capture time, so the on and off arms need their own graph (12 graphs total, captured
   once per spine, before any timing).
2. The on arm's benefit only exists while the pinned lines are already resident when a replay starts, and
   releasing the carve-out needs `cudaCtxResetPersistingL2Cache` (guide §4.13.5: a persisting line survives
   until an explicit reset; setting the limit to 0 alone does not demote it). A per-round
   on/off/on/off interleave would therefore wipe the pin at every off round and measure *zero* benefit.

So: **blocks of one untimed warm-up round + `L2P_REPS`=4 timed cold-flushed rounds per arm, arm order
`(on16, on32, off)` repeated `L2P_BLOCKS`=4 times** → 16 timed rounds per arm, and the state transition
(`cuCtxSetLimit` / release + reset) happens once per block boundary, outside the timed window. The block
order cycles, so clock drift is shared between the arms. Between `on16` and `on32` there is no reset (both
keep the pin; only the set-aside size changes), so no round sees a wiped pin. A graph captured under one arm
replays correctly under the other: the access policy is a hint on the load instructions and the limit/reset
are context state — neither is validated against the graph.

**The spine** is this file's own launches, in the row's order: qkv sk2 (pinned) → the production MoE ext with
`_mf_gen_input` routing → o sk2 → unpooler sk2 → fused pooler+router, with a 256 MB `flush.add_(1)` between
rounds as the cold-L2 sweep (it can only use the non-persisting part of L2, which is exactly the asymmetry
being measured). Attention is not driven (it needs the judge's KV pages): it is the row's biggest stream but
has no intra-row reuse, so its absence understates the interference, not the effect. The MoE is only used if
prepare already built an ext for it (`_MA_EXTS` lookup, no compile); the line says `moe=yes|no`.

At T = 128/256 the spine **drops the qkv launch**, because the judge reads qkv from `rq` there (unpinned), so
those two spines measure the carve-out's cost alone; they are gated on cost (`≤ L2P_COST_US` = 0.5 µs), not
on gain.

**Gate** (per spine `pick` = the smaller median of 16/32 MB; the decision is global because the tag is frozen
at the judge's capture, so one state has to serve all 12 shapes): median gain over the four pinned spines
≥ `max(L2P_MIN_US = 1.5 µs, 2% of that shape's floor)`, paired round-win rate ≥ 75%, and unpinned cost
≤ 0.5 µs. The 2% test is taken against the task's published floors (40.0/62.3/76.7/84.5 µs), not against the
spine's own time — the spine runs the MoE at bs 32 (the full local expert set, the worst-case interference),
so it is several hundred µs and a 2%-of-spine threshold would be unattainable by construction.

Lines it prints (one per token count, covering the 12 scored shapes; `sliding`/`full` share the spine since
attention is not in it):

```
[l2p] persisting max <n> B; target qkv8_r176 28311552 B contig True at 0x…
[l2p] spine: qkv/o/unpooler sk2 + fused pooler+router, moe=yes
[l2p] T=  1 decode 1 B1 L1 m=1 qkv=sk2 off=  701.0 on16=  697.1 on32=  694.5 us pick=32 wins=15/16 pin=yes gain=6.5us=16.2%of40.0
[l2p] T=128 verify 16 B16 L8 m=32 qkv=rq(unpinned) off=  828.0 on16=  828.1 on32=  828.1 us pick=- wins=0/16 pin=no
[l2p] AB verdict: on, carve-out 33554432 B (median gain 6.5 us = 9.1% of the shape floors, wins 94%, unpinned cost +0.1 us)
```

**Worst case**: if the carve-out costs more than the pin saves, the pinned spines show no median gain / a
losing round rate, the verdict is `off`, and the file then sets the limit back to 0, clears the tag and resets
the persisting lines — the same state a plain `klr_qd2.py` run has (the judge's graphs are captured *after*
prepare, so they carry no bit 20). Same on any exception, a missing `sk2` stream, a spine that will not
capture, or a device with no persisting-L2 carve-out at all (MIG/MPS; the `persisting max` line reports it).

Gates for this file: `ast` ok; `okc lint` **30 error(s) / 72 more, identical to `klr_qd2.py`** (no new
finding); the two asm forms still nvrtc/ptxas-clean for sm_90a (4 registers, 0 spills).
