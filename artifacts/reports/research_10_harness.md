# research_10_harness — the judge's measurement regime, and what a submission may legitimately do with it

Scope: (1) CUDA-graph capture semantics usable by a submission (PDL, node attributes, conditional nodes,
graph-vs-eager dependencies); (2) cold-cache effects after the per-replay L2 flush, and the algebra of the
fixed ~4.1 µs per-replay subtraction; (3) what `prepare()` can precompute that survives into the captured
graph; (4) how the organizer's rerun median interacts with a kernel tuned on the exact scored shapes.

Every external claim has a URL. Internal numbers are cited to our files/reports. `[DERIVED]` = my
inference from the cited material, not a measured fact. `[UNVERIFIED]` = asserted by the ticket or a
third party and not reproduced here.

---

## 0. Verdict

| question | answer | evidence |
|---|---|---|
| Is PDL legal inside a captured graph? | **Yes, CUDA 12.3+.** Stream capture translates the launch attribute into a graph edge (`cudaGraphDependencyTypeProgrammatic` + `cudaGraphKernelNodePortProgrammatic`). | CUDA Prog. Guide §4.5.3 (URL below) |
| Does capture bake the ordering or re-evaluate it? | **Ordering is baked at capture/instantiation; the overlap is dynamic at each replay.** The edge says "this kernel is visible to `cudaGridDependencySynchronize()`"; whether CTAs actually overlap is "opportunistic and not guaranteed". | §4.5.2, §4.2.2, `cudaGraphKernelNodePortProgrammatic` |
| Can a submission set PDL via `cudaGraphKernelNodeSetAttribute`? | **No documented form, and no need.** NVIDIA documents PDL in graphs only as edge data. The graph *is* reachable during capture (`cudaStreamGetCaptureInfo`), but the launch attribute already converts to the same edge, and nothing is mutable after Torch instantiates. | Runtime API graph group; Graphs §4.2.1.2–4.2.2; §4.5.3 |
| Conditional graph nodes? | **Not reachable from a stream-captured graph** (they exist only via the graph API / capture-into-a-body), they are evaluated **on device at node execution**, and the task rules forbid branching on input values anyway. | Graphs §4.2.4; `TASK.md:100` |
| Is a graph replay's kernel-to-kernel dependency the same as an eager launch's? | **For a captured single stream, yes:** the default edge is "standard full serialization of two nodes with memory visibility". But a graph may run any nodes whose edges are satisfied concurrently, and instantiation pre-does setup. | `cudaGraphEdgeData`; Graphs §4.2.1–4.2.2 |
| Does the L2 flush make code/constants cold every replay? | **Yes, in L2; L1 instruction/constant caches are per-SM and survive the flush but are capacity-thrashed by an 8-kernel binary.** Instruction and constant caches reach L2 through L1.5 and the i-cache has automatic prefetch (NVIDIA staff). | Volta microbenchmark paper; NVIDIA forum |
| Does the 4.1 µs subtraction make small kernels free? | **No — the opposite.** `r = max(0, T − k)` gives `∂score/∂T = −S·w/(6·C_load)` while `r>0`: absolute gain is 1:1, relative gain is amplified by `1/(T−k)`. At the clamp (`T≤k`) the derivative drops to exactly 0. | §2.4 below |
| Can `prepare()` workspace be reused across replays? | **Yes, if the address is stable and contents are scratch.** Replays reuse baked pointers; graph allocations get fixed VAs at node creation; freed/relocated memory requires a graph update or recapture. | Graphs §4.2.3/§4.2.5; PyTorch CUDA semantics |
| Do rerun medians punish exact-shape tuning? | **Yes.** Board score is one run; final positions are medians of organizer reruns on scored *and nearby* shapes (public competitor). Our own judge also always runs nearby + edge correctness cases. | `LEARNINGS.md`; `cases.py:199`; harness `_cases` |

---

## 1. CUDA graph capture semantics a submission can use

### 1.1 What capture and replay actually are

- Stream capture does **not execute** the work; it appends nodes to an internal graph
  ([Graphs §4.2.2.1.2](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)).
- Instantiation "takes a **snapshot** of the graph template, validates it, and performs much of the setup
  and initialization of work with the aim of minimizing what needs to be done at launch"; the executable
  graph "may be launched any number of times without repeating the instantiation" (§4.2.2).
- PyTorch: "the graph **reads from and writes to the same memory addresses in every replay**", capture must
  be on a non-default stream, no CPU syncs, and capture-time allocations come from a graph-private pool
  ([PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-graphs)).

**Consequence that matters more than any other:** the only candidate host code that runs on a timed replay is
`graph.replay()`. Python/`kernel_fn` host code, `torch.empty`, `.item()`, `tensor.zero_()`, descriptor
construction, cuBLASLt heuristic queries — all of these run at **capture time only** (3 eager warm-ups + 2
side-stream calls + 1 capture in our harness, `colab_harness.py:205-235`). Anything that must happen *per
replay* has to be a captured device node; anything computed per shape is free (and legal: "Branching on
shapes and `config` is fine", `TASK.md:100`).

### 1.2 PDL inside a captured graph — legal, and the ordering becomes topology

CUDA Prog. Guide §4.5.3, "Use in CUDA Graphs"
([URL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)):

> Programmatic Dependent Launch can be used in CUDA Graphs via stream capture or directly via edge data. …
> use a `cudaGraphDependencyType` value of `cudaGraphDependencyTypeProgrammatic` on an edge connecting two
> kernel nodes. This edge type makes the upstream kernel visible to a `cudaGridDependencySynchronize()` in
> the downstream kernel.

and the capture equivalence table on the same page:

| stream code | resulting graph edge |
|---|---|
| `cudaLaunchAttributeProgrammaticStreamSerialization = 1` | `type = cudaGraphDependencyTypeProgrammatic`, `from_port = cudaGraphKernelNodePortProgrammatic` |
| `cudaLaunchAttributeProgrammaticEvent` (`triggerAtBlockStart=0`) | same as above |
| `cudaLaunchAttributeProgrammaticEvent` (`triggerAtBlockStart=1`) | `from_port = cudaGraphKernelNodePortLaunchCompletion` |

So: **capture converts the attribute into a structural edge.** At replay the driver does not re-interpret
the launch attribute; it executes the graph. The port semantics are hardware-evaluated per replay:
`cudaGraphKernelNodePortProgrammatic` "activates when all blocks of the kernel have performed
`cudaTriggerProgrammaticLaunchCompletion()` **or have terminated**"
([Runtime API types, CUDA 13](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-runtime-api/group__CUDART__TYPES.html)).
`cudaGraphDependencyTypeProgrammatic` "allows the downstream node to use `cudaGridDependencySynchronize()`"
(same page).

What is dynamic, and what the docs are explicit about: §4.5.2 says the concurrent execution is
"**opportunistic and not guaranteed**" and "Reliance on concurrent execution in this manner is unsafe and
can lead to deadlock." §4.5.1 states the benefit plainly: "concurrent launch also allows us to **hide the
launch latency of `secondary_kernel` behind the execution of `primary_kernel`**."

Correctness rule (PTX ISA `griddepcontrol`,
[URL](https://docs.nvidia.com/cuda/parallel-thread-execution/)): `.wait` "causes the executing thread to
wait until all prerequisite grids in flight have completed and all the memory operations from the
prerequisite grids are performed and made visible"; "If the prerequisite grid is using
`griddepcontrol.launch_dependents`, then the dependent grid must use `griddepcontrol.wait`". The primary's
trigger is optional: "If the primary kernel doesn't execute the trigger, it implicitly occurs after all
thread blocks in the primary kernel exit" (§4.5.2). So adding the attribute to a secondary is **safe even
if the primary never triggers** — it only loses the overlap.

`cudaGraphKernelNodeSetAttribute(cudaGraphNode_t, cudaLaunchAttributeID, const cudaLaunchAttributeValue*)`
exists in the CUDA 13 runtime API ("Sets node attribute",
[URL](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__GRAPH.html)) and is used
in practice for e.g. node priority. Its second parameter is a `cudaLaunchAttributeID`, i.e. the same
attribute namespace as a launch. But NVIDIA publishes **no** kernel-node attribute for programmatic
stream serialization; the documented representation is edge data.

Can a submission reach the graph at all? During capture, yes: `cudaStreamGetCaptureInfo()` "allows the user
to obtain the status of the capture, a unique (per-process) ID for the capture, the **underlying graph
object**, and dependencies/edge data for the next node to be captured in the stream" (Graphs §4.2.2.1.2.4),
and `cudaStreamUpdateCaptureDependencies()` can attach edge data to the dangling edge (Graphs §4.2.1.2).
That is a valid way to build a programmatic edge without the launch attribute. It buys nothing over the
attribute (which capture already converts, §1.2), it runs at capture time only, and it must happen before
Torch instantiates the graph at the end of `torch.cuda.graph`. After instantiation, changes require
`cudaGraphExec…` update or re-instantiation (§4.2.3), and a submission has no hook then.

**Operational conclusion:** PDL already in `klr_fci5.py` (`cudaLaunchAttributeProgrammaticStreamSerialization`
at :3606/:6849/:8625/:11882/:12088/:13022; `griddepcontrol.wait` at :607/:6257/:8170/:11820/:12947) is
**not** silently dropped by graph capture. It becomes an edge and works on every replay. Our PDLKV
experiment did not test this mechanism: it tested reordering copies *around* an existing wait, and measured
~0 (1.5–4% on B=1 cells, flat elsewhere, and the three runs' score deltas were within reference drift —
`pdlkv.md:130-160`). Adding PDL to boundaries that *lack* it is a different, untested move.

### 1.3 Conditional graph nodes

- "Evaluation of the condition value is performed **on the device** when the dependencies of the
  conditional node have been met" (Graphs §4.2.4). Types: IF / WHILE / SWITCH.
- The default can be re-applied "at the beginning of each graph execution" only if
  `cudaGraphCondAssignDefault` was passed; otherwise the value is undefined at the start of each launch
  (§4.2.4.1).
- They are created with `cudaGraphConditionalHandleCreate()` + `cudaGraphAddNode()`; their body graph can be
  populated with the graph API or `cudaStreamBeginCaptureToGraph()`, but **there is no stream API that
  creates a conditional node** (§4.2.4).
- Body graphs forbid CUDA Dynamic Parallelism and Device Graph Launch; cooperative launches are allowed
  (§4.2.4.2). Conditional-node parameters cannot be changed by `cudaGraphExecUpdate` beyond handle order
  (§4.2.3.4).

So a Torch stream-captured submission cannot contain one, and `TASK.md:100` ("no … branching on input
values") forbids the only use that would be interesting. `cudaGraphNodeSetEnabled()` (§4.2.3.3) likewise
needs a host graph handle and pre-instantiation, which a submission does not have.

### 1.4 Graph dependencies vs eager launches

- "An operation may be scheduled at any time once the nodes on which it depends are complete. **Scheduling
  is left up to the CUDA system**" (Graphs §4.2.1).
- A zero-initialised edge is "a standard full serialization of two nodes **with memory visibility**"
  ([`cudaGraphEdgeData`](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/structcudaGraphEdgeData.html)).
- Capture of one stream therefore yields a chain equivalent to same-stream eager ordering, but the graph
  *may* overlap any nodes whose edges permit it (only reachable if you build edges yourself), and
  instantiation removes most per-launch setup (§4.2.2).
- PDL is the only documented way a captured graph is *weaker* than eager same-stream ordering: the
  downstream node may start early and must fence itself with `griddepcontrol.wait`.

---

## 2. Cold-cache effects that are legitimate to optimise

### 2.1 What the flush evicts

The judge reads a buffer larger than L2 before every replay (`TASK.md:93`; ticket says ~1 GiB read). Our
replica instead zeroes 256 MiB (`colab_harness.py:186,223`) — both exceed the 60 MB physical / ~39.5 MB
effective H200 L2 (`research_06_microarch.md` §0), so both are full flushes; the difference is write vs read
backing, not coverage.

Measured hierarchy (Volta microbenchmarking, arXiv:1804.06826,
[PDF](https://arxiv.org/pdf/1804.06826), §3.3):

| level | size | scope |
|---|---|---|
| L0 instruction cache | 12 KiB | per processing block |
| L1 instruction cache | 128 KiB | per SM |
| L1 constant cache | 2 KiB | per SM |
| L1.5 constant cache | >64 KiB | per SM; **coincides with the L1 instruction cache on Volta** |
| L2 | 6 MiB (V100) | **unified: data + constant + instruction**, shared by all SMs |

NVIDIA staff (Greg@NV), on the same hierarchy
([forum](https://forums.developer.nvidia.com/t/instruction-cache-and-instruction-fetch-stalls/76883)):

> Instruction and constant caches access L2 through the GPC L1.5 cache but all local/global/texture/surface
> access go through the unified L1/TEX cache to the L2 cache. … **The instruction cache supports automatic
> prefetch.**

`[DERIVED]` Consequences for our 8-kernel layer graph:
1. The L1 instruction cache is **not** flushed by the buffer read, but our graph's total code is far larger
   than 128 KiB per SM, and 8 kernels execute back-to-back on the same SMs. Each kernel's entry re-fetches
   code that the previous kernel evicted, from L2 — which the flush just emptied. So **every timed replay
   pays cold instruction fetch per kernel per SM**, not just the first.
2. The 5 warm-up replays do not help: each is followed by the same flush (`colab_harness.py:220-235`), and
   the timed loop flushes before every sample too. There is no "first replay is slow" bias to exploit.
3. The flush evicts constant-bank backing as well; the constant hierarchy's L1.5 level shares the L1
   instruction cache on Volta, so large `.const`/param footprints compete with code for the same per-SM
   window (`[INFERENCE]` for Hopper; geometry is not published for Hopper).

### 2.2 Kernel parameters and the constant bank

- "CUDA kernel function parameters are passed to the device **through constant memory** and have been
  limited to 4,096 bytes. CUDA 12.1 increases this parameter limit … to 32,764 bytes"
  ([NVIDIA blog](https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/)). On H100,
  replacing a large explicit constant copy with the same data passed as kernel parameters gave a **9%
  improvement in kernel execution time** (same blog, Fig. 2).
- PTX: `.param` is a per-grid read-only state space accessed only via `ld.param`; its location is
  implementation-specific, and constant space is a separate 64 KB + 10×64 KB space
  ([PTX ISA §5.1](https://docs.nvidia.com/cuda/parallel-thread-execution/)).
- A 128-byte `CUtensorMap` passed as `const __grid_constant__` is a kernel parameter: it is copied into the
  node's parameter block at capture and is read by the kernel from constant space
  ([Async Data Copies §4.12.2](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)).

`[DERIVED]` A kernel whose `Params` struct is large and whose first accesses are cold pays a constant-cache
miss chain after the flush. This is the same class of cost the two external teams recovered by executing the
epilogue early: it is *fetch latency exposed at a point where the kernel has nothing else to do*.

### 2.3 Why our own i-cache attempt measured ~0

`OKC_FINWARM` is exactly this mechanism: "dry call of the last-CTA finalize during the prologue (instruction
cache warm-up)" (`att7_c_knobs_attn3.md:82,137`). Measured on the three q_len-1 sliding loads:
8.42→8.51, 11.20→11.01, 12.86→12.74 µs (`attn_dual.md:98-100,118`) — i.e. **+0.09, −0.19, −0.12 µs**, two
of three better. The roster records +0.16% and 0/12 shapes selected. The external 10.8→8.9 µs result
(ticket, two teams `[UNVERIFIED]`) is therefore real in principle but **the easy part of it is already
harvested in our kernel**; the remaining exposure is limited to the code paths that are still cold at their
first execution and are not already dry-run.

Two implementation details if it is revisited: the dry run must be gated by a **runtime** predicate that the
compiler cannot fold (ours is `p.cs <= 1 && !p.coop && p.nc == 1`, `attn_dual.md:15`, `att7_c:137`), and the
callee should be `__noinline__` — an inlined copy is a second instruction stream that does not warm the
out-of-line path's lines, and an unused-result call can be eliminated (the general compiler argument is
standard; the concrete gate is ours).

### 2.4 The algebra of the fixed per-replay subtraction — exact

Definitions (from `TASK.md:94-96`, `scoring.py:1-30`, and the replica's calibration in `judge_replica.md`
§3):

- Per shape *i*: measured replay `T_i = kernel_i + G_i` (kernel time + graph-launch/event cost).
- The judge reports `r_i = max(0, T_i − k)`, with `k ≈ 4.1 µs` (the "about 4 µs" of `TASK.md:96`),
  computed from a 1-kernel vs 2-kernel graph so the kernel cancels: `k = one − (two − one)`.
- Per load *l*: `C_l = Σ_{i∈l} w_i r_i`, `R_l = Σ w_i r_i^{ref}`.
- Score: `S = (Π_{l=1..6} R_l / C_l)^{1/6}` (six-load geometric mean; `scoring.py`).

Derivative for one shape, while `r_i > 0` (so `dr_i/dT_i = 1`):

```
∂S/∂T_i   = − S · w_i / (6 · C_l)
∂lnS/∂T_i = − w_i / (6 · C_l)          [per µs]
```

The subtraction `k` is a constant offset applied to **both sides**; it cancels from the derivative but
**shrinks the denominator**, so a saved µs is worth `1/(T_i − k)` in relative terms instead of `1/T_i`.
Once `T_i ≤ k`, `r_i = 0` and the derivative is **exactly 0** (the replica's `max(0, …)` is the only
sensible reading; a literal division by zero would be undefined).

Numbers for `muse_decode_layer`, computed from the judge's own per-shape times (`klr_fci5` job
`9743b6e71fbd42fd`, profile in `src/okc_loop/vendored_work/layer/profiles/`):

| load | C_l (µs) | sliding %/µs (w=46) | full %/µs (w=16) |
|---|---:|---:|---:|
| decode 1 | 4262 | **0.180** | 0.063 |
| decode 16 | 8685 | 0.088 | 0.031 |
| decode 32 | 11008 | 0.070 | 0.024 |
| verify 1 | 6746 | **0.114** | 0.040 |
| verify 16 | 16148 | 0.048 | 0.017 |
| verify 32 | 21697 | 0.035 | 0.012 |

These reproduce the last column of the profile table exactly (e.g. D16S 0.0883, V1S 0.1137), so the
profile's last column **is** this derivative. (Caveat: the inputs are the judge-reported per-shape µs as
printed; if those are pre-subtraction, `C_l` grows by `62·4.1 ≈ 254 µs` and every entry falls by ≤6%. The
ordering and the conclusion do not change.) Interpretation: **1 µs saved on the four B=1 shapes is worth
≈0.40% score; 1 µs on the four V32/D32 shapes ≈0.11%.** Spend the remaining budget on B=1.

For a shape whose *measured* time is ~5 µs (`T = 5`, `k = 4.1`, so `r = 0.9`): one load with both shapes at
5 µs has `C_l = 62·0.9 = 55.8 µs`, so 1 µs on the w=46 shape is `46/(6·55.8) = 13.7%` score. A plain
"5 µs shape" without the subtraction would be 2.5%. **The subtraction does not make a 1–2 µs kernel free;
it makes each µs worth ~5.6× more, until the clamp, after which further savings are worth exactly 0.**
For the layer's shapes (66–471 µs reported), `k/T` is 0.9–6%, so the amplification is ≤1.06× and the
practical statement is: absolute kernel-time deltas count 1:1, full stop.

What the subtraction *does* neutralise: graph-launch and event overhead up to `k`. We cannot win by shaving
the *graph launch*; we can only win by shaving `Σ w_i(kernel_i + per-node dispatch_i)`. If our 8-node graph's
`G_8` exceeds the 2-node `k` that the judge calibrated, the excess is charged on every replay — that is the
one place where node count has a direct, quantifiable price (§5, item 3).

### 2.5 The other side of the ledger

`TASK.md:94`: a shape's speedup is the **median of paired ratios** (SGLang/yours, order swapped), and
`TASK.md:95`: the run's 95% interval is "at least ±1.5%". So a sub-1.5% change on one shape is not
distinguishable from noise in a single run; the board score is still a point estimate, but any strategy
whose gain lives in <1% per-shape deltas is a coin flip under reruns (§4).

---

## 3. What `prepare()` can precompute that survives into the graph

### 3.1 What survives

`prepare(weights)` runs once, untimed, up to 300 s (`TASK.md:27`; the replica's budget key `prepare_s`).
Because capture is a **snapshot** (Graphs §4.2.2) and replay reuses "the same memory addresses"
(PyTorch), everything whose **value** is baked at capture survives; everything whose value is computed by
host code at replay does not.

Survives (baked at capture):
- **Tensor addresses** and any layout/repack done to weights, provided the tensors stay alive (PyTorch:
  "you must maintain long-lived references to tensors that hold input and output data during capture").
  Input buffers are also stable: the judge writes new draws into the captured buffers' own addresses
  (`judge_replica.md` §2), which is why a graph can be replayed on "fresh inputs" at all.
- **TMA descriptors** built with `cuTensorMapEncodeTiled` on the host in `prepare()` and passed as
  `const __grid_constant__ CUtensorMap`; the 128-byte descriptor is copied into the node's parameter block
  at capture (Async Data Copies §4.12.2). Rebuild only if the base address/strides change.
- **cuBLASLt plans**: heuristics are host work ("could take tens of microseconds") and are meant to be
  "queried once … and then reused" ([cuBLAS §3.1.2](https://docs.nvidia.com/cuda/cublas/index.html)).
  cuBLAS routines are capturable "without restrictions in most situations"; the exception is host-pointer
  output (§2.1.12). Note §2.1.12's warning: during capture the library may use `cudaMallocAsync` for
  workspace, and "attempts to capture cuBLAS routines in [child graphs / device-launched graphs] may fail.
  To avoid this issue, use `cublasSetWorkspace()` … user-owned workspace memory." Allocate the workspace in
  `prepare()` and pin it.
- **Launch configuration tables / per-shape plans** measured in `prepare()` on the actual judge node
  (300 s of GPU time). `kernel_fn` runs on the host at capture and may choose config from shape/`config`
  (`TASK.md:100` explicitly allows "branching on shapes and `config`").

Does **not** survive:
- Host code in `kernel_fn` (it does not run at replay).
- Allocations made after capture, or any pointer that moves. Graph memory allocations get virtual addresses
  fixed "for the lifetime of the allocation node", and their "contents are not persistent past the freeing
  operation" (Graphs §4.2.5) — so a graph that captured a pointer cannot be pointed elsewhere without
  `cudaGraphExecKernelNodeSetParams`/`cudaGraphExecUpdate` or recapture (§4.2.3).
- Host-side mutation of the 128-byte TMA descriptor object after capture: the captured parameter block is a
  copy. (Device-side tensormap edits are a different, fenced mechanism, sm_90a only, §4.12.2.5.)

### 3.2 Workspace reuse across replays

Yes — with three documented/observed conditions:

1. **Address stability.** PyTorch's caching allocator satisfies capture allocations from a graph-private
   pool that "stays alive until its `CUDAGraph` object and all tensors created during capture go out of
   scope" (PyTorch CUDA semantics). Workspace allocated in `prepare()` lives in the ordinary pool and is
   fine as long as a reference is held in `state`.
2. **Contents must be scratch.** Replays do not restore anything. If an algorithm needs a zeroed counter or
   a barrier generation per replay, that reset must be a captured device operation (memset node or the
   kernel's own first CTA), never a host `.zero_()` between replays.
3. **Interleaved replays must be safe.** Our own harness captures **two** graphs per case on separate
   buffers and alternates `g1/g2` with the same `state`, precisely to prove nothing is cached
   (`colab_harness.py:528-561`). If the judge does anything similar, a shared workspace that carries state
   across replays will be caught as a correctness failure. Treat all `state` memory as per-replay scratch.

### 3.3 The 300 s budget as an on-node autotuner

`prepare()` is the only place with untimed GPU time and the judge's own hardware. It is therefore the
legitimate way to (a) run a plan/config sweep on the public scored shapes plus the nearby/edge shapes, (b)
rebuild the descriptor/plan cache for the judged weights, and (c) keep the per-shape choice out of
`kernel_fn`'s replay path (it is a capture-time dict lookup). The rules only forbid caching data derived
from `data` (`TASK.md:101`) and importing the reference; benchmarking your own kernels on your own
synthetic buffers is allowed. It counts against the 300 s wall clock and is event-gated in the harness
(`colab_harness.py:355-375`).

---

## 4. Robustness / anti-overfit

### 4.1 What is actually measured and rerun

- Official statistics: per-shape **median of paired ratios**, order-swapped, 95% interval at least ±1.5%
  (`TASK.md:94-95`). One run produces the board row; the leaderboard is best-run.
- Final positions: **medians of organizer reruns on scored and nearby shapes** — `Ranoobaba/okc-campaign`
  `LEARNINGS.md`, [URL](https://github.com/Ranoobaba/okc-campaign/blob/master/LEARNINGS.md), quoted in
  `research_01_gh_code.md:152`: "Final positions are medians of organizer reruns on scored *and nearby*
  shapes. No overfitting to the exact m/ctx — overfitting is punished at scoring time." `[UNVERIFIED]` by us,
  but it matches the judge's public shape list and the always-run fringe below.
- Our own harness always runs `scored + nearby + edge` correctness cases (`colab_harness.py:473-497`):
  `LAYER_NEARBY` = batch 1/ctx 60000/sliding, 8/70000/full, 2/20000/sliding, 4/20000/full, 33/70000/full
  q_len 8 (`cases.py:199-205`), plus `EDGE_CASES` including ctx 10 with page 64, ctx 2047, batch 2 q_len 8
  ctx 3000 with `kv_scale 1/64`, and batch 2 ctx 20000 page 64 (`colab_harness.py:422-450`). The task's 13
  acceptance tests add ctx 300000 and 1048576 and batch 9/33 (`TASK.md:56-70`); the competitor's notes put
  page 64 and `kv_scale 1/64` in the hidden tests (`research_01_gh_code.md:158`).

### 4.2 Where a shape-specialised kernel degrades on a nearby shape

Each of these is a real discontinuity in the source or in the hardware, not a guess:

1. **Exact-match dispatch.** `moe_selftune.md:73`: the `sel` table hits only an exact `cfg["bs"]`, so
   nearby/test shapes (4, 33, 257, ~128-skewed) take the generic path. Fine for correctness; a 5–10%
   performance cliff if the generic path is tuned only for "it works".
2. **Tile remainders.** Tile sizes chosen for 1/8/16/32 tokens leave partial tiles at 33 or 130; a mask
   predicate that is correct can still cost a full tile of idle warps. The profile's own kernel names are
   templated on tile params (`sk2_kernel<1, 8, 4, 1, 8, 1, 6, …>`), so a new shape instantiates a *different*
   kernel — one that was never benchmarked.
3. **Occupancy / wave cliffs.** Grid sizes derived from token count or expert count can cross an SM-count
   boundary between 32 and 33 tokens; `research_06_microarch.md` §0 documents the related small-workload
   penalty on H200 for slice-ordered/persistent-block schemes ("on small workloads the persistent-block
   probe can make it *slower*").
4. **Routing-distribution cliffs (MoE).** Nearby cases switch `dist` (soft → mixed/outliers) and token count
   ×8; the number of distinct local experts can go from ~21 to 32, i.e. weight traffic from ~590 MB to
   ~905 MB. A kernel tuned around "few experts are hot" (the scored `router_distinct` 21/31) can fall off a
   bandwidth cliff on `dist=outliers` or batch 33. The judge's own `router_distinct = 31` for verify 32 is
   the warning: it is nearly all 32 local experts already.
5. **Feature-axis cliffs.** page 64 (tests) vs page 1 (scored) changes the KV gather stride; `kv_scale
   1/64`; ctx 1M changes the full-layer loop count by 12×; `q_len 8` vs 1 changes whether the sliced
   unpooler path is legal (`T=8` is the boundary, `reference.py:57-58`).

### 4.3 Cheap guards

- **One generic path that is always correct and within a few % of the specialised one**, with the
  specialised path gated by a *range* test (not an exact set membership). The `sel`-table pattern already
  has this; measure the generic path's *time*, not just its correctness.
- **Capture-time formula, not a table**: compute the launch config from `(tokens, distinct-expert upper
  bound, page, ctx)` with monotone arithmetic, so an unseen shape lands between two measured ones instead of
  on a default. Legal (`TASK.md:100`).
- **Prepare-time sweep over the shape space** (1..64 tokens, 96/128/192/256/384/512, page 1/64, ctx
  10/2047/6000/80000/1e6) and record the *worst* nearby deltas; reject a config that wins on scored shapes
  but loses >2% anywhere in the fringe.
- **Timing gate, not just correctness**: the replica already emits per-case `replay_us` for nearby shapes
  (`_weight: 0` keeps them out of the aggregate). Treat a >3% regression on any nearby/edge shape as a
  blocker, exactly as the organizer's rerun median would.
- **Round grids to the SM count** with a persistent-CTA stride loop, so 33 tokens and 32 tokens use the same
  code and differ only by one loop iteration.

### 4.4 What the median does to a tuned kernel

Score is a geometric mean over six loads, so a shape that gains 3% on decode-1 and loses 3% on verify-32 is
not neutral: the loss is multiplied by 46/16 weights and by whichever denominator is smaller. Then a rerun
median over scored+nearby shapes punishes *run-to-run variance* a second time. Concretely, a config with a
2% mean gain but 4% variance across nearby shapes has a median gain near 0. This is the same conclusion as
`TASK.md:95`'s ±1.5% floor, applied twice.

---

## 5. Top 3 by expected value for the remaining budget

Ranked by `P(gain) × Δscore / effort`, using the per-µs table in §2.4 and the profile in
`profiles/klr_fci5.9743b6e71fbd42fd.txt`.

### 1. Put a PDL edge on every kernel boundary that lacks one; absorb the two glue nodes

The graph has **8 nodes** at B=1: `moe_fused` 23.7 µs, `sk2_gle…` 17.3, `sk2_kernel` 10.2, `sk2_gle…` 6.3,
`attn_kernel` 5.3, `sk2_kernel` 4.3, `_gle_route` 2.7, `_sl_cluster_finish` 1.8 (GPU time, profile 0). The
last two are **4.5 µs, 6.7% of the shape, and are pure fixed cost** (routing 2.7 µs and a launch/finalize
1.8 µs). PDL hides "the launch latency of secondary_kernel behind the execution of primary_kernel"
(§4.5.1) and is capture-legal (§1.2), so it works on every replay. There are 7 boundaries; recovering
0.5–1 µs at each of 4–6 of them is 2–6 µs per shape, and the four B=1 shapes carry 0.40%/µs combined —
**0.8–2.4% score if most boundaries pay, 0.4–0.8% if only half do**; killing/absorbing the two glue nodes
outright is up to **+1.8%** on the B=1 loads. Risk: low (implicit trigger at primary exit; `griddepcontrol.wait`
still required before any dependent read). Our PDLKV null result does not cover this — it reordered copies
around an *existing* wait.

### 2. Use `prepare()` for an on-node autotune sweep + a scoring-side nearby gate

300 s untimed on the judge's own H200 (TASK.md:27): rebuild heuristic/plan tables, sweep launch configs over
the scored shapes **and** the nearby/edge fringe, store a capture-time lookup. No replay cost, legal
(`TASK.md:100`), and it is the only defence against the rerun median: a 1–3% rerun penalty is larger than
most kernel micro-wins left on the board. Upside if the sweep finds 0–1% per shape; downside avoided 1–3%.
This also subsumes the workspace/descriptor precomputation rules of §3 (no allocation at capture, stable
pointers, scratch-only reuse).

### 3. Calibrate our own graph's fixed cost against the judge's `k` (cheap, decides item 1's size)

Capture the same 8-node `kernel_fn` and a 1-node no-op graph with the judge's ritual (flush → record →
replay → record, median of 30) and compute `G_8` the same way the judge computes `k`. If `G_8 ≈ k`, per-node
dispatch is free and all remaining EV is in kernel bodies (item 1's absorption case is then the only
node-count lever). If `G_8 − k` is ~1 µs or more, that excess is charged every replay on **every** shape and
node reduction is the single best code change available. Cost: one replica run. This also closes the
`±2%` reproduction gate in `judge_replica.md`, which is currently open.

**Deliberately not in the top 3: a general cold-i-cache hoist.** The mechanism is documented
(§2.1) and external teams report 10.8→8.9 µs, but our `OKC_FINWARM` already implements exactly it and
measured −0.19, +0.09, −0.12 µs on the three relevant loads (`attn_dual.md:98-100`). Re-spending budget there
is re-running a closed axis; only revisit it inside item 1's glue-node work, where the whole tail is still
cold.

---

## 6. Not verified / risks

- The ticket's two-team 10.8→8.9 µs cold-code result is `[UNVERIFIED]` here; our own measurement of the
  same idea is ~0.
- Hopper's instruction/constant cache geometry is unpublished; the L1i/L1.5-constant sharing claim is a
  **Volta** measurement (arXiv:1804.06826) and is marked `[INFERENCE]` for H200.
- Our replica flushes 256 MiB with a write (`colab_harness.py:186`); the judge reads ~1 GiB. Both exceed L2,
  but a write flush leaves dirty lines where a read leaves clean ones; the `±2%` reproduction gate is the
  check that this does not matter.
- The "final ranks are medians of reruns on nearby shapes" claim is a public competitor's, not an official
  rule; the official rules only say "best run is your row" (`TASK.md:132`). Treat the median as the
  worst-case scoring model, which is the safe assumption.
- The judge's conditional-node/device-graph behaviour is irrelevant to us, but note we also cannot use
  `cudaGraphNodeSetEnabled` per replay: no graph handle is exposed to `kernel_fn`.
