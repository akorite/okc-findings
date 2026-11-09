# fd_kernel_sources — upstream mechanism sweep (2026-11-04, final day)

Lane R3. Six upstream repos read at these commits (local clones): flashinfer `08d74a4`,
DeepGEMM `5081f85`, TensorRT-LLM `a649e41`, FlashMLA `395cb3a` (fresh clone, /tmp/flashmla),
vLLM `ffdf267` (`tmpbuild/engines/vllm`), LeanAttention via MLSys'25 paper
(`artifacts/reports/research_03_venues.md` §2.2). For each: the ONE surviving mechanism
most transferable to our boards, plus what is already-owned or dominated so it is not re-derived.
Dead-mechanism baseline: `docs/HANDOFF-2026-11-04-FINAL.md` §6.

## Lever table (all surviving levers; est. values are board %)

| # | mechanism | source | board | est. value | vs dead list |
|---|---|---|---|---|---|
| 1 | **Expert-synchronous schedule order**: interleave same-expert units across CTAs so concurrent CTAs stream the same weight stage; duplicate expert reads hit L2 by *time alignment*, not cache hints | DeepGEMM `scheduler/gemm.cuh:185-269` (strided `iter*numSMs+blockIdx` over expert-major m-blocks) vs ours `moe_dup.py:2339-2370` (contiguous cost ranges ⇒ CTAs on *different* experts) | moe, decode-layer MoE | +0.3–1.7% moe board (dup-byte bound 59.9 MB ⇒ +1.67%, `moe_schedule.md` §3; realistic ~half) | MOEDUP was an **L2 hint** and *failed judge checks* (rc 112). This is job/unit **order**, correctness-neutral; it is the residual lever moe_schedule.md §3 named but deferred |
| 2 | **Equal-work chunk boundaries on sliding batched cells**: global constant `chunk` (per request `nact` derives from data — the XQA/our-own idiom), so chunk size doesn't scale with each request's `n_keys` | FlashMLA `get_decoding_sched_meta.cu:64-104` (payload = blocks + `fixed_overhead_num_blocks` per request); DeepGEMM `sm90_paged_mqa_logits.cuh:60-115` (same, SPLIT_KV granularity); LeanAttention stream-K tile→CTA map | attention | +0.03–0.1% (ctx spread ±7.5% ⇒ ~4% makespan on the ~3-9 µs stream term of D16S/D32S/V16S/V32S, ×0.241 %/µs) | Dead list killed split *count* (`attn_plan.md` 0/12, `attn_xqa_grid` in flight); this changes boundary *placement*, not count. Our FLAT mode already does exactly this (`dual.py:2602-2660`, incl. the `SEGKF` fixed per-request charge = FlashMLA's `fixed_overhead_num_blocks`) — the port is the sliding `!FLAT` path only |
| 3 | **KV-tile striping across splits** (`idxKTile = idxKTileInit + idxIter*nbSubSeq`): each split reads interleaved 128-key tiles rather than a contiguous span — every split touches the same page-table region distribution | XQA `mha_sm90.cu:716,761,805` (via `attn_xqa_grid.md` §3) | attention | ~0 (±noise; contiguous is arguably better for DRAM rows) | Not on the dead list only because nobody tried it; 5-line variant worth folding into the next grid A/B, not a standalone check |
| 4 | **Compile-time stage index for narrow-N buckets** (`kStaticStages`): token descriptor built in the uniform datapath, no per-stage R2UR/vector math | DeepGEMM #473 diff :4356-4419, quoted −1.4…−2.8% on *their* issue-bound loop; ours is 20% MMA-utilised (`moe_nbucket.md` §4 item 1) | moe | ≤0.5%, WGS loads only | Not dead — deferred as unverifiable-without-GPU; cheap (one `switch`→template hoist) if a batch slot exists |

## Per-repo verdicts

### 1. FlashInfer MonoMoE — `csrc/fused_moe/monomoe/` — verdict: **already-owned, no new lever**

The single-launch 5-phase MoE (routing ∥ BF16 prefetch / quantize / up-proj / down-proj /
writeback) for bs≤8 (`moe.cuh:24-28,111-255`), co-resident grid `GRID_SIZE ≤ SM count`
(`monomoe_wrapper.cuh:120-128`). Its distinctive machinery:

- **Sentinel-in-data phase handoff**: up-proj epilogue release-publishes each `temp_act_scale`
  cell after its fp8 payload; down-proj polls cells until non-sentinel → per-*expert* early
  start, no barrier (`moe.cuh:174-189`, `moe_down_projection.cuh:63-97`). **We already have the
  finer version**: phase-B stages wait on the 2 phase-A *units* that wrote their inter columns
  (`moe_dup.py:2041`, `wait_A` :1404) — per-unit, not per-expert.
- **Inter-expert lookahead**: next expert's K=0 activation TMA issued while the current expert
  computes, after a single-thread sweep-poll (`moe_down_projection.cuh:99-134,328`). Our
  producer streams stage space in cost order continuously; the wait is the data, not a phase.
- **Launch-parity double-buffered sentinel/counter reset** (graph-replayable, zero host work;
  `moe.cuh:75-109`, `spec->launch_flip`). Our attention kernel already does this exact trick
  (`dual.py:2317-2326`, epoch counters).
- fp32 `atomicAdd` down-projection accumulation (`moe.cuh:250-255`) — we accumulate in-kernel
  already; their combine adds nothing.
- Dtypes: block-wise FP8 (`moe_interface.h:93-95`), not NVFP4 W4A16 — math does not port.

### 2. DeepGEMM PR #473 + grouped-GEMM scheduler — the ONE real lever

N-bucketing is confirmed closed (`moe_nbucket.md`: ours is a superset). The other tricks:

- **Surviving**: the grouped scheduler's *order*. `get_next_block` strides `next_block_idx =
  iter*kNumSMs + blockIdx.x` (`scheduler/gemm.cuh:186`) and `get_swizzled_block_idx` maps
  consecutive indices onto m-blocks of the same expert group (`:195-204`, MGroupedMasked walks
  groups cumulatively). Effect: at any instant the resident CTAs share m-block neighbors of the
  **same expert**, so concurrent CTAs fetch the same weight tiles — L2 (and, with clusters, TMA
  multicast `kIsTMAMulticastOnA`, `:137-143`) serves the copies. **Ours**: `wgs_range` gives CTA c
  a contiguous equal-cost interval of the expert-major job list (`moe_dup.py:2365-2369`), so
  concurrent CTAs sit on *different* experts and the two jobs of a duplicated expert are read
  ~45 µs apart — guaranteed DRAM re-read (`moe_schedule.md` §1b). Interleaving the unit→CTA map
  inside the same cost-prefix (keep equal-cost totals, change *which* stages a CTA sees when)
  aligns the copies to ~µs — the fix moe_schedule.md §3 sketched but never priced. This is why
  it is not MOEDUP: MOEDUP kept the schedule and asked L2 to hold 28 MB across 45 µs; this
  removes the reuse distance.
- TMA multicast itself is **unexpressible** for us: no cluster launch, weights fetched by 1-D
  `cp.async.bulk` (`moe_schedule.md` §2).
- `kStaticStages` (lever 4 above) is #473's only other remaining N-adjacent trick.
- Bonus confirm: `scheduler/sm90_paged_mqa_logits.cuh:60-115` is independent evidence for
  lever 2 — prefix-sum per-request `ceil(len/SPLIT_KV)` split counts, reversed remainder
  allocation (`:75-80`).

### 3. TRT-LLM XQA — verdict: **occupancy axis already adjudicated; striping is the only residue**

- The occupancy trick *itself* is a measured failure: `splits = round(0.75·SM/(batch·kv_heads))`
  (`mha_sm90.cu:2960-2972`) produced the 66-block/18%-of-peak launch of flashinfer #5998
  (`attn_xqa_grid.md` §2). The `sqrt(8·blocks)` cap + 8-wave gate + `kEnableMinBlockFactor=4`
  admission (`decoderXQARunnerUtils.h:409-431`, `decoderXQAConstants.h:27-30`,
  `decoderXQARunner.cpp:154-176`) are all inside the `attn_xqa` arm whose check
  `01b835e2a93744ab` is in flight — do not re-run.
- Residue not in the check: **KV-tile striping** `idxKTileInit + idxIter*nbSubSeq` (lever 3)
  and 4 dedicated merge warps over a 2-deep smem ring (`mha_sm90.cu:1473-1550`) — the latter is
  dominated by our cooperative finalize (all arrival CTAs slice the output, `dual.py:2308`).
- `nbSubSeq` derived from data (`multiBlockMinNbTilesPerCta`, `mha_sm90.cu:713-723`) = our
  `nact = ceil(n_keys/chunk)` (`dual.py:2308`). Already ours — and it is exactly the primitive
  lever 2 needs: a global `chunk` with per-request `nact` requires **no merge changes**.

### 4. FlashMLA — verdict: **confirms lever 2; we already own its substance on FLAT**

- `get_decoding_sched_meta` (`smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu:
  64-104`): one warp packs KV blocks into `num_sm_parts` equal payloads **plus
  `fixed_overhead_num_blocks` per request** — a per-request completion tax inside the balancing
  itself. Our FLAT partition is byte-identical in form: equal `T/G` ranges over the flattened
  key space *plus* `SEGKF` (400/600 keys) per request boundary (`dual.py:2602-2660`). Full cells
  already run this; sliding batched cells run per-request `n_splits` (`dual.py:2531-2558`,
  `_plan` :3096-3124) — that gap is lever 2.
- Combine kernel: PDL-gated, `num_splits` cumsum early-exit, one warp per head
  (`smxx/decode/combine/combine.cu:64-98`) — our merge is in-kernel, strictly better.
- Everything sm100 in this tree (umma, cluster pairs) does not port to sm_90a.

### 5. LeanAttention — verdict: **the formula is already shipped; same residual as #4**

`G = NumSMs × MaxCTAsPerSM`, LeanTile 256 @hd64, host-block/flag merge — `attn_xqa_grid.md` §1-2
proves `_plan`'s `2·SM/(B·pairs)` **is** this grid at our real occupancy (that's the
`XQA_CTAS_PER_SM=2` control in the in-flight check). Its stream-K tile→CTA mapping crosses
request boundaries (a CTA's LeanTiles span requests); our `!FLAT` mode doesn't — same residual,
same lever 2.

### 6. vLLM Marlin MoE — verdict: **dominated on every axis; nothing survives**

- Work order: `moe_align_block_size` orders m-blocks by **expert index** with uniform
  `block_size_m` (`csrc/moe/moe_align_sum_kernels.cu:195-228`), block size chosen so
  `M*topk/E/block < 0.9` (`marlin_moe.py:344-351`). Our LPT-by-ntt job list + per-job ntt
  bucketing strictly dominates (uniform 8/16/64 pads every expert to one size).
- Balance: persistent CTAs sweep ordered m-blocks with a K-slice `locks[]` protocol
  (`marlin_template.h:389-432,535-596`) — static assignment vs our stream-K cost prefix;
  Marlin's own comment admits the design is for conflict avoidance, not balance.
- Occupancy heuristic (`ops.cu:219-238`: `blocks_per_sm` from reg/smem attrs, `allow_count`
  ≤4 for `thread_m_blocks==1`) — same inputs as our occupancy table, no new formula.
- The only instruction-level item, "m as the inner loop to overlap dequant and matmul"
  (`marlin_template.h:1155-1157`), is mma.sync-era advice; our WGS path is wgmma with the dequant
  folded into the A fragment before the MMA (`moe_nbucket.md` §2.2).

## Where this leaves the day

Two actionable levers, both schedule-order rather than kernel-math:

1. **Lever 1 (expert-synchronous unit interleave)** — largest unpriced byte lever on the MoE
   board; the only upstream mechanism that attacks bytes our schedule creates. Implementation =
   reorder `wgs_range`'s stage→CTA map inside the existing cost prefix; correctness-free
   (order only). Needs a GPU slot; estimate +0.3–1.0% moe.
2. **Lever 2 (global chunk on sliding batched cells)** — 30-line `_plan`/kernel-boundary change
   reusing the FLAT idiom and the existing `nact` merge; +0.03–0.1% attention. Must wait for
   check `01b835e2a93744ab` to adjudicate the sliding `n_splits` count first (same lines).

Everything else either *is* our code (sentinel handoffs, parity counters, grid-first splits,
FLAT equal-work + fixed-charge), was already measured dead (split counts, stage depth,
L2 hints, multicast), or is dominated (Marlin scheduling, XQA merge warps, MonoMoE phases).
