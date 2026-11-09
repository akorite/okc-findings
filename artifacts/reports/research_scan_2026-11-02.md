# Research scan 2026-11-02 (scout LegislativeAnglerfish)

## Ranked levers (mechanisms only; closed list filtered out)

**L1 — Warp-parallel `wgs_ranges` prologue** *(NEW, MoE)* — Incumbent computes the stream-K
stage ranges serially on one lane (`wgs_prefix` + two `wgs_range` binary searches,
`wgs_ranges` at cas…e5.py:2344/4008, called from a single thread at :3925). keell0renz
measured **+0.9% MoE board** making the job-cost prefix a lane-parallel `__shfl` scan and the
boundary search warp-parallel (their LESSONS.md §3; repo fetched 2026-11-02). Small isolated
edit; bundle-able.

**L2 — `OKC_SYNC0` per-warp slot release** *(specced NEXT.md §1)* — alygg77's H200 microbench:
consuming a ring fully before refill drops per-SM stream from ~140–207 GB/s to **60–64 GB/s**;
"the loss is boundaries, not peak" matches our 1.3715 vs 1.134 µs/unit. CAUTION from
keell0renz: dual-lane wgmma + per-unit barrier variant ran **8–20% slower** — SYNC0 must
REMOVE sync, not add it.

**L3 — `OKC_SC6` 6-bit scale codec** *(specced NEXT.md §2)* — keell0renz shipped "lossless
6-bit block scales" at 2.42 vs our 2.375 (+~1.5–2%). Confirmed estimate; ship it.

**L4 — A→B transition x-gather bubble** *(BFLAG prefetch exists; retune not incumbent)* —
alygg77: phase B slow start at bs1, silent **6–9 µs tail at d16–32**, end-spread 7.5 µs,
cost-table r²≈0. Levers: finer phase-B units (256→128-row probe neutral-positive) or fix
phase-B cost weights vs measured n-tile counts. ≤1–2%, speculative.

**L5 — Fence-site reduction (`NOFENCE`-style)** *(NEW, MoE)* — alygg77 measured GPU-scope
fences ≈1 µs/call; removing unneeded acquire-release sites on the hot path gave **+0.234%**.
Bundle-only.

**L6 — Layer-side: cluster-scope residual/norm boundary** *(NEW)* — keell0renz credits "a
cluster-based residual/normalization boundary" among their top layer steps (board 2.2659 vs
our 2.194, ≈3%). Mechanism: fuse post-attention residual add + post-expert norm into
cluster-scope producer/consumer; kills one elementwise round-trip. Needs design read of their
`muse_decode_layer.py`.

**L7 — I-cache warmup warp** *(NEW, small)* — muse-spark-open attn log A13: cold-L2
instruction fetch is a measured cost; warm once-per-CTA phases via `__noinline__` variant on
idle warp, stores off. Our MoE kernel is huge; serial prologue/epilogue paths run cold every
launch. Sub-µs, bundle-only.

**Already in incumbent — do NOT re-propose:** KPERM3 4-step scale-pair sharing, warp scans in
routing, deferred `wgs_flush` fixup (WGSFCI=3), early `launch_dependents` before grid barrier,
PDL wait placement, FT_T=64 + WGSREORD, 3-slot TMA ring (NS=4 falsified), 12-bit dense path,
per-shape autotune tournament, Triton router at T≥128.

**New negatives:** WGS dual-lane wgmma/PUB −8–20%; dynamic per-unit claiming/sub-unit claims
≈ break-even; speculative split-unit expert prefetch ≈ break-even; fused-slice MoE dead; bf16
split-K multi-ring for m≥128 dead (E10); cluster-push tail >MT=1 dead (E14); FP32-pooler
Triton dead.

**Skipped:** redteam/ai-review remainder (HYPOTHESES verdicts cover it);
`.vendor/openkernels/notes/` does not exist. Risk: L4/L6 rest on single public claims —
gate behind check budget, never unbundled.
