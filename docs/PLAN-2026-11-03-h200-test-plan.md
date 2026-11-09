# H200 test plan — what to test once compute is free

**2026-11-03.** Research + planning only; nothing in this file has been run. Board state at writing:
`muse_decode_layer` **2.28370 / 9,905.1 µs, #80 of ~1215**; top-20 = **2.38339 (+4.37%)**.
Judge quota: 20 checks + 10 submits per board per UTC day, reset 00:00; contest closes
**2026-11-05 07:59 UTC**, so two buckets remain.

The point of this document: the judge is a *ranking* oracle at ~0.63% SD per run. An H200 turns
measurement into a *debugging* tool. Everything below is ordered by what that unlock buys first.

---

## 0. Why borrowed H200 time changes the decision

| | judge only | with an H200 |
|---|---|---|
| cost of one A/B | 1 check, 40-min queue, 1/20 of the day | ~2 min, unlimited |
| resolution | σ ≈ 0.63% per run, ±1.5–1.9% check-to-check → **a real 1% needs 6–8 runs per arm** (keell0renz, 460 experiments) | paired A/B, 30+ reps, ±0.2% |
| what you can ask | "is this faster?" | "**where are the idle edges?**" per-SM timelines, per-kernel, per-phase |
| variant throughput | 1 per 40 min | 8–16 per process (build all in `prepare()`, time back to back) |

Our last ~40 judge checks bought a 1.62% win and a very complete map of what is *not* the problem.
The one thing that map still cannot answer is why a 28 MB read takes 29 µs — that is a profiler
question, not a quota question.

---

## 1. The arithmetic that makes +4.37% plausible

Computed from `problems/muse_decode_layer/reference.py` (HIDDEN 8192, D_EXPERT 4096,
N_EXPERTS 256, **LOCAL_EXPERTS 32**, top-8, NVFP4 2-codes/byte + one e4m3 scale per 16 values):

| | bytes |
|---|---|
| one expert: w13 codes 16.78 MB + w13_sf 2.10 MB + w2 codes 8.39 MB + w2_sf 1.05 MB | **28.31 MB** |
| all 32 local experts | **906 MB** (its BF16 twin is the 3.2 GB the starter expands) |
| floor for all 32 at the measured 4.2–4.45 TB/s HBM ceiling (keell0renz) | **≥204 µs** |

Measured in-layer MoE time per load (profile `82c098ae…7cd40fdd1a934e27`, the row's parent).
The routed-expert count is **not** the uniform model's: `generate_input(rank="busiest")` rotates the
expert ids so this GPU holds *the busiest group of 32*, so at bs=1 the local GPU gets **2–4 of the 8
routes (mean ≈ 2.5)** — verified against `problems/muse_moe_w4a16/reference.py`.

| load | T | MoE µs | share of step | routed bytes | implied rate | vs 3.5 TB/s reference |
|---|---|---|---|---|---|---|
| **decode 1** | 1 | **29.4** | 37.7% (of 78.0) | ~71 MB | **2.4 TB/s** | **69%** |
| decode 16 | 16 | 66.1–74.6 | 52.1% | ≥ ~14 experts | ≈ wall | 100%+ |
| decode 32 | 32 | 83.9 | 48.9% | larger | ≈ wall | 100%+ |
| verify 1 | 8 | 58.0–64.6 | 52.3% | ~8–12 experts | ≈ wall | 90–100% |
| verify 16 | 128 | 146.2–161.2 | 44.5–64% | near all 32 | ≈ wall | 100%+ |
| verify 32 | 256 | 173.5–190.5 | 37.1–61.6% | near all 32 | ≈ wall | 90–100% |

The reference rate for a read this size is **3.5 TB/s** (keell0renz: a 55 MB read from kernel entry,
ramp included — the 4.2–4.45 TB/s figure is a sustained ring, not a 71 MB burst). Against that,
**`decode 1` is the only load with real headroom, and it is ~7–9 µs, not 15**:

- the **standalone** MoE at bs=1 is **~22 µs** for the same 71 MB = **3.2 TB/s = 91% of reference**;
- our **in-layer** MoE is **29.4 µs** for that work = 2.4 TB/s = 69%.

So the gap is the **in-layer epilogue/skeleton delta**, not the MoE algorithm: the incumbent is
already weight-major (weights on the mma M dimension, tokens on N, per-expert sentinels, persistent
132-CTA launch), which is why MonoMoE's *structure* has little left to give here. Closing
29.4 → 22 µs is −7.4 µs = **−11% of the `decode 1` step**, i.e. **≈ +1.3% to +1.9% of the layer
score** at a 70–100% step-translation band; the absolute floor (71 MB at 3.5 TB/s) is ~20 µs, worth
at most **+2.5%**.

Second-largest unexamined term: the auxiliary kernels in the same steps —
`_graft_row_boundary` 8.8–13.9 µs (2 calls) at verify 16/32 = **2.7–4.5% of those steps**,
`_sl_tile_route`/`_sl_cluster_finish`/`_gle_route`/`_graft_route` 1.7–3.7 µs each at the small loads.

---

## 2. Research findings that change what we do

### 2.1 keell0renz `LESSONS.md` (they forked into our row; 460 logged experiments)
Fetch: `raw.githubusercontent.com/keell0renz/muse-open-kernel-challenge/main/LESSONS.md`

- **Noise**: σ ≈ 0.63% per run on board speedup; rows are biased upward ~1 SD **for everyone**.
  A 1% change needs 6–8 judge runs per arm. Decide sub-1% changes locally with 30+ paired reps.
- **Hardware**: TMA bulk ring HBM read ceiling **4.2–4.45 TB/s** (spec 4.8); a 55 MB read from
  kernel entry averages **~3.5 TB/s** (ramp, Little's law, not latency); per-SM steady state one
  36 KB unit per ~1.09 µs.
- **Cross-SM sync costs 3–6× more under HBM saturation** (132-CTA flag hop: 1.4 µs idle, 6.3 µs
  under a 4.3 TB/s background stream). This is why their megakernel died.
- **Their MoE diagnosis**: the "streams at 3.3 TB/s vs 4.4" story is a myth — ncu PM sampling shows
  steady DRAM equals the bare ring. The loss is in **events**: ~3 µs ramp, the A→B hand-off dip,
  phase B's slow start at bs1, and a silent 6–9 µs tail. **They never removed those events; they
  measured them.** That is the gap our decode-1 number sits in.
- **Verbatim dead list** (do not re-open without a new idea): megakernel/persistent whole layer;
  fused-slice MoE (six variants, 5–13% slower); prefetching first weights into L2 (x0.99–1.00);
  more CTAs via 4-CTA K-splitting for the qkv GEMV (x0.87 at decode 1); speculative expert prefetch;
  L2 persistence windows; lookahead rings; dual-lane wgmma (WGS2) and per-unit barriers (PUB),
  8–20% slower. **Our own 40+ measurements land on the same list independently.**
- **Their wins we already hold**: 12-bit lossless BF16 re-encoding, SC6 6-bit block scales
  (verified present and auto-enabled in `klr_fci5.py`), PDL chain, per-batch-size MoE builds,
  stream-K + cost tables, data-flag A→B, warp-parallel prologue, per-shape bf16 dispatcher.
- **Method**: same-codegen controls (nvcc shifts ~5% from unrelated edits in a hot loop); compile on
  CPU; **per-SM timelines** (`%globaltimer` + `%smid`, one row per SM) — "showed every idle edge,
  wait and tail at a glance", which beat tables of averages.

### 2.2 MonoMoE — arXiv 2609.04244 (H200, open source in FlashInfer)
`flashinfer/csrc/fused_moe/monomoe/` (`moe_routing.cuh`, `moe_up_projection.cuh`,
`moe_down_projection.cuh`, `moe_scale_inputs.cuh`, `moe_internal.h`). Read: it is **FP8 block-wise
with fp32 scales** (`dl_float8_e4m3fn` + `dl_float32`), so it is not drop-in for NVFP4 W4A16 — but
its *architecture* is the published answer to the exact failure mode above:

- **weight-major**, not token-major: CTAs are partitioned **over expert-weight tiles**, and the
  whole decode token tile goes on the tensor-core **N** dimension. No expert-local token
  materialization, no tile padding — this is what removes the per-unit prologue that our T=1 kernel
  pays 8× for.
- **one persistent launch** fusing routing, top-k, quantization, both projections, activation and
  reduction — no A→B phase barrier, so no hand-off dip.
- **sentinel/flag hand-offs under an explicit co-residency invariant** (their design doc §2–3):
  flags work here because they are inside one operator, not across a whole layer — this is the
  distinction that keeps the "cross-SM sync" objection from applying.
- Claims up to **1.54×** over vLLM Triton Grouped GEMM on H200 and −18.7% inter-token latency.

Their baseline is weak (Triton grouped GEMM); ours is not. The transferable part is the *structure
for the few-tokens-per-expert regime*, which is precisely `decode 1`, `verify 1`, `decode 16`.

### 2.3 Auxiliary-stage cost is a recognised class
flashinfer issue #6683: on the micro W4A16 path a standalone NVFP4 scale-expansion kernel costs
**5.8% of GPU busy time**, and the requested fix is to inline it or overlap it on a side stream. Our
`_graft_row_boundary` (8.8–13.9 µs) and `_sl_*` helpers are the same class of cost — separate
launches for boundary/residual work that the timeline will show as idle edges.

### 2.4 Negative intel, recorded so nobody chases it
- The current **#1 decode-layer submit is judged-maxxed** — keell0renz say so about their own
  submission in their README. Top rows are not all kernels; do not reverse-engineer them as if they were.
- ZhimaoL (decode #29) **deleted** their published branches on 2026-11-01; `zhebrak/muse-spark-kernels`
  and `chenwainuo/okc` have not moved since Oct 28/29 and are already vendored or superseded.
- `agents.kernarc.com` exposes no unauthenticated API and has not moved since 2026-10-29.

---

## 3. P0 — build the instrument before testing anything (≈4–6 h of H200)

**P0.1 Judge replica.** `tools/bench_bundle.py` already packages real `prepare()`, CUDA-graph
replays, a 256 MiB cold-L2 sweep and the oracle checks, but it deliberately omits the baseline and
therefore returns absolute µs. Add three things it lacks:

1. **the router-seed load search** — replicate `reference.weights_fields` (target
   `router_load`/`router_distinct`, `LOAD_TOL 1.5%`, 5000 attempts). Without it, the number of
   distinct experts varies per run and the T=1 MoE reading is confounded by routing.
2. **fresh inputs per repetition**, new seeds — the judge does this (`MAX_REPEATS`), and a small
   fixed input pool is exactly what a cached answer would exploit.
3. **the ~4.1 µs per-replay subtraction** (graph launch + events, measured as `one − (two − one)` on
   empty-kernel graphs). If the judge subtracts it and we do not, every shape reads +4.1 µs locally —
   6% at decode 1, where we are about to make decisions.

Acceptance: an **A/A run** on two copies of `klr_fci5.py` whose CI contains 0, and a check that the
replica reproduces the judge's own 12 per-shape µs within ±2% on a matching node.

**P0.2 Per-SM timelines.** Wrap the layer's kernels with `%globaltimer` + `%smid` stamps (CTA start,
each phase mark, PDL wait, end) and plot one row per SM for all 12 shapes. This is the deliverable
that names the decode-1 gap: it will show whether the single routed expert's 28.31 MB is spread over
all 132 CTAs (then it is a supply problem) or confined to a few (then it is a partition problem, and
T1b applies).

**P0.3 Variant sweeps in one process.** Build 8–16 compile-time variants inside `prepare()` and time
them back to back on the same inputs — one session covers what used to be two weeks of checks.

---

## 4. Ranked tickets

### T1 — decode-1 MoE: close the **in-layer epilogue** gap, 29.4 → 22 µs  ⟵ do this first
`decode 1` is 37.7% MoE, and the standalone build of the *same* kernel already does that work at
91% of the achievable rate. The gap is what the layer adds around it.
- **T1a (the whole ticket, in essence)**: diff the standalone bs=1 path against the layer's `TINY2`
  path and remove the delta without breaking the layer's epilogue contract (the layer still owes
  the residual/boundary output; only its *cost* is in question). Expected **−7 µs** on decode 1
  = **+1.3% to +1.9%** of the layer score.
- **T1b (only if T1a leaves a gap)**: deepen the tiny build's weight-tile ring / issue the CTA's whole
  slice at launch (shipped as `OKC_WMAJ` in `src/okc_loop/vendored_work/moe/wm_t1.py`, default off).
  Its author's own ceiling estimate is **≤ 8%** of the MoE, so treat it as a small top-up, not a lever.
- **Kill criterion**: if the replica shows the standalone at bs=1 is itself ≥25 µs on our node, the
  22 µs reference is node-specific and the ticket is closed.
- **Do not** build a MonoMoE-style rewrite: the partition it proposes is what the incumbent already
  does. That was the working hypothesis for one afternoon and the generator's `busiest` rotation
  killed it — recorded here so it is not re-opened.
- **Why this survives every previous falsification**: every probe we and keell0renz ran
  (`WGSFCI`, `WGSNX`, `WGSGS`, `WGSDFIN`, `WGSU/WGSUB`, cost tables, `SEGK`, `PDL`, `WPAR`, `SYNC0`,
  `NX`) tuned the **stream-K/phase machinery for T ≥ 17**. At T ≤ 8 the kernel runs a different build
  whose structure nobody changed. The MoE board's own row (2.3746 vs dolanzhao's 2.764) is a separate
  question; this is the layer's largest term at its weakest point.

### T2 — the auxiliary kernels in the small-load steps
`_graft_row_boundary` **8.8–13.9 µs** (2 calls) at verify 16/32 → 2.7–4.5% of those steps;
`_sl_tile_route` + `_sl_cluster_finish` + `_gle_route` + `_graft_route` = 3.7–7.9 µs at decode 1 /
verify 1. Fuse into the adjacent kernel's epilogue or overlap on a side stream (the flashinfer #6683
pattern). Expected **2–15 µs depending on load**, i.e. up to +1.5%. Timeline-driven: fix only what
P0.2 shows as exposed.

### T3 — verify-1 MoE (T = 8): 58–64.6 µs, ~8 distinct experts
If T1b's weight-major structure generalises to T = 8, this is the second-biggest single term
(52% of the `verify 1` step, and `verify 1`/`verify 16` are two of six loads). Reuse T1b's kernel;
only the occupancy target changes. Expected −8 to −20 µs at verify 1 if T1b lands.

### T4 — the 4.1 µs algebra, and whether our harness mis-predicts small loads
Quantify the judge's flat subtraction on the replica. If it is real and asymmetric across loads, some
of our historical "−0.5%" results were measurements of the harness, not of the kernel. Cheap, and it
changes how every future number is read. **Do not queue this behind anything.**

### T5 — the dense/GLE path at the big loads
Already at the wall (cuBLAS picks optimal, falsified from two sides: 0 of 20 routes on the bf16 board,
0 re-picks in the sk2 bake). **Do not spend H200 time here.**

### T6 — remaining PDL edges
`OKC_PDL` is dead code in the current build (its only caller passes `pdl`); `klr_pdlkv` measured
−0.08% of its own time. Only revisit if P0.2 shows a genuine exposed gap between two chained kernels.

### T7 — moonshot, only if T1–T3 land early
Merged-heads attention (16 q heads as one 1024-wide head — halves per-request fixed cost on the
sliding cells). Estimated 0.5–0.8%; large build; strictly after the MoE work.

---

## 5. Judge-quota ladder (unchanged rules, new information)

1. A candidate goes to the judge only after the **replica** shows ≥1.0% on a load-weighted basis with
   a paired A/B CI that excludes 0.
2. One check confirms conformance; if it lands within ±0.3% of the replica's prediction, **submit**.
3. **Two checks per idea, maximum.** Keep ≥4 checks and ≥4 submits for the final bucket
   (00:00 UTC Nov 5 → 07:59 UTC close).
4. No byte-different equivalents to fish for a better draw — the judge dedups identical bytes, and a
   variant that only differs in whitespace is the same measurement with extra variance.

---

## 6. Closed list — do not re-open

Everything in §"Measured and BELOW the row" of `artifacts/bench/dispatch_queue.md`, plus
keell0renz's verbatim dead list (§2.1). Combined: megakernel; fused-slice MoE; L2 prefetch of first
weights; 4-CTA K-split for the qkv GEMV; speculative expert prefetch; L2 persistence windows; lookahead
rings; dual-lane wgmma; per-unit barriers; `WGSFCI` 6/7; `WGSNX` 2/4; `WGSGS` 3; `WGSDFIN` 0/3;
`WGSU/WGSUB` ×1.1; cost tables; `SEGK` up and down; `ST1F`/`ST8F`; `NP8`; `WGSREORD`; PDL-KV
reordering; dense geometry; pooler 12-bit pack secondary; the five-way stack; GLE-at-T8;
per-T MoE self-tune; TMA tile mode; graph-level PDL; code warming; the attention plan sweep; the ATT7
dual build; the VFG sliding route; and the whole drifted `drate` bundle.

---

## 7. Risks

- **Borrowed-node variance.** keell0renz's Modal H200 matched the judge within ~1%; ours must be
  measured, not assumed. P0.1's acceptance test is exactly that check, and until it passes every
  local number is relative-only.
- **The judge's router targets are not public** (`router_load`/`router_distinct` live in the judge's
  case table). Our replica must *search* for the same load, not assume a seed — otherwise T1 is
  measured against a different amount of work than it will be scored on.
- **Top-20 may be contaminated by judge-maxxing** (the current #1 says so of itself). If kernel work
  tops out near 2.35, the remaining gap may not be closable by engineering at all; that is a finding
  to report, not a reason to keep spending.
- **Borrowed time is finite and unscheduled.** Order is therefore strict: P0 before T1 before T2/T3.
  If only four hours exist, spend all four on P0.1 + P0.2 and the T1a build diff — that pair is what
  turns the decode-1 hypothesis into either a number or a refutation.
