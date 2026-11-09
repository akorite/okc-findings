# fd_gh_round2 — GitHub sweep round 2, final day (2026-11-04 ~06:30 UTC)

Re-ran the research_01 query set (`problem=muse_*` ×5, `okc: problem`, `openkernelchallenge.dev`,
plus targeted `wst_ct14b` / `NyoomFloat12` / site-quoted searches), sorted by index date, and
re-checked all priority handles. **Delta vs research_01: 4 real new submission repos + 2
meta/benchmark finds.** All previously-known repos are stale or already harvested — nobody
published a final-day drop overnight.

Hit-count drift vs Nov 3: decode 80→252, moe 133→163, attention 161→120, bf16 38→219, prefill
22→69 (index coverage changes; every unique repo on all fetched pages was inspected).

## 1. NEW repos (not in research_01)

### a) `Madhavan113/ParadigmKernelChallenge` — highest-value find
https://github.com/Madhavan113/ParadigmKernelChallenge (created 2026-10-23, last push 2026-10-29)

A multi-agent (Claude + Codex) campaign that **started from vjeux/okc-2026's published kernels** and
documents ~2 weeks of judged layer work in `notes/decode_layer_map.md` (14 sections, per-shape µs
budgets, every judged job id). Live board: decode **2.20 (#356)**, attention 2.5883 (#340),
moe 2.357 (#335), bf16 1.4933 (#345), prefill 1.796 (#230). Best judged layer 2.1388
(job `75a47c1c`, notes §12).

Mechanisms / findings not on our closed list:

- **v4 CUDA glue kernels** (boundary + routing): one 512-thread CTA per token, 16 coalesced
  columns/thread, one fused four-sum block reduction, no inter-CTA traffic; routing = 1024-thread
  CTA doing top-8 by rank on 64-bit `(ordered_score, 255-e)` keys. **+1..+4 µs on every shape vs the
  Triton glue; took the file 2.098→2.1388.** notes/decode_layer_map.md §12. [We carry a comparable
  grafted glue (`glue_cluster_boundary`, `_boundary_kernel`, route-finish) — parity unknown, worth a
  diff if minutes allow.]
- **PDL max-T = 128**: PDL on at T≤128 gains v16s −1.6..−3.7 µs, v16f −0.5..−1.5; wash at T=256.
  §12. **Already in our file** (`kitchen_sink.py:100` PDL_MAX_T=128).
- **`const __restrict__` PDL hazard**: nvcc emits `ld.global.nc` for const-restrict pointers and
  ptxas **hoists them above `griddepcontrol.wait`** (asm "memory" does not pin them) → dependent
  kernel reads garbage on the first replay, stale-but-identical values after (fails only in graph
  replay, only PDL, invisible in eager). Rule: read predecessor outputs via `__ldcg` or plain loads,
  never `const __restrict__`/`__ldg`. §12, commit 2f4cf4c. **Check our grafted kernels for this.**
- **M1_ROUTE_PIN = all-sk2 at m=1**: the vjeux-fork m=1 autotuner flips between prepares (cgemv vs
  sk2 for qkv/o); pinning sk2 at m=1 is worth up to +7.7 µs at T=1 vs a prepare that drew cgemv
  (§13, lock 682afcb). Our `kitchen_sink.py:6938` sets `GEMV_M1=True` = the *opposite* default
  (vjeux's 12-bit GEMV) — verify our tuner's draw, or pin.
- **TMA bulk-reduction merge for q_len-1 attention** (`OKC_BULKRED`, commit 2026-10-29): stage the
  rescaled partial as a 72-float smem row and use ONE `cp.reduce.async.bulk .add.f32` instead of 272
  `red.global.v4`; ticket release stops waiting on contended atomic acks. Measured on H200:
  ticket 1.37→0.86 µs; b16 sliding −0.5, b1 full −0.3, b32 ~0 µs. Small but positive; we already
  use `cp.reduce.async.bulk` elsewhere (`attn_fill1.py:484`).
- **Last-CTA epilogue folds are dead** (3 attempts): any tail that must wait for the producer's last
  row is a one-CTA memory pipeline — ~130 KB fresh loads/token, 20–50 GB/s effective → +3..+76 µs.
  Boundary folds, route tails, ticket tails all closed. §6/§10/§14.
- **PDL kernels must not end with a grid barrier**: early-launched dependent CTAs are held → +3-5 µs. §8.
- **MoE WGS32 policy build** (`-DOKC_FT_T=32` for 17≤M≤32): bs32 −0.9..−3.8 µs on judge seeds.
  Commit 2026-10-28. MoE-side only.
- **Judge measurement facts**: sglang reference column drifts 2–4% per run; MoE row moves 5–20 µs
  on identical code via the routing draw; "judge component changes by kernel rows, never the
  geomean." Sequential capture-then-time harness gives the first variant +2.6-3.7 µs (round-robin
  required). §6/§11.
- `tools/compose_decode_layer.py` — a component-splicing layer composer (their layer is generated
  from the three problem files). Analog of our vendoring; no new mechanism.

### b) `NathenFernandes/muse-bf16-proj` — a different bf16 12-bit kernel family
https://github.com/NathenFernandes/muse-bf16-proj (pushed 2026-10-24)

v18 = 1.5186× (job `66d8b007f6c641ee`, CI 1.4958–1.5413, 9/9); claims a historical **#1 on the
bf16 board** (results/ranking-history.md). Live board now: **1.502 (#322)** — below our 1.5891 (#80).

- Independent implementation of the same closed axis (12-bit lossless + exact escapes), with
  per-warp **stream-ordered escape lists** and a "geometry" layer separate from vjeux's
  CTA-table design (`submission.py` sm_/m32_* machinery). Same family, different plumbing;
  nothing above our closed 12-bit axis.
- ⚠️ **Compliance data point**: `submission.py` zlib+base64-compresses its CUDA source strings to fit
  the 1 MB cap (`submission.py:2-6`). It passed the judge on Oct 24 — **before** the 2026-11-02
  "no compressed/encoded/obfuscated source" clarification (llms.txt line 36). Do NOT read this as
  precedent that compression is still allowed; it almost certainly is not now, and our tree is
  clean (HANDOFF §8).

### c) `Kushagrabainsla/muse-spark-open` — full campaign writeup
https://github.com/Kushagrabainsla/muse-spark-open (created 2026-10-31, pushed 2026-11-01)

Attention 1.525 (#897), moe 1.704 (#840), decode 1.496 (#811), bf16 1.190 (#1131). Below our ranks
everywhere — value is the experiment log, not the code.

- **Cold-L2 instruction-fetch finding** (attention A13): judge's L2 flush makes once-per-CTA code
  (prologue/epilogue) fetch from DRAM each replay; merge 6928→2298 cycles warm. Fix: `__noinline__`
  merge function run once early by an idle producer warp (stores off). ≈ Our OKC_FINWARM — already
  measured +0.16% and stacked in kitchen_sink.
- **PDL `__ldcg` rule, independently found** (layer L11): predecessor outputs through L1 → residual
  error 0.19. Same rule as Madhavan's (b). Confirms the hazard is real across codebases.
- **Layer L13 finding (new, worth knowing)**: their own FP32 residual-boundary kernel changed
  routing inputs → the judge's fixed expert range is assigned under *reference* routing, so
  exacter FP32 routing gave their MoE ~27 MB / ~7 µs per expert more work (decode 32 +30-40 µs).
  **"Keep SGLang's numerics on the router's input."** Layer-only gotcha: glue math that changes
  routing *at all* changes MoE expert traffic even though routing is judge-fixed.
- Negative results confirming our list: L2 prefetch of next weights during attention = worse
  (latency-bound phases) — twice (layer + MoE M20); 12-bit lossless proj weights ≈ +1% after escape
  fix-up (their E12, a *worse* escape scheme than vjeux's); H100 is not an H200 proxy (M21).

### d) `Mithun-033/Muse-Spark-Open-Kernel-Challenge` — bf16 1.255× writeup
https://github.com/Mithun-033/Muse-Spark-Open-Kernel-Challenge (created + pushed 2026-11-03)

bf16 1.255× (job `79f65b29d2394b45`), rank #518. README is a careful per-shape engineering log:
two-phase non-atomic split-K, wave-quantized grids, `gemv_bf16_vec8` (16B `ld.global.nc.v4` = 98.4%
SOL on qkv_gate m=1), adopted 12-bit compression, **router FP32 hi/lo decomposition**, and
`routertail_small` PTX cache-bypass router (2.19× m=1, **6.47× m=8** — 26.2→4.1 µs).
- Router speedups are the interesting bit but are relative to a weak baseline path (10.6–26.2 µs
  SGLang); absolute numbers ≈ vjeux's sk2 router. Mechanism family (hi/lo router, nc loads,
  warp-shuffle GEMV) already in the closed/published set.
- "NyoomFloat12" attribution in his README (Tri Dao / Daniel Fu / vjeux) is garbled — no such
  artifact exists; the community 12-bit method is vjeux's. [INFERENCE]

## 2. Meta / low-value new finds

- `Pareton-ai/pareton` — a **Bittensor subnet (SN10) running inference-optimization campaigns**;
  contains a draft campaign fixture mirroring the OKC **serving** track (same SGLang commit
  a55ab7d5, 8×H200, ≥10% C16 speedup target, TTFT ≤2× stock, paid in TAO). Not a kernel repo, but a
  signal that serving-track work will also happen on a paid subnet after Nov 10.
  fixtures/campaigns/sglang_muse_spark_open_h200_okc/campaign-fields.json
- `hieptran1812/my-website` — a process blog ("Setting Up the Lab", 2026-10-24): local-judge loop
  on L40S, quota economics. No mechanisms.
- `4gatepylon/LearningML` — a monorepo that *vendored* a copy of `meta_muse_spark100k_challenge`
  (includes lee101/openkernels + their own bf16 submission). Pushed today but just mirrors known work.
- `Tetraslam/muse-kernel-challenge` — stub README only.

## 3. Priority-handle recheck (all stale — nobody published overnight)

| handle | result |
|---|---|
| `pjmullin3` (decode #12, attention #1) | still **0 public repos** — `gh api users/pjmullin3/repos` empty |
| `dolanzhao` (decode #3 now) | 16 repos, all HPC tooling; nothing OKC |
| `gauravjain14` (decode **#1 2.5401**) | only `rl-at-home` (pushed Nov 1, not OKC) |
| `gogowhy` (decode #5) | **404 — not a GitHub user** (like `Abarbir12`) |
| `colinlikescode`, `DTennant`, `zro-inference` | no OKC repos |
| `ZhimaoL` | the wst_ct14b kernels live in **`keell0renz/muse-open-kernel-challenge`** (confirmed via code search); ZhimaoL's own GitHub has only an SGLang fork + DeepGEMM fork, no OKC repo |
| `infraax`, `operatorstack`, `tommyyliu`, `ruwwww` | no pushes since Oct 26–Nov 2; already harvested |
| `lee101/openkernels` | **last push 2026-10-28** (attention v84/v85 parked: "swk batched sliding decode, swf whole-row 128B gathers, cluster per request" — diag experiments, all off by default). Newest versions since research_01 = v82–v85 attention + prefill pf33; all parked/diag, **no unharvested shipped mechanisms** |
| `keell0renz` | pushed 2026-11-02 19:51 UTC — README still describes the same Nov-1 snapshot (decode 2.2659 self-report; live board now 2.2713 #140). No new mechanisms in the diff |
| `rohanarun` (decode #60!, attention #123, bf16 #25, moe #115, prefill #32 — climbed a lot) | Auto-research-agent pushes are benchmark-chart PRs; the 6-model candidate corpus is unchanged |

**Live leaderboard snapshot** (cheap `okc leaderboard --json`, ~06:10 UTC): decode top5 =
gauravjain14 2.5401 / AlexCuadron 2.5226 / dolanzhao 2.5179 / caoshiyi 2.5021 / gogowhy 2.4972;
us #111 (slid 2 from handoff). `Lurrobert` is now a monster account: decode #38, attention **#31
2.8355 screened**, moe #99, bf16 #252 — and publishes only his guild service.

## 4. infraax legality verdict: the untimed-prepare() sweep

Rules text (live, https://openkernelchallenge.dev/llms.txt):
- L135: "`prepare`… is untimed and can run for up to 300 seconds" — an explicit, budgeted resource.
- L129: "Compilation time counts toward the run's time limit but not your score."
- L183: banned = "tamper with the timing or harness, do work outside the timed calls, cache or
  replay outputs, compute at lower precision than the rules allow."
- L181: "final positions = medians of organizer reruns"; "we'll read the top submissions ourselves."

**Verdict: legal, and already mainstream.** Running N variant *timings* inside `prepare()` is
ordinary autotuning — vjeux's own bf16 file does 24-rep cold-L2 graph-replay autotune of every
(layer, m) dispatch table inside prepare (its `_autotune`), and Madhavan's 12-timings-per-T tuner
does the same. The infraax twist is only that the variants are *experiments* (twins deleting one
cost) whose results inform the next submission rather than dispatch — the file still performs
identically at scoring time. It consumes no extra quota (one run = one experiment batch), touches
no timed-call mechanics, and the only cost is prepare's own 300 s budget. Gray edge to watch:
sweeping variants that would *themselves* be submissions purely to map the landscape is defensible
by the letter but could draw review if it delays the judge queue; the variant-count per prepare is
capped by 300 s, not by any rule. **Adopting the pattern: our `prepare()` already times candidates;
the honest version of infraax's trick for us = pack 6-candidate H200 benches into one Supers batch
job — which we already built (tools/supers_batch.py).**

## 5. Mechanisms found, filtered against the §6 dead list

| # | mechanism | board | evidence | value |
|---|---|---|---|---|
| 1 | v4-style CUDA glue (fused 4-sum block reduce + rank-key top-8) replacing Triton glue | decode_layer | Madhavan notes §12, job 75a47c1c | +1..+4 µs/shape (2.098→2.1388 with PDL_MAX_T 128 + locks) — **mostly already ours via grafted glue; diff to confirm** |
| 2 | TMA `cp.reduce.async.bulk` q1 attention merge (one bulk add vs 272 `red.global.v4`) | attention/layer | Madhavan commit 2026-10-29 | −0.3..−0.7 µs on b1/b16 rows; small |
| 3 | M1_ROUTE_PIN (pin sk2 at m=1, kill the per-prepare autotune flip) | decode_layer | Madhavan notes §13 | up to +7.7 µs at T=1 vs a cgemv draw — **check which way our GEMV_M1 draw went** |
| 4 | `const __restrict__` hoists over `griddepcontrol.wait` (use `__ldcg`) | all PDL code | Madhavan §12; Kushagra L11 | correctness hazard audit, not a speedup |
| 5 | Router hi/lo + nc-cache-bypass GEMV | bf16 | Mithun README | family known; 6.47× claim is vs weak baseline |
| 6 | MoE `-DOKC_FT_T=32` WGS32 build for 17≤M≤32 | moe | Madhavan commit 2026-10-28 | bs32 −0.9..−3.8 µs |
| 7 | Pareton SN10 mirrors serving track on Bittensor | serving | pareton fixture | not a mechanism; serving-track intel |

Nothing found changes the HANDOFF §7 plan: kitchen_sink at 00:00 UTC stays first, and no published
kernel on GitHub beats our current row on any board (best published layer-family file = Madhavan's
re-tuned vjeux at ~2.14, below our 2.2837; best published bf16 = NathenFernandes 1.5186 vs our 1.5891;
best published attention mechanism base = the same vjeux file we already carry).

## 6. Not re-investigated (unchanged from research_01)

sid19991 mirror, operatorstack, ruwwww, tommyyliu, PostTrainLLM, iyerramji, jacobpmiller,
Aravind-11, Xuefeng-Zhu, xharshvardhan, lilaclilac09, ozgureyilmaz, Ranoobaba, lee101 (§3),
msaroufim audit, theluckystrike/KingCobra52/Reedo3/arudomanenko stubs, Lurrobert guild,
alygg77, trxphxle — all either stale since Oct 28 or already fully harvested.
