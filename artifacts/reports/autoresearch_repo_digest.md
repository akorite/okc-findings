# `rohanarun/Auto-research-agent` — digest for the OKC Muse loop

**Provenance.** Shallow clone of `github.com/rohanarun/Auto-research-agent` at commit
`018b7a102348ea3b73701e8be52d24f82d8c91ee` (2026-10-30, "Add Claude Opus/Sonnet/Haiku 5.5
benchmark cohort"), 430 files. All `path:line` citations below are against that clone
(`/tmp/ara` at time of reading). Nothing here was run on our hardware; every number is quoted
from the repo, and anything I infer across runs is marked `[INFERENCE]`.

**What the repo actually is.** A macOS SwiftUI harness for long-lived research agents (the app),
plus `benchmarks/` — a *public re-implementation of the same OKC Muse challenge we are in*: five
task snapshots (`benchmarks/data/kernel-starters/muse_*/`), the real scoring rule, a judge that
runs on H200 (`benchmarks/gpu/judge.py`), a calibrated reference table
(`benchmarks/data/calibrated-targets.json`), and 35 sealed candidate sources from six models
(`benchmarks/data/candidates/*/*/submission.py`) with their measured speedups.

---

## 0. Headline (read this if nothing else)

1. **There are no buried CUDA kernel practices.** A repo-wide grep of every `.md`, `.txt`,
   `.json` and `.swift` file for `wgmma`, `tma`, `cp.async`, `launch_bounds`, `register
   pressure`, `occupancy` returns **zero** hits for `wgmma`/`tma`/`cp.async`, and only two
   prose hits for the rest — both of them generic or negative (see rows 15–16 below). The repo
   is an *agent harness*, not a kernel-engineering knowledge base. The only CUDA advice in it is
   five boilerplate "Jump-Start Engineering Guidelines" lists, one per board, which are the
   SGLang-baseline-level obvious wins (preallocate, fuse epilogues, don't sync). We do all of
   them and have closed the harder versions.
2. **The candidate sources are our own upstream.** `gemini-3.8-flash-opencode`'s five
   submissions are the vjeux/`okc-2026` family — the same lineage as our
   `src/okc_loop/vendored_work/bf16/port_rohan.py`. Their bf16 file is *the same document* as
   our port (identical header `# Ported from team vjeux's Open Kernel Challenge entry…` and
   identical module docstring, `benchmarks/data/candidates/gemini-3.8-flash-opencode/kernel-muse_bf16_proj/submission.py:1-20`
   vs `src/okc_loop/vendored_work/bf16/port_rohan.py:1-20`). Our vendored copies for
   attn/moe/layer/prefill are *derived and further along* than these (our layer carries ~80
   `OKC_*` feature flags, `KPERM3`, cuBLASLt plan pins; theirs is upstream +
   `h2uv_v2` + `ub2`). So for kernel technique this repo is **behind** us on 4 of 5 boards.
3. **The three things genuinely worth taking are not kernels:** the judge's timing protocol,
   the agent-loop prompts, and the per-shape calibration table (which is the only public
   headroom map for our boards).

---

## 1. Every extractable practice / finding

"Already?" = does our tree already do this. Column 4 is the verdict.

| # | Practice / finding | Where it lives | Already? | Actionable here? |
|---|---|---|---|---|
| 1 | Preallocate all buffers in `prepare()`; no `torch.empty` inside `kernel_fn` (allocator locks cost µs per call) | `Resources/StarterPrompts/muse_moe_w4a16.md:13`, `muse_decode_layer.md:13`, `muse_prefill_layer.md:14`, `muse_attention_decode.md:13` | Yes — every vendored kernel preallocates; `TASK.md` graph rule forbids allocation in the timed path (`benchmarks/data/kernel-starters/muse_decode_layer/TASK.md:100`) | No |
| 2 | Pre-transpose weights in `prepare()` so reduction-axis loads are coalesced 128 B lines | `muse_bf16_proj.md:16` | Yes — `port_rohan.py` repacks into `mma.m16n8k16` A-fragment order in prepare | No |
| 3 | Specialise `M=1` into a 1-D streaming GEMV (avoid 2-D GEMM launch latency) | `muse_bf16_proj.md:17` | Yes — our bf16 entry has the `m==1` path (`OKC_W0`/`Q8_W0` split in `vendored_work/attn/muse_attention_decode.py:3-5`) | No |
| 4 | Branch-free, statically unrolled inner loops; drop runtime boundary masks when K divides | `muse_bf16_proj.md:18` | Yes — magic-divide + static unroll already in our layer/moe (`OKC_MAGICDIV`) | No |
| 5 | Decouple NVFP4 word decode from the MMA so 8-element decode doesn't spill registers ("decode Word 0 → MMA → Word 1 → MMA") | `muse_moe_w4a16.md:15` | Same symptom, different fix — our MoE does `KPERM3` (permute K inside each 256-K stage so one scale pair decodes once per four k16 steps, `vendored_work/layer/muse_decode_layer.py:13-15`). The repo has **no measurement** showing the interleaved staging beats it | No — no evidence attached; ours is measured at 2.37 |
| 6 | In-register SwiGLU fusion (`silu(gate)*up` in accumulator registers before the store) | `muse_moe_w4a16.md:16` | Yes — fused SiLU·up epilogue | No |
| 7 | Predicated masking (`tl.where`) instead of divergent `if/else` in causal/window loops | `muse_attention_decode.md:16`, `muse_prefill_layer.md:16` | Yes | No |
| 8 | One fused prologue for QK-RMSNorm + RoPE + FP8 cast + paged KV store (replaces 5 SGLang launches) | `muse_attention_decode.md:14`; done by 3 of the 6 models (e.g. `benchmarks/data/candidates/claude-opus-5-5/kernel-muse_attention_decode/evaluation.json` audit: "One fused Triton launch … replacing five launches") | Yes — this is exactly our attention entry's design | No |
| 9 | Align paged-KV loads to 64/128 B transactions along the head dim | `muse_attention_decode.md:15` | Yes — cp.async 4-stage, 128 B staging | No |
| 10 | Cross-stage buffer reuse: one contiguous residual workspace, no intermediate tensors between Attn→Proj→MoE→Norm | `muse_decode_layer.md:13` | Yes — single `s` stream, `OKC_BALPART`/`arena` reuse | No |
| 11 | Epilogue-fuse residual add + RMSNorm into the preceding GEMM instead of separate small launches | `muse_decode_layer.md:15` | Yes — our boundary kernels do this (`_boundary_kernel`, `vendored_work/prefill/prefill_gap1.py` selftest region) | No |
| 12 | Overlap the router GEMM with the tail of attention on a second CUDA stream | `muse_decode_layer.md:14` | **Not applicable** — our layer is a single launch; there is no stream to overlap onto. The advice targets a multi-launch composition (the SGLang-style baseline), i.e. it is advice to *build* what we already replaced | No |
| 13 | Never `.item()`/`.cpu()`/`synchronize()` inside the decoder step; keep routing on device | `muse_decode_layer.md:16`; hard rule `muse_decode_layer/TASK.md:100` | Yes | No |
| 14 | Chunked activation streaming for prefill: materialising `[T,4096]` across 256 experts thrashes L2 | `muse_prefill_layer.md:13` | Not done as stated (we use cuBLAS dense + grouped expert GEMM) — but prefill **does not contribute to the overall board** (`muse_prefill_layer/TASK.md` page: "This problem has its own leaderboard and does not contribute to the overall decode board", quoted at `Resources/StarterPrompts/muse_prefill_layer.md:9`) | No — wrong board for our rank |
| 15 | Nothing on TMA, wgmma, multicast, cluster shape, launch geometry, or occupancy tuning | absent: 0 grep hits for `wgmma`/`tma`/`cp.async` in all prose/config; 6 hits only inside candidate `.py` | We do these; repo adds nothing | No |
| 16 | Explicitly *anti*-heuristic: "a static improvement such as fewer registers, higher occupancy, less shared memory, fewer instructions or a smaller intermediate is a hypothesis, not a speedup" | `Resources/StarterPrompts/OfficialJudge.md:35` | Partly — we have this instinct, but it is not a written gate in our loop | **Yes — make it a gate** (see §2c) |
| 17 | Bounded, data-independent prepare-time autotuner on synthetic inputs with private CUDA events; sanctioned by policy | `benchmarks/PROTOCOL.md` ("the sealed candidate may contain a bounded data-independent prepare-time tuner"); used by 4 of the 6 models (`claude-opus-5-5/kernel-muse_bf16_proj` audit: "a bounded data-independent prepare-time tuner… baseline fallback") | Yes — `moe_selftune`, `bf16_selftune`, `layer_autotune_dense` | No |
| 18 | Diagnostic budget: ≤3 requests, ≤2 cases each, correctness first then the strongest revision | `benchmarks/prompts/kernel-v1.txt:1`; `benchmarks/gpu/research-policy.json` (`max_diagnostic_requests: 3`, `max_cases_per_request: 2`) | We have a check budget + quota ledger (`src/okc_loop/ledger.py`, `orchestrator.py:506-527`), not a per-candidate diagnostic cap | Marginal — our replica makes diagnostics cheap |
| 19 | Reserve a diagnostic for the **exact final source**; if the source changes after the last diagnostic, label it unverified | `benchmarks/prompts/research-v2-lessons.txt:8` | Yes in spirit (we freeze+hash before submit, `SetupTemplate.md:39` behaviour) | No |
| 20 | Choose diagnostic cases that exercise *different* dtype/routing/shape branches — "a passing case0 does not validate multi-token FP32 router inputs or an eight-token attention reduction" | `research-v2-lessons.txt:8` | Yes — we run all scored + nearby cases locally (`cases.py`, `--nearby`) | No |
| 21 | Structured hypothesis portfolio: exactly N objects with `bottleneck`, `mechanism`, `expected_effect`, `lineage`, `verbalized_probability`; malformed/missing portfolio **rejects the probe** | `Resources/ResearchMethodsPolicy.md` (RSI-top §2); enforced in `Resources/ResearchMethods.py:256-257` and validated at `:290-301` | **No** — we have flat "directions" strings + a `seen` dedup (`src/okc_loop/generator.py:146-200`) | **Yes — highest-value loop change** |
| 22 | Randomized *metric views*: each exploration branch sees a random subset of the metric list (`metric_subset`), so branches don't all chase the same dashboard row | `ResearchMethods.py:566-569`; schema `ResearchMethodsPolicy.md` plan JSON (`"metric_subset": 1`, `"portfolio_size": 4`) | No — we broadcast the same context to peers (`agent://all`) | **Yes** — cheap, counteracts the herd |
| 23 | Stall-triggered exploration: `stalled = now - best_time >= stall_seconds`; exploration only then; promotion needs `gain > meaningful_gain` | `ResearchMethods.py:554-555`, `:607`; defaults `Resources/ResearchMethodsDefaults.json` (`branches 4`, `explorationMinutes 5`, `exploitationMinutes 15`, `rounds 3`) | Yes — plateau rule at `src/okc_loop/autoresearch.py:820` ("two consecutive checks moving the geomean by under 1.5%"), which is better calibrated than a wall-clock stall | No |
| 24 | "A source permutation that compiles to identical code is not a new candidate" — check emitted code/resource use before timing | `OfficialJudge.md:37` | **No** explicit check; we rely on measured µs | **Yes — cheap guard** (compare cubin/PTX hash or `--ptxas-options=-v` regs/smem between candidate and parent) |
| 25 | A/A control: unchanged-baseline source is flagged by AST equality and its measured deviation is reported as a *control*, not a gain | `benchmarks/gpu/validate.py:29` (`source_equals_baseline_ast=ast.dump(...)==ast.dump(...)`); `benchmarks/RESULTS.md:7` ("Two GPT-5.6 sources are AST-identical to the baseline; their small measured deviations are controls") | No A/A in `tools/replica_selftest.py` or `judge_replica.py` | **Yes — one A/A run gives us our real noise floor** (we currently quote the judge's ±1.5% instead of measuring ours) |
| 26 | Promotion ledger: incumbent, challenger, frozen source hash, job, decision | `OfficialJudge.md:37`, `Resources/ResearchPolicy.md:27` | Yes — SQLite ledger with provenance (`src/okc_loop/ledger.py`) | No |
| 27 | Keep negative results *at their tested scope* and record what new evidence would reopen them | `ResearchPolicy.md` ("Preserve negative results at their tested scope and identify what new evidence would justify reopening them") | Partly — we keep reports, but our closed-list has no reopen conditions | Marginal |
| 28 | Never resubmit unchanged bytes for timing noise; a duplicate returns the earlier result | `SetupTemplate.md:39`, `OfficialJudge.md:31` | Yes (judge dedups by source hash) | No |
| 29 | The stop condition must require a **new** improvement, not reproduction of a known-good baseline — they had to withdraw a published claim over exactly this | `docs/benchmarks/2026-10-15/README.md:3-5` ("The old `valid=true` stop condition did not require a new improvement … Any implication of such a speedup is withdrawn"); `benchmarks/RESULTS.md:7` | Yes — our loop promotes only on geomean gain | No |
| 30 | Composition beat novel construction: in their parallel run the coordinator got the improvement by *composing existing advances* while workers chased harder local changes and re-audited status | `Resources/ResearchPolicy.md:17` | This is the failure mode of our own long session (94 parked peers, dozens of `artifacts/reports/*.md`) | **Yes — see §2c** |

---

## 2. The prompts, verbatim

### (a) The kernel-run protocol — `benchmarks/prompts/kernel-v1.txt:1`

> GPU BENCHMARK PROTOCOL: Your Mac shell has no CUDA. A file-only bridge submits a source file to
> an isolated H200 with Torch 2.14.1+cu130 / Triton 3.8.0 and SGLang. Run `python gpu_test.py
> solution/submission.py` to queue diagnostic public case 0, or add `--cases 0,1` for at most two
> public case indices. Poll with the printed command. Do not invoke okc, remote services, package
> installation, Internet or account tools. GPU tests are serialized fairly across all contestants;
> queue time counts in the 1,800-second wall-clock cap and is reported separately. The GPU is
> deleted whenever its queue empties. You may continue CPU analysis while waiting. Budget at most
> three diagnostic requests, preferably focused on correctness, then the strongest revision.
> Selected-case speed is not the full target. Preserve all output dtypes and shapes, dynamic fresh
> graph replay, original precision and TASK rules. Optimizing by input/output caching, skipped
> work, checker/timer manipulation, monkeypatching evaluator globals or reading private files
> invalidates a submission. A later full independent held-out check determines the score. Save the
> final self-contained kernel to solution/submission.py AND create solution/submission.json with
> claim and assumptions, even if diagnostics are pending. Set done=true only after the final files
> are saved. Be candid about unverified changes.

**What it would change in our loop:** nothing structural — we already run the full scored set
locally via `tools/bench_bundle.py`/`judge_replica.py`, which is strictly better than a 3-diagnostic
budget. Worth stealing: *"Selected-case speed is not the full target"* as a literal line in the
generator prompt, and the `submission.json` **claim + assumptions** artifact — we have no
machine-readable "what I claim and what I did not verify" record per candidate.

### (b) The post-mortem lessons — `benchmarks/prompts/research-v2-lessons.txt:8`

> Kernel lessons from the initial pilot: reserve a diagnostic for the exact final source version
> and select cases that exercise different dtype, routing, and shape branches. A passing case0
> does not validate multi-token FP32 router inputs or an eight-token attention reduction. Check
> Triton tile ranks, mixed input dtypes and mask identities before spending a GPU request. If the
> final source changes after the last diagnostic, label it unverified; CPU emulation is supporting
> evidence rather than a substitute for actual CUDA compilation and fresh replay. Never repair a
> sealed submission during adjudication.

and the research-side twin at `:6`:

> Lessons from the initial pilot: reserve enough of the research budget to seal a complete,
> self-contained candidate and submission manifest. A promising experiment that never creates an
> in-budget checkpoint is unadmitted. If verification is incomplete, seal the candidate with that
> uncertainty stated plainly rather than implying a pass. Freeze each candidate version before
> starting a full verifier; never mutate its inputs while that verification is running. Run quick
> schema, path and input-binding checks before expensive complete replays. Preserve all mandatory
> verification stages and original mathematical assumptions. These instructions define a new
> prompt version; they must not be represented as the policy used for the original cohort.

**What it would change in our loop:** this is the strongest transferable text in the repo. It is
the "seal before you run out" rule — an agent that has a promising unsealed experiment has
*earned nothing*. Our loop's failure mode is the inverse: many candidates, each measured, none
promoted. Adopt the *freeze → check cheaply → seal* ordering as an explicit instruction, plus
"never repair a sealed submission during adjudication" (we do edit sources between a check and a
submit; that is exactly the hole this closes).

### (c) The measurement discipline — `Resources/StarterPrompts/OfficialJudge.md:35,37`

> A leaderboard result is a whole-problem measurement, not a component estimate. Compare a frozen
> candidate with the current scored incumbent on the judge's own cases, across every scored case
> and the objective's stated aggregation; for local screening, use independent repeats with fresh
> weights and inputs. Use the evaluator's paired ratio and the same timing definition for both
> arms; do not compare eager with graph-replay, raw with overhead-subtracted, or a sub-kernel with
> a whole call. A single-shape win, a parameter-sweep lead, or a static improvement such as fewer
> registers, higher occupancy, less shared memory, fewer instructions or a smaller intermediate is
> a hypothesis, not a speedup. Re-measure a helper or sub-kernel gain after integration into the
> whole call.

> Before timing a variant, check that it changes emitted code or resource use; a source permutation
> that compiles to identical code is not a new candidate. Compose compatible, already-verified
> advances when their interfaces, precision and assumptions match, but validate the composition as
> a new whole candidate instead of adding isolated percentages. Shape- or case-specialized paths
> must preserve the original semantics and path for every other case and include dispatch cost in
> the full-set measurement. Keep a promotion ledger naming the incumbent, each challenger, the
> frozen source hash, the job and the decision. Preserve negative and inconclusive results at their
> tested scope; do not retry unchanged source for timing noise or reopen a closed family without
> new evidence.

**What it would change in our loop:** three concrete gates we do not have —
(i) *emitted-code check* before a timing run (parent vs child cubin/PTX hash, or `ptxas -v`
regs/smem): this would have caught some of our "mechanism scans" that changed nothing;
(ii) *composition is a candidate, not a sum of percentages* — directly relevant to our 2 days left
and our pile of per-mechanism reports;
(iii) the reopen-condition clause, which our closed list (deeper MoE rings, megakernel, …) lacks.

---

## 3. Measured numbers, and the negative results

### 3a. The calibration table (the only public headroom map for our boards)

`benchmarks/data/calibrated-targets.json` (frozen `2026-10-30T02:12:03Z`, policy sha
`07da851d…`) contains the curated private reference kernels' per-shape times on H200, measured
with the same judge. Overall: attention **2.2879**, bf16 **1.3893**, layer **2.1649**, MoE
**2.4527**, prefill **1.5688**.

Cross-referencing that against our own per-shape profiles `[INFERENCE — different runs, so compare
µs and ratios with care; baseline drift between banks is up to ~5%]`:

| board | our row | repo reference | ours vs reference |
|---|---|---|---|
| layer | 2.2837 | 2.1649 | we are **ahead** |
| attention | 2.6590 | 2.2879 | we are **far ahead** |
| bf16 | 1.5891 | 1.3893 | we are **ahead** |
| prefill | 1.9087 | 1.5688 | we are **ahead** |
| MoE | 2.3746 | 2.4527 | reference *appears* ahead — but see below |

Per-load MoE, our `vendored_work/moe/profiles/moe_dup.*.json` (`best_ns`) against the reference's
`candidate_us`:

| load | ours µs | ref µs | our baseline µs | ref baseline µs |
|---|---:|---:|---:|---:|
| decode 1 | 25.47 | 25.42 | 76.78 | 83.76 |
| decode 16 | 66.62 | 93.79 | 187.87 | 193.78 |
| decode 32 | 88.90 | 143.53 | 202.67 | 238.90 |
| verify 1 | 40.54 | 78.22 | 142.69 | 168.45 |
| verify 16 | 140.70 | 174.95 | 457.62 | 481.14 |
| verify 32 | 173.89 | 187.55 | 611.09 | 516.85 |

We are faster in absolute µs on five of six loads and tied on `decode 1`. The reference's higher
*score* comes from a slower baseline bank, not a faster kernel. **Conclusion: the 2.4527 MoE
number implies no real headroom for us; do not chase it.**

Layer, same exercise (`layer/profiles/klr_best.*.json`, our 2.2513 run vs the 2.1649 reference):
we win decode 1 (both layers), verify 1 full, and verify 16; we lose `verify 1 / sliding`
92.48 µs vs 70.22 µs, and lose decode 32 by ~5%. `[INFERENCE]` `verify 1 / sliding` is the one
shape where a known-better number exists in this repo — but their baseline on that shape was also
lower (228.42 vs our 260.69), so part of the 22 µs is machine, not kernel.

### 3b. Per-shape SOL floors (authoritative, from the task pages)

`benchmarks/data/kernel-starters/muse_decode_layer/TASK.md:37-50` — our board's floors at 4.8 TB/s
or 989 TFLOP/s, whichever binds: decode 1 sliding **40.0 µs**, decode 1 full **44.2**, decode 16
76.7 / 137.5, decode 32 84.5 / 208.1, verify 1 62.3 / 66.4, verify 16 126.6 / 188.7, verify 32
143.8 / 267.2. Weighting is 46 for sliding shapes and 16 for full, so sliding shapes carry 74% of
every load. MoE floors at `muse_moe_w4a16/TASK.md`: 11.4 / 52.5 / 70.8 / 37.3 / 101.3 / 108.9 µs.

### 3c. Negative results with provenance

- **Layout-only repacking of one projection is worthless on bf16** — "the starter already uses a
  transpose view, and repacking one ordinary projection did not demonstrate a gain; the unchanged
  router dominated the observed cost" (`Resources/StarterPrompts/muse_bf16_proj.md:11`).
- **65-trial parallel-agent study produced no discovery**: "All 65 valid constructions matched the
  eligible baseline. **No trial demonstrated a new milestone improvement.**" and the endpoint was
  later declared wrong — "The old `valid=true` stop condition did not require a new improvement"
  (`docs/benchmarks/2026-10-15/README.md:3-5, 26-28`).
- **Fast checkpoint sharing showed no measured benefit**: "Thirty of 45 fast-sharing successes
  occurred without a handoff… These data do not establish causal benefits from sharing, diversity
  or agent count" (`docs/benchmarks/2026-10-15/README.md:28-33`). Their own control document says
  the earlier comparison "cannot establish discovery speedup" and was withdrawn
  (`docs/benchmarks/2026-10-15/matched-comparison-controls.md:3-7`).
- **Cohort-wide**: only 23/35 kernel cells beat 1.01× baseline at all, and the best non-Gemini
  result on any board is 1.43× (`benchmarks/RESULTS.md` table). Two "wins" were AST-identical to
  baseline and are reported as controls.

---

## 4. Benchmarking hygiene: theirs vs ours

Their protocol (`benchmarks/PROTOCOL.md` "Admission", `benchmarks/gpu/judge.py:95-108`,
`benchmarks/gpu/timing_stats.py:1-16`, `benchmarks/gpu/scoring-policy.json`):

| element | theirs | ours (`tools/judge_replica.py`, `tools/bench_bundle.py`, `src/okc_loop/colab_harness.py`) | verdict |
|---|---|---|---|
| cold cache | 64 MiB `uint8` buffer zeroed **outside** the graph, `torch.cuda.synchronize()` before each block (`judge.py:97-101`) | 256 MiB flush before **every** replay, plus 5 warmup replays (`colab_harness.py:186,220-235`) | ours stricter |
| timing unit | one event pair around N replays (N=min(20,repeats)), divide by N → block **mean** (`judge.py:29-33`) | one event pair **per replay**, median of 30 (`colab_harness.py:226-235`) | ours more outlier-robust; theirs is the number the leaderboard actually reports — when reproducing a judge number, use a block mean |
| A/B pairing | 5 rounds; **order shuffled per round**; both sides timed in the same loop; median per side over rounds (`judge.py:98-103`) | two candidates run as separate bundles "on the same box in the same ten minutes" (`bench_bundle.py:17-18`); interleaved *replays* within one run, but no cross-candidate order swap | **their model is better for A/B** — adopt the paired round with swapped order |
| aggregation | weighted sum of per-case medians (weight 46/16), ratio per bank, geometric mean across loads (`scoring-policy.json`) | matches (`src/okc_loop/scoring.py:82-110`) | parity |
| uncertainty | bootstrap over `paired_rounds`, 10 000 resamples, **seed 20261030**, reports one-sided 95% lower (`timing_stats.py:5-16`) | deterministic percentile bootstrap on ratios, widened to the judge's ±1.5% (`scoring.py:157-196`) | parity; their fixed seed makes reruns bit-identical — worth copying for our own reports |
| replay overhead | judge subtracts "about 4 µs" as a constant (`muse_decode_layer/TASK.md:96`) | we **measure** it from one-vs-two empty-kernel graphs (`judge_replica.py:17-19,128-164`) | ours better |
| A/A control | AST-identical-to-baseline sources flagged and reported as controls (`validate.py:29`, `RESULTS.md:7`) | **absent** | adopt: one A/A run on our replica |
| fresh inputs | new draw per case/bank; graph replays re-checked after mutating `x`/`topk_ids` and again after restoring routing (`judge.py:73-94`) | new draw per repetition (`judge_replica.py:2` doc, `colab_harness.py:17`) | parity |
| scoring separation | point estimate and timing-support are separate fields; the 1.01× cutoff is a point estimate, CI does not gate it (`scoring-policy.json`, `scoring.py:1-7`) | we gate on CI overlap ("two consecutive … under 1.5%") | theirs is the more honest framing: *report* the CI, *gate* on the point estimate |

---

## 5. Conclusion

**Nothing new on CUDA kernels beyond `port_rohan`.** Concretely:

- The five boards' advice blocks (`Resources/StarterPrompts/muse_*.md:11-19`) are all either
  already in our tree, already closed by us, or inapplicable to a single-launch layer (§1 rows
  1–14). None of them mentions TMA, wgmma, cluster geometry, occupancy targets, or a measurement
  method.
- The repo's kernel sources are the **upstream** of our vendored family, at lower scores on
  attention (2.65 vs 2.29 ref), bf16, prefill and layer. The one apparent exception (MoE 2.4527)
  is baseline-bank drift, disproved load-by-load in §3a.
- Therefore the only transferable value is procedural: the **paired-round + bootstrap timing
  protocol**, the **seal-before-you-run-out** prompt discipline, the **structured hypothesis
  portfolio with randomized metric views**, and two cheap gates (**emitted-code check**,
  **A/A noise floor**).

### Three actions, in value order

1. **A/A control + paired-round timing** — run the same source twice through
   `tools/bench_bundle.py` and once through the paired-round model of `judge.py:98-103`; this
   gives our real noise floor instead of the judge's quoted ±1.5%, and tells us whether any of the
   sub-1.5% deltas we have been arguing about in `artifacts/reports/*` are real.
2. **Emitted-code gate** — before spending a run on a variant, diff its cubin/PTX hash (or
   `ptxas -v` regs+smem) against its parent; reject identical emissions as non-candidates
   (`OfficialJudge.md:37`).
3. **Portfolio prompt** — replace the flat `DEFAULT_DIRECTIONS` list in
   `src/okc_loop/generator.py:151-162` with the RSI-top portfolio contract from
   `Resources/ResearchMethods.py:256-257` (bottleneck / mechanism / expected_effect / lineage /
   verbalized_probability, exactly N, malformed ⇒ reject), and give each subagent a randomized
   `metric_subset` view rather than the same broadcast context.

*Not checked: the Swift app's own prompts (`Sources/AutoResearch/*`) beyond the kernel-launch
templates; the Lean/math half of the repo, which is irrelevant to us; and I did not run any of
their code. Risk: the per-shape comparisons in §3a are cross-run, so treat the layer `verify 1 /
sliding` gap as a hint to profile, not a measured deficit.*
