# research_01 — GitHub code search for OKC submissions (2026-11-03)

Scope: find OKC submission repos beyond the eleven we knew; check whether any top-20 team published;
record published judge scores and mechanisms not on our closed-axes list.

## 0. Method and credentials

`gh` CLI is authenticated (account `akorite`), so **GitHub code search worked** (unauthenticated `api.github.com/search/code`
returns 401; `gh api -X GET search/code` returns 200). Queries run:

- code: `"problem=muse_decode_layer"`, `"problem=muse_moe_w4a16"`, `"problem=muse_attention_decode"`,
  `"problem=muse_bf16_proj"`, `"problem=muse_prefill_layer"` (all pages, deduped by repo)
- code: `"okc: problem"`, `"openkernelchallenge.dev"`, `okc_set_pdl`, `OKC_WGS`, `wst_ct14b`
- repo: `open kernel challenge`, `muse spark kernel`, `okc h200`
- handle sweep: `api.github.com/users/<handle>` + `/repos` for all **59 distinct top-20 handles** (§3)

`problem=muse_*` totals were 80 / 133 / 161 / 38 / 22 hits across the five tags. Every non-`lee101` hit was inspected.

## 1. Headline

**20 new submission-relevant repos** (plus 4 meta/negligible). Three are high-value:

| repo | why it matters |
|---|---|
| [`sid19991/open-kernel-challenge`](https://github.com/sid19991/open-kernel-challenge) | offline copy of the site + **`docs/api/leaderboard-*.json`: full per-board leaderboard dumps with team names** (fetched 2026-10-28) |
| [`lee101/openkernels`](https://github.com/lee101/openkernels) | 109 judged snapshots across all 5 boards + per-shape weak-spot analysis + measured negative results |
| [`tommyyliu/okc`](https://github.com/tommyyliu/okc) | published attention kernel at 1.559× scored / **1.717× unsubmitted**, full design note, per-shape judged times |

## 2. Every repo found

### 2a. NEW — real submissions with published judge scores

| repo | team | published scores (their own numbers) | mechanisms |
|---|---|---|---|
| [operatorstack/open-kernel-challenge-solutions](https://github.com/operatorstack/open-kernel-challenge-solutions) | `bigboateng` | decode_layer **1.7352** (#132/702), moe **1.8320** (#217/794), attention **1.8228** (#196/752), bf16 **1.3066** (#196/1019); peak overall #121/1155 | NVFP4 decode in registers as MMA **A**-operand; swap-AB weight streaming via host-built TMA descriptors; **split-K inside thread-block clusters through DSEM, last-arriver fixed order, release/acquire**; **PDL `griddepcontrol` + L2 prefetch of constant weights before the dependency wait**; fused FP8 attention (QK-norm→RoPE→store→split-K→exact LSE merge→out-norm) in one kernel; warp-specialised **Gluon** kernels; cuBLASLt algorithm pinning with bit-exact self-tests |
| [ruwwww/open-kernel-challenge](https://github.com/ruwwww/open-kernel-challenge) | `ruwwww` (committer `kobeni`) | attention **2.6032** (#69), moe **2.3370** (#85), prefill **1.7950** (#69), decode **2.0751** (#148), bf16 **1.5056** (#206); overall #167–171 | 132-CTA persistent grid over MoE with zero-DRAM intermediates; `.L2::cache_hint` evict-weights/keep-activations; **dynamic dual work queues: fat jobs streamed along K, 16-row skinny jobs with split-K warps, atomic dequeue**; **cuBLASLt heuristic pinning of all 14 prefill GEMMs (nvjet_sm90 descriptors)**; multi-stream router/pooler overlap joined by `cudaStreamWaitEvent` |
| [tommyyliu/okc](https://github.com/tommyyliu/okc) | `tommyyliu` | attention **1.559×** scored (`3819260aba274207`), 1.574× check, **1.717× check unsubmitted**; bf16 1.28312× selected (3764.88 µs), PDL default 1.2311× vs control 1.2048× | **2 KV heads packed per program with a block-diagonal `[16×128]` query so each cache row is one contiguous 128-byte read** (random 64 B ≈ 2 TB/s vs 128 B ≈ 3.0–3.3 TB/s); new tokens on an extra split writing+attending causally in parallel; **two-level tree merge with self-resetting arrival counters** (graph-replay safe); **PDL measured as a +2.2 % bf16 win** |
| [PostTrainLLM/posttrainllm](https://github.com/PostTrainLLM/posttrainllm) | `PostTrainLLM` (issue #206) | attention **1.3723**, moe **1.2236**, bf16 **1.0673**, decode_layer **1.2503**; overall 283/749 on 2026-10-24 | "joint-head FP8 paged attention"; **local-only Marlin scheduling** (schedule computed locally, no global/host sync); norm + local-ID fusion; whole-row small projections. 55 jobs, 13 scored, all fresh-correct |
| [infraax/okc-kernel-challenge](https://github.com/infraax/okc-kernel-challenge) | `infraax` | attention **1.770** (#313), decode **1.433** (#427), moe **1.486** (#525), bf16 **1.170** (#799); overall 437. Archived 2026-10-27 | **measurement-only submissions: 10–40 variants swept inside untimed `prepare()`**, one judge run = one experiment ("twins" deleting one cost at a time) → found exp2/MUFU was only 7 % of attention and a 1.13–1.16× tile-size win in the same run; predict-then-kill promote gate; PTX audit compiled with the judge's real launch attributes |
| [MadanChaollaPark/open-kernel-experiments](https://github.com/MadanChaollaPark/open-kernel-experiments) | `MadanChaollaPark` | attention **1.4807**, bf16 **1.2195**, decode_layer **1.1852**, moe **1.0971**, prefill **1.1206** (all scored, job ids in README) | shape-based fallback attention; fused decoder boundaries; grouped-TMA prefill; compact expert dispatch (histogram/prefix/scatter alignment) 1.1715× vs 1.1595 control. Negatives: custom Marlin stripe scheduling 76/76 correct but 1.066–1.077× vs 1.167× control; exact register-byte FP4 lookup table correct but slower; async weight prefetch slower |
| [iyerramji/open-kernel-challenge](https://github.com/iyerramji/open-kernel-challenge) | `iyerramji` | bf16 best **1.1583×** (#1039/1354), plus 5 more scored bf16 runs with CIs | **"does Modal predict the official result?"** section — Modal whole-workload numbers do not track official scores; typed/validated cuBLASLt records; TinyGEMM restricted to small batches; 4-token router weight reuse |
| [jacobpmiller/open-kernel-challenge-2026](https://github.com/jacobpmiller/open-kernel-challenge-2026) | `jacobpmiller` | decode 1.437, moe 1.632, attention 1.329, bf16 1.179, prefill 1.005; peak overall #155 | `sim/`: CPU-only simulator to estimate speed without GPU runs; `PROGRESS.md` strategy log; `*_dev/` dead ends |
| [Aravind-11/muse-moe-w4a16](https://github.com/Aravind-11/muse-moe-w4a16) | `Aravind-11` | moe **1.629×** (8254 µs), #493/936 | per-shape table: decode-1 1.630, decode-16 1.566, decode-32 1.333, verify-8 1.556, verify-128 1.896, verify-256 1.865 |
| [Xuefeng-Zhu/open-kernel-challenge](https://github.com/Xuefeng-Zhu/open-kernel-challenge) | `Xuefeng-Zhu` | bf16 1.18971×, attention 1.24292× (ranked); 240-case oracle sweep for Quack configs (both rejected) | FA3 "fusion + lookahead" attention; pooler column batch2 hybrid; three-branch integration with per-experiment ledgers |
| [xharshvardhan/open-kernel-challenge](https://github.com/xharshvardhan/open-kernel-challenge) | `xharshvardhan` | bf16 **1.1991** (#639/1109), moe **1.3749** (#579/877), attention **1.2990** (#663/836), decode **1.2252** (#611/758); overall 647/1271 | lossless BF16 repack + fused/split-K router with FP32 accumulation; grouped-query decode with parallel partial combine; notes pin SGLang commit `a55ab7d5` |
| [lilaclilac09/open-kernel-challenge](https://github.com/lilaclilac09/open-kernel-challenge) | `lilaclilac09` | none (starters only) | — (valuable only as a starter snapshot) |
| [ozgureyilmaz/okc-muse-kernels](https://github.com/ozgureyilmaz/okc-muse-kernels) | `ozgureyilmaz` | bf16 **1.1718×** scored (#820/1160 on 10-27); 16 checks, all 9/9 | **the best-argued "how far can we get" document we have** — see §4 |
| [Ranoobaba/okc-campaign](https://github.com/Ranoobaba/okc-campaign) | `Ranoobaba` | campaign, mostly discarded | **LEARNINGS.md = the single densest fact sheet on judge mechanics, shapes and floors** — see §5 |
| [lee101/openkernels](https://github.com/lee101/openkernels) | `lee101` | attention 1.599 (v38), moe 1.418 (v17), decode 1.348 (v7), bf16 1.149 (v11), prefill 1.034 (v6) — all `okc check`, never submitted | see §6 (negative results + per-shape dilations) |
| [rohanarun/Auto-research-agent](https://github.com/rohanarun/Auto-research-agent) | `rohanarun` (**top-20: attention #20**) | starters + 6 model-generated candidate dirs (`gpt-6.1-sol`, `claude-opus-5-5`, …) | none of its own; it is a **model-vs-model candidate corpus** (same problem, 6 different models' submissions) |
| [msaroufim/openkernelchallenge-audit](https://github.com/msaroufim/openkernelchallenge-audit) | — (audit) | n/a | **OKC is KernelBot-derived**: 60/60 benchmark seeds fit GPU-MODE's Cantor-pairing formula with an extra mod 2^63−1; exit code 112 = `VALIDATE_FAIL`; identical stats set; same `submission.py` + directive-comment convention |
| [theluckystrike/muse-spark-open-kernels](https://github.com/theluckystrike/muse-spark-open-kernels) | — | none | 1 KB stub; "published after the boards close" |
| [KingCobra52/muse_bf16_optimization](https://github.com/KingCobra52/muse_bf16_optimization) | — | none | empty (README only) |
| [Reedo3/Muse_Attention_Decode](https://github.com/Reedo3/Muse_Attention_Decode) | — | none | empty repo |
| [arudomanenko/open_kernel_challenge](https://github.com/arudomanenko/open_kernel_challenge) | — | none | empty (gitignore + LICENSE) |

### 2b. NEW — meta / not submissions

| repo | what |
|---|---|
| [Lurrobert/guild](https://github.com/Lurrobert/guild) | a competitor-coordination service: encrypted agent guilds + **public forum** (`guild.1nterface.ai`), live leaderboard-rank badges, top-10/25/50 feed filters. Author `Lurrobert` is **attention #11**. Public posts may be readable without joining — worth a pass if we want competitor technique signal. No kernels. |
| [danyloooah-web/okc-claims](https://github.com/danyloooah-web/okc-claims) | 5 KB of encrypted "Cacheon claim archives"; no plaintext |
| [ruwwww/agent-skills](https://github.com/ruwwww/agent-skills) | same author's agent skill definitions |
| [Okohedeki/autoinference](https://github.com/Okohedeki/autoinference) | market-map of inference techniques; cites OKC, no kernels |
| [lachydotmcg/muse-sputter](https://github.com/lachydotmcg/muse-sputter) | joke repo ("143.8× slower than PyTorch on an RTX 3060") |
| [Saikat-too/muse_spark_kernel_challenge](https://github.com/Saikat-too/muse_spark_kernel_challenge) | 13 KB, README 404 — unverified, likely empty |

### 2c. KNOWN repos — new information only

| repo | new information |
|---|---|
| [keell0renz/muse-open-kernel-challenge](https://github.com/keell0renz/muse-open-kernel-challenge) | self-reported (Nov 1): decode **2.2659**, overall **#38/1085**; attention #35, bf16 #26, prefill #29, moe #58; "~460 experiments" |
| [alygg77/okc-muse-spark-kernels](https://github.com/alygg77/okc-muse-spark-kernels) | self-reported: decode 2.21×, moe 2.40×, bf16 1.55×, prefill 1.82×, attention 2.65× |
| [trxphxle/okc-2026-muse-kernels](https://github.com/trxphxle/okc-2026-muse-kernels) | "peaked top 200 (#185)" |

## 3. Do the top-20 teams publish? (answer: almost never)

Per-board **#1 / #10 / #20** from the 2026-10-28 leaderboard dump
([`docs/api/leaderboard-*.json`](https://github.com/sid19991/open-kernel-challenge/tree/main/docs/api)), sorted by screened speedup,
with the team that holds each position:

| board | n | #1 | #10 threshold | #20 threshold |
|---|---|---:|---:|---:|
| muse_decode_layer | 863 | **colinlikescode 2.3797** (pjmullin3 2.3744 raw) | 2.2609 | 2.2157 |
| muse_attention_decode | 943 | **pjmullin3 3.0316** | 2.6928 | 2.6537 |
| muse_moe_w4a16 | 989 | **Abarbir12 2.6196** | 2.5026 | 2.4362 |
| muse_bf16_proj | 1240 | **colinlikescode 1.6367** | 1.5764 | 1.5624 |
| muse_prefill_layer | 572 | **ericligit05 2.0043** | 1.8829 | 1.8180 |

Cross-checked against two independent snapshots that agree: `infraax` (2026-10-27 07:30 UTC: attention #1 3.03 / #10 2.62;
decode #1 2.38 / #10 2.23; moe #1 2.62 / #10 2.45; bf16 #1 1.64 / #10 1.56) and `Ranoobaba` (2026-10-25: attention #1 2.910 / #50 2.185;
decode #1 2.285 / #50 1.885; moe #1 2.523 / #50 2.126; bf16 #1 1.610 / #50 1.444; prefill #1 1.888 / #50 1.626; 1077 teams).

**Handle sweep result.** I queried GitHub for all 59 distinct top-20 handles. **Exactly one top-20 team publishes a
submission repo we can read: `rohanarun` (attention #20), and it is a model-candidate corpus, not their kernel.**
`Lurrobert` (attention #11) publishes only the guild service. The per-board leaders have **nothing**:

| handle | boards led | GitHub presence |
|---|---|---|
| `pjmullin3` (#1 attention, #2 decode, #2 moe, #5 prefill, #7 bf16) | strongest single account on the board | **user exists, 0 public repos** |
| `colinlikescode` (#1 decode, #1 bf16, #2 attention, #3 moe) | | 8 repos, none OKC (`NanoGPT-Speedrun-Winner`, `Open-Inference`, …) |
| `Abarbir12` (#1 moe) | | **404, no such user** |
| `ericligit05` (#1 prefill), `cu788` (#2 prefill) | | no OKC repos |
| `dolanzhao` (moe **#11**, not #1 — the roster's premise is stale) | | 16 repos, all HPC profiling (HPCToolkit, DynamoRIO, XLA); none OKC |
| `zzz0906` (#3 prefill, #3 attention, #9 moe, #11 bf16, #13 decode) | the most consistently strong account | 21 repos: `vllm`/`flashinfer`/`sglang` forks + blog; **no OKC repo** |
| `tjiang-fair` | — | **absent from all five leaderboards** in the dump; no GitHub user |
| `arashattar` | — | **absent from all five leaderboards**; no GitHub user |

Only three top-20 handles hold *any* kernel-ish repo, and none is an OKC entry:
[`caoshiyi/eat-kernels`](https://github.com/caoshiyi/eat-kernels) (Kimi-Delta-Attention kernels for MI350X/B200, agent-evolved,
NeurIPS'26 paper — `caoshiyi` is decode #6, moe #5, prefill #9), [`tie-pilot-qxw/pdl-megakernel-reconstruction`](https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction)
(Aug 2026, pre-dates OKC; `tie-pilot-qxw` is decode #12, moe #7, prefill #17), and
[`gauravjain14/kernel-lens-jev`](https://github.com/gauravjain14/kernel-lens-jev) (moe #4, prefill #14).

**Rank-drift caveat (important).** This dump is one-per-team and dated 2026-10-28; the boards densified hard afterwards.
`keell0renz` reports decode 2.2659 → **#38** on Nov 1, while the same 2.26 band was #5–#6 on Oct 28. Treat the **team names**
as durable and the numeric thresholds as a lower bound: today's #10 thresholds are higher than the table above
(`infraax`'s own "top-10 target" for bf16 was 1.5696 / 3077.632 µs on 2026-10-27, and ozgureyilmaz's Sep-of-latency arithmetic
in §4 shows how tight it is). [INFERENCE] — I could not read the live board (no `okc` runs allowed), so the current
per-board #1 identity may have changed; the *publication* answer (they don't publish) is robust because a repo, once public, stays public.

## 4. `ozgureyilmaz` bf16 gap analysis — the most useful single document found

From [`notes/top10-next-round.md`](https://github.com/ozgureyilmaz/okc-muse-kernels/blob/master/notes/top10-next-round.md),
their v027 measured 1.1804× check / 1.1718× scored, **rank 820/1160**, and the **live rank-10 threshold was 1.5696× / 3077.632 µs**
(2026-10-27 16:53 UTC) — i.e. closing the gap needed another **24.8 % latency reduction** (~33 % more score).

Their per-load budget table (TASK floors = theoretical minimum per scored load, µs):

| load | v027 total | budget for 1.5696× | TASK floor |
|---|---:|---:|---:|
| Decode 1 | 3008 | 2331 | **1792** |
| Decode 16 | 3994 | 2955 | **2573** |
| Decode 32 | 4208 | 3044 | **2585** |
| Verify 1 | 3198 | 2972 | **1798** |
| Verify 16 | 4786 | 3380 | **2716** |
| Verify 32 | 5988 | 3978 | **3212** |

Two hard ceilings worth carrying into our bf16 plan:
- Replacing **every** router call by its TASK floor caps the score at **≈1.306×** — router work alone cannot reach the top ten.
- Verify-32 is the weakest load; its pooler is 31.7 %, qkv_gate 20.2 %, unpooler 25.0 % of candidate time. **The dense layers dominate.**
- The TASK floors assume **uncompressed** weights, so lossless packing legitimately moves the floor (their proposed
  scheme: 32-value block, 8-bit sign+mantissa, 4-bit exponent deltas, shared exponent byte = 49 B vs 64 B = 12.25 bits/value, exact fallback).
  This is our closed 12-bit axis, and `ruwwww` independently shipped it (1.5056×) — so it is **not** unpublished.

## 5. Judge/board facts from `Ranoobaba/okc-campaign/LEARNINGS.md` (dense, verified 2026-10-25)

[URL](https://github.com/Ranoobaba/okc-campaign/blob/master/LEARNINGS.md). Items that change strategy rather than micro-tuning:

- **The prefill board carries no overall credit** ("separate board, no overall credit"). Overall requires **all four decode problems solved**
  (every top row has `solved=4`). Prefill is prize/board-only.
- **Final positions are medians of organizer reruns on scored *and nearby* shapes.** No overfitting to the exact m/ctx — overfitting is punished at scoring time.
- Judge = CUDA-graph replay, cold L2 before each replay, paired A/B ratios, **~4 µs replay cost subtracted**.
- Environment in `lmsysorg/sglang:dev-muse-spark-open`: torch 2.14.1+cu130, triton 3.8.0, flashinfer 0.7.0.post1,
  sgl_kernel 0.4.9, deep_gemm 0.2.1.post1+muse, tilelang 0.1.12, cutlass 4.8.0, nvcc 13.0, py 3.12; H200 SXM 141 GB, SM max 1980 MHz.
- Scored shapes are public (`problems/*/meta.json`, including seeds); the *tests* use different seeds.
- Attention: "**ctx ~80K ±7.5 %** agentic spread", sliding window 2048 with interleaved RoPE, **full layers have no RoPE**,
  page_size **1** in scored cases and **64** in tests; test heads include **"dp" (128/16)**, ctx up to **1M**, `kv_scale 1/64`, window-edge keys.
  Sliding floor ≈ 0 → "launch count and latency dominate decode-1 loads".
- MoE: output = `sum_g g * rmsnorm((y*(1+w)).bf16())`, tolerance **1.5 %**, 28 MB per touched expert, all scored shapes weight-read bound.
- Their replica calibration: baseline-vs-itself 1.0036 (proj) / 1.0012 (moe); per-shape noise ±2–3 % at 10 reps → **≥20 reps** for keeps under 5 %.
- bf16 floors at m=1 (SGLang vs HBM floor, µs): qkv_gate 18.3/7.9, o 9.7/3.5, pooler 29/14, unpooler(1024 rows) 9.4/1.7, router 13.0/1.8.
  Biggest single wins: router at every m, unpooler slice, `o`.

## 6. Mechanisms not on our closed-axes list

Ranked by expected value; each is measured by the team that published it.

1. **Head-packed 128-byte cache reads with a block-diagonal query** — `tommyyliu`, attention.
   Two KV heads per program, query staged as `[16 rows × 128 cols]` block-diagonal so each gathered paged-cache row is one
   contiguous **128-byte** read: "random 64-byte reads top out at ~2 TB/s on H200; 128-byte reads reach ~3.0–3.3 TB/s."
   They score 1.559× published and 1.717× on the same code, and their kernel is one Triton launch per call.
   [README](https://github.com/tommyyliu/okc/blob/main/muse_attention_decode/README.md) · [design](https://github.com/tommyyliu/okc/tree/main/muse_attention_decode)
2. **PDL (`griddepcontrol`) between dependent kernels, plus L2 prefetch of constant weights *before* the dependency wait.**
   `tommyyliu` measures PDL as their bf16 default: 1.2048× control → **1.2311×** PDL (`e6f6e1af0bd446a1`), and separately
   tested "earlier PDL signaling" (no further win). `operatorstack` uses PDL + prefetch and ships `okc_set_pdl`-style helpers.
   The starter already exposes `okc_set_pdl`, so this is legal-by-construction and not on our closed list.
   [tommyyliu results](https://github.com/tommyyliu/okc/blob/main/muse_bf16_proj/results/README.md) · [operatorstack](https://github.com/operatorstack/open-kernel-challenge-solutions)
3. **Two-level tree merge with self-resetting arrival counters** (`tommyyliu`): split-K partials → workspace; last arriver per
   group merges the group, last group-merger finishes and applies the BF16 output boundary + per-head RMSNorm + sigmoid gate.
   Counters reset themselves so CUDA-graph replay is safe. Not the same as our closed "per-unit barriers" (that is intra-CTA);
   this is an inter-CTA tree with no device sync.
4. **Dynamic dual work queues over experts** (`ruwwww`): fat jobs streamed along K, 16-row skinny jobs get split-K warps,
   dequeued by `atom.inc`. Our closed list has "heaviest-expert-first (already present)" but not a **two-class** split; adjacent, worth a look.
5. **cuBLASLt descriptor pinning** (`ruwwww` 14 GEMMs → `nvjet_sm90`; `iyerramji` "typed and validated cuBLASLt records";
   `tommyyliu` "pinned cuBLASLt algorithms with bit-exact self-tests"). Purpose: remove `prepare()`-time heuristic jitter so the
   timed GEMM is deterministic. Cheap, small, and orthogonal to every closed axis.
6. **Measurement-only submissions** (`infraax`): run 10–40 variants inside the untimed `prepare()` so one judge quota slot yields a full
   factorial experiment ("twins" that delete one cost at a time). Their finding: attention's expected bottleneck (exp2/MUFU) was **7 %**,
   and the same run surfaced a 1.13–1.16× tile-size win. This is the highest-leverage *method* in the whole corpus for our per-board quota.
7. **`red.release` add instead of `fence` + per-tile atomic** (`lee101` moe v17, 1.418×): they measured that ptxas warp-aggregates
   `atomicAdd` (atomic + shfl) and the shfl waits on the contended queue atomic → use `atom.global.inc` asm. Micro, but they credit the queue.
8. **P row-sums on the tensor core via a ones-wgmma** (`lee101` attention v38, 1.599×): the softmax denominator is produced by an
   extra WGMMA against a ones vector rather than a separate reduction pass.

### Negative results we should not spend quota on (all measured on H200 by the publisher)

- `mma.sync ... e4m3` is **emulated on sm_90** (ptxas expands to F2FP+HMMA) → no gain; real FP8 needs WGMMA (`lee101`).
- **L2 bulk prefetch ahead of the cp.async ring is much slower on full shapes** (`lee101`, PFF/PFD 2–12).
- **Warp-specialised MoE GEMM parked after three builds** (`infraax`): copy-issue bound → consumer bound → ring-protocol floor ≈ old kernel's rate.
- **Custom Marlin stripe scheduling**: 76/76 correct, 1.066–1.077× vs a 1.167× control (`MadanChaollaPark`).
- **Exact register-byte FP4 lookup table**: bit-proof and 76/76 checks, GEMV slower than native (`MadanChaollaPark`).
- **All-small-M TinyGEMM, transposed `torch.mm` layout flip, explicit K-partial split-K on short-K O/unpooler, deeper WGMMA pipeline, cluster variants**
  — all regressed for `ozgureyilmaz`/`tommyyliu`; `ozgureyilmaz` states "do not rerun this split-K design".
- **Modal/self-hosted replica scores do not predict official scores** (`iyerramji`, dedicated section; `MadanChaollaPark`, `ozgureyilmaz` separate check vs scored by ~1 %).

## 7. Cross-cutting detail worth stealing

- **OKC is KernelBot-derived and its scored inputs may be reproducible**: 60/60 benchmark seeds fit GPU-MODE's Cantor-pairing
  seed formula plus `mod 2^63−1`; exit code 112 = `VALIDATE_FAIL`; six-statistic output identical.
  [`msaroufim/openkernelchallenge-audit`](https://github.com/msaroufim/openkernelchallenge-audit) (probe scripts + JSON evidence in-repo).
  If seeds are deterministic we can regenerate the judge's exact scored inputs instead of guessing at shape spread. [INFERENCE on usability]
- **`sid19991/open-kernel-challenge`** is a full offline mirror of the site, `okc` 0.7.2 binaries for 4 platforms, the pinned
  SGLang image refs, and per-problem `TASK.md`/`baseline.py`/`reference.py`/`tests.json` — a local source of truth that needs no network.
- `MadanChaollaPark` and `GiulianoIntelligence` publish **judge profile dumps per job** (`profiles/submission.<job>.{json,txt,log}`) —
  the closest thing to ncu output from inside the judge image that anyone has released.
- Quota as observed: 10 scored submissions and 10–20 checks per problem per day, resetting 00:00 UTC (`infraax`, `ozgureyilmaz`, `Ranoobaba`).

## 8. What I did not find

No repo anywhere contains a decode-layer kernel beating 2.38×, no repo contains a `muse_decode_layer` score above our 2.2837 except
`colinlikescode` (2.3797, unpublished), and no top-20 leader has published source. **The published frontier on every board is
1.5–2.0 rank-tiers below the board frontier**, so repo mining is now a source of *mechanisms and negative results*, not of scores.
