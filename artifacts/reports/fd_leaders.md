# fd_leaders — reverse-the-leaders (2026-11-04, ~06:15 UTC)

Final-day snapshot of all five boards + public-artifact hunt for every top-10 team.
Sources: `okc leaderboard <board> --json` (live, 2026-11-04), the 2026-10-28 dumps in
[`sid19991/open-kernel-challenge/docs/api`](https://github.com/sid19991/open-kernel-challenge/tree/main/docs/api),
GitHub API (`gh api users/<handle>/repos`, `/gists`), and the public Guild forum at <https://guild.1nterface.ai>.
[INFERENCE] tags mark guesses; everything else is cited.

## 0. Headline deltas since the Nov-3 sweep

- **All five #1s and the entire top-10 shifted in 7 days.** 2026-10-28 → now: decode #1 2.4029→**2.5401** (gauravjain14, was unranked top-10),
  attention #1 3.0316→**3.3389** (pjmullin3, same person +0.31), moe #1 2.6853→**2.7642** (dolanzhao), bf16 #1 1.6367→**1.6940** (gogowhy),
  prefill #1 2.0302→**2.0754** (gauravjain14). ex-leaders colinlikescode (#1 decode+bf16) and Abarbir12 (#1 moe) are out of every top-10.
- **The Guild forum (<https://guild.1nterface.ai>) is readable without an account** — post bodies public, only comments from non-top-50
  authors are hidden. Newest post is 1 h old (arashattar). It is the single best public signal on what works/doesn't (§4).
- **Three new submission repos since Nov 3:** `Mithun-033/Muse-Spark-Open-Kernel-Challenge` (bf16 1.255×, published 11-03,
  cites the community "C12" lossless codec), `Aravind-11/muse-moe-w4a16-kernels` (moe 1.8696, PR-per-speedup log, 11-03),
  `Kushagrabainsla/muse-spark-open` (4 boards, 10-31, mid-pack but good mechanism notes). Plus `vjeux/okc-2026`
  (full 5-problem source, Apache-2.0, 10-27) and `trxphxle/okc-2026-muse-kernels` which contains the best decode-leader decomposition.
- **No top-10 team on any board has published their leading kernel.** Consistent with the Nov-3 finding — the published frontier
  remains ~1.5–2 rank-tiers behind the board frontier.

## 1. Top-10 tables (live, `okc leaderboard --json`)

submitted_at = UTC; value = µs. Full JSON kept in /tmp during run; regenerated on demand.

### muse_decode_layer (1251 ranked; akorite #111 @ 2.2837)

| # | team | speedup | µs | att | job | submitted |
|--|------|--|--|--|--|--|
| 1 | gauravjain14 | 2.5401 | 8905.2 | 19 | bd4cc135d74d43c6 | 11-04 04:52 |
| 2 | AlexCuadron | 2.5226 | 8967.0 | 29 | 22dcc6ecab784f56 | 11-03 20:58 |
| 3 | dolanzhao | 2.5179 | 8983.9 | 35 | ef285182cb0647ba | 11-04 02:57 |
| 4 | caoshiyi | 2.5021 | 9040.5 | 22 | 3338748e39a74d1f | 11-04 01:28 |
| 5 | gogowhy | 2.4972 | 9058.3 | 19 | 439e45c3116e4308 | 11-03 22:43 |
| 6 | tonybolivar | 2.4854 | 9101.3 | 81 | f483c380607b43b9 | 11-04 04:14 |
| 7 | tjiang-fair | 2.4844 | 9105.0 | 43 | c3aec8f7f1c54442 | 11-03 16:38 |
| 8 | lxe | 2.4708 | 9155.1 | 72 | c3e4a6ed8ebc4a1c | 11-03 21:37 |
| 9 | Charbel199 | 2.4688 | 9162.5 | 9 | 3c281dd65a8d41bb | 11-03 19:34 |
| 10 | mpeng19 | 2.4656 | 9174.4 | 49 | 8cd42483edfa47f0 | 11-03 22:27 |

### muse_moe_w4a16 (1381 ranked; akorite #260 @ 2.3806)

| # | team | speedup | µs | att | job | submitted |
|--|------|--|--|--|--|--|
| 1 | dolanzhao | 2.7642 | 4865.2 | 40 | e0287e5044384e8d | 10-31 22:36 |
| 2 | pjmullin3 | 2.7373 | 4913.1 | 66 | 799e065688e34520 | 11-03 09:08 |
| 3 | Fangzhou66 | 2.7172 | 4949.3 | 4 | 19628bd551b44a54 | 11-03 23:21 |
| 4 | tjiang-fair | 2.6969 | 4986.7 | 61 | 045b7b56922d4fa8 | 11-02 23:31 |
| 5 | Charbel199 | 2.6794 | 5019.3 | 11 | fbbd67d9bcac4fe6 | 11-03 11:23 |
| 6 | olety | 2.6783 | 5021.3 | 97 | 60a8cf126e0a4ff6 | 11-04 00:01 |
| 7 | reyanshsolis | 2.6720 | 5033.1 | 26 | 0154ecc6c797405a | 11-03 12:25 |
| 8 | k9thebeast | 2.6714 | 5034.2 | 14 | 6837b060ad674bdb | 11-02 23:20 |
| 9 | mpeng19 | 2.6700 | 5037.0 | 30 | 0e2add92fca542ca | 11-01 03:00 |
| 10 | DTennant | 2.6613 | 5053.4 | 38 | f918590a48694a6c | 11-03 17:40 |

### muse_attention_decode (1328 ranked; akorite #149 @ 2.659)

| # | team | speedup | µs | att | job | submitted |
|--|------|--|--|--|--|--|
| 1 | pjmullin3 | 3.3389 | 1128.0 | 19 | d64f5ae375bd434d | 10-31 22:05 |
| 2 | tjiang-fair | 3.2221 | 1168.9 | 18 | 7d1f783733cf49c6 | 10-30 00:39 |
| 3 | olety | 3.1293 | 1203.6 | 97 | b7ff567fad554bd7 | 11-03 06:11 |
| 4 | jt914 | 3.1125 | 1210.0 | 18 | 47adadf28dec4415 | 11-04 04:42 |
| 5 | gauravjain14 | 3.0579 | 1231.7 | 23 | dc32b1f7cb454af4 | 11-04 04:12 |
| 6 | LeshengJin | 3.0542 | 1233.2 | 2 | 342ee78f9ebf4858 | 10-28 22:26 |
| 7 | Fangzhou66 | 3.0322 | 1242.1 | 7 | 17b0901e110c442a | 11-03 23:41 |
| 8 | mgpai22 | 3.0133 | 1249.9 | 75 | b4de548c799b4a89 | 11-02 23:39 |
| 9 | zzz0906 | 3.0077 | 1252.2 | 25 | f6664be67ca54e57 | 10-28 12:44 |
| 10 | colinlikescode | 2.9969 | 1256.7 | 31 | 1c4f41dbee7f49ba | 10-26 07:23 |

### muse_bf16_proj (1694 ranked; akorite #80 @ 1.5891)

| # | team | speedup | µs | att | job | submitted |
|--|------|--|--|--|--|--|
| 1 | gogowhy | 1.6940 | 2851.7 | 13 | 71cf794843564b0c | 11-03 08:29 |
| 2 | olety | 1.6881 | 2861.6 | 95 | b7ed6892b1e64b58 | 11-04 03:09 |
| 3 | dolanzhao | 1.6872 | 2863.2 | 34 | 78f3016daa774337 | 11-04 04:16 |
| 4 | reyanshsolis | 1.6835 | 2869.5 | 24 | e021b190a83c47e2 | 11-04 04:44 |
| 5 | gauravjain14 | 1.6713 | 2890.4 | 17 | 64a069a7d172434b | 11-03 08:05 |
| 6 | jt914 | 1.6609 | 2908.6 | 9 | e6bddb7ba34a493e | 11-04 04:25 |
| 7 | tjiang-fair | 1.6596 | 2910.8 | 50 | fcbd655f49b043e9 | 11-02 04:44 |
| 8 | sudodaksh | 1.6534 | 2921.7 | 40 | 4358d22fd08e4888 | 11-03 21:59 |
| 9 | Fangzhou66 | 1.6468 | 2933.5 | 6 | 74c619de872b4683 | 11-03 23:41 |
| 10 | miketung | 1.6460 | 2934.8 | 25 | 9f7efe7b29894953 | 11-03 17:47 |

### muse_prefill_layer (858 ranked; akorite #99 @ 1.9087)

| # | team | speedup | µs | att | job | submitted |
|--|------|--|--|--|--|--|
| 1 | gauravjain14 | 2.0754 | 186664.2 | 25 | da8d262247244d48 | 11-04 00:39 |
| 2 | ericligit05 | 2.0727 | 186907.0 | 15 | 929d13bac04748c4 | 10-31 21:40 |
| 3 | olety | 2.0623 | 187854.9 | 89 | 67166e3ca39e4796 | 11-03 08:18 |
| 4 | Fangzhou66 | 2.0568 | 188357.1 | 6 | c630c58de7db40ed | 11-03 23:41 |
| 5 | pjmullin3 | 2.0433 | 189601.8 | 52 | f931bb0d83d54fd0 | 11-02 21:01 |
| 6 | DTennant | 2.0427 | 189650.1 | 28 | 7ed71111217f4a3a | 11-01 06:23 |
| 7 | k9thebeast | 2.0378 | 190110.5 | 11 | d58bf485f69f4493 | 11-04 05:22 |
| 8 | zzz0906 | 2.0325 | 190604.4 | 25 | b08e5d36bce14fb2 | 10-29 00:01 |
| 9 | vaktibabat | 2.0267 | 191151.5 | 36 | ea07c46750b5407c | 11-04 01:03 |
| 10 | mgpai22 | 2.0195 | 191834.5 | 68 | c007fdbe7f90417a | 11-01 00:06 |

## 2. Leaderboard diff, 2026-10-28 dump → today

Old ranks from `sid19991/.../docs/api/leaderboard-*.json` (one row per team, sorted).

| board | notable movers |
|---|---|
| decode | **gauravjain14 NEW → #1**; AlexCuadron #123→#2; dolanzhao NEW→#3; gogowhy NEW→#5; tjiang-fair NEW→#7; Charbel199 NEW→#9. pjmullin3 was raw-#1 (2.4029) now #12; colinlikescode #2→#36. |
| moe | dolanzhao #2→**#1**; Fangzhou66 NEW→#3; tjiang-fair NEW→#4; Charbel199 NEW→#5; k9thebeast NEW→#8; olety #20→#6; reyanshsolis #16→#7. Abarbir12 (#1→out), zzz0906 (#4→#18). |
| attention | pjmullin3 #1→**#1, +0.31 to 3.3389**; tjiang-fair NEW→#2; jt914 NEW→#4; gauravjain14 NEW→#5; LeshengJin NEW→#6; Fangzhou66 NEW→#7. zzz0906 #2→#9 (identical 3.0077 score → only their rank moved, i.e. the frontier moved past them). |
| bf16 | gogowhy NEW→**#1**; dolanzhao NEW→#3; reyanshsolis **#260→#4** (+0.28 in a week); gauravjain14 NEW→#5; jt914 NEW→#6; Fangzhou66 NEW→#9; tjiang-fair #162→#7. colinlikescode #1→#15. |
| prefill | gauravjain14 **#16→#1** (+0.23); Fangzhou66 NEW→#4; k9thebeast NEW→#7; vaktibabat NEW→#9; zzz0906 #1→#8 (same 2.030 score). |

Pattern: a cohort (gauravjain14, Fangzhou66, jt914, tjiang-fair, Charbel199, olety, k9thebeast, gogowhy) landed >2.4-decode / >2.6-moe /
>3.0-attention / >1.64-bf16 all within the last week — consistent with a shared mechanism family (agent-driven search + the same public
lineages: vjeux/okc-2026, alygg77, keell0renz lineage named on Guild). [INFERENCE that they share, not copy; attempts 4–97 vary widely.]

## 3. One line per leader (all 25 top-10 teams, all five boards)

Published mechanism = anything public; else "nothing public". Inferred family from scores + repos + lee101/trxphxle/guild analyses.

| team | boards+rank | published | inferred mechanism family | citation |
|---|---|---|---|---|
| **pjmullin3** | attn **#1 3.339**, moe #2, prefill #5, bf16 #14, decode #12 | nothing public (GitHub user exists, 0 repos, 0 gists) | Attention at ~gather-ceiling on full shapes + near-zero fixed latency on sliding/B=1 (see §5); MoE at ~97% of the 4.35 TB/s stream floor | [users/pjmullin3](https://github.com/pjmullin3); lee101 gap doc §below |
| **gauravjain14** | decode **#1**, prefill **#1**, attn #5, bf16 #5, moe #14 | no OKC repo; `kernel-lens-jev` (Sep), `rl-at-home` (Nov-01) | Composite: near-floor MoE + fused attention + per-stage glue elimination (the decode-leader margin is MoE ~15%/attn ~21%/dense ~6.5% ahead of vjeux-class base) | [trxphxle README §"leader margin"](https://github.com/trxphxle/okc-2026-muse-kernels); [repos](https://github.com/gauravjain14?tab=repositories) |
| **dolanzhao** | moe **#1 2.764**, decode #3, bf16 #3 | nothing OKC; 15 HPC/profiling repos (HPCToolkit author — "Performance Engineer / Floating-Point Therapist") | MoE ≈97.4% of the 2.84 geomean stream-floor ceiling → every load within ~1.05 of the 4.35 TB/s bulk-stream floor; zero-DRAM intermediates + fused rmsnorm/gate | lee101 moe_gap_analysis floor table; [repos](https://github.com/dolanzhao?tab=repositories) |
| **gogowhy** | bf16 **#1 1.694**, decode #5, moe #16, attn #16, prefill #14 | **no GitHub user at all** (404) | Every bf16 shape within ~1.5 µs of its C12-compressed/FP32 streaming floor + beating cuBLAS at m128 with compressed weights (lee101's model of the 1.637 leader, now extended to 1.69) | lee101 bf16_gap_analysis §6; gh api users/gogowhy 404 |
| **tjiang-fair** | attn #2 3.222, moe #4, decode #7, bf16 #7 | nothing public (user exists, 0 repos) | Same attention family as pjmullin3 but ~4% weaker (mostly sliding/B=1 latency); strong all-rounder | [users/tjiang-fair](https://github.com/tjiang-fair) |
| **AlexCuadron** | decode #2 | nothing OKC; OpenHands/Letta agent repos | decode-only specialist; same composite as gauravjain14 minus prefill | [repos](https://github.com/AlexCuadron?tab=repositories) |
| **olety** | attn #3, bf16 #2, moe #6, prefill #3, decode #46 | nothing OKC (83 repos, all unrelated craft/tooling) | Broad all-rounder; bf16 1.688 within 0.4% of #1 → same compressed-floor regime | [repos](https://github.com/olety?tab=repositories) |
| **jt914** | attn #4, bf16 #6, decode #11 | nothing public (0 repos, "Joseph Tso") | Same attention family (3.11) | [users/jt914](https://github.com/jt914) |
| **LeshengJin** | attn #6 (2 attempts only) | nothing OKC (sglang/flashinfer forks, stale) | Attention specialist — 3.054 in 2 attempts suggests an off-board-proven kernel or lineage reuse [INFERENCE] | [repos](https://github.com/LeshengJin?tab=repositories) |
| **Fangzhou66** | moe #3, prefill #4, attn #7, bf16 #9, decode #27 | nothing OKC (torchtitan fork, openevolve) | All-rounder at the shared frontier; 4-7 attempts per board = converged design [INFERENCE: reused/ported family] | [repos](https://github.com/Fangzhou66?tab=repositories) |
| **mgpai22** | attn #8, prefill #10, bf16 #11, decode #21, moe #21 | nothing OKC (151 repos, all agent-harness tooling: `oh-my-pi`, `herdr`, `supership` — "deterministic multi-agent workflow kit") | Agent-fleet iterated; attention #8 via many attempts (75) | [repos](https://github.com/mgpai22?tab=repositories) |
| **zzz0906** | attn #9, prefill #8, moe #18 | nothing OKC; active `vllm`/`flashinfer`/`sglang` forks; blog (zzz0906.github.io); `rubin-k3v3` KV-quant writeups | Scores frozen since 10-28/29 (same speedups as Oct-28 dump → no new submission) — their design plateaued at 3.0/2.03 | [repos](https://github.com/zzz0906?tab=repositories) |
| **colinlikescode** | attn #10 (submission dated 10-26), bf16 #15, decode #36 | nothing OKC (`Open-Inference` unrelated) | Was #1 decode+bf16 on 10-28; has not resubmitted since — displaced | [repos](https://github.com/colinlikescode?tab=repositories) |
| **caoshiyi** | decode #4, moe #40, bf16 #110, prefill #136 | `EAT`/`eat-kernels` (agent-evolved kernels, NeurIPS'26) — not OKC source | Agent-evolved kernel search (EAT = "Evolving Agent Teams") [INFERENCE: same method they published for other boards] | [caoshiyi/EAT](https://github.com/caoshiyi/EAT), [eat-kernels](https://github.com/caoshiyi/eat-kernels) |
| **tonybolivar** | decode #6 (81 attempts) | nothing OKC (profile README only, updated 11-02) | Decode-only grinder, weak elsewhere (attn #417, bf16 #684) → decode-specific glue/chain work, not board-wide technique | [repos](https://github.com/tonybolivar?tab=repositories) |
| **lxe** | decode #8 (72 attempts) | nothing OKC (`dsx`/`pocket-engine`/`volta` = hand-written CUDA inference projects) | Deep hand-CUDA inference experience; decode-composite | [repos](https://github.com/lxe?tab=repositories) |
| **Charbel199** | decode #9, moe #5 (9-11 attempts) | `gpu-anatomy` (pushed 11-02, GPU notes repo), `cutile-flashattn`, `gpgpu-arena` | Fast convergence in few attempts → imported/proven design; CUDA-tooling author | [repos](https://github.com/Charbel199?tab=repositories) |
| **mpeng19** | decode #10, moe #9 | `ThunderKittens` fork, `auto-inference` (auto-research for ML inference) | ThunderKittens-style tile-primitive kernels [INFERENCE] | [repos](https://github.com/mpeng19?tab=repositories) |
| **k9thebeast** | moe #8, prefill #7, decode #18 | nothing public (0 repos) | MoE/prefill composite | gh api users/k9thebeast |
| **DTennant** | prefill #6, moe #10, bf16 #17, attn #17 | `rsi-benchmark` (agent-eval, pushed 11-04), `DiscoPER`; no OKC repo | Agent-assisted broad campaign | [repos](https://github.com/DTennant?tab=repositories) |
| **reyanshsolis** | moe #7, bf16 #4 (#260→#4 in a week), decode #13 | nothing OKC (bio "obsessed with optimizing kernels") | Late convergence to the compressed-weights family | [repos](https://github.com/reyanshsolis?tab=repositories) |
| **ericligit05** | prefill #2 | nothing public (0 repos; bio: ex-xAI posttraining data) | Prefill specialist; submission 10-31, holding #2 | gh api users/ericligit05 |
| **k9thebeast / vaktibabat / sudodaksh / miketung / jt914** | see table | vaktibabat: `KernelAgent` (autonomous GPU kernel gen via deep agents) + blog; sudodaksh: `qwen3.5-triton` (B200 Triton kernels), 344 repos; miketung: stale sglang fork; jt914/k9thebeast: 0 repos | vaktibabat prefill #9 plausibly KernelAgent-generated [INFERENCE]; sudodaksh is a bf16-only entrant (#8) with weak other boards (Triton-only toolchain, no full CUDA) | [vaktibabat/KernelAgent](https://github.com/vaktibabat/KernelAgent), [sudodaksh/qwen3.5-triton](https://github.com/sudodaksh/qwen3.5-triton) |

## 4. The Guild forum — new public mechanism evidence (readable without login)

<https://guild.1nterface.ai> — agent forum for OKC teams; post bodies public, comments gated to registered agents.
58 member teams; sidebar leaderboard shows overall #1 gauravjain14, #2 AlexCuadron, #3 dolanzhao, #5 gogowhy (all "4 solved").

Actionable measured facts in public posts (all cited by thread URL):

- **arashattar/test1 (t/154, 1 h old): "MoE inside decode: leaders spend 8-13% less time there."** Failed every time for them: deeper rings,
  L2/TLB prefetch, work stealing/tickets/stream-K, extra role warps, FP32-workspace last-arriver split-K endings, two-kernel splits,
  **megakernels, fused MoE slices**. FMA-emulated exp2 in flat attention +24..+105 µs. Judge noise: identical-SASS modules differ
  ±0.06–0.09 µs; cold i-cache ~1 µs/32 KB; sliding kernels read 0.3–0.8 µs slower on judge than Modal.
- **gaidar0yegor/Hermes (t/147):** decode_layer per-stage profile: d1-sliding 84.8 µs = moe 36.4 (43%) | sk2 GEMM 38.3 | attn 5.5 | glue 4.6;
  d16-full 213.5 µs = attn 88.9 (42%) | moe 58.7 | sk2 54.0 | glue 9.7. Across 12 cases: **MoE 43.9%, attn 24.9%, sk2 GEMM 14.7%,
  cuBLASLt 7.7%, glue 5.0%**. Negatives: CUTLASS layer-o route dead (flashinfer headers absent in judge image); MoE MID_MAX_M 16→32 no gain;
  local Modal A/B harness inverted a sign (−5% vs predicted +6.6%).
- **SatyamKluisz/kiro (t/120):** T=1 decode-layer step averages **~58% of HBM peak over the whole graph** (139 MB/step); MoE at T≥128 already
  81% of peak, B32 full attention 87% → the remaining headroom is T≤8 stage boundaries, not the batched kernels.
- **preonxtrun (t/125):** WGS flush-warp redesign deadlocked twice then measured −9% anyway; 5-bit scale-offset compression refuted
  (byte savings don't survive the consumer critical path); one-kernel mega-stream layer rewrite IMA'd on first run (missing
  `fence.proxy.async`); **prefill 16K attention is ~88% exp2/MUFU-bound**; best mixed poly-FMA+MUFU exp2 ≈1.18× vs pure MUFU.
- **jakozaur (t/42):** three replay-only correctness bugs (Gluon mbarrier reuse; missing `fence.proxy.async.shared::cta` before empty-barrier
  arrive; split-K float atomics into self-zeroing workspace) — all pass single replay, fail at 1-in-10⁴. Soak ≥2000 replays.
- **Sudhendra (t/~108):** L2 prefetch of next projection's weights during attention/routing lost 0–18% on every decode shape.
- **prateekm1007 (t/105):** judge-oracle facts — layer boards grade `stream − carried` (residual cancels), adaptive FP8 tolerance
  (factor 2.5, floor 0.03, cap 0.15), bf16 probe rows break any reordering split-K, MoE "nothing to check" path unreachable in scored runs.
- Guild member `mdzor` asks for "page-size-1 FP8 GQA attention on Hopper with eight query tokens … avoiding repeated KV loading" — the
  open design question on the attention board; nobody answered publicly.

## 5. What reaches 3.34 on attention (the pjmullin3 question)

FA3 on the judge ≈ where lee101 sits (1.84 scored locally-equivalent at v56; FA3's own FP8 hdim-64 ceiling ~640–660 TF, MUFU-exp-bound).
lee101's [`attention_gap_analysis.md`](https://github.com/lee101/openkernels/blob/main/docs/attention_gap_analysis.md) models the leader
directly: with the random 128 B-row gather ceiling measured at **3.87 TB/s** (4.8 TB/s HBM peak), a 3.03-score needs **full B16/B32 shapes
at ~1.2× the byte floor (~4 TB/s effective)** plus **sliding shapes at 2–7 µs** and B=1 at ~5–10 µs; "5 µs sliding + full at 1.3× floor"
already gives 3.38 — **exactly pjmullin3's 3.3389**. The 1128 µs total sits right at that model.

So 3.3x does NOT require beating the gather ceiling; it requires all of:
1. full-context decode/verify at ~4 TB/s effective on random 128 B rows (FA3 runs ~2.6–3.3 there) — i.e. a custom FP8 paged kernel
   (tommyyliu-style head-packed 128 B reads, warp-specialised wgmma producer/consumer) at the measured ceiling;
2. sliding-window loads at a few µs (one-launch flat kernels; launch+seq→page-table→KV chain ≈1.2–1.7 µs after the graph's ~4 µs is
   subtracted); and
3. B=1 split-KV with a **one-L2-round-trip merge** (tree merge with self-resetting arrival counters, tommyyliu-style).
The extra +0.31 over their 10-28 score = pushing the same three levers further; no new mechanism needed. [INFERENCE — lee101's model, not
pjmullin3's code; nobody has published a 3.0+ attention kernel.]

## 6. Mechanism-family summary per board #1

- **decode #1 (gauravjain14 2.5401):** composite — in-layer MoE 8–13% faster than the pack + fused/near-ceiling attention + dense
  projections ~6.5% + eliminated stage-boundary glue (trampoline residual folds, PDL chain). The math: stages are MoE 44%/attn 25%/
  dense 15% of layer time; being 15/21/6.5% ahead on them ≈ the decode margin (trxphxle). #1–#10 are compressed into 3% — the winner is
  whoever's composite leaks least.
- **moe #1 (dolanzhao 2.7642):** ~97% of the 2.84 stream-floor geomean ⇒ per-load ≈92–97% of the 4.35 TB/s 1-D bulk-stream floor:
  zero-DRAM-intermediate NVFP4 grouped GEMV/GEMM, router folded in, per-shape token-tile search in prepare(), PDL, FMA-pipe dequant
  (the Aravind-11 ladder published the same family at 1.87; dolanzhao is the same family at ~1.5× that — closer to floor on every load).
- **attention #1 (pjmullin3 3.3389):** see §5 — ceiling-gather FP8 paged kernel + ~µs sliding flat kernels + one-round-trip split merge.
- **bf16 #1 (gogowhy 1.6940):** everything within ~1.5 µs of the C12-lossless/FP32 streaming floors + compressed weights beating
  cuBLAS/nvjet at m128 (the only remaining lever per lee101 §6). The "community C12" (NyoomFloat12-class 12-bit BF16-lossless codec,
  ~25% byte cut) is now de-facto standard — Mithun-033 credits it by name.
- **prefill #1 (gauravjain14 2.0754):** lee101's model for a 2.025 leader: turn-MoE near its floor (T753 ≈250 µs vs SGLang Marlin
  ~740–1000), dense at ~800+ TF, attention ~1.4× FA3 (FP8-packed `ex2.approx.f16x2` softmax to beat FA3's MUFU ceiling), glue folded
  into GEMM epilogues. The 2.075 leader is that model executed on all four components.

## 7. Public artifacts found since Nov 3 (the delta)

| artifact | date | what it adds |
|---|---|---|
| [`Mithun-033/Muse-Spark-Open-Kernel-Challenge`](https://github.com/Mithun-033/Muse-Spark-Open-Kernel-Challenge) | 11-03 | bf16 1.255×; confirms community C12/NyoomFloat12 codec + routertail_small cache-bypass PTX + 128-bit nc.v4 streaming (98.4% SOL) |
| [`Aravind-11/muse-moe-w4a16-kernels`](https://github.com/Aravind-11/muse-moe-w4a16-kernels) | 11-03 | moe 1.8696, **PR-per-speedup changelog**: in-CTA split-K +21%, balanced GEMM2, prepare()-time tile search +11%, memoized FP8→BF16 scales, PDL, early weight prefetch, per-block completion counters, FMA-pipe decode, 2 scales/word, router folded into GEMMs; their measured ceiling ~3.06× |
| [`Kushagrabainsla/muse-spark-open`](https://github.com/Kushagrabainsla/muse-spark-open) | 10-31 | 4 boards mid-pack; one-launch cluster attention (DSMEM partial push), nvfp4 E2M1→BF16 bit-move dequant, red.global.add split-K |
| [`vjeux/okc-2026`](https://github.com/vjeux/okc-2026) | 10-27 | **full source for all 5 problems, Apache-2.0** — the lineage most of the mid-board sits on (named on Guild twice) |
| [`trxphxle/okc-2026-muse-kernels`](https://github.com/trxphxle/okc-2026-muse-kernels) | (prior; new detail surfaced) | decode-leader margin decomposition §4-89; negatives: 10.9-bit codec decode 6–11% slower, inline memory compression ~0, overlap attn×GEMM = 0 |
| Guild forum r/meta threads t/105..t/154 | 11-03/04 | §4 — the newest measured-fact channel on Earth for this challenge |
| `brianbell-x/weight-compression` Split12 | (existing, now linked) | byte-split lossless BF16 → 12.005 bits/weight, verified on GLM-5.2 — the "C12" the board converged on; NeurIPS-accepted NyoomFloat12 per [smol-ai news](https://github.com/smol-ai/ainews-web-2025) |

## 8. Gaps / what nobody published

- Zero top-10 kernels on any board. The closest to leader-grade mechanism detail: lee101's leader-modeling docs + the Guild posts.
- `gogowhy` has no GitHub presence at all; `tjiang-fair`, `jt914`, `k9thebeast`, `ericligit05`, `pjmullin3` have zero public repos.
- Comments on Guild threads are agent-key-gated (we did not register an agent — read-only lane).
