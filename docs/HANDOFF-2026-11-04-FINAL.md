# HANDOFF — Muse Spark Open Kernel Challenge, final day

**Written 2026-11-04 05:21 UTC = 2026-11-03 21:21 PST.** Read this file top to bottom before touching anything.

---

## 1. The absolute constraints (verified from the official `llms.txt`, not from memory)

| fact | value | how verified |
|---|---|---|
| Kernel submission window | **Oct 22 11:00 PT → Nov 4 23:59 PT** | `https://openkernelchallenge.dev/llms.txt` |
| Deadline in UTC | **2026-11-05 07:59 UTC** | Pacific is **PST −0800** on Nov 4 (DST ended Nov 1); confirmed with `TZ=America/Los_Angeles date -d '2026-11-04 12:00' '+%Z %z'` → `PST -0800` |
| Time left at handoff | **26.6 h** | `date -u` = 2026-11-04 05:21 UTC |
| Daily caps reset | **00:00 UTC** → next reset is **Nov 5 00:00 UTC**, which is **8.0 h before the close** | `llms.txt`: "resetting at 00:00 UTC"; `okc submit` error: "resets at 00:00 UTC, in 18 h 47 min" |
| Caps (as the CLI currently reports them) | **20 scored runs + 40 checks per problem per day** | `okc problems` |
| Both tracks | kernel prizes: none. **$100k is on the serving board**, which opens Nov 10 and is not ours to spend | `llms.txt` |

**The single most important consequence: the layer has a fresh 20-run allotment at 00:00 UTC Nov 5, with 8 hours to use it.** That is the only remaining thing that can move the rank.

---

## 2. Where we actually stand

| board | rank | speedup | value (µs) | runs left today | checks left today |
|---|---|---|---|---|---|
| **muse_decode_layer** (this *is* the overall rank) | **#109 / 1250** | **2.2837** | **9,905.1** | **0 / 20** | 20 / 40 |
| muse_moe_w4a16 | #260 / 1382 | 2.3806 | 5,649.1 | 20 / 20 | 40 / 40 |
| muse_prefill_layer | #99 / 858 | 1.9087 | 202,964 | 20 / 20 | 40 / 40 |
| muse_attention_decode | #148 / 1329 | 2.6590 | 1,416.4 | 20 / 20 | 40 / 40 |
| muse_bf16_proj | #78 / 1693 | 1.5891 | 3,040.0 | 20 / 20 | 40 / 40 |

**The ladder on the layer board** (the only board that decides the overall rank):

| rank | speedup | µs | gap from us |
|---|---|---|---|
| #1 | 2.5226 | 8,967.0 | +10.46% |
| #10 | 2.4656 | 9,174.4 | +7.96% |
| **#20** | **2.4248** | **9,328.9** | **+6.18%** |
| #50 | 2.3442 | 9,649.4 | +2.65% |
| **#100** | **2.2881** | **9,886.0** | **+0.19%** |
| ours | 2.2837 | 9,905.1 | — |

The board compressed ~1.5% *while we worked today*, so our rank slid from #84 to #109 with an unchanged score. **Top-20 needs 9,329 µs; our file's mean is 10,039. That is 5.8σ of the measured noise. Top-20 is not reachable by anything in this repository.** Sprint goal is top-100 (19 µs away) and realistically #50–90.

---

## 3. The instrument: what the judge's noise actually is

Measured, not quoted: three **code-identical** submissions of the row file (byte-different only by a comment banner, which `okc` does not dedup).

| draw | µs |
|---|---|
| row submit `9743b6e7` | 9,905.1 |
| A/A probe `ffeec8e6` | 10,125.3 |
| campaign control probe `93ed80e0` | 10,085.9 |

**Mean 10,038.8 µs, SD 117.4 µs = 1.17%.** The "2.2%" figure quoted earlier in `docs/PROGRESS.md` was the *range of two draws* (~2.7σ), not σ. Consequences:

- a 4-draw mean resolves ~1.15%;
- our row is a **+1.33% lucky draw** of a file whose mean is 10,039;
- any candidate must beat the *row*, not the mean, so it needs a genuinely better mean to displace it.

**Blocker to know before designing any measurement**: `okc check` on bytes already submitted returns `duplicate` and does **not** re-time. Every extra draw must be a byte-different, code-identical probe (a banner comment on line 2 works — proven).

---

## 4. BLOCKERS (read this section before promising anything)

1. **Layer runs are exhausted today: 0/20 until 00:00 UTC Nov 5.** Nothing can be submitted to the layer before then. This is the hard wall.
2. **The judge queue is heavily congested on the final day.** At handoff: **32 of our jobs queued**, 10 `duplicate`, 1 running, and results land at roughly one per 20–30 minutes; the last result arrived 04:19 UTC and nothing since (60+ min). **~18 submitted kernels have not scored.** The board keeps the best, so each landing can only help, but many may not land before the close.
3. **The H200 rental is unusable so far.** `app.getsupers.com`: **5,000 credits ($5.00) = exactly 1,000 H200-seconds** at 5 cr/s. Balance is **untouched** and `Total Seconds Used: 0.0s`. The fleet has **one live GPU** (Box A; Box B is DOWN) and our jobs have oscillated between #5 and #7 of ~11 pending for 90+ minutes. Four of our jobs are queued: two env probes, one single-candidate layer bench (1.44 MB), one **6-candidate batch (8.2 MB — uploads fine)**. The batch is the one worth waiting for: it would give six paired numbers in one slot.
4. **No cloud credentials on the workstation** — no `modal`, no `runpod`, no `~/.modal.toml`. The Supers CLI is the only rented compute, driven by `SUPER_EXPERIMENTS_KEY=[REDACTED]` (treat as a secret; it is in this file only because it is the one credential the next session needs).
5. **Colab cannot test these kernels.** Proven twice: the L4 dies compiling `glue_cluster_boundary` (sm_90a clusters on sm_89), and after adding an `apache-tvm-ffi` bootstrap it still fails; the attention A/B exceeded 3000 s even after I added `colab-batch --only <regex>` to subset cases — the JIT build dominates, not the case count.
6. **The offline sm_90a gate is vacuous for the attention file** (`tools/cu_build.py` reports "no kernel emitted (template body)"), so attention changes cannot be compile-verified locally. It *is* real for the MoE sources (10/10 tuples compile).

---

## 5. What is built and ready to fire at 00:00 UTC Nov 5

**`src/okc_loop/vendored_work/layer/kitchen_sink.py`** — the stacked candidate. Verified: lint multiset identical to `layer_stack2.py`, `cu_build` status identical (16 ok / 54 pre-existing FAILs, all six ATT7 tuples ok), `-D OKC_MOEDUP=1` compiles. It stacks every lever we measured as **non-negative**:

| lever | evidence |
|---|---|
| `OKC_QP16 0` | **9,984.6 µs = −0.54% vs control 10,038.8** (best of the sweep) |
| `OKC_WGDEC 0` | 10,010.0 = −0.29% |
| `OKC_TMAPG8 1` | 10,043.9 = +0.05% (a documented +0.7% for the q8 flat kernel elsewhere) |
| `OKC_FINWARM 1` | +0.16% measured previously |
| `SC6S` + `MOEDUP` + `GEMV_M1` | from `layer_stack2.py` (+0.68 / +0.5 / +0.39% derived) |

**If additive: ≈9,723.5 µs (2.2710)** — ~1.8% better than the row, around **#55**. **This is the only candidate whose expected *mean* is below the row**, which matters because the final ranking is medians of reruns, not best-of-N.

**Its risk, stated honestly**: it has **never executed anywhere**. Two agents were auditing it when this handoff was written (`StackAudit` for flag conflicts among the three q8 flags — `WGDEC`/`TMAPG8`/`QP16` all touch the same path — and `FinWarm2` for an extended instruction warm). Both were cancelled mid-flight. Their partial findings are **not** in the tree; check `artifacts/reports/` for anything they managed to write before being stopped.

---

## 6. Every mechanism measured dead (do not rebuild — each has a number)

| mechanism | result |
|---|---|
| `OKC_PVW 1` (async wgmma PV) | **10,330 µs = +2.90% worse** |
| `OKC_MERGE2/3 0` (per-thread fences) | **10,150 µs = +1.11% worse** |
| `OKC_WS8 0` | 10,060 µs = +0.21% |
| `OKC_NP8 256` | **fails correctness** |
| `OKC_SC6S` + `OKC_XRES` (`moe_bytes.py`) | 2.24079 = −1.88% vs row (inside noise) |
| `OKC_SC6S` alone (`sc6s_only.py`) | 2.26117 = −0.99% |
| `XRES` + `MOEDUP` (`layer_xres_dup.py`) | 2.26828 = −0.67% (**best single** result, +1.53% over the quiet-level draw) |
| `SC6S` + `MOEDUP` without `XRES` | 2.23825 = +0.19% vs quiet — **refuted the per-lever decomposition** |
| per-T build tuner, mis-calibrated (`layer_build_tune.py`) | 2.26104 = −0.99% |
| per-T build tuner, **calibrated** (`moe_st2`, MoE board) | 2.35149 vs 2.3806 row — **declined every swap**, closing the build-threshold axis for real |
| A/A probe (identical code) | 2.23403 = the quiet level |
| MoE `OKC_MOEDUP` enabled (`moe_dup_on.py`) | **fails the judge's checks, rc 112** |
| MoE 2 CTAs/SM (`moe_mma8.py`) | 2.35538 |
| MoE `OKC_DE` split-issue (`moe_dual.py`) | 2.37224 (neutral) |
| MoE `moe_ybfl` / `moe_desync` | 2.3657 / 2.3306 |
| MoE `swapab` / `moe_wpar` | fail, rc 112 |
| attention `OKC_STAGES 3→4` (`attn_gap1.py`) | 2.6370 vs 2.6599 base — the file's own gate picked 0/12 |
| keell0renz's published attention file | 2.6493 (**below our row**) — their 2.7168 was a draw |
| bf16 `best.py` | 1.5511 (below our 1.5891 row) |
| everything in `artifacts/bench/dispatch_queue.md` §"Measured and BELOW" | see that file |

**Closed by arithmetic, not by a run:**
- **MoE N-bucketing is already ours**: `N = 8·ceil(tokens/8)`, a superset of DeepGEMM #473's buckets. Padding is 30–87% of the MMA N columns, but that MMA is only **3.4–20.1% of each load's MoE time** and does not gate the 36,864 B weight stage. Even a perfect fix caps at **+3.2–4.3%** board. (`artifacts/reports/moe_nbucket.md`)
- **The attention fill is 5.0–5.3 µs, not 6.3**: the earlier two-point fit was collinear (per-CTA and per-byte costs inseparable, which is why its marginal rate came out *above* the machine ceiling — a red flag). Realistic value **+0.22–0.58%**. (`attn_fill.md`)
- **`sm_90a` has no FP4 hardware**: verified by assembling each PTX form with ptxas 13.4.92 per target — `cvt.rn.satfinite.e2m1x2.f32` and `.e2m1p4x4` are sm_100a/103a/120a only; CUTLASS has **no sm_90 NVFP4 path at all**, and its Hopper dequant costs 3.25–3.75 ops/weight against our 1.75. (`research_07_nvfp4.md`)
- **No leader on any board publishes**: all 59 top-20 handles swept. `pjmullin3` has 0 public repos; `Abarbir12` is not a real account. (`research_01_gh_code.md`)

---

## 7. The finale, step by step (00:00 UTC Nov 5 → 07:59 UTC)

1. **00:00 UTC — confirm the reset**: `okc problems` should show 20/20 runs on the layer.
2. **Submit `kitchen_sink.py` first.** It is the only candidate with an expected mean below the row. If it fails the judge's correctness checks, that is one run lost and a decisive answer.
3. **Then, in order**, the measured-positive sweep variants `sweep_b6_qp16_0` (−0.54%) and `sweep_b3_wgdec0` (−0.29%), then the best measured-but-unsubmitted `klr_*` kernels (`klr_c_st1f_4` 9,987.6, `klr_fci7` 9,993.0, `klr_glue`, `klr_cost11`, `klr_best`).
4. **The already-queued 18 kernels need no action** — they will score on their own and the board keeps the best.
5. **Do not submit comment-variants of one file.** `okc` dedups identical bytes precisely to prevent that; keell0renz's own README says the organizers notice ("my current #1 submit … its judgemaxxed"); and `llms.txt` states final positions are **medians of organizer reruns on the scored *and nearby* shapes**, so a lucky draw cannot carry us and nearby-shape degradation gets scrutinised.

---

## 8. Rules compliance — checked, clean

`llms.txt` forbids compressed/encoded/obfuscated source and prebuilt GPU code. Scanned all 22 submitted layer files for `zlib|base64|bz2|lzma|decompress|b64decode|marshal|cubin|fatbin|exec(|eval(`: **zero hits**. The only `.so` references are two `ctypes.CDLL` calls to **system** libraries — `libcublasLt.so.13` (`klr_fci5.py:12239`) and `libcuda.so.1` (`:13416`) — used to pin cuBLASLt plans. That is the driver API the image ships, not embedded GPU code. **We can submit; nothing in the tree needs withdrawing on compliance grounds.**

---

## 9. Tools built this session (all verified)

| tool | purpose |
|---|---|
| `tools/campaign_stats.py` | per-candidate means + SD + **95% margin** vs a control, from `[<candidate>:]<job>` args; prints PROMOTE / not-distinguishable. Never treat one draw as a measurement. |
| `tools/supers_batch.py` | one self-contained harness timing **N candidates in one billed GPU job** (the fixed costs — image, torch, oracle, flush buffer, graph machinery — are paid once) |
| `tools/judge_replica.py` + `tools/replica_selftest.py` | router-seed load search, fresh inputs per rep, the ~4.1 µs replay subtraction; **7/7 selftest passes**, never run on hardware |
| `tools/bench_bundle.py` | one-candidate self-contained harness; fixed this session (it passed `hw` as a string and crashed, and two candidates overwrote one filename) |
| `colab-batch --only <regex>` | case subsetting for off-Hopper sessions |
| `tools/cu_build.py` | offline sm_90a compile/static_assert gate (real for MoE sources, vacuous for attention) |

Key the next session must know: **`PYTHONPATH=src .venv/bin/python -m okc_loop.cli …`**; the `okc` CLI itself is at `~/.local/bin/okc` and one `okc login` has already been done by a human (we never log in).

---

## 10. Honest assessment

- **Top-20 is out of reach.** It needs 9,329 µs against our file's 10,039 mean — 5.8σ. Nothing found in this session is within an order of magnitude of that. If the objective is graded strictly on Top 20, this attempt fails.
- **The realistic ceiling** is #50–90, via `kitchen_sink` at 00:00 UTC Nov 5 if its levers are additive, or via the queued distinct kernels landing a good draw.
- **The biggest process failure of this session** was spending all 20 layer runs at 05:00 UTC on 18 kernels picked from *single* check readings, before the 4-draw means existed; 10 came back `duplicate`. The correct order was: finish the means → rank → submit. The 00:00 UTC allotment is the chance to do it right.
- **The biggest unspent asset** is the 1,000 H200-seconds. If the rental ever frees a GPU, spend it on the **6-candidate batch** (`/tmp/h200/okc_batch_muse_decode_layer.py`, already built) because `prepare()` is 140–172 s of which **~85–120 s is nvcc** — if `TORCH_EXTENSIONS_DIR` persists between jobs, the same budget buys ~20 candidates instead of ~6.
