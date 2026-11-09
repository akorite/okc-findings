# Final-Day Autoresearch Loop (all five boards, close 2026-11-05 07:59 UTC)

> **For agentic workers:** this document is a **runbook**: execute §1 (bootstrap), then run the §2 LOOP until the terminal condition. Do not end your turn while any lane in §4–§6 has an actionable item. Research lanes may fan out; `okc` calls never do.

**Spec:** `docs/HANDOFF-2026-11-04-FINAL.md` (read once at bootstrap, then rely on this file).

**State at bootstrap (measured 05:45 UTC Nov 4):**
layer `0/20 runs, 20/40 checks` · moe/attn/bf16/prefill `20/20 runs, 40/40 checks` · 31 of our jobs queued, draining ~09:30 UTC · judge cadence ≈7.5 min/job serial · row 2.2837 (#109/1250).

**Live targets:**
| board | ours | #1 | #50 | gap to #50 |
|---|---|---|---|---|
| muse_decode_layer (= overall rank) | 2.2837 | 2.5401 | 2.3493 | +2.87% |
| muse_moe_w4a16 | 2.3806 | 2.7642 | 2.5036 | +5.2% |
| muse_attention_decode | 2.6590 | **3.3389** | 2.7838 | +4.7% |
| muse_bf16_proj | 1.5891 | 1.6940 | 1.5953 | +0.39% |
| muse_prefill_layer | 1.9087 | 2.0754 | 1.9338 | +1.3% |

#1 scores are proof of possible speed, not verdicts. Demotivation is a bug — see Standing Orders.

---

## 1. BOOTSTRAP (once, ~20 min)

1. Read `docs/HANDOFF-2026-11-04-FINAL.md` §2, §4, §6, §7. Read this file.
2. `okc problems`, `okc jobs --json`, `curl -s https://openkernelchallenge.dev/api/v1/queue`, `date -u` — snapshot into `artifacts/bench/final_day_ledger.md`.
3. Build `tools/dedup_guard.py` + `tests/test_dedup_guard.py`: `already_judged(path)->job_id|None`, sha256 of file bytes vs every prior job's `content_hash` (verify hash semantics on one known file first; fallback: compare file bytes against `payload_ref` payloads). CLI form exits 1 on collision. Run it over `vendored_work/*/**.py`; write the forbidden-file list into the ledger.
4. Build `tools/queue_watch.py` (~80 lines, runs detached all day): every 120 s — `okc jobs --json` + `okc problems` + `GET /queue` → append new landings (job, µs, status, created→started latency) to `final_day_ledger.md`, and recompute `CADENCE = median(created→started)`; also polls `okc leaderboard muse_decode_layer --json` every 15 min and logs rank. This is the loop's sense organ; the agent never blocks on `okc status` again.
5. `mkdir -p artifacts/reports artifacts/bench`; touch `final_day_ledger.md` with a header: lever | board | file | job | µs | verdict | banked.
6. Start the first research wave (Lane R, 3 concurrent max, cheap model).

## 2. THE LOOP — run until terminal condition

```
TERMINAL: now ≥ 2026-11-05 04:00 UTC for submits; ≥ 07:59 UTC for everything else.

loop forever:
  # OBSERVE (cheap, no okc calls — read queue_watch's ledger tail)
  state = { clock, cadence, quota per board, queue depth, pending landings,
            H200 state, candidate backlog, research backlog }

  # INGEST — every new landing since last tick
  for job in new_finished_jobs:
      ledger += (job, µs, status); verdict via tools/campaign_stats.py vs control
      if failed/rc112: record stderr tail → lever DEAD → spawn 2 new hypotheses (Standing Order 3)
      if run result: re-read board row + rank

  # DECIDE — pick the highest-priority ACTIONABLE item (priority order below)
  act on it  →  LOG it  →  next iteration immediately; NEVER wait on a result
  if nothing actionable: pick next research/offline item; STILL log each iteration
```

**Priority order (first actionable wins; gates in §3):**

| pri | action | gate |
|---|---|---|
| 1 | **kitchen_sink correctness check** (1 job, alone) | queue empty-ish; layer checks > 0 |
| 2 | ingest/verdict every landed job | always |
| 3 | layer check campaign (ks ×5, ks_noq8 ×4, ks_qp16only ×4, ctrl ×4, interleaved) | after P1; cadence < 15 min |
| 4 | lever builds from backlog (§4 K/S/M items), offline | 90-min box each |
| 5 | research lanes (§5 R1–R7), 3 concurrent max | always (subagents) |
| 6 | component-board checks (moe T1/midband/pf, attn xqa/bulk_red, bf16/prefill new levers) | layer campaign scheduled; dedup-clean |
| 7 | H200 batch if GPU frees | Lane R4 poll |
| 8 | **00:00 UTC: run waves** (§2.1) | layer 20/20 |
| 9 | M-lane moonshots (§6) when K/S backlog empty | offline |

**2.1 The 00:00 UTC run waves (replace loop during Nov 5):**
- `okc problems` → layer 20/20 or stop-and-report.
- Wave 1 (00:00): kitchen_sink winner → best second variant → dedup-clean distinct `klr_*` (`klr_c_st1f_4`, `klr_fci7`, `klr_glue`, `klr_cost11`, `klr_best`), 6 runs, 11 s apart. If first job's created→started > 60 min: submit all 20 distinct bytes NOW and revert to loop.
- Waves 2–3 (≈01:00, ≈02:30): 6 runs each of distinct-code recombinations of the levers that moved µs; hold 2 runs, last submit ≤ 04:00 UTC.
- Component runs only after layer waves, ≤4/board, dedup-clean, never delaying a layer job.
- 06:00 UTC: snapshot all boards into the ledger. No withdrawals.

## 3. GATES (the rules that were missing)

1. **No layer `okc submit` before 00:00 UTC Nov 5.** Other boards' runs are allowed only by Priority 6 rule.
2. **dedup_guard before every dispatch** — `duplicate` costs a serial slot and maybe a run.
3. **Distinct code for scored runs.** Banner clones on `check` only (it's off-board). Final rank = medians of organizer reruns on scored + nearby shapes.
4. **Means decide, never draws.** σ=1.17%; promote on ≥2 draws via `campaign_stats.py` vs interleaved control; prefer lower mean even inside noise. Max 2 checks/idea except the 2 finalists.
5. **One file = the stack.** Every non-negative lever becomes a `#define` in the kitchen_sink lineage; variants are flag flips. Interactions have been wrong before — re-check after every merge.
6. **Hand-write kernels.** Generator only for mechanical splices; no model-written kernels (40× slower, measured).
7. Never prompt a human; never weaken `assert_automated`. Never withdraw.

## 4. WORK QUEUE — lever backlog (each: file/status/delta vs board/gate)

**K — kernel work, hand-written**
- **K1 decode-1 MoE in-layer epilogue: 29.4→22 µs (+1.3–1.9% layer)** — THE open term. `t1_layer_build.py` queued as `c2ae9737` — ingest, then diff standalone bs=1 vs `TINY2` path.
- K2 aux-kernel fusion: `_graft_row_boundary` 8.8–13.9 µs @v16/v32, `_sl_*`/`_gle`/`_graft_route` 3.7–7.9 µs @small loads → +1.5% max. Needs per-SM timeline or careful blind fuse.
- K3 attention XQA prefetched-fold + `bulk_red_add_f32` merge — mechanism our layer file lacks; `attn_xqa.py` arm is default-off; unmeasured.
- K4 kitchen_sink q8-flag ablations (`ks_noq8`, `ks_qp16only`) — settles WGDEC/TMAPG8/QP16 interaction.
- K5 verify-1 (T=8) MoE occupancy, −8..−20 µs if K1 generalises. Depends: K1.

**S — small-wins ledger (write all, skip none; verify status before work)**
S1 `GEMV_M1` +0.39% deriv — in stack2 · S2 `MOEDUP` +0.5% bound; standalone rc112, layer run `2db47a2b` pending · S3 `SC6S` +0.68% deriv, sign unresolved — in stack2 · S4 `QP16 0` −0.54% 1-draw — in ks · S5 `WGDEC 0` −0.29% — in ks · S6 `FINWARM 1` +0.16% — in ks · S7 `TMAPG8 1` ~0 — in ks · S8 `XRES+MOEDUP` +1.53% vs quiet — submitted, mean unknown · S9 `EXH8` +0.10% expected, trades QDIRECT/PK2 · S10 `WGSFR1` unmeasured → 1 check · S11 bulk_red merge → K3 · S12 `klr_sk2` dense cells 2–6 µs — queued · S13 `STAGES 3→4` DEAD · S14 XQA fold → K3 · S15 merged-heads → M1 · S16 attention-#1 gap → R2 · S17 heaviest-expert-first +1.1–1.3% moe — "MoeSchedule in flight", find job · S18 `moe_midband` `eea609ed` — ingest · S19 `moe_pf` — ingest · S20 `WMAJ` tiny ring ≤8% moe — K1 fallback · S21 standalone-vs-layer bs=1 gap → K1 · S22 `bf16_selftune` — ingest · S23 bf16 #1 gap +6.6% → R2/R3 · S24 `ROUTER_MM_SMALL` banked +0.65% · S25 prefill = MoE 38–40% + dense 37–43% → I2 ports · S26 `t1_layer_build` — ingest · S27 `layer_build_tune` — ingest · S28 codegen lottery → M8.

## 5. RESEARCH LANES (parallel, 3 max, cheap model; each → `artifacts/reports/fd_<slug>.md` + ≤6 lines back; every claim: URL or path:line)

- **R1 GitHub round-2** (`gh` authed as `akorite`): re-run research_01 queries **recently-updated first**; re-check `pjmullin3` (attn #1, 0 repos Nov 3), `dolanzhao` (moe #1), `gauravjain14` (overall+prefill #1), `gogowhy` (bf16 #1), `colinlikescode`, `DTennant`, `zro-inference`, `tommyyliu`, `ruwwww`, `operatorstack`, `infraax` (untimed-`prepare()` sweep trick — legality check), `lee101/openkernels` (109 snapshots — unharvested mechanisms?), `ZhimaoL` forks/web-archive.
- **R2 reverse-the-leaders:** mechanism per #1–#10 on every board from public artifacts (repos, gists, sid19991 leaderboard JSONs, profiles). One line each.
- **R3 kernel-source sweep:** FlashInfer MonoMoE, DeepGEMM #473, TRT-LLM XQA, FlashMLA, LeanAttention, vLLM Marlin scheduler — one surviving mechanism per repo, ≤10 levers.
- **R4 H200 rental** (`SUPER_EXPERIMENTS_KEY` — secret): poll hourly; run `/tmp/h200/okc_batch_muse_decode_layer.py` if a slot opens; kill pendings at 20:00 UTC.
- **R5 weight-entropy probe** (feeds M2): generate oracle weights locally (`problems/muse_moe_w4a16/`); entropy of NVFP4 codes + scales + block deltas; one-line verdict ≥3%?
- **R6 compliance re-scan** of all queue-able files: encoded/obfuscated patterns; `ctypes` = system libs only → GO/NO-GO per file.
- **R7 nearby-shape audit:** grep finalists for scored-shape conditionals (`page_size`, `kv_scale`, `m == ` literals); flag any lever without a general fallback.

## 6. MOONSHOTS (90-min boxes; run when §4 is empty)

M1 merged-heads attention (+0.5–0.8% est) · M2 NVFP4 lossless recompression — bandwidth-bound term; R5 gives the ceiling today for free · M3 shared-exponent scales (SC6→~4.5 bits) · M4 router BF16 hi+lo at ALL sizes (explicitly legal; verify current coverage) · M5 P→FP8 + poly exp2 (audit current use; keep ≥BF16 precision — screening risk otherwise) · M6 TMA multicast PTX re-probe (closed axis says inexpressible; 30 min to verify, not trust) · M7 cluster-pair DSM KV share · M8 codegen lottery self-tournament in `prepare()` (nvcc jitter ~5%; cap builds at prepare budget) · M9 per-SM `%globaltimer` timelines — first thing on any real H200 · M10 tail-shrink (silent 6–9 µs tail; needs M9) · M11 wgmma A/B-swap for M=64 padding (verify DeepGEMM claim before building).

## 7. STANDING ORDERS & ANTI-STALL

1. Never idle: priority table always has an actionable row; if truly blocked, next offline item — and that gets logged too.
2. `queue_watch.py` is always running; the agent reads its ledger tail, never polls `okc status` itself.
3. A dead lever spawns 2 new hypotheses before filing.
4. 90-min box per hypothesis → lever file or a falsification number; both are wins.
5. Banned output: "unreachable/ceiling/hopeless" without measured arithmetic.
6. Ledger append every iteration; the ledger is the loop's memory — keep it current.
7. Small wins compound: +0.3% × 6 = rank move. Never skip for size.

## 8. DO-NOT-REOPEN (measured numbers exist; needs a NEW mechanism, not a retry)
Handoff §6 + test-plan §6 in full: megakernel, fused-slice MoE, L2 prefetch/persistence, 4-CTA K-splits, speculative prefetch, dual-lane wgmma, per-unit barriers, `WGSFCI`≠5, `WGSNX`, `WGSGS`3, `WGSDFIN` 0/3, `WGSU/SUB` ×1.1, cost tables, `SEGK`, `ST1F`/`ST8F`, `NP8`, `WGSREORD`, PDL-KV, dense-geometry @big loads, pooler-pack route-side, GLE@T8, per-T self-tune, TMA tile mode, graph PDL, STAGES 3→4, VFG sliding, `PVW`, `MERGE2/3 0`, `WS8 0`, `SYNC0`, sm_90a FP4 hw (none, ptxas-verified), CUTLASS Hopper NVFP4 dequant (we're ahead), banner-clone submits.

## Review Focus
- Duplicate bytes → dedup guard. · kitchen_sink never executed → Priority 1. · cadence collapse at 00:00 → re-measure + fallback submit-all. · serial-queue inference wrong → queue_watch flags it (a job starting while another of ours runs = parallel lanes; update priorities). · nearby-shape correctness unverified offline → R7.
