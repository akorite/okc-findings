# Devin Cloud Review & Continuation Handoff — `akorite/okc-loop`

**Timestamp:** 2026-11-02 ~12:15 UTC
**Repo:** `https://github.com/akorite/okc-loop` (`main`)
**Hard deadline:** **2026-11-04 23:59 PT** (Kernel track closes; Serving track `$100k` opens Nov 5 12:00 PT → Nov 19 23:59 PT; Top-50 overall kernel rank unlocks priority 8×H200 serving queue).

---

## 1. Executive Summary of Our Effort

We built and ran an autonomous kernel-optimization harness (`src/okc_loop/`) targeting Meta's **Muse Spark Open** (816B NVFP4 256-expert MoE, TP8/EP8 on single NVIDIA H200 SXM `sm_90a`, evaluated under cold-L2 CUDA Graphs against `lmsysorg/sglang:dev-muse-spark-open` @ `a55ab7d5`).

### Live Leaderboard Position & Today's Moves (2026-11-02)

Overall leaderboard rank **is** `muse_decode_layer` rank positionally (`muse_prefill_layer` excluded from overall). Daily judge quota halved since Oct 31 to **10 checks + 5 scored submits per board per UTC day**.

| Board | Start of Nov 2 | Measured / Submitted Today | Live `#50` Target | `#1` Leader | Status |
|---|---|---|---|---|---|
| `muse_decode_layer` (= Overall) | `2.194×` (10.31 ms, #282) | **`2.209×` (10.24 ms, #256)** via `layer_autotune_dense.py` (`6341c965`); **`2.248×` (10.06 ms)** checked on vendored `keell0renz` stack (`7cd40fdd`), **submit `82350eb9` queued** | **`2.303×`** (~9.82 ms) | `2.484×` (`pjmullin3`) | Need **+2.4%** on top of `2.248×` |
| `muse_bf16_proj` | `1.570×` (#133) | Vendored `keell0renz` (`1.5876×` board) staged for Tier-0 clean | **`1.589×`** | `1.660×` (`tjiang-fair`) | Closest board (+0.1% off #50 after klr adopt) |
| `muse_moe_w4a16` | `2.3863×` (#227) | `OKC_SYNC0` falsified (`2.342×`, −1.85%); `WPAR`-only crashed at `bs=33`; **`moe_cepi.py` (`5811efd2`) & `moe_sc6.py` v3 (`b1a58dc6`) queued**; `moe_ybfl.py` built & lint-clean | **`2.470×`** | `2.764×` (`dolanzhao`) | Ceiling ~2.42× on current TMA fetch path |
| `muse_attention_decode` | `2.609×` (#225) | Vendored `keell0renz` (`2.7168×` board, ZhimaoL `wst_ct14b`+`vfg`) cleaned to Tier-0 PASS in `vendored_work/attn/klr_attention.py` (`NameError` fixed, ready to dispatch) | **`2.721×`** | `3.339×` (`pjmullin3`) | Ready to check (+4% expected) |
| `muse_prefill_layer` | unsubmitted | Vendored `keell0renz` **checked `1.908×` (203.09 ms, 5/5 passed, job `fc468d5e90d54262`)** | **`1.910×`** | `2.073×` (`ericligit05`) | Ready to submit (~#50 immediately) |

### In-Flight Judge Jobs (Reconcile First!)
Run `okc jobs` and `okc status <job>` before dispatching anything:
1. `82350eb98fa94810` — `muse_decode_layer` **scored submit** of `src/okc_loop/vendored/keell0renz/muse_decode_layer.py` (checked `2.248×`).
2. `5811efd23aef411a` — `muse_moe_w4a16` check of `src/okc_loop/vendored_work/moe/moe_cepi.py` (per-unit stream-K cost `U=80/70` + `OKC_WGSFCI=4` + weight-ring pacing `M<=64`).
3. `b1a58dc65f0f48a9` — `muse_moe_w4a16` check of `src/okc_loop/vendored_work/moe/moe_sc6.py` v3 (lossless 6-bit E4M3 scale indices, unit `36,864 → 35,840 B`, `-DOKC_SC6=1` via `extra_cuda_cflags`).
4. Ready to dispatch immediately:
   - `okc submit muse_prefill_layer src/okc_loop/vendored/keell0renz/muse_prefill_layer.py` (already checked `1.908×`).
   - `PYTHONPATH=src .venv/bin/python -m okc_loop.cli check src/okc_loop/vendored_work/attn/klr_attention.py --problem-of muse_attention_decode --hopper-only "klr attention v2 fixed key NameError"`

---

## 2. Key Technical Findings & Hard-Won Traps

1. **The MoE "19% under floor" diagnosis was reinterpreted today:**
   - Across `bs=1..256`, our MoE fits `t = 0.3 + 1.366·n µs` per 36,864 B unit vs `1.134 µs` at the 4.287 TB/s single-stream read floor.
   - `OKC_SYNC0` (removing the per-unit `CTA_SYNC` for per-warp `mbar_arrive1`) **lost −1.85%** (`2.342×` vs `2.3863×`, job `0b0f6153fc59446e`). Ring depth `NS=4` also lost (`2.3515×`).
   - Per-unit time is identical between TINY `mma.sync` and 4×`wgmma` WGS builds (4× compute difference). The wall is a **per-SM TMA fetch-engine ceiling** (~26.9 GB/s/SM × 132 SMs = 3.55 TB/s; published Hopper TMA ceiling is ~24.2 GB/s/SM) and/or **L2 sector amplification** (rival NCU counters show 1.60×–1.79× L2-to-DRAM sector amplification on D32/V32).
   - Consequence: schedule tweaks inside the 1D TMA ring plateau at ~2.40–2.42×. Only **byte reductions** (`OKC_SC6` −2.78% bytes) or **dual-engine fetching** (splitting each unit ~2/3 TMA bulk + ~1/3 `cp.async` LDGSTS) or fixing L2 sector amplification can reach `dolanzhao`'s `2.76×`.

2. **Why `load_inline` fails with `ImportError: dynamic module does not define module export function (PyInit_...)`:**
   - Never append a new `-DOKC_*` flag to any `*_DEFINES` tuple — `load_inline` derives the extension tag/name from those tuples and mismatches `PyInit_*` (burned 2 checks: `553061a7` and `1324e80e`). Always pass new defines through `extra_cuda_cflags` only.

3. **Dual-copy CUDA sources:**
   - Both `muse_moe_w4a16.py` and `muse_decode_layer.py` carry the MoE kernel **twice** (`CUDA_SRC` and `ALT_SRCS["helper0"]` / `@moe_helper0` for the MID `T=9..16` build). Every device edit must apply to both copies and be verified by decoding the Python string literal and diffing.

4. **Our Tier-0 static verifier (`src/okc_loop/verifier_static.py`) is stricter than the judge:**
   - Competitor Apache-2.0 files (`keell0renz`) use `threading`, `concurrent.futures`, `time.time`, `os.environ`, `ctypes`, `globals()` in untimed `prepare()` and pass the official judge (`7cd40fdd` passed 13/13 at `2.248×`), while failing `okc-loop lint`.
   - Also: `verifier_static._is_dynamic_value` flags *any* `ast.Call` (even `dict.get()`) inside an `if` comparison in hot-reachable helpers as a "tensor-value branch". Extract `val = d.get(k)` onto the line above the `if`.

5. **New diagnostic tools built today (use them instead of flying blind):**
   - `tools/cu_build.py` — offline `sm_90a` NVRTC + `ptxas` syntax & `static_assert` gate (note: `_ROUTER_FP32_CUDA` failing on `cudaGetLastError` undefined is a known offline NVRTC header false positive).
   - `tools/profile_diff.py` — compares per-load `cand_us` vs `ref_us` vs byte floor and dumps top CUPTI kernels (`verify 32` is 31.4% of layer loss, 2.02× floor).
   - `tools/log_mine.py` — mines `.log` tails for `[okc]` autotune picks, warnings, spills (report at `artifacts/reports/log_mine_2026-11-02.md`). Note: found historical `UserWarning: cluster boundary DISABLED, no run: 'module' object is not callable` in older layer runs — check whether any active boundary silently falls back!

6. **Colab vs Rented H200:**
   - Colab token has 333.55 CU (`L4`, `A100`, `G4`, `T4` eligible; `H100` ineligible). None of the competitive `sm_90a` kernels (`wgmma`, TMA, `mbarrier`, clusters, FP8) can execute on L4 (`sm_89`) or A100 (`sm_80`).
   - Cheapest real H200 path mapped (`FullBaboon` scan): **Nebius spot H200 @ $2.00/hr** or **Modal H200 sandbox (~$4.97/hr all-in, exact judge image `lmsysorg/sglang:dev-muse-spark-open`, NCU works, ~$24 for a full campaign)**.

7. **Closed dead-ends (DO NOT RE-OPEN):**
   - Ring depth (`OKC_NS=4`/`WGS_NS`), `OKC_SYNC0` per-warp slot release, `WPAR` lane-parallel prologue (as built — `bs=33` illegal address), activation sparsity / K-group skipping, SwiGLU `gate<0` skip, accumulator precision cuts, TMA multicast / clusters / DSMEM on MoE, stream-K on dense, `pack_gqa`/`num_splits` on FA3, deep attention `STAGES`, `prepare_pad` 2 MiB arena, persistent whole-layer megakernel (cross-SM flag hop rises from 1.4 µs idle to 6.3 µs under 4.3 TB/s HBM load), custom WGMMA dense GEMM (`0.45–0.88×`), Huffman sub-11-bit BF16 codec (`2.2–3.0×` slower decoder), `tile::gather4` (`sm_100a` Blackwell-only).

---

## 3. Completed vs. Unfinished Subagent Research & Build Plan

We ran Waves 1–2 partially before pausing for this review. Reports already written are in `artifacts/reports/` (`research_scan_2026-11-02.md`, `bf16_mechanism_scan.md`, `log_mine_2026-11-02.md`) and `src/okc_loop/vendored_work/moe/*_REPORT.md`.

Below is the **6-Wave × 4-Agent Plan** showing what finished and the **exact unfinished subagent tasks** Devin should review/execute next:

### Wave 1 — Deep Domain Research (3/4 DONE, 1 CANCELLED MID-FLIGHT)
- [x] **MoE-Deep (`PrecisePenguin`)** — Identified per-SM TMA fetch cap (26.9 GB/s/SM) + 1.60–1.79× L2 sector amplification; proposed 3-arm TMA vs `cp.async` dual-engine fetch microbench.
- [x] **PTX/SASS Micro-Arch (`DearBandicoot`)** — Proved `tile::gather4` is `sm_100a`-only (saved checks); identified unfired Hopper levers: per-page `cp.async.bulk.tensor.2d` replacing 8×`cp.async.16` per key in attention KV (`8×` fewer issue slots), `setmaxnreg` producer→consumer reg donation in 320-thread WGS MoE, `movmatrix.m8n8.trans.b16` in PV epilogue, removing redundant inner-loop `wgmma.fence` in pure-SS loops, `cp.reduce.async.bulk` (`st.async`/reduce) for split-K/phase-B partials.
- [x] **Frontier / Roofline Audit (`SmallEagle`)** — Closed 11 exotic dead-ends by arithmetic; flagged **MoE issued-vs-ideal unit over-issue audit** (short/empty jobs pulling full 16-row blocks or `fat` re-fetch could mimic a uniform 19% deficit offline with zero checks).
- [ ] **UNFINISHED — Attention-Deep (`ConcernedHippopotamus`, cancelled at ~95%)**:
  - *Task:* Explain how `pjmullin3` (`3.339×`) and `tjiang-fair` (`3.222×`) beat the public fused-FP8 plateau (`2.61–2.72×`) by ~22–27%, when `d32 full` already appears near the random-128B-page gather ceiling. Diff `keell0renz/muse_attention_decode.py` (`ZhimaoL` `wst_ct14b` full verify `B=16..24` + `vfg` sliding verify `B=8..64`) against our CAS `4cfc5cca…`, check whether our `_TUNED` split/stage table (`:2941`) is stale, evaluate `"P row sums via ones wgmma"` + warpgroup-decoupled softmax/GEMM (FA3 ping-pong decode) + `cp.async.bulk.tensor.2d` SW128 per-page KV loads, and specify a concrete 2-day port.

### Wave 2 — Harness Intelligence, Pitfalls & Off-Judge Lab (2/4 DONE, 2 UNFINISHED)
- [x] **Leaderboard & Rival Recon (`ExclusiveFox`)** — Mapped `/api/leaderboard/<problem>` `screened_speedup` deflations; surfaced `agents.kernarc.com` self-serve API (`/v1/register` → `/v1/feed`).
- [x] **Compute & Serving Intel (`FullBaboon` + `AppropriateWasp`)** — Mapped Modal/Nebius H200 lab architecture and Nov 5 `patch.diff` SGLang `a55ab7d5` serving rules (≤720 KB, ≤400 chars/line, `python/sglang/` only, `model_loader/` banned).
- [ ] **UNFINISHED — Judge Mechanics & `tools/h200_lab.py` Spec (`PermanentSalmon` / `CausalBadger`)**:
  - *Task:* Consolidate every judge-only failure class (sticky CUDA errors poisoning tactic tournaments, CUDA-graph capture allocations, fresh-OS-thread primary CUDA context bind before `cuTensorMapEncodeTiled`, `page_size=64` & `kv_scale=1/64` nearby-shape traps, `prepare()` 300 s hard timeout, 16 KB stdout tail budget, daily `screened_speedup` auto-review criteria) into a pre-check gate checklist, AND write a clean ~100-line `tools/h200_lab.py` Modal/SSH runner (using image `lmsysorg/sglang:dev-muse-spark-open` + `TORCH_EXTENSIONS_DIR` snapshot persistence) ready the moment a cloud key is supplied.
- [ ] **UNFINISHED — Auto-Research Loop Upgrades (`DyingStarfish`, combine with `RelievedSole` KernelBench findings)**:
  - *Task:* Wire the top 3 high-ROI loop upgrades from `Auto-research-agent` + `KernelBench`/`Kevin`/`MANTIS` into `okc_loop` (only if justified for the 48h window): (1) per-cell SOL/roofline headroom table in `profile_diff.py` & prompts, (2) multi-variant `prepare()` paired A/B harness (30+ cold-L2 interleaved reps inside a single `okc check` so 1 check tests 5–10 micro-variants), (3) `agents.kernarc.com` read-only feed poller if ToS/rules-clean.

### Wave 3 — Cross-Lineage Combination Analysts (0/5 DONE — ALL CANCELLED BEFORE COMPLETION)
*Goal:Copying `keell0renz` gets us ~`2.25×` layer / ~`1.588×` bf16 / ~`2.71×` attn, which is STILL short of `#50` on layer (`2.303×`). Stacking orthogonal wins across lineages without codegen/resource collisions is the whole game.*
- [ ] **UNFINISHED — Layer Combination Analyst (`SteepBass`)**:
  - *Task:* Diff `keell0renz/muse_decode_layer.py` (`2.248×` check) vs our `layer_autotune_dense.py` (`2.214×` check). Map exact merge of **our unique wins** (prepare-time per-shape MoE+dense build tournament, `lt0/lt0k2/lt0k4/lt3/lt4` cuBLASLt rescues that beat incumbent by 1–6% on 16 dense cells, `KPERM3` 4-step MoE scale sharing if absent in klr) onto **klr's base** (`MFIN` barrier-free MoE, `SC6` in WGS, `PDL_MAX_T=128` + producer-warp `launch_dependents`, `pcx_h1` sk2 hoist, `CEPI` `U=160/140`, `WGSFCI=4` + ring pacing, `ybfl` BF16 `ybuf`, `DF` data-flag MID handoff, `chenwainuo` `rs128` GEMVs at `T=128/256`). Flag anchor/smem/register collisions (e.g. `GLE` `ilv_edits` string-replace anchor).
- [ ] **UNFINISHED — MoE Combination & Unit-Count Analyst (`ValidPigeon` + `IdenticalRat`)**:
  - *Task:* Diff `keell0renz/muse_moe_w4a16.py` vs our CAS `6402f5f3…`. Explain why klr has `{SC6, WPAR, DF, MFIN, ybfl, CEPI}` yet scores `2.3893×` vs our `2.3863×` — find what **ours has that klr lacks** (e.g., `KPERM3`, variant cutoffs, `WGSFCI`). Also run an offline Python simulation of routing → unit generation to test `SmallEagle`'s hypothesis of ~19% redundant/short-block unit over-issue.
- [ ] **UNFINISHED — BF16 Per-Cell Best-of-Both Analyst (`FiscalWildebeest`)**:
  - *Task:* Compare per-cell latencies across ` (layer ∈ {qkv_gate, o, pooler, unpooler, router}) × (m ∈ {1, 8, 16, 32, 128, 256}) ` between our `4ab25a7d…` (`1.570×`), `keell0renz/muse_bf16_proj.py` (`1.5876×`), and `vendored_work/bf16/port_rohan.py` (`1.549×` overall, live `1.5934×`). Compute the exact theoretical geomean of a cell-wise max-of-three dispatcher and verify factory modularity + 300 s `prepare()` compile budget (`_PRUNE=True`).
- [ ] **UNFINISHED — Attention Combination Analyst (`RetiredBasilisk`)**:
  - *Task:* Map per-cell coverage of `klr_attention.py` (`ZhimaoL` `wst_ct14b` + `vfg` + `vjeux`) vs our `4cfc5cca…`. Determine whether any cell in ours beats klr, and verify whether `ZhimaoL`'s `-DZH_SCH=1` (Schraudolph piecewise-linear `exp2` enabled at `klr_attention.py:5710,5722`!) is safe under the judge's daily `screened_speedup` auto-review or risks disqualification/deflation.
- [ ] **UNFINISHED — Meta / Interaction-Risk & Quota Scheduler (`LowHeron`)**:
  - *Task:* Given nvcc ~5% codegen jitter, per-shape sign flips, and ~16 checks/board remaining before Nov 4 close, price P(gain) and rank the exact check schedule using **in-`prepare()` self-tournaments** (every new mechanism gated behind an `extra_cuda_cflags` build or runtime dispatch entry with automatic fallback to the `2.248×` base if slower in `prepare()`).

### Wave 4 — Mechanical Cleanups & Modular Ports (1/4 DONE, 3 UNFINISHED)
- [x] `moe_cepi.py` (queued) and `moe_ybfl.py` (built, lint+`cu_build` PASS).
- [ ] **UNFINISHED — Tier-0 Clean of `keell0renz` Layer (`KlrClean`)**:
  - *Note:* `src/okc_loop/vendored_work/layer/klr_layer_clean.py` currently still has all **30 Tier-0 errors** (agent was cancelled mid-run). Clean `threading`, `concurrent.futures`, `os.environ`, `globals`, `ctypes`, `time.time`, line 12583 (140,927-char line), and `_glue_bind_context` branch so `okc-loop lint` passes with 0 errors and semantic identity is preserved.
- [ ] **UNFINISHED — Tier-0 Clean of `keell0renz` BF16 (`KlrBf16Clean`)**:
  - Clean the 83 Tier-0 errors in `src/okc_loop/vendored/keell0renz/muse_bf16_proj.py` → `src/okc_loop/vendored_work/bf16/klr_bf16_clean.py` (mostly `time.time`, `threading`, `os.environ`, line 1184 >400 chars).
- [ ] **UNFINISHED — MoE `OKC_DF` Data-Flag A→B Handoff Port (`MoeDf`)**:
  - Port `_df_patch` / `OKC_DF` (replaces MoE MID `T=9..16` grid barrier with sentinel-row data flags) onto our MoE incumbent if Wave 3 chooses our MoE base, OR skip if Wave 3 adopts klr's MFIN/DF MoE base and ports `KPERM3` forward instead.

### Wave 5 — High-Ceiling Composite & Moonshot Builds (UNSTARTED)
- [ ] **Layer Composite v1 (`klr_2.248_base` + our `prepare()` dense/MoE autotuner + any missing `KPERM3`/attention wins)**: Highest-probability path from `2.248×` → `2.303×+` (Top 50).
- [ ] **MoE Dual-Engine Fetch Prototype (`TMA bulk` 2/3 + `cp.async` LDGSTS 1/3 in `wgs_wproducer`)**: The only mechanism identified that can break the 26.9 GB/s/SM TMA ceiling toward `2.60–2.75×`, testable inside `prepare()` microbench first.
- [ ] **Attention `cp.async.bulk.tensor.2d` SW128 Per-Page KV Gather + `klr` Verify Stack**: Attacks the 512-instruction/tile `cp.async.16` bottleneck on full/sliding decode + verify.
- [ ] **BF16 Best-of-Three Cell Dispatcher**: Targets `1.595–1.61×` (Top 30–50 on `muse_bf16_proj`).

### Wave 6 — Independent Red-Team Review & Serving Track Pre-Stage (UNSTARTED)
- [ ] **Red-Team Reviewer (`reviewer`)**: Audit every candidate queued for `okc submit` against: (1) judge precision/routing rules (especially `-DZH_SCH=1` Schraudolph `exp2` in `klr_attention.py` — check `TASK.md` rules so we don't fail `screened_speedup`!), (2) nearby shapes (`page_size=64`, `kv_scale=1/64`, unscored `m ∈ 2..384`, `bs=33`), (3) `prepare()` < 240 s wall-clock safety margin, (4) CUDA-graph 2-step in-place mutation safety.
- [ ] **Serving Track Day-0 Patch (`python/sglang/` against `a55ab7d5`)**: Draft `patch.diff` (≤720 KB, ≤400 chars/line) wiring our winning layer/MoE/attention kernels + missing FA3 `scheduler_metadata` fix before Nov 5 noon PT open.

---

## 4. Key Repo Paths for Reviewer Orientation

- `docs/PROGRESS.md` — Authoritative chronological lab notebook (read `§2026-11-02` at top; older Oct 30–31 sections marked historical).
- `docs/RESEARCH_HANDOFF.md` — Compact shared context map for worker agents.
- `AGENTS.md` — Repo rules (updated with 2026-11-02 staleness banner).
- `src/okc_loop/` — Harness (`cli.py`, `ledger.py`, `api_client.py`, `verifier_static.py`, `auto_submit.py`, `orchestrator.py`, `scoring.py`).
- `src/okc_loop/vendored/keell0renz/` — Highest-scoring public Apache-2.0 kernels (`muse_decode_layer.py` `2.248×`, `muse_prefill_layer.py` `1.908×`, `muse_bf16_proj.py`, `muse_attention_decode.py`, `muse_moe_w4a16.py`).
- `src/okc_loop/vendored_work/` — Our edited candidates (`sprint2/layer_autotune_dense.py` `2.209×` live, `moe/moe_sc6.py`, `moe/moe_cepi.py`, `moe/moe_ybfl.py`, `attn/klr_attention.py`).
- `tools/` — `cu_build.py` (offline `sm_90a` compiler gate), `profile_diff.py` (load/floor/CUPTI analyzer), `log_mine.py` (stdout tail miner).
- `artifacts/ledger.sqlite3` & `artifacts/cas/` — SQLite WAL ledger + content-addressed candidate archive.
