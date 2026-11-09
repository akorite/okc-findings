# GPU budget: the highest-information experiment per H200-second

**State at writing (2026-11-04).** 1000 H200-s of rented compute (5000 cr at 5 cr/s), ~27 h to close
(2026-11-05 07:59 UTC). Decode layer row **2.28370 / 9,905.1 µs, #88/1227**; top-20 = **2.3900 (+4.65%)**;
board noise **SD 1.17%** per draw (a 4-draw mean resolves ~1.15%). Quota: 40 checks + 20 runs/day (brief); judge
queue ~30–40 min/job (`docs/HANDOFF-2026-11-03.md:205`).

Every number below is either measured in this repo (cited `path:line`) or marked **[ESTIMATE]**.

---

## 1. Where a submission's seconds actually go (measured)

The judge reports `result.records[0].prepare_s` and timestamps `created_at/started_at/finished_at`. Three
complete jobs, all one-candidate:

| job | mode | exec wall (s) | `prepare_s` (s) | everything else (s) | source |
|---|---|---|---|---|---|
| `7cd40fdd1a934e27` (layer 2.248) | check | **259.6** | **140.19** | 119.4 (12 shape profiles + 13 tests + image + upload) | `artifacts/cas/profiles/82c098ae…7cd40fdd1a934e27.json` |
| `a857807a067a4ad9` (MoE) | run | **219.0** | **167.54** | 51.5 | `artifacts/analysis/moe_board_job.json` (`result.records[0]`, timestamps) |
| `83179200c2124bd8` (bf16) | run | **215.0** | **171.53** | 43.5 | `artifacts/analysis/bf16_job.json` |
| `attn_job` (custom harness run) | run | **339.6** | — | — | `artifacts/analysis/attn_job.json` |

`prepare_s` is taken with CUDA events around `prepare()` (`src/okc_loop/colab_harness.py:357-366`); on an
idle stream the event pair tracks submission time, so it is effectively wall clock — which is also how the
judge enforces its 300 s cap (`docs/HANDOFF-2026-11-03.md` §1).

**Consequences, before any experiment choice:**

* `prepare()` is **54–80 %** of a job's wall (76–80 % on `run` jobs; 54 % on the check, which also pays for the
  12 per-shape profiles); the timed 12-shape pass is ~1 s of GPU. A job is a *build*,
  not a measurement, and the rental's "H200-seconds" are therefore almost entirely CPU/nvcc seconds if it
  meters wall clock.
* The floor of a "measure-only" job is the 43–52 s non-prepare overhead seen on `run` jobs, plus whatever
  `prepare()` still has to do.
* A check costs ~68–76 s more than a run at equal prepare — that delta *is* the CUPTI profiling pass
  (compare 119.4 s vs 43.5–51.5 s), and it buys 12 per-shape kernel tables (§4).

---

## 2. Q1 — what dominates a `load_inline` first run, and how to separate it

### 2.1 Measured split

| component | number | source |
|---|---|---|
| CUDA device source in the incumbent file | **9,252 lines / 458 KB** — `ATT7` 3,592 l (187.5 KB), `MOE` 2,982 l (150.1 KB), bfp family ~1.9 k l, 3 glue modules 714 l | measured by AST over `src/okc_loop/vendored_work/layer/klr_fci5.py` |
| `load_inline` builds inside `prepare()` | **10**, launched as 10 threads | `klr_fci5.py:11245` (`_att7, _moe, _moe_tiny, _moe_tiny2, _moe_mid, bfp__get_ext, _graft_glue_module, _graft_b2_module, _sl_cluster_module, _moe(32)`) |
| nvcc for those builds | **~85 s** (the file's own annotation: `# ~85 s of nvcc, in parallel`) | `klr_fci5.py:11245` |
| independent compile gate (7 of 10 modules, concurrent, cold cache, A100) | **89.8 s wall**; the 3 `tvm_ffi` modules cannot be gated there | `docs/PROGRESS.md:129-136` |
| offline device-only front end (`nvrtc` + `ptxas`, no GPU, every define tuple the file declares) | **1.2–1.9 s** for the whole file | `tools/cu_build.py` — measured today on this workstation |
| weight repack in `prepare()` (906 MB NVFP4) | 2–4 s **[ESTIMATE]** (≈2 GB of traffic at 2–3 TB/s) | `klr_fci5.py:11223-11228` |
| prepare-time tuner + attention self-test (GPU work) | 10–40 s **[ESTIMATE]**, ≤1 s of it kernel time: 9 T values × ~15 cells × `_graph_time(reps=5)` | `klr_fci5.py:11062-11125`, `11286` |

**So: compilation dominates the first run — 85–120 s of a 140–172 s `prepare()`, i.e. 55–75 % of it and
33–55 % of the whole billed job. GPU work in `prepare()` is single-digit seconds, and the timed pass is
~1 s.** The offline `cu_build.py` pass being 1.2–1.9 s also shows the *device* front end is cheap: the wall
time is the host C++ pass (torch/extension.h, templates), the ten ninja links, and their parallel scheduling.

### 2.2 Does the meter bill it? — one 25 s experiment settles it

The repo cannot tell us. Do not assume; measure, because the answer inverts the entire plan:

* **Wall-clock billing** ⇒ compile is 33–55 % of the bill ⇒ every trick that avoids a recompile is worth
  10× more than any kernel idea, and only ~4 jobs fit in 1000 s.
* **GPU-active billing** ⇒ compiles are free ⇒ skip all cache engineering and spend the budget on
  measurement fidelity and repetitions instead.

**S0 (25 s):** a script that `time.sleep(15)`, then prints the box inventory (`torch.__version__`,
`nvcc --version`, `ninja`, `import tvm_ffi`, `torch.cuda.get_device_name`, `nvidia-smi -q -d CLOCK`,
`echo $TORCH_EXTENSIONS_DIR`, free HBM/host RAM), writes a marker file into the persistent volume, and
exits. Read the credit delta (5 cr/s): ~75 cr ⇒ wall; ~0–25 cr ⇒ GPU-active. The marker is read for free
by S1 to prove volume persistence.

### 2.3 The two-submission separation (only if billing is wall-clock)

1. **Compile-only submission (~130 s).** Not `cu_build.py` — that produces no reusable artifact (device-only
   nvrtc, host `#include <torch/` section stripped). It is `cu_build.py`'s *role* that matters (a 2 s, GPU-free
   syntax/`static_assert`/register gate, `tools/cu_build.py:1-30`), and it should keep being run locally for
   free before anything is uploaded. The billed compile-only job instead does: set
   `TORCH_EXTENSIONS_DIR=/vol/te`, import the candidate module, call its ten build entry points
   (`_att7()`, `_moe()`, … `_sl_cluster_module()`, `_moe(32)`) **without** `prepare(weights)` and **without**
   weights generation or measurement, then exit. Cost = compile wall only.
2. **Measure-only submission (~60 s + measurement).** Same image, same `TORCH_EXTENSIONS_DIR`, same source
   bytes ⇒ torch's per-extension cache key (source + flags + torch/CUDA version + arch list) hits and ninja
   does nothing; `tvm_ffi` modules are named by an md5 of their source (`klr_fci5.py:12102`) and are cached
   the same way. `prepare()` then costs only repack + tuner + self-test (15–45 s **[ESTIMATE]**).
3. **Fail-fast check inside S1:** the same job imports the candidate in a *subprocess* twice (second process,
   same `TORCH_EXTENSIONS_DIR`) and prints both import wall times. Cold vs warm in one job proves the
   mechanism; the S0 marker proves whether the volume survives between jobs. If warm ≈ cold, the cache does
   not work (wrong image, wrong flags, or `torch` re-installed) — abandon this line immediately and run every
   experiment as a single fat job instead.

Do **not** spend a job on a compile-only pass until S0 says billing is wall-clock *and* the volume persists.

---

## 3. Q2 — 4–6 candidates in one billed session

### 3.1 What can and cannot be shared

The batch path already exists: `colab_runner.build_payload(candidates=[…])` takes a list and embeds it in one
self-contained harness (`src/okc_loop/colab_runner.py:107-160`, `write_harness`), and `bench_bundle.py` is
just the one-candidate wrapper. Precedent for a long custom run: 339.6 s (`artifacts/analysis/attn_job.json`).

| cost | shared? | where it lives now |
|---|---|---|
| torch import, image start | once per job | — |
| oracle modules (`reference/checks/nvfp4`) | **yes, already** | `colab_harness.py:650` (`install_oracle` once) |
| 256 MiB cold-L2 flush buffer + CUDA-graph machinery | **yes, already** | `colab_harness.py:186-190`, `Graphs()` built once at `:650` |
| weights generation (3.2 GB randn + NVFP4 quantise) | **no today — 3-line hoist** | `colab_harness.py:357`, inside `verify_candidate`, so it re-runs *per candidate* |
| compiled modules | **yes iff same source+flags** | `TORCH_EXTENSIONS_DIR` (torch cache) / tvm_ffi name-hash |
| each candidate's `prepare()` | **never** | compile + repack + tuner + attention self-test |
| each candidate's graph capture per shape | **never** | `colab_harness.py:483-560` (2 graphs + mutation checks per shape) |

Patch list for the batch session (small, mechanical):
1. hoist `weights = oracle.generate_weights(...)` from `verify_candidate` (`colab_harness.py:357`) into
   `main()` and pass it in;
2. print per-phase timers — `import`, `weights`, `compile` (time the ten build calls), `repack`, `tuner`,
   `12-shape pass` — one JSON line per candidate, so S1 yields the cold/warm split for free;
3. keep candidates in **one file** and vary them by `extra_cuda_cflags` / host constants, never as N files
   (§3.3).

### 3.2 Expected seconds per candidate (wall, single job, warm page cache)

| candidate shape | compile | prepare rest | 12-shape pass | total |
|---|---|---|---|---|
| **host-side only** (constant, plan table, `_TUNED7`, `BFP_PDL_N`, `GLE_RC`, `FT_T`, `mm_pick`, moe band edges — the 12 "free to re-time" constants in `docs/PROGRESS.md:209-224`) | 0 | 15–30 s **[ESTIMATE]** | ~1 s | **16–31 s** |
| **one small module flag** (ATT7 `-DOKC_QDIRECT/-DPK2`, bfp `SK2_DELAY`) | 25–60 s **[ESTIMATE]** (one TU, 174–3,592 l) | 15–30 s | ~1 s | **40–90 s** |
| **MoE define** (the 5 MoE builds share `MOE_DEFINES`) | 60–90 s **[ESTIMATE]** (5 modules, parallel; bounded by the measured 89.8 s for all ten) | 15–30 s | ~1 s | **75–120 s** |
| a different *file* | 85–120 s (measured) | 15–45 s | ~1 s | **100–165 s** |

Per-job fixed cost: image+import 20–30 s **[ESTIMATE]**, weights once 4–10 s **[ESTIMATE]**, cold shared
compile 85–120 s (measured), non-prepare job overhead 43–52 s (measured, §1).

**Therefore:**
* 6 candidates in one job, all host-side variants of one file ≈ **130 + 6×25 ≈ 280 s**;
* 4 host variants + 2 flag variants ≈ **380 s**;
* 6 different files ≈ **700–1,000 s** — never do this.
* With the S1 warm cache, the second and later jobs drop to **~60 s fixed + 25 s/candidate**.

### 3.3 Design rules for the batch

* **One file, N build variants.** A variant that changes a `*_DEFINES` tuple changes the extension name and
  burned two judge checks before (`docs/DEVIN_HANDOFF.md` §2 item 2) — pass new defines through
  `extra_cuda_cflags` only.
* **A/A first.** Make candidate 1 and candidate 2 byte-identical. Their spread is the session's noise floor.
  Each shape is already a median of 30 replays with a cold-L2 flush (`colab_harness.py:220-233`), so a
  same-node A/A should land **≤0.2 %** **[ESTIMATE]** vs ±0.5 % whole-file on the judge
  (`docs/PROGRESS.md:54`) and 1.17 % SD per draw. If A/A > 0.5 %, the session cannot resolve the 0.3–1.5 %
  levers we own — stop and add repetitions rather than candidates.
* **Only shape-determined things may be self-tuned in `prepare()`.** Routing-dependent bake tournaments lost
  twice because synthetic routing is not the judge's (`docs/HANDOFF-2026-11-03.md` §5); the incumbent already
  does the sound half (per-T MoE pick, dense `mm_pick`, attention plan) in `_tune`
  (`klr_fci5.py:11062-11252`).
* **Memory is not the limit.** ~1.5 GB of state per candidate against 141 GB of HBM; wall clock is.

---

## 4. Q3 — profiler artifacts for ~free

### 4.1 The measurement that decides the question

The judge's own per-shape profile already contains **both** the sum of kernel times and the union of kernel
intervals ("GPU busy (kernels overlap: PDL or streams)") — i.e. it already measures the gaps between the 7–9
kernels of the PDL chain. Extracted from the two layer rows
(`artifacts/cas/profiles/88ef4019….6341c965….txt`, `…88ef4019….b49ac780….txt`):

| shape | kernels | Σ kernel µs | union µs | gap µs | gap/edge | gap % of step |
|---|---|---|---|---|---|---|
| d16 sliding | 9 | 126.7 | 124.3 | 2.4 | 0.30 | 1.9 % |
| d16 full | 9 | 212.5 | 206.3 | 6.2 | 0.77 | 2.9 % |
| d32 sliding | 9 | 151.9 | 148.8 | 3.1 | 0.39 | 2.0 % |
| d32 full | 9 | 324.8 | 314.0 | 10.8 | 1.35 | 3.3 % |
| v1 sliding | 9 | 128.6 | 125.6 | 3.0 | 0.38 | 2.3 % |
| v1 full | 9 | 114.0 | 112.4 | 1.6 | 0.20 | 1.4 % |
| second layer run (6 shapes) | 9 | — | — | 1.7–7.8 | 0.21–0.98 | **1.97 % mean** |

and union/measured-wall = **1.046** (run 1) and **1.064** (run 2) — the profiled pass reproduces the timed
replay to ~5 %, and **the kernels sum to the wall**.

**Answer to "where is the idle time in a 9-kernel PDL chain": there is almost none between the kernels.
2.0–2.3 % of the step total, 0.20–1.35 µs per edge, already measured, for free, inside every check.** An nsys
timeline would re-derive exactly this at 1.05–1.10× wall overhead. (For contrast, on the *attention* board the
same numbers are union/wall = **0.81** — 19 % of that step is host-side, outside any kernel; that is a real
timeline finding, but it is not our board.)

The residual is **inside** the kernels: the ~3 µs ramp, the A→B hand-off dip, the 6–9 µs tail that keell0renz
measured and never removed (`docs/PLAN-2026-11-03-h200-test-plan.md` §2.1). Of the three candidate tools, only the
`%globaltimer` stamps can see them.

### 4.2 Tool comparison

| question | tool | cost | verdict |
|---|---|---|---|
| gaps / launch latency / PDL wait between the 9 kernels | `nsys profile --trace=cuda,nvtx` | activity tracing ≈1.05–1.10× wall (measured equivalent: the judge's CUPTI pass, union/wall 1.046–1.064) | **already answered at 2 % — do not run** |
| per-kernel time & share, 12 shapes | the judge's own check profile | included in a check (+68–76 s over a run: 259.6 s check vs 215–219 s runs at similar prepare) | **free — read it from every check** |
| per-kernel DRAM throughput, occupancy, stall reasons | `ncu --set detailed` | 4–8× kernel replay + serialisation **[ESTIMATE]**; ≈3–10 s wall per kernel-config **[ESTIMATE]** ⇒ 12 shapes × 9 kernels = 108 configs = **5–15 min**, i.e. 0.5–1.5× the whole budget | only for `moe_fused` on **decode 1** (23.0 µs of the 69.2 µs step, `…6341c965….txt:25-33`) |
| **intra**-kernel idle edges (ramp, A→B dip, tail) per SM | own `%globaltimer` + `%smid` stamps, one row per SM | ~1.0× (a timestamp read + a 16 B store per mark) + **one module rebuild** (25–60 s) | **the only tool that answers the remaining question** |

Also note the two hard limits on `ncu` already recorded in this repo: it locks clocks and serialises replays,
so an ncu µs must never be compared to a judge µs, and our replays are graph-based and clock-uncontrolled
(`artifacts/reports/research_02_analyzers.md:83`).

### 4.3 What one `--export` submission should run

Run it **only** if S2 leaves a load with ≥2 % unexplained (below). One job, one candidate, ~200 s:

1. warm `prepare()` (ideally from the S1 cache), then
2. `ncu --kernel-name regex:moe_fused --launch-count 1 --set detailed` on **decode 1 and verify 1 only**
   (~2 kernel-configs, 20–60 s incl. per-launch overhead **[ESTIMATE]**), and
3. the 12-shape pass with `nsys profile --trace=cuda --sample=none` written to the exported `profiles/` dir
   (one shot, 1.05–1.10× of ~1 s), so the artifact set is complete for later offline reading.

If instead the question is a specific *load's* internal idle, skip the profilers and spend the same 200 s on
the stamped build (§4.2, row 4): that is the deliverable that names the decode-1 gap
(`docs/PLAN-2026-11-03-h200-test-plan.md:154-160`).

---

## 5. Q4 — the ranked plan

Ordering principle: cheapest decisive experiment first; each row must be able to *end* a line of work.
Budget 1000 s.

| # | submission | expected s | what it answers | kill criterion / next |
|---|---|---|---|---|
| **S0** | meter + inventory probe: sleep 15 s, print `torch/nvcc/ninja/tvm_ffi` versions, GPU names, `TORCH_EXTENSIONS_DIR`, RAM/HBM, `nvidia-smi -q -d CLOCK`; write volume marker | **25** | wall-clock vs GPU-active billing (credit delta); image completeness (`tvm_ffi` present?); is the cache dir writable/persistent | none — it gates everything. GPU-active ⇒ drop §2.3 entirely and go straight to S2 with 8 candidates |
| **S1** | cache-rate probe: same job imports the base candidate **twice in two subprocesses** with `TORCH_EXTENSIONS_DIR` on the volume (cold, then warm), prints both wall times and per-phase timers | **150** (cold 130 + import 20) | the real cold/warm compile split (the number every later estimate rests on); does the torch/tvm_ffi cache hit; do the 3 tvm_ffi modules build at all on this image | warm < 40 s ⇒ every later job is cheap; warm ≈ cold ⇒ cache broken, run one fat job per experiment and never pay twice |
| **S2** | 6-candidate batch: **A/A (identical bytes) + 4 host-side-constant variants** + per-phase timers, 30 reps, cold-L2, same node | **280** (130 fixed + 6×25) | within-session noise floor (A/A); per-load ordering of the 4 constants; where the 12 "free to re-time" constants (`docs/PROGRESS.md:209-224`) actually stand | A/A > 0.5 % ⇒ session cannot resolve 0.3–1.5 % levers, stop batching and raise reps; all 4 within ±0.2 % ⇒ host-constant surface is exhausted, do not revisit |
| **S3a** | 4 compile-time flag variants, one **small** module each (ATT7 `-DOKC_QDIRECT/-DPK2`, bfp `-DSK2_DELAY_ESC`, `bfp_WIDE`/`GQ` paths) — keep the MoE family out of this row: it is 5 modules ≈ 60–90 s | **300** (130 fixed + 4×42) | whether any build flag still moves a load by ≥0.5 % | nothing ≥0.5 % (2.5× A/A noise) ⇒ close the flag sweep (`OKC_WGSFCI` 3/4/5/6/7 is already closed: 5 is best, `docs/PROGRESS.md:44-56`) |
| **S3b** | *(alternative to S3a, mutually exclusive)* profiler job per §4.3 | **200** | intra-kernel idle at decode 1 / verify 1; MoE DRAM vs the 4.287 TB/s ring ceiling | MoE already ≥90 % of the ring ⇒ close the MoE; <80 % ⇒ inspect occupancy/stalls before any further MoE build |
| **S4** | confirm: winner + base, paired, second node, 30 reps | **100** | is the S2/S3 win real across nodes (board noise 1.17 % SD, judge ±0.5 % whole file) | win < 0.5 % ⇒ do not spend a check; ≥1 % ⇒ check, then submit (board keeps the best row) |
| — | reserve | **145** | overrun / one extra inventory or batch job | — |

**Allocation:** S0 25 + S1 150 + S2 280 + S3a 300 + S4 100 = **855 s**, leaving ~145 s for overrun or one extra
inventory/batch job. If S3b replaces S3a: 755 s used, ~245 s spare for a second batch. A single MoE-family flag
sweep is 5 modules per variant (~60–90 s each) and does **not** fit beside S2 — it needs its own 450 s job, which
is why it is not in this table.

**Why this order and not "probe the board":** one judge check is 259.6 s of billed time for **one** number
(SD 1.17 %); one S2 job is 280 s for **six** paired numbers plus the noise floor plus the phase split — 6–7×
more information per second. Judge checks are quota-cheap (40/day) but time-expensive; spend them only on
candidates the rental has already ranked (S4).

**Also free, and strictly before any of the above:** `tools/cu_build.py` (1.2–1.9 s, no GPU, this workstation)
for syntax/`static_assert`/register checks, and `tools/colab_gate.py` on free Colab A100 (89.8 s measured,
`docs/PROGRESS.md:129-136`) for the full `load_inline` build. Three judge checks were already lost to the
`ptxas`/PyInit class these two catch.

---

## 6. Risks

* **Billing basis is an assumption until S0.** If wall-clock, ~55 % of every second we own is nvcc — and the
  correct response is fewer, fatter jobs, not more kernel ideas.
* **Cache-key fragility.** torch's extension cache keys on source + flags + torch/CUDA version + arch list;
  tvm_ffi keys on an md5 of the source (`klr_fci5.py:12102`). Any image difference (a re-installed torch, a
  different `-gencode`) silently invalidates 85–120 s of work. S1 exists to catch exactly this.
* **Volume persistence is not guaranteed** by anything in the repo; the S0 marker is the test.
* **`prepare()` has 300 s and we use 140–172 s** (measured) — there is ~100 s of headroom for a
  prepare-time tournament, which is a *free* A/B channel on the judge itself, but only for shape-determined
  choices (routing-dependent tournaments have lost twice).
* **Profiler numbers are not judge numbers.** ncu locks clocks and serialises replays
  (`artifacts/reports/research_02_analyzers.md:83`); use it to rank causes, never to predict µs.
* **The board row is 1.5σ lucky** (`docs/HANDOFF-2026-11-03.md` §2). Anything under +1 % will not survive the
  draw; S4's cross-node confirm is the gate, not the single best session number.
