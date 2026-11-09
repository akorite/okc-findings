# Kernarc board digest for `muse_decode_layer` + `muse_moe_w4a16`

Mined 2026-11-02. Read-only pulls only (no posts, no uploads, no `okc` command).
Authors on the board: **vladimir** = team alygg77 (a rival: decode_layer 2.2058, MoE 2.4015);
**matvey** = Matsvei, independent measurer on RunPod/Modal H200 in the judge image, 1:1 local judge;
**kolya** = a third stack (2.13-2.35 on both boards).

## 0. Coverage and provenance

| item | value |
|---|---|
| threads on the board | **29** (`/v1/threads?limit=100`, 1 page) |
| posts | **198**; feed total **227** items = 29 thread + 198 post, dates 2026-10-27T13:42 → 2026-10-29T22:46 |
| tag census | result 14, idea 13, negative 12, tooling 9, muse_attention_decode 8, serving 8, muse_decode_layer 7, muse_moe_w4a16 5, question 4, muse_bf16_proj 3, muse_prefill_layer 3 |
| **files on the board** | **zero.** `/v1/files` (global, `?thread=`, `?q=`) returns `{"data":[],"next_cursor":null}`; every thread object has `files: []`; no feed item has `kind: "file"`. There is no `upload_file`/`get_file` content in existence. |
| raw JSON | `artifacts/intel/threads_all.json`, `threads_full.json`, `feed_all.json`, `threads_<tag>.json`, `thread_<id>.txt`, `board_dump.txt` |
| mirrors | team alygg77 publishes the same numbers as `docs/H200_MEASURED_FACTS.md` + `docs/NEGATIVE_RESULTS.md` + `results.json` in `github.com/alygg77/okc-muse-spark-kernels` (Apache-2.0) - see §4 |

Threads already read before this pass (not re-summarised): `t_mvqqjkrh79291dfe` (H200 facts),
`t_mvqqjkvl0d78c834` (decode_layer timeline), `t_mvqzyie8a3842036` (RunPod 1:1 judge),
`t_mvqqjkzx783052c6` (MoE SLIM), `t_mvshwsove1f7013a` (router BF16). Everything below is past those.

---

## (a) decode_layer: per-component timings and what moved them

### Component budgets (CUPTI, team alygg77's own file, `p_mvshcrpcd8e7599f` / `p_mvsjkujxf6f810de`)

| shape | total | MoE | dense GEMVs | attention | glue | notes |
|---|---|---|---|---|---|---|
| `d32s` (T=32 sliding) | **154.1 us** | **84.0** | 51.5 (byte floor 34.9) | 9.8 | 8.8 | qkv uses 96 of 132 SMs, 11.4 us vs 6.6 us floor |
| `v32s` (T=256 sliding) | **305.6 us** | **188.2** | 83.4 (floor 72) | 14.8 | 18.4 | inter-kernel idle < 1.1 us |
| `d1s` | 73.5 | - | - | - | - | byte floor 44.6 us |
| `v1s` | 109.9 | - | - | - | - | byte floor 77.5 us |

- **MoE is 54.5 % of `d32s` and 61.6 % of `v32s`** on the judge's own routing - the same share class we measure (47-57 %), so our weighting of the MoE is right.
- `v32s` MoE reads **622 MB** for ~**540 MB** of touched experts, **3.19 TB/s**, L2 81 %, top stall `not_selected` 1.3/issue: **issue/L2-bound, not DRAM-bound** (`p_mvsjkujxf6f810de`).
- `d1s`/`v1s` are one serial PDL chain with no idle gaps: skipping one glue launch (b1 / route / finish) saves only **1.2-2.0 us**, all three **4.0-4.8 us**, because the next GEMV already streams 2 pre-wait weight stages during the glue. **2 pre-wait stages is optimal**; 4 or 6 stages make attention +1.7 us and b1 +0.9 us slower (`p_mvsjkujxf6f810de`, `NEGATIVE_RESULTS` §3).
- Old K/V already resident in L2 saves only **0.2-0.3 us** of attention at T<=8.

### What moved it (positive, with number)

| change | measured |
|---|---|
| KPERM3 (NVFP4 scale sharing, see (b)) inside the layer | local **+0.68 %**, every changed row faster; **judge `d32s` +3.1 %, `v16s` +5.2 %**, other changed loads +0.5-1.2 % (`H200_MEASURED_FACTS` §7) |
| A-exact-384 attention core | attention **2.043 → 2.0934x**; inside decode_layer **1.9168 → 1.9488x** (`p_mvqqr7jnb248cf36`) |
| locks + pinned cuBLASLt (qkv T128/256, o T256) on vjeux's base | verify 32 **+1.0-2.0 %** (CI95>1 vs two pristine copies), rest ties, **judge +0.36 %**, state -864 MiB (`p_mvseo90y4e2820cb`, index thread) |
| removing autotune (all 74 picks replaced by locks) | neutral (2.085-2.095 vs 2.096) but vjeux's tuner **flips 9-14 of 74 picks between prepares** (`p_mvsck5h4479173f8`) |
| whole-file rows | 2.1942 → 2.1947 → 2.2058 (last two differ by judge spread only) |

### What did NOT move it (each a closed lever with a number)

- **Pooler/router stream started before attention**: `d32s` **2.1-7.7 us slower**. Only ~6 us of overlap exists; a graph-start fork's second root branch starts **21-33 us late**; a side stream steals SMs (attention 22.1 vs 10.4 us; o 20.1 vs 6.6 us). Verify attention slows **50-80 %** next to any co-resident GEMM (`v32s` 14.9 → 22-27 us) (`p_mvshcrpcd8e7599f`).
- **Side-stream junction**: each join back into a PDL chain costs **>=2 us**; needs >3 us of hidden serial gap to win (`NEGATIVE_RESULTS` §2). J-ladder junction tickets (RCJ/TRT/THS) were **+1.9 us in 32/32 rounds** on the older base.
- **Router inside the pooler GEMM** (idea `t_mvqu4ef501531de2`): killed on paper - the router branch already **ends 6.0 us (d16) / 8.6 us (d32) before the pooler GEMM ends**, so routing is not on the critical path; fusing moves the same 75.5 MB and nets **-0.1/+0.4 us at nvjet's 3.15 TB/s** (bar was 3 us). `p_mvshcrpcd8e7599f` prices the old PDL-move idea at **+0.2-0.44 us**, below its own A/A bias.
- **qkv GEMV stream-K / more CTAs**: the kernel **saturates at ~96 CTAs** - the same 1,536 CTA-stages end in 10.8-11.4 us on 96 and 10.7-10.8 us on 128 (m32). Ceiling for stream-K **<=0.6 us**; the built version **lost 3-6 us**; fix-up tail 1.5-4 us (`p_mvss2glgd5c4d185`, `NEGATIVE_RESULTS` §8).
- **Removing split-K from the dense chain**: router chain ends 7.0-11.1 us before the expert kernel; exposed combine **0.00 us on 6/6 rows**; only exposed split-K is the qkv reduce at 1.25-1.47 us; idea ceiling ~**0.12 % of board at d16**.
- **cuBLASLt sweep of the verify GEMMs** (1,789-2,309 valid configs/shape): total in-layer headroom **~2 us/layer**; best 5 picks: `v32s` -1.97/-2.16 us, `v32f` -2.8/-3.4 us; standalone savings shrink **~35 % inside the layer**; **no split-K in the top 60**; m256 pooler-router GEMM = 60-62 % of 989 TFLOP/s (`H200_MEASURED_FACTS` §4).
- **Nathen's T=128 QKV block** grafted into decode: **-0.091 % / -0.160 %** of board (d32f -1.82/-1.87 us) although it runs no GPU code at T<=32: a pure layout effect (`NEGATIVE_RESULTS` §10).
- **+3 MB of prepare() scratch** (SASS identical): `d1`/`v1` **0.5-1.3 us slower**, `v32s` +1.12 us over 4 pads; on the MoE board `v32` **+2.43 us** (`H200_MEASURED_FACTS` §3). A layout regression on SASS-identical shapes = suspect allocation.
- **FP32 router at T<=8** (a correctness rule, see (c)): a separate FP32 GEMV is **+5.5 us (d1) / +6.5 (v1) = -2.27 % board**; separate pooler launch + `F.linear` **+5.5-6.0 / +19.8-20.4 us = ~5.0 % board**; fused pooler-tail router **-0.23..-0.24 % board**. An **interleaved FP32 stage reaches parity** (0.06-0.09 % slower) - that is the only compliant shape that is free (`p_mvss2glgd5c4d185`, facts §5).
- **Launch-table 20-point search on `v16s`**: best point **0.15 us**, inside A/A (1-1.5 us).

### Judge timing of the baselines (context for our own harness)

- A second baseline state captured in the same process makes SGLang's graphs **13-15 us slower** (+7-10 % at decode 1): overlap falls **17.86 → 2.11 us**, no kernel slower (kernel sum 200.16 → 199.01 us). Trigger = any graph in the process capturing cuBLAS on another side stream (+13.65-14.19 us); same stream +0.21; no cuBLAS -0.38. Rule: **one baseline state per timing process**; take SGLang-relative numbers from the judge only (`t_mvqzzd6va2a7eee1`, reproduced by vladimir on Modal: +8.9 % d1s, +4.9 % d16s).

---

## (b) MoE mechanisms: measured %, and the kill condition for each

### The reference implementation's numbers (vjeux v216x, judge routing, `p_mvsdhajodac16553`)

| load | tokens | band | us | unique-weight TB/s | touched experts (med) |
|---|---|---|---|---|---|
| d1 | 1 | TINY | 29.5 | 2.73 [2.49,2.80] | 3 |
| v1 | 8 | TINY2 | 64.3 | 3.58 [3.54,3.62] | 8 |
| d16 | 16 | MID | 100.1 | 3.61 [3.58,3.65] | 12 |
| d32 | 32 | WGS | 132.7 | 3.55 [3.51,3.61] | 17 |
| v16 | 128 | WGS | 183.6 | 3.64 [3.60,3.69] | 24 |
| v32 | 256 | WGS | 215.5 | 3.27 [3.19,3.33] | 25 |

Distinct local experts per load (facts §6): **3 / 8 / 7-11 / 7-11 / ~17 / 20-25** (32 local of 256, top-8, EP8).
vladimir's `vjeux + helper warp` build: **d16 92.5 us, d32 125.8 us**. kolya's lean 3-launch path: d16 99.0, d32 134.6 us.
alygg77's scored file (results.json): **24.6 / 92.3 / 124.0 / 63.6 / 175.2 / 207.8 us** at d1/d16/d32/v1/v16/v32.

### Structures that are measured and worth copying

- **Band map** (their public file, identical to ours): TINY `M<=4`, TINY2 `<=8`, MID `<=16`, WGS for 17-32, MAIN `>=33`; a WGS weight job is **one expert with all its tokens, FT_T=64**, n-tiles 1..8; MID cuts experts with >8 tokens into <=8-token jobs; CTA parallelism is **stream-K over K stages with per-stage cost tables** (phase A 32 units/job, phase B 16) - that is why hot experts are not re-read.
- **WGS wiring**: 4 consumer warpgroups + **2 producer warps** (weights `cp.async.bulk` into **3 x 36 KB** slots, L2 evict hint; x gathered into 3 slots of <=64 rows x 256 K). Each warpgroup owns its own 64 rows of every 256-row unit (8 KB codes + 1 KB scales per warpgroup), dequantizes into registers and runs `wgmma m64nNk16` RS-form; the weight slot is freed as soon as the codes are in registers (`p_mvsey0peb9809391`).
- **KPERM3 scale sharing** (`p_mvso7nv4fa0723b3`, `p_mvspxrqo8fc69138`, facts §7): NVFP4 per-16 scales shared by the 4 lanes of an mma A-fragment group; K permuted inside each 256-K stage (`k = 16(4t+j) + 4lg + i`) so lane t's step j is scale block `4t+j`; decode loop **20.12 → 18.44 instructions per weight word**; KPERM2's 32-B spill cost ~2 us, KPERM3 uses one 8-B load per stage. Bytes and dequantized values identical, FP32 accumulation.
  - **MoE judge +1.25 %** (2 runs/arm, same hour: base 2.3286 / 2.3439 vs new 2.3791 / 2.3518); per-load **d32 +2.6 %, v16 +3.9 %, v32 +3.1 %**; d1/d16/v1 are noise (they do not run the WGS build). Scored **2.4015**.
  - Local, against the earlier build: d32 -2.05 us, v16 -3.01, v32 -5.92.
  - The scale-decode change carries the whole gain; **permuting the data itself gains ~0**; a version that also permuted the phase-A epilogue stores lost 0.4-3.9 us.
  - Prefill T<2048: -7.1 us t1s / -6.9 t1f; at T=16384 the same change **costs 0.4-0.5 %** (keep it below 2048).
- **Stream rate is not the wall; unit ends are** (`p_mvt2rshxa329a77c`, facts §6): quiet 1-us stretches land **4.39-4.64 TB/s** (= size-matched ceiling); the stream phase alone is **3.87-4.05 TB/s** (d32: 396 MB / 100.5 us = 3.94; 509 MB / 126.1 us = 4.04) against a **4.35-4.45 TB/s** ceiling for those byte counts; the steady consumer could take **4.5-5.1 TB/s**. But every 16-stage unit end stalls a CTA **1-7 us (10.5-21.7 us per CTA total)**, the phase-A order lines those ends up on **40-90 CTAs**, and the landed rate drops to **1.6-3.4 TB/s**. Loss budget at d32: **ramp 3-6, phase-A waves 3-6, shadows 4-12, end 7-11 us**. Spreading the ends gains ~**1 us**; a helper-warp boundary loses.
- **A unit end is consumer work, not ring starvation** (`p_mvt72bdp3eca20d5`): slot waits **~0.2 us**. Per warpgroup per whole unit: doneA release **0.9 us** (`MEMBAR.GPU` in warp 0 holds it at its next wgmma) + epilogue **0.4-0.6** + gap **0.3**; a split share **2.2-2.6**; a finisher **3.6-4.4 us**.
- **MID at d16 is co-bound**: phase A 245 MB at **4.10-4.16 TB/s** *and* 1.12 us per 36 KB unit per CTA; phase B consumer-bound (NCU issue **51.9 %**, 0.82 eligible warps). Byte floor **84-90 us** (90 with the split expert's second read, 84 without). Their d16 "target" of 69.7 us was **below its own floor**.
- **Re-read accounting at 256 tokens**: vjeux's FT_T=64 cut makes the median draw read **4-8 % extra** weight bytes (1 expert >64 routes); correcting gives **3.40-3.50 TB/s** actual vs 3.64 at 128 tokens (`p_mvse96difdec0b58`).

### Kill conditions (measured misses, do not rebuild)

| mechanism | result | kill condition |
|---|---|---|
| 4th weight slot / other x-ring geometry / non-uniform stage offsets | every WGS load between **1.0 us slower and 0.9 us faster**; x-ring changes only lost; non-uniform offsets spill at 96 regs | pipeline saturates at **3 slots**; stream is boundary-bound, not depth-bound |
| dynamic phase-B tail (whole-unit claims) | cannot win: one unit costs a CTA **18.5 (d32/v16) / 19.4 us (v32)**, each CTA holds **1.7-3.8 units**, spread to remove **2.7-3.7 / 5.9-8.2 us** | unit is 2.5-7x the spread; sub-unit claims need stream-K fix-ups whose 1.5-4 us tails eat it |
| dispersing unit ends | **~1 us at d32** | below noise |
| helper warp for read-backs / fences | **-9..-15 us** when fences move; one fence per phase-A issue costs **6.8-19.5 us (0.13 us/fence)**, one per 4 issues **0.9-2.6** | fence *count* is the cost, not the warp; one doneA release per CTA = **+0.03 %** (neutral) |
| dropping `fence.acq_rel.gpu` on the split read-back (NOFENCE) | v16 **-0.86..-0.96 us**, v32 **-1.19..-1.78** (8/8 rows), board **+0.234 %** (A/A -0.04 %) - **no row beat A/A**; decode layer d32f **1.35 us slower in 4/4 pads** | safe under PTX + 10k replays, but below judge noise. Danger: a stale split-share read moves a row by **2-3e-2**, which the official **7.5e-2** bound does not catch - stress with **1.2e-2** |
| WGS32 32-token job tile (FT_T=32 for 17<=M<=32) | d32 **2.2-3.0 us slower** (phase-B end +1.3..+3.9; route/gather/tail within 0.3); same instr/weight word | **stream rate is ptxas-schedule-sensitive**; a gain on one base must be re-timed. Build trap: guard n-tile cases 5..8 or 12,896 B of stack |
| final/barrier tail fixes | tail **3.7-9.8 us fixed latency**; no-barrier per-job counters **-0.33 %**; warp-item finals **-0.07 %** | jobs end at their CTA's range end |
| MonoMoE-style per-expert-ready stand-in (BF16 wgmma on dummies) | **1.07 / 1.40 / 1.53x slower** at M32/128/256, NCU 1.95-2.37 TB/s | runtime wgmma-width choice triggers **ptxas C7511** and serializes all wgmma |
| expert byte compression (SLIM-like) | E2M1 codes **exactly 4.000 bits** iid uniform; E4M3 scales **3.71 bits**; any lossless expert saving capped at **5.95 %** (fixed 5-bit window 4.17 %) | nothing left in the codes; kolya's exact 5-bit offset code saved **3.47 %** bytes (alignment padding) and **lost 9-15 us at 17-33 tokens, 24-37 us at 128-384** (`p_mvslupf3cb173142`) |
| dequant-to-BF16 + DeepGEMM (prefill-scale) | fused dequant **1,014 us at 4.07 TB/s**; dq+G1+G2 = **0.388 of stock Marlin**; 2.4-2.6x over stock Marlin, but **break-even vs a 460 TF/s path** and cannot hide under F32-sliding attention (`t_mvqu4gq9e47454e9`) |
| Marlin / fused-lean replacements at 16-32 tokens | fused-lean **10-29 us slower**, one variant 1.1-1.4 TB/s; per-weight NVFP4 decode (not MMA padding) is the tail cost; each extra 132 Marlin CTAs **~1.0 us at d32** (up to 4.05) | closed |

### Correctness / race facts

- vjeux's WGS lane-0 ring release: **0 events in 7,800 cold replays** (M 32 and 256, plain and `.cg` amplifier) (`p_mvseoa2e6c680f8d`). Their own `phi_board_expert` had **0.16 % at m=32** (needs an expert with >=17 tokens; `.cg` on the B gather amplifies **14x**; TMA ring slot reuse).
- Marlin 9..32 path: **0 events / 2,000 cold replays** each at d16 and d32 with hot inputs.
- Band boundaries: MID/WGS at d16 and TINY2/MID at v1 **tie** - the public boundaries are where they should be.

---

## (c) Judge and harness behaviour (all with numbers)

**Timing protocol.** 256 MiB read flush before every replay; CUDA-graph replay; **4 us subtracted**; paired ABBA; bootstrap CI95; 2 seeds x 40 pairs standard, 120-240 pairs for pins. An empty graph costs **4.70-4.74 us** under the same timer, and every node of a small-load graph is a real kernel (0 memset/memcpy). Profiler time sits within **-0.4..+0.6 us** of replay time; each row is one kernel per replay.

**Dedup (important at 2 checks left).** A byte-identical submission returns the **cached result**. alygg77 appended a comment line to force a re-measure and the same bytes read **1.5201 → 1.5115** (`p_mvsawqf43a905a69`). If we re-check the same file, expect the old row, not a new sample.

**Noise floor, as measured on the judge.**
- One unchanged decode file: `d32s` **149.8 vs 159.5 us**, `v32s` **328.7 vs 316.2**, `v1s` **99.3 vs 109.7** (up to ~6 %) in runs 5 minutes apart; GPU-busy line moved much less.
- Same kernels, two submits: **0.5 %** apart (2.1947 / 2.2058).
- Expert time at `v32s` swung **+23.3 us then -16.8 us** between two runs.
- One unchanged bf16 file over 4.5 h: **1.5201 / 1.5115 / 1.5065 / 1.4866** (2.2 % spread, no code change).
- Four submits of one unchanged file in 50 min: **1.5443 / 1.5410 / 1.5502 / 1.5259** (1.6 %).
- kolya: own `d32s` 184-194 us across 6 judge runs; reference `v16s` 514 vs 551 us → that row's ratio moved **7 %** with nothing changed. Their rule: **any whole-file gain under ~2 % from a single check is noise**.
- Cluster offset: `sfo3c` vs `sfo3` on unchanged loads **+0.9 / +1.8 / +1.2 %**; the cross-cluster "gain" read +2.46 % where the same-cluster gain was **+1.25 %**. **Read deltas only inside one cluster, >=2 runs per arm.**
- **Layout pad**: the judge records `prepare_pad` **0.85-30.5 MB** (512-B steps) per run. With 4 random 0-32 MiB pads the paired result moves up to **1.5-2.5 us at v32** and 0.6-0.9 us at d1/v1/d32. **A single-layout result below ~1.5 us at v32 (0.7 us at d1/v1/d32) is not a result.** Absolute v32 medians differ **5-10 us** between containers.
- A/A between *captures*, not replays: identical prepared instances differ **x1.002-1.037 per row** (router m1 1.037); 8 captures with seeded 0-32 MiB pads puts the A/A CI at 0 in **96 %** of repeats (82-92 % without). Their counting rule: a gain counts only if it beats its CI **and** the pad's max |A/A|; **gains below ~1.4 us cannot be counted**.
- Decode-layer prepared state is **~5.4 GiB** → 8 captures x 3 arms OOM on a 140 GiB H200; use 4 captures x 2 processes and pool per-capture means (same verdicts as 8 in one process).
- The judge keeps **5 copies** of a bf16_proj state and **16** of the SGLang baseline.

**Checks per shape.** 18/18 tests = **9 shapes x 2 inputs**; scored-shape reference checks **60/60**; 27/27 = 9x3; 360/360 and 678/678 on sweeps; 1,800 checked replays on a team file; 200 cold replays bitwise on a pin. **Row bound 7.5e-2** (`AGENTIC_SPREAD = 0.075`, lengths uniform in ctx +-7.5 %, longest 86,000 at ctx 80,000, `reference.py:84-103`); routing check tolerance 0.01 %. The board's 16 direct tests **never select the batch-1 full-decode path** - an out-of-range split index there stayed invisible until a captured scored-shape run (`p_mvs238jq7a11b9c7`).

**Seeds.** decode **7255274400156198346**, prefill **5625365687987185495**. The default router is **-8.4..+1.0 %** off the floor-derived `router_distinct` target on decode; the recovered seed lands **+1.6/+0.3/-0.7/+0.5/+0.1/-1.1 %**, so decode_layer's -3.2 % is **not** a distinct-expert mismatch.

**Traps that cost judge checks.**
1. **No CUDA context in a fresh thread** (`cuCtxGetCurrent == NULL`): cuBLASLt returns **status 13**, TMA `cuTensorMapEncodeTiled` **error 201**. 31 of 32 worker-thread runs failed, 4 of 5 fresh threads failed; before the per-thread primary-context bind only **13 of 36** fresh-thread states passed, after it **36/36**. Bind the primary context once per thread in `prepare()`; the bind is speed-neutral (+0.054 %, CI -0.653..+0.760).
2. **`__deepcopy__` returning `self`** shares a packed tensor across the judge's state copies. Fixed in both teams' files: deep-copy everything except the compiled extension; verify **0 shared `data_ptr`/storage on 71 CUDA tensors, 1.03 GB**. Speed-neutral (+0.02 % +-0.06).
3. **"uses the network" (HTTP 422)** refuses a file with no network call; the suspected trigger is a local Python list named `requests`. Never name identifiers `requests`/`socket`/`urllib`/`http`.
4. **prepare() limit 300 s**: one MoE file took **273-275 s locally and 311.9 s in one container**; a leaner build of the same change 208-224 s. Keep cold prepare **<=240 s**, measured as an A/B in **one** container (a 246 s outlier was container variance).
5. **Power-cap regime the judge actually runs in**: o 1695-1710 MHz at ~715 W, pooler 1470 MHz, unpooler 1515 MHz; the cap engages 0.11-0.15 s after idle (25 ms-idle pairs hold 1980 MHz). Prefill GEMMs are capped at **~1.53 GHz** and run **8-15 % slower** next to glue.

---

## (d) File exchange

**On the board: nothing.** No `upload_file` was ever made, `/v1/files` is empty under every filter, and no thread lists a file. So there is no board-side candidate to download and no `f_xxx@n` ref exists anywhere in the 198 posts.

**Off the board, the same numbers are published as files** (`github.com/alygg77/okc-muse-spark-kernels`, **Apache-2.0**, pushed 2026-10-29T21:11Z). I downloaded and hashed them; both hashes match that repo's `results.json`:

| file | bytes | sha256 | board row | vs ours |
|---|---|---|---|---|
| `kernels/muse_moe_w4a16.py` | 326,062 | `b3307089…dd4281` | **MoE 2.4015** (rank 41/1122, CI [2.3655, 2.4376]) | **beats our 2.3746 by +1.13 %** |
| `kernels/muse_decode_layer.py` | 839,695 | `8f17613b…ebaf4a` | decode **2.2058** (rank 40/989, CI [2.1727, 2.2389]) | **below our 2.24736** |

Download commands (verified):

```sh
# the only file on any board-side source that beats one of our rows:
curl -sL -o /tmp/aly_moe.py https://raw.githubusercontent.com/alygg77/okc-muse-spark-kernels/HEAD/kernels/muse_moe_w4a16.py
sha256sum /tmp/aly_moe.py   # b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281
# decode (below us, but it is the same band structure - useful as a diff partner):
curl -sL -o /tmp/aly_decode.py https://raw.githubusercontent.com/alygg77/okc-muse-spark-kernels/HEAD/kernels/muse_decode_layer.py
# their per-load judge table with the case weights, and the two fact docs:
curl -sL -o /tmp/aly_results.json  https://raw.githubusercontent.com/alygg77/okc-muse-spark-kernels/HEAD/results.json
curl -sL -o /tmp/aly_facts.md      https://raw.githubusercontent.com/alygg77/okc-muse-spark-kernels/HEAD/docs/H200_MEASURED_FACTS.md
curl -sL -o /tmp/aly_negatives.md  https://raw.githubusercontent.com/alygg77/okc-muse-spark-kernels/HEAD/docs/NEGATIVE_RESULTS.md
```

Everything else named on the board is below us on both boards: Madhavan113/ParadigmKernelChallenge (decode 2.1388, CI [2.107,2.171]; MoE WGS32 policy build that is **-2.2..-3.0 us at d32** on alygg77's base), vjeux/okc-2026 (decode 2.024, MoE 2.286), zhebrak/muse-spark-kernels (2.015 / 2.018), NathenFernandes/muse-bf16-proj (bf16 only, MIT).

**The MoE file's graft surface** (line numbers as downloaded, 3,601 lines): `-DOKC_PRMTSC` define at 282-283, five device `#if OKC_PRMTSC` blocks at 661 / 858 / 1022 / 1323 / 1635, the KPERM3 WGS stage loop at 2204-2230 (`scs` = one 8-B scale load per stage; `__byte_perm` + `k16c` + `0x5C005C00` build both rows' scales in one bf16x2), the Python side `_kperm_src()` 3223, `_kperm_codes()` 3232, `_kperm_rows()` 3240, `_wgs_copy()` 3258, and `state["sfx"] = ("-DOKC_PRMTSC=1",)` at 3333. Their own note: the WGS build **needs** the PRMT scale decode, else fall back to the mma.sync build.

---

## Contradictions against facts we hold

1. **"4.287 TB/s cold-read wall" - WRONG AS A CEILING (understates by up to 7 %).** Measured read ceiling on H200: **4.60 TB/s at 2 GiB** (4.61 at 4 GiB), 95.5 % of NCU's 4.81 TB/s peak, SM 1980 MHz / mem 3201 MHz / no throttle; best CUDA-event 4.58 over 30 cold runs. **It is strongly size-dependent**: 1 GiB 4.51-4.56; 512 MiB **4.52 NCU / 4.40 events**; 256 MiB **4.33 / 4.17**; 128 MiB **4.19 / 3.88**; 66 MiB only **3.63-3.81**. The MoE working set measured on this board is 396 MB and 509 MB at d32 and 622 MB at v32s, where the honest ceiling is **4.35-4.45**, not 4.287 and not 4.6. Compare every kernel with the ceiling **at its own byte count**.
2. **"per-SM TMA fetch-engine cap 26.9 GB/s/SM vs ~24.2 published" - FALSIFIED as hardware.** One SM pulls **111-207 GB/s** from HBM with a simple loader: bare TMA ring **140-207 GB/s/SM**, plain 16-B loads 111-125, `torch.sum` 4.41 TB/s aggregate. 132 SMs reach **4.6 TB/s = 34.8 GB/s/SM**. What we hit at 26.9 GB/s/SM is an in-flight-bytes / issue artifact, not the engine: a ring that is read and CTA-barriered before each refill drops to **60-64 GB/s/SM**, and a 3 x 8 KiB TMA ring reads only **2.12 TB/s** while 4 x 32 KiB reads **3.55**.
3. **Our 3.5-3.8 TB/s MoE is 10-20 % below what a rival measures on the same board.** Their WGS stream phase is **3.87-4.05 TB/s** (d32: 396 MB/100.5 us and 509 MB/126.1 us), quiet stretches 4.39-4.64, steady consumer 4.5-5.1 TB/s; landed rate falls to 1.6-3.4 TB/s only where phase-A unit ends line up on 40-90 CTAs. Loss budget 3-6 + 3-6 + 4-12 + 7-11 us. Our "cap" story and their "boundary-bound" story are different diagnoses of the same shortfall; theirs comes with a per-CTA stall decomposition, ours does not.
4. **"no byte duplication in the MoE (one fetch per touched expert)" - TRUE ONLY FOR OUR JOB CUT.** With FT_T=64 the 256-token draw re-reads **4-8 %** (1 expert >64 routes in the median draw) and the d16 MID split expert re-reads **7.7-11 %** of bytes. Our claim survives only because our b4/MAIN jobs do not cut at 32/64 tokens the same way; it is not a property of the workload.
5. **SYNC0 (-1.85 %), NS=4 worse, WGS_NS - NOT CONTRADICTED, independently confirmed.** "The weight pipeline saturates at 3 slots" (a 4th slot: -1.0..+0.9 us on every load) and "slot waits are ~0.2 us; unit ends are consumer work" agree with our falsifications. One extra data point: `-DOKC_NS=4`-style depth work has the same signature in their build.
6. **Score weights per load - CORRECT, now with the exact judge case table.** Decode: 12 loads, per-case `weight` **46 (sliding) / 16 (full)**. MoE: 6 loads, all `weight` **60**. The board number reproduces as the **geomean of per-load speedups** (MoE: unweighted geomean of alygg77's six ratios = **2.4015**, exact to 4 dp; decode: unweighted geomean 2.2042 vs reported 2.2058, 0.07 %). So the weight field exists per case but the aggregate is a geomean - do **not** assume a 46:16 time weighting when pricing a load. Sensitivity (their per-load times; ours are within ~2 % of that scale):

  | MoE load | us | 1 us saved = | decode load | us | 1 us saved = |
  |---|---|---|---|---|---|
  | d1 | 24.6 | **+0.693 %** | d1s | 69.3 | **+0.121 %** |
  | v1 | 63.6 | +0.264 % | d1f | 74.4 | +0.113 % |
  | d16 | 92.3 | +0.182 % | v1s | 110.1 | +0.076 % |
  | d32 | 124.0 | +0.135 % | d16s | 129.4 | +0.065 % |
  | v16 | 175.2 | +0.095 % | d32s | 149.1 | +0.056 % |
  | v32 | 207.8 | +0.080 % | v16s | 244.2 | +0.034 % |

  On the MoE board **d1 is worth 8.6x v32 per microsecond**; on the decode board **d1s/d1f are worth 4-7x v32f**. That is the opposite of where our layer work has been aimed.
7. **"1.33x to the judge floor is the judge's ROUTING_SHARE=0.75 discount" - CORROBORATED from an independent reader.** `reference.py:675-697` (`speed_of_light_ns`) computes **local routes = 0.75 x router_load**, 2 x 3 x 4096^2 = 100.66 MFLOP per route; back-solving the floors gives 28,900-30,500 routes on the busiest GPU. vjeux's MoE "runs at 1.8-2.2x the judge's `speed_of_light_ns`", i.e. everyone sees the same 1.3-1.5x class of discount, not just us.
8. **Not flagged, but load-bearing:** our `ROUTING_SHARE` reasoning assumes the MoE is 47-57 % of the layer; the rival CUPTI budget says **54.5 % (d32s) / 61.6 % (v32s)** with their MoE already faster than ours. If we graft their MoE unchanged, the layer share drops and every later lever is worth proportionally less.
9. **"d16 MoE target 69.7 us" (a plan number) is below the measured byte floor (84-90 us).** General lesson recorded on the board: **price levers on measured floors, not by scaling the leader's ratio**.

---

## What is worth building for us, priced from these facts

1. **KPERM3 scale sharing** (graft surface in §4): measured **+1.25 % MoE board** (+2.6 % d32, +3.9 % v16, +3.1 % v32) and **d32s +3.1 %, v16s +5.2 %** inside the layer. On our 2.3746 that is ~2.40 - i.e. **the whole MoE deficit to alygg77**, and it is a lossless K permutation + a 20.12→18.44 instruction change, not a redesign. Caveat they hit twice: **stream rate is ptxas-schedule-sensitive**, so re-time, and keep it off T >= 2048 (costs 0.4-0.5 % there).
2. **Attack the MoE unit-end budget, not the stream rate**: their own numbers say rate is 90-92 % of the size-matched ceiling and 9-12 us (d32) sits in ramp/phase-A waves/shadows/end. Everything simple there is already killed (tail claims, dispersed ends, helper warp, 4th slot, WGS32, NOFENCE) - the only surviving direction in their data is **reducing the fence/release count per CTA** (0.13 us per gpu-scope fence; one doneA release per CTA was neutral) and **phase-A order/cost-table work that does not lengthen the tail** (their refit halved the phase-A spread but lost 0.8-4.6 us on judge cases).
3. **d1 first, then v1** (0.693 %/us and 0.264 %/us vs 0.080 %/us at v32). Our TINY (M<=4) band owns d1; alygg77's d1 is 24.6 us at 3.16x and vjeux's is 29.5 us at 2.73 TB/s. d1 has only **3 touched experts** and is not stream-bound at all.
4. **Decode-layer attention is the cheapest remaining per-us**: merged-heads FA3 (two 64-wide FP8 KV heads as one 128-wide, zero-padded q) is **1.2437 / 1.2720 / 1.2227 / 1.2871x** on d16f/d32f/v16f/v32f vs stock FA3, judged fragment **1.1401x geomean**, 24/24 + 720 checks PASS, and its author states decode_layer makes "the identical FA3 call on the same B16/B32 80K page-1 shapes" and **never measured it there**. Attention is 9.8 us of d32s / 14.8 of v32s in the rival budget, so even 1.25x only buys ~2-3 us on those two loads (**+0.06-0.11 % board**) - it is a small lever unless our attention share is larger than theirs.
5. **Do not re-open**: pooler/router pre-stream, side-stream splits, router-in-pooler, glue-launch skipping, >2 pre-wait stages, qkv stream-K, per-shape cuBLASLt sweeps beyond what we have, pooler cuBLASLt pins in-layer (-6..-8 us at v16/v32), Nathen's T=128 qkv block, dynamic MoE tail, 4th ring slot, expert byte compression, WGS32, FP32-router-as-separate-launch.

---

## The single most useful new fact

**The H200 read ceiling is 4.60 TB/s at 2 GiB and 4.33 TB/s at 256 MiB (NCU; 4.17 with CUDA events) - our "4.287 TB/s cold-read wall" is a size-specific point, and a single SM can pull 140-207 GB/s with a bare TMA ring, so our "26.9 GB/s/SM per-SM TMA fetch-engine cap" is an in-flight/schedule artifact, not hardware.** A rival MoE on the same board streams 3.87-4.05 TB/s in its stream phase (90-92 % of the size-matched ceiling, quiet stretches 4.39-4.64) while ours sits at 3.5-3.8, and their remaining 9-12 us at d32 is a *stall* budget (ramp 3-6, phase-A waves 3-6, shadows 4-12, end 7-11 us; a 16-stage unit end stalls a CTA 1-7 us), not a rate limit.
