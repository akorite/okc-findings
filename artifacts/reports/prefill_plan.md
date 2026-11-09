# `muse_prefill_layer` — the 7.2%, the one mechanism worth the most, and the shipped candidate

Read `artifacts/reports/prefill_budget.md` first: it has the per-shape byte/FLOP budget, the fitted
`router_load = 1.35·T`, and the measured per-component shares. Everything below is a consequence of
that table.

Baseline to beat on this board: **1.8963** (ours, job `30a94d86…`; the same bytes read 1.9075 in
`fc468d5e…`). #1 = **2.0325** → each of the three loads must come down **6.7–7.7%**.

## 1. Top mechanism — settle the MoE launch geometry under the graph (it is 38–40% of every load)

The grouped NVFP4 expert GEMM is **38–40% of every load** and is the only component more than 2× off
its own roofline: **63% of the BF16 peak at T=16384, 55% at T=3012, 79% of HBM at T=753** (budget
report §2). The measured loss is *per item*, not per byte:

```
T=16384:  2227 GFLOP in 3587.5 us  = 621 TF/s = 63% of 989 TF/s
queue   = tiles x NT = 96 x (64 g1 + 32 g2) = 9216 items over 132 CTAs (1 CTA/SM)
per item = 268 MFLOP (256 token rows x 128 channels x K=4096)  -> 35.8 us of wgmma time
measured = 51.4 us/item  ->  15.6 us/item (30%) is the K=4096 loop's per-stage barrier/issue cost,
not math.  The weight stream (2.65 GB = 3 reads/expert at ~691 routes/expert) is only 740 GB/s,
15% of HBM, so it is not the weight stream either.
```

The tile family is register-exhausted, not design-exhausted: 2 consumer warpgroups at
`setmaxnreg 232` = 59 k of the SM's 64 k registers, so a wider A or N tile (m256, or 4 warpgroups at
`acc[128]`) does not fit — that is why this is a **sweep of the incumbent's own unclaimed launch
geometries**, each a one-#define change, and not a rewrite:

| arm | one change | why it can pay | expected at T=16384 |
|---|---|---|---|
| **A: n=128 token tiles** | `plan_warp`: full tile `c>>8`→`c>>7`, rows `full*256`→`full*128`, 256-class → 128-class (10 lines, kernel already instantiates `consumer_tile<128>` and `kps_for(128)`) | halves the activation bytes staged per FLOP; costs 2× the expert-weight stream (5.3 GB, 1.1 ms of HBM that still overlaps a 2.6 ms compute floor) | **−400 to −900 µs** |
| **B: 2 CTAs/SM** | `SKINNY_TEST = True` (already written: `MOE_SMEM_SKINNY_TEST = 100 KiB`, grid `nsm*2`) | two independent stage streams per SM hide the per-stage barrier; halves `acc` per CTA | −200 to **+300** µs (sign unknown) |
| C: `MOE_KPS*` 2→4 | one constant | one producer issue per 2 k-blocks, unchanged accumulator | −100 to +100 µs |

Only T ≥ 8192 is affected by A (at T=3012 an expert already has 127 rows ⇒ a 128-row tile; at
T=753 it is 32 rows), so the blast radius of arm A is exactly the `first turn` load = 1/3 of the
score:

| shape | MoE now | arm A | Δ shape | Δ load |
|---|---|---|---|---|
| turn 1 / turn 4 (T=753, T=3012) | 239–751 µs | unchanged | 0 | 0 |
| first ctx0 full | 3587.5 µs | −400..−900 | −4.4..−9.9% | first turn **−3.1..−6.9%** |
| first ctx32K sliding (weight 138) | 3750.0 µs | −400..−900 | −4.5..−10% | (same load) |
| first ctx32K/65K full | 3421/3488.9 µs | −400..−900 | −3.7..−8.2% | (same load) |

Mid-range arm A alone: **≈ −3.5% whole-file → 1.96**; arm A + the §3 runner-ups: 2.02–2.06.

Why this is not the shipped candidate: its sign is not decidable off a GPU (arm B in particular), and
`bench_bundle` builds a new extension per arm, so settling it costs GPU minutes, not judge checks.
Each arm is guarded and validated by the harness's own correctness tests before any timing.

## 2. Shipped candidate — `prefill_gap1.py` (router GEMM out of split-K at T ≤ 4096)

`src/okc_loop/vendored_work/prefill/prefill_gap1.py` is `vendored/keell0renz/muse_prefill_layer.py`
plus **one** mechanism behind `ROUTER_MM_SMALL = False` (shipped path byte-identical when off):
at `T ≤ ROUTER_MM_MAX_T` the prepare-time tuner skips the router GEMM, so `_mm` uses `torch.mm`.

Evidence it is worth a check (CUPTI, job `30a94d86…`): the router is pinned to a 4-way split-K plan
(`LT_PIN[(753,"router")] = (33,4,2,3)`) that, under the captured graph, runs on the side stream
against the pooler GEMM and drags a **separate `cublasLt::splitKreduce_kernel` behind it** —
`nvjet_..._splitK` 13.1 µs + `splitKreduce` **63.2 µs** next to a 64.7 µs pooler at T=753, plus
`moe_route` 9.9 µs. The tuner that chose it times each candidate **alone and eagerly** (the exact
measurement `AGENTS.md` warns about: *"configurations that win eagerly can lose under graphs"*),
and `LT_PIN` bypasses the race entirely. `torch.mm` has no split-K plan and no reduction launch, so
8–20 µs of chain can hide in the pooler's shadow instead of extending the critical path.

| shape | chain cost now | expected | Δ shape | Δ load |
|---|---|---|---|---|
| turn 1 sliding (T=753) | ~20 µs (13.1 splitK + reduce + 5.2 route) | ~0 exposed | −2.5 to −3.7% | turn 1 **−2 to −3%** |
| turn 1 full (T=753) | same chain, partly overlapped | ~0 exposed | −1.5 to −2.5% | (same load) |
| turn 4 (T=3012) | split-K router under a 251 µs pooler | −15 to −30 µs | −1 to −2% | turn 4 **−1 to −2%** |
| first turn (T=16384) | unchanged (`T > ROUTER_MM_MAX_T`) | 0 | 0 | 0 |

Expected whole-file: **+1.5 to +3%** (1.8963 → 1.92–1.95). Honest label: this is the largest defect
a **host-only** change can reach; it is not the 7%.

### Exact local A/B (no judge quota)

```bash
# build both bundles (the tool only packages; the numbers come from the GPU box)
python tools/bench_bundle.py src/okc_loop/vendored_work/prefill/prefill_gap1.py muse_prefill_layer --out artifacts/bench/gap1
cp src/okc_loop/vendored_work/prefill/prefill_gap1.py src/okc_loop/vendored_work/prefill/prefill_gap1_on.py
# flip the one line in the copy:  ROUTER_MM_SMALL = False  ->  True   (line 163)
python tools/bench_bundle.py src/okc_loop/vendored_work/prefill/prefill_gap1_on.py muse_prefill_layer --out artifacts/bench/gap1

# on the H200, same ten minutes, interleaved (the harness flushes L2 per repetition):
python artifacts/bench/gap1/okc_bench_prefill_gap1.py      # arm A: flag off
python artifacts/bench/gap1/okc_bench_prefill_gap1_on.py   # arm B: flag on
python artifacts/bench/gap1/okc_bench_prefill_gap1_on.py
python artifacts/bench/gap1/okc_bench_prefill_gap1.py
```

Compare the per-shape µs for `batch=1 ctx=80000 q_len=753` (both layers) and `batch=4 … q_len=753`;
`first turn` rows must be *identical* (the flag does not touch them — that is the A/A control).

### Kill criterion

Kill if **any** of:
* the two `turn 1` rows do not both improve by ≥1.0% in the ABBA median (below the ±0.6% same-code
  spread of this board, a single row is not a result), **or**
* the `first turn` rows move by more than 0.3% (the flag must be inert there; a move means the
  router choice leaked into a shared buffer and the file is wrong), **or**
* `prepare()` fails to build or `LT unavailable` appears in the log (the `torch.mm` fallback broken).
Keep it only if the whole-file geomean improves ≥1.5%; then submit once, expect +1.5 to +3%, and
spend the remaining checks on §1.

## 3. Runner-ups, in expected-value order (all measured, all need a GPU A/B)

| # | change | evidence | expected per shape | load effect |
|---|---|---|---|---|
| 1 | **MoE arm A: n=128 token tiles** (§1) — the biggest single lever | 63% of peak; 15.6 µs of the 51.4 µs/item is per-stage cost | −400 to −900 µs at T=16384; 0 at T=3012 and T=753 | first turn −3 to −7% |
| 2 | dense pin sweep: `(16384,"qkv")` has no `LT_PIN` entry; `(3012,"unpooler"/"router")` have neither pin | the five GEMMs are at **81%** of the file's own floor at 16K, 80% at 3012, 75% at 753 | −230 µs @16K, −50 @3012, −25 @753 | −6%/−4%/−1% |
| 3 | boundary kernels: one CTA per row, **71–75% of HBM** | `boundary_kernel<512,…>` 460 µs + `<128,…>` 268 µs at 16K for ~1.9 GB | −145 µs @16K, −30 @3012 | −1.5% |
| 4 | glue (`moe_gather` 135 + `_combine` 128 + `gather_kp` + `qk_prep`) at **52% of HBM** | 319 µs at 16K for ~800 MB | −64 µs @16K | −0.7% |
| 5 | attention on full layers (48–62% of the FP8 peak = FA3's own band) | 2291 µs @ctx32K, 4016 @ctx65K | −5% is −115/−200 µs | −0.4% of a load |

Do **not** chase: the `o` GEMM at T=753/3012 (39/148 µs) is *tile-traffic*-bound, not kernel-bound
(216 MB of re-reads for a 12.6 GFLOP GEMM at a 112×128 tile); `LT_PIN2` already pins the algo-66
winner and the tuner already raced it. And **do not** replace the expert GEMM with
dequant-to-BF16 + DeepGEMM: that path breaks even only against a ≤460 TF/s expert GEMM
(`artifacts/reports/kernarc_digest.md` §(b), measured on the board), and this file already runs at
621 TF/s at T=16384 and 545 at T=3012. The N72 attention core and the `MOE_FAST_SCALE` dequant are
already at their measured optimum (`scale_pair` = 1 byte-permute + 1 IMAD per pair; 9 ALU ops per 8
weights).

## 4. Research: what the N72 PV layout is, and why 72

The incumbent's attention is ZhimaoL's `fmha7` + `gather_kp` (Apache-2.0, cited in the file header),
and the piece that is his is the **N72 PV layout**: the `P·V` product and the softmax row sum are
**one** tensor-core instruction,

```c
wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3   // _CUDA_SRC, lines 1514 and 5195
```

* **64** of the 72 N columns are V's head dim (`HEAD_DIM = 64`);
* the **extra 8** are an added N group fed a ones-vector that every stage materialises in shared
  memory ("each slot's 1 KB ones group (N72's 9th N group)", line 1759), so `Σ_j p_ij · 1` lands in
  the same accumulator as `Σ_j p_ij v_j`. The denominator is then kept in fp32 registers
  (`fold_l72` / `rescale_acc72`, lines 1534-1556) because the fp8 wgmma accumulator drops its low
  bits.
* **Why 72 and not 65:** wgmma's N is a multiple of 8 (`m64n8k16 … m64n256k16` are the only legal
  shapes; the file's own wrapper list at lines 3293-3342 is exactly 32/64/96/128/160/192/224/256).
  Sixty-five columns do not exist, so 64 + 8 is the shortest legal width that can also carry the
  denominator. Cost: 12.5% of the PV tensor-core time produces 8 identical copies of the row sum,
  which is still cheaper than the standard alternative — a separate reduction pass over the P tile
  every key block, as Hopper FA2/CUTLASS do
  ([CUTLASS Hopper FMHA](https://github.com/NVIDIA/cutlass/tree/main/examples/88_hopper_fmha),
  [Colfax FA2-on-Hopper case study](https://arxiv.org/html/2312.11918)).
* `gather_kp` exists because the paged KV cache has page_size 1 on the scored rows: it copies the
  attended range into the N-group-major V^T tile layout that N72 reads, which is why the V tile in
  SMEM is `72 × 128` (`ATTN_SMEM`, line 183).

Hopper FP8 FMHA state of the art is ~1.0–1.2 PFLOPS on H100/H200
([Colfax FP8 FA2](https://research.colfax-intl.com/wp-content/uploads/2024/03/FP8-FMHA-blog-031024.pdf),
[FA3](https://pytorch.org/blog/flashattention-3/)) = 50–60% of the 1979 TF/s FP8 peak; this kernel
measures 48–62%. There is no 2× sitting in the attention.

## 5. Order of operations

1. Run §2's A/B on the GPU box. If it clears the kill criterion, submit it (one check) — cheap
   insurance against the ±0.6% board spread.
2. Hand §1 to a GPU-capable session as a 3-arm sweep of `prefill_gap1.py`, arms A/B/C each behind
   its own define (default off), run through the same bundle A/B with `--nearby`: only the
   `first turn` rows can move, and they are 1/3 of the score with 5.5–6.9% of upside. Keep the arm
   that wins by >2% on the two 16K sliding/full rows; the harness's correctness tests run before
   any timing, so a wrong plan costs GPU minutes, not a judge check.
3. Only then the dense pin sweep (§3 #2) — the cheapest remaining µs and the only one that needs no
   new kernel.

Shipped candidate = `src/okc_loop/vendored_work/prefill/prefill_gap1.py`; lint (the only gate run
here): **3 errors / 59 warnings, identical to its base** (`okc_loop.cli lint`, only line numbers
shift by +12).
