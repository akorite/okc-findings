# `muse_prefill_layer` — bytes/FLOPs budget and where the 7.2% is

Board: ours **1.8963** (#91/729), #1 **2.0325** (zzz0906). Gap **7.19%** per load
(geometric mean over the three loads, so *every* load must come down ~7.2%).

Sources, all on disk, no re-runs:

| what | where |
|---|---|
| judge's floor model | `problems/muse_prefill_layer/reference.py:681-697` (`speed_of_light_ns`) |
| constants | `problems/muse_prefill_layer/reference.py:29-58, 678` |
| scored rows + weights | `problems/muse_prefill_layer/TASK.md:37-44`, `src/okc_loop/cases.py:213-230` |
| measured per-shape µs | `src/okc_loop/vendored/keell0renz/profiles/muse_prefill_layer.30a94d861e9f4851.txt` (job `30a94d86…`, score 1.8963) |
| per-kernel CUPTI report | same file, `profile 0..7` (+ `…fc468d5e….txt`, score 1.9075) |
| incumbent | `src/okc_loop/vendored/keell0renz/muse_prefill_layer.py` |

The two profile jobs are the same bytes measured 40 min apart: **1.8963 / 1.9075**, so the
run-to-run spread of the whole file is ~0.6% and any candidate below ~2% needs ≥2 checks.

## 0. Constants that decide everything

`HIDDEN 8192 · Q_SIZE 1024 (HQ 16 × HD 64) · KV_SIZE 128 (HKV 2) · QKV_SIZE 2304 · D_EXPERT 4096 ·
N_EXPERTS 256 · TOP_K 8 · LOCAL_EXPERTS 32 · BLOCK 16 · SLIDING_WINDOW 2048 · ROUTING_SHARE 0.75 ·
UNPOOL_SLICE 1024 · SLICED_MAX_TOKENS 8`

Hardware the floor assumes (H200 at spec): **4.8 TB/s HBM, 989 TFLOP/s BF16 dense.**

## 1. The floor is a pure BF16-dense model, and it is *math*-bound on all 8 shapes

`reference.py:681-697` counts, per call:

```
dense     = 2304·8192 + 8192·1024 + 4096·8192 + unpooled·4096      = 94.37 MFLOP/token (unpooled = 8192 for T>8)
bytes     = 2·dense + 4·256·8192 + touched·3·4096²·(0.5+1/16) + kv_tokens·128·2 + T·8192·12
flops     = 2·T·(dense + 256·8192) + 2·local_routes·3·4096² + 4·kv_tokens·q_len·1024
kv_tokens = B · (min(ctx,2048) if sliding else ctx) · shortest_share(spread)     # spread="fixed" on the scored rows → 1.0
local_routes = max(T, 0.75·router_load)     touched = max(6, 0.75·router_distinct)
```

`router_load`/`router_distinct` are not in `TASK.md`. **Fitting the 8 published floors gives
`router_load = 1.35·T` routes on the busiest GPU and `touched = 24..32` experts** (1.0·T
under-predicts every floor by 10%, 1.35·T reproduces all 8 to ≤1.4%). That 1.35 is the "dist soft"
routing skew: each GPU takes ~1/8 of the top-8s = T routes on average, but the busiest 32-expert
range takes `1.35·T`.

| load | shape (B, ctx, layer, L) | w | judge floor µs | bytes MB | bytes µs | FLOP G | math µs | bound | ours µs | ours/floor |
|---|---|---|---|---|---|---|---|---|---|---|
| turn 1 | 1, 80000, sliding, 753 | 46 | **256.95** | 1178 | 245 | 254 | **257** | math | 537.19 | 2.09× |
| turn 1 | 1, 80000, full, 753 | 16 | **496.69** | 1198 | 250 | 494 | **500** | math | 769.51 | 1.55× |
| turn 4 | 4, 80000, sliding, 753 | 46 | **1036.28** | 1401 | 292 | 1016 | **1027** | math | 1758.84 | 1.70× |
| turn 4 | 4, 80000, full, 753 | 16 | **1992.42** | 1481 | 309 | 1977 | **1999** | math | 2589.92 | 1.30× |
| first | 1, 0, full, 16384 | 16 | **5518.43** | 2714 | 565 | 5388 | **5448** | math | 9101.13 | 1.65× |
| first | 1, 32768, sliding, 16384 | 138 | **5663.35** | 2714 | 565 | 5525 | **5586** | math | 8911.53 | 1.57× |
| first | 1, 32768, full, 16384 | 16 | **7710.00** | 2722 | 567 | 7587 | **7671** | math | 10866.07 | 1.41× |
| first | 1, 65536, full, 16384 | 16 | **9847.68** | 2730 | 569 | 9786 | **9894** | math | 12729.20 | 1.29× |

(bytes at `touched = 32` experts, the upper bound: every local expert is touched at T ≥ 753, and the
judge's own floor cannot pin `touched` because the math term binds on all eight rows; the model
reproduces all eight published floors to −1.3%..+1.3%)

**No scored shape is bandwidth-bound** — but `turn 1` is only 5% away from it (245 vs 257 µs), and
at T=16384 bytes are 10× below math. So the large shapes must come out of tensor-core time (dense
GEMMs, the grouped NVFP4 expert GEMM, the FP8 attention); at T=753 there is no bandwidth headroom
either, which is why that row is the hardest to move (below).

The floor splits into three terms that sum to the judge's number:

| shape | dense+router | MoE (`2·1.35T·3·4096²`) | attention (`4·kv·L·1024`) |
|---|---|---|---|
| turn 1 sliding | 147.0 µs (57%) | 103.5 µs (40%) | 6.5 µs (3%) |
| turn 1 full | 147.0 (29%) | 103.5 (21%) | 249.5 (50%) |
| turn 4 sliding | 587.6 (57%) | 414.0 (40%) | 25.5 (2%) |
| turn 4 full | 587.6 (29%) | 414.0 (21%) | 998 (50%) |
| first ctx0 | 3196 (59%) | 2252 (41%) | **0** |
| first ctx32K sliding | 3196 (52%) | 2252 (37%) | 139 (2%) |
| first ctx32K full | 3196 (39%) | 2252 (27%) | 2224 (27%) |
| first ctx65K full | 3196 (31%) | 2252 (22%) | 4447 (43%) |

Note the ctx0 row: the judge's attention term is **zero** there (no cached tokens), so the 505 µs
of causal-in-block attention that the incumbent pays is entirely *above the floor*. A submission is
not required to be near the floor; it is scored against SGLang.

## 2. Where the measured time actually goes

Per-kernel CUPTI, job `30a94d86…` (µs; share of the kernel-sum; the `… N more` residual is 1–3%).
`dense` = the five cuBLASLt/nvjet GEMMs (qkv, o, pooler, unpooler, router-hilo); `glue` = KV gather,
`moe_route`, `moe_gather`, `_combine_kernel`, `qk_prep`, hidden rows.

| shape | total | MoE | dense | attention | boundary | glue |
|---|---|---|---|---|---|---|
| turn 1 sliding | 537.2 | 239.3 (45%) | 201.2 (37%) | 13.5 (2.5%) | 39.4 (7%) | 38.9 (7%) |
| turn 1 full | 769.5 | 241.9 (31%) | 275.0 (36%) | 236.3 (31%) | 39.7 (5%) | 47.2 (6%) |
| turn 4 sliding | 1758.8 | 751.4 (43%) | 754.8 (43%) | 42.7 (2%) | 151.8 (9%) | 75.3 (4%) |
| turn 4 full | 2589.9 | 732.9 (28%) | 751.0 (29%) | 840.6 (32%) | 150.3 (6%) | 140.7 (5%) |
| first ctx0 | 9101.1 | 3587.5 (39%) | 4010.6 (44%) | 505.5 (6%) | 726.0 (8%) | 319.9 (4%) |
| first ctx32K sliding | 8911.5 | 3750.0 (42%) | 4011.7 (45%) | 195.6 (2%) | 728.4 (8%) | 328.8 (4%) |
| first ctx32K full | 10866.1 | 3421.0 (31%) | 4081.6 (38%) | 2290.6 (21%) | 728.1 (7%) | 355.6 (3%) |
| first ctx65K full | 12729.2 | 3488.9 (27%) | 4085.0 (32%) | 4016.0 (32%) | 728.2 (6%) | 352.5 (3%) |

### Efficiency against each component's own floor

Floors here use the *file's* arithmetic (the router is one `[512,8192]` hi/lo GEMM, i.e. 2× the
judge's router term; `2·T·98.57 MFLOP` per token):

| component | T=753 floor | obs | eff | T=3012 floor | obs | eff | T=16384 floor | obs | eff |
|---|---|---|---|---|---|---|---|---|---|
| **MoE** (math at 1.35·T routes) | 103.5 | 239.3 | **43%** | 414.0 | 751.4 | **55%** | 2252 | 3587.5 | **63%** |
| **MoE** (weight bytes: 906 MB / 4.8 TB/s) | 188.8 | 239.3 | **79%** | 188.8 | 751.4 | — | 188.8 | 3587.5 | — |
| **dense+router** | 150.1 | 201.2 | **75%** | 600.4 | 754.8 | **80%** | 3265 | 4010.6 | **81%** |
| **attention** (judge count) | 6.5–249.5 | 13.5–236.3 | 48–105% | 25.5–998 | 42.7–840.6 | 60–119% | 0–4447 | 195.6–4016 | 71–111% |
| **boundary** (stream bytes) | 28.5 | 39.4 | **72%** | 114 | 151.8 | **75%** | 517 | 726.0 | **71%** |
| **glue** (~800 MB at 16K) | 8 | 38.9 | 21% | 45 | 75.3 | 60% | 167 | 319.9 | 52% |

The attention ratios above 100% are the causal discount (the judge counts `L·kv` pairs; a causal
block only visits `L·ctx + L²/2`), and the real rate is ~1.0–1.3× the judge's number, i.e.
**48–62% of the 1979 TFLOP/s FP8 peak** — the same band Colfax/FA3 report for Hopper FP8 FMHA
([Colfax FP8 FA2, ~1 PFLOPS](https://research.colfax-intl.com/wp-content/uploads/2024/03/FP8-FMHA-blog-031024.pdf),
[FA3, ~1.2 PFLOPS](https://pytorch.org/blog/flashattention-3/)). So attention is *normal*, not broken.

Three things follow:

1. **The 16K shapes do 3× the weight traffic the floor assumes.** `touched·28.31 MB = 906 MB` is
   read once in the floor; the kernel reads an expert's weights once **per token tile**
   (`row0`-tiles of 256 rows ⇒ ~691 routes/expert ⇒ 3 tiles at T=16384), so the real weight stream
   is ~2.65 GB. It is *not* binding (740 GB/s of 4.8 TB/s, 15%) but it is 25 ns of HBM time per
   route against a 102 ns compute floor.
2. **The MoE is the only component with a >2× gap to its own roofline** (63% at 16K, 55% at 3012).
3. The dense path (75–81%) and the boundary (71–75%) are ordinary for cuBLASLt and a one-CTA-per-row
   streaming kernel; there is ~600 µs at 16K and ~50 µs at 753 of theoretical slack in each, but no
   gross defect.

## 3. Which load carries what (weights from `cases.py`)

Load time = Σ (shape µs × weight); the score is the geometric mean of `SGLang/ours` per load, so a
component's effect on the score is its **share inside each load**, not its absolute µs.

| load | total µs | MoE | dense | attention | boundary | glue |
|---|---|---|---|---|---|---|
| turn 1 (`46×slide + 16×full`) | 37,023 | **40.2%** | 36.9% | 11.9% | 6.6% | 6.9% |
| turn 4 (`46×slide + 16×full`) | 122,343 | **37.8%** | 38.2% | 12.6% | 7.7% | 4.7% |
| first turn (`16+138+16+16`) | 1,752,930 | **39.1%** | 42.7% | 7.8% | 7.7% | 3.5% |

The `first turn` load is 70% the **sliding** shape (weight 138 = 3 chunks × 46 sliding layers), and
that shape has essentially no attention cost (196 µs, 2%). The scored workload is therefore
**69% sliding layers** (138+46+46 = 230 of 334 layer-runs), where the layer is
`MoE 42% + dense 45% + boundary 8% + attention 2%`.

## 4. What it takes to close 7.19%

Required: each load's time × 0.9281. With the shares above, one consistent split of the target is

| component | needed | what that means per shape |
|---|---|---|
| MoE | **−10%** | 239→215 (T=753, already 79% of HBM), 751→676 (T=3012), 3587→3229 (T=16384) |
| dense GEMMs | −5% | 201→191, 755→717, 4011→3810 (81%→86% of peak) |
| boundary | −20% | 39→32, 152→122, 726→581 |
| attention (full layers only) | −10% | 236→213, 841→757, 2290→2062, 4016→3614 |
| glue | −20% | 39→31, 75→60, 320→256 |

Cheaper alternatives that reach the same number: **MoE −18% alone** (⇒ 60–63% → 74–77% of the
roofline), or MoE −12% + dense −7%, or attention −25% on full layers alone (impossible: that is
above the FP8 state of the art). **No single host-level knob in the incumbent is worth 7%.**

## 5. What the baseline is, and why the remaining 7% is not "beat one SGLang kernel"

`problems/muse_prefill_layer/baseline.py` is SGLang at `a55ab7d5…` (`MuseSpark1xDecoderLayer`), with
these kernel choices on the scored path:

* attention: **FA3 `flash_attn_with_kvcache`**, paged, causal, window (2047,0) on sliding layers;
* experts: `moe_align_block_size` (block-size heuristic `BLOCK_SIZE_M_CANDIDATES`) then **two
  `moe_wna16_marlin_gemm` (W4A16)**, `silu_and_mul`, and a Triton post-expert norm weighted sum;
* the router above 8 tokens as the BF16 hi/lo pair, everything else `bf16_gemm_dispatch` (cuBLAS).

Measured SGLang per shape (§1 table, `sglang` column of the profile) vs ours:

| shape | SGLang µs | ours µs | ratio | our floor µs |
|---|---|---|---|---|
| turn 1 sliding | 1026.45 | 537.19 | 1.91× | 257 |
| turn 1 full | 1415.81 | 769.51 | 1.84× | 497 |
| turn 4 sliding | 3299.65 | 1758.84 | 1.88× | 1036 |
| turn 4 full | 4841.20 | 2589.92 | 1.87× | 1992 |
| first ctx0 | 17364.03 | 9101.13 | 1.91× | 5518 |
| first ctx32 sliding | 17191.49 | 8911.53 | 1.93× | 5663 |
| first ctx32 full | 20884.51 | 10866.07 | 1.92× | 7710 |
| first ctx65 full | 24741.97 | 12729.20 | 1.94× | 9848 |

The ratio is flat (1.84–1.94×) because both sides pay the same *floor* work: the dense projection
FLOPs and the expert GEMM are the same on both, and the incumbent already replaced SGLang's padded
Marlin path with a persistent grouped kernel, FA3 with ZhimaoL's N72 fmha7, and cuBLAS's dispatch
with pinned cuBLASLt signatures (`LT_PIN`, 7 of them) and algo-66 tournament winners (`LT_PIN2`). Beating SGLang by another 7% therefore means beating **our own
kernels' distance to the floor**, and the ordered list is: the MoE plan/tile (63–79% of roofline,
38–40% of every load), the five dense GEMMs' pins (75–81%), the boundary kernels (71–75%), the glue
kernels (52%), and only then the PV layout / attention epilogue — which is already at the published
Hopper FP8 band (§4 of `prefill_plan.md`).
