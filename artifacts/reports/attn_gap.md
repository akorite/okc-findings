# attn_gap — where the 21% to the leader lives, and the one knob that can pay for part of it

Board row **2.6590** (`dual.py`, value 1415.9 us) against **#1 = 3.2221** (tjiang-fair).
Candidate: `src/okc_loop/vendored_work/attn/attn_gap1.py` (copy of `dual.py`, one mechanism, off by default).
Nothing here was run: no judge, no `colab`, no local nvcc. All numbers below are either from the judge profile
`09b60f3ae8bf4c2b` (`klr_attention.py`), from `artifacts/ledger.sqlite3`, or from published sources — each one is
labelled.

## 1. What the three files actually differ by

`diff` of `vendored/keell0renz/muse_attention_decode.py` (their board: 2.7168) against ours:

| file pair | hunks | substance |
|---|---|---|
| `keell0renz/muse_attention_decode.py` vs `attn/klr_attention.py` | **3, 24 lines** | `_workspace()` caches `n_rows` in `_WS_CAP` instead of reading `ws[0].numel()` (their file reads numel; ours was changed to satisfy the Tier-0 tensor-value-branch screen); `_zh_start_builds` launches the three JIT builds on threads instead of serially |
| `attn/klr_attention.py` vs `attn/dual.py` | 290 lines | import-time `_plan_sweep` + `_dual_sweep` and their pick tables; `_build(src=...)`; `_launch(_ret=...)`; the `OKC_FINWARM` / `OKC_FLATCOOP1C` gates |

So **their file has no mechanism ours lacks**: not one kernel, plan or tile change. The 2.7168-vs-2.6497 delta is
run-to-run variance, and the ledger bounds that variance: one sha (`4cfc5cca`) scored geomean **2.6667** (check) and
**2.6089** (submit) with values **1412.3 / 1443.6 us** — a 2.2% geomean / 2.2% value spread for identical bytes. The
threaded JIT builds are untimed (`prepare()` is outside the timed section), and the `_workspace` numel read is host
metadata. **The gap to #1 is not in keell0renz's file, and it is not in the plan or the finalize either** — the plan
sweep's medians are flat within ±8% on all twelve shapes (`attn_plan.md`), and the two finalize flags measured
`0/12 picks` (`attn_dual.md`).

### Per-shape, from the profile (`klr_attention.py`, `09b60f3ae8bf4c2b`)

`eff` = the harness's own `sol` column divided by the kernel's GPU time: how much of the shape's bound the kernel
reaches. `tiles/CTA` and `CTAs` are read off `_plan`+`_TUNED` at 132 SMs; one tile = 64 KV positions = 16 KB of K+V.

| case | ours us | sol us | eff | plan (splits/nc/coop) | CTAs | tiles/CTA | kernel |
|---|---|---|---|---|---|---|---|
| decode 1 sliding | 4.81 | 0.11 | **2.6%** | 32/1/0 | 32 | 1 | `attn_kernel<1,false,13>` 4.3 |
| decode 1 full | 10.66 | 4.27 | **46%** | 252/8/1 | 252 | 5 | `<1,false,11>` 9.2 |
| decode 16 sliding | 7.84 | 1.65 | **25%** | 16/1/0 | 256 | 2 | `<1,false,1>` 6.7 |
| decode 16 full | 86.78 | 63.18 | 74% | flat 264 | 264 | 76 | `<1,true,0>` 84.8 |
| decode 32 sliding | 10.04 | 3.29 | **37%** | 8/1/0 | 256 | 4 | `<1,false,1>` 8.8 |
| decode 32 full | 166.3 | 126.4 | 76% | flat 264 | 264 | 151 | `<1,true,0>` 165.8 |
| verify 1 sliding | 6.02 | 0.12 | **2.2%** | 33/1/1 | 33 | 1 | `vfgns::b1vp2::vp2_kernel` 5.4 |
| verify 1 full | 13.65 | 4.28 | **32%** | 252/8/1 | 252 | 5 | `<8,false,11>` 13.5 |
| verify 16 sliding | 11.64 | 1.86 | **17%** | 8/1/1 | 128 | 4 | `<8,false,3>` 10.8 |
| verify 16 full | 99.24 | 63.39 | 64% | flat 264 | 264 | 75 | `wst_ct14b<4,2,4,1,0,0>` 99.3 |
| verify 32 sliding | 14.58 | 3.72 | **27%** | 4/1/1 | 128 | 8 | `<8,false,3>` 14.0 |
| verify 32 full | 185.2 | 126.8 | 69% | flat 264 | 264 | 151 | `<8,true,0>` 184.6 |

Two families, cleanly separated:

* **the four flat cells** run 64–76% of their bound with 76–151 tiles per CTA — long runs, the producer/consumer
  pipeline streams, nothing left (`attn_cores.md`, three independent audits);
* **the eight `!FLAT` per-request cells** run 2–46% with 1–8 tiles per CTA. Every one of the six sliding cells is in
  this family, which is why the sliding weight carries the deficit.

## 2. Mechanism table — what each candidate mechanism is worth, per shape

The two papers below give the right frame: a decode kernel's bandwidth is set by how many bytes it keeps in flight
(Little's law), and a CTA that only has a couple of tiles never amortises its fill — it "pays a full DRAM round trip
before it streams". Our `!FLAT` pipeline keeps `(STAGES-1)` tiles = 32 KB in flight per CTA at 3 stages, and the
scored sliding shapes give a CTA 1–8 tiles to spend it on.

| mechanism | where it reaches | per-shape estimate (us, base → est) | why / why not |
|---|---|---|---|
| **deeper `!FLAT` pipeline** (this candidate: `OKC_STAGES` 3 → 4) | !FLAT with ≥3 tiles/CTA | dec32 9.84→7.6, v16 11.34→8.8, v32 14.58→10.0, d1f 10.96→8.7, v1f 13.88→11.1 | in-flight per CTA 32→48 KB, same grid, same math: the only clean 1.5x left. Flat cells unaffected (their `Cfg<L,true>` is untouched) |
| more CTAs per SM | all !FLAT | 0 | `MINB=2` + 48–107 KB of smem already cap it at 2/SM; 264-CTA grids are already 2/SM |
| fewer/more splits | all | 0 | swept: ±8% neighbourhood, shipped plan is the minimum (`attn_plan.md`) |
| coop flip / finalize flags | 3 !FLAT, 2 flat | 0 | measured 0/12 picks, ≤0.5 us (`attn_dual.md`) |
| persistent one-wave over all requests (the `wst_ct14b` / stream-K shape, applied to sliding) | 7 !FLAT | dec1 4.81→? , v1 6.02→?, dec16 7.84→? | the only mechanism that can reach the **1–2 tile** cells, and the one axis never tried: `_plan` never routes a windowed shape to `flat`, and the plan sweep never crosses the flat/per-request line. Needs a host segment plan for windows — a new code path, not a knob |
| batch the two KV heads differently / q-head split | sliding | 0 or negative | halves bytes per CTA without adding bytes in flight, or duplicates KV traffic |

### Why the 1–2-tile cells are the hard ones

`bytes-in-flight/SM` explains the ordering at the small end but *not* dec16 sliding: at 2 tiles/CTA and 256 CTAs it
already holds ~62 KB/SM in flight (the level at which dec32 full reaches 76%) and still measures 25%. What it lacks is
**runway**: 2 tiles per CTA means the pipeline's fill is the kernel. Same for decode 1 and verify 1 (one tile each,
32–33 CTAs = a quarter of the machine). Those three shapes are fill-bound, and the published answer for a fill-bound
decode is a persistent grid with long contiguous runs (one wave, equal shares per CTA), which is exactly what
`wst_ct14b` does for the full verify cells — where it beats our flat kernel (99.3 vs 108 us) at B16.

## 3. Outward: what the literature says closes this

* **Little's law, and the size of the regime.** *Concurrency Response of Plain Global Loads on the NVIDIA H100*
  (arXiv 2608.15764): the bandwidth a memory-bound kernel sustains "is set by how many bytes it keeps in flight";
  the paper measures the crossover scale of the load mechanisms in **bytes in flight per SM**: parity only at
  ≥ 32 KiB, still 2x apart at 16 KiB, 2–3x apart at 0.5–8 KiB
  ([arXiv 2610.34907](https://arxiv.org/html/2610.34907v1)). Our sliding cells sit at 3.9 KB/SM (decode 1: 32 CTAs x
  16 KB over 132 SMs) to 62 KB/SM (decode 16) — i.e. at or below the knee.
* **Striped page lookup, producer one tile ahead, persistent one-wave grid.** *Paged Flash-Decoding at Memory Speed*
  (Heidari et al.) reaches **86% of H100 peak with FP16 / 83% with FP8** KV using: a page translation fused into each
  16-byte `cp.async`, one page-table read per 16 bytes rather than per element, a producer that "runs a full tile
  ahead of the math, so its latency overlaps with the loads already in flight", and a **persistent stream-K grid of
  exactly one wave** — after which "bandwidth swung by up to 30 points between shapes" stopped happening. Two of
  their findings bound what we can win: (a) *"deeper pipelines (3–4 stages) were within noise or worse"* on their
  long-context shapes — the same conclusion our flat cells already encode (L==8 flat runs 4 stages, !FLAT 3);
  (b) **at page size 1, bandwidth falls to ~54% of peak even in their kernel** — the scored `page_size=1` costs
  everyone, so the leader's 21% is not coming from a page-size trick.
* **What `wst_ct14b` does differently** (ZhimaoL, sglang `okc_patch/_cuda_src_wst.py`, vendored here): a
  **cooperative launch** (`cudaLaunchAttributeCooperative`, co-residency guaranteed) of a fixed **`G` = one-wave
  grid** whose CTAs are cut **balanced over the batch** — every CTA gets an equal, contiguous share of the
  (request, q-block, key-range) work instead of one split of one request; the merge is **in-kernel** through a
  workspace row plus a cooperative barrier, so there is no second kernel and no last-CTA serial tail; and the
  q-block is the unit (`q_len 8 x d 64` per head-pair), so one CTA folds 8 query rows over its whole key range with
  the m64n64k32 e4m3 wgmma pair. On `muse_attention_decode` that is 99.3 us at verify 16 full against 108 us for our
  flat partition (`attn_dual.md`), and it is why the file already routes B16 there. Its structure is the published
  stream-K answer to the same problem our sliding cells have; it is *not* applied to sliding (`ZH_VFG_SLIDING = False`,
  and `_plan` never emits a flat windowed plan).
* **Gated attention**: the gate is a per-(batch, q, head) scalar folded into the epilogue (`sigmoid(gate)` times the
  normalised row), so it is not a source of skips — no gate sparsity to exploit.

Conclusion of the survey: nobody publishes a 21% pointer for *our* shapes, because the sliding shapes are
fill/lookahead bound, not algorithm bound. The available levers are (i) lookahead depth at fixed grid — cheap, this
candidate — and (ii) the persistent one-wave partition for windowed shapes — the same idea one level up, a new host
path.

## 4. The candidate: `attn/attn_gap1.py`

One mechanism, behind a define that defaults to today's behaviour:

```
#define OKC_STAGES 0       // !FLAT cp.async stages (0 = 3); the variant build sets 4
static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : (OKC_STAGES ? OKC_STAGES : 3);
```

* `OKC_STAGES 0` (shipped default) reproduces `dual.py` byte for byte in behaviour: `!FLAT` 3 stages, flat 3 / 4.
* The variant build is this file's **own source** with that one line patched to 4 (`_dual_var()`, one asserted
  occurrence — a `-D` would be overridden by the later `#define`). Tag `_st4`, its own extension dir, +1 nvcc TU
  (71.7 s measured for a TU of this file on the gate box; the dual check took 472 s of the 900 s budget).
* The two nvcc builds are A/B'd **per scored shape at import** (the existing `_dual_sweep` hook, before the harness
  warms and captures a case), through the real `_launcher`, on interleaved cold-L2 CUDA-graph replays: keep the
  variant only if it wins `max(3%, 2 us)` on the median **and** ≥ 75% of 8 rounds. `_DUAL_PICK` is empty for every
  shape that does not earn it, so all correctness shapes and all four flat cells run the shipped build.
* What 3 → 4 stages means physically: prologue issues 3 tiles instead of 2 and the steady state holds 3 tiles
  (48 KB) in flight instead of 2 (32 KB), grid and split count unchanged. Same tiles, same boundaries, same merge
  order, so the output must be **bit-identical**; the sweep enforces that with
  `torch.equal(alt.view(torch.int16), ref.view(torch.int16))` (bit-pattern compare, NaN-safe — `dual.py`'s rel-L2
  guard returned `nan` on the sweep's synthetic case and refused every pick, which is why its flags could never
  have landed).
* Occupancy is preserved: L==1 `!FLAT` smem ~73 KB, L==8 ~107 KB, both under the 114 KB that would drop the CTA to
  1/SM, so the coop grids (v16/v32 sliding, 128 CTAs) stay resident and their finalize path does not change. If a
  grid still lost residency the launcher already clears `coop` and re-measures, so it fails closed.

### Expected per-shape delta

Assumes the streaming part of each cell scales with bytes in flight (1.5x) and the fill/merge part does not;
fill share 30% at 4 tiles/CTA, 20% at 8, 25% at 5.

| shape | base us | est us | Δ | reaches the score as |
|---|---|---|---|---|
| decode 32 sliding | 10.04 | 7.6 | −24% | 1.31x |
| verify 16 sliding | 11.64 | 8.8 | −24% | 1.32x |
| verify 32 sliding | 14.58 | 10.0 | −31% | 1.46x |
| decode 1 full | 10.66 | 8.7 | −18% | 1.22x |
| verify 1 full | 13.65 | 11.1 | −19% | 1.23x |
| decode 16 sliding (2 tiles/CTA) | 7.84 | 7.8 | ~0 | 1.00x |
| decode 1 / verify 1 sliding (1 tile/CTA) | 4.81 / 6.02 | same | 0 | 1.00x |
| the four flat cells | — | same | 0 | 1.00x (SASS identical) |

Geomean over 12: **+12%** (2.66 → 2.97); weighting the cells 46/16 as the ledger does: **+14%** (→ 3.02). That is
**55–60% of the 21% gap**, and it stacks with nothing else currently pending on this board.

## 5. Local A/B

```
# both bundles on the same H200 box, back to back, same ten minutes
python tools/bench_bundle.py src/okc_loop/vendored_work/attn/dual.py     muse_attention_decode --out artifacts/bench/gap0
python tools/bench_bundle.py src/okc_loop/vendored_work/attn/attn_gap1.py muse_attention_decode --out artifacts/bench/gap1
python artifacts/bench/gap0/okc_bench_muse_attention_decode.py     # base
python artifacts/bench/gap1/okc_bench_muse_attention_decode.py     # candidate: expect the [dual] lines per shape
```

The candidate's own `[dual]` lines (`base=.. var=.. pick=.. wins=../8 bits=same`) are the direct read: `pick=var`
on the five estimated shapes with `bits=same` is the win; `bits=DIFF` anywhere means the mechanism touched the math
and must not ship.

## 6. Kill criteria

1. **Build**: `[dual] variant build FAILED` → the `STAGES=4` instantiation does not compile (unverified here: no
   nvcc on this box, and `tools/cu_build.py` compiles the template body only). Kill the candidate, keep the report.
2. **Pick**: `0/12 shapes on the variant build` → worth 0, same verdict as `dual.py`. Do not submit. If instead the
   medians show `var` faster by 5–15% but under the 2 us floor on the small cells, the floor is the blocker, not the
   mechanism: that is the signal to go to `OKC_STAGES 5` for L==1 only (smem 88 KB, still 2 CTAs/SM) or to drop the
   absolute floor to 1 us for this knob.
3. **Score**: any check-level geomean below the base row (2.6590) or value above 1415.9 us → kill, even if some
   shapes improved: the picks are per shape, so a regression means the sweep's gate is mis-calibrated.
4. **Correctness**: `bits=DIFF` on any shape, or fewer than 16/16 harness tests → kill.

## Gates run here

* `PYTHONPATH=src .venv/bin/python -m okc_loop.cli lint src/okc_loop/vendored_work/attn/attn_gap1.py` → **PASS
  1/1 clean**, zero new findings against `dual.py` (which is PASS).
* `ast.parse` ok; diff against `dual.py` is 47 insertions / 43 deletions, of which the CUDA source change is the one
  `#define` plus one `constexpr` line.
* Not run, by design: judge, `colab`, `tools/cu_build.py` (no nvcc here). The `STAGES=4` instantiation is proven by
  the H200 check, not locally.

## Next lever if this one lands (not in this candidate)

Route the **windowed** shapes through the flat/persistent partition — one wave, equal contiguous runs, the
`wst_ct14b` structure — which is the only mechanism that can move the three fill-bound cells (4.81 / 6.02 / 7.84 us,
2.2–25% of bound). It needs a host segment plan for `window != 0` (`_plan`/`_launch` build `cum`/`seg` from
`seq_lens` only today), so it is a new code path, and the plan sweep has never crossed that line.
