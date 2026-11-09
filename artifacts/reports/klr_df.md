# DF (data-flag A→B hand-off): can it extend to the b4 (T=17..32) and main (T≥33) builds?

Verdict: **no — the extension is a no-op at minimum and wrong at worst.** All eight textual anchors
exist in `MOE_CUDA_SRC`, the patch applies cleanly to it, and the result compiles for the WGS define
sets — but under `-DOKC_WGS=1` the patched code is not in the translation unit at all, and the two
edits that *do* survive are one out-of-bounds pointer shift and one silently-wrong sentinel scope.
Deliverable: `src/okc_loop/vendored_work/layer/klr_df.py` (default-off, `DF_MAIN = False`); the base
file is untouched (999,964 B).

## 1. What `_df_patch` rewrites

`klr_merged3.py:12884-12958` applies six `str.replace`s (each asserted to hit exactly once) to
`_MA_ALT_SRCS["moe_helper0"]`, producing `_MA_ALT_SRCS["moe_helper0df"]`:

| # | anchor in the CUDA source | edit |
|---|---|---|
| 1 | `#define OKC_NOLOAD 0` | adds `#define OKC_DF OKC_PRMTSC` and `INTER_SENT_ROWS 128` |
| 2 | `constexpr int X1P = 512 + 8;` | inserts `df_bad()` (bf16 `0xFFFF` test on a `uint4`) and `load_x_inter_df()` |
| 3 | `if (PB) { while (ld_acquire(p.doneA + j) < RB_A) … } load_x_regs<PB>(…)` inside `skinny_ring_ws` (segment `skinny_ring_ws` → `ws_helper`, unique there) | `if (PB && OKC_DF) load_x_inter_df(...) else {original}` |
| 4 | `if (q == nA - 1) {` (phase-A credit `atomicAdd(p.doneA + jj, b - a)`) | `if (!OKC_DF && q == nA - 1)` — the credit is skipped entirely |
| 5 | `// final: out[t] = …` | `#if !OKC_WGS` block that re-writes the sentinel over `min(128, p.M * 8)` inter rows at the end of the launch |
| 6 | `p.inter = (__nv_bfloat16*)scratch[2].data_ptr();` | `p.inter += 128 * 4096` (host code) |

**The barrier it removes**: the job-granular phase-A→B barrier `while (ld_acquire(p.doneA + j) < RB_A)`
plus the `p.doneA` crediting that feeds it. DF replaces "wait until every phase-A unit of job *j* is
done" with "load job *j*'s `inter` rows and spin per 16-byte chunk until no word is the bf16-NaN
sentinel `0xFFFF`" — i.e. **row-granular** (ID = 4096 columns), not per-token, not per-column-block.
The sentinel is self-perpetuating: the next launch starts poisoned because the previous launch's
phase-B kernel re-poisons the rows before the final epilogue.

**Why MID-only**: after the E173 measurement the patch was scoped to the band where a *grid barrier*
was still the alternative. In-repo trace: `docs/PROGRESS.md:461` — "data-flag A→B beats grid barrier
on small batches" (the `harness/make_dl_df.py` / "moe-small E173" record itself is not vendored; that
one line plus the header comment is all that survives). MID is also the only band where the poison
region is exactly the used region (§5).

## 2. Do the anchors exist in the main/b4 source? Yes — and the patch lands

`MOE_CUDA_SRC` (153,703 B) is a different source from `moe_helper0` (138,010 B), but every anchor is
present with the same multiplicity (1 each; #3 is 3× file-wide but 1× inside the
`skinny_ring_ws`…`ws_helper` segment, which is all `_df_patch` asserts). `_df_patch(MOE_CUDA_SRC)`
succeeds, +1,977 B, and produces a source that compiles for `MOE_DEFINES` and for
`MOE_DEFINES + -DOKC_B4FIN=1` (gate table, §4). `skinny_ring_ws` sits at `MOE_CUDA_SRC:1445`,
`ws_helper` at 1553 — same order as in the MID source, so the segment slice is non-empty.

## 3. Why it still cannot work there — three independent blockers

**(a) Dead code.** `skinny_ring_ws` is inside `#if OKC_WS` and has exactly one call site,
`run_phase` (`MOE_CUDA_SRC:2352`), which is called only from the `#else` of `#if OKC_WGS`
(`:2584-2589`). `MOE_DEFINES` is `-DOKC_WGS=1` with no `-DOKC_WS`, and b4 is `MOE_DEFINES +
-DOKC_B4FIN=1`. Preprocessed with those defines, `void skinny_ring_ws` and `run_phase<` occur **0
times**, so `load_x_inter_df(...)` occurs 0 times; with MID defines, 1. Mechanical check, not a
reading: `klr_df.py`'s `__main__` runs `cpp -P` and asserts exactly that (MAIN: 0, MID: 1).

**(b) The WGS builds have no grid barrier left to beat, and their existing hand-off is finer.**
`wgs_xproducer` (`:1880`) already waits per *(job, phase-B stage)*:
`while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8)` — 256 columns (one of 16 stages), counted by the
8 phase-A warpgroup-epilogues that wrote exactly those columns, and pipelined by the `WGS_NX = 3` x
ring so only the first stage can expose. That is `ID/16` **coarser-to-finer than DF**: DF waits for a
whole 4096-column row, WGS waits for the 256-column slice it is about to consume. DF at T ≥ 17 would
not add a data flag, it would replace a 16× finer one with a coarser one. This is the same point
`artifacts/reports/klr_unitend_fences.md` §2 makes from the other end ("coarsening it to per-job =
serialising A→B").

**(c) The one edit outside every `#if` is an out-of-bounds write.** Edit #6 is host code and is
guarded only by `#if OKC_DF`, i.e. `OKC_DF = OKC_PRMTSC = 1` in the WGS builds too. It shifts
`p.inter` by 128 rows, and `:2934` computes `p.xA = p.inter + EMAX * HD` from it, so `xA`/`xB` shift
too. The host allocates exactly `(EMAX + 2 * WGS_XROWS) * 4096` bf16
(`klr_merged3.py:10341`; `WGS_XROWS = 3072 + 1088`), i.e. 11,392 rows — the shift pushes the last
`xB` row 128 rows past the end: a 1 MiB write beyond the buffer and past `SCRATCH_GUARD` (64 Ki
words). Nothing else in the DF payload is reachable, so this shift is the *entire* effect DF would
have on a b4/main launch.

## 4. Gates

| gate | result |
|---|---|
| `ast.parse(klr_df.py)` | ok |
| `ruff check klr_df.py` | clean; `klr_merged3.py` unchanged at its 218 pre-existing errors |
| `klr_df.py` self-check (`cpp -P` reachability) | `MAIN: DF load sites = 0`, `MID: = 1` → passes |
| `cu_build.py` patched vs base, `--variant MAIN` (`-DOKC_WGS=1 -DOKC_PRMTSC=1`) | both `ok`; PTX byte-identical (sha `622c5b43`), as expected for dead code. Inputs regenerate in one line: a candidate `MOE_CUDA_SRC = klr_df.df_src()` (patched) or `= klr_merged3.MOE_CUDA_SRC` (base) plus the two define tuples |
| `cu_build.py` patched vs base, `--variant MID` (`-DOKC_WS=1 -DOKC_PRMTSC=1`) | both `ok`; PTX byte-identical (`29684a78`) — the MoE `__global__` is a template, so nvrtc emits an empty module (`cu_build`'s documented limit): this proves the DF payload *parses* under both define sets, nothing more |
| `colab run --gpu A100` sm_90a compile gate, candidate = layer file + `_MA_ALT_SRCS["moe_dfmain"] = _df_patch(MOE_CUDA_SRC)` + `@moe_dfmain` on `MOE_DEFINES` (`artifacts/bench/gate_df.py`) | `bad: [tvm_ffi ×3]` only (expected: those three modules cannot be gated on Colab); `_moe` ok 88.6 s, `_moe(32)` ok, `_moe_mid` ok, `_moe_tiny`/`_moe_tiny2`/`_att7`/`bfp__get_ext` ok, wall 88.8 s |
| same gate with `OKC_DF` **on** (`_MA_SFX = ("-DOKC_PRMTSC=1",)`, `artifacts/bench/gate_df_prmt.py`, narrowed to `['_moe', '_moe_mid']`) | `bad: []`, `_moe` ok 87.9 s, `_moe_mid` ok 63.6 s, `_moe(32)` ok, wall 87.9 s — i.e. `OKC_DF=1` compiles for `-DOKC_WGS=1` under the judge's nvcc/torch 2.11+cu130. Two earlier attempts on this file returned `TimeoutError` from the client at ~200 s: `colab run`'s default cell timeout is 30 s, so it needs `--timeout 900` (harmless, the build never exceeded 88 s) |

What the compile gates can and cannot say: they settle the ticket's stated risk ("the DF source has
never been compiled with `-DOKC_WGS=1`") in the cheapest possible way — with `-DOKC_WGS=1` the DF
code is *not compiled*, only parsed, so the instantiation risk does not exist. The residual risk is
runtime, and it is (c): a 1 MiB OOB store.

## 5. Reverse check: what DF assumes, and where it breaks at T ≥ 17

Two assumptions, both tied to the MID band:

1. **`p.M * 8 ≤ INTER_SENT_ROWS = 128`, i.e. M ≤ 16 tokens per launch.** Phase A writes rows
   `[0, M*8)` of the shifted `inter`; the end-of-launch re-poison covers `min(128, p.M * 8)` rows —
   capped at 128. At T = 32 that leaves rows `[128, 256)` holding the *previous* launch's real values
   (never poisoned, never rewritten for jobs that do not reach them), which are non-sentinel, so
   `df_bad()` returns 0 and phase B consumes stale data immediately: silently wrong, no hang. The
   `spin > 200000` bail-out only bounds the opposite failure (never-written rows), not this one.
2. **The re-poison needs "every CTA is past phase B"** — true in the non-WGS build, where the grid
   barrier at `:2615-2621` always runs. In the WGS build that barrier is compiled out whenever
   `OKC_MFIN && (OKC_MFQ & 4)` (`:2612`), which is the shipping configuration; that is what the
   patch's own `#if !OKC_WGS` around the re-poison encodes. So even with a widened allocation, DF on
   WGS restores the grid barrier MFIN removed and pays for it.

Additionally `load_x_inter_df` reads `p.inter` with `__ldcg` in the *phase-B register layout* — for
the non-WGS ring the layout matches `load_x_regs`; the WGS build's phase B reads x from smem in wgmma
B layout, filled by TMA from `p.xB` (KPERM3-permuted). Porting DF there means replacing the x TMA path
in `wgs_xproducer`/`wgs_seg`, not reusing the patch.

## 6. Price today at T=32/128/256

- What DF would remove in the WGS build: nothing (blocker b). The job-granular `doneA` wait it deletes
  is only compiled in the non-WGS builds; the WGS builds never wait on `p.doneA` for x.
- What the live hand-off costs: one wait per *(job, stage)*, pipelined 3 deep, so at most the first
  stage of the CTA's first phase-B job is exposed. One phase-A unit is ~1.37 µs at d32 (`1.3715 µs/unit`,
  `artifacts/reports/research_scan_2026-11-02.md` L2); a stage is 2 units spread over 8
  warpgroup-epilogues on different CTAs, and the phase-A end spread at d32 is 7-11 µs. The *exposed*
  part is one producer drain that the CTA had no other work for, i.e. **≈0.1-1.4 µs at d32**, and less
  at d128/256 where a CTA's phase A is 4-8× longer and the relative skew shrinks.
- DF's own cost: 1 MiB of sentinel stores per launch (128 rows × 4096 × 2 B) ≈ **0.35-0.7 µs** at
  H200 store bandwidth, plus `__ldcg` polling instead of a counter read.

So: **under 1 µs each way** — the recoverable amount is ≲1.4 µs at d32 and ~0 at d128/256, against
DF's ≈0.5 µs floor, inside the board's ±1.5-1.9% noise (which is 3-5 µs here). Not measurable, and in
the wrong direction.

## 7. If someone still wants it

`klr_df.py`: `df_src()` returns `MOE_CUDA_SRC` with the patch applied; `register(mod)` installs it as
`_MA_ALT_SRCS["moe_dfmain"]` and, only with `DF_MAIN = True`, appends `@moe_dfmain` to
`MOE_DEFINES` (which b4 inherits). Before `DF_MAIN` can be anything but dead weight, three things must
change in the source, none of them a string patch: the `p.inter += 128 * 4096` shift must be dropped
(or `_ma_make_scratch` widened by 128 rows), `load_x_inter_df` must be called from `wgs_xproducer`
with the sentinel covering all `EMAX` rows, and the re-poison must move to the point where the WGS
build actually knows every CTA is past phase B (i.e. re-adding the barrier MFIN removed).
