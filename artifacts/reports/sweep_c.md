# Family C: the MoE's tile / cost-table constants, and the dense dispatch

Base: `src/okc_loop/vendored_work/layer/klr_fci5.py` (row **2.28370**, 979,654 B). Board noise SD 1.17 %.
Every variant is a byte copy with **one** coherent change plus one header line (line 2) naming it; line 1 keeps
`# okc: problem=muse_decode_layer`.

## 1. What is actually live (this is the result that shapes the ranking)

Read, not guessed. Line numbers are the base's.

| constant | site | reachable? |
|---|---|---|
| `_MA_MMAX = 384` | `L10177`, loop `L11615/12500/13127/13333/14144` | **only downward.** Scored loads are T = 1, 8, 16, 32, 128, 256 (`L11648`), so `range(0, T, 384)` is always one chunk today. `384` is also the kernel's `constexpr MMAX` (`L4040`), so the scratch is sized for it (`TORCH_CHECK` `L6779-6823`) — a smaller chunk must keep `_MA_EMAX = 384*8` or `prepare()` fails its own checks. |
| `OKC_FCOSTB` / `OKC_FCOSTBB` (35 / 36) | `MAIN_FALLBACK_DEFINES` `L7188`, consumed only by `skinny_split` `L5515` | **dead.** `_main_defines` returns that tuple only when `-DOKC_WGS=1` (`L7191-7194`); `skinny_split` is called only from `run_phase` (`L6215`, `L6224`), which is the `#else` of `#if OKC_WGS` in `moe_fused` (`L6433-6467`), and the fat queue it prices does not exist in the WGS build (`L4275` "the WGS build has no fat queue"). In the TINY builds `nF` is 0 by construction (`OKC_SPLIT` zeroes `nfj` at `L4249`, `TINY` traps on `nF>0` at `L6208`). |
| `OKC_DYNK` (7) | same tuple, `dyn_pool` `L4037` | **dead**: `SKDYN` (`L3940`) is 0 in every TINY build and `dyn_pool` is only called from the non-WGS paths (`L5298`, `L6214`). |
| `OKC_FPF` (5), `OKC_EPI4` (0) | same tuple / `MOE_*_DEFINES` `L6877-6882` | **dead identifiers**: `grep -o 'OKC_FPF[^ "]*'` finds exactly one hit (the definition), `OKC_EPI4` five (the five define tuples). No CUDA source in the file references either. |
| `OKC_FINCOL` / `OKC_FINPF` | `L3918`, `L3983` | live, but only under `!OKC_WGS` (`L6477`, `L6661`), i.e. only in the builds whose define tuples already set them (`L6880`, `L13359`). |
| `OKC_WGSA` / `OKC_WGSB` | `L5597-5598` | live, **measured dead**: `artifacts/reports/costtab.md` §5 (job `3edcd41d4421463f`, geomean 2.2386 vs base 2.2474) — ceiling ~0.1 % of the board. Lane closed. |
| `OKC_WGSU` / `OKC_WGSUB` (160 / 140) | `L6122-6123` | live, already owned by `klr_cost11.py` (176 / 154) and by `moe_cepi`; a further point is a duplicate, not a widening. |
| `OKC_FT_T` (64) | `L3941`, `JMAX` `L4040` | **pinned.** `FT_T = 32` needs `JMAX = 136 > MF_XRS = 96` (`static_assert` `L5565`), i.e. a flag-region change too; `FT_T = 128` needs a 64 KB x slot ×3 (`WGS_XS` `L5552`) — no smem. |
| `RB_A/RB_B`, `UB_A/UB_B`, `WGS_NS` | `L4031-4033`, `L5547` | structural (`static_assert`s, smem at 1 CTA/SM). Not knobs. |
| `BFP_MAX_T = 32` | `L10365` | live and **asymmetric**: the tuner times `bfp` at every T in `(1..256)` (`L11079-11130`) and may set `mm_pick="bfp"` at T = 128 / 256, but `dense()` (`L10374-10380`) then runs `torch.mm` for those picks, so the measurement is discarded. |
| split-K tile cap `600` | `L10387` in `dense()` (live via `_vj_dense = dense` `L12385`) | live for T ≤ 32 and never swept. Note `dense_ng`'s `S` is **not** free: its caller bails on `S != 1` (`L13298`), so only `dense()`'s cap can move. |

Two premises in the brief are therefore false for this base and are worth recording: (a) the *layer's* dense picks are
**not** inherited — `prepare()` times bfp / Triton / cuBLAS per `(T, name)` for T = 1…256 on this GPU (`L11079-11130`);
(b) `_MA_MMAX` was never varied because a value below max T is structurally wrong (§3, rank 3).

## 2. Candidates (all on disk, all verified)

| file | constant | current → proposed | mechanism | expected effect on which loads | verification |
|---|---|---|---|---|---|
| `sweep_c3.py` | `dense()` split-K tile cap `L10387` | `600 → 1200` | `S` doubles while `tiles*S*2 <= cap`; at T ≤ 8 the pooler/router GEMM is `tiles = (N/64)*1` = 72, so 600 stops at `S=4` (288 CTAs ≈ 2.2 waves of 132) and 1200 reaches `S=8` (576 CTAs, `K/S = 512 ≥ 2*BK` still holds) — finer stream-K granularity, no extra weight bytes | decode 1 / verify 1, where dense GEMM is 52.5 % / 34.9 % of the load; only on cells the tuner leaves on the Triton route, and the tuner re-times the new `S` in `prepare()`, so it can only be *chosen* if it measures faster | lint multiset identical; `cu_build` identical; diff = 1 line |
| `sweep_c4.py` | `BFP_MAX_T` `L10365` | `32 → 256` | above 32 a `"bfp"` pick is silently downgraded to cuBLAS; with the cap raised the pick runs the `"wide"` route (`bfp__route` `L9892`), which is otherwise dead code | verify 16 / verify 32 (dense GEMM 12.9 % / 18.9 %), only where the tuner's own timing already prefers bfp at T = 128 / 256 | lint identical; build identical; diff = value + 1 comment line |
| `sweep_c1.py` | `_MA_MMAX` `L10177` (with `_MA_EMAX` pinned to `384*8`) | `384 → 128` | verify 32's MoE becomes two 128-token `okc_run` launches | verify 32 only (MoE 54.3 %) — **predicted loss**, see §3 | lint identical; build identical; diff = 2 lines |
| `sweep_c2.py` | `_MA_MMAX` `L10177` (same pin) | `384 → 64` | 2 launches at verify 16, 4 at verify 32 | verify 16 / 32 — **predicted loss, worse than c1** | lint identical; build identical; diff = 2 lines |
| `sweep_c5.py` | `MAIN_FALLBACK_DEFINES` `L7188-7189` | `FCOSTB 35 → 33`, `FCOSTBB 36 → 34` | prices fat units in `skinny_split` | **exactly 0**: no build we run executes `skinny_split` with `nF > 0` (§1). Emitted as a control, not a bet | lint identical; build identical; diff = 2 lines |

Verification run for all five (`src/okc_loop/vendored_work/layer/`):

```
ruff check --output-format=concise   # 54 distinct messages, sorted multiset byte-identical to the base, 5/5
python tools/cu_build.py <file>      # 10 ok + 54 FAIL, identical set (sorted, define-set text stripped), 5/5
diff -u klr_fci5.py sweep_cN.py      # only the header line and the one edit, pasted below
```

The 54 FAILs are the same pre-existing set as the base (the host-wrapper sources `_ROUTER_FP32_CUDA`, `bfp_*_SRC`,
`_GRAFT_*`, `_SL_GLUE_CUDA`: nvrtc has no `cudaGetLastError`/ATen host headers). `MOE_CUDA_SRC` itself compiles
`ok` for all four MoE define sets in every variant, so c1/c2/c5's define changes are compile-clean — but note
`cu_build` cannot see the *Python* side, so c1/c2's scratch arithmetic is verified by reading the `TORCH_CHECK`s
(`L6779-6823`), not by the build.

## 3. Ranked

1. **`sweep_c3.py` — split-K cap 600 → 1200.** Only candidate whose mechanism points the right way and whose
   downside is bounded by the tuner's own re-measurement. [ESTIMATE] −0.1…−0.4 % on decode 1 / verify 1, ≤0.1 % of the
   board: below the 1.15 % screen, so treat it as a cheap lottery ticket, not a plan.
2. **`sweep_c4.py` — `BFP_MAX_T` 32 → 256.** Pure optionality: the tuner already times this path at T = 128 / 256 and
   its pick is currently thrown away. [ESTIMATE] 0 ± 0.2 % — the base's own note says the wide route was "a wash" at
   T = 128 and 10 µs slower at T = 256 *on the board's 32×8 sliding cell*, which is not the layer's cell.
3. **`sweep_c1.py` — `_MA_MMAX` 384 → 128.** The knob the brief wanted, and the answer is that it is *pinned*: the
   unit is weight-side (a job = one local expert's gate+up matrix, 32 units of 512 rows, `L4031-4033`, `wgs_phase`
   `L6101`), so chunking the tokens does not reduce the unit count — it makes the same routed experts' weights stream
   **twice**. The base's own profile says the MoE is weight-bound (T = 16 → 32 costs 0.2 µs/token against 73.6 µs at
   T = 16), and an expert hit by 128 tokens is still hit at 256 (P(hit) 98 % vs 99.97 %), so [ESTIMATE] the MoE at
   verify 32 roughly doubles its weight traffic — tens of percent on that load, not a 1–2 % variant. Do not spend a
   check; kept to close the knob with a stated mechanism.
4. **`sweep_c2.py` — `_MA_MMAX` 384 → 64.** Same argument, two loads instead of one, strictly worse.
5. **`sweep_c5.py` — `FCOSTB/FCOSTBB` −2.** Expected exactly 0 (dead path, §1). A control for anyone who doubts the
   deadness proof: if this ever scores ≠ the base's noise, the exclusion in `const_sweep_fci5.md` E1 is wrong.

## 4. Diffs

```
=== sweep_c1 ===
+# sweep_c1: MoE chunk _MA_MMAX 384 -> 128 (verify 32's MoE becomes two 128-token okc_run launches; scratch stays at EMAX 384*8).
-_MA_MMAX = 384
-_MA_EMAX = _MA_MMAX * 8
+_MA_MMAX = 128   # sweep_c1: was 384, which never chunked (every scored T <= 256)
+_MA_EMAX = 384 * 8   # the kernel's constexpr MMAX is still 384: its TORCH_CHECKs size the scratch from okc::EMAX

=== sweep_c2 ===
+# sweep_c2: MoE chunk _MA_MMAX 384 -> 64 (verify 16 / 32 split into 2 / 4 okc_run launches; scratch stays at EMAX 384*8).
-_MA_MMAX = 384
-_MA_EMAX = _MA_MMAX * 8
+_MA_MMAX = 64    # sweep_c2: was 384, which never chunked (every scored T <= 256)
+_MA_EMAX = 384 * 8   # the kernel's constexpr MMAX is still 384: its TORCH_CHECKs size the scratch from okc::EMAX

=== sweep_c3 ===
+# sweep_c3: dense split-K tile cap 600 -> 1200 (one more K split for the T <= 32 Triton route; the per-cell tuner gates it).
-    while tiles * S * 2 <= 600 and K // (S * 2) >= 2 * BK:
+    while tiles * S * 2 <= 1200 and K // (S * 2) >= 2 * BK:   # sweep_c3: was 600 (one more split at T <= 32)

=== sweep_c4 ===
+# sweep_c4: BFP_MAX_T 32 -> 256, so a bfp pick the tuner measures at T = 128 / 256 actually runs (the "wide" route).
-BFP_MAX_T = 32   # bf16_proj's qkv / o / unpooler kernels where they win inside the chain (T=1..32: +4-9% on those
+BFP_MAX_T = 256  # bf16_proj's qkv / o / unpooler kernels where they win inside the chain (T=1..32: +4-9% on those
                  # loads); at T=256 their routes measured 10 us slower than cuBLAS on 32x8 sliding, at T=128 a wash
+                 # sweep_c4: 32 -> 256, so a "bfp" pick the tuner measures at T=128/256 runs instead of being forced to cuBLAS

=== sweep_c5 ===
+# sweep_c5: MAIN_FALLBACK_DEFINES FCOSTB 35 -> 33 / FCOSTBB 36 -> 34 (control: that fat-queue path is not in any build we run).
-... "-DOKC_FCOSTB=35",
-    "-DOKC_FCOSTBB=36", ... "-DOKC_FPF=5")
+... "-DOKC_FCOSTB=33",
+    "-DOKC_FCOSTBB=34", ... "-DOKC_FPF=5")   # sweep_c5: FCOSTB 35->33, FCOSTBB 36->34
```

## 5. What is left in this family, if anyone wants a sixth

Nothing with a positive prior. The remaining live edges are owned elsewhere (`OKC_WGSU/WGSUB` → cost11/cepi,
`OKC_WGSFCI` → st2, `OKC_WGSGS` → gs3, `OKC_WGSDFIN` → dfin, `OKC_WGSNX` → nx4) or closed by measurement
(`OKC_WGSA/WGSB` → `costtab.md`). One untested-but-blocked idea is recorded so it is not re-derived:
`dense_ng`'s fused gate GEMM runs `S = 1` (no split-K) with 72–128 CTAs at T ≤ 8 and was never timed on its own, but
the fast-path caller is gated on `S == 1 and bf16` (`L13298`, `L13344`; `dense_ng` hands back fp32 partials), so it is
not a local tile knob — raising `S` there changes nothing on that path and changes the *other* one.
