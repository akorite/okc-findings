# `OKC_MOEDUP` in the layer's MoE — what it is, what it would buy, and the no-go

Candidate: `src/okc_loop/vendored_work/layer/layer_dup.py` (copy of `klr_fci5.py`; `MOEDUP = False`, so with the
switch off the file compiles and packs exactly as the base — verified below). Nothing shipped was touched.

Mechanism ported from `src/okc_loop/vendored_work/moe/moe_dup.py` (`OKC_MOEDUP`, `:300-332`, `:2003`), which measured
2.38393 check / 2.38062 submit vs a 2.37458 row on the layer board.

**Verdict: expressible, implemented, and NOT worth one of the four runs — expected +0.3…+0.8% of the layer board
against a 2.2% A/A σ.** The layer's 82 MB of amplification at v32 is *not* the duplication this lever fixes; it is the
x-stage refetch, which is the already-priced `OKC_XRES` lever (+1.18%, `moe_bytes.py`). See §5-6.

---

## 1. What `moe_dup`'s MOEDUP actually does

Two separate uses of the same 64-bit constant, and only one of them is MOEDUP:

**The constant.** `0x12F0…` and `0x14F0…` are opaque 64-bit L2 *cache-policy handles* — the operand `.L2::cache_hint`
takes. They are hardcoded here rather than produced by `createpolicy.fractional.L2::evict_last.b64 …, 1.0` (which is
what the layer's `OKC_XRES` and its attention path do, `klr_fci5.py:7314-7317`): `0x12F0…` = evict-first, `0x14F0…` =
evict-last, fraction 1.0 in both. The repo's own comments state those two meanings (`moe_dup.py:307-316`,
`klr_fci5.py:7311-7312`); the nibble that differs (`0x2` → `0x4`) is the eviction-priority field
[INFERENCE from those two comments + the CUTLASS `CacheHintSm90` they name].

**Read side — this is MOEDUP.** `moe_dup.py:709/1155/2003/2531`:

```c
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [smem], [src], bytes, [mbar], policy;
```

with `policy = L2H_DUP_ARG(S, j)`: `0x14F0…` (keep) when job `j` has a *following job of the same expert*
(`j + 1 < njF + njS && jE[j+1] == jE[j]`), else the shipped `0x12F0…` (stream through). `route()` emits an expert's
chunks back to back, so an expert with 65-128 tokens is jobs `j, j+1` and its second job re-reads the whole expert;
keeping the first copy's tiles in L2 turns that re-read into a hit. The `#else` arm of `OKC_MOEDUP` expands to plain
`L2H_ARG`, i.e. byte-identical instruction stream to shipped. Hints only: no result can change.

**Write side — not MOEDUP.** `st_keep_bf16` / `st_keep_f32` (`moe_dup.py:320-325`, called on `inter` and `ybuf` at
`:637,651,938,946`) are the *pre-existing* `OKC_L2H` lever: `st.global.L2::cache_hint.b16/.b32 …, 0x14F0` marks those
*stores* evict-last so the later re-reads (phase B reads `inter`; the final pass reads `ybuf`) hit L2. MOEDUP neither
adds nor changes a store — it reuses the same constant on a different instruction. So: a hint on a **TMA load**, with
a policy handle baked in as an operand; not a warm, not a store, not a `createpolicy`.

## 2. Can the layer's MoE express it? Yes — one site

The layer's WGS build streams every weight stage from exactly one place, `wgs_wproducer`
(`klr_fci5.py:5698-5726`):

```c
const int j = u >> ubl, ub = u & ((1 << ubl) - 1);          // :5717  job index, unit-in-job, stage-in-unit
const unsigned char* src = base + (S.jE[j] << ubl + ub) * 16 + st) * WGS_STG;
cp.async.bulk … L2H_SUFFIX … L2H_ARG;                       // :5719-5720  shipped: evict-first on every stage
```

`j`, `S.jE`, `S.njF`, `S.njS` are all in scope, the job list is the standalone's (`route()`, `:4246-4274`: fat chunks
of `FT_T = 64` first, an expert's chunks consecutive, then the skinny jobs), and the fat queue's LPT reorder is
`#if !OKC_WGS` (`:4276`) so it cannot break the adjacency — exactly the standalone's precondition. The only site is
this one: for `OKC_WGS` the `run_phase`/`fat_stream` path is not compiled into the kernel (`:6457-6461`
`wgs_phase<false>/<true>` vs `:6463-6467` `run_phase`), and the tiny/tiny2/mid builds (`ring_issue :4422`, `fat_issue :4863`) never see an
expert above 64 tokens (§3), so swapping their `L2H_ARG` too would be a conditional that is always false.

The change (`layer_dup.py`, 4 hunks, `git diff klr_fci5.py`):

| where | change |
|---|---|
| `:3991-3993` | `#ifndef OKC_MOEDUP / #define OKC_MOEDUP 0` |
| `:4005-4012` | `L2H_DUP_ARG(S_, j_)` — the standalone's ternary — under `#if OKC_MOEDUP`, else `L2H_ARG` |
| `:4023` | `#define L2H_DUP_ARG(S_, j_)` in the `OKC_L2H=0` arm (the call site is compiled regardless of `OKC_L2H`) |
| `:5734` | the `wgs_wproducer` weight copy: `L2H_ARG` → `L2H_DUP_ARG(S, j)` |
| `:6916`, `:6934-6935` | `MOEDUP = False`; `_ma_ext()` appends `-DOKC_MOEDUP=1` only when `-DOKC_WGS=1` |

That is every WGS build, so T = 32's B4FIN build carries the flag too; its conditional is false on every job
(`dup = 1.000`, §3) and costs one compare per stage in the single producer lane, off the consumer critical path.

Note the layer's stages are `WGS_STG = 35,840 B` (SC6W 6-bit scales on), not the standalone's 36,864 B, so one expert
per job is **27.525 MB**, not 28.31 MB.

## 3. How much duplication does the layer actually have — measured, not assumed

Job model at a scored T: `ep8`, 32 local experts, top-8 of 256, `POPULARITY = 0.03`, busiest rank
(`problems/muse_moe_w4a16/reference.py:66-88`). Replica in numpy (2000 seeds/shape, `scores = U + 0.03·z/√(−log u)`,
top-8, busiest-32-group relabel): it reproduces `moe_bytes.md` §3-4 to <0.3%, and that replica in turn back-solves
the judge's published floor `touched = 0.75·distinct` to <1%:

| T (= bs) | load | routes | distinct | jobs | dup = Σjobs/Σdistinct | extra streams | extra bytes @ 27.525 MB |
|---|---|---|---|---|---|---|---|
| 1 | D1 | 2.6 | 2.57 | 2.57 | 1.000 | 0 | 0 |
| 8 | V1 | 13.7 | 8.48 | 8.48 | 1.000 | 0 | 0 |
| 16 | D16 | 26.0 | 11.99 | 11.99 | 1.000 | 0 | 0 |
| 32 | D32 | 50.0 | 15.85 | 15.85 | 1.000 | 0 | 0 |
| 128 | V16 | 194 | 22.60 | 22.95 | 1.016 | 0.36 | **9.9 MB** |
| 256 | V32 | 385 | 24.58 | 26.02 | 1.059 | 1.44 | **39.6 MB** |

So the mechanism has something to act on at only two of the six scored T (four of the twelve cells), and the number of
duplicated experts is 1.44 — the same 1.46 the standalone pays. This is *not* where the layer's 622-vs-540 MB gap
comes from.

## 4. Price at the layer, not at the standalone

`dup` bytes at the layer's MoE rates: `kernarc_digest.md` measures v32 at 3.19 TB/s (the byte report uses a
conservative 2.39-2.54); V16 at ~3.5 TB/s.

| shape | saved | µs | board %/µs (layer) | board % |
|---|---|---|---|---|
| V32S | 39.6 MB | 12.4 | 0.0357 | 0.44 |
| V32F | 39.6 MB | 12.4 | 0.0125 | 0.16 |
| V16S+F | 9.9 MB | 2.8 | ≈0.030 + ≈0.011 | ≈0.11 |
| **total** | 49.5 MB | | | **≈ +0.7%** |

The V32 sensitivities are the ones the `OKC_XRES` pricing used (0.86% / 24.1 µs, 0.32% / 25.7 µs), so the table is
consistent with the file's own byte account; at 2.5 TB/s instead of 3.19 it is +0.85%. **The honest cross-check is the
hardware measurement of the same mechanism on the same shape: +0.27% layer board** (11.7 µs there vs 12.4 here), i.e.
the derived row is if anything optimistic by ~2.5×. Quote the range as **+0.3…+0.8%, expected ≈ +0.5%**.

Two reasons it could land under even that:

* In WGS the stream-K split hands job `j` and job `j+1` to *different* CTAs, so the two copies of an address are
  in flight concurrently; the standalone's re-read follows its first read within one CTA's ordered stream. Concurrent
  duplicates cannot hit L2 in the first place, only coalesce - and the evict-first tag job `j+1` puts on the same
  address can retroactively demote the line job `j` pinned.
* The pinned set (up to 2 × 27.5 MB per duplicated expert, and 1.44 of them) is evict-last, competing with the
  evict-last `inter`/`ybuf` stores the same kernel relies on. A regression is possible, not just a null.

## 5. The 82 MB is a different site — and it is already priced

`layer_moe_bytes.md` §Lever B and `moe_struct.md` §0: the v32 MoE fetches 622 MB for 540 MB touched, and the 82 MB is
the **x refetch** — one stage carries `npad·512 = ntt·4096` B of `x`, `st = gs & 15` depends only on the stage, so a
job's 16 x-blocks are copied 32× (phase A) + 16× (phase B): 120 MB of demand for 2.4 MB of unique data, of which
~79.6 MB is recoverable. That is `OKC_XRES` (`moe_bytes.py:5547-5549`, `5742-5748`, `5773-5779`), priced at **+1.18%**
(−24.1 / −26.0 µs on V32S/V32F). It is a *different* instruction (the `wgs_xproducer` copy, `:5731-5784`) and a
different policy handle (`createpolicy.fractional.L2::evict_last.b64`).

Weight duplication is 40 MB; x refetch is 82 MB. The port in this report therefore captures at most a third of what
`moe_dup` captures on its own board, on the same shape.

## 6. Go / no-go

**No-go on spending a run on `OKC_MOEDUP` alone**, on four grounds:

1. **Under the resolution bar by 3-7×.** ≈ +0.5% expected (0.3-0.8 range, +0.27% measured for the identical mechanism
   on the identical shape) against a 2.2% A/A σ: 0.23σ. A single run reports a number that cannot be distinguished
   from the null; even a lucky +1.5% reading would sit inside the A/A band and would not survive as evidence.
2. **Wrong byte.** It touches 1.44 duplicated expert streams (40 MB); the layer's measured amplification is the x
   refetch (82 MB). The brief's premise - "the same duplication `moe_dup` fixes" - does not hold: the layer re-reads
   x, not weights, and the weight job model is identical to the standalone's (§3).
3. **Only 4 of 12 cells.** Dup is 1.000 at T = 1, 8, 16, 32 and 1.016 at T = 128. Even at full capture the board
   gain is bounded by the V32 pair's 0.048 %/µs.
4. **Known-negative risk, not just null.** The WGS producer interleaves a duplicated expert's two copies across CTAs
   and the pin competes with the kernel's own evict-last `inter`/`ybuf` streams.

**What to spend the run on instead, in the same family.** If one of the four is spent on the L2-policy family, spend
it on the *combined* file — `OKC_XRES` (+1.18%) plus `OKC_MOEDUP` (+0.5%) — because they are disjoint (different
producer, different policy handle, no smem, no state, no extra bytes) and their sum, ≈ +1.7…2.0%, is the only version
of this family that reaches the bar at all, at ~0.8-0.9σ. That is still a coin flip and I do not recommend it on a
board 4.7% short of top-20; but it is strictly better than the port alone. Composition is mechanical: `moe_bytes.py` is the
same base (its weight copy is the identical line at `:5723`, 3 lines later from its XRES define block), so my 4 hunks
apply there unchanged - same `#if OKC_L2H` block, same `wgs_wproducer` call, same `_ma_ext` - and both switches
default off, so one merged file is one edit away.

Nothing else in the ring clears 2%: the remaining recoverable amount is the 2.4-3.2 TB/s bandwidth gap
(`moe_bytes.md` §7: +20% of the MoE board if the fetch path reached 4.287 TB/s), which is a different lever and not an
L2-policy one.

## 7. Acceptance

* `PYTHONPATH=src python -m okc_loop.cli lint --limit 500`: **84 findings, 30 error(s)** on both files; normalized
  multisets identical, the only textual difference is the inner line reference of the "line 12583 is 140927 chars"
  error (`12583` → `12601`, the shift the added lines imply).
* `python tools/cu_build.py … --variant MOE`: **identical status** to `klr_fci5.py` for all 44 (source × build)
  entries — `ok MOE_CUDA_SRC` on all four MOE builds, the same pre-existing `_ROUTER_FP32`/`_GRAFT_`/`_SL_GLUE`/`bfp_*`
  nvrtc failures. With `-D OKC_MOEDUP=1` the same four MOE builds compile `ok`, and a deliberately corrupted policy
  constant in the `OKC_MOEDUP` arm makes the WGS build FAIL — the new operand is on the compiled path, not in dead
  code. `tools/cu_build.py` itself unmodified.

## 8. Merged candidate

`src/okc_loop/vendored_work/layer/layer_xres_dup.py` = `moe_bytes.py` (SC6W + `OKC_XRES`) + the four `OKC_MOEDUP`
hunks of `layer_dup.py` (§2), `SC6_SKINNY = False`.

**Flag states:** `SC6_SKINNY = False` (SC6S alone −0.99%, run `f41d1b56`), `XRES_L2 = True`, `MOEDUP = True`.
`_ma_ext` appends `-DOKC_MOEDUP=1` on `-DOKC_WGS=1` builds only, exactly as `layer_dup.py:6934-6935`; the four C hunks
are byte-identical to `layer_dup.py:3993-3995`, `:4005-4012`, `:4023`, `:5734` (`diff` of each range: empty).
Header unchanged from the base, first line `# okc: problem=muse_decode_layer`.

`diff -u moe_bytes.py layer_xres_dup.py` — the only differences are the MOEDUP block and the lever-A default:

```diff
@@ -3988,6 +3988,9 @@
 
 namespace okc {
 
+#ifndef OKC_MOEDUP
+#define OKC_MOEDUP 0  // an expert split into >1 fat job re-reads its 27.53 MB per job: keep the earlier copy in L2
+#endif
 #ifndef OKC_L2H
 #define OKC_L2H 1
 #endif
@@ -3997,6 +4000,14 @@
 #define L2H_SUFFIX ".L2::cache_hint"
 #define L2H_OPERAND ", %4"
 #define L2H_ARG , "l"(0x12F0000000000000ull)
+// jobs of one expert are consecutive (route() emits an expert's chunks back to back), so job j's stream is re-read by
+// job j + 1: fetch that first copy evict-last, which the evict-first streams then evict around instead of through.
+// Hints only: results are unchanged.
+#if OKC_MOEDUP
+#define L2H_DUP_ARG(S_, j_) , "l"(((j_) + 1 < (S_).njF + (S_).njS && (S_).jE[(j_) + 1] == (S_).jE[j_]) ? 0x14F0000000000000ull : 0x12F0000000000000ull)
+#else
+#define L2H_DUP_ARG(S_, j_) L2H_ARG
+#endif
 __device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) {
   asm volatile("st.global.L2::cache_hint.b16 [%0], %1, %2;\n" ::"l"(a), "h"(*reinterpret_cast<unsigned short*>(&v)), "l"(0x14F0000000000000ull) : "memory");
 }
@@ -4007,6 +4018,7 @@
 #define L2H_SUFFIX ""
 #define L2H_OPERAND ""
 #define L2H_ARG
+#define L2H_DUP_ARG(S_, j_)
 __device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) { *a = v; }
 __device__ __forceinline__ void st_keep_f32(float* a, float v) { *a = v; }
 #endif
@@ -5720,7 +5732,7 @@
       const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)WGS_STG;
       asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(WGS_STG) : "memory");
       asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
-                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(WGS_STG), "r"(smem_u32(bar)) L2H_ARG : "memory");
+                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(WGS_STG), "r"(smem_u32(bar)) L2H_DUP_ARG(S, j) : "memory");
       if (++ws == WGS_NS) { ws = 0; wph ^= 1; }
     }
   }
@@ -6914,8 +6926,10 @@
 
 
 SC6_WGS = True      # lossless 6-bit block scales in the WGS stages (moe-scale SC6, harness/make_sc6.py)
+MOEDUP = True       # an expert with 65..128 tokens is split into 2 fat jobs and the second re-reads its 27.53 MB;
+                    # fetch the first copy evict-last (-DOKC_MOEDUP). T=256 only: 26.0 jobs / 24.6 experts; T<=32 1.000
 _MA_SC6W = False    # set in prepare(): scale bytes all in 0x40..0x7F and the PRMT decode and a WGS build
-SC6_SKINNY = True   # lever A: the same 6-bit scales in the skinny (!WGS, -DOKC_TINY=1) unit copies (-DOKC_SC6S)
+SC6_SKINNY = False  # lever A off: OKC_SC6S alone measured -0.99% (run f41d1b56), not part of this family
 _MA_SC6S = False    # set in prepare(): SC6_SKINNY, scale bytes all in 0x40..0x7F, and the probe agrees bit-exactly
 XRES_L2 = True      # lever B: the job's x blocks stay L2-resident in the WGS stage refill (-DOKC_XRES)
 _MA_XRES = False    # set in prepare(): XRES_L2 and the probe agrees bit-exactly
@@ -6940,6 +6954,8 @@
         defines = defines + ("-DOKC_XRES=1",)
     if DQ7 and "-DOKC_DQ7=1" not in defines:
         defines = defines + ("-DOKC_DQ7=1",)
+    if MOEDUP and "-DOKC_WGS=1" in defines and "-DOKC_MOEDUP=1" not in defines:
+        defines = defines + ("-DOKC_MOEDUP=1",)  # only the WGS builds stream weights from wgs_wproducer
     if defines not in _MA_EXTS:
         src = MOE_CUDA_SRC
         defs = [d for d in defines if not d.startswith("@")]
```

**Lint** — `PYTHONPATH=src .venv/bin/python -m okc_loop.cli lint --limit 500 <file>`:

```
FAIL  layer_xres_dup.py
      FAIL tier0 muse_decode_layer: 30 error(s)     (84 findings, 0/1 clean)
```

Same 84 findings / 30 error(s) as `moe_bytes.py`; normalized multisets identical (84 vs 84, 30 vs 30 errors, empty
symmetric difference). Only the quoted line numbers differ: +12 for the C-half findings (`4374`→`4386`, `4992`→`5004`,
…), +16 for the Python-half ones (`6993`→`7009`, …, `16428`→`16444`), and the long-line error's inner reference
`line 12731 is 140927 chars` → `line 12747 is 140927 chars`.

**Build** — `PYTHONPATH=src .venv/bin/python tools/cu_build.py src/okc_loop/vendored_work/layer/layer_xres_dup.py -D OKC_MOEDUP=1`:

```
ok   MOE_CUDA_SRC [OKC_NW=16 OKC_FAST1=0 OKC_EPI4=0 OKC_WGS=1 OKC_NS=3 OKC_NSF=3] 
ok   MOE_CUDA_SRC [OKC_NW=8 OKC_TINY=1 OKC_EPI4=0 OKC_FAST1=2 OKC_L2H=0 OKC_NACC=4 OKC_UWARP=0 OKC_WS=1 OKC_WSB=1 OKC_WSBL=2 OKC_WSBLK=2 OKC_NS=3 OKC_NSF=3] 
ok   MOE_CUDA_SRC [OKC_NW=8 OKC_TINY=1 OKC_CPF=0 OKC_EPI4=0 OKC_FAST1=2 OKC_NACC=4 OKC_UWARP=0 OKC_WS=1 OKC_WSB=1 OKC_WSBL=2 OKC_WSBLK=1 OKC_FINPF=1 OKC_NS=3 OKC_NSF=3] 
ok   MOE_CUDA_SRC [OKC_NW=8 OKC_TINY=1 OKC_SPLIT=1 OKC_CPF=0 OKC_EPI4=0 OKC_FAST1=2 OKC_NACC=4 OKC_UWARP=0 OKC_WS=1 OKC_WSB=1 OKC_WSBL=2 OKC_WSBLK=2 OKC_FINPF=1 OKC_NS=3 OKC_NSF=3] 
```

All four MOE builds `ok`; the remaining 54 entries are the pre-existing `_ROUTER_FP32` / `_GRAFT_` / `_SL_GLUE` /
`bfp_*` nvrtc failures, and the full 64-entry `ok`/`FAIL` status set is identical to `moe_bytes.py`'s (`10 ok / 54 FAIL`
both). Negative control on a throwaway copy: replacing the `0x14F0…` arm's constant with an undefined token makes the
`OKC_NW=16 OKC_WGS=1` MOE build FAIL under `-D OKC_MOEDUP=1` and leaves all four `ok` without it — the merged operand
is on the compiled path and the off arm is the shipped one.

