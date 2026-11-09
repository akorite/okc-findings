# layer_stack2: vjeux's m == 1 GEMV on qkv / o / unpooler (stack addition to the SC6S + MOEDUP row)

Candidate: `src/okc_loop/vendored_work/layer/layer_stack2.py` (**999,807 B**), base
`layer_sc6s_dup.py` (989,712 B: `MOEDUP = True`, `SC6_SKINNY = True`, `XRES_L2 = False`, unchanged here).
New flag `GEMV_M1 = True`, new CUDA define `-DOKC_GEMV_M1`.

**Expected: −1.74 µs on the decode-1 step (73.5 µs) = +0.39 % of the layer score** (range +0.34…+0.43 % for
−1.5…−1.9 µs). That is 0.18σ of the board's 2.2 % A/A σ: **a stack addition, not a standalone candidate** — it
needs ≥ 3 averaged runs to be distinguishable from the null, and even a lucky reading leaves the row at
≈ 2.275, i.e. ~4.6 % short of top-20 (2.3900). The +0.82 % ceiling needs the router too, and the router needs the
fused `pr` launch split, which costs +5.5 µs at d1 (`kernarc_digest.md:61`) — so it is not reachable.

Basis: the mechanism is the only unported upstream kernel with a measured effect (`layer_unported.md` §2 #1-2).

---

## 1. The three call sites, and what each is worth

Measured by upstream's own tuner on the same board, per-kernel CUDA-graph replay with a cold-L2 sweep
(`vendored_work/bf16/profiles/geo.c8c64c72c95a4ae9.log:49-53`); a ~4 µs graph fixed cost is in every number, so
only the differences are used:

| our call site | weight | kernel instantiation | grid | cgemv | sk2 | Δ |
|---|---|---|---|---|---|---|
| `dense(x, state["qkv"], T)` (:11865) | 2304 × 8192 | `cgemv_kernel<2, 256, 8192>` | 1152 × 288 | 12.48 | 13.06 | **−0.58** |
| `dense(o_in, state["o"], T, state, "o")` (:11942, :12820, :13458, :13664) | 8192 × 1024 | `cgemv_kernel<2, 64, 1024>` | 4096 × 96 | 8.77 | 9.22 | **−0.45** |
| `dense(m, wu, T, state, "un")` (:11992, :12876, :13503, :13709), `wu = state["unpool_slice"]` = 1024 × 4096 | 1024 × 4096 | `cgemv_kernel<2, 128, 4096>` | 512 × 160 | 7.10 | 7.81 | **−0.71** |

Sum **−1.74 µs**. The layer's six-load geomean gives `Δscore/score = (Δt/t)/6` (`scoring.py:98-107`), so
−1.74/73.5/6 = **+0.395 %**; the per-load share is 0.79 / 0.61 / 0.97 % of d1s.

The brief's "unpooler −0.5…−0.9" is the −0.71 above (`cgemv=7.1 sgemv=7.49 sk2=7.81` is the same log line). The
`17.34/18.21 = −0.87` pair (log:51) that `layer_unported.md` §2 #1 labels is the **pooler**, which is excluded (§6), so quote −1.74 as
the sum, not −1.87.

### How the dispatch reaches them

No new dispatcher. The layer already pins the dense backend by `(T, name)` in the final `prepare()` override
(:13063-13072: `mm_pick[(T, "qkv"/"o"/"un")] = "bfp"` for T ∈ {1,8,16,32}, and
`bfp["choice"][(layer, T)] = bfp__static_choice(layer, T, bfp)`), and `bfp_kernel_fn` → `bfp__run` executes that
choice. So the m == 1 GEMV is one more arm of the **same `sk2` route selection**:

* `bfp__static_choice` (:10386-10388) returns `("cgemv",)` when `GEMV_M1 and m == 1 and layer in state["cg"]` and the
  probe has not disabled the route; otherwise the pre-existing `sk2` logic runs unchanged.
* `bfp__run` (:10365-10371) launches `cgemv(L, C, x, tab, pmax, out, base4, rows, nth)` and returns the bf16
  `[1, N]` block as one partial (`dense` wraps it as `view(1, T, N), 1`, the same shape the `sk2`/`bfp` arm returns).
* `bfp__prepare` (:10173-10180) builds the row-major planes for `qkv_gate`, `o` and the unpooler's 1024-row slice
  (`st["unpool_slice"] = weights["unpooler"][:1024]`, :11567 — the same tensor the T ≤ 8 call sites pass).
* `bfp__candidates` is untouched: the static choice *is* candidate 0, so the selftest still probes it (§3).

## 2. Flag and build gate

* `GEMV_M1 = True` (:6937). Off ⇒ nothing of this mechanism exists in the build.
* `GEMV_M1_SRC` (device half, :9471) + `GEMV_M1_HOST` (ATen launcher, :9597) are appended to the single TU **only**
  when the flag is on (:9680), `"cgemv"` is added to `load_inline(functions=...)` only when it is on (:9686), the
  define `-DOKC_GEMV_M1=1` is passed only when it is on (:9688-9689), and the extension name gains `_g1` so a cached
  build of the old TU can never be reused. With the flag off the compiled TU, the exported function list and the
  extension name are exactly `layer_sc6s_dup.py`'s.
* Inside the CUDA source the arm is `#if OKC_GEMV_M1` (default 1 in an appended-only-when-on string), so `-D
  OKC_GEMV_M1=0` compiles the same text to an empty arm — that is the control in §4.

## 3. Guardrails

1. **Inert for T > 1 and for every nearby shape.** The route needs `m == 1`; verified by offline execution of the
   real functions (`bfp__static_choice` / `bfp__nearby_choice` / `bfp__route_key` extracted from the file):

   | layer | m=1 | m=2 | m=8 | m=16 | m=33 |
   |---|---|---|---|---|---|
   | qkv_gate, o, unpooler | **cgemv** | sk2 | sk2 | sk2 | torch/wide |
   | pooler, router | sk2 | sk2 | sk2 | sk2 | torch/wide |

   `bfp__nearby_choice(m=2…8)` returns the m=8 bucket choice, so an untuned m can never inherit `cgemv`.
2. **Inert for every misshapen matrix.** The planes are built only for `name in {"qkv_gate","o","unpooler"}` **and**
   only when `(K, rows per CTA, threads)` is one of the three compiled instantiations (`bfp_CG_INST`, :9867); the
   compression itself declines (`None`) when the escape share exceeds 1/200 or when its exactness check fails. Any of
   those leaves no `state["cg"]` entry ⇒ `bfp__static_choice` falls through to `sk2` with no error. Zero extra
   device memory beyond the planes (qkv 28.3 + o 12.6 + un 6.3 ≈ **47 MB** of read-only buffers).
3. **Probe, and the fallback to sk2.** `bfp__selftest` (:10273-10296) runs every candidate of every layer at
   m = 1, 8, 16, 32, 33, 128, 200, 256 against an FP64 reference (relative tolerance 5e-3, bf16 output) and adds
   `bfp__route_key(layer, 1, "cgemv")` to `state["disabled"]` on mismatch or launch failure. `bfp__static_choice`
   checks that set (:10387), so a failing probe silently restores the sk2 route. Verified offline: with the key
   disabled, 0 entries in `state["cg"]`, or `GEMV_M1 = False`, the m = 1 choice is `sk2`.
4. **No numerical path change; the rounding order differs.** Both paths compute the *same exact* products: the
   lossless 12-bit decode (`L` low byte + 4-bit code + base-4 window, `gemv_hib`) reproduces the stored BF16 weight
   bit-exactly, and escapes are corrected with the exact FP32 delta `wtrue − decoded` (asserted in
   `bfp_cg_compress`: `dec + delta == wtrue`, and replayed in §4c). x is bf16 and enters both paths as the same real
   numbers (cgemv shifts the bf16 into an fp32 register; sk2 feeds it to the tensor core). The accumulation trees
   differ:

   * **cgemv**: thread `tid` owns `k = 16·tid + 16·j·NTH + [0..15]`; 16 sequential FMAs per group; partials reduced
     by an xor butterfly over the 32 lanes (pairwise tree); then the `NWARP` warp partials **and** the escape warp's
     partial summed in ascending index order (the escape correction is the last add); one final
     `__float2bfloat16_rn`.
   * **sk2 (m ≤ 32)**: one CTA owns its row block over the whole K (no cross-CTA reduction); per 64-k unit the
     `mma.m16n8k16` adds 16 products in the tensor core's fixed tree order, units accumulate in ascending k order;
     the epilogue reduces lanes/warps in a fixed order and rounds once to bf16.

   So this is a **re-association, not an approximation**: expect ~1e-7 relative difference from sk2 (four orders
   inside the 5e-3 selftest tolerance), not bit equality. It is the identical numeric contract upstream ships on the
   same component board; it must still pass the judge's own accuracy check on the layer row.
5. **PDL edge (the one risk that can eat part of the gain).** At T ≤ 128 the layer launches with programmatic
   dependent launch and `sk2` participates (`bfp_SK2_PDL`, `sk2_set_pdl`): it triggers its dependents after the
   first weight stages. `cgemv` is a plain launch, so at T = 1 the three edges lose that early trigger and simply
   serialise at kernel completion (safe — the *dependent* kernel's attribute is what makes it start early, and its
   absence only removes overlap). The upstream µs above are isolated-kernel deltas, so the in-chain gain can be
   smaller than −1.74 µs. If the run shows it, the follow-up is to add `griddepcontrol.launch_dependents` after the
   first loads and `griddepcontrol.wait` before the first operand use, mirroring `SK2_WAIT`; not done here because a
   wrong wait placement is a correctness bug and it cannot be measured offline.

## 4. Verification (no GPU on this workstation; nothing was run on the judge)

a. **Lint**: `PYTHONPATH=src python -m okc_loop.cli lint src/.../layer_stack2.py` → **30 error(s)**, the same finding
   multiset as `layer_sc6s_dup.py` after normalising line numbers (the only textual delta is the shifted inner line
   reference of the "line N is 140,927 chars" error).
b. **`tools/cu_build.py`**: the 64 pre-existing (source × build) entries are status-identical to
   `layer_sc6s_dup.py` — `ok` for `MOE_CUDA_SRC` ×4 and `ATT7_CUDA_SRC` ×7, the same pre-existing
   `bfp_TMA/GEMM/SK2/WIDE/GQ`, `_GRAFT_*`, `_SL_GLUE`, `_ROUTER_FP32` nvrtc failures — plus **6 × `ok
   GEMV_M1_SRC`** (one per define set the tool tries). As documented in the tool, a template body compiled without
   its host launcher emits no `.visible .entry`, so the entry is `compiled, no kernel emitted (template body)`;
   no register report is obtainable in that mode.
c. **Corrupted-constant control** (the new operand is on the compiled path, not in dead code): a copy of the
   candidate with one digit changed in the *only* explicit instantiation (`cgemv_kernel<2, 64, 1024>` →
   `<2, 65, 1024>`) makes all 6 `GEMV_M1_SRC` builds **FAIL** with

   ```
   error: static assertion failed with "cgemv: K >= 16 NTH"
   detected during instantiation of "void cgemv_kernel<ROWS,NTH,K>(...) [with ROWS=2, NTH=65, K=1024]"
   ```

   i.e. the kernel body, its 4-bit code decode and its escape path are instantiated. The same corrupted copy with
   `#define OKC_GEMV_M1 0` compiles `ok` — the define really is the gate.
d. **Packing round trip** (pure-Python replay of the kernel's register decode plus the escape warp against the
   host-side packing): 3 shapes (16×64, 32×128, 8×256), escapes deliberately placed at CTA boundaries →
   **0 mismatches / 7,168 weights**. This is the check that would catch a wrong `C` byte order, a wrong escape-table
   slot (`((row_local << 16) | k, delta bits)`, entry 0 = count) or a wrong `base4` replication.
e. **Dispatch table**: see the table in §3.1 (extracted functions executed against fake states).
f. **Byte budget**: the base was 989,712 B and this addition is 10,095 B, which put the file 605 B over the lint's
   `1000000`-byte single-file budget. Freed, comments only (the `rq_pool.md` §3.6 precedent): 8 decorative
   `# -----…` rule lines, i.e. the rule pairs around 4 section headers (dense skinny GEMM, helpers, QK norm + RoPE,
   attention) collapsed to their title lines. Each removed line was verified to be a `tokenize` COMMENT token
   (not inside any multi-line string), so no code, string or define changed. File now 999,807 B (193 B of headroom).

## 5. Expected effect, honestly

| | µs | share of d1s (73.5 µs) | layer score |
|---|---|---|---|
| qkv | −0.58 | 0.79 % | +0.13 % |
| o | −0.45 | 0.61 % | +0.10 % |
| unpooler | −0.71 | 0.97 % | +0.16 % |
| **sum** | **−1.74** | **2.37 %** | **+0.39 %** |

* 2.28370 → ≈ **2.2747** (−0.39 %). Ceiling if the router's −1.00 and the pooler's −0.87 were also reachable:
  −3.61 µs = **+0.82 %** → 2.265; both need the fused `pr` launch split, which is +5.5 µs.
* Against σ_A/A = 2.2 % this is **0.18σ** per run. One run cannot resolve it; take ≥ 3 averaged runs, and only as a
  stack on top of the SC6S + MOEDUP row (`4201a60c`, pending), which is where the +1.2 %/+2.5 % levers already are.
* If the PDL edge (§3.5) eats the unpooler's or qkv's overlap, the realised gain can fall to ~+0.2 %.

## 6. Not ported, and why

* **`rgemv` (router, −1.00 µs isolated)**: at every scored T the router is inside the fused
  `bfp_fused_pooler_router` launch; splitting it back out costs +5.5 µs at d1 (`kernarc_digest.md:61`).
* **`sgemv` (unpooler BF16 fallback, −0.32 vs sk2)**: dominated by `cgemv` at the same shape in upstream's own log
  (`cgemv=7.1 sgemv=7.49 sk2=7.81`), and `cgemv`'s 4096-K instantiation already covers the unpooler slice.
* **pooler `cgemv` (2 × 512, −0.87)**: same fused-`pr` reason as the router.
* Consequence: the file carries no `RGEMV_SRC`/`sgemv_kernel` code at all — three instantiation arms, not the
  upstream five.

## 7. Kill criterion

Set `GEMV_M1 = False` (one line; the file then compiles and runs exactly `layer_sc6s_dup.py`, whose signals are the
pending `4201a60c` row) if any of:

1. **No effect**: over ≥ 3 averaged runs at the row's settings the layer score does not improve by ≥ 0.27 %
   (≈ 1.2 µs on the d1 load, two thirds of the expected −1.74) — inside noise ⇒ the mechanism is not worth a slot.
2. **Accuracy**: any judge accuracy/precision check regresses at T = 1 (the rounding order of §3.4 is the only
   numerical change; a regression means the re-association is not tolerable on this row, not that the kernel is wrong).
3. **The lever is silently dead**: no `state["cg"]` entry for one of the three layers (the compression declined) —
   the file then runs the unmodified sk2 path and no run will ever show the gain.
4. **The PDL edge dominates**: the d1 load moves by less than the isolated kernel deltas while the decode-16/32 loads
   do not move at all, i.e. the loss of `sk2`'s early dependent trigger at T = 1 cancelled the kernel win — in that
   case port the `griddepcontrol` handshake instead of reverting, and re-price (the other two mechanisms in this
   family are worth ≥ 1 % each only if the same `pr` split is paid, see `layer_unported.md` §3).
