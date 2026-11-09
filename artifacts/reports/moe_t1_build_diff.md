# The T=1 (bs=1) build in three files — what actually differs, and what it costs

Question: the in-layer MoE costs **29.4 µs at T=1** (`moe_fused<false>`, profile
`82c098ae…7cd40fdd1a934e27`, 37.7% of a 66 µs decode-1 step) while the standalone MoE's bs=1 case is
**~22 µs** for the same bytes. Is that a build difference (defines / epilogue), and if so which build
wins at bs=1?

Files compared, all three at the T=1 dispatch (M = token count, so M = 1):

| | file | what it compiles at M ≤ 4 |
|---|---|---|
| **layer** | `src/okc_loop/vendored_work/layer/klr_fci5.py` (row file, 2.28370) | `MOE_TINY_DEFINES` (:6878) **+ `(" @moe_helper0", "-DOKC_FINPF=1")` appended at :13359** → the 2671-line "v2" body stored in `_MA_ALT_SRCS["moe_helper0"]` (:12583) |
| **CAS / standalone** | `artifacts/cas/6402f5f3…b5d2b3e5.py` (MoE board incumbent, 2.3863) | `TINY_DEFINES` (:133-136) → its own 2698-line `CUDA_SRC` (:194) |
| **vendored keell0renz** | `src/okc_loop/vendored/keell0renz/muse_moe_w4a16.py` | `TINY_DEFINES` (:145-148) → its own 2885-line `CUDA_SRC` (:211). The per-batch builds (`PS_SHAPES`, :6815) exist **only for bs = 8** (`tiny2` + `t2_wsi`, the ring-issuer warp) — bs = 1 has no special build |

Dispatch proof (layer): `_moe(T, state)` (:13512) → `M <= MOE_TINY_MAX_M (4)` → `"tiny"` →
`_ma_ext(MOE_TINY_DEFINES + _MA_SFX)`, and `_ma_ext` (:6912) swaps `src` for `_MA_ALT_SRCS[name]`
when a define starts with `@`. `prepare()` builds it up front in the thread pool (:11245).

## 1. Define table (values as the code has them, not as they were assumed)

| define | layer T=1 | CAS bs=1 | vendored bs=1 | live at T=1? what it costs |
|---|---|---|---|---|
| `OKC_NW=8` | ✔ | ✔ | ✔ | 8 compute warps |
| `OKC_TINY=1` | ✔ | ✔ | ✔ | skinny-only: no fat units, `nF == 0` |
| `OKC_EPI4=0` | ✔ | ✔ | ✔ | **0 in all three** — this is not a T=1 knob at all |
| `OKC_FAST1=2` | ✔ | ✔ | ✔ | routing warp splits off early |
| `OKC_L2H=0` | ✔ | ✔ | ✔ | plain `cp.async.bulk` (no L2 hint) |
| `OKC_NACC=4` | ✔ | ✔ | ✔ | |
| `OKC_UWARP=0` | ✔ | ✔ | ✔ | |
| `OKC_WS=1` | ✔ | ✔ | ✔ | +1 helper warp (32 threads), `NTL = NT+32 = 288` |
| `OKC_WSB=1` `OKC_WSBL=2` `OKC_WSBLK=2` | ✔ | ✔ | ✔ | ring pacing of the last units |
| `OKC_FINPF=1` | ✔ (:13359) | ✔ (in the tuple) | ✔ (in the tuple) | **live**: the first final-pass item's `topk_w`/`pos` loads and its `ss`+`ybuf` loads issue *before* the grid barrier. Same code in all three |
| `OKC_DQ7=1` | auto | auto | auto | weight-decode layout (host + device agree) |
| `OKC_PRMTSC=1` (sfx) | auto (`_MA_SFX`) | auto (`sfx`) | auto (`sfxs`) | PRMT scale decode |
| `OKC_NS=3` `OKC_NSF=3` | auto | auto | auto | ring slots |
| `OKC_FNST=4` | injected (`_ma_ext`) | not passed | not passed | **0** — no line of any of the three bodies reads `OKC_FNST` |
| `OKC_SC6W` | WGS builds only → not T=1 | absent | absent | 0 at T=1 |
| `OKC_MFIN` `OKC_B4L` `OKC_B4FIN` `OKC_SC6S` | in the layer's **main** `MOE_CUDA_SRC` only; the T=1 build compiles the helper0 body, which has none of them | absent | absent | 0 at T=1 |
| `OKC_WGSFCI` | 5 (main source) | 4 | 3 | WGS-only → 0 at T=1 |
| alt source | `@moe_helper0` | — | — | **the only live difference** (see §2) |

**Net: the define tuples are identical.** Nothing in the layer's T=1 tuple is missing from, or extra
to, the standalone's — `FINPF` is on in both, `EPI4` off in both. The two `*_DEFINES` tuples differ in
zero elements; the layer merely moves `FINPF` from the literal into a `+=` at :13359 while attaching an
alternate body.

## 2. The one live code difference: `TIDX`

Diffing the layer's T=1 body against the CAS's `CUDA_SRC` (decode `_MA_ALT_SRCS["moe_helper0"]`, then
map `TIDX → threadIdx.x` and drop its `#define`) gives **27 hunks, all classified**:

| hunks | what |
|---|---|
| 9 | comments and `#else`/`#endif` comment text |
| 10 | inline-asm operand lists re-wrapped (`wgmma` `"+f"(...)` lines); identical after removing whitespace |
| 4 | inert declarations: `constexpr int XP_SK/NST` (defined, never used), `int unit`, `int unit_next, unit_next2` in `Params` (the CAS has neither; the launcher text is identical, so the host never writes them), `#ifndef OKC_NACC` default |
| 4 | inside the source's own `#if OKC_WGS … #endif` (CAS lines 1620-2169): the KPERM3 x-gather and the KPERM3 per-stage scale words. `OKC_WGS=0` at T=1 |
| 0 | unexplained |

So the T=1 build difference between the layer file and the standalone is **one macro**:

```c
// layer T=1 (klr_fci5 MoE_CUDA_SRC/helper0, CAS file's `threadIdx.x` counterpart):
#define TIDX ((int)((threadIdx.x + NT) % NTL))      // NTL = 288 at TINY: 19 call sites
// standalone bs=1:
                                                // no macro: plain threadIdx.x at the same 19 sites
```

`NT = 256`, so the rotation is a bijection that makes **physical warp 0 the WS helper warp** and the
last warp (physical 8) the routing warp; every `tid`, `warp`, `lane`, `wg`, `wt` and named-barrier group
in the body is derived from it. All 19 `threadIdx.x` sites in the standalone correspond one-to-one to
the layer's 19 `TIDX` uses (19 = 19), so `TIDX = threadIdx.x` converts one build into the other.

Consequences the two candidates test:
* **cost**: the rotated id has to survive as a value; `(threadIdx.x + NT) % 288` cannot be a no-op for
  `threadIdx.x ≥ 32` (nvcc reduces it to a compare + select, so ~2 ALU ops per use), and it is used in
  the skinny ring's lane/warp decode, the early-issue branch and the final pass;
* **role assignment**: routing (which reads `topk_ids` and builds the tables, on the critical path at
  T=1) moves from physical warp 0 to physical warp 8, and the helper warp's TMA issues move to warp 0.

## 3. Epilogue: how many bytes, and are they on the critical path at T=1?

The T=1 body is **not** the body that carries the in-layer epilogue additions. `OKC_YBF` (bf16 `ybuf`),
`OKC_MFIN` (per-expert counters, no grid barrier), `OKC_B4L`/`B4FIN` and `OKC_EPI4` live in the layer's
*main* `MOE_CUDA_SRC`, which is compiled for `M ≥ 5`; the T=1 build compiles the helper0 body. In it —
and in the standalone's body, identically — the phase-B epilogue writes **fp32** `ybuf`
(`st_keep_f32`, `float* ybuf; // [EMAX, HD]`) and the final pass re-reads it as `float4`. Same in all
three files.

Per local route (one `(token, route)` pair whose expert this rank owns), M = 1:

| buffer | dtype | write | read | round trip |
|---|---|---|---|---|
| `inter` (phase A → phase B, SwiGLU) | bf16 | 4096 × 2 B = 8 KB | 8 KB | 16 KB |
| `ybuf` (phase B → final pass) | **fp32** | 4096 × 4 B = 16 KB | 16 KB | 32 KB |
| `ss` (+ `ss1` copy) | fp32 | 4 B | 4 B | 8 B |
| `out` (per token, not per route) | bf16 | 8 KB | — | 8 KB |

At bs=1 the rank sees the *local* part of 8 routes out of 256 experts / 32 local: L ≈ 3.5 on average
(the busiest-rank case the board uses), worst case L = 8.

| L | intermediate traffic | at 4.2 TB/s | vs the 22-29 µs kernel |
|---|---|---|---|
| 3.5 | ≈ 0.17 MB | **0.04 µs** | 0.2 % |
| 8 | ≈ 0.40 MB | **0.095 µs** | 0.3-0.4 % |

Weight streaming is the whole cost: L × 28.31 MB ≈ 99 MB → 22-24 µs at the 4.2-4.45 TB/s wall. So the
epilogue's **bytes are three orders of magnitude below** the difference we are chasing, and they are
overlapped with the weight ring anyway: they cannot be on the critical path at T=1. The only
non-byte epilogue costs are the grid barrier before the final pass (all 132 CTAs must arrive; the
`FINPF` item hides the first item's loads behind it) and the final pass itself — `M × FSEG = 4` items
over 132 CTAs, one item each on 4 CTAs. Both are present, identically, in the standalone.

**Where the 29.4 vs 22 µs can come from that is not a build:** the layer runs the MoE with
programmatic dependent launch (`PDL = True`, `PDL_MAX_T = 128` → on at T=1; `okc_set_pdl(1)` at the call
site :13332) so the kernel is resident while its predecessor (the router, which produces `xm`/`lids`/
`wts`) still runs and its profiled duration includes the `griddepcontrol.wait` at the top of the body.
The standalone submission has `PDL = False` (`okc_set_pdl` never turns it on), so its 22 µs starts at
`kernel_fn`'s launch. 5-8 µs of router overlap is exactly the size of the gap; the layer buys wall-clock
with it, which the *layer* board sees and the MoE board does not.

## 4. The two candidates

**A — `src/okc_loop/vendored_work/moe/t1_standalone_build.py`** (`# okc: problem=muse_moe_w4a16`,
copy of the CAS file). `TINY_DEFINES` gains `"@t1_rot"`; `ALT_SRCS["t1_rot"]` is the file's own
`CUDA_SRC` with the layer's `TIDX` macro restored (`_t1_rotated`, 19 substitutions + the macro line).
Nothing else changes: bands M ≥ 5, packing, scratch, PDL stay the incumbent's. Reading: the MoE board
prices **the layer's T=1 build** at bs=1 against 2.3863 with the other five scored cases held fixed, so
the delta is the bs=1 delta (≈ 1/6 of the board score).

**B — `src/okc_loop/vendored_work/layer/t1_layer_build.py`** (`# okc: problem=muse_decode_layer`,
copy of `klr_fci5.py`). `MOE_TINY_DEFINES` compiles `@moe_t1_cas` instead of `@moe_helper0`, where
`@moe_t1_cas` is the helper0 body with `TIDX` defined as `threadIdx.x` (`_t1_plain`) — i.e. the
standalone's bs=1 body, which by §2 differs from the layer's only in that macro. Same define tuple
(`TINY` + `FINPF=1`), same `Params`/launcher/scratch contract, so the epilogue contract is untouched:
the kernel still writes `out[T, HIDDEN]` bf16 through the same final pass and norm. Bands M ≥ 5 are
not touched (TINY2/MID keep `@moe_helper0`/`@moe_helper0df`).

**Winning hypothesis**: plain `threadIdx.x` (the standalone's bs=1 body) is the cheaper T=1 build, so
B is the candidate to send to the layer board. Read-out:
* A **down** at bs=1 → the rotation loses; B should gain the same relative amount on the layer's
  decode-1 cases: those are 2 of the 12 scored layer cases and carry 62/372 of the score weight
  (the `verify 1` cases are 8 query tokens, i.e. the TINY2 band, which neither candidate touches).
* A **flat** → the build is not the 7.4 µs; the PDL-overlap explanation stands, B earns nothing but
  costs nothing (it is the same machine code the standalone already ships), and the next lever is the
  router overlap, not the MoE.
* A **up** → do not send B; flip A's change the other way in the layer (keep `@moe_helper0`) and look
  for the layer's T=1 win in the rotation instead.

## 5. Verification actually run (no `okc`, no GPU)

| gate | result |
|---|---|
| `okc_loop.cli lint` A | **PASS**, 1/1 clean (base CAS: PASS, 0 findings) — **0 new findings** |
| `okc_loop.cli lint` B | 30 errors, 84 findings — **identical multiset** to `klr_fci5.py` (rule+severity+message with digits folded); the only change is line shift +14 (13409 → 13423). **0 new, 0 gone** |
| source-level equality | with `TIDX → threadIdx.x`, A's body vs the layer's helper0 body and B's body vs the CAS body are the same token stream: 27 hunks, all classified (§2), 0 unexplained |
| offline compile of the derived sources | `nvrtc` 12.9 with the TINY tuple: A's patched source and B's patched source both compile (`cu_build.to_ptx`), each with the same PTX size as its unpatched control (2500 / 2653 bytes) — that covers every **non-template** device function the bodies contain (`route`, `ws_helper`, `wgs_*`, epilogues). The `moe_fused` template body is not instantiated by nvrtc (`cu_build`'s documented limit: nvrtc drops explicit instantiations, and the host-side `<<<>>>` route needs non-`-default-device` mode, which rejects this file's `__device__` forward declarations) — for that part the guarantee is §2: each derived body is token-identical to a body **the other file compiles for real at T=1 today** |
| bytes | file sizes 327,375 (A) and 980,433 (B) — both under the 1,000,000-byte budget, max line unchanged |

## 6. Risks

* The layer's `@moe_t1_cas` body is byte-for-byte the standalone's bs=1 body **modulo the layer's own
  weight packing**: `_pack_codes`/`_pack_scales`/`_unit_order`/`_fat_order` are identical (checked
  function by function), `_wgs_copy` differs only by the `sc6` flag (WGS-only, not read at T=1), and the
  `Params`/`okc_run` launcher text is identical, so the packed-layout contract holds. A wrong guess
  here faults or produces wrong numbers rather than running slower — the correctness gate belongs to
  `okc check`.
* `TIDX` is a deliberate design (it puts the helper warp on physical warp 0), so candidate B is an
  A/B of that design, not a pure no-op; if A regresses, B is the wrong direction.
* The 29.4 µs in-layer figure is a *different measurement context* (PDL on, router in flight). Treat
  "the layer's MoE is 34% slower" as unestablished until A reports.
