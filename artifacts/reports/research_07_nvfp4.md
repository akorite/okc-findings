# research_07 — NVFP4 / E2M1 on sm_90a: instructions, CUTLASS, scale layouts, re-encoding

Scope: the four questions in the ticket, for **sm_90a (H200)** only. Two evidence classes are used and
kept apart:
- **MEASURED here**: `ptxas` 13.4.92 (`/tmp/cuda13/usr/local/cuda-13.4/bin/ptxas`) assembling hand-written
  PTX for 8 targets. Reproduce: `.version 8.7 / .target <tgt> / .visible .entry k() { …instr…; ret; }`,
  `ptxas -arch=<tgt> t.ptx -o t.cubin`. This is the compilers' own answer to "does this exist on this chip".
- **CITED**: PTX ISA 9.4 text (fetched, `docs.nvidia.com/cuda/parallel-thread-execution/index.html`),
  CUTLASS 4.8.1 at `/home/ubuntu/kr/cutlass` (commit `e13f5cd`, 2026-10-28), FlashInfer tree at
  `tmpbuild/engines/flashinfer`, and two arXiv papers.

Our own decode, for reference (`klr_attn3.py:4175-4190`, the live `DQ7` path):
`r0 = q&M` (1 LOP3), `r1 = (q*16)&M` (IMAD.SHL+LOP3), `r2 = funnel_l(q,q,10)&M` (SHF+LOP3),
`r3 = ((q*64)&0x80C080C0) | (funnel_l(q,q,14)&0x01000100)` (IMAD+LOP3+SHF+LOP3+LOP3) — **≈9-10 ALU per 8
weights (1.25/weight)**, then 4 `HMUL2` for the two scale pairs → **1.75 ops/weight all-in**, scales via
`cvt.rn.f16x2.e4m3x2` (`klr_attn3.py:6022`) and the SC6W 6-bit unpack (`:5995-6016`).

---

## 1. The hardware instruction set on sm_90a — there is none for E2M1

Measured matrix (ptxas 13.4.92; `OK` = assembles, `no` = rejected):

| PTX instruction | sm_90 | **sm_90a** | sm_100 | sm_100a | sm_103a | sm_110a | sm_120a | sm_121a |
|---|---|---|---|---|---|---|---|---|
| `cvt.rn.f16x2.e2m1x2 %rd, %rb8;` | no | **no** | no | OK | OK | OK | OK | OK |
| `cvt.rn.satfinite.e2m1x2.f32 %rb8, %f, %f;` | no | **no** | no | OK | OK | OK | OK | OK |
| `cvt.rn.satfinite.e2m1x2.f16x2 %rb8, %rd;` | no | **no** | no | OK | OK | OK | OK | OK |
| `mul.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4 %d,%a,%c;` | no | **no** | no | OK | OK | no | no | no |
| `mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4` | no | **no** | no | no | no | no | no | no |
| `cvt.rn.f16x2.e4m3x2 %rd, %rh;` | OK | **OK** | OK | OK | OK | OK | OK | OK |
| `cvt.rn.satfinite.e4m3x2.f32 %rh, %f, %f;` | OK | **OK** | OK | OK | OK | OK | OK | OK |
| `prmt.b32`, `lop3.b32`, `bmsk.clamp.b32`, `dp4a.u32.u32` | OK | **OK** | OK | OK | OK | OK | OK | OK |
| `mma.sync…m16n8k32.f32.e4m3.e4m3` | OK | **OK** | OK | OK | OK | OK | OK | OK |

Cross-checked against the spec, which agrees exactly:
- `cvt.rn{.relu}.f16x2.{e2m1x2/e2m3x2/e3m2x2}` — PTX ISA 8.6 — *"is supported on following architectures:
  **sm_100a**, sm_101a (renamed sm_110a from PTX ISA 9.0), **sm_120a**, and … family-specific architectures
  from PTX ISA version 8.8: sm_100f / sm_101f / sm_120f / sm_110f"*. sm_90/sm_90a is absent from every list.
- `cvt.rn.satfinite{.relu}.{e2m1x2/e2m3x2/e3m2x2/ue8m0x2}.f32` — PTX ISA 8.6 — same target list.
- `add/sub/mul/fma` with `.e2m1p4x4`/`.e2m1x4` as `.atype` — **PTX ISA 9.4** — *"is supported on the
  following architectures: **sm_100a, sm_103a**"*. This is the `mul.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4`
  the ticket asks about: **two targets in the world, neither of them Hopper**.
- `tcgen05.mma … .kind::mxf4nvf4 / .kind::mxf4` — PTX ISA 8.6 (kind 8.7) — *"supported on: **sm_100a,
  sm_101a→sm_110a, sm_103a, sm_107a, sm_110a**"*. Block-scaled FP4 tensor-core MMA is Blackwell-only.
- The warp-level `mma.sync` with `.kind::mxf4nvf4/.kind::mxf4` is **sm_120a/sm_121a only** — i.e. on
  sm_100a you *must* go through tcgen05; on Hopper neither exists.
- **`wgmma` — the only tensor-core instruction sm_90a has — lists `.atype/.btype = {.f16/.bf16, .e4m3/.e5m2,
  .s8/.u8}`. There is no FP4 operand type at all** ("Requires sm_90a", PTX ISA 9.7.17.5.2). So even the
  "dequantize to FP8 then wgmma" route cannot express a 4-bit input.
- `e2m1` register representation: *"e2m1 values must be used in a packed format specified as e2m1x2. A
  register variable containing two e2m1 values must be declared with `.b8` type"* (PTX §"Packed Floating
  Point Data Types", Table 9).
- `viota` is **not a PTX mnemonic at all** — ptxas reports `Not a name of any known instruction: 'viota'`
  on every target. It is a SASS-only op; the PTX-level equivalents are `vote.sync.ballot.b32` and
  `redux.sync`. (Do not budget an instruction for it.)

**Verdict: on sm_90a the only E2M1 path is software** — exactly what we ship. Blackwell-only, and therefore
irrelevant to this board: `cvt.*.e2m1x2` (sm_100a+), `mul/add/sub/fma .e2m1p4x4` (sm_100a/sm_103a),
`tcgen05` block-scaled MMA (sm_10xa), warp `mma.sync … i4` kinds (sm_120a). Available on sm_90a and already
used by us: `cvt.rn.f16x2.e4m3x2` (scales), `prmt`/`lop3`/`shf`/`imad` (codes), `HMUL2`.

---

## 2. What CUTLASS emits on sm_90, and whether it beats us

**There is no sm_90 NVFP4 block-scaled GEMM in CUTLASS 4.8.1.** `include/cutlass/gemm/collective/` contains
block-scaled collectives for `sm100`, `sm103`, `sm107`, `sm120` only (`sm1??_blockscaled_mma_*.hpp`); the
`sm90` files are the plain, FP8 and `mixed_input` variants. The FP4 conversion intrinsics are gated:

```
float_subbyte.h:47-53   #if (SM100A || SM101A || SM103A || SM107A || SM110A || SM120A || SM121A)
                        #   define CUDA_PTX_FP4FP6_CVT_ENABLED 1
numeric_conversion.h:3715  #if defined(CUDA_PTX_FP4FP6_CVT_ENABLED) && __CUDA_ARCH__ >= 1100
                           #  define USE_PTX_CONVERT 1
```

so a Hopper compile takes the `#else` branch: the **LUT path**, `_e2m1_to_bf16_x8` (`:3861-3864`) →
`_e2m1_to_bf16_x4` (`:3796-3859`), which builds each bf16 from a 64-bit PRMT lookup of the exponent bits:

```
and.b32 prmt_ctrl0123, src, 0x7777        shr.b32 …, src, 2      and.b32 …, 0x3333
prmt.b32 target0123, lut_lo, lut_hi, ctrl (×1)   prmt.b32 target0_1_/target2_3_ (×2)
prmt.b32 upper0123 (×1)  prmt.b32 upper_0_1/upper_2_3 (×2)
shl.b32 …,6 (×2)         or.b32 …, … (×2)
```

= **13 ops per 4 values (3.25/value)**, 15 for the `shift_count=16` variant, per 8 values **26-30 ops ≈
3.25-3.75/weight** — *before* the block-scale multiply, which CUTLASS then does as a plain
`__hmul2` per bf16x2 (`detail/collective/mixed_input_utils.hpp:1036-1050`), i.e. ≥0.5 more ops/weight.
On Blackwell the same conversion is 4× `cvt.rn.f16x2.e2m1x2` for 8 values ≈ 1.0/weight — that is the whole
gap, and it is not purchasable on Hopper.

CUTLASS's *fastest* 4-bit register path is not E2M1 at all: `NumericArrayConverter<bfloat16_t, int4b_t, 8>`
(`numeric_conversion.h:7870-7960`) is 1 `shr` + 4 `prmt` + 4 `lop3` + 2 bfma = **11 ops/8 values = 1.375**,
plus the same 0.5 HMUL2 ⇒ **≈1.875 ops/weight vs our 1.75**. Its cheapest published 4-bit dequant is the
W4A8 `lookup_table_convert` (`mixed_input_utils.hpp:763-800`): 1 `lop3` for the sign word + **3 `prmt` per
4 weights**, because the destination is a *byte* (LUT of 8 pre-scaled magnitudes fits in 2 registers).
That trick is unavailable to us: an E2M1→**bf16** codebook is 16 B (4 registers) and `prmt` selects from 8
bytes only, so a 16-entry bf16 table costs ≥2 `prmt` + merge per 2 values — strictly worse than our layout.

**Verdict (Q2): no faster register-only decode than PRMT+LOP3+SHF+HADD2 exists for sm_90a, and ours is
already at or below CUTLASS's best.** Our `_pack_codes` custom nibble permutation is what buys it: fixed
nibble positions make each grab a fixed funnel-shift+mask, while CUTLASS must carry a general PRMT control
word per caller. Do not re-derive CUTLASS's LUT path.

---

## 3. Scale-factor layouts: nothing published reduces the bytes; ours is already the TMA-optimal one

CUTLASS's SF machinery is **a permutation, not a compression**: 1 byte per 16 values either way.

`detail/sm100_blockscaled_layout.hpp:59-73`:

```cpp
struct Sm1xxBlockScaledBasicChunk {          // one atom = 128 MN x 4 k-blocks of SFVecSize
  using Blk_MN = _128;  using Blk_SF = _4;
  using SfKMajorAtom = Layout< Shape<Shape<_32,_4>, Shape<Int<SFVecSize>,_4>>,
                              Stride<Stride<_16,_4>, Stride<              _0,_1>>>;
};
```

so `offset(mn, kb) = (mn%32)*16 + (mn/32)*4 + kb` — the atom is 512 contiguous bytes holding 128 rows × 4
k-blocks, i.e. **one `uint32` per row = the 4 scales of 4 consecutive K-blocks**, which is precisely what a
`tcgen05` block-scaled MMA wants in one 32-bit TMEM word. Its purpose is to make the SF operand a single
contiguous 512-byte TMA box loadable into TMEM — a Blackwell requirement (SF is an *instruction operand*
there, `scale_A: M x 4` per `block16`, PTX Table 68). The only "vector sizes" the hardware accepts are the
two MX ones: `block16 ≡ scale_vec::4X` (NVFP4, 1 scale / 16) and `block32 ≡ scale_vec::2X` (MXFP4, 1
scale / 32) — PTX Table 68/69. Coarsening to `block32` halves the SF bytes but changes the quantization
(needs the fp32 weights to requantize; we only have the codes) so it is lossy and off the table.

For us: our `w13_sf`/`w2_sf` are plain `[rows][256]` e4m3 = one byte per 16 values = **the same information
at the same size**, already re-ordered per-unit by `_pack_scales`/`_fat_order` (`klr_attn3.py:7060-7076`) so
that each unit's scale section is one contiguous `cp.async.bulk` and each thread's 8 scales are an 8-byte
load. On sm_90a there is no SF operand and no TMEM, so the CUTLASS atom's only property (a 32-bit word = 4
scales of one row) buys us nothing we don't already have from KPERM3 + `_fat_order`. **Adopting
`Sm1xxBlockScaledConfig` would change the TMA box from a 4 KB contiguous 16-row×256-B piece to 64 boxes of
[128 rows × 4 B] — a 4-byte-row 2D box, i.e. worse.** Reject; the layout question is closed.

The only published *byte* reductions for the scale stream are (a) MX/E8M0 coarsening (lossy, above) and
(b) entropy coding, §4.

---

## 4. Anything cheaper than 4 bits per weight? Codes: published answer is *no*; scales: *yes, 2:1*

Exact per-expert composition (D_HIDDEN 8192 / D_EXPERT 4096, top-8, 4-bit codes + e4m3 per-16 scales):

| component | bytes | share |
|---|---|---|
| w13 codes (8192×4096 @4b) | 16,777,216 | 59.3% |
| w2 codes (4096×4096 @4b) | 8,388,608 | 29.6% |
| w13_sf (8192×256) | 2,097,152 | 7.4% |
| w2_sf (4096×256) | 1,048,576 | 3.7% |
| **total** | **28,311,552** (=28.31 MB ✓) | |

So **codes 88.9%, scales 11.1%** — the scale stream is the only place with more than 0.8 MB to find.

**Codes (25.17 MB): closed, on published measurement.** Heilper & Choukroun, *Lossless Compression of
Neural Network Components…* (arXiv:2508.19263v1, §3.4 + §4.4) tested exactly this: they concatenated 2 bits
from each of 4 consecutive FP4 values into byte-aligned streams for MXFP4 **and NVFP4**, and report *"did
not yield meaningful compression. The resulting streams appeared statistically uniform… we conclude that
the quantized values in FP4 do not exhibit consistent statistical redundancy and are not suitable for
lossless compression"*, so their scheme *"targets only the scaling factors and stores the quantized values
uncompressed."* This matches our own independent negative result (roster: Hopper inline memory compression
on dense weights ≈ 0). A hard ceiling also exists analytically: E2M1 has 15 distinct values (±0 merge
losslessly), so even a perfectly uniform code stream caps at `log2(15)/4 = 0.9767`, i.e. **2.07% of the
expert** — and reaching it needs ≥3-4 ops/symbol of entropy decode against a measured deletion value of
≤0.9% for the entire existing unpack.

**Conflict to note, not to act on.** `github.com/drxddy/ecq` reports 4-bit weight entropy of 1.12-1.54 bits
(3.27× "lossless" compression) on Qwen2.5/GPT-2/SmolLM. That is a 3× disagreement with the measured NVFP4
result above; ECQ's models are ≤1.5 B and its numbers are incompatible even with a 15-symbol alphabet's
`log2(15)`, and NVFP4's per-16 block scale pins the block max to code ±6, which flattens the code
distribution by construction. Treat ECQ as unverified. A **20-line `prepare()` census** (per-bit entropy of
the 4 code bits + histogram) settles it for free — do that before ever writing a decoder.

**Scales (3.15 MB): the lever, with two independent confirmations.**
- Published: same paper, §4.4 + Fig. 9 measures NVFP4 scales on DeepSeek-R1 at 39 GB of a 395 GB dataset and
  reports Huffman coding takes the dataset to *"5% … as the scaling factors composed 10% of the overall
  dataset"* — i.e. **the SF stream halves (~2:1)**, a 5.6%-of-expert byte win, while the codes give nothing.
- Ours: `_sc6_ok` (`klr_attn3.py:7071`) requires every SF byte ∈ `0x40..0x7F`, and `v = byte-0x40` is a
  lossless 6-bit index — measured +2.2-3.3% on the MoE board for 8→6 bits (2.78% of expert bytes).
  The published 2:1 ⇔ 4 bits/scale is the *next* step: 6→4 bits is another 786,432 B/expert = **2.78%**.

Ceiling on that direction: at 4 bits/scale the total stream is 4.11 bits/weight (codes 4.0 + scales 0.5). A
3-bit index (needs ≤8 distinct SF bytes) would reach 5.56% total saving but is almost certainly too
aggressive given `0x40..0x7F` is the *entire* positive e4m3 range ≥2.0 — the 64-value window is not
evidence of a narrow alphabet, which is exactly why the census must come first.

---

## 5. Ranked top-3, by expected value on the MoE board (1% ≈ 101 ranks)

Leverage anchor: SC6W's own measurement, 2.78% of expert bytes → +2.2-3.3% board, i.e. **0.79-1.19 board-%
per byte-%**.

| # | lever | bytes/expert | board % | ranks | evidence / gate |
|---|---|---|---|---|---|
| 1 | **SF 6→4 bits**: replace `v = byte-0x40` with an index into a per-tensor ≤16-byte alphabet table, shipped once per expert | −786,432 B = **−2.78%** | **+2.2 to +3.3** | **+222 to +334** | arXiv:2508.19263 §4.4 (2:1 on real NVFP4 scales) + our own SC6W slope. Gate: `len(unique(sf)) ≤ 16` per layer, measured in `prepare()`; else 5-bit (needs ≤32, +1.4%) or top-16 + 6-bit escape channel. Decode cost ≈ neutral (one table select per 4 scales replaces the current 8-ALU 6-bit reassembly). |
| 2 | **`OKC_SC6S` — the same 8→6 bit cut on the skinny/tiny builds**, where SC6W is inert (WGS-only, M≥17) | −1,024 B/unit = **−2.78%** of every skinny load | **+0.68 layer board** (prior report's own price, `layer_moe_bytes.md` §A); if the timed M ≤ 16 this is the *whole* scale lever on that path | ~+69 layer-equivalent, ~+220-330 MoE-equivalent in the small-M band | already ported + modelled; blocked only by `_sc6_unit` never being set in `prepare()` and the fat-unit scale section not being repacked (`moe_klrbands.md` §4). No new research, pure completion. |
| 3 | **E2M1 decode** (any further instruction-level work) | 0 | **≤ +0.9** | **≤ +90** | Independent deletion test (nearly all unpack + half the scale multiplies = +0.5-0.9%); no sm_90a instruction exists (§1); ours already ≤ CUTLASS (§2). Close the axis. |

Note on #1's arithmetic: 8→6 bits gave 2.78%; 6→4 gives the same 2.78% again because the scale stream is
2^15 bytes for 2^12 scales — halving the bits always halves the stream. Going all the way 8→4 is 5.56% of
expert bytes, matching the published 2:1.

Two cheap carries, not ranked: (a) the code-stream census of §4 — 20 lines, expected 0, resolves a 3×
literature disagreement and would be worth up to +9% if ECQ were ever right; (b) nothing else in §1-§3 is
actionable on sm_90a.

---

## 6. Risks / what would falsify this

- #1 depends on the SF alphabet actually being ≤16. `_sc6_ok`'s `0x40..0x7F` window is *not* evidence of
  that (it is the full positive e4m3 range from 2.0 up), so the census is a hard prerequisite, not a
  formality. If the alphabet is 17-64, use the published Huffman result as the fallback *design* but expect
  its decode cost to eat the win on GPU (this is the same failure the 10.9-bit BF16 codec measured,
  −6..−11% for an entropy decode against a byte saving that did not cover it).
- #2's blocker is a repack, not a flag: enabling `OKC_SC6S` with the 4,096-byte fat section live reads 3,072 B
  as 4,096 B and faults (`moe_klrbands.md` §4). Fault, not a wrong number — treat as a code change.
- `layer_moe_bytes.md` §A's unit table has an arithmetic slip (it lists the 6-bit unit as 34,816 B; 32,768 +
  3,072 = 35,840 B). The 2.78% it uses is correct; don't propagate the absolute.
- Byte savings assume the kernel stays bytes-bound. That is measured at large T (SC6 slope ≈1:1) and
  *assumed* at bs 1-16, where the roster says the MoE is per-unit/fixed-cost bound — #2 is precisely the
  experiment that tests the small-T half of that assumption.

## 7. Sources

- PTX ISA 9.4, `docs.nvidia.com/cuda/parallel-thread-execution/index.html`: §"Packed Floating Point Data
  Types" Table 9 (`e2m1x2` = `.b8`); §9.7.1 `cvt` (target lists + PTX-version notes for `e2m1x2`, 8.6);
  §9.7.6.3 `mul` with `.e2m1p4x4` (PTX 9.4, sm_100a/sm_103a); §9.7.17.5.2 `wgmma.mma_async`
  (`.atype={.f16/.bf16,.e4m3/.e5m2,.s8/.u8}`, requires sm_90a); §9.7.18 `tcgen05.mma` kind::mxf4nvf4 target
  list; Table 68/69 (`scale_vec`/`block16`/`block32` shapes).
- Measured: ptxas 13.4.92 sm_{90,90a,100,100a,103a,110a,120a,121a} matrix in §1.
- CUTLASS 4.8.1 `/home/ubuntu/kr/cutlass`: `numeric_types.h:76-80`; `float_subbyte.h:47-53,83`;
  `numeric_conversion.h:3715, 3796-3859, 3861-3864, 4231-4234, 7870-7960`;
  `detail/sm100_blockscaled_layout.hpp:59-73,85-130`; `detail/collective/mixed_input_utils.hpp:763-800,
  832-860, 1036-1050`; `gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp:875,891,906`;
  `gemm/collective/sm1{00,03,07,20}_blockscaled_*.hpp` (existence set).
- FlashInfer `tmpbuild/engines/flashinfer/csrc/cake_fmha/cuda/decode_quant_nvfp4/default.cu:94-130`
  (`mul.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4` gated on `__CUDA_ARCH_SPECIFIC__ == 1000 || 1030`; the
  portable 8-instruction-per-4-value chain for everything else) — independent agreement with §1.
- A. Heilper, B. Choukroun, *Lossless Compression of Neural Network Components: Weights, Checkpoints, and
  K/V Caches in Low-Precision Formats*, arXiv:2508.19263v1, §3.4, §4.4, Fig. 9, §5, §6.
- `github.com/drxddy/ecq` README (the conflicting 1.12-1.54-bit claim).
- Repo: `klr_attn3.py:4175-4190` (`dq4`), `:5995-6016` (SC6W), `:6022` (`cvt2h`), `:7060-7115`
  (`_pack_scales`/`_sc6_ok`/`_sc6_unit`/`_sc6_wgs`), `:4134-4140` (`bmul2l/h`);
  `artifacts/reports/moe_klrbands.md`, `layer_moe_bytes.md`, `prior_art_trxphxle.md`, `klr_df.md`.
