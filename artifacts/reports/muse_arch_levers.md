# Muse Spark Open decoder layer — architectural levers a general kernel cannot see

Base row: `src/okc_loop/vendored_work/layer/klr_fci5.py`, judge **2.28370 / 9,905.1 µs**, job
`9743b6e71fbd42fd`, profile `src/okc_loop/vendored_work/layer/profiles/klr_fci5.9743b6e71fbd42fd.txt`
(12/12 scored shapes + 9 per-kernel rows per shape). Every µs in this file is from that profile or from
`problems/muse_decode_layer/{TASK.md,reference.py}`, or is marked `[DERIVED]` where I computed it from those.
Nothing here was re-derived from `prior_art_trxphxle.md` that is already in it.

**Bottom line up front.** No wasted bytes and no architectural redundancy left to delete: at all 12 scored
shapes the layer moves *exactly* the judge's floor byte set (byte-for-byte on the dense term; the only excess
anywhere is the MoE's measured 1.15× at v32) and achieves 2.24–3.43 TB/s of it, 47–72% of 4.8 TB/s. The
components separate cleanly by achieved rate — dense **3.0–3.6 TB/s**, full attention **3.3–3.7**, boundary
glue **4.1**, MoE **1.9–2.7** — and the two levers that are real are not bandwidth levers at all: **the
attention kernel's 6.3–8.3 µs fixed cost on the path that the 46 sliding layers are forced onto (+1.5%)**, and
**6-bit NVFP4 block scales on the three skinny MoE builds (+0.7%)**. The MoE's 2.4 TB/s is 4× bigger than both
together (+14.5% if brought to the dense chain's rate) and has no lever left in the source.

---

## 1. The architecture, corrected and cited

Per GPU under TP8/EP8 (`problems/muse_decode_layer/reference.py`):

| item | value | line |
|---|---|---|
| hidden | 8192 | `:29` |
| head_dim | **64** | `:30` |
| q heads / kv heads on this shard | 16 / 2 → **GQA 8:1** | `:31` |
| `Q_SIZE, KV_SIZE, QKV_SIZE` | 1024, 128, **2304** | `:32-33` |
| sliding window | 2048 | `:34` |
| RoPE theta | 500000 (**sliding only**) | `:35`, `:317-318` |
| qk scale factor folded into the softmax scale | 5.4730064863875 | `:36`, `:309` |
| RMS eps / post-norm eps | 1e-5 / **1e-8** | `:38-39` |
| page size | 64 default, **1 scored**; cache shape `(total, page, HKV, 64)` | `:40`, `:270` |
| experts | 256 total, **32 local**, top-8, `D_EXPERT = 4096` | `:45-48` |
| router multiplier | 0.1767766952966369 (logit scale) | `:51` |
| residual gate temperature / eps | 0.3 / 1e-3 | `:54-55` |
| unpooler slice at T ≤ 8 | 1024 of 8192 columns, rest **exactly zero** | `:57-58`, :469-470 |

Five corrections to the brief's summary, each of which changes a byte or a tolerance:

1. **`window=0` and `rope_theta=0.0` on full layers** (`:317-318`) — the full-attention layers are **NoPE**:
   no rotation is applied to q or k at all. Confirmed by HF: *"`layer_rope_theta` … Defaults to the global
   theta everywhere except every 4th layer counted backward from the last, which is NoPE"* ([HF
   MuseSparkTextConfig](https://huggingface.co/docs/transformers/en/model_doc/muse_spark)). 62 layers with
   every 4th-from-the-last NoPE = **16 NoPE + 46 RoPE**, which is *exactly* the judge's 46:16 weighting
   (`TASK.md` shapes table). The board's weights are the model's own layer census, not a judge artifact.
2. **The residual gate is norm-preserving in a specific way** (`:339-345`): `beta = sigmoid(g/0.3)`,
   `alpha = sqrt(clamp(sigmoid(-w)*(1+beta), 1e-3))`. `alpha²+beta² ≠ 1`; the invariant is
   `alpha² = sigmoid(-w)(1+beta)`. Note `alpha` can be **clamped at 1e-3 for w ≥ ~+8**, i.e. for very
   negative gate entries the branch is suppressed but not zeroed — a kernel that assumes `alpha>0` always
   holds, one that assumes `alpha` unclamped does not.
3. **`post_expert_norm` is scale-then-norm** (`:427-431`): `y ← y·(1+w)`, *then* an RMS over the 4096-wide
   MoE width with eps **1e-8** (not 1e-5). The RMS is therefore of a column-weighted y, i.e. the quadratic
   form `‖y ⊙ (1+w)‖²/4096` — not `‖y‖²` with a post-scale.
4. **The router is FP64 in the reference** (`:419-425`), and the acceptance slack is enormous relative to the
   score spread: `ROUTE_TOL = 5e-2` (`:72`) on biased scores whose own std is ≈0.14 `[DERIVED: sigmoid of
   N(0,0.72) × 0.25 slope ≈ 0.14; bias std 0.055, `:84`]`. So a single-sigmoid-unit error is 3× the whole
   tolerance budget, and `KEEP_TOL = 0.03/0.04` (`:75-76`) against an 8th-vs-9th gap of ~0.01–0.02 `[DERIVED]`
   means the "must keep" set is nearly always empty.
5. **`local_start` is a runtime input** (`:48`, `:480-486`) and experts are `(ids − local_start) % 256`
   (`:468`), so the four-in-256 slab↔id mapping rotates per call. Only ~1/8 of the top-8 routes are local.

Global (full-model) config, from the HF text-config defaults: hidden 8192, **128 q heads / 16 kv heads**,
head_dim 64, 62 layers, `moe_hidden_size 4096`, `moe_intermediate_size 4096`, 256 experts top-8,
`sliding_window 2048`, `qk_scale_factor 5.4730064863875`, `output_multiplier 0.1767766952966369`,
`post_norm_eps 1e-8`, `residual_gate_temperature 0.3`, all-sparse MLP. The per-GPU numbers in `reference.py`
are this divided by TP8 on the head axis and EP8 on the experts (`:8-13` of TASK.md: "Communication is left
out"). Parameter check `[DERIVED]`: 62 × (8192·8192 qkv + 8192·8192 o + 2·8192·32768 pool/unpool + 256·3·4096²
experts) ≈ 841 B ≈ the 816 B release.

### Byte budget the judge actually charges (`reference.py:681-697`)

```
dense   = 2304·8192 + 8192·1024 + 4096·8192 + unpooled·4096     (unpooled = 1024 at T≤8, else 8192)
bytes   = 2·dense + 4·256·8192 (router FP32) + touched·3·4096²·(0.5+1/16) + kv_tokens·128·2 + tokens·8192·12
```

Our kernels read exactly this set — the `2·dense` term is 197.1 MB at T>8 and 138.4 MB at T≤8, and the
profile's per-kernel byte account reproduces both to within the 12-bit encoder's 25% `[DERIVED from the
profile's kernel times × the achieved rates below]`. The one place we read **more** than the floor is the MoE,
which the standalone board measured at **622 MB fetched for 540 MB touched at v32 (1.15×)**
(`kernarc_digest.md`, `moe_struct.md`). So: no deletable byte except that 15%, and the 12-bit encoder.

---

## 2. The measurement base

### 2.1 Per shape (µs, from the profile) and the achieved rate per component

`sens` = board-% per µs saved on that shape, `100/6 · w/(62·t_load)` with `t_load = (46·t_S + 16·t_F)/62`.

| shape | ours | floor | ×floor | dense | attn | MoE | glue | **sens %/µs** |
|---|---|---|---|---|---|---|---|---|
| D1S | 66.68 | 40.02 | 1.67 | 3.39 TB/s (138 MB) | 0.48 MB, 5.3 µs — latency | 53.1 MB, **2.24** | 4.5 µs | **0.1799** |
| D1F | 74.69 | 44.18 | 1.69 | 3.46 | 18.9 MB, 1.65 | 54.6 MB, 1.86 | 4.8 | 0.0626 |
| D16S | 120.09 | 76.72 | 1.57 | 3.57 (197 MB) | 7.8 MB, 0.98 | 161.8 MB, 2.55 | 11.0 | 0.0883 |
| D16F | 197.53 | 137.46 | 1.44 | 3.24 | 303.1 MB, 3.52 | 158.0 MB, 2.48 | 11.3 | 0.0307 |
| D32S | 138.02 | 84.51 | 1.63 | 3.02 | 15.5 MB, 1.63 | 189.9 MB, 2.20 | 10.7 | 0.0696 |
| D32F | 291.21 | 208.10 | 1.40 | 3.11 | 606.2 MB, 3.72 | 192.4 MB, 2.29 | 12.4 | 0.0242 |
| V1S | 106.53 | 62.29 | 1.71 | 3.56 | 0.48 MB, 6.6 µs | 160.0 MB, 2.27 | 6.8 | **0.1137** |
| V1F | 115.32 | 66.44 | 1.74 | 3.58 | 18.9 MB, 1.33 | 161.5 MB, 2.32 | 8.0 | 0.0395 |
| V16S | 238.37 | 126.61 | **1.88** | 3.07 | 7.8 MB, 0.66 | 401.3 MB, 2.59 | 12.3 | 0.0475 |
| V16F | 323.96 | 188.68 | 1.72 | 3.13 | 303.1 MB, 2.97 | 403.9 MB, 2.73 | 11.8 | 0.0165 |
| V32S | 307.78 | 143.81 | **2.14** | **2.31** (peak-bound) | 15.5 MB, 1.03 | 474.5 MB, 2.54 | 17.7 | 0.0353 |
| V32F | 471.18 | 267.18 | 1.76 | 2.34 | 606.2 MB, 3.34 | 476.0 MB, 2.39 | 17.8 | 0.0123 |

`glue` = `glue_cluster_boundary` + `_graft_row_boundary` + `_sl_cluster_finish` + `_sl_tile_route` +
`_gle_route` + `_graft_route` summed from the profile.

The **attn** column's TB/s is a raw µs÷bytes ratio and is only meaningful above ~100 MB: below that the
kernel is dominated by a fixed cost, so §3/A1 refits the same twelve numbers as `t = fixed + S/R` and that
fit — not the ratio — is what any attention lever must be argued against.

### 2.2 Where the board's per-µs value lives (the judge-arithmetic table)

`Σ sens = 0.7201 %/µs` — i.e. 1 µs off **every** shape is +0.72%, and 1 µs off every *sliding* shape is
+0.534% while 1 µs off every *full* shape is +0.187% (ratio 46:16, as designed).

**The ranking is steep and it is not the ranking of the kernels' own sizes:**

| µs saved on | worth | µs saved on | worth |
|---|---|---|---|
| D1 sliding | **0.180 %** | V16 sliding | 0.048 % |
| V1 sliding | 0.114 % | V32 sliding | 0.035 % |
| D16 sliding | 0.088 % | D32 full | 0.024 % |
| D32 sliding | 0.070 % | V16 full | 0.017 % |
| D1 full | 0.063 % | V32 full | **0.012 %** |

**1 µs at D1-sliding is worth 14.6× the same µs at V32-full.** So:
* a 5 µs kernel on D1S (where the whole attention kernel is 5.3 µs) is worth **42%** as much board as a
  181.7 µs kernel on V32F — exactly the "5 µs ≈ 300 µs" effect the brief suspected;
* the four B∈{1} shapes (D1S, D1F, V1S, V1F) carry **0.396 of 0.720 = 55% of the board's total per-µs
  value**, and they are the four shapes where the layer is *latency*-dominated (192–319 MB in 67–115 µs);
* the four V ∈ {16,32} shapes carry **0.086**, i.e. 4.6× less per µs — and that is where most of our recent
  engineering went (WGS MoE, verify megakernels, the cost tables).

### 2.3 Component weight on the board ("if this kernel were free")

| | board-% if free |
|---|---|
| MoE | **51.3** |
| dense (qkv/o/pooler/unpooler + router) | **36.8** |
| attention | 15.9 |
| boundary glue + route | 6.2 |
| Σ | 110.2 (the 10% excess is PDL overlap: a kernel's GPU time includes its wait for the previous one) |

---

## 3. The levers

### ACCEPTED

#### A1. Every attention cell is dominated by a *fixed* cost, and the fixed cost is 40–120% of the cell — **+1.3 to +1.9%**

`klr_fci5.py` compiles `attn_kernel<L, WS, MODE>` and picks `WS = false` for **every sliding shape and for
both B=1 full shapes**, `WS = true` only for full at B≥16. The gate at `:3795` is structural, not tuning:
`if not window and 1 < B <= 512:` — the flat path needs one contiguous key span ("equal shares of all
requests' keys, one CTA pass per SM pair"), so **all 46 sliding layers are locked into the per-request
`grid = (n_splits, B)` split-merge path** (`:3087` grid, `:3101` `chunk = fdiv_i(n_keys + n_splits - 1,
n_splits)`). (Template arg 2 in the profile's mangled names: `<1,false,13>` D1S, `<1,false,11>` D1F,
`<1,false,1>` D16S/D32S, `<8,false,15>` V1S, `<8,false,11>` V1F, `<8,false,3>` V16S/V32S, `<1,true,0>`
D16F/D32F, `<8,true,0>` V16F/V32F.)

The sliding cells are small by construction: the window is bounded at 2048 (`:34`, `:383-384`
`first = max(0, seq - window + 1)`, and every scored length is 74–86 K ≫ 2048), so per request the read is
*exactly* 2047 past + L new positions = 524 KB — 16.8 MB at B=32, 8.4 at B=16, 0.52 at B=1.

**Fit each path as `t = fixed + S/R` over the two sizes it runs at** (B=16 and B=32, `[DERIVED]` from §2.1):

| path | fixed | marginal | as TB/s | cells it runs on |
|---|---|---|---|---|
| WS, q_len 1 | **9.20 µs** | 0.253 µs/MB | 3.95 | D16F, D32F |
| WS, q_len 8 | **22.3 µs** | 0.263 µs/MB | 3.80 | V16F, V32F |
| non-WS, q_len 1 | **6.30 µs** | 0.206 µs/MB | 4.85 | D16S, D32S |
| non-WS, q_len 8 | **8.30 µs** | 0.438 µs/MB | 2.28 | V16S, V32S |

Two readings of that table, and the second is the one that matters:

1. **The per-request path is not slow per byte — it is the fastest of the four.** Doubling the sliding gather
   from 7.8 MB to 15.5 MB costs only **1.6 µs** (q_len 1) / **3.4 µs** (q_len 8). So there is no bandwidth
   headroom to chase on the sliding cells: `D32S` at "1.63 TB/s" is not a 3× opportunity, it is a 6.3 µs
   constant plus 3.2 µs of streaming.
2. **What it does have is a 6.3–8.3 µs constant, against cell totals of 5.3–15.1 µs.** Every one of the eight
   non-WS cells is 55–126% fixed cost:

| | D1S | V1S | D16S | V16S | D32S | V32S | D1F | V1F |
|---|---|---|---|---|---|---|---|---|
| total µs | 5.3 | 6.6 | 7.9 | 11.7 | 9.5 | 15.1 | 11.5 | 14.2 |
| fixed ÷ total | **119%** | **126%** | 80% | 71% | 66% | 55% | 55% | 58% |
| board %/µs | 0.180 | 0.114 | 0.088 | 0.048 | 0.070 | 0.035 | 0.063 | 0.040 |

3 µs off that constant on all eight cells is **+1.91%**; 2 µs is **+1.27%**. The constituent costs are already
enumerated on this kernel (`pdlkv.md` §1): launch+dispatch ≈2 µs, page-index fetch ≈0.7, the 16 KB KV tile
≈0.8, q-load + QKᵀ/softmax/PV ≈1.2, and the `red_v4` merge / `bar.sync` / atom.acq_rel counter / last-CTA
`ws_acc` re-read / finalize ≈**2.5–3 µs**. At B=1 the gather is 0.5 MB, so 100% of D1S's 5.3 µs is those five
round trips.

**Why a general kernel cannot see it:** a decode-attention kernel is sized for "S is large, stream it" and
amortises a 9.2 µs prologue over 100–300 µs of bytes. Here 46 of 62 layers (72% of every load's weight) run a
*fixed* 524 KB gather where the same prologue is never amortised. The fixes, in order of expected value:
(i) for S ≤ 2056 the split axis buys nothing — collapse it to (request, kv_head) with one merge owner instead
of `red_v4` over `n_splits = min(tiles, 2·sms)`; (ii) give the split that owns position ≥ seq−64 the final
tile so the merge is a write-through; (iii) at B=1 there is only a 2048-entry index walk and 0.5 MB — that is
a 2 µs kernel, not a 5.3 µs one.
Risk: `pdlkv.md` measured two attempts at re-ordering *this exact prologue* (variants A, B) at −0.9%/−1.7%
because they lengthened the `cp_wait` drain — the *split/merge* axis is the untouched part of the same kernel,
and an A/A gate (±0.3%) is mandatory before a check.

#### A2. `OKC_SC6S` — 6-bit NVFP4 block scales in the three *skinny* MoE builds — **+0.68%**
Site: `klr_fci5.py:4363-4369` `#define OKC_SC6S 0` / `UNIT_SC = OKC_SC6S ? 3072 : 4096`, with `UNIT_BYTES`
already fully parameterised and the decode already an opaque IMAD (`:4381-4390`). The WGS twin
(`OKC_SC6W`) **already ships and is added automatically by the build** (`:6914-6915`), so the WGS loads (d32,
v16, v32) already pay 6-bit scales — **d1, v1 and d16 are the only three still carrying 4,096 B of FP8 scales
per 36,864 B unit.** One unit = 32,768 B of codes + 4,096 B of scales; 6 bits per scale is 3,072 B →
**−2.78% of every byte those three loads read**.

**Why this is architectural and not a codec hack:** NVFP4 as the judge defines it is
`x ≈ e2m1 × block_scale(E4M3) × global_scale(FP32)` (`nvfp4.py:1-6`) — three fields, and the format's bit
budget is 4 + 8/16 for the first two. Once the FP32 global scale has hoisted the exponent, the *stored*
per-block byte is confined to one exponent class, **0x40..0x7F** — a 64-value window, i.e. exactly 6 bits, and
`:7066` `_sc6_ok` is precisely the losslessness predicate for that ("bits 7:6 == 01, so byte − 0x40 is a
lossless 6-bit value"). The generator's `SCALE_LOG2_STD = 0.4` (`nvfp4.py:17`) makes the block scales span
`2^±1.2 ≈ 5.3×`, which at 3 mantissa bits is ≈42 codes ≈ 6 bits — so 6 is the exact width and a 4-bit field
(16 codes ≈ 2×) would be lossy. 4.5 bits/element is the judge's own floor model (`0.5 + 1/16`); **4.375 is
reachable, and is the only place in this problem where we can go below the judge's floor.**

`[DERIVED]` from §2.1, applying −2.78% to the MoE time of the three skinny loads (moe_struct §0 established
time ∝ bytes on exactly these loads: 2.20/2.64/2.76 TB/s):

| | D1S | D1F | V1S | V1F | D16S | D16F | total |
|---|---|---|---|---|---|---|---|
| µs | −0.7 | −0.8 | −2.0 | −1.9 | −1.8 | −1.8 | |
| board % | 0.12 | 0.05 | 0.22 | 0.08 | 0.16 | 0.05 | **+0.68** |

Risk: `_sc6_ok` must gate per band and fall back — a scale byte outside the window decodes a block off by
≥2×, which is wrong output, not a small error; the `uslot`/barrier layout assumes a constant unit stride (`:4107` `uslot[8]`) so the
ring arithmetic must take `UNIT_BYTES`, which it already does; and the prepare-time repack mirrors
`make_sc6.py` for the `w13/w2` unit-contiguous copies. Effort: one flag, one prepare-time repack, one check.

#### A3. The MoE's 1.15× read amplification at v32 — **+1.2%**
Our own measurement (`kernarc_digest.md` §a, `moe_struct.md` §0): `v32s` MoE "reads **622 MB** for ~**540 MB**
of touched experts, 3.19 TB/s, L2 81%, top stall `not_selected` 1.3/issue". 540 MB is exactly the judge's
floor for 18.5 experts, so the 82 MB is pure re-read. The stage account in `moe_struct.md` says where:
"one stage = 36,864 B of weights … **plus `npad·512 = ntt·4096` B of x, refetched per stage**". The 15%
appears only where `ntt = 2` is populated. With `tok/expert ~ Poisson(T·8/256)`, `P(>8)` is 1e-6 at T=32,
**2.1% at T=128 (V16 — so V16 gets essentially nothing)** and **41–44% at T=256 (V32)**, which is exactly the
`56% ntt=1 / 44% ntt=2` job mix `moe_struct.md` §0 reports for v32.

| | V32S | V32F | total |
|---|---|---|---|
| µs | −24.3 | −26.0 | |
| board % | 0.86 | 0.32 | **+1.18** |

The fix is a smem trade the file has already made elsewhere: the ring holds
`WGS_NS(3)×36,864 + WGS_NX(3)×32,768 = 204 KB`/CTA (`:5547-5555`: `WGS_NS = 3`, `WGS_NX`, `WGS_SLOTS = 160`) where 25 KB/SM of bytes-in-flight
saturates 4.8 TB/s (`moe_struct.md` §0), and `OKC_NS = 4` was measured at nothing — so **ring depth is free
currency**. Trading one weight slot (36.9 KB) for the job's resident x (≤ 14 tokens × 4096 × 2 B = 115 KB at
V32; 8 KB at D16) buys 15% of bytes at a prefetch depth whose marginal value is measured at 0.
Risk: 115 KB does not fit alongside a 108 KB weight ring in 226 KB — this works at V32 only if x is held at
4-bit/e8m0 or split across two stages; and `WGS_SLOTS = 160` (`:5555`) constrains `grid`.

#### A4. Boundary-epilogue folding on the four D16/D32 shapes — **+1.9% (prior-art measured, not ours)**
Not an architecture lever; included because its board value is 2× what `prior_art_trxphxle.md` implies.
They measured "every deletable dense-projection epilogue + consumer cost = **+1.0% decode at D16/D32**
(+2.75% over all four affected loads)". Applying 2.75% per shape to all four: `+0.62 + 0.37 + 0.47 + 0.41 =
+1.87%`. Our glue *runs at the wall already* (V32S: 14.1 µs for 2 × 28 MB = **4.1 TB/s**), so only the
round-trip is deletable, never the bytes. Prior art also fixes the ceiling: deleting a whole glue launch
saves 1.2–2.0 µs because the next GEMV streams 2 pre-wait weight stages during it, and side-stream junctions
cost ≥2 µs each.

### REJECTED, with the number that kills it

| candidate | verdict |
|---|---|
| **GQA 8:1 — batch the 8 q heads sharing a KV head into one MMA so a position's 128 B key is read once** | **Already exploited; nothing to recover.** A position's key for *both* KV heads is 128 B and both K and V are 256 B (`:32`, page 1). D32F reads 606.2 MB in 162.8 µs = **3.72 TB/s**; per-q-head reading would be 4.85 GB and 8× the time. The MMA is m=8 (8 heads × 64) against a WGMMA m=64, i.e. 12.5% tensor-core use — irrelevant, the cell is 116 B/flop. |
| **Sliding 2048 — is our KV traffic exactly the bound?** | **Yes, exactly.** `:383-384`: `first = max(0, seq-window+1)`; every scored length is 74–86 K ≫ 2048, so the read is `2047+L` positions × 256 B = 524.0 KB/request, 16.8 MB at B=32, 8.4 at B=16, 0.52 at B=1. **And it is not the only unbounded term:** the full cells are 18.9 / 303.1 / 606.2 MB — the full layer is un-bounded in ctx and the sliding one is O(1). The unbounded term is already at its gather wall (3.3–3.7 TB/s); the bounded term's problem is not traffic but a 6.3–8.3 µs constant → A1. |
| **Per-expert post-norm + weighted sum folded into the expert epilogue** | **Already fused** — one `moe_fused` launch for the whole block, with `fused_post_expert_norm_weighted_sum` called directly (`:10881`) and `g_postexp` formed as `1+w` in-kernel (`:11219`, `:10218`). And the *cross-expert* sum genuinely cannot fold: `:427-436` normalises each (token, expert) pair with its own RMS before the weight and the sum, so 8 different scalars must be applied to 8 different vectors. Ceiling if it could: 0. |
| **The two residual boundaries and the pre-ffn / pre-expert norms at T ≤ 8** | **Priced and at the wall.** D1S glue = 4.5 µs of 66.68 (0.81% of board); V1S 6.8 µs (0.77%). They run at 4.1 TB/s (V32S 14.1 µs / 56 MB). Prior art already bounded the *deletable* part: 1.2–2.0 µs per launch, all three at B=1 = 4.0–4.8 µs ⇒ ≤ +0.8% board at D1 and less elsewhere. Folded into A4. |
| **`prepare()` has 300 s — what else is precomputable?** | **Enumerated, and it is already fully spent.** What is done: 12-bit lossless BF16 packing for qkv/o/pooler/unpooler + hi/lo router (`:9400`, `:9634`, `:10152-10174`, `BFP_MAX_T = 32` at `:10365`); KPERM3 expert K-permutation; pinned cuBLASLt plans; per-shape routes; `_rqbake`/`_rqsbake` per-`(layer,m)` GEMM variant bakes with bit-exactness gates (`:16201-16212`); and a full **in-prepare paired A/B of the MoE kernel** (`MFAB_LAYER`, 4 seeds × 24 reps, `:16250-16260`). What is left and its price: (a) **SC6S repack, +0.68%** — A2; (b) folding `(1+pre_expert_norm)` and `(1+w)` diagonals into BF16 weight columns — **0 µs**, it removes a `[T,4096]` multiply that is already off any critical path; (c) a precomputed 2048×32 cos/sin RoPE table — 0.5 MB, already negligible against 5.3 µs of latency; (d) the *only* large precomputable object left is a low-rank router sketch, and it **fails**: see next row. |
| **Router prefilter — score 256 experts from a `256×5` projection of the stream's rank-4+mean subspace (all five basis vectors are fixed by `STREAM_SEED = 5354`, `:81-83`), then refine ~48** | **Rejected on error, not on rules.** `STREAM_SHARED = 0.9` puts only 60% of the per-element variance in the 4 shared dims; the residual noise (10%) contributes score error `0.316·|w| = 1.29` logit units `[DERIVED]` vs `ROUTE_TOL = 0.05` — a 26× overshoot, because 0.05 score ≈ 0.006 logit. The sketch would need to keep 60–80% of the 256 rows, saving ≤1.4 µs. Not worth the fact that it also hard-codes a generator seed into `prepare()`. |
| **Router FP32 → BF16 only (8.39 MB → 4.19 MB)** | **Legal-blocked, and the prize is small.** The error would be 250× inside `ROUTE_TOL` (`[DERIVED]` dot-product error 5e-5/element × ‖h‖ = 0.0045 logit × 0.177 × σ'0.25 = 0.0002 score). But `TASK.md` "Rules → Weights" says *"the router's FP32 weights may run as a BF16 **hi + lo pair** with FP32 accumulation"* and *"don't store them at a different precision … or smaller, losslessly"*. Dropping the `lo` is lossy, i.e. a rule violation that costs a day's attempts. Worth if ever legalised: **+0.97%** (`[DERIVED]` 4.19 MB at the route kernel's 3.1 TB/s = 1.35 µs × 0.72). |
| **Deeper dense compression (12 → 10.5 bits, entropy 10.46)** | **Measured to lose.** `prior_art_trxphxle.md`: "~10.9-bit lossless BF16 codec (measured entropy 10.46 bits) → decode **6–11% slower**". Independently, `[DERIVED]`: 130 MB of BF16 weights = 65 M elements; 3 extra ALU ops each = 195 M ops at 132 SM × 128 lane × 1.7 GHz = 6.8 µs against 3.6 µs of byte time saved at D1. Our 12-bit is the optimum. |
| **Full attention 3.5 → 4.5 TB/s** | **At the random-128-B-row wall.** D32F 3.72, V32F 3.34, D16F 3.52 on a gather of 128 B rows from a 328 MB pool whose page table is a `randperm` (:263-266). The parent's own summary: measured three times, treat as closed. The price if the four WS full cells moved 3.5→4.5 TB/s would be +2.33%, and it is not reachable. |
| **Run the dense projections in FP8 / FP4 at T ≥ 128** | **Rule-blocked**: "Activations. BF16 wherever SGLang's kernel keeps BF16" (`TASK.md`). V32 is the one shape where it would bind: the dense is **2.31–2.34 TB/s but compute-bound at 52% of 989 TFLOP/s** `[DERIVED: 43.5 GFLOP / 85.2 µs]`. Prior art swept 1,789–2,309 cuBLASLt configs per shape and found **~2 µs/layer** ("no split-K in the top 60"), so the 52% is what M=256 with N∈{2304,4096,8192} gives on 132 SMs. Closed. |
| **Start the pooler/router or attention on a side stream / fuse the router into the pooler** | **Measured dead twice**: `kernarc_digest.md` — "pooler/router stream started before attention: d32s **2.1–7.7 µs slower**"; "router inside the pooler GEMM: killed on paper, the router branch already ends 6.0 µs (d16) / 8.6 µs (d32) before the pooler GEMM ends". Verify attention slows **50–80%** next to any co-resident GEMM. |
| **Megakernel / glue fusion** (ZhimaoL's four Helion roots) | **Bounded by the launch boundary, not the work.** v1 7.85% attention share, glue 10.5–12.5 µs, one megakernel removes ≤2–4 µs (1.8–3.6% of v1, 0.3–0.7% of v32); our own D1 glue is 4.5 µs and prior art prices each deleted launch at 1.2–2.0 µs. Already counted in A4. |
| **Expert order / multicast / occupancy / cost tables** | All four already measured at ~0% on the MoE board with the numbers: heaviest-first +1.1–1.3% *expert board* (= +0.6% layer), no duplicate reads to multicast (unit list is a bijection), 1 CTA/SM correct because the ring is 8× oversized, cost table already in-kernel (`moe_struct.md` §1–4). |
| **`decode 32 sliding` is our worst load (1.779×) — attack it** | **Wrong target.** Its own time swings ±8% between repeat runs of *different* files (base 138.0, A 145.5, B 139.6, C 150.0 µs, `pdlkv.md` §4) and it is worth **0.070 %/µs**, half of D1S. Fixing D32S by 10 µs = +0.70%; fixing D1S by 10 µs = +1.80%. |

---

## 4. Ranked top-3

| # | lever | expected on the layer board | where | effort | confidence |
|---|---|---|---|---|---|
| **1** | **The attention kernel's fixed cost on the per-request path.** `attn_kernel<L,false,*>` is *forced* on all 46 sliding layers by `if not window` (`:3795`) and on both B=1 full cells; its `t = fixed + S/R` fit is a **6.3 µs constant (q_len 1) / 8.3 µs (q_len 8)** against cell totals of 5.3–15.1 µs and a marginal cost of only 0.21–0.44 µs/MB. Kill 2–3 µs of the constant (merge/finalize ≈2.5–3 µs, launch ≈2, index ≈0.7 — `pdlkv.md` §1) and those 8 cells, which hold 88% of the board's per-µs value, all shrink. | **+1.5%** (band +1.3 to +1.9) | `:3795` the `not window` gate, `:3087`/`:3101` grid `(n_splits,B)` + `chunk`, `segment()` merge path (`:1521-1633`), `:3585-3639` `kn`/`kn1` split choice | high | **high** — a difference of cells, not an extrapolation: 7.8→15.5 MB costs 1.6 µs, so the cell *is* the constant |
| **2** | **The MoE's 1.15× read amplification at v32** (622 MB fetched / 540 MB touched) — the only byte-level waste anywhere in the layer. Trade measured-free ring depth for resident x. | **+1.2%** (V32S +0.86, V32F +0.32) | stage/x ring `:5547-5555` (`WGS_NS = 3`, `WGS_NX`), the `npad·512` x refetch per stage | medium | medium-high — the 622/540 and the 81% L2 hit rate are both directly measured |
| **3** | **`OKC_SC6S`** — 6-bit lossless NVFP4 block scales for the d1/v1/d16 skinny MoE builds (the WGS twin already ships, so those three loads are the only ones still paying 4,096 B of scales per 36,864 B unit). | **+0.68%** | `:4363-4369` flag, `:4381-4390` decode, `:4107` `uslot[8]` stride, prepare-time repack | low | high — the transform is already shipped and measured on the other three loads |

**The prize that is 4× all of the above, and has no lever yet: the MoE runs at 2.24–2.73 TB/s of the judge's
byte count on all six loads** (dense: 3.0–3.6; glue: 4.1; full attention: 3.3–3.7). Every structural angle on
it is already closed with a number (`moe_struct.md` §1–4: no duplicate reads, 1 CTA/SM correct, epilogue off
the critical path, routing prebuilt, `OKC_NS=4` worth nothing). Bringing it to the dense chain's rate is
**+14.5%** of the layer board. That is a per-byte, per-stage efficiency problem — i.e. exactly the one thing
the judge's 989 TFLOP/s and 4.8 TB/s model says should not exist — and it is where the next 10% lives, not in
the model's structure.

---

## 5. What not to spend another check on

Dense compression past 12 bits; full-attention *bandwidth* (it is at its random-128 B gather wall at
3.3–3.7 TB/s; only its 9.2/22.3 µs fixed cost is left, and it is amortised over 86–182 µs); cuBLASLt sweeps; side streams and stream joins;
router/pooler fusion; attention↔GEMM overlap; megakernels for the launch boundary; expert order and multicast;
occupancy; smem size; the cost tables; and `decode 32 sliding` as a target (0.070 %/µs against D1S's 0.180).
