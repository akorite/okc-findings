# Dense path (32.1% of the layer, never attacked): the pooler+router projection is the gap

Every per-kernel number is from job `7cd40fdd1a934e27` (12 Nsight single replays,
`artifacts/cas/profiles/82c098ae…7cd40fdd1a934e27.json`, `result.records[i].report`).
File: `src/okc_loop/vendored/keell0renz/muse_decode_layer.py` (sha 82c098ae), board line numbers.

## 0. Cell sizes (these decide everything below)

`HIDDEN = 8192`, `D_EXPERT = 4096`, `N_EXPERTS = 256` (78-81); `bfp_w_rows` = rows (8971);
`bfp_SK2_K` = K (9560).

| cell | shape [N, K] | bf16 MB | stream MB (12-bit) | FLOPS/token |
|---|---|---|---|---|
| qkv_gate | [2304, 8192] | 37.75 | 28.3 | 37.7 M |
| o | [8192, 1024] | 16.78 | 12.6 | 16.8 M |
| pooler | [4096, 8192] | 67.10 | 50.3 | 67.1 M |
| unpooler | [8192, 4096] | 67.10 | 50.3 | 67.1 M |
| router | [256, 8192] hi+lo | 8.39 | 8.4 (BF16) | 4.2 M |
| **five cells** | | **197.1** | **149.9** | |

149.9 MB is exactly the number the layer model uses (`artifacts/analysis/layer_model.py:35`,
">= 149.9 MB and >= 35.0 us at 4.287 TB/s"), so the model's floor is the *compressed* volume:
(37.75+16.78+67.1+67.1)*0.75 + 8.39 = 149.95.

Note the unpooler is **not** the o cell's twin (a 4x-larger matrix): any per-byte comparison of the
two is invalid, and un is in fact the *fastest* cell per byte at d32 (2.96 TB/s).

## 1. Which backend each cell actually runs (quoted picks)

The last `prepare()` decides, not `_tune` and not `PIN_PICK`:

```
12692 _stack_unpinned_prepare = prepare
12698         for T in (1, 8, 16, 32):
12701             for name in ("qkv", "o", "un"):
12702                 st["mm_pick"][(T, name)] = "bfp"
12705             st["mm_pick"][(T, "pr")] = "bfpf" if T <= 32 and can_fuse else "bfp"
12709     for T in (128, 256):
12710         for name in ("qkv", "o", "pr", "un"):
12711             st["mm_pick"][(T, name)] = True
```

`_tune` (11062-11190) times four arms per (T, matrix) — `t_tri` (Triton split-K), `t_mm` (cuBLAS
`torch.mm`), `t_bfp`, `t_bfpf` (fused) — and 12701/12705/12711 then **overwrite** the winner for all
six scored token counts. `PIN_PICK` (112) is overwritten too. So at 16/32/128/256 the backend is
forced; the tuner only matters at T = 1/2/4/64.

Secondary pick tables:

```
9526 bfp_SK2_CONFIGS = [  # (name, layer, n_rows, R rows/CTA, NT, NWR, WSK, UPW, NACC, CL)
9543 bfp_SK2_PICK = {  # (layer, tokr) -> config name;  tokr = 8 if m<=8 else 16 if m<=16 else 32
9584 bfp__sk2_geom: nstage = max(2, min(16, nst, ring//sst)),  ring = min(cap*1024, 232448-extra)
                   cap parsed from the NAME suffix "_rNNN"
12304 _GRAFT_LT_PICKS   # (base matrix, T) -> (mode, batch, fp32 out, 8 raw cuBLASLt algo words)
16209 RQ_USE = {('un', 128), ('qkv', 256), ('qkv', 128)}   # custom rs128 kernel runs before the Lt plan
```

| cell | T=16 | T=32 | T=128 | T=256 |
|---|---|---|---|---|
| qkv_gate | sk2 `qkv16_r176` **10.7** | sk2 `qkv32_r208` **11.7** | `rq` (in the 2-call 31.2) | `rq` **17.8/18.1** |
| o | sk2 `o8h_r88` **5.9** | sk2 `o32h_r100` **6.4** | Lt `('nn',8,bf16)` **8.6** | Lt `('nn',2)` **9.1/9.5** |
| pooler+router | sk2**f** fused, pooler default **18.3** | sk2**f** fused **21.2** | Lt `('tn',1,fp32)` **26.6** | Lt `('tn',2,fp32)` **32.2/32.5** |
| unpooler | sk2 `unpooler16h_r100` **15.0** | sk2 `unpooler32_r144` **17.0** | `rq` (same 31.2) | Lt `('nn',2)` **25.5/27.4** |
| router | in the fused `pr` launch (T<=32) | same | in the TNN `pr` plan | in the TNN `pr` plan |

Kernel-name decode: `sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL>` (8049).
d32s: `<2,32,3,1,4,2,4>` = `qkv32_r208`; `<2,32,2,2,2,2,3>` = `o32h_r100`; `<2,32,4,2,2,2,3>` =
`unpooler32_r144`; `<2,32,4,1,8,1,3>` = the *fused* pooler+router (`sk2f` instantiates only the
`'pooler'` default config: `bfp__sk2_src`, `if name != "pooler": continue`), which is why there is no
separate router row at T<=32.
`PDL_MAX_T = 128` (98), so T=256 spans are the kernels' own time; T<=128 spans include any
`griddepcontrol.wait` on the previous kernel (2 pre-wait stages, `_bfp_pdl`/`BFP_PDL_N = 2`).

## 2. Per-cell time vs floor

Floor = stream MB / 4.287 TB/s (the model's own number). "honest" = the size-dependent read ceiling
measured on this GPU (`kernarc_digest` §5: 66 MiB only 3.63-3.81 TB/s, 128 MiB 3.88-4.19) and, where a
sibling cell in the same profile already does better, that sibling's rate.

### d32s (wall 144.94, GPU busy 145.2 = 4.8 us of overlap; dense 56.3 us = 39% of the row)

| cell | stream | floor | actual | excess | rate |
|---|---|---|---|---|---|
| **pooler+router** | 58.7 | 13.7 | **21.2** | **+7.5** | 2.77 TB/s |
| unpooler | 50.3 | 11.7 | 17.0 | +5.3 | 2.96 |
| qkv_gate | 28.3 | 6.6 | 11.7 | +5.1 | 2.42 |
| o | 12.6 | 2.9 | 6.4 | +3.5 | 1.97 |
| five cells | 149.9 | 35.0 | 56.3 | +21.3 | 2.66 |

### d16s

| cell | actual | excess | rate |
|---|---|---|---|
| pooler+router | 18.3 | +4.6 | 3.21 |
| unpooler | 15.0 | +3.3 | 3.35 |
| qkv_gate | 10.7 | +4.1 | 2.65 |
| o | 5.9 | +3.0 | 2.14 |
| five cells | 49.9 | +14.9 | 3.00 |

### v16s / v32s (BF16, no codec above m=32; PDL off at 256)

| cell | v16 actual | v16 excess | v32 actual | v32 excess | v32 rate |
|---|---|---|---|---|---|
| **pooler+router** | **26.6** | **+9.0** | **32.2/32.5** | **+14.6/+14.9** | 2.34 |
| unpooler | (in 31.2 rq pair) | — | 25.5/27.4 | +9.9/+11.8 | 2.63 |
| qkv_gate | (in 31.2 rq pair) | — | 17.8/18.1 | +9.0/+9.3 | 2.12 |
| o | 8.6 | +4.7 | 9.1/9.5 | +5.2/+5.6 | 1.82 |
| five cells | 66.4 | +20.4 | 85.6 | +39.6 | 2.30 |

The `rq` pair at T=128 is 31.2 us for qkv+un = 104.85 MB = **3.36 TB/s**, the best rate anywhere in the
dense path — i.e. the custom rs128 kernel is not the problem at v16; the 4608-row `pr` Lt plan is.

### alygg77, where per-cell numbers exist

d32s CUPTI split: total 154.1 = MoE 84.0 + dense 51.5 (byte floor 34.9) + attn 9.8 + glue 8.8, with
"qkv 96 of 132 SMs, 11.4 us vs 6.6 us floor". Their qkv floor 6.6 us = 28.3 MB / 4.287 TB/s, i.e. their
dense cells are also 12-bit — so the two files are directly comparable cell by cell: **qkv 11.4 vs our
11.7** (parity), dense total **51.5 vs our 56.3** (+4.8 us, +9%). The whole 4.8 us is inside `pr`
(their dense is one budget number, so the per-cell split is ours to infer) and our `pr` is the only
cell where we have no rival number that matches.

## 3. The single biggest in-layer dense gap

**The pooler+router projection (4608x8192, 75.5 MB BF16 / 58.7 MB at d32) — the largest cell and the
largest absolute excess at *all four* scored dense T (+4.6 d16, +7.5 d32, +9.0 v16, +14.9 v32): 36 of
the dense path's ~96 us of excess.**

Why it cannot be picked away:

* **v32**: the `('tn', 2, fp32)` plan emits `nvjet_sm90_tss_144x64_64x8_1x1_v_bz_TNN`, grid =
  ceil(4608/64) x ceil(256/144) = **144 CTAs, full machine, 8 stages**, 19.33 GFLOP in 32.2 us =
  **600 TFLOP/s = 60% of the 989 TFLOP/s BF16 peak** and 2.34 TB/s = 55-63% of the honest byte ceiling.
  Both bounds are *unreached*, so neither bytes nor tensor cores is the limiter and no tile/split-K in
  cuBLASLt closes it: the file's own sweep covered 1,789-2,309 valid configs per shape for **~2 us
  in-layer total with no split-K in the top 60**, `bfp_WIDE_BF16 = False` (the BF16 wide kernel never
  beat cuBLAS), `gq` exists only for ("pooler",64/128), and `12709-12711` force `pr = True` so the bfp
  pooler kernel is not even offered at 128/256.
* **d32**: the fused sk2f runs the only pooler config it instantiates, whose stage is 56 KB
  (UPS 8, `sst = 8*32*128 + 24576`) so the 200 KB ring admits **3 stages** and 4 stages (229 KB) does
  not fit in the 227 KB block limit; a cap change cannot help. The same kernel at T=16 gets 5 stages and
  runs the same 58.7 MB at **3.21 TB/s instead of 2.77** — the 13-15% at d32 is the shallower ring, and
  buying it back needs a smaller stage (UPS 4), i.e. a new instantiation of `sk2f`, not a table pick.

**The mechanism that closes it is the 12-bit lossless weight stream above m = 32.** The file already
has the codec (`bfp_sk2_compress`, 8961: 512 B low bytes + 256 B of 4-bit exponent codes + exact escape
corrections, the pooler's 12.07 bit/weight that both teams use) and it is enabled for exactly the four
cells in `bfp_COMPRESS = {qkv_gate, o, pooler, unpooler}` (8972) — but only inside `sk2_kernel`, which
is M <= 32 by construction (`TOKR <= 32`, `MT <= 2`). Above m = 32 the file falls back to plain BF16
(`state["raw"]`, used by `gemm90`/`wide`/`gq`/Lt). Extending the stream to the verify sizes cuts
197.1 -> 149.7 MB, **-38 MB at both v16 and v32**: `pr` 75.5 -> 56.6 (-8.0 us at 2.34 TB/s), `qkv`
37.75 -> 28.3 (-4.5), `un` 67.1 -> 50.3 (-5.5) => **~-18 us at v32 (350.5 us row, -5.1%) and ~-18 at
v16 (-6.9%)**, i.e. +0.86% and +1.15% of score, ~2.0% total. Those are *byte-proportional* estimates:
the kernels sit at 55-63% of the byte ceiling, so part of a smaller stream converts into more idle
latency instead of time saved; treat -18 us as the upper bound and the +2% score as its scale. `o` is the exception (12.6 MB, latency-bound
at 1.8 TB/s; the codec buys it ~0).

This is a **kernel we would have to write**, not a pick: a wgmma/`gemm90` mainloop that decodes the
12-bit B-fragment stream in-loop at m = 128/256. The decode cost is 0.5-1.5 us at m<=8 (`EXPERIMENTS.md`
E12) and amortizes over 128+ MACs per weight word at m>=128, so the arithmetic is the payload, not the
decode — but nobody in this workspace has proven a decode-in-mainloop at those sizes (`bf16_mechanism_scan`
calls it "12-bit format at m>=128: EV +2% theoretical, decode-in-mainloop unproven, OPEN, high risk"),
and both rival teams stopped at the m<=32 form. Design sketch: reuse `bfp_sk2_compress` byte-for-byte
(the stream is a pure function of the weights, already built in prepare for `bfp_SK2_CONFIGS`), add a
`pmax`-entry escape table per 128-row CTA, and decode into the wgmma B fragment the way `sk2_kernel`
already does per 64-k unit — the change is the tile (64 or 128 rows x 64 k) plus the escape walk over
4x more k-units per stage, not a new codec.

## 4. Second-order, cheap, not built

* **d32 ring depth.** All four sk2 cells at T=32 are smem-capped to 3 stages while the T=16 kernels get
  5 and beat them per byte (pr 3.21 vs 2.77, un 3.35 vs 2.96 TB/s on identical byte counts). One config
  can be deepened by *renaming* it: `unpooler32_r144` -> a `_r175` suffix lifts `ring` to 179 KB and the
  same geometry then gets 4 stages (4*40960 + 35040 = 194 KB of the 227 KB block limit). Zero correctness risk (same kernel,
  same stream), ~-2 us at d32 = +0.18% score — under the noise bar, which is why it is not built.
* **`("un",256)` custom nibble.** `_GRAFT_LT_PICKS` gives `("o",256)` custom=6 (`…286`,
  `nvjet_..._NNN`, 9.1 us on 16.78 MB) and `("un",256)` custom=7 (`…287`, `..._NNT`, 25.5-27.4 us) on
  identical descriptors. **Do not spend a check on swapping them**: the unpooler is 4x the bytes
  (67.1 vs 16.78 MB) and already runs at a higher rate (2.63 vs 1.82 TB/s), so there is nothing to win.
* **`pr` through `rq`.** `rq` is the fastest kernel in the path (3.36 TB/s at T=128) but `_rqbake` only
  covers qkv_gate/unpooler (`_RQ_NAMES = {"qkv_gate": "qkv", "unpooler": "un"}`, 16174) and the plan is
  per-matrix, so `pr` (4608 rows fused) cannot reuse the 4096-row pooler plan; unfusing was measured
  +10-12 us by the author (12706).

## 5. Status

Built: **nothing**. The fix is a new kernel at m >= 128, so per the ticket this is design-only; the
candidate I started (`klr_dense.py`, forcing the unpooler's sk2 geometry and the un Lt nibble) was
deleted — it rested on the false premise that the unpooler and `o` are the same matrix, which they are
not (`D_EXPERT = 4096` vs K=1024). No check was spent.
