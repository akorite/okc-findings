# attn_xqa_grid — the literature's KV-split grid, extracted from source, ported to `attn_xqa.py` (off by default)

Target: the six sliding cells of the attention board (row **2.6590**), 55–126% fixed cost at 1–8 tiles/CTA,
1.07–1.67 TB/s, against the four flat cells at 76–151 tiles/CTA / 3.07–3.65 TB/s.

**Verdict up front.** Two independent MLSys'25 sources disagree about how to size the split, and the
disagreement is the useful result:

* **XQA's split-first heuristic under-fills our machine.** FlashInfer [#5998] measured the shipped XQA
  launcher on sm_90: `grid (split, KV heads, batch) = (3, 2, 11) = 66 blocks for 78 SMs`, 1 CTA/SM,
  129.6 KiB smem, 12.5% warp occupancy, **721 GB/s = 18.0% of 4.0 TB/s** — the maintainer confirms the
  split is `floor(78/(11×2)) = 3`. That is our failure mode, produced by the formula the brief asked for.
* **The grid-first formula is already what we ship.** LeanAttention (and FlashInfer Alg. 1) fix the grid at
  `NumSMs × MaxCTAsPerSM` and derive the KV chunk from it. Our `_plan` uses `2·SM/(B·pairs)` splits with
  `MINB = 2` and 2 CTAs/SM of occupancy — i.e. LeanAttention evaluated at **this** kernel's
  `MaxCTAsPerSM = 2`. The split count is not where the 6.3/8.3 µs lives; `research_03_venues.md` §"deliberately
  not ranked" reaches the same conclusion from the other direction ("Split-K / more KV splits … recorded as
  *confirmation that the closed axis was right*, not as a proposal").
* What the flag ships is therefore the one arm that satisfies the brief's tiebreak — **every SM busy *and*
  ≥ 8 tiles/CTA at B=32** — which at 132 SMs means **one CTA per SM** (LeanAttention's `MaxCTAsPerSM = 1`,
  not 2). Predicted effect: **+0.09% board** under the arrivals-priced model, **−0.53%** under the
  per-CTA-serial model, i.e. noise: **this axis is closed by measurement, not by further tuning.**

Sources read at these commits (cloned under `~/kr`, `~/Projects/OKC/tmpbuild/engines`):

| tree | file | commit |
|---|---|---|
| flashinfer | `csrc/xqa/{mha.cu,mha_sm90.cu}` (the kernel TRT-LLM JITs) | `08d74a47163138273af02f034a500449b9a2a6d0` |
| TensorRT-LLM | `…/decoderMaskedMultiheadAttention/decoderXQARunner{Utils.h,.cpp}`, `decoderXQAConstants.h` | `a649e41d2cc28944e3b76bc553dbd6fd52dbe195` |
| LeanAttention | MLSys'25, `proceedings.mlsys.org/paper_files/paper/2025/…16ec6494…-Paper-Conference.pdf` | — (via `research_03_venues.md` §2.2) |
| FlashInfer issue | [#5998] | — (via `research_03_venues.md` §2.3) |

**SM count: 132 (H200).** Every published constant is from a 78-SM (H20) or 108-SM (A100) part
(`research_03_venues.md` §2.4); nothing below lifts one unchanged.

## 1. The three formulas, verbatim

**(a) XQA host default** (`mha_sm90.cu:2960-2972`):

```
float const factor = 0.25f;
return mha::min<uint32_t>(
    mha::max<uint32_t>(
        1U, (uint32_t)round(prop.multiProcessorCount * 3 / (batchSize * nbKHeads) * factor)),
    divUp(maxSeqLen, gemm0CtaTileNbTokens));
```

`splits = clamp(round(0.75·SM/(batch·kv_heads)), 1, ceil(kv_len/128))` — 3 resident CTAs/SM times a 0.25
under-subscription. Grid `{divUp(qSeqLen,inputTokensPerCta), nbSubSeqPerSeq, nbKHeads*batchSize}`
(`:2981`); the fully-subscribed twin is `mha.cu:3016-3030` (`SM/(batch·nbKHeads)`, no factor). In the field
the launcher instead runs `floor(SM/(batch·kv_heads))` ([#5998] maintainer reply), which is (a) without the
0.75 — that is the 66-block launch.

**(b) XQA's scheduler gate** (`decoderXQARunnerUtils.h:409-431`), the `sqrt` cap and the 8-wave target:

```
multiBlockCount = historyLength / kMinHistoryTokensPerBlock;                        // 128 tokens/block
multiBlockCount = std::min(multiBlockCount, (int)std::round(std::sqrt(multiBlockCount * 8.F)));
multiBlockCount = std::max(multiBlockCount, 1);
double waveCount = (double) batchSize * numKVHeads * multiBlockCount / (double) multiprocessorCount;
double adjFactor = waveCount / (double) kTargetWaveFactor;                           // = 8
if (adjFactor > 1.0) multiBlockCount = floor(multiBlockCount / adjFactor);
```

with `kMinHistoryTokensPerBlock = 128`, `kTargetWaveFactor = 8` (`decoderXQAConstants.h:27-30`) and
admission `numKVHeads·batch·multiBlockCount·kEnableMinBlockFactor >= SM`, `kEnableMinBlockFactor = 4.0`
(`decoderXQARunner.cpp:154-176`). Note (b)'s wave reduction **can never bind** on top of (a):
`reqs·splits/SM = 0.75 < 8`. Composite form used here:

```
blocks = ceil(kv_len/128);  splits = clamp(min(round(0.75·SM/(batch·kv_heads)), round(sqrt(8·blocks))), 1, blocks)
```

**(c) LeanAttention (MLSys'25)** — grid first, chunk second:

```
G              = NumSMs × MaxCTAsPerSM                       # fixed grid; only the tile→CTA map moves
TilesPerCTA    = (BatchSize × NumHeads × ContextLen) / (TileSize × NumSMs × MaxCTAsPerSM)
LeanTile       = 256 tokens at head_dim 64, 128 at head_dim 128 (measured)
```

plus **FlashInfer Alg. 1** (§2.1 of the venues report), which states the same relation from the split side:
"**KV-split counts** are *derived from the CTA count*, not the reverse. Total work is divided by `#CTA` and
that quotient is the chunk cap. So on a short grid the chunks get bigger and the grid stays short."
The merge is a **host-block/flag** reduction in one launch (see §3).

## 2. Both formulas at SM = 132, on our grid

Our grid is `(n_splits, B, Hkv/2)` (`dual.py:2963`): one CTA covers **both** KV heads as one 128-byte row,
so the formulas' request×KV dimension is `B·pairs` (`pairs = Hkv/2 = 1` at tp8), not `B·Hkv`. One KV tile
here is 64 keys = 16 KB of K+V; a cell's window is 2048 keys ⇒ **32 tiles** (L=1) / **33** (L=8) per request.

| cell | shipped splits / CTAs / tiles·CTA | XQA (a)+(b): splits / CTAs / tiles·CTA | LeanAttention (c) @1 CTA/SM: splits / CTAs / tiles·CTA | LeanAttention @2 CTA/SM |
|---|---|---|---|---|
| D16S (B16 L1) | 16 / 256 / 2 | min(6, 11) = **6 / 96 / 5.3** | 132/16 = 8.25 → **8 / 128 / 4.0** | 16 / 256 / 2.0 |
| D32S (B32 L1) | 8 / 256 / 4 | min(3, 11) = **3 / 96 / 10.7** | 132/32 = 4.1 → **4 / 128 / 8.0** | 8 / 256 / 4.0 |
| V16S (B16 L8) | 8 / 128 / 4 | min(6, 12) = **6 / 96 / 5.5** | 132/16 = 8.25 → **8 / 128 / 4.1** | 16 / 256 / 2.1 |
| V32S (B32 L8) | 4 (`_TUNED`) / 128 / 8 | min(3, 12) = **3 / 96 / 11.0** | 132/32 = 4.1 → **4 / 128 / 8.25** | 8 / 256 / 4.1 |
| D1S (B1 L1) | 32 / 32 / 1 | min(99, 11) = **11 / 11 / 2.9** | min(132, 32) = **32 / 32 / 1.0** | 32 / 32 / 1.0 |
| V1S (B1 L8) | 33 / 33 / 1 | min(99, 12) = **12 / 12 / 2.8** | min(132, 33) = **33 / 33 / 1.0** | 33 / 33 / 1.0 |
| D1F…V32F | flat 252/264, untouched (`window > 0` guard) | | | |

LeanAttention's literal `TilesPerCTA` at those shapes, `H = 16` q heads, `Tile = 256`, `Ctx = 2048`:
`B·15.5` ⇒ **15.5 (B16) / 31.0 (B32) LeanTiles per CTA** over a grid of `SM·MaxCTAsPerSM = 132`; in our
64-key tiles (a LeanTile is 4 of them, and a CTA here holds all 16 q heads at once, so the KV-shared count
is the same 132-CTA quotient) that is 512/132 = 3.9 and 1024/132 = 7.8 — the table's 4.0 / 8.0.

**Which one wins the brief's tiebreak** ("every SM busy *and* tiles/CTA ≥ 8 at B=32"):

* XQA (a)+(b) at B=32: 96 CTAs = 0.73 CTA/SM — **15 SMs idle**, and it is exactly the #5998 shape
  (0.85 CTA/SM) that runs at 18% of HBM peak. Fails "every SM busy".
* XQA's 8-wave gate alone (`XQA_OCC = 8`): 11–12 splits, 176–384 CTAs, 3 tiles/CTA — fails "≥ 8".
* LeanAttention @1 CTA/SM: 4 splits, **128 CTAs (97% of the 132 SMs — all but 4 busy), 8 tiles/CTA at
  B=32** — the only arm that satisfies both. **Chosen.**
* LeanAttention @2 CTA/SM equals the shipped plan (256 CTAs, 4 tiles/CTA): the paper-faithful port for *our*
  occupancy (`Cfg::MINB = 2`, 2 CTAs/SM by smem), but it fails the ≥ 8-tile criterion, so it is recorded as
  the anchor rather than shipped.

That anchor is the interesting number: `_plan`'s shipped `2·SM/(B·pairs)` **is** LeanAttention's grid at this
kernel's real occupancy, which is why the plan sweep found ±8% and no pick (`attn_plan.md`: 0/12 shapes). The
arm the flag ships is a deliberate under-subscription to 1 CTA/SM — the paper's own regime (their CTA needs
129.6 KiB, ours 48–107 KiB) — traded for double the runway per CTA.

## 3. The merge, and why a variable split count needs no kernel change

**XQA's single-launch merge** (`mha_sm90.cu`):

* effective count from data, not grid (`:713-723`): `nbSubSeq = min(nbTilesInUse/multiBlockMinNbTilesPerCta,
  maxNbSubSeq)`, `if (idxSubSeq >= nbSubSeq) return;` — surplus CTAs leave *before* the semaphore, and every
  CTA derives the same count from the tile total (`multiBlockMinNbTilesPerCta = 1; // 3; // @fixme`,
  `:102-103`);
* KV striped over tiles (`:716,761,805`): `idxKTile = idxKTileInit + idxIter*nbSubSeq`;
* arrival (`:1424-1442`): `atom.acq_rel.gpu.global.inc.u32 old, [sem], lastOld` with
  `lastOld = nbSubSeq-1`, result kept in smem `isLastCta` — release of this CTA's scratch, acquire of the
  others', no `__threadfence`, no combine launch;
* the merge itself is spread over `nbIOWarps = 4` warps (`:1473-1550`) with a two-deep smem ring
  (`nbBuf = 2`, mbarrier parity) pulling `rowSum`/`rowMax`/out from global scratch;
* **the semaphore is never reset in the kernel** — the only references in `mha.cu` are the parameter and the
  `inc` at `:2762`. The caller still owns a workspace reset.

**LeanAttention's merge** does not even use a last-CTA election: the CTA holding an output tile's *first*
LeanTile is the **host block**; every other contributing CTA `StorePartials(O,m,l)` then `Signal(flags[g])`;
the host `Wait`s, folds with the associative softmax rescale (`m = max`, `l = e^{m−m}l_cta + e^{m−m}l`,
`O = …`, proven for unequal block sizes), and the finishing CTA writes `diag(l)^{-1}O` from registers. One
launch, no global atomics. It generalizes FlashAttention-2 (>1 output tile) and FlashDecoding (grid a
multiple of output tiles); 1.73× average over FlashDecoding.

**Our kernel already has the stronger version of both.** The arrival is counted in **keys**, not in splits or
CTAs: `chunk = ceil(n_keys/n_splits)` (tile-rounded on the host-supplied `ns_magic`, `dual.py:2546-2556`),
each CTA adds `end − start`, the last one is the CTA with `old + mine == n_keys` (`dual.py:2319-2323`), and a
split with an empty range returns **before** the counter (`dual.py:2564`) — a contiguous cover has no gaps,
so the sum is exactly `n_keys` for **any** `n_splits ≤ 1024`. The cooperative finalize counts
`nact = ceil(n_keys/chunk)` (`dual.py:2308`) — the same "effective count from the data" XQA derives from its
tile total — and slices `L*16` rows over it. And unlike XQA it is graph-replayable with **no host reset**:
two counters alternate by launch parity and the last arrival re-zeroes the *other* copy
(`dual.py:2317-2326`, epoch in `ws_cnt + 24576`).

Consequences for this port: **no kernel change**. The only code-side coupling is that a smaller `n_splits`
drops the launcher out of the one-tile-per-split builds (`kn1` when `n_splits·64 >= kmax`, `dual.py:2866`)
into the ordinary multi-tile builds `kn[1]`/`kn[3]` that the B=1 full cells already run.

## 4. What shipped

`src/okc_loop/vendored_work/attn/attn_xqa.py` — byte copy of `dual.py` (sha `147ea0ed`) plus three hunks,
all behind `XQA_SPLIT = False`:

| change | effect with the flag off |
|---|---|
| `XQA_SPLIT`, `XQA_FORM = "lean"`, `XQA_CTAS_PER_SM = 1`, `XQA_OCC = 0.75`, `XQA_BLOCK_KEYS = 128` | none |
| `_plan`: `if window and XQA_SPLIT:` — `n = round(XQA_CTAS_PER_SM·SM/reqs)` (`"xqa"`: the §1(a)+(b) form), clamped to `[1, tiles, 1024]`, `nc = 1`, `coop = 1 if L == 8` | none |
| `_launch`: `tuned = pick = None` when the flag is on and the shape slides | none |

The third hunk is needed because both import-time sweeps were fitted against the old heuristic (`_TUNED`
pins `(32,8,True,1)` to 4 splits) and a pick applies whenever the pre-override `n_splits` matches — with the
flag on the new plan must not be folded back.

`XQA_FORM` is the A/B: `"lean"` is the shipped arm, `"xqa"` reproduces §1(a)+(b) for the kill-criterion
comparison; `XQA_CTAS_PER_SM = 2` reproduces the shipped grid and is the "is the plan formula itself right"
control. Verified by exec'ing `_plan` with a stubbed SM count (132) — the table in §2 is that run.

Gates: lint `PASS 1/1 clean` (base `PASS`); `tools/cu_build.py`: `CUDA_SRC ok`, `ZH_WST_CUDA ok`,
`ZH_VFG_CUDA` FAIL with the identical offline `cuda_bf16.h` `__shfl_*_sync` "C linkage" errors as base —
environment artifact, unchanged. No attention math changed: the wgmma/`mma.sync` question from
`research_03_venues.md` (wgmma M fixed at 64; no non-arXiv claim that `mma.sync` wins at small M) does not
bear on this candidate — the MMA path, A/B swap, tile size and stage depth are untouched.

## 5. Predicted µs at the chosen split count

Model (`t = fixed + S/R`, the fitted one, since the fixed term is what we are attacking): `S/R` unchanged
(marginal 0.206 µs/MB at q_len 1, 0.438 at q_len 8 — roofline, 4.85/2.28 TB/s, insensitive to the grid); the
fixed term's only split-dependent part is the arrival/merge chain, priced at **0.09 µs per arriving split**
(calibrated at the cell with the most arrivals: 3.0 µs / 32 at D1S, the top of the published 2.5–3 µs).
Base µs and the "board %/µs" weights are from `muse_arch_levers.md` §A1.

| cell | splits 32/16/8/33/8/4 → new | merge µs | **predicted µs** | Δ | weight | board Δ |
|---|---|---|---|---|---|---|
| D1S | 32 → 32 (capped by tiles) | 2.88 → 2.88 | 5.3 → **5.3** | 0 | 0.180 | 0 |
| V1S | 33 → 33 | 2.97 → 2.97 | 6.6 → **6.6** | 0 | 0.114 | 0 |
| D16S | 16 → 8 | 1.44 → 0.72 | 7.9 → **7.2** | −0.72 | 0.088 | +0.063 |
| V16S | 8 → 8 | 0.72 → 0.72 | 11.7 → **11.7** | 0 | 0.048 | 0 |
| D32S | 8 → 4 | 0.72 → 0.36 | 9.5 → **9.1** | −0.36 | 0.070 | +0.025 |
| V32S | 4 → 4 | 0.36 → 0.36 | 15.1 → **15.1** | 0 | 0.035 | 0 |
| the six full cells | untouched (`window > 0`) | | | | | 0 |

**Σ = +0.09% → row 2.6590 → 2.6614, value 1415.9 → 1414.7 µs.** (Weight sanity: the same weights price
"3 µs off all eight non-WS cells" at +1.91%, matching `muse_arch_levers.md`.)

**The counter-model, and why this is a measure-me item.** The table prices only arrivals. Calibrating a
per-CTA serial chain instead — `t = fill + n·τ` on the current 256-CTA points (D16S 7.84 µs at n=2, D32S
10.04 at n=4 ⇒ τ ≈ 1.1 µs/tile, fill ≈ 5.6; V16S 11.64 at n=4, V32S 14.58 at n=8 ⇒ τ ≈ 0.735, fill ≈ 8.7) —
says the 1-CTA/SM arm *loses*: D16S 10.0 µs, D32S 14.4 µs, **−0.53% board**. The XQA arm on the same model
is far worse (−1.9%), so **lean dominates xqa under both models**; what the models disagree about is whether
either beats the shipped plan. They cannot: the shipped plan *is* the grid-first formula at our occupancy,
and both models put this axis inside the board's own 2.2% same-sha spread (`attn_gap.md` §1).

## 6. Kill criterion

Adjudicate on the interleaved per-shape medians (`_PLAN_ROUNDS = 16`, `_PLAN_MARGIN` 3%/2 µs), never on a
board delta.

* **Numerics first.** rel-L2 of the bf16 rows vs the same file with `XQA_SPLIT = False` must stay under
  `1e-3` (`_PLAN_RTOL`) on all twelve scored shapes. The split moves only chunk boundaries and the fp32
  atomic accumulation order; a larger deviation means the key-counted arrival is double-counting or missing a
  split at the new count, and the port is wrong rather than slow.
* **The shipped path must be inert**: with the flag off every `[plan]` line and per-shape median must match
  `dual.py` to the last digit, else the comparison is void.
* **Control**: `XQA_CTAS_PER_SM = 2` must reproduce the shipped plans exactly (it does, §2, except V32S whose
  4 splits come from `_TUNED`, which the flag disables). If it does not, the plan wrapper is lying.
* **Keep** only if the six-cell sum improves ≥ 3% on `"lean"` **and** no sliding shape regresses > 2%; of the
  two arms keep the better. **Kill** if it is flat — which both models predict — and then stop treating the
  grid as an axis: #5998 already shows a 0.85 CTA/SM sm_90 decode at 18% of HBM peak with page size provably
  not a lever, and 8-wave/`sqrt` gates cannot manufacture parallelism a 96-CTA grid does not have.
* Next lever if this is flat: stage depth (`Cfg<L,false>::STAGES` 3 → 4, `(STAGES-1)·16 KB` in flight per
  CTA), i.e. *occupancy per CTA* rather than CTA count — deliberately **not** in this file: 4 stages costs
  smem, which is the quantity that decides 1 vs 2 CTAs/SM, and that cannot be checked without a device.

## 7. Explicitly unverified (no GPU here)

* Every predicted µs: no launch was made. Both models, the anchor arm and the A/B are in §5/§6.
* End-to-end numerics at a variable split count. The invariants are checked by reading (§3), not by running:
  empty splits return before the counter, the arrival sum equals `n_keys`, `nact` matches, and coop row
  slices cover `L*16` rows for `nact = 1…12`. `coop` is 1 for L=8 (as shipped at small grids) and 0 for L=1;
  the launcher downgrades `coop` when the grid is not resident — shipped behaviour, unchanged.
* Whether 128 CTAs at 1/SM sustain the 2.28–4.85 TB/s marginal the 256-CTA grids do. That single assumption
  is the difference between the two models in §5.
* Stage depth (brief item 3) not attempted: one untested mechanism per file, and it perturbs the exact
  quantity (CTAs/SM) this change already moves.
