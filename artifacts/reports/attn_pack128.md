# attn_pack128 — the decisive question: do we already issue one 128 B load per position?

**Answer: yes. Shipped, in every gather path, since before this question was asked.** `dual.py` reads both
KV heads of a position as a single contiguous, 128 B aligned region — eight lanes × one 16 B `cp.async` over one
row — and it does not re-fetch the second head anywhere. There is therefore no `attn_pack128.py` to write: the
competitor's mechanism is not a mechanism we lack. The rest of this report prices what is actually left.

---

## 1. The source, line by line

### 1.1 A cache row *is* one 128 B pair of KV heads

The row width is `Hkv * 64` bytes, and the scored shape has `Hkv = 2`, FP8 E4M3 → 128 B. It is a compile-time
constant *and* a runtime-derived log2, so the kernel indexes rows by shifting, not multiplying:

```
dual.py:165    constexpr int ROWB = 128;                    // bytes per tile row: 2 KV heads x 64
dual.py:166    constexpr int STAGE_BYTES = TILE * ROWB;     // 8 KB per stage for K (same for V)
```
```
dual.py:2953     p.row_bytes = (int)(Hkv * 64);
dual.py:2954     TORCH_CHECK((p.row_bytes & (p.row_bytes - 1)) == 0, "Hkv * 64 must be a power of two");
dual.py:2955     p.lrb = 0;
dual.py:2956     while ((1 << p.lrb) < p.row_bytes) ++p.lrb;          // lrb = 7
```
```
dual.py:2918   // [rows, row_bytes] byte view of a cache, boxes of one 128-byte row (one KV pair), 128-byte swizzle
```
The last line is the TMA descriptor's own contract: **one box == one row == one KV pair == 128 B**. The
architecture already names the thing the competitor is being credited for.

### 1.2 The gather: eight lanes per row, one 128 B contiguous window

`issue_kv` is the only place a sliding-window cell fetches K or V. Its own header says the design intent:

```
dual.py:1160   // K/V: thread -> 16-byte chunk (tid & 7) of rows (tid >> 3) + 16*k: a warp copies 4 whole 128-byte rows.
dual.py:1162   const int cchunk = tid & 7, crow0 = tid >> 3;
dual.py:1163   constexpr int RPP = NT / 8;   // rows per pass
dual.py:1193       const uint32_t off = ((((uint32_t)idxs[r] << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb)
                                        + (uint32_t)(pair * 128 + cchunk * 16);
dual.py:1194       cp_async16(smem_u32(dk), p.k_cache + off, 16);
dual.py:1195       cp_async16(smem_u32(dv), p.v_cache + off, 16);
```

Unpacked:

* `cchunk = tid & 7` — lanes 0–7 of every warp take chunks 0–7 of **the same row**.
* `pair * 128 + cchunk * 16` — the eight chunks span exactly `[pair·128, pair·128 + 128)`: **one 128 B window,
  128 B aligned** (the row base is `row << 7` plus a 512 B-aligned tensor base).
* So one warp instruction covers `row(pair·128+0..127)` for three different rows → **4 line-fills of 128 B each,
  512 B, zero waste.** There is no `+64 B` second fetch, and no second head left over to fetch: with `Hkv = 2`,
  `pairs = Hkv >> 1 = 1` and `pair` is always 0, so the 128 B window *is* the whole row.
* The same expression appears at `dual.py:1173` (prologue), `dual.py:1330` (the one-tile fast path),
  `dual.py:1119/1124` (flat kernel) and `vfg.py:1035/1119/1124/1160` (the few-request path). The `L == 8` verify
  kernel shares the identical `issue_kv` body (`Cfg<L,false>::NT` at `dual.py:676`/`804`), only `RPP` differs.

`Hkv > 2` generalises without a change: `pairs = Hkv >> 1` gives each half-row its own `pair·128` base, still
one contiguous 128 B per pair.

### 1.3 The pair is shared, not re-read

```
dual.py:1578   const int hk = __shfl_sync(0xffffffffu, (L == 1) ? (warp & 1) : ((warp & 7) >> 2), 0);   // L==8: warpgroup = KV head
```
One KV head per warp *for the math*, out of a stage buffer that already holds both. The global 128 B line is paid
for once and consumed twice from shared memory. The competitor's `[16×128]` block-diagonal query is how they fold
the pair into one wgmma; we fold it with two warps over one stage. Same bytes on the wire, which is the only thing
that matters for the 2 TB/s-vs-3.3 TB/s claim.

### 1.4 What the 64 B path would have looked like

The only 64 B-granular global traffic in the file is a **store**, not a gather: the write of the block's own new
K/V tokens into the cache (`dual.py:1544`, `(2 * pair + hh) * 64 + partL * 8`), i.e. 8 positions × 64 B × 2, a
few hundred bytes per request. That is the write side of the KV cache, not the read side the mechanism is about.

---

## 2. Transfer-size table

Byte counts are the harness's own accounting (`reference.py:389 speed_of_light_ns`: `tokens * hkv * HEAD_DIM * 2`
with `_shortest_share(spread)` ≈ 0.94 on `spread agentic`, K+V as two separate FP8 caches), recovered from its
`sol` column at `HBM_BYTES_PER_S = 4.8e12` (`scoring.py:47`). `eff = sol / ours`.

| cell | K+V bytes | ours µs | achieved TB/s | eff | `implied` row bytes/position | plan | CTAs | tiles/CTA |
|---|---|---|---|---|---|---|---|---|
| decode 16 sliding | 7.92 MB | 7.84 | **1.01** | 21% | 256 (2 × 128 B, full lines) | 16/1/0 | 256 | **2** |
| decode 32 sliding | 15.79 MB | 10.04 | **1.57** | 33% | 256 | 8/1/0 | 256 | **4** |
| verify 16 sliding | 8.93 MB | 11.64 | **0.77** | 16% | 256 | 8/1/1 | 128 | **4** |
| verify 32 sliding | 17.86 MB | 14.58 | **1.22** | 26% | 256 | 4/1/1 | 128 | **8** |
| *decode 16 full* | 303.3 MB | 86.78 | ***3.49*** | 73% | 256 (same rows, same page size) | flat 264 | 264 | 76 |
| *decode 32 full* | 606.7 MB | 166.30 | ***3.65*** | 76% | 256 | flat 264 | 264 | 151 |
| *verify 16 full* | 304.3 MB | 99.24 | ***3.07*** | 64% | 256 | flat 264 | 264 | 75 |
| *verify 32 full* | 608.6 MB | 185.20 | ***3.29*** | 68% | 256 | flat 264 | 264 | 151 |

(The brief's 8.4 / 16.8 MB and 1.07–1.67 TB/s assume `share = 1.0`; the harness's own share is 0.94, which moves
the rate ~6%, not the conclusion.)

**The four italic rows are the argument.** They are the same kernel, the same 128 B rows, the same `page_size=1`
random page table, the same FP8 caches — they move **3.07–3.65 TB/s**, i.e. *at or above* the 3.0–3.3 TB/s the
competitor cites as the 128 B ceiling. Our flat path therefore already proves the row size is right and the
bandwidth is reachable. The only variable that separates 3.65 from 1.57 TB/s is **tiles per CTA: 151 versus 4**.

---

## 3. Where the remaining gap is, with the source evidence

### 3.1 Outstanding bytes per CTA — 32 KB, and it is a compile-time constant

```
dual.py:157    static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : 3;
dual.py:1612       cp_wait<STAGES - 2>();          // ≤ 1 group pending -> tile i landed
dual.py:1613       if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
dual.py:1353       for (int k = 0; k < STAGES - 1; ++k) { if (k < PRE && k < ntiles) issue_kv(k, false); ...
```
Per-request kernels run `STAGES = 3` (`F == false`), one tile = `TILE * ROWB = 64 × 128 = 8 KB` per buffer
(`dual.py:163–166`), two buffers ⇒ **`(STAGES-1) × 16 KB = 32 KB` in flight per CTA**, and the prologue issues
only two tiles. That is the whole lookahead. The flat kernel's `Cfg<L,true>` runs 3/4 stages but keeps the *same*
32–48 KB — the difference is that it has 76–151 tiles to spend it on, so it stays in steady state; the sliding
cells have 2–8.

### 3.2 The structural reason the sliding cells never get runway

```
dual.py:3083   if not window and 1 < B <= MAX_BATCH:
dual.py:3084       # Full layer, several requests: equal shares of all requests' keys, one CTA pass per SM pair.
dual.py:3085       return dict(n_splits=max(1, target // pairs), flat=1, cs=0, ncl=1, nc=1)
dual.py:3086   keys_max = min(window + L - 1, max_pages * P) if window else max_pages * P
dual.py:3087   tiles = max(1, (keys_max + TILE_KEYS - 1) // TILE_KEYS)
```
`flat=1` is gated on `not window`. A windowed shape can *never* reach the one-wave partition: it always falls
through to `_splits(tiles, target // reqs)`, which cuts 32 tiles into 16 shares (dec16), 8 (dec32) or 4 (v32).
So the sliding cells are **2–8-tile split-KV CTAs by construction**, and 32 KB of lookahead is amortised over
four tiles instead of a hundred and fifty.

### 3.3 Which mechanism, evidenced rather than asserted

* **Depth is worth +1.29x on the ≥3-tile cells, measured.** `attn_gap1.py` ships the identical file with
  `OKC_STAGES 3 → 4` (32 → 48 KB in flight, same grid, same math, bit-identical output). On the H200 box:
  dec32 `9.84 → 7.6`, v16 `11.34 → 8.8`, v32 `14.58 → 10.0` µs. Depth at fixed row size bought 24–31% on exactly
  the three cells with runway. If the binding term were row size, raising STAGES could not have done that.
* **Depth is worth ~0 on the 2-tile cell.** Same measurement: dec16 `7.84 → 7.8`. With 2 tiles per CTA the
  prologue *is* the kernel; there is nothing to overlap. This is the part of the gap that only the persistent
  one-wave windowed partition can reach (`attn_gap.md` §2, last row) — a host segment plan, not a kernel edit.
* **`page_size = 1` caps everyone.** Each position is its own page, so the gather is a random walk over 128 B
  lines with no sequential neighbour to share a burst. The published page-1 ceiling is ~54% of peak
  (Heidari et al., quoted in `attn_gap.md` §3) = **2.59 TB/s on H200**. Our sliding cells sit at 0.77–1.57 of that,
  and our *flat* cells at 3.07–3.65 prove the row size is not what stops them.

Verdict on the decisive question, priced: **row size 0%; lookahead and runway 100% of the remaining gap.**

---

## 4. Board arithmetic

The board is a geometric mean over 6 loads, each load = `46 × sliding + 16 × full` (weights from
`problems/muse_attention_decode/TASK.md:35–48`). A shape change therefore moves one sixth of the log-sum. Isolated
gain for a single cell, and the gain when all four move together:

| scenario | dec16s | dec32s | v16s | v32s | board geomean | all four |
|---|---|---|---|---|---|---|
| **A** — 3.30 TB/s (== what our own flat cells already do at page 1) | 2.40 µs (3.27x) | 4.79 (2.10x) | 2.71 (4.30x) | 5.41 (2.69x) | **+10.03%** | 2.66 → 2.93 |
| **B** — 2.59 TB/s (4.8 × 0.54, the published page-1 ceiling) | 3.06 (2.56x) | 6.10 (1.65x) | 3.45 (3.38x) | 6.89 (2.11x) | **+8.55%** | 2.66 → 2.89 |
| **C** — depth only, `OKC_STAGES 3→4`, measured | 7.80 (~0) | 7.75 | 9.03 | 10.00 | **+2.59%** | 2.659 → 2.728 |

Per-cell share of the board, scenario A: dec16 **+2.61%**, v16 **+3.65%**, v32 **+2.08%**, dec32 **+1.35%**.
Scenario C: v32 **+1.00%**, v16 **+0.97%**, dec32 **+0.57%**, dec16 **+0.02%**.

Note what A and B are *not*: they are not achievable by any row-size change, because the read is already 128 B.
They are the ceiling of the lever we already have — a persistent windowed partition (needed for dec16/v16) plus
deeper lookahead (dec32/v32) — and C is the part of it already built and measured. The single most useful reading
of the table: **the 128 B mechanism the competitor credits for +0.158x is worth 0% here, and the depth mechanism
we already have in `attn_gap1.py` is worth +2.6% of it before any windowed partition work.**

---

## 5. Kill criterion

Kill the whole "128 B head-packing" line — do not ship a candidate, do not spend a build on it:

1. **Static (already met):** if any sliding-window gather's address arithmetic reads a head at `+64 B` from a
   second, separate fetch, it is a real gap. It does not: `pair*128 + cchunk*16` over `cchunk ∈ [0,8)` covers one
   aligned 128 B window, and `row_bytes = Hkv*64 = 128`. `grep -n 'pair \* 128 + ' dual.py` returns one pattern
   in seven places (1035, 1119, 1124, 1173, 1193, 1199, 1330), all 128 B wide. **Kill.**
2. **Measured (already met):** if a long-grid sliding cell could beat 1.57 TB/s by fixing rows, then the same
   rows at long grid would not already run at 3.65 TB/s. dec32 full does, on the same page-1 random table.
   **Kill.**
3. **Would revive it:** a scored cell reading < 10 000 B per position while a `64` literal appears in its global
   offset. None does.
4. **Applies to the surviving lever instead:** `attn_gap1.py`'s `OKC_STAGES 3→4` is killed by `0/12 picks` or by
   any check-level geomean below the base row (2.6590) — its own criteria, unchanged, in `attn_gap.md` §6.

## 6. What this report did not do

* No `attn_pack128.py`. The brief's second branch (implement behind a default-off define) does not apply: the
  paired read is shipped, so a define would be a no-op fork of the file. Lint/`cu_build.py` therefore have
  nothing new to check.
* Not run: `okc`, `colab`, the judge. Every number above is either read from the CUDA source with a line number
  or taken from `attn_gap.md`'s H200 profile of this file; the board arithmetic is a formula, reproducible from
  §4's inputs.
