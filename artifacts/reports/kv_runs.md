# Page-table run collapse for the KV stream — DEAD for muse_decode_layer (layout proof, op census, price)

Verdict: **stop.** The address arithmetic supports the idea, but the judge's page ids are a
*random permutation*, so a 2,048-position window is not a run — it is 2,048 scattered 128 B rows.
Expected runs of >= 4 pages per window: **4.8e-15**. The mechanism can never fire, and there is
almost nothing left to win anyway: the KV stream already runs at **4.17 TB/s = 93 %** of the
size-matched H200 read ceiling.

---

## 1. Layout: contiguous in page id — but the ids are not adjacent

`k_cache` / `v_cache` are `float8_e4m3fn [pages, page_size, Hkv=2, 64]` (`problems/muse_decode_layer/
composite.py:107-108`); the kernel views each as rows of `row_bytes = Hkv*64 = 128` B
(`klr_qd2.py:3677`), `lp = log2(page_size) = 0`, `lrb = 7`.

Per-row byte address used by the producer (`klr_qd2.py:1450`, `1435`):

    off = ((id << lp) + (pos & (P-1))) * 128 + pair*128 + chunk*16
        = id * 128 + chunk*16                    (P = 1, pair = Hkv/2 - 1 = 0)

So **page id `p` owns exactly `[p*128, p*128 + 128)` of that tensor** — one full KV pair (both `Hkv`
heads, 2 x 64 dims, inside one 128 B row; `pair*128 == 0` for `Hkv = 2`). A run of L adjacent ids
occupies `L*128` contiguous bytes, and K and V are **two separate tensors, hence two runs** of
`L*128` B each (`p.k_cache` / `p.v_cache`), never one interleaved run. The row is 4 full 32 B
sectors, so a scattered row wastes no DRAM bytes (matches the paper's 1.001x).

That is the whole good news. The ids themselves come from

    perm = torch.randperm(batch * max_pages)                     # reference.py:261, "-1 unused"
    page_table[b, :n] = perm[offsets[b] : offsets[b] + n]

a **random permutation of the whole pool**, regenerated for every judge repetition
(`judge.py:55` → `generate_input(seed + 1009*(i+1))`, plus a `+1000003` freshness pass). For a
uniform random permutation of N pages, P(id[i+1] = id[i]+1) = 1/N, so in one request's window:

| quantity (decode 32 full: N = 32 x 80,001 = 2.56 M, window = 80,000 pages) | expectation |
|---|---|
| adjacent-increasing pairs (a 2-page run of 256 B) | **0.031** |
| 4-page runs (the flag's minimum) | **4.8e-15** |
| 8-page runs (one TMA swizzle atom, 1 KB) | ~0 |
| 64-page runs (one whole tile, 8 KB) | ~0 |

The sliding shapes are worse: a 2,048-page window over the same N gives 8e-4 expected 2-runs.
`muse_attention_decode` uses the identical generator (`reference.py:187`), so nothing changes there.

The obvious counter — "sort the tile's ids and merge physically contiguous spans, not logically
contiguous ones" — also fails: a window is a random 2,048-subset of 2.56 M pool slots, so it contains
**~1.6 physically adjacent pairs** on average (m(m-1)/N), each a 256 B span, far below the 4-page
minimum; and a physical span maps to *arbitrary* key positions, whose smem slots are not adjacent, so
it would need a scatter of exactly the same 16 B ops it was meant to replace.

## 2. Producer op census (was 16 copy ops per 128 B page; a run producer would be 2 per tile)

Scored full-attention shapes are `flat=1` (B > 1) with wargp specialized producers (`Cfg::WS`,
`OKC_WS8 = 1`, `NP = 128`, `PR = 16`, `PK = 4`). Per 64-page tile = 16 KB of KV (`klr_qd2.py:1353-1382`):

- **1024 `cp.async.cg` 16 B ops per tile** (128 threads x 4 rows x K,V) = **16 ops / 128 B page**;
- 512 page-table `__ldg` per tile (4/thread, 8x redundant — the 8 chunk threads of a row re-load it);
- ~25 instructions per thread per tile, i.e. **~100 warp-instructions per 16 KB = 0.006 warp-instr/byte**.

Budget: 4.17 TB/s over 132 CTAs = 31.6 GB/s/CTA, so a 16 KB tile lasts **0.52 us = 830 cycles**;
100 warp-instructions over 4 schedulers = **25 cycles = 3.0 % of the tile**. The producer is not the
limit — which is exactly why the paper's 54 %-of-peak regime does not apply: that is a single-warp
per-page producer (2,048 sequential loads, one copy op each), while ATT7 has 128 producer threads,
8-way row splitting and a 4-stage pipeline, and lands **655.4 MB / 157 us = 4.17 TB/s**.

A run-collapsed producer would issue **2 ops/tile** (one 8 KB bulk per stream, 512x fewer) or 16 with
8-row boxes. Both are unreachable (section 1) and both would buy >= nothing measurable (section 3).

Independent confirmation that the *form* of the copy is inert here: a unified harness on H200 NVL
(sm_90a, producer thread + mbarrier'd smem ring, 256 B-16 KB chunks, 1-32 stages) measures 1-D
`cp.async.bulk` at 4,287 GB/s, `cp.async` at 4,277 and plain loads at 4,139 — 89.3 % of the 4.8 TB/s
peak for the *best* form, i.e. no copy instruction closes the DRAM gap. Tile mode and bulk also lower
to the same per-SM TMA engine (`UTMALDG` / `UBLKCP`) and are statistically identical above ~32 KiB
bytes-in-flight per SM; ATT7 holds 4 stages x 8 KB x 2 = 64 KB per CTA, so it is already in the
converged regime. Our 4.17 TB/s sits within 3 % of that harness's best.

## 3. Price (upper bound, assuming a perfect producer and the run precondition somehow held)

- decode 32 full KV = 32 x 80,000 x 256 B = **655.4 MB**, 157 us today at 4.17 TB/s.
- Size-matched DRAM ceiling at 655 MB = **~4.50 TB/s** (kernarc_digest: 4.60 @ 2 GiB, 4.52 @ 512 MiB).
  Floor = 146 us -> **<= 10.7 us off a 473 us step (-2.1 % of that load)** and that is the *entire*
  DRAM gap, not the issue-cost part (issue cost is 3.0 % of the tile, <= 4.7 us).
- Score: one load owns 1/6 in log space (PROGRESS:334) -> -2.1 % on one load = **-0.35 % of the
  layer score**, so ~**-0.1 %** from the issue-cost share (0.1 score-%, vs MoE's +8.2 %).
- Byte efficiency is already 1.0: a 128 B row is 4 full sectors, so there are no wasted bytes to
  reclaim either.

## 4. The other instrument, considered and closed: per-row `cp.async.bulk.tensor.2d`

The only way to get to 1 op per 128 B row. `row_map()` (`klr_qd2.py:3622`, box {128 B, 1 row},
`CU_TENSOR_MAP_SWIZZLE_128B`) already exists and `p.tk` / `p.tv` are built for it, but the attention
kernel never uses them (`p.tma` is set and read nowhere; `OKC_TMA` = 0). Closed for two reasons:

1. `cp.async.bulk` **1-D cannot produce the smem layout**: the QK^T operand is read by a wgmma with
   `desc_sw128` (128 B swizzle, chunk `c` of row `r` stored at `c ^ (r & 7)`, `klr_qd2.py:521-522`),
   and no linear copy yields that permutation; no-swizzle core-matrix layouts (as sk2 uses) are a
   row *transpose* per 1 KB atom, also not linear. Only the TMA tensor path carries the swizzle.
2. **The 1-row box does work** (resolved after this analysis started): PTX ISA §5.5.7 keys the SW128
   XOR on the *absolute* `dstMem` shared address — "swizzle base offset = (`dstMem` / 128) % 8, 0 at a
   1024-Byte boundary" — not on the box row. So dst `= stage_base(1024-aligned) + r*128` lands chunk
   `c` at `c ^ (r & 7)`: byte-identical to the `cchunk ^ (r & 7)` cp.async16 write, 1 op per 128 B row,
   128 ops/tile instead of 1024, no self-check needed. It is closed on *economics*, not mechanism: the
   copy form is measured inert at our in-flight depth (64 KB/CTA; 1-D bulk 4,287 GB/s vs cp.async
   4,277 vs plain loads 4,139, and TMA-2D loses its ~170-cycle issue tax only below ~16 KiB BIF/SM),
   the producer's whole issue budget is 3.0 % of the tile, and the stream is at 93 % of the ceiling.
   ~128 lines plus one Colab gate for <= 0.1 score-%; revisit under section 6.

## 5. Gates

**No candidate, no delta, by design** (this ticket's own stop condition: the pages are not contiguous).
No file was written to `klr_runs.py`; an unreachable 200-line producer in a 979 KB file is exactly the
weight to avoid. Nothing to lint, nothing new to compile:

- `okc_loop.cli lint klr_qd2.py`: n/a (unchanged).
- Colab sm_90a gate / `cu_build.py --variant MID`: not run — no source changed, and the board build
  (`82c098ae`, judge 2.2536) is already known-good at sm_90a.

## 6. Revive only if

**Run collapse** (2 ops/tile) needs all three:

1. The harness stops scattering: `page_table` contiguous per request (vAttention/VMM or a
   `page_size >= 8` scored case) — then a tile is one 8 KB run and 2 ops/tile is real;
2. the KV stream measures below ~80 % of the size-matched read ceiling (it is at 93 %);
3. the producer's issue share is measured above ~10 % of the tile window (it is at 3 %).

**Per-row TMA** (128 ops/tile, layout proven) needs only 2 and 3 — and would still cap at ~3 % of the
KV stream, i.e. ~0.05 score-%. Measure the producer's issue share with NCU (`smsp__inst_executed` on the
producer warps vs the tile window) before writing it.

Otherwise the money is in MoE (47.1 % of the score, 2.51 TB/s against a 3.19 TB/s per-SM fetch cap,
worth +8.2 % of the layer), not in the 15.2 % attention term that is already at its byte floor.
