# Does `okc::moe_fused` move more bytes than the judge's floor, or the same bytes slower?

**Answer: the same bytes, slower.** At all six scored shapes the kernel fetches exactly
768 units x 36,864 B = **28.3116 MB per touched expert per job** — byte-for-byte the floor's own
per-expert weight term — and reads each touched expert's matrix **once** (Σjobs/Σdistinct =
1.000–1.060). What separates us from the floor is delivered bandwidth: **2.91–3.80 TB/s**
(22–29 GB/s/SM on 132 SMs), i.e. 61–79% of the 4.8 TB/s the floor is priced at.

Sources: board candidate `artifacts/cas/6402f5f3…d2b3e5.py` (2.3746, job `a857807a067a4ad9`),
its profile `.txt`, `problems/muse_moe_w4a16/{reference,TASK}.md`, layer lineage
`src/okc_loop/vendored_work/layer/klr_layer_clean.py` (= sha 82c098ae).

---

## 1. Bytes per call, from the source

| fact | value | source |
|---|---|---|
| unit (one `cp.async.bulk`/TMA copy) | `UNIT_CODES 32768` + `UNIT_SC 4096` = **36,864 B** = 16 rows x 4,096 K of NVFP4 codes + e4m3 scales | `…d2b3e5.py:657` |
| 16-row blocks: gate/up / down | `RB_A = 2*ID/16 = 512`, `RB_B = HD/16 = 256` | `:333` |
| w13 per expert | 512 x 36,864 = 18,874,368 B = 8192x4096x(0.5+1/16) | `:333,657` |
| w2 per expert | 256 x 36,864 = 9,437,184 B = 4096x4096x(0.5+1/16) | `:334` |
| **per expert, per job** | **768 units = 28,311,552 B** | sum of the two |
| fat path: one 36 KB "stage" = 16 rows x 256 K, 16 stages per 256-row unit | `NKS = HD/KS = 16`, `UB_A = 32`, `UB_B = 16` -> 32x16 + 16x16 = 768 stages | `:1378-1382`, `:2368`, `:1131` |

No per-expert padding: 768 units x 36,864 B is *exactly* `3*D_EXPERT*D_HIDDEN*(0.5 + 1/BLOCK)`,
the judge's `per_expert * (0.5 + 1/BLOCK)` term (`reference.py:132-134`). The weight layout has no
rounding waste (8192 and 4096 rows are both multiples of 16).

## 2. Routing -> jobs -> CTAs: can one expert's weights be fetched twice in one call?

`route()` (`…d2b3e5.py:545-583`) counts, per local expert `lane`, its tokens `tot`, then:

```
nfj = tot > SK_T  ? (tot + FT_T - 1) / FT_T : 0     # fat jobs of <= FT_T = 64 tokens   (:551)
nsj = 0 < tot <= SK_T ? 1 : 0                       # one skinny job of <= SK_T = 8      (:552)
S.jE[fj+q]=lane; S.jS[fj+q]=off+q*FT_T; S.jC[fj+q]=min(FT_T, tot-q*FT_T)   (:563)
```

A job is a (expert, token-slice) pair and each job streams the expert's whole matrix
(`fat_stream`: `job_of(uu) = uu/UBS`, `nF = njF*ubs` at `:2368`; skinny: `blk = jE[j]*rbs + rb`,
`:686`). Units inside a phase are partitioned, never re-issued: the CTA range is
`muldiv_g(nS, blockIdx.x, …)` over a static part plus an atomic-claimed pool tail (`:1574`,
`:2492`), and in stream-K (WGS) the stages are split by cost prefix (`wgs_range`, `:2338`) so each
(stage) is in exactly one CTA's range. So:

**bytes_fetched = 768 x 36,864 x Σ_e jobs(e)** — with duplication only if `tot_e > FT_T` (64),
which requires a hot expert at bs>=128.

Variant -> cap: M<=4 TINY and M<=8 TINY2 have `OKC_SPLIT 0` -> one job per expert whatever `tot`
(`:545-552`, `:245`); M<=16 MID sets `OKC_SPLIT 1` -> `ceil(tot/8)` jobs (`:549`, `MID_MAX_M 16`);
M>16 MAIN/WGS -> `ceil(tot/64)` (`:243,551`, `WGS_NT = FT_T/8` `:1825`). Scored shapes:
bs1=TINY, bs8=TINY2, bs16=MID, bs32/128/256=MAIN/WGS (`kernel_fn`, `:3386-3391`).

## 3. Real routing per shape (replica)

`reference.generate_input` needs a CUDA generator; this box has no GPU/torch, so the routing was
replicated in numpy exactly as `generate_input` draws it (`scores = U + 0.03*z/sqrt(-log u)`,
top-8, busiest-GPU relabel), 4000 seeds/shape. **Validation**: the floor's own
`touched = max(1, 0.75*router_distinct)` back-solves to distinct = floor_bytes/28.3116 MB/0.75;

| shape | distinct implied by the published floor | replica |
|---|---|---|
| decode 1 | 2.582 | 2.57 |
| verify 32 | 24.63 | 24.61 |

Agreement to <1%: the replica and the byte model are right.

## 4. Table

`floor_bytes = floor_us x 4.8e12`; `bytes` = replica mean; `TB/s` uses the scored `yours` time;
`dup = Σjobs/Σdistinct` (the kernel's real duplication, 1.000 = each expert read once);
`vs floor` = bytes_fetched/floor_bytes, whose *no-duplication reference is 1/0.75 = 1.333*.

| shape | bs | build | distinct | jobs | bytes_fetched | floor_bytes | vs floor | dup | yours µs | TB/s | GB/s/SM | vs 4.287 TB/s | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode 1 | 1 | TINY | 2.57 | 2.57 | **72.7 MB** | 54.8 MB | 1.325 | 1.000 | 24.93 | 2.91 | 22.1 | 0.68 | bandwidth-limited |
| decode 16 | 16 | MID (cap 8) | 11.93 | 12.26 | **347.0 MB** | 252.1 MB | 1.376 | 1.027 | 91.28 | 3.80 | 28.8 | 0.89 | at ceiling |
| decode 32 | 32 | MAIN/WGS (cap 64) | 15.96 | 15.96 | **451.9 MB** | 339.6 MB | 1.331 | 1.000 | 125.28 | 3.61 | 27.3 | 0.84 | at ceiling |
| verify 1 | 8 | TINY2 | 8.42 | 8.42 | **238.4 MB** | 179.1 MB | 1.331 | 1.000 | 67.82 | 3.52 | 26.6 | 0.82 | bandwidth-limited |
| verify 16 | 128 | MAIN/WGS (cap 64) | 22.56 | 22.91 | **648.6 MB** | 486.4 MB | 1.333 | 1.015 | 172.75 | 3.75 | 28.4 | 0.88 | at ceiling |
| verify 32 | 256 | MAIN/WGS (cap 64) | 24.61 | 26.08 | **738.3 MB** | 522.9 MB | 1.412 | 1.060 | 207.99 | 3.55 | 26.9 | 0.83 | bandwidth-limited (5.6% dup) |

Cross-check against the ticket's second reference frame: `bytes_fetched / (yours x 4.287e12)` =
0.68 / 0.89 / 0.84 / 0.82 / 0.88 / 0.83 — all **< 1.0**, so the kernel never exceeds the bytes a
4.287 TB/s stream would carry; nothing is hidden.

Non-weight traffic (ESTIMATE): inter 8 KB write+read and ybuf f32 16 KB write+2 reads per route,
plus x/out — ~0.2 MB at bs=1, ~29 MB (4%) at bs=256. Included nowhere above.

## 5. Verdicts

* **No traffic duplication** on any scored shape. The three "ratio" values that sit above 1.15
  (1.325–1.412) are the judge's own 25% discount: `ROUTING_SHARE = 0.75` (`reference.py:122`), so a
  dedup-perfect kernel lands at exactly 1.333. `dup` (the honest measure) is 1.000–1.060.
* **Bandwidth-limited / at ceiling.** 2.91–3.80 TB/s vs 4.8 TB/s priced peak (61–79%), 0.68–0.89 of
  the 4.287 TB/s achievable read rate. decode 1 is the one shape with a real gap (0.68 of
  achievable, 22.1 GB/s/SM); the rest sit at 0.82–0.89, i.e. 26.6–28.8 GB/s/SM.
* Per-unit cost from the profile GPU times: 36,864 B per CTA in 1.27–1.57 µs = 23.5–29.0 GB/s/SM;
  the constant is set by the **single-issue TMA fetch path per SM**
  (`ring_issue :683`, `fat_issue :1131`, `ws_issue :1702`, WGS producer `:1981`), not by bytes,
  compute or occupancy (1 CTA/SM, `__launch_bounds__(NTL,1)`; 4x more wgmma compute in WGS does not
  move the per-unit time). Ring depth is `OKC_NS`/`WGS_NS = 3` (`:1823`).

### Layer lineage (also checked)
`klr_layer_clean.py` has the identical job model (`nfj/nsj` at `:4254-4255`, `FT_T 64` `:3941`), so
its MoE is byte-identical, only 2.8% smaller per expert: SC6W 6-bit scales make a stage 35,840 B
(`WGS_STG` `:5551`, on — the job log says `[sc6] scales ok -> wgs sc6 True`), i.e.
768 x 35,840 = **27.525 MB/expert** -> 70.7 / 337.4 / 439.4 / 231.8 / 630.6 / 717.8 MB per shape
(ESTIMATE, same jobs). The verdict transfers unchanged; the layer's MoE µs are not in the profile
JSON, so its GB/s is not computed here.

## 6. The only real duplication, priced

verify 32: 1.46 extra expert streams (26.08 jobs vs 24.61 distinct) = **41.5 MB = 11.7 µs** at
3.55 TB/s, i.e. **+0.79% of the MoE board**. Cause: `OKC_FT_T 64` (`:243`) -> `FT_T:332`,
`WGS_NT = FT_T/8 :1825`; an expert with 65–128 tokens is split into two jobs that each re-stream
28.31 MB (`:551,563`). Fix if wanted: raise the fat/WGS job cap to 128 (16 n-tiles) — blocked by the
96-register cap (`:1860` comment) — or, cheaper, keep an expert's split jobs adjacent so the second
reads its 28.31 MB out of L2 (H200 L2 = 50 MB): WGS already keeps expert order; the non-WGS fat
`forder` LPT reorder (`:572-583`) is what would break it. Not worth the quota: ≤0.8%.

## 7. The largest recoverable amount

Byte-optimal already, so the recoverable amount is the bandwidth gap, and 1 µs is not worth the
same on every shape (`+% = ln(yours/(yours-1))/6`, each shape 1/6 of the MoE geomean):

| shape | 1 µs = | to 4.287 TB/s | gain |
|---|---|---|---|
| decode 1 | **+0.682%** | 24.93 -> 16.95 µs (8.0 µs) | **+6.43%** |
| verify 1 | +0.248% | 67.82 -> 55.61 µs (12.2 µs) | +3.31% |
| verify 32 | +0.080% | 207.99 -> 172.22 µs (35.8 µs) | +3.15% |
| decode 32 | +0.134% | 125.28 -> 105.42 µs (19.9 µs) | +2.88% |
| verify 16 | +0.097% | 172.75 -> 151.29 µs (21.5 µs) | +2.21% |
| decode 16 | +0.184% | 91.28 -> 80.94 µs (10.3 µs) | +2.00% |
| all six | | 107.6 µs total | **+19.98% -> board 2.3746 -> 2.8997** |

**Largest recoverable: decode 1's 8.0 µs = +6.43% of the MoE board** (it is the only shape under
3.5 TB/s: 22.1 GB/s/SM, 15 units/CTA so ring fill and cold-L2 start dominate). Fix: make the M<=4
TINY build fetch at the rate its own family already reaches — MID (`M<=16`, same `NW=8`/`WS`/
skinny-only skeleton, only `OKC_SPLIT`/`WSBLK=2` + the `moe_helper0` helper source differ,
`:3455-3458`) delivers 28.8 GB/s/SM where TINY delivers 22.1 — so A/B `bs=1` and `bs=8` on
`MID_DEFINES` before touching the fetch path; if that does not transfer, the lever is the per-SM
issue path (`ws_issue :1702` / WGS producer `:1981`) and ring depth, i.e. the dual-engine
TMA+`cp.async` prologue already specced in `docs/DEVIN_HANDOFF.md:113`.

## 8. Correction to prior notes

`docs/PROGRESS.md:54-56` reads "523 MB = 32 local experts x 16.3 MB, i.e. each expert once". The
conclusion (each expert once) is right, the per-expert size is not: one expert's matrix is
**28.31 MB**, and 523 MB is 18.5 touched experts = `0.75 x 24.6 distinct`, exactly the floor's
discount. Same shape, correct arithmetic.
