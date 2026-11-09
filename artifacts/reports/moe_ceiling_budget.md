# MoE board: per-bs budget, the 2.764 arithmetic, and the one default-off lever

**2026-11-03.** Board read live: `muse_moe_w4a16` rank **245** of **1,342** (the brief's 1,219 is the
earlier snapshot), score **2.3746** / 5,663.5 µs, job `a857807a067a4ad9`, sha `6402f5f3…d2b3e5`.
Row 20 (`colinlikescode`) = **2.5760** → **+8.48%**; #1 (`dolanzhao`) = **2.7642** → **+16.41%**.
Times are the submit profile's `median_ns` (judge subtraction applied); bytes are the replica in
`moe_bytes.md`, validated to <1% against the published floor.

## 1. The budget: every scored bs at the size-matched ceiling

`4.287 TB/s` is **not** a universal ceiling: it is a 256 MiB point on a strongly size-dependent
curve (NCU/events: 66 MiB 3.63–3.81, 256 MiB 4.33/4.17, 512 MiB 4.40–4.52, 1 GiB 4.51–4.56,
2 GiB 4.60). Use each load's own byte count. The ceilings below interpolate that curve (≈±3%);
`ceiling µs` = actual bytes / ceiling, `alone` = board score if only that load reaches it.

| load | bs | base µs | ours µs | ratio | actual MB | ours TB/s | ceiling TB/s | ceiling µs | save µs | alone | alone gain | per µs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode 1 | 1 | 77.09 | **24.93** | 3.093 | 72.7 | 2.917 | 3.72 | **19.54** | 5.38 | 2.4729 | +4.14% | 0.669% |
| verify 1 | 8 | 149.49 | 67.82 | 2.204 | 238.4 | 3.515 | 4.30 | 55.44 | 12.38 | 2.4557 | +3.42% | 0.246% |
| decode 32 | 32 | 206.69 | 125.28 | 1.650 | 451.9 | 3.607 | 4.45 | 101.55 | 23.73 | 2.4592 | +3.56% | 0.133% |
| verify 16 | 128 | 458.05 | 172.75 | 2.652 | 648.6 | 3.755 | 4.50 | 144.13 | 28.62 | 2.4474 | +3.06% | 0.096% |
| decode 16 | 16 | 185.66 | 91.28 | 2.034 | 347.0 | 3.802 | 4.40 | 78.86 | 12.41 | 2.4332 | +2.47% | 0.183% |
| verify 32 | 256 | 614.77 | 207.99 | 2.956 | 738.3 | 3.550 | 4.53 | 162.98 | 45.01 | 2.4731 | +4.15% | 0.080% |
| **all six** | | | | | | **3.511 geo** | | | **127.5** | **2.9129** | **+22.66%** | |

`our geomean landed TB/s = 3.511`. But the kernel's **stream phase** — the bursts between unit
ends — already measures **3.87–4.05 TB/s**, i.e. ~90% of these size-matched ceilings; the landed
rate is lower because every 16-stage unit end costs a CTA **1–7 µs**, and at d32 the decomposed event
budget is **9–12 µs of 125 µs** (ramp 3–6, phase-A waves 3–6, shadows 4–12, end 7–11). Decode 1 is
the outlier *landed* (2.917) and worth 8.6× verify 32 per µs, but it is not a streaming shape.

## 2. What 2.7642 must be doing

With fixed bytes, `score = geomean(base_i · B_i / bytes_i)`, so

```
2.7642 / 2.3746 = 1.1641 = geomean(B_i) / 3.511   ⇒   geomean landed B = 4.087 TB/s
```

The ceiling scenario, using each load's own ceiling:

* The **maximum possible board is 2.9129** (+22.66%). Dolanzhao's 2.7642 is **94.9% of that max**.
* If the five streaming loads land at their size-matched ceilings, decode 1 only needs **26.78 µs
  (2.715 TB/s) — 7.4% slower than ours**. So 2.764 does **not** require fixing decode 1 at all.
* Conversely, with decode 1 at its 3.72 TB/s ceiling, the other five need only **4.165 TB/s** —
  just 9.5–15.5% above the five's landed 3.52–3.80 TB/s.
* **Bytes cannot explain it.** One expert is exactly the floor's 28.31 MB; the only duplicated
  stream is verify 32's Σjobs/Σdistinct = 1.060 (≤0.8% of the board). The lossless 6-bit-scale
  option is −2.78% bytes. Both together leave `B_req ≈ 3.97`, still above every landed load we have.

**Conclusion.** Dolanzhao is landing roughly **4.05–4.5 TB/s on the five streaming loads**; how
good his decode 1 has to be depends where in that band he sits — at the five loads' own ceilings
decode 1 may be **2.72 TB/s, worse than ours**, while at 4.165 on the five it must be at its 3.72
ceiling. Either way the discriminating change is not decode 1: it is that his five streaming loads
land at **our own measured stream-phase rate (3.87–4.05) or above**, not a new GEMM or a byte trick.
In other words: he has the same stream and pays far less per unit end/ramp/tail; or, equivalently,
he runs at the 512 MiB-class ceiling where the size-matched measurements say a clean 1-D read
should. Our per-load landed 3.5–3.8 (2.9 at d1) is precisely the 10–20% that the
ramp/phase-A/shadows/unit-end stalls remove.

Two actionable readings:

1. **The rate is not the wall** — our bursts are already 90–92% of the size-matched ceiling. The
   remaining 10–12 µs at d32 lives in the event budget; the surviving untested direction there is
   fewer fences/releases per CTA and phase order that does not lengthen the tail (the published
   refits halved phase-A spread but lost 0.8–4.6 µs on judge cases).
2. **Decode 1 is still the cheapest point per microsecond** (0.669%/µs, 8.6× verify 32), so the
   TINY build is the right first move for *rank*, even though 2.764 does not mathematically need it.

## 3. The default-off candidate

`src/okc_loop/vendored_work/moe/moe_dual.py` — the 2.3746 incumbent plus one compile-time flag,
**`OKC_DE` (default 0; the one-line switch is `DE = False`, next to `DQ7`)**. Under `OKC_DE`, the
WGS weight producer splits each 36,864 B stage into

* **24,576 B TMA bulk** (`cp.async.bulk`, lane 0, `.L2::cache_hint` preserved), and
* **12,288 B `cp.async.cg` (LDGSTS)**, all 32 producer lanes, 24 × 16 B each,

into the same slot; the full mbarrier is initialised with **33 arrivals** (1 TMA + 32 cp.async via
`cp.async.mbarrier.arrive.noinc`) instead of 1. The split is 128 B aligned, so no sector is written
by both engines. WGS is the build for bs 32/128/256, so the candidate measures three loads.

This is the **rate** probe, and it is deliberately the only untested mechanism left on that axis:
the quiet stretches of a rival ring measure 4.39–4.64 TB/s and the size-matched ceiling is
4.35–4.45, so two engines on one stage can close the stream-phase gap if the single issue path is
what limits it. It does **not** address the stall budget; if the stall diagnosis is the real one,
`OKC_DE` measures flat, and that is the decisive negative result that closes the rate axis.

Bounded expectation (bytes unchanged, other three loads unchanged):

| if WGS loads land at | board |
|---|---|
| 3.90 TB/s | 2.4593 |
| 4.05 TB/s (our measured stream-phase rate) | 2.5061 |
| 4.35 TB/s (size-matched ceiling) | 2.5973 |
| 4.45 TB/s | 2.6269 |
| all six at 4.05 (needs the same split in TINY/TINY2/MID) | 2.7390 |
| all six at their size-matched ceilings | 2.9129 |

Offline verification: `okc-loop lint` PASS, and `tools/cu_build.py` (nvrtc → ptxas, sm_90a) ok on
all five build tuples with `OKC_DE` both 0 and 1. No GPU, no judge run.

## 4. Spend order if a check is available

1. Check `moe_dual.py` with `DE=True` once. A 5–8% move on decode 32 / verify 16 / verify 32
   confirms the rate axis and prices the extension; a flat result confirms the event budget and
   closes the rate axis for good.
2. If (1) lands, add the same 2/3 TMA + 1/3 cp.async split to `ring_issue` for TINY/TINY2/MID
   (decode 1 alone is +4.14% at its own ceiling and 0.669%/µs).
3. Do not spend a check on byte tricks: SC6 is −2.78% bytes and the only duplicated stream
   (verify 32) is ≤0.8%.
