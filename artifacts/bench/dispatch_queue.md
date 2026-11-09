# Layer queue — state at 2026-11-03 07:00 UTC

**Banked: 2.28370 / 9,905.1 µs, rank #84 of 1221** (board #1 is now 2.4902 and rising while our score is fixed).
Layer quota: **0 checks, 7 runs**; fresh 20/10 at 00:00 UTC Nov 4, contest closes Nov 5 07:59 UTC.

| # | candidate | evidence | expected | state |
|---|---|---|---|---|
| 1 | `layer/layer_build_tune.py` | per-T MoE build selection measured in `prepare()`; the thresholds (`MOE_TINY_MAX_M`/`TINY2`/`MID`) were inherited from the standalone and never measured on this file; non-regressive (exact fallback, swaps need `max|d|==0` and +1.5%/1.5 µs) | +0.0 … +1.3% | **submitted `6536485f21d443a7`** |
| 2 | `layer/t1_layer_build.py` | T=1 path: the three builds are define-identical; the only delta is the `TIDX` warp-role rotation, and the 29.4-vs-22 µs gap is at least partly PDL overlap attribution (the profile itself says an overlapped kernel's time includes its predecessor's wait) | unproven | **hold** — needs the per-SM timeline, not a run |
| 3 | `layer/klr_fci5.py` | the row | — | control |

**Nothing else on the layer has evidence.** Every mechanism we hold is measured and below the row
(see `docs/PROGRESS.md` §"Measured and BELOW the row" and keell0renz's own `LESSONS.md` dead list).
The 20-check bucket at 00:00 is therefore **held for whatever the borrowed H200's timeline produces**,
not spent on re-tests. If no H200 arrives, the bucket's best use is (a) one conformance check of
`layer_build_tune.py` to see whether its tuner actually swapped anything, and (b) a second draw of an
equivalent mechanism — which is explicitly *not* to be done (noise farming).

## Other boards (all advanced today, all with their own quota)
| board | row | rank | ranks/+1% | banked today |
|---|---|---|---|---|
| prefill | **1.9087** | #85/831 | 29 | +1.04% check / +0.65% submitted (`ROUTER_MM_SMALL`) |
| bf16 | 1.5891 | #74/1672 | 48 | `bf16_selftune.py` in flight (per-cell route timing, non-regressive) |
| MoE | 2.3746 | #246/1346 | **101** | `moe_selftune.py` in flight (per-bs build selection) |
| attention | 2.6590 | #126/1289 | 33 | 4 candidates measured today, all below; STAGES 3->4 dead |
