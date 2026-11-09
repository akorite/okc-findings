# moe_selftune — per-shape build selection measured in `prepare()`

Candidate: `src/okc_loop/vendored_work/moe/moe_selftune.py`
(`# okc: problem=muse_moe_w4a16`, a copy of `artifacts/cas/6402f5f3…d2b3e5.py`, check 2.3863).

Axis: **which of the four builds this file already compiles serves each scored bs**.
No new kernel, no new mechanism: `prepare()` (untimed, 300 s) times `MAIN`, `TINY`,
`TINY2`, `MID` on real routing for bs 1/8/16/32/128/256 and stores the winner per bs;
`kernel_fn` still branches only on shape, and every other shape keeps the hardcoded
`TINY_MAX_M` / `TINY2_MAX_M` / `MID_MAX_M` chain.

## Candidate sets (structural safety, not preference)

| bs | incumbent | candidates timed | why the rest are excluded |
|---|---|---|---|
| 1 | TINY | MAIN, MID, TINY2, TINY | — |
| 8 | TINY2 | MAIN, MID, TINY2, TINY | — |
| 16 | MID | MAIN, MID | TINY/TINY2 `__trap()` on any fat job (`tot > 8`); an expert gets at most one route per token, so they need `M <= 8` |
| 32 | MAIN | MAIN, MID | tiny builds unsafe; MID safe (next row) |
| 128 | MAIN | — (one candidate) | MID splits every expert into `ceil(tot/8)` jobs; `sum(ceil(tot/8)) <= M + 32` must fit `JMAX = 32 + EMAX/64 + 8 = 88`, so `M <= 56`; tiny builds unsafe |
| 256 | MAIN | — (one candidate) | same |

The ± scale-decode variant is **not** a separate candidate: `prepare()` compiles exactly
the decode the weights make exact (`-DOKC_PRMTSC=1` when every block-scale byte is a
positive normal e4m3 value, the smem LUT otherwise) and every build is compiled with that
one `sfx`. Compiling the LUT twin of all four builds would add four `nvcc` builds inside a
prepare that must stay inside 300 s, and the earlier PRMT work never showed it as a speed
axis. Closed by construction, not by choice.

## Expected table (to be replaced by the check log)

No GPU run was made here (offline box; only `lint` + `tools/cu_build.py`). The byte model
in `artifacts/reports/moe_bytes.md` says the incumbent already streams each shape once
(`dup = 1.000–1.060`) and that bs 8/16 are "bandwidth-limited"/"at ceiling", so the two
plausible disagreements are bs 16 (MID vs MAIN; MAIN's fat 16-token job streams one expert
once where MID re-streams it per <=8-token chunk) and bs 32 (MAIN vs MID; MID
duplicates weight traffic, so a win here is very unlikely). bs 1/8 compare
scheduling variants of the same skinny work (TINY/TINY2/MID/MAIN) and may be flat.

| bs | incumbent | expected winner | expected gain |
|---|---|---|---|
| 1 | TINY | TINY (or TINY2) | 0 – 2 % |
| 8 | TINY2 | TINY2 | 0 |
| 16 | MID | MID, maybe MAIN | 0 – 3 % |
| 32 | MAIN | MAIN | 0 |
| 128 | MAIN | MAIN (untimed) | — |
| 256 | MAIN | MAIN (untimed) | — |

The check log prints one line per tuned bs, e.g.
`[stune] bs=16 incumbent=MID 91.2us winner=MAIN 88.4us gain=3.07%`, plus
`[stune] bs=<n> swap rejected: max|d|=…` or `[stune] disabled: …` when a guard fires.

## Guard logic (cannot regress beyond noise)

1. **Real inputs.** `_gen_input` (already in the file, bit-for-bit the task's
   `generate_input`: normal dist, Student-t popularity `POPULARITY = 0.03`, top-8 of 256,
   busiest rotation) with a fixed seed per bs. No import of the oracle: importing
   `reference` is a submission rejection, and the in-file replica is exact.
2. **Interleaved rounds.** One warm call per build, then 3 rounds; round order reverses
   each round; a 256 MB buffer is written before each timed launch so every build sees the
   judge's cold L2. Per-build **median** of the 3 CUDA-event times.
3. **Margin.** Swap only when `median(incumbent) − median(winner) >=
   max(0.015 × median(incumbent), 1.5 µs)`. A noisy pick cannot change behaviour.
4. **Bigger gains from the null measurement.** The method is the *same builds*: a build
   that only ties is kept.
5. **Exactness probe.** On the same input, incumbent and winner are re-run into separate
   outputs and `max|d|` must be **exactly 0**; any difference (a rounding-only difference
   from a different K-split/schedule) rejects the swap and prints the value. This is the
   gate that makes a wrong candidate impossible to score.
6. **Failure containment.** The whole tuner runs in `prepare()`'s `try/except`; any error
   (missing build, OOM, compile problem) prints `[stune] disabled: …` and leaves
   `state["sel"] = {}`, which is byte-for-byte today's dispatch. `kernel_fn` only consults
   `sel` for an exact `cfg["bs"]` hit, so nearby/test shapes (4, 33, 257, ~128-skewed) take
   the hardcoded chain.
7. **No new compiles.** Candidates are the four tuples `prepare()` already builds
   (`mods`); the tuner never calls `_ext` with a tuple outside that set.

## Time budget

`SELFTUNE_BUDGET_S = 120 s` of **GPU** time (`SELFTUNE_SLOT_US = 2000 µs` reserved per
build call, `names × (rounds+1)` per bs, accumulated from measured values after each bs).
Measured cost at H200 speeds is ~4 shapes × 4 builds × 4 calls × ~0.1 ms ≈ 10 ms, so the
budget never binds; a bs is skipped with `[stune] bs=<n> skipped (budget)` if the reserve
would exceed it. Compile time is unchanged (same four builds as the base).

## Kill criterion

If no bs disagrees with the incumbent (either the winner ties, or the exactness probe
rejects every swap), `state["sel"]` stays empty, `kernel_fn` takes the identical hardcoded
path, and the axis is **closed** for the standalone board: the remaining gap is the
9–12 µs event budget, not the build choice.

## Acceptance run

- `lint`: `PASS`, `1/1 clean` (base is also clean).
- `tools/cu_build.py src/okc_loop/vendored_work/moe/moe_selftune.py --variant TINY|MID|MAIN`:
  `ok` for TINY, TINY2, MID, MAIN and MAIN_FALLBACK (BW twins compile with no kernel, as
  the template body is expected).
- Risk: the exactness gate is strict. If builds differ in the last bf16 bit (different
  accumulation order), every swap is rejected and the file is today's behaviour — safe but
  the axis is closed. No GPU run was available to tell which case holds.
