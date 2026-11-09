# layer_build_tune — per-token-count MoE build selection in the layer's `prepare()`

Candidate: `src/okc_loop/vendored_work/layer/layer_build_tune.py`
(`# okc: problem=muse_decode_layer`, a copy of `klr_fci5.py`).

Axis: the layer's MoE is 37.7–64 % of every scored step, and its build is chosen by
four hardcoded thresholds (`MOE_TINY_MAX_M` / `MOE_TINY2_MAX_M` / `MOE_MID_MAX_M` over
tiny/tiny2/mid/b4/main) that were set outside this file and never measured on this
file's own token counts. The existing `[mfab]` self-tune only compares the SC6 twin
against its barrier twin; it does not choose among the builds.

Change: `prepare()` times each build this file already compiles against the incumbent
for T = 1, 8, 16, 32, 128, 256, stores the winner in `state["ma_sel"]`, and the
state-aware `_moe(M, state)` serves that ext for the scored T. **No** change to
`klr_fci5.py`, to the ATT7/attention path, the dense/sk2 path, the PDL predicate, or to
any build's defines.

## T → build table

| T | incumbent (`_moe`'s chain) | candidates timed | excluded and why |
|---|---|---|---|
| 1 | tiny | tiny, tiny2, mid, main, b4 | — |
| 8 | tiny2 | tiny, tiny2, mid, main, b4 | — |
| 16 | mid | mid, main, b4, tiny, tiny2 | tiny/tiny2 `__trap()` on a fat job (`tot > 8`); an expert gets at most one route per token, so they need T <= 8 |
| 32 | b4 (`MOE_DEFINES + -DOKC_B4FIN=1`) | b4, main, mid | tiny/tiny2 unsafe as above; MID's split jobs must fit `JMAX = 32 + EMAX/64 + 8 = 88` with `sum(ceil(tot/8)) <= T + 32`, so MID is safe to T = 56 |
| 128 | main | — (one candidate) | MID over the JMAX bound; B4FIN's launcher checks `M <= 32`; tiny builds unsafe |
| 256 | main | — (one candidate) | same |

The builds are exactly the tuples the file's build thread already compiles
(`_moe`, `_moe_tiny`, `_moe_tiny2`, `_moe_mid`, `_moe(32)`), so the tuner adds **no nvcc
work**; the ± SC6/PRMT suffix is the one `_ma_ext` derives from the weights, not a
candidate axis.

Expected outcome (no GPU run here — offline box; `lint` + `tools/cu_build.py` only):
the two real disagreements are T = 16 (mid vs main/b4) and T = 32 (b4 vs main); T = 1/8
compare scheduling variants of identical skinny work and are likely flat. The check log
prints, per T:
`[ltune] T=32 incumbent=b4 41.2us winner=b4 41.2us gain=0.00%` (or a swap with the
measured percentage).

## Guard logic (identical to the standalone brief)

1. **Real routing.** `_mf_gen_input` (already in the file: the task's `generate_input`
   distribution, `POPULARITY = 0.03`, top-8 of 256, busiest rotation) with a fixed seed
   per T; global ids become the layer's `lids` (`>= 32 → -1`, exactly what the router
   emits). No oracle import.
2. **Interleaved rounds.** One warm call per build, then 3 rounds with the order
   reversed each round and a 256 MB cold-L2 write before each launch; per-build medians
   from CUDA events. PDL is set to the layer's own `_pdl(T)` for every measured launch,
   since the scored call sets it too.
3. **Margin.** Swap only when `median(incumbent) − median(winner) >=
   max(1.5 % × median(incumbent), 1.5 µs)`.
4. **Exactness probe.** Incumbent and winner are re-run on the same input into separate
   outputs; `max|d| == 0` is required before `state["ma_sel"][T]` is set. A rounding-only
   difference rejects the swap and prints `max|d|=…`.
5. **Fallback.** `state["ma_sel"] = {}` on any exception (`[ltune] disabled: …`), and
   `_moe` only consults it for an exact T hit; every other T and every path keeps today's
   thresholds byte-for-byte. `st.get("ma_ok")`/`ma_w` are checked first.
6. **Safety net.** The wrapper ends with the layer's own `_check_guards()`, so the tuned
   launches get the same out-of-bounds sentinel check as the file's other MoE builds.

## Time budget

`_LT_BUDGET_US = 90 s` of measured GPU time, accounted after each T with a
`names × (rounds+1) × 4000 µs` reserve; a T that would exceed it prints
`[ltune] T=<n> skipped (budget)` and keeps the incumbent. The real cost is ~6 T × 5
builds × 4 calls × 0.05 ms ≈ a few ms, so the layer's ~140 s prepare and its 205 s `[mfab]`
cutoff are untouched. `_LT_MID_MAX_M = _MA_EMAX/64 + 8 = 56` (the JMAX bound).

## Kill criterion

If no T disagrees with the incumbent (ties, or every exactness probe rejects),
`state["ma_sel"]` stays empty and the layer behaves exactly as `klr_fci5.py`; the axis is
closed for the layer too, and the MoE bytes/event budget remains the only lever.

## Acceptance run

- `lint`: same finding multiset as `klr_fci5.py` — `30 error(s)` / 84 findings, only the
  filename and line numbers shift (verified with `diff` of both `--limit 500` lists:
  `IDENTICAL_FINDINGS`). One `globals()` guard was removed after it added a 31st error.
- `tools/cu_build.py … --variant MOE`: `ok` for MAIN(WGS), TINY, TINY2, MID; with
  `-D OKC_B4FIN=1` the same four compile (the B4FIN build's `TORCH_CHECK` is host-side,
  `M <= 32`). The `_ROUTER_FP32_CUDA`/glue/bfp failures are the pre-existing nvrtc
  false positives (no torch headers), unchanged from the base.
- Risk: if the builds are not bit-identical at a T, the strict probe rejects every swap
  and the file collapses to today's dispatch (safe, axis closed). No GPU run was
  available to measure which case holds.
