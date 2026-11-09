# bf16 mechanism scan (scout HushedAnaconda, 2026-11-02)

## Per-cell time decomposition (incumbent, job 83179200c2124bd8, profile artifacts/cas/profiles/4ab2…349631.83179200c2124bd8.txt)

Every scored cell launches **exactly one kernel** (calls = 1 for all 30). Split:

| cell | kernel µs | wall µs | kernel % | non-kernel µs |
| --- | --- | --- | --- | --- |
| qkv/o/pooler/unpooler/router m1 | 7.2/4.0/12.7/2.7/2.6 | 9.05/5.10/13.55/3.75/3.78 | 69–94% | 0.85–1.85 |
| sk2 m8–32 | 3.5–14.8 | 4.3–14.7 | ~90–99% | 0.2–1.3 |
| nvjet/qb/ol/rk m128/256 | 4.6–24.5 | 6.0–25.2 | ~78–101% | −0.4–1.8 |

(a) GEMM kernel ≈ 69–101% (avg ~88%); (b) launch/idle ≈ 0.4–1.9 µs/cell — graph-node launch
latency, irreducible while 1 kernel remains; (c) packed 12-bit decode fused inside
sk2/cgemv/specialist kernels (E12 ~0.5–1.5 µs decode penalty at m≤8,
`.vendor/muse-spark-open/muse_bf16_proj/EXPERIMENTS.md`). Tournament log: incumbent wins 29/30
cells; `ltpin` never beats incumbent; only unpooler m128 → torchT. `_LT_PINS` 6745–6757;
tournament 6830–6880.

## Unadopted mechanisms >1.2%: none found

All surveyed sources already in incumbent (Nathen v18 kernels rr_small/v18s/smallm2/dd_cap2/
qb/ol/rl/rwdl — lines 4146, 4959, 5278, 6017, 4331 ff.), measured dead (rival TMA+wgmma
m128/256 lost 0.45–0.82×; Huffman 2.2–3× slower decode; Triton split-K 0.83–1.17×), or closed.

Persistent-fusion EV = **0** (1 kernel/cell already).

## #1 on board

`arashattar` 1.6523 job `62b1572a4de843a4` (artifacts/analysis/boards.json) — no source on
disk. Best-known on disk: rohanarun #17/1.5934 = vjeux-family + Nathen `_ALT_SRC` cells +
`rq_lt` plans (`.vendor/tmp/rohan_muse_bf16_proj.py:3072,4367`); ours already contains the
same mechanisms and beats Nathen v18 1.5186.

## EV table

- Wide-cell GEMM m128/256 (qkv256 18.4 µs vs 9.8 floor, ~2.0 TB/s): EV up to +2% — measured
  dead twice (E10, rival tma_dense). OPEN, high risk.
- 12-bit format at m≥128 (bytes ×0.75, amortizes decode): EV +2% theoretical;
  decode-in-mainloop unproven. OPEN, high risk.
- Per-cell specialist swaps à la rohan `_ALT_ROUTES` (pooler16/unpooler16/router32 differ
  from our picks): EV ~+0.3–0.8%. OPEN, cheap.
- Router m128/256 toward floor (rk 4.6–5.8 vs 2.2/2.7): EV ~+0.3%. OPEN.
- Fused multi-projection / PDL / stream-K / Huffman / custom WGMMA / CUTLASS beyond
  pooler256: CLOSED.

**Verdict: no surveyed mechanism cheaply clears +1.2%.** Only >1.2% path is a new m≥128 GEMM
family — attempted and failed by two teams. Per-cell specialist swapping (~0.5%) is the cheap
open item; a per-cell diff of our 1.5434 run vs rohan's 1.5934 would pin the exact ~0.5%
recoverable cells (needs his job JSON, not on disk).
