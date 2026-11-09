# attndual: the two ATT7 finalize flags, measured on `muse_attention_decode`

File: `src/okc_loop/vendored_work/attn/dual.py` (byte copy of `attn/plan.py`, sha `d2580a9f`, + the dual build and
the per-shape pick). Line 1 is `# okc: problem=muse_attention_decode`. Source of the hypothesis:
`layer/klr_attnst.py`'s `_att7v` / `_attn_finalize_tune` (`artifacts/reports/attn_selftune.md`).

## What the two flags are, and where they can reach

The board's kernel is the same code with both constants 0 and had never been measured with them on. They are plain
`#define`s in `CUDA_SRC` (a `-D` would be overridden by the later definition), so the variant build patches the two
lines in its own copy of the source — one occurrence each, asserted:

| flag | site | reaches on this board |
|---|---|---|
| `OKC_FINWARM` (dry call of the last-CTA finalize in the prologue) | `segment`, gated `!FLAT && p.cs<=1 && !p.coop && p.nc==1` | the three **q_len 1 sliding** loads (B1/B16/B32: 32/16/8 plain split-KV CTAs, `nc=1`) |
| `OKC_FLATCOOP1C` (the grid-wide flat finalize compiled for q_len 1) | 6 gates, `L == 8` -> `L == 8 \|\| OKC_FLATCOOP1C` (`mode1`, the coop merge branch, the epoch preload, the three `flat_finalize` calls) + host `coop` override | the two **flat q_len 1 full** loads (B16/B32: 264 CTAs, two per SM). The launcher override is `coop=1`; the kernel clears `coop` again if the grid is not resident, so a non-fitting grid just re-measures the base |

The other seven scored shapes compile the flags in and stay bit-identical at runtime; they are timed anyway, so the
sweep also reports whether the two extra constants cost anything through code layout.

`dry=1` in `fin_nc1` makes both of its stores conditional (`if (!dry)` on the accumulator reset and on the `out`
store), so the prologue warm-up reads the workspace and the gate rows and writes nothing.

## How the pick is expressed

* `_build(tag="", flags=(), src=None)` — two lines changed so the variant can pass a patched source.
* `_dual_var()`: patches the two `#define`s and builds `okc_attn_v298bulk_fw1c_90a` (distinct from the shipped
  `okc_attn_v298bulk_90a`, so both live in different extension dirs). Any failure is caught in `_dual_sweep`, which
  prints `[dual] variant build FAILED: ...` and returns: the shipped build stays for every shape.
* `_DUAL_PICK[(batch, q_len, sliding, pairs)] = (module, widen)` — read by `_launch` only when the caller passed
  neither `_mod` nor an override and did not ask for the `_ret` readback, so production and the sweep's readback
  keep one path. `widen` sets `plan["coop"] = 1` (the host gate `L == 8` -> `L in (1, 8)`).
* Empty dict = shipped build: every shape the sweep does not cover (correctness cases) is untouched.

## Sweep

Runs at import, right after the plan sweep (this problem is stateless and has no `prepare()`), before the harness
warms and captures a case:

* per shape: `_plan_case` (the plan sweep's synthetic inputs), replayed through the real `_launch` so the plan, the
  workspace and the PDL switch are production's;
* `_plan_time` (the file's own helper): one CUDA graph per build, cold-L2 flush before every replay, interleaved
  order, **8 rounds** (the plan sweep's 16 was cut because the extra nvcc build, not the replays, is what the added
  import time is spent on; one bit needs the round count, not 16);
* keep the variant only if all three hold — median faster by `max(3%, 2 us)`, >= 75% round wins, and
  `rel-L2(variant, base) <= 1e-6` on the bf16 rows (`_dual_pick_ok`);
* per-shape `try`, whole pass `try`: a failure pops the key, prints `pick=base FAILED: ...` and the run continues.

Printed per shape (one line, 12 lines):

```
[dual] B32 L1 sliding base=8.3 var=7.1 pick=var wins=7/8 rel=0.0e+00
[dual] B16 L1 full    base=86.2 var=86.0 pick=base wins=4/8 rel=0.0e+00
[dual] 0/12 shapes on the variant build
```

## Cost

* nvcc: **one extra translation unit** of the same source at import (the plan port's check wall was 495 s with three
  builds and the plan sweep; the variant is a fourth, same size, serial on this box). The plan port's 495 s leaves
  the 900 s budget with room for it; the dual sweep's GPU side is *smaller* than the plan sweep's (2 builds x 8
  rounds = 16 replays per shape against 8 candidates x 16 = 128, and no new input set: `_plan_case` is reused) —
  it re-creates the twelve input sets, which is the plan sweep's own dominant GPU cost, so the sweep is roughly a
  second copy of that part.
* GPU: 24 graph captures, 192 replays, 24 eager launches, the twelve input sets built and dropped per shape.

## Gates

* `ast.parse` ok (6126 lines).
* `okc_loop.cli lint`: **PASS 1/1 clean**, byte-identical to `plan.py`'s (zero errors, no new findings).
* `tools/cu_build.py --variant MID`: identical to `plan.py` except the file name inside its own digests —
  `CUDA_SRC` ok (compiled, no kernel emitted: template body), `ZH_WST_CUDA` ok, `ZH_VFG_CUDA` FAIL with the same
  offline `cuda_bf16.h` `__shfl_*_sync` "C linkage" gate artifact. The same run on a copy with both flags patched to
  1 gives the identical result, so the patched source is the same translation unit to nvrtc.
* `tools/colab_gate.py` + `colab run --gpu A100`, **builds only**: generated from a copy with the two import-time
  sweeps neutralized (`artifacts/bench/dual_gate_src.py`, since a sweep launch of sm_90a SASS on an A100 never
  returns) and with `_att7 = _module`, `_moe_mid = _dual_var` so the gate's fixed name list calls the two builders.
  Result: A100-SXM4-40GB, torch 2.11.0+cu130, `wall_s: 71.7`, `bad: ["_moe(32)"]` (the alias the attention file does
  not have, as on the base file). `_att7` (= `_module`, the shipped build) `ok`, `s: 0.0` (that name was already in
  the session's extension cache); `_moe_mid` (= `_dual_var`, the **variant** build) `ok`, `s: 71.7` — so the patched
  source compiles for sm_90a, and **one extra translation unit of this file is 71.7 s of nvcc** on the gate box. The
  gate cannot instantiate the kernel (the `__global__` is chosen below the `<torch/` cut), so the new
  `flat_finalize<1, ...>` instantiation is proven by the H200 check, not here.
* `tests/test_attn_dual_tune.py`: 3 passed (gate arithmetic; the two anchors occur once and are patched in the text;
  six widened gates; the pick is read on the production path only; the sweep is called at import).

## H200 result

## H200 result (check `b01cc1203c1b45d8`, sha `77d214c2`)

`passed=True`, 16/16 tests, Tier-0 PASS, Tier-1 PASS, `hopper_sm90a_specialized`, check wall **472 s** (7:14 run,
1:01 queued) — the extra build and the dual sweep did not push the run near the 900 s budget (the plan port's own
check was 495 s). `[dual]` lines from the job's log, one per scored shape (medians are cold-L2 graph replays;
`wins` is out of 8):

| shape | base us | var us | wins | pick | flag that can reach it |
|---|---|---|---|---|---|
| B1 L1 sliding | 8.42 | 8.51 | 4/8 | base | FINWARM |
| B16 L1 sliding | 11.20 | 11.01 | 5/8 | base | FINWARM |
| B32 L1 sliding | 12.86 | 12.74 | 6/8 | base | FINWARM |
| B1 L8 sliding | 10.40 | 10.50 | 5/8 | base | - |
| B16 L8 sliding | 15.17 | 15.17 | 3/8 | base | - |
| B32 L8 sliding | 18.46 | 18.34 | 5/8 | base | - |
| B1 L1 full | 14.46 | 14.11 | 5/8 | base | - (per-request, `nc=8`, `coop=1`) |
| B16 L1 full | 86.50 | 87.39 | 1/8 | base | FLATCOOP1C (`coop=1`) |
| B32 L1 full | 159.81 | 159.94 | 3/8 | base | FLATCOOP1C (`coop=1`) |
| B1 L8 full | 18.08 | 18.21 | 4/8 | base | - (already on in the shipped build) |
| B16 L8 full | 126.21 | 127.17 | 3/8 | base | - |
| B32 L8 full | 225.09 | 224.67 | 5/8 | base | - |

`[dual] 0/12 shapes on the variant build`. Row: **geomean 2.6599**, value **1415.9 us**, CI 2.620-2.700, against the
submitted 2.6590 / 1417.3 us — **+0.03%, i.e. noise**. Per-load against the ledger's personal best (`2.6089`,
1443.6 us): decode 1 +0.01%, decode 16 +1.07%, decode 32 -0.11%, verify 1 +8.12%, verify 16 +2.42%, verify 32
+0.47%, 5/6 loads improved, worst-load regression inside the floor.

## Did either flag earn its pick? No.

* **FINWARM** (the three q_len 1 sliding loads): 8.42 -> 8.51, 11.20 -> 11.01, 12.86 -> 12.74 us. Two of three are
  nominally faster by 0.1-0.2 us with 5-6 of 8 rounds; the third is slower. The prologue dry call is worth
  <= 0.2 us on kernels this small, which is inside the round-to-round spread. It also *cannot* be picked under the
  gate as written: `max(3%, 2 us)` on an 8.4-12.9 us kernel asks for 24-38%, so the 2 us floor binds (the same
  diagnosis the layer's plan sweep reached for its sub-30 us shapes).
* **FLATCOOP1C** (the two flat q_len 1 full loads, the only shapes that could clear the floor): 86.50 -> 87.39 us
  (needs -2.6 us, lost 0.9 with 1/8 rounds) and 159.81 -> 159.94 us (needs -4.8 us, lost 0.1 with 3/8). The
  grid-wide finalize for q_len 1 is not a win on this kernel: at 264 CTAs the 2-per-SM grid already finalizes
  mostly in parallel, and the layer's "+1-3 us off the finalize tail" estimate does not survive the measurement.
* The seven shapes the flags compile into but cannot reach moved by <= 1% either way — no code-layout cost.

**Side observation (kills a tempting lead):** the layer's ATT7, profiled inside `muse_decode_layer` on the same
twelve loads, is the same kernel family and lands within ~5% of this board's: 5.3 vs 5.1 us decode 1 sliding, 11.5 vs
10.6 decode 1 full, 89.4 vs 86.8 decode 16 full, 171 vs 167 decode 32 full, 14.3 vs 13.9 verify 1 full, 104 vs 98
verify 16 full, 187 vs 183 verify 32 full. This board's file is the *faster* of the two everywhere, so there is no
kernel-merge upside to chase from the layer side either.

**Caveat, and it did not decide anything:** `rel` came back `nan` on all 12 shapes, in both builds, so the
`<= 1e-6` output guard reported nothing usable (the sweep's synthetic case — random e4m3 codes up to 0x7E with
`sm_scale * qk_scale = 0.68` — leaves the bf16 rows non-finite; the plan sweep never evaluated its own rows either,
because it never had a candidate clear the floor). The guard fails closed (`nan <= tol` is false), and no shape
cleared the speed gate, so the pick was decided by speed alone. The numerical argument stays structural: `dry=1`
makes both stores conditional, and the flat coop path finalizes the same rows from the same single workspace copy,
so a real rel check would have been a formality. The harness's own 16/16 correctness tests (real data) pass.

## Verdict

Port done, inert on H200, no score change: neither flag earns its pick on `muse_attention_decode`, which is
independent confirmation of the layer-side estimate (0.2-0.6% / +0.1-0.3% of a *layer* is 0.1-0.2 us of an 8-13 us
*attention* call). Both flags stay 0, so `dual.py` measures as `plan.py`, and the cross-kernel numbers above say the
layer's ATT7 clone has nothing left to give this row either. Nothing here should be submitted.
