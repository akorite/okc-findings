# attnplan: the layer's prepare-time plan sweep, ported to `muse_attention_decode`

File: `src/okc_loop/vendored_work/attn/plan.py` (byte copy of `attn/klr_attention.py`, sha `55fd129a`, + the
sweep). Base row `2.6297 / 1432.2 us` (that file checked `2.6497`); layer candidate `layer/klr_plan.py`
(`_PLAN7_PICK`/`_plan_cand7s`/`_plan_time7`/`_plan_sweep7`) is the source of the port.

## Hook diff: what ports verbatim, what had to change

| hook | layer (`klr_plan.py`) | attention (`plan.py`) |
|---|---|---|
| where it runs | `prepare()` (untimed, has `state`, 300 s budget) | **import**, right after `_zh_start_builds()`: this problem has *no* `prepare()` and no `state` (harness: `STATELESS`), and import is the only hook that is certainly outside capture and outside the timed section |
| clock | `_rq_time.time()` | dropped — `time` is not on the judge's import allowlist and `time.time` is a Tier-0 error |
| inputs | `_plan_case7` synthetic, ctx 80000 / window, page 1 | **verbatim**, minus `st`: `st["qkv"].shape[0]` → the real fused width `2*(hq+hkv)*HEAD_DIM`; qkv built 3-D `[B, L, *]` and `seq_lens` int32 / `cache_loc` int64 / caches e4m3 `(B*per, 1, 2, 64)` to match `generate_input` exactly |
| head layout | constants `NUM_Q_HEADS/NUM_KV_HEADS` | `_PLAN_HQ, _PLAN_HKV = 16, 2` (tp8 = one pair, `pairs=1`); `qkv` row = 2304 |
| launch readback | `_decode_attention7(..., _ret=True)` → `(raw, n, nc, coop, flat)` | same `_ret` hook added to `_launch` (4 lines, returns before the workspace alloc) — not duplicated host logic |
| timing | `_plan_time7`: 16 interleaved cold-L2 graph replays, 256 MB flush | **verbatim** |
| candidates | `_plan_cands7`: shipped, coop flip, 2 `nc`, `n_splits` x0.5/0.75/1.25/1.5, SM-multiple CTAs, `out[:8]` | **verbatim** (`_plan_cands`) |
| threshold | `max(3%, 2 us)` median, `>=75%` wins, rel-L2 `<=1e-3` on the bf16 rows | **verbatim** |
| apply | `_launch7`: `_PLAN7_PICK[(B,L,bool(window),pairs)]`, applied only when the *pre-override* `n_splits` matches | **verbatim** (`raw_n`, `pick`, tuned-then-pick order) |
| failure | one `try` around the whole sweep in `prepare` | `try` around the call **and** per shape (one bad shape keeps its shipped plan; the process still imports) |

Only two behavioural differences: the hook (import vs prepare) and per-shape exception containment. The scored
shapes' shipped plans were re-derived from the file's own `_plan` + `_TUNED` + coop default at 132 SMs and are
quota-capped (`raw` = 32/16/8/33/8 sliding, 264/252 full), so the regime guard resolves identically for ctx 80000
and for the agentic-spread width 86001 — verified by exec'ing `_plan`/`_plan_cands` against stubs.

## H200 result (check `6e04b0e305734e62`, sha `bae63690ec14`)

`[plan]` lines from the job's `log_tail`, one per scored shape, `n_splits/nc/coop`:

| shape | shipped | best | med us (candidates) | wins | pick |
|---|---|---|---|---|---|
| B1 L1 sliding | 32/1/0 | 32/1/0 | 8.45 8.83 9.25 8.83 9.60 9.63 | 0/16 | shipped |
| B16 L1 sliding | 16/1/0 | 16/1/0 | 10.88 11.42 13.79 11.10 11.71 10.94 11.07 11.04 | 0/16 | shipped |
| B32 L1 sliding | 8/1/0 | 8/1/0 | 12.64 13.18 15.84 12.99 14.34 13.76 13.34 12.90 | 0/16 | shipped |
| B1 L8 sliding | 33/1/1 | 33/1/1 | 10.30 11.81 10.37 10.43 12.54 11.20 | 0/16 | shipped |
| B16 L8 sliding | 8/1/1 | 8/2/1 | 14.82 15.68 15.55 **14.34** 17.34 15.23 16.32 15.30 | 13/16 | shipped (fails the 2 us floor: gain 0.48 us) |
| B32 L8 sliding | 4/1/1 | 4/1/1 | 18.11 19.36 19.39 18.46 25.70 21.02 20.22 19.07 | 0/16 | shipped |
| B1 L1 full | 252/8/1 | 252/4/1 | 13.95 14.98 15.10 13.92 15.65 15.17 14.82 14.78 | 10/16 | shipped |
| B16 L1 full | 264/1/0 | 264/2/0 | 86.24 86.18 87.01 86.02 93.63 91.97 116.16 101.38 | 9/16 | shipped |
| B32 L1 full | 264/1/0 | 264/1/1 | 159.62 159.33 160.48 159.81 175.65 169.70 215.90 185.28 | 9/16 | shipped |
| B1 L8 full | 252/8/1 | 252/8/1 | 17.73 27.33 19.90 18.40 19.42 19.81 32.16 32.42 | 0/16 | shipped |
| B16 L8 full | 264/1/1 | 264/1/1 | 126.34 128.80 127.52 126.91 148.67 155.81 156.26 139.33 | 0/16 | shipped |
| B32 L8 full | 264/1/1 | 264/1/1 | 226.08 227.20 230.50 226.30 263.81 282.88 273.50 238.98 | 0/16 | shipped |

**0/12 shapes re-planned.** The shipped plan is the best of its own neighbourhood on every shape; the one
candidate that beat the 3% bar (B16 L8 sliding, `nc=2`, 13/16 wins) is a 0.48 us gain, 4x short of the 2 us
absolute floor. Every shape's `_TUNED`/`_plan` entry survives, so this file measures as the vendored one.

## Check numbers (1 of 2 spent, no submit)

`passed=True`, Tier-0 PASS, Tier-1 PASS, `hopper_sm90a_specialized`, 40 records clean, check wall 495 s.

* geomean **2.6574**, value **1417.3 us**, CI `[2.6175, 2.6973]`; board row 2.6297 / 1432.2 us → **+1.05%**
  (2.6497 for the same file's earlier check → +0.3%). No plan changed, so this is check-level variance, not a win.
* per-load vs the ledger's personal best (`2.6089 / 1443.6 us`, sha `4cfc5cca`): decode 1 +3.12%, decode 16
  +0.34%, decode 32 +0.35%, verify 1 +5.72%, verify 16 +1.81%, verify 32 -0.07% — 5/6 loads up, worst-load
  regression well inside the floor.
* `auto_submit` would still refuse on `nearby_shapes` (no H200 nearby-shape verification for this sha), so a submit
  would need that verification first; not spent here.

## Gates

* `ast.parse` ok; `okc_loop.cli lint` **PASS 1/1 clean** (base PASS; zero errors, no new findings — the two
  `time.time` hits and the `time` import were the only deltas and were removed).
* `tools/cu_build.py --variant MID`: identical to base — `CUDA_SRC` ok (template body, no kernel emitted),
  `ZH_WST_CUDA` ok, `ZH_VFG_CUDA` FAIL with the same offline `cuda_bf16.h` `__shfl_*_sync` "C linkage" errors as
  base, i.e. a gate-environment artifact, not this change.
* `tools/colab_gate.py` + `colab run --gpu A100` (sm_90a forced), per build: all three extensions exist in the
  session — `/root/.cache/torch_extensions/py313_cu130/{okc_attn_v298bulk_90a, okc_zh_wst_efb8c04999,
  okc_zh_vfg_55f1ac74b0}` each hold their `.so` — so `CUDA_SRC`, `ZH_WST_CUDA` and `ZH_VFG_CUDA` all compile for
  sm_90a from this file. The gate's JSON never came back: with `get_device_capability` forced to (9, 0) the script
  then *runs* the import-time sweep on the A100, and the first real launch of sm_90a SASS there never returns (that
  session sat 75 min inside the cell and was killed). The same gate on the unmodified file (`gate_attnbase.py`)
  returns its JSON normally — `===BEGIN=== {"bad": ["_moe(32)"], "builds": {"_att7": absent, ...}} ===END===`, i.e.
  every name this gate knows (`_att7`, `_moe`, ...) is `absent` for an attention file, so the gate cannot report
  ATT7 here at all. The real compile proof is the H200 check: it built and ran all three extensions and passed
  Tier-1. Caveat, not a regression: the kernel is Hopper-only (`hopper_sm90a_specialized`), the judge never runs it
  on a non-Hopper, and any launch of this kernel on the gate's forced-capability A100 would be the same hang.

## What the med table teaches

The shipped plan is the minimum of its own neighbourhood on 8/12 shapes outright and within 0.5 us on the other
four. The `2 us` absolute floor is what binds on every sub-30 us shape (it needs a 24% win at 8.45 us); the 3%
relative floor is what binds on the 86-226 us shapes, where the candidate spread (132-396 CTAs) is already flat.
The one candidate with a repeatable edge — B16 L8 sliding with `nc=2`, 13/16 round wins, 14.34 vs 14.82 us — is
0.48 us, 4x short. Two corollaries for the layer's own sweep: its 2 us floor blocks the same size class, and its
`_plan_cands7` never tries the flat/per-request kernel-mode flip (n_splits is only ever rescaled inside the
shipped mode), which is the one axis still unmeasured here; on this kernel the flat plan wins that mode anyway
(the flat shapes' `n_splits` sweep 132-396 never beat 264).

## Verdict

Port done, inert on H200, no score change. The plan axis is exhausted for this kernel: what is left for
`muse_attention_decode` is in the kernel, not the launch. For the layer the port is *negative* evidence —
the sweep costs nothing and adopts nothing on the same 12 shapes at the same thresholds, so `klr_plan.py`'s
2.24934 is not coming from its plan picks. Nothing here should be submitted.
