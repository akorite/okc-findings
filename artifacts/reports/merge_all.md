# merge_all: the four self-gating mechanisms + FCI 5 in one layer candidate

File: `artifacts/reports/klr_merged_all.py` (999,552 B < 1,000,000 cap).
Base: `src/okc_loop/vendored_work/layer/klr_drate.py` (991,109 B), copied, never edited in place.

## 1. Hook map (what each append touches, and the names it needs)

| # | mechanism | hooks | state it reads / writes | gate |
|---|---|---|---|---|
| 1 | `klr_drate.py` (base) | `bfp_sk2_prepare` → `bfp__sk2_bake`; `bfp_GEMM_ALT` + `bfp__gemm_items` + `_bfp_config` dispatch; `bfp_SK2_ALT`; fused `pr` in `bfp__run_gemm` / `_pool_router_fp32` | `bfp_SK2_PICK[(layer, tokr)]`, `state["sk2"]`, `state["choice"]` | 1e-3 parity + ≤0.97x and ≥0.3 us on 24 cold-L2 graph rounds |
| 2 | `klr_plan.py` | `_PLAN7_PICK` (new table); `_launch7` gains `_ret=` + `raw_n`/`pick`/regime guard; `_decode_attention7(**_ov)`; `_plan_sweep7`/`_plan_cands7`/`_plan_case7`; call in the root `prepare` after `_attn_selftest` | `st` only (synthetic inputs); writes `_PLAN7_PICK[(B, L, sliding, pairs)]` | max(3%, 2 us) median + ≥75% round wins + 1e-3 rel-L2 |
| 3 | `klr_attnst.py` | `_att7_build`/`_att7v`/`_att7v_opt` (second ATT7 build with `OKC_FINWARM 1` + `OKC_FLATCOOP1C 1`); `_ATT7_PICK` + `_launch7` wrapper (`_launch7_attnst` = the outer name); `_attv` added to the nvcc fan-out; `_att_shape_inputs`/`_att_pick_ok`/`_attn_finalize_tune` + final `prepare` | `_ATT7V`, `_ATT7_PICK[(B, L, window)] = (module, {"coop": 1} on flat q_len 1)` | max(3%, 2 us) + ≥75% wins + 1e-6 rel-L2 |
| 4 | `klr_pool.py` | `_pool_router_fp32` rq_pool branch; `RQLAYERS += "pooler"`, `_RQ_NAMES["pooler"]="pool"`, `RQ_USE += {("pool",128),("pool",256)}`; `RQPOOL_*`, `_rqpool_T`, `BND2_REPS`/`_bnd2_rel`/`_selftune_time`, final `prepare` | `st["chen_rq"][("pool", T)]`, `st["rq_pool"][T]` | ids exact + 1e-5 rel-L2 on out and s2, ≥max(4%, 3 us), 24+/32 round wins |
| 5 | `klr_fci5.py` | `#define OKC_WGSFCI 4` → `5` (line 5635) | — | none (plain constant, coin flip) |

Chaining, never clobbering: the pre-existing `_stack_unpinned_prepare` / `_policy_parent_prepare` /
`_router_fp32*_parent_prepare` / `_pin_parent_prepare` / `_rq_prior_prepare` chain is untouched; the two new
overrides are appended as `_rqpool_prior_prepare` and `_att_prior_prepare` (rq_pool outer of the chen bake, attnst
outermost), so the execution order is root → … → chen rq bake → rq_pool → attnst. `_launch7` is wrapped, not
replaced: `_launch7_attnst` keeps the base (with the plan hooks) and the wrapper only injects `_mod`.

De-duplication done to fit the byte cap (each pair is the same code twice otherwise):
`_plan_case7` reuses `_att_shape_inputs`; `_plan_time7` → `_selftune_time(...)[0]`; the pool gate's round-win
count is computed from `_selftune_time`'s samples (2 variants: the round winner is the faster one), so `_bnd2_ab`
is gone; `_bnd2_rel` + `BND2_REPS` remain.

## 2. Binding audit (the gate that matters) — 0 unresolved names, 1 weights key fixed

Method: `symtable` over the merged file. For each top-level `def prepare` (14 of them) every symbol that is
`is_global()` and loaded (including from nested closures under the block) is checked against the module-level
namespace ∪ builtins.

```
def prepare( blocks: 14
  [ 0] line 11367  params=['weights']  bound: weights, st, torch, dev    ok
  [ 1] line 12838  params=['weights']  bound: weights, st, torch         ok
  [ 2] line 12983  params=['weights']  bound: weights, st                ok
  [ 3] line 13022  params=['weights']  bound: weights, st, torch, gen    ok
  [ 4] line 13446  params=['weights']  bound: weights, st                ok
  [ 5] line 13686  params=['weights']  bound: weights                    ok   (no st, no weights[...] use)
  [ 6] line 13840  params=['weights']  bound: weights, st                ok
  [ 7] line 13931  params=['weights']  bound: weights, state             ok   (indexes `state`, which it binds)
  [ 8] line 14479  params=['weights']  bound: weights                    ok
  [ 9] line 14590  params=['weights']  bound: weights                    ok
  [10] line 14599  params=['weights']  bound: weights, st                ok
  [11] line 16466  params=['weights']  bound: weights, st, torch         ok   (chen rq bake)
  [12] line 16561  params=['weights']  bound: weights, st, torch, gen    ok   (rq_pool, appended)
  [13] line 16694  params=['weights']  bound: weights, st                ok   (attnst tune, appended)
unresolved names: 0
```

`weights[...]` keys per block (the KeyError class that killed two checks today):

```
line 11367 (root): 19 keys: pooled/qkv_gate/o/router/unpooler/w13*/w2*/norms/router_bias   <- the checkpoint contract
line 12838: next_pre_attn_norm, post_ffn_norm, pre_ffn_norm
line 16466: pooler, qkv_gate, unpooler
line 16561: qkv_gate      <- was `weights["qkv"]` in klr_pool.py: KeyError at prepare. FIXED here.
others: none
```

Every key now used anywhere in the file is in the root block's set (the reference `generate_weights` contract;
the vendored upstream layer never has a `qkv` key — it is `qkv_gate`, and `st["qkv"]` is the state alias).
**`klr_pool.py` line 16747 still carries `weights["qkv"].device` and would die the same way.**

## 3. Interaction risk between the four mechanisms

* Both sweeps build their own synthetic inputs and time with cold-L2 graph replays; all four run on the default
  stream, one thread, `torch.cuda.synchronize()` between rounds, so they serialize: no race, only wall clock.
* `prepare` order means the plan sweep runs *before* `_attn_finalize_tune`; each tune measures a base that already
  contains the earlier picks (the plan pick is inside `_launch7`, the attnst pick is a wrapper) — so no pick is
  measured against a stale incumbent.
* One deliberate precedence: where the attnst pick wins on a flat q_len-1 shape it passes `{"coop": 1}` as an
  override, and the base `_launch7` skips `_PLAN7_PICK`/`_TUNED7` when an override is present (its own regime
  guard). So on those two shapes the attention build pick wins over the plan pick, and the attnst gate was measured
  against the plan-picked base — conservative, not unsafe.
* `st["rq_pool"]` is consumed only by `_pool_router_fp32` at `T > 32`; the sk2 bake, the plan sweep and the ATT7
  tune never read it.
* Row cost of the mechanisms is additive only where they touch disjoint shapes: drate = dense GEMMs, plan = the
  attention launch, attnst = the attention finalize build, pool = the pooler+router cell.

### Prepare budget (299 s)

| added | seconds |
|---|---|
| drate sk2/GEMM bake (5 alt cells x ~2 kernels x 24 rounds, 64 MB L2 flush each) | 0.05-0.2 |
| plan sweep (12 shapes x up to 8 candidates x 16 rounds + graph capture + k/v caches) | 1-2 |
| ATT7 finalize tune (12 shapes x 2 builds x 16 rounds) + the extra nvcc slot in the fan-out | 0.3-0.8 (+0 wall, parallel) |
| rq_pool gate (2 T x 2 chains + 32 rounds x 2) | 0.1-1 |
| pooler pack in the chen bake (4th layer, 5+2 probes x 2 m) | 0.5-1.5 |
| **total added** | **~2-5.5 s** |

Well under the ~240 s bar, so nothing has to be dropped. They do contend for the same GPU minutes but only for
seconds.

Freed instead: the `MFAB_LAYER` MoE A/B diagnostic (`_mf_ab` + `_mf_start_base_builds`, 7,905 B of source, 3 extra
nvcc builds on a thread, `th.join(timeout=195 - elapsed)` = up to 195 s of prepare for print-only output) is
removed. That is both the byte budget (the four mechanisms needed ~22 KB and the file had 8.9 KB of headroom) and
the largest single consumer of the same prepare window. Everything else kept bytewise: the 17 pure `# ----`
divider lines (1,573 B) and the pw11-pw31 `_make_chen` history (now the same pointer to
`bf16_mechanism_scan.md` that `klr_pool.py` already uses).

## 4. Gates

| gate | result |
|---|---|
| `ast.parse` | OK |
| `okc_loop.cli lint` | FAIL 28 errors vs the base's 30 (delta **-2**, the removed diagnostic's `globals`); identical error set otherwise, no new line/category. Tier-0 is not clean in the base either. |
| `tools/cu_build.py --variant MID` | **degenerate here**: 9/9 builds fail on base *and* merged with the same first diagnostics (`cannot open source file "nv/target"`, `cudaGetLastError is undefined` — no CUDA toolkit headers on this box). Delta 0, so this gate cannot discriminate. |
| size | 999,552 B < 1,000,000 |
| Colab sm_90a compile gate (`colab run --gpu A100 --timeout 900 artifacts/bench/gate_merged.py`) | **OK**: A100-SXM4-40GB, torch 2.11.0+cu130, wall 111.8 s. `_att7` ok, `_att7v` ok (the text-patched second build), `_moe` ok, `_moe_tiny` ok, `_moe_tiny2` ok, `_moe_mid` ok, `bfp__get_ext` ok, `_moe(32)` ok; `bad` = only `_graft_glue_module`, `_graft_b2_module`, `_sl_cluster_module` (`No module named 'tvm_ffi'`, expected absent on Colab). `--timeout 900` is required: the default 30 s expires before the fan-out. |

Colab: `python tools/colab_gate.py artifacts/reports/klr_merged_all.py --out artifacts/bench/gate_merged.py`,
with `_att7v` added to the generated gate's build list (the text-patched second ATT7 build is the one new
compile that could fail; it is a compile gate, so it belongs in the list).
