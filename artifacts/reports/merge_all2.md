# merge_all2: + the glue packed top-8 route (PACK) on top of merge_all

File: `src/okc_loop/vendored_work/layer/klr_merged2.py` — 999,545 B (**net -7 B** against the
999,552 B base `klr_merged.py`, cap 1,000,000). Base copied, never edited in place.
The previous merge's report (`merge_all.md`) still describes the four self-gating mechanisms + FCI 5.

## 1. What was merged (from `klr_glue.py`, one mechanism)

| hook | where | names it needs |
|---|---|---|
| `_route_split(..., PACK: tl.constexpr = False, ...)` + the program-split branch (program (t,0) = norm/store, (t,1) = one `tl.topk` over the ordered-bits packing) | kernel body, before the shipped loop | `tl.program_id(1)`, `tl.topk`, `tl.gather` (Triton; compiles at first launch, inside the self-check) |
| launch: `gpk = state.get("glue_pack", {}).get(T, False)`, grid `(T, 2 if gpk else 1)`, `PACK=gpk` | `_after_attention`'s `if S == -1:` branch (the only `_route_split[` call site in the file) | `state`, `T`, `P` |
| `GLUE_PACK = None` + `_glue_pack_selfcheck(st)` | appended at EOF | `st["bias"]`, `st["g_preexp"]`, `TUNED_T`, `_pdl`, `torch` |
| the self-check call | the `_policy` prepare, immediately before `st['compose_selection'] = []` | `st` |

Chaining: nothing is clobbered. `_route_split`'s extra keyword defaults to `False`, so the non-PACK path
(including the graft chain's own `_graft_route(PACKED=...)`, which is a different kernel) is byte-identical
in behaviour; `_after_attention` is untouched apart from the `gpk` lookup; the self-check hangs off the
existing `_policy_parent_prepare` chain, which keeps its 15 other overrides.

## 2. The two specific checks

**(a) Is the PACK path reachable from the launch site?**
`grep -n '_route_split\[' klr_merged2.py` returns exactly one call, and it is the one that carries `gpk`:

```
11929:    if S == -1:  # pooler and router came from the bf16_proj kernels as two separate tensors
11930:        pooled, logits = P
11931:        gpk = state.get("glue_pack", {}).get(T, False)
11932:        _route_split[(T, 2 if gpk else 1)](pooled, logits, state["bias"], state["g_preexp"], ids, wts, xm,
11933:                                           data["local_start"], lids, PERMUTED=not (use_marlin or use_ma),
11934:                                           PACK=gpk, num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
```

`S == -1` is what `_pool_router_fp32` returns for the split form (`bfpf`, `T <= 8`, and the merged rq_pool
path) — i.e. exactly where the two tensors exist. The kernel's `if PACK:` branch returns before the shipped
loop, so the two paths cannot interleave.

**(b) Does the self-check still run before the compose sweep?**
Yes — same function scope, and the sweep reads `st` afterwards:

```
13031:    st = _policy_parent_prepare(weights)
13035:    try:      # before the sweep, so the sweep's base variant is the route the run will use
13036:        st["glue_pack"] = {} if GLUE_PACK is False else _glue_pack_selfcheck(st)
13038:        st["glue_pack"] = {}                     # (self-check raised: every T stays on the shipped kernel)
13040:    print("glue: pack", st["glue_pack"], flush=True)
13041:    st['compose_selection'] = []
13061:        ts = _compose_times([lambda ss=ss: _after_attention(data, ss, o_in, None, T) for ss in variants], reps=32)
```

Each sweep variant is `dict(st)` (`_policy_variant`), so every timed chain carries the measured
`glue_pack`; the sweep cannot time a route the run will not use. Had the self-check been placed after the
loop, the sweep would have measured the unpacked route and the pick would be wrong for the run — that is the
ordering bug the check above rules out.

## 3. Binding audit (all 14 `def prepare(`)

`symtable`: every loaded global in every prepare block (and its nested closures) resolved against the
module namespace + builtins → **0 unresolved**.

```
prepare line 11388 span 11388..11480  weights[19 keys: the checkpoint contract]
prepare line 12846 span 12846..12872  weights[next_pre_attn_norm, post_ffn_norm, pre_ffn_norm]
prepare line 12991 span 12991..13012  weights[]
prepare line 13030 span 13030..13067  weights[]        <- the _policy prepare (self-check + compose sweep)
prepare line 13460 span 13460..13463  weights[]
prepare line 13700 span 13700..13702  weights[]
prepare line 13854 span 13854..13858  weights[]
prepare line 13945 span 13945..13948  weights[]
prepare line 14489 span 14489..14491  weights[]
prepare line 14599 span 14599..14601  weights[]
prepare line 14608 span 14608..14613  weights[]
prepare line 16475 span 16475..16503  weights[pooler, qkv_gate, unpooler]     <- chen rq bake
prepare line 16569 span 16569..16583  weights[qkv_gate]                       <- rq_pool gate
prepare line 16698 span 16698..16705  weights[]                               <- attnst tune
```

Every `weights[...]` key anywhere in the file is in the root block's set (the `generate_weights` contract);
no new key was introduced by this merge (the glue hooks read `st` only).

## 4. Gates

| gate | result |
|---|---|
| `ast.parse` | OK |
| `okc_loop.cli lint` (vs `klr_merged.py`) | 28 errors vs 28 — identical set, delta **0** (only the file name differs in the output); Tier-0 is not clean in the base either |
| `tools/cu_build.py --variant MID` | 9/9 builds FAIL on the base *and* on this file with the same first diagnostics (`cannot open source file "nv/target"`, `cudaGetLastError is undefined`: no CUDA toolkit headers on this box) — delta **0**, the gate cannot discriminate here |
| size | 999,545 B < 1,000,000 (net -7 B) |
| Colab sm_90a gate (`colab run --gpu A100 --timeout 900 artifacts/bench/gate_m2.py`) | **OK**: A100-SXM4-40GB, torch 2.11.0+cu130, wall 87.3 s. `_att7`, `_moe`, `_moe_tiny`, `_moe_tiny2`, `_moe_mid`, `bfp__get_ext`, `_moe(32)` all ok; `bad` = only `_graft_glue_module`, `_graft_b2_module`, `_sl_cluster_module` (`No module named 'tvm_ffi'`, expected absent on Colab). |

## 5. Byte budget: what had to go (net -7 B)

The PACK change itself is ~4.6 KB (kernel branch + launch + self-check). It was paid for entirely with
comments, no code: the stack lineage in the header (kept as 3 lines, every attribution line intact), the
`_plan7` per-branch timing notes, `_TUNED7`/`bfp_GEMM_ALT` measurement notes, the opus_glue design
derivations (pointed at `artifacts/reports/gle_t8.md`), the graft-block banner (attribution kept), and the
banners/docstrings of my own four mechanisms. The self-check is untouched, as are every kernel, every gate
constant and every measured number that steers a pick.
