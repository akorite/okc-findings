# deepcopy + fresh-thread CUDA context audit — `src/okc_loop/vendored/keell0renz/muse_decode_layer.py`

Verdict: **both traps are already correct in the board file. No fix, no `klr_safe.py`.**
Proof: `artifacts/reports/deepcopy_probe.py` (exec's the real `__deepcopy__` bodies against ctypes/stub tensors —
there is no torch or GPU on this box; all PASS, exit 0).

## 1. Copy hooks (the file has exactly three)

| hook | line | covers | holds CUDA tensors / raw pointers? | shares across a state copy? |
|---|---|---|---|---|
| `_GraftLt.__deepcopy__` | 12268 | cuBLASLt wrapper | `workspace` (64 MB CUDA tensor), `handle`+`desc` (`c_void_p`), `alpha`/`beta` (`c_float`), `lib` (`CDLL`) | **tensor copied**; handles/lib shared. Handles are immutable after `__init__` and only read by `cublasLtMatmul`, and `c_void_p` cannot be deep-copied at all (default copy raises `ValueError`), so sharing is the only option. Correct. |
| `_GraftPlan.__deepcopy__` | 12280 | one plan dict per (matrix, m) | `weight` (CUDA tensor, the state's own `pr`/`o`/`unpool`/`qkv`), `lt` (`_GraftLt`), `layouts` (3 × matrix-layout `c_void_p`) | **`weight` copied** (and stays aliased with the copy's own `st['pr']` via `memo`), `lt` copied through `_GraftLt`, `layouts` shared — they carry shape/stride/stride metadata only; the weight address is passed per call (`_graft_lt_run`: `_GP(plan["weight"].data_ptr())`), never baked into a layout. Correct. |
| `_RouterExt.__deepcopy__` | 13508 | compiled-extension wrapper (`_ma_ext(...)`, `_router_fp32_ext()`) | **no tensors**: only pybind function attributes. No Python code assigns attributes onto an ext module (grep `setattr`/`ext.<attr> =` → none) and the C++ has no static tensors or pointer-keyed caches (only the `static PFN_cuTensorMapEncodeTiled` function pointer; every tensor map is re-encoded per call from the live `data_ptr()`). | returns `self`. This is exactly both rivals' fix ("deep-copy everything except the compiled extension"). Correct. |

No `__copy__`, `__getstate__`, `__setstate__`, `__reduce__` anywhere in the file.

Every other state entry is a torch tensor, `dict`, `list`, `tuple`, `set`, `int`, `bool` or a function, i.e. default
`copy.deepcopy` semantics (fresh storage per tensor; functions are atomic and share). There is no custom class in the
state that could leak a tensor through a `self`-returning hook, and no non-leaf tensor (every state tensor is built
from `requires_grad=False` inputs), so `Tensor.__deepcopy__` cannot raise either.

Verified with `copy.deepcopy(state)` twice: 0 shared tensor objects between the two copies; the plan weight stays
aliased *within* each copy, layouts/handle/desc/`CDLL`/ext-module shared by design.

## 2. Per-thread primary CUDA context

- Bind site: `_glue_bind_context` (13433), called as the **first statement** of the last `kernel_fn` (13451), which
  wraps the 12529 `kernel_fn` (`_glue_context_parent_kernel`), which wraps 11465. No later rebinding of `kernel_fn`
  exists (AST check in the probe).
- TLS-keyed (`_GLUE_CONTEXT_TLS`), caches the device index, and re-binds if `cuCtxGetDevice` disagrees — stricter than
  the rival's null-context-only check. Retains the primary context and `cuCtxSetCurrent`s it: this is the exact fix
  for cuBLASLt status 13 / `cuTensorMapEncodeTiled` 201.
- Every launcher the judge's timed path can reach descends from that one entry:
  - graft Lt: `_graft_lt_run` ← `dense`'s `(T,name)` plan branch (12395) and the qkv graft at 12537;
  - rq: `_rq_gemm` ← `dense`, 12535;
  - `dense()` itself ← `_pool_router_fp32` (13484) ← `kernel_fn`;
  - triton / TMA / router_fp32 launches: all inside the same `kernel_fn` call graph.
- prepare-time fresh threads carry no launches: the `_build` fan-out at 11245 (`_att7`, `_moe*`, `bfp__get_ext`,
  `_graft_glue_module`, `_graft_b2_module`, `_sl_cluster_module`), `_rqstart`'s `build` (15998) and
  `_mf_start_base_builds.run` (16259) only run `load_inline` compiles. `prepare()` itself performs torch CUDA ops
  before its raw-cuBLASLt work (`_graft_lt_plan`), so the prepare thread's primary context is current even when the
  judge calls `prepare()` on a fresh worker thread.

## 3. Residual notes (not these two traps, no action)

1. `_GUARDS` is a module-global list of the **original** `ma_scratch` buffers and `_check_guards()` only walks it.
   A copy's guarded views do keep their sentinel tail (torch deepcopy copies the whole storage, offset/size/stride
   preserved), but nothing ever checks the copy's tail — guard coverage is copy-1 only.
2. `_PF7` (11384) and `_GLE_W_CUR` (14283) are plain module globals, not thread-local. Each is re-pointed from the
   *currently passed* state immediately before use and cleared right after, so it is safe for the judge's sequential
   timing, but two state copies timed concurrently on two threads would cross wires (wrong but valid pointer — no UAF).
3. `_glue_bind_context` never releases the per-thread `cuDevicePrimaryCtxRetain`; refcount grows with the number of
   fresh worker threads. Harmless at judge thread counts.

## 4. State-key audit (every `st[...]` / `state[...]` / `bst[...]` assignment in the file)

`bst` = `state["bfp"]`. "tensor" = a CUDA tensor, deep-copied into fresh storage.

| key | content | shares after deepcopy? |
|---|---|---|
| `router_fp32, qkv, o, pr, unpool, unpool_slice, gate1, gate2, g_preffn, g_preexp, g_postexp, g_postffn, g_next, bias, w13q, w13s, gs13, w2q, w2s, gs2, invf0, invf` | tensor | no |
| `ma_w` | list of 15 tensors (packed experts, scales, LUT, norm, fat/unit/WGS copies) | no |
| `ma_scratch` | list of 3 `_guarded` tensor views into sentinel-tailed buffers | no |
| `graft_t_<name>` (Lt NN transposes) | tensor | no |
| `graft` | dict: `alpha1/2, beta1/2, pre1/2, post2` tensors | no |
| `graft_lt` | dict `(name,m) -> _GraftPlan` | weight copied; `layouts`/handle/desc shared (no tensor) |
| `graft_lt_status` | dict `str -> (int, int)` | no |
| `ma_ext` | dict `build -> _RouterExt` (compiled ext) | yes — no tensors (correct per both rivals) |
| `router_fp32_ext` | `_RouterExt` (compiled ext) | yes — no tensors |
| `ma_sfx` (`tuple`), `nsm`, `pbmax`, `att_ok`, `ma_ok` | immutable scalars | n/a (shared values) |
| `fa3` | function `flash_attn_with_kvcache` | atomic copy (expected) |
| `moe_pick, gluev2, marlin_cfg, mm_pick, choice, compose_base_mm, compose_graft_enabled, compose_boundary_warps` | dict of scalars | no |
| `compose_base_graft_lt` | `dict(st['graft_lt'])` — same `_GraftPlan` objects | same copied plans as `graft_lt` (memo keeps the alias) |
| `compose_active` (`set`), `compose_selection` (`list`) | scalars | no |
| `bfp` | dict, see next rows | no |
| `bfp.raw, cnt, comp, mid_cnt, streams, tabs, base4` | dict of layer -> tensor | no |
| `bfp.sk2` | dict `name -> (ci:int, stream:tensor, tab:tensor, pmax:int, b4:int, n_rows:int)` | no |
| `bfp.dummy, ws_shared, mid_ws_shared, ws_shared_diag, router_hi_t, router_lo_t, router_cat, fused_part, fused_cnt` | tensor | no |
| `bfp.wtile, bfp.diag_cnt` | dict -> tensor | no |
| `bfp.old_paths` (bool), `bfp.sm90` (bool), `bfp.disabled` (set), `bfp.tune_log` (list) | scalars | no |
| `rq` | dict `(layer,m) -> {i,lay,esc,esc2,cnt,KS,tab,n}` (4 tensors + ints; no module) | no |
| `rq_pk` | dict layer -> pack tuple of tensors | no |
| `rq_e, rq_e2e, rq_sink` | tensor | no |
| `chen_rq`, `chen_st` | dict of tensors (`rq_e`, `rq_sink`) | no |
| `bank` | `None` | n/a |
