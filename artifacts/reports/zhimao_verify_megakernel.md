# ZhimaoL verify megakernels — fetched, read, priced

## 0. Where they actually live (the 404 path was one level too deep)

`GET /repos/ZhimaoL/sglang/git/trees/<sha>?recursive=1` returns 200 and `truncated:false` for **both**
dangling heads (`x-eod` 87c325b0…, `x-w1b` f01c7e17…). The guessed
`…/muse_spark_v12/okc_patch/helion_megakernels/` does not exist; the real prefix drops `okc_patch/`:

    python/sglang/kernels/ops/muse_spark_v12/helion_megakernels/{__init__,attention_verify_megakernel,
      attention_boundary_verify_megakernel,moe_tp4_verify_megakernel,lm_head_verify_megakernel,
      megakernel_verify,integration,fp4_utils}.py
    python/sglang/kernels/ops/muse_spark_v12/helion_{norm_fp4_quant,norm_gate,post_expert,qk_norm,runtime}.py
    test/registered/kernels/ops/muse_spark/test_helion_{attention_verify,attention_boundary_verify,
      moe_tp4_verify,lm_head_verify}_megakernel.py
    test/registered/unit/models/test_muse_spark_helion_megakernel_integration.py

The three requested files are **byte-identical** at the two heads (`cmp` SAME for all three), so there is one
version, not two. Saved to `artifacts/cas/third_party/zhimao/sglang_x-w1b/…` (paths preserved; sha256 prefixes:
`db40fc4645d71d72` attention, `c98266717d17c0b3` boundary, `29cfa46cd2e2572a` moe). `x-w1b` only adds
`okc_patch/w1b.py` + `w1b_trace.py`; `x-eod` only adds `okc_patch/{eod.py,_cuda_src_eod.py}`.

`okc_patch/__init__.py` is **not** a layer-board file: its header is "OKC serving patch for Muse Spark: the pg
grouped-GEMM MoE pair on prefill chunks", i.e. the serving-track patch. Nothing in this fork is a
`# okc: problem=muse_decode_layer` file, so the 2.3455 row cannot be read out of this tree.

## 1. What the megakernel is

One layer of the Muse Spark 1.2 verify pass (B=1, T=8/16 draft tokens) is cut into **four** persistent Helion
kernels, each a single grid that runs several "roots" back to back, chained by PDL
(`hl.pdl_wait()` at entry, `hl.pdl_launch_dependents()` immediately after):

| kernel | roots it fuses (stock stages it replaces) | grid |
|---|---|---|
| `attention_verify_megakernel` | FusedQKNorm + RoPE + FP8 Q/K and K/V store, **plus** FI `trtllm_batch_decode` split-KV paged attention and its split merge, **plus** `rms_norm_gate` | ATTEND `(kv_head, split)` = 4 × (SMs/4) tiles = **148 CTAs on B200**, 1 CTA/SM, `num_warps=8`, `num_stages=1`; then MERGE `(kv_head, token)`. Sliding-window layers only — global layers keep stock attention |
| `attention_boundary_verify_megakernel` | o_proj GEMM + one-shot **all-reduce** + post-attn norm / residual / pre-FFN norm + pooler + router top-8 + pre-expert NVFP4 quant | roots A (o_proj K-shard, `[N/R/CB, CB]`) → B (rank-order AR + moments) → C (norms) → R1/P1 split-K partials → P2/R2 symm sum → R3 top-8 → Q (RMSNorm+NVFP4) |
| `moe_tp4_verify_megakernel` | FI TRT-LLM MoE routing + GEMM1/SwiGLU + NVFP4 quant of z + GEMM2 + `finalize_with_post_expert_norm`; the **tail** (unpooler, post-FFN norm, next pre-attn norm, next layer's QKV GEMV) is a second launch | R1 `[q_tiles, slots]`, R2 `[h_tiles, slots]`, R3 `[tokens, H/W/128]` |
| `lm_head_verify_megakernel` | FP32 lm head GEMM + logit scale + softcap | 1 root |

Language: **Helion IR only** (`helion.kernel`, `hl.tile`, `hl.dot_scaled`, `hl.inline_asm_elementwise`,
`hl.ProcessGroupName`, `cross_loop_pipeline`, `pid_type="persistent_blocked"`). No hand-written CUDA, no
Triton source in these files — the Triton comes out of `backend="triton"` at compile time.

Caller requirements (`integration.py`, `helion_runtime.py`): `helion >= 1.5.0` with `hl.pdl_wait`,
`hl.pdl_launch_dependents` and `Config(cross_loop_pipeline=…)`; `msgspec`; an sglang **serve** stack —
`TRTLLMHAAttnBackend` (unscaled FP8 KV, page 128, window 2047, RoPE), `get_token_to_kv_pool`,
`get_tp_group`, `is_cp_active`, `is_dp_attention_enabled`, `forward_batch.is_target_verify()`,
`muse_spark_expert_tp.VERIFY_TOKENS`, `fused_norm_residual.FusedPostNormResidualPreNorm`, a prepared
FlashInfer TRT-LLM NVFP4 weight bank, and **TP4/EP4**: `world = group.size()`, `symm_mem.empty(...)`,
`torch.ops.symm_mem.get_remote_tensors`, `hl.ProcessGroupName`, `dist.all_to_all_single` in `_bank`.
`_check_model_dims` hard-fails any other shape.

## 2. Price against our layer

Facts used (already measured, not re-derived): verify replay totals and ATT7 share from
`klr_qd_candidate.md` (82c098ae profile) and `kernarc_digest.md`; glue/launch behaviour from
`p_mvsjkujxf6f810de`.

| load | replay (sliding / full) us | weighted ATT7 share | glue | what one megakernel can remove |
|---|---|---|---|---|
| v1 | 110.9 / 125.8 | 7.85 % | 4-5 launches, 10.5-12.5 us | ≤2-4 us (1.8-3.6 %) |
| v16 | 251.7 / 328.7 | 12.02 % | same | ≤2-4 us (0.8-1.6 %) |
| v32 | 309.2 / 467.6 | 14.23 % | same | ≤1-2 us (0.3-0.7 %) |

Why that is the whole prize, and why it shrinks with T:

1. **A fusion does not delete work, only boundaries.** o_proj, the norms, the router and attention all still
   execute; what a merged grid saves is the launch boundary and the intermediate round-trip. The round-trips
   are `T×2048 bf16` = 0.13 MB at T=32 (L2-resident, ~0.05 us) — qkv is already direct-loaded.
2. **Our layer is already the megakernel's end state.** Verify is one serial PDL chain with no idle gaps,
   inter-kernel idle **<1.1 us** at v32s; deleting a glue launch outright buys only **1.2-2.0 us**, all three
   **4.0-4.8 us** at T<=8 (because the next GEMV streams 2 pre-wait weight stages over the glue). Fusion
   recovers the boundary of that 1.2-2.0 us, not the number.
3. **The co-residency penalty is measured and it dominates.** "Verify attention slows **50-80 %** next to any
   co-resident GEMM (`v32s` 14.9 → 22-27 us)." A persistent megakernel with ATTEND tasks and o_proj/router
   CTAs in one grid *is* that regime by construction. At v32 that is **-7 to -12 us (2.3-3.9 %)** against the
   +1-2 us the fusion can win: net negative, twice the removable amount.
4. **The MoE third of the idea is already shipped in our file.** `okc::moe_fused` is a single persistent
   launch per layer (`muse_decode_layer.py:6853`), with `griddepcontrol.wait/launch_dependents` inside
   (`:6257`, `:6472`), and the file's own MoE source is commented "one persistent kernel: routing prologue,
   GEMM1+SwiGLU, GEMM2+norm stats, final norm/sum" (`:12583`) — their R1/R2/R3 design. The remainder is
   TP4/symm_mem, which does not exist in a single-GPU problem.
5. **Their attention is sliding-window only**, which is exactly our 4.8-6.5 % arm — the arm where a rival
   attention core is worth **+1.55 % on the attention board**, i.e. **0.1-0.2 % of the layer**.

Layer-board conversion (1/6 per load in log space, verify loads only): best case
≈ (1.5 + 1.0 + 0.5)/6 ≈ **+0.3 %**, before the v32 penalty; realistic **≤0.2 %**, inside the ±6 % per-shape
judge noise and the "single check under ~2 % is noise" rule.

## 3. Port cost, in files

- **Blocking, not just expensive.** `helion` and `msgspec` are **not in the judge image allowlist**
  (`verifier_static.py:140`, "modules present in lmsysorg/sglang:dev-muse-spark-open"): `lint` on
  `import helion` prints `[warning] portability: 'helion' is not in the locked judge image's allowlist` and the
  module is not installed in our env or the image. The source would have to be **transliterated to
  Triton/CUDA**; the Helion IR cannot ship as-is.
- **Size.** 13 modules, 4,233 lines, 142 KB. `SERVING_MAX_BYTES = 1_000_000` is an **error**-severity cap and
  our board file is already **979,654 bytes**: 20,346 bytes of headroom for a rewrite whose attention arm
  alone is 563 Helion lines + the `helion_norm_gate` intrinsics (estimated 1,500-2,500 Triton/CUDA lines).
- **Structure.** Their `attention_verify` + `attention_boundary` are 1,202 Helion lines that replace a
  TP4 serving path we do not have; the single-GPU parts are the norm/RoPE/softmax algebra and the MERGE root,
  both of which ATT7 and `_attn_combine` already implement.

## 4. Verdict

**Do not port.** The megakernel is an sglang TP4 serving-path artifact (Helion IR, symm_mem all-reduce,
FlashInfer NVFP4 banks, `group.size()==4`), not a kernel we can drop into a single-GPU single-file layer; and
even a faithful single-GPU transliteration has a ≤2-4 us / ≤1-2 us prize on v1 / v32 against a **measured
7-12 us co-residency penalty** at v32, with ≤20 KB of file budget left to spend. The one portable residue —
folding `_attn_combine` (and optionally `_qk_prep`) into `_attn_split`'s launch, which is their MERGE root —
is 1-2 launches ≈ 1-2 us on verify loads ≈ **+0.1-0.3 % board**, below the single-check noise floor.

## 5. Skip / risk

- Did not build or run any Helion kernel (helion absent; the judge image is the only place it could run).
- The +0.3 % ceiling is an arithmetic bound from published component shares, not a measurement: nobody has
  built a single-GPU attention+boundary fusion to compare. The co-residency figure (50-80 %) is the one
  measured fact that turns the bound negative, and it was measured on alygg77's lineage, not 82c098ae.
