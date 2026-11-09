# R7 — Nearby-shape robustness of the 23 wave files

Judge reruns "the scored shapes and nearby ones" (AGENTS.md, Validation). All 23 files share one dispatch skeleton; line cites below are ks_full.py (klr_* siblings same functions at their own offsets).

## Attention (hardest surface)
- `_supported7` ks_full.py:11665-11667 — head_dim==64, q_heads==8*kv_heads, kv_heads even, q_len in (1,8), batch<=512, page_size power of two, batch*(kv_heads//2)<=4096: all static shape checks, graph-legal.
- `_att_fits7` :11741-11748 — 32-bit index-math ceilings; exceeding shapes fall through to FA3 instead of corrupting.
- Fallback ladder kernel_fn: agent kernel -> try/except RuntimeError -> _kernel_fn_fa3 :11758-11771 -> FA3 flash_attn_with_kvcache when state["fa3"] -> own Triton _attn_split/_attn_combine :11804-11817 when sgl_kernel absent (guarded import :11564-11569). Any q_len/batch/context/page_size the custom kernel declines still computes correctly.
- q_len outside {1,8}: _supported7 False -> FA3 -> correct (max_seqlen_q passed, window honored :11799-11801).

## Token count T
- `_bucket(T)` :10668-10674 maps any T to nearest tuned count at-or-above, saturates at 256 — every pick lookup tolerates untuned T; T>256 -> 256 row.
- `dense()` :10677-10693 — T>32 -> plain torch.mm fp32-out; Triton split-K only serves T<=32 — no tuned kernel ever sees an untuned T.
- `_pick_expert_backend` :11819-11823 — Marlin <-> agent mutual fallback; marlin path re-derives block_size_m for any num_tokens when cfgm None (:11260-11268) — no hard T cap.

## Remaining risk
- `torch.mm(..., out_dtype=torch.float32)` in T>32 dense + qkv fast-path (:10685, :11763) — needs torch>=2.3, fine on judge image.
- verify loads are all q_len=8 (TASK.md); a nearby rerun at q_len!=1,8 uses FA3 — slower but legal.
- No shape hits a bare assert/IndexError in kernel_fn: every dispatch is .get-with-default or _bucket-clamped.
- Conclusion: safe for nearby reruns; expected nearby outcome is "slower via FA3/Triton fallback", not a crash. Only a much-slower-than-scored nearby result invites "a closer look" (AGENTS.md).

## Per-file deltas (all SAFE)
ks_full/ks_full_t1/kitchen_sink/ks_flatwin/ks_isync/ks_routev4_xqa/ks_xqa/ks_ft32/ks_exh8/ks_bulkred/ks_br_*/ks_noq8_ft32/ks_ft32_exh8/klr_*/sweep_b7_qss_off: all dispatch guards are .get-with-default or _bucket-clamped; every scored shape has >=2 fallback tiers (ATT7->FA3->Triton; tuned pick->_bucket->torch.mm; graft->base kernel). Watch: kv_scale=1/64+q_len-8 ATT7 rounding coarser (accuracy); MoE agent-build fail + bank=None -> TypeError (uniform all shapes).
