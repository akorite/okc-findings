# fd_r1_gh_round2 — GitHub round 2, recently-updated first (2026-11-04 ~11:45 UTC)

## Headline
- **AutoclickerI/muse-spark-open-kernels** — opened public 11-03 12:45Z, last push 14:58Z. Scored decode file S22 (job 210d40e0c9cb4b71, 1.8892x #534/1233) byte-exact in `final/muse_decode_layer/` (8,113-line layer.py + 61 src files, Apache-2.0). 490-run judge ledger, docs/FINDINGS.md (63 closed-recipe negatives), tools.
- **nvbkdw/muse-spark-open-challenge** — pushed 11-01, missed earlier. Prefill 1.543x (#~375) + bf16 1.202x.
- nibague/open-kernel-challenge — 11-03; starter only, skip.
- Aravind-11/muse-moe-w4a16-kernels — 13 PRs v35 1.8696; all mechanism families known.
- No other watched handle changed since Nov 3. Nobody published a leader kernel.

## Per-handle deltas
pjmullin3 0 repos/gists; dolanzhao stale; gauravjain14 rl-at-home (RL infra, no kernels); gogowhy 404 (renamed/deleted); colinlikescode unrelated; DTennant unrelated; zro-inference 0 repos; tommyyliu unrelated; ruwwww attn 2.6032/moe 2.337 known; operatorstack/infraax unchanged; lee101 last commit 10-28 (v84/v85 parked, same kernel we carry); ZhimaoL sglang = upstream mirror, no wayback.

## infraax legality
Re-confirmed legal: prepare untimed ≤300s (L135), compile counts toward run limit not score (L129), bans = timing/harness tamper + work outside timed calls + output caching (L183). AutoclickerI `tools/h200_lab.py` proves the pattern is mainstream.

## New-repo detail
### AutoclickerI/muse-spark-open-kernels
decode 1.889 / moe 2.032 / attn 1.670 / bf16 1.202 / prefill 1.454 — all below ours. Mechanisms all known (PDL+prefetch split-K GEMV FP32 partials; 12-bit coded BF16 GEMV vjeux; persistent NVFP4 expert kernel w/ repack + 8-bit scale recode; prepare-time per-shape dispatch). Real assets: (1) byte-exact scored decode file w/ per-(load,layer-kind) attn port table; (2) VQ8/CS verify-block attention page-ring+DSM clusters +4.4%/+8.2% @verify16/32 receipt 863f502c; (3) C3 verify kernel reuses dead stage for new token (idea only); (4) tools/h200_lab.py, tools/board_model.py (rank ~= decode-dominated, tau 0.95 top-50), tools/promotion.py 95%-CI admission gate; (5) docs/FINDINGS.md 63 measured negatives — cross-check our dead list.

### nvbkdw/muse-spark-open-challenge
Prefill 1.543x: humming NVFP4xBF16 grouped-masked wgmma already in judge image (~640-700 TFLOP/s @T=16K vs Marlin 265); big/small concurrent two-launch split (160/176-row tile >32 routes, 16-row rest, side stream + stream-K locks) = 543→370 us experts @T=753. Custom NVRTC+ctypes FP8 GQA-packed wgmma attention +7-13% vs FA3 on 5 long full shapes (H100); passes static checks. Negatives matching ours: swap-AB NVFP4 MoE 3-12% slower; FA3 knobs/3-way split/SM-partitioning/routing-atomics no-gain. bf16 1.202x 12-bit lossless + cluster skinny-GEMM.

### Aravind-11
v35=1.8696: in-CTA split-K, per-shape token-tile search, activation-gather addr hoist, prepare-memoized scales, PDL, GEMM2 prefetches weights before GEMM1 wait, per-token-block completion counters, decode ALU→FMA pipe, 1B scales 2/reg, routing folded into GEMM CTAs (decode-1), wgmma variants (tied mma.sync — negative). All known.

## Searches re-run (deltas only)
sort=updated sweeps for all 5 problem names: top = AutoclickerI; nothing new past it. created:>2026-11-04 = 0. Code-search API 401 (unchanged). wst_ct14b: NF12 = NeurIPS paper only, no implementation.

## Impact
Plan unchanged: kitchen_sink/ks_full at 00:00 stays first. Optional: AutoclickerI FINDINGS.md dead-list cross-check; board_model.py sanity vs our rank model; VQ8 page-ring+DSM verify kernel only if a spare verify lane exists.
