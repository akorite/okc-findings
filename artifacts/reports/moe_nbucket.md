# MoE N-selection: do we tile wgmma N to the token count, or pad? (DeepGEMM #473 check)

Base: `src/okc_loop/vendored_work/layer/klr_fci5.py`, judge **2.28370 / 9,905.1 µs**, job
`9743b6e71fbd42fd`, profile `profiles/klr_fci5.9743b6e71fbd42fd.txt`. Line numbers below are that file.
DeepGEMM side: **PR #473's own diff** (11,441 lines, `deepseek-ai/DeepGEMM` `main`, opened 2026-10-14), read in full
and quoted; cached as `artifact://5918`.

---

## 0. Verdict

| question | answer |
|---|---|
| Do we pad the MMA N to a fixed width? | **No.** N is `ceil(tokens/8)` per expert tile, chosen at run time, exactly #473's rule. |
| Which N is used on the scored mix? | **8** on 5 of 6 loads, **8 and 16** at verify 32 (`18 × ntt=1, 14 × ntt=2`, measured). |
| How much of the MMA work is padding? | **30.4–87.5 % of the N columns** (the last 8-row tile, which #473 also pays: at M=1 they run N8 = 87.5 %). |
| How much of the *load time* is that MMA? | **3.4–20.1 %** — the MMA is overlapped with a weight stream whose stage length it does not set. |
| Is there a concrete change to port? | **No.** The brief's premise ("pads to a fixed width") is false for this kernel; `moe_nbucket.py` is deliberately **not** written (see §4). |
| Expected layer % if "fixed" | **0.** Upper bound in the impossible limit (MMA 100 % exposed *and* padding freely removable): **+3.2…+4.3 board %**. |

The 1.44×/1.36× in #473 is **not** a padding result. Their own A/B comment prices the N-bucket family at
"−5..−14 % at M80…M256" (§2), i.e. at *large* M; the small-M gain they attribute to **register-A decode instead of
shared-memory expanded tiles (−21/−22/−20/−15 % at M8/16/32/64)** — a mechanism this kernel already has (`dq4()` →
A-fragment registers, `wgs_seg` :6028-6040, no expanded weight tile anywhere).

---

## 1. What our kernel actually selects (Q1)

### 1.1 The code path

| thing | cite | what it does |
|---|---|---|
| band ladder | :6956-6974 | M ≤ 4 TINY, ≤ 8 TINY2, ≤ 16 MID (all **no** `-DOKC_WGS`), 16 < M ≤ 32 B4 (= WGS), else MAIN WGS |
| fat-job cap | `OKC_FT_T 64` :3941, `WGS_NT = FT_T/8` :5549 | a WGS job = **one expert**, its whole token count, chunked at 64 |
| **the N decision** | **:6104** `const int ntt = (S.jC[u >> UBL] + 7) >> 3;` | n-tiles = `ceil(job tokens / 8)` — **per job, at run time** |
| N dispatch | :6105-6115 | `switch (ntt) case 1..8 → wgs_seg<PB, NTT>` — eight compile-time instantiations |
| the instructions | :5606-5686 | `WGMMA<1..8>` = `wgmma.mma_async.sync.aligned.m64n{8,16,24,32,40,48,56,64}k16.f32.bf16.bf16` |
| register bounding | :5969-5977 | `float d[WGS_NT][4]` but only `NTT` entries are ever named in an instantiation ⇒ the live footprint is `NTT×4` f32; `GS = OKC_WGSGS = 2` k16-steps per commit group because "GS=4 serializes the N=16..32 variants at 96 registers (C7512)" |
| x staging granularity | :5754 `npad = ((jC[j] + 7) >> 3) << 3`, :5761 `bytes = npad * 512` | the copied x block is `ntt` swizzled 8-row groups = **the same 8-row quantum**, not a fixed width |
| T ≤ 16 instruction | :4561 `two = (OKC_SKT > 8) && S.jC[j] > 8` with `OKC_SKT 8` :4026 and `static_assert(!OKC_WS \|\| OKC_SKT == 8)` | compile-time **false**: one `mma.sync.aligned.m16n8k16` n-tile per job, 16 weight rows in the A fragment (swap-AB, already ours) |

Two independent consequences, both checked:

* `ntt` is the *minimum legal* value: 8 is the hardware floor on both paths (`mma.sync` n8; `wgmma` n8 — there is no
  n4/n2/n1 wgmma, and the swizzled x block is 4,096 B per n8 group, so even the copy cannot be truncated further).
* Nothing anywhere rounds N up to a build-wide constant. `WGS_NT`/`WGS_FIXW`/`WGS_XS` are *allocations* sized for the
  64-token worst case; the executed instruction width is `NTT`.

### 1.2 The table

Jobs = expert tiles on one rank. Two estimators, both reported because they bracket: **(a)** the judge's floor
inversion (`moe_struct.md` §0, `touched = 0.75·distinct`) and **(b)** the measured n-tile histogram
(`costtab.md` §1: `d32 22 × ntt 1`; `v16 31 × ntt 1`; `v32 18 × ntt 1, 14 × ntt 2`). They disagree at decode 32 only,
and the profile's own time rules (b) out there (22 jobs × 28.31 MB = 623 MB cannot move in 86.5 µs; see §3.1).

| load | T | build | instruction | **N per expert tile** | jobs | Σ padded N cols | real routes | **MMA-N padding** |
|---|---|---|---|---|---|---|---|---|
| decode 1 | 1 | TINY (nw8, `mma.sync`) | m16n8k16 | **8** (one tile) | 1 | 8 | 1 | **87.5 %** |
| verify 1 | 8 | TINY2 | m16n8k16 | **8** | 6–7 | 48–56 | 8 | **83–86 %** |
| decode 16 | 16 | MID (`SPLIT`) | m16n8k16 | **8** | 10–13 | 80–104 | 16 | **80–85 %** |
| decode 32 | 32 | B4 (WGS) | **m64n8k16** | **8** | 12–16 | 96–128 | 32 | **67–75 %** |
| verify 16 | 128 | main (WGS) | **m64n8k16** | **8** | 23–31 | 184–248 | 128 | **30–48 %** |
| verify 32 | 256 | main (WGS) | **m64n8k16 / m64n16k16** | **8 or 16** (18/14, measured) | 32 | 368 (measured) | 256 | **30.4 %** |

Padding here is `(8·ntt − cnt)/(8·ntt)` summed over jobs — i.e. the share of issued MMA columns that compute a
padding row. It is *large* at small T and unavoidable: at T=1 you cannot issue less than one n8 tile, and #473 at M=1
issues exactly `MMA_64x8x32` too.

The only load with any bucket above 8 is verify 32, where 14 of 32 experts hold 9–16 tokens — and 16 is the bucket our
`WGSMMA<2>` already provides.

---

## 2. DeepGEMM #473's exact recipe (Q2)

Source: `deep_gemm/include/deep_gemm/impls/sm90_fp8_fused_mega_moe.cuh` (new, 5,780 lines),
`common/sm90_mxfp4_decode.cuh` (new, 150), `mma/sm90.cuh` (+52), `csrc/jit_kernels/heuristics/sm90_fused_mega_moe.hpp` (new, 852).
Quoted text is the PR's own comment.

### 2.1 What they added

1. **Register-A swap wgmma.** `MMA_64x{8,16,24,32,40,48}x32_F32E4M3E4M3_RS_TN` (`mma/sm90.cuh`, diff :8131-8153):
   `RS` = A from **registers**, B from smem; `M=64` is the **weight-row** axis, `N` is the **token** axis,
   `kNumAccum = M·N/128`. "the decoded weights are the WGMMA A operand (two M64 halves per warpgroup) and the tile's
   tokens the N operand, **in a bucket of 8 / 16 / 32 / 64 chosen per tile from valid_m**" (diff :2174-2176).
2. **The bucket rule**, run time, per tile (diff :5898-5918 for the L1 pass, :6014-6034 for L2):
   ```cpp
   if (n_swap <= 8)      run_swap_ab_l1.template operator()<8>();
   else if (n_swap <= 16) ... <16>();
   else if (n_swap <= 32) ... <32>();
   else                   ... <64>();
   // modes 5..7 replace the ladder with an exact switch: 8/16/24/32/40/48/56/64
   ```
   `n_swap` is `valid_m`, "runtime valid-row counts select the actual bucket" (docs, diff :10853). Buckets above
   `kMXFP4SwapMaxValidM` fall back to the regular mainloop ("per-tile hybrid", diff :5797-5800).
3. **Tiled register decode.** "Pack each 256-row x K128 weight stage into a contiguous 16 KiB tile and decode directly
   into register A fragments, avoiding expanded-weight shared-memory traffic and associated synchronization" (PR body).
   Implementation: E2M1→E4M3 bit patterns by `PRMT`+`LOP3` against a per-K32-group mantissa lookup built by one
   `mad.lo` (`sm90_mxfp4_decode.cuh`, diff :1883-1930), sign order re-permuted at **model load** so no sign gather
   runs in the loop, and `ld.shared.u32 [addr+imm]` with the byte offset as the instruction immediate so 16 word loads
   share one address register (diff :1944-1972).
4. **K handling.** `block_k = 128`; the wgmma is **k32** (FP8), so `frag[4][4]` = 4 K32 groups × 4 registers (diff :1965).
5. **Scale application.** Folded into the A fragment *at decode*: the UE8M0 byte of the K32 group is turned into a
   mantissa bias and merged by the same `PRMT`/`LOP3` that expands the nibbles (`sm90_mxfp4_e4m3_lookup`, diff
   :1888-1902); "the x64 is restored together with the per-expert secondary scale in the accumulator promotion".
   Activation scales are per **accumulator column (token)**, "padded tokens get scale 0" (diff :4264-4265).
6. **Register bounding.** `float acc[2][kSwapAccum]` (two M64 halves, diff :4290-4293), `kSwapAccum = N/2`, so
   N=8→8 regs … N=48→24; `kMXFP4SwapMaxBucket = 32` (diff :2526) and modes 6/7 raise it to 48. Their notes price the
   limit: "N56 every instance (STACK 64..160)", "an N48 bucket (double or single fragment set) spilled the
   cap-8192/M4096 instances" (diff :5225-5227). Code size is gated too: the N24 bucket "code alone, never executed,
   costs the M8..M32 tiles ~1.5 %", N48 "costs the small calls +2..+5 %" (diff :1069-1080) — so the host *compiles*
   buckets 5-7 only above an expected-tokens threshold.
7. **Epilogue.** Swap accumulators are transposed back through the (then idle) expanded-B tiles "so the regular
   epilogue runs unchanged", or fed directly to SwiGLU + activation quantisation ("direct swap epilogue", diff :6358).
8. **Split-K** for tiny M, gated on `split_k_max_tokens` (diff :2178, :5216-5227), plus dual-N96 for 65–96 rows.

### 2.2 Line-by-line vs our WGS path

| property | #473 (MXFP4, sm_90) | ours (`klr_fci5.py`, WGS) | same? |
|---|---|---|---|
| operand roles | A = **weights** (decoded to registers, M64), B = tokens (smem) — `RS_TN` | A = weights (`dq4`→`af[]` regs, m64), B = x tile in smem (`wg_desc`) | **same** (the swap-AB premise was already true) |
| N axis = tokens | yes, `N ∈ {8,16,24,32,40,48,56,64}` | yes, `N = 8·ntt`, `ntt ∈ 1..8` ⇒ same set | **same, ours is a superset at 64** |
| selection rule | `valid_m` rounded up to the bucket, per **tile** | `cnt = jC[j]` rounded up to the multiple of 8, per **job** (a tile ≡ a job here) | **same** |
| selection site | runtime `if/switch` in the mainloop, bucket = template arg | runtime `switch (ntt)` at :6105, bucket = template arg | **same shape** |
| fallback for big N | tile > 48 → regular mainloop | none needed: `FT_T = 64` caps a job at 8 tiles (`WGS_NT`) | equivalent |
| K | k32 per wgmma, `block_k = 128`, 4 groups per stage | k16 per wgmma (bf16), 256 K per stage, 16 stages per 16-row unit | differs with dtype only |
| scale placement | folded into the A fragment during decode (UE8M0→mantissa bias) | folded into the A fragment before the MMA (`PRMT`+`IMAD`, `sgp`, :6035) | **same** (pre-MMA fold) |
| weight expansion to smem | none (decode straight to registers) | none (`dq4` from the ring slot) | **same** |
| accumulator registers | `acc[2][kSwapAccum]`, capped by spill/code-size | `d[NTT][4]` per instantiation, commit width `GS=2` to fit 96 regs | **same** |
| epilogue | swap layout, transposed via the idle expanded tiles, or direct SwiGLU+quant | `wgs_epi_tile` on the swap layout; SwiGLU/norm/quant fused in phases A/B | **same** |
| small-M extra | split-K under a per-expert token bound | stream-K over 132 CTAs + 4 warpgroups splitting K (per-unit cost tables) | equivalent |
| what actually differs | `kStaticStages`: for N≤16 buckets with ≥48 k-blocks the stage index becomes compile-time "so the token descriptors come from the uniform datapath (no R2UR / vector address math per WGMMA)" (diff :4356-4419, measured −1.4…−2.8 %) | `xd = xd0 + xs*(WGS_XS>>4)` with `xs = seq % WGS_NX` live per stage (:5985); only the warp index is broadcast (`OKC_UWARP`) | **differs** — but this is a descriptor-address micro-opt, not N selection, and it is worth ≤ 0.5 % of the layer (§4) |

Nothing in the N dimension differs. The published 1.44× is against #411, whose small-M path lacked the register decode;
#473's own bucket A/B is quoted at M80…M256, and it *hurts* at M8 (+5.3 %).

---

## 3. Pricing (Q3)

### 3.1 Bytes first: the census is constrained by the clock

MoE µs per load (profile, real GPU time): 23.7 / 29.4 (decode 1), 70.4 / 69.6 (verify 1), 63.4 / 63.6 (decode 16),
86.5 / 83.9 (decode 32), 155.2 / 148.1 (verify 16), 186.6 / 199.4 (verify 32) — sliding / full.
One expert = 28.31 MB (`moe_struct.md` §0). At 4.8 TB/s, decode 32's 86.5 µs admits **≤ 14.7 experts**, so the
`costtab.md` census of 22 jobs cannot be this load's routing; the judge's floor inversion (12 touched / 16 distinct)
is the one to use there. verify 32's 186.6 µs admits ≤ 31.6 experts, so its 32 jobs (all local experts, mean 8 tokens)
is consistent. Hence the brackets in §1.2.

### 3.2 The MMA's share of each load

Padded MMA time = `Σcols · 12288 rows · 4096 K / (989 TFLOPS / 2)` — the judge's own dense bf16 number, 132 SMs.
Removable = the same minus the unpadded (`real routes`) work. `sens` = board %/µs from `muse_arch_levers.md` §2.1.

| load | MoE µs | padded MMA µs | **MMA % of load** | removable µs | board % if MMA were fully exposed |
|---|---|---|---|---|---|
| decode 1 S / F | 23.7 / 29.4 | 0.8 | **3.4 / 2.8** | 0.7 | 0.13 + 0.04 |
| verify 1 S / F | 70.4 / 69.6 | 4.9–5.7 | **7–8** | 4.1–4.9 | 0.56 + 0.19 |
| decode 16 S / F | 63.4 / 63.6 | 8.1–10.6 | **13–17** | 6.5–9.0 | 0.79 + 0.27 |
| decode 32 S / F | 86.5 / 83.9 | 9.8–13.0 | **11–15** | 6.5–9.8 | 1.02 + 0.35 |
| verify 16 S / F | 155.2 / 148.1 | 18.7–25.2 | **12–17** | 5.7–12.2 | 0.58 + 0.20 |
| verify 32 S / F | 186.6 / 199.4 | 37.5 | **20 / 19** | 11.4 | 0.40 + 0.24 |
| **Σ** | | | | | **+3.19 … +4.33** |

### 3.3 Why even that ceiling is unreachable

1. **The stage is weight-bound and the MMA does not set its length.** One WGS stage = `36,864 B` of weights
   (:5761 keeps `UNIT_BYTES` for the smem slot) + `4,096·ntt B` of x. At 4.8 TB/s / 132 SMs that is **1.01 µs**.
   The MMA in the same stage at ntt=1 is `64·8·256` MACs per warpgroup × 4 = 0.52 MMAC ⇒ **0.14 µs = 14 %**, and it
   is an async wgmma whose waits sit behind the ring (`wg_wait<1>`), never in front of the fetch. Measured per-stage
   time at ntt=1 is 1.156 µs (`aly_moe_diff.md:106` via `costtab.md` §2) — the weights, not the tensor core.
2. **The one place the schedule even sees `ntt` is the cost table, and mispricing it measured dead.** `OKC_WGSA/WGSB`
   ×`OKC_WGSU/WGSUB` are indexed by `ntt`; the shipped A over-states `cs(2)/cs(1)` by 16 % vs the byte model and the
   alternative tables moved the board **2.2386 vs 2.2474** (`costtab.md` §5), i.e. a ±16 % mispricing of the ntt
   dimension is worth **0.1 %**. If per-stage MMA work mattered, that experiment would have found it.
3. **Removing padding below 8 is physically impossible** on both instruction paths and in the swizzled x copy
   (4,096 B = one 8-row group, `bytes = npad·512`). #473 pays the same quantum.

### 3.4 The byte-side padding, priced separately (and already on the board)

The only traffic the 8-row rounding creates is the x refetch: `4,096·ntt B` per stage against `512·cnt B` real — 4–8 %
of a stage's bytes at our mix (e.g. decode 32: 4,096 vs ≈810 B real of a 40,960 B stage). Those 4 KB are the *same*
block re-read across the job's 768 stages, and the kernel already relies on the padding rows being garbage (the gather
"counts padding rows, does not write them", :5756). This is exactly the measured read amplification at verify 32
(622 MB for 540 MB, `muse_arch_levers.md` §A3, **+1.2 % board, medium risk**) — a *residency* fix (keep x resident,
trade ring depth for it), not an N-bucket fix, and not re-priced here.

---

## 4. Recommendation (Q4): **no port, no `moe_nbucket.py`**

The ticket's condition ("if a concrete change exists") is not met: the change the brief describes is already in the
kernel, with a bucket set that is a superset of the published one, on both instruction paths (mma.sync n8 for T ≤ 16,
wgmma m64n{8…64}k16 for T ≥ 17). Writing a default-off twin of a code path that is already optimal would be dead
weight, and the honesty cost of shipping an unvalidated kernel for a non-existent delta is worse than no candidate.

What would have to be true for this lever to pay, and does not hold here: (a) the MMA must be on the critical path —
it is ≤ 20 % of the load and overlapped; (b) N must be able to go below 8 — the ISA and the swizzled x copy both
forbid it; (c) there must be a fixed-width pad — there is none.

Two things I found and deliberately did **not** attempt (recorded so they are not re-derived):

1. **`kStaticStages`-style uniform stage index** (#473's only remaining N-adjacent trick, diff :4356-4419): make the x
   ring slot `xs = seq % WGS_NX` compile-time for the narrow buckets so the token descriptor is built in the uniform
   datapath. Their measured −1.4…−2.8 % is on *their* issue-bound loop; our `wgs_seg` is 20 % MMA-utilised and already
   hoists `xd0` per segment (:5977). Upside ≤ 0.5 % board, WGS-only (4 of 6 loads), unverifiable without a GPU, and a
   wrong descriptor is silent numeric corruption. Not worth a slot.
2. **The x-refetch / residency lever** (§3.4) — belongs to `muse_arch_levers.md` §A3, priced at +1.2 %, still open.

Incidental finding worth passing to whoever owns the bands: the profile's decode-32 MoE (86.5 µs) and verify-32
(186.6 µs) bound the expert count at ≤ 14.7 and ≤ 31.6 respectively, which means `costtab.md`'s `d32 = 22 jobs` census
and `moe_struct.md`'s "median 125.3 µs at bs 32" come from a more-spread routing draw than the scored load. The cost
tables were fitted on that draw; the shipped table is dead anyway (0.1 %), so it changes no decision.

## 5. What was verified here

| check | result |
|---|---|
| N selection in the scored file | read: :6104 `ntt = (jC+7)>>3`, :6105-6115 `switch`, :5606-5686 the eight `m64n{8..64}k16` instantiations |
| T ≤ 16 path | read: :4026 `OKC_SKT 8`, :4561 `two = (OKC_SKT > 8) && …` ⇒ compile-time false; MID/TINY/TINY2 tuples :6879-6885 carry no `-DOKC_WGS` |
| x copy quantum | read: :5754 `npad = ((jC+7)>>3)<<3`, :5761 `bytes = npad*512`, :5764 `cp.async.bulk` |
| #473 recipe | PR diff read end-to-end (`artifact://5918`): selector, bucket ladder/switch, register decode, K/scale placement, register bounds, mode gating, split-K |
| padding + MMA arithmetic | computed (§1.2, §3.2), anchored on the profile's real µs and the judge's 989 TFLOPS; census brackets cross-checked against the 4.8 TB/s byte bound |
| **not** verified | nothing was compiled or run: there is no GPU here, and no file was added or changed, so there is no `lint`/`cu_build`/`okc check` delta to report. The kernel was not touched. |
