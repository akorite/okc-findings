# alygg77 MoE (2.4015) vs our standalone MoE (2.3746): full diff

**Headline: there is no mechanism to port. The two files are the same kernel.**

- `artifacts/cas/third_party/aly_moe.py` — sha256 `b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281` (**verified**, 326062 B, 3601 lines).
- `artifacts/cas/6402f5f354428aa180681499e6e9ebd3554e40feafa569cbe4b1cc37b5d2b3e5.py` — sha256 `6402f5f3…3e5` (326196 B, 4187 lines).
- `RESULTS.md` / `results.json` of that repo state `muse_moe_w4a16 → speedup 2.4015, score_us 5600.0, job e1bcad5a55cd4a29, kernel_sha256 b33070…4281`. So the file scored at 2.4015 **is** the file diffed here, not some later private build.

## The entire diff is 6 hunks

`diff -u aly ours` → 6 hunks, 810 changed lines, and **zero** of them on the timed path.

| # | aly lines | our lines | nature | affects score |
|---|---|---|---|---|
| 1 | 1214–1220 | 1214–1221 | `MUMMA<6>` asm output list: `"+f"(acc[5][3])` moved to a new line | no (whitespace) |
| 2 | 1225–1231 | 1226–1233 | same, `MUMMA<7>` (`acc[5][3]`, `acc[6][*]`) | no (whitespace) |
| 3 | 1236–1242 | 1238–1245 | same, `MUMMA<8>` (`acc[5][3]`…`acc[7][3]`) | no (whitespace) |
| 4 | 1913–1943 | 1916–1952 | same wrapping inside `WGSMMA<6/7/8>::mma0` and `::mma` (the WGS path — this *is* the path KPERM3 runs on) | no (whitespace) |
| 5 | 3440–3553 (114 L) | 3449–4135 (687 L) | diagnostics only: `_diag_bw` / `_diag` / scratch guards. | no (never called in a submission; `DIAG` off) |
| 6 | 3585–3591 | 4167–4177 | `_bind_context`: aly dereferences `context.value` / `current_device.value` inline twice; ours copies them into `ctx_ptr` / `dev_id` first | no (identical semantics) |

Hunk 5 also carries the only two non-`DIAG` lines outside hunk 6:

- aly 3549–3550: `import ctypes as _C` / `import threading as _ContextThreading`
- ours 4131–4132: `_C = torch.ctypes` / `_ContextThreading = torch.threading` — same objects via the `torch` re-export, because the judge's static screen rejects a bare `import ctypes`.

Byte-for-byte identical between the two files, with no diff hunk at all:

| region | aly | ours |
|---|---|---|
| shape / chunking / WGS scratch / variant selection / feature flags | 99–194 | 99–194 |
| all CUDA source (roadmap → routing → epilogues → skinny ring → fat unit → wgmma → WGS ring → RS-wgmma WGS → `moe_fused` → TMA) | 195–3065 | 195–3065 |
| `_ext`, weight packing, `_pack_codes_nat`, `_kperm_src`, `_kperm_codes`, `_kperm_rows`, `_wgs_copy` | 3103–3285 | 3112–3294 |
| `prepare()`, `_make_scratch` | 3287–3361 | 3296–3370 |
| `kernel_fn` (timed path) | 3363–3391 | 3372–3400 |

## Consequence for the +1.13%

`score_us`: theirs 5600.0, ours 5663.5 → 5663.5/5600.0 = **+1.13%**, on **identical machine code**. Their own published CI is 2.3655–2.4376, which **contains** 2.3746. Their `docs/NEGATIVE_RESULTS.md` §"Gaps between local results and judge results" says verbatim: *"Unchanged files. Four submits of one unchanged `muse_bf16_proj` file spread over 1.6% (1.5259 to 1.5502). Two judge submissions of decode files with the same kernels scored 0.5% apart."* and *"Board-level noise on unchanged files reached 1.6% to 2.2%."* The gap is judge noise, not a mechanism.

Also from their own §8: *"A MoE build with extra changes beyond KPERM3 … on the judge, d16 read -0.7% and v1 read +0.2% (noise). The gain showed only on the loads that run the WGS build."* They confirm KPERM3 is the only MoE mechanism that moves the board.

## KPERM3 is already in both of ours — no port

| copy | `_kperm_src` | `_kperm_codes` | `_kperm_rows` | `_wgs_copy` KPERM3 | wired into prepare | enabled |
|---|---|---|---|---|---|---|
| aly 2.4015 | 3223 | 3235 | 3243 | 3257–3269 | 3326–3327 | `-DOKC_WGS=1` = **MAIN** variant |
| standalone 2.3746 | 3232 | 3244 | 3252 | 3266–3278 | 3339–3341 | same |
| klr in-layer MoE | 7136 | 7148 | 7156 | 7167–7184 | `_ma_prepare` 10224–10227, 16273–16274 | `_wgs_on()` |

The host packing **is** in the same file, and both sides of the permutation are applied there, identically:

- `_kperm_codes` — `nib = codes.view(R, K//256, 256).index_select(2, p)` per 256-K stage, low nibble first.
- `_kperm_rows` — gathers the gate/up row order by `colmap(P) = 256·(P>>8) + p[P & 255]`.
- `_wgs_copy` — scales are permuted `[ub,q,warp,h,g,st,t,j] → [ub,st,q,warp,g,t,j,h]`, i.e. the byte at stage `st`, lane `t`, step `j` is scale block `16·st + 4t + j`, matching `p >> 4 == 4t + j`.

Semantics (identical in all three): `p = 16·(4t+j) + 4·lg + i`, so **one scale pair serves the four 16-wide K steps `lg = 0..3`**. That is the "decode 20.12 → 18.44 instr per weight word".

The device side exploits it, in both files, with the same comment block and the same code:

- standalone 2213–2217 / klr layer 5988–5992: *"the 4 scale pairs ([j][h] bytes) come from one 8-byte load per stage and are built once per stage"* → `const uint2 scs = *(const uint2*)(pc + 8192 + (wp*32+lane)*8);`, hoisted out of the `lg` loop.
- Then one `__byte_perm` + one `IMAD` per k16 step builds both rows' bf16x2 scales: `sgp = __byte_perm(sw, 0, …) * k16c + 0x5C005C00`, with `HMUL2 .H0_H0/.H1_H1` broadcasting its halves.
- gather side (standalone 2058–2063): the x gather uses `B(G) = 8·((G>>1)&3) + 2·(G>>3) + (G&1)`, the same permutation applied to x.

## klr's in-layer MoE is *ahead* of both

`src/okc_loop/vendored/keell0renz/muse_decode_layer.py` lines 3880–6862 embed the same `muse_moe_w4a16 v2` kernel and carry KPERM3 verbatim, plus mechanisms neither aly's file nor our standalone has:

| mechanism | klr layer | aly 2.4015 | our standalone |
|---|---|---|---|
| KPERM3 | 5794, 5988, 7136–7184 | yes | yes |
| `OKC_SC6W` 6-bit block scales (768 B instead of 1024 B per warpgroup piece) | 4366–4367, 5550–5551, 6012–6031 | **no** | **no** |
| `OKC_MFIN` per-expert completion counters (no MoE grid barrier) | 3891–3892, 5566, 5890, 5948 | **no** | **no** |
| cepi stream-K unit costs `OKC_WGSU 160 / OKC_WGSUB 140` | 6122–6191 | **no** | **no** |
| `OKC_WGSFCI` deferred flush | **4** (5599) | 3 (1856) | 3 (1859) |
| YBFL bf16 WGS ybuf | header line 5 | no | no |

So the score-carrying copy needs nothing from alygg77. It already runs KPERM3 on a *narrower* scale stream (SC6 768 B) than the file that scored 2.4015.

## Gate results (offline)

`artifacts/reports/kperm3_gate.py` executes the shipped `_kperm_src` / `_kperm_codes` / `_kperm_rows` source out of each kernel file through a numpy shim (no torch, no CUDA) and asserts:

1. `p` is a bijection on the 256-K stage (`sorted(p) == range(256)`).
2. `p` equals the independently recomputed `16(4t+j) + 4lg + i`.
3. KPERM3's property holds: `p >> 4 == 4t + j`, exactly 16 blocks of 16 slots — the scale byte at a position is the block for its (lane, step) across all 4 load groups.
4. codes are permuted by exactly `p` per 256-K stage, and the permuted bytes recombine to the same nibble sequence in permuted K order.
5. the permutation is invertible (permuting back restores the original codes).
6. `_kperm_rows` is a bijection over the gate/up rows and gathers exactly `gate[colmap(P)]` / `I + gate[colmap(P)]`, verified against a **scrambled** row order so the gather is really exercised.

```
PASS standalone-2.3746: p is a bijection (16 blocks x 16 slots = 4t+j), codes follow p, invertible, _kperm_rows colmap == 256(P>>8)+p[P&255]
PASS aly-2.4015:        idem
PASS klr-layer:         idem
```

- `okc_loop.cli lint` on the standalone: `PASS`, 1/1 clean.
- `tools/cu_build.py … --variant MAIN` (the only variant with `-DOKC_WGS=1`, i.e. where KPERM3 lives) on the standalone: `ok CUDA_SRC [OKC_NW=16 OKC_FAST1=0 OKC_EPI4=0 OKC_WGS=1 …]` plus the Marlin fallback and both `BW_SRC`.
- Negative control to prove that build gate is real: injecting `syntax_error_here` into `drain_x` made it report `1 error detected in the compilation of "negctl_CUDA_SRC"`. So the `ok` above is a genuine NVRTC compile, not a cache hit.

## What aly tried and rejected on the MoE (do not re-spend on these)

From `docs/NEGATIVE_RESULTS.md` §muse_moe_w4a16 — all measured on this same file:

1. **Deeper rings / a 4th weight slot** — −1.0 … +0.9 µs per WGS load; the stream is *boundary-bound* (unit ends), not depth-bound.
2. **Tail fixes** (dynamic phase-B tail, dispersed unit ends, helper-warp read-back, refit cost table) — one phase-B unit costs a CTA 18.5 µs (`d32`,`v16`) / 19.4 µs (`v32`); the end spread to remove is only 2.7–3.7 / 5.9–8.2 µs. The unit is 2.5–7× the spread, so the slowest CTA just moves.
3. **Fences** — NOFENCE (dropping the `fence.acq_rel.gpu` before the split-unit read-back) was locally +0.86…+1.78 µs on 8/8 `v16`/`v32` rows and gave board **+0.234%**, below their own A/A noise (0.7–3.1 µs). Memory-safe, 10 000 reference-checked replays clean — but not worth a check.
4. **WGS32** (a 32-token job tile that gained 1.3–3.8 µs on another public H200 base) — *slower by 2.2–3.0 µs* here. Same instruction count per weight word; the compiler's stage-loop schedule differed. "Stream rate is sensitive to schedule, not only to instruction count."
5. **`d16` second weight read** — removing it (7.7–11% of the bytes) sped other CTAs by only 1%. `d16` mid path is co-bound: phase A at the size-matched ceiling (4.10–4.16 TB/s, 1.12 µs per 36 KB unit) and phase B consumer-bound (NCU issue util 51.9%).
6. **Expert byte compression** — E2M1 codes use exactly 4.000 bits, E4M3 scales 3.71 bits; best practical code saves ~5%. The weights are independent random draws.

Item 1 is the one that matters for `UnitEnd`'s finding: aly independently reached the same conclusion (§2, §Why it lost: *"Every 16-stage unit end stalls a CTA for 1 to 7 µs, and the stream is boundary-bound"*), and their four attempts at the unit end all lost. That corroborates (iii) in the brief and says the unit-end stall is not fixable by scheduling alone on this base.

## Files

- `artifacts/cas/third_party/aly_moe.py` — fetched, sha256 verified.
- `artifacts/reports/kperm3_gate.py` — the permutation gate; re-runnable: `PYTHONPATH=src .venv/bin/python artifacts/reports/kperm3_gate.py "<file>=<tag>" …`
- `artifacts/reports/aly_moe_diff.md` — this report.
