# MoE ptxas/SASS audit — every build the boards run

Standalone MoE = `artifacts/cas/6402f5f3…b5d2b3e5.py` (MoE board 2.3746), embedded =
`src/okc_loop/vendored/keell0renz/muse_decode_layer.py:3880-6862` (`MOE_CUDA_SRC`). Compiled
offline as the judge does: nvrtc→PTX (compute_90a, -O3, -lineinfo) then ptxas -v sm_90a; SASS via
`nvdisasm`/`cuobjdump` 13.4 (wheels `nvidia-cuda-nvdisasm` / `-cuobjdump`, not in the venv).

Method: device text cut at `#include <torch/`, the two template instantiations forced with
`namespace okc { template __global__ void moe_fused<false>(Params); … <true> … }`, and dynamic smem
read straight from a probe array holding the host's own `okc_smem_bytes()` expression (so smem is
the compiler's value, not hand arithmetic). Defines are the host's exact ones: the tuple plus
`-DOKC_NS=3 -DOKC_NSF=3 -DOKC_DQ7=1` (CAS: `DQ7 = True`, no PRMTSC/FNST) and additionally
`-DOKC_FNST=4 -DOKC_PRMTSC=1` for the embedded file (`_ma_ext` adds DQ7/NS/FNST, `_moe_build`
appends `_MA_SFX`). `moe_fused<false>` is the entry the board actually runs at M = 32 (< OKC_BIGM 64);
both instantiations come out identical in regs/spills for every build below.

Registers/smem/stack are per thread / per CTA as ptxas reports them.

| build | defines (delta from MAIN) | thr | regs | spill st/ld (B) | stack | smem/CTA (B) | CTAs/SM | resident thr |
|---|---|---|---|---|---|---|---|---|
| **CAS** MAIN WGS, +`-DOKC_PRMTSC=1` (the M=17..63 build) | — | 576 | 96 | 4 / 32 | 8 | 231168 | 1 | 28.1 % |
| CAS MAIN, scales not PRMT-decodable | same source, no `sfx` | 576 | 96 | 4 / 32 | 8 | 231168 | 1 | 28.1 % |
| CAS MAIN fallback (`_main_defines(())`, mma.sync, NW=8) | CPF/CPFF/DYNK/EARLYN/EPI4/FCOSTB(B)/FINCOL | 256 | **255** | 4 / 4 (`<false>`, 280 B stack) · 8 / 8 (`<true>`) | 280 | 206464 | 1 | 12.5 % |
| CAS TINY | NW=8 TINY WS WSBLK=2 FINPF | 288 | 142 | 0 / 0 | 0 | 214784 | 1 | 14.1 % |
| CAS TINY2 | NW=8 TINY WSBLK=1 FINPF | 288 | 142 | 0 / 0 | 0 | 214784 | 1 | 14.1 % |
| CAS MID | NW=8 TINY SPLIT WSBLK=2 FINPF | 288 | 151 | 0 / 0 | 0 | 214784 | 1 | 14.1 % |
| **vend** MAIN (WGS+PRMT, host also passes FNST=4) | — | 576 | 96 | 32 / 84 | 16 | 231168 | 1 | 28.1 % |
| vend b4 (16 < M <= 32) | + `-DOKC_B4FIN=1` | 576 | 96 | 60 / 188 | 24 | 231168 | 1 | 28.1 % |
| vend MAIN fallback | CPF/CPFF/DYNK/EARLYN/EPI4/FCOSTB(B)/FINCOL/**FPF=5**, no WGS → NW=8, NT=256 | 256 | **255** | 4 / 4 (`<false>`, 296 B stack) · 8 / 8 (`<true>`) | **296** | 206464 | 1 | 12.5 % |
| vend TINY | NW=8 TINY WS WSBLK=2 | 288 | 145 | 0 / 0 | 0 | 214784 | 1 | 14.1 % |
| vend TINY2 | NW=8 TINY WSBLK=1 FINPF | 288 | 145 | 0 / 0 | 0 | 214784 | 1 | 14.1 % |
| vend MID | NW=8 TINY SPLIT WSBLK=2 FINPF | 288 | 149 | 0 / 0 | 0 | 214784 | 1 | 14.1 % |

Barriers: 16 for the two WGS builds, 3 for TINY/TINY2/MID, 1 for the fallbacks. `-DOKC_B4FIN=1` is
inert in the standalone CAS (0 occurrences of `OKC_B4FIN` in that file), so the CAS has no b4 band;
its M=17..63 slots are MAIN. PRMTSC and FNST cost nothing: CAS MAIN is 96 regs / 4+32 B with or
without `-DOKC_PRMTSC=1`, and the embedded MAIN is 96 regs / 32+84 B with or without `-DOKC_FNST=4`.
The embedded build spills 7× more than the CAS one purely because its source is 14 KB larger (the
B4FIN / FPF code paths), not because of any define.

## Occupancy

Every MoE build is **1 CTA/SM, and only 1 is arithmetically possible**: sm_90 has 65536 regs and
233472 B of shared memory per SM, and each CTA asks for 206464–231168 B, so a second CTA never fits
(2 × 206464 > 233472). The host also launches `grid = nsm` (`state["nsm"]`), and the kernel's
cross-CTA grid barrier needs every CTA resident — occupancy 2 was never an option, by construction.

| thr | regs/thread available for 1 CTA (8-reg grain) | used | headroom to the launch-failure cliff | regs/thread for 2 CTAs | gap |
|---|---|---|---|---|---|
| 576 | 112 | 96 | 16 | 56 | +40 |
| 288 | 224 | 145–151 | 73–79 | 112 | +33…+39 |
| 256 | 255 (ISA ceiling) | 255 | **0** | 128 | +127 |

So the 64-register-style boundary is not 64 here: for 576 threads a second CTA needs ≤56 regs, for
288 threads ≤112. No build is within 8 registers of either (MAIN is 40 above 56, TINY 33 above 112),
and crossing it would buy nothing because smem forbids the second CTA anyway.

## Is any build's occupancy/register choice provably wrong?

**No.** The MoE build is smem-pinned to exactly 1 CTA/SM in every band (206–231 KiB/CTA vs 228 KiB/SM)
with `grid = #SMs`, so register count only has to stay under the launch-bounds cap — MAIN sits 16
registers under it (96 vs 112) and its spills are 29 `STL`/`LDL` in total: 5 in the kernel prologue
(0x9e0–0xbb0) and 24 single `LDL`s scattered through the per-unit prologue/epilogue sections
(one every ~0x4000 B, 0x30160–0x58560), none inside an expander body (0x7570–0x51640 between two
`HGMMA.64`) — 24 local loads per 8192 weights, ~0.3 % of the dynamic instruction count.
`--maxrregcount=104/112/120` reproduces 96 regs and the same 16-byte stack frame, so the spill is
structural (one staged 16 B value reloaded per unit), not a register-budget shortfall: nothing is left
on the table for ptxas to take.
The one build at a cliff is the never-scored **fallback MAIN** (`_main_defines` when the block scales
are not PRMT-decodable): 255 registers is the ISA ceiling at 256 threads (65280/65536 = 99.6 % of the
file) and it carries a 296-byte stack frame — it has zero headroom, but it also cannot use a second
CTA (206464 B smem), so even that is not an occupancy error, only a build with no slack.

## SASS of the per-unit loop (`moe_fused<false>`, MAIN and b4)

`vend_MAIN` entry = 23520 instructions, `vend_b4` = 24912; both contain 512 `HGMMA.64`, 2048 `HMUL2.BF16_V2`,
550 `LDS`. The expander (`fat_compute`, `dq4` = one 32-bit word/lane = 8 codes × 32 lanes = 256 weights)
is fully unrolled: 16 groups of 823 instructions, each 128 `HMUL2` + 32 `dq4`.

Per `fat_compute` warp-call (16 k-steps, 8192 weights, 32 `dq4` calls), counted from the SASS:

| pipe | count | of which |
|---|---|---|
| ALU | 436 | LOP3 166, IMAD 107, SHF 72, LEA 31, IADD3 21, ISETP 19, VIADD 12 |
| heavy (FP) | 166 | **HMUL2.BF16_V2 128**, FMUL 22, F2FP 6, MUFU 4, FADD 4, FSETP 2 |
| MMA | 57 | HGMMA.64 28.5 + WARPGROUP.ARRIVE/DEPEND 28.4 |
| uniform datapath | 53 | UIADD3 33, UMOV 9, UIMAD 3.5, ULDC 1.5, USHF 1.1, ULOP3 1.3, ULEA 2 |
| LSU | 34 | LDS 17.5, STG 9, LDG 7, STS 0.5 |
| control / sync | 57 | BRA 20, BSSY+BSYNC 12, MEMBAR 8, ERRBAR+CGAERRBAR 8, SYNCS 4.7, BAR 3, NOP/WARPSYNC 1 |
| constant datapath | 13 | LDC 6.8, S2R 6.6 |
| misc | 7 | CCTL 2.5, REDG 2.5, ATOMG 0.5, VOTEU 1.5 |

Per 256 weights (one `dq4` call): **ALU 13.6, heavy 5.2, LSU 1.1, MMA 1.8**; per thread over 256
weights (32 words): ALU 436, heavy 166, LSU 34. The repo memo's "ALU 130 / heavy-FMA 53 / LSU 23 per
256 weights" does not reproduce at either unit (3.4×, 3.1×, 1.5× of it at the warp-call unit), but its
shape does: ALU:heavy = 2.45 in the memo vs **2.63 measured**, and the ALU count is dominated by the
DQ7 decode's LOP3 masks (5.2 per 256 weights) exactly as the source comment claims (7 ALU + 2 IMAD.SHL
per 8 weights, the IMADs on the FMA pipe).

## Offline-gate defect found on the way (affects `tools/cu_build.py`)

nvrtc 12.9 + `-default-device` emits an **empty PTX** (headers and a few `.global` constants, no `.entry`,
no `.func`, rc = 0, no diagnostic) for any TU that includes glibc's `<stdint.h>` — measured with a
3-line file: `#include <stdint.h>` + `__global__ void k(int*)` yields no `.entry`, deleting the include
yields `.visible .entry _Z1kIPi…`. Both MoE sources and every ATTN/bf16 source include `<stdint.h>`, so
`cu_build.py`'s "compiled, no kernel emitted (template body)" line has been vacuous for them and none of
its register/spill reporting has ever fired. Replacing the include with the plain typedefs
(`int8_t…uint64_t`, `intptr_t`, `uintptr_t`) restores codegen; that is what produced this table.
Second gate nit: nvrtc 12.9 also rejects `wgs_cta_of(S, PB, u << 4, gridDim.x)` inside `wgs_phase`
("no instance matches") because the callee is declared only later in the TU — a dependent-call lookup
quirk the judge's nvcc 13 does not have; a forward declaration makes the WGS builds compile offline.
