# Off-Hopper (A100) build shim for `muse_moe_w4a16`

File under test: `src/okc_loop/vendored_work/moe/moe_offarch.py` (byte copy of
`artifacts/cas/6402f5f3…b5d2b3e5.py` plus the guards listed in §1).

## Verdict

* The shim exists, is inert on sm_90a (§3) and changes nothing in the gate or lint (§3).
* **It did not make the kernel run off-Hopper.** On a Colab **A100-SXM4-40GB** the build still
  dies in `ptxas`, but on a *different* and much larger set of constructs: the TMA/mbarrier/wgmma
  data-movement layer (§4). The three constructs named in the ticket are gone from the sm_80 build
  — there is no `griddepcontrol`, `.launch_dependents` or vector-`red` error left in the log — and
  what remains is not shim-able: `cp.async.bulk` and `wgmma` have no sm_80 equivalent.
* The harness itself is fine off-Hopper: the session booted the GPU, compiled the oracle, and got
  as far as `prepare()`'s first `load_inline`. The failure is the kernel's arch floor, not the
  Colab path. **One** of the three allowed sessions was spent; see §4.4 for why more cannot help.

---

## 1. Sites guarded

The candidate carries **two** copies of the device source: `CUDA_SRC` (compiled for MAIN / TINY /
TINY2 / MAIN-fallback) and `ALT_SRCS["helper0"]` — the string assigned at module level and pulled
in by `MID_DEFINES += ("-DOKC_WSH0=1", "@helper0")`. Both had the same three unguarded constructs,
so both are guarded. Line numbers are in `moe_offarch.py`; the `base` column is the same site in
`artifacts/cas/6402f5f3…b5d2b3e5.py`.

| # | source | site (new) | base | construct | sm_90 branch | off-sm_90 branch |
|---|--------|-----------|------|-----------|--------------|------------------|
| 1 | `CUDA_SRC` | 2274–2288 | 2274–2277 | `red.global.add.v4.f32` in `wgs_seg`'s split-unit fixup (stream-K partials) | the original `red.global.add.v4.f32` inside the original `#pragma unroll` / `for` | same loop, body = 4 × `red.global.add.f32` (offsets `+0,+1,+2,+3`) |
| 2 | `CUDA_SRC` | 2431–2436 | 2420 | `griddepcontrol.wait` at the top of `moe_fused` | the original asm | nothing (comment states why) |
| 3 | `CUDA_SRC` | 2636–2640 | 2620 | `griddepcontrol.launch_dependents` before the intra-launch grid barrier | the original asm | nothing |
| 4 | `ALT_SRCS["helper0"]` | 4010–4017 | 3990–3991 | same vector `red` site in the MID source | the original literal, unchanged | same loop, 4 × scalar `red.global.add.f32` |
| 5 | `ALT_SRCS["helper0"]` | 4052–4055 | 4025 | `griddepcontrol.wait` | the original literal | nothing |
| 6 | `ALT_SRCS["helper0"]` | 4099–4102 | 4069 | `griddepcontrol.launch_dependents` | the original literal | nothing |

Each guard is exactly `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 … #else … #endif`;
`defined(__CUDA_ARCH__)` keeps the host pass (where the macro is undefined) from taking the sm_90
branch. No tile, ring depth, define, threshold or sm_90 instruction was touched.

There are no other `griddepcontrol`, `.launch_dependents` or vector-`red` sites: `grep -c` over the
base gives exactly the 3+3 above (the only other `red.…` uses, at base 2078/2131/2142/2181, are
scalar `.s32` reductions, which are sm_70-legal and were left alone).

## 2. Why the fallbacks are exact

**PDL (sites 2/3/5/6).** `PDL = False` (line 164) → `okc_set_pdl(1 if PDL else 0)` (3128) →
`g_okc_pdl = 0` (2832–2834) → the launcher takes the plain `<<<>>>` path (2892), so the
programmatic-stream-serialization attribute is never set. The source says so itself at the guard:
"without the launch attribute this returns at once". A `griddepcontrol.wait` therefore already
returns immediately, and `launch_dependents` is already a hint that nothing consumes. On sm_90
*in this configuration* the pair is a no-op too, so emitting nothing off sm_90 is not an
approximation of the sm_90 semantics — it is the same semantics. (If `PDL` were ever flipped on,
this fallback would need revisiting: PDL is a Hopper launch-API feature and the host must not set
the attribute on a device that cannot express it.)

**Vector `red` (sites 1/4).** PTX `red` with a vector type performs *four independent reductions*,
not one 16-byte atomic, so four scalar `red.global.add.f32` are the same operation. The addends are
per-lane disjoint here — the address is `fb + (slot*4+wg)*WGS_FIXW + wt*4 + nt*512`, with `wt`
distinguishing lanes — so replacing one v4 by four scalars changes neither the value added to any
address nor the order *within* a lane, which is the only order the kernel controls. Cross-CTA order
into a fixup slot was already nondeterministic under the v4 form. **Numeric risk: none beyond the
pre-existing run-to-run nondeterminism** (both forms sum the same multiset per address).

## 3. Proof the sm_90 path is untouched

**(a) Projection equality.** For each of the two sources, project the shimmed string through its
`__CUDA_ARCH__` blocks with `__CUDA_ARCH__ = 900` and compare against the base projected the same
way; and with `__CUDA_ARCH__ = 0`, assert no sm_90-only instruction survives. Result:

```
CUDA_SRC:            sm_90 projection IDENTICAL to base (2531 lines)
   arch<900 projection: instruction 'asm volatile("griddepcontrol' x0
   arch<900 projection: instruction 'red.global.add.v4' x0
   arch<900 projection: scalar red.global.add.f32 x4
ALT_SRCS[helper0]:   sm_90 projection IDENTICAL to base (2518 lines)
   arch<900 projection: instruction 'asm volatile("griddepcontrol' x0
   arch<900 projection: instruction 'red.global.add.v4' x0
   arch<900 projection: scalar red.global.add.f32 x4
```

i.e. on sm_90a the preprocessor emits **exactly** the base's token stream for both sources — the
shim only inserts preprocessor lines. Reproduce with:

```python
OURS = '#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900'
def project(src, arch):                     # strips directives, keeps the branch for `arch`
    out, stack, keep = [], [], True
    for line in src.split('\n'):
        t = line.strip()
        if t.startswith('#if'):
            stack.append((t == OURS, keep))
            keep = keep and arch >= 900 if t == OURS else keep
        elif t.startswith('#else') and stack:
            ours, outer = stack[-1]
            if ours: keep = outer and arch < 900
        elif t.startswith('#endif') and stack:
            keep = stack.pop()[1]
        elif keep: out.append(line)
    return '\n'.join(out)
# sources via ast.literal_eval of CUDA_SRC and ALT_SRCS['helper0']; assert project(new, 900) == project(base, 900)
```

**(b) `tools/cu_build.py` (sm_90a gate): identical to the base.**

```
$ PYTHONPATH=src .venv/bin/python tools/cu_build.py src/okc_loop/vendored_work/moe/moe_offarch.py
ok   CUDA_SRC [OKC_NW=16 OKC_FAST1=0 OKC_EPI4=0 OKC_WGS=1 OKC_NS=3 OKC_NSF=3]
ok   CUDA_SRC [OKC_NW=8 OKC_TINY=1 OKC_EPI4=0 OKC_FAST1=2 OKC_L2H=0 OKC_NACC=4 OKC_UWARP=0 OKC_WS=1 OKC_WSB=1 OKC_WSBL=2 OKC_WSBLK=2 OKC_FINPF=1 OKC_NS=3 OKC_NSF=3]
ok   CUDA_SRC [OKC_NW=8 OKC_TINY=1 OKC_CPF=0 OKC_EPI4=0 OKC_FAST1=2 OKC_NACC=4 OKC_UWARP=0 OKC_WS=1 OKC_WSB=1 OKC_WSBL=2 OKC_WSBLK=1 OKC_FINPF=1 OKC_NS=3 OKC_NSF=3]
ok   CUDA_SRC [OKC_NW=8 OKC_TINY=1 OKC_SPLIT=1 OKC_CPF=0 OKC_EPI4=0 OKC_FAST1=2 OKC_NACC=4 OKC_UWARP=0 OKC_WS=1 OKC_WSB=1 OKC_WSBL=2 OKC_WSBLK=2 OKC_FINPF=1 OKC_NS=3 OKC_NSF=3]
ok   CUDA_SRC [OKC_CPF=0 OKC_CPFF=0 OKC_DYNK=7 OKC_EARLYN=2 OKC_EPI4=0 OKC_FCOSTB=35 OKC_FCOSTBB=36 OKC_FINCOL=0 OKC_NS=3 OKC_NSF=3]
ok   BW_SRC [OKC_NW=16 OKC_FAST1=0 OKC_EPI4=0 OKC_WGS=1 OKC_NS=3 OKC_NSF=3] compiled, no kernel emitted (template body)
ok   BW_SRC [OKC_NW=8 OKC_TINY=1 OKC_EPI4=0 OKC_FAST1=2 OKC_L2H=0 OKC_NACC=4 OKC_UWARP=0 OKC_WS=1 OKC_WSB=1 OKC_WSBL=2 OKC_WSBLK=2 OKC_FINPF=1 OKC_NS=3 OKC_NSF=3] compiled, no kernel emitted (template body)
ok   BW_SRC [OKC_NW=8 OKC_TINY=1 OKC_CPF=0 OKC_EPI4=0 OKC_FAST1=2 OKC_NACC=4 OKC_UWARP=0 OKC_WS=1 OKC_WSB=1 OKC_WSBL=2 OKC_WSBLK=1 OKC_FINPF=1 OKC_NS=3 OKC_NSF=3] compiled, no kernel emitted (template body)
ok   BW_SRC [OKC_NW=8 OKC_TINY=1 OKC_SPLIT=1 OKC_CPF=0 OKC_EPI4=0 OKC_FAST1=2 OKC_NACC=4 OKC_UWARP=0 OKC_WS=1 OKC_WSB=1 OKC_WSBL=2 OKC_WSBLK=2 OKC_FINPF=1 OKC_NS=3 OKC_NSF=3] compiled, no kernel emitted (template body)
ok   BW_SRC [OKC_CPF=0 OKC_CPFF=0 OKC_DYNK=7 OKC_EARLYN=2 OKC_EPI4=0 OKC_FCOSTB=35 OKC_FCOSTBB=36 OKC_FINCOL=0 OKC_NS=3 OKC_NSF=3] compiled, no kernel emitted (template body)
```
10 `ok`, 0 `FAIL`, exit 0 — and the output is **byte-identical** to the base file's run
(`diff <(cu_build.py moe_offarch.py) <(cu_build.py …b5d2b3e5.py)` is empty; both exit 0 with the
same five `CUDA_SRC` define sets and five `BW_SRC` sets). Note this gate cannot see SASS for this
candidate: the `__global__` is a template instantiated by the host wrapper below the
`#include <torch/` cut, so the PTX is empty and ptxas has nothing to assemble — the gate is a
frontend check (syntax, `static_assert`s) here, which is also why it reports an identical `ok` count.

**(c) Lint: no new findings.**

```
$ … cli lint src/okc_loop/vendored_work/moe/moe_offarch.py --problem-of muse_moe_w4a16
PASS  src/okc_loop/vendored_work/moe/moe_offarch.py
1/1 clean
```
(The base is also PASS. A first draft of the `helper0` red site left an 845-char Python line and
tripped the 400-char `portability` rule — that literal was re-split; the emitted CUDA is unchanged,
which §3(a) re-verifies after the reflow.)

**(d) The fallback instruction is legal and the sm_90 branch is the v4.** Local nvrtc → ptxas on a
two-line kernel carrying both branches through the same macro condition:

```
compute_80  → entry emitted, only `red.global.add.f32`, ptxas OK
compute_90a → entry emitted, `red.global.add.v4.f32` present, ptxas OK
```

## 4. The A100 run

### 4.1 Command

```
PYTHONPATH=src .venv/bin/python -m okc_loop.cli colab-batch \
    src/okc_loop/vendored_work/moe/moe_offarch.py \
    --problem-of muse_moe_w4a16 --gpu A100 --timeout 1800
```

### 4.2 Result — `ok: false`, `stage: "prepare"`, no shapes ran

```
gpu        NVIDIA A100-SXM4-40GB
wall_s     225.8
stage      prepare
total_us   null          sol_fraction null          prepare_s null
```

`prepare()` compiles four variants. The first one (`MAIN`, `-DOKC_WGS=1`, the RS-wgmma build) fails
`nvcc`, so variants 2–4 were never attempted:

```
nvcc -O3 -gencode=arch=compute_80,code=sm_80 -DOKC_NS=1 -DOKC_NSF=1 -DOKC_NW=16 -DOKC_FAST1=0 \
     -DOKC_EPI4=0 -DOKC_WGS=1 -DOKC_PRMTSC=1 -DOKC_DQ7=1 … cuda.cu
FAILED: cuda.cuda.o   (ninja exit 255)
```

### 4.3 Exact remaining `ptxas` errors (complete inventory, 2742 error lines)

| count | error |
|------:|-------|
| 1024 | `Instruction 'wgmma.mma_async with floating point types' not supported on .target 'sm_80'` |
| 544 | `Instruction 'wgmma.wait_group' not supported on .target 'sm_80'` |
| 512 | `Instruction 'wgmma.fence' not supported on .target 'sm_80'` |
| 512 | `Instruction 'wgmma.commit_group' not supported on .target 'sm_80'` |
| 132 | `Feature 'mbarrier.try_wait.parity' requires .target sm_90 or higher` |
| 4 | `Feature 'mbarrier.arrive.expect_tx' requires .target sm_90 or higher` |
| 4 | `Feature 'cp.async.bulk' requires .target sm_90 or higher` |
| 4 | `Feature '.mbarrier::complete_tx::bytes' requires .target sm_90 or higher` |
| 2 | `Feature '.op_restrict' requires .target sm_90 or higher` |
| 2 | `Modifier '.mbarrier_init' requires .target sm_90 or higher` |
| 2 | `Feature '.cluster scope' requires .target sm_90 or higher` |
| — | **0** `griddepcontrol` / `.launch_dependents` / `red instruction with vector types` |

The last row is the positive result of this ticket: the three constructs the brief named no longer
exist in the sm_80 build. What stops it now is
`cp.async.bulk` + `mbarrier.arrive.expect_tx` + `.mbarrier::complete_tx::bytes` (the 16 KB TMA unit
ring), `mbarrier.try_wait.parity` (the sm_80 form is `test_wait`), `fence.mbarrier_init.release.cluster`,
and `wgmma` (the RS-wgmma phases). Those are the kernel's data-movement and math engines, not
stray modifiers: there is no arch-legal substitution that keeps the code shape.

Side effect to be aware of: `colab-batch` freezes a source path into the ledger before running it,
so this attempt left one new `muse_moe_w4a16` row (tag `1b74bd42900b`, tier-1 `FAIL`, no CU) for the
shimmed source. It has no measurement attached and cannot outrank a measured candidate.

### 4.4 Why no second/third session

The remaining blockers are structural, not iterative: no re-run of the same source can change which
instructions `prepare()` emits. A session that "iterates on the remaining compile errors" would have
to *replace the TMA ring and wgmma with a cp.async / mma.sync implementation* first. That is a port
with a new pipeline protocol (mbarrier accounting changes from bytes-of-tx to cp.async arrivals; the
ring's expected-arrival counts, parity phases and per-thread copy loops all move), it cannot be
compile-checked on this workstation (no `nvcc`, and the kernel template is instantiated only by the
host wrapper, so `ptxas` never sees it), and its correctness gate would be a Colab run whose failure
mode is a hang. It is also *not* inert on sm_90 by construction — it is a new code path — so it
contradicts this ticket's "build shim only" constraint and needs its own ticket.

---

## 5. What an A100 measurement can and cannot tell us

**This candidate cannot produce an A100 measurement at all** — so the honest summary for now is:
nothing measured on an A100 may be read as an H200 result, because there is nothing to measure
until the port in §4.4 exists. The table says what *would* transfer if it did.

| layer | portable to A100? | why |
|-------|-------------------|-----|
| routing (`route()`: counts, offsets, sorted `(token, slot)` list, fat/skinny job split) | **yes** | pure integer work over `topk_ids`; no arch dependency |
| cost-weighted stream-K split (`wgs_cost/wgs_prefix/wgs_range`, `skinny_split`) | **yes** (arithmetic) | plain integer math — but see SM count below |
| FP4 (NVFP4) decode: `dq4`, PRMT scale decode, LUT build | **yes** | integer/`__byte_perm`, no tensor-core feature |
| SwiGLU, `1+post_norm_w` epilogue, per-route sum of squares, final `rsqrt`/weighted sum | **yes** | plain bf16/f32 math, same instructions on both |
| `mma.sync` GEMM core (the non-WGS builds) | **yes** | sm_80-native (`m16n8k16` bf16), same instruction on sm_90 |
| graph capture / fresh-input / no-host-sync protocol | **yes** | harness-level, arch-independent |
| in-launch grid barrier (`atomicAdd` + `ld_acquire` spin) | **yes** | global atomics, works on both |
| unit ring via `cp.async.bulk` + `mbarrier.arrive.expect_tx` (16 KB TMA copies) | **no** | Hopper TMA; sm_80 has only 16 B `cp.async.cg` + different completion accounting |
| `mbarrier.try_wait.parity` | **no** (form) | sm_90 form; sm_80 must poll `mbarrier.test_wait.parity` |
| `fence.mbarrier_init.release.cluster` | **no** (scope) | `.cluster` scope is sm_90 |
| wgmma phases (`wgs_*`, RS-wgmma, 4 warpgroups × 64 rows) | **no** | no sm_80 equivalent; the mma.sync fallback build is a *different algorithm* |
| PDL (`griddepcontrol`) | **no** | Hopper launch-API feature (a no-op here, see §2) |
| ring depth (`OKC_NS/OKC_NSF`) | **no** | the host *already* drops them to 1 on `major < 9` (line 3104–3105 of the base): an A100 run is a different pipeline depth, not the same kernel slower |
| occupancy | **no** | 164 KB smem/SM on A100 vs 227 KB on H200 → the 16 KB-unit ring and the fat 256-row staging do not fit the same way |
| stream-K balance | **no** | 108 SMs (A100-SXM4-40GB) vs 132 (H200): pool-tail and split boundaries land differently |
| L2 residency | **no** | 40 MB vs 50 MB; the L2 hints and the 256 MiB cold-L2 flush are calibrated for the larger part |

So an A100 number for a hypothetical ported build could speak to **the scheduling and arithmetic
layer only** — routing, job split, stream-K balance, epilogue math, capture/freshness — and never to
the three mechanisms this kernel actually spends its time in (TMA ring overlap, wgmma issue rate,
227 KB-smem occupancy). With the ring depth also differing, per-shape µs would not be comparable to
the judge's at all; only *ratios between two candidates measured the same way on the same A100* would
carry information. That is the ceiling: a coherence/regression instrument for the portable half of
the kernel, never a performance proxy.

The one thing the A100 session does confirm cheaply is that `colab-batch --gpu A100` works end to end
on our side (session boot, GPU identification, oracle install, `prepare()` reached) — the instrument
is sound; this candidate is simply below its arch floor.
