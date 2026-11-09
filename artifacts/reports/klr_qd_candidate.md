# klr_qd: keell0renz board layer + 3 proven flags

Candidate: `src/okc_loop/vendored_work/layer/klr_qd.py` (16349 lines).
Base: `src/okc_loop/vendored/keell0renz/muse_decode_layer.py`, sha256
`82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`, check `7cd40fdd1a934e27` = **2.247521**.
Line 1 (`# okc: problem=muse_decode_layer`) is untouched; the provenance header sits directly under it
(static contract `verifier_static.HEADER_RE` requires the tag on line 1, so it cannot be prepended above it).

## Diff (everything that is not the 4 header comment lines)

```
-#define OKC_PDLKV 0     // PDL launches: K/V issued before griddepcontrol.wait ...
+#define OKC_PDLKV 1     // PDL launches: K/V issued before griddepcontrol.wait ...
-  ... extra_cuda_cflags=["-DOKC_QDIRECT=0", "-DOKC_PK2=0", "-DOKC_W0ENABLE=1", "-O3", f"-DOKC_SM90={...}", "--expt-relaxed-constexpr"]
+  ... extra_cuda_cflags=["-DOKC_QDIRECT=1", "-DOKC_PK2=1", "-DOKC_W0ENABLE=1", "-O3", f"-DOKC_SM90={...}", "--expt-relaxed-constexpr"]
```
(lines 182 and 3740). Verified with `diff -u` against the board file: no other byte differs.

## Gates

`ast.parse` — OK. Problem tag — line 1 `# okc: problem=muse_decode_layer`; `okc lint` identifies the file as
`tier0 muse_decode_layer`.

`PYTHONPATH=src .venv/bin/python -m okc_loop.cli lint src/okc_loop/vendored_work/layer/klr_qd.py` — **FAIL**,
30 tier-0 errors. These are **pre-existing**: the unmodified vendored board file fails with the identical finding
set (normalized diff of the two lint outputs differs only in the file path; 17 lines each, same counts, same
text). The board file scored 2.247521 on the judge in this state, so the judge tolerates all of them.

The 30 errors are 8 distinct constructs, all in harness/dev/build-setup code, none in the kernel path:
`threading` imports (71, 13413, 15992), `os.environ` (3738 TORCH_CUDA_ARCH_LIST; 7211 SK2_DELAY_ESC debug knob),
`globals()`/`locals()` feature probes (6972, 6978, 10227, 11127, 16172, 16260, 16284), `ctypes` (11654),
`time.time` (23 x, 14350…16150), the 140 927-char `_MA_ALT_SRCS["moe_helper0"]` literal (12587, limit 400),
and the value-dependent branch in `_glue_bind_context` (13447).
Our fork `vendored_work/layer/muse_decode_layer.py` **PASSES** lint, i.e. every rule has a known-good fix already
in the repo: it replaced 3738's `os.environ` with a `-gencode=arch=compute_{cap},code=sm_{cap}` flag (its line
3937) and deleted the threading/ctypes/time.time/globals/locals/dev-knob sites. `vendored_work/layer/klr_layer_clean.py`
still fails with the same 30.

`TMPDIR=… .venv/bin/python tools/cu_build.py src/okc_loop/vendored_work/layer/klr_qd.py --variant MID` — 2 ok,
9 FAIL, exit 1. The 9 FAILs are identical (same 9 sources, same messages) on the unmodified board file, i.e.
tooling limits, not this change: `bfp_TMA/GEMM/SK2/WIDE/GQ_SRC` (Python-side `@@...@@` placeholders it must not
guess, 14 errors), `_GRAFT_GLUE_CUDA` / `_SL_GLUE_CUDA` (`nv/target` not on the nvrtc include path),
`_GRAFT_B2_CUDA` (`tvm/ffi/container/tensor.h` absent offline), `_ROUTER_FP32_CUDA` (`cudaGetLastError` needs the
runtime header). The two that matter:
* `ok ATT7_CUDA_SRC` — the kernel whose flags changed: compiles clean (reported "no kernel emitted (template
  body)", the documented limit: the `attn_kernel<>` instantiations live below the `<torch/` cut, so nvrtc cannot
  instantiate them and no register/spill count is produced — the `if constexpr (QDIR)` / `IDXF` bodies are
  therefore not code-generated offline),
* `ok MOE_CUDA_SRC`.
Note `cu_build` does not read inline `extra_cuda_cflags`; the ATT7 pass above is with the source defaults plus the
define set it infers. Re-running both files with `-D OKC_QDIRECT=1 -D OKC_PK2=1` gives the identical result
(2 ok / 9 FAIL on each), so the flag pair itself is not what fails.

## Why PDLKV=1 is the low-risk one of the three

klr's comment at the macro: "PDL launches: K/V issued before griddepcontrol.wait, q / new k,v as direct loads
after it (decode_layer: build with 1; standalone it costs ~0.3% from code layout)". This file *is* the decode
layer and runs `PDL = True` for T <= 128 (line 97), so the comment applies — but his own build line never passes
it, so **no file in this repo has ever been built with OKC_PDLKV=1**: the 2.247521 came from 0. It is a
compile-time prologue ordering with exactly one consumer, `IDXF` (line 1519: `OKC_IDXFIRST && !WS &&
!Cfg<L,FLAT>::S2 && !OKC_PDLKV`), whose only other uses are `cp_wait<IDXF?1:0>` (1571) and one
`if constexpr (IDXF)` block (1633) — both orders are correct (every qkv read is preceded by `pdl_wait()` at 1532).

Blast radius: `OKC_WS=1` (+`OKC_WS8=1`) makes `WS` true for every flat kernel and `OKC_S2=0` leaves `S2` false for
per-request ones, so PDLKV=1 changes only the **per-request** kernels — the sliding-window arms. From the saved
profile of 82c098ae that is 4.3% of weighted layer time (see below); the dominant flat kernels
(`attn_kernel<1,true,…>`, 40.7-51.7% of the full-layer shapes) are untouched. Worst case even a 3% ATT7 swing
there is 0.13% of the layer, and the revert is one character.

Rejected alternative: passing `-DOKC_PDLKV=1` at the build site instead of editing the `#define` — the in-source
`#define OKC_PDLKV 0` would redefine the `-D` and win (with a warning), so the macro itself had to change.

## Expected gain

Weighted attention share per load, from `artifacts/cas/profiles/82c098ae….7cd40fdd1a934e27.json` (each load's
two cases weighted 46 sliding / 16 full; ATT7 = `attn_kernel<…>` GPU time / total):

| load | sliding | full | weighted attn share |
|---|---|---|---|
| decode 1 | 5.3/78.0 = 6.8% | 11.7/83.5 = 14.0% | 8.66% |
| decode 16 | 7.6/143.2 = 5.3% | 90.0/221.8 = 40.6% | 14.41% |
| decode 32 | 9.3/150.0 = 6.2% | 170.3/329.5 = 51.7% | 17.94% |
| verify 1 | 7.2/110.9 = 6.5% | 14.8/125.8 = 11.7% | 7.85% |
| verify 16 | 12.0/251.7 = 4.8% | 108.1/328.7 = 32.9% | 12.02% |
| verify 32 | 15.5/309.2 = 5.0% | 190.4/467.6 = 40.7% | 14.23% |
| **mean** | | | **12.52%** |

The six loads own 1/6 of the log score each, so a layer gain is the mean over loads of (attn share x ATT7 gain).
`QDIRECT` is gated on `L == 8` (line 1517), i.e. only the verify loads: 11.37% weighted. `PK2` and `PDLKV` hit
every shape (12.52%) / the sliding arms only (4.28%).
Fork evidence is +1.55% on ATT7 (2.6497 vs klr's 2.6089 on the attention board) but that figure also carries the
fork's `FPART` term and its missing L2-prefetch block, so QDIRECT+PK2 alone is 1-1.5%.

**Expected layer gain: +0.2% (range +0.1 to +0.4%)** — 2.247521 -> ~2.252; PDLKV adds +-0.04%.

## For Main

1. This candidate is **not lint-clean** and cannot be without the harness clean-up (8 constructs above). It is
   usable only on the judge's actual tolerance, which the board file proved. If the check budget can only be spent
   once, prefer applying the same three flags to the **lint-clean fork**, which already has QDIRECT=1/PK2=1 at its
   line 3941 and passes lint — my candidate's real delta over the fork is klr's newer MoE lineage
   (cepil U=160/140, MFIN/B4L, SC6W, WGSFCI 4, DF patch) plus his PRF32 L2 prefetch and the
   `pdl_launch_dependents` placement.
2. PDLKV=1 is the one of the three with no empirical backing in this repo; drop it first if the candidate must be
   flattened to the two board-proven flags (revert = `#define OKC_PDLKV 0` on line 182).
3. ATT7's register/spill count is still unverified offline (template body); a real build is the only check.
