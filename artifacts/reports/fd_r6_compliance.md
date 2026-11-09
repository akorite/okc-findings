# R6 — Wave-file static-compliance re-scan (23 files, tools/wave_submit.sh ORDERED)

## Verdict
All 23 wave files PASS the judge's prepare-time screen. Evidence: siblings of the same skeleton already ran the real harness clean (`result.checks.ok: true`, `"rejected": []` in artifacts/bench/job_statuses/*.json — 101d3eb0fe0c49b1 probe_sweep_b6_qp16_0_1, d0823a0e3b24411c probe_klr_fci5_3), same evaluator harness/static_checks.py sha c5588b69. Our Tier-0 (src/okc_loop/verifier_static.py:97-120) is STRICTER than the judge (final_day_ledger.md:601-604 records identical-pattern klr_fci5 scoring), so local banned-pattern hits do NOT predict a prepare reject.

AGENTS.md/checks.py define no import banned set; the only written gate is TASK.md "Checked before running" (timing patches, harness frames, harness I/O/secrets, module edits, reference imports, processes, network) — none in any wave file.

## Shared skeleton hits (every file)
- `import os` / `import threading` top-level (ks_full.py:35-36 and ±30 lines per family).
- `import ctypes as _C` + `CDLL("libcublasLt.so.13")` + `CDLL("libcuda.so.1")` (ks_full.py:11922,12869,14037; siblings ±250 lines).
- `os.environ["TORCH_CUDA_ARCH_LIST"]` prepare-time arch pin (ks_full.py:3690; siblings).
- `os.environ.get("SK2_DELAY_ESC","0")` — latent env hook, dead at default (ks_full.py:7328; siblings).
- `threading.Thread` (prepare-time parallel nvcc + _rqstart daemon; ks_full.py:~11533,16588,16600; siblings).
- `import threading as _GlueContextThreading` TLS ctx-bind (ks_full.py:14030; siblings).

## Kernel-library imports — present on judge image, none unguarded-fatal
- sglang/sgl_kernel only inside `_marlin_bank`/`_marlin_experts` bodies (ks_full.py:11220-11252), same as baseline.py:41-63 which the judge itself ships.
- `from sgl_kernel.flash_attn import flash_attn_with_kvcache` inside try/except with "fallback (no sgl_kernel/sglang)" print (ks_full.py:11564-11569; every file).
- `flashinfer`/`deep_gemm`/`marlin` top-level imports: ZERO hits in all 23.

## Clean categories (zero hits)
subprocess, multiprocessing, __import__, eval(, exec(, open(, socket, urllib, inspect, frame attrs, setattr-on-modules, base64/hex/marshal/pickle payloads (all CUDA is plaintext r""" strings; only `.b64` hits are PTX `mbarrier...b64`).

## Human-review risk notes
1. `SK2_DELAY_ESC` env hook — latent; env set on judge host changes kernel behavior.
2. `OKC_ROUTEV4` env read at import (ks_full.py:12189, ks_full_t1.py:12189, ks_routev4_xqa.py:12158) — default on, judge never sets.
3. ctypes dlopen libcuda/libcublasLt + TLS primary-ctx bind inside kernel_fn — NOTE.md-documented; post-hoc screening surface.
4. Sizes ~976-977 KiB, under wave_submit's 1,000,000-byte skip guard.
