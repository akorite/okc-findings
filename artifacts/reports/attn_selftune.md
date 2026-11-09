# attnst: the two ATT7 finalize flags as one measured per-shape choice

File: `src/okc_loop/vendored_work/layer/klr_attnst.py` (copy of `klr_qd3.py` + appended block, nothing above
the append changed except the pieces listed under *Edits*). Base = board 82c098ae + ATT7
`-DOKC_QDIRECT=1 -DOKC_PK2=1` + the `rq` 12-bit codec + the MoE build self-tune.

## What the launch actually chooses

`_launch7` (11339) builds the plan with `_plan7`, applies `_TUNED7[(B, L, window, pairs)]`, then derives two
bits that reach the kernel: `coop` (bit 1) and `w0`, plus `flat`/`n_splits`/`cs`/`nc` as grid arguments.
`_decode_attention7` (11402) is the only entry the row uses for all 12 scored shapes
(`decode/verify {1,16,32} x {sliding,full}`, page 1, ctx 80000, window 2048), so all of them pass through
`_launch7` and therefore through the pick.

Which sides of the two flags each scored shape can reach — computed from `_plan7` + `_TUNED7` + the `coop`
default in `_launch7` with 132 SMs (verified by exec'ing the file's own `_plan7` against stubs):

| shape | plan | coop | `OKC_FINWARM` gate `!FLAT && cs<=1 && !coop && nc==1` | `OKC_FLATCOOP1C` reachable |
|---|---|---|---|---|
| b1 L1 sliding | 32 splits, nc=1 | 0 | **yes** | no (not flat) |
| b16 L1 sliding | 16 splits, nc=1 | 0 | **yes** | no |
| b32 L1 sliding | 8 splits, nc=1 | 0 | **yes** | no |
| b1/b16/b32 L8 sliding | `_TUNED7` b32 → 4 splits coop=1; others coop=1 | 1 | no | no |
| b1 L1 full | `_TUNED7` → 252 splits, nc=8, coop=1 | 1 | no (nc, coop) | no (not flat) |
| b16/b32 L1 full | flat, 264 CTAs, nc=1 | 0→1 | no (FLAT) | **yes** |
| b1 L8 full | 252 splits, nc=8 | 1 | no | no |
| b16/b32 L8 full | flat, 264 CTAs | 1 | no (FLAT) | already on |

So the variant build can only ever change five shapes; the other seven are compiled with the flags but stay
bit-identical at runtime (the gates compile out). That is why the sweep times all 12 anyway: it also reports
whether the two extra compile-time constants cost anything through code layout.

## How the pick is expressed

* `_att7()` / `_att7v()` are two `load_inline` builds of the same source, module names
  `muse_att_v298w1pf_<arch>` (unchanged, so an existing cached build is reused) and
  `..._<arch>_fw1c`. `OKC_FINWARM`/`OKC_FLATCOOP1C` are plain `#define`s in the source, so a `-D` would be
  silently overridden by the later definition — the variant patches the two lines in its own source string
  and asserts each occurs exactly once, instead of adding them to `extra_cuda_cflags`.
* `_att7v_opt()` wraps the build so a failed second compile prints one line and leaves base in place;
  `prepare` raises whatever its fan-out hits, and a measurement must never fail a run.
* `_ATT7_PICK[(batch, q_len, window)] = (module, launcher override)` (the window value, not just
  sliding/full, so a shape whose window differs from the sweep's 2048 keeps the base build); the `_launch7` wrapper (11419) reads it
  only when the caller passed neither `_mod` nor an override, so production and the tuner's base pass share
  one code path. The variant's *only* launcher difference is `coop=1` on the flat `L == 1` shapes, i.e. the
  host gate widened from `L == 8` to `L in (1, 8)`; `w0` is `OKC_W0_L1` for `L == 1` flat either way, and the
  kernel re-checks residency (`_launch7` host side at 3557), clearing `coop` if the grid does not fit.
* Empty dict = base build, so every shape the tuner does not cover (correctness cases) is unchanged.

## Tuner

Appended `prepare` wrapper runs after every other prepare and calls `_attn_finalize_tune`:

* per shape: synthetic inputs from `_att_shape_inputs` (same construction as `_attn_selftest`, page 1, ctx
  80000 full / 2048 sliding, a sliding layer only needs window + q_len - 1 keys, finite random e4m3 cache),
  replayed through the real `_decode_attention7` so `_PF7`, the plan and the PDL switch are production's;
* 2 CUDA graphs per shape, cold-L2 read sweep before every replay, 16 interleaved rounds (the file's own
  `_selftune_time`);
* keep the variant only if all three hold — median faster by `max(3%, 2 us)`, >= 75% round wins, and
  `rel-L2(variant, base) <= 1e-6` on the attention output (`_att_pick_ok`, unit-tested);
* per-shape `try`, whole pass `try`: any failure pops the key, prints `pick=base FAILED: ...` and the run
  continues on the base build. `_check_guards()` after the pass inspects the workspace sentinels the variant's
  finalize wrote.

Printed per shape (one line, 12 lines):

```
[att-selftune] 32/1/sliding pick=var med=[8.3,7.1] rel=0.0e+00 wins=16/16
[att-selftune] 16/1/full    pick=base med=[90.4,91.0] rel=3.1e-07 wins=5/16
[att-selftune] 1/8/full     pick=base FAILED: RuntimeError('...')      # only if a shape raises
[att-selftune] no variant build: base everywhere                        # only if the second nvcc failed
```

Reading it: `pick=var` means the variant is in use for that shape this run and was faster by >3%/>2 us with
matching output; `pick=base` with a `rel` around 1e-7 means it was numerically fine but not faster — the flag
is a loss for that shape, which is the answer the check was bought for. `rel` above 1e-6 with a large speedup
means the faster path drifts and was refused. `wins=k/16` is the round count, so a 3% median with 7/16 wins
is noise, not a win.

## Cost

* nvcc: one extra ATT7 translation unit added to prepare's existing 10-way `threading` fan-out (11278), so it
  overlaps the builds already running there (~85 s of nvcc wall in total before). With >= 12 cores the added
  wall time is ~0; on an 8-core box the tail can grow by up to one ATT7 compile (order 30-60 s). Either way
  far inside the 300 s prepare.
* GPU: 24 graph captures and 384 replays, 12 input sets built and dropped per shape; ~0.3-0.8 s of wall
  (~2 ms of flush + <=7 ms of replays per shape, the rest is filling up to 330 MB of fp8 cache per full-layer
  shape). Peak transient ~1.7 GB on the b32 full shapes: the allocations are sequential and freed per shape.

## Gates run here

* `ast` parse: ok.
* `okc_loop.cli lint`: 74 findings, byte-identical to `klr_qd3.py`'s set; only the file name and the number
  inside one message (`line 12586` -> `12625`) differ. Zero new findings.
* `tools/cu_build.py --variant MID|MAIN`: identical status lines (`ok`/`FAIL` per embedded source, including
  `ok ATT7_CUDA_SRC ... compiled, no kernel emitted (template body)`) and identical diagnostic sets. This
  container has no `nv/target`, `cuda/std/type_traits`, `tvm/ffi`, and `cu_build` stops ATT7 at the
  `<torch/` cut, so the two patched `#define`s and both builds' kernel bodies are **not** compiled here:
  the only thing verified about them is that each anchor string occurs exactly once (assert at build time,
  mirrored in `tests/test_attn_finalize_tune.py`).
* `tests/test_attn_finalize_tune.py`: 2 passed (gate arithmetic; build plumbing: anchors unique, `-D`-free
  patch, distinct module name, optional build wired into the fan-out).
