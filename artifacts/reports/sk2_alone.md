# sk2 dense geometry, ALONE on the FCI5 base (`klr_sk2.py`)

Candidate: `src/okc_loop/vendored_work/layer/klr_sk2.py`, sha `4c70405d6c10`, 984,011 bytes,
line 1 = `# okc: problem=muse_decode_layer`. Built as `klr_fci5.py` (sha `ee74c9e4`) + the sk2
geometry mechanism of `klr_drate.py` and nothing else. Nothing was submitted.

## 1. Hunks taken from the fci5 -> drate diff (4 edit regions, 29 hunks total in drate)

| drate hunk | content | taken? |
|---|---|---|
| `@-9536` | `bfp_SK2_CONFIGS += qkv32c2 / un32c2_r144 / o16h_r100` + `bfp_SK2_ALT` map | **yes** |
| `@-9598` | `bfp__sk2_src` instantiation gate: `name in bfp_SK2_ALT[(layer,tokr)]` also compiles | **yes** |
| `@-9737` | end of `bfp_sk2_prepare` calls `bfp__sk2_bake`; `bfp_SK2_MARGIN`; `bfp__sk2_bake`; `bfp__run_sk2(nm=)` | **yes** |
| `@-13395` | `_de` shim `bfp__run_sk2` passes `nm` through (`_DE_RUN(...,dg,dbg,nm)`) | **yes** |
| `@-5596` | `OKC_WGSFCI 5 -> 4` | no (that is fci5's win, drate is stale) |
| `@-8968`, `@-9042`, `@-9060`, `@-9793`, `@-9871`, `@-9895`, `@-9930`, `@-10007`, `@-10101`, `@-10149`, `@-10174`, `@-11119`, `@-12707`, `@-13477` | the `pr` fused pooler+router GEMM (rows entry, CONFIGS, route, FP32 tolerance, `_bfp_pr_ok`, `bfpp` in `_tune`) | no |
| `@-9042`(2nd), `@-9060`(2nd), `@-10037`, `@-10101`, `@-10149` | `bfp_GEMM_ALT` / `bfp__gemm_items` / `cfg=` in `bfp__run_gemm` / `TUNE` print / `tune_log` init | no |
| `@-14650`, `@-15975`, `@-16074`, `@-16090` | rq `@KS` split-K probes (`_ksof`, `_tag`, wider `RQBAKE_L`) | no |
| `@-10037` | `state.get("disabled", ())` guard in `bfp__candidates` | no |

Only `bfp__ctas` was trimmed: fci5 gets the sk2 branch only (the gemm branch exists to print
`bfp_GEMM_ALT` candidates, which are not in this file). Everything else is verbatim drate — checked by
extracting the blocks from both files and comparing (ALT cfgs+map IDENTICAL, bake IDENTICAL, shim head IDENTICAL).

## 2. Binding audit

`diff -u klr_fci5.py klr_sk2.py` = **4 hunks**, all outside every `def prepare(` body:
`@-9536` configs, `@-9598` src gate, `@-9737` `bfp_sk2_prepare` tail + two new defs before
`bfp__run_sk2`, `@-13395` the `_de` shim. The 13 `def prepare(` definitions are byte-identical to
fci5 (the hunk at 13470 sits between `prepare` and the separate `bfp__run_sk2` def, not inside it).
`weights["qkv"]` appears **0** times; the only qkv bindings are `weights["qkv_gate"]` (9856, 11277)
and `st["qkv"] = weights["qkv_gate"].contiguous()` (11283) — unchanged from fci5. The bake writes
only `bfp_SK2_PICK`, never `weights`.

Mechanism (what the bake compares, computed from `bfp__sk2_geom`): every alternative doubles the CTAs
at the same stage count, i.e. it is a pure grid-shape change.

| cell | pick (CTAs) | alternative (CTAs) | nstage pick/alt |
|---|---|---|---|
| o t16 | `o8h_r88` 256c | `o16h_r100` 512c | 4 / 4 |
| o t32 | `o32h_r100` 256c | `o16h_r100` 512c | 3 / 4 |
| qkv_gate t16 | `qkv16_r176` 96c | `qkv32c2` 192c | 5 / 5 |
| qkv_gate t32 | `qkv32_r208` 96c | `qkv32c2` 192c | 4 / 3 |
| unpooler t32 | `unpooler32_r144` 128c | `un32c2_r144` 256c | 3 / 3 |

## 3. Gates

- `ast.parse` ok; 984,011 bytes (< 1,000,000); line 1 tag ok.
- ruff delta vs fci5: 180 -> 182 errors, +2 `UP031` at 9811 (the bake's own `%` log line, same style as
  the 45 pre-existing `UP031`). No new rule class.
- sm_90a compile gate on A100, 92.3 s wall: `bfp__get_ext` **ok (55.2 s)** — this is the module that now
  contains the 5 extra `launch_sk2<...>` instantiations (CL=2 / o16h) — plus `_moe_mid 76.8 ok`,
  `_moe_tiny 76.9 ok`, `_moe_tiny2 77.0 ok`, `_moe 91.3 ok`, `_att7 92.3 ok`, `_moe(32) ok`.
  `bad` = only `_graft_glue_module`, `_graft_b2_module`, `_sl_cluster_module` (no `tvm_ffi` on Colab, the
  documented environment limit, present for fci5 too).

## 4. Judge checks (2 used, no submit)

| check | job | geomean | value | vs board 2.28370 |
|---|---|---|---|---|
| 1 | `e969350ba6d24d8d` | 2.2471 | 10,066.5 us | -1.60 % |
| 2 | `863fdc486f504a4b` | `duplicate` of check 1 | - | no sample |

Check 2 returned `duplicate_of=e969350ba6d24d8d`: **the judge dedups identical bytes**, so the log mine's
"a sub-1 % win needs a second check" cannot be done on the same file — it needs a byte-different twin
(like `klr_glue`/`klr_fci5`). A same-day re-measure of the incumbent is equally impossible.

**Why the -1.6 % is node drift, not the change.** The MoE is untouched by this candidate, and its
self-tune in this very job reads `T=32 145.3 / T=128 191.6 / T=256 223.2 us` against the reference
`189.9 / 220.9 us` at T=128/256 — i.e. **~0.9-1.0 % slower on this node at unchanged code**, with the
twin ratio unchanged at 1.0309/1.0157 (SC6 intact).

Per-load ratios, check 1 vs the PB row the gate compares against (82c098ae, 2.2474) and vs fci5's own
check `a407ad5e` (2.2730):

| load | PB 82c098ae | fci5 check | sk2 check 1 | vs fci5 |
|---|---|---|---|---|
| decode 1 | 2.5604 | 2.5604 | 2.5274 | -1.29 % |
| decode 16 | 2.1048 | 2.1312 | 2.0966 | -1.62 % |
| decode 32 | 1.9217 | 1.9716 | 1.9407 | -1.57 % |
| verify 1 | 2.4781 | 2.4961 | 2.4593 | -1.47 % |
| verify 16 | 2.2077 | 2.2322 | 2.1939 | -1.72 % |
| verify 32 | 2.2750 | 2.3003 | 2.3202 | +0.86 % |

Gate verdict from `ingest_check`: `allowed=false, failed=gain` (2.2471 < 2.2474, the ledger PB) — i.e. the
candidate is not a submit, and the loss is ~equal to the PB and ~1.5 % behind fci5's own measurement.

## 5. What the bake actually picked in the judge

`SK2` lines from job `e969350ba6d24d8d`'s `result.log_tail` — **no cell was re-picked**, all five keep the
shipped geometry (alt in us, CTAs from `bfp__ctas`):

```
SK2 o         t16 o8h_r88:9.22us/256c o16h_r100:11.49us/512c -> o8h_r88
SK2 o         t32 o32h_r100:9.47us/256c o16h_r100:12.19us/512c -> o32h_r100
SK2 qkv_gate  t16 qkv16_r176:13.76us/96c qkv32c2:17.50us/192c -> qkv16_r176
SK2 qkv_gate  t32 qkv32_r208:14.78us/96c qkv32c2:20.32us/192c -> qkv32_r208
SK2 unpooler  t32 unpooler32_r144:18.14us/128c un32c2_r144:22.91us/256c -> unpooler32_r144
```

merged3 (in the bundle) read 9.18/9.41/13.50/14.37/18.11 for the same picks, so the numbers reproduce to
0.4-3 % — the alternatives lose by **2.0-5.5 us in every cell**, they never win. The lead is closed: the
"2-6 us per cell" in the log mine is the *pick's* margin over its alternative, not a gain available.
`prepare_s = 141.5 s` with the bake in (the bake itself is ~5 x 3 graphs, a few ms).

Consequence: the hot path is byte-identical to fci5 (same `ci`, same stream/table, same dispatch), so sk2
alone can only differ from fci5 by machine drift — up to ~1.5 % on 5 of 6 loads in check 1, and the MoE
cross-check above shows this node is ~1 % slow. **sk2 dense geometry is inert: it buys nothing, costs
nothing in the timed path (~5 ms of prepare), and its own gate keeps every shipped pick every run.** The
"never tested alone" lead is now closed — do not spend another check on it, and do not carry it into a
bundle (it re-instantiated 5 extra `launch_sk2` templates and ~4 ms of prepare for zero win).
