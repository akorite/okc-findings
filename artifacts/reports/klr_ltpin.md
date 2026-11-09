# cuBLASLt plan pins: do they transfer into the board file?

Verdict: **no**. No `klr_ltpin.py` produced. Both T=256 cells the fork pins are already covered by the
board file's own Lt path, and one of them by a strictly newer path ahead of it.

## The fork's pins (src/okc_loop/vendored_work/layer/muse_decode_layer.py:114)

```
_OKC_LT_PINS = {
    ("qkv", 256): (14, (66, 124, 1, 0, 0, 0, 35, 0, 2)),   # C/D bf16; [256, 8192] @ [8192, 2304]
    ("o", 256): (0, (66, 20, 1, 0, 0, 6, 35, 0, 2)),       # C/D fp32; [256, 1024] @ [1024, 8192]
}
```
cfg word order = (algoId, tile, splitK, reduction, swizzle, custom, stages, [inner unused], cluster).
`_CFG_WORDS` maps them to cublasLt algo attributes 1,2,3,4,5,6,8; the algo is built with
`cublasLtMatmulAlgoInit(68, 0, 14, 14, cdtype, cdtype, algoId)` and then `AlgoCheck`.

## What the board file already runs at T=256 (src/okc_loop/vendored/keell0renz/muse_decode_layer.py)

`_GRAFT_LT_T = (128, 256)` (11658); `prepare()` runs `_graft_lt_plan(st)` (12572), and a later
`prepare()` forces `st["mm_pick"][(T, name)] = True` for `("qkv","o","pr","un")` at T=128,256 (12708-12714)
with the comment "The graft's fixed Lt plans handle these shapes ... qkv must be True for the graft's
direct Lt/attention path." The compose policy at 12720-12759 only compares glue/boundary variants and
never drops these plan entries (`modes` all carry `drops=()`).

Pick table quoted (12313-12332):

```
_GRAFT_LT_PICKS = {
    ("pr", 256): ('tn', 2, True, [...]),
    ("pr", 128): ('tn', 1, True, [...]),
    ("o", 256):  ('nn', 2, False, [85899345986, ..., 1407374883553286, ...]),   # bf16 out, batch 2
    ("un", 256): ('nn', 2, False, [...]),
    ("un", 128): ('nn', 2, False, [...]),
    ("qkv",128): ('tn',16, False, [395136991298, ..., 1407374883553282, ...]),
    ("o", 128):  ('nn', 8, False, [77309411394,  ..., 1407374883553283, ...]),
}
_GRAFT_QKV256 = {"id": 66, "tile": 124, "stages": 35, "splitk": 1, "red": 0, "swz": 0, "custom": 0, "cluster": 2}
_GRAFT_QKV256_ATTRS = (("tile",1),("splitk",2),("red",3),("swz",4),("custom",5),("stages",6),("cluster",8))
```

* **qkv 256 — duplicate, not an addition.** `_GRAFT_QKV256` + `_GRAFT_QKV256_ATTRS` is the *same*
  attribute set as the fork's pin, applied through the same `AlgoInit(68,0,14,14,14,14,66)` +
  `ConfigSetAttribute` + `AlgoCheck` sequence. Descriptors match too: `_graft_layouts(lt_tn, 256, n, k,
  "tn", 1, False)` builds A=(k,n,k) dtype 14, B=(k,m,k) dtype 14, C=(n,m,n) dtype 14 for the `st["qkv"]`
  weight [2304, 8192], i.e. `[256,8192]@[8192,2304]` bf16-out — identical to `("qkv",256)` in the fork
  apart from the count=1 / stride batch attributes (no-ops at batch 1). Same algo, same tile 124, same
  stage 35, same cluster 2 => the same kernel launch. Adding the pin would duplicate the config and buy
  nothing but a second handle and two more prepare-time AlgoCheck/guard runs.
* **o 256 — different cell.** The board's o GEMM writes **bf16** (`('nn', 2, False, ...)`; `_graft_lt_run`
  allocates `torch.bfloat16` when `fp32` is False) in `nn` mode batched over n/2=4096. The fork's o pin is
  built with cdtype 0 = **fp32** C/D for a single 8192-wide n. Different C/D dtype and different layout
  => the fork's measured fp32 cell does not transfer, and the board's own pick (decoded words:
  algoId 66, tile 0x14=20, stages 0x23=35, splitK 1, custom 6) already uses the same algo 66 / tile 20 /
  custom 6 kernel family, from a measurement taken in this pipeline (`dl_lt_kdl_finish c5e0be33`, gpu03 CUPTI).

## Two more reasons a pin could not win here

1. `("qkv",256)` has a *newer* path ahead of the Lt plan: `_rq_gemm` / `RQ_USE = {('un',128), ('qkv',256),
   ('qkv',128)}` (16209); `kernel_fn` runs `_rq_gemm` first and only falls through to `_graft_lt_run`
   (12535-12537). A pin would sit behind two measured layers.
2. Guards can only lose, never add: the fork's plan requires 0 B workspace, the graft's AlgoCheck accepts
   up to 64 MiB. If graft's AlgoCheck refused a descriptor, the fork's would refuse it too.
   The only cell where the fork's pin would still run is the fallback where `att_ok`/`_supported7`/
   `_att_fits7` fail (12533) — i.e. when the attention build is unavailable, a state in which a 1.5% GEMM
   gain is irrelevant.

## Files touched

None. Nothing added to the board file, nothing compiled.
