# attn_cores — the layer vs the three attention-side facts (co-residency, pre-wait stages, coop launch)

File audited (untouched): `src/okc_loop/vendored/keell0renz/muse_decode_layer.py`, sha `82c098ae`, 16345 lines.
Line refs below are that file unless prefixed. No judge command run; no file produced (nothing qualifies
as a one-liner — see "Change" per section).

Ground truth for the in-layer attention cells: judge profile of this exact candidate
(`artifacts/cas/profiles/82c098ae…7cd40fdd1a934e27.json`, 12/12 loads), kernel `attn_kernel<...>` GPU time
per replay:

| load | attn kernel | replay total (µs) | attn share |
| --- | --- | --- | --- |
| decode 1 sliding | 5.3 µs | 78.0 | 6.8% |
| decode 1 full | 11.7 | ~84 | 14.0% |
| decode 16 sliding | 7.6 | ~143 | 5.3% |
| decode 16 full | 90.0 | 221.7 | 40.6% |
| decode 32 sliding | 9.3 | 150.0 | 6.2% |
| decode 32 full | 170.3 | 329.3 | 51.7% |
| verify 1 sliding | 7.2 | 110.8 | 6.5% |
| verify 1 full | 14.8 | 126.5 | 11.7% |
| verify 16 sliding | 12.0 | 250.0 | 4.8% |
| verify 16 full | 108.1 | 328.6 | 32.9% |
| verify 32 sliding | 15.5 | 310.0 | 5.0% |
| verify 32 full | 190.4 | 467.8 | 40.7% |

(1 µs on `verify 32` = 1.9 score-µs; 1 µs on `decode 1` = 5.17× that, per PROGRESS:119-121.)

## 1. Attention co-resident with a GEMM — we are on the correct side, 0 µs to recover

**What we do.** One stream for every launch: the ATen current stream for ATT7 (`3690`) and for every other
kernel (`6839`, `8614`, `9295`, …). The only `torch.cuda.Stream()` uses are the offline tuner's
microbenchmarks (`14596`, `16297`), not the timed forward. The only cross-kernel overlap is PDL:
`PDL = True`, `PDL_MAX_T = 128` (`101-103`), attention gets `pdl={"pdl": _pdl(B*L)}` (`11384`), so **T=8 and
T=128 launch with the programmatic-dependency attribute and T=256 does not**.

Per verify load (`_plan7`): v1s 33 splits × 2 pairs = 66 CTAs (window 2048 → 33 tiles, `3826-3830`),
v1f 126×2 = 252 (`3813-3815`), v16s 4×16×2 = 128 (`3836`, `sms//reqs`), v16f 132 flat (`3795-3803`),
v32s 4×32×2 = 256 (`3836`, `264//64`), v32f 132 flat. Occupancy floor is 2 CTAs/SM (`Cfg::MINB = 2`, `287`)
= 264 slots, and the coop plans are host-clamped to that (`3557`, `3575`).

**Mechanism.** PDL never preempts: a dependent grid's CTAs are placed only on resources the producer has
already released, so a co-resident kernel can only *fill holes* (66-132 of the 264 slots at v1s/v16s/v16f/v32f),
never displace attention's CTAs. The producer window is the qkv GEMM's drain, and ATT7's
`pdl_wait()` sits at `1532`, before its first read of `p.qkv` — the pre-wait work is page indices only
(`1521-1526`). `pdl_launch_dependents()` at `1533` lets the *successor* (o-projection) begin early, which is
the win the attribute exists for. At **v32 (T=256) there is no overlap at all** — the fact's noisy case is
already disabled there; `102` records the measurement that produced that cut ("a launch-latency win at decode
sizes, a ~1% loss at T = 256"), and PROGRESS:147 lists `PDL_MAX_T 32→128` as measured and closed.

**Measurement check.** The fact's un-slowed v32 attention is 14.9 µs and its GEMM-resident figure 22-27 µs.
Our in-layer v32s attention is **15.5 µs** (and v32f is at 3.4 TB/s = the KV-byte floor, 1.03-1.25× per
PROGRESS:128-130) — i.e. we sit at the un-slowed number, not the slowed one. Nothing in our verify path runs
attention beside a kernel that holds its SMs.

**Worth.** 0 µs/replay as it stands. The exposure the fact describes would be 7-12 µs on `verify 32`
(2.3-3.9% of its 310 µs replay ≈ 0.4-0.65% of the layer score) and 7-12 µs on `verify 16 full`
(2-4% of that load). We do not pay it. **Change: none** — "moving a kernel off the same SMs" has no
target: the co-resident kernels are dependents filling idle slots, and the one load where a large GEMM
could overlap (v32) already has PDL off.

## 2. Pre-wait stage count — dense/glue is at exactly 2 (the optimum); attention is at 0 K/V stages and its switch is dead; 0 µs

**Dense / glue = 2, exactly the measured optimum.** `BFP_PDL_N = 2` — "weight stages issued before
`griddepcontrol.wait`" (`10155`); `PCX_CFG = {}` (`10158`) so `_pcx_val` returns that default for every
`(role, T)` (`10162`) and `_bfp_pdl` forwards it per launch (`10166-10170`). In the sk2 kernel the producer
lane streams `P` weight stages before the wait (`8141-8168`), every thread waits at `8183`, and
`launch_dependents` fires after the last pre-wait stage (`8209`). No path sets 4 or 6; there is no
`num_stages`-style knob above 2 anywhere in the dense chain. The Triton glue kernels run at **0** pre-wait
stages — `gdc_launch_dependents()` first, `gdc_wait()` immediately after (`10280-10282`, `10771-10773`,
`10437-10439`) — so they also cannot be carrying the 4/6 penalty (they are 1-3 µs kernels; the fact's
+0.9 µs b1-boundary figure has no purchase).

**Attention = 0 K/V stages pre-wait, by construction.** Prologue (`1512-1535`): page indices for
`STAGES-1` tiles (`1521-1523`; `STAGES` = 3 for the per-request builds `OKC_STN 3`, 3 for q1-flat
`OKC_ST1F 3`, 4 for q8-flat `OKC_ST8F 4`), `cp_commit()` `1526`, `pdl_wait()` `1532`; then q rows (direct
cp.async `1537`), the new k/v rows, and the K/V flight gated by `cp_wait<IDXF ? 1 : 0>()` at `1571` (only the
index group), with the q group awaited at `1633-1635` — so K/V is in flight while the q group lands, but
*after* the griddepcontrol wait. Effective pre-wait work = 2-3 index groups (256 B each), not weight stages.

**The dormant note is a dead switch in our copy.** `OKC_PDLKV 0` (`178`) has **no `#if OKC_PDLKV` block**
in this file: the name is read only at `1519` (`IDXF = OKC_IDXFIRST && … && !OKC_PDLKV`) and `p.pdlf`
(`330`, set at `3520`) is never read by device code. So flipping the macro to 1 does *not* add pre-wait K/V
staging — it only turns off `OKC_IDXFIRST` (`147`), which would make K/V wait for the q rows too: a
regression. (`PROGRESS.md:145` already flags `-DOKC_PDLKV=1` as "still unpriced in both files"; this is why —
the priced switch no longer exists.)

**What "2 stages" would really cost.** The variant exists verbatim in another vendored copy of the same
lineage: `.vendor/Auto-research-agent/benchmarks/data/candidates/gemini-3.8-flash-opencode/kernel-muse_decode_layer/submission.py:2167-2228`
(`#if OKC_PDLKV`, ~62 lines: indices + constants + `cp_wait/csync` + `issue_kv(k,false)` ×2 + next indices,
then `pdl_wait`, then q/new-kv as direct 16-byte loads). That is a second prologue path compiled into every
attention instance — new code, not a one-liner — and the file's own note prices the *standalone* order at
"~0.3% from code layout" (`178-179`). Bounded upside: the flight can only hide under the qkv GEMM's drain,
≤ ~1-2 µs on the small-T attention cells.

**Worth.** 0 µs/replay: we are at 2 (dense) and 0-K/V (attention), and the measured penalty for going the
wrong way (4/6 stages: attention +1.7 µs, b1 boundary +0.9 µs) is not carried anywhere — +1.7 µs would be
1.5% of v1s's 110.8 µs replay and +0.9 µs 1.15% of d1s's 78.0 µs. **Change: none**; the only one-line edit
available (OKC_PDLKV 1) is a no-op-plus-regression, and the real 2-stage prologue is a new kernel path.

## 3. Cooperative-launch attribute — never paid; the alternative was already measured slower; 0 µs

**What we do.** `coop` is a *software* grid barrier over resident CTAs, not `cudaLaunchCooperativeKernel`:
host residency clamp `3557` and the `OKC_COOPGUARD` re-check `3575` (macro `235`), per-launch alternating
counters `OKC_PARITY 1` (`210`), device finalize `3504-3510`, trap guard `239`. Launches go through `<<<>>>`
(`3622`) or `cudaLaunchKernelEx` carrying only the cluster attribute (`3594`) and/or the PDL attribute
(`3607`) — `3613`. **`cudaLaunchCooperativeKernel` appears 0 times in the file** (the sole neighbouring
attribute is `cudaFuncAttributeNonPortableClusterSizeAllowed`, `3544`, a cluster property).

**Worth.** The fact's +2.47 µs is the cost of the cooperative-launch API; we never pay it on any of the six
loads → 0 µs/replay, nothing to recover.

**The alternative is already priced, and it loses on our grids.** Replacing the coop finalize with the
last-CTA finalize on the small grids is a plan-field flip (`coop=0`), no new kernel — but the file measured
it on exactly our verify shapes and coop won: b16 q8 8 splits 16.2 → 15.85 µs (+0.35 for coop) and b32 q8
4 splits 20.15 → 19.55 µs (+0.6 for coop) (`3833-3835`); at b1 q8 with one tile per CTA coop is −2.8 µs
(13.0 → 10.2, `3827-3830`). The one branch where the last-CTA finalize wins is q1 per-request
(coop 13.0-13.1 vs 12.6-12.9, `3817-3819`) and it is **already `coop=0`** (`3819`). Flipping v16s/v32s would
therefore cost 0.35-0.6 µs (0.11-0.19% of the load) — far inside judge noise (unchanged files vary up to 6%
per shape). **Change: none**; the genuine one-liner exists and is a measured regression.

## Verdict table

| fact | our state | better per the measurement | worth on which load | kind of change |
| --- | --- | --- | --- | --- |
| 1 co-residency | single stream; PDL only fills free slots; v32 (T=256) has no PDL; in-layer v32s attention 15.5 µs = the un-slowed figure | keep attention off SMs a GEMM holds | 0 µs; 7-12 µs on v32s is what we are *not* paying | none needed |
| 2 pre-wait stages | dense/glue exactly 2 (`BFP_PDL_N`, sk2 `8141-8209`); attention 0 K/V stages + 2 index groups; `OKC_PDLKV` dead | 2 (4/6 cost +1.7 µs attn, +0.9 µs b1 boundary) | 0 µs; the penalty avoided is 1.7 µs on v1s | not a one-liner (62-line prologue variant) |
| 3 coop launch | software barrier + `cudaLaunchKernelEx`; no cooperative launch anywhere | avoid `cudaLaunchCooperativeKernel` | 0 µs; +2.47 µs not paid anywhere | one-liner exists (coop=0) but measured +0.35/+0.6 µs worse on v16s/v32s |

**Nothing implemented, no `src/okc_loop/vendored_work/layer/klr_attn2.py` produced** — all three checks land
on the side the measurements endorse, and the two edits that do not require a new kernel (OKC_PDLKV=1,
coop=0 on the L==8 reqs>=16 grids) are respectively a dead-switch regression and a measured regression.
