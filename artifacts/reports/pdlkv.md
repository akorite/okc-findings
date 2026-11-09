# OKC_PDLKV: the K/V copies issued before `griddepcontrol.wait`

Candidate: `src/okc_loop/vendored_work/layer/klr_pdlkv.py` (985,068 B, line 1 = `# okc: problem=muse_decode_layer`),
copied from the board file `klr_fci5.py` (979,654 B) with the ATT7 `QDIRECT/PK2` + `OKC_WGSFCI 5` payload intact.
Board row at the time of writing: **2.28370 / 9,905.1 µs, #77 of 1212** (top-20 at 2.38333).

## 1. What the ordering is today, and what it must be

`segment()` (per-request, `!WS`, `!S2`) runs this prologue. The five dependent L2/HBM round trips measured on the
sliding path are launch+dispatch (≈2 µs), page-index fetch (≈0.7), the 16 KB K/V tile (≈0.8), q-load+QK^T/softmax/PV
(≈1.2) and the `red_v4` merge / `bar.sync` / `atom.acq_rel` counter / last-CTA `ws_acc` re-read / finalize (≈2.5-3),
plus ~0.5 µs of straggler skew — 8.42 µs total for B=1.

```
klr_fci5.py
:1521  for y < STAGES-1: issue_idx(y)          cp.async page indices  (reads p.page_table)
:1525  cp_commit()                             group 0
:1526  OKC_LATETID re-read of tid
:1532  pdl_wait()            <-- BLOCKS here: every later round trip is behind the predecessor's tail
:1533  pdl_launch_dependents()
:1537  q rows          cp.async  (reads p.qkv  -- the predecessor's output)
:1546  new k / v rows  cp.async  (reads p.qkv)
:1549  cluster_stage_gate  (reads p.qkv, gate rows)
:1551  inv_freq / k_scale / v_scale / sense / cache_loc  cp.async  (inputs)
:1560  cp_commit()                             group 1
:1562  gate-row L2 prefetch (reads p.qkv)
:1571  cp_wait<IDXF ? 1 : 0>(); csync()        IDXF=0 under OKC_PDLKV -> cp_wait<0>: groups 0 and 1 drained
:1575  OKC_SPEC miss check (MODE & 8)
:1583  for k < STAGES-1: issue_kv(k, false)    <-- the K/V copies, after the wait
:1626    issue_idx(k + STAGES-1); cp_commit()  groups 2 .. STAGES
```

`OKC_PDLKV`'s only live `#if` site is `:1519` (`IDXF = OKC_IDXFIRST && ... && !OKC_PDLKV`), i.e. the flag today only
*disables* the early index wait. `p.pdlf` (`:330`, set at `:3520` from the launcher's `pdl` bit) is **never read by
the kernel** — it only gates the launch attribute (`cudaLaunchAttributeProgrammaticStreamSerialization`, `:3606`).
PDL launches are already live in the layer: `_decode_attention7` passes `pdl=_pdl(T)` (`:11381`, `PDL = True`,
`PDL_MAX_T = 128`), and the predecessor `_qk_prep` fires `gdc_launch_dependents()` at its **entry** (`:10443`), so the
whole of `_qk_prep` overlaps this kernel and `pdl_wait()` at `:1532` blocks for all of it.

New ordering (gated by `constexpr bool PKV = OKC_PDLKV && !WS && !Cfg<L,FLAT>::S2`, `:1574`):

```
issue_idx(y) for y < STAGES-1 ; cp_commit()                          group 0
cp_wait<0>() ; csync()                                               page indices only
OKC_SPEC miss check
issue_kv_one(1) / issue_kv_g(0, 1)   <-- tile 0's stable rows (pos + L < seq)
issue_idx(STAGES-1) ; cp_commit()                                    group 1
pdl_wait() ; pdl_launch_dependents()
issue_kv_one(2) / issue_kv_g(0, 2)   <-- tile 0's rows the predecessor may still be writing
q rows / new k,v / cluster_stage_gate / constants ; cp_commit()      group 2
for k in 1..STAGES-2: issue_kv(k, false); issue_idx(k+STAGES-1); cp_commit()   groups 3..STAGES
gate-row L2 prefetch
cp_wait<STAGES-2>() ; csync()        covers groups 0..2, leaves exactly the tiles the base leaves in flight
```

Only tile 0 goes early. That is the whole change: `STAGES + 1` groups in the base's order, and the tiles behind the
q group are issued and waited exactly as before — an earlier attempt that pre-issued every prologue tile and waited
with `cp_wait<0>` (because the q group is the newest) lengthened the drain and cost 2.5-14.8 % on the attention
kernel itself (variant A below).

`issue_kv_one` is the `OKC_ONETILE` straight-line predicated issue of tile 0, split the same way, because the
B=1-sliding cell takes that build: `_plan7` gives `n_splits = min(tiles, 2*sms)` one tile per CTA (`:3798`), and the
host then picks `kn1` — `MODE | 4`, `ONE` — whenever `n_splits * 64 >= kmax` (`:3639`). Without it the split would
have replaced the branch-free prologue with the general per-row loop in exactly the cell that carries the weight.

What actually moves: the base already overlaps launch+dispatch, the CTA prologue and the page-index *issue* (they
sit above `pdl_wait()` at `:1532`), and the index copies have landed by the time the wait returns. What it does not
overlap is the K/V tile fetch — `issue_kv` runs after `cp_wait<0>` at `:1571` — so the new ordering buys tile 0's
16 KB (≈0.8 µs of the 8.42 at B=1), not the full 3.3 the ticket's decomposition attributes to the pre-wait chain.
The measured deltas in §4 are the answer: the B=1 cells gain 1.5-4 %, everything else is flat, and the layer's
geomean is unchanged.

The group count and its order relative to the q group are unchanged
(`1 + 1 + PRE` = `STAGES + 1`), which is what the main loop's `cp_wait<STAGES-2>` accounting assumes; the loop then
issues tile `i + STAGES-1` at `i = 0` exactly as before.

### The RAW guard

Two grids can be writing the cache rows this prologue reads, both still resident under PDL:

* the predecessor `_qk_prep` writes `k_cache[loc*128]` / `v_cache[loc*128]` at `loc = cache_loc[b*L + jj]`, i.e. the
  **new tokens'** rows, positions `seq .. seq+L-1` — already excluded by the existing `pos < seq` test;
* the *previous call's* grid: `_qk_prep` fires `launch_dependents()` before its own `gdc_wait()`, so this kernel can
  start while the **previous decode step's attention kernel is still running**, and that kernel writes its new-token
  rows at positions `seq-L .. seq-1` (`:1817`, `:1838`).

A row at `pos` is therefore stable iff `pos + L < seq`. `issue_kv_g(x, 1)` copies only those; `issue_kv_g(x, 2)`
copies the complement, after the wait. Rows past the range are zero-filled in pass 1 (`cp_async16_zero` issues a
copy with `src_bytes = 0` — no global read), and the block's own new rows are still left to the patch pass, exactly
as in `issue_kv(k, false)`.

The page table is read before the wait on the same footing the kernel already reads `p.seq_lens` at its top
(`:3168`, before any `pdl_wait`): it is an input of the whole layer chain — no grid in this stream writes it (the
only `page_table` uses are reads, `:306`, `:11396`, `:11433`). `p.cache_loc`, `inv_freq`, the scales and `qkv` are
all read after the wait, as before.

## 2. Why the results cannot change

Tile 0 is issued in two passes; tiles `1..STAGES-2` keep the base's single pass. The two passes issue the **same
multiset of `cp.async` operations** — same shared-memory destination
(`r * ROWB + ((cchunk ^ (r & 7)) << 4)`), same global source
(`((idx[r] << lp) + (pos & (P-1))) << lrb + pair*128 + cchunk*16`, or `row * row_bytes + …` when `!off32`), same 16
bytes — only their position relative to `griddepcontrol.wait` differs:

* `{pos < end && pos < seq && pos + L < seq}` (pass 1) ∪ `{pos < end && pos < seq && pos + L >= seq}` (pass 2) =
  `{pos < end && pos < seq}`, the single-pass set; `{pos >= end}` is zero-filled in pass 1 only;
* `{seq <= pos < end}` stays untouched in both passes (patch pass, unchanged).

`artifacts/reports/pdlkv_guard.py` replays both predicates over `L ∈ {1,8} × P ∈ {1,64} × off32 ∈ {0,1} ×
sliding/full × T ∈ {1,2,4,8,16,32} × seq ∈ [0, 192] ∪ {4096, 8191}` and all 4 prologue tiles: **57,120 tile cases
identical**.

Not runnable as a bitwise test: the file cannot be built for `sm_80` at all, base or new. `segment()` calls
`pdl_wait()` unconditionally, and `ptxas -arch=sm_80` rejects it —
`error: Modifier '.wait' requires .target sm_90 or higher` (reproduced locally with the wheel's `ptxas` 13 on a
two-line snippet). The `-DOKC_SM90=0` path at `:3809` therefore only ever means "emulate clusters on an sm_90a
cubin" (`kn[0]`, `:3585`), not a real sm_80 build, so the A100 can compile the candidate but never execute it.

## 3. Gates

* `ast.parse` ok; 985,068 B (limit 1,000,000); line 1 unchanged.
* lint delta vs `klr_fci5.py`: **none** (ruff `--output-format concise`, 180 findings in each, identical message
  multiset after stripping line numbers).
* **sm_90a compile gate** on the shipped file, `tools/colab_gate.py` + `colab run --gpu A100 --timeout 1500`:
  `_att7` **ok 89.7 s** (89.7 s wall incl. VM) — the whole attention TU, i.e. `segment<1,false>`, `segment<8,false>`, `segment<1,true>`,
  `segment<8,true>` and every launch-mode build (`kn[1..3]`, `kn1`, `ksp`, `ksp1`); `_moe`, `_moe(32)`, `_moe_tiny`,
  `_moe_tiny2`, `_moe_mid`, `bfp__get_ext` ok. The only `bad` entries are the three tvm_ffi glue modules
  (`ModuleNotFoundError: No module named 'tvm_ffi'`) — the documented Colab limitation. The **base file gates
  identically** (86.5 s, same three failures, `_att7` ok), so the change costs no build.
* `tools/cu_build.py --variant MID`: **vacuous for this file**, base and new alike. `ATT7_CUDA_SRC` opens with
  `#include <torch/extension.h>` (`:123`), and `cu_build` compiles `src.split("#include <torch/")[0]` — the empty
  string — so it prints `ok ATT7_CUDA_SRC … compiled, no kernel emitted (template body)` without touching the code.
  (The device part alone does not compile under that tool's nvrtc either: libstdc++ 13 `cstdlib`/`cstddef` errors.)

## 4. Measured — three variants, and what they say

All three on the judge (`okc check`, off the board, **13/13 tests passed** each). The base is the board row itself:
`klr_fci5.py` job `9743b6e71fbd42fd` = **2.28370 / 9,905.1 µs**, whose profile
`profiles/klr_fci5.9743b6e71fbd42fd.txt` gives the exact per-case and per-kernel reference, so nothing below is
inferred.

| variant | what it does | job | score / µs | vs 2.2837 |
|---|---|---|---|---|
| base | the board row | `9743b6e71fbd42fd` | 2.28370 / 9,905.1 | — |
| **A** | tiles `0..STAGES-2` split across the wait, `cp_wait<0>` before the q math | `c95bb5e748d447df` | 2.26380 / 9,992.2 | −0.87 % |
| **B** | A + the `OKC_ONETILE` split (`issue_kv_one`) | `866706d73b7a40f1` | 2.24604 / 10,071.2 | −1.65 % |
| **C** (shipped) | **tile 0 only** goes early; tiles `1..STAGES-2` keep the plain issue behind the q group, waited with `cp_wait<STAGES-2>` | `017cc4163e5b46ab` | 2.26760 / 9,975.4 | −0.70 % |

The score is a geomean of `sglang / ours`, and the *reference* moves between runs by up to ±4 % per case (the same
H200, different minute). Splitting each run into "our time" and "the reference's time", both relative to the base
run, isolates the kernel:

```
        score    ref/base   own/base   per-case own/base (C)
A     2.26380    +1.09%     +1.68%
B     2.24604    -1.24%     +0.37%
C     2.26760    -0.49%     -0.08%    decode 1 sliding  -1.48%   verify 1 sliding  -2.65%
                                     decode 1 full     -3.94%   verify 1 full     -3.72%
                                     decode 16 sliding +0.32%    verify 16 sliding +3.22%
                                     decode 16 full    +0.34%    verify 16 full    -1.90%
                                     decode 32 sliding +8.69%    verify 32 sliding -0.18%
                                     decode 32 full    +2.21%    verify 32 full    -1.25%
```

**C's own time is −0.08 % — neutral.** Its −0.70 % score is the reference being 0.49 % faster in that run, not the
kernel. And the per-case shape is exactly the predicted one: the two latency-bound cells (B=1, one tile per CTA)
gain 1.5-4 %, the B>=16 cells are flat, and `decode 32 sliding` is the one cell whose own time swings ±8 % between
repeat runs of *different* files (base 138.0, A 145.5, B 139.6, C 150.0 µs) — that +8.69 % is that cell's noise, not
this change.

### A regressed, and the per-kernel profile says why

The attention kernel itself, base → A:

```
case                   MODE   base µs    A µs    delta
decode 1 sliding         13     5.30     5.80   +9.43%
decode 1 full            11    11.50    13.20  +14.78%
decode 16 sliding         1     7.90     8.10   +2.53%
decode 32 sliding         1     9.50     9.90   +4.21%
verify 1 sliding         15     6.60     6.80   +3.03%
verify 1 full            11    14.20    14.80   +4.23%
verify 16 sliding         3    11.70    12.80   +9.40%
verify 32 sliding         3    15.10    16.30   +7.95%
```

Neither half of A is a bandwidth problem:

1. `cp_wait<0>` in front of the q math drains **all** the K/V groups, because the q group is the newest. In the base
   only tile 0 has to land before the q / softmax work; tiles 1.. overlap it. With 2 CTAs per SM (decode 1 full: 264
   splits) that second tile's 16 KB is ~1 µs of real time on the critical path — the +14.8 % on MODE 11, the cell
   with the most CTAs.
2. Losing the `OKC_ONETILE` straight-line prologue in the ONE build (`MODE 13`, decode 1 sliding) is the +9.4 %; B
   restores it and that cell goes 67.82 → 66.63 µs.

C is the fix: `1 + 1 + 1 + (STAGES-2)` = `STAGES + 1` groups, the same count and order as the base, and
`cp_wait<STAGES-2>` leaves exactly the tiles the base leaves in flight.

`OKC_PDL` (`:3780`) was left at 0: it is dead for this candidate. `use_pdl = OKC_PDL if pdl is None else bool(pdl)`
(`:11352`) and the only `_launch7` caller — `_decode_attention7` (`:11381`) — always passes `pdl=_pdl(T)`, which is
already 1 for T <= 128; the other attention entry (`:11433`) calls the extension's `run` directly and never reads it.
Flipping it would change no launch, and the file measured by job `017cc4163e5b46ab` is byte-for-byte the file shipped.

### The ticket's premise

`launch+dispatch ≈ 2 µs` and the `0.7 µs` page-index fetch were **already** hidden: they sit above `pdl_wait()` at
`:1532` in the base, and `_qk_prep` fires `gdc_launch_dependents()` at its entry. The only round trip still behind
the wait is the 16 KB K/V tile (≈0.8 µs), and moving it in front of the wait costs a prologue drain of the same
order — so the lever is worth ~0, not 40 % of 8.42 µs. That is what the three runs show.
