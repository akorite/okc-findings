# OKC_SEGK1 / OKC_SEGK8: the flat partition's boundary charge

## 1. Mechanism (klr_fci5.py:3310-3425, 3233-3280)

`attn_kernel<L, FLAT>` splits, per launch, the *keys* of all B requests across `n_splits`
CTAs (`_plan7`: flat, full layer, `1 < B <= 512` -> `n_splits = 2*sms//pairs` = 264 on 132 SMs).
The split is made in a **virtual key axis**:

    T = T0 + (B - 1) * SEGK                     # T0 = sum_b (seq_b + L)
    [cs, ce) = CTA's share of T (even, or W0 1135/2048 for q_len 8)
    vmap(v): b = last request with cum[b] + b*SEGK <= v
             real = min(v - b*SEGK, cum[b+1])

so **every request boundary after the first is charged SEGK virtual keys**: a CTA whose share
crosses a boundary loses SEGK real keys (the `min` clamps it to the request it starts in). SEGK is
the price, in keys, of one extra segment (q staging + pipeline refill + one more merge, "~7 us at
q_len 8" in the kernel comment). SEGK = 0 is a plain even split; large SEGK makes boundary-spanning
CTAs progressively lighter, at the cost of `(B-1)*SEGK/G` extra keys on every other CTA.

Which constant governs which load (`constexpr int SEGK = (L == 8) ? OKC_SEGK8 : OKC_SEGK1`):

| load | L | B | CTA share | constant |
|---|---|---|---|---|
| d16f `decode 16` full | 1 | 16 | 264 CTAs, even (coop=0, w0=1024) | `OKC_SEGK1 400` |
| d32f `decode 32` full | 1 | 32 | 264 CTAs, even | `OKC_SEGK1 400` |
| v16f `verify 16` full | 8 | 16 | 264 CTAs, W0 pair split 1135/2048, coop=1 | `OKC_SEGK8 600` |
| v32f `verify 32` full | 8 | 32 | 264 CTAs, W0 pair split, coop=1 | `OKC_SEGK8 600` |

(`fpart` is compiled out here: `OKC_W0ENABLE=1` via `extra_cuda_cflags`, so even the q_len-1 grids
take the `vmap` path; both constants are live on all four loads.)

## 2. What the optimum is: SEGK* = P / a

The tail CTA is either non-spanning, at `K + (B-1)S/G` keys, or spanning one boundary, at the same
minus `S` plus the penalty `P` (in key-equivalents). The max is minimised when the two are equal:

    S* = P                                    (in keys)
    S* = P_us / a,     a = per-CTA us per key (SEGK only redistributes; the time saved
                       going 0 -> S* is exactly P_us, and none of it depends on B)

Fitting that to the vendored numbers (this build's delta, not theirs):

| load | 0 -> 600/400 gain | keys/CTA | us/CTA | a (keys/us) | implied S* |
|---|---|---|---|---|---|
| b16 q8 | 107.4 -> 103 = 4.4 us | 4849 | 103 | 47.1 | **207** |
| b32 q8 | 187.4 -> 184 = 3.4 us | 9705 | 184 | 52.7 | **179** |
| b16 q1 | 90.4 -> 88.7 = 1.7 us | 4848 | 90.4 | 53.6 | **91** |

The gain being B-independent (~4 us at q8) is the check that the model is the right one: it is the
one-off boundary penalty, not a fraction of the work. Both shipped values therefore overshoot:
600 is ~3x, 400 is ~4.4x the implied optimum. Overshoot is cheap but not free — it hands
`(B-1)*S/G` keys to everybody else (b16: 0.057 keys per unit of S):

| S (q8) | max non-spanning CTA, keys | vs S=200 |
|---|---|---|
| 200 | 4849 + 11 | 0 |
| 500 | 4849 + 28 | +0.36 % |
| 600 (base) | 4849 + 34 | +0.48 % |
| 720 | 4849 + 41 | +0.62 % |

Second-order caveat, and it points the same way: on our build the tall CTA is the W0-heavy one
(1135/2048 -> +11 % keys, 5410 for v16f), while the boundary penalty lands on whichever CTA spans
a boundary — often a *light* one that then finishes early and wastes the relief. That dilution
makes our optimum <= theirs, never above.

`artifacts/reports/segk_part_sim.py` replays the kernel's partition exactly (same vmap, same
1135/2048 split) over 60 agentic length draws: keys/CTA mean stay flat (4847 -> 4853 for d16f,
4846 -> 4841 for v16f from S=0 to 720), no CTA goes empty at any of these values, and the only
thing that moves is the spanning CTA's share (min 3752 -> 3632 for v16f, 600 -> 720).

## 3. Variants

All three are byte-identical to `klr_fci5.py` except lines 186-187, `ast.parse`-clean, and
`_att7` + `_moe` compile on the sm_90a gate (`artifacts/bench/gate_segk_{lo,hi,min}.out`;
the three tvm_ffi glue modules stay ungatable on Colab, as always).

| file | SEGK8 | SEGK1 | why |
|---|---|---|---|
| `klr_segk_lo.py` | 500 | 340 | -17 % on both, toward the implied optimum |
| `klr_segk_hi.py` | 720 | 480 | +20 % on both, the "more is better" side |
| `klr_segk_min.py` | 200 | 100 | the model optimum from §2 |

## 4. Measured: klr_segk_hi.py (SEGK8 720 / SEGK1 480)

Job `1bc2a20158c04d0c`, check only (not submitted): **geomean 2.2411** (CI 2.1963-2.2828) vs the
`klr_fci5.py` check's 2.2730 (CI 2.2242-2.3170) — i.e. -1.4 % against the very base it came from,
and *below* the board row (2.2474), so it would not even have landed.

Raw `cand_us` against the fci5 check (`a407ad5e63ae4e2d`), same shapes, same harness:

| load | fci5 (us) | hi (us) | delta | kernel |
|---|---|---|---|---|
| decode 16 (d16f) | 8898.8 | 8991.6 | **+1.04 %** | flat, SEGK1 400 -> 480 |
| decode 32 (d32f) | 11146.4 | 11385.4 | **+2.14 %** | flat, SEGK1 400 -> 480 |
| verify 16 (v16f) | 15916.7 | 16203.9 | **+1.80 %** | flat, SEGK8 600 -> 720 |
| verify 32 (v32f) | 21774.5 | 22298.7 | **+2.41 %** | flat, SEGK8 600 -> 720 |
| decode 1 | 4252.2 | 4247.4 | -0.11 % | **no flat grid** (B = 1) |
| verify 1 | 6697.2 | 6620.0 | -1.15 % | **no flat grid** (B = 1) |

The two B = 1 loads never enter the flat kernel (`_plan7` needs `1 < B`), so -0.11 / -1.15 % there
is the per-load noise of a single check: +-1.2 %. The four flat loads all move the *same* way by
+1.0..+2.4 %, which is outside that noise (4/4 sign test) and is the whole reason the geomean fell.

So: **more is worse**, and by more than the linear overshoot model of §2 predicts (it estimates
+0.13 %, measured +1.8 % on average). Two readings, both pointing down:
* the shipped 600/400 sit on the falling side of the curve, slope ~9 % of load time per +100 % SEGK,
  so the peak is *below* 600/400 (the §2 estimate is ~200/100);
* on our build the tall CTA is the W0-heavy one (1135/2048 = +11 % keys), so the relief SEGK hands
  a boundary-spanning CTA is mostly wasted and only the `(B-1)S/G` surcharge on everybody else is
  left. That is why the measured penalty is ~10x the key-arithmetic estimate.

Next (unmeasured, both emitted and gated): `klr_segk_lo.py` (500/340, the conservative step in the
measured direction) and `klr_segk_min.py` (200/100, the model optimum). Anything at or above the
shipped 600/400 is now falsified, including the `OKC_SEGK8 700` row on the constant-sweep list.

