# research_06_microarch — Hopper/H200 microarchitecture, non-arXiv where it exists

Scope: what the literature actually *measures* about (1) the achievable bandwidth of an 8–17 MB burst
on H200, (2) L2 partitioning / NUMA locality on H200, (3) clock and power behaviour, and (4) why a short
paged-KV grid lands at 1.15–1.67 TB/s where the flat cells on the same paged table land at 3.07–3.65 TB/s.

Every external claim carries a URL. Internal numbers are cited to `artifacts/reports/*.md` (our own
measurements; not independently published). `[DERIVED]` = my inference, not a cited measurement.
Where the literature is silent I say so instead of filling the gap.

---

## 0. Verdict

| question | answer the literature supports | confidence |
|---|---|---|
| Is 4.2–4.45 TB/s a real H200 ceiling? | Yes. Best published cold-HBM microbenchmark on H200 NVL: **4287 GB/s = 89.3 % of the 4.8 TB/s spec** (TMA), 4139 GB/s with plain `ld.global` | measured |
| Does rate fall with burst size? | Yes, monotonically, down to at least **64 MiB → ~3 TB/s** (given); smallest *published* point on this hardware family is a 1 GiB single pass. **The 8–17 MB regime is unpublished** | measured at ≥64 MB; **silent** below |
| Per-SM issue-rate ceiling? | **No binding ceiling.** SM data pipe = **124–128 B/clk/SM**; we use **7.0 B/clk** at 1.67 TB/s. TMA per-issue tax = **170 cycles (~94 ns)**; a CTA issuing 4–8 tiles pays ≤0.8 µs | measured |
| Saturation needs how much in flight? | **~32 KiB bytes-in-flight per SM** for both TMA and `cp.async` on H100 (3.1–3.2 TB/s there). One 16 KB bulk copy ≈ **128 outstanding 128-B requests** — i.e. roughly one SM's whole MSHR budget per issue | measured |
| Deeper pipelines past 3 stages? | **Measured worthless.** "Having more than 2 stages doesn't help much… suggests the TMA unit has an implicit internal window" | measured (RTX 5090) |
| L2 partitioning on H200 | Real and **H200-specific**: effective L2 is only **65.8 %** of the 60 MB physical (≈39.5 MB), **4 KB** partition-interleave, remote HBM throttled by the LTC fabric | measured |
| Does slice-ordered work pay? | **Up to 1.22×** on GroupGEMM (B200); on H200 the address-remap overhead **eats the GroupGEMM gain**; small-KV attention "gains little", and on small workloads the persistent-block probe can make it *slower* | measured |
| Do short kernels pay a clock-ramp tax? | **No.** A frequency transition on GH200 costs **5.5 ms min / 23.4 ms mean / 477 ms max** — 3.5× to 300× the 1.4–15 µs cell, so the kernel runs entirely at whatever clock was already in force | measured |
| Does HBM clock drop? | Not under power-capping: **3201 MHz, "strictly static"** from 200 W to 700 W. It *does* drop under **HBM thermal** throttle (1593→1297 MHz on H100) with no public flag pre-535 | measured |
| Is page scatter / TLB / DRAM banks the sliding-vs-flat delta? | **No — bounded at ≤13 %** by our own flat cells on the identical table and rows, and TLB pre-warming measurably changes nothing on H100 (NVIDIA staff) | measured + bounded |

---

## 1. Finite-transfer bandwidth on HBM3e: what sets the rate for an 8–17 MB burst

### 1.1 The published curves

The only public H200 cold-HBM microbenchmark with a clean eviction protocol is Zhengrong Wang's TMA/cp.async/normal-load
sweep on an **H200 NVL**, CUDA 12.8: a 256 MiB eviction buffer read outside the timed interval, then a
single pass over a **1 GiB** working set (`repeat=1`), plus a separate L2-resident build (40 MiB, 32 repeats,
`-dlcm=cg`), Nsight counters unavailable so the regimes are separated by capacity + eviction + policy
([github.com/seanzw/random/tree/main/tma_bw](https://github.com/seanzw/random/blob/main/tma_bw/README.md)):

| mechanism | L2-resident | cold HBM (1 GiB, single pass) | % of 4.8 TB/s |
|---|---:|---:|---:|
| TMA (`cp.async.bulk`) | 7787 GB/s | **4287 GB/s** | 89.3 % |
| `cp.async` | 7749 GB/s | 4277 GB/s | 89.1 % |
| plain `ld.global` (`float4`) | 7158 GB/s | **4139 GB/s** | 86.2 % |

Notes that matter for us: **(a)** the three mechanisms differ by only **3.6 %** at 1 GiB, so the choice of
load instruction is *not* what sets a large-burst rate; **(b)** the L2-resident number (7.8 TB/s) is 1.8×
HBM, so a cache-resident ceiling cannot be what we are hitting; **(c)** the H100 STREAM/Triad equivalent
saturates at **2.43–2.47 TB/s = 73 % of 3.35 TB/s** ([Kane Walter](https://www.kanewalter.com/writing/17-memory-bandwidth-experiment/)),
i.e. "89 % of spec" is a good result and "73 % of spec" is the normal result for a naive copy.

The burst-size dependence is the settled fact of this ticket: ~4.6 TB/s at 2 GiB, ~4.2 at 128 MiB, ~3 at
64 MB, lower below. **I found no published measurement at 8–17 MB on H200.** The smallest published
finite-transfer points for this family are 64 MiB (ours) and 1 GiB (Wang). *The literature is silent
below 64 MB*; treat any claim there as `[DERIVED]` from the local measurements.

### 1.2 Latency × concurrency: how many requests are actually needed

The correct accounting is Little's law at the *request* level, and both the number and its unit are
published:

* **H100 unloaded HBM latency 353 ns**, L2 145 ns, L1 17 ns, from a single-SM pointer chase; streaming
  read bandwidth is "a median **91–94 % of the 3.35**" TB/s ([Concurrency Response, Table/§4.1](https://arxiv.org/html/2608.15764v1)).
* Wang's H100/H200-adjacent L2 study measures **L2 hit = 282.5 cycles ≈ 141 ns** and, using Little's law,
  shows that sustaining **3.2 TB/s at 141 ns needs ≈451 KB of outstanding traffic** across the GPU
  ([kanewalter L2](https://www.kanewalter.com/writing/18-l2-cache-bandwidth/)).
* Scaled to H200 at 4.2 TB/s and 353 ns: **4.2e12 × 353e-9 ≈ 1.48 MB in flight** = **11.2 KiB per SM** =
  **~350 outstanding 32-B sectors per SM** = **~88 outstanding 128-B requests per SM**. At a loaded
  latency of 600–800 ns it is 19–25 KiB/SM and ~150 requests/SM. `[DERIVED]` from the two cited numbers.
* The bytes-in-flight threshold is measured directly on H100: **`cp.async` reaches 3.1 TB/s (94 % of peak)
  and TMA 3.2 TB/s (96 %) at 32 KiB bytes-in-flight per SM**, TMA doing it "with a single CTA's single
  descriptor with a few pipeline stages" ([When and Why to Use TMA, §4.2](https://arxiv.org/html/2610.34907v1)).

**Consequence for a 16 KB tile.** A 16 KB bulk copy is 128 × 128-B requests = **more than one SM's whole
outstanding-request budget in a single instruction**. So:

1. a 3-stage pipeline at 2 CTAs/SM already holds **32–64 KiB/SM**, i.e. at or above the published knee;
2. deeper pipelines cannot add in-flight capacity because the SM has already run out of MSHRs — which is
   exactly what Wang measures on the TMA path: *"Having more than 2 stages doesn't help much… This suggests
   the TMA unit has an implicit internal window for processing requests, so issuing more than 2 TMA requests
   doesn't improve bandwidth"* ([seanzw](https://seanzw.github.io/posts/gpu_load_bandwidth)).

That second point is a **negative result for the `STAGES 3 → 4` lever** that `attn_gap.md` §2 lists as "the
only clean 1.5× left". At 16 KB per tile we are already past the point where stage depth buys anything.

### 1.3 `cp.async.bulk` vs `cp.async` vs `ld.global` at this size

The measured discriminator is **per-issue fixed cost vs per-byte cost**, not peak rate
([TMA paper §4.2, Fig. 4](https://arxiv.org/html/2610.34907v1)):

| bytes-in-flight/SM | TMA latency | LDGSTS latency | ratio |
|---|---:|---:|---:|
| 0.5 KiB | 1.4 µs | 0.43 µs | **3.1×** |
| 1 KiB | — | — | 2.1× |
| 2–8 KiB | — | — | 2.8–3.0× |
| 16 KiB | — | — | <2× |
| ≥32 KiB | 0.07 ns/B | 0.05 ns/B | parity |

* TMA's fixed tax is the descriptor walk + bulk issue + completion machinery: **~170 cycles per access
  before the transfer begins** ([Dissecting Hopper §5.1](https://arxiv.org/html/2501.12084v2), same number
   reused in the TMA paper).
* LDGSTS needs *thread count* to fabricate in-flight bytes — 3.1 TB/s required
  "4 CTAs/SM × stages × 256 threads × float4"; TMA gets the same with one elected thread.
* Wang's independent sweep on the same code base agrees in shape: TMA and `cp.async` both show a **~4 KB
  chunk-size threshold** below which achievable bandwidth collapses (normal loads: 2033 GB/s at 4 KB vs
  653 GB/s at 2 KB; TMA at 512 B chunks peaks at **9.1 %** utilisation)
  ([seanzw](https://seanzw.github.io/posts/gpu_load_bandwidth)).

**So at our granularity the three mechanisms are equivalent per byte and differ only in the fixed tax,
which a 16 KB tile already amortises to ~0.6–3 %.** The choice is not the lever.

*Silent:* no published comparison of `cp.async.bulk` (non-tensor) *one-dimensional* vs descriptor-based TMA
*at 8–17 MB total transfer*, and no published data on H200 at all for the mechanism comparison. Luo et al.
do note that 1-D **tensor** TMA is capped at 256 elements/box ≈ 2 KB and recommend the non-tensor form,
which is what our `bulk_g2s` uses ([Dissecting Hopper §5.2](https://arxiv.org/html/2501.12084v2)).

### 1.4 Per-SM issue-rate ceiling — measured, and not the limit

| quantity | measured value | source |
|---|---|---|
| L1 / shared-memory data path | **124.1–127.9 B/clk/SM** (Hopper, `float4`) | [Luo Table 5](https://arxiv.org/html/2501.12084v2) |
| whole-GPU L2 | 3942–4472 B/clk | ibid. |
| TMA per-issue overhead | ~170 cycles | ibid. §5.1 |
| TMA engine count | **one per SM** | [TMA paper §3.1](https://arxiv.org/html/2610.34907v1) |
| kernel launch overhead (host, H100) | 6.31 µs per launch (removed by CUDA graphs) | [CarmitHaas](https://github.com/CarmitHaas/roofline-to-cuda-graphs) |

At 1.67 TB/s over 132 SMs we move **12.6 GB/s per SM = 7.0 B/clk at 1.81 GHz**, i.e. **5.6 % of the SM's
124 B/clk pipe**. Even at the machine ceiling (4.2 TB/s → 31.8 GB/s/SM → 17.6 B/clk) we would be at 14 %.
**No per-SM issue-rate ceiling can explain 1.67 TB/s**; and 4–8 TMA issues × 170 cycles = 0.4–0.75 µs
against a 10 µs kernel is ≤7.5 %. `[DERIVED from the two cited tables + our µs]`

### 1.5 Does more parallelism or a bigger burst help more? Both, but parallelism wins at our size

Three independent measurements say the **grid**, not the burst, is the knob at 8–17 MB:

* **H100, plain `float4` copy, fixed 1 GiB** ([Kane Walter](https://www.kanewalter.com/writing/17-memory-bandwidth-experiment/)):
  64 blocks → 323 GB/s; 128 → 607; 256 → 1109; 512 → 1865; **1024 → 2430 (saturated)**; 4096 → 2465.
  The curve is ~linear in block count until ~8 blocks/SM, then flat. Nsight on the saturated point:
  *Memory Throughput 2.43 Tbyte/s, Max Bandwidth 72.6 %*.
* **H800, TMA, fixed 4 GB** ([Luo §5.2](https://arxiv.org/html/2501.12084v2)): CTAs at 1/2/3/4 × SM count
  with loads of 1–16 KB — "larger load sizes and more thread blocks generally achieve higher memory
  bandwidth", topping out "over 1800 GB/s" (of 2039 GB/s theoretical), i.e. **more CTAs keeps helping up to
  4/SM even with 16 KB loads**.
* **H100, TMA/`cp.async`** ([TMA paper](https://arxiv.org/html/2610.34907v1)): the whole crossover is
  expressible as bytes-in-flight per SM, saturating at 32 KiB/SM.

`[DERIVED]` Reconciling them: at 1 CTA/SM issuing 16 KB tiles we sit at the 32 KiB/SM knee *only while two
tiles are actually outstanding*, and a 4–8 tile CTA is at that level for a minority of its lifetime
(§4).

### 1.6 Modelling a finite burst

For a burst of S bytes with an asymptote R and a fill/drain of F:
`t = F + S/R`. Our own sliding cells fit this with a **large F and a near-ceiling R**
(`muse_arch_levers.md` §A1, from the B=16/B=32 pairs):

| path | F | marginal | as TB/s |
|---|---:|---:|---:|
| non-WS, q_len 1 (D16S, D32S) | **6.30 µs** | 0.206 µs/MB | **4.85** |
| non-WS, q_len 8 (V16S, V32S) | **8.30 µs** | 0.438 µs/MB | 2.28 |
| WS flat, q_len 1 (D16F, D32F) | 9.20 µs | 0.253 µs/MB | 3.95 |
| WS flat, q_len 8 (V16F, V32F) | 22.3 µs | 0.263 µs/MB | 3.80 |

The marginal rate of the sliding decode path is **4.85 TB/s — above the machine's measured 4.2–4.45 TB/s**,
which is the whole story: **the bytes are not slow, the burst is too short to pay for its own fill.**
Independently, Wang's curve for the *same* kernel family (rate vs block count, saturating only near 8
blocks/SM) explains why 128–256 CTAs at 2/SM cannot substitute for runway.

---

## 2. L2 partitioning on H200 (the "uGPU" question)

### 2.1 Geometry — same as H100, and confirmed on H200

* Hopper's L2 is **two partitions** joined by a cross-partition link, with **GPCs partitioned the same way**;
  Nsight counts 60 MiB of L2 on the H200 NVL used in Wang's tests
  ([Luo §3, §4.2 insight](https://arxiv.org/html/2501.12084v2); [tma_bw README](https://github.com/seanzw/random/blob/main/tma_bw/README.md)).
* Latency, fine-grained P-chase on H800 (Hopper, 114 SM) — **near vs far is a 60 % gap**:

  | | H800 | A100 |
  |---|---:|---:|
  | L2 near hit | **258 cyc** | 208.0 |
  | L2 far hit | **414 cyc** | 356.6 |
  | L2 near miss | **555 cyc** | 474.9 |
  | L2 far miss | **744 cyc** | 622.7 |
  | global (stable) | 656 cyc | 566 |

  ([Luo Table 4](https://arxiv.org/html/2501.12084v2).)
* Chips and Cheese on H100 PCIe: "accessing the 'far' partition takes nearly twice as long… H100 also
  **regresses compared to A100 when accessing the entire 50 MB L2, 3.8 TB/s vs 4.5 TB/s**"
  ([chipsandcheese](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth)).
* **H200-specific**: effective L2 is only **65.8 % of physical** (≈39.5 of 60 MB) because the LTC fabric
  *replicates* remote data into the local partition instead of forwarding to L1 — so remote lines compete
  for local capacity; **effective L2 per partition = 30 MB**; partitions are interleaved by an **XOR hash on
  physical address bits at 4 KB granularity**; and "**remote HBM bandwidth is bottlenecked by the LTC fabric
  that connects the two L2 cache partitions, not by the HBM itself**"
  ([Dissecting How Die Scaling Breaks GPU Fine-grained Scheduling, §6.1–6.2](https://arxiv.org/html/2609.24270v1)).

### 2.2 Does ordering work by slice/local partition change achieved bandwidth? Yes, with caveats

Same paper, real H200 and B200 hardware, using FlashAttention-3 (H200) / FA-4 (B200) and CUTLASS GroupGEMM
as baselines, **indirect 4 KB remapping on H200** (kernel-side, O(1) address arithmetic) or a driver
allocation interface on B200, plus **persistent thread blocks with one block pinned per SM** so that
block→SM→partition is planned:

* **GroupGEMM (the MoE shape) up to 1.22×** — but that figure is **B200**; the text is explicit that
  "NUMA-aware execution performs better on B200 for two reasons. B200's cross-die NUMA effect is larger than
  H200's, and direct allocation incurs negligible address-calculation overhead. **On H200, indirect allocation
  requires address remapping, which adds overhead, particularly for GroupGEMM**" and on H200 that overhead
  "outweigh[ed] GroupGEMM's gains".
* **Attention**: "Most kernels on both GPUs benefit from NUMA awareness in some configurations", but
  "attention kernels with small KV working sets gain little", and the **PTB startup probe can make small
  workloads slower**: "PTB-based workload assignment requires each block to query which SM it is running on,
  adding startup overhead… For small workloads, this overhead outweighs the gains".
* Crossover warning that bites anything we build: without topology awareness, a 1.10× becomes **0.78×**
  (MLA decode, B200, 70/78 SM split).
* Their own summary table prices "Full-GPU kernels, *example: Attention & MoE*" at **up to 1.22×**, and
  "intra-application multiplexing (prefill/decode)" at **+14.3 % decode**
  ([Table 2](https://arxiv.org/html/2609.24270v1)).

### 2.3 `cudaLimitPersistingL2CacheSize` / access-policy window: H200 vs H100

**The literature is silent on any H200-vs-H100 difference**, and it should be: the two parts are the same
GH100 die and the same 60 MB two-partition L2 (§2.1). The documented no-ops are platform-level, not
SKU-level: the set-aside is disabled under MIG and cannot be changed under MPS
([CUDA C Programming Guide §4.13.1–4.13.8, as quoted in our own `klr_l2p.md` §1](klr_l2p.md)).
Internally this axis is already **falsified** for our problem for a reason that has nothing to do with
H200: the judge flushes L2 with a read larger than L2 before every replay, so there is no cross-step reuse
for a carve-out to capture (`klr_l2p.md` header, §4).

What the partition data *does* say, quantitatively, about our 1 GiB paged KV: with a **4 KB** interleave and
**random page order**, every 128-B row lands in a random 4 KB chunk, so ~50 % of requests are remote and
pay 555 → 744 cycles (+34 %), lifting the average miss latency ~17 % and thus the bytes-in-flight needed by
the same ~17 %. `[DERIVED]` That is a real but *modest* tax — and, critically, **the flat cells pay exactly
the same tax** (same table, same rows), so it cannot be the 2.3× sliding/flat gap.

---

## 3. Clock and power behaviour

### 3.1 What the clocks are, and what moves them

| | H100 | H200 |
|---|---|---|
| SM base / boost | 1665 / **1980 MHz** | 1665 / **1980 MHz** |
| memory clock | 1593 MHz | **3201 MHz** |
| memory spec | 2.41 TB/s | 4.89 TB/s |
| *(measured, 4 nodes × 4 GPUs, 50 runs/cap)* | | |

Source: [Architectural Trade-offs in the Energy-Efficient Era, Table 1](https://arxiv.org/html/2604.11391v2).
That study's central clock finding:

* "**Across all tested GPU power limits, the memory frequency remained strictly static: 1593 MHz for the
  H100 and 3201 MHz for the H200**" — 200 W to 700 W, no HBM clock movement from *power* capping.
* **SM frequency does move, and the H200 moves more**, precisely because its memory draws more: memory power
  "saturates at approximately 125 W" on H100 vs "roughly 220 W" on H200, and that "reduces the available
  power budget for the SMs. Consequently, **the H200 experiences more aggressive SM frequency throttling**".
* A memory-bound Triad needs "only around 350 W" on H100 but "**at least 550 W**" on H200 to sustain peak
  bandwidth; H200 throttles as far as its 345 MHz floor at ≤300 W.
* Chips and Cheese see the same on H100 PCIe: "**H100 occasionally dropped down to as low as 1395 MHz, or
  just under 80 % of its maximum boost clock**… Other metrics from nvidia-smi suggest we could be hitting a
  power limit, particularly when pulling data from L2"
  ([chipsandcheese](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth)).

`[DERIVED]` for our 1.81 vs 1.97 GHz observation: 1.81/1.97 = **92 %**, which is inside the range both
sources report for a loaded H100/H200 (80–100 % of boost), and the mechanism they identify — memory power
displacing SM power headroom — is exactly the shape of a dequant+MMA kernel that also streams. It is
**not** evidence of an HBM clock drop: HBM clock is static under power capping.

### 3.2 HBM clock *can* drop — thermally, silently

A separate path from power capping, and documented for Hopper: per-stack HBM3/HBM3e DTS sensors feed
GSP-RM, which steps `MEMCLK` down in 100 MHz decrements when the hottest stack passes ~95 °C, and the
default `nvidia-smi` view does **not** show it (`Memory Current Temp` = N/A on H100/H200; only NVML
`NVML_TEMPERATURE_MEMORY` / DCGM field 252). Measured consequence on H100: **MEMCLK 1593 → 1297 MHz (−19 %),
achieved bandwidth 3.1 → 2.3 TB/s (−26 %), TTFT +32 %, with `temperature.gpu`, `sm clock` and `power draw`
all unchanged** ([manishklach](https://manishklach.github.io/writings/why-hbm-thermal-throttling-is-silent.html)).
This is a *fleet-health* claim from a practitioner blog, not a vendor doc — treat the exact numbers as
one operator's measurements, but the *existence* of an HBM-only throttle flag (`HW Memory Thermal Slowdown`,
driver ≥535.104.12) is checkable and is the thing worth asserting.

**Implication for us:** a run-to-run bandwidth spread that tracks node/rack rather than code could be HBM
thermal throttling. It is checkable on the judge node with `dcgmi dmon -e 252,204` if we are ever allowed
telemetry — the cheap guard is that our own same-sha spread is 2.2 % (`attn_gap.md` §1), which is far too
small for a −26 % clock event, so this is *not* currently confounding the board.

### 3.3 Do short kernels pay a clock-ramp tax? No — and the number is decisive

Measured frequency-switching latency, statistics over all frequency pairs, on **GH200 (Hopper: 1980 MHz max
SM, 2619 MHz memory)**, plus A100 and RTX 6000:

| | GH200 worst-case | GH200 best-case |
|---|---:|---:|
| min | **5.554 ms** | 4.914 ms |
| mean | **23.448 ms** | 7.866 ms |
| max | **477.318 ms** | 140.352 ms |

([Methodology for GPU Frequency Switching Latency Measurement, Table II](https://arxiv.org/html/2502.20075v1).)

A transition costs **at least 5 ms**; our whole attention cell is **1.4 ms** and the individual shapes are
**4.8–185 µs**. So a kernel cannot observe a ramp *within itself*: it executes entirely at whatever clock
was already in force when it launched. Two consequences, both useful:

1. **Short kernels are not penalised relative to long ones by ramp** — the "short kernel gets a cold clock"
   hypothesis is dead, at least for kernels launched into an already-busy GPU.
2. **The observed clock is inherited from the preceding workload**, and it drifts on a 5–23 ms timescale —
   i.e. *between* replays, not inside one. That is the correct frame for the 1.81-vs-1.97 observation: it
   says the load average before/during the timed region sets the clock, not the kernel's own first
   microseconds.

**Literature silence:** nothing published measures clock behaviour on a *CUDA-graph-replayed, back-to-back,
sub-millisecond* kernel sequence. `[DERIVED]` back-to-back graph replays keep the load average high and the
clock therefore pinned; the mechanism to worry about is the judge's per-replay L2 flush and any host gap,
not the kernel itself.

---

## 4. Why a short paged-KV grid reaches only 1.15–1.67 TB/s where flat cells reach 3.65

### 4.1 The four candidate mechanisms, each tested against published numbers

**(a) Memory-level parallelism per CTA — SUPPORTED, and the strongest.**
Rate scales with concurrency until a knee, on every published measurement: blocks (323 → 2430 GB/s from 64
to 1024 blocks, [Kane Walter](https://www.kanewalter.com/writing/17-memory-bandwidth-experiment/)); CTAs
(1→4 per SM, [Luo §5.2](https://arxiv.org/html/2501.12084v2)); bytes-in-flight per SM (knee at **32 KiB**,
[TMA paper](https://arxiv.org/html/2610.34907v1)). Our sliding cells have **4 tiles/CTA at 256 CTAs** and
**8 tiles/CTA at 128 CTAs**; the flat cells have **75–151 tiles/CTA at 264 CTAs**
(`attn_gap.md` §1). The variable that separates 3.65 TB/s from 1.57 TB/s is **tiles per CTA: 151 versus 4**
(`attn_pack128.md` §3). At 1 CTA/SM the per-CTA outstanding window *is* the per-SM budget, so halving the
CTA count halves in-flight unless tile count rises.

**(b) Sector-level coalescing of 128-B rows across a random page table — RULED OUT as the explanation.**
The flat cells read the *same* paged table, the *same* 128-byte paired rows, at page_size 1, and reach
**3.65 TB/s over 606.7 MB** and **3.07 TB/s over 304.3 MB** (`attn_pack128.md` §3). Whatever scatter costs,
the flat cells pay it too. Quantitatively, the published size-matched ceiling for 128 MiB is **4.2 TB/s**
(§1.1), so the entire scatter + partition + page-table tax is **≤13 %** at 304–607 MB — and our marginal
sliding rate is **4.85 TB/s**, i.e. *no* scatter tax at all. Corroboration that page scatter barely matters
on Hopper: sweeping KV **page size 16 → 256 tokens** on FA3's `cp.async`-paged path moves all nine
(model × phase) curves by **±4 % with no systematic trend**
([TMA paper §4.1](https://arxiv.org/html/2610.34907v1)) — a 16× change in contiguity per page, invisible.

**(c) TLB reach for a 1 GiB paged cache — RULED OUT (measured negative).**
H100, device-only `cudaMalloc`, 2 MB assumed pages, pages pre-touched and then thrashed with junk, GEMV
working set spanning hundreds of pages: **no measurable improvement in either direction**. The explanation
given is the right one for us: "A GPU is a latency hiding machine… perhaps this is just another latency to
be hidden", the TLB-fetch bandwidth is ~16 B per 2 MiB page (0.0008 %), and page-table walks overlap across
threads — classic pointer chasing serialises, bulk loads do not
([NVIDIA forums, incl. Robert Crovella](https://forums.developer.nvidia.com/t/inquiry-about-gpu-on-device-tlb-behavior/345240)).
`[DERIVED]` 1 GiB at 2 MB pages is **512 entries**, comfortably inside any plausible GPU L2-TLB reach.

**(d) DRAM bank conflicts from a randperm page order — RULED OUT as the explanation, unmeasurable here.**
Same argument as (b): the flat cells use the identical randperm order. **The literature is silent on
HBM3e row-buffer hit rate as a function of GPU access pattern** — the DRAM-interaction literature
([Mutlu group, SIGMETRICS'19](https://people.inf.ethz.ch/omutlu/pub/Workload-DRAM-Interaction-Analysis_sigmetrics19_pomacs19.pdf),
[DRAMSim3](https://terpconnect.umd.edu/~blj/papers/memsys2018-dramsim.pdf)) is simulation-calibrated for
DDR/HBM2 and does not publish HBM3e row-hit-vs-random numbers on H200 hardware. I found **no** measured
H200 DRAM-bank-conflict figure. The upper bound from (b) (≤13 %) stands in for it.

### 4.2 What is left: the arithmetic of a 10 µs burst

Take `dec32 sliding` (`attn_gap.md` §1): **16.8 MB in 10.04 µs = 1.67 TB/s**, 256 CTAs (2/SM), 4 tiles/CTA,
3 stages. The fitted model (§1.6) says **t = 6.30 µs + 16.8 MB / 4.85 TB/s = 6.30 + 3.46 = 9.76 µs**, which
reproduces the measurement to 3 %. So:

* the **bytes cost 3.46 µs** — i.e. the transfer, if it were happening at the machine's measured ceiling,
  would take a third of the cell;
* **6.30 µs is fill + drain + completion**, invariant to bytes.

At 1.81 GHz, 6.3 µs = **11,400 cycles**. Accounting for it, with cited components:

| component | cost | cite |
|---|---:|---|
| one HBM round trip (unloaded) | 353 ns → 640 cyc | [Concurrency](https://arxiv.org/html/2608.15764v1) |
| one HBM round trip (loaded, est. 2–3× unloaded) | 0.7–1.1 µs | `[DERIVED]` |
| 4–8 TMA issues × 170 cyc | 0.38–0.75 µs | [Luo §5.1](https://arxiv.org/html/2501.12084v2) |
| pipeline fill: 2 tiles × 16 KB at 4.2 TB/s | 7.6 ns (negligible) | — |
| grid dispatch + CTA launch of 128–256 CTAs | unpublished on Hopper | **silent** |
| split-merge arrival chain (8 splits) | ~2.9 µs at D1S | `muse_arch_levers.md` §A1, `attn_xqa_grid.md` §5 |

The listed components do **not** add to 6.3 µs; several µs are unaccounted, and the honest statement is
that **the fill/drain is a serial latency chain whose individual terms are each ~0.1–1 µs.** This is the
same phenomenon the wave-quantization literature names and fixes structurally: "if the number of work
units isn't evenly divisible by the number of available SMs… the final wave is a partial wave"; on a
114-SM part, "a computation with 115 work units will require 2 waves — exactly the same as one with 228…
adding the 115th work unit approximately **halves** the utilization"
([Colfax, CUTLASS tutorial](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/),
citing [NVIDIA's own wave-quantization section](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#wave-quant)).
Our 4-tile cell is a **half-filled final wave**, by construction.

The structural fix in that literature is exactly the one `attn_gap.md` §2 identifies as "the only mechanism
that can reach the 1–2 tile cells": **one wave, persistent CTAs, fractional work per CTA** (Stream-K), and
its **hybrid** form — "use a data-parallel schedule for as long as possible, reserving Stream-K for just
enough tiles to handle the wave quantization effect", because pure Stream-K "introduces skew… making cache
hits much harder" ([Colfax](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)).
FlashAttention-3 already ships a persistent tile-scheduler class, so the pattern is production-proven for
attention (ibid., linking `hopper/tile_scheduler.hpp`).

### 4.3 The direct answer to "why does a 32-CTA / 128-CTA grid get 1.67 TB/s?"

1. **It is not bandwidth.** The kernel's marginal rate is 4.85 TB/s (§1.6) — the bytes are moving at
   *above* the machine's measured cold-HBM asymptote.
2. **It is not per-SM issue rate.** 7.0 B/clk/SM against a 124 B/clk/SM pipe (§1.4), and TMA issue tax ≤7.5 %.
3. **It is not coalescing, TLB, or DRAM banks.** Bounded at ≤13 % by the flat cells on the same table
   (§4.1b–d), and the page-size sweep is flat to ±4 %.
4. **It is concurrency × runway.** At 4 tiles per CTA the CTA is inside its own fill for most of its life,
   so the *time-average* bytes-in-flight is a fraction of the 32 KiB/SM knee even though the *peak* is at
   or above it. Every published concurrency curve (blocks, CTAs/SM, bytes-in-flight) has a knee at 8× to
   60× our grid, and our flat cells sit **on the far side of it** with 151 tiles/CTA.

---

## 5. Ranked mechanisms by expected value

Board arithmetic from `muse_arch_levers.md` §2.2: **Σ sens = 0.7201 %/µs**, sliding-only **0.534 %/µs**,
per-shape D1S 0.180, V1S 0.114, D16S 0.088, D32S 0.070, V16S 0.048, V32S 0.035 %/µs; **MoE is 51.3 % of the
board's per-µs value**.

### #1 — Delete the 6.3/8.3 µs fill from the six sliding cells — **+2.2 %** on the attention row (2.659 → ~2.72)
Work-centric one-wave plan (Stream-K *hybrid*: data-parallel over the long spans, stream-K only over the
tail), so each CTA holds 32 KiB in flight for ~its whole life instead of 4 tiles' worth. Priced at
F: 6.30/8.30 → 1.5 µs using the fitted model:
D16S 0.088×4.8 + D32S 0.070×4.8 + V16S 0.048×6.8 + V32S 0.035×6.8 ≈ **+0.76 + 0.56 %**, plus ~3 µs at
D1S/V1S 0.294×3 ≈ **+0.88 %**, ≈ **+2.2 %**; half the fill removed → **+1.1 %**.
Literature basis: [Colfax](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/),
[wave quantization](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#wave-quant),
32 KiB/SM knee ([TMA paper](https://arxiv.org/html/2610.34907v1)).
**Do not** pair this with `STAGES 3→4`: Wang measures that >2 TMA stages buys nothing because of the TMA
unit's internal window ([seanzw](https://seanzw.github.io/posts/gpu_load_bandwidth)) — a 16 KB tile already
exceeds one SM's outstanding-request budget.

### #2 — NUMA/L2-partition placement for the MoE group-GEMM — **+1.5–2.6 % board at 1.03–1.05×, +11 % at the paper's 1.22×**
(MoE is 51.3 % of the board's per-µs value, so board-% ≈ the MoE speedup × 0.513.)
Basis: H200 effective L2 65.8 % of physical, 4 KB interleave, LTC fabric caps remote HBM bandwidth, remote
data replicated locally ([die scaling, §6](https://arxiv.org/html/2609.24270v1)); the paper's own
"Attention & MoE, full-GPU kernel" row is **up to 1.22×** with NUMA-aware allocation + persistent blocks.
Our MoE at bs=1 is a grouped GEMM, the closest shape to their GroupGEMM, and reads **71 MB at ~2.4 TB/s
vs 3.2 TB/s standalone** — a 33 % gap of exactly the kind a partition-local read can close.
**Risk, also from that paper:** on *H200* (not B200) the remap overhead ate GroupGEMM's gain, small-workload
attention gained little, and the persistent-block SM probe can make small cells *slower* — so price it at
1.03–1.05×, not 1.22×.

### #3 — Page scatter / TLB / DRAM-bank order — **≤+0.4 % board; treat as a bound, not a lever**
The whole family is capped at the ≤13 % gap between our flat cells (3.07–3.65 TB/s at 304–607 MB) and the
size-matched published ceiling (4.2 TB/s at 128 MiB; Wang's 4287 GB/s at 1 GiB). Recovering all of it on the
flat cells is 0.187 %/µs × ~30 µs over the two largest cells ≈ **+0.4 %**. Ruled out as the sliding explanation
by construction (same table, same rows) and by the ±4 % page-size sweep and the TLB null result.

### Not a mechanism
**Clock/power.** A clock transition costs ≥5 ms on GH200, i.e. ≥3.5× the whole attention cell and ≥250× an
individual shape ([FGCS/arXiv 2502.20075, Table II](https://arxiv.org/html/2502.20075v1)); HBM clock is
static under power caps (3201 MHz), moves only under HBM *thermal* throttle, and our same-sha spread (2.2 %)
is far too small for a −26 % clock event. The 1.81-vs-1.97 GHz figure is just the SM clock sitting at 92 %
of boost, the range both hardware sources report under load.

---

## 6. Where the literature is silent (do not quote a number here)

1. **Achievable bandwidth for an 8–17 MB burst on H200.** No published measurement. Smallest published
   finite-transfer points: 64 MiB (ours) and 1 GiB ([Wang](https://github.com/seanzw/random/blob/main/tma_bw/README.md)).
2. **Grid-dispatch / CTA-launch ramp cost *inside* a kernel duration on Hopper.** Nobody publishes it; the
   6.31 µs figure is host-side launch, which CUDA graphs remove.
3. **H200-vs-H100 differences in `cudaLimitPersistingL2CacheSize` or access-policy-window behaviour.**
   Undocumented and unlikely (same die, same 60 MB partitioned L2); MIG/MPS no-ops are the only documented
   caveats. Internally already falsified for our problem (`klr_l2p.md`).
4. **HBM3e row-buffer hit rate vs access pattern on H200.** No hardware measurement exists; the DRAM
   literature is simulation-calibrated for DDR/HBM2.
5. **TMA `cp.async.bulk` *non-tensor* at 1-D sizes above 2 KB, at 8–17 MB total.** Luo et al. recommend the
   non-tensor form for 1-D but only sweep ≤16 KB per issue and fixed 4 GB totals.
6. **Clock behaviour of back-to-back CUDA-graph-replay microkernels.** Nothing published; the ≥5 ms
   transition latency is the only anchor.

## 7. Checks worth making on the device (one launch each, no source change)

1. `dcgmi dmon -e 252,204` during a scored run — DCGM field 204 (`MEM_CLOCK`) against its max settles #3.2
   for the judge node in one number.
2. `nvmlDeviceGetClockInfo(NVML_CLOCK_MEM)` vs max, sampled between replays — if MEMCLK = max and SM clock
   ≈1.81 GHz, §3.1 is confirmed and the clock axis can be closed permanently.
3. A 4-tile vs 8-tile vs 16-tile variant of *one* sliding cell at fixed bytes — this is the direct
   discriminator between "runway" (#1) and "anything else"; the fitted `t = F + S/R` predicts 6.3 µs of
   invariance to bytes, and any other mechanism predicts otherwise.
