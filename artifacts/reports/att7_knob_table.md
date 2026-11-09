| # | knob | default | comment (verbatim) | class | note |
|---|------|---------|--------------------|-------|------|
| 134 | `OKC_NP` | `128` | producer threads (q_len 1 flat; 128 since v111: f16 -0.7%, f32 -0.4% vs 256) | **a** | - |
| 135 | `OKC_WS` | `1` | producer warpgroup for the flat (B > 1, full layer) kernel; 0 = v10's flat kernel | **a** | - |
| 136 | `OKC_WS8` | `1` | producer warps for the q_len 8 flat kernel too | **a** | - |
| 137 | `OKC_NP8` | `128` | q_len 8: 4 producer warps (its consumers need the registers) | **a** | - |
| 138 | `OKC_WSMINB` | `2` | (none) | **d** | structural constant, no comment |
| 139 | `OKC_SETMAXNREG` | `1` | (none) | **d** | structural constant, no comment |
| 140 | `OKC_PREGS` | `0` | (none) | **c** | register split knob (0 = built-in 56/32); a value to sweep, not a flag |
| 141 | `OKC_TMAPG` | `1` | producer mode: page indices by TMA (one copy per tile) instead of per-thread loads | **a** | - |
| 142 | `OKC_XHALFPV` | `0` | (probe) q_len 1: half the PV work, to see whether the q1 flat loop is consumer-bound | **d** |  |
| 143 | `OKC_XNOWAIT` | `0` | (probe) per-request loop: no wait for the tile's copies (garbage), the compute-bound tile time | **d** |  |
| 144 | `OKC_MRGDBG` | `0` | (debug builds) phase marks around the merge's fence (columns precl1 / cl2) | **d** |  |
| 145 | `OKC_MERGE2` | `1` | last-CTA merges: one acq_rel atomic after the CTA barrier instead of per-thread fences | **a** | - |
| 146 | `OKC_MERGE3` | `1` | cooperative merges: one acq_rel barrier arrive after the CTA barrier instead of per-thread fences | **a** | - |
| 147 | `OKC_IDXFIRST` | `1` | per-request kernels: K/V issued once the page indices land (not after the q rows too) | **a** | - |
| 148 | `OKC_MAGICDIV` | `1` | per-request chunk: host magic multiplier instead of fdiv_i | **a** | - |
| 149 | `OKC_SC1` | `1` | coop finalize: sum_copies without the copy loop when nc == 1 | **a** | - |
| 150 | `OKC_QP16` | `1` | q_len 8 QDIRECT q pass: four threads per row (16 dims each) | **a** | - |
| 151 | `OKC_ONENEW` | `1` | ONE builds: the straight-line prologue copies also for the tile holding new rows | **a** | - |
| 152 | `OKC_SC8` | `1` | coop finalize: the eight workspace copies' loads in one round trip | **a** | - |
| 153 | `OKC_PINDIV` | `1` | coop finalize: the slice divisions pinned above the barrier wait | **a** | - |
| 154 | `OKC_ONEPRED` | `1` | ONE builds: predicated (branch-free) straight-line prologue copies | **a** | - |
| 155 | `OKC_LATETID` | `1` | per-request prologue: tid re-read (volatile) after the page-index copies, so the q-pass math stays late | **a** | - |
| 156 | `OKC_PGCONT` | `0` | (racecheck control) page-index TMA barriers once per launch, slots continued across segments | **d** |  |
| 157 | `OKC_TMAPG8` | `0` | ... for the q_len 8 flat kernel too (measured ~+0.7%) | **b** | measured ~+0.7% (slower) |
| 158 | `OKC_ST8F` | `4` | pipeline stages of the q_len 8 flat kernel | **a-** |  |
| 159 | `OKC_COOP` | `1` | cooperative finalize for single-request grids (0: always the last-CTA finalize) | **a** | - |
| 160 | `OKC_PVW` | `0` | q_len 8 on sm_90: PV as an asynchronous wgmma from a double-buffered f16 V tile | **c** |  |
| 161 | `OKC_WGDEC` | `1` | q_len 8 producer mode: the two consumer warpgroups sync separately (one per head) | **a-** |  |
| 162 | `OKC_SEGDBG` | `0` | (none) | **d** |  |
| 164 | `OKC_PV8` | `1` | q_len 8 on sm_90: PV as an FP8 wgmma (P in e4m3, as SGLang's FA3 FP8 path does: pod/p16probe*.py) | **a** | - |
| 165 | `OKC_PROF` | `0` | per-warp phase profile of the main loops (diagnostic builds) | **d** |  |
| 166 | `OKC_PROFPV8` | `0` | profile builds: keep the FP8 PV path | **d** |  |
| 168 | `OKC_PV8FLAT` | `1` | FP8 PV in the flat (B > 1 full) kernel too (needs OKC_QSS: 104-register consumers) | **a** | - |
| 169 | `OKC_QSS` | `1` | with FP8 PV: q in shared memory (QK^T as an SS wgmma), 8 fewer live registers | **a** | - |
| 170 | `OKC_PV8SYNC` | `0` | wait for each PV wgmma inside its tile (fewer live registers) | **c** |  |
| 171 | `OKC_Q8RAW` | `1` | q_len 8 sm_90: FP8 query from the unit-RMS query (scores scaled after QK^T) | **a** | - |
| 172 | `OKC_FLATCOOP1C` | `0` | compile the grid-wide flat finalize for q_len 1 too | **c** | SHIPPED in klr_attn3.py |
| 173 | `OKC_RAWTHR` | `0` | q8 lazy max: compare raw scores with (offset + 8) / ssc (measured slower: +10 static fp32 instr) | **b** | measured slower: +10 static fp32 instr |
| 174 | `OKC_PCHK` | `1` | q8: the lazy-max offset test on the packed e4m3 P (no per-tile row-max scan) | **a** | - |
| 175 | `OKC_TURN` | `0` | q8 flat: the heads' warpgroups alternate in the exponential phase (1: every tile, measured +1.3%; 2: offset once, on the first tile) | **b** | measured +1.3% (slower) |
| 177 | `OKC_CLSM` | `0` | (experiment) minimum dynamic shared memory of cluster launches (forces one CTA per SM) | **c** | no scored load uses clusters (all six plans are cs=0/ncl=1) |
| 178 | `OKC_PDLKV` | `0` | PDL launches: K/V issued before griddepcontrol.wait, q / new k,v as direct loads after it (decode_layer: build with 1; standalone it costs ~0.3% from code layout) | **b** | decode_layer: build with 1; standalone ~0.3% worse |
| 180 | `OKC_RARE` | `1` | q8 flat: common tiles (no mask, no new rows) found once; one compare per tile | **a** | - |
| 181 | `OKC_PCHKTEST` | `0` | test hook for OKC_PCHK (offset moves on most tiles) | **d** |  |
| 182 | `OKC_UNEV` | `0` | per-request coop grids with more splits than SMs: uneven tiles by dispatch wave (w0; no gain on b1 q8 full: whole-tile granularity, 5 tiles per split, cannot shorten the slow wave) | **b** | no gain on b1 q8 full |
| 184 | `OKC_W0L1` | `0` | per-SM uneven shares for the non-cooperative (q_len 1) flat grids too (measured +3..5 us: the generation-tagged registration costs more than the q_len 1 balance gains) | **b** | measured +3..5 us (slower) |
| 186 | `OKC_SEGK8` | `600` | flat q_len 8: virtual keys per request boundary in the partition (b16 107.4 -> 103, b32 187.4 -> 184) | **a** | - |
| 187 | `OKC_SEGK1` | `400` | flat q_len 1: the same (b16 90.4 -> 88.7) | **a** | - |
| 188 | `OKC_W0SPREAD` | `0` | W0 registration barrier over 8 arrival counters (measured slower: 8 sequential polls) | **b** | measured slower |
| 189 | `OKC_W0EARLY` | `1` | the W0 registration atomics issued before the prefix sums (warp 1), the wait after them | **a-** |  |
| 190 | `OKC_W0FORCEBAD` | `0` | test only: force the W0 registration's fallback (every CTA takes the even split) | **d** |  |
| 191 | `OKC_W0BID` | `1` | rank the two CTAs of an SM by blockIdx (the lower one gets the warp schedulers' priority: it finishes first on every SM) instead of by arrival | **a-** |  |
| 193 | `OKC_DBGON` | `0` | phase timestamps (diagnostic builds): compiled out, the checks cost cold instructions | **d** |  |
| 194 | `OKC_BALPART` | `0` | flat: balanced integer shares (one 32-bit division) instead of c*T/G | **c** | DEAD: only read as a disabling term of fpart; no implementation |
| 195 | `OKC_CONT` | `0` | flat producer mode: one continuous stage pipeline across segments (no drain per segment) | **c** | ranked #2 by size; not shipped (see risk note) |
| 196 | `OKC_DYN` | `0` | flat producer mode: dynamic chunks claimed from a global counter (needs OKC_CONT) | **c** | requires OKC_CONT |
| 197 | `OKC_FINWARM` | `0` | dry call of the last-CTA finalize during the prologue (instruction cache warm-up) | **c** | SHIPPED in klr_attn3.py |
| 198 | `OKC_PVW1` | `1` | flat q_len 8: wait for the previous PV wgmma before the V transpose (frees its registers) | **a-** |  |
| 199 | `OKC_NOXCH1` | `0` | q_len 1, workspace merge: no smem exchange of the two key halves (both reduce; measured slower) | **b** | measured slower |
| 200 | `OKC_PVSPLIT` | `1` | q8 PV: each half issued as soon as its P is packed; 1: waited at the tile end (no ptxas C7515 wgmma serialization: nothing in flight across the loop edge), 2: at the next tile's top | **a** | - |
| 202 | `OKC_QKZ` | `0` | q8 QK^T: first wgmma with write-only accumulators (not needed for C7515; QKZ=0 vf32 -0.4%) | **a** | - |
| 203 | `OKC_INV2` | `1` | per-tile invariants: hk / wsub / hi_all as lane-0 broadcasts, vld_off1 as vld_off0 ^ 32 | **a-** |  |
| 204 | `OKC_OZF` | `1` | q8: fence the O / row-sum zero init before the tile loop (scout: the C7515 trigger) | **a** | - |
| 205 | `OKC_XP_ARRALL` | `1` | every consumer thread arrives on the stage-empty barriers (compute-sanitizer racecheck clean; the one-lane arrive after bar.sync / __syncwarp was flagged on every ring reuse; vf32 -0.8%) | **a** | - |
| 207 | `OKC_XP_NOMASK` | `0` | timing/compile experiment only (wrong results): no per-element masks | **d** |  |
| 208 | `OKC_IRANGE` | `0` | q8 loop: interior-tile test as one unsigned compare on the tile index (range found once; neutral) | **b** | neutral |
| 209 | `OKC_PATCHNI` | `1` | q8 flat loop: the rare new-row patch as a noinline call (no hoisted bounds math per tile; -1.5%) | **a** | - |
| 210 | `OKC_PARITY` | `1` | per-request coop barrier: per-launch alternating counters (no departure counter) | **a** | - |
| 211 | `OKC_SENSE2` | `0` | per-request coop barrier: sense-reversing with the flag read in the prologue (slower: the flag store delays the waiters) | **b** | slower |
| 212 | `OKC_UNRN` | `1` | q_len 8 loop unroll in the plain split-KV per-request kernel (b32 q8 sliding -0.5 us) | **a** | - |
| 213 | `OKC_ADDRN` | `1` | OKC_ADDR in the per-request kernels too (few tiles per CTA: the hoisting does not pay) | **a** | - |
| 214 | `OKC_OFF32` | `1` | 32-bit index math for qkv / workspace / caches (host-checked sizes) | **a** | - |
| 215 | `OKC_UNR8` | `2` | q_len 8 flat loop unroll (2: the V^T buffer / stage parity become constants per copy; -4.5%) | **a** | - |
| 216 | `OKC_RED4` | `1` | q_len 8 workspace merge with 16-byte atomics (lane-pair swaps) | **a-** |  |
| 217 | `OKC_SWAPHALF` | `0` | (none) | **c** | no comment; use site says experiment |
| 218 | `OKC_LATESYNC` | `1` | with OKC_PK2 (flat): the V^T proxy fence and warpgroup barrier move from after the transpose to just before the PV issue (each warp runs its exponentials without waiting for the others' V^T) | **a** | - |
| 220 | `OKC_LATESYNCN` | `1` | OKC_LATESYNC for the per-request q8 kernels (split PV): the V^T proxy fence and warpgroup barrier just before the first PV half instead of right after the transpose | **a** | - |
| 222 | `OKC_QKFIRST` | `1` | per-request kernels (one tile stream): the tile's QK^T wgmma issued right after the stage barrier, before this thread's K/V copies for tile i + STAGES - 1 (they refill tile i-1's stage, which no wgmma of tile i reads), so the tensor core starts while the copy instructions issue | **a** | - |
| 225 | `OKC_FPART` | `1` | q_len 1 flat grids (<= 32 requests: ballots): warp 0 computes this CTA's key range and segment list right after its prefix sums (no all-thread binary search, no serial scan by thread 0, one barrier less: v288 vf16 spent 1.18 us between the prefix sums and the segment list) | **a** | - |
| 228 | `OKC_EARLYBAR` | `1` | flat producer mode: the first segment's stage barriers set up at kernel start (in the shadow of the seq_lens load) instead of after the partition (one init + CTA barrier off the critical path) | **a** | - |
| 230 | `OKC_PTPF` | `1` | (with OKC_FPART) warp 0 prefetches the first segment's first STAGES tiles of page indices into L2 as soon as the segment list is known (the producers' index loads then hit L2) | **a** | - |
| 232 | `OKC_FPRO` | `0` | (debug) flat prologue stamps in dbg slots 40-44 (needs OKC_SEGDBG=1: stride 64) | **d** |  |
| 234 | `OKC_COOPGUARD` | `1` | host: a cooperative (grid-barrier) launch only when the instance actually launched (ONE / SPEC builds have their own register counts) holds the whole grid, else plain split-KV; device: the grid barriers trap after 1 s instead of spinning forever (never a hang on any data or node) | **a** | - |
| 240 | `OKC_ST1F` | `3` | q_len 1 flat kernel: pipeline stages (3 measured best: b16 full 93 -> 91.5 us vs 6) | **a** | - |
| 241 | `OKC_STN` | `3` | per-request kernels: pipeline stages | **a-** |  |
| 242 | `OKC_UWARP` | `1` | (none) | **a-** |  |
| 243 | `OKC_INEW` | `0` | q_len 8 flat loop: the new-row patch test as one compare (first such tile found per segment; measured slower) | **b** | measured slower |
| 244 | `OKC_ADDR` | `1` | q_len 8 loop: loop-carried stage / phase / descriptors / V^T buffer (fewer address ops per tile; slower before OKC_UWARP/UARGS (vector registers), -1% since (uniform registers)) | **a** | - |
| 246 | `OKC_STSM` | `0` | q_len 8 FP8 PV: V^T stores via stmatrix (2 per thread per tile instead of 8; measured ~1.5% slower) | **b** | measured ~1.5% slower |
| 247 | `OKC_STSMPR` | `1` | ... in the per-request kernels (where it wins) | **a** | - |
| 248 | `OKC_TCHUNK` | `1` | per-request splits: whole 64-key tiles per CTA | **a-** |  |
| 249 | `OKC_EXPPOLY` | `0` | q_len 8 FP8 PV: how many of every 4 exponentials run on the FMA pipe (0..4; measured slower: issue-bound) | **b** | measured slower: issue-bound |
| 250 | `OKC_EXH8` | `0` | q_len 8 FP8 PV: exponentials as packed f16 pairs (experiment) | **c** | NOT IMPLEMENTED in this copy: the name appears only in the LSYNC/PCHK/PK2 constexprs, so =1 just disables all three |
| 251 | `OKC_NOSYNC0` | `1` | flat q_len 8 FP8 PV: no warpgroup barrier at the start of each tile | **a-** |  |
| 252 | `OKC_LAZYMAX` | `1` | q_len 8 sm_90: no per-tile row-max shuffles (only for the rare offset move) | **a** | - |
| 253 | `OKC_SENSE` | `0` | coop finalizes: sense-reversing barriers (no departure counter round trip at the end) | **c** | DEAD while OKC_PARITY=1: that branch is only reached when PAR is false |
| 254 | `OKC_WSPLIT` | `500` | flat: per mille of the keys for the first half of the grid (500 = even) | **a-** |  |
| 255 | `OKC_F2` | `0` | q_len 8 flat: one CTA per SM, two streams claiming tiles per KV head (measured slower: one producer warpgroup per SM limits) | **b** | measured slower |
| 256 | `OKC_S2` | `0` | q_len 8 per-request kernels: two tile streams per CTA, one CTA per SM (measured slower) | **b** | measured slower |
| 257 | `OKC_PTR` | `0` | flat q_len 8: the producer warps build V^T (measured much slower: the producers stall on their copies) | **b** | measured much slower |
| 258 | `OKC_PTRD` | `2` | OKC_PTR: tile y is transposed after the copies of tile y + OKC_PTRD are issued | **b** | inert: OKC_PTR=0 |
| 259 | `OKC_XP_NOTR` | `0` | timing experiments only (wrong results): skip the V transpose | **d** |  |
| 260 | `OKC_XP_NOFENCE` | `0` | timing experiments only: skip the proxy fence after the transpose | **d** |  |
| 261 | `OKC_XP_NOEXP` | `0` | timing experiments only: P = scores (no exponentials) | **d** |  |
| 262 | `OKC_CLDBG` | `0` | (none) | **d** |  |
| 263 | `OKC_SPINDBG` | `0` | (none) | **d** |  |
