# R5 — NVFP4 weight-entropy probe (muse_moe_w4a16 oracle weights)

VERDICT: NO — lossless smaller encoding is not worth >3%. Codes are at 8.000 bits/byte
(uniform randint in nvfp4.py:117 — incompressible). Only scales have headroom (11% of
weight bytes, H=3.7 bits/scale stored 8): 6-bit repack (already-shipped SC6) = 2.78%,
5-bit fixed fails losslessness (alphabet 32–47 > 32), entropy floor 5.96% requires
Huffman/rANS decode in the weight-load path (precedent: measured codec lost −6…−11% net).
Block deltas dead: scales i.i.d. per block. M2 stays dead beyond the 2.78% SC6 slice.
(Full table from SoundAmphibian in transcript ~10:4xZ.)
