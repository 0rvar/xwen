# 2026-07-29 — The 27B prefill gap was the dense FFN's gemm, not the DeltaNet scan: a Q4_K cooperative-tensor kernel takes prefill from 263 to 702 tok/s @925

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


**Context.** Two entries below, the DeltaNet arc refuted its own premise and handed off a
question: the 27B's 1.8-2.1x prefill loss to llama.cpp is not in the DeltaNet layers, it
is "in the dense projections", and the next arc "should start from a per-stage profile
rather than from a reading of llama.cpp's kernels". A profiling pass did exactly that and
found the answer in one stage. This arc fixed it.

**The profile.** Transcribed from the profiling pass, which produced no docs of its own.
Conditions: `pmset -g` reported `lowpowermode 0` — NOT low-power mode, but this machine
emits no `powermode` key at all, so high-power mode was never positively confirmed and
must not be claimed. `XWEN_BENCH=1` (warm, GPU-complete), GPU exclusivity verified before
every run, fixtures at 880 and 3851 tokens. Rows are tagged MEASURED or DERIVED because
two of the headline figures are derived and one carries a known bias.

Wall (MEASURED, reproduced twice): **2.66-2.68 s @880 (330.7 tok/s)**, **13.86 s @3851
(277.9 tok/s)**.

| stage | @880 ms (% wall) | @3851 ms (% wall) | µs/tok @880 → @3851 | |
|---|---|---|---|---|
| Dense FFN (64 layers) | 2268 (85.3%) | 9926 (71.6%) | 2578 → 2578 | DERIVED |
| DeltaNet non-scan (48) | 278 (10.4%) | 1216 (8.8%) | 316 → 316 | DERIVED |
| Attention (16) | 155 (5.8%) | 914 (6.6%) | 176 → 237 | MEASURED |
| DeltaNet scan (48) | 136 (5.1%) | 377 (2.7%) | 154 → 98 | MEASURED/DERIVED |
| Prefill mask (hoisted) | 4.01 (0.2%) | 51.22 (0.37%) | 5 → 13 | MEASURED |
| **sum** | **2841 (106.8%)** | **12484 (90.1%)** | 3229 → 3242 | |
| unaccounted | −181 (−6.8%) | +1376 (+9.9%) | −206 → +357 | |

**Read the −181 ms.** A negative residual is physically impossible: the stage sum exceeds
the wall clock. The cause is that the FFN row is DERIVED from an isolated T=512 rate that
is ~7-8% pessimistic against what the FFN achieves inside a real forward. So the honest
claim is a **band: the dense FFN is 66-85% of prefill wall time**, and the two DERIVED
rows carry that bias. Everything else is measured directly.

**Two protocol traps the profiling pass documented, both easy to repeat.** (1) All rates
above are AMORTIZED — BATCH=8 dispatches per sync, outputs held alive so the allocator
pool cannot inject a false write-after-write barrier. Per-dispatch numbers charge a
command-buffer round trip a real forward never pays, and are wildly different (attention
102.12 vs 57.13 ms/layer, DeltaNet 8.903 vs 3.367, FFN 24.589 vs 20.620); building the
budget from them sums to **127% of wall**, i.e. nonsense. (2) These benches throttle: the
same T=3851 q4_K gate measured 53.773 ms in a 9.3 s run, 54.118 ms in a low-duty run, and
66.639 ms (23% slower) in a 36 s run. Low-duty-cycle only; a profiling run long enough to
heat the GPU inverts conclusions.

**The mechanism, and it is not subtle.** The FFN runs `DenseMlp` → `QLinear::forward` →
candle `QMatMul` → `kernel_mul_mm_q4_K_f32`; the same shapes through `ops::matmul_f16` →
`kernel_mul_mm_f16_f32_t` are 2.4-3.0x faster:

| [T, 5120] × [5120, 17408] | Q4_K (`QMatMul`) | f16 cooperative-tensor gemm |
|---|---|---|
| T = 512 | 7.893 ms (11.6 TFLOP/s) | 3.242 ms (28.2 TFLOP/s) |
| T = 880 | 13.143 ms (11.9) | 4.412 ms (35.6) |
| T = 3851 | 54.118 ms (12.7) | 19.979 ms (34.4) |

(Profiling pass's figures. Re-measured independently for this entry with
`dense_ffn_prefill_timing`: 8.170 / 2.988 at T=512, 12.390 / 4.188 at 880, 55.904 /
19.932 at 3851 — same conclusion, within 8% on every cell.) The f16 rate is not a
synthetic-shape artifact: the DeltaNet projections, which DO take the f16 plane in
production, measure 27.7-30.4 TFLOP/s at T=880. The reason the FFN does not is that the
dual-plane machinery (`Weights::attn_proj`, gguf.rs:657) has exactly one caller,
`Proj::load` in attention.rs:108.

**Why this is kernel efficiency and not a memory wall — stated without appeal to a peak
bandwidth nobody measured.** At T=512 the Q4_K arm reads 50.14 MB of weights and the f16
arm 178.32 MB. The Q4_K arm moves **3.6x fewer weight bytes and takes 2.4x longer**
(10.9 vs 66.0 GB/s of weights+output). If either arm were bandwidth-bound, the one moving
fewer bytes would be the faster one. It is not.

Counterfactual (DERIVED, and an **upper bound** — it assumes a quantized gemm could reach
the full f16-plane rate, which is not guaranteed): 694 tok/s @880 and 496 @3851 against
llama.cpp's 486 / 502.

The same profile refuted a standing suspect and left two others open. All three are now
TODO.md ledger items rather than folklore. Their figures are the profiling pass's, carried
over rather than re-measured here — with one exception, flagged where it appears: the
attention-glue split came from the arc's briefing, not from the raw profile.

**Refuted: the materialized causal mask.** `flash.metal` is genuinely unreachable at head
dim 256 (`ops::flash_attn` hard-bails at `head_dim != 128`, dispatch.rs:3324/3361, zero
production callers), so prefill runs candle sdpa against a host-built mask — but that
mask is **hoisted**: built once per chunk in `model.rs` `run_stack` and shared across all
16 full-attention layers, NOT rebuilt per layer. The profiling pass's own first run got
this wrong and multiplied by 16, turning a 51 ms non-event into a ~682 ms scare; the
corrected figure is **0.37% of wall at 3851** (51.22 ms) and 0.15% at 880. Mask + sdpa
together grow ~1.2 percentage points between the two lengths against an observed 16%
throughput drop — an order of magnitude short of what the hypothesis needs. The 402 MB is
DERIVED (Σ over chunks of `n_head × seq × k_seq × 2`), not measured.

**Open: a length-dependent residual outside every measured stage.** Per-token wall goes
3023 → 3599 µs between the two fixtures (+576, the 330.7 → 277.9 drop), and the four big
stages account for only **+13 µs/token** of it — the FFN and DeltaNet non-scan rates are
flat, and the scan actually gets *faster* per token with length. But part of the residual
swing is an artifact of the DERIVED FFN row's 7-8% bias, so the defensible statement is
**+350 to +560 µs/token**, not a hard +576. Not attributable without per-layer
instrumentation inside `run_stack`.

**Open: attention glue.** Attention is 57.13 ms/layer amortized at T=3851 (up from 5.15
at a single 512-token chunk at position 0, growing monotonically with position), and the
brief that opened this arc put ~42.43 ms/layer of that in ~10 unfused eager passes —
a figure that appeared in the briefing rather than in the raw profile, so treat it as
indicative until re-measured. `ops::attn_gate` already exists but is wired only into the
DFlash path.

**Built: `src/ops/dense_mm.metal`.** A dense Metal-4 cooperative-tensor gemm that reads
Q4_K directly and dequantizes each weight tile in registers. It is `f16_t.metal`'s kernel
with exactly one substitution — the A-tile staging phase gains ggml's block-quant
dequant instead of a half widen-copy — so the 64-row × 128-token tiles, the f32
activation read straight from device as a cooperative tensor with no B staging, the
reduced-precision matmul2d descriptor and the extent-clipped store all carry over
unchanged. That is also how ggml writes it: templated over
`(block_q, nl, dequantize_func)`. Instantiated for q8_0/q4_K/q5_K/q6_K; q4_K is the
production one.

**The alternative was priced, not hand-waved.** Dequantizing to a transient f16 scratch
and feeding the existing gemm needs 178 MB written and 178 MB read back per projection
per chunk against the 50 MB of Q4_K the kernel otherwise streams — ~8x the weight
traffic, ~0.68 ms per projection at 600 GB/s against the gemm's own 3.2 ms, so about a
fifth of the win handed straight back plus a 178 MB scratch buffer. Permanent f16 planes,
the trick the attention projections use, are not even arithmetically available here:
17.1e9 FFN parameters at 2 bytes is 34 GB. The planes that did ship cost nothing —
`Weights::qlinear_with_plane` reuses `qlinear_with_buffer`'s one-upload construction, so
the `QLinear` decode path and the prefill gemm index the same allocation.

**Threshold from the tile geometry, not from tuning.** `DENSE_MM_MIN_SEQ = 32`,
exclusive (`seq > 32`), following `F16_MM_MIN_SEQ` and ggml's `ne11_mm_min`. candle tiles
tokens 32 wide and the vendored kernel 128, so up to 32 tokens both sit on the same
launch-latency floor — 1.01-1.05x, a wash — and at 33 candle takes a second token tile
while the vendored gemm does not. Isolation, plateau ms, interleaved arms:

| tokens | 33 | 48 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| [17408, 5120] speedup | 1.48x | 1.30x | 1.94x | 2.19x | **2.41x** |
| [5120, 17408] speedup | 1.20x | 1.19x | 1.65x | 2.71x | **3.02x** |

Production prefill runs in 512-token chunks (`PREFILL_CHUNK`), so the kernel always sees
the end of that curve.

**End to end** (27B, `lowpowermode 0` — not LPM, high-power mode not separately
confirmable on this machine — `XWEN_BENCH=1`, `--no-draft`, committed fixtures,
interleaved arms F/C/F/C, 5 reps, medians with ranges; the classic arm doubles as the
calibration — at 263 tok/s @925 it reproduces the documented ~270 baseline, which is how
this run was distinguished from three earlier contended ones that read 3x low in **both**
arms. Note the profiling pass's own baseline on a quieter machine was higher still,
330.7 tok/s @880, so the classic column here is a valid control but not a machine-best
number):

| | fused | classic (`XWEN_DENSE_MM_CLASSIC=1`) | |
|---|---|---|---|
| prefill @925 | **702.2** [637-715] | 263.1 [230-307] | **2.67x** |
| prefill @3851 | **444.9** [409-450] | 199.9 [194-205] | **2.23x** |
| decode n=128 | 24.6 [20.6-24.7] | 22.1 [19.5-23.4] | no regression |

Decode cannot change and did not: at seq 1 the threshold sends it down the `QLinear`
chain, and `dense_mlp_below_threshold_takes_the_classic_path` asserts bit-identical
output there against a block built with no planes at all. The 1.11x in that row is
between-run drift, not an effect — the two ranges overlap almost completely. The 35B-A3B
is untouched by construction (every layer is MoE; `DenseMlp` never constructs) and
spot-checks at 1.01x prefill / 0.94x decode, both noise.

So 27B prefill goes from 1.8-2.1x behind llama.cpp to ahead of it at 925 (702 vs 486)
and near parity at 4k (445 vs 502). Against the counterfactual — which was an upper
bound, assuming a quantized gemm could reach the full f16-plane rate — the short prompt
landed on it (702 vs 694 predicted) and the long one came in under (445 vs 496). The
shortfall at 4k is where the remaining work is: the length-dependent residual above is
now a much larger share of what is left than it was of what it came out of.

**The cost, stated plainly.** This kernel is **less accurate than the `QMatMul` chain it
replaces**. Against a dequantize-then-f32 oracle at the 27B FFN shapes it lands ~4.1e-4
rel_l2 where candle's kernel lands ~1.9e-4; the two differ from each other by ~3.7e-4.
Both stage the weight tile as half and accumulate in f32 — the extra ~2e-4 is matmul2d's
reduced-precision tensor-core path, which is where the throughput comes from. It is not a
new precision class (the attention prefill gemm made the identical trade, and llama.cpp
sets the same descriptor flag for its own dense FFN), but it is a real one, so the kernel
is graded like the DeltaNet scan and not like the glue kernels:
`XWEN_DENSE_MM_CLASSIC=1` is pinned on **both** sides of the strict tier, a `dense_mm`
provenance field (parity_schema v7, grandfather `classic`) proves which side ran what,
and the bounded mm/decode/ppl tiers carry the signal against frozen floors. Cached pre-v7
references stay valid — no binary of that era had the gemm.

**Both gates pass on frozen floors.** 27B: strict `cos=1.000000`, mm `cos=1.000000`,
decode 64/64 with 0 mismatches on all three fixtures, ppl `Δnll=0.000243`. 35B-A3B
(mandatory here because the provenance schema moved to v7, not because any 35B code
path changed): strict `1.000000`, mm `0.999631`, decode 63/63/62 with 0 mismatches, ppl
`Δnll=0.000791` — reproducing the DeltaNet entry's numbers exactly, which is the
confirmation that the 35B is bit-for-bit untouched.

**Shipped.** `src/ops/dense_mm.metal` + `src/ops/dense_mm.rs` (new library, compiled
lazily so nothing else gains a Metal-4 dependency), `dispatch::run_matmul_dense_q_mm` and
its dtype/geometry support matrix, `gguf::QuantPlane` + `Weights::qlinear_with_plane`,
`DenseMlp`'s prefill entry, and the `XWEN_DENSE_MM_CLASSIC` / `XWEN_DENSE_MM_MIN_SEQ`
switches wired into `parity-gate.ts`'s strip list, `referenceEnv()`, the strict candidate
env and `isReferenceDump()`. New tests: kernel-vs-oracle and kernel-vs-`QMatMul` at the
production FFN shapes, all four dtypes, the tile-edge cases (token counts across the
128-wide tile, out-dims across the 64-wide one), the instantiation-matrix and
geometry-vs-`#define` cross-checks, a `DenseMlp`-level fused-vs-classic comparison with
per-projection attribution, and the threshold-gating test above. 696 lib tests and 66
parity tests pass, 0 failures.
