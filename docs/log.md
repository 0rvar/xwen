# Engineering log

Reverse-chronological. Heading convention: `## YYYY-MM-DD — headline stating what
shipped, ideally with the number`. Same-day entries disambiguate in the heading text.
Superseded entries are marked in the headline, never deleted.

## 2026-07-29 (latest) — K-snapshot fused verify lands: spec decode goes from single digits to +8-21%, the 35B flips from -12% to +13-20%, and --draft becomes opt-out

**Context.** P9 shipped speculation as a 27B-only single-digit win with the 35B losing
12%, and its own annotation named the cause: under an armed rollback trail a multi-token
verify chunk fell back to the frozen reference scan, token by token — 39 ms per verified
position on the 27B, so the 48-of-64 DeltaNet layers got no batching win at all inside a
verify forward. TODO.md P9(a) called the K-snapshot fused verify "the precondition for
speculation to pay, not an optimization of it". This arc built it.

**What shipped.** The two fused scan kernels (`kernel_delta_scan`, `kernel_delta_scan_v2`
in `src/ops/delta.metal`) now optionally spill per-token states: a new
`ops::delta_scan_with_trail(.., state_planes)` widens the state output to
`[planes, v_heads, 128, 128]`, most-recent-first (plane s = state after token
`seq-1-s`), mirroring llama.cpp's `kernel_gated_delta_net` K>1 snapshot slots
(ggml-metal.metal:2740-2749) so the CPU oracle stays diffable. Plane 0 is the unchanged
after-loop store — at planes = 1 (every unarmed prefill and decode call) the kernel is
byte-identical to before, proven by a bitwise test. The armed clause is gone from
`linear_attn.rs`'s fused gate; an armed chunk runs the fused scan with planes = seq and
builds the rollback trail from unmaterialized plane views (delta) plus the same
host-side conv-stream slices the reference records. `XWEN_DELTA_CLASSIC=1` still routes
everything, armed chunks included, to `forward_classic` (untouched, still the frozen
oracle). `n_planes` rides the args struct, not a specialized pipeline. Details in
decisions.md "Model math" (the superseded armed-chunk entry).

**Verification.** 700 lib + 66 parity-harness tests green (4 net new; the test encoding
the old fallback was rewritten to assert the new invariant, and the new trail test was
mutation-tested — flipping the kernel's slot mapping fails it while every pre-existing
scan test stays green, so it is not vacuous). Two-model-family adversarial review
(Claude + Codex gpt-5.6-sol at xhigh): zero correctness findings; one doc-accuracy nit
each, both fixed. Both parity gates re-ran and pass with numbers identical to the
pre-change run (35B mm cos 0.999631, Δnll 0.000791; 27B all tiers) — the schema is
untouched.

**Measured** (`lowpowermode 0` — this machine exposes no `powermode` key, so high-power
is never claimed; warm, one model process at a time, interleaved arms, greedy, 128 new
tokens, `p_min` 0.3; two independent end-to-end runs, per-rep values in the raw logs).
The verify A/B is same-day and same-harness: `XWEN_DELTA_CLASSIC=1` IS the pre-P9a
verify path, and that arm reproduced the historical 245 ms @ span ~6 baseline.

- 27B verify round (`n_past` 512): fit over spans 2-32, the marginal cost fell
  **9.42 → 3.57 ms/position** (2.6x) over a fixed cost of ~171 → ~149 ms. At span 6:
  244 → 187 ms/round. In-loop `--stats` corroborate (~248 ms/8.3-position round →
  ~182 ms/7.3).
- 27B end-to-end: code **+19.3 to +21.0%** (29.7-30.0 vs 24.8-24.9 tok/s median), chat
  **+7.6 to +8.4%** — up from +4.8-6.8% / +1.5-7.4%. Acceptance 83.2% / 75.2% (down
  ~4 points from pre-P9a: the batched verify reassociates sums and accepts a slightly
  different token set; throughput improved regardless).
- 35B-A3B end-to-end: code **+18.1 to +19.8%** (124.0-126.7 vs 105.0-105.8), chat
  **+12.6 to +12.8%** — from **-11.5/-12.7%**. The mechanism is the pause controller,
  not the drafter: 35B code went 54-of-66 rounds paused → **0-of-20 paused, 159
  drafted**. The ledger's attribution of the 35B loss to the drafter cache sync (P9b)
  was measured right but read wrong — the ~1.2 ms sync is only fatal when the
  controller pauses and pays it for nothing; cheap verify made it stop pausing.
- spec-equivalence: six of eight comparisons byte-identical; the two 27B-chat forks are
  at the SAME points with the SAME words as the pre-P9a run (the known near-tie class).
  No desync — the sampled stream stays in step for 100+ tokens before forking.

**The `--draft` default flipped to opt-out** — P9(d)'s pre-registered bar ("the 35B at
or above plain, not merely closer") is met with margin on both prompt kinds in both
runs. Zero-flag `generate`/`serve` now load the dflash sidecar; `--no-draft` opts out.

**New open items** (ledgered under P9/TODO): the verify round's ~149 ms fixed cost
(~113 ms above a plain step, now ~60% of a typical round and the new ceiling — price
checkpoint/rollback/readback before attacking); `p_min`/`pause_margin` retune against
the new cost curve (0.3 was fitted to the reference-scan curve); an unexplained
superlinear jump at span 48 in every verify arm (outside the production regime —
block_size 16 caps real spans near 17 — refuted as the dense-mm threshold, cause
unknown); `spec-verify-bench.rs` shipped broken (fixture id `long-swa` predates the P7
rename to `long-mixed` — it had never run in this repo; fixed this arc). The flip's own
review (Codex, clean on the diff) surfaced two default-on consequences, both ledgered:
serve slots persisted without drafter planes now silently decode plain forever while
reporting draft ON (the common hydration path against `--no-draft`-era slots, not the
flag-change edge the code comment was written for), and a custom `--model` GGUF that
fails the drafter preflight now hard-errors at startup where it previously ran plain.

## 2026-07-29 — The 27B prefill gap was the dense FFN's gemm, not the DeltaNet scan: a Q4_K cooperative-tensor kernel takes prefill from 263 to 702 tok/s @925

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

## 2026-07-29 — DFlash adapted to the Qwen sidecars (P9): both drafters load and accept 85-95%, and speculation is a 27B-only win because the verify forward runs the per-token reference scan

**Context.** The DFlash subsystem came over from laguna whole and inert: `dflash.rs`
described a six-layer, hidden-3072, 72-head drafter with a per-head softplus attention
gate and a per-tap encoder norm, gated behind a `dflash.decoder_arch == "laguna"` check.
The ggml-org sidecars for Qwen 3.6 are a different model — 5 layers at hidden 5120 (27B)
or 6 at 2048 (35B-A3B), 32 Q / 8 KV heads, no gate tensor, no `enc.aux_norm`, no
`decoder_arch` key at all, and sliding-window attention on every layer but the last. Two
tests (`real_file_load_and_shapes`, `real_file_bf16_alias_load_and_forward`) were the
suite's only red ones and asserted the laguna geometry on purpose, as the arc's gate.

**What the oracle actually says.** Three of the graph's differences were not in the
briefed list and were found by reading `reference/llama.cpp/src/models/dflash.cpp`
against the shipped headers:

- **The noise block is non-causal.** `common/speculative.cpp:1004` calls
  `llama_set_causal_attn(ctx_dft, false)`, and the causal branch of the KV mask builder
  is guarded by `if (causal)` (llama-kv-cache.cpp:1793). The drafter is a block-diffusion
  model: it denoises `[id_last, MASK × 15]` in one forward, so every block position sees
  every other in both directions. The inherited code masked the block causally.
- **The injection path applies no `attn_norm`.** dflash.cpp:252-253 projects the raw
  encoder output into `wk`/`wv`; `enc.output_norm` is the only norm on that path. The
  query path in `draft_forward` does apply `attn_norm`, so the two deliberately disagree.
  The inherited code normed both, citing the laguna branch — which no longer exists.
- **The encoder is three ops.** dflash.cpp:109-123: concatenate the taps tap-major,
  `fc`, `enc.output_norm`. No per-tap RMS-norm, no per-tap scale.

Everything else the mapping pass had established held up: taps are the residual
*entering* the named target block, so the `t - 1` translation to our `l_out` capture
points is unchanged; rope is plain NEoX over all 128 dims (no `rope.dimension_count` key
→ `n_rot = n_embd_head_k`), theta 1e7 from the GGUF; QK-norm before rope; SwiGLU with a
real `ffn_norm` tensor.

**Sliding windows, implemented as a narrow rather than a mask.** llama.cpp masks
`p1 - p0 >= n_swa`, i.e. a query at position `p` keeps `[p - window + 1, p]` on the past
side and — being non-causal — everything on the future side. The obvious implementation
is an additive mask over the full score row, but the block's windows are a contiguous
union: query 0 has the deepest floor, and the 16 queries' floors span at most 15
positions. `attention` therefore narrows the cache to `[lo, committed + n_block)` and
masks only the ≤15 columns between the individual floors, falling back to no mask at all
while the context still fits inside the window. A windowed layer costs O(window) per
round instead of O(context), which retires half of the argument behind
`DEFAULT_DRAFT_CTX` being small: only the final full-attention layer — one of five or six
— still grows with depth.

**Measurements** (`lowpowermode 0`, warm, greedy, 128 decoded tokens, interleaved arms
within one process-per-run sweep, 3 reps, medians; scripts in the session scratchpad,
`scripts/spec-equivalence.ts` committed):

| | plain | `--draft` (p_min 0.3) | acceptance |
|---|---|---|---|
| 27B, code prompt | 25.0-25.2 tok/s | 26.4-26.7 (+4.8 to +6.8%) | 87.4% |
| 27B, chat prompt | 20.6-21.5 tok/s | 20.9-23.1 (+1.5 to +7.4%) | 65.9-73.7% |
| 35B-A3B, code prompt | 105.1 tok/s | 93.0 (-11.5%) | 81.3% |
| 35B-A3B, chat prompt | 105.5 tok/s | 92.1 (-12.7%) | 82.8% |

The 27B rows are ranges across two independent interleaved runs rather than one run's
median, because that model's run-to-run spread is wide enough to matter (docs/decisions.md
"Measurement discipline", and the 27B caveat under TODO P11): within a run the reps are
tight — 26.9/26.7/26.7 against 24.8/25/25 on the code prompt — but the level shifts
between runs. The sign is stable; the magnitude is ±2 points. The 35B rows repeated to
within 1%.

The drafter proposes well on both models — 85-95% acceptance at `p_min` 0.5-0.9, and a
27B run at `p_min` 0.9 accepted 54 of 54. The 35B's loss is not a drafting failure.

**The 35B loses ~12% before it drafts anything.** An arm with `--draft-p-min 1.1`, where
no drafter token can ever clear the threshold and 119 of 127 rounds pause, still decodes
at 92.6-92.7 tok/s against 105.1-105.5 plain — the same loss as the best drafting arm.
The cost is the drafter's per-round cache sync: every committed token runs `encode` (an
8-tap concat through a [2048, 16384] `fc`) plus six layers of `wk`/`wv` projections,
QK-norm, rope and two `slice_set`s, about 14 small Metal dispatches for ~1.2 ms. That is
12% of a 9.5 ms plain step and 2.8% of the 27B's 43 ms one, which is the whole difference
between the two models' verdicts. It is dispatch-bound, not FLOP-bound — the same disease
the MoE glue fusion cured — and it is mandatory while a drafter is attached, because a
drafter whose cache falls out of step with the target's can never resume speculating.

**The verify forward gets almost no batching win, which is the real ceiling.** Under an
armed rollback trail a multi-token chunk falls back to the frozen reference scan
(linear_attn.rs:194-205), and that scan walks tokens one at a time in candle ops. So the
layers that are 48 of the 27B's 64 and 30 of the 35B's 40 cost the same per position in a
16-token verify as in 16 single-token steps: measured 245 ms for a ~6-position verify on
the 27B against a 43 ms plain step, i.e. 39 ms per verified position. Speculative decoding
is a bet that verifying N tokens costs far less than decoding them one at a time; on this
architecture that bet currently pays only in the attention and FFN layers. **The
K-snapshot fused verify is therefore not an optimization of P9 but the precondition for
it** — TODO.md P9 carries it with the structural note that both scan kernels already hold
each thread's state slice in registers across the timestep loop, so emitting per-token
snapshots is one guarded store plus a wider output buffer.

**Tuning.** `draft_p_min` swept over {0.2, 0.3, 0.5, 0.7, 0.9} on the 27B: 0.3 is the only
value that came out ahead of plain on both prompt kinds in every run (0.5 lost on the chat
prompt twice), so the default moves 0.5 → 0.3 in `SpecParams`, the CLI and the serve
config. Lower `p_min` drafts longer at lower acceptance, and that wins precisely because
the verify cost is near-linear in its span — with no batching penalty to pay, a longer
span amortizes the round's fixed cost. That reasoning is fitted to the reference-scan cost
curve and should be re-run when the fused verify lands. `pause_margin` stays 1.0: the
controller earns it on the 27B, where always-drafting (`--draft-pause-margin 0`) measured
21.8 tok/s against the controller's 25.8 and plain's 23.3. It cannot help the 35B, whose
loss is charged to rounds the controller has already paused.

**`--draft` stays opt-in.** The flip to opt-out was conditional on the controller holding
a never-materially-slower property on both models. It does not: no setting of `p_min` or
`pause_margin` can recover a 12% loss that is incurred on paused rounds. The CLI, serve
config and `--init` template text all lose their "not adapted yet, fails at load" wording
and gain the measured reason instead.

**Equivalence, in two modes.** `scripts/spec-equivalence.ts` diffs `--draft` against
`--no-draft`. Greedy mode (temperature 0) checks the verify walk's token selection; it
found 11 of 12 comparisons byte-identical, the twelfth forking on the 27B chat prompt at an
adjective ("accessible" vs "educational"). That is the batched verify forward reassociating
its f32 sums differently from the single-token forward and flipping a near-tie — the same
class the decode parity tier's near-tie rule grades, and the same one already recorded for
the fused delta scan under P8a.

Greedy mode has a structural blind spot, though, raised by the second-family review: at
temperature 0 the argmax path never draws from the seeded RNG, so no amount of greedy
agreement can show that the spec loop advances the SAMPLER STREAM the same number of times
the plain loop does — one extra or missing draw would reroute every subsequent token. A
`sampled` mode now covers it: temperature 0.8 at a fixed seed with `--draft-p-min 0` and
`--draft-pause-margin 0`, so every round drafts a full block and nothing pauses (auto-pause
is what makes a temperature>0 run irreproducible from a seed, so it has to be off).
Result: the 35B is byte-identical on both prompts with 360 and 435 drafted tokens over 384
and 464 verified positions, and the 27B is identical on the code prompt with 315 drafted.
**The sampler stream is in lockstep.** The 27B chat prompt forks in sampled mode too, at
line 12 — deep enough that the stream was demonstrably in step for over a hundred tokens,
which is the near-tie signature rather than the desync one (a desync reroutes the first
sampled token, and the script says so when a sampled-mode fork lands on line 1).

Both modes now also refuse to report OK on a run that drafted nothing: a comparison that
paused into plain decoding exercises no verify and no rollback, so its agreement means
nothing, and the script says NO COVERAGE instead. It rebuilds before comparing and checks
the binary is not older than the newest source, so it cannot bless a stale build.

**Deleted, added, changed.** Deleted: the `decoder_arch` requirement, the `enc.aux_norm`
tensor and its per-tap norm-and-scale, the `attn_gate` tensor and its softplus output
gate, the `softplus` helper, and the within-block causal mask. Added: `sliding_window` +
`swa_layers` on `DflashConfig` with a `layer_window(il)` accessor, `value_bool`, the
narrow-plus-mask windowed attention, `Model::draft_kv_bytes_per_token()` (40 KiB/token on
the 27B's five layers, 48 on the 35B's six — `serve/config.rs`'s hardcoded 35B-only
constant now derives from it), and `scripts/spec-equivalence.ts`. `attach_drafter` also
gained the mismatched-pairing check the CLI path never needed while every drafter load
failed: nothing in a sidecar's metadata names its target, so `--draft <path>` can pair the
27B drafter with the 35B, which used to reach `set_spec_taps`'s `assert!` and panic. It is
now an error naming both numbers, and the hidden-size mismatch is caught at attach rather
than at the first forward. Review caught that serve had the same hole one level earlier:
`check_draft_geometry` compared head counts but not hidden size, and the tap check alone
does not separate the two sidecars in one direction — the 35B-A3B drafter's translated
taps top out at 37, inside the 27B's 64 layers — so `xwen serve --model-size 27b --draft
<35B drafter>` passed startup validation and failed the first job instead. The check is
now threaded through `read_draft_config` from the target's `XwenConfig`, which is what
`validate_model` exists to do. Tests: the two red ones
rewritten against real sidecar geometry and parameterized over both models (the alias test
now injects 2100 positions, past the 27B's 2048 window, so the narrow-plus-mask path runs
on real weights), plus three new ones — a windowed forward graded against the scalar
reference and against an unwindowed twin, a perturbation test proving the last block row
informs the first, and a config test for the window keys. `ops/bf16.rs`'s `DRAFTER_SHAPES`
covers both sidecars' twelve production matmul shapes. Suite: **760 passing, 0 failing**,
the two deliberately-red tests among them.

## 2026-07-29 — The DeltaNet scan is 3% of 27B prefill: llama.cpp's decomposition measured slower, and the premise behind P8b refuted

**Context.** The head-to-head entry below named the sequential DeltaNet scan as the cause
of the 27B's 1.8-2.1x prefill loss, and a mapping pass then established that llama.cpp's
Metal path does NOT run the chunked form it advertises — its fused
`ggml_gated_delta_net` op pre-empts the chunked graph (delta-net-base.cpp:437-446), so it
runs the same sequential scan we do, under a far more parallel decomposition. That made
re-decomposing our kernel the obvious lever and demoted P8b's chunked scan. This arc
built the re-decomposition. It lost, and finding out why cost the premise as well.

**Built.** `kernel_delta_scan_v2` is llama.cpp's shape adapted to our layouts: one
SIMDGROUP owns one state value-column for the whole T loop, both key-dim contractions
collapse to `simd_sum`, no barrier appears anywhere in the timestep loop, and the grid is
1536 threadgroups at the 27B geometry against the shipped kernel's 192. It needs q and k
pre-normalized, so `kernel_delta_l2norm` hoists the L2 clamp-norm out into its own
dispatch. Both are bounded against the frozen reference scan exactly as the shipped
kernel is, at the same tolerances — no parity schema change, floors untouched.

**Measured, and it lost.** Isolation timing per DeltaNet layer (`delta_scan_timing`,
plateau ms, each arm inclusive of its q/k norm), and interleaved end-to-end (median of 3
rounds, arm order flipped each round, full power `lowpowermode 0` start and end,
`XWEN_BENCH=1`, `--no-draft`, committed fixtures):

| isolation, one layer | 27B @880 | 27B @4096 | 35B @880 | 35B @4096 |
|---|---|---|---|---|
| shipped | 1.97 | 8.56 | 1.57 | 6.31 |
| llama.cpp shape | 2.73 | 14.81 | 1.88 | 8.93 |

| end to end, tok/s | 27B shipped | 27B v2 | 35B shipped | 35B v2 |
|---|---|---|---|---|
| prefill @880 | 296.9 | 307.6 | 2504.9 | 2497.2 |
| prefill @3851 | 262.1 | 257.7 | 2312.3 | 2398.6 |
| decode n=256 @630 | 22.0 | 22.1 | 103.3 | 102.2 |

The end-to-end arms are a tie in both directions — which is the actual finding, not a
measurement failure. **The scan is 3% of 27B prefill.** 48 layers × 1.97 ms is 95 ms of a
2.96 s prefill at 880 tokens; × 8.56 ms is 411 ms of 14.2 s at 3851. A free scan buys
~297 → ~307 tok/s against llama.cpp's 486. The 1.7x isolation regression at 4096 shows up
end-to-end as exactly the 1.7% it should, which is the cross-check that the share is
right. The mechanism of the loss is q/k read amplification: our threadgroup stages one
normalized q and k vector per threadgroup per timestep, llama.cpp's shape gives every
simdgroup its own copy — 32x the L2 traffic — and at head dim 128 each lane owns 4 state
entries, so two `simd_sum`s ride on 8 useful FMAs. Hoisting the norm alone loses too: the
extra dispatch rewrites the whole q|k plane (0.52 / 1.80 ms) to save a reduction already
amortized over a threadgroup.

**Shipped: nothing.** The scan kernel is byte-identical to what it was before this arc,
which both gates confirm by reproducing every number exactly (35B strict 1.000000, mm
0.999631, decode 63/63/62 with 0 mismatches, ppl Δnll 0.000791; 27B strict and mm both
1.000000, decode 64/64/64 with 0 mismatches, ppl Δnll 0.000330). `parity-gate.ts`'s
`baseEnv()` strips `XWEN_DELTA_SCAN_V2` alongside the other presence-based switches —
unlike its siblings it has no provenance field, so a stray shell value would have applied
to both sides and passed the gate while grading the wrong kernel. The two new kernels stay
behind `XWEN_DELTA_SCAN_V2=1` as a runnable refutation — the kernel they mirror is
vendored in this repo and will invite the same proposal again — along with
`ops::delta_l2norm`, its bounded test, and the `delta_scan_timing` bench that produced
the table. 681 lib tests pass with the two dflash `real_file` tests staying deliberately
red, 63 parity tests pass (3 ignored, the ones the gate feeds), and the delta and
linear-attn suites pass under `XWEN_DELTA_SCAN_V2=1` as well as under the default.
`scan_matches_reference` picked up seq 67 — prime, and a multiple of no tile or simd
width the scan is built from.

**What this hands off.** The 27B prefill gap is still ~1.8-2.1x and is now known NOT to
be in the DeltaNet layers. It is in the dense projections; that is the next arc's
question, and it should start from a per-stage profile rather than from a reading of
llama.cpp's kernels. P8b's chunked scan keeps its rollback-replay rationale and loses its
prefill bounty (see TODO.md and decisions.md, "The DeltaNet scan decomposition").

## 2026-07-29 — Two-family review of the MoE-glue + top-p diff: kernels clean twice over, one real hole in the gate script

**Context.** Standard post-arc review of what became commit `bec5fa2`, one Claude
reviewer and one outside-model pass (Codex CLI, gpt-5.6-sol, xhigh effort) over the full
diff. Both independently cleared the fused kernels line by line — barriers, the bitonic
network's pad ordering, the epilogue's fp pragmas, wrapper validation. The findings that
survived were all at the edges.

**Fixed.** (1) `parity-gate.ts`'s `baseEnv()` stripped every presence-based kernel
switch EXCEPT the two new ones, so a stray `XWEN_MOE_GLUE_CLASSIC` in the shell (even
`=0`) would have silently classic-ed both sides and passed the gate without dispatching
either new kernel. Both are stripped now, and the 35B gate was re-run under an
explicitly clean env: ALL PASS, every number identical (strict 1.000000, mm 0.999631,
decode 63/63/62 with 0 mismatches, ppl Δnll 0.000791) — which also retroactively
validates the pre-fix run. (2) `top_p == 0.0` — reachable through the serve layers —
had no test; it now pins the keep-one behavior against the llama.cpp oracle.

**Documented, not changed.** The top-p cut is one ulp off llama.cpp at exact f32
boundaries (they re-softmax survivor logits, we divide full-softmax survivors by their
sum — algebraically equal, reproduced counterexample in decisions.md), and the
`llamacpp_filtered` oracle is structurally blind to that class. **Ledgered.** `top_k=0`
semantics (greedy here, disabled there), the unpinned
`MTLMathFloatingPointFunctions` compile axis under the epilogue's bare `exp` (the
bitwise suites are the tripwire), and `mul_mv_id_dual`'s trusting wrapper. Codex's one
discounted finding — the ragged q4_K tail overread — is the documented, deliberate
ggml-matching inheritance, predating this arc.

## 2026-07-29 — First llama.cpp head-to-head: xwen wins decode on both models, loses 27B prefill 2x to the sequential DeltaNet scan

**Context.** Three perf arcs (sampler tail, fused DeltaNet, fused MoE glue) had moved
xwen's own numbers, but nobody had ever measured llama.cpp on this machine. Same GGUF
files fed to both engines (the blessed Q4_K_M pair), llama.cpp at the pinned oracle
build e9fa078 with `-fa 1`, strictly interleaved arms per the measurement-discipline
entry, 6-7 reps per cell, decode at matched context depth 630. Power mode: AC,
`lowpowermode 0`, no `highpowermode` key — Automatic, the whole run.

**Results** (median tok/s, ratio = xwen/llama.cpp):

| cell | llama.cpp | xwen | ratio |
|---|---|---|---|
| 35B decode tg256@630 | 98.0 | 103.1 | 1.05 |
| 35B prefill @925 / @4096 | 2725.7 / 2497.9 | 2546.6 / 2316.6 | 0.93 / 0.93 |
| 27B decode tg256@630 | 19.2 | 19.6 | 1.02 |
| 27B prefill @925 / @4096 | 486.2 / 501.8 | 268.9 / 235.6 | 0.55 / 0.47 |

**Reading.** Decode is won on both models — the design target metric. The 35B prefill
deficit is mostly thermal-boost asymmetry: llama.cpp's early reps boost to ~3140 and
settle to ~2500-2700 (-17%) while xwen only drifts -5%; restricted to settled reps the
ratios are 0.96 @925 and 0.90 @4k, i.e. near parity at 925. The 27B prefill gap is
real, large, and length-growing: xwen *degrades* 269→236 from 925 to 4k while llama.cpp
*improves* 486→502. The cause is the one CLAUDE.md's P8 note predicted — the 27B runs
48 sequential DeltaNet reference scans at inner width 6144 (vs the 35B's 30 at 4096),
and llama.cpp prefills those layers with its chunked (chunk=64) form. P8b (chunked
scan) is the named fix and now has a measured bounty: ~2x on 27B prefill.

**Sampler asymmetry checked.** llama-bench's tg loop runs no sampler; xwen's decode
number carries full 1.0/0.95/20 sampling. Greedy control run: 104.9 vs 103.1 — the
sampler costs ~1.5%, so the decode win is not an artifact of what each side skips.

**Also confirmed.** The 2026-07-29 MoE-glue numbers reproduce independently (35B decode
103.1 vs 102.8 quoted, 27B 19.6 vs 19.0). Raw per-rep data and harness scripts are in
the session scratchpad (results-{35b,27b}.txt), not the repo.

## 2026-07-29 — Top-p switched to llama.cpp's renormalizing convention

**Context.** The 2026-07-28 sampler rewrite kept candle's top-p rule — cut against
full-vocabulary mass, skip the cut entirely when the top-k set holds less than `top_p`
of the total — and ledgered the divergence rather than fixing it inside a perf change.
llama.cpp is this project's ground truth for everything else, and `--top-p 0.95` did not
mean what a llama.cpp user would read it as.

**Change.** `truncate_top_p` is now `llama_sampler_top_p_apply`. `top_p >= 1.0` returns
without touching anything (llama.cpp builds an empty sampler there); otherwise the top-k
survivors are renormalized to sum to one and the walk keeps the shortest prefix whose
cumulative mass reaches `top_p`, comparing `cum_sum >= top_p` with the crossing token
INCLUDED. `min_keep` is not carried — llama.cpp's default is 0, disabled, and the loop's
own at-least-one guarantee is all that default gives. The two branches of
`candidate_set` collapsed into one: the "is the top-k mass already under `top_p`" guard
existed only to express the absolute-mass rule and has no analogue here. The
renormalization is CPU-side over the ≤20 candidates and costs nothing; the device-side
full-vocabulary softmax stays for now.

**What this changes.** Sampled outputs, one-directionally: renormalizing cuts the same
or more, never less, so a seeded stochastic run draws a narrower candidate set than a
pre-2026-07-29 build wherever the cut bites. Accepted. Greedy is untouched and the
parity gate is greedy end to end, so nothing there moves. A side effect worth naming:
the cut no longer compares absolute mass, so the softmax denominator divides back out of
it, and the device fast path and the CPU `SampleControl` path can no longer truncate
differently by an ulp at the threshold.

**Tests.** `top_p_measures_absolute_mass_not_renormalized_mass` is replaced by
`top_p_measures_mass_renormalized_over_the_candidate_set`, built on two hand-computed
rows where the conventions give different answers — one where the old rule's cut ran and
stopped short, one where it was skipped outright because the top-k set held only 0.76 of
the mass. A `llamacpp_filtered` transcription joins the candle transcription as an
oracle, and the candle-equivalence matrix now claims candle agreement only at `top_p`
1.0, where no cut applies; below that it is llama.cpp the candidate set is checked
against, at every shape and every p. 29 sampler tests pass, clippy clean.

**Not done.** The perf half of the ledger item — the fast path no longer needs a
full-vocabulary softmax, since renormalizing over k survivors IS a k-wide softmax — is
split out as its own TODO entry. It is gated on a Metal top-k, not on this change.

## 2026-07-29 — Fused MoE glue: an MoE layer goes from 24 dispatches per token to 14, and 35B decode from 92.6 to 102.8 tok/s

**Context.** With the DeltaNet layers fused, the 35B-A3B's remaining decode cost is the
MoE half, and it was launch-bound rather than bandwidth-bound: 24 dispatches per layer
at seq==1, 960 per token across the 40 layers, of which exactly 8 are real matmuls.
Everything else is glue — a softmax, an arg-sort, a gather, a sum, two clamp halves that
each upload a fresh 4-byte scalar buffer, a divide, a sigmoid, three elementwise
multiplies and three adds.

**Change.** Three fusions, each bit-identical to the candle chain it replaces, behind one
kill-switch (`XWEN_MOE_GLUE_CLASSIC=1`).

`kernel_moe_router` (new `src/ops/moe_glue.metal`) takes the router logits and returns
the selected ids and their renormalized weights in one threadgroup per token, replacing
seven dispatches. `kernel_moe_epilogue`, in the same file, computes
`Σ_k down[k]·w[k] + shexp·sigmoid(gate)` in one pass over the uncombined down
projection, replacing the weighted combine, the shared-expert gate sigmoid, its
broadcast multiply and the routed+shared add — four dispatches. And the shared expert's
`silu(g)*u` now calls the existing vendored `ops::silu_mul` instead of candle's two
elementwise passes. The routed experts' `ExpertFfn` grew a `project` method returning
the down projection BEFORE the combine, so the epilogue can own the combine; the
reference oracle declines it (it scatters as it goes) and so does the f16-tile prefill
branch (its projection carries an L2 rescale the epilogue has no term for), and both
fall back to the untouched classic composition.

**The router matmul stays candle's, deliberately.** It lowers to MLX's
`gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4`, whose K-partition is strided across lanes
(thread t owns `{i·32 + 4t + tm}`) and whose combine is a specific shuffle-down tree —
reproducible only by lifting the kernel verbatim, and even then resting on the compiler
contracting `result += vc * inter` to an fma the same way. Worse, its accumulation order
depends on the output width, so concatenating the shared-expert gate row onto the router
weight to save a dispatch would have changed that gate's bits. Both matmuls stayed
candle's; the fusion starts at the logits. This is branch (b) of the briefing's decision
tree, and it costs one dispatch out of the ten saved.

**Bit-identity held everywhere, including the two places it looked unlikely.** candle's
`softmax_last_dim` is not a max/exp/sum triple but the online Welford form, with `fast::exp`
in the merge, `fast::divide(1, d)` hoisted once per thread, and a five-step
`simd_shuffle_down` tree over `MD{m, d}` pairs (the merge op is not a built-in simd op,
so candle takes the shuffle tree, not `simd_sum`); the kernel reproduces its
compile-time BLOCKSIZE switch over a runtime width. And candle's Metal arg-sort is
llama.cpp's bitonic network, which is deterministic but NOT stable — its comparators are
strict and never consult the index, so equal probabilities do not come out in ascending
expert order. That matters more than any rounding here: a tie flip swaps an entire
expert. The network is reproduced verbatim rather than replaced by a selection
algorithm, and `router_ties_match_candle_bitwise` pins it against candle's on four
tie families (all-equal, a twelve-way plateau with eight slots to fill, a
sixteen-value block repeated across the row, and a single winner over thirty-two tied
runners-up).

**Perf.** 35B-A3B Q4_K_M, `lowpowermode 0`, warm, batch 1, `--no-draft`, 630-token
prompt, 256 decoded tokens. Interleaved arms, median of 5, per docs/decisions.md
"Measurement discipline":

| | classic | fused | |
|---|---|---|---|
| decode | 92.6 tok/s | 102.8 tok/s | +11.0% |
| prefill 925 | 2475.1 tok/s | 2488.9 tok/s | +0.6% |
| prefill 4k | 2101.4 tok/s | 2105.8 tok/s | +0.2% |

The decode arms never overlap across the five reps (classic 91.9–93.7, fused 102.2–104.3),
and an earlier independent run on a cooler machine put the same ratio at +11.1%. Prefill
is unmoved, as it should be: above `MM_ID_MIN_SEQ` the epilogue declines and only the
router kernel and the shared expert's activation fuse, against a chunk that is
compute-bound rather than launch-bound.

**A fourth fusion was built, measured, and switched off.**
`kernel_mul_mv_id_q4_K_f32_dual` computes the gate and up expert matvecs and their
SwiGLU activation in one dispatch instead of three, and is bitwise against that chain —
it runs the same `mul_mv_q4_K_sums` body twice, factored out of the verbatim ggml
`mul_mv_q4_K_impl` so neither projection's accumulation order moves. It measured
SLOWER: 99.5 against 102.8 tok/s decode, five interleaved reps apart with no overlap. It
is not register pressure — both pipelines report `max_total_threads_per_threadgroup` of
1024. It is the grid: two independent `_v` dispatches expose twice the threadgroups, and
a bandwidth-bound gather turns that into outstanding memory requests, so folding them
into one grid that walks both weight planes serially per thread costs more than the two
saved launches buy. Kept behind an opt-in `XWEN_MOE_DUAL=1` as a measured artifact worth
re-pricing on another device, not because anything runs it.

**Parity provenance needs no schema change, and that is the point of the bitwise work.**
Every shipped kernel here reproduces its chain's rounding boundaries exactly, so there
is no path for the fused/classic split to change a dump — no new provenance field, no new
pin, no grandfather clause. Both gates confirm it, reporting numbers identical to the
pre-change run: 35B ALL PASS (strict cos=1.000000, mm cos=0.999631, decode 63/63/62 of
64 with 0 mismatches, ppl Δnll=0.000791), 27B ALL PASS (strict and mm cos=1.000000,
decode 64/64 on all three fixtures, ppl Δnll=0.000330). The dense 27B never enters this
code — `DenseMlp` keeps the free `swiglu` helper untouched — and was re-gated only
because the ledger says a shared file was edited.

`XWEN_MOE_GLUE_CLASSIC=1` joined `parity-gate.ts`'s `referenceEnv()` on hygiene, so the
oracle stays a pure candle composition, and it is deliberately NOT on the strict
candidate: the strict tier now grades the FUSED glue against a reference that ran the
unfused one. Regenerating the reference under the new pin (`--regen-ref`) and re-running
returns `cos=1.000000 top5=5/5` with every other tier unchanged — the full-model
confirmation that the ops-level bitwise tests generalize past synthetic shapes.

**Tests.** 679 lib tests pass (the two deliberately-red dflash `real_file` tests
unchanged), plus 63 in the parity binary. Eleven are new: eight in `ops::moe_glue`
(bitwise router across eight geometries including a padded bitonic network and a softmax
width below candle's shared-memory threshold, the tie families, offset views, a clamp
whose floor actually binds, the bitwise epilogue across three sequence lengths with gate
logits in both sigmoid saturation tails, a `#define`-vs-Rust geometry cross-check, and
the shape/dtype contracts), two in `ops::mv_id` for the dual kernel, and
`moe::fused_block_matches_candle_bitwise`, which runs a whole synthetic MoE block
through `MoeBlock::forward` and compares it against a reference built entirely from
candle ops — the wiring test, where a swapped operand or a gate that picked up a stray
sigmoid would show up and the per-kernel tests would not.

## 2026-07-28 — Fused DeltaNet kernels: a layer goes from ~65 dispatches per token to 8, and 35B prefill from 305 to 2183 tok/s

**Context.** Three of every four Qwen 3.6 layers are gated DeltaNet, and all of them
ran the P3 reference: composed candle ops, one scan step per token. Decode spent about
65 Metal dispatches on a layer containing seven matmuls. Prefill was worse in kind, not
just degree — the scan is a Rust `for t in 0..seq` loop issuing eight dispatches per
timestep per layer, so a 512-token chunk on the 35B cost roughly 123k dispatches. This
is P8's decode-side package. The chunked (chunk 64, tri-solve) scan was explicitly out
of scope and stays open as P8b.

**What landed.** `src/ops/delta.metal` + `src/ops/delta.rs`, four kernels wired into a
`forward_fused` beside the untouched reference (renamed `forward_classic`, math
verbatim — it is still the oracle):

- `kernel_delta_conv` — causal depthwise conv, silu, and the next conv window, reading
  the carried window and the fresh qkv rows as two buffers. That kills the `cat`, and
  writing the window directly kills the `zeros_like` + `slice_set` materialization.
- `kernel_delta_ba` — `beta = sigmoid(b_raw)` and `g = ssm_a * softplus(a_raw +
  dt_bias)` from ONE `[hidden, 2·v_heads]` projection, built at load time by
  concatenating `ssm_beta` and `ssm_alpha`. Two gemvs become one. It emits the LOG
  decay and lets the scan exponentiate, folding away another pass.
- `kernel_delta_scan` — the whole recurrence, all T timesteps, one dispatch.
- `kernel_delta_gnorm` — the gated output RMSNorm.

Eight dispatches per layer at any sequence length.

**The scan's shape is the whole trick.** Value-dim columns of the state are completely
independent — `sk[j]`, `d[j]`, the rank-1 update of column `j` and `o[j]` all touch only
column `j` — so the only cross-thread folds are the two key-dim contractions. A
threadgroup owns one V-head and 32 of its 128 columns; thread `(r, jl)` holds 32 state
rows of one column IN REGISTERS for the entire scan. The state is read once and written
once no matter how long the chunk is, which is what turns prefill's per-timestep
dispatch storm into a single launch. Consecutive threads in a simdgroup share `r` and
take consecutive `j`, so both state passes are contiguous 32-float runs. q and k are
read straight out of the conv output with the tiled K-head mapping (V-head `h` reads
K-head `h % k_heads`) and L2 clamp-normalized in the load stage, so the reference's
materialized tile-and-broadcast disappears too.

**Measured, warm (`XWEN_BENCH=1`), batch 1, greedy (`--temp 0 --seed 7`), `-n 128`.**
Power: `pmset -g` reports `lowpowermode 0` and exposes NO `highpowermode` key on this
machine, so low-power mode is confirmed off but the High Power tier is neither
confirmed nor available — do not read these as laguna's "full power" anchors. Protocol:
fused and classic runs INTERLEAVED, three reps per arm, median reported — see the
measurement-discipline note below, which is the reason these numbers are not the ones a
naive sequential matrix produces.

35B-A3B Q4_K_M:

| prompt | prefill classic | prefill fused | decode classic | decode fused |
|---|---|---|---|---|
| 596 tokens | 305.4 tok/s | **2183.2** (7.15x) | 57.8 tok/s | **91.2** (1.58x) |
| 1929 tokens | 300.3 tok/s | **2274.1** (7.57x) | 56.6 tok/s | **88.0** (1.55x) |

27B dense Q4_K_M, which had no perf numbers at all before today:

| prompt | prefill classic | prefill fused | decode classic | decode fused |
|---|---|---|---|---|
| 596 tokens | 77.3 tok/s | **290.4** (3.76x) | 14.3 tok/s | **19.0** (1.33x) |
| 1929 tokens | 77.9 tok/s | **209.3** (2.69x) | 14.3 tok/s | **17.9** (1.25x) |

The 27B's decode gain is smaller (1.25-1.33x vs the 35B's 1.55-1.58x) because 64 dense
SwiGLU layers at hidden 5120 dominate its per-token budget; dispatch count was never
its problem. Its per-rep spread is also visibly wider than the 35B's — the 35B's
classic arm repeated at 303.2/305.4/305.6 tok/s prefill, while the 27B's fused 596-token
decode walked 21.7/19.0/17.9 across its three reps as the machine heated.

**A measurement finding worth more than the numbers.** The first pass of this matrix ran
the eight cells sequentially and reported the 27B at 13.9 tok/s decode and 131.3 tok/s
prefill. A cooled, interleaved re-run of the identical binary put the same cells at 19.0
and 290.4. The whole matrix drifts 20-35% slower over roughly ten minutes of continuous
GPU load, uniformly across BOTH arms and both checkpoints, and `pmset -g therm` records
nothing while it happens — the only tell is the control arm moving too. So a sequential
A/B on this machine is not an A/B; the arms have to be interleaved. Recorded in
decisions.md "Measurement discipline", along with the sibling trap found the same hour:
`pgrep -f "logits-dump"` matches the argv of the shell running the check, so it reports
a model process that does not exist (this bit parity-gate's own preflight too — use
`pgrep -x`).

**This is the first vendored kernel family that is not bit-identical, and that changed
the parity gate.** Every earlier fused kernel reproduced its candle chain's rounding
boundaries exactly, so its `*_CLASSIC` pin was pure provenance discipline. The scan
cannot: the reference contracts k and q against the state with a candle gemm and
normalizes q/k with a candle reduce, and the kernel partitions both across threads —
that is the point. Reassociating an f32 sum is not something a kernel can undo. So
`XWEN_DELTA_CLASSIC=1` is now pinned on BOTH sides of the strict tier (with the fused
scan on, strict stops being a bitwise tier), and a `delta` provenance field
(parity_schema v6, grandfather `classic`) proves which path each dump ran. The cached
reference dumps were written at v5 and grandfather correctly — every reference in both
gate runs was reused, no regeneration.

The other three kernels ARE bitwise, using block-scope `fp contract(off)` /
`reassociate(off)` so the scan stays free to contract into fma — its two inner loops
are the entire prefill cost, and file-scope pragmas (the sibling glue files'
convention) would have doubled their instruction count.

**Parity: ALL PASS on both checkpoints.** 35B-A3B — strict 1.000000, mm 0.999631
(was 0.999540 pre-change: slightly *better*), decode 63/64, 63/64, 62/64 agreements
with 1, 1 and 2 excused near-ties and zero non-excused mismatches, ppl Δnll 0.000791.
27B — strict 1.000000, mm 1.000000, decode 64/64 on all three fixtures with zero
excusals, ppl Δnll 0.000330.

The 35B result reproduced **three times across two independent builds**, matching to
every digit each time — twice here, and once accidentally on the parity owner's build
(an unguarded `import.meta.main` in parity-gate.ts fired a whole four-tier run as an
import side effect, since fixed). The fused kernels are deterministic: same dumps, same
cosines, same agreement counts, same Δnll, regardless of who compiled them.

The one number that moved in the wrong direction is perplexity, and it is worth stating
precisely because it is the only place the kernel's fidelity cost is visible at all.
The 35B's ppl delta went 0.000511 → 0.000791 and the 27B's 0.000221 → 0.000330 —
+55% and +49%, proportionally the same on two different architectures. And the SIGN is
systematic: the candidate is worse (higher NLL) in all four measurements, so this is
bias, not symmetric rounding noise. Everything else about the two candidates was
identical, which makes the attribution clean — that is a real, measured cost of the
fused scan, still comfortably inside the 0.002 bound.

The floor stays at 0.002 (parity owner's call, and the right one): `max(3 × measured,
0.002)` is a one-time floor-SETTING heuristic anchored to the reference-scan baseline,
not an invariant to maintain against whatever the candidate currently measures.
Re-fitting it to 0.000791 would widen the bound to fit the change under test, and a
bound re-fitted to each new implementation ratchets outward forever and catches
nothing. So the constant deliberately no longer reproduces from `3 × measured` — it is
tighter and more sensitive than the recipe would now give. Trip-wire: from 0.000791, a
further ~2.5x rise fails the gate. Since the 35B mm cosine went the OTHER way
(0.999540 → 0.999631), perplexity — not cosine — is the number to watch on further
DeltaNet kernel work.

The 35B's long-mixed fixture also picked up two near-tie excusals it did
not have before (margins 0.0161 and 0.1391 — both would clear even the standard 0.5
window), which is exactly where you would expect them: long-mixed is the fixture that
carries the DeltaNet recurrence over 600+ tokens.

**Greedy output is not preserved at longer prompts, by construction.** At a 596-token
prompt, fused and classic produce byte-identical 128-token greedy continuations on both
models. At 1929 tokens the 35B shares 69 words and then forks at a near-tie; the 27B
stays identical at both lengths (dense, no router, far less tie-prone). This is the
expected consequence of reassociated f32 sums, not a kill-switch bug — and it is why
the decode tier grades against the llama.cpp-anchored oracle with a near-tie rule
rather than against the previous build.

**One deliberate carve-out.** A `seq > 1` chunk under an armed rollback checkpoint
stays on the reference scan: the one-dispatch scan can only report the state after the
LAST token, and an armed layer needs one per token for the trail. Single tokens still
take the fused path even when armed, since their only state IS the final one — so spec
decode's per-token verify steps keep the win and only a batched verify forward pays.

**Tests.** 660 lib pass / 2 known-red (the dflash `real_file` pair, P9's), 63 parity
pass (was 60). Eleven new: four per-kernel tests at both shipped geometries (16/48 and
16/32 at head dim 128) with the conv and beta/decay ones asserting `f32::to_bits`
equality against the reference chain, a no-mutation and a streaming test for the scan,
shape/geometry rejection, offset-view handling, a block-level fused-vs-reference test
that grades the kernels as a package, and three parity rejection tests for the new
`delta` pin. The scan test was mutation-checked: flipping the K-head mapping from tiled
to interleaved moves its relative L2 from ~1e-6 to 1.37.

**Hardening follow-up, 2026-07-29.** Guards, an assertion mechanism, and doc
corrections on top of the kernels above. Nothing here moves a computed value; the
kernel math, barriers and indexing were re-read and stand.

- *Geometry, asserted twice over.* The scan's threadgroup shape lived as `#define`s in
  delta.metal (which index the state slice) and as independent Rust constants in
  dispatch.rs (which size the grid), with nothing tying the two together — drift meant
  silent out-of-bounds device writes. delta.metal now carries `static_assert`s for the
  three relations, and `scan_geometry_matches_metal` parses the `#define`s out of the
  source and compares them against dispatch.rs's copies.
- *Simd width.* Both the scan (`red[2][DELTA_D/32]`, indexed by simdgroup index) and the
  gated norm assume 32-wide simdgroups. `check_delta_simd_width` reads
  `threadExecutionWidth` at pipeline setup, so a device that ever differed fails at load
  instead of quietly folding the wrong lanes.
- *Empty chunks.* `delta_ba` and `delta_gnorm` accepted `seq == 0` and encoded a
  zero-dimension grid; they now bail like `delta_conv` and `delta_scan` already did.
- *Provenance is observed, not assumed.* `LinearAttnBlock::forward` also falls back to
  the reference scan on a non-128 head dim, a non-Metal device, or an armed multi-token
  chunk — none of which the environment shows — so an env-derived `delta` field could
  stamp "fused" on a dump that never dispatched a delta kernel, and the bounded tiers
  would grade the reference against itself and pass on nothing. Two `AtomicU64`s now
  count layer forwards down each path (`linear_attn::delta_path_counts`), and
  logits-dump derives the field from them, refusing to write a dump whose observed path
  contradicts the environment or splits across both. The field's value vocabulary is
  unchanged, so parity_schema and the gate's checks are untouched.
- *Docs that overclaimed.* TWO of the four kernels are bitwise, not three: the gated
  norm reassociates its sum of squares through `simd_sum` and grades at 2e-6, in the
  same class as the scan. The `math_mode(fast)` pragma pins the math mode but NOT the
  fast-vs-precise math-function compile option (this source compiles with no
  `MTLCompileOptions`), so what holds the intrinsics to candle's rounding is the pair of
  on-device bitwise tests, not the pragma. `ba_matches_reference_bitwise`'s decay
  assertion exponentiates through candle on both sides, so it grades `g` — not the
  scan's fast-math `exp`. And docs/parity.md's manual env table omitted
  `XWEN_DELTA_CLASSIC=1` from the reference and strict-candidate rows, so the documented
  by-hand procedure would have failed the tier the script passes.

One new lib test (`scan_geometry_matches_metal`), taking the delta family to nine, plus
four empty-chunk rejections folded into `shape_and_geometry_errors`; 63 parity
unchanged, since the `delta` field's value vocabulary did not move. The `static_assert`s
were mutation-checked out of band: halving `DELTA_TG_COLS` fails two of the three at
compile time.

## 2026-07-28 — Sampler tail: 0.82 → 0.41 ms/token, by moving the softmax off the CPU

**Context.** The per-token sampling tail was suspected of costing multiple milliseconds
of the ~16.9 ms decode budget. The bench that would have shown it (`sampler_decode_bench`
in moe.rs) was still carrying laguna's shapes — hidden 3072, vocab 100352, top-k 10,
47 MoE layers — so it had never measured Qwen geometry. Fixed first: the whole
`decode_bench` constant block now reads 35B-A3B (hidden 2048, expert_ff 512, top-k 8,
vocab 248320, 40 MoE layers), and the tiled `[VOCAB, HIDDEN]` synthetic tables tile at
512 rows because 1024 does not divide 248320.

**Measured, then fixed.** Baseline at real width: 0.819 ms/token for the whole draw, of
which 0.204 ms is the GPU→CPU copy of the 993 KB logit row and 0.615 ms is CPU work.
The CPU work was candle `LogitsProcessor::sample_topk_topp`: a full-vocabulary
temperature divide, a full softmax (three more passes plus a 993 KB allocation), a
`to_vec1`, and a `select_nth_unstable_by` over 248320 indices with an indirect
comparator. Micro-benched in isolation, the exp pass alone is 0.347 ms and the
`select_nth` 0.270 ms — together they are the tail.

**Change.** sampler.rs no longer uses `LogitsProcessor`; it owns its RNG and its
filtering. Two things moved. The full-vocabulary softmax now runs on whatever device
holds the logits — one Metal kernel instead of 248320 CPU `expf` calls — and the draw
reads back probabilities instead of logits, so the number of bus crossings is
unchanged at one. The candidate set comes from a streaming top-k (one comparison per
entry against the running k-th best, insertion on the order of `k·ln(n/k)` times)
instead of `select_nth` over an index vector. After: 0.406 ms/token, CPU work 0.206 ms.
0.41 ms/token back, ~2.4% of the budget, ~1.5 tok/s at the 59 tok/s baseline.

**The op-order finding.** candle's `TopKThenTopP` is temperature → full softmax → top-k
→ top-p, and its top-p cut is measured against FULL-vocabulary probabilities: it
compares the running cumulative mass of the top-k survivors to `top_p` without
renormalizing them first, and skips the cut entirely when the top-k mass is already at
or below `top_p`. llama.cpp and HF transformers both do the opposite — truncate to k,
renormalize over the survivors, then cut. The two disagree whenever the top-k set does
not hold nearly all the mass. This rewrite deliberately preserves candle's order, so no
distribution changed; it is why the fast path still needs a full-vocabulary softmax
rather than a 20-wide one. The divergence from llama.cpp is a ledger item, not a fix
made in passing.

**What did change: seeded token streams.** candle's candidate list came out of
`select_nth_unstable_by` in unspecified order; this one is sorted by descending
probability. A weighted draw maps its single uniform through the cumulative weights, so
the same uniform now lands on a different token. The distribution is identical, the
seeded sequence is not. Nothing depended on it — the parity gate is greedy end to end,
and the argmax path is untouched (still a CPU first-maximal scan). Pinned by two new
tests: the candidate set is compared id-for-id and weight-for-weight against a literal
transcription of candle's filtering, and 40000 draws are compared against the real
`LogitsProcessor` as a live oracle.

**Verdict.** The bench measures Qwen now, which was the precondition for any of this.
The remaining 0.41 ms is 0.20 ms of readback (mostly command-buffer sync, not copy) and
0.11 ms of streaming top-k; taking the selection to the GPU so only k values cross the
bus is the next lever, and a ledger item.

**Review follow-up, 2026-07-29.** Two reviewers went over the replacement. Nothing they
found moves the distribution on a well-formed row; three of the four are about rows that
are not well-formed, and the fourth is a claim the tests were making too broadly.

- *The padded tail was drawable.* The sampler read the whole logit width, but the output
  layer is wider than the tokenizer (248320 rows against 248070 encodable ids) and the
  rows between them decode to nothing. A padded row winning a slot puts a textless id
  into the stream. `Sampler::new` now takes the encodable bound — `tok.vocab_size()` at
  the one production construction site, never a literal — validates it against the width
  of every row it is handed, and narrows the row to it before anything else runs. The
  narrow is a view, so this is cheaper than what it replaced rather than an extra pass,
  and it keeps the padding out of the softmax denominator as well as out of the
  selection, which is what leaves the two softmax backends (device on the fast path, CPU
  on the controlled one) looking at identical values.
- *NaN was being skipped.* NaN loses every ordered comparison, so the argmax scan walked
  past a corrupt row and returned the best of the survivors — a silent answer where
  `LogitsProcessor` had at least pinned index 0. Every path now errors on a NaN, greedy
  included, which is the one the parity gate runs. `-inf` is not corruption (it is how
  the controls exclude an id) and stays skippable. The streaming top-k catches its NaNs
  inside the branch it already takes for a genuine improvement, so the hot loop still
  costs one comparison per entry.
- *The tie-break contract was overstated.* `top_k_desc`'s strict `>` against the floor
  gives lowest-id-wins at an exact top-k boundary tie; candle's `select_nth_unstable_by`
  leaves the same case unspecified. The determinism is the better contract — it is what
  makes the candidate set a function of the probabilities rather than of the traversal —
  so it is now claimed as such: distribution equivalence for untied inputs, deterministic
  low-id selection at exact boundary ties, and a test pinning the tie behavior itself
  without reference to what candle does with it.
- *Equivalence was mostly tested against a transcription.* The broad oracle is a
  hand-copy of candle's filtering, so a shared misreading would pass; only one narrow
  case ran against the real `LogitsProcessor`. That comparison is now a matrix — widths
  64 / 2048 / the checkpoint's own 248320, k of 1 / 20 / 64, top_p of 0.5 / 0.95 / 1.0,
  over flat, peaked and exactly-tied rows, 81 cases, ~4 s in release. Untied rows assert
  containment (candle must never draw an id this sampler excluded) plus matching
  frequencies; tied rows assert only eligibility, since candle's answer there is
  unspecified. Mutation-checked: shifting the top-p threshold by 30% and flipping the
  tie-break direction each fail it.

Five new lib tests, 666 passing. The one behavior change a user could observe is the
padded tail no longer being drawable, which was never a legitimate outcome.

## 2026-07-28 (later still) — Second-model review: one finding, and it was our own docstring overclaiming

**Context.** The external-model reviewer (GPT-5.6 via codex, ~35 min, xhigh) ran over
the full retarget alongside the two Claude reviewers. It cleared every model-math trap
independently (interleaved gate split, partial rope, rollback indexing, router order,
chunk carry) and missed the tool-call parser bug the integration reviewer caught — the
two reviewer families found disjoint issues, which is the argument for running both.

**Change.** Its one finding (MEDIUM): `kv_rollback`'s docstring promised caches
"byte-identical to ones that only ever advanced over the committed tokens" — but the
q8/f16 dual-storage split makes cache-mutating projections partition-dependent
(a verify batch of 9+ tokens runs the f16 plane where one-token decode runs q8 GEMV),
so the counterfactual-identity half of that promise is false in exactly the
speculative case it describes; the parity gate cannot see it because it runs one fixed
partitioning. Resolution: the mechanism is kept (the bytes-halving is why dual storage
exists), the CONTRACT was corrected — restore is a bit-exact replay of recorded bytes,
cross-partition agreement is numeric and parity-gated, `XWEN_ATTN_DEQUANT` pins one
representation when partition-independence matters. decisions.md records it under
Kernel policy; drift magnitude at the 8↔9 boundary is a ledger item, unmeasured.

**Verdict.** No code behavior changed; a claimed identity was demoted to the observed
one, per this project's own first rule. Review pass complete: three reviewers, one
critical fix (tool-call parser), one contract correction, model math clean.

## 2026-07-28 (late night) — Serve integration fixes: the tool-call parser was reading `:` and `;` as span markers

**Context.** An adversarial review of the inherited serve/ tree against the Qwen
retarget turned up four integration defects. One is severe and had no chance of being
caught by the suite that covered it.

**The headline: `serve/engine.rs` opened tool-call spans on ordinary punctuation.** The
span parser carried laguna's token ids as literals, `TOOL_CALL_OPEN: u32 = 25` and
`TOOL_CALL_CLOSE: u32 = 26`. In Qwen's vocabulary 25 is `:` and 26 is `;`; the real
`<tool_call>` pair is 248058/248059, and `tokenizer.rs` has held those constants since
the fork. So for any request carrying tools, every colon the model wrote in prose opened
a span and every semicolon closed one. What followed a colon stopped being answer text
and started being parsed as a call — delivered as a fabricated tool call if it happened
to parse, silently discarded by the heal path if it did not, which is to say the reply
truncated at the first colon. Genuine `<tool_call>` tokens meanwhile fell through to the
`_` arm and reached the client as literal text. The interior grammar was laguna's too:
`<arg_key>`/`<arg_value>`, strings that are not in Qwen's vocabulary and that chat.rs
has never emitted — so even a correctly-framed span parsed to nothing.

**Why the suite missed it.** There were seventeen tests over this parser and they all
passed. Every one of them scripted the token stream by hand as `(TOOL_CALL_OPEN,
"<tool_call>")` — the constant paired with the text a correct constant would decode to.
That pairing is the bug, asserted as a fixture. The tests agreed with the parser about
what id 25 meant, and neither had ever been asked what the tokenizer thought. A test
that builds its input from the same wrong constant as the code cannot fail when the
constant is wrong; it can only fail when the code stops being self-consistent. The fix
is not more assertions, it is a different input source, so the tests now drive the
emitter over ids produced by the real embedded tokenizer — round-tripping conversations
that `chat.rs` rendered, and one hostile case that feeds prose full of `:`, `;` and
`<function=` text and asserts zero calls with byte-identical output. Under the old
constants that case produces a tool call named `name`.

**Change.** The ids come from `LagunaTokenizer` now; the interior parser reads what
chat.rs writes (`<function=NAME>`, `<parameter=KEY>\nVALUE\n</parameter>`,
`</function>`, one function per span, framing newlines stripped from values). Two
behaviors changed on purpose rather than by translation. `</tool_call>` is structural
wherever it lands, mid-value included — the template writes a literal one inside an
argument as content so it never encodes to the added token, and treating the token as
content is what let a malformed value swallow the rest of a reply. And a span that never
names a callable tool now degrades: raw text, markers included, to the client as answer
text with a logged warning (`ServeLog::ToolSpanDegraded`, counted separately from
`healed` in the per-request report). Never discard, never fabricate.

**Three smaller items.** Drafting was on by default, inherited from laguna, which made
every zero-flag `xwen generate` and `xwen serve` abort at startup — `xwen serve` before
the listener bound. The error turns out to be `missing GGUF key dflash.decoder_arch`
rather than the `decoder_arch == "laguna"` mismatch the review predicted: the shipped
sidecars have no such key, so the failure precedes the check adaptation was expected to
repoint, with `enc.aux_norm` and `blk.N.attn_gate` absent behind it. Default flipped to
off; asking for a drafter still fails loudly. `serve/config.rs` was sizing caches with
laguna's geometry — 48 KiB/token from 12 full layers × 8 KV heads × 128 head_dim, and a
72 MiB snapshot described as copies of 36 SWA rings, in a model with no SWA layer at
all. Real figures are 20 KiB/token (35B-A3B) and 64 KiB (27B), with snapshots of
DeltaNet recurrent state at a fixed 62.8/149.6 MiB; they are derived on `hub::Model` now
rather than carried as constants. And `scan_banned` was protecting the compile-time EOG
set while the decode loop stopped on the GGUF-derived one — harmless today, since the
two match on both checkpoints, but only by coincidence.

**Verdict.** 645 lib tests pass, up from 642. Two pre-existing failures remain, both
`dflash::tests::real_file_*`, both the unadapted-drafter story above; they cannot pass
until TODO.md P9 lands and were failing before this arc. The tool-call parser is the
first inherited subsystem found to be not merely unadapted but actively wrong on Qwen
input, and the way it survived — tests built from the same constant as the code — is
worth carrying into the review of every other inherited dialect layer.

## 2026-07-28 (late night) — Parity harness live (P7): both checkpoints match upstream llama.cpp, floors an order of magnitude tighter than laguna's

**Context.** Everything before this rested on cited source lines, hand-computed unit
tests, and one greedy eyeball. The engine had never been compared against another
implementation on the same weights, and the 27B had never been run at all. P7 was the
item standing between "it produces fluent text" and "it computes the right thing".

**Change.** Upstream `ggml-org/llama.cpp` shallow-cloned into `reference/llama.cpp`
and PINNED at `e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0` (recorded in docs/parity.md).
This session materialized it as a plain clone and gitignored the path, reasoning that
a submodule puts the reference tree in the index and the index is the human's review
surface; a `.gitmodules` entry declaring the same path as a submodule was staged
concurrently from elsewhere, and the owner settled it that night in favour of the
submodule — the gitlink makes the oracle sha reviewable in the diff, so moving the pin
becomes a staged change someone approves, which is the property the pin exists to
have. `scripts/build-llamacpp.sh` retargeted at the path.
`tests/fixtures/parity-prompts.json` regenerated with Qwen ids from the oracle's own
`llama-tokenize --no-bos`; the SWA-specific `long-swa` fixture replaced by
`long-mixed`, 612 tokens of prose that stresses the DeltaNet recurrence instead of a
sliding window that does not exist here. `hf.ts` repointed at the two ggml-org repos
with a size selector; `parity-gate.ts` gained `--model-size 27b|35b` and namespaces
every artifact by checkpoint basename (two official files, two architectures, two
sets of floors — they must never share a parity dir or a frozen ppl fixture).
`ref-dump.sh` retargeted. `scripts/parity.ts` learned the tap-name mapping
(`refTapNames`) rather than renaming engine taps. Laguna's `reference-ppl.json`
deleted; the committed-fixture test now validates every per-checkpoint fixture.

**Two latent parser bugs in the inherited `parity.ts`, both silent corrupters.**
First, the node-header regex captured names as `(\S+)`, so headers with spaces
(`cache_r_l0 (reshaped)`, `(view)`) were skipped and their value rows were attributed
to the previous node — which kept that node's `sum` but replaced its sampled row with
an unrelated tensor's. Symptom: `attn_norm-0` reporting `rowRelL2 = 2.29e+6` while its
values were in fact digit-identical to the oracle's. Second, `FLOAT_RE.test(line)` on a
shared `/g` regex advanced `lastIndex`, dropping every other value row. Neither would
ever produce a false PASS, but both produced convincing false divergences — the first
one cost a real detour before it was pinned down.

**Result.** Both checkpoints agree with upstream on identical GGUF weights. The final
four-tier run with the calibrated constants in place is `ALL PASS (6 graded)` on each —
42 s warm for the 35B, 2.0 min for the 27B, since the Reference dumps are cached and
only the candidates regenerate.

Track A (first-divergence bisection, 35B, code-short, 242 taps compared): no cliff.
The sampled-row rel-L2 profile is smooth and *flattens* rather than compounding —
`l_out` runs 1.8e-3 at layer 0, 1.2e-3 at 7, 2.4e-2 at 23, 1.4e-2 at 39 — and the
final-logits sampled cosine is 0.999995. Individual `sumRelErr` spikes up to 1.9e-1
are near-cancelling residual sums, not divergences; their own `rowRelL2` stays in the
neighbourhood's band.

Track B, `bun scripts/parity-gate.ts`, 35B-A3B Q4_K_M:

| tier | fixture | result |
|---|---|---|
| strict | code-short | PASS, cosine 0.999999861, top-1 = ref, top5 5/5 |
| mm | code-short | PASS, cosine 0.999539782, top-1 = ref, top5 5/5 |
| decode | code-short | PASS, 63/64 agree, 1 excused near-tie (0.0040 logit) |
| decode | text-mixed | PASS, 62/64 agree, 2 excused (0.5567, 0.2606) |
| decode | long-mixed | PASS, **64/64 agree**, 0 excused |
| ppl | — | PASS, Δmean_nll 0.000511 (fused 1.694170 vs reference 1.693659 over 4218 tokens) |

**The 27B's first forward was correct.** No bisection was needed: it loaded (18.2 GB
resident), prefilled 58 tokens in 9.6 s cold, and produced a top-5 that tracks the
35B's on the same prompt. Its parity numbers are *better* than the 35B's across the
board — strict bitwise 1.000000000, mm ≥ 0.999993294, decode **64/64 with zero excused
near-ties on all three fixtures**, ppl Δ 0.000221 (fused 1.748093 vs reference
1.747872) — with the caveat that on a dense model the
strict tier is near-vacuous, since `--moe-impl reference` and `fused` run the same
`DenseMlp` and the strict env pins everything else classic on both sides. The 27B's
real signal is the mm/decode tiers, which exercise the f16 attention path and the
fused glue.

**Floors, calibrated across both checkpoints and all three fixtures** (the constants
are global, so they are set under the WORST observed value): `COS_MIN_STRICT = 0.9998`
(worst achieved 0.999894, 35B long-mixed) and `COS_MIN_MM = 0.999` (worst achieved
0.999540, 35B code-short — ~1.7x the observed prompt-to-prompt spread). Both are an
order of magnitude tighter than laguna's 0.9955 / 0.985: these kernels track the
oracle much more closely on this architecture, largely because the Qwen Q4_K_M mix
keeps attention, ssm and shared-expert weights at q8_0.

**Verdict.** The engine is parity-validated. This is the checkpoint that makes every
later kernel change checkable instead of hopeful, and it is the gate P8's DeltaNet
kernels will be graded against. Four things are explicitly NOT proven and are in the
ledger: Track A cannot localize inside a layer (the tap set is still laguna's six —
no DeltaNet core, router logits, or shared-expert gate taps, which would need plumbing
in the model-math files); `provenance.flash` says `"fused"` while `flash.metal` is
compiled at head dim 128 and cannot serve Qwen's 256, so prefill is really candle sdpa
with a materialized mask; the dense strict tier as noted above; and the `_Q8` widened
bands were inherited, not recalibrated — measured worst per-step l2 deviation is
1.0211 against a 1.5 band (far too loose), while the near-tie window genuinely needed
its widened 1.0 (text-mixed step 15 excused at 0.5567).

## 2026-07-28 (night) — First real-weights decode: correct output, clean stop, 59 tok/s on the 35B-A3B

**Context.** Everything to this point rested on cited source lines and hand-computed
tests; nothing had touched real weights. The 20.4 GB ggml-org Q4_K_M finished
downloading and the release binary was fresh.

**Change.** None — this is the first execution of the P2-P6 stack:
`xwen generate -p "What is 2+2? Answer with just the number." --temp 0 --no-draft`.

**Result.** Correct end to end on the first attempt: a coherent five-step thinking
block, `</think>`, the answer `4`, and a natural stop on `<|im_end|>` (143 tokens
emitted with 369 of budget left — the stop list works). Load 2.8-3.0s via the mmap
path, 19.2 GB resident as predicted (weights 19.0 + KV 0.2 + recurrent state 0.1 at
max_ctx 8192). Warm numbers, power mode unmeasured: prefill 23 tokens at 167.6 tok/s,
decode 58.6-59.0 tok/s. The 8.76s cold-run prefill was Metal pipeline compilation, not
the DeltaNet scan — the "will look like a hang" warning in the P2-P4 entry overstated
short-prompt cost; long-prompt prefill through the per-token scan remains unmeasured
and P8 still owns it. Coherence at temp 0 is strong indirect evidence for the whole
trap set (tiled V-heads, norm baking, gate split, router order): any of those wrong
produces fluent garbage, not correct arithmetic under a formatting constraint.

**Verdict.** The engine runs Qwen 3.6 correctly by the greedy-eyeball half of the
fallback gate. Not yet parity-validated — P7 is unchanged as the next honest
checkpoint. Decode at ~59 tok/s pre-kernel-work is 2.7x laguna's decode on 2.7x
fewer active params, i.e. exactly the bandwidth story, and the DeltaNet reference's
~240 extra dispatches per token are not yet visibly the bottleneck at this length.

## 2026-07-28 (evening) — ChatML port, tokenizer swap, and a latent constrain-width bug (P5+P6): suite fully green, 642/0

**Context.** The fork still rendered Laguna's angle-bracket template over Qwen's vocab,
tokenizer.rs carried Laguna ids, and two tests failed on vocab assumptions. hub/CLI
still resolved poolside checkpoints.

**Change.** chat.rs rewritten against the vendored chat_template.jinja (ChatML, tools
preamble, XML-ish call format with string-args-raw, collapsed tool-result turns, the
last_query_index thinking-retention rule, open-`<think>` generation seeding), keeping
laguna's content/structure separation; typed `ChatError` mirroring the template's raise
cases plus one deliberate divergence (tool-result-first refused — the template emits a
malformed boundary there). tokenizer.rs repointed at Qwen ids and made the single owner
of every token id, exposing both vocab sizes (248070 encodable / 248320 logit width).
hub.rs got a per-model checkpoint table (`--model-size 27b|35b`, 35b default,
API-verified filenames); sampling defaults aligned to 1.0/20/0.95. The constrain trie
is now sized to the logit width — which surfaced and fixed a latent bug where every
constrained serve request would have died on a 248,096-bit mask against 248,320 logits.
Control-token safety under grammars turned out to rest on toktrie's angle-bracket
heuristic rather than the special flag; documented and pinned by a full-range sweep
test. Design calls recorded in decisions.md "Tokenization, chat, tool calls".

**Result.** 662 lib tests: 642 pass, 0 fail, 20 ignored (perf benches). The five
template vectors are byte-exact; a differential harness ran 20,000 fuzzed conversations
plus exhaustive role-shape and Unicode-whitespace sweeps against the reference jinja
with zero divergences. Adversarial review by a second model family found two real bugs
pre-merge (silent second-system drop; trim whitespace set), both fixed.

**Verdict.** The prompt/token surface is done and independently verified. What remains
before trusting the whole stack is a forward pass against real weights — the 35B
Q4_K_M was downloading as this entry was written.

## 2026-07-28 — Model core retargeted to Qwen 3.6 (P2-P4): config, loader, DeltaNet reference, attention and MoE

**Context.** The mechanical fork built green but still computed Laguna: 48 uniform
attention layers, a sliding-window ring every fourth layer, a softplus per-head output
gate, and a sigmoid/bias/scale MoE router. P2-P4 replaced that core wholesale with the
Qwen 3.6 hybrid, on the critical path for everything downstream. Two research agents ran
alongside the implementation — one extracting llama.cpp master's `qwen35.cpp` /
`qwen35moe.cpp` / `delta-net-base.cpp` graphs, one range-parsing the shipped ggml-org
GGUF headers — so every load-bearing form was written first from the CLAUDE.md cheat
sheet and then confirmed or corrected against a cited source line.

**Change.** `LagunaConfig`/`LagunaModel` → `XwenConfig`/`XwenModel`. config.rs parses
both archs, rejects anything else, and derives per-layer `LayerKind::{Full, Linear}`
from the `full_attention_interval` key rather than a hardcoded 4. New `linear_attn.rs`
holds the gated-DeltaNet layer as a frozen oracle in the `ReferenceExperts` sense:
composed candle ops, recurrent form only, fp32 state, one sequential scan step per
token. attention.rs gained the double-width `attn_q` with its per-head interleaved gate
(strided split, not a halving of the row), QK-RMSNorm over 256 dims, partial NEoX rope
at n_rot 64 / theta 1e7, sdpa scale 1/√256, and an ELEMENTWISE `sigmoid(gate)` — 4096
independent values per token, not one scalar per head. moe.rs swapped the router for
softmax-over-all-256 → top-8 → renormalize with the f16 floor, no selection bias and no
weight scale, and gave the shared expert its scalar sigmoid gate. kv_cache.rs grew a
third `LayerCache::Linear` variant carrying conv window and delta state in f32 through
checkpoint, rollback, snapshot and the on-disk framing; the SWA ring machinery is left
in place but nothing on the model path constructs it. model.rs dispatches per layer and
loads the pre-MLP norm from `post_attention_norm`. The mmap/no-copy loader, ExpertStack,
QLinear and the dual-storage attention planes were not touched — only the name table.

**Result.** `cargo build` and `cargo test --no-run` green. 659 lib tests: 637 pass, 20
ignored, 2 fail — `generate::ban_scan_catches_every_em_dash_token` and
`constrain::walks_a_valid_document_and_completes`, both in tokenizer/constrain code this
arc never touched, and the second is the already-logged `<think>`-is-not-special trap.
23 tests are new, including the three the recurrence needed: the delta rule walked
against a scalar f64 reimplementation of the update equations at head dim 2 over three
tokens, the gated-norm ordering pinned with a non-uniform weight and opposite-sign gate
factors, and conv-state continuity checked by feeding seven tokens one at a time versus
one batch. Six corrections came out of the ground-truth pass: no fused `ssm_ba` (two
separate tensors), `ssm_conv1d.weight` is 2-D not 3-D, `full_attention_interval` exists
and should be read, `value_length` exists and should be asserted square, neither
`eot_token_id` nor `eom_token_id` exists so the second stop id is a named constant, and
the L2 norm is ggml's clamp form rather than HF's rsqrt form. The tiled-versus-
interleaved k-head broadcast — the one assumption that could have silently corrupted
every DeltaNet layer — came back tiled, as the cheat sheet said, traced to
`ggml_compute_forward_repeat_f32`'s destination index.

One finding is a real regression rather than a correction: **the vendored flash kernel
is unreachable on this architecture.** `flash.metal` is compiled at `BD == 128` and
Qwen 3.6 is head dim 256, so prefill falls back to candle's sdpa with a materialized
`[1, n_head, seq, k_seq]` f16 mask — precisely the allocation laguna's flash path was
written to avoid. Correct but slower, and now a ledger item alongside the deleted
attention benches and the DeltaNet rollback trail's ~1 GB verify-walk cost.

**Verdict.** The engine computes Qwen 3.6. Nothing has been run against real weights
yet, and it should not be mistaken for validated: the fallback gate (reference unit
tests plus a greedy eyeball) has only had its first half satisfied, and prefill through
a per-token sequential scan will make that first real load slow enough to be
mistaken for a hang. The next honest checkpoint is P7's parity harness; until it lands,
every claim here rests on cited source lines and hand-computed tests, not on a number
measured against llama.cpp.

## 2026-07-28 — Fork bootstrap: laguna mapped, Qwen 3.6 architecture pinned down, mechanical fork started

**Context.** xwen is a manual fork of ../laguna (crate `maxuna`, ~72k lines: candle+Metal
GGUF inference engine for poolside Laguna S 2.1) retargeted at Qwen3.6-27B and
Qwen3.6-35B-A3B. Bootstrap ran as five parallel agent workstreams: laguna codebase map,
Qwen 3.6 architecture research, GGUF header survey (range-request parsing of ggml-org
files, no downloads), llama.cpp reference extraction, and the cp-based mechanical fork.

**Findings that set the design.** Qwen 3.6 is not Qwen3 — it is the Qwen3-Next-derived
hybrid: 3 gated-DeltaNet linear-attention layers per full-attention layer (full at
indices 3,7,11,…), sigmoid-gated attention output fused into a double-width q_proj
(per-head interleaved [q_h, gate_h]), QK-RMSNorm over head_dim 256, partial RoPE (n_rot
64 of 256, theta 1e7; IMROPE in llama.cpp, but provably identical to NEoX rope for
text-only), MoE with 256 experts / top-8 / softmax-then-renorm plus a sigmoid-gated
shared expert (35B), MTP head shipped as sidecar GGUFs. candle has zero support for any
of this — the DeltaNet recurrence is new code. Full config/tensor tables live in
CLAUDE.md's cheat sheet; conversion traps (norm +1 baking, tiled V-heads, ssm_a =
-exp(A_log), no ffn_norm, no ssm_in) are recorded in decisions.md "Weights and loading".

**The dflash reversal.** The plan dropped dflash.rs as Laguna-specific (a diffusion
drafter bound to a poolside checkpoint). The GGUF survey then found ggml-org publishes
DFlash sidecar drafters for both Qwen 3.6 models under the same `dflash` architecture —
the subsystem is portable after all. The removal was cancelled mid-flight; the fork
keeps dflash.rs and all drafter wiring, with Qwen adaptation tracked as its own TODO
item. Lesson, same one laguna's decisions.md preaches: check the artifact before
deleting the code that consumes it.

**Fork state.** Mechanical fork (cp-based copy, maxuna→xwen rename, MAXUNA_*→XWEN_* env
prefix, Qwen tokenizer/template vendored into reference/) running as this entry is
written; build gate is `cargo build` on the unmodified-logic tree. Docs
(this file, decisions.md, parity.md, CLAUDE.md, TODO.md) written fresh, mirroring
laguna's documentation system.

**Verdict.** Research phase complete with high confidence on every load-bearing fact
(all numbers read from shipped GGUF headers, llama.cpp master source, and live HF
repos, not from memory or blogs). Implementation fan-out next: config/loader, DeltaNet
reference, attention/MoE adaptation, chat.rs, parity harness — in that order.
