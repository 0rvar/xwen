# Refuted perf directions — do not reopen without new evidence

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic; moved
here verbatim on 2026-09-06. Dated paragraphs, newest additions appended within their topic.


**Refuted: the f32-tile mm_id family (`_t_hp`) as a way to delete the rescale chain
on the 35B.** The rescale chain exists only because the default f16 tensor tiles
stage the activation as half; the `_t_hp` family stages f32 and needs none of it.
But at the 2048 chunk the `XWEN_MM_ID_TENSOR_HP=1` arm reads 2286-2314 prefill
tok/s against the default's 3081-3090 in the same sweep — the f32 tiles cost ~25% of
the gemm where the whole rescale chain costs a few percent of the stage (and the L2
fold has since collapsed it to one dispatch). Decode unchanged (108.4-108.7).
Measured 2026-08-30, log.md "FFN glue".

Laguna's refuted list (death-by-dispatch, encoder takeover, all-f16 activation chains,
mixed-operand matmul2d as a tensor speedup, sub-32-seq mm_id) transfers as *prior
evidence, not law* — same machine, same candle, same kernel geometry, different model.
Anything re-opened here needs a measurement, and its entry moves into this section with
the number that killed or revived it (2026-07-28).

**The dual-weight expert gather — fusing gate, up and their SwiGLU into one dispatch —
is REFUTED on this device.** `kernel_mul_mv_id_q4_K_f32_dual` (mv.metal) does exactly
that, and is bitwise against the three dispatches it replaces: it calls the same
`mul_mv_q4_K_sums` body once per projection, factored out of the verbatim ggml
`mul_mv_q4_K_impl`, so neither accumulation order moves, and the epilogue is
`silu_mul.metal`'s expression with its roundings pinned. It still measured slower —
99.5 tok/s decode against 102.8 for the split chain, 35B-A3B, interleaved, median of 5,
five reps apart with no overlap between the arms.

The mechanism is worth recording because it generalizes. It is NOT register pressure:
the single and dual pipelines both report `max_total_threads_per_threadgroup` of 1024.
It is the grid. Two independent `mul_mv_id` dispatches expose twice the threadgroups,
and the routed-expert gather is bandwidth-bound, so that parallelism is what keeps
memory requests outstanding. The dual kernel keeps ONE dispatch's grid while walking both
weight planes serially per thread, halving the memory-level parallelism to save two
launches. Dispatch count is the right thing to attack in the MoE glue — 24 to 14 bought
11% — but it stops being the right thing to attack the moment the dispatches being
merged were already saturating bandwidth in parallel.

Kept behind an opt-in `XWEN_MOE_DUAL=1` rather than deleted: it is a measured, gated,
bitwise-proven artifact, and the conclusion is device-specific. Re-price it before
reusing it, do not assume it (2026-07-29).

**The DeltaNet scan decomposition: llama.cpp's Metal shape is REFUTED here, and the
scan was never the 27B prefill gap in the first place.** Two claims, measured together,
because the second is what makes the first not worth revisiting.

llama.cpp's `kernel_gated_delta_net_impl` (ggml-metal.metal) gives ONE SIMDGROUP each
state value-column end to end: both key-dim contractions become `simd_sum`, the timestep
loop holds no barrier at all, and the grid is `(S_v/nsg) × H × n_seqs` — 1536
threadgroups at the 27B geometry against our 192, eight times the parallelism. It is
transplanted here as `kernel_delta_scan_v2` plus the hoisted `kernel_delta_l2norm` the
shape requires, adapted to our transposed state layout and our end-of-scan `1/sqrt(128)`.
It is SLOWER, at every geometry and every length. Isolation timing of one layer's whole
recurrence (`delta_scan_timing`, plateau ms, both arms inclusive of their q/k norm):

| geometry | seq 880 | seq 4096 |
|---|---|---|
| 27B (16 K / 48 V), shipped | 1.97 | 8.56 |
| 27B, llama.cpp shape | 2.73 | 14.81 |
| 35B-A3B (16 / 32), shipped | 1.57 | 6.31 |
| 35B-A3B, llama.cpp shape | 1.88 | 8.93 |

The mechanism is the same one the dual-weight gather taught, running the other way.
Parallelism was not the binding constraint; q/k READ AMPLIFICATION is. Our kernel stages
one normalized q and k vector into threadgroup memory per threadgroup per timestep, so
that traffic scales with the 192 threadgroups. Giving every simdgroup its own column
also gives every simdgroup its own copy of those reads — 6144 of them per timestep at
the 27B geometry, 32x the L2 traffic — and at head dim 128 each lane owns only 4 state
entries, so two `simd_sum` reductions sit on top of 8 useful FMAs per lane per timestep.
The barriers and the redundant in-register norm that the shape removes are cheaper than
that. Hoisting the norm out on its own is a loss too, for a related reason: the separate
dispatch reads and rewrites the whole q|k plane (0.52 ms at seq 880, 1.80 ms at 4096,
27B) to save a reduction that was already amortized over a threadgroup.

The second claim is the one that closes the direction. **The scan is 3% of 27B prefill,
so no decomposition of it can be the 1.8-2.1x llama.cpp prefill gap.** 48 DeltaNet
layers × 1.97 ms is 95 ms of a 2.96 s prefill at 880 tokens, and × 8.56 ms is 411 ms of
a 14.2 s prefill at 3851. Deleting the scan outright would move 27B prefill from ~297 to
~307 tok/s against llama.cpp's 486. The end-to-end A/B corroborates the isolation
numbers to within noise — the llama.cpp shape costs 48 × 6.25 ms ≈ 300 ms at 4096, which
is 1.9% of the wall time, and the measured arms differ by 1.7%. On the 35B the scan is a
larger slice (30 layers, 11-13% of prefill) but that model is already near llama.cpp
parity. Whatever the 27B loses, it loses in the dense projections, not here (2026-07-29).

`kernel_delta_scan_v2` and `kernel_delta_l2norm` are kept behind `XWEN_DELTA_SCAN_V2=1`
rather than deleted, on the `XWEN_MOE_DUAL` precedent and for a sharper reason: the
kernel they mirror is sitting in `reference/llama.cpp` inviting the same proposal, so the
refutation is worth more as a runnable arm than as a paragraph. `delta_scan_timing`
(src/ops/delta.rs, `#[ignore]`d) is the instrument — re-run it before believing any
future claim about the scan's cost, including this one.

**A decode-specialized scan kernel is a WASH, and the "60 GB/s scan" that motivated it
was a profiler artifact.** `kernel_delta_scan_decode` (delta.metal) is the seq == 1
shape the general kernel's loop body cannot be: no timestep loop, the state read and
written as float4 rather than as 32 scalars at a 512-byte stride, the row-slice folds
done with a `simd_shuffle_xor` butterfly inside a simdgroup, and the q/k clamp-norm
computed once per threadgroup instead of once per timestep. It is correct — graded at
1e-5 against both the general kernel and the reference over consecutive state-carrying
steps at both geometries — and it buys nothing.

| arm (Flash-Next, unprofiled, interleaved, `--no-draft`) | round 1 | round 2 |
|---|---|---|
| general kernel (shipped) | 44.7 tok/s | 44.8 |
| decode kernel (`XWEN_DELTA_DECODE_KERNEL=1`) | 44.6 | 44.7 |

The 35B-A3B is the same story (105.5 against 105.4/104.4; its round-1 general arm read
96.8 cold and is not comparable), and both checkpoints emit byte-identical 64-token
greedy text under either kernel.

The premise is what actually moved. `XWEN_GDN_PROFILE` reported the scan at 3.79-7.19 ms
per token and ~30 GB/s of its declared byte floor, which reads like a kernel running at a
tenth of what `out_proj` achieves in the same block. It is not: every step in that line
is bracketed by a device sync, so a step that dispatches one kernel pays a full GPU round
trip the real forward pipelines away, and the module's own dispatch-floor correction does
not recover it. Priced the way CLAUDE.md's benching rules require — batched dispatches per
sync, a whole token's worth of layers per iteration, states as cold as the real ones —
the scan costs **1.35-1.43 ms/token at 36 layers of the 48-V-head geometry (160-170
GB/s), against 0.98-1.02 ms for a candle affine moving exactly the same state bytes with
no arithmetic at all** (`delta_scan_decode_timing`, src/ops/delta.rs, which carries that
floor arm as its third arm; medians of two runs on a shared machine, the second the
quieter). The decode kernel reads 1.27-1.41 — 0-5% better in isolation, which is the
0.3% of a token that the end-to-end arms could not see. So the whole prize between "as
shipped" and "free" is ~0.4 ms of a ~22 ms token, and the state traffic itself — 3.1 MB read plus
3.1 MB written per layer per token — is most of what remains. Differencing the two
geometries puts the scan's marginal rate at 525-564 GB/s, which is `out_proj`'s own
number: the kernel is bandwidth-bound, and it was already bandwidth-bound before this
kernel existed.

Kept OPT-IN behind `XWEN_DELTA_DECODE_KERNEL=1` rather than shipped as the seq == 1
default, on the `XWEN_DELTA_SCAN_V2` / `XWEN_MOE_DUAL` precedent: a second bounded
kernel on the decode path is a permanent parity surface, and a wash does not pay for one.
What it does pay for is the arm — the bench that priced the scan honestly needs something
to price it against, and the next person told the decode step is starving for bandwidth
should be able to run the fix rather than rebuild it. `delta_scan_decode_timing` calls
both kernels directly and needs no switch (2026-08-30).

**"`attn_qkv` runs at a third of its siblings' rate" is REFUTED — at DRAM it is the
FASTEST of the three GDN projections, and the profiler's ordering of them is inverted.**
`XWEN_GDN_PROFILE`'s decode line reported the `attn_qkv` plane (`[2560 → 10240]`, 1003
MB/token over 36 layers) at 346 GB/s where `attn_gate` (`[2560 → 6144]`) read 523 and
`ssm_out` (`[6144 → 2560]`) read 537 through the same `kernel_mul_mv_q8_0_f32_attn` —
which reads like an output width the grid or the cache falls off at, on the single
largest weight read in the block. `q8_gemv_shape_sweep` (src/ops/q8.rs) sweeps eight
shapes across both axes in three arms: `reuse` (one plane, 32 dispatches per sync —
cache-resident, **never quotable as a kernel rate**), `rotate` (distinct planes covering
~512 MB round-robin, which is the situation a real decode step is in) and `synced` (one
dispatch per flush, the condition the profiler itself measures in).

The rotate arm, medians of three runs of 41 rounds: `attn_gate` **464 GB/s** (36.1 µs),
`ssm_out` **465** (35.9), `attn_qkv` **510** (54.6), and off the production shapes 493 at
8192, 525 at 12288, 505 at `[10240 → 2560]`, 531 at `[6144 → 6144]`. Four K=2560 shapes
fit **t = 8.41 µs + bytes / 604 GB/s** by least squares (R² 0.99996, max residual 0.09
µs); the three off-axis shapes were held out and land within 0.8 µs. There is no cliff at
any width — the rate rises monotonically with bytes moved, because the only thing varying
is how much traffic the fixed 8.41 µs amortizes over, which is exactly why the widest
production plane is the fastest one. `[2560 → 2560]` is the one shape off the line (30.8
µs against 19.9 predicted, 55% over) and is EXCLUDED from the fit: 1280 threadgroups is
too small a grid to fill 40 cores, and it is also the only cell that moved run to run.

**The 346 was not reproduced, and the entry does not claim it was.** The synced arm reads
122 GB/s for `attn_qkv`, because its raw number still carries the ~157 µs sync floor that
`gdn_profile` subtracts; reconstructing the profiler's own arithmetic (solve the floor
from the reported `attn_gate` figure, 160.3 µs, subtract from `attn_qkv`'s 227.8) gives
67.5 µs = 413 GB/s against production's 346. Right mechanism, right direction, right
order of magnitude, wrong number. What is established is that **the kernel cannot produce
a 346/523/537 spread and the measurement condition demonstrably can** — the condition
turns a true 18.5 µs gap between the two shapes into a measured 35.6 µs one, and a
constant floor subtraction from ~200 µs numbers amplifies it further. The synced arm has
its own fit, `t = 157 µs + bytes / 476 GB/s`, so that condition also costs ~21% of the
marginal rate on top of the floor.

The 604 GB/s is a marginal slope differenced between two arms of the same bench, NOT an
appeal to a peak-bandwidth figure — at the time this machine's peak had never been
measured and the repo rule forbade arguing from the nominal one (MEASURED 2026-09-05:
537-565 GB/s streaming read, see "Measurement discipline"; the 604 slope sits 7-12%
above that range). Machine conditions, stated because they
were not ideal: `pgrep` was clean at the START of the session and not re-verified per
run; at least one other agent was on the machine during part of it (the unstable
`[2560 → 2560]` cell is the visible contention); the four cells the conclusion rests on
held to ±0.3 µs across three runs; and power mode was read only afterwards (`powermode
0`), which does not establish what was in force during the runs.

A geometry retune was priced at the same time and there is none: six `(NR0, NSG)`
configurations A/B'd by temporary edits to q8.metal and dispatch.rs (reverted), all six
passing `q8_decode_production_shapes`. The shipped `(2, 4)` wins at all three production
shapes (36.1 / 54.2 / 36.1 µs); `(4, 4)`, `(8, 4)` and `(2, 2)` are clean and worse,
`(4, 2)` and `(4, 8)` are much worse but came out non-monotonic in `n_out` and count as
"clearly not a win" rather than as figures.

Do not re-open a "slow plane" reading off the profile line without an amortized arm; the
sweep is `#[ignore]`d and takes seconds (2026-08-30).

**How to read `XWEN_GDN_PROFILE`: it RANKS steps, it does not PRICE them.** Every step on
the line is bracketed by a device sync, so a step that issues one kernel pays a GPU round
trip the real forward overlaps away. The module's dispatch-floor correction is one global
number, while the inflation is per step and roughly inverse to the step's byte count — so
the correction does not recover it, and the printed shares are shares of an inflated
total (the raw mixer total, 78 ms, is more than three whole unprofiled decode tokens).
Two figures off that line have now been measured properly and both were wrong by 2-3x in
the same direction: the scan (3.79-7.19 ms/token on the line, 1.35-1.43 amortized) and
`attn_qkv` (346 GB/s on the line, 510 at DRAM). Neither inflated figure was reproduced
EXACTLY from outside — the qkv reconstruction lands at 413 rather than 346 — so the
correction to make when reading the line is directional, not a divisor: assume a step is
faster than it says, by more the fewer dispatches and bytes it has, and re-measure rather
than rescale. Use it to decide WHICH step to
investigate — that is what the 27B prefill work used it for and it was right every time.
Then price the step with an amortized bench (batched dispatches per sync, outputs held
alive, per CLAUDE.md's benching rules) or with end-to-end tok/s before believing a cost,
and never quote a decode figure from it as one. The same rule already applies to
`XWEN_STACK_PROFILE`'s decode stages. Fixing the instrument — bracket the whole block
once and attribute by difference, or run every step under `XWEN_GDN_REPS` and say so on
the line — is ledgered, not done (2026-08-30).

**Chunk-boundary device syncs and command-buffer batching granularity are both REFUTED
as levers on the 27B prefill residual.** The residual — +350 to +560 µs/token of
length-dependent prefill cost outside every measured stage — was reproduced in situ at
+410.3 and +437.9 µs/token across two rounds, and per-stage serialization accounts for
only +102.8 of it (mixer_full_attn +53.5, ffn +42.2, residual_ffn +16.8, mask_upload
+7.9, mixer_delta −9.1). So ~335 µs/token exists ONLY when stages pipeline, and the two
obvious pipelining suspects were tested directly and both cleared.

*Cross-chunk accumulation.* `XWEN_CHUNK_SYNC` waits for each 512-token chunk's forward
to complete before enqueueing the next, which prunes candle's buffer pool, clears its
fence map and drops the encoder's barrier history — everything a chunk boundary could
reset. The length delta with it on is +431.1 µs/token against +437.9 without, a
difference of −6.8. Whatever accumulates re-develops inside a single chunk. (The flag
itself costs +9.2 µs/token at 925 and +2.4 at 4k — a per-chunk fixed price, not a
length-dependent one.)

*Command-buffer batching.* `CANDLE_METAL_COMPUTE_PER_BUFFER` swept over 10 / 200 / 1000
against candle's default 50 at 4k: all four means within 0.9%. (Decode-side coda,
2026-08-30: the same knob at 200/1000 LOSES decode monotonically with context — −3.6%
@1937 to −6.8% @7606 at 1000, greedy byte-identical — so 50 is not merely adequate, it
is the right side of the curve; see the technique-survey ledger item.) A 100x range of batching
granularity moves nothing, so the cost is neither submission overhead nor a per-buffer
fixed price.

What is left standing is intra-chunk, and unconfirmed: barrier storms from
buffer-pointer recycling (candle's encoder emits a full `MTLBarrierScope::Buffers`
barrier when a pool-recycled pointer is reused within one encoder session) and
fence-wait pileup (every new encoder waits on every fence in the growing
`prev_ce_outputs` map). Both are consistent with the `XWEN_CHUNK_SYNC` result, and
separating them needs a counter candle does not expose — a patched candle or a Metal
capture. Do not re-propose either refuted lever without an instrument that can see
inside a chunk (2026-08-08).

**Extending the `mul_mv_ext` window past seq 8 is REFUTED — ggml's `ne11_mm_min` 8 is
the right ceiling and is now measured rather than inherited.** The obvious follow-on to
the small-batch win was to raise the ceiling: spans 9-32 are where the verify round's
fixed cost actually lives, and the window's upper edge was ggml's tested envelope rather
than anything anyone here had priced. `XWEN_MV_EXT_MAX_SEQ=32` prices it, and the
multi-row mat-vec loses: spans 16 / 24 / 32 come in **worse than classic by 1.11x /
1.42x / 1.69x**, with span 12 a wash at 0.98x. The degradation is monotonic in span,
which is the signature of the mechanism — a mat-vec reusing one dequantized block across
r1ptg rows stops paying once the token count is large enough for the tiled path's
threadgroup grid to fill, and past that it is strictly the worse decomposition. The
crossover is where ggml put it. Do not re-raise the ceiling without a kernel that
changes this shape (2026-08-08).

**Refuted: tensor cores for the decode gemv.** The cooperative-tensor path has paid
twice on this machine (dense_mm's 2.2-2.7x, the `mm_id` `_t` family) and the obvious
next thought is to point it at decode. Three independent sources say no, and they agree
closely enough that spending a bench on it would be spending it to confirm a
consensus: Apple's own M5 figures put the decode gain at **+19-27% against a +28%
memory-bandwidth gain** — it tracks bandwidth, not arithmetic — and both BaseRT-M5
(arXiv 2607.00501) and MLX keep decode on SIMD kernels while using cooperative tensors
for prefill. The mechanism is the one the `mul_mv_id` dual-weight refutation above
already taught in another form: a batch-1 mat-vec reads a weight byte per FMA, and the
arithmetic units are not what it waits on. Cooperative tensors stay a PREFILL lever
here. Re-open only with a measurement on this machine (2026-08-30, log.md "technique
survey").

**Refuted: reducing the number of graph splits.** llama.cpp PR #27880 measured exactly
this, on qwen4exp, on an M5: splits 4 → 2 moved prefill **665.65 → 665.27** tok/s and
decode **27.99 → 27.29**. Nothing, in both directions of nothing, with decode slightly
the wrong way. Their graph and their measurement, but the same architecture family and
the same GPU generation, and there is no version of this cheap enough to be worth our
own arm (2026-08-30).

**Refuted AS ALREADY PRESENT: "adopt MLX-style concurrent encoding".** The most-cited
Apple-silicon inference technique — one long-lived encoder in concurrent dispatch mode
with dependency-derived barriers instead of an encoder per kernel — is the top item on
every survey of this area, and candle already does all of it. Anchors, pinned rev
21cca0b, candle-metal-kernels: `computeCommandEncoderWithDispatchType(Concurrent)` at
`command_buffer.rs:24`; the dependency-tracked `auto_barrier` with hazard sets spanning
the whole window since the last barrier at `encoder.rs:104-149`; fences plus untracked
buffers across encoder boundaries; commit every `CANDLE_METAL_COMPUTE_PER_BUFFER`
dispatches (default 50 — per DISPATCH, not per op as the doc comment says) at
`commands.rs:18,162`. xwen's 137 dispatch sites bind through
`set_input_buffer`/`set_output_buffer` and participate fully, so there is no xwen-side
adoption left to do; encoders break only at the readbacks (`sampler.rs:257` and the
scoring path), explicit synchronizes, blits, and the ~77-dispatch decode step's two
rollovers. What remains is not the scheme but its granularity — see "Dispatch-floor
levers ranked" under Kernel policy, and do not re-propose the scheme itself
(2026-08-30).
