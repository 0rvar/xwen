# 2026-09-07 — The Z-Image transformer's linears on the Metal-4 tensor gemm: a 1024x1024 step from 5.0-5.25 s to 3.06-3.59 s, image PSNR 29.71 to 47.03 dB

The first performance arc on the image pipeline, and the first one that had to start by
measuring rather than by profiling: `Timings` records one number per step and nothing
inside it, so no lever in the graph was priced before this. The architecture is in
[docs/zimage.md](../zimage.md), the decisions in
[docs/decisions/zimage.md](../decisions/zimage.md), the pipeline arcs in
[zimage-pipeline.md](zimage-pipeline.md), and the current figures in
[docs/perf-state.md](../perf-state.md).

Two commits, e54801f for the gemm arc and ab1cde2 for the per-stage profiler that ranked
what it left. Everything below was measured on this machine on 2026-09-07 with
`pmset -g` reading `lowpowermode 0`; no high-power claim is made, because neither
`lowpowermode` nor `powermode` can confirm it. Builds are dev-tree release builds, not
pinned binaries. The first run of the microbench arm that links the xwen lib was built in
a detached worktree at HEAD under `/tmp/xwen-microbench`, the main tree being mid-edit at
the time; the later quiet-tree re-run was built in the main tree once it compiled again,
which is immaterial because `src/ops` carried none of those edits.

## The measurement

### Baseline, before anything changed

| Figure | 1024x1024 | 512x512 |
| --- | --- | --- |
| transformer step, each of 8 | 5.02-5.25 s | 1.17-1.25 s |
| VAE decode | 4.88 s | 1.13 s |
| total wall, warm | 50.2 s | 14.7 s |

At roughly 62 TFLOP of arithmetic in a 1024x1024 step that is about 11.6-12.4 TFLOPS
end to end.

### candle's gemm at the model's own shapes, and its dtype-blindness

`tests/zimage_microbench.rs` is the ignored bench that priced this
(`cargo test --release --test zimage_microbench -- --ignored --nocapture`, 60 s). Every
figure is per-iteration wall time between two `Device::synchronize` calls after three
warm-up iterations. Shapes are the model's: dim 3840, SwiGLU 10240, 30 heads of 128,
T 4128 for 1024x1024 and 1056 for 512x512.

candle dispatches every `matmul` through the MLX steel GEMM, and it lands at the same
rate whatever dtype it is handed:

| square GEMM, TFLOPS | bf16 | f16 | f32 |
| --- | --- | --- | --- |
| 2048 cubed | 14.88 | 14.92 | 14.75 |
| 4096 cubed | 15.60 | 15.59 | 15.02 |
| 8192 cubed | 15.70 | 15.69 | 14.67 |

A reduced-precision path that gives nothing over f32 is not a reduced-precision path.
On the model's own shapes bf16 reached 15.6, f16 14.8 and f32 12.0, so bf16 was the
fastest of the three but by 30% where a tensor path should be 2x or more.

Two layout findings from the same run, worth knowing only so nobody undoes them.
Pre-transposing the stored `[out, in]` weight into a contiguous `[in, out]` buys nothing
(15.43 against 15.38 at T 4128), so weight layout was never a lever. And
`broadcast_matmul` on a `[1, T, K]` activation is 13-18% slower at T 4128 and 36-38% at
T 1056 than the 2-D `matmul`; the pipeline never paid that, because `candle_nn::Linear`
reshapes a contiguous rank-3 input to 2-D first.

### The A/B that named the cause: kernel class, not tuning

Three host paths over the same weights and the same shapes, in separate process runs
because the choice is a process-global `OnceLock` over `XWEN_ATTN_MM_CLASSIC`. TFLOPS:

| shape | T | candle steel | xwen classic simdgroup | xwen Metal-4 tensor |
| --- | --- | --- | --- | --- |
| K 3840, N 3840 | 4128 | 15.43 | 15.30 | 41.56 |
| K 3840, N 10240 | 4128 | 15.57 | 14.91 | 38.99 |
| K 10240, N 3840 | 4128 | 14.81 | 14.48 | 36.05 |
| K 3840, N 3840 | 1056 | 14.08 | 13.91 | 32.41 |
| K 3840, N 10240 | 1056 | 14.64 | 13.63 | 34.97 |
| K 10240, N 3840 | 1056 | 14.22 | 12.23 | 32.54 |

xwen's own classic simdgroup kernel lands on candle's rate to within 5%. The same host
code switched to the cooperative-tensor kernel is 2.4 to 2.7x faster. That rules out
tuning: candle's GEMM is the wrong kernel class for this chip, and it explains the
dtype-blindness above, since a simdgroup-matrix kernel with f32 accumulate gets little
from a narrower input type where the tensor path is exactly what narrow types are for.

Those figures are the isolated run. Measured inside the full suite, after 60 s of GPU
work and with multi-gigabyte attention tensors alive, the same arm read 36.36, 32.36 and
30.41 at T 4128. A later quiet-tree re-run of the arm read 38.64, 38.08 and 36.59
against candle's 15.44, 15.57 and 14.08, a 2.45 to 2.60x, and those are the numbers the
projections below use. Contention moved the small rows most: the `[T,3840]` widen
measured 0.331 ms under contention and 0.182 ms clean.

### The step budget at the baseline rates

Projection FLOPs count the seven token-width linears per block, 176.9 M parameters, over
34 blocks. Attention charges all 34 blocks for full-T attention, which is slightly high
because the two `context_refiner` blocks see only the caption.

| | T 4128 | T 1056 |
| --- | --- | --- |
| observed step | 5.1 s | 1.2 s |
| projections at 15.6 TFLOPS | 3.19 s (62%) | 0.87 s (73%) |
| fused sdpa, 34 blocks | 0.72 s (14%) | 0.05 s (4%) |
| adaLN `broadcast_mul`, 4 per block | 0.18 s (3.5%) | 0.04 s (3.2%) |
| norms, silu, residuals, q/k/v copies | ~0.06 s | ~0.01 s |
| launch overhead at 3 us per dispatch | ~0.02 s | ~0.02 s |
| accounted | 4.17 s (82%) | 0.99 s (82%) |
| unaccounted | 0.93 s (18%) | 0.21 s (18%) |

Three secondary readings from the same bench. Fused `candle_nn::ops::sdpa` runs 21.27 ms
per block at T 4128 against 33.93 for the explicit matmul, scale, softmax, matmul chain,
a 1.6x win, but the fused kernel itself reaches only 12.30 TFLOPS: it wins by never
materializing the 1 GB T-by-T score matrix, not by arithmetic. `broadcast_mul` by a
`[1, 3840]` row runs at 48 GB/s where the contiguous `mul` does 530, and materializing
the row first then multiplying is 4.7x faster end to end at 0.281 ms even though it
writes an extra 31.7 MB. Per-dispatch overhead is 3 us, so even 200 dispatches a block
over 34 blocks is 20 ms and launches are not a lever here.

### The hardware ceiling, and how much of it is confirmed

State the peak as **about 70 TFLOPS fp16 and bf16**, at medium confidence: it is an
extrapolation from a 5-core A19 microbenchmark measuring 1024 fp16 FLOP per core per
cycle, scaled to 40 cores at an assumed 1.75 GHz, not a measurement. It is consistent
with Apple's "over 4x the peak GPU compute" claim for the M5 Max. Do not size against
the published measured figures instead: the public literature tops out well below that
and this machine has already beaten it, torch MPS bf16 having reached about 38 TFLOPS on
this very model at 512x512 and xwen's tensor gemm 36-38 in isolation. Treat the
literature as a floor and roughly 40-55% of the peak as locally demonstrated.

A 1024x1024 step is about 59-62 TFLOP against 12.3 GB of weight traffic, an arithmetic
intensity of 4800 to 5100 FLOP/byte where the ridge point at 70 TFLOPS and 614 GB/s is
114.
Reading the weights is about 20 ms of a multi-second step. This graph is compute-bound by
40x or more under every assumption, so achieved GEMM rate is the only thing that sets
the time:

| achieved rate | per step | 8 steps |
| --- | --- | --- |
| 14 TFLOPS | 4.23 s | 33.8 s |
| 20 TFLOPS | 2.96 s | 23.7 s |
| 28 TFLOPS | 2.11 s | 16.9 s |
| 36 TFLOPS | 1.64 s | 13.2 s |
| 38 TFLOPS (torch MPS, local) | 1.56 s | 12.5 s |
| 70 TFLOPS (peak) | 0.85 s | 6.8 s |

Attention is 15% of the step FLOPs at 1024x1024 and grows quadratically in token count,
so it becomes the dominant term at higher resolutions: at 2048x2048 the GEMM term goes
4x and the attention term 16x.

## The gemm arc

Every projection in the transformer now runs `crate::ops::matmul_bf16`, the Metal-4
cooperative-tensor kernel the language models prefill on. `src/zimage/linear.rs` is the
new seam: `Projection` holds a bf16 `[out, in]` weight and an optional f32 bias, reshapes
a rank-2 or rank-3 input to `[t, k]`, and calls the kernel on Metal.

**The activation stream is f32, and that was a design choice rather than a consequence.**
The kernel's contract is a bf16 weight against an f32 activation with f32 accumulation,
returning f32, so a drop-in inside a bf16 stream pays a widen going in and a narrow
coming out. Those casts were measured: 14 of them for the seven token-width linears cost
3.58 ms per block and 0.122 s per step, and the 11 achievable by sharing one widen
across q, k and v and one across the SwiGLU pair cost 3.04 ms and 0.103 s, about 9% of
the xwen linear figure. Keeping the whole block in f32 removes all of them. It costs nothing in
resident memory, the weights staying bf16 on the device at 12.3 GB, and it turned out to
cost nothing in time on the elementwise tail either, since the `candle` bisect arm also
runs the f32 stream and reproduces the old step time exactly. What it bought was parity,
below.

Norm weights, pad tokens and biases load f32; projection weights are fetched bf16 by
`Projection::new` through `vb.set_dtype(BF16)`. The FFN uses `ops::silu_mul` on Metal.
The pipeline's `dtype` field now means the activation dtype.

**The guard.** The kernel stages each weight tile to f16 on its way into the tensor unit,
so a weight past f16's finite range would be silent garbage. `ensure_weights_fit_f16`
refuses any projection with `|w| > 65504`, naming the tensor, at the end of
`Config`-driven construction. It costs 0.2 s of load, 3.0 s to 3.2 s. Two facts about the
shipped checkpoint make it safe: the largest weight is 14.0, in
`layers.6.feed_forward.w2.weight`, and 0.0347% of values (2,136,677 of 6,153,863,168)
sit below f16's normal floor and round or flush there, which the parity bars hold with.
The subnormal count was dropped from the load path rather than shipped, costing another
1.2 s for a number that only needed measuring once.

One loose end worth naming. A first version of the guard copied every plane to f32 and
batched 300 scalars through one `Tensor::stack`; it cost 17 s of load and once reported
a max of 11.0 in `layers.28.w1` where every other run said 14.0. That nondeterminism was
not chased. The shipped per-tensor `max_keepdim` then `max_all` form has reported 14.0 on
every run since, and a unit test pins it by agreeing Metal against CPU on a 1024x3840
plane.

**The bisect arm** is `XWEN_ZIMAGE_LINEAR=candle`, beside `XWEN_ZIMAGE_ATTN`: candle's
own bf16 gemm over bf16-rounded activations, the path every projection ran before this.
It shares no matmul code with the shipped arm, and a unit test asserts the two agree to
a nonzero relative L2 of 2.34e-3 under a 5e-3 bar, which is the shape a reference arm's
test has to have (AGENTS.md "Verification workflow").

**The weights were bf16 all along.** The three shards are labelled F32 and hold bf16
values upcast: across a 117 M-word sample the low mantissa bits are zero everywhere. So
the load-time cast to bf16 loses nothing at all, and the f16-range question was only ever
about magnitude, never about mantissa.

### Results

| | before | after |
| --- | --- | --- |
| 1024x1024 step | 5.02-5.25 s | 3.06-3.59 s |
| 1024x1024 VAE decode | 4.88 s | 5.01 s (untouched) |
| 1024x1024 total, warm | 50.2 s | 37.7 s |
| 512x512 step | 1.17-1.25 s | 0.63-0.68 s |
| 512x512 VAE decode | 1.13 s | 1.08 s |
| 512x512 total, warm | 14.7 s | 10.3 s |
| 512x512 step, `XWEN_ZIMAGE_LINEAR=candle` | | 1.20-1.26 s |
| transformer and VAE load, warm | 3.0 s | 3.1-3.2 s |

That is 1.5x per step at 1024x1024 and 1.85x at 512x512, and about 17-20 TFLOPS end to
end at 1024x1024 against 11.6-12.4 before. A second run of the same build in the same
session read the 1024x1024 step at 3.04-3.52 s and the total at 36.2 s, which is the
spread to expect. The step now creeps upward across the eight steps of a run, 3.06 early
and 3.59 late. The 24-step run below settled what that is: a bounded warm-up ramp that
plateaus at 3.55-3.67, so **3.6 s is the figure and 3.06 s is a first step**. The `candle` arm
reproducing 1.20-1.26 s at 512x512 is what proves the arm is live and that the f32
elementwise traffic costs nothing visible.

Parity improved, which is the part that was not predicted:

| 512x512, prompt 1, seed 0 | before | after | bar |
| --- | --- | --- | --- |
| step-0 velocity cosine | 0.999302 | 0.999999 | >= 0.998 |
| step-0 velocity mean rel | 0.0205 | 0.0008 | <= 0.04 |
| step-0 velocity max rel | 0.1006 | 0.0045 | reported |
| final latent cosine / mean rel | 0.990845 / 0.0631 | 0.999708 / 0.0064 | reported |
| image PSNR against the fp32 reference | 29.71 dB | 47.03 dB | reported |
| VAE alone | 92.62 dB | 92.62 dB | >= 60 |

Both wrong-graph brackets stayed outside, at 0.610825 for a timestep one grid point off
and 0.870767 for a reversed caption. The reference's own bf16 arm sits at 0.999560 and
32.40 dB, so xwen has gone from 1.6x noisier than torch's bf16 to well inside its
spread, and the old narrative that the graph was right but its arithmetic a little
noisier than torch's is now history: it was the bf16 activation stream, and there is no
longer one.

### The attention cast experiment

candle's full steel sdpa kernel accepts f32 at head_dim 128, so with the stream in f32
there were two options: hand it f32 q, k and v as they are, or cast to bf16 and back
around it. Both were run back to back at 1024x1024: f32 read 3.04-3.49 s per step and
the bf16 round trip 3.02-3.61 s. No gain, so no casts, and `attention_metal` carries a
comment recording it.

## What the arc left open, before the profiler ran

This was the ranking at the end of the gemm arc, kept because it is what the profiler was
sent to settle and because two of its guesses were wrong. The settled version is the
lever ledger below.

- **A bf16 mixed-operand instantiation of the tensor gemm**, taking a bf16 activation
  tile directly. It was the alternative to the f32 stream, and the f32 stream made it
  moot: with nothing to widen there is no cast to avoid, and the 0.10-0.12 s per step it
  would have saved does not exist any more. The f16 family already has the probe
  (`src/ops/f16_t_mixed.metal`, reachable as `F16MmKernel::TensorMixed` and constructed
  only by tests), so the pattern is there if the stream ever has to go back to bf16 for
  an attention or elementwise reason.
- **Attention** is already fused and still reaches only 12.30 TFLOPS, a third of what the
  tensor gemm does on the same chip. It is 0.72 s of a step that is now 3.6 s rather than
  5.1, so its share has roughly doubled to 21%. A flash kernel at the gemm's rate would be
  worth about 0.5 s per step at 1024x1024. The profiler confirmed the share at 740 ms and
  it is a Front item now; it is also the first term at any resolution above 1024x1024,
  where attention grows 4x faster than the GEMM term.
- **The adaLN `broadcast_mul` chain** runs at 48 GB/s against 530 for the contiguous
  kernel, four applications per block, 180 ms at 1024x1024 and 38 ms at 512x512, down to
  under 40 ms and 8 ms by materializing the `[1, 3840]` row before multiplying. It was
  3.5% of the old step and is about 5% of the new one. A small local change in the
  modulation path, no new kernel.
- **The unattributed 18%**, at both resolutions, was the reason there was a profiler pass
  at all, and the candidates named here were the right ones: the per-block f32 rope
  rotation at ~465 ms, the four attention copies at ~235 ms and the norms at ~370 ms
  cover it and more. Two of the guesses around them were wrong. The 32 adaLN gemv
  dispatches are 7.7 ms a step and irrelevant, and the two `to_dtype(F32)` calls the ops
  map charged to rope are now free, candle returning a clone when the dtype already
  matches and the stream being f32 since e54801f. The rope row is 6-7x its own traffic
  for a reason no capture was needed to find: strided views.
- **The VAE decode**, untouched at 4.93-5.15 s and now 13% of a 1024x1024 image rather
  than 10%. It is 9.89 TFLOP of f32 convolution running at about 2.0 effective TFLOPS through
  candle's im2col-plus-matmul path, whose 9x buffer inflation and power-of-two rounding
  leave a 16 GB pooled Metal buffer resident for the life of any process that has decoded
  at 1024x1024. Cheapest known wins, unpriced: `Activation::Swish` to the fused
  `Activation::Silu` at 28 sites (numerically identical, one kernel instead of two plus a
  full-size intermediate), SDPA for the mid-block's naive 16384-token materialized
  softmax with its two 1.07 GB allocations, and a fused GroupNorm to replace nine
  full-tensor passes. torch MPS extrapolates to roughly 2.8 s at 1024x1024, so the gap
  is about 1.75x and not an order of magnitude.
- **candle's steel tile config** (`TILE_64_64_16_1_2` for these shapes) was on the lever
  list before the A/B and is now unreachable: the shipped path does not call candle's
  gemm at all.

## The profiler, and how to read it

`XWEN_ZIMAGE_PROFILE=1` is a third env switch beside `XWEN_ZIMAGE_ATTN` and
`XWEN_ZIMAGE_LINEAR`, read the same way and refusing an unrecognized value.
`src/zimage/profile.rs` is a `Profiler` holding the device and a mutex, `mark(label)`
syncing and charging the interval to that label, and two tables printed at the end of a
run: one for the transformer, the mean per step over steps 2 through 8, and one for the
VAE decode. The transformer marks are per phase and per stage, so `noise_refiner`,
`context_refiner` and the 30 main layers are separated, and everything outside the block
loops has its own rows. Off, the cost at each site is one `Option` check. No math
changed: the parity gate is green with the profiler in.

**A profiled table is not a set of figures**, and the reason is not only the syncs.
candle's `wait_until_completed` calls `drop_unused_buffers`, so every mark also evicts the
buffer pool and the next op re-allocates and re-zeros a power-of-two-rounded buffer.
Measured inflation against the same run's unprofiled steady state:

| phase | unprofiled | profiled table total | factor |
| --- | --- | --- | --- |
| transformer step, 1024x1024 | 3.45-3.51 s | 4.86 s | 1.39x |
| transformer step, 512x512 | 0.63-0.69 s | 1.07 s | 1.60x |
| VAE decode, 1024x1024 | 4.93 s | 7.14 s | 1.45x |
| VAE decode, 512x512 | 1.08 s | 1.70 s | 1.57x |

The four gemm rows and the sdpa row are single large dispatches, and their milliseconds
hold up against arithmetic: `attn.qkv` 4.06 ms and `attn.out` 3.85 ms for 121.7 GFLOP
each is 30.0 and 31.6 TFLOP/s, `ffn.w1w3` 13.24 ms and `ffn.w2` 8.39 ms for 324.6 GFLOP
each is 24.5 and 38.7. That is what licenses deflating the rest. Real gemm work per step
is 46.7 TFLOP at those rates, about 1.53 s, and sdpa is 8.4 TFLOP at 0.74 s; against a
3.45 s unprofiled step that leaves about 1.18 s of real elementwise and copy work where
the table claims 2.27 s, so the small rows read about 1.9x high and are deflated by that
factor below. Take a gemm or sdpa row as a measurement and a small row as an upper bound
(decisions.md "A Z-Image step is quoted at steady state").

**A step is quoted at steady state.** Over 24 steps at 1024x1024 from a cool machine the
per-step time ramps 3.06, 3.11, 3.22, 3.36, 3.41, 3.46, 3.49, 3.57 and then goes flat at
3.55-3.67 for the remaining sixteen with no further trend; a later 8-step run on a
machine that had cooled started at 3.05 again. So the ramp is reproducible, reversible
and bounded at +19%. **Quote 3.6 s for a 1024x1024 step and 3.06 s for a first step, and
never an 8-step mean**, which is an artifact of how many steps the run happened to ask
for.

## Lever ledger

At 1024x1024, against a 3.6 s steady-state step and a 37.7 s warm image of which about
33.8 s is the render and about 4 s is load and encode. Millisecond figures are the
deflated ones above; the gemm and sdpa rows are measured, the rest are deflated
estimates. Gain per image is the row's saving times eight steps, or the row itself for
the VAE.

| lever | measured today | ceiling, and how it was derived | gain per image | cost class |
| --- | --- | --- | --- | --- |
| the four gemms | 1530 ms/step, 30-39 TFLOP/s | about 70 TFLOP/s hardware peak, extrapolated from a 5-core A19 at an assumed clock, medium confidence | none planned | no work: this is the rate the kernel gives |
| `attn.sdpa` | 740 ms/step, 11.3 TFLOP/s | ~300 ms at the gemms' own rate, via a bidirectional flag on `ops::flash_attn` | ~3.5 s | new kernel mode: one mask site, two block-skip bounds, a host guard, f16 K/V, parity re-run |
| `attn.rope` | ~465 ms/step | ~70-120 ms fused: 700 MB per call at 400 GB/s is 119 ms per step over 68 calls, so today is 6-7x its own traffic | ~2.9 s | new kernel, interleaved-pair, parity re-run |
| norms and modulation | ~370 ms/step | ~100 ms with a norm-with-scale kernel folding five full-tensor passes per block | ~2.2 s | new kernel, no math change |
| the f32 write at 10240 wide | ~300 ms/step | ~0: `ffn.w1w3` at 24.5 TFLOP/s against `ffn.w2`'s 38.7 for identical FLOPs is the 169 MB it writes | ~2.4 s | a `silu_mul` gemm epilogue (no precision change) or a bf16 intermediate (a precision change the gate arbitrates) |
| the four attention copies | ~235 ms/step | partly removable; the output-side copy alone if candle's SDPA cannot read token-major | up to ~1.9 s | layout question first, then existing `ops::permute_01` |
| `ffn.silu_mul` | ~118 ms/step | absorbed by the epilogue above | ~0.9 s | same edit as the f32 write |
| VAE decode | 4.93 s cool, 5.15 s hot | 9.89 TFLOP at the gemms' rate is ~0.3 s; 1-1.5 s realistic with a direct 3x3 conv or MPSGraph | 3.4-3.9 s | new conv path; bf16 is refuted, below |

Taking the rope, sdpa, norm and f32-write rows puts a step at about 2.2 s; adding
`silu_mul` and the removable half of the copies puts it near **2.1 s**. Eight of those is
16.8 s against today's 28.8, so with the VAE at 1-1.5 s the **render goes from about
33.8 s to about 18 s** and the warm wall from 37.7 s to about 22 s, the 4 s of load and
encode being untouched by any of it. Two Front items carry the two largest rows and the
rest are area items (TODO.md, "Image generation").

Three things are priced as non-levers rather than argued away.

- **The per-step readback is free.** `latents.flatten_all()?.get(0)?.to_scalar::<f32>()?`
  reaches `drop_unused_buffers` and evicts the buffer pool once per step, which looked
  like a real cost. Removing it entirely reads 35.8 s total against a 35.3 s baseline at
  1024x1024, the work moving into the VAE number rather than disappearing. Keep it: it is
  what makes per-step timing mean anything.
- **`context_refiner` is 13 ms/step** at a 32-token caption, and everything outside the
  block loops, meaning patchify, the embedders, the rope tables, the final layer and the
  Euler step, is 18 ms/step. Neither is worth an arc.
- **The VAE in bf16 is refuted**, measured and reverted: 4.79 s against 4.93 at
  1024x1024, 1.09 against 1.08 at 512x512, and the VAE-alone PSNR falls to 54.38 dB
  against the 60 dB bar. Halving every byte of conv traffic being worth 140 ms is the
  interesting half: the decode is not bandwidth-bound at f32, so only the conv structure
  can move it (decisions.md "The VAE decodes in f32 and bf16 is refuted").
