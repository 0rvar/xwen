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

## 2026-09-08 — The elementwise rows: a one-pass rope kernel, the adaLN scale folded into the norm weight, a fused gated residual

bee11da, the first of two arcs off the ledger below. Three rows, no math change, and the
parity gate re-run on all of them. Measured on this machine on 2026-09-08 with `pmset -g`
reading `lowpowermode 0` and no `powermode` key; no high-power claim is made. Dev-tree
release builds rather than pinned binaries, GPU otherwise idle under the lock.

### The interleaved-pair rope, 892 ms of profiled table to 126

`apply_rotary_emb` ran as a candle chain over stride-2 views of the even and the odd
elements, so all six of its binary ops took candle's strided path with uncoalesced reads
and `Tensor::stack` copied the result back. `ops::rope_pair` is one kernel over
`[batch, seq, heads, head_dim]` f32 with `[seq, head_dim/2]` f32 cos and sin tables, one
thread per pair, reading and writing its own two adjacent floats.

It is **bitwise identical** to the chain, and the reason it can be is that it evaluates
the same expressions with FP contraction and reassociation pinned off, so every multiply
and add rounds where candle's separate dispatches rounded. The test asserts that identity
at the production shape `[1, 4128, 30, 128]` and at two small ones, which matters because
the f32 rotation is one of the four deliberate corrections toward the reference
(zimage.md "What the vendored candle module got wrong"): a kernel that quietly rounded
somewhere else would have undone a correction while looking like a speedup.
`ops::rope_neox` was never a candidate, being by-halves where this graph is
interleaved-pair.

### The adaLN scale folded into the norm weight, and no kernel at all

The modulated block computed `rms_norm(x) * w` and then a full-tensor `broadcast_mul` by
`1 + scale`. Both factors are `[dim]` vectors, so `w * (1 + scale)` is a `[3840]` multiply
and the second full-tensor pass over 63 MB is not needed at all. `BlockNorm` holds the f32
weight and the eps, `forward` is the same fused `candle_nn::ops::rms_norm` the block's four
norms already ran, and `forward_scaled(x, scale)` does the fold and runs one norm. **No
kernel was written**: the ledger's norm-with-scale ceiling turned out to be reachable
without one.

It is the one part of the arc that is not bitwise. `rms(x) * w * s` and `rms(x) * (w * s)`
are the same math in a different rounding order of two f32 multiplies, and the difference
is invisible at step 0 and visible at step 8: the velocity field is unchanged to every
printed digit and the image PSNR moves 47.03 to 45.75 dB. That cost is accepted rather
than absorbed, and the reasoning is a decision paragraph (decisions.md "The adaLN scale
folds into the norm weight").

### The gated residual

`h + gate * y` ran as candle's broadcast multiply, which the microbench priced at 48 GB/s
against 530 for the contiguous kernel, followed by an add. `ops::gated_residual` is
`out = h + gate[c] * y` in one pass, one thread per element, bitwise identical to the pair
it replaces.

### Results

Steady state is steps 9 through 16 of a 16-step run (decisions.md "A Z-Image step is
quoted at steady state"). The first timing session ran straight off a test build and came
out thermally noisy; the figures below are from the second, after 45 s idle, which is the
one to quote.

| | before (5bfbe10) | after (bee11da) |
| --- | --- | --- |
| 1024x1024 steady-state step | 3.55-3.67 s | 2.62-2.74 s |
| 1024x1024 first step of a cool run | 3.06 s | 2.14 s |
| 512x512 step | 0.63-0.68 s | 0.45-0.52 s |
| VAE decode, 1024 / 512 | 4.9-5.0 / 1.08 s | 5.21 / 1.38 s, untouched, thermal spread |

The sixteen per-step times of that run, in order: 2.14, 2.24, 2.54, 2.48, 2.55, 2.60,
2.65, 2.63, 2.65, 2.64, 2.65, 2.62, 2.74, 2.69, 2.71, 2.70. The warm-up ramp keeps the
bounded, reversible shape it had before, on a plateau about 0.9 s lower.

Profiled rows, mean per step over steps 2 through 8 at 1024x1024, in profiled
milliseconds and so inflated:

| row | before | after |
| --- | --- | --- |
| `attn.rope` | 892.43 | 125.82 |
| `attn.norm+scale` | 235.75 | 159.19 |
| `ffn.norm+scale` | 200.54 | 124.71 |
| `attn.gate+residual` | 66.48 | 26.11 |
| `ffn.gate+residual` | 65.84 | 25.16 |
| `attn.qknorm`, untouched | 128.10 | 130.55 |
| transformer total | 4862.91 | 3831.88 |

Parity, the whole gate:

| metric | before | after | bar |
| --- | --- | --- | --- |
| step-0 velocity cosine | 0.999999 | 0.999999 | >= 0.998 |
| step-0 mean relative error | 0.0008 | 0.0008 | <= 0.04 |
| step-0 max relative error | 0.0045 | 0.0037 | reported |
| final latent cosine / mean rel | 0.999708 / 0.0064 | 0.999659 / 0.0073 | reported |
| image PSNR | 47.03 dB | 45.75 dB | reported |
| VAE alone | 92.62 dB | 92.62 dB | >= 60 |

Both wrong-graph brackets stayed outside, at 0.610812 for a timestep one grid point off
and 0.870471 for a reversed caption. The reference's own bf16 arm sits at 0.999560 and
32.40 dB, so 45.75 is still 13 dB above the arithmetic this pipeline exists to match.
`attn.qknorm`, two per-head norms per block, was outside this arc and is the norm-shaped
row that remains.

## 2026-09-08 — Bidirectional flash attention by query position, and the SwiGLU dual gemm refuted

66e7202, the second arc off the ledger, and the one that priced its own row differently
than the ledger had. Same conditions: `pmset -g` read `lowpowermode 0` with no
`powermode` key, no high-power claim, dev-tree release builds, GPU idle under the lock. It
was measured against 5bfbe10 rather than against bee11da, the two arcs having run in
parallel worktrees, so its before column is the pre-rope base and its gains do not add to
the section above.

### Bidirectional by query position, with no kernel edit

The ledger's plan was a bidirectional flag through `FlashAttnArgs`: drop the future test
in the one mask block, open the two block-skip bounds, relax the two host causal guards.
What shipped touches the kernel not at all. The causal kernel masks a key when
`col_abs > row_abs` (future) or `row_abs - col_abs >= window` (expired), and both tests
are vacuous if the queries sit at absolute positions K through K+T-1 while the keys sit at
0 through K-1 and the window is unbounded: every key is in the past of every query and
nothing is expired. The block-skip bound `kb_lim = min(NK, (q_hi - k_off) / BK + 1)` then
evaluates to NK on its own, so it needed no plumbing to open, and the in-kernel masking
loop runs only on an unaligned last key block. `run_flash_attn` dispatches on a private
`FlashMask { Causal { pos, k_off, window }, Bidirectional }`; the bidirectional arm sets
`q_off = K`, `k_off = 0` and `window = i32::MAX`, checks that `K + seq` fits in an i32,
and the one host guard it had to bypass is "each query's own key must be present". The
causal callers keep their guards verbatim and their tests still report bitwise identity.

**`ops::flash_attn_bidirectional` is bitwise identical to candle's unmasked f32 sdpa** at
every shape tested: the production one at 30 heads and T = K = 4128, an unaligned 203, a
two-token sequence, 40 queries over 64 keys, and a GQA 8-into-2 case at 45. A second test
confirms the entry differs from the causal one on row 0 and agrees bitwise on the last
row, which is what makes the first assertion evidence instead of one kernel compared with
itself (AGENTS.md "Verification workflow").

`AttnImpl` replaces the old `use_accelerated_attn` boolean with three arms, and
`XWEN_ZIMAGE_ATTN` gained a value: `flash` or `xwen` or unset is the shipped path, `fused`
or `sdpa` is candle's kernel and was the previous default, `basic` is the reference matmul
chain. The head_dim check is in `ZImageAttention::new`, so a config with another head dim
takes the fused path at load rather than failing in the middle of a render. k and v go
through `ops::permute_01_f16`, which does the permute and the f16 cast in one pass, so
they reach the kernel at half width; q and the output go through `ops::permute_01`. It is
still four passes, but each of them is single-pass now. A masked call, which only the
tests make and the pipeline never does, runs candle's sdpa as before and equals the fused
arm bit for bit. On the tiny model the flash arm sits 5.398e-6 from the basic chain
against a 1e-4 bar, with the mutation bracket above 1e-2.

### The finding: the flash kernel runs at candle's rate, because it is candle's kernel

The ledger priced this row at about 300 ms at the gemms' own rate, and that number was
wrong for a reason worth writing down. The vendored flash kernel is a copy of candle's MLX
steel attention: simdgroup matmul with f32 accumulate, which is the kernel class the gemm
A/B of 2026-09-07 identified as the wrong one for this chip. Switching to it did not change
the arithmetic path at all. **`attn.sdpa` moved 740 to 687 ms profiled**, 655 on a second
run, which on the profiled basis is roughly 11.3 to 12.5 TFLOP/s, and that is the same rate
within noise. Deflated against the merged step the real rate is about 16 TFLOP/s, so the
f16 k and v did lift it; what did not move is the kernel class.

What the arm bought is traffic and copies. `attn.transpose` went 328 to 158 ms profiled and
`attn.untranspose` 121 to 76, from the f16 k and v and the fused permutes, and that is where
the measured step-time gain comes from. Attention at the gemms' rate needs a Metal-4
tensor-op attention kernel: `matmul2d` for both products with an online softmax over
cooperative-tensor elements and P staged through threadgroup memory. It is a new kernel and
not a flag. The merged profiler pass sized it properly at 0.53 s per step against 0.23 at
the gemms' rate, so 0.30 s per step and **2.4 s per image**, and it is the Front item this
arc promoted in place of itself.

### The SwiGLU dual gemm: built, correct, and slower

`kernel_mul_mm_bf16_f32_swiglu_t` staged the w1 and w3 bf16 planes into two half tiles
against one activation tensor, ran two `matmul2d` accumulations into two destination
cooperative tensors, and applied `cGate[i] = silu(cGate[i]) * cUp[i]` over
`get_capacity()` under `is_valid_element(i)` before one store. Two notes for whoever writes
the next epilogue: the header's `get_mask`, which the vendor doc example uses, does not
exist, and `is_valid_element` is what does; and the `t <= 8` and `XWEN_ATTN_MM_CLASSIC`
chain fallbacks have to be carried through, which the entry did.

It was numerically right. Relative L2 against the two-gemm plus `silu_mul` chain was 5e-8
at every shape tried, 7.4e-4 against a CPU f32 reference, which is the tensor path's own
class, and the `t <= 8` entry was bitwise the chain.

It was also slower, isolated and in situ. Three tile variants at the model's own shape,
T 4128, K 3840, N 10240, both products counted:

| variant | dual kernel | two gemms plus `silu_mul` |
| --- | --- | --- |
| 64 rows per plane, 4 simdgroups | 33.7 ms, 19.2 TFLOP/s | 15.1 ms, 43.1 |
| 32 rows per plane, 4 simdgroups | 17.7 ms, 36.7 | 15.9 ms, 40.8 |
| 64 rows per plane, 8 simdgroups | 17.3 ms, 37.6 | 14.9 ms, 43.5 |

The best variant is 15% off the chain isolated. In the pipeline, alternating unprofiled A/B
at 1024x1024 over 16 steps with the steady state read from steps 9 through 16:

| run order | flash only | flash plus dual gemm |
| --- | --- | --- |
| first pair | 3.36-3.55 s, ~3.42 | 3.43-3.52 s, ~3.49 |
| second pair | 3.39-3.49 s, ~3.46 | 3.55-3.63 s, ~3.58 |

Two to four percent slower in situ, in both orders. It was removed whole, kernel, host
entry, tests and wiring, and `bf16_t.metal` is byte-identical to what it was. The reopen
condition is a variant whose per-thread accumulator footprint matches the single kernel and
whose activation tile is read once; the 8-simdgroup variant did both and still lost 15%, so
the loss is most likely the second staging pass plus the epilogue's exponential per element
inside a tile that is already compute-bound. A bf16 output store was not tried, being a
precision change (decisions.md "The SwiGLU dual gemm is REFUTED").

**And the profiler said it was a 410 ms win.** The fused row read 769.5 ms against the
chain's 904.6 plus 275.1, which is 1179.7, so the profiled table credited the fusion with
410 ms per step while the unprofiled A/B measured a regression. The mechanism is the one
already recorded for the small rows: `wait_until_completed` at every mark evicts candle's
buffer pool, so the chain is charged for re-allocating and re-zeroing the 169 MB
intermediate the fused kernel never materializes, and a fusion whose whole point is to
remove an intermediate is precisely the change that penalty flatters. The rule that came out
of it is a decision (decisions.md "A profiled row that shows a fusion win is not a result
until the fusion is confirmed unprofiled").

### Results

| | before (5bfbe10) | after (66e7202) |
| --- | --- | --- |
| 1024x1024 steady-state step | 3.55-3.67 s | 3.42-3.46 s, hot A/B session |
| 1024x1024 8-step run, cool | 3.06 first, 3.57 by step 8 | 2.89 first, 3.04-3.25 |
| 512x512 step, and total wall | 0.63-0.68 s, 10.3 s | 0.56-0.58 s, 9.7 s |
| VAE decode, 1024 / 512 | 5.01 / 1.08 s | 5.20 / 1.08 s, untouched |

Read that as 0.15 to 0.2 s per step at 1024x1024, about 5%, and 0.07 s per step at 512,
about 11%. The machine ran hot for the whole session, both arcs timing back to back, so the
cool 8-step run's 3.04-3.25 is the optimistic reading and the hot A/B the conservative one.

Parity held, the attention itself being bitwise:

| metric | before | after |
| --- | --- | --- |
| step-0 velocity cosine / mean rel / max rel | 0.999999 / 0.0008 / 0.0045 | 0.999999 / 0.0009 / 0.0039 |
| final latent cosine / mean rel | 0.999708 / 0.0064 | 0.999684 / 0.0066 |
| image PSNR | 47.03 dB | 46.98 dB |
| VAE alone | 92.62 dB | 92.62 dB |

Brackets outside at 0.6107 and 0.8709.

### On the merged master

Timed after both arcs landed, bee11da and 66e7202, warm, `pmset -g` reading
`lowpowermode 0`. **The two arcs compose almost perfectly**: 2.15 s of steady state against
2.62-2.74 for the rope arc alone and 3.42-3.46 for the attention arc alone, from a
3.55-3.67 s base.

| | 1024x1024 | 512x512 |
| --- | --- | --- |
| steady-state step | **2.15 s** | 0.37-0.41 s |
| first step of a warm run | 1.78 s | |
| VAE decode | 5.19 s | 1.10 s |
| render, 8 steps plus the decode | ~21 s | |
| total wall, warm | **~25 s** | 8.6 s |

The twelve per-step times, in order: 1.78, 1.77, 1.85, 1.93, 1.98, 1.99, 2.10, 2.13, 2.15,
2.17, 2.15, 2.15. The ramp is the familiar bounded, reversible one, now +21% over 8 steps
onto a 2.15 s plateau.

Against where the day before started, a 1024x1024 step went **5.0-5.25 s to 2.15 s and a
warm image 50.2 s to about 25 s**, of which 3.6 s and 37.7 s was the state this record
opened at.

Parity on the merged tree, and the norm fold's rounding order is still the only thing that
moved:

| metric | 2026-09-07 | merged | bar |
| --- | --- | --- | --- |
| step-0 velocity cosine | 0.999999 | 0.999999 | >= 0.998 |
| step-0 mean / max relative error | 0.0008 / 0.0045 | 0.0008 / 0.0041 | <= 0.04 / reported |
| final latent cosine / mean rel | 0.999708 / 0.0064 | 0.999672 / 0.0070 | reported |
| image PSNR | 47.03 dB | 46.09 dB | reported |
| VAE alone | 92.62 dB | 92.62 dB | >= 60 |

Both brackets outside. The reference's own bf16 arm is 32.40 dB, so the merged tree sits
14 dB above it. The full suite is green: 1336 lib tests passed and 34 ignored, every
integration target passing.

**A 2.15 s step is about 29 TFLOP/s end to end**, against 17 at 3.6 s and 11.6-12.4 before
any of this work.

### The merged profile, and the deflator refitted

Profiled at 1024x1024, mean per step over steps 2 through 8, 3371 ms of profiled total
against the 2.15 s real step. Profiled milliseconds on the left, real on the right where the
row can carry one.

| row | profiled | real |
| --- | --- | --- |
| `ffn.w1w3` | 799 | see the caveat below |
| `attn.sdpa` | 636 | ~536 |
| `attn.qkv` | 368 | |
| `ffn.w2` | 248 | |
| `ffn.silu_mul` | 236 | |
| `attn.norm+scale` | 162 | |
| `attn.qknorm` | 135 | |
| `attn.rope` | 129 | |
| `attn.transpose` | 126 | |
| `ffn.norm+scale` | 115 | |
| `attn.out` | 80 | |
| `attn.untranspose` | 48 | |
| `attn.gate+residual` | 25 | |
| `ffn.gate+residual` | 25 | |
| `embed` / `adaln` / `final` / `euler` | 7 / 7 / 6 / 3 | |
| the four gemms together | 1495 | **~1260** |
| the nine elementwise and copy rows together | 1001 | **~354** |

**This is where the budget closes, and it closes with two deflators rather than one.** The
four gemm rows sum to 1495 ms profiled. Their arithmetic is 46.7 TFLOP, and at the
~37 TFLOP/s the kernel measures in isolation at T 4128 that is **1.26 s real**, so the gemm
marks carry **1.19x**. The important half of that is what it says about the instrument: the
sync-and-evict costs the gemms too, not only the small rows, so the earlier reading that
treated a gemm row as a measurement was already about 19% high. `attn.sdpa` deflates by the
same factor to **~0.53 s**, which is about **16 TFLOP/s** on roughly 8.4 TFLOP of attention,
up from 11.3 before the f16 k and v. The remainder, 2.15 less 1.26 less 0.53, is
**~0.34 s real** for every elementwise row and every copy, against 1001 ms of profiled rows,
so those carry about **3x** and **only their sum may be quoted**. Attributed inside that sum,
in real seconds: norm+scale ~0.09, `attn.qknorm` ~0.05, `ffn.silu_mul` ~0.08, the four
copies ~0.06, `attn.rope` ~0.04, the two gated residuals ~0.02.

One loose end, stated rather than smoothed. The listed rows sum to 3155 of the 3371 profiled
total, leaving 216 ms in the per-phase refiner rows that are not broken out here. And one
number to know the basis of: `ffn.w1w3` is quoted below at 0.90 s from its own FLOPs,
22.1 TFLOP at the 24.5 TFLOP/s the profiler measures for it, where the 1.19x aggregate
deflation would put it at ~0.67 s. The two bases differ because the 1.26 s bucket prices all
four gemms at one average rate and the per-row figure prices this one at its own. The
SwiGLU lever is sized on the per-row basis, so if the aggregate basis holds instead its gain
is nearer 1 s per image than 2.6, and `tests/zimage_microbench.rs` settles it cheaply at
these shapes.

The VAE decode profiles at 6792 ms against 5.19 s real, so 1.31x, and its rows are where
the conv work actually sits:

| VAE row | profiled |
| --- | --- |
| `up3.resnets` | 2343 |
| `up2.resnets` | 1294 |
| `up2.upsample` | 984 |
| `up1.resnets` | 567 |
| `up1.upsample` | 523 |
| `norm_out+conv_out` | 547 |
| `up0` | 252 |
| `mid` | 242 |

`up3.resnets` alone is a third of the decode and the top two resolutions are about two
thirds of it, which is the same reading as 2026-09-07 and is what makes a direct 3x3 conv
the lever rather than anything global.

## Lever ledger

Rewritten 2026-09-08, after the two arcs above and the profiler pass on the merged tree.
The base is the **measured 2.15 s steady-state step** and a warm image of about 25 s, whose
render is **22.4 s, being 17.2 s of eight steady-state steps plus the 5.19 s decode**, with
about 3 s of load and encode outside it. Every figure below is a REAL figure off the
deflation in "The merged profile, and the deflator refitted" above: gemms and sdpa at 1.19x,
the elementwise sum at about 3x. Gain per image is the row's saving times eight steps, or
the row itself for the VAE.

| lever | today | ceiling, and how it was derived | gain per image | cost class |
| --- | --- | --- | --- | --- |
| the VAE conv path | 5.2 s per image | 1.0-1.5 s with a direct 3x3 conv or MPSGraph; 9.89 TFLOP at the gemms' rate is ~0.3 s, so 1.0-1.5 is the realistic form | **3.7-4.2 s** | new conv path, but the cheap wins come first and may be worth 0.5-1 s alone: fused silu at the 28 `Activation::Swish` sites, SDPA for the mid-block's materialized 16384-token softmax, a fused GroupNorm for nine full-tensor passes. bf16 is refuted, below |
| the f32 store in the SwiGLU pair | 0.90 s/step, `ffn.w1w3`'s 22.1 TFLOP at 24.5 TFLOP/s | 0.57 s at `ffn.w2`'s own 38.7 for the same kernel and the same shape class; the difference is the 169 MB of f32 it writes against w2's 63, so the route is a bf16 SwiGLU intermediate | **2.6 s**, on the per-row basis; ~1 s if the aggregate basis holds | a precision change the parity gate arbitrates. The `silu_mul` gemm-epilogue route is REFUTED (decisions.md "The SwiGLU dual gemm is REFUTED") |
| a Metal-4 tensor-op attention kernel | 0.53 s/step at ~16 TFLOP/s | 0.23 s at the gemms' own rate; the flag route is spent, the vendored flash kernel being a copy of candle's steel attention | **2.4 s** | new kernel: `matmul2d` for both products, online softmax over cooperative-tensor elements, P through threadgroup memory, head_dim 128, bidirectional, f32 accumulate |
| the gemm rate beyond that | ~1.26 s/step for the four gemms at 30-39 TFLOP/s | ~1.0 s, UNPRICED: fuse q, k and v into one N=11520 gemm and the SwiGLU pair into one N=20480, then tune the tiles for M around 4000, which no sweep has covered | ~2 s | host-side plus tile work, no new kernel class, and **cheap to price**: `tests/zimage_microbench.rs` already runs these shapes |
| the remaining elementwise and copies | ~0.34 s/step for all of them together: norm+scale ~0.09, `attn.qknorm` ~0.05, `ffn.silu_mul` ~0.08, copies ~0.06, `attn.rope` ~0.04, gates ~0.02 | ~0.15 s with gemm epilogues and a fused per-head `attn.qknorm`; no single row is worth an arc, which is why they are one line | ~1.5 s | small kernels, no math change |

**The dense bf16 floor is a ~1.25 s step, ~10 s of steps, a ~1.2 s decode and a ~11.5 s
render**, against 22.4 s today, with every row above taken. It is not the sum of the
savings, because the SwiGLU f32 store lives inside the four-gemm bucket: taking the store
and then fusing the pair does not pay twice. Read the floor as the figure and the rows as
the ranking.

Below that floor nothing is a kernel, and all three are record lines with reopen conditions
rather than ledger items.

- **Fewer denoising steps.** Six instead of eight saves 4.3 s of today's render and four
  saves 8.6 s; after the kernel work above those become 2.5 s and 5 s, so this gets less
  attractive as the step gets cheaper, not more. Turbo is distilled for eight and the model
  card says nine ([zimage.md](../zimage.md)), so it is an image-quality judgement and not a
  measurement, and the stock ComfyUI node's `quality` field is the natural switch to hang it
  on, that being the one knob the node already sends. Reopen on a product decision about how
  many steps an image gets.
- **Step caching**, TeaCache or a first-block cache, reusing part of a step's result across
  neighbouring steps. Worth 0 to 2 s and UNPRICED, and the reason the range starts at zero is
  structural: an 8-step schedule at static shift 3.0 spaces its sigmas far apart, which is
  the regime these caches work worst in, having been built for 30-plus-step schedules where
  adjacent steps barely differ. Reopen on a priced experiment, which is a day.
- **int8 gemms on the tensor units**, worth about 1.8x fp16 compute and so about 4.5 s of
  today's render. Weeks of work: W8A8 calibration and a new set of parity bars, because the
  existing ones are read against bf16 arithmetic. Reopen if sub-10 s at eight steps becomes a
  requirement rather than a nice number (decisions.md "The transformer runs bf16 end to
  end").

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
