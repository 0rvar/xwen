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

## 2026-09-08 — The VAE on a direct conv kernel, attention on the tensor units, and the SwiGLU bf16 store refuted

The third arc of the day, run as three worktrees off e630ebb against the three largest rows
of the lever ledger below as it stood that morning: the VAE conv path (3.7-4.2 s per image),
the SwiGLU f32 store (2.6 s on the per-row basis, ~1 s on the aggregate one) and a Metal-4
tensor-op attention kernel (2.4 s). Two shipped, a763c61 and e5d9775 on master; the third
was built, measured, refuted and kept on the branch `zimage-ffn` as 5e7a6ea so the code
stays readable beside its numbers. Conditions: `pmset -g` read `lowpowermode 0` in every
session, no high-power claim. The three implementers shared the machine behind a GPU lock,
so their absolute step times ran 2.2-3.6 s against the 2.15 s recorded steady state and only
their same-session A/B pairs are quoted here; the clean figures at the end were taken on
master e5d9775 alone.

### The VAE on an implicit-gemm conv, 5.2 s to 1.4 s

The decode ran candle's `conv2d`, which at these shapes is nine im2col copies of the input,
a narrow gemm, an NHWC permute and 16 GB of pooled buffers per decode, at 1.2-4.4 TFLOP/s.
`src/ops/conv2d_direct.metal` replaces it with an implicit-gemm f32 3x3 and 1x1 convolution
over NCHW as candle stores it, so nothing is permuted: a 256-thread threadgroup owns a
`TH x 16` pixel tile (TH 8, or 16 when `c_in <= 128`) for 64 output channels, stages the
input tile with its one-pixel halo and eight input channels of weights in about 26 KB of
threadgroup memory, and runs one k-step per (tap, 8 channels) on simdgroup 8x8 f32 matrix
ops, the A operand being a transposed `simdgroup_load` of the staged tile at the tap's
offset. The weights are permuted once at load to `[k*k, c_in, c_out]`. The bias is in the
store, and everything the decoder wraps around a conv is folded into the conv itself: the
GroupNorm affine and the silu on the input read, the 2x nearest upsample as a read at half
coordinates so the 4x tensor never exists, and the residual add in the store. GroupNorm is
then `src/ops/group_norm.metal`, one statistics read (sum and sum of squares of `x - x0`
over up to 64 slices per group, `x0` the group's first element so the single-pass variance
does not cancel) plus a per-channel fold into `scale` and `shift`; the normalized tensor is
written only for the mid-block attention's norm and the fallback path. Four kernel
instantiations cover every decoder conv in the shipped config; anything outside `k = 3` or
`k = 1`, stride 1, `c_in % 8 == 0` falls back to the candle chain.

| conv, at 1024x1024 output scale | direct | candle `conv2d` |
| --- | --- | --- |
| 512->512 at 128x128 | 7.1 ms, 10.9 TFLOP/s | 17.5 ms, 4.4 |
| 512->512 at 256x256 | 27.4 ms, 11.3 | 71.5 ms, 4.3 |
| 512->256 at 512x512 | 54.1 ms, 11.4 | 220.7 ms, 2.8 |
| 256->256 at 512x512 | 28.6 ms, 10.8 | 121.4 ms, 2.5 |
| 256->128 at 1024x1024 | 55.5 ms, 11.1 | 459.6 ms, 1.3 |
| 128->128 at 1024x1024 | 30.7 ms, 10.1 (6.3 at TH 8) | 266.2 ms, 1.2 |

At 10-11 TFLOP/s the decode's 9.89 TFLOP of convolution is about 0.95 s, and the rest of
the 1.4 s decode is the norms' statistics reads, the mid-block attention and the shortcuts.
Same binary, same session, arms back to back on the shared box:

| VAE decode | `XWEN_ZIMAGE_VAE=xwen` (default) | `XWEN_ZIMAGE_VAE=candle` | master e630ebb, pinned |
| --- | --- | --- | --- |
| 1024x1024 | 1.36 s, 1.44 s | 6.45 s, 6.52 s | 5.94 s, 6.09 s |
| 512x512 | 0.25 s, 0.26 s | 1.06 s, 1.27 s | |

The profiled table, for ranking only, went 8826 ms on the baseline binary to 1471 on the
xwen arm, with `up3.resnets` 3605 to 335, `up2.resnets` 1613 to 301, `up2.upsample` 1176
to 181 and `norm_out+conv_out` 615 to 5; the four conv-heavy rows are 70% of what is left
and `mid.attn` at 108 ms is 7%. Parity: VAE alone 92.62 to 92.32 dB against the 60 dB bar
(same f32 math, a different summation order and the shifted single-pass variance), image
PSNR 46.09 dB unchanged, step-0 lines unchanged. The rendered PNGs are the same picture on
both arms.

Two traps cost the arc time and are pinned in the code. The first kernel ran a flat
1.04 TFLOP/s at every shape, slower than candle: the simdgroup accumulator array
`acc[NPB][NCB]` was not staying in registers because the tile loops over constexpr bounds
were not unrolled, so every MMA went through memory, and `#pragma clang loop unroll(full)`
took it to 10-11 TFLOP/s with no other change. The fold kernel bounded its threads on
`threads_per_grid`, the rounded-up launch count, so stray threads wrote past `scale` and
`shift` into pooled buffers; the kernel tests passed by luck and the decoder-level A/B test
caught it as NaN. Glue kernels here bound on an explicit `n` in their args, every one.

Not taken now, each with its reopen condition:

- The mid-block attention, 108-166 ms profiled across the runs and now 7% of the decode.
  Reopen when the decode is wanted under 1 s; the fix is candle's SDPA or a flash arm on the
  `(1, 1, 16384, 512)` q/k/v, head_dim 512 being outside the flash kernels' shapes.
- The 16-row tile for the 256- and 512-channel convs. It was priced only for `c_in <= 128`,
  where it took 6.3 to 10.1 TFLOP/s; the deeper convs already read 10.8-11.4 at 8 rows.
  Reopen with the microbench if the last 10% is wanted.
- Folding the GroupNorm statistics read into the preceding conv's epilogue, one fewer read of
  each 537 MB activation at 1024x1024, about 14 of them at the top two resolutions, perhaps
  30-60 ms. Reopen with a profile that shows the norm rows; today they sit inside the resnet
  rows.
- An f32 cooperative-tensor conv through `matmul2d`. Unpriced; the `_t_hp` MoE family read
  25% below the f16 tiles, and 10-11 TFLOP/s on simdgroup MMA is probably near this device's
  f32 FMA peak. The tensor units pay for f16 or bf16 operands, which the 60 dB bar refused
  (decisions.md "The VAE decodes in f32 and bf16 is refuted"). Reopen only with an
  f16-weights, f32-activations variant that passes the bar.
- Footprint, not measured. The xwen arm allocates no im2col buffers, so the 16 GB and 8 GB
  pooled buckets the path map found should be gone on this arm; the permuted weight planes
  add about 200 MB of f32 beside the candle copies, which stay for the bisect arm. Worth a
  `footprint` check from a user shell for the serve route.

### Attention on the cooperative tensor ops, 13 to 45 TFLOP/s

The ledger said a kernel and not a flag, and `src/ops/flash_t.metal` is that kernel:
bidirectional flash attention for head_dim 128 whose QK^T and PV products both run through
`mpp::tensor_ops::matmul2d`, the primitive the gemms run on. One threadgroup per
(64-query block, head), four simdgroups. The threadgroup stages its 64 Q rows as half into
16 KB of threadgroup memory once, then every tensor op runs at `execution_simdgroup` scope,
each simdgroup owning 16 rows and walking the whole key range in blocks of 32 with no further
barrier: `S = Q K^T` with K read from device as an f16 tensor with no staging, columns past
T zero-filled by the extent clip and set to the finite minimum in the tail block, row max
and row sum through `reduce_rows` into row-reduction cooperative tensors, `P = exp2(S *
scale * log2e - m)` in place, O rescaled per row and skipped under `simd_any` when no
row's max moved, then `P` converted into the left input of the PV op through
`get_left_input_cooperative_tensor` and V read from device. O lives in its cooperative
destination tensor for the whole key loop and never touches threadgroup memory; the per-row
rescale works because `map_iterator` from O onto the S op's row-reduction tensor is
compatible on this device, queried once before the loop and packed into one bit per element,
so the hot loop does two-way selects and no index arithmetic. A device whose row-reduction
capacity is not two gets NaN rather than a wrong answer, by a guard in the kernel.

The per-simdgroup structure was forced, not chosen: input cooperative tensors,
`reduce_rows` and `map_iterator` are all `static_assert`ed to simdgroup scope in this SDK
(MacOSX26.5), so the threadgroup-scope design the brief sketched, 64x32 S tiles with the
softmax through threadgroup memory, cannot be written against these headers. Two more facts
for the next kernel here: `get_mask`, which the header's own example uses, does not exist
and `is_valid_element` does; and an f32 operand under `relaxed_precision` is consumed at
less than f16 precision (Q as f32 from device read rel L2 9.6e-4 against 4.5e-4 with Q staged
as half), which is why Q is staged and why every operand is half.

Isolated, 30 heads x 128, f32 q, f16 k and v, same session
(`cargo test --release --test zimage_microbench zimage_flash_kernels_only -- --ignored`):

| kernel | T=4128 | TFLOP/s | T=1056 | TFLOP/s |
| --- | --- | --- | --- | --- |
| `flash_attn_bidirectional`, the steel copy (`flash`) | 20.05 ms | 13.1 | 1.380 ms | 12.4 |
| `flash_attn_tensor`, 16 rows x BK 32 (`tensor`, default) | 5.80 ms | 45.1 | 0.473 ms | 36.2 |
| `flash_attn_tensor`, 16 rows x BK 64 | 7.28 ms | 36.0 | 0.669 ms | 25.6 |

3.5x at the 1024x1024 shape and 2.9x at 512x512, and by the attention flop count the kernel
is now past the gemms' 36-38 TFLOP/s on the same primitive. The attempts, in order and in
one session at T 4128: the first version with device f32 Q and per-element `map_iterator`
inside the loop, 18.3 ms; unrolled index loops, slot masks computed once and the lazy
rescale, 11.3 ms; Q held in a left-input cooperative tensor, 28.4 ms (refuted, 64 more
registers per lane); Q staged as half in threadgroup memory, 6.07-6.44 ms and more accurate,
which shipped; K and V tiles staged through threadgroup memory for the four simdgroups,
13.1 ms (refuted, the two barriers per block couple simdgroups that are otherwise
independent and the tensor op reads device operands well); P through a per-simdgroup half
threadgroup tile so PV runs half x half, 8.3 ms and wrong as written (refuted on time, never
debugged); BK 64, slower at both shapes and kept instantiated so the choice stays priced. A
diagnostic split put QK^T alone at 4.4 ms and QK^T plus softmax at 6.1, so the softmax is
about 1.7 ms of the 5.8 and the PV product hides behind it.

Whole image on the shared box, unprofiled, two runs per arm alternating, steps 6 to 8:

| arm | run | step 6 | step 7 | step 8 | image total |
| --- | --- | --- | --- | --- | --- |
| flash | 1 | 2.15 s | 2.27 s | 2.36 s | 27.9 s |
| tensor | 1 | 1.83 s | 1.85 s | 1.89 s | 24.6 s |
| flash | 2 | 2.27 s | 2.34 s | 2.32 s | 27.5 s |
| tensor | 2 | 1.77 s | 1.88 s | 1.97 s | 24.2 s |

About 0.4 s per step and 3.3-3.5 s per image in that session; profiled `attn.sdpa` 697.6 to
246.4 ms per step, no other row moving. Accuracy against candle's unmasked f32 sdpa over
five shapes including the production one: rel L2 1.6e-4 to 4.5e-4 against a 2e-3 bar, max
abs 2.0e-4 to 8.5e-4 against 2e-2, both bars derived from the f16 rounding of q, k and v.
Layer-level against the basic chain 6.8e-6 (the steel arm 5.4e-6), and the test asserts the
tensor and steel arms are NOT bit-identical, the shape a reference arm's test has to have.

| parity, 512x512 fixture, T = 1120 | step-0 cosine | mean rel | max rel | final latent | image PSNR |
| --- | --- | --- | --- | --- | --- |
| `tensor` | 0.999999 | 0.0010 | 0.0050 | 0.999627 / 0.0077 | 45.60 dB |
| `flash` | 0.999999 | 0.0008 | 0.0041 | 0.999672 / 0.0070 | 46.09 dB |
| reference bf16 against fp32, its own spread | 0.999560 | 0.0175 | 0.0890 | 0.994750 / 0.0492 | 32.40 dB |
| bracket, timestep one grid point off | 0.610796 | 0.6224 | | | |
| bracket, caption tokens reversed | 0.870805 | 0.3144 | | | |

The tensor arm costs 0.5 dB of image PSNR against the steel arm and sits 13 dB above torch's
own bf16 arm. `XWEN_ZIMAGE_ATTN` now names four arms, `tensor` (the default, alias `xwen`),
`flash` (alias `steel`, the previous default), `fused` (candle's sdpa) and `basic`, and
`AttnImpl::SHIPPED` names the default once so the pipeline's log guard and the serde default
cannot drift apart.

Not taken now: two independent 16-row groups per simdgroup sharing each K/V load, reopen if a
profile shows the kernel L1/L2-bound rather than register-bound (the Q-in-registers result
says 64 more registers per lane costs 2.5x today); the threadgroup-scope kernel, reopen only
if a future SDK lifts the simdgroup-scope static_asserts or the per-simdgroup kernel stalls
below the gemm rate at a new shape; fewer `reduce_rows` per block, about 1.7 ms of the 5.8,
reopen if attention is again the largest `attn.*` row, `attn.qkv` being larger now.

### The SwiGLU bf16 store: built, bit-exact, and not faster

The ledger row carried two bases for the same lever, 2.6 s per image if `ffn.w1w3`'s
profiled 24.5 TFLOP/s was the f32 store's price and about 1 s if the aggregate 1.19x
deflation held, and said the microbench would settle it. It did, and neither basis survived.
The arc built `kernel_mul_mm_bf16_f32_t_bf16out`, the tensor gemm with a bf16 rounding
epilogue (the cooperative tensor's own `store` static-asserts that the destination element
type equals the accumulator's, so a bfloat destination would mean a bfloat accumulator, and
the epilogue rounds per element instead), `silu_mul` over bf16 inputs,
`Projection::forward_bf16`, an `XWEN_ZIMAGE_FFN_STORE` arm and an `XWEN_ZIMAGE_FFN_STATS`
activation-max instrument. The store is bit-exact by construction, the f32 result rounded
once to nearest even, identical to `matmul_bf16(...).to_dtype(BF16)` at every shape tested.

Isolated (`zimage_ffn_store_only`), the chain being w1 x, w3 x, `silu_mul`, min over seven
interleaved reps; the GPU was under outside load in the middle runs:

| run | T | f32 chain | bf16 chain | f32 gemm | bf16 gemm | `silu_mul` f32 / bf16 |
| --- | --- | --- | --- | --- | --- | --- |
| scalar epilogue, first | 4128 | 15.11 ms | 14.28 | 8.76 | 9.76 | 1.00 / 0.69 |
| scalar epilogue, first | 1056 | 4.82 | 5.56 | 2.63 | 2.63 | 0.25 / 0.16 |
| threadgroup-staged bfloat4 store | 4128 | 15.47 | 16.95 | 13.14 | 14.47 | 1.07 / 0.76 |
| bfloat2 pair epilogue | 4128 | 15.61 | 16.54 | 14.56 | 15.30 | 1.03 / 0.79 |
| scalar epilogue, last, quiet GPU | 4128 | 15.41 | 15.28 | 7.25 | 7.79 | 1.01 / 0.72 |
| scalar epilogue, last, quiet GPU | 1056 | 3.83 | 4.06 | 2.27 | 2.20 | 0.25 / 0.16 |

`silu_mul_bf16` reliably saves 0.3 ms at 4128 and 0.09 at 1056, which is exactly its bytes at
470-550 GB/s. Every bf16 store form costs the gemm 0.5 to 1.3 ms at 4128 in most runs, more
than the activation saves, so the chain nets to parity within 5% with an unstable sign. The
bandwidth ceiling closes the question independently of any run: the store saves 338 MB per
block, 11.5 GB per step, about 25 ms per step at 450 GB/s and about 0.2 s per image, 1% of
the render. Whole-image A/B was indistinguishable inside a noise band where the f32 arm alone
spanned 2.2 to 3.4 s per step across identical runs.

The finding that matters beyond this row: **the f32-store gemm isolated runs at 37 to 45
TFLOPS, w2's own class, so the profiled 24.5 TFLOP/s was never the store.** It was the
profiler's buffer-pool eviction: every mark evicts the pool, so every w1/w3 dispatch
first-touches a fresh 169 MB buffer, and the profiled row reads 14.8 ms per gemm against
7.3-8.8 isolated. The same run shows unrelated rows moving 25% between arms, which is the
profiled table's own spread. The stretch, a half `h` for w2, is closed by measurement: max
|silu(w1 x) * (w3 x)| over an 8-step 1024x1024 run is 284,507, 64,695 already at step 1,
against f16's 65,504. Parity on the bf16 arm held the bars but cost 3.9 dB of image PSNR
(42.19 against 46.09) and doubled step-0 mean rel to 0.0018; the default arm read master's
numbers exactly. The default stays f32, the branch keeps the code, and master carries only
the decision (decisions.md "The bf16 SwiGLU store is REFUTED").

Not taken now: a converting store, reopen if a future MPP release adds one (the `_b16` store
intrinsics exist in the header behind the same-element-type static_assert) or documents the
tile lane layout so a vectorized epilogue can skip the index math; a per-row scaled
activation folded into w2's gemm, which is what a half intermediate would need and a
different arc; and a profiler mode that syncs without evicting, or reuses a pre-allocated
scratch, which would give a true per-row rate and was not built.

### Results on master e5d9775, clean

Nothing else on the GPU, `pmset -g` reading `lowpowermode 0`, dev-tree release build, seed 7,
"a lighthouse on a rocky shore at dusk, oil painting". The per-step times of three warm
1024x1024 runs, in order:

| run | steps 1 through 8 | VAE | wall (load) |
| --- | --- | --- | --- |
| 1 | 1.62 1.81 2.33 2.14 2.09 2.12 1.96 2.02 | 1.36 s | 25.2 s (5.0) |
| 2 | 1.47 1.56 1.65 1.71 1.80 1.93 2.04 3.27 | 1.78 s | 22.0 s (3.5) |
| 3, cold page cache | 1.35 1.36 1.61 1.77 1.94 1.91 2.06 2.12 | 1.44 s | load 44.7, excluded |

512x512: 0.35 0.37 0.38 0.39 0.39 0.39 0.40 0.39, VAE 0.26 s, 8.1 s wall with 3.7 s of load.

| | master e630ebb, clean | master e5d9775, clean |
| --- | --- | --- |
| 1024x1024 first step | 1.78 s | 1.35 s |
| 1024x1024 step by step 8 | 2.15 s | 2.0-2.1 s |
| 1024x1024 eight steps | 17.2 s at steady state | 14.1-16.1 s |
| VAE decode, 1024x1024 | 5.19 s | 1.36-1.44 s (one 1.78) |
| render, steps plus decode | ~21.5 s | 15.5-17.5 s |
| total wall, warm | ~25 s | 22-25 s |
| 512x512 step | 0.37-0.41 s | 0.35-0.40 s |
| VAE decode, 512x512 | 1.06-1.10 s | 0.26 s |
| 512x512 total wall | 8.6 s | 8.1 s |

**The step profile changed shape, and that is the finding to carry.** The first step fell
1.78 to 1.35 s, 24%, the isolated attention kernel is 3.5x, and the step still reaches
1.9-2.1 s by step 8, so the plateau moved by 0.1-0.2 s where the first step moved by 0.43.
The ramp is now 1.35 to 2.1, 55% over eight steps, where it was 1.78 to 2.15, 20%, and
before the rope arc 3.06 to 3.57, 17%. Every arc has made the first step faster and the ramp
steeper. That is the signature of a power or thermal cap: faster kernels reach the throttle
sooner, and past it the step is governed by the envelope and not by the kernel. The budget
says the same thing from the other side. At 1.35 s a step is about 46 TFLOP/s end to end; the
four gemms are 46.7 TFLOP, which at the 37 TFLOP/s the kernel measured in isolation on
2026-09-07 would already be 1.26 s, more than the whole step leaves after attention, so the
cool chip runs the gemm above any isolated figure taken on a warm one (the FFN arc's
isolated gemm spanned 37 to 45). This is an observation and not a confirmed cause. The
confirmation is a `powermetrics` reading during a run, which needs sudo from a user shell and
is the first thing the next arc does, because it decides whether any further kernel win
converts to wall time at steady state or only to a lower first step. Until then a step is
quoted as a range with its ramp, "1.35 s first step rising to 2.0-2.1 s by step 8", and
never as one steady number (decisions.md "A Z-Image step is quoted at steady state").

The profiled run on e5d9775, for ranking only (`/tmp/arc3-1024-prof.log`, transformer total
3120.88 ms per step against unprofiled steps of 1.35-2.1): `ffn.w1w3` 863, `attn.qkv` 392,
`ffn.w2` 272, `ffn.silu_mul` 254, `attn.sdpa` 231, `attn.norm+scale` 184, `attn.qknorm`
138, `ffn.norm+scale` 124, `attn.transpose` 121, `attn.rope` 119, `attn.out` 85,
`attn.untranspose` 58, the two gated residuals 26 each. `attn.sdpa` fell 698 to 231 and is
the fifth row now, below `ffn.silu_mul`. The VAE profiles at 1395 ms against 1.36-1.44 s
real, so its marks cost almost nothing now that the im2col buffers are gone: `up3.resnets`
324, `up2.resnets` 283, `up1.resnets` 207, `up2.upsample` 170, `up1.upsample` 147,
`mid.attn` 119.

Parity on the merged tree is the tensor arm's row above: step-0 cosine 0.999999, mean rel
0.0010, max rel 0.0050, final latent 0.999627 / 0.0077, image PSNR 45.60 dB, VAE alone
92.32 dB, both brackets outside. The lib suite is green at 1347 tests.

### Review fixes, c43a3e9

The outside review of a763c61 and e5d9775 (Codex, `/tmp/agent-report-review-arc3-codex.md`)
found one defect on the shipped path and three contract gaps in the exported wrappers, all
fixed in c43a3e9 and pushed. The decoder's entry conv (16 to 512 channels at 128x128) went
through the `Module` impl, which always called candle's conv2d, so its permuted weight plane
sat unused; the impl now takes the direct kernel where the arm resolved to it, and the
parity gate's VAE-alone line moved from 92.32 to 93.11 dB with `conv_in` at 4.3 ms profiled
where it read 12.6. The attention wrapper now requires q, k and v to be addressable in i32
as whole tensors (the kernel forms offsets in `int`, and the per-extent check alone let
about 16.7M query rows overflow), refuses a non-positive or non-finite scale (padded key
columns hold the lowest finite float and rely on the scale to keep them at zero weight; at
scale 0 they would each contribute 1 to the softmax sum), and refuses a q that does not
start on a 16-byte boundary, the Q tile being staged through float4 loads. The GroupNorm
kernels take their float4 path only when x is 16-byte aligned. No shipped caller changed
behaviour: every one hands over freshly allocated tensors and a positive scale. Classes the
review checked and found clean: conv tile, halo and partial-channel bounds; explicit-n
bounding; barriers; threadgroup memory sizing; the online softmax at a positive scale; the
encoder staying on candle; env parsing; the shared dispatch changes for the language-model
paths.

## Lever ledger

Rewritten 2026-09-08 for the third time, after the arc above. The base is master e5d9775:
a 1024x1024 step of **1.35 s first rising to 2.0-2.1 s by step 8**, eight steps in 14.1-16.1 s,
a 1.4 s decode, a render of **15.5-17.5 s** and a warm wall of 22-25 s. Two rows of the
previous ledger shipped and one is refuted, and the base itself is the open question: the
ramp says the steady state is governed by a power or thermal envelope, so every gain below is
a gain in the FIRST step until `powermetrics` says otherwise, and the instrument comes first.

| lever | today | ceiling, and how it was derived | gain per image | cost class |
| --- | --- | --- | --- | --- |
| the power envelope, an INSTRUMENT | the ramp: 1.35 to 2.0-2.1 s over eight steps, 55%, where it was 20% before this arc and 17% before the rope arc | a `powermetrics` trace of GPU power, frequency and the thermal state across a 1024x1024 run, read beside the per-step times | prices every row below: if the plateau is the envelope, a kernel win converts to wall time only until the cap and the render's floor is the cap's, not the kernels' | sudo from a user shell, an hour; the sandbox cannot run it |
| gemm launch fusion and tile tuning | the four gemms, 46.7 TFLOP, at 37-45 TFLOP/s isolated, so 1.04-1.26 s and most of the step | UNPRICED: q, k and v as one N=11520 gemm and the SwiGLU pair as one N=20480, then tiles tuned for M around 4000 at those N, which no sweep has covered | unknown, and `tests/zimage_microbench.rs` prices it in an hour at these shapes | host-side plus tile work, no new kernel class |
| the elementwise tail | ~1.05 s profiled per step over nine rows (norm+scale 184, `attn.qknorm` 138, `ffn.norm+scale` 124, transpose 121, rope 119, untranspose 58, `ffn.silu_mul` 254, the gates 52); real is the residual after the gemms and attention, at most ~0.1-0.3 s depending on where in the ramp the step sits | about half of that with gemm epilogues and a fused per-head `attn.qknorm` | under 1 s, probably well under; no single row is worth an arc | small kernels, no math change |
| the VAE mid-block attention | 108-119 ms profiled, 7-8% of the 1.4 s decode | candle's SDPA or a flash arm at head_dim 512 | ~0.1 s | one call site; reopen when the decode is wanted under 1 s |
| the SwiGLU bf16 store | REFUTED, see the arc above | the row's basis was the profiler's pool eviction; the gemm runs at 37-45 TFLOP/s with the f32 store, bandwidth caps the lever at ~0.2 s per image, and the built kernel measured parity within 5% with an unstable sign | none | on the branch `zimage-ffn` (5e7a6ea), not on master (decisions.md "The bf16 SwiGLU store is REFUTED") |

The dense bf16 floor of the previous ledger, a 1.25 s step, is where the FIRST step already
sits; there is no floor to quote for the plateau until the envelope is read.

Below the kernels, three record lines with reopen conditions, re-based on the new step.

- **Fewer denoising steps.** At 1.35-2.1 s a step, six instead of eight saves 3.5-4 s of the
  render and four saves 7-8 s, which is now the largest number on this page and the one that
  is not a measurement: Turbo is distilled for eight and the model card says nine
  ([zimage.md](../zimage.md)), so it is an image-quality judgement, and the stock ComfyUI
  node's `quality` field is the natural switch. Reopen on a product decision about how many
  steps an image gets.
- **Step caching**, TeaCache or a first-block cache. Worth 0 to 2 s and UNPRICED, the range
  starting at zero because an 8-step schedule at static shift 3.0 spaces its sigmas far
  apart, the regime these caches were not built for. Reopen on a priced experiment, a day.
- **int8 gemms on the tensor units**, about 1.8x fp16 compute on the gemm plane, so perhaps
  4-5 s of today's render if the envelope is not the cap and much less if it is. Weeks: W8A8
  calibration and a new set of parity bars. Reopen if sub-10 s at eight steps becomes a
  requirement (decisions.md "The transformer runs bf16 end to end").

Priced as non-levers, unchanged from the previous ledger: the per-step readback is free
(removing it read 35.8 s against 35.3 at the time, the work moving into the VAE number);
`context_refiner` is 13 ms a step and everything outside the block loops 18; the VAE in
bf16 is refuted (decisions.md "The VAE decodes in f32 and bf16 is refuted"), and the direct
conv path that replaced the dtype question is what shipped above.
