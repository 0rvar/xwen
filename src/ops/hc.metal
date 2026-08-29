// Vendored fused hyper-connection kernels — the qwen4exp residual carrier's
// read and write gates (src/qwen4exp/hc.rs, frozen oracle src/qwen4exp/ref_hc.rs).
//
// The carrier is `hc_count` parallel residual streams concatenated into one
// `hc_count * hidden` row. Every attention and every MLP block reads a single
// `hidden`-wide vector out of it and writes its output back into all streams,
// which on the candle chain costs ~20 dispatches per read and 3 per write —
// 4 gates per layer, 48 layers.
//
// Four kernels replace all of that except the two Q8_0 bottleneck matmuls,
// which stay on QLinear/QMatMul:
//   kernel_hc_norm          the grouped RMS norm (per-stream statistics, FULL
//                           width weight), with kernel_hc_norm_inject its
//                           sibling that also folds the injection head's
//                           hc_count full-row dot products and their
//                           2*sigmoid(./hc_count) into the same threadgroup —
//                           one templated body, the arm chosen by name at
//                           dispatch (the tail mixer has no injection head).
//                           kernel_hc_norm_split and kernel_hc_inject are the
//                           same two steps as separate launches over an
//                           n * hc_count grid, for batches too small to fill
//                           the machine one-threadgroup-per-token (bottom of
//                           this file; bit-identical, not merely equivalent).
//   kernel_hc_silu_quarter  silu(d / hc_count) on the bottleneck activation.
//   kernel_hc_mix           sigmoid(up_out) * normed, meaned over streams.
//   kernel_hc_write         stream + block_out (x) inject, out of place.
//
// ROUNDING. kernel_hc_silu_quarter and kernel_hc_write are BIT-IDENTICAL to the
// candle chains they replace and pin FP contraction and reassociation off at
// block scope:
//   - hc_silu_quarter reproduces candle's affine (`x * mul + 0`, where an fma
//     with a zero addend rounds exactly like the bare product) followed by
//     candle's usilu `x / (1 + exp(-x))`.
//   - hc_write reproduces the broadcast multiply `block_out[j] * inject[s]`
//     and the separate add onto the raw carrier, in that order.
// The other two are BOUNDED, not bitwise, because a reduction the reference
// runs in one order is partitioned across threads here: hc_norm's per-stream
// sum of squares and the injection head's `width`-long dot products fold
// through hardware simd_sum plus a threadgroup pass, and hc_mix's hc_count-term
// mean is a per-thread accumulation where candle runs a reduce kernel. Their
// tests grade at rel_l2 <= 1e-6 against the candle chain and <= 1e-5 against
// ref_hc, the same tolerance the classic path is held to.
//
// `#pragma METAL fp math_mode(fast)` pins the library's math MODE at the source
// level to what nil compile options resolve to today (candle compiles its own
// kernels with an explicit MTLMathMode::Fast), exactly as delta.metal does.
//
// A SEPARATE library from the other vendored sources (own runtime compile via
// src/ops/pipelines.rs, no Metal-4 dependency).

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp math_mode(fast)

// The largest hc_count the injection accumulators are sized for. The host
// (dispatch.rs HC_MAX_STREAMS) refuses anything wider, so the per-thread array
// below is always a compile-time-bounded, register-resident 8.
#define HC_MAX_STREAMS 8

// Matches dispatch.rs HcNormArgs (#[repr(C)]).
typedef struct {
    int32_t hc_count;
    int32_t hidden;
    int32_t width; // hc_count * hidden
    float eps;
    float inv_hc; // 1 / hc_count, as candle's affine multiplier
} hc_norm_args;

// Grouped RMS norm over one carrier row, optionally plus the injection head.
//
// Statistics are per stream — the streams sit decades apart in scale and one
// set of statistics over the whole row would flatten that — while the weight is
// FULL width, one value per element of the carrier (docs/qwen4exp-port.md trap
// #16). Accumulation is f32; ref_hc uses f64, which is the oracle's insurance
// and not a requirement of the math.
//
// One threadgroup per token, `tcount` threads striding each stream's slice, so
// every thread's elements belong to exactly one stream at a time and the
// hc_count reductions stay disjoint. `tcount` divides `hidden` (the host picks
// it) and is a multiple of the simd width.
//
// With HAS_INJECT the same threadgroup then contracts the normed row against
// the injection head's hc_count rows — accumulated per thread while the normed
// values are still in registers, so the head costs one extra read of its own
// [hc_count, width] weight and no extra pass over the carrier.
//
// `partial` and `scales` are passed in rather than declared here: MSL allows
// threadgroup variables only at a kernel function's own scope.
template <bool HAS_INJECT>
static inline void hc_norm_body(
        constant hc_norm_args & args,
        device const float * x,
        device const float * w,
        device const float * inj,
        device       float * normed,
        device       float * inject,
        threadgroup  float * partial, // [32], one slot per simdgroup
        threadgroup  float * scales,  // [HC_MAX_STREAMS]
        uint tgid,
        uint tid,
        uint sgid,
        uint lane,
        uint sgcount,
        uint tcount) {
    const size_t row = (size_t) tgid * (size_t) args.width;

    // Per-stream sum of squares -> the stream's 1/sqrt(mean + eps).
    for (int s = 0; s < args.hc_count; ++s) {
        const size_t base = row + (size_t) (s * args.hidden);
        float acc = 0.0f;
        for (int j = (int) tid; j < args.hidden; j += (int) tcount) {
            const float v = x[base + (size_t) j];
            acc += v * v;
        }
        const float lane_sum = simd_sum(acc);
        if (lane == 0) {
            partial[sgid] = lane_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            for (uint u = 0; u < sgcount; ++u) {
                total += partial[u];
            }
            scales[s] = 1.0f / sqrt(total / (float) args.hidden + args.eps);
        }
        // Also the write-after-read fence that lets the next stream reuse
        // `partial`.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize, weight, write — and contract against the injection head on the
    // way past. The accumulator loop is bounded by the compile-time
    // HC_MAX_STREAMS so it unrolls and every index stays constant: a runtime
    // index would push the array out of registers and into thread scratch.
    float acc_inj[HC_MAX_STREAMS];
    if (HAS_INJECT) {
        for (int o = 0; o < HC_MAX_STREAMS; ++o) {
            acc_inj[o] = 0.0f;
        }
    }
    for (int s = 0; s < args.hc_count; ++s) {
        const float scale = scales[s];
        const int off = s * args.hidden;
        for (int j = (int) tid; j < args.hidden; j += (int) tcount) {
            const int i = off + j;
            const float n = x[row + (size_t) i] * scale * w[i];
            normed[row + (size_t) i] = n;
            if (HAS_INJECT) {
                for (int o = 0; o < HC_MAX_STREAMS; ++o) {
                    if (o < args.hc_count) {
                        acc_inj[o] += inj[(size_t) (o * args.width + i)] * n;
                    }
                }
            }
        }
    }

    if (!HAS_INJECT) {
        return;
    }
    // 2*sigmoid(dot / hc_count) per stream: spans (0, 2), centered on 1, so an
    // untrained gate leaves the carrier's scale alone.
    for (int o = 0; o < args.hc_count; ++o) {
        float mine = 0.0f;
        for (int k = 0; k < HC_MAX_STREAMS; ++k) {
            if (k == o) {
                mine = acc_inj[k];
            }
        }
        const float lane_sum = simd_sum(mine);
        if (lane == 0) {
            partial[sgid] = lane_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            for (uint u = 0; u < sgcount; ++u) {
                total += partial[u];
            }
            const float z = total * args.inv_hc;
            inject[(size_t) tgid * (size_t) args.hc_count + (size_t) o] =
                2.0f * (1.0f / (1.0f + exp(-z)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kernel_hc_norm(
        constant hc_norm_args & args [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device const float * w       [[buffer(2)]],
        device const float * inj     [[buffer(3)]],
        device       float * normed  [[buffer(4)]],
        device       float * inject  [[buffer(5)]],
        uint tgid   [[threadgroup_position_in_grid]],
        uint tid    [[thread_position_in_threadgroup]],
        uint sgid   [[simdgroup_index_in_threadgroup]],
        uint lane   [[thread_index_in_simdgroup]],
        uint sgcount [[simdgroups_per_threadgroup]],
        uint tcount [[threads_per_threadgroup]]) {
    threadgroup float partial[32];
    threadgroup float scales[HC_MAX_STREAMS];
    hc_norm_body<false>(args, x, w, inj, normed, inject, partial, scales, tgid, tid, sgid, lane,
                        sgcount, tcount);
}

// The injection-head sibling (a block gate rather than the tail mixer).
// Identical but for the extra contraction and its [n, hc_count] output.
kernel void kernel_hc_norm_inject(
        constant hc_norm_args & args [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device const float * w       [[buffer(2)]],
        device const float * inj     [[buffer(3)]],
        device       float * normed  [[buffer(4)]],
        device       float * inject  [[buffer(5)]],
        uint tgid   [[threadgroup_position_in_grid]],
        uint tid    [[thread_position_in_threadgroup]],
        uint sgid   [[simdgroup_index_in_threadgroup]],
        uint lane   [[thread_index_in_simdgroup]],
        uint sgcount [[simdgroups_per_threadgroup]],
        uint tcount [[threads_per_threadgroup]]) {
    threadgroup float partial[32];
    threadgroup float scales[HC_MAX_STREAMS];
    hc_norm_body<true>(args, x, w, inj, normed, inject, partial, scales, tgid, tid, sgid, lane,
                       sgcount, tcount);
}

// Matches dispatch.rs HcSiluArgs (#[repr(C)]).
typedef struct {
    int32_t n;
    float scale; // 1 / hc_count
} hc_silu_args;

// The bottleneck activation, `silu(d / hc_count)`. The 1/hc_count scale is on
// the ACTIVATION, before the silu: the carrier is a sum over hc_count streams,
// so the down projection's output grows with the stream count.
//
// Bit-identical to candle's `affine(1/hc_count, 0)` + `silu` pair: the affine's
// fma against a zero addend rounds exactly like the bare product, and the silu
// is candle's usilu written out.
kernel void kernel_hc_silu_quarter(
        constant hc_silu_args & args [[buffer(0)]],
        device const float * src     [[buffer(1)]],
        device       float * dst     [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    if ((int) tid >= args.n) {
        return;
    }
    const float x = src[tid] * args.scale;
    dst[tid] = x / (1.0f + exp(-x));
}

// Matches dispatch.rs HcMixArgs (#[repr(C)]).
typedef struct {
    int32_t n;   // rows * hidden, the output element count
    int32_t hc_count;
    int32_t hidden;
    float inv_hc; // 1 / hc_count
} hc_mix_args;

// The mix and the collapse: `mixed[j] = mean_s sigmoid(up_out[s*hidden+j]) *
// normed[s*hidden+j]`. MEAN, not sum. `up` arrives as the up projection's raw
// pre-sigmoid logits, so the sigmoid is folded in here rather than costing its
// own full-width pass.
//
// One thread per OUTPUT element, so each thread walks the hc_count strided
// stream slots of its own column.
kernel void kernel_hc_mix(
        constant hc_mix_args & args [[buffer(0)]],
        device const float * up     [[buffer(1)]],
        device const float * normed [[buffer(2)]],
        device       float * dst    [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if ((int) tid >= args.n) {
        return;
    }
    const int j = (int) tid % args.hidden;
    const int t = (int) tid / args.hidden;
    const size_t row = (size_t) t * (size_t) (args.hc_count * args.hidden);
    float acc = 0.0f;
    for (int s = 0; s < args.hc_count; ++s) {
        const size_t i = row + (size_t) (s * args.hidden + j);
        const float u = up[i];
        acc += (1.0f / (1.0f + exp(-u))) * normed[i];
    }
    dst[tid] = acc * args.inv_hc;
}

// Matches dispatch.rs HcWriteArgs (#[repr(C)]).
typedef struct {
    int32_t n; // rows * hc_count * hidden, the output element count
    int32_t hc_count;
    int32_t hidden;
} hc_write_args;

// The write-back, onto the RAW carrier: `new[s*hidden+j] = x[s*hidden+j] +
// block_out[j] * inject[s]`. Out of place — the caller's carrier tensor is
// never mutated, so a tap or a snapshot holding the old one stays valid.
//
// Bit-identical to the candle chain (broadcast multiply, then add), which is
// why the fp pragmas pin contraction off: an fma here would fold the two
// roundings into one.
kernel void kernel_hc_write(
        constant hc_write_args & args [[buffer(0)]],
        device const float * x         [[buffer(1)]],
        device const float * block_out [[buffer(2)]],
        device const float * inject    [[buffer(3)]],
        device       float * dst       [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    if ((int) tid >= args.n) {
        return;
    }
    const int width = args.hc_count * args.hidden;
    const int col = (int) tid % width;
    const int t = (int) tid / width;
    const int s = col / args.hidden;
    const int j = col - s * args.hidden;
    const float scaled =
        block_out[(size_t) t * (size_t) args.hidden + (size_t) j] *
        inject[(size_t) t * (size_t) args.hc_count + (size_t) s];
    dst[tid] = x[tid] + scaled;
}

// ---------------------------------------------------------------------------
// The small-batch split path.
//
// kernel_hc_norm[_inject] puts ONE threadgroup on a whole token: at decode
// (n = 1) that is 97 launches per forward each running a two-pass read of a
// 10240-wide carrier — plus, on the gated arm, the whole [hc_count, width]
// injection head — on a single 256-thread threadgroup, with the rest of the GPU
// idle. It is the right shape at prefill, where the token grid fills the machine
// on its own, and the wrong one below a handful of tokens (dispatch.rs
// HC_SPLIT_MAX_N picks between them).
//
// The split pair spreads the same work over an `n * hc_count` grid: one
// threadgroup per (token, stream) for the norm, one per (token, injection row)
// for the head. Both keep the SINGLE-threadgroup kernel's thread count and its
// per-thread strided partition, so every reduction folds in the same
// association order and both outputs are BIT-IDENTICAL to the fused kernel's
// (`split_norm_matches_single_bitwise` pins that; the cost of getting it is
// nothing but writing the loops in the same order).
// ---------------------------------------------------------------------------

// One threadgroup per (token, stream): the stream's own sum of squares, then
// its `hidden`-wide slice of the normed carrier. The weight is still indexed at
// FULL width (`s * hidden + j`) — it is one value per carrier element, not per
// stream element.
kernel void kernel_hc_norm_split(
        constant hc_norm_args & args [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device const float * w       [[buffer(2)]],
        device       float * normed  [[buffer(3)]],
        uint2 tgid  [[threadgroup_position_in_grid]],
        uint2 tpos  [[thread_position_in_threadgroup]],
        uint sgid   [[simdgroup_index_in_threadgroup]],
        uint lane   [[thread_index_in_simdgroup]],
        uint sgcount [[simdgroups_per_threadgroup]],
        uint2 tdim  [[threads_per_threadgroup]]) {
    // The launch is `threads x 1`, so the row component is the whole partition.
    const uint tid = tpos.x;
    const uint tcount = tdim.x;
    threadgroup float partial[32];
    // A one-slot array rather than a scalar: the barrier below is what makes
    // the write visible, which the compiler's uninitialized-use warning cannot
    // see through on a plain threadgroup scalar.
    threadgroup float scale_tg[1];

    const int s = (int) tgid.y;
    const size_t row = (size_t) tgid.x * (size_t) args.width;
    const int off = s * args.hidden;
    const size_t base = row + (size_t) off;

    float acc = 0.0f;
    for (int j = (int) tid; j < args.hidden; j += (int) tcount) {
        const float v = x[base + (size_t) j];
        acc += v * v;
    }
    const float lane_sum = simd_sum(acc);
    if (lane == 0) {
        partial[sgid] = lane_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint u = 0; u < sgcount; ++u) {
            total += partial[u];
        }
        scale_tg[0] = 1.0f / sqrt(total / (float) args.hidden + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = scale_tg[0];
    for (int j = (int) tid; j < args.hidden; j += (int) tcount) {
        const int i = off + j;
        normed[row + (size_t) i] = x[row + (size_t) i] * scale * w[i];
    }
}

// One threadgroup per (token, injection row): `2*sigmoid(dot(I[o], normed[t]) /
// hc_count)`, the same gate kernel_hc_norm_inject folds into its second pass.
//
// The dot walks the carrier STREAM-MAJOR with the same `tcount` stride the
// fused kernel's accumulators saw — outer loop over streams, inner strided walk
// inside each — so each thread's partial sum, and therefore the simd_sum and
// the serial fold over `partial`, land on exactly the fused kernel's bits.
// Reading `normed` back from device memory rather than recomputing it costs one
// extra full-width read and is what lets the norm above stay a separate launch.
kernel void kernel_hc_inject(
        constant hc_norm_args & args [[buffer(0)]],
        device const float * inj     [[buffer(1)]],
        device const float * normed  [[buffer(2)]],
        device       float * inject  [[buffer(3)]],
        uint2 tgid  [[threadgroup_position_in_grid]],
        uint2 tpos  [[thread_position_in_threadgroup]],
        uint sgid   [[simdgroup_index_in_threadgroup]],
        uint lane   [[thread_index_in_simdgroup]],
        uint sgcount [[simdgroups_per_threadgroup]],
        uint2 tdim  [[threads_per_threadgroup]]) {
    // The launch is `threads x 1`, so the row component is the whole partition.
    const uint tid = tpos.x;
    const uint tcount = tdim.x;
    threadgroup float partial[32];

    const int o = (int) tgid.y;
    const size_t row = (size_t) tgid.x * (size_t) args.width;
    const size_t inj_row = (size_t) o * (size_t) args.width;

    float acc = 0.0f;
    for (int s = 0; s < args.hc_count; ++s) {
        const int off = s * args.hidden;
        for (int j = (int) tid; j < args.hidden; j += (int) tcount) {
            const int i = off + j;
            acc += inj[inj_row + (size_t) i] * normed[row + (size_t) i];
        }
    }
    const float lane_sum = simd_sum(acc);
    if (lane == 0) {
        partial[sgid] = lane_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint u = 0; u < sgcount; ++u) {
            total += partial[u];
        }
        const float z = total * args.inv_hc;
        inject[(size_t) tgid.x * (size_t) args.hc_count + (size_t) o] =
            2.0f * (1.0f / (1.0f + exp(-z)));
    }
}
