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
// which stay on QLinear/QMatMul (a fifth and sixth, at the bottom of this file,
// swallow those two as well for the small token counts a decode step runs —
// kernel_hc_gate_down and kernel_hc_gate_up_mix, three dispatches per gate
// instead of seven):
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
        const size_t base = row + (size_t) s * (size_t) args.hidden;
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
    // Every operand of an index product is widened BEFORE the multiply, here
    // and in the split pair. `o * args.width` is a row offset into the
    // injection head and the carrier row stride is `args.width`, both of which
    // scale with the geometry, and a product computed in `int` and cast
    // afterwards would already have wrapped by the time the cast sees it.
    for (int s = 0; s < args.hc_count; ++s) {
        const float scale = scales[s];
        const size_t off = (size_t) s * (size_t) args.hidden;
        for (int j = (int) tid; j < args.hidden; j += (int) tcount) {
            const size_t i = off + (size_t) j;
            const float n = x[row + i] * scale * w[i];
            normed[row + i] = n;
            if (HAS_INJECT) {
                for (int o = 0; o < HC_MAX_STREAMS; ++o) {
                    if (o < args.hc_count) {
                        acc_inj[o] += inj[(size_t) o * (size_t) args.width + i] * n;
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
    const size_t row = (size_t) t * (size_t) args.hc_count * (size_t) args.hidden;
    float acc = 0.0f;
    for (int s = 0; s < args.hc_count; ++s) {
        const size_t i = row + (size_t) s * (size_t) args.hidden + (size_t) j;
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
// (`split_matches_single_bitwise` pins that; the cost of getting it is
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
    const size_t off = (size_t) s * (size_t) args.hidden;
    const size_t base = row + off;

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
        const size_t i = off + (size_t) j;
        normed[row + i] = x[row + i] * scale * w[i];
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
        const size_t off = (size_t) s * (size_t) args.hidden;
        for (int j = (int) tid; j < args.hidden; j += (int) tcount) {
            const size_t i = off + (size_t) j;
            acc += inj[inj_row + i] * normed[row + i];
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

// ---------------------------------------------------------------------------
// The fused decode gate.
//
// The seven dispatches above (norm, head, down gemv, activation, up gemv, mix,
// write) are seven launch latencies per gate, and a decode step runs one gate
// per attention block and one per MLP block. Below a handful of tokens that
// latency, not bandwidth, is what the gate costs: the two Q8_0 projections are
// already near the machine's streaming rate, while the five glue kernels around
// them move a few tens of kilobytes each.
//
// These two kernels swallow the projections and their glue whole:
//   kernel_hc_gate_down     norm + injection head + down gemv + silu
//   kernel_hc_gate_up_mix   up gemv + sigmoid + mix and collapse
// leaving three dispatches per gate with kernel_hc_write, and two on the
// headless tail mixer. The normed carrier is never materialized: the first
// kernel keeps it in registers for the head and the down rows and hands the
// second kernel the per-stream `scales` instead, which recomputes
// `x * scale * w` for the one element each of its threads needs.
//
// ROUNDING: BOUNDED, not bitwise, against the chain they replace. Both dot
// products are reassociated — the down rows fold per-thread partials through
// simd_sum where the chain's gemv folds its own partition, and the mix's
// hc_count-term sum runs as a simd_shuffle_xor butterfly where kernel_hc_mix
// runs a serial loop. The per-stream statistics are partitioned differently
// again (per q8_0 block rather than per strided slice). Graded at rel_l2 <= 1e-5
// against both the split chain and ref_hc, the tolerance the classic path is
// held to; XWEN_HC_GATE_CLASSIC restores the seven-dispatch path.
//
// LAYOUT, and why it is not one threadgroup per token. `kernel_hc_norm_inject`
// already proved the one-threadgroup-per-token shape is a 6% decode LOSS here
// (see the split section above), so both kernels keep a wide grid: the first
// launches one threadgroup per (row tile of the down weight, token) plus one
// for the head, the second one per (column tile of the carrier, token). Each
// weight byte is still read exactly once per token; what the first kernel
// re-reads across its threadgroups is the carrier row and the norm weight,
// 40 KiB each at the production geometry, which is a cache working set rather
// than a bandwidth cost.
//
// The k partition is the same in both: `HC_GATE_THREADS` threads own
// `nblk / HC_GATE_THREADS` q8_0 blocks each, interleaved (thread t takes blocks
// t, t + HC_GATE_THREADS, ...), so adjacent lanes read adjacent 34-byte blocks
// and adjacent 128-byte runs of the carrier. The host refuses any geometry that
// does not divide (dispatch.rs `hc_gate_fused_supported`) and keeps the seven
// -dispatch path for it.
// ---------------------------------------------------------------------------

// block_q8_0 (ggml-common.h): one f16 delta then 32 int8 quants, 34 bytes. Both
// bottleneck projections ship q8_0, and this library reads them directly rather
// than through QMatMul. Declared here rather than shared with q8.metal: each
// vendored .metal is its own runtime-compiled library.
#define QK8_0 32
typedef struct {
    half   d;
    int8_t qs[QK8_0];
} hc_block_q8_0;

// Threads per threadgroup in kernel_hc_gate_down: five simdgroups, and a
// divisor of the 320 q8_0 blocks a production carrier row holds. Mirrored by
// dispatch.rs HC_GATE_THREADS.
#define HC_GATE_THREADS 160
#define HC_GATE_SIMDGROUPS (HC_GATE_THREADS / 32)

// q8_0 blocks of the carrier one thread of kernel_hc_gate_down owns, at most.
// It stages them in registers (32 floats each) and every step below reads them
// from there, so this bounds the kernel's register footprint; the host refuses a
// carrier wider than HC_GATE_THREADS * HC_GATE_MAX_BLK_PER_THREAD blocks. The
// production carrier is 320 blocks = 2 per thread. Mirrored by dispatch.rs
// HC_GATE_MAX_BLK_PER_THREAD.
#define HC_GATE_MAX_BLK_PER_THREAD 2

// Down-projection output rows one threadgroup of kernel_hc_gate_down computes.
// Each is a full-width dot against the same staged carrier, so the rows share
// one pass over it and cost only their own weight bytes; the accumulators are
// registers, which is what bounds this. Mirrored by dispatch.rs
// HC_GATE_ROWS_PER_TG.
#define HC_GATE_ROWS_PER_TG 8

// Threads per threadgroup in kernel_hc_gate_up_mix: hc_count adjacent lanes take
// the hc_count streams of one carrier column, so a threadgroup covers
// HC_GATE_MIX_THREADS / hc_count columns. Mirrored by dispatch.rs
// HC_GATE_MIX_THREADS.
#define HC_GATE_MIX_THREADS 256

// The widest bottleneck kernel_hc_gate_up_mix stages in threadgroup memory (one
// f32 per low_rank column, read by every thread of the threadgroup). Mirrored by
// dispatch.rs HC_GATE_MAX_LOW_RANK.
#define HC_GATE_MAX_LOW_RANK 1024

// Matches dispatch.rs HcGateDownArgs (#[repr(C)]).
typedef struct {
    int32_t hc_count;
    int32_t hidden;
    int32_t width;     // hc_count * hidden
    int32_t low_rank;
    int32_t nblk;      // width / 32, the q8_0 blocks in one carrier row
    int32_t n_down_tg; // ceil(low_rank / HC_GATE_ROWS_PER_TG)
    float eps;
    float inv_hc;      // 1 / hc_count
} hc_gate_down_args;

// Grouped RMS norm, injection head, down projection and bottleneck activation
// for one token, in one launch.
//
// Grid is (n_down_tg + has_inject, n): threadgroup `g < n_down_tg` computes
// output rows `g * HC_GATE_ROWS_PER_TG ..` of the down projection, and the last
// one — present only on a gated block, never on the tail mixer — computes the
// injection head's hc_count full-row dot products. Every threadgroup repeats the
// norm for its token, which is a read of the carrier row and the norm weight
// plus hc_count simd reductions; that is cheaper than a second launch and a
// materialized `normed`.
//
// Outputs: `low[t, low_rank]` = silu(down . normed / hc_count), `inject[t,
// hc_count]` = 2*sigmoid(head . normed / hc_count) (untouched without a head),
// and `scales[t, hc_count]`, the per-stream 1/sqrt(mean+eps) that
// kernel_hc_gate_up_mix needs to rebuild `normed` one element at a time.
kernel void kernel_hc_gate_down(
        constant hc_gate_down_args & args    [[buffer(0)]],
        device const float * x               [[buffer(1)]],
        device const float * w               [[buffer(2)]],
        device const float * inj             [[buffer(3)]],
        device const hc_block_q8_0 * down    [[buffer(4)]],
        device       float * low             [[buffer(5)]],
        device       float * inject          [[buffer(6)]],
        device       float * scales_out      [[buffer(7)]],
        uint2 tgid   [[threadgroup_position_in_grid]],
        uint2 tpos   [[thread_position_in_threadgroup]],
        uint  sgid   [[simdgroup_index_in_threadgroup]],
        uint  lane   [[thread_index_in_simdgroup]],
        uint  sgcount [[simdgroups_per_threadgroup]]) {
    // The launch is `HC_GATE_THREADS x 1`, so the row component is the whole
    // partition (the vector arity has to match tgid's).
    const uint tid = tpos.x;
    // One slot per (row of this tile, simdgroup); the norm and the head reduce
    // one value at a time and use the first HC_GATE_SIMDGROUPS of it.
    threadgroup float partial[HC_GATE_ROWS_PER_TG * HC_GATE_SIMDGROUPS];
    threadgroup float scales[HC_MAX_STREAMS];

    const int nbpt = args.nblk / HC_GATE_THREADS;
    const size_t row = (size_t) tgid.y * (size_t) args.width;
    // `hidden` is a whole number of q8_0 blocks (the host refuses otherwise), so
    // every block belongs to exactly one stream and the hc_count sum-of-squares
    // reductions stay disjoint.
    const int blk_per_stream = args.hidden / QK8_0;

    // This thread's slice of the carrier, staged once. The loop bound is the
    // compile-time maximum with a runtime guard inside, so every index into the
    // array is a constant after unrolling and the array stays in registers.
    float xv[HC_GATE_MAX_BLK_PER_THREAD * QK8_0];
    for (int p = 0; p < HC_GATE_MAX_BLK_PER_THREAD; ++p) {
        if (p < nbpt) {
            const size_t base = row + (size_t) ((int) tid + p * HC_GATE_THREADS) * QK8_0;
            for (int i = 0; i < QK8_0; ++i) {
                xv[p * QK8_0 + i] = x[base + (size_t) i];
            }
        }
    }

    // Per-stream sum of squares -> the stream's 1/sqrt(mean + eps). Statistics
    // are per stream — the streams sit decades apart in scale and one set over
    // the whole row would flatten that (docs/qwen4exp-port.md trap #16).
    for (int s = 0; s < args.hc_count; ++s) {
        float acc = 0.0f;
        for (int p = 0; p < HC_GATE_MAX_BLK_PER_THREAD; ++p) {
            if (p < nbpt && ((int) tid + p * HC_GATE_THREADS) / blk_per_stream == s) {
                for (int i = 0; i < QK8_0; ++i) {
                    const float v = xv[p * QK8_0 + i];
                    acc += v * v;
                }
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
            scales[s] = 1.0f / sqrt(total / (float) args.hidden + args.eps);
        }
        // Also the write-after-read fence that lets the next stream reuse
        // `partial`.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Every threadgroup recomputes the scales; one of them publishes them.
    if (tgid.x == 0 && tid < (uint) args.hc_count) {
        scales_out[(size_t) tgid.y * (size_t) args.hc_count + (size_t) tid] = scales[tid];
    }

    // Normalize in place, in the same left-to-right order as
    // kernel_hc_norm_split. The weight is indexed at FULL width — one value per
    // carrier element, not per stream element.
    for (int p = 0; p < HC_GATE_MAX_BLK_PER_THREAD; ++p) {
        if (p < nbpt) {
            const int blk = (int) tid + p * HC_GATE_THREADS;
            const float scale = scales[blk / blk_per_stream];
            const size_t off = (size_t) blk * QK8_0;
            for (int i = 0; i < QK8_0; ++i) {
                xv[p * QK8_0 + i] = xv[p * QK8_0 + i] * scale * w[off + (size_t) i];
            }
        }
    }

    if ((int) tgid.x < args.n_down_tg) {
        // HC_GATE_ROWS_PER_TG rows of the down projection against the staged
        // normed slice. Per q8_0 block: an f32 dot of the 32 quants, then one
        // multiply by the block's delta — the same form as
        // kernel_mul_mv_q8_0_f32_attn (q8.metal).
        const int row0 = (int) tgid.x * HC_GATE_ROWS_PER_TG;
        float acc[HC_GATE_ROWS_PER_TG];
        for (short r = 0; r < HC_GATE_ROWS_PER_TG; ++r) {
            acc[r] = 0.0f;
        }
        for (short r = 0; r < HC_GATE_ROWS_PER_TG; ++r) {
            // Uniform across the threadgroup: a ragged last tile skips the same
            // rows in every thread, so nothing below diverges.
            if (row0 + (int) r >= args.low_rank) {
                continue;
            }
            device const hc_block_q8_0 * arow =
                down + (size_t) (row0 + (int) r) * (size_t) args.nblk;
            for (int p = 0; p < HC_GATE_MAX_BLK_PER_THREAD; ++p) {
                if (p < nbpt) {
                    device const hc_block_q8_0 * b = arow + ((int) tid + p * HC_GATE_THREADS);
                    // packed_char4, not char4: the quants start two bytes into
                    // a 34-byte block, so a naturally aligned vector load would
                    // fault.
                    device const packed_char4 * q4 = (device const packed_char4 *) b->qs;
                    float sumq = 0.0f;
                    for (short i = 0; i < QK8_0 / 4; ++i) {
                        const packed_char4 v = q4[i];
                        sumq += (float) v.x * xv[p * QK8_0 + 4 * i + 0];
                        sumq += (float) v.y * xv[p * QK8_0 + 4 * i + 1];
                        sumq += (float) v.z * xv[p * QK8_0 + 4 * i + 2];
                        sumq += (float) v.w * xv[p * QK8_0 + 4 * i + 3];
                    }
                    acc[r] += sumq * (float) b->d;
                }
            }
        }
        // One reduction pass for all HC_GATE_ROWS_PER_TG rows: the per-row
        // simd_sums need no barrier between them, only the single fence before
        // the serial fold.
        for (short r = 0; r < HC_GATE_ROWS_PER_TG; ++r) {
            const float lane_sum = simd_sum(acc[r]);
            if (lane == 0) {
                partial[(uint) r * HC_GATE_SIMDGROUPS + sgid] = lane_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < HC_GATE_ROWS_PER_TG && row0 + (int) tid < args.low_rank) {
            float total = 0.0f;
            for (uint u = 0; u < sgcount; ++u) {
                total += partial[tid * HC_GATE_SIMDGROUPS + u];
            }
            // The 1/hc_count scale is on the ACTIVATION, before the silu: the
            // carrier is a sum over hc_count streams, so the projection's output
            // grows with the stream count.
            const float z = total * args.inv_hc;
            low[(size_t) tgid.y * (size_t) args.low_rank + (size_t) (row0 + (int) tid)] =
                z / (1.0f + exp(-z));
        }
        return;
    }

    // The injection head. Accumulated per thread while the normed values are
    // still in registers; the accumulator loop is bounded by the compile-time
    // HC_MAX_STREAMS so it unrolls and every index stays constant.
    float acc_inj[HC_MAX_STREAMS];
    for (int o = 0; o < HC_MAX_STREAMS; ++o) {
        acc_inj[o] = 0.0f;
    }
    for (int o = 0; o < HC_MAX_STREAMS; ++o) {
        if (o < args.hc_count) {
            for (int p = 0; p < HC_GATE_MAX_BLK_PER_THREAD; ++p) {
                if (p < nbpt) {
                    // Every operand of an index product is widened BEFORE the
                    // multiply: `o * width` scales with the geometry and would
                    // already have wrapped by the time a cast saw it.
                    const size_t base = (size_t) o * (size_t) args.width +
                                        (size_t) ((int) tid + p * HC_GATE_THREADS) * QK8_0;
                    for (int i = 0; i < QK8_0; ++i) {
                        acc_inj[o] += inj[base + (size_t) i] * xv[p * QK8_0 + i];
                    }
                }
            }
        }
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
            inject[(size_t) tgid.y * (size_t) args.hc_count + (size_t) o] =
                2.0f * (1.0f / (1.0f + exp(-z)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Matches dispatch.rs HcGateUpArgs (#[repr(C)]).
typedef struct {
    int32_t hc_count;
    int32_t hidden;
    int32_t width;    // hc_count * hidden
    int32_t low_rank;
    int32_t nblk_low; // low_rank / 32, the q8_0 blocks in one up row
    float inv_hc;     // 1 / hc_count
} hc_gate_up_args;

// Up projection, sigmoid, and the mix and collapse for one token, in one launch:
// `mixed[j] = mean_s sigmoid(up[s*hidden+j] . low) * normed[s*hidden+j]`. MEAN,
// not sum. The up projection's raw logits are never materialized — the sigmoid
// is applied where the dot finishes.
//
// One thread per (column j, stream s), the hc_count streams of a column in
// ADJACENT lanes, so the hc_count-term mean folds through simd_shuffle_xor with
// no threadgroup memory and every thread's up row is its own. The bottleneck
// activation is staged once per threadgroup and read by all of them.
//
// `normed[s*hidden+j]` is rebuilt from the raw carrier and the scales
// kernel_hc_gate_down published, in that kernel's expression order — one f32
// each rather than a materialized full-width intermediate.
kernel void kernel_hc_gate_up_mix(
        constant hc_gate_up_args & args   [[buffer(0)]],
        device const float * low          [[buffer(1)]],
        device const hc_block_q8_0 * up   [[buffer(2)]],
        device const float * x            [[buffer(3)]],
        device const float * w            [[buffer(4)]],
        device const float * scales       [[buffer(5)]],
        device       float * mixed        [[buffer(6)]],
        uint2 tgid [[threadgroup_position_in_grid]],
        uint  sgid [[simdgroup_index_in_threadgroup]],
        uint  lane [[thread_index_in_simdgroup]]) {
    // Built from (simdgroup, lane) rather than from thread_position_in_threadgroup
    // because the fold below is over LANES: a column's hc_count streams have to
    // sit in the low bits of `lane` for simd_shuffle_xor to reach them, which
    // this makes structural rather than an assumption about the 1-D mapping.
    // hc_count divides the simd width (the host refuses otherwise), so no
    // column straddles two simdgroups.
    const uint tid = sgid * 32u + lane;

    threadgroup float lo[HC_GATE_MAX_LOW_RANK];
    for (uint i = tid; i < (uint) args.low_rank; i += HC_GATE_MIX_THREADS) {
        lo[i] = low[(size_t) tgid.y * (size_t) args.low_rank + (size_t) i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int s = (int) tid % args.hc_count;
    const int j = (int) tgid.x * (HC_GATE_MIX_THREADS / args.hc_count) +
                  (int) tid / args.hc_count;

    // A ragged last tile leaves some threads with no column. They still take the
    // butterfly below — a simd shuffle needs every lane it reads from — and
    // contribute the zero this thread's term would have been.
    float acc = 0.0f;
    if (j < args.hidden) {
        const size_t urow = (size_t) s * (size_t) args.hidden + (size_t) j;
        device const hc_block_q8_0 * blk = up + urow * (size_t) args.nblk_low;
        for (int b = 0; b < args.nblk_low; ++b) {
            // packed_char4, not char4: the quants start two bytes into a
            // 34-byte block, so a naturally aligned vector load would fault.
            device const packed_char4 * q4 = (device const packed_char4 *) blk[b].qs;
            threadgroup const float * lb = lo + b * QK8_0;
            float sumq = 0.0f;
            for (short i = 0; i < QK8_0 / 4; ++i) {
                const packed_char4 v = q4[i];
                sumq += (float) v.x * lb[4 * i + 0];
                sumq += (float) v.y * lb[4 * i + 1];
                sumq += (float) v.z * lb[4 * i + 2];
                sumq += (float) v.w * lb[4 * i + 3];
            }
            acc += sumq * (float) blk[b].d;
        }
        const size_t i = (size_t) tgid.y * (size_t) args.width + urow;
        const float n = x[i] * scales[(size_t) tgid.y * (size_t) args.hc_count + (size_t) s] *
                        w[urow];
        acc = (1.0f / (1.0f + exp(-acc))) * n;
    }

    for (uint m = 1; m < (uint) args.hc_count; m <<= 1) {
        acc += simd_shuffle_xor(acc, m);
    }
    if (s == 0 && j < args.hidden) {
        mixed[(size_t) tgid.y * (size_t) args.hidden + (size_t) j] = acc * args.inv_hc;
    }
}
