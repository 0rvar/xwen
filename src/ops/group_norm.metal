// Fused GroupNorm for the Z-Image VAE decoder, f32 NCHW, in two passes plus a
// per-channel fold.
//
// candle's GroupNorm is about nine full-tensor dispatches (reshape, two
// reductions, the centering, the square, the divide and two broadcasts). Here
// the statistics come from ONE read of the tensor and the normalization is
// folded into a per-(batch, channel) affine, y = x * scale + shift, which the
// caller either applies with `kernel_group_norm_apply` (one more pass, with
// an optional silu) or hands to the direct convolution, which applies it on
// its input read so the normalized tensor is never written at all.
//
// A group is `channels / groups` channels of one batch element, one
// contiguous run of `len` floats in NCHW. `kernel_group_norm_partials`
// splits each run into P slices, one threadgroup each, and accumulates the
// sum and the sum of squares of `x - x0`, x0 being the run's first element:
// the shift keeps the single-pass variance from cancelling when the mean is
// far from zero, and a sample of the data is a shift within a standard
// deviation or so of the mean. `kernel_group_norm_fold` sums the P partials
// per group and writes scale = rstd * gamma[c], shift = beta[c] - mean *
// scale for every (batch, channel).
//
// The arithmetic is not candle's chain (which centers first, then squares),
// so results agree to f32 rounding and not bitwise; the group_norm.rs tests
// hold rel_l2 under 1e-5 against candle, with the data offset from zero.
// Fast math like the rest of the vendored libraries.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp math_mode(fast)

// Matches dispatch.rs GroupNormPartialsArgs (#[repr(C)]).
typedef struct {
    int32_t len;      // floats per (batch, group) run
    int32_t partials; // slices per run, P
    int32_t chunk;    // floats per slice, a multiple of 4 when vec4 is set
    int32_t vec4;     // 1 when len is a multiple of 4 and float4 loads apply
} group_norm_partials_args;

// Matches dispatch.rs GroupNormFoldArgs (#[repr(C)]).
typedef struct {
    int32_t n;        // batch * channels, the threads with work
    int32_t channels;
    int32_t groups;
    int32_t partials;
    int32_t len;
    float   eps;
} group_norm_fold_args;

// Matches dispatch.rs GroupNormApplyArgs (#[repr(C)]).
typedef struct {
    int32_t n;    // elements, or float4s when vec4 is set
    int32_t hw;   // elements per channel plane, or float4s when vec4 is set
    int32_t silu;
    int32_t vec4;
} group_norm_apply_args;

constant constexpr int GN_THREADS = 256;

// Grid: (P, batch * groups). Writes partials[(run * P + slice) * 2 + {0, 1}]
// as the slice's sum and sum of squares of (x - x0).
kernel void kernel_group_norm_partials(
        constant group_norm_partials_args & args [[buffer(0)]],
        device const float * x                  [[buffer(1)]],
        device       float * partials           [[buffer(2)]],
        uint2 tgid [[threadgroup_position_in_grid]],
        uint   tid [[thread_index_in_threadgroup]],
        uint  sgid [[simdgroup_index_in_threadgroup]],
        uint  lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[2 * (GN_THREADS / 32)];
    const int run = (int) tgid.y;
    const int slice = (int) tgid.x;
    device const float * xr = x + (size_t) run * args.len;
    const float x0 = xr[0];
    const int start = slice * args.chunk;
    const int end = min(start + args.chunk, args.len);
    float s1 = 0.0f;
    float s2 = 0.0f;
    if (args.vec4 != 0) {
        device const float4 * xv = (device const float4 *) (xr + start);
        const int n4 = (end - start) / 4;
        for (int i = (int) tid; i < n4; i += GN_THREADS) {
            const float4 d = xv[i] - x0;
            s1 += d.x + d.y + d.z + d.w;
            s2 += dot(d, d);
        }
    } else {
        for (int i = start + (int) tid; i < end; i += GN_THREADS) {
            const float d = xr[i] - x0;
            s1 += d;
            s2 += d * d;
        }
    }
    s1 = simd_sum(s1);
    s2 = simd_sum(s2);
    if (lane == 0) {
        red[sgid * 2] = s1;
        red[sgid * 2 + 1] = s2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t1 = 0.0f;
        float t2 = 0.0f;
        for (int i = 0; i < GN_THREADS / 32; i++) {
            t1 += red[i * 2];
            t2 += red[i * 2 + 1];
        }
        const size_t o = ((size_t) run * args.partials + slice) * 2;
        partials[o] = t1;
        partials[o + 1] = t2;
    }
}

// One thread per (batch, channel), `batch * channels` in all.
kernel void kernel_group_norm_fold(
        constant group_norm_fold_args & args [[buffer(0)]],
        device const float * x              [[buffer(1)]],
        device const float * partials       [[buffer(2)]],
        device const float * gamma          [[buffer(3)]],
        device const float * beta           [[buffer(4)]],
        device       float * scale          [[buffer(5)]],
        device       float * shift          [[buffer(6)]],
        uint tid [[thread_position_in_grid]]) {
    // Bounded by the argument, not the grid: the launch rounds up to whole
    // threadgroups and the stray threads would write past scale and shift.
    if (tid >= (uint) args.n) {
        return;
    }
    const int c = (int) tid % args.channels;
    const int b = (int) tid / args.channels;
    const int cpg = args.channels / args.groups;
    const int run = b * args.groups + c / cpg;
    const float x0 = x[(size_t) run * args.len];
    float s1 = 0.0f;
    float s2 = 0.0f;
    device const float * p = partials + (size_t) run * args.partials * 2;
    for (int i = 0; i < args.partials; i++) {
        s1 += p[i * 2];
        s2 += p[i * 2 + 1];
    }
    const float inv_n = 1.0f / (float) args.len;
    const float m = s1 * inv_n;
    const float var = max(s2 * inv_n - m * m, 0.0f);
    const float rstd = 1.0f / sqrt(var + args.eps);
    const float sc = rstd * gamma[c];
    scale[tid] = sc;
    shift[tid] = beta[c] - (x0 + m) * sc;
}

// y = x * scale[b, c] + shift[b, c], then silu when asked. One thread per
// element, or per float4 when the channel plane is a multiple of four.
kernel void kernel_group_norm_apply(
        constant group_norm_apply_args & args [[buffer(0)]],
        device const float * x               [[buffer(1)]],
        device const float * scale           [[buffer(2)]],
        device const float * shift           [[buffer(3)]],
        device       float * dst             [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= (uint) args.n) {
        return;
    }
    const int bc = (int) tid / args.hw;
    const float sc = scale[bc];
    const float sh = shift[bc];
    if (args.vec4 != 0) {
        float4 v = ((device const float4 *) x)[tid] * sc + sh;
        if (args.silu != 0) {
            v = v / (1.0f + exp(-v));
        }
        ((device float4 *) dst)[tid] = v;
    } else {
        float v = x[tid] * sc + sh;
        if (args.silu != 0) {
            v = v / (1.0f + exp(-v));
        }
        dst[tid] = v;
    }
}
