// Per-pixel L2 normalisation over the channel axis of an NCHW tensor, the
// Qwen-Image 2.1 VAE's norm: out[b, c, p] = x[b, c, p] / max(|x[b, :, p]|_2,
// eps) * gamma[c], with gamma carrying the reference's sqrt(C).
//
// It replaces six candle dispatches (sqr, a reduce over channels, sqrt,
// maximum, broadcast_div, broadcast_mul), four of them full passes over the
// activation and two on the broadcast kernels. This kernel reads the
// activation twice and writes it once.
//
// The sum of squares runs over channels in index order in one float, which is
// not the order candle's reduce uses, so the result agrees with the chain to
// rounding and not to the bit (channel_l2_norm.rs bounds it).
//
// Own library, no Metal-4 dependency.

#include <metal_stdlib>

using namespace metal;

// Matches dispatch.rs ChannelL2NormArgs (#[repr(C)]).
typedef struct {
    int32_t n;        // pixels: batch * plane
    int32_t channels; // C
    int32_t plane;    // H * W
    float   eps;      // floor of the norm
} channel_l2_norm_args;

// x, dst: batch * channels * plane contiguous f32, NCHW. gamma: `channels`
// contiguous f32. One thread per pixel.
kernel void kernel_channel_l2_norm(
        constant channel_l2_norm_args & args [[buffer(0)]],
        device const float * x             [[buffer(1)]],
        device const float * gamma         [[buffer(2)]],
        device       float * dst           [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    // Unsigned compare: the rounded-up grid emits stray threads past the end.
    if (tid >= (uint) args.n) {
        return;
    }
    const size_t plane = (size_t) args.plane;
    const size_t b = (size_t) tid / plane;
    const size_t p = (size_t) tid % plane;
    const size_t base = b * (size_t) args.channels * plane + p;
    float ss = 0.0f;
    for (int c = 0; c < args.channels; c++) {
        const float v = x[base + (size_t) c * plane];
        ss += v * v;
    }
    const float inv = 1.0f / max(sqrt(ss), args.eps);
    for (int c = 0; c < args.channels; c++) {
        const size_t i = base + (size_t) c * plane;
        dst[i] = x[i] * inv * gamma[c];
    }
}
