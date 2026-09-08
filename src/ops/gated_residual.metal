// Gated residual add, the tail of every modulated Z-Image transformer
// sub-block: out = h + g[c] * y, where g is one gate value per channel
// (`tanh(gate)` from the adaLN modulation, shape [1, 1, C]) and h, y are the
// residual stream and the sub-block output, [.., C].
//
// It replaces two candle dispatches, `g.broadcast_mul(y)` then `h + (...)`,
// each a full pass over the activation, and the first on candle's broadcast
// kernel, which the Z-Image microbench measured at 48 GB/s against 530 for the
// contiguous binary kernel. This kernel reads h and y once and writes out once.
//
// The arithmetic is the chain's: the multiply rounds, then the add rounds,
// with FP contraction pinned OFF so the two never fuse into an fma. The
// gated_residual.rs test compares the two bitwise on production shapes.
//
// Own library, no Metal-4 dependency.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp math_mode(fast)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

// Matches dispatch.rs GatedResidualArgs (#[repr(C)]).
typedef struct {
    int32_t n;        // total elements
    int32_t channels; // C, the gate's length; a row of h/y is C elements
} gated_residual_args;

// h, y, dst: n contiguous f32 elements, rows of `channels`. gate: `channels`
// contiguous f32. One thread per element.
kernel void kernel_gated_residual(
        constant gated_residual_args & args [[buffer(0)]],
        device const float * h             [[buffer(1)]],
        device const float * y             [[buffer(2)]],
        device const float * gate          [[buffer(3)]],
        device       float * dst           [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    // Unsigned compare: the rounded-up grid emits stray threads past the end.
    if (tid >= (uint) args.n) {
        return;
    }
    const uint c = tid % (uint) args.channels;
    const float gy = gate[c] * y[tid];
    dst[tid] = h[tid] + gy;
}
