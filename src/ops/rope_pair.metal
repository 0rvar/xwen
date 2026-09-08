// Interleaved-pair rotary embedding, the form the Z-Image transformer uses
// (`view_as_complex(x.reshape(..., -1, 2))` in the reference): dims (2i, 2i+1)
// rotate together by the angle in column i of the cos/sin tables. This is NOT
// the NEoX by-halves form of rope.metal, which pairs (i, i + n_rot/2).
//
// It replaces the candle chain in src/zimage/transformer.rs `apply_rotary_emb`:
// two strided views onto the even and odd elements, four broadcast multiplies,
// a subtract, an add and a `stack` back to interleaved. Every op in that chain
// runs on candle's strided binary kernel because the views have stride 2 on
// the last axis, so it costs 6-7x its traffic. This kernel reads x once and
// writes it once, coalesced.
//
// The arithmetic is the chain's, expression for expression:
//   y_re = x_re * c - x_im * s
//   y_im = x_re * s + x_im * c
// with FP contraction and reassociation pinned OFF so each multiply and each
// add rounds separately, the way candle's separate dispatches round. The
// rope_pair.rs test compares the two bitwise on production shapes; the parity
// gate grades the whole transformer with this kernel in place. Rotation in f32
// with one rounding at the store is one of the four deliberate corrections
// toward the reference (docs/zimage.md), and it is kept.
//
// A SEPARATE library from rope.metal, whose header explains why that file must
// stay pragma-free. No Metal-4 dependency.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp math_mode(fast)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

// Matches dispatch.rs RopePairArgs (#[repr(C)]).
typedef struct {
    int32_t batch;
    int32_t seq;
    int32_t heads;
    int32_t half_dim;  // head_dim / 2: pairs per row
} rope_pair_args;

// src, dst: [batch, seq, heads, 2 * half_dim] f32 contiguous.
// cos_t, sin_t: [seq, half_dim] f32 contiguous, one angle per (token, pair),
// shared by every head and every batch row.
// One thread per pair: it reads the two adjacent floats of its pair and writes
// the two rotated ones, so a simdgroup reads a contiguous 256-byte span.
kernel void kernel_rope_pair(
        constant rope_pair_args & args [[buffer(0)]],
        device const float * src       [[buffer(1)]],
        device const float * cos_t     [[buffer(2)]],
        device const float * sin_t     [[buffer(3)]],
        device       float * dst       [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    const uint n_pairs = (uint) args.batch * (uint) args.seq * (uint) args.heads
                       * (uint) args.half_dim;
    // Unsigned compare: the rounded-up grid emits stray threads past the end.
    if (tid >= n_pairs) {
        return;
    }
    const uint i = tid % (uint) args.half_dim;
    const uint row = tid / (uint) args.half_dim;      // (b * seq + s) * heads + h
    const uint s = (row / (uint) args.heads) % (uint) args.seq;
    const uint x_idx = row * (uint) (2 * args.half_dim) + 2 * i;
    const uint cs_idx = s * (uint) args.half_dim + i;

    const float x_re = src[x_idx];
    const float x_im = src[x_idx + 1];
    const float c = cos_t[cs_idx];
    const float sn = sin_t[cs_idx];

    dst[x_idx]     = x_re * c - x_im * sn;
    dst[x_idx + 1] = x_re * sn + x_im * c;
}
