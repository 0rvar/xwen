// Vendored fused MoE weighted-combine kernels — the routed-expert combine tail
// of FusedExperts::forward (src/moe.rs). They replace the 3-4 strided candle
// broadcast/affine/sum passes over down=[seq, top_k, n_out] with ONE pass that
// reads `down` a single time, and are BIT-IDENTICAL to the candle chain by
// construction (compared bitwise in the combine.rs ops test), so the fused path
// is safe under every parity tier including strict.
//
// The identity rests on reproducing candle's `sum(1)` reduction order EXACTLY.
// candle dispatches `fast_sum_f32_strided` with out_length = seq*n_out
// threadgroups, each `block_dim` threads wide where block_dim =
// min(pipeline_max, next_power_of_two(top_k/2)); the reduced axis is top_k. Its
// per-thread loader accumulates `value = value + src[strided_idx(i)]` for i =
// tid, tid+block_dim, ... < top_k (candle reduce.metal loader), then the
// BLOCKSIZE<=32 path reduces the lane partials with the hardware `simd_sum`, and
// lane 0 stores. We mirror that per output: `block_dim` lanes, the same
// ascending per-lane loader partition, a shuffle-xor butterfly over strides 1, 2,
// ..., block_dim/2 (the order `simd_sum` folds those lanes in; the bitwise test
// fails on the reversed strides), lane-0 store — but we compute each loaded
// element in-register with candle's per-op rounding boundaries instead of reading
// a pre-materialized product. What we do NOT mirror is the launch: one
// threadgroup of block_dim (4 or 8) threads per output left the GPU idling on
// threadgroup dispatch, 2.6-3 ms for a 2048-token chunk that moves 0.1 GB, so
// each 256-thread threadgroup packs 256/block_dim outputs instead.
//
// SINGLE-SIMDGROUP CONSTRAINT: an output's lanes must sit inside one 32-lane
// simdgroup for the shuffles to reach them, so these kernels implement ONLY the
// block_dim <= 32 path. candle's >32 threadgroup-reduction tree is deliberately
// NOT reproduced (production top_k is 6 to 10, so candle's width
// next_pow2(top_k/2) is 4 or 8). `run_combine` (combine.rs dispatch) refuses
// host-side any top_k whose candle width would exceed 32, so a reduction wider
// than one simdgroup can never reach these kernels.
//   rescale-free : r3 = down * w                                (one f32 mul)
//   rescale      : r1 = down * col_l2 (f32 mul), r2 = r1 * 2^-15 (exact,
//                  power of two), r3 = r2 * w (f32 mul)
// then `value = value + r3` per element. The three scalars are NOT folded into
// one multiplier (that would reorder rounding), and no fma is used anywhere.
//
// FP contraction is disabled at file scope: candle's own combine kernels are
// single-op (a lone mul or a lone add — nothing to contract), so their per-op
// results are correctly-rounded f32; our fused loop has an adjacent mul and add
// that fast-math would otherwise contract into one fma (one rounding instead of
// two), breaking the identity. `#pragma clang fp contract(off)` pins the two
// roundings regardless of the library's math mode, and `#pragma clang fp
// reassociate(off)` pins the r1→r2→r3 multiply chain's written order (fast math
// would otherwise be licensed to regroup it, e.g. hoisting col_l2·2^-15·w into
// one scalar — different rounding). clang REJECTS unknown `fp` pragma options,
// so these compiling proves they are honored; together they close every
// fast-math transform that could touch a bit (the per-lane partition is at most
// two elements wide, so the add chain has no reassociation freedom either).
// `#pragma METAL fp math_mode(fast)` additionally pins the library math-mode
// axis to the value nil compile options resolve to today (candle's own kernels
// are compiled with explicit MTLMathMode::Fast), so a future OS default change
// cannot move this library's mode; the clang fp pragmas above constrain
// contraction/reassociation WITHIN that mode.
//
// A SEPARATE library from mm_id.metal / mv.metal / f16.metal (own runtime
// compile via src/ops/pipelines.rs, no Metal-4 dependency).

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp math_mode(fast)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

// Matches dispatch.rs CombineArgs (#[repr(C)]).
typedef struct {
    int32_t top_k;
    int32_t n_out;
    int32_t width;
    int32_t n;
} combine_args;

// One output per `width` adjacent lanes, `width` being candle's block_dim: lane
// t of an output's group loads k = t, t+width, ... < top_k (candle's loader
// partition), and the group folds its partials with a shuffle-xor butterfly over
// strides 1, 2, ..., width/2, the pairing `simd_sum` uses (the bitwise test
// holds it). `width` divides 32 and the threadgroup is a whole number of
// simdgroups, so a group never straddles a simdgroup. Bounded on the explicit
// `args.n` (seq * n_out), never on the grid; lanes past it still join the
// shuffles with a zero partial.
template <bool RESCALE>
inline void moe_combine(
        constant combine_args & args,
        device const float * down,
        device const float * col_l2,
        device const float * w,
        device       float * dst,
        uint gid) {
    const int top_k = args.top_k;
    const int n_out = args.n_out;
    const int width = args.width;
    const int did = (int) (gid / (uint) width);
    const int tid = (int) (gid % (uint) width);
    const bool live = did < args.n;

    float value = 0.0f;
    if (live) {
        const int s = did / n_out;
        const int c = did % n_out;
        const int down_base = s * top_k * n_out + c;
        const int sk_base   = s * top_k;
        for (int k = tid; k < top_k; k += width) {
            float d  = down[down_base + k * n_out];
            float ww = w[sk_base + k];
            float r3;
            if (RESCALE) {
                float l  = col_l2[sk_base + k];
                float r1 = d * l;
                float r2 = r1 * 0x1p-15f;
                r3 = r2 * ww;
            } else {
                r3 = d * ww;
            }
            value = value + r3;
        }
    }
    for (ushort m = 1; m < (ushort) width; m <<= 1) {
        value = value + simd_shuffle_xor(value, m);
    }
    if (live && tid == 0) {
        dst[did] = value;
    }
}

// Rescale-free combine: dst[s, c] = Σ_k down[s, k, c] * w[s, k].
kernel void kernel_moe_combine(
        constant combine_args & args [[buffer(0)]],
        device const float * down    [[buffer(1)]],
        device const float * w       [[buffer(2)]],
        device       float * dst     [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    moe_combine<false>(args, down, nullptr, w, dst, gid);
}

// Rescale combine: dst[s, c] = Σ_k (down[s, k, c] * col_l2[s, k]) * 2^-15 * w[s, k],
// with each multiply rounded separately (2^-15 is exact) — the per-column L2
// rescale the f16-tile down projection needs, undone here per candle's chain.
kernel void kernel_moe_combine_rescale(
        constant combine_args & args [[buffer(0)]],
        device const float * down    [[buffer(1)]],
        device const float * col_l2  [[buffer(2)]],
        device const float * w       [[buffer(3)]],
        device       float * dst     [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    moe_combine<true>(args, down, col_l2, w, dst, gid);
}
