// Vendored fused MoE SwiGLU-activation kernel — the silu*mul glue between the
// up/gate expert matvecs and the down matvec in FusedExperts::forward
// (src/moe.rs). It replaces candle's two elementwise dispatches (a `silu` unary
// pass then a `mul` binary pass over the [seq, top_k, expert_ff] activation)
// with ONE pass that reads `gate`/`up` once and writes `act`, and is
// BIT-IDENTICAL to that candle chain by construction (compared bitwise in the
// silu_mul.rs ops test), so the fused path is safe under every parity tier
// including strict.
//
// The identity rests on reproducing candle's two ops with the SAME per-op
// rounding boundaries:
//   silu : candle's usilu (metal_src/unary.metal) is `x / (1 + exp(-x))` for the
//          f32 unary kernel — copied here VERBATIM (`g / (1 + exp(-g))`). The
//          integer `1` widens to exactly 1.0f, matching candle.
//   mul  : candle's bmul (metal_src/binary.metal) is `x * y` — the silu result
//          times `up`, a separate f32 rounding.
// candle materializes the silu output to an f32 buffer before the mul reads it,
// so silu and mul each round to f32 independently. This kernel keeps `s` in an
// f32 register (already f32-rounded by the division) and then multiplies, which
// is the same two roundings PROVIDED no fast-math transform fuses or reorders
// across the boundary — which the fp pragmas below pin.
//
// FP contraction / reassociation are disabled at file scope: without them,
// fast math could reassociate `(g / D) * up` into e.g. `g * (up / D)` (a
// different rounding), fusing the silu result and the multiply into one
// expression. `#pragma clang fp reassociate(off)` fixes the written expression
// tree — `s = g / D` rounded, then `s * up` rounded — so the boundary matches
// candle's two separate kernels; `#pragma clang fp contract(off)` pins the (here
// vacuous — there is no multiply-add) contraction axis for parity with the
// sibling vendored glue files. clang REJECTS unknown `fp` pragma options, so
// these compiling proves they are honored. `#pragma METAL fp math_mode(fast)`
// pins the library math-mode axis to what nil compile options resolve to today
// (candle's own kernels are compiled with explicit MTLMathMode::Fast), so a
// future OS default change cannot move this library's mode WITHIN which the
// silu expression's own arithmetic (the `exp`, and any fast-math reciprocal
// lowering of the division) stays identical to candle's fast-compiled usilu.
//
// A SEPARATE library from mm_id.metal / mv.metal / f16.metal / combine.metal
// (own runtime compile via src/ops/pipelines.rs, no Metal-4 dependency).

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp math_mode(fast)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

// Matches dispatch.rs SiluMulArgs (#[repr(C)]).
typedef struct {
    int32_t n;
} silu_mul_args;

// act[i] = silu(gate[i]) * up[i], with silu = x / (1 + exp(-x)) (candle's usilu)
// and the multiply each rounded separately (see file header). `gate`, `up` and
// `dst` are contiguous f32 of the same length; one thread per element.
kernel void kernel_moe_silu_mul(
        constant silu_mul_args & args [[buffer(0)]],
        device const float * gate     [[buffer(1)]],
        device const float * up       [[buffer(2)]],
        device       float * dst      [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    // Unsigned compare: at n == i32::MAX (the host-side launch cap) the
    // rounded-up grid emits a stray thread whose tid wraps negative under an
    // (int) cast, slipping past a signed guard into a one-element OOB write.
    if (tid >= (uint) args.n) {
        return;
    }
    const float g = gate[tid];
    const float s = g / (1 + exp(-g));
    dst[tid] = s * up[tid];
}

// Matches dispatch.rs SiluMulL2Args (#[repr(C)]).
typedef struct {
    int32_t ff;        // row width (expert_ff), <= SILU_MUL_L2_MAX_FF
    int32_t n_rows;    // seq * top_k
    float   scale;     // the f16-headroom factor (32768)
    float   clamp_min; // col_l2 floor (1e-8)
    float   clamp_max; // col_l2 ceiling (1e30)
} silu_mul_l2_args;

#define SILU_MUL_L2_THREADS 256
#define SILU_MUL_L2_MAX_FF 1024
// The kernel keeps each thread's slice of the row in a fixed-size register
// array, so the ceiling is exactly this many columns per thread; the host
// mirrors both constants (dispatch.rs SILU_MUL_L2_*, cross-checked by the
// silu_mul.rs `metal_and_host_constants_agree` test).
#define SILU_MUL_L2_COLS_PER_THREAD (SILU_MUL_L2_MAX_FF / SILU_MUL_L2_THREADS)
static_assert(SILU_MUL_L2_MAX_FF == SILU_MUL_L2_COLS_PER_THREAD * SILU_MUL_L2_THREADS,
              "the per-thread register array must cover the row ceiling exactly");

// The f16-tile prefill branch's activation glue in ONE pass: for each
// [token, slot] row of the [seq, top_k, expert_ff] gate/up pair,
//   act    = silu(gate) * up              (the kernel above's expression, verbatim)
//   col_l2 = clamp(sqrt(Σ act²), clamp_min, clamp_max)
//   act_s  = (act * scale) / col_l2
// which is what the candle chain in FusedExperts::project_inner computes as
// sqr → sum_keepdim → sqrt → clamp → affine(scale) → broadcast_div (six
// dispatches over the activation). `act_s` is what the down gemm consumes and
// `col_l2` is what `combine` divides back out; the per-op rounding of each
// elementwise step matches the chain (`sqr` as `a*a`, the scale as a separate
// multiply, then the divide), so the only place the two can differ is the SUM.
//
// Reduction order (fixed, so the result is deterministic run over run): thread
// `i` of the 256 accumulates its elements SEQUENTIALLY in ascending column order
// (columns i, i+256, i+512, i+768), then a threadgroup tree halves 256 → 1
// (partial[i] += partial[i + s] for s = 128, 64, ..., 1), barrier between
// levels. candle's `sum_keepdim` reduces in its own strided/simd order, so the
// two agree to accumulation-order noise (~1e-7 relative on the sum, which the
// sqrt halves) — bounded in the silu_mul.rs test, NOT bitwise. This kernel is
// therefore off the strict parity tier by construction (the strict candidate runs
// mv_id, which never takes the rescale branch) and graded by mm / decode / ppl.
//
// One threadgroup of 256 threads per row; rows wider than 1024 columns fall
// back to the chain on the host (`run_silu_mul_l2` bails, the caller decides).
kernel void kernel_moe_silu_mul_l2(
        constant silu_mul_l2_args & args [[buffer(0)]],
        device const float * gate       [[buffer(1)]],
        device const float * up         [[buffer(2)]],
        device       float * act_s      [[buffer(3)]],
        device       float * col_l2     [[buffer(4)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[SILU_MUL_L2_THREADS];

    if (row >= (uint) args.n_rows) {
        return;
    }
    const uint ff = (uint) args.ff;
    const uint base = row * ff;

    // Each thread's up-to-4 activation values stay in registers between the
    // reduction and the rescale store — no threadgroup staging of the row.
    float act[SILU_MUL_L2_COLS_PER_THREAD];
    float acc = 0.0f;
    for (uint r = 0; r < SILU_MUL_L2_COLS_PER_THREAD; ++r) {
        const uint c = tid + r * SILU_MUL_L2_THREADS;
        if (c < ff) {
            const float g = gate[base + c];
            const float s = g / (1 + exp(-g));
            const float a = s * up[base + c];
            act[r] = a;
            acc += a * a;
        }
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = SILU_MUL_L2_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // clamp(x, lo, hi) as candle's maximum(lo) then minimum(hi).
    const float l2 = min(max(sqrt(partial[0]), args.clamp_min), args.clamp_max);
    if (tid == 0) {
        col_l2[row] = l2;
    }
    for (uint r = 0; r < SILU_MUL_L2_COLS_PER_THREAD; ++r) {
        const uint c = tid + r * SILU_MUL_L2_THREADS;
        if (c < ff) {
            act_s[base + c] = (act[r] * args.scale) / l2;
        }
    }
}
