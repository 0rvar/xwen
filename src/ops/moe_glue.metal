// Vendored fused MoE glue kernels — the routing decision and the block tail of
// MoeBlock::forward (src/moe.rs). Each replaces a chain of candle dispatches
// with ONE pass and is BIT-IDENTICAL to the chain it replaces (compared bitwise
// in the moe_glue.rs ops tests), so the fused path is safe under every parity
// tier including strict. The kill-switch back to the candle chains is
// XWEN_MOE_GLUE_CLASSIC.
//
// kernel_moe_router replaces the 7-dispatch routing chain that follows the
// router matmul — softmax_last_dim, arg_sort_last_dim(desc), narrow+contiguous,
// gather, sum_keepdim, clamp (maximum then minimum, each uploading a fresh
// 4-byte scalar buffer), broadcast_div — with one threadgroup per token. The
// router matmul itself stays a candle dispatch: it lowers to MLX's
// `gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4`, whose K-partition and shuffle-tree
// combine are not reproducible from a differently-shaped hand-written gemv, and
// whose accumulation order additionally depends on the output width — so
// concatenating the shared-expert gate row onto the router weight would change
// that gate's bits. Both matmuls therefore stay candle's.
//
// The identity rests on reproducing three candle kernels in sequence, each in
// its own launch geometry:
//   1. `softmax_f32` (candle reduce.metal): the ONLINE (Welford/Milakov) form,
//      NOT a max/exp/sum triple. One (m, d) pair per thread merged by
//      `MDReduceOp`, reduced through candle's block_reducer — threadgroup array
//      for width >= 64, then the 5-step `simd_shuffle_down` tree over lanes
//      0..31 (MDReduceOp is not a built-in simd op, so candle takes the shuffle
//      tree, not `simd_max`/`simd_sum`) — then `fast::exp(x - m) * fast::divide(1, d)`
//      with the reciprocal hoisted once per thread. Candle's threadgroup width
//      is `min(pipeline_max, next_pow2(n_expert/2))`; the host passes that width
//      in as `softmax_width` and this kernel branches on it exactly as candle's
//      compile-time BLOCKSIZE switch does. The per-thread element partition is
//      ASCENDING and strided by that width, in both the load and the finalize.
//   2. `asort_desc_f32` (candle sort.metal, itself llama.cpp's bitonic network):
//      run VERBATIM over the softmax probabilities, one thread per padded
//      column. This network is deterministic but NOT stable — its comparators
//      are strict and never consult the index — so equal probabilities do not
//      generally come out in ascending expert order. Reproducing the network is
//      the only way to reproduce which expert wins a tie, and a tie flip swaps
//      an entire expert's contribution, so no selection shortcut is admissible
//      here.
//   3. `fast_sum_f32` over the gathered top_k weights (candle reduce.metal
//      again, contiguous branch): threadgroup width `next_pow2(top_k/2)`, the
//      same ascending strided per-lane partition, folded by the hardware
//      `simd_sum`. Lanes beyond that width hold +0.0 here where candle simply
//      has no such thread; softmax probabilities are non-negative, so those
//      lanes are an exact additive identity.
// The tail then mirrors candle's `clamp` (a `maximum` then a `minimum`, both
// ternary selects — `MAX(x,y)` is `x > y ? x : y`, NOT `fmax`, so their NaN
// behavior differs) and its `bdiv` (a true per-element divide, no reciprocal
// hoist). The weight-sum floor arrives as a kernel argument so the Rust
// constant stays the single source of truth.
//
// kernel_moe_epilogue replaces the 4-dispatch block tail — the routed weighted
// combine, the shared-expert gate sigmoid, the gate broadcast_mul, and the
// routed+shared add — with one pass over `down`. Its reduction is
// `kernel_moe_combine`'s (see combine.metal): candle's `sum(1)` launch geometry,
// the same ascending per-lane partition, the same hardware `simd_sum`, lane-0
// store. The shared-expert term is folded in AFTER the reduction, in candle's
// order (`routed + shared`, where `shared` is `shexp * sigmoid(gate)`), and the
// sigmoid is candle's `usigmoid` — `1 / (1 + exp(-x))`, a true divide — computed
// per output element from the raw gate logit rather than read from a
// materialized tensor. The recomputation is deterministic, so every element
// sees the bits candle's materialized sigmoid tensor held.
//
// FP PRAGMA SPLIT — the two kernels need OPPOSITE treatment, which is why the
// pragmas are at BLOCK scope here (the delta.metal pattern) and not at file
// scope:
//   * kernel_moe_epilogue carries `contract(off)` / `reassociate(off)`, like
//     combine.metal: candle's own chain ops are single-rounding (a lone mul, a
//     lone add), so there is nothing fast math could fuse on candle's side,
//     while this kernel's adjacent mul and add would otherwise contract into one
//     fma and lose a rounding.
//   * kernel_moe_router carries NEITHER, deliberately — the rope.metal
//     rationale. Its inner expressions are transcriptions of candle's own kernel
//     bodies (`bigger.d + smaller.d * fast::exp(...)` in particular is a
//     multiply-add that candle's fast-math compile is free to contract), so it
//     must compile under the same latitude candle's kernels do. Pinning
//     contraction off here would pin it off on ONE side of the comparison only.
// `#pragma METAL fp math_mode(fast)` stays at file scope: it pins the library
// math-mode axis to what nil compile options resolve to today (candle compiles
// its kernels with explicit MTLMathMode::Fast), so a future OS default change
// cannot move this library off candle's mode. clang REJECTS unknown `fp` pragma
// options, so these compiling proves they are honored.
//
// A SEPARATE library from the other vendored sources (own runtime compile via
// src/ops/pipelines.rs, no Metal-4 dependency).

#include <metal_stdlib>
#include <metal_limits>

using namespace metal;

#pragma METAL fp math_mode(fast)

// Threadgroup-array bounds. The router's shared state is
// `MOE_ROUTER_MAX_SOFTMAX` (m, d) pairs + `MOE_ROUTER_MAX_EXPERTS` floats +
// `MOE_ROUTER_MAX_EXPERTS` uints + `MOE_ROUTER_MAX_TOP_K` floats; dispatch.rs
// refuses any geometry that would exceed one of these, and the moe_glue.rs
// geometry test cross-checks these numbers against the Rust constants.
#define MOE_ROUTER_MAX_EXPERTS 512
#define MOE_ROUTER_MAX_SOFTMAX 256
#define MOE_ROUTER_MAX_TOP_K 32

// Matches dispatch.rs MoeRouterArgs (#[repr(C)]).
typedef struct {
    int32_t n_expert;      // routed experts per token (softmax width in elements)
    int32_t n_expert_pad;  // next power of two >= n_expert; the bitonic network width
    int32_t top_k;         // experts selected per token
    int32_t softmax_width; // candle's softmax threadgroup width for n_expert
    int32_t sum_width;     // candle's fast_sum threadgroup width for top_k
    float   sum_floor;     // renormalization denominator floor
} moe_router_args;

// candle's `MD<float>`: the running (max, denominator) pair of the online
// softmax. `d` is always float on candle's side too.
typedef struct {
    float m;
    float d;
} moe_md;

// candle's MDReduceOp: merge two online-softmax partials. The `>` (not `>=`)
// picks `b` on a tie, exactly as candle's `a_bigger ? a : b` does.
static inline moe_md moe_md_merge(moe_md a, moe_md b) {
    bool a_bigger = a.m > b.m;
    moe_md bigger = a_bigger ? a : b;
    moe_md smaller = a_bigger ? b : a;
    moe_md res;
    res.d = bigger.d + smaller.d * fast::exp(smaller.m - bigger.m);
    res.m = bigger.m;
    return res;
}

// candle's `simd_shuffle_down(MD<T>)`: the two fields shuffle independently.
static inline moe_md moe_md_shuffle_down(moe_md v, ushort delta) {
    moe_md r;
    r.m = simd_shuffle_down(v.m, delta);
    r.d = simd_shuffle_down(v.d, delta);
    return r;
}

// One token's routing decision: softmax over `n_expert` logits, descending
// bitonic arg-sort, top_k gather, sum, floor-clamp, renormalize.
// `logits` is [seq, n_expert] f32 contiguous; `ids` [seq, top_k] u32 and
// `weights` [seq, top_k] f32 are the outputs. One threadgroup per token, with
// `n_expert_pad` threads (the bitonic network's width, which is always >= the
// softmax width since next_pow2(n) >= next_pow2(n/2)).
kernel void kernel_moe_router(
        constant moe_router_args & args [[buffer(0)]],
        device const float * logits  [[buffer(1)]],
        device       uint  * ids     [[buffer(2)]],
        device       float * weights [[buffer(3)]],
        uint tid [[thread_index_in_threadgroup]],
        uint row [[threadgroup_position_in_grid]]) {
    threadgroup moe_md md_shared[MOE_ROUTER_MAX_SOFTMAX];
    threadgroup moe_md md_total;
    threadgroup float probs[MOE_ROUTER_MAX_EXPERTS];
    threadgroup uint order[MOE_ROUTER_MAX_EXPERTS];
    threadgroup float selected[MOE_ROUTER_MAX_TOP_K];
    threadgroup float denom;

    const int n_expert = args.n_expert;
    const int pad = args.n_expert_pad;
    const int top_k = args.top_k;
    const int sw = args.softmax_width;
    const int col = (int) tid;

    device const float * row_logits = logits + (int) row * n_expert;

    // --- 1. candle softmax_f32, online form (see file header) --------------
    moe_md value = moe_md{ numeric_limits<float>::lowest(), 0.0f };
    if (col < sw) {
        for (int i = col; i < n_expert; i += sw) {
            value = moe_md_merge(value, moe_md{ row_logits[i], 1.0f });
        }
    }
    // candle's block_reducer, with its compile-time BLOCKSIZE switch expressed
    // over the runtime width. `sw` is threadgroup-uniform, so every barrier
    // below is reached by every thread.
    if (sw >= 64) {
        if (col < sw) {
            md_shared[col] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = sw / 2; s >= 64; s >>= 1) {
            if (col < s) {
                md_shared[col] = moe_md_merge(md_shared[col], md_shared[col + s]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (col < 32) {
        if (sw >= 64) {
            value = moe_md_merge(md_shared[col], md_shared[col + 32]);
            simdgroup_barrier(mem_flags::mem_none);
        }
        // Lanes past `sw` (which candle simply does not have, its threadgroup
        // being `sw` wide) still hold the {lowest, 0} identity, which merges
        // away exactly.
        if (sw >= 32) value = moe_md_merge(value, moe_md_shuffle_down(value, 16));
        if (sw >= 16) value = moe_md_merge(value, moe_md_shuffle_down(value, 8));
        if (sw >= 8) value = moe_md_merge(value, moe_md_shuffle_down(value, 4));
        if (sw >= 4) value = moe_md_merge(value, moe_md_shuffle_down(value, 2));
        if (sw >= 2) value = moe_md_merge(value, moe_md_shuffle_down(value, 1));
    }
    if (col == 0) {
        md_total = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // candle's finalize_softmax: one reciprocal per thread, then the same
    // ascending strided partition over the row.
    if (col < sw) {
        const float d_total_inverse = fast::divide(1.0f, md_total.d);
        for (int i = col; i < n_expert; i += sw) {
            probs[i] = fast::exp(row_logits[i] - md_total.m) * d_total_inverse;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- 2. candle asort_desc_f32: llama.cpp's bitonic network, verbatim ---
    order[col] = (uint) col;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 2; k <= pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (order[col] >= (uint) n_expert ||
                        (order[ixj] < (uint) n_expert &&
                         probs[order[col]] < probs[order[ixj]])) {
                        const uint t = order[col];
                        order[col] = order[ixj];
                        order[ixj] = t;
                    }
                } else {
                    if (order[ixj] >= (uint) n_expert ||
                        (order[col] < (uint) n_expert &&
                         probs[order[col]] > probs[order[ixj]])) {
                        const uint t = order[col];
                        order[col] = order[ixj];
                        order[ixj] = t;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // --- 3. candle gather: the top_k probabilities, in sorted order --------
    if (col < top_k) {
        selected[col] = probs[order[col]];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- 4. candle fast_sum_f32 over those top_k, then clamp --------------
    if (col < 32) {
        float acc = 0.0f;
        if (col < args.sum_width) {
            for (int i = col; i < top_k; i += args.sum_width) {
                acc = acc + selected[i];
            }
        }
        acc = simd_sum(acc);
        if (col == 0) {
            // candle's clamp: bmaximum then bminimum, both ternary selects.
            const float floored = acc > args.sum_floor ? acc : args.sum_floor;
            denom = floored < INFINITY ? floored : INFINITY;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- 5. candle bdiv: renormalize, one true divide per weight ----------
    if (col < top_k) {
        const int out = (int) row * top_k + col;
        ids[out] = order[col];
        weights[out] = selected[col] / denom;
    }
}

// Matches dispatch.rs MoeEpilogueArgs (#[repr(C)]).
typedef struct {
    int32_t top_k;
    int32_t n_out;
} moe_epilogue_args;

// dst[s, c] = (Σ_k down[s, k, c] * w[s, k]) + shexp[s, c] * sigmoid(gate[s]).
// `down` is [seq, top_k, n_out] f32, `w` [seq, top_k] f32, `shexp` [seq, n_out]
// f32, `gate` [seq, 1] f32 (the RAW shared-expert gate logit — the sigmoid runs
// here). One threadgroup per output element, `next_pow2(top_k/2)` threads wide,
// reproducing candle's `sum(1)` reduction geometry.
//
// SINGLE-SIMDGROUP CONSTRAINT, as in combine.metal: the reduction is one
// `simd_sum` folding exactly one 32-lane simdgroup, so a candle width above 32
// would leave lanes 32.. unfolded. `run_moe_epilogue` refuses such a top_k
// host-side.
kernel void kernel_moe_epilogue(
        constant moe_epilogue_args & args [[buffer(0)]],
        device const float * down  [[buffer(1)]],
        device const float * w     [[buffer(2)]],
        device const float * shexp [[buffer(3)]],
        device const float * gate  [[buffer(4)]],
        device       float * dst   [[buffer(5)]],
        uint tid       [[thread_index_in_threadgroup]],
        uint dst_id    [[threadgroup_position_in_grid]],
        uint block_dim [[threads_per_threadgroup]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    const int top_k = args.top_k;
    const int n_out = args.n_out;
    const int did = (int) dst_id;
    const int s = did / n_out;
    const int c = did % n_out;
    const int down_base = s * top_k * n_out + c;
    const int sk_base = s * top_k;

    float value = 0.0f;
    for (int k = (int) tid; k < top_k; k += (int) block_dim) {
        float d = down[down_base + k * n_out];
        float ww = w[sk_base + k];
        float r3 = d * ww;
        value = value + r3;
    }
    value = simd_sum(value);
    if (tid == 0) {
        // candle's usigmoid: recip(1 + exp(-x)) — a true divide, no fma.
        const float g = gate[s];
        const float sig = 1.0f / (1.0f + exp(-g));
        // candle's bmul, then badd in the block's order: routed + shared.
        const float shared = shexp[did] * sig;
        dst[did] = value + shared;
    }
}
