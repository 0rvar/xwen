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

// ---------------------------------------------------------------------------
// The fused shared expert at decode.
//
// A decode step's MoE block spends five of its twelve dispatches on the shared
// expert: the gate gemv, the up gemv, `kernel_moe_silu_mul`, the down gemv, and
// the `ffn_gate_inp_shexp` scalar logit — four candle launches and one vendored
// one, moving 1.74 MB each at the production geometry where the routed experts
// next door move thirty. Below a handful of tokens that is five launch
// latencies for bytes the machine streams in a fraction of one.
//
// These two kernels swallow all five:
//   kernel_moe_shexp_gate_up    gate gemv + up gemv + silu*mul + the gate logit
//   kernel_moe_epilogue_shexp   the epilogue with the shexp DOWN gemv folded in
// leaving ONE dispatch for the shared expert's projections and none for its
// glue — the sigmoid, the broadcast multiply and the routed+shared add were
// already inside `kernel_moe_epilogue`, and the down projection now joins them.
// Four launches per MoE layer disappear.
//
// ROUNDING: BOUNDED, not bitwise, unlike everything above it in this file. All
// three dot products are reassociated — the gate/up rows fold per-thread
// partials through `simd_sum` where candle's `QMatMul` gemv folds its own
// partition, the shexp down row folds per-lane block partials the same way, and
// the routed combine runs over 32 lanes here where `kernel_moe_epilogue` runs
// over `next_pow2(top_k/2)` of them. That is why `kernel_moe_epilogue` is NOT
// touched and stays the strict tier's bitwise anchor: this is a separate kernel
// taken only at `n <= XWEN_MOE_SHEXP_FUSED_MAX_N`, and `XWEN_MOE_SHEXP_CLASSIC`
// (or `XWEN_MOE_GLUE_CLASSIC`, which disables the whole epilogue path) restores
// the five-dispatch chain. The bound is rel_l2 <= 1e-5 against an f32 host
// reference (`shexp_fused_matches_reference`).
//
// LAYOUT, and why neither kernel is one threadgroup per token. The refuted
// XWEN_MOE_DUAL experiment is the constraint that shapes both: merging two
// bandwidth-saturating `mul_mv_id` gathers into one dispatch LOST 3 tok/s on
// the 35B, because the parallelism was what kept memory requests outstanding.
// Nothing here merges a saturating dispatch — the routed gemms are untouched —
// and both kernels keep a wide grid anyway. The first launches one threadgroup
// per (row tile of the two projections, token) plus one for the gate logit,
// exactly as `kernel_hc_gate_down` carries its injection head as an extra
// threadgroup. The second inherits `kernel_moe_epilogue`'s grid unchanged: one
// threadgroup per OUTPUT ELEMENT, `n * n_out` of them, each reading one
// `ffn_down_shexp` row — so the whole 1.74 MB plane is read once per token
// across a grid that was already there, with no extra allocation. What the
// first kernel re-reads across its threadgroups is the token's activation row,
// 10 KiB at the production geometry: a cache working set, not a bandwidth cost.
//
// The two partition k differently because their reductions are different
// lengths. kernel_moe_shexp_gate_up splits each `hidden/32`-block row across
// MOE_SHEXP_THREADS threads, interleaved, and stages the token's own slice of
// the activation in registers ONCE — reused across the tile's rows, both
// planes, and the gate logit, which is the whole reason the logit rides along
// for free. kernel_moe_epilogue_shexp gives its single simdgroup the
// `inner/32`-block row, at most MOE_SHEXP_MAX_BLK_PER_LANE blocks per lane.
// ---------------------------------------------------------------------------

// block_q8_0 (ggml-common.h): one f16 delta then 32 int8 quants, 34 bytes. All
// three shared-expert projections ship q8_0 on every shipped checkpoint, and
// these kernels read them directly rather than through QMatMul. Declared here
// rather than shared with q8.metal or hc.metal: each vendored .metal is its own
// runtime-compiled library.
#define QK8_0 32
typedef struct {
    half   d;
    int8_t qs[QK8_0];
} moe_block_q8_0;

// Threads per threadgroup in kernel_moe_shexp_gate_up: four simdgroups. It does
// NOT have to divide the row's block count — a hidden of 2560 is 80 blocks and
// leaves threads 80.. with a +0.0 accumulator, an exact additive identity — so
// the host admits any hidden that is a whole number of blocks. Mirrored by
// dispatch.rs MOE_SHEXP_THREADS.
#define MOE_SHEXP_THREADS 128
#define MOE_SHEXP_SIMDGROUPS (MOE_SHEXP_THREADS / 32)

// q8_0 blocks of the activation row one thread of kernel_moe_shexp_gate_up
// owns, at most. It stages them in registers (32 floats each) and the row dots,
// both planes and the gate logit all read them from there, so this bounds the
// kernel's register footprint; the host refuses a hidden wider than
// MOE_SHEXP_THREADS * MOE_SHEXP_MAX_BLK_PER_THREAD blocks (8192 elements).
// Production is 80 blocks (Flash-Next, hidden 2560) or 64 (35B-A3B, hidden
// 2048) — one per thread. Mirrored by dispatch.rs
// MOE_SHEXP_MAX_BLK_PER_THREAD.
#define MOE_SHEXP_MAX_BLK_PER_THREAD 2

// Bottleneck rows one threadgroup of kernel_moe_shexp_gate_up computes. Each
// costs two full-width dots against the same staged activation, so the rows
// share one pass over it and pay only their own weight bytes; the per-row
// accumulator pairs are registers, which is what bounds this. Mirrored by
// dispatch.rs MOE_SHEXP_ROWS_PER_TG.
#define MOE_SHEXP_ROWS_PER_TG 4

// q8_0 blocks of one `ffn_down_shexp` row a single lane of
// kernel_moe_epilogue_shexp owns, at most — the host refuses an `inner` wider
// than 32 * MOE_SHEXP_MAX_BLK_PER_LANE (4096). Production is 20 blocks (inner
// 640) or 16 (inner 512), so under one per lane. Mirrored by dispatch.rs
// MOE_SHEXP_MAX_BLK_PER_LANE.
#define MOE_SHEXP_MAX_BLK_PER_LANE 4

// Matches dispatch.rs MoeShexpGateUpArgs (#[repr(C)]).
typedef struct {
    int32_t hidden;
    int32_t inner;
    int32_t nblk;     // hidden / QK8_0, the q8_0 blocks in one gate/up row
    int32_t n_row_tg; // ceil(inner / MOE_SHEXP_ROWS_PER_TG)
} moe_shexp_gate_up_args;

// The shared expert's two input projections, their SwiGLU activation, and the
// scalar gate logit, in one launch.
//
// Grid is (n_row_tg + 1, n): threadgroup `g < n_row_tg` computes bottleneck rows
// `g * MOE_SHEXP_ROWS_PER_TG ..` of `silu(gate . x) * (up . x)`, and the LAST
// one computes `<x[s], ffn_gate_inp_shexp>`. Every threadgroup stages its own
// slice of the token's activation row; that is a read of 10 KiB per threadgroup
// at the production geometry, cheaper than a second launch.
//
// Outputs: `h[t, inner]`, the ungated SwiGLU bottleneck, and `logit[t, 1]`, the
// RAW pre-sigmoid gate logit — the sigmoid stays in the epilogue, where it
// already was.
kernel void kernel_moe_shexp_gate_up(
        constant moe_shexp_gate_up_args & args [[buffer(0)]],
        device const float * x                 [[buffer(1)]],
        device const moe_block_q8_0 * gate_w   [[buffer(2)]],
        device const moe_block_q8_0 * up_w     [[buffer(3)]],
        device const float * gate_inp          [[buffer(4)]],
        device       float * h                 [[buffer(5)]],
        device       float * logit             [[buffer(6)]],
        uint2 tgid    [[threadgroup_position_in_grid]],
        uint2 tpos    [[thread_position_in_threadgroup]],
        uint  sgid    [[simdgroup_index_in_threadgroup]],
        uint  lane    [[thread_index_in_simdgroup]],
        uint  sgcount [[simdgroups_per_threadgroup]]) {
    // The launch is `MOE_SHEXP_THREADS x 1`, so the row component is the whole
    // partition (the vector arity has to match tgid's).
    const uint tid = tpos.x;
    // One slot per (row of this tile, simdgroup), for each of the two
    // projections. The gate-logit threadgroup reduces one value and uses the
    // first MOE_SHEXP_SIMDGROUPS slots of the first array.
    threadgroup float partial_g[MOE_SHEXP_ROWS_PER_TG * MOE_SHEXP_SIMDGROUPS];
    threadgroup float partial_u[MOE_SHEXP_ROWS_PER_TG * MOE_SHEXP_SIMDGROUPS];

    const size_t row = (size_t) tgid.y * (size_t) args.hidden;

    // This thread's slice of the token's activation, staged once. The loop bound
    // is the compile-time maximum with a runtime guard inside, so every index
    // into the array is a constant after unrolling and the array stays in
    // registers.
    float xv[MOE_SHEXP_MAX_BLK_PER_THREAD * QK8_0];
    for (int p = 0; p < MOE_SHEXP_MAX_BLK_PER_THREAD; ++p) {
        const int blk = (int) tid + p * MOE_SHEXP_THREADS;
        if (blk < args.nblk) {
            const size_t base = row + (size_t) blk * QK8_0;
            for (int i = 0; i < QK8_0; ++i) {
                xv[p * QK8_0 + i] = x[base + (size_t) i];
            }
        }
    }

    if ((int) tgid.x < args.n_row_tg) {
        // MOE_SHEXP_ROWS_PER_TG rows of BOTH projections against the staged
        // activation. Per q8_0 block: an f32 dot of the 32 quants, then one
        // multiply by the block's delta — the same form as
        // kernel_mul_mv_q8_0_f32_attn (q8.metal).
        const int row0 = (int) tgid.x * MOE_SHEXP_ROWS_PER_TG;
        float acc_g[MOE_SHEXP_ROWS_PER_TG];
        float acc_u[MOE_SHEXP_ROWS_PER_TG];
        for (short r = 0; r < MOE_SHEXP_ROWS_PER_TG; ++r) {
            acc_g[r] = 0.0f;
            acc_u[r] = 0.0f;
        }
        for (short r = 0; r < MOE_SHEXP_ROWS_PER_TG; ++r) {
            // Uniform across the threadgroup: a ragged last tile skips the same
            // rows in every thread, so nothing below diverges.
            if (row0 + (int) r >= args.inner) {
                continue;
            }
            device const moe_block_q8_0 * grow =
                gate_w + (size_t) (row0 + (int) r) * (size_t) args.nblk;
            device const moe_block_q8_0 * urow =
                up_w + (size_t) (row0 + (int) r) * (size_t) args.nblk;
            for (int p = 0; p < MOE_SHEXP_MAX_BLK_PER_THREAD; ++p) {
                const int blk = (int) tid + p * MOE_SHEXP_THREADS;
                if (blk < args.nblk) {
                    device const moe_block_q8_0 * bg = grow + blk;
                    device const moe_block_q8_0 * bu = urow + blk;
                    // packed_char4, not char4: the quants start two bytes into
                    // a 34-byte block, so a naturally aligned vector load would
                    // fault.
                    device const packed_char4 * qg = (device const packed_char4 *) bg->qs;
                    device const packed_char4 * qu = (device const packed_char4 *) bu->qs;
                    float sg = 0.0f;
                    float su = 0.0f;
                    for (short i = 0; i < QK8_0 / 4; ++i) {
                        const packed_char4 vg = qg[i];
                        const packed_char4 vu = qu[i];
                        const float x0 = xv[p * QK8_0 + 4 * i + 0];
                        const float x1 = xv[p * QK8_0 + 4 * i + 1];
                        const float x2 = xv[p * QK8_0 + 4 * i + 2];
                        const float x3 = xv[p * QK8_0 + 4 * i + 3];
                        sg += (float) vg.x * x0;
                        sg += (float) vg.y * x1;
                        sg += (float) vg.z * x2;
                        sg += (float) vg.w * x3;
                        su += (float) vu.x * x0;
                        su += (float) vu.y * x1;
                        su += (float) vu.z * x2;
                        su += (float) vu.w * x3;
                    }
                    acc_g[r] += sg * (float) bg->d;
                    acc_u[r] += su * (float) bu->d;
                }
            }
        }
        // One reduction pass for all MOE_SHEXP_ROWS_PER_TG rows: the per-row
        // simd_sums need no barrier between them, only the single fence before
        // the serial fold.
        for (short r = 0; r < MOE_SHEXP_ROWS_PER_TG; ++r) {
            const float sum_g = simd_sum(acc_g[r]);
            const float sum_u = simd_sum(acc_u[r]);
            if (lane == 0) {
                partial_g[(uint) r * MOE_SHEXP_SIMDGROUPS + sgid] = sum_g;
                partial_u[(uint) r * MOE_SHEXP_SIMDGROUPS + sgid] = sum_u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < MOE_SHEXP_ROWS_PER_TG && row0 + (int) tid < args.inner) {
            float tot_g = 0.0f;
            float tot_u = 0.0f;
            for (uint u = 0; u < sgcount; ++u) {
                tot_g += partial_g[tid * MOE_SHEXP_SIMDGROUPS + u];
                tot_u += partial_u[tid * MOE_SHEXP_SIMDGROUPS + u];
            }
            // kernel_moe_silu_mul's expression verbatim (silu_mul.metal):
            // candle's usilu, `x / (1 + exp(-x))`, then a separately rounded
            // multiply. The chain this replaces runs exactly these two lines.
            const float s = tot_g / (1 + exp(-tot_g));
            h[(size_t) tgid.y * (size_t) args.inner + (size_t) (row0 + (int) tid)] = s * tot_u;
        }
        return;
    }

    // The gate logit, `<x[t], ffn_gate_inp_shexp>`. Accumulated per thread while
    // the activation is still in registers, over the same block partition the
    // projections used.
    float acc = 0.0f;
    for (int p = 0; p < MOE_SHEXP_MAX_BLK_PER_THREAD; ++p) {
        const int blk = (int) tid + p * MOE_SHEXP_THREADS;
        if (blk < args.nblk) {
            const size_t base = (size_t) blk * QK8_0;
            for (int i = 0; i < QK8_0; ++i) {
                acc += gate_inp[base + (size_t) i] * xv[p * QK8_0 + i];
            }
        }
    }
    const float lane_sum = simd_sum(acc);
    if (lane == 0) {
        partial_g[sgid] = lane_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint u = 0; u < sgcount; ++u) {
            total += partial_g[u];
        }
        logit[tgid.y] = total;
    }
}

// Threads per threadgroup in kernel_moe_epilogue_shexp: ONE simdgroup, so both
// of its reductions stay inside the single `simd_sum` that
// kernel_moe_epilogue's contract is built on. Mirrored by dispatch.rs
// MOE_SHEXP_EPILOGUE_THREADS.
#define MOE_SHEXP_EPILOGUE_THREADS 32

// Matches dispatch.rs MoeEpilogueShexpArgs (#[repr(C)]).
typedef struct {
    int32_t top_k;
    int32_t n_out;
    int32_t inner;
    int32_t nblk_inner; // inner / QK8_0, the q8_0 blocks in one down_shexp row
} moe_epilogue_shexp_args;

// dst[s, c] = (Σ_k down[s, k, c] * w[s, k])
//           + (Σ_j down_shexp[c, j] * h[s, j]) * sigmoid(gate[s]).
//
// kernel_moe_epilogue with the shared expert's DOWN projection folded in: the
// `shexp` operand it read as a materialized `[seq, n_out]` tensor is computed
// here instead, one row of `ffn_down_shexp` per threadgroup against
// kernel_moe_shexp_gate_up's bottleneck `h`. The grid is unchanged — one
// threadgroup per output element — so the plane is read exactly once per token
// across a grid that already existed, and nothing is allocated for it.
//
// Widened from `next_pow2(top_k/2)` threads to a full simdgroup, because the
// down row needs `inner/32` blocks folded where the routed combine needs only
// `top_k` terms; both reductions are one `simd_sum` over the same 32 lanes,
// each in its own accumulator. That widening is why this kernel is bounded and
// not bitwise against kernel_moe_epilogue, which is left untouched as the
// strict tier's anchor.
kernel void kernel_moe_epilogue_shexp(
        constant moe_epilogue_shexp_args & args  [[buffer(0)]],
        device const float * down                [[buffer(1)]],
        device const float * w                   [[buffer(2)]],
        device const float * h                   [[buffer(3)]],
        device const moe_block_q8_0 * down_shexp [[buffer(4)]],
        device const float * gate                [[buffer(5)]],
        device       float * dst                 [[buffer(6)]],
        uint lane   [[thread_index_in_threadgroup]],
        uint dst_id [[threadgroup_position_in_grid]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    const int top_k = args.top_k;
    const int n_out = args.n_out;
    const int did = (int) dst_id;
    const int s = did / n_out;
    const int c = did % n_out;
    const int down_base = s * top_k * n_out + c;
    const int sk_base = s * top_k;

    // The routed combine, in its own accumulator — kernel_moe_epilogue's loop
    // over a wider partition.
    float routed = 0.0f;
    for (int k = (int) lane; k < top_k; k += MOE_SHEXP_EPILOGUE_THREADS) {
        float d = down[down_base + k * n_out];
        float ww = w[sk_base + k];
        float r3 = d * ww;
        routed = routed + r3;
    }

    // The shared expert's down projection for this output channel, in a second
    // accumulator. The loop bound is the compile-time maximum with a runtime
    // guard inside, so the whole walk unrolls.
    device const moe_block_q8_0 * drow =
        down_shexp + (size_t) c * (size_t) args.nblk_inner;
    const size_t hbase = (size_t) s * (size_t) args.inner;
    float shexp = 0.0f;
    for (int p = 0; p < MOE_SHEXP_MAX_BLK_PER_LANE; ++p) {
        const int b = (int) lane + p * MOE_SHEXP_EPILOGUE_THREADS;
        if (b < args.nblk_inner) {
            device const moe_block_q8_0 * blk = drow + b;
            // packed_char4, not char4: the quants start two bytes into a
            // 34-byte block.
            device const packed_char4 * q4 = (device const packed_char4 *) blk->qs;
            const size_t hb = hbase + (size_t) b * QK8_0;
            float sumq = 0.0f;
            for (short i = 0; i < QK8_0 / 4; ++i) {
                const packed_char4 v = q4[i];
                sumq += (float) v.x * h[hb + (size_t) (4 * i + 0)];
                sumq += (float) v.y * h[hb + (size_t) (4 * i + 1)];
                sumq += (float) v.z * h[hb + (size_t) (4 * i + 2)];
                sumq += (float) v.w * h[hb + (size_t) (4 * i + 3)];
            }
            shexp += sumq * (float) blk->d;
        }
    }

    routed = simd_sum(routed);
    shexp = simd_sum(shexp);
    if (lane == 0) {
        // candle's usigmoid: recip(1 + exp(-x)) — a true divide, no fma. The
        // same expression kernel_moe_epilogue applies to the same raw logit.
        const float g = gate[s];
        const float sig = 1.0f / (1.0f + exp(-g));
        // candle's bmul, then badd in the block's order: routed + shared.
        const float shared = shexp * sig;
        dst[did] = routed + shared;
    }
}
