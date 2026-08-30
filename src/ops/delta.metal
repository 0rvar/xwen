// Vendored fused gated-DeltaNet kernels — the linear-attention layers of
// src/linear_attn.rs, whose composed-candle form (the frozen reference scan)
// spends ~65 Metal dispatches per decoded token on a layer that contains only
// seven matmuls, and one dispatch PER TIMESTEP PER LAYER during prefill. These
// four kernels collapse the glue:
//
//   kernel_delta_conv    depthwise causal conv + silu + next conv window, reading
//                        the carried window and the fresh qkv rows as two
//                        buffers (no concatenation, no slice_set state write).
//   kernel_delta_ba      beta = sigmoid(b_raw) and g = ssm_a * softplus(a_raw +
//                        dt_bias) from ONE fused [hidden, 2*v_heads] projection.
//   kernel_delta_ba_fused_t{1,4}
//                        that same head with the projection itself folded in:
//                        one dispatch reads x and the [hidden, 2*v_heads]
//                        weight and writes beta and g. Small n only (decode and
//                        short verify chunks); above the host's threshold the
//                        candle gemm's weight reuse wins. XWEN_DELTA_BA_CLASSIC
//                        reverts to the gemv + kernel_delta_ba pair.
//   kernel_delta_scan    the whole delta-rule recurrence over T timesteps, with
//                        the head's state slice resident in registers for the
//                        entire scan: one dispatch per layer per chunk. It folds
//                        the q/k L2 clamp-norm into its own load stage.
//   kernel_delta_scan_decode  the same recurrence for ONE token, the shape a
//                        decode step could take: no timestep loop, the state
//                        moved as float4, the row-slice folds done inside a
//                        simdgroup. OPT-IN behind XWEN_DELTA_DECODE_KERNEL and
//                        a measured wash; kernel_delta_scan runs decode by
//                        default.
//   kernel_delta_gnorm   the gated output RMSNorm (rms -> ssm_norm.weight ->
//                        silu(z)), with kernel_delta_gnorm_sigmoid its
//                        sigmoid(z) sibling for the qwen4exp graph — one
//                        templated body, the arm chosen by name at dispatch.
//
// A layer therefore costs eight dispatches at any sequence length. The
// kill-switch back to the reference scan is XWEN_DELTA_CLASSIC.
//
// The decode scan is a WASH against the general one end to end, so it is
// OPT-IN (XWEN_DELTA_DECODE_KERNEL) and kept for the measurement, not for a
// speedup — docs/decisions.md, "A decode-specialized scan kernel is a WASH".
//
// Two further kernels are not on the shipped path and exist as MEASURED
// ARTIFACTS of a refuted direction (docs/decisions.md, "The DeltaNet scan
// decomposition"): kernel_delta_scan_v2, llama.cpp's Metal decomposition of the
// same recurrence, and kernel_delta_l2norm, the hoisted q/k norm it needs.
// XWEN_DELTA_SCAN_V2 selects the pair.
//
// ROUNDING. TWO of this file's kernels are BIT-IDENTICAL to the candle chain
// they replace, and pin FP contraction and reassociation off at block scope
// (`#pragma clang fp contract(off)` / `reassociate(off)`):
//   - kernel_delta_conv reproduces the reference's tap chain exactly: the first
//     tap is a bare product and each later tap is a separate f32 multiply
//     followed by a separate f32 add (the reference materializes one tensor per
//     tap), then candle's usilu `x / (1 + exp(-x))`.
//   - kernel_delta_ba reproduces candle's usigmoid `1/(1 + exp(-x))` and the
//     stable softplus chain of linear_attn.rs (abs, add, affine-as-fma, exp,
//     affine-as-fma, log, add) with the same rounding boundaries, then one
//     multiply by ssm_a.
// The rest are BOUNDED, not bitwise, because a reduction the reference runs in
// one order is partitioned here:
//   - kernel_delta_gnorm's arithmetic matches the reference's ops one for one,
//     but the 128-term sum-of-squares reduction reassociates (hardware simd_sum
//     instead of candle's reduce partition). Its test grades at 2e-6.
//   - kernel_delta_l2norm is the same story for the q/k norm: the ops match
//     `linear_attn::l2_norm` one for one (sum of squares, sqrt, floor at eps,
//     divide) and only the 128-term sum reassociates. Its test grades at 2e-6.
//   - kernel_delta_scan_decode is the same story at seq == 1: same per-thread
//     arithmetic in the same order as kernel_delta_scan, with the cross-thread
//     fold reassociated once more (a simd_shuffle_xor butterfly over the row
//     slices, then the simdgroup partials). Graded at 1e-5 against both the
//     general kernel and the reference.
//   - kernel_delta_scan partitions the k- and q-contractions across threads and
//     folds them through threadgroup memory, where the reference runs a candle
//     gemm. It deliberately does NOT carry the fp pragmas — its two inner loops
//     are the entire prefill cost and must be free to contract into fma.
//
// `#pragma METAL fp math_mode(fast)` pins the library's math MODE at the source
// level to what nil compile options resolve to today (candle compiles its own
// kernels with an explicit MTLMathMode::Fast). That is all it pins: the
// separate fast-vs-precise math-FUNCTION compile option is left at the OS
// default (pipelines.rs hands this source no MTLCompileOptions), so what
// actually guarantees the exp/log/silu intrinsics still round like candle's is
// the pair of on-device bitwise tests — conv_matches_reference_bitwise and
// ba_matches_reference_bitwise fail loudly the moment a toolchain or OS shift
// moves them.
//
// A SEPARATE library from the other vendored sources (own runtime compile via
// src/ops/pipelines.rs, no Metal-4 dependency).

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp math_mode(fast)

// Matches dispatch.rs DeltaConvArgs (#[repr(C)]).
typedef struct {
    int32_t seq;
    int32_t conv_dim;
    int32_t taps;
    int32_t tail; // taps - 1, the carried window depth
} delta_conv_args;

// Causal depthwise conv over the fused qkv stream, silu'd, plus the window the
// next call starts from.
//
// The logical stream is the carried window followed by this chunk's rows:
//   stream[u] = u < tail ? state[u] : qkv[u - tail]
// so output row t reads taps stream[t .. t+taps-1] (the last tap is token t
// itself) and the next window is stream[seq .. seq+tail-1]. Reading `state` and
// `qkv` as two buffers is what removes the reference's concatenation; writing
// `nstate` here is what removes its zeros_like+slice_set materialization.
//
// One thread per output element. The `tail` window rows are claimed by the
// threads of column `c` whose row index is congruent to the window row modulo
// `seq`, so a single-token step (seq == 1) has thread t == 0 write all of them
// and a chunk of seq >= tail has one thread per row.
kernel void kernel_delta_conv(
        constant delta_conv_args & args [[buffer(0)]],
        device const float * state      [[buffer(1)]],
        device const float * qkv        [[buffer(2)]],
        device const float * w          [[buffer(3)]],
        device       float * dst        [[buffer(4)]],
        device       float * nstate     [[buffer(5)]],
        uint tid [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    const int n = args.seq * args.conv_dim;
    // Unsigned compare (the silu_mul.metal idiom): a signed guard lets a stray
    // grid thread at tid == 2^31 wrap negative and slip through to an OOB write.
    if (tid >= (uint) n) {
        return;
    }
    const int c = (int) tid % args.conv_dim;
    const int t = (int) tid / args.conv_dim;

    float acc = 0.0f;
    for (int j = 0; j < args.taps; ++j) {
        const int u = t + j;
        const float x = (u < args.tail) ? state[u * args.conv_dim + c]
                                        : qkv[(u - args.tail) * args.conv_dim + c];
        const float p = x * w[j * args.conv_dim + c];
        acc = (j == 0) ? p : (acc + p);
    }
    // candle's usilu, verbatim.
    dst[tid] = acc / (1 + exp(-acc));

    for (int i = t; i < args.tail; i += args.seq) {
        const int u = args.seq + i;
        nstate[i * args.conv_dim + c] = (u < args.tail)
                                            ? state[u * args.conv_dim + c]
                                            : qkv[(u - args.tail) * args.conv_dim + c];
    }
}

// Matches dispatch.rs DeltaBaArgs (#[repr(C)]).
typedef struct {
    int32_t seq;
    int32_t v_heads;
} delta_ba_args;

// ===========================================================================
// The beta/decay head. Two entry points over ONE epilogue:
//   kernel_delta_ba        the epilogue alone, over a projection candle
//                          already computed (prefill, and the kill switch).
//   kernel_delta_ba_fused  the [n, hidden] x [hidden, 2*v_heads] projection
//                          AND the epilogue in one dispatch (small n).
// The two helpers below are that shared epilogue: the arithmetic that has to
// round like candle's exists once. Both pin FP contraction and reassociation
// off and keep the reference's per-op rounding boundaries, which is what makes
// kernel_delta_ba bit-identical to the candle chain
// (ba_matches_reference_bitwise); the fused kernel differs from it only in the
// dot product it computes for itself.
// ===========================================================================

// candle's usigmoid: recip(1 + exp(-x)).
static inline float delta_ba_beta(float b_raw) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    return 1.0f / (1 + exp(-b_raw));
}

// The LOG decay `a * softplus(a_raw + dt_bias)`, softplus in the stable
// relu + ln(1 + exp(-|x|)) form with the reference's per-op rounding
// boundaries: candle's affine kernel is an explicit fma, its badd/bmul single
// f32 operations. `a` is `ssm_a`, pre-baked as -exp(A_log), so the result is
// <= 0 and the decay the scan exponentiates lands in (0, 1].
static inline float delta_ba_logdecay(float a_raw, float dt_bias, float a) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    const float x = a_raw + dt_bias;
    const float ax = abs(x);
    const float sum = x + ax;
    const float relu = fma(sum, 0.5f, 0.0f);
    const float e = exp(-ax);
    const float one_plus = fma(e, 1.0f, 1.0f);
    const float tail = log(one_plus);
    const float sp = relu + tail;
    return sp * a;
}

// beta and the log-decay from one fused projection. `ba` is [seq, 2*v_heads]:
// the beta logits occupy the first v_heads columns of a row and the alpha
// logits the second v_heads — the layout produced by concatenating ssm_beta and
// ssm_alpha into a single [hidden, 2*v_heads] weight at load time, which is why
// one gemv now does the work of two.
//
// `g` is emitted as the LOG decay, not exp(g): the scan kernel exponentiates it
// per head per timestep, which folds the reference's separate exp pass away.
// `ssm_a` arrives pre-baked as -exp(A_log), so g <= 0 and the decay lands in
// (0, 1]. One thread per (token, head).
kernel void kernel_delta_ba(
        constant delta_ba_args & args [[buffer(0)]],
        device const float * ba       [[buffer(1)]],
        device const float * ssm_a    [[buffer(2)]],
        device const float * dt_bias  [[buffer(3)]],
        device       float * beta     [[buffer(4)]],
        device       float * g        [[buffer(5)]],
        uint tid [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    const int n = args.seq * args.v_heads;
    if (tid >= (uint) n) {
        return;
    }
    const int h = (int) tid % args.v_heads;
    const int t = (int) tid / args.v_heads;
    const int base = t * 2 * args.v_heads;

    beta[tid] = delta_ba_beta(ba[base + h]);
    g[tid] = delta_ba_logdecay(ba[base + args.v_heads + h], dt_bias[h], ssm_a[h]);
}

// Matches dispatch.rs DeltaBaFusedArgs (#[repr(C)]).
typedef struct {
    int32_t seq;
    int32_t hidden;
    int32_t v_heads;
} delta_ba_fused_args;

// The threadgroup shape of kernel_delta_ba_fused, mirrored by dispatch.rs
// (DELTA_BA_* there) which sizes the grid from it. COLS x ROWS threads:
// DELTA_BA_COLS output columns wide, DELTA_BA_ROWS deep in row chunks of the
// hidden dim, each thread holding one partial dot product per column per token
// of its tile.
#define DELTA_BA_COLS 8
#define DELTA_BA_ROWS 128
#define DELTA_BA_TOKS 4

static_assert(DELTA_BA_COLS * DELTA_BA_ROWS <= 1024,
              "a threadgroup is at most 1024 threads");
static_assert((DELTA_BA_ROWS & (DELTA_BA_ROWS - 1)) == 0,
              "the partial-sum tree halves DELTA_BA_ROWS to 1");
static_assert(DELTA_BA_TOKS * DELTA_BA_ROWS * DELTA_BA_COLS * 4 <= 32768,
              "the partial buffer must fit threadgroup memory");

// The beta|alpha PROJECTION and its epilogue in ONE dispatch: reads x
// [n, hidden] and the concatenated [hidden, 2*v_heads] weight and writes beta
// and g directly, so the candle gemv that produced `ba` for kernel_delta_ba
// disappears along with its dispatch. At decode that gemv was 10% of the whole
// GDN mixer for a matmul of 96 dot products (docs/log.md).
//
// The partition is over OUTPUT COLUMNS, which is the only split of this shape
// that needs no cross-threadgroup reduction: column c of the weight is read by
// exactly one threadgroup, and beta (c < v_heads) and g (c >= v_heads) are
// independent per column. Inside a threadgroup, lane j of a row chunk reads
// w[i][colbase + j], so the DELTA_BA_COLS lanes of one chunk read one
// contiguous run of the row-major weight; the DELTA_BA_ROWS chunks then fold
// their partials through threadgroup memory in a tree.
//
// TOKS tokens are tiled into one threadgroup so a multi-token chunk reads the
// weight once per tile rather than once per token — the whole reason this shape
// is confined to small n. Above the host's threshold the weight stops fitting
// the reuse pattern and candle's gemm wins; see dispatch.rs.
//
// Deliberately NOT bit-identical to kernel_delta_ba over the same input: the
// dot product is summed as DELTA_BA_ROWS partials folded in a tree, where
// candle's gemv sums in its own order. The epilogue is the same helpers, so
// the only difference is that reassociation (graded at 2e-6 in delta.rs).
template <int TOKS>
static inline void delta_ba_fused_body(
        constant delta_ba_fused_args & args,
        device const float * x,
        device const float * w,
        device const float * ssm_a,
        device const float * dt_bias,
        device       float * beta,
        device       float * g,
        threadgroup  float * part,
        uint2 tgid,
        uint tid) {
    const int two_vh = 2 * args.v_heads;
    const int j = (int) (tid % DELTA_BA_COLS);
    const int r = (int) (tid / DELTA_BA_COLS);
    const int col = (int) tgid.x * DELTA_BA_COLS + j;
    const int t0 = (int) tgid.y * TOKS;
    // At least one, because the grid has ceil(seq / TOKS) tiles in y.
    const int live = min(TOKS, args.seq - t0);

    // Rows past the tile's end clamp onto its last live token: the duplicate
    // reads are harmless and their accumulator is never written out, which
    // keeps `acc` a fully unrolled register array with no branch in the loop.
    const device float * xp[TOKS];
    for (int u = 0; u < TOKS; ++u) {
        xp[u] = x + (size_t) (t0 + min(u, live - 1)) * args.hidden;
    }

    float acc[TOKS];
    for (int u = 0; u < TOKS; ++u) {
        acc[u] = 0.0f;
    }
    if (col < two_vh) {
        device const float * wp = w + col;
        for (int i = r; i < args.hidden; i += DELTA_BA_ROWS) {
            const float wv = wp[(size_t) i * two_vh];
            for (int u = 0; u < TOKS; ++u) {
                acc[u] = fma(xp[u][i], wv, acc[u]);
            }
        }
    }
    for (int u = 0; u < TOKS; ++u) {
        part[(u * DELTA_BA_ROWS + r) * DELTA_BA_COLS + j] = acc[u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int s = DELTA_BA_ROWS / 2; s > 0; s >>= 1) {
        if (r < s) {
            for (int u = 0; u < TOKS; ++u) {
                part[(u * DELTA_BA_ROWS + r) * DELTA_BA_COLS + j] +=
                    part[(u * DELTA_BA_ROWS + r + s) * DELTA_BA_COLS + j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (r != 0 || col >= two_vh) {
        return;
    }
    const bool is_beta = col < args.v_heads;
    const int h = is_beta ? col : col - args.v_heads;
    for (int u = 0; u < live; ++u) {
        const float dot = part[(u * DELTA_BA_ROWS) * DELTA_BA_COLS + j];
        const int idx = (t0 + u) * args.v_heads + h;
        if (is_beta) {
            beta[idx] = delta_ba_beta(dot);
        } else {
            g[idx] = delta_ba_logdecay(dot, dt_bias[h], ssm_a[h]);
        }
    }
}

// The decode specialization: one token, so no tile clamping survives constant
// folding and the inner loop is one fma per weight load.
kernel void kernel_delta_ba_fused_t1(
        constant delta_ba_fused_args & args [[buffer(0)]],
        device const float * x              [[buffer(1)]],
        device const float * w              [[buffer(2)]],
        device const float * ssm_a          [[buffer(3)]],
        device const float * dt_bias        [[buffer(4)]],
        device       float * beta           [[buffer(5)]],
        device       float * g              [[buffer(6)]],
        uint2 tgid [[threadgroup_position_in_grid]],
        uint2 tid  [[thread_position_in_threadgroup]]) {
    threadgroup float part[1 * DELTA_BA_ROWS * DELTA_BA_COLS];
    delta_ba_fused_body<1>(args, x, w, ssm_a, dt_bias, beta, g, part, tgid, tid.x);
}

// The short-chunk specialization: DELTA_BA_TOKS tokens share one pass over the
// weight.
kernel void kernel_delta_ba_fused_t4(
        constant delta_ba_fused_args & args [[buffer(0)]],
        device const float * x              [[buffer(1)]],
        device const float * w              [[buffer(2)]],
        device const float * ssm_a          [[buffer(3)]],
        device const float * dt_bias        [[buffer(4)]],
        device       float * beta           [[buffer(5)]],
        device       float * g              [[buffer(6)]],
        uint2 tgid [[threadgroup_position_in_grid]],
        uint2 tid  [[thread_position_in_threadgroup]]) {
    threadgroup float part[DELTA_BA_TOKS * DELTA_BA_ROWS * DELTA_BA_COLS];
    delta_ba_fused_body<DELTA_BA_TOKS>(args, x, w, ssm_a, dt_bias, beta, g, part, tgid, tid.x);
}

// Matches dispatch.rs DeltaGnormArgs (#[repr(C)]).
typedef struct {
    int32_t v_heads;
    int32_t head_dim;
    float eps;
} delta_gnorm_args;

// Gated output RMSNorm: normalize over the head dim, scale by
// ssm_norm.weight, and only THEN multiply by the activated gate. The gate is
// outside the norm, so it does not affect the statistic.
//
// The activation is the checkpoint's: `silu(z)` on the qwen35/qwen35moe
// graphs, `sigmoid(z)` on qwen4exp (config::ZGate). One body, two kernels —
// the branch is a template parameter so each specialization compiles to the
// same straight-line arithmetic a hand-written kernel would, and the silu
// arm's instruction stream is unchanged by the sigmoid arm's existence.
//
// One threadgroup per (token, head), `head_dim` threads wide; the sum of
// squares folds through simd_sum and one threadgroup-memory pass.
//
// `partial` is passed in rather than declared here: MSL allows threadgroup
// variables only at a kernel function's own scope.
template <bool SIGMOID_GATE>
static inline void delta_gnorm_body(
        constant delta_gnorm_args & args,
        device const float * o,
        device const float * z,
        device const float * w,
        device       float * dst,
        threadgroup  float * partial,
        uint tgid,
        uint tid,
        uint sgid,
        uint lane,
        uint sgcount) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    const int d = (int) tid;
    const size_t idx = (size_t) tgid * args.head_dim + d;
    const float x = o[idx];

    const float lane_sum = simd_sum(x * x);
    if (lane == 0) {
        partial[sgid] = lane_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint u = 0; u < sgcount; ++u) {
        total += partial[u];
    }

    const float ms = total / (float) args.head_dim;
    const float den = sqrt(ms + args.eps);
    const float zv = z[idx];
    // candle's usilu / usigmoid on the gate, applied after the weight (see the
    // ordering test in linear_attn.rs).
    const float gate = SIGMOID_GATE ? (1 / (1 + exp(-zv))) : (zv / (1 + exp(-zv)));
    dst[idx] = ((x / den) * w[d]) * gate;
}

kernel void kernel_delta_gnorm(
        constant delta_gnorm_args & args [[buffer(0)]],
        device const float * o           [[buffer(1)]],
        device const float * z           [[buffer(2)]],
        device const float * w           [[buffer(3)]],
        device       float * dst         [[buffer(4)]],
        uint tgid [[threadgroup_position_in_grid]],
        uint tid  [[thread_position_in_threadgroup]],
        uint sgid [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sgcount [[simdgroups_per_threadgroup]]) {
    threadgroup float partial[32];
    delta_gnorm_body<false>(args, o, z, w, dst, partial, tgid, tid, sgid, lane, sgcount);
}

// The sigmoid-gated sibling (qwen4exp). Identical but for the activation.
kernel void kernel_delta_gnorm_sigmoid(
        constant delta_gnorm_args & args [[buffer(0)]],
        device const float * o           [[buffer(1)]],
        device const float * z           [[buffer(2)]],
        device const float * w           [[buffer(3)]],
        device       float * dst         [[buffer(4)]],
        uint tgid [[threadgroup_position_in_grid]],
        uint tid  [[thread_position_in_threadgroup]],
        uint sgid [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sgcount [[simdgroups_per_threadgroup]]) {
    threadgroup float partial[32];
    delta_gnorm_body<true>(args, o, z, w, dst, partial, tgid, tid, sgid, lane, sgcount);
}

// The scan kernels' fixed head dim. Both checkpoints run gated DeltaNet at 128
// (27B: 16 K-heads / 48 V-heads; 35B-A3B: 16 / 32), so the kernels are
// specialized to it and the host refuses any other head dim (falling back to
// the reference scan).
#define DELTA_D 128

// Matches dispatch.rs DeltaL2NormArgs (#[repr(C)]).
typedef struct {
    int32_t k_heads;
    int32_t conv_dim;
    float eps; // rms_norm_eps, the L2 norm FLOOR
} delta_l2norm_args;

// The q/k L2 clamp-norm, `x / max(||x||, eps)` with eps a FLOOR ON THE NORM
// rather than a term under the root (ggml_l2_norm's form).
//
// The conv output's leading `2 * k_heads * DELTA_D` columns are the q planes
// followed by the k planes, one plane per K-HEAD; everything after them is v,
// which is not normalized. This kernel walks exactly those leading columns and
// writes them normalized to a `[seq, 2 * k_heads * DELTA_D]` buffer in the same
// order, so the scan reads q and k from `qk` and v from `conv` and the tiled
// K-head broadcast stays a read-side index rather than a materialized tensor.
//
// Normalizing here rather than inside the scan is what keeps this work
// proportional to the K-HEADS: the scan's threadgroups outnumber the K-head
// planes by the V-head ratio times the value-column split, and each of them
// would otherwise recompute the same two norms on every timestep.
//
// One threadgroup per (token, plane), DELTA_D threads wide; the sum of squares
// folds through simd_sum and one threadgroup-memory pass, exactly as
// kernel_delta_gnorm's does.
kernel void kernel_delta_l2norm(
        constant delta_l2norm_args & args [[buffer(0)]],
        device const float * conv         [[buffer(1)]],
        device       float * dst          [[buffer(2)]],
        uint tgid [[threadgroup_position_in_grid]],
        uint tid  [[thread_position_in_threadgroup]],
        uint sgid [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sgcount [[simdgroups_per_threadgroup]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    threadgroup float partial[DELTA_D / 32];

    const int planes = 2 * args.k_heads;
    const int t = (int) tgid / planes;
    const int plane = (int) tgid % planes;
    const int d = (int) tid;

    const float x = conv[(size_t) t * args.conv_dim + (size_t) plane * DELTA_D + d];

    const float lane_sum = simd_sum(x * x);
    if (lane == 0) {
        partial[sgid] = lane_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint u = 0; u < sgcount; ++u) {
        total += partial[u];
    }

    dst[(size_t) tgid * DELTA_D + d] = x / max(sqrt(total), args.eps);
}

// The shipped scan's threadgroup geometry (see kernel_delta_scan below).
#define DELTA_TG_COLS 32  // value-dim columns owned by one threadgroup
#define DELTA_TG_ROWS 4   // key-dim slices the columns are split across
#define DELTA_S_SLICE 32  // DELTA_D / DELTA_TG_ROWS, state rows per thread
#define DELTA_COL_BLOCKS 4 // DELTA_D / DELTA_TG_COLS, threadgroups per head

// The four are not independent: the threadgroup is DELTA_D threads laid out as
// DELTA_TG_ROWS x DELTA_TG_COLS, each thread owns DELTA_S_SLICE state rows of
// one column, and DELTA_COL_BLOCKS threadgroups tile a head's value dim. The
// host sizes the grid from its own copies of these numbers (dispatch.rs
// DELTA_HEAD_DIM / DELTA_COL_BLOCKS, cross-checked by a test), so a value that
// drifted out of this relation would index outside a head's state slice.
static_assert(DELTA_TG_ROWS * DELTA_TG_COLS == DELTA_D,
              "the scan threadgroup must be exactly DELTA_D threads");
static_assert(DELTA_S_SLICE == DELTA_D / DELTA_TG_ROWS,
              "each thread owns DELTA_D / DELTA_TG_ROWS state rows");
static_assert(DELTA_COL_BLOCKS == DELTA_D / DELTA_TG_COLS,
              "DELTA_COL_BLOCKS threadgroups must tile a head's value dim exactly");

// Matches dispatch.rs DeltaScanArgs (#[repr(C)]).
typedef struct {
    int32_t seq;
    int32_t k_heads;
    int32_t v_heads;
    int32_t conv_dim;
    int32_t n_planes; // state snapshot planes in s_out, most-recent-first
    float scale; // 1 / sqrt(head_dim)
    float eps;   // rms_norm_eps, the L2 norm FLOOR
} delta_scan_args;

// The delta-rule recurrence, all T timesteps inside one dispatch.
//
// Per timestep, with S the head's [d_k, d_v] state (first axis contracts with k
// and q, second carries the value dimension):
//
//     S *= exp(g);  d = (v - k·S) * beta;  S += k (x) d;  o = q·S / sqrt(d_k)
//
// q and k are read straight out of the conv output with the TILED K-head
// mapping (V-head h reads K-head h % k_heads — ggml's plain repeat, the order
// the GGUF's permuted V-side weights expect), so the reference's materialized
// tile-and-broadcast disappears. Their L2 normalization is folded into this
// load stage in ggml_l2_norm's clamp form, `x / max(||x||, eps)` with eps a
// FLOOR ON THE NORM rather than a term under the root.
//
// Decomposition: value-dim columns are fully independent of each other (sk[j],
// d[j], the rank-1 update of column j and o[j] all touch only column j), so a
// threadgroup owns one head and DELTA_TG_COLS of its columns, and the only
// cross-thread folds are the two key-dim contractions. Thread (r, jl) holds
// rows [r*DELTA_S_SLICE, (r+1)*DELTA_S_SLICE) of column j0+jl IN REGISTERS for
// the whole scan — the state is read once and written once no matter how long
// the chunk is, which is what makes prefill one dispatch per layer.
//
// Lane order also makes both state passes coalesced: consecutive threads within
// a simdgroup share `r` and take consecutive `j`, so each state row access is a
// contiguous 32-float run.
//
// Staging q and k through threadgroup memory is also what keeps the q/k reads
// proportional to the THREADGROUPS rather than to the individual state columns.
// kernel_delta_scan_v2 below gives that up, and it is the measured reason this
// decomposition is the one that ships.
//
// s_out holds args.n_planes state planes MOST-RECENT-FIRST: plane p is the state
// after token seq-1-p, so plane 0 is always the final state and n_planes = 1 is
// the plain scan. A rollback trail asks for n_planes = seq and reads plane
// seq-1-t for token t. The ordering mirrors llama.cpp's snapshot slots
// (ggml/src/ggml-metal/ggml-metal.metal, `target_slot = n_tokens - 1 - t`).
kernel void kernel_delta_scan(
        constant delta_scan_args & args [[buffer(0)]],
        device const float * conv       [[buffer(1)]],
        device const float * beta       [[buffer(2)]],
        device const float * g          [[buffer(3)]],
        device const float * s_in       [[buffer(4)]],
        device       float * out        [[buffer(5)]],
        device       float * s_out      [[buffer(6)]],
        uint tgid [[threadgroup_position_in_grid]],
        uint tid  [[thread_position_in_threadgroup]],
        uint sgid [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float qn[DELTA_D];
    threadgroup float kn[DELTA_D];
    threadgroup float part[DELTA_TG_ROWS][DELTA_TG_COLS];
    threadgroup float red[2][DELTA_D / 32];

    const int h = (int) tgid / DELTA_COL_BLOCKS;
    const int jl = (int) tid % DELTA_TG_COLS;
    const int j = ((int) tgid % DELTA_COL_BLOCKS) * DELTA_TG_COLS + jl;
    const int r = (int) tid / DELTA_TG_COLS;
    const int i0 = r * DELTA_S_SLICE;

    const int k_dim = args.k_heads * DELTA_D;
    const int kh = h % args.k_heads; // TILED, not interleaved
    const int q_off = kh * DELTA_D;
    const int k_off = k_dim + kh * DELTA_D;
    const int v_off = 2 * k_dim + h * DELTA_D;

    // The state slice this thread owns, resident for the whole scan.
    const size_t s_base = (size_t) h * DELTA_D * DELTA_D + j;
    float s[DELTA_S_SLICE];
#pragma unroll
    for (int a = 0; a < DELTA_S_SLICE; ++a) {
        s[a] = s_in[s_base + (size_t) (i0 + a) * DELTA_D];
    }

    for (int t = 0; t < args.seq; ++t) {
        device const float * row = conv + (size_t) t * args.conv_dim;

        // q/k load + L2 clamp-norm. One thread per head dim (DELTA_D threads
        // per threadgroup), so `tid` is the dim index here.
        const int d = (int) tid;
        const float qr = row[q_off + d];
        const float kr = row[k_off + d];
        const float q_lane = simd_sum(qr * qr);
        const float k_lane = simd_sum(kr * kr);
        if (lane == 0) {
            red[0][sgid] = q_lane;
            red[1][sgid] = k_lane;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (int u = 0; u < DELTA_D / 32; ++u) {
            q_sum += red[0][u];
            k_sum += red[1][u];
        }
        // Every thread has read `red`, and the previous timestep's readers of
        // qn/kn are past this point, before either is rewritten.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        qn[d] = qr / max(sqrt(q_sum), args.eps);
        kn[d] = kr / max(sqrt(k_sum), args.eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Decay the state, then contract it with k over the key dim.
        const float dec = exp(g[t * args.v_heads + h]);
        float sk = 0.0f;
#pragma unroll
        for (int a = 0; a < DELTA_S_SLICE; ++a) {
            s[a] *= dec;
            sk += s[a] * kn[i0 + a];
        }
        part[r][jl] = sk;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sk_col = 0.0f;
        for (int u = 0; u < DELTA_TG_ROWS; ++u) {
            sk_col += part[u][jl];
        }
        // All reads of `part` complete before the second fold overwrites it.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float delta = (row[v_off + j] - sk_col) * beta[t * args.v_heads + h];

        // Rank-1 update, then read the UPDATED state out with q.
        float ov = 0.0f;
#pragma unroll
        for (int a = 0; a < DELTA_S_SLICE; ++a) {
            s[a] += kn[i0 + a] * delta;
            ov += qn[i0 + a] * s[a];
        }
        part[r][jl] = ov;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float o_col = 0.0f;
        for (int u = 0; u < DELTA_TG_ROWS; ++u) {
            o_col += part[u][jl];
        }
        if (r == 0) {
            out[((size_t) t * args.v_heads + h) * DELTA_D + j] = o_col * args.scale;
        }

        // Trail plane for this timestep. Plane 0 is the after-loop store below,
        // so only the older planes are written here — at n_planes = 1 the loop
        // costs one compare per timestep and nothing else.
        const int target_slot = args.seq - 1 - t;
        if (target_slot >= 1 && target_slot < args.n_planes) {
            device float * s_plane =
                s_out + (size_t) target_slot * (size_t) args.v_heads * DELTA_D * DELTA_D;
#pragma unroll
            for (int a = 0; a < DELTA_S_SLICE; ++a) {
                s_plane[s_base + (size_t) (i0 + a) * DELTA_D] = s[a];
            }
        }
    }

#pragma unroll
    for (int a = 0; a < DELTA_S_SLICE; ++a) {
        s_out[s_base + (size_t) (i0 + a) * DELTA_D] = s[a];
    }
}

// The decode scan's threadgroup geometry (see kernel_delta_scan_decode below).
// A threadgroup is still DELTA_D threads owning DELTA_DEC_TG_COLS value
// columns of one head, but the columns are handed out FOUR AT A TIME so every
// state touch is a float4: a thread owns DELTA_DEC_SLICE rows of one float4
// column group, and the DELTA_DEC_ROWS threads sharing a group fold through
// simd_shuffle_xor first and threadgroup memory only across simdgroups.
#define DELTA_DEC_VEC 4        // value columns per thread — one float4
#define DELTA_DEC_LANES 8      // float4 column groups across a threadgroup
#define DELTA_DEC_ROWS 16      // key-dim slices those groups are split across
#define DELTA_DEC_SLICE 8      // DELTA_D / DELTA_DEC_ROWS, state rows per thread
#define DELTA_DEC_TG_COLS 32   // DELTA_DEC_VEC * DELTA_DEC_LANES
#define DELTA_DEC_COL_BLOCKS 4 // DELTA_D / DELTA_DEC_TG_COLS, threadgroups per head
#define DELTA_DEC_SGS 4        // DELTA_D / 32, simdgroups per threadgroup

// The seven are not independent: the threadgroup is DELTA_D threads laid out as
// DELTA_DEC_ROWS x DELTA_DEC_LANES, each thread owns DELTA_DEC_SLICE rows of
// DELTA_DEC_VEC columns, and DELTA_DEC_COL_BLOCKS threadgroups tile a head's
// value dim. The host sizes the grid from its own copies (dispatch.rs
// DELTA_HEAD_DIM / DELTA_DEC_COL_BLOCKS, cross-checked by a test), so a value
// that drifted out of this relation would index outside a head's state slice.
static_assert(DELTA_DEC_VEC * DELTA_DEC_LANES == DELTA_DEC_TG_COLS,
              "a threadgroup's float4 groups must tile its columns exactly");
static_assert(DELTA_DEC_ROWS * DELTA_DEC_LANES == DELTA_D,
              "the decode threadgroup must be exactly DELTA_D threads");
static_assert(DELTA_DEC_SLICE == DELTA_D / DELTA_DEC_ROWS,
              "each thread owns DELTA_D / DELTA_DEC_ROWS state rows");
static_assert(DELTA_DEC_COL_BLOCKS == DELTA_D / DELTA_DEC_TG_COLS,
              "DELTA_DEC_COL_BLOCKS threadgroups must tile a head's value dim exactly");
static_assert(DELTA_DEC_SGS * 32 == DELTA_D, "DELTA_DEC_SGS simdgroups per threadgroup");
static_assert(32 % DELTA_DEC_LANES == 0,
              "a simdgroup must hold a whole number of column groups for the "
              "shuffle fold to reduce over row slices only");

// The delta-rule recurrence for ONE token — the decode step, hoisted out of
// kernel_delta_scan's timestep loop. Same math, same operands, and the same
// most-recent-first `s_out` contract at the single plane a one-token chunk can
// name (n_planes is 1 by construction, so plane 0 is the whole trail and a
// rollback restores it unchanged).
//
// Why a second kernel: at seq == 1 the general scan's loop body IS the kernel,
// so everything it amortizes over a chunk it pays in full — a barrier-separated
// two-phase fold through part[][], the q/k staging, and 32 scalar state loads
// per thread at a 512-byte stride. This one drops the loop and rebuilds the
// same decomposition around the memory: the state is read once and written once
// as float4 (each load instruction covers 512 contiguous bytes of a state row
// instead of 128), the row-slice fold happens inside a simdgroup with
// simd_shuffle_xor, and threadgroup memory carries only the DELTA_DEC_SGS
// per-simdgroup partials. The q/k L2 clamp-norm is computed once per
// threadgroup, one thread per head dim, and broadcast through `qn`/`kn`.
//
// Arithmetic is the general kernel's, in the same order per thread; only the
// cross-thread fold reassociates, so this is bounded against the reference in
// exactly the class kernel_delta_scan already sits in (docs/parity.md).
kernel void kernel_delta_scan_decode(
        constant delta_scan_args & args [[buffer(0)]],
        device const float  * conv  [[buffer(1)]],
        device const float  * beta  [[buffer(2)]],
        device const float  * g     [[buffer(3)]],
        device const float4 * s_in  [[buffer(4)]],
        device       float  * out   [[buffer(5)]],
        device       float4 * s_out [[buffer(6)]],
        uint tgid [[threadgroup_position_in_grid]],
        uint tid  [[thread_position_in_threadgroup]],
        uint sgid [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float qn[DELTA_D];
    threadgroup float kn[DELTA_D];
    threadgroup float red[2][DELTA_D / 32];
    // Two fold buffers, not one reused: a third barrier to separate the k fold
    // from the q fold would cost more than 512 bytes of threadgroup memory.
    threadgroup float4 part_k[DELTA_DEC_SGS][DELTA_DEC_LANES];
    threadgroup float4 part_o[DELTA_DEC_SGS][DELTA_DEC_LANES];

    const int h = (int) tgid / DELTA_DEC_COL_BLOCKS;
    const int c = (int) tid % DELTA_DEC_LANES; // float4 column group
    const int r = (int) tid / DELTA_DEC_LANES; // row slice
    const int i0 = r * DELTA_DEC_SLICE;
    const int j0 = ((int) tgid % DELTA_DEC_COL_BLOCKS) * DELTA_DEC_TG_COLS + c * DELTA_DEC_VEC;

    const int k_dim = args.k_heads * DELTA_D;
    const int kh = h % args.k_heads; // TILED, not interleaved
    const int v_off = 2 * k_dim + h * DELTA_D;

    // The state slice this thread owns, in float4s: DELTA_D / DELTA_DEC_VEC of
    // them to a state row, DELTA_D rows to a head.
    const int s4_row = DELTA_D / DELTA_DEC_VEC;
    const size_t s4_base = (size_t) h * DELTA_D * s4_row + (size_t) (j0 / DELTA_DEC_VEC);
    float4 s[DELTA_DEC_SLICE];
#pragma unroll
    for (int a = 0; a < DELTA_DEC_SLICE; ++a) {
        s[a] = s_in[s4_base + (size_t) (i0 + a) * s4_row];
    }

    // q/k load + L2 clamp-norm, one thread per head dim, published to the whole
    // threadgroup. Once per dispatch, where the general kernel does this per
    // timestep.
    const int d = (int) tid;
    const float qr = conv[kh * DELTA_D + d];
    const float kr = conv[k_dim + kh * DELTA_D + d];
    const float q_lane = simd_sum(qr * qr);
    const float k_lane = simd_sum(kr * kr);
    if (lane == 0) {
        red[0][sgid] = q_lane;
        red[1][sgid] = k_lane;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (int u = 0; u < DELTA_D / 32; ++u) {
        q_sum += red[0][u];
        k_sum += red[1][u];
    }
    qn[d] = qr / max(sqrt(q_sum), args.eps);
    kn[d] = kr / max(sqrt(k_sum), args.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Decay the state, then contract it with k over the key dim. The row slices
    // a simdgroup holds fold with a shuffle butterfly over the lane bits that
    // carry `r`; only the DELTA_DEC_SGS simdgroup partials reach memory.
    const float dec = exp(g[h]);
    float4 sk = 0.0f;
#pragma unroll
    for (int a = 0; a < DELTA_DEC_SLICE; ++a) {
        s[a] *= dec;
        sk += s[a] * kn[i0 + a];
    }
    for (uint m = DELTA_DEC_LANES; m < 32; m <<= 1) {
        sk += simd_shuffle_xor(sk, m);
    }
    if (lane < DELTA_DEC_LANES) {
        part_k[sgid][lane] = sk;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 sk_col = part_k[0][c];
    for (int u = 1; u < DELTA_DEC_SGS; ++u) {
        sk_col += part_k[u][c];
    }

    // v is read as scalars: the conv buffer may start at any f32 offset, and
    // four loads per thread is nothing next to the state.
    const float4 v = float4(conv[v_off + j0], conv[v_off + j0 + 1],
                            conv[v_off + j0 + 2], conv[v_off + j0 + 3]);
    const float4 delta = (v - sk_col) * beta[h];

    // Rank-1 update, then read the UPDATED state out with q.
    float4 ov = 0.0f;
#pragma unroll
    for (int a = 0; a < DELTA_DEC_SLICE; ++a) {
        s[a] += kn[i0 + a] * delta;
        ov += qn[i0 + a] * s[a];
    }
    for (uint m = DELTA_DEC_LANES; m < 32; m <<= 1) {
        ov += simd_shuffle_xor(ov, m);
    }
    if (lane < DELTA_DEC_LANES) {
        part_o[sgid][lane] = ov;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
    for (int a = 0; a < DELTA_DEC_SLICE; ++a) {
        s_out[s4_base + (size_t) (i0 + a) * s4_row] = s[a];
    }

    if (r == 0) {
        float4 o_col = part_o[0][c];
        for (int u = 1; u < DELTA_DEC_SGS; ++u) {
            o_col += part_o[u][c];
        }
        o_col *= args.scale;
        device float * o_ptr = out + (size_t) h * DELTA_D + j0;
        o_ptr[0] = o_col.x;
        o_ptr[1] = o_col.y;
        o_ptr[2] = o_col.z;
        o_ptr[3] = o_col.w;
    }
}

// The v2 scan's geometry. A SIMDGROUP, not a threadgroup, is the unit of
// ownership: one simdgroup owns one value column j of one head's state for the
// whole scan, holding the column's DELTA_D key entries DELTA_V2_KPL to a lane.
// Both key-dim contractions are then simd_sum reductions inside that
// simdgroup — no threadgroup memory, and no barrier anywhere in the timestep
// loop. DELTA_V2_SGS simdgroups share a threadgroup purely to reach a sensible
// launch width; they never talk to each other.
#define DELTA_V2_KPL 4      // DELTA_D / 32, key entries per lane
#define DELTA_V2_SGS 4      // simdgroups (= value columns) per threadgroup
#define DELTA_V2_COL_TGS 32 // DELTA_D / DELTA_V2_SGS, threadgroups per V-head

// The three are not independent: a lane's DELTA_V2_KPL entries times the 32
// lanes of a simdgroup must cover the key dim exactly, and the threadgroups of a
// head must tile its value dim exactly. The host sizes the grid and the
// threadgroup from its own copies (dispatch.rs DELTA_V2_SGS / DELTA_V2_COL_TGS,
// cross-checked by a test), so a value drifting out of these relations would
// leave part of the state unowned.
static_assert(DELTA_V2_KPL * 32 == DELTA_D,
              "a simdgroup's lanes must cover the key dim exactly");
static_assert(DELTA_V2_COL_TGS * DELTA_V2_SGS == DELTA_D,
              "a head's threadgroups must tile its value dim exactly");

// Matches dispatch.rs DeltaScanV2Args (#[repr(C)]). No eps: this kernel reads q
// and k already normalized, from kernel_delta_l2norm.
typedef struct {
    int32_t seq;
    int32_t k_heads;
    int32_t v_heads;
    int32_t conv_dim;
    int32_t n_planes; // state snapshot planes in s_out, most-recent-first
    float scale; // 1 / sqrt(head_dim)
} delta_scan_v2_args;

// The same recurrence and the same inputs as kernel_delta_scan (bar the already
// normalized q and k), under a decomposition that hands each SIMDGROUP its own
// state value-column instead of splitting one head across a whole threadgroup.
// Selected by XWEN_DELTA_SCAN_V2; its doc comment in src/ops/mod.rs carries the
// measured reason it is not the default.
//
// Decomposition: value-dim columns are fully independent of each other (sk, d,
// the rank-1 update of column j and o[j] all touch only column j), so ONE
// SIMDGROUP owns column j of head h end to end, and both key-dim contractions
// collapse to simd_sum within it. Lane `l` holds state rows
// [l*DELTA_V2_KPL, (l+1)*DELTA_V2_KPL) of that column in registers for the whole
// scan: the state is read once and written once no matter how long the chunk is,
// the timestep loop touches no threadgroup memory, and nothing in it waits on a
// barrier. Each lane's q/k reads are DELTA_V2_KPL consecutive floats, and the
// lanes of a simdgroup cover a plane contiguously.
//
// The state's value axis is the fastest-varying one, so a column is strided in
// memory and the load and store below are not coalesced. That is a deliberate
// trade: it costs two strided passes per dispatch and buys a timestep loop with
// no cross-lane traffic at all.
kernel void kernel_delta_scan_v2(
        constant delta_scan_v2_args & args [[buffer(0)]],
        device const float * qk    [[buffer(1)]],
        device const float * conv  [[buffer(2)]],
        device const float * beta  [[buffer(3)]],
        device const float * g     [[buffer(4)]],
        device const float * s_in  [[buffer(5)]],
        device       float * out   [[buffer(6)]],
        device       float * s_out [[buffer(7)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]]) {
    const uint lane = tpitg.x;
    const int h = (int) tgpig.y;
    const int j = (int) (tgpig.x * DELTA_V2_SGS + tpitg.y);

    const int k_dim = args.k_heads * DELTA_D;
    const int qk_stride = 2 * k_dim;
    const int kh = h % args.k_heads; // TILED, not interleaved
    const int i0 = (int) lane * DELTA_V2_KPL;

    // The state column this simdgroup owns, DELTA_V2_KPL rows of it per lane,
    // resident for the whole scan.
    const size_t s_base =
        (size_t) h * DELTA_D * DELTA_D + (size_t) i0 * DELTA_D + (size_t) j;
    float s[DELTA_V2_KPL];
#pragma unroll
    for (int a = 0; a < DELTA_V2_KPL; ++a) {
        s[a] = s_in[s_base + (size_t) a * DELTA_D];
    }

    device const float * q_ptr = qk + (size_t) kh * DELTA_D + i0;
    device const float * k_ptr = qk + (size_t) (k_dim + kh * DELTA_D + i0);
    device const float * v_ptr = conv + (size_t) (2 * k_dim + h * DELTA_D + j);
    device const float * b_ptr = beta + h;
    device const float * g_ptr = g + h;
    device       float * o_ptr = out + (size_t) h * DELTA_D + j;

    for (int t = 0; t < args.seq; ++t) {
        // Decay the state, then contract it with k over the key dim.
        const float dec = exp(*g_ptr);
        float sk = 0.0f;
#pragma unroll
        for (int a = 0; a < DELTA_V2_KPL; ++a) {
            s[a] *= dec;
            sk += s[a] * k_ptr[a];
        }
        sk = simd_sum(sk);

        const float delta = (*v_ptr - sk) * (*b_ptr);

        // Rank-1 update, then read the UPDATED state out with q.
        float y = 0.0f;
#pragma unroll
        for (int a = 0; a < DELTA_V2_KPL; ++a) {
            s[a] += k_ptr[a] * delta;
            y += s[a] * q_ptr[a];
        }
        y = simd_sum(y);

        if (lane == 0) {
            *o_ptr = y * args.scale;
        }

        // Trail plane for this timestep, most-recent-first as in
        // kernel_delta_scan; plane 0 is the after-loop store below.
        const int target_slot = args.seq - 1 - t;
        if (target_slot >= 1 && target_slot < args.n_planes) {
            device float * s_plane =
                s_out + (size_t) target_slot * (size_t) args.v_heads * DELTA_D * DELTA_D;
#pragma unroll
            for (int a = 0; a < DELTA_V2_KPL; ++a) {
                s_plane[s_base + (size_t) a * DELTA_D] = s[a];
            }
        }

        q_ptr += qk_stride;
        k_ptr += qk_stride;
        v_ptr += args.conv_dim;
        b_ptr += args.v_heads;
        g_ptr += args.v_heads;
        o_ptr += (size_t) args.v_heads * DELTA_D;
    }

#pragma unroll
    for (int a = 0; a < DELTA_V2_KPL; ++a) {
        s_out[s_base + (size_t) a * DELTA_D] = s[a];
    }
}
