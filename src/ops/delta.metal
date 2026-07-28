// Vendored fused gated-DeltaNet kernels — the linear-attention layers of
// src/linear_attn.rs, whose composed-candle form (the frozen reference scan)
// spends ~65 Metal dispatches per decoded token on a layer that contains only
// seven matmuls, and one dispatch PER TIMESTEP PER LAYER during prefill. These
// four kernels collapse the glue:
//
//   kernel_delta_conv   depthwise causal conv + silu + next conv window, reading
//                       the carried window and the fresh qkv rows as two
//                       buffers (no concatenation, no slice_set state write).
//   kernel_delta_ba     beta = sigmoid(b_raw) and g = ssm_a * softplus(a_raw +
//                       dt_bias) from ONE fused [hidden, 2*v_heads] projection.
//   kernel_delta_scan   the whole delta-rule recurrence over T timesteps, with
//                       the head's state slice resident in registers for the
//                       entire scan: one dispatch per layer per chunk.
//   kernel_delta_gnorm  the gated output RMSNorm (rms -> ssm_norm.weight ->
//                       silu(z)).
//
// A layer therefore costs eight dispatches at any sequence length. The
// kill-switch back to the reference scan is XWEN_DELTA_CLASSIC.
//
// ROUNDING. TWO of the four kernels are BIT-IDENTICAL to the candle chain they
// replace, and pin FP contraction and reassociation off at block scope
// (`#pragma clang fp contract(off)` / `reassociate(off)`):
//   - kernel_delta_conv reproduces the reference's tap chain exactly: the first
//     tap is a bare product and each later tap is a separate f32 multiply
//     followed by a separate f32 add (the reference materializes one tensor per
//     tap), then candle's usilu `x / (1 + exp(-x))`.
//   - kernel_delta_ba reproduces candle's usigmoid `1/(1 + exp(-x))` and the
//     stable softplus chain of linear_attn.rs (abs, add, affine-as-fma, exp,
//     affine-as-fma, log, add) with the same rounding boundaries, then one
//     multiply by ssm_a.
// The other two are BOUNDED, not bitwise, because a reduction the reference
// runs in one order is partitioned here:
//   - kernel_delta_gnorm's arithmetic matches the reference's ops one for one,
//     but the 128-term sum-of-squares reduction reassociates (hardware simd_sum
//     instead of candle's reduce partition). Its test grades at 2e-6.
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

    // candle's usigmoid: recip(1 + exp(-x)).
    const float b = ba[base + h];
    beta[tid] = 1.0f / (1 + exp(-b));

    // softplus(a_raw + dt_bias), the stable relu + ln(1 + exp(-|x|)) form, with
    // the reference's per-op rounding boundaries: candle's affine kernel is an
    // explicit fma, its badd/bmul single f32 operations.
    const float x = ba[base + args.v_heads + h] + dt_bias[h];
    const float ax = abs(x);
    const float sum = x + ax;
    const float relu = fma(sum, 0.5f, 0.0f);
    const float e = exp(-ax);
    const float one_plus = fma(e, 1.0f, 1.0f);
    const float tail = log(one_plus);
    const float sp = relu + tail;
    g[tid] = sp * ssm_a[h];
}

// Matches dispatch.rs DeltaGnormArgs (#[repr(C)]).
typedef struct {
    int32_t v_heads;
    int32_t head_dim;
    float eps;
} delta_gnorm_args;

// Gated output RMSNorm: normalize over the head dim, scale by
// ssm_norm.weight, and only THEN multiply by silu(z). The gate is outside the
// norm, so it does not affect the statistic.
//
// One threadgroup per (token, head), `head_dim` threads wide; the sum of
// squares folds through simd_sum and one threadgroup-memory pass.
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
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    threadgroup float partial[32];

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
    // candle's usilu on the gate, applied after the weight (see the ordering
    // test in linear_attn.rs).
    dst[idx] = ((x / den) * w[d]) * (zv / (1 + exp(-zv)));
}

// The scan kernel's fixed geometry. Both checkpoints run gated DeltaNet at head
// dim 128 (27B: 16 K-heads / 48 V-heads; 35B-A3B: 16 / 32), so the kernel is
// specialized to it and the host refuses any other head dim (falling back to
// the reference scan).
#define DELTA_D 128       // head dim, and the threads per threadgroup
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
    }

#pragma unroll
    for (int a = 0; a < DELTA_S_SLICE; ++a) {
        s_out[s_base + (size_t) (i0 + a) * DELTA_D] = s[a];
    }
}
