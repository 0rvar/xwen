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
//   kernel_delta_scan    the whole delta-rule recurrence over T timesteps, with
//                        the head's state slice resident in registers for the
//                        entire scan: one dispatch per layer per chunk. It folds
//                        the q/k L2 clamp-norm into its own load stage.
//   kernel_delta_gnorm   the gated output RMSNorm (rms -> ssm_norm.weight ->
//                        silu(z)), with kernel_delta_gnorm_sigmoid its
//                        sigmoid(z) sibling for the qwen4exp graph — one
//                        templated body, the arm chosen by name at dispatch.
//
// A layer therefore costs eight dispatches at any sequence length. The
// kill-switch back to the reference scan is XWEN_DELTA_CLASSIC.
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
