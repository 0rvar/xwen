// Bidirectional flash attention on the Metal-4 cooperative-tensor path: the
// diffusion transformer's self-attention (30 heads of 128, one unordered set of
// image and caption tokens, no mask), with both products of every key block,
// S = Q K^T and O += P V, issued through `mpp::tensor_ops::matmul2d`, the same
// tensor-core primitive the transformer's linears run on (bf16_t.metal,
// dense_mm.metal). flash.metal is the classic simdgroup-matrix kernel and stays
// the causal path's kernel unchanged; this file is bidirectional only, head_dim
// 128 only, and the `tensor` arm of XWEN_ZIMAGE_ATTN.
//
// Structure. One threadgroup per (query block of BQ = 4 x ROWS rows, head), four
// simdgroups. The threadgroup stages its BQ query rows into a half tile once
// (16 KB, the only threadgroup memory), then every tensor op runs at
// `execution_simdgroup` scope: each simdgroup owns ROWS of those rows and walks
// the whole key range on its own, so after the one staging barrier the four
// never synchronize. That scope is not a choice: the library's input
// cooperative tensors, row reductions and iterator mapping are all
// static_asserted to a single SIMD group (MPPTensorOpsMatMul2dImpl.h), and the
// whole online softmax below is built from those three facilities.
//
// Per key block of BK columns, per simdgroup:
//   1. S = Q K^T into a cooperative destination tensor (f32): the staged half Q
//      rows against the device f16 K rows read directly with transpose_right.
//      Columns past the last key come back zero from the extent clip and are
//      set to the finite minimum here.
//   2. Row max of S through `reduce_rows` into a row-reduction cooperative
//      tensor; the running max, running sum and the rescale factor alpha are
//      three more of those, so element i names the same row in all of them.
//   3. P = exp2(S * scale * log2(e) - m) in place; row sum through
//      `reduce_rows`.
//   4. O *= alpha per row, skipped when no row's max moved (alpha is then
//      exactly 1). O never leaves its cooperative destination tensor between
//      key blocks.
//   5. P becomes the LEFT INPUT cooperative tensor of the P V op without a
//      round trip through memory (`get_left_input_cooperative_tensor(S)`), in
//      f32, the library requiring the input's element type to equal the
//      source's; `float x half -> float` is a supported combination. V is the
//      device f16 tensor read directly. O accumulates in multiply_accumulate.
// After the last block O *= 1 / l per row and stores through the device tensor,
// whose extents clip the rows of a partial last query block.
//
// Which row an element belongs to is asked of the library ONCE, before the key
// loop: `map_iterator` from S and from O onto the row-reduction tensor gives
// the slot, and the slots are packed into one bit per element. The layout is
// fixed for the kernel, only the statistics change, so the loop does two-way
// selects and no index arithmetic. The row-reduction layout of a 16-row op
// gives every lane two rows (flash_t.rs's layout probe reads it off the
// device); a device that lays them out otherwise poisons its output rather
// than computing a wrong one.
//
// Operand precision, measured (tests/zimage_microbench.rs, flash_t.rs): the
// tensor core consumes an f32 operand under relaxed_precision at LESS than
// f16 precision, and reading Q from device memory as f32 was both slower
// (10.5 vs 6.1 ms at 30 x 4128) and less accurate (rel L2 9.6e-4 vs 4.5e-4)
// than the half staging. P stays f32 because the input-from-destination
// conversion demands it; routing it through a half threadgroup tile instead
// was slower. Staging the K and V tiles into threadgroup memory for the four
// simdgroups was slower still: the per-block barriers couple simdgroups that
// are otherwise independent.
//
// Masked scores use the finite minimum rather than -INFINITY: the library is
// compiled math_mode(fast) like every vendored kernel, and exp2 of a huge
// negative finite value is exactly 0 without depending on infinity semantics.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// Pin the library math-mode axis to the value nil compile options resolve to
// today (and that candle's kernels are explicitly compiled with), so a future
// OS default change cannot silently alter this library's codegen.
#pragma METAL fp math_mode(fast)

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Matches dispatch.rs's FlashTArgs field for field.
typedef struct {
    int32_t n_q;          // query rows per head
    int32_t n_k;          // key rows per head
    int32_t gqa_factor;   // query heads per kv head
    float   scale_log2;   // softmax scale times log2(e)
    int64_t q_stride_h;   // head strides, in elements
    int64_t k_stride_h;
    int64_t v_stride_h;
    int64_t o_stride_h;
} flash_t_args;

constexpr constant int FLASH_T_BD = 128;
constexpr constant int FLASH_T_NSG = 4;

// Stage the threadgroup's Q rows as half: 128 threads move 8 consecutive
// elements each per pass, rows past `valid` zero-filled. `src` is the head's
// f32 plane at the block's first row.
template <int TILE_ROWS>
inline void stage_q_tile(threadgroup half * dst, device const float * src, int valid, ushort tiitg) {
    constexpr int CHUNKS = TILE_ROWS * FLASH_T_BD / 8;
    for (int c = tiitg; c < CHUNKS; c += 32 * FLASH_T_NSG) {
        const int row = c / (FLASH_T_BD / 8);
        const int col = (c % (FLASH_T_BD / 8)) * 8;
        threadgroup half4 * d = (threadgroup half4 *)(dst + row * FLASH_T_BD + col);
        if (row < valid) {
            device const float4 * s = (device const float4 *)(src + row * FLASH_T_BD + col);
            d[0] = half4(s[0]);
            d[1] = half4(s[1]);
        } else {
            d[0] = half4(0.0h);
            d[1] = half4(0.0h);
        }
    }
}

template <int ROWS, int BK>
kernel void flash_attn_t(
        constant flash_t_args & args [[buffer(0)]],
        device float * Q [[buffer(1)]],
        device half  * K [[buffer(2)]],
        device half  * V [[buffer(3)]],
        device float * O [[buffer(4)]],
        threadgroup half * sQ [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int BD = FLASH_T_BD;
    constexpr int BQ = ROWS * FLASH_T_NSG;
    // Row statistics a lane holds (see the header).
    constexpr int R_SLOTS = 2;
    static_assert(ROWS == 16, "the row-slot selects are written for 16-row tiles");

    const int h  = tgpig.y;
    const int qb = tgpig.x * BQ;
    const int q0 = qb + sgitg * ROWS;
    const int hk = h / args.gqa_factor;

    device float * q = Q + h  * args.q_stride_h;
    device half  * k = K + hk * args.k_stride_h;
    device half  * v = V + hk * args.v_stride_h;
    device float * o = O + h  * args.o_stride_h;

    stage_q_tile<BQ>(sQ, q + qb * BD, args.n_q - qb, tiitg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // A simdgroup with no query row is uniform in this decision, and every
    // tensor op below is simdgroup-scoped, so leaving is safe once the shared
    // staging is done.
    if (q0 >= args.n_q) {
        return;
    }

    // Row-major [rows, 128] planes as (128, rows) tensors: dim 0 is the
    // contiguous head dim. Q and O are (K, M) / (N, M) of their ops, K is the
    // right operand transposed, (K, N), and V the right operand plain, (N, K).
    auto tQ = tensor(sQ, dextents<int32_t, 2>(BD, BQ));
    auto tK = tensor(k, dextents<int32_t, 2>(BD, args.n_k), array<int, 2>({1, BD}));
    auto tV = tensor(v, dextents<int32_t, 2>(BD, args.n_k), array<int, 2>({1, BD}));
    auto tO = tensor(o, dextents<int32_t, 2>(BD, args.n_q), array<int, 2>({1, BD}));

    // S = Q K^T: (ROWS x 128) x (BK x 128)^T -> ROWS x BK, fresh each block.
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            ROWS, BK, BD, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroup> mmS;
    // O += P V: (ROWS x BK) x (BK x 128) -> ROWS x 128, accumulated over blocks.
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            ROWS, BD, BK, false, false, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroup> mmO;

    auto mQ = tQ.slice(0, sgitg * ROWS);

    auto S = mmS.template get_destination_cooperative_tensor<decltype(mQ), decltype(tK), float>();
    // Row statistics, all in the S op's row-reduction layout: element i is the
    // same query row in each of them.
    auto m_run = mmS.template get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(tK), float>();
    auto l_run = mmS.template get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(tK), float>();
    auto r_new = mmS.template get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(tK), float>();
    auto alpha = mmS.template get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(tK), float>();

    using P_t = decltype(mmO.template get_left_input_cooperative_tensor<float, half, float>(S));
    auto Oacc = mmO.template get_destination_cooperative_tensor<P_t, decltype(tV), float>();

    if (alpha.get_capacity() != R_SLOTS) {
        FOR_UNROLL (uint16_t i = 0; i < Oacc.get_capacity(); ++i) {
            Oacc[i] = as_type<float>(0x7fc00000u);
        }
        Oacc.store(tO.slice(0, q0));
        return;
    }

    // Which of the lane's two row slots each element of S and of O belongs
    // to, one bit per element.
    uint32_t s_slot = 0;
    FOR_UNROLL (uint16_t i = 0; i < S.get_capacity(); ++i) {
        const uint32_t slot = uint32_t(alpha.map_iterator(S.begin() + i) - alpha.begin());
        s_slot |= (slot & 1u) << i;
    }
    uint64_t o_slot = 0;
    FOR_UNROLL (uint16_t i = 0; i < Oacc.get_capacity(); ++i) {
        const uint64_t slot = uint64_t(alpha.map_iterator(Oacc.begin() + i) - alpha.begin());
        o_slot |= (slot & 1u) << i;
    }

    const float lowest = numeric_limits<float>::lowest();
    FOR_UNROLL (uint16_t i = 0; i < R_SLOTS; ++i) {
        m_run[i] = lowest;
        l_run[i] = 0.0f;
    }
    FOR_UNROLL (uint16_t i = 0; i < Oacc.get_capacity(); ++i) {
        Oacc[i] = 0.0f;
    }

    const float c = args.scale_log2;
    const int nk = (args.n_k + BK - 1) / BK;

    for (int kb = 0; kb < nk; ++kb) {
        const int k0 = kb * BK;

        auto mK = tK.slice(0, k0);
        mmS.run(mQ, mK, S);

        // Columns past the last key read as zero through the extent clip;
        // give them the finite minimum so their exp2 is exactly 0.
        if (k0 + BK > args.n_k) {
            FOR_UNROLL (uint16_t i = 0; i < S.get_capacity(); ++i) {
                auto idx = S.get_multidimensional_index(i);
                if (k0 + idx[0] >= args.n_k) {
                    S[i] = lowest;
                }
            }
        }

        // New running max in log2 units; alpha rescales the old sum and O.
        mpp::tensor_ops::reduce_rows(S, r_new, mpp::tensor_ops::reduction_operation::max, lowest);
        FOR_UNROLL (uint16_t i = 0; i < R_SLOTS; ++i) {
            const float m_new = max(r_new[i] * c, m_run[i]);
            alpha[i] = exp2(m_run[i] - m_new);
            m_run[i] = m_new;
        }
        const float m0 = m_run[0];
        const float m1 = m_run[1];

        // P = exp2(S * c - m) in place.
        FOR_UNROLL (uint16_t i = 0; i < S.get_capacity(); ++i) {
            const float m = ((s_slot >> i) & 1u) ? m1 : m0;
            S[i] = exp2(S[i] * c - m);
        }

        mpp::tensor_ops::reduce_rows(S, r_new, mpp::tensor_ops::reduction_operation::sum, 0.0f);
        FOR_UNROLL (uint16_t i = 0; i < R_SLOTS; ++i) {
            l_run[i] = l_run[i] * alpha[i] + r_new[i];
        }

        // Rescale O only when some row's max moved: alpha is exactly 1
        // otherwise, and after the first blocks it almost always is.
        const float a0 = alpha[0];
        const float a1 = alpha[1];
        if (simd_any(a0 != 1.0f || a1 != 1.0f)) {
            FOR_UNROLL (uint16_t i = 0; i < Oacc.get_capacity(); ++i) {
                Oacc[i] *= ((o_slot >> i) & 1u) ? a1 : a0;
            }
        }

        auto P  = mmO.template get_left_input_cooperative_tensor<float, half, float>(S);
        auto mV = tV.slice(0, k0);
        mmO.run(P, mV, Oacc);
    }

    const float r0 = 1.0f / l_run[0];
    const float r1 = 1.0f / l_run[1];
    FOR_UNROLL (uint16_t i = 0; i < Oacc.get_capacity(); ++i) {
        Oacc[i] *= ((o_slot >> i) & 1u) ? r1 : r0;
    }

    Oacc.store(tO.slice(0, q0));
}

typedef decltype(flash_attn_t<16, 32>) flash_attn_t_fn;

// BK 32 is the shipped geometry; 64 stays instantiated so the choice remains
// priced by tests/zimage_microbench.rs (it measured slower at both shapes).
template [[host_name("kernel_flash_attn_t_k32")]] kernel flash_attn_t_fn flash_attn_t<16, 32>;
template [[host_name("kernel_flash_attn_t_k64")]] kernel flash_attn_t_fn flash_attn_t<16, 64>;

// Test-only probe of the cooperative-tensor facts the kernel above rests on,
// for the shipped geometry and operand types: per-lane capacities of S, the
// row reduction and O, whether the iterator mapping from S and from O onto the
// row reduction is compatible on this device, and the (column, row) coordinate
// of every element of S and O for every lane, so flash_t.rs can assert the
// layout covers each tile exactly once. Output (int32): [cap_S, cap_R, cap_O,
// compat_S, compat_O, 0, 0, 0], then lane-major coordinate dumps, S at 32 lanes
// x cap_S x 2 and O at 32 lanes x cap_O x 2, an invalid element as (-1, -1).
kernel void kernel_flash_attn_t_probe(
        device half  * K [[buffer(2)]],
        device half  * V [[buffer(3)]],
        device int32_t * out [[buffer(4)]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr int ROWS = 16;
    constexpr int BK = 32;
    constexpr int BD = FLASH_T_BD;

    threadgroup half sQ[ROWS * BD];
    auto tQ = tensor(sQ, dextents<int32_t, 2>(BD, ROWS));
    auto tK = tensor(K, dextents<int32_t, 2>(BD, BK), array<int, 2>({1, BD}));
    auto tV = tensor(V, dextents<int32_t, 2>(BD, BK), array<int, 2>({1, BD}));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            ROWS, BK, BD, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroup> mmS;
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            ROWS, BD, BK, false, false, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroup> mmO;

    auto S = mmS.get_destination_cooperative_tensor<decltype(tQ), decltype(tK), float>();
    auto R = mmS.get_row_reduction_destination_cooperative_tensor<decltype(tQ), decltype(tK), float>();
    using P_t = decltype(mmO.get_left_input_cooperative_tensor<float, half, float>(S));
    auto Oacc = mmO.get_destination_cooperative_tensor<P_t, decltype(tV), float>();

    const int cap_s = S.get_capacity();
    const int cap_r = R.get_capacity();
    const int cap_o = Oacc.get_capacity();
    if (tiisg == 0) {
        out[0] = cap_s;
        out[1] = cap_r;
        out[2] = cap_o;
        out[3] = mpp::tensor_ops::is_iterator_compatible(S, R) ? 1 : 0;
        out[4] = mpp::tensor_ops::is_iterator_compatible(Oacc, R) ? 1 : 0;
    }
    device int32_t * dump = out + 8;
    for (int i = 0; i < cap_s; ++i) {
        auto idx = S.get_multidimensional_index(i);
        const bool valid = S.is_valid_element(i);
        dump[(tiisg * cap_s + i) * 2 + 0] = valid ? idx[0] : -1;
        dump[(tiisg * cap_s + i) * 2 + 1] = valid ? idx[1] : -1;
    }
    dump += 32 * cap_s * 2;
    for (int i = 0; i < cap_o; ++i) {
        auto idx = Oacc.get_multidimensional_index(i);
        const bool valid = Oacc.is_valid_element(i);
        dump[(tiisg * cap_o + i) * 2 + 0] = valid ? idx[0] : -1;
        dump[(tiisg * cap_o + i) * 2 + 1] = valid ? idx[1] : -1;
    }
}
