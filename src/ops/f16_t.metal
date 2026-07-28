// Vendored Metal-4 cooperative-tensor port of the f16-weight x f32-activation
// attention PREFILL gemm — the tensor analogue of f16.metal's classic simdgroup
// kernel_mul_mm_f16_f32_v. The SHIPPED DEFAULT for matmul_f16's mm branch
// (ne11 > 8); XWEN_ATTN_MM_CLASSIC reverts to the classic simdgroup kernel.
// The decode gemv (ne11 <= 8) never reaches here — it always runs f16.metal's
// classic kernel_mul_mv_f16_f32_v.
//
// Ported from ggml's DEDICATED dense cooperative-tensor kernel_mul_mm (the
// tensor variant, ggml-metal.metal), NOT from mm_id.metal's MoE gather geometry.
// Only the A tile (the f16 weight) is staged to threadgroup memory as half; the
// B operand (the f32 activation) is wrapped as a DEVICE-memory cooperative tensor
// and read directly — NO threadgroup staging of B, which is the throughput win
// over the prior MoE-geometry port (~2x: it staged both operands and drained the
// output through threadgroup memory). The matmul2d descriptor's reduced-precision
// flag is set (ggml's dense-kernel value), so the Metal-4 tensor cores compute at
// reduced precision regardless of the f32 B declaration — the result is the
// fork's ~2e-4 prefill precision class (docs/parity.md §3b, the "mm" parity
// tier), NOT f32 accumulation-order noise. Reading B as f32 buys the SPEED (no
// staging), not precision. Accumulation is f32 and the output stores f32.
//
// Geometry (ggml-metal-impl.h): NR0 = 64 out rows (M, NRA), NR1 = 128 tokens
// (N, NRB), NK = 32 K-step (N_MM_NK_TOTAL), 4 simdgroups / 128 threads. The A
// tile is 64x32 half = 4096 B of threadgroup memory; there is no B tile and no C
// tile in threadgroup memory. Token-edge (N not a multiple of 128) and out-edge
// (M not a multiple of 64) tiles are handled by the cooperative-tensor extents:
// A zero-pads rows >= M, tB's N extent zero-fills columns >= N, and cT.store
// clips to tD's (M, N) extents. We require K % 32 == 0 (asserted on the host in
// dispatch.rs — every attention K is a multiple of 1024), so ggml's element-wise
// bc_inp path and its K-tail zero-pad are structurally unreachable and omitted.
//
// DELIBERATELY a SEPARATE library from f16.metal: this file needs Metal-4
// (<metal_tensor> + matmul2d), so it is compiled lazily on first tensor-path
// dispatch (src/ops/pipelines.rs::f16_t_pipeline) and the classic f16.metal
// library stays Metal-4-free. Mirrors the mm_id.metal / mm_id_t_hp.metal split.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// Pin the library math-mode axis to the value nil compile options resolve to
// today (and that candle's kernels are explicitly compiled with), so a future
// OS default change cannot silently alter this library's codegen. clang
// hard-errors on bad `METAL fp` options, so compiling proves it is honored.
#pragma METAL fp math_mode(fast)

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Byte-for-byte the fork's ggml_metal_kargs_mul_mm (== f16.metal's mm_args and
// dispatch.rs's MmArgs); the host writes the identical layout.
typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} mm_args;

// Dense f16-weight gemm on the cooperative-tensor path: half A tile (threadgroup),
// float B tensor (device, read directly), f32 accumulate, f32 output. Grid:
// (ceil(ne1/NR1), ceil(ne0/NR0), 1); 128 threads (4 simdgroups); 4096 B
// threadgroup memory (the A tile only). Single-batch attention projection
// (ne12 == 1, r2 == r3 == 1) — the only case matmul_f16 dispatches — so no batch
// offset is threaded through.
kernel void kernel_mul_mm_f16_f32_t(
        constant mm_args   & args [[buffer(0)]],
        device const char * src0  [[buffer(1)]],
        device const char * src1  [[buffer(2)]],
        device       char * dst   [[buffer(3)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    (void) tiisg;
    (void) sgitg;

    // Matrix dimensions: A(M,K) x B(K,N) -> C(M,N)
    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;

    constexpr int NR0 = 64;   // out rows per tile (M, NRA)
    constexpr int NR1 = 128;  // tokens per tile (N, NRB)
    constexpr int NK  = 32;   // K-step per iteration (N_MM_NK_TOTAL)

    constexpr int NK_CHUNK = 16;             // halves per work item (SZ_SIMDGROUP)
    constexpr int NCHUNK   = NK / NK_CHUNK;  // K chunks per row per step (N_MM_NK)
    constexpr int A_WORK_ITEMS = NR0 * NCHUNK;
    constexpr int NUM_THREADS  = 128;

    // Tile offsets in the output matrix.
    const int ra = tgpig.y * NR0;  // first out row (M) of this tile
    const int rb = tgpig.x * NR1;  // first token (N) of this tile

    // Threadgroup memory for the dequantized A tile only.
    threadgroup half * sa = (threadgroup half *)(shmem);

    // tA wraps threadgroup memory as (NK, NR0); tB wraps device src1 directly as
    // (K, N) with the token-row stride, read as f32 with no staging.
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));

    // Non-const element type: the MPP cooperative tensor rejects `const float` as
    // an operand element type (ggml casts srcB to a non-const `device T1 *` too);
    // the kernel only ever reads through it.
    device float * ptrB = (device float *)(src1);
    const int strideB = args.nb11 / sizeof(float);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    // Configure the matmul: descriptor(N, M, K, transpose_left=false,
    // transpose_right=true, reduced_precision=true), 4 simdgroups. ggml's
    // dense-kernel value for the last flag: the reduced-precision tensor-core
    // path (~2x the precise path — the `false` variant runs at classic speed even
    // reading B from device), which computes at reduced precision even over the
    // f32 B operand, giving the fork's ~2e-4 prefill precision class.
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NR1, NR0, NK, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    // CRITICAL operand order: B (device float) is the FIRST template/run operand,
    // A (threadgroup half) the second. The (A, B) order is only valid for
    // same-type operands; mixed half x float operands require this ggml order.
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    // Accumulate partial results over the K dimension.
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        // === PHASE 1: stage the f16 weight tile into threadgroup memory ===
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
            const int   row     = work / NCHUNK;
            const int   k_chunk = work % NCHUNK;
            const int   k_pos   = loop_k + k_chunk * NK_CHUNK;
            const short k_base  = k_chunk * NK_CHUNK;

            if (ra + row < M) {
                // Dense f16 weight: one half4x4 (16 halves) per K chunk. K % 32 == 0
                // (host-asserted), so k_pos + i < K holds for the whole chunk and no
                // K-tail zero-pad is needed.
                device const half4x4 * row_ptr =
                    (device const half4x4 *)(src0 + args.nb01 * (ra + row));
                half4x4 temp_a = row_ptr[k_pos / NK_CHUNK];

                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row * NK + (k_base + i)] = temp_a[i/4][i%4];
                }
            } else {
                // Zero-pad rows beyond the out-dimension (M not a multiple of 64).
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row * NK + (k_base + i)] = (half)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === PHASE 2: tensor matmul (B first) ===
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, rb);

        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store the result tile directly to device. tD's (M, N) extents clip the
    // out-edge and token-edge partial tiles; dst is [token][out] row-major with
    // leading dim ne0 == M (stride {1, M}), matching the classic kernel's layout.
    device float * dstD = (device float *)dst;
    auto tD = tensor(dstD, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
    cT.store(tD.slice(ra, rb));
}
