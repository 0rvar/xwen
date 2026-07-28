// Vendored Metal-4 cooperative-tensor bf16-weight x f32-activation prefill
// gemm — the BF16 twin of f16_t.metal, the DEFAULT mm branch (ne11 > 8) of
// matmul_bf16 (the DFlash drafter's mmap-aliased BF16 planes);
// XWEN_ATTN_MM_CLASSIC reverts to bf16.metal's classic simdgroup kernel. The
// decode gemv (ne11 <= 8) never reaches here — it always runs bf16.metal's
// kernel_mul_mv_bf16_f32_v.
//
// The body is line-for-line f16_t.metal's with ONLY the A-tile staging loads
// changed: the weight is read as bfloat4 quads and CONVERTED TO HALF while
// staging into the (unchanged) half threadgroup tile, so `matmul2d` never sees
// a bfloat operand — ggml runtime-probes whether the tensor API accepts bfloat
// at all (ggml-metal-device.m has_bfloat probe); converting at the staging
// boundary sidesteps that question entirely. The drafter load scans each BF16
// plane and hard-errors on any value that would OVERFLOW f16 (dflash.rs
// `ensure_bf16_fits_f16`, the same bound the old materialize-to-f16 load
// enforced via its finiteness check). Within that bound the staging narrow is
// exact in f16's normal range (7 mantissa bits ⊂ 10); the ~2e-5 of drafter
// weights down in f16's SUBNORMAL range round/flush here exactly as the old
// materialized-f16 path did (the gemv and classic gemm instead keep the full
// bf16 value — the documented seq-boundary asymmetry, pinned by an ops/bf16.rs
// test). Keep the body in sync with f16_t.metal.
//
// DELIBERATELY a SEPARATE library from f16_t.metal (attention-critical) and
// bf16.metal (Metal-4-free): compiled lazily on first bf16 tensor-path
// dispatch, so a compile problem here can break only the drafter path.

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

// Byte-for-byte the fork's ggml_metal_kargs_mul_mm (== f16_t.metal's mm_args
// and dispatch.rs's MmArgs); the host writes the identical layout.
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

// Dense bf16-weight gemm on the cooperative-tensor path: bfloat device loads
// staged to a half A tile (threadgroup), float B tensor (device, read
// directly), f32 accumulate, f32 output. Grid: (ceil(ne1/NR1), ceil(ne0/NR0),
// 1); 128 threads (4 simdgroups); 4096 B threadgroup memory (the A tile only).
// Single-matrix (ne02 == ne12 == 1, r2 == r3 == 1) — the only case
// matmul_bf16 dispatches — so no batch offset is threaded through.
kernel void kernel_mul_mm_bf16_f32_t(
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

    constexpr int NK_CHUNK = 16;             // weight elements per work item (SZ_SIMDGROUP)
    constexpr int NCHUNK   = NK / NK_CHUNK;  // K chunks per row per step (N_MM_NK)
    constexpr int A_WORK_ITEMS = NR0 * NCHUNK;
    constexpr int NUM_THREADS  = 128;

    // Tile offsets in the output matrix.
    const int ra = tgpig.y * NR0;  // first out row (M) of this tile
    const int rb = tgpig.x * NR1;  // first token (N) of this tile

    // Threadgroup memory for the staged (bf16 -> half) A tile only.
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
        // === PHASE 1: stage the bf16 weight tile into threadgroup memory as half ===
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
            const int   row     = work / NCHUNK;
            const int   k_chunk = work % NCHUNK;
            const int   k_pos   = loop_k + k_chunk * NK_CHUNK;
            const short k_base  = k_chunk * NK_CHUNK;

            if (ra + row < M) {
                // Dense bf16 weight: one 16-element chunk = 4 bfloat4 quads per
                // K chunk, widened to float (exact) and narrowed to half (exact
                // for the f16-range values the load scan admits). K % 32 == 0
                // (host-asserted), so k_pos + i < K holds for the whole chunk
                // and no K-tail zero-pad is needed.
                device const bfloat4 * row_ptr =
                    (device const bfloat4 *)(src0 + args.nb01 * (ra + row));

                FOR_UNROLL (short j = 0; j < 4; j++) {
                    const half4 v = half4(float4(row_ptr[k_pos / 4 + j]));
                    FOR_UNROLL (short e = 0; e < 4; e++) {
                        sa[row * NK + (k_base + 4*j + e)] = v[e];
                    }
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
