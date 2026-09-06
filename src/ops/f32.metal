// Vendored f32-weight x f32-activation mat-vec — the F32 twin of f16.metal's
// decode gemv, consumed by the MoE ROUTER projection (`ffn_gate_inp`, the one
// f32 matmul plane in the graph). Kernel body is line-for-line
// kernel_mul_mv_f16_f32_v's with ONLY the device weight loads changed
// (half/half4 -> float/float4, no widening conversion needed): the operands are
// bit-identical to what candle's mlx gemv reads, so the difference is pure
// accumulation ORDER, bounded and not bitwise.
//
// Why it exists at all: candle's `Tensor::matmul` sends a `[1, hidden] x
// [hidden, n_expert]` f32 product to the mlx `gemv_t` family, which picks a tile
// giving EIGHT threadgroups for the whole 5.24 MB router plane (Flash-Next
// hidden 2560 x 512 experts; the 35B-A3B's 2048 x 256 also lands on 8). This
// kernel's grid is ceil(n_out/NR0) = 256 threadgroups over the same bytes.
//
// ONE kernel, not two: above 8 tokens the host keeps the caller on candle
// (`ops::matmul_f32` bails), because the per-token weight re-read this gemv
// pays (grid.y = ne11) stops being free there and the router is a tiny fraction
// of prefill wall.
//
// DELIBERATELY a SEPARATE library from f16.metal and mv.metal, for the same
// isolation reason those two are separate from each other: this one is
// MoE-router-critical, f16.metal is attention-critical. Compiled at runtime by
// src/ops/pipelines.rs via candle's new_library_with_source, so `cargo build`
// proves nothing about it — `xcrun -sdk macosx metal -c src/ops/f32.metal` does.

#include <metal_stdlib>

using namespace metal;

// Pin the library math-mode axis to the value nil compile options resolve to
// today (and that candle's kernels are explicitly compiled with), so a future
// OS default change cannot silently alter this library's codegen. clang
// hard-errors on bad `METAL fp` options, so compiling proves it is honored.
#pragma METAL fp math_mode(fast)

#define N_SIMDWIDTH 32

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// ---- Argument struct --------------------------------------------------------
// Byte-for-byte the fork's ggml_metal_kargs_mul_mv (ggml-metal-impl.h) and
// f16.metal's / bf16.metal's mv_args; the host writes the identical layout
// (dispatch.rs MvArgs), which is why all three share one launcher.

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
} mv_args;

// ---- kernel_mul_mv_f32_f32_v (decode gemv) ----------------------------------
// The fork's host constants for our shapes, matching f16.metal: nr0 = 2 src0
// rows per threadgroup and nsg = min(4, ceil(ne00/128)) = 4 simdgroups
// splitting the K reduction (every router K is >= 2048).
#define MV_NR0 2
#define MV_NSG 4

// Verbatim from ggml-metal.metal helper_mv_reduce_and_write<NR0> (and
// f16.metal's copy): per-row simd_sum, cross-simdgroup combine via shmem (NW
// floats per row), single writer, ragged-tail row guard on the store.
template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = tot;
        }
    }
}

// kernel_mul_mv_t_t_4_impl<float, float4, float, float4, 2> with the broadcast
// function constants resolved (ne12 == 1, r2 == r3 == 1 -> i12 == i13 == 0).
// Grid: (ceil(ne01/NR0), ne11, 1); threads (32, MV_NSG, 1); each simdgroup
// covers a disjoint slice of the K reduction for the same NR0 rows. Weights are
// read as float4 — no conversion at all, so the products are the same f32
// products candle's gemv forms and only the summation order differs.
kernel void kernel_mul_mv_f32_f32_v(
        constant mv_args   & args [[buffer(0)]],
        device const char * src0  [[buffer(1)]],
        device const char * src1  [[buffer(2)]],
        device       char * dst   [[buffer(3)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NR0 = MV_NR0;
    constexpr short NSG = MV_NSG;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;

    const uint64_t offset1 = r1*args.nb11;

    device const float  * y  = (device const float  *) (src1 + offset1);
    device const float4 * y4 = (device const float4 *) (src1 + offset1);

    // pointers to src0 rows
    device const float  * ax [NR0];
    device const float4 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01;

        ax [row] = (device const float  *) ((device const char *) src0 + offset0);
        ax4[row] = (device const float4 *) ((device const char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    float4 yl4[NF4];

    device const float4 * yb4 = y4 + (ib0*NB + il*NF)/4;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const float4 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(xb4[i], yl4[i]);
            }

            sumf[row] += sumq;
        }

        yb4 += NSG*NF*NW/4;
    }

    // K tail (ne00 % 32 != 0) — never taken at our shapes (host requires
    // ne00 % 32 == 0), kept verbatim from the fork.
    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}
