// Vendored bf16-weight x f32-activation matmul kernels — the BF16 twin of
// f16.metal, consumed by the DFlash drafter's mmap-aliased BF16 matmul planes
// (the official drafter GGUF stores every matmul weight BF16; aliasing them
// no-copy means the kernels must read bfloat, not half). Kernel bodies are
// line-for-line f16.metal's with ONLY the device weight loads changed
// (half/half4/half4x4 -> bfloat reads, widened through float4()): bfloat ->
// float widening is exact, so over weights whose values are representable in
// BOTH formats these kernels are bit-identical to their f16 twins — the
// bf16.rs tests pin exactly that. Keep the bodies in sync with f16.metal.
//
// Two kernels, mirroring the same host-side split on ne11 (token count):
//   kernel_mul_mv_bf16_f32_v — the decode gemv (t <= 8).
//   kernel_mul_mm_bf16_f32_v — the classic simdgroup prefill gemm
//     (XWEN_ATTN_MM_CLASSIC; the default t > 8 path is bf16_t.metal). Float
//     tiles, like f16.metal's deliberate deviation from the fork — ggml's own
//     bf16 instantiation stages bfloat tiles with simdgroup_bfloat8x8, which we
//     deliberately do NOT copy (the float-tile body needs no bfloat simdgroup
//     support and keeps the "stored weights are the only sub-f32 rounding"
//     contract).
//
// `bfloat` needs MSL >= 3.1; runtime compilation on this OS resolves far above
// that (mm_id.metal's Metal-4 tensor code compiles in the same process), and
// the M5-only mandate applies. DELIBERATELY a SEPARATE library from f16.metal:
// that file is attention-critical and this one is drafter-critical — a bfloat
// compile problem must be able to break only the drafter path, never attention.
// Compiled lazily on first bf16 dispatch by src/ops/pipelines.rs.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// Pin the library math-mode axis to the value nil compile options resolve to
// today (and that candle's kernels are explicitly compiled with), so a future
// OS default change cannot silently alter this library's codegen. clang
// hard-errors on bad `METAL fp` options, so compiling proves it is honored.
#pragma METAL fp math_mode(fast)

#define N_SIMDWIDTH 32

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// ---- Argument structs -------------------------------------------------------
// Byte-for-byte the fork's ggml_metal_kargs_mul_mv / _mul_mm (ggml-metal-impl.h)
// and f16.metal's mv_args / mm_args; the host writes the identical layout
// (dispatch.rs MvArgs / MmArgs).

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

// ---- kernel_mul_mv_bf16_f32_v (decode gemv) ---------------------------------
// The fork's host constants for our shapes (see f16.metal's header).
#define MV_NR0 2
#define MV_NSG 4

// Verbatim from ggml-metal.metal helper_mv_reduce_and_write<NR0> (== f16.metal's
// copy): per-row simd_sum, cross-simdgroup combine via shmem (NW floats per
// row), single writer, ragged-tail row guard on the store.
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

// f16.metal's kernel_mul_mv_f16_f32_v with the weight loads read as bfloat4
// (ggml's kernel_mul_mv_bf16_f32_4 is the same template instantiated at
// T0=bfloat). Grid: (ceil(ne01/NR0), ne11, 1); threads (32, MV_NSG, 1); each
// simdgroup covers a disjoint slice of the K reduction for the same NR0 rows.
// Weights are read as bfloat4 and widened to float in the dot — an EXACT
// widening (bf16 ⊂ f32), f32 products, f32 accum.
kernel void kernel_mul_mv_bf16_f32_v(
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
    device const bfloat  * ax [NR0];
    device const bfloat4 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01;

        ax [row] = (device const bfloat  *) ((device const char *) src0 + offset0);
        ax4[row] = (device const bfloat4 *) ((device const char *) src0 + offset0);
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
            device const bfloat4 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), yl4[i]);
            }

            sumf[row] += sumq;
        }

        yb4 += NSG*NF*NW/4;
    }

    // K tail (ne00 % 32 != 0) — never taken at our shapes (host requires
    // ne00 % 32 == 0), kept verbatim from the fork.
    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += (float) ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

// ---- kernel_mul_mm_bf16_f32_v (classic prefill gemm) ------------------------
// f16.metal's kernel_mul_mm_f16_f32_v with the weight tile loaded as bfloat4
// quads and widened through float4() (exact) instead of a half4x4 matrix
// conversion — ggml's dequantize_bf16 is the same widen-copy. Float tiles
// (f16.metal's deviation from the fork, kept here — NOT ggml's bfloat-tile
// instantiation). Grid: (ceil(ne1/NR1), ceil(ne0/NR0), 1); 128 threads
// (4 simdgroups); a 64(out-row) x 32(token) tile per threadgroup with the
// guarded threadgroup store-back (bc_out resolved true).
kernel void kernel_mul_mm_bf16_f32_v(
        constant mm_args   & args [[buffer(0)]],
        device const char * src0  [[buffer(1)]],
        device const char * src1  [[buffer(2)]],
        device       char * dst   [[buffer(3)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    // sa holds an NR0(64) x NK(32) weight tile, sb an NR1(32) x NK(32)
    // activation tile, both float; the store-back reuses the region as an
    // NR0 x NR1 float tile (8192 B, within sa's span).
    threadgroup float * sa = (threadgroup float *)(shmem);
    threadgroup float * sb = (threadgroup float *)(shmem + sizeof(float) * 64 * 32);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    // nl == 1: il stays il0 and x advances NK elements per K step. bfloat4
    // quads instead of f16.metal's half4x4 (4 quads = one 16-element chunk),
    // so the chunk index scales by 4 and the K-step advance is 8 quads.
    device const bfloat4 * x = (device const bfloat4 *)(src0 + args.nb01*(r0 + lr0)) + 4*il0;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    simdgroup_float8x8 ma[4];
    simdgroup_float8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // load data and store to threadgroup memory
        {
            // dequantize_bf16: a plain 16-element widen-copy (4 bfloat4 quads),
            // exact bf16 -> f32.
            float4 temp_a[4];
            FOR_UNROLL (short j = 0; j < 4; j++) {
                temp_a[j] = float4(x[j]);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup float2x4 *)(sb + 64*ib + 8*ly) = *((device const float2x4 *) y);
        }

        x += 8;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const float * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const float * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    // Guarded store-back (the fork's bc_out branch): stage the tile in
    // threadgroup memory, then copy only the in-bounds nr0 x nr1 region.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1) {
            device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0;
            device float4 * D4 = (device float4 *) D;

            threadgroup float  * C  = ((threadgroup float *) shmem) + (j*NR0);
            threadgroup float4 * C4 = (threadgroup float4 *) C;

            int i = 0;
            for (; i < nr0/4; i++) {
                *(D4 + i) = *(C4 + i);
            }

            i *= 4;
            for (; i < nr0; i++) {
                *(D + i) = *(C + i);
            }
        }
    }
}
