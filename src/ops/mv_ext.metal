// Vendored SMALL-BATCH quantized mat-vec ("mul_mv_ext") — the 2..8-token window
// between the plain gemv and the tiled gemm. Ported from the pinned llama.cpp
// clone at reference/llama.cpp (commit e9fa0781f1c2, 2026-07-28):
// ggml/src/ggml-metal/ggml-metal.metal:3903-4228 (the two impl templates and
// their instantiations) plus the dequantize helpers at :630-660 and :731-810,
// with the host gates at ggml/src/ggml-metal/ggml-metal-ops.cpp:2120-2223.
//
// What it is for. At a few tokens both shipping paths waste the weight read.
// candle's `mul_mm` reads each weight byte once but collapses to one
// threadgroup column and a handful of row blocks, so it leaves most of the
// device idle; the plain gemv (mv.metal) saturates bandwidth but re-reads the
// ENTIRE weight matrix once per token. This kernel does neither: a thread
// dequantizes a weight chunk into registers once and dots it against r1ptg
// (2..5) src1 rows before moving on, so the weight passes memory once and the
// token count rides along in registers.
//
// Geometry, baked. ggml carries nsg / nxpsg / ne12 / r2 / r3 as function
// constants; xwen has no function-constant plumbing (pipelines.rs compiles
// named entry points), so this file bakes one geometry: nsg = 2 simdgroups,
// nxpsg = 8 threads along a row, hence nypsg = 4 rows per simdgroup and
// r0ptg = 8 rows per threadgroup. Batch and broadcast are gone with them: xwen
// is batch 1 and never broadcasts, so ne12 = r2 = r3 = 1 and the i12/i13
// machinery is deleted — the same simplification mv.metal already made. Only
// r1ptg stays variable, as four entry points per dtype (r1_2 .. r1_5), because
// the host picks it from the token count.
//
// nxpsg = 8 is ggml's choice for MOST of what lands here, but not all of it,
// and the divergence is deliberate. ggml picks 16 when ne00 % 256 == 0 AND
// ne11 < 3, so it would run a K-quant at t = 2 with 16 lanes per row where this
// file runs 8. Specializing that case would mean a second geometry — another
// four entry points per K-quant dtype and a host rule to pick between them —
// to serve one token count, and the 8-lane path is correct there, just with
// half the row parallelism. (ggml would not take this kernel for a K-quant at
// t = 2..3 at all: it gates K-quants at ne11 >= 4. xwen routes them from 2
// because our fallback below 4 is candle's mul_mm rather than ggml's tuned
// alternative — recorded in TODO.md, and one of the things the measurement pass
// is meant to check.)
//
// Dtypes. q4_K, q6_K and q8_0 — the three the shipped Q4_K_M files actually
// present to a small-batch matmul (Q4_K experts and dense FFN, Q6_K lm_head,
// Q8_0 attention/ssm/shared-expert). q5_K is instantiated in ggml and
// deliberately skipped here: no supported checkpoint stores a weight in it on a
// path this kernel serves.
//
// ACCURACY. This is NOT bit-identical to anything shipping, and it cannot be:
// it changes the summation ORDER of the K reduction. The plain gemv reduces a
// row across 32 lanes; this reduces across 8, with each lane having summed a
// different interleave of chunks. No operand is narrowed anywhere — the weight
// chunk dequantizes into f32 registers, the activation is read f32, the
// accumulation is f32 — and there is no reduced-precision tensor path here
// (unlike dense_mm.metal), so the residual against a dequantize-then-f32 oracle
// is pure float-reassociation noise: measured 4e-7 .. 8e-6 rel_l2 across the
// three dtypes and the production reduction lengths.
//
// That makes it MORE accurate than what it displaces, not less. candle's
// quantized `mul_mm` — what `QMatMul` runs at these token counts — stages its
// dequantized weight tile as half, which puts it ~1.8e-4 from the same oracle
// at every one of those shapes. So the small-batch window is an accuracy
// improvement of 20x to 400x alongside the bandwidth one (the oracle tests in
// src/ops/mv_ext.rs assert the direction, not just the bound).
//
// It is still a non-bitwise change, so it is gated and graded like one: the
// XWEN_MV_EXT_CLASSIC kill-switch reverts every routing decision, the parity
// gate pins "classic" on both sides of the strict tier, and the bounded mm /
// decode / ppl tiers carry the signal (docs/parity.md).
//
// Compiled at runtime by src/ops/pipelines.rs via candle's
// new_library_with_source.

#include <metal_stdlib>

using namespace metal;

// Pin the library math-mode axis to the value nil compile options resolve to
// today (and that candle's kernels are explicitly compiled with), so a future
// OS default change cannot silently alter this library's codegen. clang
// hard-errors on bad `METAL fp` options, so compiling proves it is honored.
#pragma METAL fp math_mode(fast)

#define QK_K 256
#define QK8_0 32

// ---- Quantized block layouts (from ggml-common.h; unions there don't change
// the byte layout) — the same declarations mv.metal carries, repeated because
// each vendored .metal file is its own translation unit. ------------------

#define K_SCALE_SIZE 12

typedef struct {
    half    d;                    // super-block scale for quantized scales
    half    dmin;                 // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4-bit quants
} block_q4_K;

typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    half    d;               // super-block scale
} block_q6_K;

typedef struct {
    half   d;          // delta
    int8_t qs[QK8_0];  // quants
} block_q8_0;

// ---- Argument struct --------------------------------------------------------
// ggml's `ggml_metal_kargs_mul_mv_ext` minus everything batch/broadcast carries
// (ne02/ne12/nb02/nb03/nb12/nb13/r2/r3 are all 1 or unreachable at batch 1).
// The host writes the identical layout (dispatch.rs `MvExtArgs`).

typedef struct {
    int32_t  ne00;   // K, the reduction length
    int32_t  ne01;   // weight rows (n_out)
    uint64_t nb01;   // weight row stride, bytes
    int32_t  ne11;   // src1 rows (tokens)
    uint64_t nb11;   // src1 row stride, bytes
    int32_t  ne0;    // dst row length (== ne01)
} mv_ext_args;

// ---- Baked geometry ---------------------------------------------------------
// nsg = 2 is ggml's value unconditionally; nxpsg = 8 is its choice whenever
// ne00 % 128 == 0, which every shape here satisfies — except that it would
// prefer 16 for a K-quant at ne11 < 3, which this file does not specialize (see
// the header). dispatch.rs `MV_EXT_*` mirrors these for the grid;
// `mv_ext::tests::geometry_matches_metal` cross-checks the two.

#define MV_EXT_NSG   2
#define MV_EXT_NXPSG 8
#define MV_EXT_NYPSG (32/MV_EXT_NXPSG)

// ---- Dequantize helpers -----------------------------------------------------
// Verbatim from ggml-metal.metal, specialized to float4 / float4x4 outputs (the
// only instantiations the ext kernels use) so no function-pointer template
// specialization has to be named at the call site.

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

// 16 contiguous weights at [16*il, 16*il + 16) of a q4_K super-block.
static inline void deq_q4_K_f4x4(device const block_q4_K * xb, short il, thread float4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

// 16 contiguous weights at [16*il, 16*il + 16) of a q6_K super-block.
static inline void deq_q6_K_f4x4(device const block_q6_K * xb, short il, thread float4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
}

// 4 contiguous weights of a q8_0 block, in ggml's t4 chunk order.
static inline void deq_q8_0_f4(device const block_q8_0 * xb, short il, thread float4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    for (int i = 0; i < 4; i++) {
        reg[i] = (qs[4*(il%4) + i + 16*(il/4)] * d);
    }
}

// ---- Impl: float4 chunks (32-element blocks, q8_0) --------------------------
// chpb - chunks per quantization block; chpt - chunks per thread per pass.

template<short r1ptg, typename q_t, short chpb, void (*deq_t4)(device const q_t *, short, thread float4 &)>
static inline void mul_mv_ext_q4_f32_impl(
        constant mv_ext_args & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    constexpr short NSG   = MV_EXT_NSG;
    constexpr short nxpsg = MV_EXT_NXPSG;
    constexpr short nypsg = MV_EXT_NYPSG;
    constexpr short chpt  = 4;

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;

    const uint64_t offset0 = i01*args.nb01;
    const uint64_t offset1 = i11*args.nb11;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4 * y4[r1ptg];

    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4 *) src1;
    }

    float sumf[r1ptg];
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        sumf[ir1] = 0.0f;
    }

    short cch = tx%chpb; // current chunk index within the block

    for (int ich = tx; 4*ich < args.ne00; ich += chpt*nxpsg) {
        float4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch*nxpsg]);
            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4[ir1] += chpt*nxpsg;
        }
    }

    // Reduce only the nxpsg threads that share a row (the shuffle ladder ggml
    // runs for nxpsg == 8).
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        sumf[ir1] += simd_shuffle_down(sumf[ir1], 4);
        sumf[ir1] += simd_shuffle_down(sumf[ir1], 2);
        sumf[ir1] += simd_shuffle_down(sumf[ir1], 1);
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// ---- Impl: float4x4 chunks (256-element super-blocks, q4_K / q6_K) ----------

template<short r1ptg, typename q_t, short chpb, void (*deq_t4x4)(device const q_t *, short, thread float4x4 &)>
static inline void mul_mv_ext_q4x4_f32_impl(
        constant mv_ext_args & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    constexpr short NSG   = MV_EXT_NSG;
    constexpr short nxpsg = MV_EXT_NXPSG;
    constexpr short nypsg = MV_EXT_NYPSG;
    constexpr short chpt  = 1;

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;

    const uint64_t offset0 = i01*args.nb01;
    const uint64_t offset1 = i11*args.nb11;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4x4 * y4x4[r1ptg];

    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4x4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4x4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4x4 *) src1;
    }

    float sumf[r1ptg];
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        sumf[ir1] = 0.0f;
    }

    short cch = tx%chpb;

    for (int ich = tx; 16*ich < args.ne00; ich += chpt*nxpsg) {
        float4x4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4x4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] +=
                    dot(lx[ch][0], y4x4[ir1][ch*nxpsg][0]) +
                    dot(lx[ch][1], y4x4[ir1][ch*nxpsg][1]) +
                    dot(lx[ch][2], y4x4[ir1][ch*nxpsg][2]) +
                    dot(lx[ch][3], y4x4[ir1][ch*nxpsg][3]);
            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4x4[ir1] += chpt*nxpsg;
        }
    }

    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        sumf[ir1] += simd_shuffle_down(sumf[ir1], 4);
        sumf[ir1] += simd_shuffle_down(sumf[ir1], 2);
        sumf[ir1] += simd_shuffle_down(sumf[ir1], 1);
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// ---- Entry points -----------------------------------------------------------
// Four r1ptg widths per dtype; the host picks one from the token count
// (`ops::mv_ext_window`, the single authority on the plan — it consults ggml's
// table in `dispatch::mv_ext_r1ptg`). Names are globally unique across xwen's
// vendored sources, which pipelines.rs's pipeline cache requires.

#define MV_EXT_Q4X4_ENTRY(NAME, R1PTG, QT, CHPB, DEQ)                       \
    kernel void NAME(                                                       \
            constant mv_ext_args & args [[buffer(0)]],                      \
            device const char * src0    [[buffer(1)]],                      \
            device const char * src1    [[buffer(2)]],                      \
            device       char * dst     [[buffer(3)]],                      \
            uint3  tgpig [[threadgroup_position_in_grid]],                  \
            ushort tiisg [[thread_index_in_simdgroup]],                     \
            ushort sgitg [[simdgroup_index_in_threadgroup]]) {              \
        mul_mv_ext_q4x4_f32_impl<R1PTG, QT, CHPB, DEQ>(                     \
            args, src0, src1, dst, tgpig, tiisg, sgitg);                    \
    }

#define MV_EXT_Q4_ENTRY(NAME, R1PTG, QT, CHPB, DEQ)                         \
    kernel void NAME(                                                       \
            constant mv_ext_args & args [[buffer(0)]],                      \
            device const char * src0    [[buffer(1)]],                      \
            device const char * src1    [[buffer(2)]],                      \
            device       char * dst     [[buffer(3)]],                      \
            uint3  tgpig [[threadgroup_position_in_grid]],                  \
            ushort tiisg [[thread_index_in_simdgroup]],                     \
            ushort sgitg [[simdgroup_index_in_threadgroup]]) {              \
        mul_mv_ext_q4_f32_impl<R1PTG, QT, CHPB, DEQ>(                       \
            args, src0, src1, dst, tgpig, tiisg, sgitg);                    \
    }

// q4_K / q6_K: 256-element super-blocks, 16 float4x4 chunks each.
MV_EXT_Q4X4_ENTRY(kernel_mul_mv_ext_q4_K_f32_r1_2, 2, block_q4_K, 16, deq_q4_K_f4x4)
MV_EXT_Q4X4_ENTRY(kernel_mul_mv_ext_q4_K_f32_r1_3, 3, block_q4_K, 16, deq_q4_K_f4x4)
MV_EXT_Q4X4_ENTRY(kernel_mul_mv_ext_q4_K_f32_r1_4, 4, block_q4_K, 16, deq_q4_K_f4x4)
MV_EXT_Q4X4_ENTRY(kernel_mul_mv_ext_q4_K_f32_r1_5, 5, block_q4_K, 16, deq_q4_K_f4x4)

MV_EXT_Q4X4_ENTRY(kernel_mul_mv_ext_q6_K_f32_r1_2, 2, block_q6_K, 16, deq_q6_K_f4x4)
MV_EXT_Q4X4_ENTRY(kernel_mul_mv_ext_q6_K_f32_r1_3, 3, block_q6_K, 16, deq_q6_K_f4x4)
MV_EXT_Q4X4_ENTRY(kernel_mul_mv_ext_q6_K_f32_r1_4, 4, block_q6_K, 16, deq_q6_K_f4x4)
MV_EXT_Q4X4_ENTRY(kernel_mul_mv_ext_q6_K_f32_r1_5, 5, block_q6_K, 16, deq_q6_K_f4x4)

// q8_0: 32-element blocks, 8 float4 chunks each.
MV_EXT_Q4_ENTRY(kernel_mul_mv_ext_q8_0_f32_r1_2, 2, block_q8_0, 8, deq_q8_0_f4)
MV_EXT_Q4_ENTRY(kernel_mul_mv_ext_q8_0_f32_r1_3, 3, block_q8_0, 8, deq_q8_0_f4)
MV_EXT_Q4_ENTRY(kernel_mul_mv_ext_q8_0_f32_r1_4, 4, block_q8_0, 8, deq_q8_0_f4)
MV_EXT_Q4_ENTRY(kernel_mul_mv_ext_q8_0_f32_r1_5, 5, block_q8_0, 8, deq_q8_0_f4)
