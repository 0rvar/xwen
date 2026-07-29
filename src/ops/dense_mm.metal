// Vendored Metal-4 cooperative-tensor DENSE quantized-weight x f32-activation
// prefill gemm — the 27B's SwiGLU FFN projections (ffn_gate / ffn_up /
// ffn_down, Q4_K in the shipped Q4_K_M file). The SHIPPED DEFAULT for
// `DenseMlp`'s prefill (seq > DENSE_MM_MIN_SEQ); XWEN_DENSE_MM_CLASSIC reverts
// to candle's `QMatMul` chain. Decode never reaches here.
//
// This file is `f16_t.metal`'s kernel with ONE substitution: the A-tile staging
// phase reads block-quantized weights and dequantizes them in-kernel instead of
// widening stored halves. Everything else — the 64(out) x 128(token) tile
// geometry, the device-memory cooperative tensor over the f32 activation (no B
// staging), the reduced-precision matmul2d descriptor, the extent-clipped
// store — is unchanged, so the two kernels share a precision class and a host
// argument struct. That is also ggml's own structure: its dense tensor
// `kernel_mul_mm` is templated over (block_q, nl, dequantize_func) and
// instantiated for the same dtype set.
//
// Why an in-kernel tile dequant rather than a dequantize-to-f16 scratch feeding
// the existing f16 gemm: at the 27B's FFN shapes a materialized f16 plane of
// [17408, 5120] is 178 MB written and re-read per projection per chunk against
// the 50 MB of Q4_K the kernel would otherwise stream — ~8x the weight traffic,
// which costs back a fifth of the win and needs a large scratch buffer. The tile
// dequant reads each Q4_K super-block exactly once and never leaves registers.
//
// The A tile is reused across NR1 = 128 tokens (not mm_id.metal's 32), so each
// dequantized weight element feeds 128 MACs. That ratio is why this kernel is
// not dequant-bound the way the MoE gather is at its narrower token tile.
//
// Ported geometry (ggml-metal-impl.h): NR0 = 64 out rows (M, NRA), NR1 = 128
// tokens (N, NRB), NK = 32 K-step (N_MM_NK_TOTAL), 4 simdgroups / 128 threads.
// The A tile is 64x32 half = 4096 B of threadgroup memory; no B tile, no C tile.
// Out-edge (M % 64) and token-edge (N % 128) tiles are handled by the
// cooperative-tensor extents exactly as in f16_t.metal: A zero-pads rows >= M,
// tB's N extent zero-fills columns >= N, and cT.store clips to tD's (M, N).
//
// DELIBERATELY a SEPARATE library from mm_id.metal (whose dequant helpers this
// file re-vendors): that one is MoE-prefill-critical, this one is
// dense-FFN-prefill-critical, and neither can break the other. Compiled lazily
// on first dense-prefill dispatch (src/ops/pipelines.rs::dense_mm_pipeline).

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// Pin the library math-mode axis to the value nil compile options resolve to
// today (and that candle's kernels are explicitly compiled with), so a future
// OS default change cannot silently alter this library's codegen. clang
// hard-errors on bad `METAL fp` options, so compiling proves it is honored.
#pragma METAL fp math_mode(fast)

#define QK_K 256
#define K_SCALE_SIZE 12
#define QK8_0 32
#define QK_NL 16

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// ---- Quantized block layouts (flattened from ggml-common.h; the aggregate
// unions there do not change the byte layout). Re-vendored from mm_id.metal so
// the two libraries stay independent translation units. ----------------------

typedef struct {
    half   d;          // delta
    int8_t qs[QK8_0];  // quants
} block_q8_0;

typedef struct {
    half    d;                    // super-block scale for quantized scales
    half    dmin;                 // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4-bit quants
} block_q4_K;

typedef struct {
    half    d;
    half    dmin;
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;

typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    half    d;               // super-block scale
} block_q6_K;

// ---- Dequantize functions (verbatim from the fork / mm_id.metal) ------------
// Each call produces the 16 CONTIGUOUS weight elements at super-block offset
// [16*il, 16*il + 16) — the property the A-tile staging below indexes by.

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }

    reg = (type4x4) reg_f;
}

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
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

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;

    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask  = il<2 ? 0x0F : 0xF0;
    const float qh_val = il<2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg) {
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

// ---- Argument struct --------------------------------------------------------
// Byte-for-byte the fork's ggml_metal_kargs_mul_mm (== f16.metal's / f16_t.metal's
// mm_args and dispatch.rs's MmArgs); the host writes the identical layout.
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

// ---- The dense quantized cooperative-tensor gemm ----------------------------
// Grid: (ceil(ne1/NR1), ceil(ne0/NR0), 1); 128 threads (4 simdgroups); 4096 B
// threadgroup memory (the A tile only). Single matrix (ne02 == ne12 == 1,
// r2 == r3 == 1) — the only case the dense FFN dispatches — so no batch offset
// is threaded through.
//
// `nl` is the number of 16-element dequant sub-blocks per quantized super-block
// (QK_NL = 16 for the K-quants, 2 for q8_0), so the super-block spans 16*nl
// elements. The host requires ne00 % 32 == 0 AND ne00 % (16*nl) == 0, which
// makes every A-tile read land on a whole sub-block and removes ggml's K-tail
// zero-pad.
template<typename S0_4x4, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &)>
kernel void kernel_mul_mm_q_f32_t(
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

    constexpr int NK_CHUNK = 16;             // elements per dequant call (SZ_SIMDGROUP)
    constexpr int NCHUNK   = NK / NK_CHUNK;  // K chunks per row per step (N_MM_NK)
    constexpr int A_WORK_ITEMS = NR0 * NCHUNK;
    constexpr int NUM_THREADS  = 128;

    // Elements per quantized super-block, and the byte size of one.
    constexpr int QK = NK_CHUNK * nl;

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

    // descriptor(N, M, K, transpose_left=false, transpose_right=true,
    // reduced_precision=true), 4 simdgroups — ggml's dense-kernel values. The
    // reduced-precision tensor-core path is ~2x the precise one and puts this
    // kernel in the fork's ~2e-4 prefill precision class (docs/parity.md §3b).
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
        // === PHASE 1: dequantize the weight tile into threadgroup memory ===
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
            const int   row     = work / NCHUNK;
            const int   k_chunk = work % NCHUNK;
            const int   k_pos   = loop_k + k_chunk * NK_CHUNK;
            const short k_base  = k_chunk * NK_CHUNK;

            if (ra + row < M) {
                // One dequant call yields the 16 contiguous elements at k_pos:
                // super-block k_pos/QK, sub-block (k_pos/16) % nl. K is a whole
                // multiple of QK (host-asserted), so the chunk never straddles a
                // super-block and no K-tail zero-pad is needed.
                device const block_q * xb =
                    (device const block_q *)(src0 + args.nb01 * (ra + row)) + k_pos / QK;

                S0_4x4 temp_a;
                dequantize_func(xb, (short)((k_pos / NK_CHUNK) % nl), temp_a);

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

typedef decltype(kernel_mul_mm_q_f32_t<half4x4, block_q8_0, 2, dequantize_q8_0>) mul_mm_q_t;

// Half A tiles only — matmul2d's validated operand type on this device (the
// mm_id.metal probe). q4_K is the production instantiation (the shipped Q4_K_M
// file's ffn_gate/ffn_up/ffn_down); the other three exist so the dtype support
// matrix is not a single-entry special case and the tests can sweep them.
template [[host_name("kernel_mul_mm_q8_0_f32_t")]] kernel mul_mm_q_t kernel_mul_mm_q_f32_t<half4x4, block_q8_0, 2,     dequantize_q8_0>;
template [[host_name("kernel_mul_mm_q4_K_f32_t")]] kernel mul_mm_q_t kernel_mul_mm_q_f32_t<half4x4, block_q4_K, QK_NL, dequantize_q4_K>;
template [[host_name("kernel_mul_mm_q5_K_f32_t")]] kernel mul_mm_q_t kernel_mul_mm_q_f32_t<half4x4, block_q5_K, QK_NL, dequantize_q5_K>;
template [[host_name("kernel_mul_mm_q6_K_f32_t")]] kernel mul_mm_q_t kernel_mul_mm_q_f32_t<half4x4, block_q6_K, QK_NL, dequantize_q6_K>;
