// Direct 2D convolution for the Z-Image VAE decoder: 3x3 stride 1 pad 1 and
// 1x1, f32 in, f32 out, on candle's NCHW layout, bias fused into the store.
//
// candle's Metal conv2d is an im2col that materializes nine copies of the
// input, a narrow-N gemm over them and an NHWC to NCHW permute of the result.
// This kernel is the implicit form of that gemm: a threadgroup owns a tile of
// output pixels (TH rows by TW columns) for a block of CO output channels,
// stages the input tile with its halo and the weights for a chunk of CI input
// channels in threadgroup memory, and accumulates with simdgroup 8x8 f32
// matrix ops, one k-step per (tap, 8 input channels). The A operand of each
// step is the staged tile read at the tap's offset and transposed on load, so
// the im2col matrix is never written anywhere.
//
// The input read is where the decoder's elementwise work folds in, all of it
// optional and selected by `flags`:
//   AFFINE    v = v * scale[b, c] + shift[b, c], the GroupNorm apply with its
//             statistics already folded per channel (see group_norm.metal),
//   SILU      v = v / (1 + exp(-v)) after the affine, the resnet's activation,
//   UPSAMPLE  the source is read at half coordinates, the decoder's 2x
//             nearest upsample, so the 4x tensor is never materialized,
//   RESIDUAL  out += residual[b, co, y, x], the resnet's skip add.
// Zero padding applies after the affine and the activation, as the reference
// pads the activated tensor.
//
// Weights arrive permuted to [taps][c_in][c_out] (done once at load), so a
// weight stage is a contiguous run per (tap, ci) and the B operand needs no
// transpose. Output channels past c_out and pixels past the image are padded
// with zeros on the way in and skipped on the way out, so any c_out and any
// H, W work; c_in must be a multiple of CI (the wrapper refuses otherwise).
//
// Accumulation order differs from candle's gemm, so results agree to f32
// rounding, not bitwise; the conv2d_direct.rs tests hold rel_l2 under 1e-5.
// Fast math like the rest of the vendored libraries; nothing here has a
// bitwise contract with a candle chain.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

#pragma METAL fp math_mode(fast)

// Matches dispatch.rs Conv2dDirectArgs (#[repr(C)]).
typedef struct {
    int32_t batch;
    int32_t c_in;
    int32_t c_out;
    int32_t height;     // output rows
    int32_t width;      // output columns
    int32_t src_height; // input rows as stored (height / 2 under UPSAMPLE)
    int32_t src_width;
    int32_t flags;
} conv2d_direct_args;

constant int32_t CONV_FLAG_SILU     = 1;
constant int32_t CONV_FLAG_AFFINE   = 2;
constant int32_t CONV_FLAG_RESIDUAL = 4;
constant int32_t CONV_FLAG_UPSAMPLE = 8;

// Output tile per threadgroup and the input-channel chunk per k-step. TW is
// two 8-pixel blocks; TH rows make 16 pixel blocks in all, spread over the
// 8 simdgroups of a 256-thread group. CI is the 8 the simdgroup matrix
// wants as K.
constant constexpr int CONV_TH = 8;
constant constexpr int CONV_TW = 16;
constant constexpr int CONV_CI = 8;
constant constexpr int CONV_THREADS = 256;
constant constexpr int CONV_SIMDGROUPS = CONV_THREADS / 32;

// CO: output channels per threadgroup, a multiple of 8. NCB: 8-wide output
// channel blocks per simdgroup; the simdgroups split CO/8 blocks into
// CO/8/NCB groups and the 2*TH pixel blocks into the rest. TH: tile rows; 16
// halves the staging phases per FLOP for the shallow (128-channel) convs,
// where a chunk stages more bytes per MMA than it can hide, at twice the
// accumulator registers.
template <int CO, int KS, int NCB, int TH>
kernel void kernel_conv2d_direct(
        constant conv2d_direct_args & args [[buffer(0)]],
        device const float * x            [[buffer(1)]],
        device const float * w            [[buffer(2)]],
        device const float * bias         [[buffer(3)]],
        device       float * dst          [[buffer(4)]],
        device const float * scale        [[buffer(5)]],
        device const float * shift        [[buffer(6)]],
        device const float * residual     [[buffer(7)]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint   tid [[thread_index_in_threadgroup]],
        uint  sgid [[simdgroup_index_in_threadgroup]],
        uint  lane [[thread_index_in_simdgroup]]) {
    constexpr int TW = CONV_TW;
    constexpr int CI = CONV_CI;
    constexpr int HALO = KS / 2;
    constexpr int PH = TH + 2 * HALO;
    constexpr int PW = TW + 2 * HALO;
    constexpr int PS = PH * PW;
    constexpr int TAPS = KS * KS;
    constexpr int CB = CO / 8;                       // output channel blocks
    constexpr int SG_PER_CB = CB / NCB;              // simdgroups sharing a pixel group
    constexpr int NPB = 2 * TH * SG_PER_CB / CONV_SIMDGROUPS; // pixel blocks per simdgroup
    static_assert(CB % NCB == 0, "NCB must divide CO/8");
    static_assert(CONV_SIMDGROUPS % SG_PER_CB == 0, "simdgroups must split evenly");
    static_assert((2 * TH * SG_PER_CB) % CONV_SIMDGROUPS == 0, "pixel blocks must split evenly");
    static_assert((CONV_SIMDGROUPS / SG_PER_CB) * NPB == 2 * TH, "pixel blocks must be covered");

    threadgroup float tile[CI * PS];                 // [ci][py][px]
    threadgroup float wts[TAPS * CI * CO];           // [tap][ci][co]
    threadgroup float scratch[CONV_SIMDGROUPS * 64]; // one 8x8 per simdgroup

    const int flags = args.flags;
    const int c_in = args.c_in;
    const int c_out = args.c_out;
    const int H = args.height;
    const int W = args.width;
    const int SH = args.src_height;
    const int SW = args.src_width;
    const int co_blocks = (c_out + CO - 1) / CO;
    const int b = (int) tgid.z / co_blocks;
    const int co0 = ((int) tgid.z % co_blocks) * CO;
    const int tx0 = (int) tgid.x * TW;
    const int ty0 = (int) tgid.y * TH;
    const bool upsample = (flags & CONV_FLAG_UPSAMPLE) != 0;
    const bool affine = (flags & CONV_FLAG_AFFINE) != 0;
    const bool silu = (flags & CONV_FLAG_SILU) != 0;

    // This simdgroup's pixel blocks and output channel blocks.
    const int sg = (int) sgid;
    const int cb0 = (sg % SG_PER_CB) * NCB;
    const int pb0 = (sg / SG_PER_CB) * NPB;

    simdgroup_float8x8 acc[NPB][NCB];
    #pragma clang loop unroll(full)
    for (int i = 0; i < NPB; i++) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < NCB; j++) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    device const float * xb = x + (size_t) b * c_in * SH * SW;
    device const float * scale_b = scale + (size_t) b * c_in;
    device const float * shift_b = shift + (size_t) b * c_in;

    for (int c0 = 0; c0 < c_in; c0 += CI) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Input tile with halo, zero outside the image. Consecutive threads
        // take consecutive px, so device reads run along a row.
        for (int i = (int) tid; i < CI * PS; i += CONV_THREADS) {
            const int ci = i / PS;
            const int p = i - ci * PS;
            const int py = p / PW;
            const int px = p - py * PW;
            const int iy = ty0 + py - HALO;
            const int ix = tx0 + px - HALO;
            float v = 0.0f;
            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                const int sy = upsample ? (iy >> 1) : iy;
                const int sx = upsample ? (ix >> 1) : ix;
                const int c = c0 + ci;
                v = xb[((size_t) c * SH + sy) * SW + sx];
                if (affine) {
                    v = v * scale_b[c] + shift_b[c];
                }
                if (silu) {
                    v = v / (1.0f + exp(-v));
                }
            }
            tile[i] = v;
        }
        // Weights for this chunk, zero past c_out. Consecutive threads take
        // consecutive co, a contiguous run of the permuted plane.
        for (int i = (int) tid; i < TAPS * CI * CO; i += CONV_THREADS) {
            const int co = i % CO;
            const int tc = i / CO; // tap * CI + ci
            const int tap = tc / CI;
            const int ci = tc - tap * CI;
            const int cog = co0 + co;
            float v = 0.0f;
            if (cog < c_out) {
                v = w[((size_t) tap * c_in + c0 + ci) * c_out + cog];
            }
            wts[i] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)

        for (int tap = 0; tap < TAPS; tap++) {
            const int kh = tap / KS;
            const int kw = tap - kh * KS;
            simdgroup_float8x8 bm[NCB];
            #pragma clang loop unroll(full)
            for (int j = 0; j < NCB; j++) {
                simdgroup_load(bm[j], wts + tap * CI * CO + (cb0 + j) * 8, CO);
            }
            #pragma clang loop unroll(full)
            for (int i = 0; i < NPB; i++) {
                const int pb = pb0 + i;
                const int r = pb >> 1;
                const int cx = (pb & 1) * 8;
                // Rows of the staged tile are input channels, so the
                // transposed load yields [pixel][ci] for this tap's offset.
                simdgroup_float8x8 am;
                simdgroup_load(am, tile + (r + kh) * PW + cx + kw, PS, ulong2(0, 0), true);
                #pragma clang loop unroll(full)
                for (int j = 0; j < NCB; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], am, bm[j], acc[i][j]);
                }
            }
        }
    }

    // Epilogue: each 8x8 goes through the simdgroup's scratch so the store
    // can add the bias and the residual, clip to the image and to c_out, and
    // write runs along a row.
    threadgroup float * sc = scratch + sg * 64;
    const bool has_residual = (flags & CONV_FLAG_RESIDUAL) != 0;
    const size_t plane = (size_t) H * W;
    #pragma clang loop unroll(full)
    for (int i = 0; i < NPB; i++) {
        const int pb = pb0 + i;
        const int r = pb >> 1;
        const int cx = (pb & 1) * 8;
        const int y = ty0 + r;
        #pragma clang loop unroll(full)
        for (int j = 0; j < NCB; j++) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(acc[i][j], sc, 8); // sc[px][co]
            simdgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (int e = (int) lane; e < 64; e += 32) {
                const int co = e >> 3;
                const int px = e & 7;
                const int cog = co0 + (cb0 + j) * 8 + co;
                const int xg = tx0 + cx + px;
                if (y < H && xg < W && cog < c_out) {
                    const size_t idx = ((size_t) b * c_out + cog) * plane + (size_t) y * W + xg;
                    float v = sc[px * 8 + co] + bias[cog];
                    if (has_residual) {
                        v += residual[idx];
                    }
                    dst[idx] = v;
                }
            }
        }
    }
}

typedef decltype(kernel_conv2d_direct<64, 3, 4, CONV_TH>) conv2d_direct_t;

// 64 output channels per threadgroup, 4 blocks per simdgroup: the resnet
// convs at 128 to 512 channels.
template [[host_name("kernel_conv2d_direct_3x3_co64")]] kernel conv2d_direct_t kernel_conv2d_direct<64, 3, 4, CONV_TH>;
template [[host_name("kernel_conv2d_direct_1x1_co64")]] kernel conv2d_direct_t kernel_conv2d_direct<64, 1, 4, CONV_TH>;
// The 16-row tile for the shallow convs (c_in up to 128).
template [[host_name("kernel_conv2d_direct_3x3_co64_th16")]] kernel conv2d_direct_t kernel_conv2d_direct<64, 3, 4, 2 * CONV_TH>;
// 8 output channels per threadgroup: conv_out's three channels, which a
// 64-wide block would compute eight times over.
template [[host_name("kernel_conv2d_direct_3x3_co8")]] kernel conv2d_direct_t kernel_conv2d_direct<8, 3, 1, CONV_TH>;
