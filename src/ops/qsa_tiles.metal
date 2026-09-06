// Tile-batched sparse attention for a QSA prefill chunk: the pieces that turn
// a chunk's per-query block lists (kernel_qsa_select_mask) into the compact,
// padded operands candle's batched sdpa runs over, so the sparse layers stop
// computing every masked-out column of the cache at prefill.
//
// A tile is T consecutive queries. Per tile:
//   union  — the DISTINCT blocks any query of the tile selected, ascending, and
//            their count (kernel_qsa_tile_union; one threadgroup per tile, a
//            bitmap over the blocks in threadgroup memory).
//   gather — the union's K and V rows packed to `[tile, head, S, head_dim]`
//            (kernel_qsa_tile_gather_kv), where S is the host-chosen padded
//            column count; padding rows copy row 0 and are masked below.
//   mask   — the tile's `[T, S]` f16 additive mask, read off the chunk's full
//            `[n, n_kv]` mask at the union's columns; padding columns and the
//            tail block's columns past the row end are -inf,
//            and a padding query row (past the chunk's last query) copies the
//            last real row so its softmax has something finite to see
//            (kernel_qsa_tile_mask).
//   q      — the queries re-laid as `[tile, head, T, head_dim]` f16, padding
//            rows copying the last real query (kernel_qsa_tile_q).
// The batched sdpa then attends each tile over its own S columns; the union is
// exact (every column a query needs is in it) and every column a query must
// not see carries -inf, so the result is dense attention's up to summation
// order.

#include <metal_stdlib>

using namespace metal;

#define QSA_TILES_MAX_BLOCKS 65536
#define QSA_TILES_MAX_WORDS (QSA_TILES_MAX_BLOCKS / 32)
#define QSA_TILES_MAX_SIMDGROUPS 32

static inline uint tiles_exclusive_scan(
        uint v,
        threadgroup uint * simd_tot,
        uint lane,
        uint sg) {
    const uint local = simd_prefix_exclusive_sum(v);
    const uint total = simd_sum(v);
    if (lane == 0) {
        simd_tot[sg] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = 0;
    for (uint i = 0; i < sg; i++) {
        base += simd_tot[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return base + local;
}

// Matches dispatch.rs QsaTileUnionArgs (#[repr(C)]).
typedef struct {
    int32_t n;         // queries in the chunk
    int32_t t;         // queries per tile
    int32_t cap_in;    // block-list row stride (keep_max + 1)
    int32_t n_blocks;  // bitmap width: every block id a list may hold is below
                       // it, the tail's incomplete block included; <= QSA_TILES_MAX_BLOCKS
    int32_t cap_out;   // union row stride
} qsa_tile_union_args;

kernel void kernel_qsa_tile_union(
        constant qsa_tile_union_args & args  [[buffer(0)]],
        device const uint32_t * blocks       [[buffer(1)]],
        device const uint32_t * nsel         [[buffer(2)]],
        device uint32_t * union_out          [[buffer(3)]],
        device uint32_t * count              [[buffer(4)]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint tid   [[thread_index_in_threadgroup]],
        uint3 tg3  [[threads_per_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]],
        uint sg    [[simdgroup_index_in_threadgroup]]) {
    threadgroup atomic_uint words[QSA_TILES_MAX_WORDS];
    threadgroup uint simd_tot[QSA_TILES_MAX_SIMDGROUPS];

    const uint width = tg3.x;
    const uint tile = tgid.x;
    const uint n = (uint) args.n;
    const uint t = (uint) args.t;
    const uint cap_in = (uint) args.cap_in;
    const uint n_words = ((uint) args.n_blocks + 31) / 32;
    const uint row_lo = tile * t;
    const uint row_hi = min(row_lo + t, n);

    for (uint w = tid; w < n_words; w += width) {
        atomic_store_explicit(&words[w], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = row_lo; r < row_hi; r++) {
        const uint m = nsel[r];
        device const uint32_t * list = blocks + (size_t) r * cap_in;
        for (uint s = tid; s < m; s += width) {
            const uint b = list[s];
            atomic_fetch_or_explicit(&words[b >> 5], 1u << (b & 31u), memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compaction in word order: a contiguous run of words per thread, so the
    // scanned slots come out ascending by block index.
    const uint chunk = (n_words + width - 1) / width;
    const uint lo = min(tid * chunk, n_words);
    const uint hi = min(lo + chunk, n_words);
    uint mine = 0;
    for (uint w = lo; w < hi; w++) {
        mine += popcount(atomic_load_explicit(&words[w], memory_order_relaxed));
    }
    uint slot = tiles_exclusive_scan(mine, simd_tot, lane, sg);
    device uint32_t * out = union_out + (size_t) tile * (uint) args.cap_out;
    for (uint w = lo; w < hi; w++) {
        uint bits = atomic_load_explicit(&words[w], memory_order_relaxed);
        while (bits != 0) {
            const uint bit = ctz(bits);
            out[slot++] = w * 32 + bit;
            bits &= bits - 1;
        }
    }
    if (tid == width - 1) {
        count[tile] = slot;
    }
}

// Matches dispatch.rs QsaTileGatherArgs (#[repr(C)]).
typedef struct {
    int32_t n;          // queries in the chunk
    int32_t t;          // queries per tile
    int32_t n_tiles;
    int32_t s;          // padded columns per tile
    int32_t ratio;
    int32_t cap_out;    // union row stride
    int32_t heads;      // kv heads
    int32_t head_dim;   // multiple of 4
    int32_t len;        // cache rows (n_kv)
    int64_t src_stride_h;
    int64_t src_stride_r;
} qsa_tile_gather_args;

// One threadgroup per (column j, tile, head): the union's row for column j of
// the tile, or row 0 when j is padding, copied as vec4s.
kernel void kernel_qsa_tile_gather_kv(
        constant qsa_tile_gather_args & args [[buffer(0)]],
        device const uint32_t * union_in     [[buffer(1)]],
        device const uint32_t * count        [[buffer(2)]],
        device const half * k_src            [[buffer(3)]],
        device const half * v_src            [[buffer(4)]],
        device half * k_dst                  [[buffer(5)]],
        device half * v_dst                  [[buffer(6)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tid3  [[thread_position_in_threadgroup]],
        uint3 tg3   [[threads_per_threadgroup]]) {
    const uint j = tgpig.x;
    const uint tile = tgpig.y;
    const uint h = tgpig.z;
    const uint s = (uint) args.s;
    const uint ratio = (uint) args.ratio;
    if (j >= s) {
        return;
    }
    const uint u = j / ratio;
    uint row = 0;
    if (u < count[tile]) {
        row = union_in[(size_t) tile * (uint) args.cap_out + u] * ratio + (j % ratio);
        if (row >= (uint) args.len) {
            row = 0;
        }
    }
    const uint vec_count = (uint) args.head_dim / 4;
    const size_t dst_off = (((size_t) tile * (uint) args.heads + h) * s + j) * (uint) args.head_dim;
    const size_t src_off = (size_t) h * args.src_stride_h + (size_t) row * args.src_stride_r;
    device const half4 * kin = (device const half4 *) (k_src + src_off);
    device const half4 * vin = (device const half4 *) (v_src + src_off);
    device half4 * kout = (device half4 *) (k_dst + dst_off);
    device half4 * vout = (device half4 *) (v_dst + dst_off);
    for (uint v = tid3.x; v < vec_count; v += tg3.x) {
        kout[v] = kin[v];
        vout[v] = vin[v];
    }
}

// Matches dispatch.rs QsaTileMaskArgs (#[repr(C)]).
typedef struct {
    int32_t n;
    int32_t t;
    int32_t n_tiles;
    int32_t s;
    int32_t ratio;
    int32_t cap_out;
    int32_t n_kv;       // full mask row stride
} qsa_tile_mask_args;

// One thread per (column j, query row t, tile).
kernel void kernel_qsa_tile_mask(
        constant qsa_tile_mask_args & args   [[buffer(0)]],
        device const uint32_t * union_in     [[buffer(1)]],
        device const uint32_t * count        [[buffer(2)]],
        device const float * mask            [[buffer(3)]],
        device half * out                    [[buffer(4)]],
        uint3 gid [[thread_position_in_grid]]) {
    const uint j = gid.x;
    const uint tq = gid.y;
    const uint tile = gid.z;
    const uint s = (uint) args.s;
    const uint t = (uint) args.t;
    if (j >= s || tq >= t) {
        return;
    }
    const uint ratio = (uint) args.ratio;
    const uint u = j / ratio;
    float v = -INFINITY;
    if (u < count[tile]) {
        const uint col = union_in[(size_t) tile * (uint) args.cap_out + u] * ratio + (j % ratio);
        const uint row = min(tile * t + tq, (uint) args.n - 1);
        // The tail's incomplete block reaches past the row end; those columns
        // do not exist and stay hidden (the K/V gather clamps them to row 0).
        if (col < (uint) args.n_kv) {
            v = mask[(size_t) row * (uint) args.n_kv + col];
        }
    }
    out[((size_t) tile * t + tq) * s + j] = half(v);
}

// Matches dispatch.rs QsaTileQArgs (#[repr(C)]).
typedef struct {
    int32_t n;
    int32_t t;
    int32_t n_tiles;
    int32_t heads;
    int32_t head_dim;   // multiple of 4
} qsa_tile_q_args;

// One threadgroup per (query row t, tile, head): `[head, n, head_dim]` f32 in,
// `[tile, head, T, head_dim]` f16 out, RTNE like to_dtype.
kernel void kernel_qsa_tile_q(
        constant qsa_tile_q_args & args      [[buffer(0)]],
        device const float * q               [[buffer(1)]],
        device half * out                    [[buffer(2)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tid3  [[thread_position_in_threadgroup]],
        uint3 tg3   [[threads_per_threadgroup]]) {
    const uint tq = tgpig.x;
    const uint tile = tgpig.y;
    const uint h = tgpig.z;
    const uint t = (uint) args.t;
    const uint n = (uint) args.n;
    const uint hd = (uint) args.head_dim;
    const uint row = min(tile * t + tq, n - 1);
    device const float4 * in = (device const float4 *) (q + ((size_t) h * n + row) * hd);
    device half4 * o = (device half4 *) (out + (((size_t) tile * (uint) args.heads + h) * t + tq) * hd);
    for (uint v = tid3.x; v < hd / 4; v += tg3.x) {
        o[v] = half4(in[v]);
    }
}
