// QSA decode row gather: pack the K (or V) rows a QSA selection names out of a
// `[heads, len, head_dim]` cache view into a contiguous `[heads, n_sel,
// head_dim]` plane. One threadgroup per (selected row, head); each thread
// copies 4-element vectors, striding across the row. A pure copy — the output
// bits are the input bits — so the kernel is bit-identical to the per-head
// `index_select` + `stack` chain it replaces (`XWEN_QSA_CLASSIC`).
//
// The source is a cache view: rows within a head are contiguous (`head_dim`
// apart) while the head axis carries the cache's `max_ctx` gap, which is why
// the strides arrive as arguments instead of being derived from the shape.
//
// A row index at or beyond `len` names a slot the cache has not written;
// rather than read it, the kernel writes zeros for that row. The host never
// produces one (the indexer's rows are all below the position it selected
// at), so this is a guard against garbage, not a semantics.

#include <metal_stdlib>

using namespace metal;

// Matches dispatch.rs QsaGatherArgs (#[repr(C)]).
typedef struct {
    int32_t heads;
    int32_t len;
    int32_t head_dim;   // multiple of 4
    int32_t n_sel;
    int64_t src_stride_h; // elements between heads in the source
    int64_t src_stride_r; // elements between rows in the source
} qsa_gather_args;

template <typename T>
METAL_FUNC void qsa_gather_impl(
        constant qsa_gather_args & args,
        device const T * src,
        device const uint32_t * rows,
        device       T * dst,
        uint3 tgpig,
        uint3 tid3,
        uint3 tg3) {
    const uint tid = tid3.x;
    const uint tg_width = tg3.x;
    const int sel = (int) tgpig.x;
    const int h = (int) tgpig.y;
    if (sel >= args.n_sel || h >= args.heads) {
        return;
    }
    const int row = (int) rows[sel];
    const int vec_count = args.head_dim / 4;
    device T * out = dst + ((int64_t) h * args.n_sel + sel) * (int64_t) args.head_dim;
    if (row < 0 || row >= args.len) {
        for (int v = (int) tid; v < vec_count; v += (int) tg_width) {
            ((device vec<T, 4> *) out)[v] = vec<T, 4>(0);
        }
        return;
    }
    device const T * in = src + (int64_t) h * args.src_stride_h + (int64_t) row * args.src_stride_r;
    for (int v = (int) tid; v < vec_count; v += (int) tg_width) {
        ((device vec<T, 4> *) out)[v] = ((device const vec<T, 4> *) in)[v];
    }
}

#define QSA_GATHER_KERNEL(NAME, T)                                             \
kernel void NAME(                                                              \
        constant qsa_gather_args & args   [[buffer(0)]],                       \
        device const T * src              [[buffer(1)]],                       \
        device const uint32_t * rows      [[buffer(2)]],                       \
        device       T * dst              [[buffer(3)]],                       \
        uint3 tgpig [[threadgroup_position_in_grid]],                          \
        uint3 tid   [[thread_position_in_threadgroup]],                        \
        uint3 tg    [[threads_per_threadgroup]]) {                             \
    qsa_gather_impl<T>(args, src, rows, dst, tgpig, tid, tg);                  \
}

QSA_GATHER_KERNEL(kernel_qsa_gather_f16, half)
QSA_GATHER_KERNEL(kernel_qsa_gather_f32, float)
