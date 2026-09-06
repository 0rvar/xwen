// QSA decode block selection: the top-`keep` of `nb` block scores, expanded
// into the ascending token-row list the attention gathers — on device, so the
// step never reads the scores back to the host. One threadgroup per call, one
// contiguous stripe of `ceil(nb / width)` blocks per thread.
//
// The selection it must reproduce is `QsaIndexer::top_blocks` +
// `expand_into` (src/qwen4exp/indexer.rs), bit for bit: the `keep` blocks of
// highest score under the total order (score descending, block index
// ascending), each block's `ratio` positions in order, then the `tail`
// positions `nb*ratio ..` that never formed a complete block. Ties toward the
// LOWER index are the load-bearing part — the per-head relu floors most
// scores at exactly 0 at long context, so the cut usually runs through a tie.
//
// Keys. Both arms rank by ONE function, `score_key` (indexer.rs has the host
// copy): a non-negative finite float orders by its bit pattern, denormals
// included, so the integer order IS the float order for every score the relu
// can produce; a set sign bit (-0.0) or a NaN keys as 0. The host comparator
// is built on the same key, so the two arms name the same rows for EVERY
// input, contract or not — a NaN is a tie with the zero-scored blocks on both
// sides rather than a partial-order wildcard on one.
//
// Threshold. Radix select, MSB-first, 8 bits per pass: a threadgroup
// histogram over the keys matching the prefix fixed so far, a walk from bin
// 255 downward to the bin where the cumulative count reaches what is still
// needed, fix that digit, repeat. Four passes give the threshold key `T` and
// `need_eq` = how many of the keys EQUAL to `T` complete the `keep` (>= 1);
// every key above `T` is kept outright.
//
// Compaction. Each thread counts its stripe's `> T` and `== T` keys; an
// exclusive scan of the equal counts ranks every equal key in index order, so
// a thread keeps the first `need_eq - eq_prefix` (clamped) of its own; a
// second scan of `c_gt + selected_eq` gives each thread its first output
// slot. Stripes are contiguous and in thread order, so slot order is block
// index order and the rows come out ascending, as `Rows` promises.
//
// `keep == nb` (every block fits, only the tail decides anything) is an
// identity fill and takes neither the histogram nor the scans.

#include <metal_stdlib>

using namespace metal;

// Matches dispatch.rs QsaSelectArgs (#[repr(C)]).
typedef struct {
    int32_t nb;     // block count (score count), >= 1
    int32_t keep;   // blocks to keep, 1..=nb
    int32_t ratio;  // positions per block
    int32_t tail;   // positions after the last complete block, < ratio
} qsa_select_args;

// The widest threadgroup this kernel is dispatched with; `simd_tot` is sized
// for it (1024 / 32 simdgroups). dispatch.rs caps the width to match.
#define QSA_SELECT_MAX_THREADS 1024
#define QSA_SELECT_MAX_SIMDGROUPS (QSA_SELECT_MAX_THREADS / 32)

// The SAME function as indexer.rs `score_key`; keep them identical. Bit work
// only: no float `max`, because `max(-0.0f, 0.0f)` may return either zero
// (and -0.0 reinterpreted is the LARGEST key), and no flush-to-zero compare
// on the value, which would rank a denormal equal to 0 where the host ranks
// it above. A set sign bit (-0.0, or a negative the relu makes impossible)
// keys as 0; so does a NaN, on both arms.
static inline uint score_key(float s) {
    const uint u = as_type<uint>(s);
    return ((u & 0x80000000u) || isnan(s)) ? 0u : u;
}

// Exclusive prefix sum of `v` across the threadgroup (width a multiple of 32,
// every thread participating). `simd_tot` is scratch; the trailing barrier
// frees it for the next call.
static inline uint tg_exclusive_scan(
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

kernel void kernel_qsa_select(
        constant qsa_select_args & args   [[buffer(0)]],
        device const float * scores       [[buffer(1)]],
        device uint32_t * rows            [[buffer(2)]],
        uint tid  [[thread_index_in_threadgroup]],
        uint3 tg3 [[threads_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint simd_tot[QSA_SELECT_MAX_SIMDGROUPS];
    threadgroup uint sh_prefix;
    threadgroup uint sh_need;

    const uint width = tg3.x;
    const uint nb = (uint) args.nb;
    // dispatch.rs refuses keep > nb; clamped again here so the identity
    // branch below can never leave output slots unwritten.
    const uint keep = min((uint) args.keep, nb);
    const uint ratio = (uint) args.ratio;
    const uint tail = (uint) args.tail;

    const uint chunk = (nb + width - 1) / width;
    const uint lo = min(tid * chunk, nb);
    const uint hi = min(lo + chunk, nb);

    // The tail: the positions above the last complete block, always visible.
    for (uint j = tid; j < tail; j += width) {
        rows[keep * ratio + j] = nb * ratio + j;
    }

    if (keep >= nb) {
        for (uint b = lo; b < hi; b++) {
            for (uint j = 0; j < ratio; j++) {
                rows[b * ratio + j] = b * ratio + j;
            }
        }
        return;
    }

    // ---- threshold: MSB-first radix select over the 32-bit keys ----
    uint prefix = 0;
    uint mask = 0;
    uint need = keep;
    for (uint pass = 0; pass < 4; pass++) {
        const uint shift = 24 - 8 * pass;
        for (uint i = tid; i < 256; i += width) {
            atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint b = lo; b < hi; b++) {
            const uint key = score_key(scores[b]);
            if ((key & mask) == prefix) {
                atomic_fetch_add_explicit(&hist[(key >> shift) & 0xFFu], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            // Walk from the highest digit down. `need` keys are still wanted
            // among those matching `prefix`, and at least that many match
            // (invariant from the previous pass; `keep <= nb` at the first),
            // so the walk always lands on a bin. Every key with a larger
            // digit is kept outright, which is what `need -= acc` records.
            uint acc = 0;
            uint digit = 0;
            for (int bin = 255; bin >= 0; bin--) {
                const uint c = atomic_load_explicit(&hist[bin], memory_order_relaxed);
                if (acc + c >= need) {
                    digit = (uint) bin;
                    need -= acc;
                    break;
                }
                acc += c;
            }
            sh_prefix = prefix | (digit << shift);
            sh_need = need;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = sh_prefix;
        need = sh_need;
        mask |= 0xFFu << shift;
    }
    const uint T = prefix;
    const uint need_eq = need;

    // ---- compaction ----
    uint c_gt = 0;
    uint c_eq = 0;
    for (uint b = lo; b < hi; b++) {
        const uint key = score_key(scores[b]);
        c_gt += (key > T) ? 1u : 0u;
        c_eq += (key == T) ? 1u : 0u;
    }
    const uint eq_prefix = tg_exclusive_scan(c_eq, simd_tot, lane, sg);
    const uint sel_eq = (eq_prefix >= need_eq) ? 0u : min(c_eq, need_eq - eq_prefix);
    uint slot = tg_exclusive_scan(c_gt + sel_eq, simd_tot, lane, sg);

    uint taken = 0;
    for (uint b = lo; b < hi; b++) {
        const uint key = score_key(scores[b]);
        bool emit = key > T;
        if (key == T && taken < sel_eq) {
            emit = true;
            taken++;
        }
        if (emit) {
            for (uint j = 0; j < ratio; j++) {
                rows[slot * ratio + j] = b * ratio + j;
            }
            slot++;
        }
    }
}

// ---------------------------------------------------------------------------
// QSA prefill selection + mask: one threadgroup per QUERY of a chunk, writing
// that query's row of the `[n, n_kv]` additive f32 mask the attention consumes
// (`-inf` everywhere, `0` at the selected positions) straight from its row of
// the `[n, n_blocks]` score plane. The selection per row is the decode kernel's
// exactly — same keys, same radix threshold, same tie rank — with the row's
// own `nb` (query `i` at absolute position `pos + i` sees `(pos + i + 1) /
// ratio` complete blocks) and `keep = min(keep_max, nb)`; the compaction scan
// is not needed, since a mask is written by POSITION, not by slot. It must
// reproduce `QsaIndexer::top_blocks` + `expand_into` + the host fill bit for
// bit (`device_mask_matches_host_mask_bitwise`).
//
// A row is written in two orderings: `-inf` over the whole row first, then
// zeros over the tail and the selected blocks, with a device-memory barrier
// between so no thread's background store lands over another's zero.

// Matches dispatch.rs QsaSelectMaskArgs (#[repr(C)]).
typedef struct {
    int32_t n_blocks;  // score row stride: blocks scored per query, >= 1
    int32_t n_kv;      // mask row stride: pos + n
    int32_t pos;       // absolute position of query 0
    int32_t ratio;     // positions per block
    int32_t keep_max;  // budget / ratio
} qsa_select_mask_args;

kernel void kernel_qsa_select_mask(
        constant qsa_select_mask_args & args [[buffer(0)]],
        device const float * scores          [[buffer(1)]],
        device float * mask                  [[buffer(2)]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint tid   [[thread_index_in_threadgroup]],
        uint3 tg3  [[threads_per_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]],
        uint sg    [[simdgroup_index_in_threadgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint simd_tot[QSA_SELECT_MAX_SIMDGROUPS];
    threadgroup uint sh_prefix;
    threadgroup uint sh_need;

    const uint width = tg3.x;
    const uint i = tgid.x;
    const uint n_blocks = (uint) args.n_blocks;
    const uint n_kv = (uint) args.n_kv;
    const uint ratio = (uint) args.ratio;
    const uint visible = (uint) args.pos + i + 1;
    const uint nb = min(visible / ratio, n_blocks);
    const uint keep = min((uint) args.keep_max, nb);

    device const float * row_scores = scores + (size_t) i * n_blocks;
    device float * row = mask + (size_t) i * n_kv;

    for (uint t = tid; t < n_kv; t += width) {
        row[t] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // The tail: the positions above the last complete block, always visible.
    for (uint t = visible - visible % ratio + tid; t < visible; t += width) {
        row[t] = 0.0f;
    }
    if (keep == 0) {
        return;
    }

    const uint chunk = (nb + width - 1) / width;
    const uint lo = min(tid * chunk, nb);
    const uint hi = min(lo + chunk, nb);

    if (keep >= nb) {
        for (uint t = lo * ratio; t < hi * ratio; t++) {
            row[t] = 0.0f;
        }
        return;
    }

    // ---- threshold: MSB-first radix select, as in kernel_qsa_select ----
    uint prefix = 0;
    uint kmask = 0;
    uint need = keep;
    for (uint pass = 0; pass < 4; pass++) {
        const uint shift = 24 - 8 * pass;
        for (uint b = tid; b < 256; b += width) {
            atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint b = lo; b < hi; b++) {
            const uint key = score_key(row_scores[b]);
            if ((key & kmask) == prefix) {
                atomic_fetch_add_explicit(&hist[(key >> shift) & 0xFFu], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint acc = 0;
            uint digit = 0;
            for (int bin = 255; bin >= 0; bin--) {
                const uint c = atomic_load_explicit(&hist[bin], memory_order_relaxed);
                if (acc + c >= need) {
                    digit = (uint) bin;
                    need -= acc;
                    break;
                }
                acc += c;
            }
            sh_prefix = prefix | (digit << shift);
            sh_need = need;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = sh_prefix;
        need = sh_need;
        kmask |= 0xFFu << shift;
    }
    const uint T = prefix;
    const uint need_eq = need;

    // ---- emit: every key above T, and the first need_eq keys equal to it
    // in block-index order ----
    uint c_eq = 0;
    for (uint b = lo; b < hi; b++) {
        c_eq += (score_key(row_scores[b]) == T) ? 1u : 0u;
    }
    const uint eq_prefix = tg_exclusive_scan(c_eq, simd_tot, lane, sg);
    const uint sel_eq = (eq_prefix >= need_eq) ? 0u : min(c_eq, need_eq - eq_prefix);

    uint taken = 0;
    for (uint b = lo; b < hi; b++) {
        const uint key = score_key(row_scores[b]);
        bool emit = key > T;
        if (key == T && taken < sel_eq) {
            emit = true;
            taken++;
        }
        if (emit) {
            for (uint j = 0; j < ratio; j++) {
                row[b * ratio + j] = 0.0f;
            }
        }
    }
}
