#include <metal_stdlib>
using namespace metal;

// Achievable-bandwidth probes. Not model math: they exist so bytes-moved can be
// priced against a MEASURED streaming rate for this machine instead of the
// spec-sheet figure (see `ops::bandwidth`). `n4` is the element count in float4s.
// Every thread walks the buffer with a grid stride of `groups * 256` threads,
// consecutive threads touching consecutive float4s, with four independent loads
// in flight so the loop is bandwidth-bound rather than latency-bound.
struct BwArgs {
    uint n4;
    uint groups;
};

kernel void kernel_bw_read(constant BwArgs& a [[buffer(0)]],
                           device const float4* src [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           uint tid [[thread_position_in_grid]],
                           uint tg [[threadgroup_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint stride = a.groups * 256u;
    float4 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    uint i = tid;
    for (; i + 3u * stride < a.n4; i += 4u * stride) {
        acc0 += src[i];
        acc1 += src[i + stride];
        acc2 += src[i + 2u * stride];
        acc3 += src[i + 3u * stride];
    }
    for (; i < a.n4; i += stride) {
        acc0 += src[i];
    }
    const float4 acc = (acc0 + acc1) + (acc2 + acc3);
    const float s = simd_sum((acc.x + acc.y) + (acc.z + acc.w));
    threadgroup float part[8];
    if (lane == 0u) {
        part[sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float t = 0.0f;
        for (uint j = 0u; j < 8u; j++) {
            t += part[j];
        }
        out[tg] = t;
    }
}

kernel void kernel_bw_copy(constant BwArgs& a [[buffer(0)]],
                           device const float4* src [[buffer(1)]],
                           device float4* dst [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {
    const uint stride = a.groups * 256u;
    for (uint i = tid; i < a.n4; i += stride) {
        dst[i] = src[i];
    }
}
