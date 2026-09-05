// PLE tail: normalized key and shared value arrive from the projections.
// Reductions partition the frozen oracle's sums; scalar products keep their
// original rounding and order. Safe math mode (not the fast mode every other
// library here uses) keeps the isnan guard in the gate, which fast mode would
// fold away and with it the oracle's NaN-propagation contract.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp math_mode(safe)

typedef struct {
    int n, hidden, width, k, dilation, state_len;
    float eps, dot_scale;
} ple_args;

static inline float ple_sum(float v, threadgroup float *partial,
                            uint tid, uint lane, uint sg, uint threads) {
    float sum = simd_sum(v);
    if (lane == 0) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = simd_sum(tid < threads / 32 ? partial[tid] : 0.0f);
    if (tid == 0) partial[0] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result = partial[0];
    // Reuse fence: the next call writes partial[sg] again, so every thread must
    // have read partial[0] before any simdgroup gets there.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

kernel void kernel_ple_gate(
    constant ple_args &a [[buffer(0)]],
    device const float *key [[buffer(1)]],
    device const float *value [[buffer(2)]],
    device const float *stream [[buffer(3)]],
    device const float *query_w [[buffer(4)]],
    device const float *norm_w [[buffer(5)]],
    device float *gated [[buffer(6)]],
    device float *normed [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 thread_shape [[threads_per_threadgroup]]) {
    #pragma clang fp contract(off) reassociate(off)
    // Sized for the most simdgroups a threadgroup can hold, not for the 256-thread
    // launch dispatch.rs chooses (8 simdgroups), so the two cannot drift apart.
    threadgroup float partial[32];
    uint threads = thread_shape.x;
    int base = group.x * a.width + group.y * a.hidden;
    int wb = group.y * a.hidden;
    float ss = 0.0f;
    for (uint j = tid; j < uint(a.hidden); j += threads) {
        float x = stream[base+j];
        ss += x*x;
    }
    float scale = 1.0f / sqrt(ple_sum(ss,partial,tid,lane,sg,threads) / float(a.hidden) + a.eps);
    float dot = 0.0f;
    for (uint j = tid; j < uint(a.hidden); j += threads) {
        float q = stream[base+j] * scale;
        q = q * query_w[wb+j];
        dot += key[base+j] * q;
    }
    float raw = ple_sum(dot,partial,tid,lane,sg,threads) * a.dot_scale;
    float sign = raw > 0.0f ? 1.0f : (raw < 0.0f ? -1.0f : 0.0f);
    float signed_root = isnan(raw) ? raw : sign * sqrt(max(abs(raw),1e-6f));
    float gate = 1.0f / (1.0f + exp(-signed_root));
    ss = 0.0f;
    for (uint j = tid; j < uint(a.hidden); j += threads) {
        float g = gate * value[group.x*a.hidden+j];
        gated[base+j] = g;
        ss += g*g;
    }
    scale = 1.0f / sqrt(ple_sum(ss,partial,tid,lane,sg,threads) / float(a.hidden) + a.eps);
    for (uint j = tid; j < uint(a.hidden); j += threads) {
        float g = gate * value[group.x*a.hidden+j];
        normed[base+j] = (g*scale) * norm_w[wb+j];
    }
}

kernel void kernel_ple_conv(
    constant ple_args &a [[buffer(0)]],
    device const float *gated [[buffer(1)]],
    device const float *normed [[buffer(2)]],
    device const float *prior [[buffer(3)]],
    device const float *weight [[buffer(4)]],
    device float *out [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    #pragma clang fp contract(off) reassociate(off)
    if (index >= uint(a.n*a.width)) return;
    int t = index / a.width;
    int c = index % a.width;
    float acc = 0.0f;
    for (int j = 0; j < a.k; ++j) {
        int pos = t - (a.k - 1 - j)*a.dilation;
        float x = pos < 0 ? prior[c*a.state_len + a.state_len + pos] : normed[pos*a.width+c];
        acc += weight[c*a.k+j] * x;
    }
    float sigmoid = 1.0f / (1.0f + exp(-acc));
    out[index] = gated[index] + acc*sigmoid;
}
