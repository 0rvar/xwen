// Single-row (decode) attention with GQA grouping and per-head sinks, derived
// from candle's vendored MLX `sdpa_vector` / `sdpa_vector_2pass`
// (candle-metal-kernels scaled_dot_product_attention.metal, MIT license; MLX is
// Copyright © 2023-2024 Apple Inc.). The per-key online softmax with one
// simdgroup per key, 4 head dims per lane and a simd_sum for the dot product
// is theirs; the deviations, all deliberate:
//
//  - GQA-grouped. One threadgroup serves one KV head and all G query heads
//    that read it, so each K/V row is loaded once and dotted against G
//    queries, where sdpa_vector reloads it once per query head.
//
//  - Head-strided f16 K/V: rows are 128 contiguous halves, the head stride is
//    an argument, so the cache's capacity views and the sliding ring are read
//    in place. Q is f32, every accumulator f32, the output f32.
//
//  - Split-K over a variable number of splits. Grid (n_split, n_kv): split s
//    covers keys [s*chunk, min(n_keys, (s+1)*chunk)). With one split the
//    kernel finishes the row itself; otherwise it writes its partial output,
//    max and sum, and `attn_decode_combine` merges the splits.
//
//  - A learned sink per query head: a raw logit (not scaled by 1/√d) that
//    joins the softmax denominator and attends to nothing. It enters only at
//    the end, as one more term of the final max-and-sum merge.
//
//  - No mask. A decode row sees every key of the view it is handed.
//
//  - log2 domain: q is premultiplied by scale·log2(e) and the sink by log2(e),
//    so every exponential is exp2. EMPTY is a finite floor in place of -inf
//    (fast math assumes no infinities): every exponent taken is x - M with
//    M >= x and both >= EMPTY, so it is finite and <= 0, and a split or
//    simdgroup that saw no key carries l = 0 and o = 0 into the merge.
//
// Compiled math_mode(fast), like candle's metallib.

#include <metal_stdlib>
using namespace metal;

constant constexpr int D = 128;     // head dim
constant constexpr int EPT = 4;     // head dims per lane, D / 32
constant constexpr int NSG = 4;     // simdgroups per threadgroup
constant constexpr float EMPTY = -3.0e38f;
constant constexpr float LOG2E = 1.44269504088896340736f;

// Matches dispatch.rs AttnDecodeArgs (#[repr(C)]): six 4-byte fields (24
// bytes, a multiple of 8), then two i64 ELEMENT strides between KV heads.
typedef struct {
    int n_keys;
    int n_split;
    int chunk;        // keys per split
    int has_sinks;
    float scale_log2; // scale * log2(e)
    int pad;
    int64_t k_stride_h;
    int64_t v_stride_h;
} attn_decode_args;

// The sink of query head h in the log2 domain, or EMPTY without sinks.
static inline float sink_log2(constant attn_decode_args& a, const device float* sinks, int h) {
    return a.has_sinks ? sinks[h] * LOG2E : EMPTY;
}

template <int G>
[[kernel, max_total_threads_per_threadgroup(128)]] void attn_decode(
    constant attn_decode_args& a [[buffer(0)]],
    const device float* Q [[buffer(1)]],     // [n_kv * G, 128]
    const device half* K [[buffer(2)]],      // head h at K + h * k_stride_h
    const device half* V [[buffer(3)]],
    const device float* sinks [[buffer(4)]], // [n_kv * G]
    device float* O [[buffer(5)]],           // one split: [n_head, 128] final;
                                             // else [n_head, n_split, 128]
    device float* ML [[buffer(6)]],          // [n_head, n_split, 2]: max, sum
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  const int split = int(tg.x);
  const int kvh = int(tg.y);
  const int h0 = kvh * G;

  float4 q[G];
  for (int g = 0; g < G; g++) {
    q[g] = *(const device float4*)(Q + (h0 + g) * D + lane * EPT) * a.scale_log2;
  }
  float m[G];
  float l[G];
  float4 o[G];
  for (int g = 0; g < G; g++) {
    m[g] = EMPTY;
    l[g] = 0.0f;
    o[g] = float4(0.0f);
  }

  const int k_begin = split * a.chunk;
  const int k_end = min(a.n_keys, k_begin + a.chunk);
  const device half* kp = K + int64_t(kvh) * a.k_stride_h + lane * EPT;
  const device half* vp = V + int64_t(kvh) * a.v_stride_h + lane * EPT;

  for (int i = k_begin + int(sg); i < k_end; i += NSG) {
    const float4 k = float4(*(const device half4*)(kp + int64_t(i) * D));
    const float4 v = float4(*(const device half4*)(vp + int64_t(i) * D));
    for (int g = 0; g < G; g++) {
      const float s = simd_sum(dot(q[g], k));
      const float mn = max(m[g], s);
      const float f = fast::exp2(m[g] - mn);
      const float p = fast::exp2(s - mn);
      m[g] = mn;
      l[g] = l[g] * f + p;
      o[g] = o[g] * f + p * v;
    }
  }

  // Merge the simdgroups of this threadgroup.
  threadgroup float tg_m[NSG][G];
  threadgroup float tg_l[NSG][G];
  threadgroup float tg_o[NSG][G][D];
  for (int g = 0; g < G; g++) {
    if (lane == 0) {
      tg_m[sg][g] = m[g];
      tg_l[sg][g] = l[g];
    }
    *(threadgroup float4*)(&tg_o[sg][g][lane * EPT]) = o[g];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int idx = int(tid); idx < G * D; idx += NSG * 32) {
    const int g = idx / D;
    const int d = idx % D;
    float mx = EMPTY;
    for (int s = 0; s < NSG; s++) {
      mx = max(mx, tg_m[s][g]);
    }
    float sum = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < NSG; s++) {
      const float f = fast::exp2(tg_m[s][g] - mx);
      sum += tg_l[s][g] * f;
      acc += tg_o[s][g][d] * f;
    }
    const int h = h0 + g;
    if (a.n_split == 1) {
      const float sk = sink_log2(a, sinks, h);
      const float mt = max(mx, sk);
      const float f = fast::exp2(mx - mt);
      const float denom = sum * f + (a.has_sinks ? fast::exp2(sk - mt) : 0.0f);
      O[h * D + d] = acc * f / denom;
    } else {
      O[(int64_t(h) * a.n_split + split) * D + d] = acc;
      if (d == 0) {
        ML[(h * a.n_split + split) * 2 + 0] = mx;
        ML[(h * a.n_split + split) * 2 + 1] = sum;
      }
    }
  }
}

// Merge the splits of `attn_decode` and fold the sink in. Grid (n_head), one
// thread per head dim.
[[kernel, max_total_threads_per_threadgroup(128)]] void attn_decode_combine(
    constant attn_decode_args& a [[buffer(0)]],
    const device float* P [[buffer(1)]],     // [n_head, n_split, 128]
    const device float* ML [[buffer(2)]],    // [n_head, n_split, 2]
    const device float* sinks [[buffer(3)]], // [n_head]
    device float* O [[buffer(4)]],           // [n_head, 128]
    uint h [[threadgroup_position_in_grid]],
    uint d [[thread_index_in_threadgroup]]) {
  const device float* ml = ML + int64_t(h) * a.n_split * 2;
  const device float* p = P + int64_t(h) * a.n_split * D + d;
  const float sk = sink_log2(a, sinks, int(h));
  float mx = sk;
  for (int s = 0; s < a.n_split; s++) {
    mx = max(mx, ml[s * 2]);
  }
  float sum = a.has_sinks ? fast::exp2(sk - mx) : 0.0f;
  float acc = 0.0f;
  for (int s = 0; s < a.n_split; s++) {
    const float f = fast::exp2(ml[s * 2] - mx);
    sum += ml[s * 2 + 1] * f;
    acc += p[s * D] * f;
  }
  O[int64_t(h) * D + d] = acc / sum;
}

#define instantiate_attn_decode(g)                                        \
  template [[host_name("kernel_attn_decode_g" #g)]] [[kernel]]             \
      decltype(attn_decode<g>) attn_decode<g>;

instantiate_attn_decode(1)
instantiate_attn_decode(2)
instantiate_attn_decode(4)
instantiate_attn_decode(8)
