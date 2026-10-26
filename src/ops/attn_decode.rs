//! Host side of the single-row (decode) attention kernel (attn_decode.metal,
//! derived from candle's `sdpa_vector`): one query row per head against every
//! key of a cache view, GQA-grouped so each K/V row is read once per KV head,
//! with an optional learned sink per query head and a split-K form for long
//! contexts. The tests grade it against Flint's frozen f32 oracle,
//! [`crate::flint::attention::reference_attention`], on identical operands.

use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

/// `softmax([q kᵀ·scale | sink]) [v | 0]` for one query row per head. `q` is
/// `[n_head, 1, 128]` f32 contiguous; `k`/`v` are `[n_kv, K, 128]` f16 cache
/// views with contiguous rows and any head stride (a full layer's capacity
/// view, a sliding layer's ring, read in place); `sinks` is `[n_head]` f32 raw
/// logits (not multiplied by `scale`), or `None` for none. The row sees every
/// key of the view and the order of the keys does not matter, which is what
/// lets an unordered ring be passed as it is. Query head `h` reads KV head
/// `h / (n_head / n_kv)`, for 1, 2, 4 or 8 query heads per KV head. The key
/// range is split across threadgroups by `attn_decode_splits` and merged in a
/// second pass. Returns `[n_head, 1, 128]` f32. Metal only.
pub fn attn_decode(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: Option<&Tensor>,
    scale: f32,
) -> Result<Tensor> {
    dispatch::run_attn_decode(q, k, v, sinks, scale, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flint::attention::reference_attention;
    use crate::gguf::metal_device;
    use candle_core::{DType, Device};

    const HD: usize = 128;

    /// Deterministic pseudo-random f32s in [lo, hi] (xorshift, no deps).
    fn rand(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = (s >> 11) as f64 / (1u64 << 53) as f64;
                lo + (hi - lo) * u as f32
            })
            .collect()
    }

    fn rand_t(seed: u64, dims: (usize, usize, usize), dev: &Device) -> Tensor {
        Tensor::from_vec(rand(seed, dims.0 * dims.1 * dims.2, -2.0, 2.0), dims, dev).unwrap()
    }

    /// K/V drawn f32 and rounded through f16, the cache's dtype.
    fn rand_kv(seed: u64, dims: (usize, usize, usize), dev: &Device) -> Tensor {
        rand_t(seed, dims, dev).to_dtype(DType::F16).unwrap()
    }

    /// Sinks spanning the shipped file's range: a head switched nearly off at
    /// 10, the rest spread below it.
    fn sinks_t(n_head: usize, seed: u64, dev: &Device) -> Tensor {
        let mut s = rand(seed, n_head, -4.0, 8.0);
        s[0] = 10.0;
        Tensor::from_vec(s, n_head, dev).unwrap()
    }

    fn vanishing_sinks(n_head: usize, dev: &Device) -> Tensor {
        Tensor::from_vec(vec![-1e30f32; n_head], n_head, dev).unwrap()
    }

    /// Worst error of `got` against `want` relative to each output row's
    /// largest magnitude, so rows that a large sink shrinks are graded at
    /// their own scale. Returns (worst row-relative, worst absolute).
    pub(crate) fn row_relative_error(got: &Tensor, want: &Tensor) -> (f32, f32) {
        assert_eq!(got.dims(), want.dims());
        let hd = *got.dims().last().unwrap();
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let w: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        let (mut rel, mut abs) = (0f32, 0f32);
        for (gr, wr) in g.chunks(hd).zip(w.chunks(hd)) {
            let scale = wr.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30);
            for (a, b) in gr.iter().zip(wr) {
                assert!(a.is_finite(), "non-finite output {a}");
                let e = (a - b).abs();
                abs = abs.max(e);
                rel = rel.max(e / scale);
            }
        }
        (rel, abs)
    }

    fn assert_bits_eq(got: &Tensor, want: &Tensor, label: &str) {
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let w: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(g.len(), w.len(), "{label}: length");
        for (i, (a, b)) in g.iter().zip(&w).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label}: element {i} ({a} vs {b})"
            );
        }
    }

    /// Row-relative tolerance against the f32 oracle. Both sides accumulate
    /// in f32 from identical f16 K/V; they differ in order (per-simdgroup
    /// online softmax and a split merge against one matmul and one softmax)
    /// and in exp2-of-prescaled against exp, which leaves a few ulps per
    /// term. The error grows with the keys one accumulator sums: under 1.3e-6
    /// at the 513-key ring and up to 9.4e-6 at 32k keys in one split, where a
    /// simdgroup sums 8k of them. Every case prints its measured worst.
    const TOL: f32 = 2e-5;

    /// The Flint geometry (32 query heads over 4 KV heads of 128) on a NoPE
    /// full layer past 32k keys, read through a head-strided capacity view,
    /// at the default split count and at several forced ones, including more
    /// splits than a split has keys to spare, with realistic sinks and with
    /// sinks that vanish.
    #[test]
    fn attn_decode_matches_the_oracle_on_a_long_full_layer() {
        let dev = metal_device().unwrap();
        let (nh, nkv) = (32usize, 4usize);
        let keys = 32 * 1024 + 37;
        let cap = 40 * 1024;
        let scale = 1.0 / (HD as f32).sqrt();
        let q = rand_t(1, (nh, 1, HD), &dev);
        let k = rand_kv(2, (nkv, cap, HD), &dev).narrow(1, 0, keys).unwrap();
        let v = rand_kv(3, (nkv, cap, HD), &dev).narrow(1, 0, keys).unwrap();
        assert!(!k.is_contiguous(), "the view must be head-strided");
        let mut worst = 0f32;
        for (label, sinks) in [
            ("sinks", sinks_t(nh, 4, &dev)),
            ("sinks -1e30", vanishing_sinks(nh, &dev)),
        ] {
            let want = reference_attention(&q, &k, &v, None, &sinks, HD).unwrap();
            let default = dispatch::attn_decode_splits(keys, nkv);
            for split in [None, Some(1), Some(7), Some(64), Some(128)] {
                let got =
                    dispatch::run_attn_decode(&q, &k, &v, Some(&sinks), scale, split).unwrap();
                let (rel, abs) = row_relative_error(&got, &want);
                println!(
                    "full {keys} keys, {label}, split {}: row-rel {rel:.2e}, abs {abs:.2e}",
                    split.unwrap_or(default)
                );
                assert!(rel < TOL, "{label} split {split:?}: row-rel {rel:.2e}");
                worst = worst.max(rel);
            }
        }
        println!("worst row-relative error {worst:.2e}");
    }

    /// A sliding layer's decode view is the whole ring, unordered once it has
    /// wrapped. The kernel reads it as it is and must match the oracle over
    /// the same view, including fewer keys than splits.
    #[test]
    fn attn_decode_matches_the_oracle_on_a_ring() {
        let dev = metal_device().unwrap();
        let (nh, nkv, w) = (32usize, 4usize, 513usize);
        let scale = 1.0 / (HD as f32).sqrt();
        for (keys, seed) in [(w, 10u64), (200, 20), (3, 30), (1, 40)] {
            let q = rand_t(seed, (nh, 1, HD), &dev);
            let ring_k = rand_kv(seed + 1, (nkv, w, HD), &dev);
            let ring_v = rand_kv(seed + 2, (nkv, w, HD), &dev);
            let k = ring_k.narrow(1, 0, keys).unwrap();
            let v = ring_v.narrow(1, 0, keys).unwrap();
            let sinks = sinks_t(nh, seed + 3, &dev);
            let want = reference_attention(&q, &k, &v, None, &sinks, HD).unwrap();
            for split in [None, Some(1), Some(2), Some(8)] {
                let got =
                    dispatch::run_attn_decode(&q, &k, &v, Some(&sinks), scale, split).unwrap();
                let (rel, abs) = row_relative_error(&got, &want);
                println!("ring {keys} keys, split {split:?}: row-rel {rel:.2e}, abs {abs:.2e}");
                assert!(rel < TOL, "{keys} keys split {split:?}: row-rel {rel:.2e}");
            }
        }
    }

    /// Without sinks the kernel is plain softmax attention, which the oracle
    /// computes with sinks that vanish; and a head-strided view reads exactly
    /// what its packed copy holds, bit for bit.
    #[test]
    fn attn_decode_without_sinks_and_on_strided_views() {
        let dev = metal_device().unwrap();
        let (nh, nkv, keys) = (32usize, 4usize, 3000usize);
        let scale = 1.0 / (HD as f32).sqrt();
        let q = rand_t(50, (nh, 1, HD), &dev);
        let k_big = rand_kv(51, (nkv, keys + 300, HD), &dev);
        let v_big = rand_kv(52, (nkv, keys + 300, HD), &dev);
        let k = k_big.narrow(1, 17, keys).unwrap();
        let v = v_big.narrow(1, 17, keys).unwrap();
        let want = reference_attention(&q, &k, &v, None, &vanishing_sinks(nh, &dev), HD).unwrap();
        let got = attn_decode(&q, &k, &v, None, scale).unwrap();
        let (rel, _) = row_relative_error(&got, &want);
        println!("no sinks: row-rel {rel:.2e}");
        assert!(rel < TOL, "no sinks: row-rel {rel:.2e}");

        let sinks = sinks_t(nh, 53, &dev);
        for split in [Some(1), None] {
            let strided =
                dispatch::run_attn_decode(&q, &k, &v, Some(&sinks), scale, split).unwrap();
            let packed = dispatch::run_attn_decode(
                &q,
                &k.contiguous().unwrap(),
                &v.contiguous().unwrap(),
                Some(&sinks),
                scale,
                split,
            )
            .unwrap();
            assert_bits_eq(&strided, &packed, "head-strided vs packed");
        }
    }

    /// Every GQA ratio the kernel instantiates matches the oracle.
    #[test]
    fn attn_decode_covers_each_group_size() {
        let dev = metal_device().unwrap();
        let scale = 1.0 / (HD as f32).sqrt();
        for (nh, nkv) in [(4usize, 4usize), (8, 4), (8, 2), (16, 2)] {
            let keys = 777;
            let q = rand_t(60 + nh as u64, (nh, 1, HD), &dev);
            let k = rand_kv(61, (nkv, keys, HD), &dev);
            let v = rand_kv(62, (nkv, keys, HD), &dev);
            let sinks = sinks_t(nh, 63, &dev);
            let want = reference_attention(&q, &k, &v, None, &sinks, HD).unwrap();
            let got = attn_decode(&q, &k, &v, Some(&sinks), scale).unwrap();
            let (rel, _) = row_relative_error(&got, &want);
            assert!(rel < TOL, "{nh}/{nkv}: row-rel {rel:.2e}");
        }
    }

    #[test]
    fn attn_decode_rejects_bad_inputs() {
        let dev = metal_device().unwrap();
        let q = rand_t(70, (8, 1, HD), &dev);
        let k = rand_kv(71, (2, 9, HD), &dev);
        let v = rand_kv(72, (2, 9, HD), &dev);
        let sinks = sinks_t(8, 73, &dev);
        let scale = 0.1f32;
        attn_decode(&q, &k, &v, Some(&sinks), scale).unwrap();

        // Two query rows.
        let q2 = rand_t(74, (8, 2, HD), &dev);
        assert!(attn_decode(&q2, &k, &v, Some(&sinks), scale).is_err());
        // f16 q, f32 k.
        let q16 = q.to_dtype(DType::F16).unwrap();
        assert!(attn_decode(&q16, &k, &v, Some(&sinks), scale).is_err());
        let k32 = k.to_dtype(DType::F32).unwrap();
        assert!(attn_decode(&q, &k32, &v, Some(&sinks), scale).is_err());
        // Wrong head dim.
        let q64 = rand_t(75, (8, 1, 64), &dev);
        assert!(attn_decode(&q64, &k, &v, Some(&sinks), scale).is_err());
        // K and V disagree.
        let v_short = v.narrow(1, 0, 8).unwrap();
        assert!(attn_decode(&q, &k, &v_short, Some(&sinks), scale).is_err());
        // Head-dim-strided k.
        let k_t = rand_kv(76, (2, HD, HD), &dev).transpose(1, 2).unwrap();
        assert!(attn_decode(&q, &k_t, &k_t, Some(&sinks), scale).is_err());
        // Group sizes with no kernel, and heads that do not group.
        let q24 = rand_t(77, (24, 1, HD), &dev);
        let k1 = rand_kv(78, (1, 9, HD), &dev);
        assert!(attn_decode(&q24, &k1, &k1, None, scale).is_err());
        let k3 = rand_kv(79, (3, 9, HD), &dev);
        assert!(attn_decode(&q, &k3, &k3, None, scale).is_err());
        // Sinks of the wrong length or dtype.
        let short = sinks.narrow(0, 0, 4).unwrap();
        assert!(attn_decode(&q, &k, &v, Some(&short), scale).is_err());
        let s16 = sinks.to_dtype(DType::F16).unwrap();
        assert!(attn_decode(&q, &k, &v, Some(&s16), scale).is_err());
        // Split counts outside the kernel's range.
        assert!(dispatch::run_attn_decode(&q, &k, &v, None, scale, Some(0)).is_err());
        assert!(
            dispatch::run_attn_decode(
                &q,
                &k,
                &v,
                None,
                scale,
                Some(dispatch::ATTN_DECODE_MAX_SPLIT + 1)
            )
            .is_err()
        );
        // CPU operands.
        let cq = rand_t(80, (8, 1, HD), &Device::Cpu);
        let ck = rand_kv(81, (2, 9, HD), &Device::Cpu);
        assert!(attn_decode(&cq, &ck, &ck, None, scale).is_err());
    }

    /// Kernel time at Flint's decode geometry against the f32 candle chain it
    /// replaces (`reference_attention` computes what the old production arm
    /// did). Unpinned: run alone, and quote with the `pmset -g` power mode.
    /// `cargo test --release --lib attn_decode_microbench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn attn_decode_microbench() {
        let dev = metal_device().unwrap();
        let (nh, nkv) = (32usize, 4usize);
        let scale = 1.0 / (HD as f32).sqrt();
        let sync = || dev.synchronize().unwrap();
        let time = |f: &dyn Fn() -> Tensor, iters: usize| -> f64 {
            for _ in 0..3 {
                f();
            }
            sync();
            let t = std::time::Instant::now();
            for _ in 0..iters {
                f();
            }
            sync();
            t.elapsed().as_secs_f64() * 1e6 / iters as f64
        };
        for keys in [513usize, 4096, 32768] {
            let q = rand_t(1, (nh, 1, HD), &dev);
            let k = rand_kv(2, (nkv, keys, HD), &dev);
            let v = rand_kv(3, (nkv, keys, HD), &dev);
            let sinks = sinks_t(nh, 4, &dev);
            let chain = time(
                &|| reference_attention(&q, &k, &v, None, &sinks, HD).unwrap(),
                20,
            );
            // An f16 affine over K and V moves twice the bytes the kernel reads: half
            // its time is the bandwidth floor at this moment.
            let copy = time(
                &|| {
                    k.affine(1.0, 0.0).unwrap();
                    v.affine(1.0, 0.0).unwrap()
                },
                50,
            );
            let mut line = format!(
                "{keys} keys: f32 chain {chain:.1} us, read floor {:.1} us",
                copy / 2.0
            );
            for split in [1usize, 2, 4, 8, 16, 32, 64, 128] {
                if split > keys {
                    continue;
                }
                let t = time(
                    &|| {
                        dispatch::run_attn_decode(&q, &k, &v, Some(&sinks), scale, Some(split))
                            .unwrap()
                    },
                    200,
                );
                line += &format!(", split {split} {t:.1} us");
            }
            line += &format!(" (default {})", dispatch::attn_decode_splits(keys, nkv));
            println!("{line}");
        }
    }
}
