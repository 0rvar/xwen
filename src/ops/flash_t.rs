//! Host side of the cooperative-tensor bidirectional flash attention kernel
//! (flash_t.metal): the diffusion transformer's self-attention with both
//! products, Q K^T and P V, on the Metal-4 tensor ops. Same operand contract
//! as `flash_attn_bidirectional` (flash.rs), which stays the classic
//! simdgroup-matrix path and the `flash` arm of `XWEN_ZIMAGE_ATTN`; this is the
//! `tensor` arm. Numerics: the tensor ops consume the f32 operands at reduced
//! input precision (`relaxed_precision`, the gemms' setting), so unlike the
//! flash kernel this one is graded against candle's f32 sdpa at a bound rather
//! than expected to match it bitwise.

use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

pub use dispatch::FlashTGeometry;

/// Bidirectional flash attention on the cooperative-tensor ops,
/// `softmax(q·kᵀ·scale)·v` with every query seeing every key. `q` is
/// `[n_head, seq, 128]` f32 contiguous, `k`/`v` are `[n_kv, K, 128]` f16 with
/// contiguous rows (head-strided views are consumed via their strides), and
/// the result is `[n_head, seq, 128]` f32 contiguous: the layouts
/// `flash_attn_bidirectional` takes and returns, so the two are
/// interchangeable at the call site. Metal only.
pub fn flash_attn_tensor(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    dispatch::run_flash_attn_tensor(q, k, v, scale, FlashTGeometry::DEFAULT)
}

/// [`flash_attn_tensor`] on an explicit tile geometry, for the tests and the
/// microbench that pick [`FlashTGeometry::DEFAULT`].
pub fn flash_attn_tensor_with(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    geometry: FlashTGeometry,
) -> Result<Tensor> {
    dispatch::run_flash_attn_tensor(q, k, v, scale, geometry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::ops::flash_attn;
    use candle_core::{DType, Device, Tensor};

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

    fn rand_kv(seed: u64, dims: (usize, usize, usize), dev: &Device) -> Tensor {
        rand_t(seed, dims, dev).to_dtype(DType::F16).unwrap()
    }

    /// candle's unmasked f32 sdpa over the widened k/v, with the kv heads
    /// expanded the way the kernel maps them (query head h reads kv head
    /// h / gqa_factor).
    fn reference(q: &Tensor, k16: &Tensor, v16: &Tensor, scale: f32) -> Tensor {
        let (n_head, _, hd) = q.dims3().unwrap();
        let (n_kv, k_len, _) = k16.dims3().unwrap();
        let expand = |t: &Tensor| -> Tensor {
            let factor = n_head / n_kv;
            t.to_dtype(DType::F32)
                .unwrap()
                .unsqueeze(1)
                .unwrap()
                .expand((n_kv, factor, k_len, hd))
                .unwrap()
                .reshape((n_head, k_len, hd))
                .unwrap()
                .contiguous()
                .unwrap()
        };
        candle_nn::ops::sdpa(
            &q.unsqueeze(0).unwrap(),
            &expand(k16).unsqueeze(0).unwrap(),
            &expand(v16).unsqueeze(0).unwrap(),
            None,
            false,
            scale,
            1.0,
        )
        .unwrap()
        .squeeze(0)
        .unwrap()
    }

    /// (relative L2 error, max abs error) of `got` against `want`.
    fn errors(got: &Tensor, want: &Tensor) -> (f64, f32) {
        assert_eq!(got.dims(), want.dims(), "shape");
        assert_eq!(got.dtype(), DType::F32, "dtype");
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let w: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        let mut num = 0f64;
        let mut den = 0f64;
        let mut max_abs = 0f32;
        for (a, b) in g.iter().zip(w.iter()) {
            assert!(a.is_finite(), "non-finite output {a}");
            let d = a - b;
            num += (d as f64) * (d as f64);
            den += (*b as f64) * (*b as f64);
            max_abs = max_abs.max(d.abs());
        }
        ((num / den.max(1e-30)).sqrt(), max_abs)
    }

    /// The accuracy bars. Relative L2 over the whole output: 2e-3, the bar
    /// the arc set. Max abs error: inputs are uniform on [-2, 2], so every
    /// output is a convex combination of values in [-2, 2]; the f16 rounding
    /// of k and v is 2^-11 relative (at most 1e-3 on a value of 2), and the
    /// tensor core's reduced operand precision on q and p perturbs the
    /// logits by a few 1e-3, which the softmax passes through to the weights
    /// at the same order. Two times 2 times a few 1e-3 is a few 1e-2; the
    /// bar is set at 2e-2 and the measured values print beside it.
    const REL_L2_BAR: f64 = 2e-3;
    const MAX_ABS_BAR: f32 = 2e-2;

    /// The tensor kernel matches candle's unmasked f32 sdpa across the
    /// diffusion shape (30 heads of 128, 4128 tokens), an unaligned length,
    /// the minimal one, fewer queries than keys and GQA, on every
    /// instantiated geometry.
    #[test]
    fn flash_attn_tensor_matches_f32_sdpa() {
        let dev = metal_device().unwrap();
        let hd = 128usize;
        let scale = 1.0f32 / (hd as f32).sqrt();

        // (label, n_head, n_kv, seq, K).
        type Case = (&'static str, usize, usize, usize, usize);
        let cases: &[Case] = &[
            ("z-image 1024x1024", 30, 30, 4128, 4128),
            ("unaligned seq and K", 6, 6, 203, 203),
            ("seq=2", 4, 4, 2, 2),
            ("fewer queries than keys", 4, 4, 40, 64),
            ("gqa 8/2 unaligned", 8, 2, 45, 45),
        ];

        for geometry in FlashTGeometry::ALL {
            for &(label, n_head, n_kv, seq, k_len) in cases {
                let seed = (n_head * 1000 + seq * 3 + k_len) as u64;
                let q = rand_t(seed, (n_head, seq, hd), &dev);
                let k16 = rand_kv(seed + 1, (n_kv, k_len, hd), &dev);
                let v16 = rand_kv(seed + 2, (n_kv, k_len, hd), &dev);

                let got = flash_attn_tensor_with(&q, &k16, &v16, scale, geometry).unwrap();
                let want = reference(&q, &k16, &v16, scale);
                let (rel_l2, max_abs) = errors(&got, &want);
                println!(
                    "{label} {}: rel_l2 {rel_l2:.3e} (bar {REL_L2_BAR:.0e}), \
                     max abs {max_abs:.3e} (bar {MAX_ABS_BAR:.0e})",
                    geometry.label()
                );
                assert!(
                    rel_l2 <= REL_L2_BAR,
                    "{label} {geometry:?}: rel_l2 {rel_l2:.3e}"
                );
                assert!(
                    max_abs <= MAX_ABS_BAR,
                    "{label} {geometry:?}: max abs {max_abs:.3e}"
                );
            }
        }
    }

    /// Every query sees every key: against the causal flash kernel on the
    /// same inputs, the last row agrees (it sees all keys under both rules,
    /// so the two differ only by operand precision) and the first row does
    /// not (it sees one key causally and all of them here).
    #[test]
    fn flash_attn_tensor_sees_future_keys() {
        let dev = metal_device().unwrap();
        let (n_head, seq, hd) = (4usize, 48usize, 128usize);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let q = rand_t(300, (n_head, seq, hd), &dev);
        let k16 = rand_kv(301, (n_head, seq, hd), &dev);
        let v16 = rand_kv(302, (n_head, seq, hd), &dev);
        let causal = flash_attn(&q, &k16, &v16, 0, 0, None, scale).unwrap();
        let both = flash_attn_tensor(&q, &k16, &v16, scale).unwrap();
        let c: Vec<f32> = causal.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = both.flatten_all().unwrap().to_vec1().unwrap();
        let row =
            |v: &[f32], h: usize, r: usize| v[(h * seq + r) * hd..(h * seq + r + 1) * hd].to_vec();
        for h in 0..n_head {
            let last_diff = row(&c, h, seq - 1)
                .iter()
                .zip(row(&b, h, seq - 1).iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                last_diff <= MAX_ABS_BAR,
                "head {h}: the last row sees all keys under both rules, differs by {last_diff:.3e}"
            );
            let first_diff = row(&c, h, 0)
                .iter()
                .zip(row(&b, h, 0).iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                first_diff > MAX_ABS_BAR,
                "head {h}: the first row must differ once future keys are visible, differs by {first_diff:.3e}"
            );
        }
    }

    /// Head-strided k/v views (a nonzero start offset and a head-axis gap)
    /// match the packed copy bitwise: the strides feed the kernel, not a
    /// forced contiguous copy.
    #[test]
    fn flash_attn_tensor_head_strided_views() {
        let dev = metal_device().unwrap();
        let (n_head, hd, seq) = (6usize, 128usize, 70usize);
        let (max_slots, off) = (100usize, 5usize);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let q = rand_t(90, (n_head, seq, hd), &dev);
        let k_big = rand_kv(91, (n_head, max_slots, hd), &dev);
        let v_big = rand_kv(92, (n_head, max_slots, hd), &dev);
        let k_view = k_big.narrow(1, off, seq).unwrap();
        let v_view = v_big.narrow(1, off, seq).unwrap();
        assert!(
            !k_view.is_contiguous(),
            "the view must exercise the strided path"
        );
        let strided = flash_attn_tensor(&q, &k_view, &v_view, scale).unwrap();
        let packed = flash_attn_tensor(
            &q,
            &k_view.contiguous().unwrap(),
            &v_view.contiguous().unwrap(),
            scale,
        )
        .unwrap();
        let s: Vec<f32> = strided.flatten_all().unwrap().to_vec1().unwrap();
        let p: Vec<f32> = packed.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            s.iter()
                .zip(p.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "strided and packed views must agree bitwise"
        );
    }

    /// Rejection paths: dtype, head_dim and head-count preconditions fail
    /// cleanly.
    #[test]
    fn flash_attn_tensor_rejects_bad_inputs() {
        let dev = metal_device().unwrap();
        let q = rand_t(50, (6, 4, 128), &dev);
        let k = rand_kv(51, (2, 9, 128), &dev);
        let v = rand_kv(52, (2, 9, 128), &dev);
        let scale = 0.1f32;
        flash_attn_tensor(&q, &k, &v, scale).unwrap();
        let k32 = k.to_dtype(DType::F32).unwrap();
        assert!(flash_attn_tensor(&q, &k32, &v, scale).is_err());
        let q64 = rand_t(53, (6, 4, 64), &dev);
        assert!(flash_attn_tensor(&q64, &k, &v, scale).is_err());
        let k4 = rand_kv(55, (4, 9, 128), &dev);
        assert!(flash_attn_tensor(&q, &k4, &k4, scale).is_err());
        let k_t = rand_kv(54, (2, 128, 128), &dev).transpose(1, 2).unwrap();
        assert!(flash_attn_tensor(&q, &k_t, &k_t, scale).is_err());
    }

    /// The cooperative-tensor layout facts the kernel rests on, read off the
    /// device through the test-only probe kernel: the per-lane capacities of
    /// S, the row reduction and O; that the iterator mapping from S and from O
    /// onto the row-reduction tensor is compatible (the softmax and the O
    /// rescale both go through it); and that the valid elements of S and of O
    /// across the 32 lanes cover their tiles exactly once each.
    #[test]
    fn flash_attn_tensor_layout_probe() {
        let dev = metal_device().unwrap();
        let words = dispatch::run_flash_attn_t_probe(&dev).unwrap();
        let (cap_s, cap_r, cap_o) = (words[0] as usize, words[1] as usize, words[2] as usize);
        let (compat_s, compat_o) = (words[3], words[4]);
        println!(
            "flash_t probe: capacity S {cap_s}, row reduction {cap_r}, O {cap_o}; \
             iterator compatible S->R {compat_s}, O->R {compat_o}"
        );
        assert!(cap_s > 0 && cap_o > 0);
        assert!(
            8 + 32 * cap_s * 2 + 32 * cap_o * 2 <= dispatch::FLASH_T_PROBE_WORDS,
            "probe dump exceeded its buffer"
        );
        assert_eq!(
            compat_s, 1,
            "S -> row reduction iterator mapping incompatible"
        );
        assert_eq!(
            compat_o, 1,
            "O -> row reduction iterator mapping incompatible"
        );

        let coverage = |base: usize, cap: usize, cols: usize, rows: usize, what: &str| {
            let mut seen = vec![0u32; cols * rows];
            for lane in 0..32 {
                for i in 0..cap {
                    let col = words[base + (lane * cap + i) * 2];
                    let row = words[base + (lane * cap + i) * 2 + 1];
                    if col < 0 {
                        continue;
                    }
                    let (col, row) = (col as usize, row as usize);
                    assert!(
                        col < cols && row < rows,
                        "{what}: coordinate ({col}, {row}) out of tile"
                    );
                    seen[row * cols + col] += 1;
                }
            }
            let dup = seen.iter().filter(|&&n| n != 1).count();
            assert_eq!(
                dup, 0,
                "{what}: {dup} tile elements not covered exactly once"
            );
        };
        coverage(8, cap_s, 32, 16, "S");
        coverage(8 + 32 * cap_s * 2, cap_o, 128, 16, "O");
    }
}
