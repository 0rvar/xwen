use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

/// Interleaved-pair rotary embedding against `kernel_rope_pair`
/// (rope_pair.metal): dims `(2i, 2i+1)` of `x` `[batch, seq, heads, head_dim]`
/// f32 rotate together by column `i` of `cos`/`sin` `[seq, head_dim/2]` f32,
/// the `view_as_complex` form the Z-Image transformer uses. One read and one
/// write of `x`, replacing the strided-view chain in
/// `zimage::transformer::apply_rotary_emb`, which this reproduces bit for bit
/// (`fused_matches_candle_bitwise` proves it). Metal only; the caller keeps the
/// candle chain for other devices.
pub fn rope_pair(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    dispatch::run_rope_pair(x, cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use candle_core::{D, DType, Device, IndexOp};

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

    /// The candle chain `apply_rotary_emb` runs off Metal, verbatim: strided
    /// even/odd views, four broadcast multiplies, a subtract, an add, a stack.
    fn candle_chain(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let (b, seq, heads, head_dim) = x.dims4().unwrap();
        let half = head_dim / 2;
        let x = x.reshape((b, seq, heads, half, 2)).unwrap();
        let x_re = x.i((.., .., .., .., 0)).unwrap();
        let x_im = x.i((.., .., .., .., 1)).unwrap();
        let cos = cos.unsqueeze(0).unwrap().unsqueeze(2).unwrap();
        let sin = sin.unsqueeze(0).unwrap().unsqueeze(2).unwrap();
        let y_re = (x_re.broadcast_mul(&cos).unwrap() - x_im.broadcast_mul(&sin).unwrap()).unwrap();
        let y_im = (x_re.broadcast_mul(&sin).unwrap() + x_im.broadcast_mul(&cos).unwrap()).unwrap();
        Tensor::stack(&[y_re, y_im], D::Minus1)
            .unwrap()
            .reshape((b, seq, heads, head_dim))
            .unwrap()
    }

    fn tables(seq: usize, half: usize, dev: &Device) -> (Tensor, Tensor) {
        let angles = rand(0x77, seq * half, -6.5, 6.5);
        let a = Tensor::from_vec(angles, (seq, half), &Device::Cpu)
            .unwrap()
            .to_device(dev)
            .unwrap();
        (a.cos().unwrap(), a.sin().unwrap())
    }

    /// The kernel reproduces the candle chain bit for bit on the production
    /// shape (4128 tokens, 30 heads of 128, the 1024x1024 image) and on small
    /// ragged ones. Both sides round each multiply and each add separately,
    /// so any difference is a real divergence, never noise.
    #[test]
    fn fused_matches_candle_bitwise() {
        let dev = metal_device().unwrap();
        for &(b, seq, heads, head_dim) in &[
            (1usize, 4128usize, 30usize, 128usize),
            (2, 37, 3, 8),
            (1, 5, 1, 2),
        ] {
            let n = b * seq * heads * head_dim;
            let x = Tensor::from_vec(
                rand(0x10 + seq as u64, n, -4.0, 4.0),
                (b, seq, heads, head_dim),
                &Device::Cpu,
            )
            .unwrap()
            .to_device(&dev)
            .unwrap();
            let (cos, sin) = tables(seq, head_dim / 2, &dev);
            let got = rope_pair(&x, &cos, &sin).unwrap();
            assert_eq!(got.dims(), x.dims());
            assert_eq!(got.dtype(), DType::F32);
            let want = candle_chain(&x, &cos, &sin);
            let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
            let w: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
            let max_x = x
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let mut worst = 0f32;
            for (i, (a, b)) in g.iter().zip(&w).enumerate() {
                worst = worst.max((a - b).abs());
                assert!(
                    (a - b).abs() <= 1e-5 * max_x,
                    "rope_pair [{b},{seq},{heads},{head_dim}]: element {i} got {a}, chain {b}"
                );
            }
            let bitwise = g.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits());
            eprintln!(
                "rope_pair [{b},{seq},{heads},{head_dim}]: max |delta| {worst:.3e}, bitwise {bitwise}"
            );
            assert!(
                bitwise,
                "the kernel and the chain round identically by construction"
            );
        }
    }

    #[test]
    fn shape_and_dtype_errors() {
        let dev = metal_device().unwrap();
        let x = Tensor::zeros((1, 6, 2, 8), DType::F32, &dev).unwrap();
        let (cos, sin) = tables(6, 4, &dev);
        assert!(rope_pair(&x, &cos, &sin).is_ok());
        let short = tables(5, 4, &dev);
        assert!(rope_pair(&x, &short.0, &short.1).is_err()); // fewer table rows than tokens
        let wide = tables(6, 5, &dev);
        assert!(rope_pair(&x, &wide.0, &wide.1).is_err()); // columns != head_dim / 2
        let odd = Tensor::zeros((1, 6, 2, 7), DType::F32, &dev).unwrap();
        assert!(rope_pair(&odd, &cos, &sin).is_err()); // odd head_dim
        let x16 = Tensor::zeros((1, 6, 2, 8), DType::F16, &dev).unwrap();
        assert!(rope_pair(&x16, &cos, &sin).is_err());
        let strided = x.transpose(1, 2).unwrap();
        assert!(rope_pair(&strided, &cos, &sin).is_err());
    }
}
