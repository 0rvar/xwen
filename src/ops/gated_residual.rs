use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

/// Gated residual add against `kernel_gated_residual` (gated_residual.metal):
/// `h + gate ⊙ y` with `gate` one f32 per channel (any shape whose element
/// count is the last dim of `h`), `h` and `y` same-shape f32 contiguous. One
/// pass, replacing candle's `broadcast_mul` then `add`, which this reproduces
/// bit for bit (`fused_matches_candle_bitwise` proves it). Metal only; the
/// caller keeps the candle chain for other devices.
pub fn gated_residual(h: &Tensor, y: &Tensor, gate: &Tensor) -> Result<Tensor> {
    dispatch::run_gated_residual(h, y, gate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use candle_core::{DType, Device};

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

    /// The candle chain the modulated block ran: `gate.broadcast_mul(y)` then
    /// the residual add.
    fn candle_chain(h: &Tensor, y: &Tensor, gate: &Tensor) -> Tensor {
        (h + gate.broadcast_mul(y).unwrap()).unwrap()
    }

    /// The kernel reproduces the chain bit for bit at the production shape
    /// (4128 tokens by 3840 channels) and on ragged small ones. Multiply then
    /// add, each rounded once on both sides.
    #[test]
    fn fused_matches_candle_bitwise() {
        let dev = metal_device().unwrap();
        for &(t, c) in &[(4128usize, 3840usize), (7, 5), (1, 3)] {
            let n = t * c;
            let to_dev = |v: Vec<f32>, shape: (usize, usize, usize)| {
                Tensor::from_vec(v, shape, &Device::Cpu)
                    .unwrap()
                    .to_device(&dev)
                    .unwrap()
            };
            let h = to_dev(rand(0x1 + t as u64, n, -30.0, 30.0), (1, t, c));
            let y = to_dev(rand(0x2 + t as u64, n, -30.0, 30.0), (1, t, c));
            let gate = to_dev(rand(0x3 + c as u64, c, -1.0, 1.0), (1, 1, c));
            let got = gated_residual(&h, &y, &gate).unwrap();
            assert_eq!(got.dims(), h.dims());
            assert_eq!(got.dtype(), DType::F32);
            let want = candle_chain(&h, &y, &gate);
            let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
            let w: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
            for (i, (a, b)) in g.iter().zip(&w).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "gated_residual [{t},{c}]: element {i} got {a:?}, chain {b:?}"
                );
            }
        }
    }

    #[test]
    fn shape_and_dtype_errors() {
        let dev = metal_device().unwrap();
        let h = Tensor::zeros((1, 4, 6), DType::F32, &dev).unwrap();
        let y = Tensor::zeros((1, 4, 6), DType::F32, &dev).unwrap();
        let gate = Tensor::zeros((1, 1, 6), DType::F32, &dev).unwrap();
        assert!(gated_residual(&h, &y, &gate).is_ok());
        let flat_gate = Tensor::zeros(6, DType::F32, &dev).unwrap();
        assert!(gated_residual(&h, &y, &flat_gate).is_ok());
        let bad_gate = Tensor::zeros((1, 1, 5), DType::F32, &dev).unwrap();
        assert!(gated_residual(&h, &y, &bad_gate).is_err());
        let bad_y = Tensor::zeros((1, 4, 5), DType::F32, &dev).unwrap();
        assert!(gated_residual(&h, &bad_y, &gate).is_err());
        let h16 = Tensor::zeros((1, 4, 6), DType::F16, &dev).unwrap();
        assert!(gated_residual(&h16, &y, &gate).is_err());
        let strided = h.transpose(1, 2).unwrap();
        assert!(gated_residual(&strided, &y, &gate).is_err());
    }
}
