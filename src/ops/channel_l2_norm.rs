use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

/// Per-pixel channel L2 norm against `kernel_channel_l2_norm`
/// (channel_l2_norm.metal): `x / max(‖x‖₂ over C, eps) * gamma[c]` for `x`
/// `[B, C, H, W]` f32 contiguous and `gamma` any shape of `C` f32 elements.
/// One dispatch where the candle chain is six. Not bitwise against that chain,
/// the sum of squares being taken in a different order; the tests bound it.
/// Metal only; the caller keeps the candle chain for other devices.
pub fn channel_l2_norm(x: &Tensor, gamma: &Tensor, eps: f32) -> Result<Tensor> {
    dispatch::run_channel_l2_norm(x, gamma, eps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use candle_core::Device;

    fn rand(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64 * 8.0 - 4.0) as f32
            })
            .collect()
    }

    /// The chain the VAE ran, on the CPU in f32.
    fn reference(x: &Tensor, gamma: &Tensor, eps: f64) -> Tensor {
        let c = gamma.elem_count();
        let norm = x
            .sqr()
            .unwrap()
            .sum_keepdim(1)
            .unwrap()
            .sqrt()
            .unwrap()
            .maximum(eps)
            .unwrap();
        x.broadcast_div(&norm)
            .unwrap()
            .broadcast_mul(&gamma.reshape((1, c, 1, 1)).unwrap())
            .unwrap()
    }

    #[test]
    fn the_kernel_matches_the_chain_at_ragged_shapes() {
        let dev = metal_device().unwrap();
        // Widths and planes that are not multiples of any launch width, two
        // batches, and the decoder's own narrowest and widest channel counts.
        for (seed, (b, c, h, w)) in [(1, 8, 5, 7), (2, 144, 17, 3), (1, 1152, 4, 4), (1, 3, 1, 1)]
            .into_iter()
            .enumerate()
        {
            let x = Tensor::from_vec(
                rand(seed as u64 + 1, b * c * h * w),
                (b, c, h, w),
                &Device::Cpu,
            )
            .unwrap();
            let gamma = Tensor::from_vec(rand(seed as u64 + 99, c), c, &Device::Cpu).unwrap();
            let want = reference(&x, &gamma, 1e-12);
            let got = channel_l2_norm(
                &x.to_device(&dev).unwrap(),
                &gamma.to_device(&dev).unwrap(),
                1e-12,
            )
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();
            assert_eq!(got.dims(), want.dims());
            let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
            let r: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
            let scale = r.iter().fold(0f32, |m, v| m.max(v.abs()));
            for (a, e) in g.iter().zip(&r) {
                assert!(a.is_finite());
                assert!(
                    (a - e).abs() <= 2e-6 * scale,
                    "{a} vs {e} at [{b},{c},{h},{w}]"
                );
            }
        }
    }

    /// A pixel whose channels are all zero has norm 0: the floor keeps the
    /// division finite and the output is exactly zero, as the chain's is.
    #[test]
    fn a_zero_pixel_stays_zero() {
        let dev = metal_device().unwrap();
        let (c, h, w) = (16, 3, 3);
        let mut data = rand(7, c * h * w);
        for ch in 0..c {
            data[ch * h * w + 4] = 0.0;
        }
        let x = Tensor::from_vec(data, (1, c, h, w), &dev).unwrap();
        let gamma = Tensor::from_vec(rand(8, c), c, &dev).unwrap();
        let out: Vec<f32> = channel_l2_norm(&x, &gamma, 1e-12)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for ch in 0..c {
            assert_eq!(out[ch * h * w + 4], 0.0);
        }
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().any(|v| *v != 0.0));
    }

    fn assert_close(got: &Tensor, want: &Tensor, what: &str) {
        assert_eq!(got.dims(), want.dims(), "{what}");
        let g: Vec<f32> = got
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let r: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        let scale = r.iter().fold(0f32, |m, v| m.max(v.abs()));
        for (a, e) in g.iter().zip(&r) {
            assert!(
                a.is_finite() && e.is_finite(),
                "{what}: non-finite {a} / {e}"
            );
            assert!((a - e).abs() <= 2e-6 * scale, "{what}: {a} vs {e}");
        }
    }

    /// A view that starts past its buffer's first element, for the activation
    /// and for gamma: the kernel must read from the view's offset.
    #[test]
    fn views_at_a_nonzero_offset_read_their_own_elements() {
        let dev = metal_device().unwrap();
        let (c, h, w) = (24, 5, 3);
        let both = Tensor::from_vec(rand(11, 2 * c * h * w), (2, c, h, w), &Device::Cpu).unwrap();
        let gammas = Tensor::from_vec(rand(12, 2 * c), 2 * c, &Device::Cpu).unwrap();
        let (x_cpu, g_cpu) = (
            both.narrow(0, 1, 1).unwrap(),
            gammas.narrow(0, c, c).unwrap(),
        );
        let want = reference(
            &x_cpu.contiguous().unwrap(),
            &g_cpu.contiguous().unwrap(),
            1e-12,
        );

        let x = both.to_device(&dev).unwrap().narrow(0, 1, 1).unwrap();
        let g = gammas.to_device(&dev).unwrap().narrow(0, c, c).unwrap();
        assert!(x.is_contiguous() && g.is_contiguous());
        let got = channel_l2_norm(&x, &g, 1e-12).unwrap();
        assert_close(&got, &want, "offset views");

        // The first batch element and the first gammas give another answer,
        // so an ignored offset cannot pass by coincidence.
        let wrong = reference(
            &both.narrow(0, 0, 1).unwrap().contiguous().unwrap(),
            &gammas.narrow(0, 0, c).unwrap().contiguous().unwrap(),
            1e-12,
        );
        let d = (wrong - want).unwrap().abs().unwrap().max_all().unwrap();
        assert!(d.to_scalar::<f32>().unwrap() > 1e-2);
    }

    /// A strided view is refused by name; the caller makes it contiguous.
    #[test]
    fn a_non_contiguous_input_is_refused() {
        let dev = metal_device().unwrap();
        let x = Tensor::from_vec(rand(13, 8 * 4 * 6), (1, 8, 4, 6), &dev).unwrap();
        let strided = x.transpose(2, 3).unwrap();
        assert!(!strided.is_contiguous());
        let gamma = Tensor::ones(8, candle_core::DType::F32, &dev).unwrap();
        let err = channel_l2_norm(&strided, &gamma, 1e-12)
            .unwrap_err()
            .to_string();
        assert!(err.contains("contiguous"), "{err}");
        let made = channel_l2_norm(&strided.contiguous().unwrap(), &gamma, 1e-12).unwrap();
        let want = reference(
            &strided
                .contiguous()
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap(),
            &Tensor::ones(8, candle_core::DType::F32, &Device::Cpu).unwrap(),
            1e-12,
        );
        assert_close(&made, &want, "made contiguous");
    }

    /// Around the floor the form is `x / max(‖x‖, eps)`: a pixel whose norm is
    /// under `eps` is divided by `eps` and not by its norm, one just over it
    /// by its norm. Expected values are worked out in f64 from that form.
    #[test]
    fn pixels_near_the_floor_follow_the_clamp_form() {
        let dev = metal_device().unwrap();
        let c = 4usize;
        let eps = 1e-12f32;
        // One pixel per case: far under the floor, just under, just over,
        // well over.
        let cases = [1e-20f32, 3e-13, 8e-13, 1e-6];
        let mut data = vec![0f32; c * cases.len()];
        for (p, v) in cases.iter().enumerate() {
            for ch in 0..c {
                data[ch * cases.len() + p] = *v * (1.0 + ch as f32 * 0.25);
            }
        }
        let gamma: Vec<f32> = vec![2.0, -1.0, 0.5, 3.0];
        let x = Tensor::from_vec(data.clone(), (1, c, 1, cases.len()), &dev).unwrap();
        let g = Tensor::from_vec(gamma.clone(), c, &dev).unwrap();
        let got: Vec<f32> = channel_l2_norm(&x, &g, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let mut under = 0;
        for p in 0..cases.len() {
            let norm = (0..c)
                .map(|ch| (data[ch * cases.len() + p] as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            if norm < eps as f64 {
                under += 1;
            }
            for ch in 0..c {
                let i = ch * cases.len() + p;
                let want = data[i] as f64 / norm.max(eps as f64) * gamma[ch] as f64;
                let got = got[i] as f64;
                assert!(got.is_finite());
                assert!(
                    (got - want).abs() <= 1e-5 * want.abs().max(1e-30),
                    "pixel {p} channel {ch}: {got} against {want}"
                );
            }
        }
        assert!(
            under >= 2 && under < cases.len(),
            "the cases must straddle the floor"
        );
    }

    #[test]
    fn operands_off_contract_are_refused() {
        let dev = metal_device().unwrap();
        let x = Tensor::zeros((1, 8, 2, 2), candle_core::DType::F32, &dev).unwrap();
        let short = Tensor::zeros(7, candle_core::DType::F32, &dev).unwrap();
        assert!(channel_l2_norm(&x, &short, 1e-12).is_err());
        let half = Tensor::zeros(8, candle_core::DType::F16, &dev).unwrap();
        assert!(channel_l2_norm(&x, &half, 1e-12).is_err());
        let cpu = Tensor::zeros((1, 8, 2, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        let gamma = Tensor::zeros(8, candle_core::DType::F32, &Device::Cpu).unwrap();
        assert!(channel_l2_norm(&cpu, &gamma, 1e-12).is_err());
    }
}
