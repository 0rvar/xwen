use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

/// GroupNorm statistics folded to a per-(batch, channel) affine against
/// `kernel_group_norm_partials` and `kernel_group_norm_fold`
/// (group_norm.metal): `(scale, shift)`, each `[B, C]` f32, such that
/// `x * scale[b, c] + shift[b, c]` is `group_norm(x)` with `gamma` and
/// `beta` applied. One read of `x`. The pair feeds [`group_norm_apply`] or the
/// direct convolution's input read (`ops::conv2d_direct`), which then never
/// writes the normalized tensor. Metal only.
pub fn group_norm_fold(
    x: &Tensor,
    groups: usize,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    dispatch::run_group_norm_fold(x, groups, gamma, beta, eps)
}

/// `x * scale + shift` per (batch, channel), then silu when asked, against
/// `kernel_group_norm_apply` (group_norm.metal). One pass.
pub fn group_norm_apply(x: &Tensor, scale: &Tensor, shift: &Tensor, silu: bool) -> Result<Tensor> {
    dispatch::run_group_norm_apply(x, scale, shift, silu)
}

/// The whole GroupNorm, [`group_norm_fold`] then [`group_norm_apply`]: two
/// passes over `x` where candle's chain takes about nine.
pub fn group_norm(
    x: &Tensor,
    groups: usize,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
    silu: bool,
) -> Result<Tensor> {
    let (scale, shift) = group_norm_fold(x, groups, gamma, beta, eps)?;
    group_norm_apply(x, &scale, &shift, silu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use candle_core::{DType, Device, Module};

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

    fn to_dev(v: Vec<f32>, shape: &[usize], dev: &Device) -> Tensor {
        Tensor::from_vec(v, shape, &Device::Cpu)
            .unwrap()
            .to_device(dev)
            .unwrap()
    }

    fn rel_l2(got: &Tensor, want: &Tensor) -> f64 {
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let w: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(g.len(), w.len());
        let mut num = 0f64;
        let mut den = 0f64;
        for (a, b) in g.iter().zip(&w) {
            assert!(a.is_finite(), "non-finite output {a}");
            num += ((*a - *b) as f64).powi(2);
            den += (*b as f64).powi(2);
        }
        (num / den.max(1e-30)).sqrt()
    }

    const BAR: f64 = 1e-5;

    /// Against candle's GroupNorm, on data offset from zero (the shifted
    /// single-pass variance is what makes that safe) with a per-channel
    /// drift so the groups have different statistics.
    fn check(
        dev: &Device,
        seed: u64,
        batch: usize,
        c: usize,
        groups: usize,
        h: usize,
        w: usize,
        silu: bool,
    ) {
        let mut data = rand(seed, batch * c * h * w, -1.0, 1.0);
        for (i, v) in data.iter_mut().enumerate() {
            let ch = (i / (h * w)) % c;
            *v = *v * (0.5 + ch as f32 * 0.01) + 4.0 + ch as f32 * 0.1;
        }
        let x = to_dev(data, &[batch, c, h, w], dev);
        let gamma = to_dev(rand(seed + 1, c, 0.5, 1.5), &[c], dev);
        let beta = to_dev(rand(seed + 2, c, -0.5, 0.5), &[c], dev);
        let eps = 1e-6f32;
        let reference =
            candle_nn::GroupNorm::new(gamma.clone(), beta.clone(), c, groups, eps as f64).unwrap();
        let mut want = reference.forward(&x).unwrap();
        if silu {
            want = want.silu().unwrap();
        }
        let got = group_norm(&x, groups, &gamma, &beta, eps, silu).unwrap();
        assert_eq!(got.dims(), x.dims());
        assert_eq!(got.dtype(), DType::F32);
        let err = rel_l2(&got, &want);
        assert!(
            err < BAR,
            "group_norm b{batch} c{c} g{groups} {h}x{w} silu={silu}: rel_l2 {err:.3e} over {BAR:.0e}"
        );
    }

    /// The decoder's channel counts at 32 groups, at a plane size that takes
    /// the float4 path and one that does not, with and without the silu.
    #[test]
    fn matches_candle_group_norm() {
        let dev = metal_device().unwrap();
        check(&dev, 1, 1, 512, 32, 16, 16, false);
        check(&dev, 2, 1, 256, 32, 12, 20, true);
        check(&dev, 3, 2, 128, 32, 7, 9, false);
        check(&dev, 4, 1, 128, 32, 5, 5, true);
        check(&dev, 5, 1, 64, 32, 1, 1, false);
    }

    /// A group long enough to split into many statistics slices: 4 channels
    /// of 256x256 is 262144 floats, 16 slices at the kernel's slice size.
    #[test]
    fn long_groups_reduce_across_slices() {
        let dev = metal_device().unwrap();
        check(&dev, 6, 1, 128, 32, 256, 256, true);
    }

    #[test]
    fn fold_then_apply_is_the_norm() {
        let dev = metal_device().unwrap();
        let x = to_dev(rand(7, 64 * 8 * 8, -1.0, 1.0), &[1, 64, 8, 8], &dev);
        let gamma = to_dev(rand(8, 64, 0.5, 1.5), &[64], &dev);
        let beta = to_dev(rand(9, 64, -0.5, 0.5), &[64], &dev);
        let (scale, shift) = group_norm_fold(&x, 32, &gamma, &beta, 1e-6).unwrap();
        assert_eq!(scale.dims(), &[1, 64]);
        assert_eq!(shift.dims(), &[1, 64]);
        let via_affine = x
            .broadcast_mul(&scale.reshape((1, 64, 1, 1)).unwrap())
            .unwrap()
            .broadcast_add(&shift.reshape((1, 64, 1, 1)).unwrap())
            .unwrap();
        let fused = group_norm_apply(&x, &scale, &shift, false).unwrap();
        let err = rel_l2(&fused, &via_affine);
        assert!(err < BAR, "apply against the affine: rel_l2 {err:.3e}");
    }

    #[test]
    fn shape_errors() {
        let dev = metal_device().unwrap();
        let x = Tensor::zeros((1, 64, 4, 4), DType::F32, &dev).unwrap();
        let gamma = Tensor::ones(64, DType::F32, &dev).unwrap();
        let beta = Tensor::zeros(64, DType::F32, &dev).unwrap();
        assert!(group_norm_fold(&x, 32, &gamma, &beta, 1e-6).is_ok());
        assert!(group_norm_fold(&x, 48, &gamma, &beta, 1e-6).is_err());
        let short = Tensor::ones(32, DType::F32, &dev).unwrap();
        assert!(group_norm_fold(&x, 32, &short, &beta, 1e-6).is_err());
        let x3 = Tensor::zeros((64, 4, 4), DType::F32, &dev).unwrap();
        assert!(group_norm_fold(&x3, 32, &gamma, &beta, 1e-6).is_err());
        let (scale, shift) = group_norm_fold(&x, 32, &gamma, &beta, 1e-6).unwrap();
        assert!(group_norm_apply(&x, &scale, &short, false).is_err());
        assert!(group_norm_apply(&x, &scale, &shift, false).is_ok());
    }
}
