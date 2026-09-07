// Vendored from candle rev 21cca0b (candle-transformers/src/models/z_image/sampling.rs,
// PR #3261, SpenserCai), MIT / Apache-2.0. Rewritten here: the noise is drawn from a
// seeded CPU generator rather than the device's, and the CFG helpers are gone —
// Z-Image-Turbo is a distilled model that runs without guidance.
//! Sampling utilities for the Z-Image pipeline: the initial latent noise and
//! the pixel post-processing.

use candle_core::{DType, Device, Result, Tensor};
use rand::{Rng, SeedableRng};

/// Standard-normal noise of the given shape, drawn on the CPU from a generator
/// seeded with `seed`, so that one seed gives the same latent on every run
/// and every device. (candle's CPU device cannot be seeded, and its Metal
/// generator is not stable across builds; torch's is not reproducible from
/// here at all — a reference comparison injects the latent instead.)
///
/// Box-Muller over `rand`'s `StdRng`. Not torch's `randn`, so a seed here and
/// a seed in diffusers are unrelated draws.
pub fn seeded_noise(
    seed: u64,
    shape: (usize, usize, usize, usize),
    device: &Device,
) -> Result<Tensor> {
    let (b, c, h, w) = shape;
    let n = b * c * h * w;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        // Two uniforms in (0, 1]: the log needs u1 > 0.
        let u1: f64 = 1.0 - rng.random::<f64>();
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        out.push((r * theta.sin()) as f32);
    }
    out.truncate(n);
    Tensor::from_vec(out, shape, &Device::Cpu)?.to_device(device)
}

/// The VAE's `[-1, 1]` float image as `[0, 255]` u8, the way diffusers'
/// `VaeImageProcessor` does it: `(x / 2 + 0.5)` clamped to `[0, 1]`, times
/// 255, ROUNDED to the nearest integer.
pub fn postprocess_image(image: &Tensor) -> Result<Tensor> {
    let image = ((image.to_dtype(DType::F32)? / 2.0)? + 0.5)?.clamp(0.0, 1.0)?;
    (image * 255.0)?.round()?.to_dtype(DType::U8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_noise_is_reproducible_and_roughly_standard_normal() {
        let a = seeded_noise(7, (1, 4, 16, 16), &Device::Cpu).unwrap();
        let b = seeded_noise(7, (1, 4, 16, 16), &Device::Cpu).unwrap();
        let c = seeded_noise(8, (1, 4, 16, 16), &Device::Cpu).unwrap();
        let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cv = c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(av, bv);
        assert_ne!(av, cv);
        let n = av.len() as f64;
        let mean = av.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var = av.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 0.1, "mean {mean}");
        assert!((var - 1.0).abs() < 0.15, "var {var}");
    }

    #[test]
    fn postprocess_rounds_to_the_nearest_byte() {
        // -1 -> 0, 0 -> 127.5 -> 128 (round half away from zero), 1 -> 255,
        // and out-of-range values clamp.
        let x = Tensor::new(&[-1.0f32, 0.0, 1.0, 1.5, -2.0, 0.5], &Device::Cpu).unwrap();
        let y = postprocess_image(&x).unwrap().to_vec1::<u8>().unwrap();
        assert_eq!(y, [0, 128, 255, 255, 0, 191]);
    }
}
