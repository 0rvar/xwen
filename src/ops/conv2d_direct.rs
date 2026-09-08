use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

pub use crate::ops::dispatch::Conv2dDirectFusion as Conv2dFusion;
pub use crate::ops::dispatch::conv2d_direct_supported;

/// A candle `[c_out, c_in, k, k]` convolution weight in the layout
/// `kernel_conv2d_direct` stages from, `[k * k, c_in, c_out]` contiguous: one
/// contiguous run of output channels per (tap, input channel). Done once at
/// load; the direct path never reads the candle layout.
pub fn permute_conv_weight(w: &Tensor) -> Result<Tensor> {
    let (c_out, c_in, kh, kw) = w.dims4()?;
    Ok(w.permute((2, 3, 1, 0))?
        .reshape((kh * kw, c_in, c_out))?
        .contiguous()?)
}

/// Direct f32 convolution against `kernel_conv2d_direct_*`
/// (conv2d_direct.metal): `x` `[B, c_in, H, W]` NCHW, `w` from
/// [`permute_conv_weight`], `bias` `[c_out]`, `kernel` 3 (stride 1, pad 1)
/// or 1. The fusions in [`Conv2dFusion`] are applied on the input read (a
/// folded GroupNorm, silu, a 2x nearest upsample) and on the store (a
/// residual). Metal only; the caller keeps candle's conv2d for other devices
/// and for shapes [`conv2d_direct_supported`] declines.
pub fn conv2d_direct(
    x: &Tensor,
    w: &Tensor,
    bias: &Tensor,
    kernel: usize,
    fusion: Conv2dFusion<'_>,
) -> Result<Tensor> {
    dispatch::run_conv2d_direct(x, w, bias, kernel, fusion)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use candle_core::{DType, Device, Module};
    use candle_nn::{Conv2d, Conv2dConfig};

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

    /// The candle chain the decoder ran: optional norm-affine and silu, an
    /// optional nearest upsample, candle's conv2d, an optional residual.
    #[allow(clippy::too_many_arguments)]
    fn candle_chain(
        x: &Tensor,
        w: &Tensor,
        b: &Tensor,
        kernel: usize,
        norm: Option<(&Tensor, &Tensor)>,
        silu: bool,
        upsample: bool,
        residual: Option<&Tensor>,
    ) -> Tensor {
        let (batch, c, h, wd) = x.dims4().unwrap();
        let mut v = x.clone();
        if let Some((scale, shift)) = norm {
            let scale = scale.reshape((batch, c, 1, 1)).unwrap();
            let shift = shift.reshape((batch, c, 1, 1)).unwrap();
            v = v
                .broadcast_mul(&scale)
                .unwrap()
                .broadcast_add(&shift)
                .unwrap();
        }
        if silu {
            v = v.silu().unwrap();
        }
        if upsample {
            v = v.upsample_nearest2d(h * 2, wd * 2).unwrap();
        }
        let cfg = Conv2dConfig {
            padding: kernel / 2,
            ..Default::default()
        };
        let conv = Conv2d::new(w.clone(), Some(b.clone()), cfg);
        let mut out = conv.forward(&v).unwrap();
        if let Some(r) = residual {
            out = (out + r).unwrap();
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn check(
        dev: &Device,
        seed: u64,
        batch: usize,
        c_in: usize,
        c_out: usize,
        h: usize,
        w: usize,
        kernel: usize,
        norm: bool,
        silu: bool,
        upsample: bool,
        residual: bool,
    ) {
        let x = to_dev(
            rand(seed, batch * c_in * h * w, -3.0, 3.0),
            &[batch, c_in, h, w],
            dev,
        );
        let wt = to_dev(
            rand(seed + 1, c_out * c_in * kernel * kernel, -0.2, 0.2),
            &[c_out, c_in, kernel, kernel],
            dev,
        );
        let bias = to_dev(rand(seed + 2, c_out, -0.5, 0.5), &[c_out], dev);
        let (oh, ow) = if upsample { (h * 2, w * 2) } else { (h, w) };
        let norm_t = norm.then(|| {
            (
                to_dev(rand(seed + 3, batch * c_in, 0.5, 1.5), &[batch, c_in], dev),
                to_dev(rand(seed + 4, batch * c_in, -1.0, 1.0), &[batch, c_in], dev),
            )
        });
        let res_t = residual.then(|| {
            to_dev(
                rand(seed + 5, batch * c_out * oh * ow, -2.0, 2.0),
                &[batch, c_out, oh, ow],
                dev,
            )
        });
        let norm_ref = norm_t.as_ref().map(|(s, t)| (s, t));
        let want = candle_chain(
            &x,
            &wt,
            &bias,
            kernel,
            norm_ref,
            silu,
            upsample,
            res_t.as_ref(),
        );
        let wp = permute_conv_weight(&wt).unwrap();
        let fusion = Conv2dFusion {
            norm: norm_ref,
            silu,
            upsample,
            residual: res_t.as_ref(),
        };
        let got = conv2d_direct(&x, &wp, &bias, kernel, fusion).unwrap();
        assert_eq!(got.dims(), &[batch, c_out, oh, ow]);
        assert_eq!(got.dtype(), DType::F32);
        let err = rel_l2(&got, &want);
        assert!(
            err < BAR,
            "conv {kernel}x{kernel} b{batch} {c_in}->{c_out} {h}x{w} norm={norm} silu={silu} \
             up={upsample} res={residual}: rel_l2 {err:.3e} over {BAR:.0e}"
        );
    }

    /// Every channel count the decoder uses, at a size with whole tiles.
    #[test]
    fn matches_candle_on_decoder_channel_counts() {
        let dev = metal_device().unwrap();
        for (i, &(c_in, c_out)) in [
            (16usize, 512usize),
            (512, 512),
            (512, 256),
            (256, 256),
            (256, 128),
            (128, 128),
            (128, 3),
        ]
        .iter()
        .enumerate()
        {
            check(
                &dev,
                100 + i as u64,
                1,
                c_in,
                c_out,
                16,
                32,
                3,
                false,
                false,
                false,
                false,
            );
        }
    }

    /// Sizes that leave partial tiles on both axes and a batch of two, so
    /// the halo clipping, the pixel-block skipping and the batch stride are
    /// all exercised.
    #[test]
    fn matches_candle_on_ragged_sizes() {
        let dev = metal_device().unwrap();
        for (i, &(h, w)) in [(5usize, 7usize), (9, 17), (8, 16), (13, 33), (1, 1)]
            .iter()
            .enumerate()
        {
            check(
                &dev,
                200 + i as u64,
                2,
                16,
                24,
                h,
                w,
                3,
                false,
                false,
                false,
                false,
            );
        }
    }

    /// The same ragged sizes on the deep 3x3 template: `c_in` past the
    /// shallow bound selects the CO=64, TH=8 kernel, which is every resnet
    /// convolution of the real decode and was otherwise only ever run at a
    /// whole tile. The last case pairs a batch of two with a `c_out` past one
    /// 64-wide block, so the threadgroup z-index carries both.
    #[test]
    fn matches_candle_on_deep_ragged_sizes() {
        let dev = metal_device().unwrap();
        check(
            &dev, 500, 1, 256, 128, 13, 33, 3, false, false, false, false,
        );
        check(&dev, 501, 1, 512, 128, 9, 17, 3, false, false, false, false);
        check(&dev, 502, 2, 256, 128, 9, 17, 3, true, true, false, true);
    }

    /// The fused input read and store: the folded norm, silu, the upsample
    /// and the residual, alone and together.
    #[test]
    fn fusions_match_the_candle_chain() {
        let dev = metal_device().unwrap();
        check(&dev, 300, 1, 32, 40, 12, 20, 3, true, false, false, false);
        check(&dev, 301, 1, 32, 40, 12, 20, 3, false, true, false, false);
        check(&dev, 302, 1, 32, 40, 12, 20, 3, false, false, true, false);
        check(&dev, 303, 1, 32, 40, 12, 20, 3, false, false, false, true);
        check(&dev, 304, 2, 32, 40, 7, 9, 3, true, true, false, true);
        check(&dev, 305, 1, 32, 40, 7, 9, 3, false, false, true, true);
        // conv_out's shape with its silu fold, on the 8-wide kernel.
        check(&dev, 306, 1, 128, 3, 9, 17, 3, true, true, false, false);
    }

    /// The 1x1 shortcut convolutions, with and without a residual.
    #[test]
    fn one_by_one_matches_candle() {
        let dev = metal_device().unwrap();
        check(&dev, 400, 1, 512, 256, 8, 16, 1, false, false, false, false);
        check(&dev, 401, 1, 256, 128, 9, 17, 1, false, false, false, true);
        check(&dev, 402, 2, 16, 40, 5, 7, 1, false, false, false, false);
    }

    /// The direct kernel against candle's conv2d at the decoder's production
    /// shapes, in GFLOP/s, for pricing kernel variants. Ignored: it is a
    /// measurement, not a gate. Run with `--ignored --nocapture` under the
    /// GPU lock.
    #[test]
    #[ignore]
    fn microbench_production_shapes() {
        let dev = metal_device().unwrap();
        let shapes: [(usize, usize, usize, usize); 6] = [
            (512, 512, 128, 128),
            (512, 512, 256, 256),
            (512, 256, 512, 512),
            (256, 256, 512, 512),
            (256, 128, 1024, 1024),
            (128, 128, 1024, 1024),
        ];
        for &(c_in, c_out, h, w) in &shapes {
            let x = to_dev(rand(1, c_in * h * w, -1.0, 1.0), &[1, c_in, h, w], &dev);
            let wt = to_dev(
                rand(2, c_out * c_in * 9, -0.1, 0.1),
                &[c_out, c_in, 3, 3],
                &dev,
            );
            let bias = to_dev(rand(3, c_out, -0.5, 0.5), &[c_out], &dev);
            let wp = permute_conv_weight(&wt).unwrap();
            let flops = 2.0 * (c_in * c_out * 9 * h * w) as f64;
            let cfg = Conv2dConfig {
                padding: 1,
                ..Default::default()
            };
            let conv = Conv2d::new(wt.clone(), Some(bias.clone()), cfg);
            let time = |f: &dyn Fn() -> Tensor| -> f64 {
                let _ = f();
                dev.synchronize().unwrap();
                let reps = 5;
                let started = std::time::Instant::now();
                for _ in 0..reps {
                    let _ = f();
                }
                dev.synchronize().unwrap();
                started.elapsed().as_secs_f64() / reps as f64
            };
            let direct =
                time(&|| conv2d_direct(&x, &wp, &bias, 3, Conv2dFusion::default()).unwrap());
            let candle = time(&|| conv.forward(&x).unwrap());
            eprintln!(
                "conv 3x3 {c_in}->{c_out} at {h}x{w}: direct {:.1} ms ({:.0} GFLOP/s), candle {:.1} ms ({:.0} GFLOP/s)",
                direct * 1e3,
                flops / direct / 1e9,
                candle * 1e3,
                flops / candle / 1e9
            );
        }
    }

    #[test]
    fn refuses_uncovered_shapes() {
        let dev = metal_device().unwrap();
        assert!(conv2d_direct_supported(16, 3));
        assert!(conv2d_direct_supported(8, 1));
        assert!(!conv2d_direct_supported(3, 3));
        assert!(!conv2d_direct_supported(12, 3));
        assert!(!conv2d_direct_supported(16, 5));
        assert!(!conv2d_direct_supported(0, 3));
        let x = Tensor::zeros((1, 3, 4, 4), DType::F32, &dev).unwrap();
        let w = Tensor::zeros((9, 3, 8), DType::F32, &dev).unwrap();
        let b = Tensor::zeros(8, DType::F32, &dev).unwrap();
        assert!(conv2d_direct(&x, &w, &b, 3, Conv2dFusion::default()).is_err());
        let x = Tensor::zeros((1, 16, 4, 4), DType::F32, &dev).unwrap();
        let w = Tensor::zeros((9, 16, 8), DType::F32, &dev).unwrap();
        assert!(conv2d_direct(&x, &w, &b, 3, Conv2dFusion::default()).is_ok());
        let bad_w = Tensor::zeros((9, 8, 8), DType::F32, &dev).unwrap();
        assert!(conv2d_direct(&x, &bad_w, &b, 3, Conv2dFusion::default()).is_err());
        let bad_b = Tensor::zeros(7, DType::F32, &dev).unwrap();
        assert!(conv2d_direct(&x, &w, &bad_b, 3, Conv2dFusion::default()).is_err());
        let bad_res = Tensor::zeros((1, 8, 4, 5), DType::F32, &dev).unwrap();
        let fusion = Conv2dFusion {
            residual: Some(&bad_res),
            ..Default::default()
        };
        assert!(conv2d_direct(&x, &w, &b, 3, fusion).is_err());
        let x16 = x.to_dtype(DType::F16).unwrap();
        assert!(conv2d_direct(&x16, &w, &b, 3, Conv2dFusion::default()).is_err());
    }
}
