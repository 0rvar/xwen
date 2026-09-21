//! Op-level timings for the Qwen-Image 2.1 VAE decoder at the exact shapes it
//! runs (src/qwen_image/vae.rs): widths 1152, 1152, 1152, 576, 288, 144 from
//! a 64-channel latent at 1/16 scale, three residual blocks per stage, a
//! nearest 2x and a convolution between stages, RGBA out.
//!
//! ```text
//! cargo test --release --test qwen_image_microbench -- --ignored --nocapture
//! ```
//!
//! That command selects both tests here, and the harness would start them on
//! two threads. They take [`GPU`] before touching the device, so they run one
//! after the other whatever the thread count: two benchmarks sharing the GPU
//! time each other and measure nothing. Name one to run it alone, e.g.
//! `-- --ignored --nocapture --exact qwen_image_vae_microbench`.
//!
//! It prices the two convolution arms of `XWEN_QWEN_IMAGE_VAE` against each
//! other shape by shape, and then the parts of the decode neither arm touches:
//! the per-pixel channel L2 norm as the candle chain and as
//! `ops::channel_l2_norm`, the silu the candle arm runs as its own op,
//! and the mid block's single-head attention with its full score matrix. The
//! attribution at the end multiplies each row by how often the decoder runs
//! it, so what is left after the convolutions is ranked and not guessed.
//!
//! It loads no weights. It is still a GPU benchmark, so nothing else large may
//! run alongside it (AGENTS.md "Operational hazards"), and candle's im2col
//! column buffer for the widest full-resolution convolution is 10.9 GB.
//!
//! Every figure is per-iteration wall time between two `Device::synchronize`
//! calls around a loop, after warm-up. Convolution rows report TFLOP/s =
//! 2 * H * W * c_out * c_in * k^2 / t.

use anyhow::Result;
use candle_core::{Device, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig};
use xwen::ops::{self, Conv2dFusion};

/// Held for the whole of a benchmark, so the tests in this file never share
/// the device. A poisoned lock still serialises, which is all it is for.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

const TARGET_SECS: f64 = 0.7;
const MAX_ITERS: usize = 200;

/// Runs `f` to a steady state and returns milliseconds per iteration. The one
/// timed calibration iteration only sizes the measured loop.
fn time_op(dev: &Device, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..2 {
        f()?;
    }
    dev.synchronize()?;
    let probe = std::time::Instant::now();
    f()?;
    dev.synchronize()?;
    let est = probe.elapsed().as_secs_f64();
    let iters = if est <= 0.0 {
        MAX_ITERS
    } else {
        ((TARGET_SECS / est).ceil() as usize).clamp(2, MAX_ITERS)
    };
    let start = std::time::Instant::now();
    for _ in 0..iters {
        f()?;
    }
    dev.synchronize()?;
    Ok(start.elapsed().as_secs_f64() / iters as f64 * 1e3)
}

/// One convolution shape of the decoder: `side` is the INPUT side, the output
/// being twice it under `upsample`. `count` is how many times one decode runs
/// the shape.
#[derive(Clone, Copy)]
struct ConvShape {
    what: &'static str,
    c_in: usize,
    c_out: usize,
    kernel: usize,
    side: usize,
    upsample: bool,
    count: usize,
}

/// The decoder's convolutions for an image of `side` pixels, `l = side / 16`
/// being the latent side. Three residual blocks per stage, two 3x3 each; the
/// first block of a narrowing stage has a 1x1 shortcut; every stage but the
/// last ends in a nearest 2x and a 3x3.
fn decoder_convs(side: usize) -> Vec<ConvShape> {
    let l = side / 16;
    let c = |what, c_in, c_out, kernel, side, upsample, count| ConvShape {
        what,
        c_in,
        c_out,
        kernel,
        side,
        upsample,
        count,
    };
    vec![
        c("post_quant 1x1", 64, 64, 1, l, false, 1),
        c("conv_in", 64, 1152, 3, l, false, 1),
        c("mid+up0 3x3", 1152, 1152, 3, l, false, 10),
        c("attn to_qkv 1x1", 1152, 3456, 1, l, false, 1),
        c("attn proj 1x1", 1152, 1152, 1, l, false, 1),
        c("up0 upsample 3x3", 1152, 1152, 3, l, true, 1),
        c("up1 3x3", 1152, 1152, 3, 2 * l, false, 6),
        c("up1 upsample 3x3", 1152, 1152, 3, 2 * l, true, 1),
        c("up2 narrow 3x3", 1152, 576, 3, 4 * l, false, 1),
        c("up2 shortcut 1x1", 1152, 576, 1, 4 * l, false, 1),
        c("up2 3x3", 576, 576, 3, 4 * l, false, 5),
        c("up2 upsample 3x3", 576, 576, 3, 4 * l, true, 1),
        c("up3 narrow 3x3", 576, 288, 3, 8 * l, false, 1),
        c("up3 shortcut 1x1", 576, 288, 1, 8 * l, false, 1),
        c("up3 3x3", 288, 288, 3, 8 * l, false, 5),
        c("up3 upsample 3x3", 288, 288, 3, 8 * l, true, 1),
        c("up4 narrow 3x3", 288, 144, 3, 16 * l, false, 1),
        c("up4 shortcut 1x1", 288, 144, 1, 16 * l, false, 1),
        c("up4 3x3", 144, 144, 3, 16 * l, false, 5),
        c("conv_out", 144, 4, 3, 16 * l, false, 1),
    ]
}

fn conv_tflops(s: &ConvShape, ms: f64) -> f64 {
    let out = if s.upsample { 2 * s.side } else { s.side };
    let flops = 2.0 * (out * out) as f64 * (s.c_out * s.c_in * s.kernel * s.kernel) as f64;
    flops / (ms * 1e-3) / 1e12
}

/// Bytes of candle's im2col column buffer for the shape.
fn column_bytes(s: &ConvShape) -> usize {
    let out = if s.upsample { 2 * s.side } else { s.side };
    out * out * s.c_in * s.kernel * s.kernel * 4
}

struct ConvRow {
    candle_ms: Option<f64>,
    direct_ms: f64,
}

/// Both arms at one shape. The candle arm materialises the upsample and then
/// convolves, as the decoder's candle arm does; the direct arm reads at half
/// coordinates. `candle` false skips the im2col arm where its column buffer
/// alone would not fit.
fn bench_conv(dev: &Device, s: ConvShape, candle: bool) -> Result<ConvRow> {
    let x = Tensor::randn(0f32, 1f32, (1, s.c_in, s.side, s.side), dev)?;
    let w = (Tensor::randn(0f32, 1f32, (s.c_out, s.c_in, s.kernel, s.kernel), dev)? * 0.02)?;
    let b = Tensor::zeros(s.c_out, candle_core::DType::F32, dev)?;
    let conv = Conv2d::new(
        w.clone(),
        Some(b.clone()),
        Conv2dConfig {
            padding: s.kernel / 2,
            ..Default::default()
        },
    );
    let candle_ms = if candle {
        Some(time_op(dev, || {
            let v = if s.upsample {
                x.upsample_nearest2d(2 * s.side, 2 * s.side)?
            } else {
                x.clone()
            };
            conv.forward(&v)?;
            Ok(())
        })?)
    } else {
        None
    };
    let pw = ops::permute_conv_weight(&w)?;
    let direct_ms = time_op(dev, || {
        ops::conv2d_direct(
            &x,
            &pw,
            &b,
            s.kernel,
            Conv2dFusion {
                upsample: s.upsample,
                ..Default::default()
            },
        )?;
        Ok(())
    })?;
    Ok(ConvRow {
        candle_ms,
        direct_ms,
    })
}

/// The decoder's norm as it runs: `x / max(|x|_2, eps) * gamma` over the
/// channel axis per pixel, six candle ops.
fn channel_l2_norm(x: &Tensor, gamma: &Tensor) -> candle_core::Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?.maximum(1e-12)?;
    x.broadcast_div(&norm)?.broadcast_mul(gamma)
}

/// (channels, input side in latent units, count) of every norm the decoder
/// runs: two per residual block, one in the attention, one before `conv_out`.
fn decoder_norms(side: usize) -> Vec<(usize, usize, usize)> {
    let l = side / 16;
    vec![
        (1152, l, 4 + 1 + 6),
        (1152, 2 * l, 6),
        (1152, 4 * l, 1),
        (576, 4 * l, 5),
        (576, 8 * l, 1),
        (288, 8 * l, 5),
        (288, 16 * l, 1),
        (144, 16 * l, 5 + 1),
    ]
}

fn print_size(dev: &Device, side: usize, candle: bool) -> Result<()> {
    println!("\n=== {side}x{side} decode, per op ===");
    println!(
        "{:<20} {:>5}>{:<5} {:>5} {:>3} {:>10} {:>8} {:>10} {:>8} {:>9}",
        "conv",
        "c_in",
        "c_out",
        "side",
        "n",
        "candle ms",
        "TFLOP/s",
        "direct ms",
        "TFLOP/s",
        "column GB"
    );
    let mut candle_total = 0.0;
    let mut direct_total = 0.0;
    for s in decoder_convs(side) {
        let row = bench_conv(dev, s, candle)?;
        let n = s.count as f64;
        direct_total += row.direct_ms * n;
        let (cms, ctf) = match row.candle_ms {
            Some(ms) => {
                candle_total += ms * n;
                (format!("{ms:.2}"), format!("{:.2}", conv_tflops(&s, ms)))
            }
            None => ("-".into(), "-".into()),
        };
        println!(
            "{:<20} {:>5}>{:<5} {:>4}{} {:>3} {:>10} {:>8} {:>10.2} {:>8.2} {:>9.2}",
            s.what,
            s.c_in,
            s.c_out,
            s.side,
            if s.upsample { "^" } else { " " },
            s.count,
            cms,
            ctf,
            row.direct_ms,
            conv_tflops(&s, row.direct_ms),
            column_bytes(&s) as f64 / 1e9,
        );
    }

    println!(
        "\n{:<28} {:>3} {:>10} {:>10} {:>10}",
        "norm / silu", "n", "chain ms", "fused ms", "silu ms"
    );
    let mut norm_total = 0.0;
    let mut fused_total = 0.0;
    let mut silu_total = 0.0;
    for (c, s, n) in decoder_norms(side) {
        let x = Tensor::randn(0f32, 1f32, (1, c, s, s), dev)?;
        let gamma = Tensor::ones((1, c, 1, 1), candle_core::DType::F32, dev)?;
        let norm_ms = time_op(dev, || {
            channel_l2_norm(&x, &gamma)?;
            Ok(())
        })?;
        let flat = gamma.flatten_all()?;
        let fused_ms = time_op(dev, || {
            ops::channel_l2_norm(&x, &flat, 1e-12)?;
            Ok(())
        })?;
        fused_total += fused_ms * n as f64;
        let silu_ms = time_op(dev, || {
            x.silu()?;
            Ok(())
        })?;
        norm_total += norm_ms * n as f64;
        // The attention's norm has no silu after it; every other norm does.
        let silus = if c == 1152 && s == side / 16 {
            n - 1
        } else {
            n
        };
        silu_total += silu_ms * silus as f64;
        println!(
            "{:<28} {:>3} {:>10.2} {:>10.2} {:>10.2}",
            format!("{c} ch at {s}x{s}"),
            n,
            norm_ms,
            fused_ms,
            silu_ms
        );
    }

    // The mid block's attention: one head of 1152 over every latent position,
    // the score matrix formed in full.
    let l = side / 16;
    let t = l * l;
    let q = Tensor::randn(0f32, 1f32, (1, t, 1152), dev)?;
    let attn_ms = time_op(dev, || {
        let scores = (q.matmul(&q.transpose(1, 2)?)? * (1.0 / (1152f64).sqrt()))?;
        candle_nn::ops::softmax_last_dim(&scores)?.matmul(&q)?;
        Ok(())
    })?;
    println!(
        "\nmid attention, {t} positions, scores {:.2} GB: {attn_ms:.2} ms",
        (t * t * 4) as f64 / 1e9
    );

    println!("\n--- attribution, one {side}x{side} decode ---");
    if candle {
        println!(
            "candle arm: convs {:.0} ms + norms {:.0} + silus {:.0} + attention {:.0} = {:.0} ms",
            candle_total,
            norm_total,
            silu_total,
            attn_ms,
            candle_total + norm_total + silu_total + attn_ms
        );
    }
    println!(
        "xwen arm: convs {:.0} ms + fused norms {:.0} + attention {:.0} = {:.0} ms (silu read inside the conv)",
        direct_total,
        fused_total,
        attn_ms,
        direct_total + fused_total + attn_ms
    );
    Ok(())
}

#[test]
#[ignore = "GPU microbenchmark; run alone and read docs/benching.md first"]
fn qwen_image_vae_microbench() -> Result<()> {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_metal(0)?;
    print_size(&dev, 512, true)?;
    print_size(&dev, 1024, true)?;
    Ok(())
}

/// The native size, direct arm only: candle's column buffer for the widest
/// full-resolution convolution is 43 GB there.
#[test]
#[ignore = "GPU microbenchmark; run alone and read docs/benching.md first"]
fn qwen_image_vae_microbench_2048_direct() -> Result<()> {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_metal(0)?;
    print_size(&dev, 2048, false)
}
