//! Op-level timings for candle's stock Metal kernels at the exact shapes the
//! Z-Image-Turbo diffusion transformer runs (docs/zimage.md): dim 3840, 30
//! heads of 128, SwiGLU 10240, 34 modulated blocks over T = 4096 + 32 image
//! plus caption tokens at 1024x1024, T = 1024 + 32 at 512x512.
//!
//! ```text
//! cargo test --release --test zimage_microbench -- --ignored --nocapture
//! ```
//!
//! This measures ops in isolation, one class at a time, so a whole-step time
//! can be attributed: what the projections alone cost at the rate the machine
//! actually sustains, what attention costs fused against unfused, and what
//! the elementwise tail costs against memory bandwidth. It loads no weights
//! and reads no checkpoint, so it is safe to run on a clean cache — but it is
//! still a GPU benchmark, so nothing else large may run alongside it
//! (AGENTS.md "Operational hazards").
//!
//! Every figure is per-iteration wall time between two `Device::synchronize`
//! calls around a loop, after three warm-up iterations. Matmul rows report
//! TFLOPS = 2*M*N*K/t; memory-bound rows report the bytes the op must move
//! divided by t.

use anyhow::{Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::Module;

/// Model shape constants, from `docs/zimage.md`.
const DIM: usize = 3840;
const N_HEADS: usize = 30;
const HEAD_DIM: usize = 128;
const FFN_HIDDEN: usize = 10240;
/// Blocks that actually execute per denoising step: 30 `layers` + 2
/// `noise_refiner` + 2 `context_refiner`.
const BLOCKS: usize = 34;
/// Joint sequence lengths: 1024x1024 is 4096 image tokens, 512x512 is 1024,
/// both plus a 32-token caption.
const SEQ_LENS: [usize; 2] = [4128, 1056];

/// Wall time each measured loop aims for. Long enough that a 10 ms op is
/// averaged over dozens of iterations, short enough that the whole file stays
/// under a couple of minutes.
const TARGET_SECS: f64 = 0.7;
const MAX_ITERS: usize = 300;

struct Row {
    group: &'static str,
    what: String,
    seq: usize,
    dtype: &'static str,
    ms: f64,
    iters: usize,
    metric: String,
}

/// Runs `f` to a steady state and returns milliseconds per iteration.
///
/// The single timed calibration iteration is thrown away; it only sizes the
/// measured loop so that a fast op and a slow op are both averaged over
/// enough work to be stable.
fn time_op(dev: &Device, mut f: impl FnMut() -> Result<()>) -> Result<(f64, usize)> {
    for _ in 0..3 {
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
        ((TARGET_SECS / est).ceil() as usize).clamp(3, MAX_ITERS)
    };

    let start = std::time::Instant::now();
    for _ in 0..iters {
        f()?;
    }
    dev.synchronize()?;
    let total = start.elapsed().as_secs_f64();

    Ok((total / iters as f64 * 1e3, iters))
}

fn tflops(m: usize, n: usize, k: usize, ms: f64) -> String {
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    format!("{:.2} TFLOPS", flops / (ms * 1e-3) / 1e12)
}

fn gbps(bytes: usize, ms: f64) -> String {
    format!("{:.0} GB/s", bytes as f64 / (ms * 1e-3) / 1e9)
}

fn rand_tensor(shape: &[usize], dtype: DType, dev: &Device) -> Result<Tensor> {
    Ok(Tensor::randn(0f32, 1f32, shape, dev)?.to_dtype(dtype)?)
}

/// (a) and (b): the projection matmuls, in each layout a caller could choose,
/// in both dtypes.
///
/// The layouts are not interchangeable in candle: `Linear::forward` on a
/// contiguous rank-3 input reshapes to 2-D and calls `matmul` with a
/// transposed view of the stored `[out, in]` weight, while `broadcast_matmul`
/// keeps the batch dimension. Pre-transposing the weight to a contiguous
/// `[in, out]` is the third option and the only one that changes what the
/// kernel sees rather than how it is reached.
fn bench_matmuls(dev: &Device, rows: &mut Vec<Row>) -> Result<()> {
    let shapes: [(usize, usize, &str); 3] = [
        (DIM, DIM, "K=3840 N=3840 (qkv/out)"),
        (DIM, FFN_HIDDEN, "K=3840 N=10240 (ffn up/gate)"),
        (FFN_HIDDEN, DIM, "K=10240 N=3840 (ffn down)"),
    ];

    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let dname = match dtype {
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            _ => "f32",
        };
        for &t in SEQ_LENS.iter() {
            for &(k, n, label) in shapes.iter() {
                let x2 = rand_tensor(&[t, k], dtype, dev)?;
                let x3 = x2.reshape((1, t, k))?;
                // Stored the way every checkpoint stores it: [out, in].
                let w = rand_tensor(&[n, k], dtype, dev)?;
                let wt = w.t()?;
                let wt_contig = w.t()?.contiguous()?;
                let linear = candle_nn::Linear::new(w.clone(), None);

                let (ms, iters) = time_op(dev, || {
                    let _ = x3.broadcast_matmul(&wt)?;
                    Ok(())
                })?;
                rows.push(Row {
                    group: "matmul",
                    what: format!("broadcast_matmul [1,T,K] x w.t()  {label}"),
                    seq: t,
                    dtype: dname,
                    ms,
                    iters,
                    metric: tflops(t, n, k, ms),
                });

                let (ms, iters) = time_op(dev, || {
                    let _ = x2.matmul(&wt)?;
                    Ok(())
                })?;
                rows.push(Row {
                    group: "matmul",
                    what: format!("matmul [T,K] x w.t() (view)      {label}"),
                    seq: t,
                    dtype: dname,
                    ms,
                    iters,
                    metric: tflops(t, n, k, ms),
                });

                let (ms, iters) = time_op(dev, || {
                    let _ = x2.matmul(&wt_contig)?;
                    Ok(())
                })?;
                rows.push(Row {
                    group: "matmul",
                    what: format!("matmul [T,K] x [K,N] contiguous  {label}"),
                    seq: t,
                    dtype: dname,
                    ms,
                    iters,
                    metric: tflops(t, n, k, ms),
                });

                let (ms, iters) = time_op(dev, || {
                    let _ = linear.forward(&x3)?;
                    Ok(())
                })?;
                rows.push(Row {
                    group: "matmul",
                    what: format!("candle_nn::Linear on [1,T,K]     {label}"),
                    seq: t,
                    dtype: dname,
                    ms,
                    iters,
                    metric: tflops(t, n, k, ms),
                });
            }
        }
    }
    Ok(())
}

/// xwen's own Metal-4 cooperative-tensor gemm at the same projection shapes,
/// beside the candle arm.
///
/// `ops::matmul_bf16` is the DFlash drafter's matmul: a bf16 `[n_out, k]`
/// weight in stored orientation against an f32 `[t, k]` activation, f32 out.
/// It is not a drop-in for this graph — the pipeline's activations are bf16,
/// so a caller would pay a widen on the way in and a narrow on the way out —
/// so `bench_conversions` prices that too.
///
/// Which gemm runs is a process-global `OnceLock` over `XWEN_ATTN_MM_CLASSIC`,
/// so the Metal-4 kernel and the classic simdgroup one cannot be compared
/// inside one process. `xwen_bf16_gemm_only` exists to be run twice.
fn bench_xwen_bf16(dev: &Device, rows: &mut Vec<Row>) -> Result<()> {
    let shapes: [(usize, usize, &str); 3] = [
        (DIM, DIM, "K=3840 N=3840 (qkv/out)"),
        (DIM, FFN_HIDDEN, "K=3840 N=10240 (ffn up/gate)"),
        (FFN_HIDDEN, DIM, "K=10240 N=3840 (ffn down)"),
    ];
    let classic = std::env::var("XWEN_ATTN_MM_CLASSIC").is_ok();
    let label = if classic {
        "xwen matmul_bf16 CLASSIC simdgroup"
    } else {
        "xwen matmul_bf16 Metal-4 tensor"
    };

    for &t in SEQ_LENS.iter() {
        for &(k, n, shape) in shapes.iter() {
            // Stored orientation, no transpose, and f32 activations: the
            // kernel's contract, not a layout choice.
            let w = rand_tensor(&[n, k], DType::BF16, dev)?;
            let x = rand_tensor(&[t, k], DType::F32, dev)?;

            match xwen::ops::matmul_bf16(&w, &x) {
                Ok(out) => {
                    assert_eq!(out.dims(), &[t, n]);
                    let (ms, iters) = time_op(dev, || {
                        let _ = xwen::ops::matmul_bf16(&w, &x)?;
                        Ok(())
                    })?;
                    rows.push(Row {
                        group: "xwen-gemm",
                        what: format!("{label}  {shape}"),
                        seq: t,
                        dtype: "bf16xf32",
                        ms,
                        iters,
                        metric: tflops(t, n, k, ms),
                    });
                }
                Err(e) => rows.push(Row {
                    group: "xwen-gemm",
                    what: format!("{label} REFUSED {shape}: {e}"),
                    seq: t,
                    dtype: "bf16xf32",
                    ms: f64::NAN,
                    iters: 0,
                    metric: "-".to_string(),
                }),
            }
        }
    }
    Ok(())
}

/// The two casts a `matmul_bf16` drop-in would pay per linear layer: the
/// activation widened to f32 going in, the result narrowed back to bf16
/// coming out. Priced at the widths the graph uses them at.
fn bench_conversions(dev: &Device, rows: &mut Vec<Row>) -> Result<()> {
    let t = SEQ_LENS[0];
    let cases: [(&str, usize, usize, DType, DType); 4] = [
        (
            "cast in : [T,3840] bf16 -> f32",
            t,
            DIM,
            DType::BF16,
            DType::F32,
        ),
        (
            "cast out: [T,3840] f32 -> bf16",
            t,
            DIM,
            DType::F32,
            DType::BF16,
        ),
        (
            "cast in : [T,10240] bf16 -> f32",
            t,
            FFN_HIDDEN,
            DType::BF16,
            DType::F32,
        ),
        (
            "cast out: [T,10240] f32 -> bf16",
            t,
            FFN_HIDDEN,
            DType::F32,
            DType::BF16,
        ),
    ];
    for (what, rows_n, cols, from, to) in cases {
        let x = rand_tensor(&[rows_n, cols], from, dev)?;
        let (ms, iters) = time_op(dev, || {
            let _ = x.to_dtype(to)?;
            Ok(())
        })?;
        let bytes = rows_n * cols * (from.size_in_bytes() + to.size_in_bytes());
        rows.push(Row {
            group: "conversion",
            what: what.to_string(),
            seq: rows_n,
            dtype: "-",
            ms,
            iters,
            metric: gbps(bytes, ms),
        });
    }
    Ok(())
}

/// The matmul kernel's own ceiling, away from the model's shapes.
///
/// The projection rows below come out around 15 TFLOPS, which is only worth
/// acting on if it is the kernel's limit rather than an artifact of a 4128 x
/// 3840 x 10240 shape. Square GEMMs at three sizes and three dtypes answer
/// that: if bf16 never clears the f32 figure by much, the fast dtype path is
/// not being taken, and if none of the sizes clears the model's own rate,
/// the rate is the kernel.
fn bench_gemm_ceiling(dev: &Device, rows: &mut Vec<Row>) -> Result<()> {
    for n in [2048usize, 4096, 8192] {
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            let dname = match dtype {
                DType::BF16 => "bf16",
                DType::F16 => "f16",
                _ => "f32",
            };
            let a = rand_tensor(&[n, n], dtype, dev)?;
            let b = rand_tensor(&[n, n], dtype, dev)?;
            let (ms, iters) = time_op(dev, || {
                let _ = a.matmul(&b)?;
                Ok(())
            })?;
            rows.push(Row {
                group: "gemm-ceil",
                what: format!("square matmul [{n},{n}] x [{n},{n}]"),
                seq: n,
                dtype: dname,
                ms,
                iters,
                metric: tflops(n, n, n, ms),
            });
        }
    }
    Ok(())
}

/// (c) and (d): attention, unfused piece by piece and through the fused
/// kernel the pipeline actually calls.
fn bench_attention(dev: &Device, rows: &mut Vec<Row>) -> Result<()> {
    for &t in SEQ_LENS.iter() {
        for dtype in [DType::BF16] {
            let dname = "bf16";
            let q = rand_tensor(&[1, N_HEADS, t, HEAD_DIM], dtype, dev)?;
            let k = rand_tensor(&[1, N_HEADS, t, HEAD_DIM], dtype, dev)?;
            let v = rand_tensor(&[1, N_HEADS, t, HEAD_DIM], dtype, dev)?;
            let kt = k.transpose(2, 3)?;
            let scale = 1.0 / (HEAD_DIM as f64).sqrt();

            let (ms, iters) = time_op(dev, || {
                let _ = q.matmul(&kt)?;
                Ok(())
            })?;
            rows.push(Row {
                group: "attention",
                what: "scores = q @ k^T  [1,30,T,T]".to_string(),
                seq: t,
                dtype: dname,
                ms,
                iters,
                metric: tflops(N_HEADS * t, t, HEAD_DIM, ms),
            });

            let scores = (q.matmul(&kt)? * scale)?;
            let (ms, iters) = time_op(dev, || {
                let _ = candle_nn::ops::softmax_last_dim(&scores)?;
                Ok(())
            })?;
            rows.push(Row {
                group: "attention",
                what: "softmax_last_dim over [1,30,T,T]".to_string(),
                seq: t,
                dtype: dname,
                ms,
                iters,
                metric: gbps(2 * N_HEADS * t * t * 2, ms),
            });

            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            let (ms, iters) = time_op(dev, || {
                let _ = probs.matmul(&v)?;
                Ok(())
            })?;
            rows.push(Row {
                group: "attention",
                what: "out = probs @ v".to_string(),
                seq: t,
                dtype: dname,
                ms,
                iters,
                metric: tflops(N_HEADS * t, HEAD_DIM, t, ms),
            });

            let (ms, iters) = time_op(dev, || {
                let s = (q.matmul(&kt)? * scale)?;
                let p = candle_nn::ops::softmax_last_dim(&s)?;
                let _ = p.matmul(&v)?;
                Ok(())
            })?;
            rows.push(Row {
                group: "attention",
                what: "UNFUSED whole chain (matmul/softmax/matmul)".to_string(),
                seq: t,
                dtype: dname,
                ms,
                iters,
                metric: tflops(N_HEADS * t, t, 2 * HEAD_DIM, ms),
            });

            // The softmax in f32, which is what an upcast-for-stability arm
            // would pay on top of the bf16 chain.
            let scores_f32 = scores.to_dtype(DType::F32)?;
            let (ms, iters) = time_op(dev, || {
                let _ = candle_nn::ops::softmax_last_dim(&scores_f32)?;
                Ok(())
            })?;
            rows.push(Row {
                group: "attention",
                what: "softmax_last_dim over [1,30,T,T]".to_string(),
                seq: t,
                dtype: "f32",
                ms,
                iters,
                metric: gbps(2 * N_HEADS * t * t * 4, ms),
            });

            match candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0) {
                Ok(_) => {
                    let (ms, iters) = time_op(dev, || {
                        let _ = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0)?;
                        Ok(())
                    })?;
                    rows.push(Row {
                        group: "attention",
                        what: "FUSED candle_nn::ops::sdpa (no mask)".to_string(),
                        seq: t,
                        dtype: dname,
                        ms,
                        iters,
                        metric: tflops(N_HEADS * t, t, 2 * HEAD_DIM, ms),
                    });
                }
                Err(e) => rows.push(Row {
                    group: "attention",
                    what: format!("FUSED sdpa REFUSED: {e}"),
                    seq: t,
                    dtype: dname,
                    ms: f64::NAN,
                    iters: 0,
                    metric: "-".to_string(),
                }),
            }
        }
    }
    Ok(())
}

/// (e): the elementwise tail, at the activation shape every block carries it
/// at. Bytes counted are the minimum the op must move: inputs read once,
/// output written once.
fn bench_elementwise(dev: &Device, rows: &mut Vec<Row>) -> Result<()> {
    for &t in SEQ_LENS.iter() {
        let n = t * DIM;
        let x = rand_tensor(&[t, DIM], DType::BF16, dev)?;
        let y = rand_tensor(&[t, DIM], DType::BF16, dev)?;
        let row = rand_tensor(&[1, DIM], DType::BF16, dev)?;
        let alpha = rand_tensor(&[DIM], DType::BF16, dev)?;
        let x_f32 = x.to_dtype(DType::F32)?;

        let cases: Vec<(&'static str, usize, Box<dyn Fn() -> Result<()>>)> = vec![
            (
                "rms_norm",
                2 * n * 2,
                Box::new({
                    let x = x.clone();
                    let alpha = alpha.clone();
                    move || {
                        let _ = candle_nn::ops::rms_norm(&x, &alpha, 1e-5)?;
                        Ok(())
                    }
                }),
            ),
            (
                "silu",
                2 * n * 2,
                Box::new({
                    let x = x.clone();
                    move || {
                        let _ = x.silu()?;
                        Ok(())
                    }
                }),
            ),
            (
                "mul (elementwise, two [T,3840])",
                3 * n * 2,
                Box::new({
                    let x = x.clone();
                    let y = y.clone();
                    move || {
                        let _ = (&x * &y)?;
                        Ok(())
                    }
                }),
            ),
            (
                "add (elementwise, two [T,3840])",
                3 * n * 2,
                Box::new({
                    let x = x.clone();
                    let y = y.clone();
                    move || {
                        let _ = (&x + &y)?;
                        Ok(())
                    }
                }),
            ),
            (
                "broadcast_mul by [1,3840] row",
                2 * n * 2,
                Box::new({
                    let x = x.clone();
                    let row = row.clone();
                    move || {
                        let _ = x.broadcast_mul(&row)?;
                        Ok(())
                    }
                }),
            ),
            (
                // What the same modulation costs if the row is materialised
                // to the full activation shape first and multiplied with the
                // plain elementwise kernel. Prices the workaround for
                // broadcast_mul's rate against its own extra write.
                "broadcast_as().contiguous() then mul",
                4 * n * 2,
                Box::new({
                    let x = x.clone();
                    let row = row.clone();
                    move || {
                        let full = row.broadcast_as(x.shape())?.contiguous()?;
                        let _ = (&x * &full)?;
                        Ok(())
                    }
                }),
            ),
            (
                "to_dtype bf16 -> f32",
                n * 2 + n * 4,
                Box::new({
                    let x = x.clone();
                    move || {
                        let _ = x.to_dtype(DType::F32)?;
                        Ok(())
                    }
                }),
            ),
            (
                "to_dtype f32 -> bf16",
                n * 4 + n * 2,
                Box::new({
                    let x_f32 = x_f32.clone();
                    move || {
                        let _ = x_f32.to_dtype(DType::BF16)?;
                        Ok(())
                    }
                }),
            ),
        ];

        for (what, bytes, f) in cases {
            let (ms, iters) = time_op(dev, || f())?;
            rows.push(Row {
                group: "elementwise",
                what: what.to_string(),
                seq: t,
                dtype: "bf16",
                ms,
                iters,
                metric: gbps(bytes, ms),
            });
        }
    }
    Ok(())
}

/// (f): per-dispatch overhead. 1000 independent tiny adds, one synchronize at
/// the end, so the figure is the host-side cost of getting a kernel onto the
/// queue rather than anything the GPU does with it.
fn bench_launch_overhead(dev: &Device, rows: &mut Vec<Row>) -> Result<()> {
    let a = rand_tensor(&[1, 64], DType::BF16, dev)?;
    let b = rand_tensor(&[1, 64], DType::BF16, dev)?;
    const N: usize = 1000;

    for _ in 0..3 {
        for _ in 0..N {
            let _ = (&a + &b)?;
        }
    }
    dev.synchronize()?;

    let start = std::time::Instant::now();
    for _ in 0..N {
        let _ = (&a + &b)?;
    }
    dev.synchronize()?;
    let total = start.elapsed().as_secs_f64();

    rows.push(Row {
        group: "overhead",
        what: format!("{N} tiny [1,64] adds, one synchronize"),
        seq: 1,
        dtype: "bf16",
        ms: total / N as f64 * 1e3,
        iters: N,
        metric: format!("{:.0} us/dispatch", total / N as f64 * 1e6),
    });
    Ok(())
}

fn print_table(rows: &[Row]) {
    println!(
        "\n{:<11} {:<6} {:<6} {:<48} {:>10} {:>7} {:>16}",
        "group", "T", "dtype", "op", "ms/iter", "iters", "rate"
    );
    println!("{}", "-".repeat(110));
    for r in rows {
        let ms = if r.ms.is_nan() {
            "-".to_string()
        } else {
            format!("{:.3}", r.ms)
        };
        println!(
            "{:<11} {:<6} {:<6} {:<48} {:>10} {:>7} {:>16}",
            r.group, r.seq, r.dtype, r.what, ms, r.iters, r.metric
        );
    }
}

/// Attributes a whole denoising step to op classes using the rates just
/// measured, so the table can be read against the 5.1 s / 1.2 s step totals
/// in docs/perf-state.md.
fn print_attribution(rows: &[Row]) {
    // Per-block projection parameters: q, k, v and out at dim^2 each for the
    // first three plus one more, then SwiGLU's gate, up and down.
    let proj_params = 3 * DIM * DIM + DIM * DIM + 2 * DIM * FFN_HIDDEN + FFN_HIDDEN * DIM;

    println!("\n=== step attribution at the measured rates ===");
    println!(
        "per-block projection params: {:.1} M",
        proj_params as f64 / 1e6
    );

    for &t in SEQ_LENS.iter() {
        let proj_flops = 2.0 * proj_params as f64 * t as f64 * BLOCKS as f64;
        // Two matmuls per head over the T x T score matrix.
        let attn_flops =
            4.0 * N_HEADS as f64 * HEAD_DIM as f64 * (t as f64) * (t as f64) * BLOCKS as f64;

        // The best projection rate measured at this T, in bf16.
        let best = rows
            .iter()
            .filter(|r| r.group == "matmul" && r.seq == t && r.dtype == "bf16")
            .filter_map(|r| {
                r.metric
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse::<f64>().ok())
            })
            .fold(0f64, f64::max);

        let fused = rows
            .iter()
            .find(|r| r.group == "attention" && r.seq == t && r.what.starts_with("FUSED c"))
            .map(|r| r.ms);
        let unfused = rows
            .iter()
            .find(|r| r.group == "attention" && r.seq == t && r.what.starts_with("UNFUSED"))
            .map(|r| r.ms);

        println!("\nT = {t}");
        println!("  projection FLOPs / step : {:.2} TFLOP", proj_flops / 1e12);
        println!("  attention FLOPs / step  : {:.2} TFLOP", attn_flops / 1e12);
        println!("  best measured bf16 matmul rate: {best:.2} TFLOPS");
        if best > 0.0 {
            println!(
                "  projections alone would take {:.2} s at that rate",
                proj_flops / (best * 1e12)
            );
        }
        if let Some(ms) = fused {
            println!(
                "  fused sdpa, {BLOCKS} blocks   : {:.2} s ({:.3} ms/block)",
                ms * BLOCKS as f64 / 1e3,
                ms
            );
        }
        if let Some(ms) = unfused {
            println!(
                "  unfused chain, {BLOCKS} blocks: {:.2} s ({:.3} ms/block)",
                ms * BLOCKS as f64 / 1e3,
                ms
            );
        }
    }
}

/// Prices a `matmul_bf16` drop-in against the candle arm it would replace,
/// including the casts its f32 activation contract forces.
///
/// Two cast counts, because they bracket what an implementation would pay. The
/// naive one widens the input of every linear and narrows every output. The
/// shared one widens once per distinct activation — one cast feeds q, k and v,
/// one feeds the SwiGLU gate and up — which is what a careful port would do
/// without restructuring anything else.
fn print_dropin_accounting(rows: &[Row]) {
    let t = SEQ_LENS[0];
    let find = |group: &str, needle: &str, seq: usize| -> Option<f64> {
        rows.iter()
            .find(|r| r.group == group && r.seq == seq && r.what.contains(needle) && !r.ms.is_nan())
            .map(|r| r.ms)
    };

    println!("\n=== matmul_bf16 drop-in accounting at T = {t} ===");

    // Per-linear times at each of the three projection shapes, candle bf16
    // (2-D contiguous, the layout Linear actually reaches) against xwen.
    let shapes = [
        ("K=3840 N=3840", 1usize, 3usize), // one out_proj, plus q/k/v below
        ("K=3840 N=10240", 2, 0),
        ("K=10240 N=3840", 1, 0),
    ];

    let mut candle_total = 0.0;
    let mut xwen_total = 0.0;
    for (shape, extra, qkv) in shapes {
        let count = extra + qkv;
        let Some(c) = find("matmul", &format!("contiguous  {shape}"), t) else {
            println!("  missing candle row for {shape}");
            continue;
        };
        let Some(x) = find("xwen-gemm", shape, t) else {
            println!("  missing xwen row for {shape}");
            continue;
        };
        println!(
            "  {shape}: {count} per block, candle {c:.3} ms, xwen {x:.3} ms ({:.2}x)",
            c / x
        );
        candle_total += c * count as f64;
        xwen_total += x * count as f64;
    }

    let in_3840 = find("conversion", "cast in : [T,3840]", t).unwrap_or(0.0);
    let out_3840 = find("conversion", "cast out: [T,3840]", t).unwrap_or(0.0);
    let in_10240 = find("conversion", "cast in : [T,10240]", t).unwrap_or(0.0);
    let out_10240 = find("conversion", "cast out: [T,10240]", t).unwrap_or(0.0);

    // Naive: q, k, v, out, gate and up read a 3840-wide activation; down reads
    // a 10240-wide one. q, k, v, out and down write 3840 wide; gate and up
    // write 10240 wide.
    let naive_cast = 6.0 * in_3840 + in_10240 + 5.0 * out_3840 + 2.0 * out_10240;
    // Shared: one widen for q/k/v, one for out_proj's input, one for the
    // SwiGLU pair, one for down's input. Outputs are unchanged.
    let shared_cast = 3.0 * in_3840 + in_10240 + 5.0 * out_3840 + 2.0 * out_10240;

    let b = BLOCKS as f64;
    println!("\n  per block (7 token-width linears; the adaLN linear is a [1,3840] vector):");
    println!(
        "    candle bf16 linears              : {:.3} ms",
        candle_total
    );
    println!(
        "    xwen matmul_bf16 linears         : {:.3} ms",
        xwen_total
    );
    println!(
        "    casts, naive (14 = 2 per linear)  : {:.3} ms",
        naive_cast
    );
    println!(
        "    casts, shared inputs (11)         : {:.3} ms",
        shared_cast
    );

    println!("\n  per step, {BLOCKS} blocks:");
    println!(
        "    candle bf16                      : {:.3} s",
        candle_total * b / 1e3
    );
    println!(
        "    xwen + naive casts               : {:.3} s ({:.2}x)",
        (xwen_total + naive_cast) * b / 1e3,
        candle_total / (xwen_total + naive_cast)
    );
    println!(
        "    xwen + shared casts              : {:.3} s ({:.2}x)",
        (xwen_total + shared_cast) * b / 1e3,
        candle_total / (xwen_total + shared_cast)
    );
    println!(
        "    xwen, casts excluded (f32 block) : {:.3} s ({:.2}x)",
        xwen_total * b / 1e3,
        candle_total / xwen_total
    );
}

/// The xwen gemm arm alone, so it can be run a second time under
/// `XWEN_ATTN_MM_CLASSIC=1`: the kernel choice is a process-global `OnceLock`
/// and cannot be switched inside one process.
#[test]
#[ignore = "GPU microbenchmark; run alone"]
fn xwen_bf16_gemm_only() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut rows = Vec::new();
    bench_xwen_bf16(&dev, &mut rows)?;
    print_table(&rows);
    Ok(())
}

#[test]
#[ignore = "GPU microbenchmark; run alone and read docs/benching.md first"]
fn zimage_op_microbench() -> Result<()> {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(e) => bail!("this benchmark needs a Metal device: {e}"),
    };

    println!("pmset -g, verbatim as of this run:");
    let out = std::process::Command::new("pmset").arg("-g").output();
    match out {
        Ok(o) => print!("{}", String::from_utf8_lossy(&o.stdout)),
        Err(e) => println!("  (pmset unavailable: {e})"),
    }
    println!(
        "\nNote: neither `lowpowermode` nor `powermode` can confirm high-power mode; \
         no such claim is made here (AGENTS.md \"Perf state\")."
    );

    let mut rows = Vec::new();
    bench_gemm_ceiling(&dev, &mut rows)?;
    bench_matmuls(&dev, &mut rows)?;
    bench_xwen_bf16(&dev, &mut rows)?;
    bench_conversions(&dev, &mut rows)?;
    bench_attention(&dev, &mut rows)?;
    bench_elementwise(&dev, &mut rows)?;
    bench_launch_overhead(&dev, &mut rows)?;

    print_table(&rows);
    print_attribution(&rows);
    print_dropin_accounting(&rows);
    Ok(())
}
