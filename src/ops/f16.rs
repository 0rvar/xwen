use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

/// Dense f16-weight x f32-activation matmul against the vendored ggml-geometry
/// kernels — the attention projections. `weight` is a rank-2 `[n_out, k]` dense
/// f16 tensor, `x` is `[t, k]` f32; returns `[t, n_out]` f32. Semantically the
/// fork's mixed-dtype mul_mat: f32 products/accumulation, f32 output.
///
/// Dispatches the classic mat-vec (`f16.metal`) for t <= 8 tokens; above that,
/// the prefill gemm — by default the Metal-4 cooperative-tensor kernel
/// (`f16_t.metal`, ggml's dense geometry), or the classic simdgroup kernel
/// (`f16.metal`) under the `XWEN_ATTN_MM_CLASSIC` kill-switch. The tensor
/// prefill gemm reads the f32 activation directly from device memory (no
/// threadgroup staging — the throughput win), but its reduced-precision
/// cooperative-tensor path computes at reduced precision, so it carries the
/// fork's ~2e-4 prefill precision class (docs/parity.md §3b), not f32
/// accumulation-order noise. Metal only; the caller's fallback is the dequant-f32
/// `QMatMul` path (`XWEN_ATTN_F32`), which bypasses this module entirely.
pub fn matmul_f16(weight: &Tensor, x: &Tensor) -> Result<Tensor> {
    dispatch::run_matmul_f16(weight, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::ops::dispatch::F16MmKernel;
    use crate::ops::dispatch::testutil::{max_abs, pseudo_random, rel_l2};
    use candle_core::{DType, Device, Tensor};

    /// Kernel output vs a CPU f32 reference matmul over the SAME f16-rounded
    /// weights, on the CLASSIC path (float tiles): the kernel's only rounding is
    /// the stored weights, which the reference shares, so the residual is pure f32
    /// accumulation-order noise. The tensor prefill kernel's reduced-precision
    /// cooperative-tensor path adds ~2e-4 error on top, so it is graded against the
    /// classic kernel (not the tight f32 reference) in `f16_tensor_matches_classic`.
    /// `run_shape` pins Classic; `run_shape_kernel` selects the variant.
    fn run_shape(n_out: usize, k: usize, t: usize, seed: u64) -> f32 {
        run_shape_kernel(n_out, k, t, F16MmKernel::Classic, seed)
    }

    fn run_shape_kernel(n_out: usize, k: usize, t: usize, kernel: F16MmKernel, seed: u64) -> f32 {
        let device = metal_device().unwrap();
        let cpu = Device::Cpu;

        let w = Tensor::from_vec(pseudo_random(n_out * k, seed, -0.5, 0.5), (n_out, k), &cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let x =
            Tensor::from_vec(pseudo_random(t * k, seed ^ 0xF00D, -1.0, 1.0), (t, k), &cpu).unwrap();

        let got = dispatch::run_matmul_f16_variant(
            &w.to_device(&device).unwrap(),
            &x.to_device(&device).unwrap(),
            kernel,
        )
        .unwrap();
        assert_eq!(got.dims(), &[t, n_out]);
        assert_eq!(got.dtype(), DType::F32);
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let want = x
            .matmul(&w.to_dtype(DType::F32).unwrap().t().unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        rel_l2(&got, &want)
    }

    // The classic kernels accumulate in f32 with f16 rounding only in the stored
    // weights (which the reference shares), so the measured error is f32
    // accumulation-order noise: rel_l2 3.3e-7..7.1e-7 on the gemv and
    // 9.3e-7..1.8e-6 on the gemm (worst at t=512, K=9216) across the shapes
    // below. Bound at 1e-5 (~5x headroom over the worst).
    const TOL: f32 = 1e-5;

    /// Decode gemv (t <= 8) at every production projection shape, including
    /// the tiny 48/72-row gate projections and the o_proj K=9216 shape.
    #[test]
    fn f16_mv_production_shapes() {
        for (n_out, k) in [
            (6144, 3072),
            (9216, 3072),
            (1024, 3072),
            (48, 3072),
            (72, 3072),
            (3072, 9216),
        ] {
            let rel = run_shape(n_out, k, 1, 0x51 + n_out as u64);
            assert!(rel < TOL, "mv [{n_out}x{k}] t=1 rel_l2 {rel}");
        }
        // The mv/mm boundary: t = 8 is the last gemv seq.
        let rel = run_shape(1024, 3072, 8, 0x61);
        assert!(rel < TOL, "mv t=8 rel_l2 {rel}");
    }

    /// Classic prefill gemm (t > 8): the first mm seq (9), a real fixture seq
    /// (58, matching the code-short parity prompt), and a full 512-token chunk,
    /// over production out-dims including the sub-tile 48/72 gate projections
    /// (nr0 < 64: guarded store-back) and the o_proj K=6144/9216 shapes.
    #[test]
    fn f16_mm_production_shapes() {
        for (n_out, k, t) in [
            (1024, 3072, 9),
            (9216, 3072, 58),
            (48, 3072, 58),
            (72, 3072, 58),
            (3072, 6144, 58),
            (6144, 3072, 512),
            (3072, 9216, 512),
        ] {
            let rel = run_shape(n_out, k, t, 0x71 + n_out as u64 + t as u64);
            assert!(rel < TOL, "mm [{n_out}x{k}] t={t} rel_l2 {rel}");
        }
    }

    /// The two kernels are one op behind a seq threshold: at adjacent seqs the
    /// gemv (t=8) and the CLASSIC gemm (t=9) must agree with the shared reference
    /// to the same bound (implicitly covered above) AND with each other row-for-row
    /// on the overlapping tokens.
    #[test]
    fn f16_mv_mm_boundary_agrees() {
        let device = metal_device().unwrap();
        let cpu = Device::Cpu;
        let (n_out, k) = (1024, 3072);
        let w = Tensor::from_vec(pseudo_random(n_out * k, 0x81, -0.5, 0.5), (n_out, k), &cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&device)
            .unwrap();
        let x9 = Tensor::from_vec(pseudo_random(9 * k, 0x82, -1.0, 1.0), (9, k), &device).unwrap();
        let mm = dispatch::run_matmul_f16_variant(&w, &x9, F16MmKernel::Classic).unwrap(); // t=9: classic gemm
        let mv = matmul_f16(&w, &x9.narrow(0, 0, 8).unwrap()).unwrap(); // t=8: gemv
        let a = mm
            .narrow(0, 0, 8)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = mv.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rel = rel_l2(&a, &b);
        assert!(rel < TOL, "mv/mm boundary rel_l2 {rel}");
    }

    /// The production attention projection shapes at a full prefill chunk, as
    /// (n_out, k): SWA q (72h -> 9216), FULL q (48h -> 6144), k/v (8kv -> 1024),
    /// o_proj (9216 -> 3072), and the sub-tile gate projections (48/72 rows, the
    /// guarded store-back). Every shape the tensor prefill gemm runs in production.
    const PREFILL_SHAPES: [(usize, usize); 6] = [
        (9216, 3072),
        (6144, 3072),
        (1024, 3072),
        (3072, 9216),
        (48, 3072),
        (72, 3072),
    ];

    /// The shipped tensor prefill gemm vs the classic simdgroup gemm on every
    /// production projection shape at a 512-token chunk. The tensor kernel reads
    /// the f32 activation directly from device memory (ggml's dense geometry) but
    /// runs the reduced-precision cooperative-tensor path (the descriptor flag
    /// ggml sets for ~2x throughput), so it carries the fork's ~2e-4 prefill
    /// precision relative to the classic float-tile kernel (both share the f16
    /// weights). Bound at 5e-4 (the fork's prefill precision class). This is the
    /// transitive correctness link: classic is pinned to the f32 CPU reference
    /// above, tensor is pinned to classic here.
    #[test]
    fn f16_tensor_matches_classic() {
        let device = metal_device().unwrap();
        const T: usize = 512;
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for (n_out, k) in PREFILL_SHAPES {
            let w = Tensor::from_vec(
                pseudo_random(n_out * k, 0x300 + n_out as u64, -0.5, 0.5),
                (n_out, k),
                &device,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
            let x = Tensor::from_vec(
                pseudo_random(T * k, 0x400 + n_out as u64, -1.0, 1.0),
                (T, k),
                &device,
            )
            .unwrap();

            let tensor =
                flat(&dispatch::run_matmul_f16_variant(&w, &x, F16MmKernel::Tensor).unwrap());
            let classic =
                flat(&dispatch::run_matmul_f16_variant(&w, &x, F16MmKernel::Classic).unwrap());

            // rel_l2 is the relative (scale-invariant) error, ~2e-4 on the worst
            // shape — the fork's own prefill precision class. max_abs is a raw
            // absolute diff (diagnostic only): it scales with the output magnitude
            // (K=3072..9216 dot products), so a small relative error still reads as
            // a larger absolute number there.
            let rel = rel_l2(&tensor, &classic);
            let mabs = max_abs(&tensor, &classic);
            assert!(
                rel < 5e-4,
                "tensor vs classic [{n_out}x{k}] t={T}: rel_l2 {rel} (max_abs {mabs})"
            );
        }
    }

    /// The shipped tensor prefill gemm (t > 8) vs the classic simdgroup gemm on the
    /// tile-edge cases ggml's cooperative-tensor extents must cover, which the
    /// 512-token `PREFILL_SHAPES` sweep above does not: out-edge tiles (48/72 rows,
    /// not a multiple of NR0=64), token-edge tiles (413 = a ragged tail past the
    /// NR1=128 token tile; 16 = a single sub-128 tile), and the k/v out-dim. Same
    /// precision class as `f16_tensor_matches_classic` (5e-4, the fork's prefill
    /// class); classic is itself pinned to the f32 CPU reference in `run_shape`.
    #[test]
    fn f16_tensor_edge_tiles_match_classic() {
        let device = metal_device().unwrap();
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for (n_out, k, t) in [
            (9216, 3072, 16),
            (6144, 3072, 128),
            (1024, 3072, 413),
            (48, 3072, 413),
            (72, 3072, 413),
            (3072, 9216, 512),
        ] {
            let seed = 0x91 + n_out as u64 + t as u64;
            let w = Tensor::from_vec(
                pseudo_random(n_out * k, seed, -0.5, 0.5),
                (n_out, k),
                &device,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
            let x = Tensor::from_vec(
                pseudo_random(t * k, seed ^ 0xF00D, -1.0, 1.0),
                (t, k),
                &device,
            )
            .unwrap();

            let tensor =
                flat(&dispatch::run_matmul_f16_variant(&w, &x, F16MmKernel::Tensor).unwrap());
            let classic =
                flat(&dispatch::run_matmul_f16_variant(&w, &x, F16MmKernel::Classic).unwrap());
            let rel = rel_l2(&tensor, &classic);
            assert!(rel < 5e-4, "tensor edge [{n_out}x{k}] t={t}: rel_l2 {rel}");
        }
    }

    /// PROBE: does matmul2d accept MIXED operand element types — half weight
    /// tile x FLOAT activation tile, f32 accumulate (`f16_t_mixed.metal`)? The
    /// MPP header documents `float(left) x half(right) -> float` as supported;
    /// this test is the empirical check. A compile rejection surfaces as a loud
    /// failure carrying the compiler diagnostic (that outcome IS the probe's
    /// answer). On success the kernel's inputs are bit-identical to the classic
    /// simdgroup kernel's (both consume the stored f16 weights and the raw f32
    /// activations — no staging rounding), so the residual is pure f32 tile
    /// accumulation-order noise, the same ~1e-6 class as classic-vs-CPU above;
    /// bound at the shared 1e-5 TOL. A ~2e-4 result here would mean the
    /// implementation silently staged the float operand to half (the f16_t.metal
    /// precision class) — that must FAIL: it would make the mixed kernel
    /// pointless as a default candidate.
    #[test]
    fn f16_tensor_mixed_matches_classic() {
        let device = metal_device().unwrap();
        const T: usize = 512;
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut worst = 0f32;

        for (n_out, k) in PREFILL_SHAPES {
            let w = Tensor::from_vec(
                pseudo_random(n_out * k, 0x300 + n_out as u64, -0.5, 0.5),
                (n_out, k),
                &device,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
            let x = Tensor::from_vec(
                pseudo_random(T * k, 0x400 + n_out as u64, -1.0, 1.0),
                (T, k),
                &device,
            )
            .unwrap();

            let mixed = match dispatch::run_matmul_f16_variant(&w, &x, F16MmKernel::TensorMixed) {
                Ok(t) => flat(&t),
                Err(e) => panic!(
                    "PROBE ANSWER: mixed-operand matmul2d (half weights x float activations) \
                     did NOT compile/dispatch:\n{e:#}"
                ),
            };
            let classic =
                flat(&dispatch::run_matmul_f16_variant(&w, &x, F16MmKernel::Classic).unwrap());

            let rel = rel_l2(&mixed, &classic);
            let mabs = max_abs(&mixed, &classic);
            eprintln!(
                "mixed vs classic [{n_out}x{k}] t={T}: rel_l2 {rel:.3e} (max_abs {mabs:.3e})"
            );
            worst = worst.max(rel);
            assert!(
                rel < TOL,
                "mixed vs classic [{n_out}x{k}] t={T}: rel_l2 {rel} (max_abs {mabs}) — \
                 ~2e-4 here means the float activation operand was silently rounded to half"
            );
        }
        eprintln!("mixed vs classic worst rel_l2 {worst:.3e}");
    }

    #[test]
    fn f16_shape_and_dtype_errors() {
        let device = metal_device().unwrap();
        let w = Tensor::zeros((64, 32), DType::F16, &device).unwrap();
        // k mismatch.
        let x = Tensor::zeros((1, 64), DType::F32, &device).unwrap();
        assert!(matmul_f16(&w, &x).is_err());
        // f32 weight (must be pre-cast f16 at load, not here).
        let wf = Tensor::zeros((64, 32), DType::F32, &device).unwrap();
        let x = Tensor::zeros((1, 32), DType::F32, &device).unwrap();
        assert!(matmul_f16(&wf, &x).is_err());
        // k not a multiple of 32 (the kernels have no K tail at our shapes).
        let w20 = Tensor::zeros((64, 20), DType::F16, &device).unwrap();
        let x20 = Tensor::zeros((1, 20), DType::F32, &device).unwrap();
        assert!(matmul_f16(&w20, &x20).is_err());
    }

    /// Isolation timing: classic simdgroup gemm vs Metal-4 cooperative-tensor gemm
    /// on each production projection shape at a 512-token chunk. `#[ignore]`d —
    /// run on a `pgrep`-verified free GPU with:
    ///   cargo test --release -p xwen f16_tensor_vs_classic_timing -- --ignored --nocapture
    /// `XWEN_BENCH_WARMUP` / `XWEN_BENCH_ITERS` override the loop counts.
    #[test]
    #[ignore = "perf bench"]
    fn f16_tensor_vs_classic_timing() {
        use std::time::Instant;

        let device = metal_device().unwrap();
        const T: usize = 512;
        let read_scalar = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        let get = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let (warm, iters) = (get("XWEN_BENCH_WARMUP", 10), get("XWEN_BENCH_ITERS", 100));

        // Warm-up then a timed loop, each iter ending in a small readback (the
        // per-iter command-buffer flush). Returns (mean, plateau = mean of the last
        // half) ms/iter — the LPM burst→clamp makes the plateau the honest figure.
        let bench = |name: &str, mut f: Box<dyn FnMut() -> f32>| {
            let mut sink = 0f32;
            for _ in 0..warm {
                sink += f();
            }
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = Instant::now();
                sink += f();
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            let plateau: f64 = times[iters / 2..].iter().sum::<f64>() / (iters - iters / 2) as f64;
            eprintln!("{name}: mean {mean:.3} ms | plateau {plateau:.3} ms (sink {sink:.1})");
        };

        for (n_out, k) in PREFILL_SHAPES {
            let w = Tensor::from_vec(
                pseudo_random(n_out * k, 0x100 + n_out as u64, -0.5, 0.5),
                (n_out, k),
                &device,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
            let x = Tensor::from_vec(
                pseudo_random(T * k, 0x200 + n_out as u64, -1.0, 1.0),
                (T, k),
                &device,
            )
            .unwrap();

            eprintln!("--- [{n_out}x{k}] t={T} ---");
            let (w0, x0) = (w.clone(), x.clone());
            bench(
                "  classic",
                Box::new(move || {
                    read_scalar(
                        &dispatch::run_matmul_f16_variant(&w0, &x0, F16MmKernel::Classic).unwrap(),
                    )
                }),
            );
            let (w1, x1) = (w.clone(), x.clone());
            bench(
                "  tensor ",
                Box::new(move || {
                    read_scalar(
                        &dispatch::run_matmul_f16_variant(&w1, &x1, F16MmKernel::Tensor).unwrap(),
                    )
                }),
            );
        }
    }

    /// Amortized isolation timing: a batch of back-to-back dispatches per timed
    /// round with a single flush+readback at the round boundary (all outputs
    /// held alive until the readback so the buffer pool cannot recycle one
    /// mid-round and inject a false WAW barrier). This mirrors how a real
    /// prefill forward issues the gemm — many dispatches per command buffer, no
    /// per-op sync — and how ggml's `test-backend-ops perf` measures, so these
    /// are the numbers to put next to the fork's per-op figures. The
    /// per-iter-synced `f16_tensor_vs_classic_timing` above additionally pays
    /// per-dispatch submit+sync latency the production path never pays per op.
    #[test]
    #[ignore = "perf bench"]
    fn f16_amortized_timing() {
        use std::time::Instant;

        let device = metal_device().unwrap();
        const T: usize = 512;
        const BATCH: usize = 32;
        let read_scalar = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        let get = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let (warm, rounds) = (get("XWEN_BENCH_WARMUP", 3), get("XWEN_BENCH_ITERS", 20));

        for (n_out, k) in PREFILL_SHAPES {
            let w = Tensor::from_vec(
                pseudo_random(n_out * k, 0x100 + n_out as u64, -0.5, 0.5),
                (n_out, k),
                &device,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
            let x = Tensor::from_vec(
                pseudo_random(T * k, 0x200 + n_out as u64, -1.0, 1.0),
                (T, k),
                &device,
            )
            .unwrap();

            eprintln!("--- [{n_out}x{k}] t={T}, {BATCH} dispatches/round ---");
            for (name, kernel) in [
                ("classic", F16MmKernel::Classic),
                ("tensor ", F16MmKernel::Tensor),
            ] {
                let mut sink = 0f32;
                let mut round = || {
                    let outs: Vec<Tensor> = (0..BATCH)
                        .map(|_| dispatch::run_matmul_f16_variant(&w, &x, kernel).unwrap())
                        .collect();
                    read_scalar(outs.last().unwrap())
                };
                for _ in 0..warm {
                    sink += round();
                }
                let mut times = Vec::with_capacity(rounds);
                for _ in 0..rounds {
                    let t = Instant::now();
                    sink += round();
                    times.push(t.elapsed().as_secs_f64() * 1e3 / BATCH as f64);
                }
                let mean = times.iter().sum::<f64>() / times.len() as f64;
                let plateau: f64 =
                    times[rounds / 2..].iter().sum::<f64>() / (rounds - rounds / 2) as f64;
                eprintln!(
                    "  {name}: mean {mean:.3} ms | plateau {plateau:.3} ms per dispatch (sink {sink:.1})"
                );
            }
        }
    }

    /// 3-way isolation timing: classic simdgroup vs half-tile tensor vs the
    /// mixed-operand tensor probe, on each production projection shape at a
    /// 512-token chunk. Unlike `f16_tensor_vs_classic_timing`'s back-to-back
    /// per-variant loops, the variants are INTERLEAVED inside one timed loop so
    /// every variant samples the same DVFS/LPM clock trajectory. `#[ignore]`d —
    /// run on a `pgrep`-verified free GPU with:
    ///   cargo test --release -p xwen f16_tensor_mixed_vs_classic_timing -- --ignored --nocapture
    /// `XWEN_BENCH_WARMUP` / `XWEN_BENCH_ITERS` override the loop counts.
    #[test]
    #[ignore = "perf bench"]
    fn f16_tensor_mixed_vs_classic_timing() {
        use std::time::Instant;

        let device = metal_device().unwrap();
        const T: usize = 512;
        let read_scalar = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        let get = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let (warm, iters) = (get("XWEN_BENCH_WARMUP", 10), get("XWEN_BENCH_ITERS", 100));

        const VARIANTS: [(&str, F16MmKernel); 3] = [
            ("classic", F16MmKernel::Classic),
            ("tensor ", F16MmKernel::Tensor),
            ("mixed  ", F16MmKernel::TensorMixed),
        ];

        for (n_out, k) in PREFILL_SHAPES {
            let w = Tensor::from_vec(
                pseudo_random(n_out * k, 0x100 + n_out as u64, -0.5, 0.5),
                (n_out, k),
                &device,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
            let x = Tensor::from_vec(
                pseudo_random(T * k, 0x200 + n_out as u64, -1.0, 1.0),
                (T, k),
                &device,
            )
            .unwrap();

            let mut sink = 0f32;
            let run = |kernel: F16MmKernel| {
                read_scalar(&dispatch::run_matmul_f16_variant(&w, &x, kernel).unwrap())
            };
            for _ in 0..warm {
                for (_, kernel) in VARIANTS {
                    sink += run(kernel);
                }
            }

            // One timed loop, variants interleaved per iteration; each dispatch
            // ends in a small readback (the per-iter command-buffer flush).
            let mut times: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(iters));
            for _ in 0..iters {
                for (i, (_, kernel)) in VARIANTS.iter().enumerate() {
                    let t0 = Instant::now();
                    sink += run(*kernel);
                    times[i].push(t0.elapsed().as_secs_f64() * 1e3);
                }
            }

            eprintln!("--- [{n_out}x{k}] t={T} ---");
            for (i, (name, _)) in VARIANTS.iter().enumerate() {
                let mean = times[i].iter().sum::<f64>() / times[i].len() as f64;
                let plateau: f64 =
                    times[i][iters / 2..].iter().sum::<f64>() / (iters - iters / 2) as f64;
                eprintln!("  {name}: mean {mean:.3} ms | plateau {plateau:.3} ms (sink {sink:.1})");
            }
        }
    }
}
