use anyhow::Result;
use candle_core::Tensor;
use candle_core::quantized::GgmlDType;

use crate::gguf::QuantPlane;
use crate::ops::dispatch;

/// Dense quantized-weight x f32-activation matmul against the vendored Metal-4
/// cooperative-tensor gemm (`dense_mm.metal`) — the 27B's SwiGLU FFN
/// projections at PREFILL. `plane` is a rank-2 `[out_dim, in_dim]` quantized
/// weight's raw device bytes, `x` is `[t, in_dim]` f32; returns `[t, out_dim]`
/// f32.
///
/// The weight tile is dequantized in-kernel to half; the activation is read
/// straight from device memory as an f32 cooperative tensor with no staging, and
/// the accumulation and output are f32. The matmul2d descriptor sets ggml's
/// reduced-precision flag — the source of the throughput — so this carries the
/// fork's prefill precision class (docs/parity.md §3b) exactly like the
/// attention prefill gemm: ~4.1e-4 rel_l2 from the f32 oracle at the 27B FFN
/// shapes, against the `QMatMul` chain's ~1.9e-4. Bounded and fork-equivalent,
/// but looser than what it replaces; the mm / decode / ppl parity tiers grade
/// the model-level effect.
///
/// No gemv branch: decode stays on `QLinear::forward`. Callers gate on
/// `ops::dense_mm_min_seq()`.
pub fn matmul_dense_q(plane: &QuantPlane, x: &Tensor) -> Result<Tensor> {
    dispatch::run_matmul_dense_q_mm(
        &plane.buffer,
        plane.base_off,
        plane.dtype,
        plane.out_dim,
        plane.in_dim,
        x,
    )
}

/// Whether a `[n_out, k]` quantized weight of this dtype can take the dense
/// prefill gemm at all (kernel instantiated for the dtype, and `k` a whole
/// multiple of both the NK step and the dtype's super-block). The loader uses
/// this to decide whether to hand out a `QuantPlane`.
pub fn dense_mm_supported(dt: GgmlDType, k: usize) -> bool {
    dispatch::dense_mm_supported(dt, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::ops::dispatch::testutil::{max_abs, pseudo_random, rel_l2};
    use candle_core::quantized::{QStorage, QTensor};
    use candle_core::{Device, Module, Tensor};
    use std::sync::Arc;

    /// A `[n_out, k]` weight quantized to `dt`, returned as both the raw-bytes
    /// `QuantPlane` the vendored gemm consumes and the `QTensor` the classic
    /// `QMatMul` chain consumes — sharing ONE device allocation, exactly as
    /// `Weights::qlinear_with_plane` builds them in production. Sharing matters
    /// for the comparison tests: it proves the two paths read identical bytes,
    /// so any difference is the kernel's and not the quantizer's.
    fn build_plane(
        device: &Device,
        dt: GgmlDType,
        n_out: usize,
        k: usize,
        seed: u64,
    ) -> (QuantPlane, Arc<QTensor>) {
        let cpu = Device::Cpu;
        let dense =
            Tensor::from_vec(pseudo_random(n_out * k, seed, -0.5, 0.5), (n_out, k), &cpu).unwrap();
        // Quantize on the CPU, then upload the quantized bytes ONCE and retain
        // the buffer before the storage moves into the QTensor.
        let qcpu = QTensor::quantize(&dense, dt).unwrap();
        let raw = qcpu.data().unwrap();
        let storage = QStorage::from_data(raw, device, dt).unwrap();
        let QStorage::Metal(qms) = &storage else {
            panic!("expected Metal quantized storage")
        };
        let buffer = Arc::new(qms.buffer().clone());
        let qtensor = Arc::new(QTensor::new(storage, (n_out, k)).unwrap());
        (
            QuantPlane {
                buffer,
                base_off: 0,
                dtype: dt,
                out_dim: n_out,
                in_dim: k,
            },
            qtensor,
        )
    }

    /// The classic path the new kernel replaces: candle's `QMatMul` over the
    /// same quantized bytes.
    fn classic(qt: &Arc<QTensor>, x: &Tensor) -> Vec<f32> {
        candle_core::quantized::QMatMul::from_arc(qt.clone())
            .unwrap()
            .forward(x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// The exact reference both kernels approximate: dequantize the weights and
    /// multiply in f32. Neither path's quantization error appears here — the
    /// stored bytes are shared — so the residual is purely what each kernel's
    /// tiling and operand precision cost.
    ///
    /// This also validates the kernel's block/sub-block indexing, which is the
    /// one thing in the port that a numeric test could plausibly miss.
    /// `dense_mm.metal` locates each 16-element dequant call at super-block
    /// `k_pos / (16*nl)`, sub-block `(k_pos / 16) % nl`, on the assumption that a
    /// call returns the 16 CONTIGUOUS elements at `[16*il, 16*il + 16)`.
    /// `QTensor::dequantize` is candle's own independent implementation of the
    /// same GGUF layout, so if that assumption were off by a sub-block the
    /// weights would be permuted within every super-block and the error would be
    /// order 1, not the ~4e-4 these tests bound. Agreement at K = 5120 and 17408
    /// (20 and 68 super-blocks per row) is what pins the indexing.
    fn oracle(qt: &Arc<QTensor>, x: &Tensor) -> Vec<f32> {
        let w = qt
            .dequantize(x.device())
            .unwrap()
            .to_dtype(candle_core::DType::F32)
            .unwrap();
        flat(&x.matmul(&w.t().unwrap()).unwrap())
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// The 27B's production dense-FFN shapes as (n_out, k): gate and up
    /// (hidden 5120 -> ff 17408) and down (ff 17408 -> hidden 5120).
    const FFN_SHAPES: [(usize, usize); 2] = [(17408, 5120), (5120, 17408)];

    /// The tolerance this path is graded at, and the reason it is not tighter.
    ///
    /// This kernel is measurably LOOSER than the `QMatMul` chain it replaces:
    /// against the dequantize-then-f32 oracle it lands ~4.1e-4 rel_l2 at the 27B
    /// FFN shapes where candle's `kernel_mul_mm_q4_K_f32` lands ~1.9e-4. Both
    /// stage the weight tile as half and accumulate in f32; the extra ~2e-4 is
    /// matmul2d's reduced-precision tensor-core path, which is where this
    /// kernel's throughput comes from. That is a deliberate, already-shipped
    /// trade — the attention prefill gemm (`ops::matmul_f16`'s tensor branch)
    /// made exactly it, docs/parity.md §3b names the resulting ~2e-4 band the
    /// fork's prefill precision class, and llama.cpp sets the same descriptor
    /// flag for its own dense FFN prefill. Whether the model-level effect is
    /// acceptable is not this test's call: the mm / decode / ppl parity tiers
    /// decide that against frozen floors.
    ///
    /// 1e-3 is the bound mm_id's f16-operand tiled variants are already held to
    /// against the same kind of oracle
    /// (`mm_id::tests::mm_variants_precision_and_throughput`, `rel_t < 1e-3`).
    /// This path is that kernel family with a wider token tile, so it inherits
    /// the bound rather than inventing one.
    const TOL: f32 = 1e-3;

    /// The oracle-relative error the CLASSIC path shows at the same shapes. The
    /// new kernel is allowed to be looser than this (the reduced-precision
    /// trade above) but not unboundedly so: a regression past this multiple
    /// would mean something other than the descriptor flag changed.
    const CLASSIC_MULTIPLE: f32 = 4.0;

    /// The vendored dense gemm at the 27B's production FFN shapes and a full
    /// prefill chunk, graded against the dequantize-then-f32 oracle — and
    /// alongside candle's `QMatMul` over the SAME quantized bytes, which must
    /// land in the same class. The pair is the load-bearing correctness link:
    /// `QMatMul` is what ships today, so the new path has to be no further from
    /// exact than it is, not merely close to it.
    #[test]
    fn dense_q4k_matches_oracle_production_shapes() {
        let device = metal_device().unwrap();
        const T: usize = 512;

        for (n_out, k) in FFN_SHAPES {
            let (plane, qt) = build_plane(&device, GgmlDType::Q4K, n_out, k, 0x210 + n_out as u64);
            let x = Tensor::from_vec(
                pseudo_random(T * k, 0x310 + n_out as u64, -1.0, 1.0),
                (T, k),
                &device,
            )
            .unwrap();

            let want = oracle(&qt, &x);
            let got = flat(&matmul_dense_q(&plane, &x).unwrap());
            let old = classic(&qt, &x);
            let (rel, rel_classic, rel_pair) =
                (rel_l2(&got, &want), rel_l2(&old, &want), rel_l2(&got, &old));
            eprintln!(
                "dense q4_K [{n_out}x{k}] t={T}: vs oracle {rel:.3e}  \
                 QMatMul vs oracle {rel_classic:.3e}  vs QMatMul {rel_pair:.3e}"
            );
            assert!(
                rel < TOL,
                "dense q4_K vs oracle [{n_out}x{k}] t={T}: rel_l2 {rel} (max_abs {})",
                max_abs(&got, &want)
            );
            // The known, bounded cost of the reduced-precision tensor path (see
            // CLASSIC_MULTIPLE). Measured ~2.2x; anything past 4x means the gap
            // is no longer just the descriptor flag.
            assert!(
                rel <= rel_classic * CLASSIC_MULTIPLE,
                "dense q4_K is {:.1}x further from exact than QMatMul [{n_out}x{k}] \
                 ({rel} vs {rel_classic}) — more than the reduced-precision path accounts for",
                rel / rel_classic
            );
            assert!(rel_pair < TOL, "dense q4_K vs QMatMul rel_l2 {rel_pair}");
        }
    }

    /// Every instantiated dtype at a small shape, against `QMatMul` over the
    /// same bytes. q4_K is production; the others guard the dtype matrix.
    #[test]
    fn dense_all_dtypes_match_qmatmul() {
        let device = metal_device().unwrap();
        const T: usize = 64;
        let (n_out, k) = (128usize, 512usize);

        for dt in [
            GgmlDType::Q8_0,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            let (plane, qt) = build_plane(&device, dt, n_out, k, 0x220);
            let x =
                Tensor::from_vec(pseudo_random(T * k, 0x320, -1.0, 1.0), (T, k), &device).unwrap();
            let got = flat(&matmul_dense_q(&plane, &x).unwrap());
            let want = classic(&qt, &x);
            let rel = rel_l2(&got, &want);
            assert!(
                rel < TOL,
                "{dt:?} rel_l2 {rel} (max_abs {})",
                max_abs(&got, &want)
            );
        }
    }

    /// The tile-edge cases the cooperative-tensor extents must cover, which the
    /// 512-token production sweep does not: token counts that are not multiples
    /// of NR1 = 128 (a ragged tail, a single sub-tile token count, and the exact
    /// tile width), and out-dims that are not multiples of NR0 = 64 (the
    /// zero-padded A rows and the clipped store).
    #[test]
    fn dense_edge_tiles_match_qmatmul() {
        let device = metal_device().unwrap();

        for (n_out, k, t) in [
            (128usize, 512usize, 128usize), // exact token tile
            (128, 512, 129),                // one token past a tile
            (128, 512, 200),                // ragged tail
            (100, 512, 200),                // out-dim not a multiple of 64
            (36, 512, 45),                  // sub-tile in both axes
            (64, 256, 512),                 // one k super-block per row
        ] {
            let seed = 0x230 + n_out as u64 + t as u64;
            let (plane, qt) = build_plane(&device, GgmlDType::Q4K, n_out, k, seed);
            let x = Tensor::from_vec(
                pseudo_random(t * k, seed ^ 0xBEEF, -1.0, 1.0),
                (t, k),
                &device,
            )
            .unwrap();
            let out = matmul_dense_q(&plane, &x).unwrap();
            assert_eq!(out.dims(), &[t, n_out]);
            let got = flat(&out);
            let want = classic(&qt, &x);
            let rel = rel_l2(&got, &want);
            assert!(
                rel < TOL,
                "edge [{n_out}x{k}] t={t}: rel_l2 {rel} (max_abs {})",
                max_abs(&got, &want)
            );
        }
    }

    #[test]
    fn dense_shape_and_dtype_errors() {
        let device = metal_device().unwrap();
        let (plane, _) = build_plane(&device, GgmlDType::Q4K, 64, 256, 0x240);

        // k mismatch.
        let x = Tensor::zeros((4, 128), candle_core::DType::F32, &device).unwrap();
        assert!(matmul_dense_q(&plane, &x).is_err());
        // f16 activation (the kernel reads x as f32).
        let xf = Tensor::zeros((4, 256), candle_core::DType::F16, &device).unwrap();
        assert!(matmul_dense_q(&plane, &xf).is_err());
        // A dtype/k combination outside the support matrix must be refused, not
        // dispatched: k must be a whole multiple of the super-block.
        assert!(!dense_mm_supported(GgmlDType::Q4K, 288));
        assert!(!dense_mm_supported(GgmlDType::Q4_0, 512));
        assert!(dense_mm_supported(GgmlDType::Q4K, 5120));
        assert!(dense_mm_supported(GgmlDType::Q4K, 17408));
    }

    /// Strip `/* ... */` block comments and `//` line comments from Metal source
    /// so a commented-out `[[host_name(...)]]` instantiation cannot satisfy a
    /// `contains` check.
    fn strip_metal_comments(src: &str) -> String {
        let b = src.as_bytes();
        let mut out: Vec<u8> = Vec::with_capacity(b.len());
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
                i += 2;
                while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                    i += 1;
                }
                i = (i + 2).min(b.len());
            } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
                i += 2;
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            } else {
                out.push(b[i]);
                i += 1;
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    const DENSE_MM_SRC: &str = include_str!("dense_mm.metal");

    /// The dtype support matrix (`dense_mm_kernel_name`, which `dense_mm_supported`
    /// and therefore the loader's plane decision gate on) must exactly track the
    /// kernels instantiated in dense_mm.metal: a dtype it claims with no
    /// `[[host_name(...)]]` line would fault the pipeline lookup instead of
    /// falling back to `QMatMul`.
    #[test]
    fn instantiation_matrix_matches_metal() {
        let src = strip_metal_comments(DENSE_MM_SRC);
        for dt in [
            GgmlDType::Q8_0,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::Q4_0,
            GgmlDType::Q2K,
            GgmlDType::F16,
        ] {
            let claimed = dispatch::dense_mm_kernel_name(dt);
            match claimed {
                Ok(name) => assert!(
                    src.contains(&format!("host_name(\"{name}\"")),
                    "dense_mm_kernel_name claims {name:?} for {dt:?}, but no host_name in dense_mm.metal"
                ),
                Err(_) => {
                    // Nothing to check by name for a denied dtype, but the ggml
                    // naming is mechanical, so the would-be name must be absent.
                    let ghost = format!("kernel_mul_mm_{dt:?}_f32_t");
                    assert!(
                        !src.contains(&format!("host_name(\"{ghost}\"")),
                        "dense_mm_kernel_name denies {dt:?}, but dense_mm.metal instantiates {ghost:?}"
                    );
                }
            }
        }
    }

    /// The host sizes the grid and the threadgroup allocation from
    /// `DENSE_MM_NR0` / `DENSE_MM_NR1` / `DENSE_MM_A_TILE_SMEM`; the kernel bakes
    /// the same numbers as `constexpr`. A silent divergence would under-allocate
    /// the A tile or leave output tiles uncomputed, so pin them against the
    /// source.
    #[test]
    fn geometry_matches_metal() {
        let src = strip_metal_comments(DENSE_MM_SRC);
        for (decl, want) in [
            ("constexpr int NR0 = ", dispatch::DENSE_MM_NR0),
            ("constexpr int NR1 = ", dispatch::DENSE_MM_NR1),
            ("constexpr int NK  = ", 32),
            ("constexpr int NK_CHUNK = ", 16),
            ("constexpr int NUM_THREADS  = ", 128),
        ] {
            let rest = src
                .split_once(decl)
                .unwrap_or_else(|| panic!("dense_mm.metal has no `{decl}` declaration"))
                .1;
            let got: usize = rest
                .split(';')
                .next()
                .unwrap()
                .trim()
                .parse()
                .unwrap_or_else(|e| panic!("parsing `{decl}` value: {e}"));
            assert_eq!(got, want, "dense_mm.metal `{decl}` disagrees with the host");
        }
        // The A tile is the ONLY threadgroup allocation: NR0 rows x NK halves.
        assert_eq!(
            dispatch::DENSE_MM_A_TILE_SMEM,
            dispatch::DENSE_MM_NR0 * 32 * 2
        );
    }

    /// Isolation timing: the vendored dense gemm vs candle's `QMatMul` at the
    /// 27B's production FFN shapes, over the token counts that bracket the
    /// mv/mm crossover. The variants are INTERLEAVED inside one timed loop so
    /// both sample the same DVFS/LPM clock trajectory. Prints achieved TFLOP/s
    /// for each. `#[ignore]`d — run on a `pgrep`-verified free GPU with:
    ///   cargo test --release -p xwen dense_mm_vs_qmatmul_timing -- --ignored --nocapture
    #[test]
    #[ignore = "perf bench"]
    fn dense_mm_vs_qmatmul_timing() {
        use std::time::Instant;

        let device = metal_device().unwrap();
        let candle_core::Device::Metal(mdev) = &device else {
            unreachable!("metal_device is Metal")
        };
        let get = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let (warm, iters) = (get("XWEN_BENCH_WARMUP", 5), get("XWEN_BENCH_ITERS", 30));

        for (n_out, k) in FFN_SHAPES {
            let (plane, qt) = build_plane(&device, GgmlDType::Q4K, n_out, k, 0x250 + n_out as u64);
            let qmm = candle_core::quantized::QMatMul::from_arc(qt).unwrap();

            // Dense across the crossover region: the vendored gemm's token tile
            // is NR1 = 128 wide, so it is flat (launch-latency bound) up to
            // there, while candle's is 32 wide and steps at every multiple.
            for t in [8usize, 16, 24, 32, 33, 40, 48, 64, 96, 128, 256, 512] {
                let x = Tensor::from_vec(pseudo_random(t * k, 0x350, -1.0, 1.0), (t, k), &device)
                    .unwrap();

                let mut times = [Vec::with_capacity(iters), Vec::with_capacity(iters)];
                let run = |which: usize| {
                    if which == 0 {
                        matmul_dense_q(&plane, &x).unwrap();
                    } else {
                        qmm.forward(&x).unwrap();
                    }
                    mdev.wait_until_completed().unwrap();
                };
                for _ in 0..warm {
                    run(0);
                    run(1);
                }
                for _ in 0..iters {
                    for which in 0..2 {
                        let t0 = Instant::now();
                        run(which);
                        times[which].push(t0.elapsed().as_secs_f64() * 1e3);
                    }
                }
                let plateau = |v: &Vec<f64>| -> f64 {
                    v[iters / 2..].iter().sum::<f64>() / (iters - iters / 2) as f64
                };
                let (dense, cl) = (plateau(&times[0]), plateau(&times[1]));
                let tf = |ms: f64| 2.0 * t as f64 * k as f64 * n_out as f64 / (ms * 1e9);
                eprintln!(
                    "[{n_out}x{k}] t={t:4}  dense_mm {dense:7.3} ms ({:6.2} TFLOP/s)  \
                     QMatMul {cl:7.3} ms ({:6.2} TFLOP/s)  speedup {:.2}x",
                    tf(dense),
                    tf(cl),
                    cl / dense
                );
            }
        }
    }
}
