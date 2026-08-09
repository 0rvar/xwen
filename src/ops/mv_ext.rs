use anyhow::Result;
use candle_core::Tensor;
use candle_core::quantized::GgmlDType;

use crate::gguf::QuantPlane;
use crate::ops::dispatch;

/// Quantized-weight x f32-activation matmul against the vendored small-batch
/// mat-vec (`mv_ext.metal`) — the window between the plain gemv and a tiled
/// gemm. `plane` is a rank-2 `[out_dim, in_dim]` quantized weight's raw device
/// bytes, `x` is `[t, in_dim]` f32; `r1ptg` is the plan
/// `ops::mv_ext_window(t)` returned, and passing anything else is the caller's
/// bug (the dispatcher rejects a width with no instantiated kernel).
///
/// A thread dequantizes a weight chunk into registers once and dots it against
/// `r1ptg` token rows before advancing, so the weight matrix crosses memory
/// once per call instead of once per token (the gemv) — and it does so at the
/// gemv's occupancy rather than a gemm's single tile column.
///
/// Every operand stays f32; what changes is the K reduction's summation order,
/// so this is bounded-close to both paths it displaces rather than bit-identical
/// to either. Callers gate on `ops::mv_ext_window`, which folds in the
/// `XWEN_MV_EXT_CLASSIC` kill-switch.
pub fn matmul_mv_ext(plane: &QuantPlane, x: &Tensor, r1ptg: usize) -> Result<Tensor> {
    dispatch::run_matmul_mv_ext(
        &plane.buffer,
        plane.base_off,
        plane.dtype,
        plane.out_dim,
        plane.in_dim,
        x,
        r1ptg,
    )
}

/// Whether a `[n_out, k]` quantized weight of this dtype can take the
/// small-batch mat-vec at all (kernel instantiated for the dtype, and `k` a
/// whole multiple of both the 128-element pass width and the dtype's
/// super-block). Routing sites ask this before they ask about the token count.
pub fn mv_ext_supported(dt: GgmlDType, k: usize) -> bool {
    dispatch::mv_ext_supported(dt, k)
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
    /// `QuantPlane` the vendored kernel consumes and the `QTensor` the classic
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

    /// One of the two paths this kernel displaces: candle's `QMatMul` over the
    /// same quantized bytes (its `mul_mm` branch at these token counts).
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

    /// The exact reference both paths approximate: dequantize the weights and
    /// multiply in f32. Neither path's quantization error appears here — the
    /// stored bytes are shared — so the residual is purely each kernel's
    /// reduction order and operand precision.
    ///
    /// This also validates the port's chunk indexing, which is the one thing a
    /// numeric test could plausibly miss. The ext kernel walks a row as chunks
    /// of 16 (K-quants) or 4 (q8_0) elements, striding `nxpsg` chunks per thread
    /// and wrapping super-blocks by hand; `QTensor::dequantize` is candle's own
    /// independent implementation of the same GGUF layout, so a chunk index off
    /// by one would permute the weights within every super-block and the error
    /// would be order 1, not the ~1e-4 these tests bound.
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

    /// The width production would use for `t`, minus the env ceiling and the
    /// kill-switch — the same function `ops::mv_ext_window` composes, so these
    /// tests exercise the shipped plan rather than a copy of it.
    fn plan(t: usize) -> usize {
        crate::ops::mv_ext_plan(t)
    }

    /// Production shapes as (n_out, k), downscaled on the out dimension only —
    /// `k` is what the kernel's chunk walk depends on, so the real reduction
    /// lengths are kept exactly: 5120 (27B hidden, and the lm_head's k), 17408
    /// (27B ff), 2048 (35B hidden), 512 (35B expert width).
    const SHAPES: [(usize, usize); 4] = [(1024, 5120), (1024, 17408), (768, 2048), (768, 512)];

    /// The tolerance this path is graded at.
    ///
    /// Unlike `dense_mm`, nothing here is narrowed: weights dequantize to f32
    /// registers, the activation is read f32, the accumulation is f32. The only
    /// difference from the paths it displaces is the ORDER of the K reduction —
    /// eight partial sums per row, each over a different interleave of chunks,
    /// combined by a shuffle ladder — which is float-reassociation noise.
    ///
    /// 1e-3 is the bound mm_id's tiled variants are already held to against the
    /// same kind of oracle (`mm_id::tests::mm_variants_precision_and_throughput`,
    /// `rel_t < 1e-3`); this inherits it rather than inventing one. In practice
    /// the kernel lands two orders of magnitude inside it (4e-7 .. 8e-6).
    const TOL: f32 = 1e-3;

    /// This kernel must be at least as close to exact as the path it replaces.
    ///
    /// Not a formality: `QMatMul` at these token counts runs candle's quantized
    /// `mul_mm`, which stages its dequantized weight tile as HALF, and that puts
    /// it ~1.8e-4 from the oracle at every shape here. Keeping f32 end to end
    /// puts this kernel at 4e-7 .. 8e-6 — 20x to 400x closer. So the small-batch
    /// window is an accuracy improvement as well as a bandwidth one, and the
    /// assertion is worth making at 1.0 rather than with a slack multiple: with
    /// two orders of magnitude of measured headroom, anything that reached
    /// parity with `QMatMul` would mean an f16 staging path had crept back in.
    const CLASSIC_MULTIPLE: f32 = 1.0;

    /// The same accuracy-direction assert against the OTHER path this kernel
    /// displaces: the vendored q8_0 gemv (`ops::matmul_q8`) at the attention
    /// projections. That comparison cannot be made at 1.0 the way the `QMatMul`
    /// one is: the gemv narrows nothing either — it multiplies the raw int8
    /// quants by their f32 delta and accumulates in f32, exactly as this kernel
    /// does — so the two sit in the same class and their ratio is pure
    /// reduction-order noise. A 2x band is wide enough that which one rounds
    /// better at a given shape does not matter, and narrow enough to catch a
    /// real precision regression, which would show up as the
    /// ~1e-4 of an f16 staging path (two orders out from the ~1e-6 both sit at).
    /// Measured, the two are within 2% of each other at every shape below.
    const GEMV_MULTIPLE: f32 = 2.0;

    /// Every dtype x every r1ptg width, at production reduction lengths, graded
    /// against the dequantize-then-f32 oracle and alongside `QMatMul` over the
    /// SAME quantized bytes. The pair is the load-bearing correctness link:
    /// `QMatMul` is what ships at these token counts today, so the new path has
    /// to be no further from exact than it is, not merely close to it.
    #[test]
    fn mv_ext_matches_oracle_every_dtype_and_width() {
        let device = metal_device().unwrap();
        // One token count per instantiated width (2 -> r1_2, 3 -> r1_3,
        // 5 -> r1_5, 8 -> r1_4 with a second, fully-populated threadgroup).
        const TOKENS: [usize; 5] = [2, 3, 4, 5, 8];

        for dt in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            for (n_out, k) in SHAPES {
                assert!(
                    mv_ext_supported(dt, k),
                    "{dt:?} k={k} should be a supported mv_ext shape"
                );
                let (plane, qt) = build_plane(&device, dt, n_out, k, 0x510 + k as u64);
                for t in TOKENS {
                    let x = Tensor::from_vec(
                        pseudo_random(t * k, 0x610 + (k + t) as u64, -1.0, 1.0),
                        (t, k),
                        &device,
                    )
                    .unwrap();

                    let want = oracle(&qt, &x);
                    let got = flat(&matmul_mv_ext(&plane, &x, plan(t)).unwrap());
                    let old = classic(&qt, &x);
                    let (rel, rel_classic) = (rel_l2(&got, &want), rel_l2(&old, &want));
                    eprintln!(
                        "mv_ext {dt:?} [{n_out}x{k}] t={t}: vs oracle {rel:.3e}  \
                         QMatMul vs oracle {rel_classic:.3e}"
                    );
                    assert!(
                        rel < TOL,
                        "mv_ext {dt:?} vs oracle [{n_out}x{k}] t={t}: rel_l2 {rel} \
                         (max_abs {})",
                        max_abs(&got, &want)
                    );
                    assert!(
                        rel <= rel_classic * CLASSIC_MULTIPLE,
                        "mv_ext {dt:?} is {:.1}x further from exact than QMatMul \
                         [{n_out}x{k}] t={t} ({rel} vs {rel_classic}) — an f32-throughout \
                         kernel must beat one that stages its weight tile as half",
                        rel / rel_classic
                    );
                }
            }
        }
    }

    /// The q8_0 attention and DeltaNet projections (`Proj::DenseF16Q8`) at their
    /// production shapes, graded against the same f32 oracle as above AND against
    /// `ops::matmul_q8` — the vendored gemv this window displaces at those sites,
    /// which reads the identical q8_0 bytes once per token.
    ///
    /// Shapes are the real `[out_dim, in_dim]` pairs of both checkpoints, named
    /// by the projections that carry them. The widest out dims (27B `attn_q`
    /// 12288x5120 and `attn_qkv` 10240x5120, 35B `attn_q` 8192x2048) are left out
    /// as duplicates of cost, not of coverage: out_dim only scales the grid's row
    /// axis, which the ragged-row test already walks, while `in_dim` is what the
    /// kernel's chunk walk depends on and every distinct one appears here.
    #[test]
    fn mv_ext_q8_matches_the_gemv_it_replaces_at_projection_shapes() {
        let device = metal_device().unwrap();
        // (out_dim, in_dim, the projections shipping that shape)
        const PROJECTIONS: [(usize, usize, &str); 5] = [
            (1024, 5120, "27B attn_k / attn_v"),
            (5120, 6144, "27B attn_output / ssm_out"),
            (512, 2048, "35B attn_k / attn_v"),
            (8192, 2048, "35B attn_qkv"),
            (2048, 4096, "35B attn_output / ssm_out"),
        ];
        // Bottom, middle and top of the shipped window (r1_2, r1_5, and r1_4
        // over two threadgroups).
        const TOKENS: [usize; 3] = [2, 5, 8];

        for (n_out, k, what) in PROJECTIONS {
            let (plane, qt) = build_plane(&device, GgmlDType::Q8_0, n_out, k, 0xF10 + n_out as u64);
            for t in TOKENS {
                let x = Tensor::from_vec(
                    pseudo_random(t * k, 0x1010 + (n_out + t) as u64, -1.0, 1.0),
                    (t, k),
                    &device,
                )
                .unwrap();

                let want = oracle(&qt, &x);
                let got = flat(&matmul_mv_ext(&plane, &x, plan(t)).unwrap());
                let gemv = flat(
                    &crate::ops::matmul_q8(&plane.buffer, plane.base_off, n_out, k, &x).unwrap(),
                );
                let (rel, rel_gemv) = (rel_l2(&got, &want), rel_l2(&gemv, &want));
                eprintln!(
                    "mv_ext q8_0 [{n_out}x{k}] ({what}) t={t}: vs oracle {rel:.3e}  \
                     gemv vs oracle {rel_gemv:.3e}"
                );
                assert!(
                    rel < TOL,
                    "mv_ext q8_0 vs oracle [{n_out}x{k}] t={t}: rel_l2 {rel} (max_abs {})",
                    max_abs(&got, &want)
                );
                assert!(
                    rel <= rel_gemv * GEMV_MULTIPLE,
                    "mv_ext q8_0 is {:.1}x further from exact than the gemv it \
                     replaces [{n_out}x{k}] t={t} ({rel} vs {rel_gemv})",
                    rel / rel_gemv
                );
            }
        }
    }

    /// A row count that is not a whole multiple of the 8 rows a threadgroup
    /// covers, and a token count that is not a whole multiple of r1ptg, so the
    /// kernel's two ragged-tail guards both fire. The masked lanes must neither
    /// write out of bounds nor perturb the rows that do get written.
    ///
    /// Across all three dtypes, because the guards live in two structurally
    /// different templates: the K-quants run the float4x4 impl (chpb 16, one
    /// chunk per thread per pass) and q8_0 the float4 one (chpb 8, four), so
    /// covering only a K-quant would leave half the tail logic untested.
    #[test]
    fn mv_ext_handles_ragged_row_and_token_tails() {
        let device = metal_device().unwrap();
        let k = 512;
        for dt in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            // 1027 = 128*8 + 3 rows: the last threadgroup has 3 live rows of 8.
            // 8 is exactly one threadgroup; 5 is a single partly-masked one.
            for n_out in [1027usize, 8, 5] {
                let (plane, qt) = build_plane(&device, dt, n_out, k, 0x710 + n_out as u64);
                // t = 3 runs r1_3 exactly; t = 7 runs r1_4 with a second
                // threadgroup holding 3 live rows of 4.
                for t in [3usize, 7] {
                    let x = Tensor::from_vec(
                        pseudo_random(t * k, 0x810 + (n_out + t) as u64, -1.0, 1.0),
                        (t, k),
                        &device,
                    )
                    .unwrap();
                    let got = flat(&matmul_mv_ext(&plane, &x, plan(t)).unwrap());
                    let want = oracle(&qt, &x);
                    assert_eq!(got.len(), t * n_out);
                    let rel = rel_l2(&got, &want);
                    assert!(
                        rel < TOL,
                        "ragged mv_ext {dt:?} [{n_out}x{k}] t={t}: rel_l2 {rel} (max_abs {})",
                        max_abs(&got, &want)
                    );
                }
            }
        }
    }

    /// Above ggml's window the plan is xwen's own (r1ptg 4, `ops::mv_ext_window`
    /// under a raised `XWEN_MV_EXT_MAX_SEQ`), and the extension has to actually
    /// compute: the grid takes `ceil(t/4)` threadgroups and the last one masks
    /// its unused token rows. Exercised by passing the width directly rather
    /// than through the env knob, so the test covers the mechanism the knob
    /// selects without depending on process-wide cached state.
    #[test]
    fn mv_ext_matches_oracle_beyond_the_default_window() {
        let device = metal_device().unwrap();
        let k = 512;
        const EXTENDED_R1PTG: usize = 4;
        for dt in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            let (plane, qt) = build_plane(&device, dt, 256, k, 0xB10);
            // 9 and 10 straddle the ceiling; 11 leaves 3 live rows in the last
            // group of 4. 12 is the exact multiple, for contrast.
            for t in [9usize, 10, 11, 12] {
                let x = Tensor::from_vec(
                    pseudo_random(t * k, 0xC10 + t as u64, -1.0, 1.0),
                    (t, k),
                    &device,
                )
                .unwrap();
                let got = flat(&matmul_mv_ext(&plane, &x, EXTENDED_R1PTG).unwrap());
                let want = oracle(&qt, &x);
                assert_eq!(got.len(), t * 256);
                let rel = rel_l2(&got, &want);
                assert!(
                    rel < TOL,
                    "extended mv_ext {dt:?} t={t} r1ptg={EXTENDED_R1PTG}: rel_l2 {rel} \
                     (max_abs {})",
                    max_abs(&got, &want)
                );
            }
        }
    }

    /// The plan `mv_ext_window` hands out must be one the dispatcher accepts, at
    /// every token count the window can admit — including the extended range a
    /// raised `XWEN_MV_EXT_MAX_SEQ` opens, where the two used to disagree and a
    /// 9-token projection aborted the forward.
    #[test]
    fn every_admitted_plan_is_dispatchable() {
        use crate::ops::dispatch::mv_ext_kernel_name;
        for t in 2..=64usize {
            // `mv_ext_window` consults the cached env ceiling, so ask the plan
            // rule directly: in ggml's range its table, above it the r1ptg-4
            // extension — exactly what `mv_ext_window` composes.
            let r = plan(t);
            for dt in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
                mv_ext_kernel_name(dt, r)
                    .unwrap_or_else(|e| panic!("t={t} plans r1ptg {r}, undispatchable: {e}"));
            }
        }
    }

    /// The ACTIVATION reaches the kernel as `start_offset * 4` bytes into a
    /// shared buffer, like every other vendored op. A dropped offset would read
    /// the buffer head, so run a mid-buffer view against the same rows in their
    /// own fresh allocation and require bitwise agreement.
    ///
    /// The flat arm is rebuilt from host data rather than `x.contiguous()`:
    /// a dim-0 narrow is already contiguous, so `contiguous()` returns a clone
    /// carrying the SAME nonzero offset and the two arms would be the identical
    /// configuration — the comparison would hold no matter what the kernel did
    /// with the offset.
    #[test]
    fn mv_ext_handles_offset_activation_views() {
        let device = metal_device().unwrap();
        let (n_out, k, t) = (256usize, 512usize, 4usize);
        let (plane, qt) = build_plane(&device, GgmlDType::Q8_0, n_out, k, 0x910);

        let rows = pseudo_random((t + 3) * k, 0xA10, -1.0, 1.0);
        let x_big = Tensor::from_vec(rows.clone(), (t + 3, k), &device).unwrap();
        let x = x_big.narrow(0, 3, t).unwrap();
        assert!(x.is_contiguous(), "narrowed x view must stay contiguous");
        assert_ne!(
            x.layout().start_offset(),
            0,
            "the view must actually be offset or this test proves nothing"
        );

        let got = flat(&matmul_mv_ext(&plane, &x, plan(t)).unwrap());
        let want = oracle(&qt, &x);
        let rel = rel_l2(&got, &want);
        assert!(
            rel < TOL,
            "offset-view mv_ext: rel_l2 {rel} (max_abs {}) — a dropped start_offset \
             reads the buffer head",
            max_abs(&got, &want)
        );

        // The same rows uploaded as their own allocation: offset 0, identical
        // bytes. Anything but bitwise agreement means the offset moved the math.
        let flatx = Tensor::from_vec(rows[3 * k..].to_vec(), (t, k), &device).unwrap();
        assert_eq!(flatx.layout().start_offset(), 0);
        let direct = flat(&matmul_mv_ext(&plane, &flatx, plan(t)).unwrap());
        assert_eq!(got, direct, "offset view and fresh copy must agree exactly");
    }

    /// The WEIGHT reaches the kernel at `QuantPlane::base_off` bytes into its
    /// buffer. Every production plane is built at offset 0 today, so nothing
    /// else would notice the dispatcher dropping it — construct a plane that
    /// genuinely points mid-buffer by quantizing two stacked weights into one
    /// allocation and aiming at the second.
    #[test]
    fn mv_ext_honors_a_nonzero_weight_base_off() {
        let device = metal_device().unwrap();
        let (n_out, k, t) = (64usize, 512usize, 4usize);

        for dt in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            // One allocation holding [2*n_out, k]; the plane describes its
            // second half.
            let (stacked_plane, stacked_qt) = build_plane(&device, dt, 2 * n_out, k, 0xD10);
            let bytes_per_row = k / dt.block_size() * dt.type_size();
            let base_off = bytes_per_row * n_out;
            assert_ne!(base_off, 0);
            let plane = QuantPlane {
                buffer: stacked_plane.buffer.clone(),
                base_off,
                dtype: dt,
                out_dim: n_out,
                in_dim: k,
            };

            let x = Tensor::from_vec(
                pseudo_random(t * k, 0xE10 + k as u64, -1.0, 1.0),
                (t, k),
                &device,
            )
            .unwrap();

            // Oracle over the SAME half the plane points at.
            let w = stacked_qt
                .dequantize(&device)
                .unwrap()
                .to_dtype(candle_core::DType::F32)
                .unwrap()
                .narrow(0, n_out, n_out)
                .unwrap();
            let want = flat(&x.matmul(&w.t().unwrap()).unwrap());
            let got = flat(&matmul_mv_ext(&plane, &x, plan(t)).unwrap());
            let rel = rel_l2(&got, &want);
            assert!(
                rel < TOL,
                "offset-weight mv_ext {dt:?}: rel_l2 {rel} (max_abs {}) — a dropped \
                 base_off reads the first half of the stack",
                max_abs(&got, &want)
            );
        }
    }

    /// `dispatch::mv_ext_r1ptg` is ggml's table verbatim
    /// (ggml-metal-ops.cpp:2176-2190) and reports `None` outside ggml's own
    /// 2..=8 window — that is the vendored fact, NOT xwen's routing decision
    /// (`ops::mv_ext_window` composes this with the ceiling and the extension;
    /// `every_admitted_plan_is_dispatchable` covers the composed rule).
    #[test]
    fn r1ptg_plan_covers_the_window_and_nothing_else() {
        use crate::ops::dispatch::{mv_ext_kernel_name, mv_ext_r1ptg};
        assert_eq!(mv_ext_r1ptg(1), None, "one token is the plain gemv's job");
        assert_eq!(mv_ext_r1ptg(9), None, "past the window the gemm takes over");
        assert_eq!(mv_ext_r1ptg(0), None);
        let expected = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 3), (7, 4), (8, 4)];
        for (t, r) in expected {
            assert_eq!(mv_ext_r1ptg(t), Some(r), "r1ptg plan for t={t}");
            for dt in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
                mv_ext_kernel_name(dt, r)
                    .unwrap_or_else(|e| panic!("no kernel for {dt:?} r1ptg {r}: {e}"));
            }
        }
    }

    /// `mv_ext_supported` must refuse any K the kernel cannot walk: it has no
    /// tail path, so K must be a whole multiple of both the 128-element pass
    /// width and the dtype's super-block.
    #[test]
    fn support_predicate_tracks_the_kernels_k_requirement() {
        // 384 is a multiple of 128 but not of the 256-element K-quant block.
        assert!(!mv_ext_supported(GgmlDType::Q4K, 384));
        assert!(!mv_ext_supported(GgmlDType::Q6K, 384));
        // ... while q8_0's 32-element block divides it.
        assert!(mv_ext_supported(GgmlDType::Q8_0, 384));
        // Not a multiple of the pass width at all.
        assert!(!mv_ext_supported(GgmlDType::Q8_0, 320));
        // Dtypes with no instantiation, at a shape that would otherwise pass.
        assert!(!mv_ext_supported(GgmlDType::Q5K, 512));
        assert!(!mv_ext_supported(GgmlDType::F16, 512));
        // Production shapes.
        for k in [512usize, 2048, 4096, 5120, 6144, 17408] {
            assert!(mv_ext_supported(GgmlDType::Q4K, k), "q4_K k={k}");
            assert!(mv_ext_supported(GgmlDType::Q8_0, k), "q8_0 k={k}");
        }
    }

    /// The host sizes the grid from constants the kernel bakes in; a change to
    /// either side that is not mirrored would silently drop or double-cover
    /// rows. Cross-check the Rust constants against the .metal source text.
    #[test]
    fn geometry_matches_metal() {
        use crate::ops::dispatch::{MV_EXT_NSG, MV_EXT_NXPSG, MV_EXT_NYPSG, MV_EXT_R0PTG};
        let src = include_str!("mv_ext.metal");
        assert!(
            src.contains(&format!("#define MV_EXT_NSG   {MV_EXT_NSG}")),
            "mv_ext.metal must bake NSG {MV_EXT_NSG}"
        );
        assert!(
            src.contains(&format!("#define MV_EXT_NXPSG {MV_EXT_NXPSG}")),
            "mv_ext.metal must bake NXPSG {MV_EXT_NXPSG}"
        );
        assert_eq!(MV_EXT_NYPSG, 32 / MV_EXT_NXPSG);
        assert_eq!(MV_EXT_R0PTG, MV_EXT_NYPSG * MV_EXT_NSG);
        // The baked shuffle ladder reduces exactly NXPSG lanes.
        for half in [4, 2, 1] {
            assert!(
                src.contains(&format!("simd_shuffle_down(sumf[ir1], {half})")),
                "the nxpsg = {MV_EXT_NXPSG} reduce needs a shuffle by {half}"
            );
        }
    }

    /// Every kernel name the host can ask for must exist in the source, and the
    /// source must not carry entry points the host cannot reach.
    #[test]
    fn instantiation_matrix_matches_metal() {
        use crate::ops::dispatch::mv_ext_kernel_name;
        let src = include_str!("mv_ext.metal");
        let mut expected = Vec::new();
        for dt in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            for r in 2..=5usize {
                let name = mv_ext_kernel_name(dt, r).unwrap();
                assert!(
                    src.contains(name),
                    "mv_ext.metal is missing entry point {name}"
                );
                expected.push(name.to_string());
            }
        }
        let found = src
            .match_indices("kernel_mul_mv_ext_")
            .filter_map(|(i, _)| {
                let rest = &src[i..];
                let end = rest
                    .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                    .unwrap_or(rest.len());
                let name = &rest[..end];
                // The macro invocations name each entry point once as an
                // argument; the impl templates are named differently.
                name.contains("_f32_r1_").then(|| name.to_string())
            })
            .collect::<std::collections::BTreeSet<_>>();
        let expected: std::collections::BTreeSet<_> = expected.into_iter().collect();
        assert_eq!(
            found, expected,
            "mv_ext.metal entry points and mv_ext_kernel_name disagree"
        );
    }
}
