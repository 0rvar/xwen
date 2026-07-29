use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

/// Dense bf16-weight x f32-activation matmul against the vendored bf16 twin
/// kernels — the DFlash drafter's mmap-aliased BF16 matmul planes. `weight` is
/// a rank-2 `[n_out, k]` dense BF16 tensor (usually a no-copy view of the GGUF
/// bytes), `x` is `[t, k]` f32; returns `[t, n_out]` f32. Same host split and
/// contract as `matmul_f16`: the classic mat-vec (`bf16.metal`) for t <= 8
/// tokens; above that the gemm — by default the Metal-4 cooperative-tensor
/// kernel (`bf16_t.metal`, bfloat loads staged to the half A tile), or the
/// classic simdgroup kernel under `XWEN_ATTN_MM_CLASSIC`.
///
/// Numerics: bf16 -> f32 widening is exact, so the gemv and classic gemm are
/// bit-identical to their f16 twins over weights representable in both formats
/// (the tests pin this), and keep the FULL bf16 value even in f16's subnormal
/// range. The tensor gemm stages the weight to half — the drafter's load-time
/// scan guards overflow; normal-range values stage exactly, f16-subnormal-range
/// values round/flush as a materialized-f16 weight would (the documented
/// seq-boundary asymmetry, pinned by the subnormal-split test below) — and
/// carries the same reduced-precision ~2e-4 class as `matmul_f16`'s tensor
/// path. Metal only.
pub fn matmul_bf16(weight: &Tensor, x: &Tensor) -> Result<Tensor> {
    dispatch::run_matmul_bf16(weight, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::ops::dispatch::F16MmKernel;
    use crate::ops::dispatch::testutil::{pseudo_random, rel_l2};
    use candle_core::{DType, Device, Tensor};

    /// The two sidecars' production matmul shapes as (n_out, k): q (32 heads ->
    /// 4096), k/v (8 KV heads -> 1024), o_proj, the FFN trio, and the encoder fc
    /// (k = n_aux * n_embd, 5 x 5120 on the 27B and 8 x 2048 on the 35B-A3B).
    const DRAFTER_SHAPES: [(usize, usize); 12] = [
        // 27B sidecar: n_embd 5120, n_ff 17408, 5 taps.
        (4096, 5120),
        (1024, 5120),
        (5120, 4096),
        (17408, 5120),
        (5120, 17408),
        (5120, 25600),
        // 35B-A3B sidecar: n_embd 2048, n_ff 6144, 8 taps.
        (4096, 2048),
        (1024, 2048),
        (2048, 4096),
        (6144, 2048),
        (2048, 6144),
        (2048, 16384),
    ];

    /// Weights whose values are exactly representable in BOTH bf16 and f16:
    /// bf16-rounded (8 mantissa bits ⊂ f16's 10) with tiny magnitudes bumped
    /// away from f16's subnormal floor, where bf16 can represent values f16
    /// cannot. Returns (bf16 weights, the SAME values as f16).
    fn common_weights(n_out: usize, k: usize, seed: u64, device: &Device) -> (Tensor, Tensor) {
        let vals: Vec<f32> = pseudo_random(n_out * k, seed, -0.5, 0.5)
            .into_iter()
            .map(|v| if v.abs() < 1e-3 { 0.123 } else { v })
            .collect();
        let cpu = Device::Cpu;
        let wb = Tensor::from_vec(vals, (n_out, k), &cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let wf = wb
            .to_dtype(DType::F32)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        (wb.to_device(device).unwrap(), wf.to_device(device).unwrap())
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// Kernel output vs a CPU f32 reference matmul over the SAME bf16-rounded
    /// weights, on the classic-library paths (gemv + classic gemm, both float
    /// accumulation with the stored weights as the only rounding — the same
    /// ~1e-6 accumulation-noise class as the f16 twins, bound at their shared
    /// 1e-5).
    #[test]
    fn bf16_classic_matches_reference() {
        let device = metal_device().unwrap();
        let cpu = Device::Cpu;
        // t=1/8 exercise the gemv, t=9/16/512 the classic gemm (16 = a draft
        // block, 512 = an inject prefill chunk).
        for (n_out, k) in DRAFTER_SHAPES {
            for t in [1usize, 8, 9, 16, 512] {
                let seed = 0xB16 + n_out as u64 + t as u64;
                let w =
                    Tensor::from_vec(pseudo_random(n_out * k, seed, -0.5, 0.5), (n_out, k), &cpu)
                        .unwrap()
                        .to_dtype(DType::BF16)
                        .unwrap();
                let x =
                    Tensor::from_vec(pseudo_random(t * k, seed ^ 0xF00D, -1.0, 1.0), (t, k), &cpu)
                        .unwrap();

                let got = dispatch::run_matmul_bf16_variant(
                    &w.to_device(&device).unwrap(),
                    &x.to_device(&device).unwrap(),
                    F16MmKernel::Classic,
                )
                .unwrap();
                assert_eq!(got.dims(), &[t, n_out]);
                assert_eq!(got.dtype(), DType::F32);

                let want = x
                    .matmul(&w.to_dtype(DType::F32).unwrap().t().unwrap())
                    .unwrap();
                let rel = rel_l2(&flat(&got), &flat(&want));
                assert!(rel < 1e-5, "bf16 classic [{n_out}x{k}] t={t} rel_l2 {rel}");
            }
        }
    }

    /// The bf16 kernels are line-for-line twins of the f16 ones and bf16 -> f32
    /// widening is exact, so over weights representable in BOTH formats every
    /// path must be BIT-IDENTICAL to its f16 twin: gemv, classic gemm, and the
    /// tensor gemm (whose bf16 A-tile staging narrows to half — exact on the
    /// common set). This is the correctness anchor for the whole family: the
    /// f16 kernels are pinned to references in f16.rs, and this pins bf16 to
    /// f16.
    #[test]
    fn bf16_bitwise_matches_f16_on_common_values() {
        let device = metal_device().unwrap();
        // k/v (27B), q (35B), o_proj (35B), encoder fc (35B).
        for (n_out, k) in [(1024, 5120), (4096, 2048), (2048, 4096), (2048, 16384)] {
            let (wb, wf) = common_weights(n_out, k, 0xC0FFEE + n_out as u64, &device);
            for (t, kernel) in [
                (4usize, F16MmKernel::Classic), // gemv (kernel arg unused at t<=8)
                (16, F16MmKernel::Classic),     // classic simdgroup gemm
                (16, F16MmKernel::Tensor),      // tensor gemm, sub-tile token edge
                (512, F16MmKernel::Tensor),     // tensor gemm, full tiles
            ] {
                let x = Tensor::from_vec(
                    pseudo_random(t * k, 0xACE ^ (n_out as u64 + t as u64), -1.0, 1.0),
                    (t, k),
                    &device,
                )
                .unwrap();
                let got = flat(&dispatch::run_matmul_bf16_variant(&wb, &x, kernel).unwrap());
                let want = flat(&dispatch::run_matmul_f16_variant(&wf, &x, kernel).unwrap());
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "bf16 vs f16 [{n_out}x{k}] t={t} {kernel:?} differs at element {i}: {g} vs {w}"
                    );
                }
            }
        }
    }

    /// The shipped tensor gemm vs the classic simdgroup gemm over GENERIC bf16
    /// weights (not the common-representable set), drafter shapes and tile-edge
    /// token counts. Same transitive link and 5e-4 reduced-precision bound as
    /// f16's `f16_tensor_matches_classic`: classic is pinned to the f32
    /// reference above, tensor is pinned to classic here.
    #[test]
    fn bf16_tensor_matches_classic() {
        let device = metal_device().unwrap();
        let cpu = Device::Cpu;
        for (n_out, k, t) in [
            (4096, 5120, 16),   // 27B q, sub-tile token edge
            (1024, 5120, 413),  // 27B k/v, token count off every tile boundary
            (4096, 2048, 413),  // 35B q, same
            (2048, 6144, 512),  // 35B ffn_down, full tiles
            (2048, 16384, 512), // 35B encoder fc, full tiles over a wide k
        ] {
            let seed = 0xD00 + n_out as u64 + t as u64;
            let w = Tensor::from_vec(pseudo_random(n_out * k, seed, -0.5, 0.5), (n_out, k), &cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
                .to_device(&device)
                .unwrap();
            let x = Tensor::from_vec(
                pseudo_random(t * k, seed ^ 0xF00D, -1.0, 1.0),
                (t, k),
                &device,
            )
            .unwrap();

            let tensor =
                flat(&dispatch::run_matmul_bf16_variant(&w, &x, F16MmKernel::Tensor).unwrap());
            let classic =
                flat(&dispatch::run_matmul_bf16_variant(&w, &x, F16MmKernel::Classic).unwrap());
            let rel = rel_l2(&tensor, &classic);
            assert!(
                rel < 5e-4,
                "bf16 tensor vs classic [{n_out}x{k}] t={t}: rel_l2 {rel}"
            );
        }
    }

    /// An offset weight VIEW (the mmap alias shape: nonzero layout
    /// start_offset) must produce bit-identical output to the same weights at
    /// offset 0, on both the gemv and the tensor gemm. 96 elements = 192 bytes,
    /// 16-byte aligned, sub-page — exactly what `dense_alias_tensor` narrows.
    #[test]
    fn bf16_offset_view_matches() {
        let device = metal_device().unwrap();
        let (n_out, k) = (1024usize, 5120usize); // the 27B sidecar's k/v projection
        let pad = 96usize;
        let vals = pseudo_random(pad + n_out * k, 0xE0, -0.5, 0.5);
        let flat_all = Tensor::from_vec(vals, pad + n_out * k, &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w_view = flat_all
            .narrow(0, pad, n_out * k)
            .unwrap()
            .reshape((n_out, k))
            .unwrap();
        let w_zero = w_view
            .to_dtype(DType::F32)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap(); // materialized at offset 0, identical values
        for t in [2usize, 64] {
            let x = Tensor::from_vec(
                pseudo_random(t * k, 0xE1 + t as u64, -1.0, 1.0),
                (t, k),
                &device,
            )
            .unwrap();
            let got = flat(&matmul_bf16(&w_view, &x).unwrap());
            let want = flat(&matmul_bf16(&w_zero, &x).unwrap());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "bf16 offset view t={t} differs at element {i}: {g} vs {w}"
                );
            }
        }
    }

    /// The documented seq-boundary asymmetry for weights in f16's SUBNORMAL
    /// range (2.2e-5 of the real drafter — see dflash.rs
    /// `ensure_bf16_fits_f16`): the widening gemv and the float-tile classic
    /// gemm keep the FULL bf16 value, while the tensor gemm's bf16 -> half
    /// staging reproduces the old materialized-f16 path bit-for-bit (whatever
    /// the tensor core does with an f16 subnormal, it must match the f16 twin
    /// fed the f16-rounded weight).
    #[test]
    fn bf16_subnormal_range_weights_follow_the_documented_split() {
        let device = metal_device().unwrap();
        let cpu = Device::Cpu;
        // 129 * 2^-25: exactly representable in bf16 (7-bit mantissa,
        // exponent -18) but NOT on f16's fixed 2^-24 subnormal grid — f16
        // RTNE rounds it to 64 * 2^-24 = 2^-18.
        let w_val = 129.0f32 * (2.0f32).powi(-25);
        let (n_out, k, t) = (4usize, 32usize, 9usize);
        let mut vals = vec![0.0f32; n_out * k];
        vals[0] = w_val;
        let wb_cpu = Tensor::from_vec(vals, (n_out, k), &cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let wf_cpu = wb_cpu
            .to_dtype(DType::F32)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        // Premises: the bf16 round-trip kept the full value, and the f16
        // narrowing genuinely moved it (the asymmetry is real).
        let widened = flat(&wb_cpu.to_dtype(DType::F32).unwrap())[0];
        assert_eq!(widened.to_bits(), w_val.to_bits());
        let f16_widened = flat(&wf_cpu.to_dtype(DType::F32).unwrap())[0];
        assert_ne!(f16_widened.to_bits(), w_val.to_bits());

        let wb = wb_cpu.to_device(&device).unwrap();
        let wf = wf_cpu.to_device(&device).unwrap();
        let x1 = Tensor::ones((1, k), DType::F32, &device).unwrap();
        let xt = Tensor::ones((t, k), DType::F32, &device).unwrap();

        // With a single nonzero weight and all-ones x, out[0] IS the weight —
        // one product, no accumulation ambiguity.
        let mv = flat(&matmul_bf16(&wb, &x1).unwrap());
        assert_eq!(
            mv[0].to_bits(),
            w_val.to_bits(),
            "the gemv must keep the full bf16 value ({} vs {w_val})",
            mv[0]
        );
        let classic =
            flat(&dispatch::run_matmul_bf16_variant(&wb, &xt, F16MmKernel::Classic).unwrap());
        assert_eq!(
            classic[0].to_bits(),
            w_val.to_bits(),
            "the classic gemm's float tiles must keep the full bf16 value ({} vs {w_val})",
            classic[0]
        );

        let bt = flat(&dispatch::run_matmul_bf16_variant(&wb, &xt, F16MmKernel::Tensor).unwrap());
        let ft = flat(&dispatch::run_matmul_f16_variant(&wf, &xt, F16MmKernel::Tensor).unwrap());
        for (i, (b, f)) in bt.iter().zip(&ft).enumerate() {
            assert_eq!(
                b.to_bits(),
                f.to_bits(),
                "tensor gemm subnormal handling diverged from the f16 twin at element {i}: {b} vs {f}"
            );
        }
    }

    #[test]
    fn bf16_shape_and_dtype_errors() {
        let device = metal_device().unwrap();
        // f16 weight must be rejected (that is matmul_f16's job).
        let wf = Tensor::zeros((64, 32), DType::F16, &device).unwrap();
        let x = Tensor::zeros((1, 32), DType::F32, &device).unwrap();
        assert!(matmul_bf16(&wf, &x).is_err());
        // k mismatch.
        let w = Tensor::zeros((64, 32), DType::BF16, &device).unwrap();
        let x64 = Tensor::zeros((1, 64), DType::F32, &device).unwrap();
        assert!(matmul_bf16(&w, &x64).is_err());
        // k not a multiple of 32.
        let w20 = Tensor::zeros((64, 20), DType::BF16, &device).unwrap();
        let x20 = Tensor::zeros((1, 20), DType::F32, &device).unwrap();
        assert!(matmul_bf16(&w20, &x20).is_err());
        // The TensorMixed probe is f16-only.
        let w = Tensor::zeros((64, 32), DType::BF16, &device).unwrap();
        let x9 = Tensor::zeros((9, 32), DType::F32, &device).unwrap();
        assert!(dispatch::run_matmul_bf16_variant(&w, &x9, F16MmKernel::TensorMixed).is_err());
    }
}
