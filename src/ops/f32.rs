use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

/// Dense f32-weight x f32-activation mat-vec against the vendored `f32.metal`
/// kernel — the MoE ROUTER projection (`ffn_gate_inp`, the graph's only f32
/// matmul plane). `weight` is a rank-2 `[n_out, k]` contiguous f32 tensor in
/// its GGUF orientation (NOT the `[k, n_out]` transpose candle's `matmul`
/// wants), `x` is `[t, k]` f32; returns `[t, n_out]` f32.
///
/// GEMV ONLY: `t` above 8 is an error, not a fallback. The kernel re-reads the
/// whole weight plane per token row (grid.y = t), which is the right trade at
/// decode and the wrong one at prefill, so the caller keeps larger batches on
/// candle's `Tensor::matmul` — [`matmul_f32_supported`] is the predicate that
/// says which side a call is on.
///
/// Why it exists: candle's mlx `gemv_t` gives a `[1, 2560] x [2560, 512]` f32
/// product EIGHT threadgroups for the whole 5.24 MB plane (the 35B-A3B's
/// 2048 x 256 also lands on 8), each streaming ~655 KB serially. This kernel's
/// grid is `ceil(n_out/2) x t` — 256 threadgroups at Flash-Next's 512 experts.
///
/// Numerics: both paths accumulate f32 products of the same f32 operands, so
/// the difference is accumulation ORDER — bounded (~1e-7 class at these
/// geometries), never bitwise. Top-k routing is discrete, so a near-tie CAN
/// flip an expert; that is why the strict parity tier pins the classic path
/// (`XWEN_ROUTER_MV_CLASSIC`). Metal only.
pub fn matmul_f32(weight: &Tensor, x: &Tensor) -> Result<Tensor> {
    dispatch::run_matmul_f32(weight, x)
}

/// Whether [`matmul_f32`] covers this SHAPE: `t` inside the gemv's token window
/// (1..=8, the same `F16_MM_MIN_SEQ` boundary the f16 family splits on), `k`
/// a multiple of 32 (the float4 K walk has no tail at our shapes) and `n_out`
/// a multiple of 4 (shared with the f16 family).
///
/// Shape only. Dtype, device, contiguity and view alignment are the caller's to
/// check — they are properties of the tensors rather than of the geometry, and
/// `MoeBlock::router_mv` asks them alongside its own, the way
/// `MoeBlock::fused_shexp` does. Both production routers pass: Flash-Next's
/// `[512, 2560]` and the 35B-A3B's `[256, 2048]`.
pub fn matmul_f32_supported(t: usize, k: usize, n_out: usize) -> bool {
    dispatch::matmul_f32_shape_supported(t, k, n_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::ops::dispatch::testutil::{pseudo_random, rel_l2};
    use candle_core::{DType, Device, Tensor};

    /// The two production router planes as (n_out, k) = (n_expert, hidden):
    /// Qwen3.8-Flash-Next and Qwen3.6-35B-A3B.
    const ROUTER_SHAPES: [(usize, usize); 2] = [(512, 2560), (256, 2048)];

    /// Sequential f32 dot per (row, token) — the host oracle. Deliberately the
    /// naive order: the kernel's residual against it IS the reassociation this
    /// change introduces, and a "smarter" host summation would hide it.
    fn host_matmul(w: &[f32], x: &[f32], t: usize, k: usize, n_out: usize) -> Vec<f32> {
        let mut out = vec![0f32; t * n_out];
        for r in 0..t {
            for c in 0..n_out {
                let mut acc = 0f32;
                for i in 0..k {
                    acc += w[c * k + i] * x[r * k + i];
                }
                out[r * n_out + c] = acc;
            }
        }
        out
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// Both production router geometries at every token count the gemv covers
    /// the router at, graded twice: against a sequential host f32 dot, and
    /// against the path this replaces (candle's `matmul` over the `[k, n_out]`
    /// transpose). Both operands are f32 in every arm and no arm rounds
    /// anything, so all three differ only in summation order.
    #[test]
    fn matmul_f32_matches_host_reference() {
        let device = metal_device().unwrap();
        let cpu = Device::Cpu;

        for (n_out, k) in ROUTER_SHAPES {
            for t in [1usize, 3, 8] {
                assert!(
                    matmul_f32_supported(t, k, n_out),
                    "[{n_out}x{k}] t={t} must be inside the gemv's window"
                );
                let seed = 0x30F + n_out as u64 + t as u64;
                let wv = pseudo_random(n_out * k, seed, -0.5, 0.5);
                let xv = pseudo_random(t * k, seed ^ 0xF00D, -1.0, 1.0);

                let w = Tensor::from_vec(wv.clone(), (n_out, k), &cpu).unwrap();
                let x = Tensor::from_vec(xv.clone(), (t, k), &cpu).unwrap();

                let got = matmul_f32(
                    &w.to_device(&device).unwrap(),
                    &x.to_device(&device).unwrap(),
                )
                .unwrap();
                assert_eq!(got.dims(), &[t, n_out]);
                assert_eq!(got.dtype(), DType::F32);
                let got = flat(&got);

                let host = host_matmul(&wv, &xv, t, k, n_out);
                let rel_host = rel_l2(&got, &host);
                assert!(
                    rel_host < 1e-5,
                    "host oracle [{n_out}x{k}] t={t} rel_l2 {rel_host}"
                );

                // The route this replaces: x [t, k] @ router_t [k, n_out].
                let candle = flat(&x.matmul(&w.t().unwrap().contiguous().unwrap()).unwrap());
                let rel_candle = rel_l2(&got, &candle);
                assert!(
                    rel_candle < 1e-5,
                    "candle path [{n_out}x{k}] t={t} rel_l2 {rel_candle}"
                );
                eprintln!(
                    "[{n_out}x{k}] t={t}: rel_l2 vs host {rel_host:.3e}, vs candle {rel_candle:.3e}"
                );
            }
        }
    }

    /// Neither operand has to start at its buffer's origin: the launcher binds
    /// each at its layout's byte offset. A weight view is what a sliced router
    /// plane looks like, and an activation view is what a narrowed residual row
    /// looks like. Both offset arms must equal the same product computed from
    /// freshly materialized (offset-zero) copies, BIT for bit — the kernel reads
    /// identical bytes in identical order, so anything less is a binding bug.
    #[test]
    fn matmul_f32_honours_offset_views() {
        let device = metal_device().unwrap();
        let (n_out, k, t) = (256usize, 2048usize, 3usize);

        // A [n_out+4, k] plane whose rows 4.. are the weight; narrowing on dim 0
        // keeps it contiguous and moves the start offset to 4*k floats.
        let wide = Tensor::from_vec(
            pseudo_random((n_out + 4) * k, 0x4A1, -0.5, 0.5),
            (n_out + 4, k),
            &device,
        )
        .unwrap();
        let w_view = wide.narrow(0, 4, n_out).unwrap();
        assert!(w_view.is_contiguous());

        let tall = Tensor::from_vec(
            pseudo_random((t + 2) * k, 0x4A2, -1.0, 1.0),
            (t + 2, k),
            &device,
        )
        .unwrap();
        let x_view = tall.narrow(0, 1, t).unwrap();
        assert!(x_view.is_contiguous());

        // The oracle re-uploads the same values as fresh offset-ZERO tensors.
        // `.contiguous()` would not do: candle hands back the same view when it
        // already is contiguous, so the comparison would be against itself.
        let materialize = |v: &Tensor| {
            let dims = v.dims2().unwrap();
            Tensor::from_vec(flat(v), dims, &device).unwrap()
        };
        let got = flat(&matmul_f32(&w_view, &x_view).unwrap());
        let want = flat(&matmul_f32(&materialize(&w_view), &materialize(&x_view)).unwrap());
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "an offset weight/activation view must read the same bytes as a materialized copy"
        );

        // A NON-contiguous view is refused rather than silently read as dense.
        let strided = tall.narrow(1, 0, k - 32).unwrap();
        assert!(!strided.is_contiguous());
        let w_small = wide
            .narrow(0, 4, n_out)
            .unwrap()
            .narrow(1, 0, k - 32)
            .unwrap();
        assert!(matmul_f32(&w_small, &strided).is_err());
    }

    /// Every shape rule is an error, not a fallback: the caller decides which
    /// side of them a call belongs on, and a silent reroute would hide a
    /// mis-shaped plane. `matmul_f32_supported` must agree with the launcher on
    /// each.
    #[test]
    fn matmul_f32_refuses_unsupported_shapes() {
        let device = metal_device().unwrap();

        // t = 9: one past the gemv window (larger batches belong on candle).
        assert!(!matmul_f32_supported(9, 2048, 256));
        let w = Tensor::zeros((256, 2048), DType::F32, &device).unwrap();
        let x = Tensor::zeros((9, 2048), DType::F32, &device).unwrap();
        assert!(matmul_f32(&w, &x).is_err());
        // t = 8 is the last accepted one.
        assert!(matmul_f32_supported(8, 2048, 256));

        // k = 2561: the float4 K walk has no tail.
        assert!(!matmul_f32_supported(1, 2561, 256));
        let w = Tensor::zeros((256, 2561), DType::F32, &device).unwrap();
        let x = Tensor::zeros((1, 2561), DType::F32, &device).unwrap();
        assert!(matmul_f32(&w, &x).is_err());

        // n_out = 510: not a multiple of 4.
        assert!(!matmul_f32_supported(1, 2048, 510));
        let w = Tensor::zeros((510, 2048), DType::F32, &device).unwrap();
        let x = Tensor::zeros((1, 2048), DType::F32, &device).unwrap();
        assert!(matmul_f32(&w, &x).is_err());

        // Wrong weight dtype (the shape predicate cannot see this — the caller
        // checks dtype itself, and the launcher is the backstop).
        let wf16 = Tensor::zeros((256, 2048), DType::F16, &device).unwrap();
        let x = Tensor::zeros((1, 2048), DType::F32, &device).unwrap();
        assert!(matmul_f32(&wf16, &x).is_err());
        // Wrong activation dtype.
        let w = Tensor::zeros((256, 2048), DType::F32, &device).unwrap();
        let xf16 = Tensor::zeros((1, 2048), DType::F16, &device).unwrap();
        assert!(matmul_f32(&w, &xf16).is_err());

        // k mismatch between the two operands.
        let x = Tensor::zeros((1, 1024), DType::F32, &device).unwrap();
        assert!(matmul_f32(&w, &x).is_err());
    }
}
