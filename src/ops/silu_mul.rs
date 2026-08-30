use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

/// Fused MoE SwiGLU activation against the vendored `silu_mul.metal` kernel — the
/// silu*mul glue between the up/gate expert matvecs and the down matvec in
/// `FusedExperts::forward`. Reads `gate` and `up` (same-shape f32 contiguous, the
/// `[seq, top_k, expert_ff]` expert-matvec outputs) once and returns their
/// `silu(gate) * up`, same shape and dtype. Bit-identical to the candle
/// `silu(gate) * up` chain it replaces (silu_mul.rs `fused_matches_candle_bitwise`
/// proves it), so the fused path is safe under every parity tier. Metal only; the
/// caller's kill-switch is the candle chain (`XWEN_ACT_CLASSIC`).
pub fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    dispatch::run_silu_mul(gate, up)
}

/// The f16-tile prefill branch's activation glue in one pass, against
/// `kernel_moe_silu_mul_l2` (silu_mul.metal). From the `[seq, top_k, expert_ff]`
/// f32 `gate`/`up` pair returns `(act_s, col_l2)`:
///   `act    = silu(gate) * up`
///   `col_l2 = clamp(sqrt(Σ_ff act²), clamp_min, clamp_max)`   `[seq, top_k, 1]`
///   `act_s  = (act * scale) / col_l2`                          `[seq, top_k, expert_ff]`
/// — exactly what `FusedExperts::project_inner`'s candle chain (`silu_mul`, then
/// sqr → sum_keepdim → sqrt → clamp → affine → broadcast_div, seven dispatches)
/// computes, in one. The elementwise steps keep the chain's per-op rounding; the
/// SUM runs in a fixed sequential-then-tree order that differs from candle's, so
/// the two agree to accumulation-order noise (silu_mul.rs
/// `l2_fold_matches_candle_chain` bounds it) — bounded, NOT bitwise, which is
/// why the strict parity tier never sees this kernel (its candidate runs mv_id,
/// which has no rescale branch) and mm / decode / ppl grade it. Metal only; the
/// caller's kill-switch is the candle chain (`XWEN_ACT_L2_CLASSIC`).
pub fn silu_mul_l2(
    gate: &Tensor,
    up: &Tensor,
    scale: f32,
    clamp_min: f32,
    clamp_max: f32,
) -> Result<(Tensor, Tensor)> {
    dispatch::run_silu_mul_l2(gate, up, scale, clamp_min, clamp_max)
}

/// Whether `silu_mul_l2` can take a `[.., expert_ff]` activation: the kernel
/// holds one row in a fixed threadgroup array, so wider rows stay on the chain.
pub fn silu_mul_l2_supported(expert_ff: usize) -> bool {
    expert_ff >= 1 && expert_ff <= dispatch::SILU_MUL_L2_MAX_FF
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use candle_core::{DType, Device};
    use candle_nn::ops::silu;

    /// A deterministic f32 with a wide magnitude span (`10^-6 .. 10^4`) and a
    /// random sign — silu's `exp(-x)` saturates at both ends, so this exercises
    /// the small-, mid- and large-magnitude regimes where the fused kernel and
    /// candle's silu could round differently if the arithmetic diverged.
    fn wide(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 // [0, 1)
        };
        (0..n)
            .map(|_| {
                let exp = -6.0 + next() * 10.0; // 10^-6 .. 10^4
                let mag = 10f64.powf(exp) as f32;
                if next() < 0.5 { -mag } else { mag }
            })
            .collect()
    }

    /// The exact candle chain `FusedExperts::forward` runs when `XWEN_ACT_CLASSIC`
    /// is set — the ground truth the fused kernel must reproduce bit-for-bit.
    fn candle_chain(gate: &Tensor, up: &Tensor) -> Tensor {
        (silu(gate).unwrap() * up).unwrap()
    }

    /// The fused activation kernel must reproduce the live candle `silu(gate) * up`
    /// chain BIT-FOR-BIT (compared on `f32::to_bits`, not a tolerance), across the
    /// production seq / top_k / expert_ff grid with wide-magnitude signed inputs.
    /// Bit-identity is the whole justification for shipping the fused kernel on the
    /// strict parity tier, so any mismatch is a hard failure — never loosen this to
    /// a tolerance.
    #[test]
    fn fused_matches_candle_bitwise() {
        let device = metal_device().unwrap();
        let cpu = Device::Cpu;
        let top_k = 10usize;
        // Decode (seq 1) and prefill (seq 8, 512) shapes at the production
        // expert_ff (1024) plus a non-multiple width to catch tail-thread bugs.
        for &seq in &[1usize, 8, 512] {
            for &expert_ff in &[1024usize, 1000] {
                let n = seq * top_k * expert_ff;
                let gate_v = wide(0x100 + seq as u64 * 31 + expert_ff as u64, n);
                let up_v = wide(0x900 + seq as u64 * 17 + expert_ff as u64, n);

                let gate = Tensor::from_vec(gate_v, (seq, top_k, expert_ff), &cpu)
                    .unwrap()
                    .to_device(&device)
                    .unwrap();
                let up = Tensor::from_vec(up_v, (seq, top_k, expert_ff), &cpu)
                    .unwrap()
                    .to_device(&device)
                    .unwrap();

                let fused = silu_mul(&gate, &up).unwrap();
                assert_eq!(fused.dims(), &[seq, top_k, expert_ff]);
                assert_eq!(fused.dtype(), DType::F32);
                let want = candle_chain(&gate, &up);

                let fb: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();
                let wb: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
                assert_eq!(fb.len(), wb.len());
                for (i, (f, w)) in fb.iter().zip(wb.iter()).enumerate() {
                    assert_eq!(
                        f.to_bits(),
                        w.to_bits(),
                        "silu_mul seq={seq} expert_ff={expert_ff}: element {i} differs \
                         (fused {f:?} bits {:#010x}, candle {w:?} bits {:#010x})",
                        f.to_bits(),
                        w.to_bits(),
                    );
                }
            }
        }
    }

    #[test]
    fn shape_and_dtype_errors() {
        let device = metal_device().unwrap();
        // Shape mismatch between gate and up.
        let gate = Tensor::zeros((4, 10, 8), DType::F32, &device).unwrap();
        let bad_up = Tensor::zeros((4, 10, 9), DType::F32, &device).unwrap();
        assert!(silu_mul(&gate, &bad_up).is_err());
        // Non-f32 operands.
        let up = Tensor::zeros((4, 10, 8), DType::F32, &device).unwrap();
        let gate_f16 = Tensor::zeros((4, 10, 8), DType::F16, &device).unwrap();
        assert!(silu_mul(&gate_f16, &up).is_err());
        let up_f16 = Tensor::zeros((4, 10, 8), DType::F16, &device).unwrap();
        assert!(silu_mul(&gate, &up_f16).is_err());
    }

    /// The exact candle chain `FusedExperts::project_inner` runs on the rescale
    /// branch when `XWEN_ACT_L2_CLASSIC` is set (after the bitwise-identical
    /// fused activation), returning `(act_s, col_l2)`.
    fn candle_l2_chain(act: &Tensor) -> (Tensor, Tensor) {
        let col_l2 = act
            .sqr()
            .unwrap()
            .sum_keepdim(2)
            .unwrap()
            .sqrt()
            .unwrap()
            .clamp(1e-8_f32, 1e30_f32)
            .unwrap();
        let act_s = (act * 32768.0_f64).unwrap().broadcast_div(&col_l2).unwrap();
        (act_s, col_l2)
    }

    fn max_rel(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let d = (*x as f64 - *y as f64).abs();
                let m = (*y as f64).abs().max(1e-30);
                d / m
            })
            .fold(0.0, f64::max)
    }

    /// The L2-fold kernel reproduces the candle rescale chain to accumulation-
    /// order noise: `col_l2` is a 640..1024-term sum of squares reduced in a
    /// different order than candle's `sum_keepdim`, and `act_s` inherits that
    /// through one divide. Bounded at 1e-5 relative — an ulp-level class
    /// (observed ~1e-7), with two orders of headroom — never bitwise. The
    /// production expert_ff widths (640 Flash-Next, 512 the 35B) plus the
    /// kernel's ceiling (1024) and a ragged width.
    #[test]
    fn l2_fold_matches_candle_chain() {
        let device = metal_device().unwrap();
        let cpu = Device::Cpu;
        let top_k = 10usize;
        let mut worst_act = 0.0f64;
        let mut worst_l2 = 0.0f64;
        for &seq in &[1usize, 8, 512] {
            for &expert_ff in &[512usize, 640, 1024, 1000] {
                let n = seq * top_k * expert_ff;
                let gate_v = wide(0x100 + seq as u64 * 31 + expert_ff as u64, n);
                let up_v = wide(0x900 + seq as u64 * 17 + expert_ff as u64, n);
                let gate = Tensor::from_vec(gate_v, (seq, top_k, expert_ff), &cpu)
                    .unwrap()
                    .to_device(&device)
                    .unwrap();
                let up = Tensor::from_vec(up_v, (seq, top_k, expert_ff), &cpu)
                    .unwrap()
                    .to_device(&device)
                    .unwrap();

                let (act_s, col_l2) = silu_mul_l2(&gate, &up, 32768.0, 1e-8, 1e30).unwrap();
                assert_eq!(act_s.dims(), &[seq, top_k, expert_ff]);
                assert_eq!(col_l2.dims(), &[seq, top_k, 1]);
                let (want_s, want_l2) = candle_l2_chain(&candle_chain(&gate, &up));

                let a: Vec<f32> = act_s.flatten_all().unwrap().to_vec1().unwrap();
                let b: Vec<f32> = want_s.flatten_all().unwrap().to_vec1().unwrap();
                let l: Vec<f32> = col_l2.flatten_all().unwrap().to_vec1().unwrap();
                let m: Vec<f32> = want_l2.flatten_all().unwrap().to_vec1().unwrap();
                let ra = max_rel(&a, &b);
                let rl = max_rel(&l, &m);
                worst_act = worst_act.max(ra);
                worst_l2 = worst_l2.max(rl);
                assert!(
                    ra < 1e-5 && rl < 1e-5,
                    "silu_mul_l2 seq={seq} expert_ff={expert_ff}: max rel act_s {ra:.3e}, col_l2 {rl:.3e}"
                );
            }
        }
        eprintln!("silu_mul_l2 worst max-rel: act_s {worst_act:.3e}, col_l2 {worst_l2:.3e}");
    }

    /// Rows wider than the kernel's threadgroup array are refused, not silently
    /// truncated; the caller (`FusedExperts`) asks `silu_mul_l2_supported` and
    /// keeps the chain there.
    #[test]
    fn l2_fold_refuses_wide_rows() {
        let device = metal_device().unwrap();
        assert!(silu_mul_l2_supported(1024));
        assert!(!silu_mul_l2_supported(1025));
        let gate = Tensor::zeros((2, 4, 1025), DType::F32, &device).unwrap();
        let up = Tensor::zeros((2, 4, 1025), DType::F32, &device).unwrap();
        assert!(silu_mul_l2(&gate, &up, 32768.0, 1e-8, 1e30).is_err());
    }

    /// The kernel sizes its per-thread register array and its reduction tree
    /// from SILU_MUL_L2_{THREADS,MAX_FF}; the host dispatches threadgroups and
    /// bounds rows from its own copies (dispatch.rs). The two sets must agree,
    /// and neither side can see the other at compile time — so pin the .metal
    /// text to the host constants here.
    #[test]
    fn metal_and_host_constants_agree() {
        let src = include_str!("silu_mul.metal");
        for (name, host) in [
            ("SILU_MUL_L2_THREADS", dispatch::SILU_MUL_L2_THREADS),
            ("SILU_MUL_L2_MAX_FF", dispatch::SILU_MUL_L2_MAX_FF),
        ] {
            let define = format!("#define {name} {host}");
            assert!(
                src.contains(&define),
                "silu_mul.metal must contain {define:?} to match dispatch.rs"
            );
        }
    }
}
