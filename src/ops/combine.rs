use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

/// Fused MoE weighted combine against the vendored `combine.metal` kernels — the
/// routed-expert combine tail of `FusedExperts::forward`. Reads `down`
/// (`[seq, top_k, n_out]` f32 contiguous) once and returns `[seq, n_out]` f32:
///   - `col_l2 == None`  (rescale-free): `dst[s,c] = Σ_k down[s,k,c] * w[s,k]`
///   - `col_l2 == Some`  (`[seq, top_k, 1]` f32): also undoes the per-column L2
///     rescale — `dst[s,c] = Σ_k down[s,k,c] * col_l2[s,k] * 2^-15 * w[s,k]`.
/// `weights` is `[seq, top_k]` f32. Bit-identical to the candle broadcast/affine/
/// sum chain it replaces (combine.rs `fused_matches_candle_bitwise` proves it),
/// so the fused path is safe under every parity tier. Metal only; the caller's
/// kill-switch is the candle chain (`XWEN_COMBINE_CLASSIC`).
pub fn combine(down: &Tensor, col_l2: Option<&Tensor>, weights: &Tensor) -> Result<Tensor> {
    dispatch::run_combine(down, col_l2, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use candle_core::{DType, Device};

    /// F16's safe headroom — the exact `f16_safe` constant `FusedExperts::forward`
    /// scales the down projection by; `1.0 / f16_safe` is `2^-15`, exact.
    const F16_SAFE: f64 = 32768.0;

    /// A deterministic f32 with a wide magnitude span (`10^-6 .. 10^4`) and a
    /// random sign — the stress the bitwise test needs: catastrophic cancellation
    /// in the reduction would surface any accumulation-order difference between the
    /// fused kernel and the candle chain.
    fn wide(seed: u64, n: usize, signed: bool) -> Vec<f32> {
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
                if signed && next() < 0.5 { -mag } else { mag }
            })
            .collect()
    }

    /// The exact candle chain `FusedExperts::forward` runs today, for the given
    /// branch — the ground truth the fused kernel must reproduce bit-for-bit.
    fn candle_chain(
        down: &Tensor,
        col_l2: Option<&Tensor>,
        weights: &Tensor,
        seq: usize,
        top_k: usize,
    ) -> Tensor {
        let w = weights.reshape((seq, top_k, 1)).unwrap();
        match col_l2 {
            Some(l2) => {
                let d = (down.broadcast_mul(l2).unwrap() * (1.0 / F16_SAFE)).unwrap();
                d.broadcast_mul(&w).unwrap().sum(1).unwrap()
            }
            None => down.broadcast_mul(&w).unwrap().sum(1).unwrap(),
        }
    }

    /// The fused combine kernel must reproduce the live candle broadcast/affine/sum
    /// chain BIT-FOR-BIT (compared on `f32::to_bits`, not a tolerance), for both
    /// branches, with wide-magnitude signed inputs, at the 3.6 MoE shapes (top_k
    /// 8 and 10), Flint's (top_k 6, hidden 2048, up to a 2048-token chunk), every
    /// candle width from 1 to 8, and an output count that leaves the last
    /// threadgroup partly empty. Bit-identity is the whole justification for
    /// shipping the fused kernel on the strict parity tier, so any mismatch is a
    /// hard failure — never loosen this to a tolerance.
    #[test]
    fn fused_matches_candle_bitwise() {
        let device = metal_device().unwrap();
        let cpu = Device::Cpu;

        let mut shapes = Vec::new();
        for &top_k in &[6usize, 8, 10] {
            for &seq in &[1usize, 8, 512] {
                for &n_out in &[1024usize, 2048, 3072] {
                    shapes.push((top_k, seq, n_out));
                }
            }
        }
        shapes.extend([(6, 2048, 2048), (8, 2048, 2048)]);
        for top_k in 1..=5 {
            shapes.push((top_k, 3, 1000));
        }
        shapes.push((6, 7, 9));

        // The second scale pushes `down` to the f16 ceiling the rescaled projection
        // is held under and `col_l2` to 1e7, past the norm of a 512-wide row of
        // Flint's largest routed activations (~3e5).
        for (top_k, seq, n_out) in shapes {
            for (down_scale, l2_scale) in [(1.0f32, 1.0f32), (6.5, 1000.0)] {
                let down_v: Vec<f32> = wide(
                    0x100 + seq as u64 * 31 + n_out as u64 + top_k as u64 * 7,
                    seq * top_k * n_out,
                    true,
                )
                .into_iter()
                .map(|x| x * down_scale)
                .collect();
                // col_l2 is a positive per-column norm in production.
                let l2_v: Vec<f32> = wide(0x200 + seq as u64, seq * top_k, false)
                    .into_iter()
                    .map(|x| x * l2_scale)
                    .collect();
                let w_v = wide(0x300 + n_out as u64, seq * top_k, true);

                let down = Tensor::from_vec(down_v, (seq, top_k, n_out), &cpu)
                    .unwrap()
                    .to_device(&device)
                    .unwrap();
                let col_l2 = Tensor::from_vec(l2_v, (seq, top_k, 1), &cpu)
                    .unwrap()
                    .to_device(&device)
                    .unwrap();
                let weights = Tensor::from_vec(w_v, (seq, top_k), &cpu)
                    .unwrap()
                    .to_device(&device)
                    .unwrap();

                for (label, l2) in [("rescale", Some(&col_l2)), ("plain", None)] {
                    let fused = combine(&down, l2, &weights).unwrap();
                    assert_eq!(fused.dims(), &[seq, n_out]);
                    assert_eq!(fused.dtype(), DType::F32);
                    let want = candle_chain(&down, l2, &weights, seq, top_k);

                    let fb: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();
                    let wb: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
                    assert_eq!(fb.len(), wb.len());
                    for (i, (f, w)) in fb.iter().zip(wb.iter()).enumerate() {
                        assert_eq!(
                            f.to_bits(),
                            w.to_bits(),
                            "{label} combine top_k={top_k} seq={seq} n_out={n_out} scale={down_scale}: \
                             element {i} differs \
                             (fused {f:?} bits {:#010x}, candle {w:?} bits {:#010x})",
                            f.to_bits(),
                            w.to_bits(),
                        );
                    }
                }
            }
        }
    }

    /// The combine kernels reduce each output inside one 32-lane simdgroup, so a
    /// top_k whose candle threadgroup width `next_pow2(top_k/2)` exceeds 32 would
    /// silently drop lanes 32.. — `run_combine` must bail (an error, not a
    /// fallback). top_k=66 is the first such width (66/2=33 → next_pow2=64 > 32);
    /// production top_k is 10 (width 8), so this is out of contract.
    #[test]
    fn combine_bails_on_wide_top_k() {
        let device = metal_device().unwrap();
        let top_k = 66usize;
        let down = Tensor::zeros((1, top_k, 4), DType::F32, &device).unwrap();
        let w = Tensor::zeros((1, top_k), DType::F32, &device).unwrap();
        let err = combine(&down, None, &w).unwrap_err().to_string();
        assert!(err.contains("> 32"), "unexpected error: {err}");
    }

    /// Kernel time at Flint's (top_k 6) and the Qwen MoE checkpoints' (top_k 8)
    /// prefill chunk shapes against the candle chain it replaces. Unpinned: run
    /// alone, and quote with the `pmset -g` power mode.
    /// `cargo test --release --lib combine_microbench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn combine_microbench() {
        let dev = metal_device().unwrap();
        let cpu = Device::Cpu;
        let sync = || dev.synchronize().unwrap();
        let time = |f: &dyn Fn() -> Tensor, iters: usize| -> f64 {
            for _ in 0..10 {
                f();
            }
            sync();
            let t = std::time::Instant::now();
            for _ in 0..iters {
                f();
            }
            sync();
            t.elapsed().as_secs_f64() * 1e6 / iters as f64
        };
        for &(top_k, n_out) in &[(6usize, 2048usize), (8, 2048)] {
            for &seq in &[512usize, 2048] {
                let to = |v: Vec<f32>, shape: &[usize]| {
                    Tensor::from_vec(v, shape, &cpu)
                        .unwrap()
                        .to_device(&dev)
                        .unwrap()
                };
                let down = to(wide(1, seq * top_k * n_out, true), &[seq, top_k, n_out]);
                let l2 = to(wide(2, seq * top_k, false), &[seq, top_k, 1]);
                let w = to(wide(3, seq * top_k, true), &[seq, top_k]);
                let fused = time(&|| combine(&down, Some(&l2), &w).unwrap(), 50);
                let chain = time(&|| candle_chain(&down, Some(&l2), &w, seq, top_k), 20);
                println!(
                    "combine_rescale top_k={top_k} n_out={n_out} seq={seq}: \
                     fused {fused:.1} us, candle chain {chain:.1} us"
                );
            }
        }
    }

    #[test]
    fn shape_and_dtype_errors() {
        let device = metal_device().unwrap();
        // weights shape mismatch.
        let down = Tensor::zeros((4, 10, 8), DType::F32, &device).unwrap();
        let bad_w = Tensor::zeros((4, 9), DType::F32, &device).unwrap();
        assert!(combine(&down, None, &bad_w).is_err());
        // col_l2 wrong last dim.
        let w = Tensor::zeros((4, 10), DType::F32, &device).unwrap();
        let bad_l2 = Tensor::zeros((4, 10, 2), DType::F32, &device).unwrap();
        assert!(combine(&down, Some(&bad_l2), &w).is_err());
        // down not f32.
        let down_f16 = Tensor::zeros((4, 10, 8), DType::F16, &device).unwrap();
        assert!(combine(&down_f16, None, &w).is_err());
    }
}
