use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

/// Fused MoE routing decision against the vendored `moe_glue.metal` router — the
/// seven candle dispatches that follow the router matmul in `MoeBlock::route`
/// (softmax over all experts, descending arg-sort, narrow, gather, sum, clamp,
/// renormalize) collapsed into one threadgroup per token. `logits` is
/// `[seq, n_expert]` f32 contiguous, the router matmul's output; `sum_floor` is
/// the renormalization denominator floor. Returns
/// `(ids [seq, top_k] u32, weights [seq, top_k] f32)`.
///
/// Bit-identical to the candle chain it replaces (`router_matches_candle_bitwise`
/// proves it), so the fused path is safe under every parity tier — which matters
/// more here than anywhere else in the MoE block: the arg-sort's tie order
/// decides WHICH experts run, so a single differing bit is a whole expert's
/// contribution, not a rounding. Metal only; the caller's kill-switch is the
/// candle chain (`XWEN_MOE_GLUE_CLASSIC`).
pub fn moe_router(logits: &Tensor, top_k: usize, sum_floor: f32) -> Result<(Tensor, Tensor)> {
    dispatch::run_moe_router(logits, top_k, sum_floor)
}

/// Whether `moe_router` covers this routing geometry. The router kernel sizes
/// its threadgroup arrays from compile-time bounds and folds the top-k sum in a
/// single simdgroup, so a wider expert set or selection is out of contract —
/// callers ask first and keep the candle chain for a checkpoint outside them,
/// rather than turning an unsupported shape into a hard failure.
pub fn moe_router_supported(n_expert: usize, top_k: usize) -> bool {
    n_expert > 0
        && top_k > 0
        && top_k <= n_expert
        && top_k <= dispatch::MOE_ROUTER_MAX_TOP_K
        && n_expert.next_power_of_two() <= dispatch::MOE_ROUTER_MAX_EXPERTS
}

/// Fused MoE block epilogue against the vendored `moe_glue.metal` kernel — the
/// routed weighted combine, the shared-expert gate sigmoid, its broadcast
/// multiply and the routed+shared add, in one pass over `down`:
/// `dst[s,c] = Σ_k down[s,k,c] * w[s,k] + shexp[s,c] * sigmoid(gate[s])`.
/// `down` is `[seq, top_k, n_out]` f32 contiguous, `w` `[seq, top_k]` f32,
/// `shexp` `[seq, n_out]` f32 (the shared expert's SwiGLU output, ungated), and
/// `gate` `[seq, 1]` f32 the RAW pre-sigmoid shared-expert gate logit. Returns
/// `[seq, n_out]` f32. Bit-identical to the candle chain it replaces
/// (`epilogue_matches_candle_bitwise` proves it). Metal only; the caller's
/// kill-switch is the candle chain (`XWEN_MOE_GLUE_CLASSIC`).
pub fn moe_epilogue(
    down: &Tensor,
    weights: &Tensor,
    shexp: &Tensor,
    gate: &Tensor,
) -> Result<Tensor> {
    dispatch::run_moe_epilogue(down, weights, shexp, gate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::moe::WEIGHTS_SUM_FLOOR;
    use candle_core::{DType, Device};
    use candle_nn::ops::{sigmoid, softmax_last_dim};

    /// Deterministic pseudo-random f32 in `[lo, hi)`.
    fn uniform(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
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

    /// A deterministic f32 with a wide magnitude span (`10^-6 .. 10^4`) and a
    /// random sign — the combine reduction's stress case, where catastrophic
    /// cancellation would surface any accumulation-order difference.
    fn wide(seed: u64, n: usize, signed: bool) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let exp = -6.0 + next() * 10.0;
                let mag = 10f64.powf(exp) as f32;
                if signed && next() < 0.5 { -mag } else { mag }
            })
            .collect()
    }

    fn on_device(v: Vec<f32>, dims: (usize, usize), dev: &Device) -> Tensor {
        Tensor::from_vec(v, dims, &Device::Cpu)
            .unwrap()
            .to_device(dev)
            .unwrap()
    }

    /// The exact candle chain `MoeBlock::route` runs on the classic path — the
    /// ground truth the fused router must reproduce bit-for-bit. Kept a literal
    /// transcription (not a call into moe.rs) so that a future edit to the
    /// production chain shows up here as a test failure rather than silently
    /// moving the target.
    fn candle_route(logits: &Tensor, top_k: usize) -> (Tensor, Tensor) {
        let probs = softmax_last_dim(&logits.to_dtype(DType::F32).unwrap()).unwrap();
        let order = probs
            .contiguous()
            .unwrap()
            .arg_sort_last_dim(false)
            .unwrap();
        let ids = order.narrow(1, 0, top_k).unwrap().contiguous().unwrap();
        let weights = probs.gather(&ids, 1).unwrap();
        let sum = weights
            .sum_keepdim(1)
            .unwrap()
            .clamp(WEIGHTS_SUM_FLOOR as f32, f32::INFINITY)
            .unwrap();
        (ids, weights.broadcast_div(&sum).unwrap())
    }

    /// The exact candle chain the fused epilogue replaces: the fused combine
    /// (itself bit-identical to candle's broadcast/sum chain), the shared
    /// expert's sigmoid gate and broadcast multiply, then `routed + shared`.
    fn candle_epilogue(down: &Tensor, w: &Tensor, shexp: &Tensor, gate: &Tensor) -> Tensor {
        let (seq, top_k, _) = down.dims3().unwrap();
        let wb = w.reshape((seq, top_k, 1)).unwrap();
        let routed = down.broadcast_mul(&wb).unwrap().sum(1).unwrap();
        let shared = shexp.broadcast_mul(&sigmoid(gate).unwrap()).unwrap();
        (routed + shared).unwrap()
    }

    fn assert_f32_bits_eq(got: &Tensor, want: &Tensor, what: &str) {
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let w: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(g.len(), w.len(), "{what}: length");
        for (i, (a, b)) in g.iter().zip(w.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{what}: element {i} differs (fused {a:?} bits {:#010x}, candle {b:?} bits {:#010x})",
                a.to_bits(),
                b.to_bits(),
            );
        }
    }

    fn assert_u32_eq(got: &Tensor, want: &Tensor, what: &str) {
        let g: Vec<u32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let w: Vec<u32> = want.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(g, w, "{what}: selected expert ids differ");
    }

    /// The fused router must reproduce the live candle routing chain
    /// BIT-FOR-BIT — the same selected expert ids and the same renormalized
    /// weights on `f32::to_bits`, not a tolerance. Bit-identity is the whole
    /// justification for shipping it on the strict parity tier, and the ids in
    /// particular are a discrete decision: never loosen this to a tolerance.
    ///
    /// The grid covers the production geometry (256 experts, top-8), a
    /// non-power-of-two expert count (the bitonic network's padding path), a
    /// small count whose candle softmax width drops below the threadgroup-memory
    /// threshold, multi-token rows, and logit scales from near-uniform to
    /// saturating (where most probabilities underflow to zero and the arg-sort
    /// runs on a sea of exact ties).
    #[test]
    fn router_matches_candle_bitwise() {
        let device = metal_device().unwrap();

        let cases: [(usize, usize, usize, f32); 8] = [
            // (seq, n_expert, top_k, logit scale)
            (1, 256, 8, 1.0),   // production decode
            (1, 256, 8, 0.001), // near-uniform: probabilities crowd together
            (1, 256, 8, 60.0),  // saturating: most probabilities underflow to +0
            (7, 256, 8, 1.0),   // several tokens at once
            (1, 250, 8, 1.0),   // padded bitonic network
            (3, 250, 10, 4.0),
            (1, 8, 3, 1.0),   // softmax width below the shared-memory threshold
            (5, 64, 16, 2.0), // wider selection
        ];

        for (seq, n_expert, top_k, scale) in cases {
            let v = uniform(
                0x51 + seq as u64 * 131 + n_expert as u64 * 7 + top_k as u64,
                seq * n_expert,
                -scale,
                scale,
            );
            let logits = on_device(v, (seq, n_expert), &device);
            let (ids, weights) = moe_router(&logits, top_k, WEIGHTS_SUM_FLOOR as f32).unwrap();
            assert_eq!(ids.dims(), &[seq, top_k]);
            assert_eq!(ids.dtype(), DType::U32);
            assert_eq!(weights.dims(), &[seq, top_k]);
            let (want_ids, want_w) = candle_route(&logits, top_k);
            let what = format!("router seq={seq} n_expert={n_expert} top_k={top_k} scale={scale}");
            assert_u32_eq(&ids, &want_ids, &what);
            assert_f32_bits_eq(&weights, &want_w, &what);
        }
    }

    /// Ties are the arg-sort's only ambiguous input and the one place a wrong
    /// answer costs a whole expert. candle's Metal arg-sort is llama.cpp's
    /// bitonic network, which is deterministic but NOT stable — equal
    /// probabilities do not come out in ascending expert order — so these rows
    /// pin the fused network against candle's on inputs built to tie: all-equal
    /// logits, a handful of equal winners among losers, and duplicated blocks.
    #[test]
    fn router_ties_match_candle_bitwise() {
        let device = metal_device().unwrap();
        let n_expert = 256usize;
        let top_k = 8usize;

        let all_equal = vec![0.0f32; n_expert];

        // Twelve exactly-equal winners, the rest exactly equal losers: the
        // selection has to pick 8 of the 12, and WHICH 8 is decided purely by
        // the network's comparator order.
        let mut plateau = vec![0.0f32; n_expert];
        for i in [3usize, 9, 17, 40, 55, 77, 90, 101, 120, 200, 201, 250] {
            plateau[i] = 1.0;
        }

        // A repeating block: every value has 15 exact duplicates spread across
        // the row, so ties appear at every level of the network.
        let block = uniform(0xBEEF, 16, -2.0, 2.0);
        let repeated: Vec<f32> = (0..n_expert).map(|i| block[i % 16]).collect();

        // Two-level ties with a distinct maximum, so the top slot is decided
        // but the remaining seven are all tied.
        let mut one_winner = vec![-1.0f32; n_expert];
        for i in 0..32 {
            one_winner[i * 8] = 0.5;
        }
        one_winner[137] = 9.0;

        for (label, row) in [
            ("all-equal", all_equal),
            ("plateau", plateau),
            ("repeated-block", repeated),
            ("one-winner", one_winner),
        ] {
            let logits = on_device(row, (1, n_expert), &device);
            let (ids, weights) = moe_router(&logits, top_k, WEIGHTS_SUM_FLOOR as f32).unwrap();
            let (want_ids, want_w) = candle_route(&logits, top_k);
            assert_u32_eq(&ids, &want_ids, &format!("router ties {label}"));
            assert_f32_bits_eq(&weights, &want_w, &format!("router ties {label}"));
        }
    }

    /// Every op resolves its operand via `start_offset * dtype_size`; the other
    /// tests build inputs at offset 0. Feed the router a contiguous view that
    /// starts mid-buffer — a dropped offset would route on the buffer head.
    #[test]
    fn router_handles_offset_views() {
        let device = metal_device().unwrap();
        let (seq, n_expert, top_k, skip) = (4usize, 256usize, 8usize, 3usize);
        let big = on_device(
            uniform(0x0FF5, (seq + skip) * n_expert, -3.0, 3.0),
            (seq + skip, n_expert),
            &device,
        );
        let view = big.narrow(0, skip, seq).unwrap();
        assert!(view.is_contiguous());
        let (ids, weights) = moe_router(&view, top_k, WEIGHTS_SUM_FLOOR as f32).unwrap();
        let (want_ids, want_w) = candle_route(&view, top_k);
        assert_u32_eq(&ids, &want_ids, "router offset view");
        assert_f32_bits_eq(&weights, &want_w, "router offset view");
    }

    /// The renormalization denominator floor is unreachable through a real
    /// softmax row (the top-k share of a distribution over `n_expert` experts is
    /// at least `top_k / n_expert`), so the kernel's clamp is exercised here
    /// against a hand-picked floor LARGER than the true sum: with the floor
    /// binding, the emitted weights must still match candle's chain bit-for-bit,
    /// which is only true if both sides clamp the same way.
    #[test]
    fn router_clamped_denominator_matches_candle_bitwise() {
        let device = metal_device().unwrap();
        let (n_expert, top_k) = (256usize, 8usize);
        let logits = on_device(uniform(0xC1A3, n_expert, -2.0, 2.0), (1, n_expert), &device);

        // A floor of 1.0 always binds (the weights sum to at most 1) and one of
        // exactly the true sum's own value exercises the `>` boundary.
        let (_, unclamped) = candle_route(&logits, top_k);
        let sum: f32 = unclamped
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .sum();
        assert!((sum - 1.0).abs() < 1e-5, "renormalized weights sum to one");

        for floor in [1.0f32, 0.5, WEIGHTS_SUM_FLOOR as f32] {
            let (ids, weights) = moe_router(&logits, top_k, floor).unwrap();
            // The candle side has to clamp with the same floor to be a fair
            // comparison, so re-run its tail here.
            let probs = softmax_last_dim(&logits).unwrap();
            let order = probs
                .contiguous()
                .unwrap()
                .arg_sort_last_dim(false)
                .unwrap();
            let want_ids = order.narrow(1, 0, top_k).unwrap().contiguous().unwrap();
            let w = probs.gather(&want_ids, 1).unwrap();
            let s = w
                .sum_keepdim(1)
                .unwrap()
                .clamp(floor, f32::INFINITY)
                .unwrap();
            let want_w = w.broadcast_div(&s).unwrap();
            assert_u32_eq(&ids, &want_ids, &format!("router floor={floor}"));
            assert_f32_bits_eq(&weights, &want_w, &format!("router floor={floor}"));
        }
    }

    /// The fused epilogue must reproduce the live candle chain (weighted
    /// combine, sigmoid gate, gate multiply, routed+shared add) BIT-FOR-BIT,
    /// across the production decode shape and prefill widths, with
    /// wide-magnitude signed inputs and gate logits spanning both sigmoid
    /// saturation tails.
    #[test]
    fn epilogue_matches_candle_bitwise() {
        let device = metal_device().unwrap();
        let top_k = 8usize;

        for &seq in &[1usize, 5, 31] {
            for &n_out in &[2048usize, 1000] {
                let down_v = wide(
                    0x100 + seq as u64 * 31 + n_out as u64,
                    seq * top_k * n_out,
                    true,
                );
                let w_v = wide(0x300 + n_out as u64, seq * top_k, true);
                let shexp_v = wide(0x500 + seq as u64, seq * n_out, true);
                // Gate logits from deep in the negative tail to deep in the
                // positive one, so both sigmoid saturations are covered.
                let gate_v = uniform(0x700 + seq as u64, seq, -90.0, 90.0);

                let down = Tensor::from_vec(down_v, (seq, top_k, n_out), &Device::Cpu)
                    .unwrap()
                    .to_device(&device)
                    .unwrap();
                let w = on_device(w_v, (seq, top_k), &device);
                let shexp = on_device(shexp_v, (seq, n_out), &device);
                let gate = on_device(gate_v, (seq, 1), &device);

                let fused = moe_epilogue(&down, &w, &shexp, &gate).unwrap();
                assert_eq!(fused.dims(), &[seq, n_out]);
                assert_eq!(fused.dtype(), DType::F32);
                let want = candle_epilogue(&down, &w, &shexp, &gate);
                assert_f32_bits_eq(&fused, &want, &format!("epilogue seq={seq} n_out={n_out}"));
            }
        }
    }

    /// Operand offsets, as for the router: every epilogue input is resolved via
    /// `start_offset`, and the test inputs above all start at zero.
    #[test]
    fn epilogue_handles_offset_views() {
        let device = metal_device().unwrap();
        let (seq, top_k, n_out, skip) = (3usize, 8usize, 512usize, 2usize);

        let down_big = Tensor::from_vec(
            wide(0x9001, (seq + skip) * top_k * n_out, true),
            (seq + skip, top_k, n_out),
            &Device::Cpu,
        )
        .unwrap()
        .to_device(&device)
        .unwrap();
        let down = down_big.narrow(0, skip, seq).unwrap();
        let w = on_device(
            wide(0x9002, (seq + skip) * top_k, true),
            (seq + skip, top_k),
            &device,
        )
        .narrow(0, skip, seq)
        .unwrap();
        let shexp = on_device(
            wide(0x9003, (seq + skip) * n_out, true),
            (seq + skip, n_out),
            &device,
        )
        .narrow(0, skip, seq)
        .unwrap();
        let gate = on_device(
            uniform(0x9004, (seq + skip) * 1, -10.0, 10.0),
            (seq + skip, 1),
            &device,
        )
        .narrow(0, skip, seq)
        .unwrap();
        for t in [&down, &w, &shexp, &gate] {
            assert!(t.is_contiguous(), "narrowed views must stay contiguous");
        }

        let fused = moe_epilogue(&down, &w, &shexp, &gate).unwrap();
        let want = candle_epilogue(&down, &w, &shexp, &gate);
        assert_f32_bits_eq(&fused, &want, "epilogue offset views");
    }

    /// The router kernel's threadgroup-array bounds are spelled out twice: as
    /// `#define`s in moe_glue.metal, which size the shared arrays, and as Rust
    /// constants in dispatch.rs, which refuse an over-large geometry before the
    /// dispatch. Drift between the two is silent — a geometry the Rust side
    /// waves through would write past a shared array — so parse the kernel's
    /// numbers out of the source and compare.
    #[test]
    fn router_geometry_matches_metal() {
        const SRC: &str = include_str!("moe_glue.metal");

        /// The integer in `#define <name> <int>`, ignoring any trailing comment.
        fn define(name: &str) -> usize {
            SRC.lines()
                .find_map(|line| {
                    let rest = line.trim_start().strip_prefix("#define ")?;
                    let rest = rest.strip_prefix(name)?;
                    rest.strip_prefix(' ')?
                        .split_whitespace()
                        .next()?
                        .parse()
                        .ok()
                })
                .unwrap_or_else(|| panic!("moe_glue.metal has no `#define {name} <integer>`"))
        }

        assert_eq!(
            define("MOE_ROUTER_MAX_EXPERTS"),
            dispatch::MOE_ROUTER_MAX_EXPERTS,
            "moe_glue.metal and dispatch.rs disagree on the padded expert bound"
        );
        assert_eq!(
            define("MOE_ROUTER_MAX_SOFTMAX"),
            dispatch::MOE_ROUTER_MAX_SOFTMAX,
            "moe_glue.metal and dispatch.rs disagree on the softmax reduction bound"
        );
        assert_eq!(
            define("MOE_ROUTER_MAX_TOP_K"),
            dispatch::MOE_ROUTER_MAX_TOP_K,
            "moe_glue.metal and dispatch.rs disagree on the selection bound"
        );
        // The softmax phase reuses the low lanes of the bitonic network's
        // threadgroup, so its bound can never be the binding one; and the sum
        // folds a single simdgroup, so the selection bound cannot exceed 32.
        assert!(
            dispatch::MOE_ROUTER_MAX_SOFTMAX <= dispatch::MOE_ROUTER_MAX_EXPERTS,
            "the softmax reduction is narrower than the sort network it shares a threadgroup with"
        );
        assert!(
            dispatch::MOE_ROUTER_MAX_TOP_K <= 32,
            "the top-k sum folds one 32-lane simdgroup"
        );
    }

    #[test]
    fn shape_and_dtype_errors() {
        let device = metal_device().unwrap();
        let floor = WEIGHTS_SUM_FLOOR as f32;

        // Router: rank, dtype, and selection-size contracts.
        let logits = Tensor::zeros((2, 256), DType::F32, &device).unwrap();
        assert!(moe_router(&logits, 0, floor).is_err());
        assert!(moe_router(&logits, 257, floor).is_err());
        assert!(
            moe_router(&logits, 33, floor).is_err(),
            "top_k over one simdgroup"
        );
        let logits_f16 = Tensor::zeros((2, 256), DType::F16, &device).unwrap();
        assert!(moe_router(&logits_f16, 8, floor).is_err());
        let logits_r3 = Tensor::zeros((2, 4, 256), DType::F32, &device).unwrap();
        assert!(moe_router(&logits_r3, 8, floor).is_err());
        // An expert count whose padding overruns the kernel's shared arrays.
        let too_wide = Tensor::zeros((1, 1024), DType::F32, &device).unwrap();
        assert!(moe_router(&too_wide, 8, floor).is_err());
        // Empty rows dispatch a zero-dimension grid, which writes nothing and
        // hands the caller a shape it did not ask for.
        let empty = Tensor::zeros((0, 256), DType::F32, &device).unwrap();
        assert!(moe_router(&empty, 8, floor).is_err());

        // Epilogue: shape agreement across the four operands.
        let down = Tensor::zeros((4, 8, 16), DType::F32, &device).unwrap();
        let w = Tensor::zeros((4, 8), DType::F32, &device).unwrap();
        let shexp = Tensor::zeros((4, 16), DType::F32, &device).unwrap();
        let gate = Tensor::zeros((4, 1), DType::F32, &device).unwrap();
        assert!(moe_epilogue(&down, &w, &shexp, &gate).is_ok());
        let bad_w = Tensor::zeros((4, 7), DType::F32, &device).unwrap();
        assert!(moe_epilogue(&down, &bad_w, &shexp, &gate).is_err());
        let bad_shexp = Tensor::zeros((4, 15), DType::F32, &device).unwrap();
        assert!(moe_epilogue(&down, &w, &bad_shexp, &gate).is_err());
        let bad_gate = Tensor::zeros((4, 2), DType::F32, &device).unwrap();
        assert!(moe_epilogue(&down, &w, &shexp, &bad_gate).is_err());
        let down_f16 = Tensor::zeros((4, 8, 16), DType::F16, &device).unwrap();
        assert!(moe_epilogue(&down_f16, &w, &shexp, &gate).is_err());
        // A top_k whose candle reduction width would exceed one simdgroup.
        let wide_down = Tensor::zeros((1, 66, 4), DType::F32, &device).unwrap();
        let wide_w = Tensor::zeros((1, 66), DType::F32, &device).unwrap();
        let one_shexp = Tensor::zeros((1, 4), DType::F32, &device).unwrap();
        let one_gate = Tensor::zeros((1, 1), DType::F32, &device).unwrap();
        let err = moe_epilogue(&wide_down, &wide_w, &one_shexp, &one_gate)
            .unwrap_err()
            .to_string();
        assert!(err.contains("> 32"), "unexpected error: {err}");
    }
}
