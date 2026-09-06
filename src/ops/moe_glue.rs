use anyhow::Result;
use candle_core::Tensor;
use candle_core::quantized::GgmlDType;

use crate::gguf::QuantPlane;
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

/// Whether the FUSED SHARED EXPERT covers this block's geometry and weight
/// dtypes — the pair of kernels that swallow the shared expert's five decode
/// dispatches (gate gemv, up gemv, `silu_mul`, down gemv, gate logit) into one
/// plus a shexp-aware epilogue. The bounds are the kernels': q8_0 planes
/// throughout, a `hidden` that is a whole number of q8_0 blocks no wider than
/// the first kernel's staged register array, an `inner` its epilogue's single
/// simdgroup can fold, and a `top_k` inside that simdgroup. A block outside them
/// keeps the five-dispatch chain rather than failing. See
/// [`moe_shexp_gate_up`] for the split.
pub fn moe_shexp_fused_supported(
    hidden: usize,
    inner: usize,
    top_k: usize,
    gate_dtype: GgmlDType,
    up_dtype: GgmlDType,
    down_dtype: GgmlDType,
) -> bool {
    dispatch::moe_shexp_fused_supported(hidden, inner, top_k, gate_dtype, up_dtype, down_dtype)
}

/// The shared expert's gate and up q8_0 projections, their SwiGLU activation and
/// the scalar gate logit, in ONE dispatch (`kernel_moe_shexp_gate_up`) where the
/// classic chain spends four: two `QMatMul` gemvs, `ops::silu_mul`, and the
/// candle f32 matmul against `ffn_gate_inp_shexp`.
///
/// `x` is the block's normed input `[n, hidden]` f32, `gate` / `up` the
/// `[inner, hidden]` q8_0 projections' raw bytes, and `gate_inp` the
/// `ffn_gate_inp_shexp` row as the `[hidden, 1]` f32 tensor `SharedExpert`
/// already holds. Returns `(h [n, inner], logit [n, 1])` — the UNGATED SwiGLU
/// bottleneck and the RAW pre-sigmoid gate logit, both of which
/// [`moe_epilogue_shexp`] consumes. Metal only; the caller's kill switches are
/// `XWEN_MOE_SHEXP_CLASSIC` and `XWEN_MOE_GLUE_CLASSIC`.
pub fn moe_shexp_gate_up(
    x: &Tensor,
    gate: &QuantPlane,
    up: &QuantPlane,
    gate_inp: &Tensor,
    hidden: usize,
    inner: usize,
) -> Result<(Tensor, Tensor)> {
    dispatch::run_moe_shexp_gate_up(x, gate, up, gate_inp, hidden, inner)
}

/// [`moe_epilogue`] with the shared expert's q8_0 DOWN projection folded into
/// the same pass (`kernel_moe_epilogue_shexp`):
/// `dst[s,c] = Σ_k down[s,k,c] * w[s,k] + (Σ_j down_shexp[c,j] * h[s,j]) *
/// sigmoid(gate[s])`. `h` is [`moe_shexp_gate_up`]'s bottleneck `[seq, inner]`
/// and `down_shexp` the `[n_out, inner]` projection's raw bytes; the materialized
/// `shexp` tensor [`moe_epilogue`] takes is never built.
///
/// BOUNDED, not bitwise, against [`moe_epilogue`]: this kernel's routed combine
/// folds over a full simdgroup where that one folds over
/// `next_pow2(top_k/2)` lanes. [`moe_epilogue`] itself is untouched and stays
/// the bit-identical anchor the strict parity tier runs. Metal only.
pub fn moe_epilogue_shexp(
    down: &Tensor,
    weights: &Tensor,
    h: &Tensor,
    down_shexp: &QuantPlane,
    gate: &Tensor,
    inner: usize,
) -> Result<Tensor> {
    dispatch::run_moe_epilogue_shexp(down, weights, h, down_shexp, gate, inner)
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

    // ---------------------------------------------------------------- shexp

    /// The two production geometries the fused shared expert ships for:
    /// `(hidden, inner, top_k, n_expert)` on Flash-Next and on the 35B-A3B.
    /// `n_out` is `hidden` — the shared expert projects back to the block width.
    const SHEXP_CASES: [(usize, usize, usize); 2] = [(2560, 640, 10), (2048, 512, 8)];

    /// The shared expert's three q8_0 planes plus everything the two comparison
    /// paths need from them.
    ///
    /// Each plane VIEWS the last `n_out` rows of a `[skip + n_out, k]` upload,
    /// so `skip > 0` is the nonzero-`base_off` case with no other change; the
    /// `QTensor` alongside is a quantization of exactly those rows, and the
    /// constructor asserts the two agree byte for byte, which is what makes the
    /// classic chain a fair comparison rather than a second quantizer.
    struct ShexpWeights {
        gate: QuantPlane,
        up: QuantPlane,
        down: QuantPlane,
        gate_q: std::sync::Arc<candle_core::quantized::QTensor>,
        up_q: std::sync::Arc<candle_core::quantized::QTensor>,
        down_q: std::sync::Arc<candle_core::quantized::QTensor>,
        /// The `QTensor`s that own the planes' uploads. Never read — held only
        /// so the allocations stay claimed. A `QuantPlane` retains the MTLBuffer
        /// OBJECT, not candle's pool `Arc`, so dropping the tensor that made the
        /// upload lets the pool hand the same memory to the next `new_buffer`
        /// and a later kernel writes its output over these weights (the trap
        /// `dispatch::testutil::build_stack` documents).
        _owners: Vec<std::sync::Arc<candle_core::quantized::QTensor>>,
        gate_deq: Vec<f32>,
        up_deq: Vec<f32>,
        down_deq: Vec<f32>,
        /// `ffn_gate_inp_shexp` in the `[hidden, 1]` shape `SharedExpert` holds.
        gate_inp: Tensor,
        gate_inp_v: Vec<f32>,
    }

    /// A `[skip + n_out, k]` q8_0 weight uploaded once, handed back as a plane
    /// whose `base_off` skips the first `skip` rows, plus the `QTensor` and the
    /// dequantized f32 of exactly the rows the plane covers.
    fn build_q8_plane(
        dev: &Device,
        skip: usize,
        n_out: usize,
        k: usize,
        data: &[f32],
    ) -> (
        QuantPlane,
        std::sync::Arc<candle_core::quantized::QTensor>,
        std::sync::Arc<candle_core::quantized::QTensor>,
        Vec<f32>,
    ) {
        use candle_core::quantized::{QStorage, QTensor};
        let cpu = Device::Cpu;
        assert_eq!(data.len(), (skip + n_out) * k);
        let whole = QTensor::quantize(
            &Tensor::from_vec(data.to_vec(), (skip + n_out, k), &cpu).unwrap(),
            GgmlDType::Q8_0,
        )
        .unwrap();
        // Quantizing the tail rows alone must give the same bytes: q8_0 blocks
        // are 32 contiguous elements of one row, so a row's encoding cannot
        // depend on the rows above it. Asserted rather than assumed, because it
        // is what lets the offset plane and the QTensor below be compared.
        let tail = QTensor::quantize(
            &Tensor::from_vec(data[skip * k..].to_vec(), (n_out, k), &cpu).unwrap(),
            GgmlDType::Q8_0,
        )
        .unwrap();
        let row_bytes = k / GgmlDType::Q8_0.block_size() * GgmlDType::Q8_0.type_size();
        assert_eq!(
            &whole.data().unwrap()[skip * row_bytes..],
            &tail.data().unwrap()[..],
            "a q8_0 row's bytes must not depend on the rows above it"
        );
        let deq = tail
            .dequantize(&cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // Upload the WHOLE weight once and retain its buffer before the storage
        // moves into a QTensor, the way the loader does.
        let storage = QStorage::from_data(whole.data().unwrap(), dev, GgmlDType::Q8_0).unwrap();
        let QStorage::Metal(qms) = &storage else {
            panic!("expected Metal quantized storage")
        };
        let buffer = std::sync::Arc::new(qms.buffer().clone());
        // MOVED into a QTensor the caller must keep alive: the plane's
        // `buffer` above is a retained handle on the same MTLBuffer, NOT a
        // clone of candle's pool `Arc`, so once this tensor drops the pool is
        // free to hand the same memory out again — and the next kernel output
        // allocated from it lands on top of these weights.
        let owner = std::sync::Arc::new(QTensor::new(storage, (skip + n_out, k)).unwrap());

        // The classic chain's QMatMul reads its own upload of the same bytes.
        let tail_storage = QStorage::from_data(tail.data().unwrap(), dev, GgmlDType::Q8_0).unwrap();
        let qtensor = std::sync::Arc::new(QTensor::new(tail_storage, (n_out, k)).unwrap());

        (
            QuantPlane {
                buffer,
                base_off: skip * row_bytes,
                dtype: GgmlDType::Q8_0,
                out_dim: n_out,
                in_dim: k,
            },
            owner,
            qtensor,
            deq,
        )
    }

    /// Rows generated AHEAD of the ones a plane covers, so that `skip` only
    /// moves where the plane's view starts and never which values it sees: every
    /// plane views the LAST `n_out` rows of its data, and the slice handed to
    /// `build_q8_plane` begins `skip` rows before them. That is what makes the
    /// offset and zero-offset weight sets comparable bit for bit.
    const SHEXP_PAD_ROWS: usize = 5;

    fn shexp_weights(dev: &Device, skip: usize, hidden: usize, inner: usize) -> ShexpWeights {
        use crate::ops::dispatch::testutil::pseudo_random;
        assert!(skip <= SHEXP_PAD_ROWS);
        let head = SHEXP_PAD_ROWS - skip;
        let gate_all = pseudo_random(
            (SHEXP_PAD_ROWS + inner) * hidden,
            0xA1 + hidden as u64,
            -0.06,
            0.06,
        );
        let up_all = pseudo_random(
            (SHEXP_PAD_ROWS + inner) * hidden,
            0xA2 + hidden as u64,
            -0.05,
            0.05,
        );
        let down_all = pseudo_random(
            (SHEXP_PAD_ROWS + hidden) * inner,
            0xA3 + hidden as u64,
            -0.04,
            0.04,
        );
        let (gate, gate_own, gate_q, gate_deq) =
            build_q8_plane(dev, skip, inner, hidden, &gate_all[head * hidden..]);
        let (up, up_own, up_q, up_deq) =
            build_q8_plane(dev, skip, inner, hidden, &up_all[head * hidden..]);
        let (down, down_own, down_q, down_deq) =
            build_q8_plane(dev, skip, hidden, inner, &down_all[head * inner..]);
        let gate_inp_v = pseudo_random(hidden, 0xA4 + hidden as u64, -0.1, 0.1);
        let gate_inp = Tensor::from_vec(gate_inp_v.clone(), (hidden, 1), &Device::Cpu)
            .unwrap()
            .to_device(dev)
            .unwrap();
        ShexpWeights {
            gate,
            up,
            down,
            gate_q,
            up_q,
            down_q,
            _owners: vec![gate_own, up_own, down_own],
            gate_deq,
            up_deq,
            down_deq,
            gate_inp,
            gate_inp_v,
        }
    }

    /// The whole fused pair on the host, in f32, over the dequantized weights
    /// the kernels read: `h = silu(gate . x) * (up . x)`, the raw gate logit,
    /// and the epilogue's `routed + (down_shexp . h) * sigmoid(logit)`. Every
    /// dot is a sequential f32 sum, which is a different association from every
    /// path under test — that is the point: the bound holds all of them.
    #[allow(clippy::too_many_arguments)]
    fn host_shexp(
        w: &ShexpWeights,
        x: &[f32],
        routed: &[f32],
        rw: &[f32],
        n: usize,
        hidden: usize,
        inner: usize,
        top_k: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut h = vec![0f32; n * inner];
        let mut logit = vec![0f32; n];
        let mut dst = vec![0f32; n * hidden];
        for s in 0..n {
            let xr = &x[s * hidden..(s + 1) * hidden];
            for r in 0..inner {
                let mut g = 0f32;
                let mut u = 0f32;
                let grow = &w.gate_deq[r * hidden..(r + 1) * hidden];
                let urow = &w.up_deq[r * hidden..(r + 1) * hidden];
                for ((gw, uw), xv) in grow.iter().zip(urow).zip(xr) {
                    g += gw * xv;
                    u += uw * xv;
                }
                // kernel_moe_silu_mul's silu, then a separate multiply.
                h[s * inner + r] = (g / (1.0 + (-g).exp())) * u;
            }
            let mut acc = 0f32;
            for (gw, xv) in w.gate_inp_v.iter().zip(xr) {
                acc += gw * xv;
            }
            logit[s] = acc;
            let sig = 1.0f32 / (1.0 + (-logit[s]).exp());
            for c in 0..hidden {
                let mut combined = 0f32;
                for k in 0..top_k {
                    combined += routed[(s * top_k + k) * hidden + c] * rw[s * top_k + k];
                }
                let mut shared = 0f32;
                for j in 0..inner {
                    shared += w.down_deq[c * inner + j] * h[s * inner + j];
                }
                dst[s * hidden + c] = combined + shared * sig;
            }
        }
        (h, logit, dst)
    }

    /// The five-dispatch chain the fused pair replaces, over the SAME quantized
    /// bytes: two `QMatMul` gemvs, `ops::silu_mul`, a third gemv, the candle f32
    /// gate matmul, and the plain `moe_epilogue`. Spelled out rather than called
    /// into moe.rs so an edit to the production chain shows up here as a failure
    /// instead of silently moving the target.
    fn classic_shexp_chain(
        w: &ShexpWeights,
        x: &Tensor,
        routed: &Tensor,
        weights: &Tensor,
    ) -> Tensor {
        use candle_core::Module;
        let mm = |qt: &std::sync::Arc<candle_core::quantized::QTensor>, x: &Tensor| {
            candle_core::quantized::QMatMul::from_arc(qt.clone())
                .unwrap()
                .forward(x)
                .unwrap()
        };
        let g = mm(&w.gate_q, x);
        let u = mm(&w.up_q, x);
        let h = crate::ops::silu_mul(&g, &u).unwrap();
        let shexp = mm(&w.down_q, &h);
        let logit = x.matmul(&w.gate_inp).unwrap();
        moe_epilogue(routed, weights, &shexp, &logit).unwrap()
    }

    fn assert_rel_l2(got: &Tensor, want: &[f32], tol: f32, what: &str) {
        use crate::ops::dispatch::testutil::rel_l2;
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(g.len(), want.len(), "{what}: length");
        // A non-finite element is its own diagnosis — a NaN rel_l2 says only
        // "somewhere" — so name the first one and which side it is on.
        for (i, (a, b)) in g.iter().zip(want).enumerate() {
            assert!(
                a.is_finite() && b.is_finite(),
                "{what}: element {i} is not finite (got {a}, want {b})"
            );
        }
        let e = rel_l2(&g, want);
        assert!(e <= tol, "{what}: rel_l2 {e} > {tol}");
    }

    /// The fused shared expert at both production geometries, across the token
    /// window it is taken in, graded against BOTH an f32 host reference and the
    /// five-dispatch chain it replaces.
    ///
    /// Bounded, never bitwise: all three dots are reassociated (the gate/up rows
    /// fold per-thread `simd_sum` partials where `QMatMul`'s gemv folds its own
    /// partition, the shexp down row folds per-lane block partials, and the
    /// routed combine runs over 32 lanes where `kernel_moe_epilogue` runs over
    /// `next_pow2(top_k/2)`). The HOST bound — rel_l2 <= 1e-5 — is the one that
    /// holds the pair. The cross-path bound is looser above one token for the
    /// same reason the fused hc gate's is
    /// (`gate_fused_matches_reference`): `QMatMul` takes its matmul kernel past
    /// one row and stages half activation tiles, so above n = 1 it is the
    /// CLASSIC side that moves away from the oracle, measured ~2.2e-4 at the
    /// Flash-Next geometry where the pair stays near 1e-6. That is asserted
    /// below rather than assumed, so the loose 5e-4 cross-path bound cannot
    /// quietly absorb a regression in the pair itself.
    #[test]
    fn shexp_fused_matches_reference() {
        use crate::ops::dispatch::testutil::{pseudo_random, rel_l2};
        let device = metal_device().unwrap();

        for (hidden, inner, top_k) in SHEXP_CASES {
            let w = shexp_weights(&device, 0, hidden, inner);
            assert!(moe_shexp_fused_supported(
                hidden,
                inner,
                top_k,
                GgmlDType::Q8_0,
                GgmlDType::Q8_0,
                GgmlDType::Q8_0
            ));

            for &n in &[1usize, 3, 8] {
                let case = format!("hidden={hidden} inner={inner} top_k={top_k} n={n}");
                let x_v = pseudo_random(n * hidden, 0xB1 + n as u64, -2.0, 2.0);
                let routed_v = pseudo_random(n * top_k * hidden, 0xB2 + n as u64, -1.0, 1.0);
                let rw_v = pseudo_random(n * top_k, 0xB3 + n as u64, 0.0, 1.0);

                let x = on_device(x_v.clone(), (n, hidden), &device);
                let routed = Tensor::from_vec(routed_v.clone(), (n, top_k, hidden), &Device::Cpu)
                    .unwrap()
                    .to_device(&device)
                    .unwrap();
                let rw = on_device(rw_v.clone(), (n, top_k), &device);

                let (h, logit) =
                    moe_shexp_gate_up(&x, &w.gate, &w.up, &w.gate_inp, hidden, inner).unwrap();
                assert_eq!(h.dims(), &[n, inner]);
                assert_eq!(logit.dims(), &[n, 1]);
                let dst = moe_epilogue_shexp(&routed, &rw, &h, &w.down, &logit, inner).unwrap();
                assert_eq!(dst.dims(), &[n, hidden]);
                assert_eq!(dst.dtype(), DType::F32);

                let (want_h, want_logit, want_dst) =
                    host_shexp(&w, &x_v, &routed_v, &rw_v, n, hidden, inner, top_k);
                assert_rel_l2(&h, &want_h, 1e-5, &format!("fused shexp h {case}"));
                assert_rel_l2(
                    &logit,
                    &want_logit,
                    1e-5,
                    &format!("fused shexp logit {case}"),
                );
                assert_rel_l2(&dst, &want_dst, 1e-5, &format!("fused shexp dst {case}"));

                let classic = classic_shexp_chain(&w, &x, &routed, &rw);
                let classic_v: Vec<f32> = classic.flatten_all().unwrap().to_vec1().unwrap();
                let tol = if n == 1 { 1e-6 } else { 5e-4 };
                assert_rel_l2(
                    &dst,
                    &classic_v,
                    tol,
                    &format!("fused shexp vs the classic chain {case}"),
                );
                if n > 1 {
                    // Which side the cross-path gap belongs to, stated as an
                    // assertion rather than a comment: past one row `QMatMul`
                    // takes its matmul kernel and stages half activation tiles,
                    // so the CLASSIC chain leaves the oracle (measured ~2e-4 at
                    // the Flash-Next geometry) while the fused pair stays at the
                    // ~1e-6 the assertion above holds it to. If that ever
                    // inverts, the loose cross-path bound above is hiding a
                    // regression in the pair and this fails first.
                    let fused_err = rel_l2(
                        &dst.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                        &want_dst,
                    );
                    let classic_err = rel_l2(&classic_v, &want_dst);
                    assert!(
                        fused_err <= classic_err,
                        "{case}: the fused pair sits {fused_err} from the f32 oracle where the \
                         classic chain sits {classic_err} — the pair must be the closer of the two"
                    );
                }
            }
        }
    }

    /// Every operand is resolved through a start offset — the planes through
    /// `base_off`, the tensors through `start_offset * 4` — and the tests above
    /// build all of them at zero. A dropped offset would silently read the
    /// buffer head, so run the same arithmetic over offset views and over
    /// copies at zero and require the two BITWISE.
    #[test]
    fn shexp_fused_honours_offset_views() {
        use crate::ops::dispatch::testutil::pseudo_random;
        let device = metal_device().unwrap();
        let (hidden, inner, top_k) = SHEXP_CASES[1];
        let (n, skip) = (3usize, 5usize);

        // Same source values in both weight sets; only the plane offset differs.
        let flat = shexp_weights(&device, 0, hidden, inner);
        let offset = shexp_weights(&device, skip, hidden, inner);
        // The offset planes must really carry one, or this test proves nothing.
        assert_ne!(offset.gate.base_off, 0);
        assert_ne!(offset.down.base_off, 0);

        let big_x = on_device(
            pseudo_random((n + skip) * hidden, 0xC1, -2.0, 2.0),
            (n + skip, hidden),
            &device,
        );
        let x_view = big_x.narrow(0, skip, n).unwrap();
        assert_ne!(
            x_view.layout().start_offset(),
            0,
            "the narrowed view must actually carry an offset"
        );
        let x_copy = on_device(
            x_view.flatten_all().unwrap().to_vec1().unwrap(),
            (n, hidden),
            &device,
        );

        let routed = Tensor::from_vec(
            pseudo_random(n * top_k * hidden, 0xC2, -1.0, 1.0),
            (n, top_k, hidden),
            &Device::Cpu,
        )
        .unwrap()
        .to_device(&device)
        .unwrap();
        let rw = on_device(
            pseudo_random(n * top_k, 0xC3, 0.0, 1.0),
            (n, top_k),
            &device,
        );

        let run = |w: &ShexpWeights, x: &Tensor, offset_h: bool| {
            let (h, logit) =
                moe_shexp_gate_up(x, &w.gate, &w.up, &w.gate_inp, hidden, inner).unwrap();
            // Offset the bottleneck too: pad it with a leading row and hand the
            // epilogue the narrowed view of that.
            let h = if offset_h {
                let padded = Tensor::cat(
                    &[&Tensor::zeros((1, inner), DType::F32, &device).unwrap(), &h],
                    0,
                )
                .unwrap();
                let view = padded.narrow(0, 1, n).unwrap();
                assert_ne!(view.layout().start_offset(), 0);
                view
            } else {
                h
            };
            let dst = moe_epilogue_shexp(&routed, &rw, &h, &w.down, &logit, inner).unwrap();
            (dst, logit)
        };

        let (v_dst, v_logit) = run(&offset, &x_view, true);
        let (c_dst, c_logit) = run(&flat, &x_copy, false);
        assert_f32_bits_eq(&v_dst, &c_dst, "fused shexp over offset views");
        assert_f32_bits_eq(&v_logit, &c_logit, "fused shexp logit over offset views");
    }

    /// Geometry outside the kernels' bounds is refused by the predicate the
    /// caller asks — which is what keeps such a block on the five-dispatch
    /// chain — and the launchers hard-error rather than reading past an array
    /// their partition does not cover. Each case below breaks exactly one bound.
    #[test]
    fn shexp_fused_refuses_unsupported_geometry() {
        use crate::ops::dispatch::testutil::pseudo_random;
        let device = metal_device().unwrap();
        let q8 = GgmlDType::Q8_0;
        let (hidden, inner, top_k) = SHEXP_CASES[0];

        assert!(moe_shexp_fused_supported(hidden, inner, top_k, q8, q8, q8));
        // Any dtype but q8_0: the kernels read that block layout directly.
        assert!(!moe_shexp_fused_supported(
            hidden,
            inner,
            top_k,
            GgmlDType::Q4K,
            q8,
            q8
        ));
        assert!(!moe_shexp_fused_supported(
            hidden,
            inner,
            top_k,
            q8,
            q8,
            GgmlDType::Q4K
        ));
        // A hidden that is not a whole number of q8_0 blocks: the staged
        // activation slice would straddle one.
        assert!(!moe_shexp_fused_supported(
            hidden + 16,
            inner,
            top_k,
            q8,
            q8,
            q8
        ));
        // ... and one wider than the staged register array (128 threads x 2
        // blocks = 8192 elements).
        assert!(moe_shexp_fused_supported(8192, inner, top_k, q8, q8, q8));
        assert!(!moe_shexp_fused_supported(
            8192 + 32,
            inner,
            top_k,
            q8,
            q8,
            q8
        ));
        // An inner outside the epilogue simdgroup's per-lane block bound
        // (32 lanes x 4 blocks = 4096), or not a whole number of blocks.
        assert!(moe_shexp_fused_supported(hidden, 4096, top_k, q8, q8, q8));
        assert!(!moe_shexp_fused_supported(
            hidden,
            4096 + 32,
            top_k,
            q8,
            q8,
            q8
        ));
        assert!(!moe_shexp_fused_supported(
            hidden,
            inner + 16,
            top_k,
            q8,
            q8,
            q8
        ));
        // A top_k past one simdgroup has no reduction.
        assert!(moe_shexp_fused_supported(hidden, inner, 32, q8, q8, q8));
        assert!(!moe_shexp_fused_supported(hidden, inner, 33, q8, q8, q8));
        assert!(!moe_shexp_fused_supported(hidden, inner, 0, q8, q8, q8));

        // The launchers refuse what the predicate refuses. The planes here are
        // real q8_0 weights of a covered shape; the GEOMETRY passed alongside
        // them is the unsupported one.
        let (h_small, i_small, k_small) = (256usize, 128usize, 4usize);
        let w = shexp_weights(&device, 0, h_small, i_small);
        let x = on_device(
            pseudo_random(h_small, 0xD1, -1.0, 1.0),
            (1, h_small),
            &device,
        );
        assert!(
            moe_shexp_gate_up(&x, &w.gate, &w.up, &w.gate_inp, h_small + 16, i_small).is_err(),
            "a hidden that is not a whole number of blocks"
        );
        assert!(
            moe_shexp_gate_up(&x, &w.down, &w.up, &w.gate_inp, h_small, i_small).is_err(),
            "a plane whose shape contradicts the geometry"
        );
        let (h, logit) =
            moe_shexp_gate_up(&x, &w.gate, &w.up, &w.gate_inp, h_small, i_small).unwrap();
        let routed = Tensor::zeros((1, k_small, h_small), DType::F32, &device).unwrap();
        let rw = Tensor::zeros((1, k_small), DType::F32, &device).unwrap();
        assert!(moe_epilogue_shexp(&routed, &rw, &h, &w.down, &logit, i_small).is_ok());
        assert!(
            moe_epilogue_shexp(&routed, &rw, &h, &w.down, &logit, i_small + 16).is_err(),
            "an inner that is not a whole number of blocks"
        );
        assert!(
            moe_epilogue_shexp(&routed, &rw, &h, &w.gate, &logit, i_small).is_err(),
            "a down plane whose shape contradicts the geometry"
        );

        // A plane whose declared shape AGREES with the geometry but outruns its
        // own allocation. Nothing else in the call carries the weight's length —
        // it is raw bytes, not a tensor — so without the buffer bound the
        // kernels would read off the end of device memory.
        let lying_gate = QuantPlane {
            out_dim: i_small * 4,
            ..w.gate.clone()
        };
        assert!(
            moe_shexp_gate_up(&x, &lying_gate, &w.up, &w.gate_inp, h_small, i_small * 4).is_err()
        );
        let lying_down = QuantPlane {
            out_dim: h_small * 4,
            ..w.down.clone()
        };
        let wide_routed = Tensor::zeros((1, k_small, h_small * 4), DType::F32, &device).unwrap();
        assert!(moe_epilogue_shexp(&wide_routed, &rw, &h, &lying_down, &logit, i_small).is_err());
    }

    /// The fused pair's launch geometry is spelled out in BOTH languages — a
    /// `#define` shaping the kernels' threadgroups and register arrays, and the
    /// host constants that size the grid and refuse a geometry those bounds do
    /// not cover. Nothing links them, and moving only one side would produce a
    /// launch whose threads and arrays disagree with no diagnostic, so this test
    /// is the link (same shape as `router_geometry_matches_metal`).
    #[test]
    fn moe_shexp_constants_match_metal() {
        const SRC: &str = include_str!("moe_glue.metal");
        let parse = |name: &str| -> usize {
            let mut found = None;
            for line in SRC.lines() {
                if let Some(rest) = line.trim().strip_prefix(&format!("#define {name} ")) {
                    assert!(
                        found.is_none(),
                        "moe_glue.metal defines {name} more than once"
                    );
                    found = Some(
                        rest.trim()
                            .parse::<usize>()
                            .unwrap_or_else(|_| panic!("{name} must be a plain integer literal")),
                    );
                }
            }
            found.unwrap_or_else(|| panic!("moe_glue.metal must #define {name}"))
        };
        for (name, host) in [
            ("MOE_SHEXP_THREADS", dispatch::MOE_SHEXP_THREADS),
            (
                "MOE_SHEXP_MAX_BLK_PER_THREAD",
                dispatch::MOE_SHEXP_MAX_BLK_PER_THREAD,
            ),
            ("MOE_SHEXP_ROWS_PER_TG", dispatch::MOE_SHEXP_ROWS_PER_TG),
            (
                "MOE_SHEXP_MAX_BLK_PER_LANE",
                dispatch::MOE_SHEXP_MAX_BLK_PER_LANE,
            ),
            (
                "MOE_SHEXP_EPILOGUE_THREADS",
                dispatch::MOE_SHEXP_EPILOGUE_THREADS,
            ),
        ] {
            assert_eq!(
                parse(name),
                host,
                "moe_glue.metal's {name} and dispatch.rs's ({host}) must agree"
            );
        }
        // The epilogue's reduction is ONE simd_sum, so its threadgroup can never
        // be wider than a simdgroup — that is the contract it inherits from
        // kernel_moe_epilogue, not a tunable.
        assert_eq!(
            dispatch::MOE_SHEXP_EPILOGUE_THREADS,
            32,
            "the shexp-aware epilogue folds a single 32-lane simdgroup"
        );
        // And the first kernel's threadgroup must be a whole number of them, or
        // its cross-simdgroup fold walks slots no simdgroup wrote.
        assert_eq!(dispatch::MOE_SHEXP_THREADS % 32, 0);
    }
}
