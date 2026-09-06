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

/// Whether one of the fused shared expert's three planes is bindable, as
/// opposed to whether its geometry is covered ([`moe_shexp_fused_supported`] is
/// the geometry half). Two conditions, both of which the launchers turn into a
/// hard error: a 2-byte-aligned `base_off` — the kernels index the weight
/// through `device const moe_block_q8_0 *`, whose alignment is that of its
/// `half` scale, and rows are whole blocks so only the bound offset can break it
/// — and a declared `[out_dim, in_dim]` shape that fits the allocation the plane
/// views. Neither is implied by the tensor shapes: a plane is raw bytes, so
/// nothing else in the call carries its length.
///
/// A custom GGUF whose shared expert fails either must fall back to the
/// five-dispatch chain rather than fail the forward, which is why
/// `MoeBlock::fused_shexp` asks this before dispatching.
pub fn moe_shexp_plane_bindable(plane: &QuantPlane) -> bool {
    plane.base_off.is_multiple_of(2)
        && dispatch::check_plane_fits(plane, "shared expert projection").is_ok()
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
    /// `(hidden, inner, top_k)` on Flash-Next and on the 35B-A3B. `n_out` is
    /// `hidden` — the shared expert projects back to the block width.
    const SHEXP_CASES: [(usize, usize, usize); 2] = [(2560, 640, 10), (2048, 512, 8)];

    /// Geometries at the TOP of what the predicate admits, in the same
    /// `(hidden, inner, top_k)` shape. Neither is a shipped checkpoint; both are
    /// here because production is one block per thread in both kernels
    /// (hidden 2560/2048 is 80/64 blocks over 128 threads, inner 640/512 is
    /// 20/16 blocks over 32 lanes), so the strided `p > 0` iterations of the
    /// staging loop in `kernel_moe_shexp_gate_up` and of the row fold in
    /// `kernel_moe_epilogue_shexp` never execute at a shipped shape. `hidden`
    /// 8192 is exactly `MOE_SHEXP_THREADS * MOE_SHEXP_MAX_BLK_PER_THREAD` q8_0
    /// blocks wide and `inner` 4096 exactly `32 * MOE_SHEXP_MAX_BLK_PER_LANE` —
    /// each the last width the predicate admits, so a partition that dropped its
    /// final strided iteration would show up here and nowhere else.
    const SHEXP_PARTITION_CASES: [(usize, usize, usize); 2] = [(8192, 2048, 10), (5120, 4096, 8)];

    /// The shared expert's three q8_0 planes plus everything the two comparison
    /// paths need from them.
    ///
    /// Each plane VIEWS the last `n_out` rows of a `[skip + n_out, k]` upload,
    /// so `skip > 0` is the nonzero-`base_off` case with no other change; the
    /// `QLinear` alongside wraps a quantization of exactly those rows, and the
    /// constructor asserts the two agree byte for byte, which is what makes the
    /// classic chain a fair comparison rather than a second quantizer.
    struct ShexpWeights {
        gate: QuantPlane,
        up: QuantPlane,
        down: QuantPlane,
        /// The same three weights as PLANED `QLinear`s, built the way
        /// `SharedExpert::new` builds them (`Weights::qlinear_with_buffer`), so
        /// the classic arm of the comparison takes the route production's
        /// classic arm takes — `QMatMul` at one token, `ops::matmul_mv_ext` at
        /// 2..=8 — rather than `QMatMul` at every count.
        gate_lin: crate::gguf::QLinear,
        up_lin: crate::gguf::QLinear,
        down_lin: crate::gguf::QLinear,
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
    /// whose `base_off` skips the first `skip` rows, plus a planed `QLinear`
    /// over the same rows and the dequantized f32 of exactly the rows the plane
    /// covers.
    fn build_q8_plane(
        dev: &Device,
        skip: usize,
        n_out: usize,
        k: usize,
        data: &[f32],
    ) -> (
        QuantPlane,
        std::sync::Arc<candle_core::quantized::QTensor>,
        crate::gguf::QLinear,
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

        // The classic chain reads its own upload of the same bytes, through a
        // planed `QLinear` — `QLinear::from_qtensor_with_buffer` pairs the
        // tensor with a raw view of its OWN allocation exactly as
        // `Weights::qlinear_with_buffer` does for a file-loaded weight, which is
        // what opens the 2..=8-token mv_ext window production's classic arm
        // takes. The layer owns the QTensor, so the allocation the plane views
        // stays claimed for as long as the layer does.
        let tail_storage = QStorage::from_data(tail.data().unwrap(), dev, GgmlDType::Q8_0).unwrap();
        let QStorage::Metal(tail_qms) = &tail_storage else {
            panic!("expected Metal quantized storage")
        };
        let tail_buffer = std::sync::Arc::new(tail_qms.buffer().clone());
        let qtensor = std::sync::Arc::new(QTensor::new(tail_storage, (n_out, k)).unwrap());
        let qlinear = crate::gguf::QLinear::from_qtensor_with_buffer(qtensor, tail_buffer).unwrap();

        (
            QuantPlane {
                buffer,
                base_off: skip * row_bytes,
                dtype: GgmlDType::Q8_0,
                out_dim: n_out,
                in_dim: k,
            },
            owner,
            qlinear,
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
        let (gate, gate_own, gate_lin, gate_deq) =
            build_q8_plane(dev, skip, inner, hidden, &gate_all[head * hidden..]);
        let (up, up_own, up_lin, up_deq) =
            build_q8_plane(dev, skip, inner, hidden, &up_all[head * hidden..]);
        let (down, down_own, down_lin, down_deq) =
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
            gate_lin,
            up_lin,
            down_lin,
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
    /// bytes: three planed `QLinear` projections, `ops::silu_mul` between the
    /// first two, the candle f32 gate matmul, and the plain `moe_epilogue`.
    /// Spelled out rather than called into moe.rs so an edit to the production
    /// chain shows up here as a failure instead of silently moving the target —
    /// which is also why the projections go through `QLinear::forward_gemm`
    /// under the same two conditions `SharedExpert::swiglu_out` applies them:
    /// inside the fused pair's token window that lands on `forward`, so the
    /// route is `QMatMul` at one token and `ops::matmul_mv_ext` at 2..=8.
    fn classic_shexp_chain(
        w: &ShexpWeights,
        x: &Tensor,
        routed: &Tensor,
        weights: &Tensor,
    ) -> Tensor {
        let proj = |lin: &crate::gguf::QLinear, x: &Tensor| {
            if crate::ops::shexp_qmatmul() {
                lin.forward(x).unwrap()
            } else {
                lin.forward_gemm(x).unwrap()
            }
        };
        let g = proj(&w.gate_lin, x);
        let u = proj(&w.up_lin, x);
        let h = crate::ops::silu_mul(&g, &u).unwrap();
        let shexp = proj(&w.down_lin, &h);
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

    /// The fused shared expert at both production geometries and at the two
    /// geometries that make the kernels' strided partitions execute, across the
    /// token window the pair is taken in, graded against BOTH an f32 host
    /// reference and the five-dispatch chain it replaces.
    ///
    /// Bounded, never bitwise: all three dots are reassociated (the gate/up rows
    /// fold per-thread `simd_sum` partials where the classic projection folds
    /// its own partition, the shexp down row folds per-lane block partials, and
    /// the routed combine runs over 32 lanes where `kernel_moe_epilogue` runs
    /// over `next_pow2(top_k/2)`). The HOST bound — rel_l2 <= 1e-5 — is the one
    /// that holds the pair.
    ///
    /// The CROSS-PATH bound grades the pair against the route production's
    /// classic arm really takes, which is not `QMatMul` throughout: the shared
    /// expert's three projections are planed `QLinear`s, so at one token each is
    /// candle's `QMatMul` gemv and at 2..=8 tokens `QLinear::forward` takes the
    /// vendored small-batch mat-vec (`ops::matmul_mv_ext`), which splits every K
    /// reduction across 8 lanes holding interleaved chunks. `QMatMul`'s f16-tile
    /// matmul — the ~2e-4 arm — is never reached inside this window, so both
    /// sides are f32 reassociations of the same dots and the gap between them is
    /// small in both halves of the window.
    ///
    /// Measured 2026-09-06, rel_l2 on `dst` as
    /// pair-vs-classic | pair-vs-oracle | classic-vs-oracle:
    /// - 2560/640:  n=1 5.4e-8 | 8.2e-8 | 8.4e-8; n=3 2.3e-7 | 1.0e-6 | 1.0e-6;
    ///   n=8 1.9e-7 | 6.8e-7 | 7.0e-7
    /// - 2048/512:  n=1 7.5e-8 | 2.1e-7 | 1.9e-7; n=3 2.0e-7 | 6.9e-7 | 6.9e-7;
    ///   n=8 2.1e-7 | 8.3e-7 | 8.4e-7
    /// - 8192/2048: n=1 5.5e-8 | 8.0e-8 | 8.1e-8; n=3 5.6e-7 | 2.7e-6 | 2.5e-6
    /// - 5120/4096: n=1 3.1e-7 | 2.5e-6 | 2.4e-6; n=3 5.9e-7 | 2.6e-6 | 2.8e-6
    ///
    /// The bounds below — 1e-6 at one token, 2e-6 above it — are those maxima
    /// (3.1e-7 and 5.9e-7) with roughly 3x headroom, and the split is real: the
    /// classic side changes kernel at n = 2 and the fused side does not.
    ///
    /// Against the oracle the two arms are LEVEL, ratio 0.95..1.11 in either
    /// direction, so the assertion below is a levelness guard rather than the
    /// ordering one it would be if the classic side still ran an f16-tile
    /// matmul: it fails when the pair becomes materially the worse of the two,
    /// which is what would let the cross-path bound absorb a regression in it.
    #[test]
    fn shexp_fused_matches_reference() {
        use crate::ops::dispatch::testutil::{pseudo_random, rel_l2};
        let device = metal_device().unwrap();

        // The shipped shapes run the whole token window; the partition shapes
        // run one count below the mv_ext window and one inside it. Not the full
        // window there only because the host oracle is O(n * hidden * inner) and
        // these are the widest shapes the predicate admits — the partition
        // iterations they exist to reach do not depend on the token count.
        let cases = SHEXP_CASES
            .iter()
            .map(|g| (*g, &[1usize, 3, 8][..]))
            .chain(SHEXP_PARTITION_CASES.iter().map(|g| (*g, &[1usize, 3][..])));

        for ((hidden, inner, top_k), token_counts) in cases {
            let w = shexp_weights(&device, 0, hidden, inner);
            assert!(moe_shexp_fused_supported(
                hidden,
                inner,
                top_k,
                GgmlDType::Q8_0,
                GgmlDType::Q8_0,
                GgmlDType::Q8_0
            ));

            for &n in token_counts {
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
                // The classic side runs `QMatMul`'s gemv at one token and the
                // small-batch mv_ext kernel above it, so the gap has two
                // regimes; both bounds are the measured maxima in the doc
                // comment with headroom, not round numbers.
                let tol = if n == 1 { 1e-6 } else { 2e-6 };
                assert_rel_l2(
                    &dst,
                    &classic_v,
                    tol,
                    &format!("fused shexp vs the classic chain {case}"),
                );
                if n > 1 {
                    // Neither arm owns the cross-path gap: both are f32
                    // reassociations of the same dots and they sit level against
                    // the oracle, within 8% of each other either way across the
                    // four geometries. What must not happen is the pair becoming
                    // materially the worse of the two — that is the failure the
                    // bound above could otherwise absorb — so this is a
                    // levelness guard with a 1.5x margin over the measured
                    // spread, not a claim about which side is closer.
                    let fused_err = rel_l2(
                        &dst.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                        &want_dst,
                    );
                    let classic_err = rel_l2(&classic_v, &want_dst);
                    assert!(
                        fused_err <= classic_err * 1.5,
                        "{case}: the fused pair sits {fused_err} from the f32 oracle where the \
                         classic chain sits {classic_err} — the pair must stay level with the \
                         chain it replaces, not fall behind it"
                    );
                }
            }
        }
    }

    /// Every geometry the predicate admits, DISPATCHED — not merely asked about.
    ///
    /// `moe_shexp_fused_supported` admits a `hidden` of up to
    /// `MOE_SHEXP_THREADS * MOE_SHEXP_MAX_BLK_PER_THREAD` q8_0 blocks (8192
    /// elements) and an `inner` of up to `32 * MOE_SHEXP_MAX_BLK_PER_LANE`
    /// (4096), while both shipped checkpoints sit at one block per thread and
    /// under one per lane. This crosses the two partitions across their whole
    /// admitted range so that every `(blocks per thread, blocks per lane)`
    /// combination the kernels can be handed is executed at least once against
    /// the f32 host oracle — a strided iteration dropped from either loop, or a
    /// ragged tail mishandled in one, cannot pass the whole grid.
    ///
    /// One token: the partitions are over the WEIGHT rows and the reduction, not
    /// over the token count, so the token window is `shexp_fused_matches_reference`'s
    /// job and the widths are this test's.
    #[test]
    fn shexp_fused_covers_every_admitted_shape() {
        use crate::ops::dispatch::testutil::pseudo_random;
        let device = metal_device().unwrap();
        let q8 = GgmlDType::Q8_0;
        let (n, top_k) = (1usize, 4usize);

        for hidden in [2048usize, 2560, 4096, 8192] {
            for inner in [512usize, 640, 1024, 2048, 4096] {
                let case = format!("hidden={hidden} inner={inner}");
                assert!(
                    moe_shexp_fused_supported(hidden, inner, top_k, q8, q8, q8),
                    "{case}: the predicate must admit this shape"
                );

                let w = shexp_weights(&device, 0, hidden, inner);
                let seed = 0xE1 + (hidden * 31 + inner) as u64;
                let x_v = pseudo_random(n * hidden, seed, -2.0, 2.0);
                let routed_v = pseudo_random(n * top_k * hidden, seed + 1, -1.0, 1.0);
                let rw_v = pseudo_random(n * top_k, seed + 2, 0.0, 1.0);

                let x = on_device(x_v.clone(), (n, hidden), &device);
                let routed = Tensor::from_vec(routed_v.clone(), (n, top_k, hidden), &Device::Cpu)
                    .unwrap()
                    .to_device(&device)
                    .unwrap();
                let rw = on_device(rw_v.clone(), (n, top_k), &device);

                let (h, logit) =
                    moe_shexp_gate_up(&x, &w.gate, &w.up, &w.gate_inp, hidden, inner).unwrap();
                let dst = moe_epilogue_shexp(&routed, &rw, &h, &w.down, &logit, inner).unwrap();
                assert_eq!(dst.dims(), &[n, hidden], "{case}");

                let (want_h, want_logit, want_dst) =
                    host_shexp(&w, &x_v, &routed_v, &rw_v, n, hidden, inner, top_k);
                assert_rel_l2(&h, &want_h, 1e-5, &format!("{case}: h"));
                assert_rel_l2(&dst, &want_dst, 1e-5, &format!("{case}: dst"));

                // The gate logit is ONE number per token, so a relative bound on
                // it is a relative bound on a single f32 — and this dot cancels
                // hard: `hidden` terms of a [-2, 2] activation against a
                // [-0.1, 0.1] row sum to something far smaller than the terms
                // themselves, and at some seeds land near zero, where any
                // reassociation reads huge relatively while being tiny
                // absolutely (hidden 2560 / inner 512 does exactly that: rel_l2
                // 1.6e-4 where every other shape in this grid reads under 1e-5).
                // Grade it against the dot's own condition scale instead. The
                // bound is above the sqrt(n) *
                // eps rounding at every admitted width (5.4e-6 * mag at
                // hidden 8192) and orders below what a dropped strided iteration
                // would produce, which is a fraction of `mag` itself.
                let mag: f32 = x_v
                    .iter()
                    .zip(&w.gate_inp_v)
                    .map(|(a, b)| (a * b).abs())
                    .sum();
                let got_logit = logit.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
                assert!(
                    (got_logit - want_logit[0]).abs() <= 1e-4 * mag,
                    "{case}: gate logit {got_logit} against the oracle's {}, off by more than \
                     1e-4 of the dot's {mag} magnitude",
                    want_logit[0]
                );
            }
        }
    }

    /// Every operand is resolved through a start offset — the planes through
    /// `base_off`, the tensors through `start_offset * 4` — and the tests above
    /// build all of them at zero. A dropped offset would silently read the
    /// buffer head, so run the same arithmetic with EVERY binding a narrowed
    /// view into a larger buffer, and again with every one of them a copy at
    /// zero, and require the two BITWISE. Every binding means all of them: the
    /// three planes, the activation, the `ffn_gate_inp_shexp` row, the routed
    /// projection, the routing weights, and the bottleneck and gate logit
    /// flowing from the first kernel into the second.
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
        assert_ne!(offset.up.base_off, 0);
        assert_ne!(offset.down.base_off, 0);

        // The operand values, generated once and shared by the two arms.
        let x_v = pseudo_random(n * hidden, 0xC1, -2.0, 2.0);
        let routed_v = pseudo_random(n * top_k * hidden, 0xC2, -1.0, 1.0);
        let rw_v = pseudo_random(n * top_k, 0xC3, 0.0, 1.0);

        let contiguous = |v: &[f32], dims: &[usize]| -> Tensor {
            Tensor::from_vec(v.to_vec(), dims, &Device::Cpu)
                .unwrap()
                .to_device(&device)
                .unwrap()
        };
        // The same values as the LAST rows of a wider buffer, narrowed back to
        // them: identical elements, a nonzero start offset, and junk ahead of
        // them that a dropped offset would read instead.
        let viewed = |v: &[f32], dims: &[usize], seed: u64| -> Tensor {
            let row: usize = dims[1..].iter().product();
            let mut data = pseudo_random(skip * row, seed, -3.0, 3.0);
            data.extend_from_slice(v);
            let mut wide = dims.to_vec();
            wide[0] += skip;
            let big = Tensor::from_vec(data, wide, &Device::Cpu)
                .unwrap()
                .to_device(&device)
                .unwrap();
            let view = big.narrow(0, skip, dims[0]).unwrap();
            assert_ne!(
                view.layout().start_offset(),
                0,
                "the narrowed view must actually carry an offset"
            );
            view
        };
        // A kernel output is written at offset 0, so the only way to hand the
        // NEXT kernel an offset one is to pad it and narrow the padding off.
        let repad = |t: &Tensor| -> Tensor {
            let cols = t.dim(1).unwrap();
            let padded = Tensor::cat(
                &[&Tensor::ones((skip, cols), DType::F32, &device).unwrap(), t],
                0,
            )
            .unwrap();
            let view = padded.narrow(0, skip, n).unwrap();
            assert_ne!(view.layout().start_offset(), 0);
            view
        };

        let run = |w: &ShexpWeights, offsets: bool| {
            let (x, gate_inp, routed, rw) = if offsets {
                (
                    viewed(&x_v, &[n, hidden], 0xD1),
                    viewed(&w.gate_inp_v, &[hidden, 1], 0xD2),
                    viewed(&routed_v, &[n, top_k, hidden], 0xD3),
                    viewed(&rw_v, &[n, top_k], 0xD4),
                )
            } else {
                (
                    contiguous(&x_v, &[n, hidden]),
                    contiguous(&w.gate_inp_v, &[hidden, 1]),
                    contiguous(&routed_v, &[n, top_k, hidden]),
                    contiguous(&rw_v, &[n, top_k]),
                )
            };
            let (h, logit) =
                moe_shexp_gate_up(&x, &w.gate, &w.up, &gate_inp, hidden, inner).unwrap();
            let (h, logit) = if offsets {
                (repad(&h), repad(&logit))
            } else {
                (h, logit)
            };
            let dst = moe_epilogue_shexp(&routed, &rw, &h, &w.down, &logit, inner).unwrap();
            (dst, logit)
        };

        let (v_dst, v_logit) = run(&offset, true);
        let (c_dst, c_logit) = run(&flat, false);
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
