//! Host side of the fused hyper-connection ops — the qwen4exp residual
//! carrier's read gate (grouped norm + injection head, bottleneck activation,
//! stream mix) and its write-back. Kernel-side rounding contracts live in
//! hc.metal: the activation and the write-back are bit-identical to the candle
//! chains they replace, while the norm and the mix are bounded — each
//! partitions across threads a reduction the reference runs in one order.
//!
//! The two Q8_0 bottleneck matmuls stay on `QLinear`/`QMatMul` for the
//! four-kernel read gate — the up projection's raw pre-sigmoid logits feeding
//! [`hc_mix`] are that matmul's output. The FUSED DECODE GATE
//! ([`hc_gate_down`], [`hc_gate_up_mix`], `XWEN_HC_GATE_CLASSIC`) swallows them
//! too, reading the q8_0 blocks directly, for the small token counts where the
//! seven launch latencies cost more than the bytes do.
//!
//! `src/qwen4exp/hc.rs` is the caller and `src/qwen4exp/ref_hc.rs` the frozen
//! CPU oracle both paths grade against; the kill-switch back to the candle
//! chains is `XWEN_HC_CLASSIC`.

use anyhow::Result;
use candle_core::Tensor;
use candle_core::quantized::GgmlDType;

use crate::gguf::QuantPlane;
use crate::ops::dispatch;

/// Whether the fused read kernels cover this gate's geometry. The bounds are
/// the kernel's — at most [`dispatch::HC_MAX_STREAMS`] streams, and a `hidden`
/// some multiple of the 32-wide simdgroup up to 256 divides — so a gate outside
/// them keeps the candle chain rather than failing.
pub fn hc_norm_supported(hc_count: usize, hidden: usize) -> bool {
    dispatch::hc_norm_supported(hc_count, hidden)
}

/// The carrier's grouped RMS norm — statistics per stream, weight over the FULL
/// `hc_count * hidden` width — plus, when `inject_w` is given, the injection
/// head.
///
/// `stream` is the raw carrier `[n, hc_count * hidden]` f32, `norm_w` the
/// `[hc_count * hidden]` multiply-ready norm weight and `inject_w` the dense
/// `[hc_count, hc_count * hidden]` head (`None` on the tail mixer, which has
/// none). Returns the normed carrier and, with a head, the per-stream write
/// strengths `2·sigmoid((I·n)/hc_count)` as `[n, hc_count]`.
///
/// TWO launch shapes over the same arithmetic, picked by token count
/// ([`crate::ops::hc_split_max_n`]). At prefill one threadgroup takes a whole
/// token, folding the `hc_count` sum-of-squares reductions and the head's
/// `hc_count` full-row dot products together, so the carrier is read once for a
/// step the candle chain spends a dozen dispatches on. Below the threshold —
/// decode, where that one threadgroup is the whole launch — the work splits
/// over an `n * hc_count` grid instead, and the head becomes its own dispatch.
/// The two agree BIT-FOR-BIT (`split_matches_single_bitwise`).
pub fn hc_norm(
    stream: &Tensor,
    norm_w: &Tensor,
    inject_w: Option<&Tensor>,
    hc_count: usize,
    hidden: usize,
    eps: f32,
) -> Result<(Tensor, Option<Tensor>)> {
    dispatch::run_hc_norm(stream, norm_w, inject_w, hc_count, hidden, eps)
}

/// The bottleneck activation `silu(x / hc_count)`, one dispatch in place of
/// candle's affine + silu pair. The `1/hc_count` scale is on the ACTIVATION,
/// before the silu: the carrier is a sum over `hc_count` streams, so the down
/// projection's output grows with the stream count. Bit-identical to the pair
/// it replaces (`silu_quarter_matches_candle_bitwise` proves it).
pub fn hc_silu_quarter(x: &Tensor, hc_count: usize) -> Result<Tensor> {
    dispatch::run_hc_silu_quarter(x, hc_count)
}

/// The mix and collapse: `mixed[j] = mean_s sigmoid(up[s·hidden+j]) ·
/// normed[s·hidden+j]`, a MEAN over streams and not a sum. `up` is the up
/// projection's RAW pre-sigmoid logits `[n, hc_count · hidden]` — the sigmoid is
/// folded in here rather than costing its own full-width pass — and `normed` the
/// normed carrier. Returns `[n, hidden]`, the vector the block runs on.
pub fn hc_mix(up: &Tensor, normed: &Tensor, hc_count: usize, hidden: usize) -> Result<Tensor> {
    dispatch::run_hc_mix(up, normed, hc_count, hidden)
}

/// Whether the FUSED DECODE GATE covers this gate's geometry and bottleneck
/// dtype — the two kernels that swallow the norm, the head, both q8_0
/// projections and all the glue between them into three dispatches per gate.
/// The bounds are the kernels' (q8_0 weights, a power-of-two stream count, a
/// carrier that tiles the fixed thread partition, a bottleneck that fits
/// threadgroup memory), so a gate outside them keeps the seven-dispatch split
/// path rather than failing. See [`hc_gate_down`] for the split.
pub fn hc_gate_fused_supported(
    hc_count: usize,
    hidden: usize,
    low_rank: usize,
    dtype: GgmlDType,
) -> bool {
    dispatch::hc_gate_fused_supported(hc_count, hidden, low_rank, dtype)
}

/// First half of the fused decode gate: the grouped norm, the injection head,
/// the q8_0 down projection and `silu(·/hc_count)`, in ONE dispatch.
///
/// `stream` is the raw carrier `[n, hc_count · hidden]` f32, `norm_w` the
/// full-width multiply-ready norm weight, `inject_w` the dense
/// `[hc_count, hc_count · hidden]` head (`None` on the tail mixer) and `down`
/// the `[low_rank, hc_count · hidden]` q8_0 projection's raw device bytes.
/// Returns the bottleneck activation `[n, low_rank]`, the write strengths
/// `[n, hc_count]` when there is a head, and the per-stream scales
/// `[n, hc_count]` — which [`hc_gate_up_mix`] needs, because neither kernel
/// materializes the normed carrier the split path passes between its launches.
///
/// BOUNDED against that path, not bitwise: the down rows fold per-thread
/// partials through `simd_sum` where the gemv folds its own partition, and the
/// per-stream statistics are partitioned per q8_0 block rather than per strided
/// slice. `XWEN_HC_GATE_CLASSIC` restores it.
#[allow(clippy::too_many_arguments)]
pub fn hc_gate_down(
    stream: &Tensor,
    norm_w: &Tensor,
    inject_w: Option<&Tensor>,
    down: &QuantPlane,
    hc_count: usize,
    hidden: usize,
    low_rank: usize,
    eps: f32,
) -> Result<(Tensor, Option<Tensor>, Tensor)> {
    dispatch::run_hc_gate_down(
        stream, norm_w, inject_w, down, hc_count, hidden, low_rank, eps,
    )
}

/// Second half of the fused decode gate: the q8_0 up projection, its sigmoid and
/// the mix-and-collapse, in ONE dispatch. `low` and `scales` are
/// [`hc_gate_down`]'s outputs; `stream` and `norm_w` are what the normed carrier
/// is rebuilt from, one element per thread. Returns `[n, hidden]`.
///
/// BOUNDED against `hc_mix`: the `hc_count`-term mean runs as a
/// `simd_shuffle_xor` butterfly over adjacent lanes where that kernel runs a
/// serial loop.
#[allow(clippy::too_many_arguments)]
pub fn hc_gate_up_mix(
    low: &Tensor,
    up: &QuantPlane,
    stream: &Tensor,
    norm_w: &Tensor,
    scales: &Tensor,
    hc_count: usize,
    hidden: usize,
    low_rank: usize,
) -> Result<Tensor> {
    dispatch::run_hc_gate_up_mix(low, up, stream, norm_w, scales, hc_count, hidden, low_rank)
}

/// The write-back onto the RAW carrier: `new[s·hidden+j] = stream[s·hidden+j] +
/// block_out[j] · inject[s]`, one pass, OUT OF PLACE (the caller's carrier is
/// never mutated, so a tap or a snapshot holding it stays valid). `stream` is
/// `[n, hc_count · hidden]`, `block_out` `[n, hidden]`, `inject` `[n,
/// hc_count]`. Bit-identical to the candle broadcast-multiply-then-add chain
/// (`write_matches_candle_bitwise` proves it).
pub fn hc_write(stream: &Tensor, block_out: &Tensor, inject: &Tensor) -> Result<Tensor> {
    dispatch::run_hc_write(stream, block_out, inject)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::ops::dispatch::testutil::{pseudo_random, rel_l2};
    use crate::qwen4exp::ref_hc::{GatedResidualRef, grouped_rms_norm_batch};
    use candle_core::quantized::{QStorage, QTensor};
    use candle_core::{DType, Device, Module, Tensor};
    use candle_nn::ops::{sigmoid, silu};

    /// The checkpoint's real hyper-connection geometry (Flash-Next): four
    /// streams of 2560, a 320-wide bottleneck, `rms_norm_eps` 1e-6.
    const HC: usize = 4;
    const HIDDEN: usize = 2560;
    const LOW_RANK: usize = 320;
    const EPS: f32 = 1e-6;

    fn t2(v: Vec<f32>, rows: usize, cols: usize, dev: &Device) -> Tensor {
        Tensor::from_vec(v, (rows, cols), &Device::Cpu)
            .unwrap()
            .to_device(dev)
            .unwrap()
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn assert_f32_bits_eq(got: &Tensor, want: &Tensor, what: &str) {
        assert_eq!(got.dims(), want.dims(), "{what}: shape");
        let g = flat(got);
        let w = flat(want);
        for (i, (a, b)) in g.iter().zip(w.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{what}: element {i} differs (fused {a:?} bits {:#010x}, candle {b:?} bits \
                 {:#010x})",
                a.to_bits(),
                b.to_bits(),
            );
        }
    }

    fn assert_rel_l2(got: &Tensor, want: &[f32], tol: f32, what: &str) {
        let g = flat(got);
        let e = rel_l2(&g, want);
        assert!(e <= tol, "{what}: rel_l2 {e} > {tol}");
    }

    /// The grouped norm exactly as `HcRead::grouped_norm` runs it on the classic
    /// path. Kept a literal transcription rather than a call into qwen4exp so an
    /// edit to the production chain shows up here as a failure instead of
    /// silently moving the target.
    fn candle_grouped_norm(
        stream: &Tensor,
        norm_w: &Tensor,
        hc_count: usize,
        hidden: usize,
    ) -> Tensor {
        let (n, width) = stream.dims2().unwrap();
        let grouped = stream.reshape((n, hc_count, hidden)).unwrap();
        let inv_rms = grouped
            .sqr()
            .unwrap()
            .mean_keepdim(2)
            .unwrap()
            .affine(1.0, EPS as f64)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        grouped
            .broadcast_mul(&inv_rms)
            .unwrap()
            .reshape((n, width))
            .unwrap()
            .broadcast_mul(norm_w)
            .unwrap()
    }

    /// The classic injection head: `2·sigmoid((I·n)/hc_count)`.
    fn candle_inject(normed: &Tensor, inject_w: &Tensor, hc_count: usize) -> Tensor {
        let logits = normed
            .matmul(&inject_w.t().unwrap().contiguous().unwrap())
            .unwrap();
        sigmoid(&logits.affine(1.0 / hc_count as f64, 0.0).unwrap())
            .unwrap()
            .affine(2.0, 0.0)
            .unwrap()
    }

    /// The classic mix and collapse.
    fn candle_mix(up: &Tensor, normed: &Tensor, hc_count: usize, hidden: usize) -> Tensor {
        let (n, _) = up.dims2().unwrap();
        (sigmoid(up).unwrap() * normed)
            .unwrap()
            .reshape((n, hc_count, hidden))
            .unwrap()
            .sum(1)
            .unwrap()
            .affine(1.0 / hc_count as f64, 0.0)
            .unwrap()
    }

    /// The classic write-back: broadcast multiply, then add onto the raw
    /// carrier. Literal transcription of `qwen4exp::hc::hc_write`.
    fn candle_write(stream: &Tensor, block_out: &Tensor, inject: &Tensor) -> Tensor {
        let (n, width) = stream.dims2().unwrap();
        let (_, hidden) = block_out.dims2().unwrap();
        let (_, hc_count) = inject.dims2().unwrap();
        let scaled = block_out
            .reshape((n, 1, hidden))
            .unwrap()
            .broadcast_mul(&inject.reshape((n, hc_count, 1)).unwrap())
            .unwrap();
        (stream.reshape((n, hc_count, hidden)).unwrap() + scaled)
            .unwrap()
            .reshape((n, width))
            .unwrap()
    }

    /// The write-back kernel must reproduce the candle chain BIT-FOR-BIT
    /// (compared on `f32::to_bits`, not a tolerance): it is a multiply and an
    /// add in a fixed order, and nothing about it reassociates. Bit-identity is
    /// the whole justification for shipping it under the strict parity tier —
    /// never loosen this to a tolerance.
    #[test]
    fn write_matches_candle_bitwise() {
        let dev = metal_device().unwrap();
        // Decode (1 token) and prefill shapes, at the real geometry and at a
        // small one whose width is not a multiple of the launch width.
        for &(n, hc_count, hidden) in &[(1usize, HC, HIDDEN), (64, HC, HIDDEN), (3, 3, 33)] {
            let width = hc_count * hidden;
            let stream = t2(pseudo_random(n * width, 0x51, -3.0, 3.0), n, width, &dev);
            let block_out = t2(pseudo_random(n * hidden, 0x52, -3.0, 3.0), n, hidden, &dev);
            let inject = t2(
                pseudo_random(n * hc_count, 0x53, 0.0, 2.0),
                n,
                hc_count,
                &dev,
            );

            let fused = hc_write(&stream, &block_out, &inject).unwrap();
            assert_eq!(fused.dims(), &[n, width]);
            let want = candle_write(&stream, &block_out, &inject);
            assert_f32_bits_eq(&fused, &want, &format!("hc_write n={n} hidden={hidden}"));
        }
    }

    /// A zero block output must leave the carrier bit-identical whatever the
    /// injection weights say — the write is purely additive, and the kernel adds
    /// an exact zero rather than scaling the carrier.
    #[test]
    fn write_of_zero_is_identity() {
        let dev = metal_device().unwrap();
        let (n, width) = (5usize, HC * HIDDEN);
        let stream = t2(pseudo_random(n * width, 0x61, -3.0, 3.0), n, width, &dev);
        let zeros = Tensor::zeros((n, HIDDEN), DType::F32, &dev).unwrap();
        let inject = t2(pseudo_random(n * HC, 0x62, 0.0, 2.0), n, HC, &dev);
        let out = hc_write(&stream, &zeros, &inject).unwrap();
        assert_f32_bits_eq(&out, &stream, "hc_write with a zero block output");
    }

    /// The bottleneck activation must reproduce candle's `affine(1/hc, 0)` +
    /// `silu` pair BIT-FOR-BIT across a wide magnitude span, where `exp(-x)`
    /// saturates at both ends.
    #[test]
    fn silu_quarter_matches_candle_bitwise() {
        let dev = metal_device().unwrap();
        for &hc_count in &[HC, 3] {
            for &(n, low_rank) in &[(1usize, LOW_RANK), (64, LOW_RANK), (7, 13)] {
                let v: Vec<f32> = pseudo_random(n * low_rank, 0x71 + hc_count as u64, -30.0, 30.0);
                let x = t2(v, n, low_rank, &dev);
                let fused = hc_silu_quarter(&x, hc_count).unwrap();
                let want = silu(&x.affine(1.0 / hc_count as f64, 0.0).unwrap()).unwrap();
                assert_f32_bits_eq(
                    &fused,
                    &want,
                    &format!("hc_silu_quarter hc={hc_count} n={n} low_rank={low_rank}"),
                );
            }
        }
    }

    /// The norm + injection head (K1) and the mix (K3) at the checkpoint's real
    /// geometry, graded both against the candle chain they replace and against
    /// the frozen CPU oracle.
    ///
    /// Both kernels partition reductions the classic path runs in one order — a
    /// 2560-term sum of squares per stream, a 10240-term dot per injection row —
    /// so they are bounded, not bitwise. 1e-6 against candle is the same floor
    /// the delta kernels' partitioned reductions hold; 1e-5 against `ref_hc` is
    /// the tolerance the classic path itself is graded at.
    #[test]
    fn norm_and_mix_match_candle_and_reference() {
        let dev = metal_device().unwrap();
        let width = HC * HIDDEN;
        // Norm weights sit near 1 (the converter's baked `1 + w`), everything
        // else is zero-centered.
        let norm_w_v: Vec<f32> = pseudo_random(width, 0x81, -0.5, 0.5)
            .iter()
            .map(|v| 1.0 + 0.1 * v)
            .collect();
        let down_w_v = pseudo_random(LOW_RANK * width, 0x82, -0.03, 0.03);
        let up_w_v = pseudo_random(width * LOW_RANK, 0x83, -0.05, 0.05);
        let inject_w_v = pseudo_random(HC * width, 0x84, -0.02, 0.02);

        let norm_w = Tensor::from_vec(norm_w_v.clone(), width, &Device::Cpu)
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let down_w = t2(down_w_v.clone(), LOW_RANK, width, &dev);
        let up_w = t2(up_w_v.clone(), width, LOW_RANK, &dev);
        let inject_w = t2(inject_w_v.clone(), HC, width, &dev);
        let down_t = down_w.t().unwrap().contiguous().unwrap();
        let up_t = up_w.t().unwrap().contiguous().unwrap();

        let reference = GatedResidualRef {
            hc_count: HC,
            hidden: HIDDEN,
            low_rank: LOW_RANK,
            norm_w: norm_w_v.clone(),
            down_w: down_w_v,
            up_w: up_w_v,
            inject_w: inject_w_v,
        };

        // One token count per LAUNCH ARM, so both are graded against candle and
        // the oracle here: n=5 is below HC_SPLIT_MAX_N and takes the split pair
        // (one threadgroup per token AND stream, plus a separate injection
        // kernel), n=64 takes the fused single-threadgroup-per-token kernel.
        for &n in &[5usize, 64] {
            let stream_v = pseudo_random(n * width, 0x85 + n as u64, -2.0, 2.0);
            let stream = t2(stream_v.clone(), n, width, &dev);

            let (normed, inject) =
                hc_norm(&stream, &norm_w, Some(&inject_w), HC, HIDDEN, EPS).unwrap();
            let inject = inject.expect("a gate with an injection head returns one");
            assert_eq!(normed.dims(), &[n, width]);
            assert_eq!(inject.dims(), &[n, HC]);

            // The bottleneck matmuls are the same dense f32 products on both
            // sides; only K2 and K3 differ from the classic chain here.
            let low = hc_silu_quarter(&normed.matmul(&down_t).unwrap(), HC).unwrap();
            let up = low.matmul(&up_t).unwrap();
            let mixed = hc_mix(&up, &normed, HC, HIDDEN).unwrap();
            assert_eq!(mixed.dims(), &[n, HIDDEN]);

            // --- against the candle chain.
            let c_normed = candle_grouped_norm(&stream, &norm_w, HC, HIDDEN);
            assert_rel_l2(
                &normed,
                &flat(&c_normed),
                1e-6,
                &format!("normed vs candle n={n}"),
            );
            let c_inject = candle_inject(&c_normed, &inject_w, HC);
            assert_rel_l2(
                &inject,
                &flat(&c_inject),
                1e-6,
                &format!("inject vs candle n={n}"),
            );
            let c_low =
                silu(&c_normed.matmul(&down_t).unwrap().affine(0.25, 0.0).unwrap()).unwrap();
            let c_up = c_low.matmul(&up_t).unwrap();
            let c_mixed = candle_mix(&c_up, &c_normed, HC, HIDDEN);
            assert_rel_l2(
                &mixed,
                &flat(&c_mixed),
                1e-6,
                &format!("mixed vs candle n={n}"),
            );

            // --- against the frozen oracle.
            let r_normed = grouped_rms_norm_batch(&stream_v, &norm_w_v, HIDDEN, EPS);
            assert_rel_l2(&normed, &r_normed, 1e-5, &format!("normed vs ref_hc n={n}"));
            let (r_mixed, r_inject) = reference.read_batch(&stream_v, EPS);
            assert_rel_l2(&inject, &r_inject, 1e-5, &format!("inject vs ref_hc n={n}"));
            assert_rel_l2(&mixed, &r_mixed, 1e-5, &format!("mixed vs ref_hc n={n}"));
        }
    }

    /// K1 with no injection head — the tail mixer's shape — must produce the
    /// same normed carrier as the head-bearing arm and no injection tensor. The
    /// head is the only difference between the two gates.
    ///
    /// Both arms are PINNED to the fused kernel, so this is what covers its
    /// `HAS_INJECT=false` specialization: leaving the launch shape to the token
    /// count would put a small `n` on the split pair and never compile the
    /// headless fused variant. The split pair's own head/no-head equality is
    /// asserted in `split_matches_single_bitwise`.
    #[test]
    fn norm_without_inject_matches_the_inject_arm() {
        let dev = metal_device().unwrap();
        let width = HC * HIDDEN;
        let n = 3usize;
        let norm_w = Tensor::from_vec(pseudo_random(width, 0x91, 0.9, 1.1), width, &Device::Cpu)
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let inject_w = t2(
            pseudo_random(HC * width, 0x92, -0.02, 0.02),
            HC,
            width,
            &dev,
        );
        let stream = t2(pseudo_random(n * width, 0x93, -2.0, 2.0), n, width, &dev);

        let (with, inject) = hc_norm_arm(false, &stream, &norm_w, Some(&inject_w), HC, HIDDEN);
        assert!(inject.is_some());
        let (without, none) = hc_norm_arm(false, &stream, &norm_w, None, HC, HIDDEN);
        assert!(none.is_none(), "the tail mixer has no injection head");
        assert_f32_bits_eq(&without, &with, "normed carrier, head vs no head");
    }

    /// Every kernel must honour a tensor whose storage starts at a nonzero
    /// offset — a row-slice of a larger allocation, which candle hands out
    /// without materializing. The dispatch binds each operand at its own
    /// `f32_off`, so a slice must produce exactly the bits the materialized copy
    /// does.
    #[test]
    fn offset_views_read_the_right_rows() {
        let dev = metal_device().unwrap();
        let (hc_count, hidden, n, skip) = (4usize, 64usize, 5usize, 2usize);
        let width = hc_count * hidden;
        let rows = n + skip + 1;

        let big_stream = t2(
            pseudo_random(rows * width, 0xA1, -2.0, 2.0),
            rows,
            width,
            &dev,
        );
        let big_block = t2(
            pseudo_random(rows * hidden, 0xA2, -2.0, 2.0),
            rows,
            hidden,
            &dev,
        );
        let big_inject = t2(
            pseudo_random(rows * hc_count, 0xA3, 0.0, 2.0),
            rows,
            hc_count,
            &dev,
        );
        let norm_w = Tensor::from_vec(pseudo_random(width, 0xA4, 0.9, 1.1), width, &Device::Cpu)
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let inject_w = t2(
            pseudo_random(hc_count * width, 0xA5, -0.05, 0.05),
            hc_count,
            width,
            &dev,
        );

        // A dim-0 narrow is contiguous with a nonzero start offset; the copies
        // are the same rows at offset zero.
        let view_stream = big_stream.narrow(0, skip, n).unwrap();
        let view_block = big_block.narrow(0, skip, n).unwrap();
        let view_inject = big_inject.narrow(0, skip, n).unwrap();
        let copy_stream = t2(flat(&view_stream), n, width, &dev);
        let copy_block = t2(flat(&view_block), n, hidden, &dev);
        let copy_inject = t2(flat(&view_inject), n, hc_count, &dev);
        assert_ne!(
            view_stream.layout().start_offset(),
            0,
            "the narrowed view must actually carry an offset"
        );

        let (v_normed, v_inj) = hc_norm(
            &view_stream,
            &norm_w,
            Some(&inject_w),
            hc_count,
            hidden,
            EPS,
        )
        .unwrap();
        let (c_normed, c_inj) = hc_norm(
            &copy_stream,
            &norm_w,
            Some(&inject_w),
            hc_count,
            hidden,
            EPS,
        )
        .unwrap();
        assert_f32_bits_eq(&v_normed, &c_normed, "hc_norm over an offset carrier");
        assert_f32_bits_eq(
            &v_inj.unwrap(),
            &c_inj.unwrap(),
            "hc_norm inject over an offset carrier",
        );

        let v_silu = hc_silu_quarter(&view_block, hc_count).unwrap();
        let c_silu = hc_silu_quarter(&copy_block, hc_count).unwrap();
        assert_f32_bits_eq(&v_silu, &c_silu, "hc_silu_quarter over an offset input");

        // The mix reads two full-width operands; feed the offset carrier as the
        // logits so both bindings are exercised.
        let v_mixed = hc_mix(&view_stream, &v_normed, hc_count, hidden).unwrap();
        let c_mixed = hc_mix(&copy_stream, &c_normed, hc_count, hidden).unwrap();
        assert_f32_bits_eq(&v_mixed, &c_mixed, "hc_mix over an offset carrier");

        let v_written = hc_write(&view_stream, &view_block, &view_inject).unwrap();
        let c_written = hc_write(&copy_stream, &copy_block, &copy_inject).unwrap();
        assert_f32_bits_eq(&v_written, &c_written, "hc_write over offset operands");
    }

    /// Geometry and shape the kernels do not cover must be REFUSED, not
    /// silently mis-dispatched: an empty batch encodes a zero-dimension grid, a
    /// `hidden` the launch width cannot divide would leave a stream's reduction
    /// split across threadgroups, and a stream count past the kernel's
    /// accumulator array would read past it.
    #[test]
    fn degenerate_geometry_is_refused() {
        let dev = metal_device().unwrap();
        let z2 = |r: usize, c: usize| Tensor::zeros((r, c), DType::F32, &dev).unwrap();
        let z1 = |c: usize| Tensor::zeros(c, DType::F32, &dev).unwrap();

        // hc_norm: hidden must tile whole simdgroups.
        assert!(hc_norm(&z2(2, 4 * 40), &z1(4 * 40), None, 4, 40, EPS).is_err());
        assert!(!hc_norm_supported(4, 40));
        assert!(hc_norm_supported(4, HIDDEN));
        // ... and hidden 0 has no launch width at all (every width divides
        // zero, so the divisor search has to reject it explicitly).
        assert!(!hc_norm_supported(4, 0));
        // More streams than the kernel's accumulator array holds.
        assert!(hc_norm(&z2(2, 9 * 64), &z1(9 * 64), None, 9, 64, EPS).is_err());
        assert!(!hc_norm_supported(9, 64));
        assert!(!hc_norm_supported(0, 64));
        // An empty batch dispatches nothing and hands back a shape nobody asked
        // for.
        assert!(hc_norm(&z2(0, 4 * 64), &z1(4 * 64), None, 4, 64, EPS).is_err());
        // Carrier width, norm-weight width and injection-head shape must all
        // agree with the declared geometry.
        assert!(hc_norm(&z2(2, 4 * 64 - 1), &z1(4 * 64), None, 4, 64, EPS).is_err());
        assert!(hc_norm(&z2(2, 4 * 64), &z1(64), None, 4, 64, EPS).is_err());
        assert!(
            hc_norm(
                &z2(2, 4 * 64),
                &z1(4 * 64),
                Some(&z2(3, 4 * 64)),
                4,
                64,
                EPS
            )
            .is_err()
        );
        assert!(hc_norm(&z2(2, 4 * 64), &z1(4 * 64), Some(&z2(4, 64)), 4, 64, EPS).is_err());
        // f16 operands are not the carrier's dtype.
        let half = Tensor::zeros((2, 4 * 64), DType::F16, &dev).unwrap();
        assert!(hc_norm(&half, &z1(4 * 64), None, 4, 64, EPS).is_err());

        // silu_quarter: an empty input, a zero stream count, a non-f32 operand.
        assert!(hc_silu_quarter(&z2(0, 8), 4).is_err());
        assert!(hc_silu_quarter(&z2(2, 8), 0).is_err());
        assert!(hc_silu_quarter(&half, 4).is_err());

        // mix: the two operands must be the same full-width shape.
        assert!(hc_mix(&z2(2, 4 * 64), &z2(2, 4 * 64 - 1), 4, 64).is_err());
        assert!(hc_mix(&z2(2, 4 * 64), &z2(3, 4 * 64), 4, 64).is_err());
        assert!(hc_mix(&z2(0, 4 * 64), &z2(0, 4 * 64), 4, 64).is_err());
        assert!(hc_mix(&z2(2, 4 * 64), &z2(2, 4 * 64), 0, 64).is_err());

        // write: row counts, the width identity, and an empty batch.
        assert!(hc_write(&z2(2, 4 * 64), &z2(3, 64), &z2(2, 4)).is_err());
        assert!(hc_write(&z2(2, 4 * 64), &z2(2, 64), &z2(2, 3)).is_err());
        assert!(hc_write(&z2(2, 4 * 64), &z2(2, 63), &z2(2, 4)).is_err());
        assert!(hc_write(&z2(0, 4 * 64), &z2(0, 64), &z2(0, 4)).is_err());
    }

    /// [`hc_norm`] with the launch shape pinned rather than picked by token
    /// count, so a test can put both arms on the same input.
    fn hc_norm_arm(
        split: bool,
        stream: &Tensor,
        norm_w: &Tensor,
        inject_w: Option<&Tensor>,
        hc_count: usize,
        hidden: usize,
    ) -> (Tensor, Option<Tensor>) {
        dispatch::run_hc_norm_with(stream, norm_w, inject_w, hc_count, hidden, EPS, Some(split))
            .unwrap()
    }

    /// The split launch (one threadgroup per token AND stream, plus a separate
    /// injection kernel) must produce EXACTLY the single-threadgroup kernel's
    /// bits — not merely a close answer. Both arms give a thread the same
    /// strided slice of the same stream and fold the same simd_sum and the same
    /// serial pass over the simdgroup partials, so no reduction reassociates
    /// between them; the injection dot deliberately walks the carrier
    /// stream-major for the same reason. That identity is what lets the host
    /// pick an arm by token count without the choice being visible in a result,
    /// so a divergence here is a bug in the split kernels, never a tolerance to
    /// relax.
    #[test]
    fn split_matches_single_bitwise() {
        let dev = metal_device().unwrap();
        let width = HC * HIDDEN;
        let norm_w = Tensor::from_vec(pseudo_random(width, 0xB1, 0.9, 1.1), width, &Device::Cpu)
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let inject_w = t2(
            pseudo_random(HC * width, 0xB2, -0.02, 0.02),
            HC,
            width,
            &dev,
        );

        // A decode step, a small speculative batch, and the token right below
        // the shipped threshold.
        for &n in &[1usize, 5, 31] {
            let stream = t2(
                pseudo_random(n * width, 0xB3 + n as u64, -2.0, 2.0),
                n,
                width,
                &dev,
            );

            let (s_normed, s_inject) =
                hc_norm_arm(true, &stream, &norm_w, Some(&inject_w), HC, HIDDEN);
            let (f_normed, f_inject) =
                hc_norm_arm(false, &stream, &norm_w, Some(&inject_w), HC, HIDDEN);
            assert_eq!(s_normed.dims(), &[n, width]);
            assert_f32_bits_eq(&s_normed, &f_normed, &format!("split normed n={n}"));
            assert_f32_bits_eq(
                &s_inject.unwrap(),
                &f_inject.unwrap(),
                &format!("split inject n={n}"),
            );

            // The headless (tail mixer) arm takes the same split norm kernel
            // with no second launch behind it.
            let (s_bare, none) = hc_norm_arm(true, &stream, &norm_w, None, HC, HIDDEN);
            assert!(none.is_none(), "the tail mixer has no injection head");
            let (f_bare, _) = hc_norm_arm(false, &stream, &norm_w, None, HC, HIDDEN);
            assert_f32_bits_eq(&s_bare, &f_bare, &format!("split normed, no head, n={n}"));

            // Dropping the head changes only whether an injection comes back —
            // the normed carrier is the same bits either way, on BOTH arms.
            assert_f32_bits_eq(
                &f_normed,
                &f_bare,
                &format!("single normed, head vs no head, n={n}"),
            );
            assert_f32_bits_eq(
                &s_normed,
                &s_bare,
                &format!("split normed, head vs no head, n={n}"),
            );
        }
    }

    /// The split arm graded against the frozen CPU oracle, not just against the
    /// other kernel: two arms agreeing bit-for-bit would still both be wrong if
    /// the split kernels indexed the full-width norm weight or the injection
    /// head's rows by the stream slice instead of the carrier.
    #[test]
    fn split_matches_reference() {
        let dev = metal_device().unwrap();
        let width = HC * HIDDEN;
        let norm_w_v: Vec<f32> = pseudo_random(width, 0xC1, -0.5, 0.5)
            .iter()
            .map(|v| 1.0 + 0.1 * v)
            .collect();
        let down_w_v = pseudo_random(LOW_RANK * width, 0xC2, -0.03, 0.03);
        let up_w_v = pseudo_random(width * LOW_RANK, 0xC3, -0.05, 0.05);
        let inject_w_v = pseudo_random(HC * width, 0xC4, -0.02, 0.02);

        let norm_w = Tensor::from_vec(norm_w_v.clone(), width, &Device::Cpu)
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let down_t = t2(down_w_v.clone(), LOW_RANK, width, &dev)
            .t()
            .unwrap()
            .contiguous()
            .unwrap();
        let up_t = t2(up_w_v.clone(), width, LOW_RANK, &dev)
            .t()
            .unwrap()
            .contiguous()
            .unwrap();
        let inject_w = t2(inject_w_v.clone(), HC, width, &dev);

        let reference = GatedResidualRef {
            hc_count: HC,
            hidden: HIDDEN,
            low_rank: LOW_RANK,
            norm_w: norm_w_v.clone(),
            down_w: down_w_v,
            up_w: up_w_v,
            inject_w: inject_w_v,
        };

        for &n in &[1usize, 5] {
            let stream_v = pseudo_random(n * width, 0xC5 + n as u64, -2.0, 2.0);
            let stream = t2(stream_v.clone(), n, width, &dev);

            let (normed, inject) = hc_norm_arm(true, &stream, &norm_w, Some(&inject_w), HC, HIDDEN);
            let inject = inject.expect("a gate with an injection head returns one");
            let low = hc_silu_quarter(&normed.matmul(&down_t).unwrap(), HC).unwrap();
            let mixed = hc_mix(&low.matmul(&up_t).unwrap(), &normed, HC, HIDDEN).unwrap();

            let r_normed = grouped_rms_norm_batch(&stream_v, &norm_w_v, HIDDEN, EPS);
            assert_rel_l2(&normed, &r_normed, 1e-5, &format!("split normed n={n}"));
            let (r_mixed, r_inject) = reference.read_batch(&stream_v, EPS);
            assert_rel_l2(&inject, &r_inject, 1e-5, &format!("split inject n={n}"));
            assert_rel_l2(&mixed, &r_mixed, 1e-5, &format!("split mixed n={n}"));
        }
    }

    /// The split launch must honour a carrier that starts at a nonzero storage
    /// offset, the same way the single-threadgroup kernel does. Its norm
    /// dispatch binds the carrier at `f32_off` while its injection dispatch
    /// reads the freshly written `normed` at offset zero, so the two launches
    /// disagree about offsets if either binding is wrong.
    #[test]
    fn split_honours_offset_views() {
        let dev = metal_device().unwrap();
        let (hc_count, hidden, n, skip) = (4usize, 64usize, 3usize, 2usize);
        let width = hc_count * hidden;
        let rows = n + skip + 1;

        let big = t2(
            pseudo_random(rows * width, 0xD1, -2.0, 2.0),
            rows,
            width,
            &dev,
        );
        let norm_w = Tensor::from_vec(pseudo_random(width, 0xD2, 0.9, 1.1), width, &Device::Cpu)
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let inject_w = t2(
            pseudo_random(hc_count * width, 0xD3, -0.05, 0.05),
            hc_count,
            width,
            &dev,
        );

        let view = big.narrow(0, skip, n).unwrap();
        assert_ne!(
            view.layout().start_offset(),
            0,
            "the narrowed view must actually carry an offset"
        );
        let copy = t2(flat(&view), n, width, &dev);

        let (v_normed, v_inject) =
            hc_norm_arm(true, &view, &norm_w, Some(&inject_w), hc_count, hidden);
        let (c_normed, c_inject) =
            hc_norm_arm(true, &copy, &norm_w, Some(&inject_w), hc_count, hidden);
        let v_inject = v_inject.unwrap();
        let c_inject = c_inject.unwrap();
        assert_f32_bits_eq(&v_normed, &c_normed, "split hc_norm over an offset carrier");
        assert_f32_bits_eq(
            &v_inject,
            &c_inject,
            "split hc_inject over an offset carrier",
        );

        // ... and it must still be the single kernel's answer on that view.
        let (f_normed, f_inject) =
            hc_norm_arm(false, &view, &norm_w, Some(&inject_w), hc_count, hidden);
        assert_f32_bits_eq(
            &v_normed,
            &f_normed,
            "split vs single over an offset carrier",
        );
        assert_f32_bits_eq(
            &v_inject,
            &f_inject.unwrap(),
            "split vs single inject over an offset carrier",
        );
    }

    /// `HC_MAX_STREAMS` is spelled out in BOTH languages — a `#define` sizing
    /// the kernel's per-thread injection accumulator array, and the host bound
    /// that refuses a wider carrier. Nothing links them, and raising only the
    /// host one would let a carrier index past the array with no diagnostic, so
    /// this test is the link. (Same shape as `mm_id.rs`'s
    /// `instantiation_matrix_matches_metal`: read the Metal source, parse the
    /// value, compare.)
    #[test]
    fn hc_max_streams_matches_metal() {
        const SRC: &str = include_str!("hc.metal");
        let mut found = None;
        for line in SRC.lines() {
            if let Some(rest) = line.trim().strip_prefix("#define HC_MAX_STREAMS") {
                assert!(
                    found.is_none(),
                    "hc.metal defines HC_MAX_STREAMS more than once"
                );
                found = Some(
                    rest.trim()
                        .parse::<usize>()
                        .expect("HC_MAX_STREAMS must be a plain integer literal"),
                );
            }
        }
        let metal = found.expect("hc.metal must #define HC_MAX_STREAMS");
        assert_eq!(
            metal,
            dispatch::HC_MAX_STREAMS,
            "hc.metal's HC_MAX_STREAMS ({metal}) and dispatch.rs's ({}) must agree — the host \
             bound is what keeps a carrier inside the kernel's accumulator array",
            dispatch::HC_MAX_STREAMS,
        );
    }

    /// A `[n_out, k]` weight quantized to q8_0, returned as the raw-bytes
    /// `QuantPlane` the gate kernels read, the `QTensor` the `QMatMul` split
    /// chain reads, and the dequantized f32 values the frozen oracle needs. All
    /// three view ONE quantization of one source, so a difference between the
    /// three paths is the path's and never the quantizer's.
    fn build_q8_plane(
        dev: &Device,
        n_out: usize,
        k: usize,
        data: &[f32],
    ) -> (QuantPlane, std::sync::Arc<QTensor>, Vec<f32>) {
        let cpu = Device::Cpu;
        let dense = Tensor::from_vec(data.to_vec(), (n_out, k), &cpu).unwrap();
        let qcpu = QTensor::quantize(&dense, GgmlDType::Q8_0).unwrap();
        let deq = qcpu
            .dequantize(&cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Upload the quantized bytes ONCE and retain the buffer before the
        // storage moves into the QTensor.
        let storage = QStorage::from_data(qcpu.data().unwrap(), dev, GgmlDType::Q8_0).unwrap();
        let QStorage::Metal(qms) = &storage else {
            panic!("expected Metal quantized storage")
        };
        let buffer = std::sync::Arc::new(qms.buffer().clone());
        let qtensor = std::sync::Arc::new(QTensor::new(storage, (n_out, k)).unwrap());
        (
            QuantPlane {
                buffer,
                base_off: 0,
                dtype: GgmlDType::Q8_0,
                out_dim: n_out,
                in_dim: k,
            },
            qtensor,
            deq,
        )
    }

    /// The per-stream `1/sqrt(mean(x^2) + eps)` the grouped norm is built on,
    /// summed sequentially in f32 on the host. The kernel partitions that sum
    /// over threads, so this is a bounded reference for its `scales` output, not
    /// a bitwise one.
    fn host_scales(stream: &[f32], hc_count: usize, hidden: usize) -> Vec<f32> {
        let width = hc_count * hidden;
        let n = stream.len() / width;
        let mut out = Vec::with_capacity(n * hc_count);
        for t in 0..n {
            for s in 0..hc_count {
                let base = t * width + s * hidden;
                let mut acc = 0.0f32;
                for j in 0..hidden {
                    let v = stream[base + j];
                    acc += v * v;
                }
                out.push(1.0 / (acc / hidden as f32 + EPS).sqrt());
            }
        }
        out
    }

    /// One gate's weights at a geometry the fused decode gate covers, with the
    /// bottleneck stored q8_0 the way the checkpoint stores it.
    struct GateWeights {
        norm_w: Tensor,
        inject_w: Tensor,
        down: QuantPlane,
        up: QuantPlane,
        down_q: std::sync::Arc<QTensor>,
        up_q: std::sync::Arc<QTensor>,
        reference: GatedResidualRef,
    }

    fn gate_weights(dev: &Device, hc_count: usize, hidden: usize, low_rank: usize) -> GateWeights {
        let width = hc_count * hidden;
        let norm_w_v: Vec<f32> = pseudo_random(width, 0xF1, -0.5, 0.5)
            .iter()
            .map(|v| 1.0 + 0.1 * v)
            .collect();
        let inject_w_v = pseudo_random(hc_count * width, 0xF2, -0.02, 0.02);
        let (down, down_q, down_deq) = build_q8_plane(
            dev,
            low_rank,
            width,
            &pseudo_random(low_rank * width, 0xF3, -0.03, 0.03),
        );
        let (up, up_q, up_deq) = build_q8_plane(
            dev,
            width,
            low_rank,
            &pseudo_random(width * low_rank, 0xF4, -0.05, 0.05),
        );
        GateWeights {
            norm_w: Tensor::from_vec(norm_w_v.clone(), width, &Device::Cpu)
                .unwrap()
                .to_device(dev)
                .unwrap(),
            inject_w: t2(inject_w_v.clone(), hc_count, width, dev),
            down,
            up,
            down_q,
            up_q,
            // The oracle is frozen and takes dense f32 weights, so it gets the
            // dequantized bytes the kernels actually read.
            reference: GatedResidualRef {
                hc_count,
                hidden,
                low_rank,
                norm_w: norm_w_v,
                down_w: down_deq,
                up_w: up_deq,
                inject_w: inject_w_v,
            },
        }
    }

    /// The split path the fused gate replaces, over the SAME quantized bytes:
    /// the split norm arm, both bottleneck projections through `QMatMul`, the
    /// activation and the mix. Spelled out rather than called into qwen4exp so
    /// an edit to the production chain shows up here as a failure instead of
    /// silently moving the target.
    fn split_chain(
        w: &GateWeights,
        stream: &Tensor,
        with_inject: bool,
        hc_count: usize,
        hidden: usize,
    ) -> (Tensor, Option<Tensor>) {
        let (normed, inject) = hc_norm_arm(
            true,
            stream,
            &w.norm_w,
            with_inject.then_some(&w.inject_w),
            hc_count,
            hidden,
        );
        let mm = |qt: &std::sync::Arc<QTensor>, x: &Tensor| {
            candle_core::quantized::QMatMul::from_arc(qt.clone())
                .unwrap()
                .forward(x)
                .unwrap()
        };
        let low = hc_silu_quarter(&mm(&w.down_q, &normed), hc_count).unwrap();
        let mixed = hc_mix(&mm(&w.up_q, &low), &normed, hc_count, hidden).unwrap();
        (mixed, inject)
    }

    /// The fused decode gate at the checkpoint's real geometry, graded against
    /// BOTH the seven-dispatch split path it replaces and the frozen CPU oracle.
    ///
    /// Both dot products are reassociated against that path — the down rows fold
    /// per-thread simd_sum partials where the gemv folds its own partition, and
    /// the stream mean is a simd-shuffle butterfly where `hc_mix` runs a serial
    /// loop — so this is bounded at the tolerance the classic path is held to,
    /// never bitwise.
    #[test]
    fn gate_fused_matches_reference() {
        let dev = metal_device().unwrap();
        let width = HC * HIDDEN;
        assert!(hc_gate_fused_supported(
            HC,
            HIDDEN,
            LOW_RANK,
            GgmlDType::Q8_0
        ));
        let w = gate_weights(&dev, HC, HIDDEN, LOW_RANK);

        for &n in &[1usize, 3] {
            let stream_v = pseudo_random(n * width, 0xF5 + n as u64, -2.0, 2.0);
            let stream = t2(stream_v.clone(), n, width, &dev);

            let (low, inject, scales) = hc_gate_down(
                &stream,
                &w.norm_w,
                Some(&w.inject_w),
                &w.down,
                HC,
                HIDDEN,
                LOW_RANK,
                EPS,
            )
            .unwrap();
            let inject = inject.expect("a gate with an injection head returns one");
            let mixed = hc_gate_up_mix(
                &low, &w.up, &stream, &w.norm_w, &scales, HC, HIDDEN, LOW_RANK,
            )
            .unwrap();
            assert_eq!(mixed.dims(), &[n, HIDDEN]);
            assert_eq!(inject.dims(), &[n, HC]);
            assert_eq!(scales.dims(), &[n, HC]);

            // The scales are an interface between the two kernels, not a
            // by-product: the second rebuilds the normed carrier from them, so a
            // wrong one is a silently wrong mix rather than a failure.
            assert_rel_l2(
                &scales,
                &host_scales(&stream_v, HC, HIDDEN),
                1e-6,
                &format!("fused gate scales n={n}"),
            );

            let (c_mixed, c_inject) = split_chain(&w, &stream, true, HC, HIDDEN);
            // At one token the split path's two projections are f32 gemvs and
            // the two paths differ only by dot-product association (measured
            // 7e-8). Above one token `QMatMul` takes its matmul kernel, whose
            // activation tiles are half precision, and it is the SPLIT path
            // that moves away from the oracle (1.7e-5 at n = 3 against 1.6e-7
            // for the fused gate), so the cross-path bound is that path's own
            // class there; the oracle bound below is the one that holds the
            // fused gate.
            let split_tol = if n == 1 { 1e-6 } else { 5e-5 };
            assert_rel_l2(
                &mixed,
                &flat(&c_mixed),
                split_tol,
                &format!("fused gate mixed vs the split path n={n}"),
            );
            assert_rel_l2(
                &inject,
                &flat(&c_inject.unwrap()),
                1e-5,
                &format!("fused gate inject vs the split path n={n}"),
            );

            let (r_mixed, r_inject) = w.reference.read_batch(&stream_v, EPS);
            assert_rel_l2(
                &mixed,
                &r_mixed,
                1e-5,
                &format!("fused gate mixed vs ref_hc n={n}"),
            );
            assert_rel_l2(
                &inject,
                &r_inject,
                1e-5,
                &format!("fused gate inject vs ref_hc n={n}"),
            );
        }
    }

    /// The tail mixer's arm: no injection head, so the grid carries no head
    /// threadgroup and nothing comes back for the write-back to scale. The mix
    /// is unaffected — dropping the head changes only whether an injection is
    /// produced.
    #[test]
    fn gate_fused_tail_arm_has_no_inject() {
        let dev = metal_device().unwrap();
        let width = HC * HIDDEN;
        let w = gate_weights(&dev, HC, HIDDEN, LOW_RANK);
        let stream_v = pseudo_random(width, 0xF6, -2.0, 2.0);
        let stream = t2(stream_v.clone(), 1, width, &dev);

        let (low, inject, scales) =
            hc_gate_down(&stream, &w.norm_w, None, &w.down, HC, HIDDEN, LOW_RANK, EPS).unwrap();
        assert!(inject.is_none(), "the tail mixer has no injection head");
        let mixed = hc_gate_up_mix(
            &low, &w.up, &stream, &w.norm_w, &scales, HC, HIDDEN, LOW_RANK,
        )
        .unwrap();

        let (c_mixed, c_inject) = split_chain(&w, &stream, false, HC, HIDDEN);
        assert!(c_inject.is_none());
        assert_rel_l2(
            &mixed,
            &flat(&c_mixed),
            1e-5,
            "headless fused gate vs the split path",
        );
        // And the same answer the gated arm gives on the same carrier: the head
        // must not feed the mix.
        let (g_low, _, g_scales) = hc_gate_down(
            &stream,
            &w.norm_w,
            Some(&w.inject_w),
            &w.down,
            HC,
            HIDDEN,
            LOW_RANK,
            EPS,
        )
        .unwrap();
        assert_f32_bits_eq(&low, &g_low, "headless vs gated bottleneck activation");
        assert_f32_bits_eq(&scales, &g_scales, "headless vs gated scales");
    }

    /// Both kernels must honour operands that start at a nonzero storage offset:
    /// the carrier (a narrowed batch), and the bottleneck activation the second
    /// kernel reads back from the first.
    #[test]
    fn gate_fused_honours_offset_views() {
        let dev = metal_device().unwrap();
        let width = HC * HIDDEN;
        let (n, skip) = (2usize, 3usize);
        let w = gate_weights(&dev, HC, HIDDEN, LOW_RANK);

        let big = t2(
            pseudo_random((n + skip + 1) * width, 0xF7, -2.0, 2.0),
            n + skip + 1,
            width,
            &dev,
        );
        let view = big.narrow(0, skip, n).unwrap();
        assert_ne!(
            view.layout().start_offset(),
            0,
            "the narrowed view must actually carry an offset"
        );
        let copy = t2(flat(&view), n, width, &dev);

        let run = |stream: &Tensor| {
            let (low, inject, scales) = hc_gate_down(
                stream,
                &w.norm_w,
                Some(&w.inject_w),
                &w.down,
                HC,
                HIDDEN,
                LOW_RANK,
                EPS,
            )
            .unwrap();
            // Offset the bottleneck activation too: pad it with a leading row
            // and hand the second kernel the narrowed view of that.
            let padded = Tensor::cat(
                &[
                    &Tensor::zeros((1, LOW_RANK), DType::F32, &dev).unwrap(),
                    &low,
                ],
                0,
            )
            .unwrap();
            let low_view = padded.narrow(0, 1, stream.dim(0).unwrap()).unwrap();
            assert_ne!(low_view.layout().start_offset(), 0);
            let mixed = hc_gate_up_mix(
                &low_view, &w.up, stream, &w.norm_w, &scales, HC, HIDDEN, LOW_RANK,
            )
            .unwrap();
            (mixed, inject.unwrap(), scales)
        };

        let (v_mixed, v_inject, v_scales) = run(&view);
        let (c_mixed, c_inject, c_scales) = run(&copy);
        assert_f32_bits_eq(
            &v_mixed,
            &c_mixed,
            "fused gate mixed over an offset carrier",
        );
        assert_f32_bits_eq(
            &v_inject,
            &c_inject,
            "fused gate inject over an offset carrier",
        );
        assert_f32_bits_eq(
            &v_scales,
            &c_scales,
            "fused gate scales over an offset carrier",
        );
    }

    /// Geometry outside the kernels' bounds is refused by the predicate the
    /// caller asks — which is what keeps such a gate on the split path — and the
    /// launchers hard-error rather than computing something on a partition that
    /// does not tile. Each case below breaks exactly one bound.
    #[test]
    fn gate_fused_refuses_unsupported_geometry() {
        let dev = metal_device().unwrap();
        // `hidden` not a whole number of q8_0 blocks: a carrier block would
        // straddle two streams and the per-stream statistics would mix.
        assert!(!hc_gate_fused_supported(4, 2560 + 16, 320, GgmlDType::Q8_0));
        // The carrier's block count must tile the fixed thread partition ...
        assert!(!hc_gate_fused_supported(4, 2560 - 32, 320, GgmlDType::Q8_0));
        // ... by no more blocks per thread than the register bound allows
        // (3 * 160 blocks here).
        assert!(!hc_gate_fused_supported(4, 3840, 320, GgmlDType::Q8_0));
        // A stream count that is not a power of two has no shuffle butterfly.
        assert!(!hc_gate_fused_supported(3, 2560, 320, GgmlDType::Q8_0));
        // A bottleneck wider than the staged threadgroup array.
        assert!(!hc_gate_fused_supported(4, 2560, 2048, GgmlDType::Q8_0));
        // Any dtype but q8_0: the kernels read that block layout directly.
        assert!(!hc_gate_fused_supported(4, 2560, 320, GgmlDType::Q4K));
        assert!(hc_gate_fused_supported(4, 2560, 320, GgmlDType::Q8_0));

        // The launchers refuse what the predicate refuses. The plane here is a
        // real q8_0 weight of a covered shape; the GEOMETRY passed alongside it
        // is the unsupported one.
        let w = gate_weights(&dev, HC, HIDDEN, LOW_RANK);
        let stream = t2(
            pseudo_random(HC * HIDDEN, 0xF8, -1.0, 1.0),
            1,
            HC * HIDDEN,
            &dev,
        );
        assert!(hc_gate_down(&stream, &w.norm_w, None, &w.down, 3, HIDDEN, LOW_RANK, EPS).is_err());
        let (low, _, scales) =
            hc_gate_down(&stream, &w.norm_w, None, &w.down, HC, HIDDEN, LOW_RANK, EPS).unwrap();
        assert!(
            hc_gate_up_mix(
                &low, &w.up, &stream, &w.norm_w, &scales, 3, HIDDEN, LOW_RANK
            )
            .is_err()
        );
        // And a plane whose shape contradicts the geometry, which would
        // otherwise index off the end of the weight.
        assert!(
            hc_gate_up_mix(
                &low, &w.down, &stream, &w.norm_w, &scales, HC, HIDDEN, LOW_RANK
            )
            .is_err()
        );

        // A plane whose declared shape AGREES with the geometry but outruns its
        // own allocation. Nothing else in the call carries the weight's length —
        // it is raw bytes, not a tensor — so without the buffer bound the
        // kernels would read off the end of device memory.
        let (small, _, _) = build_q8_plane(
            &dev,
            32,
            LOW_RANK,
            &pseudo_random(32 * LOW_RANK, 0xF9, -1.0, 1.0),
        );
        let lying_down = QuantPlane {
            out_dim: LOW_RANK,
            in_dim: HC * HIDDEN,
            ..small.clone()
        };
        assert!(
            hc_gate_down(
                &stream,
                &w.norm_w,
                None,
                &lying_down,
                HC,
                HIDDEN,
                LOW_RANK,
                EPS
            )
            .is_err()
        );
        let lying_up = QuantPlane {
            out_dim: HC * HIDDEN,
            in_dim: LOW_RANK,
            ..small
        };
        assert!(
            hc_gate_up_mix(
                &low, &lying_up, &stream, &w.norm_w, &scales, HC, HIDDEN, LOW_RANK
            )
            .is_err()
        );
    }

    /// The fused gate's launch geometry is spelled out in BOTH languages — a
    /// `#define` shaping the kernels' threadgroups, register arrays and staged
    /// threadgroup memory, and the host constants that size the grid and refuse
    /// a geometry those bounds do not cover. Nothing links them, and moving only
    /// one side would produce a launch whose threads and arrays disagree with no
    /// diagnostic, so this test is the link (same shape as
    /// `hc_max_streams_matches_metal`).
    #[test]
    fn hc_gate_constants_match_metal() {
        const SRC: &str = include_str!("hc.metal");
        let parse = |name: &str| -> usize {
            let mut found = None;
            for line in SRC.lines() {
                if let Some(rest) = line.trim().strip_prefix(&format!("#define {name} ")) {
                    assert!(found.is_none(), "hc.metal defines {name} more than once");
                    found = Some(
                        rest.trim()
                            .parse::<usize>()
                            .unwrap_or_else(|_| panic!("{name} must be a plain integer literal")),
                    );
                }
            }
            found.unwrap_or_else(|| panic!("hc.metal must #define {name}"))
        };
        for (name, host) in [
            ("HC_GATE_THREADS", dispatch::HC_GATE_THREADS),
            (
                "HC_GATE_MAX_BLK_PER_THREAD",
                dispatch::HC_GATE_MAX_BLK_PER_THREAD,
            ),
            ("HC_GATE_ROWS_PER_TG", dispatch::HC_GATE_ROWS_PER_TG),
            ("HC_GATE_MIX_THREADS", dispatch::HC_GATE_MIX_THREADS),
            ("HC_GATE_MAX_LOW_RANK", dispatch::HC_GATE_MAX_LOW_RANK),
        ] {
            assert_eq!(
                parse(name),
                host,
                "hc.metal's {name} ({}) and dispatch.rs's ({host}) must agree",
                parse(name),
            );
        }
    }

    /// The widest carrier the host admits. The shipped geometry is 4 streams,
    /// so 5..=8 is exercised nowhere else — and 8 is exactly where the kernel's
    /// `HC_MAX_STREAMS`-bounded accumulator loop runs to its last slot, which
    /// is the index a fencepost error would walk off. Both launch arms, graded
    /// against the frozen oracle.
    #[test]
    fn eight_streams_match_reference() {
        let dev = metal_device().unwrap();
        let (hc_count, hidden, low_rank, n) = (8usize, 64usize, 32usize, 3usize);
        let width = hc_count * hidden;
        assert!(hc_norm_supported(hc_count, hidden));

        let norm_w_v: Vec<f32> = pseudo_random(width, 0xE1, -0.5, 0.5)
            .iter()
            .map(|v| 1.0 + 0.1 * v)
            .collect();
        let down_w_v = pseudo_random(low_rank * width, 0xE2, -0.03, 0.03);
        let up_w_v = pseudo_random(width * low_rank, 0xE3, -0.05, 0.05);
        let inject_w_v = pseudo_random(hc_count * width, 0xE4, -0.02, 0.02);

        let norm_w = Tensor::from_vec(norm_w_v.clone(), width, &Device::Cpu)
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let down_t = t2(down_w_v.clone(), low_rank, width, &dev)
            .t()
            .unwrap()
            .contiguous()
            .unwrap();
        let up_t = t2(up_w_v.clone(), width, low_rank, &dev)
            .t()
            .unwrap()
            .contiguous()
            .unwrap();
        let inject_w = t2(inject_w_v.clone(), hc_count, width, &dev);

        let reference = GatedResidualRef {
            hc_count,
            hidden,
            low_rank,
            norm_w: norm_w_v.clone(),
            down_w: down_w_v,
            up_w: up_w_v,
            inject_w: inject_w_v,
        };
        let stream_v = pseudo_random(n * width, 0xE5, -2.0, 2.0);
        let stream = t2(stream_v.clone(), n, width, &dev);
        let r_normed = grouped_rms_norm_batch(&stream_v, &norm_w_v, hidden, EPS);
        let (r_mixed, r_inject) = reference.read_batch(&stream_v, EPS);

        for split in [true, false] {
            let (normed, inject) =
                hc_norm_arm(split, &stream, &norm_w, Some(&inject_w), hc_count, hidden);
            let inject = inject.expect("a gate with an injection head returns one");
            let low = hc_silu_quarter(&normed.matmul(&down_t).unwrap(), hc_count).unwrap();
            let mixed = hc_mix(&low.matmul(&up_t).unwrap(), &normed, hc_count, hidden).unwrap();
            assert_rel_l2(
                &normed,
                &r_normed,
                1e-5,
                &format!("hc=8 normed split={split}"),
            );
            assert_rel_l2(
                &inject,
                &r_inject,
                1e-5,
                &format!("hc=8 inject split={split}"),
            );
            assert_rel_l2(&mixed, &r_mixed, 1e-5, &format!("hc=8 mixed split={split}"));
        }
    }
}
