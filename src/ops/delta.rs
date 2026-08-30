//! Host side of the fused gated-DeltaNet ops (conv+silu+state, beta/decay head,
//! q/k L2 norm, recurrent scan, gated output norm). Kernel-side rounding
//! contracts live in delta.metal: the conv and beta/decay kernels are
//! bit-identical to the candle chains they replace, while the two norms and the
//! scan are bounded — each partitions across threads a reduction the reference
//! runs in one order (simd_sum over the head dim, and the scan's contractions
//! where the reference runs a gemm).

use anyhow::Result;
use candle_core::Tensor;

use crate::config::ZGate;
use crate::ops::dispatch;

/// The head dim the scan kernel is specialized to — the production geometry of
/// both checkpoints (27B: 16 K-heads / 48 V-heads at 128; 35B-A3B: 16 / 32 at
/// 128). A block at any other head dim keeps the reference scan.
pub const DELTA_HEAD_DIM: usize = dispatch::DELTA_HEAD_DIM;

/// Causal depthwise conv over the fused qkv stream, silu'd, plus the window the
/// next call starts from. `state` `[taps - 1, conv_dim]` is the carried window,
/// `qkv` `[seq, conv_dim]` this chunk's rows, `w` `[taps, conv_dim]` the taps
/// oldest-first; all f32 contiguous. Returns `([seq, conv_dim], [taps - 1,
/// conv_dim])`. Replaces the reference's cat + per-tap broadcast_mul chain +
/// silu + zeros_like/slice_set state write with ONE pass, bit-identically
/// (`conv_matches_reference_bitwise` proves it). Metal only; the caller's
/// kill-switch is the reference scan (`XWEN_DELTA_CLASSIC`).
pub fn delta_conv(state: &Tensor, qkv: &Tensor, w: &Tensor) -> Result<(Tensor, Tensor)> {
    dispatch::run_delta_conv(state, qkv, w)
}

/// `beta = sigmoid(b_raw)` and the LOG decay `g = ssm_a * softplus(a_raw +
/// dt_bias)` from the fused `[seq, 2 * v_heads]` beta|alpha projection output.
/// `ssm_a` is the pre-baked `-exp(A_log)` and `dt_bias` the dt offset, both
/// `[v_heads]` f32. Returns `(beta, g)`, both `[seq, v_heads]`. The scan kernel
/// exponentiates `g`, so the reference's separate exp pass is folded away.
/// Bit-identical to the candle chain given the same `ba`
/// (`ba_matches_reference_bitwise` proves it).
pub fn delta_ba(ba: &Tensor, ssm_a: &Tensor, dt_bias: &Tensor) -> Result<(Tensor, Tensor)> {
    dispatch::run_delta_ba(ba, ssm_a, dt_bias)
}

/// The beta|alpha PROJECTION and that same head in ONE dispatch: `x` is the
/// layer input `[seq, hidden]` f32, `w` the concatenated `[hidden, 2 *
/// v_heads]` beta|alpha weight built at load. Returns the same `(beta, g)` as
/// `delta_ba` over `x.matmul(w)`, with `g` the LOG decay — one kernel instead
/// of a candle gemv plus [`delta_ba`].
///
/// Only for the token counts [`delta_ba_fused_applies`] admits: the kernel
/// reads the whole weight once per token tile, so a prefill chunk keeps the
/// gemm. NOT bit-identical to the chain it replaces — the dot product
/// reassociates against candle's gemv (`ba_fused_matches_the_gemv_chain`
/// grades it at 2e-6); the epilogue is the same Metal helper, so nothing else
/// differs.
pub fn delta_ba_fused(
    x: &Tensor,
    w: &Tensor,
    ssm_a: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    dispatch::run_delta_ba_fused(x, w, ssm_a, dt_bias)
}

/// Whether [`delta_ba_fused`] can serve this input and weight — the caller's
/// fork between it and the `matmul` + [`delta_ba`] chain. False under the
/// `XWEN_DELTA_BA_CLASSIC` kill switch, off a Metal device, on a
/// non-contiguous or non-f32 operand, and outside the geometry the kernel's
/// grid is built for (token count, V-heads, hidden dim).
pub fn delta_ba_fused_applies(x: &Tensor, w: &Tensor) -> bool {
    !crate::ops::delta_ba_classic() && dispatch::delta_ba_fused_applies(x, w)
}

/// Gated output RMSNorm: `rms_norm(o, eps) * ssm_norm_weight * gate(z)` per
/// head, one pass. `o` and `z` are `[seq, v_heads, head_dim]` f32 contiguous,
/// `w` is `[head_dim]`. The gate multiplies AFTER the weight and outside the
/// norm, so it never enters the statistic.
///
/// `gate` is the checkpoint's z-gate activation ([`ZGate`], from
/// `Arch::z_gate`): silu on qwen35/qwen35moe, sigmoid on qwen4exp.
pub fn delta_gnorm(o: &Tensor, z: &Tensor, w: &Tensor, eps: f32, gate: ZGate) -> Result<Tensor> {
    dispatch::run_delta_gnorm(o, z, w, eps, gate)
}

/// The q/k L2 clamp-norm, `x / max(||x||, eps)`, over the leading
/// `2 * k_heads * 128` columns of the conv output — the per-K-head q planes
/// followed by the k planes. `conv` is `[seq, conv_dim]` f32 contiguous; returns
/// `[seq, 2 * k_heads * 128]` in the same order. v is not normalized and stays
/// in `conv`.
///
/// Off the shipped path: `delta_scan`'s default kernel normalizes q and k in its
/// own load stage, and only the `XWEN_DELTA_SCAN_V2` artifact needs this
/// hoisted. Kept and tested alongside it.
pub fn delta_l2norm(conv: &Tensor, k_heads: usize, eps: f32) -> Result<Tensor> {
    dispatch::run_delta_l2norm(conv, k_heads, eps)
}

/// The delta-rule recurrence in ONE dispatch, however long the chunk:
/// `S *= exp(g); d = (v - k·S) * beta; S += k (x) d; o = q·S / sqrt(d_k)` per
/// timestep. `conv` `[seq, conv_dim]` is the silu'd conv output (q | k | v
/// fused) — q and k are L2 clamp-normalized in the kernel's load stage (or by a
/// separate `delta_l2norm` dispatch under `XWEN_DELTA_SCAN_V2`) and read with
/// the tiled K-head mapping, so no
/// materialized tile-and-broadcast is needed. `XWEN_DELTA_DECODE_KERNEL` sends
/// a one-token chunk to the decode-specialized kernel instead (same math, same
/// bound, a measured wash). `beta`/`g` are `[seq, v_heads]`,
/// `s` is the incoming `[v_heads, 128, 128]` f32 state. Returns the per-token
/// output `[seq, v_heads, 128]` and the state after the last token; `s` itself
/// is left untouched.
pub fn delta_scan(
    conv: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    s: &Tensor,
    k_heads: usize,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let (o, states) = dispatch::run_delta_scan(conv, beta, g, s, k_heads, eps, 1)?;
    Ok((o, states.squeeze(0)?))
}

/// `delta_scan`, plus the per-token state trail a speculative verify walk rolls
/// back through. `state_planes` must be in `1..=seq`; the second returned tensor
/// is `[state_planes, v_heads, 128, 128]` f32 MOST-RECENT-FIRST — plane p is the
/// state after token `seq - 1 - p`, so plane 0 is the final state and
/// `state_planes == 1` is exactly `delta_scan` (same kernel, same dispatch, same
/// bits). A caller wanting the state after token t asks for `seq` planes and
/// reads plane `seq - 1 - t`.
///
/// The ordering is llama.cpp's snapshot-slot convention, kept so the two are
/// readable against each other. The trail is one buffer, not one per token: at
/// the 27B's 48 V-heads a 16-token verify block is ~48 MiB per DeltaNet layer,
/// live only for the walk.
pub fn delta_scan_with_trail(
    conv: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    s: &Tensor,
    k_heads: usize,
    eps: f32,
    state_planes: usize,
) -> Result<(Tensor, Tensor)> {
    dispatch::run_delta_scan(conv, beta, g, s, k_heads, eps, state_planes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::linear_attn::{l2_norm, softplus, tile_heads};
    use crate::ops::dispatch::testutil::{pseudo_random, rel_l2};
    use candle_core::{DType, Device};
    use candle_nn::ops::{sigmoid, silu};

    /// The two shipped DeltaNet geometries, `(k_heads, v_heads)` at head dim
    /// 128: the dense 27B (16 K-heads, 48 V-heads, conv width 10240) and the
    /// 35B-A3B (16 / 32, conv width 8192). Every kernel test runs both, so a
    /// K-head tiling or width assumption that only holds for one is caught.
    ///
    /// Qwen3.8-Flash-Next is the third checkpoint and needs no third entry: its
    /// gated-DeltaNet layers are 16 K-heads / 48 V-heads at 128, byte-for-byte
    /// the 27B's geometry (docs/qwen4exp-tensors.md). It differs in how many
    /// layers run it, which is a model-level number, not a kernel one.
    const GEOMETRIES: [(usize, usize); 2] = [(16, 48), (16, 32)];

    const HD: usize = DELTA_HEAD_DIM;
    /// The models' `rms_norm_eps`, which doubles as the L2 norm floor.
    const EPS: f64 = 1e-6;

    fn on_device(v: Vec<f32>, shape: impl Into<candle_core::Shape>, dev: &Device) -> Tensor {
        Tensor::from_vec(v, shape, dev).unwrap()
    }

    fn assert_bits_eq(got: &Tensor, want: &Tensor, what: &str) {
        assert_eq!(got.dims(), want.dims(), "{what}: shape mismatch");
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let w: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (a, b)) in g.iter().zip(w.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{what}: element {i} differs (fused {a:?} bits {:#010x}, reference {b:?} bits {:#010x})",
                a.to_bits(),
                b.to_bits(),
            );
        }
    }

    fn assert_close(got: &Tensor, want: &Tensor, tol: f32, what: &str) {
        assert_eq!(got.dims(), want.dims(), "{what}: shape mismatch");
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let w: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        let r = rel_l2(&g, &w);
        assert!(r < tol, "{what}: relative l2 {r:e} exceeds {tol:e}");
        assert!(
            g.iter().all(|v| v.is_finite()),
            "{what}: produced a non-finite value"
        );
    }

    /// The reference conv chain of `LinearAttnBlock::forward_classic`: build the
    /// stream by concatenating the carried window in front of the chunk, sum the
    /// taps one broadcast_mul at a time, then silu. Returns the conv output and
    /// the window the next call starts from.
    fn reference_conv(state: &Tensor, qkv: &Tensor, w: &Tensor, taps: usize) -> (Tensor, Tensor) {
        let seq = qkv.dim(0).unwrap();
        let tail = taps - 1;
        let stream = Tensor::cat(&[state, qkv], 0).unwrap();
        let mut conv = stream
            .narrow(0, 0, seq)
            .unwrap()
            .broadcast_mul(&w.narrow(0, 0, 1).unwrap())
            .unwrap();
        for j in 1..taps {
            conv = (conv
                + stream
                    .narrow(0, j, seq)
                    .unwrap()
                    .broadcast_mul(&w.narrow(0, j, 1).unwrap())
                    .unwrap())
            .unwrap();
        }
        (
            silu(&conv).unwrap(),
            stream.narrow(0, seq, tail).unwrap().contiguous().unwrap(),
        )
    }

    /// UNIT 1: the fused conv kernel must reproduce the reference's cat +
    /// per-tap broadcast_mul + silu chain BIT-FOR-BIT, and hand back exactly the
    /// window the reference's `stream.narrow(0, seq, tail)` slice holds. Both
    /// conv widths, and every seq class the state write has to distinguish:
    /// a single decode token (one thread claims all three window rows), chunks
    /// shorter than the window, the exact-window length, and a prefill chunk.
    /// Bit-identity is what keeps this kernel off the parity gates' books —
    /// never loosen it to a tolerance.
    #[test]
    fn conv_matches_reference_bitwise() {
        let device = metal_device().unwrap();
        let taps = 4usize;
        for (ki, &(k_heads, v_heads)) in GEOMETRIES.iter().enumerate() {
            let conv_dim = (2 * k_heads + v_heads) * HD;
            for &seq in &[1usize, 2, 3, 4, 17, 512] {
                let seed = 0x1000 + ki as u64 * 977 + seq as u64;
                let state = on_device(
                    pseudo_random((taps - 1) * conv_dim, seed, -3.0, 3.0),
                    (taps - 1, conv_dim),
                    &device,
                );
                let qkv = on_device(
                    pseudo_random(seq * conv_dim, seed + 1, -3.0, 3.0),
                    (seq, conv_dim),
                    &device,
                );
                let w = on_device(
                    pseudo_random(taps * conv_dim, seed + 2, -1.0, 1.0),
                    (taps, conv_dim),
                    &device,
                );

                let (got, got_state) = delta_conv(&state, &qkv, &w).unwrap();
                let (want, want_state) = reference_conv(&state, &qkv, &w, taps);
                let label = format!("conv k={k_heads} v={v_heads} seq={seq}");
                assert_bits_eq(&got, &want, &label);
                assert_bits_eq(&got_state, &want_state, &format!("{label} window"));
            }
        }
    }

    /// UNIT 2: the fused beta/decay head must reproduce the reference's
    /// `sigmoid(beta_logits)` and `ssm_a * softplus(alpha_logits + dt_bias)`
    /// BIT-FOR-BIT given the same projection output, and must read the fused
    /// projection's column blocks the right way round (beta first, alpha
    /// second) — logits spanning both softplus branches make a swap visible.
    /// The kernel emits the LOG decay, which the scan kernel exponentiates; the
    /// decay assertion below exponentiates through candle on BOTH sides, so it
    /// grades the same `g` bit-identity in the domain the reference's separate
    /// `exp` pass consumed — it says nothing about the scan's own fast-math
    /// `exp`, which `scan_matches_reference` bounds instead.
    #[test]
    fn ba_matches_reference_bitwise() {
        let device = metal_device().unwrap();
        for (ki, &(_, v_heads)) in GEOMETRIES.iter().enumerate() {
            for &seq in &[1usize, 17, 512] {
                let seed = 0x2000 + ki as u64 * 131 + seq as u64;
                // Wide logits: past ±40 softplus is pure relu on one side and
                // pure tail on the other, and sigmoid saturates.
                let ba = on_device(
                    pseudo_random(seq * 2 * v_heads, seed, -60.0, 60.0),
                    (seq, 2 * v_heads),
                    &device,
                );
                // ssm_a is the pre-baked -exp(A_log): strictly negative.
                let a: Vec<f32> = pseudo_random(v_heads, seed + 1, 0.1, 4.0)
                    .into_iter()
                    .map(|v| -v)
                    .collect();
                let ssm_a = on_device(a, v_heads, &device);
                let dt_bias = on_device(
                    pseudo_random(v_heads, seed + 2, -4.0, 4.0),
                    v_heads,
                    &device,
                );

                let (beta, g) = delta_ba(&ba, &ssm_a, &dt_bias).unwrap();

                let b_raw = ba.narrow(1, 0, v_heads).unwrap().contiguous().unwrap();
                let a_raw = ba
                    .narrow(1, v_heads, v_heads)
                    .unwrap()
                    .contiguous()
                    .unwrap();
                let want_beta = sigmoid(&b_raw).unwrap();
                let dt = a_raw.broadcast_add(&dt_bias).unwrap();
                let want_g = softplus(&dt).unwrap().broadcast_mul(&ssm_a).unwrap();

                let label = format!("ba v={v_heads} seq={seq}");
                assert_bits_eq(&beta, &want_beta, &format!("{label} beta"));
                assert_bits_eq(&g, &want_g, &format!("{label} g"));
                assert_bits_eq(
                    &g.exp().unwrap(),
                    &want_g.exp().unwrap(),
                    &format!("{label} decay"),
                );
            }
        }
    }

    /// The three shipped `(hidden, v_heads)` pairs the fused beta|alpha
    /// projection runs at: the dense 27B / 3.8-27B (5120, 48), the 35B-A3B
    /// (2048, 32) and Flash-Next (2560, 48). Two different hidden dims at the
    /// same V-head count, and two different V-head counts, so a kernel that
    /// confused the two axes fails on at least one.
    const BA_GEOMETRIES: [(usize, usize); 3] = [(5120, 48), (2048, 32), (2560, 48)];

    /// A `(hidden, v_heads)` pair that takes BOTH of the kernel's tail
    /// branches, which no shipped geometry does: every one of those has
    /// `2 * v_heads` a multiple of `DELTA_BA_COLS` (so the column guard
    /// `col < two_vh` never fires) and `hidden` a multiple of
    /// `DELTA_BA_ROWS` (so the strided row walk never stops a thread short of
    /// the others). 12 columns leaves the second threadgroup four lanes idle
    /// and 2000 rows leaves 80 of the 128 row chunks one iteration shorter, so
    /// a guard dropped from either loop shows up as garbage in the columns or
    /// the partials past the end. Not a shape any Qwen checkpoint has — the
    /// kernel is offered to whatever custom GGUF fits its ceilings, which is
    /// what makes the tails reachable in the first place.
    const BA_TAIL_GEOMETRY: (usize, usize) = (2000, 6);

    /// The gemv + `delta_ba` chain the fused projection replaces, run through
    /// candle exactly as `forward_fused` runs it off the fused path.
    fn reference_ba_chain(
        x: &Tensor,
        w: &Tensor,
        ssm_a: &Tensor,
        dt_bias: &Tensor,
    ) -> (Tensor, Tensor) {
        let ba = x.matmul(w).unwrap();
        delta_ba(&ba, ssm_a, dt_bias).unwrap()
    }

    /// Build one geometry's operands: the layer input, the concatenated
    /// beta|alpha weight, the pre-baked (strictly negative) `ssm_a`, and the dt
    /// offset.
    fn ba_operands(
        hidden: usize,
        v_heads: usize,
        seq: usize,
        seed: u64,
        device: &Device,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let x = on_device(
            pseudo_random(seq * hidden, seed, -2.0, 2.0),
            (seq, hidden),
            device,
        );
        let w = on_device(
            pseudo_random(hidden * 2 * v_heads, seed + 1, -0.08, 0.08),
            (hidden, 2 * v_heads),
            device,
        );
        let a: Vec<f32> = pseudo_random(v_heads, seed + 2, 0.1, 4.0)
            .into_iter()
            .map(|v| -v)
            .collect();
        let ssm_a = on_device(a, v_heads, device);
        let dt_bias = on_device(pseudo_random(v_heads, seed + 3, -4.0, 4.0), v_heads, device);
        (x, w, ssm_a, dt_bias)
    }

    /// UNIT 2b: the one-dispatch beta|alpha projection must reproduce the gemv
    /// + `delta_ba` chain it replaces, at every shipped geometry and every
    /// token-tiling class the kernel distinguishes: one token (the `_t1`
    /// specialization), a chunk that straddles a tile boundary, an exact tile,
    /// and the largest chunk the host will hand it. `BA_TAIL_GEOMETRY` runs the
    /// short token counts again at a width and depth that fire the column and
    /// row tail guards no shipped geometry reaches.
    ///
    /// NOT bit-identity, deliberately: the kernel sums each dot product as
    /// `DELTA_BA_ROWS` per-thread partials folded in a tree where candle's gemv
    /// sums in its own order, so f32 reassociation alone separates them. The
    /// epilogue is the same Metal helper on both sides, so 2e-6 — the bound the
    /// two reassociating norms carry — is the right class of tolerance: the
    /// widest geometry's 5120-term dot measures 1.0e-6 on `g`, whose softplus
    /// passes the dot's absolute error through as a relative one wherever the
    /// decay is small. Anything looser would stop catching a swapped
    /// beta/alpha column block or a mis-tiled token, both of which fail here by
    /// orders of magnitude.
    #[test]
    fn ba_fused_matches_the_gemv_chain() {
        let device = metal_device().unwrap();
        let shipped_seqs: &[usize] = &[1, 3, 4, 5, 16, dispatch::DELTA_BA_MAX_SEQ];
        let tail_seqs: &[usize] = &[1, 3, 5];
        let cases = BA_GEOMETRIES
            .iter()
            .map(|&g| (g, shipped_seqs))
            .chain(std::iter::once((BA_TAIL_GEOMETRY, tail_seqs)));
        for (gi, ((hidden, v_heads), seqs)) in cases.enumerate() {
            for &seq in seqs {
                let seed = 0x2600 + gi as u64 * 401 + seq as u64;
                let (x, w, ssm_a, dt_bias) = ba_operands(hidden, v_heads, seq, seed, &device);

                assert!(
                    crate::ops::delta_ba_fused_applies(&x, &w),
                    "hidden={hidden} v={v_heads} seq={seq} is inside the kernel's ceilings"
                );
                let (beta, g) = delta_ba_fused(&x, &w, &ssm_a, &dt_bias).unwrap();
                let (want_beta, want_g) = reference_ba_chain(&x, &w, &ssm_a, &dt_bias);

                let label = format!("ba_fused hidden={hidden} v={v_heads} seq={seq}");
                assert_eq!(beta.dims(), &[seq, v_heads]);
                assert_close(&beta, &want_beta, 2e-6, &format!("{label} beta"));
                assert_close(&g, &want_g, 2e-6, &format!("{label} g"));
            }
        }
    }

    /// The fused projection is a shape, not a fallback: everything outside the
    /// geometry its grid is built for must be REFUSED by the dispatch and
    /// reported as inapplicable by the predicate, so the block takes the gemv
    /// chain instead of running a kernel that would leave columns or tokens
    /// unwritten.
    #[test]
    fn ba_fused_refuses_a_geometry_its_grid_cannot_cover() {
        let device = metal_device().unwrap();
        let (hidden, v_heads) = (2560usize, 48usize);
        let (x, w, ssm_a, dt_bias) = ba_operands(hidden, v_heads, 1, 0x2700, &device);
        let f32z = |d: (usize, usize)| Tensor::zeros(d, DType::F32, &device).unwrap();
        let vec_f32 = |n: usize| Tensor::zeros(n, DType::F32, &device).unwrap();

        // Above the token ceiling the candle gemm's weight reuse wins, so the
        // kernel is not merely slower there — it is not offered at all.
        let long = ba_operands(
            hidden,
            v_heads,
            dispatch::DELTA_BA_MAX_SEQ + 1,
            0x2701,
            &device,
        );
        assert!(!crate::ops::delta_ba_fused_applies(&long.0, &long.1));
        assert!(delta_ba_fused(&long.0, &long.1, &long.2, &long.3).is_err());

        // An empty chunk encodes a zero-dimension grid.
        let empty = f32z((0, hidden));
        assert!(!crate::ops::delta_ba_fused_applies(&empty, &w));
        assert!(delta_ba_fused(&empty, &w, &ssm_a, &dt_bias).is_err());

        // An odd column count cannot split into beta|alpha.
        let odd = f32z((hidden, 2 * v_heads - 1));
        assert!(!crate::ops::delta_ba_fused_applies(&x, &odd));
        assert!(delta_ba_fused(&x, &odd, &ssm_a, &dt_bias).is_err());

        // A weight whose rows are not x's hidden dim.
        let short = f32z((hidden - 1, 2 * v_heads));
        assert!(!crate::ops::delta_ba_fused_applies(&x, &short));
        assert!(delta_ba_fused(&x, &short, &ssm_a, &dt_bias).is_err());

        // Beyond the V-head and hidden ceilings.
        let wide = f32z((hidden, 2 * (dispatch::DELTA_BA_MAX_V_HEADS + 1)));
        assert!(!crate::ops::delta_ba_fused_applies(&x, &wide));
        let tall_x = f32z((1, dispatch::DELTA_BA_MAX_HIDDEN + 1));
        let tall_w = f32z((dispatch::DELTA_BA_MAX_HIDDEN + 1, 2 * v_heads));
        assert!(!crate::ops::delta_ba_fused_applies(&tall_x, &tall_w));
        assert!(delta_ba_fused(&tall_x, &tall_w, &ssm_a, &dt_bias).is_err());

        // Per-head vectors that do not match the weight's V-head count.
        assert!(delta_ba_fused(&x, &w, &vec_f32(v_heads + 1), &dt_bias).is_err());
        assert!(delta_ba_fused(&x, &w, &ssm_a, &vec_f32(v_heads - 1)).is_err());
    }

    /// UNIT 3: the fused gated output norm must match the reference's
    /// rms → ssm_norm.weight → silu(z) chain. Only the 128-term sum of squares
    /// reassociates (hardware simd_sum vs candle's reduce partition), so this is
    /// a tight bound rather than bit-identity. A non-uniform weight and gate
    /// logits of both signs make an ordering swap (gate before weight, or gate
    /// inside the norm) visible as a gross failure, not a rounding one.
    #[test]
    fn gnorm_matches_reference() {
        let device = metal_device().unwrap();
        for (ki, &(_, v_heads)) in GEOMETRIES.iter().enumerate() {
            for &seq in &[1usize, 17, 512] {
                let seed = 0x3000 + ki as u64 * 71 + seq as u64;
                let n = seq * v_heads * HD;
                let o = on_device(
                    pseudo_random(n, seed, -6.0, 6.0),
                    (seq, v_heads, HD),
                    &device,
                );
                let z = on_device(
                    pseudo_random(n, seed + 1, -8.0, 8.0),
                    (seq, v_heads, HD),
                    &device,
                );
                let w = on_device(pseudo_random(HD, seed + 2, 0.2, 2.0), HD, &device);

                let got = delta_gnorm(&o, &z, &w, EPS as f32, ZGate::Silu).unwrap();

                let ms = (o.sqr().unwrap().sum_keepdim(2).unwrap() / HD as f64).unwrap();
                let want = o
                    .broadcast_div(&(ms + EPS).unwrap().sqrt().unwrap())
                    .unwrap()
                    .broadcast_mul(&w.reshape((1, 1, HD)).unwrap())
                    .unwrap();
                let want = (want * silu(&z).unwrap()).unwrap();

                assert_close(&got, &want, 2e-6, &format!("gnorm v={v_heads} seq={seq}"));
            }
        }
    }

    /// UNIT 3a: the qwen4exp arm gates the same norm with `sigmoid(z)` instead
    /// of `silu(z)`, and `ZGate` is what selects between the two kernels.
    ///
    /// Both arms are graded against `ref_hc::gated_rms_norm_batch`, the frozen
    /// f32 oracle the transformers fixture pins — a second, independent
    /// reference for the silu arm as well as the first one for sigmoid. Same
    /// bound as the candle comparison above and for the same reason: only the
    /// 128-term sum of squares reassociates.
    ///
    /// The arms must also disagree grossly on the same inputs. Without that,
    /// a dispatch that resolved every request to one kernel name would satisfy
    /// every bound in this file.
    #[test]
    fn gnorm_sigmoid_arm_matches_reference() {
        use crate::qwen4exp::ref_hc::{ZGateRef, gated_rms_norm_batch};

        let device = metal_device().unwrap();
        for (ki, &(_, v_heads)) in GEOMETRIES.iter().enumerate() {
            for &seq in &[1usize, 17, 512] {
                let seed = 0x3500 + ki as u64 * 71 + seq as u64;
                let n = seq * v_heads * HD;
                let o_h = pseudo_random(n, seed, -6.0, 6.0);
                let z_h = pseudo_random(n, seed + 1, -8.0, 8.0);
                let w_h = pseudo_random(HD, seed + 2, 0.2, 2.0);
                let o = on_device(o_h.clone(), (seq, v_heads, HD), &device);
                let z = on_device(z_h.clone(), (seq, v_heads, HD), &device);
                let w = on_device(w_h.clone(), HD, &device);

                let mut arms = Vec::new();
                for (gate, gate_ref, name) in [
                    (ZGate::Sigmoid, ZGateRef::Sigmoid, "sigmoid"),
                    (ZGate::Silu, ZGateRef::Silu, "silu"),
                ] {
                    let got = delta_gnorm(&o, &z, &w, EPS as f32, gate).unwrap();
                    let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
                    let want = gated_rms_norm_batch(&o_h, &w_h, &z_h, gate_ref, EPS as f32);
                    let r = rel_l2(&g, &want);
                    assert!(
                        r < 2e-6,
                        "gnorm {name} v={v_heads} seq={seq}: relative l2 {r:e} exceeds 2e-6"
                    );
                    arms.push(g);
                }
                let spread = rel_l2(&arms[0], &arms[1]);
                assert!(
                    spread > 0.1,
                    "the sigmoid and silu arms agree to {spread:e} at v={v_heads} seq={seq}; \
                     one kernel is serving both"
                );
            }
        }
    }

    /// UNIT 3b: the hoisted q/k norm must reproduce `linear_attn::l2_norm` over
    /// exactly the conv output's leading q and k planes, and must leave the v
    /// columns alone. Only the 128-term sum of squares reassociates, so this is a
    /// tight bound rather than bit-identity. Two input scales: ordinary
    /// activations, whose norms clear the eps floor, and activations so small
    /// that every norm is floored — a relative comparison at that scale is what
    /// grades the clamp form (`x / max(||x||, eps)`) rather than the rsqrt form.
    #[test]
    fn l2norm_matches_reference() {
        let device = metal_device().unwrap();
        for (ki, &(k_heads, v_heads)) in GEOMETRIES.iter().enumerate() {
            let conv_dim = (2 * k_heads + v_heads) * HD;
            for &seq in &[1usize, 17, 512] {
                for &scale in &[1.0f32, 1e-9] {
                    let seed = 0x3800 + ki as u64 * 53 + seq as u64;
                    let raw: Vec<f32> = pseudo_random(seq * conv_dim, seed, -2.0, 2.0)
                        .into_iter()
                        .map(|v| v * scale)
                        .collect();
                    let conv = on_device(raw, (seq, conv_dim), &device);

                    let got = delta_l2norm(&conv, k_heads, EPS as f32).unwrap();

                    let qk_dim = 2 * k_heads * HD;
                    let want = l2_norm(
                        &conv
                            .narrow(1, 0, qk_dim)
                            .unwrap()
                            .contiguous()
                            .unwrap()
                            .reshape((seq, 2 * k_heads, HD))
                            .unwrap(),
                        EPS,
                    )
                    .unwrap()
                    .reshape((seq, qk_dim))
                    .unwrap();

                    assert_close(
                        &got,
                        &want,
                        2e-6,
                        &format!("l2norm k={k_heads} v={v_heads} seq={seq} scale={scale:e}"),
                    );
                }
            }
        }
    }

    /// The reference scan of `LinearAttnBlock::forward_classic`, lifted verbatim
    /// onto raw tensors: split the conv output, L2-normalize q and k, tile them
    /// up to the V-head count, then walk the delta rule one token at a time.
    /// Returns the per-token output and the state after the last token.
    fn reference_scan(
        conv: &Tensor,
        beta: &Tensor,
        g: &Tensor,
        s0: &Tensor,
        k_heads: usize,
        v_heads: usize,
    ) -> (Tensor, Tensor) {
        let (o, states) = reference_scan_states(conv, beta, g, s0, k_heads, v_heads);
        let last = states.last().unwrap().clone();
        (o, last)
    }

    /// `reference_scan`, keeping the state after EVERY token in token order —
    /// the trail a rollback reads, and what `delta_scan_with_trail`'s planes are
    /// graded against.
    fn reference_scan_states(
        conv: &Tensor,
        beta: &Tensor,
        g: &Tensor,
        s0: &Tensor,
        k_heads: usize,
        v_heads: usize,
    ) -> (Tensor, Vec<Tensor>) {
        let seq = conv.dim(0).unwrap();
        let k_dim = k_heads * HD;
        let v_dim = v_heads * HD;
        let heads = |off: usize, width: usize, n: usize| {
            conv.narrow(1, off, width)
                .unwrap()
                .contiguous()
                .unwrap()
                .reshape((seq, n, HD))
                .unwrap()
        };
        let q = tile_heads(&l2_norm(&heads(0, k_dim, k_heads), EPS).unwrap(), v_heads).unwrap();
        let k = tile_heads(
            &l2_norm(&heads(k_dim, k_dim, k_heads), EPS).unwrap(),
            v_heads,
        )
        .unwrap();
        let v = heads(2 * k_dim, v_dim, v_heads);
        let decay = g.exp().unwrap();

        let scale = 1.0 / (HD as f64).sqrt();
        let mut s = s0.clone();
        let mut outs = Vec::with_capacity(seq);
        let mut states = Vec::with_capacity(seq);
        for t in 0..seq {
            let row = |t3: &Tensor| {
                t3.narrow(0, t, 1)
                    .unwrap()
                    .reshape((v_heads, 1, HD))
                    .unwrap()
            };
            let (qt, kt, vt) = (row(&q), row(&k), row(&v));
            let dec = decay
                .narrow(0, t, 1)
                .unwrap()
                .reshape((v_heads, 1, 1))
                .unwrap();
            let bt = beta
                .narrow(0, t, 1)
                .unwrap()
                .reshape((v_heads, 1, 1))
                .unwrap();

            s = s.broadcast_mul(&dec).unwrap();
            let sk = kt.matmul(&s).unwrap();
            let d = (vt - sk).unwrap().broadcast_mul(&bt).unwrap();
            s = (&s + kt.reshape((v_heads, HD, 1)).unwrap().matmul(&d).unwrap()).unwrap();
            outs.push(
                (qt.matmul(&s).unwrap() * scale)
                    .unwrap()
                    .reshape((1, v_heads, HD))
                    .unwrap(),
            );
            states.push(s.clone());
        }
        (Tensor::cat(&outs, 0).unwrap(), states)
    }

    /// UNIT 4: the fused scan must reproduce the reference recurrence — both
    /// the per-token output and the state it leaves behind — at both shipped
    /// geometries, for a single decode token and for prefill chunks. The
    /// contractions are partitioned across threads where the reference runs a
    /// gemm, so this is a tight fp32 bound. The V-head count differing from the
    /// K-head count is what pins the TILED head mapping: an interleaving
    /// broadcast pairs q/k with the wrong values and fails grossly.
    #[test]
    fn scan_matches_reference() {
        let device = metal_device().unwrap();
        for (ki, &(k_heads, v_heads)) in GEOMETRIES.iter().enumerate() {
            let conv_dim = (2 * k_heads + v_heads) * HD;
            // 67 is deliberately awkward: prime, and not a multiple of any tile
            // or simd width the scan is built out of.
            for &seq in &[1usize, 2, 33, 67, 512] {
                let seed = 0x4000 + ki as u64 * 313 + seq as u64;
                let conv = on_device(
                    pseudo_random(seq * conv_dim, seed, -2.0, 2.0),
                    (seq, conv_dim),
                    &device,
                );
                // beta in (0, 1) as sigmoid produces it, and a strictly
                // negative log-decay so the decay lands in (0, 1).
                let beta = on_device(
                    pseudo_random(seq * v_heads, seed + 1, 0.01, 0.99),
                    (seq, v_heads),
                    &device,
                );
                let g = on_device(
                    pseudo_random(seq * v_heads, seed + 2, -0.6, -0.001),
                    (seq, v_heads),
                    &device,
                );
                // A nonzero incoming state: a zero one would hide a decay or
                // state-load bug on the first token.
                let s0 = on_device(
                    pseudo_random(v_heads * HD * HD, seed + 3, -0.5, 0.5),
                    (v_heads, HD, HD),
                    &device,
                );

                let (o, s) = delta_scan(&conv, &beta, &g, &s0, k_heads, EPS as f32).unwrap();
                let (want_o, want_s) = reference_scan(&conv, &beta, &g, &s0, k_heads, v_heads);

                let label = format!("scan k={k_heads} v={v_heads} seq={seq}");
                assert_close(&o, &want_o, 1e-5, &format!("{label} out"));
                assert_close(&s, &want_s, 1e-5, &format!("{label} state"));
            }
        }
    }

    /// UNIT 4b: asked for one state plane per token, the scan must report the
    /// state after EVERY token — the trail a speculative verify walk rolls back
    /// through — under the same bound as its final state. The planes are
    /// most-recent-first, so plane `seq - 1 - t` is the state after token t; a
    /// kernel that wrote them in token order instead would pass at seq = 1 and
    /// fail here from seq = 2 up.
    #[test]
    fn scan_trail_records_the_state_after_every_token() {
        let device = metal_device().unwrap();
        for (ki, &(k_heads, v_heads)) in GEOMETRIES.iter().enumerate() {
            let conv_dim = (2 * k_heads + v_heads) * HD;
            for &seq in &[2usize, 5, 16] {
                let seed = 0x4400 + ki as u64 * 313 + seq as u64;
                let conv = on_device(
                    pseudo_random(seq * conv_dim, seed, -2.0, 2.0),
                    (seq, conv_dim),
                    &device,
                );
                let beta = on_device(
                    pseudo_random(seq * v_heads, seed + 1, 0.01, 0.99),
                    (seq, v_heads),
                    &device,
                );
                let g = on_device(
                    pseudo_random(seq * v_heads, seed + 2, -0.6, -0.001),
                    (seq, v_heads),
                    &device,
                );
                let s0 = on_device(
                    pseudo_random(v_heads * HD * HD, seed + 3, -0.5, 0.5),
                    (v_heads, HD, HD),
                    &device,
                );

                let (o, trail) =
                    delta_scan_with_trail(&conv, &beta, &g, &s0, k_heads, EPS as f32, seq).unwrap();
                assert_eq!(trail.dims(), &[seq, v_heads, HD, HD]);
                let (want_o, want_states) =
                    reference_scan_states(&conv, &beta, &g, &s0, k_heads, v_heads);

                let label = format!("trail k={k_heads} v={v_heads} seq={seq}");
                assert_close(&o, &want_o, 1e-5, &format!("{label} out"));
                for (t, want) in want_states.iter().enumerate() {
                    let plane = trail.narrow(0, seq - 1 - t, 1).unwrap().squeeze(0).unwrap();
                    assert_close(
                        &plane,
                        want,
                        1e-5,
                        &format!("{label} state after token {t}"),
                    );
                }
            }
        }
    }

    /// The trail is the same kernel and the same dispatch as the plain scan, so
    /// its most-recent plane is not merely close to `delta_scan`'s state — it is
    /// the same bits. This is what lets the block take one path whether or not a
    /// rollback checkpoint is armed.
    #[test]
    fn scan_trail_plane_zero_is_the_plain_scan_state() {
        let device = metal_device().unwrap();
        for (ki, &(k_heads, v_heads)) in GEOMETRIES.iter().enumerate() {
            let conv_dim = (2 * k_heads + v_heads) * HD;
            for &seq in &[1usize, 2, 16] {
                let seed = 0x4500 + ki as u64 * 313 + seq as u64;
                let conv = on_device(
                    pseudo_random(seq * conv_dim, seed, -2.0, 2.0),
                    (seq, conv_dim),
                    &device,
                );
                let beta = on_device(
                    pseudo_random(seq * v_heads, seed + 1, 0.01, 0.99),
                    (seq, v_heads),
                    &device,
                );
                let g = on_device(
                    pseudo_random(seq * v_heads, seed + 2, -0.6, -0.001),
                    (seq, v_heads),
                    &device,
                );
                let s0 = on_device(
                    pseudo_random(v_heads * HD * HD, seed + 3, -0.5, 0.5),
                    (v_heads, HD, HD),
                    &device,
                );

                let (plain_o, plain_s) =
                    delta_scan(&conv, &beta, &g, &s0, k_heads, EPS as f32).unwrap();
                let (o, trail) =
                    delta_scan_with_trail(&conv, &beta, &g, &s0, k_heads, EPS as f32, seq).unwrap();

                let label = format!("plane0 k={k_heads} v={v_heads} seq={seq}");
                assert_bits_eq(&o, &plain_o, &format!("{label} out"));
                assert_bits_eq(
                    &trail.narrow(0, 0, 1).unwrap().squeeze(0).unwrap(),
                    &plain_s,
                    &format!("{label} state"),
                );
            }
        }
    }

    /// A plane count outside `1..=seq` names a state no chunk of that length
    /// has, so the wrapper refuses rather than dispatching a kernel that would
    /// leave planes unwritten.
    #[test]
    fn scan_trail_rejects_a_plane_count_no_chunk_can_fill() {
        let device = metal_device().unwrap();
        let (k_heads, v_heads) = (16usize, 32usize);
        let conv_dim = (2 * k_heads + v_heads) * HD;
        let seq = 4usize;
        let conv = on_device(
            pseudo_random(seq * conv_dim, 0x5100, -2.0, 2.0),
            (seq, conv_dim),
            &device,
        );
        let beta = on_device(
            pseudo_random(seq * v_heads, 0x5101, 0.01, 0.99),
            (seq, v_heads),
            &device,
        );
        let g = on_device(
            pseudo_random(seq * v_heads, 0x5102, -0.6, -0.001),
            (seq, v_heads),
            &device,
        );
        let s0 = Tensor::zeros((v_heads, HD, HD), DType::F32, &device).unwrap();

        for planes in [0usize, seq + 1] {
            assert!(
                delta_scan_with_trail(&conv, &beta, &g, &s0, k_heads, EPS as f32, planes).is_err(),
                "{planes} planes over a {seq}-token chunk must be refused"
            );
        }
        assert!(delta_scan_with_trail(&conv, &beta, &g, &s0, k_heads, EPS as f32, seq).is_ok());
    }

    /// UNIT 4e: the decode kernel is a SECOND implementation of the same
    /// recurrence, taken by every seq == 1 step, so it is graded twice — against
    /// the general kernel it replaces and against the frozen reference — over
    /// several CONSECUTIVE steps that carry their own state forward. Carrying
    /// the state is the point: a per-step difference that a single step hides
    /// compounds, and a kernel that dropped or double-decayed part of its state
    /// slice diverges by step two.
    ///
    /// Both arms are called explicitly rather than through `delta_scan`, so this
    /// grades both kernels whichever way `XWEN_DELTA_DECODE_KERNEL` is set —
    /// which matters more now that the decode kernel is opt-in and a default
    /// test run never reaches it through the router.
    #[test]
    fn decode_scan_matches_the_general_kernel_and_the_reference() {
        let device = metal_device().unwrap();
        for (ki, &(k_heads, v_heads)) in GEOMETRIES.iter().enumerate() {
            let conv_dim = (2 * k_heads + v_heads) * HD;
            let seed = 0x4700 + ki as u64 * 313;
            let s0 = on_device(
                pseudo_random(v_heads * HD * HD, seed, -0.5, 0.5),
                (v_heads, HD, HD),
                &device,
            );
            let (mut s_dec, mut s_gen, mut s_ref) = (s0.clone(), s0.clone(), s0.clone());

            for step in 0..4u64 {
                let seed = seed + 1 + step * 7;
                let conv = on_device(
                    pseudo_random(conv_dim, seed, -2.0, 2.0),
                    (1, conv_dim),
                    &device,
                );
                let beta = on_device(
                    pseudo_random(v_heads, seed + 1, 0.01, 0.99),
                    (1, v_heads),
                    &device,
                );
                let g = on_device(
                    pseudo_random(v_heads, seed + 2, -0.6, -0.001),
                    (1, v_heads),
                    &device,
                );

                let (o_dec, next_dec) = dispatch::run_delta_scan_decode(
                    &conv, &beta, &g, &s_dec, k_heads, EPS as f32, 1,
                )
                .unwrap();
                let (o_gen, next_gen) = dispatch::run_delta_scan_default(
                    &conv, &beta, &g, &s_gen, k_heads, EPS as f32, 1,
                )
                .unwrap();
                let (o_ref, next_ref) = reference_scan(&conv, &beta, &g, &s_ref, k_heads, v_heads);

                let label = format!("decode k={k_heads} v={v_heads} step={step}");
                assert_eq!(o_dec.dims(), &[1, v_heads, HD]);
                assert_eq!(next_dec.dims(), &[1, v_heads, HD, HD]);
                s_dec = next_dec.squeeze(0).unwrap();
                s_gen = next_gen.squeeze(0).unwrap();
                assert_close(&o_dec, &o_gen, 1e-5, &format!("{label} out vs general"));
                assert_close(&s_dec, &s_gen, 1e-5, &format!("{label} state vs general"));
                assert_close(&o_dec, &o_ref, 1e-5, &format!("{label} out vs reference"));
                assert_close(
                    &s_dec,
                    &next_ref,
                    1e-5,
                    &format!("{label} state vs reference"),
                );
                s_ref = next_ref;
            }
        }
    }

    /// The decode kernel writes a fresh state buffer, like the general one: the
    /// state it was handed comes back unchanged, and a re-run of the same
    /// dispatch reproduces the same bits. A rollback trail holds the incoming
    /// state (and, once armed, every state the trail recorded), so an in-place
    /// update here would corrupt a checkpoint rather than fail a test.
    #[test]
    fn decode_scan_does_not_mutate_the_incoming_state() {
        let device = metal_device().unwrap();
        let (k_heads, v_heads) = (16usize, 48usize);
        let conv_dim = (2 * k_heads + v_heads) * HD;
        let conv = on_device(
            pseudo_random(conv_dim, 0x5100, -2.0, 2.0),
            (1, conv_dim),
            &device,
        );
        let beta = on_device(
            pseudo_random(v_heads, 0x5101, 0.01, 0.99),
            (1, v_heads),
            &device,
        );
        let g = on_device(
            pseudo_random(v_heads, 0x5102, -0.6, -0.001),
            (1, v_heads),
            &device,
        );
        let s0 = on_device(
            pseudo_random(v_heads * HD * HD, 0x5103, -0.5, 0.5),
            (v_heads, HD, HD),
            &device,
        );
        let before: Vec<f32> = s0.flatten_all().unwrap().to_vec1().unwrap();

        let (o1, s1) =
            dispatch::run_delta_scan_decode(&conv, &beta, &g, &s0, k_heads, EPS as f32, 1).unwrap();
        let after: Vec<f32> = s0.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(before, after, "the incoming state was overwritten");

        let (o2, s2) =
            dispatch::run_delta_scan_decode(&conv, &beta, &g, &s0, k_heads, EPS as f32, 1).unwrap();
        assert_bits_eq(&o2, &o1, "rerun output");
        assert_bits_eq(&s2, &s1, "rerun state");
    }

    /// The decode kernel is the seq == 1 kernel and says so rather than
    /// mis-indexing: a longer chunk, and a plane count no one-token chunk can
    /// name, are refused. `run_delta_scan` never sends it either, but the
    /// entry point is `pub(crate)` and the next caller should not have to know.
    #[test]
    fn decode_scan_refuses_a_multi_token_chunk() {
        let device = metal_device().unwrap();
        let (k_heads, v_heads) = (16usize, 32usize);
        let conv_dim = (2 * k_heads + v_heads) * HD;
        let s0 = Tensor::zeros((v_heads, HD, HD), DType::F32, &device).unwrap();
        let rows = |seq: usize| {
            (
                Tensor::zeros((seq, conv_dim), DType::F32, &device).unwrap(),
                Tensor::zeros((seq, v_heads), DType::F32, &device).unwrap(),
            )
        };
        let (conv, bg) = rows(3);
        assert!(
            dispatch::run_delta_scan_decode(&conv, &bg, &bg, &s0, k_heads, EPS as f32, 1).is_err(),
            "the decode kernel accepted a three-token chunk"
        );
        let (conv1, bg1) = rows(1);
        assert!(
            dispatch::run_delta_scan_decode(&conv1, &bg1, &bg1, &s0, k_heads, EPS as f32, 2)
                .is_err(),
            "the decode kernel accepted two state planes for one token"
        );
        assert!(
            dispatch::run_delta_scan_decode(&conv1, &bg1, &bg1, &s0, k_heads, EPS as f32, 1)
                .is_ok()
        );
    }

    /// The scan leaves the state it was handed untouched: the kernel writes a
    /// fresh buffer rather than updating in place, which is what lets the
    /// caller keep the incoming state alive for a rollback trail (and what
    /// makes a re-run of the same dispatch reproducible).
    #[test]
    fn scan_does_not_mutate_the_incoming_state() {
        let device = metal_device().unwrap();
        let (k_heads, v_heads) = (16usize, 32usize);
        let conv_dim = (2 * k_heads + v_heads) * HD;
        let seq = 4usize;
        let conv = on_device(
            pseudo_random(seq * conv_dim, 0x5000, -2.0, 2.0),
            (seq, conv_dim),
            &device,
        );
        let beta = on_device(
            pseudo_random(seq * v_heads, 0x5001, 0.01, 0.99),
            (seq, v_heads),
            &device,
        );
        let g = on_device(
            pseudo_random(seq * v_heads, 0x5002, -0.6, -0.001),
            (seq, v_heads),
            &device,
        );
        let s0 = on_device(
            pseudo_random(v_heads * HD * HD, 0x5003, -0.5, 0.5),
            (v_heads, HD, HD),
            &device,
        );
        let before: Vec<f32> = s0.flatten_all().unwrap().to_vec1().unwrap();

        let (o1, s1) = delta_scan(&conv, &beta, &g, &s0, k_heads, EPS as f32).unwrap();
        let after: Vec<f32> = s0.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(before, after, "the incoming state was overwritten");

        let (o2, s2) = delta_scan(&conv, &beta, &g, &s0, k_heads, EPS as f32).unwrap();
        assert_bits_eq(&o2, &o1, "rerun output");
        assert_bits_eq(&s2, &s1, "rerun state");
    }

    /// Stepping a chunk one token at a time through the carried state must land
    /// exactly where one batched call lands — the property decode-after-prefill
    /// depends on, and the one a scan that mishandled its register-resident
    /// state across timesteps would break.
    #[test]
    fn scan_streams_to_the_same_place_as_one_batch() {
        let device = metal_device().unwrap();
        let (k_heads, v_heads) = (16usize, 48usize);
        let conv_dim = (2 * k_heads + v_heads) * HD;
        let seq = 9usize;
        let conv = on_device(
            pseudo_random(seq * conv_dim, 0x6000, -2.0, 2.0),
            (seq, conv_dim),
            &device,
        );
        let beta = on_device(
            pseudo_random(seq * v_heads, 0x6001, 0.01, 0.99),
            (seq, v_heads),
            &device,
        );
        let g = on_device(
            pseudo_random(seq * v_heads, 0x6002, -0.6, -0.001),
            (seq, v_heads),
            &device,
        );
        let s0 = Tensor::zeros((v_heads, HD, HD), DType::F32, &device).unwrap();

        let (batched, batched_s) = delta_scan(&conv, &beta, &g, &s0, k_heads, EPS as f32).unwrap();

        let mut s = s0.clone();
        let mut rows = Vec::with_capacity(seq);
        for t in 0..seq {
            let row = |x: &Tensor| x.narrow(0, t, 1).unwrap().contiguous().unwrap();
            let (o, next) =
                delta_scan(&row(&conv), &row(&beta), &row(&g), &s, k_heads, EPS as f32).unwrap();
            rows.push(o);
            s = next;
        }
        let streamed = Tensor::cat(&rows, 0).unwrap();
        assert_close(&streamed, &batched, 1e-6, "streamed vs batched output");
        assert_close(&s, &batched_s, 1e-6, "streamed vs batched state");
    }

    /// Shape, dtype and geometry contracts. Each op refuses rather than
    /// silently mis-indexing: the scan is specialized to head dim 128, needs a
    /// V-head count that is a whole multiple of the K-head count (the tiling),
    /// and needs a conv width that matches the split it is about to make.
    #[test]
    fn shape_and_geometry_errors() {
        let device = metal_device().unwrap();
        let z1 = |d: (usize, usize)| Tensor::zeros(d, DType::F32, &device).unwrap();
        let z3 = |d: (usize, usize, usize)| Tensor::zeros(d, DType::F32, &device).unwrap();

        // conv: the carried window must be taps-1 rows of the same width.
        let qkv = z1((4, 64));
        let w = z1((4, 64));
        assert!(delta_conv(&z1((2, 64)), &qkv, &w).is_err());
        assert!(delta_conv(&z1((3, 32)), &qkv, &w).is_err());
        assert!(delta_conv(&z1((3, 64)), &qkv, &z1((4, 32))).is_err());

        // ba: an odd column count cannot split into beta|alpha.
        let a = Tensor::zeros(4usize, DType::F32, &device).unwrap();
        assert!(delta_ba(&z1((2, 7)), &a, &a).is_err());
        assert!(
            delta_ba(
                &z1((2, 8)),
                &a,
                &Tensor::zeros(5usize, DType::F32, &device).unwrap()
            )
            .is_err()
        );

        // gnorm: the head dim must tile whole simdgroups, and z must match o.
        let w128 = Tensor::zeros(HD, DType::F32, &device).unwrap();
        assert!(
            delta_gnorm(
                &z3((2, 3, 40)),
                &z3((2, 3, 40)),
                &Tensor::zeros(40usize, DType::F32, &device).unwrap(),
                1e-6,
                ZGate::Silu
            )
            .is_err()
        );
        assert!(delta_gnorm(&z3((2, 3, HD)), &z3((2, 4, HD)), &w128, 1e-6, ZGate::Silu).is_err());

        // l2norm: the conv row must be wide enough to hold the q|k planes it is
        // asked to normalize.
        assert!(delta_l2norm(&z1((2, 2 * 16 * HD - 1)), 16, 1e-6).is_err());
        assert!(delta_l2norm(&z1((2, 2 * 16 * HD)), 0, 1e-6).is_err());

        // scan: head dim, head-count divisibility, conv width.
        let s_ok = z3((32, HD, HD));
        let conv_ok = z1((2, (2 * 16 + 32) * HD));
        let beta_ok = z1((2, 32));
        assert!(delta_scan(&conv_ok, &beta_ok, &beta_ok, &z3((32, 64, 64)), 16, 1e-6).is_err());
        assert!(delta_scan(&conv_ok, &beta_ok, &beta_ok, &s_ok, 12, 1e-6).is_err());
        assert!(delta_scan(&z1((2, 128)), &beta_ok, &beta_ok, &s_ok, 16, 1e-6).is_err());
        assert!(delta_scan(&conv_ok, &z1((2, 31)), &beta_ok, &s_ok, 16, 1e-6).is_err());

        // An empty chunk encodes a zero-dimension grid, which dispatches
        // nothing and hands the caller a shape it did not ask for. Every op
        // refuses it.
        assert!(delta_conv(&z1((3, 64)), &z1((0, 64)), &w).is_err());
        assert!(delta_ba(&z1((0, 8)), &a, &a).is_err());
        assert!(delta_gnorm(&z3((0, 3, HD)), &z3((0, 3, HD)), &w128, 1e-6, ZGate::Silu).is_err());
        assert!(delta_l2norm(&z1((0, 2 * 16 * HD)), 16, 1e-6).is_err());
        assert!(delta_gnorm(&z3((2, 0, HD)), &z3((2, 0, HD)), &w128, 1e-6, ZGate::Silu).is_err());
        assert!(
            delta_scan(
                &z1((0, (2 * 16 + 32) * HD)),
                &z1((0, 32)),
                &z1((0, 32)),
                &s_ok,
                16,
                1e-6
            )
            .is_err()
        );
    }

    /// The scan kernel's threadgroup geometry is spelled out twice: as
    /// `#define`s in delta.metal, which index the state slice, and as Rust
    /// constants in dispatch.rs, which size the grid. Drift between the two is
    /// silent — the kernel would index state rows the grid never covers, or
    /// threadgroups would write past a head's slice — so parse the kernel's
    /// numbers out of the source and compare.
    /// The integer in delta.metal's `#define <name> <int>`, ignoring any
    /// trailing comment.
    fn define(name: &str) -> usize {
        const SRC: &str = include_str!("delta.metal");
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
            .unwrap_or_else(|| panic!("delta.metal has no `#define {name} <integer>`"))
    }

    /// The fused beta|alpha projection's threadgroup shape is spelled out
    /// twice — as `#define`s in delta.metal, which size its partial-sum buffer
    /// and its reduction tree, and as Rust constants in dispatch.rs, which size
    /// the grid and the threadgroup. Drift between the two is silent: columns
    /// or tokens the grid never covers are simply never written.
    #[test]
    fn ba_fused_geometry_matches_metal() {
        assert_eq!(
            define("DELTA_BA_COLS"),
            dispatch::DELTA_BA_COLS,
            "delta.metal DELTA_BA_COLS and dispatch.rs DELTA_BA_COLS disagree; \
             the launch's column blocks are sized from the Rust copy"
        );
        assert_eq!(
            define("DELTA_BA_ROWS"),
            dispatch::DELTA_BA_ROWS,
            "delta.metal DELTA_BA_ROWS and dispatch.rs DELTA_BA_ROWS disagree; \
             the threadgroup width is sized from the Rust copy"
        );
        assert_eq!(
            define("DELTA_BA_TOKS"),
            dispatch::DELTA_BA_TOKS,
            "delta.metal DELTA_BA_TOKS and dispatch.rs DELTA_BA_TOKS disagree; \
             the launch's token tiles are sized from the Rust copy"
        );
        // The same relations the kernel's own static_asserts hold it to.
        let (cols, rows, toks) = (
            dispatch::DELTA_BA_COLS,
            dispatch::DELTA_BA_ROWS,
            dispatch::DELTA_BA_TOKS,
        );
        assert!(cols * rows <= 1024, "a threadgroup is at most 1024 threads");
        assert_eq!(rows & (rows - 1), 0, "the reduction tree halves the rows");
        assert!(
            toks * rows * cols * 4 <= 32768,
            "the partial buffer must fit threadgroup memory"
        );
    }

    #[test]
    fn scan_geometry_matches_metal() {
        let d = define("DELTA_D");
        let cols = define("DELTA_TG_COLS");
        let rows = define("DELTA_TG_ROWS");
        let slice = define("DELTA_S_SLICE");
        let col_blocks = define("DELTA_COL_BLOCKS");
        let dec_vec = define("DELTA_DEC_VEC");
        let dec_lanes = define("DELTA_DEC_LANES");
        let dec_rows = define("DELTA_DEC_ROWS");
        let dec_slice = define("DELTA_DEC_SLICE");
        let dec_cols = define("DELTA_DEC_TG_COLS");
        let dec_blocks = define("DELTA_DEC_COL_BLOCKS");
        let dec_sgs = define("DELTA_DEC_SGS");
        let kpl = define("DELTA_V2_KPL");
        let sgs = define("DELTA_V2_SGS");
        let col_tgs = define("DELTA_V2_COL_TGS");

        assert_eq!(
            d, DELTA_HEAD_DIM,
            "delta.metal DELTA_D and dispatch.rs DELTA_HEAD_DIM disagree"
        );
        assert_eq!(
            col_blocks,
            dispatch::DELTA_COL_BLOCKS,
            "delta.metal DELTA_COL_BLOCKS and dispatch.rs DELTA_COL_BLOCKS disagree; \
             the v1 scan grid is sized from the Rust copy"
        );
        assert_eq!(
            dec_blocks,
            dispatch::DELTA_DEC_COL_BLOCKS,
            "delta.metal DELTA_DEC_COL_BLOCKS and dispatch.rs DELTA_DEC_COL_BLOCKS disagree; \
             the decode scan grid is sized from the Rust copy"
        );
        assert_eq!(
            sgs,
            dispatch::DELTA_V2_SGS,
            "delta.metal DELTA_V2_SGS and dispatch.rs DELTA_V2_SGS disagree; \
             the v2 threadgroup is sized from the Rust copy"
        );
        assert_eq!(
            col_tgs,
            dispatch::DELTA_V2_COL_TGS,
            "delta.metal DELTA_V2_COL_TGS and dispatch.rs DELTA_V2_COL_TGS disagree; \
             the v2 scan grid is sized from the Rust copy"
        );
        // The same relations the kernels' own static_asserts hold them to, so a
        // `#define` edit that broke them is caught here too and not only by the
        // Metal compiler on a device.
        assert_eq!(rows * cols, d, "the v1 threadgroup must be DELTA_D threads");
        assert_eq!(slice, d / rows, "v1 state rows per thread");
        assert_eq!(col_blocks, d / cols, "v1 threadgroups per head");
        assert_eq!(
            dec_vec * dec_lanes,
            dec_cols,
            "a decode threadgroup's float4 groups must tile its columns"
        );
        assert_eq!(
            dec_rows * dec_lanes,
            d,
            "the decode threadgroup must be DELTA_D threads"
        );
        assert_eq!(dec_slice, d / dec_rows, "decode state rows per thread");
        assert_eq!(dec_blocks, d / dec_cols, "decode threadgroups per head");
        assert_eq!(dec_sgs * 32, d, "decode simdgroups per threadgroup");
        assert_eq!(
            32 % dec_lanes,
            0,
            "a simdgroup must hold a whole number of decode column groups"
        );
        assert_eq!(kpl * 32, d, "a v2 simdgroup's lanes must cover the key dim");
        assert_eq!(col_tgs * sgs, d, "v2 threadgroups per head");
    }

    /// Isolation timing for the recurrent scan (plus its q/k norm) at both
    /// production geometries and both prefill chunk lengths the bench harness
    /// uses. Multiply a number here by the model's DeltaNet layer count (27B: 48,
    /// 35B-A3B: 30) to get the scan's share of one prefill, which is what says
    /// whether the scan is worth optimizing at all.
    ///
    /// The decomposition is chosen by the cached `XWEN_DELTA_SCAN_V2` switch, so
    /// one process times one arm; run it twice to compare. `#[ignore]`d — run on
    /// a `pgrep`-verified free GPU with:
    ///   cargo test --release -p xwen delta_scan_timing -- --ignored --nocapture
    /// `XWEN_BENCH_WARMUP` / `XWEN_BENCH_ITERS` override the loop counts.
    #[test]
    #[ignore = "perf bench"]
    fn delta_scan_timing() {
        use std::time::Instant;

        let device = metal_device().unwrap();
        let get = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let (warm, iters) = (get("XWEN_BENCH_WARMUP", 5), get("XWEN_BENCH_ITERS", 20));
        let arm = if crate::ops::delta_scan_v2() {
            "v2     "
        } else {
            "shipped"
        };

        for &(k_heads, v_heads) in GEOMETRIES.iter() {
            let conv_dim = (2 * k_heads + v_heads) * HD;
            for &seq in &[880usize, 4096] {
                let seed = 0x9000 + seq as u64;
                let conv = on_device(
                    pseudo_random(seq * conv_dim, seed, -2.0, 2.0),
                    (seq, conv_dim),
                    &device,
                );
                let beta = on_device(
                    pseudo_random(seq * v_heads, seed + 1, 0.01, 0.99),
                    (seq, v_heads),
                    &device,
                );
                let g = on_device(
                    pseudo_random(seq * v_heads, seed + 2, -0.6, -0.001),
                    (seq, v_heads),
                    &device,
                );
                let s0 = on_device(
                    pseudo_random(v_heads * HD * HD, seed + 3, -0.5, 0.5),
                    (v_heads, HD, HD),
                    &device,
                );

                // Each iteration ends by waiting on the device rather than by
                // reading the output back: these ops produce tens of megabytes,
                // and a readback would time the memcpy as much as the kernel.
                let Device::Metal(mdev) = &device else {
                    unreachable!("metal_device() returned a non-Metal device")
                };
                let bench = |label: &str, f: &mut dyn FnMut()| {
                    for _ in 0..warm {
                        f();
                        mdev.wait_until_completed().unwrap();
                    }
                    let mut times = Vec::with_capacity(iters);
                    for _ in 0..iters {
                        let t = Instant::now();
                        f();
                        mdev.wait_until_completed().unwrap();
                        times.push(t.elapsed().as_secs_f64() * 1e3);
                    }
                    let mean = times.iter().sum::<f64>() / times.len() as f64;
                    let plateau: f64 =
                        times[iters / 2..].iter().sum::<f64>() / (iters - iters / 2) as f64;
                    eprintln!(
                        "{label} {arm} k={k_heads} v={v_heads} seq={seq}: mean {mean:.3} ms | \
                         plateau {plateau:.3} ms"
                    );
                };

                bench("scan  ", &mut || {
                    delta_scan(&conv, &beta, &g, &s0, k_heads, EPS as f32).unwrap();
                });
                // The norm alone, so the hoisted arms' extra dispatch can be
                // priced against what removing it from the timestep loop saves.
                bench("l2norm", &mut || {
                    delta_l2norm(&conv, k_heads, EPS as f32).unwrap();
                });
            }
        }
    }

    /// Isolation timing for the DECODE scan: the seq == 1 step every decoded
    /// token runs once per DeltaNet layer, and at Flash-Next's 36 of them the
    /// largest single share of the GDN mixer (`XWEN_GDN_PROFILE`).
    ///
    /// Both kernels in ONE process — they are called directly rather than
    /// through the cached `XWEN_DELTA_DECODE_KERNEL` switch — and the shape is
    /// a real token's: `XWEN_BENCH_LAYERS` distinct states, each advanced once
    /// per iteration, so no state is warm from the dispatch before it and the
    /// rate is AMORTIZED over a token's worth of dispatches instead of measured
    /// one dispatch at a time (CLAUDE.md's benching rules).
    ///
    /// `#[ignore]`d — run on a `pgrep`-verified free GPU with:
    ///   cargo test --release -p xwen delta_scan_decode_timing -- --ignored --nocapture
    #[test]
    #[ignore = "perf bench"]
    fn delta_scan_decode_timing() {
        use std::time::Instant;

        let device = metal_device().unwrap();
        let Device::Metal(mdev) = &device else {
            unreachable!("metal_device() returned a non-Metal device")
        };
        let get = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let (warm, iters) = (get("XWEN_BENCH_WARMUP", 5), get("XWEN_BENCH_ITERS", 20));
        let layers = get("XWEN_BENCH_LAYERS", 36);

        for &(k_heads, v_heads) in GEOMETRIES.iter() {
            let conv_dim = (2 * k_heads + v_heads) * HD;
            let seed = 0x9500 + v_heads as u64;
            let conv = on_device(
                pseudo_random(conv_dim, seed, -2.0, 2.0),
                (1, conv_dim),
                &device,
            );
            let beta = on_device(
                pseudo_random(v_heads, seed + 1, 0.01, 0.99),
                (1, v_heads),
                &device,
            );
            let g = on_device(
                pseudo_random(v_heads, seed + 2, -0.6, -0.001),
                (1, v_heads),
                &device,
            );
            // The state read once and written once per layer, plus the streaming
            // operands — the same floor `LinearAttnBlock::byte_floors` declares.
            let bytes = 4.0
                * layers as f64
                * (2.0 * (v_heads * HD * HD) as f64
                    + conv_dim as f64
                    + (v_heads * HD) as f64
                    + 2.0 * v_heads as f64);

            // The floor arm: candle's affine over the same state, which reads
            // and writes exactly the bytes the scan does and computes nothing.
            // A scan arm sitting on this number is bandwidth-bound and not
            // worth another decomposition.
            {
                let mut states: Vec<Tensor> = (0..layers)
                    .map(|l| {
                        on_device(
                            pseudo_random(v_heads * HD * HD, seed + 100 + l as u64, -0.5, 0.5),
                            (v_heads, HD, HD),
                            &device,
                        )
                    })
                    .collect();
                let mut token = || {
                    for l in 0..layers {
                        states[l] = states[l].affine(1.0, 0.0).unwrap();
                    }
                };
                for _ in 0..warm {
                    token();
                    mdev.wait_until_completed().unwrap();
                }
                let mut times = Vec::with_capacity(iters);
                for _ in 0..iters {
                    let t = Instant::now();
                    token();
                    mdev.wait_until_completed().unwrap();
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                }
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let (median, best) = (times[iters / 2], times[0]);
                let state_bytes = 4.0 * layers as f64 * 2.0 * (v_heads * HD * HD) as f64;
                eprintln!(
                    "decode-scan floor   k={k_heads} v={v_heads} layers={layers}: \
                     median {median:.3} ms/token ({:.1} GB/s) | best {best:.3} ms ({:.1} GB/s)",
                    state_bytes / (median * 1e6),
                    state_bytes / (best * 1e6),
                );
            }

            for (label, decode) in [("general", false), ("decode ", true)] {
                let mut states: Vec<Tensor> = (0..layers)
                    .map(|l| {
                        on_device(
                            pseudo_random(v_heads * HD * HD, seed + 100 + l as u64, -0.5, 0.5),
                            (v_heads, HD, HD),
                            &device,
                        )
                    })
                    .collect();
                let mut token = || {
                    for l in 0..layers {
                        let next = if decode {
                            dispatch::run_delta_scan_decode(
                                &conv, &beta, &g, &states[l], k_heads, EPS as f32, 1,
                            )
                        } else {
                            dispatch::run_delta_scan_default(
                                &conv, &beta, &g, &states[l], k_heads, EPS as f32, 1,
                            )
                        }
                        .unwrap()
                        .1;
                        states[l] = next.squeeze(0).unwrap();
                    }
                };
                for _ in 0..warm {
                    token();
                    mdev.wait_until_completed().unwrap();
                }
                let mut times = Vec::with_capacity(iters);
                for _ in 0..iters {
                    let t = Instant::now();
                    token();
                    mdev.wait_until_completed().unwrap();
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                }
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = times[iters / 2];
                let best = times[0];
                eprintln!(
                    "decode-scan {label} k={k_heads} v={v_heads} layers={layers}: \
                     median {median:.3} ms/token ({:.1} GB/s) | best {best:.3} ms ({:.1} GB/s)",
                    bytes / (median * 1e6),
                    bytes / (best * 1e6),
                );
            }
        }
    }

    /// Isolation timing for the two beta|alpha arms — the fused one-dispatch
    /// kernel against the candle gemv plus `delta_ba` it replaces — at every
    /// shipped geometry, across the token counts that decide where
    /// `DELTA_BA_MAX_SEQ` belongs. The fused kernel reads the whole weight once
    /// per token tile and candle's gemm reads it once per chunk, so the fused
    /// arm's advantage decays with n and the crossover is what this measures.
    ///
    /// Multiply a per-call figure by the model's DeltaNet layer count (27B: 48,
    /// 35B-A3B: 30, Flash-Next: 36) for the step's share of one token.
    ///
    /// The FIRST cell printed reads several times high in both arms and is not
    /// a measurement of the geometry it names — the process's first delta
    /// dispatches walk a cold buffer pool. Read from the second geometry on, or
    /// compare cells at the same position across runs.
    ///
    /// `#[ignore]`d — run on a `pgrep`-verified free GPU with:
    ///   cargo test --release -p xwen delta_ba_timing -- --ignored --nocapture
    #[test]
    #[ignore = "perf bench"]
    fn delta_ba_timing() {
        use std::time::Instant;

        let device = metal_device().unwrap();
        let get = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let (warm, iters) = (get("XWEN_BENCH_WARMUP", 20), get("XWEN_BENCH_ITERS", 100));
        let Device::Metal(mdev) = &device else {
            unreachable!("metal_device() returned a non-Metal device")
        };

        for &(hidden, v_heads) in BA_GEOMETRIES.iter() {
            for &seq in &[1usize, 2, 4, 8, 16, 32] {
                let (x, w, ssm_a, dt_bias) =
                    ba_operands(hidden, v_heads, seq, 0x9100 + seq as u64, &device);
                // Amortized: the whole batch of iterations is one sync, so a
                // per-call figure is not one command buffer's round trip.
                let bench = |label: &str, f: &mut dyn FnMut()| {
                    for _ in 0..warm {
                        f();
                    }
                    mdev.wait_until_completed().unwrap();
                    let t = Instant::now();
                    for _ in 0..iters {
                        f();
                    }
                    mdev.wait_until_completed().unwrap();
                    let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
                    eprintln!("{label} hidden={hidden} v={v_heads} seq={seq}: {us:.2} us/call");
                };

                bench("ba fused ", &mut || {
                    delta_ba_fused(&x, &w, &ssm_a, &dt_bias).unwrap();
                });
                bench("ba classic", &mut || {
                    let ba = x.matmul(&w).unwrap();
                    delta_ba(&ba, &ssm_a, &dt_bias).unwrap();
                });
            }
        }
    }

    /// Every op resolves its operands via `start_offset * dtype_size`; the other
    /// tests build inputs with `Tensor::from_vec` (offset 0). Feed each op a
    /// CONTIGUOUS view that starts mid-buffer and compare against the same view
    /// through the reference — a dropped offset would read the buffer head.
    #[test]
    fn delta_ops_handle_offset_views() {
        let device = metal_device().unwrap();
        let (k_heads, v_heads) = (16usize, 32usize);
        let conv_dim = (2 * k_heads + v_heads) * HD;
        let (seq, taps) = (5usize, 4usize);

        let offset = |rows: usize, cols: usize, skip: usize, seed: u64, lo: f32, hi: f32| {
            let big = on_device(
                pseudo_random((rows + skip) * cols, seed, lo, hi),
                (rows + skip, cols),
                &device,
            );
            let view = big.narrow(0, skip, rows).unwrap();
            assert!(view.is_contiguous(), "narrowed view must stay contiguous");
            view
        };

        let state = offset(taps - 1, conv_dim, 2, 0x7000, -3.0, 3.0);
        let qkv = offset(seq, conv_dim, 3, 0x7001, -3.0, 3.0);
        let w = offset(taps, conv_dim, 1, 0x7002, -1.0, 1.0);
        let (got, got_state) = delta_conv(&state, &qkv, &w).unwrap();
        let (want, want_state) = reference_conv(&state, &qkv, &w, taps);
        assert_bits_eq(&got, &want, "offset conv");
        assert_bits_eq(&got_state, &want_state, "offset conv window");

        let ba = offset(seq, 2 * v_heads, 2, 0x7003, -20.0, 20.0);
        let a_big = on_device(
            pseudo_random(v_heads + 4, 0x7004, -3.0, -0.1),
            v_heads + 4,
            &device,
        );
        let ssm_a = a_big.narrow(0, 4, v_heads).unwrap();
        let dt_big = on_device(
            pseudo_random(v_heads + 5, 0x7005, -2.0, 2.0),
            v_heads + 5,
            &device,
        );
        let dt_bias = dt_big.narrow(0, 5, v_heads).unwrap();
        let (beta_out, g_out) = delta_ba(&ba, &ssm_a, &dt_bias).unwrap();
        let b_raw = ba.narrow(1, 0, v_heads).unwrap().contiguous().unwrap();
        let a_raw = ba
            .narrow(1, v_heads, v_heads)
            .unwrap()
            .contiguous()
            .unwrap();
        assert_bits_eq(&beta_out, &sigmoid(&b_raw).unwrap(), "offset ba beta");
        assert_bits_eq(
            &g_out,
            &softplus(&a_raw.broadcast_add(&dt_bias).unwrap())
                .unwrap()
                .broadcast_mul(&ssm_a)
                .unwrap(),
            "offset ba g",
        );

        // The fused projection, whose four operands all resolve through
        // `start_offset`: the layer input, the weight, and both per-head
        // vectors. Graded against the same views through the gemv chain.
        let hidden = 512usize;
        let xv = offset(seq, hidden, 2, 0x7010, -2.0, 2.0);
        let wv = offset(hidden, 2 * v_heads, 3, 0x7011, -0.08, 0.08);
        let (fb, fg) = delta_ba_fused(&xv, &wv, &ssm_a, &dt_bias).unwrap();
        let (wb, wg) = reference_ba_chain(&xv, &wv, &ssm_a, &dt_bias);
        assert_close(&fb, &wb, 1e-6, "offset ba_fused beta");
        assert_close(&fg, &wg, 1e-6, "offset ba_fused g");

        let conv = offset(seq, conv_dim, 2, 0x7006, -2.0, 2.0);
        let beta = offset(seq, v_heads, 3, 0x7007, 0.01, 0.99);
        let g = offset(seq, v_heads, 1, 0x7008, -0.6, -0.001);
        let s_big = Tensor::from_vec(
            pseudo_random((v_heads + 2) * HD * HD, 0x7009, -0.5, 0.5),
            (v_heads + 2, HD, HD),
            &device,
        )
        .unwrap();
        let s0 = s_big.narrow(0, 2, v_heads).unwrap();
        assert!(s0.is_contiguous(), "narrowed state must stay contiguous");
        let (o, s) = delta_scan(&conv, &beta, &g, &s0, k_heads, EPS as f32).unwrap();
        let (want_o, want_s) = reference_scan(&conv, &beta, &g, &s0, k_heads, v_heads);
        assert_close(&o, &want_o, 1e-5, "offset scan out");
        assert_close(&s, &want_s, 1e-5, "offset scan state");

        let o_view = offset(seq * v_heads, HD, 4, 0x700a, -6.0, 6.0)
            .reshape((seq, v_heads, HD))
            .unwrap();
        let z_view = offset(seq * v_heads, HD, 6, 0x700b, -8.0, 8.0)
            .reshape((seq, v_heads, HD))
            .unwrap();
        let w_big = on_device(pseudo_random(HD + 3, 0x700c, 0.2, 2.0), HD + 3, &device);
        let nw = w_big.narrow(0, 3, HD).unwrap();
        let got = delta_gnorm(&o_view, &z_view, &nw, EPS as f32, ZGate::Silu).unwrap();
        let ms = (o_view.sqr().unwrap().sum_keepdim(2).unwrap() / HD as f64).unwrap();
        let want = (o_view
            .broadcast_div(&(ms + EPS).unwrap().sqrt().unwrap())
            .unwrap()
            .broadcast_mul(&nw.reshape((1, 1, HD)).unwrap())
            .unwrap()
            * silu(&z_view).unwrap())
        .unwrap();
        assert_close(&got, &want, 2e-6, "offset gnorm");
    }
}
