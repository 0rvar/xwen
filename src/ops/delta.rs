//! Host side of the fused gated-DeltaNet ops (conv+silu+state, beta/decay head,
//! recurrent scan, gated output norm). Kernel-side rounding contracts live in
//! delta.metal: the conv and beta/decay kernels are bit-identical to the candle
//! chains they replace, while the gated norm and the scan are bounded — each
//! partitions across threads a reduction the reference runs in one order
//! (simd_sum over the head dim, and the scan's contractions where the reference
//! runs a gemm).

use anyhow::Result;
use candle_core::Tensor;

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

/// Gated output RMSNorm: `rms_norm(o, eps) * ssm_norm_weight * silu(z)` per
/// head, one pass. `o` and `z` are `[seq, v_heads, head_dim]` f32 contiguous,
/// `w` is `[head_dim]`. The gate multiplies AFTER the weight and outside the
/// norm, so it never enters the statistic.
pub fn delta_gnorm(o: &Tensor, z: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    dispatch::run_delta_gnorm(o, z, w, eps)
}

/// The delta-rule recurrence in ONE dispatch, however long the chunk:
/// `S *= exp(g); d = (v - k·S) * beta; S += k (x) d; o = q·S / sqrt(d_k)` per
/// timestep. `conv` `[seq, conv_dim]` is the silu'd conv output (q | k | v
/// fused) — q and k are read from it with the tiled K-head mapping and L2
/// clamp-normalized (`x / max(||x||, eps)`) inside the kernel, so no
/// materialized tile-and-broadcast is needed. `beta`/`g` are `[seq, v_heads]`,
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
    dispatch::run_delta_scan(conv, beta, g, s, k_heads, eps)
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

                let got = delta_gnorm(&o, &z, &w, EPS as f32).unwrap();

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
        }
        (Tensor::cat(&outs, 0).unwrap(), s)
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
            for &seq in &[1usize, 2, 33, 512] {
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
                1e-6
            )
            .is_err()
        );
        assert!(delta_gnorm(&z3((2, 3, HD)), &z3((2, 4, HD)), &w128, 1e-6).is_err());

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
        assert!(delta_gnorm(&z3((0, 3, HD)), &z3((0, 3, HD)), &w128, 1e-6).is_err());
        assert!(delta_gnorm(&z3((2, 0, HD)), &z3((2, 0, HD)), &w128, 1e-6).is_err());
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
    #[test]
    fn scan_geometry_matches_metal() {
        const SRC: &str = include_str!("delta.metal");

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
                .unwrap_or_else(|| panic!("delta.metal has no `#define {name} <integer>`"))
        }

        let d = define("DELTA_D");
        let cols = define("DELTA_TG_COLS");
        let rows = define("DELTA_TG_ROWS");
        let slice = define("DELTA_S_SLICE");
        let col_blocks = define("DELTA_COL_BLOCKS");

        assert_eq!(
            d, DELTA_HEAD_DIM,
            "delta.metal DELTA_D and dispatch.rs DELTA_HEAD_DIM disagree"
        );
        assert_eq!(
            col_blocks,
            dispatch::DELTA_COL_BLOCKS,
            "delta.metal DELTA_COL_BLOCKS and dispatch.rs DELTA_COL_BLOCKS disagree; \
             the scan grid is sized from the Rust copy"
        );
        // The same relations the kernel's own static_asserts hold it to, so a
        // `#define` edit that broke them is caught here too and not only by the
        // Metal compiler on a device.
        assert_eq!(rows * cols, d, "the threadgroup must be DELTA_D threads");
        assert_eq!(slice, d / rows, "state rows per thread");
        assert_eq!(col_blocks, d / cols, "threadgroups per head");
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
        let got = delta_gnorm(&o_view, &z_view, &nw, EPS as f32).unwrap();
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
