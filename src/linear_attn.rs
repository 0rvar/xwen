//! Gated DeltaNet — the linear-attention layers, three of every four.
//!
//! This is the FROZEN CORRECTNESS ORACLE for the recurrence, in the same sense
//! `ReferenceExperts` is for the MoE FFN: composed candle ops, the recurrent
//! (token-at-a-time) form only, fp32 state, written to be read against
//! llama.cpp's `delta-net-base.cpp` rather than to be fast. Prefill costs one
//! sequential scan step per token. Do not optimize it; the chunked scan and the
//! fused decode step are separate vendored kernels (TODO P8) that are graded
//! against this.
//!
//! Per token, with `x` the pre-attention normed residual:
//!
//! ```text
//! qkv    = attn_qkv(x)                       [conv_dim]  (q | k | v, fused)
//! qkv    = silu(depthwise_conv1d(qkv))       causal, kernel 4, no bias
//! q,k,v  = split(qkv);  q,k = l2norm(q,k);  q,k tiled up to the V-head count
//! beta   = sigmoid(ssm_beta(x))              [v_heads]
//! g      = ssm_a * softplus(ssm_alpha(x) + ssm_dt_bias)      [v_heads], <= 0
//! S     *= exp(g);  d = (v - k·S) * beta;  S += k (x) d;  o = q·S / sqrt(d_k)
//! o      = rms_norm(o) * ssm_norm_w * silu(attn_gate(x))
//! out    = ssm_out(o)
//! ```
//!
//! `S` is `[v_heads, d_k, d_v]`: the first matrix axis contracts with k and q,
//! the second carries the value dimension. State carried between forwards is the
//! conv window (the last `kernel-1` columns of the PRE-conv fused stream) and
//! `S`, both f32, both held by the layer's `LayerCache`.

use anyhow::{Context, Result, ensure};
use candle_core::Tensor;
use candle_nn::ops::{sigmoid, silu};

use crate::attention::{AttnWeights, Proj};
use crate::config::XwenConfig;
use crate::gguf::Weights;
use crate::kv_cache::{LayerCache, materialize};

/// `x / max(||x||_2, eps)` over the last axis of a `[seq, heads, dim]` tensor —
/// the q/k L2 normalization in `ggml_l2_norm`'s form, where eps is a FLOOR ON
/// THE NORM and not a term under the root:
///
/// ```text
/// ggml/src/ggml-cpu/ops.cpp, ggml_compute_forward_l2_norm_f32:
///     const float scale = 1.0f/fmaxf(sqrtf(sum), eps);
/// ```
///
/// HF/FLA instead computes `x * rsqrt(sum + eps)`. The two agree to rounding for
/// any vector whose norm clears eps — which a silu'd conv output always does —
/// so this choice is invisible in practice. It is made deliberately anyway:
/// llama.cpp is the parity ground truth, and a form that merely "cannot matter"
/// is the kind of thing that turns out to matter once a strict tier asks for
/// bitwise agreement.
fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let norm = x
        .sqr()?
        .sum_keepdim(2)?
        .sqrt()?
        .clamp(eps as f32, f32::INFINITY)?;
    Ok(x.broadcast_div(&norm)?.contiguous()?)
}

/// One gated-DeltaNet layer.
pub struct LinearAttnBlock {
    /// `attn_qkv`: the fused q|k|v projection the conv runs over. DeltaNet ships
    /// under attention tensor names; there is no `ssm_in`.
    qkv: Proj,
    /// `attn_gate`: the z gate multiplied in after the output norm.
    z_proj: Proj,
    /// `ssm_out`: back to the residual width.
    out_proj: Proj,
    /// Depthwise conv taps, `[kernel, conv_dim]` f32, oldest tap first.
    conv_w: Tensor,
    /// `[hidden, v_heads]` f32, pre-transposed for the per-forward matmul.
    beta_wt: Tensor,
    alpha_wt: Tensor,
    /// `ssm_a`, `[v_heads]` f32 — already `-exp(A_log)`; used as-is.
    ssm_a: Tensor,
    /// `ssm_dt.bias`, `[v_heads]` f32 — the dt offset, not a projection bias.
    dt_bias: Tensor,
    /// `ssm_norm.weight`, `[head_dim]` f32.
    norm_w: Tensor,
    k_heads: usize,
    v_heads: usize,
    head_dim: usize,
    conv_kernel: usize,
    rms_eps: f64,
}

impl LinearAttnBlock {
    /// `w` is positioned at the block prefix (e.g. `blk.2`).
    pub fn new(w: &Weights, cfg: &XwenConfig, weights: AttnWeights) -> Result<Self> {
        let (k_heads, v_heads, head_dim) =
            (cfg.linear_k_heads, cfg.linear_v_heads, cfg.linear_head_dim);

        // `ssm_conv1d.weight` is stored kernel-major — GGUF ne `[kernel, conv_dim]`,
        // which candle reads reversed as `[conv_dim, kernel]`. The scan wants tap
        // j to be one contiguous row, so transpose to `[kernel, conv_dim]`.
        // Reshaping first tolerates a conversion that adds the singleton channel
        // axis a depthwise conv1d would conventionally carry.
        //
        // The reshape is also the load-time check on `conv_dim`: the fused width
        // has no metadata key and is derived as
        // `2 * group_count * state_size + inner_size`, so a wrong derivation
        // fails here against the real tensor rather than silently mis-splitting
        // q/k/v later.
        let conv_w = w
            .dense_f32("ssm_conv1d")?
            .reshape((cfg.conv_dim(), cfg.conv_kernel))
            .with_context(|| {
                format!(
                    "ssm_conv1d does not hold {} taps over the derived conv width {}",
                    cfg.conv_kernel,
                    cfg.conv_dim()
                )
            })?
            .t()?
            .contiguous()?;

        // Two separate projections, not the fused `ssm_ba` of llama.cpp's
        // qwen3next arch — the qwen35 conversion maps HF's `in_proj_b` and
        // `in_proj_a` one-to-one.
        let (beta_w, alpha_w) = (w.dense_f32("ssm_beta")?, w.dense_f32("ssm_alpha")?);
        ensure!(
            beta_w.dim(0)? == v_heads && alpha_w.dim(0)? == v_heads,
            "ssm_beta/ssm_alpha have {}/{} rows, expected one per V-head ({v_heads})",
            beta_w.dim(0)?,
            alpha_w.dim(0)?
        );

        let proj = |name: &str| -> Result<Proj> { Proj::load(w, name, weights) };
        Ok(Self {
            qkv: proj("attn_qkv")?,
            z_proj: proj("attn_gate")?,
            out_proj: proj("ssm_out")?,
            conv_w,
            beta_wt: beta_w.t()?.contiguous()?,
            alpha_wt: alpha_w.t()?.contiguous()?,
            ssm_a: w.dense_f32_any("ssm_a")?.flatten_all()?,
            dt_bias: w.dense_f32_any("ssm_dt")?.flatten_all()?,
            norm_w: w.dense_f32("ssm_norm")?.flatten_all()?,
            k_heads,
            v_heads,
            head_dim,
            conv_kernel: cfg.conv_kernel,
            rms_eps: cfg.rms_eps,
        })
    }

    /// `x_normed`: `[seq, hidden]` f32, already through the layer's `attn_norm`.
    /// Advances `cache`'s recurrent state by `seq` tokens. Returns
    /// `[seq, hidden]` f32.
    pub fn forward(&self, x_normed: &Tensor, cache: &mut LayerCache) -> Result<Tensor> {
        let (seq, _hidden) = x_normed.dims2()?;
        let (hd, kh, vh) = (self.head_dim, self.k_heads, self.v_heads);
        let (k_dim, v_dim) = (kh * hd, vh * hd);
        let tail = self.conv_kernel - 1;

        let qkv = self.qkv.forward(x_normed)?; // [seq, conv_dim] f32
        let (conv_state, mut s) = cache.linear_state()?;

        // Causal depthwise conv, computed by shifting the carried window in front
        // of this chunk: row t of `stream` is pre-conv column t - tail, so tap j
        // of output token t reads `stream[t + j]` and tap `kernel-1` is the token
        // itself.
        let stream = Tensor::cat(&[&conv_state, &qkv], 0)?; // [tail + seq, conv_dim]
        let mut conv = stream
            .narrow(0, 0, seq)?
            .broadcast_mul(&self.conv_w.narrow(0, 0, 1)?)?;
        for j in 1..self.conv_kernel {
            conv = (conv
                + stream
                    .narrow(0, j, seq)?
                    .broadcast_mul(&self.conv_w.narrow(0, j, 1)?)?)?;
        }
        let conv = silu(&conv)?; // [seq, conv_dim]

        // q and k arrive at the K-head count and are broadcast up to the V-head
        // count by TILING (head j reads K-head j % k_heads). The GGUF converter
        // permuted every V-side weight to suit exactly this; an interleaving
        // broadcast (HF's layout) is wrong against these weights.
        let heads = |off: usize, width: usize, n: usize| -> Result<Tensor> {
            Ok(conv
                .narrow(1, off, width)?
                .contiguous()?
                .reshape((seq, n, hd))?)
        };
        let q = tile_heads(&l2_norm(&heads(0, k_dim, kh)?, self.rms_eps)?, vh)?;
        let k = tile_heads(&l2_norm(&heads(k_dim, k_dim, kh)?, self.rms_eps)?, vh)?;
        let v = heads(2 * k_dim, v_dim, vh)?;

        // beta and the decay come from the layer INPUT, not the conv output.
        // `ssm_a` is the pre-baked -exp(A_log), so g is non-positive and the decay
        // lands in (0, 1].
        let beta = sigmoid(&x_normed.matmul(&self.beta_wt)?)?; // [seq, v_heads]
        let dt = x_normed
            .matmul(&self.alpha_wt)?
            .broadcast_add(&self.dt_bias)?;
        let decay = softplus(&dt)?.broadcast_mul(&self.ssm_a)?.exp()?;

        let z = self.z_proj.forward(x_normed)?; // [seq, v_dim]

        // The scan. One step per token; every intermediate state is a distinct
        // tensor, so recording the whole trail for a rollback costs handles
        // rather than copies.
        let record = cache.linear_trail_armed();
        let scale = 1.0 / (hd as f64).sqrt();
        let mut outs = Vec::with_capacity(seq);
        let mut states: Vec<(Tensor, Tensor)> = Vec::with_capacity(if record { seq } else { 1 });
        for t in 0..seq {
            let row = |t3: &Tensor| t3.narrow(0, t, 1)?.reshape((vh, 1, hd));
            let qt = row(&q)?;
            let kt = row(&k)?;
            let vt = row(&v)?;
            let dec = decay.narrow(0, t, 1)?.reshape((vh, 1, 1))?;
            let bt = beta.narrow(0, t, 1)?.reshape((vh, 1, 1))?;

            s = s.broadcast_mul(&dec)?; // S *= exp(g)
            let sk = kt.matmul(&s)?; // k contracted over the key axis
            let d = (vt - sk)?.broadcast_mul(&bt)?;
            s = (s + kt.reshape((vh, hd, 1))?.matmul(&d)?)?; // S += k (x) d
            outs.push((qt.matmul(&s)? * scale)?.reshape((1, vh, hd))?);

            if record {
                states.push((materialize(&stream.narrow(0, t + 1, tail)?)?, s.clone()));
            }
        }
        if !record {
            states.push((materialize(&stream.narrow(0, seq, tail)?)?, s.clone()));
        }
        cache.advance_linear(seq, states)?;

        // Gated RMSNorm: normalize over the head dim, scale by ssm_norm.weight,
        // and only THEN multiply by silu(z). The gate is outside the norm, so it
        // does not affect the statistic.
        let o = Tensor::cat(&outs, 0)?; // [seq, v_heads, hd]
        let ms = (o.sqr()?.sum_keepdim(2)? / hd as f64)?;
        let o = o
            .broadcast_div(&(ms + self.rms_eps)?.sqrt()?)?
            .broadcast_mul(&self.norm_w.reshape((1, 1, hd))?)?;
        let o = (o * silu(&z)?.reshape((seq, vh, hd))?)?;
        self.out_proj.forward(&o.reshape((seq, v_dim))?)
    }
}

/// Broadcast `[seq, k_heads, dim]` up to `[seq, v_heads, dim]` by TILING: output
/// head j reads input head `j % k_heads`. This is ggml's plain repeat, and the
/// order the GGUF's permuted V-side weights expect.
fn tile_heads(x: &Tensor, v_heads: usize) -> Result<Tensor> {
    let k_heads = x.dim(1)?;
    if k_heads == v_heads {
        return Ok(x.contiguous()?);
    }
    let parts: Vec<&Tensor> = (0..v_heads / k_heads).map(|_| x).collect();
    Ok(Tensor::cat(&parts, 1)?.contiguous()?)
}

/// Numerically stable softplus, ln(1 + exp(x)) = relu(x) + ln(1 + exp(-|x|)).
pub(crate) fn softplus(x: &Tensor) -> Result<Tensor> {
    let ax = x.abs()?;
    let relu = x.broadcast_add(&ax)?.affine(0.5, 0.0)?;
    let tail = ax.neg()?.exp()?.affine(1.0, 1.0)?.log()?;
    Ok(relu.broadcast_add(&tail)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Arch, LayerKind, RopeKind};
    use candle_core::Device;
    use candle_core::quantized::{GgmlDType, QTensor};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn dev() -> Device {
        Device::Cpu
    }

    /// A DeltaNet-only config: the fields the block reads, sized for a test.
    fn cfg_for(k_heads: usize, v_heads: usize, hd: usize, hidden: usize) -> XwenConfig {
        XwenConfig {
            arch: Arch::Moe,
            n_layer: 1,
            hidden,
            vocab: 8,
            n_head: vec![1],
            n_kv_head: 1,
            head_dim: hd,
            layer_kind: vec![LayerKind::Linear],
            linear_k_heads: k_heads,
            linear_v_heads: v_heads,
            linear_head_dim: hd,
            conv_kernel: 4,
            dense_ff: 0,
            n_expert: 0,
            n_expert_used: 0,
            expert_ff: 0,
            shared_expert_ff: 0,
            rms_eps: 1e-6,
            n_ctx_train: 64,
            rope: RopeKind::Plain {
                freq_base: 1e7,
                n_rot: 2,
            },
            eog_tokens: vec![0],
        }
    }

    fn t2(data: Vec<f32>, rows: usize, cols: usize) -> Tensor {
        Tensor::from_vec(data, (rows, cols), &dev()).unwrap()
    }

    fn t1(data: Vec<f32>) -> Tensor {
        let n = data.len();
        Tensor::from_vec(data, n, &dev()).unwrap()
    }

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    /// The raw per-layer weights, written to a throwaway GGUF under the real
    /// tensor names so the loader's name table (no `ssm_in`, `ssm_a` bare,
    /// `ssm_dt.bias`) is exercised rather than bypassed.
    struct Raw {
        qkv: Tensor,     // [conv_dim, hidden]
        gate: Tensor,    // [v_dim, hidden]
        out: Tensor,     // [hidden, v_dim]
        conv: Tensor,    // [conv_dim, kernel] — GGUF ne [kernel, conv_dim]
        beta: Tensor,    // [v_heads, hidden]
        alpha: Tensor,   // [v_heads, hidden]
        a: Tensor,       // [v_heads]
        dt_bias: Tensor, // [v_heads]
        norm: Tensor,    // [head_dim]
    }

    fn build(raw: &Raw, cfg: &XwenConfig) -> LinearAttnBlock {
        let q = |t: &Tensor| QTensor::quantize(t, GgmlDType::F32).unwrap();
        let (qkv, gate, out, conv, beta, alpha, a, dt, norm) = (
            q(&raw.qkv),
            q(&raw.gate),
            q(&raw.out),
            q(&raw.conv),
            q(&raw.beta),
            q(&raw.alpha),
            q(&raw.a),
            q(&raw.dt_bias),
            q(&raw.norm),
        );
        let tensors: Vec<(&str, &QTensor)> = vec![
            ("blk.0.attn_qkv.weight", &qkv),
            ("blk.0.attn_gate.weight", &gate),
            ("blk.0.ssm_out.weight", &out),
            ("blk.0.ssm_conv1d.weight", &conv),
            ("blk.0.ssm_beta.weight", &beta),
            ("blk.0.ssm_alpha.weight", &alpha),
            ("blk.0.ssm_a", &a),
            ("blk.0.ssm_dt.bias", &dt),
            ("blk.0.ssm_norm.weight", &norm),
        ];
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("xwen_delta_test_{}_{id}.gguf", std::process::id()));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            candle_core::quantized::gguf_file::write(&mut f, &[], &tensors).unwrap();
        }
        let src = crate::gguf::open(&path, &dev()).unwrap();
        let w = Weights::from_gguf(src).pp("blk.0");
        let block = LinearAttnBlock::new(&w, cfg, AttnWeights::DequantF32).unwrap();
        let _ = std::fs::remove_file(&path);
        block
    }

    /// Weights that make the block a transparent probe of the recurrence: the
    /// qkv projection is the identity on the first `conv_dim` inputs, the conv is
    /// a pass-through of the current token, and the output projection is the
    /// identity. Everything the delta rule does is then visible in the output.
    fn passthrough(cfg: &XwenConfig, hidden: usize) -> Raw {
        let conv_dim = cfg.conv_dim();
        let v_dim = cfg.linear_v_dim();
        let vh = cfg.linear_v_heads;
        let eye = |rows: usize, cols: usize| {
            let mut d = vec![0f32; rows * cols];
            for i in 0..rows.min(cols) {
                d[i * cols + i] = 1.0;
            }
            t2(d, rows, cols)
        };
        // Conv taps: only the newest (index kernel-1) is 1, so the conv is the
        // identity on the current token and the carried window is inert.
        let mut conv = vec![0f32; conv_dim * cfg.conv_kernel];
        for c in 0..conv_dim {
            conv[c * cfg.conv_kernel + (cfg.conv_kernel - 1)] = 1.0;
        }
        Raw {
            qkv: eye(conv_dim, hidden),
            gate: t2(vec![0f32; v_dim * hidden], v_dim, hidden),
            out: eye(hidden, v_dim),
            conv: Tensor::from_vec(conv, (conv_dim, cfg.conv_kernel), &dev()).unwrap(),
            beta: t2(vec![0f32; vh * hidden], vh, hidden),
            alpha: t2(vec![0f32; vh * hidden], vh, hidden),
            a: t1(vec![0f32; vh]),
            dt_bias: t1(vec![0f32; vh]),
            norm: t1(vec![1f32; cfg.linear_head_dim]),
        }
    }

    fn cache_for(cfg: &XwenConfig) -> LayerCache {
        LayerCache::new(cfg, 0, 16, &dev()).unwrap()
    }

    fn to_vec2(t: &Tensor) -> Vec<Vec<f32>> {
        t.to_dtype(candle_core::DType::F32)
            .unwrap()
            .to_vec2()
            .unwrap()
    }

    fn silu_f64(v: f64) -> f64 {
        v / (1.0 + (-v).exp())
    }

    /// The delta rule, at head dim 2 with a single head over three tokens,
    /// against a scalar f64 walk of the update equations written out below. The
    /// reference multiplies nothing through a tensor: it is the equations from
    /// this module's header, indexed by hand, so a rewrite of the tensor form
    /// that changes what the recurrence MEANS fails here.
    ///
    /// The block is wired transparent — identity qkv projection, newest-tap-only
    /// conv, identity output projection — so the only transforms between the
    /// input row and the output row are the ones under test (silu, L2, the
    /// recurrence, the gated norm), and each appears explicitly in the reference.
    #[test]
    fn delta_rule_step_matches_hand_computation() {
        let (hd, kh, vh) = (2usize, 1usize, 1usize);
        let hidden = 6; // conv_dim = (2*1 + 1) * 2 = 6, so qkv is the identity
        let cfg = cfg_for(kh, vh, hd, hidden);
        assert_eq!(cfg.conv_dim(), hidden);

        let mut raw = passthrough(&cfg, hidden);
        // beta = sigmoid(0) = 0.5 for every token (zero beta weights); the decay
        // is exp(ssm_a * softplus(0)) with ssm_a = 0, i.e. exactly 1 — no
        // forgetting, so the state carries forward untouched.
        //
        // A zero gate weight would give silu(0) = 0 and hide the whole output, so
        // the gate reads input column 0 (which is 1 in every test token) with a
        // factor of 4: silu(z) is the fixed constant silu(4).
        let mut gate = vec![0f32; cfg.linear_v_dim() * hidden];
        for r in 0..cfg.linear_v_dim() {
            gate[r * hidden] = 4.0;
        }
        raw.gate = t2(gate, cfg.linear_v_dim(), hidden);
        let block = build(&raw, &cfg);
        let mut cache = cache_for(&cfg);

        // Columns are [q(2) | k(2) | v(2)]; column 0 doubles as the gate input.
        let rows: [[f64; 6]; 3] = [
            [1.0, 0.0, 1.0, 0.0, 2.0, 3.0],
            [1.0, 0.0, 0.0, 1.0, 5.0, 7.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        ];
        let x = t2(
            rows.iter().flatten().map(|v| *v as f32).collect(),
            3,
            hidden,
        );
        let got = to_vec2(&block.forward(&x, &mut cache).unwrap());

        // --- the reference, by hand -----------------------------------------
        // S is [d_k][d_v]: index 0 contracts with k and q, index 1 is the value
        // dimension.
        let mut s = [[0f64; 2]; 2];
        let beta = 0.5;
        let scale = 1.0 / (hd as f64).sqrt();
        let gate_factor = silu_f64(4.0);
        for (t, row) in rows.iter().enumerate() {
            // The conv passes the token through, then silu runs over the WHOLE
            // fused stream — q, k and v alike.
            let act: Vec<f64> = row.iter().map(|v| silu_f64(*v)).collect();
            // q and k are L2-normalized; v is not.
            let l2 = |a: f64, b: f64| {
                let n = (a * a + b * b).sqrt().max(1e-6);
                [a / n, b / n]
            };
            let q = l2(act[0], act[1]);
            let k = l2(act[2], act[3]);
            let v = [act[4], act[5]];

            // decay is 1 here, so S is unchanged before the update.
            let sk = [
                s[0][0] * k[0] + s[1][0] * k[1],
                s[0][1] * k[0] + s[1][1] * k[1],
            ];
            let d = [(v[0] - sk[0]) * beta, (v[1] - sk[1]) * beta];
            for i in 0..2 {
                for j in 0..2 {
                    s[i][j] += k[i] * d[j];
                }
            }
            let o = [
                (s[0][0] * q[0] + s[1][0] * q[1]) * scale,
                (s[0][1] * q[0] + s[1][1] * q[1]) * scale,
            ];

            // Gated RMS norm: rms over the value dims, x weight (1 here), then
            // x silu(z).
            let inv = 1.0 / ((o[0] * o[0] + o[1] * o[1]) / 2.0 + 1e-6).sqrt();
            let want = [o[0] * inv * gate_factor, o[1] * inv * gate_factor];

            // The identity output projection leaves the value dims in columns 0,1.
            for d in 0..hd {
                assert!(
                    (got[t][d] as f64 - want[d]).abs() < 2e-5,
                    "token {t} dim {d}: got {} want {}",
                    got[t][d],
                    want[d]
                );
            }
        }
    }

    /// The output norm applies rms → ssm_norm.weight → silu(z), in that order.
    /// Folding the gate in before the norm would change the statistic; applying
    /// the weight after the gate would scale a different value. A non-uniform
    /// weight and per-channel gate factors of opposite sign make either swap
    /// visible.
    #[test]
    fn gated_norm_applies_weight_before_the_gate() {
        let (hd, kh, vh) = (2usize, 1usize, 1usize);
        let hidden = 6;
        let cfg = cfg_for(kh, vh, hd, hidden);
        let mut raw = passthrough(&cfg, hidden);
        raw.norm = t1(vec![0.5, 2.0]);
        let mut gate = vec![0f32; cfg.linear_v_dim() * hidden];
        gate[0] = 1.0; // channel 0: z = x[0]
        gate[hidden] = -3.0; // channel 1: z = -3 * x[0]
        raw.gate = t2(gate, cfg.linear_v_dim(), hidden);
        let block = build(&raw, &cfg);
        let mut cache = cache_for(&cfg);

        let row = [1.0f64, 0.0, 1.0, 0.0, 2.0, 3.0];
        let x = t2(row.iter().map(|v| *v as f32).collect(), 1, hidden);
        let got = to_vec2(&block.forward(&x, &mut cache).unwrap());

        // One token, so S = k (x) (beta * v) with beta = 0.5, and o = q·S/sqrt(2).
        let act: Vec<f64> = row.iter().map(|v| silu_f64(*v)).collect();
        let n = |a: f64, b: f64| (a * a + b * b).sqrt().max(1e-6);
        let q = [act[0] / n(act[0], act[1]), act[1] / n(act[0], act[1])];
        let k = [act[2] / n(act[2], act[3]), act[3] / n(act[2], act[3])];
        let d = [act[4] * 0.5, act[5] * 0.5];
        let scale = 1.0 / (hd as f64).sqrt();
        let o = [
            (k[0] * d[0] * q[0] + k[1] * d[0] * q[1]) * scale,
            (k[0] * d[1] * q[0] + k[1] * d[1] * q[1]) * scale,
        ];

        let inv = 1.0 / ((o[0] * o[0] + o[1] * o[1]) / 2.0 + 1e-6).sqrt();
        let w = [0.5f64, 2.0];
        let z = [1.0f64, -3.0];
        for d in 0..hd {
            let want = o[d] * inv * w[d] * silu_f64(z[d]);
            assert!(
                (got[0][d] as f64 - want).abs() < 2e-5,
                "dim {d}: got {} want {want}",
                got[0][d]
            );
        }
    }

    /// The conv window and the delta state carried in the cache make a
    /// token-at-a-time walk identical to one batched forward. This is what makes
    /// decode after a prefill correct, and it is the only place the carried conv
    /// tail is load-bearing — a conv that dropped or misordered it would agree on
    /// the batched path and diverge here.
    #[test]
    fn streaming_one_token_at_a_time_matches_one_batch() {
        let (hd, kh, vh) = (4usize, 2usize, 4usize);
        let hidden = 12;
        let cfg = cfg_for(kh, vh, hd, hidden);
        let conv_dim = cfg.conv_dim();
        let v_dim = cfg.linear_v_dim();

        let seeded = |n: usize, seed: u64| -> Vec<f32> {
            let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 33) as f32 / u32::MAX as f32) - 0.5
                })
                .collect()
        };
        let raw = Raw {
            qkv: t2(seeded(conv_dim * hidden, 1), conv_dim, hidden),
            gate: t2(seeded(v_dim * hidden, 2), v_dim, hidden),
            out: t2(seeded(hidden * v_dim, 3), hidden, v_dim),
            conv: Tensor::from_vec(
                seeded(conv_dim * cfg.conv_kernel, 4),
                (conv_dim, cfg.conv_kernel),
                &dev(),
            )
            .unwrap(),
            beta: t2(seeded(vh * hidden, 5), vh, hidden),
            alpha: t2(seeded(vh * hidden, 6), vh, hidden),
            // Real decay: negative a, so exp(g) is strictly inside (0, 1).
            a: t1(seeded(vh, 7).iter().map(|v| v - 1.0).collect()),
            dt_bias: t1(seeded(vh, 8)),
            norm: t1(seeded(hd, 9).iter().map(|v| 1.0 + v).collect()),
        };
        let block = build(&raw, &cfg);

        let seq = 7;
        let x = t2(seeded(seq * hidden, 10), seq, hidden);

        let mut batch_cache = cache_for(&cfg);
        let batched = to_vec2(&block.forward(&x, &mut batch_cache).unwrap());

        let mut step_cache = cache_for(&cfg);
        let mut streamed = Vec::new();
        for t in 0..seq {
            let row = x.narrow(0, t, 1).unwrap().contiguous().unwrap();
            streamed.extend(to_vec2(&block.forward(&row, &mut step_cache).unwrap()));
        }

        assert_eq!(step_cache.len(), seq, "the cache advanced once per token");
        for (t, (a, b)) in batched.iter().zip(&streamed).enumerate() {
            for (d, (x, y)) in a.iter().zip(b).enumerate() {
                assert!(
                    (x - y).abs() < 1e-4,
                    "token {t} dim {d}: batched {x} vs streamed {y}"
                );
            }
        }
    }

    /// q and k are broadcast to the V-heads by tiling, so V-head j reads K-head
    /// `j % k_heads`. An interleaving broadcast would pair them the other way,
    /// which reads the GGUF's permuted V-side weights against the wrong keys.
    #[test]
    fn head_broadcast_tiles_rather_than_interleaves() {
        let x = Tensor::from_vec(
            vec![
                0.0f32, 1.0, // seq 0, k-head 0
                2.0, 3.0, // seq 0, k-head 1
            ],
            (1, 2, 2),
            &dev(),
        )
        .unwrap();
        let tiled = tile_heads(&x, 4).unwrap();
        assert_eq!(
            to_vec2(&tiled.reshape((4, 2)).unwrap()),
            vec![
                vec![0.0, 1.0],
                vec![2.0, 3.0],
                vec![0.0, 1.0],
                vec![2.0, 3.0]
            ]
        );
    }

    /// The L2 normalization floors the NORM at eps rather than adding eps under
    /// the root — `ggml_l2_norm`'s form. A unit vector therefore comes back
    /// exactly unchanged (the rsqrt form would shrink it by sqrt(1+eps)), and a
    /// vector far below eps is divided by eps rather than by sqrt(eps). Those
    /// two constants are the whole observable difference between the forms, and
    /// this test is what pins which one the engine implements.
    #[test]
    fn l2_norm_floors_the_norm() {
        let eps = 1e-6f64;
        let unit = Tensor::from_vec(vec![0.6f32, 0.8], (1, 1, 2), &dev()).unwrap();
        let out: Vec<f32> = l2_norm(&unit, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!((out[0] - 0.6).abs() < 1e-7 && (out[1] - 0.8).abs() < 1e-7);

        let tiny = Tensor::from_vec(vec![1e-12f32, 0.0], (1, 1, 2), &dev()).unwrap();
        let out: Vec<f32> = l2_norm(&tiny, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            (out[0] as f64 - 1e-12 / eps).abs() < 1e-12,
            "expected division by the eps floor, got {out:?}"
        );
    }
}
