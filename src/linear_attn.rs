//! Gated DeltaNet — the linear-attention layers, three of every four.
//!
//! This is the FROZEN CORRECTNESS ORACLE for the recurrence, in the same sense
//! `ReferenceExperts` is for the MoE FFN: composed candle ops, the recurrent
//! (token-at-a-time) form only, fp32 state, written to be read against
//! llama.cpp's `delta-net-base.cpp` rather than to be fast. Prefill costs one
//! sequential scan step per token. Do not optimize it: it is what the vendored
//! kernels (`ops::delta_*`, reached through `forward_fused`) are graded against,
//! and what `XWEN_DELTA_CLASSIC` falls back to.
//!
//! `forward` picks between the two. Everything below `forward_classic` is the
//! oracle; `forward_fused` is the shipped path.
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

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, ensure};
use candle_core::Tensor;
use candle_nn::ops::{sigmoid, silu};

use crate::attention::{AttnWeights, Proj};
use crate::config::XwenConfig;
use crate::gguf::Weights;
use crate::kv_cache::{LayerCache, materialize};

/// DeltaNet layer forwards this process has run down each path.
///
/// `forward` decides per call — on the head dim, the device and the
/// `XWEN_DELTA_CLASSIC` switch — so the switch alone does not say which path a
/// run took (a block at an unsupported head dim takes the reference either way).
/// Anything recording provenance reads these instead of the environment, which
/// is what keeps a bounded parity tier from grading the reference scan against
/// itself. Counting only; the increment carries no ordering obligation.
static FUSED_FORWARDS: AtomicU64 = AtomicU64::new(0);
static CLASSIC_FORWARDS: AtomicU64 = AtomicU64::new(0);

/// `(fused, classic)` DeltaNet layer forwards since process start.
pub fn delta_path_counts() -> (u64, u64) {
    (
        FUSED_FORWARDS.load(Ordering::Relaxed),
        CLASSIC_FORWARDS.load(Ordering::Relaxed),
    )
}

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
pub(crate) fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
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
    /// `[hidden, 2 * v_heads]` f32 — `beta_wt` and `alpha_wt` side by side, so
    /// the fused path runs ONE gemv where the reference runs two. Column block
    /// 0 is beta, block 1 is alpha; `ops::delta_ba` reads that layout.
    ba_wt: Tensor,
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

        let (beta_wt, alpha_wt) = (beta_w.t()?.contiguous()?, alpha_w.t()?.contiguous()?);
        let ba_wt = Tensor::cat(&[&beta_wt, &alpha_wt], 1)?.contiguous()?;

        let proj = |name: &str| -> Result<Proj> { Proj::load(w, name, weights) };
        Ok(Self {
            qkv: proj("attn_qkv")?,
            z_proj: proj("attn_gate")?,
            out_proj: proj("ssm_out")?,
            conv_w,
            beta_wt,
            alpha_wt,
            ba_wt,
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
    ///
    /// Runs the fused Metal kernels when they apply and the frozen reference
    /// scan otherwise. The fused path needs a Metal device and the production
    /// head dim the scan kernel is specialized to; a live rollback trail does
    /// not push it off, because the scan emits one state plane per token when
    /// asked (`ops::delta_scan_with_trail`).
    pub fn forward(&self, x_normed: &Tensor, cache: &mut LayerCache) -> Result<Tensor> {
        let fused = self.head_dim == crate::ops::DELTA_HEAD_DIM
            && !crate::ops::delta_classic()
            && x_normed.device().is_metal();
        if fused {
            self.forward_fused(x_normed, cache)
        } else {
            self.forward_classic(x_normed, cache)
        }
    }

    /// The fused path: eight dispatches per layer at any sequence length —
    /// three projections, the conv+silu+state kernel, the beta/decay head, the
    /// whole recurrent scan, the gated output norm, and the output projection.
    ///
    /// A multi-token chunk under an armed rollback trail keeps those eight
    /// scan-side dispatches (the scan writes one state plane per token instead
    /// of one) but adds the trail's host-side copies: one `cat` for the
    /// shifted stream plus one materialized conv window per token, so an armed
    /// chunk of `seq` tokens costs roughly `8 + 1 + seq` dispatches. The conv
    /// windows come from the same shifted stream the reference builds.
    fn forward_fused(&self, x_normed: &Tensor, cache: &mut LayerCache) -> Result<Tensor> {
        FUSED_FORWARDS.fetch_add(1, Ordering::Relaxed);
        let seq = x_normed.dim(0)?;
        let v_dim = self.v_heads * self.head_dim;
        let tail = self.conv_kernel - 1;

        let qkv = self.qkv.forward(x_normed)?; // [seq, conv_dim]
        let (conv_state, s) = cache.linear_state()?;
        let (conv, next_conv_state) = crate::ops::delta_conv(&conv_state, &qkv, &self.conv_w)?;

        // beta and the decay come from the layer INPUT, not the conv output.
        let ba = x_normed.matmul(&self.ba_wt)?; // [seq, 2 * v_heads]
        let (beta, g) = crate::ops::delta_ba(&ba, &self.ssm_a, &self.dt_bias)?;

        let z = self.z_proj.forward(x_normed)?; // [seq, v_dim]

        let eps = self.rms_eps as f32;
        let o = if seq > 1 && cache.linear_trail_armed() {
            let (o, trail) = crate::ops::delta_scan_with_trail(
                &conv,
                &beta,
                &g,
                &s,
                self.k_heads,
                eps,
                seq, // one state plane per token of the verify span
            )?;
            // The conv window a rollback to token t restarts from is rows
            // t+1..t+1+tail of the carried-window-plus-chunk stream, exactly as
            // the reference records it. The delta entries are views into the
            // trail buffer: nothing mutates it, and `rollback` materializes the
            // one entry it keeps.
            let stream = Tensor::cat(&[&conv_state, &qkv], 0)?; // [tail + seq, conv_dim]
            let mut states = Vec::with_capacity(seq);
            for t in 0..seq {
                states.push((
                    materialize(&stream.narrow(0, t + 1, tail)?)?,
                    trail.narrow(0, seq - 1 - t, 1)?.squeeze(0)?,
                ));
            }
            cache.advance_linear(seq, states)?;
            o
        } else {
            let (o, next_s) = crate::ops::delta_scan(&conv, &beta, &g, &s, self.k_heads, eps)?;
            cache.advance_linear(seq, vec![(next_conv_state, next_s)])?;
            o
        };

        let o = crate::ops::delta_gnorm(
            &o,
            &z.reshape((seq, self.v_heads, self.head_dim))?,
            &self.norm_w,
            self.rms_eps as f32,
        )?;
        self.out_proj.forward(&o.reshape((seq, v_dim))?)
    }

    /// The FROZEN reference scan (see the module header). Never optimize it: it
    /// is the oracle the fused kernels are graded against, and the
    /// `XWEN_DELTA_CLASSIC` fallback.
    fn forward_classic(&self, x_normed: &Tensor, cache: &mut LayerCache) -> Result<Tensor> {
        CLASSIC_FORWARDS.fetch_add(1, Ordering::Relaxed);
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
pub(crate) fn tile_heads(x: &Tensor, v_heads: usize) -> Result<Tensor> {
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

    /// `delta_path_counts` is process-global, so reading it around a single
    /// forward only means something with every other forward in this module out
    /// of the way. Every test that runs a DeltaNet layer takes this lock; a
    /// panicking test hands it on rather than poisoning the rest.
    static PATH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn path_lock() -> std::sync::MutexGuard<'static, ()> {
        PATH_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

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
        build_on(raw, cfg, &dev())
    }

    /// `build`, with the block's weights landing on `device`. The raw tensors
    /// are written to a throwaway GGUF from the CPU either way; only the load
    /// device differs, so a Metal block and a CPU block hold the same bits.
    fn build_on(raw: &Raw, cfg: &XwenConfig, device: &Device) -> LinearAttnBlock {
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
        let src = crate::gguf::open(&path, device).unwrap();
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
        cache_on(cfg, &dev(), 16)
    }

    fn cache_on(cfg: &XwenConfig, device: &Device, max_ctx: usize) -> LayerCache {
        LayerCache::new(cfg, 0, max_ctx, device).unwrap()
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
        let _guard = path_lock();
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
        let _guard = path_lock();
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
        let _guard = path_lock();
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

    /// Deterministic pseudo-random f32s in [-0.5, 0.5] (xorshift-free LCG, no
    /// deps) — the generator the streaming test already uses, lifted so the
    /// Metal tests can share it.
    fn seeded_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / u32::MAX as f32) - 0.5
            })
            .collect()
    }

    /// A block whose weights are all pseudo-random at head dim 128 — the dim the
    /// fused kernels are specialized to — so both paths run the real recurrence
    /// rather than a degenerate one.
    fn random_block(cfg: &XwenConfig, hidden: usize, seed: u64) -> Raw {
        let conv_dim = cfg.conv_dim();
        let v_dim = cfg.linear_v_dim();
        let vh = cfg.linear_v_heads;
        Raw {
            qkv: t2(seeded_vec(conv_dim * hidden, seed), conv_dim, hidden),
            gate: t2(seeded_vec(v_dim * hidden, seed + 1), v_dim, hidden),
            out: t2(seeded_vec(hidden * v_dim, seed + 2), hidden, v_dim),
            conv: Tensor::from_vec(
                seeded_vec(conv_dim * cfg.conv_kernel, seed + 3),
                (conv_dim, cfg.conv_kernel),
                &dev(),
            )
            .unwrap(),
            beta: t2(seeded_vec(vh * hidden, seed + 4), vh, hidden),
            alpha: t2(seeded_vec(vh * hidden, seed + 5), vh, hidden),
            // Real decay: negative a, so exp(g) is strictly inside (0, 1).
            a: t1(seeded_vec(vh, seed + 6).iter().map(|v| v - 1.0).collect()),
            dt_bias: t1(seeded_vec(vh, seed + 7)),
            norm: t1(seeded_vec(hd_of(cfg), seed + 8)
                .iter()
                .map(|v| 1.0 + v)
                .collect()),
        }
    }

    fn hd_of(cfg: &XwenConfig) -> usize {
        cfg.linear_head_dim
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }

    fn assert_all_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            assert!(
                (a - b).abs() <= tol * (1.0 + b.abs()),
                "{what}: element {i} fused {a} vs reference {b}"
            );
        }
    }

    /// The fused Metal path must reproduce the frozen reference scan end to end
    /// — the layer output AND the recurrent state it leaves in the cache — at
    /// the head dim the kernels are specialized to. Two head ratios: V-heads
    /// twice the K-heads (so the tiled broadcast is exercised) and the shipped
    /// 16-to-32 ratio. Decode, a short chunk and a prefill chunk each go
    /// through, because the scan kernel's state handling differs between one
    /// timestep and many.
    ///
    /// This is the test that grades the kernels as a package: a wrong fused
    /// beta|alpha column order, a mis-shaped gate reshape or a cache advance
    /// that stored the wrong state all show up here and nowhere in the per-op
    /// tests.
    #[test]
    fn fused_path_matches_the_reference_scan() {
        let Ok(metal) = crate::gguf::metal_device() else {
            return;
        };
        let _guard = path_lock();
        for &(kh, vh, hidden) in &[(2usize, 4usize, 256usize), (16, 32, 256)] {
            let cfg = cfg_for(kh, vh, 128, hidden);
            let raw = random_block(&cfg, hidden, 0x100 + kh as u64 * 7 + vh as u64);
            let block = build_on(&raw, &cfg, &metal);

            for &seq in &[1usize, 5, 64] {
                let x = Tensor::from_vec(
                    seeded_vec(seq * hidden, 0x900 + seq as u64),
                    (seq, hidden),
                    &metal,
                )
                .unwrap();

                let mut fused_cache = cache_on(&cfg, &metal, 128);
                let fused = block.forward(&x, &mut fused_cache).unwrap();
                let (fused_conv, fused_delta) = fused_cache.linear_state().unwrap();

                let mut ref_cache = cache_on(&cfg, &metal, 128);
                let reference = block.forward_classic(&x, &mut ref_cache).unwrap();
                let (ref_conv, ref_delta) = ref_cache.linear_state().unwrap();

                let label = format!("k={kh} v={vh} seq={seq}");
                assert_all_close(
                    &flat(&fused),
                    &flat(&reference),
                    2e-4,
                    &format!("{label} out"),
                );
                assert_all_close(
                    &flat(&fused_conv),
                    &flat(&ref_conv),
                    1e-6,
                    &format!("{label} conv window"),
                );
                assert_all_close(
                    &flat(&fused_delta),
                    &flat(&ref_delta),
                    2e-5,
                    &format!("{label} delta state"),
                );
            }
        }
    }

    /// Decoding one token at a time through the fused path must land where one
    /// batched fused prefill lands — the property that makes decode-after-
    /// prefill correct, now that the conv window and the delta state cross the
    /// call boundary through kernel-written buffers rather than candle slices.
    #[test]
    fn fused_streaming_matches_one_fused_batch() {
        let Ok(metal) = crate::gguf::metal_device() else {
            return;
        };
        let _guard = path_lock();
        let (kh, vh, hidden, seq) = (16usize, 32usize, 256usize, 7usize);
        let cfg = cfg_for(kh, vh, 128, hidden);
        let block = build_on(&random_block(&cfg, hidden, 0x300), &cfg, &metal);
        let x = Tensor::from_vec(seeded_vec(seq * hidden, 0x301), (seq, hidden), &metal).unwrap();

        let mut batch_cache = cache_on(&cfg, &metal, 32);
        let batched = flat(&block.forward(&x, &mut batch_cache).unwrap());

        let mut step_cache = cache_on(&cfg, &metal, 32);
        let mut streamed = Vec::new();
        for t in 0..seq {
            let row = x.narrow(0, t, 1).unwrap().contiguous().unwrap();
            streamed.extend(flat(&block.forward(&row, &mut step_cache).unwrap()));
        }
        assert_eq!(step_cache.len(), seq, "the cache advanced once per token");
        assert_all_close(&streamed, &batched, 2e-4, "streamed vs batched");
    }

    /// A live rollback checkpoint needs the state after EVERY token of the
    /// verify span, and the fused scan reports all of them in one dispatch — so
    /// an armed multi-token chunk takes the same fused path an unarmed one
    /// takes, with the trail coming out one entry per token. `advance_linear`
    /// refuses an armed advance carrying anything else, so a forward that
    /// returns at all is the proof of the count; the rollback over the full span
    /// at the end is the second.
    ///
    /// The kill switch still outranks the checkpoint: under `XWEN_DELTA_CLASSIC`
    /// every DeltaNet forward is the frozen reference scan, armed chunks
    /// included. That switch is read once per process, so this grades whichever
    /// arm the test process was launched in.
    #[test]
    fn an_armed_multi_token_chunk_stays_on_the_fused_scan() {
        let Ok(metal) = crate::gguf::metal_device() else {
            return;
        };
        let _guard = path_lock();
        let (kh, vh, hidden) = (16usize, 32usize, 256usize);
        let cfg = cfg_for(kh, vh, 128, hidden);
        let block = build_on(&random_block(&cfg, hidden, 0x400), &cfg, &metal);

        let mut cache = cache_on(&cfg, &metal, 32);
        let ckpt = cache.checkpoint(6).unwrap();
        assert!(cache.linear_trail_armed(), "the checkpoint arms the layer");

        let x = Tensor::from_vec(seeded_vec(4 * hidden, 0x401), (4, hidden), &metal).unwrap();
        let (fused_before, classic_before) = delta_path_counts();
        block.forward(&x, &mut cache).unwrap();
        let (fused, classic) = delta_path_counts();
        let switched = crate::ops::delta_classic();
        assert_eq!(
            (fused - fused_before, classic - classic_before),
            if switched { (0, 1) } else { (1, 0) },
            "armed four-token chunk took the wrong path (XWEN_DELTA_CLASSIC set: {switched})"
        );
        assert_eq!(cache.len(), 4);

        // Two more single tokens, so the trail spans a mix of chunk lengths.
        for t in 0..2usize {
            let row = Tensor::from_vec(seeded_vec(hidden, 0x402 + t as u64), (1, hidden), &metal)
                .unwrap();
            block.forward(&row, &mut cache).unwrap();
        }
        assert_eq!(cache.len(), 6, "the armed span filled exactly");

        cache.rollback(&ckpt, 0, 6, 5).unwrap();
        assert_eq!(cache.len(), 5, "the trail held an entry for every token");
    }

    /// Rolling back through a fused armed chunk must land exactly where rolling
    /// back through the reference scan lands, at every accepted prefix: the conv
    /// window bit-for-bit (both paths slice the same shifted stream) and the
    /// delta state within the fused scan's bound. This is the invariant the
    /// speculative verify walk rests on — the trail is only useful if the state
    /// it restores is the state the accepted tokens actually produced.
    #[test]
    fn fused_armed_rollback_lands_where_the_reference_scan_lands() {
        let Ok(metal) = crate::gguf::metal_device() else {
            return;
        };
        let _guard = path_lock();
        let (kh, vh, hidden, span) = (16usize, 32usize, 256usize, 4usize);
        let cfg = cfg_for(kh, vh, 128, hidden);
        let block = build_on(&random_block(&cfg, hidden, 0x500), &cfg, &metal);
        // Two committed tokens before the span, so the verify chunk starts from
        // a nonzero conv window and a nonzero delta state.
        let prefix = Tensor::from_vec(seeded_vec(2 * hidden, 0x501), (2, hidden), &metal).unwrap();
        let x = Tensor::from_vec(seeded_vec(span * hidden, 0x502), (span, hidden), &metal).unwrap();

        for commit in 0..=span {
            let run = |fused: bool| -> (Vec<f32>, Vec<f32>) {
                let mut cache = cache_on(&cfg, &metal, 32);
                block.forward(&prefix, &mut cache).unwrap();
                let ckpt = cache.checkpoint(span).unwrap();
                if fused {
                    block.forward(&x, &mut cache).unwrap();
                } else {
                    block.forward_classic(&x, &mut cache).unwrap();
                }
                cache.rollback(&ckpt, 2, span, commit).unwrap();
                assert_eq!(cache.len(), 2 + commit);
                let (conv, delta) = cache.linear_state().unwrap();
                (flat(&conv), flat(&delta))
            };
            let (fused_conv, fused_delta) = run(true);
            let (ref_conv, ref_delta) = run(false);
            assert_eq!(fused_conv, ref_conv, "commit {commit}: conv window");
            assert_all_close(
                &fused_delta,
                &ref_delta,
                2e-5,
                &format!("commit {commit} delta state"),
            );
        }
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

    /// Isolation timing for the 27B DeltaNet layer's NON-scan dispatches — the
    /// three f16 projections, the eager beta/alpha matmul, and the conv, ba and
    /// gnorm kernels. `delta_scan_timing` (src/ops/delta.rs) prices the scan
    /// itself; together they account for all eight dispatches of
    /// `forward_fused`, and the 27B runs 48 of these layers.
    ///
    /// The projections dominate the layer's FLOPs (qkv is [T,5120]x[5120,10240],
    /// z is [T,5120]x[5120,6144], out is [T,6144]x[6144,5120]) and, unlike the
    /// dense FFN, they DO get a dequantized f16 plane, so their achieved
    /// TFLOP/s is the reference point for what this machine's gemm can do.
    ///
    /// `#[ignore]`d — run on a `pgrep`-verified free GPU with:
    ///   cargo test --release -p xwen deltanet_prefill_stage_timing -- --ignored --nocapture
    /// `XWEN_BENCH_WARMUP` / `XWEN_BENCH_ITERS` override the loop counts.
    #[test]
    #[ignore = "perf bench"]
    fn deltanet_prefill_stage_timing() {
        use std::time::Instant;

        const HIDDEN: usize = 5120;
        const K_HEADS: usize = 16;
        const V_HEADS: usize = 48;
        const HD: usize = 128;
        const TAPS: usize = 4;
        const LAYERS: usize = 48;
        const CONV_DIM: usize = (2 * K_HEADS + V_HEADS) * HD; // 10240
        const V_DIM: usize = V_HEADS * HD; // 6144

        let device = crate::gguf::metal_device().unwrap();
        let Device::Metal(mdev) = &device else {
            unreachable!("metal_device() returned a non-Metal device")
        };
        let get = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let (warm, iters) = (get("XWEN_BENCH_WARMUP", 3), get("XWEN_BENCH_ITERS", 10));

        let bench = |f: &mut dyn FnMut()| -> f64 {
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
            times[iters / 2..].iter().sum::<f64>() / (iters - iters / 2) as f64
        };

        // Deterministic values in [lo, hi) from a plain LCG — timing is
        // value-independent for every stage here, so the only requirement is
        // that the buffers are real device allocations of the right shape.
        let rnd = |n: usize, seed: u64, lo: f32, hi: f32, shape: (usize, usize)| {
            let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v: Vec<f32> = (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    lo + (hi - lo) * ((s >> 33) as f32 / u32::MAX as f32)
                })
                .collect();
            Tensor::from_vec(v, shape, &device).unwrap()
        };
        let f16w = |rows: usize, cols: usize, seed: u64| {
            rnd(rows * cols, seed, -0.5, 0.5, (rows, cols))
                .to_dtype(candle_core::DType::F16)
                .unwrap()
        };

        let w_qkv = f16w(CONV_DIM, HIDDEN, 0x601);
        let w_z = f16w(V_DIM, HIDDEN, 0x602);
        let w_out = f16w(HIDDEN, V_DIM, 0x603);
        let ba_wt = rnd(
            HIDDEN * 2 * V_HEADS,
            0x604,
            -0.1,
            0.1,
            (HIDDEN, 2 * V_HEADS),
        );
        let conv_w = rnd(TAPS * CONV_DIM, 0x605, -1.0, 1.0, (TAPS, CONV_DIM));
        let conv_state = rnd(
            (TAPS - 1) * CONV_DIM,
            0x606,
            -1.0,
            1.0,
            (TAPS - 1, CONV_DIM),
        );
        let ssm_a = rnd(V_HEADS, 0x607, -3.0, -0.1, (1, V_HEADS))
            .flatten_all()
            .unwrap();
        let dt_bias = rnd(V_HEADS, 0x608, -2.0, 2.0, (1, V_HEADS))
            .flatten_all()
            .unwrap();
        let norm_w = rnd(HD, 0x609, 0.5, 1.5, (1, HD)).flatten_all().unwrap();

        for &t in &[512usize, 880, 3851] {
            let x = rnd(t * HIDDEN, 0x60a, -1.0, 1.0, (t, HIDDEN));
            let qkv = rnd(t * CONV_DIM, 0x60b, -2.0, 2.0, (t, CONV_DIM));
            let ba = rnd(t * 2 * V_HEADS, 0x60c, -20.0, 20.0, (t, 2 * V_HEADS));
            let o = rnd(t * V_DIM, 0x60d, -1.0, 1.0, (t, V_DIM))
                .reshape((t, V_HEADS, HD))
                .unwrap();
            let z = rnd(t * V_DIM, 0x60e, -1.0, 1.0, (t, V_DIM))
                .reshape((t, V_HEADS, HD))
                .unwrap();
            let ov = rnd(t * V_DIM, 0x60f, -1.0, 1.0, (t, V_DIM));
            let tf =
                |ms: f64, k: usize, n: usize| 2.0 * t as f64 * k as f64 * n as f64 / (ms * 1e9);

            let t_qkv = bench(&mut || {
                crate::ops::matmul_f16(&w_qkv, &x).unwrap();
            });
            let t_z = bench(&mut || {
                crate::ops::matmul_f16(&w_z, &x).unwrap();
            });
            let t_out = bench(&mut || {
                crate::ops::matmul_f16(&w_out, &ov).unwrap();
            });
            let t_ba_mm = bench(&mut || {
                x.matmul(&ba_wt).unwrap();
            });
            let t_conv = bench(&mut || {
                crate::ops::delta_conv(&conv_state, &qkv, &conv_w).unwrap();
            });
            let t_ba = bench(&mut || {
                crate::ops::delta_ba(&ba, &ssm_a, &dt_bias).unwrap();
            });
            let t_gnorm = bench(&mut || {
                crate::ops::delta_gnorm(&o, &z, &norm_w, 1e-6).unwrap();
            });

            // A real forward issues all 48 DeltaNet layers back to back, so the
            // per-call commit-and-wait each measurement above pays is overhead
            // the model never sees. Batch the whole non-scan sequence and sync
            // once; this is the rate to multiply by the layer count.
            const BATCH: usize = 8;
            let t_amort = bench(&mut || {
                let mut keep = Vec::with_capacity(BATCH);
                for _ in 0..BATCH {
                    let q = crate::ops::matmul_f16(&w_qkv, &x).unwrap();
                    let (conv, _) = crate::ops::delta_conv(&conv_state, &q, &conv_w).unwrap();
                    let raw = x.matmul(&ba_wt).unwrap();
                    crate::ops::delta_ba(&raw, &ssm_a, &dt_bias).unwrap();
                    let z = crate::ops::matmul_f16(&w_z, &x).unwrap();
                    let gn = crate::ops::delta_gnorm(
                        &conv
                            .narrow(1, 2 * K_HEADS * HD, V_DIM)
                            .unwrap()
                            .contiguous()
                            .unwrap()
                            .reshape((t, V_HEADS, HD))
                            .unwrap(),
                        &z.reshape((t, V_HEADS, HD)).unwrap(),
                        &norm_w,
                        1e-6,
                    )
                    .unwrap();
                    keep.push(
                        crate::ops::matmul_f16(&w_out, &gn.reshape((t, V_DIM)).unwrap()).unwrap(),
                    );
                }
            }) / BATCH as f64;

            let non_scan = t_qkv + t_z + t_out + t_ba_mm + t_conv + t_ba + t_gnorm;
            eprintln!("--- 27B DeltaNet non-scan, T={t} ---");
            eprintln!(
                "  qkv proj  {t_qkv:7.3} ms  {:6.2} TFLOP/s",
                tf(t_qkv, HIDDEN, CONV_DIM)
            );
            eprintln!(
                "  z proj    {t_z:7.3} ms  {:6.2} TFLOP/s",
                tf(t_z, HIDDEN, V_DIM)
            );
            eprintln!(
                "  out proj  {t_out:7.3} ms  {:6.2} TFLOP/s",
                tf(t_out, V_DIM, HIDDEN)
            );
            eprintln!(
                "  ba matmul {t_ba_mm:7.3} ms  {:6.2} TFLOP/s (candle eager)",
                tf(t_ba_mm, HIDDEN, 2 * V_HEADS)
            );
            eprintln!("  conv      {t_conv:7.3} ms");
            eprintln!("  ba head   {t_ba:7.3} ms");
            eprintln!("  gnorm     {t_gnorm:7.3} ms");
            eprintln!(
                "  NON-SCAN  {non_scan:7.3} ms/layer => {:.1} ms for {LAYERS} layers",
                non_scan * LAYERS as f64
            );
            eprintln!(
                "  AMORTIZED {t_amort:7.3} ms/layer => {:.1} ms for {LAYERS} layers",
                t_amort * LAYERS as f64
            );
        }
    }
}
