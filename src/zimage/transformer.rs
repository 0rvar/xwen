// Vendored from candle rev 21cca0b (candle-transformers/src/models/z_image/transformer.rs,
// PR #3261, SpenserCai), MIT / Apache-2.0. Changed here against the reference
// implementations: the caption is padded to a 32-multiple with the learned
// `cap_pad_token` AFTER the embedder and left unmasked (upstream zero-padded before it
// and masked); RoPE rotates in f32; the CUDA flash-attention arm is gone.
//! Z-Image Transformer (ZImageTransformer2DModel)
//!
//! Core transformer implementation for Z-Image text-to-image generation.
//! Batch 1 only, which is the one case this machine runs: at batch 1 every
//! sequence is its own length and no attention mask exists in either
//! reference implementation.

use candle_core::{D, DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{VarBuilder, linear, linear_no_bias};
use candle_transformers::models::with_tracing::RmsNorm;

// ==================== Constants ====================

/// AdaLN embedding dimension (256)
pub const ADALN_EMBED_DIM: usize = 256;
/// Sequence padding alignment (32)
pub const SEQ_MULTI_OF: usize = 32;
/// Frequency embedding size for timestep encoding
pub const FREQUENCY_EMBEDDING_SIZE: usize = 256;
/// Max period for sinusoidal encoding
pub const MAX_PERIOD: f64 = 10000.0;

/// The shipped `axes_lens`: how many positions each of the three RoPE tables
/// holds, and therefore the highest position any coordinate grid may name.
///
/// A position past the end of a table is NOT an error on Metal — candle's
/// `index_select` kernel clamps out-of-range ids to the last row rather than
/// failing (`candle-metal-kernels/src/metal_src/indexing.metal`, "Force
/// prevent out of bounds indexing") while the CPU backend errors. So an
/// oversized request would come back as a plausible image built from the
/// wrong rotations, on the shipped device only. The explicit checks in
/// [`ZImageTransformer2DModel::forward`] and
/// [`crate::zimage::pipeline::ZImagePipeline::check_size`] are the whole
/// defence; these are the numbers they check against when the caller has no
/// config in hand.
pub const AXES_LENS: [usize; 3] = [1536, 512, 512];

// ==================== Config ====================

/// Z-Image Transformer configuration
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    #[serde(default = "default_patch_size")]
    pub all_patch_size: Vec<usize>,
    #[serde(default = "default_f_patch_size")]
    pub all_f_patch_size: Vec<usize>,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_dim")]
    pub dim: usize,
    #[serde(default = "default_n_layers")]
    pub n_layers: usize,
    #[serde(default = "default_n_refiner_layers")]
    pub n_refiner_layers: usize,
    #[serde(default = "default_n_heads")]
    pub n_heads: usize,
    #[serde(default = "default_n_kv_heads")]
    pub n_kv_heads: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default = "default_qk_norm")]
    pub qk_norm: bool,
    #[serde(default = "default_cap_feat_dim")]
    pub cap_feat_dim: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_t_scale")]
    pub t_scale: f64,
    #[serde(default = "default_axes_dims")]
    pub axes_dims: Vec<usize>,
    #[serde(default = "default_axes_lens")]
    pub axes_lens: Vec<usize>,
    /// Whether attention runs through candle's fused Metal SDPA kernel (the
    /// default) or through the plain matmul-softmax-matmul chain, which is the
    /// reference arm for an A/B. Not a key in the shipped config.json, so the
    /// `serde` default is what decides it, and that default reads
    /// [`ATTN_ENV`] — `XWEN_ZIMAGE_ATTN=basic` selects the reference arm.
    #[serde(default = "default_use_accelerated_attn")]
    pub use_accelerated_attn: bool,
}

/// Padding length that takes `ori_len` up to the next multiple of
/// [`SEQ_MULTI_OF`] (zero when it already is one).
#[inline]
pub fn compute_padding_len(ori_len: usize) -> usize {
    (SEQ_MULTI_OF - (ori_len % SEQ_MULTI_OF)) % SEQ_MULTI_OF
}

/// The environment switch that picks the attention implementation, read when
/// a [`Config`] is built — by `serde` from `transformer/config.json`, which
/// carries no such key, or by [`Config::z_image_turbo`]. Same shape as
/// `XWEN_QWEN3_ATTN` on the language stack and for the same purpose: a
/// bisect arm that shares no attention kernel with the shipped one.
pub const ATTN_ENV: &str = "XWEN_ZIMAGE_ATTN";

/// Which attention chain a built [`Config`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnImpl {
    /// The shipped path on Metal: candle's fused SDPA kernel.
    Fused,
    /// The explicit chain — Q·Kᵀ, additive mask, softmax, P·V — through
    /// candle's plain matmul and softmax. It is the reference arm: it shares
    /// no attention kernel with the fused one, so an A/B between them is a
    /// comparison of two computations rather than of one kernel with itself.
    Basic,
}

impl AttnImpl {
    /// Resolve from [`ATTN_ENV`]: unset means the shipped path, anything else
    /// must name an arm.
    pub fn from_env() -> Result<Self> {
        match std::env::var(ATTN_ENV) {
            Err(std::env::VarError::NotPresent) => Ok(Self::Fused),
            Err(std::env::VarError::NotUnicode(_)) => {
                candle_core::bail!("{ATTN_ENV} is not valid UTF-8")
            }
            Ok(value) => Self::parse(&value),
        }
    }

    /// [`Self::from_env`] with a bad value read as the shipped path, for the
    /// `serde` default, which cannot fail. Nothing runs on that reading:
    /// `ZImagePipeline::load` calls [`Self::from_env`] with a `?` before it
    /// opens anything, so a typo in a bisect run is a load error rather than
    /// a silent measurement of the wrong arm.
    pub fn from_env_or_default() -> Self {
        Self::from_env().unwrap_or(Self::Fused)
    }

    /// `fused` / `flash` (or empty) select the shipped path, `basic` the
    /// reference chain; anything else is refused rather than defaulted.
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "fused" | "flash" | "sdpa" => Ok(Self::Fused),
            "basic" => Ok(Self::Basic),
            other => candle_core::bail!(
                "{ATTN_ENV}={other:?}: expected `fused` (the default) or `basic`"
            ),
        }
    }

    /// Whether this arm is the fused kernel, which is how [`Config`] stores it.
    pub fn is_accelerated(self) -> bool {
        matches!(self, Self::Fused)
    }

    /// The name a dump or a log records for provenance.
    pub fn label(self) -> &'static str {
        match self {
            Self::Fused => "fused",
            Self::Basic => "basic",
        }
    }
}

fn default_use_accelerated_attn() -> bool {
    AttnImpl::from_env_or_default().is_accelerated()
}

fn default_patch_size() -> Vec<usize> {
    vec![2]
}
fn default_f_patch_size() -> Vec<usize> {
    vec![1]
}
/// The latent channel count, `in_channels` in the shipped
/// `transformer/config.json` and the VAE's `latent_channels`. A constant
/// because the shape of an injected latent has to be checkable before any
/// config is open; the loaded config stays authoritative for what runs.
pub const LATENT_CHANNELS: usize = 16;

fn default_in_channels() -> usize {
    LATENT_CHANNELS
}
fn default_dim() -> usize {
    3840
}
fn default_n_layers() -> usize {
    30
}
fn default_n_refiner_layers() -> usize {
    2
}
fn default_n_heads() -> usize {
    30
}
fn default_n_kv_heads() -> usize {
    30
}
fn default_norm_eps() -> f64 {
    1e-5
}
fn default_qk_norm() -> bool {
    true
}
fn default_cap_feat_dim() -> usize {
    2560
}
fn default_rope_theta() -> f64 {
    256.0
}
fn default_t_scale() -> f64 {
    1000.0
}
fn default_axes_dims() -> Vec<usize> {
    vec![32, 48, 48]
}
fn default_axes_lens() -> Vec<usize> {
    AXES_LENS.to_vec()
}

impl Config {
    /// Create configuration for Z-Image Turbo model
    pub fn z_image_turbo() -> Self {
        Self {
            all_patch_size: vec![2],
            all_f_patch_size: vec![1],
            in_channels: LATENT_CHANNELS,
            dim: 3840,
            n_layers: 30,
            n_refiner_layers: 2,
            n_heads: 30,
            n_kv_heads: 30,
            norm_eps: 1e-5,
            qk_norm: true,
            cap_feat_dim: 2560,
            rope_theta: 256.0,
            t_scale: 1000.0,
            axes_dims: vec![32, 48, 48],
            axes_lens: AXES_LENS.to_vec(),
            use_accelerated_attn: AttnImpl::from_env_or_default().is_accelerated(),
        }
    }

    /// Which attention chain this config runs.
    pub fn attn_impl(&self) -> AttnImpl {
        if self.use_accelerated_attn {
            AttnImpl::Fused
        } else {
            AttnImpl::Basic
        }
    }

    /// Pick the attention chain explicitly, overriding what [`ATTN_ENV`] said
    /// when this config was built. What a test uses to run both arms in one
    /// process without touching the environment its siblings share.
    pub fn set_attn_impl(&mut self, attn: AttnImpl) {
        self.use_accelerated_attn = attn.is_accelerated();
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    /// Get hidden dimension for FFN
    /// Matches Python: int(dim / 3 * 8) = 10240 for dim=3840
    pub fn hidden_dim(&self) -> usize {
        (self.dim / 3) * 8
    }
}

// ==================== TimestepEmbedder ====================

/// Timestep embedding using sinusoidal encoding + MLP
#[derive(Debug, Clone)]
pub struct TimestepEmbedder {
    linear1: candle_nn::Linear,
    linear2: candle_nn::Linear,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    pub fn new(out_size: usize, mid_size: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = linear(FREQUENCY_EMBEDDING_SIZE, mid_size, vb.pp("mlp").pp("0"))?;
        let linear2 = linear(mid_size, out_size, vb.pp("mlp").pp("2"))?;
        Ok(Self {
            linear1,
            linear2,
            frequency_embedding_size: FREQUENCY_EMBEDDING_SIZE,
        })
    }

    fn timestep_embedding(&self, t: &Tensor, device: &Device, dtype: DType) -> Result<Tensor> {
        let half = self.frequency_embedding_size / 2;
        let freqs = Tensor::arange(0u32, half as u32, device)?.to_dtype(DType::F32)?;
        let freqs = (freqs * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
        let args = t
            .unsqueeze(1)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&freqs.unsqueeze(0)?)?;
        let embedding = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?;
        embedding.to_dtype(dtype)
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let device = t.device();
        let dtype = self.linear1.weight().dtype();
        let t_freq = self.timestep_embedding(t, device, dtype)?;
        t_freq.apply(&self.linear1)?.silu()?.apply(&self.linear2)
    }
}

// ==================== FeedForward (SwiGLU) ====================

/// SwiGLU feedforward network
#[derive(Debug, Clone)]
pub struct FeedForward {
    w1: candle_nn::Linear,
    w2: candle_nn::Linear,
    w3: candle_nn::Linear,
}

impl FeedForward {
    pub fn new(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let w1 = linear_no_bias(dim, hidden_dim, vb.pp("w1"))?;
        let w2 = linear_no_bias(hidden_dim, dim, vb.pp("w2"))?;
        let w3 = linear_no_bias(dim, hidden_dim, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = x.apply(&self.w1)?.silu()?;
        let x3 = x.apply(&self.w3)?;
        (x1 * x3)?.apply(&self.w2)
    }
}

// ==================== QkNorm ====================

/// QK normalization using RMSNorm
#[derive(Debug, Clone)]
pub struct QkNorm {
    norm_q: RmsNorm,
    norm_k: RmsNorm,
}

impl QkNorm {
    pub fn new(head_dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let norm_q = RmsNorm::new(head_dim, eps, vb.pp("norm_q"))?;
        let norm_k = RmsNorm::new(head_dim, eps, vb.pp("norm_k"))?;
        Ok(Self { norm_q, norm_k })
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        // q, k shape: (B, seq_len, n_heads, head_dim)
        let q = self.norm_q.forward(q)?;
        let k = self.norm_k.forward(k)?;
        Ok((q, k))
    }
}

// ==================== RopeEmbedder (3D) ====================

/// 3D Rotary Position Embedding for video/image generation
#[derive(Debug, Clone)]
pub struct RopeEmbedder {
    #[allow(dead_code)]
    theta: f64,
    axes_dims: Vec<usize>,
    #[allow(dead_code)]
    axes_lens: Vec<usize>,
    /// Pre-computed cos cache per axis
    cos_cached: Vec<Tensor>,
    /// Pre-computed sin cache per axis
    sin_cached: Vec<Tensor>,
}

impl RopeEmbedder {
    /// The tables are f32 whatever the model dtype: the reference builds them
    /// in float64 and rotates in float32, and a bf16 table at theta 256 loses
    /// the high positions.
    pub fn new(
        theta: f64,
        axes_dims: Vec<usize>,
        axes_lens: Vec<usize>,
        device: &Device,
    ) -> Result<Self> {
        assert_eq!(axes_dims.len(), axes_lens.len());
        let mut cos_cached = Vec::with_capacity(axes_dims.len());
        let mut sin_cached = Vec::with_capacity(axes_dims.len());

        for (d, e) in axes_dims.iter().zip(axes_lens.iter()) {
            let half_d = d / 2;
            // float64 like the reference, then cast: the table is what a
            // position rotates by, and the cast happens once here rather
            // than once per angle.
            let mut table = Vec::with_capacity(e * half_d * 2);
            for pos in 0..*e {
                for i in 0..half_d {
                    let inv_freq = 1.0 / theta.powf((2 * i) as f64 / *d as f64);
                    let angle = (pos as f64 * inv_freq) as f32;
                    table.push(angle);
                }
            }
            let angles = Tensor::from_vec(table, (*e, half_d), device)?;
            cos_cached.push(angles.cos()?);
            sin_cached.push(angles.sin()?);
        }

        Ok(Self {
            theta,
            axes_dims,
            axes_lens,
            cos_cached,
            sin_cached,
        })
    }

    /// Get RoPE cos/sin from position IDs
    /// ids: (seq_len, 3) - [frame_id, height_id, width_id]
    ///
    /// Every id must already be inside its axis's `axes_lens`: the
    /// `index_select` below CLAMPS an out-of-range id to the table's last row
    /// on Metal instead of failing, so a caller that skips the check gets
    /// wrong rotations rather than an error. The one caller,
    /// [`ZImageTransformer2DModel::forward`], checks the grid extents before
    /// building the ids, which costs nothing per step; reading the ids back
    /// here to check them would cost a device sync per call.
    pub fn forward(&self, ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut cos_parts = Vec::with_capacity(self.axes_dims.len());
        let mut sin_parts = Vec::with_capacity(self.axes_dims.len());

        for (i, _) in self.axes_dims.iter().enumerate() {
            let axis_ids = ids.i((.., i))?.contiguous()?; // (seq_len,) - must be contiguous for Metal
            let cos_i = self.cos_cached[i].index_select(&axis_ids, 0)?;
            let sin_i = self.sin_cached[i].index_select(&axis_ids, 0)?;
            cos_parts.push(cos_i);
            sin_parts.push(sin_i);
        }

        let cos = Tensor::cat(&cos_parts, D::Minus1)?; // (seq_len, head_dim/2)
        let sin = Tensor::cat(&sin_parts, D::Minus1)?;
        Ok((cos, sin))
    }
}

/// Apply RoPE (real-number form, equivalent to PyTorch complex multiplication)
///
/// INTERLEAVED pairs — dims `(2i, 2i+1)` rotate together, as
/// `view_as_complex(x.reshape(..., -1, 2))` pairs them — not the NEoX
/// split-half form every Qwen rope kernel in this repo uses. Rotates in f32
/// and returns `x`'s dtype, as the reference does (`x_in.float()` ...
/// `.type_as(x_in)`).
///
/// x: (B, seq_len, n_heads, head_dim)
/// cos, sin: (seq_len, head_dim/2), f32
pub fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, seq_len, n_heads, head_dim) = x.dims4()?;
    let half_dim = head_dim / 2;
    let x_dtype = x.dtype();

    // Reshape x to interleaved real/imag form: (B, seq_len, n_heads, half_dim, 2)
    let x = x
        .to_dtype(DType::F32)?
        .reshape((b, seq_len, n_heads, half_dim, 2))?;

    // Extract real and imag parts
    let x_real = x.i((.., .., .., .., 0))?; // (B, seq_len, n_heads, half_dim)
    let x_imag = x.i((.., .., .., .., 1))?;

    // Expand cos/sin for broadcasting: (seq_len, half_dim) -> (1, seq_len, 1, half_dim)
    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

    // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    let y_real = (x_real.broadcast_mul(&cos)? - x_imag.broadcast_mul(&sin)?)?;
    let y_imag = (x_real.broadcast_mul(&sin)? + x_imag.broadcast_mul(&cos)?)?;

    // Interleave back
    Tensor::stack(&[y_real, y_imag], D::Minus1)?
        .reshape((b, seq_len, n_heads, head_dim))?
        .to_dtype(x_dtype)
}

// ==================== ZImageAttention ====================

/// Z-Image attention with QK normalization and 3D RoPE
#[derive(Debug, Clone)]
pub struct ZImageAttention {
    to_q: candle_nn::Linear,
    to_k: candle_nn::Linear,
    to_v: candle_nn::Linear,
    to_out: candle_nn::Linear,
    qk_norm: Option<QkNorm>,
    n_heads: usize,
    head_dim: usize,
    use_accelerated_attn: bool,
}

impl ZImageAttention {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.dim;
        let n_heads = cfg.n_heads;
        let head_dim = cfg.head_dim();

        let to_q = linear_no_bias(dim, n_heads * head_dim, vb.pp("to_q"))?;
        let to_k = linear_no_bias(dim, cfg.n_kv_heads * head_dim, vb.pp("to_k"))?;
        let to_v = linear_no_bias(dim, cfg.n_kv_heads * head_dim, vb.pp("to_v"))?;
        let to_out = linear_no_bias(n_heads * head_dim, dim, vb.pp("to_out").pp("0"))?;

        let qk_norm = if cfg.qk_norm {
            Some(QkNorm::new(head_dim, cfg.norm_eps, vb.clone())?)
        } else {
            None
        };

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            qk_norm,
            n_heads,
            head_dim,
            use_accelerated_attn: cfg.use_accelerated_attn,
        })
    }

    /// Bidirectional attention over the whole sequence. `attention_mask` is
    /// a `(B, seq_len)` key keep-mask (1 = attend, 0 = ignore) for callers that
    /// batch sequences of unequal length; the pipeline runs batch 1 and passes
    /// `None`, which is what both reference implementations do there.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = hidden_states.dims3()?;

        // Project to Q, K, V
        let q = hidden_states.apply(&self.to_q)?;
        let k = hidden_states.apply(&self.to_k)?;
        let v = hidden_states.apply(&self.to_v)?;

        // Reshape: (B, seq_len, n_heads * head_dim) -> (B, seq_len, n_heads, head_dim)
        let q = q.reshape((b, seq_len, self.n_heads, self.head_dim))?;
        let k = k.reshape((b, seq_len, self.n_heads, self.head_dim))?;
        let v = v.reshape((b, seq_len, self.n_heads, self.head_dim))?;

        // Apply QK norm
        let (q, k) = if let Some(ref norm) = self.qk_norm {
            norm.forward(&q, &k)?
        } else {
            (q, k)
        };

        // Apply RoPE
        let q = apply_rotary_emb(&q, cos, sin)?;
        let k = apply_rotary_emb(&k, cos, sin)?;

        // Transpose for attention: (B, n_heads, seq_len, head_dim)
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();

        let context = if self.use_accelerated_attn && hidden_states.device().is_metal() {
            self.attention_metal(&q, &k, &v, attention_mask, scale)?
        } else {
            self.attention_basic(&q, &k, &v, attention_mask, scale)?
        };

        // Reshape back: (B, n_heads, seq_len, head_dim) -> (B, seq_len, dim)
        let context = context.transpose(1, 2)?.reshape((b, seq_len, ()))?;

        context.apply(&self.to_out)
    }

    /// Metal: candle's fused SDPA kernel (bf16/f16/f32, head_dim 128).
    fn attention_metal(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f64,
    ) -> Result<Tensor> {
        let sdpa_mask = self.prepare_sdpa_mask(mask, q)?;
        candle_nn::ops::sdpa(q, k, v, sdpa_mask.as_ref(), false, scale as f32, 1.0)
    }

    /// The plain chain: scores, softmax, weighted sum. The reference arm.
    fn attention_basic(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f64,
    ) -> Result<Tensor> {
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(m) = mask {
            // mask: (B, seq_len) -> (B, 1, 1, seq_len)
            let m = m.unsqueeze(1)?.unsqueeze(2)?;
            let m = m.to_dtype(attn_weights.dtype())?;
            // 1=valid, 0=padding -> 0=valid, -inf=padding
            let m = ((m - 1.0)? * 1e9)?;
            attn_weights = attn_weights.broadcast_add(&m)?;
        }

        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        attn_probs.matmul(v)
    }

    /// The additive `(B, n_heads, seq_len, seq_len)` mask candle's Metal SDPA
    /// takes, from a `(B, seq_len)` keep-mask.
    ///
    /// The four-axis shape is not a waste and it is not optional. It is not a
    /// waste because `broadcast_as` sets a stride of 0 on the two expanded
    /// axes and copies nothing (`candle-core/src/layout.rs`
    /// `Layout::broadcast_as`), and candle hands those strides to the kernel
    /// rather than making the mask contiguous first. It is not optional
    /// because that kernel refuses anything else: `candle-nn/src/ops.rs`
    /// bails with "Mask shape must be (bs, qheads, qseq, kseq)" unless the
    /// dims equal `[b, q_heads, q_seq, k_seq]` exactly, so the
    /// `(B, 1, 1, seq_len)` form the reference's torch SDPA broadcasts
    /// internally is an error here. [`Self::attention_basic`] takes that
    /// broadcastable form instead, `broadcast_add` being happy with it.
    ///
    /// Neither mask branch is reachable from `xwen image`: every call site
    /// passes `None`, batch 1 having no ragged sequence to mask. The tests
    /// cover them so that whoever enables the batched or image-padded path
    /// starts from something that has run.
    fn prepare_sdpa_mask(&self, mask: Option<&Tensor>, q: &Tensor) -> Result<Option<Tensor>> {
        match mask {
            Some(m) => {
                let (b, _, seq_len, _) = q.dims4()?;
                let m = m.unsqueeze(1)?.unsqueeze(2)?;
                let m = m.to_dtype(q.dtype())?;
                // SDPA uses additive mask: 0=valid, -inf=masked
                let m = ((m - 1.0)? * 1e9)?;
                let m = m.broadcast_as((b, self.n_heads, seq_len, seq_len))?;
                Ok(Some(m))
            }
            None => Ok(None),
        }
    }
}

// ==================== ZImageTransformerBlock ====================

/// Z-Image transformer block with optional AdaLN modulation
#[derive(Debug, Clone)]
pub struct ZImageTransformerBlock {
    attention: ZImageAttention,
    feed_forward: FeedForward,
    attention_norm1: RmsNorm,
    attention_norm2: RmsNorm,
    ffn_norm1: RmsNorm,
    ffn_norm2: RmsNorm,
    adaln_modulation: Option<candle_nn::Linear>,
}

impl ZImageTransformerBlock {
    pub fn new(cfg: &Config, modulation: bool, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.dim;
        let hidden_dim = cfg.hidden_dim();

        let attention = ZImageAttention::new(cfg, vb.pp("attention"))?;
        let feed_forward = FeedForward::new(dim, hidden_dim, vb.pp("feed_forward"))?;

        let attention_norm1 = RmsNorm::new(dim, cfg.norm_eps, vb.pp("attention_norm1"))?;
        let attention_norm2 = RmsNorm::new(dim, cfg.norm_eps, vb.pp("attention_norm2"))?;
        let ffn_norm1 = RmsNorm::new(dim, cfg.norm_eps, vb.pp("ffn_norm1"))?;
        let ffn_norm2 = RmsNorm::new(dim, cfg.norm_eps, vb.pp("ffn_norm2"))?;

        let adaln_modulation = if modulation {
            let adaln_dim = dim.min(ADALN_EMBED_DIM);
            Some(linear(
                adaln_dim,
                4 * dim,
                vb.pp("adaLN_modulation").pp("0"),
            )?)
        } else {
            None
        };

        Ok(Self {
            attention,
            feed_forward,
            attention_norm1,
            attention_norm2,
            ffn_norm1,
            ffn_norm2,
            adaln_modulation,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        adaln_input: Option<&Tensor>,
    ) -> Result<Tensor> {
        if let Some(ref adaln) = self.adaln_modulation {
            let adaln_input = adaln_input.expect("adaln_input required when modulation=true");
            // (B, 256) -> (B, 4*dim) -> (B, 1, 4*dim) -> chunk into 4
            let modulation = adaln_input.apply(adaln)?.unsqueeze(1)?;
            let chunks = modulation.chunk(4, D::Minus1)?;
            let (scale_msa, gate_msa, scale_mlp, gate_mlp) =
                (&chunks[0], &chunks[1], &chunks[2], &chunks[3]);

            // Apply tanh gate
            let gate_msa = gate_msa.tanh()?;
            let gate_mlp = gate_mlp.tanh()?;
            let scale_msa = (scale_msa + 1.0)?;
            let scale_mlp = (scale_mlp + 1.0)?;

            // Attention block
            let normed = self.attention_norm1.forward(x)?;
            let scaled = normed.broadcast_mul(&scale_msa)?;
            let attn_out = self.attention.forward(&scaled, attn_mask, cos, sin)?;
            let attn_out = self.attention_norm2.forward(&attn_out)?;
            let x = (x + gate_msa.broadcast_mul(&attn_out)?)?;

            // FFN block
            let normed = self.ffn_norm1.forward(&x)?;
            let scaled = normed.broadcast_mul(&scale_mlp)?;
            let ffn_out = self.feed_forward.forward(&scaled)?;
            let ffn_out = self.ffn_norm2.forward(&ffn_out)?;
            x + gate_mlp.broadcast_mul(&ffn_out)?
        } else {
            // Without modulation
            let normed = self.attention_norm1.forward(x)?;
            let attn_out = self.attention.forward(&normed, attn_mask, cos, sin)?;
            let attn_out = self.attention_norm2.forward(&attn_out)?;
            let x = (x + attn_out)?;

            let normed = self.ffn_norm1.forward(&x)?;
            let ffn_out = self.feed_forward.forward(&normed)?;
            let ffn_out = self.ffn_norm2.forward(&ffn_out)?;
            x + ffn_out
        }
    }
}

// ==================== FinalLayer ====================

/// LayerNorm without learnable parameters (elementwise_affine=False)
#[derive(Debug, Clone)]
pub struct LayerNormNoParams {
    eps: f64,
}

impl LayerNormNoParams {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }
}

impl Module for LayerNormNoParams {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        // Subtract mean
        let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        // Divide by std
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(x_dtype)
    }
}

/// Final layer for output projection
#[derive(Debug, Clone)]
pub struct FinalLayer {
    norm_final: LayerNormNoParams,
    linear: candle_nn::Linear,
    adaln_silu: candle_nn::Linear,
}

impl FinalLayer {
    pub fn new(hidden_size: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let norm_final = LayerNormNoParams::new(1e-6);
        let linear = candle_nn::linear(hidden_size, out_channels, vb.pp("linear"))?;
        let adaln_dim = hidden_size.min(ADALN_EMBED_DIM);
        let adaln_silu =
            candle_nn::linear(adaln_dim, hidden_size, vb.pp("adaLN_modulation").pp("1"))?;

        Ok(Self {
            norm_final,
            linear,
            adaln_silu,
        })
    }

    pub fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let scale = c.silu()?.apply(&self.adaln_silu)?;
        let scale = (scale + 1.0)?.unsqueeze(1)?;
        let x = self.norm_final.forward(x)?.broadcast_mul(&scale)?;
        x.apply(&self.linear)
    }
}

// ==================== Patchify / Unpatchify ====================

/// Convert image to patch sequence
/// Matches Python: image.view(C, F_t, pF, H_t, pH, W_t, pW).permute(1,3,5,2,4,6,0)
///
/// For Z-Image with F=1, pF=1, we optimize to use 6D operations.
/// input: (B, C, 1, H, W)
/// output: (B, num_patches, patch_dim), (F, H, W) original size
pub fn patchify(
    x: &Tensor,
    patch_size: usize,
    f_patch_size: usize,
) -> Result<(Tensor, (usize, usize, usize))> {
    let (b, c, f, h, w) = x.dims5()?;
    let ph = patch_size;
    let pw = patch_size;
    let pf = f_patch_size;

    let f_tokens = f / pf;
    let h_tokens = h / ph;
    let w_tokens = w / pw;
    let num_patches = f_tokens * h_tokens * w_tokens;
    let patch_dim = pf * ph * pw * c;

    // For F=1, pF=1 case (image generation), use optimized 6D path
    if f == 1 && pf == 1 {
        // Step 1: Squeeze F dimension: (B, C, 1, H, W) -> (B, C, H, W)
        let x = x.squeeze(2)?;

        // Step 2: Reshape H into (H_tokens, pH): (B, C, H, W) -> (B, C, H_t, pH, W)
        let x = x.reshape((b, c, h_tokens, ph, w))?;

        // Step 3: Reshape W into (W_tokens, pW): (B, C, H_t, pH, W) -> (B, C, H_t, pH, W_t, pW)
        let x = x.reshape((b, c, h_tokens, ph, w_tokens, pw))?;

        // Step 4: Permute to match Python: (C, H_t, pH, W_t, pW) -> (H_t, W_t, pH, pW, C)
        // For batch: (B, C, H_t, pH, W_t, pW) -> (B, H_t, W_t, pH, pW, C)
        // Permutation: (0, 2, 4, 3, 5, 1)
        let x = x.permute((0, 2, 4, 3, 5, 1))?;

        // Step 5: Reshape to patches: (B, H_t, W_t, pH, pW, C) -> (B, H_t*W_t, pH*pW*C)
        let x = x.reshape((b, num_patches, patch_dim))?;

        Ok((x, (f, h, w)))
    } else {
        // General case: use contiguous + reshape approach
        // This is less common for Z-Image image generation
        let x = x.permute((0, 2, 3, 4, 1))?.contiguous()?; // (B, F, H, W, C)
        let x = x.reshape((b, f_tokens, pf, h_tokens, ph, w_tokens * pw * c))?;
        let x = x.permute((0, 1, 3, 5, 2, 4))?.contiguous()?;
        let x = x.reshape((b, num_patches, patch_dim))?;
        Ok((x, (f, h, w)))
    }
}

/// Convert patch sequence back to image
/// Matches Python: x.view(F_t, H_t, W_t, pF, pH, pW, C).permute(6,0,3,1,4,2,5)
///
/// For Z-Image with F=1, pF=1, we optimize to use 6D operations.
/// input: (B, seq_len, patch_dim)
/// output: (B, C, F, H, W)
pub fn unpatchify(
    x: &Tensor,
    size: (usize, usize, usize),
    patch_size: usize,
    f_patch_size: usize,
    out_channels: usize,
) -> Result<Tensor> {
    let (f, h, w) = size;
    let ph = patch_size;
    let pw = patch_size;
    let pf = f_patch_size;

    let f_tokens = f / pf;
    let h_tokens = h / ph;
    let w_tokens = w / pw;
    let ori_len = f_tokens * h_tokens * w_tokens;

    let (b, _, _) = x.dims3()?;
    let x = x.narrow(1, 0, ori_len)?; // Remove padding

    // For F=1, pF=1 case (image generation), use optimized 6D path
    if f == 1 && pf == 1 {
        // Step 1: Reshape to (B, H_t, W_t, pH, pW, C)
        let x = x.reshape((b, h_tokens, w_tokens, ph, pw, out_channels))?;

        // Step 2: Permute to match Python: (H_t, W_t, pH, pW, C) -> (C, H_t, pH, W_t, pW)
        // For batch: (B, H_t, W_t, pH, pW, C) -> (B, C, H_t, pH, W_t, pW)
        // Permutation: (0, 5, 1, 3, 2, 4)
        let x = x.permute((0, 5, 1, 3, 2, 4))?;

        // Step 3: Reshape to combine H and W: (B, C, H_t, pH, W_t, pW) -> (B, C, H, W)
        let x = x.reshape((b, out_channels, h, w))?;

        // Step 4: Add back F dimension: (B, C, H, W) -> (B, C, 1, H, W)
        let x = x.unsqueeze(2)?;

        Ok(x)
    } else {
        // General case
        let x = x.reshape((b, f_tokens, h_tokens, w_tokens, pf * ph * pw * out_channels))?;
        let x = x.reshape((b, f_tokens, h_tokens, w_tokens * pf, ph, pw * out_channels))?;
        let x = x.permute((0, 5, 1, 3, 2, 4))?.contiguous()?;
        let x = x.reshape((b, out_channels, f, h, w))?;
        Ok(x)
    }
}

/// Create 3D coordinate grid for RoPE position IDs
/// size: (F, H, W)
/// start: (f0, h0, w0)
/// output: (F*H*W, 3)
pub fn create_coordinate_grid(
    size: (usize, usize, usize),
    start: (usize, usize, usize),
    device: &Device,
) -> Result<Tensor> {
    let (f, h, w) = size;
    let (f0, h0, w0) = start;

    let mut coords = Vec::with_capacity(f * h * w * 3);
    for fi in 0..f {
        for hi in 0..h {
            for wi in 0..w {
                coords.push((f0 + fi) as u32);
                coords.push((h0 + hi) as u32);
                coords.push((w0 + wi) as u32);
            }
        }
    }

    Tensor::from_vec(coords, (f * h * w, 3), device)
}

// ==================== ZImageTransformer2DModel ====================

/// Z-Image Transformer 2D Model
#[derive(Debug, Clone)]
pub struct ZImageTransformer2DModel {
    t_embedder: TimestepEmbedder,
    cap_embedder_norm: RmsNorm,
    cap_embedder_linear: candle_nn::Linear,
    x_embedder: candle_nn::Linear,
    final_layer: FinalLayer,
    /// Learned embedding for the image rows that pad the token count up to a
    /// multiple of [`SEQ_MULTI_OF`]. They attend and are attended to, unmasked.
    x_pad_token: Tensor,
    /// The same for the caption rows.
    cap_pad_token: Tensor,
    noise_refiner: Vec<ZImageTransformerBlock>,
    context_refiner: Vec<ZImageTransformerBlock>,
    layers: Vec<ZImageTransformerBlock>,
    rope_embedder: RopeEmbedder,
    cfg: Config,
}

impl ZImageTransformer2DModel {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device();

        // TimestepEmbedder
        let adaln_dim = cfg.dim.min(ADALN_EMBED_DIM);
        let t_embedder = TimestepEmbedder::new(adaln_dim, 1024, vb.pp("t_embedder"))?;

        // Caption embedder
        let cap_embedder_norm = RmsNorm::new(
            cfg.cap_feat_dim,
            cfg.norm_eps,
            vb.pp("cap_embedder").pp("0"),
        )?;
        let cap_embedder_linear = linear(cfg.cap_feat_dim, cfg.dim, vb.pp("cap_embedder").pp("1"))?;

        // Patch embedder (assuming patch_size=2, f_patch_size=1)
        let patch_dim = cfg.all_f_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.in_channels;
        let x_embedder = linear(patch_dim, cfg.dim, vb.pp("all_x_embedder").pp("2-1"))?;

        // Final layer
        let out_channels = cfg.all_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.all_f_patch_size[0]
            * cfg.in_channels;
        let final_layer =
            FinalLayer::new(cfg.dim, out_channels, vb.pp("all_final_layer").pp("2-1"))?;

        // Pad tokens
        let x_pad_token = vb.get((1, cfg.dim), "x_pad_token")?;
        let cap_pad_token = vb.get((1, cfg.dim), "cap_pad_token")?;

        // Noise refiner (with modulation)
        let mut noise_refiner = Vec::with_capacity(cfg.n_refiner_layers);
        for i in 0..cfg.n_refiner_layers {
            noise_refiner.push(ZImageTransformerBlock::new(
                cfg,
                true,
                vb.pp("noise_refiner").pp(i),
            )?);
        }

        // Context refiner (without modulation)
        let mut context_refiner = Vec::with_capacity(cfg.n_refiner_layers);
        for i in 0..cfg.n_refiner_layers {
            context_refiner.push(ZImageTransformerBlock::new(
                cfg,
                false,
                vb.pp("context_refiner").pp(i),
            )?);
        }

        // Main layers (with modulation)
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(ZImageTransformerBlock::new(
                cfg,
                true,
                vb.pp("layers").pp(i),
            )?);
        }

        // RoPE embedder
        let rope_embedder = RopeEmbedder::new(
            cfg.rope_theta,
            cfg.axes_dims.clone(),
            cfg.axes_lens.clone(),
            device,
        )?;

        Ok(Self {
            t_embedder,
            cap_embedder_norm,
            cap_embedder_linear,
            x_embedder,
            final_layer,
            x_pad_token,
            cap_pad_token,
            noise_refiner,
            context_refiner,
            layers,
            rope_embedder,
            cfg: cfg.clone(),
        })
    }

    /// One denoising forward at batch 1.
    ///
    /// * `x` - the latent, `(1, C, F, H, W)`, in the model dtype
    /// * `t` - the timestep, `(1,)`, already `1 - σ` and in `[0, 1]`
    /// * `cap_feats` - the caption's encoder hidden states, `(1, T, cap_feat_dim)`
    ///
    /// The sequence the 30 main layers see is `[image tokens, caption tokens]`,
    /// each run independently padded up to a multiple of [`SEQ_MULTI_OF`] with
    /// its learned pad token AFTER its embedder, and nothing masked. Caption
    /// positions count `1..=cap_len` on axis 0; image tokens all sit at
    /// `cap_len + 1` on axis 0 and take their `(h, w)` on axes 1 and 2. The
    /// image token count has to be a multiple of 32 already (true at every
    /// size whose 16-pixel grid has a multiple-of-32 cell count, 1024x1024
    /// included): padded image rows would sit at position `(0, 0, 0)`, and
    /// that path has no reference comparison yet, so it is refused rather
    /// than run unverified.
    pub fn forward(&self, x: &Tensor, t: &Tensor, cap_feats: &Tensor) -> Result<Tensor> {
        let device = x.device();
        let (b, _c, f, h, w) = x.dims5()?;
        if b != 1 || cap_feats.dim(0)? != 1 {
            candle_core::bail!("the Z-Image transformer runs batch 1 only");
        }
        let patch_size = self.cfg.all_patch_size[0];
        let f_patch_size = self.cfg.all_f_patch_size[0];

        // 1. Timestep embedding
        let t_scaled = (t * self.cfg.t_scale)?;
        let adaln_input = self.t_embedder.forward(&t_scaled)?; // (B, 256)

        // 2. Patchify and embed image
        let (x_patches, orig_size) = patchify(x, patch_size, f_patch_size)?;
        let mut x = x_patches.apply(&self.x_embedder)?; // (B, img_seq, dim)
        let img_seq_len = x.dim(1)?;
        if compute_padding_len(img_seq_len) != 0 {
            candle_core::bail!(
                "{img_seq_len} image tokens is not a multiple of {SEQ_MULTI_OF}; a size whose \
                 token grid needs padding is not supported yet"
            );
        }

        // 3. Caption embedding, then the learned pad rows up to a 32-multiple
        let cap_normed = self.cap_embedder_norm.forward(cap_feats)?;
        let mut cap = cap_normed.apply(&self.cap_embedder_linear)?; // (B, text_len, dim)
        let text_len = cap.dim(1)?;
        let cap_pad = compute_padding_len(text_len);
        if cap_pad > 0 {
            let pad = self
                .cap_pad_token
                .to_dtype(cap.dtype())?
                .unsqueeze(0)?
                .broadcast_as((b, cap_pad, self.cfg.dim))?
                .contiguous()?;
            cap = Tensor::cat(&[&cap, &pad], 1)?;
        }
        let cap_len = text_len + cap_pad;

        // 4. Position ids: caption 1..=cap_len on axis 0; image at cap_len + 1
        let f_tokens = f / f_patch_size;
        let h_tokens = h / patch_size;
        let w_tokens = w / patch_size;
        // Every position named below indexes a RoPE table, and an index past
        // the end of one is silently clamped to its last row by candle's Metal
        // `index_select` kernel — wrong rotations, a plausible image and no
        // error, on the device this actually runs on. The caption occupies
        // axis 0 up to `cap_len` and the image sits above it at
        // `cap_len + 1 ..= cap_len + f_tokens`, so `cap_len + f_tokens` is the
        // highest index axis 0 ever sees.
        let axes_lens = self.cfg.axes_lens.as_slice();
        if axes_lens.len() != 3 {
            candle_core::bail!(
                "axes_lens has {} entries; the coordinate grid is 3-axis",
                axes_lens.len()
            );
        }
        let top = cap_len + f_tokens;
        if top >= axes_lens[0] {
            candle_core::bail!(
                "the caption needs position {top} on rope axis 0 ({cap_len} caption tokens \
                 then the image above them), and its table holds {}; a shorter prompt is \
                 the only fix, the table being the checkpoint's",
                axes_lens[0]
            );
        }
        if h_tokens > axes_lens[1] || w_tokens > axes_lens[2] {
            candle_core::bail!(
                "the image token grid is {h_tokens}x{w_tokens} (rows x columns) and the rope \
                 tables hold {}x{}; that size is past what this checkpoint can position",
                axes_lens[1],
                axes_lens[2]
            );
        }
        let x_pos_ids =
            create_coordinate_grid((f_tokens, h_tokens, w_tokens), (cap_len + 1, 0, 0), device)?;
        let (x_cos, x_sin) = self.rope_embedder.forward(&x_pos_ids)?;
        let cap_pos_ids = create_coordinate_grid((cap_len, 1, 1), (1, 0, 0), device)?;
        let (cap_cos, cap_sin) = self.rope_embedder.forward(&cap_pos_ids)?;

        // 5. Noise refiner (image, modulated)
        for layer in &self.noise_refiner {
            x = layer.forward(&x, None, &x_cos, &x_sin, Some(&adaln_input))?;
        }

        // 6. Context refiner (caption, unmodulated)
        for layer in &self.context_refiner {
            cap = layer.forward(&cap, None, &cap_cos, &cap_sin, None)?;
        }

        // 7. Joint sequence: [image, caption]
        let mut unified = Tensor::cat(&[&x, &cap], 1)?; // (B, img_seq + cap_len, dim)
        let unified_pos_ids = Tensor::cat(&[&x_pos_ids, &cap_pos_ids], 0)?;
        let (unified_cos, unified_sin) = self.rope_embedder.forward(&unified_pos_ids)?;

        // 8. Main transformer layers
        for layer in &self.layers {
            unified = layer.forward(
                &unified,
                None,
                &unified_cos,
                &unified_sin,
                Some(&adaln_input),
            )?;
        }

        // 9. Final layer on the image rows, then unpatchify
        let x_out = unified.narrow(1, 0, img_seq_len)?;
        let x_out = self.final_layer.forward(&x_out, &adaln_input)?;
        unpatchify(
            &x_out,
            orig_size,
            patch_size,
            f_patch_size,
            self.cfg.in_channels,
        )
    }

    /// The learned image pad row, for a caller building padded sequences.
    pub fn x_pad_token(&self) -> &Tensor {
        &self.x_pad_token
    }

    /// Get model configuration
    pub fn config(&self) -> &Config {
        &self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    /// The five values from working the reference's `polar(theta=256)` tables
    /// by hand for the position `(5, 3, 7)`: axis 0 fills complex pairs 0..16,
    /// axis 1 pairs 16..40, axis 2 pairs 40..64. Pins the axis layout, the
    /// frequency ladder and the interleaved application together.
    #[test]
    fn the_three_axis_rope_table_matches_the_reference_values() {
        let rope =
            RopeEmbedder::new(256.0, vec![32, 48, 48], vec![1536, 512, 512], &cpu()).unwrap();
        let ids = Tensor::new(&[[5u32, 3, 7]], &cpu()).unwrap();
        let (cos, sin) = rope.forward(&ids).unwrap();
        assert_eq!(cos.dims(), &[1, 64]);
        let cos = cos.to_vec2::<f32>().unwrap().remove(0);
        let sin = sin.to_vec2::<f32>().unwrap().remove(0);
        let close = |a: f32, b: f32| (a - b).abs() < 2e-5;
        // Axis 0, position 5: freqs 1, 2^-1/2, 2^-1, ...
        assert!(
            close(cos[0], 0.283662) && close(sin[0], -0.958924),
            "{} {}",
            cos[0],
            sin[0]
        );
        assert!(
            close(cos[1], -0.923403) && close(sin[1], -0.383831),
            "{} {}",
            cos[1],
            sin[1]
        );
        assert!(
            close(cos[2], -0.801144) && close(sin[2], 0.598472),
            "{} {}",
            cos[2],
            sin[2]
        );
        // Axis 1, position 3, first frequency 1.
        assert!(
            close(cos[16], -0.989992) && close(sin[16], 0.141120),
            "{} {}",
            cos[16],
            sin[16]
        );
        // Axis 2, position 7, first frequency 1.
        assert!(
            close(cos[40], 0.753902) && close(sin[40], 0.656987),
            "{} {}",
            cos[40],
            sin[40]
        );

        // Interleaved application: dims (0, 1) are one complex pair. A unit
        // real vector there rotates to (cos, sin); dims (2, 3) hold a zero pair
        // and stay zero, which the NEoX split-half form would not leave alone.
        let mut x = vec![0f32; 128];
        x[0] = 1.0;
        x[64] = 1.0; // dim 64 pairs with 65 (pair 32, axis 1 at frequency index 16)
        let x = Tensor::from_vec(x, (1, 1, 1, 128), &cpu()).unwrap();
        let cos_t = Tensor::from_vec(cos.clone(), (1, 64), &cpu()).unwrap();
        let sin_t = Tensor::from_vec(sin.clone(), (1, 64), &cpu()).unwrap();
        let y = apply_rotary_emb(&x, &cos_t, &sin_t).unwrap();
        let y = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(close(y[0], cos[0]) && close(y[1], sin[0]), "{:?}", &y[..4]);
        assert_eq!((y[2], y[3]), (0.0, 0.0));
        assert!(close(y[64], cos[32]) && close(y[65], sin[32]));
    }

    #[test]
    fn rope_keeps_the_input_dtype() {
        let rope = RopeEmbedder::new(256.0, vec![32, 48, 48], vec![16, 16, 16], &cpu()).unwrap();
        let ids = Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], &cpu()).unwrap();
        let (cos, sin) = rope.forward(&ids).unwrap();
        assert_eq!(cos.dtype(), DType::F32);
        let x = Tensor::randn(0f32, 1.0, (1, 2, 4, 128), &cpu())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let y = apply_rotary_emb(&x, &cos, &sin).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
        assert_eq!(y.dims(), x.dims());
    }

    /// `patchify` orders each 64-vector `(ph, pw, c)` with the channel fastest,
    /// as the reference's `view(...).permute(1,3,5,2,4,6,0)` does, and
    /// `unpatchify` is its exact inverse.
    #[test]
    fn patchify_orders_channel_fastest_and_unpatchify_inverts_it() {
        let (c, h, w) = (16usize, 4usize, 6usize);
        let data: Vec<f32> = (0..c * h * w)
            .map(|i| {
                let ci = i / (h * w);
                let hi = (i / w) % h;
                let wi = i % w;
                (ci * 1000 + hi * 10 + wi) as f32
            })
            .collect();
        let x = Tensor::from_vec(data, (1, c, 1, h, w), &cpu()).unwrap();
        let (patches, size) = patchify(&x, 2, 1).unwrap();
        assert_eq!(size, (1, h, w));
        assert_eq!(patches.dims(), &[1, (h / 2) * (w / 2), 4 * c]);
        let first = patches.i((0, 0)).unwrap().to_vec1::<f32>().unwrap();
        // (ph=0, pw=0, c=0..16), then (ph=0, pw=1, c=0..16), then ph=1 ...
        assert_eq!(first[0], 0.0);
        assert_eq!(first[1], 1000.0);
        assert_eq!(first[16], 1.0);
        assert_eq!(first[17], 1001.0);
        assert_eq!(first[32], 10.0);
        assert_eq!(first[48], 11.0);
        // The second patch is the next pair of columns.
        let second = patches.i((0, 1)).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(second[0], 2.0);
        let back = unpatchify(&patches, size, 2, 1, c).unwrap();
        assert_eq!(back.dims(), x.dims());
        let diff = (back - &x)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap();
        assert_eq!(diff.to_scalar::<f32>().unwrap(), 0.0);
    }

    #[test]
    fn padding_takes_a_length_up_to_the_next_multiple_of_32() {
        assert_eq!(compute_padding_len(0), 0);
        assert_eq!(compute_padding_len(32), 0);
        assert_eq!(compute_padding_len(4096), 0);
        assert_eq!(compute_padding_len(1), 31);
        assert_eq!(compute_padding_len(33), 31);
        assert_eq!(compute_padding_len(63), 1);
    }

    #[test]
    fn the_attention_env_switch_names_its_arms_and_refuses_a_typo() {
        assert_eq!(AttnImpl::parse("").unwrap(), AttnImpl::Fused);
        assert_eq!(AttnImpl::parse("fused").unwrap(), AttnImpl::Fused);
        assert_eq!(AttnImpl::parse(" Flash ").unwrap(), AttnImpl::Fused);
        assert_eq!(AttnImpl::parse("basic").unwrap(), AttnImpl::Basic);
        assert_eq!(AttnImpl::parse("BASIC").unwrap(), AttnImpl::Basic);
        assert!(AttnImpl::parse("reference").is_err());
        assert_eq!(AttnImpl::Fused.label(), "fused");
        assert_eq!(AttnImpl::Basic.label(), "basic");
        assert!(AttnImpl::Fused.is_accelerated());
        assert!(!AttnImpl::Basic.is_accelerated());

        // What a Config built from the shipped `config.json` gets, that file
        // carrying no key for it, and what an explicit override does.
        let mut cfg = Config::z_image_turbo();
        cfg.set_attn_impl(AttnImpl::Basic);
        assert_eq!(cfg.attn_impl(), AttnImpl::Basic);
        assert!(!cfg.use_accelerated_attn);
        cfg.set_attn_impl(AttnImpl::Fused);
        assert_eq!(cfg.attn_impl(), AttnImpl::Fused);
    }

    /// Deterministic pseudo-random weights for any name and shape, so a whole
    /// tiny transformer can be built with no checkpoint at all.
    ///
    /// Seeded from the tensor's NAME, so two builders hand the same tensor the
    /// same numbers: an A/B between two models built this way differs only in
    /// the code under test.
    struct RandomWeights;

    impl candle_nn::var_builder::SimpleBackend for RandomWeights {
        fn get(
            &self,
            s: candle_core::Shape,
            name: &str,
            _init: candle_nn::Init,
            dtype: DType,
            dev: &Device,
        ) -> Result<Tensor> {
            let mut state = name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
            }) | 1;
            let values: Vec<f32> = (0..s.elem_count())
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                    (u as f32 - 0.5) * 0.2
                })
                .collect();
            Tensor::from_vec(values, s, &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(dev)
        }

        fn get_unchecked(&self, name: &str, _dtype: DType, _dev: &Device) -> Result<Tensor> {
            candle_core::bail!("{name} was asked for without a shape to make up")
        }

        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    fn random_vb(dev: &Device) -> VarBuilder<'static> {
        VarBuilder::from_backend(Box::new(RandomWeights), DType::F32, dev.clone())
    }

    /// The real head dim (128) and the real axis split, at one head pair and
    /// one layer of each kind, with the rope tables cut down to 40 positions
    /// on axis 0 and 8 on each spatial axis so a test can reach past them.
    fn tiny_config() -> Config {
        Config {
            all_patch_size: vec![2],
            all_f_patch_size: vec![1],
            in_channels: 4,
            dim: 256,
            n_layers: 1,
            n_refiner_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            norm_eps: 1e-5,
            qk_norm: true,
            cap_feat_dim: 16,
            rope_theta: 256.0,
            t_scale: 1000.0,
            axes_dims: vec![32, 48, 48],
            axes_lens: vec![40, 8, 8],
            use_accelerated_attn: true,
        }
    }

    fn metal_or_skip(what: &str) -> Option<Device> {
        match crate::gguf::metal_device() {
            Ok(dev) => Some(dev),
            Err(e) => {
                eprintln!("skipping {what}: no Metal device ({e})");
                None
            }
        }
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        let b = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        let a = a.to_vec1::<f32>().unwrap();
        let b = b.to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        let mut worst = 0f32;
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            // Checked first: `f32::max` returns its non-NaN operand, so a fold
            // over a NaN-filled output would read as a perfect match.
            assert!(
                x.is_finite() && y.is_finite(),
                "non-finite value at {i}: {x} vs {y}"
            );
            worst = worst.max((x - y).abs());
        }
        worst
    }

    /// A position past the end of a rope table is refused, not clamped.
    ///
    /// Clamped is what it would otherwise be: candle's Metal `index_select`
    /// returns the table's last row for an out-of-range id rather than
    /// failing, so an over-long caption or an over-large image would come back
    /// as a plausible picture built from the wrong rotations. Runs on the CPU,
    /// where the same ids would have errored inside `index_select` — the point
    /// is that the refusal happens up front, with a sentence about the size,
    /// on every device.
    #[test]
    fn a_position_past_a_rope_table_is_refused_rather_than_clamped() {
        let dev = cpu();
        let cfg = tiny_config();
        let model = ZImageTransformer2DModel::new(&cfg, random_vb(&dev)).unwrap();
        let t = Tensor::new(&[0.5f32], &dev).unwrap();
        let legal_latent = Tensor::zeros((1, 4, 1, 16, 8), DType::F32, &dev).unwrap();
        let legal_cap = Tensor::zeros((1, 16, 16), DType::F32, &dev).unwrap();

        // The control. 16 caption tokens pad to 32, so the image sits at axis-0
        // position 33 of 40, and its 8x4 cell grid fits the 8x8 spatial tables.
        let out = model.forward(&legal_latent, &t, &legal_cap).unwrap();
        assert_eq!(out.dims(), &[1, 4, 1, 16, 8]);

        // 48 caption tokens pad to 64, putting the image at position 65 of a
        // 40-row table.
        let long_cap = Tensor::zeros((1, 48, 16), DType::F32, &dev).unwrap();
        let err = model
            .forward(&legal_latent, &t, &long_cap)
            .unwrap_err()
            .to_string();
        assert!(err.contains("position 65"), "{err}");
        assert!(err.contains("rope axis 0"), "{err}");

        // A 16x2 cell grid is still 32 image tokens, so this is the table
        // bound firing and not the multiple-of-32 rule.
        let tall_latent = Tensor::zeros((1, 4, 1, 32, 4), DType::F32, &dev).unwrap();
        let err = model
            .forward(&tall_latent, &t, &legal_cap)
            .unwrap_err()
            .to_string();
        assert!(err.contains("16x2"), "{err}");
        assert!(err.contains("8x8"), "{err}");
    }

    /// The basic attention arm agrees with the fused Metal kernel, within a
    /// bar and NOT to the bit.
    ///
    /// The nonzero half is the point. The two arms are different computations
    /// — an explicit matmul, additive mask, softmax and second matmul against
    /// candle's fused SDPA — and a bit-identical result would mean the switch
    /// selected one kernel twice and `XWEN_ZIMAGE_ATTN=basic` bisects nothing.
    /// That is exactly how the dense Qwen3 sdpa ablation was vacuous for an
    /// arc (AGENTS.md, "Verification workflow").
    ///
    /// It also runs the two mask branches, which `xwen image` never reaches:
    /// the fused arm's mask has to be the four-axis shape candle's kernel
    /// demands and the basic arm's the broadcastable one, and only running
    /// both says whether either is right.
    #[test]
    fn the_basic_attention_arm_matches_the_fused_kernel() {
        let Some(dev) = metal_or_skip("the_basic_attention_arm_matches_the_fused_kernel") else {
            return;
        };
        let mut fused_cfg = tiny_config();
        fused_cfg.set_attn_impl(AttnImpl::Fused);
        let mut basic_cfg = tiny_config();
        basic_cfg.set_attn_impl(AttnImpl::Basic);
        let fused = ZImageAttention::new(&fused_cfg, random_vb(&dev)).unwrap();
        let basic = ZImageAttention::new(&basic_cfg, random_vb(&dev)).unwrap();

        let seq = 64;
        let hidden = random_vb(&dev)
            .get((1, seq, fused_cfg.dim), "hidden_states")
            .unwrap();
        let ids = create_coordinate_grid((1, 8, 8), (1, 0, 0), &dev).unwrap();
        let rope = RopeEmbedder::new(
            fused_cfg.rope_theta,
            fused_cfg.axes_dims.clone(),
            fused_cfg.axes_lens.clone(),
            &dev,
        )
        .unwrap();
        let (cos, sin) = rope.forward(&ids).unwrap();

        let a = fused.forward(&hidden, None, &cos, &sin).unwrap();
        let b = basic.forward(&hidden, None, &cos, &sin).unwrap();
        assert_eq!(a.dims(), &[1, seq, fused_cfg.dim]);
        let diff = max_abs_diff(&a, &b);
        eprintln!("z-image attention, fused vs basic: max |delta| {diff:.3e}");
        assert!(diff <= 2e-3, "max abs diff {diff}");
        assert!(
            diff > 0.0,
            "the two arms produced bit-identical output, so they ran the same kernel"
        );

        // The mask branches, on a keep-mask that drops the second half of the
        // sequence. The masked arms agree with each other and differ from the
        // unmasked run, which is what says the mask was applied at all.
        let keep: Vec<f32> = (0..seq)
            .map(|i| if i < seq / 2 { 1.0 } else { 0.0 })
            .collect();
        let mask = Tensor::from_vec(keep, (1, seq), &dev).unwrap();
        let a_masked = fused.forward(&hidden, Some(&mask), &cos, &sin).unwrap();
        let b_masked = basic.forward(&hidden, Some(&mask), &cos, &sin).unwrap();
        let diff = max_abs_diff(&a_masked, &b_masked);
        eprintln!("z-image attention with a mask, fused vs basic: max |delta| {diff:.3e}");
        assert!(diff <= 2e-3, "masked: max abs diff {diff}");
        assert!(
            max_abs_diff(&a, &a_masked) > 0.0,
            "the mask changed nothing, so it was not applied"
        );
    }
}
