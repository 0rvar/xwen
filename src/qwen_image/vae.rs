//! Qwen-Image 2.1 VAE (`AutoencoderKLQwenImage21`), written from diffusers
//! `autoencoder_kl_qwenimage21.py` at 6256aa7.
//!
//! The reference is Wan 2.2's residual video VAE specialised to ONE frame: its
//! "causal 3-D" convolution subclasses `nn.Conv2d`, squeezes the frame axis and
//! raises when handed a feature cache, so the weights on disk are 2-D
//! convolutions (`[out, in, k, k]`, F32) and this module is a plain NCHW
//! network with no frame axis at all. Three things survive from the video
//! model and each is a silent trap:
//!
//! 1. **`RMS_norm` is not an RMS norm.** It is `F.normalize(x, dim=1) *
//!    sqrt(C) * gamma`: an L2 norm over the CHANNEL axis per pixel, in the clamp
//!    form `x / max(‖x‖₂, 1e-12)`, with a per-channel `gamma` and no bias.
//! 2. **The residual shortcuts are the video ones evaluated at T = 1**
//!    ([`DownShortcut`], [`UpShortcut`]). `AvgDown3D` pads a ZERO frame in front
//!    at a temporal stage, so half of that stage's shortcut channels are
//!    identically zero; `DupUp3D` keeps only the LAST temporal copy of the first
//!    chunk, so a channel-halving stage reads the odd input channels and the
//!    spatial-only stage interleaves a channel pair into even and odd rows.
//!    The closed forms here are derived from the reference's
//!    `view`/`permute`/`view` as written and each is tested against a literal
//!    5-D transcription of it.
//! 3. **`time_conv` never runs for one frame.** The reference's first chunk
//!    marks an `upsample3d` cache slot `"Rep"` and stores a `downsample3d` one
//!    without calling `time_conv` either time, and an image is only ever a
//!    first chunk. Those tensors are in the file and are dead
//!    ([`dead_tensors`] names them one by one).
//!
//! The latent normalisation lives INSIDE this API on both sides: [`decode`]
//! takes the transformer's normalised latent and applies `z * std + mean` per
//! channel itself, and [`encode`] returns `(mean(posterior) - mean) / std`, the
//! tensor the transformer reads as a clean condition latent. The 64 constants
//! per side come from `vae/config.json`.
//!
//! Everything runs in f32, which is also what ships on disk.
//!
//! [`decode`]: QwenImageVae::decode
//! [`encode`]: QwenImageVae::encode

use candle_core::{D, DType, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder, conv2d};

/// The environment switch that picks the VAE's convolution kernels, read when
/// the VAE loads. `candle` (or unset) is candle's conv2d chain, the only arm
/// there is; `xwen` / `direct` name the direct-convolution arm and are refused
/// until it exists, rather than silently running candle under that label.
pub const VAE_ENV: &str = "XWEN_QWEN_IMAGE_VAE";

/// Which kernels the VAE runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VaeImpl {
    /// candle's conv2d for every convolution.
    Candle,
}

impl VaeImpl {
    /// Resolve from [`VAE_ENV`]: unset means `candle`, anything else must name
    /// an arm.
    pub fn from_env() -> Result<Self> {
        match std::env::var(VAE_ENV) {
            Err(std::env::VarError::NotPresent) => Ok(Self::Candle),
            Err(std::env::VarError::NotUnicode(_)) => {
                candle_core::bail!("{VAE_ENV} is not valid UTF-8")
            }
            Ok(value) => Self::parse(&value),
        }
    }

    /// `candle` (or empty) selects candle's chain. `xwen` and `direct` are
    /// refused by name; anything else is refused as unknown.
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "candle" => Ok(Self::Candle),
            "xwen" | "direct" => candle_core::bail!(
                "{VAE_ENV}={value:?}: the Qwen-Image VAE has no direct-convolution arm, \
                 `candle` is the only one"
            ),
            other => candle_core::bail!("{VAE_ENV}={other:?}: expected `candle`"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Candle => "candle",
        }
    }
}

// ==================== Config ====================

/// `vae/config.json`. Every key the graph depends on is read and the ones this
/// module does not implement are refused in [`VaeConfig::validate`], so a
/// config that would need the other branch of the reference fails at load.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct VaeConfig {
    pub base_dim: usize,
    /// The decoder's base width; the reference falls back to `base_dim`.
    #[serde(default)]
    pub decoder_base_dim: Option<usize>,
    pub z_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    #[serde(default)]
    pub attn_scales: Vec<f64>,
    pub temperal_downsample: Vec<bool>,
    pub is_residual: bool,
    pub in_channels: usize,
    pub out_channels: usize,
    #[serde(default)]
    pub patch_size: Option<usize>,
    pub scale_factor_spatial: usize,
    pub latents_mean: Vec<f64>,
    pub latents_std: Vec<f64>,
}

impl VaeConfig {
    /// Refuse what this module does not implement: the non-residual block
    /// layout, attention inside the down and up paths, and the patchified
    /// input. The shipped config uses none of them.
    pub fn validate(&self) -> Result<()> {
        if !self.is_residual {
            candle_core::bail!("qwen-image vae: is_residual=false is not implemented");
        }
        if !self.attn_scales.is_empty() {
            candle_core::bail!("qwen-image vae: attn_scales is not implemented");
        }
        if self.patch_size.is_some() {
            candle_core::bail!("qwen-image vae: patch_size is not implemented");
        }
        if self.dim_mult.is_empty() || self.num_res_blocks == 0 || self.z_dim == 0 {
            candle_core::bail!("qwen-image vae: empty dim_mult, num_res_blocks or z_dim");
        }
        if self.temperal_downsample.len() + 1 != self.dim_mult.len() {
            candle_core::bail!(
                "qwen-image vae: {} temperal_downsample flags for {} stages, expected one per \
                 resampling stage",
                self.temperal_downsample.len(),
                self.dim_mult.len()
            );
        }
        if self.latents_mean.len() != self.z_dim || self.latents_std.len() != self.z_dim {
            candle_core::bail!(
                "qwen-image vae: latents_mean/latents_std hold {}/{} values for z_dim {}",
                self.latents_mean.len(),
                self.latents_std.len(),
                self.z_dim
            );
        }
        if self
            .latents_std
            .iter()
            .any(|s| !(s.is_finite() && *s > 0.0))
        {
            candle_core::bail!("qwen-image vae: latents_std must be finite and positive");
        }
        let stride = 1usize << (self.dim_mult.len() - 1);
        if self.scale_factor_spatial != stride {
            candle_core::bail!(
                "qwen-image vae: scale_factor_spatial {} against {} resampling stages ({stride}x)",
                self.scale_factor_spatial,
                self.dim_mult.len() - 1
            );
        }
        Ok(())
    }

    /// Pixels per latent position on each side.
    pub fn spatial_factor(&self) -> usize {
        self.scale_factor_spatial
    }

    /// The encoder's stage widths, `base_dim * [1, dim_mult...]`.
    fn encoder_dims(&self) -> Vec<usize> {
        std::iter::once(1)
            .chain(self.dim_mult.iter().copied())
            .map(|m| self.base_dim * m)
            .collect()
    }

    /// The decoder's stage widths, `decoder_base_dim * [last, reversed dim_mult...]`.
    fn decoder_dims(&self) -> Vec<usize> {
        let base = self.decoder_base_dim.unwrap_or(self.base_dim);
        let last = *self.dim_mult.last().expect("validated non-empty");
        std::iter::once(last)
            .chain(self.dim_mult.iter().rev().copied())
            .map(|m| base * m)
            .collect()
    }

    /// Whether encoder stage `i` resamples, and whether that stage is a
    /// temporal one in the video model. The last stage does neither.
    fn down_stage(&self, i: usize) -> (bool, bool) {
        let down = i != self.dim_mult.len() - 1;
        (down, down && self.temperal_downsample[i])
    }

    /// The decoder's mirror of [`Self::down_stage`]: the temporal flags are
    /// read reversed.
    fn up_stage(&self, i: usize) -> (bool, bool) {
        let up = i != self.dim_mult.len() - 1;
        let n = self.temperal_downsample.len();
        (up, up && self.temperal_downsample[n - 1 - i])
    }
}

// ==================== Tensor table ====================

fn conv_entries(
    out: &mut Vec<(String, Vec<usize>)>,
    name: &str,
    c_in: usize,
    c_out: usize,
    k: usize,
) {
    out.push((format!("{name}.weight"), vec![c_out, c_in, k, k]));
    out.push((format!("{name}.bias"), vec![c_out]));
}

fn resnet_entries(out: &mut Vec<(String, Vec<usize>)>, name: &str, c_in: usize, c_out: usize) {
    out.push((format!("{name}.norm1.gamma"), vec![c_in, 1, 1, 1]));
    conv_entries(out, &format!("{name}.conv1"), c_in, c_out, 3);
    out.push((format!("{name}.norm2.gamma"), vec![c_out, 1, 1, 1]));
    conv_entries(out, &format!("{name}.conv2"), c_out, c_out, 3);
    if c_in != c_out {
        conv_entries(out, &format!("{name}.conv_shortcut"), c_in, c_out, 1);
    }
}

fn mid_entries(out: &mut Vec<(String, Vec<usize>)>, name: &str, dim: usize) {
    resnet_entries(out, &format!("{name}.resnets.0"), dim, dim);
    out.push((format!("{name}.attentions.0.norm.gamma"), vec![dim, 1, 1]));
    conv_entries(out, &format!("{name}.attentions.0.to_qkv"), dim, dim * 3, 1);
    conv_entries(out, &format!("{name}.attentions.0.proj"), dim, dim, 1);
    resnet_entries(out, &format!("{name}.resnets.1"), dim, dim);
}

/// Every tensor the loader reads, by checkpoint name with its on-disk shape.
/// Together with [`dead_tensors`] this is the whole file, which the tests
/// assert in both directions.
pub fn live_tensors(cfg: &VaeConfig) -> Vec<(String, Vec<usize>)> {
    let mut out = Vec::new();
    let enc = cfg.encoder_dims();
    conv_entries(&mut out, "encoder.conv_in", cfg.in_channels, enc[0], 3);
    for i in 0..cfg.dim_mult.len() {
        let (c_in, c_out) = (enc[i], enc[i + 1]);
        for j in 0..cfg.num_res_blocks {
            let from = if j == 0 { c_in } else { c_out };
            resnet_entries(
                &mut out,
                &format!("encoder.down_blocks.{i}.resnets.{j}"),
                from,
                c_out,
            );
        }
        if cfg.down_stage(i).0 {
            let name = format!("encoder.down_blocks.{i}.downsampler.resample.1");
            conv_entries(&mut out, &name, c_out, c_out, 3);
        }
    }
    let top = *enc.last().expect("non-empty");
    mid_entries(&mut out, "encoder.mid_block", top);
    out.push(("encoder.norm_out.gamma".into(), vec![top, 1, 1, 1]));
    conv_entries(&mut out, "encoder.conv_out", top, cfg.z_dim * 2, 3);
    conv_entries(&mut out, "quant_conv", cfg.z_dim * 2, cfg.z_dim * 2, 1);
    conv_entries(&mut out, "post_quant_conv", cfg.z_dim, cfg.z_dim, 1);

    let dec = cfg.decoder_dims();
    conv_entries(&mut out, "decoder.conv_in", cfg.z_dim, dec[0], 3);
    mid_entries(&mut out, "decoder.mid_block", dec[0]);
    for i in 0..cfg.dim_mult.len() {
        let (c_in, c_out) = (dec[i], dec[i + 1]);
        for j in 0..=cfg.num_res_blocks {
            let from = if j == 0 { c_in } else { c_out };
            resnet_entries(
                &mut out,
                &format!("decoder.up_blocks.{i}.resnets.{j}"),
                from,
                c_out,
            );
        }
        if cfg.up_stage(i).0 {
            let name = format!("decoder.up_blocks.{i}.upsampler.resample.1");
            conv_entries(&mut out, &name, c_out, c_out, 3);
        }
    }
    let bottom = *dec.last().expect("non-empty");
    out.push(("decoder.norm_out.gamma".into(), vec![bottom, 1, 1, 1]));
    conv_entries(&mut out, "decoder.conv_out", bottom, cfg.out_channels, 3);
    out
}

/// The tensors in the file that a single frame never evaluates: the
/// `time_conv` of every temporal resampling stage, the video model's mixing
/// across frames. Listed one by one so that any OTHER unread tensor is a
/// loader bug and not something a pattern swallowed.
pub fn dead_tensors(cfg: &VaeConfig) -> Vec<(String, Vec<usize>)> {
    let mut out = Vec::new();
    let enc = cfg.encoder_dims();
    for i in 0..cfg.dim_mult.len() {
        if cfg.down_stage(i).1 {
            let name = format!("encoder.down_blocks.{i}.downsampler.time_conv");
            conv_entries(&mut out, &name, enc[i + 1], enc[i + 1], 1);
        }
    }
    let dec = cfg.decoder_dims();
    for i in 0..cfg.dim_mult.len() {
        if cfg.up_stage(i).1 {
            let name = format!("decoder.up_blocks.{i}.upsampler.time_conv");
            conv_entries(&mut out, &name, dec[i + 1], dec[i + 1] * 2, 1);
        }
    }
    out
}

// ==================== Layers ====================

/// A square stride-1 convolution with `padding = kernel / 2`. The arm is a
/// constructor argument so a second one lands here and nowhere else; note that
/// the encoder's `conv_in` (4 input channels) and its strided downsample
/// convolutions are outside what `ops::conv2d_direct` accepts (`c_in % 8 == 0`,
/// stride 1) and stay on candle under any arm.
#[derive(Debug, Clone)]
struct Conv {
    candle: Conv2d,
}

impl Conv {
    fn new(c_in: usize, c_out: usize, kernel: usize, arm: VaeImpl, vb: VarBuilder) -> Result<Self> {
        let VaeImpl::Candle = arm;
        let cfg = Conv2dConfig {
            padding: kernel / 2,
            ..Default::default()
        };
        Ok(Self {
            candle: conv2d(c_in, c_out, kernel, cfg, vb)?,
        })
    }
}

impl Module for Conv {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.candle.forward(xs)
    }
}

/// The reference's `RMS_norm`: an L2 normalisation over the channel axis per
/// pixel, `x / max(‖x‖₂, 1e-12) * sqrt(C) * gamma`. `gamma` is stored as
/// `[C, 1, 1, 1]` in the residual blocks and the output norms and as
/// `[C, 1, 1]` in the attention block, and is held here as `[1, C, 1, 1]`
/// with `sqrt(C)` folded in. There is no bias.
#[derive(Debug, Clone)]
struct ChannelL2Norm {
    gamma: Tensor,
}

impl ChannelL2Norm {
    const EPS: f64 = 1e-12;

    fn new(dim: usize, stored: &[usize], vb: VarBuilder) -> Result<Self> {
        let gamma = vb.get(stored, "gamma")?.reshape((1, dim, 1, 1))?;
        Ok(Self {
            gamma: (gamma * (dim as f64).sqrt())?,
        })
    }

    fn residual(dim: usize, vb: VarBuilder) -> Result<Self> {
        Self::new(dim, &[dim, 1, 1, 1], vb)
    }

    fn attention(dim: usize, vb: VarBuilder) -> Result<Self> {
        Self::new(dim, &[dim, 1, 1], vb)
    }
}

impl Module for ChannelL2Norm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let norm = xs.sqr()?.sum_keepdim(1)?.sqrt()?.maximum(Self::EPS)?;
        xs.broadcast_div(&norm)?.broadcast_mul(&self.gamma)
    }
}

/// `norm1, silu, conv1, norm2, silu, conv2`, added to the input (through a 1x1
/// `conv_shortcut` when the width changes).
#[derive(Debug, Clone)]
struct ResidualBlock {
    norm1: ChannelL2Norm,
    conv1: Conv,
    norm2: ChannelL2Norm,
    conv2: Conv,
    conv_shortcut: Option<Conv>,
}

impl ResidualBlock {
    fn new(c_in: usize, c_out: usize, arm: VaeImpl, vb: VarBuilder) -> Result<Self> {
        let conv_shortcut = if c_in != c_out {
            Some(Conv::new(c_in, c_out, 1, arm, vb.pp("conv_shortcut"))?)
        } else {
            None
        };
        Ok(Self {
            norm1: ChannelL2Norm::residual(c_in, vb.pp("norm1"))?,
            conv1: Conv::new(c_in, c_out, 3, arm, vb.pp("conv1"))?,
            norm2: ChannelL2Norm::residual(c_out, vb.pp("norm2"))?,
            conv2: Conv::new(c_out, c_out, 3, arm, vb.pp("conv2"))?,
            conv_shortcut,
        })
    }
}

impl Module for ResidualBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = match &self.conv_shortcut {
            Some(conv) => conv.forward(xs)?,
            None => xs.clone(),
        };
        let x = self.conv1.forward(&self.norm1.forward(xs)?.silu()?)?;
        let x = self.conv2.forward(&self.norm2.forward(&x)?.silu()?)?;
        x + h
    }
}

/// Single-head self-attention over all `H * W` positions, scale `1/sqrt(C)`,
/// with the fused 1x1 `to_qkv` split as `[q | k | v]` along channels.
#[derive(Debug, Clone)]
struct AttentionBlock {
    norm: ChannelL2Norm,
    to_qkv: Conv,
    proj: Conv,
    dim: usize,
}

impl AttentionBlock {
    fn new(dim: usize, arm: VaeImpl, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: ChannelL2Norm::attention(dim, vb.pp("norm"))?,
            to_qkv: Conv::new(dim, dim * 3, 1, arm, vb.pp("to_qkv"))?,
            proj: Conv::new(dim, dim, 1, arm, vb.pp("proj"))?,
            dim,
        })
    }
}

impl Module for AttentionBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = xs.dims4()?;
        let qkv = self
            .to_qkv
            .forward(&self.norm.forward(xs)?)?
            .reshape((b, c * 3, h * w))?
            .transpose(1, 2)?;
        let q = qkv.narrow(D::Minus1, 0, c)?.contiguous()?;
        let k = qkv.narrow(D::Minus1, c, c)?.contiguous()?;
        let v = qkv.narrow(D::Minus1, 2 * c, c)?.contiguous()?;
        let scores = (q.matmul(&k.transpose(1, 2)?)? * (1.0 / (self.dim as f64).sqrt()))?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        let out = attn.transpose(1, 2)?.reshape((b, c, h, w))?;
        self.proj.forward(&out)? + xs
    }
}

#[derive(Debug, Clone)]
struct MidBlock {
    resnet0: ResidualBlock,
    attention: AttentionBlock,
    resnet1: ResidualBlock,
}

impl MidBlock {
    fn new(dim: usize, arm: VaeImpl, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            resnet0: ResidualBlock::new(dim, dim, arm, vb.pp("resnets.0"))?,
            attention: AttentionBlock::new(dim, arm, vb.pp("attentions.0"))?,
            resnet1: ResidualBlock::new(dim, dim, arm, vb.pp("resnets.1"))?,
        })
    }
}

impl Module for MidBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.resnet0.forward(xs)?;
        self.resnet1.forward(&self.attention.forward(&x)?)
    }
}

// ==================== Shortcuts ====================

/// The encoder stage's parameter-free shortcut: the reference's `AvgDown3D`
/// for one frame. With `factor_t` frames per output frame, `factor_s` pixels
/// per side and `g = c_in * factor_t * factor_s² / c_out`, the reference
/// space-to-depths each input channel `c` into `factor_t * factor_s²` expanded
/// channels ordered `(t, i, j)` and averages consecutive runs of `g` of them.
/// For one frame and `factor_t = 2` the frame at `t = 0` is the zero frame the
/// reference pads IN FRONT, so that half of the expanded channels is zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DownShortcut {
    /// No resampling, same width (`g = 1`): the input itself.
    Identity,
    /// Spatial only, same width (`factor_t = 1`, `g = 4`): `out[c]` is the 2x2
    /// average of `in[c]`.
    Pool,
    /// Temporal stage doubling the width (`factor_t = 2`, `g = 4`): a run of
    /// four covers exactly one `(c, t)`, so `out[2c]` is the zero frame's
    /// average, identically zero, and `out[2c + 1]` is the 2x2 average of
    /// `in[c]`.
    ZeroInterleavedPool,
}

impl DownShortcut {
    fn new(c_in: usize, c_out: usize, down: bool, temporal: bool) -> Result<Self> {
        match (down, temporal) {
            (false, false) if c_in == c_out => Ok(Self::Identity),
            (true, false) if c_in == c_out => Ok(Self::Pool),
            (true, true) if c_out == 2 * c_in => Ok(Self::ZeroInterleavedPool),
            _ => candle_core::bail!(
                "qwen-image vae: no single-frame form of the {c_in} -> {c_out} down shortcut \
                 (down {down}, temporal {temporal})"
            ),
        }
    }

    fn forward(self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Identity => Ok(xs.clone()),
            Self::Pool => xs.avg_pool2d(2),
            Self::ZeroInterleavedPool => {
                let pooled = xs.avg_pool2d(2)?;
                let (b, c, h, w) = pooled.dims4()?;
                Tensor::stack(&[&pooled.zeros_like()?, &pooled], 2)?.reshape((b, 2 * c, h, w))
            }
        }
    }
}

/// The decoder stage's parameter-free shortcut: the reference's `DupUp3D` for
/// the first (and only) chunk. It repeats every input channel
/// `r = c_out * factor_t * 4 / c_in` times, reads the expanded channels of
/// output channel `o` as `(t, i, j)`, writes them to frame `t` and pixel
/// `(2y + i, 2x + j)`, and for the first chunk keeps only the LAST frame,
/// `t = factor_t - 1`. So the expanded channel behind output `o` at `(i, j)` is
/// `o * 4 * factor_t + (factor_t - 1) * 4 + 2i + j`, and the input channel is
/// that divided by `r`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UpShortcut {
    /// `r = 4 * factor_t`: all four positions read `in[o]`, a nearest 2x.
    Nearest,
    /// Temporal stage halving the width (`factor_t = 2`, `r = 4`): the kept
    /// frame reads `in[2o + 1]` at all four positions, a nearest 2x of the ODD
    /// input channels; the even ones belong to the dropped frame.
    OddChannelNearest,
    /// Spatial-only stage halving the width (`factor_t = 1`, `r = 2`): position
    /// `(i, j)` reads `in[2o + i]`, so even output rows come from `in[2o]` and
    /// odd ones from `in[2o + 1]`, each pixel doubled along the row.
    RowInterleave,
}

impl UpShortcut {
    fn new(c_in: usize, c_out: usize, temporal: bool) -> Result<Self> {
        match temporal {
            _ if c_in == c_out => Ok(Self::Nearest),
            true if c_in == 2 * c_out => Ok(Self::OddChannelNearest),
            false if c_in == 2 * c_out => Ok(Self::RowInterleave),
            _ => candle_core::bail!(
                "qwen-image vae: no single-frame form of the {c_in} -> {c_out} up shortcut \
                 (temporal {temporal})"
            ),
        }
    }

    fn forward(self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = xs.dims4()?;
        match self {
            Self::Nearest => xs.upsample_nearest2d(2 * h, 2 * w),
            Self::OddChannelNearest => xs
                .reshape((b, c / 2, 2, h, w))?
                .narrow(2, 1, 1)?
                .reshape((b, c / 2, h, w))?
                .upsample_nearest2d(2 * h, 2 * w),
            Self::RowInterleave => xs
                .reshape((b, c / 2, 2, h, w))?
                .permute((0, 1, 3, 2, 4))?
                .unsqueeze(5)?
                .broadcast_as((b, c / 2, h, 2, w, 2))?
                .contiguous()?
                .reshape((b, c / 2, 2 * h, 2 * w)),
        }
    }
}

// ==================== Encoder ====================

/// `ZeroPad2d((0, 1, 0, 1))` then a 3x3 stride-2 convolution with no padding of
/// its own: the pad is on the right and the bottom only.
#[derive(Debug, Clone)]
struct Downsample {
    conv: Conv2d,
}

impl Downsample {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            stride: 2,
            ..Default::default()
        };
        Ok(Self {
            conv: conv2d(dim, dim, 3, cfg, vb.pp("resample.1"))?,
        })
    }
}

impl Module for Downsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let padded = xs.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?;
        self.conv.forward(&padded)
    }
}

#[derive(Debug, Clone)]
struct DownBlock {
    resnets: Vec<ResidualBlock>,
    downsampler: Option<Downsample>,
    shortcut: DownShortcut,
}

impl DownBlock {
    fn new(
        c_in: usize,
        c_out: usize,
        n_res: usize,
        (down, temporal): (bool, bool),
        arm: VaeImpl,
        vb: VarBuilder,
    ) -> Result<Self> {
        let resnets = (0..n_res)
            .map(|j| {
                let from = if j == 0 { c_in } else { c_out };
                ResidualBlock::new(from, c_out, arm, vb.pp("resnets").pp(j))
            })
            .collect::<Result<Vec<_>>>()?;
        let downsampler = down
            .then(|| Downsample::new(c_out, vb.pp("downsampler")))
            .transpose()?;
        Ok(Self {
            resnets,
            downsampler,
            shortcut: DownShortcut::new(c_in, c_out, down, temporal)?,
        })
    }
}

impl Module for DownBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();
        for resnet in &self.resnets {
            x = resnet.forward(&x)?;
        }
        if let Some(down) = &self.downsampler {
            x = down.forward(&x)?;
        }
        x + self.shortcut.forward(xs)?
    }
}

#[derive(Debug, Clone)]
struct Encoder {
    conv_in: Conv,
    down_blocks: Vec<DownBlock>,
    mid_block: MidBlock,
    norm_out: ChannelL2Norm,
    conv_out: Conv,
}

impl Encoder {
    fn new(cfg: &VaeConfig, arm: VaeImpl, vb: VarBuilder) -> Result<Self> {
        let dims = cfg.encoder_dims();
        let down_blocks = (0..cfg.dim_mult.len())
            .map(|i| {
                DownBlock::new(
                    dims[i],
                    dims[i + 1],
                    cfg.num_res_blocks,
                    cfg.down_stage(i),
                    arm,
                    vb.pp("down_blocks").pp(i),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let top = *dims.last().expect("non-empty");
        Ok(Self {
            conv_in: Conv::new(cfg.in_channels, dims[0], 3, arm, vb.pp("conv_in"))?,
            down_blocks,
            mid_block: MidBlock::new(top, arm, vb.pp("mid_block"))?,
            norm_out: ChannelL2Norm::residual(top, vb.pp("norm_out"))?,
            conv_out: Conv::new(top, cfg.z_dim * 2, 3, arm, vb.pp("conv_out"))?,
        })
    }
}

impl Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(xs)?;
        for block in &self.down_blocks {
            x = block.forward(&x)?;
        }
        let x = self.mid_block.forward(&x)?;
        self.conv_out.forward(&self.norm_out.forward(&x)?.silu()?)
    }
}

// ==================== Decoder ====================

#[derive(Debug, Clone)]
struct UpBlock {
    resnets: Vec<ResidualBlock>,
    /// The nearest 2x upsample's convolution, with the shortcut added after
    /// it. `None` on the last stage, which neither upsamples nor has a
    /// shortcut.
    upsampler: Option<(Conv, UpShortcut)>,
}

impl UpBlock {
    fn new(
        c_in: usize,
        c_out: usize,
        n_res: usize,
        (up, temporal): (bool, bool),
        arm: VaeImpl,
        vb: VarBuilder,
    ) -> Result<Self> {
        let resnets = (0..=n_res)
            .map(|j| {
                let from = if j == 0 { c_in } else { c_out };
                ResidualBlock::new(from, c_out, arm, vb.pp("resnets").pp(j))
            })
            .collect::<Result<Vec<_>>>()?;
        let upsampler = if up {
            let conv = Conv::new(c_out, c_out, 3, arm, vb.pp("upsampler.resample.1"))?;
            Some((conv, UpShortcut::new(c_in, c_out, temporal)?))
        } else {
            None
        };
        Ok(Self { resnets, upsampler })
    }
}

impl Module for UpBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();
        for resnet in &self.resnets {
            x = resnet.forward(&x)?;
        }
        match &self.upsampler {
            Some((conv, shortcut)) => {
                let (_, _, h, w) = x.dims4()?;
                // `nearest-exact` and `nearest` pick the same source pixel at
                // an integer factor of 2.
                let x = conv.forward(&x.upsample_nearest2d(2 * h, 2 * w)?)?;
                x + shortcut.forward(xs)?
            }
            None => Ok(x),
        }
    }
}

#[derive(Debug, Clone)]
struct Decoder {
    conv_in: Conv,
    mid_block: MidBlock,
    up_blocks: Vec<UpBlock>,
    norm_out: ChannelL2Norm,
    conv_out: Conv,
}

impl Decoder {
    fn new(cfg: &VaeConfig, arm: VaeImpl, vb: VarBuilder) -> Result<Self> {
        let dims = cfg.decoder_dims();
        let up_blocks = (0..cfg.dim_mult.len())
            .map(|i| {
                UpBlock::new(
                    dims[i],
                    dims[i + 1],
                    cfg.num_res_blocks,
                    cfg.up_stage(i),
                    arm,
                    vb.pp("up_blocks").pp(i),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let bottom = *dims.last().expect("non-empty");
        Ok(Self {
            conv_in: Conv::new(cfg.z_dim, dims[0], 3, arm, vb.pp("conv_in"))?,
            mid_block: MidBlock::new(dims[0], arm, vb.pp("mid_block"))?,
            up_blocks,
            norm_out: ChannelL2Norm::residual(bottom, vb.pp("norm_out"))?,
            conv_out: Conv::new(bottom, cfg.out_channels, 3, arm, vb.pp("conv_out"))?,
        })
    }
}

impl Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = self.mid_block.forward(&self.conv_in.forward(xs)?)?;
        for block in &self.up_blocks {
            x = block.forward(&x)?;
        }
        self.conv_out.forward(&self.norm_out.forward(&x)?.silu()?)
    }
}

// ==================== The autoencoder ====================

/// The Qwen-Image 2.1 autoencoder for one image: RGBA in `[-1, 1]` on one side,
/// the transformer's normalised 64-channel latent at 1/16 scale on the other.
#[derive(Debug, Clone)]
pub struct QwenImageVae {
    encoder: Encoder,
    quant_conv: Conv,
    post_quant_conv: Conv,
    decoder: Decoder,
    /// `latents_mean` and `latents_std` as `[1, z_dim, 1, 1]`.
    latents_mean: Tensor,
    latents_std: Tensor,
    cfg: VaeConfig,
    arm: VaeImpl,
}

impl QwenImageVae {
    /// Load with the arm [`VAE_ENV`] names. `vb` is rooted at the VAE file
    /// (names start `encoder.` / `decoder.`) and must be F32, the dtype the
    /// file ships and the one every op here runs in.
    pub fn load(vb: VarBuilder, cfg: &VaeConfig) -> Result<Self> {
        Self::load_with_impl(vb, cfg, VaeImpl::from_env()?)
    }

    pub fn load_with_impl(vb: VarBuilder, cfg: &VaeConfig, arm: VaeImpl) -> Result<Self> {
        cfg.validate()?;
        if vb.dtype() != DType::F32 {
            candle_core::bail!(
                "qwen-image vae: runs in f32, got a {:?} VarBuilder",
                vb.dtype()
            );
        }
        let z = cfg.z_dim;
        let constants = |values: &[f64]| -> Result<Tensor> {
            let values: Vec<f32> = values.iter().map(|v| *v as f32).collect();
            Tensor::from_vec(values, (1, z, 1, 1), vb.device())
        };
        Ok(Self {
            encoder: Encoder::new(cfg, arm, vb.pp("encoder"))?,
            quant_conv: Conv::new(z * 2, z * 2, 1, arm, vb.pp("quant_conv"))?,
            post_quant_conv: Conv::new(z, z, 1, arm, vb.pp("post_quant_conv"))?,
            decoder: Decoder::new(cfg, arm, vb.pp("decoder"))?,
            latents_mean: constants(&cfg.latents_mean)?,
            latents_std: constants(&cfg.latents_std)?,
            cfg: cfg.clone(),
            arm,
        })
    }

    pub fn config(&self) -> &VaeConfig {
        &self.cfg
    }

    pub fn arm(&self) -> VaeImpl {
        self.arm
    }

    /// Decode the transformer's NORMALISED latent `[B, z_dim, h, w]` (f32) to
    /// `[B, out_channels, 16h, 16w]` in `[-1, 1]`, clamped; the four channels
    /// are R, G, B, A. The per-channel `z * std + mean` is applied here.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let (_, c, _, _) = z.dims4()?;
        if c != self.cfg.z_dim || z.dtype() != DType::F32 {
            candle_core::bail!(
                "qwen-image vae: decode takes an f32 [B, {}, h, w] latent, got {:?} {:?}",
                self.cfg.z_dim,
                z.dtype(),
                z.shape()
            );
        }
        let z = z
            .broadcast_mul(&self.latents_std)?
            .broadcast_add(&self.latents_mean)?;
        self.decode_trunk(&z)?.clamp(-1f32, 1f32)
    }

    /// `post_quant_conv` and the decoder over a latent already in the VAE's
    /// own scale, before the clamp.
    fn decode_trunk(&self, z: &Tensor) -> Result<Tensor> {
        self.decoder.forward(&self.post_quant_conv.forward(z)?)
    }

    /// The encoder and `quant_conv`: the posterior's `2 * z_dim` moments, the
    /// mean in the first half and the log-variance in the second.
    fn moments(&self, image: &Tensor) -> Result<Tensor> {
        self.quant_conv.forward(&self.encoder.forward(image)?)
    }

    /// Encode `[B, in_channels, H, W]` (f32, RGBA in `[-1, 1]`, both sides
    /// multiples of 16) to the NORMALISED posterior mean `[B, z_dim, H/16,
    /// W/16]`: the first `z_dim` of the encoder's `2 * z_dim` moments, then
    /// `(z - mean) / std`. The log-variance half is never used; a condition
    /// image is its posterior's mode.
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        let (_, c, h, w) = image.dims4()?;
        let f = self.cfg.spatial_factor();
        if c != self.cfg.in_channels || image.dtype() != DType::F32 || h % f != 0 || w % f != 0 {
            candle_core::bail!(
                "qwen-image vae: encode takes an f32 [B, {}, H, W] image with H and W multiples \
                 of {f}, got {:?} {:?}",
                self.cfg.in_channels,
                image.dtype(),
                image.shape()
            );
        }
        self.moments(image)?
            .narrow(1, 0, self.cfg.z_dim)?
            .broadcast_sub(&self.latents_mean)?
            .broadcast_div(&self.latents_std)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use candle_core::{Device, Shape};
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    const REPO: &str = "Qwen/Qwen-Image-2.1";
    const CONFIG_FILE: &str = "vae/config.json";
    const WEIGHTS_FILE: &str = "vae/diffusion_pytorch_model.safetensors";

    /// The shipped geometry with the 64 constants replaced by placeholders,
    /// for the tests that need no file.
    fn shipped_shape() -> VaeConfig {
        VaeConfig {
            base_dim: 96,
            decoder_base_dim: Some(144),
            z_dim: 64,
            dim_mult: vec![1, 2, 4, 8, 8],
            num_res_blocks: 2,
            attn_scales: vec![],
            temperal_downsample: vec![false, true, true, true],
            is_residual: true,
            in_channels: 4,
            out_channels: 4,
            patch_size: None,
            scale_factor_spatial: 16,
            latents_mean: vec![0.0; 64],
            latents_std: vec![1.0; 64],
        }
    }

    /// A narrow config with every stage kind of the shipped one, cheap enough
    /// to run on the CPU.
    pub(crate) fn tiny() -> VaeConfig {
        VaeConfig {
            base_dim: 4,
            decoder_base_dim: Some(6),
            z_dim: 3,
            latents_mean: vec![0.25, -0.5, 1.0],
            latents_std: vec![2.0, 0.5, 1.5],
            ..shipped_shape()
        }
    }

    fn ramp(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| ((i * 37 % 101) as f32) * 0.1 - 5.0)
            .collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        assert_eq!(a.dims(), b.dims());
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// The reference's `AvgDown3D.forward` line by line on a `[B, C, 1, H, W]`
    /// input: the zero frames padded in front, the 8-D view, the permute, the
    /// two views and the group mean. Returns the single output frame.
    fn avg_down_3d(x: &Tensor, c_out: usize, ft: usize, fs: usize) -> Tensor {
        let x = x.unsqueeze(2).unwrap();
        let pad_t = (ft - x.dim(2).unwrap() % ft) % ft;
        let x = x.pad_with_zeros(2, pad_t, 0).unwrap();
        let (b, c, t, h, w) = x.dims5().unwrap();
        let factor = ft * fs * fs;
        let group = c * factor / c_out;
        let x = x
            .reshape(Shape::from(vec![b, c, t / ft, ft, h / fs, fs, w / fs, fs]))
            .unwrap()
            .permute(vec![0, 1, 3, 5, 7, 2, 4, 6])
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((b, c * factor, t / ft, h / fs, w / fs))
            .unwrap()
            .reshape(Shape::from(vec![b, c_out, group, t / ft, h / fs, w / fs]))
            .unwrap()
            .mean(2)
            .unwrap();
        assert_eq!(x.dim(2).unwrap(), 1);
        x.squeeze(2).unwrap()
    }

    /// The reference's `DupUp3D.forward` with `first_chunk=True` on a
    /// `[B, C, 1, H, W]` input: `repeat_interleave`, the 8-D view, the
    /// permute, the view, and the slice that drops all but the last frame.
    fn dup_up_3d_first_chunk(x: &Tensor, c_out: usize, ft: usize, fs: usize) -> Tensor {
        let (b, c, h, w) = x.dims4().unwrap();
        let repeats = c_out * ft * fs * fs / c;
        assert_eq!(repeats * c, c_out * ft * fs * fs);
        let x = x
            .reshape((b, c, 1, 1, h, w))
            .unwrap()
            .broadcast_as(Shape::from(vec![b, c, repeats, 1, h, w]))
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape(Shape::from(vec![b, c_out, ft, fs, fs, 1, h, w]))
            .unwrap()
            .permute(vec![0, 1, 5, 2, 6, 3, 7, 4])
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((b, c_out, ft, h * fs, w * fs))
            .unwrap()
            .narrow(2, ft - 1, 1)
            .unwrap();
        x.squeeze(2).unwrap()
    }

    /// Every encoder stage's shortcut, as the config builds it, is bit-equal to
    /// the reference arithmetic for one frame.
    #[test]
    fn each_down_shortcut_is_the_reference_arithmetic_for_one_frame() {
        let cfg = tiny();
        let dims = cfg.encoder_dims();
        let mut kinds = Vec::new();
        for i in 0..cfg.dim_mult.len() {
            let (down, temporal) = cfg.down_stage(i);
            let shortcut = DownShortcut::new(dims[i], dims[i + 1], down, temporal).unwrap();
            kinds.push(shortcut);
            let x = ramp(&[2, dims[i], 6, 4]);
            let want = avg_down_3d(
                &x,
                dims[i + 1],
                if temporal { 2 } else { 1 },
                if down { 2 } else { 1 },
            );
            let got = shortcut.forward(&x).unwrap();
            assert_eq!(max_abs_diff(&got, &want), 0.0, "encoder stage {i}");
        }
        use DownShortcut::*;
        assert_eq!(
            kinds,
            [
                Pool,
                ZeroInterleavedPool,
                ZeroInterleavedPool,
                ZeroInterleavedPool,
                Identity
            ]
        );
    }

    /// A temporal down stage's even output channels are identically zero and
    /// its odd ones carry the pooled input: the zero frame sits in FRONT.
    #[test]
    fn a_temporal_down_shortcut_zeroes_the_even_channels() {
        let x = ramp(&[1, 3, 4, 4]);
        let out = DownShortcut::ZeroInterleavedPool.forward(&x).unwrap();
        let pooled = x.avg_pool2d(2).unwrap();
        for c in 0..3 {
            let even = out.narrow(1, 2 * c, 1).unwrap();
            assert_eq!(max_abs_diff(&even, &even.zeros_like().unwrap()), 0.0);
            let odd = out.narrow(1, 2 * c + 1, 1).unwrap();
            assert_eq!(max_abs_diff(&odd, &pooled.narrow(1, c, 1).unwrap()), 0.0);
        }
    }

    /// Every decoder stage's shortcut, as the config builds it, is bit-equal
    /// to the reference arithmetic for the first chunk.
    #[test]
    fn each_up_shortcut_is_the_reference_arithmetic_for_the_first_chunk() {
        let cfg = tiny();
        let dims = cfg.decoder_dims();
        let mut kinds = Vec::new();
        for i in 0..cfg.dim_mult.len() {
            let (up, temporal) = cfg.up_stage(i);
            if !up {
                continue;
            }
            let shortcut = UpShortcut::new(dims[i], dims[i + 1], temporal).unwrap();
            kinds.push(shortcut);
            let x = ramp(&[2, dims[i], 3, 5]);
            let want = dup_up_3d_first_chunk(&x, dims[i + 1], if temporal { 2 } else { 1 }, 2);
            let got = shortcut.forward(&x).unwrap();
            assert_eq!(max_abs_diff(&got, &want), 0.0, "decoder stage {i}");
        }
        use UpShortcut::*;
        assert_eq!(kinds, [Nearest, Nearest, OddChannelNearest, RowInterleave]);
    }

    /// The two width-halving up shortcuts, spelled out per pixel, so the
    /// mapping is pinned independently of the transcription above.
    #[test]
    fn the_halving_up_shortcuts_read_the_channels_they_say() {
        let x = ramp(&[1, 4, 2, 3]);
        let src = x.squeeze(0).unwrap().to_vec3::<f32>().unwrap();
        let odd = UpShortcut::OddChannelNearest.forward(&x).unwrap();
        let odd = odd.squeeze(0).unwrap().to_vec3::<f32>().unwrap();
        let rows = UpShortcut::RowInterleave.forward(&x).unwrap();
        let rows = rows.squeeze(0).unwrap().to_vec3::<f32>().unwrap();
        for o in 0..2 {
            for y in 0..4 {
                for xx in 0..6 {
                    assert_eq!(odd[o][y][xx], src[2 * o + 1][y / 2][xx / 2]);
                    assert_eq!(rows[o][y][xx], src[2 * o + y % 2][y / 2][xx / 2]);
                }
            }
        }
    }

    /// `RMS_norm` is `F.normalize(x, dim=1) * sqrt(C) * gamma`: every pixel's
    /// channel vector comes out with L2 norm `sqrt(C)` under a unit gamma, and
    /// an all-zero pixel stays zero instead of dividing by zero.
    #[test]
    fn the_norm_is_an_l2_norm_over_channels_per_pixel() {
        let c = 5;
        let norm = ChannelL2Norm {
            gamma: Tensor::full((c as f32).sqrt(), (1, c, 1, 1), &Device::Cpu).unwrap(),
        };
        let x = ramp(&[1, c, 2, 3]);
        let x = Tensor::cat(&[&x, &x.zeros_like().unwrap()], 2).unwrap();
        let y = norm.forward(&x).unwrap();
        let lens = y.sqr().unwrap().sum(1).unwrap().sqrt().unwrap();
        let lens = lens.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, len) in lens.iter().enumerate() {
            let want = if i < 6 { (c as f32).sqrt() } else { 0.0 };
            assert!((len - want).abs() < 1e-5, "pixel {i}: {len} against {want}");
        }
    }

    /// The norm as the loader builds it, from a stored `gamma` in each of the
    /// two shapes the file uses, against `F.normalize(x, dim=1) * sqrt(C) *
    /// gamma` worked out per pixel by hand: the `sqrt(C)` folded in at load is
    /// there exactly once, and a zero pixel stays zero.
    #[test]
    fn the_loaded_norm_carries_sqrt_c_and_gamma() {
        let c = 5;
        let gamma: Vec<f32> = vec![0.5, -1.25, 2.0, 1.0, 0.125];
        let x = ramp(&[1, c, 2, 3]);
        let x = Tensor::cat(&[&x, &x.zeros_like().unwrap()], 2).unwrap();
        let src = x.squeeze(0).unwrap().to_vec3::<f32>().unwrap();
        type Build = fn(usize, VarBuilder) -> Result<ChannelL2Norm>;
        let cases: [(&[usize], Build); 2] = [
            (&[5, 1, 1, 1], ChannelL2Norm::residual),
            (&[5, 1, 1], ChannelL2Norm::attention),
        ];
        for (stored, build) in cases {
            let tensors = std::collections::HashMap::from([(
                "norm.gamma".to_string(),
                Tensor::from_vec(gamma.clone(), stored, &Device::Cpu).unwrap(),
            )]);
            let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
            let norm = build(c, vb.pp("norm")).unwrap();
            let got = norm.forward(&x).unwrap();
            let got = got.squeeze(0).unwrap().to_vec3::<f32>().unwrap();
            for y in 0..4 {
                for xx in 0..3 {
                    let len = (0..c)
                        .map(|ch| (src[ch][y][xx] as f64).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        .max(1e-12);
                    for ch in 0..c {
                        let want =
                            src[ch][y][xx] as f64 / len * (c as f64).sqrt() * gamma[ch] as f64;
                        let got = got[ch][y][xx] as f64;
                        assert!(
                            (got - want).abs() < 1e-5,
                            "{stored:?} ch {ch} ({y},{xx}): {got} against {want}"
                        );
                        assert!(y < 2 || got == 0.0, "zero pixel gave {got}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_arm_switch_refuses_the_arm_that_does_not_exist() {
        assert_eq!(VaeImpl::parse("").unwrap(), VaeImpl::Candle);
        assert_eq!(VaeImpl::parse(" Candle ").unwrap(), VaeImpl::Candle);
        for value in ["xwen", "direct", "steel"] {
            assert!(VaeImpl::parse(value).is_err(), "{value}");
        }
    }

    #[test]
    fn the_config_refuses_the_branches_this_module_does_not_implement() {
        shipped_shape().validate().unwrap();
        let cases: [fn(&mut VaeConfig); 6] = [
            |c| c.is_residual = false,
            |c| c.attn_scales = vec![1.0],
            |c| c.patch_size = Some(2),
            |c| c.temperal_downsample = vec![false, true, true],
            |c| c.latents_std[3] = 0.0,
            |c| c.scale_factor_spatial = 8,
        ];
        for (i, mutate) in cases.iter().enumerate() {
            let mut cfg = shipped_shape();
            mutate(&mut cfg);
            assert!(cfg.validate().is_err(), "case {i}");
        }
    }

    /// A backend that serves zeros for exactly the tensors of a table, checks
    /// the requested shape against it, and records what was asked for.
    struct Recording {
        table: BTreeMap<String, Vec<usize>>,
        asked: Arc<Mutex<BTreeSet<String>>>,
    }

    impl candle_nn::var_builder::SimpleBackend for Recording {
        fn get(
            &self,
            s: Shape,
            name: &str,
            _: candle_nn::Init,
            dtype: DType,
            dev: &Device,
        ) -> Result<Tensor> {
            match self.table.get(name) {
                Some(shape) if shape.as_slice() == s.dims() => {
                    self.asked.lock().unwrap().insert(name.to_string());
                    Tensor::zeros(s, dtype, dev)
                }
                Some(shape) => candle_core::bail!("{name}: asked for {s:?}, table has {shape:?}"),
                None => candle_core::bail!("{name}: not in the table"),
            }
        }

        fn get_unchecked(&self, name: &str, _: DType, _: &Device) -> Result<Tensor> {
            candle_core::bail!("{name}: unchecked read")
        }

        fn contains_tensor(&self, name: &str) -> bool {
            self.table.contains_key(name)
        }
    }

    /// The loader reads exactly [`live_tensors`], each at the table's shape:
    /// nothing outside it, nothing in it left unread, and no `time_conv`.
    #[test]
    fn the_loader_reads_exactly_the_live_table() {
        for cfg in [tiny(), shipped_shape()] {
            let table: BTreeMap<_, _> = live_tensors(&cfg).into_iter().collect();
            assert_eq!(table.len(), live_tensors(&cfg).len(), "duplicate names");
            let asked = Arc::new(Mutex::new(BTreeSet::new()));
            let backend = Recording {
                table: table.clone(),
                asked: asked.clone(),
            };
            let vb = VarBuilder::from_backend(Box::new(backend), DType::F32, Device::Cpu);
            QwenImageVae::load_with_impl(vb, &cfg, VaeImpl::Candle).unwrap();
            let asked = asked.lock().unwrap();
            let expected: BTreeSet<_> = table.keys().cloned().collect();
            assert_eq!(*asked, expected);
            assert!(
                dead_tensors(&cfg)
                    .iter()
                    .all(|(name, _)| !asked.contains(name))
            );
        }
    }

    /// The tiny model end to end on the CPU: shapes on both sides, the clamp,
    /// and the normalisation living inside the API (a zero-weight decoder's
    /// output does not depend on it, so that half is checked on `encode`,
    /// whose zero-weight moments are zero and come back as `-mean / std`).
    #[test]
    fn the_api_owns_the_shapes_and_the_latent_normalisation() {
        let cfg = tiny();
        let table: BTreeMap<_, _> = live_tensors(&cfg).into_iter().collect();
        let backend = Recording {
            table,
            asked: Arc::new(Mutex::new(BTreeSet::new())),
        };
        let vb = VarBuilder::from_backend(Box::new(backend), DType::F32, Device::Cpu);
        let vae = QwenImageVae::load_with_impl(vb, &cfg, VaeImpl::Candle).unwrap();

        let image = vae.decode(&ramp(&[1, 3, 2, 3])).unwrap();
        assert_eq!(image.dims(), [1, 4, 32, 48]);
        let latent = vae.encode(&ramp(&[1, 4, 32, 48])).unwrap();
        assert_eq!(latent.dims(), [1, 3, 2, 3]);
        let got = latent.mean((0, 2, 3)).unwrap().to_vec1::<f32>().unwrap();
        for (c, got) in got.iter().enumerate() {
            let want = (-cfg.latents_mean[c] / cfg.latents_std[c]) as f32;
            assert!(
                (got - want).abs() < 1e-6,
                "channel {c}: {got} against {want}"
            );
        }

        assert!(vae.encode(&ramp(&[1, 4, 30, 48])).is_err());
        assert!(vae.encode(&ramp(&[1, 3, 32, 48])).is_err());
        assert!(vae.decode(&ramp(&[1, 4, 2, 3])).is_err());
    }

    /// The tiny model with deterministic NON-zero weights: every element a
    /// function of its tensor's name and its index, within `amplitude` of zero
    /// (of one, for a `gamma`). No two output channels share a filter, so the
    /// two halves of the posterior differ and every stage carries signal.
    pub(crate) fn patterned(cfg: &VaeConfig, amplitude: f32) -> QwenImageVae {
        let tensors = live_tensors(cfg)
            .into_iter()
            .map(|(name, shape)| {
                let seed = name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                    (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
                });
                let n: usize = shape.iter().product();
                let base = if name.ends_with(".gamma") { 1.0 } else { 0.0 };
                let data: Vec<f32> = (0..n as u64)
                    .map(|i| {
                        let h = (seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
                            .wrapping_mul(0xD6E8_FEB8_6659_FD93);
                        let unit = (h >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                        base + 2.0 * amplitude * unit
                    })
                    .collect();
                let tensor = Tensor::from_vec(data, shape, &Device::Cpu).unwrap();
                (name, tensor)
            })
            .collect();
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        QwenImageVae::load_with_impl(vb, cfg, VaeImpl::Candle).unwrap()
    }

    fn per_channel(values: &[f64]) -> Tensor {
        let values: Vec<f32> = values.iter().map(|v| *v as f32).collect();
        let n = values.len();
        Tensor::from_vec(values, (1, n, 1, 1), &Device::Cpu).unwrap()
    }

    /// `encode` is the FIRST half of the moments, normalised: equal to that
    /// worked out from the raw moments, and not the log-variance half.
    #[test]
    fn encode_normalises_the_mean_half_of_the_posterior() {
        let cfg = tiny();
        let vae = patterned(&cfg, 0.2);
        let image = (ramp(&[1, 4, 32, 48]) * 0.2).unwrap();
        let moments = vae.moments(&image).unwrap();
        assert_eq!(moments.dims(), [1, 6, 2, 3]);
        let normalise = |half: usize| {
            moments
                .narrow(1, half * cfg.z_dim, cfg.z_dim)
                .unwrap()
                .broadcast_sub(&per_channel(&cfg.latents_mean))
                .unwrap()
                .broadcast_div(&per_channel(&cfg.latents_std))
                .unwrap()
        };
        let got = vae.encode(&image).unwrap();
        assert!(max_abs_diff(&got, &normalise(0)) < 1e-6);
        assert!(max_abs_diff(&got, &normalise(1)) > 1e-3);
    }

    /// `decode` is the trunk over `z * std + mean`, clamped: equal to that, and
    /// neither the trunk over the raw latent nor over `(z - mean) / std`.
    #[test]
    fn decode_denormalises_the_latent_before_the_trunk() {
        let cfg = tiny();
        let vae = patterned(&cfg, 0.05);
        let z = (ramp(&[1, 3, 2, 3]) * 0.2).unwrap();
        let (mean, std) = (
            per_channel(&cfg.latents_mean),
            per_channel(&cfg.latents_std),
        );
        let trunk = |z: &Tensor| vae.decode_trunk(z).unwrap().clamp(-1f32, 1f32).unwrap();
        let forward = z.broadcast_mul(&std).unwrap().broadcast_add(&mean).unwrap();
        let reversed = z.broadcast_sub(&mean).unwrap().broadcast_div(&std).unwrap();
        let got = vae.decode(&z).unwrap();
        assert_eq!(got.dims(), [1, 4, 32, 48]);
        assert!(max_abs_diff(&got, &trunk(&forward)) < 1e-6);
        assert!(max_abs_diff(&got, &trunk(&z)) > 1e-3);
        assert!(max_abs_diff(&got, &trunk(&reversed)) > 1e-3);
    }

    /// With weights large enough that the trunk leaves `[-1, 1]` on both
    /// sides, `decode` returns exactly the clamped trunk.
    #[test]
    fn decode_clamps_to_the_unit_range() {
        let cfg = tiny();
        let vae = patterned(&cfg, 4.0);
        let z = (ramp(&[1, 3, 2, 3]) * 0.2).unwrap();
        let denormalised = z
            .broadcast_mul(&per_channel(&cfg.latents_std))
            .unwrap()
            .broadcast_add(&per_channel(&cfg.latents_mean))
            .unwrap();
        let raw = vae.decode_trunk(&denormalised).unwrap();
        let raw = raw.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(raw.iter().any(|v| *v > 1.0) && raw.iter().any(|v| *v < -1.0));
        let got = vae.decode(&z).unwrap();
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (got, raw) in got.iter().zip(&raw) {
            assert_eq!(*got, raw.clamp(-1.0, 1.0));
        }
        assert!(got.iter().any(|v| *v == 1.0) && got.iter().any(|v| *v == -1.0));
    }

    /// The snapshot root to read the shipped VAE from: `XWEN_QWEN_IMAGE_DIR`
    /// when set, else the Hugging Face cache, else a visible skip.
    fn shipped(file: &str) -> Option<PathBuf> {
        if let Some(root) = std::env::var_os("XWEN_QWEN_IMAGE_DIR") {
            let path = PathBuf::from(root).join(file);
            assert!(path.exists(), "XWEN_QWEN_IMAGE_DIR has no {file}");
            return Some(path);
        }
        crate::test_support::repo_file_or_skip(
            REPO,
            file,
            "uvx --from huggingface_hub hf download Qwen/Qwen-Image-2.1",
        )
    }

    fn shipped_config() -> Option<VaeConfig> {
        let path = shipped(CONFIG_FILE)?;
        Some(serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
    }

    /// Name, dtype and shape of every tensor, from the safetensors header
    /// alone.
    fn header(path: &std::path::Path) -> BTreeMap<String, (String, Vec<usize>)> {
        use std::io::Read;
        let mut file = std::fs::File::open(path).unwrap();
        let mut len = [0u8; 8];
        file.read_exact(&mut len).unwrap();
        let mut json = vec![0u8; u64::from_le_bytes(len) as usize];
        file.read_exact(&mut json).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&json).unwrap();
        json.as_object()
            .unwrap()
            .iter()
            .filter(|(name, _)| name.as_str() != "__metadata__")
            .map(|(name, entry)| {
                let shape = entry["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect();
                let dtype = entry["dtype"].as_str().unwrap().to_string();
                (name.clone(), (dtype, shape))
            })
            .collect()
    }

    /// The shipped file is the live table plus the dead list and nothing else,
    /// shape for shape, all of it F32 with 2-D convolution weights.
    #[test]
    fn the_shipped_file_is_the_live_table_plus_the_dead_list() {
        let (Some(cfg), Some(weights)) = (shipped_config(), shipped(WEIGHTS_FILE)) else {
            return;
        };
        cfg.validate().unwrap();
        let mut expected = BTreeMap::new();
        for (name, shape) in live_tensors(&cfg).into_iter().chain(dead_tensors(&cfg)) {
            assert!(
                expected.insert(name.clone(), shape).is_none(),
                "{name} listed twice"
            );
        }
        let file = header(&weights);
        for (name, (dtype, shape)) in &file {
            assert_eq!(dtype, "F32", "{name}");
            assert_eq!(expected.get(name), Some(shape), "{name} in the file");
        }
        for name in expected.keys() {
            assert!(file.contains_key(name), "{name} is not in the file");
        }
    }

    /// The shipped config has the geometry the file-free tests assume.
    #[test]
    fn the_shipped_config_is_the_geometry_the_tests_assume() {
        let Some(cfg) = shipped_config() else {
            return;
        };
        let mut want = shipped_shape();
        want.latents_mean = cfg.latents_mean.clone();
        want.latents_std = cfg.latents_std.clone();
        assert_eq!(format!("{want:?}"), format!("{cfg:?}"));
    }

    /// Smoke, not a gate: the shipped weights decode a fixed random latent at
    /// 256x256 to a finite, non-constant RGBA image inside the clamp, and
    /// encode that image to a finite latent of the shape it came from. The
    /// timing and the correlation of `z` with `encode(decode(z))` are printed
    /// and deliberately not asserted, a random latent being far off the
    /// decoder's manifold; the numeric gate for this module is the comparison
    /// against the reference fixture.
    #[test]
    #[ignore = "loads the shipped VAE and runs it on the GPU"]
    fn the_shipped_vae_round_trips_a_latent() {
        let (Some(cfg), Some(weights)) = (shipped_config(), shipped(WEIGHTS_FILE)) else {
            return;
        };
        let device = Device::new_metal(0).unwrap_or(Device::Cpu);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device).unwrap()
        };
        let vae = QwenImageVae::load_with_impl(vb, &cfg, VaeImpl::Candle).unwrap();

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut uniform = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 + 0.5) / (1u64 << 24) as f32
        };
        let n = cfg.z_dim * 16 * 16;
        let z: Vec<f32> = (0..n)
            .map(|_| {
                let (u1, u2) = (uniform(), uniform());
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();
        let z = Tensor::from_vec(z, (1, cfg.z_dim, 16, 16), &device).unwrap();

        let t0 = std::time::Instant::now();
        let image = vae.decode(&z).unwrap();
        let flat = image.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let decode_s = t0.elapsed().as_secs_f64();
        assert_eq!(image.dims(), [1, 4, 256, 256]);
        assert!(
            flat.iter()
                .all(|v| v.is_finite() && (-1.0..=1.0).contains(v))
        );

        let centre = flat.iter().map(|v| *v as f64).sum::<f64>() / flat.len() as f64;
        let spread = flat
            .iter()
            .map(|v| (*v as f64 - centre).powi(2))
            .sum::<f64>();
        assert!(spread > 0.0, "the decoded image is constant");

        let t1 = std::time::Instant::now();
        let back = vae.encode(&image).unwrap();
        assert_eq!(back.dims(), [1, cfg.z_dim, 16, 16]);
        let back = back.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let encode_s = t1.elapsed().as_secs_f64();
        let z = z.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(back.iter().all(|v| v.is_finite()));

        let mean = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
        let (ma, mb) = (mean(&z), mean(&back));
        let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
        for (a, b) in z.iter().zip(&back) {
            let (a, b) = (*a as f64 - ma, *b as f64 - mb);
            ab += a * b;
            aa += a * a;
            bb += b * b;
        }
        let alpha = mean(&flat[3 * 256 * 256..]);
        eprintln!(
            "qwen-image vae smoke on {device:?}: decode {decode_s:.3} s, encode {encode_s:.3} s, \
             corr(z, encode(decode(z))) = {:.4}, mean alpha {alpha:.4}",
            ab / (aa * bb).sqrt()
        );
    }
}
