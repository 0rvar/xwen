// Vendored from candle rev 21cca0b (candle-transformers/src/models/z_image/vae.rs,
// PR #3261, SpenserCai), MIT / Apache-2.0. The module structure and the candle
// arm are that file; the xwen arm (the direct convolution with the norm, the
// activation, the upsample and the residual folded into it) is ours.
//! Z-Image VAE (AutoEncoderKL) - Diffusers Format
//!
//! This VAE implementation uses the diffusers weight naming format,
//! which is different from the Flux autoencoder original format.
//!
//! Key differences from Flux autoencoder:
//! 1. Weight paths: `encoder.down_blocks.{i}.resnets.{j}.*` vs `encoder.down.{i}.block.{j}.*`
//! 2. Attention naming: `to_q/to_k/to_v/to_out.0.*` vs `q/k/v/proj_out.*`
//! 3. Shortcut naming: `conv_shortcut.*` vs `nin_shortcut.*`
//!
//! The decoder runs on one of two arms, [`VaeImpl`], chosen by [`VAE_ENV`]
//! at load. The shipped arm replaces every candle `GroupNorm`, `Swish` and
//! `conv2d` of the decoder with xwen's kernels: the GroupNorm statistics are
//! folded to a per-channel affine (`ops::group_norm_fold`) and the direct
//! convolution (`ops::conv2d_direct`) applies that affine and the silu on its
//! input read, reads the 2x nearest upsample at half coordinates, and adds the
//! resnet's residual on its store. The normalized tensor, the activated tensor,
//! the upsampled tensor and candle's im2col are never written. `candle` is the
//! bisect arm, the chain as vendored.

use candle_core::{D, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, GroupNorm, VarBuilder, conv2d};

use super::profile::Profiler;
use crate::ops::{self, Conv2dFusion};

/// The environment switch that picks the VAE decoder's kernels, read when a
/// `ZImagePipeline` loads. `xwen` (or unset) is the direct-convolution path;
/// `candle` is the bisect arm.
pub const VAE_ENV: &str = "XWEN_ZIMAGE_VAE";

/// Which kernels the VAE decoder runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VaeImpl {
    /// The shipped path: xwen's direct convolution with the GroupNorm, silu,
    /// upsample and residual folded in, on a Metal device. Off Metal every
    /// block falls through to the candle chain, the kernels being Metal-only.
    Xwen,
    /// candle's GroupNorm, Swish and im2col conv2d, the chain as vendored.
    Candle,
}

impl VaeImpl {
    /// Resolve from [`VAE_ENV`]: unset means the shipped path, anything else
    /// must name an arm.
    pub fn from_env() -> Result<Self> {
        match std::env::var(VAE_ENV) {
            Err(std::env::VarError::NotPresent) => Ok(Self::Xwen),
            Err(std::env::VarError::NotUnicode(_)) => {
                candle_core::bail!("{VAE_ENV} is not valid UTF-8")
            }
            Ok(value) => Self::parse(&value),
        }
    }

    /// `xwen` / `direct` (or empty) select the shipped path, `candle` the
    /// bisect arm; anything else is refused rather than defaulted.
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "xwen" | "direct" => Ok(Self::Xwen),
            "candle" => Ok(Self::Candle),
            other => {
                candle_core::bail!("{VAE_ENV}={other:?}: expected `xwen` (the default) or `candle`")
            }
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Xwen => "xwen",
            Self::Candle => "candle",
        }
    }

    /// The arm a block runs for `xs`: the xwen kernels need a Metal device.
    fn resolve(self, xs: &Tensor) -> Self {
        match self {
            Self::Xwen if xs.device().is_metal() => Self::Xwen,
            _ => Self::Candle,
        }
    }
}

fn wrap(e: anyhow::Error) -> candle_core::Error {
    candle_core::Error::Msg(format!("{e:#}"))
}

/// The profile labels of the decoder's up blocks, resnet stack and upsample
/// convolution apart. One pair per block of the shipped four; a config with
/// more blocks than this reports the rest under a shared row rather than
/// failing a profiled run.
const UP_LABELS: [(&str, &str); 4] = [
    ("up0.resnets", "up0.upsample"),
    ("up1.resnets", "up1.upsample"),
    ("up2.resnets", "up2.upsample"),
    ("up3.resnets", "up3.upsample"),
];

// ==================== Config ====================

/// VAE configuration
#[derive(Debug, Clone, serde::Deserialize)]
pub struct VaeConfig {
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_out_channels")]
    pub out_channels: usize,
    #[serde(default = "default_latent_channels")]
    pub latent_channels: usize,
    #[serde(default = "default_block_out_channels")]
    pub block_out_channels: Vec<usize>,
    #[serde(default = "default_layers_per_block")]
    pub layers_per_block: usize,
    #[serde(default = "default_scaling_factor")]
    pub scaling_factor: f64,
    #[serde(default = "default_shift_factor")]
    pub shift_factor: f64,
    #[serde(default = "default_norm_num_groups")]
    pub norm_num_groups: usize,
}

fn default_in_channels() -> usize {
    3
}
fn default_out_channels() -> usize {
    3
}
fn default_latent_channels() -> usize {
    16
}
fn default_block_out_channels() -> Vec<usize> {
    vec![128, 256, 512, 512]
}
fn default_layers_per_block() -> usize {
    2
}
fn default_scaling_factor() -> f64 {
    0.3611
}
fn default_shift_factor() -> f64 {
    0.1159
}
fn default_norm_num_groups() -> usize {
    32
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self::z_image()
    }
}

impl VaeConfig {
    /// Create configuration for Z-Image VAE
    pub fn z_image() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 16,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            scaling_factor: 0.3611,
            shift_factor: 0.1159,
            norm_num_groups: 32,
        }
    }
}

// ==================== The two-arm layers ====================

/// A square convolution with `padding = kernel / 2`: candle's `Conv2d` for
/// the candle arm and for any shape the direct kernel declines, plus, on the
/// xwen arm, the weight permuted to the direct kernel's `[k*k, c_in, c_out]`
/// plane and the bias it stores.
#[derive(Debug, Clone)]
struct Conv {
    candle: Conv2d,
    kernel: usize,
    direct: Option<(Tensor, Tensor)>,
}

impl Conv {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        arm: VaeImpl,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: kernel / 2,
            ..Default::default()
        };
        let candle = conv2d(in_channels, out_channels, kernel, cfg, vb)?;
        let direct = if arm == VaeImpl::Xwen && ops::conv2d_direct_supported(in_channels, kernel) {
            let w = ops::permute_conv_weight(candle.weight()).map_err(wrap)?;
            let b = match candle.bias() {
                Some(b) => b.clone(),
                None => Tensor::zeros(out_channels, w.dtype(), w.device())?,
            };
            Some((w, b))
        } else {
            None
        };
        Ok(Self {
            candle,
            kernel,
            direct,
        })
    }

    /// The convolution with `fusion` folded in: the direct kernel does it in
    /// one dispatch; a shape it declines runs the fused GroupNorm apply, the
    /// upsample and candle's conv2d, then adds the residual. `xs` is on a
    /// Metal device either way (the xwen arm resolved).
    fn forward_fused(&self, xs: &Tensor, fusion: Conv2dFusion<'_>) -> Result<Tensor> {
        if let Some((w, b)) = &self.direct {
            return ops::conv2d_direct(xs, w, b, self.kernel, fusion).map_err(wrap);
        }
        let mut v = match fusion.norm {
            Some((scale, shift)) => {
                ops::group_norm_apply(xs, scale, shift, fusion.silu).map_err(wrap)?
            }
            None if fusion.silu => xs.silu()?,
            None => xs.clone(),
        };
        if fusion.upsample {
            let (_, _, h, w) = v.dims4()?;
            v = v.upsample_nearest2d(h * 2, w * 2)?;
        }
        let out = self.candle.forward(&v)?;
        match fusion.residual {
            Some(r) => out + r,
            None => Ok(out),
        }
    }
}

impl Module for Conv {
    /// candle's conv2d, the candle arm.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.candle.forward(xs)
    }
}

/// A GroupNorm: candle's for the candle arm, and its weight, bias and
/// geometry for the fold on the xwen arm.
#[derive(Debug, Clone)]
struct Norm {
    candle: GroupNorm,
    weight: Tensor,
    bias: Tensor,
    groups: usize,
    eps: f32,
}

impl Norm {
    fn new(groups: usize, channels: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(channels, "weight", candle_nn::Init::Const(1.))?;
        let bias = vb.get_with_hints(channels, "bias", candle_nn::Init::Const(0.))?;
        let candle = GroupNorm::new(weight.clone(), bias.clone(), channels, groups, eps)?;
        Ok(Self {
            candle,
            weight,
            bias,
            groups,
            eps: eps as f32,
        })
    }

    /// The statistics of `xs` folded to `(scale, shift)` per (batch, channel),
    /// for the direct convolution's input read or `ops::group_norm_apply`.
    fn fold(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        ops::group_norm_fold(xs, self.groups, &self.weight, &self.bias, self.eps).map_err(wrap)
    }
}

impl Module for Norm {
    /// candle's GroupNorm, the candle arm.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.candle.forward(xs)
    }
}

// ==================== Attention ====================

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(v)
}

/// VAE Attention block (diffusers format)
///
/// Note: VAE attention uses Linear with bias (2D weight shape)
/// Unlike Transformer attention which uses linear_no_bias
#[derive(Debug, Clone)]
struct Attention {
    group_norm: Norm,
    to_q: candle_nn::Linear,
    to_k: candle_nn::Linear,
    to_v: candle_nn::Linear,
    to_out: candle_nn::Linear,
    arm: VaeImpl,
}

impl Attention {
    fn new(channels: usize, num_groups: usize, arm: VaeImpl, vb: VarBuilder) -> Result<Self> {
        let group_norm = Norm::new(num_groups, channels, 1e-6, vb.pp("group_norm"))?;
        // VAE attention uses Linear with bias
        let to_q = candle_nn::linear(channels, channels, vb.pp("to_q"))?;
        let to_k = candle_nn::linear(channels, channels, vb.pp("to_k"))?;
        let to_v = candle_nn::linear(channels, channels, vb.pp("to_v"))?;
        let to_out = candle_nn::linear(channels, channels, vb.pp("to_out").pp("0"))?;
        Ok(Self {
            group_norm,
            to_q,
            to_k,
            to_v,
            to_out,
            arm,
        })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let (b, c, h, w) = xs.dims4()?;

        // GroupNorm. The attention itself stays on candle on both arms: at the
        // decoder's fixed 128x128 mid block it is a small row of the decode.
        let xs = match self.arm.resolve(xs) {
            VaeImpl::Xwen => {
                let (scale, shift) = self.group_norm.fold(xs)?;
                ops::group_norm_apply(xs, &scale, &shift, false).map_err(wrap)?
            }
            VaeImpl::Candle => xs.apply(&self.group_norm)?,
        };

        // (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
        let xs = xs.permute((0, 2, 3, 1))?.reshape((b * h * w, c))?;

        // Linear projections
        let q = xs.apply(&self.to_q)?; // (B*H*W, C)
        let k = xs.apply(&self.to_k)?;
        let v = xs.apply(&self.to_v)?;

        // Reshape for attention: (B*H*W, C) -> (B, H*W, C) -> (B, 1, H*W, C)
        let q = q.reshape((b, h * w, c))?.unsqueeze(1)?;
        let k = k.reshape((b, h * w, c))?.unsqueeze(1)?;
        let v = v.reshape((b, h * w, c))?.unsqueeze(1)?;

        // Scaled dot-product attention
        let xs = scaled_dot_product_attention(&q, &k, &v)?;

        // (B, 1, H*W, C) -> (B*H*W, C)
        let xs = xs.squeeze(1)?.reshape((b * h * w, c))?;

        // Output projection
        let xs = xs.apply(&self.to_out)?;

        // (B*H*W, C) -> (B, H, W, C) -> (B, C, H, W)
        let xs = xs.reshape((b, h, w, c))?.permute((0, 3, 1, 2))?;

        // Residual connection
        xs + residual
    }
}

// ==================== ResnetBlock2D ====================

/// ResNet block (diffusers format)
#[derive(Debug, Clone)]
struct ResnetBlock2D {
    norm1: Norm,
    conv1: Conv,
    norm2: Norm,
    conv2: Conv,
    conv_shortcut: Option<Conv>,
    arm: VaeImpl,
}

impl ResnetBlock2D {
    fn new(
        in_channels: usize,
        out_channels: usize,
        num_groups: usize,
        arm: VaeImpl,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = Norm::new(num_groups, in_channels, 1e-6, vb.pp("norm1"))?;
        let conv1 = Conv::new(in_channels, out_channels, 3, arm, vb.pp("conv1"))?;
        let norm2 = Norm::new(num_groups, out_channels, 1e-6, vb.pp("norm2"))?;
        let conv2 = Conv::new(out_channels, out_channels, 3, arm, vb.pp("conv2"))?;

        let conv_shortcut = if in_channels != out_channels {
            Some(Conv::new(
                in_channels,
                out_channels,
                1,
                arm,
                vb.pp("conv_shortcut"),
            )?)
        } else {
            None
        };

        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            conv_shortcut,
            arm,
        })
    }
}

impl Module for ResnetBlock2D {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self.arm.resolve(xs) {
            VaeImpl::Candle => {
                let h = xs
                    .apply(&self.norm1)?
                    .apply(&candle_nn::Activation::Swish)?
                    .apply(&self.conv1)?
                    .apply(&self.norm2)?
                    .apply(&candle_nn::Activation::Swish)?
                    .apply(&self.conv2)?;

                match &self.conv_shortcut {
                    Some(conv) => xs.apply(conv)? + h,
                    None => xs + h,
                }
            }
            VaeImpl::Xwen => {
                // norm1 and the silu ride on conv1's input read, norm2 and
                // the silu on conv2's, and the skip add on conv2's store.
                let (scale1, shift1) = self.norm1.fold(xs)?;
                let h = self.conv1.forward_fused(
                    xs,
                    Conv2dFusion {
                        norm: Some((&scale1, &shift1)),
                        silu: true,
                        ..Default::default()
                    },
                )?;
                let (scale2, shift2) = self.norm2.fold(&h)?;
                let shortcut = match &self.conv_shortcut {
                    Some(conv) => Some(conv.forward_fused(xs, Conv2dFusion::default())?),
                    None => None,
                };
                let residual = shortcut.as_ref().unwrap_or(xs);
                self.conv2.forward_fused(
                    &h,
                    Conv2dFusion {
                        norm: Some((&scale2, &shift2)),
                        silu: true,
                        residual: Some(residual),
                        ..Default::default()
                    },
                )
            }
        }
    }
}

// ==================== DownEncoderBlock2D ====================

#[derive(Debug, Clone)]
struct Downsample2D {
    conv: Conv2d,
}

impl Downsample2D {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: 2,
            padding: 0,
            ..Default::default()
        };
        let conv = conv2d(channels, channels, 3, conv_cfg, vb.pp("conv"))?;
        Ok(Self { conv })
    }
}

impl Module for Downsample2D {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Manual padding: (0, 1, 0, 1) for right=1, bottom=1
        let xs = xs.pad_with_zeros(D::Minus1, 0, 1)?; // width: right
        let xs = xs.pad_with_zeros(D::Minus2, 0, 1)?; // height: bottom
        xs.apply(&self.conv)
    }
}

#[derive(Debug, Clone)]
struct DownEncoderBlock2D {
    resnets: Vec<ResnetBlock2D>,
    downsampler: Option<Downsample2D>,
}

impl DownEncoderBlock2D {
    fn new(
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        num_groups: usize,
        add_downsample: bool,
        arm: VaeImpl,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        let vb_resnets = vb.pp("resnets");

        for i in 0..num_layers {
            let in_c = if i == 0 { in_channels } else { out_channels };
            resnets.push(ResnetBlock2D::new(
                in_c,
                out_channels,
                num_groups,
                arm,
                vb_resnets.pp(i),
            )?);
        }

        let downsampler = if add_downsample {
            Some(Downsample2D::new(
                out_channels,
                vb.pp("downsamplers").pp("0"),
            )?)
        } else {
            None
        };

        Ok(Self {
            resnets,
            downsampler,
        })
    }
}

impl Module for DownEncoderBlock2D {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.clone();
        for resnet in &self.resnets {
            h = h.apply(resnet)?;
        }
        if let Some(ds) = &self.downsampler {
            h = h.apply(ds)?;
        }
        Ok(h)
    }
}

// ==================== UpDecoderBlock2D ====================

#[derive(Debug, Clone)]
struct Upsample2D {
    conv: Conv,
    arm: VaeImpl,
}

impl Upsample2D {
    fn new(channels: usize, arm: VaeImpl, vb: VarBuilder) -> Result<Self> {
        let conv = Conv::new(channels, channels, 3, arm, vb.pp("conv"))?;
        Ok(Self { conv, arm })
    }
}

impl Module for Upsample2D {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self.arm.resolve(xs) {
            VaeImpl::Candle => {
                let (_, _, h, w) = xs.dims4()?;
                xs.upsample_nearest2d(h * 2, w * 2)?.apply(&self.conv)
            }
            // The nearest upsample is a read at half coordinates inside the
            // convolution; the 4x tensor is never written.
            VaeImpl::Xwen => self.conv.forward_fused(
                xs,
                Conv2dFusion {
                    upsample: true,
                    ..Default::default()
                },
            ),
        }
    }
}

#[derive(Debug, Clone)]
struct UpDecoderBlock2D {
    resnets: Vec<ResnetBlock2D>,
    upsampler: Option<Upsample2D>,
}

impl UpDecoderBlock2D {
    fn new(
        in_channels: usize,
        out_channels: usize,
        num_layers: usize, // decoder has num_layers + 1 resnets per block
        num_groups: usize,
        add_upsample: bool,
        arm: VaeImpl,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers + 1);
        let vb_resnets = vb.pp("resnets");

        for i in 0..=num_layers {
            let in_c = if i == 0 { in_channels } else { out_channels };
            resnets.push(ResnetBlock2D::new(
                in_c,
                out_channels,
                num_groups,
                arm,
                vb_resnets.pp(i),
            )?);
        }

        let upsampler = if add_upsample {
            Some(Upsample2D::new(
                out_channels,
                arm,
                vb.pp("upsamplers").pp("0"),
            )?)
        } else {
            None
        };

        Ok(Self { resnets, upsampler })
    }
}

impl UpDecoderBlock2D {
    /// [`Module::forward`] with the resnet stack and the upsample
    /// convolution timed under `labels` (see [`Profiler`]).
    fn forward_profiled(
        &self,
        xs: &Tensor,
        prof: &Profiler,
        labels: (&'static str, &'static str),
    ) -> Result<Tensor> {
        let mut h = xs.clone();
        for resnet in &self.resnets {
            h = h.apply(resnet)?;
        }
        prof.mark(labels.0);
        if let Some(us) = &self.upsampler {
            h = h.apply(us)?;
            prof.mark(labels.1);
        }
        Ok(h)
    }
}

impl Module for UpDecoderBlock2D {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.clone();
        for resnet in &self.resnets {
            h = h.apply(resnet)?;
        }
        if let Some(us) = &self.upsampler {
            h = h.apply(us)?;
        }
        Ok(h)
    }
}

// ==================== UNetMidBlock2D ====================

#[derive(Debug, Clone)]
struct UNetMidBlock2D {
    resnet_0: ResnetBlock2D,
    attention: Attention,
    resnet_1: ResnetBlock2D,
}

impl UNetMidBlock2D {
    fn new(channels: usize, num_groups: usize, arm: VaeImpl, vb: VarBuilder) -> Result<Self> {
        let resnet_0 = ResnetBlock2D::new(
            channels,
            channels,
            num_groups,
            arm,
            vb.pp("resnets").pp("0"),
        )?;
        let attention = Attention::new(channels, num_groups, arm, vb.pp("attentions").pp("0"))?;
        let resnet_1 = ResnetBlock2D::new(
            channels,
            channels,
            num_groups,
            arm,
            vb.pp("resnets").pp("1"),
        )?;
        Ok(Self {
            resnet_0,
            attention,
            resnet_1,
        })
    }
}

impl UNetMidBlock2D {
    /// [`Module::forward`] with the two resnets summed into one row and the
    /// attention in its own.
    fn forward_profiled(&self, xs: &Tensor, prof: &Profiler) -> Result<Tensor> {
        let h = xs.apply(&self.resnet_0)?;
        prof.mark("mid.resnet");
        let h = h.apply(&self.attention)?;
        prof.mark("mid.attn");
        let h = h.apply(&self.resnet_1)?;
        prof.mark("mid.resnet");
        Ok(h)
    }
}

impl Module for UNetMidBlock2D {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.resnet_0)?
            .apply(&self.attention)?
            .apply(&self.resnet_1)
    }
}

// ==================== Encoder ====================

/// VAE Encoder
#[derive(Debug, Clone)]
pub struct Encoder {
    conv_in: Conv2d,
    down_blocks: Vec<DownEncoderBlock2D>,
    mid_block: UNetMidBlock2D,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Encoder {
    /// The encoder runs the candle chain: nothing in the image pipeline
    /// encodes, so it carries no permuted planes.
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let arm = VaeImpl::Candle;
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = conv2d(
            cfg.in_channels,
            cfg.block_out_channels[0],
            3,
            conv_cfg,
            vb.pp("conv_in"),
        )?;

        let mut down_blocks = Vec::with_capacity(cfg.block_out_channels.len());
        let vb_down = vb.pp("down_blocks");

        for (i, &out_channels) in cfg.block_out_channels.iter().enumerate() {
            let in_channels = if i == 0 {
                cfg.block_out_channels[0]
            } else {
                cfg.block_out_channels[i - 1]
            };
            let add_downsample = i < cfg.block_out_channels.len() - 1;
            down_blocks.push(DownEncoderBlock2D::new(
                in_channels,
                out_channels,
                cfg.layers_per_block,
                cfg.norm_num_groups,
                add_downsample,
                arm,
                vb_down.pp(i),
            )?);
        }

        let mid_channels = *cfg.block_out_channels.last().unwrap();
        let mid_block =
            UNetMidBlock2D::new(mid_channels, cfg.norm_num_groups, arm, vb.pp("mid_block"))?;

        let conv_norm_out = candle_nn::group_norm(
            cfg.norm_num_groups,
            mid_channels,
            1e-6,
            vb.pp("conv_norm_out"),
        )?;
        let conv_out = conv2d(
            mid_channels,
            2 * cfg.latent_channels,
            3,
            conv_cfg,
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            down_blocks,
            mid_block,
            conv_norm_out,
            conv_out,
        })
    }
}

impl Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.apply(&self.conv_in)?;
        for block in &self.down_blocks {
            h = h.apply(block)?;
        }
        h.apply(&self.mid_block)?
            .apply(&self.conv_norm_out)?
            .apply(&candle_nn::Activation::Swish)?
            .apply(&self.conv_out)
    }
}

// ==================== Decoder ====================

/// VAE Decoder
#[derive(Debug, Clone)]
pub struct Decoder {
    conv_in: Conv,
    mid_block: UNetMidBlock2D,
    up_blocks: Vec<UpDecoderBlock2D>,
    conv_norm_out: Norm,
    conv_out: Conv,
    arm: VaeImpl,
}

impl Decoder {
    pub fn new(cfg: &VaeConfig, arm: VaeImpl, vb: VarBuilder) -> Result<Self> {
        let mid_channels = *cfg.block_out_channels.last().unwrap();

        let conv_in = Conv::new(cfg.latent_channels, mid_channels, 3, arm, vb.pp("conv_in"))?;
        let mid_block =
            UNetMidBlock2D::new(mid_channels, cfg.norm_num_groups, arm, vb.pp("mid_block"))?;

        // Decoder up_blocks order is reversed from encoder down_blocks
        let reversed_channels: Vec<usize> = cfg.block_out_channels.iter().rev().cloned().collect();
        let mut up_blocks = Vec::with_capacity(reversed_channels.len());
        let vb_up = vb.pp("up_blocks");

        for (i, &out_channels) in reversed_channels.iter().enumerate() {
            let in_channels = if i == 0 {
                mid_channels
            } else {
                reversed_channels[i - 1]
            };
            let add_upsample = i < reversed_channels.len() - 1;
            up_blocks.push(UpDecoderBlock2D::new(
                in_channels,
                out_channels,
                cfg.layers_per_block,
                cfg.norm_num_groups,
                add_upsample,
                arm,
                vb_up.pp(i),
            )?);
        }

        let final_channels = *reversed_channels.last().unwrap();
        let conv_norm_out = Norm::new(
            cfg.norm_num_groups,
            final_channels,
            1e-6,
            vb.pp("conv_norm_out"),
        )?;
        let conv_out = Conv::new(final_channels, cfg.out_channels, 3, arm, vb.pp("conv_out"))?;

        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            arm,
        })
    }

    /// The decoder's tail: `conv_norm_out`, the activation and `conv_out`.
    fn tail(&self, h: &Tensor) -> Result<Tensor> {
        match self.arm.resolve(h) {
            VaeImpl::Candle => h
                .apply(&self.conv_norm_out)?
                .apply(&candle_nn::Activation::Swish)?
                .apply(&self.conv_out),
            VaeImpl::Xwen => {
                let (scale, shift) = self.conv_norm_out.fold(h)?;
                self.conv_out.forward_fused(
                    h,
                    Conv2dFusion {
                        norm: Some((&scale, &shift)),
                        silu: true,
                        ..Default::default()
                    },
                )
            }
        }
    }
}

impl Decoder {
    /// [`Module::forward`] with one row per stage. Same arithmetic; the
    /// marks synchronize the device, so a profiled decode is slower than the
    /// decode it describes.
    fn forward_profiled(&self, xs: &Tensor, prof: &Profiler) -> Result<Tensor> {
        let mut h = xs.apply(&self.conv_in)?;
        prof.mark("conv_in");
        h = self.mid_block.forward_profiled(&h, prof)?;
        for (i, block) in self.up_blocks.iter().enumerate() {
            let labels = UP_LABELS
                .get(i)
                .copied()
                .unwrap_or(("up*.resnets", "up*.upsample"));
            h = block.forward_profiled(&h, prof, labels)?;
        }
        let out = self.tail(&h)?;
        prof.mark("norm_out+conv_out");
        Ok(out)
    }
}

impl Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.apply(&self.conv_in)?.apply(&self.mid_block)?;
        for block in &self.up_blocks {
            h = h.apply(block)?;
        }
        self.tail(&h)
    }
}

// ==================== DiagonalGaussian ====================

/// Diagonal Gaussian distribution sampling (VAE reparameterization trick)
#[derive(Debug, Clone)]
pub struct DiagonalGaussian {
    sample: bool,
}

impl DiagonalGaussian {
    pub fn new(sample: bool) -> Self {
        Self { sample }
    }
}

impl Module for DiagonalGaussian {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let chunks = xs.chunk(2, 1)?; // Split along channel dimension
        let mean = &chunks[0];
        let logvar = &chunks[1];

        if self.sample {
            let std = (logvar * 0.5)?.exp()?;
            mean + (std * mean.randn_like(0., 1.)?)?
        } else {
            Ok(mean.clone())
        }
    }
}

// ==================== AutoEncoderKL ====================

/// Z-Image VAE (AutoEncoderKL) - Diffusers Format
#[derive(Debug, Clone)]
pub struct AutoEncoderKL {
    encoder: Encoder,
    decoder: Decoder,
    reg: DiagonalGaussian,
    scale_factor: f64,
    shift_factor: f64,
}

impl AutoEncoderKL {
    /// [`Self::new_with_impl`] on the shipped decoder arm.
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_impl(cfg, vb, VaeImpl::Xwen)
    }

    /// Build the VAE with its decoder on `arm`.
    pub fn new_with_impl(cfg: &VaeConfig, vb: VarBuilder, arm: VaeImpl) -> Result<Self> {
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let decoder = Decoder::new(cfg, arm, vb.pp("decoder"))?;
        let reg = DiagonalGaussian::new(true);

        Ok(Self {
            encoder,
            decoder,
            reg,
            scale_factor: cfg.scaling_factor,
            shift_factor: cfg.shift_factor,
        })
    }

    /// Encode image to latent space
    /// xs: (B, 3, H, W) RGB image, range [-1, 1]
    /// Returns: (B, latent_channels, H/8, W/8)
    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let z = xs.apply(&self.encoder)?.apply(&self.reg)?;
        (z - self.shift_factor)? * self.scale_factor
    }

    /// Decode latent to image
    /// xs: (B, latent_channels, H/8, W/8)
    /// Returns: (B, 3, H, W) RGB image, range [-1, 1]
    pub fn decode(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = ((xs / self.scale_factor)? + self.shift_factor)?;
        xs.apply(&self.decoder)
    }

    /// [`Self::decode`] with one profile row per decoder stage.
    pub fn decode_profiled(&self, xs: &Tensor, prof: &Profiler) -> Result<Tensor> {
        let xs = ((xs / self.scale_factor)? + self.shift_factor)?;
        self.decoder.forward_profiled(&xs, prof)
    }

    /// Get scaling factor
    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    /// Get shift factor
    pub fn shift_factor(&self) -> f64 {
        self.shift_factor
    }
}

impl Module for AutoEncoderKL {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.decode(&self.encode(xs)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vae_impl_parses_its_arms() {
        assert_eq!(VaeImpl::parse("").unwrap(), VaeImpl::Xwen);
        assert_eq!(VaeImpl::parse("xwen").unwrap(), VaeImpl::Xwen);
        assert_eq!(VaeImpl::parse(" Direct ").unwrap(), VaeImpl::Xwen);
        assert_eq!(VaeImpl::parse("candle").unwrap(), VaeImpl::Candle);
        let err = VaeImpl::parse("fast").unwrap_err().to_string();
        assert!(err.contains(VAE_ENV), "{err}");
        assert_eq!(VaeImpl::Xwen.label(), "xwen");
        assert_eq!(VaeImpl::Candle.label(), "candle");
    }

    /// A small decoder built from random weights decodes the same latent on
    /// both arms, so the fused path is graded against the vendored chain on
    /// every block shape the shipped config has, at a size with partial
    /// tiles.
    #[test]
    fn xwen_arm_matches_candle_arm() {
        use candle_core::DType;
        use candle_nn::VarMap;
        let Ok(dev) = crate::gguf::metal_device() else {
            return;
        };
        let cfg = VaeConfig {
            block_out_channels: vec![16, 32, 64, 64],
            norm_num_groups: 8,
            layers_per_block: 1,
            ..VaeConfig::z_image()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let candle_dec = Decoder::new(&cfg, VaeImpl::Candle, vb.pp("decoder")).unwrap();
        // Perturbed weights, so the norms have non-unit weights and non-zero
        // biases and every fold has something to fold; small enough that 14
        // random resnets stay finite.
        for (_, var) in varmap.data().lock().unwrap().iter() {
            let noise = var.randn_like(0.0, 0.02).unwrap();
            var.set(&(var.as_tensor() + noise).unwrap()).unwrap();
        }
        let xwen_dec = Decoder::new(&cfg, VaeImpl::Xwen, vb.pp("decoder")).unwrap();
        let latent = Tensor::randn(0f32, 1.0, (1, 16, 3, 5), &dev).unwrap();
        let want = candle_dec.forward(&latent).unwrap();
        let got = xwen_dec.forward(&latent).unwrap();
        assert_eq!(got.dims(), want.dims());
        assert_eq!(got.dims(), &[1, 3, 24, 40]);
        let norm = want
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            norm.is_finite() && norm > 0.0,
            "the candle arm is not finite: {norm}"
        );
        let diff = (&got - &want)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff.is_finite(), "the xwen arm is not finite");
        let rel = (diff / norm.max(1e-30)).sqrt();
        assert!(rel < 1e-4, "xwen arm against candle arm: rel_l2 {rel:.3e}");
        assert!(
            rel > 0.0,
            "the two arms are different code and should not agree to the bit"
        );
    }
}
