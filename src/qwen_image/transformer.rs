//! The Qwen-Image 2.1 single-stream diffusion transformer, ported from
//! diffusers `transformer_qwenimage21.py` at 6256aa7.
//!
//! One sequence carries everything: the text tokens, each condition image's
//! clean latent tokens dropped in at the slots the vision-language encoder
//! reserved for them, and the target image's noisy latent tokens last. 32
//! identical blocks run over it, all modulated from ONE shared projection on
//! the model. What is particular to this model, and silent when wrong:
//!
//! - **Block-causal attention.** `allowed = (q >= kv) OR same_image_block`.
//!   Text is token-causal, every image block is bidirectional within itself
//!   and sees everything before it, and nothing sees a later block. Here that
//!   is a list of [`Segment`]s and one attention call per segment, not a mask:
//!   a segment's queries read the keys `[0, end)`, and only a text segment
//!   needs a triangle, over its own keys.
//! - **The t = 0 row.** The timestep embedder runs on `[t, 0]`. The target
//!   image's tokens are modulated from the first row, text and condition
//!   tokens from the second, in every block and in the final norm. So the
//!   whole prefix is independent of the denoising step, and its per-block
//!   keys and values are computed once ([`PrefixCache`]); every later step
//!   runs the target rows alone against `[cached prefix ; fresh target]`.
//! - **Rope** is the interleaved-pair rotation over axes (16, 56, 56) at theta
//!   1e4: a text token sits at `(p, p, p)`, an image block freezes the frame
//!   axis at the running position, lays its rows and columns out CENTRED on
//!   zero (`[-(H - H/2), H/2)`, so negative ids), and advances the position by
//!   `max(H, W)`.
//! - **Modulation** is `norm(x) * (1 + scale)` with no shift and the residual
//!   is `h + tanh(gate) * y`; the shared projection's output is chunked
//!   `[scale1, gate1, scale2, gate2]`. The block norms and the final norm are
//!   affine-free LayerNorm. `txt_in` opens with the model's one zero-centred
//!   RMSNorm, whose checkpoint weight is `scale - 1`.
//! - The timestep embedding is `cat([cos, sin])` of `1000 * t * freqs`, and
//!   the output is the velocity as-is: `x += (sigma_next - sigma) * v`.
//!
//! Layout conventions of this port. Batch is 1 and has no axis. A latent image
//! is PACKED, `[H * W, channels]` in raster order ([`pack_latents`]), and so is
//! the returned velocity. The activation stream is f32 over bf16 weights, the
//! contract of [`Projection`]. The prefix and the target are kept as two
//! tensors through the blocks, because each reads one modulation row, which
//! keeps every scale and gate a `[dim]` vector.

use candle_core::{D, DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use crate::zimage::linear::{LinearImpl, Projection, WeightRange, ensure_weights_fit_f16};

/// The environment switch that picks the linear-layer kernel for this model:
/// `xwen` (or unset) is the Metal-4 tensor gemm, `candle` the bisect arm.
pub const LINEAR_ENV: &str = "XWEN_QWEN_IMAGE_LINEAR";

/// The environment switch that picks the attention arm for this model.
pub const ATTN_ENV: &str = "XWEN_QWEN_IMAGE_ATTN";

/// The head width `ops::flash_attn_tensor` is compiled for.
const FLASH_HEAD_DIM: usize = 128;

/// Width of the sinusoidal timestep embedding.
const TIMESTEP_DIM: usize = 256;
/// The factor the model applies to `t` before the sinusoid.
const TIME_FACTOR: f32 = 1000.0;
const TIMESTEP_MAX_PERIOD: f32 = 10000.0;

/// Rope base. A constant of the reference model class, not a config key.
const ROPE_THETA: f32 = 10000.0;
/// The positions the reference's rope table holds per axis: `[0, 8192)` and
/// `[-1024, 0)`. A layout reaching past either is refused, the reference
/// failing there too (by an index error, or silently by wrapping a negative
/// index into the positive rows).
const ROPE_MAX_POSITION: usize = 8192;
const ROPE_MAX_NEGATIVE: usize = 1024;

/// The longest image side, in latent tokens, whose centred positions
/// `[-(side - side/2), side/2)` stay inside the rope.
pub const MAX_IMAGE_SIDE: usize = 2 * ROPE_MAX_NEGATIVE;

/// Latent tokens per vision-language image slot, a 2x2 group.
const TOKENS_PER_SLOT: usize = 4;

/// Which attention computation runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnArm {
    /// The shipped path: one attention call per segment and no mask except a
    /// text segment's own triangle. On Metal every unmasked call is
    /// `ops::flash_attn_tensor`, f32 queries over f16 keys and values;
    /// elsewhere, and for the text triangles, it is the explicit f32 chain.
    Tensor,
    /// The reference arm: one dense f32 attention over the whole sequence
    /// under the explicit boolean mask `(q >= kv) OR same_image_block`. It
    /// shares no segment arithmetic with the shipped arm, so an A/B between
    /// them compares two computations.
    Basic,
}

impl AttnArm {
    pub const SHIPPED: Self = Self::Tensor;

    /// `tensor` / `xwen` (or empty) select the shipped arm, `basic` the
    /// reference; anything else is refused rather than defaulted.
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "tensor" | "xwen" => Ok(Self::Tensor),
            "basic" => Ok(Self::Basic),
            other => candle_core::bail!(
                "{ATTN_ENV}={other:?}: expected `tensor` (the default) or `basic`"
            ),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Tensor => "tensor",
            Self::Basic => "basic",
        }
    }
}

/// The two bisect switches, resolved once at load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arms {
    pub linear: LinearImpl,
    pub attn: AttnArm,
}

impl Arms {
    pub const SHIPPED: Self = Self {
        linear: LinearImpl::Xwen,
        attn: AttnArm::SHIPPED,
    };

    /// Read [`LINEAR_ENV`] and [`ATTN_ENV`]. Unset means shipped; a value that
    /// names no arm is an error, so a typo in a bisect run cannot measure the
    /// wrong arm silently.
    pub fn from_env() -> Result<Self> {
        let read = |name: &str| -> Result<Option<String>> {
            match std::env::var(name) {
                Ok(value) => Ok(Some(value)),
                Err(std::env::VarError::NotPresent) => Ok(None),
                Err(std::env::VarError::NotUnicode(_)) => {
                    candle_core::bail!("{name} is not valid UTF-8")
                }
            }
        };
        let linear = match read(LINEAR_ENV)? {
            None => LinearImpl::Xwen,
            Some(value) => LinearImpl::parse(&value).map_err(|_| {
                candle_core::Error::Msg(format!(
                    "{LINEAR_ENV}={value:?}: expected `xwen` (the default) or `candle`"
                ))
            })?,
        };
        let attn = match read(ATTN_ENV)? {
            None => AttnArm::SHIPPED,
            Some(value) => AttnArm::parse(&value)?,
        };
        Ok(Self { linear, attn })
    }
}

fn default_patch_size() -> usize {
    1
}
fn default_in_channels() -> usize {
    64
}
fn default_num_layers() -> usize {
    32
}
fn default_attention_head_dim() -> usize {
    128
}
fn default_num_attention_heads() -> usize {
    32
}
fn default_context_in_dim() -> usize {
    4096
}
fn default_mlp_ratio() -> usize {
    3
}
fn default_axes_dims_rope() -> Vec<usize> {
    vec![16, 56, 56]
}
fn default_eps() -> f64 {
    1e-6
}
fn default_causal_condition() -> bool {
    true
}

/// `transformer/config.json`. The defaults are the reference class's, which
/// are also what the checkpoint ships.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default)]
    pub out_channels: Option<usize>,
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    #[serde(default = "default_attention_head_dim")]
    pub attention_head_dim: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_context_in_dim")]
    pub context_in_dim: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: usize,
    #[serde(default = "default_axes_dims_rope")]
    pub axes_dims_rope: Vec<usize>,
    #[serde(default = "default_eps")]
    pub eps: f64,
    #[serde(default = "default_causal_condition")]
    pub causal_condition: bool,
}

impl Config {
    /// The shipped Qwen-Image 2.1 configuration.
    pub fn qwen_image_21() -> Self {
        Self {
            patch_size: 1,
            in_channels: 64,
            out_channels: Some(64),
            num_layers: 32,
            attention_head_dim: 128,
            num_attention_heads: 32,
            context_in_dim: 4096,
            mlp_ratio: 3,
            axes_dims_rope: vec![16, 56, 56],
            eps: 1e-6,
            causal_condition: true,
        }
    }

    /// The model width, `heads * head_dim`.
    pub fn dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// The SwiGLU inner width, `mlp_ratio * dim`.
    pub fn hidden_dim(&self) -> usize {
        self.mlp_ratio * self.dim()
    }

    /// Channels the model returns per token; falls back to `in_channels`.
    pub fn out_channels(&self) -> usize {
        self.out_channels.unwrap_or(self.in_channels)
    }

    /// Refuse every value that would change the graph into one this port does
    /// not implement.
    pub fn validate(&self) -> Result<()> {
        if self.patch_size != 1 {
            candle_core::bail!(
                "qwen-image transformer: patch_size {} is not implemented; 2.1 consumes latents \
                 unpatched",
                self.patch_size
            );
        }
        if !self.causal_condition {
            candle_core::bail!(
                "qwen-image transformer: causal_condition is false; this port modulates text and \
                 condition tokens from t = 0, which is also what makes the prefix cacheable"
            );
        }
        if self.num_layers == 0
            || self.num_attention_heads == 0
            || self.attention_head_dim == 0
            || self.in_channels == 0
            || self.out_channels() == 0
            || self.context_in_dim == 0
            || self.mlp_ratio == 0
        {
            candle_core::bail!("qwen-image transformer: a zero dimension in {self:?}");
        }
        if self.axes_dims_rope.len() != 3
            || self.axes_dims_rope.iter().any(|d| d % 2 != 0)
            || self.axes_dims_rope.iter().sum::<usize>() != self.attention_head_dim
        {
            candle_core::bail!(
                "qwen-image transformer: axes_dims_rope {:?} must be three even widths summing \
                 to the head dim {}",
                self.axes_dims_rope,
                self.attention_head_dim
            );
        }
        if self.eps <= 0.0 || self.eps.is_nan() {
            candle_core::bail!("qwen-image transformer: eps {} is not positive", self.eps);
        }
        Ok(())
    }
}

// ==================== The joint sequence ====================

/// One run of the joint sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment {
    /// `len` text tokens, token-causal.
    Text { len: usize },
    /// One image block of `height x width` latent tokens in raster order,
    /// bidirectional within itself. Two adjacent images are two segments and
    /// never see each other bidirectionally.
    Image { height: usize, width: usize },
}

impl Segment {
    pub fn len(&self) -> usize {
        match *self {
            Self::Text { len } => len,
            Self::Image { height, width } => height.saturating_mul(width),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The joint sequence: text runs and condition images in order, the target
/// image last. Text-to-image is `[Text, Image]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    segments: Vec<Segment>,
}

impl Layout {
    /// `segments` in sequence order. The last one is the target image; every
    /// other image is a condition image.
    pub fn new(segments: Vec<Segment>) -> Result<Self> {
        match segments.last() {
            Some(Segment::Image { .. }) => {}
            _ => candle_core::bail!(
                "qwen-image layout: the sequence must end with the target image block"
            ),
        }
        if let Some(i) = segments.iter().position(Segment::is_empty) {
            candle_core::bail!("qwen-image layout: segment {i} is empty");
        }
        Ok(Self { segments })
    }

    /// The text-to-image layout: the prompt, then a `height x width` target.
    pub fn text_to_image(text_len: usize, height: usize, width: usize) -> Result<Self> {
        Self::new(vec![
            Segment::Text { len: text_len },
            Segment::Image { height, width },
        ])
    }

    /// The layout of a vision-language sequence with image slots.
    ///
    /// `slot_mask` is over the encoder's kept tokens, true at an image slot,
    /// each slot standing for a 2x2 group of latent tokens. `images` is every
    /// image's `(height, width)` in latent tokens, condition images first and
    /// the target last.
    ///
    /// `slot_mask` covers the ENCODER's sequence only and must NOT carry slots
    /// for the target: the target is appended here. That differs from the
    /// reference transformer's `img_mask`, to which the pipeline has already
    /// appended `target_tokens / 4` trailing ones; a mask in that form is
    /// refused by name rather than read as one more condition image. Block boundaries come from `images`, not from runs of true:
    /// two condition images with no text between them stay two blocks.
    ///
    /// Also returns the positions in `slot_mask` of the text tokens, which are
    /// the only encoder rows the transformer reads: a slot's own hidden state
    /// is replaced by the image's latents.
    pub fn from_slots(slot_mask: &[bool], images: &[(usize, usize)]) -> Result<(Self, Vec<u32>)> {
        let Some((&(target_h, target_w), conditions)) = images.split_last() else {
            candle_core::bail!("qwen-image layout: no image shapes, not even the target's");
        };
        let slots = slot_mask.iter().filter(|&&s| s).count();
        let mut tokens = 0usize;
        for &(h, w) in conditions {
            if (h * w) % TOKENS_PER_SLOT != 0 {
                candle_core::bail!(
                    "qwen-image layout: a {h}x{w} condition image is not a whole number of \
                     {TOKENS_PER_SLOT}-token slots"
                );
            }
            tokens += h * w;
        }
        let target_slots = target_h * target_w / TOKENS_PER_SLOT;
        if tokens + target_slots * TOKENS_PER_SLOT == slots * TOKENS_PER_SLOT
            && target_slots > 0
            && slot_mask.len() >= target_slots
            && slot_mask[slot_mask.len() - target_slots..]
                .iter()
                .all(|&s| s)
        {
            candle_core::bail!(
                "qwen-image layout: the slot mask ends with {target_slots} slots for the \
                 {target_h}x{target_w} target image; pass the encoder's mask without them, the \
                 target block is appended by the layout"
            );
        }
        if tokens != slots * TOKENS_PER_SLOT {
            candle_core::bail!(
                "qwen-image layout: the image shapes account for {tokens} condition tokens but \
                 the slot mask marks {}",
                slots * TOKENS_PER_SLOT
            );
        }

        let mut segments = Vec::new();
        let mut text_rows = Vec::new();
        let mut conditions = conditions.iter();
        let mut cursor = 0usize;
        let mut text_run = 0usize;
        while cursor < slot_mask.len() {
            if !slot_mask[cursor] {
                text_rows.push(cursor as u32);
                text_run += 1;
                cursor += 1;
                continue;
            }
            if text_run > 0 {
                segments.push(Segment::Text { len: text_run });
                text_run = 0;
            }
            let &(height, width) = conditions
                .next()
                .expect("the slot count was checked against the shapes");
            let block_slots = height * width / TOKENS_PER_SLOT;
            if slot_mask[cursor..].iter().take(block_slots).any(|&s| !s)
                || cursor + block_slots > slot_mask.len()
            {
                candle_core::bail!(
                    "qwen-image layout: the {height}x{width} condition image at token {cursor} \
                     is interrupted by text before its {block_slots} slots end"
                );
            }
            segments.push(Segment::Image { height, width });
            cursor += block_slots;
        }
        if text_run > 0 {
            segments.push(Segment::Text { len: text_run });
        }
        segments.push(Segment::Image {
            height: target_h,
            width: target_w,
        });
        Ok((Self::new(segments)?, text_rows))
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// The target image's `(height, width)` in latent tokens.
    pub fn target(&self) -> (usize, usize) {
        match self.segments.last() {
            Some(&Segment::Image { height, width }) => (height, width),
            _ => unreachable!("Layout::new requires a trailing image"),
        }
    }

    pub fn target_len(&self) -> usize {
        let (h, w) = self.target();
        h * w
    }

    /// Everything before the target: text and condition images.
    pub fn prefix_len(&self) -> usize {
        self.len() - self.target_len()
    }

    pub fn len(&self) -> usize {
        self.segments.iter().map(Segment::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn text_len(&self) -> usize {
        self.segments
            .iter()
            .filter(|s| matches!(s, Segment::Text { .. }))
            .map(Segment::len)
            .sum()
    }

    /// Every image block's `(height, width)`, condition images first.
    pub fn images(&self) -> Vec<(usize, usize)> {
        self.segments
            .iter()
            .filter_map(|s| match *s {
                Segment::Image { height, width } => Some((height, width)),
                Segment::Text { .. } => None,
            })
            .collect()
    }

    /// `(start, end, is_text)` of every prefix segment.
    fn prefix_spans(&self) -> Vec<(usize, usize, bool)> {
        let mut start = 0;
        let mut spans = Vec::new();
        for segment in &self.segments[..self.segments.len() - 1] {
            let end = start + segment.len();
            spans.push((start, end, matches!(segment, Segment::Text { .. })));
            start = end;
        }
        spans
    }

    /// The per-token image block id, `-1` at text, which is what the dense
    /// mask is written in.
    fn block_ids(&self) -> Vec<i64> {
        let mut ids = Vec::with_capacity(self.len());
        let mut block = 0i64;
        for segment in &self.segments {
            match segment {
                Segment::Text { len } => ids.extend(std::iter::repeat_n(-1, *len)),
                Segment::Image { .. } => {
                    ids.extend(std::iter::repeat_n(block, segment.len()));
                    block += 1;
                }
            }
        }
        ids
    }

    /// The `(frame, row, column)` rope position of every token.
    ///
    /// One counter runs through the sequence. A text token takes it on all
    /// three axes and advances it by one. An image block takes it as the frame
    /// of all its tokens, lays rows out over `[-(H - H/2), H/2)` and columns
    /// likewise, and advances it by `max(H, W)`.
    ///
    /// Errors when a position WRITTEN falls outside the reference's rope
    /// table, `[-1024, 8192)` per axis: a text position or an image frame at
    /// or past 8192, or an image side centring below -1024. The counter itself
    /// is not bounded, so a trailing image may advance it past the table as
    /// long as its own frame is inside, which is what the reference accepts.
    ///
    /// Every bound is checked before anything is allocated, so a layout of
    /// absurd extents is an error and never an allocation.
    pub fn positions(&self) -> Result<Vec<[i64; 3]>> {
        let mut total = 0usize;
        for segment in &self.segments {
            if let Segment::Image { height, width } = *segment
                && let Some(side) = [height, width].into_iter().find(|&s| s > MAX_IMAGE_SIDE)
            {
                candle_core::bail!(
                    "qwen-image rope: an image side of {side} latent tokens centres past \
                     -{ROPE_MAX_NEGATIVE}, the rope's lowest position"
                );
            }
            // Bounded sides make an image at most MAX_IMAGE_SIDE squared; a
            // text run past the table is refused below, and here only has to
            // not overflow.
            total = total.saturating_add(segment.len());
        }
        let text: usize = self
            .segments
            .iter()
            .filter(|s| matches!(s, Segment::Text { .. }))
            .fold(0usize, |n, s| n.saturating_add(s.len()));
        if text > ROPE_MAX_POSITION {
            candle_core::bail!(
                "qwen-image rope: {text} text tokens, past the {ROPE_MAX_POSITION} positions the \
                 rope holds"
            );
        }
        let mut out = Vec::with_capacity(total);
        let mut position = 0usize;
        for segment in &self.segments {
            match *segment {
                Segment::Text { len } => {
                    if position + len > ROPE_MAX_POSITION {
                        candle_core::bail!(
                            "qwen-image rope: a text token sits at position {}, past the \
                             {ROPE_MAX_POSITION} the rope holds",
                            position + len - 1
                        );
                    }
                    for p in position..position + len {
                        out.push([p as i64; 3]);
                    }
                    position += len;
                }
                Segment::Image { height, width } => {
                    if position >= ROPE_MAX_POSITION {
                        candle_core::bail!(
                            "qwen-image rope: an image block sits at frame {position}, past the \
                             {ROPE_MAX_POSITION} the rope holds"
                        );
                    }
                    let frame = position as i64;
                    let (top, left) = ((height - height / 2) as i64, (width - width / 2) as i64);
                    for h in 0..height as i64 {
                        for w in 0..width as i64 {
                            out.push([frame, h - top, w - left]);
                        }
                    }
                    position += height.max(width);
                }
            }
        }
        Ok(out)
    }
}

/// The interleaved-pair rope tables `cos`/`sin`, `[len, head_dim / 2]` f32,
/// for a layout: per token the frame axis's `axes[0] / 2` angles, then the
/// row axis's, then the column axis's.
///
/// Built per layout rather than indexed from a cached table, the reference's
/// table being a function of the position alone. The arithmetic is the
/// reference's: the inverse frequency `theta^-(2i / d)` and its product with
/// the position are both taken in f32, which is what decides the angle at a
/// large position; the cosine and sine of that f32 angle are then rounded
/// once from f64.
pub fn rope_tables(layout: &Layout, axes: &[usize], device: &Device) -> Result<(Tensor, Tensor)> {
    let positions = layout.positions()?;
    let inv_freqs: Vec<Vec<f32>> = axes
        .iter()
        .map(|&d| {
            (0..d / 2)
                .map(|i| 1.0 / ROPE_THETA.powf((2 * i) as f32 / d as f32))
                .collect()
        })
        .collect();
    let half: usize = axes.iter().map(|d| d / 2).sum();
    let mut cos = Vec::with_capacity(positions.len() * half);
    let mut sin = Vec::with_capacity(positions.len() * half);
    for position in &positions {
        for (axis, inv_freq) in inv_freqs.iter().enumerate() {
            let p = position[axis] as f32;
            for &f in inv_freq {
                let angle = f64::from(p * f);
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
    }
    Ok((
        Tensor::from_vec(cos, (positions.len(), half), device)?,
        Tensor::from_vec(sin, (positions.len(), half), device)?,
    ))
}

/// Rotate dims `(2i, 2i + 1)` of `x` `[seq, heads, head_dim]` f32 by column
/// `i` of `cos`/`sin` `[seq, head_dim / 2]`: `ops::rope_pair` on Metal, the
/// candle chain it reproduces bit for bit elsewhere.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (seq, heads, head_dim) = x.dims3()?;
    if x.device().is_metal() {
        let x = x.unsqueeze(0)?.contiguous()?;
        return crate::ops::rope_pair(&x, &cos.contiguous()?, &sin.contiguous()?)
            .map_err(|e| candle_core::Error::Msg(format!("qwen-image rope_pair kernel: {e:#}")))?
            .squeeze(0);
    }
    let x = x.reshape((seq, heads, head_dim / 2, 2))?;
    let real = x.i((.., .., .., 0))?;
    let imag = x.i((.., .., .., 1))?;
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;
    let out_real = (real.broadcast_mul(&cos)? - imag.broadcast_mul(&sin)?)?;
    let out_imag = (real.broadcast_mul(&sin)? + imag.broadcast_mul(&cos)?)?;
    Tensor::stack(&[out_real, out_imag], D::Minus1)?.reshape((seq, heads, head_dim))
}

/// `[1, C, H, W]` or `[C, H, W]` to the packed `[H * W, C]`, a raster flatten:
/// 2.1 has no patch folding.
pub fn pack_latents(latents: &Tensor) -> Result<Tensor> {
    let latents = match latents.rank() {
        4 if latents.dim(0)? == 1 => latents.squeeze(0)?,
        3 => latents.clone(),
        _ => candle_core::bail!(
            "qwen-image pack_latents: expected [1, C, H, W] or [C, H, W], got {:?}",
            latents.shape()
        ),
    };
    let (c, h, w) = latents.dims3()?;
    latents.reshape((c, h * w))?.t()?.contiguous()
}

/// The packed `[H * W, C]` back to `[1, C, H, W]`.
pub fn unpack_latents(packed: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    let (n, c) = packed.dims2()?;
    if n != height * width {
        candle_core::bail!(
            "qwen-image unpack_latents: {n} tokens are not a {height}x{width} image"
        );
    }
    packed.t()?.contiguous()?.reshape((1, c, height, width))
}

// ==================== Small numerics ====================

/// Affine-free LayerNorm over the last axis times a per-channel factor,
/// `layer_norm(x) * factor` with `factor` `[dim]`: the modulated norm of every
/// block and of the final layer, `factor` being `1 + scale`.
///
/// The factor rides in as the fused kernel's weight, with a zero bias, so the
/// product is the kernel's own `normed * weight` and no full-tensor multiply
/// follows the norm. A kernel that also folds the norm into the projection's
/// read is a possible later step and would be priced by microbench first.
fn layer_norm_scaled(x: &Tensor, factor: &Tensor, zero_bias: &Tensor, eps: f64) -> Result<Tensor> {
    candle_nn::ops::layer_norm(&x.contiguous()?, factor, zero_bias, eps as f32)
}

/// `h + gate * y` with `gate` `[dim]`: one kernel on Metal, the two candle
/// ops it reproduces bit for bit elsewhere.
fn gated_residual(h: &Tensor, y: &Tensor, gate: &Tensor) -> Result<Tensor> {
    if h.device().is_metal() {
        return crate::ops::gated_residual(&h.contiguous()?, &y.contiguous()?, gate).map_err(|e| {
            candle_core::Error::Msg(format!("qwen-image gated_residual kernel: {e:#}"))
        });
    }
    h + y.broadcast_mul(gate)?
}

/// `silu(gate) * up`, fused on Metal.
fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.device().is_metal() {
        return crate::ops::silu_mul(&gate.contiguous()?, &up.contiguous()?)
            .map_err(|e| candle_core::Error::Msg(format!("qwen-image silu_mul kernel: {e:#}")));
    }
    gate.silu()? * up
}

/// `[seq, heads, d]` to head-major `[heads, seq, d]`, contiguous, in `dtype`
/// (f32 or f16).
fn head_major(x: &Tensor, dtype: DType) -> Result<Tensor> {
    if x.device().is_metal() && x.dtype() == DType::F32 {
        let x = x.contiguous()?;
        let out = match dtype {
            DType::F32 => crate::ops::permute_01(&x),
            DType::F16 => crate::ops::permute_01_f16(&x),
            other => candle_core::bail!("qwen-image head_major: unsupported dtype {other:?}"),
        };
        return out.map_err(|e| candle_core::Error::Msg(format!("qwen-image permute: {e:#}")));
    }
    x.transpose(0, 1)?.contiguous()?.to_dtype(dtype)
}

/// Head-major `[heads, seq, d]` f32 back to `[seq, heads * d]`.
fn token_major(x: &Tensor) -> Result<Tensor> {
    let (heads, seq, d) = x.dims3()?;
    let x = if x.device().is_metal() {
        crate::ops::permute_01(&x.contiguous()?)
            .map_err(|e| candle_core::Error::Msg(format!("qwen-image permute: {e:#}")))?
    } else {
        x.transpose(0, 1)?.contiguous()?
    };
    x.reshape((seq, heads * d))
}

/// The explicit f32 chain over head-major operands: scores, an optional
/// additive `[q, k]` mask, softmax, weighted sum.
fn attend_basic(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let mut scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * scale)?;
    if let Some(mask) = mask {
        scores = scores.broadcast_add(mask)?;
    }
    candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v.contiguous()?)
}

/// An additive f32 mask from `allowed(q, k)`: 0 where allowed, `-inf` where
/// not. Every row a caller builds allows at least its own diagonal.
fn additive_mask(
    q_len: usize,
    k_len: usize,
    device: &Device,
    allowed: impl Fn(usize, usize) -> bool,
) -> Result<Tensor> {
    let mut values = Vec::with_capacity(q_len * k_len);
    for q in 0..q_len {
        for k in 0..k_len {
            values.push(if allowed(q, k) {
                0f32
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    Tensor::from_vec(values, (q_len, k_len), device)
}

/// The sinusoidal embedding of `[t, 0]`, two rows of [`TIMESTEP_DIM`] in row
/// order: `cat([cos(a), sin(a)])` with `a = 1000 * t * exp(-ln(1e4) * i / 128)`,
/// all in f32.
fn timestep_sinusoid(t: f32) -> Vec<f32> {
    let half = TIMESTEP_DIM / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-TIMESTEP_MAX_PERIOD.ln() * i as f32 / half as f32).exp())
        .collect();
    let mut rows = Vec::with_capacity(2 * TIMESTEP_DIM);
    for t in [t, 0.0] {
        let t = TIME_FACTOR * t;
        rows.extend(freqs.iter().map(|f| (t * f).cos()));
        rows.extend(freqs.iter().map(|f| (t * f).sin()));
    }
    rows
}

// ==================== Modules ====================

/// QK-normed, roped self-attention projections. The attention itself is
/// assembled by the model, which knows the layout.
#[derive(Debug, Clone)]
struct Attention {
    to_q: Projection,
    to_k: Projection,
    to_v: Projection,
    to_out: Projection,
    norm_q: Tensor,
    norm_k: Tensor,
    heads: usize,
    head_dim: usize,
    eps: f32,
}

impl Attention {
    fn new(cfg: &Config, arm: LinearImpl, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.dim();
        let head_dim = cfg.attention_head_dim;
        Ok(Self {
            to_q: Projection::new(dim, dim, false, vb.pp("to_q"), arm)?,
            to_k: Projection::new(dim, dim, false, vb.pp("to_k"), arm)?,
            to_v: Projection::new(dim, dim, false, vb.pp("to_v"), arm)?,
            to_out: Projection::new(dim, dim, false, vb.pp("to_out").pp("0"), arm)?,
            norm_q: vb.get(head_dim, "norm_q.weight")?.to_dtype(DType::F32)?,
            norm_k: vb.get(head_dim, "norm_k.weight")?.to_dtype(DType::F32)?,
            heads: cfg.num_attention_heads,
            head_dim,
            eps: cfg.eps as f32,
        })
    }

    fn projections(&self) -> [&Projection; 4] {
        [&self.to_q, &self.to_k, &self.to_v, &self.to_out]
    }

    /// `x` `[seq, dim]` to `(q, k, v)`, each `[seq, heads, head_dim]` f32, q
    /// and k RMS-normed per head and then rotated.
    fn qkv(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let seq = x.dim(0)?;
        let shape = (seq, self.heads, self.head_dim);
        let q = self.to_q.forward(x)?.reshape(shape)?.contiguous()?;
        let k = self.to_k.forward(x)?.reshape(shape)?.contiguous()?;
        let v = self.to_v.forward(x)?.reshape(shape)?;
        let q = candle_nn::ops::rms_norm(&q, &self.norm_q, self.eps)?;
        let k = candle_nn::ops::rms_norm(&k, &self.norm_k, self.eps)?;
        Ok((apply_rope(&q, cos, sin)?, apply_rope(&k, cos, sin)?, v))
    }
}

/// SwiGLU: `out(silu(gate_layer(x)) * proj(x))`. `proj` is the ungated up
/// projection.
#[derive(Debug, Clone)]
struct FeedForward {
    proj: Projection,
    gate_layer: Projection,
    out: Projection,
}

impl FeedForward {
    fn new(cfg: &Config, arm: LinearImpl, vb: VarBuilder) -> Result<Self> {
        let (dim, hidden) = (cfg.dim(), cfg.hidden_dim());
        Ok(Self {
            proj: Projection::new(dim, hidden, false, vb.pp("proj"), arm)?,
            gate_layer: Projection::new(dim, hidden, false, vb.pp("gate_layer"), arm)?,
            out: Projection::new(hidden, dim, false, vb.pp("out"), arm)?,
        })
    }

    fn projections(&self) -> [&Projection; 3] {
        [&self.proj, &self.gate_layer, &self.out]
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.out.forward(&silu_mul(
            &self.gate_layer.forward(x)?,
            &self.proj.forward(x)?,
        )?)
    }
}

#[derive(Debug, Clone)]
struct Block {
    attn: Attention,
    mlp: FeedForward,
}

/// One modulation row cut into what a block reads: the two norm factors
/// `1 + scale` and the two gates `tanh(gate)`, each `[dim]`.
struct RowModulation {
    factor1: Tensor,
    gate1: Tensor,
    factor2: Tensor,
    gate2: Tensor,
}

impl RowModulation {
    /// `row` is `[4 * dim]`, chunked `[scale1, gate1, scale2, gate2]`.
    fn new(row: &Tensor) -> Result<Self> {
        let chunks = row.chunk(4, 0)?;
        Ok(Self {
            factor1: (&chunks[0] + 1.0)?.contiguous()?,
            gate1: chunks[1].tanh()?.contiguous()?,
            factor2: (&chunks[2] + 1.0)?.contiguous()?,
            gate2: chunks[3].tanh()?.contiguous()?,
        })
    }
}

/// A stream through the blocks: its hidden states `[n, dim]`, the rope rows
/// of its tokens and the modulation row they read.
struct Stream<'a> {
    hidden: Tensor,
    cos: &'a Tensor,
    sin: &'a Tensor,
    modulation: &'a RowModulation,
}

/// Which graph a full forward runs. Everything but [`Self::Reference`] is a
/// deliberately wrong model, there so a parity gate can show its bar refuses
/// them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphVariant {
    /// The model as shipped.
    Reference,
    /// Every token attends to every token, the attention pattern of the
    /// dual-stream Qwen-Image 1.0 and of Z-Image.
    FullyBidirectional,
    /// Text and condition tokens modulated from the real `t` instead of the
    /// `t = 0` row.
    RealTimestepForText,
}

/// The per-block keys and values of the prefix, post QK-norm and post rope,
/// head-major `[heads, prefix, head_dim]`, in the dtype the attention arm
/// reads keys in (f16 for the flash kernel, f32 for the explicit chain). It
/// owns its storage, so the prefill's activations are freed after step 0.
///
/// Valid for one prompt, one set of condition images and one target size, at
/// every denoising step: nothing in the prefix depends on `t`.
#[derive(Debug, Clone)]
pub struct PrefixCache {
    layers: Vec<(Tensor, Tensor)>,
    layout: Layout,
    target_cos: Tensor,
    target_sin: Tensor,
}

impl PrefixCache {
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn prefix_len(&self) -> usize {
        self.layout.prefix_len()
    }

    /// Bytes of cached keys and values.
    pub fn size_in_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|(k, v)| {
                k.elem_count() * k.dtype().size_in_bytes()
                    + v.elem_count() * v.dtype().size_in_bytes()
            })
            .sum()
    }
}

/// The Qwen-Image 2.1 transformer.
#[derive(Debug, Clone)]
pub struct QwenImageTransformer {
    cfg: Config,
    arms: Arms,
    time_linear_1: Projection,
    time_linear_2: Projection,
    /// `txt_in.text_norm` as the multiply-ready `1 + w`, f32.
    txt_norm: Tensor,
    txt_in_layer: Projection,
    txt_out_layer: Projection,
    img_in: Projection,
    modulation: Projection,
    blocks: Vec<Block>,
    norm_out_linear: Projection,
    proj_out: Projection,
    /// `[dim]` zeros, the bias the fused LayerNorm takes.
    zero_bias: Tensor,
    weight_range: WeightRange,
}

impl QwenImageTransformer {
    /// Load every tensor of `transformer/` through `vb`, projections as bf16
    /// and the three norm weights as f32, and check the projections against
    /// the tensor gemm's f16 staging.
    pub fn load(cfg: &Config, arms: Arms, vb: VarBuilder) -> Result<Self> {
        cfg.validate()?;
        let device = vb.device().clone();
        if arms.attn == AttnArm::Tensor
            && device.is_metal()
            && cfg.attention_head_dim != FLASH_HEAD_DIM
        {
            candle_core::bail!(
                "{ATTN_ENV}=tensor needs head_dim {FLASH_HEAD_DIM}, this model has {}; run it \
                 with {ATTN_ENV}=basic",
                cfg.attention_head_dim
            );
        }
        let dim = cfg.dim();
        let arm = arms.linear;
        let time = vb.pp("time_text_embed").pp("timestep_embedder");
        let txt = vb.pp("txt_in");
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let vb = vb.pp("transformer_blocks").pp(i);
            blocks.push(Block {
                attn: Attention::new(cfg, arm, vb.pp("attn"))?,
                mlp: FeedForward::new(cfg, arm, vb.pp("img_mlp"))?,
            });
        }
        let mut model = Self {
            cfg: cfg.clone(),
            arms,
            time_linear_1: Projection::new(TIMESTEP_DIM, dim, false, time.pp("linear_1"), arm)?,
            time_linear_2: Projection::new(dim, dim, false, time.pp("linear_2"), arm)?,
            txt_norm: (txt
                .get(cfg.context_in_dim, "text_norm.weight")?
                .to_dtype(DType::F32)?
                + 1.0)?,
            txt_in_layer: Projection::new(cfg.context_in_dim, dim, false, txt.pp("in_layer"), arm)?,
            txt_out_layer: Projection::new(dim, dim, false, txt.pp("out_layer"), arm)?,
            img_in: Projection::new(cfg.in_channels, dim, false, vb.pp("img_in"), arm)?,
            modulation: Projection::new(dim, 4 * dim, false, vb.pp("modulation").pp("1"), arm)?,
            blocks,
            norm_out_linear: Projection::new(dim, dim, false, vb.pp("norm_out").pp("linear"), arm)?,
            proj_out: Projection::new(dim, cfg.out_channels(), false, vb.pp("proj_out"), arm)?,
            zero_bias: Tensor::zeros(dim, DType::F32, &device)?,
            weight_range: WeightRange {
                max_abs: 0.0,
                max_abs_tensor: String::new(),
                total: 0,
            },
        };
        model.weight_range = ensure_weights_fit_f16(model.projections(), &device)?;
        Ok(model)
    }

    /// Every tensor name a checkpoint of this configuration holds, which is
    /// exactly what [`Self::load`] reads.
    pub fn tensor_names(cfg: &Config) -> Vec<String> {
        let mut names: Vec<String> = [
            "img_in.weight",
            "modulation.1.weight",
            "norm_out.linear.weight",
            "proj_out.weight",
            "time_text_embed.timestep_embedder.linear_1.weight",
            "time_text_embed.timestep_embedder.linear_2.weight",
            "txt_in.in_layer.weight",
            "txt_in.out_layer.weight",
            "txt_in.text_norm.weight",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        for i in 0..cfg.num_layers {
            for suffix in [
                "attn.norm_k.weight",
                "attn.norm_q.weight",
                "attn.to_k.weight",
                "attn.to_out.0.weight",
                "attn.to_q.weight",
                "attn.to_v.weight",
                "img_mlp.gate_layer.weight",
                "img_mlp.out.weight",
                "img_mlp.proj.weight",
            ] {
                names.push(format!("transformer_blocks.{i}.{suffix}"));
            }
        }
        names
    }

    fn projections(&self) -> Vec<&Projection> {
        let mut all = vec![
            &self.time_linear_1,
            &self.time_linear_2,
            &self.txt_in_layer,
            &self.txt_out_layer,
            &self.img_in,
            &self.modulation,
            &self.norm_out_linear,
            &self.proj_out,
        ];
        for block in &self.blocks {
            all.extend(block.attn.projections());
            all.extend(block.mlp.projections());
        }
        all
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn arms(&self) -> Arms {
        self.arms
    }

    /// The projections' weight range, as checked at load.
    pub fn weight_range(&self) -> &WeightRange {
        &self.weight_range
    }

    fn device(&self) -> &Device {
        self.zero_bias.device()
    }

    /// Whether unmasked attention runs on the flash kernel, which is also what
    /// decides the dtype keys and values are held in.
    fn uses_flash(&self) -> bool {
        self.arms.attn == AttnArm::Tensor && self.device().is_metal()
    }

    fn kv_dtype(&self) -> DType {
        if self.uses_flash() {
            DType::F16
        } else {
            DType::F32
        }
    }

    fn scale(&self) -> f64 {
        1.0 / (self.cfg.attention_head_dim as f64).sqrt()
    }

    /// The conditioning embedding of `[t, 0]`, `[2, dim]`: row 0 is the real
    /// timestep's and row 1 the `t = 0` row the prefix reads.
    ///
    /// The sinusoid is the reference's in f32: `freqs = exp(-ln(1e4) * i /
    /// 128)`, the argument `1000 * t * freqs`, cosines in the first half.
    fn timestep_embedding(&self, t: f32) -> Result<Tensor> {
        let sinusoid = Tensor::from_vec(timestep_sinusoid(t), (2, TIMESTEP_DIM), self.device())?;
        self.time_linear_2
            .forward(&self.time_linear_1.forward(&sinusoid)?.silu()?)
    }

    /// `txt_in`: the zero-centred RMSNorm, then `Linear, GELU(tanh), Linear`.
    fn embed_text(&self, text: &Tensor) -> Result<Tensor> {
        let text = text.to_dtype(DType::F32)?.contiguous()?;
        let normed = candle_nn::ops::rms_norm(&text, &self.txt_norm, self.cfg.eps as f32)?;
        self.txt_out_layer
            .forward(&self.txt_in_layer.forward(&normed)?.gelu()?)
    }

    fn embed_image(&self, packed: &Tensor) -> Result<Tensor> {
        self.img_in
            .forward(&packed.to_dtype(DType::F32)?.contiguous()?)
    }

    fn check_inputs(&self, layout: &Layout, text: &Tensor, images: &[&Tensor]) -> Result<()> {
        let (rows, width) = text.dims2()?;
        if rows != layout.text_len() || width != self.cfg.context_in_dim {
            candle_core::bail!(
                "qwen-image transformer: text features are {:?}, the layout needs [{}, {}]",
                text.shape(),
                layout.text_len(),
                self.cfg.context_in_dim
            );
        }
        let shapes = layout.images();
        if shapes.len() != images.len() {
            candle_core::bail!(
                "qwen-image transformer: {} latent images for a layout of {}",
                images.len(),
                shapes.len()
            );
        }
        for (i, (&(h, w), image)) in shapes.iter().zip(images).enumerate() {
            if image.dims() != [h * w, self.cfg.in_channels] {
                candle_core::bail!(
                    "qwen-image transformer: latent image {i} is {:?}, the layout needs a packed \
                     [{}, {}]",
                    image.shape(),
                    h * w,
                    self.cfg.in_channels
                );
            }
        }
        Ok(())
    }

    /// The embedded prefix `[prefix, dim]`, text runs and condition images in
    /// layout order, or `None` when the layout has no prefix.
    fn embed_prefix(
        &self,
        layout: &Layout,
        text: &Tensor,
        conditions: &[&Tensor],
    ) -> Result<Option<Tensor>> {
        let text = self.embed_text(text)?;
        let mut parts = Vec::new();
        let mut text_at = 0;
        let mut conditions = conditions.iter();
        for segment in &layout.segments()[..layout.segments().len() - 1] {
            match segment {
                Segment::Text { len } => {
                    parts.push(text.narrow(0, text_at, *len)?);
                    text_at += len;
                }
                Segment::Image { .. } => {
                    let image = conditions.next().expect("check_inputs counted the images");
                    parts.push(self.embed_image(image)?);
                }
            }
        }
        if parts.is_empty() {
            return Ok(None);
        }
        Ok(Some(Tensor::cat(&parts, 0)?))
    }

    /// Step 0 under the prefix cache: the whole sequence runs, and every
    /// block's prefix keys and values are kept.
    ///
    /// `text` is `[layout.text_len(), context_in_dim]`, the encoder's rows at
    /// the text positions in order. `images` are the packed latents of every
    /// image block, clean condition images first and the noisy target last.
    /// `t` is the sigma of the step, in `[0, 1]`. Returns the target's
    /// velocity, packed `[target_len, out_channels]` f32.
    pub fn forward_prefill(
        &self,
        layout: &Layout,
        text: &Tensor,
        images: &[&Tensor],
        t: f32,
    ) -> Result<(Tensor, PrefixCache)> {
        let (velocity, cache) =
            self.forward_joint(layout, text, images, t, GraphVariant::Reference, true)?;
        Ok((
            velocity,
            cache.expect("forward_joint was asked for the cache"),
        ))
    }

    /// The same step with nothing kept: the no-cache arm, and the entry the
    /// wrong-graph variants run through.
    pub fn forward_full(
        &self,
        layout: &Layout,
        text: &Tensor,
        images: &[&Tensor],
        t: f32,
        variant: GraphVariant,
    ) -> Result<Tensor> {
        Ok(self
            .forward_joint(layout, text, images, t, variant, false)?
            .0)
    }

    /// Every step after the first: the target's tokens alone, against the
    /// cached prefix. `target` is the packed noisy latent, `[target_len,
    /// in_channels]`.
    pub fn forward_cached(&self, cache: &PrefixCache, target: &Tensor, t: f32) -> Result<Tensor> {
        if cache.layers.len() != self.blocks.len() {
            candle_core::bail!(
                "qwen-image transformer: a prefix cache of {} blocks for a model of {}",
                cache.layers.len(),
                self.blocks.len()
            );
        }
        let n = cache.layout.target_len();
        if target.dims() != [n, self.cfg.in_channels] {
            candle_core::bail!(
                "qwen-image transformer: the target latent is {:?}, the cached layout needs a \
                 packed [{n}, {}]",
                target.shape(),
                self.cfg.in_channels
            );
        }
        let temb = self.timestep_embedding(t)?;
        let real = RowModulation::new(&self.modulation.forward(&temb.silu()?)?.i(0)?)?;
        let mut hidden = self.embed_image(target)?;
        let kv_dtype = self.kv_dtype();
        for (block, (cached_k, cached_v)) in self.blocks.iter().zip(&cache.layers) {
            let normed = layer_norm_scaled(&hidden, &real.factor1, &self.zero_bias, self.cfg.eps)?;
            let (q, k, v) = block
                .attn
                .qkv(&normed, &cache.target_cos, &cache.target_sin)?;
            let q = head_major(&q, DType::F32)?;
            let k = Tensor::cat(&[cached_k, &head_major(&k, kv_dtype)?], 1)?;
            let v = Tensor::cat(&[cached_v, &head_major(&v, kv_dtype)?], 1)?;
            let attn = block
                .attn
                .to_out
                .forward(&token_major(&self.attend(&q, &k, &v)?)?)?;
            hidden = gated_residual(&hidden, &attn, &real.gate1)?;
            let normed = layer_norm_scaled(&hidden, &real.factor2, &self.zero_bias, self.cfg.eps)?;
            hidden = gated_residual(&hidden, &block.mlp.forward(&normed)?, &real.gate2)?;
        }
        self.final_layer(&hidden, &temb)
    }

    /// Unmasked attention over head-major operands, keys and values in
    /// [`Self::kv_dtype`].
    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        if self.uses_flash() {
            return crate::ops::flash_attn_tensor(q, k, v, self.scale() as f32).map_err(|e| {
                candle_core::Error::Msg(format!("qwen-image flash attention: {e:#}"))
            });
        }
        attend_basic(q, k, v, None, self.scale())
    }

    /// `norm_out` and `proj_out` over the target's rows, which are the only
    /// rows of the output anything reads.
    fn final_layer(&self, target: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let scale = self.norm_out_linear.forward(&temb.silu()?)?.i(0)?;
        let factor = (scale + 1.0)?.contiguous()?;
        let normed = layer_norm_scaled(target, &factor, &self.zero_bias, self.cfg.eps)?;
        self.proj_out.forward(&normed)
    }

    fn forward_joint(
        &self,
        layout: &Layout,
        text: &Tensor,
        images: &[&Tensor],
        t: f32,
        variant: GraphVariant,
        keep_cache: bool,
    ) -> Result<(Tensor, Option<PrefixCache>)> {
        self.check_inputs(layout, text, images)?;
        let (cos, sin) = rope_tables(layout, &self.cfg.axes_dims_rope, self.device())?;
        let prefix_len = layout.prefix_len();
        let target_len = layout.target_len();
        let target_cos = cos.narrow(0, prefix_len, target_len)?.contiguous()?;
        let target_sin = sin.narrow(0, prefix_len, target_len)?.contiguous()?;
        let prefix_cos = cos.narrow(0, 0, prefix_len)?.contiguous()?;
        let prefix_sin = sin.narrow(0, 0, prefix_len)?.contiguous()?;

        let temb = self.timestep_embedding(t)?;
        let modulation = self.modulation.forward(&temb.silu()?)?;
        let real = RowModulation::new(&modulation.i(0)?)?;
        let zero = RowModulation::new(&modulation.i(1)?)?;
        let prefix_modulation = match variant {
            GraphVariant::RealTimestepForText => &real,
            _ => &zero,
        };

        let (conditions, target) = images.split_at(images.len() - 1);
        let mut streams = Vec::with_capacity(2);
        if let Some(prefix) = self.embed_prefix(layout, text, conditions)? {
            streams.push(Stream {
                hidden: prefix,
                cos: &prefix_cos,
                sin: &prefix_sin,
                modulation: prefix_modulation,
            });
        }
        streams.push(Stream {
            hidden: self.embed_image(target[0])?,
            cos: &target_cos,
            sin: &target_sin,
            modulation: &real,
        });

        let dense_mask = match (self.arms.attn, variant) {
            (_, GraphVariant::FullyBidirectional) => None,
            (AttnArm::Tensor, _) => None,
            (AttnArm::Basic, _) => {
                let ids = layout.block_ids();
                Some(additive_mask(
                    ids.len(),
                    ids.len(),
                    self.device(),
                    |q, k| q >= k || (ids[q] >= 0 && ids[q] == ids[k]),
                )?)
            }
        };

        let mut layers = Vec::with_capacity(if keep_cache { self.blocks.len() } else { 0 });
        for block in &self.blocks {
            let mut qs = Vec::with_capacity(2);
            let mut ks = Vec::with_capacity(2);
            let mut vs = Vec::with_capacity(2);
            for stream in &streams {
                let normed = layer_norm_scaled(
                    &stream.hidden,
                    &stream.modulation.factor1,
                    &self.zero_bias,
                    self.cfg.eps,
                )?;
                let (q, k, v) = block.attn.qkv(&normed, stream.cos, stream.sin)?;
                qs.push(q);
                ks.push(k);
                vs.push(v);
            }
            let q = Tensor::cat(&qs, 0)?;
            let k = Tensor::cat(&ks, 0)?;
            let v = Tensor::cat(&vs, 0)?;
            let (attn, kept) =
                self.attend_joint(layout, &q, &k, &v, variant, dense_mask.as_ref(), keep_cache)?;
            if let Some(kept) = kept {
                layers.push(kept);
            }
            let attn = block.attn.to_out.forward(&attn)?;

            let mut at = 0;
            for stream in &mut streams {
                let n = stream.hidden.dim(0)?;
                let m = stream.modulation;
                let h = gated_residual(&stream.hidden, &attn.narrow(0, at, n)?, &m.gate1)?;
                let normed = layer_norm_scaled(&h, &m.factor2, &self.zero_bias, self.cfg.eps)?;
                stream.hidden = gated_residual(&h, &block.mlp.forward(&normed)?, &m.gate2)?;
                at += n;
            }
        }

        let target = &streams.last().expect("the target stream").hidden;
        let velocity = self.final_layer(target, &temb)?;
        let cache = keep_cache.then(|| PrefixCache {
            layers,
            layout: layout.clone(),
            target_cos,
            target_sin,
        });
        Ok((velocity, cache))
    }

    /// Attention over the whole joint sequence. `q`, `k`, `v` are
    /// `[len, heads, head_dim]` f32; the result is `[len, dim]`. With
    /// `keep_cache`, also the prefix's head-major keys and values as owned
    /// copies in [`Self::kv_dtype`].
    #[allow(clippy::too_many_arguments)]
    fn attend_joint(
        &self,
        layout: &Layout,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        variant: GraphVariant,
        dense_mask: Option<&Tensor>,
        keep_cache: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let prefix_len = layout.prefix_len();
        let kv_dtype = self.kv_dtype();
        let q = head_major(q, DType::F32)?;
        let k_kernel = head_major(k, kv_dtype)?;
        let v_kernel = head_major(v, kv_dtype)?;
        // `narrow` on the sequence axis of a one-image batch-free tensor is a
        // view of the whole, and a cache holding a view would pin the
        // prefill's keys for the whole denoising loop: copy.
        let kept = if keep_cache {
            Some((
                k_kernel.narrow(1, 0, prefix_len)?.copy()?,
                v_kernel.narrow(1, 0, prefix_len)?.copy()?,
            ))
        } else {
            None
        };

        let bidirectional = variant == GraphVariant::FullyBidirectional;
        let out = if bidirectional {
            self.attend(&q, &k_kernel, &v_kernel)?
        } else if self.arms.attn == AttnArm::Basic {
            attend_basic(&q, &k_kernel, &v_kernel, dense_mask, self.scale())?
        } else {
            // One call per segment, its queries over the keys `[0, end)`. A
            // text segment adds the triangle over its own keys and runs the
            // f32 chain on f32 keys, whatever the kernel arm holds them in.
            let mut outputs = Vec::with_capacity(layout.segments().len());
            let text_keys = if kv_dtype == DType::F32 {
                None
            } else {
                let text_end = layout
                    .prefix_spans()
                    .iter()
                    .filter(|span| span.2)
                    .map(|span| span.1)
                    .max()
                    .unwrap_or(0);
                Some((
                    head_major(&k.narrow(0, 0, text_end)?, DType::F32)?,
                    head_major(&v.narrow(0, 0, text_end)?, DType::F32)?,
                ))
            };
            for (start, end, is_text) in layout.prefix_spans() {
                let q_seg = q.narrow(1, start, end - start)?.contiguous()?;
                if is_text {
                    let (k32, v32) = match &text_keys {
                        Some((k32, v32)) => (k32, v32),
                        None => (&k_kernel, &v_kernel),
                    };
                    let mask =
                        additive_mask(end - start, end, self.device(), |q, k| k <= start + q)?;
                    outputs.push(attend_basic(
                        &q_seg,
                        &k32.narrow(1, 0, end)?,
                        &v32.narrow(1, 0, end)?,
                        Some(&mask),
                        self.scale(),
                    )?);
                } else {
                    outputs.push(self.attend(
                        &q_seg,
                        &k_kernel.narrow(1, 0, end)?,
                        &v_kernel.narrow(1, 0, end)?,
                    )?);
                }
            }
            let q_target = q.narrow(1, prefix_len, layout.target_len())?.contiguous()?;
            outputs.push(self.attend(&q_target, &k_kernel, &v_kernel)?);
            Tensor::cat(&outputs, 1)?
        };
        Ok((token_major(&out)?, kept))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Deterministic pseudo-random weights for any name and shape, seeded
    /// from the tensor's NAME so two builders hand the same tensor the same
    /// numbers, and recording every name asked for.
    struct RandomWeights {
        asked: Arc<Mutex<Vec<String>>>,
    }

    impl candle_nn::var_builder::SimpleBackend for RandomWeights {
        fn get(
            &self,
            s: candle_core::Shape,
            name: &str,
            _init: candle_nn::Init,
            dtype: DType,
            dev: &Device,
        ) -> Result<Tensor> {
            self.asked.lock().unwrap().push(name.to_string());
            let mut state = name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
            }) | 1;
            let values: Vec<f32> = (0..s.elem_count())
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                    (u as f32 - 0.5) * 0.4
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

    fn random_vb(dev: &Device) -> (VarBuilder<'static>, Arc<Mutex<Vec<String>>>) {
        let asked = Arc::new(Mutex::new(Vec::new()));
        let backend = RandomWeights {
            asked: asked.clone(),
        };
        (
            VarBuilder::from_backend(Box::new(backend), DType::F32, dev.clone()),
            asked,
        )
    }

    /// Two heads of 16 with the axis split (4, 6, 6), two blocks.
    pub(crate) fn tiny_config() -> Config {
        Config {
            patch_size: 1,
            in_channels: 8,
            out_channels: Some(8),
            num_layers: 2,
            attention_head_dim: 16,
            num_attention_heads: 2,
            context_in_dim: 12,
            mlp_ratio: 3,
            axes_dims_rope: vec![4, 6, 6],
            eps: 1e-6,
            causal_condition: true,
        }
    }

    fn tiny_model(attn: AttnArm) -> QwenImageTransformer {
        tiny_model_of(&tiny_config(), attn)
    }

    /// The random tiny model at another geometry, for the pipeline tests.
    pub(crate) fn tiny_model_of(cfg: &Config, attn: AttnArm) -> QwenImageTransformer {
        let arms = Arms {
            linear: LinearImpl::Candle,
            attn,
        };
        QwenImageTransformer::load(cfg, arms, random_vb(&Device::Cpu).0).unwrap()
    }

    /// Deterministic inputs in a plausible range.
    fn ramp(rows: usize, cols: usize, phase: f32) -> Tensor {
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.37 + phase).sin())
            .collect();
        Tensor::from_vec(values, (rows, cols), &Device::Cpu).unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        // `f32::max` drops a NaN operand, so a nonfinite output would reduce to
        // an exact match.
        assert!(
            a.iter().chain(&b).all(|x| x.is_finite()),
            "a compared tensor holds a nonfinite value"
        );
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// The text-to-image layout and one with two condition images, the second
    /// following the first with no text between them.
    fn layouts() -> Vec<Layout> {
        vec![
            Layout::text_to_image(5, 4, 6).unwrap(),
            Layout::new(vec![
                Segment::Text { len: 3 },
                Segment::Image {
                    height: 2,
                    width: 4,
                },
                Segment::Image {
                    height: 4,
                    width: 2,
                },
                Segment::Text { len: 4 },
                Segment::Image {
                    height: 4,
                    width: 4,
                },
            ])
            .unwrap(),
        ]
    }

    fn inputs(layout: &Layout, cfg: &Config) -> (Tensor, Vec<Tensor>) {
        let text = ramp(layout.text_len(), cfg.context_in_dim, 0.1);
        let images = layout
            .images()
            .iter()
            .enumerate()
            .map(|(i, &(h, w))| ramp(h * w, cfg.in_channels, 1.0 + i as f32))
            .collect();
        (text, images)
    }

    #[test]
    fn the_segmented_attention_equals_the_dense_block_causal_mask() {
        let segmented = tiny_model(AttnArm::Tensor);
        let dense = tiny_model(AttnArm::Basic);
        for layout in layouts() {
            let (text, images) = inputs(&layout, segmented.config());
            let images: Vec<&Tensor> = images.iter().collect();
            let a = segmented
                .forward_full(&layout, &text, &images, 0.7, GraphVariant::Reference)
                .unwrap();
            let b = dense
                .forward_full(&layout, &text, &images, 0.7, GraphVariant::Reference)
                .unwrap();
            assert_eq!(a.dims(), [layout.target_len(), 8]);
            let diff = max_abs_diff(&a, &b);
            assert!(
                diff < 1e-4,
                "{layout:?}: segmented vs dense differ by {diff}"
            );
        }
    }

    #[test]
    fn a_cached_step_equals_the_full_forward() {
        for arm in [AttnArm::Tensor, AttnArm::Basic] {
            let model = tiny_model(arm);
            for layout in layouts() {
                let (text, images) = inputs(&layout, model.config());
                let refs: Vec<&Tensor> = images.iter().collect();
                let (first, cache) = model.forward_prefill(&layout, &text, &refs, 1.0).unwrap();
                let full_first = model
                    .forward_full(&layout, &text, &refs, 1.0, GraphVariant::Reference)
                    .unwrap();
                assert_eq!(max_abs_diff(&first, &full_first), 0.0);
                assert_eq!(cache.prefix_len(), layout.prefix_len());

                // A later step: another target latent at another sigma.
                let (h, w) = layout.target();
                let target = ramp(h * w, 8, 9.0);
                let mut refs = refs.clone();
                *refs.last_mut().unwrap() = &target;
                let cached = model.forward_cached(&cache, &target, 0.4).unwrap();
                let full = model
                    .forward_full(&layout, &text, &refs, 0.4, GraphVariant::Reference)
                    .unwrap();
                let diff = max_abs_diff(&cached, &full);
                assert!(
                    diff < 1e-4,
                    "{arm:?} {layout:?}: cached vs full differ by {diff}"
                );
            }
        }
    }

    #[test]
    fn the_wrong_graphs_are_different_models() {
        let model = tiny_model(AttnArm::Tensor);
        let layout = layouts().remove(1);
        let (text, images) = inputs(&layout, model.config());
        let refs: Vec<&Tensor> = images.iter().collect();
        let run = |variant| {
            model
                .forward_full(&layout, &text, &refs, 0.7, variant)
                .unwrap()
        };
        let reference = run(GraphVariant::Reference);
        for variant in [
            GraphVariant::FullyBidirectional,
            GraphVariant::RealTimestepForText,
        ] {
            let diff = max_abs_diff(&reference, &run(variant));
            assert!(diff > 1e-4, "{variant:?} changed nothing: {diff}");
        }
        // And the fully bidirectional graph is the same on both arms, so the
        // bracket does not depend on which one a gate runs.
        let dense = tiny_model(AttnArm::Basic);
        let a = run(GraphVariant::FullyBidirectional);
        let b = dense
            .forward_full(&layout, &text, &refs, 0.7, GraphVariant::FullyBidirectional)
            .unwrap();
        assert!(max_abs_diff(&a, &b) < 1e-4);
    }

    #[test]
    fn the_target_does_not_reach_back_into_the_prefix() {
        // Block-causal means the prefix never reads the target, so the cache
        // taken under one target latent serves another.
        let model = tiny_model(AttnArm::Tensor);
        let layout = layouts().remove(1);
        let (text, images) = inputs(&layout, model.config());
        let refs: Vec<&Tensor> = images.iter().collect();
        let (_, cache_a) = model.forward_prefill(&layout, &text, &refs, 1.0).unwrap();
        let (h, w) = layout.target();
        let other = ramp(h * w, 8, 5.0);
        let mut refs_b = refs.clone();
        *refs_b.last_mut().unwrap() = &other;
        let (_, cache_b) = model.forward_prefill(&layout, &text, &refs_b, 0.3).unwrap();
        for ((ka, va), (kb, vb)) in cache_a.layers.iter().zip(&cache_b.layers) {
            assert_eq!(max_abs_diff(ka, kb), 0.0);
            assert_eq!(max_abs_diff(va, vb), 0.0);
        }
    }

    #[test]
    fn rope_positions_follow_the_shared_counter_and_centre_the_images() {
        let layout = Layout::new(vec![
            Segment::Text { len: 2 },
            Segment::Image {
                height: 3,
                width: 2,
            },
            Segment::Text { len: 1 },
            Segment::Image {
                height: 2,
                width: 2,
            },
        ])
        .unwrap();
        let p = layout.positions().unwrap();
        assert_eq!(&p[..2], &[[0, 0, 0], [1, 1, 1]]);
        // Frame 2; rows over [-(3 - 1), 1) = -2..=0, columns over -1..=0.
        assert_eq!(
            &p[2..8],
            &[
                [2, -2, -1],
                [2, -2, 0],
                [2, -1, -1],
                [2, -1, 0],
                [2, 0, -1],
                [2, 0, 0]
            ]
        );
        // The counter advanced by max(3, 2), so the text resumes at 5.
        assert_eq!(p[8], [5, 5, 5]);
        assert_eq!(&p[9..], &[[6, -1, -1], [6, -1, 0], [6, 0, -1], [6, 0, 0]]);
    }

    #[test]
    fn rope_tables_hold_each_axis_angles_in_order() {
        let layout = Layout::text_to_image(1, 2, 2).unwrap();
        let (cos, sin) = rope_tables(&layout, &[4, 6, 6], &Device::Cpu).unwrap();
        assert_eq!(cos.dims(), [5, 8]);
        let cos = cos.to_vec2::<f32>().unwrap();
        let sin = sin.to_vec2::<f32>().unwrap();
        // Token 0 is text at position 0: no rotation on any axis.
        assert!(cos[0].iter().all(|&c| c == 1.0) && sin[0].iter().all(|&s| s == 0.0));
        // Token 1 is the image's top-left: frame 1, row -1, column -1. The
        // first slot of each axis has inverse frequency 1.
        let one = 1f64;
        assert_eq!(cos[1][0], one.cos() as f32);
        assert_eq!(sin[1][0], one.sin() as f32);
        assert_eq!(sin[1][2], (-one).sin() as f32);
        assert_eq!(sin[1][5], (-one).sin() as f32);
        // The second frame slot: theta^-(2/4).
        let f = 1.0 / ROPE_THETA.powf(0.5);
        assert_eq!(cos[1][1], f64::from(f).cos() as f32);
    }

    #[test]
    fn rope_refuses_a_layout_past_the_table() {
        // Extents no allocation could hold are refused before one is tried.
        for (text, h, w) in [
            (1, usize::MAX, usize::MAX),
            (usize::MAX, 2, 2),
            (1, 2, usize::MAX),
        ] {
            assert!(
                Layout::text_to_image(text, h, w)
                    .unwrap()
                    .positions()
                    .is_err()
            );
        }
        let tall = Layout::text_to_image(1, 2 * ROPE_MAX_NEGATIVE + 2, 1).unwrap();
        assert!(
            tall.positions()
                .unwrap_err()
                .to_string()
                .contains("centres")
        );
        let long = Layout::text_to_image(ROPE_MAX_POSITION + 1, 2, 2).unwrap();
        assert!(
            long.positions()
                .unwrap_err()
                .to_string()
                .contains("position")
        );
        Layout::text_to_image(10, 2 * ROPE_MAX_NEGATIVE, 2)
            .unwrap()
            .positions()
            .unwrap();
    }

    /// The bound is on the positions written, not on the counter: a block
    /// whose frame is inside the table is accepted even though it advances the
    /// counter past the end.
    #[test]
    fn rope_bounds_the_frames_written_and_not_the_counter() {
        let wide = Segment::Image {
            height: 2,
            width: 2 * ROPE_MAX_NEGATIVE,
        };
        let with_conditions = |n: usize| {
            let mut segments = vec![Segment::Text { len: 1 }];
            segments.extend(std::iter::repeat_n(wide, n + 1));
            Layout::new(segments).unwrap()
        };
        // Frames 1, 2049, 4097 and the target at 6145; the counter ends at 8193.
        let p = with_conditions(3).positions().unwrap();
        assert_eq!(p.last().unwrap()[0], 6145);
        // One more block puts the target at frame 8193.
        let err = with_conditions(4).positions().unwrap_err().to_string();
        assert!(err.contains("frame 8193"), "{err}");

        // The last frame the table holds, and the first it does not.
        let last = Layout::text_to_image(ROPE_MAX_POSITION - 1, 2, 2).unwrap();
        assert_eq!(last.positions().unwrap().last().unwrap()[0], 8191);
        let past = Layout::text_to_image(ROPE_MAX_POSITION, 2, 2).unwrap();
        assert!(
            past.positions()
                .unwrap_err()
                .to_string()
                .contains("frame 8192")
        );
    }

    #[test]
    fn rope_rotates_interleaved_pairs() {
        let x = Tensor::new(&[[[1f32, 0.0, 0.0, 2.0]]], &Device::Cpu).unwrap();
        let quarter = std::f32::consts::FRAC_PI_2;
        let cos = Tensor::new(&[[quarter.cos(), 1.0]], &Device::Cpu).unwrap();
        let sin = Tensor::new(&[[quarter.sin(), 0.0]], &Device::Cpu).unwrap();
        let y = apply_rope(&x, &cos, &sin).unwrap().flatten_all().unwrap();
        let y = y.to_vec1::<f32>().unwrap();
        // (1 + 0i) turned a quarter is i; the second pair is untouched.
        assert!(y[0].abs() < 1e-6 && (y[1] - 1.0).abs() < 1e-6, "{y:?}");
        assert_eq!(&y[2..], &[0.0, 2.0]);
    }

    #[test]
    fn a_slot_mask_becomes_segments_with_adjacent_images_kept_apart() {
        // text, image A (2 slots = 8 tokens), image B (1 slot), text.
        let mask = [false, false, true, true, true, false];
        let (layout, text_rows) = Layout::from_slots(&mask, &[(2, 4), (2, 2), (4, 4)]).unwrap();
        assert_eq!(
            layout.segments(),
            &[
                Segment::Text { len: 2 },
                Segment::Image {
                    height: 2,
                    width: 4
                },
                Segment::Image {
                    height: 2,
                    width: 2
                },
                Segment::Text { len: 1 },
                Segment::Image {
                    height: 4,
                    width: 4
                },
            ]
        );
        assert_eq!(text_rows, vec![0, 1, 5]);
        assert_eq!(layout.prefix_len(), 2 + 8 + 4 + 1);
        assert_eq!(
            layout.block_ids()[2..14],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        );

        // The reference's `img_mask` form, the target's four slots appended.
        let mut with_target = mask.to_vec();
        with_target.extend([true; 4]);
        let err = Layout::from_slots(&with_target, &[(2, 4), (2, 2), (4, 4)]).unwrap_err();
        assert!(err.to_string().contains("without them"), "{err}");

        let err = Layout::from_slots(&mask, &[(2, 4), (4, 4)]).unwrap_err();
        assert!(err.to_string().contains("account for 8"), "{err}");
        // No slots at all is text-to-image.
        let (t2i, rows) = Layout::from_slots(&[false; 3], &[(4, 4)]).unwrap();
        assert_eq!(t2i, Layout::text_to_image(3, 4, 4).unwrap());
        assert_eq!(rows, vec![0, 1, 2]);
    }

    #[test]
    fn packing_is_a_raster_flatten_and_round_trips() {
        let values: Vec<f32> = (0..2 * 2 * 3).map(|i| i as f32).collect();
        let latents = Tensor::from_vec(values, (1, 2, 2, 3), &Device::Cpu).unwrap();
        let packed = pack_latents(&latents).unwrap();
        assert_eq!(packed.dims(), [6, 2]);
        // Token 4 is row 1, column 1: channel 0 holds 4, channel 1 holds 10.
        assert_eq!(
            packed.i(4).unwrap().to_vec1::<f32>().unwrap(),
            vec![4.0, 10.0]
        );
        let back = unpack_latents(&packed, 2, 3).unwrap();
        assert_eq!(max_abs_diff(&back, &latents), 0.0);
    }

    #[test]
    fn the_text_norm_weight_is_stored_zero_centred() {
        let model = tiny_model(AttnArm::Basic);
        let stored = random_vb(&Device::Cpu)
            .0
            .pp("txt_in")
            .get(12, "text_norm.weight")
            .unwrap();
        // The checkpoint holds `scale - 1`, so the norm multiplies by one more
        // than what is stored.
        let scale = (&stored + 1.0).unwrap();
        assert_eq!(max_abs_diff(&model.txt_norm, &scale), 0.0);

        let x = ramp(3, 12, 0.3);
        let rrms = (x.sqr().unwrap().mean_keepdim(1).unwrap() + 1e-6)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let want = x
            .broadcast_mul(&rrms)
            .unwrap()
            .broadcast_mul(&scale)
            .unwrap();
        let got = candle_nn::ops::rms_norm(&x, &model.txt_norm, 1e-6).unwrap();
        assert!(max_abs_diff(&got, &want) < 1e-6);
    }

    /// Values worked out from the reference formula at sigma 0.5, so the
    /// argument of slot 0 is 500 and of slot 1 is `500 * 1e4^(-1/128)`.
    #[test]
    fn the_timestep_sinusoid_matches_the_reference_formula() {
        let rows = timestep_sinusoid(0.5);
        assert_eq!(rows.len(), 512);
        // An f32 argument near 500 is good to about 3e-5.
        let close = |slot: usize, want: f32| {
            let got = rows[slot];
            assert!((got - want).abs() < 1e-4, "slot {slot}: {got} vs {want}");
        };
        close(0, -0.883_849_3); // cos(500)
        close(1, 0.945_942_6); // cos(465.286...)
        close(127, 0.998_556_9); // cos(500 * 1e4^(-127/128))
        close(128, -0.467_771_8); // sin(500)
        close(129, 0.324_334_1); // sin(465.286...)
        close(255, 0.053_704_5);
        // The t = 0 row: every cosine 1, every sine 0.
        assert!(rows[256..384].iter().all(|&c| c == 1.0));
        assert!(rows[384..].iter().all(|&s| s == 0.0));
    }

    #[test]
    fn the_timestep_embedding_puts_cosines_first_and_carries_the_zero_row() {
        let model = tiny_model(AttnArm::Basic);
        // The second row is t = 0 whatever t is, so it is the same for any t.
        let a = model.timestep_embedding(0.9).unwrap();
        let b = model.timestep_embedding(0.2).unwrap();
        assert_eq!(a.dims(), [2, 32]);
        assert_eq!(max_abs_diff(&a.i(1).unwrap(), &b.i(1).unwrap()), 0.0);
        assert!(max_abs_diff(&a.i(0).unwrap(), &b.i(0).unwrap()) > 0.0);
    }

    #[test]
    fn unsupported_configs_and_inputs_are_refused() {
        let refused = |edit: fn(&mut Config), needle: &str| {
            let mut cfg = tiny_config();
            edit(&mut cfg);
            let err = cfg.validate().unwrap_err().to_string();
            assert!(err.contains(needle), "{err}");
        };
        refused(|c| c.patch_size = 2, "patch_size");
        refused(|c| c.causal_condition = false, "causal_condition");
        refused(|c| c.axes_dims_rope = vec![4, 6, 4], "axes_dims_rope");
        Config::qwen_image_21().validate().unwrap();
        assert_eq!(Config::qwen_image_21().dim(), 4096);
        assert_eq!(Config::qwen_image_21().hidden_dim(), 12288);

        assert!(Layout::new(vec![Segment::Text { len: 3 }]).is_err());
        assert!(Layout::text_to_image(0, 2, 2).is_err());

        let model = tiny_model(AttnArm::Basic);
        let layout = Layout::text_to_image(3, 2, 2).unwrap();
        let text = ramp(4, 12, 0.0);
        let image = ramp(4, 8, 0.0);
        let err = model
            .forward_full(&layout, &text, &[&image], 0.5, GraphVariant::Reference)
            .unwrap_err();
        assert!(err.to_string().contains("text features"), "{err}");

        assert_eq!(AttnArm::parse(" Tensor ").unwrap(), AttnArm::Tensor);
        assert_eq!(AttnArm::parse("basic").unwrap(), AttnArm::Basic);
        assert!(AttnArm::parse("flash").is_err());
    }

    #[test]
    fn the_loader_reads_exactly_the_names_it_declares() {
        let cfg = tiny_config();
        let (vb, asked) = random_vb(&Device::Cpu);
        QwenImageTransformer::load(&cfg, Arms::SHIPPED, vb).unwrap();
        let mut asked = asked.lock().unwrap().clone();
        asked.sort();
        let mut declared = QwenImageTransformer::tensor_names(&cfg);
        declared.sort();
        assert_eq!(asked, declared);
    }

    /// The shipped checkpoint's index names exactly the tensors the loader
    /// reads, and its config is the one this port implements.
    #[test]
    fn the_shipped_index_names_exactly_the_loaders_tensors() {
        const REPO: &str = "Qwen/Qwen-Image-2.1";
        const FETCH: &str = "xwen fetch --model qwen-image-2.1";
        let Some(index) = crate::test_support::repo_file_or_skip(
            REPO,
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            FETCH,
        ) else {
            return;
        };
        let Some(config) =
            crate::test_support::repo_file_or_skip(REPO, "transformer/config.json", FETCH)
        else {
            return;
        };
        let cfg: Config = serde_json::from_slice(&std::fs::read(config).unwrap()).unwrap();
        cfg.validate().unwrap();
        assert_eq!((cfg.dim(), cfg.num_layers), (4096, 32));

        let index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(index).unwrap()).unwrap();
        let mut shipped: Vec<String> = index["weight_map"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        shipped.sort();
        let mut declared = QwenImageTransformer::tensor_names(&cfg);
        declared.sort();
        assert_eq!(shipped, declared);
    }

    #[test]
    fn the_flash_arm_agrees_with_the_explicit_chain_on_metal() {
        // No skip: like the ops tests this needs a Metal device, and a machine
        // without one fails here rather than reporting a pass.
        let dev = crate::gguf::metal_device().unwrap();
        // The kernel's head width and the tensor gemm's contract, at two heads.
        let cfg = Config {
            in_channels: 32,
            out_channels: Some(32),
            num_layers: 2,
            attention_head_dim: 128,
            num_attention_heads: 2,
            context_in_dim: 64,
            axes_dims_rope: vec![16, 56, 56],
            ..tiny_config()
        };
        let build = |arms| QwenImageTransformer::load(&cfg, arms, random_vb(&dev).0).unwrap();
        let shipped = build(Arms::SHIPPED);
        let reference = build(Arms {
            linear: LinearImpl::Xwen,
            attn: AttnArm::Basic,
        });
        let layout = Layout::new(vec![
            Segment::Text { len: 7 },
            Segment::Image {
                height: 4,
                width: 4,
            },
            Segment::Text { len: 3 },
            Segment::Image {
                height: 8,
                width: 8,
            },
        ])
        .unwrap();
        let text = ramp(layout.text_len(), 64, 0.1).to_device(&dev).unwrap();
        let images: Vec<Tensor> = layout
            .images()
            .iter()
            .map(|&(h, w)| ramp(h * w, 32, 2.0).to_device(&dev).unwrap())
            .collect();
        let refs: Vec<&Tensor> = images.iter().collect();

        let (first, cache) = shipped.forward_prefill(&layout, &text, &refs, 0.8).unwrap();
        let want = reference
            .forward_full(&layout, &text, &refs, 0.8, GraphVariant::Reference)
            .unwrap();
        let to_cpu = |t: &Tensor| t.to_device(&Device::Cpu).unwrap();
        let scale = to_cpu(&want)
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let diff = max_abs_diff(&to_cpu(&first), &to_cpu(&want));
        // f16 keys and values through the flash kernel against the f32 chain:
        // close, and provably not the same computation.
        assert!(
            diff > 0.0 && diff < 5e-3 * scale,
            "prefill: {diff} against a scale of {scale}"
        );

        let cached = shipped.forward_cached(&cache, refs[1], 0.8).unwrap();
        let diff = max_abs_diff(&to_cpu(&cached), &to_cpu(&first));
        assert!(
            diff < 1e-3 * scale,
            "cached vs prefill: {diff} against a scale of {scale}"
        );
    }
}
