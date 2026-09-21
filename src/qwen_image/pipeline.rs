//! The Qwen-Image 2.1 text-to-image pipeline: conditioning in, pixels out.
//!
//! What diffusers' `QwenImage21Pipeline.__call__` does after `encode_prompt`,
//! at batch 1 and without guidance (`true_cfg_scale` 1.0, the release's
//! default): draw the latent noise unscaled, run the transformer once per
//! scheduler step with the prefix keys and values kept from step 0
//! (`use_kv_cache=True`), Euler-step the packed latent in f32, decode with the
//! VAE in f32, and quantize to bytes. The conditioning tensor comes from
//! [`crate::XwenModel::encode_tap`] at depth 36 before the final norm, with
//! the system turn's rows dropped, and is not produced here.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::scheduler::{DynamicShiftScheduler, SchedulerConfig};
use super::transformer::{
    Arms, Config, GraphVariant, Layout, MAX_IMAGE_SIDE, QwenImageTransformer, pack_latents,
    unpack_latents,
};
use super::vae::{QwenImageVae, VaeConfig, VaeImpl};
use crate::zimage::linear::LinearImpl;
use crate::zimage::sampling::{postprocess_image, seeded_noise};

pub use crate::zimage::pipeline::{Timings, encode_png, write_png};

/// The VAE compresses 16x and the transformer reads latents unpatched, so one
/// image token is a 16-pixel cell.
pub const PIXELS_PER_TOKEN: usize = 16;

/// The pipeline's divisibility rule. The reference floors each side to
/// `2 * (side // 32)` latent cells before anything else, so a side that is not
/// a multiple of 32 px renders a smaller image than was asked for; here it is
/// refused instead.
pub const SIZE_MULTIPLE: usize = 32;

/// The step count the release samples with.
pub const DEFAULT_STEPS: usize = 40;

/// Selects the step arm: unset, `on` or `prefix` keeps the prefix keys and
/// values from step 0, which is what ships; `off` or `full` runs the whole
/// joint sequence at every step. The two are the same mathematics and differ
/// by rounding alone, so `off` attributes a difference to the cache and is
/// not a wrong-graph bracket.
pub const CACHE_ENV: &str = "XWEN_QWEN_IMAGE_CACHE";

/// Whether later steps reuse step 0's prefix keys and values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheArm {
    Prefix,
    Full,
}

impl CacheArm {
    pub const SHIPPED: Self = Self::Prefix;

    pub fn from_env() -> Result<Self> {
        match std::env::var(CACHE_ENV) {
            Ok(value) => Self::parse(&value),
            Err(std::env::VarError::NotPresent) => Ok(Self::SHIPPED),
            Err(std::env::VarError::NotUnicode(_)) => bail!("{CACHE_ENV} is not unicode"),
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "" | "on" | "prefix" => Ok(Self::Prefix),
            "off" | "full" => Ok(Self::Full),
            other => bail!("{CACHE_ENV}={other:?}: expected `on` (the default) or `off`"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Prefix => "prefix",
            Self::Full => "full",
        }
    }
}

/// What one image run is asked for.
#[derive(Debug, Clone)]
pub struct ImageOptions {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// The noise seed. Meaningless when `latents` is given.
    pub seed: u64,
    /// The initial latent, `(1, 64, height/16, width/16)` f32, in place of the
    /// seeded draw: the way a reference comparison makes both sides start
    /// from the same noise.
    pub latents: Option<Tensor>,
}

/// What one run produced: the image, and the two intermediate tensors the
/// parity gate grades.
#[derive(Debug, Clone)]
pub struct Rendered {
    /// u8 on the CPU: `[3, H, W]` RGB, or `[4, H, W]` RGBA when the decoded
    /// alpha plane holds a transparent region ([`keeps_alpha`]).
    pub image: Tensor,
    /// How far the decoded alpha plane strays from opaque, as the largest
    /// `1 - alpha` over the image with alpha in `[0, 1]`. A text-to-image run
    /// without the transparency prompt is expected to read near zero.
    pub alpha_max_distance_from_opaque: f32,
    /// The lowest alpha byte of the decoded image, 255 being opaque.
    pub alpha_min: u8,
    /// How many pixels are clear: alpha at or under [`CLEAR_ALPHA_MAX`].
    pub clear_pixels: usize,
    pub timings: Timings,
    /// The transformer's output at the first sigma, `(1, 64, H/16, W/16)`
    /// f32, as the Euler step consumes it. None when no step runs.
    pub velocity0: Option<Tensor>,
    /// The latent after the last step, same shape, f32, in the normalised
    /// space the transformer works in: the VAE's `z * std + mean` is applied
    /// inside the decode.
    pub final_latents: Tensor,
}

/// A decoded image split into what is written and what is reported.
#[derive(Debug, Clone)]
pub struct Decoded {
    /// u8 on the CPU, `[3, H, W]` RGB or `[4, H, W]` RGBA ([`keeps_alpha`]).
    pub image: Tensor,
    pub alpha_max_distance_from_opaque: f32,
    pub alpha_min: u8,
    pub clear_pixels: usize,
}

impl Decoded {
    /// The RGB planes alone, `[3, H, W]`, whichever form the image took: what
    /// a comparison against an RGB reference reads.
    pub fn rgb(&self) -> Result<Tensor> {
        Ok(self.image.narrow(0, 0, 3)?.contiguous()?)
    }
}

impl Rendered {
    /// The RGB planes alone, `[3, H, W]`, whichever form the image took.
    pub fn rgb(&self) -> Result<Tensor> {
        Ok(self.image.narrow(0, 0, 3)?.contiguous()?)
    }
}

/// An alpha byte at or under this is a clear pixel.
pub const CLEAR_ALPHA_MAX: u8 = 8;

/// This many clear pixels make a transparent region, and an image with one
/// keeps its alpha plane.
pub const CLEAR_PIXELS_MIN: usize = 10;

/// Whether an image with this alpha plane is written as RGBA: at least
/// [`CLEAR_PIXELS_MIN`] pixels at or under [`CLEAR_ALPHA_MAX`]. Everything else
/// is written as RGB with the plane dropped.
///
/// The decoder always emits an alpha plane, and on a prompt that asks for no
/// transparency it is only NEARLY opaque: the reference's fp32 render of an
/// ordinary prompt reads 252 to 255, with about 17% of its pixels at 254. So
/// "any pixel short of opaque" would fire on every image and hand every caller
/// a faintly translucent PNG. A clear pixel is one the model drew as
/// background, which no amount of that noise produces, and asking for ten of
/// them keeps a stray one from deciding the format. An image that is
/// translucent throughout and clear nowhere is written as RGB.
pub fn keeps_alpha(alpha: &[u8]) -> bool {
    clear_pixels(alpha) >= CLEAR_PIXELS_MIN
}

fn clear_pixels(alpha: &[u8]) -> usize {
    alpha.iter().filter(|&&a| a <= CLEAR_ALPHA_MAX).count()
}

/// The transformer, the VAE and the scheduler config, resident on one device.
pub struct QwenImagePipeline {
    transformer: QwenImageTransformer,
    vae: QwenImageVae,
    scheduler_cfg: SchedulerConfig,
    cache: CacheArm,
    device: Device,
}

/// The check the plain entry points run under: a render nothing external can
/// interrupt.
fn never_cancelled() -> Result<()> {
    Ok(())
}

impl QwenImagePipeline {
    /// Open the pipeline at a repo snapshot root: the directory holding
    /// `model_index.json`, `transformer/`, `vae/` and `scheduler/`.
    pub fn load(root: &Path, device: &Device) -> Result<Self> {
        Self::load_cancellable(root, device, &never_cancelled)
    }

    /// The transformer ships bf16 and its projections stay bf16 on the device;
    /// the three norm weights among its tensors are widened to f32 as they are
    /// read. The `VarBuilder` is f32 because that is the dtype those small
    /// tensors want; a projection asks for its plane in bf16 itself, so
    /// nothing the size of the model is ever held in f32. The VAE ships f32
    /// and decodes in f32.
    pub fn load_cancellable(
        root: &Path,
        device: &Device,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        check()?;
        // Before anything opens: a typo in a bisect switch is a load error,
        // not a run that quietly measures the shipped arm twice.
        let arms = Arms::from_env()?;
        let vae_arm = VaeImpl::from_env()?;
        let cache = CacheArm::from_env()?;
        if arms.attn != Arms::SHIPPED.attn {
            eprintln!("xwen: qwen-image attention arm: {}", arms.attn.label());
        }
        if arms.linear != LinearImpl::Xwen {
            eprintln!("xwen: qwen-image linear arm: {}", arms.linear.label());
        }
        if cache != CacheArm::SHIPPED {
            eprintln!("xwen: qwen-image step arm: {}", cache.label());
        }

        // An unsupported scheduler config fails here, before 15 GB of weights
        // are resident.
        let scheduler_cfg: SchedulerConfig =
            read_json(&root.join("scheduler").join("scheduler_config.json"))?;
        DynamicShiftScheduler::new(scheduler_cfg.clone())?;

        let transformer_dir = root.join("transformer");
        let transformer_cfg: Config = read_json(&transformer_dir.join("config.json"))?;
        let vae_dir = root.join("vae");
        let vae_cfg: VaeConfig = read_json(&vae_dir.join("config.json"))?;
        vae_cfg.validate()?;
        ensure!(
            vae_cfg.z_dim == transformer_cfg.in_channels,
            "the VAE's {} latent channels are not the transformer's {}",
            vae_cfg.z_dim,
            transformer_cfg.in_channels
        );
        ensure!(
            vae_cfg.spatial_factor() == PIXELS_PER_TOKEN,
            "the VAE compresses {}x and this pipeline is built for {PIXELS_PER_TOKEN}x",
            vae_cfg.spatial_factor()
        );

        let shards = shard_paths(&transformer_dir)?;
        check()?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, DType::F32, device)? };
        let transformer = QwenImageTransformer::load(&transformer_cfg, arms, vb)
            .context("building the Qwen-Image transformer")?;
        let range = transformer.weight_range();
        eprintln!(
            "xwen: qwen-image projections: max |w| {:.4} in {} over {} values, inside f16's range",
            range.max_abs, range.max_abs_tensor, range.total
        );
        check()?;

        let vae_file = vae_dir.join("diffusion_pytorch_model.safetensors");
        ensure!(vae_file.is_file(), "missing {}", vae_file.display());
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&vae_file], DType::F32, device)? };
        let vae = QwenImageVae::load_with_impl(vb, &vae_cfg, vae_arm)
            .context("building the Qwen-Image VAE")?;
        check()?;

        Ok(Self::from_parts(
            transformer,
            vae,
            scheduler_cfg,
            cache,
            device,
        ))
    }

    /// A pipeline over parts already built, which is how a test runs the loop
    /// on a model small enough for the CPU.
    pub fn from_parts(
        transformer: QwenImageTransformer,
        vae: QwenImageVae,
        scheduler_cfg: SchedulerConfig,
        cache: CacheArm,
        device: &Device,
    ) -> Self {
        Self {
            transformer,
            vae,
            scheduler_cfg,
            cache,
            device: device.clone(),
        }
    }

    /// Whether a `width x height` is one this pipeline runs, and why not.
    ///
    /// Two rules. Both sides are positive multiples of [`SIZE_MULTIPLE`]; and
    /// the layout's rope positions exist, which the transformer's own
    /// [`Layout::positions`] decides and which caps a side at 32768 px. There
    /// is no token-count rule: 2.1 has no pad tokens. The second rule is
    /// checked again in the forward against the real caption length.
    pub fn check_size(width: usize, height: usize) -> Result<()> {
        ensure!(
            width > 0
                && height > 0
                && width.is_multiple_of(SIZE_MULTIPLE)
                && height.is_multiple_of(SIZE_MULTIPLE),
            "image size {width}x{height}: both sides must be positive multiples of \
             {SIZE_MULTIPLE}"
        );
        // In closed form first, so an absurd size is refused by arithmetic
        // and never reaches a layout; the layout's own check stays the
        // authority for what the rope holds.
        let max_side = MAX_IMAGE_SIDE * PIXELS_PER_TOKEN;
        ensure!(
            width <= max_side && height <= max_side,
            "image size {width}x{height}: a side is at most {max_side} px, the extent of the \
             transformer's rope"
        );
        let (lat_h, lat_w) = Self::latent_size(width, height);
        Layout::text_to_image(1, lat_h, lat_w)
            .and_then(|layout| layout.positions())
            .with_context(|| format!("image size {width}x{height}"))?;
        Ok(())
    }

    /// Whether a caption of `text_len` rows and a `width x height` image fit
    /// the rope together, by the rule the forward enforces
    /// ([`Layout::positions`]) and before anything is loaded: the text takes the
    /// first positions and the image's frame comes after them, so a caption can
    /// be too long for an image that alone would fit.
    pub fn check_layout(text_len: usize, width: usize, height: usize) -> Result<()> {
        Self::check_size(width, height)?;
        ensure!(text_len > 0, "the caption has no tokens");
        let (lat_h, lat_w) = Self::latent_size(width, height);
        Layout::text_to_image(text_len, lat_h, lat_w)?.positions()?;
        Ok(())
    }

    /// `(latent_h, latent_w)` for a `width x height` that passed
    /// [`Self::check_size`].
    pub fn latent_size(width: usize, height: usize) -> (usize, usize) {
        (height / PIXELS_PER_TOKEN, width / PIXELS_PER_TOKEN)
    }

    /// The image tokens of a `width x height` run, which is what the
    /// scheduler's shift is computed from.
    pub fn target_tokens(width: usize, height: usize) -> usize {
        let (h, w) = Self::latent_size(width, height);
        h * w
    }

    /// The bytes one run holds at its peak, the encoder excluded (it is gone
    /// before the transformer loads): `PEAK_BASE + PEAK_PER_MEGAPIXEL * pixels`,
    /// a line through two measured runs with 15% on top.
    ///
    /// Measured 2026-09-21 as the kernel's `phys_footprint_peak` of `xwen
    /// image`, 40 steps, the VAE on the candle arm: 25 GiB at 512x512
    /// ([`MEASURED_PEAK_512`]) and 56 GiB at 1024x1024
    /// ([`MEASURED_PEAK_1024`]). Both peaks are the VAE DECODE, not the
    /// denoising loop, which held 19 and 27 GiB: candle's convolution builds a
    /// column buffer nine times its input, 10.9 GB for one 288-channel
    /// 1024x1024 layer, and the buffer pool rounds that up. The decode already
    /// returns its buffers between layers, so what is left is one layer's
    /// worth, and it scales with pixels. A decoder that forms no column buffer
    /// moves both constants, which is when they are measured again.
    pub fn peak_bytes(width: usize, height: usize) -> Result<u64> {
        Self::check_size(width, height)?;
        let pixels = (width as u64)
            .checked_mul(height as u64)
            .context("image pixel count overflow")?;
        pixels
            .checked_mul(PEAK_PER_MEGAPIXEL)
            .map(|scaled| scaled >> 20)
            .and_then(|scaled| scaled.checked_add(PEAK_BASE))
            .context("image memory estimate overflow")
    }

    /// Read an injected latent from a safetensors file holding it under
    /// `latents`, `(1, 64, height/16, width/16)` or the same without the batch
    /// axis, and check it fits a `width x height` run, as f32 on the CPU.
    /// Called before anything loads, so a wrong path or a latent from another
    /// resolution costs nothing.
    pub fn read_latents(path: &Path, width: usize, height: usize) -> Result<Tensor> {
        Self::check_size(width, height)?;
        let latents = read_tensor(path, "latents")?;
        let latents = if latents.rank() == 3 {
            latents.unsqueeze(0)?
        } else {
            latents
        };
        let (lat_h, lat_w) = Self::latent_size(width, height);
        let want = (1, LATENT_CHANNELS, lat_h, lat_w);
        let got = latents
            .dims4()
            .with_context(|| format!("the latents in {} are not 3- or 4-axis", path.display()))?;
        ensure!(
            got == want,
            "the latents in {} are {:?}, and {width}x{height} needs {:?}",
            path.display(),
            latents.dims(),
            want
        );
        latents
            .to_dtype(DType::F32)
            .with_context(|| format!("casting the latents in {} to f32", path.display()))
    }

    /// Read injected caption features, the encoder's kept `[T, 4096]` rows,
    /// from a safetensors file holding them under `cap_feats`, in whatever
    /// float dtype they were saved in.
    pub fn read_cap_feats(path: &Path) -> Result<Tensor> {
        let cap = read_tensor(path, "cap_feats")?;
        let (t, dim) = cap
            .dims2()
            .with_context(|| format!("the caption in {} is not [T, dim]", path.display()))?;
        ensure!(t > 0, "the caption in {} has no tokens", path.display());
        ensure!(
            dim == CAPTION_DIM,
            "the caption in {} is {dim} wide, and this model reads {CAPTION_DIM}",
            path.display()
        );
        ensure!(
            cap.dtype().is_float(),
            "the caption in {} is {:?}, not a float dtype",
            path.display(),
            cap.dtype()
        );
        Ok(cap)
    }

    /// Generate one image from `cap_feats`, the text encoder's kept
    /// `[T, 4096]` rows (any float dtype, any device).
    pub fn generate(&self, cap_feats: &Tensor, opts: &ImageOptions) -> Result<Rendered> {
        self.generate_cancellable(cap_feats, opts, &never_cancelled)
    }

    pub fn generate_cancellable(
        &self,
        cap_feats: &Tensor,
        opts: &ImageOptions,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Rendered> {
        check()?;
        Self::check_size(opts.width, opts.height)?;
        ensure!(opts.steps > 0, "an image needs at least one step");
        let cap_feats = self.caption(cap_feats)?;
        let channels = self.transformer.config().in_channels;
        let (lat_h, lat_w) = Self::latent_size(opts.width, opts.height);
        let shape = (1, channels, lat_h, lat_w);
        // The reference draws standard normal noise and does not scale it.
        let noise = match &opts.latents {
            Some(latents) => {
                ensure!(
                    latents.dims4()? == shape,
                    "the injected latents are {:?}, and {}x{} needs {:?}",
                    latents.dims(),
                    opts.width,
                    opts.height,
                    shape
                );
                latents.to_device(&self.device)?.to_dtype(DType::F32)?
            }
            None => seeded_noise(opts.seed, shape, &self.device)?,
        };
        let layout = self.layout(&cap_feats, lat_h, lat_w)?;
        let mut scheduler = DynamicShiftScheduler::new(self.scheduler_cfg.clone())?;
        scheduler.set_timesteps(opts.steps, layout.target_len())?;

        // The latent is stepped packed, `[N, 64]`, the form the transformer
        // reads and returns.
        let mut latents = pack_latents(&noise)?;
        let mut timings = Timings::default();
        let mut velocity0 = None;
        let mut prefix = None;
        for step in 0..opts.steps {
            check()?;
            let started = Instant::now();
            let sigma = scheduler.current_sigma();
            let velocity = match (self.cache, &prefix) {
                (CacheArm::Prefix, Some(cache)) => {
                    self.transformer.forward_cached(cache, &latents, sigma)?
                }
                (CacheArm::Prefix, None) => {
                    let (velocity, cache) = self.transformer.forward_prefill(
                        &layout,
                        &cap_feats,
                        &[&latents],
                        sigma,
                    )?;
                    prefix = Some(cache);
                    velocity
                }
                (CacheArm::Full, _) => self.transformer.forward_full(
                    &layout,
                    &cap_feats,
                    &[&latents],
                    sigma,
                    GraphVariant::Reference,
                )?,
            };
            if step == 0 {
                velocity0 = Some(unpack_latents(&velocity, lat_h, lat_w)?);
            }
            latents = scheduler.step(&velocity, &latents)?;
            // The step's arithmetic is asynchronous on Metal; reading one
            // element back is what makes the timing mean anything.
            let _ = latents.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            timings.steps.push(started.elapsed().as_secs_f64());
        }
        drop(prefix);
        check()?;

        let final_latents = unpack_latents(&latents, lat_h, lat_w)?;
        let started = Instant::now();
        let decoded = self.decode_latents(&final_latents)?;
        timings.vae_decode = started.elapsed().as_secs_f64();
        check()?;
        Ok(Rendered {
            image: decoded.image,
            alpha_max_distance_from_opaque: decoded.alpha_max_distance_from_opaque,
            alpha_min: decoded.alpha_min,
            clear_pixels: decoded.clear_pixels,
            timings,
            velocity0,
            final_latents,
        })
    }

    /// One forward at `sigma` with nothing kept: the transformer's velocity
    /// for `latents` `(1, 64, h, w)`, same shape, f32. The parity gate's entry,
    /// and the one the wrong-graph variants run through.
    pub fn velocity(
        &self,
        cap_feats: &Tensor,
        latents: &Tensor,
        sigma: f32,
        variant: GraphVariant,
    ) -> Result<Tensor> {
        let cap_feats = self.caption(cap_feats)?;
        let (_, _, lat_h, lat_w) = latents.dims4()?;
        let layout = self.layout(&cap_feats, lat_h, lat_w)?;
        let packed = pack_latents(&latents.to_device(&self.device)?.to_dtype(DType::F32)?)?;
        let velocity =
            self.transformer
                .forward_full(&layout, &cap_feats, &[&packed], sigma, variant)?;
        Ok(unpack_latents(&velocity, lat_h, lat_w)?)
    }

    /// The sigma grid a `steps`-step run of `width x height` walks, trailing
    /// zero included, and the shift `mu` it was built from.
    pub fn sigmas(&self, steps: usize, width: usize, height: usize) -> Result<(Vec<f32>, f64)> {
        Self::check_size(width, height)?;
        let tokens = Self::target_tokens(width, height);
        let mut scheduler = DynamicShiftScheduler::new(self.scheduler_cfg.clone())?;
        scheduler.set_timesteps(steps, tokens)?;
        Ok((scheduler.sigmas().to_vec(), scheduler.mu(tokens)))
    }

    /// Decode a normalised latent `(1, 64, h, w)`: the tail of
    /// [`Self::generate`], public so a reference latent can be decoded through
    /// this VAE alone. The VAE applies `z * std + mean` itself.
    pub fn decode_latents(&self, latents: &Tensor) -> Result<Decoded> {
        let latents = latents.to_device(&self.device)?.to_dtype(DType::F32)?;
        let rgba = self.vae.decode(&latents)?;
        split_rgba(&rgba)
    }

    pub fn transformer_config(&self) -> &Config {
        self.transformer.config()
    }

    pub fn cache_arm(&self) -> CacheArm {
        self.cache
    }

    /// The caption on the device in f32, checked against the transformer.
    fn caption(&self, cap_feats: &Tensor) -> Result<Tensor> {
        let (t, dim) = cap_feats
            .dims2()
            .context("the caption features are not [T, dim]")?;
        ensure!(t > 0, "the caption has no tokens");
        let want = self.transformer.config().context_in_dim;
        ensure!(
            dim == want,
            "the caption features are {dim} wide, and the transformer reads {want}"
        );
        Ok(cap_feats.to_device(&self.device)?.to_dtype(DType::F32)?)
    }

    /// The joint sequence of one run, built here and nowhere else: the
    /// caption's rows, then the target image. A run with condition images
    /// builds its layout from the encoder's slot mask instead
    /// ([`Layout::from_slots`]) and hands the same loop more image blocks.
    fn layout(&self, cap_feats: &Tensor, lat_h: usize, lat_w: usize) -> Result<Layout> {
        Ok(Layout::text_to_image(cap_feats.dim(0)?, lat_h, lat_w)?)
    }
}

const GIB: u64 = 1 << 30;

/// The measured peaks [`QwenImagePipeline::peak_bytes`] is fitted to.
pub const MEASURED_PEAK_512: u64 = 25 * GIB;
pub const MEASURED_PEAK_1024: u64 = 56 * GIB;

/// The fit through those two points is 14.7 GiB plus 41.3 GiB per megapixel
/// (1,048,576 pixels); these are that line with 15% on top, rounded up.
const PEAK_BASE: u64 = 17 * GIB;
const PEAK_PER_MEGAPIXEL: u64 = 48 * GIB;

/// The latent channels and the caption width of the shipped checkpoint, which
/// the file readers check against before any config is open. The loaded
/// configs stay the authority for what runs.
const LATENT_CHANNELS: usize = 64;
const CAPTION_DIM: usize = 4096;

/// The decoder's `[1, 4, H, W]` RGBA in `[-1, 1]` as u8 on the CPU, RGB or RGBA
/// as [`keeps_alpha`] decides over the quantised alpha plane, plus what that
/// plane measured. A decoder with no alpha plane is RGB and opaque by
/// definition.
fn split_rgba(image: &Tensor) -> Result<Decoded> {
    let (b, c, _, _) = image.dims4()?;
    ensure!(
        b == 1 && (c == 3 || c == 4),
        "the VAE decoded {:?}, not one RGB or RGBA image",
        image.shape()
    );
    let bytes = postprocess_image(image)?;
    let bytes = bytes.squeeze(0)?.to_device(&Device::Cpu)?;
    let (alpha_min, clear, keep) = if c == 4 {
        let alpha = bytes.narrow(0, 3, 1)?.flatten_all()?.to_vec1::<u8>()?;
        (
            alpha.iter().copied().min().unwrap_or(u8::MAX),
            clear_pixels(&alpha),
            keeps_alpha(&alpha),
        )
    } else {
        (u8::MAX, 0, false)
    };
    let written = if keep { bytes } else { bytes.narrow(0, 0, 3)? };
    let alpha_max_distance_from_opaque = if c == 4 {
        let alpha = ((image.narrow(1, 3, 1)?.to_dtype(DType::F32)? / 2.0)? + 0.5)?;
        let lowest = alpha.flatten_all()?.min(0)?.to_scalar::<f32>()?;
        (1.0 - lowest).max(0.0)
    } else {
        0.0
    };
    Ok(Decoded {
        image: written.contiguous()?,
        alpha_max_distance_from_opaque,
        alpha_min,
        clear_pixels: clear,
    })
}

fn read_tensor(path: &Path, key: &str) -> Result<Tensor> {
    let tensors = candle_core::safetensors::load(path, &Device::Cpu)
        .with_context(|| format!("reading {}", path.display()))?;
    tensors.get(key).cloned().with_context(|| {
        format!(
            "{} has no `{key}` tensor; it holds {}",
            path.display(),
            if tensors.is_empty() {
                "nothing".to_string()
            } else {
                tensors.keys().cloned().collect::<Vec<_>>().join(", ")
            }
        )
    })
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// The transformer's shard files, from its index's `weight_map`, sorted.
fn shard_paths(transformer_dir: &Path) -> Result<Vec<PathBuf>> {
    #[derive(serde::Deserialize)]
    struct Index {
        weight_map: std::collections::BTreeMap<String, String>,
    }
    let index: Index =
        read_json(&transformer_dir.join("diffusion_pytorch_model.safetensors.index.json"))?;
    let mut names: Vec<&String> = index.weight_map.values().collect();
    names.sort();
    names.dedup();
    let paths: Vec<PathBuf> = names.into_iter().map(|n| transformer_dir.join(n)).collect();
    for path in &paths {
        ensure!(
            path.is_file(),
            "missing transformer shard {}",
            path.display()
        );
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen_image::transformer::AttnArm;

    /// A pipeline small enough for the CPU: the transformer's tiny test model
    /// narrowed to the tiny VAE's three latent channels.
    fn tiny_pipeline(cache: CacheArm) -> QwenImagePipeline {
        let vae_cfg = crate::qwen_image::vae::tests::tiny();
        let mut cfg = crate::qwen_image::transformer::tests::tiny_config();
        cfg.in_channels = vae_cfg.z_dim;
        cfg.out_channels = Some(vae_cfg.z_dim);
        let transformer =
            crate::qwen_image::transformer::tests::tiny_model_of(&cfg, AttnArm::Basic);
        let vae = crate::qwen_image::vae::tests::patterned(&vae_cfg, 0.2);
        QwenImagePipeline::from_parts(
            transformer,
            vae,
            SchedulerConfig::qwen_image_21(),
            cache,
            &Device::Cpu,
        )
    }

    fn caption(pipeline: &QwenImagePipeline) -> Tensor {
        let dim = pipeline.transformer_config().context_in_dim;
        let values: Vec<f32> = (0..5 * dim).map(|i| (i as f32 * 0.41).sin()).collect();
        Tensor::from_vec(values, (5, dim), &Device::Cpu).unwrap()
    }

    fn options() -> ImageOptions {
        ImageOptions {
            width: 64,
            height: 32,
            steps: 3,
            seed: 11,
            latents: None,
        }
    }

    fn floats(t: &Tensor) -> Vec<f32> {
        let values = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|x| x.is_finite()), "a nonfinite value");
        values
    }

    #[test]
    fn sizes_are_multiples_of_32_inside_the_rope() {
        for (w, h) in [
            (32, 32),
            (1024, 1024),
            (2048, 2048),
            (2752, 1536),
            (64, 32768),
        ] {
            QwenImagePipeline::check_size(w, h).unwrap_or_else(|e| panic!("{w}x{h}: {e:#}"));
        }
        for (w, h) in [(0, 32), (32, 0), (16, 32), (1024, 1040), (1000, 1000)] {
            assert!(QwenImagePipeline::check_size(w, h).is_err(), "{w}x{h}");
        }
        // One 32 px step past the rope's lowest row.
        assert!(QwenImagePipeline::check_size(32, 32768 + 32).is_err());
        // Sizes no allocation could hold are errors, not panics.
        let huge = (u32::MAX as usize / 32) * 32;
        for (w, h) in [
            (huge, huge),
            (32, huge),
            (huge, 32),
            (usize::MAX / 32 * 32, 32),
        ] {
            assert!(QwenImagePipeline::check_size(w, h).is_err(), "{w}x{h}");
            assert!(QwenImagePipeline::peak_bytes(w, h).is_err(), "{w}x{h}");
        }
        assert_eq!(QwenImagePipeline::latent_size(1024, 512), (32, 64));
        assert_eq!(QwenImagePipeline::target_tokens(1024, 1024), 4096);
        assert!(
            QwenImagePipeline::peak_bytes(2048, 2048).unwrap()
                > QwenImagePipeline::peak_bytes(1024, 1024).unwrap()
        );
        // A caption and an image share the rope: the text takes the first
        // positions, so one that alone would fit is refused beside an image
        // whose frame it pushes past the end, and before anything allocates.
        assert!(QwenImagePipeline::check_layout(73, 1024, 1024).is_ok());
        assert!(QwenImagePipeline::check_layout(8191, 1024, 1024).is_ok());
        assert!(QwenImagePipeline::check_layout(8192, 1024, 1024).is_err());
        assert!(QwenImagePipeline::check_layout(usize::MAX, 1024, 1024).is_err());
        assert!(QwenImagePipeline::check_layout(0, 1024, 1024).is_err());
        assert!(QwenImagePipeline::check_layout(73, 1000, 1024).is_err());
        // The estimate covers both measured runs with its margin.
        for (side, measured) in [(512, MEASURED_PEAK_512), (1024, MEASURED_PEAK_1024)] {
            let estimate = QwenImagePipeline::peak_bytes(side, side).unwrap();
            assert!(
                estimate as f64 >= measured as f64 * 1.15,
                "{side}x{side}: {estimate} against a measured {measured}"
            );
        }
    }

    #[test]
    fn the_cache_switch_names_two_arms_and_refuses_a_typo() {
        assert_eq!(CacheArm::parse("").unwrap(), CacheArm::Prefix);
        assert_eq!(CacheArm::parse("on").unwrap(), CacheArm::Prefix);
        assert_eq!(CacheArm::parse("off").unwrap(), CacheArm::Full);
        assert_eq!(CacheArm::parse("full").unwrap(), CacheArm::Full);
        assert!(CacheArm::parse("of").is_err());
    }

    #[test]
    fn the_cached_loop_matches_the_full_loop() {
        let cached = tiny_pipeline(CacheArm::Prefix);
        let full = tiny_pipeline(CacheArm::Full);
        let cap = caption(&cached);
        let a = cached.generate(&cap, &options()).unwrap();
        let b = full.generate(&cap, &options()).unwrap();
        assert_eq!(a.timings.steps.len(), 3);
        assert_eq!(a.final_latents.dims(), [1, 3, 2, 4]);
        // A random decoder's alpha plane lands wherever it lands, so the form
        // follows the rule and the RGB planes are there either way.
        let channels = if a.clear_pixels >= CLEAR_PIXELS_MIN {
            4
        } else {
            3
        };
        assert_eq!(a.image.dims(), [channels, 32, 64]);
        assert_eq!(a.rgb().unwrap().dims(), [3, 32, 64]);
        let (x, y) = (floats(&a.final_latents), floats(&b.final_latents));
        let diff = x
            .iter()
            .zip(&y)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f32::max);
        assert!(diff < 1e-4, "cached and full loops differ by {diff}");
        // The loop moved the latent: three steps of a nonzero velocity.
        let start = floats(&seeded_noise(11, (1, 3, 2, 4), &Device::Cpu).unwrap());
        assert!(x.iter().zip(&start).any(|(p, q)| (p - q).abs() > 1e-3));
    }

    #[test]
    fn the_runs_first_velocity_is_the_standalone_forward() {
        let pipeline = tiny_pipeline(CacheArm::Prefix);
        let cap = caption(&pipeline);
        let noise = seeded_noise(11, (1, 3, 2, 4), &Device::Cpu).unwrap();
        let opts = ImageOptions {
            latents: Some(noise.clone()),
            ..options()
        };
        let run = pipeline.generate(&cap, &opts).unwrap();
        let (sigmas, _) = pipeline.sigmas(3, 64, 32).unwrap();
        assert_eq!(sigmas.len(), 4);
        assert_eq!(sigmas[0], 1.0);
        assert_eq!(sigmas[3], 0.0);
        let alone = pipeline
            .velocity(&cap, &noise, sigmas[0], GraphVariant::Reference)
            .unwrap();
        assert_eq!(floats(&run.velocity0.unwrap()), floats(&alone));
        // An injected latent of the wrong shape is refused.
        let wrong = ImageOptions {
            latents: Some(seeded_noise(1, (1, 3, 4, 4), &Device::Cpu).unwrap()),
            ..options()
        };
        assert!(pipeline.generate(&cap, &wrong).is_err());
    }

    #[test]
    fn the_alpha_plane_is_measured_and_dropped() {
        // R, G, B at -1, 0, 1 and an alpha plane whose lowest value is 0.5
        // in [-1, 1], which is 0.75 in [0, 1].
        let plane = |v: f32| Tensor::full(v, (1, 1, 2, 2), &Device::Cpu).unwrap();
        let alpha = Tensor::new(&[1.0f32, 0.5, 1.0, 0.9], &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 2, 2))
            .unwrap();
        let rgba = Tensor::cat(&[plane(-1.0), plane(0.0), plane(1.0), alpha], 1).unwrap();
        let decoded = split_rgba(&rgba).unwrap();
        assert_eq!(decoded.image.dims(), [3, 2, 2]);
        assert_eq!(
            decoded
                .image
                .flatten_all()
                .unwrap()
                .to_vec1::<u8>()
                .unwrap(),
            [0, 0, 0, 0, 128, 128, 128, 128, 255, 255, 255, 255]
        );
        assert!((decoded.alpha_max_distance_from_opaque - 0.25).abs() < 1e-6);
        let rgb = Tensor::cat(&[plane(-1.0), plane(0.0), plane(1.0)], 1).unwrap();
        assert_eq!(
            split_rgba(&rgb).unwrap().alpha_max_distance_from_opaque,
            0.0
        );
    }

    /// The alpha plane is kept from the tenth clear pixel on, and a pixel is
    /// clear through alpha 8 and not at 9.
    #[test]
    fn ten_clear_pixels_keep_the_alpha_plane() {
        let plane = |clear: usize, value: u8| {
            let mut alpha = vec![254u8; 64];
            alpha[..clear].fill(value);
            alpha
        };
        assert!(!keeps_alpha(&plane(CLEAR_PIXELS_MIN - 1, 0)));
        assert!(keeps_alpha(&plane(CLEAR_PIXELS_MIN, 0)));
        assert!(keeps_alpha(&plane(CLEAR_PIXELS_MIN, CLEAR_ALPHA_MAX)));
        assert!(!keeps_alpha(&plane(CLEAR_PIXELS_MIN, CLEAR_ALPHA_MAX + 1)));
        // The noise floor of an ordinary render keeps nothing, however much
        // of the image sits one step short of opaque.
        assert!(!keeps_alpha(&[252, 253, 254, 254, 254, 255].repeat(1000)));
        assert!(!keeps_alpha(&[]));
    }

    /// A decode with a clear region comes out as four channels with the alpha
    /// bytes intact; one without comes out as three, and both report what the
    /// plane held.
    #[test]
    fn a_clear_region_makes_the_image_rgba() {
        let (h, w) = (4usize, 4usize);
        let plane = |v: f32| Tensor::full(v, (1, 1, h, w), &Device::Cpu).unwrap();
        let alpha_plane = |clear: usize| {
            let mut alpha = vec![1.0f32; h * w];
            alpha[..clear].fill(-1.0);
            Tensor::from_vec(alpha, (1, 1, h, w), &Device::Cpu).unwrap()
        };
        let decode = |clear: usize| {
            let rgba = Tensor::cat(
                &[plane(-1.0), plane(0.0), plane(1.0), alpha_plane(clear)],
                1,
            )
            .unwrap();
            split_rgba(&rgba).unwrap()
        };

        let kept = decode(CLEAR_PIXELS_MIN);
        assert_eq!(kept.image.dims(), [4, h, w]);
        assert_eq!(kept.clear_pixels, CLEAR_PIXELS_MIN);
        assert_eq!(kept.alpha_min, 0);
        let alpha = kept
            .image
            .narrow(0, 3, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap();
        assert!(alpha[..CLEAR_PIXELS_MIN].iter().all(|&a| a == 0));
        assert!(alpha[CLEAR_PIXELS_MIN..].iter().all(|&a| a == 255));
        assert_eq!(kept.rgb().unwrap().dims(), [3, h, w]);

        let dropped = decode(CLEAR_PIXELS_MIN - 1);
        assert_eq!(dropped.image.dims(), [3, h, w]);
        assert_eq!(dropped.clear_pixels, CLEAR_PIXELS_MIN - 1);
        assert_eq!(dropped.alpha_min, 0);
        assert_eq!(
            dropped
                .rgb()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<u8>()
                .unwrap(),
            kept.rgb()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<u8>()
                .unwrap()
        );
    }
}
