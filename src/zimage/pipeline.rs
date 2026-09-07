//! The Z-Image-Turbo text-to-image pipeline: conditioning in, pixels out.
//!
//! What diffusers' `ZImagePipeline.__call__` does after `encode_prompt`, at
//! batch 1 and without guidance (Turbo is distilled and runs at
//! `guidance_scale` 0): draw the latent noise, run the transformer once per
//! scheduler step, Euler-step the latent in f32, decode with the VAE in f32
//! (`force_upcast`), and quantize to bytes. The conditioning tensor comes
//! from [`crate::XwenModel::encode`] and is not produced here.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::sampling::{postprocess_image, seeded_noise};
use super::scheduler::{FlowMatchEulerDiscreteScheduler, SchedulerConfig};
use super::transformer::{
    AXES_LENS, AttnImpl, Config, LATENT_CHANNELS, SEQ_MULTI_OF, ZImageTransformer2DModel,
};
use super::vae::{AutoEncoderKL, VaeConfig};

/// The pipeline's own divisibility rule: the VAE compresses 8x and the
/// transformer patches 2x, so a side is a whole number of 16-pixel cells.
pub const PIXELS_PER_TOKEN: usize = 16;

/// The number of denoising steps the release is fitted for. The model card's
/// sample says 9 against a diffusers whose ninth step was a skipped terminal
/// zero; the official pipeline and current diffusers both run 8 forwards.
pub const DEFAULT_STEPS: usize = 8;

/// What one image run is asked for.
#[derive(Debug, Clone)]
pub struct ImageOptions {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// The noise seed. Meaningless when `latents` is given.
    pub seed: u64,
    /// The initial latent, `(1, 16, height/8, width/8)` f32, in place of the
    /// seeded draw — the way a reference comparison makes both sides start
    /// from the same noise.
    pub latents: Option<Tensor>,
}

/// Wall time of each phase of one run, for the caller to print.
#[derive(Debug, Clone, Default)]
pub struct Timings {
    /// One entry per transformer forward, seconds.
    pub steps: Vec<f64>,
    pub vae_decode: f64,
}

/// What one run produced: the image, and the two intermediate tensors the
/// parity gate grades (docs/zimage.md, Stages 3 and 4). Both are small and on
/// the device, so keeping them costs nothing a caller would notice.
#[derive(Debug, Clone)]
pub struct Rendered {
    /// `[3, H, W]` u8 RGB on the CPU.
    pub image: Tensor,
    pub timings: Timings,
    /// The velocity fed to the first Euler step, `(1, 16, H/8, W/8)` f32:
    /// the negated transformer output at sigma 1, which is the one forward
    /// whose inputs are known exactly on both sides of a comparison.
    pub velocity0: Tensor,
    /// The latent after the last step and before the VAE, same shape, f32.
    pub final_latents: Tensor,
}

/// The transformer, the VAE and the scheduler config, resident on one device.
pub struct ZImagePipeline {
    transformer: ZImageTransformer2DModel,
    transformer_cfg: Config,
    vae: AutoEncoderKL,
    scheduler_cfg: SchedulerConfig,
    device: Device,
    /// The transformer's compute dtype. bf16, as the reference runs it.
    dtype: DType,
}

impl ZImagePipeline {
    /// Open the pipeline at a repo snapshot root — the directory holding
    /// `model_index.json`, `transformer/`, `vae/` and `scheduler/`.
    ///
    /// The transformer's fp32 shards are cast to bf16 as they load; the VAE's
    /// bf16 weights are widened to f32, which is the dtype it decodes in.
    pub fn load(root: &Path, device: &Device) -> Result<Self> {
        // Before anything opens: a typo in the bisect switch is a load error,
        // not a run that quietly measures the shipped arm twice.
        let attn = AttnImpl::from_env()?;
        let transformer_dir = root.join("transformer");
        let mut transformer_cfg: Config = read_json(&transformer_dir.join("config.json"))?;
        transformer_cfg.set_attn_impl(attn);
        if attn != AttnImpl::Fused {
            eprintln!("xwen: z-image attention arm: {}", attn.label());
        }
        let shards = shard_paths(&transformer_dir)?;
        let dtype = DType::BF16;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, dtype, device)? };
        let transformer = ZImageTransformer2DModel::new(&transformer_cfg, vb)
            .context("building the Z-Image transformer")?;

        let vae_dir = root.join("vae");
        let vae_cfg: VaeConfig = read_json(&vae_dir.join("config.json"))?;
        let vae_file = vae_dir.join("diffusion_pytorch_model.safetensors");
        ensure!(vae_file.is_file(), "missing {}", vae_file.display());
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&vae_file], DType::F32, device)? };
        let vae = AutoEncoderKL::new(&vae_cfg, vb).context("building the Z-Image VAE")?;

        let scheduler_cfg: SchedulerConfig =
            read_json(&root.join("scheduler").join("scheduler_config.json"))?;
        // Fail on an unsupported scheduler config at load, not at the first
        // step, after 12 GB of weights are resident.
        FlowMatchEulerDiscreteScheduler::new(scheduler_cfg.clone())?;

        Ok(Self {
            transformer,
            transformer_cfg,
            vae,
            scheduler_cfg,
            device: device.clone(),
            dtype,
        })
    }

    /// Whether a `width x height` is one this pipeline runs, and why not.
    ///
    /// Three rules. Both sides are whole 16-pixel cells; the cell count is a
    /// multiple of 32 so the image sequence needs no pad rows (see
    /// `ZImageTransformer2DModel::forward`); and each side's cell count fits
    /// inside its RoPE table, [`AXES_LENS`] rows on axis 1 for the image's
    /// rows and axis 2 for its columns, which caps a side at 8192 px. That
    /// third rule is not a formality: candle's Metal `index_select` clamps an
    /// out-of-range position to the table's last row instead of failing, so
    /// past the cap the run would return a plausible image built from the
    /// wrong rotations. [`AXES_LENS`] is the shipped `transformer/config.json`
    /// and this is a static check with no config in hand; the same bound is
    /// re-checked against the loaded `axes_lens` inside the transformer.
    pub fn check_size(width: usize, height: usize) -> Result<()> {
        ensure!(
            width > 0
                && height > 0
                && width.is_multiple_of(PIXELS_PER_TOKEN)
                && height.is_multiple_of(PIXELS_PER_TOKEN),
            "{width}x{height}: both sides must be positive multiples of {PIXELS_PER_TOKEN}"
        );
        let cols = width / PIXELS_PER_TOKEN;
        let rows = height / PIXELS_PER_TOKEN;
        // Checked, because the caller's width and height are unvalidated
        // `usize`s and the product of two of them is not obviously one.
        let tokens = rows.checked_mul(cols).with_context(|| {
            format!("{width}x{height} is more image tokens than a usize can count")
        })?;
        ensure!(
            tokens.is_multiple_of(SEQ_MULTI_OF),
            "{width}x{height} is {tokens} image tokens ({cols}x{rows} cells of \
             {PIXELS_PER_TOKEN} px), not a multiple of {SEQ_MULTI_OF}; sizes whose cell count \
             needs padding are not supported yet (1024x1024, 1024x768, 768x1024, 512x512 all \
             are)"
        );
        let (max_rows, max_cols) = (AXES_LENS[1], AXES_LENS[2]);
        ensure!(
            rows <= max_rows && cols <= max_cols,
            "{width}x{height} is a {rows}x{cols} cell grid (rows x columns) and the rope \
             tables position only {max_rows}x{max_cols}, so a side is capped at {} px",
            max_cols * PIXELS_PER_TOKEN
        );
        Ok(())
    }

    /// The latent grid `(height/8, width/8)` an image size maps to.
    pub fn latent_size(width: usize, height: usize) -> (usize, usize) {
        (
            2 * (height / PIXELS_PER_TOKEN),
            2 * (width / PIXELS_PER_TOKEN),
        )
    }

    /// Read an injected latent from a safetensors file and check it fits a
    /// `width x height` run, as f32 on the CPU.
    ///
    /// Separate from [`Self::generate`]'s own check, and called BEFORE
    /// anything loads: a typo in the path or a latent from another resolution
    /// otherwise costs the encoder and the transformer — about 33 GB and a
    /// minute — before it is noticed. `generate` re-checks against the loaded
    /// `in_channels` and that stays the authority for what runs.
    pub fn read_latents(path: &Path, width: usize, height: usize) -> Result<Tensor> {
        Self::check_size(width, height)?;
        let tensors = candle_core::safetensors::load(path, &Device::Cpu)
            .with_context(|| format!("reading the latents file {}", path.display()))?;
        let latents = tensors.get("latents").with_context(|| {
            format!(
                "{} has no `latents` tensor; it holds {}",
                path.display(),
                if tensors.is_empty() {
                    "nothing".to_string()
                } else {
                    tensors.keys().cloned().collect::<Vec<_>>().join(", ")
                }
            )
        })?;
        let latents = if latents.rank() == 3 {
            latents.unsqueeze(0)?
        } else {
            latents.clone()
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

    /// Read injected caption features, the encoder's `[T, 2560]` hidden state,
    /// from a safetensors file holding them under `cap_feats`, in whatever
    /// float dtype they were saved in. The width is checked against the
    /// loaded transformer by [`Self::generate`]; this only refuses what cannot
    /// be a caption at all, and it does so before anything loads, like
    /// [`Self::read_latents`].
    pub fn read_cap_feats(path: &Path) -> Result<Tensor> {
        let tensors = candle_core::safetensors::load(path, &Device::Cpu)
            .with_context(|| format!("reading the caption file {}", path.display()))?;
        let cap = tensors.get("cap_feats").with_context(|| {
            format!(
                "{} has no `cap_feats` tensor; it holds {}",
                path.display(),
                if tensors.is_empty() {
                    "nothing".to_string()
                } else {
                    tensors.keys().cloned().collect::<Vec<_>>().join(", ")
                }
            )
        })?;
        let (t, _dim) = cap
            .dims2()
            .with_context(|| format!("the caption in {} is not [T, dim]", path.display()))?;
        ensure!(t > 0, "the caption in {} has no tokens", path.display());
        ensure!(
            cap.dtype().is_float(),
            "the caption in {} is {:?}, not a float dtype",
            path.display(),
            cap.dtype()
        );
        Ok(cap.clone())
    }

    /// Generate one image from `cap_feats`, the text encoder's `[T, 2560]`
    /// hidden state (any float dtype, any device).
    pub fn generate(&self, cap_feats: &Tensor, opts: &ImageOptions) -> Result<Rendered> {
        Self::check_size(opts.width, opts.height)?;
        ensure!(opts.steps >= 1, "steps must be at least 1");
        let (t, cap_dim) = cap_feats
            .dims2()
            .context("cap_feats is [T, cap_feat_dim]")?;
        ensure!(
            cap_dim == self.transformer_cfg.cap_feat_dim,
            "cap_feats is {cap_dim} wide, the transformer takes {}",
            self.transformer_cfg.cap_feat_dim
        );
        ensure!(t > 0, "cap_feats has no tokens");
        let cap_feats = cap_feats
            .to_device(&self.device)?
            .to_dtype(self.dtype)?
            .unsqueeze(0)?;

        let (lat_h, lat_w) = Self::latent_size(opts.width, opts.height);
        let latent_shape = (1, self.transformer_cfg.in_channels, lat_h, lat_w);
        let mut latents = match &opts.latents {
            Some(given) => {
                let given = if given.rank() == 3 {
                    given.unsqueeze(0)?
                } else {
                    given.clone()
                };
                ensure!(
                    given.dims4()? == latent_shape,
                    "the given latents are {:?}, this size needs {:?}",
                    given.dims(),
                    latent_shape
                );
                given.to_device(&self.device)?.to_dtype(DType::F32)?
            }
            None => seeded_noise(opts.seed, latent_shape, &self.device)?,
        };

        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(self.scheduler_cfg.clone())?;
        scheduler.set_timesteps(opts.steps)?;
        let mut timings = Timings::default();
        let mut velocity0 = None;

        for _ in 0..opts.steps {
            let started = Instant::now();
            let t = scheduler.current_timestep_normalized();
            let velocity = self.velocity_batched(&latents, &cap_feats, t as f32)?;
            if velocity0.is_none() {
                velocity0 = Some(velocity.clone());
            }
            latents = scheduler.step(&velocity, &latents)?;
            // The step's arithmetic is asynchronous on Metal; reading one
            // element back is what makes the timing mean anything.
            let _ = latents.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            timings.steps.push(started.elapsed().as_secs_f64());
        }
        let velocity0 = velocity0.expect("at least one step ran");

        let started = Instant::now();
        let image = self.decode(&latents)?;
        timings.vae_decode = started.elapsed().as_secs_f64();
        Ok(Rendered {
            image,
            timings,
            velocity0,
            final_latents: latents,
        })
    }

    /// One transformer forward: the velocity towards data at normalized
    /// timestep `t` (`1 - sigma`) for a `(1, 16, H/8, W/8)` f32 latent and
    /// the encoder's `[T, 2560]` caption. The body of one Euler step in
    /// [`Self::generate`], public so the parity gate can grade a single
    /// forward and so a deliberately wrong `t` can bracket its bar.
    pub fn velocity(&self, latents: &Tensor, cap_feats: &Tensor, t: f32) -> Result<Tensor> {
        let cap_feats = cap_feats
            .to_device(&self.device)?
            .to_dtype(self.dtype)?
            .unsqueeze(0)?;
        let latents = latents.to_device(&self.device)?.to_dtype(DType::F32)?;
        self.velocity_batched(&latents, &cap_feats, t)
    }

    /// [`Self::velocity`] with the caption already batched and in the model
    /// dtype, which the step loop does once rather than per step.
    fn velocity_batched(&self, latents: &Tensor, cap_feats: &Tensor, t: f32) -> Result<Tensor> {
        // The timestep stays f32: the sinusoidal embedding is computed in
        // f32 from it (as the reference does, autocast off) and only the
        // embedding is cast to the model dtype. A bf16 timestep would
        // round 0.0454 to 0.0454102 before anything used it.
        let t = Tensor::new(&[t], &self.device)?;
        let x = latents.to_dtype(self.dtype)?.unsqueeze(2)?;
        let pred = self.transformer.forward(&x, &t, cap_feats)?;
        // The transformer predicts the flow towards noise; the Euler step
        // wants the velocity towards data. Kept in f32 like the latent.
        Ok(pred.squeeze(2)?.to_dtype(DType::F32)?.neg()?)
    }

    /// Decode a `(1, 16, H/8, W/8)` latent through the VAE into `[3, H, W]`
    /// u8 RGB on the CPU. The tail of [`Self::generate`], public so a
    /// reference latent can be decoded through this VAE alone.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let latents = latents.to_device(&self.device)?.to_dtype(DType::F32)?;
        let image = self.vae.decode(&latents)?; // (1, 3, H, W) f32
        Ok(postprocess_image(&image)?
            .squeeze(0)?
            .to_device(&Device::Cpu)?)
    }

    pub fn transformer_config(&self) -> &Config {
        &self.transformer_cfg
    }

    pub fn scheduler_config(&self) -> &SchedulerConfig {
        &self.scheduler_cfg
    }
}

/// Encode a `[3, H, W]` u8 tensor as an RGB8 PNG at `path`.
///
/// Written to a sibling temporary file and renamed into place, so a failure
/// anywhere in the encode leaves whatever was at `path` untouched. Creating
/// the destination first truncates it, which on a re-run over the previous
/// image destroys the one comparable artefact the run had.
///
/// The temporary name is unique per WRITER, not per process, and it is claimed
/// by exclusive creation rather than chosen. A pid is enough to separate two
/// `xwen image` runs and not enough to separate two callers inside one process,
/// which the image serve route will be: they would share the name, interleave
/// their pixels into it, and each rename a half-written file over the other's
/// output. The counter only supplies the next candidate; the filesystem decides
/// who got it.
pub fn write_png(image: &Tensor, path: &Path) -> Result<()> {
    let bytes = encode_png(image)?;
    // A sibling, so the rename is within one filesystem and therefore atomic.
    let Some(name) = path.file_name().map(|n| n.to_string_lossy().into_owned()) else {
        bail!("{} is not a file path to write a PNG to", path.display());
    };
    let pid = std::process::id();
    let (temp, file) = {
        let mut attempt = 0u32;
        loop {
            let candidate = path.with_file_name(format!(".{name}.{pid}.{attempt}.tmp"));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(file) => break (candidate, file),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    attempt += 1;
                    // Bounded, because a thousand taken names beside one output
                    // is a directory this cannot win in — leftovers from killed
                    // runs, most likely — and a spin there is worse than saying
                    // so.
                    ensure!(
                        attempt < 1024,
                        "no free temporary name beside {} after {attempt} tries; \
                         stale `.{name}.*.tmp` files may need clearing",
                        path.display()
                    );
                }
                Err(e) => {
                    return Err(e).with_context(|| format!("creating {}", candidate.display()));
                }
            }
        }
    };
    let write = || -> Result<()> {
        let mut file = file;
        std::io::Write::write_all(&mut file, &bytes)?;
        std::io::Write::flush(&mut file)?;
        Ok(())
    };
    if let Err(e) = write() {
        let _ = std::fs::remove_file(&temp);
        return Err(e);
    }
    // The temp is this writer's alone, so a rename that failed leaves a file
    // nothing will ever pick up. Removed on the way out, or it accumulates one
    // whole image per failure beside the output.
    if let Err(e) = std::fs::rename(&temp, path) {
        let _ = std::fs::remove_file(&temp);
        return Err(e).with_context(|| {
            format!(
                "renaming {} into place as {}",
                temp.display(),
                path.display()
            )
        });
    }
    Ok(())
}

/// Encode a `[3, H, W]` u8 tensor as RGB8 PNG bytes, the form an HTTP
/// response wants and [`write_png`] writes.
pub fn encode_png(image: &Tensor) -> Result<Vec<u8>> {
    let (c, h, w) = image.dims3().context("the image is [3, H, W]")?;
    ensure!(c == 3, "the image has {c} channels, PNG RGB needs 3");
    // Channel-last, interleaved, as PNG wants it.
    let pixels: Vec<u8> = image
        .permute((1, 2, 0))?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<u8>()?;
    let mut bytes = Vec::with_capacity(pixels.len() / 2);
    let mut encoder = png::Encoder::new(&mut bytes, w as u32, h as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&pixels)?;
    writer.finish()?;
    Ok(bytes)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// The transformer's shard files, from its index's `weight_map`, sorted.
fn shard_paths(transformer_dir: &Path) -> Result<Vec<PathBuf>> {
    let index = transformer_dir.join("diffusion_pytorch_model.safetensors.index.json");
    if index.is_file() {
        #[derive(serde::Deserialize)]
        struct Index {
            weight_map: std::collections::BTreeMap<String, String>,
        }
        let index: Index = read_json(&index)?;
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
        return Ok(paths);
    }
    let single = transformer_dir.join("diffusion_pytorch_model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    bail!(
        "no transformer weights under {}: neither an index nor a single safetensors file",
        transformer_dir.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_size_rule_admits_the_common_sizes_and_names_the_reason_otherwise() {
        for (w, h) in [
            (1024, 1024),
            (1024, 768),
            (768, 1024),
            (512, 512),
            (1536, 1024),
        ] {
            ZImagePipeline::check_size(w, h).unwrap_or_else(|e| panic!("{w}x{h}: {e}"));
        }
        // 1024x1024 is 64x64 = 4096 tokens: no padding, the reference case.
        assert_eq!(ZImagePipeline::latent_size(1024, 1024), (128, 128));
        // Not a multiple of 16.
        let err = ZImagePipeline::check_size(1000, 1000)
            .unwrap_err()
            .to_string();
        assert!(err.contains("multiples of 16"), "{err}");
        // 528x528 is 33x33 = 1089 tokens, which would need pad rows.
        let err = ZImagePipeline::check_size(528, 528)
            .unwrap_err()
            .to_string();
        assert!(err.contains("1089 image tokens"), "{err}");
        assert!(ZImagePipeline::check_size(0, 1024).is_err());
    }

    /// The rope tables hold 512 positions per spatial axis, so 8192 px is the
    /// largest side, and one cell past it is refused rather than clamped.
    ///
    /// Clamped is what it would otherwise be: candle's Metal `index_select`
    /// returns the table's last row for an out-of-range position, so the run
    /// would produce an image rather than an error, and the image would be
    /// built from the wrong rotations everywhere past row or column 511.
    #[test]
    fn a_size_past_the_rope_tables_is_refused_rather_than_clamped() {
        // The boundary itself is a 512x512 cell grid and 262144 tokens.
        ZImagePipeline::check_size(8192, 8192).unwrap();

        // One cell wider: 513 columns, still a multiple-of-32 token count, so
        // this is the new rule firing and not the old one.
        let tokens = (8208 / PIXELS_PER_TOKEN) * (1024 / PIXELS_PER_TOKEN);
        assert!(tokens.is_multiple_of(SEQ_MULTI_OF), "{tokens}");
        let err = ZImagePipeline::check_size(8208, 1024)
            .unwrap_err()
            .to_string();
        assert!(err.contains("8192 px"), "{err}");
        let err = ZImagePipeline::check_size(1024, 8208)
            .unwrap_err()
            .to_string();
        assert!(err.contains("8192 px"), "{err}");

        // And the token count is counted, not wrapped: two sides that are each
        // legal multiples of 16 can still multiply past a usize.
        let huge = 1usize << 40;
        let err = ZImagePipeline::check_size(huge, huge)
            .unwrap_err()
            .to_string();
        assert!(err.contains("than a usize can count"), "{err}");
    }

    /// An injected latent is read and checked before anything loads, so a
    /// typo in the path costs a millisecond rather than the encoder and the
    /// transformer.
    ///
    /// Every rejection names the file, because the caller passed a path and
    /// that is what they can go and look at.
    #[test]
    fn an_injected_latent_is_validated_from_the_file() {
        let dir = std::env::temp_dir().join(format!("xwen-latents-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // A missing file, which is the typo case.
        let err = ZImagePipeline::read_latents(&dir.join("nope.safetensors"), 1024, 1024)
            .unwrap_err()
            .to_string();
        assert!(err.contains("nope.safetensors"), "{err}");

        // The right shape under the wrong name.
        let good = Tensor::zeros((1, LATENT_CHANNELS, 128, 128), DType::F32, &Device::Cpu).unwrap();
        let named_wrong = dir.join("named-wrong.safetensors");
        candle_core::safetensors::save(
            &std::collections::HashMap::from([("latent".to_string(), good.clone())]),
            &named_wrong,
        )
        .unwrap();
        let err = ZImagePipeline::read_latents(&named_wrong, 1024, 1024)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no `latents` tensor"), "{err}");
        assert!(err.contains("latent"), "{err}");

        // The right name at another resolution's shape, which is the mistake a
        // reference comparison actually makes.
        let path = dir.join("latents.safetensors");
        candle_core::safetensors::save(
            &std::collections::HashMap::from([("latents".to_string(), good.clone())]),
            &path,
        )
        .unwrap();
        let err = ZImagePipeline::read_latents(&path, 512, 512)
            .unwrap_err()
            .to_string();
        assert!(err.contains("512x512 needs"), "{err}");
        assert!(err.contains("[1, 16, 128, 128]"), "{err}");

        // And the matching one, cast to f32 whatever it was stored as. Rank 3
        // is accepted too: a torch dump of one latent often has no batch axis.
        let read = ZImagePipeline::read_latents(&path, 1024, 1024).unwrap();
        assert_eq!(read.dims(), &[1, LATENT_CHANNELS, 128, 128]);
        assert_eq!(read.dtype(), DType::F32);

        let unbatched = dir.join("unbatched.safetensors");
        candle_core::safetensors::save(
            &std::collections::HashMap::from([(
                "latents".to_string(),
                good.squeeze(0).unwrap().to_dtype(DType::BF16).unwrap(),
            )]),
            &unbatched,
        )
        .unwrap();
        let read = ZImagePipeline::read_latents(&unbatched, 1024, 1024).unwrap();
        assert_eq!(read.dims(), &[1, LATENT_CHANNELS, 128, 128]);
        assert_eq!(read.dtype(), DType::F32);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A directory of its own per test: these run in parallel in one process,
    /// and the temporary names below are derived from the process id, so a
    /// shared directory is one test staging a failure for another.
    fn png_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("xwen-png-{}-{label}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A failed PNG write leaves the previous image at that path alone.
    ///
    /// The failure is staged by making the output's DIRECTORY unwritable, so
    /// the temporary file cannot be created at all. What it pins is the
    /// ordering: nothing truncates the destination, so a re-run that dies
    /// before the rename still has yesterday's image to compare against.
    #[test]
    fn a_failed_png_write_does_not_destroy_the_previous_image() {
        use std::os::unix::fs::PermissionsExt;
        let dir = png_dir("failure");
        let out = dir.join("out.png");

        let image = Tensor::zeros((3, 16, 16), DType::U8, &Device::Cpu).unwrap();
        write_png(&image, &out).unwrap();
        let good = std::fs::read(&out).unwrap();
        assert!(good.starts_with(b"\x89PNG"), "not a PNG: {:?}", &good[..4]);

        let mode = std::fs::metadata(&dir).unwrap().permissions().mode();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o555)).unwrap();
        let err = write_png(&image, &out).unwrap_err();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(mode)).unwrap();
        assert!(format!("{err:#}").contains("creating"), "{err:#}");
        assert_eq!(std::fs::read(&out).unwrap(), good);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A taken temporary name is stepped over rather than reused, and nothing
    /// is left behind on the way through.
    ///
    /// The name used to be `.{output}.{pid}.tmp` and nothing else, so two
    /// writers to one destination inside one process shared a file: each would
    /// encode into it and each would rename it, and one of the two images was
    /// whatever the interleaving produced. Exclusive creation is what makes
    /// that impossible, and the observable consequence is this — a candidate
    /// somebody else holds is skipped, not opened.
    #[test]
    fn a_taken_temporary_name_is_skipped_and_no_temporary_survives() {
        let dir = png_dir("collision");
        let out = dir.join("out.png");
        let pid = std::process::id();

        // Candidate 0 held by something the writer cannot open: exactly the
        // situation a second writer in this process creates.
        let taken = dir.join(format!(".out.png.{pid}.0.tmp"));
        std::fs::create_dir(&taken).unwrap();

        let image = Tensor::zeros((3, 16, 16), DType::U8, &Device::Cpu).unwrap();
        write_png(&image, &out).unwrap();
        assert!(std::fs::read(&out).unwrap().starts_with(b"\x89PNG"));
        assert!(taken.is_dir(), "the held candidate was consumed");
        assert!(
            !dir.join(format!(".out.png.{pid}.1.tmp")).exists(),
            "the temporary the write used is still there"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
