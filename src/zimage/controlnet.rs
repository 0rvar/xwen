//! Fun Union 2.1 distilled ControlNet, following VideoX-Fun's joint-token graph.
//!
//! Only the side weights live here. The transformer owns and evaluates its
//! shared embedders and refiners, so adapters on them also condition this graph.

use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use super::linear::{LinearImpl, Projection, WeightRange, ensure_weights_fit_f16};
use super::transformer::{Config, ZImageTransformerBlock, patchify};

pub const CONTROL_CHANNELS: usize = 33;
pub const CONTROL_REPO: &str = "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1";
pub const DEFAULT_CONTROL_FILE: &str =
    "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors";

/// Offline resolution; a control request never downloads its side weights.
pub fn cached_default() -> anyhow::Result<PathBuf> {
    if let Some(path) = std::env::var_os("XWEN_CONTROLNET_FILE") {
        let path = PathBuf::from(path);
        ensure!(
            path.is_file(),
            "XWEN_CONTROLNET_FILE {} is not a file",
            path.display()
        );
        return Ok(path);
    }
    for filename in [
        DEFAULT_CONTROL_FILE,
        "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps.safetensors",
        "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors",
        "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2601-8steps.safetensors",
    ] {
        if let Some(path) = crate::hub::cached_file(CONTROL_REPO, filename) {
            return Ok(path);
        }
    }
    anyhow::bail!(
        "ControlNet is not cached; fetch it with: bun scripts/hf-fetch.ts {CONTROL_REPO} {DEFAULT_CONTROL_FILE}"
    )
}

/// The two distilled architectures. The two noise refiners are additional to
/// these main blocks, and both emit residuals into the base noise refiners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlVariant {
    Full,
    Lite,
}

impl ControlVariant {
    pub fn from_filename(name: &str) -> anyhow::Result<Self> {
        match name {
            "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2601-8steps.safetensors"
            | "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors" => Ok(Self::Full),
            "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps.safetensors"
            | "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors" => {
                Ok(Self::Lite)
            }
            _ => anyhow::bail!(
                "unsupported ControlNet {name:?}; use the Fun Union 2.1 2601 or 2602 full/lite -8steps file"
            ),
        }
    }

    pub fn blocks(self) -> usize {
        match self {
            Self::Full => 15,
            Self::Lite => 3,
        }
    }

    pub fn file_bytes(self) -> u64 {
        match self {
            Self::Full => 6_712_485_600,
            Self::Lite => 2_016_627_488,
        }
    }

    pub fn injection_indices(self) -> impl Iterator<Item = usize> {
        (0..self.blocks()).map(move |i| i * (30 / self.blocks()))
    }
}

#[derive(Debug)]
struct ControlBlock {
    block: ZImageTransformerBlock,
    before: Option<Projection>,
    after: Projection,
}

impl ControlBlock {
    fn new(cfg: &Config, first: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            block: ZImageTransformerBlock::new(cfg, true, vb.clone())?,
            before: if first {
                Some(Projection::new(
                    cfg.dim,
                    cfg.dim,
                    true,
                    vb.pp("before_proj"),
                    cfg.linear_impl(),
                )?)
            } else {
                None
            },
            after: Projection::new(
                cfg.dim,
                cfg.dim,
                true,
                vb.pp("after_proj"),
                cfg.linear_impl(),
            )?,
        })
    }

    fn forward(
        &self,
        c: &Tensor,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        adaln: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let c = match &self.before {
            Some(before) => (before.forward(c)? + x)?,
            None => c.clone(),
        };
        let c = self.block.forward(&c, None, cos, sin, Some(adaln))?;
        Ok((self.after.forward(&c)?, c))
    }

    fn projections(&self) -> Vec<&Projection> {
        let mut projections = self.block.projections();
        projections.extend(self.before.iter());
        projections.push(&self.after);
        projections
    }
}

/// Prepared side state after the two image-only control refiners.
pub(crate) struct RefinerSamples {
    pub hidden: Tensor,
    pub residuals: Vec<Tensor>,
}

#[derive(Debug)]
pub struct ControlNet {
    embedding: Projection,
    refiners: Vec<ControlBlock>,
    layers: Vec<ControlBlock>,
    variant: ControlVariant,
    weight_range: WeightRange,
}

impl ControlNet {
    /// Load a registry-named, distilled 2601/2602 checkpoint. Safetensors shape
    /// checks and the f16 range gate cover every side projection at load.
    pub fn load(path: &Path, cfg: &Config, device: &Device) -> anyhow::Result<Self> {
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .context("ControlNet filename must be UTF-8")?;
        let variant = ControlVariant::from_filename(name)?;
        ensure!(
            std::fs::metadata(path)?.len() == variant.file_bytes(),
            "{name}: size does not match the vetted distilled checkpoint"
        );
        ensure!(
            cfg.dim == 3840
                && cfg.n_layers == 30
                && cfg.n_refiner_layers == 2
                && cfg.n_heads == 30
                && cfg.n_kv_heads == 30
                && cfg.in_channels == 16
                && cfg.all_patch_size == [2]
                && cfg.all_f_patch_size == [1],
            "Fun Union requires the Z-Image-Turbo architecture"
        );
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        Ok(Self::new(cfg, variant, vb)?)
    }

    fn new(cfg: &Config, variant: ControlVariant, vb: VarBuilder) -> Result<Self> {
        // 33 channels * 2x2 patches is 132, outside the tensor gemm's
        // multiple-of-32 input contract. This small projection uses candle.
        let embedding = Projection::new(
            CONTROL_CHANNELS * 4,
            cfg.dim,
            true,
            vb.pp("control_all_x_embedder").pp("2-1"),
            LinearImpl::Candle,
        )?;
        let refiners = (0..2)
            .map(|i| ControlBlock::new(cfg, i == 0, vb.pp("control_noise_refiner").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let layers = (0..variant.blocks())
            .map(|i| ControlBlock::new(cfg, i == 0, vb.pp("control_layers").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let mut projections = vec![&embedding];
        for block in refiners.iter().chain(&layers) {
            projections.extend(block.projections());
        }
        let weight_range = ensure_weights_fit_f16(projections, vb.device())?;
        Ok(Self {
            embedding,
            refiners,
            layers,
            variant,
            weight_range,
        })
    }

    #[cfg(test)]
    pub(crate) fn synthetic(cfg: &Config, vb: VarBuilder, refiner_only: bool) -> Result<Self> {
        let mut model = Self::new(cfg, ControlVariant::Lite, vb)?;
        if refiner_only {
            for layer in &mut model.layers {
                layer.after = Projection::from_weights(
                    Tensor::zeros(
                        (cfg.dim, cfg.dim),
                        DType::F32,
                        model.embedding.weight().device(),
                    )?,
                    None,
                    "test.zero_after",
                    LinearImpl::Candle,
                )?;
            }
        }
        Ok(model)
    }

    pub fn variant(&self) -> ControlVariant {
        self.variant
    }
    pub fn weight_range(&self) -> &WeightRange {
        &self.weight_range
    }

    /// Context is `[control latent (16), keep mask (1), masked-source latent (16)]`.
    /// Both latent planes are already Flux-scaled; source masking happens in
    /// pixels before encoding. Missing auxiliary planes contain literal zeros.
    pub fn validate_context(context: &Tensor, latent: &Tensor) -> Result<()> {
        let (b, _, f, h, w) = latent.dims5()?;
        if b != 1 || f != 1 || context.dims() != [1, CONTROL_CHANNELS, 1, h, w] {
            candle_core::bail!(
                "ControlNet context must be [1, 33, 1, {h}, {w}], got {:?}",
                context.dims()
            );
        }
        if context.dtype() != DType::F32 {
            candle_core::bail!("ControlNet context must be f32, got {:?}", context.dtype());
        }
        Ok(())
    }

    pub(crate) fn refine(
        &self,
        context: &Tensor,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        adaln: &Tensor,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<RefinerSamples> {
        check()?;
        let (patches, _) = patchify(context, 2, 1)?;
        let mut c = self.embedding.forward(&patches)?;
        let mut residuals = Vec::with_capacity(2);
        for block in &self.refiners {
            check()?;
            let (skip, next) = block.forward(&c, x, cos, sin, adaln)?;
            residuals.push(skip);
            c = next;
        }
        Ok(RefinerSamples {
            hidden: c,
            residuals,
        })
    }

    pub(crate) fn main_samples(
        &self,
        image_hidden: &Tensor,
        cap: &Tensor,
        unified: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        adaln: &Tensor,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Vec<Tensor>> {
        check()?;
        let mut c = Tensor::cat(&[image_hidden, cap], 1)?;
        let mut residuals = Vec::with_capacity(self.layers.len());
        for block in &self.layers {
            check()?;
            let (skip, next) = block.forward(&c, unified, cos, sin, adaln)?;
            residuals.push(skip);
            c = next;
        }
        Ok(residuals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distilled_files_select_the_trained_injection_schedule() {
        assert_eq!(
            ControlVariant::Lite.injection_indices().collect::<Vec<_>>(),
            [0, 10, 20]
        );
        assert_eq!(
            ControlVariant::Full.injection_indices().collect::<Vec<_>>(),
            (0..30).step_by(2).collect::<Vec<_>>()
        );
        assert!(
            ControlVariant::from_filename("Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors")
                .is_err()
        );
        assert!(
            ControlVariant::from_filename(
                "Z-Image-Turbo-Fun-Controlnet-Tile-2.1-2601-8steps.safetensors"
            )
            .is_err()
        );
    }

    #[test]
    fn context_requires_all_33_planes_at_the_latent_size() {
        let dev = Device::Cpu;
        let x = Tensor::zeros((1, 16, 1, 8, 16), DType::F32, &dev).unwrap();
        let context = Tensor::zeros((1, 33, 1, 8, 16), DType::F32, &dev).unwrap();
        ControlNet::validate_context(&context, &x).unwrap();
        assert!(ControlNet::validate_context(&x, &x).is_err());
        let wrong_size = Tensor::zeros((1, 33, 1, 16, 8), DType::F32, &dev).unwrap();
        assert!(ControlNet::validate_context(&wrong_size, &x).is_err());
    }
    #[test]
    fn control_blocks_add_the_base_once_and_emit_projected_skips() {
        let dev = Device::Cpu;
        let mut cfg = Config::z_image_turbo();
        cfg.dim = 48;
        cfg.n_heads = 1;
        cfg.n_kv_heads = 1;
        cfg.attn_impl = super::super::transformer::AttnImpl::Basic;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let mut first = ControlBlock::new(&cfg, true, vb.clone()).unwrap();
        let eye = Tensor::eye(48, DType::F32, &dev).unwrap();
        first.before = Some(
            Projection::from_weights((&eye * 2.).unwrap(), None, "before", LinearImpl::Candle)
                .unwrap(),
        );
        first.after =
            Projection::from_weights((&eye * 3.).unwrap(), None, "after", LinearImpl::Candle)
                .unwrap();
        let c = Tensor::ones((1, 32, 48), DType::F32, &dev).unwrap();
        let x = (&c * 4.).unwrap();
        let cos = Tensor::ones((32, 24), DType::F32, &dev).unwrap();
        let sin = Tensor::zeros((32, 24), DType::F32, &dev).unwrap();
        let adaln = Tensor::zeros((1, 48), DType::F32, &dev).unwrap();
        let (skip, hidden) = first.forward(&c, &x, &cos, &sin, &adaln).unwrap();
        assert!(
            hidden
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|v| *v == 6.)
        );
        assert!(
            skip.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|v| *v == 18.)
        );
        let mut second = ControlBlock::new(&cfg, false, vb).unwrap();
        second.after =
            Projection::from_weights((&eye * -1.).unwrap(), None, "after", LinearImpl::Candle)
                .unwrap();
        let (skip, hidden) = second
            .forward(&hidden, &(&x * 100.).unwrap(), &cos, &sin, &adaln)
            .unwrap();
        assert!(
            hidden
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|v| *v == 6.)
        );
        assert!(
            skip.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|v| *v == -6.)
        );
    }
}
