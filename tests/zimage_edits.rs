//! Image-edit parity against the actual diffusers pipelines with saved random draws.
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use std::path::Path;
use xwen::zimage::{
    inputs,
    pipeline::{ImageEdit, ImageOptions, ZImagePipeline},
};

fn tensor(root: &Path, file: &str, key: &str) -> Result<Tensor> {
    candle_core::safetensors::load(root.join(file), &Device::Cpu)?
        .remove(key)
        .with_context(|| format!("{file}: no {key}"))
}

fn spread(actual: &Tensor, expected: &Tensor) -> Result<(f64, f64)> {
    ensure!(actual.dims() == expected.dims(), "tensor shapes differ");
    let a = actual
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let b = expected
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let (mut dot, mut aa, mut bb, mut error) = (0., 0., 0., 0.);
    for (&a, &b) in a.iter().zip(&b) {
        let (a, b) = (a as f64, b as f64);
        dot += a * b;
        aa += a * a;
        bb += b * b;
        error += (a - b).abs();
    }
    Ok((
        dot / (aa * bb).sqrt(),
        error / a.len() as f64 / (bb / a.len() as f64).sqrt(),
    ))
}

#[test]
#[ignore = "requires cached Z-Image weights and scripts/zimage-ref-dump.py --stage edits"]
fn edited_images_match_diffusers_with_explicit_random_draws() -> Result<()> {
    let fixtures = std::env::var_os("XWEN_IMAGE_EDIT_REF")
        .map(std::path::PathBuf::from)
        .context("set XWEN_IMAGE_EDIT_REF to the edits directory from --stage edits")?;
    let checkpoint = xwen::hub::cached_model(xwen::hub::Model::ZImageTurbo)
        .context("fetch Z-Image-Turbo first")?;
    let pipeline =
        ZImagePipeline::load(checkpoint.parent().unwrap(), &xwen::gguf::metal_device()?)?;
    for mode in ["img2img", "inpaint"] {
        let dir = fixtures.join(mode);
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
        let width = meta["width"].as_u64().unwrap() as usize;
        let height = meta["height"].as_u64().unwrap() as usize;
        let source = inputs::prepare_image(
            &std::fs::read(dir.join("source.png"))?,
            Some((width, height)),
        )?;
        let mask = if mode == "inpaint" {
            Some(inputs::prepare_mask(
                &std::fs::read(dir.join("mask.png"))?,
                width,
                height,
                0.,
            )?)
        } else {
            None
        };
        let cap = tensor(&dir, "cap_feats.safetensors", "cap_feats")?;
        let posterior = tensor(&dir, "posterior-noise.safetensors", "noise")?;
        let encoded = pipeline.encode_image(&source, &posterior)?;
        let expected_source = tensor(&dir, "source-latents.safetensors", "latents")?;
        let (cos, rel) = spread(&encoded, &expected_source)?;
        eprintln!("{mode}: VAE encode cosine={cos:.9} mean_rel={rel:.7}");
        ensure!(
            cos >= 0.99999 && rel < 0.001,
            "VAE encoder differs from fp32 reference"
        );
        let opts = ImageOptions {
            width,
            height,
            steps: meta["steps"].as_u64().unwrap() as usize,
            seed: meta["seed"].as_u64().unwrap(),
            latents: Some(tensor(&dir, "noise.safetensors", "latents")?),
        };
        let edit = ImageEdit {
            init_image: source.clone(),
            mask: mask.clone(),
            strength: meta["strength"].as_f64().unwrap(),
            posterior_noise: Some(posterior),
        };
        let rendered = pipeline.generate_edited(&cap, &opts, &edit)?;
        assert_eq!(
            rendered.start_step,
            meta["start_step"].as_u64().unwrap() as usize
        );
        assert_eq!(
            rendered.timings.steps.len(),
            opts.steps - rendered.start_step
        );
        let (cos, rel) = spread(
            rendered
                .velocity0
                .as_ref()
                .context("missing first velocity")?,
            &tensor(&dir, "velocity.safetensors", "velocity")?,
        )?;
        eprintln!("{mode}: first velocity cosine={cos:.9} mean_rel={rel:.7}");
        ensure!(
            cos >= 0.998 && rel <= 0.04,
            "edit velocity fails existing transformer parity bar"
        );
        let final_spread = spread(
            &rendered.final_latents,
            &tensor(&dir, "final.safetensors", "latents")?,
        )?;
        eprintln!(
            "{mode}: final latent cosine={} mean_rel={}",
            final_spread.0, final_spread.1
        );
        if let Some(mask) = mask {
            let generated = rendered.image.flatten_all()?.to_vec1::<u8>()?;
            let original = source.flatten_all()?.to_vec1::<u8>()?;
            let mask = mask.flatten_all()?.to_vec1::<f32>()?;
            let mut preserved = 0;
            for (i, (&a, &b)) in generated.iter().zip(&original).enumerate() {
                if mask[i % mask.len()] == 0. {
                    assert_eq!(a, b);
                    preserved += 1;
                }
            }
            ensure!(preserved > 0, "fixture must exercise preserved pixels");
        }
    }
    Ok(())
}
