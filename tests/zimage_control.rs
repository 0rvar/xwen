//! Author VideoX-Fun ControlNet parity with explicit conditioning tensors.
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use std::path::Path;
use xwen::zimage::{
    controlnet, inputs,
    lora::PreparedLoras,
    pipeline::{ImageControl, ImageEdit, ImageOptions, ZImagePipeline},
};

fn tensor(root: &Path, file: &str, key: &str) -> Result<Tensor> {
    candle_core::safetensors::load(root.join(file), &Device::Cpu)?
        .remove(key)
        .context("missing reference tensor")
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
#[ignore = "requires --stage control reference, cached base and distilled ControlNet"]
fn control_plain_and_inpaint_match_author() -> Result<()> {
    let fixtures = std::env::var_os("XWEN_CONTROL_REF")
        .map(std::path::PathBuf::from)
        .context("set XWEN_CONTROL_REF to the control directory from --stage control")?;
    let checkpoint = xwen::hub::cached_model(xwen::hub::Model::ZImageTurbo)
        .context("fetch Z-Image-Turbo first")?;
    let control_file = controlnet::cached_default()?;
    let dev = xwen::gguf::metal_device()?;
    let pipeline = ZImagePipeline::load_with_loras_and_control(
        checkpoint.parent().unwrap(),
        &dev,
        &PreparedLoras::load(&[])?,
        Some(&control_file),
    )?;
    for case in ["plain", "inpaint"] {
        let root = fixtures.join(case);
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("meta.json"))?)?;
        ensure!(
            meta["author_revision"] == "968f0e2192ba4c7a12868bf36d73260d135424ca",
            "unexpected author oracle"
        );
        let reference_file = Path::new(
            meta["control_file"]
                .as_str()
                .context("reference ControlNet file")?,
        );
        ensure!(
            reference_file.file_name() == control_file.file_name(),
            "loaded ControlNet differs from the fixture; set XWEN_CONTROLNET_FILE to {}",
            reference_file.display()
        );
        let control = ImageControl {
            image: inputs::prepare_image(
                &std::fs::read(root.join("control.png"))?,
                Some((512, 512)),
            )?,
            scale: 0.75,
            start: 0.,
            end: 1.,
        };
        let edit = if case == "inpaint" {
            Some(ImageEdit {
                init_image: inputs::prepare_image(
                    &std::fs::read(root.join("source.png"))?,
                    Some((512, 512)),
                )?,
                mask: Some(inputs::prepare_mask(
                    &std::fs::read(root.join("mask.png"))?,
                    512,
                    512,
                    0.,
                )?),
                strength: 1.,
                posterior_noise: None,
            })
        } else {
            None
        };
        let context = pipeline.control_context(&control, edit.as_ref())?;
        let expected = tensor(&root, "context.safetensors", "context")?;
        let context_spread = spread(&context, &expected)?;
        eprintln!(
            "{case} context cosine={:.9} mean_rel={:.7}",
            context_spread.0, context_spread.1
        );
        ensure!(
            context_spread.0 >= 0.99999 && context_spread.1 < 0.001,
            "33-channel VAE/mask contract differs"
        );
        let cap = tensor(&root, "cap_feats.safetensors", "cap_feats")?;
        let opts = ImageOptions {
            width: 512,
            height: 512,
            steps: 8,
            seed: 0,
            latents: Some(tensor(&root, "noise.safetensors", "latents")?),
        };
        let rendered = pipeline.generate_controlled(&cap, &opts, edit.as_ref(), &control)?;
        let velocity = spread(
            rendered
                .velocity0
                .as_ref()
                .context("missing first velocity")?,
            &tensor(&root, "velocity.safetensors", "velocity")?,
        )?;
        eprintln!(
            "{case} first velocity cosine={:.9} mean_rel={:.7}",
            velocity.0, velocity.1
        );
        ensure!(
            velocity.0 >= 0.998 && velocity.1 <= 0.04,
            "ControlNet velocity fails existing transformer gate"
        );
        let final_spread = spread(
            &rendered.final_latents,
            &tensor(&root, "final.safetensors", "latents")?,
        )?;
        eprintln!(
            "{case} final latent cosine={:.9} mean_rel={:.7}; image {:?}",
            final_spread.0,
            final_spread.1,
            rendered.image.dims()
        );
        let reference_image =
            inputs::prepare_image(&std::fs::read(root.join("image.png"))?, Some((512, 512)))?;
        let reference_decode = pipeline.decode(&tensor(&root, "final.safetensors", "latents")?)?;
        let expected = reference_image.flatten_all()?.to_vec1::<u8>()?;
        let decoded = reference_decode.flatten_all()?.to_vec1::<u8>()?;
        let mse = decoded
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum::<f64>()
            / expected.len() as f64;
        let vae_psnr = 10. * (255.0f64.powi(2) / mse).log10();
        eprintln!("{case} reference-latent VAE PSNR={vae_psnr:.3} dB");
        ensure!(vae_psnr >= 60., "VAE reference decode differs");
        let generated = rendered.image.flatten_all()?.to_vec1::<u8>()?;
        let mask = edit
            .as_ref()
            .and_then(|e| e.mask.as_ref())
            .map(|m| m.flatten_all()?.to_vec1::<f32>())
            .transpose()?;
        let mut total = 0.;
        let mut count = 0;
        for (i, (&a, &b)) in generated.iter().zip(&expected).enumerate() {
            if mask.as_ref().is_none_or(|m| m[i % m.len()] != 0.) {
                total += (a as f64 - b as f64).powi(2);
                count += 1;
            }
        }
        eprintln!(
            "{case} generated-region image PSNR={:.3} dB",
            10. * (255.0f64.powi(2) / (total / count as f64)).log10()
        );
        xwen::zimage::pipeline::write_png(&rendered.image, &root.join("image-xwen.png"))?;
        if case == "plain" {
            let plain = pipeline.velocity(opts.latents.as_ref().unwrap(), &cap, 0.)?;
            let zero = pipeline.velocity_controlled(
                opts.latents.as_ref().unwrap(),
                &cap,
                0.,
                &context,
                0.,
            )?;
            let a = plain
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let b = zero
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            ensure!(
                a == b,
                "zero-scale controlled velocity must exactly recover the base"
            );
            eprintln!("full real model zero-scale velocity is bitwise equal to base");
        }
    }
    Ok(())
}
