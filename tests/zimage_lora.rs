//! A real split-QKV adapter against diffusers' fp32 merge and denoising.
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;
use xwen::zimage::{
    lora::{LoraSpec, PreparedLoras},
    pipeline::{ImageOptions, ZImagePipeline},
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
#[ignore = "requires --stage lora fixture, cached base and the real adapter"]
fn real_split_attention_lora_matches_diffusers() -> Result<()> {
    let fixture = std::env::var_os("XWEN_LORA_REF")
        .map(std::path::PathBuf::from)
        .context("set XWEN_LORA_REF to the lora directory from --stage lora")?;
    let root = fixture.join("merged");
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("meta.json"))?)?;
    let loras = PreparedLoras::prepare(&[LoraSpec {
        name: meta["lora_file"].as_str().context("lora path")?.into(),
        weight: meta["lora_weight"].as_f64().context("lora weight")?,
    }])?;
    let checkpoint =
        xwen::hub::cached_model(xwen::hub::Model::ZImageTurbo).context("fetch base first")?;
    let snapshot = checkpoint.parent().unwrap();
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(
        snapshot.join("transformer/diffusion_pytorch_model.safetensors.index.json"),
    )?)?;
    let key = "layers.0.attention.to_q.weight";
    let shard = index["weight_map"][key]
        .as_str()
        .context("attention shard")?;
    {
        let path = snapshot.join("transformer").join(shard);
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &Device::Cpu)? };
        let base = vb.get((3840, 3840), key)?;
        let (merged, _) = loras.wrap(vb);
        let merged = merged.get((3840, 3840), key)?;
        let expected = tensor(&fixture, "attention-q-merged.safetensors", "weight")?;
        let s = spread(&merged, &expected)?;
        eprintln!("real Q plane merge cosine={:.12} mean_rel={:.9}", s.0, s.1);
        ensure!(
            s.0 > 0.999999 && s.1 < 0.00001,
            "real attention merge differs from diffusers"
        );
        let delta = (&merged - &base)?;
        let reference_delta = tensor(&fixture, "attention-q-delta.safetensors", "weight")?;
        let s = spread(&delta, &reference_delta)?;
        eprintln!(
            "real Q adapter delta cosine={:.12} mean_rel={:.9}",
            s.0, s.1
        );
        ensure!(
            s.0 > 0.99999 && s.1 < 0.001,
            "split Q adapter delta is missing or incorrectly scaled"
        );
    }
    let pipeline = ZImagePipeline::load_with_loras(snapshot, &xwen::gguf::metal_device()?, &loras)?;
    let cap = tensor(&root, "cap_feats.safetensors", "cap_feats")?;
    let opts = ImageOptions {
        width: 512,
        height: 512,
        steps: 8,
        seed: 0,
        latents: Some(tensor(&root, "noise.safetensors", "latents")?),
    };
    let rendered = pipeline.generate(&cap, &opts)?;
    let velocity = spread(
        rendered.velocity0.as_ref().context("missing velocity")?,
        &tensor(&root, "velocity.safetensors", "velocity")?,
    )?;
    eprintln!(
        "LoRA first velocity cosine={:.9} mean_rel={:.7}",
        velocity.0, velocity.1
    );
    ensure!(
        velocity.0 >= 0.998 && velocity.1 <= 0.04,
        "LoRA velocity fails existing transformer gate"
    );
    let final_spread = spread(
        &rendered.final_latents,
        &tensor(&root, "final.safetensors", "latents")?,
    )?;
    eprintln!(
        "LoRA final latent cosine={:.9} mean_rel={:.7}",
        final_spread.0, final_spread.1
    );
    let expected = xwen::zimage::inputs::prepare_image(
        &std::fs::read(root.join("image.png"))?,
        Some((512, 512)),
    )?
    .flatten_all()?
    .to_vec1::<u8>()?;
    let actual = rendered.image.flatten_all()?.to_vec1::<u8>()?;
    let mse = actual
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum::<f64>()
        / actual.len() as f64;
    eprintln!(
        "LoRA generated image PSNR={:.3} dB",
        10. * (255.0f64.powi(2) / mse).log10()
    );
    xwen::zimage::pipeline::write_png(&rendered.image, &root.join("image-xwen.png"))?;
    Ok(())
}
