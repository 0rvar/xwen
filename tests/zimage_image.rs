//! The first end-to-end check of the Z-Image-Turbo pipeline: one 1024x1024
//! image from a fixed prompt at a fixed seed, judged for being a picture at
//! all rather than for being the right one — there is no reference image yet
//! (the reference comparison is the next arc, with injected latents).
//!
//! ```text
//! cargo test --release --test zimage_image -- --ignored --nocapture
//! ```
//!
//! `XWEN_ZIMAGE_DIR` may point at a snapshot root (the directory holding
//! `model_index.json`); otherwise the registry entry's cached snapshot is
//! used, and the test says to run `xwen fetch --model-size zimage-turbo` when
//! it is absent. The PNG lands in `$TMPDIR/xwen-zimage-test.png` for a look.

use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use xwen::hub::Model;
use xwen::zimage::pipeline::{ImageOptions, ZImagePipeline, write_png};

fn snapshot_root() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("XWEN_ZIMAGE_DIR") {
        let dir = PathBuf::from(dir);
        ensure!(
            dir.is_dir(),
            "$XWEN_ZIMAGE_DIR {} is not a directory",
            dir.display()
        );
        return Ok(dir);
    }
    let index = xwen::hub::cached_model(Model::ZImageTurbo).context(
        "Z-Image-Turbo is not in the Hugging Face cache: run \
         `xwen fetch --model-size zimage-turbo`, or point $XWEN_ZIMAGE_DIR at a snapshot root",
    )?;
    Ok(index
        .parent()
        .context("the cached model_index.json has no parent directory")?
        .to_path_buf())
}

#[test]
#[ignore = "needs the Z-Image checkpoint (33 GB) and a Metal device"]
fn zimage_turbo_produces_a_non_degenerate_image() -> Result<()> {
    let root = snapshot_root()?;
    let encoder_entry = Model::ZImageTurbo.text_encoder().unwrap();
    let spec = encoder_entry.encoder_spec().unwrap();
    let device = xwen::gguf::metal_device().context("this test needs the Metal device")?;

    // Conditioning, from the encoder entry exactly as `xwen image` opens it.
    let encoder_dir = root.join("text_encoder");
    let source = xwen::CheckpointSource::open(&encoder_dir, &device, Some(encoder_entry))?;
    let set = source
        .safetensors()
        .context("text_encoder is a safetensors set")?
        .clone();
    let tokenizer = xwen::tokenizer::LagunaTokenizer::from_file(set.tokenizer_path())?;
    let chat_opts = xwen::chat::ChatOptions::for_dialect(encoder_entry.chat_dialect());
    let prompt = "A photo of a red bicycle leaning against a white brick wall, golden hour";
    let (text, ranges, _) = xwen::chat::build_prompt_with_spans(
        &[xwen::chat::Message::User(prompt.to_string())],
        &chat_opts,
    )?;
    let ids = tokenizer.encode_prompt(&text, &ranges)?;
    let mut encoder = xwen::XwenModel::load_encoder(source, spec.max_tokens)?;
    let (cap_feats, n_tokens) = encoder.encode(&ids, spec.layer)?;
    assert_eq!(n_tokens, ids.len());

    let pipeline = ZImagePipeline::load(&root, &device)?;
    let opts = ImageOptions {
        width: 1024,
        height: 1024,
        steps: 8,
        seed: 0,
        latents: None,
    };
    let (image, timings) = pipeline.generate(&cap_feats, &opts)?;
    eprintln!("steps {:?} vae {:.2}s", timings.steps, timings.vae_decode);
    assert_eq!(image.dims(), &[3, 1024, 1024]);

    let out = std::env::temp_dir().join("xwen-zimage-test.png");
    write_png(&image, &out)?;
    eprintln!("wrote {}", out.display());

    // Non-degenerate: every channel has spread, and the image is neither
    // black nor white nor grey — a broken graph produces one of those.
    let f = image.to_dtype(candle_core::DType::F32)?;
    for c in 0..3 {
        let plane = f.get(c)?.flatten_all()?;
        let n = plane.elem_count() as f64;
        let mean = plane.sum_all()?.to_scalar::<f32>()? as f64 / n;
        let var = plane
            .broadcast_sub(&plane.mean_all()?)?
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()? as f64
            / n;
        let std = var.sqrt();
        eprintln!("channel {c}: mean {mean:.1} std {std:.1}");
        ensure!(std > 20.0, "channel {c} has std {std:.2}: a flat image");
        ensure!(
            (10.0..245.0).contains(&mean),
            "channel {c} has mean {mean:.1}: saturated"
        );
    }
    // Not noise either: neighbouring pixels correlate in a picture.
    let row = f.get(0)?.get(512)?;
    let a = row.narrow(0, 0, 1023)?;
    let b = row.narrow(0, 1, 1023)?;
    let mean_abs_diff = (a - b)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    eprintln!("mean |dx| on the middle row of R: {mean_abs_diff:.2}");
    ensure!(
        mean_abs_diff < 40.0,
        "neighbouring pixels differ by {mean_abs_diff:.1} on average: noise"
    );
    Ok(())
}
