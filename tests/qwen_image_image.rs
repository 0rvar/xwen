//! The end-to-end check of the Qwen-Image 2.1 pipeline: one 512x512 image from
//! a fixed prompt at a fixed seed, encoder included, judged for being a picture
//! at all. Whether it is the RIGHT picture is `tests/qwen_image_parity.rs`'s
//! question, which starts from injected caption features and so never runs the
//! path this one does: render the prompt, encode it, drop the system rows, free
//! the encoder, load the transformer, denoise, decode.
//!
//! ```text
//! cargo test --release --test qwen_image_image -- --ignored --nocapture
//! ```
//!
//! `XWEN_QWEN_IMAGE_DIR` may point at a snapshot root (the directory holding
//! `model_index.json`); otherwise the registry entry's cached snapshot is used,
//! and the test says to run `xwen fetch --model qwen-image-2.1` when it is
//! absent. Eight steps, not the release's forty: a picture is recognisable as
//! one long before it is finished, and this is not the quality check. The PNG
//! lands in `$TMPDIR/xwen-qwen-image-test.png` for a look.

use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use xwen::hub::Model;
use xwen::qwen_image::pipeline::{CLEAR_PIXELS_MIN, ImageOptions, QwenImagePipeline, write_png};

fn snapshot_root() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("XWEN_QWEN_IMAGE_DIR") {
        let dir = PathBuf::from(dir);
        ensure!(
            dir.is_dir(),
            "$XWEN_QWEN_IMAGE_DIR {} is not a directory",
            dir.display()
        );
        return Ok(dir);
    }
    let index = xwen::hub::cached_model(Model::QwenImage21).context(
        "Qwen-Image 2.1 is not in the Hugging Face cache: run \
         `xwen fetch --model qwen-image-2.1`, or point $XWEN_QWEN_IMAGE_DIR at a snapshot root",
    )?;
    Ok(index
        .parent()
        .context("the cached model_index.json has no parent directory")?
        .to_path_buf())
}

#[test]
#[ignore = "needs the Qwen-Image 2.1 checkpoint (33 GB) and a Metal device"]
fn qwen_image_produces_a_non_degenerate_image() -> Result<()> {
    const SIDE: usize = 512;
    let root = snapshot_root()?;
    let encoder_entry = Model::QwenImage21.text_encoder().unwrap();
    let spec = encoder_entry.encoder_spec().unwrap();
    let device = xwen::gguf::metal_device().context("this test needs the Metal device")?;

    // Conditioning, from the encoder entry exactly as `xwen image` opens it,
    // and gone from the device before the transformer arrives.
    let cap_feats = {
        let encoder_dir = root.join("text_encoder");
        let source = xwen::CheckpointSource::open(&encoder_dir, &device, Some(encoder_entry))?;
        let tokenizer_path = source
            .safetensors()
            .context("text_encoder is a safetensors set")?
            .tokenizer_path()
            .to_path_buf();
        let prompt = "A photo of a red bicycle leaning against a white brick wall, golden hour";
        let rendered = xwen::qwen_image::conditioning::prompt_ids(
            encoder_entry,
            &tokenizer_path,
            prompt,
            &[],
        )?;
        let mut encoder = xwen::XwenModel::load_encoder(source, spec.max_tokens)?;
        let (hidden, n_tokens) = encoder.encode_spec(&rendered.ids, &spec)?;
        assert_eq!(n_tokens, rendered.ids.len());
        hidden
            .narrow(0, rendered.drop, n_tokens - rendered.drop)?
            .to_device(&candle_core::Device::Cpu)?
            .contiguous()?
    };
    device.synchronize()?;

    let pipeline = QwenImagePipeline::load(&root, &device)?;
    let opts = ImageOptions {
        width: SIDE,
        height: SIDE,
        steps: 8,
        seed: 0,
        latents: None,
    };
    let rendered = pipeline.generate(&cap_feats, &opts)?;
    eprintln!(
        "steps {:?} vae {:.2}s; alpha min {}, {} clear pixels",
        rendered.timings.steps,
        rendered.timings.vae_decode,
        rendered.alpha_min,
        rendered.clear_pixels
    );
    // An ordinary prompt draws no transparent region, so the alpha plane the
    // decoder emits is measured and dropped.
    assert_eq!(rendered.image.dims(), &[3, SIDE, SIDE]);
    assert!(
        rendered.clear_pixels < CLEAR_PIXELS_MIN,
        "{} clear pixels on a prompt that asks for no transparency",
        rendered.clear_pixels
    );

    let out = std::env::temp_dir().join("xwen-qwen-image-test.png");
    write_png(&rendered.image, &out)?;
    eprintln!("wrote {}", out.display());

    // Non-degenerate: every channel has spread, and the image is neither
    // black nor white nor grey, a broken graph producing one of those.
    let f = rendered.image.to_dtype(candle_core::DType::F32)?;
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
        ensure!(
            std.is_finite() && std > 20.0,
            "channel {c} has std {std:.2}: a flat image"
        );
        ensure!(
            (10.0..245.0).contains(&mean),
            "channel {c} has mean {mean:.1}: saturated"
        );
    }
    // Not noise either: neighbouring pixels correlate in a picture.
    let row = f.get(0)?.get(SIDE / 2)?;
    let a = row.narrow(0, 0, SIDE - 1)?;
    let b = row.narrow(0, 1, SIDE - 1)?;
    let mean_abs_diff = (a - b)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    eprintln!("mean |dx| on the middle row of R: {mean_abs_diff:.2}");
    ensure!(
        mean_abs_diff < 40.0,
        "neighbouring pixels differ by {mean_abs_diff:.1} on average: noise"
    );
    Ok(())
}
