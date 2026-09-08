//! Stages 3 and 4 of the Z-Image verification (docs/zimage.md): the diffusion
//! transformer against diffusers' fp32 run of the same weights on the same
//! two inputs, and the decoded image against the reference PNG.
//!
//! ```text
//! cargo test --release --test zimage_parity -- --ignored --nocapture
//! ```
//!
//! Each case under `tests/fixtures/zimage-transformer/` was written by
//! `scripts/zimage-ref-dump.py --stage transformer` and holds the noise both
//! sides start from (`latents0`), the caption both sides read (`cap_feats`,
//! the encoder's hidden state rounded once to bf16, so the encoder is out of
//! the picture and Stage 2 stays its only gate), and the reference's step-0
//! velocity, final latent and image, all from the transformer in fp32 on mps.
//! `meta.json` also carries the reference's OWN gap when it reruns the
//! transformer in bf16 on the same inputs, which is what the bars below were
//! set from.
//!
//! What is gated and what is reported, per docs/decisions/zimage.md:
//!
//! - Stage 3, the step-0 velocity field, is the gate. One forward at sigma 1
//!   exercises all 34 blocks, the three-axis rope, every modulation and the
//!   final layer, and its inputs are known exactly on both sides. The bar is
//!   bracketed: the same forward with the timestep one grid point off, and
//!   with the caption tokens in reverse order, must both land outside it.
//! - Stage 4, the final latent and the image after eight Euler steps, is
//!   reported. Eight steps compound the bf16 differences, so a bar there
//!   would be a bar on the reference's own arithmetic as much as on ours.
//! - The VAE alone is gated too, at PSNR: the reference's final latent
//!   decoded through this VAE against the reference PNG. Both decode in f32,
//!   so the number is a check on the decoder and on nothing else.
//!
//! `XWEN_ZIMAGE_DIR` may point at a snapshot root; otherwise the registry
//! entry's cached snapshot is used.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use xwen::hub::Model;
use xwen::zimage::pipeline::{ImageOptions, ZImagePipeline, write_png};

/// Stage 3 bars on xwen's bf16 velocity against the fp32 reference, set
/// 2026-09-07 from the 512x512 case: the reference's own bf16 arm sits at
/// cosine 0.99956 and mean relative error 0.0175, xwen at 0.99930 and
/// 0.0205, the timestep one grid point off at 0.60 / 0.63 and the reversed
/// caption at 0.86 / 0.33. The bars sit at about three times the bf16
/// arithmetic's own loss and a hundred times under the nearest bracket, so
/// they separate a wrong graph from a rounding difference without flapping
/// on the latter.
const VELOCITY_COS_MIN: f64 = 0.998;
const VELOCITY_MEAN_REL_MAX: f64 = 0.04;

/// The VAE-only bar: the reference latent decoded here against the
/// reference's own decode of it. Both run in f32 and the measured figure is
/// 92.6 dB (2026-09-07), a handful of pixels one level off; 60 dB is where a
/// decoder that is merely close, rather than the same computation, lands.
const VAE_PSNR_MIN_DB: f64 = 60.0;

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

#[derive(serde::Deserialize)]
struct Meta {
    case: String,
    width: usize,
    height: usize,
    steps: usize,
    sigmas: Vec<f64>,
    bf16_vs_fp32: RefSpread,
}

#[derive(serde::Deserialize)]
struct RefSpread {
    velocity0: Spread,
    final_latents: Spread,
    image_psnr_db: f64,
}

#[derive(serde::Deserialize, Debug, Clone, Copy)]
struct Spread {
    cosine: f64,
    max_rel: f64,
    mean_rel: f64,
}

impl Spread {
    /// Whole-field cosine, `max|a-b| / max|b|` and `mean|a-b| / rms(b)`, in
    /// f64, the same three numbers the dump script computes for the
    /// reference's own bf16 arm.
    fn between(a: &Tensor, b: &Tensor) -> Result<Self> {
        let a = a.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let b = b.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        ensure!(a.len() == b.len(), "{} vs {} elements", a.len(), b.len());
        let (mut dot, mut na, mut nb, mut max_abs, mut sum_abs, mut ref_max) =
            (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        for (&x, &r) in a.iter().zip(&b) {
            let (x, r) = (x as f64, r as f64);
            dot += x * r;
            na += x * x;
            nb += r * r;
            let d = (x - r).abs();
            max_abs = max_abs.max(d);
            sum_abs += d;
            ref_max = ref_max.max(r.abs());
        }
        let n = a.len() as f64;
        Ok(Self {
            cosine: dot / (na.sqrt() * nb.sqrt()),
            max_rel: max_abs / ref_max,
            mean_rel: (sum_abs / n) / (nb / n).sqrt(),
        })
    }

    fn passes_velocity_bar(&self) -> bool {
        self.cosine >= VELOCITY_COS_MIN && self.mean_rel <= VELOCITY_MEAN_REL_MAX
    }
}

fn load_one(path: &Path, key: &str) -> Result<Tensor> {
    let tensors = candle_core::safetensors::load(path, &Device::Cpu)
        .with_context(|| format!("reading {}", path.display()))?;
    tensors
        .get(key)
        .cloned()
        .with_context(|| format!("{} has no `{key}` tensor", path.display()))
}

/// Read an RGB8 PNG into `[3, H, W]` u8, the layout the pipeline returns.
fn read_png(path: &Path) -> Result<Tensor> {
    let decoder = png::Decoder::new(std::io::BufReader::new(
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?,
    ));
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size().context("PNG too large")?];
    let info = reader.next_frame(&mut buf)?;
    ensure!(
        info.color_type == png::ColorType::Rgb && info.bit_depth == png::BitDepth::Eight,
        "{} is not RGB8",
        path.display()
    );
    let (h, w) = (info.height as usize, info.width as usize);
    let hwc = Tensor::from_vec(buf[..info.buffer_size()].to_vec(), (h, w, 3), &Device::Cpu)?;
    Ok(hwc.permute((2, 0, 1))?.contiguous()?)
}

/// PSNR in dB between two `[3, H, W]` u8 images.
fn psnr_db(a: &Tensor, b: &Tensor) -> Result<f64> {
    ensure!(a.dims() == b.dims(), "{:?} vs {:?}", a.dims(), b.dims());
    let a = a.flatten_all()?.to_vec1::<u8>()?;
    let b = b.flatten_all()?.to_vec1::<u8>()?;
    let mse = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    Ok(if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64.powi(2) / mse).log10()
    })
}

struct Row {
    label: &'static str,
    spread: Spread,
    must_pass: bool,
}

fn grade_case(pipeline: &ZImagePipeline, dir: &Path) -> Result<Vec<String>> {
    let meta: Meta = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?)
        .with_context(|| format!("parsing {}/meta.json", dir.display()))?;
    let latents0 =
        ZImagePipeline::read_latents(&dir.join("latents0.safetensors"), meta.width, meta.height)?;
    let cap_feats = ZImagePipeline::read_cap_feats(&dir.join("cap_feats.safetensors"))?;
    let ref_velocity = load_one(&dir.join("velocity0-fp32.safetensors"), "velocity")?;
    let ref_final = load_one(&dir.join("latents-final-fp32.safetensors"), "latents")?;
    let ref_image = read_png(&dir.join("image-fp32.png"))?;
    ensure!(
        meta.sigmas.len() == meta.steps + 1,
        "{}: sigma grid",
        meta.case
    );
    let (t_cap, _) = cap_feats.dims2()?;
    let mut failures = Vec::new();

    // Stage 3: one forward at sigma 1, then the two brackets.
    let t0 = (1.0 - meta.sigmas[0]) as f32;
    let velocity = pipeline.velocity(&latents0, &cap_feats, t0)?;
    let real = Spread::between(&velocity, &ref_velocity)?;

    let t_off = (1.0 - meta.sigmas[1]) as f32;
    let off_grid = pipeline.velocity(&latents0, &cap_feats, t_off)?;
    let off_grid = Spread::between(&off_grid, &ref_velocity)?;

    let reversed_idx = Tensor::from_vec(
        (0..t_cap as u32).rev().collect::<Vec<_>>(),
        t_cap,
        &Device::Cpu,
    )?;
    let reversed = cap_feats.index_select(&reversed_idx, 0)?;
    let reversed = pipeline.velocity(&latents0, &reversed, t0)?;
    let reversed = Spread::between(&reversed, &ref_velocity)?;

    let rows = [
        Row {
            label: "xwen bf16 vs fp32 reference",
            spread: real,
            must_pass: true,
        },
        Row {
            label: "reference bf16 vs fp32 (its own spread)",
            spread: meta.bf16_vs_fp32.velocity0,
            must_pass: true,
        },
        Row {
            label: "bracket: timestep one grid point off",
            spread: off_grid,
            must_pass: false,
        },
        Row {
            label: "bracket: caption tokens reversed",
            spread: reversed,
            must_pass: false,
        },
    ];
    eprintln!(
        "\n{} ({}x{}, {} caption tokens): step-0 velocity, bar cosine >= {VELOCITY_COS_MIN} \
         and mean rel <= {VELOCITY_MEAN_REL_MAX}",
        meta.case, meta.width, meta.height, t_cap
    );
    eprintln!(
        "  {:<44} {:>10} {:>9} {:>9}  bar",
        "arm", "cosine", "mean_rel", "max_rel"
    );
    for row in &rows {
        let passes = row.spread.passes_velocity_bar();
        eprintln!(
            "  {:<44} {:>10.6} {:>9.4} {:>9.4}  {}",
            row.label,
            row.spread.cosine,
            row.spread.mean_rel,
            row.spread.max_rel,
            if passes { "inside" } else { "outside" }
        );
        if passes != row.must_pass {
            failures.push(format!(
                "{}: {} is {} the velocity bar (cosine {:.6}, mean rel {:.4})",
                meta.case,
                row.label,
                if passes { "inside" } else { "outside" },
                row.spread.cosine,
                row.spread.mean_rel
            ));
        }
    }

    // Stage 4: the full run, reported.
    let rendered = pipeline.generate(
        &cap_feats,
        &ImageOptions {
            width: meta.width,
            height: meta.height,
            steps: meta.steps,
            seed: 0,
            latents: Some(latents0.clone()),
        },
    )?;
    let velocity_in_run = Spread::between(
        rendered
            .velocity0
            .as_ref()
            .context("generation must produce a velocity")?,
        &velocity,
    )?;
    ensure!(
        velocity_in_run.max_rel < 1e-6,
        "{}: the run's step-0 velocity differs from the single forward's (max rel {:.2e}), \
         so the two are not the same computation",
        meta.case,
        velocity_in_run.max_rel
    );
    let final_spread = Spread::between(&rendered.final_latents, &ref_final)?;
    let image_psnr = psnr_db(&rendered.image, &ref_image)?;
    let out = std::env::temp_dir().join(format!("xwen-zimage-parity-{}.png", meta.case));
    write_png(&rendered.image, &out)?;
    eprintln!(
        "  final latent after {} steps: cosine {:.6} mean rel {:.4} (reference bf16 arm: \
         {:.6} / {:.4}); image PSNR {:.2} dB (reference bf16 arm: {:.2} dB); wrote {}",
        meta.steps,
        final_spread.cosine,
        final_spread.mean_rel,
        meta.bf16_vs_fp32.final_latents.cosine,
        meta.bf16_vs_fp32.final_latents.mean_rel,
        image_psnr,
        meta.bf16_vs_fp32.image_psnr_db,
        out.display()
    );

    // The VAE alone: the reference latent through this decoder.
    let vae_only = pipeline.decode(&ref_final)?;
    let vae_psnr = psnr_db(&vae_only, &ref_image)?;
    eprintln!(
        "  VAE alone (reference latent through this decoder): PSNR {:.2} dB, bar >= {VAE_PSNR_MIN_DB}",
        vae_psnr
    );
    if vae_psnr < VAE_PSNR_MIN_DB {
        failures.push(format!(
            "{}: the VAE decode of the reference latent is {:.2} dB from the reference image, \
             under {VAE_PSNR_MIN_DB}",
            meta.case, vae_psnr
        ));
    }
    Ok(failures)
}

#[test]
#[ignore = "needs the Z-Image checkpoint (25 GB) and a Metal device"]
fn zimage_transformer_matches_the_fp32_reference() -> Result<()> {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/zimage-transformer");
    let mut cases: Vec<PathBuf> = std::fs::read_dir(&fixtures)
        .with_context(|| format!("listing {}", fixtures.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("meta.json").is_file())
        .collect();
    cases.sort();
    ensure!(!cases.is_empty(), "no cases under {}", fixtures.display());

    let root = snapshot_root()?;
    let device = xwen::gguf::metal_device().context("this test needs the Metal device")?;
    let pipeline = ZImagePipeline::load(&root, &device)?;

    let mut failures = Vec::new();
    for dir in &cases {
        failures.extend(grade_case(&pipeline, dir)?);
    }
    ensure!(
        failures.is_empty(),
        "{} finding(s):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
    Ok(())
}

/// The metric helpers, on the CPU without a checkpoint.
#[cfg(test)]
mod metric_tests {
    use super::*;

    #[test]
    fn spread_of_a_tensor_with_itself_is_exact() -> Result<()> {
        let t = Tensor::from_vec(vec![1.0f32, -2.0, 3.5, 0.25], 4, &Device::Cpu)?;
        let s = Spread::between(&t, &t)?;
        assert!((s.cosine - 1.0).abs() < 1e-12);
        assert_eq!(s.max_rel, 0.0);
        assert_eq!(s.mean_rel, 0.0);
        Ok(())
    }

    #[test]
    fn spread_measures_a_scaled_copy_the_way_the_dump_script_does() -> Result<()> {
        let r = Tensor::from_vec(vec![1.0f32, -2.0, 4.0, 0.0], 4, &Device::Cpu)?;
        let a = (&r * 1.1)?;
        let s = Spread::between(&a, &r)?;
        assert!((s.cosine - 1.0).abs() < 1e-9, "{}", s.cosine);
        // max |a-r| = 0.4 over max |r| = 4.
        assert!((s.max_rel - 0.1).abs() < 1e-6, "{}", s.max_rel);
        // mean |a-r| = 0.7/4 over rms(r) = sqrt(21/4).
        let want = (0.7 / 4.0) / (21.0f64 / 4.0).sqrt();
        assert!((s.mean_rel - want).abs() < 1e-6, "{}", s.mean_rel);
        Ok(())
    }

    #[test]
    fn psnr_of_identical_images_is_infinite_and_one_step_off_is_48_db() -> Result<()> {
        let a = Tensor::from_vec(vec![10u8; 3 * 4 * 4], (3, 4, 4), &Device::Cpu)?;
        assert!(psnr_db(&a, &a)?.is_infinite());
        let b = Tensor::from_vec(vec![11u8; 3 * 4 * 4], (3, 4, 4), &Device::Cpu)?;
        let db = psnr_db(&a, &b)?;
        assert!((db - 48.1308).abs() < 1e-3, "{db}");
        Ok(())
    }

    #[test]
    fn png_round_trip_keeps_the_pixels() -> Result<()> {
        let pixels: Vec<u8> = (0..3 * 6 * 5).map(|i| (i * 7 % 256) as u8).collect();
        let img = Tensor::from_vec(pixels, (3, 6, 5), &Device::Cpu)?;
        let dir = std::env::temp_dir().join(format!("xwen-parity-png-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("rt.png");
        write_png(&img, &path)?;
        let back = read_png(&path)?;
        assert_eq!(back.dims(), img.dims());
        assert_eq!(
            back.flatten_all()?.to_vec1::<u8>()?,
            img.flatten_all()?.to_vec1::<u8>()?
        );
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }
}
