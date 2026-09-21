//! The Qwen-Image 2.1 transformer and VAE against diffusers' fp32 run of the
//! same weights on the same two inputs, and the decoded image against the
//! reference PNG.
//!
//! ```text
//! cargo test --release --test qwen_image_parity -- --ignored --nocapture
//! ```
//!
//! Each case under `tests/fixtures/qwen-image-transformer/` was written by
//! `scripts/qwen-image-ref-dump.py --stage transformer` and holds the noise
//! both sides start from (`latents0`, unpacked and unscaled), the caption both
//! sides read (`cap_feats`, the encoder's kept pre-norm rows rounded once to
//! bf16, so the encoder is out of the picture and `tests/qwen_image_encoder.rs`
//! stays its only gate), and the reference's step-0 velocity, final latent and
//! image, from `QwenImage21Pipeline.__call__` itself with the transformer in
//! fp32 and `use_kv_cache=True`. `meta.json` also carries the reference's OWN
//! gap when it reruns the transformer in bf16 on the same inputs, which the
//! bars below are read against.
//!
//! What is gated and what is reported:
//!
//! - The step-0 velocity field is the gate. One forward at sigma 1 exercises
//!   all 32 blocks, the three-axis rope, the shared modulation with its t=0
//!   row, the block-causal attention and the final layer, and its inputs are
//!   known exactly on both sides. The bar is bracketed by two wrong graphs
//!   that must land outside it: attention fully bidirectional over the joint
//!   sequence, and the text rows modulated from the real timestep.
//! - The sigma grid and `mu` are gated against the reference's, to 1e-6.
//! - The final latent and the image after the full schedule are reported.
//!   Forty steps compound the bf16 differences, so a bar there would be a bar
//!   on the reference's own arithmetic as much as on ours. The run uses the
//!   shipped prefix-cache arm, as the reference does.
//! - The VAE alone is gated at PSNR: the reference's final latent decoded
//!   through this VAE against the reference PNG's RGB planes. Both decode in
//!   f32, so the number is a check on the decoder and on nothing else.
//!
//! A missing fixture or checkpoint FAILS, naming the command that supplies
//! it: the test is opt-in, and a gate that skips is not a gate.
//!
//! `XWEN_QWEN_IMAGE_DIR` may point at a snapshot root; otherwise the registry
//! entry's cached snapshot is used.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use xwen::hub::Model;
use xwen::qwen_image::pipeline::{ImageOptions, QwenImagePipeline, write_png};
use xwen::qwen_image::transformer::GraphVariant;

/// The step-0 bars on xwen's velocity against the fp32 reference: Z-Image's,
/// whose transformer runs on the same kernels with the same bf16 weights and
/// f32 stream. They are a claim about this model only while they sit OUTSIDE
/// the reference's own bf16-against-fp32 spread and inside both brackets, and
/// the test asserts all three every run rather than trusting the numbers.
const VELOCITY_COS_MIN: f64 = 0.998;
const VELOCITY_MEAN_REL_MAX: f64 = 0.04;

/// The VAE-only bar: the reference latent decoded here against the
/// reference's own decode of it, both in f32. 60 dB is where a decoder that
/// is merely close, rather than the same computation, lands.
const VAE_PSNR_MIN_DB: f64 = 60.0;

const SIGMA_TOLERANCE: f64 = 1e-6;

const DUMP_COMMANDS: &str = "/tmp/qwen-image-venv/bin/python scripts/qwen-image-ref-dump.py \
     --stage transformer --dtype fp32 && /tmp/qwen-image-venv/bin/python \
     scripts/qwen-image-ref-dump.py --stage transformer --dtype bf16";

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
         `xwen fetch --model qwen-image-2.1`, or point $XWEN_QWEN_IMAGE_DIR at a snapshot \
         root",
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
    use_kv_cache: bool,
    sigmas: Vec<f64>,
    mu: f64,
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
    /// reference's own bf16 arm. A nonfinite value on either side is an
    /// error: it would otherwise fall out of the maxima and read as agreement.
    fn between(a: &Tensor, b: &Tensor) -> Result<Self> {
        let a = a.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let b = b.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        ensure!(a.len() == b.len(), "{} vs {} elements", a.len(), b.len());
        ensure!(
            a.iter().chain(&b).all(|x| x.is_finite()),
            "a compared tensor holds a nonfinite value"
        );
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

/// Read the RGB planes of an RGB8 or RGBA8 PNG into `[3, H, W]` u8, the layout
/// the pipeline returns. The reference writes RGBA; the alpha plane is not
/// part of the comparison.
fn read_png_rgb(path: &Path) -> Result<Tensor> {
    let decoder = png::Decoder::new(std::io::BufReader::new(
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?,
    ));
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size().context("PNG too large")?];
    let info = reader.next_frame(&mut buf)?;
    ensure!(
        info.bit_depth == png::BitDepth::Eight,
        "{} is not 8-bit",
        path.display()
    );
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => anyhow::bail!("{} is {other:?}, not RGB or RGBA", path.display()),
    };
    let (h, w) = (info.height as usize, info.width as usize);
    let hwc = Tensor::from_vec(
        buf[..info.buffer_size()].to_vec(),
        (h, w, channels),
        &Device::Cpu,
    )?;
    Ok(hwc.narrow(2, 0, 3)?.permute((2, 0, 1))?.contiguous()?)
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

fn grade_case(pipeline: &QwenImagePipeline, dir: &Path) -> Result<Vec<String>> {
    let meta: Meta = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?)
        .with_context(|| format!("parsing {}/meta.json", dir.display()))?;
    ensure!(
        meta.use_kv_cache,
        "{}: the reference ran without its prefix cache, which is not what ships",
        meta.case
    );
    let latents0 = QwenImagePipeline::read_latents(
        &dir.join("latents0.safetensors"),
        meta.width,
        meta.height,
    )?;
    let cap_feats = QwenImagePipeline::read_cap_feats(&dir.join("cap_feats.safetensors"))?;
    let ref_velocity = load_one(&dir.join("velocity0-fp32.safetensors"), "velocity")?;
    let ref_final = load_one(&dir.join("latents-final-fp32.safetensors"), "latents")?;
    let ref_image = read_png_rgb(&dir.join("image-fp32.png"))?;
    let (t_cap, _) = cap_feats.dims2()?;
    let mut failures = Vec::new();

    // The schedule: the reference's grid, entry for entry, and its shift.
    let (sigmas, mu) = pipeline.sigmas(meta.steps, meta.width, meta.height)?;
    ensure!(
        sigmas.len() == meta.sigmas.len() && meta.sigmas.len() == meta.steps + 1,
        "{}: {} sigmas here, {} in the reference, for {} steps",
        meta.case,
        sigmas.len(),
        meta.sigmas.len(),
        meta.steps
    );
    let worst_sigma = sigmas
        .iter()
        .zip(&meta.sigmas)
        .map(|(&ours, &theirs)| (f64::from(ours) - theirs).abs())
        .fold(0.0, f64::max);
    eprintln!(
        "\n{} ({}x{}, {} caption tokens): sigma grid max |diff| {:.2e}, mu {:.6} vs {:.6}",
        meta.case, meta.width, meta.height, t_cap, worst_sigma, mu, meta.mu
    );
    if worst_sigma > SIGMA_TOLERANCE {
        failures.push(format!(
            "{}: the sigma grid is {worst_sigma:.2e} from the reference's, over {SIGMA_TOLERANCE}",
            meta.case
        ));
    }
    if (mu - meta.mu).abs() > SIGMA_TOLERANCE {
        failures.push(format!(
            "{}: mu {mu} against the reference's {}",
            meta.case, meta.mu
        ));
    }

    // One forward at the first sigma, then the two wrong graphs.
    let sigma0 = meta.sigmas[0] as f32;
    let velocity = pipeline.velocity(&cap_feats, &latents0, sigma0, GraphVariant::Reference)?;
    let real = Spread::between(&velocity, &ref_velocity)?;
    let bidirectional = pipeline.velocity(
        &cap_feats,
        &latents0,
        sigma0,
        GraphVariant::FullyBidirectional,
    )?;
    let bidirectional = Spread::between(&bidirectional, &ref_velocity)?;
    let real_t = pipeline.velocity(
        &cap_feats,
        &latents0,
        sigma0,
        GraphVariant::RealTimestepForText,
    )?;
    let real_t = Spread::between(&real_t, &ref_velocity)?;

    let rows = [
        Row {
            label: "xwen vs fp32 reference",
            spread: real,
            must_pass: true,
        },
        Row {
            label: "reference bf16 vs fp32 (its own spread)",
            spread: meta.bf16_vs_fp32.velocity0,
            must_pass: true,
        },
        Row {
            label: "bracket: fully bidirectional attention",
            spread: bidirectional,
            must_pass: false,
        },
        Row {
            label: "bracket: text modulated from the real t",
            spread: real_t,
            must_pass: false,
        },
    ];
    eprintln!(
        "  step-0 velocity, bar cosine >= {VELOCITY_COS_MIN} and mean rel <= \
         {VELOCITY_MEAN_REL_MAX}"
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

    // The full run on the shipped arm, reported.
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
    let out = std::env::temp_dir().join(format!("xwen-qwen-image-parity-{}.png", meta.case));
    write_png(&rendered.image, &out)?;
    let steps = &rendered.timings.steps;
    eprintln!(
        "  final latent after {} steps ({} arm): cosine {:.6} mean rel {:.4} (reference bf16 \
         arm: {:.6} / {:.4}); image PSNR {:.2} dB (reference bf16 arm: {:.2} dB); alpha strays \
         {:.4} from opaque; wrote {}",
        meta.steps,
        pipeline.cache_arm().label(),
        final_spread.cosine,
        final_spread.mean_rel,
        meta.bf16_vs_fp32.final_latents.cosine,
        meta.bf16_vs_fp32.final_latents.mean_rel,
        image_psnr,
        meta.bf16_vs_fp32.image_psnr_db,
        rendered.alpha_max_distance_from_opaque,
        out.display()
    );
    eprintln!(
        "  step seconds: first {:.2}, second {:.2}, last {:.2}; vae decode {:.2}",
        steps[0],
        steps.get(1).copied().unwrap_or(f64::NAN),
        steps[steps.len() - 1],
        rendered.timings.vae_decode
    );

    // The VAE alone: the reference latent through this decoder.
    let vae_only = pipeline.decode_latents(&ref_final)?;
    let vae_psnr = psnr_db(&vae_only.image, &ref_image)?;
    eprintln!(
        "  VAE alone (reference latent through this decoder): PSNR {vae_psnr:.2} dB, bar >= \
         {VAE_PSNR_MIN_DB}"
    );
    if vae_psnr < VAE_PSNR_MIN_DB {
        failures.push(format!(
            "{}: the VAE decode of the reference latent is {vae_psnr:.2} dB from the reference \
             image, under {VAE_PSNR_MIN_DB}",
            meta.case
        ));
    }
    Ok(failures)
}

#[test]
#[ignore = "needs the Qwen-Image 2.1 checkpoint (33 GB) and a Metal device"]
fn qwen_image_transformer_matches_the_fp32_reference() -> Result<()> {
    let fixtures =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen-image-transformer");
    let mut cases: Vec<PathBuf> = std::fs::read_dir(&fixtures)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.join("meta.json").is_file())
                .collect()
        })
        .unwrap_or_default();
    cases.sort();
    ensure!(
        !cases.is_empty(),
        "no cases under {}: write one with\n  {DUMP_COMMANDS}",
        fixtures.display()
    );

    let root = snapshot_root()?;
    let device = xwen::gguf::metal_device().context("this test needs the Metal device")?;
    let pipeline = QwenImagePipeline::load(&root, &device)?;

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
    fn spread_refuses_a_nonfinite_value() -> Result<()> {
        let r = Tensor::from_vec(vec![1.0f32, 2.0], 2, &Device::Cpu)?;
        let a = Tensor::from_vec(vec![1.0f32, f32::NAN], 2, &Device::Cpu)?;
        assert!(Spread::between(&a, &r).is_err());
        assert!(Spread::between(&r, &a).is_err());
        Ok(())
    }

    #[test]
    fn the_bars_refuse_each_side_separately() {
        let inside = Spread {
            cosine: 0.9995,
            max_rel: 0.5,
            mean_rel: 0.02,
        };
        assert!(inside.passes_velocity_bar());
        assert!(
            !Spread {
                cosine: 0.99,
                ..inside
            }
            .passes_velocity_bar()
        );
        assert!(
            !Spread {
                mean_rel: 0.05,
                ..inside
            }
            .passes_velocity_bar()
        );
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
    fn an_rgba_png_reads_back_as_its_rgb_planes() -> Result<()> {
        let (h, w) = (5usize, 6usize);
        let rgba: Vec<u8> = (0..h * w * 4).map(|i| (i * 7 % 256) as u8).collect();
        let dir = std::env::temp_dir().join(format!("xwen-qi-parity-png-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("rgba.png");
        {
            let file = std::io::BufWriter::new(std::fs::File::create(&path)?);
            let mut encoder = png::Encoder::new(file, w as u32, h as u32);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.write_header()?.write_image_data(&rgba)?;
        }
        let back = read_png_rgb(&path)?;
        assert_eq!(back.dims(), [3, h, w]);
        let planes = back.flatten_all()?.to_vec1::<u8>()?;
        for c in 0..3 {
            for i in 0..h * w {
                assert_eq!(planes[c * h * w + i], rgba[i * 4 + c]);
            }
        }
        // An RGB PNG from the pipeline's own writer reads back unchanged.
        let rgb = Tensor::from_vec(planes.clone(), (3, h, w), &Device::Cpu)?;
        let path = dir.join("rgb.png");
        write_png(&rgb, &path)?;
        assert_eq!(read_png_rgb(&path)?.flatten_all()?.to_vec1::<u8>()?, planes);
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }
}
