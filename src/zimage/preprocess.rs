//! Control-map preparation. Models load from the HF cache on first use; requests never download.
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Module, Tensor};
use image::{GrayImage, Luma, Rgb, RgbImage, imageops::FilterType};
use std::{path::PathBuf, sync::Arc};
mod depth_model;
mod dino;
mod pose;
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    None,
    Canny,
    Pose,
    Depth,
}
impl std::str::FromStr for Kind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "none" => Ok(Self::None),
            "canny" => Ok(Self::Canny),
            "pose" => Ok(Self::Pose),
            "depth" => Ok(Self::Depth),
            _ => anyhow::bail!("unknown preprocessor {s}; expected none, canny, pose or depth"),
        }
    }
}
pub struct Preprocessor {
    depth: Option<depth_model::DepthAnythingV2>,
    pose: Option<pose::Pose>,
}
impl Default for Preprocessor {
    fn default() -> Self {
        Self::new()
    }
}
impl Preprocessor {
    pub fn new() -> Self {
        Self {
            depth: None,
            pose: None,
        }
    }
    pub fn validate(kind: Kind) -> Result<()> {
        match kind {
            Kind::None | Kind::Canny => Ok(()),
            Kind::Depth => {
                cached(
                    "jeroenvlek/depth-anything-v2-safetensors",
                    "depth_anything_v2_vits.safetensors",
                )?;
                Ok(())
            }
            Kind::Pose => {
                cached("yzd-v/DWPose", "yolox_l.onnx")?;
                cached("yzd-v/DWPose", "dw-ll_ucoco_384.onnx")?;
                Ok(())
            }
        }
    }
    pub fn run(&mut self, image: &Tensor, kind: Kind) -> Result<Tensor> {
        let (channels, height, width) = image.dims3()?;
        ensure!(
            channels == 3 && height > 0 && width > 0 && image.dtype() == DType::U8,
            "preprocessor expects RGB8 [3,height,width]"
        );
        if kind == Kind::None {
            return Ok(image.clone());
        }
        let rgb = to_rgb(image)?;
        let result = match kind {
            Kind::None => unreachable!(),
            Kind::Canny => canny(&rgb),
            Kind::Pose => {
                if self.pose.is_none() {
                    self.pose = Some(pose::Pose::load()?)
                }
                self.pose.as_mut().unwrap().run(&rgb)?
            }
            Kind::Depth => {
                if self.depth.is_none() {
                    self.depth = Some(load_depth()?)
                }
                // The pinned DPT head has a square 37x37 patch grid. Fixed 518 input also
                // uses the trained DINO position embeddings without interpolation.
                let resized = image::imageops::resize(&rgb, 518, 518, FilterType::CatmullRom);
                let mut data = vec![0f32; 3 * 518 * 518];
                for (i, p) in resized.pixels().enumerate() {
                    for c in 0..3 {
                        data[c * 518 * 518 + i] = (p[c] as f32 / 255. - [0.485, 0.456, 0.406][c])
                            / [0.229, 0.224, 0.225][c]
                    }
                }
                let input = Tensor::from_vec(data, (1, 3, 518, 518), &Device::Cpu)?;
                let depth = self.depth.as_ref().unwrap().forward(&input)?;
                let depth = bilinear_tensor(&depth, height, width)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                ensure!(
                    depth.len() == height * width && depth.iter().all(|x| x.is_finite()),
                    "invalid depth prediction"
                );
                let lo = depth.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = depth.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                RgbImage::from_fn(width as u32, height as u32, |x, y| {
                    let v = if hi > lo {
                        ((depth[y as usize * width + x as usize] - lo) / (hi - lo) * 255.)
                            .clamp(0., 255.) as u8
                    } else {
                        0
                    };
                    Rgb([v, v, v])
                })
            }
        };
        from_rgb(result, image.device())
    }
}
fn cached(repo: &str, file: &str) -> Result<PathBuf> {
    let root = crate::hub::hub_cache_root().context("cannot resolve Hugging Face cache")?;
    hf_hub::Cache::new(root)
        .model(repo.to_owned())
        .get(file)
        .with_context(|| {
            format!("missing preprocessor weights; run `bun scripts/hf-fetch.ts {repo} {file}`")
        })
}
fn load_depth() -> Result<depth_model::DepthAnythingV2> {
    let path = cached(
        "jeroenvlek/depth-anything-v2-safetensors",
        "depth_anything_v2_vits.safetensors",
    )?;
    let tensors = candle_core::safetensors::load(path, &Device::Cpu)?;
    let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let backbone = Arc::new(dino::vit_small(vb.pp("pretrained"))?);
    Ok(depth_model::DepthAnythingV2::new(
        backbone,
        depth_model::DepthAnythingV2Config::vit_small(),
        vb,
    )?)
}
fn to_rgb(t: &Tensor) -> Result<RgbImage> {
    let (_, h, w) = t.dims3()?;
    let bytes = t
        .to_device(&Device::Cpu)?
        .permute((1, 2, 0))?
        .flatten_all()?
        .to_vec1::<u8>()?;
    RgbImage::from_raw(w as u32, h as u32, bytes).context("invalid RGB dimensions")
}
fn from_rgb(rgb: RgbImage, dev: &Device) -> Result<Tensor> {
    let (w, h) = rgb.dimensions();
    Ok(
        Tensor::from_vec(rgb.into_raw(), (h as usize, w as usize, 3), dev)?
            .permute((2, 0, 1))?
            .contiguous()?,
    )
}
/// PyTorch bilinear interpolation with align_corners=true, used by the depth author.
fn bilinear_tensor(t: &Tensor, oh: usize, ow: usize) -> candle_core::Result<Tensor> {
    let (b, c, h, w) = t.dims4()?;
    let input = t.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0.; b * c * oh * ow];
    for y in 0..oh {
        let fy = if oh > 1 {
            y as f32 * (h - 1) as f32 / (oh - 1) as f32
        } else {
            0.
        };
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(h - 1);
        let dy = fy - y0 as f32;
        for x in 0..ow {
            let fx = if ow > 1 {
                x as f32 * (w - 1) as f32 / (ow - 1) as f32
            } else {
                0.
            };
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let dx = fx - x0 as f32;
            for p in 0..b * c {
                let row = &input[p * h * w..];
                out[p * oh * ow + y * ow + x] =
                    (row[y0 * w + x0] * (1. - dx) + row[y0 * w + x1] * dx) * (1. - dy)
                        + (row[y1 * w + x0] * (1. - dx) + row[y1 * w + x1] * dx) * dy;
            }
        }
    }
    Tensor::from_vec(out, (b, c, oh, ow), t.device())
}
fn canny(rgb: &RgbImage) -> RgbImage {
    let (w, h) = rgb.dimensions();
    let n = (w * h) as usize;
    if w < 3 || h < 3 {
        return RgbImage::new(w, h);
    }
    let gray = GrayImage::from_fn(w, h, |x, y| {
        let p = rgb.get_pixel(x, y);
        Luma([(0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).round() as u8])
    });
    let gray = image::imageops::blur(&gray, 1.0);
    let w = w as usize;
    let h = h as usize;
    let mut mag = vec![0f32; n];
    let mut dir = vec![0u8; n];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let p = |dx: isize, dy: isize| {
                gray.get_pixel((x as isize + dx) as u32, (y as isize + dy) as u32)[0] as f32
            };
            let gx = -p(-1, -1) + p(1, -1) - 2. * p(-1, 0) + 2. * p(1, 0) - p(-1, 1) + p(1, 1);
            let gy = -p(-1, -1) - 2. * p(0, -1) - p(1, -1) + p(-1, 1) + 2. * p(0, 1) + p(1, 1);
            mag[y * w + x] = gx.hypot(gy);
            let angle = gy.atan2(gx).to_degrees().rem_euclid(180.);
            dir[y * w + x] = if !(22.5..157.5).contains(&angle) {
                0
            } else if angle < 67.5 {
                1
            } else if angle < 112.5 {
                2
            } else {
                3
            };
        }
    }
    let mut state = vec![0u8; n];
    let mut queue = Vec::new();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let (a, b) = match dir[i] {
                0 => (i - 1, i + 1),
                1 => (i - w - 1, i + w + 1),
                2 => (i - w, i + w),
                _ => (i - w + 1, i + w - 1),
            };
            if mag[i] >= mag[a] && mag[i] >= mag[b] && mag[i] >= 100. {
                state[i] = 1;
                if mag[i] >= 200. {
                    state[i] = 2;
                    queue.push(i)
                }
            }
        }
    }
    while let Some(i) = queue.pop() {
        let y = i / w;
        let x = i % w;
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                let yy = y as isize + dy;
                let xx = x as isize + dx;
                if yy >= 0 && xx >= 0 && yy < h as isize && xx < w as isize {
                    let j = yy as usize * w + xx as usize;
                    if state[j] == 1 {
                        state[j] = 2;
                        queue.push(j)
                    }
                }
            }
        }
    }
    RgbImage::from_fn(w as u32, h as u32, |x, y| {
        Rgb([if state[y as usize * w + x as usize] == 2 {
            255
        } else {
            0
        }; 3])
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canny_finds_rectangle_edges() {
        let rgb = RgbImage::from_fn(64, 64, |x, y| {
            Rgb([if (16..48).contains(&x) && (16..48).contains(&y) {
                255
            } else {
                0
            }; 3])
        });
        let map = canny(&rgb);
        let count = map.pixels().filter(|p| p[0] != 0).count();
        assert!((80..400).contains(&count), "edge count {count}");
        assert_eq!(map.get_pixel(32, 32)[0], 0);
    }
    #[test]
    fn bilinear_preserves_corners() {
        let t = Tensor::new(&[[[[0f32, 2.], [4., 6.]]]], &Device::Cpu).unwrap();
        let o = bilinear_tensor(&t, 3, 3)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(o, vec![0., 1., 2., 2., 3., 4., 4., 5., 6.]);
    }
    #[test]
    fn no_preprocessing_preserves_pixels() {
        let t = Tensor::zeros((3, 2, 2), DType::U8, &Device::Cpu).unwrap();
        assert_eq!(
            Preprocessor::new().run(&t, Kind::None).unwrap().dims(),
            t.dims()
        );
    }
    #[test]
    #[ignore = "requires cached depth and pose models; CPU inference"]
    fn real_preprocessors() {
        let file = std::env::var("XWEN_PREPROCESS_TEST_IMAGE").unwrap_or_else(|_| {
            "tests/fixtures/zimage-transformer/512x512-p1-s0/image-fp32.png".into()
        });
        let rgb = image::open(file).unwrap().to_rgb8();
        let input = from_rgb(rgb, &Device::Cpu).unwrap();
        let mut p = Preprocessor::new();
        for kind in [Kind::Depth, Kind::Pose] {
            let started = std::time::Instant::now();
            let map = p.run(&input, kind).unwrap();
            let bytes = map.flatten_all().unwrap().to_vec1::<u8>().unwrap();
            assert!(bytes.iter().any(|v| *v != 0));
            to_rgb(&map)
                .unwrap()
                .save(format!("/tmp/xwen-preprocess-{kind:?}.png"))
                .unwrap();
            eprintln!("{kind:?}: {:?}", started.elapsed());
        }
    }

    #[test]
    #[ignore = "requires author depth fixture generated by zimage-ref-dump.py"]
    fn author_depth_reference() {
        let root = PathBuf::from(
            std::env::var("XWEN_PREPROCESS_REF_DIR")
                .unwrap_or_else(|_| "tests/fixtures/zimage-preprocess".into()),
        );
        let input =
            candle_core::safetensors::load(root.join("depth-input.safetensors"), &Device::Cpu)
                .unwrap()
                .remove("input")
                .unwrap();
        let reference =
            candle_core::safetensors::load(root.join("depth-reference.safetensors"), &Device::Cpu)
                .unwrap()
                .remove("depth")
                .unwrap();
        let output = load_depth().unwrap().forward(&input).unwrap();
        let a = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        let squared_error: f64 = a
            .iter()
            .zip(&b)
            .map(|(a, b)| ((*a as f64) - (*b as f64)).powi(2))
            .sum();
        let norm: f64 = b.iter().map(|v| (*v as f64).powi(2)).sum();
        let relative_l2 = (squared_error / norm).sqrt();
        eprintln!("Depth author relative L2 {relative_l2}");
        assert!(
            relative_l2 < 0.001,
            "depth author relative L2 {relative_l2}"
        );
    }
}
