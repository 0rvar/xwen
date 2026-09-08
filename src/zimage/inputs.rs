//! Pixel preparation shared by image clients and the diffusion loop.
use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};
use image::imageops::{self, FilterType};

/// Snap to the nearest valid grid by squared pixel displacement, then area.
pub fn snap_size(width: usize, height: usize) -> Result<(usize, usize)> {
    ensure!(width > 0 && height > 0, "image dimensions must be positive");
    let mut best = None;
    for rows in 1usize..=512 {
        for cols in 1usize..=512 {
            if !(rows * cols).is_multiple_of(32) {
                continue;
            }
            let w = cols * 16;
            let h = rows * 16;
            let dx = w.abs_diff(width) as u128;
            let dy = h.abs_diff(height) as u128;
            let rank = (
                dx.saturating_mul(dx).saturating_add(dy.saturating_mul(dy)),
                w * h,
                w,
                h,
            );
            if best.is_none_or(|old| rank < old) {
                best = Some(rank);
            }
        }
    }
    let (_, _, w, h) = best.expect("there are valid image grids");
    Ok((w, h))
}

/// Decode PNG/JPEG pixels and resize to an explicit or snapped source size.
pub fn prepare_image(bytes: &[u8], size: Option<(usize, usize)>) -> Result<Tensor> {
    let image = image::load_from_memory(bytes)?.to_rgb8();
    let (w, h) = match size {
        Some(size) => size,
        None => snap_size(image.width() as usize, image.height() as usize)?,
    };
    super::pipeline::ZImagePipeline::check_size(w, h)?;
    let image = imageops::resize(&image, w as u32, h as u32, FilterType::Lanczos3);
    Ok(Tensor::from_vec(image.into_raw(), (h, w, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .contiguous()?)
}

/// White repaints. Blur is measured in pixels at the requested output size.
pub fn prepare_mask(bytes: &[u8], width: usize, height: usize, blur: f32) -> Result<Tensor> {
    ensure!(
        blur.is_finite() && blur >= 0.0,
        "mask blur must be finite and nonnegative"
    );
    super::pipeline::ZImagePipeline::check_size(width, height)?;
    let gray = image::load_from_memory(bytes)?.to_luma8();
    let mut gray = imageops::resize(&gray, width as u32, height as u32, FilterType::Lanczos3);
    if blur > 0.0 {
        gray = imageops::blur(&gray, blur);
    }
    let values: Vec<f32> = gray
        .into_raw()
        .into_iter()
        .map(|x| x as f32 / 255.0)
        .collect();
    Ok(Tensor::from_vec(values, (1, height, width), &Device::Cpu)?)
}

/// Resize a floating mask without quantizing its soft edges to bytes.
pub fn resize_mask(mask: &Tensor, width: usize, height: usize) -> Result<Tensor> {
    let (channels, h, w) = mask.dims3()?;
    ensure!(
        channels == 1 && width > 0 && height > 0,
        "invalid mask dimensions"
    );
    let data = mask
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    // torch interpolate(mode="nearest") samples floor(dst * src_size / dst_size),
    // without the half-pixel offset image libraries use for nearest resizing.
    let mut resized = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            resized.push(data[(y * h / height) * w + x * w / width]);
        }
    }
    Ok(Tensor::from_vec(resized, (1, height, width), &Device::Cpu)?.clamp(0.0, 1.0)?)
}

/// Composite in pixel space, preserving source bytes exactly wherever mask is zero.
pub fn composite(generated: &Tensor, source: &Tensor, mask: &Tensor) -> Result<Tensor> {
    ensure!(
        generated.dims() == source.dims(),
        "composite images must have matching dimensions"
    );
    let (_, h, w) = source.dims3()?;
    ensure!(
        source.dim(0)? == 3 && mask.dims() == [1, h, w],
        "invalid composite shape"
    );
    let a = generated
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<u8>()?;
    let b = source
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<u8>()?;
    let m = mask
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let out: Vec<u8> = a
        .iter()
        .zip(b)
        .enumerate()
        .map(|(i, (&a, b))| {
            let m = m[i % (w * h)];
            if m == 0.0 {
                b
            } else if m == 1.0 {
                a
            } else {
                (a as f32 * m + b as f32 * (1.0 - m))
                    .round()
                    .clamp(0.0, 255.0) as u8
            }
        })
        .collect();
    Ok(Tensor::from_vec(out, (3, h, w), &Device::Cpu)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn latent_masks_sample_the_top_left_of_each_source_cell() -> Result<()> {
        let mask = Tensor::from_vec(vec![0f32, 0.25, 0.75, 1.0], (1, 1, 4), &Device::Cpu)?;
        assert_eq!(
            resize_mask(&mask, 2, 1)?.flatten_all()?.to_vec1::<f32>()?,
            [0.0, 0.75]
        );
        Ok(())
    }
    #[test]
    fn snapping_preserves_valid_sizes_and_finds_nearest_grid() -> Result<()> {
        assert_eq!(snap_size(896, 1152)?, (896, 1152));
        let (w, h) = snap_size(513, 511)?;
        assert_eq!((w, h), (512, 512));
        super::super::pipeline::ZImagePipeline::check_size(w, h)?;
        Ok(())
    }
    #[test]
    fn pixel_composite_preserves_black_mask_bytes() -> Result<()> {
        let a = Tensor::from_vec(vec![255u8; 6], (3, 1, 2), &Device::Cpu)?;
        let b = Tensor::from_vec(vec![1u8, 3, 5, 7, 9, 11], (3, 1, 2), &Device::Cpu)?;
        let m = Tensor::from_vec(vec![0f32, 1.], (1, 1, 2), &Device::Cpu)?;
        assert_eq!(
            composite(&a, &b, &m)?.flatten_all()?.to_vec1::<u8>()?,
            vec![1, 255, 5, 255, 9, 255]
        );
        Ok(())
    }
}
