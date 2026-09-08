//! Native image controls and the multipart OpenAI edit surfaces.
use super::{
    ApiError, AppState,
    images::{self, ImageInputs, ImageParams},
    openai::bad_request,
};
use crate::zimage::{inputs, lora::LoraSpec, pipeline::ImageEdit};
use anyhow::{Context, Result, ensure};
use axum::{
    body::Bytes,
    extract::{Multipart, State},
    response::{IntoResponse, Response},
};
use base64::Engine as _;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ControlRequest {
    pub image: String,
    pub preprocess: Option<String>,
    pub scale: Option<f64>,
    pub start: Option<f64>,
    pub end: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RenderRequest {
    pub prompt: String,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub steps: Option<usize>,
    pub seed: Option<u64>,
    pub n: Option<u32>,
    #[serde(default)]
    pub loras: Vec<LoraSpec>,
    pub init_image: Option<String>,
    pub strength: Option<f64>,
    pub mask: Option<String>,
    pub mask_blur: Option<f32>,
    pub control: Option<ControlRequest>,
}

/// Native image strings are local files, data URIs, or bare base64.
pub(crate) fn image_bytes(value: &str) -> Result<Vec<u8>> {
    if let Some(data) = value.strip_prefix("data:") {
        let (header, body) = data.split_once(',').context("invalid image data URI")?;
        ensure!(
            header.ends_with(";base64"),
            "image data URI must use base64"
        );
        return Ok(base64::engine::general_purpose::STANDARD.decode(body)?);
    }
    let path = std::path::Path::new(value);
    if path.is_file() {
        return Ok(std::fs::read(path)?);
    }
    base64::engine::general_purpose::STANDARD
        .decode(value)
        .context("image must be an existing local path, data URI, or base64")
}

pub(crate) fn prepare(
    request: RenderRequest,
    server_steps: Option<usize>,
) -> Result<(ImageParams, ImageInputs), ApiError> {
    let fail = |name: &str, e: anyhow::Error| images::bad_param(name, format!("{e:#}"));
    if request.prompt.trim().is_empty() {
        return Err(images::bad_param(
            "prompt",
            "a non-empty prompt is required",
        ));
    }
    if request.init_image.is_none() && (request.mask.is_some() || request.strength.is_some()) {
        return Err(images::bad_param(
            "init_image",
            "mask and strength require init_image",
        ));
    }
    if request.mask_blur.is_some() && request.mask.is_none() {
        return Err(images::bad_param("mask_blur", "mask_blur requires mask"));
    }
    if request
        .strength
        .is_some_and(|strength| !strength.is_finite() || !(0.0..=1.0).contains(&strength))
    {
        return Err(images::bad_param(
            "strength",
            "strength must be finite and between 0 and 1",
        ));
    }
    if request
        .mask_blur
        .is_some_and(|blur| !blur.is_finite() || blur < 0.0)
    {
        return Err(images::bad_param(
            "mask_blur",
            "mask blur must be finite and nonnegative",
        ));
    }
    let size = match (request.width, request.height) {
        (Some(w), Some(h)) => Some((w, h)),
        (None, None) => None,
        _ => {
            return Err(images::bad_param(
                "width",
                "width and height must be supplied together",
            ));
        }
    };
    let source = request
        .init_image
        .as_deref()
        .map(|s| {
            let bytes = image_bytes(s)?;
            inputs::prepare_image(&bytes, size)
        })
        .transpose()
        .map_err(|e| fail("init_image", e))?;
    let (width, height) = match &source {
        Some(source) => (
            source.dim(2).map_err(|e| fail("init_image", e.into()))?,
            source.dim(1).map_err(|e| fail("init_image", e.into()))?,
        ),
        None => size.unwrap_or((1024, 1024)),
    };
    let compatibility: images::ImagesRequest = serde_json::from_value(json!({
        "prompt":request.prompt,"width":width,"height":height,"steps":request.steps,"seed":request.seed,"n":request.n
    })).map_err(|e| bad_request(e.to_string()))?;
    let params = images::validate(compatibility, false, server_steps)?;
    let edit = match source {
        Some(source) => {
            let mask = request
                .mask
                .as_deref()
                .map(|s| {
                    inputs::prepare_mask(
                        &image_bytes(s)?,
                        width,
                        height,
                        request.mask_blur.unwrap_or(0.),
                    )
                })
                .transpose()
                .map_err(|e| fail("mask", e))?;
            let edit = ImageEdit {
                init_image: source,
                mask,
                strength: request.strength.unwrap_or(if request.mask.is_some() {
                    1.0
                } else {
                    0.6
                }),
                posterior_noise: None,
            };
            edit.validate(width, height)
                .map_err(|e| fail("init_image", e))?;
            Some(edit)
        }
        None => None,
    };
    let loras = request
        .loras
        .iter()
        .map(LoraSpec::resolve)
        .collect::<Result<Vec<_>>>()
        .map_err(|e| fail("loras", e))?;
    let control = request
        .control
        .map(|control| -> Result<_> {
            let kind = control
                .preprocess
                .as_deref()
                .unwrap_or("none")
                .parse::<crate::zimage::preprocess::Kind>()?;
            let (scale, start, end) = (
                control.scale.unwrap_or(0.75),
                control.start.unwrap_or(0.),
                control.end.unwrap_or(0.8),
            );
            ensure!(
                scale.is_finite() && (0.0..=1.0).contains(&scale),
                "control scale must be between zero and one"
            );
            ensure!(
                start.is_finite()
                    && end.is_finite()
                    && (0.0..=1.0).contains(&start)
                    && (0.0..=1.0).contains(&end)
                    && start <= end,
                "control window must satisfy 0 <= start <= end <= 1"
            );
            let image =
                inputs::prepare_image(&image_bytes(&control.image)?, Some((width, height)))?;
            let path = match std::env::var_os("XWEN_CONTROLNET_FILE") {
                Some(path) => std::path::PathBuf::from(path),
                None => crate::zimage::controlnet::cached_default()?,
            };
            let checkpoint = control_fingerprint(&path)?;
            Ok(images::ControlInput {
                image: crate::zimage::pipeline::ImageControl {
                    image,
                    scale,
                    start,
                    end,
                },
                preprocess: kind,
                checkpoint,
            })
        })
        .transpose()
        .map_err(|e| fail("control", e))?;
    Ok((
        params,
        ImageInputs {
            edit,
            loras,
            control,
        },
    ))
}

pub(crate) async fn render(State(state): State<AppState>, body: Bytes) -> Response {
    let request = match serde_json::from_slice::<RenderRequest>(&body) {
        Ok(r) => r,
        Err(e) => {
            return bad_request(format!("could not parse image render request: {e}"))
                .into_response();
        }
    };
    match prepare(request, state.settings.image_steps) {
        Ok((params, inputs)) => images::submit_image(state, params, inputs).await,
        Err(e) => e.into_response(),
    }
}

async fn multipart_fields(mut multipart: Multipart) -> Result<HashMap<String, Vec<u8>>, ApiError> {
    let mut fields = HashMap::new();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| bad_request(format!("invalid multipart body: {e}")))?
    {
        let name = field.name().unwrap_or("").to_owned();
        let name = if name == "image[]" {
            "image".to_owned()
        } else {
            name
        };
        let bytes = field
            .bytes()
            .await
            .map_err(|e| bad_request(format!("invalid multipart field: {e}")))?;
        if fields.insert(name.clone(), bytes.to_vec()).is_some() {
            return Err(images::bad_param(
                &name,
                "duplicate multipart field; this route accepts one input image",
            ));
        }
    }
    Ok(fields)
}

fn prepare_multipart(
    mut fields: HashMap<String, Vec<u8>>,
    variation: bool,
    server_steps: Option<usize>,
) -> Result<(ImageParams, ImageInputs), ApiError> {
    let bytes = fields
        .remove("image")
        .ok_or_else(|| images::bad_param("image", "image is required"))?;
    let mask = fields.remove("mask");
    if variation && mask.is_some() {
        return Err(images::bad_param("mask", "variations do not accept a mask"));
    }
    let mut value = serde_json::Map::new();
    for (name, bytes) in fields {
        let text = String::from_utf8(bytes)
            .map_err(|_| images::bad_param(&name, "text field must be UTF-8"))?;
        let field = match name.as_str() {
            "n" => json!(
                text.parse::<u32>()
                    .map_err(|_| images::bad_param(&name, "n must be an integer"))?
            ),
            "prompt" | "model" | "size" | "response_format" | "output_format" => json!(text),
            _ => continue,
        };
        value.insert(name, field);
    }
    let prompt = value
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if !variation && prompt.trim().is_empty() {
        return Err(images::bad_param("prompt", "edits require a prompt"));
    }
    // Compatibility validation requires a prompt; variations intentionally encode an empty one.
    if prompt.is_empty() {
        value.insert("prompt".into(), json!("variation"));
    }
    let explicit_size = value
        .get("size")
        .and_then(|v| v.as_str())
        .filter(|s| *s != "auto")
        .map(images::parse_size)
        .transpose()?;
    let source = inputs::prepare_image(&bytes, explicit_size)
        .map_err(|e| images::bad_param("image", format!("{e:#}")))?;
    let (_, height, width) = source.dims3().map_err(|e| bad_request(e.to_string()))?;
    value.insert("width".into(), json!(width));
    value.insert("height".into(), json!(height));
    let request = serde_json::from_value(serde_json::Value::Object(value))
        .map_err(|e| bad_request(e.to_string()))?;
    let mut params = images::validate(request, false, server_steps)?;
    params.prompt = prompt;
    let repaint = mask
        .as_ref()
        .map(|mask| -> Result<_> {
            let original = image::load_from_memory(&bytes)?;
            let mask = image::load_from_memory(mask)?;
            ensure!(
                mask.width() == original.width() && mask.height() == original.height(),
                "mask dimensions must match the input image"
            );
            ensure!(
                mask.color().has_alpha(),
                "OpenAI edit masks must have an alpha channel; transparent pixels repaint"
            );
            let rgba = mask.to_rgba8();
            let values: Vec<f32> = rgba.pixels().map(|p| 1.0 - p[3] as f32 / 255.0).collect();
            let tensor = candle_core::Tensor::from_vec(
                values,
                (1, rgba.height() as usize, rgba.width() as usize),
                &candle_core::Device::Cpu,
            )?;
            inputs::resize_mask(&tensor, width, height)
        })
        .transpose()
        .map_err(|e| images::bad_param("mask", format!("{e:#}")))?;
    let edit = ImageEdit {
        init_image: source,
        mask: repaint,
        strength: if mask.is_some() { 1.0 } else { 0.6 },
        posterior_noise: None,
    };
    Ok((
        params,
        ImageInputs {
            edit: Some(edit),
            loras: vec![],
            control: None,
        },
    ))
}

pub(crate) async fn edits(
    State(state): State<AppState>,
    multipart: Result<Multipart, axum::extract::multipart::MultipartRejection>,
) -> Response {
    compatibility_edit(state, multipart, false).await
}
pub(crate) async fn variations(
    State(state): State<AppState>,
    multipart: Result<Multipart, axum::extract::multipart::MultipartRejection>,
) -> Response {
    compatibility_edit(state, multipart, true).await
}
async fn compatibility_edit(
    state: AppState,
    multipart: Result<Multipart, axum::extract::multipart::MultipartRejection>,
    variation: bool,
) -> Response {
    let multipart = match multipart {
        Ok(m) => m,
        Err(e) => return bad_request(format!("invalid multipart request: {e}")).into_response(),
    };
    let prepared = match multipart_fields(multipart).await {
        Ok(fields) => prepare_multipart(fields, variation, state.settings.image_steps),
        Err(e) => Err(e),
    };
    match prepared {
        Ok((params, inputs)) => images::submit_image(state, params, inputs).await,
        Err(e) => e.into_response(),
    }
}

fn control_fingerprint(path: &std::path::Path) -> Result<images::ControlIdentity> {
    use std::os::unix::fs::MetadataExt;
    // HF files are symlinks into hash-named blobs; the published basename identifies the variant.
    let name = path
        .file_name()
        .context("control checkpoint needs a filename")?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    let path = parent.canonicalize()?.join(name);
    let metadata = std::fs::metadata(&path)?;
    ensure!(metadata.is_file(), "control checkpoint must be a file");
    let variant = crate::zimage::controlnet::ControlVariant::from_filename(
        path.file_name()
            .and_then(|s| s.to_str())
            .context("control checkpoint needs a UTF-8 filename")?,
    )?;
    ensure!(
        metadata.len() == variant.file_bytes(),
        "control checkpoint has {} bytes, expected {}",
        metadata.len(),
        variant.file_bytes()
    );
    Ok(images::ControlIdentity {
        path,
        fingerprint: crate::zimage::lora::Fingerprint {
            bytes: metadata.len(),
            modified_ns: metadata
                .modified()?
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos(),
            inode: metadata.ino(),
            changed_s: metadata.ctime(),
            changed_ns: metadata.ctime_nsec(),
        },
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PreprocessRequest {
    image: String,
    #[serde(rename = "type")]
    kind: crate::zimage::preprocess::Kind,
}
pub(crate) async fn preprocess(State(state): State<AppState>, body: Bytes) -> Response {
    let request = match serde_json::from_slice::<PreprocessRequest>(&body) {
        Ok(request) => request,
        Err(e) => return bad_request(format!("invalid preprocess request: {e}")).into_response(),
    };
    if request.kind == crate::zimage::preprocess::Kind::None {
        return images::bad_param("type", "preprocess type must be canny, pose, or depth")
            .into_response();
    }
    let image =
        match image_bytes(&request.image).and_then(|bytes| inputs::prepare_image(&bytes, None)) {
            Ok(image) => image,
            Err(e) => return images::bad_param("image", format!("{e:#}")).into_response(),
        };
    images::submit_preprocess(state, image, request.kind).await
}

#[cfg(test)]
mod tests {
    use super::*;
    fn png(width: u32, height: u32, alpha: Option<u8>) -> Vec<u8> {
        let mut bytes = std::io::Cursor::new(Vec::new());
        match alpha {
            Some(a) => image::DynamicImage::ImageRgba8(image::ImageBuffer::from_pixel(
                width,
                height,
                image::Rgba([120, 40, 10, a]),
            )),
            None => image::DynamicImage::ImageRgb8(image::ImageBuffer::from_pixel(
                width,
                height,
                image::Rgb([120, 40, 10]),
            )),
        }
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
        bytes.into_inner()
    }
    fn request(value: serde_json::Value) -> RenderRequest {
        serde_json::from_value(value).unwrap()
    }
    fn error_param(value: serde_json::Value) -> String {
        prepare(request(value), None).err().unwrap().body["error"]["param"]
            .as_str()
            .unwrap()
            .to_owned()
    }
    #[test]
    fn native_rejects_unknown_fields_and_dependent_controls() {
        for value in [
            json!({"prompt":"x","strenth":0.5}),
            json!({"prompt":"x","loras":[{"name":"x","weight":1,"typo":0}]}),
            json!({"prompt":"x","control":{"image":"x","typo":0}}),
        ] {
            assert!(serde_json::from_value::<RenderRequest>(value).is_err());
        }
        for value in [
            json!({"prompt":"x","strength":0.5}),
            json!({"prompt":"x","mask":"x"}),
            json!({"prompt":"x","mask_blur":1}),
            json!({"prompt":"x","width":512}),
        ] {
            assert!(prepare(request(value), None).is_err());
        }
    }
    #[test]
    fn native_edit_errors_name_the_failing_field() {
        let image = base64::engine::general_purpose::STANDARD.encode(png(512, 512, None));
        assert_eq!(
            error_param(json!({"prompt":"x","init_image":image,"strength":2})),
            "strength"
        );
        assert_eq!(
            error_param(json!({"prompt":"x","init_image":image,"mask":"%%%"})),
            "mask"
        );
        let mask = base64::engine::general_purpose::STANDARD.encode(png(512, 512, None));
        assert_eq!(
            error_param(json!({
                "prompt":"x","init_image":image,"mask":mask,"mask_blur":-1
            })),
            "mask_blur"
        );
    }
    #[test]
    fn control_identity_keeps_the_hub_filename_and_checks_the_blob_size() -> Result<()> {
        let root = std::env::temp_dir().join(format!("xwen-control-link-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let blob = root.join("hash-blob");
        let file = std::fs::File::create(&blob)?;
        let expected = crate::zimage::controlnet::ControlVariant::Lite.file_bytes();
        file.set_len(expected)?;
        let path = root.join(crate::zimage::controlnet::DEFAULT_CONTROL_FILE);
        let _ = std::fs::remove_file(&path);
        std::os::unix::fs::symlink(&blob, &path)?;
        let identity = control_fingerprint(&path)?;
        assert_eq!(identity.path.file_name(), path.file_name());
        assert_eq!(identity.fingerprint.bytes, expected);
        file.set_len(128)?;
        assert!(control_fingerprint(&path).is_err());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn native_defaults_to_source_size_and_preserves_full_seed() {
        let image = base64::engine::general_purpose::STANDARD.encode(png(513, 511, None));
        let (params, inputs) = prepare(
            request(json!({"prompt":"edit","init_image":image,"seed":u64::MAX})),
            None,
        )
        .unwrap();
        assert_eq!((params.width, params.height), (512, 512));
        assert_eq!(params.seed, Some(u64::MAX));
        assert_eq!(inputs.edit.unwrap().strength, 0.6);
    }
    #[test]
    fn openai_alpha_mask_means_transparent_repaints() {
        let mut fields = HashMap::from([
            ("image".into(), png(512, 512, None)),
            ("mask".into(), png(512, 512, Some(0))),
            ("prompt".into(), b"edit".to_vec()),
        ]);
        let (_, inputs) = prepare_multipart(fields.clone(), false, None).unwrap();
        let edit = inputs.edit.unwrap();
        assert_eq!(edit.strength, 1.0);
        assert_eq!(
            edit.mask
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0],
            1.0
        );
        fields.insert("mask".into(), png(256, 256, Some(0)));
        assert!(prepare_multipart(fields, false, None).is_err());
    }
    #[test]
    fn variations_encode_an_empty_prompt_and_use_fixed_strength() {
        let fields = HashMap::from([("image".into(), png(512, 512, None))]);
        let (params, inputs) = prepare_multipart(fields, true, None).unwrap();
        assert!(params.prompt.is_empty());
        assert_eq!(inputs.edit.unwrap().strength, 0.6);
    }
    #[tokio::test]
    async fn sdk_multipart_fields_reach_the_edit_preparer() {
        use axum::extract::FromRequest;
        let boundary = "image-control-test";
        let mut body = Vec::new();
        for (name, bytes, file) in [
            ("image", png(512, 512, None), true),
            ("prompt", b"a painted room".to_vec(), false),
            ("n", b"2".to_vec(), false),
        ] {
            body.extend_from_slice(
                format!(
                    "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"{}\r\n\r\n",
                    if file {
                        "; filename=\"source.png\""
                    } else {
                        ""
                    }
                )
                .as_bytes(),
            );
            body.extend_from_slice(&bytes);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        let request = axum::http::Request::builder()
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(axum::body::Body::from(body))
            .unwrap();
        let multipart = Multipart::from_request(request, &()).await.unwrap();
        let fields = multipart_fields(multipart).await.unwrap();
        let (params, inputs) = prepare_multipart(fields, false, None).unwrap();
        assert_eq!(params.n, 2);
        assert_eq!(params.prompt, "a painted room");
        assert!(inputs.edit.is_some());
    }
}
