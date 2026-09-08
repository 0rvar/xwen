//! `POST /v1/images/generations`: Z-Image-Turbo behind the OpenAI images shape.
//!
//! Three paths reach one handler. `/v1/images/generations` is the canonical
//! one; `/images/generations` is where a client that `urljoin`s a base URL
//! without a trailing slash lands; `/proxy/openai/images/generations` is the
//! path ComfyUI's stock OpenAI image node POSTs to, relative to its
//! `--comfy-api-base`, which is what makes that node work against this server
//! with nothing installed on the ComfyUI side (docs/decisions/zimage.md). The
//! paths differ in exactly one rule, the `model` field: the first two take the
//! full name or nothing, like every LM route, and the proxy path takes whatever
//! the node's dropdown says, because there is one image model here and the
//! dropdown cannot name it.
//!
//! The work runs on its own OS thread, `image-engine`, beside the language
//! engine and independent of it: its own lazy load, its own idle unload on the
//! same `--idle-unload` setting, one render at a time, a short bounded queue.
//! The two engines do not know about each other's residency, so a language
//! model and the image pipeline can both be resident inside one idle window;
//! the image side is about 20 GB. Never 401, 402, 409 or 429 from here: the
//! ComfyUI client rewrites those four statuses into comfy.org login and credit
//! messages before it reads the body.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::body::Bytes;
use axum::extract::{OriginalUri, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use serde::Deserialize;
use serde_json::json;

use super::config::ServeSettings;
use super::log::{ServeLog, ServeLogger};
use super::openai::{bad_request, error};
use super::{ApiError, AppState};
use crate::hub::Model;
use crate::zimage::pipeline::{DEFAULT_STEPS, ImageOptions, ZImagePipeline, encode_png};

/// How many requests may wait for the engine. A render is seconds to a minute,
/// the ComfyUI client waits up to an hour, and past this a client is told to
/// come back rather than queued into a wait it did not ask for.
pub(crate) const QUEUE_CAPACITY: usize = 4;
const MAX_N: u32 = 4;
/// The step ceiling a request may ask for, and equally the ceiling on the
/// server-wide default: the two bounds are the same number so an operator
/// cannot configure a default no request could have named.
pub(crate) const MAX_STEPS: usize = 50;
const DEFAULT_SIZE: (usize, usize) = (1024, 1024);
const PROXY_PATH: &str = "/proxy/openai/images/generations";

/// The three paths the handler answers on.
pub(crate) fn is_images_path(path: &str) -> bool {
    matches!(
        path,
        "/v1/images/generations"
            | "/images/generations"
            | PROXY_PATH
            | "/v1/images/render"
            | "/v1/images/edits"
            | "/v1/images/variations"
            | "/v1/images/preprocess"
            | "/v1/images/loras"
    )
}

/// One render request as the engine takes it: validated, with every default
/// applied.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ImageParams {
    pub prompt: String,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub n: u32,
    /// The first image's seed; the rest follow at +1 each. Drawn by the engine
    /// when absent.
    pub seed: Option<u64>,
    /// The `model` string the proxy path accepted in place of the real name,
    /// for the log line.
    pub model_note: Option<String>,
}

/// One rendered image on its way back to the handler.
pub(crate) struct RenderedImage {
    pub png: Vec<u8>,
    pub seed: u64,
    pub start_step: usize,
    pub control_map: Option<Vec<u8>>,
}

/// Why a job failed, split by whose fault it is: a request fault is a 400, a
/// failure in the render is a 500.
#[derive(Debug)]
pub(crate) enum ImageError {
    Request(String),
    Render(String),
}

#[derive(Default)]
pub(crate) struct ImageInputs {
    pub edit: Option<crate::zimage::pipeline::ImageEdit>,
    pub loras: Vec<crate::zimage::lora::ResolvedLora>,
    pub control: Option<ControlInput>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ControlIdentity {
    pub path: std::path::PathBuf,
    pub fingerprint: crate::zimage::lora::Fingerprint,
}

pub(crate) struct ControlInput {
    pub image: crate::zimage::pipeline::ImageControl,
    pub preprocess: crate::zimage::preprocess::Kind,
    pub checkpoint: ControlIdentity,
}

pub(crate) enum ImageJob {
    Render {
        inputs: ImageInputs,
        params: ImageParams,
        reply: tokio::sync::oneshot::Sender<Result<Vec<RenderedImage>, ImageError>>,
    },
    Preprocess {
        image: candle_core::Tensor,
        kind: crate::zimage::preprocess::Kind,
        reply: tokio::sync::oneshot::Sender<Result<Vec<u8>, ImageError>>,
    },
}

/// The handler's end of the engine: the queue and the residency flag.
pub struct Handle {
    sender: crossbeam_channel::Sender<ImageJob>,
    /// Whether the encoder and the pipeline are resident, set and cleared by
    /// the engine thread, read by `/health`.
    pub resident: Arc<AtomicBool>,
}

impl Handle {
    pub fn is_loaded(&self) -> bool {
        self.resident.load(Ordering::Relaxed)
    }

    /// A handle with no engine behind it, for tests that build an `AppState`
    /// and never render: a send fails as disconnected.
    #[cfg(test)]
    pub(crate) fn detached() -> Self {
        let (sender, _receiver) = crossbeam_channel::bounded(0);
        Self {
            sender,
            resident: Arc::new(AtomicBool::new(false)),
        }
    }
}

pub(crate) const IMAGE_ENGINE_THREAD: &str = "image-engine";

/// Start the image engine. The thread exits when every [`Handle`] is dropped
/// and the queue has drained, or when `shutdown` fires between jobs.
pub(crate) fn spawn(
    settings: &ServeSettings,
    shutdown: Arc<super::types::Cancel>,
    logger: ServeLogger,
) -> (Handle, std::thread::JoinHandle<()>) {
    let (sender, receiver) = crossbeam_channel::bounded::<ImageJob>(QUEUE_CAPACITY);
    let resident = Arc::new(AtomicBool::new(false));
    // Said once at startup, because a step count nobody asked for in a request
    // is otherwise only visible in each render's own log line. The pipeline's
    // own default needs no announcement.
    if let Some(steps) = settings.image_steps.filter(|&n| n != DEFAULT_STEPS) {
        logger.log(ServeLog::HostLine(format!(
            "xwen serve: images render in {steps} steps unless a request names its own \
             (pipeline default {DEFAULT_STEPS})"
        )));
    }
    let idle = settings.idle_unload;
    let thread_resident = Arc::clone(&resident);
    let thread = std::thread::Builder::new()
        .name(IMAGE_ENGINE_THREAD.to_string())
        .spawn(move || engine_loop(receiver, idle, thread_resident, shutdown, logger))
        .expect("spawning the image engine thread");
    (Handle { sender, resident }, thread)
}

/// The encoder, the pipeline and what they were opened from.
struct Loaded {
    encoder: crate::XwenModel,
    pipeline: ZImagePipeline,
    encoder_entry: Model,
    spec: crate::hub::EncoderSpec,
    tokenizer_path: std::path::PathBuf,
    loras: Vec<crate::zimage::lora::ResolvedLora>,
    control: Option<ControlIdentity>,
}

impl Loaded {
    /// Open everything `xwen image` opens, from the cached snapshot.
    fn open(
        logger: &ServeLogger,
        loras: &[crate::zimage::lora::ResolvedLora],
        control: Option<&ControlIdentity>,
    ) -> Result<Self, ImageError> {
        let index = crate::hub::cached_model(Model::ZImageTurbo).ok_or_else(|| {
            ImageError::Request(format!(
                "{} is not in the Hugging Face cache: run `xwen fetch --model-size {}` on the \
                 server first",
                Model::ZImageTurbo.full_name(),
                Model::ZImageTurbo
            ))
        })?;
        let adapters = crate::zimage::lora::PreparedLoras::load(loras)
            .map_err(|e| ImageError::Request(format!("invalid LoRA: {e:#}")))?;
        adapters
            .validate_base(index.parent().ok_or_else(|| {
                ImageError::Request("cached pipeline has no parent directory".into())
            })?)
            .map_err(|e| ImageError::Request(format!("invalid LoRA: {e:#}")))?;
        let open = || -> Result<(Self, Duration, Duration)> {
            let root = index
                .parent()
                .context("the cached model_index.json has no parent directory")?;
            let encoder_entry = Model::ZImageTurbo
                .text_encoder()
                .context("the pipeline entry names no text encoder")?;
            let spec = encoder_entry
                .encoder_spec()
                .context("the text encoder entry carries no encoder spec")?;
            let encoder_dir = root.join(
                Path::new(encoder_entry.file())
                    .parent()
                    .context("the encoder entry's config has no parent directory")?,
            );
            let device = crate::gguf::metal_device()?;

            let started = Instant::now();
            let source = crate::CheckpointSource::open(&encoder_dir, &device, Some(encoder_entry))?;
            let tokenizer_path = source
                .safetensors()
                .with_context(|| {
                    format!(
                        "{} did not open as a safetensors set",
                        encoder_dir.display()
                    )
                })?
                .tokenizer_path()
                .to_path_buf();
            let encoder = crate::XwenModel::load_encoder(source, spec.max_tokens)?;
            let encoder_s = started.elapsed();

            let started = Instant::now();
            let pipeline = ZImagePipeline::load_with_loras_and_control(
                root,
                &device,
                &adapters,
                control.map(|c| c.path.as_path()),
            )?;
            let pipeline_s = started.elapsed();
            Ok((
                Self {
                    encoder,
                    pipeline,
                    encoder_entry,
                    spec,
                    tokenizer_path,
                    loras: loras.to_vec(),
                    control: control.cloned(),
                },
                encoder_s,
                pipeline_s,
            ))
        };
        let (loaded, encoder_s, pipeline_s) = open().map_err(classify_load_error)?;
        logger.log(ServeLog::HostLine(format!(
            "xwen serve: image pipeline loaded in {:.1}s (encoder {:.1}s, transformer+VAE {:.1}s)",
            (encoder_s + pipeline_s).as_secs_f64(),
            encoder_s.as_secs_f64(),
            pipeline_s.as_secs_f64()
        )));
        Ok(loaded)
    }

    fn render(
        &mut self,
        params: &ImageParams,
        inputs: &ImageInputs,
        logger: &ServeLogger,
    ) -> Result<Vec<RenderedImage>> {
        let rendered = crate::zimage::conditioning::prompt_ids(
            self.encoder_entry,
            &self.tokenizer_path,
            &params.prompt,
        )?;
        if let Some(from) = rendered.truncated_from {
            logger.log(ServeLog::HostLine(format!(
                "xwen serve: image prompt is {from} tokens, truncated to {}",
                rendered.ids.len()
            )));
        }
        let (cap_feats, _n_tokens) = self.encoder.encode(&rendered.ids, self.spec.layer)?;
        // A drawn seed stays under 2^53: the seed goes back to the client as a
        // JSON number, and a JavaScript client rounds anything wider, so a
        // seed it echoed back would not reproduce the image it came with. A
        // seed the client chose is used as given, whatever its width.
        let first_seed = params.seed.unwrap_or_else(|| rand::random::<u64>() >> 11);
        let mut out = Vec::with_capacity(params.n as usize);
        for i in 0..params.n as u64 {
            let seed = first_seed.wrapping_add(i);
            let started = Instant::now();
            let options = ImageOptions {
                width: params.width,
                height: params.height,
                steps: params.steps,
                seed,
                latents: None,
            };
            let run = match (&inputs.edit, &inputs.control) {
                (edit, Some(control)) => self.pipeline.generate_controlled(
                    &cap_feats,
                    &options,
                    edit.as_ref(),
                    &control.image,
                )?,
                (Some(edit), None) => self.pipeline.generate_edited(&cap_feats, &options, edit)?,
                (None, None) => self.pipeline.generate(&cap_feats, &options)?,
            };
            let png = encode_png(&run.image)?;
            logger.log(ServeLog::HostLine(format!(
                "xwen serve: image rendered {}x{} in {} steps, {:.1}s (seed {seed})",
                params.width,
                params.height,
                params.steps,
                started.elapsed().as_secs_f64()
            )));
            out.push(RenderedImage {
                png,
                seed,
                start_step: run.start_step,
                control_map: inputs
                    .control
                    .as_ref()
                    .map(|c| encode_png(&c.image.image))
                    .transpose()?,
            });
        }
        Ok(out)
    }
}

fn classify_load_error(error: anyhow::Error) -> ImageError {
    if error
        .downcast_ref::<crate::zimage::lora::AdapterError>()
        .is_some()
    {
        ImageError::Request(format!("invalid LoRA: {error:#}"))
    } else {
        ImageError::Render(format!("loading the image pipeline: {error:#}"))
    }
}

fn engine_loop(
    jobs: crossbeam_channel::Receiver<ImageJob>,
    idle_unload: Option<Duration>,
    resident: Arc<AtomicBool>,
    shutdown: Arc<super::types::Cancel>,
    logger: ServeLogger,
) {
    let mut loaded: Option<Loaded> = None;
    let mut preprocessor = crate::zimage::preprocess::Preprocessor::new();
    let mut preprocessed = false;
    loop {
        // The idle timer only matters while something is loaded, and it is
        // measured from the moment the previous job returned.
        let idle = idle_unload.filter(|_| loaded.is_some() || preprocessed);
        let waiting_since = Instant::now();
        let job = match idle {
            Some(window) => match jobs.recv_timeout(window) {
                Ok(job) => job,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    loaded = None;
                    preprocessor = crate::zimage::preprocess::Preprocessor::new();
                    preprocessed = false;
                    resident.store(false, Ordering::Relaxed);
                    logger.log(ServeLog::HostLine(format!(
                        "xwen serve: image pipeline unloaded after {:.0}s idle (configured {}s)",
                        waiting_since.elapsed().as_secs_f64(),
                        window.as_secs()
                    )));
                    continue;
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            },
            None => match jobs.recv() {
                Ok(job) => job,
                Err(_) => break,
            },
        };
        if shutdown.is_cancelled() {
            match job {
                ImageJob::Render { reply, .. } => {
                    let _ = reply.send(Err(ImageError::Render(
                        "the server is shutting down".into(),
                    )));
                }
                ImageJob::Preprocess { reply, .. } => {
                    let _ = reply.send(Err(ImageError::Render(
                        "the server is shutting down".into(),
                    )));
                }
            }
            continue;
        }
        let (params, mut inputs, reply) = match job {
            ImageJob::Preprocess { image, kind, reply } => {
                preprocessed = true;
                let result =
                    preprocess_on_worker(&mut preprocessor, &image, kind).and_then(|image| {
                        encode_png(&image).map_err(|e| ImageError::Render(format!("{e:#}")))
                    });
                let _ = reply.send(result);
                continue;
            }
            ImageJob::Render {
                params,
                inputs,
                reply,
            } => (params, inputs, reply),
        };
        if let Some(control) = &mut inputs.control {
            preprocessed = true;
            match preprocess_on_worker(&mut preprocessor, &control.image.image, control.preprocess)
            {
                Ok(image) => control.image.image = image,
                Err(e) => {
                    let _ = reply.send(Err(e));
                    continue;
                }
            }
        }
        if let Some(note) = &params.model_note {
            logger.log(ServeLog::HostLine(format!(
                "xwen serve: images: model {note:?} served by {} (proxy path)",
                Model::ZImageTurbo.full_name()
            )));
        }
        if loaded.as_ref().is_some_and(|held| {
            held.loras != inputs.loras
                || held.control.as_ref() != inputs.control.as_ref().map(|c| &c.checkpoint)
        }) {
            loaded = None;
            resident.store(false, Ordering::Relaxed);
        }
        let result = match loaded.as_mut() {
            Some(held) => Ok(held),
            None => match Loaded::open(
                &logger,
                &inputs.loras,
                inputs.control.as_ref().map(|c| &c.checkpoint),
            ) {
                Ok(opened) => {
                    resident.store(true, Ordering::Relaxed);
                    Ok(loaded.insert(opened))
                }
                Err(e) => Err(e),
            },
        }
        .and_then(|held| {
            held.render(&params, &inputs, &logger)
                .map_err(|e| ImageError::Render(format!("{e:#}")))
        });
        // A client that hung up is the only reason this fails, and its answer
        // has nowhere to go.
        let _ = reply.send(result);
    }
    // A pipeline dropped here unregisters its buffers on the way out.
    drop(loaded);
    resident.store(false, Ordering::Relaxed);
}

/// The request body. Every field but `prompt` is optional and anything not
/// named here is accepted and dropped, which is what lets the OpenAI SDK's and
/// ComfyUI's own payloads (`quality`, `background`, `moderation`, `style`,
/// `user`, `partial_images`, `output_compression`) through untouched.
#[derive(Debug, Deserialize)]
pub(crate) struct ImagesRequest {
    pub prompt: String,
    pub model: Option<String>,
    pub n: Option<u32>,
    pub size: Option<String>,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub response_format: Option<String>,
    pub output_format: Option<String>,
    pub stream: Option<bool>,
    pub negative_prompt: Option<String>,
    pub guidance_scale: Option<f64>,
    pub seed: Option<u64>,
    pub rng_seed: Option<u64>,
    pub steps: Option<usize>,
    pub num_inference_steps: Option<usize>,
}

pub(super) fn bad_param(param: &str, message: impl Into<String>) -> ApiError {
    let mut err = bad_request(message);
    err.body["error"]["param"] = json!(param);
    err
}

/// `"WxH"` or `"auto"`, in the OpenAI spelling.
pub(super) fn parse_size(text: &str) -> Result<(usize, usize), ApiError> {
    let text = text.trim();
    if text.eq_ignore_ascii_case("auto") {
        return Ok(DEFAULT_SIZE);
    }
    let malformed = || {
        bad_param(
            "size",
            format!("Invalid size format: '{text}'. Expected WIDTHxHEIGHT."),
        )
    };
    let (w, h) = text.split_once(['x', 'X']).ok_or_else(malformed)?;
    let w: usize = w.trim().parse().map_err(|_| malformed())?;
    let h: usize = h.trim().parse().map_err(|_| malformed())?;
    Ok((w, h))
}

/// Turn a parsed body into what the engine runs, or the 400 saying why not.
/// `proxy` is whether the request came in on the ComfyUI proxy path.
/// `server_steps` is the server-wide step default, which a request's own
/// `steps` overrides and `None` falls through to [`DEFAULT_STEPS`]. It is a
/// parameter rather than a read of the settings because the step count is the
/// one render knob a client cannot always send: ComfyUI's stock OpenAI image
/// node has no field for it.
pub(crate) fn validate(
    request: ImagesRequest,
    proxy: bool,
    server_steps: Option<usize>,
) -> Result<ImageParams, ApiError> {
    let prompt = request.prompt.trim();
    if prompt.is_empty() {
        return Err(bad_param(
            "prompt",
            "prompt: a non-empty prompt is required",
        ));
    }

    let full_name = Model::ZImageTurbo.full_name();
    let model_note = match request.model.as_deref().map(str::trim) {
        None | Some("") => None,
        Some(name) if name == full_name => None,
        Some(name) if proxy => Some(name.to_string()),
        Some(name) => {
            return Err(bad_param(
                "model",
                format!(
                    "unknown model {name:?}: this route serves {full_name} only; name it in \
                     full or leave the field out"
                ),
            ));
        }
    };

    let (width, height) = match (request.width, request.height) {
        (Some(w), Some(h)) => (w, h),
        (None, None) => match request.size.as_deref() {
            Some(text) => parse_size(text)?,
            None => DEFAULT_SIZE,
        },
        _ => {
            return Err(bad_request(
                "width and height go together: pass both, or `size` as WIDTHxHEIGHT",
            ));
        }
    };
    ZImagePipeline::check_size(width, height).map_err(|e| bad_param("size", e.to_string()))?;

    let n = request.n.unwrap_or(1);
    if !(1..=MAX_N).contains(&n) {
        return Err(bad_param(
            "n",
            format!("n must be between 1 and {MAX_N}, got {n}"),
        ));
    }

    match request.response_format.as_deref().map(str::trim) {
        None | Some("") | Some("b64_json") => {}
        Some("url") => {
            return Err(bad_param(
                "response_format",
                "this server returns b64_json only; it hosts no URLs",
            ));
        }
        Some(other) => {
            return Err(bad_param(
                "response_format",
                format!("unknown response_format {other:?}; this server returns b64_json only"),
            ));
        }
    }
    match request.output_format.as_deref().map(str::trim) {
        None | Some("") | Some("png") => {}
        Some(other) => {
            return Err(bad_param(
                "output_format",
                format!("output_format {other:?} is not supported; this server writes png"),
            ));
        }
    }
    if request.stream == Some(true) {
        return Err(bad_param(
            "stream",
            "streaming is not supported on this route; the image comes back in one response",
        ));
    }
    let guided = format!(
        "{full_name} is distilled to run without guidance, so a negative prompt or a \
         guidance_scale would be silently ignored; leave them out"
    );
    if request
        .negative_prompt
        .as_deref()
        .is_some_and(|s| !s.trim().is_empty())
    {
        return Err(bad_param("negative_prompt", guided));
    }
    if request.guidance_scale.is_some_and(|g| g != 0.0) {
        return Err(bad_param("guidance_scale", guided));
    }

    let seed = match (request.seed, request.rng_seed) {
        (Some(a), Some(b)) if a != b => {
            return Err(bad_param(
                "seed",
                format!("seed {a} and rng_seed {b} disagree; pass one of them"),
            ));
        }
        (a, b) => a.or(b),
    };
    let steps = match (request.steps, request.num_inference_steps) {
        (Some(a), Some(b)) if a != b => {
            return Err(bad_param(
                "steps",
                format!("steps {a} and num_inference_steps {b} disagree; pass one of them"),
            ));
        }
        (a, b) => a.or(b).or(server_steps).unwrap_or(DEFAULT_STEPS),
    };
    if !(1..=MAX_STEPS).contains(&steps) {
        return Err(bad_param(
            "steps",
            format!("steps must be between 1 and {MAX_STEPS}, got {steps}"),
        ));
    }

    Ok(ImageParams {
        prompt: prompt.to_string(),
        width,
        height,
        steps,
        n,
        seed,
        model_note,
    })
}

fn server_error(status: StatusCode, message: impl Into<String>) -> ApiError {
    error(status, "server_error", None, message)
}

pub(crate) async fn generations(
    State(state): State<AppState>,
    OriginalUri(uri): OriginalUri,
    body: Bytes,
) -> Response {
    let request: ImagesRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(e) => {
            return bad_request(format!("could not parse the request body: {e}")).into_response();
        }
    };
    let params = match validate(
        request,
        uri.path() == PROXY_PATH,
        state.settings.image_steps,
    ) {
        Ok(params) => params,
        Err(err) => return err.into_response(),
    };
    submit_image(state, params, ImageInputs::default()).await
}

pub(crate) async fn submit_image(
    state: AppState,
    params: ImageParams,
    inputs: ImageInputs,
) -> Response {
    let (width, height, steps) = (params.width, params.height, params.steps);

    let (reply, answer) = tokio::sync::oneshot::channel();
    match state.images.sender.try_send(ImageJob::Render {
        params,
        inputs,
        reply,
    }) {
        Ok(()) => {}
        Err(crossbeam_channel::TrySendError::Full(_)) => {
            return server_error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("the image engine has {QUEUE_CAPACITY} requests queued; retry shortly"),
            )
            .with_header("retry-after", "5")
            .into_response();
        }
        Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
            return server_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "the image engine is not running",
            )
            .into_response();
        }
    }
    let images = match answer.await {
        Ok(Ok(images)) => images,
        Ok(Err(ImageError::Request(message))) => return bad_request(message).into_response(),
        Ok(Err(ImageError::Render(message))) => {
            return server_error(StatusCode::INTERNAL_SERVER_ERROR, message).into_response();
        }
        Err(_) => {
            return server_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "the image engine stopped before answering",
            )
            .into_response();
        }
    };

    let data: Vec<_> = images
        .iter()
        .map(|image| {
            json!({
                "b64_json": base64::engine::general_purpose::STANDARD.encode(&image.png),
                "seed": image.seed,
                "start_step": image.start_step,
                "control_map": image.control_map.as_ref().map(|map|base64::engine::general_purpose::STANDARD.encode(map)),
            })
        })
        .collect();
    axum::Json(json!({
        "created": super::unix_now(),
        "model": Model::ZImageTurbo.full_name(),
        "size": format!("{width}x{height}"),
        "steps": steps,
        "output_format": "png",
        "data": data,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adapter_load_faults_are_request_errors_and_device_faults_are_not() {
        let adapter = anyhow::Error::new(crate::zimage::lora::AdapterError("overflow".into()))
            .context("building transformer");
        assert!(matches!(
            classify_load_error(adapter),
            ImageError::Request(_)
        ));
        assert!(matches!(
            classify_load_error(anyhow::anyhow!("device unavailable")),
            ImageError::Render(_)
        ));
    }

    fn parse(json: &str) -> ImagesRequest {
        serde_json::from_str(json).expect("the request parses")
    }

    fn message(err: &ApiError) -> String {
        err.body["error"]["message"].as_str().unwrap().to_string()
    }

    fn param(err: &ApiError) -> Option<String> {
        err.body["error"]["param"].as_str().map(str::to_string)
    }

    #[test]
    fn image_routes_share_the_image_auth_policy() {
        assert!(is_images_path("/v1/images/generations"));
        assert!(is_images_path("/images/generations"));
        assert!(is_images_path("/proxy/openai/images/generations"));
        for path in [
            "/v1/images/edits",
            "/v1/images/variations",
            "/v1/images/render",
            "/v1/images/preprocess",
            "/v1/images/loras",
        ] {
            assert!(is_images_path(path));
        }
        assert!(!is_images_path("/v1/chat/completions"));
        assert!(!is_images_path("/v1/images/generations/"));
    }

    #[test]
    fn a_bare_prompt_gets_every_default() {
        let params = validate(parse(r#"{"prompt":"a cat"}"#), false, None).unwrap();
        assert_eq!(
            params,
            ImageParams {
                prompt: "a cat".into(),
                width: 1024,
                height: 1024,
                steps: DEFAULT_STEPS,
                n: 1,
                seed: None,
                model_note: None,
            }
        );
    }

    #[test]
    fn the_comfyui_stock_node_payload_is_accepted_on_the_proxy_path_only() {
        let body = r#"{"model":"gpt-image-1","prompt":"a cat","quality":"low","background":"auto","n":1,"size":"1024x1024","moderation":"low"}"#;
        let params = validate(parse(body), true, None).unwrap();
        assert_eq!(params.model_note.as_deref(), Some("gpt-image-1"));
        assert_eq!((params.width, params.height), (1024, 1024));

        let err = validate(parse(body), false, None).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(param(&err).as_deref(), Some("model"));
        assert!(message(&err).contains("Z-Image-Turbo"), "{}", message(&err));
    }

    #[test]
    fn the_full_name_or_nothing_selects_the_model_on_the_v1_path() {
        for model in [r#""model":"Z-Image-Turbo","#, r#""model":"","#, ""] {
            let params =
                validate(parse(&format!(r#"{{{model}"prompt":"x"}}"#)), false, None).unwrap();
            assert_eq!(params.model_note, None);
        }
        // The CLI alias is refused on the wire, as it is on every LM route.
        let err = validate(
            parse(r#"{"model":"zimage-turbo","prompt":"x"}"#),
            false,
            None,
        )
        .unwrap_err();
        assert!(message(&err).contains("zimage-turbo"), "{}", message(&err));
    }

    #[test]
    fn size_spellings() {
        assert_eq!(parse_size("1024x1536").unwrap(), (1024, 1536));
        assert_eq!(parse_size(" 512X512 ").unwrap(), (512, 512));
        assert_eq!(parse_size("auto").unwrap(), DEFAULT_SIZE);
        for bad in ["1024x", "x1024", "1024", "big", "1024x1024x3", "-1x5"] {
            let err = parse_size(bad).unwrap_err();
            assert_eq!(err.status, StatusCode::BAD_REQUEST);
            assert_eq!(param(&err).as_deref(), Some("size"));
            assert_eq!(
                message(&err),
                format!("Invalid size format: '{bad}'. Expected WIDTHxHEIGHT.")
            );
        }
    }

    #[test]
    fn a_size_the_pipeline_refuses_is_a_400_with_its_reason() {
        let err = validate(parse(r#"{"prompt":"x","size":"1000x1000"}"#), false, None).unwrap_err();
        assert_eq!(param(&err).as_deref(), Some("size"));
        assert!(
            message(&err).contains("multiples of 16"),
            "{}",
            message(&err)
        );
        // A cell count that would need pad rows is refused rather than padded.
        let err = validate(parse(r#"{"prompt":"x","size":"1040x1040"}"#), false, None).unwrap_err();
        assert!(
            message(&err).contains("not a multiple of 32"),
            "{}",
            message(&err)
        );
    }

    #[test]
    fn width_and_height_override_size_and_come_as_a_pair() {
        let params = validate(
            parse(r#"{"prompt":"x","size":"1024x1024","width":512,"height":768}"#),
            false,
            None,
        )
        .unwrap();
        assert_eq!((params.width, params.height), (512, 768));
        let err = validate(parse(r#"{"prompt":"x","width":512}"#), false, None).unwrap_err();
        assert!(message(&err).contains("go together"), "{}", message(&err));
    }

    #[test]
    fn n_is_one_through_four() {
        assert_eq!(
            validate(parse(r#"{"prompt":"x","n":4}"#), false, None)
                .unwrap()
                .n,
            4
        );
        for n in [0, 5, 10] {
            let err =
                validate(parse(&format!(r#"{{"prompt":"x","n":{n}}}"#)), false, None).unwrap_err();
            assert_eq!(param(&err).as_deref(), Some("n"));
        }
    }

    #[test]
    fn refusals_name_their_field() {
        let cases = [
            (r#"{"prompt":"   "}"#, "prompt", "non-empty"),
            (
                r#"{"prompt":"x","response_format":"url"}"#,
                "response_format",
                "b64_json only",
            ),
            (
                r#"{"prompt":"x","response_format":"webp"}"#,
                "response_format",
                "b64_json only",
            ),
            (
                r#"{"prompt":"x","output_format":"jpeg"}"#,
                "output_format",
                "png",
            ),
            (r#"{"prompt":"x","stream":true}"#, "stream", "not supported"),
            (
                r#"{"prompt":"x","negative_prompt":"blurry"}"#,
                "negative_prompt",
                "without guidance",
            ),
            (
                r#"{"prompt":"x","guidance_scale":3.5}"#,
                "guidance_scale",
                "without guidance",
            ),
            (
                r#"{"prompt":"x","seed":1,"rng_seed":2}"#,
                "seed",
                "disagree",
            ),
            (
                r#"{"prompt":"x","steps":8,"num_inference_steps":9}"#,
                "steps",
                "disagree",
            ),
            (r#"{"prompt":"x","steps":0}"#, "steps", "between 1 and 50"),
            (
                r#"{"prompt":"x","num_inference_steps":51}"#,
                "steps",
                "between 1 and 50",
            ),
        ];
        for (body, field, needle) in cases {
            let err = validate(parse(body), false, None).unwrap_err();
            assert_eq!(err.status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(param(&err).as_deref(), Some(field), "{body}");
            assert!(message(&err).contains(needle), "{body}: {}", message(&err));
            assert_eq!(err.body["error"]["type"], "invalid_request_error");
        }
    }

    #[test]
    fn harmless_spellings_pass() {
        let params = validate(
            parse(
                r#"{"prompt":"x","response_format":"b64_json","output_format":"png","stream":false,
                    "negative_prompt":"","guidance_scale":0,"quality":"hd","style":"vivid",
                    "background":"opaque","moderation":"low","user":"u","partial_images":0,
                    "output_compression":80,"rng_seed":9,"num_inference_steps":4}"#,
            ),
            false,
            None,
        )
        .unwrap();
        assert_eq!(params.seed, Some(9));
        assert_eq!(params.steps, 4);
        // The same value under both names is not a disagreement.
        let params = validate(
            parse(r#"{"prompt":"x","seed":3,"rng_seed":3,"steps":2,"num_inference_steps":2}"#),
            false,
            None,
        )
        .unwrap();
        assert_eq!((params.seed, params.steps), (Some(3), 2));
    }

    /// The step count resolves request first, then the server-wide default,
    /// then the pipeline's own. The middle rung is what a client with no step
    /// field of its own — ComfyUI's stock OpenAI image node — renders at.
    #[test]
    fn the_server_wide_step_default_fills_in_for_a_request_that_names_none() {
        let bare = r#"{"prompt":"a cat"}"#;
        assert_eq!(
            validate(parse(bare), false, None).unwrap().steps,
            DEFAULT_STEPS
        );
        // 4 is a real turbo step count and not the pipeline's, so a pass-through
        // of the default would fail this rather than read the same either way.
        assert_ne!(4, DEFAULT_STEPS);
        assert_eq!(validate(parse(bare), false, Some(4)).unwrap().steps, 4);
        // Either spelling of the request's own count wins over the server's.
        for named in [r#""steps":6"#, r#""num_inference_steps":6"#] {
            let body = format!(r#"{{"prompt":"a cat",{named}}}"#);
            assert_eq!(validate(parse(&body), false, Some(4)).unwrap().steps, 6);
        }
        // The proxy path resolves it the same way; nothing about the step count
        // is per-path.
        assert_eq!(validate(parse(bare), true, Some(4)).unwrap().steps, 4);
        // A request out of range is refused whatever the server default is: the
        // bound belongs to the pipeline, not to who supplied the number.
        let err = validate(parse(r#"{"prompt":"x","steps":0}"#), false, Some(4)).unwrap_err();
        assert_eq!(param(&err).as_deref(), Some("steps"));
    }
}

/// Queue preprocessing on the same worker that owns the image models.
pub(crate) async fn submit_preprocess(
    state: AppState,
    image: candle_core::Tensor,
    kind: crate::zimage::preprocess::Kind,
) -> Response {
    let (reply, answer) = tokio::sync::oneshot::channel();
    if let Err(err) = state
        .images
        .sender
        .try_send(ImageJob::Preprocess { image, kind, reply })
    {
        return server_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("image engine unavailable: {err}"),
        )
        .into_response();
    }
    match answer.await {
        Ok(Ok(map)) => {
            axum::Json(json!({"b64_json":base64::engine::general_purpose::STANDARD.encode(map)}))
                .into_response()
        }
        Ok(Err(ImageError::Request(message))) => bad_request(message).into_response(),
        Ok(Err(ImageError::Render(message))) => {
            server_error(StatusCode::INTERNAL_SERVER_ERROR, message).into_response()
        }
        Err(_) => server_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "image engine stopped before answering",
        )
        .into_response(),
    }
}

fn preprocess_on_worker(
    preprocessor: &mut crate::zimage::preprocess::Preprocessor,
    image: &candle_core::Tensor,
    kind: crate::zimage::preprocess::Kind,
) -> Result<candle_core::Tensor, ImageError> {
    crate::zimage::preprocess::Preprocessor::validate(kind)
        .map_err(|e| ImageError::Request(format!("{e:#}")))?;
    preprocessor
        .run(image, kind)
        .map_err(|e| ImageError::Render(format!("{e:#}")))
}
