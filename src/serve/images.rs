//! `POST /v1/images/generations`: the image pipelines behind the OpenAI images
//! shape, Z-Image-Turbo by default and Qwen-Image 2.1 when a request names it.
//!
//! Three paths reach one handler. `/v1/images/generations` is the canonical
//! one; `/images/generations` is where a client that `urljoin`s a base URL
//! without a trailing slash lands; `/proxy/openai/images/generations` is the
//! path ComfyUI's stock OpenAI image node POSTs to, relative to its
//! `--comfy-api-base`, which is what makes that node work against this server
//! with nothing installed on the ComfyUI side (docs/decisions/zimage.md). The
//! paths differ in exactly one rule, the `model` field: the first two take the
//! full name or nothing, like every LM route, and the proxy path takes whatever
//! the node's dropdown says, because the dropdown cannot name a model here: a
//! string that is no pipeline's full name is served by the default one.
//!
//! One image pipeline is resident at a time ([`Pipeline`]). A request for the
//! other one unloads the resident one first, through the same drain the idle
//! unload uses, so the two are never held together.
//!
//! The work runs on its own OS thread, `image-engine`, beside the language
//! engine, with shared memory ownership and an independent idle timer. A
//! waiting engine receives ownership after the resident model is unloaded.
//! Never 401, 402, 409 or 429 from here: the
//! ComfyUI client rewrites those four statuses into comfy.org login and credit
//! messages before it reads the body.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
use super::log::{ImageActivity, ImageRecord, ServeLog, ServeLogger};
use super::openai::{bad_request, error};
use super::{ApiError, AppState};
use crate::hub::Model;
use crate::qwen_image::conditioning::{PromptTooLong, RenderedPrompt};
use crate::qwen_image::pipeline::QwenImagePipeline;
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
            | "/v1/images/models"
            | "/images/models"
            | "/proxy/openai/images/models"
    )
}

/// The image pipelines this surface runs. Every per-model rule of the routes
/// is a match on this, so a third pipeline is a compile error at each of them
/// rather than a request quietly held to another model's rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pipeline {
    ZImage,
    QwenImage,
}

/// The inputs a pipeline can take beside the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Controls {
    pub init_image: bool,
    pub mask: bool,
    pub control: bool,
    pub loras: bool,
}

impl Pipeline {
    pub(crate) const ALL: [Self; 2] = [Self::ZImage, Self::QwenImage];
    /// What a request that names no model runs.
    pub(crate) const DEFAULT: Self = Self::ZImage;

    /// The registry entry, which owns the full name and the files.
    pub(crate) const fn entry(self) -> Model {
        match self {
            Self::ZImage => Model::ZImageTurbo,
            Self::QwenImage => Model::QwenImage21,
        }
    }

    /// The pipeline a full name selects, the way every LM route reads a name.
    fn from_full_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|pipeline| pipeline.entry().full_name() == name)
    }

    pub(crate) const fn default_steps(self) -> usize {
        match self {
            Self::ZImage => DEFAULT_STEPS,
            Self::QwenImage => crate::qwen_image::pipeline::DEFAULT_STEPS,
        }
    }

    /// The server-wide step default is fitted to the distilled model's eight
    /// steps, so it fills in for Z-Image alone: at that count Qwen-Image 2.1
    /// renders an unfinished image, and a client that wants fewer than its
    /// forty says so in the request.
    fn steps_default(self, server_steps: Option<usize>) -> usize {
        match self {
            Self::ZImage => server_steps.unwrap_or(self.default_steps()),
            Self::QwenImage => self.default_steps(),
        }
    }

    /// The multiple both sides of a size must be.
    pub(crate) const fn size_multiple(self) -> usize {
        match self {
            Self::ZImage => crate::zimage::pipeline::PIXELS_PER_TOKEN,
            Self::QwenImage => crate::qwen_image::pipeline::SIZE_MULTIPLE,
        }
    }

    fn size_rule(self) -> &'static str {
        match self {
            Self::ZImage => {
                "both sides multiples of 16, and (width/16) * (height/16) a multiple of 32"
            }
            Self::QwenImage => "both sides multiples of 32",
        }
    }

    fn check_size(self, width: usize, height: usize) -> Result<()> {
        match self {
            Self::ZImage => ZImagePipeline::check_size(width, height),
            Self::QwenImage => QwenImagePipeline::check_size(width, height),
        }
    }

    pub(crate) const fn controls(self) -> Controls {
        let all = matches!(self, Self::ZImage);
        Controls {
            init_image: all,
            mask: all,
            control: all,
            loras: all,
        }
    }

    /// The projected peak of one request, which admission and the 400 for an
    /// oversized image both read.
    fn peak(self, width: usize, height: usize, control: bool, loras: usize) -> Result<u64> {
        let (width, height) = (u32::try_from(width)?, u32::try_from(height)?);
        match self {
            Self::ZImage => crate::memory::image_peak(width, height, control, loras),
            Self::QwenImage => crate::memory::qwen_image_serve_peak(width, height),
        }
    }

    /// A lower bound on the weights that stay allocated between requests,
    /// which a warm request does not reserve again. Z-Image keeps its encoder
    /// and its pipeline; Qwen-Image 2.1 keeps the pipeline alone, 15.7 GB, its
    /// encoder being loaded and released inside every request.
    const fn resident_floor(self) -> u64 {
        match self {
            Self::ZImage => 16 * 1024 * 1024 * 1024,
            Self::QwenImage => 14 * 1024 * 1024 * 1024,
        }
    }

    /// Why neither a negative prompt nor a guidance scale is taken.
    fn guidance_refusal(self) -> String {
        let name = self.entry().full_name();
        match self {
            Self::ZImage => format!(
                "{name} is distilled to run without guidance, so a negative prompt or a \
                 guidance_scale would be silently ignored; leave them out"
            ),
            Self::QwenImage => format!(
                "{name} is served without classifier-free guidance, the way its model card \
                 samples it, so a negative prompt or a guidance_scale would be silently \
                 ignored; leave them out"
            ),
        }
    }
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
    /// Which pipeline renders it.
    pub pipeline: Pipeline,
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
    Unavailable(String),
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

impl ImageJob {
    fn is_closed(&self) -> bool {
        match self {
            Self::Render { reply, .. } => reply.is_closed(),
            Self::Preprocess { reply, .. } => reply.is_closed(),
        }
    }

    fn fail(self, error: ImageError) {
        match self {
            Self::Render { reply, .. } => {
                let _ = reply.send(Err(error));
            }
            Self::Preprocess { reply, .. } => {
                let _ = reply.send(Err(error));
            }
        }
    }

    fn peak(&self) -> Result<u64> {
        match self {
            Self::Render { params, inputs, .. } => params.pipeline.peak(
                params.width,
                params.height,
                inputs.control.is_some(),
                inputs.loras.len(),
            ),
            Self::Preprocess { image, .. } => {
                Pipeline::ZImage.peak(image.dim(2)?, image.dim(1)?, true, 0)
            }
        }
    }
}

static IMAGE_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// The guard travels with the queued job, including failed sends and shutdown drains.
struct ImageTrace {
    record: ImageRecord,
    logger: ServeLogger,
    started: Option<Instant>,
}

impl ImageTrace {
    fn new(job: &ImageJob, logger: ServeLogger) -> Self {
        let (width, height, steps, images, model) = match job {
            ImageJob::Render { params, .. } => (
                params.width,
                params.height,
                params.steps,
                params.n as usize,
                Some(params.pipeline.entry().full_name()),
            ),
            ImageJob::Preprocess { image, .. } => (
                image.dim(2).unwrap_or(0),
                image.dim(1).unwrap_or(0),
                0,
                1,
                None,
            ),
        };
        let preprocessing = model.is_none();
        let activity = ImageActivity {
            id: IMAGE_REQUEST_ID.fetch_add(1, Ordering::Relaxed),
            queued_at: Instant::now(),
            model: model.unwrap_or("preprocess").into(),
            width,
            height,
            steps,
            images,
            preprocessing,
        };
        logger.log(ServeLog::ImageQueued(activity.clone()));
        Self {
            record: ImageRecord {
                activity,
                completed_images: 0,
                elapsed_secs: 0.0,
                encode_secs: 0.0,
                denoise_secs: 0.0,
                vae_secs: 0.0,
                executed_steps: 0,
                error: Some("image request interrupted before completion".into()),
                cancelled: true,
            },
            logger,
            started: None,
        }
    }

    fn picked(&mut self) {
        self.started = Some(Instant::now());
        self.logger.log(ServeLog::ImagePicked {
            id: self.record.activity.id,
        });
    }

    fn failed(&mut self, error: &ImageError, cancelled: bool) {
        self.record.error = Some(match error {
            ImageError::Request(message)
            | ImageError::Render(message)
            | ImageError::Unavailable(message) => message.clone(),
        });
        self.record.cancelled = cancelled;
    }

    fn finished<T>(&mut self, result: &Result<T, ImageError>, cancelled: bool) {
        match result {
            Ok(_) => {
                self.record.error = None;
                self.record.cancelled = cancelled;
            }
            Err(error) => self.failed(error, cancelled),
        }
    }

    fn completed_image(&mut self) {
        self.record.completed_images += 1;
        self.logger.log(ServeLog::ImageProgress {
            id: self.record.activity.id,
            completed_images: self.record.completed_images,
        });
    }
}

impl Drop for ImageTrace {
    fn drop(&mut self) {
        self.record.elapsed_secs = self.started.map_or(0.0, |at| at.elapsed().as_secs_f64());
        self.logger
            .log(ServeLog::ImageDone(Box::new(self.record.clone())));
    }
}

struct QueuedImage {
    job: ImageJob,
    trace: ImageTrace,
}

impl QueuedImage {
    fn new(job: ImageJob, logger: ServeLogger) -> Self {
        let trace = ImageTrace::new(&job, logger);
        Self { job, trace }
    }
}

/// The handler's end of the engine: the queue and the residency flag.
pub struct Handle {
    sender: crossbeam_channel::Sender<QueuedImage>,
    logger: ServeLogger,
    /// Whether the encoder and the pipeline are resident, set and cleared by
    /// the engine thread, read by `/health`.
    pub resident: Arc<AtomicBool>,
    /// Which pipeline that is, written by the engine thread beside the flag.
    resident_model: Arc<ResidentModel>,
}

/// The resident pipeline as `/health` reads it. The flag above is what the TUI
/// holds; this names the entry, and is cleared in the same place.
#[derive(Debug, Default)]
pub(crate) struct ResidentModel(std::sync::Mutex<Option<Pipeline>>);

impl ResidentModel {
    pub(crate) fn set(&self, pipeline: Option<Pipeline>) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = pipeline;
    }

    fn get(&self) -> Option<Pipeline> {
        *self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The full name to show for a resident pipeline. A caller that knows one
    /// is resident and finds no name here gets the default pipeline's.
    pub(crate) fn name(&self) -> &'static str {
        self.get().unwrap_or(Pipeline::DEFAULT).entry().full_name()
    }
}

impl Handle {
    fn try_send(&self, job: ImageJob) -> Result<(), crossbeam_channel::TrySendError<ImageJob>> {
        match self
            .sender
            .try_send(QueuedImage::new(job, self.logger.clone()))
        {
            Ok(()) => Ok(()),
            Err(crossbeam_channel::TrySendError::Full(mut queued)) => {
                queued.trace.failed(
                    &ImageError::Unavailable("the image queue is full".into()),
                    false,
                );
                Err(crossbeam_channel::TrySendError::Full(queued.job))
            }
            Err(crossbeam_channel::TrySendError::Disconnected(mut queued)) => {
                queued.trace.failed(
                    &ImageError::Unavailable("the image engine is not running".into()),
                    false,
                );
                Err(crossbeam_channel::TrySendError::Disconnected(queued.job))
            }
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.resident.load(Ordering::Relaxed)
    }

    /// The full name of the resident image pipeline, or `None` when none is.
    pub fn resident_model(&self) -> Option<&'static str> {
        self.resident_model
            .get()
            .map(|pipeline| pipeline.entry().full_name())
    }

    /// A handle with no engine behind it, for tests that build an `AppState`
    /// and never render: a send fails as disconnected.
    #[cfg(test)]
    pub(crate) fn detached() -> Self {
        let (sender, _receiver) = crossbeam_channel::bounded(0);
        Self {
            sender,
            logger: ServeLogger::discarding(),
            resident: Arc::new(AtomicBool::new(false)),
            resident_model: Arc::default(),
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
    resident: Arc<AtomicBool>,
    resident_model: Arc<ResidentModel>,
) -> (Handle, std::thread::JoinHandle<()>) {
    let (sender, receiver) = crossbeam_channel::bounded::<QueuedImage>(QUEUE_CAPACITY);
    // Said once at startup, because a step count nobody asked for in a request
    // is otherwise only visible in each render's own log line. The pipeline's
    // own default needs no announcement.
    if let Some(steps) = settings.image_steps.filter(|&n| n != DEFAULT_STEPS) {
        logger.log(ServeLog::HostLine(format!(
            "xwen serve: {} images render in {steps} steps unless a request names its own \
             (pipeline default {DEFAULT_STEPS})",
            Pipeline::ZImage.entry().full_name()
        )));
    }
    let idle = settings.idle_unload;
    let wait_timeout = settings.queue_timeout;
    let thread_resident = Residency {
        flag: Arc::clone(&resident),
        model: Arc::clone(&resident_model),
    };
    let handle_logger = logger.clone();
    let thread = std::thread::Builder::new()
        .name(IMAGE_ENGINE_THREAD.to_string())
        .spawn(move || {
            engine_loop(
                receiver,
                idle,
                wait_timeout,
                thread_resident,
                shutdown,
                logger,
            )
        })
        .expect("spawning the image engine thread");
    (
        Handle {
            sender,
            resident,
            resident_model,
            logger: handle_logger,
        },
        thread,
    )
}

/// The engine thread's end of the residency flags: the bool the TUI and
/// `/health` read, and the name `/health` puts beside it.
struct Residency {
    flag: Arc<AtomicBool>,
    model: Arc<ResidentModel>,
}

impl Residency {
    fn loaded(&self, pipeline: Pipeline, logger: &ServeLogger) {
        self.model.set(Some(pipeline));
        self.flag.store(true, Ordering::Relaxed);
        logger.log(ServeLog::ImageResidency { loaded: true });
    }

    fn cleared(&self, logger: &ServeLogger) {
        self.model.set(None);
        if self.flag.swap(false, Ordering::Relaxed) {
            logger.log(ServeLog::ImageResidency { loaded: false });
        }
    }
}

/// What is resident, as the swap decision reads it.
#[derive(Clone, Copy)]
struct Held<'a> {
    pipeline: Pipeline,
    loras: &'a [crate::zimage::lora::ResolvedLora],
    control: Option<&'a ControlIdentity>,
}

impl Held<'_> {
    /// Whether what is resident renders `wanted` as it stands. Anything else
    /// is unloaded before the load that replaces it, so two pipelines, or one
    /// pipeline under two adapter sets, are never held together.
    fn serves(&self, wanted: &Held<'_>) -> bool {
        self.pipeline == wanted.pipeline
            && self.loras == wanted.loras
            && self.control == wanted.control
    }
}

/// The one resident image pipeline.
enum Loaded {
    ZImage(Box<ZImageLoaded>),
    QwenImage(QwenImageLoaded),
}

impl Loaded {
    fn device(&self) -> &candle_core::Device {
        match self {
            Self::ZImage(held) => held.encoder.device(),
            Self::QwenImage(held) => &held.device,
        }
    }

    /// The pipeline whose weights are on the device, which for Qwen-Image 2.1
    /// is none until its first request has loaded them.
    fn resident_pipeline(&self) -> Option<Pipeline> {
        match self {
            Self::ZImage(_) => Some(Pipeline::ZImage),
            Self::QwenImage(held) => held.pipeline.as_ref().map(|_| Pipeline::QwenImage),
        }
    }

    fn held(&self) -> Held<'_> {
        match self {
            Self::ZImage(held) => Held {
                pipeline: Pipeline::ZImage,
                loras: &held.loras,
                control: held.control.as_ref(),
            },
            Self::QwenImage(_) => Held {
                pipeline: Pipeline::QwenImage,
                loras: &[],
                control: None,
            },
        }
    }
}

/// Z-Image's encoder, its pipeline and what they were opened from.
struct ZImageLoaded {
    encoder: crate::XwenModel,
    pipeline: ZImagePipeline,
    encoder_entry: Model,
    spec: crate::hub::EncoderSpec,
    tokenizer_path: std::path::PathBuf,
    loras: Vec<crate::zimage::lora::ResolvedLora>,
    control: Option<ControlIdentity>,
}

impl Drop for ZImageLoaded {
    fn drop(&mut self) {
        let _drain = crate::memory::DeviceDrain(self.encoder.device().clone());
    }
}

fn uncached(pipeline: Pipeline) -> ImageError {
    let entry = pipeline.entry();
    ImageError::Request(format!(
        "{} is not in the Hugging Face cache: run `xwen fetch --model {entry}` on the server \
         first",
        entry.full_name()
    ))
}

impl ZImageLoaded {
    /// Open everything `xwen image` opens, from the cached snapshot.
    fn open(
        logger: &ServeLogger,
        loras: &[crate::zimage::lora::ResolvedLora],
        control: Option<&ControlIdentity>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Self, ImageError> {
        check().map_err(|e| ImageError::Unavailable(e.to_string()))?;
        let index = crate::hub::cached_model(Model::ZImageTurbo)
            .ok_or_else(|| uncached(Pipeline::ZImage))?;
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
            let _drain = crate::memory::DeviceDrain(device.clone());

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
            check()?;

            let started = Instant::now();
            let pipeline = ZImagePipeline::load_cancellable(
                root,
                &device,
                &adapters,
                control.map(|c| c.path.as_path()),
                check,
            )?;
            check()?;
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
        trace: &mut ImageTrace,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Vec<RenderedImage>> {
        check()?;
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
        self.encoder.device().synchronize()?;
        let encoding = Instant::now();
        let encoded = self.encoder.encode_spec(&rendered.ids, &self.spec);
        let drained = self.encoder.device().synchronize();
        trace.record.encode_secs += encoding.elapsed().as_secs_f64();
        drained?;
        let (cap_feats, _n_tokens) = encoded?;
        check()?;
        let first_seed = params.seed.unwrap_or_else(drawn_seed);
        let mut out = Vec::with_capacity(params.n as usize);
        for i in 0..params.n as u64 {
            check()?;
            let seed = first_seed.wrapping_add(i);
            let started = Instant::now();
            let options = ImageOptions {
                width: params.width,
                height: params.height,
                steps: params.steps,
                seed,
                latents: None,
            };
            let run = self.pipeline.generate_cancellable(
                &cap_feats,
                &options,
                inputs.edit.as_ref(),
                inputs.control.as_ref().map(|c| &c.image),
                check,
            )?;
            trace.record.denoise_secs += run.timings.steps.iter().sum::<f64>();
            trace.record.vae_secs += run.timings.vae_decode;
            trace.record.executed_steps += run.timings.steps.len();
            check()?;
            let png = encode_png(&run.image)?;
            logger.log(ServeLog::HostLine(format!(
                "xwen serve: image rendered {}x{} in {} steps, {:.1}s (seed {seed})",
                params.width,
                params.height,
                params.steps,
                started.elapsed().as_secs_f64()
            )));
            let control_map = inputs
                .control
                .as_ref()
                .map(|c| encode_png(&c.image.image))
                .transpose()?;
            out.push(RenderedImage {
                png,
                seed,
                start_step: run.start_step,
                control_map,
            });
            trace.completed_image();
        }
        Ok(out)
    }
}

/// Whether Qwen-Image 2.1's text encoder outlives the request that loaded it.
/// It does not: it is 15.7 GB beside a pipeline of the same size and it works
/// for a fraction of a second per request, so every request loads it, encodes
/// and releases it, which costs the load and nothing else. Keeping it warm is
/// this constant, together with the admission figure
/// (`memory::qwen_image_serve_peak`), which prices the encoder as a transient
/// of the request rather than as weights that stay.
const KEEP_QWEN_IMAGE_ENCODER: bool = false;

/// Qwen-Image 2.1 as the engine holds it: where the cached snapshot is, and
/// whichever of the encoder and the pipeline is on the device right now.
struct QwenImageLoaded {
    device: candle_core::Device,
    root: std::path::PathBuf,
    encoder_dir: std::path::PathBuf,
    encoder_entry: Model,
    spec: crate::hub::EncoderSpec,
    encoder: Option<crate::XwenModel>,
    pipeline: Option<QwenImagePipeline>,
}

impl Drop for QwenImageLoaded {
    fn drop(&mut self) {
        let _drain = crate::memory::DeviceDrain(self.device.clone());
    }
}

impl QwenImageLoaded {
    /// Resolve the cached snapshot. Nothing loads here: the encoder goes first
    /// and leaves before the pipeline arrives, as it does on `xwen image`.
    fn open() -> Result<Self, ImageError> {
        let entry = Pipeline::QwenImage.entry();
        let index = crate::hub::cached_model(entry).ok_or_else(|| uncached(Pipeline::QwenImage))?;
        let resolve = || -> Result<Self> {
            let root = index
                .parent()
                .context("the cached model_index.json has no parent directory")?
                .to_path_buf();
            let encoder_entry = entry
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
            Ok(Self {
                device: crate::gguf::metal_device()?,
                root,
                encoder_dir,
                encoder_entry,
                spec,
                encoder: None,
                pipeline: None,
            })
        };
        resolve().map_err(classify_load_error)
    }

    /// The caption rows for `rendered`, the prompt the plan already validated,
    /// on the CPU so that nothing of the encoder's is referenced from the device
    /// once it goes.
    fn caption(
        &mut self,
        rendered: &RenderedPrompt,
        logger: &ServeLogger,
        trace: &mut ImageTrace,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<candle_core::Tensor, ImageError> {
        let mut encode = || -> Result<candle_core::Tensor> {
            check()?;
            if self.encoder.is_none() {
                let started = Instant::now();
                let source = crate::CheckpointSource::open(
                    &self.encoder_dir,
                    &self.device,
                    Some(self.encoder_entry),
                )?;
                self.encoder = Some(crate::XwenModel::load_encoder(
                    source,
                    self.spec.max_tokens,
                )?);
                self.device.synchronize()?;
                let secs = started.elapsed().as_secs_f64();
                trace.record.encode_secs += secs;
                logger.log(ServeLog::HostLine(format!(
                    "xwen serve: image text encoder loaded in {secs:.1}s"
                )));
                crate::memory::log_event("image text encoder loaded", Some(&self.device));
            }
            check()?;
            let encoder = self.encoder.as_mut().expect("the encoder was just loaded");
            let started = Instant::now();
            let encoded = encoder.encode_spec(&rendered.ids, &self.spec);
            let drained = self.device.synchronize();
            trace.record.encode_secs += started.elapsed().as_secs_f64();
            drained?;
            let (hidden, n_tokens) = encoded?;
            let cap_feats = hidden
                .narrow(0, rendered.drop, n_tokens - rendered.drop)?
                .to_device(&candle_core::Device::Cpu)?
                .contiguous()?;
            Ok(cap_feats)
        };
        let encoded = encode();
        if !KEEP_QWEN_IMAGE_ENCODER || encoded.is_err() {
            self.release_encoder();
        }
        // After the cleanup, and by the same rule the pipeline load and the
        // render follow: a cancelled or shut-down request is a 503, not a 500.
        encoded.map_err(|e| {
            let message = format!("encoding the prompt: {e:#}");
            if check().is_err() {
                ImageError::Unavailable(message)
            } else {
                ImageError::Render(message)
            }
        })
    }

    /// Drop the encoder and drain, so its buffers are back with the device
    /// before the pipeline loads or a step allocates.
    fn release_encoder(&mut self) {
        if self.encoder.is_none() {
            return;
        }
        drain_or_abort(&self.device, "image text encoder");
        self.encoder = None;
        drain_or_abort(&self.device, "image text encoder buffers");
        crate::memory::log_event("image text encoder released", Some(&self.device));
    }

    /// Load the transformer and the VAE when they are not resident. True when
    /// this call loaded them.
    fn ensure_pipeline(
        &mut self,
        logger: &ServeLogger,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<bool, ImageError> {
        if self.pipeline.is_some() {
            return Ok(false);
        }
        let started = Instant::now();
        let pipeline = QwenImagePipeline::load_cancellable(&self.root, &self.device, check)
            .and_then(|pipeline| {
                self.device.synchronize()?;
                Ok(pipeline)
            })
            .map_err(classify_load_error)?;
        self.pipeline = Some(pipeline);
        logger.log(ServeLog::HostLine(format!(
            "xwen serve: {} transformer and VAE loaded in {:.1}s",
            Pipeline::QwenImage.entry().full_name(),
            started.elapsed().as_secs_f64()
        )));
        Ok(true)
    }

    fn render(
        &mut self,
        params: &ImageParams,
        cap_feats: &candle_core::Tensor,
        logger: &ServeLogger,
        trace: &mut ImageTrace,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Vec<RenderedImage>> {
        let pipeline = self
            .pipeline
            .as_ref()
            .context("the Qwen-Image 2.1 pipeline is not loaded")?;
        // The request was admitted on the VAE arm the environment names. The
        // loaded pipeline knows the arm it resolved to, and a render it prices
        // above what was admitted is refused rather than run on a wrong number.
        let admitted = Pipeline::QwenImage.peak(params.width, params.height, false, 0)?;
        let loaded = pipeline.loaded_peak_bytes(params.width, params.height)?;
        anyhow::ensure!(
            loaded <= admitted,
            "the loaded Qwen-Image 2.1 pipeline prices this render at {loaded} bytes, above the \
             {admitted} it was admitted on: its VAE resolved to another arm than admission assumed"
        );
        let first_seed = params.seed.unwrap_or_else(drawn_seed);
        let mut out = Vec::with_capacity(params.n as usize);
        for i in 0..params.n as u64 {
            check()?;
            let seed = first_seed.wrapping_add(i);
            let started = Instant::now();
            let options = crate::qwen_image::pipeline::ImageOptions {
                width: params.width,
                height: params.height,
                steps: params.steps,
                seed,
                latents: None,
            };
            let run = pipeline.generate_cancellable(cap_feats, &options, check)?;
            trace.record.denoise_secs += run.timings.steps.iter().sum::<f64>();
            trace.record.vae_secs += run.timings.vae_decode;
            trace.record.executed_steps += run.timings.steps.len();
            check()?;
            // Whatever the pipeline decided: RGBA when the decoded alpha plane
            // holds a transparent region, RGB otherwise.
            let png = encode_png(&run.image)?;
            logger.log(ServeLog::HostLine(format!(
                "xwen serve: image rendered {}x{} in {} steps, {:.1}s (seed {seed}, alpha min {}, \
                 {} clear pixels, {})",
                params.width,
                params.height,
                params.steps,
                started.elapsed().as_secs_f64(),
                run.alpha_min,
                run.clear_pixels,
                if run.image.dim(0)? == 4 {
                    "RGBA"
                } else {
                    "RGB"
                }
            )));
            out.push(RenderedImage {
                png,
                seed,
                start_step: 0,
                control_map: None,
            });
            trace.completed_image();
        }
        Ok(out)
    }
}

/// What a render settled before it was allowed to cost anything: whether the
/// resident pipeline has to go first, and the prompt as the encoder will read
/// it.
#[derive(Debug)]
struct Plan {
    unload_first: bool,
    prompt: Option<RenderedPrompt>,
}

/// Everything about a render that can be refused without touching the device,
/// decided before the swap and before memory is admitted, so that a request
/// which was never going to run evicts nothing and waits for nothing.
///
/// `resident` is what is loaded now. `cached` answers whether an entry's whole
/// file set is on disk, which is asked only when this request loads or replaces
/// a pipeline: a resident pipeline that serves the request reopens none of its
/// files, and a file that went missing under it is no reason to refuse.
/// `render_prompt` renders and checks the prompt from the tokenizer alone; it
/// runs on every Qwen-Image 2.1 request, that model reopening its tokenizer and
/// its encoder each time.
fn plan(
    params: &ImageParams,
    wanted: &Held<'_>,
    resident: Option<&Held<'_>>,
    cached: &dyn Fn(Model) -> bool,
    render_prompt: &dyn Fn(&ImageParams) -> Result<RenderedPrompt, ImageError>,
) -> Result<Plan, ImageError> {
    let serves = resident.is_some_and(|held| held.serves(wanted));
    if !serves && !cached(params.pipeline.entry()) {
        return Err(uncached(params.pipeline));
    }
    let prompt = match params.pipeline {
        Pipeline::ZImage => None,
        Pipeline::QwenImage => Some(render_prompt(params)?),
    };
    Ok(Plan {
        unload_first: resident.is_some() && !serves,
        prompt,
    })
}

/// Qwen-Image 2.1's prompt, rendered and checked from the cached tokenizer.
/// The prompt's own faults are request faults: too many tokens, or too many to
/// sit beside an image of this size on the rope. A tokenizer that is not there
/// is the operator's to fetch, and anything else is the server's.
fn qwen_image_prompt(params: &ImageParams) -> Result<RenderedPrompt, ImageError> {
    let entry = Pipeline::QwenImage.entry();
    let encoder_entry = entry
        .text_encoder()
        .ok_or_else(|| ImageError::Render("the pipeline entry names no text encoder".into()))?;
    // What every request reopens: the tokenizer here, the encoder's files in
    // `caption`. Both come from the encoder entry's file set.
    if crate::hub::cached_model(encoder_entry).is_none() {
        return Err(uncached(Pipeline::QwenImage));
    }
    let tokenizer = encoder_entry
        .safetensors_tokenizer()
        .and_then(|file| crate::hub::cached_file(entry.repo(), file))
        .ok_or_else(|| uncached(Pipeline::QwenImage))?;
    check_qwen_image_prompt(encoder_entry, &tokenizer, params)
}

fn check_qwen_image_prompt(
    encoder_entry: Model,
    tokenizer: &Path,
    params: &ImageParams,
) -> Result<RenderedPrompt, ImageError> {
    let rendered =
        crate::qwen_image::conditioning::prompt_ids(encoder_entry, tokenizer, &params.prompt, &[])
            .map_err(|e| match e.downcast_ref::<PromptTooLong>() {
                Some(too_long) => ImageError::Request(format!("prompt: {too_long}")),
                None => ImageError::Render(format!("rendering the prompt: {e:#}")),
            })?;
    QwenImagePipeline::check_layout(rendered.kept_ids().len(), params.width, params.height)
        .map_err(|e| ImageError::Request(format!("prompt: {e:#}")))?;
    Ok(rendered)
}

/// A drawn seed stays under 2^53: the seed goes back to the client as a JSON
/// number, and a JavaScript client rounds anything wider, so a seed it echoed
/// back would not reproduce the image it came with. A seed the client chose is
/// used as given, whatever its width.
fn drawn_seed() -> u64 {
    rand::random::<u64>() >> 11
}

/// A synchronization failure means ownership cannot safely transfer.
fn drain_or_abort(device: &candle_core::Device, what: &str) {
    if let Err(e) = device.synchronize() {
        eprintln!("xwen: {what} could not drain: {e}; terminating before another load");
        std::process::abort();
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
    jobs: crossbeam_channel::Receiver<QueuedImage>,
    idle_unload: Option<Duration>,
    wait_timeout: Duration,
    resident: Residency,
    shutdown: Arc<super::types::Cancel>,
    logger: ServeLogger,
) {
    // Declared before model storage so unwinding drops GPU owners first.
    let mut lease: Option<crate::memory::Lease> = None;
    let mut loaded: Option<Loaded> = None;
    let mut preprocessor = crate::zimage::preprocess::Preprocessor::new();
    let mut last_finished = Instant::now();
    loop {
        let yielding = lease.as_ref().is_some_and(|held| held.should_yield());
        let expired =
            lease.is_some() && idle_unload.is_some_and(|idle| last_finished.elapsed() >= idle);
        if yielding || expired {
            unload_images(&mut loaded, &mut preprocessor, &resident, &logger);
            lease = None;
            logger.log(ServeLog::HostLine(format!(
                "xwen serve: image resources unloaded ({})",
                if yielding {
                    "memory ownership requested"
                } else {
                    "idle"
                }
            )));
        }
        if shutdown.is_cancelled() {
            break;
        }
        let QueuedImage { job, mut trace } = match jobs.recv_timeout(Duration::from_millis(100)) {
            Ok(job) => job,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        };
        if job.is_closed() {
            continue;
        }
        trace.picked();
        let peak = match job.peak() {
            Ok(peak) => peak,
            Err(e) => {
                let error = ImageError::Request(e.to_string());
                trace.failed(&error, false);
                job.fail(error);
                continue;
            }
        };
        let mut prompt = None;
        if let ImageJob::Render { inputs, params, .. } = &job {
            let wanted = Held {
                pipeline: params.pipeline,
                loras: &inputs.loras,
                control: inputs.control.as_ref().map(|c| &c.checkpoint),
            };
            let settled = plan(
                params,
                &wanted,
                loaded.as_ref().map(Loaded::held).as_ref(),
                &|entry| crate::hub::cached_model(entry).is_some(),
                &qwen_image_prompt,
            );
            let settled = match settled {
                Ok(settled) => settled,
                Err(error) => {
                    trace.failed(&error, false);
                    job.fail(error);
                    continue;
                }
            };
            if settled.unload_first {
                let leaving = loaded.as_ref().map(|held| held.held().pipeline);
                unload_images(&mut loaded, &mut preprocessor, &resident, &logger);
                if leaving.is_some_and(|leaving| leaving != params.pipeline) {
                    logger.log(ServeLog::HostLine(format!(
                        "xwen serve: image resources unloaded ({} requested)",
                        params.pipeline.entry().full_name()
                    )));
                }
            }
            prompt = settled.prompt;
        }
        let waiting_since = Instant::now();
        let check = || -> Result<()> {
            anyhow::ensure!(!shutdown.is_cancelled(), "the server is shutting down");
            anyhow::ensure!(!job.is_closed(), "the image client disconnected");
            anyhow::ensure!(
                waiting_since.elapsed() < wait_timeout,
                "timed out waiting for image memory ownership; retry shortly"
            );
            Ok(())
        };
        if lease.is_none() {
            match crate::memory::acquire("images", &check) {
                Ok(held) => lease = Some(held),
                Err(e) => {
                    let error = ImageError::Unavailable(e.to_string());
                    trace.failed(&error, shutdown.is_cancelled() || job.is_closed());
                    job.fail(error);
                    continue;
                }
            }
        }
        if let Err(e) = check().and_then(|_| {
            crate::memory::admit_additional_until(
                "image request",
                image_allocation_reserve(peak, loaded.as_ref().and_then(Loaded::resident_pipeline)),
                &check,
            )
        }) {
            let error = ImageError::Unavailable(e.to_string());
            trace.failed(&error, shutdown.is_cancelled() || job.is_closed());
            job.fail(error);
            unload_images(&mut loaded, &mut preprocessor, &resident, &logger);
            lease = None;
            continue;
        }
        let (params, mut inputs, reply) = match job {
            ImageJob::Preprocess { image, kind, reply } => {
                let result =
                    preprocess_on_worker(&mut preprocessor, &image, kind).and_then(|image| {
                        encode_png(&image).map_err(|e| ImageError::Render(format!("{e:#}")))
                    });
                trace.finished(&result, shutdown.is_cancelled() || reply.is_closed());
                if result.is_ok() {
                    trace.completed_image();
                }
                let _ = reply.send(result);
                last_finished = Instant::now();
                crate::memory::log_event("image preprocess finished", None);
                continue;
            }
            ImageJob::Render {
                params,
                inputs,
                reply,
            } => (params, inputs, reply),
        };
        let check = || -> Result<()> {
            anyhow::ensure!(!shutdown.is_cancelled(), "the server is shutting down");
            anyhow::ensure!(!reply.is_closed(), "the image client disconnected");
            Ok(())
        };
        let result = (|| -> Result<Vec<RenderedImage>, ImageError> {
            if let Some(control) = &mut inputs.control {
                control.image.image = preprocess_on_worker(
                    &mut preprocessor,
                    &control.image.image,
                    control.preprocess,
                )?;
            }
            check().map_err(|e| ImageError::Unavailable(e.to_string()))?;
            if let Some(note) = &params.model_note {
                logger.log(ServeLog::HostLine(format!(
                    "xwen serve: images: model {note:?} served by {} (proxy path)",
                    params.pipeline.entry().full_name()
                )));
            }
            logger.log(ServeLog::HostLine(format!(
                "xwen serve: image request for {}, {}x{}, {} steps, {} images, {} LoRAs, \
                 control {}",
                params.pipeline.entry().full_name(),
                params.width,
                params.height,
                params.steps,
                params.n,
                inputs.loras.len(),
                inputs.control.is_some()
            )));
            let interrupted = |e: anyhow::Error| {
                if check().is_err() {
                    ImageError::Unavailable(format!("{e:#}"))
                } else {
                    ImageError::Render(format!("{e:#}"))
                }
            };
            match params.pipeline {
                Pipeline::ZImage => {
                    if loaded.is_none() {
                        let opened = ZImageLoaded::open(
                            &logger,
                            &inputs.loras,
                            inputs.control.as_ref().map(|c| &c.checkpoint),
                            &check,
                        )?;
                        crate::memory::log_event(
                            "image model loaded",
                            Some(opened.encoder.device()),
                        );
                        loaded = Some(Loaded::ZImage(Box::new(opened)));
                        resident.loaded(Pipeline::ZImage, &logger);
                    }
                    let Some(Loaded::ZImage(held)) = loaded.as_mut() else {
                        unreachable!("the swap left another pipeline resident");
                    };
                    held.render(&params, &inputs, &logger, &mut trace, &check)
                        .map_err(interrupted)
                }
                Pipeline::QwenImage => {
                    if loaded.is_none() {
                        loaded = Some(Loaded::QwenImage(QwenImageLoaded::open()?));
                    }
                    let Some(Loaded::QwenImage(held)) = loaded.as_mut() else {
                        unreachable!("the swap left another pipeline resident");
                    };
                    let rendered = prompt.as_ref().ok_or_else(|| {
                        ImageError::Render("the prompt was not rendered before the load".into())
                    })?;
                    let cap_feats = held.caption(rendered, &logger, &mut trace, &check)?;
                    check().map_err(|e| ImageError::Unavailable(e.to_string()))?;
                    if held.ensure_pipeline(&logger, &check).map_err(|e| match e {
                        ImageError::Render(message) if check().is_err() => {
                            ImageError::Unavailable(message)
                        }
                        other => other,
                    })? {
                        crate::memory::log_event("image model loaded", Some(&held.device));
                        resident.loaded(Pipeline::QwenImage, &logger);
                    }
                    held.render(&params, &cap_feats, &logger, &mut trace, &check)
                        .map_err(interrupted)
                }
            }
        })();
        trace.finished(&result, shutdown.is_cancelled() || reply.is_closed());
        // A request fault leaves a resident pipeline where it is: an adapter
        // that will not load is not a reason to make the next request pay the
        // load again.
        let failed = match &result {
            Ok(_) => false,
            Err(ImageError::Request(_)) => loaded
                .as_ref()
                .and_then(Loaded::resident_pipeline)
                .is_none(),
            Err(_) => true,
        };
        let _ = reply.send(result);
        last_finished = Instant::now();
        crate::memory::log_event(
            "image request finished",
            loaded.as_ref().map(Loaded::device),
        );
        if failed {
            unload_images(&mut loaded, &mut preprocessor, &resident, &logger);
            lease = None;
        }
    }
    // Pending requests must finish their telemetry before server handles are dropped.
    for queued in jobs.try_iter() {
        drop(queued);
    }
    unload_images(&mut loaded, &mut preprocessor, &resident, &logger);
    drop(lease);
}

/// Only credit a lower bound on weights that remain allocated. Every request
/// reserves its temporary working space again, even after an earlier warm render.
fn image_allocation_reserve(peak: u64, resident: Option<Pipeline>) -> u64 {
    peak.saturating_sub(resident.map_or(0, Pipeline::resident_floor))
}

fn unload_images(
    loaded: &mut Option<Loaded>,
    preprocessor: &mut crate::zimage::preprocess::Preprocessor,
    resident: &Residency,
    logger: &ServeLogger,
) {
    let device = loaded.as_ref().map(|held| held.device().clone());
    if let Some(device) = &device {
        // A synchronization failure means ownership cannot safely transfer.
        if let Err(e) = device.synchronize() {
            eprintln!("xwen: image device could not drain: {e}; terminating before another load");
            std::process::abort();
        }
    }
    *loaded = None;
    *preprocessor = crate::zimage::preprocess::Preprocessor::new();
    if let Some(device) = &device {
        if let Err(e) = device.synchronize() {
            eprintln!("xwen: image buffers could not drain: {e}; terminating before another load");
            std::process::abort();
        }
    }
    resident.cleared(logger);
    crate::memory::log_event("image resources released", device.as_ref());
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
/// `steps` overrides and `None` falls through to the pipeline's own
/// ([`Pipeline::steps_default`]). It is a
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

    let (pipeline, model_note) = select_pipeline(request.model.as_deref(), proxy)?;

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
    pipeline
        .check_size(width, height)
        .map_err(|e| bad_param("size", e.to_string()))?;
    pipeline
        .peak(width, height, false, 0)
        .map_err(|e| bad_param("size", e.to_string()))?;

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
    let guided = pipeline.guidance_refusal();
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
        (a, b) => a
            .or(b)
            .unwrap_or_else(|| pipeline.steps_default(server_steps)),
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
        pipeline,
    })
}

/// The pipeline a request's `model` selects, and on the proxy path the string
/// that was accepted in place of a name.
///
/// A full name selects its pipeline on every path. Nothing selects the
/// default. Anything else is a 400 on the two canonical paths, as it is on
/// every LM route, and on the proxy path it is served by the default pipeline
/// and logged, the ComfyUI node's dropdown holding only OpenAI's names.
pub(crate) fn select_pipeline(
    model: Option<&str>,
    proxy: bool,
) -> Result<(Pipeline, Option<String>), ApiError> {
    match model.map(str::trim) {
        None | Some("") => Ok((Pipeline::DEFAULT, None)),
        Some(name) => match Pipeline::from_full_name(name) {
            Some(pipeline) => Ok((pipeline, None)),
            None if proxy => Ok((Pipeline::DEFAULT, Some(name.to_string()))),
            None => Err(bad_param(
                "model",
                format!(
                    "unknown model {name:?}: the images routes serve {}; name one in full or \
                     leave the field out for {}",
                    Pipeline::ALL.map(|p| p.entry().full_name()).join(" and "),
                    Pipeline::DEFAULT.entry().full_name()
                ),
            )),
        },
    }
}

/// The 400 for an input `pipeline` has no path for, or `Ok` when it has one.
/// `given` is the request's fields of that kind which are present, by name.
pub(crate) fn refuse_missing_controls(pipeline: Pipeline, given: &[&str]) -> Result<(), ApiError> {
    let Some(first) = given.first() else {
        return Ok(());
    };
    let controls = pipeline.controls();
    if controls.init_image && controls.mask && controls.control && controls.loras {
        return Ok(());
    }
    Err(bad_param(
        first,
        format!(
            "{} renders from a prompt alone on this server: {} {} {}'s; leave {} out or \
             name that model",
            pipeline.entry().full_name(),
            given.join(", "),
            if given.len() == 1 { "is" } else { "are" },
            Pipeline::ZImage.entry().full_name(),
            if given.len() == 1 { "it" } else { "them" },
        ),
    ))
}

/// One entry of `GET /v1/images/models`.
fn model_listing(pipeline: Pipeline, server_steps: Option<usize>) -> serde_json::Value {
    let controls = pipeline.controls();
    json!({
        "id": pipeline.entry().full_name(),
        "object": "model",
        "default": pipeline == Pipeline::DEFAULT,
        // What a request naming no step count renders at, which for Z-Image is
        // the operator's `--image-steps` when there is one.
        "default_steps": pipeline.steps_default(server_steps),
        "max_steps": MAX_STEPS,
        "default_size": format!("{}x{}", DEFAULT_SIZE.0, DEFAULT_SIZE.1),
        "size_multiple": pipeline.size_multiple(),
        "size_rule": pipeline.size_rule(),
        "max_pixels": crate::memory::IMAGE_MAX_PIXELS,
        "max_references": 0,
        "controls": {
            "init_image": controls.init_image,
            "mask": controls.mask,
            "control": controls.control,
            "loras": controls.loras,
        },
    })
}

/// The listing body over the pipelines `cached` says are on disk. Only those
/// are listed, so every listed id renders without a fetch.
fn models_body(cached: impl Fn(Model) -> bool, server_steps: Option<usize>) -> serde_json::Value {
    let data: Vec<_> = Pipeline::ALL
        .into_iter()
        .filter(|pipeline| cached(pipeline.entry()))
        .map(|pipeline| model_listing(pipeline, server_steps))
        .collect();
    json!({"object": "list", "data": data})
}

/// `GET /v1/images/models`: the image pipelines this server can render with
/// right now, each with the defaults and limits a client needs to build a
/// request. `/v1/models` lists the language models and none of these, the chat
/// routes refusing every one of them.
pub(crate) async fn models(State(state): State<AppState>) -> Response {
    let server_steps = state.settings.image_steps;
    let body = tokio::task::spawn_blocking(move || {
        models_body(
            |entry| crate::hub::cached_model(entry).is_some(),
            server_steps,
        )
    })
    .await;
    match body {
        Ok(body) => (
            [(axum::http::header::CACHE_CONTROL, "no-store")],
            axum::Json(body),
        )
            .into_response(),
        Err(e) => server_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("listing the image models failed: {e}"),
        )
        .into_response(),
    }
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
    let pipeline = params.pipeline;
    let peak = pipeline.peak(width, height, inputs.control.is_some(), inputs.loras.len());
    if let Err(e) = peak {
        return bad_param("size", e.to_string()).into_response();
    }

    let (reply, answer) = tokio::sync::oneshot::channel();
    match state.images.try_send(ImageJob::Render {
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
        Ok(Err(ImageError::Unavailable(message))) => {
            return server_error(StatusCode::SERVICE_UNAVAILABLE, message)
                .with_header("retry-after", "5")
                .into_response();
        }
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

    axum::Json(envelope(pipeline, (width, height), steps, &images)).into_response()
}

/// The response body: the OpenAI images shape, plus what this server adds to
/// it. `model` is the pipeline that rendered, which on the proxy path is not
/// necessarily the string the request carried.
fn envelope(
    pipeline: Pipeline,
    (width, height): (usize, usize),
    steps: usize,
    images: &[RenderedImage],
) -> serde_json::Value {
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
    json!({
        "created": super::unix_now(),
        "model": pipeline.entry().full_name(),
        "size": format!("{width}x{height}"),
        "steps": steps,
        "output_format": "png",
        "data": data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn test_image_job() -> (
        ImageJob,
        tokio::sync::oneshot::Receiver<Result<Vec<RenderedImage>, ImageError>>,
    ) {
        let (reply, answer) = tokio::sync::oneshot::channel();
        let params = validate(parse(r#"{"prompt":"test"}"#), false, None).unwrap();
        (
            ImageJob::Render {
                params,
                inputs: ImageInputs::default(),
                reply,
            },
            answer,
        )
    }

    #[test]
    fn rejected_image_submissions_close_their_queue_entries() {
        for disconnected in [false, true] {
            let (logger, events) = super::super::log::collecting();
            let (sender, receiver) = crossbeam_channel::bounded(0);
            if disconnected {
                drop(receiver);
            }
            let handle = Handle {
                sender,
                logger,
                resident: Arc::new(AtomicBool::new(false)),
                resident_model: Arc::default(),
            };
            let (job, _answer) = test_image_job();
            assert!(handle.try_send(job).is_err());
            let events = events.drain();
            assert_eq!(events.len(), 2);
            let ServeLog::ImageQueued(activity) = &events[0] else {
                panic!("queue event missing")
            };
            let ServeLog::ImageDone(record) = &events[1] else {
                panic!("terminal event missing")
            };
            assert_eq!(record.activity.id, activity.id);
            assert!(!record.cancelled);
            assert_eq!(record.completed_images, 0);
            assert!(record.error.as_ref().unwrap().contains(if disconnected {
                "not running"
            } else {
                "full"
            }));
        }
    }

    #[test]
    fn dropping_the_image_queue_finishes_every_accepted_job() {
        let (logger, events) = super::super::log::collecting();
        let (sender, receiver) = crossbeam_channel::bounded(2);
        let handle = Handle {
            sender,
            logger,
            resident: Arc::new(AtomicBool::new(false)),
            resident_model: Arc::default(),
        };
        let (first, _first_answer) = test_image_job();
        let (second, _second_answer) = test_image_job();
        handle.try_send(first).unwrap();
        handle.try_send(second).unwrap();
        drop(handle);
        drop(receiver);
        let events = events.drain();
        let queued: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                ServeLog::ImageQueued(a) => Some(a.id),
                _ => None,
            })
            .collect();
        let done: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                ServeLog::ImageDone(r) => {
                    assert!(r.cancelled);
                    Some(r.activity.id)
                }
                _ => None,
            })
            .collect();
        assert_eq!(queued, done);
        assert_eq!(done.len(), 2);
    }

    #[test]
    fn image_failure_preserves_completed_work_and_phase_measurements() {
        let (logger, events) = super::super::log::collecting();
        let (job, _answer) = test_image_job();
        let mut trace = ImageTrace::new(&job, logger);
        trace.picked();
        trace.record.encode_secs = 0.5;
        trace.record.denoise_secs = 12.0;
        trace.record.vae_secs = 2.0;
        trace.record.executed_steps = 6;
        trace.completed_image();
        trace.failed(&ImageError::Render("second image failed".into()), false);
        drop(trace);
        let events = events.drain();
        assert!(matches!(
            &events[..],
            [
                ServeLog::ImageQueued(_),
                ServeLog::ImagePicked { .. },
                ServeLog::ImageProgress {
                    completed_images: 1,
                    ..
                },
                ServeLog::ImageDone(_)
            ]
        ));
        let ServeLog::ImageDone(record) = &events[3] else {
            unreachable!()
        };
        assert_eq!(record.completed_images, 1);
        assert_eq!(record.executed_steps, 6);
        assert_eq!(
            record.encode_secs + record.denoise_secs + record.vae_secs,
            14.5
        );
        assert_eq!(record.error.as_deref(), Some("second image failed"));
        assert!(!record.cancelled);
    }

    #[test]
    fn warm_renders_reserve_their_temporary_allocations_again() {
        let gib = 1024 * 1024 * 1024;
        let peak = crate::memory::image_peak(1024, 1024, false, 0).unwrap();
        assert_eq!(image_allocation_reserve(peak, None), 40 * gib);
        assert_eq!(
            image_allocation_reserve(peak, Some(Pipeline::ZImage)),
            24 * gib
        );
        // Qwen-Image 2.1 keeps its pipeline alone, so a warm request is
        // credited less than the 15.7 GB that stays and reserves its encoder,
        // loaded for the request, in full.
        let peak = Pipeline::QwenImage.peak(512, 512, false, 0).unwrap();
        assert_eq!(peak, crate::memory::QWEN_IMAGE_SERVE_ENCODE_PEAK);
        assert_eq!(
            image_allocation_reserve(peak, Some(Pipeline::QwenImage)),
            peak - 14 * gib
        );
        assert!(Pipeline::QwenImage.resident_floor() < 15_700_000_000);
        // The envelope is the taller of the two phases at every size. Which
        // one that is depends on the VAE arm: the shipped decoder keeps the
        // render under the encode phase up to the pixel cap.
        let large = Pipeline::QwenImage.peak(1024, 1024, false, 0).unwrap();
        assert_eq!(
            large,
            crate::memory::qwen_image_serve_peak(1024, 1024).unwrap()
        );
        assert!(large >= crate::memory::qwen_image_peak(1024, 1024).unwrap());
        assert!(large >= crate::memory::QWEN_IMAGE_SERVE_ENCODE_PEAK);
    }

    fn fingerprint() -> crate::zimage::lora::Fingerprint {
        crate::zimage::lora::Fingerprint {
            bytes: 1,
            modified_ns: 0,
            inode: 0,
            changed_s: 0,
            changed_ns: 0,
        }
    }

    fn held(pipeline: Pipeline) -> Held<'static> {
        Held {
            pipeline,
            loras: &[],
            control: None,
        }
    }

    #[test]
    fn a_request_for_the_other_pipeline_unloads_the_resident_one() {
        for resident in Pipeline::ALL {
            for wanted in Pipeline::ALL {
                assert_eq!(
                    held(resident).serves(&held(wanted)),
                    resident == wanted,
                    "{resident:?} resident, {wanted:?} wanted"
                );
            }
        }
        // The same pipeline under another adapter set or another ControlNet
        // is a reload too, which is the rule the swap extends.
        let lora = crate::zimage::lora::ResolvedLora {
            path: "/adapters/a.safetensors".into(),
            weight: 0.8,
            fingerprint: fingerprint(),
        };
        let with_lora = [lora];
        let adapted = Held {
            loras: &with_lora,
            ..held(Pipeline::ZImage)
        };
        assert!(!held(Pipeline::ZImage).serves(&adapted));
        assert!(adapted.serves(&adapted));
        let checkpoint = ControlIdentity {
            path: "/control.safetensors".into(),
            fingerprint: fingerprint(),
        };
        let controlled = Held {
            control: Some(&checkpoint),
            ..held(Pipeline::ZImage)
        };
        assert!(!held(Pipeline::ZImage).serves(&controlled));
    }

    fn qwen_params(prompt: &str) -> ImageParams {
        let body = json!({"model": "Qwen-Image-2.1", "prompt": prompt, "size": "512x512"});
        validate(serde_json::from_value(body).unwrap(), false, None).unwrap()
    }

    fn rendered() -> Result<RenderedPrompt, ImageError> {
        Ok(RenderedPrompt {
            text: String::new(),
            ids: vec![1, 2, 3],
            drop: 1,
        })
    }

    #[test]
    fn a_request_that_cannot_run_evicts_nothing() {
        let params = qwen_params("x");
        let wanted = held(Pipeline::QwenImage);
        let zimage = held(Pipeline::ZImage);
        // A bad prompt is found before the swap is decided: with Z-Image
        // resident the plan is an error, and an error unloads nothing.
        let refused = plan(&params, &wanted, Some(&zimage), &|_| true, &|_| {
            Err(ImageError::Request("prompt: too long".into()))
        });
        assert!(matches!(refused, Err(ImageError::Request(_))));
        // So is a pipeline that is not on disk, and the prompt is not even
        // rendered for it.
        let missing = plan(&params, &wanted, Some(&zimage), &|_| false, &|_| {
            panic!("an uncached pipeline renders no prompt")
        });
        assert!(matches!(
            missing,
            Err(ImageError::Request(text)) if text.contains("xwen fetch --model qwen-image-2.1")
        ));
        // A good request for the other pipeline unloads first, and carries the
        // validated prompt on to the encoder.
        let swap = plan(&params, &wanted, Some(&zimage), &|_| true, &|_| rendered()).unwrap();
        assert!(swap.unload_first);
        assert_eq!(swap.prompt.unwrap().ids, [1, 2, 3]);
        // Nothing resident: nothing to unload.
        let cold = plan(&params, &wanted, None, &|_| true, &|_| rendered()).unwrap();
        assert!(!cold.unload_first);
    }

    #[test]
    fn a_resident_pipeline_keeps_serving_when_its_cache_entry_is_gone() {
        let params = qwen_params("x");
        let wanted = held(Pipeline::QwenImage);
        let warm = plan(&params, &wanted, Some(&wanted), &|_| false, &|_| rendered()).unwrap();
        assert!(!warm.unload_first);
        let zparams = validate(parse(r#"{"prompt":"x"}"#), false, None).unwrap();
        let zimage = held(Pipeline::ZImage);
        let warm = plan(&zparams, &zimage, Some(&zimage), &|_| false, &|_| {
            panic!("Z-Image renders its prompt inside the render")
        })
        .unwrap();
        assert!(!warm.unload_first && warm.prompt.is_none());
        // The same missing cache refuses a load.
        assert!(plan(&zparams, &zimage, None, &|_| false, &|_| rendered()).is_err());
    }

    #[test]
    fn an_overlong_prompt_is_the_clients_fault_and_a_broken_tokenizer_is_not() {
        let encoder = Model::QwenImage21.text_encoder().unwrap();
        // No tokenizer at this path: the server's fault, whatever the prompt.
        let missing = check_qwen_image_prompt(
            encoder,
            Path::new("/nonexistent/tokenizer.json"),
            &qwen_params("x"),
        );
        assert!(matches!(missing, Err(ImageError::Render(_))), "{missing:?}");

        let Some(tokenizer) = crate::test_support::repo_file_or_skip(
            Model::QwenImage21.repo(),
            "processor/tokenizer.json",
            "xwen fetch --model qwen-image-2.1",
        ) else {
            return;
        };
        let fine = check_qwen_image_prompt(encoder, &tokenizer, &qwen_params("a red bicycle"));
        assert_eq!(fine.unwrap().drop, 14);
        let long = "lighthouse ".repeat(5000);
        let refused = check_qwen_image_prompt(encoder, &tokenizer, &qwen_params(&long));
        assert!(
            matches!(&refused, Err(ImageError::Request(text)) if text.contains("shorten it")),
            "{refused:?}"
        );
    }

    #[test]
    fn a_job_is_shown_under_the_pipeline_it_asked_for() {
        let (logger, events) = super::super::log::collecting();
        let (reply, _answer) = tokio::sync::oneshot::channel();
        let job = ImageJob::Render {
            params: qwen_params("x"),
            inputs: ImageInputs::default(),
            reply,
        };
        drop(QueuedImage::new(job, logger));
        let queued = events.drain().into_iter().find_map(|event| match event {
            ServeLog::ImageQueued(activity) => Some(activity.model),
            _ => None,
        });
        assert_eq!(queued.as_deref(), Some("Qwen-Image-2.1"));
    }

    #[test]
    fn health_names_the_resident_pipeline_and_forgets_it_on_unload() {
        let (logger, events) = super::super::log::collecting();
        let residency = Residency {
            flag: Arc::new(AtomicBool::new(false)),
            model: Arc::default(),
        };
        let handle = Handle {
            sender: crossbeam_channel::bounded(0).0,
            logger: logger.clone(),
            resident: Arc::clone(&residency.flag),
            resident_model: Arc::clone(&residency.model),
        };
        assert_eq!(handle.resident_model(), None);
        residency.loaded(Pipeline::QwenImage, &logger);
        assert!(handle.is_loaded());
        assert_eq!(handle.resident_model(), Some("Qwen-Image-2.1"));
        residency.cleared(&logger);
        residency.cleared(&logger);
        assert!(!handle.is_loaded());
        assert_eq!(handle.resident_model(), None);
        let flips: Vec<_> = events
            .drain()
            .into_iter()
            .filter_map(|event| match event {
                ServeLog::ImageResidency { loaded } => Some(loaded),
                _ => None,
            })
            .collect();
        assert_eq!(flips, [true, false], "a second clear says nothing");
    }
    #[test]
    fn image_worker_skips_disconnected_jobs_and_refuses_oversized_work_before_loading() {
        let (send, receive) = crossbeam_channel::bounded(2);
        let (logger, events) = super::super::log::collecting();
        let resident = Arc::new(AtomicBool::new(false));
        let (closed_reply, closed_answer) = tokio::sync::oneshot::channel();
        drop(closed_answer);
        let params = validate(parse(r#"{"prompt":"test"}"#), false, None).unwrap();
        let closed = ImageJob::Render {
            params: params.clone(),
            inputs: ImageInputs::default(),
            reply: closed_reply,
        };
        assert!(closed.is_closed());
        send.send(QueuedImage::new(closed, logger.clone())).unwrap();
        let (reply, answer) = tokio::sync::oneshot::channel();
        send.send(QueuedImage::new(
            ImageJob::Render {
                params: ImageParams {
                    width: 8192,
                    height: 8192,
                    ..params
                },
                inputs: ImageInputs::default(),
                reply,
            },
            logger.clone(),
        ))
        .unwrap();
        drop(send);
        engine_loop(
            receive,
            None,
            Duration::from_secs(1),
            Residency {
                flag: Arc::clone(&resident),
                model: Arc::default(),
            },
            Arc::new(super::super::types::Cancel::default()),
            logger,
        );
        let records: Vec<_> = events
            .drain()
            .into_iter()
            .filter_map(|event| match event {
                ServeLog::ImageDone(record) => Some(record),
                _ => None,
            })
            .collect();
        assert_eq!(records.len(), 2);
        assert!(records[0].cancelled);
        assert!(!records[1].cancelled);
        assert!(records[1].error.as_ref().unwrap().contains("pixels"));
        assert!(
            matches!(answer.blocking_recv().unwrap(), Err(ImageError::Request(message)) if message.contains("pixels"))
        );
        assert!(!resident.load(Ordering::Relaxed));
    }

    #[test]
    fn architecture_valid_images_above_the_memory_envelope_are_bad_requests() {
        let err = validate(
            parse(r#"{"prompt":"test","size":"1536x1024"}"#),
            false,
            None,
        )
        .unwrap_err();
        assert_eq!(param(&err).as_deref(), Some("size"));
        assert!(message(&err).contains("pixels"));
    }

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
            "/v1/images/models",
            "/images/models",
            "/proxy/openai/images/models",
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
                pipeline: Pipeline::ZImage,
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
    fn a_full_name_selects_its_pipeline_on_every_path() {
        for proxy in [false, true] {
            let params = validate(
                parse(r#"{"model":"Qwen-Image-2.1","prompt":"x"}"#),
                proxy,
                None,
            )
            .unwrap();
            assert_eq!(params.pipeline, Pipeline::QwenImage);
            assert_eq!(params.model_note, None);
            let params = validate(parse(r#"{"prompt":"x"}"#), proxy, None).unwrap();
            assert_eq!(params.pipeline, Pipeline::ZImage, "nothing is the default");
        }
        // The proxy path takes any string, and a string that is no pipeline's
        // name is served by the default one and noted for the log.
        let params =
            validate(parse(r#"{"model":"gpt-image-1","prompt":"x"}"#), true, None).unwrap();
        assert_eq!(params.pipeline, Pipeline::ZImage);
        assert_eq!(params.model_note.as_deref(), Some("gpt-image-1"));
        // Aliases and near misses are refused on the canonical paths, the
        // refusal naming every pipeline that is served.
        for name in [
            "qwen-image-2.1",
            "qwen-image-2.1-encoder",
            "Qwen-Image-2.1-Encoder",
        ] {
            let err = validate(
                parse(&format!(r#"{{"model":"{name}","prompt":"x"}}"#)),
                false,
                None,
            )
            .unwrap_err();
            assert_eq!(param(&err).as_deref(), Some("model"));
            let text = message(&err);
            assert!(
                text.contains("Z-Image-Turbo") && text.contains("Qwen-Image-2.1"),
                "{text}"
            );
        }
        // Every pipeline is an image-capable registry entry, and every such
        // entry is a pipeline here: a new one cannot be registered and missed.
        let registered: Vec<Model> = crate::hub::MODELS
            .into_iter()
            .filter(|model| model.text_encoder().is_some())
            .collect();
        assert_eq!(registered, Pipeline::ALL.map(Pipeline::entry));
    }

    #[test]
    fn each_pipeline_is_held_to_its_own_defaults_and_size_rule() {
        let qwen = |body: &str, server_steps| {
            validate(
                parse(&format!(r#"{{"model":"Qwen-Image-2.1",{body}}}"#)),
                false,
                server_steps,
            )
        };
        let params = qwen(r#""prompt":"x""#, None).unwrap();
        assert_eq!((params.width, params.height), (1024, 1024));
        assert_eq!(params.steps, 40);
        // The server-wide default is Z-Image's eight-step figure and does not
        // reach the forty-step model; a request's own count does.
        assert_eq!(qwen(r#""prompt":"x""#, Some(8)).unwrap().steps, 40);
        assert_eq!(
            qwen(r#""prompt":"x","steps":25"#, Some(8)).unwrap().steps,
            25
        );
        assert_eq!(
            validate(parse(r#"{"prompt":"x"}"#), false, Some(6))
                .unwrap()
                .steps,
            6
        );
        // 48x512 is a Z-Image size (3 x 32 = 96 tokens) and 48 is no multiple
        // of 32, so the same size is one model's and not the other's.
        let err = qwen(r#""prompt":"x","size":"48x512""#, None).unwrap_err();
        assert_eq!(param(&err).as_deref(), Some("size"));
        assert!(message(&err).contains("32"), "{}", message(&err));
        assert!(validate(parse(r#"{"prompt":"x","size":"48x512"}"#), false, None).is_ok());
        // The pixel cap is the same number and says it is about measurement.
        let err = qwen(r#""prompt":"x","size":"2048x2048""#, None).unwrap_err();
        assert_eq!(param(&err).as_deref(), Some("size"));
        assert!(
            message(&err).contains("not been measured"),
            "{}",
            message(&err)
        );
    }

    #[test]
    fn guidance_is_refused_on_both_pipelines_without_claiming_distillation_for_the_second() {
        for field in [r#""negative_prompt":"blurry""#, r#""guidance_scale":4.0"#] {
            let err = validate(
                parse(&format!(
                    r#"{{"model":"Qwen-Image-2.1","prompt":"x",{field}}}"#
                )),
                false,
                None,
            )
            .unwrap_err();
            let text = message(&err);
            assert!(text.contains("Qwen-Image-2.1"), "{text}");
            assert!(text.contains("classifier-free guidance"), "{text}");
            assert!(!text.contains("distilled"), "{text}");
            let err =
                validate(parse(&format!(r#"{{"prompt":"x",{field}}}"#)), false, None).unwrap_err();
            assert!(message(&err).contains("distilled"), "{}", message(&err));
        }
    }

    #[test]
    fn inputs_a_pipeline_has_no_path_for_are_refused_by_field_name() {
        assert!(refuse_missing_controls(Pipeline::ZImage, &["init_image", "loras"]).is_ok());
        assert!(refuse_missing_controls(Pipeline::QwenImage, &[]).is_ok());
        for field in [
            "init_image",
            "strength",
            "mask",
            "mask_blur",
            "control",
            "loras",
            "image",
        ] {
            let err = refuse_missing_controls(Pipeline::QwenImage, &[field]).unwrap_err();
            assert_eq!(param(&err).as_deref(), Some(field));
            let text = message(&err);
            assert!(
                text.contains("Qwen-Image-2.1")
                    && text.contains(field)
                    && text.contains("Z-Image-Turbo"),
                "{text}"
            );
        }
        let err = refuse_missing_controls(Pipeline::QwenImage, &["mask", "loras"]).unwrap_err();
        assert!(
            message(&err).contains("mask, loras are"),
            "{}",
            message(&err)
        );
    }

    #[test]
    fn the_listing_holds_the_cached_pipelines_and_what_a_client_needs_of_each() {
        assert_eq!(
            models_body(|_| false, None),
            json!({"object": "list", "data": []})
        );
        let only_qwen = models_body(|entry| entry == Model::QwenImage21, Some(6));
        assert_eq!(
            only_qwen,
            json!({"object": "list", "data": [{
                "id": "Qwen-Image-2.1",
                "object": "model",
                "default": false,
                "default_steps": 40,
                "max_steps": 50,
                "default_size": "1024x1024",
                "size_multiple": 32,
                "size_rule": "both sides multiples of 32",
                "max_pixels": 1_048_576,
                "max_references": 0,
                "controls": {"init_image": false, "mask": false, "control": false, "loras": false},
            }]})
        );
        // The operator's step default is what a Z-Image request without one
        // renders at, so it is what the listing says; Qwen-Image 2.1 keeps 40.
        let tuned = models_body(|_| true, Some(6));
        assert_eq!(tuned["data"][0]["default_steps"], 6);
        assert_eq!(tuned["data"][1]["default_steps"], 40);
        let both = models_body(|_| true, None);
        let ids: Vec<_> = both["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["Z-Image-Turbo", "Qwen-Image-2.1"]);
        let zimage = &both["data"][0];
        assert_eq!(zimage["default"], true);
        assert_eq!(zimage["default_steps"], 8);
        assert_eq!(zimage["size_multiple"], 16);
        assert_eq!(
            zimage["controls"],
            json!({"init_image": true, "mask": true, "control": true, "loras": true})
        );
        // Every listed default is one the route accepts for that model.
        for entry in both["data"].as_array().unwrap() {
            let body = json!({
                "model": entry["id"], "prompt": "x", "size": entry["default_size"],
                "steps": entry["default_steps"],
            });
            validate(serde_json::from_value(body).unwrap(), false, None).unwrap();
        }
    }

    #[test]
    fn the_envelope_names_the_pipeline_that_rendered() {
        let image = RenderedImage {
            png: vec![1, 2, 3],
            seed: 7,
            start_step: 0,
            control_map: None,
        };
        for pipeline in Pipeline::ALL {
            let body = envelope(pipeline, (512, 768), 12, std::slice::from_ref(&image));
            assert_eq!(body["model"], pipeline.entry().full_name());
            assert_eq!(body["size"], "512x768");
            assert_eq!(body["steps"], 12);
            assert_eq!(body["output_format"], "png");
            assert_eq!(body["data"][0]["b64_json"], "AQID");
            assert_eq!(body["data"][0]["seed"], 7);
        }
    }

    #[test]
    fn an_uncached_pipeline_is_a_request_fault_naming_the_fetch() {
        for pipeline in Pipeline::ALL {
            let ImageError::Request(text) = uncached(pipeline) else {
                panic!("an uncached pipeline is the operator's to fetch: a 400, not a 500");
            };
            assert!(text.contains(pipeline.entry().full_name()), "{text}");
            assert!(
                text.contains(&format!("xwen fetch --model {}", pipeline.entry())),
                "{text}"
            );
        }
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
        Ok(Err(ImageError::Unavailable(message))) => {
            server_error(StatusCode::SERVICE_UNAVAILABLE, message)
                .with_header("retry-after", "5")
                .into_response()
        }
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
