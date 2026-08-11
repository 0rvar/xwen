//! `xwen serve` — local HTTP server exposing the Anthropic Messages API, the
//! OpenAI Chat Completions API and this engine's own native generate endpoint
//! ([`native`]) over one model process.
//!
//! Every API is a thin translation over the same job protocol ([`types`]): a
//! handler normalizes its request into a [`JobRequest`], hands it to the single
//! inference thread over the bounded, scheduled [`queue::JobQueue`], and renders the resulting
//! [`EngineEvent`] stream in its own dialect. Everything in this module is the
//! part neither dialect owns — state, routing, auth, and the two ways a handler
//! consumes an event stream (buffered into one response, or forwarded as SSE).

pub mod anthropic;
pub(crate) mod batch;
pub mod config;
pub mod disk_cache;
mod disk_tier;
pub mod engine;
pub mod log;
pub(crate) mod native;
pub mod openai;
pub mod queue;
pub mod tui;
pub mod types;

use std::collections::VecDeque;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result};
use axum::Router;
use axum::extract::{Request, State};
use axum::http::{Method, StatusCode, Uri};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::Stream;

use crate::chat::{self, ChatOptions, Message};
use crate::sampler::SamplerOptions;
use crate::tokenizer::LagunaTokenizer;
use config::ServeSettings;
use log::{ServeLog, ServeLogger};
use queue::{JobQueue, Queued, SchedulePolicy};
use types::{
    Cancel, CancelGuard, CancelReason, Dialect, EngineEvent, GenerationJob, RequestOrigin, StopKind,
};

/// Per-job event channel depth. The engine blocks once a slow client falls this
/// far behind, which throttles generation to the reader rather than buffering
/// an unbounded transcript.
const EVENT_CHANNEL_CAPACITY: usize = 128;

/// Comment-line heartbeat, well under the idle timeouts proxies and SDK clients
/// apply to a stream that has produced nothing yet (a long prefill looks
/// exactly like that).
const KEEP_ALIVE_INTERVAL: Duration = Duration::from_secs(15);

/// How long a graceful shutdown may wait for connections to close before the
/// process leaves on its own.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(30);

/// Everything a handler needs, cloned per request. `Clone + Send + Sync` is a
/// hard requirement of axum's `State`, which the queue's mutex satisfies.
#[derive(Clone)]
pub struct AppState {
    pub jobs: Arc<JobQueue>,
    pub tokenizer: Arc<LagunaTokenizer>,
    pub settings: Arc<ServeSettings>,
    pub model_loaded: Arc<AtomicBool>,
    /// The process-wide shutdown token. Cancelled by `shutdown_signal`, polled by
    /// the engine, so the running generation aborts and the process exits through
    /// its destructors instead of the shutdown watchdog.
    pub shutdown: Arc<Cancel>,
    /// Reported as the model id by `/v1/models`, and echoed by a request that
    /// names no model of its own.
    pub model_id: String,
    /// Which official checkpoint the default GGUF is, read from its
    /// architecture at startup. A request whose `model` names a known
    /// checkpoint runs on that one; everything else runs on this.
    pub default_size: crate::hub::Model,
    /// The resolved context length: what the config asks for, capped at what the
    /// checkpoint was converted with. Serves the handler-side "does the prompt
    /// fit" check; the engine re-derives its own copy at load time, which stays
    /// authoritative for engine-side arithmetic.
    pub max_ctx: usize,
    /// The next request id [`submit`] hands out. Ids exist to correlate the
    /// events one request produces — they are never sent to a client and never
    /// appear in a log line — so a plain counter is all they have to be.
    pub next_request_id: Arc<AtomicU64>,
}

/// A request for a graceful shutdown from something that is not a signal.
///
/// The dashboard's `q` and Ctrl-C go through it — raw mode generates no SIGINT,
/// so a keypress has to reach [`shutdown_signal`] some other way — and by waking
/// that same future they take exactly the path a signal does, watchdog included.
/// A request that arrives before anything is waiting is held rather than lost.
#[derive(Clone, Default)]
pub struct QuitSignal(Arc<tokio::sync::Notify>);

impl QuitSignal {
    pub fn request(&self) {
        self.0.notify_one();
    }

    async fn requested(&self) {
        self.0.notified().await;
    }
}

/// Assemble the runtime, the router and the inference worker, and serve until
/// shutdown.
pub fn run(settings: ServeSettings) -> Result<()> {
    let model_id = model_id(&settings);
    let quit = QuitSignal::default();
    // The sink outlives everything that logs: it is started before the first
    // line the server can produce, and the handle's `Drop` stops and joins it on
    // every way out of this function, error paths included. Which sink is the
    // only thing `--tui` decides; every site logs the same events either way.
    let (logger, _sink) = if settings.tui {
        log::spawn_tui_sink(tui::Vitals::new(model_id.clone(), &settings), quit.clone())
    } else {
        log::spawn_stderr_sink()
    };
    log::set_global(logger.clone());

    // Before binding anything: a bad model path or an unreadable tokenizer is a
    // startup error, not a surprise on the first request. Which official
    // checkpoint the default GGUF is comes from its architecture — the
    // authoritative answer, whatever flag the path travelled with.
    let (tokenizer, max_ctx, default_size) = engine::validate_model(&settings, &logger)?;

    let model_loaded = Arc::new(AtomicBool::new(false));
    let shutdown = Arc::new(Cancel::default());
    let jobs = Arc::new(JobQueue::new(
        settings.queue_capacity,
        SchedulePolicy {
            schedule: settings.schedule,
            queue_timeout: settings.queue_timeout,
            age_limit: settings.schedule_age_limit,
        },
        logger.clone(),
    ));
    let engine = engine::spawn_engine(
        settings.clone(),
        default_size,
        Arc::clone(&jobs),
        Arc::clone(&model_loaded),
        Arc::clone(&shutdown),
        logger.clone(),
    );

    let address = format!("{}:{}", settings.host, settings.port);
    let state = AppState {
        jobs: Arc::clone(&jobs),
        tokenizer,
        settings: Arc::new(settings),
        model_loaded,
        shutdown: Arc::clone(&shutdown),
        model_id,
        default_size,
        max_ctx,
        next_request_id: Arc::new(AtomicU64::new(1)),
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building the serve runtime")?;

    let served = runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(&address)
            .await
            .with_context(|| format!("binding {address}"))?;
        let bound = listener
            .local_addr()
            .unwrap_or_else(|_| ([0, 0, 0, 0], 0).into());
        logger.log(ServeLog::Listening {
            address: bound.to_string(),
            anthropic: state.settings.anthropic,
            openai: state.settings.openai,
        });
        logger.log(ServeLog::ServingModel {
            path: state.settings.model.clone(),
        });

        axum::serve(listener, router(state))
            .with_graceful_shutdown(shutdown_signal(
                Arc::clone(&shutdown),
                Arc::clone(&jobs),
                logger.clone(),
                quit,
            ))
            .await
            .context("serving")
    });
    // Whatever ended the serve — the graceful shutdown or an error — the engine
    // comes down before any error propagates: cancelling the shutdown token
    // aborts a running generation within one poll, closing the queue unblocks a
    // `take`, and the join runs the engine's destructors (the model's mmap
    // unregisters from the residency set there) on the error path too.
    shutdown.cancel(CancelReason::Shutdown);
    jobs.close();
    logger.log(ServeLog::ShuttingDown);
    let engine_panicked = engine.join().is_err();
    served?;
    if engine_panicked {
        anyhow::bail!("the inference thread panicked");
    }
    Ok(())
}

/// The GGUF's basename without its extension, e.g. `laguna-s-2.1-Q4_K_M`.
fn model_id(settings: &ServeSettings) -> String {
    settings
        .model
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "xwen".to_string())
}

/// A disabled API's routes are simply never registered, so it 404s like any
/// unknown path — with a JSON body saying which, from the fallback below.
/// `/health` sits outside the auth layer: a readiness probe that needs a
/// credential is a readiness probe nobody wires up.
///
/// Both fallbacks are installed after `route_layer`, which is what keeps them
/// out from behind the auth middleware: a client that guessed the path wrong
/// should be told so, not asked for a key it would not have needed.
fn router(state: AppState) -> Router {
    let mut api = Router::new();
    if state.settings.anthropic {
        api = api
            .route("/v1/messages", post(anthropic::messages))
            .route("/v1/messages/count_tokens", post(anthropic::count_tokens));
    }
    if state.settings.openai {
        api = api.route("/v1/chat/completions", post(openai::chat_completions));
    }
    // The native surface is not a compatibility dialect and has no opt-out: it
    // is the only way to reach the engine capabilities the other two cannot
    // spell.
    api = api
        .route("/xwen/v1/generate", post(native::generate))
        .route("/xwen/v1/batch", post(batch::batch));
    let api = api
        .route("/v1/models", get(models))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            require_api_key,
        ));

    api.route("/health", get(health))
        .method_not_allowed_fallback(method_not_allowed)
        .fallback(not_found)
        .with_state(state)
}

/// Register a terminating signal, or `None` when the process cannot.
///
/// A listener that fails to register costs this one way down, not the server:
/// Ctrl-C, the dashboard's `q` and an axum error all still reach the same
/// shutdown. Refusing to start over it would be the worse trade.
fn terminating_signal(
    kind: tokio::signal::unix::SignalKind,
    name: &'static str,
    logger: &ServeLogger,
) -> Option<tokio::signal::unix::Signal> {
    match tokio::signal::unix::signal(kind) {
        Ok(stream) => Some(stream),
        Err(e) => {
            logger.log(ServeLog::SignalUnavailable {
                signal: name,
                error: e.to_string(),
            });
            None
        }
    }
}

/// Wait for one of the registered signals, or forever for one that is not.
async fn signalled(stream: Option<&mut tokio::signal::unix::Signal>) {
    match stream {
        Some(stream) => {
            stream.recv().await;
        }
        None => std::future::pending().await,
    }
}

async fn shutdown_signal(
    shutdown: Arc<Cancel>,
    jobs: Arc<JobQueue>,
    logger: ServeLogger,
    quit: QuitSignal,
) {
    // Every way a `xwen serve` is asked to stop — a signal, a keypress in the
    // dashboard — is the same request, and there is deliberately only one path
    // down from here. SIGTERM and SIGHUP matter as much as Ctrl-C does: a
    // `pkill` or a service manager stopping the server must take the graceful
    // path, which is what tears the dashboard's screen down before the watchdog
    // thread starts and leaves the operator's terminal usable.
    let mut terminate = terminating_signal(
        tokio::signal::unix::SignalKind::terminate(),
        "SIGTERM",
        &logger,
    );
    let mut hangup =
        terminating_signal(tokio::signal::unix::SignalKind::hangup(), "SIGHUP", &logger);
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        () = quit.requested() => {}
        () = signalled(terminate.as_mut()) => {}
        () = signalled(hangup.as_mut()) => {}
    }
    logger.log(ServeLog::ShutdownRequested);
    // The engine polls this token, so the running generation aborts within one
    // poll interval and the process normally exits through its destructors —
    // the model's mmap unregisters from the residency set on the way out.
    shutdown.cancel(CancelReason::Shutdown);
    // Closing the queue unblocks an engine parked in `take`, so it drains the
    // accepted jobs — each dropped unstarted by its own shutdown check — and
    // exits promptly instead of waiting out the watchdog below.
    jobs.close();
    // Graceful shutdown still waits for every open connection, and a stream whose
    // client stopped reading can hold one open indefinitely. The watchdog bounds
    // that wait. It exits the process outright, which runs no destructors; as a
    // last resort that is fine, since the OS reclaims the Metal buffers and the
    // model's mmap either way.
    std::thread::spawn(move || {
        std::thread::sleep(SHUTDOWN_GRACE);
        logger.log(ServeLog::ShutdownGraceExpired {
            grace: SHUTDOWN_GRACE,
        });
        // The exit runs no destructors, so the line has to be on its way out
        // before the process is.
        logger.flush();
        std::process::exit(0);
    });
}

/// `POST /v1/messages` on a server started with `--no-anthropic` is the case
/// this exists for: without a body a client sees a bare 404 and reports a
/// network problem, when the answer is a one-line config change.
fn missing_route_message(settings: &ServeSettings, method: &Method, path: &str) -> String {
    if is_anthropic_path(path) && !settings.anthropic {
        return "the Anthropic Messages API is disabled on this server (anthropic = false)"
            .to_string();
    }
    if path.starts_with("/v1/chat/completions") && !settings.openai {
        return "the OpenAI Chat Completions API is disabled on this server (openai = false)"
            .to_string();
    }
    format!("no route for {method} {path}")
}

/// Which dialect a path answers in. The Anthropic routes own `/v1/messages`;
/// the native API owns `/xwen`; every other path — including one that
/// matched nothing — answers in the OpenAI shape, which is what the rest of
/// the surface uses.
pub(crate) fn is_anthropic_path(path: &str) -> bool {
    path.starts_with("/v1/messages")
}

pub(crate) fn is_native_path(path: &str) -> bool {
    path.starts_with("/xwen")
}

/// An error for a request that never reached a handler, in the dialect its path
/// belongs to.
pub(crate) fn routing_error(
    path: &str,
    status: StatusCode,
    anthropic_kind: &str,
    openai_code: &str,
    message: String,
) -> ApiError {
    if is_anthropic_path(path) {
        anthropic::error(status, anthropic_kind, message)
    } else if is_native_path(path) {
        native::error(status, anthropic_kind, message)
    } else {
        openai::error(status, "invalid_request_error", Some(openai_code), message)
    }
}

async fn not_found(State(state): State<AppState>, method: Method, uri: Uri) -> Response {
    routing_error(
        uri.path(),
        StatusCode::NOT_FOUND,
        "not_found_error",
        "unknown_url",
        missing_route_message(&state.settings, &method, uri.path()),
    )
    .into_response()
}

/// A known path reached with the wrong verb. Axum answers this one with a bare
/// 405 by default, which is a body an SDK cannot parse; the `Allow` header it
/// attaches to the response says which method the path does take.
async fn method_not_allowed(method: Method, uri: Uri) -> Response {
    routing_error(
        uri.path(),
        StatusCode::METHOD_NOT_ALLOWED,
        "invalid_request_error",
        "method_not_supported",
        format!(
            "{method} is not allowed on {}: see the Allow header for the method it takes",
            uri.path()
        ),
    )
    .into_response()
}

/// `GET /health` — cheap enough to poll, and truthful about the lazy load.
async fn health(State(state): State<AppState>) -> Response {
    axum::Json(json!({
        "status": "ok",
        "model_loaded": state.model_loaded.load(Ordering::Relaxed),
    }))
    .into_response()
}

/// `GET /v1/models` — the default checkpoint under the id requests echo, then
/// the two official checkpoints under the names a request's `model` field
/// selects them by. Each entry carries both APIs' field names so either SDK's
/// model list parses it.
async fn models(State(state): State<AppState>) -> Response {
    let created = model_mtime(&state.settings);
    let entry = |id: &str| {
        json!({
            "id": id,
            "object": "model",
            "created": created,
            "owned_by": "xwen",
            "type": "model",
            "display_name": id,
            "created_at": rfc3339_utc(created),
        })
    };
    let mut ids = vec![state.model_id.clone()];
    ids.extend(
        [crate::hub::Model::Qwen27B, crate::hub::Model::Qwen35BA3B]
            .iter()
            .map(ToString::to_string),
    );
    let data: Vec<Value> = ids.iter().map(|id| entry(id)).collect();
    axum::Json(json!({
        "object": "list",
        "data": data,
        "has_more": false,
        "first_id": ids.first(),
        "last_id": ids.last(),
    }))
    .into_response()
}

/// The checkpoint's mtime, which is the closest thing a local file has to a
/// release date. Falls back to now for a model on a filesystem without one.
fn model_mtime(settings: &ServeSettings) -> u64 {
    std::fs::metadata(&settings.model)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|since| since.as_secs())
        .unwrap_or_else(unix_now)
}

pub(crate) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// UTC RFC 3339, via Howard Hinnant's civil-from-days. A date formatter is all
/// `/v1/models` needs from a calendar, which does not justify a date crate.
fn rfc3339_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let time = secs % 86_400;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    let year = if month <= 2 { year + 1 } else { year };

    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        time / 3600,
        (time % 3600) / 60,
        time % 60
    )
}

/// Rejects a request whose credentials do not match `settings.api_key`, in the
/// dialect of the API it was aimed at. With no key configured every request
/// passes, credentials or not — which is why the default bind is loopback.
async fn require_api_key(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let expected = state.settings.api_key.as_deref();
    let headers = request.headers();
    let presented = presented_key(
        headers.get("x-api-key").and_then(|v| v.to_str().ok()),
        headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok()),
    );
    if authorized(expected, presented) {
        return next.run(request).await;
    }

    let message = "invalid or missing API key: send it as x-api-key or Authorization: Bearer";
    let path = request.uri().path();
    if is_anthropic_path(path) {
        anthropic::error(StatusCode::UNAUTHORIZED, "authentication_error", message).into_response()
    } else if is_native_path(path) {
        native::error(StatusCode::UNAUTHORIZED, "authentication_error", message).into_response()
    } else {
        openai::error(
            StatusCode::UNAUTHORIZED,
            "invalid_request_error",
            Some("invalid_api_key"),
            message,
        )
        .into_response()
    }
}

/// The key a request presents. `x-api-key` (Anthropic's spelling) wins over
/// `Authorization: Bearer` (OpenAI's) when both are set; an Authorization
/// header with any other scheme presents nothing.
pub(crate) fn presented_key<'a>(
    x_api_key: Option<&'a str>,
    authorization: Option<&'a str>,
) -> Option<&'a str> {
    if let Some(key) = x_api_key {
        return Some(key);
    }
    let (scheme, value) = authorization?.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| value.trim())
}

/// No configured key accepts everything, including a request with no
/// credentials at all.
pub(crate) fn authorized(expected: Option<&str>, presented: Option<&str>) -> bool {
    match (expected, presented) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some(expected), Some(presented)) => secrets_match(expected, presented),
    }
}

/// Compare two keys without an early exit, so how long the answer takes says
/// nothing about how many leading bytes were right. The lengths are folded in
/// the same way, and the loop runs over the longer of the two: a caller cannot
/// learn the expected key's length from the timing either.
fn secrets_match(expected: &str, presented: &str) -> bool {
    let (expected, presented) = (expected.as_bytes(), presented.as_bytes());
    let mut difference = expected.len() ^ presented.len();
    for index in 0..expected.len().max(presented.len()) {
        let a = expected.get(index).copied().unwrap_or(0);
        let b = presented.get(index).copied().unwrap_or(0);
        difference |= usize::from(a ^ b);
    }
    difference == 0
}

/// An error response in one API's envelope, built by that API's module.
#[derive(Debug)]
pub(crate) struct ApiError {
    pub status: StatusCode,
    pub body: Value,
    /// Extra response headers. `Retry-After` on the queue-full answer is the
    /// one user; almost every error carries none, which is why the builders
    /// default it empty and [`ApiError::with_header`] adds to it.
    pub headers: Vec<(&'static str, String)>,
}

impl ApiError {
    pub fn with_header(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.headers.push((name, value.into()));
        self
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut response = (self.status, axum::Json(self.body)).into_response();
        for (name, value) in self.headers {
            match axum::http::HeaderValue::from_str(&value) {
                Ok(value) => {
                    response.headers_mut().insert(name, value);
                }
                // The names and values are compile-time strings from the
                // builders; an unencodable one is a bug worth hearing about,
                // not worth failing the response over.
                // `into_response` is a trait method with no access to the
                // server's state, so this one site reports through the
                // process-wide logger instead of a threaded handle.
                Err(_) => log::log_global(ServeLog::InvalidHeaderDropped { name, value }),
            }
        }
        response
    }
}

/// A normalized request. The messages are still unrendered here; [`submit`]
/// renders and encodes them, so the engine only ever deals in token ids.
pub(crate) struct JobRequest {
    pub messages: Vec<Message>,
    pub enable_thinking: bool,
    pub max_think: Option<usize>,
    pub max_tokens: usize,
    pub sampling: SamplerOptions,
    pub stop_sequences: Vec<String>,
    /// Tool definitions in the OpenAI object shape, or empty for a request that
    /// carries none (including one whose tools the server was told to drop).
    pub tools: Vec<Value>,
    /// Schema constraint compiled from `response_format` /
    /// `output_config.format`, or `None` for an unconstrained reply. Mutually
    /// exclusive with `tools` — the dialects reject the combination.
    pub grammar: Option<crate::constrain::Grammar>,
    /// The start of the turn to hand the model, written into the generation
    /// header. Only the native dialect can express one; the compat dialects
    /// always send `None`.
    pub continuation: Option<chat::Continuation>,
}

/// Why a job never reached the engine.
#[derive(Debug)]
pub(crate) enum SubmitError {
    /// The request cannot be served as written — a prompt that does not fit the
    /// context, an impossible parameter, a conversation the template cannot
    /// render. A 400 in either dialect.
    Invalid(String),
    /// The queue is full: too many requests are already waiting on the one
    /// model. Retryable, and both APIs say so with their "overloaded" status.
    Overloaded,
    /// The inference thread is gone, which no retry fixes.
    EngineGone,
}

/// A rendered conversation, encoded into the exact ids the engine prefills.
/// Built by [`encode_conversation`], the one render+encode implementation both
/// dialects' generation paths and `count_tokens` share — so a counted prompt is
/// byte-for-byte the prompt a generation would prefill.
pub(crate) struct EncodedPrompt {
    pub tokens: Vec<u32>,
    /// Length of the shared-context span; the generation header starts here.
    pub boundary: usize,
    /// Number of tokens up to and including the leading system block, when the
    /// conversation renders one. It is the deepest position a different
    /// conversation from the same client still agrees with, which is why the
    /// engine snapshots the KV cache there as well as at the boundary.
    pub anchor: Option<usize>,
    /// Whether the generation header leaves the model inside an open thinking
    /// span, so the first decoded token is reasoning rather than answer text.
    pub starts_in_thinking: bool,
    /// Token count of the response prefix a continuation put at the end of the
    /// header, or 0 when the request supplied none. Those are the last
    /// `prefix_len` ids of `tokens`.
    pub prefix_len: usize,
}

/// Render a conversation and encode it, split at the generation header.
///
/// The two halves tokenize independently: the header opens with the added token
/// `<assistant>`, which BPE never merges into its neighbours, so
/// `encode(context) ++ encode(header) == encode(context ++ header)` —
/// `chat::tests::prompt_parts_split_on_a_token_boundary` guards exactly this.
/// No special tokens are added; the template already wrote the BOS into the
/// context half.
///
/// The context is encoded with the client-content byte ranges, so an added-token
/// string inside a message body — a user quoting `<tool_call>` — encodes as
/// plain text instead of a control token the parser would act on. The header is
/// encoded the same way: it is template-emitted apart from a continuation's
/// injected reasoning and response prefix, and with no continuation the ranges
/// are empty, which encodes exactly as plain `encode` does.
///
/// The context is itself encoded in two spans, split at the end of the leading
/// system block, so the anchor position is counted rather than guessed. That
/// split is a token boundary as well (see [`chat::PromptParts::system_end`]);
/// `tests::the_system_block_ends_on_a_token_boundary` pins the whole three-span
/// encoding against a single-shot one. A rendered response prefix splits the
/// header in the same way, so its token count is counted rather than guessed;
/// the seam there is the added token `</think>`.
pub(crate) fn encode_conversation(
    tokenizer: &LagunaTokenizer,
    messages: &[Message],
    enable_thinking: bool,
    tools: Vec<Value>,
    continuation: Option<&chat::Continuation>,
) -> Result<EncodedPrompt> {
    let chat::PromptParts {
        context,
        header,
        content_ranges,
        header_content_ranges,
        header_prefix_start,
        starts_in_thinking,
        system_end,
    } = chat::build_prompt_parts_with_spans_continued(
        messages,
        &ChatOptions {
            enable_thinking,
            // The wire protocols have no knob for this yet, so replayed turns
            // drop their reasoning once a later query supersedes them — the
            // template's own default.
            preserve_thinking: false,
            tools,
        },
        continuation,
    )?;
    let (mut tokens, anchor) = match system_end {
        Some(split) => {
            let (system_ranges, body_ranges) = split_content_ranges(&content_ranges, split);
            let mut tokens = tokenizer.encode_prompt(&context[..split], &system_ranges)?;
            let anchor = tokens.len();
            tokens.extend(tokenizer.encode_prompt(&context[split..], &body_ranges)?);
            (tokens, Some(anchor))
        }
        None => (tokenizer.encode_prompt(&context, &content_ranges)?, None),
    };
    let boundary = tokens.len();
    let prefix_len = match header_prefix_start {
        Some(split) => {
            let (head_ranges, prefix_ranges) = split_content_ranges(&header_content_ranges, split);
            tokens.extend(tokenizer.encode_prompt(&header[..split], &head_ranges)?);
            let prefix = tokenizer.encode_prompt(&header[split..], &prefix_ranges)?;
            let prefix_len = prefix.len();
            tokens.extend(prefix);
            prefix_len
        }
        None => {
            tokens.extend(tokenizer.encode_prompt(&header, &header_content_ranges)?);
            0
        }
    };
    Ok(EncodedPrompt {
        tokens,
        boundary,
        anchor,
        starts_in_thinking,
        prefix_len,
    })
}

/// Divide client-content byte ranges at `at` into the ranges of the text before
/// it and the ranges of the text after, the latter rebased onto that half.
///
/// A range is clipped rather than assigned to one side, so a range spanning the
/// split would keep both of its halves marked as client content. None does: the
/// split falls between two template-emitted pushes and every range covers one
/// client string.
fn split_content_ranges(
    ranges: &[std::ops::Range<usize>],
    at: usize,
) -> (Vec<std::ops::Range<usize>>, Vec<std::ops::Range<usize>>) {
    let head = ranges
        .iter()
        .map(|r| r.start.min(at)..r.end.min(at))
        .filter(|r| !r.is_empty())
        .collect();
    let tail = ranges
        .iter()
        .map(|r| r.start.max(at) - at..r.end.max(at) - at)
        .filter(|r| !r.is_empty())
        .collect();
    (head, tail)
}

/// Render, encode and hand a job to the inference thread, returning the channel
/// its events will arrive on and the guard that cancels the job when the
/// response it feeds is dropped. Never blocks on the queue: a full one is
/// reported, not waited on.
///
/// Everything that can be judged from the request alone is judged here, before
/// the queue, so the engine only ever sees a prompt that fits and a rejected
/// request costs it nothing.
///
/// `dialect` and `streaming` are the caller's own two facts about the request,
/// and only a handler knows them; they travel with the job as its
/// [`RequestOrigin`] so the events one request produces can be told from
/// another's.
pub(crate) fn submit(
    state: &AppState,
    request: JobRequest,
    dialect: Dialect,
    streaming: bool,
    model: crate::hub::Model,
) -> std::result::Result<(mpsc::Receiver<EngineEvent>, CancelGuard), SubmitError> {
    if request.max_tokens == 0 {
        return Err(SubmitError::Invalid(
            "max_tokens must be at least 1".to_string(),
        ));
    }
    let prompt = encode_conversation(
        &state.tokenizer,
        &request.messages,
        request.enable_thinking,
        request.tools.clone(),
        request.continuation.as_ref(),
    )
    .map_err(|e| SubmitError::Invalid(format!("rendering the prompt failed: {e:#}")))?;
    if prompt.tokens.len() >= state.max_ctx {
        return Err(SubmitError::Invalid(format!(
            "the prompt is {} tokens, which leaves no room to reply inside the server's \
             {}-token context: shorten the conversation or raise context_length",
            prompt.tokens.len(),
            state.max_ctx
        )));
    }

    // Armed with the prompt's own thinking state: a header that opens `<think>`
    // keeps the grammar dormant until `</think>` commits. A response prefix the
    // header already wrote is fed in ahead of the first draw, so the mask
    // continues that document instead of starting a second one.
    let grammar = match request.grammar {
        None => None,
        Some(grammar) => {
            let mut state = grammar.into_state(prompt.starts_in_thinking);
            if prompt.prefix_len > 0 {
                debug_assert!(
                    !prompt.starts_in_thinking,
                    "a prefix is only renderable past the closing </think>"
                );
                let prefix = &prompt.tokens[prompt.tokens.len() - prompt.prefix_len..];
                state.consume_prefix(prefix).map_err(|e| {
                    SubmitError::Invalid(format!(
                        "response prefix rejected by the output schema: {e:#}"
                    ))
                })?;
            }
            Some(state)
        }
    };

    let (events, receiver) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    let cancel = Arc::new(Cancel::default());
    let prompt_tokens = prompt.tokens.len();
    let job = GenerationJob {
        origin: RequestOrigin {
            id: state.next_request_id.fetch_add(1, Ordering::Relaxed),
            dialect,
            streaming,
        },
        model,
        prompt: prompt.tokens,
        boundary: prompt.boundary,
        anchor: prompt.anchor,
        starts_in_thinking: prompt.starts_in_thinking,
        // A thinking budget governs an OPEN reasoning span. When the header
        // already closed one (a native continuation), an armed ThinkBudget
        // would wait forever for a decoded </think> and eventually force its
        // wrap-up into the answer — the budget must die with the span.
        max_think: request.max_think.filter(|_| prompt.starts_in_thinking),
        max_tokens: request.max_tokens,
        sampling: request.sampling,
        stop_sequences: request.stop_sequences,
        tools: request.tools,
        grammar,
        cancel: Arc::clone(&cancel),
        deadline: None,
        events,
    };
    state
        .jobs
        .push(Queued {
            job: types::Job::Generation(Box::new(job)),
            submitted: Instant::now(),
            prompt_tokens,
        })
        .map(|()| (receiver, CancelGuard::new(cancel)))
}

/// One generation, buffered — what a non-streaming request answers with.
pub(crate) struct Completion {
    pub input_tokens: usize,
    pub cached_tokens: usize,
    pub thinking: String,
    pub text: String,
    /// In the order the model wrote them.
    pub tool_calls: Vec<CompletedToolCall>,
    pub stop: StopKind,
    pub output_tokens: usize,
    pub thinking_tokens: usize,
}

/// A tool call with its argument deltas already joined. The engine guarantees
/// `arguments` is one complete JSON object, so a handler that has to parse it —
/// Anthropic's `input` field does — never has to cope with a fragment.
pub(crate) struct CompletedToolCall {
    pub name: String,
    pub arguments: String,
}

/// How a generation failed after the job was accepted.
pub(crate) enum EngineFailure {
    /// The engine reported this message. `request_fault` is the engine's own
    /// classification: true when the request could not have been served as
    /// written, false when the server failed to serve a valid one.
    Reported {
        message: String,
        request_fault: bool,
    },
    /// The engine dropped the channel without a terminal event.
    Hangup,
}

/// Drain a job's events into a single completion.
pub(crate) async fn collect_completion(
    events: &mut mpsc::Receiver<EngineEvent>,
) -> std::result::Result<Completion, EngineFailure> {
    let mut completion = Completion {
        input_tokens: 0,
        cached_tokens: 0,
        thinking: String::new(),
        text: String::new(),
        tool_calls: Vec::new(),
        stop: StopKind::EndTurn,
        output_tokens: 0,
        thinking_tokens: 0,
    };
    while let Some(event) = events.recv().await {
        match event {
            EngineEvent::Start {
                input_tokens,
                cached_tokens,
            } => {
                completion.input_tokens = input_tokens;
                completion.cached_tokens = cached_tokens;
            }
            EngineEvent::Thinking(delta) => completion.thinking.push_str(&delta),
            EngineEvent::Text(delta) => completion.text.push_str(&delta),
            EngineEvent::ToolCallStart { name } => completion.tool_calls.push(CompletedToolCall {
                name,
                arguments: String::new(),
            }),
            // Deltas belong to the call the engine most recently started; it never
            // interleaves calls, and never sends a delta before a start.
            EngineEvent::ToolCallDelta(delta) => {
                if let Some(call) = completion.tool_calls.last_mut() {
                    call.arguments.push_str(&delta);
                }
            }
            EngineEvent::ToolCallEnd => {}
            EngineEvent::Done {
                stop,
                output_tokens,
                thinking_tokens,
            } => {
                completion.stop = stop;
                completion.output_tokens = output_tokens;
                completion.thinking_tokens = thinking_tokens;
                return Ok(completion);
            }
            EngineEvent::Error {
                message,
                request_fault,
            } => {
                return Err(EngineFailure::Reported {
                    message,
                    request_fault,
                });
            }
            // A generation's channel never carries a batch document; one here is
            // a protocol bug, reported as the server fault it is.
            EngineEvent::BatchDone(_) => {
                return Err(EngineFailure::Reported {
                    message: "the engine answered a generation with a batch response".to_string(),
                    request_fault: false,
                });
            }
        }
    }
    Err(EngineFailure::Hangup)
}

/// One server-sent event, still in wire form. Keeping the name and the payload
/// as plain strings until the last moment is what makes an API's event sequence
/// assertable: axum's `Event` is write-only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SseFrame {
    /// The `event:` line, or `None` for an unnamed event (OpenAI's dialect).
    pub name: Option<&'static str>,
    pub data: String,
}

impl SseFrame {
    pub fn named(name: &'static str, data: Value) -> Self {
        Self {
            name: Some(name),
            data: data.to_string(),
        }
    }

    pub fn unnamed(data: Value) -> Self {
        Self {
            name: None,
            data: data.to_string(),
        }
    }

    /// Named events write `event:` before `data:`. Both orders are legal SSE,
    /// but the Messages API documents this one and parsers written against its
    /// transcripts dispatch on the name before they have read the payload.
    /// Axum's `Event` appends each field as it is set, so the call order here is
    /// the wire order.
    fn into_event(self) -> Event {
        match self.name {
            Some(name) => Event::default().event(name).data(self.data),
            None => Event::default().data(self.data),
        }
    }
}

/// Renders the engine's events in one API's dialect. Encoders are stateful
/// because both dialects decide what to emit from what came before — the
/// Anthropic one cannot even assign a block index until it knows whether the
/// turn produced any thinking text.
pub(crate) trait SseEncoder: Unpin + Send + 'static {
    /// Append the frames for one engine event. Returns true once the stream is
    /// complete, after which the engine's channel is not polled again.
    fn on_event(&mut self, event: EngineEvent, out: &mut VecDeque<SseFrame>) -> bool;
    /// Append the frames for the engine hanging up without a terminal event.
    fn on_hangup(&mut self, out: &mut VecDeque<SseFrame>);
}

/// The engine's event channel, rendered as SSE. The receiver and the cancel
/// guard both live inside the stream, so a client that disconnects drops the
/// response body and with it cancels the generation nobody is listening to.
struct EventStream<E> {
    events: mpsc::Receiver<EngineEvent>,
    /// Fires `ClientGone` when the response body is dropped — before the engine
    /// would have noticed the closed channel at its next send.
    _guard: CancelGuard,
    encoder: E,
    pending: VecDeque<SseFrame>,
    finished: bool,
}

impl<E: SseEncoder> Stream for EventStream<E> {
    type Item = std::result::Result<Event, Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(frame) = this.pending.pop_front() {
                return Poll::Ready(Some(Ok(frame.into_event())));
            }
            if this.finished {
                return Poll::Ready(None);
            }
            match this.events.poll_recv(cx) {
                Poll::Ready(Some(event)) => {
                    this.finished = this.encoder.on_event(event, &mut this.pending);
                }
                Poll::Ready(None) => {
                    this.encoder.on_hangup(&mut this.pending);
                    this.finished = true;
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Wrap a job's events in an SSE response, heartbeat included. The guard rides
/// along so the generation is cancelled the moment the body is dropped.
pub(crate) fn sse_response<E: SseEncoder>(
    events: mpsc::Receiver<EngineEvent>,
    guard: CancelGuard,
    encoder: E,
) -> Response {
    let stream = EventStream {
        events,
        _guard: guard,
        encoder,
        pending: VecDeque::new(),
        finished: false,
    };
    Sse::new(stream)
        .keep_alive(
            KeepAlive::new()
                .interval(KEEP_ALIVE_INTERVAL)
                .text("keep-alive"),
        )
        .into_response()
}

/// A random request id, in the `<prefix><suffix>` shape both APIs use.
pub(crate) fn random_id(prefix: &str) -> String {
    use rand::Rng;
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut rng = rand::rng();
    let mut id = String::with_capacity(prefix.len() + 24);
    id.push_str(prefix);
    for _ in 0..24 {
        id.push(ALPHABET[rng.random_range(0..ALPHABET.len())] as char);
    }
    id
}

#[cfg(test)]
pub(crate) mod testutil {
    use super::*;
    use std::path::PathBuf;

    /// Settings with the documented defaults, for the request-preparation
    /// tests: they exercise the "request said nothing, fall back to the server"
    /// half of every mapping. One deliberate exception: `draft` is pinned to
    /// `None` where production zero-flag resolves to `Some("official")` —
    /// these tests never load a model, so a drafter path would be dead weight,
    /// but it means nothing routed through this helper exercises
    /// draft-enabled defaults.
    pub(crate) fn settings() -> ServeSettings {
        ServeSettings {
            model: PathBuf::from("/models/laguna-s-2.1-Q4_K_M.gguf"),
            host: config::DEFAULT_HOST.to_string(),
            port: config::DEFAULT_PORT,
            context_length: config::DEFAULT_CONTEXT_LENGTH,
            idle_unload: Some(Duration::from_secs(300)),
            anthropic: true,
            openai: true,
            api_key: None,
            tools_mode: config::DEFAULT_TOOLS_MODE,
            queue_capacity: config::DEFAULT_QUEUE_CAPACITY,
            queue_timeout: Duration::from_secs(config::DEFAULT_QUEUE_TIMEOUT_SECS),
            request_prefill_rate: config::DEFAULT_REQUEST_PREFILL_RATE,
            request_decode_rate: config::DEFAULT_REQUEST_DECODE_RATE,
            request_slack: Duration::from_secs(config::DEFAULT_REQUEST_SLACK_SECS),
            schedule: config::DEFAULT_SCHEDULE,
            schedule_age_limit: Duration::from_secs(config::DEFAULT_SCHEDULE_AGE_LIMIT_SECS),
            tui: config::DEFAULT_TUI,
            thinking_force: config::DEFAULT_THINKING_FORCE,
            thinking_budget: None,
            temperature: config::DEFAULT_TEMPERATURE,
            top_k: config::DEFAULT_TOP_K,
            top_p: config::DEFAULT_TOP_P,
            cache_snapshots: config::DEFAULT_CACHE_SNAPSHOTS,
            cache_slots: config::DEFAULT_CACHE_SLOTS,
            // No disk tier for a handler test: nothing here reaches the engine, and
            // a test that touched the owner's real cache directory would be a bug.
            cache_dir: None,
            disk_cache: false,
            disk_max_gib: config::DEFAULT_DISK_MAX_GIB,
            disk_min_tokens: config::DEFAULT_DISK_MIN_TOKENS,
            draft: None,
            draft_max: config::DEFAULT_DRAFT_MAX,
            draft_p_min: None,
            draft_pause_margin: config::DEFAULT_DRAFT_PAUSE_MARGIN,
            draft_ctx: config::DEFAULT_DRAFT_CTX,
        }
    }

    /// A conversation as `role:text` lines. `chat::Message` has no `PartialEq`,
    /// and the rendered shape is what the assertions are actually about.
    pub(crate) fn shape(messages: &[Message]) -> Vec<String> {
        messages
            .iter()
            .map(|message| match message {
                Message::System(text) => format!("system:{text}"),
                Message::User(text) => format!("user:{text}"),
                Message::Assistant {
                    content, reasoning, ..
                } => match reasoning {
                    Some(reasoning) => format!("assistant:{content}|think:{reasoning}"),
                    None => format!("assistant:{content}"),
                },
                Message::ToolResponse(text) => format!("tool:{text}"),
            })
            .collect()
    }

    /// Run a scripted event list through an encoder, as the SSE stream would.
    pub(crate) fn encode_all(
        encoder: &mut impl SseEncoder,
        events: Vec<EngineEvent>,
    ) -> Vec<SseFrame> {
        let mut out = VecDeque::new();
        for event in events {
            if encoder.on_event(event, &mut out) {
                break;
            }
        }
        out.into_iter().collect()
    }

    /// The `event:` names of a frame sequence, which is the part an SDK
    /// dispatches on.
    pub(crate) fn names(frames: &[SseFrame]) -> Vec<&str> {
        frames
            .iter()
            .map(|frame| frame.name.unwrap_or("<unnamed>"))
            .collect()
    }

    /// A frame's payload, parsed back for field-level assertions.
    pub(crate) fn payload(frame: &SseFrame) -> Value {
        serde_json::from_str(&frame.data)
            .unwrap_or_else(|e| panic!("frame data is not JSON ({e}): {:?}", frame.data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_configured_key_accepts_anything() {
        assert!(authorized(None, None));
        assert!(authorized(None, Some("whatever")));
    }

    #[test]
    fn a_configured_key_is_required_and_must_match() {
        assert!(authorized(Some("secret"), Some("secret")));
        assert!(!authorized(Some("secret"), Some("Secret")));
        assert!(!authorized(Some("secret"), None));
    }

    #[test]
    fn either_sdks_header_spelling_presents_the_key() {
        assert_eq!(presented_key(Some("secret"), None), Some("secret"));
        assert_eq!(presented_key(None, Some("Bearer secret")), Some("secret"));
        // Case in the scheme is not the client's problem.
        assert_eq!(presented_key(None, Some("bearer secret")), Some("secret"));
        // x-api-key wins when both are set.
        assert_eq!(
            presented_key(Some("from-header"), Some("Bearer other")),
            Some("from-header")
        );
        // Any other scheme presents nothing.
        assert_eq!(presented_key(None, Some("Basic secret")), None);
        assert_eq!(presented_key(None, Some("Bearer")), None);
        assert_eq!(presented_key(None, None), None);
    }

    #[test]
    fn model_id_is_the_gguf_basename() {
        let mut settings = testutil::settings();
        assert_eq!(model_id(&settings), "laguna-s-2.1-Q4_K_M");
        settings.model = std::path::PathBuf::from("model.gguf");
        assert_eq!(model_id(&settings), "model");
    }

    #[test]
    fn timestamps_render_as_utc_rfc3339() {
        assert_eq!(rfc3339_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_utc(1_000_000_000), "2001-09-09T01:46:40Z");
        // A leap day, which the month-shifted civil algorithm is the part that
        // gets wrong when it is wrong.
        assert_eq!(rfc3339_utc(1_709_164_800), "2024-02-29T00:00:00Z");
    }

    #[test]
    fn an_unknown_path_answers_in_the_dialect_it_was_aimed_at() {
        let anthropic = routing_error(
            "/v1/messages",
            StatusCode::NOT_FOUND,
            "not_found_error",
            "unknown_url",
            "nope".to_string(),
        );
        assert_eq!(anthropic.status, StatusCode::NOT_FOUND);
        assert_eq!(anthropic.body["type"], "error");
        assert_eq!(anthropic.body["error"]["type"], "not_found_error");
        assert_eq!(anthropic.body["error"]["message"], "nope");

        let openai = routing_error(
            "/v1/chat/completions",
            StatusCode::NOT_FOUND,
            "not_found_error",
            "unknown_url",
            "nope".to_string(),
        );
        assert_eq!(openai.body["error"]["type"], "invalid_request_error");
        assert_eq!(openai.body["error"]["code"], "unknown_url");
        assert_eq!(openai.body["error"]["message"], "nope");

        // The native surface answers in its own two-field envelope: no
        // top-level "type", no OpenAI "code"/"param".
        let native = routing_error(
            "/xwen/v1/generate",
            StatusCode::NOT_FOUND,
            "not_found_error",
            "unknown_url",
            "nope".to_string(),
        );
        assert_eq!(native.body["error"]["type"], "not_found_error");
        assert_eq!(native.body["error"]["message"], "nope");
        assert_eq!(native.body.get("type"), None);
        assert_eq!(native.body["error"].get("code"), None);
        // count_tokens belongs to the Anthropic surface too.
        assert!(is_anthropic_path("/v1/messages/count_tokens"));
        assert!(!is_anthropic_path("/v1/models"));
        assert!(is_native_path("/xwen/v1/generate"));
        assert!(!is_native_path("/v1/chat/completions"));
    }

    #[test]
    fn a_disabled_api_says_so_rather_than_reading_as_a_typo() {
        let mut settings = testutil::settings();
        settings.anthropic = false;
        settings.openai = false;
        let message = missing_route_message(&settings, &Method::POST, "/v1/messages");
        assert!(message.contains("Anthropic"), "{message}");
        assert!(message.contains("anthropic = false"), "{message}");
        let message = missing_route_message(&settings, &Method::POST, "/v1/chat/completions");
        assert!(message.contains("OpenAI"), "{message}");

        // With the API enabled, an unknown path is just an unknown path.
        let settings = testutil::settings();
        let message = missing_route_message(&settings, &Method::POST, "/v1/mesages");
        assert_eq!(message, "no route for POST /v1/mesages");
    }

    /// Named frames put `event:` first, which is the framing the Messages API
    /// documents. Asserted on the serialized bytes, since that is the only
    /// place the order exists.
    #[test]
    fn a_named_frame_serializes_its_event_line_first() {
        let frames = vec![
            SseFrame::named("message_start", json!({"type": "message_start"})),
            SseFrame::unnamed(json!({"id": "chatcmpl-1"})),
        ];
        let wire = serialize_frames(frames);
        assert!(
            wire.starts_with("event: message_start\ndata: {\"type\":\"message_start\"}\n\n"),
            "{wire:?}"
        );
        // An unnamed frame carries a data line and nothing else.
        assert!(
            wire.ends_with("data: {\"id\":\"chatcmpl-1\"}\n\n"),
            "{wire:?}"
        );
    }

    /// Render frames exactly as the SSE response would, and read the bytes back.
    fn serialize_frames(frames: Vec<SseFrame>) -> String {
        struct Scripted(VecDeque<SseFrame>);
        impl SseEncoder for Scripted {
            fn on_event(&mut self, _: EngineEvent, _: &mut VecDeque<SseFrame>) -> bool {
                true
            }
            fn on_hangup(&mut self, out: &mut VecDeque<SseFrame>) {
                out.append(&mut self.0);
            }
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        // Inside the runtime: the keep-alive timer the response carries needs a
        // reactor the moment it is built.
        runtime.block_on(async move {
            let (sender, events) = mpsc::channel(1);
            // Dropping the sender ends the stream at once, which flushes the
            // scripted frames through `on_hangup`.
            drop(sender);
            let guard = CancelGuard::new(Arc::new(Cancel::default()));
            let response = sse_response(events, guard, Scripted(frames.into()));
            let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
                .await
                .expect("the stream body collects");
            String::from_utf8(body.to_vec()).expect("SSE frames are UTF-8")
        })
    }

    #[test]
    fn a_wrong_key_is_compared_without_an_early_exit() {
        assert!(secrets_match("secret", "secret"));
        assert!(!secrets_match("secret", "secreT"));
        // A prefix of the key is not the key, in either direction.
        assert!(!secrets_match("secret", "sec"));
        assert!(!secrets_match("sec", "secret"));
        assert!(secrets_match("", ""));
    }

    #[test]
    fn random_ids_carry_the_prefix_and_differ() {
        let first = random_id("msg_");
        let second = random_id("msg_");
        assert!(first.starts_with("msg_") && first.len() == 28);
        assert_ne!(first, second);
    }

    /// Submit-side state over the real tokenizer and a probe queue standing in
    /// for the engine, so the render+encode+validate path runs without a model.
    fn probe_state(max_ctx: usize) -> (AppState, Arc<JobQueue>) {
        let jobs = Arc::new(JobQueue::new(
            4,
            SchedulePolicy {
                schedule: config::Schedule::ShortestPrefill,
                queue_timeout: Duration::from_secs(300),
                age_limit: Duration::from_secs(20),
            },
            ServeLogger::discarding(),
        ));
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/tokenizer.json");
        let state = AppState {
            jobs: Arc::clone(&jobs),
            tokenizer: Arc::new(
                LagunaTokenizer::from_file(path).expect("load reference tokenizer"),
            ),
            settings: Arc::new(testutil::settings()),
            model_loaded: Arc::new(AtomicBool::new(false)),
            shutdown: Arc::new(Cancel::default()),
            model_id: "xwen-test".to_string(),
            default_size: crate::hub::Model::default(),
            max_ctx,
            next_request_id: Arc::new(AtomicU64::new(1)),
        };
        (state, jobs)
    }

    /// What the engine's dequeue would see right now: a non-blocking take with
    /// nothing cached.
    fn try_take(queue: &JobQueue) -> Option<Queued> {
        queue.take(Some(Duration::ZERO), &|_| 0)
    }

    /// The generation inside a queued job; these tests submit nothing else.
    fn generation(job: types::Job) -> GenerationJob {
        match job {
            types::Job::Generation(job) => *job,
            types::Job::Batch(_) => panic!("these tests submit no batch jobs"),
        }
    }

    /// `submit` with the default checkpoint, which is what every call in these
    /// tests means — model selection has its own test below.
    fn submit_default(
        state: &AppState,
        request: JobRequest,
        dialect: Dialect,
        streaming: bool,
    ) -> std::result::Result<(mpsc::Receiver<EngineEvent>, CancelGuard), SubmitError> {
        submit(state, request, dialect, streaming, state.default_size)
    }

    fn probe_request(max_tokens: usize) -> JobRequest {
        JobRequest {
            messages: vec![Message::User("Hi".into())],
            enable_thinking: true,
            max_think: None,
            max_tokens,
            sampling: SamplerOptions::default(),
            stop_sequences: Vec::new(),
            tools: Vec::new(),
            grammar: None,
            continuation: None,
        }
    }

    /// The object schema the constrained-submit tests decode under.
    fn probe_schema() -> Value {
        json!({
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
            "additionalProperties": false,
        })
    }

    /// The checks a request alone can fail are judged at submit, as a 400 in the
    /// handler's dialect; the engine never sees the job.
    #[test]
    fn submit_refuses_a_zero_output_budget_before_the_queue() {
        let (state, queue) = probe_state(4096);
        match submit_default(&state, probe_request(0), Dialect::Anthropic, false) {
            Err(SubmitError::Invalid(message)) => {
                assert!(message.contains("max_tokens"), "{message}");
            }
            _ => panic!("a zero output budget must be refused as invalid"),
        }
        assert!(try_take(&queue).is_none(), "nothing may reach the engine");
    }

    #[test]
    fn submit_refuses_a_prompt_the_context_cannot_hold() {
        let (state, queue) = probe_state(8);
        match submit_default(&state, probe_request(64), Dialect::Anthropic, false) {
            Err(SubmitError::Invalid(message)) => {
                assert!(message.contains("8-token context"), "{message}");
            }
            _ => panic!("an oversize prompt must be refused as invalid"),
        }
        assert!(try_take(&queue).is_none(), "nothing may reach the engine");
    }

    /// The job the engine dequeues carries the prompt already encoded, split at
    /// the generation header: the ids equal a single-shot encode of the full
    /// rendering (the token-split invariant), and the header is what says
    /// whether the decode starts inside a thinking span.
    #[test]
    fn a_submitted_job_carries_the_prompt_split_at_the_generation_header() {
        let (state, queue) = probe_state(4096);
        let whole = chat::build_prompt(
            &[Message::User("Hi".into())],
            &ChatOptions {
                enable_thinking: true,
                preserve_thinking: false,
                tools: Vec::new(),
            },
        )
        .expect("the prompt renders");
        let expected = state.tokenizer.encode(&whole).expect("the prompt encodes");

        let (_events, _guard) = submit_default(&state, probe_request(64), Dialect::Anthropic, true)
            .expect("the job submits");
        let queued = try_take(&queue).expect("the job reached the queue");
        assert_eq!(queued.prompt_tokens, queued.job.prompt().len());
        let job = generation(queued.job);
        // The two facts only the handler knows travel with the job.
        assert_eq!(job.origin.dialect, Dialect::Anthropic);
        assert!(job.origin.streaming);
        assert_eq!(job.prompt, expected);
        assert!(job.boundary < job.prompt.len());
        assert!(job.starts_in_thinking);
        assert_eq!(job.max_tokens, 64);

        // With thinking disabled the header closes the span instead of opening
        // it, and the job says so.
        let (_events, _guard) = submit_default(
            &state,
            JobRequest {
                enable_thinking: false,
                ..probe_request(64)
            },
            Dialect::OpenAi,
            false,
        )
        .expect("the job submits");
        let second = generation(try_take(&queue).expect("the job reached the queue").job);
        assert!(!second.starts_in_thinking);
        // Ids are handed out in order, so a consumer can tell one request's
        // events from another's without the handlers agreeing on anything.
        assert_eq!(second.origin.id, job.origin.id + 1);
        assert_eq!(second.origin.dialect, Dialect::OpenAi);
    }

    /// A thinking budget governs an open reasoning span; a continuation that
    /// already closed the span must shed it, or the armed controller would wait
    /// for a decoded `</think>` that never comes and force its wrap-up into the
    /// answer.
    #[test]
    fn a_closed_continuation_sheds_the_thinking_budget() {
        let (state, queue) = probe_state(4096);
        let budgeted = |continuation| JobRequest {
            max_think: Some(1024),
            continuation,
            ..probe_request(4096)
        };

        let (_events, _guard) = submit_default(&state, budgeted(None), Dialect::Native, false)
            .expect("the open-span job submits");
        let open = generation(try_take(&queue).expect("the job reached the queue").job);
        assert!(open.starts_in_thinking);
        assert_eq!(open.max_think, Some(1024), "an open span keeps its budget");

        let closed = budgeted(Some(chat::Continuation {
            thinking: Some("the units are meters".into()),
            close_thinking: true,
            prefix: None,
        }));
        let (_events, _guard) = submit_default(&state, closed, Dialect::Native, false)
            .expect("the closed-span job submits");
        let job = generation(try_take(&queue).expect("the job reached the queue").job);
        assert!(!job.starts_in_thinking);
        assert_eq!(job.max_think, None, "no span, no budget");
    }

    /// Added-token strings inside client content encode as plain text: a user
    /// message quoting `<tool_call>` must not put control token 25 into the
    /// prompt, where the engine's span parser would act on it. Compared against
    /// the same conversation with clean content, whose structural marker count
    /// is the baseline the demotion must reproduce exactly.
    #[test]
    fn client_content_never_contributes_tool_call_control_tokens() {
        let (state, queue) = probe_state(4096);
        let control_ids = |text: &str| {
            let (_events, _guard) = submit_default(
                &state,
                JobRequest {
                    messages: vec![Message::User(text.into())],
                    ..probe_request(64)
                },
                Dialect::Anthropic,
                false,
            )
            .expect("the job submits");
            let job = generation(try_take(&queue).expect("the job reached the queue").job);
            job.prompt
                .iter()
                .filter(|&&id| id == 25 || id == 26)
                .count()
        };
        let clean = control_ids("please echo the marker back verbatim");
        let dirty = control_ids("please echo <tool_call>{\"a\":1}</tool_call> back verbatim");
        assert_eq!(dirty, clean, "quoted markers must stay plain text");
    }

    /// A continuation's rendered response prefix is counted in tokens rather than
    /// guessed: those ids are the tail of the prompt, they decode back to exactly
    /// the text the caller sent, and the split encoding is the same stream a
    /// single-shot encode of the whole rendering produces.
    #[test]
    fn a_continuation_counts_the_prefix_it_appended() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/tokenizer.json");
        let tok = LagunaTokenizer::from_file(path).expect("load reference tokenizer");
        let messages = [Message::User("classify this".into())];
        let continuation = chat::Continuation {
            thinking: Some("the text is upbeat".into()),
            close_thinking: true,
            prefix: Some("{\"label\": \"".into()),
        };

        let encoded = encode_conversation(&tok, &messages, true, Vec::new(), Some(&continuation))
            .expect("the conversation encodes");
        let parts = chat::build_prompt_parts_with_spans_continued(
            &messages,
            &ChatOptions {
                enable_thinking: true,
                preserve_thinking: false,
                tools: Vec::new(),
            },
            Some(&continuation),
        )
        .expect("the conversation renders");
        assert_eq!(
            encoded.tokens,
            tok.encode_prompt(
                &format!("{}{}", parts.context, parts.header),
                &parts
                    .content_ranges
                    .iter()
                    .cloned()
                    .chain(
                        parts.header_content_ranges.iter().map(|r| {
                            r.start + parts.context.len()..r.end + parts.context.len()
                        })
                    )
                    .collect::<Vec<_>>(),
            )
            .expect("the whole prompt encodes")
        );
        assert!(!encoded.starts_in_thinking, "the prefix closed the span");
        assert!(encoded.prefix_len > 0);
        let tail = &encoded.tokens[encoded.tokens.len() - encoded.prefix_len..];
        assert_eq!(
            tok.decode(tail).expect("the prefix's ids decode"),
            "{\"label\": \""
        );

        // Without a continuation nothing is appended, and the header is the
        // template's own.
        let plain = encode_conversation(&tok, &messages, true, Vec::new(), None)
            .expect("the conversation encodes");
        assert_eq!(plain.prefix_len, 0);
        assert!(plain.starts_in_thinking);
    }

    /// A prefix the header already wrote is fed to the grammar before the first
    /// draw, so the mask continues that document rather than opening a second
    /// one — and a prefix the schema does not accept is a 400 before the queue.
    #[test]
    fn submit_feeds_a_response_prefix_to_the_grammar() {
        let (state, queue) = probe_state(4096);
        let factory = crate::constrain::shared().expect("the shared factory builds");
        let constrained = |prefix: &str| JobRequest {
            grammar: Some(
                factory
                    .compile(&probe_schema())
                    .expect("the schema compiles"),
            ),
            continuation: Some(chat::Continuation {
                thinking: None,
                close_thinking: true,
                prefix: Some(prefix.to_string()),
            }),
            ..probe_request(64)
        };

        let (_events, _guard) = submit_default(
            &state,
            constrained("{\"label\": \"sunny\""),
            Dialect::Native,
            false,
        )
        .expect("a prefix the schema accepts submits");
        let job = generation(try_take(&queue).expect("the job reached the queue").job);
        let mut grammar = job.grammar.expect("the job carries an armed grammar");
        let words = grammar
            .mask_words()
            .expect("the mask computes")
            .expect("a prefix past </think> arms the grammar");
        // With the object's one property written, the only continuation is the
        // closing brace — where a grammar that had started over would be
        // offering the opening one.
        let bit = |text: &str| {
            let ids = state.tokenizer.encode(text).expect("encodes");
            assert_eq!(ids.len(), 1, "{text:?} is one token");
            words
                .get((ids[0] / 32) as usize)
                .is_some_and(|word| word & (1 << (ids[0] % 32)) != 0)
        };
        assert!(bit("}"), "the value cannot be closed");
        assert!(
            !bit("{"),
            "the mask restarted the document instead of continuing it"
        );

        match submit_default(&state, constrained("[1, 2"), Dialect::Native, false) {
            Err(SubmitError::Invalid(message)) => {
                assert!(message.contains("schema"), "{message}");
            }
            _ => panic!("a prefix the schema rejects must be refused as invalid"),
        }
        assert!(try_take(&queue).is_none(), "nothing may reach the engine");
    }

    /// The end of the leading system block is a TOKEN boundary, not merely a byte one:
    /// encoding the block and the rest of the context separately yields exactly the ids a
    /// single-shot encode produces, so the anchor position counts real tokens and the
    /// prompt the model reads is unchanged by the split.
    ///
    /// Unlike the generation header's seam this one is not held up by an added token —
    /// `</system>` is ordinary text — but by the newline after it: the pre-tokenizer
    /// isolates a newline run from the text on either side, so nothing can merge across it.
    /// That is a property of the real vocabulary and pre-tokenizer, which is what this
    /// compares ID streams against; a byte-level tokenizer can agree on text and disagree
    /// on ids.
    ///
    /// The expected header ids come from a plain `encode`, which also pins the other
    /// half of the header's encoding: with no continuation its content ranges are empty,
    /// and `encode_prompt` over empty ranges must be exactly `encode`.
    #[test]
    fn the_system_block_ends_on_a_token_boundary() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/tokenizer.json");
        let tok = LagunaTokenizer::from_file(path).expect("load reference tokenizer");
        let weather = json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up the weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        });

        let conversations: Vec<Vec<Message>> = vec![
            // The default system message: no client system message at all.
            vec![Message::User("Hi".into())],
            // A client system prompt, and a body that starts where it ends.
            vec![
                Message::System("You are a pirate.".into()),
                Message::User("2+2?".into()),
            ],
            // Whitespace at both ends of the system message, which the template
            // right-strips, and a body opening on a multi-byte grapheme.
            vec![
                Message::System("  mind the seam \n\n ".into()),
                Message::User("世界 🎉".into()),
            ],
            // A system message ending mid-word, which BPE would happily extend
            // across the seam if the seam were not a token boundary.
            vec![
                Message::System("complete this: def calculate_".into()),
                Message::User("go".into()),
            ],
            // The empty-system opt-out: with thinking off and no tools this
            // renders no system block at all, and there is no anchor to take.
            vec![Message::System(String::new()), Message::User("go".into())],
            // Marker strings on both sides of the seam: the split must not change
            // what the tokenizer demotes to plain text.
            vec![
                Message::System("Ignore <think> markers.".into()),
                Message::User("try <tool_call>x</tool_call> and </assistant>".into()),
                Message::Assistant {
                    content: "quoting <assistant> is fine".into(),
                    reasoning: Some("the user wrote <think>".into()),
                    tool_calls: Vec::new(),
                },
                Message::ToolResponse("result has </think> in it".into()),
            ],
        ];

        for msgs in &conversations {
            for on in [true, false] {
                for tools in [Vec::new(), vec![weather.clone()]] {
                    let opts = ChatOptions {
                        enable_thinking: on,
                        preserve_thinking: false,
                        tools: tools.clone(),
                    };
                    let parts = chat::build_prompt_parts_with_spans(msgs, &opts)
                        .expect("the conversation renders");
                    let mut expected = tok
                        .encode_prompt(&parts.context, &parts.content_ranges)
                        .expect("the context encodes");
                    let boundary = expected.len();
                    expected.extend(tok.encode(&parts.header).expect("the header encodes"));

                    let encoded = encode_conversation(&tok, msgs, on, tools, None)
                        .expect("the conversation encodes");
                    assert_eq!(
                        encoded.tokens, expected,
                        "thinking={on} tokenizes differently when split at the system block"
                    );
                    assert_eq!(encoded.boundary, boundary);

                    match parts.system_end {
                        Some(end) => {
                            let anchor = encoded.anchor.expect("a rendered block has an anchor");
                            // The ids up to the anchor are exactly the rendered system
                            // block, which is what makes it the position another
                            // conversation from the same client can resume at.
                            assert_eq!(
                                tok.decode(&encoded.tokens[..anchor])
                                    .expect("the block's ids decode"),
                                parts.context[..end]
                            );
                        }
                        None => assert_eq!(encoded.anchor, None),
                    }
                }
            }
        }
    }

    /// The headers a builder attaches ride into the HTTP response.
    #[test]
    fn api_error_headers_reach_the_response() {
        let error = ApiError {
            status: StatusCode::TOO_MANY_REQUESTS,
            body: json!({"ok": false}),
            headers: Vec::new(),
        }
        .with_header("retry-after", "1");
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok()),
            Some("1")
        );
    }
}
