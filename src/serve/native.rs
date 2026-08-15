//! The native generate endpoint: POST /xwen/v1/generate.
//!
//! This is the engine's own API rather than a compatibility dialect, so it is
//! strict where the other two are permissive: it is `deny_unknown_fields`
//! throughout — a misspelled field is a 400, not a silently dropped parameter —
//! and it carries no model name, no tools and none of the SDK-shaped block
//! ceremony. What it does carry is what neither compat dialect can spell:
//! the current turn's reasoning and the start of its answer, injected into the
//! generation header for the model to continue, and either of those combined
//! with a schema constraint on the rest.

use std::collections::VecDeque;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use super::config::ServeSettings;
use super::types::{Dialect, EngineEvent, StopKind};
use super::{
    ApiError, AppState, Completion, EngineFailure, JobRequest, SseEncoder, SseFrame, SubmitError,
    collect_completion, random_id, sse_response, submit,
};
use crate::chat::{Continuation, Message};
use crate::constrain::{self, Grammar};
use crate::generate::feasible_think_budget;
use crate::sampler::SamplerOptions;

/// A conversation with no turns has nothing to answer.
const EMPTY_MESSAGES: &str = "messages: at least one message is required";
/// Injected reasoning needs a `<think>` block to live in, and with thinking off
/// the header never opens one.
const THINKING_OFF_WITH_REASONING: &str = "continue.thinking cannot be used with thinking disabled: there is no reasoning block to \
     continue";
/// With thinking off the header closes the span itself, so asking for it to be
/// closed describes a turn that does not exist.
const THINKING_OFF_WITH_CLOSE: &str = "continue.close_thinking cannot be used with thinking disabled: the turn opens no reasoning \
     block";
/// A prefix written before `</think>` would be read back as reasoning, not as
/// the answer the caller means it to be.
const PREFIX_INSIDE_THINKING: &str = "continue.prefix needs the reasoning block closed: set continue.close_thinking, or disable \
     thinking";

/// The error envelope every failure on this API wears.
pub(crate) fn error(status: StatusCode, kind: &str, message: impl Into<String>) -> ApiError {
    ApiError {
        status,
        body: json!({"error": {"type": kind, "message": message.into()}}),
        headers: Vec::new(),
    }
}

fn bad_request(message: impl Into<String>) -> ApiError {
    error(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

/// The queue-full answer: 429 with a short `Retry-After`, matching both compat
/// dialects. 529 is banned server-wide, and a `Retry-After` past 60 seconds is
/// fatal to clients that honour it.
fn overloaded() -> ApiError {
    error(
        StatusCode::TOO_MANY_REQUESTS,
        "overloaded_error",
        "the model is busy with other requests; retry shortly",
    )
    .with_header("retry-after", "1")
}

/// An engine failure, in the envelope its cause deserves. The engine says which
/// kind it was; nothing here second-guesses it from the message text.
fn engine_error(message: &str, request_fault: bool) -> ApiError {
    if request_fault {
        bad_request(message.to_string())
    } else {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            message.to_string(),
        )
    }
}

// ---------------------------------------------------------------- request ---

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GenerateRequest {
    /// The system message. Absent or empty leaves the template's default in
    /// place; there is no way to spell "no system block at all" here.
    pub system: Option<String>,
    pub messages: Vec<InputMessage>,
    /// The start of the turn the model is about to write. Named `continue` on
    /// the wire, which is a Rust keyword.
    #[serde(rename = "continue")]
    pub continuation: Option<ContinueBlock>,
    pub format: Option<OutputFormat>,
    /// Whether the turn reasons before answering. Absent follows the server's
    /// thinking policy, like the compat dialects.
    pub thinking: Option<bool>,
    /// Ceiling on newly GENERATED reasoning tokens; injected `continue.thinking`
    /// is prompt content and does not count against it. Absent follows the
    /// server's configured budget.
    pub max_thinking_tokens: Option<usize>,
    pub max_tokens: usize,
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub seed: Option<u64>,
    pub stop_sequences: Option<Vec<String>>,
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct InputMessage {
    pub role: String,
    pub content: String,
    /// An assistant turn's reasoning, replayed verbatim. Unlike the compat
    /// dialects there is no trailing-run predicate: this API replays exactly
    /// what the caller sent, and thinking being off is what drops it.
    pub thinking: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ContinueBlock {
    pub thinking: Option<String>,
    #[serde(default)]
    pub close_thinking: bool,
    pub prefix: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OutputFormat {
    #[serde(rename = "type")]
    pub kind: String,
    pub schema: Option<Value>,
}

/// A request reduced to what the engine and the response builders need. The
/// model is not among them: this server has one, and the response echoes it.
pub(crate) struct Prepared {
    pub stream: bool,
    pub job: JobRequest,
}

/// Turn the wire block into the renderer's [`Continuation`], refusing the
/// combinations that have no rendering. `chat.rs` refuses two of them again;
/// this is where they get an error a caller can act on.
fn resolve_continuation(
    block: Option<&ContinueBlock>,
    thinking: bool,
) -> Result<Option<Continuation>, ApiError> {
    let Some(block) = block else {
        return Ok(None);
    };
    // An empty string is how a client spells "not supplied" without dropping the
    // key, and the renderer treats it that way too.
    let injected = block.thinking.clone().filter(|text| !text.is_empty());
    let prefix = block.prefix.clone().filter(|text| !text.is_empty());
    if !thinking {
        if injected.is_some() {
            return Err(bad_request(THINKING_OFF_WITH_REASONING));
        }
        if block.close_thinking {
            return Err(bad_request(THINKING_OFF_WITH_CLOSE));
        }
    } else if prefix.is_some() && !block.close_thinking {
        return Err(bad_request(PREFIX_INSIDE_THINKING));
    }
    Ok(Some(Continuation {
        thinking: injected,
        close_thinking: block.close_thinking,
        prefix,
    }))
}

/// Resolve `format` into the grammar the job will decode under. Compiled here,
/// on the HTTP thread, so a schema the compiler rejects is a 400 carrying the
/// compiler's own message; only the shared factory failing is a 500.
fn resolve_format(format: Option<&OutputFormat>) -> Result<Option<Grammar>, ApiError> {
    let Some(format) = format else {
        return Ok(None);
    };
    // The type is read before the factory is touched, so an unknown one is the
    // 400 it is rather than whatever building the trie reports.
    let schema =
        match format.kind.as_str() {
            // A schema alongside json_object is a contradiction, not a hint:
            // this API refuses fields it would ignore.
            "json_object" => match format.schema {
                Some(_) => {
                    return Err(bad_request(
                        "format.schema does not apply to a json_object format; use json_schema",
                    ));
                }
                None => None,
            },
            "json_schema" => Some(format.schema.as_ref().ok_or_else(|| {
                bad_request("format.schema is required for a json_schema format")
            })?),
            other => {
                return Err(bad_request(format!(
                    "format type {other:?} is not supported; use \"json_schema\" or \"json_object\""
                )));
            }
        };
    let factory = constrain::shared().map_err(|e| {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            format!("{e:#}"),
        )
    })?;
    match schema {
        Some(schema) => factory.compile(schema),
        None => factory.compile_any_object(),
    }
    .map(Some)
    .map_err(|e| bad_request(format!("{e:#}")))
}

/// Normalize a request's conversation into the turns `chat.rs` renders.
fn normalize(request: &GenerateRequest) -> Result<Vec<Message>, ApiError> {
    let mut messages = Vec::new();
    if let Some(system) = request.system.as_ref().filter(|text| !text.is_empty()) {
        messages.push(Message::System(system.clone()));
    }
    for message in &request.messages {
        match message.role.as_str() {
            // Refused rather than dropped: accepting `thinking` here would let a
            // caller believe the text was replayed.
            "user" if message.thinking.is_some() => {
                return Err(bad_request(
                    "thinking is only valid on assistant messages; a user turn has no \
                     reasoning block to replay it into",
                ));
            }
            "user" => messages.push(Message::User(message.content.clone())),
            "assistant" => messages.push(Message::Assistant {
                content: message.content.clone(),
                reasoning: message.thinking.clone().filter(|text| !text.is_empty()),
                tool_calls: Vec::new(),
            }),
            role => {
                return Err(bad_request(format!(
                    "unknown message role {role:?}: expected \"user\" or \"assistant\""
                )));
            }
        }
    }
    Ok(messages)
}

pub(crate) fn prepare(
    request: GenerateRequest,
    settings: &ServeSettings,
) -> Result<Prepared, ApiError> {
    if request.messages.is_empty() {
        return Err(bad_request(EMPTY_MESSAGES));
    }
    if request.max_tokens == 0 {
        return Err(bad_request("max_tokens must be at least 1"));
    }
    // Absent fields follow the operator's server policy, like the compat
    // dialects; explicit fields override it.
    let enable_thinking = request.thinking.unwrap_or(settings.thinking_force);
    let continuation = resolve_continuation(request.continuation.as_ref(), enable_thinking)?;
    let grammar = resolve_format(request.format.as_ref())?;
    let messages = normalize(&request)?;
    // The ceiling has to leave the reply room to conclude in: it is lowered to
    // the largest one that fits, or dropped when none does — the same clamp both
    // compat dialects apply.
    let max_think = enable_thinking
        .then(|| {
            request
                .max_thinking_tokens
                .or(settings.thinking_budget)
                .and_then(|budget| feasible_think_budget(budget, request.max_tokens))
        })
        .flatten();

    let defaults = SamplerOptions::default();
    Ok(Prepared {
        stream: request.stream.unwrap_or(false),
        job: JobRequest {
            messages,
            enable_thinking,
            max_think,
            max_tokens: request.max_tokens,
            sampling: SamplerOptions {
                temperature: request.temperature.unwrap_or(settings.temperature),
                top_k: request.top_k.unwrap_or(settings.top_k),
                top_p: request.top_p.unwrap_or(settings.top_p),
                seed: request.seed.unwrap_or(defaults.seed),
            },
            stop_sequences: request.stop_sequences.unwrap_or_default(),
            tools: Vec::new(),
            grammar,
            continuation,
        },
    })
}

// --------------------------------------------------------------- response ---

/// `stop_reason` plus the `stop_sequence` that goes with it. This API serves no
/// tools, so a turn can never end on a call.
fn stop_fields(stop: &StopKind) -> (&'static str, Value) {
    match stop {
        StopKind::EndTurn | StopKind::ToolUse => ("end_turn", Value::Null),
        StopKind::MaxTokens => ("max_tokens", Value::Null),
        StopKind::StopSequence(sequence) => ("stop_sequence", Value::String(sequence.clone())),
    }
}

fn usage(
    input_tokens: usize,
    cached_tokens: usize,
    output_tokens: usize,
    thinking_tokens: usize,
) -> Value {
    json!({
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
    })
}

/// The buffered response. `content` is what this turn generated: an injected
/// response prefix is the caller's own text and is not echoed back.
pub(crate) fn generation_body(id: &str, model: &str, completion: &Completion) -> Value {
    let (stop_reason, stop_sequence) = stop_fields(&completion.stop);
    json!({
        "id": id,
        "model": model,
        "thinking": match completion.thinking.is_empty() {
            true => Value::Null,
            false => Value::String(completion.thinking.clone()),
        },
        "content": completion.text,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": usage(
            completion.input_tokens,
            completion.cached_tokens,
            completion.output_tokens,
            completion.thinking_tokens,
        ),
    })
}

// -------------------------------------------------------------------- SSE ---

/// Renders a generation as this API's event sequence: `start`, a `delta` per
/// engine delta, and one `done` — or an `error` frame in place of the `done`.
/// There is no block indexing to keep straight, since a delta says which of the
/// two spans it belongs to.
pub(crate) struct GenerationStream {
    id: String,
    model: String,
    input_tokens: usize,
    cached_tokens: usize,
}

impl GenerationStream {
    pub(crate) fn new(id: String, model: String) -> Self {
        Self {
            id,
            model,
            input_tokens: 0,
            cached_tokens: 0,
        }
    }

    fn delta(kind: &str, text: String, out: &mut VecDeque<SseFrame>) {
        if text.is_empty() {
            return;
        }
        out.push_back(SseFrame::named(
            "delta",
            json!({"type": kind, "text": text}),
        ));
    }

    fn push_error(message: &str, kind: &str, out: &mut VecDeque<SseFrame>) {
        out.push_back(SseFrame::named(
            "error",
            json!({"error": {"type": kind, "message": message}}),
        ));
    }
}

impl SseEncoder for GenerationStream {
    fn on_event(&mut self, event: EngineEvent, out: &mut VecDeque<SseFrame>) -> bool {
        match event {
            EngineEvent::Start {
                input_tokens,
                cached_tokens,
            } => {
                // Held for the `done` frame's usage, which is where a stream
                // reports the prompt side too.
                self.input_tokens = input_tokens;
                self.cached_tokens = cached_tokens;
                out.push_back(SseFrame::named(
                    "start",
                    json!({"id": self.id, "model": self.model}),
                ));
                false
            }
            EngineEvent::Thinking(text) => {
                Self::delta("thinking", text, out);
                false
            }
            EngineEvent::Text(text) => {
                Self::delta("content", text, out);
                false
            }
            // A request that carries no tools cannot produce a call span: the
            // engine treats `<tool_call>` in the output as ordinary text. A
            // batch document can never arrive on a generation's channel either.
            EngineEvent::ToolCallStart { .. }
            | EngineEvent::ToolCallDelta(_)
            | EngineEvent::ToolCallEnd
            | EngineEvent::BatchDone(_) => false,
            EngineEvent::Done {
                stop,
                output_tokens,
                thinking_tokens,
            } => {
                let (stop_reason, stop_sequence) = stop_fields(&stop);
                out.push_back(SseFrame::named(
                    "done",
                    json!({
                        "stop_reason": stop_reason,
                        "stop_sequence": stop_sequence,
                        "usage": usage(
                            self.input_tokens,
                            self.cached_tokens,
                            output_tokens,
                            thinking_tokens,
                        ),
                    }),
                ));
                true
            }
            EngineEvent::Error {
                message,
                request_fault,
            } => {
                let kind = if request_fault {
                    "invalid_request_error"
                } else {
                    "api_error"
                };
                Self::push_error(&message, kind, out);
                true
            }
        }
    }

    fn on_hangup(&mut self, out: &mut VecDeque<SseFrame>) {
        Self::push_error(
            "the inference engine stopped before finishing the response",
            "api_error",
            out,
        );
    }
}

// ---------------------------------------------------------------- handler ---

pub(crate) async fn generate(State(state): State<AppState>, body: Bytes) -> Response {
    let request: GenerateRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(e) => {
            return bad_request(format!("could not parse the request body: {e}")).into_response();
        }
    };
    let prepared = match prepare(request, &state.settings) {
        Ok(prepared) => prepared,
        Err(e) => return e.into_response(),
    };
    let Prepared { stream, job } = prepared;
    let id = random_id("gen_");
    let model = state.model_id.clone();

    // This endpoint carries no model name by design; it runs on the default
    // checkpoint. The batch endpoint is the native surface that names one.
    let (mut events, guard) =
        match submit(&state, job, Dialect::Native, stream, state.default_target) {
            Ok(submitted) => submitted,
            Err(SubmitError::Invalid(message)) => return bad_request(message).into_response(),
            Err(SubmitError::Overloaded) => return overloaded().into_response(),
            Err(SubmitError::EngineGone) => {
                return error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    "the inference engine is not running",
                )
                .into_response();
            }
        };

    if stream {
        return sse_response(events, guard, GenerationStream::new(id, model));
    }
    // Held across the drain: hyper dropping this future — the client hung up —
    // drops the guard, which cancels the generation with its reason stated.
    let _guard = guard;
    match collect_completion(&mut events).await {
        Ok(completion) => axum::Json(generation_body(&id, &model, &completion)).into_response(),
        Err(EngineFailure::Reported {
            message,
            request_fault,
        }) => engine_error(&message, request_fault).into_response(),
        Err(EngineFailure::Hangup) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            "the inference engine stopped before finishing the response",
        )
        .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::testutil::{encode_all, names, payload, settings, shape};

    fn parse(body: &str) -> GenerateRequest {
        serde_json::from_str(body).expect("request parses")
    }

    fn prepared(body: &str) -> Prepared {
        prepare(parse(body), &settings()).expect("request prepares")
    }

    fn rejected(body: &str) -> ApiError {
        prepare(parse(body), &settings())
            .err()
            .expect("request is rejected")
    }

    fn message(error: &ApiError) -> String {
        error.body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .to_string()
    }

    /// The rendered continuation, as the job carries it.
    fn continuation(prepared: &Prepared) -> (Option<&str>, bool, Option<&str>) {
        let continuation = prepared
            .job
            .continuation
            .as_ref()
            .expect("the request carries a continuation");
        (
            continuation.thinking.as_deref(),
            continuation.close_thinking,
            continuation.prefix.as_deref(),
        )
    }

    /// This is our own API, so a field nobody recognizes is a caller mistake to
    /// report rather than a newer SDK's addition to ignore.
    #[test]
    fn unknown_fields_are_refused_rather_than_ignored() {
        let error = serde_json::from_str::<GenerateRequest>(
            r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],"temperture":0.5}"#,
        )
        .expect_err("an unknown field is a parse error");
        assert!(error.to_string().contains("temperture"), "{error}");
        // Including inside the nested blocks.
        assert!(
            serde_json::from_str::<GenerateRequest>(
                r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],
                    "continue":{"prefix":"x","closeThinking":true}}"#,
            )
            .is_err()
        );
    }

    #[test]
    fn a_plain_request_prepares_with_the_servers_defaults() {
        let plain = prepared(r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}"#);
        assert!(!plain.stream);
        assert!(plain.job.enable_thinking, "thinking defaults on");
        assert!(plain.job.continuation.is_none());
        assert!(plain.job.grammar.is_none());
        assert!(plain.job.tools.is_empty());
        assert_eq!(plain.job.max_think, None);
        assert_eq!(shape(&plain.job.messages), vec!["user:Hi"]);
        let settings = settings();
        assert_eq!(plain.job.sampling.temperature, settings.temperature);
        assert_eq!(plain.job.sampling.top_k, settings.top_k);
        assert_eq!(plain.job.sampling.top_p, settings.top_p);
        assert_eq!(plain.job.sampling.seed, SamplerOptions::default().seed);
    }

    /// An absent `thinking` follows the operator's server policy — both the
    /// on/off switch and the configured budget — and an explicit field wins.
    #[test]
    fn absent_thinking_follows_the_server_policy() {
        let mut policy = settings();
        policy.thinking_force = false;
        policy.thinking_budget = Some(1024);
        let request = |body: &str| {
            prepare(
                serde_json::from_str::<GenerateRequest>(body).expect("parse"),
                &policy,
            )
            .expect("prepare")
        };
        let absent = request(r#"{"max_tokens":8192,"messages":[{"role":"user","content":"Hi"}]}"#);
        assert!(!absent.job.enable_thinking);
        assert_eq!(
            absent.job.max_think, None,
            "no budget survives thinking off"
        );
        let stated = request(
            r#"{"max_tokens":8192,"thinking":true,"messages":[{"role":"user","content":"Hi"}]}"#,
        );
        assert!(stated.job.enable_thinking);
        assert_eq!(stated.job.max_think, Some(1024), "server budget applies");
        let capped = request(
            r#"{"max_tokens":8192,"thinking":true,"max_thinking_tokens":256,
                "messages":[{"role":"user","content":"Hi"}]}"#,
        );
        assert_eq!(capped.job.max_think, Some(256), "a stated budget wins");
    }

    /// Sampling the request states wins over the server's configured defaults.
    #[test]
    fn stated_sampling_parameters_win() {
        let stated = prepared(
            r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],
                "temperature":0.2,"top_k":5,"top_p":0.8,"seed":7,"stop_sequences":["END"],
                "stream":true}"#,
        );
        assert!(stated.stream);
        assert_eq!(stated.job.sampling.temperature, 0.2);
        assert_eq!(stated.job.sampling.top_k, 5);
        assert_eq!(stated.job.sampling.top_p, 0.8);
        assert_eq!(stated.job.sampling.seed, 7);
        assert_eq!(stated.job.stop_sequences, vec!["END".to_string()]);
    }

    /// The conversation is replayed exactly as sent: a system message leads it,
    /// and an assistant turn keeps the reasoning the caller attached — no
    /// trailing-run predicate, which is the compat dialects' problem.
    #[test]
    fn the_conversation_is_replayed_verbatim() {
        let history = prepared(
            r#"{"max_tokens":16,"system":"Be brief.","messages":[
                {"role":"user","content":"2+2?"},
                {"role":"assistant","content":"4","thinking":"add them"},
                {"role":"user","content":"thanks"}]}"#,
        );
        assert_eq!(
            shape(&history.job.messages),
            vec![
                "system:Be brief.",
                "user:2+2?",
                "assistant:4|think:add them",
                "user:thanks"
            ]
        );
        // An empty system message leaves the template's default in place.
        let empty_system = prepared(
            r#"{"max_tokens":16,"system":"","messages":[{"role":"user","content":"Hi"}]}"#,
        );
        assert_eq!(shape(&empty_system.job.messages), vec!["user:Hi"]);
    }

    #[test]
    fn an_unknown_role_is_rejected() {
        let error = rejected(r#"{"max_tokens":16,"messages":[{"role":"tool","content":"sunny"}]}"#);
        assert!(
            message(&error).contains("unknown message role"),
            "{}",
            message(&error)
        );
    }

    /// `thinking` on a user turn has nowhere to go; accepting it would let the
    /// caller believe the text was replayed.
    #[test]
    fn thinking_on_a_user_message_is_refused_rather_than_dropped() {
        let error = rejected(
            r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi","thinking":"psst"}]}"#,
        );
        assert!(
            message(&error).contains("only valid on assistant messages"),
            "{}",
            message(&error)
        );
    }

    #[test]
    fn an_empty_conversation_and_a_zero_budget_are_rejected() {
        let error = rejected(r#"{"max_tokens":16,"messages":[]}"#);
        assert!(
            message(&error).contains("at least one message"),
            "{}",
            message(&error)
        );
        let error = rejected(r#"{"max_tokens":0,"messages":[{"role":"user","content":"Hi"}]}"#);
        assert!(
            message(&error).contains("max_tokens"),
            "{}",
            message(&error)
        );
    }

    /// The happy path this endpoint exists for: reasoning and an answer prefix
    /// injected into the turn, under a schema that constrains the rest of it.
    #[test]
    fn thinking_prefix_and_schema_prepare_together() {
        let continued = prepared(
            r#"{"max_tokens":64,"messages":[{"role":"user","content":"classify"}],
                "continue":{"thinking":"the text is upbeat","close_thinking":true,
                            "prefix":"{\"label\": \""},
                "format":{"type":"json_schema","schema":
                    {"type":"object","properties":{"label":{"type":"string"}},
                     "required":["label"],"additionalProperties":false}}}"#,
        );
        assert_eq!(
            continuation(&continued),
            (Some("the text is upbeat"), true, Some("{\"label\": \""))
        );
        assert!(continued.job.grammar.is_some());
    }

    /// Every combination the renderer has no shape for, refused with the field
    /// that has to change.
    #[test]
    fn unrenderable_continuations_are_rejected() {
        let error = rejected(
            r#"{"max_tokens":16,"thinking":false,"messages":[{"role":"user","content":"Hi"}],
                "continue":{"thinking":"reasoning"}}"#,
        );
        assert!(
            message(&error).contains("continue.thinking"),
            "{}",
            message(&error)
        );

        let error = rejected(
            r#"{"max_tokens":16,"thinking":false,"messages":[{"role":"user","content":"Hi"}],
                "continue":{"close_thinking":true}}"#,
        );
        assert!(
            message(&error).contains("continue.close_thinking"),
            "{}",
            message(&error)
        );

        let error = rejected(
            r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],
                "continue":{"prefix":"Sure,"}}"#,
        );
        assert!(
            message(&error).contains("continue.prefix"),
            "{}",
            message(&error)
        );
    }

    /// With thinking off the header closes the span itself, so a prefix needs
    /// nothing else asked for. Injected reasoning that is empty is not injected
    /// reasoning, and stays legal there.
    #[test]
    fn thinking_off_takes_a_prefix_on_its_own() {
        let answered = prepared(
            r#"{"max_tokens":16,"thinking":false,"messages":[{"role":"user","content":"Hi"}],
                "continue":{"thinking":"","prefix":"Sure,"}}"#,
        );
        assert!(!answered.job.enable_thinking);
        assert_eq!(continuation(&answered), (None, false, Some("Sure,")));
    }

    /// Reasoning left open is the other continuation shape: the model carries on
    /// thinking from where the caller left it.
    #[test]
    fn injected_reasoning_can_be_left_open() {
        let open = prepared(
            r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],
                "continue":{"thinking":"first, the units"}}"#,
        );
        assert_eq!(continuation(&open), (Some("first, the units"), false, None));
    }

    #[test]
    fn bad_formats_are_rejected_and_good_ones_compile() {
        let error = rejected(
            r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],
                "format":{"type":"json_schema"}}"#,
        );
        assert!(
            message(&error).contains("schema is required"),
            "{}",
            message(&error)
        );

        let error = rejected(
            r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],
                "format":{"type":"yaml"}}"#,
        );
        assert!(
            message(&error).contains("not supported"),
            "{}",
            message(&error)
        );

        // A schema the compiler refuses is a 400 carrying its message, not a 500.
        let error = rejected(
            r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],
                "format":{"type":"json_schema","schema":{"not":{"type":"string"}}}}"#,
        );
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(
            message(&error).contains("json_schema rejected"),
            "{}",
            message(&error)
        );

        // A schema alongside json_object would be ignored, so it is refused.
        let error = rejected(
            r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],
                "format":{"type":"json_object","schema":{"type":"object"}}}"#,
        );
        assert!(
            message(&error).contains("does not apply"),
            "{}",
            message(&error)
        );

        assert!(
            prepared(
                r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],
                    "format":{"type":"json_object"}}"#
            )
            .job
            .grammar
            .is_some()
        );
    }

    /// A thinking budget is clamped to one the reasoning schedule can express
    /// inside the output cap, and dropped entirely with thinking off.
    #[test]
    fn the_thinking_budget_is_clamped_to_the_output_cap() {
        let tight = prepared(
            r#"{"max_tokens":64,"max_thinking_tokens":100000,
                "messages":[{"role":"user","content":"Hi"}]}"#,
        );
        assert!(
            tight.job.max_think.is_none_or(|budget| budget < 64),
            "an infeasible budget must be lowered or dropped: {:?}",
            tight.job.max_think
        );
        let roomy = prepared(
            r#"{"max_tokens":8192,"max_thinking_tokens":1024,
                "messages":[{"role":"user","content":"Hi"}]}"#,
        );
        assert_eq!(roomy.job.max_think, Some(1024));
        // With thinking off the budget must not survive to arm the schedule.
        let off = prepared(
            r#"{"max_tokens":8192,"max_thinking_tokens":1024,"thinking":false,
                "messages":[{"role":"user","content":"Hi"}]}"#,
        );
        assert_eq!(off.job.max_think, None);
    }

    /// The stream's frames: one `start`, a `delta` per non-empty engine delta
    /// saying which span it belongs to, and a `done` carrying both sides of the
    /// usage.
    #[test]
    fn the_event_stream_names_its_frames() {
        let mut stream = GenerationStream::new("gen_1".into(), "laguna".into());
        let frames = encode_all(
            &mut stream,
            vec![
                EngineEvent::Start {
                    input_tokens: 12,
                    cached_tokens: 8,
                },
                EngineEvent::Thinking("hm".into()),
                EngineEvent::Text(String::new()),
                EngineEvent::Text("yes".into()),
                EngineEvent::Done {
                    stop: StopKind::EndTurn,
                    output_tokens: 5,
                    thinking_tokens: 2,
                },
            ],
        );
        assert_eq!(names(&frames), vec!["start", "delta", "delta", "done"]);
        assert_eq!(payload(&frames[0])["id"], "gen_1");
        assert_eq!(payload(&frames[0])["model"], "laguna");
        assert_eq!(
            payload(&frames[1]),
            json!({"type": "thinking", "text": "hm"})
        );
        assert_eq!(
            payload(&frames[2]),
            json!({"type": "content", "text": "yes"})
        );
        let done = payload(&frames[3]);
        assert_eq!(done["stop_reason"], "end_turn");
        assert_eq!(done["stop_sequence"], Value::Null);
        assert_eq!(
            done["usage"],
            json!({"input_tokens": 12, "cached_tokens": 8, "output_tokens": 5, "thinking_tokens": 2})
        );
    }

    /// An engine failure ends the stream with an error frame in the same
    /// envelope the buffered path answers with, and so does a hangup.
    #[test]
    fn a_failed_generation_ends_the_stream_with_an_error_frame() {
        let mut stream = GenerationStream::new("gen_1".into(), "laguna".into());
        let frames = encode_all(
            &mut stream,
            vec![
                EngineEvent::Start {
                    input_tokens: 1,
                    cached_tokens: 0,
                },
                EngineEvent::Error {
                    message: "the prompt does not fit".into(),
                    request_fault: true,
                },
            ],
        );
        assert_eq!(names(&frames), vec!["start", "error"]);
        let error = payload(&frames[1]);
        assert_eq!(error["error"]["type"], "invalid_request_error");
        assert_eq!(error["error"]["message"], "the prompt does not fit");

        let mut out = VecDeque::new();
        stream.on_hangup(&mut out);
        assert_eq!(payload(&out[0])["error"]["type"], "api_error");
    }

    /// The buffered body reports the generated text only — an injected prefix is
    /// the caller's own bytes — and nulls a thinking span that came out empty.
    #[test]
    fn the_buffered_body_reports_what_the_turn_generated() {
        let completion = Completion {
            input_tokens: 30,
            cached_tokens: 20,
            thinking: String::new(),
            text: "sunny\"}".into(),
            tool_calls: Vec::new(),
            stop: StopKind::StopSequence("\n\n".into()),
            output_tokens: 4,
            thinking_tokens: 0,
        };
        let body = generation_body("gen_1", "laguna", &completion);
        assert_eq!(body["id"], "gen_1");
        assert_eq!(body["model"], "laguna");
        assert_eq!(body["content"], "sunny\"}");
        assert_eq!(body["thinking"], Value::Null);
        assert_eq!(body["stop_reason"], "stop_sequence");
        assert_eq!(body["stop_sequence"], "\n\n");
        assert_eq!(
            body["usage"],
            json!({"input_tokens": 30, "cached_tokens": 20, "output_tokens": 4, "thinking_tokens": 0})
        );

        let body = generation_body(
            "gen_2",
            "laguna",
            &Completion {
                thinking: "hm".into(),
                stop: StopKind::MaxTokens,
                ..completion
            },
        );
        assert_eq!(body["thinking"], "hm");
        assert_eq!(body["stop_reason"], "max_tokens");
        assert_eq!(body["stop_sequence"], Value::Null);
    }
}
