//! OpenAI Chat Completions API surface: POST /v1/chat/completions.
//!
//! Same permissiveness as the Anthropic handler: unknown fields are ignored,
//! and the parameters this engine has no equivalent for (penalties, logprobs,
//! `min_p`, `repeat_penalty`) are accepted and dropped rather than rejected,
//! since a client sending them still gets a correct completion. What cannot be
//! served correctly — JSON-mode, more than one choice, a `tool_choice` that
//! insists on a call — is refused. Tool definitions are rendered into the
//! prompt and calls are parsed back out of the generation; the `reject` and
//! `strip` tools modes refuse or drop them instead, for debugging.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use super::config::{ServeSettings, ToolsMode};
use super::types::{Dialect, EngineEvent, StopKind};
use super::{
    ApiError, AppState, Completion, EngineFailure, JobRequest, SseEncoder, SseFrame, SubmitError,
    collect_completion, random_id, sse_response, submit, unix_now,
};
use crate::chat::{Message, ToolCall};
use crate::constrain::{self, Grammar};
use crate::generate::feasible_think_budget;
use crate::sampler::SamplerOptions;

/// Reasoning budgets for the effort levels, in thinking tokens. The scale is
/// this server's own: the API defines the levels, not what they cost.
const EFFORT_MINIMAL: usize = 1024;
const EFFORT_LOW: usize = 4096;
const EFFORT_MEDIUM: usize = 16384;

/// Output cap for a request that names none. This API, unlike Anthropic's, does
/// not require one, but "unspecified" is a client that did not think about the
/// length rather than one asking for the whole context window — and a reply
/// that runs to 131k tokens is minutes of generation nobody asked for. The
/// engine still clamps this to whatever the prompt leaves.
const DEFAULT_MAX_OUTPUT_TOKENS: usize = 16384;

const NO_TOOLS: &str = "tool use not yet supported by this server";
/// The debug modes answer as if tools did not exist, and a conversation that
/// already contains a call has no such rendering: dropping the calls would
/// leave the results answering nothing.
const NO_TOOL_HISTORY: &str = "tool use not yet supported by this server: a conversation containing tool calls \
     cannot be rendered";
/// Refusing beats pretending: without constrained decoding nothing here can
/// promise the model will call anything, so a request that requires a call
/// cannot be served as written.
const FORCED_TOOL_CALL: &str = "tool_choice that requires a call is not supported by this server: \
     it cannot guarantee a tool call without constrained decoding. Use \"auto\" or \"none\"";
const NO_VISION: &str = "image content is not supported by this server: the model is text-only";
/// A conversation with no turns has nothing to answer.
const EMPTY_MESSAGES: &str = "messages: at least one message is required";
/// A schema-constrained reply masks the whole answer section to the schema,
/// which leaves no way to emit a tool-call span; serving both at once would
/// silently make the tools uncallable.
const SCHEMA_WITH_TOOLS: &str = "response_format json_schema/json_object cannot be combined with tools on this \
     server: the schema constrains the entire reply, leaving no way to emit a tool call";

/// The error envelope every failure on this API wears.
pub(crate) fn error(
    status: StatusCode,
    kind: &str,
    code: Option<&str>,
    message: impl Into<String>,
) -> ApiError {
    ApiError {
        status,
        body: json!({
            "error": {
                "message": message.into(),
                "type": kind,
                "param": Value::Null,
                "code": code.map(Value::from).unwrap_or(Value::Null),
            }
        }),
        headers: Vec::new(),
    }
}

fn bad_request(message: impl Into<String>) -> ApiError {
    error(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        None,
        message,
    )
}

/// The queue-full answer. The `Retry-After` matches the Anthropic path's and
/// stays well under 60 seconds — Claude Code treats a longer one as fatal, and
/// both SDKs honour it as their backoff.
fn overloaded() -> ApiError {
    error(
        StatusCode::TOO_MANY_REQUESTS,
        "rate_limit_error",
        Some("rate_limit_exceeded"),
        "the model is busy with other requests; retry shortly",
    )
    .with_header("retry-after", "1")
}

/// An engine failure, in the envelope its cause deserves. The engine classifies
/// it; the only thing decided here is the error `code`, and the one this API
/// defines for a request fault at generation time is the context overflow.
fn engine_error(message: &str, request_fault: bool) -> ApiError {
    if request_fault {
        error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            Some("context_length_exceeded"),
            message.to_string(),
        )
    } else {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            None,
            message.to_string(),
        )
    }
}

// ---------------------------------------------------------------- request ---

#[derive(Debug, Deserialize)]
pub(crate) struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    /// The current spelling; `max_tokens` is the deprecated one and loses to it.
    pub max_completion_tokens: Option<usize>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    /// Not an OpenAI parameter, but the local-server convention, and this
    /// engine samples with it.
    pub top_k: Option<usize>,
    pub stop: Option<Stop>,
    pub seed: Option<u64>,
    pub stream: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    pub reasoning_effort: Option<String>,
    pub n: Option<u32>,
    pub tools: Option<Value>,
    /// Held as a `Value` because every shape it can take — a string, a named
    /// function object — has to reach the tools policy rather than a parse
    /// error about its fields.
    pub tool_choice: Option<Value>,
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatMessage {
    pub role: String,
    /// Null for an assistant turn that was pure tool calls.
    pub content: Option<MessageContent>,
    /// The reasoning an assistant turn did, as this API spells it on the way
    /// out. Replayed only for the turns since the last user message.
    pub reasoning_content: Option<String>,
    /// The calls an assistant turn made.
    pub tool_calls: Option<Vec<ToolCallSpec>>,
    /// Which call a `tool` message answers. The template writes no ids and
    /// pairs results with calls by position, so this is used only to put a run
    /// of results back into the order its calls were made in.
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ToolCallSpec {
    /// The id this server minted for the call, used only to pair the results
    /// that answer it.
    pub id: Option<String>,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    pub kind: Option<String>,
    pub function: Option<FunctionSpec>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct FunctionSpec {
    pub name: Option<String>,
    /// The arguments object as a JSON *string* — this API's own encoding, kept
    /// as a `Value` so that a client sending the object inline is understood
    /// rather than rejected on a type error.
    pub arguments: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
pub(crate) struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum Stop {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamOptions {
    pub include_usage: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ResponseFormat {
    #[serde(rename = "type")]
    pub kind: String,
    /// The `json_schema` payload. `name` and `strict` are accepted (clients
    /// send them) but unused: the schema is enforced by construction, which is
    /// what `strict` asks for, and the name labels nothing this server keeps.
    pub json_schema: Option<JsonSchemaFormat>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct JsonSchemaFormat {
    #[allow(dead_code)]
    pub name: Option<String>,
    pub schema: Option<Value>,
    #[allow(dead_code)]
    pub strict: Option<bool>,
}

/// A request reduced to what the engine and the response builders need.
pub(crate) struct Prepared {
    pub model: String,
    pub stream: bool,
    pub include_usage: bool,
    pub job: JobRequest,
}

fn text_of(content: &MessageContent) -> Result<String, ApiError> {
    match content {
        MessageContent::Text(text) => Ok(text.clone()),
        MessageContent::Parts(parts) => {
            let mut texts = Vec::new();
            for part in parts {
                match part.kind.as_str() {
                    "text" | "input_text" | "output_text" => {
                        let text = part.text.clone().unwrap_or_default();
                        if !text.is_empty() {
                            texts.push(text);
                        }
                    }
                    "image_url" | "input_image" => return Err(bad_request(NO_VISION)),
                    // A part type this server does not know is far more often a
                    // newer SDK's addition than content the answer needs.
                    _ => {}
                }
            }
            Ok(texts.join("\n"))
        }
    }
}

/// Join two turns' text into one block. An empty side contributes nothing, not
/// a blank line: the separator is only there to keep two texts apart, and a
/// stray newline changes the tokens the model sees.
fn join_into(previous: &mut String, text: &str) {
    if text.is_empty() {
        return;
    }
    if !previous.is_empty() {
        previous.push('\n');
    }
    previous.push_str(text);
}

/// Append a turn, merging consecutive same-role turns — the template renders
/// one block per message, and two user messages in a row mean one turn.
fn push_turn(messages: &mut Vec<Message>, turn: Message) {
    match (messages.last_mut(), turn) {
        (Some(Message::System(previous)), Message::System(text))
        | (Some(Message::User(previous)), Message::User(text)) => join_into(previous, &text),
        (
            Some(Message::Assistant {
                content: previous,
                reasoning: kept,
                tool_calls: calls,
            }),
            Message::Assistant {
                content,
                reasoning,
                tool_calls,
            },
        ) => {
            join_into(previous, &content);
            // The merged turn renders one `<think>` block and one run of calls,
            // so both messages' contributions are concatenated in order.
            match (kept.as_mut(), reasoning) {
                (Some(kept), Some(more)) => join_into(kept, &more),
                (None, reasoning) => *kept = reasoning,
                (Some(_), None) => {}
            }
            calls.extend(tool_calls);
        }
        (_, turn) => messages.push(turn),
    }
}

/// Where the trailing run of assistant and tool-result turns begins: the
/// stretch of the conversation since the last user message. Reasoning is
/// replayed only inside that run — the model needs the thinking that led to the
/// calls it is still resolving, while reasoning from turns the user has already
/// answered is stale context the template is happy to render empty.
fn trailing_run_start(messages: &[ChatMessage]) -> usize {
    let mut start = messages.len();
    while start > 0
        && matches!(
            messages[start - 1].role.as_str(),
            "assistant" | "tool" | "function"
        )
    {
        start -= 1;
    }
    start
}

/// The order to read the messages in. The template writes no ids and pairs a
/// result with a call by position, while this API pairs them by
/// `tool_call_id` — so a client that returns results in completion order rather
/// than call order needs them put back before the positions mean anything. A
/// run whose ids do not all resolve to distinct calls is left exactly as it
/// arrived: the client's own order is a better guess than a partial one.
fn result_order(messages: &[ChatMessage]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..messages.len()).collect();
    let mut index = 0;
    while index < messages.len() {
        let call_ids: Vec<&str> = messages[index]
            .tool_calls
            .iter()
            .flatten()
            .filter_map(|call| call.id.as_deref())
            .collect();
        let mut end = index + 1;
        while end < messages.len() && matches!(messages[end].role.as_str(), "tool" | "function") {
            end += 1;
        }
        if messages[index].role == "assistant" && !call_ids.is_empty() {
            reorder_results(&mut order[index + 1..end], messages, &call_ids);
        }
        index = end.max(index + 1);
    }
    order
}

/// Sort one run of tool results into the order their calls were made. Bails on
/// the first result that names no call, or names one already spoken for.
fn reorder_results(run: &mut [usize], messages: &[ChatMessage], call_ids: &[&str]) {
    let mut ranked = Vec::with_capacity(run.len());
    for &index in run.iter() {
        let Some(id) = messages[index].tool_call_id.as_deref() else {
            return;
        };
        let Some(rank) = call_ids.iter().position(|call| *call == id) else {
            return;
        };
        if ranked.iter().any(|(seen, _)| *seen == rank) {
            return;
        }
        ranked.push((rank, index));
    }
    ranked.sort_by_key(|(rank, _)| *rank);
    for (slot, (_, index)) in run.iter_mut().zip(ranked) {
        *slot = index;
    }
}

/// The arguments of one inbound call, in the client's key order.
fn tool_call_arguments(arguments: Option<&Value>) -> Result<Vec<(String, Value)>, ApiError> {
    let object = match arguments {
        None | Some(Value::Null) => return Ok(Vec::new()),
        // A call with no arguments is usually spelled with an empty string.
        Some(Value::String(text)) if text.trim().is_empty() => return Ok(Vec::new()),
        Some(Value::String(text)) => serde_json::from_str::<Value>(text).map_err(|e| {
            bad_request(format!(
                "tool_calls[].function.arguments is not valid JSON: {e}"
            ))
        })?,
        Some(other) => other.clone(),
    };
    match object {
        Value::Object(map) => Ok(map.into_iter().collect()),
        _ => Err(bad_request(
            "tool_calls[].function.arguments must be a JSON object",
        )),
    }
}

fn tool_calls_of(specs: Option<&Vec<ToolCallSpec>>) -> Result<Vec<ToolCall>, ApiError> {
    let mut calls = Vec::new();
    for spec in specs.into_iter().flatten() {
        let function = spec
            .function
            .as_ref()
            .ok_or_else(|| bad_request("tool_calls[] entries need a \"function\""))?;
        let name = match function.name.as_deref() {
            Some(name) if !name.is_empty() => name.to_string(),
            _ => return Err(bad_request("tool_calls[].function.name is required")),
        };
        calls.push(ToolCall {
            name,
            arguments: tool_call_arguments(function.arguments.as_ref())?,
        });
    }
    Ok(calls)
}

/// Turn a request's messages into the conversation the template renders.
///
/// The tools mode decides how much of a tool-using history survives: `native`
/// replays the calls and the reasoning that led to them, while the two debug
/// modes answer as if tools had never existed — which they can only do for a
/// conversation that contains none, and which means reasoning is dropped
/// everywhere, exactly as it was before this server rendered tools at all.
pub(crate) fn normalize(request: &ChatRequest, mode: ToolsMode) -> Result<Vec<Message>, ApiError> {
    let native = matches!(mode, ToolsMode::Native);
    let reasoning_from = trailing_run_start(&request.messages);
    let mut messages = Vec::new();
    for index in result_order(&request.messages) {
        let message = &request.messages[index];
        let text = match &message.content {
            Some(content) => text_of(content)?,
            None => String::new(),
        };
        let turn = match message.role.as_str() {
            "system" | "developer" => Message::System(text),
            "user" => Message::User(text),
            "assistant" => Message::Assistant {
                content: text,
                reasoning: if native && index >= reasoning_from {
                    message
                        .reasoning_content
                        .clone()
                        .filter(|reasoning| !reasoning.is_empty())
                } else {
                    None
                },
                tool_calls: match native {
                    true => tool_calls_of(message.tool_calls.as_ref())?,
                    false if message.tool_calls.iter().flatten().next().is_some() => {
                        return Err(bad_request(NO_TOOL_HISTORY));
                    }
                    false => Vec::new(),
                },
            },
            "tool" | "function" => Message::ToolResponse(text),
            role => {
                return Err(bad_request(format!(
                    "unknown message role {role:?}: expected \"system\", \"developer\", \"user\", \"assistant\" or \"tool\""
                )));
            }
        };
        push_turn(&mut messages, turn);
    }
    Ok(messages)
}

/// Map `reasoning_effort` onto this engine's thinking switch and budget.
pub(crate) fn resolve_reasoning(
    effort: Option<&str>,
    settings: &ServeSettings,
) -> Result<(bool, Option<usize>), ApiError> {
    let (enabled, budget) = match effort {
        None => (settings.thinking_force, settings.thinking_budget),
        Some("none") => (false, None),
        Some("minimal") => (true, Some(EFFORT_MINIMAL)),
        Some("low") => (true, Some(EFFORT_LOW)),
        Some("medium") => (true, Some(EFFORT_MEDIUM)),
        // The top of the scale means "think as long as you need to".
        Some("high") | Some("xhigh") | Some("max") => (true, None),
        Some(other) => {
            return Err(bad_request(format!(
                "unknown reasoning_effort {other:?}: expected \"none\", \"minimal\", \"low\", \"medium\", \"high\", \"xhigh\" or \"max\""
            )));
        }
    };
    // A budget is a property of thinking; with thinking off it must not
    // survive to arm the reasoning schedule. Armed with no `<think>` block
    // open, the schedule would bias and eventually force `</think>` into an
    // answer that never opened one — and under a grammar constraint that
    // forced token is outside the mask, poisoning the matcher. Reachable via
    // `thinking_force = false` + a configured default budget; the Anthropic
    // resolver has carried the same guard all along.
    Ok(if enabled {
        (true, budget)
    } else {
        (false, None)
    })
}

fn stop_sequences(stop: Option<&Stop>) -> Vec<String> {
    match stop {
        None => Vec::new(),
        Some(Stop::One(sequence)) => vec![sequence.clone()],
        Some(Stop::Many(sequences)) => sequences.clone(),
    }
}

/// True when the request carries tool definitions it expects to be honored. An
/// empty list is how several harnesses spell "no tools", so it is not one.
fn requests_tools(tools: Option<&Value>) -> bool {
    match tools {
        None | Some(Value::Null) => false,
        Some(Value::Array(tools)) => !tools.is_empty(),
        Some(_) => true,
    }
}

/// What a `tool_choice` asks of the model.
#[derive(Debug, Clone, PartialEq)]
enum ToolChoice {
    /// A call is permitted, not required — the only thing this server can
    /// promise, and what an absent `tool_choice` means. `allowed` is the list
    /// form's narrowing: the only functions the model may be offered.
    Auto { allowed: Option<Vec<String>> },
    /// No call: the turn is rendered as if the request carried no tools at all.
    Forbidden,
    /// A call is required. Nothing here constrains decoding, so this is a
    /// promise the server cannot keep.
    Forced,
    /// A shape this server does not recognize. Guessing which of the three it
    /// meant is how a request quietly gets answered wrong.
    Unknown,
}

/// Read `tool_choice`, in every spelling the two APIs and their SDK generations
/// use: the plain strings, the object forms Anthropic clients send, the named
/// function, and the newer `allowed_tools` list whose `mode` carries the same
/// permitted-or-required distinction.
fn tool_choice(choice: Option<&Value>) -> ToolChoice {
    let anything = ToolChoice::Auto { allowed: None };
    match choice {
        None | Some(Value::Null) => anything,
        Some(Value::String(choice)) => match choice.as_str() {
            "auto" => anything,
            "none" => ToolChoice::Forbidden,
            "required" | "any" => ToolChoice::Forced,
            _ => ToolChoice::Unknown,
        },
        Some(Value::Object(choice)) => match choice.get("type").and_then(Value::as_str) {
            Some("auto") => anything,
            Some("none") => ToolChoice::Forbidden,
            Some("function") | Some("tool") | Some("any") | Some("required") => ToolChoice::Forced,
            // The list form: the model may pick from these tools only, and the
            // mode says whether picking one is required.
            Some("allowed_tools") => match choice.get("mode").and_then(Value::as_str) {
                Some("auto") => ToolChoice::Auto {
                    allowed: Some(function_names(choice.get("tools"))),
                },
                Some("required") => ToolChoice::Forced,
                _ => ToolChoice::Unknown,
            },
            _ => ToolChoice::Unknown,
        },
        Some(_) => ToolChoice::Unknown,
    }
}

/// The function name an entry names, in either the wrapped shape this API
/// documents or the bare `{"name": …}` some clients send.
fn function_name(entry: &Value) -> Option<&str> {
    entry
        .get("function")
        .and_then(|function| function.get("name"))
        .or_else(|| entry.get("name"))
        .and_then(Value::as_str)
}

/// Every function a list of tool entries names. An entry that names none is
/// skipped: it cannot be matched against a declaration either way.
fn function_names(entries: Option<&Value>) -> Vec<String> {
    entries
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(function_name)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Apply the server's tools policy, returning the definitions the job should
/// carry: the request's own under `native`, and none under the two debug modes
/// (`reject`, which only accepts a request that asked for no call, and `strip`,
/// which answers every request as if it had sent none).
fn check_tools(request: &ChatRequest, mode: ToolsMode) -> Result<Vec<Value>, ApiError> {
    let choice = tool_choice(request.tool_choice.as_ref());
    match mode {
        ToolsMode::Reject => {
            // A choice this mode cannot read is refused with the rest: it might
            // have been the one asking for a call.
            if requests_tools(request.tools.as_ref())
                || matches!(choice, ToolChoice::Forced | ToolChoice::Unknown)
            {
                return Err(bad_request(NO_TOOLS));
            }
            Ok(Vec::new())
        }
        // Strip answers every request as if it had sent no tool parameters at
        // all, which includes never reading `tool_choice`.
        ToolsMode::Strip => Ok(Vec::new()),
        ToolsMode::Native => match choice {
            ToolChoice::Forced => Err(bad_request(FORCED_TOOL_CALL)),
            ToolChoice::Unknown => Err(bad_request(format!(
                "unknown tool_choice {}: expected \"auto\", \"none\", \"required\" or a named function",
                request.tool_choice.as_ref().unwrap_or(&Value::Null)
            ))),
            ToolChoice::Forbidden => Ok(Vec::new()),
            ToolChoice::Auto { allowed } => match request.tools.as_ref() {
                None | Some(Value::Null) => Ok(Vec::new()),
                // Already the shape the prompt renders and the engine parses
                // against, so the definitions travel verbatim — narrowed, when
                // the client narrowed them, to the ones it named. A name that
                // matches no declaration narrows to nothing, and a narrowing
                // that leaves nothing is a turn with no tools at all.
                Some(Value::Array(tools)) => Ok(match &allowed {
                    None => tools.clone(),
                    Some(allowed) => tools
                        .iter()
                        .filter(|tool| {
                            function_name(tool)
                                .is_some_and(|name| allowed.iter().any(|allowed| allowed == name))
                        })
                        .cloned()
                        .collect(),
                }),
                Some(_) => Err(bad_request(
                    "tools must be an array of function definitions",
                )),
            },
        },
    }
}

/// Resolve `response_format` into the grammar the job will decode under.
/// Compiled here, on the HTTP thread, so a schema the compiler rejects is a
/// 400 with the compiler's own message — the same before-the-queue policy as
/// every other request check.
fn resolve_response_format(
    format: Option<&ResponseFormat>,
    tools: &[Value],
) -> Result<Option<Grammar>, ApiError> {
    let Some(format) = format else {
        return Ok(None);
    };
    match format.kind.as_str() {
        "text" => Ok(None),
        kind @ ("json_object" | "json_schema") => {
            if !tools.is_empty() {
                return Err(bad_request(SCHEMA_WITH_TOOLS));
            }
            let factory = constrain::shared().map_err(|e| {
                error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    None,
                    format!("{e:#}"),
                )
            })?;
            let grammar = if kind == "json_schema" {
                let schema = format
                    .json_schema
                    .as_ref()
                    .and_then(|payload| payload.schema.as_ref())
                    .ok_or_else(|| bad_request("response_format.json_schema.schema is required"))?;
                factory.compile(schema)
            } else {
                factory.compile_any_object()
            };
            grammar.map(Some).map_err(|e| bad_request(format!("{e:#}")))
        }
        other => Err(bad_request(format!(
            "response_format {other:?} is not supported; use \"text\", \"json_object\" or \
             \"json_schema\""
        ))),
    }
}

pub(crate) fn prepare(
    request: ChatRequest,
    settings: &ServeSettings,
    default_model: &str,
) -> Result<Prepared, ApiError> {
    let tools = check_tools(&request, settings.tools_mode)?;
    if request.messages.is_empty() {
        return Err(bad_request(EMPTY_MESSAGES));
    }
    if let Some(n) = request.n {
        if n != 1 {
            return Err(bad_request(format!(
                "n = {n} is not supported: this server generates one choice per request"
            )));
        }
    }
    let grammar = resolve_response_format(request.response_format.as_ref(), &tools)?;

    // Neither cap given: a bounded default rather than the whole context.
    let max_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or_else(|| settings.context_length.min(DEFAULT_MAX_OUTPUT_TOKENS));
    if max_tokens == 0 {
        return Err(bad_request("max_completion_tokens must be at least 1"));
    }

    let messages = normalize(&request, settings.tools_mode)?;
    let (enable_thinking, budget) =
        resolve_reasoning(request.reasoning_effort.as_deref(), settings)?;
    // The ceiling has to leave the reply room to conclude in: it is lowered to
    // the largest one that fits, or dropped when none does. A `reasoning_effort`
    // this server cannot express as a ceiling still has to answer, and
    // reasoning without one always can.
    let max_think = budget.and_then(|budget| feasible_think_budget(budget, max_tokens));

    let defaults = SamplerOptions::default();
    let sampling = SamplerOptions {
        temperature: request.temperature.unwrap_or(settings.temperature),
        top_k: request.top_k.unwrap_or(settings.top_k),
        top_p: request.top_p.unwrap_or(settings.top_p),
        seed: request.seed.unwrap_or(defaults.seed),
    };

    Ok(Prepared {
        model: request
            .model
            .clone()
            .unwrap_or_else(|| default_model.to_string()),
        stream: request.stream.unwrap_or(false),
        include_usage: request
            .stream_options
            .as_ref()
            .and_then(|options| options.include_usage)
            .unwrap_or(false),
        job: JobRequest {
            messages,
            enable_thinking,
            max_think,
            max_tokens,
            sampling,
            stop_sequences: stop_sequences(request.stop.as_ref()),
            tools,
            grammar,
            continuation: None,
        },
    })
}

// --------------------------------------------------------------- response ---

/// This API has no separate reason for a stop sequence: both a natural end and
/// a matched stop sequence are "stop", and only the token cap is "length".
fn finish_reason(stop: &StopKind) -> &'static str {
    match stop {
        StopKind::EndTurn | StopKind::StopSequence(_) => "stop",
        StopKind::MaxTokens => "length",
        StopKind::ToolUse => "tool_calls",
    }
}

/// Ids for the calls in one response. They exist to pair a call with the `tool`
/// message that answers it, which is a within-conversation job, so a
/// process-wide counter is all the identity they need.
static NEXT_CALL_ID: AtomicU64 = AtomicU64::new(1);

fn call_id() -> String {
    format!("call_{:08}", NEXT_CALL_ID.fetch_add(1, Ordering::Relaxed))
}

fn usage(completion: &Completion) -> Value {
    json!({
        "prompt_tokens": completion.input_tokens,
        "completion_tokens": completion.output_tokens,
        "total_tokens": completion.input_tokens + completion.output_tokens,
        "prompt_tokens_details": {"cached_tokens": completion.cached_tokens},
        "completion_tokens_details": {"reasoning_tokens": completion.thinking_tokens},
    })
}

pub(crate) fn completion_body(
    id: &str,
    model: &str,
    created: u64,
    completion: &Completion,
) -> Value {
    // A turn that spent itself calling tools says nothing in words, and this
    // API spells that null — which is how clients tell a tool turn from a turn
    // that merely came out empty, so a wordless turn with no calls in it stays
    // the empty string.
    let content = if completion.text.is_empty() && !completion.tool_calls.is_empty() {
        Value::Null
    } else {
        Value::String(completion.text.clone())
    };
    let mut message = json!({"role": "assistant", "content": content});
    // The field is omitted rather than null when the turn did no reasoning:
    // clients that show a reasoning pane check for its presence.
    if !completion.thinking.is_empty() {
        message["reasoning_content"] = Value::String(completion.thinking.clone());
    }
    if !completion.tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(
            completion
                .tool_calls
                .iter()
                .map(|call| {
                    json!({
                        "id": call_id(),
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    })
                })
                .collect(),
        );
    }
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": Value::Null,
            "finish_reason": finish_reason(&completion.stop),
        }],
        "usage": usage(completion),
    })
}

// -------------------------------------------------------------------- SSE ---

/// Renders a generation as `chat.completion.chunk` events: a role-only opening
/// chunk, reasoning deltas, content deltas, tool-call deltas, a finish chunk,
/// an optional usage chunk, and the `[DONE]` sentinel.
pub(crate) struct ChunkStream {
    id: String,
    model: String,
    created: u64,
    include_usage: bool,
    input_tokens: usize,
    cached_tokens: usize,
    /// Which call the tool-call deltas currently belong to. Calls never
    /// interleave, so the index only ever advances, at the end of a call.
    call_index: usize,
}

impl ChunkStream {
    pub(crate) fn new(id: String, model: String, created: u64, include_usage: bool) -> Self {
        Self {
            id,
            model,
            created,
            include_usage,
            input_tokens: 0,
            cached_tokens: 0,
            call_index: 0,
        }
    }

    fn chunk(&self, choices: Value, usage: Value) -> SseFrame {
        SseFrame::unnamed(json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": choices,
            "usage": usage,
        }))
    }

    fn delta(&self, delta: Value, finish_reason: Value) -> SseFrame {
        self.chunk(
            json!([{"index": 0, "delta": delta, "logprobs": Value::Null, "finish_reason": finish_reason}]),
            Value::Null,
        )
    }

    fn done() -> SseFrame {
        SseFrame {
            name: None,
            data: "[DONE]".to_string(),
        }
    }
}

impl SseEncoder for ChunkStream {
    fn on_event(&mut self, event: EngineEvent, out: &mut VecDeque<SseFrame>) -> bool {
        match event {
            EngineEvent::Start {
                input_tokens,
                cached_tokens,
            } => {
                self.input_tokens = input_tokens;
                self.cached_tokens = cached_tokens;
                out.push_back(self.delta(json!({"role": "assistant"}), Value::Null));
                false
            }
            EngineEvent::Thinking(text) => {
                if !text.is_empty() {
                    out.push_back(self.delta(json!({"reasoning_content": text}), Value::Null));
                }
                false
            }
            EngineEvent::Text(text) => {
                if !text.is_empty() {
                    out.push_back(self.delta(json!({"content": text}), Value::Null));
                }
                false
            }
            // A call opens with its identity and an empty argument string, so a
            // client can render the call before any of its arguments arrive.
            EngineEvent::ToolCallStart { name } => {
                let call = json!({
                    "index": self.call_index,
                    "id": call_id(),
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                });
                out.push_back(self.delta(json!({"tool_calls": [call]}), Value::Null));
                false
            }
            EngineEvent::ToolCallDelta(text) => {
                if !text.is_empty() {
                    let call = json!({
                        "index": self.call_index,
                        "function": {"arguments": text},
                    });
                    out.push_back(self.delta(json!({"tool_calls": [call]}), Value::Null));
                }
                false
            }
            EngineEvent::ToolCallEnd => {
                self.call_index += 1;
                false
            }
            EngineEvent::Done {
                stop,
                output_tokens,
                thinking_tokens,
            } => {
                out.push_back(self.delta(json!({}), Value::String(finish_reason(&stop).into())));
                if self.include_usage {
                    let completion = Completion {
                        input_tokens: self.input_tokens,
                        cached_tokens: self.cached_tokens,
                        thinking: String::new(),
                        text: String::new(),
                        tool_calls: Vec::new(),
                        stop,
                        output_tokens,
                        thinking_tokens,
                    };
                    // The usage chunk carries no choices, per the spec.
                    out.push_back(self.chunk(json!([]), usage(&completion)));
                }
                out.push_back(Self::done());
                true
            }
            EngineEvent::Error {
                message,
                request_fault,
            } => {
                let (kind, code) = if request_fault {
                    (
                        "invalid_request_error",
                        Value::from("context_length_exceeded"),
                    )
                } else {
                    ("server_error", Value::Null)
                };
                out.push_back(SseFrame::unnamed(json!({
                    "error": {"message": message, "type": kind, "param": Value::Null, "code": code}
                })));
                // The sentinel still follows, so a client blocked on it does
                // not wait for a stream that has already failed.
                out.push_back(Self::done());
                true
            }
        }
    }

    fn on_hangup(&mut self, out: &mut VecDeque<SseFrame>) {
        out.push_back(SseFrame::unnamed(json!({
            "error": {
                "message": "the inference engine stopped before finishing the response",
                "type": "server_error",
                "param": Value::Null,
                "code": Value::Null,
            }
        })));
        out.push_back(Self::done());
    }
}

// ---------------------------------------------------------------- handler ---

pub(crate) async fn chat_completions(State(state): State<AppState>, body: Bytes) -> Response {
    let request: ChatRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(e) => {
            return bad_request(format!("could not parse the request body: {e}")).into_response();
        }
    };
    let prepared = match prepare(request, &state.settings, &state.model_id) {
        Ok(prepared) => prepared,
        Err(e) => return e.into_response(),
    };
    let Prepared {
        model,
        stream,
        include_usage,
        job,
    } = prepared;
    let id = random_id("chatcmpl-");
    let created = unix_now();

    let (mut events, guard) = match submit(&state, job, Dialect::OpenAi, stream) {
        Ok(submitted) => submitted,
        Err(SubmitError::Invalid(message)) => return bad_request(message).into_response(),
        Err(SubmitError::Overloaded) => return overloaded().into_response(),
        Err(SubmitError::EngineGone) => {
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                None,
                "the inference engine is not running",
            )
            .into_response();
        }
    };

    if stream {
        return sse_response(
            events,
            guard,
            ChunkStream::new(id, model, created, include_usage),
        );
    }
    // Held across the drain: hyper dropping this future — the client hung up —
    // drops the guard, which cancels the generation with its reason stated.
    let _guard = guard;
    match collect_completion(&mut events).await {
        Ok(completion) => {
            axum::Json(completion_body(&id, &model, created, &completion)).into_response()
        }
        Err(EngineFailure::Reported {
            message,
            request_fault,
        }) => engine_error(&message, request_fault).into_response(),
        Err(EngineFailure::Hangup) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            None,
            "the inference engine stopped before finishing the response",
        )
        .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::CompletedToolCall;
    use crate::serve::testutil::{encode_all, payload, settings, shape};

    fn parse(body: &str) -> ChatRequest {
        serde_json::from_str(body).expect("request parses")
    }

    fn prepared(body: &str) -> Prepared {
        prepare(parse(body), &settings(), "laguna-s-2.1").expect("request prepares")
    }

    fn rejected(body: &str) -> ApiError {
        prepare(parse(body), &settings(), "laguna-s-2.1")
            .err()
            .expect("request is rejected")
    }

    fn message(error: &ApiError) -> String {
        error.body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .to_string()
    }

    /// The tools policies, named so the tool tests state which one they are
    /// about instead of leaning on whichever is currently the default.
    fn tools(mode: ToolsMode) -> ServeSettings {
        ServeSettings {
            tools_mode: mode,
            ..settings()
        }
    }

    /// Prepare under an explicit tools policy.
    fn prepared_with(mode: ToolsMode, body: &str) -> Prepared {
        prepare(parse(body), &tools(mode), "laguna-s-2.1").expect("request prepares")
    }

    fn rejected_with(mode: ToolsMode, body: &str) -> ApiError {
        prepare(parse(body), &tools(mode), "laguna-s-2.1")
            .err()
            .unwrap_or_else(|| panic!("request is rejected: {body}"))
    }

    fn calls_of(messages: &[Message], index: usize) -> Vec<crate::chat::ToolCall> {
        match &messages[index] {
            Message::Assistant { tool_calls, .. } => tool_calls.clone(),
            other => panic!("message {index} is not an assistant turn: {other:?}"),
        }
    }

    const USER: &str = r#""messages":[{"role":"user","content":"Hi"}]"#;

    /// A full queue answers 429 with a short Retry-After, which openai-node
    /// honours as its backoff; kept under 60 seconds because Claude Code
    /// treats a longer one as fatal.
    #[test]
    fn the_queue_full_answer_is_a_retryable_429() {
        let error = overloaded();
        assert_eq!(error.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(error.body["error"]["type"], "rate_limit_error");
        assert_eq!(error.body["error"]["code"], "rate_limit_exceeded");
        assert_eq!(error.headers, vec![("retry-after", "1".to_string())]);
    }

    #[test]
    fn the_current_token_cap_wins_over_the_deprecated_one() {
        let both = prepared(&format!(
            r#"{{"max_completion_tokens":64,"max_tokens":16,{USER}}}"#
        ));
        assert_eq!(both.job.max_tokens, 64);

        let legacy = prepared(&format!(r#"{{"max_tokens":16,{USER}}}"#));
        assert_eq!(legacy.job.max_tokens, 16);

        // Neither: a bounded default, not the whole context window. A client
        // that named no cap did not ask for a 131k-token reply.
        let neither = prepared(&format!(r#"{{{USER}}}"#));
        assert_eq!(neither.job.max_tokens, DEFAULT_MAX_OUTPUT_TOKENS);
        assert!(DEFAULT_MAX_OUTPUT_TOKENS < settings().context_length);
    }

    /// A context window smaller than the default cap bounds it further; the
    /// engine still clamps to what the prompt actually leaves.
    #[test]
    fn the_default_cap_never_exceeds_the_context() {
        let mut settings = settings();
        settings.context_length = 4096;
        let request = prepare(parse(&format!(r#"{{{USER}}}"#)), &settings, "m").unwrap();
        assert_eq!(request.job.max_tokens, 4096);
    }

    /// With thinking off by default (`thinking_force = false`), a configured
    /// default budget must not survive into the job: an armed reasoning
    /// schedule with no `<think>` block open would force `</think>` into the
    /// answer — and under a grammar constraint, poison the matcher with a
    /// token outside the mask.
    #[test]
    fn a_disabled_thinking_default_carries_no_budget() {
        let mut settings = settings();
        settings.thinking_force = false;
        settings.thinking_budget = Some(4096);
        let request = prepare(parse(&format!(r#"{{{USER}}}"#)), &settings, "m").unwrap();
        assert!(!request.job.enable_thinking);
        assert_eq!(request.job.max_think, None);
    }

    #[test]
    fn an_empty_conversation_is_rejected() {
        let error = rejected(r#"{"messages":[]}"#);
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.body["error"]["type"], "invalid_request_error");
        assert!(
            message(&error).contains("at least one message"),
            "{}",
            message(&error)
        );
    }

    #[test]
    fn reasoning_effort_maps_onto_thinking_budgets() {
        let settings = settings();
        assert_eq!(
            resolve_reasoning(Some("none"), &settings).unwrap(),
            (false, None)
        );
        assert_eq!(
            resolve_reasoning(Some("minimal"), &settings).unwrap(),
            (true, Some(1024))
        );
        assert_eq!(
            resolve_reasoning(Some("low"), &settings).unwrap(),
            (true, Some(4096))
        );
        assert_eq!(
            resolve_reasoning(Some("medium"), &settings).unwrap(),
            (true, Some(16384))
        );
        for uncapped in ["high", "xhigh", "max"] {
            assert_eq!(
                resolve_reasoning(Some(uncapped), &settings).unwrap(),
                (true, None)
            );
        }
        // Absent: the server's configuration decides.
        assert_eq!(resolve_reasoning(None, &settings).unwrap(), (true, None));
        let mut off = settings.clone();
        off.thinking_force = false;
        assert_eq!(resolve_reasoning(None, &off).unwrap(), (false, None));

        let error = resolve_reasoning(Some("gigantic"), &settings).unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(
            message(&error).contains("reasoning_effort"),
            "{}",
            message(&error)
        );
    }

    /// An effort level whose budget does not fit the reply is lowered to one
    /// that does, or dropped for uncapped reasoning — the request still has to
    /// generate, since nothing about it was invalid.
    #[test]
    fn a_reasoning_budget_is_degraded_to_fit_the_reply() {
        // Room for the level's own budget: honored exactly.
        let request = prepared(&format!(
            r#"{{"max_completion_tokens":32768,"reasoning_effort":"medium",{USER}}}"#
        ));
        assert_eq!(request.job.max_think, Some(EFFORT_MEDIUM));

        // Not enough room: lowered, but still a ceiling.
        let request = prepared(&format!(
            r#"{{"max_completion_tokens":4096,"reasoning_effort":"medium",{USER}}}"#
        ));
        let lowered = request
            .job
            .max_think
            .expect("a 4096-token reply has room for some ceiling");
        assert!(lowered < EFFORT_MEDIUM, "{lowered}");
        assert_eq!(feasible_think_budget(EFFORT_MEDIUM, 4096), Some(lowered));

        // A reply too short for any expressible ceiling reasons uncapped rather
        // than failing at decode time.
        let request = prepared(&format!(
            r#"{{"max_completion_tokens":100,"reasoning_effort":"minimal",{USER}}}"#
        ));
        assert!(request.job.enable_thinking);
        assert_eq!(request.job.max_think, None);
    }

    #[test]
    fn anything_but_exactly_one_choice_is_rejected() {
        let error = rejected(&format!(r#"{{"n":2,{USER}}}"#));
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(
            message(&error).contains("one choice"),
            "{}",
            message(&error)
        );
        // Zero choices is no more servable than several.
        let error = rejected(&format!(r#"{{"n":0,{USER}}}"#));
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(
            prepared(&format!(r#"{{"n":1,{USER}}}"#)).job.max_tokens,
            DEFAULT_MAX_OUTPUT_TOKENS
        );
    }

    #[test]
    fn tool_definitions_are_rejected_but_an_empty_list_is_not() {
        let rejecting = tools(ToolsMode::Reject);
        let reject = |body: &str| {
            prepare(parse(body), &rejecting, "laguna-s-2.1")
                .err()
                .unwrap_or_else(|| panic!("reject mode refuses {body}"))
        };

        let error = reject(&format!(
            r#"{{"tools":[{{"type":"function","function":{{"name":"f"}}}}],{USER}}}"#
        ));
        assert!(message(&error).contains("tool use"), "{}", message(&error));
        // A `tool_choice` that insists on a call is a tools request too: a reply
        // with no call in it is not what was asked for.
        let error = reject(&format!(r#"{{"tool_choice":"required",{USER}}}"#));
        assert!(message(&error).contains("tool use"), "{}", message(&error));

        // The forms that leave the model free not to call anything pass, since
        // not calling anything is exactly what it does.
        for body in [
            format!(r#"{{"tools":[],{USER}}}"#),
            format!(r#"{{"tool_choice":"auto",{USER}}}"#),
            format!(r#"{{"tool_choice":"none",{USER}}}"#),
        ] {
            let request = prepare(parse(&body), &rejecting, "laguna-s-2.1")
                .unwrap_or_else(|_| panic!("reject mode accepts {body}"));
            assert_eq!(shape(&request.job.messages), vec!["user:Hi"]);
            assert!(request.job.tools.is_empty());
        }
    }

    /// Strip mode answers a tools-bearing request exactly as if the tool
    /// parameters had never been sent: same conversation, same generation
    /// settings. The model simply never calls a tool.
    #[test]
    fn strip_mode_answers_a_tools_request_as_if_it_had_none() {
        let stripping = tools(ToolsMode::Strip);
        let stripped = prepare(
            parse(&format!(
                r#"{{"tools":[{{"type":"function","function":{{"name":"f"}}}}],
                    "tool_choice":"required",{USER}}}"#
            )),
            &stripping,
            "laguna-s-2.1",
        )
        .expect("strip mode accepts tool definitions");
        let toolless = prepare(parse(&format!(r#"{{{USER}}}"#)), &stripping, "laguna-s-2.1")
            .expect("a request with no tools prepares");

        assert_eq!(shape(&stripped.job.messages), vec!["user:Hi"]);
        assert_eq!(shape(&stripped.job.messages), shape(&toolless.job.messages));
        assert_eq!(stripped.job.max_tokens, toolless.job.max_tokens);
        assert_eq!(stripped.job.enable_thinking, toolless.job.enable_thinking);
        assert_eq!(stripped.job.max_think, toolless.job.max_think);
        assert_eq!(stripped.job.stop_sequences, toolless.job.stop_sequences);
        assert_eq!(
            stripped.job.sampling.temperature,
            toolless.job.sampling.temperature
        );
        assert_eq!(stripped.job.sampling.top_k, toolless.job.sampling.top_k);
        assert_eq!(stripped.job.sampling.top_p, toolless.job.sampling.top_p);
        assert_eq!(stripped.job.sampling.seed, toolless.job.sampling.seed);
        assert_eq!(stripped.model, toolless.model);
        assert!(stripped.job.tools.is_empty());
    }

    /// A conversation the debug modes can still render byte-for-byte the way
    /// they did before this server rendered tools at all: no calls in it, and
    /// the reasoning the client echoed back dropped from every turn.
    #[test]
    fn the_debug_modes_render_a_conversation_as_they_did_before_tools() {
        const REPLAYED: &str = r#"{"messages":[
            {"role":"user","content":"first"},
            {"role":"assistant","content":"answered","reasoning_content":"old"},
            {"role":"user","content":"second"},
            {"role":"assistant","content":"again","reasoning_content":"fresh"},
            {"role":"tool","content":"result"}]}"#;
        // The same conversation as a client that never sent reasoning at all.
        const PLAIN: &str = r#"{"messages":[
            {"role":"user","content":"first"},
            {"role":"assistant","content":"answered"},
            {"role":"user","content":"second"},
            {"role":"assistant","content":"again"},
            {"role":"tool","content":"result"}]}"#;

        for mode in [ToolsMode::Reject, ToolsMode::Strip] {
            let replayed = prepared_with(mode, REPLAYED);
            let plain = prepared_with(mode, PLAIN);
            assert_eq!(shape(&replayed.job.messages), shape(&plain.job.messages));
            assert!(
                !shape(&replayed.job.messages)
                    .iter()
                    .any(|turn| turn.contains("think:")),
                "{:?}",
                shape(&replayed.job.messages)
            );
        }

        // Native is the mode that replays the trailing run's reasoning.
        let native = prepared_with(ToolsMode::Native, REPLAYED);
        assert!(shape(&native.job.messages).contains(&"assistant:again|think:fresh".to_string()));
    }

    /// A history that already contains a call cannot be answered as if tools did
    /// not exist: dropping the calls would leave the results answering nothing.
    #[test]
    fn the_debug_modes_refuse_a_conversation_that_already_called_a_tool() {
        const CALLED: &str = r#"{"messages":[
            {"role":"user","content":"weather?"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"sunny"}]}"#;
        for mode in [ToolsMode::Reject, ToolsMode::Strip] {
            let error = rejected_with(mode, CALLED);
            assert_eq!(error.status, StatusCode::BAD_REQUEST);
            assert!(
                message(&error).contains("tool calls"),
                "{}",
                message(&error)
            );
        }
        // Native renders it.
        let native = prepared_with(ToolsMode::Native, CALLED);
        assert_eq!(calls_of(&native.job.messages, 1).len(), 1);
    }

    const WEATHER: &str = r#"{"type":"function","function":{"name":"get_weather",
        "description":"Look up the weather","parameters":{"type":"object",
        "properties":{"city":{"type":"string"}},"required":["city"]}}}"#;
    const CLOCK: &str = r#"{"type":"function","function":{"name":"now",
        "parameters":{"type":"object","properties":{}}}}"#;

    /// Native mode is the one that actually serves tools: the definitions reach
    /// the job untouched, since the prompt renders and the engine parses
    /// against this exact shape.
    #[test]
    fn native_mode_passes_the_tool_definitions_through_verbatim() {
        let request = prepared_with(
            ToolsMode::Native,
            &format!(r#"{{"tools":[{WEATHER}],"parallel_tool_calls":true,{USER}}}"#),
        );
        assert_eq!(request.job.tools.len(), 1);
        assert_eq!(
            request.job.tools[0],
            serde_json::from_str::<Value>(WEATHER).unwrap()
        );

        // An empty list is how several harnesses spell "no tools".
        let empty = prepared_with(ToolsMode::Native, &format!(r#"{{"tools":[],{USER}}}"#));
        assert!(empty.job.tools.is_empty());
    }

    /// `tool_choice` says whether a call is permitted, required, or forbidden.
    /// Only the first two can be honored: nothing here constrains decoding, so a
    /// required call would be a promise this server cannot keep.
    #[test]
    fn tool_choice_none_drops_the_tools_and_a_required_call_is_refused() {
        let allowed = prepared_with(
            ToolsMode::Native,
            &format!(r#"{{"tools":[{WEATHER}],"tool_choice":"auto",{USER}}}"#),
        );
        assert_eq!(allowed.job.tools.len(), 1);

        let forbidden = prepared_with(
            ToolsMode::Native,
            &format!(r#"{{"tools":[{WEATHER}],"tool_choice":"none",{USER}}}"#),
        );
        assert!(forbidden.job.tools.is_empty());
        assert_eq!(shape(&forbidden.job.messages), vec!["user:Hi"]);

        for choice in [
            r#""required""#,
            r#"{"type":"function","function":{"name":"get_weather"}}"#,
        ] {
            let error = rejected_with(
                ToolsMode::Native,
                &format!(r#"{{"tools":[{WEATHER}],"tool_choice":{choice},{USER}}}"#),
            );
            assert_eq!(error.status, StatusCode::BAD_REQUEST);
            assert!(
                message(&error).contains("tool_choice"),
                "{}",
                message(&error)
            );
        }
    }

    /// The object spellings: the ones Anthropic's clients use, the named
    /// function, and the newer `allowed_tools` list whose `mode` says whether a
    /// call is required. Anything unreadable is refused rather than guessed at.
    #[test]
    fn tool_choice_is_read_in_every_spelling() {
        let permitted = |choice: &str| {
            prepared_with(
                ToolsMode::Native,
                &format!(r#"{{"tools":[{WEATHER}],"tool_choice":{choice},{USER}}}"#),
            )
            .job
            .tools
            .len()
        };
        for choice in [
            r#""auto""#,
            r#"{"type":"auto"}"#,
            r#"{"type":"allowed_tools","mode":"auto","tools":[{"type":"function","function":{"name":"get_weather"}}]}"#,
        ] {
            assert_eq!(permitted(choice), 1, "{choice}");
        }
        for choice in [r#""none""#, r#"{"type":"none"}"#] {
            assert_eq!(permitted(choice), 0, "{choice}");
        }

        // Required in any spelling: refused for the same reason.
        for choice in [
            r#""required""#,
            r#"{"type":"function","function":{"name":"get_weather"}}"#,
            r#"{"type":"allowed_tools","mode":"required","tools":[]}"#,
        ] {
            let error = rejected_with(
                ToolsMode::Native,
                &format!(r#"{{"tools":[{WEATHER}],"tool_choice":{choice},{USER}}}"#),
            );
            assert!(
                message(&error).contains("cannot guarantee"),
                "{choice}: {}",
                message(&error)
            );
        }

        // Unreadable: named as unknown rather than silently treated as auto.
        for choice in [r#""banana""#, r#"{"type":"banana"}"#, "7"] {
            let error = rejected_with(
                ToolsMode::Native,
                &format!(r#"{{"tools":[{WEATHER}],"tool_choice":{choice},{USER}}}"#),
            );
            assert!(
                message(&error).contains("unknown tool_choice"),
                "{choice}: {}",
                message(&error)
            );
        }
    }

    /// `allowed_tools` narrows which functions the model may call. The prompt
    /// is the only place that narrowing can happen here — a tool the model is
    /// never shown is one it cannot call — so the job carries the intersection,
    /// and a narrowing that leaves nothing behind carries no tools at all.
    #[test]
    fn allowed_tools_narrows_the_declarations_to_the_named_functions() {
        let names = |tools: &[Value]| -> Vec<String> {
            tools
                .iter()
                .map(|tool| tool["function"]["name"].as_str().unwrap().to_string())
                .collect()
        };
        let narrowed = |allowed: &str| {
            prepared_with(
                ToolsMode::Native,
                &format!(
                    r#"{{"tools":[{WEATHER},{CLOCK}],
                        "tool_choice":{{"type":"allowed_tools","mode":"auto","tools":[{allowed}]}},
                        {USER}}}"#
                ),
            )
            .job
            .tools
        };

        assert_eq!(
            names(&narrowed(
                r#"{"type":"function","function":{"name":"now"}}"#
            )),
            vec!["now"]
        );
        // The bare spelling names a function just as unambiguously.
        assert_eq!(
            names(&narrowed(r#"{"name":"get_weather"}"#)),
            vec!["get_weather"]
        );
        assert_eq!(
            names(&narrowed(
                r#"{"name":"now"},{"type":"function","function":{"name":"get_weather"}}"#
            ))
            .len(),
            2
        );
        // A name no declaration answers to narrows to nothing that exists.
        assert!(narrowed(r#"{"name":"nonexistent"}"#).is_empty());
        assert!(narrowed("").is_empty());
    }

    /// This API pairs a result with its call by id; the template pairs them by
    /// position. A client that answers the calls in whatever order they
    /// completed gets them put back into call order.
    #[test]
    fn tool_results_are_reordered_onto_their_calls() {
        const CALLS: &str = r#"
            {"role":"user","content":"weather?"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"a","type":"function","function":{"name":"first","arguments":"{}"}},
                {"id":"b","type":"function","function":{"name":"second","arguments":"{}"}}]}"#;

        // Answered out of order: the ids resolve, so the run is sorted.
        let sorted = prepared_with(
            ToolsMode::Native,
            &format!(
                r#"{{"messages":[{CALLS},
                    {{"role":"tool","tool_call_id":"b","content":"B"}},
                    {{"role":"tool","tool_call_id":"a","content":"A"}}]}}"#
            ),
        );
        assert_eq!(
            shape(&sorted.job.messages),
            vec!["user:weather?", "assistant:", "tool:A", "tool:B"]
        );

        // An id that names no call, and a result carrying no id at all: the
        // client's own order is kept rather than a partial reordering.
        for run in [
            r#"{"role":"tool","tool_call_id":"b","content":"B"},
               {"role":"tool","tool_call_id":"zzz","content":"A"}"#,
            r#"{"role":"tool","tool_call_id":"b","content":"B"},
               {"role":"tool","content":"A"}"#,
            // The same call answered twice resolves, but not to distinct calls.
            r#"{"role":"tool","tool_call_id":"b","content":"B"},
               {"role":"tool","tool_call_id":"b","content":"A"}"#,
        ] {
            let kept = prepared_with(
                ToolsMode::Native,
                &format!(r#"{{"messages":[{CALLS},{run}]}}"#),
            );
            assert_eq!(
                shape(&kept.job.messages),
                vec!["user:weather?", "assistant:", "tool:B", "tool:A"],
                "{run}"
            );
        }
    }

    /// Merging two turns of the same role is a join, not a concatenation with a
    /// blank line: an empty message contributes nothing, since a stray newline
    /// would change the tokens the model sees.
    #[test]
    fn merging_same_role_turns_never_introduces_a_blank_line() {
        let request = prepared_with(
            ToolsMode::Native,
            r#"{"messages":[
                {"role":"user","content":"a"},
                {"role":"user","content":""},
                {"role":"user","content":"b"},
                {"role":"assistant","content":"","reasoning_content":"why"},
                {"role":"assistant","content":"answer","reasoning_content":"more"}]}"#,
        );
        assert_eq!(
            shape(&request.job.messages),
            vec!["user:a\nb", "assistant:answer|think:why\nmore"]
        );
    }

    /// The calls an assistant turn made are replayed into the conversation with
    /// their arguments in the client's key order and their JSON types intact —
    /// the template renders strings raw and everything else as JSON.
    #[test]
    fn assistant_tool_calls_are_replayed_in_order() {
        let request = prepared_with(
            ToolsMode::Native,
            r#"{"messages":[
                {"role":"user","content":"weather?"},
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"call_00000001","type":"function","function":{
                        "name":"get_weather",
                        "arguments":"{\"city\":\"Oslo\",\"days\":3,\"units\":{\"temp\":\"C\"}}"}},
                    {"id":"call_00000002","type":"function","function":{
                        "name":"now","arguments":""}}]},
                {"role":"tool","tool_call_id":"call_00000001","content":"sunny"},
                {"role":"tool","tool_call_id":"call_00000002","content":"14:00"}]}"#,
        );
        assert_eq!(
            shape(&request.job.messages),
            vec!["user:weather?", "assistant:", "tool:sunny", "tool:14:00"]
        );

        let calls = calls_of(&request.job.messages, 1);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].arguments,
            vec![
                ("city".to_string(), json!("Oslo")),
                ("days".to_string(), json!(3)),
                ("units".to_string(), json!({"temp": "C"})),
            ]
        );
        // An empty argument string is a call with no arguments, not a fault.
        assert_eq!(calls[1].name, "now");
        assert!(calls[1].arguments.is_empty());
    }

    /// The arguments arrive as a JSON *string*, so the client, not the model, is
    /// at fault when they do not parse into an object.
    #[test]
    fn tool_call_arguments_that_are_not_a_json_object_are_a_request_fault() {
        for arguments in [r#""{oops""#, r#""[1,2]""#, r#""\"text\"""#, "42"] {
            let error = rejected_with(
                ToolsMode::Native,
                &format!(
                    r#"{{"messages":[{{"role":"assistant","tool_calls":[
                        {{"function":{{"name":"f","arguments":{arguments}}}}}]}}]}}"#
                ),
            );
            assert_eq!(error.status, StatusCode::BAD_REQUEST);
            assert!(
                message(&error).contains("arguments"),
                "{arguments}: {}",
                message(&error)
            );
        }

        let unnamed = rejected_with(
            ToolsMode::Native,
            r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"arguments":"{}"}}]}]}"#,
        );
        assert!(message(&unnamed).contains("name"), "{}", message(&unnamed));

        // The object spelled inline is not this API's encoding, but it is
        // unambiguous, so it is understood rather than refused.
        let inline = prepared_with(
            ToolsMode::Native,
            r#"{"messages":[{"role":"assistant","tool_calls":[
                {"function":{"name":"f","arguments":{"city":"Oslo"}}}]}]}"#,
        );
        assert_eq!(
            calls_of(&inline.job.messages, 0)[0].arguments,
            vec![("city".to_string(), json!("Oslo"))]
        );
    }

    /// Reasoning is replayed for the turns since the last user message — the
    /// thinking behind the calls still being resolved — and dropped for
    /// everything the user has already answered.
    #[test]
    fn reasoning_is_replayed_only_for_the_turns_since_the_last_user_message() {
        let request = prepared_with(
            ToolsMode::Native,
            r#"{"messages":[
                {"role":"user","content":"first"},
                {"role":"assistant","content":"old","reasoning_content":"stale"},
                {"role":"user","content":"second"},
                {"role":"assistant","content":null,"reasoning_content":"fresh",
                 "tool_calls":[{"function":{"name":"f","arguments":"{}"}}]},
                {"role":"tool","content":"result"}]}"#,
        );
        assert_eq!(
            shape(&request.job.messages),
            vec![
                "user:first",
                "assistant:old",
                "user:second",
                "assistant:|think:fresh",
                "tool:result"
            ]
        );
    }

    /// A conversation that never had a user message is all trailing run: there
    /// is no earlier turn for its reasoning to be stale relative to.
    #[test]
    fn reasoning_survives_a_conversation_with_no_user_message() {
        let request = prepared_with(
            ToolsMode::Native,
            r#"{"messages":[{"role":"assistant","content":"a","reasoning_content":"why"}]}"#,
        );
        assert_eq!(shape(&request.job.messages), vec!["assistant:a|think:why"]);
    }

    #[test]
    fn response_format_compiles_into_a_grammar() {
        // Plain text and an absent format constrain nothing.
        let text = prepared(&format!(
            r#"{{"response_format":{{"type":"text"}},{USER}}}"#
        ));
        assert!(text.job.grammar.is_none());
        assert!(prepared(&format!(r#"{{{USER}}}"#)).job.grammar.is_none());
        // json_object needs no schema; json_schema carries one.
        let object = prepared(&format!(
            r#"{{"response_format":{{"type":"json_object"}},{USER}}}"#
        ));
        assert!(object.job.grammar.is_some());
        let schema = prepared(&format!(
            r#"{{"response_format":{{"type":"json_schema","json_schema":{{"name":"x","strict":true,
                "schema":{{"type":"object","properties":{{"a":{{"type":"string"}}}},
                "required":["a"],"additionalProperties":false}}}}}},{USER}}}"#
        ));
        assert!(schema.job.grammar.is_some());
    }

    #[test]
    fn bad_response_formats_are_rejected() {
        // A json_schema request without the schema itself.
        let error = rejected(&format!(
            r#"{{"response_format":{{"type":"json_schema"}},{USER}}}"#
        ));
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(
            message(&error).contains("schema is required"),
            "{}",
            message(&error)
        );
        // A schema the compiler cannot enforce, refused with its message.
        let error = rejected(&format!(
            r#"{{"response_format":{{"type":"json_schema","json_schema":{{"schema":
                {{"not":{{"type":"string"}}}}}}}},{USER}}}"#
        ));
        assert!(
            message(&error).contains("json_schema rejected"),
            "{}",
            message(&error)
        );
        // An unrecognized type.
        let error = rejected(&format!(
            r#"{{"response_format":{{"type":"yaml"}},{USER}}}"#
        ));
        assert!(
            message(&error).contains("not supported"),
            "{}",
            message(&error)
        );
    }

    #[test]
    fn response_format_with_tools_is_rejected() {
        let body = format!(
            r#"{{"response_format":{{"type":"json_object"}},
                "tools":[{{"type":"function","function":{{"name":"f","parameters":{{}}}}}}],
                {USER}}}"#
        );
        let error = prepare(parse(&body), &tools(ToolsMode::Native), "m")
            .err()
            .expect("schema + tools must be refused");
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(
            message(&error).contains("cannot be combined"),
            "{}",
            message(&error)
        );
        // With tools stripped by policy, the same request is servable: nothing
        // reaches the prompt for the schema to conflict with.
        let stripped = prepare(parse(&body), &tools(ToolsMode::Strip), "m")
            .expect("stripped tools leave the schema free to apply");
        assert!(stripped.job.grammar.is_some());
    }

    #[test]
    fn stop_accepts_a_string_or_a_list() {
        let one = prepared(&format!(r#"{{"stop":"END",{USER}}}"#));
        assert_eq!(one.job.stop_sequences, vec!["END".to_string()]);
        let many = prepared(&format!(r#"{{"stop":["A","B"],{USER}}}"#));
        assert_eq!(
            many.job.stop_sequences,
            vec!["A".to_string(), "B".to_string()]
        );
        let none = prepared(&format!(r#"{{{USER}}}"#));
        assert!(none.job.stop_sequences.is_empty());
    }

    #[test]
    fn roles_map_onto_the_templates_turns() {
        let request = prepared(
            r#"{"messages":[
                {"role":"system","content":"Be brief."},
                {"role":"user","content":"weather?"},
                {"role":"assistant","content":null,"reasoning_content":"dropped"},
                {"role":"tool","tool_call_id":"t1","content":"sunny"},
                {"role":"user","content":[{"type":"text","text":"thanks"}]}]}"#,
        );
        assert_eq!(
            shape(&request.job.messages),
            vec![
                "system:Be brief.",
                "user:weather?",
                "assistant:",
                "tool:sunny",
                "user:thanks"
            ]
        );
    }

    #[test]
    fn developer_messages_are_system_messages() {
        let request = prepared(r#"{"messages":[{"role":"developer","content":"Be brief."}]}"#);
        assert_eq!(shape(&request.job.messages), vec!["system:Be brief."]);
    }

    #[test]
    fn image_parts_are_rejected_with_a_reason() {
        let error = rejected(
            r#"{"messages":[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":"data:image/png;base64,x"}}]}]}"#,
        );
        assert!(message(&error).contains("text-only"), "{}", message(&error));
    }

    #[test]
    fn unsupported_knobs_are_accepted_and_dropped() {
        let request = prepared(&format!(
            r#"{{"presence_penalty":0.5,"frequency_penalty":0.5,"logprobs":true,
                 "min_p":0.05,"repeat_penalty":1.1,"user":"someone",{USER}}}"#
        ));
        assert_eq!(shape(&request.job.messages), vec!["user:Hi"]);
    }

    #[test]
    fn sampling_takes_the_request_then_the_server_defaults() {
        let defaults = prepared(&format!(r#"{{{USER}}}"#));
        assert_eq!(defaults.job.sampling.temperature, settings().temperature);
        assert_eq!(defaults.job.sampling.top_p, settings().top_p);
        assert_eq!(defaults.job.sampling.top_k, settings().top_k);
        assert_eq!(defaults.job.sampling.seed, SamplerOptions::default().seed);

        let asked = prepared(&format!(
            r#"{{"temperature":0.3,"top_p":0.7,"top_k":8,"seed":1234,{USER}}}"#
        ));
        assert_eq!(asked.job.sampling.temperature, 0.3);
        assert_eq!(asked.job.sampling.top_p, 0.7);
        assert_eq!(asked.job.sampling.top_k, 8);
        assert_eq!(asked.job.sampling.seed, 1234);
    }

    #[test]
    fn the_requests_model_string_is_echoed_verbatim() {
        assert_eq!(
            prepared(&format!(r#"{{"model":"gpt-4o",{USER}}}"#)).model,
            "gpt-4o"
        );
        assert_eq!(prepared(&format!(r#"{{{USER}}}"#)).model, "laguna-s-2.1");
    }

    #[test]
    fn include_usage_is_read_from_stream_options() {
        assert!(!prepared(&format!(r#"{{"stream":true,{USER}}}"#)).include_usage);
        let asked = prepared(&format!(
            r#"{{"stream":true,"stream_options":{{"include_usage":true}},{USER}}}"#
        ));
        assert!(asked.stream && asked.include_usage);
    }

    fn completion(thinking: &str, text: &str, stop: StopKind) -> Completion {
        Completion {
            input_tokens: 100,
            cached_tokens: 40,
            thinking: thinking.to_string(),
            text: text.to_string(),
            tool_calls: Vec::new(),
            stop,
            output_tokens: 25,
            thinking_tokens: 10,
        }
    }

    fn completed_call(name: &str, arguments: &str) -> CompletedToolCall {
        CompletedToolCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
        }
    }

    /// The `call_NNNNNNNN` shape, without pinning the counter: it is
    /// process-wide, so its value depends on what else ran first.
    fn assert_call_id(id: &Value) {
        let id = id.as_str().unwrap_or_else(|| panic!("{id:?} is a string"));
        let digits = id.strip_prefix("call_").unwrap_or_else(|| panic!("{id}"));
        assert_eq!(digits.len(), 8, "{id}");
        assert!(digits.chars().all(|c| c.is_ascii_digit()), "{id}");
    }

    #[test]
    fn the_response_body_has_the_documented_field_names() {
        let body = completion_body(
            "chatcmpl-1",
            "gpt-4o",
            1_700_000_000,
            &completion("why", "answer", StopKind::EndTurn),
        );
        assert_eq!(body["id"], "chatcmpl-1");
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["created"], 1_700_000_000u64);
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["choices"][0]["index"], 0);
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert_eq!(body["choices"][0]["message"]["content"], "answer");
        assert_eq!(body["choices"][0]["message"]["reasoning_content"], "why");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["choices"][0]["logprobs"], Value::Null);
        assert_eq!(body["usage"]["prompt_tokens"], 100);
        assert_eq!(body["usage"]["completion_tokens"], 25);
        assert_eq!(body["usage"]["total_tokens"], 125);
        assert_eq!(body["usage"]["prompt_tokens_details"]["cached_tokens"], 40);
        assert_eq!(
            body["usage"]["completion_tokens_details"]["reasoning_tokens"],
            10
        );
    }

    #[test]
    fn reasoning_content_is_absent_rather_than_null_when_there_was_none() {
        let body = completion_body("id", "m", 0, &completion("", "answer", StopKind::EndTurn));
        assert!(
            body["choices"][0]["message"]
                .get("reasoning_content")
                .is_none()
        );
    }

    /// A turn that called tools carries them on the message, with the arguments
    /// as the JSON string this API encodes them in, and finishes for the reason
    /// the client dispatches on.
    #[test]
    fn the_response_body_carries_tool_calls() {
        let mut completion = completion("why", "", StopKind::ToolUse);
        completion.tool_calls = vec![
            completed_call("get_weather", r#"{"city":"Oslo"}"#),
            completed_call("now", "{}"),
        ];
        let body = completion_body("chatcmpl-1", "m", 0, &completion);
        let message = &body["choices"][0]["message"];

        // No words were spoken this turn because it was spent calling tools,
        // which this API spells as null rather than an empty string.
        assert_eq!(message["content"], Value::Null);
        assert_eq!(message["reasoning_content"], "why");
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");

        let calls = message["tool_calls"].as_array().expect("a call array");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"Oslo"}"#);
        assert_eq!(calls[1]["function"]["name"], "now");
        assert_eq!(calls[1]["function"]["arguments"], "{}");
        assert_call_id(&calls[0]["id"]);
        assert_call_id(&calls[1]["id"]);
        assert_ne!(calls[0]["id"], calls[1]["id"]);
    }

    /// A turn that called nothing says so by omission, not with an empty array.
    /// Null content is reserved for tool turns: a turn that just came out empty
    /// still answers with the empty string, which is the difference clients read
    /// to tell the two apart.
    #[test]
    fn tool_calls_are_absent_when_the_turn_made_none() {
        let body = completion_body("id", "m", 0, &completion("", "answer", StopKind::EndTurn));
        assert!(body["choices"][0]["message"].get("tool_calls").is_none());
        assert_eq!(body["choices"][0]["message"]["content"], "answer");

        let wordless = completion_body("id", "m", 0, &completion("why", "", StopKind::EndTurn));
        assert_eq!(wordless["choices"][0]["message"]["content"], "");
    }

    #[test]
    fn finish_reasons_fold_a_stop_sequence_into_stop() {
        for (stop, expected) in [
            (StopKind::EndTurn, "stop"),
            (StopKind::StopSequence("END".into()), "stop"),
            (StopKind::MaxTokens, "length"),
            (StopKind::ToolUse, "tool_calls"),
        ] {
            let body = completion_body("id", "m", 0, &completion("", "a", stop));
            assert_eq!(body["choices"][0]["finish_reason"], expected);
        }
    }

    #[test]
    fn the_chunk_sequence_opens_with_a_role_and_ends_with_the_sentinel() {
        let mut stream =
            ChunkStream::new("chatcmpl-1".into(), "gpt-4o".into(), 1_700_000_000, false);
        let frames = encode_all(
            &mut stream,
            vec![
                EngineEvent::Start {
                    input_tokens: 100,
                    cached_tokens: 40,
                },
                EngineEvent::Thinking("why".into()),
                EngineEvent::Text("ans".into()),
                EngineEvent::Text("wer".into()),
                EngineEvent::Done {
                    stop: StopKind::EndTurn,
                    output_tokens: 25,
                    thinking_tokens: 10,
                },
            ],
        );
        assert_eq!(frames.len(), 6);
        // Every chunk on this API is an unnamed event.
        assert!(frames.iter().all(|frame| frame.name.is_none()));

        let role = payload(&frames[0]);
        assert_eq!(role["object"], "chat.completion.chunk");
        assert_eq!(role["id"], "chatcmpl-1");
        assert_eq!(role["model"], "gpt-4o");
        assert_eq!(role["created"], 1_700_000_000u64);
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(role["choices"][0]["finish_reason"], Value::Null);

        // Reasoning streams before content, in its own field.
        assert_eq!(
            payload(&frames[1])["choices"][0]["delta"]["reasoning_content"],
            "why"
        );
        assert_eq!(payload(&frames[2])["choices"][0]["delta"]["content"], "ans");
        assert_eq!(payload(&frames[3])["choices"][0]["delta"]["content"], "wer");

        let finish = payload(&frames[4]);
        assert_eq!(finish["choices"][0]["delta"], json!({}));
        assert_eq!(finish["choices"][0]["finish_reason"], "stop");

        assert_eq!(frames[5].data, "[DONE]");
    }

    /// The same sequence with tool calls in it: each call opens with its
    /// identity and an empty argument string, its argument fragments follow
    /// verbatim under the same index, and the next call gets the next index.
    #[test]
    fn tool_calls_stream_as_indexed_delta_entries() {
        let mut stream =
            ChunkStream::new("chatcmpl-2".into(), "gpt-4o".into(), 1_700_000_000, false);
        let frames = encode_all(
            &mut stream,
            vec![
                EngineEvent::Start {
                    input_tokens: 100,
                    cached_tokens: 40,
                },
                EngineEvent::Thinking("why".into()),
                EngineEvent::Text("checking".into()),
                EngineEvent::ToolCallStart {
                    name: "get_weather".into(),
                },
                EngineEvent::ToolCallDelta(r#"{"city":"#.into()),
                EngineEvent::ToolCallDelta(r#""Oslo"}"#.into()),
                EngineEvent::ToolCallEnd,
                EngineEvent::ToolCallStart { name: "now".into() },
                EngineEvent::ToolCallDelta("{}".into()),
                EngineEvent::ToolCallEnd,
                EngineEvent::Done {
                    stop: StopKind::ToolUse,
                    output_tokens: 25,
                    thinking_tokens: 10,
                },
            ],
        );
        // Role, reasoning, content, three chunks for the first call, two for
        // the second, finish, sentinel. The two ends produce no chunk.
        assert_eq!(frames.len(), 10);
        let delta = |index: usize| payload(&frames[index])["choices"][0]["delta"].clone();
        assert_eq!(delta(0)["role"], "assistant");
        assert_eq!(delta(1)["reasoning_content"], "why");
        assert_eq!(delta(2)["content"], "checking");

        let opening = delta(3)["tool_calls"][0].clone();
        assert_eq!(opening["index"], 0);
        assert_eq!(opening["type"], "function");
        assert_eq!(opening["function"]["name"], "get_weather");
        assert_eq!(opening["function"]["arguments"], "");
        assert_call_id(&opening["id"]);

        // Argument fragments carry the index and nothing else identifying: the
        // client appends them to the call it already opened.
        assert_eq!(
            delta(4)["tool_calls"],
            json!([{"index": 0, "function": {"arguments": r#"{"city":"#}}])
        );
        assert_eq!(
            delta(5)["tool_calls"],
            json!([{"index": 0, "function": {"arguments": r#""Oslo"}"#}}])
        );

        let second = delta(6)["tool_calls"][0].clone();
        assert_eq!(second["index"], 1);
        assert_eq!(second["function"]["name"], "now");
        assert_ne!(second["id"], opening["id"]);
        assert_eq!(
            delta(7)["tool_calls"],
            json!([{"index": 1, "function": {"arguments": "{}"}}])
        );

        let finish = payload(&frames[8]);
        assert_eq!(finish["choices"][0]["delta"], json!({}));
        assert_eq!(finish["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(frames[9].data, "[DONE]");
    }

    #[test]
    fn include_usage_adds_a_choiceless_usage_chunk_before_the_sentinel() {
        let mut stream = ChunkStream::new("id".into(), "m".into(), 0, true);
        let frames = encode_all(
            &mut stream,
            vec![
                EngineEvent::Start {
                    input_tokens: 100,
                    cached_tokens: 40,
                },
                EngineEvent::Text("hi".into()),
                EngineEvent::Done {
                    stop: StopKind::MaxTokens,
                    output_tokens: 25,
                    thinking_tokens: 10,
                },
            ],
        );
        assert_eq!(frames.len(), 5);
        assert_eq!(payload(&frames[2])["choices"][0]["finish_reason"], "length");

        let usage = payload(&frames[3]);
        assert_eq!(usage["choices"], json!([]));
        assert_eq!(usage["usage"]["prompt_tokens"], 100);
        assert_eq!(usage["usage"]["completion_tokens"], 25);
        assert_eq!(usage["usage"]["total_tokens"], 125);
        assert_eq!(usage["usage"]["prompt_tokens_details"]["cached_tokens"], 40);
        assert_eq!(
            usage["usage"]["completion_tokens_details"]["reasoning_tokens"],
            10
        );

        assert_eq!(frames[4].data, "[DONE]");
    }

    #[test]
    fn a_mid_stream_engine_error_is_delivered_then_the_sentinel() {
        let mut stream = ChunkStream::new("id".into(), "m".into(), 0, true);
        let frames = encode_all(
            &mut stream,
            vec![
                EngineEvent::Start {
                    input_tokens: 5,
                    cached_tokens: 0,
                },
                EngineEvent::Error {
                    message: "Metal command buffer failed".into(),
                    request_fault: false,
                },
                EngineEvent::Text("unreachable".into()),
            ],
        );
        assert_eq!(frames.len(), 3);
        let error = payload(&frames[1]);
        assert_eq!(error["error"]["message"], "Metal command buffer failed");
        assert_eq!(error["error"]["type"], "server_error");
        assert_eq!(error["error"]["code"], Value::Null);
        assert_eq!(frames[2].data, "[DONE]");
    }

    /// The engine says whose fault a failure was; the envelope follows that flag
    /// rather than anything about the wording of the message.
    #[test]
    fn the_engines_fault_flag_picks_the_error_envelope() {
        let mut stream = ChunkStream::new("id".into(), "m".into(), 0, false);
        let frames = encode_all(
            &mut stream,
            vec![EngineEvent::Error {
                message: "the prompt is too long".into(),
                request_fault: true,
            }],
        );
        assert_eq!(
            payload(&frames[0])["error"]["type"],
            "invalid_request_error"
        );
        assert_eq!(
            payload(&frames[0])["error"]["code"],
            "context_length_exceeded"
        );

        let request = engine_error("anything at all", true);
        assert_eq!(request.status, StatusCode::BAD_REQUEST);
        assert_eq!(request.body["error"]["code"], "context_length_exceeded");
        // A server failure whose message happens to mention the context is still
        // a server failure.
        let server = engine_error("the context buffer could not be allocated", false);
        assert_eq!(server.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(server.body["error"]["type"], "server_error");
    }

    #[test]
    fn empty_deltas_produce_no_chunks() {
        let mut stream = ChunkStream::new("id".into(), "m".into(), 0, false);
        let frames = encode_all(
            &mut stream,
            vec![
                EngineEvent::Start {
                    input_tokens: 1,
                    cached_tokens: 0,
                },
                // The decode stream withholds partial UTF-8, so a token can
                // arrive carrying no text.
                EngineEvent::Text(String::new()),
                EngineEvent::Done {
                    stop: StopKind::EndTurn,
                    output_tokens: 1,
                    thinking_tokens: 0,
                },
            ],
        );
        assert_eq!(frames.len(), 3);
        assert_eq!(
            payload(&frames[0])["choices"][0]["delta"]["role"],
            "assistant"
        );
        assert_eq!(payload(&frames[1])["choices"][0]["finish_reason"], "stop");
    }
}
