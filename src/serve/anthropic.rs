//! Anthropic Messages API surface: POST /v1/messages and /v1/messages/count_tokens.
//!
//! Parsing is deliberately permissive. Requests arrive from SDKs and harnesses
//! that add fields faster than any server tracks them, so unknown keys are
//! ignored and every shape the spec allows for a field is accepted; the only
//! things rejected are the features this server genuinely cannot serve (images,
//! server-side tools), which fail loudly rather than quietly producing a wrong
//! answer. Client tools are served: definitions are rendered into the prompt
//! and calls are parsed back out. `tools_mode` can opt out of that — `"strip"`
//! drops the definitions and `"reject"` refuses the request — for harnesses
//! that cannot be asked to stop sending them.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use super::config::{ServeSettings, ToolsMode};
use super::types::{ClientId, Dialect, EngineEvent, StopKind};
use super::{
    ApiError, AppState, Completion, EngineFailure, JobRequest, SseEncoder, SseFrame, SubmitError,
    collect_completion, random_id, sse_response, submit,
};
use crate::chat::{Message, ToolCall};
use crate::constrain::{self, Grammar};
use crate::generate::feasible_think_budget;
use crate::sampler::SamplerOptions;

/// Thinking blocks carry a signature the client is expected to send back
/// verbatim. Nothing local verifies it, but a client that round-trips blocks
/// needs the field to be there, so every block gets this constant.
const THINKING_SIGNATURE: &str = "xwen-local";

/// What a request naming an unsupported capability gets back. Failing here is
/// the point: a harness that silently loses its tool definitions produces
/// answers that look fine and are not.
const NO_TOOLS: &str = "tool use not yet supported by this server";
/// A `tool_use` block is a tool call the model already made. With tools served
/// it replays into the prompt like any other turn, but a server told to strip
/// or reject definitions cannot render one: dropping it would silently delete
/// part of the conversation, so it is refused in those modes instead.
const NO_TOOL_HISTORY: &str = "tool use not yet supported by this server: a conversation containing tool calls \
     cannot be rendered";
/// Server tools are executed by whoever hosts the model. This server hosts
/// none, so a conversation that already contains one of their calls is a
/// conversation it cannot continue.
const NO_SERVER_TOOLS: &str =
    "server tool use is not supported by this server: it has no server-side tools";
/// A call's arguments are named values, and the prompt renders them one key at
/// a time. Anything else — a bare string, a list — has no rendering at all.
const TOOL_INPUT_NOT_OBJECT: &str = "tool_use: input must be an object";
/// Forcing a call needs the sampler constrained to the grammar of one, which
/// this server does not do. Claiming to honor the request and then answering in
/// prose is the failure mode this refuses to have.
const NO_FORCED_TOOLS: &str = "tool_choice: only \"auto\" and \"none\" are supported by this server; it cannot guarantee a \
     tool call";
const NO_VISION: &str = "image blocks are not supported by this server: the model is text-only";
const NO_DOCUMENTS: &str =
    "document blocks are not supported by this server: the model is text-only";
/// A conversation with no turns has nothing to answer. The hosted API rejects it
/// too, so a harness that sends one is already prepared for this.
const EMPTY_MESSAGES: &str = "messages: at least one message is required";
/// A schema-constrained reply masks the whole answer section to the schema,
/// which leaves no way to emit a tool-call span; serving both at once would
/// silently make the tools uncallable.
const SCHEMA_WITH_TOOLS: &str = "output_config.format cannot be combined with tools on this server: the schema \
     constrains the entire reply, leaving no way to emit a tool call";

/// The error envelope every failure on this API wears.
pub(crate) fn error(status: StatusCode, kind: &str, message: impl Into<String>) -> ApiError {
    ApiError {
        status,
        body: json!({"type": "error", "error": {"type": kind, "message": message.into()}}),
        headers: Vec::new(),
    }
}

fn bad_request(message: impl Into<String>) -> ApiError {
    error(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

/// The queue-full answer: 429 with a short `Retry-After`, not the 529
/// `overloaded_error` the hosted API documents. Measured against Claude Code,
/// three consecutive 529s escalate to model fallback or a fatal overload
/// error, and a background request's 529 is dropped outright — while a 429
/// goes through its ordinary backoff and retries. The `Retry-After` must stay
/// well under 60 seconds; the client treats a longer one as fatal.
fn overloaded() -> ApiError {
    error(
        StatusCode::TOO_MANY_REQUESTS,
        "rate_limit_error",
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
pub(crate) struct MessagesRequest {
    /// Any string is accepted and echoed back: this server has exactly one
    /// model, and rejecting a name would only break harnesses that hard-code
    /// one.
    pub model: Option<String>,
    pub max_tokens: Option<usize>,
    pub messages: Vec<InputMessage>,
    pub system: Option<Content>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub stop_sequences: Option<Vec<String>>,
    pub stream: Option<bool>,
    pub thinking: Option<ThinkingParam>,
    /// Held as a `Value` so a tool list of any shape reaches this module's own
    /// validation — and its error messages — instead of a parse error about its
    /// fields.
    pub tools: Option<Value>,
    /// Likewise: the policy check reads the `type` itself, so an unrecognized
    /// shape is answered rather than rejected by serde.
    pub tool_choice: Option<Value>,
    /// Structured-output config; `format` is the schema constraint.
    pub output_config: Option<OutputConfig>,
    /// The deprecated top-level spelling of the same thing. Parsed only to be
    /// refused by name: this server ignores unknown fields, and a client that
    /// believes it requested schema-validated output must not silently get
    /// unconstrained text.
    pub output_format: Option<Value>,
    /// Client-supplied identity. Nothing here affects the reply; it is read so
    /// the metrics history can say which client a run belonged to.
    ///
    /// A `Value`, like `tools` and `tool_choice` above and for the same reason:
    /// this dialect accepts and drops what it does not understand, and a field
    /// nothing depends on must never be the thing that fails a request. A
    /// `metadata` that is not an object, or whose `user_id` is not a string,
    /// yields no client id rather than a 400.
    pub metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OutputConfig {
    pub format: Option<OutputFormat>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OutputFormat {
    #[serde(rename = "type")]
    pub kind: String,
    pub schema: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct InputMessage {
    pub role: String,
    pub content: Option<Content>,
}

/// Every content-bearing field accepts either a bare string or a block list.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum Content {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Deserialize)]
pub(crate) struct ContentBlock {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: Option<String>,
    /// A `tool_result`'s own content, which is itself a string or a block list.
    pub content: Option<Content>,
    /// A `thinking` block's reasoning text.
    pub thinking: Option<String>,
    /// The function a `tool_use` block calls.
    pub name: Option<String>,
    /// A `tool_use`'s arguments, as the object the client sent.
    pub input: Option<Value>,
    // The rest of a block's fields are declared but never read: serde would
    // ignore them either way, and this is where the shapes this server accepts
    // are written down. Each is dropped for a reason of its own.
    /// A `tool_use`'s call id. The template pairs a call with its result by
    /// position and writes no id, so the client's is not carried.
    #[allow(dead_code)]
    pub id: Option<String>,
    /// Which call a `tool_result` answers — positional here, as above.
    #[allow(dead_code)]
    pub tool_use_id: Option<String>,
    /// A `thinking` block's signature. Never verified: it is this server's own
    /// constant on the way out, and whatever the client sends on the way back.
    #[allow(dead_code)]
    pub signature: Option<String>,
    /// A `redacted_thinking` block's opaque payload — reasoning this server
    /// cannot read, let alone replay into a prompt.
    #[allow(dead_code)]
    pub data: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ThinkingParam {
    #[serde(rename = "type")]
    pub kind: String,
    pub budget_tokens: Option<usize>,
}

/// A request reduced to what the engine and the response builders need.
pub(crate) struct Prepared {
    /// Echoed verbatim in the response, whatever the client called the model.
    pub model: String,
    pub stream: bool,
    pub job: JobRequest,
}

/// One piece of a message's content, in the order it appeared. Tool results
/// become their own turns, so they cannot simply be concatenated with the text
/// around them.
enum Segment {
    Text(String),
    /// A result the client sent back, carrying the id of the call it answers
    /// when it named one. The id never reaches the prompt — the template pairs
    /// a result with a call by position — but it is what
    /// [`order_results_by_call`] needs to put the positions right first.
    ToolResult {
        tool_use_id: Option<String>,
        text: String,
    },
    /// A call the model made on a previous turn, replayed into the prompt, with
    /// the client's id for it alongside — dropped after the pairing, as above.
    ToolUse {
        id: Option<String>,
        call: ToolCall,
    },
    /// Reasoning the client sent back. Whether it survives into the prompt is
    /// decided by position, in [`normalize`].
    Thinking(String),
}

/// Flatten one content value into ordered segments, rejecting the block types
/// this server cannot serve. `mode` decides only the fate of `tool_use`: a
/// server that is not serving tools has no rendering for one.
fn segments(content: &Content, out: &mut Vec<Segment>, mode: ToolsMode) -> Result<(), ApiError> {
    match content {
        Content::Text(text) => out.push(Segment::Text(text.clone())),
        Content::Blocks(blocks) => {
            for block in blocks {
                match block.kind.as_str() {
                    "text" => out.push(Segment::Text(block.text.clone().unwrap_or_default())),
                    "thinking" => {
                        out.push(Segment::Thinking(
                            block.thinking.clone().unwrap_or_default(),
                        ));
                    }
                    // Reasoning this server cannot read. Accepting it and
                    // contributing nothing is the whole contract: a client that
                    // round-trips its blocks must not be told one of them is an
                    // error.
                    "redacted_thinking" => {}
                    "tool_result" => {
                        let mut inner = Vec::new();
                        if let Some(content) = &block.content {
                            segments(content, &mut inner, mode)?;
                        }
                        out.push(Segment::ToolResult {
                            tool_use_id: block.tool_use_id.clone(),
                            text: join_text(inner),
                        });
                    }
                    "tool_use" if mode == ToolsMode::Native => {
                        out.push(Segment::ToolUse {
                            id: block.id.clone(),
                            call: tool_call(block)?,
                        });
                    }
                    "tool_use" => return Err(bad_request(NO_TOOL_HISTORY)),
                    "server_tool_use" => return Err(bad_request(NO_SERVER_TOOLS)),
                    "image" => return Err(bad_request(NO_VISION)),
                    "document" => return Err(bad_request(NO_DOCUMENTS)),
                    // An unrecognized block is ignored rather than rejected: it
                    // is far more often a field a newer SDK added than content
                    // the answer depends on.
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

/// One `tool_use` block's call as the prompt replays it. The argument order is
/// the client's, which `serde_json`'s `preserve_order` is what preserves.
fn tool_call(block: &ContentBlock) -> Result<ToolCall, ApiError> {
    let arguments = match &block.input {
        // A call with no arguments at all; the template renders just the name.
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Object(input)) => input
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
        Some(_) => return Err(bad_request(TOOL_INPUT_NOT_OBJECT)),
    };
    Ok(ToolCall {
        name: block.name.clone().unwrap_or_default(),
        arguments,
    })
}

/// The text of a segment list, tool results folded in. Used where a turn cannot
/// be split (a system prompt, a tool result's own content). Calls and reasoning
/// are not text and contribute nothing.
fn join_text(segments: Vec<Segment>) -> String {
    let parts: Vec<String> = segments
        .into_iter()
        .filter_map(|segment| match segment {
            Segment::Text(text) | Segment::ToolResult { text, .. } => Some(text),
            Segment::ToolUse { .. } | Segment::Thinking(_) => None,
        })
        .filter(|text| !text.is_empty())
        .collect();
    parts.join("\n")
}

/// Flatten a content value straight to text.
fn text_of(content: &Content, mode: ToolsMode) -> Result<String, ApiError> {
    let mut out = Vec::new();
    segments(content, &mut out, mode)?;
    Ok(join_text(out))
}

/// Join two turns' texts, without inventing a blank line for an empty one — an
/// assistant turn that was nothing but a tool call has no text at all.
fn join_into(previous: &mut String, text: &str) {
    if text.is_empty() {
        return;
    }
    if !previous.is_empty() {
        previous.push('\n');
    }
    previous.push_str(text);
}

/// Append a turn, merging it into the previous one when the roles match. The
/// template renders one block per message, so two consecutive user messages
/// would otherwise become two `<user>` turns where the client meant one.
///
/// A system turn past the head of the conversation demotes to a user turn: the
/// template renders exactly one system block, ahead of the conversation, while
/// harnesses (Claude Code's token-budget reminders) inject system messages
/// mid-conversation that are addressed to the model like any user text.
fn push_turn(messages: &mut Vec<Message>, turn: Message) {
    let turn = match turn {
        Message::System(text) if !matches!(messages.last(), None | Some(Message::System(_))) => {
            Message::User(text)
        }
        turn => turn,
    };
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

/// Put a user turn's tool results into the order the calls they answer were
/// made in, leaving everything else where it is.
///
/// The prompt writes no ids: a `<tool_response>` answers a call by position,
/// the way the template does it. A client is free to return its results in
/// whatever order they finished in, though, and it identifies each one by
/// `tool_use_id` when it does — so results that all name a call from the turn
/// just before are sorted into that turn's order before their ids are dropped.
/// One unidentified result leaves the whole turn alone: a partial reordering
/// would pair worse than the client's own order.
fn order_results_by_call(parts: &mut [Segment], calls: &[String]) {
    let slots: Vec<usize> = parts
        .iter()
        .enumerate()
        .filter(|(_, part)| matches!(part, Segment::ToolResult { .. }))
        .map(|(index, _)| index)
        .collect();
    if slots.len() < 2 {
        return;
    }
    let mut ranked = Vec::with_capacity(slots.len());
    for &slot in &slots {
        let rank = match &parts[slot] {
            Segment::ToolResult {
                tool_use_id: Some(id),
                ..
            } if !id.is_empty() => calls.iter().position(|call| call == id),
            _ => None,
        };
        let Some(rank) = rank else { return };
        ranked.push((rank, slot));
    }
    // Stable, so two results naming the same call keep the client's order.
    ranked.sort_by_key(|(rank, _)| *rank);
    let ordered: Vec<Segment> = ranked
        .iter()
        .map(|&(_, slot)| std::mem::replace(&mut parts[slot], Segment::Text(String::new())))
        .collect();
    // `slots` is ascending, so the results land back in the turn's own result
    // positions — text between them does not move.
    for (&slot, segment) in slots.iter().zip(ordered) {
        parts[slot] = segment;
    }
}

/// Normalize a request's conversation into the turns `chat.rs` renders.
pub(crate) fn normalize(
    request: &MessagesRequest,
    mode: ToolsMode,
) -> Result<Vec<Message>, ApiError> {
    let mut messages = Vec::new();
    if let Some(system) = &request.system {
        messages.push(Message::System(text_of(system, mode)?));
    }
    // The ids of the calls the assistant turn just before this one made, which
    // is the only turn a client's `tool_use_id`s can refer to. Cleared by
    // anything that is not an assistant turn, and accumulated across a run of
    // them, since `push_turn` merges those into one rendered turn.
    let mut preceding_calls: Vec<String> = Vec::new();
    for message in &request.messages {
        let Some(content) = &message.content else {
            continue;
        };
        match message.role.as_str() {
            "user" => {
                let mut parts = Vec::new();
                segments(content, &mut parts, mode)?;
                order_results_by_call(&mut parts, &preceding_calls);
                preceding_calls.clear();
                let mut pending = Vec::new();
                for part in parts {
                    match part {
                        Segment::ToolResult { text, .. } => {
                            // A tool result ends the user text that preceded
                            // it, so the turns keep the client's ordering.
                            let before = join_text(std::mem::take(&mut pending));
                            if !before.is_empty() {
                                push_turn(&mut messages, Message::User(before));
                            }
                            push_turn(&mut messages, Message::ToolResponse(text));
                        }
                        // Reasoning and calls are the assistant's; the template
                        // has nowhere to put either inside a user turn.
                        Segment::Thinking(_) | Segment::ToolUse { .. } => {}
                        text => pending.push(text),
                    }
                }
                let rest = join_text(pending);
                if !rest.is_empty() {
                    push_turn(&mut messages, Message::User(rest));
                }
            }
            "assistant" => {
                let mut parts = Vec::new();
                segments(content, &mut parts, mode)?;
                let mut reasoning = String::new();
                let mut tool_calls = Vec::new();
                let mut text = Vec::new();
                for part in parts {
                    match part {
                        Segment::Thinking(thought) => join_into(&mut reasoning, &thought),
                        Segment::ToolUse { id, call } => {
                            // A call with no id still takes a slot, or every
                            // later call's position would be off by one.
                            preceding_calls.push(id.unwrap_or_default());
                            tool_calls.push(call);
                        }
                        other => text.push(other),
                    }
                }
                // Replaying reasoning is part of serving tools: it is what
                // carries a plan across the results it is waiting on. It is
                // passed through on EVERY assistant turn — which turns render
                // it is the renderer's per-dialect decision (chat.rs: 3.6
                // drops superseded reasoning, 3.8's preserve_thinking default
                // keeps it), so Anthropic's own strip of non-trailing thinking
                // is deliberately not emulated here. A server told not to
                // serve tools renders the history it always did.
                let keep_reasoning = mode == ToolsMode::Native && !reasoning.is_empty();
                push_turn(
                    &mut messages,
                    Message::Assistant {
                        content: join_text(text),
                        reasoning: keep_reasoning.then_some(reasoning),
                        tool_calls,
                    },
                );
            }
            // Not a role the spec puts in `messages`, but harnesses send it.
            "system" => {
                preceding_calls.clear();
                push_turn(&mut messages, Message::System(text_of(content, mode)?));
            }
            role => {
                return Err(bad_request(format!(
                    "unknown message role {role:?}: expected \"user\" or \"assistant\""
                )));
            }
        }
    }
    Ok(messages)
}

/// Resolve the thinking mode. A request that states one wins; a request that
/// says nothing gets the server's configured default.
pub(crate) fn resolve_thinking(
    param: Option<&ThinkingParam>,
    settings: &ServeSettings,
    max_tokens: Option<usize>,
) -> Result<(bool, Option<usize>), ApiError> {
    let (enabled, budget) = match param {
        None => (settings.thinking_force, settings.thinking_budget),
        Some(param) => match param.kind.as_str() {
            "enabled" => (true, param.budget_tokens.or(settings.thinking_budget)),
            // Adaptive hands the length to the model, so a budget sent alongside
            // it is not one: the server's own default is all that applies.
            "adaptive" => (true, settings.thinking_budget),
            "disabled" => (false, None),
            kind => {
                return Err(bad_request(format!(
                    "unknown thinking type {kind:?}: expected \"enabled\", \"adaptive\" or \"disabled\""
                )));
            }
        },
    };
    if !enabled {
        return Ok((false, None));
    }
    // A ceiling has to leave the reply room to conclude in. Any request the API
    // accepts — the spec asks only for `budget_tokens < max_tokens` — must
    // still generate, so a budget the reasoning schedule cannot express is
    // lowered to the largest one that fits, or dropped entirely when none
    // does. Reasoning without a ceiling is always servable.
    let budget = match (budget, max_tokens) {
        (Some(budget), Some(max_tokens)) => feasible_think_budget(budget, max_tokens),
        // `count_tokens` has no output cap to fit a budget against, and the
        // budget does not change the prompt it is counting.
        (budget, _) => budget,
    };
    Ok((enabled, budget))
}

/// Sampling for one request: whatever it asked for, over the server's
/// configured defaults, over the model card's recommendation for the request's
/// resolved thinking mode (temp 1.0 / top_p 0.95 thinking, 0.7 / 0.80 not) and
/// for `target`, which is what the presence penalty is keyed to on top of the
/// mode.
///
/// This API has no presence-penalty field of its own — the Messages API does
/// not define one — so a request here takes the server-wide setting, or the
/// checkpoint default when that is unset.
fn sampling(
    request: &MessagesRequest,
    settings: &ServeSettings,
    target: crate::serve::types::Target,
    thinking: bool,
) -> SamplerOptions {
    let recommended = SamplerOptions::recommended_for(target.model, thinking);
    SamplerOptions {
        temperature: request
            .temperature
            .or(settings.temperature)
            .unwrap_or(recommended.temperature),
        top_k: request
            .top_k
            .or(settings.top_k)
            .unwrap_or(recommended.top_k),
        top_p: request
            .top_p
            .or(settings.top_p)
            .unwrap_or(recommended.top_p),
        presence_penalty: settings
            .presence_penalty
            .unwrap_or(recommended.presence_penalty),
        seed: recommended.seed,
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

/// What a request's `tool_choice` asks of the sampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolChoice {
    /// The model decides, which is the only thing an unconstrained sampler does.
    Auto,
    /// The tools are defined but off-limits for this turn: the prompt is
    /// rendered without them.
    None,
    /// A call is required, or a named one is. Refused — see [`NO_FORCED_TOOLS`].
    Forced,
}

/// Read the `tool_choice` policy. The object form is the documented one; a bare
/// string is accepted as its `type`, since that is the OpenAI spelling and
/// harnesses mix the two up.
fn tool_choice(choice: Option<&Value>) -> Result<ToolChoice, ApiError> {
    let kind = match choice {
        None | Some(Value::Null) => return Ok(ToolChoice::Auto),
        Some(Value::String(kind)) => kind.as_str(),
        Some(Value::Object(choice)) => match choice.get("type").and_then(Value::as_str) {
            Some(kind) => kind,
            // `disable_parallel_tool_use` on its own, say: nothing is being
            // asked of the sampler.
            None => return Ok(ToolChoice::Auto),
        },
        Some(_) => {
            return Err(bad_request(
                "tool_choice must be an object such as {\"type\":\"auto\"}",
            ));
        }
    };
    match kind {
        "auto" => Ok(ToolChoice::Auto),
        "none" => Ok(ToolChoice::None),
        "any" | "tool" => Ok(ToolChoice::Forced),
        kind => Err(bad_request(format!(
            "unknown tool_choice type {kind:?}: expected \"auto\" or \"none\""
        ))),
    }
}

/// One Anthropic tool definition in the OpenAI object shape the prompt renders
/// and the engine's parser reads. `description` is omitted rather than nulled
/// when the client sent none: the rendering is a byte-exact contract.
fn oai_tool(tool: &Value) -> Result<Value, ApiError> {
    let Some(name) = tool.get("name").and_then(Value::as_str) else {
        return Err(bad_request("tools: every tool needs a name"));
    };
    let mut function = serde_json::Map::new();
    function.insert("name".into(), json!(name));
    if let Some(description) = tool.get("description").filter(|value| !value.is_null()) {
        function.insert("description".into(), description.clone());
    }
    if let Some(schema) = tool.get("input_schema").filter(|value| !value.is_null()) {
        function.insert("parameters".into(), schema.clone());
    }
    Ok(json!({"type": "function", "function": Value::Object(function)}))
}

/// Apply the server's tools policy to a request's definitions: refuse the
/// request, drop them and answer as if there were none, or translate them into
/// the shape the prompt renders.
fn check_tools(tools: Option<&Value>, mode: ToolsMode) -> Result<Vec<Value>, ApiError> {
    match mode {
        ToolsMode::Reject if requests_tools(tools) => Err(bad_request(NO_TOOLS)),
        ToolsMode::Native => match tools {
            None | Some(Value::Null) => Ok(Vec::new()),
            Some(Value::Array(tools)) => tools.iter().map(oai_tool).collect(),
            Some(_) => Err(bad_request("tools must be an array of tool definitions")),
        },
        _ => Ok(Vec::new()),
    }
}

/// The tools a request is actually served with: its definitions under the
/// server's policy, minus the turn's own opt-out. `tool_choice` is only
/// meaningful where tools are honored, so the debug modes ignore it — a
/// stripped request is answered as if it had never mentioned tools at all.
fn resolve_tools(request: &MessagesRequest, mode: ToolsMode) -> Result<Vec<Value>, ApiError> {
    let tools = check_tools(request.tools.as_ref(), mode)?;
    if mode != ToolsMode::Native {
        return Ok(tools);
    }
    match tool_choice(request.tool_choice.as_ref())? {
        ToolChoice::Auto => Ok(tools),
        ToolChoice::None => Ok(Vec::new()),
        ToolChoice::Forced => Err(bad_request(NO_FORCED_TOOLS)),
    }
}

/// Resolve `output_config.format` into the grammar the job will decode under.
/// Compiled here, on the HTTP thread, so a schema the compiler rejects is a
/// 400 with the compiler's own message.
fn resolve_output_format(
    config: Option<&OutputConfig>,
    tools: &[Value],
) -> Result<Option<Grammar>, ApiError> {
    let Some(format) = config.and_then(|c| c.format.as_ref()) else {
        return Ok(None);
    };
    match format.kind.as_str() {
        "text" => Ok(None),
        "json_schema" => {
            if !tools.is_empty() {
                return Err(bad_request(SCHEMA_WITH_TOOLS));
            }
            let factory = constrain::shared().map_err(|e| {
                error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    format!("{e:#}"),
                )
            })?;
            let schema = format
                .schema
                .as_ref()
                .ok_or_else(|| bad_request("output_config.format.schema is required"))?;
            factory
                .compile(schema)
                .map(Some)
                .map_err(|e| bad_request(format!("{e:#}")))
        }
        other => Err(bad_request(format!(
            "output_config.format type {other:?} is not supported; use \"json_schema\""
        ))),
    }
}

pub(crate) fn prepare(
    request: MessagesRequest,
    settings: &ServeSettings,
    default_model: &str,
    target: crate::serve::types::Target,
) -> Result<Prepared, ApiError> {
    let tools = resolve_tools(&request, settings.tools_mode)?;
    if request.output_format.is_some() {
        return Err(bad_request(
            "output_format is the deprecated spelling and is not supported here: put the schema \
             in output_config.format ({\"type\": \"json_schema\", \"schema\": ...})",
        ));
    }
    let grammar = resolve_output_format(request.output_config.as_ref(), &tools)?;
    if request.messages.is_empty() {
        return Err(bad_request(EMPTY_MESSAGES));
    }
    let Some(max_tokens) = request.max_tokens else {
        return Err(bad_request("max_tokens is required"));
    };
    if max_tokens == 0 {
        return Err(bad_request("max_tokens must be at least 1"));
    }

    let messages = normalize(&request, settings.tools_mode)?;
    let (enable_thinking, max_think) =
        resolve_thinking(request.thinking.as_ref(), settings, Some(max_tokens))?;

    Ok(Prepared {
        model: request
            .model
            .clone()
            .unwrap_or_else(|| default_model.to_string()),
        stream: request.stream.unwrap_or(false),
        job: JobRequest {
            sampling: sampling(&request, settings, target, enable_thinking),
            messages,
            enable_thinking,
            // This API has no fields for the template knobs, so the encode
            // path keeps the dialect's preserve default and the server-wide
            // configured effort (or, unset, the template's own).
            preserve_thinking: None,
            reasoning_effort: settings.reasoning_effort,
            max_think,
            max_tokens,
            stop_sequences: request.stop_sequences.unwrap_or_default(),
            tools,
            grammar,
            continuation: None,
        },
    })
}

// --------------------------------------------------------------- response ---

/// `stop_reason` plus the `stop_sequence` that goes with it.
fn stop_fields(stop: &StopKind) -> (&'static str, Value) {
    match stop {
        StopKind::EndTurn => ("end_turn", Value::Null),
        StopKind::MaxTokens => ("max_tokens", Value::Null),
        StopKind::StopSequence(sequence) => ("stop_sequence", Value::String(sequence.clone())),
        StopKind::ToolUse => ("tool_use", Value::Null),
    }
}

/// Numbers the tool calls this process hands out ids for. Anthropic's ids pair a
/// call with its result inside one conversation, so uniqueness only has to hold
/// for a turn; a monotonic counter gives that for free and stays readable in a
/// transcript.
static NEXT_TOOL_USE_ID: AtomicU64 = AtomicU64::new(1);

fn tool_use_id() -> String {
    format!(
        "toolu_{:08}",
        NEXT_TOOL_USE_ID.fetch_add(1, Ordering::Relaxed)
    )
}

/// A call's arguments as the `input` object. The engine guarantees the joined
/// deltas parse — healing a truncated call closed if it has to — so the fallback
/// is unreachable rather than lossy.
fn tool_input(arguments: &str) -> Value {
    serde_json::from_str::<Value>(arguments)
        .ok()
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({}))
}

/// Prompt-side usage. `input_tokens` is the whole prompt; the cache fields
/// split it into the part served from the KV prefix cache and the part this
/// request had to prefill.
fn input_usage(input_tokens: usize, cached_tokens: usize) -> Value {
    json!({
        "input_tokens": input_tokens,
        "cache_read_input_tokens": cached_tokens,
        "cache_creation_input_tokens": input_tokens.saturating_sub(cached_tokens),
    })
}

pub(crate) fn message_body(id: &str, model: &str, completion: &Completion) -> Value {
    let mut content = Vec::new();
    // An empty thinking block is worse than none: clients that round-trip
    // blocks choke on a signature with nothing to sign.
    if !completion.thinking.is_empty() {
        content.push(json!({
            "type": "thinking",
            "thinking": completion.thinking,
            "signature": THINKING_SIGNATURE,
        }));
    }
    // An answer that came out empty still gets its text block — but a turn that
    // called tools instead of answering does not. The hosted API rejects an
    // empty text block on the way in, so a client that replays this turn would
    // be handing its next request a block it will be refused for.
    if !completion.text.is_empty() || completion.tool_calls.is_empty() {
        content.push(json!({"type": "text", "text": completion.text}));
    }
    // Calls follow the text the model wrote before them, in the order it wrote
    // them — the same order the prompt replays them in.
    for call in &completion.tool_calls {
        content.push(json!({
            "type": "tool_use",
            "id": tool_use_id(),
            "name": call.name,
            "input": tool_input(&call.arguments),
        }));
    }

    let (stop_reason, stop_sequence) = stop_fields(&completion.stop);
    let mut usage = input_usage(completion.input_tokens, completion.cached_tokens);
    usage["output_tokens"] = json!(completion.output_tokens);
    usage["output_tokens_details"] = json!({"thinking_tokens": completion.thinking_tokens});

    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": usage,
    })
}

// -------------------------------------------------------------------- SSE ---

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Thinking,
    Text,
    ToolUse,
}

/// Renders a generation as the Messages API event sequence.
///
/// The block indices are the delicate part: a turn that produces no thinking
/// text must not emit a thinking block at all, so its text block is index 0.
/// That is only knowable once the first content event arrives, which is why
/// nothing but `message_start` is emitted until then.
pub(crate) struct MessageStream {
    id: String,
    model: String,
    open: Option<(BlockKind, usize)>,
    next_index: usize,
    text_block_emitted: bool,
    tool_block_emitted: bool,
    /// The id and name of the call whose block is about to open. A tool block's
    /// header is the only one that carries data from the event that opened it.
    opening_tool: Option<(String, String)>,
}

impl MessageStream {
    pub(crate) fn new(id: String, model: String) -> Self {
        Self {
            id,
            model,
            open: None,
            next_index: 0,
            text_block_emitted: false,
            tool_block_emitted: false,
            opening_tool: None,
        }
    }

    fn message_start(&self, input_tokens: usize, cached_tokens: usize) -> SseFrame {
        let mut usage = input_usage(input_tokens, cached_tokens);
        // The spec's placeholder: real output usage arrives with message_delta.
        usage["output_tokens"] = json!(1);
        SseFrame::named(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": self.id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": usage,
                },
            }),
        )
    }

    /// Open a block of `kind`, closing a differently-typed open one first, and
    /// return its index.
    fn open_block(&mut self, kind: BlockKind, out: &mut VecDeque<SseFrame>) -> usize {
        if let Some((open, index)) = self.open {
            if open == kind {
                return index;
            }
            self.close_block(out);
        }
        let index = self.next_index;
        self.next_index += 1;
        self.open = Some((kind, index));
        let block = match kind {
            BlockKind::Thinking => json!({"type": "thinking", "thinking": "", "signature": ""}),
            BlockKind::Text => {
                self.text_block_emitted = true;
                json!({"type": "text", "text": ""})
            }
            BlockKind::ToolUse => {
                self.tool_block_emitted = true;
                let (id, name) = self.opening_tool.take().unwrap_or_default();
                // The arguments arrive as input_json_delta fragments; the block
                // opens with the empty object the client accumulates into.
                json!({"type": "tool_use", "id": id, "name": name, "input": {}})
            }
        };
        out.push_back(SseFrame::named(
            "content_block_start",
            json!({"type": "content_block_start", "index": index, "content_block": block}),
        ));
        index
    }

    fn close_block(&mut self, out: &mut VecDeque<SseFrame>) {
        let Some((kind, index)) = self.open.take() else {
            return;
        };
        if kind == BlockKind::Thinking {
            out.push_back(SseFrame::named(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "signature_delta", "signature": THINKING_SIGNATURE},
                }),
            ));
        }
        out.push_back(SseFrame::named(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": index}),
        ));
    }

    fn delta(&mut self, kind: BlockKind, text: String, out: &mut VecDeque<SseFrame>) {
        let index = self.open_block(kind, out);
        if text.is_empty() {
            return;
        }
        let delta = match kind {
            BlockKind::Thinking => json!({"type": "thinking_delta", "thinking": text}),
            BlockKind::Text => json!({"type": "text_delta", "text": text}),
            BlockKind::ToolUse => json!({"type": "input_json_delta", "partial_json": text}),
        };
        out.push_back(SseFrame::named(
            "content_block_delta",
            json!({"type": "content_block_delta", "index": index, "delta": delta}),
        ));
    }

    fn push_error(message: &str, kind: &str, out: &mut VecDeque<SseFrame>) {
        out.push_back(SseFrame::named(
            "error",
            json!({"type": "error", "error": {"type": kind, "message": message}}),
        ));
    }
}

impl SseEncoder for MessageStream {
    fn on_event(&mut self, event: EngineEvent, out: &mut VecDeque<SseFrame>) -> bool {
        match event {
            EngineEvent::Start {
                input_tokens,
                cached_tokens,
            } => {
                out.push_back(self.message_start(input_tokens, cached_tokens));
                false
            }
            EngineEvent::Thinking(text) => {
                // A thinking block is opened only by text that will fill it.
                if !text.is_empty() {
                    self.delta(BlockKind::Thinking, text, out);
                }
                false
            }
            EngineEvent::Text(text) => {
                self.delta(BlockKind::Text, text, out);
                false
            }
            EngineEvent::ToolCallStart { name } => {
                // Opened here rather than on the first delta: a call with no
                // arguments has none, and the client still needs the block.
                self.opening_tool = Some((tool_use_id(), name));
                self.open_block(BlockKind::ToolUse, out);
                false
            }
            EngineEvent::ToolCallDelta(json) => {
                self.delta(BlockKind::ToolUse, json, out);
                false
            }
            // Calls never interleave, so the open block is this call's. Closing
            // it here is what keeps the next call from joining it.
            EngineEvent::ToolCallEnd => {
                self.close_block(out);
                false
            }
            EngineEvent::Done {
                stop,
                output_tokens,
                thinking_tokens,
            } => {
                self.close_block(out);
                // A message carries a text block even for an answer that turned
                // out to be empty — unless it called tools instead of
                // answering, which is a turn the hosted API sends no text block
                // for either (and refuses an empty one back from).
                if !self.text_block_emitted && !self.tool_block_emitted {
                    self.open_block(BlockKind::Text, out);
                    self.close_block(out);
                }
                let (stop_reason, stop_sequence) = stop_fields(&stop);
                out.push_back(SseFrame::named(
                    "message_delta",
                    json!({
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
                        "usage": {
                            "output_tokens": output_tokens,
                            "output_tokens_details": {"thinking_tokens": thinking_tokens},
                        },
                    }),
                ));
                out.push_back(SseFrame::named(
                    "message_stop",
                    json!({"type": "message_stop"}),
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
            // A batch document can never arrive on a generation's channel.
            EngineEvent::BatchDone(_) => false,
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

// --------------------------------------------------------------- handlers ---

/// Who the client says it is: the body's `metadata.user_id` and the session
/// header. A `user_id` that is not a JSON string is stored as its compact JSON
/// rather than dropped — the shape has moved before, and a reader looking for a
/// session marker can find one in either spelling.
pub(crate) fn client_id(request: &MessagesRequest, headers: &HeaderMap) -> ClientId {
    let client = request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("user_id"))
        .and_then(|value| match value {
            Value::Null => None,
            Value::String(text) => Some(text.clone()),
            other => serde_json::to_string(other).ok(),
        });
    ClientId::new(
        client,
        super::session_header(headers),
        super::agent_header(headers),
    )
}

pub(crate) async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let mut request: MessagesRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(e) => {
            return bad_request(format!("could not parse the request body: {e}")).into_response();
        }
    };
    // Full names only, and an unknown one is a 400 — see `resolve_requested_model`.
    let (size, model_name) = match super::resolve_requested_model(
        request.model.as_deref(),
        state.default_target,
        &state.model_id,
    ) {
        Ok(resolved) => resolved,
        Err(message) => return bad_request(message).into_response(),
    };
    // The response echoes the canonical name of the model that answered, not
    // the client's spelling of it.
    request.model = Some(model_name);
    let who = client_id(&request, &headers);
    let prepared = match prepare(request, &state.settings, &state.model_id, size) {
        Ok(prepared) => prepared,
        Err(e) => return e.into_response(),
    };
    let Prepared { model, stream, job } = prepared;
    let id = random_id("msg_");
    let (mut events, guard) = match submit(&state, job, Dialect::Anthropic, stream, size, who) {
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
        return sse_response(events, guard, MessageStream::new(id, model));
    }
    // Held across the drain: hyper dropping this future — the client hung up —
    // drops the guard, which cancels the generation with its reason stated.
    let _guard = guard;
    match collect_completion(&mut events).await {
        Ok(completion) => axum::Json(message_body(&id, &model, &completion)).into_response(),
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

/// The exact count a generation request would report as `input_tokens`: the
/// same rendering and encoding the submit path performs, tools section included
/// — a tool list is often the larger half of a harness's prompt, and a count
/// that omitted it would be wrong by exactly the part the caller is asking
/// about.
fn counted_tokens(
    request: &MessagesRequest,
    settings: &ServeSettings,
    tokenizer: &crate::tokenizer::LagunaTokenizer,
    target: crate::serve::types::Target,
) -> Result<usize, ApiError> {
    let tools = resolve_tools(request, settings.tools_mode)?;
    if request.messages.is_empty() {
        return Err(bad_request(EMPTY_MESSAGES));
    }
    let messages = normalize(request, settings.tools_mode)?;
    // `max_tokens` is not required here, so the budget is left unclamped: it
    // caps the thinking span, which does not change the prompt being counted.
    let (enable_thinking, _) = resolve_thinking(request.thinking.as_ref(), settings, None)?;
    let prompt = crate::serve::encode_conversation(
        tokenizer,
        &messages,
        target.model.chat_dialect(),
        enable_thinking,
        // The count must describe the prompt a generation would render, so it
        // carries the same server-wide effort default; the preamble it selects
        // is tokens.
        None,
        settings.reasoning_effort,
        tools,
        None,
    )
    .map_err(|e| bad_request(format!("rendering the prompt failed: {e:#}")))?;
    Ok(prompt.tokens.len())
}

/// `POST /v1/messages/count_tokens` — renders and encodes the prompt without
/// touching the job queue, so it answers with the model unloaded.
pub(crate) async fn count_tokens(State(state): State<AppState>, body: Bytes) -> Response {
    let request: MessagesRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(e) => {
            return bad_request(format!("could not parse the request body: {e}")).into_response();
        }
    };
    // Judged by the same rule as a generation: a client counting tokens for a
    // model this server does not serve is asking a question about the wrong
    // model, and every other surface tells it so. All checkpoints share one
    // tokenizer, but the count still depends on WHICH checkpoint would answer —
    // its chat dialect decides the rendering the count describes.
    let (target, _) = match super::resolve_requested_model(
        request.model.as_deref(),
        state.default_target,
        &state.model_id,
    ) {
        Ok(resolved) => resolved,
        Err(message) => return bad_request(message).into_response(),
    };
    match counted_tokens(&request, &state.settings, &state.tokenizer, target) {
        Ok(count) => axum::Json(json!({"input_tokens": count})).into_response(),
        Err(e) => e.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat;
    use crate::serve::CompletedToolCall;
    use crate::serve::testutil::{
        encode_all, generation, names, payload, probe_state, render, settings, shape, try_take,
    };

    fn parse(body: &str) -> MessagesRequest {
        serde_json::from_str(body).expect("request parses")
    }

    fn target() -> crate::serve::types::Target {
        crate::serve::types::Target::official(crate::hub::Model::Qwen3827B)
    }

    fn prepared(body: &str) -> Prepared {
        prepare(parse(body), &settings(), "laguna-s-2.1", target()).expect("request prepares")
    }

    fn rejected(body: &str) -> ApiError {
        prepare(parse(body), &settings(), "laguna-s-2.1", target())
            .err()
            .expect("request is rejected")
    }

    fn prepared_in(body: &str, mode: ToolsMode) -> Prepared {
        prepare(parse(body), &tools(mode), "laguna-s-2.1", target())
            .unwrap_or_else(|e| panic!("request prepares under {mode}: {}", message(&e)))
    }

    fn rejected_in(body: &str, mode: ToolsMode) -> ApiError {
        prepare(parse(body), &tools(mode), "laguna-s-2.1", target())
            .err()
            .unwrap_or_else(|| panic!("request is rejected under {mode}"))
    }

    /// A conversation's tool calls, one line per assistant turn that made any:
    /// `name(key=value, …)` with the arguments in the order they will render.
    /// `shape` shows only text and reasoning.
    fn calls(messages: &[Message]) -> Vec<String> {
        messages
            .iter()
            .filter_map(|message| match message {
                Message::Assistant { tool_calls, .. } if !tool_calls.is_empty() => Some(
                    tool_calls
                        .iter()
                        .map(|call| {
                            let arguments: Vec<String> = call
                                .arguments
                                .iter()
                                .map(|(key, value)| format!("{key}={value}"))
                                .collect();
                            format!("{}({})", call.name, arguments.join(","))
                        })
                        .collect::<Vec<_>>()
                        .join(" "),
                ),
                _ => None,
            })
            .collect()
    }

    fn message(error: &ApiError) -> String {
        error.body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .to_string()
    }

    /// A full queue answers 429 with a short Retry-After, not the hosted API's
    /// 529: three consecutive 529s make Claude Code escalate to model fallback
    /// or a fatal overload error, while a 429 rides its ordinary backoff.
    #[test]
    fn the_queue_full_answer_is_a_retryable_429() {
        let error = overloaded();
        assert_eq!(error.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(error.body["error"]["type"], "rate_limit_error");
        // Anything over 60 seconds the client treats as fatal.
        assert_eq!(error.headers, vec![("retry-after", "1".to_string())]);
    }

    /// The tools policies, named so the tool tests state which one they are
    /// about instead of leaning on whichever is currently the default.
    fn tools(mode: ToolsMode) -> ServeSettings {
        ServeSettings {
            tools_mode: mode,
            ..settings()
        }
    }

    #[test]
    fn output_config_format_compiles_into_a_grammar() {
        let constrained = prepared(
            r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],
                "output_config":{"format":{"type":"json_schema","schema":
                {"type":"object","properties":{"a":{"type":"string"}},
                 "required":["a"],"additionalProperties":false}}}}"#,
        );
        assert!(constrained.job.grammar.is_some());
        let plain = prepared(r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}"#);
        assert!(plain.job.grammar.is_none());
    }

    #[test]
    fn bad_output_config_formats_are_rejected() {
        let error = rejected(
            r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],
                "output_config":{"format":{"type":"json_schema"}}}"#,
        );
        assert!(
            message(&error).contains("schema is required"),
            "{}",
            message(&error)
        );
        let error = rejected(
            r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],
                "output_config":{"format":{"type":"markdown"}}}"#,
        );
        assert!(
            message(&error).contains("not supported"),
            "{}",
            message(&error)
        );
    }

    /// The deprecated top-level spelling is refused by name rather than
    /// falling into the ignore-unknown-fields default: a client that believes
    /// it requested schema-validated output must not silently get
    /// unconstrained text.
    #[test]
    fn legacy_output_format_is_refused_by_name() {
        let error = rejected(
            r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],
                "output_format":{"type":"json_schema","schema":{"type":"object"}}}"#,
        );
        assert!(
            message(&error).contains("output_config.format"),
            "{}",
            message(&error)
        );
    }

    #[test]
    fn output_config_format_with_tools_is_rejected() {
        let body = r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],
            "tools":[{"name":"f","description":"d","input_schema":{"type":"object"}}],
            "output_config":{"format":{"type":"json_schema","schema":
            {"type":"object","additionalProperties":false}}}}"#;
        let error = rejected_in(body, ToolsMode::Native);
        assert!(
            message(&error).contains("cannot be combined"),
            "{}",
            message(&error)
        );
        // With tools stripped by policy, the schema is free to apply.
        assert!(prepared_in(body, ToolsMode::Strip).job.grammar.is_some());
    }

    #[test]
    fn string_and_block_content_normalize_identically() {
        let plain = prepared(r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}"#);
        let blocks = prepared(
            r#"{"max_tokens":16,"messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]}]}"#,
        );
        assert_eq!(shape(&plain.job.messages), vec!["user:Hi"]);
        assert_eq!(shape(&blocks.job.messages), shape(&plain.job.messages));
    }

    #[test]
    fn system_accepts_both_shapes_and_leads_the_conversation() {
        let plain = prepared(
            r#"{"max_tokens":16,"system":"Be brief.","messages":[{"role":"user","content":"Hi"}]}"#,
        );
        assert_eq!(
            shape(&plain.job.messages),
            vec!["system:Be brief.", "user:Hi"]
        );

        let blocks = prepared(
            r#"{"max_tokens":16,"system":[{"type":"text","text":"Be brief."}],
                "messages":[{"role":"user","content":"Hi"}]}"#,
        );
        assert_eq!(shape(&blocks.job.messages), shape(&plain.job.messages));
    }

    /// Claude Code injects its token-budget reminders as `system`-role turns
    /// mid-conversation. The template renders exactly one system block, ahead
    /// of the conversation, so a later system turn demotes to a user turn and
    /// merges into the user text around it.
    #[test]
    fn a_system_turn_past_the_head_demotes_to_a_user_turn() {
        let request = prepared(
            r#"{"max_tokens":16,"system":"Be brief.","messages":[
                {"role":"user","content":"Hi"},
                {"role":"system","content":"<total_tokens>1000 tokens left</total_tokens>"}]}"#,
        );
        assert_eq!(
            shape(&request.job.messages),
            vec![
                "system:Be brief.",
                "user:Hi\n<total_tokens>1000 tokens left</total_tokens>"
            ]
        );
    }

    /// Thinking blocks ride the history to the renderer even on a superseded
    /// turn — retention is the template dialect's call — while a redacted
    /// block contributes nothing.
    #[test]
    fn assistant_thinking_blocks_ride_the_history() {
        let request = prepared(
            r#"{"max_tokens":16,"messages":[
                {"role":"user","content":"2+2?"},
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"add them","signature":"x"},
                    {"type":"redacted_thinking","data":"opaque"},
                    {"type":"text","text":"4"}]},
                {"role":"user","content":"thanks"}]}"#,
        );
        assert_eq!(
            shape(&request.job.messages),
            vec!["user:2+2?", "assistant:4|think:add them", "user:thanks"]
        );
    }

    #[test]
    fn tool_results_become_their_own_turns_in_order() {
        let request = prepared(
            r#"{"max_tokens":16,"messages":[
                {"role":"user","content":[
                    {"type":"text","text":"before"},
                    {"type":"tool_result","tool_use_id":"t1","content":"sunny"},
                    {"type":"text","text":"after"}]}]}"#,
        );
        assert_eq!(
            shape(&request.job.messages),
            vec!["user:before", "tool:sunny", "user:after"]
        );
    }

    #[test]
    fn a_tool_results_own_blocks_flatten_to_text() {
        let request = prepared(
            r#"{"max_tokens":16,"messages":[
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"t1",
                     "content":[{"type":"text","text":"sunny"}]}]}]}"#,
        );
        assert_eq!(shape(&request.job.messages), vec!["tool:sunny"]);
    }

    #[test]
    fn consecutive_same_role_turns_merge() {
        let request = prepared(
            r#"{"max_tokens":16,"messages":[
                {"role":"user","content":"one"},
                {"role":"user","content":"two"},
                {"role":"assistant","content":"a"},
                {"role":"assistant","content":"b"}]}"#,
        );
        assert_eq!(
            shape(&request.job.messages),
            vec!["user:one\ntwo", "assistant:a\nb"]
        );
    }

    /// The session and agent headers and the body's `metadata.user_id` ride the
    /// job to the engine, which is the only thing any of them is read for: none
    /// touches the prompt, the sampling or the reply.
    #[test]
    fn the_client_and_session_ids_reach_the_job() {
        let (state, queue) = probe_state(4096);
        let request = parse(
            r#"{"max_tokens":16,"metadata":{"user_id":"user_1a2b_session_9f2c-0d31"},
                "messages":[{"role":"user","content":"Hi"}]}"#,
        );
        let mut headers = HeaderMap::new();
        headers.insert(
            crate::serve::SESSION_HEADER,
            "9f2ca1b4-0d31-4e77-9a02-7c1f8b6e5d40"
                .parse()
                .expect("a header value"),
        );
        headers.insert(
            crate::serve::AGENT_HEADER,
            "explore-metrics".parse().expect("a header value"),
        );
        let who = client_id(&request, &headers);
        let prepared =
            prepare(request, &settings(), "laguna-s-2.1", target()).expect("request prepares");
        let (_events, _guard) = crate::serve::submit(
            &state,
            prepared.job,
            Dialect::Anthropic,
            false,
            state.default_target,
            who,
        )
        .expect("the job submits");

        let job = generation(try_take(&queue).expect("the job reached the queue").job);
        assert_eq!(
            job.origin.client.as_deref(),
            Some("user_1a2b_session_9f2c-0d31")
        );
        assert_eq!(
            job.origin.session.as_deref(),
            Some("9f2ca1b4-0d31-4e77-9a02-7c1f8b6e5d40")
        );
        assert_eq!(job.origin.agent.as_deref(), Some("explore-metrics"));
    }

    /// A `user_id` of any other JSON type is kept as its compact JSON rather
    /// than dropped: the field is undocumented, its shape has moved before, and
    /// a reader can still find a session marker in the text.
    #[test]
    fn a_non_string_user_id_is_stored_as_its_json() {
        let request = parse(
            r#"{"max_tokens":16,"metadata":{"user_id":{"session_id":"9f2c-0d31"}},
                "messages":[{"role":"user","content":"Hi"}]}"#,
        );
        let who = client_id(&request, &HeaderMap::new());
        assert_eq!(who.client.as_deref(), Some(r#"{"session_id":"9f2c-0d31"}"#));
        assert_eq!(who.session, None);

        // Nothing said is nothing stored.
        let bare = parse(r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}"#);
        assert_eq!(client_id(&bare, &HeaderMap::new()), ClientId::default());
    }

    /// A wrongly-typed `metadata` is accepted and dropped, like every other
    /// field this dialect does not understand. Nothing depends on it, so it must
    /// never be the thing that fails a request.
    #[test]
    fn a_metadata_of_the_wrong_shape_is_accepted_and_dropped() {
        for shape in [
            r#""abc""#,
            "[]",
            "42",
            "null",
            r#"{"user_id":42}"#,
            r#"{"no_user_id_here":"x"}"#,
        ] {
            let body = format!(
                r#"{{"max_tokens":16,"metadata":{shape},
                    "messages":[{{"role":"user","content":"Hi"}}]}}"#
            );
            let request = parse(&body);
            let prepared = prepare(request, &settings(), "laguna-s-2.1", target());
            assert!(
                prepared.is_ok(),
                "metadata {shape} must not fail the request"
            );
            let request = parse(&body);
            let who = client_id(&request, &HeaderMap::new());
            // A non-string user_id is still stringified; the shapes that carry
            // no user_id at all yield nothing.
            if shape == r#"{"user_id":42}"# {
                assert_eq!(who.client.as_deref(), Some("42"));
            } else {
                assert_eq!(who.client, None, "metadata {shape} names no client");
            }
        }
    }

    /// An empty id is how a client spells "not supplied" without dropping the
    /// key, and both dialects agree it is nobody.
    #[test]
    fn an_empty_user_id_is_nobody() {
        let request = parse(
            r#"{"max_tokens":16,"metadata":{"user_id":""},
                "messages":[{"role":"user","content":"Hi"}]}"#,
        );
        assert_eq!(client_id(&request, &HeaderMap::new()).client, None);
    }

    /// A client-supplied id is bounded before it is stored: the history is a
    /// file, and this is the one field a caller gets to choose the size of.
    #[test]
    fn an_overlong_client_id_is_truncated() {
        let long = "x".repeat(4096);
        let request = parse(&format!(
            r#"{{"max_tokens":16,"metadata":{{"user_id":"{long}"}},
                "messages":[{{"role":"user","content":"Hi"}}]}}"#
        ));
        let who = client_id(&request, &HeaderMap::new());
        assert_eq!(
            who.client.expect("a client id").chars().count(),
            crate::serve::types::CLIENT_ID_MAX_CHARS
        );
    }

    #[test]
    fn unknown_fields_and_block_types_are_ignored() {
        let request = prepared(
            r#"{"max_tokens":16,"metadata":{"user_id":"u"},"service_tier":"auto",
                "some_field_from_next_year":{"a":1},
                "messages":[{"role":"user","content":[
                    {"type":"text","text":"Hi","cache_control":{"type":"ephemeral"}},
                    {"type":"something_new","payload":1}]}]}"#,
        );
        assert_eq!(shape(&request.job.messages), vec!["user:Hi"]);
    }

    #[test]
    fn max_tokens_is_required() {
        let error = rejected(r#"{"messages":[{"role":"user","content":"Hi"}]}"#);
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(
            message(&error).contains("max_tokens"),
            "{}",
            message(&error)
        );
    }

    #[test]
    fn an_empty_conversation_is_rejected() {
        let error = rejected(r#"{"max_tokens":16,"messages":[]}"#);
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.body["error"]["type"], "invalid_request_error");
        assert!(
            message(&error).contains("at least one message"),
            "{}",
            message(&error)
        );

        // A system prompt is not a turn: the model would have nothing to answer.
        let error = rejected(r#"{"max_tokens":16,"system":"Be brief.","messages":[]}"#);
        assert!(
            message(&error).contains("at least one message"),
            "{}",
            message(&error)
        );
    }

    #[test]
    fn tool_definitions_are_rejected_but_an_empty_list_is_not() {
        let error = prepare(
            parse(
                r#"{"max_tokens":16,"tools":[{"name":"get_weather","input_schema":{}}],
                    "messages":[{"role":"user","content":"Hi"}]}"#,
            ),
            &tools(ToolsMode::Reject),
            "laguna-s-2.1",
            target(),
        )
        .err()
        .expect("a tools request is rejected in reject mode");
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(message(&error).contains("tool use"), "{}", message(&error));
        assert_eq!(error.body["error"]["type"], "invalid_request_error");

        let request =
            prepared(r#"{"max_tokens":16,"tools":[],"messages":[{"role":"user","content":"Hi"}]}"#);
        assert_eq!(shape(&request.job.messages), vec!["user:Hi"]);
        assert!(request.job.tools.is_empty());
    }

    /// Strip mode answers a tools-bearing request exactly as if the definitions
    /// had never been sent: same conversation, same generation settings. The
    /// model simply never calls a tool.
    #[test]
    fn strip_mode_answers_a_tools_request_as_if_it_had_none() {
        let body = |defs: &str| {
            format!(
                r#"{{"max_tokens":16,{defs}"tool_choice":{{"type":"any"}},
                    "messages":[{{"role":"user","content":"Hi"}}]}}"#
            )
        };
        let stripped = prepare(
            parse(&body(
                r#""tools":[{"name":"get_weather","input_schema":{}}],"#,
            )),
            &tools(ToolsMode::Strip),
            "laguna-s-2.1",
            target(),
        )
        .expect("strip mode accepts tool definitions");
        let toolless = prepare(
            parse(&body("")),
            &tools(ToolsMode::Strip),
            "laguna-s-2.1",
            target(),
        )
        .expect("a request with no tools prepares");
        assert_eq!(shape(&stripped.job.messages), vec!["user:Hi"]);
        assert_eq!(shape(&stripped.job.messages), shape(&toolless.job.messages));
        // Including the definitions themselves: nothing about them reaches the
        // prompt, so the `tool_choice` the request also sent means nothing.
        assert!(stripped.job.tools.is_empty());
        assert_eq!(stripped.job.tools, toolless.job.tools);
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
    }

    /// A tool call already in the history has no rendering in the debug modes:
    /// stripping it would silently delete part of the conversation, and neither
    /// mode is putting tools in the prompt for the model to have called.
    #[test]
    fn a_tool_use_block_is_rejected_in_both_debug_modes() {
        const HISTORY: &str = r#"{"max_tokens":16,"messages":[
            {"role":"assistant","content":[{"type":"tool_use","id":"t","name":"f","input":{}}]}]}"#;
        for mode in [ToolsMode::Reject, ToolsMode::Strip] {
            let error = rejected_in(HISTORY, mode);
            assert_eq!(error.status, StatusCode::BAD_REQUEST);
            assert!(message(&error).contains("tool use"), "{}", message(&error));
        }
    }

    /// Serving tools means replaying the calls the model already made. The id
    /// is dropped — the template pairs a call with its result by position — and
    /// the argument order is the client's, which the rendering depends on.
    #[test]
    fn tool_use_blocks_replay_as_assistant_calls() {
        let request = prepared_in(
            r#"{"max_tokens":16,"messages":[
                {"role":"user","content":"weather?"},
                {"role":"assistant","content":[
                    {"type":"text","text":"checking"},
                    {"type":"tool_use","id":"toolu_a","name":"get_weather",
                     "input":{"zip":"90210","units":"metric","days":3}},
                    {"type":"tool_use","id":"toolu_b","name":"now","input":{}}]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"toolu_a","content":"sunny"},
                    {"type":"tool_result","tool_use_id":"toolu_b","content":"noon"}]}]}"#,
            ToolsMode::Native,
        );
        assert_eq!(
            shape(&request.job.messages),
            vec![
                "user:weather?",
                "assistant:checking",
                "tool:sunny",
                "tool:noon"
            ]
        );
        assert_eq!(
            calls(&request.job.messages),
            vec!["get_weather(zip=\"90210\",units=\"metric\",days=3) now()"]
        );
    }

    /// An assistant turn that was nothing but a call has no text, and must not
    /// be given any: the template renders content and calls in one block.
    #[test]
    fn a_call_only_assistant_turn_has_empty_content() {
        let request = prepared_in(
            r#"{"max_tokens":16,"messages":[
                {"role":"user","content":"weather?"},
                {"role":"assistant","content":[
                    {"type":"tool_use","id":"t","name":"get_weather","input":{"zip":"90210"}}]}]}"#,
            ToolsMode::Native,
        );
        assert_eq!(
            shape(&request.job.messages),
            vec!["user:weather?", "assistant:"]
        );
    }

    /// Arguments are named values. A call whose input is anything else has no
    /// rendering, and inventing one would put a call the client never made into
    /// the prompt.
    #[test]
    fn a_tool_use_input_must_be_an_object() {
        let error = rejected_in(
            r#"{"max_tokens":16,"messages":[{"role":"assistant","content":[
                {"type":"tool_use","id":"t","name":"f","input":"zip=90210"}]}]}"#,
            ToolsMode::Native,
        );
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(message(&error).contains("object"), "{}", message(&error));

        // An absent or null input is a call with no arguments, not an error.
        for input in [r#""input":null,"#, ""] {
            let request = prepare(
                parse(&format!(
                    r#"{{"max_tokens":16,"messages":[{{"role":"assistant","content":[
                        {{"type":"tool_use","id":"t","name":"f",{input}"cache_control":null}}]}}]}}"#
                )),
                &tools(ToolsMode::Native),
                "laguna-s-2.1",
                target(),
            )
            .expect("a call with no arguments prepares");
            assert_eq!(calls(&request.job.messages), vec!["f()"]);
        }
    }

    /// Native mode passes every assistant turn's reasoning to the renderer,
    /// superseded turns included: which turns render it is the template
    /// dialect's decision (chat.rs: 3.6 drops pre-last-query reasoning, 3.8's
    /// `preserve_thinking` default keeps it), and normalization dropping any
    /// of it would overrule the template. Anthropic's own API strips
    /// non-trailing thinking; this dialect deliberately does not emulate that
    /// — it serves Qwen checkpoints, and the 3.8 card asks for preserved
    /// thinking (decisions.md "Serving").
    #[test]
    fn reasoning_is_passed_through_on_every_assistant_turn() {
        let request = prepared_in(
            r#"{"max_tokens":16,"messages":[
                {"role":"user","content":"first"},
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"old notes","signature":"x"},
                    {"type":"text","text":"answered"}]},
                {"role":"user","content":"second"},
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"fresh notes","signature":"x"},
                    {"type":"tool_use","id":"t","name":"f","input":{}}]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"t","content":"42"}]},
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"more notes","signature":"x"},
                    {"type":"text","text":"done"}]}]}"#,
            ToolsMode::Native,
        );
        assert_eq!(
            shape(&request.job.messages),
            vec![
                "user:first",
                "assistant:answered|think:old notes",
                "user:second",
                "assistant:|think:fresh notes",
                "tool:42",
                "assistant:done|think:more notes",
            ]
        );
    }

    /// The retention rule, end to end: the same normalized history rendered
    /// under each dialect. 3.6 drops pre-last-query reasoning (the superseded
    /// turn shows its answer alone); 3.8's `preserve_thinking` default keeps
    /// every turn's. This API has no preserve field of its own, so the
    /// checkpoint's template default is the whole rule here.
    #[test]
    fn the_renderer_dialect_decides_which_replayed_reasoning_survives() {
        let job = prepared_in(
            r#"{"max_tokens":16,"messages":[
                {"role":"user","content":"first"},
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"stale","signature":"x"},
                    {"type":"text","text":"answered"}]},
                {"role":"user","content":"second"}]}"#,
            ToolsMode::Native,
        )
        .job;
        let q36 = render(&job, chat::ChatDialect::Qwen36);
        assert!(!q36.contains("stale"), "{q36}");
        let q38 = render(&job, chat::ChatDialect::Qwen38);
        assert!(q38.contains("stale"), "{q38}");
    }

    /// A user turn that returns results with something else to say — a harness
    /// attaching a reminder to them — splits into tool turns and a user turn
    /// in the client's order, and the reasoning on both sides of it passes
    /// through.
    #[test]
    fn a_user_turn_that_mixes_text_with_results_keeps_the_turns_order() {
        let request = prepared_in(
            r#"{"max_tokens":16,"messages":[
                {"role":"user","content":"look it up"},
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"plan","signature":"x"},
                    {"type":"tool_use","id":"t1","name":"f","input":{}}]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"t1","content":"42"},
                    {"type":"text","text":"<system-reminder>be brief</system-reminder>"}]},
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"still planning","signature":"x"},
                    {"type":"tool_use","id":"t2","name":"g","input":{}}]}]}"#,
            ToolsMode::Native,
        );
        assert_eq!(
            shape(&request.job.messages),
            vec![
                "user:look it up",
                "assistant:|think:plan",
                "tool:42",
                "user:<system-reminder>be brief</system-reminder>",
                "assistant:|think:still planning",
            ]
        );
    }

    /// Results are paired with calls by position, because that is all the
    /// prompt can express. A client that returns them in completion order
    /// identifies each one anyway, so they are put back in the calls' order
    /// before the ids are dropped.
    #[test]
    fn out_of_order_tool_results_are_paired_by_id() {
        const CALLS: &str = r#"{"role":"assistant","content":[
            {"type":"tool_use","id":"t1","name":"first","input":{}},
            {"type":"tool_use","id":"t2","name":"second","input":{}},
            {"type":"tool_use","id":"t3","name":"third","input":{}}]}"#;
        let conversation = |results: &str| {
            format!(
                r#"{{"max_tokens":16,"messages":[
                    {{"role":"user","content":"go"}},{CALLS},
                    {{"role":"user","content":[{results}]}}]}}"#
            )
        };

        let request = prepared_in(
            &conversation(
                r#"{"type":"tool_result","tool_use_id":"t3","content":"C"},
                   {"type":"tool_result","tool_use_id":"t1","content":"A"},
                   {"type":"tool_result","tool_use_id":"t2","content":"B"}"#,
            ),
            ToolsMode::Native,
        );
        assert_eq!(
            shape(&request.job.messages),
            vec!["user:go", "assistant:", "tool:A", "tool:B", "tool:C"]
        );

        // One result nobody can place leaves the whole turn in the client's
        // order: a partial reordering would pair worse than not trying.
        let request = prepared_in(
            &conversation(
                r#"{"type":"tool_result","tool_use_id":"t3","content":"C"},
                   {"type":"tool_result","content":"A"},
                   {"type":"tool_result","tool_use_id":"t2","content":"B"}"#,
            ),
            ToolsMode::Native,
        );
        assert_eq!(
            shape(&request.job.messages),
            vec!["user:go", "assistant:", "tool:C", "tool:A", "tool:B"]
        );

        // So does an id that names no call in the turn just before.
        let request = prepared_in(
            &conversation(
                r#"{"type":"tool_result","tool_use_id":"t3","content":"C"},
                   {"type":"tool_result","tool_use_id":"elsewhere","content":"A"}"#,
            ),
            ToolsMode::Native,
        );
        assert_eq!(
            shape(&request.job.messages),
            vec!["user:go", "assistant:", "tool:C", "tool:A"]
        );
    }

    /// Reordering moves the results among themselves and nothing else: text the
    /// client wrote between them keeps its place in the turn.
    #[test]
    fn pairing_results_by_id_leaves_the_turns_text_where_it_was() {
        let request = prepared_in(
            r#"{"max_tokens":16,"messages":[
                {"role":"user","content":"go"},
                {"role":"assistant","content":[
                    {"type":"tool_use","id":"t1","name":"first","input":{}},
                    {"type":"tool_use","id":"t2","name":"second","input":{}}]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"t2","content":"B"},
                    {"type":"text","text":"between"},
                    {"type":"tool_result","tool_use_id":"t1","content":"A"}]}]}"#,
            ToolsMode::Native,
        );
        assert_eq!(
            shape(&request.job.messages),
            vec!["user:go", "assistant:", "tool:A", "user:between", "tool:B"]
        );
    }

    /// Replaying reasoning is part of serving tools. A server told not to serve
    /// them renders the history exactly as it did before tools existed, which
    /// is what makes `--tools-mode strip` a usable A/B against that behavior.
    #[test]
    fn the_debug_modes_replay_no_reasoning() {
        const HISTORY: &str = r#"{"max_tokens":16,"messages":[
            {"role":"user","content":"2+2?"},
            {"role":"assistant","content":[
                {"type":"thinking","thinking":"add them","signature":"x"},
                {"type":"text","text":"4"}]}]}"#;
        // Native passes it through.
        assert_eq!(
            shape(&prepared_in(HISTORY, ToolsMode::Native).job.messages),
            vec!["user:2+2?", "assistant:4|think:add them"]
        );
        for mode in [ToolsMode::Reject, ToolsMode::Strip] {
            assert_eq!(
                shape(&prepared_in(HISTORY, mode).job.messages),
                vec!["user:2+2?", "assistant:4"],
                "{mode}"
            );
        }
    }

    /// The blocks this server cannot read still have to round-trip: a client
    /// that sends back what it was given must not be told one of them is an
    /// error, and redacted reasoning contributes nothing to the prompt.
    #[test]
    fn redacted_thinking_is_accepted_and_contributes_nothing() {
        let request = prepared_in(
            r#"{"max_tokens":16,"messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","content":[
                    {"type":"redacted_thinking","data":"EtcBCkYIBBgCKkDx"},
                    {"type":"text","text":"hello"}]}]}"#,
            ToolsMode::Native,
        );
        assert_eq!(
            shape(&request.job.messages),
            vec!["user:hi", "assistant:hello"]
        );

        // Even alone in the turn, and even in native mode where reasoning is
        // passed through: there is nothing to pass.
        let request = prepared_in(
            r#"{"max_tokens":16,"messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","content":[
                    {"type":"redacted_thinking","data":"EtcBCkYIBBgCKkDx"}]}]}"#,
            ToolsMode::Native,
        );
        assert_eq!(shape(&request.job.messages), vec!["user:hi", "assistant:"]);
    }

    /// Server tools are run by whoever hosts the model. This server hosts none,
    /// so a conversation that already contains one of their calls cannot be
    /// continued, tools mode notwithstanding.
    #[test]
    fn a_server_tool_use_block_is_rejected_in_every_mode() {
        const HISTORY: &str = r#"{"max_tokens":16,"messages":[
            {"role":"assistant","content":[
                {"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}]}]}"#;
        for mode in [ToolsMode::Native, ToolsMode::Reject, ToolsMode::Strip] {
            let error = rejected_in(HISTORY, mode);
            assert_eq!(error.status, StatusCode::BAD_REQUEST);
            assert!(
                message(&error).contains("server tool"),
                "{}",
                message(&error)
            );
        }
    }

    /// Definitions reach the engine in the OpenAI object shape, which is what
    /// the prompt renders. `description` is omitted rather than nulled when the
    /// client sent none — the rendering is byte-exact against the fork's.
    #[test]
    fn tool_definitions_are_translated_to_the_openai_shape() {
        let request = prepared_in(
            r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],
                "tools":[
                    {"name":"get_weather","description":"Look up the weather",
                     "input_schema":{"type":"object","properties":{"zip":{"type":"string"}}}},
                    {"name":"now","input_schema":{"type":"object"}}]}"#,
            ToolsMode::Native,
        );
        assert_eq!(
            request.job.tools,
            vec![
                json!({"type":"function","function":{
                    "name":"get_weather","description":"Look up the weather",
                    "parameters":{"type":"object","properties":{"zip":{"type":"string"}}}}}),
                json!({"type":"function","function":{
                    "name":"now","parameters":{"type":"object"}}}),
            ]
        );
        assert!(
            request.job.tools[1]["function"]
                .get("description")
                .is_none(),
            "a tool with no description carries no description key"
        );
        // The key order is prompt bytes, and a `Value` comparison cannot see it:
        // two objects built in different orders are equal. This is the only
        // assertion here that would fail if the keys were reordered.
        assert_eq!(
            chat::tojson_text(&request.job.tools[0]).expect("the definition renders"),
            r#"{"type": "function", "function": {"name": "get_weather", "description": "Look up the weather", "parameters": {"type": "object", "properties": {"zip": {"type": "string"}}}}}"#
        );

        let error = rejected_in(
            r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}],
                "tools":[{"description":"nameless"}]}"#,
            ToolsMode::Native,
        );
        assert!(message(&error).contains("name"), "{}", message(&error));
    }

    /// `tool_choice` can turn the tools off for a turn, but it cannot turn them
    /// into a guarantee: without a constrained sampler, "you must call one" is a
    /// promise this server would break silently.
    #[test]
    fn tool_choice_may_disable_tools_but_never_force_them() {
        let body = |choice: &str| {
            format!(
                r#"{{"max_tokens":16,"tool_choice":{choice},
                    "tools":[{{"name":"get_weather","input_schema":{{}}}}],
                    "messages":[{{"role":"user","content":"Hi"}}]}}"#
            )
        };

        for choice in [
            r#"{"type":"auto"}"#,
            r#"{"type":"auto","disable_parallel_tool_use":true}"#,
        ] {
            let request = prepared_in(&body(choice), ToolsMode::Native);
            assert_eq!(request.job.tools.len(), 1, "{choice}");
        }
        // Absent is auto.
        let request = prepared_in(
            r#"{"max_tokens":16,"tools":[{"name":"get_weather","input_schema":{}}],
                "messages":[{"role":"user","content":"Hi"}]}"#,
            ToolsMode::Native,
        );
        assert_eq!(request.job.tools.len(), 1);

        // None renders the turn as a tool-free request.
        let request = prepared_in(&body(r#"{"type":"none"}"#), ToolsMode::Native);
        assert!(request.job.tools.is_empty());

        for choice in [
            r#"{"type":"any"}"#,
            r#"{"type":"tool","name":"get_weather"}"#,
        ] {
            let error = rejected_in(&body(choice), ToolsMode::Native);
            assert_eq!(error.status, StatusCode::BAD_REQUEST, "{choice}");
            assert!(
                message(&error).contains("tool_choice"),
                "{choice}: {}",
                message(&error)
            );
        }
    }

    /// A count that left the tools out would be wrong by exactly the part a
    /// harness asks about: a tool list is often the larger half of its prompt.
    /// The section's exact bytes are `chat.rs`'s contract with the fork's
    /// template; all the count can show is that the definitions are in there.
    #[test]
    fn the_counted_prompt_includes_the_tools_rendering() {
        let settings = tools(ToolsMode::Native);
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/tokenizer.json");
        let tokenizer =
            crate::tokenizer::LagunaTokenizer::from_file(path).expect("load reference tokenizer");
        let target = crate::serve::types::Target::official(crate::hub::Model::Qwen35BA3B);
        let with_tools = counted_tokens(
            &parse(
                r#"{"messages":[{"role":"user","content":"Hi"}],
                    "tools":[{"name":"get_weather","description":"Look it up",
                              "input_schema":{"type":"object"}}]}"#,
            ),
            &settings,
            &tokenizer,
            target,
        )
        .expect("the prompt renders");
        let without = counted_tokens(
            &parse(r#"{"messages":[{"role":"user","content":"Hi"}]}"#),
            &settings,
            &tokenizer,
            target,
        )
        .expect("the prompt renders");
        assert!(with_tools > without, "{with_tools} vs {without}");

        // The turn's own opt-out applies here too, or the count would describe
        // a prompt the generation request never sends.
        let disabled = counted_tokens(
            &parse(
                r#"{"messages":[{"role":"user","content":"Hi"}],"tool_choice":{"type":"none"},
                    "tools":[{"name":"get_weather","input_schema":{"type":"object"}}]}"#,
            ),
            &settings,
            &tokenizer,
            target,
        )
        .expect("the prompt renders");
        assert_eq!(disabled, without);
    }

    #[test]
    fn image_blocks_are_rejected_with_a_reason() {
        let error = rejected(
            r#"{"max_tokens":16,"messages":[{"role":"user","content":[
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"x"}}]}]}"#,
        );
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(message(&error).contains("text-only"), "{}", message(&error));
    }

    #[test]
    fn thinking_param_shapes_map_to_budgets() {
        let settings = settings();
        let enabled = ThinkingParam {
            kind: "enabled".into(),
            budget_tokens: Some(2048),
        };
        assert_eq!(
            resolve_thinking(Some(&enabled), &settings, Some(4096)).unwrap(),
            (true, Some(2048))
        );

        // Adaptive means the model decides, so a budget alongside it is not one.
        let adaptive = ThinkingParam {
            kind: "adaptive".into(),
            budget_tokens: Some(2048),
        };
        assert_eq!(
            resolve_thinking(Some(&adaptive), &settings, Some(4096)).unwrap(),
            (true, None)
        );
        let mut budgeted = settings.clone();
        budgeted.thinking_budget = Some(512);
        assert_eq!(
            resolve_thinking(Some(&adaptive), &budgeted, Some(4096)).unwrap(),
            (true, Some(512)),
        );

        let disabled = ThinkingParam {
            kind: "disabled".into(),
            budget_tokens: None,
        };
        assert_eq!(
            resolve_thinking(Some(&disabled), &settings, Some(4096)).unwrap(),
            (false, None)
        );

        // Absent: the server's own configuration decides.
        assert_eq!(
            resolve_thinking(None, &settings, Some(4096)).unwrap(),
            (true, None)
        );
        let mut off = settings.clone();
        off.thinking_force = false;
        assert_eq!(
            resolve_thinking(None, &off, Some(4096)).unwrap(),
            (false, None)
        );
        let mut budgeted = settings.clone();
        budgeted.thinking_budget = Some(512);
        assert_eq!(
            resolve_thinking(None, &budgeted, Some(4096)).unwrap(),
            (true, Some(512))
        );
    }

    /// Every budget the hosted API accepts (`budget_tokens < max_tokens`) has to
    /// produce an answer here. A ceiling the reasoning schedule cannot express
    /// is lowered to one it can, or dropped for uncapped reasoning — never
    /// refused, and never a 500 later on.
    #[test]
    fn an_unservable_thinking_budget_is_degraded_rather_than_refused() {
        let settings = settings();
        let budget = |n| ThinkingParam {
            kind: "enabled".into(),
            budget_tokens: Some(n),
        };

        // Comfortably inside the reply: honored exactly.
        assert_eq!(
            resolve_thinking(Some(&budget(2048)), &settings, Some(4096)).unwrap(),
            (true, Some(2048)),
        );

        // Too large for the reply to conclude in: lowered to the largest ceiling
        // that leaves room, which is still a real ceiling.
        let (enabled, lowered) =
            resolve_thinking(Some(&budget(10_000)), &settings, Some(4096)).unwrap();
        let lowered = lowered.expect("a 4096-token reply has room for some ceiling");
        assert!(enabled && lowered < 10_000, "{lowered}");
        assert_eq!(
            crate::generate::feasible_think_budget(10_000, 4096),
            Some(lowered)
        );

        // A reply too short for any expressible ceiling reasons uncapped. This is
        // the combination the hosted API accepts and this server used to reject.
        assert_eq!(
            resolve_thinking(Some(&budget(99)), &settings, Some(100)).unwrap(),
            (true, None),
        );
        assert_eq!(
            resolve_thinking(Some(&budget(10_000)), &settings, Some(100)).unwrap(),
            (true, None),
        );
        // A zero budget expresses no ceiling at all.
        assert_eq!(
            resolve_thinking(Some(&budget(0)), &settings, Some(4096)).unwrap(),
            (true, None)
        );
    }

    /// The whole path a request takes, since the budget is what the engine is
    /// handed and an error anywhere along it would surface as a 400.
    #[test]
    fn a_request_whose_budget_cannot_be_expressed_still_prepares() {
        let request = prepared(
            r#"{"max_tokens":100,"thinking":{"type":"enabled","budget_tokens":99},
                "messages":[{"role":"user","content":"Hi"}]}"#,
        );
        assert!(request.job.enable_thinking);
        assert_eq!(request.job.max_think, None);
        assert_eq!(request.job.max_tokens, 100);
    }

    /// Sampling resolves request over server config over the model card's
    /// mode-keyed recommendation: with nothing pinned, a thinking request
    /// defaults to 1.0/20/0.95 and a `thinking: disabled` one to 0.7/20/0.80.
    #[test]
    fn sampling_falls_back_to_the_resolved_modes_defaults() {
        let thinking = prepared(r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}"#);
        assert!(thinking.job.enable_thinking);
        assert_eq!(thinking.job.sampling.temperature, 1.0);
        assert_eq!(thinking.job.sampling.top_k, 20);
        assert_eq!(thinking.job.sampling.top_p, 0.95);
        assert_eq!(thinking.job.sampling.seed, SamplerOptions::default().seed);

        let instruct = prepared(
            r#"{"max_tokens":16,"thinking":{"type":"disabled"},
                "messages":[{"role":"user","content":"Hi"}]}"#,
        );
        assert!(!instruct.job.enable_thinking);
        assert_eq!(instruct.job.sampling.temperature, 0.7);
        assert_eq!(instruct.job.sampling.top_k, 20);
        assert_eq!(instruct.job.sampling.top_p, 0.80);

        // A server-pinned value beats the mode default; the request beats both.
        let mut pinned = settings();
        pinned.temperature = Some(0.5);
        let configured = prepare(
            parse(
                r#"{"max_tokens":16,"thinking":{"type":"disabled"},
                    "messages":[{"role":"user","content":"Hi"}]}"#,
            ),
            &pinned,
            "m",
            target(),
        )
        .unwrap();
        assert_eq!(configured.job.sampling.temperature, 0.5);
        assert_eq!(
            configured.job.sampling.top_p, 0.80,
            "unpinned keys keep the mode default"
        );

        let asked = prepared(
            r#"{"max_tokens":16,"temperature":0.2,"top_p":0.8,"top_k":5,
                "stop_sequences":["STOP"],"messages":[{"role":"user","content":"Hi"}]}"#,
        );
        assert_eq!(asked.job.sampling.temperature, 0.2);
        assert_eq!(asked.job.sampling.top_p, 0.8);
        assert_eq!(asked.job.sampling.top_k, 5);
        assert_eq!(asked.job.stop_sequences, vec!["STOP".to_string()]);
    }

    /// This API defines no presence-penalty field, so a request cannot set one:
    /// the value comes from the server setting, or from the target
    /// checkpoint's card for the resolved mode when nothing is pinned.
    #[test]
    fn the_presence_penalty_takes_the_server_default_or_the_card() {
        let body = r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}"#;
        // Thinking on, and the 3.8 card asks for no penalty there.
        assert_eq!(prepared(body).job.sampling.presence_penalty, 0.0);
        let a3b = crate::serve::types::Target::official(crate::hub::Model::Qwen35BA3B);
        let on_a3b = prepare(parse(body), &settings(), "m", a3b).unwrap();
        assert_eq!(on_a3b.job.sampling.presence_penalty, 1.5);

        // Thinking off: every card asks for 1.5.
        let instruct = prepared(
            r#"{"max_tokens":16,"thinking":{"type":"disabled"},
                "messages":[{"role":"user","content":"Hi"}]}"#,
        );
        assert_eq!(instruct.job.sampling.presence_penalty, 1.5);

        let mut pinned = settings();
        pinned.presence_penalty = Some(0.25);
        let configured = prepare(parse(body), &pinned, "m", a3b).unwrap();
        assert_eq!(configured.job.sampling.presence_penalty, 0.25);
    }

    /// `prepare` echoes the `model` string it is handed, and the handler is what
    /// decides that string: it resolves the client's name to a checkpoint this
    /// server serves and writes the CANONICAL spelling back before preparing, so
    /// a name that could not resolve never reaches here (it was a 400). The two
    /// cases left for `prepare` are the canonical name and an absent field.
    #[test]
    fn the_model_prepare_is_handed_is_the_one_echoed() {
        let named = prepared(
            r#"{"model":"Qwen3.6-27B","max_tokens":16,
                "messages":[{"role":"user","content":"Hi"}]}"#,
        );
        assert_eq!(named.model, "Qwen3.6-27B");
        let unnamed = prepared(r#"{"max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}"#);
        assert_eq!(unnamed.model, "laguna-s-2.1");
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

    fn call(name: &str, arguments: &str) -> CompletedToolCall {
        CompletedToolCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
        }
    }

    #[test]
    fn the_response_body_carries_both_blocks_and_the_full_usage() {
        let body = message_body(
            "msg_1",
            "claude-x",
            &completion("reasoning", "answer", StopKind::EndTurn),
        );
        assert_eq!(body["type"], "message");
        assert_eq!(body["role"], "assistant");
        assert_eq!(body["model"], "claude-x");
        assert_eq!(body["content"][0]["type"], "thinking");
        assert_eq!(body["content"][0]["thinking"], "reasoning");
        assert_eq!(body["content"][0]["signature"], THINKING_SIGNATURE);
        assert_eq!(body["content"][1]["type"], "text");
        assert_eq!(body["content"][1]["text"], "answer");
        assert_eq!(body["stop_reason"], "end_turn");
        assert_eq!(body["stop_sequence"], Value::Null);
        assert_eq!(body["usage"]["input_tokens"], 100);
        assert_eq!(body["usage"]["cache_read_input_tokens"], 40);
        assert_eq!(body["usage"]["cache_creation_input_tokens"], 60);
        assert_eq!(body["usage"]["output_tokens"], 25);
        assert_eq!(
            body["usage"]["output_tokens_details"]["thinking_tokens"],
            10
        );
    }

    #[test]
    fn a_turn_without_thinking_has_only_a_text_block() {
        let body = message_body("msg_1", "m", &completion("", "answer", StopKind::EndTurn));
        assert_eq!(body["content"].as_array().unwrap().len(), 1);
        assert_eq!(body["content"][0]["type"], "text");
    }

    /// Calls come after the text the model wrote before them, each with its own
    /// id and its arguments parsed back into an object — in the order the model
    /// wrote them, which is the order the next turn replays them in.
    #[test]
    fn the_response_body_carries_tool_use_blocks_after_the_text() {
        let mut completion = completion("reasoning", "checking", StopKind::ToolUse);
        completion.tool_calls = vec![
            call("get_weather", r#"{"zip":"90210","days":3}"#),
            call("now", "{}"),
        ];
        let body = message_body("msg_1", "claude-x", &completion);

        assert_eq!(body["content"][0]["type"], "thinking");
        assert_eq!(body["content"][1]["type"], "text");
        assert_eq!(body["content"][1]["text"], "checking");
        assert_eq!(body["content"][2]["type"], "tool_use");
        assert_eq!(body["content"][2]["name"], "get_weather");
        assert_eq!(body["content"][2]["input"], json!({"zip":"90210","days":3}));
        assert_eq!(body["content"][3]["type"], "tool_use");
        assert_eq!(body["content"][3]["name"], "now");
        assert_eq!(body["content"][3]["input"], json!({}));
        assert_eq!(body["stop_reason"], "tool_use");
        assert_eq!(body["stop_sequence"], Value::Null);

        // Ids are this process's own, and no two calls share one.
        let first = body["content"][2]["id"].as_str().expect("an id string");
        let second = body["content"][3]["id"].as_str().expect("an id string");
        assert!(first.starts_with("toolu_") && first.len() == 14, "{first}");
        assert_ne!(first, second);
    }

    #[test]
    fn stop_reasons_map_and_a_stop_sequence_reports_itself() {
        let body = message_body("m", "m", &completion("", "a", StopKind::MaxTokens));
        assert_eq!(body["stop_reason"], "max_tokens");
        let body = message_body(
            "m",
            "m",
            &completion("", "a", StopKind::StopSequence("END".into())),
        );
        assert_eq!(body["stop_reason"], "stop_sequence");
        assert_eq!(body["stop_sequence"], "END");
    }

    #[test]
    fn the_event_sequence_with_thinking_is_the_documented_one() {
        let mut stream = MessageStream::new("msg_1".into(), "claude-x".into());
        let frames = encode_all(
            &mut stream,
            vec![
                EngineEvent::Start {
                    input_tokens: 100,
                    cached_tokens: 40,
                },
                EngineEvent::Thinking("think".into()),
                EngineEvent::Text("ans".into()),
                EngineEvent::Text("wer".into()),
                EngineEvent::Done {
                    stop: StopKind::EndTurn,
                    output_tokens: 25,
                    thinking_tokens: 10,
                },
            ],
        );
        assert_eq!(
            names(&frames),
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta", // thinking_delta
                "content_block_delta", // signature_delta
                "content_block_stop",
                "content_block_start",
                "content_block_delta", // text_delta
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );

        let start = payload(&frames[0]);
        assert_eq!(start["type"], "message_start");
        assert_eq!(start["message"]["id"], "msg_1");
        assert_eq!(start["message"]["model"], "claude-x");
        assert_eq!(start["message"]["usage"]["input_tokens"], 100);
        assert_eq!(start["message"]["usage"]["cache_read_input_tokens"], 40);
        assert_eq!(start["message"]["usage"]["output_tokens"], 1);

        assert_eq!(payload(&frames[1])["content_block"]["type"], "thinking");
        assert_eq!(payload(&frames[1])["index"], 0);
        assert_eq!(payload(&frames[2])["delta"]["type"], "thinking_delta");
        assert_eq!(payload(&frames[2])["delta"]["thinking"], "think");
        assert_eq!(payload(&frames[3])["delta"]["type"], "signature_delta");
        assert_eq!(
            payload(&frames[3])["delta"]["signature"],
            THINKING_SIGNATURE
        );
        assert_eq!(payload(&frames[4])["index"], 0);

        assert_eq!(payload(&frames[5])["content_block"]["type"], "text");
        assert_eq!(payload(&frames[5])["index"], 1);
        assert_eq!(payload(&frames[6])["delta"]["type"], "text_delta");
        assert_eq!(payload(&frames[6])["delta"]["text"], "ans");
        assert_eq!(payload(&frames[8])["index"], 1);

        // The closing usage is cumulative for the whole turn, not a delta.
        let delta = payload(&frames[9]);
        assert_eq!(delta["delta"]["stop_reason"], "end_turn");
        assert_eq!(delta["delta"]["stop_sequence"], Value::Null);
        assert_eq!(delta["usage"]["output_tokens"], 25);
        assert_eq!(
            delta["usage"]["output_tokens_details"]["thinking_tokens"],
            10
        );
        assert_eq!(payload(&frames[10])["type"], "message_stop");
    }

    #[test]
    fn a_turn_without_thinking_puts_text_at_index_zero() {
        let mut stream = MessageStream::new("msg_1".into(), "m".into());
        let frames = encode_all(
            &mut stream,
            vec![
                EngineEvent::Start {
                    input_tokens: 5,
                    cached_tokens: 0,
                },
                EngineEvent::Text("hi".into()),
                EngineEvent::Done {
                    stop: StopKind::EndTurn,
                    output_tokens: 2,
                    thinking_tokens: 0,
                },
            ],
        );
        assert_eq!(
            names(&frames),
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(payload(&frames[1])["content_block"]["type"], "text");
        assert_eq!(payload(&frames[1])["index"], 0);
        assert_eq!(payload(&frames[3])["index"], 0);
    }

    #[test]
    fn empty_thinking_deltas_never_open_a_thinking_block() {
        let mut stream = MessageStream::new("msg_1".into(), "m".into());
        let frames = encode_all(
            &mut stream,
            vec![
                EngineEvent::Start {
                    input_tokens: 5,
                    cached_tokens: 0,
                },
                // The decode stream withholds partial UTF-8, so a thinking
                // token can arrive carrying no text at all.
                EngineEvent::Thinking(String::new()),
                EngineEvent::Text("hi".into()),
                EngineEvent::Done {
                    stop: StopKind::EndTurn,
                    output_tokens: 2,
                    thinking_tokens: 1,
                },
            ],
        );
        assert_eq!(payload(&frames[1])["content_block"]["type"], "text");
        assert_eq!(payload(&frames[1])["index"], 0);
    }

    #[test]
    fn a_turn_that_produced_no_text_still_emits_an_empty_text_block() {
        let mut stream = MessageStream::new("msg_1".into(), "m".into());
        let frames = encode_all(
            &mut stream,
            vec![
                EngineEvent::Start {
                    input_tokens: 5,
                    cached_tokens: 0,
                },
                EngineEvent::Thinking("hm".into()),
                EngineEvent::Done {
                    stop: StopKind::MaxTokens,
                    output_tokens: 9,
                    thinking_tokens: 9,
                },
            ],
        );
        assert_eq!(
            names(&frames),
            vec![
                "message_start",
                "content_block_start", // thinking, index 0
                "content_block_delta",
                "content_block_delta", // signature
                "content_block_stop",
                "content_block_start", // empty text, index 1
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(payload(&frames[5])["content_block"]["type"], "text");
        assert_eq!(payload(&frames[5])["index"], 1);
        assert_eq!(payload(&frames[7])["delta"]["stop_reason"], "max_tokens");
    }

    /// A tool-call turn: each call is its own block, opened by the call rather
    /// than by its first argument fragment, and closed before the next one can
    /// start. The fragments go out verbatim — concatenated they are the
    /// arguments object, and re-encoding them here would only risk changing it.
    #[test]
    fn the_event_sequence_for_a_tool_call_turn_is_the_documented_one() {
        let mut stream = MessageStream::new("msg_1".into(), "claude-x".into());
        let frames = encode_all(
            &mut stream,
            vec![
                EngineEvent::Start {
                    input_tokens: 100,
                    cached_tokens: 40,
                },
                EngineEvent::Thinking("think".into()),
                EngineEvent::Text("checking".into()),
                EngineEvent::ToolCallStart {
                    name: "get_weather".into(),
                },
                EngineEvent::ToolCallDelta(r#"{"zip":"#.into()),
                EngineEvent::ToolCallDelta(r#""90210"}"#.into()),
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
        assert_eq!(
            names(&frames),
            vec![
                "message_start",
                "content_block_start", // thinking, index 0
                "content_block_delta", // thinking_delta
                "content_block_delta", // signature_delta
                "content_block_stop",
                "content_block_start", // text, index 1
                "content_block_delta", // text_delta
                "content_block_stop",  // closed by the first call opening
                "content_block_start", // tool_use, index 2
                "content_block_delta", // input_json_delta
                "content_block_delta",
                "content_block_stop",
                "content_block_start", // tool_use, index 3
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );

        assert_eq!(payload(&frames[7])["index"], 1);
        let start = payload(&frames[8]);
        assert_eq!(start["index"], 2);
        assert_eq!(start["content_block"]["type"], "tool_use");
        assert_eq!(start["content_block"]["name"], "get_weather");
        assert_eq!(start["content_block"]["input"], json!({}));
        let id = start["content_block"]["id"]
            .as_str()
            .expect("an id string")
            .to_string();
        assert!(id.starts_with("toolu_") && id.len() == 14, "{id}");

        assert_eq!(payload(&frames[9])["delta"]["type"], "input_json_delta");
        assert_eq!(payload(&frames[9])["delta"]["partial_json"], r#"{"zip":"#);
        assert_eq!(payload(&frames[10])["delta"]["partial_json"], r#""90210"}"#);
        assert_eq!(payload(&frames[11])["index"], 2);

        let second = payload(&frames[12]);
        assert_eq!(second["index"], 3);
        assert_eq!(second["content_block"]["name"], "now");
        assert_ne!(second["content_block"]["id"], id.as_str());
        assert_eq!(payload(&frames[13])["delta"]["partial_json"], "{}");
        assert_eq!(payload(&frames[14])["index"], 3);

        assert_eq!(payload(&frames[15])["delta"]["stop_reason"], "tool_use");
        assert_eq!(payload(&frames[15])["usage"]["output_tokens"], 25);
    }

    /// A turn that called tools instead of answering carries no text block. The
    /// hosted API emits none for one and refuses an empty one on the way in, so
    /// a client replaying this turn would be handing its next request a block
    /// it gets rejected for.
    #[test]
    fn a_call_only_turn_emits_no_text_block() {
        let mut stream = MessageStream::new("msg_1".into(), "m".into());
        let frames = encode_all(
            &mut stream,
            vec![
                EngineEvent::Start {
                    input_tokens: 5,
                    cached_tokens: 0,
                },
                EngineEvent::ToolCallStart { name: "now".into() },
                EngineEvent::ToolCallDelta("{}".into()),
                EngineEvent::ToolCallEnd,
                EngineEvent::Done {
                    stop: StopKind::ToolUse,
                    output_tokens: 9,
                    thinking_tokens: 0,
                },
            ],
        );
        assert_eq!(
            names(&frames),
            vec![
                "message_start",
                "content_block_start", // tool_use, index 0
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(payload(&frames[1])["content_block"]["type"], "tool_use");
        assert_eq!(payload(&frames[1])["index"], 0);

        // Which is also what the non-streaming body says about the same turn:
        // the two surfaces describe one message.
        let mut completion = completion("", "", StopKind::ToolUse);
        completion.tool_calls = vec![call("now", "{}")];
        let body = message_body("msg_1", "m", &completion);
        assert_eq!(body["content"].as_array().unwrap().len(), 1);
        assert_eq!(body["content"][0]["type"], "tool_use");
    }

    #[test]
    fn a_mid_stream_engine_error_ends_the_stream_with_an_error_event() {
        let mut stream = MessageStream::new("msg_1".into(), "m".into());
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
                // Never reached: the encoder reports the stream complete.
                EngineEvent::Text("unreachable".into()),
            ],
        );
        assert_eq!(names(&frames), vec!["message_start", "error"]);
        let error = payload(&frames[1]);
        assert_eq!(error["type"], "error");
        assert_eq!(error["error"]["type"], "api_error");
        assert_eq!(error["error"]["message"], "Metal command buffer failed");
    }

    /// The engine says whose fault a failure was; the envelope follows that flag
    /// rather than anything about the wording of the message.
    #[test]
    fn the_engines_fault_flag_picks_the_error_envelope() {
        let mut stream = MessageStream::new("msg_1".into(), "m".into());
        let frames = encode_all(
            &mut stream,
            vec![EngineEvent::Error {
                message: "the prompt exceeds the 131072-token context".into(),
                request_fault: true,
            }],
        );
        assert_eq!(
            payload(&frames[0])["error"]["type"],
            "invalid_request_error"
        );

        assert_eq!(
            engine_error("anything at all", true).status,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            engine_error("anything at all", true).body["error"]["type"],
            "invalid_request_error"
        );
        // A server failure whose message happens to mention the context is still
        // a server failure.
        let server = engine_error("the context buffer could not be allocated", false);
        assert_eq!(server.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(server.body["error"]["type"], "api_error");
    }

    #[test]
    fn the_stream_never_ends_with_a_done_sentinel() {
        let mut stream = MessageStream::new("msg_1".into(), "m".into());
        let frames = encode_all(
            &mut stream,
            vec![
                EngineEvent::Start {
                    input_tokens: 1,
                    cached_tokens: 0,
                },
                EngineEvent::Text("x".into()),
                EngineEvent::Done {
                    stop: StopKind::EndTurn,
                    output_tokens: 1,
                    thinking_tokens: 0,
                },
            ],
        );
        // The OpenAI dialect's `[DONE]` has no counterpart here.
        assert!(
            frames
                .iter()
                .all(|frame| frame.data != "[DONE]" && frame.name.is_some())
        );
    }
}
