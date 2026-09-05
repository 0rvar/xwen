//! Job protocol between the HTTP handlers and the inference engine thread.
//!
//! Handlers normalize either API's request into a [`GenerationJob`] and submit it into the
//! bounded [`super::queue::JobQueue`]; the engine streams [`EngineEvent`]s back over the
//! job's tokio channel with a bounded, deadline-guarded send. A failed or timed-out send means the
//! client is gone and the engine aborts the generation. Abandonment travels the other way
//! through the job's [`Cancel`] token: a [`CancelGuard`] on the HTTP side reports a
//! departed client, and the engine stamps its own deadline and the server's shutdown into
//! the same token, first reason wins.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use crate::batch::{BatchRequest, BatchResponse};
use crate::hub::Model;
use crate::sampler::SamplerOptions;

/// Which API a request arrived on. Never affects generation — the two dialects
/// submit the same job — so it exists to say who asked, not what to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    Anthropic,
    OpenAi,
    /// This engine's own endpoint, which spells the capabilities the two compat
    /// dialects have no field for.
    Native,
}

impl Dialect {
    pub fn label(self) -> &'static str {
        match self {
            Dialect::Anthropic => "anthropic",
            Dialect::OpenAi => "openai",
            Dialect::Native => "native",
        }
    }
}

/// What a client says it is, for a history that can answer "which session was
/// that". Both values are opaque here: whatever the client sent, bounded, and
/// never parsed at the point it is stored — the shape of the body id has
/// already changed once between Claude Code releases, and a reader can pick a
/// session out of an old string long after the writer has stopped knowing how.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClientId {
    /// The request body's own identifier: the Anthropic dialect's
    /// `metadata.user_id`, the OpenAI dialect's `user`. Undocumented, and its
    /// shape is not promised.
    pub client: Option<String>,
    /// The `x-claude-code-session-id` header, which is the documented
    /// per-session identifier. `x-claude-code-agent-id` is deliberately not
    /// read: it rides only subagent requests, so keying on it would split one
    /// session into a row per agent.
    pub session: Option<String>,
}

/// The longest client-supplied identifier that is stored. Nothing legitimate
/// comes close — a session uuid is 36 characters — and the history is a file a
/// hostile client would otherwise get to grow a request at a time.
pub const CLIENT_ID_MAX_CHARS: usize = 128;

impl ClientId {
    /// Both values as the client sent them, each cut to
    /// [`CLIENT_ID_MAX_CHARS`] on a character boundary. An empty string is how
    /// a client spells "not supplied" without dropping the key, and it is
    /// normalized here rather than in each dialect, so that the two cannot
    /// drift into disagreeing about what an empty id means.
    pub fn new(client: Option<String>, session: Option<String>) -> Self {
        Self {
            client: bound(client),
            session: bound(session),
        }
    }
}

fn bound(value: Option<String>) -> Option<String> {
    value
        .filter(|value| !value.is_empty())
        .map(|value| value.chars().take(CLIENT_ID_MAX_CHARS).collect())
}

/// Who a job belongs to, for correlating the events one request produces.
///
/// The id is assigned once, at submit, and is monotonic for the life of the
/// process; it never reaches the client and never appears in a log line, so
/// nothing outside the server depends on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestOrigin {
    pub id: u64,
    pub dialect: Dialect,
    /// Whether the client asked for its reply as a stream.
    pub streaming: bool,
    /// The body identifier the client sent, when it sent one.
    pub client: Option<String>,
    /// The session identifier the client's headers carried, when they did.
    pub session: Option<String>,
}

/// One unit of the engine thread's work: a chat generation or a whole batch.
/// Every job names the checkpoint it needs; the engine's pickup ensures that
/// checkpoint is the one loaded, swapping the resident one out when it is not.
///
/// Both variants are boxed for the queue's sake: it holds these by value, and
/// a `GenerationJob` alone is over a kilobyte.
pub enum Job {
    Generation(Box<GenerationJob>),
    Batch(Box<BatchJob>),
}

/// Which checkpoint a job needs — and which FILE that means.
///
/// The two are not the same question. On a server started with a GGUF that
/// identifies as none of the official checkpoints, a request for that file's own
/// id runs that file, while a request naming an official checkpoint runs the
/// official hub file, downloading it if need be: an official name must never be
/// answered by weights nobody checked, and a custom file must answer under its
/// own name and no other. Equality here is therefore file identity, which is
/// exactly what the engine's "do I have to swap?" check needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    /// The checkpoint this runs AS: what sizes its caches, which official
    /// sidecar it can draft with, and what the logs call it. For a served file
    /// that identified as nothing, this is its architecture's checkpoint — the
    /// assumption reported at startup.
    pub model: Model,
    /// Whether this means the server's own `--model` file rather than the
    /// official hub file for `model`. Only ever true when the served GGUF
    /// identified as none of the official checkpoints.
    pub served_file: bool,
}

impl Target {
    /// One of the official checkpoints, in its hub file.
    pub fn official(model: Model) -> Self {
        Self {
            model,
            served_file: false,
        }
    }

    /// The file this server was started with, running as `model`.
    pub fn served(model: Model) -> Self {
        Self {
            model,
            served_file: true,
        }
    }
}

impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.model)?;
        if self.served_file {
            f.write_str(" (the served file)")?;
        }
        Ok(())
    }
}

impl Job {
    pub fn origin(&self) -> RequestOrigin {
        match self {
            Job::Generation(job) => job.origin.clone(),
            Job::Batch(job) => job.origin.clone(),
        }
    }

    /// Which request this is, without cloning the identity strings the rest of
    /// the origin carries. The queue's snapshot wants nothing else and takes it
    /// under the lock.
    pub fn origin_id(&self) -> u64 {
        match self {
            Job::Generation(job) => job.origin.id,
            Job::Batch(job) => job.origin.id,
        }
    }

    /// The checkpoint this job runs on, and the file that means.
    pub fn model(&self) -> Target {
        match self {
            Job::Generation(job) => job.model,
            Job::Batch(job) => job.model,
        }
    }

    pub fn cancel(&self) -> &Arc<Cancel> {
        match self {
            Job::Generation(job) => &job.cancel,
            Job::Batch(job) => &job.cancel,
        }
    }

    pub fn events(&self) -> &tokio::sync::mpsc::Sender<EngineEvent> {
        match self {
            Job::Generation(job) => &job.events,
            Job::Batch(job) => &job.events,
        }
    }

    /// The encoded prompt the scheduler scores a cache discount against. A
    /// batch job exposes none — its items are rendered by the runner, so the
    /// warm slots can never discount it and it is scored by its gross size
    /// estimate alone.
    pub fn prompt(&self) -> &[u32] {
        match self {
            Job::Generation(job) => &job.prompt,
            Job::Batch(_) => &[],
        }
    }

    /// The job's output budget in tokens, for the watchdog deadline: a
    /// generation's `max_tokens`, a batch's summed item budgets.
    pub fn max_tokens(&self) -> usize {
        match self {
            Job::Generation(job) => job.max_tokens,
            Job::Batch(job) => job.max_tokens,
        }
    }

    pub fn deadline(&self) -> Option<Instant> {
        match self {
            Job::Generation(job) => job.deadline,
            Job::Batch(job) => job.deadline,
        }
    }

    pub fn set_deadline(&mut self, deadline: Option<Instant>) {
        match self {
            Job::Generation(job) => job.deadline = deadline,
            Job::Batch(job) => job.deadline = deadline,
        }
    }
}

/// A whole `xwen batch` run submitted over HTTP: the request exactly as the CLI
/// reads it on stdin, run on the engine thread, answered with the one document
/// the CLI would print. The engine sends exactly one terminal event for it —
/// [`EngineEvent::BatchDone`] or [`EngineEvent::Error`].
pub struct BatchJob {
    pub origin: RequestOrigin,
    /// The batch to run, exactly as it arrived. The runner renders, encodes
    /// and validates the items itself.
    pub request: BatchRequest,
    /// The checkpoint the request's `model` field resolved to, and the file
    /// that means — resolved by the handler so an unknown name is a 400 before
    /// the queue, never an engine error after it.
    pub model: Target,
    /// Summed item output budgets, for the watchdog deadline.
    pub max_tokens: usize,
    /// The job's cancellation token, exactly as a generation's: client gone,
    /// deadline, shutdown — first reason wins, polled between items and per
    /// decoded token.
    pub cancel: Arc<Cancel>,
    /// Wall-clock ceiling, stamped at pickup like a generation's.
    pub deadline: Option<Instant>,
    pub events: tokio::sync::mpsc::Sender<EngineEvent>,
}

pub struct GenerationJob {
    /// Which request this job is, and where it came from.
    pub origin: RequestOrigin,
    /// The checkpoint this job runs on, and the file that means. The compat
    /// dialects resolve a request `model` that names a checkpoint this server
    /// serves, and refuse anything else.
    pub model: Target,
    /// The rendered prompt, already encoded by the HTTP layer with the same
    /// tokenizer the engine decodes with. The engine prefills exactly these ids.
    pub prompt: Vec<u32>,
    /// Length of the shared-context span of `prompt`: the generation header
    /// starts here. The engine prefills up to here as a span of its own so it can
    /// snapshot the KV cache at the point every turn of a conversation shares.
    pub boundary: usize,
    /// Number of tokens up to and including the leading system block, when the
    /// conversation renders one. Every conversation a client holds shares this
    /// much of its prompt, so the engine takes a second, pinned snapshot here —
    /// the point an unrelated conversation from the same client can resume at
    /// instead of prefilling from zero. `None`, or a position too shallow to be
    /// worth a snapshot, means the prompt is prefilled in the usual two spans.
    pub anchor: Option<usize>,
    /// Whether the generation header ends inside an open thinking span, i.e.
    /// whether the first decoded token is reasoning rather than answer text.
    pub starts_in_thinking: bool,
    /// Thinking budget in tokens (`None` = uncapped).
    pub max_think: Option<usize>,
    /// Hard output cap; thinking tokens count toward it.
    pub max_tokens: usize,
    pub sampling: SamplerOptions,
    pub stop_sequences: Vec<String>,
    /// Tool definitions in the OpenAI object shape
    /// (`{"type":"function","function":{...}}`), already normalized by the HTTP
    /// layer. Empty means the request carries no tools, and the engine treats
    /// `<tool_call>` in the output as ordinary text.
    pub tools: Vec<serde_json::Value>,
    /// Schema constraint for the reply, already compiled and armed by the HTTP
    /// layer. The engine hands it to the generator whole; `None` decodes
    /// unconstrained (and clears any previous request's grammar).
    pub grammar: Option<crate::constrain::GrammarState>,
    /// The job's cancellation token. The HTTP side's drop guard sets `ClientGone`
    /// into it, and the engine stamps `Deadline` and `Shutdown` as those fire, so
    /// every consumer reads one settled reason.
    pub cancel: Arc<Cancel>,
    /// Wall-clock ceiling for the whole job. `None` means the engine derives one
    /// from the job's own size — the prompt length and the reply budget the
    /// context leaves — when it picks the job up; the configured watchdog rates
    /// govern that derivation, and a rate of 0 leaves the job unbounded.
    pub deadline: Option<Instant>,
    pub events: tokio::sync::mpsc::Sender<EngineEvent>,
}

/// Why a generation was abandoned. Set once; the first writer wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelReason {
    /// The response body was dropped: the client is gone and cannot be told
    /// anything, so the generation is abandoned without a terminal event.
    ClientGone,
    /// The job outlived its wall-clock ceiling. The client is still listening
    /// and is owed a complete, truthful terminal event.
    Deadline,
    /// The server is shutting down.
    Shutdown,
}

impl CancelReason {
    /// The label the server's log lines name the reason by.
    pub fn label(self) -> &'static str {
        match self {
            CancelReason::ClientGone => "client gone",
            CancelReason::Deadline => "deadline",
            CancelReason::Shutdown => "shutdown",
        }
    }

    /// The atomic encoding: zero is reserved for "live".
    fn state(self) -> u8 {
        match self {
            CancelReason::ClientGone => 1,
            CancelReason::Deadline => 2,
            CancelReason::Shutdown => 3,
        }
    }

    fn from_state(state: u8) -> Option<Self> {
        match state {
            1 => Some(CancelReason::ClientGone),
            2 => Some(CancelReason::Deadline),
            3 => Some(CancelReason::Shutdown),
            _ => None,
        }
    }
}

/// Shared, lock-free cancellation state for one generation. Cheap enough to poll
/// per decoded token: a relaxed load costs nanoseconds against a ~46 ms token.
#[derive(Debug, Default)]
pub struct Cancel(AtomicU8);

impl Cancel {
    const LIVE: u8 = 0;

    pub fn reason(&self) -> Option<CancelReason> {
        CancelReason::from_state(self.0.load(Ordering::Relaxed))
    }

    pub fn is_cancelled(&self) -> bool {
        self.reason().is_some()
    }

    /// Record why the generation is over. The first writer wins: a later,
    /// different reason never rewrites the one the finalization acts on.
    pub fn cancel(&self, reason: CancelReason) {
        let _ = self.0.compare_exchange(
            Self::LIVE,
            reason.state(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }
}

/// Cancels its job with [`CancelReason::ClientGone`] when dropped. It lives
/// wherever the HTTP layer's response does, so hyper dropping the response body —
/// the only departed-client signal axum exposes — is what fires it.
pub struct CancelGuard(Arc<Cancel>);

impl CancelGuard {
    pub fn new(cancel: Arc<Cancel>) -> Self {
        Self(cancel)
    }
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.0.cancel(CancelReason::ClientGone);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    /// Sent once per job, after prompt rendering + encoding. `cached_tokens` is the KV
    /// prefix reused from the previous request (0 on a cold or divergent prompt).
    Start {
        input_tokens: usize,
        cached_tokens: usize,
    },
    /// UTF-8-safe text delta from inside the thinking span.
    Thinking(String),
    /// UTF-8-safe text delta from the answer span. Stop-sequence holdback is already
    /// applied: matched stop text is never delivered.
    Text(String),
    /// A tool call opened. Followed by zero or more [`EngineEvent::ToolCallDelta`] and
    /// exactly one [`EngineEvent::ToolCallEnd`]. Calls never interleave: a second call
    /// only starts after the first one has ended.
    ToolCallStart {
        name: String,
    },
    /// A fragment of the call's arguments object, as JSON text. Concatenated over one
    /// call, the deltas form exactly one syntactically valid JSON object — including the
    /// `{`/`}` framing, and including the case where generation was cut mid-call, which
    /// the engine heals closed. A call with no arguments still yields `{}`.
    ToolCallDelta(String),
    ToolCallEnd,
    /// Terminal event on success. `output_tokens` includes thinking tokens.
    Done {
        stop: StopKind,
        output_tokens: usize,
        thinking_tokens: usize,
    },
    /// Terminal event of a [`BatchJob`]: the whole response document, exactly
    /// what the CLI would print. Batch jobs produce no other content events.
    BatchDone(Box<BatchResponse>),
    /// Terminal event on failure; handlers map this to an API error response.
    /// `request_fault` is true when the request itself is at fault (a prompt that does
    /// not fit the context, an impossible parameter combination) — handlers turn that
    /// into a 400-class error; everything else is a 500-class server error.
    Error {
        message: String,
        request_fault: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopKind {
    /// The model emitted an end-of-generation token.
    EndTurn,
    /// The `max_tokens` cap was reached.
    MaxTokens,
    /// A requested stop sequence matched (the matched sequence, never emitted).
    StopSequence(String),
    /// The turn produced at least one tool call and is waiting on its results.
    ToolUse,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reason is write-once: whatever lands first is what the finalization
    /// acts on, and everything later — including the drop guard firing on the way
    /// out — leaves it alone.
    #[test]
    fn the_first_cancel_reason_wins() {
        let cancel = Cancel::default();
        assert_eq!(cancel.reason(), None);
        assert!(!cancel.is_cancelled());

        cancel.cancel(CancelReason::Deadline);
        cancel.cancel(CancelReason::ClientGone);
        cancel.cancel(CancelReason::Shutdown);
        assert_eq!(cancel.reason(), Some(CancelReason::Deadline));
        assert!(cancel.is_cancelled());
    }

    /// Racing writers settle on exactly one reason, and it stays settled.
    #[test]
    fn racing_cancels_settle_on_one_reason() {
        let reasons = [
            CancelReason::ClientGone,
            CancelReason::Deadline,
            CancelReason::Shutdown,
        ];
        let cancel = Arc::new(Cancel::default());
        let writers: Vec<_> = reasons
            .into_iter()
            .map(|reason| {
                let cancel = Arc::clone(&cancel);
                std::thread::spawn(move || cancel.cancel(reason))
            })
            .collect();
        for writer in writers {
            writer.join().expect("a cancel call never panics");
        }
        let settled = cancel.reason().expect("some writer won");
        for reason in reasons {
            cancel.cancel(reason);
            assert_eq!(cancel.reason(), Some(settled));
        }
    }

    /// Dropping the guard is the departed-client signal.
    #[test]
    fn dropping_the_guard_reports_the_client_gone() {
        let cancel = Arc::new(Cancel::default());
        let guard = CancelGuard::new(Arc::clone(&cancel));
        assert!(!cancel.is_cancelled());
        drop(guard);
        assert_eq!(cancel.reason(), Some(CancelReason::ClientGone));
    }

    /// A response body dropped after the job was already cancelled for another
    /// reason must not relabel it: the finalization already ran on that reason.
    #[test]
    fn a_guard_dropped_after_a_deadline_does_not_overwrite_it() {
        let cancel = Arc::new(Cancel::default());
        let guard = CancelGuard::new(Arc::clone(&cancel));
        cancel.cancel(CancelReason::Deadline);
        drop(guard);
        assert_eq!(cancel.reason(), Some(CancelReason::Deadline));
    }
}
