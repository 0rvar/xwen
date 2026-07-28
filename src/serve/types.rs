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

/// Who a job belongs to, for correlating the events one request produces.
///
/// The id is assigned once, at submit, and is monotonic for the life of the
/// process; it never reaches the client and never appears in a log line, so
/// nothing outside the server depends on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestOrigin {
    pub id: u64,
    pub dialect: Dialect,
    /// Whether the client asked for its reply as a stream.
    pub streaming: bool,
}

pub struct GenerationJob {
    /// Which request this job is, and where it came from.
    pub origin: RequestOrigin,
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
