//! The server's log as data: every operational line `xwen serve` produces is a
//! [`ServeLog`] value sent over a channel to one sink thread, which owns the
//! wording.
//!
//! Two properties are the point. Sends never block and never fail: a
//! [`ServeLogger`] is a cheap `Clone` around an unbounded `std::sync::mpsc`
//! sender that discards `SendError`, so the sites that log while holding the
//! queue mutex — from the engine thread and from tokio handler threads alike —
//! cannot be parked on a slow or dead consumer. And formatting lives in exactly
//! one place ([`ServeLog::render`]), so a consumer that wants the numbers rather
//! than the sentence reads the variant instead of parsing the line.
//!
//! `render` is an `Option` because an event may be worth carrying without being
//! worth a line of stderr.

use std::fmt;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::mpsc::{Receiver, Sender, SyncSender, channel, sync_channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::config::IdleUnload;
use super::types::{CancelReason, RequestOrigin, StopKind};
use crate::generate::SpecStats;

/// Bytes per MiB: paging is the one place the engine reports a size.
const MIB: f64 = 1024.0 * 1024.0;

/// Why a stored cache segment was deleted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskEvictReason {
    /// A longer segment covers everything this one did: the history it ends at is a
    /// strict prefix of one just written, and nothing chains onto it any more.
    Superseded,
    /// The store outgrew `disk_max_gib`, and this was the least recently used
    /// childless segment in it.
    Budget,
    /// The file did not pass validation, naming the class it failed in — an
    /// intact segment for a checkpoint, tokenizer or container version that is no
    /// longer in use reads differently from bytes that do not hold together.
    Invalid { class: &'static str },
    /// The file's stored history is not the conversation it was matched to. Segment
    /// names are a hash of the token ids, so a file can be found by a conversation
    /// it does not hold — and a segment that does not hold the prompt's own tokens
    /// holds another conversation's keys for them.
    Collided,
    /// The segment names a parent that is not there, or one whose history is not the
    /// one it expects to sit behind. Its own rows are meaningless without the spans
    /// in front of them.
    Orphaned,
}

impl fmt::Display for DiskEvictReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiskEvictReason::Superseded => f.write_str("superseded"),
            DiskEvictReason::Budget => f.write_str("over the size budget"),
            DiskEvictReason::Invalid { class } => write!(f, "invalid: {class}"),
            DiskEvictReason::Collided => f.write_str("a different conversation's history"),
            DiskEvictReason::Orphaned => f.write_str("no chain to a stored root"),
        }
    }
}

/// What a segment the writer just laid down is to the conversation that wrote it.
/// Both are ordinary; the distinction is there so a log line says whether bytes went
/// into a shared prefix or into one conversation's own continuation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskSegmentRole {
    /// A prefix two conversations now share, written by cutting a stored segment at
    /// the position they diverge.
    Base,
    /// One conversation's continuation past everything already stored.
    Tail,
}

impl fmt::Display for DiskSegmentRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiskSegmentRole::Base => f.write_str("base"),
            DiskSegmentRole::Tail => f.write_str("tail"),
        }
    }
}

/// Which part of a job was running when it was abandoned before it decoded
/// anything: the cache dispatch that precedes the first prefill chunk, or the
/// prefill itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobPhase {
    Dispatch,
    Prefill,
}

impl fmt::Display for JobPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JobPhase::Dispatch => f.write_str("dispatch"),
            JobPhase::Prefill => f.write_str("prefill"),
        }
    }
}

/// Everything one finished request did, gathered as it happened and reported
/// once at the end.
///
/// Exactly one of these follows every [`ServeLog::JobPicked`] — a job that
/// failed, panicked, or was abandoned by a client that stopped listening still
/// produces one, so a consumer counting jobs in flight never leaks an entry.
#[derive(Debug, Clone)]
pub struct JobRecord {
    pub origin: RequestOrigin,
    /// What the client was told ended the reply, or `None` when nothing was:
    /// a job abandoned by a departed client is owed no terminal event, and a
    /// failed one gets an error instead.
    pub stop: Option<StopKind>,
    /// Why the job was cut short, for a job that was.
    pub abandoned: Option<CancelReason>,
    /// The failure the client was sent, for a job that failed or panicked.
    pub error: Option<String>,
    /// Prompt tokens, of which `cache_read` came out of the KV cache and
    /// `prefill_tokens` were run through the model.
    pub prompt_tokens: usize,
    pub cache_read: usize,
    pub prefill_tokens: usize,
    pub prefill_secs: f64,
    /// Emitted tokens, thinking included, as [`crate::generate::DecodeOutcome`]
    /// counts them.
    pub output_tokens: usize,
    pub thinking_tokens: usize,
    pub decode_secs: f64,
    /// Pickup to the first content event handed to the client — the wait a
    /// client actually experiences, so it covers the queue-free part of the job:
    /// the lazy model load, paging, and the whole prefill. `None` for a job that
    /// never sent anything.
    pub ttft_secs: Option<f64>,
    /// What speculation did, for a reply that was speculated at all.
    pub spec: Option<SpecStats>,
}

/// One cache slot as the engine holds it, for a consumer that wants to show
/// where the host RAM went. Derived on demand, never stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotSummary {
    /// Whether this slot's conversation is the one in the model's KV cache.
    pub live: bool,
    /// Tokens in the slot's history.
    pub tokens: usize,
    /// Positions it can be resumed from: its turn boundaries, plus the pinned
    /// anchor at the end of its system block when it has one.
    pub snapshots: usize,
    /// Host bytes it holds: snapshots plus the full-attention image plus the
    /// drafter's planes.
    pub image_bytes: usize,
    pub has_drafter: bool,
    /// The manager's clock when this slot last served a job; zero means never.
    pub last_used: u64,
    /// How far the retained image is known to agree with the history.
    pub agrees_to: usize,
}

/// One waiting request, as the queue sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueEntry {
    pub id: u64,
    pub prompt_tokens: usize,
    /// When it was submitted, rather than how long it had waited when the
    /// snapshot was taken: snapshots are only produced when the queue changes,
    /// and an entry sitting behind a five-minute prefill would otherwise be
    /// shown at the age it had on arrival for the whole of that wait.
    pub queued_at: Instant,
}

/// One thing worth saying about the running server.
#[derive(Debug, Clone)]
pub enum ServeLog {
    /// The configured context length is larger than the checkpoint was converted
    /// with, so the smaller one is served.
    ContextClamped { requested: usize, trained: usize },
    /// The listener is up.
    Listening {
        address: String,
        anthropic: bool,
        openai: bool,
    },
    /// The checkpoint the first request will load.
    ServingModel { path: PathBuf },
    /// The model finished its lazy load.
    ModelLoaded { elapsed: Duration },
    /// The next job needs the other checkpoint, so the resident one is being
    /// dropped (imaged out first when the disk tier serves it); the lazy load
    /// brings the named one in.
    CheckpointSwappingOut {
        from: crate::hub::Model,
        to: crate::hub::Model,
    },
    /// A requested checkpoint (or its drafter sidecar) is not in the Hugging
    /// Face cache and is being downloaded before the job can run.
    CheckpointDownloading {
        repo: &'static str,
        file: &'static str,
        size: &'static str,
    },
    /// The batch runner reported progress: a shared prefill, or one item done.
    BatchProgress(crate::batch::BatchProgress),
    /// The speculative drafter finished loading, alongside the model.
    DrafterLoaded { elapsed: Duration, draft_ctx: usize },
    /// The idle timer elapsed and the model was dropped. `configured` is the
    /// window that was set, printed next to the span actually measured so a
    /// report of "it unloaded early" can be settled from the log.
    IdleUnloaded {
        elapsed: Duration,
        configured: Option<Duration>,
    },
    /// The post-job cache reset failed, which costs the model.
    CacheClearFailed { error: String },
    /// A client stopped draining its event channel for the whole send deadline.
    ClientStalled { after: Duration },
    /// A requested thinking budget does not fit the reply the context leaves.
    /// `using` is `None` for a reply that will reason without a ceiling.
    ThinkBudgetClamped {
        requested: usize,
        max_new: usize,
        using: Option<usize>,
    },
    /// The KV cache and the prefix cache's token history disagree about how much
    /// is cached, so the prompt is replayed from zero.
    CacheLengthMismatch { cached: usize, expected: usize },
    /// What speculation did for one reply.
    SpecReport { stats: SpecStats },
    /// Tool spans this request could not take at face value: calls the engine had
    /// to close, finished values that were not the JSON their schema asked for,
    /// and spans that named no callable tool and went back to the client as text.
    ToolSpanReport {
        healed: usize,
        quoted: usize,
        degraded: usize,
    },
    /// One `<tool_call>` span whose contents were not a call this parser could
    /// read. The text went to the client as answer text; it is logged because a
    /// model that frames calls it cannot fill in is a prompt or template
    /// problem, and the text is the only evidence of which.
    ToolSpanDegraded { text: String },
    /// A job was abandoned mid-decode by a gone client or a shutdown.
    AbandonedDuringDecode {
        reason: CancelReason,
        tokens_out: usize,
        cache_kept: usize,
    },
    /// A job outlived its wall-clock ceiling mid-decode and was truncated.
    DeadlineDuringDecode {
        ceiling: Duration,
        tokens_out: usize,
    },
    /// A job was abandoned before it decoded anything. `done`/`total` are prompt
    /// tokens; `kept` is what the cache holds toward a retry of the same prompt.
    AbandonedBeforeDecode {
        reason: CancelReason,
        phase: JobPhase,
        done: usize,
        total: usize,
        kept: usize,
    },
    /// A job outlived its ceiling before it decoded anything.
    DeadlineBeforeDecode {
        ceiling: Duration,
        done: usize,
        total: usize,
    },
    /// A conversation was imaged out of the KV cache into host RAM.
    SlotPagedOut {
        slot: usize,
        pos: usize,
        bytes: usize,
        elapsed: Duration,
    },
    /// A conversation was uploaded back into the KV cache.
    SlotPagedIn {
        slot: usize,
        restore: usize,
        bytes: usize,
        elapsed: Duration,
    },
    /// The on-disk segment store as the startup scan found it: `segments` files
    /// totalling `bytes` are tracked under `kv/` (this checkpoint's and any other
    /// checkpoint's, since the budget spans the whole store), and `rejected` files
    /// were deleted for failing validation or for chaining onto nothing.
    DiskCacheScanned {
        segments: usize,
        bytes: u64,
        rejected: usize,
    },
    /// One segment was written to the disk tier, covering `tokens` positions of the
    /// conversation that wrote it.
    DiskSegmentWritten {
        role: DiskSegmentRole,
        tokens: usize,
        bytes: u64,
        ms: u64,
    },
    /// A stored segment was cut in two at `at` so that a conversation arriving with a
    /// different continuation shares everything before it.
    DiskSegmentSplit { at: usize },
    /// A conversation's resume point at `at` was stored into the segment ending there,
    /// which had none — so everything that forks at that boundary from now on can
    /// resume from it instead of prefilling to it.
    DiskBoundarySnapshotStored { at: usize },
    /// A chain of stored segments was read back into a cache slot, which resumes at
    /// `resume` of the `tokens` positions the chain covers.
    DiskChainHydrated {
        segments: usize,
        tokens: usize,
        resume: usize,
        bytes: u64,
        ms: u64,
    },
    /// A stored segment was deleted.
    DiskSegmentEvicted { reason: DiskEvictReason, bytes: u64 },
    /// The disk tier could not do something it was asked to. Perf-only: the
    /// engine never learns, and the worst outcome is a cache miss.
    DiskCacheFailed { action: &'static str, error: String },
    /// A queued request outlived the queue timeout without reaching the model.
    QueuedRequestDropped {
        waited: Duration,
        queue_timeout: Duration,
    },
    /// The scheduler ran a request ahead of older ones — the one observable act
    /// of scheduling, said out loud so the policy is falsifiable from the log.
    OutOfOrderPick {
        prompt_tokens: usize,
        cost: usize,
        older: usize,
        oldest_waited: Duration,
    },
    /// A response header the builders wrote could not be encoded.
    InvalidHeaderDropped { name: &'static str, value: String },
    /// A shutdown signal could not be registered, so that one way of asking the
    /// server to stop is not available. The others still are.
    SignalUnavailable { signal: &'static str, error: String },
    /// A shutdown signal arrived; in-flight requests are being waited on.
    ShutdownRequested,
    /// The serve loop returned and the engine is coming down.
    ShuttingDown,
    /// The shutdown grace period elapsed with connections still open.
    ShutdownGraceExpired { grace: Duration },
    /// A line from code that sits below the server — the model's resident-memory
    /// report at load — carried here rather than printed where it was produced,
    /// so that every line the process writes goes through one consumer. Already
    /// worded by its author: [`ServeLog::render`] passes it through untouched.
    HostLine(String),

    // The events below carry data rather than words: every one of them renders
    // `None`, so what an operator reads on stderr is exactly what it was before
    // they existed. They are how a consumer that wants to draw the server —
    // progress, queue depth, where the cache went, what each request cost —
    // learns any of it without polling engine state it must never touch.
    /// The engine took a job off the queue and is about to run it. A job dropped
    /// at the dequeue — its client gone, or the server shutting down — is never
    /// picked, and so is never reported here.
    JobPicked {
        origin: RequestOrigin,
        prompt_tokens: usize,
        /// How long it waited between submit and pickup.
        queue_wait: Duration,
        /// The wall-clock ceiling the job was given at this pickup, as the
        /// instant it expires — a consumer counting down to it wants the
        /// engine's own deadline rather than one re-derived from a duration and
        /// whenever this event was received. `None` for an unbounded job.
        deadline: Option<Instant>,
    },
    /// What the prefix cache could do for the job: `cached` tokens matched, of
    /// which `resume` are actually being reused (they differ when the KV cache
    /// turned out to disagree and the prompt is replayed), in slot `slot`.
    JobCacheResolved {
        id: u64,
        cached: usize,
        resume: usize,
        slot: usize,
    },
    /// A job is over, whatever ended it.
    JobDone(Box<JobRecord>),
    /// Prefill progress, once per chunk. `done` counts the whole prompt,
    /// cache-read tokens included, so `done`/`total` is the bar a client's wait
    /// is measured against rather than the current span's share of it.
    PrefillTick { done: usize, total: usize },
    /// One decoded token. `tokens_out` counts every token this reply has
    /// produced, thinking included, and `thinking` says which side of the
    /// reasoning boundary this one fell.
    DecodeTick { tokens_out: usize, thinking: bool },
    /// The cache slots, as they stand after something changed them.
    SlotsSnapshot(Vec<SlotSummary>),
    /// The waiting requests, as they stand after a push or a dequeue.
    QueueSnapshot(Vec<QueueEntry>),
}

impl ServeLog {
    /// The line an operator reads, or `None` for an event that carries data
    /// without being worth stderr.
    pub fn render(&self) -> Option<String> {
        Some(match self {
            ServeLog::ContextClamped { requested, trained } => format!(
                "warning: context_length {requested} exceeds the checkpoint's {trained}-token \
                 context; serving with {trained}"
            ),
            ServeLog::Listening {
                address,
                anthropic,
                openai,
            } => format!(
                "xwen serve listening on http://{address} ({})",
                enabled_apis(*anthropic, *openai)
            ),
            ServeLog::ServingModel { path } => {
                format!("serving {} (loaded on the first request)", path.display())
            }
            // Imaging is not claimed here: it happens only when the disk tier
            // serves the outgoing checkpoint, and when it does the page-out
            // reports itself (`SlotPagedOut`).
            ServeLog::CheckpointSwappingOut { from, to } => {
                format!("xwen: swapping the {from} checkpoint out for {to}")
            }
            ServeLog::CheckpointDownloading { repo, file, size } => format!(
                "xwen: {repo}/{file} is not in the Hugging Face cache; downloading ({size}, \
                 resumes in place)"
            ),
            ServeLog::BatchProgress(progress) => match progress {
                crate::batch::BatchProgress::SharedPrefix { tokens, ms } => {
                    format!("xwen: batch shared prefix {tokens} tokens prefilled in {ms:.0}ms")
                }
                crate::batch::BatchProgress::Item {
                    id,
                    completion_tokens,
                    ms,
                } => format!("xwen: batch item {id:?} {completion_tokens} tokens in {ms:.0}ms"),
            },
            ServeLog::ModelLoaded { elapsed } => {
                format!("xwen serve: model loaded in {:.1}s", elapsed.as_secs_f64())
            }
            ServeLog::DrafterLoaded { elapsed, draft_ctx } => format!(
                "xwen serve: drafter loaded in {:.1}s, speculating over the first {draft_ctx} positions",
                elapsed.as_secs_f64()
            ),
            ServeLog::IdleUnloaded {
                elapsed,
                configured,
            } => format!(
                "xwen serve: idle for {:.0}s of a configured {}, unloaded the model \
                 (the next request reloads it)",
                elapsed.as_secs_f64(),
                IdleUnload(*configured)
            ),
            ServeLog::CacheClearFailed { error } => {
                format!("xwen serve: clearing the KV cache failed ({error}); dropping the model")
            }
            ServeLog::ClientStalled { after } => format!(
                "xwen serve: a client stopped reading for {}s; abandoning its request",
                after.as_secs()
            ),
            ServeLog::ThinkBudgetClamped {
                requested,
                max_new,
                using,
            } => format!(
                "xwen serve: a thinking budget of {requested} does not fit a {max_new}-token \
                 reply; using {}",
                using.map_or("no ceiling".to_string(), |n| n.to_string())
            ),
            ServeLog::CacheLengthMismatch { cached, expected } => format!(
                "xwen serve: the KV cache holds {cached} tokens where the prefix cache expected \
                 {expected}; replaying the prompt"
            ),
            ServeLog::SpecReport { stats } => {
                let paused = if stats.paused_rounds > 0 {
                    format!(", {} paused", stats.paused_rounds)
                } else {
                    String::new()
                };
                format!(
                    "xwen serve: spec: {} rounds, drafted {}, accepted {} ({:.0}%){paused}",
                    stats.rounds,
                    stats.drafted,
                    stats.accepted,
                    stats.acceptance_rate() * 100.0,
                )
            }
            ServeLog::ToolSpanReport {
                healed,
                quoted,
                degraded,
            } => {
                let mut parts = Vec::new();
                if *healed > 0 {
                    parts.push(format!("{healed} healed"));
                }
                if *quoted > 0 {
                    parts.push(format!("{quoted} parse-or-quote"));
                }
                if *degraded > 0 {
                    parts.push(format!("{degraded} degraded to text"));
                }
                format!("xwen serve: tool spans: {}", parts.join(", "))
            }
            ServeLog::ToolSpanDegraded { text } => format!(
                "xwen serve: a <tool_call> span named no callable tool and was delivered as \
                 text: {text:?}"
            ),
            ServeLog::AbandonedDuringDecode {
                reason,
                tokens_out,
                cache_kept,
            } => format!(
                "xwen serve: abandoned a request ({}) during decode after {tokens_out} output \
                 tokens; the cache kept {cache_kept} tokens for the next request",
                reason.label(),
            ),
            ServeLog::DeadlineDuringDecode {
                ceiling,
                tokens_out,
            } => format!(
                "xwen serve: a request exceeded its {:.1}s ceiling after {tokens_out} tokens; \
                 truncated",
                ceiling.as_secs_f64(),
            ),
            ServeLog::AbandonedBeforeDecode {
                reason,
                phase,
                done,
                total,
                kept,
            } => format!(
                "xwen serve: abandoned a request ({}) during {phase} after {done} of {total} \
                 prompt tokens; the cache kept {kept} tokens for the next request",
                reason.label(),
            ),
            ServeLog::DeadlineBeforeDecode {
                ceiling,
                done,
                total,
            } => format!(
                "xwen serve: a request exceeded its {:.1}s ceiling after {done} of {total} \
                 prompt tokens; truncated",
                ceiling.as_secs_f64(),
            ),
            ServeLog::SlotPagedOut {
                slot,
                pos,
                bytes,
                elapsed,
            } => format!(
                "xwen serve: cache slot {slot} paged out at {pos} tokens: {:.0} MiB in {:.0} ms",
                *bytes as f64 / MIB,
                elapsed.as_secs_f64() * 1e3,
            ),
            ServeLog::SlotPagedIn {
                slot,
                restore,
                bytes,
                elapsed,
            } => format!(
                "xwen serve: cache slot {slot} paged in at {restore} tokens: {:.0} MiB in {:.0} ms",
                *bytes as f64 / MIB,
                elapsed.as_secs_f64() * 1e3,
            ),
            ServeLog::DiskCacheScanned {
                segments,
                bytes,
                rejected,
            } => {
                let rejected = if *rejected > 0 {
                    format!(", {rejected} rejected")
                } else {
                    String::new()
                };
                format!(
                    "xwen serve: disk cache: {segments} segment{} ({}) on disk{rejected}",
                    plural(*segments),
                    size(*bytes),
                )
            }
            ServeLog::DiskSegmentWritten {
                role,
                tokens,
                bytes,
                ms,
            } => format!(
                "xwen serve: disk cache: wrote a {tokens}-token {role} segment, {} in {ms} ms",
                size(*bytes),
            ),
            ServeLog::DiskSegmentSplit { at } => {
                format!("xwen serve: disk cache: split a stored segment at {at} tokens")
            }
            ServeLog::DiskBoundarySnapshotStored { at } => format!(
                "xwen serve: disk cache: stored a resume point at {at} tokens in the segment \
                 ending there"
            ),
            ServeLog::DiskChainHydrated {
                segments,
                tokens,
                resume,
                bytes,
                ms,
            } => format!(
                "xwen serve: disk cache: hydrated a {segments}-segment chain of {tokens} tokens, \
                 resuming at {resume}: {} in {ms} ms",
                size(*bytes),
            ),
            ServeLog::DiskSegmentEvicted { reason, bytes } => format!(
                "xwen serve: disk cache: deleted a {} segment ({reason})",
                size(*bytes),
            ),
            ServeLog::DiskCacheFailed { action, error } => {
                format!("xwen serve: disk cache: {action} failed ({error})")
            }
            ServeLog::QueuedRequestDropped {
                waited,
                queue_timeout,
            } => format!(
                "xwen serve: dropped a request that waited {:.0}s in the queue \
                 (queue_timeout {}s)",
                waited.as_secs_f64(),
                queue_timeout.as_secs()
            ),
            ServeLog::OutOfOrderPick {
                prompt_tokens,
                cost,
                older,
                oldest_waited,
            } => format!(
                "xwen serve: running a {prompt_tokens}-token request ({cost} tokens of prefill \
                 after the cache) ahead of {older} older ones (the oldest has waited {:.1}s)",
                oldest_waited.as_secs_f64()
            ),
            ServeLog::InvalidHeaderDropped { name, value } => {
                format!("xwen serve: dropped invalid header {name}: {value:?}")
            }
            ServeLog::SignalUnavailable { signal, error } => format!(
                "xwen serve: cannot listen for {signal} ({error}); \
                 Ctrl-C and the dashboard's q still shut the server down"
            ),
            ServeLog::ShutdownRequested => {
                "xwen serve: shutting down, waiting for in-flight requests".to_string()
            }
            ServeLog::ShuttingDown => "xwen serve shutting down".to_string(),
            ServeLog::ShutdownGraceExpired { grace } => format!(
                "xwen serve: still shutting down after {}s; exiting",
                grace.as_secs()
            ),
            ServeLog::HostLine(line) => line.clone(),
            // Ticks, snapshots and per-request records are data for a consumer
            // that draws the server. Saying any of them on stderr would bury the
            // lines above under thousands of their own.
            ServeLog::JobPicked { .. }
            | ServeLog::JobCacheResolved { .. }
            | ServeLog::JobDone(_)
            | ServeLog::PrefillTick { .. }
            | ServeLog::DecodeTick { .. }
            | ServeLog::SlotsSnapshot(_)
            | ServeLog::QueueSnapshot(_) => return None,
        })
    }
}

/// A byte count as an operator reads it: whole MiB up to a gibibyte, GiB with one
/// decimal above. The disk tier's sizes span three orders of magnitude — a 72 MiB
/// snapshot to a 60 GiB store — and one unit for all of them prints either 0.1 or
/// 61440.
fn size(bytes: u64) -> String {
    let mib = bytes as f64 / MIB;
    if mib < 1024.0 {
        format!("{mib:.0} MiB")
    } else {
        format!("{:.1} GiB", mib / 1024.0)
    }
}

fn plural(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}

/// The routes an operator gets, for the startup line.
fn enabled_apis(anthropic: bool, openai: bool) -> &'static str {
    match (anthropic, openai) {
        (true, true) => "Anthropic + OpenAI APIs",
        (true, false) => "Anthropic API only",
        (false, true) => "OpenAI API only",
        (false, false) => "no chat APIs enabled",
    }
}

/// What travels down the channel: the events, plus the one control message that
/// ends the sink. The stop is explicit rather than relying on the last sender
/// hanging up, because [`set_global`] parks a sender in a `static` that outlives
/// everything — with hangup as the only exit, the join at the end of `run` would
/// wait forever.
pub(super) enum SinkMessage {
    Event(ServeLog),
    /// A keypress or a resize, forwarded by the dashboard's input thread so the
    /// TUI sink has exactly one place to wait. The stderr sink never sees one.
    Term(crossterm::event::Event),
    /// A rendezvous: acknowledged once everything queued ahead of it has been
    /// written.
    Flush(SyncSender<()>),
    Stop,
}

/// How long [`ServeLogger::flush`] waits. Long enough to outlast any line the
/// sink could be mid-write on, short enough that a wedged sink cannot delay the
/// exit it is called on the way to.
const FLUSH_TIMEOUT: Duration = Duration::from_secs(1);

/// The handle every logging site holds. Cloning is cheap, and `log` neither
/// blocks nor fails: a sink that is gone or behind must never change what the
/// engine or a handler does next.
#[derive(Clone)]
pub struct ServeLogger {
    tx: Sender<SinkMessage>,
}

impl ServeLogger {
    pub fn log(&self, event: ServeLog) {
        let _ = self.tx.send(SinkMessage::Event(event));
    }

    /// Wait — for at most [`FLUSH_TIMEOUT`] — until everything logged so far has
    /// been written. The one caller is the shutdown watchdog, which exits the
    /// process outright and would otherwise race its own last line; every other
    /// site logs and moves on.
    pub fn flush(&self) {
        let (ack, written) = sync_channel(0);
        if self.tx.send(SinkMessage::Flush(ack)).is_err() {
            return;
        }
        let _ = written.recv_timeout(FLUSH_TIMEOUT);
    }

    /// A logger with nothing on the other end, for the paths that need a handle
    /// but no output.
    pub fn discarding() -> Self {
        let (tx, _rx) = channel();
        Self { tx }
    }
}

/// A logger that keeps what it is told, for tests that assert what the server
/// reported rather than how it worded it. Nothing drains it, so the events pile
/// up in the channel until [`EventLog::drain`] takes them.
#[cfg(test)]
pub fn collecting() -> (ServeLogger, EventLog) {
    let (tx, rx) = channel();
    (ServeLogger { tx }, EventLog(rx))
}

#[cfg(test)]
pub struct EventLog(Receiver<SinkMessage>);

#[cfg(test)]
impl EventLog {
    /// Everything logged since the last call, in order.
    pub fn drain(&self) -> Vec<ServeLog> {
        self.0
            .try_iter()
            .filter_map(|message| match message {
                SinkMessage::Event(event) => Some(event),
                _ => None,
            })
            .collect()
    }
}

/// Keeps the sink thread alive for as long as the server runs, and ends it on
/// the way out — including the error paths, since the stop is sent from `Drop`
/// rather than an explicit call somebody could `?` past.
pub struct SinkHandle {
    tx: Sender<SinkMessage>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for SinkHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(SinkMessage::Stop);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Start the stderr sink. Every line the server prints comes out of this thread,
/// in the order the events were sent.
pub fn spawn_stderr_sink() -> (ServeLogger, SinkHandle) {
    let (tx, rx) = channel();
    let thread = std::thread::spawn(move || stderr_sink(rx));
    (
        ServeLogger { tx: tx.clone() },
        SinkHandle {
            tx,
            thread: Some(thread),
        },
    )
}

/// Start the dashboard sink: the same events, drawn instead of printed. Nothing
/// else about the server changes — `--tui` swaps the consumer, not the stream.
pub fn spawn_tui_sink(
    vitals: super::tui::Vitals,
    quit: super::QuitSignal,
) -> (ServeLogger, SinkHandle) {
    let (tx, rx) = channel();
    let input = tx.clone();
    let thread = std::thread::spawn(move || super::tui::run_sink(rx, input, vitals, quit));
    (
        ServeLogger { tx: tx.clone() },
        SinkHandle {
            tx,
            thread: Some(thread),
        },
    )
}

/// Drain events to stderr until the stop message arrives or every sender is
/// gone.
fn stderr_sink(rx: Receiver<SinkMessage>) {
    loop {
        match rx.recv() {
            Ok(SinkMessage::Event(event)) => {
                if let Some(line) = event.render() {
                    eprintln!("{line}");
                }
            }
            // Nothing forwards terminal events to this sink; a stray one is not
            // worth ending the log over.
            Ok(SinkMessage::Term(_)) => {}
            // Everything queued ahead of this has been written by now.
            Ok(SinkMessage::Flush(ack)) => {
                let _ = ack.send(());
            }
            Ok(SinkMessage::Stop) | Err(_) => return,
        }
    }
}

/// The process-wide logger, for the one site that cannot be handed a handle:
/// `IntoResponse::into_response` is a trait method with no access to the
/// server's state. Everything else threads a [`ServeLogger`] explicitly.
static GLOBAL: OnceLock<ServeLogger> = OnceLock::new();

/// Publish the logger the running server's stateless sites report through. The
/// first caller wins; a second server in one process keeps the first's sink.
///
/// This is also where the crate's own [`crate::host_log::host_line`] output is
/// claimed, so that lines produced below `serve` — which cannot name a
/// [`ServeLog`] — join the same stream as everything else rather than reaching
/// stderr on their own. One publish point, so a sink is never half-installed.
pub fn set_global(logger: ServeLogger) {
    let winner = GLOBAL.get_or_init(|| logger).clone();
    crate::host_log::set_host_line_sink(move |line| winner.log(ServeLog::HostLine(line)));
}

/// Report an event from a site with no handle. Silent when no server has
/// published one.
pub fn log_global(event: ServeLog) {
    if let Some(logger) = GLOBAL.get() {
        logger.log(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every line the server can print, pinned to the byte. Headless workflows
    /// read this output with `2>` and grep, so the wording is a contract rather
    /// than a detail of whichever site happens to emit it.
    fn line(event: ServeLog) -> String {
        event.render().expect("the event renders a line")
    }

    #[test]
    fn the_startup_lines_name_the_address_the_apis_and_the_checkpoint() {
        assert_eq!(
            line(ServeLog::Listening {
                address: "127.0.0.1:5241".to_string(),
                anthropic: true,
                openai: true,
            }),
            "xwen serve listening on http://127.0.0.1:5241 (Anthropic + OpenAI APIs)"
        );
        assert_eq!(
            line(ServeLog::Listening {
                address: "0.0.0.0:80".to_string(),
                anthropic: true,
                openai: false,
            }),
            "xwen serve listening on http://0.0.0.0:80 (Anthropic API only)"
        );
        assert_eq!(
            line(ServeLog::Listening {
                address: "0.0.0.0:80".to_string(),
                anthropic: false,
                openai: true,
            }),
            "xwen serve listening on http://0.0.0.0:80 (OpenAI API only)"
        );
        assert_eq!(
            line(ServeLog::Listening {
                address: "0.0.0.0:80".to_string(),
                anthropic: false,
                openai: false,
            }),
            "xwen serve listening on http://0.0.0.0:80 (no chat APIs enabled)"
        );
        assert_eq!(
            line(ServeLog::ServingModel {
                path: PathBuf::from("/models/laguna-s-2.1-Q4_K_M.gguf"),
            }),
            "serving /models/laguna-s-2.1-Q4_K_M.gguf (loaded on the first request)"
        );
    }

    #[test]
    fn the_context_clamp_warning_names_both_lengths() {
        assert_eq!(
            line(ServeLog::ContextClamped {
                requested: 1_000_000,
                trained: 262_144,
            }),
            "warning: context_length 1000000 exceeds the checkpoint's 262144-token context; \
             serving with 262144"
        );
    }

    /// Load times are reported to a tenth of a second.
    #[test]
    fn the_load_lines_report_tenths_of_a_second() {
        assert_eq!(
            line(ServeLog::ModelLoaded {
                elapsed: Duration::from_millis(4249),
            }),
            "xwen serve: model loaded in 4.2s"
        );
        assert_eq!(
            line(ServeLog::DrafterLoaded {
                elapsed: Duration::from_millis(1560),
                draft_ctx: 8192,
            }),
            "xwen serve: drafter loaded in 1.6s, speculating over the first 8192 positions"
        );
    }

    /// The idle line quotes the configured window in the config file's own
    /// syntax, next to the span actually measured.
    #[test]
    fn the_idle_unload_line_quotes_the_configured_window() {
        assert_eq!(
            line(ServeLog::IdleUnloaded {
                elapsed: Duration::from_secs(600),
                configured: Some(Duration::from_secs(600)),
            }),
            "xwen serve: idle for 600s of a configured 10m, unloaded the model \
             (the next request reloads it)"
        );
        assert_eq!(
            line(ServeLog::IdleUnloaded {
                elapsed: Duration::from_millis(3599_600),
                configured: Some(Duration::from_secs(3600)),
            }),
            "xwen serve: idle for 3600s of a configured 1h, unloaded the model \
             (the next request reloads it)"
        );
    }

    #[test]
    fn the_model_failure_lines_carry_the_cause() {
        assert_eq!(
            line(ServeLog::CacheClearFailed {
                error: "the device is gone".to_string(),
            }),
            "xwen serve: clearing the KV cache failed (the device is gone); dropping the model"
        );
        assert_eq!(
            line(ServeLog::ClientStalled {
                after: Duration::from_secs(120),
            }),
            "xwen serve: a client stopped reading for 120s; abandoning its request"
        );
    }

    /// A budget that does not fit is reported with what was used instead —
    /// including the case where nothing fits and reasoning runs uncapped.
    #[test]
    fn the_think_budget_line_names_what_was_used_instead() {
        assert_eq!(
            line(ServeLog::ThinkBudgetClamped {
                requested: 4096,
                max_new: 512,
                using: Some(384),
            }),
            "xwen serve: a thinking budget of 4096 does not fit a 512-token reply; using 384"
        );
        assert_eq!(
            line(ServeLog::ThinkBudgetClamped {
                requested: 4096,
                max_new: 16,
                using: None,
            }),
            "xwen serve: a thinking budget of 4096 does not fit a 16-token reply; \
             using no ceiling"
        );
    }

    #[test]
    fn the_cache_mismatch_line_names_both_lengths() {
        assert_eq!(
            line(ServeLog::CacheLengthMismatch {
                cached: 12,
                expected: 380,
            }),
            "xwen serve: the KV cache holds 12 tokens where the prefix cache expected 380; \
             replaying the prompt"
        );
    }

    /// The speculation line rounds acceptance to whole percent and stays silent
    /// about a paused count of zero.
    #[test]
    fn the_spec_line_reports_acceptance_and_only_names_pauses_when_there_were_any() {
        let stats = SpecStats {
            rounds: 10,
            drafted: 40,
            accepted: 30,
            ..SpecStats::default()
        };
        assert_eq!(
            line(ServeLog::SpecReport { stats }),
            "xwen serve: spec: 10 rounds, drafted 40, accepted 30 (75%)"
        );
        assert_eq!(
            line(ServeLog::SpecReport {
                stats: SpecStats {
                    paused_rounds: 3,
                    ..stats
                },
            }),
            "xwen serve: spec: 10 rounds, drafted 40, accepted 30 (75%), 3 paused"
        );
    }

    /// The tool-span line names only the kinds that occurred.
    #[test]
    fn the_tool_span_line_names_each_kind_that_occurred() {
        assert_eq!(
            line(ServeLog::ToolSpanReport {
                healed: 3,
                quoted: 0,
                degraded: 0,
            }),
            "xwen serve: tool spans: 3 healed"
        );
        assert_eq!(
            line(ServeLog::ToolSpanReport {
                healed: 0,
                quoted: 1,
                degraded: 0,
            }),
            "xwen serve: tool spans: 1 parse-or-quote"
        );
        assert_eq!(
            line(ServeLog::ToolSpanReport {
                healed: 0,
                quoted: 0,
                degraded: 2,
            }),
            "xwen serve: tool spans: 2 degraded to text"
        );
        assert_eq!(
            line(ServeLog::ToolSpanReport {
                healed: 2,
                quoted: 1,
                degraded: 1,
            }),
            "xwen serve: tool spans: 2 healed, 1 parse-or-quote, 1 degraded to text"
        );
    }

    /// A degraded span names itself with the text it delivered, which is the
    /// only record of what the model tried to write.
    #[test]
    fn a_degraded_tool_span_logs_the_text_it_delivered() {
        assert_eq!(
            line(ServeLog::ToolSpanDegraded {
                text: "<tool_call>\nnot a call\n</tool_call>".into(),
            }),
            "xwen serve: a <tool_call> span named no callable tool and was delivered as text: \
             \"<tool_call>\\nnot a call\\n</tool_call>\""
        );
    }

    #[test]
    fn the_decode_abandonment_lines_name_the_reason_and_what_the_cache_kept() {
        assert_eq!(
            line(ServeLog::AbandonedDuringDecode {
                reason: CancelReason::ClientGone,
                tokens_out: 142,
                cache_kept: 2048,
            }),
            "xwen serve: abandoned a request (client gone) during decode after 142 output \
             tokens; the cache kept 2048 tokens for the next request"
        );
        assert_eq!(
            line(ServeLog::AbandonedDuringDecode {
                reason: CancelReason::Shutdown,
                tokens_out: 0,
                cache_kept: 0,
            }),
            "xwen serve: abandoned a request (shutdown) during decode after 0 output tokens; \
             the cache kept 0 tokens for the next request"
        );
        assert_eq!(
            line(ServeLog::DeadlineDuringDecode {
                ceiling: Duration::from_millis(214_240),
                tokens_out: 512,
            }),
            "xwen serve: a request exceeded its 214.2s ceiling after 512 tokens; truncated"
        );
    }

    #[test]
    fn the_pre_decode_abandonment_lines_name_the_phase_and_the_prompt_progress() {
        assert_eq!(
            line(ServeLog::AbandonedBeforeDecode {
                reason: CancelReason::ClientGone,
                phase: JobPhase::Dispatch,
                done: 0,
                total: 34_112,
                kept: 0,
            }),
            "xwen serve: abandoned a request (client gone) during dispatch after 0 of 34112 \
             prompt tokens; the cache kept 0 tokens for the next request"
        );
        assert_eq!(
            line(ServeLog::AbandonedBeforeDecode {
                reason: CancelReason::Shutdown,
                phase: JobPhase::Prefill,
                done: 2048,
                total: 3896,
                kept: 2048,
            }),
            "xwen serve: abandoned a request (shutdown) during prefill after 2048 of 3896 \
             prompt tokens; the cache kept 2048 tokens for the next request"
        );
        assert_eq!(
            line(ServeLog::DeadlineBeforeDecode {
                ceiling: Duration::from_secs(60),
                done: 512,
                total: 4096,
            }),
            "xwen serve: a request exceeded its 60.0s ceiling after 512 of 4096 prompt tokens; \
             truncated"
        );
    }

    /// Paging reports whole MiB and whole milliseconds.
    #[test]
    fn the_paging_lines_report_whole_mib_and_milliseconds() {
        assert_eq!(
            line(ServeLog::SlotPagedOut {
                slot: 0,
                pos: 380,
                bytes: 113_246_208,
                elapsed: Duration::from_micros(41_400),
            }),
            "xwen serve: cache slot 0 paged out at 380 tokens: 108 MiB in 41 ms"
        );
        assert_eq!(
            line(ServeLog::SlotPagedIn {
                slot: 2,
                restore: 380,
                bytes: 113_246_208,
                elapsed: Duration::from_micros(41_400),
            }),
            "xwen serve: cache slot 2 paged in at 380 tokens: 108 MiB in 41 ms"
        );
    }

    /// The disk tier's lines report whole MiB below a gibibyte and tenths of a
    /// GiB above, and the scan line stays quiet about a rejection count of zero.
    #[test]
    fn the_disk_cache_lines_report_sizes_and_only_name_rejections_when_there_were_any() {
        assert_eq!(
            line(ServeLog::DiskCacheScanned {
                segments: 3,
                bytes: 4_402_341_478,
                rejected: 0,
            }),
            "xwen serve: disk cache: 3 segments (4.1 GiB) on disk"
        );
        assert_eq!(
            line(ServeLog::DiskCacheScanned {
                segments: 1,
                bytes: 113_246_208,
                rejected: 2,
            }),
            "xwen serve: disk cache: 1 segment (108 MiB) on disk, 2 rejected"
        );
        assert_eq!(
            line(ServeLog::DiskCacheScanned {
                segments: 0,
                bytes: 0,
                rejected: 0,
            }),
            "xwen serve: disk cache: 0 segments (0 MiB) on disk"
        );
        assert_eq!(
            line(ServeLog::DiskSegmentWritten {
                role: DiskSegmentRole::Tail,
                tokens: 20_014,
                bytes: 1_287_651_328,
                ms: 820,
            }),
            "xwen serve: disk cache: wrote a 20014-token tail segment, 1.2 GiB in 820 ms"
        );
        assert_eq!(
            line(ServeLog::DiskSegmentWritten {
                role: DiskSegmentRole::Base,
                tokens: 19_012,
                bytes: 1_182_793_728,
                ms: 700,
            }),
            "xwen serve: disk cache: wrote a 19012-token base segment, 1.1 GiB in 700 ms"
        );
        assert_eq!(
            line(ServeLog::DiskSegmentSplit { at: 19_012 }),
            "xwen serve: disk cache: split a stored segment at 19012 tokens"
        );
        assert_eq!(
            line(ServeLog::DiskBoundarySnapshotStored { at: 20_457 }),
            "xwen serve: disk cache: stored a resume point at 20457 tokens in the segment \
             ending there"
        );
        assert_eq!(
            line(ServeLog::DiskChainHydrated {
                segments: 3,
                tokens: 20_014,
                resume: 19_012,
                bytes: 1_287_651_328,
                ms: 640,
            }),
            "xwen serve: disk cache: hydrated a 3-segment chain of 20014 tokens, \
             resuming at 19012: 1.2 GiB in 640 ms"
        );
        assert_eq!(
            line(ServeLog::DiskSegmentEvicted {
                reason: DiskEvictReason::Superseded,
                bytes: 113_246_208,
            }),
            "xwen serve: disk cache: deleted a 108 MiB segment (superseded)"
        );
        assert_eq!(
            line(ServeLog::DiskSegmentEvicted {
                reason: DiskEvictReason::Budget,
                bytes: 2_147_483_648,
            }),
            "xwen serve: disk cache: deleted a 2.0 GiB segment (over the size budget)"
        );
        assert_eq!(
            line(ServeLog::DiskSegmentEvicted {
                reason: DiskEvictReason::Invalid { class: "corrupt" },
                bytes: 4096,
            }),
            "xwen serve: disk cache: deleted a 0 MiB segment (invalid: corrupt)"
        );
        assert_eq!(
            line(ServeLog::DiskSegmentEvicted {
                reason: DiskEvictReason::Collided,
                bytes: 1_287_651_328,
            }),
            "xwen serve: disk cache: deleted a 1.2 GiB segment \
             (a different conversation's history)"
        );
        assert_eq!(
            line(ServeLog::DiskSegmentEvicted {
                reason: DiskEvictReason::Orphaned,
                bytes: 4096,
            }),
            "xwen serve: disk cache: deleted a 0 MiB segment (no chain to a stored root)"
        );
        assert_eq!(
            line(ServeLog::DiskCacheFailed {
                action: "writing an image",
                error: "No space left on device (os error 28)".to_string(),
            }),
            "xwen serve: disk cache: writing an image failed (No space left on device \
             (os error 28))"
        );
    }

    #[test]
    fn the_queue_lines_report_the_drop_and_the_out_of_order_pick() {
        assert_eq!(
            line(ServeLog::QueuedRequestDropped {
                waited: Duration::from_millis(301_400),
                queue_timeout: Duration::from_secs(300),
            }),
            "xwen serve: dropped a request that waited 301s in the queue (queue_timeout 300s)"
        );
        assert_eq!(
            line(ServeLog::OutOfOrderPick {
                prompt_tokens: 1204,
                cost: 314,
                older: 2,
                oldest_waited: Duration::from_millis(4460),
            }),
            "xwen serve: running a 1204-token request (314 tokens of prefill after the cache) \
             ahead of 2 older ones (the oldest has waited 4.5s)"
        );
    }

    /// The header value is quoted, so an unprintable one is readable in the log.
    #[test]
    fn the_invalid_header_line_quotes_the_value() {
        assert_eq!(
            line(ServeLog::InvalidHeaderDropped {
                name: "retry-after",
                value: "1\n2".to_string(),
            }),
            r#"xwen serve: dropped invalid header retry-after: "1\n2""#
        );
    }

    /// A signal that could not be registered says which ways down are left, so
    /// an operator whose `pkill` does nothing can read why.
    #[test]
    fn the_unavailable_signal_line_names_the_signal_and_what_still_works() {
        assert_eq!(
            line(ServeLog::SignalUnavailable {
                signal: "SIGTERM",
                error: "Too many open files".to_string(),
            }),
            "xwen serve: cannot listen for SIGTERM (Too many open files); \
             Ctrl-C and the dashboard's q still shut the server down"
        );
    }

    #[test]
    fn the_shutdown_lines_are_verbatim() {
        assert_eq!(
            line(ServeLog::ShutdownRequested),
            "xwen serve: shutting down, waiting for in-flight requests"
        );
        assert_eq!(line(ServeLog::ShuttingDown), "xwen serve shutting down");
        assert_eq!(
            line(ServeLog::ShutdownGraceExpired {
                grace: Duration::from_secs(30),
            }),
            "xwen serve: still shutting down after 30s; exiting"
        );
    }

    /// A line whose author is below `serve` is carried, not reworded: the sink
    /// is a route for it, not a second voice.
    #[test]
    fn a_host_line_renders_verbatim() {
        assert_eq!(
            line(ServeLog::HostLine(
                "xwen: weights 63.5GB + KV 4.5GB = 68.0GB resident (max_ctx 4096)".to_string(),
            )),
            "xwen: weights 63.5GB + KV 4.5GB = 68.0GB resident (max_ctx 4096)"
        );
    }

    /// The instrumentation events say nothing on stderr, and that is a contract:
    /// headless workflows read this output with `2>` and grep, and a tick fired
    /// per prefill chunk and per decoded token would drown every line above.
    #[test]
    fn the_instrumentation_events_print_nothing() {
        let origin = RequestOrigin {
            id: 7,
            dialect: crate::serve::types::Dialect::Anthropic,
            streaming: true,
        };
        let record = JobRecord {
            origin,
            stop: Some(StopKind::EndTurn),
            abandoned: None,
            error: None,
            prompt_tokens: 380,
            cache_read: 380,
            prefill_tokens: 0,
            prefill_secs: 0.0,
            output_tokens: 38,
            thinking_tokens: 0,
            decode_secs: 3.1,
            ttft_secs: Some(1.4),
            spec: None,
        };
        let silent = [
            ServeLog::JobPicked {
                origin,
                prompt_tokens: 380,
                queue_wait: Duration::from_millis(4),
                deadline: Some(Instant::now() + Duration::from_secs(60)),
            },
            ServeLog::JobCacheResolved {
                id: 7,
                cached: 380,
                resume: 380,
                slot: 0,
            },
            ServeLog::JobDone(Box::new(record)),
            ServeLog::PrefillTick {
                done: 512,
                total: 34_112,
            },
            ServeLog::DecodeTick {
                tokens_out: 12,
                thinking: true,
            },
            ServeLog::SlotsSnapshot(vec![SlotSummary {
                live: true,
                tokens: 380,
                snapshots: 2,
                image_bytes: 113_246_208,
                has_drafter: true,
                last_used: 4,
                agrees_to: 380,
            }]),
            ServeLog::QueueSnapshot(vec![QueueEntry {
                id: 8,
                prompt_tokens: 1204,
                queued_at: Instant::now(),
            }]),
        ];
        for event in silent {
            assert!(event.render().is_none(), "{event:?} must not reach stderr");
        }
    }

    /// A logger whose sink is gone stays usable: the sites that log while
    /// holding the queue mutex cannot afford an error path.
    #[test]
    fn logging_into_a_dead_sink_is_a_no_op() {
        let (logger, handle) = spawn_stderr_sink();
        drop(handle);
        logger.log(ServeLog::ShuttingDown);
        let discarding = ServeLogger::discarding();
        discarding.clone().log(ServeLog::ShuttingDown);
    }

    /// The sink ends on its stop message even while senders are still alive —
    /// the global one never drops — so joining it cannot hang.
    #[test]
    fn the_sink_stops_while_senders_are_still_alive() {
        let (logger, handle) = spawn_stderr_sink();
        let held = logger.clone();
        drop(handle);
        held.log(ServeLog::ShuttingDown);
    }
}
