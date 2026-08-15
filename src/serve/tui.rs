//! The `--tui` sink: the same [`ServeLog`] stream the stderr sink prints, drawn
//! as a live dashboard instead of written as lines.
//!
//! This is a consumer and nothing else. It never touches the model, the engine
//! or the queue — everything it shows arrived as an event, which is what makes a
//! dashboard safe to run next to an inference thread that owns Metal.
//!
//! The loop has one select point: a `recv_timeout` on the sink channel, into
//! which a small input thread forwards terminal events as
//! [`SinkMessage::Term`]. A keypress is therefore answered as promptly as an
//! engine event is, without a second channel and without polling.
//!
//! The terminal is handed back on every way out — a normal quit, a panic (the
//! hook below), and the shutdown lines at the end of a run, which are printed as
//! plain stderr lines because by then nobody is looking at a pane any more.

use std::collections::VecDeque;
use std::io::{self, IsTerminal, Stderr};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU8, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Once};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crossterm::cursor::Show;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Frame;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Gauge, List, ListItem, ListState, Paragraph, Row, Table};

use super::QuitSignal;
use super::config::ServeSettings;
use super::log::{JobRecord, QueueEntry, ServeLog, SinkMessage, SlotSummary};
use super::types::{RequestOrigin, StopKind};

/// How long the loop parks before redrawing anyway. The uptime, the queue ages
/// and the deadline countdown all move on their own, so a quiet server still
/// has to repaint.
const POLL: Duration = Duration::from_millis(100);

/// Floor between two draws. Decode ticks arrive about twenty a second and
/// prefill chunks in bursts; a frame per event would spend the terminal's
/// bandwidth on pictures nobody can read.
const REDRAW_INTERVAL: Duration = Duration::from_millis(100);

/// How long the input thread waits for a keypress before checking whether the
/// dashboard still wants it. Bounds how long a torn-down TUI keeps a reader on
/// the terminal, which would otherwise swallow what the operator types at their
/// shell while the server finishes shutting down.
const INPUT_POLL: Duration = Duration::from_millis(200);

/// Finished requests kept for the HISTORY pane.
const HISTORY_CAPACITY: usize = 50;

/// Lines kept for the LOG pane.
const LOG_CAPACITY: usize = 500;

/// How often the machine facts in the header — the power mode and the UTC
/// offset — are re-read.
const POWER_INTERVAL: Duration = Duration::from_secs(60);

/// Weight of the newest sample in the throughput averages. Low enough that one
/// slow chunk does not swing the number an operator is reading, high enough to
/// follow a real change within a few ticks.
const RATE_ALPHA: f64 = 0.3;

/// Shortest interval a throughput is measured over. Two ticks a millisecond
/// apart are the channel's own jitter — a batch boundary, a scheduler hiccup —
/// and dividing by that interval invents a number in the hundreds. Samples
/// closer together than this are folded into the next one instead, which loses
/// nothing: the counts are cumulative.
const MIN_RATE_INTERVAL: Duration = Duration::from_millis(10);

/// How long without a tick before the average stops being believed. Prefill
/// chunks land every few hundred milliseconds and decode tokens about twenty a
/// second, so three seconds of silence is the engine not moving rather than one
/// slow step — and "is it stuck?" is the question the dashboard exists to
/// answer.
const STALL_AFTER: Duration = Duration::from_secs(3);

/// Set by the panic hook once the terminal has been handed back, so the loop
/// stops drawing into a screen it no longer owns. Only panics nobody catches set
/// it: the engine's own recovered panics leave the dashboard running, since the
/// server is still serving and the incident reaches the LOG pane as events.
static SURRENDERED: AtomicBool = AtomicBool::new(false);

/// What the header says about the server, gathered before anything is logged.
///
/// The context length and the address are starting values: the checkpoint may
/// clamp the first and the listener resolves the second, and both arrive as
/// events the dashboard folds in.
#[derive(Debug, Clone)]
pub struct Vitals {
    pub model_id: String,
    /// The checkpoint's size on disk, or `None` if it could not be read.
    pub model_bytes: Option<u64>,
    pub context_length: usize,
    pub port: u16,
    /// Whether the checkpoint currently LOADED is speculating. Starts at
    /// `draft_configured` — nothing is loaded yet, and a header that said "off"
    /// on a drafting server would be wrong for the whole first request — and is
    /// then driven by the engine's own drafter events, because which checkpoint
    /// is resident decides this and a swap can change it.
    pub draft: bool,
    /// What the SETTINGS asked for, which is what the cell falls back to
    /// whenever nothing is loaded (a swap, an idle unload). Without it the cell
    /// would have to invent an answer for "no model resident" — and inventing
    /// "off" there would read as a configuration problem on a drafting server.
    pub draft_configured: bool,
    /// Whether the on-disk prefix cache is on, which is what decides whether the
    /// header carries a cell for it: a server that keeps nothing on disk has no
    /// business showing a size of zero.
    pub disk_cache: bool,
}

impl Vitals {
    pub fn new(model_id: String, settings: &ServeSettings) -> Self {
        Self {
            model_id,
            model_bytes: file_size(&settings.model),
            context_length: settings.context_length,
            port: settings.port,
            draft: settings.draft.is_on(),
            draft_configured: settings.draft.is_on(),
            disk_cache: settings.disk_cache && settings.cache_dir.is_some(),
        }
    }
}

fn file_size(path: &Path) -> Option<u64> {
    std::fs::metadata(path).ok().map(|meta| meta.len())
}

/// Facts about the machine the render loop wants but must not go and get: both
/// come from a subprocess, so a helper thread samples them and the loop reads
/// the answers.
#[derive(Debug, Default)]
struct Environment {
    /// 0 = not known, 1 = low-power mode, 2 = full power. Unknown draws no
    /// badge: a machine whose power mode cannot be read is not a machine at full
    /// power, and guessing either way would make the badge worthless.
    power: AtomicU8,
    /// Seconds to add to a UTC timestamp for local time, re-read on the same
    /// schedule as the power mode so a daylight-saving change lands. An offset
    /// that cannot be read leaves the clock in UTC, which is wrong by hours but
    /// never wrong about the order of two lines.
    utc_offset: AtomicI64,
}

impl Environment {
    fn sample(&self) {
        if let Some(offset) = read_utc_offset() {
            self.utc_offset.store(offset, Ordering::Relaxed);
        }
        self.sample_power();
    }

    fn sample_power(&self) {
        if let Some(low) = read_low_power_mode() {
            self.power.store(if low { 1 } else { 2 }, Ordering::Relaxed);
        }
    }
}

/// The low-power-mode flag, or `None` when this machine reports no such key.
/// The key is spelled `lowpowermode` on some OS builds and `powermode` on
/// others, so both are accepted.
fn read_low_power_mode() -> Option<bool> {
    let output = std::process::Command::new("pmset")
        .arg("-g")
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        // A blank or unnamed line is skipped rather than ending the scan: the
        // key sits in a section partway down the output.
        let Some(key) = fields.next() else { continue };
        if key == "lowpowermode" || key == "powermode" {
            return fields
                .next()
                .and_then(|value| value.parse::<u8>().ok())
                .map(|v| v != 0);
        }
    }
    None
}

/// This machine's UTC offset in seconds. `std` has no local time and the
/// dashboard's clock is there to be compared against a client's log, so the
/// offset is worth a subprocess a minute — which is also what carries a
/// long-running server across a daylight-saving change.
fn read_utc_offset() -> Option<i64> {
    let output = std::process::Command::new("date")
        .arg("+%z")
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let field = text.trim();
    let (sign, digits) = field.split_at(field.len().checked_sub(4)?);
    let hours: i64 = digits.get(..2)?.parse().ok()?;
    let minutes: i64 = digits.get(2..)?.parse().ok()?;
    let magnitude = hours * 3600 + minutes * 60;
    Some(if sign == "-" { -magnitude } else { magnitude })
}

/// A throughput average over ticks as they arrive. The ticks carry no timestamp
/// of their own, and at a hundred-millisecond loop the moment one is received is
/// as close to when it happened as anything the dashboard could show.
#[derive(Debug, Default, Clone, Copy)]
struct Rate {
    last: Option<(Instant, usize)>,
    value: Option<f64>,
}

impl Rate {
    fn observe(&mut self, at: Instant, count: usize) {
        let Some((then, before)) = self.last else {
            self.last = Some((at, count));
            return;
        };
        // Every tick a drain pulls out at once shares one instant, and only the
        // first of them moves the baseline. The counts are cumulative, so what
        // the rest of the batch produced is not lost: the next batch measures
        // from the same baseline and picks all of it up. The same holds for an
        // interval too short to divide by.
        let interval = at.saturating_duration_since(then);
        if interval < MIN_RATE_INTERVAL {
            return;
        }
        let seconds = interval.as_secs_f64();
        let produced = count.saturating_sub(before) as f64;
        if produced > 0.0 {
            let sample = produced / seconds;
            self.value = Some(match self.value {
                Some(average) => average * (1.0 - RATE_ALPHA) + sample * RATE_ALPHA,
                None => sample,
            });
        }
        self.last = Some((at, count));
    }

    /// How long since the last tick, once that is long enough to mean the work
    /// has stopped. The average is a lagging figure by construction, so a job
    /// that wedged would otherwise go on reporting the throughput it had when it
    /// was still moving.
    fn stalled_for(&self, now: Instant) -> Option<Duration> {
        let (then, _) = self.last?;
        let idle = now.saturating_duration_since(then);
        (idle >= STALL_AFTER).then_some(idle)
    }
}

/// Prefill progress over the whole prompt, cache-read tokens included.
#[derive(Debug, Default, Clone, Copy)]
struct Progress {
    done: usize,
    total: usize,
    rate: Rate,
}

/// What decoding has produced so far.
#[derive(Debug, Default, Clone, Copy)]
struct Decode {
    tokens_out: usize,
    thinking: bool,
    rate: Rate,
}

/// What the prefix cache could do for the running request.
#[derive(Debug, Clone, Copy)]
struct Cached {
    cached: usize,
    resume: usize,
    slot: usize,
}

/// The request the engine is running, as its events describe it.
#[derive(Debug, Clone)]
struct Live {
    origin: RequestOrigin,
    prompt_tokens: usize,
    /// Whether `prompt_tokens` is the queue's bytes-based overestimate (a
    /// batch at pickup) rather than a real token count — shown as such.
    estimated: bool,
    queue_wait: Duration,
    deadline: Option<Instant>,
    cache: Option<Cached>,
    prefill: Option<Progress>,
    decode: Option<Decode>,
}

/// One finished request, stamped with when the dashboard heard about it —
/// [`JobRecord`] carries durations rather than a wall clock.
#[derive(Debug, Clone)]
struct Finished {
    at: SystemTime,
    record: JobRecord,
}

/// Everything the dashboard knows, folded from the event stream.
struct Dashboard {
    vitals: Vitals,
    started: Instant,
    environment: Arc<Environment>,
    /// The address the listener actually bound, once it says so.
    address: Option<String>,
    live: Option<Live>,
    queue: Vec<QueueEntry>,
    slots: Vec<SlotSummary>,
    history: VecDeque<Finished>,
    logs: VecDeque<String>,
    log_state: ListState,
    /// Whether the LOG pane sticks to the newest line. Any scroll turns it off;
    /// `End` turns it back on.
    follow: bool,
    served: usize,
    failed: usize,
    /// What the on-disk prefix cache holds, folded from the scan and every write
    /// and deletion since.
    disk: Disk,
}

/// The disk tier's size, as the header shows it.
#[derive(Debug, Clone, Copy, Default)]
struct Disk {
    segments: usize,
    bytes: u64,
}

impl Dashboard {
    fn new(vitals: Vitals, started: Instant, environment: Arc<Environment>) -> Self {
        Self {
            vitals,
            started,
            environment,
            address: None,
            live: None,
            queue: Vec::new(),
            slots: Vec::new(),
            history: VecDeque::new(),
            logs: VecDeque::new(),
            log_state: ListState::default(),
            follow: true,
            served: 0,
            failed: 0,
            disk: Disk::default(),
        }
    }

    /// Fold one event in. `at` is when it was received, which is what the
    /// throughput averages measure against; `wall` stamps the history.
    fn apply(&mut self, event: ServeLog, at: Instant, wall: SystemTime) {
        if let Some(line) = event.render() {
            self.push_log(wall, &line);
        }
        match event {
            // The checkpoint's own limit wins over the configured one, so the
            // header stops advertising a context the server will not serve.
            ServeLog::ContextClamped { trained, .. } => self.vitals.context_length = trained,
            ServeLog::Listening { address, .. } => self.address = Some(address),
            // The drafting cell tracks what is LOADED, not what was configured:
            // speculation is per checkpoint (one may ship no sidecar while the
            // next one does), so a header that reported the setting would claim
            // drafting on a checkpoint that decodes plain.
            //
            // Only the two drafter events decide it, and only the events that
            // END a residency reset it. Deliberately NOT keyed off
            // `ModelLoaded`: that fires AFTER `EngineState::load` returns, while
            // `DrafterLoaded` fires from inside it, so a "clear on load" arm
            // runs last and pins the cell off on every drafting server. With
            // nothing loaded the cell shows the configured intent, which is the
            // honest answer before a checkpoint has been chosen.
            ServeLog::DrafterLoaded { .. } => self.vitals.draft = true,
            ServeLog::NoDrafterAvailable { .. } => self.vitals.draft = false,
            ServeLog::CheckpointSwappingOut { .. } | ServeLog::IdleUnloaded { .. } => {
                self.vitals.draft = self.vitals.draft_configured;
            }
            ServeLog::JobPicked {
                origin,
                prompt_tokens,
                estimated,
                queue_wait,
                deadline,
            } => {
                self.live = Some(Live {
                    origin,
                    prompt_tokens,
                    estimated,
                    queue_wait,
                    deadline,
                    cache: None,
                    prefill: None,
                    decode: None,
                });
            }
            ServeLog::JobCacheResolved {
                cached,
                resume,
                slot,
                ..
            } => {
                if let Some(live) = self.live.as_mut() {
                    live.cache = Some(Cached {
                        cached,
                        resume,
                        slot,
                    });
                }
            }
            ServeLog::PrefillTick { done, total } => {
                if let Some(live) = self.live.as_mut() {
                    let progress = live.prefill.get_or_insert_with(Progress::default);
                    progress.done = done;
                    progress.total = total;
                    progress.rate.observe(at, done);
                }
            }
            ServeLog::DecodeTick {
                tokens_out,
                thinking,
            } => {
                if let Some(live) = self.live.as_mut() {
                    let decode = live.decode.get_or_insert_with(Decode::default);
                    decode.tokens_out = tokens_out;
                    decode.thinking = thinking;
                    decode.rate.observe(at, tokens_out);
                }
            }
            ServeLog::JobDone(record) => {
                if record.error.is_some() {
                    self.failed += 1;
                } else {
                    self.served += 1;
                }
                if self.history.len() == HISTORY_CAPACITY {
                    self.history.pop_back();
                }
                self.history.push_front(Finished {
                    at: wall,
                    record: *record,
                });
                self.live = None;
            }
            ServeLog::SlotsSnapshot(slots) => self.slots = slots,
            ServeLog::QueueSnapshot(queue) => self.queue = queue,
            // The store as the startup scan found it, then followed by the writes
            // and deletions that change it. Every one of those events is exactly one
            // file, which is what keeps a count folded from them in step with the
            // directory without anyone going and looking.
            ServeLog::DiskCacheScanned {
                segments, bytes, ..
            } => {
                self.disk = Disk { segments, bytes };
            }
            ServeLog::DiskSegmentWritten { bytes, .. } => {
                self.disk.segments += 1;
                self.disk.bytes += bytes;
            }
            ServeLog::DiskSegmentEvicted { bytes, .. } => {
                self.disk.segments = self.disk.segments.saturating_sub(1);
                self.disk.bytes = self.disk.bytes.saturating_sub(bytes);
            }
            _ => {}
        }
    }

    fn push_log(&mut self, wall: SystemTime, line: &str) {
        if self.logs.len() == LOG_CAPACITY {
            self.logs.pop_front();
            // The selection is an index into a ring that just slid under it, so
            // it follows the line it was on. Without this the view walks
            // forward one line per eviction, which under load is a page a
            // second — an operator reading back through the log would be shown
            // something else every time they looked up.
            if !self.follow
                && let Some(selected) = self.log_state.selected()
            {
                self.log_state.select(Some(selected.saturating_sub(1)));
            }
        }
        let stamp = clock(wall, self.environment.utc_offset.load(Ordering::Relaxed));
        self.logs.push_back(format!("{stamp} {line}"));
    }

    /// A key the dashboard understands. `true` means the operator asked to quit.
    fn key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('q') => return true,
            // Raw mode generates no SIGINT, so the habitual way out has to be
            // recognised as an ordinary keypress.
            KeyCode::Char('c') if ctrl => return true,
            KeyCode::Up => self.scroll_by(-1),
            KeyCode::Down => self.scroll_by(1),
            KeyCode::PageUp => self.scroll_by(-10),
            KeyCode::PageDown => self.scroll_by(10),
            KeyCode::Home => {
                // Nothing to read back through yet, so the pane keeps following:
                // latching follow off over an empty ring would leave the next
                // line to arrive stranded off-screen.
                if self.logs.is_empty() {
                    return false;
                }
                self.follow = false;
                self.log_state.select(Some(0));
            }
            KeyCode::End => self.follow = true,
            _ => {}
        }
        false
    }

    /// Move the LOG selection, which is also what toggles follow: an operator
    /// reading back through the log must not have the pane yanked out from under
    /// them by the next event, and one who scrolls back DOWN onto the newest
    /// line has arrived at the live edge and wants it live again — without this,
    /// a pane once scrolled stays pinned to that line forever unless they know
    /// about End.
    fn scroll_by(&mut self, lines: isize) {
        if self.logs.is_empty() {
            return;
        }
        let last = self.logs.len() - 1;
        let current = self.log_state.selected().unwrap_or(last) as isize;
        let next = (current + lines).clamp(0, last as isize) as usize;
        self.follow = next == last;
        self.log_state.select(Some(next));
    }
}

// ---------------------------------------------------------------------------
// Formatting. Kept as free functions over plain values so the wording is
// testable without a terminal.
// ---------------------------------------------------------------------------

/// Thousands separators, because the token counts here run to six digits and an
/// operator is comparing them against a context length.
fn commas(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, digit) in digits.char_indices() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// A file size in the units a checkpoint is discussed in.
fn giga(bytes: u64) -> String {
    format!("{:.1}G", bytes as f64 / 1e9)
}

/// The disk tier's size in the binary units its own log lines use, so the header
/// and the LOG pane never disagree about how big the store is.
fn gibi(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

/// Host bytes, in the units the paging lines already use.
fn mib(bytes: usize) -> String {
    format!("{:.0} MiB", bytes as f64 / (1024.0 * 1024.0))
}

/// A span of server uptime, coarsening as it grows.
fn uptime(elapsed: Duration) -> String {
    let secs = elapsed.as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else {
        format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60)
    }
}

/// A short span: sub-minute to a tenth, above that to the second.
fn seconds(elapsed: Duration) -> String {
    let secs = elapsed.as_secs_f64();
    if secs < 60.0 {
        format!("{secs:.1}s")
    } else {
        format!("{}m{:02}s", secs as u64 / 60, secs as u64 % 60)
    }
}

/// Wall-clock time of day, offset into local time.
fn clock(at: SystemTime, utc_offset: i64) -> String {
    let epoch = at
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs() as i64);
    let day = (epoch + utc_offset).rem_euclid(86_400);
    format!("{:02}:{:02}:{:02}", day / 3600, (day % 3600) / 60, day % 60)
}

/// A throughput, or a placeholder while there is nothing to average yet.
fn rate(value: Option<f64>) -> String {
    value.map_or_else(|| "— tok/s".to_string(), |v| format!("{v:.1} tok/s"))
}

/// What a running phase is producing: its throughput, or how long it has been
/// producing nothing.
fn throughput(rate_of: &Rate, now: Instant) -> String {
    match rate_of.stalled_for(now) {
        Some(idle) => format!("stalled {}", seconds(idle)),
        None => rate(rate_of.value),
    }
}

/// A `Duration` from a span of seconds that was measured rather than computed.
/// The engine's own spans are never negative and never astronomical, but
/// `Duration::from_secs_f64` panics on both, and the dashboard must not be the
/// thing that takes a healthy server down. A NaN reads as zero, since `max`
/// returns the operand that is a number.
fn span(secs: f64) -> Duration {
    Duration::try_from_secs_f64(secs.max(0.0)).unwrap_or(Duration::ZERO)
}

/// How a finished request ended. An abandoned one is marked with a `*` and named
/// by what actually cut it short, which is not always what the client was told:
/// a deadline truncation reports `max_tokens` on the wire. A batch reports its
/// item arithmetic — it has no single stop reason, and per-item failures are
/// its outcome in a way one word cannot be.
fn outcome(record: &JobRecord) -> String {
    if record.error.is_some() {
        return "error".to_string();
    }
    if let Some(reason) = record.abandoned {
        return format!("{}*", reason.label().replace(' ', "-"));
    }
    if let Some(batch) = &record.batch {
        return if batch.failed > 0 {
            format!("{}/{} items", batch.items - batch.failed, batch.items)
        } else {
            format!("{} items", batch.items)
        };
    }
    match &record.stop {
        Some(StopKind::EndTurn) => "end_turn".to_string(),
        Some(StopKind::MaxTokens) => "max_tokens".to_string(),
        Some(StopKind::StopSequence(_)) => "stop_seq".to_string(),
        Some(StopKind::ToolUse) => "tool_use".to_string(),
        None => "—".to_string(),
    }
}

/// The token arithmetic of one finished request: what the cache supplied, what
/// was prefilled, and what came out.
fn token_flow(record: &JobRecord) -> String {
    format!(
        "{}+{}→{}",
        commas(record.cache_read),
        commas(record.prefill_tokens),
        commas(record.output_tokens)
    )
}

/// One phase's cell — its wall time and its own rate — or a dash for a phase
/// that never ran. Prefill and decode each get one of these; blending them
/// into a single figure was tried and read as neither.
fn phase_cell(tokens: usize, secs: f64) -> String {
    if secs <= 0.0 || tokens == 0 {
        return "—".to_string();
    }
    format!("{} · {:.0} t/s", seconds(span(secs)), tokens as f64 / secs)
}

/// The prefill column: tokens actually forwarded and the time they took. For a
/// batch that is engine work — shared prefix once, tails, scored trials — not
/// the logical prompt size the flow column shows.
fn prefill_cell(record: &JobRecord) -> String {
    match &record.batch {
        Some(batch) => phase_cell(batch.prefill_tokens, batch.prefill_secs),
        None => phase_cell(record.prefill_tokens, record.prefill_secs),
    }
}

/// The decode column: tokens actually sampled and the time they took. A scored
/// batch's assembled answers are teacher-forced, so it decodes only reasoning —
/// often nothing, and the dash is the honest cell.
fn decode_cell(record: &JobRecord) -> String {
    match &record.batch {
        Some(batch) => phase_cell(batch.decode_tokens, batch.decode_secs),
        None => phase_cell(record.output_tokens, record.decode_secs),
    }
}

/// The catch-all last column: TTFT and speculation for a generation, the whole
/// wall run for a batch (host-side work lives in neither phase column).
fn detail_cell(record: &JobRecord) -> String {
    let mut parts = Vec::new();
    if let Some(batch) = &record.batch {
        parts.push(format!("total {}", seconds(span(batch.secs))));
    }
    if let Some(ttft) = record.ttft_secs {
        parts.push(format!("TTFT {}", seconds(span(ttft))));
    }
    if let Some(spec) = &record.spec
        && spec.drafted > 0
    {
        parts.push(format!("spec {:.0}%", spec.acceptance_rate() * 100.0));
    }
    if parts.is_empty() {
        return "—".to_string();
    }
    parts.join(" · ")
}

// ---------------------------------------------------------------------------
// Drawing.
// ---------------------------------------------------------------------------

/// Below this width the QUEUE and SLOTS panes stack instead of sitting side by
/// side: two four-column tables in forty characters each is unreadable.
const SIDE_BY_SIDE_WIDTH: u16 = 96;

fn draw(frame: &mut Frame, app: &mut Dashboard, now: Instant) {
    let [header, live, middle, history, logs, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(5),
        Constraint::Length(6),
        Constraint::Percentage(30),
        Constraint::Fill(1),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    frame.render_widget(header_line(app, now), header);
    draw_live(frame, app, live, now);

    let (queue_area, slots_area) = if middle.width >= SIDE_BY_SIDE_WIDTH {
        let [left, right] =
            Layout::horizontal([Constraint::Percentage(45), Constraint::Fill(1)]).areas(middle);
        (left, right)
    } else {
        let [top, bottom] =
            Layout::vertical([Constraint::Percentage(50), Constraint::Fill(1)]).areas(middle);
        (top, bottom)
    };
    draw_queue(frame, app, queue_area, now);
    draw_slots(frame, app, slots_area);
    draw_history(frame, app, history);
    draw_logs(frame, app, logs);
    frame.render_widget(footer_line(app), footer);
}

fn header_line(app: &Dashboard, now: Instant) -> Paragraph<'static> {
    let mut parts = vec![app.vitals.model_id.clone()];
    if let Some(bytes) = app.vitals.model_bytes {
        parts.push(giga(bytes));
    }
    parts.push(format!(
        "draft {}",
        if app.vitals.draft { "ON" } else { "off" }
    ));
    parts.push(format!("ctx {}", commas(app.vitals.context_length)));
    if app.vitals.disk_cache {
        parts.push(format!(
            "disk: {} segs {}",
            app.disk.segments,
            gibi(app.disk.bytes)
        ));
    }
    parts.push(match &app.address {
        Some(address) => address.clone(),
        None => format!(":{}", app.vitals.port),
    });
    if app.environment.power.load(Ordering::Relaxed) == 1 {
        parts.push("LPM".to_string());
    }
    parts.push(format!(
        "up {}",
        uptime(now.saturating_duration_since(app.started))
    ));

    let mut spans = vec![Span::styled(
        format!(" {} ", parts.remove(0)),
        Style::new().fg(Color::Cyan).bold(),
    )];
    for part in parts {
        spans.push(Span::styled("· ", Style::new().dim()));
        spans.push(Span::raw(format!("{part} ")));
    }
    Paragraph::new(Line::from(spans))
}

fn draw_live(frame: &mut Frame, app: &Dashboard, area: Rect, now: Instant) {
    let block = Block::bordered().title(" NOW ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 {
        return;
    }
    let [top, details, tail] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Fill(1),
    ])
    .areas(inner);

    let Some(live) = &app.live else {
        frame.render_widget(
            Paragraph::new(Line::from(vec![Span::styled(
                "idle — waiting for a request",
                Style::new().dim(),
            )])),
            top,
        );
        return;
    };

    match (&live.prefill, &live.decode) {
        // Prefill only: the bar is the whole point of the pane, since this is
        // the phase that can run for minutes with nothing else to show.
        (Some(progress), None) => frame.render_widget(prefill_gauge(progress, now), top),
        _ => frame.render_widget(Paragraph::new(decode_line(live, now)), top),
    }

    frame.render_widget(Paragraph::new(detail_line(live, now)), details);

    // While prefill is running the decode line is the one that is still empty,
    // and once decode starts the finished prefill is what the wait was spent on.
    let tail_line = match (&live.prefill, &live.decode) {
        (Some(_), Some(_)) => Some(prefill_summary(live)),
        (Some(_), None) => None,
        _ => None,
    };
    if let Some(line) = tail_line {
        frame.render_widget(Paragraph::new(line), tail);
    }
}

fn prefill_gauge(progress: &Progress, now: Instant) -> Gauge<'static> {
    let ratio = if progress.total == 0 {
        0.0
    } else {
        (progress.done as f64 / progress.total as f64).clamp(0.0, 1.0)
    };
    // No estimate while nothing is moving: an ETA extrapolated from a rate the
    // work has stopped producing is the one number that would mislead here.
    let eta = match progress.rate.value {
        Some(per_second) if per_second > 0.0 && progress.rate.stalled_for(now).is_none() => {
            let left = progress.total.saturating_sub(progress.done) as f64;
            format!(" · ETA {}", seconds(span(left / per_second)))
        }
        _ => String::new(),
    };
    Gauge::default()
        .gauge_style(Style::new().fg(Color::Green))
        .ratio(ratio)
        .label(format!(
            "PREFILL {} / {} tok · {}{eta}",
            commas(progress.done),
            commas(progress.total),
            throughput(&progress.rate, now),
        ))
}

fn decode_line(live: &Live, now: Instant) -> Line<'static> {
    let Some(decode) = &live.decode else {
        return Line::from(vec![Span::styled(
            "DISPATCH · resolving the cache",
            Style::new().dim(),
        )]);
    };
    Line::from(vec![
        Span::styled("DECODE ", Style::new().fg(Color::Green).bold()),
        Span::raw(format!(
            "{} tok · {} · {}",
            commas(decode.tokens_out),
            throughput(&decode.rate, now),
            if decode.thinking {
                "thinking"
            } else {
                "answer"
            }
        )),
    ])
}

fn detail_line(live: &Live, now: Instant) -> Line<'static> {
    let mut parts = Vec::new();
    if let Some(cache) = &live.cache {
        // What was reused, which is what the wire reports as `cache_read`. The
        // prefix match is only worth a mention when it did not become a reuse:
        // that is the replay case — the KV cache disagreed and a warm
        // conversation went cold — and it is exactly what an operator watching
        // a slow request wants to see.
        let mut read = format!("cache_read {}", commas(cache.resume));
        if cache.resume != cache.cached {
            read.push_str(&format!(" (match {} discarded)", commas(cache.cached)));
        }
        parts.push(read);
        parts.push(format!("slot {}", cache.slot));
    }
    if live.estimated {
        // A batch's text is untokenized at pickup: this is the queue's loose
        // bytes-based overestimate, not a prompt anyone will prefill.
        parts.push(format!("prompt ~{} tok (est)", commas(live.prompt_tokens)));
    } else {
        parts.push(format!("prompt {} tok", commas(live.prompt_tokens)));
    }
    parts.push(format!("queued {}", seconds(live.queue_wait)));
    if let Some(deadline) = live.deadline {
        // The watchdog's kill ceiling — deliberately loose (it is derived from
        // the same overestimate), and nothing like an ETA.
        parts.push(format!(
            "watchdog in {}",
            seconds(deadline.saturating_duration_since(now))
        ));
    }
    parts.push(format!(
        "client {}{}",
        live.origin.dialect.label(),
        if live.origin.streaming {
            ", streaming"
        } else {
            ""
        }
    ));
    Line::from(Span::styled(parts.join(" · "), Style::new().dim()))
}

fn prefill_summary(live: &Live) -> Line<'static> {
    let Some(progress) = &live.prefill else {
        return Line::default();
    };
    Line::from(Span::styled(
        format!(
            "prefill {} / {} tok · {}",
            commas(progress.done),
            commas(progress.total),
            rate(progress.rate.value)
        ),
        Style::new().dim(),
    ))
}

fn draw_queue(frame: &mut Frame, app: &Dashboard, area: Rect, now: Instant) {
    let title = format!(" QUEUE ({}) ", app.queue.len());
    let rows: Vec<Row> = app
        .queue
        .iter()
        .map(|entry| {
            Row::new(vec![
                format!("#{}", entry.id),
                format!("{} tok", commas(entry.prompt_tokens)),
                // Aged at render rather than at snapshot: the snapshots only
                // arrive when the queue changes, and an entry waiting out a
                // five-minute prefill must not read "age 0.0s" for all of it.
                format!(
                    "age {}",
                    seconds(now.saturating_duration_since(entry.queued_at))
                ),
            ])
        })
        .collect();
    let widths = [
        Constraint::Length(6),
        Constraint::Fill(1),
        Constraint::Length(12),
    ];
    frame.render_widget(
        Table::new(rows, widths)
            .column_spacing(1)
            .block(Block::bordered().title(title)),
        area,
    );
}

fn draw_slots(frame: &mut Frame, app: &Dashboard, area: Rect) {
    let rows: Vec<Row> = app
        .slots
        .iter()
        .enumerate()
        .map(|(index, slot)| {
            let state = if slot.live { "LIVE" } else { "cold" };
            let mut extra = vec![format!("{} snap", slot.snapshots)];
            if slot.image_bytes > 0 {
                extra.push(mib(slot.image_bytes));
            }
            if slot.has_drafter {
                extra.push("drafter".to_string());
            }
            Row::new(vec![
                index.to_string(),
                state.to_string(),
                format!("{} tok", commas(slot.tokens)),
                extra.join(" · "),
            ])
            .style(if slot.live {
                Style::new().fg(Color::Green)
            } else {
                Style::new().dim()
            })
        })
        .collect();
    let widths = [
        Constraint::Length(2),
        Constraint::Length(4),
        Constraint::Length(12),
        Constraint::Fill(1),
    ];
    frame.render_widget(
        Table::new(rows, widths)
            .column_spacing(1)
            .block(Block::bordered().title(" SLOTS ")),
        area,
    );
}

fn draw_history(frame: &mut Frame, app: &Dashboard, area: Rect) {
    let offset = app.environment.utc_offset.load(Ordering::Relaxed);
    let rows: Vec<Row> = app
        .history
        .iter()
        .map(|finished| {
            let record = &finished.record;
            Row::new(vec![
                clock(finished.at, offset),
                record.origin.dialect.label().to_string(),
                token_flow(record),
                outcome(record),
                prefill_cell(record),
                decode_cell(record),
                detail_cell(record),
            ])
        })
        .collect();
    let widths = [
        Constraint::Length(8),
        Constraint::Length(9),
        Constraint::Length(20),
        Constraint::Length(13),
        Constraint::Length(17),
        Constraint::Length(17),
        Constraint::Fill(1),
    ];
    frame.render_widget(
        Table::new(rows, widths)
            .column_spacing(1)
            .header(
                Row::new(vec![
                    "time",
                    "api",
                    "cached+new→out",
                    "outcome",
                    "prefill",
                    "decode",
                    "detail",
                ])
                .style(Style::new().dim()),
            )
            .block(Block::bordered().title(" HISTORY ")),
        area,
    );
}

fn draw_logs(frame: &mut Frame, app: &mut Dashboard, area: Rect) {
    let items: Vec<ListItem> = app
        .logs
        .iter()
        .map(|line| ListItem::new(line.as_str()))
        .collect();
    if app.follow && !items.is_empty() {
        app.log_state.select(Some(items.len() - 1));
    }
    let mut list = List::new(items).block(Block::bordered().title(" LOG "));
    if !app.follow {
        list = list.highlight_style(Style::new().add_modifier(Modifier::REVERSED));
    }
    frame.render_stateful_widget(list, area, &mut app.log_state);
}

fn footer_line(app: &Dashboard) -> Paragraph<'static> {
    let follow = if app.follow {
        "following"
    } else {
        "scrolled (End follows)"
    };
    Paragraph::new(Line::from(Span::styled(
        format!(
            " q quit · ↑↓ PgUp/PgDn scroll · {follow} · {} served, {} failed",
            app.served, app.failed
        ),
        Style::new().dim(),
    )))
}

// ---------------------------------------------------------------------------
// Terminal glue.
// ---------------------------------------------------------------------------

/// The alternate screen, for as long as the dashboard owns it.
///
/// Everything is on stderr: stdout belongs to whatever the operator might be
/// piping, and ratatui's own `init`/`restore` helpers are hardcoded to stdout,
/// so the fifteen lines below are hand-rolled instead.
struct Screen {
    terminal: ratatui::Terminal<CrosstermBackend<Stderr>>,
}

impl Screen {
    fn open() -> io::Result<Self> {
        // The check has to be explicit and has to be on stderr — the fd the
        // dashboard draws on. Raw mode alone is no guard: crossterm resolves
        // its fd through stdin and then falls back to opening /dev/tty
        // directly, so with stderr redirected to a file it happily succeeds,
        // seizes the operator's terminal, and pours ANSI frames into the
        // redirect target. With the dashboard on by default, `2> serve.log`
        // must land on the plain-lines fallback instead.
        if !io::stderr().is_terminal() {
            return Err(io::Error::other("stderr is not a terminal"));
        }
        enable_raw_mode()?;
        if let Err(e) = execute!(io::stderr(), EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(e);
        }
        // From here the terminal is ours and nothing yet owns handing it back:
        // every step below writes to stderr and can fail — a full device, a
        // closed pipe — and returning from one of them without restoring would
        // leave the operator's shell raw on the alternate screen with the
        // dashboard reporting that it could not start.
        let mut handback = Handback { armed: true };
        install_panic_hook();
        let mut terminal = ratatui::Terminal::new(CrosstermBackend::new(io::stderr()))?;
        terminal.hide_cursor()?;
        // Not Terminal::clear(): ratatui's clear round-trips a cursor-position
        // query (ESC[6n) that crossterm writes to STDOUT and waits 2s to read
        // back. The dashboard lives on stderr precisely so stdout can be
        // redirected — in that case the query never reaches the terminal and
        // the timeout aborts the dashboard. Clearing the stderr screen
        // directly needs no reply from the terminal.
        execute!(io::stderr(), Clear(ClearType::All))?;
        handback.armed = false;
        Ok(Self { terminal })
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        restore();
    }
}

/// Hands the terminal back if [`Screen::open`] gives up after taking it, for the
/// window before a `Screen` exists to do it.
struct Handback {
    armed: bool,
}

impl Drop for Handback {
    fn drop(&mut self) {
        if self.armed {
            restore();
        }
    }
}

/// Hand the terminal back. Raw mode goes first: leaving the alternate screen
/// while the terminal is still raw redraws the shell's prompt into a line
/// discipline that will not echo.
fn restore() {
    let _ = disable_raw_mode();
    let _ = execute!(io::stderr(), LeaveAlternateScreen, Show);
}

/// Restore the terminal before anything else prints a panic message into it.
/// The previous hook is chained rather than replaced, and the flag tells the
/// draw loop the screen is no longer its own.
///
/// The engine thread is the exception, and the reason the thread has a name at
/// all: it runs every job behind a panic boundary it recovers from — a drafter
/// that cannot load fails one request and the server keeps serving — so
/// surrendering the terminal there would cost a healthy server its dashboard
/// over an incident the operator can read in the LOG pane. Every other thread's
/// panic is one nobody caught, and the terminal goes back.
fn install_panic_hook() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if std::thread::current().name() == Some(super::engine::ENGINE_THREAD) {
                previous(info);
                return;
            }
            SURRENDERED.store(true, Ordering::Relaxed);
            restore();
            previous(info);
            // The dashboard is gone but the process may well still be running,
            // so say where its output went rather than leaving a bare panic
            // message and a shell that has stopped updating.
            eprintln!(
                "xwen serve: the dashboard surrendered the terminal after a panic; \
                 logging continues on stderr"
            );
        }));
    });
}

/// Forward terminal events into the sink's own channel, so the loop has exactly
/// one place to wait.
fn spawn_input(tx: Sender<SinkMessage>, stop: Arc<AtomicBool>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            match event::poll(INPUT_POLL) {
                Ok(true) => match event::read() {
                    Ok(event) => {
                        if tx.send(SinkMessage::Term(event)).is_err() {
                            return;
                        }
                    }
                    Err(_) => return,
                },
                Ok(false) => {}
                Err(_) => return,
            }
        }
    })
}

/// Sample the machine facts the header shows, off the render loop: both are
/// subprocesses, and neither belongs anywhere near a draw.
fn spawn_environment(environment: Arc<Environment>, stop: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        environment.sample();
        loop {
            // Waited out in short slices so the thread leaves with the
            // dashboard rather than a minute after it.
            let until = Instant::now() + POWER_INTERVAL;
            while Instant::now() < until {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            environment.sample();
        }
    });
}

/// The dashboard sink: drain the same channel the stderr sink does, drawing
/// instead of printing.
///
/// Two things end it. A `Stop` (or a hung-up channel) is the server on its way
/// out; `q` or Ctrl-C is the operator, which asks for the same graceful shutdown
/// a signal does and then hands the terminal back at once, so the shutdown lines
/// that follow are read as ordinary stderr — which is where an operator watching
/// a server come down expects them.
pub(super) fn run_sink(
    rx: Receiver<SinkMessage>,
    input: Sender<SinkMessage>,
    vitals: Vitals,
    quit: QuitSignal,
) {
    let environment = Arc::new(Environment::default());
    let mut app = Dashboard::new(vitals, Instant::now(), Arc::clone(&environment));

    let mut screen = match Screen::open() {
        Ok(screen) => Some(screen),
        Err(e) => {
            // Not a terminal, or a terminal that will not go raw. Falling back
            // beats refusing to start a server over how its log looks.
            eprintln!(
                "xwen serve: the dashboard cannot take this terminal ({e}); logging to stderr instead"
            );
            None
        }
    };
    let stop_helpers = Arc::new(AtomicBool::new(false));
    let input_thread = screen
        .is_some()
        .then(|| spawn_input(input, Arc::clone(&stop_helpers)));
    if screen.is_some() {
        spawn_environment(environment, Arc::clone(&stop_helpers));
    }

    let mut last_draw: Option<Instant> = None;
    loop {
        let message = match rx.recv_timeout(POLL) {
            Ok(message) => Some(message),
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        let mut done = false;
        // One receive, then everything already queued behind it: a burst of
        // decode ticks becomes one frame rather than one frame each.
        //
        // The whole batch is stamped with one instant, and that is what makes
        // the throughput averages mean anything: the ticks a drain pulls out
        // together were produced over the interval since the last drain, not in
        // the microseconds it takes to walk them. Timing each one where it was
        // dequeued would read two tokens a microsecond apart as a million a
        // second.
        let at = Instant::now();
        let wall = SystemTime::now();
        for message in message.into_iter().chain(rx.try_iter()) {
            match handle(message, &mut app, &mut screen, &quit, at, wall) {
                Step::Continue => {}
                Step::Stop => {
                    done = true;
                    break;
                }
            }
        }
        if done {
            break;
        }
        if SURRENDERED.load(Ordering::Relaxed) {
            screen = None;
        }
        // However the screen went — quit, shutdown, or a panic elsewhere in the
        // process — the helpers go with it: a reader left on a restored terminal
        // swallows what the operator types at their shell.
        if screen.is_none() {
            stop_helpers.store(true, Ordering::Relaxed);
        }
        let now = Instant::now();
        // Read again here rather than only above: a panic on another thread
        // between the two has already handed the terminal back and printed into
        // it, and a frame drawn afterwards would paint over the message the
        // operator needs.
        if SURRENDERED.load(Ordering::Relaxed) {
            screen = None;
            stop_helpers.store(true, Ordering::Relaxed);
        }
        if let Some(screen) = screen.as_mut()
            && last_draw.is_none_or(|at| now.saturating_duration_since(at) >= REDRAW_INTERVAL)
        {
            last_draw = Some(now);
            let _ = screen.terminal.draw(|frame| draw(frame, &mut app, now));
        }
    }

    drop(screen);
    stop_helpers.store(true, Ordering::Relaxed);
    if let Some(thread) = input_thread {
        // Joined rather than abandoned: a reader left on the terminal would eat
        // what the operator types at their shell while the server shuts down.
        let _ = thread.join();
    }
}

enum Step {
    Continue,
    Stop,
}

fn handle(
    message: SinkMessage,
    app: &mut Dashboard,
    screen: &mut Option<Screen>,
    quit: &QuitSignal,
    at: Instant,
    wall: SystemTime,
) -> Step {
    match message {
        SinkMessage::Event(event) => {
            // The shutdown lines are the last thing an operator reads, and by
            // the time they arrive the dashboard is about to be torn down
            // anyway — hand the terminal back first so they land where the eye
            // already is.
            if matches!(event, ServeLog::ShutdownRequested | ServeLog::ShuttingDown) {
                *screen = None;
            }
            match screen {
                Some(_) => app.apply(event, at, wall),
                None => {
                    if let Some(line) = event.render() {
                        eprintln!("{line}");
                    }
                }
            }
            Step::Continue
        }
        SinkMessage::Term(Event::Key(key)) => {
            // Key events are delivered on press and release; acting on both
            // would run every binding twice.
            if key.kind == KeyEventKind::Press && app.key(key) {
                quit.request();
                *screen = None;
            }
            Step::Continue
        }
        SinkMessage::Term(_) => Step::Continue,
        SinkMessage::Flush(ack) => {
            let _ = ack.send(());
            Step::Continue
        }
        SinkMessage::Stop => Step::Stop,
    }
}

/// Everything the dashboard drew, as text, for the tests.
#[cfg(test)]
fn rendered(terminal: &ratatui::Terminal<ratatui::backend::TestBackend>) -> String {
    let buffer = terminal.backend().buffer();
    let mut out = String::new();
    for y in 0..buffer.area.height {
        for x in 0..buffer.area.width {
            out.push_str(buffer[(x, y)].symbol());
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::SpecStats;
    use crate::serve::log::{
        BatchSummary, DiskEvictReason, DiskSegmentRole, JobPhase, QueueEntry, SlotSummary,
    };
    use crate::serve::types::{CancelReason, Dialect};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use std::path::PathBuf;

    fn vitals() -> Vitals {
        Vitals {
            model_id: "laguna-s-2.1-Q4_K_M".to_string(),
            model_bytes: Some(68_200_000_000),
            context_length: 262_144,
            port: 5241,
            draft: true,
            draft_configured: true,
            disk_cache: true,
        }
    }

    fn dashboard() -> Dashboard {
        Dashboard::new(vitals(), Instant::now(), Arc::new(Environment::default()))
    }

    fn origin(id: u64) -> RequestOrigin {
        RequestOrigin {
            id,
            dialect: Dialect::Anthropic,
            streaming: true,
        }
    }

    fn record() -> JobRecord {
        JobRecord {
            origin: origin(7),
            stop: Some(StopKind::EndTurn),
            abandoned: None,
            error: None,
            prompt_tokens: 2048,
            cache_read: 380,
            prefill_tokens: 1668,
            prefill_secs: 5.2,
            output_tokens: 38,
            thinking_tokens: 0,
            decode_secs: 3.1,
            ttft_secs: Some(6.6),
            spec: None,
            batch: None,
        }
    }

    /// Feed a sequence in and draw it, the way the sink loop would.
    fn frame(app: &mut Dashboard, events: Vec<ServeLog>) -> String {
        let now = Instant::now();
        for event in events {
            app.apply(event, now, SystemTime::UNIX_EPOCH);
        }
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).expect("a test terminal");
        terminal
            .draw(|f| draw(f, app, now))
            .expect("the dashboard draws");
        rendered(&terminal)
    }

    /// A request being prefilled is the case the dashboard exists for: the
    /// server looks idle for minutes, and the bar is the only thing saying
    /// otherwise.
    #[test]
    fn a_prefilling_request_draws_its_progress_the_cache_and_the_client() {
        let mut app = dashboard();
        let text = frame(
            &mut app,
            vec![
                ServeLog::JobPicked {
                    origin: origin(1),
                    prompt_tokens: 34_112,
                    estimated: false,
                    queue_wait: Duration::from_millis(400),
                    deadline: None,
                },
                ServeLog::JobCacheResolved {
                    id: 1,
                    cached: 380,
                    resume: 380,
                    slot: 0,
                },
                ServeLog::PrefillTick {
                    done: 21_480,
                    total: 34_112,
                },
            ],
        );
        assert!(text.contains("PREFILL 21,480 / 34,112 tok"), "{text}");
        assert!(text.contains("cache_read 380"), "{text}");
        assert!(text.contains("slot 0"), "{text}");
        assert!(text.contains("client anthropic, streaming"), "{text}");
        assert!(text.contains("laguna-s-2.1-Q4_K_M"), "{text}");
        assert!(text.contains("ctx 262,144"), "{text}");
        assert!(text.contains("draft ON"), "{text}");
    }

    /// The header's disk cell is folded from the scan and every write and deletion
    /// since — each of those events being exactly one file — so it tracks the store
    /// without anyone going and looking at the directory. A server keeping nothing on
    /// disk shows no cell at all rather than a size of zero.
    #[test]
    fn the_header_counts_the_disk_store_from_the_events() {
        let mut app = dashboard();
        let text = frame(
            &mut app,
            vec![ServeLog::DiskCacheScanned {
                segments: 2,
                bytes: 2 << 30,
                rejected: 0,
            }],
        );
        assert!(text.contains("disk: 2 segs 2.0 GiB"), "{text}");

        let text = frame(
            &mut app,
            vec![
                ServeLog::DiskSegmentWritten {
                    role: DiskSegmentRole::Tail,
                    tokens: 20_014,
                    bytes: 1 << 30,
                    ms: 820,
                },
                ServeLog::DiskSegmentEvicted {
                    reason: DiskEvictReason::Superseded,
                    bytes: 2 << 30,
                },
            ],
        );
        assert!(text.contains("disk: 2 segs 1.0 GiB"), "{text}");

        // A hydration reads a chain without changing what is stored.
        let text = frame(
            &mut app,
            vec![ServeLog::DiskChainHydrated {
                segments: 2,
                tokens: 20_014,
                resume: 19_012,
                bytes: 1 << 30,
                ms: 640,
            }],
        );
        assert!(text.contains("disk: 2 segs 1.0 GiB"), "{text}");

        let mut off = Dashboard::new(
            Vitals {
                disk_cache: false,
                ..vitals()
            },
            Instant::now(),
            Arc::new(Environment::default()),
        );
        let text = frame(&mut off, Vec::new());
        assert!(!text.contains("disk:"), "{text}");
    }

    /// Once tokens are coming out the pane switches to what the client is
    /// actually receiving, and says which side of the reasoning boundary it is.
    #[test]
    fn a_decoding_request_reports_its_tokens_and_whether_it_is_thinking() {
        let mut app = dashboard();
        let text = frame(
            &mut app,
            vec![
                ServeLog::JobPicked {
                    origin: origin(1),
                    prompt_tokens: 2048,
                    estimated: false,
                    queue_wait: Duration::ZERO,
                    deadline: None,
                },
                ServeLog::PrefillTick {
                    done: 2048,
                    total: 2048,
                },
                ServeLog::DecodeTick {
                    tokens_out: 142,
                    thinking: true,
                },
            ],
        );
        assert!(text.contains("DECODE 142 tok"), "{text}");
        assert!(text.contains("thinking"), "{text}");
    }

    /// A finished request leaves a row saying what it cost and how it ended.
    #[test]
    fn a_finished_request_lands_in_history_with_its_tokens_and_stop_reason() {
        let mut app = dashboard();
        let text = frame(
            &mut app,
            vec![
                ServeLog::JobPicked {
                    origin: origin(1),
                    prompt_tokens: 2048,
                    estimated: false,
                    queue_wait: Duration::ZERO,
                    deadline: None,
                },
                ServeLog::JobDone(Box::new(record())),
            ],
        );
        assert!(text.contains("380+1,668→38"), "{text}");
        assert!(text.contains("end_turn"), "{text}");
        assert!(text.contains("TTFT 6.6s"), "{text}");
        assert!(text.contains("anthropic"), "{text}");
        // The pane it came from is empty again.
        assert!(text.contains("idle — waiting for a request"), "{text}");
    }

    /// The queue and the slots come from snapshots, so the panes show whatever
    /// the last one said and never ask the engine anything.
    #[test]
    fn the_queue_and_slot_panes_show_the_latest_snapshots() {
        let mut app = dashboard();
        let text = frame(
            &mut app,
            vec![
                ServeLog::QueueSnapshot(vec![QueueEntry {
                    id: 12,
                    prompt_tokens: 1204,
                    queued_at: Instant::now(),
                }]),
                ServeLog::SlotsSnapshot(vec![SlotSummary {
                    live: true,
                    tokens: 21_480,
                    snapshots: 2,
                    image_bytes: 113_246_208,
                    has_drafter: true,
                    last_used: 4,
                    agrees_to: 21_480,
                }]),
            ],
        );
        assert!(text.contains("QUEUE (1)"), "{text}");
        assert!(text.contains("#12"), "{text}");
        assert!(text.contains("1,204 tok"), "{text}");
        assert!(text.contains("LIVE"), "{text}");
        assert!(text.contains("21,480 tok"), "{text}");
        assert!(text.contains("drafter"), "{text}");
    }

    /// The header's drafting cell reports the LOADED checkpoint, through the
    /// event order the engine actually produces: `DrafterLoaded` comes from
    /// inside `EngineState::load` and `ModelLoaded` only after that load
    /// returns, so anything that keyed "not drafting yet" off `ModelLoaded`
    /// would run last and pin the cell off on every drafting server. Fed here in
    /// the real order, across a full swap cycle, so that ordering cannot regress
    /// unnoticed.
    #[test]
    fn the_draft_cell_follows_the_loaded_checkpoint_in_the_engines_own_order() {
        let mut app = dashboard();
        let loaded = |ms: u64| ServeLog::ModelLoaded {
            elapsed: Duration::from_millis(ms),
        };
        let drafter = || ServeLog::DrafterLoaded {
            elapsed: Duration::from_millis(600),
            draft_ctx: 8192,
        };
        let no_drafter = || ServeLog::NoDrafterAvailable {
            model: crate::hub::Model::Qwen3827B,
        };
        let swap = || ServeLog::CheckpointSwappingOut {
            from: crate::serve::types::Target::official(crate::hub::Model::Qwen35BA3B),
            to: crate::serve::types::Target::official(crate::hub::Model::Qwen3827B),
        };

        // A drafting checkpoint loads: drafter first, model second.
        let text = frame(&mut app, vec![drafter(), loaded(4000)]);
        assert!(app.vitals.draft, "a drafting checkpoint reads ON");
        assert!(text.contains("draft ON"), "{text}");

        // Swapped out for one that ships no sidecar.
        let text = frame(&mut app, vec![swap(), no_drafter(), loaded(3800)]);
        assert!(!app.vitals.draft, "a sidecar-less checkpoint reads off");
        assert!(text.contains("draft off"), "{text}");

        // And back: the cell recovers rather than staying off for the process.
        let text = frame(&mut app, vec![swap(), drafter(), loaded(3900)]);
        assert!(app.vitals.draft, "drafting again after the swap back");
        assert!(text.contains("draft ON"), "{text}");

        // An idle unload leaves nothing loaded, so the cell falls back to what
        // the server was configured for rather than to a stale checkpoint's answer.
        frame(
            &mut app,
            vec![
                no_drafter(),
                ServeLog::IdleUnloaded {
                    elapsed: Duration::from_secs(600),
                    configured: Some(Duration::from_secs(600)),
                },
            ],
        );
        assert_eq!(app.vitals.draft, app.vitals.draft_configured);

        // A server configured not to speculate never reads ON, whatever loads.
        let mut plain = Dashboard::new(
            Vitals {
                draft: false,
                draft_configured: false,
                ..vitals()
            },
            Instant::now(),
            Arc::new(Environment::default()),
        );
        let text = frame(&mut plain, vec![loaded(4000), swap(), loaded(4000)]);
        assert!(!plain.vitals.draft, "{text}");
    }

    /// Every event that has a line to say still says it, in the LOG pane instead
    /// of on stderr. The wording is the stderr sink's, not a second copy of it.
    #[test]
    fn every_event_that_renders_a_line_lands_in_the_log_ring() {
        let mut app = dashboard();
        let speaking = vec![
            ServeLog::ContextClamped {
                requested: 1_000_000,
                trained: 262_144,
            },
            ServeLog::Listening {
                address: "127.0.0.1:5241".to_string(),
                anthropic: true,
                openai: true,
            },
            ServeLog::ServingModel {
                path: PathBuf::from("/models/laguna.gguf"),
            },
            // The engine's own order: the drafter attaches inside the model
            // load, so its line comes first.
            ServeLog::DrafterLoaded {
                elapsed: Duration::from_millis(1600),
                draft_ctx: 32768,
            },
            ServeLog::ModelLoaded {
                elapsed: Duration::from_millis(4200),
            },
            ServeLog::IdleUnloaded {
                elapsed: Duration::from_secs(600),
                configured: Some(Duration::from_secs(600)),
            },
            ServeLog::CacheClearFailed {
                error: "the device is gone".to_string(),
            },
            ServeLog::ClientStalled {
                after: Duration::from_secs(120),
            },
            ServeLog::ThinkBudgetClamped {
                requested: 4096,
                max_new: 512,
                using: Some(384),
            },
            ServeLog::CacheLengthMismatch {
                cached: 12,
                expected: 380,
            },
            ServeLog::SpecReport {
                stats: SpecStats {
                    rounds: 10,
                    drafted: 40,
                    accepted: 30,
                    ..SpecStats::default()
                },
            },
            ServeLog::ToolSpanReport {
                healed: 1,
                quoted: 0,
                degraded: 0,
            },
            ServeLog::AbandonedDuringDecode {
                reason: CancelReason::ClientGone,
                tokens_out: 142,
                cache_kept: 2048,
            },
            ServeLog::DeadlineDuringDecode {
                ceiling: Duration::from_secs(214),
                tokens_out: 512,
            },
            ServeLog::AbandonedBeforeDecode {
                reason: CancelReason::Shutdown,
                phase: JobPhase::Prefill,
                done: 2048,
                total: 3896,
                kept: 2048,
            },
            ServeLog::DeadlineBeforeDecode {
                ceiling: Duration::from_secs(60),
                done: 512,
                total: 4096,
            },
            ServeLog::SlotPagedOut {
                slot: 0,
                pos: 380,
                bytes: 113_246_208,
                elapsed: Duration::from_millis(41),
            },
            ServeLog::SlotPagedIn {
                slot: 2,
                restore: 380,
                bytes: 113_246_208,
                elapsed: Duration::from_millis(41),
            },
            ServeLog::DiskCacheScanned {
                segments: 3,
                bytes: 4_402_341_478,
                rejected: 1,
            },
            ServeLog::DiskSegmentWritten {
                role: DiskSegmentRole::Tail,
                tokens: 20_014,
                bytes: 1_287_651_328,
                ms: 820,
            },
            ServeLog::DiskSegmentSplit { at: 19_012 },
            ServeLog::DiskChainHydrated {
                segments: 3,
                tokens: 20_014,
                resume: 19_012,
                bytes: 1_287_651_328,
                ms: 640,
            },
            ServeLog::DiskSegmentEvicted {
                reason: DiskEvictReason::Superseded,
                bytes: 1_287_651_328,
            },
            ServeLog::DiskCacheFailed {
                action: "writing a segment",
                error: "No space left on device".to_string(),
            },
            ServeLog::QueuedRequestDropped {
                waited: Duration::from_secs(301),
                queue_timeout: Duration::from_secs(300),
            },
            ServeLog::OutOfOrderPick {
                prompt_tokens: 1204,
                cost: 314,
                older: 2,
                oldest_waited: Duration::from_secs(4),
            },
            ServeLog::InvalidHeaderDropped {
                name: "retry-after",
                value: "1".to_string(),
            },
            ServeLog::ShutdownRequested,
            ServeLog::ShuttingDown,
            ServeLog::ShutdownGraceExpired {
                grace: Duration::from_secs(30),
            },
        ];
        let expected: Vec<String> = speaking
            .iter()
            .map(|event| event.render().expect("this event has a line"))
            .collect();
        for event in speaking {
            app.apply(event, Instant::now(), SystemTime::UNIX_EPOCH);
        }
        assert_eq!(app.logs.len(), expected.len());
        for (line, want) in app.logs.iter().zip(&expected) {
            assert!(line.ends_with(want), "{line} should end with {want}");
        }
    }

    /// The instrumentation events change the panes without ever adding a line —
    /// a tick per decoded token would push the log ring out in seconds.
    #[test]
    fn the_instrumentation_events_add_no_log_lines() {
        let mut app = dashboard();
        for event in [
            ServeLog::JobPicked {
                origin: origin(1),
                prompt_tokens: 8,
                estimated: false,
                queue_wait: Duration::ZERO,
                deadline: None,
            },
            ServeLog::PrefillTick { done: 8, total: 8 },
            ServeLog::DecodeTick {
                tokens_out: 1,
                thinking: false,
            },
            ServeLog::JobDone(Box::new(record())),
            ServeLog::SlotsSnapshot(Vec::new()),
            ServeLog::QueueSnapshot(Vec::new()),
        ] {
            app.apply(event, Instant::now(), SystemTime::UNIX_EPOCH);
        }
        assert!(app.logs.is_empty());
    }

    /// The rings are bounded: a server that runs for a week holds the same
    /// memory as one that just started.
    #[test]
    fn the_history_and_log_rings_drop_their_oldest() {
        let mut app = dashboard();
        for _ in 0..(HISTORY_CAPACITY + 5) {
            app.apply(
                ServeLog::JobDone(Box::new(record())),
                Instant::now(),
                SystemTime::UNIX_EPOCH,
            );
        }
        assert_eq!(app.history.len(), HISTORY_CAPACITY);
        for _ in 0..(LOG_CAPACITY + 5) {
            app.apply(
                ServeLog::ShuttingDown,
                Instant::now(),
                SystemTime::UNIX_EPOCH,
            );
        }
        assert_eq!(app.logs.len(), LOG_CAPACITY);
    }

    /// Scrolling means the operator is reading, so the pane stops jumping to the
    /// newest line — until they scroll back down onto it (or press End), which
    /// is arriving back at the live edge and re-engages follow.
    #[test]
    fn scrolling_the_log_disengages_follow_and_the_live_edge_restores_it() {
        let mut app = dashboard();
        for _ in 0..20 {
            app.apply(
                ServeLog::ShuttingDown,
                Instant::now(),
                SystemTime::UNIX_EPOCH,
            );
        }
        assert!(app.follow);
        assert!(!app.key(key(KeyCode::Up)));
        assert!(!app.follow);
        assert_eq!(app.log_state.selected(), Some(18));
        assert!(!app.key(key(KeyCode::PageUp)));
        assert_eq!(app.log_state.selected(), Some(8));
        // Scrolling past either end stops there rather than wrapping.
        assert!(!app.key(key(KeyCode::PageUp)));
        assert_eq!(app.log_state.selected(), Some(0));
        // Scrolling down but short of the newest line is still reading.
        assert!(!app.key(key(KeyCode::PageDown)));
        assert_eq!(app.log_state.selected(), Some(10));
        assert!(!app.follow);
        // Landing on the newest line is the live edge: follow re-engages, and
        // the next event moves the selection with it.
        assert!(!app.key(key(KeyCode::PageDown)));
        assert_eq!(app.log_state.selected(), Some(19));
        assert!(app.follow);
        // End still works as the long-jump spelling of the same thing.
        assert!(!app.key(key(KeyCode::Up)));
        assert!(!app.follow);
        assert!(!app.key(key(KeyCode::End)));
        assert!(app.follow);
    }

    /// `q` and Ctrl-C are the same request. Ctrl-C has to be recognised as a
    /// keypress because raw mode generates no signal for it.
    #[test]
    fn q_and_ctrl_c_both_ask_to_quit_and_nothing_else_does() {
        let mut app = dashboard();
        assert!(app.key(key(KeyCode::Char('q'))));
        assert!(app.key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)));
        assert!(!app.key(key(KeyCode::Char('c'))));
        assert!(!app.key(key(KeyCode::Char('x'))));
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    /// The gauge panics outside 0..=1, and `done` can exceed `total` for one
    /// tick when a span's last chunk overshoots the estimate.
    #[test]
    fn the_prefill_gauge_ratio_stays_inside_its_range() {
        let now = Instant::now();
        let mut over = Progress {
            done: 40,
            total: 32,
            rate: Rate::default(),
        };
        over.rate.observe(now, 40);
        let _ = prefill_gauge(&over, now);
        let empty = Progress::default();
        let _ = prefill_gauge(&empty, now);
    }

    /// An abandoned request is named by what actually cut it short, marked so it
    /// is not read as an ordinary ending — the client may well have been told
    /// `max_tokens`.
    #[test]
    fn history_names_the_real_reason_a_request_was_cut_short() {
        assert_eq!(outcome(&record()), "end_turn");
        assert_eq!(
            outcome(&JobRecord {
                stop: Some(StopKind::MaxTokens),
                abandoned: Some(CancelReason::Deadline),
                ..record()
            }),
            "deadline*"
        );
        assert_eq!(
            outcome(&JobRecord {
                stop: None,
                abandoned: Some(CancelReason::ClientGone),
                ..record()
            }),
            "client-gone*"
        );
        assert_eq!(
            outcome(&JobRecord {
                stop: None,
                error: Some("the inference engine panicked".to_string()),
                ..record()
            }),
            "error"
        );
    }

    /// The phase columns each carry their own time and their own rate — a
    /// generation's prefill and decode are different machines and blending
    /// them into one figure described neither. Speculation reports its
    /// acceptance in the detail column, beside TTFT.
    #[test]
    fn phase_columns_report_their_own_times_and_rates() {
        assert_eq!(prefill_cell(&record()), "5.2s · 321 t/s");
        assert_eq!(decode_cell(&record()), "3.1s · 12 t/s");
        assert_eq!(detail_cell(&record()), "TTFT 6.6s");
        assert_eq!(
            detail_cell(&JobRecord {
                spec: Some(SpecStats {
                    rounds: 10,
                    drafted: 40,
                    accepted: 22,
                    ..SpecStats::default()
                }),
                ..record()
            }),
            "TTFT 6.6s · spec 55%"
        );
    }

    /// A batch row reads its phases from the batch summary: prefill is engine
    /// work (shared prefix, tails, scored trials), decode only what was
    /// actually sampled — a fully scored batch decodes nothing and says so
    /// with a dash — and the whole wall run lands in the detail column.
    #[test]
    fn a_batch_row_reports_measured_phases() {
        let batch = JobRecord {
            stop: None,
            ttft_secs: None,
            batch: Some(BatchSummary {
                items: 27,
                failed: 0,
                secs: 217.0,
                prefill_tokens: 45_000,
                prefill_secs: 180.0,
                decode_tokens: 0,
                decode_secs: 0.0,
            }),
            ..record()
        };
        assert_eq!(outcome(&batch), "27 items");
        assert_eq!(prefill_cell(&batch), "3m00s · 250 t/s");
        assert_eq!(decode_cell(&batch), "—");
        assert_eq!(detail_cell(&batch), "total 3m37s");
        let failed = JobRecord {
            batch: Some(BatchSummary {
                failed: 2,
                ..batch.batch.unwrap()
            }),
            ..batch.clone()
        };
        assert_eq!(outcome(&failed), "25/27 items");
    }

    #[test]
    fn the_number_formats_are_readable_at_a_glance() {
        assert_eq!(commas(0), "0");
        assert_eq!(commas(999), "999");
        assert_eq!(commas(34_112), "34,112");
        assert_eq!(commas(1_000_000), "1,000,000");
        assert_eq!(giga(68_200_000_000), "68.2G");
        assert_eq!(mib(113_246_208), "108 MiB");
        assert_eq!(uptime(Duration::from_secs(12)), "12s");
        assert_eq!(uptime(Duration::from_secs(14 * 60)), "14m");
        assert_eq!(uptime(Duration::from_secs(2 * 3600 + 14 * 60)), "2h14m");
        assert_eq!(seconds(Duration::from_millis(3140)), "3.1s");
        assert_eq!(seconds(Duration::from_secs(125)), "2m05s");
        assert_eq!(rate(None), "— tok/s");
        assert_eq!(rate(Some(19.84)), "19.8 tok/s");
    }

    /// The clock is UTC plus a fixed offset, so it reads as the operator's own
    /// wall clock and never jumps backwards across midnight.
    #[test]
    fn the_clock_is_local_time_of_day() {
        let evening = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        assert_eq!(clock(evening, 0), "22:13:20");
        // An offset that carries the time into the next or the previous day
        // wraps rather than running past 24 or below zero.
        assert_eq!(clock(evening, 2 * 3600), "00:13:20");
        assert_eq!(clock(evening, -15 * 3600), "07:13:20");
    }

    /// Throughput is averaged over the ticks, and the first one — which has
    /// nothing to be measured against — reports nothing rather than a number
    /// made up from the moment the job was picked.
    #[test]
    fn the_throughput_average_needs_two_ticks() {
        let start = Instant::now();
        let mut rate = Rate::default();
        rate.observe(start, 10);
        assert_eq!(rate.value, None);
        rate.observe(start + Duration::from_secs(1), 30);
        assert_eq!(rate.value, Some(20.0));
        rate.observe(start + Duration::from_secs(2), 40);
        // The newest sample moves the average without replacing it.
        let value = rate.value.expect("two intervals have been seen");
        assert!(value > 10.0 && value < 20.0, "{value}");
    }

    /// Ticks drained together were produced over the interval since the last
    /// drain, and the sink stamps a whole batch with one instant so they read
    /// that way. Timing each tick where it was dequeued would turn a batch of
    /// two into a throughput of hundreds of thousands a second.
    #[test]
    fn a_coalesced_batch_of_ticks_reports_the_rate_it_was_produced_at() {
        let mut app = dashboard();
        let start = Instant::now();
        app.apply(
            ServeLog::JobPicked {
                origin: origin(1),
                prompt_tokens: 8,
                estimated: false,
                queue_wait: Duration::ZERO,
                deadline: None,
            },
            start,
            SystemTime::UNIX_EPOCH,
        );
        for (batch, at) in [start, start + Duration::from_millis(100)]
            .into_iter()
            .enumerate()
        {
            for tick in 1..=2 {
                app.apply(
                    ServeLog::DecodeTick {
                        tokens_out: batch * 2 + tick,
                        thinking: false,
                    },
                    at,
                    SystemTime::UNIX_EPOCH,
                );
            }
        }
        let live = app.live.as_ref().expect("the job is still running");
        let measured = live
            .decode
            .expect("two batches have arrived")
            .rate
            .value
            .expect("the second batch has something to measure against");
        // Two tokens over a hundred milliseconds.
        assert!((measured - 20.0).abs() < 1e-9, "{measured}");
    }

    /// The queue pane ages its entries as it draws them. Snapshots only arrive
    /// when the queue changes, so an entry waiting out a long prefill would
    /// otherwise be shown at the age it had when it arrived, for all of it.
    #[test]
    fn a_waiting_request_ages_between_snapshots() {
        let mut app = dashboard();
        let queued_at = Instant::now();
        app.apply(
            ServeLog::QueueSnapshot(vec![QueueEntry {
                id: 4,
                prompt_tokens: 33_900,
                queued_at,
            }]),
            queued_at,
            SystemTime::UNIX_EPOCH,
        );
        let mut at = |now: Instant| {
            let mut terminal = Terminal::new(TestBackend::new(120, 40)).expect("a test terminal");
            terminal
                .draw(|f| draw(f, &mut app, now))
                .expect("the dashboard draws");
            rendered(&terminal)
        };
        assert!(at(queued_at).contains("age 0.0s"));
        assert!(at(queued_at + Duration::from_secs(41)).contains("age 41.0s"));
    }

    /// The NOW panel reports what was reused, which is what the wire calls
    /// `cache_read`. A prefix that matched and was then discarded — the KV cache
    /// disagreed and the prompt is being replayed — is named alongside it,
    /// because that is the case worth noticing.
    #[test]
    fn the_now_panel_reports_reused_tokens_and_names_a_discarded_match() {
        let mut app = dashboard();
        let picked = ServeLog::JobPicked {
            origin: origin(1),
            prompt_tokens: 2048,
            estimated: false,
            queue_wait: Duration::ZERO,
            deadline: None,
        };
        let text = frame(
            &mut app,
            vec![
                picked.clone(),
                ServeLog::JobCacheResolved {
                    id: 1,
                    cached: 380,
                    resume: 380,
                    slot: 0,
                },
            ],
        );
        assert!(text.contains("cache_read 380"), "{text}");
        assert!(!text.contains("discarded"), "{text}");

        let mut app = dashboard();
        let text = frame(
            &mut app,
            vec![
                picked,
                ServeLog::JobCacheResolved {
                    id: 1,
                    cached: 380,
                    resume: 0,
                    slot: 0,
                },
            ],
        );
        assert!(
            text.contains("cache_read 0 (match 380 discarded)"),
            "{text}"
        );
    }

    /// Ticks that arrive a moment apart are not a throughput: the interval is
    /// the channel's own jitter, and dividing by it would report hundreds of
    /// tokens a second at a batch boundary.
    #[test]
    fn ticks_closer_together_than_the_floor_do_not_move_the_average() {
        let start = Instant::now();
        let mut rate = Rate::default();
        rate.observe(start, 0);
        rate.observe(start + Duration::from_millis(1), 2);
        assert_eq!(rate.value, None, "one millisecond is not a measurement");
        // The counts are cumulative, so nothing was lost: the next interval
        // long enough to divide by picks up everything since the baseline.
        rate.observe(start + Duration::from_secs(1), 20);
        assert_eq!(rate.value, Some(20.0));
    }

    /// A rate that stopped moving says so. The average is lagging by
    /// construction, so a wedged job would otherwise go on reporting the
    /// throughput it had while it was still running — which is the one question
    /// the dashboard exists to answer.
    #[test]
    fn a_stalled_phase_reports_the_silence_instead_of_its_last_average() {
        let start = Instant::now();
        let mut rate = Rate::default();
        rate.observe(start, 0);
        rate.observe(start + Duration::from_secs(1), 20);
        assert_eq!(rate.stalled_for(start + Duration::from_secs(2)), None);
        assert_eq!(
            throughput(&rate, start + Duration::from_secs(2)),
            "20.0 tok/s"
        );

        let stalled = start + Duration::from_secs(13);
        assert_eq!(
            rate.stalled_for(stalled),
            Some(Duration::from_secs(12)),
            "measured from the last tick, not the last frame"
        );
        assert_eq!(throughput(&rate, stalled), "stalled 12.0s");

        // A phase that has produced nothing at all has no silence to report:
        // there is no tick to measure one from.
        assert_eq!(Rate::default().stalled_for(stalled), None);
    }

    /// The prefill bar drops its estimate while nothing is arriving. An ETA
    /// extrapolated from a rate the work stopped producing is the one number
    /// that would actively mislead.
    #[test]
    fn a_stalled_prefill_shows_no_eta() {
        let start = Instant::now();
        let mut progress = Progress {
            done: 1000,
            total: 34_112,
            rate: Rate::default(),
        };
        progress.rate.observe(start, 0);
        progress.rate.observe(start + Duration::from_secs(1), 1000);
        let label = |now: Instant| {
            let mut terminal = Terminal::new(TestBackend::new(120, 1)).expect("a test terminal");
            terminal
                .draw(|f| f.render_widget(prefill_gauge(&progress, now), f.area()))
                .expect("the gauge draws");
            rendered(&terminal)
        };
        let running = label(start + Duration::from_secs(1));
        assert!(running.contains("ETA"), "{running}");
        let stalled = label(start + Duration::from_secs(20));
        assert!(!stalled.contains("ETA"), "{stalled}");
        assert!(stalled.contains("stalled"), "{stalled}");
    }

    /// `Home` on an empty log leaves the pane following, so the first line to
    /// arrive is not stranded off-screen behind a selection nobody made.
    #[test]
    fn home_on_an_empty_log_keeps_following() {
        let mut app = dashboard();
        assert!(!app.key(key(KeyCode::Home)));
        assert!(app.follow);
        assert_eq!(app.log_state.selected(), None);
    }

    /// A scrolled selection holds its line as the ring slides under it. Under
    /// load the log evicts several lines a second, and a view that walked
    /// forward with each of them could not be read at all.
    #[test]
    fn a_scrolled_selection_survives_the_log_ring_evicting_its_oldest() {
        let mut app = dashboard();
        for _ in 0..LOG_CAPACITY {
            app.apply(
                ServeLog::ShuttingDown,
                Instant::now(),
                SystemTime::UNIX_EPOCH,
            );
        }
        app.key(key(KeyCode::Up));
        let held = app.log_state.selected().expect("a line is selected");
        for _ in 0..10 {
            app.apply(
                ServeLog::ShuttingDown,
                Instant::now(),
                SystemTime::UNIX_EPOCH,
            );
        }
        assert_eq!(app.log_state.selected(), Some(held - 10));
        // Following instead, the pane goes on showing the newest line.
        app.key(key(KeyCode::End));
        let text = frame(&mut app, vec![ServeLog::ShuttingDown]);
        assert!(text.contains("xwen serve shutting down"), "{text}");
    }

    /// A narrow terminal stacks the two middle panes instead of squeezing both
    /// into forty columns, and neither width panics.
    #[test]
    fn the_layout_survives_a_narrow_terminal() {
        let mut app = dashboard();
        app.apply(
            ServeLog::QueueSnapshot(vec![QueueEntry {
                id: 3,
                prompt_tokens: 12,
                queued_at: Instant::now(),
            }]),
            Instant::now(),
            SystemTime::UNIX_EPOCH,
        );
        for (width, height) in [(60, 20), (200, 60), (20, 8)] {
            let mut terminal =
                Terminal::new(TestBackend::new(width, height)).expect("a test terminal");
            terminal
                .draw(|f| draw(f, &mut app, Instant::now()))
                .expect("the dashboard draws at any size");
        }
    }
}
