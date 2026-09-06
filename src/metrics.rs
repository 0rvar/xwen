//! Per-run metrics: an append-only history of what every xwen run cost, and
//! the aggregation `xwen stats` reports over it.
//!
//! One JSON object per line, appended once at the end of a run. A whole line is
//! written in a single call to an `O_APPEND` handle, so two xwen processes
//! recording at the same moment interleave records and never fragments of one.
//!
//! This is durable history rather than a cache — it lives under the XDG state
//! directory, nothing in xwen prunes or rewrites it, and a failure to write it
//! never fails the run that produced it.

use std::ffi::OsStr;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use unicode_width::UnicodeWidthStr;

/// Where the history lives under `$HOME` when nothing names a path. The XDG
/// state directory, not the cache one: a lost cache costs a re-download, a lost
/// history is gone.
pub const METRICS_RELATIVE_PATH: &str = ".local/state/xwen/metrics.jsonl";

/// Names the file to record into, or `off` (in any casing) to record nothing.
/// An empty value counts as unset and resolves the default path.
pub const METRICS_ENV: &str = "XWEN_METRICS_FILE";

/// Tags every record this process writes as harness-driven rather than real
/// use. The scripts that drive the binary export it (`bench` for the bench and
/// tuning sweeps, `parity` for the gate); nothing else sets it, and an empty
/// value counts as unset.
pub const TAG_ENV: &str = "XWEN_METRICS_TAG";

/// The current record schema. Readers accept any version and ignore fields they
/// do not know, so a newer xwen's history stays readable by an older one.
pub const SCHEMA_VERSION: u32 = 1;

fn default_true() -> bool {
    true
}

/// What one finished run cost, as every surface reports it.
///
/// The token counts are the same ones the surface tells its own caller, so a
/// row here and the `--stats` line or the API usage object of the same run
/// agree by construction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunRecord {
    /// Schema version.
    pub v: u32,
    /// Unix seconds at run completion.
    pub ts: u64,
    /// `generate` | `chat` | `batch` | `serve:openai` | `serve:anthropic` |
    /// `serve:native` | `serve:batch`.
    pub surface: String,
    /// The checkpoint's full name, or the served file's own id when a server
    /// runs a GGUF that is none of the official checkpoints.
    pub model: String,
    /// Tokens in the prompt the run was given.
    pub prompt_tokens: usize,
    /// Prompt tokens answered out of the KV cache instead of being forwarded.
    pub cached_tokens: usize,
    /// Tokens actually forwarded through the model, and the wall time they took.
    ///
    /// For an ordinary run `prompt_tokens == cached_tokens + prefill_tokens`,
    /// but a reader must not assume the identity: a scored batch forwards more
    /// than its prompt, because every teacher-forced trial is real work on no
    /// prompt token, and an abandoned or failed job forwards less than it was
    /// given. The three are measured separately and each means what it says.
    pub prefill_tokens: usize,
    pub prefill_secs: f64,
    pub decode_tokens: usize,
    pub decode_secs: f64,
    /// Emitted tokens that were reasoning, or `None` where the run did not
    /// measure them — which is not the same as zero.
    ///
    /// `serve` always counts reasoning and writes `Some(n)`, `0` included.
    /// `generate` and `chat` count it only under a thinking budget
    /// (`--max-think`), so an unbudgeted run that reasoned at length still
    /// reports `None`. A reader summing this field is summing the runs that
    /// measured it, not the reasoning the machine did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_tokens: Option<usize>,
    /// Draft positions proposed and accepted, for a run a drafter ran on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drafted: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accepted: Option<usize>,
    /// Items in a batch run. One record covers the whole batch, not each item.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<usize>,
    /// The identifier the request body carried, verbatim. Undocumented and not
    /// stable in shape, which is why it is stored raw and read by
    /// [`session_key`] rather than parsed here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client: Option<String>,
    /// The `x-claude-code-session-id` header, when the request carried one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    /// The `x-claude-code-agent-id` header, when the request carried one. It
    /// rides subagent requests only, so most runs of a session have none; it is
    /// its own field and never feeds [`session_key`], which keeps one session
    /// one row however many agents worked inside it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    /// What kind of run this was, for a history that mixes real use with the
    /// runs a harness drove. `None` is a real run — a person or a client asked
    /// for it — and every tagged value names the harness that did
    /// (`bench`, `parity`). Set from [`TAG_ENV`] at the moment the record is
    /// stamped, so a script exports it once and every surface it drives records
    /// it without knowing the field exists.
    ///
    /// `xwen stats` leaves tagged records out of its default report: a sweep's
    /// several hundred runs would otherwise read as a day of inference nobody
    /// did. The exclusion is always stated in the footer, never silent, and
    /// `--all-tags` puts them back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    /// Whether the run reached its own natural end: an end-of-generation token
    /// or its token cap, with no error.
    ///
    /// False for a failure, and false for a run cut short — a client that hung
    /// up, a deadline, a shutdown, a Ctrl-C'd chat turn. Those spent real tokens
    /// and are recorded with the counts they reached, but the counts describe an
    /// interrupted run and a reader averaging over them without filtering on
    /// this field is averaging over two different things.
    #[serde(default = "default_true")]
    pub ok: bool,
}

impl RunRecord {
    /// A record stamped now, with everything a run always has and nothing it
    /// may not: the caller fills the optional fields it can answer for.
    pub fn new(surface: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            v: SCHEMA_VERSION,
            ts: now_secs(),
            surface: surface.into(),
            model: model.into(),
            prompt_tokens: 0,
            cached_tokens: 0,
            prefill_tokens: 0,
            prefill_secs: 0.0,
            decode_tokens: 0,
            decode_secs: 0.0,
            thinking_tokens: None,
            drafted: None,
            accepted: None,
            items: None,
            client: None,
            session: None,
            agent: None,
            tag: tag_from_env(),
            ok: true,
        }
    }
}

/// The tag this process stamps on every record it writes, read from
/// [`TAG_ENV`]. Read per record rather than cached: a record is written once
/// per run, and a cached value would be one more thing to get wrong in a test.
pub fn tag_from_env() -> Option<String> {
    tag_from(std::env::var_os(TAG_ENV).as_deref())
}

/// [`tag_from_env`] over a value rather than the process environment, which is
/// what makes the rule testable — the same reason [`metrics_path_from`] exists.
/// An empty value is how a shell spells "unset" by accident and names no tag.
pub fn tag_from(env: Option<&OsStr>) -> Option<String> {
    env.and_then(OsStr::to_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// The label `--by session`, `--by client` and `--by agent` use for a run that
/// named nobody.
pub const UNATTRIBUTED: &str = "-";

/// The session a record belongs to.
///
/// The header is the answer whenever it is there: it is the documented
/// per-session identifier, and its value is the session id itself. Older
/// records have only the body id, whose shape has changed between Claude Code
/// releases — one published capture spells it `user_<hex>_account_<uuid>_session_<uuid>`,
/// another embeds a JSON `"session_id":"<uuid>"`. Both put the id right after a
/// `session_` marker, so the last one in the string is what this reads past,
/// which is why the raw value is what gets stored: a reader can learn a new
/// shape long after the run was recorded.
pub fn session_key(record: &RunRecord) -> String {
    if let Some(session) = record.session.as_deref().filter(|s| !s.is_empty()) {
        return session.to_string();
    }
    record
        .client
        .as_deref()
        .and_then(session_from_client)
        .unwrap_or_else(|| UNATTRIBUTED.to_string())
}

/// The session id embedded in a body identifier, in either shape it has worn.
fn session_from_client(client: &str) -> Option<String> {
    const MARKER: &str = "session_";
    let after = &client[client.rfind(MARKER)? + MARKER.len()..];
    // `session_id` in the JSON shape, `session_<uuid>` in the underscore one.
    // Stripping the literal `id` rather than a character class matters: a uuid
    // may well begin with a `d`.
    let after = after.strip_prefix("id").unwrap_or(after);
    let after = after.trim_start_matches(['_', '"', ':', ' ', '\t']);
    let id: String = after
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect();
    (!id.is_empty()).then_some(id)
}

/// Seconds since the Unix epoch, 0 on a clock set before it.
pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

// ------------------------------------------------------------------ writing --

/// The file runs record into, or `None` when recording is off: either
/// `XWEN_METRICS_FILE` says `off`, or there is no `$HOME` to resolve the
/// default under.
pub fn metrics_path() -> Option<PathBuf> {
    metrics_path_from(
        std::env::var_os(METRICS_ENV).as_deref(),
        std::env::var_os("HOME").as_deref(),
    )
}

/// [`metrics_path`] over values rather than the process environment, which is
/// what makes the rule testable: a test that set the variable would be changing
/// state every other thread in the runner shares.
pub fn metrics_path_from(env: Option<&OsStr>, home: Option<&OsStr>) -> Option<PathBuf> {
    // Case-insensitive: `off`, `OFF` and `Off` are one instruction, and a shell
    // that spells it the second way must not silently keep recording.
    let off = env
        .and_then(OsStr::to_str)
        .is_some_and(|value| value.eq_ignore_ascii_case("off"));
    if off {
        return None;
    }
    match env {
        // An empty value is how a shell spells "unset" by accident — `FOO=`
        // and an unexported FOO reach a process alike. It names no file, so it
        // falls through to the default rather than disabling recording, which
        // only `off` does.
        Some(value) if !value.is_empty() => Some(PathBuf::from(value)),
        _ => home.map(|home| PathBuf::from(home).join(METRICS_RELATIVE_PATH)),
    }
}

/// Append one record to `path`, creating the file and, if it is missing, the
/// directory holding it.
///
/// The open is tried first and the directory created only when that is what was
/// missing: this runs on the server's inference thread once per request, where
/// the ordinary case is a directory that has existed since the first run and a
/// syscall asking again every time would buy nothing.
pub fn append(path: &Path, rec: &RunRecord) -> Result<()> {
    let mut line = serde_json::to_string(rec)?;
    line.push('\n');
    let mut file = match open_for_append(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            if let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
            open_for_append(path).with_context(|| format!("opening {}", path.display()))?
        }
        other => other.with_context(|| format!("opening {}", path.display()))?,
    };
    // One write of the whole line, terminator included: a concurrent appender
    // to the same file must find whole records, never a spliced pair.
    file.write_all(line.as_bytes())
        .with_context(|| format!("appending to {}", path.display()))?;
    file.flush()?;
    Ok(())
}

fn open_for_append(path: &Path) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
}

/// Record one run where recording is configured to go. `Ok(())` when recording
/// is off: there was nowhere to write it, which is not a failure to write it.
pub fn record(rec: &RunRecord) -> Result<()> {
    match metrics_path() {
        Some(path) => append(&path, rec),
        None => Ok(()),
    }
}

/// Whether this process has already said that recording is failing.
static WARNED: AtomicBool = AtomicBool::new(false);

/// Record one run, returning the single line this process will ever say about
/// failing to. Metrics are a side effect of a run and never a reason for one to
/// fail, and a broken path would otherwise repeat itself once per request for
/// the life of a server.
pub fn record_warning(rec: &RunRecord) -> Option<String> {
    let error = record(rec).err()?;
    if WARNED.swap(true, Ordering::Relaxed) {
        return None;
    }
    let path = metrics_path().unwrap_or_else(|| PathBuf::from("-"));
    Some(format!(
        "warning: metrics not recorded ({}): {error:#}",
        path.display()
    ))
}

/// [`record_warning`] with the warning on stderr, for a surface that owns it.
pub fn record_quietly(rec: &RunRecord) {
    if let Some(line) = record_warning(rec) {
        eprintln!("{line}");
    }
}

// ---------------------------------------------------------------- local time --

/// This machine's UTC offset in seconds, or `None` when it cannot be read.
///
/// `std` has no local time at all, so this is worth a subprocess: a history
/// bucketed by UTC day would cut the evening off the day an operator spent it
/// in, and a dashboard clock exists to be compared against a client's own log.
/// The offset is read at the call site rather than cached, which is what lets a
/// long-running server cross a daylight-saving change — though one reading is
/// applied to every record in a report, so a report spanning a change buckets
/// the far side of it by an hour off. That is ledgered rather than fixed.
pub fn read_utc_offset() -> Option<i64> {
    let output = std::process::Command::new("date")
        .arg("+%z")
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let field = text.trim();
    // `get`, not `split_at`: a `date` that printed something else entirely —
    // a localized word, a multibyte character — would panic on a split that
    // landed inside one. Odd output yields no offset instead.
    let cut = field.len().checked_sub(4)?;
    let sign = field.get(..cut)?;
    let digits = field.get(cut..)?;
    let hours: i64 = digits.get(..2)?.parse().ok()?;
    let minutes: i64 = digits.get(2..)?.parse().ok()?;
    let magnitude = hours * 3600 + minutes * 60;
    Some(if sign == "-" { -magnitude } else { magnitude })
}

/// Days since the Unix epoch to a proleptic-Gregorian `(year, month, day)`.
/// Howard Hinnant's `civil_from_days`, exact over the whole representable range
/// and free of the leap-year special cases a hand-rolled loop gets wrong.
pub fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * shifted_month + 2) / 5 + 1) as u32;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

/// The inverse of [`civil_from_days`]: a civil date to days since the epoch.
pub fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let shifted_month = if month > 2 { month - 3 } else { month + 9 } as i64;
    let day_of_year = (153 * shifted_month + 2) / 5 + day as i64 - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// How many days a month has, leap years included. Callers pass a month already
/// known to be in 1..=12.
pub fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// The proleptic-Gregorian leap rule: every fourth year, except centuries, except
/// every fourth century.
pub fn is_leap_year(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

/// The local day a timestamp falls in, as days since the epoch.
pub fn local_day(ts: u64, utc_offset: i64) -> i64 {
    (ts as i64 + utc_offset).div_euclid(86_400)
}

/// The Monday of the week a day belongs to. 1970-01-01 was a Thursday, which is
/// what the +3 encodes.
pub fn week_start(days: i64) -> i64 {
    days - (days + 3).rem_euclid(7)
}

fn day_label(days: i64) -> String {
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}")
}

fn month_label(days: i64) -> String {
    let (year, month, _) = civil_from_days(days);
    format!("{year:04}-{month:02}")
}

// ------------------------------------------------------------------ queries --

/// What a row of the report covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupBy {
    Day,
    Week,
    Month,
    Model,
    Surface,
    /// The raw body identifier the client sent.
    Client,
    /// The session a run belonged to, as [`session_key`] derives it.
    Session,
    /// The Claude Code agent a run came from, which is the subagents only: a
    /// run the header named nobody for is [`UNATTRIBUTED`], never folded into
    /// the session it belonged to.
    Agent,
    /// Everything the filters kept, as one row.
    All,
}

impl GroupBy {
    /// What the first column of the table is called.
    pub fn column(self) -> &'static str {
        match self {
            GroupBy::Day | GroupBy::Week | GroupBy::Month => "period",
            GroupBy::All => "all",
            GroupBy::Model => "model",
            GroupBy::Surface => "surface",
            GroupBy::Client => "client",
            GroupBy::Session => "session",
            GroupBy::Agent => "agent",
        }
    }

    /// Whether rows order by their label (a period) or by size (a name).
    fn chronological(self) -> bool {
        matches!(self, GroupBy::Day | GroupBy::Week | GroupBy::Month)
    }

    fn bucket(self, rec: &RunRecord, utc_offset: i64) -> String {
        match self {
            GroupBy::Day => day_label(local_day(rec.ts, utc_offset)),
            GroupBy::Week => day_label(week_start(local_day(rec.ts, utc_offset))),
            GroupBy::Month => month_label(local_day(rec.ts, utc_offset)),
            GroupBy::Model => rec.model.clone(),
            GroupBy::Surface => rec.surface.clone(),
            GroupBy::Client => rec
                .client
                .clone()
                .filter(|client| !client.is_empty())
                .unwrap_or_else(|| UNATTRIBUTED.to_string()),
            GroupBy::Session => session_key(rec),
            GroupBy::Agent => rec
                .agent
                .clone()
                .filter(|agent| !agent.is_empty())
                .unwrap_or_else(|| UNATTRIBUTED.to_string()),
            GroupBy::All => "all".to_string(),
        }
    }
}

impl std::str::FromStr for GroupBy {
    type Err = anyhow::Error;

    fn from_str(text: &str) -> Result<Self> {
        match text {
            "day" => Ok(GroupBy::Day),
            "week" => Ok(GroupBy::Week),
            "month" => Ok(GroupBy::Month),
            "model" => Ok(GroupBy::Model),
            "surface" => Ok(GroupBy::Surface),
            "client" => Ok(GroupBy::Client),
            "session" => Ok(GroupBy::Session),
            "agent" => Ok(GroupBy::Agent),
            "all" => Ok(GroupBy::All),
            other => bail!(
                "unknown --by {other:?} \
                 (expected day|week|month|model|surface|client|session|agent|all)"
            ),
        }
    }
}

/// The earliest timestamp a `--since` value admits.
///
/// `7d` / `24h` / `4w` count back from `now`; `YYYY-MM-DD` means local midnight
/// of that day, which is the boundary someone reading a daily table means.
pub fn parse_since(spec: &str, now: u64, utc_offset: i64) -> Result<u64> {
    let spec = spec.trim();
    if let Some((count, unit)) = spec.split_at_checked(spec.len().saturating_sub(1)) {
        let seconds = match unit {
            "h" => Some(3_600u64),
            "d" => Some(86_400),
            "w" => Some(604_800),
            _ => None,
        };
        if let (Some(seconds), Ok(count)) = (seconds, count.parse::<u64>()) {
            return Ok(now.saturating_sub(count.saturating_mul(seconds)));
        }
    }
    let mut fields = spec.split('-');
    let parsed = (|| {
        let year: i64 = fields.next()?.parse().ok()?;
        let month: u32 = fields.next()?.parse().ok()?;
        let day: u32 = fields.next()?.parse().ok()?;
        // Checked against the month rather than a flat 1..=31: `days_from_civil`
        // is arithmetic, not a calendar, so it would silently read 2026-02-31 as
        // the 3rd of March and report a window the caller never asked for.
        if fields.next().is_some()
            || !(1..=12).contains(&month)
            || day < 1
            || day > days_in_month(year, month)
        {
            return None;
        }
        Some(days_from_civil(year, month, day) * 86_400 - utc_offset)
    })();
    match parsed {
        Some(midnight) => Ok(midnight.max(0) as u64),
        None => bail!("cannot read --since {spec:?} (expected 24h, 7d, 4w or YYYY-MM-DD)"),
    }
}

/// Which population of the history a report covers.
///
/// The history records everything, harness runs included — silent exclusion at
/// write time is the harder mistake to notice (decisions.md "Metrics"). The
/// separation therefore happens at read time, and the default is the question
/// the table is nearly always asked: what did real use cost.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum TagFilter {
    /// Untagged records only: real use, with every harness run left out. The
    /// count left out is reported, so the exclusion is never silent.
    #[default]
    Untagged,
    /// One tag's records and nothing else, for reading a sweep on its own.
    Only(String),
    /// Everything in the file, tagged and untagged alike.
    All,
}

impl TagFilter {
    /// Whether this record belongs in the report.
    pub fn keeps(&self, rec: &RunRecord) -> bool {
        match self {
            TagFilter::All => true,
            TagFilter::Untagged => rec.tag.is_none(),
            TagFilter::Only(want) => rec.tag.as_deref() == Some(want.as_str()),
        }
    }
}

/// What `xwen stats` was asked for.
#[derive(Debug, Clone)]
pub struct StatsQuery {
    pub by: GroupBy,
    /// Unparsed, because parsing it needs the clock and the local offset.
    pub since: Option<String>,
    pub model: Option<String>,
    pub surface: Option<String>,
    /// Substring of the raw body identifier. A substring rather than an exact
    /// match because the raw ids run to a hundred characters, so the only
    /// usable way to name one is by the part a caller recognizes.
    pub client: Option<String>,
    /// Substring of the derived session key, matched the same way.
    pub session: Option<String>,
    /// Which population to report on. Defaults to real use alone; see
    /// [`TagFilter`].
    pub tag: TagFilter,
    /// Read this file instead of the configured one. Reading only: nothing
    /// records to a `--file` path.
    pub file: Option<PathBuf>,
}

impl Default for StatsQuery {
    fn default() -> Self {
        Self {
            by: GroupBy::Day,
            since: None,
            model: None,
            surface: None,
            client: None,
            session: None,
            tag: TagFilter::default(),
            file: None,
        }
    }
}

// -------------------------------------------------------------- aggregation --

/// One row of the report: the runs of a bucket, summed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Bucket {
    pub label: String,
    pub runs: usize,
    /// Of those runs, how many did not reach their own end: an error, or a
    /// client that hung up, a deadline, a shutdown, a cancelled turn. The rows
    /// sum every run either way, so this is what lets a consumer separate the
    /// two populations without re-reading the history.
    pub unfinished: usize,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub prefill_tokens: u64,
    pub prefill_secs: f64,
    pub decode_tokens: u64,
    pub decode_secs: f64,
    pub drafted: u64,
    pub accepted: u64,
}

impl Bucket {
    fn add(&mut self, rec: &RunRecord) {
        self.runs += 1;
        self.unfinished += usize::from(!rec.ok);
        self.prompt_tokens += rec.prompt_tokens as u64;
        self.cached_tokens += rec.cached_tokens as u64;
        self.prefill_tokens += rec.prefill_tokens as u64;
        self.prefill_secs += rec.prefill_secs;
        self.decode_tokens += rec.decode_tokens as u64;
        self.decode_secs += rec.decode_secs;
        self.drafted += rec.drafted.unwrap_or(0) as u64;
        self.accepted += rec.accepted.unwrap_or(0) as u64;
    }

    /// Tokens over seconds across the whole bucket — never a mean of per-run
    /// rates, which would weigh a two-token reply like an hour of decoding.
    pub fn prefill_tps(&self) -> Option<f64> {
        (self.prefill_secs > 0.0).then(|| self.prefill_tokens as f64 / self.prefill_secs)
    }

    pub fn decode_tps(&self) -> Option<f64> {
        (self.decode_secs > 0.0).then(|| self.decode_tokens as f64 / self.decode_secs)
    }

    /// Share of the prompt that came out of a cache instead of a prefill.
    pub fn hit_rate(&self) -> Option<f64> {
        (self.prompt_tokens > 0).then(|| self.cached_tokens as f64 / self.prompt_tokens as f64)
    }

    /// Share of the drafted positions the target accepted, for buckets where a
    /// drafter ran at all.
    pub fn acceptance(&self) -> Option<f64> {
        (self.drafted > 0).then(|| self.accepted as f64 / self.drafted as f64)
    }

    /// Every row summed into one, under `label`.
    pub fn fold(rows: &[Bucket], label: &str) -> Bucket {
        let mut total = Bucket {
            label: label.to_string(),
            ..Bucket::default()
        };
        for row in rows {
            total.runs += row.runs;
            total.unfinished += row.unfinished;
            total.prompt_tokens += row.prompt_tokens;
            total.cached_tokens += row.cached_tokens;
            total.prefill_tokens += row.prefill_tokens;
            total.prefill_secs += row.prefill_secs;
            total.decode_tokens += row.decode_tokens;
            total.decode_secs += row.decode_secs;
            total.drafted += row.drafted;
            total.accepted += row.accepted;
        }
        total
    }
}

/// Group records into rows. Time buckets come out in chronological order, name
/// buckets heaviest first — a table read top-down should answer the question
/// its grouping was chosen for.
pub fn aggregate(
    records: impl Iterator<Item = RunRecord>,
    by: GroupBy,
    utc_offset: i64,
) -> Vec<Bucket> {
    let mut rows: Vec<Bucket> = Vec::new();
    let mut index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for rec in records {
        let label = by.bucket(&rec, utc_offset);
        let slot = *index.entry(label.clone()).or_insert_with(|| {
            rows.push(Bucket {
                label,
                ..Bucket::default()
            });
            rows.len() - 1
        });
        rows[slot].add(&rec);
    }
    if by.chronological() {
        rows.sort_by(|a, b| a.label.cmp(&b.label));
    } else {
        rows.sort_by(|a, b| {
            b.decode_tokens
                .cmp(&a.decode_tokens)
                .then_with(|| a.label.cmp(&b.label))
        });
    }
    rows
}

// ------------------------------------------------------------------ reading --

/// A history file as read: the records that parsed, and how many lines did not.
#[derive(Debug, Clone, Default)]
pub struct History {
    pub records: Vec<RunRecord>,
    /// Lines that were not a record. Kept rather than refused: a truncated
    /// tail from a killed process must not make the whole history unreadable.
    pub skipped: usize,
}

/// Read a history file. `None` when there is no file to read.
pub fn load(path: &Path) -> Result<Option<History>> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
    };
    // Split on newlines and decode each line on its own. Reading the whole file
    // as text would let one torn multibyte record — a process killed mid-write —
    // make every other line in the history unreadable.
    let mut history = History::default();
    for line in bytes.split(|byte| *byte == b'\n') {
        let Ok(line) = std::str::from_utf8(line) else {
            history.skipped += 1;
            continue;
        };
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<RunRecord>(line) {
            Ok(rec) => history.records.push(rec),
            Err(_) => history.skipped += 1,
        }
    }
    Ok(Some(history))
}

/// What one `xwen stats` run found.
#[derive(Debug, Clone)]
pub struct StatsReport {
    pub path: PathBuf,
    pub by: GroupBy,
    pub rows: Vec<Bucket>,
    /// Records in the file, before the query's filters.
    pub records: usize,
    /// Records the filters kept.
    pub matched: usize,
    /// Records the tag filter alone left out, counted before any other filter
    /// narrowed the file. A default report of a history a sweep has run
    /// through says this out loud rather than quietly reporting fewer runs
    /// than the file holds.
    pub excluded_by_tag: usize,
    pub skipped: usize,
    /// Whether the machine's UTC offset could be read. When it could not, the
    /// rows were bucketed in UTC — which is a correct report of the wrong days,
    /// and indistinguishable from a genuinely UTC machine unless it is said.
    pub local_offset_known: bool,
}

/// Answer a query against the history. `None` when recording is off, or when
/// there is no history yet — nothing to report is not an error.
pub fn report(query: &StatsQuery) -> Result<Option<StatsReport>> {
    // Arguments are judged before the history is looked at, so that a
    // misspelled `--since` is the same error whether or not any runs have been
    // recorded yet. Otherwise the first thing a new user sees on a typo is
    // "no metrics recorded yet", which answers a question they did not ask.
    let local_offset = read_utc_offset();
    let utc_offset = local_offset.unwrap_or(0);
    let since = query
        .since
        .as_deref()
        .map(|spec| parse_since(spec, now_secs(), utc_offset))
        .transpose()?;

    let path = match query.file.clone().or_else(metrics_path) {
        Some(path) => path,
        None => return Ok(None),
    };
    let Some(history) = load(&path)? else {
        return Ok(None);
    };
    if history.records.is_empty() && history.skipped == 0 {
        return Ok(None);
    }
    let records = history.records.len();
    // Counted over the whole file, before `--since` or any other filter has
    // narrowed it: the footer's claim is "this many records in this file are
    // not real use", which is a property of the file rather than of the query.
    let excluded_by_tag = history
        .records
        .iter()
        .filter(|rec| !query.tag.keeps(rec))
        .count();
    let kept: Vec<RunRecord> = history
        .records
        .into_iter()
        .filter(|rec| query.tag.keeps(rec))
        .filter(|rec| since.is_none_or(|since| rec.ts >= since))
        .filter(|rec| query.model.as_ref().is_none_or(|want| &rec.model == want))
        .filter(|rec| {
            query
                .surface
                .as_ref()
                .is_none_or(|want| &rec.surface == want)
        })
        .filter(|rec| {
            query.client.as_ref().is_none_or(|want| {
                rec.client
                    .as_deref()
                    .is_some_and(|client| client.contains(want.as_str()))
            })
        })
        .filter(|rec| {
            query
                .session
                .as_ref()
                .is_none_or(|want| session_key(rec).contains(want.as_str()))
        })
        .collect();
    let matched = kept.len();
    Ok(Some(StatsReport {
        path,
        by: query.by,
        rows: aggregate(kept.into_iter(), query.by, utc_offset),
        records,
        matched,
        excluded_by_tag,
        skipped: history.skipped,
        local_offset_known: local_offset.is_some(),
    }))
}

/// The path `xwen stats` would read for a query, for a message about a history
/// that is not there yet.
pub fn query_path(query: &StatsQuery) -> Option<PathBuf> {
    query.file.clone().or_else(metrics_path)
}

// ---------------------------------------------------------------- rendering --

/// An integer with thousands separators.
fn commas(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (seen, ch) in digits.chars().enumerate() {
        if seen > 0 && (digits.len() - seen).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// A rate, or a dash for a bucket that spent no time in that phase.
fn rate(value: Option<f64>, decimals: usize) -> String {
    match value {
        Some(value) => format!("{value:.decimals$}"),
        None => "-".to_string(),
    }
}

/// A share as a percentage, or a dash when nothing was measured.
fn percent(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{:.1}", value * 100.0),
        None => "-".to_string(),
    }
}

fn cells(row: &Bucket) -> Vec<String> {
    vec![
        row.label.clone(),
        commas(row.runs as u64),
        commas(row.prompt_tokens),
        commas(row.cached_tokens),
        percent(row.hit_rate()),
        commas(row.prefill_tokens),
        rate(row.prefill_tps(), 0),
        commas(row.decode_tokens),
        rate(row.decode_tps(), 1),
        percent(row.acceptance()),
    ]
}

/// The report as an aligned table, header and `total` row included. Ends in a
/// newline when there is anything at all to show.
pub fn render_table(rows: &[Bucket], by: GroupBy) -> String {
    let header: Vec<String> = [
        by.column(),
        "runs",
        "prompt",
        "cached",
        "hit%",
        "prefill",
        "pf tok/s",
        "decode",
        "dec tok/s",
        "accept%",
    ]
    .iter()
    .map(|name| (*name).to_string())
    .collect();
    let total = Bucket::fold(rows, "total");
    let mut lines: Vec<Vec<String>> = vec![header];
    lines.extend(rows.iter().map(cells));
    lines.push(cells(&total));

    // A label is whatever the world put in it: a custom GGUF's file stem, a
    // client id a caller chose. Cap it so one long name cannot push every
    // number off the right of a terminal.
    for line in lines.iter_mut() {
        line[0] = truncate_to_width(&line[0], LABEL_MAX_WIDTH);
    }

    let columns = lines[0].len();
    // Display WIDTH, not character count: a CJK label is two columns per
    // character and an emoji likewise, and padding by count would leave every
    // number in the row shifted right by the difference.
    let widths: Vec<usize> = (0..columns)
        .map(|column| {
            lines
                .iter()
                .map(|line| line[column].width())
                .max()
                .unwrap_or(0)
        })
        .collect();

    let mut out = String::new();
    for (index, line) in lines.iter().enumerate() {
        // A rule between the buckets and the row that sums them.
        if index == lines.len() - 1 {
            let rule: Vec<String> = widths.iter().map(|width| "-".repeat(*width)).collect();
            out.push_str(&join_row(&rule, &widths));
            out.push('\n');
        }
        out.push_str(&join_row(line, &widths));
        out.push('\n');
    }
    out
}

/// The widest a label column grows before it is cut. Wide enough for a session
/// uuid (36) and a checkpoint name, narrow enough that the numbers still fit
/// beside it on an ordinary terminal.
pub const LABEL_MAX_WIDTH: usize = 48;

/// `text` cut to `width` display columns, ending in an ellipsis when anything
/// was dropped. Measured in columns rather than characters, so a CJK label is
/// cut where it actually reaches the limit; a double-width character that would
/// straddle the boundary is dropped whole rather than split.
fn truncate_to_width(text: &str, width: usize) -> String {
    if text.width() <= width {
        return text.to_string();
    }
    // One column is spent on the ellipsis itself.
    let budget = width.saturating_sub(1);
    let mut out = String::new();
    let mut used = 0;
    for ch in text.chars() {
        let w = ch.to_string().width();
        if used + w > budget {
            break;
        }
        out.push(ch);
        used += w;
    }
    out.push('\u{2026}');
    out
}

/// The label column reads left-aligned; every number column right-aligned, so
/// magnitudes line up down the table.
fn join_row(line: &[String], widths: &[usize]) -> String {
    let mut out = String::new();
    for (column, cell) in line.iter().enumerate() {
        if column > 0 {
            out.push_str("  ");
        }
        let pad = widths[column].saturating_sub(cell.width());
        if column == 0 {
            out.push_str(cell);
            out.push_str(&" ".repeat(pad));
        } else {
            out.push_str(&" ".repeat(pad));
            out.push_str(cell);
        }
    }
    // Trailing padding on the last column would be invisible but real.
    out.trim_end().to_string()
}

/// The report as JSON: the rows the table shows, with the sums it derives its
/// rates from, and no `total` row — a caller summing them has the same numbers.
pub fn rows_json(rows: &[Bucket]) -> serde_json::Value {
    serde_json::Value::Array(
        rows.iter()
            .map(|row| {
                serde_json::json!({
                    "label": row.label,
                    "runs": row.runs,
                    "unfinished": row.unfinished,
                    "prompt_tokens": row.prompt_tokens,
                    "cached_tokens": row.cached_tokens,
                    "hit_rate": row.hit_rate(),
                    "prefill_tokens": row.prefill_tokens,
                    "prefill_secs": row.prefill_secs,
                    "prefill_tps": row.prefill_tps(),
                    "decode_tokens": row.decode_tokens,
                    "decode_secs": row.decode_secs,
                    "decode_tps": row.decode_tps(),
                    "drafted": row.drafted,
                    "accepted": row.accepted,
                    "acceptance": row.acceptance(),
                })
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A history file of this test's own, under a name no other test uses.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xwen-metrics-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a scratch dir");
        let path = dir.join(name);
        let _ = std::fs::remove_file(&path);
        path
    }

    fn run(ts: u64, surface: &str, model: &str) -> RunRecord {
        RunRecord {
            ts,
            surface: surface.to_string(),
            model: model.to_string(),
            // Explicit, because `RunRecord::new` reads the tag out of the
            // process environment: a runner with `XWEN_METRICS_TAG` exported
            // would otherwise hand every one of these tests a tagged record.
            tag: None,
            ..RunRecord::new(surface, model)
        }
    }

    /// The record is the wire format of the history file, so it has to survive
    /// a round trip byte for byte in the fields it carries.
    #[test]
    fn a_record_round_trips_through_json() {
        let mut rec = run(1_757_030_400, "generate", "Qwen3.6-27B");
        rec.prompt_tokens = 512;
        rec.prefill_tokens = 512;
        rec.prefill_secs = 0.8;
        rec.decode_tokens = 128;
        rec.decode_secs = 3.4;
        rec.thinking_tokens = Some(40);
        rec.drafted = Some(200);
        rec.accepted = Some(160);
        rec.session = Some("9f2ca1b4-0d31-4e77-9a02-7c1f8b6e5d40".to_string());
        rec.agent = Some("explore-metrics".to_string());
        rec.tag = Some("bench".to_string());
        let line = serde_json::to_string(&rec).expect("a record serializes");
        assert!(line.contains(r#""tag":"bench""#));
        let back: RunRecord = serde_json::from_str(&line).expect("a record parses");
        assert_eq!(rec, back);
    }

    /// The tag comes from the environment at the moment the record is stamped,
    /// so a script exports it once and every surface it drives records it.
    /// A value that is only whitespace, or empty, names no tag: that is how a
    /// shell spells "unset" by accident, and it must not invent a tag called
    /// nothing that the default report would then hide runs under.
    #[test]
    fn the_tag_comes_from_the_environment_and_an_empty_value_is_no_tag() {
        use std::ffi::OsString;
        assert_eq!(
            tag_from(Some(&OsString::from("bench"))),
            Some("bench".into())
        );
        assert_eq!(
            tag_from(Some(&OsString::from("  parity "))),
            Some("parity".into())
        );
        assert_eq!(tag_from(Some(&OsString::from(""))), None);
        assert_eq!(tag_from(Some(&OsString::from("   "))), None);
        assert_eq!(tag_from(None), None);
    }

    /// Three populations out of one file. The default is real use; `--tag`
    /// reads one harness on its own; `--all-tags` is the whole file. The
    /// excluded count is the file's, not the query's, so it is stable under
    /// the other filters — it is what the footer promises a reader.
    #[test]
    fn the_tag_filter_separates_harness_runs_from_real_use() {
        let mut real = run(10, "generate", "Qwen3.8-27B");
        real.decode_tokens = 10;
        let mut benched = run(20, "generate", "Qwen3.8-27B");
        benched.tag = Some("bench".to_string());
        benched.decode_tokens = 20;
        let mut gated = run(30, "generate", "Qwen3.8-27B");
        gated.tag = Some("parity".to_string());
        gated.decode_tokens = 40;

        let path = scratch("tags.jsonl");
        for rec in [&real, &benched, &gated] {
            append(&path, rec).expect("a record appends");
        }
        let asked = |tag: TagFilter| {
            report(&StatsQuery {
                by: GroupBy::All,
                tag,
                file: Some(path.clone()),
                ..StatsQuery::default()
            })
            .expect("the history reads")
            .expect("a history")
        };

        let default = asked(TagFilter::Untagged);
        assert_eq!(default.matched, 1, "only the run a person asked for");
        assert_eq!(default.rows[0].decode_tokens, 10);
        assert_eq!(
            default.excluded_by_tag, 2,
            "and the report says how many it left out"
        );

        let bench = asked(TagFilter::Only("bench".to_string()));
        assert_eq!(bench.matched, 1);
        assert_eq!(bench.rows[0].decode_tokens, 20);
        assert_eq!(
            bench.excluded_by_tag, 2,
            "the real run and the other harness are both outside this report"
        );

        let all = asked(TagFilter::All);
        assert_eq!(all.matched, 3);
        assert_eq!(all.rows[0].decode_tokens, 70);
        assert_eq!(all.excluded_by_tag, 0, "nothing is excluded by --all-tags");
    }

    /// Absent optional fields stay absent on the wire, and a reader accepts a
    /// record a newer version wrote: unknown keys are ignored, and `ok`
    /// defaults to a successful run for a record written before it existed.
    #[test]
    fn a_record_omits_what_it_has_none_of_and_tolerates_unknown_fields() {
        let rec = run(10, "chat", "Qwen3.8-27B");
        let line = serde_json::to_string(&rec).expect("a record serializes");
        assert!(!line.contains("thinking_tokens"));
        assert!(!line.contains("drafted"));
        assert!(!line.contains("items"));
        assert!(!line.contains("agent"));
        assert!(
            !line.contains("tag"),
            "an untagged run is real use and writes no tag key"
        );
        let back: RunRecord = serde_json::from_str(&line).expect("a record parses");
        assert_eq!(
            back.agent, None,
            "a run no agent header named reads back as nobody's"
        );

        let future = r#"{"v":2,"ts":10,"surface":"chat","model":"Qwen3.8-27B",
            "prompt_tokens":0,"cached_tokens":0,"prefill_tokens":0,"prefill_secs":0.0,
            "decode_tokens":0,"decode_secs":0.0,"gpu_watts":31.5}"#;
        let back: RunRecord = serde_json::from_str(future).expect("a newer record still parses");
        assert_eq!(back.v, 2);
        assert_eq!(back.surface, "chat");
        assert!(
            back.ok,
            "a record without the field is a run that succeeded"
        );
    }

    #[test]
    fn the_civil_calendar_matches_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        // A leap day, and the century that is not a leap year.
        assert_eq!(civil_from_days(days_from_civil(2024, 2, 29)), (2024, 2, 29));
        assert_eq!(civil_from_days(days_from_civil(1900, 3, 1)), (1900, 3, 1));
        assert_eq!(local_day(1_757_030_400, 0), days_from_civil(2025, 9, 5));
        assert_eq!(civil_from_days(local_day(1_757_030_400, 0)), (2025, 9, 5));
    }

    /// The bucket a run lands in is its LOCAL day: a run just after midnight
    /// UTC belongs to the previous evening for anyone west of it.
    #[test]
    fn a_local_offset_moves_a_run_across_midnight() {
        // 2025-09-05 00:30 UTC.
        let ts = 1_757_030_400 + 1_800;
        assert_eq!(civil_from_days(local_day(ts, 0)), (2025, 9, 5));
        assert_eq!(civil_from_days(local_day(ts, -4 * 3600)), (2025, 9, 4));
        assert_eq!(civil_from_days(local_day(ts, 2 * 3600)), (2025, 9, 5));
    }

    /// A week is labelled by its Monday, whichever day of it a run fell on.
    #[test]
    fn a_week_is_labelled_by_its_monday() {
        // 2025-09-05 was a Friday; 2025-09-08 the Monday after it.
        let friday = days_from_civil(2025, 9, 5);
        assert_eq!(civil_from_days(week_start(friday)), (2025, 9, 1));
        let monday = days_from_civil(2025, 9, 8);
        assert_eq!(week_start(monday), monday);
        let sunday = days_from_civil(2025, 9, 7);
        assert_eq!(civil_from_days(week_start(sunday)), (2025, 9, 1));
    }

    /// The header is the documented per-session id, so it wins outright and is
    /// used verbatim — no marker hunting in a value that is already the answer.
    #[test]
    fn the_session_header_is_the_session_key_when_it_is_there() {
        let mut rec = run(10, "serve:anthropic", "Qwen3.6-27B");
        rec.session = Some("9f2ca1b4-0d31-4e77-9a02-7c1f8b6e5d40".to_string());
        rec.client = Some("user_abc_session_ffffffff".to_string());
        assert_eq!(session_key(&rec), "9f2ca1b4-0d31-4e77-9a02-7c1f8b6e5d40");
        // An empty header is no header.
        rec.session = Some(String::new());
        assert_eq!(session_key(&rec), "ffffffff");
    }

    /// The body id has worn two shapes across Claude Code releases. Both put
    /// the session id straight after a `session_` marker, and a record that
    /// predates the header still has to group correctly.
    #[test]
    fn the_session_key_reads_both_shapes_of_the_body_id() {
        let mut rec = run(10, "serve:anthropic", "Qwen3.6-27B");

        rec.client =
            Some("user_1a2b3c_account_11111111-2222-3333-4444-555555555555_session_deadbeef-0000-1111-2222-333333333333".to_string());
        assert_eq!(session_key(&rec), "deadbeef-0000-1111-2222-333333333333");

        rec.client = Some(
            r#"{"user_id":"u1","session_id":"7c1f8b6e-5d40-4e77-9a02-0d3111112222"}"#.to_string(),
        );
        assert_eq!(session_key(&rec), "7c1f8b6e-5d40-4e77-9a02-0d3111112222");

        rec.client = Some("user_ab_account_x_session_11111111-2222".to_string());
        assert_eq!(session_key(&rec), "11111111-2222");

        // Keys named before the session one are passed over: the marker is
        // found from the END of the string.
        rec.client =
            Some(r#"{"device_id":"d","account_uuid":"a","session_id":"3333-4444"}"#.to_string());
        assert_eq!(session_key(&rec), "3333-4444");

        // The last marker wins: a nested blob that names a session twice is
        // reporting the outer one last.
        rec.client = Some("session_aaaa_session_bbbb".to_string());
        assert_eq!(session_key(&rec), "bbbb");
    }

    /// A run nobody claimed groups under a name of its own rather than
    /// vanishing, and a client id with no session in it is not a session.
    #[test]
    fn a_run_with_no_identity_groups_as_unattributed() {
        let mut rec = run(10, "generate", "Qwen3.6-27B");
        assert_eq!(session_key(&rec), UNATTRIBUTED);
        assert_eq!(GroupBy::Client.bucket(&rec, 0), UNATTRIBUTED);
        assert_eq!(GroupBy::Session.bucket(&rec, 0), UNATTRIBUTED);

        rec.client = Some("plain-api-key-user".to_string());
        assert_eq!(session_key(&rec), UNATTRIBUTED);
        assert_eq!(GroupBy::Client.bucket(&rec, 0), "plain-api-key-user");

        // A marker with nothing usable after it is not a session either.
        rec.client = Some("trailing_session_".to_string());
        assert_eq!(session_key(&rec), UNATTRIBUTED);
    }

    /// The raw ids run to a hundred characters, so both filters match on a
    /// substring: naming the recognizable part is the only usable way in.
    #[test]
    fn the_client_and_session_filters_match_on_a_substring() {
        let mut first = run(10, "serve:openai", "Qwen3.6-27B");
        first.client = Some("user_aaaa_session_1111-2222".to_string());
        first.decode_tokens = 10;
        let mut second = run(20, "serve:openai", "Qwen3.6-27B");
        second.client = Some("user_bbbb_session_3333-4444".to_string());
        second.decode_tokens = 20;
        let records = vec![first, second];

        let path = scratch("filters.jsonl");
        for rec in &records {
            append(&path, rec).expect("a record appends");
        }
        let asked = |query: StatsQuery| {
            report(&query)
                .expect("the history reads")
                .expect("a history")
        };

        // A fragment of the session id, not the whole thing.
        let by_session = asked(StatsQuery {
            by: GroupBy::All,
            session: Some("3333".to_string()),
            file: Some(path.clone()),
            ..StatsQuery::default()
        });
        assert_eq!(by_session.matched, 1);
        assert_eq!(by_session.rows[0].decode_tokens, 20);

        // A fragment of the raw body id, which the session key never contains.
        let by_client = asked(StatsQuery {
            by: GroupBy::All,
            client: Some("aaaa".to_string()),
            file: Some(path.clone()),
            ..StatsQuery::default()
        });
        assert_eq!(by_client.matched, 1);
        assert_eq!(by_client.rows[0].decode_tokens, 10);

        // A run with no client id at all is kept by neither filter.
        append(&path, &run(30, "generate", "Qwen3.6-27B")).expect("a record appends");
        let anonymous = asked(StatsQuery {
            by: GroupBy::All,
            client: Some("user_".to_string()),
            file: Some(path.clone()),
            ..StatsQuery::default()
        });
        assert_eq!(anonymous.records, 3);
        assert_eq!(anonymous.matched, 2);
        std::fs::remove_file(&path).expect("the history is removed");

        let sessions = aggregate(records.into_iter(), GroupBy::Session, 0);
        assert_eq!(
            sessions
                .iter()
                .map(|row| row.label.as_str())
                .collect::<Vec<_>>(),
            ["3333-4444", "1111-2222"],
            "heaviest session first"
        );
    }

    /// `--by agent` names the subagents that did the work, and a run no agent
    /// header named is its own row rather than being folded into a session.
    /// The same records under `--by session` are still one row: the agent id is
    /// a field of its own and never reaches the session key.
    #[test]
    fn grouping_by_agent_separates_the_subagents_from_the_rest() {
        let session = "9f2ca1b4-0d31-4e77-9a02-7c1f8b6e5d40";
        let mut records = Vec::new();
        for (agent, decode) in [
            (Some("explore-metrics"), 10),
            (Some("router-fixes"), 40),
            (Some("explore-metrics"), 5),
            (None, 20),
        ] {
            let mut rec = run(10, "serve:anthropic", "Qwen3.6-27B");
            rec.session = Some(session.to_string());
            rec.agent = agent.map(str::to_string);
            rec.decode_tokens = decode;
            records.push(rec);
        }

        let agents = aggregate(records.clone().into_iter(), GroupBy::Agent, 0);
        assert_eq!(
            agents
                .iter()
                .map(|row| (row.label.as_str(), row.runs, row.decode_tokens))
                .collect::<Vec<_>>(),
            [
                ("router-fixes", 1, 40),
                (UNATTRIBUTED, 1, 20),
                ("explore-metrics", 2, 15)
            ],
            "most decoding first, and the unattributed run a row of its own"
        );

        let sessions = aggregate(records.into_iter(), GroupBy::Session, 0);
        assert_eq!(sessions.len(), 1, "one session stays one row");
        assert_eq!(sessions[0].label, session);
        assert_eq!(sessions[0].runs, 4);
    }

    #[test]
    fn the_bucket_labels_read_as_dates() {
        let rec = run(1_757_030_400, "generate", "Qwen3.6-27B");
        assert_eq!(GroupBy::Day.bucket(&rec, 0), "2025-09-05");
        assert_eq!(GroupBy::Week.bucket(&rec, 0), "2025-09-01");
        assert_eq!(GroupBy::Month.bucket(&rec, 0), "2025-09");
        assert_eq!(GroupBy::Model.bucket(&rec, 0), "Qwen3.6-27B");
        assert_eq!(GroupBy::Surface.bucket(&rec, 0), "generate");
        assert_eq!(GroupBy::All.bucket(&rec, 0), "all");
    }

    #[test]
    fn since_reads_every_form_it_documents() {
        let now = 1_757_030_400;
        assert_eq!(parse_since("24h", now, 0).unwrap(), now - 86_400);
        assert_eq!(parse_since("7d", now, 0).unwrap(), now - 7 * 86_400);
        assert_eq!(parse_since("4w", now, 0).unwrap(), now - 28 * 86_400);
        // Local midnight, so a positive offset makes the boundary earlier in UTC.
        assert_eq!(parse_since("2025-09-05", now, 0).unwrap(), now);
        assert_eq!(
            parse_since("2025-09-05", now, 2 * 3600).unwrap(),
            now - 2 * 3600
        );
        assert!(parse_since("yesterday", now, 0).is_err());
        assert!(parse_since("7y", now, 0).is_err());
        assert!(parse_since("2025-13-01", now, 0).is_err());
    }

    /// A relative window never runs off the start of the epoch.
    #[test]
    fn a_since_window_longer_than_the_clock_clamps_to_zero() {
        assert_eq!(parse_since("52w", 1_000, 0).unwrap(), 0);
    }

    fn sample() -> Vec<RunRecord> {
        let mut first = run(1_757_030_400, "generate", "Qwen3.6-27B");
        first.prompt_tokens = 1_000;
        first.cached_tokens = 200;
        first.prefill_tokens = 800;
        first.prefill_secs = 2.0;
        first.decode_tokens = 100;
        first.decode_secs = 4.0;
        first.drafted = Some(80);
        first.accepted = Some(60);

        let mut second = run(1_757_030_400 + 3_600, "chat", "Qwen3.6-27B");
        second.prompt_tokens = 500;
        second.prefill_tokens = 500;
        second.prefill_secs = 2.0;
        second.decode_tokens = 300;
        second.decode_secs = 6.0;

        let mut third = run(1_757_030_400 + 86_400, "chat", "Qwen3.8-27B");
        third.prompt_tokens = 200;
        third.prefill_tokens = 200;
        third.prefill_secs = 1.0;
        third.decode_tokens = 50;
        third.decode_secs = 1.0;

        vec![first, second, third]
    }

    /// Rates are the bucket's tokens over the bucket's seconds. A mean of the
    /// two runs' own rates would be 37.5 tok/s here rather than 40.
    #[test]
    fn a_bucket_sums_tokens_and_divides_the_sums() {
        let rows = aggregate(sample().into_iter(), GroupBy::Day, 0);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].label, "2025-09-05");
        assert_eq!(rows[0].runs, 2);
        assert_eq!(rows[0].prompt_tokens, 1_500);
        assert_eq!(rows[0].decode_tokens, 400);
        assert_eq!(rows[0].decode_secs, 10.0);
        assert_eq!(rows[0].decode_tps(), Some(40.0));
        assert_eq!(rows[0].prefill_tps(), Some(325.0));
        // 200 of 1500 prompt tokens came out of a cache.
        assert!((rows[0].hit_rate().unwrap() - 0.133_333).abs() < 1e-5);
        assert_eq!(rows[0].acceptance(), Some(0.75));
        assert_eq!(rows[1].label, "2025-09-06");
    }

    /// A bucket nothing drafted in, and one nothing was timed in, report a
    /// dash instead of a number they cannot have.
    #[test]
    fn an_unmeasured_bucket_reports_no_rate() {
        let rows = aggregate(sample().into_iter(), GroupBy::Surface, 0);
        let chat = rows.iter().find(|row| row.label == "chat").expect("chat");
        assert_eq!(chat.acceptance(), None);
        assert_eq!(percent(chat.acceptance()), "-");
        let empty = Bucket::default();
        assert_eq!(empty.decode_tps(), None);
        assert_eq!(empty.hit_rate(), None);
        assert_eq!(rate(empty.prefill_tps(), 0), "-");
    }

    /// Periods read in time order; names read heaviest first, because that is
    /// the question "which model did I actually run" asks.
    #[test]
    fn name_buckets_order_by_the_decoding_they_did() {
        let rows = aggregate(sample().into_iter(), GroupBy::Model, 0);
        assert_eq!(
            rows.iter()
                .map(|row| row.label.as_str())
                .collect::<Vec<_>>(),
            ["Qwen3.6-27B", "Qwen3.8-27B"]
        );
        assert_eq!(rows[0].decode_tokens, 400);

        let surfaces = aggregate(sample().into_iter(), GroupBy::Surface, 0);
        assert_eq!(
            surfaces
                .iter()
                .map(|row| row.label.as_str())
                .collect::<Vec<_>>(),
            ["chat", "generate"]
        );

        let all = aggregate(sample().into_iter(), GroupBy::All, 0);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].runs, 3);
    }

    /// A history with a torn line still reads: the record before it and the
    /// one after it both survive, and the damage is counted rather than hidden.
    #[test]
    fn a_malformed_line_is_skipped_and_counted() {
        let path = scratch("torn.jsonl");
        for rec in sample() {
            append(&path, &rec).expect("a record appends");
        }
        {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .expect("the history opens");
            writeln!(file, "{{\"v\":1,\"ts\":1,\"surf").expect("a torn line writes");
            writeln!(file).expect("a blank line writes");
        }
        let history = load(&path).expect("the history reads").expect("a history");
        assert_eq!(history.records.len(), 3);
        assert_eq!(history.skipped, 1, "the blank line is not damage");
        std::fs::remove_file(&path).expect("the history is removed");

        assert!(
            load(&scratch("absent.jsonl"))
                .expect("a missing history is not an error")
                .is_none()
        );
    }

    /// A process killed mid-write leaves a partial line, and a partial line can
    /// cut a multibyte character in half. Decoding the file as one string would
    /// lose every record in it; decoding line by line loses only the torn one.
    #[test]
    fn a_torn_multibyte_line_costs_only_itself() {
        let path = scratch("multibyte.jsonl");
        let mut bytes: Vec<u8> = Vec::new();
        let mut first = run(1_757_030_400, "chat", "Qwen3.6-27B");
        first.decode_tokens = 11;
        bytes.extend_from_slice(serde_json::to_string(&first).unwrap().as_bytes());
        bytes.push(b"\n"[0]);
        // The lead byte of a two-byte character, and then the line ends.
        bytes.extend_from_slice(b"{\"model\":\"caf\xc3");
        bytes.push(b"\n"[0]);
        let mut third = run(1_757_030_400, "chat", "Qwen3.6-27B");
        third.decode_tokens = 22;
        bytes.extend_from_slice(serde_json::to_string(&third).unwrap().as_bytes());
        bytes.push(b"\n"[0]);
        std::fs::write(&path, &bytes).expect("the history writes");

        let history = load(&path).expect("the history reads").expect("a history");
        assert_eq!(
            history.records.len(),
            2,
            "the records on either side of the torn line survive"
        );
        assert_eq!(history.skipped, 1);
        assert_eq!(history.records[0].decode_tokens, 11);
        assert_eq!(history.records[1].decode_tokens, 22);
        std::fs::remove_file(&path).expect("the history is removed");
    }

    /// `days_from_civil` is arithmetic and will happily read a 31st of February
    /// as the 3rd of March, so the day is checked against its month first.
    /// A bad argument is a bad argument whether or not any runs have been
    /// recorded: validating after the history check would answer a typo with
    /// "no metrics recorded yet", which is a different question.
    #[test]
    fn a_bad_since_is_refused_before_the_history_is_read() {
        let query = |file: PathBuf| StatsQuery {
            since: Some("yesterday".to_string()),
            file: Some(file),
            ..StatsQuery::default()
        };
        assert!(
            report(&query(scratch("never-written.jsonl"))).is_err(),
            "a missing history must not mask the argument error"
        );

        let path = scratch("since-order.jsonl");
        append(&path, &run(1_757_030_400, "chat", "Qwen3.6-27B")).expect("a record appends");
        assert!(report(&query(path.clone())).is_err());
        // The same query with a good `--since` reads the history it was
        // refusing to reach.
        let good = StatsQuery {
            since: Some("2025-09-05".to_string()),
            file: Some(path.clone()),
            ..StatsQuery::default()
        };
        assert!(report(&good).expect("the history reads").is_some());
        std::fs::remove_file(&path).expect("the history is removed");
    }

    #[test]
    fn since_refuses_a_day_its_month_does_not_have() {
        let now = 1_757_030_400;
        assert!(parse_since("2026-02-31", now, 0).is_err());
        assert!(parse_since("2026-04-31", now, 0).is_err());
        assert!(parse_since("2026-00-10", now, 0).is_err());
        assert!(parse_since("2026-01-00", now, 0).is_err());

        // 2024 is a leap year, 2026 is not, 2000 is, 1900 is not.
        assert!(parse_since("2024-02-29", now, 0).is_ok());
        assert!(parse_since("2026-02-29", now, 0).is_err());
        assert!(parse_since("2000-02-29", now, 0).is_ok());
        assert!(parse_since("1900-02-29", now, 0).is_err());
        assert!(parse_since("2026-12-31", now, 0).is_ok());
    }

    #[test]
    fn the_table_aligns_its_columns_and_sums_them() {
        let rows = aggregate(sample().into_iter(), GroupBy::Day, 0);
        let table = render_table(&rows, GroupBy::Day);
        let lines: Vec<&str> = table.lines().collect();
        assert!(lines[0].starts_with("period"));
        assert!(lines[0].contains("dec tok/s"));
        assert_eq!(lines.len(), 5, "header, two buckets, a rule, the total");
        let widths: Vec<usize> = lines.iter().copied().map(UnicodeWidthStr::width).collect();
        assert!(
            widths.windows(2).all(|pair| pair[0] == pair[1]),
            "every line ends at the same display column: {widths:?}"
        );
        assert!(lines[4].starts_with("total"));
        assert!(
            lines[4].contains("450"),
            "450 decoded tokens across the run"
        );
        // Every column ENDS at the same offset on every line, which is what
        // right-aligned numbers under a right-aligned header means. Taking the
        // header's own column ends and checking each row breaks there is what
        // makes this fail if a cell is ever padded to the wrong width.
        let ends: Vec<usize> = ["period", "runs", "prompt", "cached", "hit%"]
            .iter()
            .map(|name| {
                let at = lines[0].find(name).expect("a named column");
                at + name.chars().count()
            })
            .collect();
        // The label column is left-aligned, so only the number columns are
        // checked against the header's right edge.
        for line in &lines[1..] {
            for end in &ends[1..] {
                assert!(
                    line.chars().count() >= *end,
                    "row is too short to reach column end {end}: {line:?}"
                );
                let cell_end = line.chars().nth(end - 1).expect("a character");
                assert!(
                    cell_end != ' ',
                    "column ending at {end} is not flush right in {line:?}"
                );
                if let Some(after) = line.chars().nth(*end) {
                    assert_eq!(
                        after, ' ',
                        "column ending at {end} runs into the next in {line:?}"
                    );
                }
            }
        }
    }

    /// A label is whatever the world put in it, and a CJK or emoji one is two
    /// display columns per character. Padding by character count would leave
    /// every number in that row shifted, so the widths are measured in columns.
    ///
    /// The last column is right-aligned, so no line is ever trimmed and every
    /// one must end at the same display column.
    #[test]
    fn wide_characters_do_not_shift_the_columns() {
        let row = |label: &str, decode: u64| Bucket {
            label: label.to_string(),
            runs: 1,
            decode_tokens: decode,
            decode_secs: 1.0,
            ..Bucket::default()
        };
        let rows = vec![
            row("Qwen3.6-27B", 100),
            row("日本語のモデル", 90),
            row("robot-\u{1f916}-drafter", 80),
            row("plain", 70),
        ];
        let table = render_table(&rows, GroupBy::Model);
        let widths: Vec<usize> = table.lines().map(UnicodeWidthStr::width).collect();
        assert!(
            widths.windows(2).all(|pair| pair[0] == pair[1]),
            "every line ends at the same display column: {widths:?}\n{table}"
        );
    }

    /// One long label must not push the numbers off the right of a terminal, so
    /// the column is capped and what is cut is marked. The cut is by display
    /// column, and a double-width character that would straddle the boundary is
    /// dropped whole rather than split.
    #[test]
    fn an_overlong_label_is_truncated_to_the_column() {
        assert_eq!(truncate_to_width("short", 48), "short");
        // Exactly at the limit is not truncated.
        let exact = "x".repeat(LABEL_MAX_WIDTH);
        assert_eq!(truncate_to_width(&exact, LABEL_MAX_WIDTH), exact);

        let long = "x".repeat(LABEL_MAX_WIDTH + 20);
        let cut = truncate_to_width(&long, LABEL_MAX_WIDTH);
        assert_eq!(cut.width(), LABEL_MAX_WIDTH);
        assert!(cut.ends_with('\u{2026}'));

        // Ten double-width characters are twenty columns; a five-column budget
        // holds two of them plus the ellipsis.
        let wide = "日".repeat(10);
        let cut = truncate_to_width(&wide, 5);
        assert_eq!(cut, "日日\u{2026}");
        assert_eq!(cut.width(), 5);

        // And the cap is applied by the renderer, not just available to it.
        let rows = vec![Bucket {
            label: "y".repeat(200),
            runs: 1,
            ..Bucket::default()
        }];
        let table = render_table(&rows, GroupBy::Client);
        let first = table.lines().nth(1).expect("a bucket row");
        assert!(first.starts_with(&"y".repeat(LABEL_MAX_WIDTH - 1)));
        assert!(
            table.lines().map(UnicodeWidthStr::width).max().unwrap() < 140,
            "a 200-character label must not widen the whole table:\n{table}"
        );
    }

    #[test]
    fn integers_read_with_thousands_separators() {
        assert_eq!(commas(0), "0");
        assert_eq!(commas(999), "999");
        assert_eq!(commas(1_000), "1,000");
        assert_eq!(commas(34_112), "34,112");
        assert_eq!(commas(1_234_567), "1,234,567");
    }

    /// The rows sum every run, finished or not, so a consumer that wants only
    /// the completed ones needs to be told how many of each a row holds. The
    /// count rides the JSON rows and is deliberately not a table column.
    #[test]
    fn a_row_counts_the_runs_that_did_not_finish() {
        let mut cancelled = run(1_757_030_400, "chat", "Qwen3.6-27B");
        cancelled.ok = false;
        cancelled.decode_tokens = 7;
        let mut failed = run(1_757_030_400, "serve:openai", "Qwen3.6-27B");
        failed.ok = false;
        let finished = run(1_757_030_400, "generate", "Qwen3.6-27B");

        let rows = aggregate(
            vec![cancelled, failed, finished].into_iter(),
            GroupBy::All,
            0,
        );
        assert_eq!(rows[0].runs, 3);
        assert_eq!(rows[0].unfinished, 2);
        assert_eq!(
            rows[0].decode_tokens, 7,
            "an unfinished run's tokens still count toward the row"
        );

        // The total row folds the counts like every other sum.
        assert_eq!(Bucket::fold(&rows, "total").unfinished, 2);
        assert_eq!(rows_json(&rows)[0]["unfinished"], 2);

        // A bucket of nothing but completed runs says so.
        let clean = aggregate(sample().into_iter(), GroupBy::All, 0);
        assert_eq!(clean[0].unfinished, 0);
    }

    #[test]
    fn the_json_rows_carry_the_sums_the_table_derives_from() {
        let rows = aggregate(sample().into_iter(), GroupBy::Day, 0);
        let json = rows_json(&rows);
        let first = &json[0];
        assert_eq!(first["label"], "2025-09-05");
        assert_eq!(first["decode_tokens"], 400);
        assert_eq!(first["decode_tps"], 40.0);
        let second = &json[1];
        assert!(second["acceptance"].is_null(), "nothing drafted that day");
    }

    /// `off` turns recording off outright; any other value names the file.
    #[test]
    fn the_environment_names_the_file_or_turns_recording_off() {
        let env = |value: &str| Some(OsStr::new(value).to_owned());
        let home = Some(OsStr::new("/home/someone").to_owned());

        // `off` disables recording in any casing, and does so even where a
        // home directory would have resolved.
        for spelling in ["off", "OFF", "Off"] {
            assert_eq!(
                metrics_path_from(env(spelling).as_deref(), home.as_deref()),
                None,
                "{spelling} turns recording off"
            );
        }
        // A value that merely starts with it is a file name, not the switch.
        assert_eq!(
            metrics_path_from(env("offline.jsonl").as_deref(), home.as_deref()),
            Some(PathBuf::from("offline.jsonl"))
        );
        // Any other value names the file outright.
        assert_eq!(
            metrics_path_from(env("/tmp/elsewhere.jsonl").as_deref(), home.as_deref()),
            Some(PathBuf::from("/tmp/elsewhere.jsonl"))
        );
        // Nothing said, or nothing but an empty string, falls back to the
        // state directory under HOME.
        for unset in [None, env("")] {
            assert_eq!(
                metrics_path_from(unset.as_deref(), home.as_deref()),
                Some(PathBuf::from("/home/someone").join(METRICS_RELATIVE_PATH))
            );
        }
        // No home to resolve under and no file named leaves recording off
        // rather than guessing at a writable path.
        assert_eq!(metrics_path_from(None, None), None);
        assert_eq!(
            metrics_path_from(env("/tmp/named.jsonl").as_deref(), None),
            Some(PathBuf::from("/tmp/named.jsonl"))
        );
    }
}
