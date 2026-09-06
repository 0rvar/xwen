//! Serve configuration: the TOML file schema, the CLI-over-file merge, and the
//! `--init` template writer.
//!
//! Precedence is CLI flag > config file > built-in default. Whenever a flag
//! replaces a value the config file set explicitly, the merge records a warning
//! naming the flag, both values and the file it came from, so a forgotten flag
//! never silently contradicts a checked-in config.

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

use crate::chat::ReasoningEffort;

/// Loopback only: the server has no auth unless `api_key` is set.
pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 5241;
/// The checkpoint's own trained window: 256k tokens. A CEILING, not an
/// allocation — the KV cache starts small and grows on demand as a
/// conversation actually lengthens (`XwenModel`'s lazy KV), and an idle unload
/// drops the grown buffers, so the next load starts small again.
/// `resolve_context_length` clamps it for any checkpoint converted smaller, so
/// this asks for everything the model can hold and no more.
pub const DEFAULT_CONTEXT_LENGTH: usize = 262144;
pub const DEFAULT_IDLE_UNLOAD: IdleUnload = IdleUnload(Some(Duration::from_secs(300)));
pub const DEFAULT_ANTHROPIC: bool = true;
pub const DEFAULT_OPENAI: bool = true;
/// The dashboard is the default: serving is what a bare `xwen` does, and the
/// dashboard is what makes a minutes-long silent prefill legible. Headless
/// workflows lose nothing — when stderr is not a terminal (a `2>` redirect, a
/// pipe, CI) the sink steps aside with a one-line notice and writes the same
/// plain lines, and `--no-tui` / `tui = false` forces those lines on a
/// terminal too.
pub const DEFAULT_TUI: bool = true;
pub const DEFAULT_THINKING_FORCE: bool = true;
/// Thinking-token budget for requests that carry no thinking config of their
/// own. Capped by default because the checkpoint is prone to open-ended
/// reasoning loops, which at low-power decode rates can outlast an agent
/// client's request timeout entirely (Claude Code aborts at 600s and
/// retries — a doomed-generation loop the server never sees). The graduated
/// `--max-think` schedule steers the model out rather than truncating, so a
/// cap degrades gracefully. 0 = uncapped; a request's own
/// `thinking.budget_tokens` / `reasoning_effort` always wins.
pub const DEFAULT_THINKING_BUDGET: usize = 4096;
pub const DEFAULT_CACHE_SNAPSHOTS: usize = 4;
/// Conversations kept warm at once. One of them occupies the GPU cache; the rest
/// are host-RAM images the engine pages back in when their conversation returns.
/// 1 reproduces the single-sequence behaviour where a switch evicts whoever was
/// speaking. Two by default (2026-08-30, was four): the default checkpoint is
/// Flash-Next, whose images cost 30 KiB/token plus a 113 MiB DeltaNet floor —
/// ~8 GB per conversation at its 262144 context — and one host image beside
/// the live conversation covers the usual two-agents case.
pub const DEFAULT_CACHE_SLOTS: usize = 2;
/// The on-disk prefix cache is opt-in (`--disk-cache` / `disk_cache = true`;
/// was on by default until 2026-08-30): a restart resuming without re-prefill
/// is worth having, but at Flash-Next image sizes it writes gigabytes per
/// conversation, which is not something a default should do unasked.
pub const DEFAULT_DISK_CACHE: bool = false;
/// Ceiling on everything under `<cache_dir>/kv/`, across every checkpoint, in
/// GiB. Enforced by deleting the least recently used image.
pub const DEFAULT_DISK_MAX_GIB: u64 = 64;
/// Conversations shorter than this are not worth an image: the fixed cost is one
/// snapshot per stored position band (see `hub::Model::snapshot_bytes` — 62.8 MiB
/// on the 35B-A3B, 149.6 MiB on the 27B, whatever position it covers) while the
/// prefill an image saves scales with the token count.
pub const DEFAULT_DISK_MIN_TOKENS: usize = 1024;
/// Where the disk tier lives when neither the config nor a flag names a
/// directory, relative to `$HOME`.
const CACHE_RELATIVE_PATH: &str = ".cache/xwen";
/// Tools are served: the definitions are rendered into the prompt and calls are
/// parsed back out of the model's output. The other two modes predate that and
/// are kept for debugging.
pub const DEFAULT_TOOLS_MODE: ToolsMode = ToolsMode::Native;
/// Requests that may wait behind the one running generation. Generous rather
/// than small: a queued streaming request costs almost nothing (keep-alive
/// frames hold the connection), while a refusal sends the client into its
/// retry-or-give-up policy.
pub const DEFAULT_QUEUE_CAPACITY: usize = 16;
/// Floor on the seconds a request may wait in the queue before it is dropped
/// with an error instead of served stale. What ships is
/// [`default_queue_timeout_secs`] over the configured context length; this is
/// the value that formula can never go below, and the value a server whose
/// context is short enough gets outright.
pub const DEFAULT_QUEUE_TIMEOUT_SECS: u64 = 300;

/// The slowest prefill this machine has been measured at, in tokens per second,
/// at the longest prompt anyone has run through it.
///
/// Measured 2026-09-06 on the long-context sweep (docs/perf-state.md, "Long
/// context"): Qwen3.8-Flash-Next reads 232 tok/s at a 131072-token prompt, the
/// slowest of the checkpoints benchmarked there and the DEFAULT one, where the
/// 35B-A3B reads 668. Rounded down to 200, because prefill cost per token keeps
/// climbing past the length that was measured and a server may be configured
/// well past it. Deliberately a floor rather than a typical rate: it exists to
/// bound a wait, and being pessimistic here costs a queued client nothing while
/// being optimistic drops it.
const SLOWEST_PREFILL_TOKENS_PER_SEC: u64 = 200;

/// Seconds a queued request may wait, derived from what the server could be busy
/// with when it arrives.
///
/// The old flat 300 was sized against nothing in particular and is well under one
/// maximal prefill on this machine: a 131072-token prompt is 567 s of prefill on
/// the default checkpoint before its own decode starts, so a request arriving
/// behind one was dropped for saturation while the server worked normally. The
/// derived value covers TWO such prefills — the one running plus one already
/// queued ahead — which is the shape of the case the timeout is there to
/// survive.
///
/// The explicit `queue_timeout` key still wins; this only moves the default.
pub fn default_queue_timeout_secs(context_length: usize) -> u64 {
    let one_prefill = context_length as u64 / SLOWEST_PREFILL_TOKENS_PER_SEC;
    // Rounded to a whole minute, UP rather than down: a default that reads
    // `1080` says "eighteen minutes" where `1048` says nothing, and rounding the
    // other way would put the value back under the two prefills it exists to
    // cover (at 131072 the pair is 524 s, and rounding down to five minutes
    // would land on 300 — exactly the flat value this replaced).
    let derived = (2 * one_prefill).div_ceil(60) * 60;
    derived.max(DEFAULT_QUEUE_TIMEOUT_SECS)
}
/// Watchdog throughput floors (tokens/second) for a job's wall-clock ceiling,
/// deliberately ~2x below measured low-power-mode throughput so the ceiling
/// only ever catches a wedged generation, never a slow one. These are not
/// performance targets; at full power they are ~4x loose.
pub const DEFAULT_REQUEST_PREFILL_RATE: u64 = 150;
pub const DEFAULT_REQUEST_DECODE_RATE: u64 = 10;
/// Seconds of fixed allowance on top of the throughput terms: model load,
/// scheduling, paging.
pub const DEFAULT_REQUEST_SLACK_SECS: u64 = 30;
/// Cheapest-prefill-first scheduling; `fifo` is the kill switch that restores
/// strict arrival order.
pub const DEFAULT_SCHEDULE: Schedule = Schedule::ShortestPrefill;
/// Seconds a queued request may be passed over before it wins on age alone.
pub const DEFAULT_SCHEDULE_AGE_LIMIT_SECS: u64 = 20;
/// Speculative-decode defaults, matching `xwen generate`'s `--draft-*` flags so
/// a conversation speculates here exactly as it does on the command line.
///
/// Speculation is opt-OUT. The K-snapshot fused verify (P9a) removed the
/// per-token cache-sync cost that made a drafter a loss on the 35B-A3B, and
/// with it drafting measured faster on both checkpoints (2026-07-29, greedy,
/// 128 tokens, warm, interleaved): 27B +19.3 to +21.0% on code and +7.6 to
/// +8.4% on chat, 35B-A3B +18.1 to +19.8% on code and +12.6 to +12.8% on chat.
/// `--no-draft`, or `enabled = false` in the config, opts out. Note the cost of
/// the default: a zero-flag run now fetches and loads the DFlash sidecar (3.5
/// GB on the 27B, 0.8 GB on the 35B-A3B). See docs/decisions.md, "Speculative
/// decoding".
///
/// `p_min` is the one drafter default that is not a constant here: it is fitted
/// per checkpoint and lives on [`crate::hub::Model::draft_p_min_default`]. The
/// merge leaves it unresolved (`None`) because one server loads whichever
/// checkpoint a request names; the engine resolves it when it attaches the
/// drafter, once the checkpoint is known. `draft.p_min` in the config file, and
/// `--draft-p-min`, pin one floor for every checkpoint instead.
pub const DEFAULT_DRAFT_ENABLED: bool = true;
pub const DEFAULT_DRAFT_MAX: usize = 15;
pub const DEFAULT_DRAFT_PAUSE_MARGIN: f32 = 1.0;
/// Positions the drafter's KV cache is sized for, which doubles as the horizon
/// speculation stays active over — past it decode continues plain. Shared with
/// the CLI (`--draft-ctx`); see the constant's doc for why it is small.
pub use crate::dflash::DEFAULT_DRAFT_CTX;

/// Which checkpoint the `--init` template quotes its cache sizes for. The
/// template is written before any model is chosen, so it has to name one; the
/// bring-up model is the one most of these numbers will be read against.
const TEMPLATE_MODEL: crate::hub::Model = crate::hub::Model::Qwen35BA3B;

/// KV bytes per token of drafter cache for the model the template quotes. The
/// figure is per-model — the 27B sidecar has five layers to the 35B-A3B's six —
/// and lives on `hub::Model` beside the target's. It informs a `--init` comment
/// and nothing that allocates.
const DRAFT_KV_BYTES_PER_TOKEN: usize = match TEMPLATE_MODEL.draft_kv_bytes_per_token() {
    Some(bytes) => bytes,
    None => panic!("the template model ships a drafter, so its cache has a size"),
};

/// Config path used when `--config` is not given.
pub const CONFIG_RELATIVE_PATH: &str = ".config/xwen/serve.toml";

/// How long the server may sit idle before dropping the model, or `None` for
/// "stay loaded forever".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdleUnload(pub Option<Duration>);

impl fmt::Display for IdleUnload {
    /// Renders back into the config file's own syntax, so warning messages
    /// quote values the user can paste.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(d) = self.0 else {
            return write!(f, "off");
        };
        let secs = d.as_secs();
        if secs % 3600 == 0 {
            write!(f, "{}h", secs / 3600)
        } else if secs % 60 == 0 {
            write!(f, "{}m", secs / 60)
        } else {
            write!(f, "{secs}s")
        }
    }
}

/// What the server does with a request that carries tool definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolsMode {
    /// Render the definitions into the prompt and parse the calls back out of
    /// the model's output.
    Native,
    /// Refuse the request. Nothing is lost silently.
    Reject,
    /// Drop the definitions and answer the request as if it had none, so a
    /// harness that always sends tools can still hold a conversation.
    Strip,
}

impl fmt::Display for ToolsMode {
    /// Renders back into the config file's own syntax, so warning messages
    /// quote values the user can paste.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ToolsMode::Native => "native",
            ToolsMode::Reject => "reject",
            ToolsMode::Strip => "strip",
        })
    }
}

/// How the engine picks the next job off the queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Schedule {
    /// Run the job with the least prefill actually required — prompt tokens
    /// minus what the KV cache already holds for it — so a short side request
    /// does not wait behind a long cold prompt. An age limit bounds starvation.
    ShortestPrefill,
    /// Strict arrival order, the pre-scheduler behaviour.
    Fifo,
}

impl fmt::Display for Schedule {
    /// Renders back into the config file's own syntax, so warning messages
    /// quote values the user can paste.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Schedule::ShortestPrefill => "shortest-prefill",
            Schedule::Fifo => "fifo",
        })
    }
}

/// One of `shortest-prefill` or `fifo`; anything else names both in the error.
pub fn parse_schedule(text: &str) -> Result<Schedule> {
    match text.trim() {
        "shortest-prefill" => Ok(Schedule::ShortestPrefill),
        "fifo" => Ok(Schedule::Fifo),
        other => {
            bail!("invalid schedule {other:?}: expected \"shortest-prefill\" or \"fifo\"")
        }
    }
}

/// One of `native`, `reject` or `strip`; anything else names all three in the
/// error.
pub fn parse_tools_mode(text: &str) -> Result<ToolsMode> {
    match text.trim() {
        "native" => Ok(ToolsMode::Native),
        "reject" => Ok(ToolsMode::Reject),
        "strip" => Ok(ToolsMode::Strip),
        other => {
            bail!("invalid tools mode {other:?}: expected \"native\", \"reject\" or \"strip\"")
        }
    }
}

/// An integer with an `s`/`m`/`h` suffix, or `off`.
pub fn parse_idle_unload(text: &str) -> Result<IdleUnload> {
    let text = text.trim();
    if text.eq_ignore_ascii_case("off") {
        return Ok(IdleUnload(None));
    }
    // Split off the last character rather than the last byte: a stray
    // multi-byte suffix must produce the error below, not a panic.
    let mut chars = text.chars();
    let unit = chars.next_back();
    let digits = chars.as_str();
    let secs_per = match unit {
        Some('s') => 1,
        Some('m') => 60,
        Some('h') => 3600,
        _ => bail!(
            "invalid duration {text:?}: expected an integer with an s/m/h suffix (\"30s\", \"5m\", \"1h\") or \"off\""
        ),
    };
    let count: u64 = digits
        .parse()
        .with_context(|| format!("invalid duration {text:?}: {digits:?} is not a whole number"))?;
    if count == 0 {
        bail!("invalid duration {text:?}: use \"off\" to keep the model loaded forever");
    }
    // Checked, because the product wraps in a release build: "18446744073709551h"
    // would otherwise resolve to a handful of seconds and unload a model the
    // operator asked to keep forever.
    let Some(secs) = count.checked_mul(secs_per) else {
        bail!("invalid duration {text:?}: longer than this server can measure; use \"off\"");
    };
    Ok(IdleUnload(Some(Duration::from_secs(secs))))
}

/// The config file. Every field is optional so the merge can tell "the user set
/// this" from "the user left it alone"; unknown keys are rejected so a typo
/// fails loudly instead of silently doing nothing.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServeToml {
    pub model: Option<PathBuf>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub context_length: Option<usize>,
    pub idle_unload: Option<String>,
    pub anthropic: Option<bool>,
    pub openai: Option<bool>,
    pub api_key: Option<String>,
    /// Raw text; parsed during the merge so the error names this file.
    pub tools_mode: Option<String>,
    pub queue_capacity: Option<usize>,
    /// Seconds.
    pub queue_timeout: Option<u64>,
    /// Tokens per second; 0 disables the deadline.
    pub request_prefill_rate: Option<u64>,
    /// Tokens per second; 0 disables the deadline.
    pub request_decode_rate: Option<u64>,
    /// Seconds.
    pub request_slack: Option<u64>,
    /// Raw text; parsed during the merge so the error names this file.
    pub schedule: Option<String>,
    /// Seconds.
    pub schedule_age_limit: Option<u64>,
    pub tui: Option<bool>,
    /// Where the on-disk prefix cache keeps its images. Absent falls back to
    /// `$HOME/.cache/xwen`.
    pub cache_dir: Option<PathBuf>,
    pub disk_cache: Option<bool>,
    pub disk_max_gib: Option<u64>,
    pub disk_min_tokens: Option<usize>,
    pub thinking: ThinkingToml,
    pub sampling: SamplingToml,
    pub cache: CacheToml,
    pub draft: DraftToml,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ThinkingToml {
    pub force: Option<bool>,
    pub default_budget: Option<usize>,
    /// Raw text; parsed during the merge so the error names this file.
    pub effort: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SamplingToml {
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub presence_penalty: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CacheToml {
    pub snapshots: Option<usize>,
    pub slots: Option<usize>,
}

/// The symbolic drafter path meaning "each checkpoint's own official sidecar" —
/// what `--draft official` and `path = "official"` spell, and the name the CLI
/// documents as the default.
pub const OFFICIAL_DRAFTER: &str = "official";

/// How this server speculates, as the merge resolves it.
///
/// A mode rather than a path because one server loads whichever checkpoint a
/// request names, and "the official sidecar" is a different file for each of
/// them — the checkpoint has to be known before the drafter can be. Collapsing
/// this to one resolved path is what made a server whose DEFAULT checkpoint
/// ships no sidecar run every OTHER checkpoint plain as well, silently.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DraftMode {
    /// `--no-draft`, or `enabled = false`: nothing speculates, whatever is
    /// loaded.
    Off,
    /// The default: each checkpoint drafts with its own official sidecar,
    /// resolved (and fetched) when that checkpoint loads. One that ships none
    /// decodes plain, with a line saying so.
    #[default]
    Official,
    /// A custom drafter GGUF. It belongs to the checkpoint it was validated
    /// against — the default one — and never transfers, so any other checkpoint
    /// this server loads speculates with its own official sidecar instead.
    Custom(PathBuf),
}

impl DraftMode {
    /// Whether this server speculates at all. Not whether the checkpoint that
    /// is loaded right now does: that also depends on whether it ships a
    /// sidecar, which only the engine knows.
    pub fn is_on(&self) -> bool {
        !matches!(self, DraftMode::Off)
    }

    /// The custom drafter GGUF, when one was named.
    pub fn custom_path(&self) -> Option<&Path> {
        match self {
            DraftMode::Custom(path) => Some(path),
            _ => None,
        }
    }
}

/// Speculative decoding with the checkpoint's drafter. Opt-out: on unless
/// `enabled = false`, which decodes one token per target forward even when a
/// `path` is set. `path` names a custom drafter; without it each checkpoint's
/// own official sidecar is used.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DraftToml {
    pub enabled: Option<bool>,
    pub path: Option<PathBuf>,
    pub max: Option<usize>,
    pub p_min: Option<f32>,
    pub pause_margin: Option<f32>,
    pub ctx: Option<usize>,
}

/// The flags `xwen serve` accepts, lifted out of clap so the merge is
/// testable without building a command line. `None` means "not passed".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CliOverrides {
    pub model: Option<PathBuf>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub context_length: Option<usize>,
    /// Raw text; parsed during the merge so errors name the flag.
    pub idle_unload: Option<String>,
    pub anthropic: Option<bool>,
    pub openai: Option<bool>,
    pub api_key: Option<String>,
    /// Raw text; parsed during the merge so errors name the flag.
    pub tools_mode: Option<String>,
    pub queue_capacity: Option<usize>,
    /// Seconds.
    pub queue_timeout: Option<u64>,
    pub request_prefill_rate: Option<u64>,
    pub request_decode_rate: Option<u64>,
    /// Seconds.
    pub request_slack: Option<u64>,
    /// Raw text; parsed during the merge so errors name the flag.
    pub schedule: Option<String>,
    /// Seconds.
    pub schedule_age_limit: Option<u64>,
    pub tui: Option<bool>,
    pub cache_dir: Option<PathBuf>,
    pub disk_cache: Option<bool>,
    pub disk_max_gib: Option<u64>,
    pub disk_min_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub presence_penalty: Option<f64>,
    /// Raw text; parsed during the merge so errors name the flag.
    pub reasoning_effort: Option<String>,
    pub cache_snapshots: Option<usize>,
    pub cache_slots: Option<usize>,
    pub draft: Option<PathBuf>,
    /// `--no-draft`: present means an explicit `false`.
    pub draft_enabled: Option<bool>,
    pub draft_max: Option<usize>,
    pub draft_p_min: Option<f32>,
    pub draft_pause_margin: Option<f32>,
    pub draft_ctx: Option<usize>,
}

/// Fully resolved configuration handed to [`super::run`].
#[derive(Debug, Clone, PartialEq)]
pub struct ServeSettings {
    pub model: PathBuf,
    pub host: String,
    pub port: u16,
    pub context_length: usize,
    /// `None` keeps the model loaded forever.
    pub idle_unload: Option<Duration>,
    pub anthropic: bool,
    pub openai: bool,
    /// `None` accepts any credentials, including none.
    pub api_key: Option<String>,
    /// What a request carrying tool definitions gets.
    pub tools_mode: ToolsMode,
    /// Jobs that may sit queued behind the one running generation, at least 1.
    pub queue_capacity: usize,
    /// How long a job may wait in the queue before it is dropped with an error.
    pub queue_timeout: Duration,
    /// Watchdog prefill floor (tokens/second) for the per-job wall-clock
    /// ceiling. 0 makes that term unbounded, which — the ceiling being one
    /// instant — disables the deadline.
    pub request_prefill_rate: u64,
    /// Watchdog decode floor (tokens/second); 0 as above.
    pub request_decode_rate: u64,
    /// Fixed allowance on top of the deadline's throughput terms.
    pub request_slack: Duration,
    /// How the engine orders the queue.
    pub schedule: Schedule,
    /// Waiting this long wins over any cost estimate (starvation guard).
    pub schedule_age_limit: Duration,
    /// Draw the live dashboard instead of writing log lines to stderr. The
    /// events are the same either way; only the sink that consumes them differs.
    pub tui: bool,
    /// Where the on-disk prefix cache keeps its images (under `<dir>/kv/`).
    /// `None` means no directory could be resolved — `$HOME` is unset and nothing
    /// named one — which leaves the disk tier off however `disk_cache` is set.
    pub cache_dir: Option<PathBuf>,
    /// Whether cache slots are also written to disk, so a restart resumes warm.
    /// Perf-only: every failure on that path degrades to a cache miss.
    pub disk_cache: bool,
    /// Ceiling in GiB on everything under `<cache_dir>/kv/`, all checkpoints
    /// included.
    pub disk_max_gib: u64,
    /// Conversations shorter than this are never written to disk.
    pub disk_min_tokens: usize,
    pub thinking_force: bool,
    /// `None` is uncapped.
    pub thinking_budget: Option<usize>,
    /// The template `reasoning_effort` rendered when a request names none, or
    /// `None` for the template's own default (xhigh). A system-preamble knob of
    /// Qwen 3.8's chat template — inert on the 3.6 checkpoints, whose template
    /// has no such parameter, which is why serving one is not a config error.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Server-wide sampling defaults for requests that omit a value, or `None`
    /// for the per-request mode default (`SamplerOptions::recommended`): the
    /// model card keys sampling to thinking on/off — temp 1.0 / top_p 0.95
    /// thinking, 0.7 / 0.80 non-thinking, top_k 20 both — and only the request
    /// knows its mode. An explicit value here pins one number for both modes.
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    /// Unlike the three above, the mode default this falls back to is keyed to
    /// the SERVED CHECKPOINT as well as the request's thinking state
    /// (`Model::recommended_presence_penalty`). Setting it here pins one number
    /// for both modes, as the others do.
    pub presence_penalty: Option<f64>,
    pub cache_snapshots: usize,
    /// Conversations kept warm at once, at least 1.
    pub cache_slots: usize,
    /// How this server speculates. Speculation is opt-out, so the resolved
    /// default is [`DraftMode::Official`] — which sidecar that is depends on the
    /// checkpoint, so the resolution stays out of the merge (which is pure) and
    /// out of the settings: one server loads whichever checkpoint a request
    /// names. The settings below apply to whatever drafter ends up attached.
    pub draft: DraftMode,
    /// The deepest draft a round asks for, or `None` to take the loaded
    /// checkpoint's own fitted default ([`crate::hub::Model::draft_max_default`])
    /// — resolved at attach time, like `draft_p_min`, because the merge cannot
    /// know which checkpoint a future job will name and the two drafter kinds
    /// want very different depths.
    pub draft_max: Option<usize>,
    /// The drafting confidence floor, or `None` for each checkpoint's own
    /// fitted default ([`crate::hub::Model::draft_p_min_default`]) — resolved
    /// at attach time, since one server loads whichever checkpoint a request
    /// names. An explicit value pins one floor for every checkpoint.
    pub draft_p_min: Option<f32>,
    pub draft_pause_margin: f32,
    /// Positions the drafter's cache is sized for, at least 1. Also how far into a
    /// conversation speculation stays active.
    pub draft_ctx: usize,
}

/// Read a config file. A missing file resolves to defaults, so the common case
/// of "no config at all" is not an error; anything else (unreadable, malformed,
/// unknown key) is.
pub fn load(path: &Path) -> Result<Option<ServeToml>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("reading config {}", path.display())),
    };
    let parsed =
        toml::from_str(&text).with_context(|| format!("parsing config {}", path.display()))?;
    Ok(Some(parsed))
}

/// The config path used when `--config` is absent.
pub fn default_config_path() -> Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .context("HOME is unset, so the default config path cannot be resolved — pass --config")?;
    Ok(PathBuf::from(home).join(CONFIG_RELATIVE_PATH))
}

/// Where the disk tier keeps its images when nothing names a directory. `None`
/// when `$HOME` is unset, which leaves the tier off rather than guessing at a
/// writable path — it is a cache, and a wrong guess would scatter gigabytes.
pub fn default_cache_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(CACHE_RELATIVE_PATH))
}

/// Merge CLI over file over defaults. Returns the settings plus one warning per
/// flag that overrode a differing value from the config file; the caller prints
/// them.
pub fn resolve(
    file: &ServeToml,
    source: Option<&Path>,
    cli: &CliOverrides,
) -> Result<(ServeSettings, Vec<String>)> {
    let mut warnings = Vec::new();
    let origin = Origin(source);

    let model = match pick_path(
        "model",
        "model",
        cli.model.clone(),
        file.model.clone(),
        origin,
        &mut warnings,
    ) {
        Some(model) => model,
        None => bail!("no model to serve: set `model` in the config file or pass --model <gguf>"),
    };

    let cli_idle = cli
        .idle_unload
        .as_deref()
        .map(|text| parse_idle_unload(text).context("--idle-unload"))
        .transpose()?;
    let file_idle = file
        .idle_unload
        .as_deref()
        .map(|text| parse_idle_unload(text).context("config idle_unload"))
        .transpose()?;

    let cli_tools_mode = cli
        .tools_mode
        .as_deref()
        .map(|text| parse_tools_mode(text).context("--tools-mode"))
        .transpose()?;
    let file_tools_mode = file
        .tools_mode
        .as_deref()
        .map(|text| parse_tools_mode(text).context("config tools_mode"))
        .transpose()?;

    let cli_schedule = cli
        .schedule
        .as_deref()
        .map(|text| parse_schedule(text).context("--schedule"))
        .transpose()?;
    let file_schedule = file
        .schedule
        .as_deref()
        .map(|text| parse_schedule(text).context("config schedule"))
        .transpose()?;

    let cli_effort = cli
        .reasoning_effort
        .as_deref()
        .map(|text| {
            text.parse::<ReasoningEffort>()
                .context("--reasoning-effort")
        })
        .transpose()?;
    let file_effort = file
        .thinking
        .effort
        .as_deref()
        .map(|text| {
            text.parse::<ReasoningEffort>()
                .context("config thinking.effort")
        })
        .transpose()?;

    // The dashboard is the one setting with a switch each way, so the conflict
    // warning has to name the flag that was actually passed.
    let tui_flag = if cli.tui == Some(false) {
        "no-tui"
    } else {
        "tui"
    };
    // The disk cache is the second, and defaults the other way round.
    let disk_cache_flag = if cli.disk_cache == Some(false) {
        "no-disk-cache"
    } else {
        "disk-cache"
    };
    // Speculation's switch has a direction too: --no-draft disables, while an
    // explicit --draft (naming a drafter is a request for one) enables over a
    // config that turned speculation off.
    let draft_flag = if cli.draft_enabled == Some(false) {
        "no-draft"
    } else {
        "draft"
    };

    // The context length the queue timeout is derived from, resolved ahead of
    // the struct so it can feed two fields. Plain precedence, no warning of its
    // own: the `pick` below emits whatever this shadows.
    let resolved_context_length = cli
        .context_length
        .or(file.context_length)
        .unwrap_or(DEFAULT_CONTEXT_LENGTH);

    let settings = ServeSettings {
        model,
        host: pick(
            "host",
            "host",
            cli.host.clone(),
            file.host.clone(),
            DEFAULT_HOST.to_string(),
            origin,
            &mut warnings,
        ),
        port: pick(
            "port",
            "port",
            cli.port,
            file.port,
            DEFAULT_PORT,
            origin,
            &mut warnings,
        ),
        context_length: pick(
            "ctx",
            "context_length",
            cli.context_length,
            file.context_length,
            DEFAULT_CONTEXT_LENGTH,
            origin,
            &mut warnings,
        ),
        idle_unload: pick(
            "idle-unload",
            "idle_unload",
            cli_idle,
            file_idle,
            DEFAULT_IDLE_UNLOAD,
            origin,
            &mut warnings,
        )
        .0,
        anthropic: pick_flag(
            "no-anthropic",
            "anthropic",
            cli.anthropic,
            file.anthropic,
            DEFAULT_ANTHROPIC,
            origin,
            &mut warnings,
        ),
        openai: pick_flag(
            "no-openai",
            "openai",
            cli.openai,
            file.openai,
            DEFAULT_OPENAI,
            origin,
            &mut warnings,
        ),
        api_key: resolve_api_key(
            pick_secret(
                "api-key",
                "api_key",
                cli.api_key.clone(),
                file.api_key.clone(),
                origin,
                &mut warnings,
            ),
            &mut warnings,
        ),
        tools_mode: pick(
            "tools-mode",
            "tools_mode",
            cli_tools_mode,
            file_tools_mode,
            DEFAULT_TOOLS_MODE,
            origin,
            &mut warnings,
        ),
        queue_capacity: pick(
            "queue-capacity",
            "queue_capacity",
            cli.queue_capacity,
            file.queue_capacity,
            DEFAULT_QUEUE_CAPACITY,
            origin,
            &mut warnings,
        ),
        queue_timeout: Duration::from_secs(pick(
            "queue-timeout",
            "queue_timeout",
            cli.queue_timeout,
            file.queue_timeout,
            default_queue_timeout_secs(resolved_context_length),
            origin,
            &mut warnings,
        )),
        request_prefill_rate: pick(
            "request-prefill-rate",
            "request_prefill_rate",
            cli.request_prefill_rate,
            file.request_prefill_rate,
            DEFAULT_REQUEST_PREFILL_RATE,
            origin,
            &mut warnings,
        ),
        request_decode_rate: pick(
            "request-decode-rate",
            "request_decode_rate",
            cli.request_decode_rate,
            file.request_decode_rate,
            DEFAULT_REQUEST_DECODE_RATE,
            origin,
            &mut warnings,
        ),
        request_slack: Duration::from_secs(pick(
            "request-slack",
            "request_slack",
            cli.request_slack,
            file.request_slack,
            DEFAULT_REQUEST_SLACK_SECS,
            origin,
            &mut warnings,
        )),
        schedule: pick(
            "schedule",
            "schedule",
            cli_schedule,
            file_schedule,
            DEFAULT_SCHEDULE,
            origin,
            &mut warnings,
        ),
        schedule_age_limit: Duration::from_secs(pick(
            "schedule-age-limit",
            "schedule_age_limit",
            cli.schedule_age_limit,
            file.schedule_age_limit,
            DEFAULT_SCHEDULE_AGE_LIMIT_SECS,
            origin,
            &mut warnings,
        )),
        tui: pick_flag(
            tui_flag,
            "tui",
            cli.tui,
            file.tui,
            DEFAULT_TUI,
            origin,
            &mut warnings,
        ),
        cache_dir: pick_path(
            "cache-dir",
            "cache_dir",
            cli.cache_dir.clone(),
            file.cache_dir.clone(),
            origin,
            &mut warnings,
        )
        .or_else(default_cache_dir),
        disk_cache: pick_flag(
            disk_cache_flag,
            "disk_cache",
            cli.disk_cache,
            file.disk_cache,
            DEFAULT_DISK_CACHE,
            origin,
            &mut warnings,
        ),
        disk_max_gib: pick(
            "disk-max-gib",
            "disk_max_gib",
            cli.disk_max_gib,
            file.disk_max_gib,
            DEFAULT_DISK_MAX_GIB,
            origin,
            &mut warnings,
        ),
        disk_min_tokens: pick(
            "disk-min-tokens",
            "disk_min_tokens",
            cli.disk_min_tokens,
            file.disk_min_tokens,
            DEFAULT_DISK_MIN_TOKENS,
            origin,
            &mut warnings,
        ),
        thinking_force: file.thinking.force.unwrap_or(DEFAULT_THINKING_FORCE),
        thinking_budget: match file
            .thinking
            .default_budget
            .unwrap_or(DEFAULT_THINKING_BUDGET)
        {
            0 => None,
            n => Some(n),
        },
        // Absent stays absent, meaning the template's own default level.
        reasoning_effort: pick_opt(
            "reasoning-effort",
            "thinking.effort",
            cli_effort,
            file_effort,
            origin,
            &mut warnings,
        ),
        // Sampling has no resolved defaults: the model card's recommendation is
        // keyed to each request's thinking mode, so absent stays absent and the
        // dialects fill in `SamplerOptions::recommended` per request.
        temperature: pick_opt(
            "temp",
            "sampling.temperature",
            cli.temperature,
            file.sampling.temperature,
            origin,
            &mut warnings,
        ),
        top_k: pick_opt(
            "top-k",
            "sampling.top_k",
            cli.top_k,
            file.sampling.top_k,
            origin,
            &mut warnings,
        ),
        top_p: pick_opt(
            "top-p",
            "sampling.top_p",
            cli.top_p,
            file.sampling.top_p,
            origin,
            &mut warnings,
        ),
        presence_penalty: pick_opt(
            "presence-penalty",
            "sampling.presence_penalty",
            cli.presence_penalty,
            file.sampling.presence_penalty,
            origin,
            &mut warnings,
        ),
        cache_snapshots: pick(
            "cache-snapshots",
            "cache.snapshots",
            cli.cache_snapshots,
            file.cache.snapshots,
            DEFAULT_CACHE_SNAPSHOTS,
            origin,
            &mut warnings,
        ),
        cache_slots: pick(
            "cache-slots",
            "cache.slots",
            cli.cache_slots,
            file.cache.slots,
            DEFAULT_CACHE_SLOTS,
            origin,
            &mut warnings,
        ),
        draft: {
            let path = pick_path(
                "draft",
                "draft.path",
                cli.draft.clone(),
                file.draft.path.clone(),
                origin,
                &mut warnings,
            );
            // Opt-out speculation: the default enables on its own, and the
            // `|| path.is_some()` keeps a config that sets only `path` drafting
            // whichever way DEFAULT_DRAFT_ENABLED is set.
            let enabled = pick_flag(
                draft_flag,
                "draft.enabled",
                cli.draft_enabled,
                file.draft.enabled,
                DEFAULT_DRAFT_ENABLED || path.is_some(),
                origin,
                &mut warnings,
            );
            // Enabled with no path named means each checkpoint's own official
            // sidecar, which the engine resolves against the Hugging Face cache
            // when that checkpoint loads — the merge stays pure and, more to the
            // point, stays out of a decision that depends on which checkpoint a
            // request names. The literal `official` is the symbolic spelling of
            // that same default, and stays spellable so a config file (or
            // `--draft official`) can restate it.
            match (enabled, path) {
                (false, _) => DraftMode::Off,
                (true, Some(path)) if path == Path::new(OFFICIAL_DRAFTER) => DraftMode::Official,
                (true, Some(path)) => DraftMode::Custom(path),
                (true, None) => DraftMode::Official,
            }
        },
        draft_max: pick_opt(
            "draft-max",
            "draft.max",
            cli.draft_max,
            file.draft.max,
            origin,
            &mut warnings,
        ),
        // The one knob with no resolved default: the drafting floor is fitted
        // per checkpoint, and one server now loads whichever checkpoint a
        // request names, so `None` here means "each checkpoint's own fitted
        // default" — resolved at attach time, when the checkpoint is known.
        // An explicit value pins one floor for every checkpoint served.
        draft_p_min: pick_opt(
            "draft-p-min",
            "draft.p_min",
            cli.draft_p_min,
            file.draft.p_min,
            origin,
            &mut warnings,
        ),
        draft_pause_margin: pick(
            "draft-pause-margin",
            "draft.pause_margin",
            cli.draft_pause_margin,
            file.draft.pause_margin,
            DEFAULT_DRAFT_PAUSE_MARGIN,
            origin,
            &mut warnings,
        ),
        draft_ctx: pick(
            "draft-ctx",
            "draft.ctx",
            cli.draft_ctx,
            file.draft.ctx,
            DEFAULT_DRAFT_CTX,
            origin,
            &mut warnings,
        ),
    };

    // Zero queued jobs is not "no queueing", it is a server that can accept
    // nothing at all: the running job's successor has to wait somewhere.
    ensure!(
        settings.queue_capacity >= 1,
        "queue_capacity must be at least 1 (a request has to be able to wait \
         behind the one running generation)"
    );
    // A zero timeout would drop every request at its first dequeue; "no
    // timeout" is not offered because a queue that holds requests forever
    // serves answers nobody is waiting for.
    ensure!(
        settings.queue_timeout >= Duration::from_secs(1),
        "queue_timeout must be at least 1 second (0 would drop every queued \
         request before the engine could pick it up)"
    );
    // The rates may be 0 (that disables the deadline), but the age limit may
    // not: at 0 every entry is instantly "aged" and the scheduler degrades to
    // FIFO silently — schedule = "fifo" is the explicit way to ask for that.
    ensure!(
        settings.schedule_age_limit >= Duration::from_secs(1),
        "schedule_age_limit must be at least 1 second (0 makes every queued \
         request aged at once, which is FIFO; set schedule = \"fifo\" to ask \
         for that explicitly)"
    );

    // Zero slots is not "no caching", it is nowhere for the conversation being
    // served to live; 1 is the setting that turns multi-conversation reuse off.
    ensure!(
        settings.cache_slots >= 1,
        "cache_slots must be at least 1 (1 keeps one conversation warm, which is \
         what disabling multi-slot reuse means)"
    );

    // The disk tier needs somewhere to live. Warned about rather than refused: it
    // is perf-only, and a server that will not start because a cache directory
    // could not be named would be the worse trade.
    if settings.disk_cache && settings.cache_dir.is_none() {
        warnings.push(
            "warning: HOME is unset and no cache_dir is set, so cache images cannot be \
             kept on disk; set cache_dir or pass --cache-dir"
                .to_string(),
        );
    }

    // Every drafter knob has a range outside which speculation is configured, loaded and
    // announced at startup, and then never actually happens. Refused rather than warned
    // about, so "speculating over the first N positions" is never a lie.
    ensure!(
        settings.draft_ctx >= 1,
        "draft_ctx must be at least 1 (it sizes the drafter's KV cache, so zero \
         positions is speculation that can never run)"
    );
    if let Some(draft_max) = settings.draft_max {
        ensure!(
            draft_max >= 1,
            "draft.max must be at least 1 (it caps the tokens a round proposes, so zero \
             drafts nothing and every round falls back to plain decode)"
        );
    }
    if let Some(p_min) = settings.draft_p_min {
        ensure!(
            (0.0..=1.0).contains(&p_min),
            "draft.p_min must be between 0 and 1, not {p_min} (it is compared against a \
             probability: above 1 no draft ever qualifies and speculation never runs, \
             while 0 keeps every draft and proposes full blocks)"
        );
    }
    ensure!(
        settings.draft_pause_margin >= 0.0,
        "draft.pause_margin must not be negative ({} would hold speculation paused \
         forever; 0 disables pausing and always drafts)",
        settings.draft_pause_margin
    );

    Ok((settings, warnings))
}

/// An empty key is no key. It reads as authentication being configured while
/// accepting every request, including one with no credentials at all, so the
/// server says out loud which of the two it is doing.
fn resolve_api_key(key: Option<String>, warnings: &mut Vec<String>) -> Option<String> {
    match key {
        Some(key) if key.trim().is_empty() => {
            warnings.push(
                "warning: api_key is empty, which configures no authentication at all; \
                 remove it or set a key"
                    .to_string(),
            );
            None
        }
        key => key,
    }
}

/// Where a config value came from, for the override warnings.
#[derive(Clone, Copy)]
struct Origin<'a>(Option<&'a Path>);

impl fmt::Display for Origin<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(path) => write!(f, "{}", path.display()),
            None => write!(f, "config"),
        }
    }
}

/// CLI beats file beats default, warning when the flag contradicts the file.
fn pick<T: PartialEq + fmt::Display>(
    flag: &str,
    key: &str,
    cli: Option<T>,
    file: Option<T>,
    default: T,
    origin: Origin<'_>,
    warnings: &mut Vec<String>,
) -> T {
    match (cli, file) {
        (Some(cli), Some(file)) => {
            if cli != file {
                warnings.push(format!(
                    "warning: --{flag} {cli} overrides config {key} = {file} ({origin})"
                ));
            }
            cli
        }
        (Some(cli), None) => cli,
        (None, Some(file)) => file,
        (None, None) => default,
    }
}

/// Same as [`pick`] for a setting with no default (absent stays absent).
fn pick_opt<T: PartialEq + fmt::Display>(
    flag: &str,
    key: &str,
    cli: Option<T>,
    file: Option<T>,
    origin: Origin<'_>,
    warnings: &mut Vec<String>,
) -> Option<T> {
    match (cli, file) {
        (Some(cli), Some(file)) => {
            if cli != file {
                warnings.push(format!(
                    "warning: --{flag} {cli} overrides config {key} = {file} ({origin})"
                ));
            }
            Some(cli)
        }
        (cli @ Some(_), None) => cli,
        (None, file) => file,
    }
}

/// Same as [`pick`] for a path setting with no default (absent stays absent).
fn pick_path(
    flag: &str,
    key: &str,
    cli: Option<PathBuf>,
    file: Option<PathBuf>,
    origin: Origin<'_>,
    warnings: &mut Vec<String>,
) -> Option<PathBuf> {
    match (cli, file) {
        (Some(cli), Some(file)) => {
            if cli != file {
                warnings.push(format!(
                    "warning: --{flag} {} overrides config {key} = {} ({origin})",
                    cli.display(),
                    file.display(),
                ));
            }
            Some(cli)
        }
        (cli @ Some(_), None) => cli,
        (None, file) => file,
    }
}

/// Like [`pick_opt`], but the values are secret and never printed.
fn pick_secret(
    flag: &str,
    key: &str,
    cli: Option<String>,
    file: Option<String>,
    origin: Origin<'_>,
    warnings: &mut Vec<String>,
) -> Option<String> {
    match (cli, file) {
        (Some(cli), Some(file)) => {
            if cli != file {
                warnings.push(format!(
                    "warning: --{flag} overrides config {key} ({origin})"
                ));
            }
            Some(cli)
        }
        (cli @ Some(_), None) => cli,
        (None, file) => file,
    }
}

/// A `--no-*` switch: present means an explicit `false`, so the warning names
/// only the flag.
fn pick_flag(
    flag: &str,
    key: &str,
    cli: Option<bool>,
    file: Option<bool>,
    default: bool,
    origin: Origin<'_>,
    warnings: &mut Vec<String>,
) -> bool {
    match (cli, file) {
        (Some(cli), Some(file)) => {
            if cli != file {
                warnings.push(format!(
                    "warning: --{flag} overrides config {key} = {file} ({origin})"
                ));
            }
            cli
        }
        (Some(cli), None) => cli,
        (None, Some(file)) => file,
        (None, None) => default,
    }
}

/// The `--init` template: every setting at its built-in default, with the
/// reasoning behind it. Values are interpolated from the same constants the
/// merge uses, so the template cannot drift from the code.
pub fn init_template() -> String {
    let host = DEFAULT_HOST;
    let port = DEFAULT_PORT;
    let ctx = DEFAULT_CONTEXT_LENGTH;
    // Cache sizes are a property of the checkpoint, not of the server, so they
    // are derived from the model this template quotes rather than restated here.
    let kv_bytes_per_token = TEMPLATE_MODEL.kv_bytes_per_token();
    let snapshot_bytes = TEMPLATE_MODEL.snapshot_bytes();
    let ctx_gb = (ctx * kv_bytes_per_token) as f64 / 1e9;
    let idle = DEFAULT_IDLE_UNLOAD;
    let anthropic = DEFAULT_ANTHROPIC;
    let openai = DEFAULT_OPENAI;
    let tools_mode = DEFAULT_TOOLS_MODE;
    let queue_capacity = DEFAULT_QUEUE_CAPACITY;
    let queue_timeout = default_queue_timeout_secs(ctx);
    let prefill_rate = DEFAULT_REQUEST_PREFILL_RATE;
    let decode_rate = DEFAULT_REQUEST_DECODE_RATE;
    let request_slack = DEFAULT_REQUEST_SLACK_SECS;
    let schedule = DEFAULT_SCHEDULE;
    let age_limit = DEFAULT_SCHEDULE_AGE_LIMIT_SECS;
    let tui = DEFAULT_TUI;
    let think_force = DEFAULT_THINKING_FORCE;
    let think_budget = DEFAULT_THINKING_BUDGET;
    // Sampling defaults are per-request and mode-dependent; the template quotes
    // both sets from the same source the dialects resolve against.
    let thinking_sampling = crate::sampler::SamplerOptions::recommended(true);
    let instruct_sampling = crate::sampler::SamplerOptions::recommended(false);
    let temp_think = format!("{:?}", thinking_sampling.temperature);
    let top_p_think = format!("{:?}", thinking_sampling.top_p);
    let temp_instruct = format!("{:?}", instruct_sampling.temperature);
    let top_p_instruct = format!("{:?}", instruct_sampling.top_p);
    let top_k = thinking_sampling.top_k;
    // The one card value that is per checkpoint, so the template quotes the
    // model it was generated for rather than a shared pair.
    let penalty_think = format!("{:?}", TEMPLATE_MODEL.recommended_presence_penalty(true));
    let penalty_instruct = format!("{:?}", TEMPLATE_MODEL.recommended_presence_penalty(false));
    let snapshots = DEFAULT_CACHE_SNAPSHOTS;
    let snapshot_mib = snapshot_bytes / (1024 * 1024);
    let snapshots_mib = snapshots * snapshot_mib;
    let slots = DEFAULT_CACHE_SLOTS;
    let draft_max = DEFAULT_DRAFT_MAX;
    // Per-checkpoint, like the cache sizes above: both quoted (commented out)
    // for the model the template names.
    let draft_p_min = format!(
        "{:?}",
        TEMPLATE_MODEL
            .draft_p_min_default()
            .expect("the template model ships a drafter, and so a fitted floor")
    );
    let draft_pause_margin = format!("{:?}", DEFAULT_DRAFT_PAUSE_MARGIN);
    let draft_enabled = DEFAULT_DRAFT_ENABLED;
    let draft_ctx = DEFAULT_DRAFT_CTX;
    let draft_kib_per_token = DRAFT_KV_BYTES_PER_TOKEN / 1024;
    let draft_ctx_gb = (draft_ctx * DRAFT_KV_BYTES_PER_TOKEN) as f64 / 1e9;
    let kib_per_token = kv_bytes_per_token / 1024;
    let snapshots_gb = (snapshots * snapshot_bytes) as f64 / 1e9;
    let slot_example_gb = (20_000 * kv_bytes_per_token) as f64 / 1e9 + snapshots_gb;
    // The worst case is EVERY slot holding a conversation that fills the context — the live
    // one included, since it keeps the image it was paged in with — and, with a drafter
    // configured, a full set of drafter planes alongside each.
    let slots_ceiling_gb = slots as f64 * (ctx_gb + snapshots_gb);
    // One slot's images, which is both the per-slot term of the ceiling and the transient a
    // swap adds on top of it: the page-out builds its fresh exports before the manager
    // releases the images they replace.
    let slot_images_gb = ctx_gb + draft_ctx_gb.min(ctx_gb) + snapshots_gb;
    let slots_ceiling_draft_gb = slots as f64 * slot_images_gb;
    let disk_cache = DEFAULT_DISK_CACHE;
    let disk_max_gib = DEFAULT_DISK_MAX_GIB;
    let disk_min_tokens = DEFAULT_DISK_MIN_TOKENS;
    // One 20k-token conversation's file, which is the size to reason about: the
    // full-attention rows plus every snapshot the slot retained, and the drafter's
    // planes when speculation is on.
    let disk_example_gb = slot_example_gb;
    let disk_example_draft_gb = slot_example_gb + (20_000 * DRAFT_KV_BYTES_PER_TOKEN) as f64 / 1e9;
    let disk_max_examples = (disk_max_gib as f64 * 1.073_741_824) / disk_example_gb;

    format!(
        r#"# xwen serve configuration.
#
# Every value below is the built-in default, so deleting a line changes nothing.
# Command-line flags win over this file, and each one that contradicts a value
# set here prints a warning naming the flag, both values and this path.

# The model GGUF to serve. Unset (and no --model), the server's default
# checkpoint is used, fetched into the Hugging Face cache on first use
# (`xwen fetch` prefetches). That default is Qwen3.6-35B-A3B, not xwen's overall
# default: the server cannot run Qwen3.8-Flash-Next until its recurrent state
# rides in a cache image, so it serves the best checkpoint it can and says so at
# startup. `--model-size` picks another.
# model = "/path/to/Qwen3.6-35B-A3B-Q4_K_M.gguf"

# Address to bind. Loopback keeps the server off the network; "0.0.0.0" accepts
# connections from other machines, which is worth pairing with api_key below.
host = "{host}"

# TCP port.
port = {port}

# Context length in tokens — a ceiling, not an allocation. The full-attention
# KV cache starts small (8192 positions) and grows on demand as a conversation
# lengthens, up to about {ctx_gb:.1} GB at {ctx} for the 35B-A3B's geometry and
# roughly three times that for the 27B, whose full-attention layers are both
# more numerous and wider; an idle unload drops the grown buffers, so a
# reloaded server starts small again. Only those layers grow with context —
# the DeltaNet layers carry a fixed recurrent state either way. The default is
# the checkpoint's own trained window, and a checkpoint converted for less is
# served at its own limit whatever is set here. Decode speed depends on the
# tokens a conversation actually uses, not on what was allocated.
context_length = {ctx}

# Drop the model after this long without a request, returning the GPU-resident
# weights (19-37 GB depending on checkpoint and quant) and whatever the KV
# cache has grown to. The next request reloads in a few seconds and starts the
# KV cache small again. Write an integer with an s, m or h suffix, or "off" to
# hold the model in memory forever.
idle_unload = "{idle}"

# Serve the Anthropic Messages API: POST /v1/messages and
# POST /v1/messages/count_tokens.
anthropic = {anthropic}

# Serve the OpenAI Chat Completions API: POST /v1/chat/completions.
openai = {openai}

# Require this key on every request, as `x-api-key` or `Authorization: Bearer`.
# Unset, the server accepts any credentials including none — which is why host
# defaults to loopback.
# api_key = "..."

# What to do with a request that carries tool definitions. "native" serves them:
# the definitions are rendered into the prompt and the calls the model writes are
# parsed back out of its output. The other two are debugging modes from before
# that worked — "reject" fails any request carrying tools, and "strip" drops the
# definitions and answers as though the request had none, which is a way to tell
# a bad tool rendering apart from a bad answer.
tools_mode = "{tools_mode}"

# Requests that may wait behind the one running generation. Waiting is cheap for
# a streaming client — keep-alive frames hold the connection open — so a short
# queue beats a refusal; at capacity the server answers 429 with Retry-After: 1
# and the client retries on its ordinary backoff.
queue_capacity = {queue_capacity}

# Drop a request that has waited this long in the queue, in seconds. The client
# gets an error naming the wait; by then it has usually given up and retried
# anyway. The default is derived from context_length — two maximal prefills at
# the slowest rate this machine has been measured at — and never below 300.
queue_timeout = {queue_timeout}

# Wall-clock ceiling for one request, derived from its size:
#
#   prompt_tokens / request_prefill_rate + max_tokens / request_decode_rate
#     + request_slack
#
# The rates (tokens/second) are WATCHDOG bounds, deliberately about 2x looser
# than this machine's measured low-power-mode throughput, so the ceiling only
# ever catches a wedged generation, never a slow one. They are not performance
# targets — at full power the same values are about 4x loose. A rate of 0 makes
# its term unbounded, and since the ceiling is a single instant an unbounded
# term disables the deadline entirely; set both to 0 to say so explicitly.
# request_slack is the fixed allowance in seconds on top: model load,
# scheduling, cache paging.
request_prefill_rate = {prefill_rate}
request_decode_rate = {decode_rate}
request_slack = {request_slack}

# Queue order. "shortest-prefill" runs the job with the least prefill actually
# required first — prompt tokens minus what the KV cache already holds for it —
# so a 4-token title request does not wait behind a 100k-token cold prompt.
# "fifo" restores strict arrival order. Note the queue can only reorder jobs
# that have not started: a request submitted just after a long generation began
# still waits for all of it.
schedule = "{schedule}"

# Starvation guard for "shortest-prefill", in seconds: a request that has waited
# this long wins over any cheaper newcomer, oldest first.
schedule_age_limit = {age_limit}

# Draw a live dashboard — the running request's progress, the queue, the cache
# slots, finished requests and the log — instead of printing log lines. On by
# default; when stderr is not a terminal (a `2>` redirect, a pipe, CI) it steps
# aside on its own and logs plainly, so a headless run needs no flag. Set false
# (or pass --no-tui) to always print the plain lines: the events behind both
# are the same, and with the dashboard off the lines are exactly what they have
# always been. Quit it with q or Ctrl-C, which shut the server down the same
# way a signal does.
tui = {tui}

# Where the on-disk prefix cache keeps its images, under <cache_dir>/kv/. Unset,
# it is ~/.cache/xwen.
# cache_dir = "~/.cache/xwen"

# Also write warm conversations to disk, so a restart resumes them instead of
# re-prefilling. A conversation's image is written whenever it leaves the GPU
# cache and on the way down, and the next server that loads the same checkpoint
# reads it back: a 20k-token system prompt that costs a minute of prefill costs
# about a second of NVMe instead.
#
# An image is the same bytes a warm slot holds — {kib_per_token} KiB per cached token plus
# {snapshot_mib} MiB per retained snapshot, so about {disk_example_gb:.1} GB for a 20k-token conversation,
# or about {disk_example_draft_gb:.1} GB with a drafter configured (its planes ride along).
#
# The tier is perf-only: every way an image can be missing, stale or damaged ends
# in an ordinary cache miss. Images are bound to the checkpoint that produced them
# and to the tokenizer rules that encoded them, and anything that does not match
# is deleted on sight.
disk_cache = {disk_cache}

# Ceiling in GiB on everything under <cache_dir>/kv/, every checkpoint included,
# enforced by deleting the least recently used image — about {disk_max_examples:.0} conversations of
# the size above. Switching models therefore cannot strand an unbounded pile.
disk_max_gib = {disk_max_gib}

# Do not write conversations shorter than this. The cost of an image is dominated
# by its snapshots ({snapshot_mib} MiB each whatever the length), while the prefill it saves
# grows with the token count, so short side requests are not worth storing.
disk_min_tokens = {disk_min_tokens}

[thinking]
# Open every assistant turn inside a <think> block, the way Anthropic's manual
# extended-thinking mode does. A request that states its own thinking setting
# wins over this.
force = {think_force}

# Thinking-token budget for requests that do not carry one of their own.
# Capped by default: the model is prone to reasoning loops, and an uncapped
# think block at low-power decode rates can outlast an agent client's request
# timeout (Claude Code gives up at 600s and retries the whole request). The
# budget steers the model out at its next sentence boundary rather than
# truncating. A request's own thinking.budget_tokens / reasoning_effort wins
# over this; 0 is uncapped.
default_budget = {think_budget}

# Reasoning-effort level rendered into Qwen 3.8's chat template when a request
# names none: "low", "medium" or "xhigh" (the template's own default). A
# system-preamble instruction only — no token-budget semantics — and inert on
# the 3.6 checkpoints, whose template has no such parameter. A request's own
# reasoning_effort wins over this.
# effort = "xhigh"

[sampling]
# Server-wide defaults for requests that omit a value. Unset (the default),
# each request samples with the model card's recommendation for its own mode,
# which is keyed to thinking on/off: temp {temp_think} / top_p {top_p_think} with thinking,
# temp {temp_instruct} / top_p {top_p_instruct} without, top_k {top_k} either way. Setting a key here
# pins one number for both modes on every request that omits it.
#
# presence_penalty is the one default that also depends on which checkpoint is
# loaded: {penalty_think} with thinking and {penalty_instruct} without, on the model this file was
# generated for. It subtracts that much from the logit of every token the reply
# has already produced, once per distinct token; 0 turns it off. top_k = 0
# means no top-k cut at all, and top_k = 1 is greedy.
# temperature = {temp_think}
# top_k = {top_k}
# top_p = {top_p_think}
# presence_penalty = {penalty_think}

[cache]
# KV snapshots kept at turn boundaries, so a conversation that edits or retries
# its last turn rolls back instead of re-prefilling from scratch. Each snapshot
# copies every DeltaNet layer's recurrent state, about {snapshot_mib} MiB whatever the
# position it covers, so {snapshots} of them cost roughly {snapshots_mib} MiB.
snapshots = {snapshots}

# Conversations kept warm at once. One of them occupies the GPU cache; each of
# the others is a host-RAM image the engine pages back in when that conversation
# speaks again, so two clients talking in turn stop evicting each other. Paging
# one in or out is a unified-memory copy, milliseconds against the minute
# re-prefilling it would take. 1 keeps a single conversation warm, which is what
# this server did before slots existed.
#
# The images are what cost memory, and the cost is NOT capped. Every warm
# conversation can hold one, the live one included — it keeps the image it was
# paged in with, so a request that fails hands its conversation back instead of
# losing it — so budget
#
#   slots x (context_length x {kib_per_token} KiB + {snapshots} x {snapshot_mib} MiB)
#
# which is about {slots_ceiling_gb:.0} GB at the values above, or about {slots_ceiling_draft_gb:.0} GB with a drafter
# configured: that adds a set of drafter planes per slot, min(draft.ctx,
# context_length) x {draft_kib_per_token} KiB, roughly doubling the per-token cost up to draft.ctx.
# The server will take and keep that much if that many conversations each grow to
# fill the context, and a swap peaks one slot's images above it — about {slot_images_gb:.0} GB more
# at these values — because paging a conversation out builds its fresh exports
# before releasing the images they replace. Ordinary conversations sit far below
# all of this (a 20k-token one holds about {slot_example_gb:.1} GB, or twice that with a drafter),
# so the ceiling is what to size the machine against rather than what to expect.
# Lower this, context_length, or draft.ctx if it does not fit.
slots = {slots}

[draft]
# Speculative decoding: the drafter proposes several tokens ahead, one batched
# target forward verifies them, and every token the target would have sampled
# anyway is committed for free. Which drafter depends on the checkpoint — the 3.6
# files ship DFlash sidecars that propose a block per forward, the 3.8 an MTP head
# that chains a few steps. Greedy output matches decoding
# without it except where a near-tie lands differently. On by default: measured
# faster on both checkpoints on 2026-07-29 — 27B +19.3 to +21.0% on code and
# +7.6 to +8.4% on chat, 35B-A3B +18.1 to +19.8% on code and +12.6 to +12.8% on
# chat (greedy, 128 tokens, warm). It uses the official drafter for the
# checkpoint, fetched into the Hugging Face cache on first use (3.5 GB on the
# 27B, 0.8 GB on the 35B-A3B, 3.2 GB on the 3.8-27B); `path` names a custom
# drafter GGUF instead, of either kind.
# `enabled = false` decodes plain.
enabled = {draft_enabled}
# path = "/path/to/custom-drafter.gguf"

# Max draft tokens proposed per verify round. Unset, each checkpoint the server
# loads uses its own fitted default, because the two drafter kinds have opposite
# economics off this one knob: a DFlash sidecar proposes its whole block in one
# forward, so {draft_max} costs no more than 3 would, while an MTP head pays a
# forward per step and is asked for 3. Setting the key pins one depth for every
# checkpoint this server is asked to serve; the value below is the DFlash one,
# and pinning it onto the 3.8-27B would draft five times deeper than that head
# earns.
# max = {draft_max}

# Stop drafting at the first token whose probability falls below this, so a round
# drafts as far as the drafter is confident and no further. Adaptive length beats
# any fixed block. Unset, each checkpoint the server loads uses its own fitted
# default — 0.5 on the 27B and 0.3 on the 35B-A3B (fitted 2026-08-08), 0.5 on the
# 3.8-27B. Setting the key
# pins one floor for every checkpoint this server is asked to serve; the value below
# is the 35B-A3B's, and pinning it onto the 27B costs that model ~11%.
# p_min = {draft_p_min}

# Pause speculation when its measured wall-clock cost per committed token exceeds a
# plain decode step's times this factor, and resume when it pays again. This is
# what keeps prose — where acceptance is low — from decoding slower with a drafter
# than without one. 0 always drafts.
#
# With pausing on, a temperature>0 reply is not reproducible from a fixed seed:
# which rounds batch-verify depends on wall-clock timing, and a batched round can
# differ from a plain one at a near-tie. Greedy output is unaffected either way.
pause_margin = {draft_pause_margin}

# Positions the drafter's KV cache is sized for, and equally how far into a
# conversation speculation stays active — past it decode continues plain. Sized
# separately from context_length because the drafter's cache costs
# {draft_kib_per_token} KiB per token: about {draft_ctx_gb:.1} GB at {draft_ctx}, and that again for every warm
# conversation that has an image (see cache.slots above), where inheriting the
# target's context would spend several times as much on positions speculation
# rarely reaches.
ctx = {draft_ctx}
"#
    )
}

/// Write the `--init` template, creating parent directories. Refuses to touch
/// an existing file: the template carries no user edits worth losing to it.
pub fn write_init_template(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating config directory {}", parent.display()))?;
    }
    let mut file = match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            bail!(
                "{} already exists; delete it or pass --config <path> to write elsewhere",
                path.display()
            )
        }
        Err(e) => return Err(e).with_context(|| format!("creating {}", path.display())),
    };
    use std::io::Write;
    file.write_all(init_template().as_bytes())
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_only() -> CliOverrides {
        CliOverrides {
            model: Some(PathBuf::from("/m.gguf")),
            ..Default::default()
        }
    }

    fn defaults() -> ServeSettings {
        let (settings, warnings) =
            resolve(&ServeToml::default(), None, &model_only()).expect("defaults resolve");
        assert!(warnings.is_empty());
        settings
    }

    /// The queue timeout follows the context length, and never drops below the
    /// 300 s floor however short the context is. The point of the derivation is
    /// that a request arriving behind one maximal prefill is not dropped for
    /// saturation while the server is working normally, so the value has to
    /// cover two of them.
    #[test]
    fn the_queue_timeout_default_covers_two_maximal_prefills() {
        // A context short enough that two prefills fit inside the floor.
        assert_eq!(default_queue_timeout_secs(1024), DEFAULT_QUEUE_TIMEOUT_SECS);
        assert_eq!(default_queue_timeout_secs(8192), DEFAULT_QUEUE_TIMEOUT_SECS);
        assert_eq!(
            default_queue_timeout_secs(16384),
            DEFAULT_QUEUE_TIMEOUT_SECS
        );
        // Past that it grows with the context, rounded up to a whole minute.
        assert_eq!(default_queue_timeout_secs(131072), 1320);
        assert_eq!(default_queue_timeout_secs(262144), 2640);
        assert_eq!(default_queue_timeout_secs(524288), 5280);
        // And it never falls under the two prefills it exists to cover.
        for ctx in [65536usize, 131072, 262144, 524288] {
            let two_prefills = 2 * ctx as u64 / SLOWEST_PREFILL_TOKENS_PER_SEC;
            assert!(
                default_queue_timeout_secs(ctx) >= two_prefills,
                "short at {ctx}"
            );
        }
        // Monotone, which is the only property a caller reasons about.
        let mut last = 0;
        for ctx in [1024, 8192, 65536, 131072, 262144, 524288] {
            let now = default_queue_timeout_secs(ctx);
            assert!(now >= last, "{ctx} went backwards: {now} < {last}");
            last = now;
        }
    }

    #[test]
    fn defaults_are_the_documented_values() {
        let s = defaults();
        assert_eq!(s.host, DEFAULT_HOST);
        assert_eq!(s.port, DEFAULT_PORT);
        assert_eq!(s.context_length, DEFAULT_CONTEXT_LENGTH);
        assert_eq!(s.idle_unload, Some(Duration::from_secs(300)));
        assert!(s.anthropic && s.openai);
        assert_eq!(s.api_key, None);
        assert_eq!(s.tools_mode, ToolsMode::Native);
        assert_eq!(s.queue_capacity, DEFAULT_QUEUE_CAPACITY);
        // Derived from the context length, not the 300 s floor: at the default
        // 262144 one maximal prefill is ~1310 s at the measured floor rate.
        assert_eq!(
            s.queue_timeout,
            Duration::from_secs(default_queue_timeout_secs(DEFAULT_CONTEXT_LENGTH))
        );
        assert_eq!(s.queue_timeout, Duration::from_secs(2640));
        assert_eq!(s.request_prefill_rate, DEFAULT_REQUEST_PREFILL_RATE);
        assert_eq!(s.request_decode_rate, DEFAULT_REQUEST_DECODE_RATE);
        assert_eq!(s.request_slack, Duration::from_secs(30));
        assert_eq!(s.schedule, Schedule::ShortestPrefill);
        assert_eq!(s.schedule_age_limit, Duration::from_secs(20));
        // The dashboard is the default; --no-tui restores the plain stderr
        // lines, and a non-TTY falls back to them on its own.
        assert!(s.tui);
        assert!(s.thinking_force);
        assert_eq!(s.thinking_budget, Some(DEFAULT_THINKING_BUDGET));
        // No resolved effort or sampling defaults: the effort falls to the
        // template's own default, and sampling to each request's mode set.
        assert_eq!(s.reasoning_effort, None);
        assert_eq!(s.temperature, None);
        assert_eq!(s.top_k, None);
        assert_eq!(s.top_p, None);
        assert_eq!(s.cache_snapshots, DEFAULT_CACHE_SNAPSHOTS);
        assert_eq!(s.cache_slots, DEFAULT_CACHE_SLOTS);
        // The disk tier is opt-in (SSD wear is the operator's call); when
        // turned on it is budgeted and lives under $HOME unless told otherwise.
        assert_eq!(s.disk_cache, DEFAULT_DISK_CACHE);
        assert!(!s.disk_cache);
        assert_eq!(s.cache_dir, default_cache_dir());
        assert_eq!(s.disk_max_gib, DEFAULT_DISK_MAX_GIB);
        assert_eq!(s.disk_min_tokens, DEFAULT_DISK_MIN_TOKENS);
        // Speculation is opt-out: a zero-flag server speculates with the
        // official sidecar, named symbolically for the caller to resolve.
        assert_eq!(s.draft, DraftMode::Official);
        // Both unresolved on purpose: the depth and the floor are fitted per
        // checkpoint, and which checkpoint is only known when the engine
        // attaches the drafter.
        assert_eq!(s.draft_max, None);
        assert_eq!(s.draft_p_min, None);
        assert_eq!(s.draft_pause_margin, DEFAULT_DRAFT_PAUSE_MARGIN);
        assert_eq!(s.draft_ctx, DEFAULT_DRAFT_CTX);
    }

    /// The drafting floor is the one drafter default that follows the
    /// checkpoint the engine loads, so the merge must leave it unresolved —
    /// while a config file that names `p_min` pins one floor for every
    /// checkpoint served.
    #[test]
    fn draft_p_min_stays_unresolved_unless_pinned() {
        let (unpinned, _) = resolve(&ServeToml::default(), None, &model_only()).unwrap();
        assert_eq!(unpinned.draft_p_min, None);

        let file: ServeToml = toml::from_str("[draft]\np_min = 0.25\n").unwrap();
        let (pinned, warnings) =
            resolve(&file, Some(Path::new("/etc/serve.toml")), &model_only()).unwrap();
        assert_eq!(pinned.draft_p_min, Some(0.25));
        assert!(warnings.is_empty());
    }

    /// The drafter settings follow the same CLI-over-file-over-default precedence as
    /// everything else, and naming a path is what selects a drafter other than the
    /// official one.
    #[test]
    fn draft_settings_follow_the_usual_precedence() {
        let file: ServeToml = toml::from_str(
            r#"
            [draft]
            path = "/from-config.gguf"
            max = 8
            p_min = 0.25
            pause_margin = 0.0
            ctx = 12288
            "#,
        )
        .unwrap();
        let (from_file, warnings) =
            resolve(&file, Some(Path::new("/etc/serve.toml")), &model_only()).unwrap();
        assert_eq!(
            from_file.draft,
            DraftMode::Custom(PathBuf::from("/from-config.gguf"))
        );
        assert_eq!(from_file.draft_max, Some(8));
        assert_eq!(from_file.draft_p_min, Some(0.25));
        assert_eq!(from_file.draft_pause_margin, 0.0);
        assert_eq!(from_file.draft_ctx, 12288);
        assert!(warnings.is_empty());

        let cli = CliOverrides {
            draft: Some(PathBuf::from("/from-cli.gguf")),
            draft_max: Some(4),
            draft_p_min: Some(0.75),
            draft_pause_margin: Some(2.0),
            draft_ctx: Some(4096),
            ..model_only()
        };
        let (merged, warnings) = resolve(&file, Some(Path::new("/etc/serve.toml")), &cli).unwrap();
        assert_eq!(
            merged.draft,
            DraftMode::Custom(PathBuf::from("/from-cli.gguf"))
        );
        assert_eq!(merged.draft_max, Some(4));
        assert_eq!(merged.draft_p_min, Some(0.75));
        assert_eq!(merged.draft_pause_margin, 2.0);
        assert_eq!(merged.draft_ctx, 4096);
        assert_eq!(
            warnings,
            vec![
                "warning: --draft /from-cli.gguf overrides config draft.path = /from-config.gguf (/etc/serve.toml)".to_string(),
                "warning: --draft-max 4 overrides config draft.max = 8 (/etc/serve.toml)".to_string(),
                "warning: --draft-p-min 0.75 overrides config draft.p_min = 0.25 (/etc/serve.toml)"
                    .to_string(),
                "warning: --draft-pause-margin 2 overrides config draft.pause_margin = 0 (/etc/serve.toml)"
                    .to_string(),
                "warning: --draft-ctx 4096 overrides config draft.ctx = 12288 (/etc/serve.toml)"
                    .to_string(),
            ]
        );

        // The flag alone, with no config file to contradict, leaves the rest at their
        // defaults — --draft only swaps which drafter speculates.
        let cli = CliOverrides {
            draft: Some(PathBuf::from("/only-cli.gguf")),
            ..model_only()
        };
        let (settings, warnings) = resolve(&ServeToml::default(), None, &cli).unwrap();
        assert_eq!(
            settings.draft,
            DraftMode::Custom(PathBuf::from("/only-cli.gguf"))
        );
        assert_eq!(settings.draft_max, None);
        assert!(warnings.is_empty());
    }

    /// Speculation is opt-out: it needs no request at all, and every way of
    /// making one anyway still selects the drafter it names.
    ///
    /// The default is on because drafting measured faster on both checkpoints
    /// once the fused verify landed — so a zero-flag run asks for a drafter.
    #[test]
    fn speculation_is_on_unless_something_declines_it() {
        // Nothing named, nothing enabled: the official drafter, by default.
        let empty: ServeToml = toml::from_str("").unwrap();
        let (settings, warnings) = resolve(&empty, None, &model_only()).unwrap();
        assert_eq!(settings.draft, DraftMode::Official);
        assert!(warnings.is_empty());

        // `enabled = true` restates the default and means the same drafter.
        let file: ServeToml = toml::from_str("[draft]\nenabled = true\n").unwrap();
        let (settings, _) = resolve(&file, None, &model_only()).unwrap();
        assert_eq!(settings.draft, DraftMode::Official);

        // A config path picks a custom drafter, and enables on its own whichever
        // way the default is set.
        let file: ServeToml = toml::from_str("[draft]\npath = \"/d.gguf\"\n").unwrap();
        let (settings, _) = resolve(&file, None, &model_only()).unwrap();
        assert_eq!(settings.draft, DraftMode::Custom(PathBuf::from("/d.gguf")));

        // So is `--draft` on the command line.
        let cli = CliOverrides {
            draft: Some(PathBuf::from("/cli.gguf")),
            draft_enabled: Some(true),
            ..model_only()
        };
        let (settings, _) = resolve(&empty, None, &cli).unwrap();
        assert_eq!(
            settings.draft,
            DraftMode::Custom(PathBuf::from("/cli.gguf"))
        );
    }

    /// `--no-draft` and `[draft] enabled = false` both resolve to no drafter,
    /// even over a config-named path, and the CLI switch wins over an explicit
    /// `enabled = true` with a warning. These are the opt-out: with speculation
    /// on by default they are the only way to decode plain.
    #[test]
    fn no_draft_disables_speculation() {
        let file: ServeToml = toml::from_str("[draft]\nenabled = false\n").unwrap();
        let (settings, warnings) = resolve(&file, None, &model_only()).unwrap();
        assert_eq!(settings.draft, DraftMode::Off);
        assert!(warnings.is_empty());

        // The switch beats a config that names a drafter path: enabled is the
        // master, path only picks which drafter.
        let file: ServeToml = toml::from_str("[draft]\npath = \"/d.gguf\"\n").unwrap();
        let cli = CliOverrides {
            draft_enabled: Some(false),
            ..model_only()
        };
        let (settings, warnings) =
            resolve(&file, Some(Path::new("/etc/serve.toml")), &cli).unwrap();
        assert_eq!(settings.draft, DraftMode::Off);
        assert!(warnings.is_empty());

        let file: ServeToml = toml::from_str("[draft]\nenabled = true\n").unwrap();
        let (settings, warnings) =
            resolve(&file, Some(Path::new("/etc/serve.toml")), &cli).unwrap();
        assert_eq!(settings.draft, DraftMode::Off);
        assert_eq!(
            warnings,
            vec![
                "warning: --no-draft overrides config draft.enabled = true (/etc/serve.toml)"
                    .to_string()
            ]
        );
    }
    /// Naming a drafter on the CLI is itself a request for one: `--draft <path>`
    /// (which the CLI layer turns into `draft_enabled: Some(true)`) beats a
    /// config file's `enabled = false`, with a warning naming the flag that won.
    #[test]
    fn an_explicit_cli_draft_beats_config_disabled() {
        let file: ServeToml = toml::from_str("[draft]\nenabled = false\n").unwrap();
        let cli = CliOverrides {
            draft: Some(PathBuf::from("/custom.gguf")),
            draft_enabled: Some(true),
            ..model_only()
        };
        let (settings, warnings) =
            resolve(&file, Some(Path::new("/etc/serve.toml")), &cli).unwrap();
        assert_eq!(
            settings.draft,
            DraftMode::Custom(PathBuf::from("/custom.gguf"))
        );
        assert_eq!(
            warnings,
            vec![
                "warning: --draft overrides config draft.enabled = false (/etc/serve.toml)"
                    .to_string()
            ]
        );
    }

    /// A drafter cache of zero positions could never be injected into, so it is
    /// refused rather than read as speculation that silently never runs.
    #[test]
    fn zero_draft_ctx_is_an_error() {
        let file: ServeToml = toml::from_str("[draft]\nctx = 0\n").unwrap();
        let err = resolve(&file, None, &model_only()).unwrap_err();
        assert!(err.to_string().contains("draft_ctx"), "{err}");

        let cli = CliOverrides {
            draft_ctx: Some(0),
            ..model_only()
        };
        let err = resolve(&ServeToml::default(), None, &cli).unwrap_err();
        assert!(err.to_string().contains("draft_ctx"), "{err}");

        // One position is legal, if useless: the cap is not the server's to judge.
        let cli = CliOverrides {
            draft_ctx: Some(1),
            ..model_only()
        };
        let (settings, _) = resolve(&ServeToml::default(), None, &cli).unwrap();
        assert_eq!(settings.draft_ctx, 1);
    }

    /// The other drafter knobs are range-checked for the same reason `draft_ctx` is: each
    /// has values that leave speculation configured, loaded and announced, and then never
    /// running. That is refused rather than warned about.
    #[test]
    fn draft_knobs_outside_their_range_are_errors() {
        let refuses = |cli: CliOverrides, wanted: &str| {
            let err = resolve(&ServeToml::default(), None, &cli).unwrap_err();
            assert!(
                err.to_string().contains(wanted),
                "expected {wanted:?}: {err}"
            );
        };

        // Zero drafts nothing, so every round would fall back to plain decode.
        refuses(
            CliOverrides {
                draft_max: Some(0),
                ..model_only()
            },
            "draft.max",
        );
        // A probability threshold above 1 no draft can ever clear.
        refuses(
            CliOverrides {
                draft_p_min: Some(1.5),
                ..model_only()
            },
            "draft.p_min",
        );
        refuses(
            CliOverrides {
                draft_p_min: Some(-0.1),
                ..model_only()
            },
            "draft.p_min",
        );
        refuses(
            CliOverrides {
                draft_p_min: Some(f32::NAN),
                ..model_only()
            },
            "draft.p_min",
        );
        // A negative margin holds speculation paused forever.
        refuses(
            CliOverrides {
                draft_pause_margin: Some(-1.0),
                ..model_only()
            },
            "draft.pause_margin",
        );

        // The edges of each range are legal, and mean something: p_min 0 keeps every draft
        // and proposes full blocks, pause_margin 0 disables pausing, p_min 1 drafts only
        // what the drafter is certain of.
        let cli = CliOverrides {
            draft_max: Some(1),
            draft_p_min: Some(0.0),
            draft_pause_margin: Some(0.0),
            ..model_only()
        };
        let (settings, warnings) = resolve(&ServeToml::default(), None, &cli).unwrap();
        assert_eq!(settings.draft_max, Some(1));
        assert_eq!(settings.draft_p_min, Some(0.0));
        assert_eq!(settings.draft_pause_margin, 0.0);
        assert!(warnings.is_empty());

        let cli = CliOverrides {
            draft_p_min: Some(1.0),
            ..model_only()
        };
        assert_eq!(
            resolve(&ServeToml::default(), None, &cli)
                .unwrap()
                .0
                .draft_p_min,
            Some(1.0)
        );
    }

    /// The scheduler settings follow the same CLI-over-file-over-default
    /// precedence as everything else, and warn when a flag contradicts the file.
    #[test]
    fn scheduler_settings_follow_the_usual_precedence() {
        let file: ServeToml = toml::from_str(
            r#"
            queue_capacity = 4
            queue_timeout = 60
            request_prefill_rate = 300
            request_decode_rate = 20
            request_slack = 10
            schedule = "fifo"
            schedule_age_limit = 5
            "#,
        )
        .unwrap();
        let (from_file, warnings) =
            resolve(&file, Some(Path::new("/etc/serve.toml")), &model_only()).unwrap();
        assert!(warnings.is_empty());
        assert_eq!(from_file.queue_capacity, 4);
        assert_eq!(from_file.queue_timeout, Duration::from_secs(60));
        assert_eq!(from_file.request_prefill_rate, 300);
        assert_eq!(from_file.request_decode_rate, 20);
        assert_eq!(from_file.request_slack, Duration::from_secs(10));
        assert_eq!(from_file.schedule, Schedule::Fifo);
        assert_eq!(from_file.schedule_age_limit, Duration::from_secs(5));

        let cli = CliOverrides {
            queue_capacity: Some(2),
            schedule: Some("shortest-prefill".into()),
            ..model_only()
        };
        let (merged, warnings) = resolve(&file, Some(Path::new("/etc/serve.toml")), &cli).unwrap();
        assert_eq!(merged.queue_capacity, 2);
        assert_eq!(merged.schedule, Schedule::ShortestPrefill);
        // Untouched keys still come from the file.
        assert_eq!(merged.request_decode_rate, 20);
        assert_eq!(
            warnings,
            vec![
                "warning: --queue-capacity 2 overrides config queue_capacity = 4 (/etc/serve.toml)"
                    .to_string(),
                "warning: --schedule shortest-prefill overrides config schedule = fifo (/etc/serve.toml)"
                    .to_string(),
            ]
        );
    }

    /// A watchdog rate of 0 disables the deadline and is legal; the queue and
    /// aging knobs have no meaningful zero and are refused instead.
    #[test]
    fn scheduler_knobs_outside_their_range_are_errors() {
        let refuses = |cli: CliOverrides, wanted: &str| {
            let err = resolve(&ServeToml::default(), None, &cli).unwrap_err();
            assert!(
                err.to_string().contains(wanted),
                "expected {wanted:?}: {err}"
            );
        };
        refuses(
            CliOverrides {
                queue_capacity: Some(0),
                ..model_only()
            },
            "queue_capacity",
        );
        refuses(
            CliOverrides {
                queue_timeout: Some(0),
                ..model_only()
            },
            "queue_timeout",
        );
        refuses(
            CliOverrides {
                schedule_age_limit: Some(0),
                ..model_only()
            },
            "schedule_age_limit",
        );

        // Rates at 0 mean "no deadline", and slack 0 is a deadline with no
        // fixed allowance: both resolve.
        let cli = CliOverrides {
            request_prefill_rate: Some(0),
            request_decode_rate: Some(0),
            request_slack: Some(0),
            ..model_only()
        };
        let (settings, warnings) = resolve(&ServeToml::default(), None, &cli).unwrap();
        assert_eq!(settings.request_prefill_rate, 0);
        assert_eq!(settings.request_decode_rate, 0);
        assert_eq!(settings.request_slack, Duration::ZERO);
        assert!(warnings.is_empty());
    }

    #[test]
    fn an_unknown_schedule_names_the_valid_values() {
        let file: ServeToml = toml::from_str("schedule = \"round-robin\"\n").unwrap();
        let err = resolve(&file, None, &model_only()).unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("config schedule"), "{text}");
        assert!(
            text.contains("shortest-prefill") && text.contains("fifo"),
            "{text}"
        );

        let cli = CliOverrides {
            schedule: Some("FIFO".into()),
            ..model_only()
        };
        let err = resolve(&ServeToml::default(), None, &cli).unwrap_err();
        assert!(format!("{err:#}").contains("--schedule"), "{err:#}");
    }

    #[test]
    fn schedules_render_back_into_config_syntax() {
        assert_eq!(Schedule::ShortestPrefill.to_string(), "shortest-prefill");
        assert_eq!(Schedule::Fifo.to_string(), "fifo");
        assert_eq!(parse_schedule(" fifo ").unwrap(), Schedule::Fifo);
        assert_eq!(
            parse_schedule("shortest-prefill").unwrap(),
            Schedule::ShortestPrefill
        );
    }

    /// A slot count of zero leaves the conversation being served nowhere to
    /// live, so it is refused rather than silently read as 1.
    #[test]
    fn zero_cache_slots_is_an_error() {
        let file: ServeToml = toml::from_str("[cache]\nslots = 0\n").unwrap();
        let err = resolve(&file, None, &model_only()).unwrap_err();
        assert!(err.to_string().contains("cache_slots"), "{err}");

        // One slot is the documented way to turn multi-conversation reuse off.
        let cli = CliOverrides {
            cache_slots: Some(1),
            ..model_only()
        };
        let (settings, warnings) = resolve(&ServeToml::default(), None, &cli).unwrap();
        assert_eq!(settings.cache_slots, 1);
        assert!(warnings.is_empty());
    }

    /// The slot count follows the same CLI-over-file-over-default precedence as
    /// every other setting, and says so when a flag contradicts the file.
    #[test]
    fn cache_slots_follows_the_usual_precedence() {
        let file: ServeToml = toml::from_str("[cache]\nslots = 8\n").unwrap();
        let (from_file, warnings) =
            resolve(&file, Some(Path::new("/etc/serve.toml")), &model_only()).unwrap();
        assert_eq!(from_file.cache_slots, 8);
        assert!(warnings.is_empty());

        let cli = CliOverrides {
            cache_slots: Some(2),
            ..model_only()
        };
        let (merged, warnings) = resolve(&file, Some(Path::new("/etc/serve.toml")), &cli).unwrap();
        assert_eq!(merged.cache_slots, 2);
        assert_eq!(
            warnings,
            vec![
                "warning: --cache-slots 2 overrides config cache.slots = 8 (/etc/serve.toml)"
                    .to_string()
            ]
        );
    }

    /// The disk tier's four settings follow the same precedence as everything
    /// else, and `--no-disk-cache` is the second setting with a switch each way —
    /// so the warning has to name whichever flag was passed, not one of them.
    #[test]
    fn disk_cache_settings_follow_the_usual_precedence() {
        let file: ServeToml = toml::from_str(
            r#"
            cache_dir = "/from-config/cache"
            disk_cache = true
            disk_max_gib = 8
            disk_min_tokens = 256
            "#,
        )
        .unwrap();
        let (from_file, warnings) =
            resolve(&file, Some(Path::new("/etc/serve.toml")), &model_only()).unwrap();
        assert!(warnings.is_empty());
        assert_eq!(
            from_file.cache_dir,
            Some(PathBuf::from("/from-config/cache"))
        );
        assert!(from_file.disk_cache);
        assert_eq!(from_file.disk_max_gib, 8);
        assert_eq!(from_file.disk_min_tokens, 256);

        let cli = CliOverrides {
            cache_dir: Some(PathBuf::from("/from-cli/cache")),
            disk_cache: Some(false),
            disk_max_gib: Some(16),
            disk_min_tokens: Some(4096),
            ..model_only()
        };
        let (merged, warnings) = resolve(&file, Some(Path::new("/etc/serve.toml")), &cli).unwrap();
        assert_eq!(merged.cache_dir, Some(PathBuf::from("/from-cli/cache")));
        assert!(!merged.disk_cache);
        assert_eq!(merged.disk_max_gib, 16);
        assert_eq!(merged.disk_min_tokens, 4096);
        assert_eq!(
            warnings,
            vec![
                "warning: --cache-dir /from-cli/cache overrides config cache_dir = /from-config/cache (/etc/serve.toml)".to_string(),
                "warning: --no-disk-cache overrides config disk_cache = true (/etc/serve.toml)"
                    .to_string(),
                "warning: --disk-max-gib 16 overrides config disk_max_gib = 8 (/etc/serve.toml)"
                    .to_string(),
                "warning: --disk-min-tokens 4096 overrides config disk_min_tokens = 256 (/etc/serve.toml)"
                    .to_string(),
            ]
        );

        // The enabling flag over a config that turned it off names itself.
        let off: ServeToml = toml::from_str("disk_cache = false\n").unwrap();
        let (settings, warnings) = resolve(
            &off,
            Some(Path::new("/etc/serve.toml")),
            &CliOverrides {
                disk_cache: Some(true),
                ..model_only()
            },
        )
        .unwrap();
        assert!(settings.disk_cache);
        assert_eq!(
            warnings,
            vec![
                "warning: --disk-cache overrides config disk_cache = false (/etc/serve.toml)"
                    .to_string()
            ]
        );
    }

    #[test]
    fn missing_model_is_an_error() {
        let err = resolve(&ServeToml::default(), None, &CliOverrides::default()).unwrap_err();
        assert!(err.to_string().contains("--model"), "{err}");
    }

    #[test]
    fn file_beats_default_and_cli_beats_file() {
        let file: ServeToml = toml::from_str(
            r#"
            model = "/from-config.gguf"
            port = 9000
            host = "0.0.0.0"
            idle_unload = "off"
            [sampling]
            temperature = 0.5
            "#,
        )
        .unwrap();

        let (from_file, warnings) = resolve(
            &file,
            Some(Path::new("/etc/serve.toml")),
            &CliOverrides::default(),
        )
        .unwrap();
        assert!(warnings.is_empty());
        assert_eq!(from_file.model, PathBuf::from("/from-config.gguf"));
        assert_eq!(from_file.port, 9000);
        assert_eq!(from_file.host, "0.0.0.0");
        assert_eq!(from_file.idle_unload, None);
        assert_eq!(from_file.temperature, Some(0.5));
        // An untouched sampling key stays absent — resolved per request.
        assert_eq!(from_file.top_k, None);

        let cli = CliOverrides {
            port: Some(8080),
            idle_unload: Some("90s".into()),
            temperature: Some(0.7),
            ..Default::default()
        };
        let (merged, warnings) = resolve(&file, Some(Path::new("/etc/serve.toml")), &cli).unwrap();
        assert_eq!(merged.port, 8080);
        assert_eq!(merged.idle_unload, Some(Duration::from_secs(90)));
        assert_eq!(merged.temperature, Some(0.7));
        // The config's model survives: no --model was passed.
        assert_eq!(merged.model, PathBuf::from("/from-config.gguf"));
        assert_eq!(
            warnings,
            vec![
                "warning: --port 8080 overrides config port = 9000 (/etc/serve.toml)".to_string(),
                "warning: --idle-unload 90s overrides config idle_unload = off (/etc/serve.toml)"
                    .to_string(),
                "warning: --temp 0.7 overrides config sampling.temperature = 0.5 (/etc/serve.toml)"
                    .to_string(),
            ]
        );
    }

    #[test]
    fn a_flag_matching_the_config_warns_about_nothing() {
        let file: ServeToml = toml::from_str("model = \"/m.gguf\"\nport = 8080\n").unwrap();
        let cli = CliOverrides {
            port: Some(8080),
            ..Default::default()
        };
        let (settings, warnings) =
            resolve(&file, Some(Path::new("/etc/serve.toml")), &cli).unwrap();
        assert_eq!(settings.port, 8080);
        assert!(warnings.is_empty());
    }

    #[test]
    fn disable_flags_override_an_enabling_config() {
        let file: ServeToml = toml::from_str("anthropic = true\nopenai = false\n").unwrap();
        let cli = CliOverrides {
            anthropic: Some(false),
            openai: Some(false),
            ..model_only()
        };
        let (settings, warnings) =
            resolve(&file, Some(Path::new("/etc/serve.toml")), &cli).unwrap();
        assert!(!settings.anthropic && !settings.openai);
        // Only the contradicted one warns; --no-openai agrees with the config.
        assert_eq!(
            warnings,
            vec![
                "warning: --no-anthropic overrides config anthropic = true (/etc/serve.toml)"
                    .to_string()
            ]
        );
    }

    /// The dashboard has a switch each way, so a config file that turns it on
    /// can be overridden for a single run — and the warning names whichever
    /// flag was actually passed.
    #[test]
    fn either_dashboard_flag_overrides_the_config() {
        let on: ServeToml = toml::from_str("tui = true\n").unwrap();
        let (settings, warnings) = resolve(
            &on,
            Some(Path::new("/etc/serve.toml")),
            &CliOverrides {
                tui: Some(false),
                ..model_only()
            },
        )
        .unwrap();
        assert!(!settings.tui);
        assert_eq!(
            warnings,
            vec!["warning: --no-tui overrides config tui = true (/etc/serve.toml)".to_string()]
        );

        let off: ServeToml = toml::from_str("tui = false\n").unwrap();
        let (settings, warnings) = resolve(
            &off,
            Some(Path::new("/etc/serve.toml")),
            &CliOverrides {
                tui: Some(true),
                ..model_only()
            },
        )
        .unwrap();
        assert!(settings.tui);
        assert_eq!(
            warnings,
            vec!["warning: --tui overrides config tui = false (/etc/serve.toml)".to_string()]
        );
    }

    #[test]
    fn api_key_override_does_not_leak_either_key() {
        let file: ServeToml = toml::from_str("api_key = \"file-secret\"\n").unwrap();
        let cli = CliOverrides {
            api_key: Some("cli-secret".into()),
            ..model_only()
        };
        let (settings, warnings) =
            resolve(&file, Some(Path::new("/etc/serve.toml")), &cli).unwrap();
        assert_eq!(settings.api_key.as_deref(), Some("cli-secret"));
        assert_eq!(
            warnings,
            vec!["warning: --api-key overrides config api_key (/etc/serve.toml)".to_string()]
        );
    }

    #[test]
    fn tools_mode_follows_cli_over_file_over_default() {
        let file: ServeToml = toml::from_str("tools_mode = \"strip\"\n").unwrap();
        let (from_file, warnings) =
            resolve(&file, Some(Path::new("/etc/serve.toml")), &model_only()).unwrap();
        assert_eq!(from_file.tools_mode, ToolsMode::Strip);
        assert!(warnings.is_empty());

        let cli = CliOverrides {
            tools_mode: Some("reject".into()),
            ..model_only()
        };
        let (merged, warnings) = resolve(&file, Some(Path::new("/etc/serve.toml")), &cli).unwrap();
        assert_eq!(merged.tools_mode, ToolsMode::Reject);
        assert_eq!(
            warnings,
            vec![
                "warning: --tools-mode reject overrides config tools_mode = strip (/etc/serve.toml)"
                    .to_string()
            ]
        );

        // The flag alone, with no config file to contradict.
        let cli = CliOverrides {
            tools_mode: Some("strip".into()),
            ..model_only()
        };
        let (settings, warnings) = resolve(&ServeToml::default(), None, &cli).unwrap();
        assert_eq!(settings.tools_mode, ToolsMode::Strip);
        assert!(warnings.is_empty());
    }

    #[test]
    fn an_unknown_tools_mode_names_the_valid_values() {
        let file: ServeToml = toml::from_str("tools_mode = \"ignore\"\n").unwrap();
        let err = resolve(&file, None, &model_only()).unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("config tools_mode"), "{text}");
        assert!(
            text.contains("native") && text.contains("reject") && text.contains("strip"),
            "{text}"
        );

        let cli = CliOverrides {
            tools_mode: Some("Strip".into()),
            ..model_only()
        };
        let err = resolve(&ServeToml::default(), None, &cli).unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("--tools-mode"), "{text}");
    }

    #[test]
    fn tools_modes_render_back_into_config_syntax() {
        assert_eq!(ToolsMode::Native.to_string(), "native");
        assert_eq!(ToolsMode::Reject.to_string(), "reject");
        assert_eq!(ToolsMode::Strip.to_string(), "strip");
        assert_eq!(parse_tools_mode(" strip ").unwrap(), ToolsMode::Strip);
        assert_eq!(parse_tools_mode("native").unwrap(), ToolsMode::Native);
    }

    #[test]
    fn thinking_budget_zero_means_uncapped() {
        let file: ServeToml =
            toml::from_str("[thinking]\nforce = false\ndefault_budget = 2048\n").unwrap();
        let (settings, _) = resolve(&file, None, &model_only()).unwrap();
        assert!(!settings.thinking_force);
        assert_eq!(settings.thinking_budget, Some(2048));

        let file: ServeToml = toml::from_str("[thinking]\ndefault_budget = 0\n").unwrap();
        let (settings, _) = resolve(&file, None, &model_only()).unwrap();
        assert_eq!(settings.thinking_budget, None);
    }

    /// `[thinking] effort` merges like every other key — flag beats file, with
    /// a warning when they disagree — and both spellings are parsed during the
    /// merge so the error names its source. Absent stays absent, which is the
    /// template's own default level.
    #[test]
    fn thinking_effort_merges_flag_over_file() {
        let file: ServeToml = toml::from_str("[thinking]\neffort = \"low\"\n").unwrap();
        let (settings, warnings) = resolve(&file, None, &model_only()).unwrap();
        assert!(warnings.is_empty());
        assert_eq!(settings.reasoning_effort, Some(ReasoningEffort::Low));

        let cli = CliOverrides {
            reasoning_effort: Some("medium".into()),
            ..model_only()
        };
        let (settings, warnings) =
            resolve(&file, Some(Path::new("/etc/serve.toml")), &cli).unwrap();
        assert_eq!(settings.reasoning_effort, Some(ReasoningEffort::Medium));
        assert_eq!(
            warnings,
            vec![
                "warning: --reasoning-effort medium overrides config thinking.effort = low \
                 (/etc/serve.toml)"
                    .to_string()
            ]
        );

        // A level neither the template nor the flag defines is refused, naming
        // the side it came from.
        let err = resolve(
            &file,
            None,
            &CliOverrides {
                reasoning_effort: Some("high".into()),
                ..model_only()
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("--reasoning-effort"), "{err}");
        let bad: ServeToml = toml::from_str("[thinking]\neffort = \"maximum\"\n").unwrap();
        let err = resolve(&bad, None, &model_only()).unwrap_err();
        assert!(err.to_string().contains("thinking.effort"), "{err}");
    }

    /// The sampling keys have no built-in fallback: a value set in the file or
    /// on a flag is pinned for both modes, and an absent one stays `None` for
    /// the dialects to resolve against each request's own thinking mode.
    #[test]
    fn sampling_keys_stay_absent_unless_pinned() {
        let (settings, _) = resolve(&ServeToml::default(), None, &model_only()).unwrap();
        assert_eq!(
            (settings.temperature, settings.top_k, settings.top_p),
            (None, None, None)
        );
        assert_eq!(settings.presence_penalty, None);

        let file: ServeToml =
            toml::from_str("[sampling]\ntop_k = 40\ntop_p = 0.9\npresence_penalty = 1.5\n")
                .unwrap();
        let (settings, warnings) = resolve(&file, None, &model_only()).unwrap();
        assert!(warnings.is_empty());
        assert_eq!(settings.temperature, None);
        assert_eq!(settings.top_k, Some(40));
        assert_eq!(settings.top_p, Some(0.9));
        assert_eq!(settings.presence_penalty, Some(1.5));

        // A zero is a pin, not an absence: it is how an operator turns the
        // checkpoint's own default off server-wide.
        let off: ServeToml = toml::from_str("[sampling]\npresence_penalty = 0.0\n").unwrap();
        let (settings, _) = resolve(&off, None, &model_only()).unwrap();
        assert_eq!(settings.presence_penalty, Some(0.0));

        // And the flag beats the file, like every other sampling key.
        let cli = CliOverrides {
            presence_penalty: Some(0.25),
            ..model_only()
        };
        let (settings, _) = resolve(&file, None, &cli).unwrap();
        assert_eq!(settings.presence_penalty, Some(0.25));
    }

    #[test]
    fn duration_parser_accepts_units_and_off() {
        assert_eq!(
            parse_idle_unload("30s").unwrap().0,
            Some(Duration::from_secs(30))
        );
        assert_eq!(
            parse_idle_unload("5m").unwrap().0,
            Some(Duration::from_secs(300))
        );
        assert_eq!(
            parse_idle_unload("2h").unwrap().0,
            Some(Duration::from_secs(7200))
        );
        assert_eq!(
            parse_idle_unload(" 5m ").unwrap().0,
            Some(Duration::from_secs(300))
        );
        assert_eq!(parse_idle_unload("off").unwrap().0, None);
        assert_eq!(parse_idle_unload("OFF").unwrap().0, None);
    }

    #[test]
    fn duration_parser_rejects_the_ambiguous_forms() {
        for bad in ["", "5", "5 m", "m", "-1m", "1.5h", "5d", "off5", "never"] {
            assert!(parse_idle_unload(bad).is_err(), "{bad:?} should not parse");
        }
        // Zero is ambiguous between "never unload" and "unload at once".
        let err = parse_idle_unload("0s").unwrap_err();
        assert!(err.to_string().contains("off"), "{err}");
    }

    /// A count that overflows the conversion to seconds is an error, not the
    /// short wait the wrapped product would silently become.
    #[test]
    fn duration_parser_rejects_an_overflowing_count() {
        let err = parse_idle_unload(&format!("{}h", u64::MAX)).unwrap_err();
        assert!(err.to_string().contains("off"), "{err}");
        let err = parse_idle_unload(&format!("{}m", u64::MAX / 59)).unwrap_err();
        assert!(err.to_string().contains("longer than"), "{err}");
        // The largest value that does fit still parses.
        assert_eq!(
            parse_idle_unload(&format!("{}s", u64::MAX))
                .unwrap()
                .0
                .unwrap()
                .as_secs(),
            u64::MAX
        );
    }

    #[test]
    fn an_empty_api_key_configures_no_authentication_and_says_so() {
        let file: ServeToml = toml::from_str("api_key = \"\"\n").unwrap();
        let (settings, warnings) = resolve(&file, None, &model_only()).unwrap();
        assert_eq!(settings.api_key, None);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("api_key is empty"), "{warnings:?}");
        // Whitespace is no more of a key than nothing at all.
        let cli = CliOverrides {
            api_key: Some("   ".into()),
            ..model_only()
        };
        let (settings, warnings) = resolve(&ServeToml::default(), None, &cli).unwrap();
        assert_eq!(settings.api_key, None);
        assert_eq!(warnings.len(), 1, "{warnings:?}");

        // A real key survives untouched, warning about nothing.
        let cli = CliOverrides {
            api_key: Some("secret".into()),
            ..model_only()
        };
        let (settings, warnings) = resolve(&ServeToml::default(), None, &cli).unwrap();
        assert_eq!(settings.api_key.as_deref(), Some("secret"));
        assert!(warnings.is_empty());
    }

    #[test]
    fn durations_render_back_into_config_syntax() {
        assert_eq!(IdleUnload(None).to_string(), "off");
        assert_eq!(IdleUnload(Some(Duration::from_secs(90))).to_string(), "90s");
        assert_eq!(IdleUnload(Some(Duration::from_secs(300))).to_string(), "5m");
        assert_eq!(
            IdleUnload(Some(Duration::from_secs(7200))).to_string(),
            "2h"
        );
    }

    #[test]
    fn unknown_keys_are_rejected_with_the_offending_name() {
        let err = toml::from_str::<ServeToml>("prot = 5241\n").unwrap_err();
        assert!(err.to_string().contains("prot"), "{err}");

        let err = toml::from_str::<ServeToml>("[sampling]\ntop_kk = 20\n").unwrap_err();
        assert!(err.to_string().contains("top_kk"), "{err}");

        let err = toml::from_str::<ServeToml>("[draft]\nmodel = \"/d.gguf\"\n").unwrap_err();
        assert!(err.to_string().contains("model"), "{err}");
    }

    #[test]
    fn init_template_parses_and_resolves_to_the_defaults() {
        let parsed: ServeToml = toml::from_str(&init_template()).expect("template is valid TOML");
        let (from_template, warnings) = resolve(&parsed, None, &model_only()).unwrap();
        assert!(warnings.is_empty());
        assert_eq!(from_template, defaults());
        // `model`, `api_key`, the drafter path and the cache directory have no
        // baked default — the last resolves from $HOME — so the template leaves
        // them commented out rather than shipping a value that cannot work or a
        // path that would be wrong on another machine.
        assert_eq!(parsed.model, None);
        assert_eq!(parsed.api_key, None);
        assert_eq!(parsed.draft.path, None);
        assert_eq!(parsed.cache_dir, None);
        assert_eq!(parsed.disk_cache, Some(DEFAULT_DISK_CACHE));
    }
}
