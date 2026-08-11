mod repl;

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};

use xwen::batch::{BatchRequest, BatchResponse};
use xwen::chat::{ChatOptions, Message, build_prompt_with_spans};
use xwen::config::XwenConfig;
use xwen::dflash::DflashDrafter;
use xwen::generate::{Generator, SpecParams};
use xwen::gguf;
use xwen::hub::Model;
use xwen::ops::ExpertRunner;
use xwen::sampler::SamplerOptions;
use xwen::serve::config::{CliOverrides, ServeToml};

#[derive(Parser)]
#[command(
    name = "xwen",
    about = "Qwen 3.6 inference on Metal. Bare `xwen` serves over HTTP \
             with the live dashboard; subcommands cover everything else.",
    args_conflicts_with_subcommands = true
)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    /// A bare `xwen` is `xwen serve`, so serve's flags parse at the top
    /// level too (`xwen --no-tui`, `xwen --port 8080`).
    #[command(flatten)]
    serve: ServeArgs,
}

/// Which official checkpoint to run.
#[derive(Parser)]
struct ModelArgs {
    /// Which official Qwen 3.6 checkpoint to run: the dense 27B or the
    /// 35B-A3B MoE. A `--model <gguf>` path overrides the target file
    /// outright; for the one-shot commands this flag still selects the family
    /// (and so the DFlash drafter sidecar), while `xwen serve` reads the
    /// family from the GGUF itself and uses the flag only to pick the default
    /// file when nothing else names one.
    #[arg(long, value_name = "27b|35b", default_value_t = Model::default())]
    model_size: Model,
}

/// Shared sampling knobs (generation_config.json defaults: temp 1.0, top-p
/// 0.95, top-k 20).
#[derive(Parser)]
struct SamplingArgs {
    #[arg(long, default_value_t = 1.0)]
    temp: f64,
    #[arg(long, default_value_t = 20)]
    top_k: usize,
    #[arg(long, default_value_t = 0.95)]
    top_p: f64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

impl SamplingArgs {
    fn options(&self) -> SamplerOptions {
        SamplerOptions {
            temperature: self.temp,
            top_k: self.top_k,
            top_p: self.top_p,
            seed: self.seed,
        }
    }
}

/// DFlash speculative-decode knobs. Speculation is opt-OUT on xwen: decoding
/// speculates with the checkpoint's official drafter unless `--no-draft` says
/// otherwise, and `--draft <gguf>` swaps in a custom one.
#[derive(Parser)]
struct DraftArgs {
    /// Speculate with a custom DFlash drafter GGUF instead of the checkpoint's
    /// official one (which the literal `official`, and the default, select).
    #[arg(long, value_name = "GGUF", conflicts_with = "no_draft")]
    draft: Option<PathBuf>,
    /// Decode without speculation. Drafting is on by default: measured faster on
    /// both checkpoints on 2026-07-29 — 27B +19.3 to +21.0% on code and +7.6 to
    /// +8.4% on chat, 35B-A3B +18.1 to +19.8% on code and +12.6 to +12.8% on
    /// chat (see docs/decisions.md, "Speculative decoding").
    #[arg(long)]
    no_draft: bool,
    /// Max draft tokens proposed per verify round (clamped to block_size-1).
    #[arg(long, default_value_t = 15)]
    draft_max: usize,
    /// Discard a round's whole draft if fewer than this many are collected.
    #[arg(long, default_value_t = 0)]
    draft_min: usize,
    /// Stop drafting at the first token whose full-vocab softmax prob is below
    /// this. Adaptive draft length; the default is per-model — 0.5 for 27b, 0.3
    /// for 35b, fitted 2026-08-08.
    #[arg(long)]
    draft_p_min: Option<f32>,
    /// Auto-pause speculation when its wall-clock cost per committed token
    /// exceeds a plain decode step's cost times this factor (keeps `--draft`
    /// from losing to plain decode on low-acceptance text). With auto-pause on,
    /// temperature>0 runs are not run-to-run reproducible for a fixed seed
    /// (which rounds batch-verify depends on wall-clock timing, and batched
    /// rounds differ from plain at near-ties); `0` disables auto-pause (always
    /// draft) and restores fully deterministic fixed-seed behavior.
    #[arg(long, default_value_t = 1.0)]
    draft_pause_margin: f32,
    /// Positions the drafter's KV cache is sized for (capped at --max-ctx),
    /// which is also the drafting depth limit: past this many tokens of
    /// context, decode continues plain. The drafter forward's cost grows
    /// linearly with depth while its proposal quality collapses, so beyond
    /// the default drafting is a pure loss on every text.
    #[arg(long, value_name = "TOKENS", default_value_t = xwen::dflash::DEFAULT_DRAFT_CTX)]
    draft_ctx: usize,
}

impl DraftArgs {
    /// `size` supplies the drafting floor for a run that did not pass
    /// `--draft-p-min`: it is fitted per checkpoint, so the flag's default
    /// cannot live on the flag itself.
    fn params(&self, size: Model) -> SpecParams {
        SpecParams {
            draft_max: self.draft_max,
            draft_min: self.draft_min,
            draft_p_min: self.draft_p_min.unwrap_or(size.draft_p_min_default()),
            pause_margin: self.draft_pause_margin,
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Ensure the official model + DFlash drafter are in the Hugging Face
    /// cache (idempotent: anything already cached is not touched), then print
    /// their paths. Every command does this lazily for whatever it needs; this
    /// just prefetches.
    Fetch {
        #[command(flatten)]
        select: ModelArgs,
    },
    /// Dump GGUF metadata and tensor listing.
    Inspect {
        /// Model GGUF (default: the official checkpoint, ensured in the Hugging
        /// Face cache — downloaded on first use, cached forever after).
        #[arg(short, long)]
        model: Option<PathBuf>,
        #[command(flatten)]
        select: ModelArgs,
    },
    /// One-shot generation from a prompt.
    Generate {
        /// Model GGUF (default: the official checkpoint, ensured in the Hugging
        /// Face cache — downloaded on first use, cached forever after).
        #[arg(short, long)]
        model: Option<PathBuf>,
        #[command(flatten)]
        select: ModelArgs,
        #[arg(short, long)]
        prompt: String,
        /// Custom tokenizer.json (default: the checkpoint tokenizer embedded
        /// in the binary).
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(short = 'n', long, default_value_t = 512)]
        max_tokens: usize,
        #[arg(long, default_value = "fused")]
        moe_impl: String,
        #[arg(long, default_value_t = 8192)]
        max_ctx: usize,
        /// Skip the chat template and feed the prompt raw (no BOS — Qwen never
        /// prepends one).
        #[arg(long)]
        raw: bool,
        #[arg(long)]
        stats: bool,
        /// Force at least this many decode tokens of <think> reasoning before
        /// `</think>` (or an EOG) may be sampled; 0 lets the model decide.
        /// Meaningful only without --raw (the chat template ends the prompt
        /// inside an open <think> block).
        #[arg(long, default_value_t = 0)]
        min_think: usize,
        /// Steer the model out of its <think> block within about this many
        /// decode tokens: wait words are biased down from 70% of the budget,
        /// `</think>` is pulled up from 80%, and past the budget a transition
        /// sentence plus `</think>` are injected at the next sentence boundary.
        /// 0 lets the model reason as long as it likes.
        #[arg(long, default_value_t = 0)]
        max_think: usize,
        /// Ban every token whose text contains this string (repeatable).
        /// Byte-level BPE can still spell a banned glyph out of raw byte
        /// tokens, so this suppresses the common spellings, not every possible
        /// one. Structural tokens are never banned, and the sentences
        /// --max-think injects bypass the ban.
        #[arg(long)]
        ban_string: Vec<String>,
        #[command(flatten)]
        sampling: SamplingArgs,
        #[command(flatten)]
        draft: DraftArgs,
    },
    /// Interactive chat REPL.
    Chat {
        /// Model GGUF (default: the official checkpoint, ensured in the Hugging
        /// Face cache — downloaded on first use, cached forever after).
        #[arg(short, long)]
        model: Option<PathBuf>,
        #[command(flatten)]
        select: ModelArgs,
        /// Custom tokenizer.json (default: the checkpoint tokenizer embedded
        /// in the binary).
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(short = 'n', long, default_value_t = 2048)]
        max_tokens: usize,
        #[arg(long, default_value = "fused")]
        moe_impl: String,
        #[arg(long, default_value_t = 8192)]
        max_ctx: usize,
        /// Show the model's <think> reasoning (dimmed) instead of hiding it.
        #[arg(long)]
        show_thinking: bool,
        /// Force at least this many decode tokens of <think> reasoning per
        /// turn before `</think>` (or an EOG) may be sampled; 0 lets the
        /// model decide (it skips reasoning on conversational prompts).
        #[arg(long, default_value_t = 0)]
        min_think: usize,
        /// Steer the model out of its <think> block within about this many
        /// decode tokens per turn: wait words are biased down from 70% of the
        /// budget, `</think>` is pulled up from 80%, and past the budget a
        /// transition sentence plus `</think>` are injected at the next
        /// sentence boundary. 0 lets the model reason as long as it likes.
        #[arg(long, default_value_t = 0)]
        max_think: usize,
        /// Ban every token whose text contains this string, for the whole
        /// session (repeatable). Byte-level BPE can still spell a banned glyph
        /// out of raw byte tokens, so this suppresses the common spellings, not
        /// every possible one. Structural tokens are never banned, and the
        /// sentences --max-think injects bypass the ban.
        #[arg(long)]
        ban_string: Vec<String>,
        #[command(flatten)]
        sampling: SamplingArgs,
        #[command(flatten)]
        draft: DraftArgs,
    },
    /// Answer a batch of chat items that share a prompt prefix: one JSON
    /// request on stdin, one JSON response on stdout.
    ///
    /// The items' shared prefix is prefilled once and the KV cache snapshotted
    /// there; every item then restores that snapshot and prefills only its own
    /// tail, so a run of N questions about the same document costs one prefill
    /// of it rather than N. Which checkpoint to run comes from the payload
    /// (`"model": "27b"` / `"35b"`), not from a flag — one request is one
    /// model's work. Sampling defaults to greedy and thinking to off, so a
    /// batch is reproducible and a tight token budget goes to the answer.
    ///
    /// Progress lines go to stderr; stdout carries the JSON alone. Setting
    /// XWEN_BATCH_NO_CACHE runs every item from a reset cache instead, which is
    /// the A/B lever for what the snapshot actually saves. The two arms decode
    /// the same answer but not always the same bytes: an item's short tail
    /// prefill takes a different MoE matmul kernel than one long prefill does,
    /// which flips the occasional near-tie (see the batch module's docs).
    Batch {
        /// Model GGUF (default: the checkpoint the payload names, ensured in
        /// the Hugging Face cache — downloaded on first use, cached forever
        /// after).
        #[arg(short, long)]
        model: Option<PathBuf>,
        /// Custom tokenizer.json (default: the checkpoint tokenizer embedded
        /// in the binary).
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value = "fused")]
        moe_impl: String,
        #[arg(long, default_value_t = 8192)]
        max_ctx: usize,
        #[command(flatten)]
        draft: DraftArgs,
    },
    /// Serve the model over HTTP (Anthropic Messages + OpenAI Chat Completions).
    Serve(ServeArgs),
}

/// `xwen serve` flags. Every value is optional so the config merge can tell a
/// flag that was passed from one that was left alone; the `--no-*` switches are
/// present-means-false.
#[derive(Parser)]
#[command(next_help_heading = "Serve options (the default command)")]
struct ServeArgs {
    /// Config file to read (default: ~/.config/xwen/serve.toml).
    #[arg(long)]
    config: Option<PathBuf>,
    /// Write a commented config template to the config path and exit. Refuses
    /// to overwrite an existing file.
    #[arg(long)]
    init: bool,
    /// Model GGUF to serve (default: the config file's `model`, else the
    /// official checkpoint from the Hugging Face cache).
    #[arg(short, long)]
    model: Option<PathBuf>,
    #[command(flatten)]
    select: ModelArgs,
    /// Address to bind.
    #[arg(long)]
    host: Option<String>,
    /// TCP port to bind.
    #[arg(long)]
    port: Option<u16>,
    /// Context length in tokens; the full-attention KV cache is preallocated
    /// to this size at load.
    #[arg(long)]
    ctx: Option<usize>,
    /// Unload the model after this long without a request: an integer with an
    /// s/m/h suffix, or "off" to keep it loaded forever.
    #[arg(long)]
    idle_unload: Option<String>,
    /// Do not serve the Anthropic Messages API.
    #[arg(long)]
    no_anthropic: bool,
    /// Do not serve the OpenAI Chat Completions API.
    #[arg(long)]
    no_openai: bool,
    /// Require this key via `x-api-key` or `Authorization: Bearer`.
    #[arg(long)]
    api_key: Option<String>,
    /// What to do with a request carrying tool definitions: serve them
    /// ("native"), "reject" the request, or "strip" the definitions and answer
    /// as if it had none.
    #[arg(long, value_name = "native|reject|strip")]
    tools_mode: Option<String>,
    /// Requests that may wait behind the one running generation; at capacity
    /// the server answers 429 with Retry-After: 1.
    #[arg(long)]
    queue_capacity: Option<usize>,
    /// Drop a request that has waited this long in the queue, in seconds.
    #[arg(long, value_name = "SECS")]
    queue_timeout: Option<u64>,
    /// Watchdog prefill floor (tokens/second) for the per-request wall-clock
    /// ceiling — a loose bound that catches wedged generations, not a perf
    /// target. 0 disables the deadline.
    #[arg(long, value_name = "TOK/S")]
    request_prefill_rate: Option<u64>,
    /// Watchdog decode floor (tokens/second); 0 disables the deadline.
    #[arg(long, value_name = "TOK/S")]
    request_decode_rate: Option<u64>,
    /// Fixed allowance in seconds on top of the ceiling's throughput terms:
    /// model load, scheduling, cache paging.
    #[arg(long, value_name = "SECS")]
    request_slack: Option<u64>,
    /// Queue order: "shortest-prefill" runs the job needing the least prefill
    /// (prompt minus cached prefix) first; "fifo" restores arrival order.
    #[arg(long, value_name = "shortest-prefill|fifo")]
    schedule: Option<String>,
    /// Starvation guard in seconds: a queued request that has waited this long
    /// wins over any cheaper newcomer.
    #[arg(long, value_name = "SECS")]
    schedule_age_limit: Option<u64>,
    /// Draw the live dashboard — current request, queue, cache slots, history
    /// and log — overriding a config file that turned it off. The dashboard is
    /// already the default on a terminal; quit with q or Ctrl-C.
    #[arg(long, conflicts_with = "no_tui")]
    tui: bool,
    /// Print plain log lines to stderr instead of drawing the dashboard.
    #[arg(long)]
    no_tui: bool,
    /// Default sampling temperature for requests that omit one.
    #[arg(long)]
    temp: Option<f64>,
    /// Default top-k for requests that omit one.
    #[arg(long)]
    top_k: Option<usize>,
    /// Default top-p for requests that omit one.
    #[arg(long)]
    top_p: Option<f64>,
    /// KV snapshots kept for turn-boundary prefix reuse.
    #[arg(long)]
    cache_snapshots: Option<usize>,
    /// Conversations kept warm at once, so clients talking in turn stop
    /// evicting each other. One lives in the GPU cache and the rest are
    /// host-RAM images, uncapped and costing per cached token what the
    /// checkpoint's full-attention layers cost — 20 KiB on the 35B-A3B, 64 KiB
    /// on the 27B — plus one snapshot's DeltaNet state per snapshot kept
    /// (62.8 / 149.6 MiB). The live one holds an image too, so budget
    /// N x (--ctx x per-token + snapshots x per-snapshot), plus
    /// min(--draft-ctx, --ctx) x 40-48 KiB of drafter planes per slot while
    /// speculation is on, plus one slot's images again while a swap is in
    /// flight. Lower this or --ctx if that does not fit. 1 keeps a single
    /// conversation warm.
    #[arg(long)]
    cache_slots: Option<usize>,
    /// Where the on-disk prefix cache keeps its images, under <DIR>/kv/
    /// (default: ~/.cache/xwen).
    #[arg(long, value_name = "DIR")]
    cache_dir: Option<PathBuf>,
    /// Keep warm conversations on disk, so a restart resumes them instead of
    /// re-prefilling. On by default; this flag overrides a config file that
    /// turns it off.
    #[arg(long, conflicts_with = "no_disk_cache")]
    disk_cache: bool,
    /// Do not keep warm conversations on disk.
    #[arg(long)]
    no_disk_cache: bool,
    /// Ceiling in GiB on everything under <cache-dir>/kv/, every checkpoint
    /// included, enforced by deleting the least recently used image.
    #[arg(long, value_name = "GIB")]
    disk_max_gib: Option<u64>,
    /// Do not store conversations shorter than this: an image's cost is
    /// dominated by its snapshots whatever its length, while the prefill it
    /// saves grows with the token count.
    #[arg(long, value_name = "TOKENS")]
    disk_min_tokens: Option<usize>,
    /// Custom DFlash drafter GGUF to speculate with, in place of the
    /// checkpoint's official one (which the literal `official`, and the
    /// default, select — fetched into the Hugging Face cache on first use).
    /// Greedy output matches decoding without it except where a near-tie lands
    /// differently.
    #[arg(long, value_name = "GGUF", conflicts_with = "no_draft")]
    draft: Option<PathBuf>,
    /// Decode without speculation. Drafting is on by default: measured faster on
    /// both checkpoints on 2026-07-29 — 27B +19.3 to +21.0% on code and +7.6 to
    /// +8.4% on chat, 35B-A3B +18.1 to +19.8% on code and +12.6 to +12.8% on
    /// chat. It costs a sidecar load per run (3.5 GB on the 27B, 0.8 GB on the
    /// 35B-A3B) plus drafter planes per cache slot (see --draft-ctx).
    #[arg(long)]
    no_draft: bool,
    /// Max draft tokens proposed per verify round.
    #[arg(long)]
    draft_max: Option<usize>,
    /// Stop drafting at the first token whose probability falls below this.
    #[arg(long)]
    draft_p_min: Option<f32>,
    /// Pause speculation when its wall-clock cost per committed token exceeds a
    /// plain decode step's times this factor; 0 always drafts. With pausing on, a
    /// temperature>0 reply is not reproducible from a fixed seed (greedy output
    /// is unaffected).
    #[arg(long)]
    draft_pause_margin: Option<f32>,
    /// Positions the drafter's KV cache is sized for, and equally how far into a
    /// conversation speculation stays active — past it decode continues plain
    /// (the drafter's proposal quality collapses with context depth, so deep
    /// drafting is a pure loss). The cache is f32 at 40 KiB per token on the
    /// 27B sidecar and 48 on the 35B-A3B — about 0.4 GB at the default 8192, and that
    /// again per warm conversation, since each cache slot images its own
    /// drafter planes (see --cache-slots).
    #[arg(long)]
    draft_ctx: Option<usize>,
}

impl ServeArgs {
    fn overrides(&self) -> CliOverrides {
        CliOverrides {
            model: self.model.clone(),
            host: self.host.clone(),
            port: self.port,
            context_length: self.ctx,
            idle_unload: self.idle_unload.clone(),
            anthropic: self.no_anthropic.then_some(false),
            openai: self.no_openai.then_some(false),
            api_key: self.api_key.clone(),
            tools_mode: self.tools_mode.clone(),
            queue_capacity: self.queue_capacity,
            queue_timeout: self.queue_timeout,
            request_prefill_rate: self.request_prefill_rate,
            request_decode_rate: self.request_decode_rate,
            request_slack: self.request_slack,
            schedule: self.schedule.clone(),
            schedule_age_limit: self.schedule_age_limit,
            // The one setting with a switch each way, so a config file can be
            // overridden in either direction for a single run.
            tui: self.tui.then_some(true).or(self.no_tui.then_some(false)),
            temperature: self.temp,
            top_k: self.top_k,
            top_p: self.top_p,
            cache_snapshots: self.cache_snapshots,
            cache_slots: self.cache_slots,
            cache_dir: self.cache_dir.clone(),
            // Naming a drafter is itself a request for one, so an explicit
            // --draft beats a config file that set `enabled = false` (clap
            // rejects combining it with --no-draft).
            draft_enabled: self
                .no_draft
                .then_some(false)
                .or(self.draft.is_some().then_some(true)),
            // The second setting with a switch each way, and the one that defaults
            // on, so the enabling flag is there for a config file that turns it off.
            disk_cache: self
                .disk_cache
                .then_some(true)
                .or(self.no_disk_cache.then_some(false)),
            disk_max_gib: self.disk_max_gib,
            disk_min_tokens: self.disk_min_tokens,
            draft: self.draft.clone(),
            draft_max: self.draft_max,
            draft_p_min: self.draft_p_min,
            draft_pause_margin: self.draft_pause_margin,
            draft_ctx: self.draft_ctx,
        }
    }
}

/// Resolve the config, then hand the settings to the server. `--init` short-
/// circuits into writing the template.
fn run_serve(args: ServeArgs) -> Result<()> {
    let explicit_config = args.config.is_some();
    let path = match &args.config {
        Some(path) => path.clone(),
        None => xwen::serve::config::default_config_path()?,
    };

    if args.init {
        xwen::serve::config::write_init_template(&path)?;
        println!("xwen: wrote config template to {}", path.display());
        return Ok(());
    }

    // A missing default config just means "all defaults", but a --config path
    // that does not exist is a typo worth failing on.
    let file = match xwen::serve::config::load(&path)? {
        Some(file) => Some(file),
        None if explicit_config => bail!("config file {} does not exist", path.display()),
        None => None,
    };
    let source = file.as_ref().map(|_| path.as_path());
    let file = file.unwrap_or_else(ServeToml::default);

    // Neither the CLI nor the config named a model: serve the hub-cached
    // official checkpoint. Injected into the CLI side of the merge (rather than
    // inside `resolve`) so config resolution itself stays pure and testable.
    let size = args.select.model_size;
    let mut overrides = args.overrides();
    if overrides.model.is_none() && file.model.is_none() {
        overrides.model = Some(resolve_model(None, size)?);
    }

    let (mut settings, warnings) = xwen::serve::config::resolve(&file, source, &overrides)?;
    for warning in &warnings {
        eprintln!("{warning}");
    }
    if !settings.model.is_file() {
        bail!("model {} does not exist", settings.model.display());
    }
    if let Some(draft) = settings.draft.take() {
        // The sidecar belongs to the checkpoint actually being served, and the
        // GGUF's own architecture is the authority on which that is — the
        // `--model-size` flag only chose the file when nothing else named one,
        // and a config-file `model` (or `-m`) can disagree with it. A
        // metadata-only read; the serve path re-reads the same header moments
        // later to validate.
        let served =
            XwenConfig::from_gguf(&gguf::open(&settings.model, &candle_core::Device::Cpu)?.content)
                .with_context(|| format!("reading {}", settings.model.display()))?
                .arch
                .model();
        settings.draft = Some(resolve_draft(&draft, served)?);
    }
    if let Some(draft) = settings.draft.as_ref().filter(|path| !path.is_file()) {
        bail!("drafter {} does not exist", draft.display());
    }

    xwen::serve::run(settings)
}

/// `-m` given: use it verbatim. Omitted: the selected official checkpoint,
/// ensured in the standard Hugging Face cache (idempotent — the cached path
/// comes back without a request; only a missing file is downloaded, with
/// progress).
fn resolve_model(model: Option<PathBuf>, size: Model) -> Result<PathBuf> {
    match model {
        Some(path) => Ok(path),
        None => {
            if xwen::hub::cached_model(size).is_none() {
                eprintln!(
                    "xwen: {}/{} is not in the Hugging Face cache; downloading ({}, resumes in place)",
                    size.repo(),
                    size.file(),
                    size.size(),
                );
            }
            xwen::hub::ensure_model(size)
        }
    }
}

/// A draft path of literally `official` (the serve config's opt-out default)
/// means the selected model's drafter, ensured in the Hugging Face cache like
/// the model itself.
fn resolve_draft(path: &std::path::Path, size: Model) -> Result<PathBuf> {
    if path == std::path::Path::new("official") {
        ensure_drafter(size)
    } else {
        Ok(path.to_path_buf())
    }
}

/// The selected model's DFlash drafter, with the same download notice the
/// target gets — the sidecar belongs to one checkpoint and never transfers.
fn ensure_drafter(size: Model) -> Result<PathBuf> {
    if xwen::hub::cached_drafter(size).is_none() {
        eprintln!(
            "xwen: {}/{} is not in the Hugging Face cache; downloading ({})",
            size.repo(),
            size.drafter_file(),
            size.drafter_size(),
        );
    }
    xwen::hub::ensure_drafter(size)
}

/// `xwen batch`: read one request from stdin, run it, write one JSON document
/// to stdout.
///
/// Stdout is JSON on every exit, success or not — a caller reading the pipe
/// parses one document either way — so a whole-request failure is caught here,
/// printed as `{"error": ...}` and reported by the exit status rather than by
/// anyhow's stderr message. Per-item failures never reach this: they ride the
/// response as an `error` on their own item.
fn run_batch(
    model: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    moe_impl: &str,
    max_ctx: usize,
    draft: &DraftArgs,
) -> Result<()> {
    match batch_request(model, tokenizer, moe_impl, max_ctx, draft) {
        Ok(response) => {
            let mut stdout = std::io::stdout();
            writeln!(stdout, "{}", serde_json::to_string_pretty(&response)?)?;
            stdout.flush()?;
            Ok(())
        }
        Err(error) => {
            let document = serde_json::json!({ "error": format!("{error:#}") });
            let mut stdout = std::io::stdout();
            // Written and flushed before the exit: `process::exit` runs no
            // destructors, so a buffered document would be lost.
            writeln!(stdout, "{}", serde_json::to_string_pretty(&document)?)?;
            stdout.flush()?;
            std::process::exit(1);
        }
    }
}

/// Everything `run_batch` does that can fail as a whole request: parse stdin,
/// resolve the checkpoint the payload names, load it, run the batch.
fn batch_request(
    model: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    moe_impl: &str,
    max_ctx: usize,
    draft: &DraftArgs,
) -> Result<BatchResponse> {
    let mut input = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut input)?;
    let request: BatchRequest = serde_json::from_str(&input)
        .context("the request on stdin is not a valid batch request")?;
    // Checked here as well as in `run_batch`, because between the two sits a
    // 20 GB load: a request with nothing to answer must be refused in
    // milliseconds rather than after the checkpoint is resident.
    ensure!(
        !request.items.is_empty(),
        "batch: the request holds no items"
    );
    // The payload names the checkpoint; `-m` still overrides the file, exactly
    // as it does for the other commands.
    let size = request.model()?;

    let load_start = std::time::Instant::now();
    let mut generator = build_generator(
        &resolve_model(model, size)?,
        size,
        tokenizer.as_deref(),
        moe_impl,
        max_ctx,
        // Batch sampling is per item and resolved inside the batch runner; this
        // is only what the generator is constructed with.
        xwen::batch::BATCH_SAMPLING,
        Some(draft),
    )?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    // Progress to stderr in the command's own format; a batch process is never
    // cancelled from inside — it runs to completion or is killed whole.
    let mut progress = |report: xwen::batch::BatchProgress| match report {
        xwen::batch::BatchProgress::SharedPrefix { tokens, ms } => {
            eprintln!("xwen: shared prefix {tokens} tokens prefilled in {ms:.0}ms");
        }
        xwen::batch::BatchProgress::Item {
            id,
            completion_tokens,
            ms,
        } => {
            eprintln!("xwen: item {id:?} {completion_tokens} tokens in {ms:.0}ms");
        }
    };
    let mut never = || false;
    xwen::batch::run_batch(
        &mut generator,
        &request,
        load_ms,
        &mut xwen::batch::BatchHooks {
            progress: &mut progress,
            cancelled: &mut never,
        },
    )
}

fn expert_runner(name: &str) -> Result<ExpertRunner> {
    match name {
        "reference" | "ref" => Ok(ExpertRunner::Reference),
        "fused" => Ok(ExpertRunner::Fused),
        other => bail!("unknown --moe-impl {other:?} (expected reference|fused)"),
    }
}

/// Load the model + tokenizer + sampler and assemble a Generator on Metal.
/// Speculation is opt-out: with `draft` present and not `--no-draft`, the
/// DFlash drafter (custom `--draft` path, else the hub-ensured official one)
/// is loaded on the SAME Metal device (its ops interleave with the target's
/// shared embeddings/lm_head, so they must share a device) and attached.
fn build_generator(
    model: &PathBuf,
    size: Model,
    tokenizer: Option<&Path>,
    moe_impl: &str,
    max_ctx: usize,
    sampling: SamplerOptions,
    draft: Option<&DraftArgs>,
) -> Result<Generator> {
    let runner = expert_runner(moe_impl)?;
    let device = gguf::metal_device()?;

    let load_start = std::time::Instant::now();
    let mut generator = Generator::load(&device, model, tokenizer, runner, max_ctx, sampling)?;
    eprintln!(
        "xwen: model loaded in {:.1}s",
        load_start.elapsed().as_secs_f64()
    );

    // Speculation is opt-out: a zero-flag run speculates with the official
    // drafter, which a first run fetches into the Hugging Face cache (3.5 GB on
    // the 27B, 0.8 GB on the 35B-A3B, with the same download notice the target
    // gets). `--no-draft` decodes plain.
    if let Some(draft) = draft.filter(|d| !d.no_draft) {
        // resolve_draft keeps `--draft official` — and the default, which is
        // that same symbolic path — meaning the same thing here as in the serve
        // config.
        let requested = draft
            .draft
            .clone()
            .unwrap_or_else(|| PathBuf::from("official"));
        let path = resolve_draft(&requested, size)?;
        let draft_start = std::time::Instant::now();
        let dgguf = gguf::open(&path, &device)?;
        // The drafter's cache is sized by --draft-ctx (capped at the target's
        // context), which is also the drafting depth limit — the same rule the
        // serve engine applies. Sizing it at the target's max_ctx would buy
        // 40-48 KiB/token of f32 cache for depths where drafting never pays (and
        // OOMs outright at 262k alongside the 68 GB target).
        let drafter = DflashDrafter::load(&dgguf, &device, draft.draft_ctx.min(max_ctx))?;
        generator.attach_drafter(drafter, draft.params(size))?;
        eprintln!(
            "xwen: drafter loaded in {:.1}s",
            draft_start.elapsed().as_secs_f64()
        );
    }

    Ok(generator)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        // No subcommand: serve, with whatever serve flags were passed at the
        // top level.
        None => run_serve(cli.serve),
        Some(Cmd::Fetch { select }) => {
            let size = select.model_size;
            let model = resolve_model(None, size)?;
            println!("model    {}", model.display());
            let drafter = ensure_drafter(size)?;
            println!("drafter  {}", drafter.display());
            Ok(())
        }
        Some(Cmd::Inspect { model, select }) => {
            let model = resolve_model(model, select.model_size)?;
            let device = candle_core::Device::Cpu;
            let gguf = gguf::open(&model, &device)?;
            print!("{}", gguf::describe(&gguf.content));
            let cfg = XwenConfig::from_gguf(&gguf.content)?;
            println!("\nparsed config: {cfg:#?}");
            Ok(())
        }
        Some(Cmd::Generate {
            model,
            select,
            prompt,
            tokenizer,
            max_tokens,
            moe_impl,
            max_ctx,
            raw,
            stats,
            min_think,
            max_think,
            ban_string,
            sampling,
            draft,
        }) => {
            // The floor bans `</think>` and the EOG ids for its first `min_think`
            // tokens, which only means "keep reasoning" when the prompt ends inside
            // the open `<think>` block the chat template writes. A raw prompt has no
            // such block, so the flag would just suppress stopping — reject it
            // rather than silently distort the generation.
            if raw && min_think > 0 {
                bail!(
                    "--min-think {min_think} is meaningless with --raw: the floor holds the model \
                     inside the chat template's <think> block, which a raw prompt never opens"
                );
            }
            // Same reasoning for the ceiling: with no open <think> block there is
            // nothing to steer out of, and the schedule would inject a transition
            // sentence and a stray `</think>` into ordinary text.
            if raw && max_think > 0 {
                bail!(
                    "--max-think {max_think} is meaningless with --raw: the ceiling steers the \
                     model out of the chat template's <think> block, which a raw prompt never opens"
                );
            }
            let mut generator = build_generator(
                &resolve_model(model, select.model_size)?,
                select.model_size,
                tokenizer.as_deref(),
                &moe_impl,
                max_ctx,
                sampling.options(),
                Some(&draft),
            )?;
            generator.set_min_think(min_think);
            generator.set_max_think(max_think)?;
            generator.set_banned_strings(&ban_string)?;

            // Qwen has no BOS; raw mode feeds the prompt bytes verbatim.
            // Raw prompts are trusted whole (no content ranges); chat prompts
            // carry the user text's ranges so literal control-token strings in
            // it encode as plain text.
            let (text, content_ranges) = if raw {
                (prompt.clone(), Vec::new())
            } else {
                build_prompt_with_spans(
                    &[Message::User(prompt)],
                    &ChatOptions {
                        enable_thinking: true,
                        ..Default::default()
                    },
                )?
            };

            let mut stdout = std::io::stdout();
            let gstats = generator.generate_with_content_ranges(
                &text,
                &content_ranges,
                max_tokens,
                &mut |chunk| {
                    print!("{chunk}");
                    let _ = stdout.flush();
                },
                &mut || false,
            )?;
            println!();

            if stats {
                eprintln!(
                    "\nprefill: {} tokens in {:.2}s ({:.1} tok/s)\ndecode:  {} tokens in {:.2}s ({:.1} tok/s)",
                    gstats.prefill_tokens,
                    gstats.prefill_secs,
                    gstats.prefill_tps(),
                    gstats.decode_tokens,
                    gstats.decode_secs,
                    gstats.decode_tps(),
                );
                if let Some(spec) = &gstats.spec {
                    // Per-round-class tok/s: did drafting pay off within THIS
                    // run? Buckets with zero rounds are skipped (a margin-0 run
                    // can have no plain rounds; a paused run no drafted ones).
                    if spec.plain_rounds > 0 && spec.plain_ms > 0.0 {
                        eprintln!(
                            "         plain:       {:>5} tok in {:.2}s ({:.1} tok/s over {} rounds)",
                            spec.plain_rounds,
                            spec.plain_ms / 1000.0,
                            spec.plain_rounds as f64 / (spec.plain_ms / 1000.0),
                            spec.plain_rounds,
                        );
                    }
                    if spec.spec_rounds > 0 && spec.spec_ms > 0.0 {
                        eprintln!(
                            "         drafted:     {:>5} tok in {:.2}s ({:.1} tok/s over {} rounds, {:.1} tok/round)",
                            spec.spec_tokens,
                            spec.spec_ms / 1000.0,
                            spec.spec_tokens as f64 / (spec.spec_ms / 1000.0),
                            spec.spec_rounds,
                            spec.spec_tokens as f64 / spec.spec_rounds as f64,
                        );
                    }
                    if spec.full_accept_rounds > 0 && spec.full_accept_ms > 0.0 {
                        eprintln!(
                            "         full-accept: {:>5} tok in {:.2}s ({:.1} tok/s over {} rounds)",
                            spec.full_accept_tokens,
                            spec.full_accept_ms / 1000.0,
                            spec.full_accept_tokens as f64 / (spec.full_accept_ms / 1000.0),
                            spec.full_accept_rounds,
                        );
                    }
                    // The comparison needs a stable plain baseline; a handful
                    // of plain rounds is noise, not a rate.
                    if spec.plain_rounds >= 8
                        && spec.spec_rounds > 0
                        && spec.plain_ms > 0.0
                        && spec.spec_ms > 0.0
                    {
                        let plain_rate = spec.plain_rounds as f64 / (spec.plain_ms / 1000.0);
                        let drafted_rate = spec.spec_tokens as f64 / (spec.spec_ms / 1000.0);
                        // Estimated net effect over the whole run: what the
                        // run's own plain rate would have cost for every
                        // committed token, vs the model time actually spent —
                        // plain forwards + full speculative rounds + drafter
                        // time wasted on rounds that ended plain. Those three
                        // partition decode-loop model time with no double
                        // counting.
                        let total_tokens = (spec.plain_rounds + spec.spec_tokens) as f64;
                        let baseline_secs = total_tokens / plain_rate;
                        let actual_secs = (spec.plain_ms
                            + spec.spec_ms
                            + (spec.draft_ms - spec.spec_draft_ms).max(0.0))
                            / 1000.0;
                        eprintln!(
                            "         drafting:    {:.2}x vs plain on drafted rounds; est. net {:+.1}% overall",
                            drafted_rate / plain_rate,
                            (baseline_secs / actual_secs - 1.0) * 100.0,
                        );
                    }
                }
                if let Some(think) = &gstats.think {
                    let exit = match (think.closed, think.forced) {
                        (true, true) => "forced",
                        (true, false) => "closed on its own",
                        (false, _) => "never closed",
                    };
                    eprintln!(
                        "think:   {} tokens of {max_think} budget, {exit}; wrap-up {}",
                        think.tokens,
                        if think.wrapup_fired {
                            "injected"
                        } else {
                            "not needed"
                        },
                    );
                }
                if let Some(spec) = &gstats.spec {
                    let paused = if spec.paused_rounds > 0 {
                        format!(" ({} paused)", spec.paused_rounds)
                    } else {
                        String::new()
                    };
                    eprintln!(
                        "spec:    {} rounds{paused}, {} drafted, {} accepted ({:.1}%), {} rejected",
                        spec.rounds,
                        spec.drafted,
                        spec.accepted,
                        spec.acceptance_rate() * 100.0,
                        spec.rejected(),
                    );
                    // Verify averages over all rounds (>= 1 here since a spec
                    // line only prints after decode ran); draft averages over
                    // the rounds the drafter actually ran — draft_ms accrues
                    // only on those, so dividing by all rounds would understate
                    // the per-draft cost whenever auto-pause skipped drafting.
                    let rounds = spec.rounds.max(1) as f64;
                    let draft_rounds = spec.draft_rounds.max(1) as f64;
                    eprintln!(
                        "         {} verified positions; draft {:.1}s ({:.0}ms/draft), verify {:.1}s ({:.0}ms/round)",
                        spec.verify_positions,
                        spec.draft_ms / 1000.0,
                        spec.draft_ms / draft_rounds,
                        spec.verify_ms / 1000.0,
                        spec.verify_ms / rounds,
                    );
                }
            }
            Ok(())
        }
        Some(Cmd::Chat {
            model,
            select,
            tokenizer,
            max_tokens,
            moe_impl,
            max_ctx,
            show_thinking,
            min_think,
            max_think,
            ban_string,
            sampling,
            draft,
        }) => {
            let mut generator = build_generator(
                &resolve_model(model, select.model_size)?,
                select.model_size,
                tokenizer.as_deref(),
                &moe_impl,
                max_ctx,
                sampling.options(),
                Some(&draft),
            )?;
            generator.set_min_think(min_think);
            generator.set_max_think(max_think)?;
            generator.set_banned_strings(&ban_string)?;
            repl::run(&mut generator, max_tokens, show_thinking)
        }
        Some(Cmd::Batch {
            model,
            tokenizer,
            moe_impl,
            max_ctx,
            draft,
        }) => run_batch(model, tokenizer, &moe_impl, max_ctx, &draft),
        Some(Cmd::Serve(args)) => run_serve(args),
    }
}
