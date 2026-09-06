mod repl;

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};

use xwen::batch::{BatchRequest, BatchResponse};
use xwen::chat::{ChatOptions, Message, ReasoningEffort, build_prompt_with_spans};
use xwen::config::{Identity, XwenConfig};
use xwen::dflash::DflashDrafter;
use xwen::generate::{Generator, SpecParams};
use xwen::gguf;
use xwen::hub::Model;
use xwen::metrics::{self, RunRecord};
use xwen::mtp::MtpDrafter;
use xwen::ops::ExpertRunner;
use xwen::sampler::SamplerOptions;
use xwen::serve::config::{CliOverrides, DraftMode, ServeToml};

/// The `--max-ctx` default on every one-shot surface — `generate`, `chat` and
/// `batch` — which had three separate copies of the number and no way to keep
/// them agreeing. A ceiling, not an allocation: the KV cache starts at
/// `KV_INITIAL_CTX` positions and doubles on demand, so this costs memory only
/// when a prompt actually reaches it.
///
/// It is deliberately BELOW the 262144 the blessed checkpoints were converted
/// with, and below serve's own default: a one-shot run has no operator watching
/// a dashboard, and 131072 is the length the envelope has been measured at
/// (docs/perf-state.md, "Long context"). `XwenModel::load` clamps whatever this
/// resolves to against the file's `n_ctx_train`, so a checkpoint converted
/// smaller is trimmed rather than run past its rope table.
const DEFAULT_MAX_CTX: usize = 131072;

#[derive(Parser)]
#[command(
    name = "xwen",
    about = "Qwen inference on Metal, defaulting to Qwen3.8-Flash-Next. Bare \
             `xwen` serves over HTTP with the live dashboard; subcommands cover \
             everything else.",
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
    /// Which official checkpoint to run: Qwen3.8-Flash-Next (`flash-next`,
    /// the default on every surface, still EXPERIMENTAL and without a
    /// drafter), the dense Qwen3.6-27B, the Qwen3.6-35B-A3B MoE, or the dense
    /// Qwen3.8-27B. Each checkpoint's full name works here too. A `--model <gguf>` path overrides the target file
    /// outright, and then the FILE says which checkpoint it is: this flag is the
    /// cross-check (it must agree, or startup fails) and the tie-break for a
    /// custom GGUF that names no release — on every surface alike.
    #[arg(long, value_name = "27b|35b|3.8-27b|flash-next")]
    model_size: Option<Model>,
}

impl ModelArgs {
    /// The selected checkpoint, or [`Model::default`] when the flag was
    /// omitted — for `fetch` and `inspect`, which name a file to act on and
    /// never load a graph. The commands that RUN one go through
    /// `one_shot_checkpoint` (which reads a `--model` file's own identity) or,
    /// for serve, through `identify_checkpoint`; both want
    /// [`Model::default_servable`] rather than the plain default wherever the
    /// surface moves cache state.
    fn size(&self) -> Model {
        self.model_size.unwrap_or_default()
    }
}

/// Shared sampling knobs. The model card keys its recommended sampling to
/// thinking on/off, so the defaults are mode-dependent and cannot live on the
/// flags themselves (the same rationale as DraftArgs' per-model defaults):
/// each unset flag resolves against `SamplerOptions::recommended` for the
/// run's thinking mode.
#[derive(Parser)]
struct SamplingArgs {
    /// Sampling temperature (default: 1.0 with thinking, 0.7 with --no-think).
    #[arg(long)]
    temp: Option<f64>,
    /// Top-k truncation (default: 20 in both modes).
    #[arg(long)]
    top_k: Option<usize>,
    /// Top-p nucleus truncation (default: 0.95 with thinking, 0.80 with
    /// --no-think).
    #[arg(long)]
    top_p: Option<f64>,
    /// Presence penalty: subtracted from the logit of every token the reply has
    /// already produced, once per distinct token, and 0 to turn it off. The
    /// default is the checkpoint's own — 1.5 with --no-think everywhere; with
    /// thinking, 1.5 on Qwen3.6-35B-A3B and 0 on the rest.
    #[arg(long)]
    presence_penalty: Option<f64>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

impl SamplingArgs {
    /// Resolve the flags over the recommended set for `model` in `thinking`
    /// mode. A caller with no chat mode at all (a raw prompt) passes `true`:
    /// the thinking set is the historical default, so those paths sample as
    /// they always have.
    ///
    /// The checkpoint is a parameter because one of the card values — the
    /// presence penalty — differs between them; the other three do not.
    fn options(&self, model: Model, thinking: bool) -> SamplerOptions {
        let recommended = SamplerOptions::recommended_for(model, thinking);
        SamplerOptions {
            temperature: self.temp.unwrap_or(recommended.temperature),
            top_k: self.top_k.unwrap_or(recommended.top_k),
            top_p: self.top_p.unwrap_or(recommended.top_p),
            presence_penalty: self
                .presence_penalty
                .unwrap_or(recommended.presence_penalty),
            seed: self.seed,
        }
    }
}

/// A [`ReasoningEffort`] as a CLI value: the 3.8 template's three levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum EffortArg {
    Low,
    Medium,
    Xhigh,
}

impl From<EffortArg> for ReasoningEffort {
    fn from(effort: EffortArg) -> Self {
        match effort {
            EffortArg::Low => Self::Low,
            EffortArg::Medium => Self::Medium,
            EffortArg::Xhigh => Self::Xhigh,
        }
    }
}

/// Chat-template knobs shared by the chat-mode commands (generate and chat).
#[derive(Parser)]
struct ThinkArgs {
    /// Answer without thinking: the prompt closes an empty <think> block ahead
    /// of the reply, and sampling switches to the instruct defaults
    /// (temp 0.7 / top_p 0.80).
    #[arg(long)]
    no_think: bool,
    /// Reasoning-effort level rendered into the Qwen 3.8 chat template
    /// (default: xhigh, the template's own). A system-preamble instruction
    /// with no budget semantics. The 3.6 template has no such parameter, so
    /// the flag is a startup error on a 3.6 checkpoint.
    #[arg(long, value_name = "low|medium|xhigh")]
    reasoning_effort: Option<EffortArg>,
}

impl ThinkArgs {
    /// The checkpoint's [`ChatOptions`] with these knobs applied.
    ///
    /// A supplied `--reasoning-effort` on a checkpoint whose template has no
    /// such parameter is refused rather than ignored — the flag would change
    /// nothing, and this repo's flags cross-check instead of shrugging (the
    /// `--model-size` rule). Unset, the default level renders nothing on 3.6
    /// anyway, so there is nothing to refuse.
    fn chat_options(&self, size: Model) -> Result<ChatOptions> {
        let mut opts = ChatOptions::for_dialect(size.chat_dialect());
        opts.enable_thinking = !self.no_think;
        if let Some(effort) = self.reasoning_effort {
            ensure!(
                size.chat_dialect() == xwen::chat::ChatDialect::Qwen38,
                "--reasoning-effort {}: {} renders the Qwen 3.6 chat template, which has no \
                 reasoning_effort parameter (it is a Qwen 3.8 template feature)",
                ReasoningEffort::from(effort),
                size.full_name(),
            );
            opts.reasoning_effort = effort.into();
        }
        Ok(opts)
    }

    /// Cross-check the think-budget flags against `--no-think`.
    ///
    /// Both budgets govern the open `<think>` block, which a no-think prompt
    /// closes itself before the reply begins. The floor would then just ban the
    /// EOG ids and suppress stopping, and the armed ceiling would wait for a
    /// decoded `</think>` that never comes, eventually forcing its wrap-up
    /// sentence and a stray `</think>` into the answer (serve guards the same
    /// hazard by dropping the budget when the prompt does not start in
    /// thinking) — so the combination is a startup error, same rule as the
    /// `--raw` combos.
    fn check_think_budgets(&self, min_think: usize, max_think: usize) -> Result<()> {
        if !self.no_think {
            return Ok(());
        }
        if min_think > 0 {
            bail!(
                "--min-think {min_think} is meaningless with --no-think: the floor holds the \
                 model inside the chat template's <think> block, which --no-think closes before \
                 the reply begins"
            );
        }
        if max_think > 0 {
            bail!(
                "--max-think {max_think} is meaningless with --no-think: the ceiling steers the \
                 model out of the chat template's <think> block, which --no-think closes before \
                 the reply begins"
            );
        }
        Ok(())
    }
}

/// DFlash speculative-decode knobs. Speculation is opt-OUT on xwen: decoding
/// speculates with the checkpoint's official drafter unless `--no-draft` says
/// otherwise, and `--draft <gguf>` swaps in a custom one. Per checkpoint,
/// though — [`Model::draft_default_on`] decides what silence means, and it is
/// false on the 35B-A3B, where `--draft official` is the opt-IN.
#[derive(Parser)]
struct DraftArgs {
    /// Speculate with a custom drafter GGUF — either kind, a DFlash sidecar or
    /// an MTP head — instead of the checkpoint's official one (which the literal
    /// `official`, and the default, select).
    #[arg(long, value_name = "GGUF", conflicts_with = "no_draft")]
    draft: Option<PathBuf>,
    /// Decode without speculation. Drafting is on by default on the 27B (+46 to
    /// +52%, 2026-08-08) and the 3.8-27B (+44 to +45% on code, +37 to +38% on
    /// chat, 2026-08-15), each at its own fitted defaults. NOT on the 35B-A3B
    /// since 2026-09-06: its drafted arm now reads below plain at every length,
    /// the router gemv having lifted plain decode past what the drafting
    /// defaults were fitted against, so `--draft official` is what turns it on
    /// there. See docs/decisions.md, "Speculative decoding".
    #[arg(long)]
    no_draft: bool,
    /// Max draft tokens proposed per verify round. The default is per-model,
    /// because the two drafter kinds have opposite economics: 15 on the DFlash
    /// checkpoints, whose sidecar proposes a whole block in one forward (and
    /// where this is clamped to block_size-1 anyway), and 4 on 3.8-27b, whose
    /// MTP head pays a forward per step (fitted 2026-08-15; depths 5, 6 and 8
    /// all measured worse).
    #[arg(long)]
    draft_max: Option<usize>,
    /// Discard a round's whole draft if fewer than this many are collected.
    #[arg(long, default_value_t = 0)]
    draft_min: usize,
    /// Stop drafting at the first token whose full-vocab softmax prob is below
    /// this. Adaptive draft length; the default is per-model — 0.5 for 27b, 0.3
    /// for 35b (both fitted 2026-08-08), 0.5 for 3.8-27b.
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
    /// cannot live on the flag itself. A checkpoint that ships no sidecar has
    /// no fitted floor either — only a custom `--draft` reaches this for one —
    /// so that falls back to the shared base.
    fn params(&self, size: Model) -> SpecParams {
        let base = SpecParams::default();
        SpecParams {
            draft_max: self
                .draft_max
                .or_else(|| size.draft_max_default())
                .unwrap_or(base.draft_max),
            draft_min: self.draft_min,
            draft_p_min: self
                .draft_p_min
                .or_else(|| size.draft_p_min_default())
                .unwrap_or(base.draft_p_min),
            pause_margin: self.draft_pause_margin,
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Ensure the official model + drafter sidecar are in the Hugging Face
    /// cache (idempotent: anything already cached is not touched), then print
    /// their paths. Every command does this lazily for whatever it needs; this
    /// just prefetches. With no --model-size that is the default checkpoint,
    /// Qwen3.8-Flash-Next: four shards and 111 GB, so name a size for anything
    /// smaller.
    Fetch {
        #[command(flatten)]
        select: ModelArgs,
    },
    /// Summarize the per-run metrics history: what every generate, chat, batch
    /// and served request cost, grouped and totalled.
    ///
    /// Every surface appends one record per run to
    /// `$HOME/.local/state/xwen/metrics.jsonl`. `XWEN_METRICS_FILE` names
    /// another file, or says `off` (in any casing) to record nothing at all;
    /// setting it to an empty string counts as not setting it.
    ///
    /// A run recorded under `XWEN_METRICS_TAG` was driven by a harness rather
    /// than asked for: the bench and parity scripts set it, and this report
    /// leaves those runs out unless `--tag` or `--all-tags` asks for them.
    Stats {
        /// What a row covers: day|week|month|model|surface|client|session|agent|all.
        #[arg(long, default_value = "day")]
        by: String,
        /// Only runs since this point: `24h`, `7d`, `4w`, or `YYYY-MM-DD`
        /// (local midnight of that day).
        #[arg(long)]
        since: Option<String>,
        /// Only runs on this checkpoint, named exactly as the table spells it.
        #[arg(long)]
        model: Option<String>,
        /// Only runs on this surface, e.g. `generate` or `serve:openai`.
        #[arg(long)]
        surface: Option<String>,
        /// Only runs whose client id contains this text. A substring, because
        /// the raw ids are long.
        #[arg(long)]
        client: Option<String>,
        /// Only runs whose session id contains this text.
        #[arg(long)]
        session: Option<String>,
        /// Only runs a harness recorded under this tag (`bench`, `parity`),
        /// instead of the real use the default report covers.
        #[arg(long, conflicts_with = "all_tags")]
        tag: Option<String>,
        /// Report on every run in the history, the harness-driven ones
        /// included. The default leaves them out and says how many.
        #[arg(long)]
        all_tags: bool,
        /// Print the rows as JSON instead of a table.
        #[arg(long)]
        json: bool,
        /// Read this history instead of the configured one. Reading only —
        /// nothing is ever recorded to it.
        #[arg(long)]
        file: Option<PathBuf>,
    },
    /// Dump GGUF metadata and tensor listing.
    Inspect {
        /// Model GGUF (default: the checkpoint --model-size names, ensured in
        /// the Hugging Face cache — downloaded on first use, cached forever
        /// after).
        #[arg(short, long)]
        model: Option<PathBuf>,
        #[command(flatten)]
        select: ModelArgs,
    },
    /// One-shot generation from a prompt.
    Generate {
        /// Model GGUF (default: the checkpoint --model-size names, ensured in
        /// the Hugging Face cache — downloaded on first use, cached forever
        /// after).
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
        /// Context ceiling in tokens. A ceiling, not an allocation: the KV
        /// cache starts small and grows on demand, so a large value costs
        /// memory only when a prompt actually reaches it.
        #[arg(long, default_value_t = DEFAULT_MAX_CTX)]
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
        think: ThinkArgs,
        #[command(flatten)]
        sampling: SamplingArgs,
        #[command(flatten)]
        draft: DraftArgs,
    },
    /// Interactive chat REPL.
    Chat {
        /// Model GGUF (default: the checkpoint --model-size names, ensured in
        /// the Hugging Face cache — downloaded on first use, cached forever
        /// after).
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
        /// Context ceiling in tokens. A ceiling, not an allocation: the KV
        /// cache starts small and grows on demand, so a large value costs
        /// memory only when a prompt actually reaches it.
        #[arg(long, default_value_t = DEFAULT_MAX_CTX)]
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
        think: ThinkArgs,
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
    /// model's work — or from the `-m` file's own identity when one is given.
    /// A payload naming nothing gets the same zero-flag checkpoint `xwen serve`
    /// resolves, the two surfaces moving cache state for the same reasons.
    /// Sampling defaults to greedy and thinking to off, so a batch is
    /// reproducible and a tight token budget goes to the answer.
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
        /// after). Given one, the file decides which checkpoint it is and the
        /// payload's `model` becomes the cross-check.
        #[arg(short, long)]
        model: Option<PathBuf>,
        /// Custom tokenizer.json (default: the checkpoint tokenizer embedded
        /// in the binary).
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value = "fused")]
        moe_impl: String,
        /// Context ceiling in tokens. A ceiling, not an allocation: the KV
        /// cache starts small and grows on demand, so a large value costs
        /// memory only when a prompt actually reaches it.
        #[arg(long, default_value_t = DEFAULT_MAX_CTX)]
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
    /// server's default checkpoint from the Hugging Face cache).
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
    /// Context length in tokens — a ceiling, not an allocation: the
    /// full-attention KV cache starts small and grows on demand, and an idle
    /// unload shrinks it back by dropping the model.
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
    /// Default sampling temperature for requests that omit one. Unset, each
    /// request uses the mode default for its resolved thinking state (1.0
    /// thinking / 0.7 not); setting this pins one value for both modes.
    #[arg(long)]
    temp: Option<f64>,
    /// Default top-k for requests that omit one (mode default: 20 either way).
    #[arg(long)]
    top_k: Option<usize>,
    /// Default top-p for requests that omit one. Unset, the mode default
    /// applies (0.95 thinking / 0.80 not); setting this pins both modes.
    #[arg(long)]
    top_p: Option<f64>,
    /// Default presence penalty for requests that omit one. Unset, the
    /// checkpoint's own mode default applies (1.5 without thinking on every
    /// checkpoint; with thinking, 1.5 on Qwen3.6-35B-A3B and 0 on the rest);
    /// setting this pins both modes.
    #[arg(long)]
    presence_penalty: Option<f64>,
    /// Default template reasoning-effort for requests that name none
    /// (low|medium|xhigh; default: the template's own, xhigh). Rendered by
    /// the Qwen 3.8 chat template; inert on the 3.6 checkpoints.
    #[arg(long, value_name = "low|medium|xhigh")]
    reasoning_effort: Option<String>,
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
    /// min(--draft-ctx, --ctx) x 4-48 KiB of drafter planes per slot (4 on the
    /// 3.8's MTP head, 40-48 on the DFlash sidecars) while
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
    /// re-prefilling. Off by default (a Flash-Next image is gigabytes per
    /// conversation, which is SSD wear nobody asked for); this flag overrides a
    /// config file that turns it off.
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
    /// Custom drafter GGUF to speculate with — either kind, a DFlash sidecar or
    /// an MTP head — in place of the checkpoint's official one (which the
    /// literal `official`, and the default, select — fetched into the Hugging
    /// Face cache on first use).
    /// Greedy output matches decoding without it except where a near-tie lands
    /// differently.
    #[arg(long, value_name = "GGUF", conflicts_with = "no_draft")]
    draft: Option<PathBuf>,
    /// Decode without speculation. Drafting is on by default on the 27B (+46 to
    /// +52%, 2026-08-08) and the 3.8-27B (+44 to +45% on code, +37 to +38% on
    /// chat, 2026-08-15), each at its own fitted defaults, and NOT on the
    /// 35B-A3B since 2026-09-06, whose drafted arm now reads below plain at
    /// every length (`--draft official` turns it on there). Where it is on it
    /// costs a sidecar load per run (3.5 GB on the 27B, 0.8 GB on the 35B-A3B,
    /// 3.2 GB on the 3.8-27B) plus drafter planes per cache slot (see
    /// --draft-ctx).
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
    /// drafting is a pure loss). What the cache costs depends on the sidecar's
    /// kind: f32 at 40 KiB per token on the 27B DFlash sidecar and 48 on the
    /// 35B-A3B — about 0.4 GB at the default 8192 — against f16 at 4 KiB on the
    /// 3.8's single-layer MTP head, an order of magnitude less. And that again
    /// per warm conversation, since each cache slot images its own drafter
    /// planes (see --cache-slots).
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
            presence_penalty: self.presence_penalty,
            reasoning_effort: self.reasoning_effort.clone(),
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

    let selected = args.select.model_size;
    let mut overrides = args.overrides();
    // Neither the CLI nor the config named a model: serve the hub-cached
    // official checkpoint. Injected into the CLI side of the merge (rather than
    // inside `resolve`) so config resolution itself stays pure and testable.
    if overrides.model.is_none() && file.model.is_none() {
        // The zero-flag checkpoint goes through `default_servable` rather than
        // `default`, because this surface moves cache state and that is the
        // question the rule answers. The two agree today; the indirection is
        // what keeps them agreeing on purpose rather than by coincidence.
        let size = selected.unwrap_or_else(Model::default_servable);
        overrides.model = Some(resolve_model(None, size)?);
    }

    let (settings, warnings) = xwen::serve::config::resolve(&file, source, &overrides)?;
    for warning in &warnings {
        eprintln!("{warning}");
    }
    if !settings.model.is_file() {
        bail!("model {} does not exist", settings.model.display());
    }

    // The served FILE decides which checkpoint this is, not the name: it settles
    // the chat dialect, the drafter and the label. Read once here and reused by
    // the drafter prefetch below.
    let served_cfg =
        XwenConfig::from_gguf(&gguf::open(&settings.model, &candle_core::Device::Cpu)?.content)
            .with_context(|| format!("reading {}", settings.model.display()))?;
    let (served_target, _) =
        xwen::serve::engine::identify_checkpoint(&settings, &served_cfg, selected)?;

    match &settings.draft {
        DraftMode::Off => {}
        DraftMode::Custom(path) => {
            if !path.is_file() {
                bail!("drafter {} does not exist", path.display());
            }
        }
        // Nothing in the config asked, and the checkpoint being served does not
        // draft unasked: nothing to prefetch, and the engine says so when it
        // loads. Other checkpoints this server may load resolve their own
        // defaults then, exactly as they resolve their own sidecars.
        DraftMode::Default if !served_target.model.draft_default_on() => {}
        DraftMode::Official | DraftMode::Default => {
            // Each checkpoint drafts with its own sidecar, resolved when it
            // loads — this only prefetches the one for the checkpoint being
            // SERVED, so a first request does not stall behind a 3.5 GB
            // download. Which checkpoint that is comes from the file, through
            // the same call the server itself uses a moment later: identifying
            // it here by any other rule would mean prefetching for one
            // checkpoint and serving another, and would report a checkpoint the
            // server is about to refuse (a `--model-size` that contradicts the
            // file is an error, raised here rather than after the notice).
            let served = served_target.model;
            // `--draft official` is a request by name and cannot be honored for
            // a checkpoint that ships no sidecar; the opt-out default asked for
            // nothing and degrades with a line, since drafting is otherwise on
            // and its absence would show up only as slower decoding.
            let by_name = args
                .draft
                .as_deref()
                .is_some_and(|path| path == Path::new(xwen::serve::config::OFFICIAL_DRAFTER));
            if ensure_drafter_explicit(served, by_name)?.is_none() {
                eprintln!(
                    "xwen: no drafter available for {}; serving it without speculative decoding \
                     (other checkpoints this server loads still draft with their own)",
                    served.full_name()
                );
            }
        }
    }

    xwen::serve::run(settings, selected)
}

/// Which checkpoint a one-shot run (`generate`, `chat`, `batch`) is against.
///
/// The FILE decides whenever there is one. `--model <gguf>` names a file whose
/// identity is a fact about its metadata, and reading it here is what keeps the
/// chat dialect, the drafter sidecar and the label the run reports agreeing with
/// what actually loaded — the same rule `xwen serve` applies to the file it
/// serves (`XwenConfig::identify`), so a custom GGUF is not one checkpoint on
/// one surface and another on the next.
///
/// `selected` is a cross-check, not an override: it must agree with a file that
/// identifies itself, and settles one that identifies as nothing. It is what the
/// run named, `selector` is what to call that in an error, and `default` is what
/// a run that named nothing gets when there is also no file to read. All three
/// differ per command — `--model-size` and the plain default on `generate` and
/// `chat`, the payload's `"model"` and the servable default on `batch` — which
/// is why none of them is inlined here.
///
/// Metadata only: a second cheap open of a file the loader is about to mmap
/// anyway, done BEFORE the load so a contradicting flag fails in milliseconds
/// rather than after 20 GB (111 on the default) is resident.
fn one_shot_checkpoint(
    model: Option<&Path>,
    selected: Option<Model>,
    selector: &str,
    default: Model,
) -> Result<Checkpoint> {
    let Some(path) = model else {
        let model = selected.unwrap_or(default);
        return Ok(Checkpoint::official(model));
    };
    let gguf = gguf::open(path, &candle_core::Device::Cpu)
        .with_context(|| format!("reading {}", path.display()))?;
    let cfg = XwenConfig::from_gguf(&gguf.content)
        .with_context(|| format!("reading {}", path.display()))?;
    Ok(match cfg.identify(path, selected, selector)? {
        Identity::Official(model) => Checkpoint::official(model),
        Identity::Assumed(assumed) => {
            // Said out loud because it decides the chat template dialect and the
            // drafter, and on the dense architecture the two 27B releases are a
            // coin-flip — the operator is the only one who can break that tie.
            eprintln!(
                "xwen: {} names no official checkpoint; running it as {} \
                 (pass {selector} to name it)",
                path.display(),
                assumed.full_name()
            );
            Checkpoint::assumed(assumed, path)
        }
    })
}

/// The checkpoint a one-shot run is against, and what to call it.
///
/// The two are not the same answer for a file that identifies as nothing: it
/// RUNS as some official checkpoint's graph, but it is not that checkpoint, and
/// a history that filed someone's finetune under `Qwen3.6-27B` would be wrong in
/// a way nothing downstream could undo. `serve` already draws this distinction
/// (`serve::model_id`), and the label here is the same string for the same file.
struct Checkpoint {
    /// What the run executes as: geometry, chat dialect, drafter sidecar.
    model: Model,
    /// What the metrics history calls it.
    label: String,
}

impl Checkpoint {
    fn official(model: Model) -> Self {
        Self {
            model,
            label: model.full_name().to_string(),
        }
    }

    /// A file that named no checkpoint answers under its own file name, as it
    /// does on the wire when a server is started with it.
    fn assumed(model: Model, path: &Path) -> Self {
        Self {
            model,
            label: path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_else(|| "xwen".to_string()),
        }
    }
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
                // A split checkpoint needs every shard beside the one the
                // loader opens, and the size quoted is the whole set's — so
                // say how many files that is rather than name one and leave
                // the figure looking like its size.
                let what = match size.files() {
                    [one] => one.to_string(),
                    shards => format!("{} ({} shards)", shards[0], shards.len()),
                };
                eprintln!(
                    "xwen: {}/{} is not in the Hugging Face cache; downloading ({}, resumes in place)",
                    size.repo(),
                    what,
                    size.size(),
                );
            }
            xwen::hub::ensure_model(size)
        }
    }
}

/// Whether a run that named no drafter decodes plain because the CHECKPOINT's
/// default says so.
///
/// The one outcome that is neither "asked for a drafter and got one" nor "there
/// was never a drafter to get", and so the one that needs a line of its own: the
/// sidecar is sitting in the cache, `--draft official` would attach it, and
/// silence about it would read as a missing file rather than a policy. False
/// once the run asks by name — an explicit request is honored on every
/// checkpoint whose graph has a verify seam, whatever its default is.
///
/// Extracted from [`configure_generator`] so the rule is testable without a
/// Metal device or a sidecar on disk.
fn draft_defaults_off(size: Model, explicit: bool) -> bool {
    !explicit && size.drafter_kind().is_some() && !size.draft_default_on()
}

/// A draft path of literally `official` (the serve config's opt-out default)
/// means the selected model's drafter, ensured in the Hugging Face cache like
/// the model itself. `None` when that checkpoint ships no sidecar — decoding
/// then runs plain, which is the only thing it could do.
fn resolve_draft(path: &std::path::Path, size: Model, explicit: bool) -> Result<Option<PathBuf>> {
    // A checkpoint whose graph has no verify seam cannot be drafted for by any
    // sidecar, so a custom `--draft <gguf>` is refused here rather than loaded
    // and rejected later by `attach_drafter`. The opt-out default asked for
    // nothing and degrades to plain decoding with a line, as it does for a
    // checkpoint that merely ships no sidecar.
    if !size.supports_drafting() {
        ensure!(!explicit, "{}", size.no_drafting_message());
        return Ok(None);
    }
    if path != std::path::Path::new("official") {
        return Ok(Some(path.to_path_buf()));
    }
    ensure_drafter_explicit(size, explicit)
}

/// The checkpoint's official sidecar, fetched on first use.
///
/// `explicit` is whether the run asked for it BY NAME (`--draft official`, as
/// opposed to the opt-out default, which asks for nothing). A named request that
/// cannot be honored is an error: answering it with a warning would leave a run
/// that was told to speculate quietly not doing so. The default degrades to
/// plain decoding instead, and says so.
/// No checkpoint currently reaches the refusal: every shipped one names a
/// sidecar. It stays because the POLICY is what matters and it is not about
/// today's registry — a named request that cannot be honored must fail rather
/// than warn — and because the alternative, deleting it, would have to be
/// rewritten the first time a release ships without one.
fn ensure_drafter_explicit(size: Model, explicit: bool) -> Result<Option<PathBuf>> {
    if explicit && size.drafter_file().is_none() {
        bail!(
            "--draft official: {} ships no drafter sidecar. Decode plain (drop the flag, \
             or --no-draft), or pass a drafter GGUF of your own",
            size.full_name()
        );
    }
    ensure_drafter(size)
}

/// The selected model's drafter sidecar, with the same download notice the
/// target gets — the sidecar belongs to one checkpoint and never transfers.
/// `None` for a checkpoint that ships none; saying what that costs is the
/// caller's, since only the caller knows whether it was about to decode with it.
fn ensure_drafter(size: Model) -> Result<Option<PathBuf>> {
    let Some(file) = size.drafter_file() else {
        return Ok(None);
    };
    if xwen::hub::cached_drafter(size).is_none() {
        eprintln!(
            "xwen: {}/{file} is not in the Hugging Face cache; downloading ({})",
            size.repo(),
            size.drafter_size().unwrap_or("unknown size"),
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
        Ok((response, label)) => {
            metrics::record_quietly(&batch_run_record(&response, &label));
            let mut stdout = std::io::stdout();
            writeln!(stdout, "{}", serde_json::to_string_pretty(&response)?)?;
            stdout.flush()?;
            Ok(())
        }
        Err(error) => {
            // A whole-request failure is a run that happened and produced
            // nothing, which is worth more in the history than a gap: the
            // counts are zero because none were ever measured.
            // The checkpoint is unknown here: most whole-request failures
            // happen before the payload has been read far enough to name one.
            let mut run = RunRecord::new("batch", "-");
            run.ok = false;
            metrics::record_quietly(&run);

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

/// `xwen stats`: read the metrics history, group it, print it.
///
/// The table goes to stdout and everything about it — where it was read from,
/// how much of it the filters kept, what did not parse — to stderr, so a piped
/// table is nothing but its own rows.
fn run_stats(query: &metrics::StatsQuery, json: bool) -> Result<()> {
    let Some(report) = metrics::report(query)? else {
        // Stdout stays machine-readable whatever happened: a caller parsing
        // `--json` gets an empty array rather than a sentence, and the reason
        // there is nothing to show goes to stderr with the rest of the notes.
        let note = match metrics::query_path(query) {
            Some(path) => format!("no metrics recorded yet ({})", path.display()),
            None => format!("metrics recording is off ({}=off)", metrics::METRICS_ENV),
        };
        if json {
            println!("[]");
            eprintln!("{note}");
        } else {
            println!("{note}");
        }
        return Ok(());
    };

    let mut stdout = std::io::stdout();
    if json {
        let rows = metrics::rows_json(&report.rows);
        writeln!(stdout, "{}", serde_json::to_string_pretty(&rows)?)?;
    } else if report.rows.is_empty() {
        writeln!(stdout, "no runs match")?;
    } else {
        write!(stdout, "{}", metrics::render_table(&report.rows, report.by))?;
    }
    stdout.flush()?;

    let mut footer = format!(
        "\n{} \u{b7} {} run{}",
        report.path.display(),
        report.matched,
        plural(report.matched)
    );
    if report.matched != report.records {
        footer.push_str(&format!(" of {}", report.records));
    }
    if report.excluded_by_tag > 0 {
        // Never silent: a history a sweep has run through holds hundreds of
        // records this report is not counting, and a reader comparing the
        // matched count against the file would otherwise have no way to know
        // why they disagree. What was left out depends on which way the filter
        // was pointed — under `--tag` the excluded runs are mostly real use —
        // so the line names the population rather than always saying "harness".
        let excluded = match &query.tag {
            metrics::TagFilter::Untagged => format!(
                "{} harness run{} excluded (--all-tags to include)",
                report.excluded_by_tag,
                plural(report.excluded_by_tag)
            ),
            metrics::TagFilter::Only(tag) => format!(
                "{} run{} outside --tag {tag} excluded",
                report.excluded_by_tag,
                plural(report.excluded_by_tag)
            ),
            // Nothing is excluded by tag under `--all-tags`, so this arm is
            // unreachable while `excluded_by_tag` is above zero.
            metrics::TagFilter::All => String::new(),
        };
        if !excluded.is_empty() {
            footer.push_str(&format!(" \u{b7} {excluded}"));
        }
    }
    if report.skipped > 0 {
        footer.push_str(&format!(
            " \u{b7} {} unreadable line{}",
            report.skipped,
            plural(report.skipped)
        ));
    }
    if !report.local_offset_known {
        // Otherwise a report bucketed in UTC because the offset could not be
        // read is indistinguishable from one on a machine that really is UTC.
        footer.push_str(" \u{b7} dates in UTC (local offset unavailable)");
    }
    eprintln!("{footer}");
    Ok(())
}

fn plural(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}

/// One whole batch run as the metrics history records it — one record for the
/// run, not one per item.
///
/// The three token counts are measured independently, because on a batch they
/// are not each other's complement.
///
/// The prompt is what the items logically asked for. The prefill is everything
/// the engine actually FORWARDED, which is `stats.prefill_tokens` plus the
/// shared prefix: the runner opens its own prefill accounting after the prefix
/// is already resident and reports that span separately as
/// `shared_prefix_tokens`/`snapshot_ms`. The seconds are folded the same way, so
/// the pair still describes a rate that was observed. On a SCORED batch this
/// exceeds the prompt outright — every teacher-forced trial is real forwarded
/// work against no prompt token — which is why the cache figure cannot be the
/// difference between the two and is summed from the items instead: the shared
/// prefix is prefilled once and restored for each item after, so the sum of the
/// items' own `cached_prefix_tokens` counts it one time too many.
fn batch_run_record(response: &BatchResponse, label: &str) -> RunRecord {
    let mut run = RunRecord::new("batch", label);
    run.prompt_tokens = response
        .items
        .iter()
        .map(|item| item.usage.prompt_tokens)
        .sum();
    let cached_per_item: usize = response
        .items
        .iter()
        .map(|item| item.usage.cached_prefix_tokens)
        .sum();
    run.cached_tokens = cached_per_item.saturating_sub(response.stats.shared_prefix_tokens);
    run.prefill_tokens = response.stats.prefill_tokens + response.stats.shared_prefix_tokens;
    run.prefill_secs = (response.stats.prefill_ms + response.stats.snapshot_ms) / 1000.0;
    run.decode_tokens = response.stats.decode_tokens;
    run.decode_secs = response.stats.decode_ms / 1000.0;
    run.items = Some(response.stats.items);
    // Every item failing is a failed run, however cleanly the machinery around
    // them worked. A batch that lost some of its items still did the rest.
    let failed = response
        .items
        .iter()
        .filter(|item| item.error.is_some())
        .count();
    run.ok = response.items.is_empty() || failed < response.items.len();
    run
}

/// Everything `run_batch` does that can fail as a whole request: parse stdin,
/// resolve the checkpoint the payload names, load it, run the batch.
///
/// The metrics label comes back beside the response because the two names can
/// differ: the response is labelled with the checkpoint the run answers AS,
/// while a custom GGUF is recorded under its own file name.
fn batch_request(
    model: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    moe_impl: &str,
    max_ctx: usize,
    draft: &DraftArgs,
) -> Result<(BatchResponse, String)> {
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
    // The payload names the checkpoint; `-m` still overrides the FILE, and when
    // it is given that file's own identity is what the run is against — the
    // payload's name (or its absence) is only the cross-check, exactly as
    // `--model-size` is for the other commands.
    //
    let named = request
        .model
        .is_some()
        .then(|| request.model())
        .transpose()?;
    // `default_servable` rather than `default`, because batch moves cache state
    // and that is the question the rule answers. The two agree today.
    let checkpoint = one_shot_checkpoint(
        model.as_deref(),
        named,
        "the request's \"model\" field",
        Model::default_servable(),
    )?;
    let size = checkpoint.model;

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
    let response = xwen::batch::run_batch(
        &mut generator,
        &request,
        load_ms,
        // The checkpoint the payload named, under its canonical name whichever
        // spelling the document used.
        size.full_name(),
        size,
        &mut xwen::batch::BatchHooks {
            progress: &mut progress,
            cancelled: &mut never,
        },
    )?;
    Ok((response, checkpoint.label))
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
/// drafter (custom `--draft` path, else the hub-ensured official one)
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

    // Speculation is opt-out per checkpoint: a zero-flag run speculates with the
    // official drafter on a checkpoint whose `draft_default_on` is true, which a
    // first run fetches into the Hugging Face cache (3.5 GB on the 27B, 3.2 GB
    // on the 3.8-27B, with the same download notice the target gets).
    // `--no-draft` decodes plain, and so does a checkpoint that ships no sidecar
    // — with a line saying so, unless the run asked for one by name.
    if let Some(draft) = draft.filter(|d| !d.no_draft) {
        // resolve_draft keeps `--draft official` — and the default, which is
        // that same symbolic path — meaning the same thing here as in the serve
        // config.
        let explicit = draft.draft.is_some();
        // Asked for nothing, on a checkpoint that ships a drafter it does not
        // attach unasked (the 35B-A3B since 2026-09-06). Decided BEFORE
        // `resolve_draft` so a default run does not fetch a sidecar it will not
        // load. A checkpoint with no sidecar at all falls through instead, so it
        // keeps saying the other, truer thing about itself below.
        if draft_defaults_off(size, explicit) {
            eprintln!(
                "xwen: {}",
                size.draft_default_off_message("pass --draft official")
            );
            return Ok(generator);
        }
        let requested = draft
            .draft
            .clone()
            .unwrap_or_else(|| PathBuf::from("official"));
        let Some(path) = resolve_draft(&requested, size, explicit)? else {
            // Two different facts, and the zero-flag default run hits the first
            // one: a checkpoint with no verify seam was never going to
            // speculate, so that is a statement of how it decodes, not a
            // warning that something went missing.
            if !size.supports_drafting() {
                eprintln!(
                    "xwen: {} decodes without speculation (no drafter exists for its graph yet)",
                    size.full_name()
                );
            } else {
                eprintln!(
                    "xwen: no drafter available for {}; decoding without speculation",
                    size.full_name()
                );
            }
            return Ok(generator);
        };
        let draft_start = std::time::Instant::now();
        let dgguf = gguf::open(&path, &device)?;
        // The drafter's cache is sized by --draft-ctx (capped at the target's
        // context), which is also the drafting depth limit — the same rule the
        // serve engine applies. Sizing it at the target's max_ctx would buy
        // 4-48 KiB/token of cache for depths where drafting never pays (and
        // OOMs outright at 262k alongside the 68 GB target).
        let draft_ctx = draft.draft_ctx.min(max_ctx);
        let drafter = load_drafter(&dgguf, &generator, &device, draft_ctx)
            .with_context(|| format!("loading the drafter {}", path.display()))?;
        generator.attach_drafter(drafter, draft.params(size))?;
        eprintln!(
            "xwen: drafter loaded in {:.1}s",
            draft_start.elapsed().as_secs_f64()
        );
    }

    Ok(generator)
}

/// Build the drafter an opened sidecar holds, whichever kind that is.
///
/// The kind comes from `drafter::classify`, the same classifier the server's
/// startup preflight uses — one classifier, so a file cannot be judged one kind
/// by the server and loaded as another here. `MtpDrafter::load` additionally
/// needs the target's config, because the head has no geometry of its own: it is
/// an extra trunk layer and reads its shapes from the trunk.
///
/// The full validation — that this sidecar actually fits this checkpoint — is
/// each kind's own `check_against_target`, which `Generator::attach_drafter`
/// runs on the way in.
fn load_drafter(
    gguf: &std::sync::Arc<xwen::gguf::GgufFile>,
    generator: &Generator,
    device: &candle_core::Device,
    draft_ctx: usize,
) -> Result<xwen::drafter::AttachedDrafter> {
    Ok(match xwen::drafter::classify(&gguf.content)? {
        xwen::drafter::DrafterKind::Dflash => {
            xwen::drafter::AttachedDrafter::Dflash(DflashDrafter::load(gguf, device, draft_ctx)?)
        }
        xwen::drafter::DrafterKind::Mtp => xwen::drafter::AttachedDrafter::Mtp(MtpDrafter::load(
            gguf,
            generator.model_config(),
            device,
            draft_ctx,
        )?),
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        // No subcommand: serve, with whatever serve flags were passed at the
        // top level.
        None => run_serve(cli.serve),
        Some(Cmd::Fetch { select }) => {
            let size = select.size();
            let model = resolve_model(None, size)?;
            println!("model    {}", model.display());
            match ensure_drafter(size)? {
                Some(drafter) => println!("drafter  {}", drafter.display()),
                None => println!("drafter  none ({} ships no sidecar)", size.full_name()),
            }
            Ok(())
        }
        Some(Cmd::Stats {
            by,
            since,
            model,
            surface,
            client,
            session,
            tag,
            all_tags,
            json,
            file,
        }) => {
            let query = metrics::StatsQuery {
                by: by.parse()?,
                since,
                model,
                surface,
                client,
                session,
                tag: match (tag, all_tags) {
                    (Some(tag), _) => metrics::TagFilter::Only(tag),
                    (None, true) => metrics::TagFilter::All,
                    (None, false) => metrics::TagFilter::Untagged,
                },
                file,
            };
            run_stats(&query, json)
        }
        Some(Cmd::Inspect { model, select }) => {
            let model = resolve_model(model, select.size())?;
            let device = candle_core::Device::Cpu;
            let gguf = gguf::open(&model, &device)?;
            print!("{}", gguf::describe_file(&gguf));
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
            think,
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
            // Both template knobs describe the chat template, which a raw
            // prompt never renders.
            if raw && think.no_think {
                bail!(
                    "--no-think is meaningless with --raw: it closes the chat template's \
                     <think> block, which a raw prompt never opens"
                );
            }
            if raw && think.reasoning_effort.is_some() {
                bail!(
                    "--reasoning-effort is meaningless with --raw: it selects the chat \
                     template's system preamble, which a raw prompt never renders"
                );
            }
            // Validated whether or not thinking is on, so a 3.6 run learns
            // about a useless flag at startup rather than never.
            think.check_think_budgets(min_think, max_think)?;
            // Read before the template knobs are resolved: with `--model` the
            // FILE decides which checkpoint this is, and the dialect, the
            // drafter and the effort preamble all key off that answer.
            let checkpoint = one_shot_checkpoint(
                model.as_deref(),
                select.model_size,
                "--model-size",
                Model::default(),
            )?;
            let size = checkpoint.model;
            let chat_opts = think.chat_options(size)?;
            let mut generator = build_generator(
                &resolve_model(model, size)?,
                size,
                tokenizer.as_deref(),
                &moe_impl,
                max_ctx,
                // The mode-dependent sampling defaults follow the resolved
                // thinking state; a raw prompt has no mode and keeps the
                // (thinking-set) historical default, --no-think being rejected
                // above. The checkpoint is threaded in for the one default that
                // is keyed to it, the presence penalty.
                sampling.options(size, !think.no_think),
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
                // The checkpoint's own template dialect, with the run's
                // thinking and effort knobs applied.
                build_prompt_with_spans(&[Message::User(prompt)], &chat_opts)?
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
            );
            // A failure is a run that happened and produced nothing, recorded
            // like the other surfaces record theirs before the error goes on.
            let gstats = match gstats {
                Ok(stats) => stats,
                Err(error) => {
                    let mut run = RunRecord::new("generate", checkpoint.label.clone());
                    run.ok = false;
                    metrics::record_quietly(&run);
                    return Err(error);
                }
            };
            println!();

            // Past here generation returned, which on this command means an EOG
            // token or the token cap: it polls no cancel, so Ctrl-C kills the
            // process outright and an interrupted run leaves no record at all.
            let mut run = RunRecord::new("generate", checkpoint.label.clone());
            // `generate` prefills from a reset cache, so nothing is ever read
            // back and the whole prompt is prefill.
            run.prompt_tokens = gstats.prefill_tokens;
            run.prefill_tokens = gstats.prefill_tokens;
            run.prefill_secs = gstats.prefill_secs;
            run.decode_tokens = gstats.decode_tokens;
            run.decode_secs = gstats.decode_secs;
            run.thinking_tokens = gstats.think.map(|think| think.tokens);
            run.drafted = gstats.spec.map(|spec| spec.drafted);
            run.accepted = gstats.spec.map(|spec| spec.accepted);
            metrics::record_quietly(&run);

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
            think,
            sampling,
            draft,
        }) => {
            // Validated before the 20 GB load, like every startup cross-check.
            think.check_think_budgets(min_think, max_think)?;
            let checkpoint = one_shot_checkpoint(
                model.as_deref(),
                select.model_size,
                "--model-size",
                Model::default(),
            )?;
            let size = checkpoint.model;
            let chat_opts = think.chat_options(size)?;
            let mut generator = build_generator(
                &resolve_model(model, size)?,
                size,
                tokenizer.as_deref(),
                &moe_impl,
                max_ctx,
                sampling.options(size, !think.no_think),
                Some(&draft),
            )?;
            generator.set_min_think(min_think);
            generator.set_max_think(max_think)?;
            generator.set_banned_strings(&ban_string)?;
            repl::run(
                &mut generator,
                max_tokens,
                show_thinking,
                chat_opts,
                &checkpoint.label,
            )
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

#[cfg(test)]
mod tests {
    use super::*;
    use xwen::batch::{BatchStats, FinishReason, ItemResponse, Usage};
    use xwen::chat::ChatDialect;

    /// The prefill a batch record reports is the whole prefill phase, which is
    /// the runner's own figure PLUS the shared prefix: the runner opens its
    /// accounting after the prefix is already resident and reports that span
    /// separately. Everything the prompt asked for beyond what was forwarded
    /// came back out of the snapshot.
    ///
    /// Eight items over a 1000-token shared prefix, each with a 500-token tail:
    /// 12,000 prompt tokens, the prefix prefilled once and read back seven
    /// times. The caller sets `stats.prefill_tokens` to whatever forwarded work
    /// it wants to describe.
    fn scored_response() -> BatchResponse {
        let item = |id: &str| ItemResponse {
            id: id.to_string(),
            content: String::new(),
            text: String::new(),
            json: None,
            finish_reason: FinishReason::Stop,
            usage: Usage {
                prompt_tokens: 1_500,
                cached_prefix_tokens: 1_000,
                completion_tokens: 150,
            },
            error: None,
        };
        BatchResponse {
            model: "Qwen3.6-35B-A3B".to_string(),
            stats: BatchStats {
                shared_prefix_tokens: 1_000,
                snapshot_ms: 4_000.0,
                items: 8,
                prefill_tokens: 4_000,
                prefill_ms: 16_000.0,
                decode_tokens: 1_200,
                decode_ms: 10_000.0,
                load_ms: 3_000.0,
                total_ms: 40_000.0,
            },
            items: (0..8).map(|n| item(&format!("item-{n}"))).collect(),
        }
    }

    #[test]
    fn a_batch_record_counts_the_shared_prefix_as_prefill_exactly_once() {
        // The unscored case: forwarded work is the tails and the prefix, and
        // nothing else, so it comes to exactly what the prompt asked for.
        let response = scored_response();
        let run = batch_run_record(&response, "Qwen3.6-35B-A3B");
        assert_eq!(run.surface, "batch");
        assert_eq!(run.items, Some(8));
        assert_eq!(run.prompt_tokens, 12_000);
        assert_eq!(
            run.prefill_tokens, 5_000,
            "the shared prefix once, plus every item's own tail"
        );
        assert_eq!(
            run.cached_tokens, 7_000,
            "the shared prefix read back once per item past the first"
        );
        assert_eq!(
            run.prompt_tokens,
            run.cached_tokens + run.prefill_tokens,
            "a batch that ran to completion read its whole prompt"
        );
        assert_eq!(
            run.prefill_secs, 20.0,
            "the shared prefill's seconds ride with its tokens"
        );
        assert_eq!(run.decode_tokens, 1_200);
        assert_eq!(run.decode_secs, 10.0);
    }

    /// A SCORED batch forwards more than its prompt: every teacher-forced trial
    /// runs through the model against no prompt token. The cache figure is
    /// summed from the items rather than taken as the difference, which would
    /// saturate to zero and hide a real cache hit. Same arithmetic the engine
    /// records for the same run.
    #[test]
    fn a_scored_batch_records_more_prefill_than_prompt() {
        let mut response = scored_response();
        // Engine work: the tails plus every teacher-forced trial, the shared
        // prefix excluded as the runner reports it.
        response.stats.prefill_tokens = 39_000;
        let run = batch_run_record(&response, "Qwen3.6-35B-A3B");
        assert_eq!(run.prompt_tokens, 12_000);
        assert_eq!(run.cached_tokens, 7_000, "the cache hit is unchanged");
        assert_eq!(run.prefill_tokens, 40_000, "39,000 plus the shared prefix");
        assert!(
            run.prefill_tokens > run.prompt_tokens,
            "scored work exceeds the prompt"
        );
    }

    /// Every item failing is a failed run. A batch that lost some of its items
    /// still did the rest, and its counts are real.
    #[test]
    fn a_batch_is_a_failure_only_when_every_item_failed() {
        let fail = |response: &mut BatchResponse, count: usize| {
            for item in response.items.iter_mut().take(count) {
                item.error = Some("no".to_string());
            }
        };
        let mut none_failed = scored_response();
        fail(&mut none_failed, 0);
        assert!(batch_run_record(&none_failed, "m").ok);

        let mut some_failed = scored_response();
        fail(&mut some_failed, 7);
        assert!(
            batch_run_record(&some_failed, "m").ok,
            "seven of eight failing still ran one"
        );

        let mut all_failed = scored_response();
        fail(&mut all_failed, 8);
        assert!(!batch_run_record(&all_failed, "m").ok);
    }

    /// The record is labelled with the checkpoint the run is against, which for
    /// a custom GGUF is not the name the response is labelled with: the
    /// response answers AS the official checkpoint whose graph ran.
    #[test]
    fn a_batch_record_takes_the_metrics_label_not_the_response_label() {
        let response = BatchResponse {
            model: "Qwen3.6-27B".to_string(),
            stats: BatchStats::default(),
            items: Vec::new(),
        };
        let run = batch_run_record(&response, "laguna-s-2.1-Q4_K_M");
        assert_eq!(run.model, "laguna-s-2.1-Q4_K_M");
        assert_eq!(
            response.model, "Qwen3.6-27B",
            "the wire document is untouched"
        );
    }

    /// With no `--model` file there is nothing to identify, so a one-shot runs
    /// what it was told to — and what "told to nothing" means differs per
    /// command, which is the whole reason the fallback is a parameter:
    /// `generate` and `chat` get the plain default, `batch` gets the one it can
    /// actually snapshot its way through.
    ///
    /// The file branch is the same rule serve applies and is pinned where that
    /// rule lives (`XwenConfig::identify`); it needs a real GGUF and is not
    /// re-tested here.
    #[test]
    fn without_a_file_a_one_shot_runs_what_it_was_told_to() {
        let resolved = |selected, default| {
            one_shot_checkpoint(None, selected, "--model-size", default).unwrap()
        };
        assert_eq!(resolved(None, Model::default()).model, Model::default());
        assert_eq!(
            resolved(None, Model::default_servable()).model,
            Model::default_servable()
        );
        // An explicit selection wins over either fallback.
        for default in [Model::default(), Model::default_servable()] {
            assert_eq!(
                resolved(Some(Model::Qwen27B), default).model,
                Model::Qwen27B
            );
        }
        // With no file to read, the label is the checkpoint the run is against.
        assert_eq!(
            resolved(Some(Model::Qwen27B), Model::default()).label,
            Model::Qwen27B.full_name()
        );
    }

    /// A GGUF that identifies as none of the official checkpoints RUNS as one
    /// of them, but the history must not file it under that name: it answers
    /// under its own file name, the same string `serve` reports for the same
    /// file. The identity rule itself needs a real GGUF and is pinned where it
    /// lives; this covers only the naming.
    #[test]
    fn a_file_that_names_no_checkpoint_is_recorded_under_its_own_name() {
        let assumed = Checkpoint::assumed(
            Model::Qwen27B,
            Path::new("/models/laguna-s-2.1-Q4_K_M.gguf"),
        );
        assert_eq!(assumed.model, Model::Qwen27B);
        assert_eq!(assumed.label, "laguna-s-2.1-Q4_K_M");
        assert_eq!(
            Checkpoint::official(Model::Qwen27B).label,
            Model::Qwen27B.full_name()
        );
    }

    /// A checkpoint whose graph has no speculative verify seam refuses a drafter
    /// it was ASKED for, and quietly decodes plain when it was not.
    ///
    /// The refusal has to happen at the flag, not at load: a custom sidecar
    /// would otherwise be fetched, opened and classified before
    /// `attach_drafter` rejected it, and the error would name a mismatch rather
    /// than the fact that this target cannot be drafted for at all.
    #[test]
    fn a_checkpoint_with_no_verify_seam_refuses_a_drafter_it_was_asked_for() {
        let official = std::path::Path::new("official");
        let custom = std::path::Path::new("/somebody/sidecar.gguf");

        // Asked by name, either spelling: an error naming the target.
        for path in [official, custom] {
            let err = resolve_draft(path, Model::Qwen38FlashNext, true)
                .unwrap_err()
                .to_string();
            assert!(err.contains("no drafter kind is supported"), "{err}");
            assert!(err.contains("Qwen3.8-Flash-Next"), "{err}");
        }

        // Not asked (the opt-out default): no drafter, no error — the caller
        // prints the "decoding without speculation" line.
        assert!(
            resolve_draft(official, Model::Qwen38FlashNext, false)
                .unwrap()
                .is_none()
        );

        // The drafting checkpoints are untouched by the same call: a custom
        // path is handed straight back, and `official` resolves to the sidecar.
        assert_eq!(
            resolve_draft(custom, Model::Qwen27B, true).unwrap(),
            Some(custom.to_path_buf())
        );
        assert!(Model::Qwen27B.supports_drafting());
    }

    /// What a zero-flag `generate`/`chat` run does per checkpoint, and what an
    /// explicit `--draft` does to that.
    ///
    /// The 35B-A3B is the interesting row: it ships a sidecar, `--draft
    /// official` still attaches it, and a run that asks for nothing decodes
    /// plain with a line — since 2026-09-06, when its drafted arm read below
    /// plain at every length.
    #[test]
    fn a_zero_flag_run_follows_the_checkpoints_own_drafting_default() {
        assert!(draft_defaults_off(Model::Qwen35BA3B, false));
        assert!(!draft_defaults_off(Model::Qwen27B, false));
        assert!(!draft_defaults_off(Model::Qwen3827B, false));
        // No sidecar at all: this is not the case that gets the line, because
        // the caller has a truer thing to say about a graph with no verify seam.
        assert!(!draft_defaults_off(Model::Qwen38FlashNext, false));

        // Asking by name is honored on every checkpoint, whatever its default.
        for model in [
            Model::Qwen27B,
            Model::Qwen35BA3B,
            Model::Qwen3827B,
            Model::Qwen38FlashNext,
        ] {
            assert!(!draft_defaults_off(model, true), "{model:?}");
        }

        // The line names the checkpoint and the flag that turns it back on, so
        // an operator who wanted the old behaviour can get it from the line
        // alone.
        let line = Model::Qwen35BA3B.draft_default_off_message("pass --draft official");
        assert!(line.contains("Qwen3.6-35B-A3B"), "{line}");
        assert!(line.contains("--draft official"), "{line}");
    }

    // Unset sampling flags resolve to the recommended set for the run's
    // thinking mode — 1.0/20/0.95 thinking, 0.7/20/0.80 with --no-think — and
    // a flag that was passed wins in either mode.
    #[test]
    fn sampling_flags_resolve_against_the_modes_recommendation() {
        let unset = SamplingArgs {
            temp: None,
            top_k: None,
            top_p: None,
            presence_penalty: None,
            seed: 42,
        };
        let thinking = unset.options(Model::Qwen27B, true);
        assert_eq!(
            (thinking.temperature, thinking.top_k, thinking.top_p),
            (1.0, 20, 0.95)
        );
        let instruct = unset.options(Model::Qwen27B, false);
        assert_eq!(
            (instruct.temperature, instruct.top_k, instruct.top_p),
            (0.7, 20, 0.80)
        );

        let pinned = SamplingArgs {
            temp: Some(0.5),
            top_k: None,
            top_p: Some(0.9),
            presence_penalty: None,
            seed: 7,
        };
        let resolved = pinned.options(Model::Qwen27B, false);
        assert_eq!(resolved.temperature, 0.5);
        assert_eq!(resolved.top_k, 20, "an unset flag keeps the mode default");
        assert_eq!(resolved.top_p, 0.9);
        assert_eq!(resolved.seed, 7);
    }

    // The presence penalty is the one card value keyed to the CHECKPOINT as
    // well as the mode, so the same unset flags resolve differently per model.
    #[test]
    fn the_presence_penalty_default_follows_the_checkpoint_and_the_mode() {
        let unset = SamplingArgs {
            temp: None,
            top_k: None,
            top_p: None,
            presence_penalty: None,
            seed: 42,
        };
        for size in [Model::Qwen27B, Model::Qwen3827B, Model::Qwen38FlashNext] {
            assert_eq!(
                unset.options(size, true).presence_penalty,
                0.0,
                "{size:?} asks for no penalty while thinking"
            );
            assert_eq!(unset.options(size, false).presence_penalty, 1.5, "{size:?}");
        }
        // The 35B-A3B card is the one that asks for a penalty in both modes.
        assert_eq!(unset.options(Model::Qwen35BA3B, true).presence_penalty, 1.5);

        // An explicit flag wins over the card, a zero included.
        let off = SamplingArgs {
            temp: None,
            top_k: None,
            top_p: None,
            presence_penalty: Some(0.0),
            seed: 42,
        };
        assert_eq!(off.options(Model::Qwen35BA3B, true).presence_penalty, 0.0);
    }

    // --reasoning-effort is a 3.8 template parameter: supplied on a 3.6
    // checkpoint it is a startup error naming the checkpoint, on the 3.8 it
    // lands in the options, and unset it is the template default everywhere.
    #[test]
    fn reasoning_effort_is_refused_on_a_36_checkpoint() {
        let supplied = ThinkArgs {
            no_think: false,
            reasoning_effort: Some(EffortArg::Low),
        };
        for size in [Model::Qwen27B, Model::Qwen35BA3B] {
            let error = supplied.chat_options(size).unwrap_err().to_string();
            assert!(error.contains(size.full_name()), "{error}");
            assert!(error.contains("Qwen 3.8 template feature"), "{error}");
        }
        let opts = supplied.chat_options(Model::Qwen3827B).unwrap();
        assert_eq!(opts.dialect, ChatDialect::Qwen38);
        assert_eq!(opts.reasoning_effort, ReasoningEffort::Low);
        assert!(opts.enable_thinking);

        let unset = ThinkArgs {
            no_think: true,
            reasoning_effort: None,
        };
        let opts = unset
            .chat_options(Model::Qwen27B)
            .expect("the default level renders nothing on 3.6, so nothing to refuse");
        assert!(!opts.enable_thinking);
        assert_eq!(opts.reasoning_effort, ReasoningEffort::Xhigh);
    }

    // The think budgets govern the <think> block that --no-think closes before
    // the reply begins, so arming either alongside it is a startup error;
    // either side alone is fine.
    #[test]
    fn think_budgets_are_refused_with_no_think() {
        let no_think = ThinkArgs {
            no_think: true,
            reasoning_effort: None,
        };
        let error = no_think
            .check_think_budgets(128, 0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("--min-think"), "{error}");
        let error = no_think
            .check_think_budgets(0, 4096)
            .unwrap_err()
            .to_string();
        assert!(error.contains("--max-think"), "{error}");
        no_think
            .check_think_budgets(0, 0)
            .expect("no armed budget, nothing to refuse");

        let thinking = ThinkArgs {
            no_think: false,
            reasoning_effort: None,
        };
        thinking
            .check_think_budgets(128, 4096)
            .expect("with thinking on the budgets govern a real block");
    }
}
