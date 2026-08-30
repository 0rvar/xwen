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
use xwen::mtp::MtpDrafter;
use xwen::ops::ExpertRunner;
use xwen::sampler::SamplerOptions;
use xwen::serve::config::{CliOverrides, DraftMode, ServeToml};

#[derive(Parser)]
#[command(
    name = "xwen",
    about = "Qwen inference on Metal, defaulting to Qwen3.8-Flash-Next. Bare \
             `xwen` serves over HTTP with the live dashboard (which serves \
             Qwen3.6-35B-A3B until Flash-Next is servable); subcommands cover \
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
    /// Which official checkpoint to run: Qwen3.8-Flash-Next (`flash-next` —
    /// the default, and EXPERIMENTAL; `xwen generate` and `xwen chat` only,
    /// since `xwen serve` and `xwen batch` both move cache state it cannot
    /// carry yet), the dense Qwen3.6-27B, the Qwen3.6-35B-A3B MoE, or the dense
    /// Qwen3.8-27B. `xwen serve` and `xwen batch` default to Qwen3.6-35B-A3B
    /// instead, because neither can run the default yet. Each checkpoint's full
    /// name works here too. A `--model <gguf>` path overrides the target file
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
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

impl SamplingArgs {
    /// Resolve the flags over the recommended set for `thinking`. A caller
    /// with no chat mode at all (a raw prompt) passes `true`: the thinking set
    /// is the historical default, so those paths sample as they always have.
    fn options(&self, thinking: bool) -> SamplerOptions {
        let recommended = SamplerOptions::recommended(thinking);
        SamplerOptions {
            temperature: self.temp.unwrap_or(recommended.temperature),
            top_k: self.top_k.unwrap_or(recommended.top_k),
            top_p: self.top_p.unwrap_or(recommended.top_p),
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
/// otherwise, and `--draft <gguf>` swaps in a custom one.
#[derive(Parser)]
struct DraftArgs {
    /// Speculate with a custom drafter GGUF — either kind, a DFlash sidecar or
    /// an MTP head — instead of the checkpoint's official one (which the literal
    /// `official`, and the default, select).
    #[arg(long, value_name = "GGUF", conflicts_with = "no_draft")]
    draft: Option<PathBuf>,
    /// Decode without speculation. Drafting is on by default: measured faster on
    /// every checkpoint, at each one's fitted defaults — 27B +46 to +52%,
    /// 35B-A3B +26 to +28% on code and +15 to +17% on chat (both 2026-08-08),
    /// 3.8-27B +44 to +45% on code and +37 to +38% on chat (2026-08-15). See
    /// docs/decisions.md, "Speculative decoding".
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
        #[arg(long, default_value_t = 131072)]
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
        #[arg(long, default_value_t = 131072)]
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
    /// Because those snapshots are how a batch runs at all, this surface cannot
    /// run Qwen3.8-Flash-Next: a payload naming it is refused up front, and a
    /// payload naming nothing gets Qwen3.6-35B-A3B (serve's default too) with a
    /// line saying so. Sampling defaults to greedy and thinking to off, so a
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
        #[arg(long, default_value_t = 131072)]
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
    /// server's default checkpoint from the Hugging Face cache — Qwen3.6-35B-A3B,
    /// not the CLI's Qwen3.8-Flash-Next, which the server cannot run yet).
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
    /// Custom drafter GGUF to speculate with — either kind, a DFlash sidecar or
    /// an MTP head — in place of the checkpoint's official one (which the
    /// literal `official`, and the default, select — fetched into the Hugging
    /// Face cache on first use).
    /// Greedy output matches decoding without it except where a near-tie lands
    /// differently.
    #[arg(long, value_name = "GGUF", conflicts_with = "no_draft")]
    draft: Option<PathBuf>,
    /// Decode without speculation. Drafting is on by default: measured faster on
    /// every checkpoint, at each one's fitted defaults — 27B +46 to +52%,
    /// 35B-A3B +26 to +28% on code and +15 to +17% on chat (both 2026-08-08),
    /// 3.8-27B +44 to +45% on code and +37 to +38% on chat (2026-08-15). It
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
    // Before anything is resolved or fetched: a `--model-size` naming a
    // checkpoint the server cannot run is refused here rather than after a
    // 111 GB download and a load that fails every request.
    if let Some(size) = selected {
        ensure!(size.servable(), "{}", size.unservable_message());
    }
    let mut overrides = args.overrides();
    // Neither the CLI nor the config named a model: serve the hub-cached
    // official checkpoint. Injected into the CLI side of the merge (rather than
    // inside `resolve`) so config resolution itself stays pure and testable.
    if overrides.model.is_none() && file.model.is_none() {
        // A run that named nothing asked for no checkpoint in particular, so an
        // unservable default is not a refusal — the server falls back to the
        // best checkpoint it can run and says which, and why, since otherwise
        // the surprise is a different model answering than the CLI would.
        // The notice belongs here rather than beside the check above: a config
        // that names its own model file is not falling back to anything.
        let size = selected.unwrap_or_else(|| {
            let default = Model::default();
            if default.servable() {
                return default;
            }
            let fallback = Model::default_servable();
            eprintln!(
                "xwen: {} cannot be served yet ({}); serving {}. Pass --model-size to choose.",
                default.full_name(),
                default.unservable_reason(),
                fallback.full_name(),
            );
            fallback
        });
        overrides.model = Some(resolve_model(None, size)?);
    }

    let (settings, warnings) = xwen::serve::config::resolve(&file, source, &overrides)?;
    for warning in &warnings {
        eprintln!("{warning}");
    }
    if !settings.model.is_file() {
        bail!("model {} does not exist", settings.model.display());
    }

    // The served FILE decides, not the name: someone's own conversion onto the
    // qwen4exp graph is as unservable as the official checkpoint is, for the
    // same reason, and it identifies as no checkpoint at all — `identify_
    // checkpoint` falls it back to `Arch::model()`, which is the qwen4exp
    // checkpoint, which is unservable. Read once here and reused by the drafter
    // prefetch below.
    let served_cfg =
        XwenConfig::from_gguf(&gguf::open(&settings.model, &candle_core::Device::Cpu)?.content)
            .with_context(|| format!("reading {}", settings.model.display()))?;
    let (served_target, _) =
        xwen::serve::engine::identify_checkpoint(&settings, &served_cfg, selected)?;
    if !served_target.model.servable() {
        bail!("{}", served_target.model.unservable_message());
    }

    match &settings.draft {
        DraftMode::Off => {}
        DraftMode::Custom(path) => {
            if !path.is_file() {
                bail!("drafter {} does not exist", path.display());
            }
        }
        DraftMode::Official => {
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
) -> Result<Model> {
    let Some(path) = model else {
        return Ok(selected.unwrap_or(default));
    };
    let gguf = gguf::open(path, &candle_core::Device::Cpu)
        .with_context(|| format!("reading {}", path.display()))?;
    let cfg = XwenConfig::from_gguf(&gguf.content)
        .with_context(|| format!("reading {}", path.display()))?;
    Ok(match cfg.identify(path, selected, selector)? {
        Identity::Official(model) => model,
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
            assumed
        }
    })
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
    // The payload names the checkpoint; `-m` still overrides the FILE, and when
    // it is given that file's own identity is what the run is against — the
    // payload's name (or its absence) is only the cross-check, exactly as
    // `--model-size` is for the other commands.
    //
    // `BatchRequest::model` has already refused a checkpoint batch cannot run,
    // so what is left to check here is a custom GGUF that turns out to BE one.
    let named = request
        .model
        .is_some()
        .then(|| request.model())
        .transpose()?;
    if model.is_none() && named.is_none() && !Model::default().servable() {
        // Nothing named a checkpoint, so an unrunnable default is a fallback
        // rather than a refusal — the same rule, notice and reason a zero-flag
        // `xwen serve` prints, because otherwise a different model answers here
        // than the one `xwen generate` would.
        eprintln!(
            "xwen: {} cannot run under `xwen batch` yet ({}); running {}. \
             Name a checkpoint in the request's \"model\" field to choose.",
            Model::default().full_name(),
            Model::default().unservable_reason(),
            Model::default_servable().full_name(),
        );
    }
    let size = one_shot_checkpoint(
        model.as_deref(),
        named,
        "the request's \"model\" field",
        Model::default_servable(),
    )?;
    ensure!(size.servable(), "batch: {}", size.unbatchable_message());

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
        // The checkpoint the payload named, under its canonical name whichever
        // spelling the document used.
        size.full_name(),
        size.chat_dialect(),
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

    // Speculation is opt-out: a zero-flag run speculates with the official
    // drafter, which a first run fetches into the Hugging Face cache (3.5 GB on
    // the 27B, 0.8 GB on the 35B-A3B, with the same download notice the target
    // gets). `--no-draft` decodes plain, and so does a checkpoint that ships no
    // sidecar — with a line saying so, unless the run asked for one by name.
    if let Some(draft) = draft.filter(|d| !d.no_draft) {
        // resolve_draft keeps `--draft official` — and the default, which is
        // that same symbolic path — meaning the same thing here as in the serve
        // config.
        let explicit = draft.draft.is_some();
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
            let size = one_shot_checkpoint(
                model.as_deref(),
                select.model_size,
                "--model-size",
                Model::default(),
            )?;
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
                // above.
                sampling.options(!think.no_think),
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
            think,
            sampling,
            draft,
        }) => {
            // Validated before the 20 GB load, like every startup cross-check.
            think.check_think_budgets(min_think, max_think)?;
            let size = one_shot_checkpoint(
                model.as_deref(),
                select.model_size,
                "--model-size",
                Model::default(),
            )?;
            let chat_opts = think.chat_options(size)?;
            let mut generator = build_generator(
                &resolve_model(model, size)?,
                size,
                tokenizer.as_deref(),
                &moe_impl,
                max_ctx,
                sampling.options(!think.no_think),
                Some(&draft),
            )?;
            generator.set_min_think(min_think);
            generator.set_max_think(max_think)?;
            generator.set_banned_strings(&ban_string)?;
            repl::run(&mut generator, max_tokens, show_thinking, chat_opts)
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
    use xwen::chat::ChatDialect;

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
        assert_eq!(
            one_shot_checkpoint(None, None, "--model-size", Model::default()).unwrap(),
            Model::default()
        );
        assert_eq!(
            one_shot_checkpoint(None, None, "--model-size", Model::default_servable()).unwrap(),
            Model::default_servable()
        );
        // An explicit selection wins over either fallback, and is the only thing
        // that can name the unservable checkpoint on a one-shot.
        for default in [Model::default(), Model::default_servable()] {
            assert_eq!(
                one_shot_checkpoint(None, Some(Model::Qwen27B), "--model-size", default).unwrap(),
                Model::Qwen27B
            );
        }
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

    // Unset sampling flags resolve to the recommended set for the run's
    // thinking mode — 1.0/20/0.95 thinking, 0.7/20/0.80 with --no-think — and
    // a flag that was passed wins in either mode.
    #[test]
    fn sampling_flags_resolve_against_the_modes_recommendation() {
        let unset = SamplingArgs {
            temp: None,
            top_k: None,
            top_p: None,
            seed: 42,
        };
        let thinking = unset.options(true);
        assert_eq!(
            (thinking.temperature, thinking.top_k, thinking.top_p),
            (1.0, 20, 0.95)
        );
        let instruct = unset.options(false);
        assert_eq!(
            (instruct.temperature, instruct.top_k, instruct.top_p),
            (0.7, 20, 0.80)
        );

        let pinned = SamplingArgs {
            temp: Some(0.5),
            top_k: None,
            top_p: Some(0.9),
            seed: 7,
        };
        let resolved = pinned.options(false);
        assert_eq!(resolved.temperature, 0.5);
        assert_eq!(resolved.top_k, 20, "an unset flag keeps the mode default");
        assert_eq!(resolved.top_p, 0.9);
        assert_eq!(resolved.seed, 7);
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
