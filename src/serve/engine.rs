//! Inference engine thread: owns the (lazily loaded) model, enforces one-inference-at-a-
//! time by construction, maintains the KV prefix cache, and applies idle unload.

use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, ensure};
use candle_core::Device;
use tokio::sync::mpsc::Sender;
use tokio::sync::mpsc::error::TrySendError;

use super::config::{DraftMode, ServeSettings};
use super::disk_cache::DiskImage;
use super::disk_tier::{DiskCache, DiskCandidate, PendingSlot};
use super::log::{JobPhase, JobRecord, ServeLog, ServeLogger, SlotSummary};
use super::queue::{JobQueue, Queued};
use super::types::{
    BatchJob, Cancel, CancelReason, EngineEvent, GenerationJob, Job, RequestOrigin, ResidentModel,
    StopKind, Target,
};
use crate::XwenConfig;
use crate::batch::{BatchHooks, BatchProgress};
use crate::config::Identity;
use crate::dflash::{DflashConfig, DflashDrafter, DrafterImage, DrafterImageKind};
use crate::drafter::DrafterKind;
use crate::generate::{GenEvent, Generator, SpecParams, SpecStats, feasible_think_budget};
use crate::gguf;
use crate::hub;
use crate::kv_cache::{HostFullKv, HostSnapshot};
use crate::mtp::{MtpConfig, MtpDrafter};
use crate::ops::ExpertRunner;
use crate::sampler::SamplerOptions;
use crate::tokenizer::{LagunaTokenizer, Specials};

/// How much prefill a mid-prefill snapshot has to save before it is worth taking.
///
/// The cost is fixed — every DeltaNet layer's recurrent state, 62.8 MiB of host RAM on
/// the 35B-A3B and 149.6 MiB on the 27B (`hub::Model::snapshot_bytes`), whatever
/// position it covers — while the saving is the prefill between the deepest
/// position a later request could already resume from and this one, about 3 seconds
/// per 1024 tokens at low-power rates. Both kinds of stop are held to it: the anchor
/// at the end of a leading system block, whose beneficiary is a fresh conversation
/// starting from zero, and the branch point where a prompt forks off the conversation
/// it matched, whose beneficiary is a future fork starting from the deepest snapshot
/// that already exists below it. A later phase may fold this into the config,
/// alongside the disk cache's own minimum image size.
const SNAPSHOT_MIN_GAIN: usize = 1024;

/// How long the engine waits for the disk tier to finish writing on the way down —
/// a graceful shutdown or an idle unload. Bounded, and deliberately inside the
/// server's own 30 s shutdown grace period: an image that does not make it costs
/// the next server a re-prefill, while a shutdown that hangs costs the operator a
/// `kill -9`. Sized for real agent conversations: a 55k-token Claude Code session
/// pages out as a ~4.2 GiB image that takes ~5 s to write, and a shutdown write
/// that also splits a stored segment writes the base and two tails — the last
/// conversation's image only ever gets this window, so it must fit several such
/// writes, not one.
const DISK_FLUSH_GRACE: Duration = Duration::from_secs(25);

/// What the disk tier writes at, in bytes per second, as a FLOOR rather than a
/// typical rate. Measured 2026-09-06 alongside the long-context envelope
/// (docs/perf-state.md, "Long context"): the page-out lines report their own
/// bytes and milliseconds, and this is the slowest of them rounded down.
///
/// It exists only to turn an image size into a wait, so being pessimistic costs
/// a shutdown a few seconds it will not use and being optimistic costs the next
/// server a re-prefill.
const DISK_WRITE_FLOOR_BYTES_PER_SEC: u64 = 700 * 1024 * 1024;

/// How long to wait for `pending` bytes of queued images to land.
///
/// [`DISK_FLUSH_GRACE`] was sized on ONE ~4.2 GiB image and is the floor here,
/// not the answer: a 131072-token conversation images several times that, and a
/// shutdown that also splits a stored segment queues more than one. So the wait
/// is the pending bytes at the measured write floor, and never less than the
/// grace. It stays bounded — an image that does not land costs a re-prefill,
/// nothing more — and the expiry is still reported rather than retried.
pub(super) fn disk_flush_budget(pending: u64) -> Duration {
    Duration::from_secs(pending / DISK_WRITE_FLOOR_BYTES_PER_SEC).max(DISK_FLUSH_GRACE)
}

/// How long one event may fail to reach the client before the connection counts
/// as dead. A slow reader filling the channel is ordinary backpressure and
/// should throttle generation, but a half-open socket never drains, and the one
/// inference thread cannot be parked on it forever. The deadline is per event
/// while the job is live, so a client that keeps reading — however slowly —
/// keeps its generation; once the job is cancelled, every remaining send shares
/// one budget of this size instead (see [`Abandon::send`]).
const SEND_DEADLINE: Duration = Duration::from_secs(120);

/// Poll interval while a client's event channel is full.
const SEND_RETRY_INTERVAL: Duration = Duration::from_millis(5);

/// The wall-clock ceiling for one job, derived from the two sizes known when it
/// starts and the configured watchdog rates (`request_prefill_rate` /
/// `request_decode_rate` / `request_slack`). Monotonic in both sizes: a bigger
/// prompt or a bigger reply budget always buys more time, so legitimate spans
/// three orders of magnitude apart each get a ceiling that fits them.
///
/// A rate of 0 makes its term unbounded, and since the ceiling is one instant
/// an unbounded term is an unbounded sum: either rate at 0 yields `None`, no
/// deadline at all. A sum too large for an `Instant` to hold — the config
/// accepts slack up to u64::MAX seconds — is the same answer: a multi-century
/// ceiling and no ceiling keep the same promise, and the arithmetic must never
/// panic the engine thread.
fn job_deadline(
    now: Instant,
    prompt_tokens: usize,
    max_new: usize,
    settings: &ServeSettings,
) -> Option<Instant> {
    if settings.request_prefill_rate == 0 || settings.request_decode_rate == 0 {
        return None;
    }
    let prefill =
        Duration::try_from_secs_f64(prompt_tokens as f64 / settings.request_prefill_rate as f64)
            .ok()?;
    let decode =
        Duration::try_from_secs_f64(max_new as f64 / settings.request_decode_rate as f64).ok()?;
    now.checked_add(prefill)?
        .checked_add(decode)?
        .checked_add(settings.request_slack)
}

/// `<tool_call>` and `</tool_call>`, the two added tokens that bracket a tool
/// call. Everything between them is ordinary BPE text — the interior markers
/// below are not in the vocabulary — so the span is entered and left by token
/// id and parsed as text in between.
///
/// The ids come from the running tokenizer's [`Specials`], which owns every
/// token id in this crate — checkpoint families number the same markers
/// differently, so the parser reads them off the vocabulary it is decoding
/// against. Spelling them out here once cost a parser that opened a span on
/// every `:` in ordinary prose; a span marker is a vocabulary fact, and the
/// vocabulary has exactly one owner.
const TOOL_CALL_OPEN_TEXT: &str = "<tool_call>";
const TOOL_CALL_CLOSE_TEXT: &str = "</tool_call>";

/// The interior grammar of a Qwen tool call, which `chat.rs` renders and this
/// parser reads back:
///
/// ```text
/// <tool_call>
/// <function=NAME>
/// <parameter=KEY>
/// VALUE
/// </parameter>
/// </function>
/// </tool_call>
/// ```
///
/// A name and a key each run to the first `>`; the template interpolates them
/// raw and defines no escape, so the first `>` is the only terminator there is.
/// A value is framed by the newline after its key's `>` and the newline before
/// `</parameter>`, and both belong to the framing rather than to the value.
const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";
/// Terminator of a `<function=` or `<parameter=` tag.
const TAG_CLOSE: &str = ">";
/// What a streaming value has to hold back: the newline before `</parameter>`
/// is framing, so it cannot be delivered until the text after it proves it was
/// content.
const VALUE_CLOSE: &str = "\n</parameter>";

/// After a tool call the model often writes `</assistant>` as ordinary text
/// instead of emitting token 24, and then keeps going. Treated as a stop
/// sequence for a job that carries tools, exactly as the fork does — but never
/// reported as one, since the client never asked for it.
const ASSISTANT_CLOSE: &str = "</assistant>";

/// What the served GGUF is: the target every request for this server's own model
/// id runs on, plus a warning when the file named no official checkpoint.
///
/// The file is the authority. An explicit `--model-size` is a tie-break for a
/// file that identifies as NOTHING, not an override of one that does — a flag
/// that contradicts the file is a startup error naming both sides, because the
/// alternative is a server that starts fine and 500s every request at load. It
/// must also agree with the architecture, which no name can change.
///
/// The returned target is `Target::official` when the file is one of the
/// checkpoints and `Target::served` when it is not: only the second answers to a
/// file name, and only the first lets an official name resolve locally.
///
/// The rule itself is [`XwenConfig::identify`], shared with the one-shot
/// commands so a custom GGUF is not one checkpoint to the server and another to
/// `xwen generate`. What is serve's own is the mapping onto `Target` and the
/// startup log line.
pub fn identify_checkpoint(
    settings: &ServeSettings,
    cfg: &XwenConfig,
    selected: Option<hub::Model>,
) -> Result<(Target, Option<ServeLog>)> {
    match cfg.identify(&settings.model, selected, "--model-size")? {
        Identity::Official(model) => Ok((Target::official(model), None)),
        Identity::Assumed(assumed) => Ok((
            Target::served(assumed),
            Some(ServeLog::CheckpointUnidentified {
                path: settings.model.clone(),
                assumed,
            }),
        )),
    }
}

/// Metadata-only read of the default checkpoint, done once at startup so the
/// identity it decides — which model id the APIs speak, which sidecar the
/// drafter comes from — is read from one look at the file.
pub fn read_startup_config(settings: &ServeSettings) -> Result<XwenConfig> {
    read_config(&settings.model)
}

/// Cheap startup validation: judges the already-parsed metadata (no tensor data, no
/// Metal allocation) and loads the tokenizer. Fails fast so a bad model path or config
/// is caught at startup rather than on the first request. Returns the tokenizer the
/// HTTP layer renders prompts with — the same vocabulary the engine decodes with (every
/// checkpoint shares it) — and the resolved context length its "does the prompt fit"
/// check applies.
pub fn validate_model(
    settings: &ServeSettings,
    cfg: &XwenConfig,
    served: Target,
    logger: &ServeLogger,
) -> Result<(Arc<LagunaTokenizer>, usize)> {
    let (max_ctx, warning) = resolve_context_length(settings.context_length, cfg.n_ctx_train)?;
    if let Some(warning) = warning {
        logger.log(warning);
    }
    // A drafter that turns out not to fit the model is a configuration mistake, and
    // the first request is far too late to learn it: the load would fail behind a
    // request that had nothing to do with it, and keep failing, forever.
    //
    // Two drafters can be judged at startup. A custom one, which is a path the
    // operator chose. And the SERVED checkpoint's official sidecar, which the CLI has
    // already fetched by now — worth checking because "official" does not mean "fits":
    // a custom GGUF served as its architecture's checkpoint gets that checkpoint's
    // sidecar, and if the file's geometry differs at all, every request fails at
    // attach. Any OTHER checkpoint's sidecar cannot be judged here — it may not even
    // be downloaded yet — and is checked when that checkpoint attaches it.
    if let Some(path) = startup_drafter(settings, served) {
        read_draft_config(&path, cfg)
            .with_context(|| format!("validating the drafter {}", path.display()))?;
    }
    Ok((Arc::new(load_tokenizer()?), max_ctx))
}

/// Which drafter startup can judge, or `None` when none can be.
///
/// Extracted from [`validate_model`] so the selection is testable without a
/// checkpoint on disk — the rule matters more than the reading of it.
fn startup_drafter(settings: &ServeSettings, served: Target) -> Option<std::path::PathBuf> {
    match &settings.draft {
        DraftMode::Off => None,
        DraftMode::Custom(path) => Some(path.clone()),
        // Nothing asked, and this checkpoint does not draft unasked: there is
        // no drafter for startup to judge, and fetching one to judge it would
        // be work for a file no request will attach.
        DraftMode::Default if !served.model.draft_default_on() => None,
        // The served checkpoint's own sidecar, offline: whatever the CLI's
        // prefetch left in the cache. A checkpoint that ships none yields
        // nothing to check, and on a cache miss the attach's own check is the
        // only one there can be.
        DraftMode::Default | DraftMode::Official => hub::cached_drafter(served.model),
    }
}

/// Spawns the dedicated inference thread. The thread lazily loads whichever checkpoint
/// the picked job names (the default one for the compat dialects, any for the batch
/// route), stamps the resident checkpoint on load and clears it on unload, swaps
/// checkpoints when a job needs the
/// other one, and drops the model after `idle_unload` of inactivity (None = never).
/// `default_target` is what `settings.model` is: the official checkpoint it identified
/// as, or — for a GGUF that identified as none of them — the served file running as its
/// architecture's checkpoint. It is the only target the served file answers for.
/// `shutdown` is the process-wide cancel token: once it fires, the running job aborts
/// and queued jobs are dropped unstarted.
pub(super) fn spawn_engine(
    settings: ServeSettings,
    default_target: Target,
    jobs: Arc<JobQueue>,
    resident: Arc<ResidentModel>,
    shutdown: Arc<Cancel>,
    disk_pending: PendingSlot,
    logger: ServeLogger,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name(ENGINE_THREAD.to_string())
        .spawn(move || {
            engine_loop(
                settings,
                default_target,
                jobs,
                resident,
                shutdown,
                disk_pending,
                logger,
            )
        })
        .expect("spawning the inference thread")
}

/// The inference thread's name. It is how a panic hook tells a panic this thread
/// will catch and recover from — a bad drafter fails one request and the server
/// keeps serving — from one nobody is going to catch.
pub(super) const ENGINE_THREAD: &str = "engine";

/// Metadata-only read of the checkpoint. The CPU device skips the mmap aliasing the
/// Metal load path sets up, so this touches the header and tensor index alone.
fn read_config(path: &Path) -> Result<XwenConfig> {
    // No registry entry: this is a metadata read of a file the server was
    // pointed at, and the identity it feeds is what decides which entry it is.
    // The only thing the entry would buy here is a safetensors set's zero-run
    // allowlist, and a set that fails that check is one no request could run.
    crate::checkpoint::CheckpointSource::open(path, &Device::Cpu, None)?.config()
}

/// Metadata-only read of a drafter sidecar, on the CPU device for the same
/// reason [`read_config`] uses it: header and tensor index only, no Metal
/// allocation.
///
/// The classification is [`crate::drafter::classify`]'s — shared with the CLI's
/// attach path, so a sidecar cannot be judged one kind here and loaded as
/// another there. Whichever kind it is, it is then judged against `target` by
/// that kind's own checks, hoisted to startup. A drafter that turns out not to
/// describe a wiring this model can provide is a configuration mistake, and the
/// lazy load is far too late to learn it: that load runs behind the panic
/// boundary around a job, so the mistake would not kill the engine — it would
/// fail a request, and every retry after it, forever.
fn read_draft_config(path: &Path, target: &XwenConfig) -> Result<DrafterKind> {
    let gguf = gguf::open(path, &Device::Cpu)?;
    let kind = crate::drafter::classify(&gguf.content)?;
    match kind {
        DrafterKind::Dflash => {
            DflashConfig::from_gguf(&gguf.content)?.check_against_target(
                target.hidden,
                target.n_layer,
                target.vocab,
            )?;
        }
        DrafterKind::Mtp => {
            MtpConfig::from_gguf(&gguf.content)?.check_against_target(target)?;
        }
    }
    Ok(kind)
}

fn load_tokenizer() -> Result<LagunaTokenizer> {
    LagunaTokenizer::embedded().context("loading the tokenizer embedded in the binary")
}

/// The context length to allocate the KV cache for: what the config asks for, capped at
/// what the checkpoint was converted with. Going past the trained context needs a rope
/// override at load, which this server does not do, so a larger request is trimmed rather
/// than honored into gibberish.
fn resolve_context_length(requested: usize, trained: usize) -> Result<(usize, Option<ServeLog>)> {
    ensure!(requested > 0, "context_length must be at least 1");
    if requested <= trained {
        return Ok((requested, None));
    }
    Ok((
        trained,
        Some(ServeLog::ContextClamped { requested, trained }),
    ))
}

/// The loaded model plus the bookkeeping that lets several conversations reuse the KV
/// cache they each built: one of them occupies the model's cache, the rest are host-RAM
/// images the engine pages back in when their conversation speaks again.
struct EngineState {
    generator: Generator,
    /// The device the model was loaded on, which is where a host snapshot has to be
    /// uploaded to before the cache can be rewound to it.
    device: Device,
    slots: Slots,
    /// Which checkpoint this state holds, and which file that is. The pickup compares
    /// it against what the next job names, and a mismatch is what swaps the state out —
    /// so it has to distinguish a served custom file from the official checkpoint it
    /// runs as, which are different weights under the same architecture.
    size: Target,
    /// The KV cache holds writes from a job that has not finished reconciling them, so
    /// nothing in it may be reused. Set at the first cache mutation and cleared once the
    /// cache and the token history agree again; a job that fails before mutating anything
    /// — a prompt that does not fit, a request nobody is listening to any more — never
    /// sets it, and so costs the next request nothing.
    dirty: bool,
}

/// The GGUF and (optional) drafter for one target.
///
/// The served file answers only for the target that IS the served file
/// (`default_target`). A custom GGUF that identified as nothing serves under its
/// own id and nothing else: a request naming an official checkpoint on such a
/// server resolves that checkpoint's real hub file, because an official name
/// must never be answered by weights nobody checked. Everything not served
/// locally resolves lazily against the Hugging Face cache, downloading on a
/// miss — the same behavior `xwen batch` has from the CLI.
///
/// The drafter follows the mode: `Off` never drafts, `Custom` belongs to the
/// default checkpoint alone (sidecars never transfer, so any other checkpoint
/// falls back to its own official one), and `Official` is per checkpoint — which
/// is the point of it being a mode rather than a path. A checkpoint that ships
/// no sidecar decodes plain with a line saying so, whatever the others do.
fn checkpoint_paths(
    settings: &ServeSettings,
    target: Target,
    default_target: Target,
    logger: &ServeLogger,
) -> Result<(std::path::PathBuf, Option<std::path::PathBuf>)> {
    let size = target.model;
    // The served file answers for this target only when it IS this target — which
    // for a custom GGUF means its own id and nothing else.
    let local = target == default_target;
    let model = if local {
        settings.model.clone()
    } else {
        // Backstop, not the gate: every API surface already refused a
        // checkpoint this server will not run (`super::checkpoint_selectable`).
        // What is left is a future caller that reaches here without passing it —
        // which must not become a 111 GB download nobody asked for, or a load
        // that then fails every request.
        ensure!(
            super::checkpoint_selectable(size),
            "{}",
            super::unselectable_model_message(size)
        );
        if hub::cached_model(size).is_none() {
            logger.log(ServeLog::CheckpointDownloading {
                repo: size.repo(),
                file: size.file(),
                size: size.size(),
            });
        }
        hub::ensure_model(size).with_context(|| {
            format!("fetching the {size} checkpoint from the Hugging Face cache")
        })?
    };
    let draft = match &settings.draft {
        DraftMode::Off => None,
        DraftMode::Custom(path) if local => Some(path.clone()),
        // Nothing in the config asked, and this checkpoint ships a sidecar it
        // does not attach unasked (the 35B-A3B since 2026-09-06). Said out
        // loud, because a drafter that is present and unused otherwise reads as
        // a fetch that failed. A checkpoint with no sidecar at all falls
        // through instead and keeps saying the truer thing about itself.
        DraftMode::Default if size.drafter_kind().is_some() && !size.draft_default_on() => {
            logger.log(ServeLog::DraftDefaultOff { model: size });
            None
        }
        DraftMode::Custom(_) | DraftMode::Official | DraftMode::Default => {
            official_drafter(size, logger)?
        }
    };
    Ok((model, draft))
}

/// One checkpoint's official sidecar, fetched on a miss — or `None`, with a line
/// saying so, for a checkpoint that ships none.
fn official_drafter(size: hub::Model, logger: &ServeLogger) -> Result<Option<std::path::PathBuf>> {
    let Some(file) = size.drafter_file() else {
        logger.log(ServeLog::NoDrafterAvailable { model: size });
        return Ok(None);
    };
    if hub::cached_drafter(size).is_none() {
        logger.log(ServeLog::CheckpointDownloading {
            repo: size.repo(),
            file,
            size: size.drafter_size().unwrap_or("unknown size"),
        });
    }
    hub::ensure_drafter(size).with_context(|| format!("fetching the {size} drafter sidecar"))
}

impl EngineState {
    fn load(
        settings: &ServeSettings,
        size: Target,
        default_target: Target,
        logger: &ServeLogger,
    ) -> Result<Self> {
        let (model_path, draft_path) = checkpoint_paths(settings, size, default_target, logger)?;
        let cfg = read_config(&model_path)?;
        // A backstop, not the contract: startup already refused a `--model-size`
        // that contradicts the served file, and every other path here resolved
        // its file FROM the checkpoint. What is left to catch is a file that
        // changed under a running server, so the message names both sides.
        ensure!(
            cfg.arch == size.model.arch(),
            "the file for the {size} checkpoint holds a {} model ({})",
            cfg.arch.key(),
            model_path.display()
        );
        if let Some(held) = cfg.checkpoint(&model_path) {
            ensure!(
                held == size.model,
                "the file for the {size} checkpoint holds {} ({})",
                held.full_name(),
                model_path.display()
            );
        }
        let (max_ctx, _) = resolve_context_length(settings.context_length, cfg.n_ctx_train)?;
        let device = gguf::metal_device()?;
        // Every job replaces this through `set_sampler`; the config defaults only cover
        // the window between load and the first draw (unpinned keys take the
        // standard thinking-mode set, since no request mode exists yet).
        let defaults = SamplerOptions::recommended_for(size.model, true);
        let sampling = SamplerOptions {
            temperature: settings.temperature.unwrap_or(defaults.temperature),
            top_k: settings.top_k.unwrap_or(defaults.top_k),
            top_p: settings.top_p.unwrap_or(defaults.top_p),
            presence_penalty: settings
                .presence_penalty
                .unwrap_or(defaults.presence_penalty),
            seed: defaults.seed,
        };
        let mut generator = Generator::load(
            &device,
            &model_path,
            Some(size.model),
            None,
            ExpertRunner::Fused,
            max_ctx,
            sampling,
        )?;
        if let Some(path) = &draft_path {
            attach_drafter(
                &mut generator,
                &device,
                path,
                settings,
                size,
                max_ctx,
                logger,
            )?;
        }
        Ok(Self {
            generator,
            device,
            slots: Slots::new(settings.cache_slots, settings.cache_snapshots),
            size,
            dirty: false,
        })
    }

    /// Why a stored image cannot be uploaded into the model this process loaded, or `None`
    /// when it can.
    ///
    /// Asked BEFORE the live conversation is paged out, so an image that does not fit
    /// costs a cache miss rather than a failed request — and asked of the LOADED model
    /// rather than of the checkpoint file the settings named, which are not the same
    /// question: the file can be replaced between the two reads, and the cache stack is
    /// what the upload actually has to fit.
    ///
    /// It delegates to the import paths' own rules (`Generator::check_importable`) rather
    /// than restating the geometry here. The checkpoint id binds where an image came
    /// from, not what shapes its records carry, so nothing but the imports themselves can
    /// say whether the bytes fit the stack — and the restore writes layer by layer, so
    /// learning it partway through is learning it too late.
    ///
    /// The drafter is deliberately not judged here. Its planes come from a separate GGUF
    /// named by a separate flag, so they are the one part of an image whose shape this
    /// server genuinely cannot predict — and losing them costs speculation, not the
    /// conversation, so `page_in` treats a refusal there as planes it simply does not
    /// have.
    fn rejects_image(&self, image: &DiskImage, restore: usize) -> Option<String> {
        let Some((_, rings)) = image.snapshots.iter().find(|(pos, _)| *pos == restore) else {
            return Some(format!(
                "the image has no snapshot at {restore} to resume from"
            ));
        };
        self.generator
            .check_importable(&image.full_kv, rings, restore)
            .err()
            .map(|e| format!("{e:#}"))
    }

    /// Forget the conversation the cache was serving, so the next job prefills its whole
    /// prompt. Used after a failed job, whose cache holds an unknown amount of a
    /// half-finished prefill. The paged-out slots were imaged before that job started and
    /// describe positions it never wrote, so they survive it.
    fn reset(&mut self, disk: Option<&DiskCache>) -> Result<()> {
        // A slot emptied outright no longer holds the conversation its stored image
        // covers, so the tier stops treating that file as one a warm conversation
        // would come back from — otherwise nothing could ever reclaim it.
        if let (Some(emptied), Some(disk)) = (self.slots.abandon_live(), disk) {
            disk.unlink(emptied);
        }
        self.dirty = false;
        // The drafter's cache mirrors the target's, so it is cleared with it: the
        // invariant every dispatch path relies on is that the drafter never holds
        // positions the target does not.
        self.generator.reset_drafter()?;
        self.generator.reset_cache()
    }
}

/// Load the drafter — whichever kind the file holds — and attach it for
/// speculative decoding. Same device as the target by requirement, not
/// convenience: both kinds borrow the target's embeddings and lm_head, so their
/// ops interleave with the target's.
///
/// The drafter's cache is sized by `draft_ctx`, capped at the target's context —
/// positions the target cannot hold are positions the drafter could never be
/// given.
fn attach_drafter(
    generator: &mut Generator,
    device: &Device,
    path: &Path,
    settings: &ServeSettings,
    size: Target,
    max_ctx: usize,
    logger: &ServeLogger,
) -> Result<()> {
    let started = Instant::now();
    let draft_ctx = settings.draft_ctx.min(max_ctx);
    // Classified through the same reader startup preflights with, so the two
    // cannot disagree about what a file is.
    let target = generator.model_config().clone();
    let load = || -> Result<crate::drafter::AttachedDrafter> {
        let kind = read_draft_config(path, &target)?;
        let gguf = gguf::open(path, device)?;
        Ok(match kind {
            DrafterKind::Dflash => crate::drafter::AttachedDrafter::Dflash(DflashDrafter::load(
                &gguf, device, draft_ctx,
            )?),
            DrafterKind::Mtp => crate::drafter::AttachedDrafter::Mtp(MtpDrafter::load(
                &gguf, &target, device, draft_ctx,
            )?),
        })
    };
    let drafter = load().with_context(|| format!("loading the drafter {}", path.display()))?;
    generator.attach_drafter(
        drafter,
        SpecParams {
            draft_max: resolved_draft_max(settings, size.model),
            draft_p_min: resolved_p_min(settings, size.model),
            pause_margin: settings.draft_pause_margin,
            // Not exposed by the server: a round that drafts anything at all is worth
            // verifying, and the CLI agrees — this is its default too.
            ..SpecParams::default()
        },
    )?;
    logger.log(ServeLog::DrafterLoaded {
        elapsed: started.elapsed(),
        draft_ctx,
    });
    Ok(())
}

/// The drafting floor for the checkpoint being loaded: the operator's pinned
/// value when one was set, else the checkpoint's own fitted default. Unset is
/// resolvable only here, at attach time — the merge cannot know which
/// checkpoint a future job will name.
///
/// The last fallback is `SpecParams::default().draft_p_min`, which is the
/// 35B-A3B's fitted 0.3 — an arbitrary value for any other checkpoint. Every
/// shipped checkpoint now carries a fitted floor of its own, so reaching it takes
/// a custom `draft.path` attached to a checkpoint that ships no sidecar, which is
/// currently no checkpoint at all. Nothing pretends otherwise; the floor item is
/// retired in docs/ledger-archive.md "Retired: Drafting".
/// Extracted from [`attach_drafter`] so the resolution is testable without a
/// Metal device.
/// The draft depth for the checkpoint being loaded: the operator's pinned value
/// when one was set, else that checkpoint's own fitted default. Resolvable only
/// here, at attach time, for the same reason `resolved_p_min` is — the merge
/// cannot know which checkpoint a future job will name.
///
/// It matters more than a shared default would suggest: the two drafter kinds
/// want opposite depths off the same knob (15 for a block that costs one forward
/// however wide, 3 for a chain that costs a forward per step), so a server that
/// applied one number to both would be drafting five times too deep on whichever
/// checkpoint it got wrong.
fn resolved_draft_max(settings: &ServeSettings, size: hub::Model) -> usize {
    settings
        .draft_max
        .or_else(|| size.draft_max_default())
        .unwrap_or_else(|| SpecParams::default().draft_max)
}

fn resolved_p_min(settings: &ServeSettings, size: hub::Model) -> f32 {
    settings
        .draft_p_min
        .or_else(|| size.draft_p_min_default())
        .unwrap_or_else(|| SpecParams::default().draft_p_min)
}

/// Closes the queue and clears the model-loaded flag when the engine thread
/// exits — however it exits. The per-job panic boundary catches everything a job
/// can throw, but a panic outside it (the dequeue's cost closure, the post-job
/// cache reset path) would otherwise leave the queue open: pushes keep
/// succeeding, handlers wait forever on events no thread will ever send, and
/// `/health` keeps reporting a model nobody serves. With the guard, handlers get
/// `EngineGone` — a clean 500 — instead.
struct EngineExitGuard {
    jobs: Arc<JobQueue>,
    resident: Arc<ResidentModel>,
}

impl Drop for EngineExitGuard {
    fn drop(&mut self) {
        self.resident.clear();
        self.jobs.close();
    }
}

fn engine_loop(
    settings: ServeSettings,
    default_target: Target,
    jobs: Arc<JobQueue>,
    resident: Arc<ResidentModel>,
    shutdown: Arc<Cancel>,
    disk_pending: PendingSlot,
    logger: ServeLogger,
) {
    let _exit = EngineExitGuard {
        jobs: Arc::clone(&jobs),
        resident: Arc::clone(&resident),
    };
    // Opened before the first job — the scan is what lets that job find a warm
    // store — and kept across idle unloads: the tier is bound to the checkpoint on
    // disk, not to a loaded model, and re-scanning after every unload would only
    // rediscover what it already knows. `None` means no disk tier at all, which
    // every call site reads as "there is nothing stored".
    //
    // The tier is bound to the SERVED FILE — `settings.model`, which is what it was
    // opened and verified against. While anything else is loaded every site below is
    // handed `None` instead ([`disk_for`]): feeding it a foreign model's images would
    // poison the store with bytes that claim the served binding, and verifying it
    // against foreign weights would disable it for the rest of the process. The
    // comparison is against the whole target rather than its checkpoint, because on a
    // custom-GGUF server the official checkpoint of the same architecture is a
    // different file with the same name for sizing.
    let disk = DiskCache::open(&settings, &logger);
    // Published for the shutdown watchdog, which sizes its own grace off the same
    // bytes this thread sizes its flush budget off — see `serve::shutdown_grace`.
    // Left empty when the tier is off, which reads as nothing owed.
    if let Some(disk) = disk.as_ref() {
        let _ = disk_pending.set(disk.pending_handle());
    }
    let disk_for = |target: Target| -> Option<&DiskCache> {
        if target == default_target {
            disk.as_ref()
        } else {
            None
        }
    };
    let mut state: Option<EngineState> = None;
    loop {
        // The idle timer only matters while something is loaded; with the model already
        // dropped there is nothing to wake up for but a job. The window is measured from
        // here — the moment the previous job returned — and `take` gets a fresh timeout
        // every time round, so each job restarts the full countdown. Requests that never
        // reach this thread (`/health`, `/v1/models`, `count_tokens`) do not.
        let idle = settings.idle_unload.filter(|_| state.is_some());
        let waiting_since = Instant::now();
        // The scheduler scores a queued prompt by the prefill it would actually need,
        // so the cost closure asks the slots what the KV cache already holds for it —
        // a `&self` read, no paging. With no model loaded nothing is cached and the
        // policy degrades to shortest-prompt-first. A job for a checkpoint other
        // than the resident one scores no discount either: the warm slots belong
        // to the resident checkpoint, and both models share a tokenizer, so a
        // token-level match against the wrong model's cache would otherwise let a
        // long re-sent conversation jump the queue and then pay a swap plus a
        // full cold prefill.
        let hot = |job: &Job| {
            state.as_ref().map_or(0, |held| {
                if held.size == job.model() {
                    held.slots.choose(job.prompt()).pos()
                } else {
                    0
                }
            })
        };
        let Some(Queued {
            job,
            submitted,
            prompt_tokens,
        }) = jobs.take(idle, &hot)
        else {
            if jobs.is_closed() {
                break;
            }
            // Only a timeout — which only an armed idle timer produces — gets here.
            // The conversation in the cache is on its way out with the model, so it
            // is imaged and stored first: an idle unload is the likeliest moment for
            // a client to come back to a conversation it was in the middle of.
            let held_disk = state.as_ref().map(|held| held.size).and_then(&disk_for);
            store_live_conversation(state.as_mut(), held_disk, &logger);
            state = None;
            resident.clear();
            // The warm conversations went with the model.
            logger.log(ServeLog::SlotsSnapshot(Vec::new()));
            // The measured span is reported alongside the configured one so that a
            // report of "it unloaded early" can be settled from the log.
            logger.log(ServeLog::IdleUnloaded {
                elapsed: waiting_since.elapsed(),
                configured: idle,
            });
            continue;
        };

        // The client may have hung up — or the server begun shutting down —
        // while this job sat in the queue.
        if shutdown.is_cancelled() || job.cancel().is_cancelled() || job.events().is_closed() {
            continue;
        }

        // The checkpoint this job needs. A resident state holding the other one is
        // imaged out first — through the same path an idle unload takes, so the
        // live conversation survives the swap in the disk tier when it is on —
        // and the lazy load below brings in the right one.
        let required = job.model();
        if let Some(held) = state
            .as_ref()
            .map(|held| held.size)
            .filter(|held| *held != required)
        {
            logger.log(ServeLog::CheckpointSwappingOut {
                from: held,
                to: required,
            });
            store_live_conversation(state.as_mut(), disk_for(held), &logger);
            state = None;
            resident.clear();
            // The warm conversations went with the model.
            logger.log(ServeLog::SlotsSnapshot(Vec::new()));
        }

        // The wall-clock ceiling is stamped at pickup, before the lazy load
        // below, so the configured slack covers the model load as the config
        // documents. The reply budget here is the requested `max_tokens` — the
        // context-capped value needs the loaded model, and a watchdog ceiling
        // only ever errs loose.
        let mut job = job;
        if job.deadline().is_none() {
            job.set_deadline(job_deadline(
                Instant::now(),
                prompt_tokens,
                job.max_tokens(),
                &settings,
            ));
        }

        // From here the job is this thread's, and it reports one `JobDone` however it
        // ends. A job dropped at the check above was never picked and reports neither.
        let mut trace = JobTrace::new(
            job.origin(),
            crate::serve::model_id(&settings, &required),
            prompt_tokens,
            matches!(job, Job::Batch(_)),
            Instant::now(),
        );
        logger.log(ServeLog::JobPicked {
            origin: job.origin(),
            prompt_tokens,
            // A batch's text is untokenized at pickup, so its figure is the
            // queue's loose bytes-based estimate; a generation's prompt was
            // rendered and encoded by the HTTP layer and the count is real.
            estimated: matches!(job, Job::Batch(_)),
            queue_wait: trace.picked.saturating_duration_since(submitted),
            deadline: job.deadline(),
        });

        let events = job.events().clone();
        // What this pickup's lazy load cost, if it ran: the batch response's
        // stats block reports it, exactly as the CLI reports its own load.
        let load_elapsed: Cell<Option<Duration>> = Cell::new(None);
        let mut drop_model = false;
        // The lazy load runs behind the same panic boundary as the job itself. A panic
        // in the model layer mid-job leaves the KV cache in a state nothing can reason
        // about, so the model goes with it; a panic during the load installs nothing,
        // so there is nothing to drop and a later request retries the load from
        // scratch. Every other failure is reported and the thread keeps serving.
        let outcome = run_behind_boundary(
            &mut state,
            || {
                let start = Instant::now();
                let loaded = EngineState::load(&settings, required, default_target, &logger)
                    .map_err(|e| JobFailure {
                        error: anyhow!("loading the model failed: {e:#}"),
                        request_fault: false,
                    })?;
                logger.log(ServeLog::ModelLoaded {
                    elapsed: start.elapsed(),
                });
                load_elapsed.set(Some(start.elapsed()));
                // The store was bound to the checkpoint on disk before the weights
                // were read. Nothing stopped that file from being replaced in
                // between, and an image bound to weights nobody is serving describes
                // another model's keys — so the tier is checked against what actually
                // loaded, here and on every reload after an idle unload. Only the
                // default checkpoint is ever checked: the tier belongs to it, and a
                // non-default load would merely disable the store it never uses.
                if let Some(disk) = disk_for(required) {
                    disk.verify(loaded.generator.checkpoint_id());
                }
                Ok(loaded)
            },
            |engine| {
                // Flipped here rather than inside the load: the run closure only
                // ever executes with the state installed, so a panic between
                // building the state and installing it can never leave `/health`
                // claiming a model nobody holds.
                resident.store(required);
                match job {
                    Job::Generation(job) => run_job(
                        engine,
                        *job,
                        disk_for(required),
                        &shutdown,
                        &logger,
                        &mut trace,
                    ),
                    Job::Batch(job) => run_batch_job(
                        engine,
                        *job,
                        load_elapsed.get().map_or(0.0, |d| d.as_secs_f64() * 1000.0),
                        disk_for(required),
                        &shutdown,
                        &logger,
                        &mut trace,
                    ),
                }
            },
        );
        match outcome {
            JobOutcome::Completed => {}
            JobOutcome::Failed(failure) => {
                trace.record.error = Some(format!("{:#}", failure.error));
                send_unless_shutdown(&events, failure.into_event(), &shutdown, &logger);
            }
            JobOutcome::Panicked { model_lost } => {
                let message = if model_lost {
                    "the inference engine panicked; the model was dropped and reloads on the \
                     next request"
                } else {
                    "loading the model panicked; the load is retried on the next request"
                };
                trace.record.error = Some(message.to_string());
                send_unless_shutdown(
                    &events,
                    EngineEvent::Error {
                        message: message.into(),
                        request_fault: false,
                    },
                    &shutdown,
                    &logger,
                );
                drop_model = model_lost;
            }
        }
        // Whatever happened, a job that did not reconcile its own writes left the KV
        // cache holding part of a prompt nothing has a token history for. That is the
        // one condition the reuse machinery cannot recover from, so it is also the
        // only one that costs the cache.
        if let Some(engine) = state.as_mut()
            && !drop_model
            && engine.dirty
        {
            // Behind a boundary of its own: the reset is a model call like any
            // other, and a panic here sits outside the per-job boundary above —
            // it would take the engine thread down without the `JobDone` that
            // every `JobPicked` is owed. A panicked reset costs the model
            // exactly as a failed one does.
            let cleared = std::panic::catch_unwind(AssertUnwindSafe(|| {
                engine.reset(disk_for(engine.size))?;
                log_slots(&engine.slots, &logger);
                anyhow::Ok(())
            }));
            match cleared {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    logger.log(ServeLog::CacheClearFailed {
                        error: format!("{e:#}"),
                    });
                    drop_model = true;
                }
                Err(_) => {
                    logger.log(ServeLog::CacheClearFailed {
                        error: "it panicked".to_string(),
                    });
                    drop_model = true;
                }
            }
        }
        if drop_model {
            state = None;
            resident.clear();
            // The warm conversations went with the model.
            logger.log(ServeLog::SlotsSnapshot(Vec::new()));
        }
        // Read out here rather than in `run_job`, so that a reply cut short — or one
        // whose job then failed or panicked — still reports the wait its client
        // actually had.
        trace.record.ttft_secs = trace
            .first_sent
            .get()
            .map(|at| at.saturating_duration_since(trace.picked).as_secs_f64());
        // The history gets the same numbers the record does. A warning about a
        // history that cannot be written goes through the logger like every
        // other line, because on a `--tui` server stderr belongs to the
        // dashboard.
        let run = run_record(&trace.record, trace.batch_job);
        if let Some(warning) = crate::metrics::record_warning(&run) {
            logger.log(ServeLog::HostLine(warning));
        }
        logger.log(ServeLog::JobDone(Box::new(trace.record)));
    }
    // The conversation still in the cache is worth as much as any other, so it is
    // imaged and queued like one; then the writer gets a bounded window to land
    // whatever it still holds. Losing an image here costs the next server a
    // re-prefill, which is why the wait is bounded and never retried.
    let held_disk = state.as_ref().map(|held| held.size).and_then(&disk_for);
    store_live_conversation(state.as_mut(), held_disk, &logger);
    if let Some(disk) = disk.as_ref() {
        disk.flush(disk_flush_budget(disk.pending_bytes()));
    }
}

/// Image the conversation in the KV cache into its slot and queue it for the disk
/// tier, on the way to dropping the model: an idle unload or a shutdown.
///
/// Both callers sit outside the per-job panic boundary, so this has one of its own —
/// a panic in the export would otherwise cost the engine thread, and with it the
/// exit path that unregisters the model's mmap, for the sake of one stored image.
fn store_live_conversation(
    state: Option<&mut EngineState>,
    disk: Option<&DiskCache>,
    logger: &ServeLogger,
) {
    let (Some(engine), Some(disk)) = (state, disk) else {
        return;
    };
    let stored = std::panic::catch_unwind(AssertUnwindSafe(|| {
        page_out_live(engine, Some(disk), logger)
    }));
    let error = match stored {
        Ok(Ok(())) => return,
        Ok(Err(e)) => format!("{e:#}"),
        Err(_) => "it panicked".to_string(),
    };
    logger.log(ServeLog::DiskCacheFailed {
        action: "imaging the live conversation",
        error,
    });
}

/// One job's measurements while it runs, and the record they become.
///
/// The engine loop owns it rather than [`run_job`] so that exactly one
/// [`ServeLog::JobDone`] follows every [`ServeLog::JobPicked`]: a job that fails, that
/// panics, or that a departed client leaves through one of the early returns fills in
/// less of the record, but still reports one.
struct JobTrace {
    /// When the engine picked the job up. Every span in the record is measured from
    /// here, which is what makes the reported time-to-first-token the wait a client
    /// actually experiences — the lazy model load and any paging included.
    picked: Instant,
    /// When the first content event of the reply reached the client's channel. A `Cell`
    /// because the emitter that stamps it holds the trace only to say this one thing,
    /// while the record's fields are being written around it; reading it out here, once,
    /// is what makes the reported figure survive every way a job can end.
    first_sent: Cell<Option<Instant>>,
    /// Whether this is a batch job, known at pickup from the job's own variant.
    /// The record's `batch` summary says the same thing only for a batch that
    /// RAN: one that failed before the runner returned has no summary and would
    /// otherwise be reported as the native generation it is not.
    batch_job: bool,
    record: JobRecord,
}

impl JobTrace {
    fn new(
        origin: RequestOrigin,
        model: String,
        prompt_tokens: usize,
        batch_job: bool,
        picked: Instant,
    ) -> Self {
        Self {
            picked,
            first_sent: Cell::new(None),
            batch_job,
            record: JobRecord {
                origin,
                model,
                stop: None,
                abandoned: None,
                error: None,
                prompt_tokens,
                cache_read: 0,
                prefill_tokens: 0,
                prefill_secs: 0.0,
                output_tokens: 0,
                thinking_tokens: 0,
                decode_secs: 0.0,
                ttft_secs: None,
                spec: None,
                batch: None,
            },
        }
    }
}

/// One finished job as the metrics history records it.
///
/// `batch_job` comes from the job's own variant rather than from the presence of
/// a summary: a batch that failed before its runner returned has no summary, and
/// reporting it as the native generation it was submitted on would put it in the
/// wrong surface with the wrong numbers.
///
/// A batch's token counts are the record's own, already summed from the items at
/// the fold — never derived from each other, because a scored batch forwards
/// more than its prompt and the difference would go negative. Only its phase
/// SECONDS come from the summary, the generation-centric spans being zero for a
/// batch. A batch that never got that far carries the queue's bytes-based prompt
/// estimate, which is not a measurement and is recorded as nothing rather than
/// as a number.
fn run_record(record: &JobRecord, batch_job: bool) -> crate::metrics::RunRecord {
    // A batch is submitted on the native dialect but is not a native
    // generation, and the two cost nothing alike: they get their own surfaces.
    let surface = if batch_job {
        "serve:batch".to_string()
    } else {
        format!("serve:{}", record.origin.dialect.label())
    };
    let mut run = crate::metrics::RunRecord::new(surface, record.model.clone());
    run.client = record.origin.client.clone();
    run.session = record.origin.session.clone();
    run.agent = record.origin.agent.clone();
    run.thinking_tokens = Some(record.thinking_tokens);
    run.drafted = record.spec.map(|spec| spec.drafted);
    run.accepted = record.spec.map(|spec| spec.accepted);
    run.items = record.batch.map(|batch| batch.items);
    // Every item failing is a failed run, however cleanly the machinery around
    // them worked. A batch that lost some of its items still did the rest.
    let every_item_failed = record
        .batch
        .is_some_and(|batch| batch.items > 0 && batch.failed == batch.items);
    // A job the client walked away from, or one the deadline or a shutdown cut,
    // did not reach its own end however healthy the engine was: its counts stop
    // wherever it was interrupted, and reporting it alongside completed runs
    // would quietly drag their averages down.
    run.ok = record.error.is_none() && record.abandoned.is_none() && !every_item_failed;
    match record.batch {
        Some(batch) => {
            run.prompt_tokens = record.prompt_tokens;
            // The record's own `prefill_tokens` is `prompt - cache_read`, which
            // the fold keeps so the record's arithmetic closes. That is not the
            // forwarded work on a scored batch, where teacher-forced trials run
            // through the model against no prompt token, so the metrics record
            // takes the measured figure and leaves the two counts independent.
            run.cached_tokens = record.cache_read;
            run.prefill_tokens = batch.prefill_tokens;
            run.prefill_secs = batch.prefill_secs;
            run.decode_tokens = batch.decode_tokens;
            run.decode_secs = batch.decode_secs;
        }
        // A batch with no summary never reached the runner, so every token
        // count on it is the estimate it was queued under.
        None if batch_job => {}
        None => {
            run.prompt_tokens = record.prompt_tokens;
            run.cached_tokens = record.cache_read;
            run.prefill_tokens = record.prefill_tokens;
            run.prefill_secs = record.prefill_secs;
            run.decode_tokens = record.output_tokens;
            run.decode_secs = record.decode_secs;
        }
    }
    run
}

/// What one trip through the panic boundary produced.
enum JobOutcome {
    /// The job ran; its own event stream carried whatever it had to say.
    Completed,
    /// The load or the job failed, and the client is owed the contained error.
    Failed(JobFailure),
    /// A panic was caught. `model_lost` is true when a loaded model was live when it
    /// fired — its caches can no longer be reasoned about, so the caller must drop the
    /// state. False means the panic happened during the load: nothing was installed,
    /// the state is still empty, and a later job retries the load from scratch.
    Panicked { model_lost: bool },
}

/// Lazily load the engine state and run one job, both behind one panic boundary, so
/// neither a load panic nor a job panic can take the engine thread down.
///
/// The load is atomic with respect to panics: `load` builds the whole state and only a
/// fully built value is installed, so a mid-load panic leaves `state` exactly as it was
/// — empty — and no half-built state can ever be observed. Generic over the state so
/// the load-panic recovery shape is testable with a stub.
fn run_behind_boundary<S>(
    state: &mut Option<S>,
    load: impl FnOnce() -> Result<S, JobFailure>,
    run: impl FnOnce(&mut S) -> Result<(), JobFailure>,
) -> JobOutcome {
    // AssertUnwindSafe: a load panic installed nothing, and after a run panic the
    // caller discards the state — the `model_lost` contract — so no torn value is
    // ever used.
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if state.is_none() {
            *state = Some(load()?);
        }
        run(state.as_mut().expect("just installed"))
    }));
    match outcome {
        Ok(Ok(())) => JobOutcome::Completed,
        Ok(Err(failure)) => JobOutcome::Failed(failure),
        Err(_) => JobOutcome::Panicked {
            model_lost: state.is_some(),
        },
    }
}

/// How a job failed, and whose fault it was.
///
/// Request faults are the ones the client could fix by sending something else; everything
/// else is the server failing to serve a request it should have been able to. The
/// distinction is made here, where the cause is known, rather than reconstructed from the
/// message text downstream.
struct JobFailure {
    error: anyhow::Error,
    request_fault: bool,
}

impl JobFailure {
    fn request(error: anyhow::Error) -> Self {
        Self {
            error,
            request_fault: true,
        }
    }

    fn into_event(self) -> EngineEvent {
        EngineEvent::Error {
            message: format!("{:#}", self.error),
            request_fault: self.request_fault,
        }
    }
}

/// Anything propagated with `?` is a server-side failure; a request fault is stated.
impl From<anyhow::Error> for JobFailure {
    fn from(error: anyhow::Error) -> Self {
        Self {
            error,
            request_fault: false,
        }
    }
}

/// Deliver one event, waiting out ordinary backpressure but not a client that has
/// stopped reading altogether, and giving up the moment the server begins shutting
/// down: no event is worth pinning the engine thread through the shutdown grace
/// period. False means the event did not land — hangup, [`SEND_DEADLINE`], or the
/// shutdown — and the generation behind it has nobody left to serve.
fn send_unless_shutdown(
    events: &Sender<EngineEvent>,
    event: EngineEvent,
    shutdown: &Cancel,
    logger: &ServeLogger,
) -> bool {
    send_until_interrupted(
        events,
        event,
        Instant::now() + SEND_DEADLINE,
        &|| shutdown.is_cancelled(),
        logger,
    )
}

/// The retry loop under every send: one delivery attempt per [`SEND_RETRY_INTERVAL`],
/// until the event lands, the channel closes, the deadline passes, or `interrupted`
/// says to stop waiting. The interrupt is polled after a failed attempt, so even an
/// interrupted send has offered the event to the channel once. The deadline is a
/// parameter so a test can watch one expire.
fn send_until_interrupted(
    events: &Sender<EngineEvent>,
    event: EngineEvent,
    deadline: Instant,
    interrupted: &dyn Fn() -> bool,
    logger: &ServeLogger,
) -> bool {
    let mut event = event;
    loop {
        match events.try_send(event) {
            Ok(()) => return true,
            Err(TrySendError::Closed(_)) => return false,
            Err(TrySendError::Full(returned)) => {
                if interrupted() {
                    return false;
                }
                if Instant::now() >= deadline {
                    logger.log(ServeLog::ClientStalled {
                        after: SEND_DEADLINE,
                    });
                    return false;
                }
                event = returned;
                std::thread::sleep(SEND_RETRY_INTERVAL);
            }
        }
    }
}

/// The signals that say nobody is waiting on the running job any more, folded into the
/// job's own cancel token so the finalization reads a single settled reason. Polled at
/// every point the engine can stop cheaply: between prefill chunks, per decoded token,
/// before a paging transfer, and inside a stalled send.
struct Abandon<'a> {
    /// The process-wide shutdown token.
    shutdown: &'a Cancel,
    /// The job's own token: the HTTP side's drop guard sets `ClientGone` into it, and
    /// [`Abandon::reason`] stamps `Shutdown`, `Deadline` and the closed-channel
    /// `ClientGone` as those fire — first writer wins, so the reason never changes
    /// once acted on.
    cancel: &'a Cancel,
    events: &'a Sender<EngineEvent>,
    /// Where the wind-down reports what became of the job, and where a stalled
    /// send says so.
    logger: &'a ServeLogger,
    /// When the job started, which against `deadline` names the ceiling in the log.
    started: Instant,
    /// The job's wall-clock ceiling, or `None` for a job running with the
    /// watchdog disabled (a rate of 0).
    deadline: Option<Instant>,
    /// When [`Abandon::reason`] first observed a cancellation. From that instant
    /// on, every remaining send shares one [`SEND_DEADLINE`] budget (see
    /// [`Abandon::send`]): the job is already over, and its wind-down events may
    /// not each park the inference thread for a full delivery deadline.
    cancelled_at: Cell<Option<Instant>>,
    /// Latched by the first send that failed to land — timeout, closed channel,
    /// or shutdown interrupt. Every later send is suppressed: a stream that
    /// already dropped an event would only become internally inconsistent by
    /// delivering the ones after it.
    dead: Cell<bool>,
}

impl<'a> Abandon<'a> {
    fn new(
        shutdown: &'a Cancel,
        cancel: &'a Cancel,
        events: &'a Sender<EngineEvent>,
        logger: &'a ServeLogger,
        started: Instant,
        deadline: Option<Instant>,
    ) -> Self {
        Self {
            shutdown,
            cancel,
            events,
            logger,
            started,
            deadline,
            cancelled_at: Cell::new(None),
            dead: Cell::new(false),
        }
    }

    /// Every abandonment signal folded into one reason, or `None` while somebody is
    /// still waiting. Shutdown is folded before the deadline so a job that hits both
    /// in the same poll is reported as the shutdown it is; a closed event channel is
    /// stamped last, as `ClientGone`, so every consumer of the token — this fold, the
    /// queue sweep, the HTTP side — converges on one settled reason.
    fn reason(&self) -> Option<CancelReason> {
        if self.shutdown.is_cancelled() {
            self.cancel.cancel(CancelReason::Shutdown);
        }
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            self.cancel.cancel(CancelReason::Deadline);
        }
        if self.events.is_closed() {
            self.cancel.cancel(CancelReason::ClientGone);
        }
        let reason = self.cancel.reason();
        if reason.is_some() && self.cancelled_at.get().is_none() {
            self.cancelled_at.set(Some(Instant::now()));
        }
        reason
    }

    /// The wall-clock budget the job was given, for the deadline log line —
    /// which only ever names it after the deadline fired, so it exists there.
    ///
    /// Saturating because the ceiling is stamped at pickup and the job starts
    /// after the lazy model load: a load that outlasts the whole budget leaves
    /// the deadline behind the start, which is a ceiling of zero and not a
    /// reason to panic the engine thread.
    fn ceiling(&self) -> Duration {
        self.deadline.map_or(Duration::ZERO, |deadline| {
            deadline.saturating_duration_since(self.started)
        })
    }

    /// Whether a blocked send should stop waiting. Shutdown is the one reason that
    /// interrupts a send in progress: a gone client closes the channel on its own,
    /// and a client whose job hit its deadline is still reading and is owed its
    /// terminal events.
    fn shutdown_pending(&self) -> bool {
        self.shutdown.is_cancelled() || self.cancel.reason() == Some(CancelReason::Shutdown)
    }

    /// Deliver one event, waiting out ordinary backpressure — except at shutdown,
    /// where a single failed attempt is all a stalled reader gets. False means the
    /// event did not land and nobody is left to serve.
    ///
    /// While the job is live each event gets a fresh [`SEND_DEADLINE`]; once a
    /// cancellation has been observed, every remaining send shares one budget from
    /// that instant, so the wind-down — healed call, held-back tail, terminal
    /// event — cannot outstay the ceiling by a full deadline per event. And the
    /// first send that fails to land is final: every later one is suppressed,
    /// because a stream that already dropped an event has nothing consistent
    /// left to say.
    fn send(&self, event: EngineEvent) -> bool {
        if self.dead.get() {
            return false;
        }
        let deadline = send_deadline(self.cancelled_at.get(), Instant::now());
        let landed = send_until_interrupted(
            self.events,
            event,
            deadline,
            &|| self.shutdown_pending(),
            self.logger,
        );
        if !landed {
            self.dead.set(true);
        }
        landed
    }
}

/// The absolute delivery deadline for one send: a fresh [`SEND_DEADLINE`] while
/// the job is live, or the tail of the single budget that started when its
/// cancellation was first observed. An exhausted budget still leaves each event
/// the one non-blocking delivery attempt [`send_until_interrupted`] always makes.
fn send_deadline(cancelled_at: Option<Instant>, now: Instant) -> Instant {
    match cancelled_at {
        Some(observed) => observed + SEND_DEADLINE,
        None => now + SEND_DEADLINE,
    }
}

/// Run one whole batch on the loaded engine and send its response document as the
/// job's single terminal event.
///
/// The batch runner owns the whole KV cache while it runs — its shared-prefix
/// snapshot machinery is its own, not the slot machinery's — so the live
/// conversation is imaged out first, through the same path an idle unload takes:
/// it survives in its slot's host image (and on disk when the tier is on), and the
/// post-job reset the `dirty` flag forces brings the cache back to a state the
/// next generation can reason about.
///
/// Cancellation (client gone, deadline, shutdown) is folded into the job's token
/// exactly as a generation's and polled by the runner between items and per
/// decoded token; items the cancellation reached report it in their own `error`
/// field, and the partial document is still sent to a client that is owed one.
fn run_batch_job(
    engine: &mut EngineState,
    job: BatchJob,
    load_ms: f64,
    disk: Option<&DiskCache>,
    shutdown: &Cancel,
    logger: &ServeLogger,
    trace: &mut JobTrace,
) -> Result<(), JobFailure> {
    page_out_live(engine, disk, logger).map_err(JobFailure::from)?;
    // Set before the first runner call: the shared prefill mutates the cache
    // immediately, and a failure anywhere after it must cost the cache.
    engine.dirty = true;

    let abandon = Abandon::new(
        shutdown,
        &job.cancel,
        &job.events,
        logger,
        trace.picked,
        job.deadline,
    );
    let mut cancelled = || abandon.reason().is_some();
    let mut progress = |report: BatchProgress| logger.log(ServeLog::BatchProgress(report));
    // The handler wrote the id this server answers under into the request; it is
    // what the document must be labeled with, and the runner no longer re-derives
    // it (a custom GGUF's id is no checkpoint's name).
    let label = job
        .request
        .model
        .clone()
        .unwrap_or_else(|| job.model.model.full_name().to_string());
    let response = crate::batch::run_batch(
        &mut engine.generator,
        &job.request,
        load_ms,
        &label,
        job.model.model,
        &mut BatchHooks {
            progress: &mut progress,
            cancelled: &mut cancelled,
        },
    )
    .map_err(JobFailure::from)?;

    // The trace reports measured totals where the estimate stood, in the
    // record's own arithmetic (`prompt = cache_read + prefill`): the summed
    // item prompts, of which every restore of the shared snapshot past its
    // first prefill was a cache read — the prefix is prefilled once and read
    // back N-1 times, so the sum of per-item `cached_prefix_tokens` overcounts
    // by exactly one span — and the rest ran through the model.
    trace.record.prompt_tokens = response.items.iter().map(|i| i.usage.prompt_tokens).sum();
    let cached_per_item: usize = response
        .items
        .iter()
        .map(|i| i.usage.cached_prefix_tokens)
        .sum();
    trace.record.cache_read = cached_per_item.saturating_sub(response.stats.shared_prefix_tokens);
    trace.record.prefill_tokens = trace
        .record
        .prompt_tokens
        .saturating_sub(trace.record.cache_read);
    trace.record.output_tokens = response
        .items
        .iter()
        .map(|i| i.usage.completion_tokens)
        .sum();
    trace.record.batch = Some(crate::serve::log::BatchSummary {
        items: response.items.len(),
        failed: response.items.iter().filter(|i| i.error.is_some()).count(),
        secs: response.stats.total_ms / 1000.0,
        // The runner's own prefill figures start counting AFTER the shared
        // prefix has been prefilled, whose tokens and time live in
        // `shared_prefix_tokens`/`snapshot_ms` instead. The summary is what
        // reports the whole prefill phase, so both halves are folded back in
        // together — dropping the tokens would understate the work and
        // dropping the seconds would overstate the rate.
        prefill_tokens: response.stats.prefill_tokens + response.stats.shared_prefix_tokens,
        prefill_secs: (response.stats.prefill_ms + response.stats.snapshot_ms) / 1000.0,
        decode_tokens: response.stats.decode_tokens,
        decode_secs: response.stats.decode_ms / 1000.0,
    });
    trace.record.abandoned = abandon.reason();

    // The one terminal event. A departed client's channel just drops it, and a
    // shutdown suppresses it — exactly as a generation's terminal events.
    send_unless_shutdown(
        &job.events,
        EngineEvent::BatchDone(Box::new(response)),
        shutdown,
        logger,
    );
    Ok(())
}

/// Prefill and decode one request, streaming events as they happen. The prompt arrives
/// already rendered and encoded by the HTTP layer. Errors reach the client as
/// `EngineEvent::Error`; the caller clears the cache behind a job that left `dirty` set.
///
/// Everything that can be judged from the request alone is judged before the first cache
/// write, so a request the server cannot serve never costs the next one its reusable
/// prefix.
fn run_job(
    engine: &mut EngineState,
    job: GenerationJob,
    disk: Option<&DiskCache>,
    shutdown: &Cancel,
    logger: &ServeLogger,
    trace: &mut JobTrace,
) -> Result<(), JobFailure> {
    let started = Instant::now();
    let GenerationJob {
        // The job's identity is already in the trace, which is where everything this
        // job reports about itself goes.
        origin: _,
        // The pickup already ensured the loaded checkpoint is this one.
        model: _,
        prompt,
        boundary,
        anchor,
        thinking_entry,
        max_think,
        max_tokens,
        sampling,
        stop_sequences,
        tools,
        grammar,
        cancel,
        deadline,
        events,
    } = job;
    let prompt_len = prompt.len();

    // The submit path already refused a prompt that does not fit, against the same
    // resolved context length. This check re-derives the bound from the checkpoint the
    // engine actually loaded, so a checkpoint swapped on disk between startup and the
    // lazy load still cannot overrun the KV cache.
    let max_ctx = engine.generator.max_ctx();
    if prompt_len >= max_ctx {
        return Err(JobFailure::request(anyhow!(
            "the prompt is {prompt_len} tokens, which leaves no room to reply inside the \
             server's {max_ctx}-token context: shorten the conversation or raise context_length"
        )));
    }
    // Whatever the request asked for, the reply cannot outgrow the cache.
    let max_new = max_tokens.min(max_ctx - prompt_len);

    // The handlers clamp a requested budget against the request's own `max_tokens`; this
    // repeats the check against what the context actually leaves, which is the number the
    // decode loop validates. A budget that still does not fit is dropped rather than
    // refused: the request is valid for the API it arrived on, so it has to produce an
    // answer, and reasoning without a ceiling is the honest way to give it one.
    let max_think = max_think.and_then(|requested| {
        let feasible = feasible_think_budget(requested, max_new);
        if feasible != Some(requested) {
            logger.log(ServeLog::ThinkBudgetClamped {
                requested,
                max_new,
                using: feasible,
            });
        }
        feasible
    });

    // Per-request sampling, thinking budget and grammar, all settled before the
    // cache is touched. The grammar is set unconditionally: `None` is what
    // clears the previous request's state.
    engine.generator.set_sampler(sampling);
    engine.generator.set_max_think(max_think.unwrap_or(0))?;
    engine.generator.set_grammar(grammar);

    // The ceiling arrives on the job, stamped at pickup — before the lazy model
    // load — so the configured slack covers the load, exactly as the config
    // documents. `None` means the watchdog is disabled (a rate of 0).
    let abandon = Abandon::new(shutdown, &cancel, &events, logger, started, deadline);

    let choice = engine.slots.choose(&prompt);
    // A stored image only competes when it would resume STRICTLY deeper than every
    // warm slot, and only when there is a slot to put it in that is not the live one
    // — with a single slot in use the conversation in the cache is never displaced
    // by a file. The read itself happens below, inside the dispatch.
    let hydrate = disk
        .zip(engine.slots.fresh_slot())
        .and_then(|(disk, target)| {
            disk.candidate(&prompt, choice.pos())
                .map(|candidate| (disk, candidate, target))
        });
    // Where this prompt stops matching anything the server holds, warm or stored —
    // asked of the matching rather than of whichever slot wins, and asked BEFORE the
    // dispatch, which can empty the very slot that shared the most: a conversation with
    // no snapshot deep enough to resume at loses to a fresh slot and is overwritten by
    // it. That prefix is still the fork the next session will arrive at.
    let shared_anywhere = engine
        .slots
        .deepest_shared(&prompt)
        .max(disk.map_or(0, |tier| tier.deepest_shared(&prompt)));
    // Whether the reply has been opened, so exactly one `Start` reaches the client
    // and no terminal event can precede it. The event carries how much of the prompt
    // came out of a cache, and that number is not known until the dispatch below has
    // settled — a stored image can be chosen and then turn out to be unusable — so it
    // is sent from one place, once the answer is a fact rather than a plan.
    let opened = Cell::new(false);

    // Past this point the KV cache is being written, and any exit that does not reconcile
    // the token history leaves it unusable.
    engine.dirty = true;
    // What the dispatch actually did, which is the hydrated slot's resume point when
    // a stored image won and the warm choice otherwise.
    let mut dispatched = choice;
    let mut hydrated = false;
    if let Some((tier, candidate, target)) = hydrate {
        match hydrate_slot(engine, tier, &candidate, &prompt, target, &abandon, logger)? {
            Hydration::Installed { slot, restore } => {
                dispatched = SlotChoice::Swap { slot, restore };
                hydrated = true;
            }
            // The file could not be used after all; the warm choice below stands.
            Hydration::Unavailable => {}
            Hydration::Abandoned(reason) => {
                finish_abandoned_before_decode(
                    &abandon,
                    &opened,
                    trace,
                    reason,
                    JobPhase::Dispatch,
                    0,
                    prompt_len,
                    0,
                );
                return Ok(());
            }
        }
    }
    // Every arm leaves the drafter's cache holding exactly what the target's does,
    // which is what lets the decode below speculate: Extend touches neither, Restore
    // rewinds both, a restart clears both, and a swap carries both in the slot image.
    if !hydrated {
        match choice {
            SlotChoice::Live { plan } => match plan {
                Resume::Extend { .. } => {}
                // A fork: this prompt shares a prefix of the live conversation and
                // diverges from it. Rewinding in place hands the slot to the arriving
                // prompt, which reconciles its own history into it at the end — so the
                // conversation that was live loses everything above the fork, in host RAM
                // and on disk both, having never been imaged. Imaging it first costs one
                // page-out (~270 ms for a 1.3 GiB slot) and is what lets the writer store
                // it and a later request come back to it; the arriving prompt is then
                // served by paging that same image back in at the fork, which is the
                // ordinary swap the machinery already performs.
                Resume::Restore { pos }
                    if engine.slots.live_history_at_risk(pos) >= SNAPSHOT_MIN_GAIN
                        && engine.slots.fresh_slot().is_some() =>
                {
                    let source = engine
                        .slots
                        .live
                        .ok_or_else(|| anyhow!("a rewind was planned with no live slot"))?;
                    let target = engine
                        .slots
                        .fresh_slot()
                        .ok_or_else(|| anyhow!("a fork was planned with nowhere to put it"))?;
                    // Taken from the ring as the CHOICE saw it, before the page-out below
                    // touches it. Its tail snapshot goes through `record`, which evicts the
                    // oldest entry of a full ring — and the fork point is the newest
                    // snapshot at or below the shared prefix, which for a client returning
                    // to an old branch of a long conversation is exactly that oldest entry.
                    // Reading the ring afterwards would hand the new slot a resume point
                    // the page-out had just thrown away.
                    let inherited = engine.slots.fork_rings(source, pos)?;
                    // The same two checks the swap arm makes, for the same reasons: this
                    // is now a page-out and a page-in, and a job nobody waits for should
                    // pay neither.
                    if let Some(reason) = abandon.reason() {
                        engine.dirty = false;
                        finish_abandoned_before_decode(
                            &abandon,
                            &opened,
                            trace,
                            reason,
                            JobPhase::Dispatch,
                            0,
                            prompt_len,
                            pos,
                        );
                        return Ok(());
                    }
                    page_out_live(engine, disk, logger)?;
                    if let Some(reason) = abandon.reason() {
                        finish_abandoned_before_decode(
                            &abandon,
                            &opened,
                            trace,
                            reason,
                            JobPhase::Dispatch,
                            0,
                            prompt_len,
                            pos,
                        );
                        return Ok(());
                    }
                    // The fork gets a slot of its own rather than the one it forked off, so
                    // both conversations stay warm and matchable. Nothing is copied: the
                    // images are `Arc`s, so the two slots share one set of rows and rings.
                    let slot = engine.slots.fork_from(source, target, pos, inherited)?;
                    // The slot may have been another conversation's a moment ago, and that
                    // conversation's stored chain must stop counting as one a warm slot
                    // would come back from — the same reason the fresh arm below unlinks.
                    if let Some(disk) = disk {
                        disk.unlink(slot);
                    }
                    page_in(engine, slot, pos, logger)?;
                    dispatched = SlotChoice::Swap { slot, restore: pos };
                }
                Resume::Restore { pos } => restore_live(engine, pos)?,
                Resume::Cold => restart_live(engine, disk)?,
            },
            SlotChoice::Swap { slot, restore } => {
                // The two transfers below move hundreds of MiB each, so a job already
                // abandoned is checked for before paying either. Here nothing has been
                // written: the cache still agrees with the live slot's history, so the
                // exit clears `dirty` and costs nobody anything.
                if let Some(reason) = abandon.reason() {
                    engine.dirty = false;
                    finish_abandoned_before_decode(
                        &abandon,
                        &opened,
                        trace,
                        reason,
                        JobPhase::Dispatch,
                        0,
                        prompt_len,
                        restore,
                    );
                    return Ok(());
                }
                page_out_live(engine, disk, logger)?;
                // Between the transfers the live conversation is safely imaged out and
                // nothing owns the cache, so this exit leaves `dirty` for the caller to
                // clear; every slot image survives that reset.
                if let Some(reason) = abandon.reason() {
                    finish_abandoned_before_decode(
                        &abandon,
                        &opened,
                        trace,
                        reason,
                        JobPhase::Dispatch,
                        0,
                        prompt_len,
                        restore,
                    );
                    return Ok(());
                }
                page_in(engine, slot, restore, logger)?;
            }
            SlotChoice::Fresh { slot } => {
                // Same as the swap arm's first check: the page-out is the expensive part,
                // and nothing has been written yet.
                if let Some(reason) = abandon.reason() {
                    engine.dirty = false;
                    finish_abandoned_before_decode(
                        &abandon,
                        &opened,
                        trace,
                        reason,
                        JobPhase::Dispatch,
                        0,
                        prompt_len,
                        0,
                    );
                    return Ok(());
                }
                page_out_live(engine, disk, logger)?;
                engine.generator.reset_cache()?;
                engine.generator.reset_drafter()?;
                engine.slots.start_fresh(slot);
                // The slot this conversation lands in may have been holding another
                // one, linked to its own stored image. That link is not this
                // conversation's, and left behind it would keep a file safe from
                // eviction on behalf of a conversation no slot holds any more.
                if let (Some(live), Some(disk)) = (engine.slots.live, disk) {
                    disk.unlink(live);
                }
            }
        }
    }
    engine.slots.touch_live()?;
    // One report for every arm above: whichever of them ran, the slots now stand as
    // this job will use them. The two paging transfers report their own state as well,
    // since each is hundreds of milliseconds nobody would otherwise see.
    log_slots(&engine.slots, logger);

    // The KV cache appends at its own length, so the position prefill resumes at has to be
    // exactly what the cache holds. A mismatch means the token history and the model have
    // drifted apart, and replaying the whole prompt is the only repair. Only the live
    // conversation is implicated — the other slots' images were written before this job.
    let cached = engine.generator.cache_len();
    let resume = if cached == dispatched.pos() {
        dispatched.pos()
    } else {
        logger.log(ServeLog::CacheLengthMismatch {
            cached,
            expected: dispatched.pos(),
        });
        restart_live(engine, disk)?;
        // The slots just lost the prefix they were reported with, which is the
        // worst moment for a consumer to be showing the old numbers.
        log_slots(&engine.slots, logger);
        0
    };
    trace.record.cache_read = resume;
    // Now the number is settled — the dispatch ran, a stored image either served or
    // did not, and the cache has been checked against the history — so this is what
    // the client is told it got for free. A send that does not land means the client
    // hung up: there is nobody left to serve, and nobody to report that to either.
    if !open_reply(&abandon, &opened, prompt_len, resume) {
        // A first send that does not land latches the channel dead, so this job will
        // never say another word. The record says that rather than describing a
        // request that simply ended.
        trace.record.abandoned = Some(CancelReason::ClientGone);
        // What the dispatch established is a valid prefix of this very prompt, so it
        // is reconciled and kept rather than reset: an identical retry — which is
        // what a client that hung up usually sends next — resumes from it.
        reconcile_partial_prefill(engine, &prompt)?;
        log_slots(&engine.slots, logger);
        return Ok(());
    }
    // The dispatch above always leaves a slot live — `touch_live` would have failed the
    // job otherwise — so the `if let` costs nothing; it is there because a report about
    // the cache must never be the thing that panics the engine and drops the model.
    if let Some(slot) = engine.slots.live {
        logger.log(ServeLog::JobCacheResolved {
            id: trace.record.origin.id,
            cached: dispatched.pos(),
            resume,
            slot,
        });
    }

    // The prompt is prefilled in one span per position worth snapshotting, plus the tail:
    // the turn boundary the next turn resumes at, the anchor where the system block ends,
    // and the point where this prompt forked off whatever it matched. Zero stops is the
    // ordinary case of a conversation extending itself, and then this is one span.
    let stops = plan_snapshot_stops(
        &engine.slots.live_slot()?.prefix,
        &prompt,
        resume,
        boundary,
        anchor,
        shared_anywhere,
    );
    let mut from = resume;
    for stop in stops {
        if let Some(reason) = prefill(
            engine,
            &abandon,
            &prompt[from..stop.at],
            from,
            prompt_len,
            trace,
        )? {
            abandon_prefill(engine, &abandon, &opened, trace, reason, &prompt)?;
            return Ok(());
        }
        let snapshot = Arc::new(engine.generator.take_cache_snapshot()?.to_host()?);
        let prefix = &mut engine.slots.live_slot()?.prefix;
        match stop.reason {
            SnapshotReason::Anchor => prefix.set_anchor(stop.at, snapshot),
            // Ordinary ring snapshots, and best-effort by design: a long conversation can
            // rotate a branch point out before its next page-out, which costs a future
            // fork a shallower resume and nothing else.
            SnapshotReason::Turn | SnapshotReason::Branch => prefix.push(stop.at, snapshot),
        }
        // Host RAM the slot did not hold a moment ago, and one of the few mutations that
        // happen mid-job.
        log_slots(&engine.slots, logger);
        from = stop.at;
    }
    if let Some(reason) = prefill(engine, &abandon, &prompt[from..], from, prompt_len, trace)? {
        abandon_prefill(engine, &abandon, &opened, trace, reason, &prompt)?;
        return Ok(());
    }

    let stopped = Cell::new(false);
    let disconnected = Cell::new(false);
    // Stamped by the first content event the client's channel accepts, which against
    // the pickup instant is the time to first token.
    let first_sent = &trace.first_sent;
    let mut emitter = Emitter::new(
        &abandon,
        stop_sequences,
        &tools,
        *engine.generator.tokenizer().specials(),
        &stopped,
        &disconnected,
        first_sent,
    );
    // The speculative loop is safe to run whenever a drafter is attached — it falls
    // back to plain rounds and draws the same tokens either way — so this asks the
    // narrower question of whether it could actually speculate from here. A drafter
    // that has fallen behind (a conversation past its context, a page-in with no
    // drafter planes) would otherwise pay the round loop's overhead for nothing.
    engine.generator.note_draft_horizon_at(prompt_len);
    let outcome = if engine.generator.spec_ready_at(prompt_len) {
        engine.generator.decode_loop_spec(
            prompt_len,
            thinking_entry,
            max_new,
            &mut |event| emitter.accept(event),
            &mut || stopped.get() || disconnected.get() || abandon.reason().is_some(),
        )
    } else {
        engine.generator.decode_loop(
            prompt_len,
            thinking_entry,
            max_new,
            &mut |event| emitter.accept(event),
            &mut || stopped.get() || disconnected.get() || abandon.reason().is_some(),
        )
    };

    // Reconcile before anything else, and against `cache_len` rather than the events
    // received: a decode that ends at its cap skips the feed-back forward for its last
    // token, and an EOG stop token is never cached at all.
    let mut history = prompt;
    history.extend(emitter.ids.iter().copied());
    history.truncate(engine.generator.cache_len());
    engine.slots.live_slot()?.prefix.set_tokens(history);
    log_slots(&engine.slots, logger);
    // A decode that returned normally leaves the cache and the history agreeing, so the
    // next request may reuse the prefix; anything that failed above does not.
    let outcome = outcome?;
    engine.dirty = false;
    trace.record.output_tokens = outcome.tokens_out;
    trace.record.thinking_tokens = outcome.thinking_tokens;
    trace.record.decode_secs = outcome.decode_secs;
    trace.record.spec = outcome.spec;
    if let Some(report) = spec_report(outcome.spec) {
        logger.log(report);
    }

    // What the abort owes the client depends on why. A gone client gets nothing —
    // not even a malformation report, since a span truncated by this server's own
    // abort is no evidence about the model's writing. A deadline or shutdown
    // client is still listening, so the turn is closed properly below — healed
    // call, tail, terminal event — with the reply truthfully reported cut short.
    let abandoned = decode_abandon_reason(
        outcome.cancelled,
        stopped.get(),
        disconnected.get(),
        abandon.reason(),
    );
    trace.record.abandoned = abandoned;
    match abandoned {
        Some(reason @ (CancelReason::ClientGone | CancelReason::Shutdown)) => {
            logger.log(ServeLog::AbandonedDuringDecode {
                reason,
                tokens_out: outcome.tokens_out,
                cache_kept: engine.generator.cache_len(),
            });
            if reason == CancelReason::ClientGone {
                return Ok(());
            }
        }
        Some(CancelReason::Deadline) => {
            logger.log(ServeLog::DeadlineDuringDecode {
                ceiling: abandon.ceiling(),
                tokens_out: outcome.tokens_out,
            });
        }
        None => {}
    }

    // A tool call the model never finished writing is closed here rather than
    // abandoned: a half-emitted call the client cannot parse is worse than a
    // complete one with the arguments the model got as far as.
    emitter.heal_open_call();

    // Nothing matched, so the holdback was ordinary text after all. A match
    // withheld it along with everything the model wrote after it.
    if emitter.matched.is_none() {
        let tail = emitter.flush_tail();
        if !tail.is_empty() {
            if !abandon.send(EngineEvent::Text(tail)) {
                return Ok(());
            }
            // A reply short enough for the stop filter to have held all of it back
            // arrives here, and this is then the first thing its client was sent.
            if first_sent.get().is_none() {
                first_sent.set(Some(Instant::now()));
            }
        }
    }
    let stop = if abandoned.is_some() {
        // Cut short by the server, which in the client's terms is the output cap:
        // neither API has a stop reason for a deadline or a shutdown, and the log
        // line above carries the real one.
        StopKind::MaxTokens
    } else {
        terminal_stop(
            emitter.matched.take(),
            emitter.internal_stop.as_deref(),
            emitter.called_tools,
            outcome.hit_eog,
        )
    };
    if let Some(report) = malformation_report(emitter.healed, emitter.quoted, emitter.degraded) {
        logger.log(report);
    }
    trace.record.stop = Some(stop.clone());
    if !abandon.send(EngineEvent::Done {
        stop,
        output_tokens: outcome.tokens_out,
        thinking_tokens: outcome.thinking_tokens,
    }) {
        // The reply is complete and its client never got the event saying so.
        // The stop stands — it is what the reply did — with the undelivered
        // ending recorded next to it, so a report of "the request hung" can be
        // told from one that answered.
        trace.record.error = Some("terminal event undelivered".to_string());
    }
    Ok(())
}

/// Why the finished decode was abandoned, or `None` for a reply that ended
/// naturally — EOG, the token cap, or a stop-sequence match.
///
/// Decided from how the decode loop actually stopped, never from a fresh poll of
/// the cancel token alone: a reason stamped between the loop's last poll and
/// finalization — a deadline racing the final token — must not relabel a
/// complete reply as truncated, or the client retries an answer it already has.
/// `loop_cancelled` is the loop's own report that its stop predicate fired, and
/// a matched stop sequence is one of that predicate's inputs, so it is excluded
/// here: the match is a natural end however the poll went.
fn decode_abandon_reason(
    loop_cancelled: bool,
    stop_matched: bool,
    disconnected: bool,
    reason: Option<CancelReason>,
) -> Option<CancelReason> {
    if !loop_cancelled || stop_matched {
        return None;
    }
    reason.or_else(|| disconnected.then_some(CancelReason::ClientGone))
}

/// Why a prefill pauses to snapshot, which is also how that snapshot is kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SnapshotReason {
    /// The end of this conversation's leading system block. Pinned outside the capacity
    /// ring: it is the shallowest resume point and the only one another conversation
    /// from the same client can reach, so it must not age out with the turn boundaries.
    Anchor,
    /// The end of the shared context, before the generation header — where the next turn
    /// of this conversation resumes.
    Turn,
    /// Where this prompt diverged from the conversation it matched. Snapshotting the fork
    /// point is what lets the next request that forks at the same place resume there
    /// rather than replaying from the last boundary below it.
    Branch,
}

/// One pause in a prefill: run up to `at`, snapshot, carry on from there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SnapshotStop {
    at: usize,
    reason: SnapshotReason,
}

/// Whether a snapshot at `pos` pays for itself on this prefill.
///
/// `already` is the deepest position a later request could resume at WITHOUT this
/// snapshot, and `ceiling` is the first position the prefill will not reach. One
/// predicate for every kind of stop, so the economics cannot drift apart between them:
/// what differs between callers is only what they can already resume from.
///
/// A position at or below `resume` is cached already and cannot be snapshotted without
/// replaying it, and one at or past `ceiling` is not a position this prefill produces.
fn snapshot_worth_taking(pos: usize, resume: usize, ceiling: usize, already: usize) -> bool {
    resume < pos && pos < ceiling && pos.saturating_sub(already) >= SNAPSHOT_MIN_GAIN
}

/// Where this job's prefill pauses to snapshot: ascending, distinct, all strictly
/// between `resume` and the end of the prompt.
///
/// Ascending is not cosmetic. [`PrefixCache::record`] drops every snapshot at or past the
/// position it stores, so a stop taken out of order would silently delete the deeper
/// resume points the earlier stops had just paid for.
///
/// The branch point is the deepest position this prompt shares with ANYTHING the server
/// holds — `shared_elsewhere` is that answer from the matching, and the live slot's own
/// history is only one of its sources.
///
/// It cannot be derived from the slot that won the dispatch. A conversation sharing a
/// long prefix but holding no snapshot beneath it resumes at nothing, loses to a fresh
/// slot, and is then invisible to a slot-derived divergence — while being exactly the
/// case a branch snapshot exists for. Two agent sessions that share a system prompt and
/// diverge where their tool sets differ produce it every time: the shared prefix is real,
/// the fork is real, and without a snapshot there the second session, and every session
/// after it, prefills the whole prefix from zero.
fn plan_snapshot_stops<S>(
    prefix: &PrefixCache<S>,
    prompt: &[u32],
    resume: usize,
    boundary: usize,
    anchor: Option<usize>,
    shared_elsewhere: usize,
) -> Vec<SnapshotStop> {
    // A slot that keeps no snapshots keeps none of these. (The tail snapshot a page-out
    // takes is not one of them, and ignores the cap for its own reasons.)
    if !prefix.keeps_snapshots() {
        return Vec::new();
    }
    let mut stops: Vec<SnapshotStop> = Vec::new();
    // The anchor's beneficiary is a conversation that shares the system block and
    // nothing else, so its saving is measured from zero however much this slot holds.
    if let Some(at) = anchor.filter(|&at| snapshot_worth_taking(at, resume, boundary, 0)) {
        stops.push(SnapshotStop {
            at,
            reason: SnapshotReason::Anchor,
        });
    }
    // The turn boundary has no floor: it is what makes the NEXT turn of this very
    // conversation cheap, which is the basis of turn-level reuse rather than a bet on a
    // fork that may never come.
    //
    // The `< prompt.len()` is belt-and-braces. Every prompt ends with a non-empty
    // generation header, so the boundary is always at least one token short of the end
    // (pinned by `serve::tests::a_submitted_job_carries_the_prompt_split_at_the_generation_header`)
    // — but the
    // final span of the prefill is what produces the logits the decode starts from, so a
    // stop at the very end would leave the decode reading stale ones.
    if resume < boundary && boundary < prompt.len() {
        stops.push(SnapshotStop {
            at: boundary,
            reason: SnapshotReason::Turn,
        });
    }
    // Capped so the prefill that follows still has a token to run, exactly as
    // `shared_with` caps its own answer: the last span is what produces the logits the
    // decode starts from.
    let at = prefix
        .shared_with(prompt)
        .unwrap_or(0)
        .max(shared_elsewhere)
        .min(prompt.len().saturating_sub(1));
    {
        // What a future fork at this position could resume from without it: the deepest
        // snapshot the slot already holds below it, the position this job resumed at, or
        // a stop planned above — a branch point the anchor or the turn boundary almost
        // reaches saves almost nothing.
        let already = [prefix.rewind_to(at).unwrap_or(0), resume]
            .into_iter()
            .chain(stops.iter().map(|stop| stop.at).filter(|pos| *pos <= at))
            .max()
            .unwrap_or(resume);
        // The `any` is an invariant rather than a filter: while the floor is positive, a
        // fork at a position already planned for measures a gain of zero and is excluded
        // above. Two stops at one position would make the second drop the first.
        if snapshot_worth_taking(at, resume, prompt.len(), already)
            && !stops.iter().any(|stop| stop.at == at)
        {
            stops.push(SnapshotStop {
                at,
                reason: SnapshotReason::Branch,
            });
        }
    }
    stops.sort_by_key(|stop| stop.at);
    stops
}

/// Prefill one span of the prompt, checking between chunks whether anyone still wants it.
/// A 100k-token prompt is a minute of GPU time, and running it to completion for a job
/// nobody is waiting on is a minute the next request waits for nothing.
///
/// `Some(reason)` means the job was abandoned mid-span. The cache holds exactly the
/// chunks already written, which is a perfectly good prefix of this prompt — the caller
/// reconciles it into the slot rather than clearing it.
///
/// `total` is the whole prompt, not this span: a span is an implementation detail of how
/// the engine snapshots turn boundaries, while what a waiting client wants to know is how
/// much of its prompt is done. `start_pos` is already the span's position in that prompt,
/// so the two are all the progress report needs.
fn prefill(
    engine: &mut EngineState,
    abandon: &Abandon,
    tokens: &[u32],
    start_pos: usize,
    total: usize,
    trace: &mut JobTrace,
) -> Result<Option<CancelReason>> {
    // The same chunk as the generate path, so splitting a prefill to check for a
    // departed client costs no extra GPU passes.
    let chunk_len = engine.generator.prefill_chunk();
    for (index, chunk) in tokens.chunks(chunk_len).enumerate() {
        if let Some(reason) = abandon.reason() {
            return Ok(Some(reason));
        }
        let at = start_pos + index * chunk_len;
        let started = Instant::now();
        engine.generator.prefill_tokens(chunk, at)?;
        trace.record.prefill_secs += started.elapsed().as_secs_f64();
        trace.record.prefill_tokens += chunk.len();
        // Reported after the chunk lands rather than before it starts, so `done` is
        // what the model has actually read.
        abandon.logger.log(ServeLog::PrefillTick {
            done: at + chunk.len(),
            total,
        });
    }
    Ok(None)
}

/// Wind down a job abandoned mid-prefill without wasting the work: the chunks already
/// written are a valid prefix of this very prompt, so the slot is reconciled to them and
/// stays live. An identical retry — Claude Code retries a failed turn with the same
/// prompt — then resumes where this one stopped instead of prefilling from zero.
fn abandon_prefill(
    engine: &mut EngineState,
    abandon: &Abandon,
    opened: &Cell<bool>,
    trace: &mut JobTrace,
    reason: CancelReason,
    prompt: &[u32],
) -> Result<()> {
    let kept = reconcile_partial_prefill(engine, prompt)?;
    log_slots(&engine.slots, abandon.logger);
    finish_abandoned_before_decode(
        abandon,
        opened,
        trace,
        reason,
        JobPhase::Prefill,
        kept,
        prompt.len(),
        kept,
    );
    Ok(())
}

/// Record exactly what the cache holds — the same reconciliation a finished decode
/// performs, against `cache_len` as the one authority — so the cache and the history
/// agree and `dirty` can be cleared. The retained image's agreement bound is untouched:
/// prefill only ever appends at or above the position the dispatch resumed at, so every
/// row below the bound still holds the keys the truncated history names.
fn reconcile_partial_prefill(engine: &mut EngineState, prompt: &[u32]) -> Result<usize> {
    let cache_len = engine.generator.cache_len();
    let mut history = prompt.to_vec();
    history.truncate(cache_len);
    engine.slots.live_slot()?.prefix.set_tokens(history);
    // The drafter was fed its taps in lockstep with the chunks above, so its rows are
    // this conversation's own; the sync only truncates one that got ahead of the cache.
    engine.generator.sync_drafter_to(cache_len)?;
    engine.dirty = false;
    Ok(cache_len)
}

/// Open the reply, once. The `Start` event is what tells a client how much of its
/// prompt came out of a cache, and it must precede every other event in the stream —
/// so the wind-down paths call this too rather than sending a terminal event into a
/// stream that never began. `false` means the client is gone.
fn open_reply(
    abandon: &Abandon,
    opened: &Cell<bool>,
    input_tokens: usize,
    cached_tokens: usize,
) -> bool {
    if opened.replace(true) {
        return true;
    }
    abandon.send(EngineEvent::Start {
        input_tokens,
        cached_tokens,
    })
}

/// Tell the log — and, when one is still listening, the client — what became of a job
/// abandoned before it decoded anything. `done`/`total` are prompt tokens; `kept` is what
/// the cache holds toward a retry of the same prompt.
///
/// A gone client gets no terminal event: there is nobody to receive one. A deadline or
/// shutdown client is still reading and is owed a complete turn, so it gets one — empty,
/// and truthfully marked as cut short at the output cap, the closest stop either API can
/// express; the log line carries the real reason. The turn is opened first when the
/// dispatch had not got that far, since a client cannot be told how a reply ended
/// before it has been told one started.
fn finish_abandoned_before_decode(
    abandon: &Abandon,
    opened: &Cell<bool>,
    trace: &mut JobTrace,
    reason: CancelReason,
    phase: JobPhase,
    done: usize,
    total: usize,
    kept: usize,
) {
    trace.record.abandoned = Some(reason);
    match reason {
        CancelReason::Deadline => abandon.logger.log(ServeLog::DeadlineBeforeDecode {
            ceiling: abandon.ceiling(),
            done,
            total,
        }),
        CancelReason::ClientGone | CancelReason::Shutdown => {
            abandon.logger.log(ServeLog::AbandonedBeforeDecode {
                reason,
                phase,
                done,
                total,
                kept,
            })
        }
    }
    if reason != CancelReason::ClientGone {
        trace.record.stop = Some(StopKind::MaxTokens);
        if open_reply(abandon, opened, total, kept) {
            abandon.send(EngineEvent::Done {
                stop: StopKind::MaxTokens,
                output_tokens: 0,
                thinking_tokens: 0,
            });
        }
    }
}

/// Rewind the live conversation to the snapshot it took at `pos`, uploading the host copy
/// of its sliding-window rings on the way. The full-attention layers are restored by being
/// shortened, which is only sound because the positions below `pos` still hold this
/// conversation's own keys — the prompt shares them, or the plan would not name this
/// snapshot.
///
/// The drafter is rewound with it. Its cache is position-indexed with no ring, so a
/// drafter that reached `pos` keeps those rows exactly and speculation continues; one
/// that never got there — it fell behind at its own smaller context — has no state
/// for them and resynchronizes on a later prefill from zero.
fn restore_live(engine: &mut EngineState, pos: usize) -> Result<()> {
    let EngineState {
        generator,
        device,
        slots,
        ..
    } = engine;
    let live = slots
        .live
        .ok_or_else(|| anyhow!("a rewind was planned with no conversation in the cache"))?;
    let rings = slots.slots[live]
        .prefix
        .snapshot_at(pos)
        .ok_or_else(|| anyhow!("cache slot {live} has no snapshot at {pos} to rewind to"))?;
    // The rings set the cache's length from the position they were taken at, while the
    // full-attention layers are shortened to `pos`; the layers only agree because those are
    // the same number. The same check `page_in` makes, for the same reason — a mismatch
    // would otherwise surface as the cache-length disagreement below and cost a silent full
    // replay of the prompt.
    ensure!(
        rings.pos == pos,
        "cache slot {live}'s snapshot at {pos} records position {} instead",
        rings.pos
    );
    generator.restore_cache_snapshot(&rings.to_snapshot(device)?)?;
    generator.sync_drafter_to(pos)?;
    slots.rewind_live(pos)
}

/// Start the live conversation over from nothing: clear the KV cache and the
/// drafter's with it, and drop the slot's history and retained image.
///
/// Both replay paths go through here — the `Cold` plan and the cache-length
/// mismatch — so neither can leave the drafter holding positions the target has just
/// forgotten, which the next prefill's tap injection would read as its own.
///
/// The slot's link to its stored image goes too: the conversation that image holds is
/// not the one this slot is about to build, so keeping the link would pin a file
/// against eviction on behalf of a conversation nothing is serving.
fn restart_live(engine: &mut EngineState, disk: Option<&DiskCache>) -> Result<()> {
    engine.generator.reset_cache()?;
    engine.generator.reset_drafter()?;
    if let (Some(live), Some(disk)) = (engine.slots.live, disk) {
        disk.unlink(live);
    }
    engine.slots.restart_live()
}

/// Page the live conversation out of the KV cache into a host image, freeing the cache for
/// another slot. Nothing is lost: the image plus the tail snapshot are everything a later
/// page-in needs to continue the conversation where it left off.
///
/// The tail snapshot is taken whatever `cache_snapshots` says. It is not a turn boundary
/// but a requirement of paging: the sliding-window rings hold the last 512 positions in
/// ring order and cannot be reconstructed by truncating a longer image, so a slot without
/// one could only ever be resumed from position zero.
///
/// The drafter's rows are imaged alongside, so a conversation that comes back keeps
/// speculating instead of decoding plain until its next prefill from zero. A drafter
/// holding nothing — none attached, or one that fell behind this conversation's
/// positions — stores no planes, and the page-in resets it instead.
///
/// This is also where the disk tier is fed: the images the page-out just built are
/// exactly what a stored file holds, so the write is queued from here rather than
/// rebuilding them anywhere else. Queueing is a channel send that cannot fail or
/// block — the engine thread never waits on the writer.
fn page_out_live(
    engine: &mut EngineState,
    disk: Option<&DiskCache>,
    logger: &ServeLogger,
) -> Result<()> {
    let Some(live) = engine.slots.live else {
        return Ok(());
    };
    let pos = engine.generator.cache_len();
    // An empty cache is nothing to come back to, and an image of zero positions would only
    // ever resume from zero anyway.
    if pos == 0 {
        if let (Some(emptied), Some(disk)) = (engine.slots.abandon_live(), disk) {
            disk.unlink(emptied);
        }
        log_slots(&engine.slots, logger);
        return Ok(());
    }
    let started = Instant::now();
    let image = Arc::new(engine.generator.export_full_kv()?);
    let rings = Arc::new(engine.generator.take_cache_snapshot()?.to_host()?);
    let planes = engine
        .generator
        .export_drafter_cache()?
        .filter(|planes| planes.pos > 0)
        .map(Arc::new);
    let moved =
        image.byte_len() + rings.byte_len() + planes.as_ref().map_or(0, |planes| planes.byte_len());
    engine.slots.page_out(pos, rings, image, planes);
    logger.log(ServeLog::SlotPagedOut {
        slot: live,
        pos,
        bytes: moved,
        elapsed: started.elapsed(),
    });
    log_slots(&engine.slots, logger);
    if let Some(disk) = disk {
        let held = &engine.slots.slots[live];
        if let Some(image) = held.full_kv.as_ref() {
            disk.queue_write(
                live,
                &held.prefix.tokens,
                image,
                held.prefix
                    .all_snapshots()
                    .map(|(pos, rings)| (*pos, Arc::clone(rings))),
                held.draft_kv.as_ref(),
            );
        }
    }
    Ok(())
}

/// What a hydration attempt did.
enum Hydration {
    /// The stored conversation is in slot `slot` and the model's cache is resumed at
    /// `restore`, exactly as an in-process swap would have left them.
    Installed { slot: usize, restore: usize },
    /// The file could not be used — it is deleted, and the caller falls back to what
    /// the warm slots offered. Nothing was uploaded and no slot was touched.
    Unavailable,
    /// Nobody is waiting for this job any more, and it was noticed before the
    /// expensive part.
    Abandoned(CancelReason),
}

/// Bring a stored conversation into `target` and page it into the model's cache.
///
/// The order is the swap arm's, for the same reasons: the read and the two transfers
/// each cost hundreds of milliseconds, so a job nobody is waiting for is checked for
/// before any of them is paid. Before the read nothing has been written and the cache
/// still agrees with the live slot's history, so that exit clears `dirty`; after the
/// page-out the cache belongs to nobody and `dirty` stays set for the caller's reset,
/// which every slot image survives.
fn hydrate_slot(
    engine: &mut EngineState,
    tier: &DiskCache,
    candidate: &DiskCandidate,
    prompt: &[u32],
    target: FreshSlot,
    abandon: &Abandon,
    logger: &ServeLogger,
) -> Result<Hydration> {
    if let Some(reason) = abandon.reason() {
        engine.dirty = false;
        return Ok(Hydration::Abandoned(reason));
    }
    let started = Instant::now();
    // Every failure mode of the disk tier ends here: a cache miss, never a wrong
    // resume. The read validates the whole file and checks the history it holds
    // against this prompt over the positions about to be resumed, so a file found by
    // a conversation it does not hold is refused rather than uploaded.
    let Some(image) = tier.load(candidate, prompt) else {
        return Ok(Hydration::Unavailable);
    };
    // Everything that decides whether this image can be uploaded at all is settled HERE,
    // while the live conversation still owns the cache and this job can still fall back
    // to what the warm slots offered. Past the page-out it cannot: the imports refuse a
    // mismatch before they write anything, so nothing would be corrupted, but the request
    // would die as a server error over a cache that was only ever an optimization.
    if let Some(why) = engine.rejects_image(&image, candidate.resume) {
        // Left on disk rather than deleted: an image this server cannot upload may be one
        // the next server can — a `--ctx` raised again, a drafter put back — and deletion
        // here would answer a question about THIS process's configuration by throwing away
        // another conversation's cache. The fail-closed rule is the same one the tier
        // applies to a class it does not recognise. It does come out of circulation for
        // this run, though: every later request sharing this prefix would otherwise read
        // and compose the same gigabytes to turn them down again.
        logger.log(ServeLog::DiskCacheFailed {
            action: "matching a stored image to the loaded model",
            error: why,
        });
        tier.set_unusable(candidate);
        return Ok(Hydration::Unavailable);
    }
    page_out_live(engine, Some(tier), logger)?;
    if let Some(reason) = abandon.reason() {
        return Ok(Hydration::Abandoned(reason));
    }
    let restore = candidate.resume;
    let slot = install_stored_image(&mut engine.slots, target, image, restore);
    // From here it is an ordinary swap: the upload, the drafter's planes and the
    // agreement bound all come from the one path that has always done them.
    page_in(engine, slot, restore, logger)?;
    tier.note_hydrated(slot, candidate, started.elapsed().as_millis() as u64);
    Ok(Hydration::Installed { slot, restore })
}

/// Rebuild a stored image into `target` and return the slot it landed in: the state
/// a slot paged out in this process would hold, so the page-in that follows is an
/// ordinary one.
///
/// Snapshots past `restore` are dropped rather than installed. `page_in` would drop
/// them a moment later anyway — they describe a branch this job is about to abandon
/// — and dropping them here is what keeps `restore` itself inside the slot's
/// snapshot capacity: it is then the deepest position installed, which is the one
/// place the capacity ring never evicts from.
fn install_stored_image(
    slots: &mut Slots,
    target: FreshSlot,
    image: DiskImage,
    restore: usize,
) -> usize {
    let DiskImage {
        tokens,
        full_kv,
        snapshots,
        drafter,
    } = image;
    let snapshots = snapshots
        .into_iter()
        .filter(|(pos, _)| *pos <= restore)
        .map(|(pos, rings)| (pos, Arc::new(rings)))
        .collect();
    slots.install(
        target,
        tokens,
        Arc::new(full_kv),
        snapshots,
        drafter.map(Arc::new),
    )
}

/// Page slot `slot`'s conversation into the KV cache, resuming at `restore` — a snapshot
/// position that slot holds, and therefore one its image covers with the same tokens the
/// arriving prompt shares. The cache must be free: the caller pages the live conversation
/// out first.
///
/// The drafter is brought back with it, from the planes the page-out imaged: a slot that
/// has none — nothing was attached then, or the drafter had fallen behind — resets it, and
/// that conversation decodes plain until it next prefills from zero.
fn page_in(
    engine: &mut EngineState,
    slot: usize,
    restore: usize,
    logger: &ServeLogger,
) -> Result<()> {
    let started = Instant::now();
    let EngineState {
        generator,
        device,
        slots,
        ..
    } = engine;
    let target = slots
        .slots
        .get(slot)
        .ok_or_else(|| anyhow!("cache slot {slot} does not exist"))?;
    let image = target
        .full_kv
        .as_ref()
        .ok_or_else(|| anyhow!("cache slot {slot} holds no image to page in"))?;
    let rings = target
        .prefix
        .snapshot_at(restore)
        .ok_or_else(|| anyhow!("cache slot {slot} has no snapshot at {restore}"))?;
    ensure!(
        restore <= image.pos,
        "cache slot {slot} was asked to resume at {restore}, past the {} positions its \
         image holds",
        image.pos
    );
    // The two uploads below set the cache's length independently — `import_full_kv` from
    // `restore`, `restore_cache_snapshot` from the snapshot's own position — and the layers
    // only end up agreeing because those are the same number. Checked rather than assumed:
    // a mismatch would leave the full-attention layers and the rings describing different
    // lengths, which nothing downstream could detect.
    ensure!(
        rings.pos == restore,
        "cache slot {slot}'s snapshot at {restore} records position {} instead",
        rings.pos
    );
    // The drafter comes back only from planes that reach `restore`, and only when there
    // is a drafter to bring back at all; anything else is skipped rather than partly
    // uploaded or refused (see `drafter_planes_usable`). The planes stay in the slot
    // either way — a later turn that resumes at or below their length can still use
    // them, and a server restarted with a drafter can still use ones stored without.
    let planes = target.draft_kv.as_ref().filter(|planes| {
        drafter_planes_usable(generator.drafter_kind(), planes.kind(), restore, planes.pos)
    });
    // Order is load-bearing since KV allocation went lazy: `import_full_kv` is
    // what GROWS the full-attention buffers to `restore` (a paged-in
    // conversation can be longer than anything this instance has run), and the
    // ring restore's own per-layer bound (`pos <= slots`) passes only after
    // that growth. Swapping these two lines fails the restore at layer zero —
    // after the live conversation was already paged out.
    generator.import_full_kv(image, restore)?;
    generator.restore_cache_snapshot(&rings.to_snapshot(device)?)?;
    // A drafter that will not take these planes costs speculation, not the request. The
    // planes are the one part of a slot image whose shape this server cannot vouch for:
    // the checkpoint id binds the TARGET's geometry, so rows and rings that read back at
    // all necessarily fit, while the drafter is a separate GGUF chosen by a separate flag
    // — restart with a different `--draft`, or a smaller `--draft-ctx`, and planes stored
    // by the last run describe a cache this one does not have. With drafting on by
    // default the plane-less case is also the COMMON hydration against slots written by
    // a `--no-draft` run or a pre-drafting build, not just a flag-change edge. Either
    // way the degradation is permanent for the conversation: the drafter's cache is fed
    // by target-layer taps during target forwards, so a reset drafter at a nonzero
    // restore point has no way to catch up (`drafter_span_rows` stays 0), and re-seeding
    // would mean re-running the target prefill the snapshot exists to avoid; accepted
    // and documented in decisions.md "Serving". Every ground a stored record can be
    // refused on — kind, shape,
    // position, capacity — is settled before a single row is written, so a refusal leaves
    // the drafter untouched; a device failure part-way through the writes does not, which
    // is why the repair is a full reset rather than a length fix, and why that reset is
    // not allowed to fail quietly. Either way the conversation carries on with the target
    // KV it just hydrated, decoding plain.
    // Split so a failure reports what actually failed: with the no-planes reset folded into
    // the same match, a reset that errored was logged as a failure to import planes that
    // were never there.
    let imported = match planes {
        Some(planes) => generator.import_drafter_cache(planes, restore),
        None => {
            generator
                .reset_drafter()
                .context("clearing the drafter for a conversation with no stored planes")?;
            Ok(())
        }
    };
    let drafter_planes = match imported {
        Ok(()) => planes,
        Err(e) => {
            // Reported as a disk-tier failure because that is the only way to get here:
            // planes a page-out wrote in this process describe this process's drafter.
            logger.log(ServeLog::DiskCacheFailed {
                action: "importing a stored drafter's planes",
                error: format!("{e:#}"),
            });
            // A refused import costs speculation and not the request, but a
            // refused RESET is a different failure: it leaves the drafter
            // holding whatever the last conversation put there, and the next
            // prefill would feed this conversation's tokens against those rows.
            // There is no local repair for that, so it propagates.
            generator.reset_drafter()?;
            None
        }
    };
    // What the uploads above actually moved: the image's rows below `restore` — its
    // layout is contiguous in position, so the share is exact — plus the rings, plus the
    // drafter rows, when those were uploaded at all. A refused drafter contributes zero
    // rather than an undercount: `import_cache` settles every shape, and allocates
    // everything it needs, before its write loop — so a refusal on any ground the stored
    // record can be wrong about uploaded no layer at all.
    let moved = image.byte_len() / image.pos.max(1) * restore
        + rings.byte_len()
        + drafter_planes.map_or(0, |planes| planes.byte_len() / planes.pos.max(1) * restore);
    slots.page_in(slot, restore);
    logger.log(ServeLog::SlotPagedIn {
        slot,
        restore,
        bytes: moved,
        elapsed: started.elapsed(),
    });
    log_slots(slots, logger);
    Ok(())
}

/// Forwards decode events to the client: reasoning verbatim, answer text through the
/// stop-sequence holdback, tool calls through the span parser, and every token id into
/// `ids` for the cache history.
///
/// The two flags live outside the struct so `should_stop` can read them while the event
/// callback holds the emitter itself mutably.
struct Emitter<'a> {
    abandon: &'a Abandon<'a>,
    stop: StopFilter,
    ids: Vec<u32>,
    /// The stop sequence that matched, once one has.
    matched: Option<String>,
    /// The engine's own `</assistant>`, when it was added — the one match that is
    /// an end of turn rather than a stop sequence.
    internal_stop: Option<String>,
    /// The declared argument types, and the switch that turns tool parsing on:
    /// with no tools in the request, `<tool_call>` is text like any other.
    schemas: ToolSchemas,
    /// The running vocabulary's marker ids, which is how a `<tool_call>` token
    /// is told from a token that merely decodes to that text.
    specials: Specials,
    /// The call currently being parsed, when the model is inside one. While it is
    /// open, answer text and stop sequences are both suspended: a stop firing
    /// mid-call would deliver a call that cannot be parsed.
    span: Option<ToolSpan>,
    /// Whether this generation has produced a call at all, which is what makes
    /// its terminal stop a `ToolUse`.
    called_tools: bool,
    /// Calls the model left unfinished, and the engine closed for it. Counted
    /// whatever the repair produced, dropped pairs and dropped calls included:
    /// the question these answer is how often the model writes a call this
    /// parser cannot take at face value, which is the evidence the decision to
    /// constrain decoding is waiting on. Reported once per request, and only
    /// when there is something to report.
    healed: usize,
    /// Values the model finished writing that were not the JSON their schema
    /// asked for, and so shipped as quoted strings.
    quoted: usize,
    /// Spans that never became a dispatchable call and went back to the client
    /// as answer text. Distinct from [`Emitter::healed`]: a heal delivers a call
    /// the engine had to close, while these delivered no call at all.
    degraded: usize,
    stopped: &'a Cell<bool>,
    disconnected: &'a Cell<bool>,
    /// When the first content event of this reply reached the client's channel, which
    /// is the honest end of a client's wait: everything before it — the queue, the
    /// model load, paging, the whole prefill — is time with nothing to show.
    first_sent: &'a Cell<Option<Instant>>,
}

impl<'a> Emitter<'a> {
    fn new(
        abandon: &'a Abandon<'a>,
        stop_sequences: Vec<String>,
        tools: &[serde_json::Value],
        specials: Specials,
        stopped: &'a Cell<bool>,
        disconnected: &'a Cell<bool>,
        first_sent: &'a Cell<Option<Instant>>,
    ) -> Self {
        let schemas = ToolSchemas::build(tools);
        // A client that asked for `</assistant>` itself gets the sequence it
        // asked for, reported under its own name; otherwise the sequence is the
        // engine's, and ends the turn without ever being named to the client.
        let internal_stop = (schemas.enabled()
            && !stop_sequences.iter().any(|s| s == ASSISTANT_CLOSE))
        .then(|| ASSISTANT_CLOSE.to_string());
        let mut sequences = stop_sequences;
        sequences.extend(internal_stop.clone());
        Self {
            abandon,
            stop: StopFilter::new(&sequences),
            ids: Vec::new(),
            matched: None,
            internal_stop,
            schemas,
            specials,
            span: None,
            called_tools: false,
            healed: 0,
            quoted: 0,
            degraded: 0,
            stopped,
            disconnected,
            first_sent,
        }
    }

    fn accept(&mut self, event: GenEvent) {
        self.ids.push(event.id());
        self.abandon.logger.log(ServeLog::DecodeTick {
            tokens_out: self.ids.len(),
            thinking: matches!(event, GenEvent::ThinkingTok { .. }),
        });
        match event {
            // Stop sequences never apply inside the reasoning block: a match there is the
            // model thinking about the sequence, not finishing its reply. Neither does
            // tool parsing — a `<tool_call>` there is the model reasoning about calling
            // something, not calling it.
            GenEvent::ThinkingTok { text, .. } => {
                if !text.is_empty() {
                    self.send(EngineEvent::Thinking(text));
                }
            }
            GenEvent::TextTok { id, text } if self.schemas.enabled() => {
                self.accept_tool_text(id, &text)
            }
            GenEvent::TextTok { text, .. } => self.push_answer(&text),
        }
    }

    /// The answer path for a job that carries tools, where the `<tool_call>` and
    /// `</tool_call>` tokens open and close a call instead of writing their own
    /// text.
    ///
    /// A decode step finalizes whatever text the tokens before it left pending, so
    /// the marker is always the tail of its own chunk and what precedes it belongs
    /// to whichever side of the boundary came first.
    ///
    /// Both tokens are structural wherever they land. The template writes a
    /// literal `<tool_call>` inside an argument as ordinary content — it never
    /// encodes to the added token — so the model emitting the token is the model
    /// framing a call, and reading it as quoted text instead would let one
    /// malformed value swallow the rest of the reply.
    fn accept_tool_text(&mut self, id: u32, text: &str) {
        if id == self.specials.tool_call_open {
            let before = before_marker(text, TOOL_CALL_OPEN_TEXT);
            self.feed_span(before, true);
            // A call left unterminated by the one that follows it is closed
            // here, so the two never merge into one.
            self.close_span(false);
            self.span = Some(ToolSpan::default());
        } else if id == self.specials.tool_call_close && self.span.is_some() {
            let before = before_marker(text, TOOL_CALL_CLOSE_TEXT);
            self.feed_span(before, true);
            self.close_span(true);
        } else {
            // A close marker with no call open closes nothing; it falls through
            // to the answer as the text the model wrote.
            self.feed_span(text, false);
        }
    }

    /// Deliver `text` to wherever the model currently is — the open call, or the
    /// answer. `boundary` says a call is about to open or close, which is what
    /// settles answer text the stop filter was holding.
    fn feed_span(&mut self, text: &str, boundary: bool) {
        let Some(mut span) = self.span.take() else {
            self.push_answer(text);
            // Answer text held back against a stop sequence cannot grow into one
            // across a call, so a call boundary settles it.
            if boundary {
                let tail = self.stop.flush();
                if !tail.is_empty() {
                    self.send(EngineEvent::Text(tail));
                }
            }
            return;
        };
        let mut out = Vec::new();
        span.push(text, &self.schemas, &mut out);
        self.span = Some(span);
        self.emit_span(out);
    }

    /// End the open call, if there is one. `closed` says the model wrote
    /// `</tool_call>`.
    ///
    /// A span that never became a dispatchable call gives its raw text back to
    /// the client as answer text. Discarding it would silently truncate a reply
    /// over a formatting slip, and inventing a call out of it would be worse:
    /// the client would dispatch something the model never asked for.
    fn close_span(&mut self, closed: bool) {
        let Some(mut span) = self.span.take() else {
            return;
        };
        let mut out = Vec::new();
        match span.finish(&self.schemas, closed, &mut out) {
            SpanEnd::Delivered { repaired } => {
                if repaired {
                    self.healed += 1;
                }
                self.quoted += span.quoted;
                self.emit_span(out);
            }
            SpanEnd::Degraded(raw) => {
                // The markers go back too: what the client receives is then
                // exactly the text the model wrote, which is the only reading
                // of it that cannot mislead.
                let close = if closed { TOOL_CALL_CLOSE_TEXT } else { "" };
                let text = format!("{TOOL_CALL_OPEN_TEXT}{raw}{close}");
                self.degraded += 1;
                self.abandon
                    .logger
                    .log(ServeLog::ToolSpanDegraded { text: text.clone() });
                self.push_answer(&text);
            }
        }
    }

    /// Answer text, through the stop-sequence holdback.
    fn push_answer(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let (ready, matched) = self.stop.push(text);
        if !ready.is_empty() {
            self.send(EngineEvent::Text(ready));
        }
        if let Some(sequence) = matched {
            self.matched = Some(sequence);
            self.stopped.set(true);
        }
    }

    fn emit_span(&mut self, out: Vec<SpanOut>) {
        for item in out {
            match item {
                SpanOut::Start(name) => {
                    self.called_tools = true;
                    self.send(EngineEvent::ToolCallStart { name });
                }
                SpanOut::Delta(delta) => self.send(EngineEvent::ToolCallDelta(delta)),
                SpanOut::End => self.send(EngineEvent::ToolCallEnd),
            }
        }
    }

    /// Close a call the model was still writing when generation ended, so the
    /// client gets a complete arguments object rather than a fragment — or, when
    /// the span never named a callable tool, the text the model actually wrote.
    /// Either way the span's content reaches the client; none of it is dropped.
    fn heal_open_call(&mut self) {
        self.close_span(false);
    }

    /// Everything the stop filter still holds, minus a half-written
    /// `</assistant>`. The engine added that sequence itself, so the start of it
    /// is no more the client's text than a complete match would be — unless it
    /// also starts a sequence the client asked for, which the client would have
    /// been given either way.
    fn flush_tail(&mut self) -> String {
        let tail = self.stop.flush();
        let Some(internal) = self.internal_stop.as_deref() else {
            return tail;
        };
        if tail.is_empty()
            || !internal.starts_with(&tail)
            || self.stop.starts_other(internal, &tail)
        {
            return tail;
        }
        String::new()
    }

    fn send(&self, event: EngineEvent) {
        if !self.abandon.send(event) {
            self.disconnected.set(true);
            return;
        }
        // Stamped after the event lands rather than before it is offered: an event
        // still waiting out a slow reader's backpressure has not been delivered.
        if self.first_sent.get().is_none() {
            self.first_sent.set(Some(Instant::now()));
        }
    }
}

/// The record of tool spans this request could not take at face value, or `None`
/// when every call arrived exactly as written.
///
/// Silent on a clean request on purpose: a log that says nothing when nothing
/// went wrong is one an operator will still read when something does. The
/// numbers exist because whether to constrain decoding is a question about the
/// real malformation rate, and normal serving is where that rate is observable.
fn malformation_report(healed: usize, quoted: usize, degraded: usize) -> Option<ServeLog> {
    if healed == 0 && quoted == 0 && degraded == 0 {
        return None;
    }
    Some(ServeLog::ToolSpanReport {
        healed,
        quoted,
        degraded,
    })
}

/// The record of what speculation did for this reply, or `None` when it was not
/// the decode path (no drafter, or one that could not speculate from where this
/// reply started).
///
/// One report per request, never per token: the numbers worth watching are the
/// acceptance rate — what says whether the drafter is earning its round trips — and
/// the paused count, which says the wall-clock controller decided it was not.
fn spec_report(spec: Option<SpecStats>) -> Option<ServeLog> {
    let stats = spec.filter(|spec| spec.rounds > 0)?;
    Some(ServeLog::SpecReport { stats })
}

/// What ended the generation, in the client's terms.
///
/// A turn that called a tool is waiting on the results however it ended — except
/// at the output cap, where the truthful answer is that the reply was cut short,
/// healed calls or not.
fn terminal_stop(
    matched: Option<String>,
    internal: Option<&str>,
    called_tools: bool,
    hit_eog: bool,
) -> StopKind {
    let stop = match matched {
        // The engine's own `</assistant>` is the end of the turn, not a sequence
        // the client is owed a report of.
        Some(sequence) if Some(sequence.as_str()) == internal => StopKind::EndTurn,
        Some(sequence) => StopKind::StopSequence(sequence),
        None if hit_eog => StopKind::EndTurn,
        None => StopKind::MaxTokens,
    };
    if called_tools && !matches!(stop, StopKind::MaxTokens) {
        StopKind::ToolUse
    } else {
        stop
    }
}

/// The part of a chunk that precedes the marker the token it came from wrote.
/// The last occurrence is the token's own: anything earlier is text the stream
/// had been withholding, and belongs to whatever came before the boundary.
///
/// Splitting at the last occurrence is safe because of how `DecodeStream` holds
/// text back: it withholds only when a decode ends mid-UTF-8-sequence, and
/// releases everything the moment a token completes the character. So a chunk is
/// at most one incomplete character's worth of earlier bytes followed by this
/// token's own text — never a spelled-out `<tool_call>` from earlier tokens
/// followed by the added token of the same name. An earlier occurrence in the
/// chunk can therefore only be the model's own text, which is exactly what the
/// prefix is meant to carry.
fn before_marker<'a>(text: &'a str, marker: &str) -> &'a str {
    text.rsplit_once(marker).map_or(text, |(before, _)| before)
}

/// One instruction from the span parser, in the order the client must see it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SpanOut {
    Start(String),
    Delta(String),
    End,
}

/// How an argument's value is turned into JSON.
///
/// The wire format writes a string value raw — unquoted and unescaped — and
/// every other type as compact JSON, so the declared type is what says which of
/// the two the model just wrote.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum ValueShape {
    /// Declared `string`, or an enum whose members are all strings: quoted and
    /// escaped as it streams.
    Text,
    /// Declared as something else, or not declared at all: held until the value
    /// is complete, then emitted as JSON if it parses as JSON and as a string if
    /// it does not.
    #[default]
    Json,
    /// A union that includes `string`, `["string","null"]` being the common
    /// optional-string shape. Held until complete, then taken as JSON only if
    /// what it parses to is one of the union's other members — so a nullable
    /// string argument of `123` stays the string it was declared to be, while a
    /// literal `null` comes through as null.
    Union(JsonTypes),
}

/// A set of JSON types, small enough to stay `Copy` so a shape lookup is cheap.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct JsonTypes(u8);

impl JsonTypes {
    const NULL: u8 = 1 << 0;
    const BOOLEAN: u8 = 1 << 1;
    const NUMBER: u8 = 1 << 2;
    const OBJECT: u8 = 1 << 3;
    const ARRAY: u8 = 1 << 4;

    /// Add one JSON Schema type name. `integer` is a JSON number, and `string`
    /// is deliberately not representable: it is the case the caller handles.
    fn insert(&mut self, name: &str) {
        self.0 |= match name {
            "null" => Self::NULL,
            "boolean" => Self::BOOLEAN,
            "number" | "integer" => Self::NUMBER,
            "object" => Self::OBJECT,
            "array" => Self::ARRAY,
            _ => 0,
        };
    }

    fn is_empty(self) -> bool {
        self.0 == 0
    }

    fn contains(self, value: &serde_json::Value) -> bool {
        let bit = match value {
            serde_json::Value::Null => Self::NULL,
            serde_json::Value::Bool(_) => Self::BOOLEAN,
            serde_json::Value::Number(_) => Self::NUMBER,
            serde_json::Value::Object(_) => Self::OBJECT,
            serde_json::Value::Array(_) => Self::ARRAY,
            // A quoted string in the text would have to be the model writing
            // JSON where the schema said raw text, which is not worth guessing.
            serde_json::Value::String(_) => 0,
        };
        self.0 & bit != 0
    }
}

/// The declared argument types, by tool name then parameter name.
///
/// A tool, parameter or type the request did not declare is simply absent: the
/// parser then decides from the value itself, which is the only thing left to
/// go on when a model calls something it was never given.
struct ToolSchemas {
    tools: HashMap<String, HashMap<String, ValueShape>>,
    /// Whether the request carried tool definitions at all — including ones this
    /// lookup could make nothing of, which still put the model in a position to
    /// call something.
    enabled: bool,
}

impl ToolSchemas {
    fn build(tools: &[serde_json::Value]) -> Self {
        let mut map: HashMap<String, HashMap<String, ValueShape>> = HashMap::new();
        for tool in tools {
            let function = &tool["function"];
            let Some(name) = function["name"].as_str() else {
                continue;
            };
            let mut parameters = HashMap::new();
            if let Some(properties) = function["parameters"]["properties"].as_object() {
                for (key, schema) in properties {
                    if let Some(shape) = declared_shape(schema) {
                        parameters.insert(key.clone(), shape);
                    }
                }
            }
            map.insert(name.to_string(), parameters);
        }
        Self {
            tools: map,
            enabled: !tools.is_empty(),
        }
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    /// Whether `name` is a tool the request declared, which is the only evidence
    /// available that a name cut short by the end of generation was complete.
    fn declares(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    fn shape(&self, tool: &str, key: &str) -> ValueShape {
        self.tools
            .get(tool)
            .and_then(|parameters| parameters.get(key))
            .copied()
            .unwrap_or(ValueShape::Json)
    }
}

/// How one parameter's schema says its value will be written, or `None` when the
/// schema does not say plainly enough to act on — `anyOf`, `$ref` and the rest
/// leave the value to be classified by what the model actually wrote.
fn declared_shape(schema: &serde_json::Value) -> Option<ValueShape> {
    match &schema["type"] {
        serde_json::Value::String(kind) if kind == "string" => Some(ValueShape::Text),
        serde_json::Value::String(_) => Some(ValueShape::Json),
        serde_json::Value::Array(kinds) => {
            let names: Vec<&str> = kinds.iter().filter_map(|kind| kind.as_str()).collect();
            if names.is_empty() {
                return None;
            }
            if !names.contains(&"string") {
                return Some(ValueShape::Json);
            }
            let mut others = JsonTypes::default();
            for name in names.iter().filter(|name| **name != "string") {
                others.insert(name);
            }
            // A union of nothing but string is a string, and can stream.
            Some(if others.is_empty() {
                ValueShape::Text
            } else {
                ValueShape::Union(others)
            })
        }
        // No type, but an enum of strings says the same thing a string type does.
        _ => match schema["enum"].as_array() {
            Some(values) if !values.is_empty() && values.iter().all(|v| v.is_string()) => {
                Some(ValueShape::Text)
            }
            _ => None,
        },
    }
}

/// Where inside a call the parser is. The interior markers are ordinary text, so
/// each state is "accumulate until this marker turns up", and the holdback keeps
/// a marker split across decode steps from leaking to the client.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum SpanState {
    /// Before `<function=`. The template writes a newline here and nothing else.
    #[default]
    Prelude,
    /// Inside `<function=`…`>`: the tool's name.
    Name,
    /// Between the name's `>` and the next `<parameter=` or `</function>`.
    AwaitParam,
    /// Inside `<parameter=`…`>`: the argument's key.
    Key,
    /// After the key's `>`: the value, up to `</parameter>`.
    Value,
    /// After `</function>`. The call is complete and the rest of the span is the
    /// template's trailing newline.
    Done,
}

/// How a span ended, which is what the emitter needs to know to report it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SpanEnd {
    /// A call went out. `repaired` when the engine had to close it — the model
    /// stopped mid-call, or wrote `</tool_call>` without `</function>`.
    Delivered { repaired: bool },
    /// Nothing dispatchable was written. Carries the raw text of the span, for
    /// the client to receive as the answer text it turned out to be.
    Degraded(String),
}

/// One tool call, parsed out of the text between the `<tool_call>` and
/// `</tool_call>` tokens.
///
/// Emits the arguments object incrementally: `{`, then `"key":value` per pair,
/// then `}`. String values stream escaped as they arrive; everything else is
/// held until `</parameter>` says the value is complete, because a value that
/// turns out not to be JSON has to be quoted, and text already sent cannot be
/// taken back.
#[derive(Debug, Default)]
struct ToolSpan {
    state: SpanState,
    /// Text whose meaning the current state has not settled yet.
    buffer: String,
    /// Everything the span has been fed, verbatim. Kept so a span that never
    /// becomes a call can be handed back as the text the model wrote instead of
    /// disappearing.
    raw: String,
    name: String,
    /// Whether `ToolCallStart` has gone out, which is also what says a partial
    /// span has an arguments object left open.
    started: bool,
    /// Values this call shipped as quoted strings after the model failed to
    /// write the JSON its schema asked for. See [`Emitter::quoted`].
    quoted: usize,
    /// Pairs emitted so far, for the comma between them.
    pairs: usize,
    key: String,
    shape: ValueShape,
    /// A held value, for the shapes that cannot stream.
    value: String,
    /// Whether the newline that frames the start of the current value is still
    /// to be stripped.
    value_lead: bool,
    /// Whether the current pair's `"key":"` has gone out. A streaming string
    /// commits it on its first content rather than when the value opens, so a
    /// generation that stopped at `<parameter=key>` — where nothing of the value
    /// arrived — can still drop the pair whole.
    value_open: bool,
}

impl ToolSpan {
    /// Feed the interior text of one decode step.
    fn push(&mut self, text: &str, schemas: &ToolSchemas, out: &mut Vec<SpanOut>) {
        self.raw.push_str(text);
        self.buffer.push_str(text);
        loop {
            let advanced = match self.state {
                // Whatever precedes `<function=` is the template's newline. A
                // span that never gets here never names anything, and `finish`
                // hands it back as text.
                SpanState::Prelude => self.take_marker(FUNCTION_OPEN).map(|_| SpanState::Name),
                SpanState::Name => self.take_marker(TAG_CLOSE).map(|before| {
                    self.name.push_str(&before);
                    self.start(out);
                    SpanState::AwaitParam
                }),
                SpanState::AwaitParam => {
                    self.take_first_marker(&[PARAM_OPEN, FUNCTION_CLOSE])
                        .map(|(_, marker)| {
                            if marker == PARAM_OPEN {
                                SpanState::Key
                            } else {
                                SpanState::Done
                            }
                        })
                }
                SpanState::Key => self.take_marker(TAG_CLOSE).map(|before| {
                    self.key = before.trim().to_string();
                    self.open_value(schemas);
                    SpanState::Value
                }),
                SpanState::Value => self.take_marker(PARAM_CLOSE).map(|before| {
                    // The newline before `</parameter>` frames the value; it is
                    // not part of it.
                    let tail = before.strip_suffix('\n').unwrap_or(&before).to_string();
                    self.feed_value(&tail, out);
                    self.close_value(false, out);
                    SpanState::AwaitParam
                }),
                // The template writes one function per span. Anything after
                // `</function>` is its trailing newline, and a second call the
                // model crammed in here would need a second `ToolCallStart`
                // inside one span — so it is absorbed rather than guessed at.
                SpanState::Done => None,
            };
            match advanced {
                Some(state) => self.state = state,
                None => break,
            }
        }
        // A streaming string value is the one place where what is buffered has
        // to be committed before the next decode step: hold back only the part
        // that could still turn into the terminator being scanned for.
        if self.state == SpanState::Value && self.shape == ValueShape::Text {
            let ready = self.take_settled_value();
            self.feed_value(&ready, out);
        }
    }

    /// The buffered value text that can no longer become its terminator. What is
    /// left behind is either completed by the next decode step or dropped when
    /// the call is healed: a trailing `\n</para` is far likelier to be a
    /// terminator the model never finished than content that happens to end that
    /// way.
    ///
    /// Both `\n</parameter>` and a bare `</parameter>` count: the newline is
    /// framing the template always writes, but a model that skips it still means
    /// to end the value, and a holdback that only knew the framed form would leak
    /// half a terminator into the value when the two arrived in separate decode
    /// steps.
    fn take_settled_value(&mut self) -> String {
        let held = partial_suffix_len(&self.buffer, VALUE_CLOSE.len(), |tail| {
            VALUE_CLOSE.starts_with(tail) || PARAM_CLOSE.starts_with(tail)
        });
        let keep = self.buffer.len() - held;
        self.buffer.drain(..keep).collect()
    }

    /// The buffered text up to the next `marker`, with the marker consumed.
    fn take_marker(&mut self, marker: &str) -> Option<String> {
        self.take_first_marker(&[marker]).map(|(before, _)| before)
    }

    /// The same for whichever of `markers` occurs first, naming the one that did.
    fn take_first_marker<'m>(&mut self, markers: &[&'m str]) -> Option<(String, &'m str)> {
        let (at, marker) = markers
            .iter()
            .filter_map(|marker| self.buffer.find(marker).map(|at| (at, *marker)))
            .min_by_key(|(at, marker)| (*at, std::cmp::Reverse(marker.len())))?;
        let before = self.buffer[..at].to_string();
        self.buffer.drain(..at + marker.len());
        Some((before, marker))
    }

    /// Commit the name and open the arguments object, once the `>` closing
    /// `<function=` has proved the name complete.
    ///
    /// A call with no name at all starts nothing — it names nothing the client
    /// could dispatch, and reporting it would hand over a call that looks valid
    /// and is not. The span then degrades to text at `finish`.
    fn start(&mut self, out: &mut Vec<SpanOut>) {
        if self.started {
            return;
        }
        self.name = self.name.trim().to_string();
        if self.name.is_empty() {
            return;
        }
        self.started = true;
        out.push(SpanOut::Start(self.name.clone()));
        out.push(SpanOut::Delta("{".to_string()));
    }

    /// Begin a value. Nothing is committed here: `<parameter=key>` proves the key
    /// but not one character of its value, so a generation that stops right at
    /// this boundary must still be able to drop the pair whole.
    fn open_value(&mut self, schemas: &ToolSchemas) {
        self.shape = schemas.shape(&self.name, &self.key);
        self.value_lead = true;
        self.value_open = false;
        self.value.clear();
    }

    /// The `"key":` that opens the next pair, comma-separated from the last, and
    /// counted as emitted.
    fn pair_head(&mut self) -> String {
        let separator = if self.pairs == 0 { "" } else { "," };
        self.pairs += 1;
        format!("{separator}\"{}\":", escape_json(&self.key))
    }

    /// Feed value text. The newline after the key's `>` frames the value rather
    /// than belonging to it, so the first chunk gives it up.
    fn feed_value(&mut self, text: &str, out: &mut Vec<SpanOut>) {
        if text.is_empty() {
            return;
        }
        let text = if std::mem::take(&mut self.value_lead) {
            text.strip_prefix('\n').unwrap_or(text)
        } else {
            text
        };
        if text.is_empty() {
            return;
        }
        match self.shape {
            ValueShape::Text => {
                if !self.value_open {
                    self.value_open = true;
                    out.push(SpanOut::Delta(format!("{}\"", self.pair_head())));
                }
                out.push(SpanOut::Delta(escape_json(text)));
            }
            _ => self.value.push_str(text),
        }
    }

    /// Finish the value the parser is holding.
    ///
    /// On a normal close the model finished writing the value, so a held one is
    /// JSON if it parses as JSON and the string it literally is otherwise —
    /// which is what keeps the delta stream a valid object even when the model
    /// wrote something the schema did not promise.
    ///
    /// `healed` means generation stopped mid-value instead. Half of a structure
    /// is not a string the client can use, and calling it one would report an
    /// argument of a type the tool never declared, so the pair is dropped
    /// entirely — the same answer a key with no value at all gets. A shape that
    /// declares `string` keeps what did arrive: there the string is the
    /// truncation, not a retyping.
    fn close_value(&mut self, healed: bool, out: &mut Vec<SpanOut>) {
        if self.shape == ValueShape::Text {
            if self.value_open {
                out.push(SpanOut::Delta("\"".to_string()));
            } else if !healed {
                // The model wrote `<parameter=key>` and closed it without a
                // character between: an empty string, which is a value. Cut
                // short at the same place it is no value at all, and the pair
                // goes the way a key with no value goes.
                out.push(SpanOut::Delta(format!("{}\"\"", self.pair_head())));
            }
            self.value_open = false;
            return;
        }
        let held = std::mem::take(&mut self.value);
        // Cut short with not one character of the value written, the pair names
        // an argument the model had not decided on yet — whatever its shape.
        if healed && held.is_empty() {
            return;
        }
        let trimmed = held.trim();
        let parsed = serde_json::from_str::<serde_json::Value>(trimmed).ok();
        let usable = match self.shape {
            // The union's other members are what the model could have meant by
            // JSON here; anything else is the string the union also allows.
            ValueShape::Union(others) => parsed.filter(|value| others.contains(value)),
            _ => parsed,
        };
        let delta = match usable {
            Some(_) => trimmed.to_string(),
            None if healed && self.shape == ValueShape::Json => return,
            None => {
                // Evidence for the constrained-decoding question: the model was
                // asked for JSON, finished writing, and did not produce it. A
                // union that also declares `string` is excluded — resolving to
                // the string there is the schema being honored, not missed.
                if !healed && self.shape == ValueShape::Json {
                    self.quoted += 1;
                }
                format!("\"{}\"", escape_json(&held))
            }
        };
        let head = self.pair_head();
        out.push(SpanOut::Delta(format!("{head}{delta}")));
    }

    /// End the call. `closed` says the model wrote `</tool_call>`.
    ///
    /// A span that never got as far as a name goes back to the client as the raw
    /// text it is. The one case worth rescuing is generation stopping partway
    /// through the name itself: what arrived is only known to be a whole name if
    /// the request declared a tool by that spelling, and a `</tool_call>` that
    /// arrived before the name's own `>` is not a truncation but a malformed
    /// call, so it degrades too.
    ///
    /// A trailing key whose value never arrived is dropped: it names an argument
    /// the model had not decided on yet.
    fn finish(&mut self, schemas: &ToolSchemas, closed: bool, out: &mut Vec<SpanOut>) -> SpanEnd {
        if !self.started {
            // Nothing commits to `name` before the `>` that closes `<function=`,
            // so a name cut short by the end of generation is still in the
            // buffer.
            let name = match self.state {
                SpanState::Name if !closed => {
                    format!("{}{}", self.name, self.buffer).trim().to_string()
                }
                _ => String::new(),
            };
            if name.is_empty() || !schemas.declares(&name) {
                return SpanEnd::Degraded(self.raw.clone());
            }
            self.name = name;
            self.start(out);
        }
        if self.state == SpanState::Value {
            // Whatever the holdback still keeps is a terminator the model never
            // finished writing, not the tail of the value.
            let pending = self.take_settled_value();
            let pending = pending.strip_suffix('\n').unwrap_or(&pending).to_string();
            self.feed_value(&pending, out);
            self.close_value(!closed, out);
        }
        out.push(SpanOut::Delta("}".to_string()));
        out.push(SpanOut::End);
        // A call the model finished writing reaches `Done` and is closed by its
        // own `</tool_call>`. Missing either — a `</tool_call>` before
        // `</function>`, or a complete function block the model never terminated
        // — is a call this parser had to close on the model's behalf.
        SpanEnd::Delivered {
            repaired: !closed || self.state != SpanState::Done,
        }
    }
}

/// The body of the JSON string literal for `text` — the escaped form without its
/// quotes, so the fragments of one value concatenate into one string.
fn escape_json(text: &str) -> String {
    let quoted = serde_json::Value::String(text.to_string()).to_string();
    quoted[1..quoted.len() - 1].to_string()
}

/// Withholds answer text that could still grow into a stop sequence.
///
/// Text is forwarded the moment it can no longer be part of a match, so a sequence spread
/// across several decoded tokens is caught exactly as one inside a single token is, and
/// the matched text itself is never delivered.
struct StopFilter {
    /// Empty sequences are dropped: they would match at every position.
    sequences: Vec<String>,
    /// Length of the longest sequence, bounding how much text is ever held back.
    longest: usize,
    held: String,
}

impl StopFilter {
    fn new(sequences: &[String]) -> Self {
        let sequences: Vec<String> = sequences
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        let longest = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        Self {
            sequences,
            longest,
            held: String::new(),
        }
    }

    /// Feed one delta. Returns the text that is safe to deliver, plus the sequence that
    /// matched when one did — in which case the match and everything after it is dropped.
    fn push(&mut self, delta: &str) -> (String, Option<String>) {
        if self.sequences.is_empty() {
            return (delta.to_string(), None);
        }
        self.held.push_str(delta);
        if let Some((at, sequence)) = self.first_match() {
            let ready = self.held[..at].to_string();
            self.held.clear();
            return (ready, Some(sequence));
        }
        let keep = self.held.len() - self.partial_tail();
        let ready: String = self.held.drain(..keep).collect();
        (ready, None)
    }

    /// Everything still held back, for a generation that ended without a match.
    fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }

    /// Whether some sequence other than `excluded` starts with `prefix` — that
    /// is, whether the holdback was owed to more than the excluded sequence.
    fn starts_other(&self, excluded: &str, prefix: &str) -> bool {
        self.sequences
            .iter()
            .any(|sequence| sequence != excluded && sequence.starts_with(prefix))
    }

    /// The earliest complete match, preferring the longest sequence when several start at
    /// the same place.
    fn first_match(&self) -> Option<(usize, String)> {
        self.sequences
            .iter()
            .filter_map(|sequence| self.held.find(sequence.as_str()).map(|at| (at, sequence)))
            .min_by_key(|(at, sequence)| (*at, std::cmp::Reverse(sequence.len())))
            .map(|(at, sequence)| (at, sequence.clone()))
    }

    /// Length of the longest suffix of the held text that is a prefix of some sequence —
    /// the part that has to stay behind. A complete match is handled before this runs, so
    /// only proper prefixes are considered.
    fn partial_tail(&self) -> usize {
        partial_suffix_len(&self.held, self.longest, |tail| {
            self.sequences
                .iter()
                .any(|sequence| sequence.starts_with(tail))
        })
    }
}

/// Length of the longest suffix of `text` that could still grow into one of the
/// markers `is_prefix` recognizes — the part a streaming consumer has to hold
/// back. `longest` is the longest marker under consideration, which bounds the
/// holdback; only character boundaries are tried, so a codepoint is never split.
///
/// Shared by the stop-sequence filter and the tool-span parser: both deliver text
/// the moment it can no longer be part of a marker, and neither may cut a
/// multi-byte character in half to do it.
fn partial_suffix_len(text: &str, longest: usize, is_prefix: impl Fn(&str) -> bool) -> usize {
    let max = longest.saturating_sub(1).min(text.len());
    (1..=max)
        .rev()
        .filter(|k| text.is_char_boundary(text.len() - k))
        .find(|k| is_prefix(&text[text.len() - k..]))
        .unwrap_or(0)
}

/// How the next prompt lines up with the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Resume {
    /// Prefill straight on top of what the cache holds, starting at `pos`.
    Extend { pos: usize },
    /// Rewind to the snapshot taken at `pos` first, then prefill from there.
    Restore { pos: usize },
    /// Nothing reusable: reset the cache and prefill the whole prompt.
    Cold,
}

impl Resume {
    /// The position prefill resumes at, and equivalently the number of cached prompt
    /// tokens this request reuses.
    fn pos(&self) -> usize {
        match self {
            Resume::Extend { pos } | Resume::Restore { pos } => *pos,
            Resume::Cold => 0,
        }
    }
}

/// What the KV cache holds, and the turn boundaries it can rewind to.
///
/// The token history is the engine's only record of the cached context — the model stores
/// no ids — and it is always reconciled against `Generator::cache_len`, never against the
/// events a decode emitted. Generic over the snapshot payload so the reuse decision is
/// testable without a model.
struct PrefixCache<S> {
    tokens: Vec<u32>,
    /// Oldest first, positions strictly increasing.
    snapshots: VecDeque<(usize, S)>,
    /// The snapshot at the end of this conversation's leading system block, kept outside
    /// the capacity ring so the turn boundaries cannot evict it: it is the shallowest
    /// resume point and the only one another conversation from the same client can reach,
    /// which is exactly the thing that would go first if it aged with the rest.
    ///
    /// Pinned against eviction, not against invalidation — a truncation past it drops it
    /// like any other snapshot, since the full-attention rows it names are then someone
    /// else's.
    anchor: Option<(usize, S)>,
    capacity: usize,
}

impl<S> PrefixCache<S> {
    fn new(capacity: usize) -> Self {
        Self {
            tokens: Vec::new(),
            snapshots: VecDeque::new(),
            anchor: None,
            capacity,
        }
    }

    fn keeps_snapshots(&self) -> bool {
        self.capacity > 0
    }

    fn clear(&mut self) {
        self.tokens.clear();
        self.snapshots.clear();
        self.anchor = None;
    }

    fn set_tokens(&mut self, tokens: Vec<u32>) {
        self.tokens = tokens;
    }

    /// Shorten the cached context to `len`, for a slot handing back only the part of a
    /// conversation it can still prove it holds.
    fn truncate_tokens(&mut self, len: usize) {
        self.tokens.truncate(len);
    }

    /// Whether anything can be rewound to at all. A slot with no snapshot has no sliding-
    /// window state to restore, and so no position it can be resumed from.
    fn has_snapshots(&self) -> bool {
        !self.snapshots.is_empty() || self.anchor.is_some()
    }

    /// Every snapshot this cache holds, the anchor included. The order is the ring's,
    /// oldest first, with the anchor last: callers use it to count and to measure, never
    /// to pick a resume point (that is [`rewind_to`]).
    fn all_snapshots(&self) -> impl Iterator<Item = &(usize, S)> {
        self.snapshots.iter().chain(self.anchor.iter())
    }

    fn snapshot_at(&self, pos: usize) -> Option<&S> {
        self.all_snapshots()
            .find(|(p, _)| *p == pos)
            .map(|(_, snapshot)| snapshot)
    }

    /// Pin the snapshot taken where this conversation's leading system block ends, replacing
    /// any it already held.
    ///
    /// Separate from [`push`] because the anchor is chosen by position rather than by
    /// recency, and because it is installed from two directions: the prefill that first
    /// crosses it, and — once cache images outlive the process — a slot hydrated from a
    /// stored image, whose shallowest snapshot is the anchor it was written with.
    fn set_anchor(&mut self, pos: usize, snapshot: S) {
        self.anchor = Some((pos, snapshot));
    }

    /// Forget every snapshot past `pos`: they belong to a branch the engine has just
    /// abandoned, and the full-attention layers no longer hold the positions they would
    /// restore to.
    fn drop_after(&mut self, pos: usize) {
        self.snapshots.retain(|(p, _)| *p <= pos);
        if self.anchor.as_ref().is_some_and(|(p, _)| *p > pos) {
            self.anchor = None;
        }
    }

    /// Record a snapshot taken at `pos`, evicting the oldest once the cap is reached.
    fn push(&mut self, pos: usize, snapshot: S) {
        if self.capacity == 0 {
            return;
        }
        self.record(pos, snapshot, self.capacity);
    }

    /// Record the snapshot a page-out took at the cache's full length. Unlike [`push`] this
    /// ignores a zero cap: the rings it holds are the only thing that can restore this
    /// conversation's sliding-window state, so a slot keeping no turn-boundary snapshots
    /// still keeps this one, or it could never be paged back in at all.
    fn push_tail(&mut self, pos: usize, snapshot: S) {
        self.record(pos, snapshot, self.capacity.max(1));
    }

    /// Store a snapshot as the newest, replacing anything at or after its position and
    /// evicting the oldest until at most `capacity` remain.
    ///
    /// The anchor is replaced by this too when it sits at or past `pos`: the pin protects it
    /// from the capacity ring, not from a newer snapshot standing where it stands. Ordinarily
    /// the anchor is the shallowest position the cache holds and nothing here touches it.
    fn record(&mut self, pos: usize, snapshot: S, capacity: usize) {
        self.snapshots.retain(|(p, _)| *p < pos);
        if self.anchor.as_ref().is_some_and(|(p, _)| *p >= pos) {
            self.anchor = None;
        }
        self.snapshots.push_back((pos, snapshot));
        while self.snapshots.len() > capacity {
            self.snapshots.pop_front();
        }
    }

    /// The furthest position at or before `pos` the cache can be rewound to without
    /// replaying from zero: the newest turn boundary that qualifies, or the anchor when it
    /// reaches deeper — which is the case exactly when a conversation shares the system
    /// block and nothing beyond it.
    fn rewind_to(&self, pos: usize) -> Option<usize> {
        let boundary = self
            .snapshots
            .iter()
            .rev()
            .find(|(p, _)| *p <= pos)
            .map(|(p, _)| *p);
        let anchor = self
            .anchor
            .as_ref()
            .map(|(p, _)| *p)
            .filter(|anchor| *anchor <= pos);
        boundary.max(anchor)
    }

    /// How much of `prompt` the cache shares with what it holds, capped so the prefill that
    /// follows always has at least one token to run: that token produces the logits the
    /// decode starts from, so even a fully cached prompt replays its last one.
    fn shared_with(&self, prompt: &[u32]) -> Option<usize> {
        let reusable = prompt.len().checked_sub(1)?;
        Some(common_prefix_len(&self.tokens, prompt).min(reusable))
    }

    /// Decide how much of `prompt` the cache can keep. Only valid for the conversation
    /// whose state is in the model: a slot that has been paged out cannot be extended, and
    /// resumes at [`rewind_to`] instead.
    fn plan(&self, prompt: &[u32]) -> Resume {
        let Some(shared) = self.shared_with(prompt) else {
            return Resume::Cold;
        };
        // Extending is only legal at the cache's own end — the cache appends at its
        // length, so a shorter resume has to rewind to a snapshot instead.
        if shared == self.tokens.len() {
            return Resume::Extend { pos: shared };
        }
        match self.rewind_to(shared) {
            Some(pos) => Resume::Restore { pos },
            None => Resume::Cold,
        }
    }
}

/// One conversation's cached state as the engine remembers it: the token history and the
/// turn boundaries it can rewind to, plus — whenever the conversation is not the one in the
/// model's cache — the host image of the full-attention KV those tokens produced.
struct Slot<S, F, D> {
    prefix: PrefixCache<S>,
    /// The host image of this conversation's full-attention KV, absent only when this slot
    /// has never been paged out (a conversation that started here and has stayed) or has
    /// been emptied. A live slot keeps the image it was paged in with, so that a job which
    /// fails partway can hand the conversation back as cold-but-intact instead of losing it.
    ///
    /// For a cold slot the image is what makes every position the snapshots name resumable:
    /// rows `[0, tokens.len())` are the keys for `tokens`, and `tokens.len() <= image.pos`.
    full_kv: Option<F>,
    /// The host image of this conversation's drafter KV, when it had any to page out: a
    /// drafter with nothing committed (none attached, or one that fell behind at its own
    /// smaller context) leaves none, and the page-in resets the drafter instead.
    ///
    /// Kept whole rather than truncated with the history: the resume decision is made at
    /// import time instead, against the same `restore` position that bounds `full_kv` (see
    /// `draft_planes_cover`), which is simpler than keeping two truncation rules in step.
    /// So a demotion may leave planes reaching past what the slot now claims, and a slot may
    /// hold planes too short to use; both are read only through that decision.
    draft_kv: Option<D>,
    /// How far the retained image is known to agree with this slot's token history, while
    /// the slot is live. Rows below it are the keys for the tokens the history names; rows
    /// above it may not be, because the job in flight prefills its own tokens over those
    /// positions. It only ever decreases — a rewind lowers it, starting the conversation
    /// over drops the image and zeroes it — so it is the position a demotion may keep, and
    /// zero means the image can back nothing. Meaningless for a cold slot, whose whole
    /// history is backed by construction.
    image_agrees_to: usize,
    /// The manager's clock when this slot last served a job. Zero means never, which is
    /// also what an emptied slot is set back to, so it is the first one reused.
    last_used: u64,
}

/// Where a conversation with nothing to reuse goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FreshSlot {
    /// There is room under the cap for a slot of its own.
    New,
    /// The cap is reached, so this least recently used cold slot is overwritten.
    Evict(usize),
}

/// What an arriving prompt does to the slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotChoice {
    /// The prompt continues the conversation the model's cache already holds, on that
    /// slot's own three-way plan.
    Live { plan: Resume },
    /// Another slot shares more of the prompt: page the live conversation out to host
    /// memory and page that one in at `restore`, a snapshot position it holds.
    Swap { slot: usize, restore: usize },
    /// Nothing warm shares a usable prefix, so the prompt starts a conversation from zero
    /// in `slot`. The live conversation is paged out rather than dropped — it is likely to
    /// come back, which is the whole point of having slots.
    Fresh { slot: FreshSlot },
}

impl SlotChoice {
    /// The position prefill resumes at, and equivalently the number of cached prompt
    /// tokens this request reuses.
    fn pos(&self) -> usize {
        match self {
            SlotChoice::Live { plan } => plan.pos(),
            SlotChoice::Swap { restore, .. } => *restore,
            SlotChoice::Fresh { .. } => 0,
        }
    }
}

/// The warm conversations, exactly one of which occupies the model's KV cache.
///
/// Slots are matched to an arriving prompt by longest common prefix over their token
/// histories, and by nothing else: with no session ids in the matching, a conversation that
/// comes back through a different API dialect still finds its own cache, and a client that
/// invents a new id per request cannot defeat it.
///
/// Generic over the three payloads — the snapshot, the full-attention image and the
/// drafter image — so that the slot choice and the paging bookkeeping, which read none of
/// them, are testable without a model.
struct SlotManager<S, F, D> {
    /// Indexed by slot id, which names a slot for the life of the manager: slots are
    /// emptied and reused, never removed.
    slots: Vec<Slot<S, F, D>>,
    /// The slot whose state is in the model's cache, if any.
    live: Option<usize>,
    /// How many conversations may be warm at once, at least 1.
    max_slots: usize,
    /// Turn-boundary snapshots each slot keeps.
    snapshots: usize,
    /// Bumped once per job, so `last_used` orders the slots by recency.
    clock: u64,
}

/// The slots as the engine holds them: host-resident snapshots, full-attention images
/// and drafter images.
///
/// Every payload is behind an `Arc`. The records are immutable once built and a slot
/// replaces rather than mutates them, so a reference costs nothing to hand out and is
/// what lets the disk tier's writer thread serialize gigabytes without copying them
/// and without racing the slot they came from.
type Slots = SlotManager<Arc<HostSnapshot>, Arc<HostFullKv>, Arc<DrafterImage>>;

impl<S, F, D> SlotManager<S, F, D> {
    fn new(max_slots: usize, snapshots: usize) -> Self {
        Self {
            slots: Vec::new(),
            live: None,
            max_slots: max_slots.max(1),
            snapshots,
            clock: 0,
        }
    }

    /// Which slot serves `prompt`, and what has to happen to the model's cache first.
    ///
    /// Slots are compared on the position each could actually resume at, which for the live
    /// slot is its own plan and for a paged-out slot is the newest snapshot at or before the
    /// shared prefix. A tie goes to the live slot — its state is already in the cache, so it
    /// costs no paging — and then to the slot used most recently.
    fn choose(&self, prompt: &[u32]) -> SlotChoice {
        match self.best_slot(prompt) {
            // Resuming at zero is no better than starting over, and paging a slot in to
            // reuse nothing is pure cost.
            Some((slot, resume)) if resume > 0 => {
                if Some(slot) == self.live {
                    SlotChoice::Live {
                        plan: self.slots[slot].prefix.plan(prompt),
                    }
                } else {
                    SlotChoice::Swap {
                        slot,
                        restore: resume,
                    }
                }
            }
            _ => match self.fresh_slot() {
                Some(slot) => SlotChoice::Fresh { slot },
                // A cap of one slot, already live: that conversation is overwritten, which
                // is the single-sequence behaviour `cache_slots = 1` asks for.
                None => SlotChoice::Live { plan: Resume::Cold },
            },
        }
    }

    /// The slot that can serve the most of `prompt`, as `(slot, resume position)`.
    fn best_slot(&self, prompt: &[u32]) -> Option<(usize, usize)> {
        self.slots
            .iter()
            .enumerate()
            .map(|(slot, held)| {
                let resume = if Some(slot) == self.live {
                    held.prefix.plan(prompt).pos()
                } else {
                    // A paged-out slot cannot be extended at its end: the rings that go
                    // with the end of it live in a snapshot, so a snapshot is where it
                    // resumes. The page-out took one at its full length, so a conversation
                    // arriving exactly where it left off still resumes there.
                    //
                    // Its full-attention rows have to come from somewhere too: without an
                    // image the snapshots name positions nothing can restore, so such a slot
                    // offers no resume point however much of the prompt it shares.
                    held.full_kv
                        .as_ref()
                        .and_then(|_| held.prefix.shared_with(prompt))
                        .and_then(|shared| held.prefix.rewind_to(shared))
                        .unwrap_or(0)
                };
                (slot, resume)
            })
            .max_by_key(|&(slot, resume)| {
                (resume, Some(slot) == self.live, self.slots[slot].last_used)
            })
    }

    /// The slot a conversation with nothing to reuse starts in, or `None` when the cap
    /// leaves only the live slot — the live slot is never evicted, so the caller recycles
    /// the conversation in it instead.
    fn fresh_slot(&self) -> Option<FreshSlot> {
        if self.slots.len() < self.max_slots {
            return Some(FreshSlot::New);
        }
        self.slots
            .iter()
            .enumerate()
            .filter(|(slot, _)| Some(*slot) != self.live)
            .min_by_key(|(_, held)| held.last_used)
            .map(|(slot, _)| FreshSlot::Evict(slot))
    }

    /// Begin a conversation that shares nothing with any warm one. The slot it lands in
    /// becomes live, holding an empty history the caller is about to prefill into.
    fn start_fresh(&mut self, target: FreshSlot) {
        let fresh = Slot {
            prefix: PrefixCache::new(self.snapshots),
            full_kv: None,
            draft_kv: None,
            image_agrees_to: 0,
            last_used: self.clock,
        };
        let slot = match target {
            FreshSlot::New => {
                self.slots.push(fresh);
                self.slots.len() - 1
            }
            FreshSlot::Evict(slot) => {
                self.slots[slot] = fresh;
                slot
            }
        };
        self.live = Some(slot);
    }

    /// Put a conversation this process never held into `target`, as a cold slot: the
    /// state a page-out would have left, rebuilt from images that came from somewhere
    /// else. The slot does not become live — the caller pages it in, which is the one
    /// path that uploads rows and sets the agreement bound.
    ///
    /// `snapshots` need not be sorted, but the shallowest becomes the anchor and the
    /// deepest is stored as a page-out's tail snapshot: the first because the shallowest
    /// position is the one another conversation from the same client can reach, and so
    /// the one the capacity ring must never evict, and the second because a slot keeping
    /// no turn boundaries at all still has to keep the position it can be paged in at.
    fn install(
        &mut self,
        target: FreshSlot,
        tokens: Vec<u32>,
        image: F,
        snapshots: Vec<(usize, S)>,
        draft: Option<D>,
    ) -> usize {
        let mut prefix = PrefixCache::new(self.snapshots);
        prefix.set_tokens(tokens);
        let mut ordered = snapshots;
        ordered.sort_by_key(|(pos, _)| *pos);
        let mut ordered = ordered.into_iter();
        if let Some((pos, rings)) = ordered.next() {
            prefix.set_anchor(pos, rings);
        }
        let mut rest: Vec<(usize, S)> = ordered.collect();
        let tail = rest.pop();
        for (pos, rings) in rest {
            prefix.push(pos, rings);
        }
        if let Some((pos, rings)) = tail {
            prefix.push_tail(pos, rings);
        }
        let slot = Slot {
            prefix,
            full_kv: Some(image),
            draft_kv: draft,
            image_agrees_to: 0,
            last_used: self.clock,
        };
        match target {
            FreshSlot::New => {
                self.slots.push(slot);
                self.slots.len() - 1
            }
            FreshSlot::Evict(at) => {
                self.slots[at] = slot;
                at
            }
        }
    }

    /// Record the live conversation's host images, freeing the model's cache. `pos` is the
    /// cache length the image and the tail snapshot were taken at; `draft` is absent when
    /// the drafter held nothing to image.
    fn page_out(&mut self, pos: usize, rings: S, image: F, draft: Option<D>) {
        let Some(live) = self.live.take() else {
            return;
        };
        let slot = &mut self.slots[live];
        slot.prefix.push_tail(pos, rings);
        // The fresh image covers the whole history, superseding whatever this slot was paged
        // in with, so the agreement position it was tracking has nothing left to bound. The
        // drafter's planes are replaced the same way, absent included: what the drafter
        // holds now is the whole truth about it.
        slot.full_kv = Some(image);
        slot.draft_kv = draft;
        slot.image_agrees_to = 0;
    }

    /// Record that `slot`'s conversation is now in the model's cache, resumed at `restore`:
    /// the snapshots past the resume point describe a branch this turn is about to abandon.
    ///
    /// The image is deliberately kept. It stays the record of this conversation's rows
    /// `[0, restore)`, which is what lets a job that fails before reconciling hand the
    /// conversation back rather than lose it — a page-in is far too expensive to throw away
    /// over a client that hung up mid-prefill.
    fn page_in(&mut self, slot: usize, restore: usize) {
        self.slots[slot].prefix.drop_after(restore);
        self.slots[slot].image_agrees_to = restore;
        self.live = Some(slot);
    }

    /// How far the deepest warm slot agrees with `prompt`, whatever any of them could be
    /// resumed at.
    ///
    /// [`SlotManager::best_slot`] answers where a slot could be picked up, which its
    /// snapshots cap: a slot sharing twenty thousand tokens but holding no snapshot below
    /// them scores zero there and loses to a fresh slot. The divergence is real all the
    /// same, and it is where the next conversation sharing that prefix will want a
    /// snapshot — so it is asked for separately rather than inferred from who won.
    fn deepest_shared(&self, prompt: &[u32]) -> usize {
        self.slots
            .iter()
            .filter_map(|held| held.prefix.shared_with(prompt))
            .max()
            .unwrap_or(0)
    }

    /// How much of the live conversation a rewind to `pos` would destroy outright: the
    /// history it holds past the fork, less whatever a retained image already covers.
    ///
    /// A rewind keeps the slot and hands it to the arriving prompt, which reconciles its
    /// OWN history into it when it finishes — so everything the live conversation held
    /// above the fork is gone, from host RAM and from the disk writer alike, unless an
    /// image was made of it first. Below the fork nothing is lost: those tokens are the
    /// prefix the arriving prompt shares, and a later request for the old conversation
    /// still matches them.
    ///
    /// Zero when nothing is live, when the fork is at or past the end of what the slot
    /// holds (an extension destroys nothing), or when a retained image already reaches
    /// that far.
    fn live_history_at_risk(&self, pos: usize) -> usize {
        let Some(live) = self.live else {
            return 0;
        };
        let slot = &self.slots[live];
        // `image_agrees_to` only means anything while an image backs it.
        let imaged = if slot.full_kv.is_some() {
            slot.image_agrees_to
        } else {
            0
        };
        slot.prefix.tokens.len().saturating_sub(pos.max(imaged))
    }

    /// The ring entries a conversation forking off `source` at `pos` inherits.
    ///
    /// Read separately from [`SlotManager::fork_from`], and before it, because the
    /// page-out that has to happen in between mutates the ring it reads: a page-out
    /// records a tail snapshot, and recording one evicts the oldest entry of a full ring.
    /// The fork point is the newest snapshot at or below the shared prefix, which for a
    /// client coming back to an old branch of a long conversation is that oldest entry —
    /// so capturing afterwards would lose exactly the position the fork resumes at.
    fn fork_rings(&self, source: usize, pos: usize) -> Result<Vec<(usize, S)>>
    where
        S: Clone,
    {
        let held = self
            .slots
            .get(source)
            .ok_or_else(|| anyhow!("cache slot {source} does not exist"))?;
        let inherited: Vec<(usize, S)> = held
            .prefix
            .all_snapshots()
            .filter(|(at, _)| *at <= pos)
            .map(|(at, rings)| (*at, rings.clone()))
            .collect();
        ensure!(
            inherited.iter().any(|(at, _)| *at == pos),
            "cache slot {source} has no snapshot at {pos} for a fork to resume from"
        );
        Ok(inherited)
    }

    /// Give a conversation that forks off `source` at `pos` a slot of its own, sharing
    /// `source`'s image rather than copying it.
    ///
    /// Called after `source` has been imaged, so its image covers its whole history, and
    /// with the ring entries [`SlotManager::fork_rings`] captured beforehand. The payloads
    /// are `Arc`s at the engine's instantiation, so what looks like a copy of a gigabyte is
    /// a refcount — which is what makes keeping both conversations warm affordable at all.
    ///
    /// The caller pages the new slot in, exactly as it would any other cold slot.
    fn fork_from(
        &mut self,
        source: usize,
        target: FreshSlot,
        pos: usize,
        inherited: Vec<(usize, S)>,
    ) -> Result<usize>
    where
        F: Clone,
        D: Clone,
    {
        let held = self
            .slots
            .get(source)
            .ok_or_else(|| anyhow!("cache slot {source} does not exist"))?;
        let image = held
            .full_kv
            .clone()
            .ok_or_else(|| anyhow!("cache slot {source} was not imaged before the fork"))?;
        // The prefix the two conversations share. What the fork writes above it is its
        // own, and the decode reconciles the whole prompt into this slot when it lands.
        let tokens = held.prefix.tokens[..pos.min(held.prefix.tokens.len())].to_vec();
        let draft = held.draft_kv.clone();
        Ok(self.install(target, tokens, image, inherited, draft))
    }

    /// Rewind the live conversation to `pos`, dropping the snapshots of the branch it is
    /// leaving. The prefill that follows overwrites the positions above `pos` with this
    /// turn's tokens, so the retained image can no longer be trusted past there either.
    fn rewind_live(&mut self, pos: usize) -> Result<()> {
        let slot = self.live_slot()?;
        slot.prefix.drop_after(pos);
        slot.image_agrees_to = slot.image_agrees_to.min(pos);
        Ok(())
    }

    /// Start the live conversation over from nothing, keeping the slot live. Used when the
    /// prompt shares too little with what the slot held to rewind to, and when the cache
    /// turns out to disagree with the history.
    ///
    /// The retained images go too, and must: the tokens about to be prefilled from zero can
    /// diverge from the conversation they record at any position, so no part of either is
    /// evidence about them any more.
    fn restart_live(&mut self) -> Result<()> {
        let slot = self.live_slot()?;
        slot.prefix.clear();
        slot.full_kv = None;
        slot.draft_kv = None;
        slot.image_agrees_to = 0;
        Ok(())
    }

    /// Give up the live conversation, because a failed job has left the cache holding part of
    /// a prompt nothing has a token history for, or because the cache holds nothing at all.
    ///
    /// Returns the slot when it was EMPTIED rather than demoted — the caller uses that to
    /// forget where the conversation was stored on disk, since a slot holding nothing is no
    /// longer a reason to keep an image alive.
    ///
    /// A slot that kept an image is demoted to cold rather than emptied, so the conversation
    /// survives one client's failure. The demotion may only leave behind resume points the
    /// image can actually back, which is what `image_agrees_to` bounds: the history is
    /// truncated to it and every snapshot above it is dropped, because the job that just
    /// failed prefilled its own tokens over those positions and the image holds the previous
    /// conversation's keys there. A slot left with no snapshot at all cannot be paged back in,
    /// so it is emptied instead of kept as an image nothing can reach.
    ///
    /// The drafter's planes ride along with the demotion untruncated, bounded when they are
    /// imported (see [`Slot::draft_kv`]); an emptied slot drops them with everything else.
    fn abandon_live(&mut self) -> Option<usize> {
        let live = self.live.take()?;
        let slot = &mut self.slots[live];
        let agreed = slot.image_agrees_to;
        slot.image_agrees_to = 0;
        if slot.full_kv.is_some() && agreed > 0 {
            slot.prefix.truncate_tokens(agreed);
            slot.prefix.drop_after(agreed);
            if slot.prefix.has_snapshots() {
                return None;
            }
        }
        slot.prefix.clear();
        slot.full_kv = None;
        slot.draft_kv = None;
        slot.last_used = 0;
        Some(live)
    }

    /// The slot whose state the model's cache holds. Every path through the job dispatch
    /// establishes one; a caller that finds none has lost track of the cache, and reporting
    /// that costs one request rather than the model and every warm slot with it.
    fn live_slot(&mut self) -> Result<&mut Slot<S, F, D>> {
        let live = self
            .live
            .ok_or_else(|| anyhow!("no conversation is in the model's KV cache"))?;
        Ok(&mut self.slots[live])
    }

    /// The slots as a consumer sees them: what each holds, and what it costs in host
    /// RAM.
    ///
    /// The payloads are measured by the caller's own byte-length functions rather than
    /// by a bound on the type parameters, which is what keeps the manager ignorant of
    /// all three — the property the bookkeeping tests rely on to run without a model.
    fn summary(
        &self,
        snapshot_bytes: impl Fn(&S) -> usize,
        full_bytes: impl Fn(&F) -> usize,
        draft_bytes: impl Fn(&D) -> usize,
    ) -> Vec<SlotSummary> {
        self.slots
            .iter()
            .enumerate()
            .map(|(slot, held)| SlotSummary {
                live: Some(slot) == self.live,
                tokens: held.prefix.tokens.len(),
                snapshots: held.prefix.all_snapshots().count(),
                image_bytes: held
                    .prefix
                    .all_snapshots()
                    .map(|(_, snapshot)| snapshot_bytes(snapshot))
                    .sum::<usize>()
                    + held.full_kv.as_ref().map_or(0, |image| full_bytes(image))
                    + held
                        .draft_kv
                        .as_ref()
                        .map_or(0, |planes| draft_bytes(planes)),
                has_drafter: held.draft_kv.is_some(),
                last_used: held.last_used,
                agrees_to: held.image_agrees_to,
            })
            .collect()
    }

    /// Note that the live conversation is serving a job, which is what makes it the most
    /// recently used.
    fn touch_live(&mut self) -> Result<()> {
        self.clock += 1;
        let clock = self.clock;
        self.live_slot()?.last_used = clock;
        Ok(())
    }
}

/// Report the slots after something changed them, measuring the payloads the engine
/// actually holds. Every slot mutation is followed by one of these, so a consumer never
/// has to reach into engine state it must not touch.
fn log_slots(slots: &Slots, logger: &ServeLogger) {
    logger.log(ServeLog::SlotsSnapshot(slots.summary(
        |rings| rings.byte_len(),
        |image| image.byte_len(),
        |planes| planes.byte_len(),
    )));
}

/// Whether drafter planes of `kind` covering `covered` positions can back a conversation
/// resuming at `restore`, in a process where `attached` says whether a drafter exists.
///
/// All-or-nothing rather than a bound, because a partial cover is worth nothing: the
/// prefill feeds the drafter only while its committed length equals the position it
/// resumes at, so planes that stop short would be uploaded — up to a gigabyte of them —
/// and then ignored for the rest of the conversation. Stopping short is routine rather
/// than exceptional, since the drafter stops taking rows at its own smaller context: any
/// conversation longer than `draft_ctx` images fewer positions than the target holds.
///
/// The two kinds differ in what "enough" means, and that difference is why this takes a
/// kind at all. A DFlash image is usable whenever it REACHES the resume point: each of its
/// rows is a function of that position's taps alone, so a prefix of a longer image is a
/// valid shorter drafter. An MTP image is usable only at EXACTLY the position it ends at,
/// because it carries one carry hidden — the one its own last position produced — and the
/// head's next row is built from the hidden before it. Handing an over-long MTP image to a
/// shorter resume is a refusal rather than a partial upload, and asking here rather than
/// discovering it inside `import_cache` is what keeps that routine case a quiet skip
/// instead of a logged disk-tier failure.
///
/// `attached` is what makes a stored image portable across configurations, and it carries
/// the KIND rather than a yes/no for the same reason the stored record does. Planes written
/// by a server running with `--draft` can arrive at one running without, and the answer
/// there is to leave them alone and decode plain — never to fail the request over a
/// speculation nobody asked for. Planes written by a server running the OTHER kind can
/// arrive too, and that is not hypothetical: `--draft <path>` accepts either kind against
/// any checkpoint, so serving 3.8 with the 3.6 DFlash sidecar writes DFlash records against
/// a checkpoint whose official drafter is an MTP head, and a restart without the flag finds
/// them. Same answer, and for the same reason: a mismatch here is a quiet skip, where
/// letting `import_cache` refuse it would log a disk-tier failure on a configuration change
/// nobody got wrong.
fn drafter_planes_usable(
    attached: Option<DrafterKind>,
    stored: DrafterImageKind,
    restore: usize,
    covered: usize,
) -> bool {
    let Some(attached) = attached else {
        return false;
    };
    if attached.image_kind() != stored {
        return false;
    }
    match stored {
        DrafterImageKind::Dflash => covered >= restore,
        DrafterImageKind::Mtp => covered == restore,
    }
}

fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(capacity: usize) -> PrefixCache<usize> {
        PrefixCache::new(capacity)
    }

    /// One finished job, as the metrics history would take it.
    fn served_record() -> JobRecord {
        JobRecord {
            origin: RequestOrigin {
                id: 3,
                dialect: crate::serve::types::Dialect::Anthropic,
                streaming: true,
                client: Some("user_1a2b_session_9f2c".to_string()),
                session: Some("9f2ca1b4-0d31".to_string()),
                agent: Some("explore-metrics".to_string()),
            },
            model: "Qwen3.6-35B-A3B".to_string(),
            stop: Some(StopKind::EndTurn),
            abandoned: None,
            error: None,
            prompt_tokens: 2048,
            cache_read: 380,
            prefill_tokens: 1668,
            prefill_secs: 5.2,
            output_tokens: 38,
            thinking_tokens: 12,
            decode_secs: 3.1,
            ttft_secs: Some(6.6),
            spec: None,
            batch: None,
        }
    }

    /// A generation is recorded under the dialect that asked for it, carrying
    /// whoever asked.
    #[test]
    fn a_served_generation_records_its_dialect_and_its_client() {
        let run = run_record(&served_record(), false);
        assert_eq!(run.surface, "serve:anthropic");
        assert_eq!(run.model, "Qwen3.6-35B-A3B");
        assert_eq!(run.client.as_deref(), Some("user_1a2b_session_9f2c"));
        assert_eq!(run.session.as_deref(), Some("9f2ca1b4-0d31"));
        assert_eq!(run.agent.as_deref(), Some("explore-metrics"));
        assert_eq!(run.prompt_tokens, 2048);
        assert_eq!(run.cached_tokens, 380);
        assert_eq!(run.prefill_tokens, 1668);
        assert_eq!(run.decode_tokens, 38);
        assert_eq!(run.thinking_tokens, Some(12));
        assert!(run.ok);
    }

    /// A batch is submitted on the native dialect but costs nothing like a
    /// native generation, so it gets a surface of its own. Its phase figures
    /// come from the summary, where the tokens and the seconds were measured
    /// together; the record's own generation spans are zero for a batch.
    ///
    /// The token counts stay the record's, which the fold already corrected
    /// against the items: eight items over a 1000-token shared prefix prefill
    /// the prefix ONCE and read it back seven times, so 7000 of the 12000
    /// prompt tokens are cache reads.
    #[test]
    fn a_served_batch_records_its_own_surface_and_measured_phases() {
        let record = JobRecord {
            batch: Some(crate::serve::log::BatchSummary {
                items: 8,
                failed: 0,
                secs: 217.0,
                // The shared prefix once (1000) plus each item's 500-token tail.
                prefill_tokens: 5_000,
                prefill_secs: 20.0,
                decode_tokens: 1_200,
                decode_secs: 10.0,
            }),
            origin: RequestOrigin {
                dialect: crate::serve::types::Dialect::Native,
                client: None,
                session: None,
                agent: None,
                ..served_record().origin
            },
            // Eight items of 1000 shared + 500 own.
            prompt_tokens: 12_000,
            cache_read: 7_000,
            prefill_tokens: 5_000,
            ..served_record()
        };
        let run = run_record(&record, true);
        assert_eq!(run.surface, "serve:batch");
        assert_eq!(run.items, Some(8));
        assert_eq!(run.prompt_tokens, 12_000);
        assert_eq!(
            run.cached_tokens, 7_000,
            "the shared prefix read back once per item past the first"
        );
        assert_eq!(
            run.prefill_tokens, 5_000,
            "the shared prefix once, plus every item's own tail"
        );
        assert_eq!(
            run.prompt_tokens,
            run.cached_tokens + run.prefill_tokens,
            "a batch that ran to completion read its whole prompt"
        );
        assert_eq!(run.prefill_secs, 20.0);
        assert_eq!(run.decode_tokens, 1_200);
        assert_eq!(run.decode_secs, 10.0);
        assert_eq!(run.client, None);
    }

    /// A batch that failed before its runner returned has no summary, and the
    /// prompt count it was queued under is the queue's bytes-based estimate.
    /// It is still a batch — reporting it as the native generation it was
    /// submitted on would file real failures under the wrong surface — and its
    /// estimate is recorded as nothing rather than as a measurement.
    #[test]
    fn a_batch_that_failed_before_it_ran_reports_no_counts() {
        let record = JobRecord {
            batch: None,
            origin: RequestOrigin {
                dialect: crate::serve::types::Dialect::Native,
                ..served_record().origin
            },
            error: Some("the batch could not be rendered".to_string()),
            // What the queue estimated from the payload's bytes.
            prompt_tokens: 48_000,
            cache_read: 0,
            prefill_tokens: 0,
            output_tokens: 0,
            ..served_record()
        };
        let run = run_record(&record, true);
        assert_eq!(run.surface, "serve:batch");
        assert!(!run.ok);
        assert_eq!(run.prompt_tokens, 0, "an estimate is not a measurement");
        assert_eq!(run.cached_tokens, 0);
        assert_eq!(run.prefill_tokens, 0);
        assert_eq!(run.items, None);
    }

    /// A SCORED batch forwards more than its prompt: every teacher-forced trial
    /// runs through the model against no prompt token. The cache figure is
    /// summed from the items rather than taken as the difference, which would
    /// saturate to zero and hide a real cache hit.
    #[test]
    fn a_scored_batch_forwards_more_than_its_prompt() {
        let record = JobRecord {
            batch: Some(crate::serve::log::BatchSummary {
                items: 8,
                failed: 0,
                secs: 300.0,
                // Engine work: the prefix, the tails, and every scored trial.
                prefill_tokens: 40_000,
                prefill_secs: 100.0,
                decode_tokens: 0,
                decode_secs: 0.0,
            }),
            origin: RequestOrigin {
                dialect: crate::serve::types::Dialect::Native,
                ..served_record().origin
            },
            prompt_tokens: 12_000,
            cache_read: 7_000,
            prefill_tokens: 5_000,
            ..served_record()
        };
        let run = run_record(&record, true);
        assert_eq!(run.prompt_tokens, 12_000);
        assert_eq!(run.cached_tokens, 7_000, "the cache hit is unchanged");
        assert!(
            run.prefill_tokens > run.prompt_tokens,
            "scored work exceeds the prompt: {} vs {}",
            run.prefill_tokens,
            run.prompt_tokens
        );
        assert_eq!(run.prefill_tokens, 40_000);
    }

    /// Every item failing is a failed run. A batch that lost some of its items
    /// still did the rest, and its counts are real.
    #[test]
    fn a_batch_is_a_failure_only_when_every_item_failed() {
        let with_failures = |failed| JobRecord {
            batch: Some(crate::serve::log::BatchSummary {
                items: 8,
                failed,
                secs: 10.0,
                prefill_tokens: 5_000,
                prefill_secs: 20.0,
                decode_tokens: 1_200,
                decode_secs: 10.0,
            }),
            ..served_record()
        };
        assert!(run_record(&with_failures(0), true).ok);
        assert!(run_record(&with_failures(7), true).ok, "seven of eight ran");
        assert!(!run_record(&with_failures(8), true).ok);
    }

    /// A run that did not reach its own end is not ok, however healthy the
    /// engine was: a client that hung up, a deadline and a shutdown all stop
    /// the counts wherever they were, and a reader averaging over them
    /// alongside completed runs is averaging over two different things.
    #[test]
    fn an_abandoned_job_is_not_a_completed_run() {
        for reason in [
            CancelReason::ClientGone,
            CancelReason::Deadline,
            CancelReason::Shutdown,
        ] {
            let record = JobRecord {
                abandoned: Some(reason),
                ..served_record()
            };
            let run = run_record(&record, false);
            assert!(!run.ok, "{reason:?} is not a natural end");
            assert_eq!(
                run.decode_tokens, 38,
                "the tokens it did reach are still recorded"
            );
        }
        assert!(
            run_record(&served_record(), false).ok,
            "a job that stopped on its own is ok"
        );
    }

    /// A single native request keeps the native surface: only a batch moves.
    #[test]
    fn a_native_generation_is_not_a_batch() {
        let record = JobRecord {
            origin: RequestOrigin {
                dialect: crate::serve::types::Dialect::Native,
                ..served_record().origin
            },
            error: Some("the inference engine panicked".to_string()),
            ..served_record()
        };
        let run = run_record(&record, false);
        assert_eq!(run.surface, "serve:native");
        assert!(!run.ok, "a job that failed is not a run that succeeded");
        assert_eq!(
            run.prompt_tokens, 2_048,
            "a generation's prompt count is real at pickup, failure or not"
        );
    }

    /// A turn whose prompt extends the cached conversation prefills only the new tail.
    #[test]
    fn a_growing_conversation_extends_the_cache() {
        let mut prefix = cache(4);
        prefix.set_tokens(vec![1, 2, 3, 4]);
        prefix.push(3, 3);
        assert_eq!(prefix.plan(&[1, 2, 3, 4, 5, 6]), Resume::Extend { pos: 4 });
    }

    /// An empty cache is an extension from zero — there is nothing to reset or rewind.
    #[test]
    fn an_empty_cache_prefills_from_zero() {
        let prefix = cache(4);
        assert_eq!(prefix.plan(&[1, 2, 3]), Resume::Extend { pos: 0 });
        assert_eq!(prefix.plan(&[]), Resume::Cold);
    }

    /// A prompt the cache already holds in full still replays its last token, because the
    /// decode needs that position's logits. With a snapshot behind it, the replay starts
    /// there.
    #[test]
    fn a_fully_cached_prompt_rewinds_far_enough_to_produce_logits() {
        let mut prefix = cache(4);
        prefix.set_tokens(vec![1, 2, 3, 4]);
        prefix.push(2, 2);
        assert_eq!(prefix.plan(&[1, 2, 3, 4]), Resume::Restore { pos: 2 });

        // Without one, the whole prompt is replayed rather than extended at its end.
        let mut prefix = cache(4);
        prefix.set_tokens(vec![1, 2, 3, 4]);
        assert_eq!(prefix.plan(&[1, 2, 3, 4]), Resume::Cold);
    }

    /// A turn that edits the previous one diverges inside the cache and rewinds to the
    /// newest snapshot at or before the divergence.
    #[test]
    fn divergence_rewinds_to_the_newest_usable_snapshot() {
        let mut prefix = cache(4);
        prefix.set_tokens(vec![1, 2, 3, 4, 5, 6]);
        prefix.push(2, 2);
        prefix.push(4, 4);
        // Shared prefix is 5 tokens; both snapshots predate it, the newer one wins.
        assert_eq!(
            prefix.plan(&[1, 2, 3, 4, 5, 9, 9]),
            Resume::Restore { pos: 4 }
        );
        // Diverging before the newer snapshot falls back to the older one.
        assert_eq!(prefix.plan(&[1, 2, 3, 9, 9]), Resume::Restore { pos: 2 });
    }

    /// Divergence with every snapshot past the split point is a cold start.
    #[test]
    fn divergence_without_a_usable_snapshot_is_cold() {
        let mut prefix = cache(4);
        prefix.set_tokens(vec![1, 2, 3, 4, 5, 6]);
        prefix.push(4, 4);
        assert_eq!(prefix.plan(&[1, 2, 9, 9]), Resume::Cold);
    }

    /// A different conversation shares only its template header, which is never a
    /// snapshot boundary.
    #[test]
    fn a_conversation_switch_is_cold() {
        let mut prefix = cache(4);
        prefix.set_tokens(vec![1, 2, 3, 4, 5, 6]);
        prefix.push(5, 5);
        assert_eq!(prefix.plan(&[7, 8, 9]), Resume::Cold);
    }

    /// The deque is bounded, and the oldest snapshot is the one that goes.
    #[test]
    fn snapshots_evict_oldest_first() {
        let mut prefix = cache(2);
        prefix.push(10, 10);
        prefix.push(20, 20);
        prefix.push(30, 30);
        assert_eq!(prefix.snapshot_at(10), None);
        assert_eq!(prefix.snapshot_at(20), Some(&20));
        assert_eq!(prefix.snapshot_at(30), Some(&30));

        // A zero cap keeps none at all.
        let mut prefix = cache(0);
        assert!(!prefix.keeps_snapshots());
        prefix.push(10, 10);
        assert_eq!(prefix.snapshot_at(10), None);
    }

    /// Snapshots past a new boundary describe an abandoned branch and are dropped, both
    /// when a rewind happens and when the next boundary lands before them.
    #[test]
    fn abandoned_branches_lose_their_snapshots() {
        let mut prefix = cache(4);
        prefix.push(10, 10);
        prefix.push(20, 20);
        prefix.push(30, 30);

        prefix.drop_after(20);
        assert_eq!(prefix.snapshot_at(30), None);
        assert_eq!(prefix.snapshot_at(20), Some(&20));

        // Pushing at a position an existing snapshot already occupies replaces it.
        prefix.push(20, 21);
        assert_eq!(prefix.snapshot_at(20), Some(&21));
        assert_eq!(prefix.snapshot_at(10), Some(&10));
    }

    /// The anchor is pinned: the turn boundaries churn through the capacity ring without
    /// evicting it, which is the whole reason it lives outside the ring. It is the position a
    /// conversation sharing only the system block resumes at, and that conversation may
    /// arrive many turns later.
    #[test]
    fn the_anchor_outlives_the_turn_boundaries() {
        let mut prefix = cache(2);
        prefix.set_anchor(10, 10);
        for pos in [20, 30, 40, 50, 60] {
            prefix.push(pos, pos);
        }
        assert_eq!(prefix.snapshot_at(10), Some(&10));
        assert_eq!(prefix.rewind_to(15), Some(10));
        // The ring itself still holds only its two newest, so nothing about the cap changed.
        assert_eq!(prefix.snapshots.len(), 2);
        assert_eq!(prefix.snapshot_at(40), None);
    }

    /// The rewind point is the deepest position on offer, whichever kind of snapshot holds
    /// it: a turn boundary the prompt still agrees with beats the anchor, and the anchor is
    /// what is left when the prompt diverges before every boundary.
    #[test]
    fn a_rewind_takes_the_anchor_only_when_no_boundary_reaches_deeper() {
        let mut prefix = cache(4);
        prefix.set_anchor(10, 10);
        prefix.push(20, 20);
        prefix.push(30, 30);
        assert_eq!(prefix.rewind_to(35), Some(30));
        assert_eq!(prefix.rewind_to(25), Some(20));
        assert_eq!(prefix.rewind_to(15), Some(10));
        // Diverging inside the system block leaves nothing, as it must: the anchor names
        // positions this prompt does not share.
        assert_eq!(prefix.rewind_to(5), None);

        prefix.set_tokens(vec![1; 40]);
        assert_eq!(prefix.plan(&[1; 12]), Resume::Restore { pos: 10 });
    }

    /// The anchor is pinned against eviction, not against invalidation: a branch the engine
    /// abandons takes with it every snapshot past the truncation, the anchor included.
    #[test]
    fn an_abandoned_branch_drops_an_anchor_past_it() {
        let mut prefix = cache(4);
        prefix.set_anchor(20, 20);
        prefix.push(30, 30);

        prefix.drop_after(20);
        assert_eq!(prefix.snapshot_at(20), Some(&20));
        assert_eq!(prefix.snapshot_at(30), None);

        prefix.drop_after(15);
        assert_eq!(prefix.snapshot_at(20), None);
        assert!(!prefix.has_snapshots());

        // Starting the conversation over drops it with everything else.
        prefix.set_anchor(20, 20);
        prefix.clear();
        assert!(!prefix.has_snapshots());
    }

    /// A snapshot recorded where the anchor stands replaces it rather than doubling the rings
    /// held for one position — a page-out at exactly the anchor is the case that happens.
    #[test]
    fn a_snapshot_at_the_anchors_position_supersedes_it() {
        let mut prefix = cache(4);
        prefix.set_anchor(20, 20);
        prefix.push_tail(20, 21);
        assert_eq!(prefix.snapshot_at(20), Some(&21));
        assert!(prefix.anchor.is_none());
    }

    /// The one snapshot predicate: deep enough to pay for itself, inside the span the
    /// prefill about to run will actually produce, and above what could be resumed
    /// without it.
    #[test]
    fn a_snapshot_is_taken_only_when_it_pays_for_itself() {
        let deep = SNAPSHOT_MIN_GAIN * 2;
        assert!(snapshot_worth_taking(deep, 0, deep + 100, 0));
        // A position at or below the resume point is cached already, and cannot be
        // snapshotted without replaying it.
        assert!(!snapshot_worth_taking(deep, deep, deep + 100, 0));
        assert!(!snapshot_worth_taking(deep, deep + 1, deep + 100, 0));
        // Too shallow to be worth a ring copy.
        assert!(!snapshot_worth_taking(SNAPSHOT_MIN_GAIN - 1, 0, 4096, 0));
        // At the ceiling it is not a position this prefill produces; past it, less so.
        assert!(!snapshot_worth_taking(deep, 0, deep, 0));
        assert!(!snapshot_worth_taking(deep, 0, deep - 1, 0));
        // What is already resumable is what the saving is measured against: a snapshot
        // that only reaches a few hundred tokens past one that exists buys those few
        // hundred tokens, whatever its absolute position.
        assert!(!snapshot_worth_taking(deep, 0, deep + 100, deep - 100));
        assert!(snapshot_worth_taking(
            deep,
            0,
            deep + 100,
            deep - SNAPSHOT_MIN_GAIN
        ));
    }

    /// The stops one prefill pauses at: the anchor, the turn boundary, and the point where
    /// the prompt forked off the conversation it matched — sorted, distinct, and each one
    /// worth its ring copy.
    ///
    /// The order is the load-bearing part: `record` drops every snapshot at or past the
    /// position it stores, so stops taken out of order would delete the deeper resume
    /// points the earlier ones just paid for.
    #[test]
    fn the_prefill_stops_are_planned_ascending_and_deduplicated() {
        let gain = SNAPSHOT_MIN_GAIN;
        // A conversation whose history the arriving prompt shares up to a fork well past
        // the anchor: system block, then a first turn, then a divergence.
        let mut prefix = cache(4);
        let shared: Vec<u32> = (0..(3 * gain) as u32).collect();
        let mut held = shared.clone();
        held.extend([7, 7, 7]);
        prefix.set_tokens(held);
        let mut prompt = shared.clone();
        prompt.extend([9, 9, 9]);

        let anchor = gain + 100;
        let boundary = prompt.len() - 2;
        let stops = plan_snapshot_stops(&prefix, &prompt, 0, boundary, Some(anchor), 0);
        assert_eq!(
            stops,
            vec![
                SnapshotStop {
                    at: anchor,
                    reason: SnapshotReason::Anchor
                },
                SnapshotStop {
                    at: 3 * gain,
                    reason: SnapshotReason::Branch
                },
                SnapshotStop {
                    at: boundary,
                    reason: SnapshotReason::Turn
                },
            ],
            "ascending, and the fork sits between the anchor and this turn's boundary"
        );

        // A fork the anchor almost reaches saves almost nothing, so it is not taken: the
        // gain is measured from the deepest position a later fork could already resume at,
        // including the stops planned above it.
        let close = plan_snapshot_stops(&prefix, &prompt, 0, boundary, Some(3 * gain - 100), 0);
        assert_eq!(
            close.iter().map(|stop| stop.reason).collect::<Vec<_>>(),
            vec![SnapshotReason::Anchor, SnapshotReason::Turn],
            "{close:?}"
        );

        // A snapshot the slot already holds just below the fork does the same.
        let mut held_snapshot = cache(4);
        held_snapshot.set_tokens(prefix.tokens.clone());
        held_snapshot.push(3 * gain - 100, 0);
        let stops = plan_snapshot_stops(&held_snapshot, &prompt, 0, boundary, None, 0);
        assert_eq!(
            stops.iter().map(|stop| stop.reason).collect::<Vec<_>>(),
            vec![SnapshotReason::Turn]
        );

        // A fork that lands exactly on the turn boundary yields that boundary's snapshot
        // and no second one at the same position: the boundary is already what a fork
        // there would resume from, so the branch point saves nothing.
        let mut at_boundary = cache(4);
        let mut held = prompt[..boundary].to_vec();
        held.extend([7, 7]);
        at_boundary.set_tokens(held);
        let stops = plan_snapshot_stops(&at_boundary, &prompt, 0, boundary, None, 0);
        assert_eq!(
            stops,
            vec![SnapshotStop {
                at: boundary,
                reason: SnapshotReason::Turn
            }]
        );

        // A fork landing exactly on the anchor is the same story as one landing on the turn
        // boundary: the anchor is what a fork there would resume from, so a second stop at
        // that position would be a ring copy for nothing — and, taken after the anchor,
        // `record` would evict the anchor it duplicates.
        let mut at_anchor = cache(4);
        let mut held = prompt[..2 * gain].to_vec();
        held.extend([7, 7]);
        at_anchor.set_tokens(held);
        let stops = plan_snapshot_stops(&at_anchor, &prompt, 0, boundary, Some(2 * gain), 0);
        assert_eq!(
            stops,
            vec![
                SnapshotStop {
                    at: 2 * gain,
                    reason: SnapshotReason::Anchor
                },
                SnapshotStop {
                    at: boundary,
                    reason: SnapshotReason::Turn
                },
            ],
            "the anchor stands in for the fork at its own position"
        );

        // A conversation extending itself forked nowhere — its shared prefix IS the resume
        // point — so the only stop is the new turn's boundary.
        let mut extending = cache(4);
        extending.set_tokens(shared.clone());
        let stops = plan_snapshot_stops(&extending, &prompt, shared.len(), boundary, None, 0);
        assert_eq!(
            stops,
            vec![SnapshotStop {
                at: boundary,
                reason: SnapshotReason::Turn
            }]
        );

        // And when the shared context reaches past this turn's boundary as well — a retry
        // of a turn already cached — there is nothing left to stop for: the whole prompt
        // but its tail is behind the resume point.
        let stops = plan_snapshot_stops(
            &extending,
            &prompt,
            shared.len(),
            shared.len() - 72,
            None,
            0,
        );
        assert!(stops.is_empty(), "{stops:?}");

        // And a slot that keeps no snapshots plans none of them.
        let mut none = cache(0);
        none.set_tokens(prefix.tokens.clone());
        assert!(plan_snapshot_stops(&none, &prompt, 0, boundary, Some(anchor), 0).is_empty());
    }

    /// A whole multi-stop plan applied to a real cache, in the order and through the calls
    /// a prefill would use: every position the plan paid a ring copy for is still a resume
    /// point afterwards.
    ///
    /// The two halves of this are pinned separately — the planner sorts, and `record` drops
    /// what sits at or past it — but nothing checked the composition, which is where a
    /// regression would actually show up. The anchor lands pinned outside the capacity
    /// ring, the branch point and the turn boundary land in it.
    #[test]
    fn a_multi_stop_plan_leaves_every_snapshot_restorable() {
        let gain = SNAPSHOT_MIN_GAIN;
        let mut prefix = cache(4);
        let shared: Vec<u32> = (0..(3 * gain) as u32).collect();
        let mut held = shared.clone();
        held.extend([7, 7, 7]);
        prefix.set_tokens(held);
        let mut prompt = shared.clone();
        prompt.extend([9, 9, 9]);
        let anchor = gain + 100;
        let boundary = prompt.len() - 2;

        let stops = plan_snapshot_stops(&prefix, &prompt, 0, boundary, Some(anchor), 0);
        assert_eq!(stops.len(), 3, "anchor, fork and turn boundary: {stops:?}");
        for stop in &stops {
            match stop.reason {
                SnapshotReason::Anchor => prefix.set_anchor(stop.at, stop.at),
                SnapshotReason::Turn | SnapshotReason::Branch => prefix.push(stop.at, stop.at),
            }
        }

        for stop in &stops {
            assert_eq!(
                prefix.snapshot_at(stop.at),
                Some(&stop.at),
                "{stop:?} stopped the prefill and must still restore"
            );
        }
        // And each is the resume point a prompt sharing exactly that much would get.
        assert_eq!(prefix.rewind_to(anchor), Some(anchor));
        assert_eq!(prefix.rewind_to(3 * gain), Some(3 * gain));
        assert_eq!(prefix.rewind_to(boundary), Some(boundary));
        // The anchor is pinned rather than occupying the ring, which is what lets it
        // outlive the turn boundaries that keep arriving after it.
        assert_eq!(prefix.anchor.as_ref().map(|(pos, _)| *pos), Some(anchor));
        assert_eq!(
            prefix
                .snapshots
                .iter()
                .map(|(pos, _)| *pos)
                .collect::<Vec<_>>(),
            vec![3 * gain, boundary]
        );
    }

    /// A fork the DISPATCH could not use is still a fork worth snapshotting.
    ///
    /// The case real traffic produced: two agent sessions share a system prompt and its
    /// tool definitions, then diverge where their tool sets differ — below the anchor. The
    /// first session's slot holds that shared prefix but no snapshot beneath the
    /// divergence, so it can be resumed at nothing, loses the dispatch to a fresh slot,
    /// and is invisible to a divergence read off whichever slot won. Nothing then
    /// snapshots the fork, the writer splits a segment with no snapshot at its boundary,
    /// and every later session sharing that prefix prefills all of it from zero.
    ///
    /// So the branch point comes from the matching — the deepest agreement with anything
    /// the server holds — not from the winner.
    #[test]
    fn a_fork_matched_outside_the_winning_slot_still_plans_its_branch_stop() {
        let gain = SNAPSHOT_MIN_GAIN;
        let shared: Vec<u32> = (0..(20 * gain) as u32).collect();
        // The arriving session: the shared prefix, then its own tools and turn.
        let mut prompt = shared.clone();
        prompt.extend([9, 9, 9]);
        let boundary = prompt.len() - 2;

        // A fresh slot — the dispatch gave this conversation nothing, because the slot
        // that shares the prefix has no snapshot below the divergence to resume at.
        let fresh = cache(4);
        assert_eq!(
            fresh.shared_with(&prompt),
            Some(0),
            "the winning slot knows nothing about this prompt"
        );
        assert!(
            plan_snapshot_stops(&fresh, &prompt, 0, boundary, None, 0)
                .iter()
                .all(|stop| stop.reason != SnapshotReason::Branch),
            "which is exactly why deriving the fork from it plans nothing"
        );

        // With the divergence carried from the matching, the fork is planned — and its
        // position is where the histories actually part, not where anything resumed.
        let stops = plan_snapshot_stops(&fresh, &prompt, 0, boundary, None, shared.len());
        assert_eq!(
            stops,
            vec![
                SnapshotStop {
                    at: shared.len(),
                    reason: SnapshotReason::Branch
                },
                SnapshotStop {
                    at: boundary,
                    reason: SnapshotReason::Turn
                },
            ]
        );

        // A divergence past the end of the prompt is capped like any other, so the last
        // span still produces the logits the decode starts from.
        let stops = plan_snapshot_stops(&fresh, &prompt, 0, boundary, None, prompt.len() + 500);
        assert!(stops.iter().all(|stop| stop.at < prompt.len()), "{stops:?}");

        // And a prompt that shares nothing with anything still plans no fork: the
        // suppression is about there being no divergence, not about where it was found.
        assert!(
            plan_snapshot_stops(&fresh, &prompt, 0, boundary, None, 0)
                .iter()
                .all(|stop| stop.reason != SnapshotReason::Branch)
        );
    }

    /// The matching's answer is the deepest agreement across ALL warm slots, not the one
    /// that wins: `best_slot` caps each slot by the snapshots it can resume at, and the
    /// slot that shares the most may offer no resume point at all.
    #[test]
    fn the_deepest_shared_prefix_is_read_across_every_slot() {
        let gain = SNAPSHOT_MIN_GAIN;
        let mut slots = manager(4, 4);
        let deep: Vec<u32> = (0..(4 * gain) as u32).collect();

        // One conversation holding the prefix and carrying on past it, whose snapshots all
        // sit ABOVE the point the arriving prompt diverges at — the shape two agent
        // sessions make when they share a system block and differ in their tools.
        let _ = start(&mut slots);
        let mut held = deep.clone();
        held.extend((0..(2 * gain) as u32).map(|i| 500_000 + i));
        live(&mut slots).prefix.push(5 * gain, 5 * gain);
        live(&mut slots).prefix.set_tokens(held);
        page_out_of_cache(&mut slots);
        // Another, sharing nothing, which is the one a fresh prompt would land next to.
        let _ = start(&mut slots);
        live(&mut slots).prefix.set_tokens(vec![7, 7, 7]);

        let mut prompt = deep.clone();
        prompt.extend([9, 9]);
        // Nothing can resume this prompt: the deep slot kept no snapshot to rewind to.
        assert!(matches!(slots.choose(&prompt), SlotChoice::Fresh { .. }));
        // But the agreement is there to be found, and it is what the fork needs.
        assert_eq!(slots.deepest_shared(&prompt), deep.len());
        // Capped so a prefill always has a token left to run.
        assert_eq!(slots.deepest_shared(&deep), deep.len() - 1);
        assert_eq!(slots.deepest_shared(&[]), 0);
        assert_eq!(slots.deepest_shared(&[42, 43]), 0);
    }

    /// Why the stop order is a correctness property and not a preference: recording a
    /// snapshot drops every one at or past its position, because a shorter cache cannot
    /// back a deeper resume point. Stops taken out of order would throw away exactly the
    /// snapshots the prefill had just paid to take.
    #[test]
    fn recording_a_snapshot_drops_the_deeper_ones() {
        let mut prefix = cache(4);
        prefix.push(1000, 1000);
        prefix.push(3000, 3000);
        assert_eq!(prefix.snapshot_at(3000), Some(&3000));

        // A stop out of order: 2000 after 3000.
        prefix.push(2000, 2000);
        assert_eq!(prefix.snapshot_at(2000), Some(&2000));
        assert_eq!(
            prefix.snapshot_at(3000),
            None,
            "the deeper snapshot is gone, which is why the planner sorts"
        );
        assert_eq!(prefix.snapshot_at(1000), Some(&1000), "shallower ones stay");
    }

    /// A rewound cache never plans an extension past what the model still holds.
    #[test]
    fn a_restore_is_planned_at_a_snapshot_the_cache_still_has() {
        let mut prefix = cache(4);
        prefix.set_tokens(vec![1, 2, 3, 4, 5, 6]);
        prefix.push(2, 2);
        prefix.push(4, 4);
        let plan = prefix.plan(&[1, 2, 3, 9]);
        assert_eq!(plan, Resume::Restore { pos: 2 });
        assert!(prefix.snapshot_at(plan.pos()).is_some());
    }

    /// The snapshot a page-out takes is what makes the slot resumable at all, so it is kept
    /// even by a slot configured to keep no turn-boundary snapshots.
    #[test]
    fn the_page_out_snapshot_survives_a_zero_snapshot_cap() {
        let mut prefix = cache(0);
        prefix.push(6, 6);
        assert_eq!(prefix.snapshot_at(6), None);
        prefix.push_tail(9, 9);
        assert_eq!(prefix.snapshot_at(9), Some(&9));
        // Still one at a time: the next page-out replaces it rather than accumulating.
        prefix.push_tail(12, 12);
        assert_eq!(prefix.snapshot_at(9), None);
        assert_eq!(prefix.snapshot_at(12), Some(&12));
    }

    /// Slots carrying stand-in payloads: the slot choice reads neither the snapshots nor
    /// the full-attention images, only the token histories and the positions they cover.
    /// The three payloads stood in for by the positions they cover, which is all the
    /// bookkeeping under test reads of them.
    type TestSlots = SlotManager<usize, usize, usize>;

    fn manager(max_slots: usize, snapshots: usize) -> TestSlots {
        SlotManager::new(max_slots, snapshots)
    }

    /// A paged-out slot holding `tokens`, with a turn-boundary snapshot at each of
    /// `boundaries` and — as every page-out leaves behind — a tail snapshot at its full
    /// length. Returns the slot it landed in.
    fn seed(slots: &mut TestSlots, tokens: &[u32], boundaries: &[usize]) -> usize {
        let slot = start(slots);
        for pos in boundaries {
            live(slots).prefix.push(*pos, *pos);
        }
        live(slots).prefix.set_tokens(tokens.to_vec());
        slots.page_out(tokens.len(), tokens.len(), tokens.len(), Some(tokens.len()));
        slot
    }

    /// Take a slot for a conversation that shares nothing with a warm one, as the dispatch
    /// does, and return the slot it landed in.
    fn start(slots: &mut TestSlots) -> usize {
        let target = slots.fresh_slot().expect("room for another slot");
        slots.start_fresh(target);
        slots.touch_live().expect("a fresh slot is live");
        slots.live.expect("a fresh slot is live")
    }

    fn live(slots: &mut TestSlots) -> &mut Slot<usize, usize, usize> {
        slots.live_slot().expect("a slot is live")
    }

    /// The page-out the engine performs before another slot takes the cache. The live
    /// conversation's cache length is what its own history says here: no model is involved.
    fn page_out_of_cache(slots: &mut TestSlots) {
        let Some(live) = slots.live else {
            return;
        };
        let pos = slots.slots[live].prefix.tokens.len();
        if pos == 0 {
            slots.abandon_live();
        } else {
            slots.page_out(pos, pos, pos, Some(pos));
        }
    }

    /// Replay one job's dispatch against the slots with the model stood in for: apply the
    /// choice, take the turn-boundary snapshot the two-span prefill takes, then reconcile
    /// the history a finished decode leaves. Returns the position prefill resumed at, which
    /// is the `cached_tokens` the client is told about.
    fn serve(slots: &mut TestSlots, prompt: &[u32], boundary: usize, reply: &[u32]) -> usize {
        serve_with_anchor(slots, prompt, None, boundary, reply)
    }

    /// [`serve`] for a prompt that carries an anchor position, which the dispatch pins as
    /// the system-block snapshot whenever the prefill about to run crosses it — the same
    /// predicate `run_job` applies, so the qualification is under test rather than restated.
    fn serve_with_anchor(
        slots: &mut TestSlots,
        prompt: &[u32],
        anchor: Option<usize>,
        boundary: usize,
        reply: &[u32],
    ) -> usize {
        let choice = slots.choose(prompt);
        let resume = choice.pos();
        // Read before the dispatch, as the engine does: a fresh conversation can
        // overwrite the slot that shared the most of this prompt.
        let shared_anywhere = slots.deepest_shared(prompt);
        match choice {
            SlotChoice::Live { plan } => match plan {
                Resume::Extend { .. } => {}
                // The dispatch's own rule: a fork that would destroy unimaged history
                // images the conversation it forks off and takes a slot of its own,
                // rather than rewinding over it.
                Resume::Restore { pos }
                    if slots.live_history_at_risk(pos) >= SNAPSHOT_MIN_GAIN
                        && slots.fresh_slot().is_some() =>
                {
                    let source = slots.live.expect("a slot is live");
                    let target = slots.fresh_slot().expect("room for the fork");
                    // Captured before the page-out, exactly as the dispatch does.
                    let inherited = slots
                        .fork_rings(source, pos)
                        .expect("the fork point is held");
                    page_out_of_cache(slots);
                    let slot = slots
                        .fork_from(source, target, pos, inherited)
                        .expect("the fork installs");
                    slots.page_in(slot, pos);
                }
                Resume::Restore { pos } => slots.rewind_live(pos).expect("a slot is live"),
                Resume::Cold => slots.restart_live().expect("a slot is live"),
            },
            SlotChoice::Swap { slot, restore } => {
                page_out_of_cache(slots);
                slots.page_in(slot, restore);
            }
            SlotChoice::Fresh { slot } => {
                page_out_of_cache(slots);
                slots.start_fresh(slot);
            }
        }
        slots.touch_live().expect("the dispatch leaves a slot live");
        // The planner the engine itself runs, so what these tests see the slots collect is
        // what a real prefill would have stopped for. The stand-in payload is the position.
        let stops = plan_snapshot_stops(
            &live(slots).prefix,
            prompt,
            resume,
            boundary,
            anchor,
            shared_anywhere,
        );
        for stop in stops {
            match stop.reason {
                SnapshotReason::Anchor => live(slots).prefix.set_anchor(stop.at, stop.at),
                SnapshotReason::Turn | SnapshotReason::Branch => {
                    live(slots).prefix.push(stop.at, stop.at)
                }
            }
        }
        let mut history = prompt.to_vec();
        history.extend_from_slice(reply);
        live(slots).prefix.set_tokens(history);
        resume
    }

    /// The generation header every turn ends with. It opens with an added token, which is
    /// what lets the prompt be split before it in the first place.
    const TEST_HEADER: [u32; 2] = [9001, 9002];

    /// The system prompt one client sends with every conversation it holds. Two
    /// conversations from the same client therefore share a real prefix, which is why a
    /// slot match cannot be judged on the shared length merely being nonzero.
    const TEST_SYSTEM: [u32; 3] = [8001, 8002, 8003];

    /// One client's side of an exchange: everything it has sent and been sent, which is
    /// exactly what its next request re-sends.
    struct Conversation {
        /// Tells this conversation's text apart from the other's.
        mark: u32,
        history: Vec<u32>,
        turn: u32,
    }

    impl Conversation {
        fn new(mark: u32) -> Self {
            Self {
                mark,
                history: TEST_SYSTEM.to_vec(),
                turn: 0,
            }
        }

        /// The next request's prompt, and the position the engine splits it at.
        fn next_turn(&mut self) -> (Vec<u32>, usize) {
            self.turn += 1;
            let mut prompt = self.history.clone();
            prompt.extend([self.mark, self.mark + self.turn, self.mark + self.turn]);
            let boundary = prompt.len();
            prompt.extend(TEST_HEADER);
            (prompt, boundary)
        }

        /// Accept what the engine committed, so the next turn re-sends it.
        fn commit(&mut self, prompt: &[u32], reply: &[u32]) {
            self.history = prompt.to_vec();
            self.history.extend_from_slice(reply);
        }
    }

    /// A slot rebuilt from a stored image is the slot a page-out would have left:
    /// same history, same resume points, same drafter planes, and after the page-in
    /// the same agreement bound — which is what lets hydration reuse the ordinary
    /// swap path instead of a second one of its own.
    #[test]
    fn a_slot_installed_from_a_stored_image_matches_one_paged_out_in_process() {
        let text: Vec<u32> = (1..=64u32).collect();
        let mut prompt = text.clone();
        prompt.push(9001);

        // The conversation as this process held it: served, snapshotted at two turn
        // boundaries, then imaged out to host RAM.
        let mut warm = manager(2, 4);
        let paged_out = seed(&mut warm, &text, &[16, 48]);

        // The same conversation as a file holds it: the token history, the rows, and
        // every snapshot position the page-out left behind.
        let mut hydrated = manager(2, 4);
        let target = hydrated.fresh_slot().expect("room for a slot");
        let installed = hydrated.install(
            target,
            text.clone(),
            text.len(),
            vec![(16, 16), (48, 48), (64, 64)],
            Some(text.len()),
        );

        // Both answer an arriving prompt identically, which is the decision the
        // dispatch is made on.
        assert_eq!(warm.choose(&prompt), hydrated.choose(&prompt));
        let SlotChoice::Swap { restore, .. } = warm.choose(&prompt) else {
            panic!("a cold slot holding the whole prompt is a swap");
        };
        assert_eq!(restore, 64);

        warm.page_in(paged_out, restore);
        warm.touch_live().expect("the page-in leaves a slot live");
        hydrated.page_in(installed, restore);
        hydrated
            .touch_live()
            .expect("the page-in leaves a slot live");

        let measured = |slots: &TestSlots, slot: usize| slots.summary(|s| *s, |f| *f, |d| *d)[slot];
        let warm_slot = measured(&warm, paged_out);
        let hydrated_slot = measured(&hydrated, installed);
        assert!(warm_slot.live && hydrated_slot.live);
        assert_eq!(warm_slot.tokens, hydrated_slot.tokens);
        assert_eq!(warm_slot.snapshots, hydrated_slot.snapshots);
        assert_eq!(warm_slot.image_bytes, hydrated_slot.image_bytes);
        assert_eq!(warm_slot.has_drafter, hydrated_slot.has_drafter);
        // The bound a demotion may keep, which the page-in sets from the resume
        // point on both.
        assert_eq!(warm_slot.agrees_to, hydrated_slot.agrees_to);
        assert_eq!(warm_slot.agrees_to, restore);
        // And the snapshot the upload reads is there in both.
        assert!(warm.slots[paged_out].prefix.snapshot_at(restore).is_some());
        assert!(
            hydrated.slots[installed]
                .prefix
                .snapshot_at(restore)
                .is_some()
        );
    }

    /// A stored image can hold more snapshots than the slot it lands in keeps. The
    /// shallowest is pinned as the anchor — it is the position another conversation
    /// from the same client reaches — and the deepest survives the ring, which is
    /// the position the arriving one resumes at.
    #[test]
    fn installing_more_snapshots_than_the_cap_keeps_the_anchor_and_the_deepest() {
        let mut slots = manager(2, 1);
        let target = slots.fresh_slot().expect("room for a slot");
        let slot = slots.install(
            target,
            (1..=64u32).collect(),
            64,
            vec![(8, 8), (24, 24), (48, 48), (64, 64)],
            None,
        );
        let prefix = &slots.slots[slot].prefix;
        assert_eq!(prefix.snapshot_at(8), Some(&8), "the anchor is pinned");
        assert_eq!(prefix.snapshot_at(64), Some(&64), "and the deepest is kept");
        assert_eq!(prefix.snapshot_at(24), None, "the ring holds one");
        assert_eq!(prefix.snapshot_at(48), None);
        // Which is what makes a conversation sharing only the system block resume
        // there rather than from zero.
        assert_eq!(prefix.rewind_to(20), Some(8));
    }

    /// A slot keeping no turn boundaries at all still keeps what a stored image can
    /// be paged in at: without it the file could only ever resume from zero, which
    /// is the same reason a page-out's tail snapshot ignores the cap.
    #[test]
    fn installing_into_a_slot_that_keeps_no_snapshots_keeps_the_resume_point() {
        let mut slots = manager(2, 0);
        let target = slots.fresh_slot().expect("room for a slot");
        let slot = slots.install(
            target,
            (1..=64u32).collect(),
            64,
            vec![(16, 16), (64, 64)],
            None,
        );
        let prefix = &slots.slots[slot].prefix;
        assert_eq!(prefix.snapshot_at(64), Some(&64));
        assert_eq!(
            prefix.snapshot_at(16),
            Some(&16),
            "and the anchor, which is not the ring's to evict"
        );
    }

    /// The slot sharing the most of the prompt is the one that serves it.
    #[test]
    fn the_slot_sharing_the_most_of_the_prompt_wins() {
        let mut slots = manager(4, 4);
        let text = [1, 2, 3, 4, 5, 6];
        let a = seed(&mut slots, &[1, 2, 3, 4, 7, 8], &[2, 4]);
        let b = seed(&mut slots, &text, &[4]);

        // Nothing is live, so the shared length alone decides and each prompt finds the
        // conversation whose text it continues.
        assert_eq!(
            slots.choose(&[1, 2, 3, 4, 5, 6, 9]),
            SlotChoice::Swap {
                slot: b,
                restore: 6
            }
        );
        assert_eq!(
            slots.choose(&[1, 2, 3, 4, 7, 8, 9]),
            SlotChoice::Swap {
                slot: a,
                restore: 6
            }
        );

        // Sharing the most of the prompt is not enough on its own: what counts is how far
        // the slot can actually rewind, and here only `a` has a boundary before the split.
        assert_eq!(
            slots.choose(&[1, 2, 3, 9]),
            SlotChoice::Swap {
                slot: a,
                restore: 2
            }
        );

        // Two slots holding the same text is a tie the more recently used one wins.
        let c = seed(&mut slots, &text, &[4]);
        assert_eq!(
            slots.choose(&[1, 2, 3, 4, 5, 6, 9]),
            SlotChoice::Swap {
                slot: c,
                restore: 6
            }
        );

        // The live slot outranks an equal match: paging another in would reuse no more of
        // the prompt and would cost two transfers to do it.
        slots.page_in(b, 6);
        assert_eq!(
            slots.choose(&[1, 2, 3, 4, 5, 6, 9]),
            SlotChoice::Live {
                plan: Resume::Extend { pos: 6 }
            }
        );
    }

    /// Two clients talking in turn each keep their own slot, so a returning conversation
    /// resumes at everything it had cached instead of prefilling from zero. This is the
    /// failure a single-sequence cache has, where each switch evicts the other speaker.
    #[test]
    fn interleaved_conversations_keep_their_own_slots() {
        let mut slots = manager(4, 4);
        let mut a = Conversation::new(100);
        let mut b = Conversation::new(200);

        // The opening turn of each is cold: the only text they share is the system prompt,
        // which is no turn boundary and so nothing either can rewind to.
        for conv in [&mut a, &mut b] {
            let (prompt, boundary) = conv.next_turn();
            let reply = [conv.mark + 50];
            assert_eq!(serve(&mut slots, &prompt, boundary, &reply), 0);
            conv.commit(&prompt, &reply);
        }

        // From here each turn resumes at its own conversation's full cached length, so only
        // the new turn is prefilled.
        for _ in 0..3 {
            for conv in [&mut a, &mut b] {
                let (prompt, boundary) = conv.next_turn();
                let reply = [conv.mark + 50];
                let resumed = serve(&mut slots, &prompt, boundary, &reply);
                assert_eq!(
                    resumed,
                    conv.history.len(),
                    "conversation {} resumed short on turn {}",
                    conv.mark,
                    conv.turn
                );
                conv.commit(&prompt, &reply);
            }
        }
        assert_eq!(slots.slots.len(), 2, "two conversations, two slots");
    }

    /// One slot is the single-sequence cache this server had before slots existed: a
    /// conversation switch overwrites it, and the conversation that left comes back cold.
    #[test]
    fn a_single_slot_holds_one_conversation_at_a_time() {
        let mut slots = manager(1, 4);
        let mut a = Conversation::new(100);
        let mut b = Conversation::new(200);

        let (prompt, boundary) = a.next_turn();
        assert_eq!(serve(&mut slots, &prompt, boundary, &[150]), 0);
        a.commit(&prompt, &[150]);

        // A second turn of the same conversation extends what the slot holds.
        let (prompt, boundary) = a.next_turn();
        assert_eq!(
            serve(&mut slots, &prompt, boundary, &[150]),
            a.history.len()
        );
        a.commit(&prompt, &[150]);

        // The other conversation takes the slot over, and A's return is cold again.
        let (prompt, boundary) = b.next_turn();
        assert_eq!(serve(&mut slots, &prompt, boundary, &[250]), 0);
        b.commit(&prompt, &[250]);
        let (prompt, boundary) = a.next_turn();
        assert_eq!(serve(&mut slots, &prompt, boundary, &[150]), 0);
        assert_eq!(slots.slots.len(), 1);
    }

    /// The cap bounds how many conversations stay warm, and the live one is never the one
    /// that goes: its state is the model's cache, and it is the likeliest to speak next.
    #[test]
    fn eviction_respects_the_cap_and_never_takes_the_live_slot() {
        let mut slots = manager(3, 4);
        // Under the cap, a conversation with nothing to reuse gets a slot of its own.
        assert_eq!(slots.fresh_slot(), Some(FreshSlot::New));
        let a = seed(&mut slots, &[1, 1, 1], &[]);
        let b = seed(&mut slots, &[2, 2, 2], &[]);
        assert_eq!(slots.fresh_slot(), Some(FreshSlot::New));
        let c = seed(&mut slots, &[3, 3, 3], &[]);

        // At the cap it is the least recently used conversation that is overwritten.
        assert_eq!(slots.fresh_slot(), Some(FreshSlot::Evict(a)));
        let clock = slots.clock;
        slots.slots[a].last_used = clock + 1;
        assert_eq!(slots.fresh_slot(), Some(FreshSlot::Evict(b)));
        slots.slots[b].last_used = clock + 2;
        assert_eq!(slots.fresh_slot(), Some(FreshSlot::Evict(c)));

        // The live slot is spared however long it has been idle.
        slots.page_in(c, 3);
        slots.slots[c].last_used = 0;
        assert_eq!(slots.fresh_slot(), Some(FreshSlot::Evict(a)));

        // A cap of one, already live, leaves nothing to evict at all: the caller recycles
        // the live conversation, which is what a single slot means.
        let mut one = manager(1, 4);
        let only = seed(&mut one, &[1, 2, 3], &[]);
        one.page_in(only, 3);
        assert_eq!(one.fresh_slot(), None);
        assert_eq!(
            one.choose(&[9, 9, 9]),
            SlotChoice::Live { plan: Resume::Cold }
        );
    }

    /// A job that fails leaves the model's cache holding part of a prompt nothing has a token
    /// history for, so the conversation in it is given up. A conversation that started in its
    /// slot and has never been paged out has no host image to fall back on, so nothing of it
    /// survives — while every other slot was imaged before that job started and describes
    /// positions it never wrote, so those stay usable.
    #[test]
    fn a_failed_job_drops_only_the_live_slot() {
        let mut slots = manager(2, 4);
        let a = seed(&mut slots, &[1, 1, 1, 1], &[]);
        let b = start(&mut slots);
        live(&mut slots).prefix.set_tokens(vec![2, 2, 2, 2]);

        slots.abandon_live();
        assert_eq!(slots.live, None);
        assert_eq!(
            slots.choose(&[1, 1, 1, 1, 5]),
            SlotChoice::Swap {
                slot: a,
                restore: 4
            },
            "the paged-out conversation is untouched by another's failure"
        );

        // The dropped slot holds nothing, and is the first one reused.
        assert!(slots.slots[b].prefix.tokens.is_empty());
        assert!(slots.slots[b].full_kv.is_none());
        assert_eq!(slots.fresh_slot(), Some(FreshSlot::Evict(b)));
    }

    /// A job that fails after paging a conversation in hands it back as cold rather than
    /// losing it: the page-in moved its whole prefix into the cache, and a client that hung
    /// up mid-prefill is no reason to make the next visit prefill from zero again.
    ///
    /// What comes back is bounded by what the image can still back. The failed job prefilled
    /// its own tokens over the positions above the resume point, so the image holds the
    /// previous conversation's keys there — the snapshot that job took is a position nothing
    /// could restore, and must not survive.
    #[test]
    fn a_failure_after_a_page_in_demotes_to_the_resume_point() {
        let mut slots = manager(2, 4);
        let a = seed(&mut slots, &[1, 2, 3, 4, 5, 6], &[4]);
        slots.page_in(a, 4);
        // A turn's worth of prefill on top of the resume point, with its boundary snapshot,
        // and then the failure — before any of it was reconciled into the history.
        live(&mut slots).prefix.push(9, 9);
        live(&mut slots)
            .prefix
            .set_tokens(vec![1, 2, 3, 4, 7, 8, 9, 9, 9]);

        slots.abandon_live();
        assert_eq!(slots.live, None);
        assert!(slots.slots[a].full_kv.is_some(), "the image is kept");
        assert_eq!(slots.slots[a].prefix.tokens, vec![1, 2, 3, 4]);
        assert_eq!(slots.slots[a].prefix.snapshot_at(4), Some(&4));
        assert_eq!(
            slots.slots[a].prefix.snapshot_at(9),
            None,
            "the failed job's snapshot names positions the image cannot back"
        );
        // And the conversation is resumable, at the point it was paged in at.
        assert_eq!(
            slots.choose(&[1, 2, 3, 4, 5, 6, 7]),
            SlotChoice::Swap {
                slot: a,
                restore: 4
            }
        );
    }

    /// A rewind inside the live conversation lowers what its retained image can back: the
    /// prefill that follows writes this turn's tokens over the positions above the rewind
    /// point, so a later failure may hand back only what lies below it.
    #[test]
    fn a_rewind_lowers_what_a_demotion_may_keep() {
        let mut slots = manager(2, 4);
        let a = seed(&mut slots, &[1, 2, 3, 4, 5, 6], &[2, 4]);
        slots.page_in(a, 6);
        // The next turn edits its way back behind the point the page-in resumed at.
        slots.rewind_live(2).expect("the slot is live");
        live(&mut slots).prefix.set_tokens(vec![1, 2, 9, 9, 9]);

        slots.abandon_live();
        assert_eq!(slots.slots[a].prefix.tokens, vec![1, 2]);
        assert_eq!(slots.slots[a].prefix.snapshot_at(2), Some(&2));
        assert_eq!(slots.slots[a].prefix.snapshot_at(4), None);
        assert_eq!(slots.slots[a].prefix.snapshot_at(6), None);
    }

    /// A demotion may keep only resume points its image can back, and the anchor is judged
    /// by that same bound: one below it survives the failure and one above it does not.
    #[test]
    fn a_demotion_caps_the_anchor_like_any_snapshot() {
        let mut slots = manager(2, 4);
        let a = seed(&mut slots, &[1, 2, 3, 4, 5, 6], &[]);
        slots.page_in(a, 6);
        // The failed job's own prefill crossed the anchor, so the image holds the previous
        // conversation's keys there.
        live(&mut slots).prefix.set_anchor(8, 8);
        live(&mut slots)
            .prefix
            .set_tokens(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

        slots.abandon_live();
        assert_eq!(slots.slots[a].prefix.snapshot_at(8), None);

        // An anchor at or below the agreement point survives, and on its own is enough to
        // keep the slot: a demotion left with no resume point at all empties it instead.
        let b = seed(&mut slots, &[1, 2, 3, 4, 5, 6], &[]);
        slots.slots[b].prefix.set_anchor(4, 4);
        // Another conversation from the same client is paged in at the anchor, and its job
        // fails before anything it prefilled is reconciled.
        slots.page_in(b, 4);
        live(&mut slots).prefix.set_tokens(vec![1, 2, 3, 4, 9, 9]);

        slots.abandon_live();
        assert!(slots.slots[b].full_kv.is_some(), "the image is kept");
        assert_eq!(slots.slots[b].prefix.tokens, vec![1, 2, 3, 4]);
        assert_eq!(slots.slots[b].prefix.snapshot_at(4), Some(&4));
        assert_eq!(
            slots.choose(&[1, 2, 3, 4, 9, 9, 9]),
            SlotChoice::Swap {
                slot: b,
                restore: 4
            },
            "the anchor is the only thing this conversation can come back at"
        );
    }

    /// The point of the anchor: a conversation that shares only the system prompt with a warm
    /// one resumes where that block ends instead of prefilling from zero. Without the anchor
    /// the shared prefix is real but unreachable — every turn boundary sits past it.
    #[test]
    fn a_new_conversation_resumes_at_the_shared_system_block() {
        // Long enough to be worth a snapshot, which is what a harness's system prompt is.
        let system: Vec<u32> = (0..SNAPSHOT_MIN_GAIN as u32 + 200)
            .map(|i| 5000 + i)
            .collect();
        let anchor = system.len();
        let turn = |mark: u32| {
            let mut prompt = system.clone();
            prompt.extend([mark, mark + 1, mark + 2]);
            let boundary = prompt.len();
            prompt.extend(TEST_HEADER);
            (prompt, boundary)
        };
        let (first, first_boundary) = turn(100);
        let (second, second_boundary) = turn(200);
        // A conversation from somewhere else entirely, which takes the cache and leaves the
        // first one paged out — the shape a second conversation actually arrives in.
        let (other, other_boundary) = (vec![1, 2, 3, 9001, 9002], 3);

        let mut slots = manager(4, 4);
        assert_eq!(
            serve_with_anchor(&mut slots, &first, Some(anchor), first_boundary, &[77]),
            0,
            "the first conversation of all is cold"
        );
        let a = slots.live.expect("the first conversation is in the cache");
        serve(&mut slots, &other, other_boundary, &[77]);

        assert_eq!(
            slots.choose(&second),
            SlotChoice::Swap {
                slot: a,
                restore: anchor
            }
        );
        assert_eq!(
            serve_with_anchor(&mut slots, &second, Some(anchor), second_boundary, &[88]),
            anchor,
            "only the new turn is prefilled"
        );

        // Without the anchor the same two conversations share the same tokens and reuse
        // nothing: the shared prefix is not a position anything can restore.
        let mut slots = manager(4, 4);
        serve(&mut slots, &first, first_boundary, &[77]);
        serve(&mut slots, &other, other_boundary, &[77]);
        assert_eq!(serve(&mut slots, &second, second_boundary, &[88]), 0);
    }

    /// A prefill cut short by a cancelled job reconciles the live slot to exactly the
    /// tokens the cache holds, so the slot stays live — `dirty` cleared, no demotion —
    /// and an identical retry extends from there instead of prefilling from zero.
    #[test]
    fn a_cancelled_prefill_leaves_the_slot_resumable_at_what_was_written() {
        let mut slots = manager(2, 4);
        let _ = start(&mut slots);
        let prompt: Vec<u32> = (1..=10).collect();
        // The abort came 6 tokens in; the reconciliation records exactly those.
        live(&mut slots).prefix.set_tokens(prompt[..6].to_vec());
        assert_eq!(
            slots.choose(&prompt),
            SlotChoice::Live {
                plan: Resume::Extend { pos: 6 }
            }
        );
    }

    /// The same reconciliation after a page-in keeps the demotion bound honest: a
    /// cancelled prefill stops at or above the position the page-in resumed at, so the
    /// rows the retained image backs still match the truncated history, and a later
    /// failure may still hand the conversation back at that bound.
    #[test]
    fn a_cancelled_prefill_after_a_page_in_keeps_the_demotion_bound() {
        let mut slots = manager(2, 4);
        let a = seed(&mut slots, &[1, 2, 3, 4, 5, 6], &[4]);
        slots.page_in(a, 4);
        // The cancelled job had written 3 prompt tokens past the resume point; the
        // reconciliation records exactly those and leaves the slot live.
        live(&mut slots)
            .prefix
            .set_tokens(vec![1, 2, 3, 4, 7, 8, 9]);
        assert_eq!(slots.slots[a].image_agrees_to, 4);

        slots.abandon_live();
        assert_eq!(slots.slots[a].prefix.tokens, vec![1, 2, 3, 4]);
        assert!(slots.slots[a].full_kv.is_some());
    }

    /// Starting the live conversation over drops its retained image. The tokens about to be
    /// prefilled from zero can diverge from the conversation the image records at any
    /// position, so no part of it is evidence about them any more.
    #[test]
    fn starting_the_live_conversation_over_drops_its_image() {
        let mut slots = manager(2, 4);
        let a = seed(&mut slots, &[1, 2, 3, 4, 5, 6], &[4]);
        slots.page_in(a, 4);
        slots.restart_live().expect("the slot is live");
        assert!(slots.slots[a].full_kv.is_none());
        assert_eq!(slots.slots[a].image_agrees_to, 0);

        // A failure now has nothing to hand back, so the slot is emptied outright.
        live(&mut slots).prefix.set_tokens(vec![7, 8, 9]);
        slots.abandon_live();
        assert!(slots.slots[a].prefix.tokens.is_empty());
        assert_eq!(slots.slots[a].last_used, 0);
    }

    /// A slot with no image offers no resume point, however much of the prompt its token
    /// history shares: its snapshots name positions whose full-attention rows nothing holds.
    #[test]
    fn a_slot_without_an_image_is_never_paged_in() {
        let mut slots = manager(2, 4);
        let a = seed(&mut slots, &[1, 2, 3, 4], &[2]);
        slots.slots[a].full_kv = None;
        assert_eq!(
            slots.choose(&[1, 2, 3, 4, 5]),
            SlotChoice::Fresh {
                slot: FreshSlot::New
            },
            "the prompt starts over rather than restoring rows that are not there"
        );
    }

    /// Paging a conversation out images its drafter rows alongside its full-attention
    /// ones, so a conversation that comes back keeps speculating. Starting it over drops
    /// both: tokens prefilled from zero can diverge at any position, which makes neither
    /// image evidence about them.
    #[test]
    fn a_page_out_images_the_drafter_rows_and_a_restart_drops_them() {
        let mut slots = manager(2, 4);
        let a = seed(&mut slots, &[1, 2, 3, 4], &[2]);
        assert_eq!(slots.slots[a].draft_kv, Some(4));

        slots.page_in(a, 2);
        slots.restart_live().expect("the slot is live");
        assert_eq!(slots.slots[a].draft_kv, None);
        assert_eq!(slots.slots[a].full_kv, None);
    }

    /// A drafter with nothing committed — none attached, or one that fell behind at its
    /// own smaller context before this page-out — leaves no planes, and the slot says so
    /// rather than storing an empty image. Its conversation resumes with speculation off.
    #[test]
    fn a_page_out_without_drafter_rows_stores_no_planes() {
        let mut slots = manager(2, 4);
        let a = start(&mut slots);
        live(&mut slots).prefix.set_tokens(vec![1, 2, 3, 4]);
        slots.page_out(4, 4, 4, None);
        assert_eq!(slots.slots[a].draft_kv, None);
        assert!(
            slots.slots[a].full_kv.is_some(),
            "the conversation itself is still resumable"
        );
        assert_eq!(
            slots.choose(&[1, 2, 3, 4, 5]),
            SlotChoice::Swap {
                slot: a,
                restore: 4
            }
        );
    }

    /// A fork off the live conversation images it before taking its slot.
    ///
    /// Rewinding in place is what the plan alone would do, and it destroys the forked-from
    /// conversation: the arriving prompt reconciles its own history into that slot when it
    /// finishes, so everything above the fork is gone from host RAM without ever having
    /// been written — the disk writer is fed from page-outs, and this path had none. The
    /// live check found exactly this, with three slots sitting empty at the time.
    #[test]
    fn a_fork_images_the_conversation_it_forks_off_before_taking_its_slot() {
        let gain = SNAPSHOT_MIN_GAIN;
        // A long conversation with a turn boundary a fork can land on.
        let mut slots = manager(4, 4);
        let system: Vec<u32> = (0..2 * gain as u32).collect();
        let fork = system.len();
        let mut first = system.clone();
        first.extend((0..2 * gain as u32).map(|i| 900_000 + i));
        let boundary = first.len();
        first.extend(TEST_HEADER);
        assert_eq!(
            serve_with_anchor(&mut slots, &first, Some(fork), boundary, &[77]),
            0,
            "the first conversation is cold"
        );
        let held = slots.live.expect("it is live");
        let history = slots.slots[held].prefix.tokens.clone();
        assert!(slots.slots[held].full_kv.is_none(), "and never imaged");

        // A sibling sharing only the system block. The plan is a rewind of the live slot.
        let mut second = system.clone();
        second.extend([1, 2, 3]);
        let second_boundary = second.len();
        second.extend(TEST_HEADER);
        assert_eq!(
            slots.choose(&second),
            SlotChoice::Live {
                plan: Resume::Restore { pos: fork }
            }
        );
        assert_eq!(
            serve_with_anchor(&mut slots, &second, Some(fork), second_boundary, &[88]),
            fork,
            "the sibling resumes at the fork"
        );

        // The conversation that was live is imaged, whole, in the slot it still owns —
        // which is what a page-out hands the disk writer — and its history is intact.
        assert_eq!(slots.slots[held].full_kv, Some(history.len()));
        assert_eq!(slots.slots[held].prefix.tokens, history);
        assert_eq!(
            slots.slots[held].image_agrees_to, 0,
            "a cold slot's whole history is backed by its own image"
        );

        // The fork took a slot of its own, sharing that image rather than copying it, and
        // both conversations are now matchable — which is the point: a request continuing
        // either one finds it.
        let forked = slots.live.expect("the fork is live");
        assert_ne!(forked, held, "the fork did not take the slot it forked off");
        assert_eq!(slots.slots[forked].full_kv, slots.slots[held].full_kv);
        assert_eq!(slots.slots[forked].image_agrees_to, fork);
        let mut back_to_first = first.clone();
        back_to_first.extend([99]);
        assert_eq!(
            slots.choose(&back_to_first),
            SlotChoice::Swap {
                slot: held,
                restore: boundary
            },
            "the conversation that was forked off is still there to come back to"
        );
    }

    /// A fork at the OLDEST snapshot of a full ring still resumes there.
    ///
    /// This is the case the ordering exists for. Imaging the forked-off conversation
    /// records a tail snapshot, and recording one evicts the oldest entry of a ring at
    /// capacity — which is precisely where a client returning to an old branch of a long
    /// conversation forks. Reading the ring after the page-out would hand the new slot a
    /// resume point that had just been thrown away, and the page-in would fail the
    /// request; the fork's inheritance is therefore taken before the page-out, not after.
    #[test]
    fn a_fork_at_the_oldest_snapshot_of_a_full_ring_still_resumes_there() {
        let gain = SNAPSHOT_MIN_GAIN;
        let mut slots = manager(4, 4);
        let tokens: Vec<u32> = (0..8 * gain as u32).collect();
        // Four turn boundaries: the ring is exactly full, so the page-out's tail snapshot
        // must evict one of them, and the oldest is the one it takes.
        let boundaries = [gain, 2 * gain, 3 * gain, 4 * gain];
        let _ = start(&mut slots);
        for at in boundaries {
            live(&mut slots).prefix.push(at, at);
        }
        live(&mut slots).prefix.set_tokens(tokens.clone());
        let source = slots.live.expect("it is live");

        // A client comes back to the oldest branch point.
        let fork = boundaries[0];
        let mut branch = tokens[..fork].to_vec();
        branch.extend([1, 2, 3]);
        let branch_boundary = branch.len();
        branch.extend(TEST_HEADER);
        assert_eq!(
            slots.choose(&branch),
            SlotChoice::Live {
                plan: Resume::Restore { pos: fork }
            }
        );

        assert_eq!(
            serve_with_anchor(&mut slots, &branch, None, branch_boundary, &[88]),
            fork,
            "the fork resumes at the position it forked at, ring pressure and all"
        );
        let forked = slots.live.expect("the fork is live");
        assert_ne!(forked, source);
        // The page-out did take the oldest boundary out of the source's own ring — the
        // fork survived it because it was captured first, not because it was spared.
        assert_eq!(slots.slots[source].prefix.snapshot_at(fork), None);
        assert_eq!(slots.slots[forked].prefix.snapshot_at(fork), Some(&fork));
        // And both conversations remain matchable.
        let mut back = tokens.clone();
        back.push(99);
        assert!(matches!(
            slots.choose(&back),
            SlotChoice::Swap { slot, .. } if slot == source
        ));
    }

    /// The page-out is skipped when it would buy nothing: a live conversation whose
    /// retained image already covers its history has nothing left to lose to a rewind, and
    /// paying a second page-out for it would be hundreds of milliseconds for a copy that
    /// already exists.
    #[test]
    fn a_fork_onto_an_imaged_conversation_rewinds_in_place() {
        let gain = SNAPSHOT_MIN_GAIN;
        let mut slots = manager(4, 4);
        let tokens: Vec<u32> = (0..4 * gain as u32).collect();
        let fork = gain;

        // Paged out and back in at its full length: the image covers everything.
        let slot = seed(&mut slots, &tokens, &[fork]);
        slots.page_in(slot, tokens.len());
        assert_eq!(slots.slots[slot].image_agrees_to, tokens.len());
        assert_eq!(slots.live_history_at_risk(fork), 0, "nothing is unimaged");

        // A fork rewinds in place, keeping the image it already has.
        let image = slots.slots[slot].full_kv;
        slots.rewind_live(fork).expect("a slot is live");
        assert_eq!(slots.slots[slot].full_kv, image);
        assert_eq!(
            slots.slots[slot].image_agrees_to, fork,
            "the rewind lowers the bound rather than making a second copy"
        );
    }

    /// And skipped when the history it would save is not worth the transfer: the floor is
    /// the one the snapshot planner uses, for the same reason — a page-out costs a fixed
    /// few hundred milliseconds, while what it saves is the prefill of the tokens it
    /// rescues.
    #[test]
    fn a_fork_that_destroys_little_rewinds_in_place() {
        let mut slots = manager(4, 4);
        let tokens: Vec<u32> = (0..SNAPSHOT_MIN_GAIN as u32 + 8).collect();
        let _ = start(&mut slots);
        live(&mut slots).prefix.push(8, 8);
        live(&mut slots).prefix.set_tokens(tokens.clone());

        // A fork right below the end: only the handful of tokens above it would go.
        let near_end = tokens.len() - 8;
        assert_eq!(slots.live_history_at_risk(near_end), 8);
        // A fork at the very start would take the lot, and is worth imaging.
        assert_eq!(slots.live_history_at_risk(8), tokens.len() - 8);
        assert!(slots.live_history_at_risk(8) >= SNAPSHOT_MIN_GAIN);
        // An extension destroys nothing at all.
        assert_eq!(slots.live_history_at_risk(tokens.len()), 0);
        // And with nothing live there is nothing to weigh.
        slots.abandon_live();
        assert_eq!(slots.live_history_at_risk(0), 0);
    }

    /// A failure demotes the slot back to cold with its drafter planes intact, untruncated:
    /// they are bounded when they are imported instead, by the same resume position that
    /// bounds the full-attention image. So the planes may reach past what the demoted slot
    /// claims, and the resume still only ever takes the rows below the bound.
    #[test]
    fn a_demotion_keeps_the_drafter_planes_and_the_import_bounds_them() {
        let mut slots = manager(2, 4);
        let a = seed(&mut slots, &[1, 2, 3, 4, 5, 6], &[4]);
        slots.page_in(a, 4);
        live(&mut slots).prefix.push(9, 9);
        live(&mut slots)
            .prefix
            .set_tokens(vec![1, 2, 3, 4, 7, 8, 9, 9, 9]);

        slots.abandon_live();
        let planes = slots.slots[a].draft_kv.expect("the planes are kept");
        assert_eq!(planes, 6, "kept whole, not truncated to the demotion bound");
        assert_eq!(slots.slots[a].prefix.tokens, vec![1, 2, 3, 4]);
        // They still reach the point the next page-in of this slot can resume at, which is
        // the only thing the extra length is judged by.
        assert!(drafter_planes_usable(
            Some(DrafterKind::Dflash),
            DrafterImageKind::Dflash,
            4,
            planes
        ));

        // A slot emptied outright — nothing the image can back — drops them with the rest.
        slots.page_in(a, 4);
        slots.restart_live().expect("the slot is live");
        live(&mut slots).prefix.set_tokens(vec![7, 8, 9]);
        slots.abandon_live();
        assert_eq!(slots.slots[a].draft_kv, None);
    }

    /// Planes are usable only when they reach the resume point, and only when there is a
    /// drafter to upload them into. A drafter that fell behind the conversation — it stops
    /// injecting at its own smaller context — covers fewer positions than the target
    /// resumes at, and those planes are skipped rather than partly uploaded: the prefill
    /// would refuse to inject on top of a short drafter, so the upload would be a gigabyte
    /// moved for nothing.
    ///
    /// With no drafter attached the answer is the same skip, and that case is not
    /// hypothetical: a stored cache image written by a server running with `--draft`
    /// arrives at one started without it, and the conversation has to come back decoding
    /// plain rather than failing the request.
    #[test]
    fn drafter_planes_are_usable_only_when_they_reach_the_resume_point() {
        let dflash = |restore, covered| {
            drafter_planes_usable(
                Some(DrafterKind::Dflash),
                DrafterImageKind::Dflash,
                restore,
                covered,
            )
        };
        // Every row the target needs is there, extra length included.
        assert!(dflash(400, 1000));
        assert!(dflash(1000, 1000));
        // A drafter that fell behind: nothing to import.
        assert!(!dflash(4000, 1000));
        // Resuming at zero needs no rows at all.
        assert!(dflash(0, 0));
        // And no drafter at all means no upload, however complete the planes are.
        assert!(!drafter_planes_usable(
            None,
            DrafterImageKind::Dflash,
            400,
            1000
        ));
        assert!(!drafter_planes_usable(None, DrafterImageKind::Dflash, 0, 0));
    }

    /// An MTP image is usable at EXACTLY its own position and nowhere else, where a
    /// DFlash image of the same length backs every resume at or below it. The reason is
    /// the carry: the image holds the one hidden its last position produced, and the
    /// head's next row is built from the hidden before it, so an over-long image has the
    /// wrong one for a shorter resume.
    ///
    /// This is the routine case, not the edge: a conversation resuming anywhere but the
    /// exact tip its planes were written at hits it. Answering here keeps it a quiet
    /// skip; letting `import_cache` answer it would log a disk-tier failure on every
    /// ordinary partial resume.
    #[test]
    fn an_mtp_image_backs_only_the_position_it_ends_at() {
        let mtp = |restore, covered| {
            drafter_planes_usable(
                Some(DrafterKind::Mtp),
                DrafterImageKind::Mtp,
                restore,
                covered,
            )
        };
        assert!(mtp(1000, 1000));
        assert!(
            !mtp(400, 1000),
            "an over-long MTP image has the wrong carry"
        );
        assert!(!mtp(4000, 1000));
        assert!(mtp(0, 0));
        // Where the same lengths are fine for a block drafter, whose every row stands on
        // its own position's taps.
        assert!(drafter_planes_usable(
            Some(DrafterKind::Dflash),
            DrafterImageKind::Dflash,
            400,
            1000
        ));
        assert!(!drafter_planes_usable(
            None,
            DrafterImageKind::Mtp,
            1000,
            1000
        ));
    }

    /// Planes written by the OTHER kind of drafter are skipped, not offered to an
    /// import that would refuse them. Reachable without anybody misconfiguring
    /// anything: `--draft <path>` takes either kind against any checkpoint, so a
    /// server told to run 3.8 with the 3.6 DFlash sidecar writes DFlash records for
    /// a checkpoint whose official drafter is an MTP head, and the next restart
    /// without the flag reads them back.
    ///
    /// Length is not what settles it — every case below reaches its resume point —
    /// so a predicate comparing only coverage would pass all of them straight into
    /// a logged import failure.
    #[test]
    fn planes_from_the_other_kind_of_drafter_are_skipped() {
        assert!(!drafter_planes_usable(
            Some(DrafterKind::Mtp),
            DrafterImageKind::Dflash,
            1000,
            1000
        ));
        assert!(!drafter_planes_usable(
            Some(DrafterKind::Dflash),
            DrafterImageKind::Mtp,
            1000,
            1000
        ));
        // And the matching pairs still pass, so the new condition is not simply
        // refusing everything.
        assert!(drafter_planes_usable(
            Some(DrafterKind::Mtp),
            DrafterImageKind::Mtp,
            1000,
            1000
        ));
        assert!(drafter_planes_usable(
            Some(DrafterKind::Dflash),
            DrafterImageKind::Dflash,
            1000,
            1000
        ));
    }

    /// What the slots report about themselves: the histories and positions the manager
    /// keeps, plus host sizes measured by the caller's own functions — the manager reads
    /// none of the three payloads, which is what lets these tests run without a model.
    #[test]
    fn the_slot_summary_reports_what_each_slot_holds() {
        // Payloads stand in for the positions they cover, so a distinct multiplier per
        // kind makes it plain which of them each byte total came from.
        let bytes = |slots: &TestSlots| slots.summary(|s| s * 10, |f| f * 100, |d| d * 1000);

        let mut slots = manager(2, 2);
        // A paged-out conversation: a turn-boundary snapshot at 2, the tail snapshot
        // every page-out leaves at 4, and both images.
        seed(&mut slots, &[1, 2, 3, 4], &[2]);
        start(&mut slots);
        live(&mut slots).prefix.set_tokens(vec![9, 9, 9]);

        let summary = bytes(&slots);
        assert_eq!(summary.len(), 2);
        assert_eq!(
            summary[0],
            SlotSummary {
                live: false,
                tokens: 4,
                snapshots: 2,
                image_bytes: (2 + 4) * 10 + 4 * 100 + 4 * 1000,
                has_drafter: true,
                last_used: 1,
                agrees_to: 0,
            }
        );
        assert_eq!(
            summary[1],
            SlotSummary {
                live: true,
                tokens: 3,
                snapshots: 0,
                // A conversation that started here and has stayed holds no host image.
                image_bytes: 0,
                has_drafter: false,
                last_used: 2,
                agrees_to: 0,
            }
        );

        // Paging the cold one back in makes it live and bounds what its retained image
        // can back; the conversation it displaced is imaged out in its place.
        page_out_of_cache(&mut slots);
        slots.page_in(0, 2);
        let summary = bytes(&slots);
        assert!(summary[0].live && !summary[1].live);
        assert_eq!(summary[0].agrees_to, 2);
        assert_eq!(
            summary[0].snapshots, 1,
            "the branch past the resume is gone"
        );
        assert_eq!(summary[1].image_bytes, 3 * 10 + 3 * 100 + 3 * 1000);
    }

    /// The 35B-A3B sidecar's metadata, and the geometry of the checkpoint it
    /// belongs to.
    const DRAFT_TARGET_HIDDEN: usize = 2048;
    const DRAFT_TARGET_LAYERS: usize = 40;
    const DRAFT_TARGET_VOCAB: usize = 248320;

    /// The 35B-A3B's shape, as much of it as the startup preflight reads: it
    /// judges the DRAFTER against these, and never opens the target's weights.
    fn draft_target_config() -> XwenConfig {
        use crate::config::{Arch, LayerKind, RopeKind};
        XwenConfig {
            arch: Arch::Moe,
            general_name: Some("Qwen3.6-35B-A3B".to_string()),
            n_layer: DRAFT_TARGET_LAYERS,
            hidden: DRAFT_TARGET_HIDDEN,
            vocab: DRAFT_TARGET_VOCAB,
            n_head: vec![16; DRAFT_TARGET_LAYERS],
            n_kv_head: 2,
            head_dim: 256,
            layer_kind: (0..DRAFT_TARGET_LAYERS)
                .map(|il| {
                    if (il + 1).is_multiple_of(4) {
                        LayerKind::Full
                    } else {
                        LayerKind::Linear
                    }
                })
                .collect(),
            linear_k_heads: 16,
            linear_v_heads: 32,
            linear_head_dim: 128,
            conv_kernel: 4,
            dense_ff: 0,
            n_expert: 256,
            n_expert_used: 8,
            expert_ff: 512,
            shared_expert_ff: 512,
            rms_eps: 1e-6,
            n_ctx_train: 262_144,
            rope: RopeKind::Plain {
                freq_base: 1e7,
                n_rot: 64,
            },
            eog_tokens: vec![248046, 248044],
            qwen4exp: None,
        }
    }

    fn draft_config() -> DflashConfig {
        DflashConfig {
            n_layer: 6,
            n_embd: DRAFT_TARGET_HIDDEN,
            n_head: 32,
            n_head_kv: 8,
            head_dim: 128,
            n_ff: 6144,
            rms_eps: 1e-6,
            rope_theta: 10_000_000.0,
            sliding_window: Some(4096),
            swa_layers: vec![true, true, true, true, true, false],
            block_size: 16,
            target_layers: vec![2, 7, 12, 17, 23, 28, 33, 38],
            mask_token_id: 248077,
            context_length: 262144,
        }
    }

    /// Serve refuses a mismatched or malformed drafter at STARTUP rather than on the
    /// first job. The cases themselves are covered exhaustively where the check lives
    /// (`dflash::tests::a_drafter_is_checked_against_the_target_it_will_serve`); what
    /// this pins is that serve's preflight runs it at all, against the config of the
    /// model being served.
    ///
    /// It matters because the lazy load runs behind the `catch_unwind` around a job:
    /// a drafter that slipped through would not kill the engine, it would fail a
    /// request and every retry after it, forever, for a configuration mistake.
    #[test]
    fn serve_preflights_the_drafter_against_the_model_being_served() {
        let ok = draft_config().check_against_target(
            DRAFT_TARGET_HIDDEN,
            DRAFT_TARGET_LAYERS,
            DRAFT_TARGET_VOCAB,
        );
        assert!(ok.is_ok(), "{ok:?}");

        // The pairing serve used to admit: the 35B-A3B drafter against the 27B. Its
        // taps top out at 37, inside the 27B's 64 layers, so nothing but the hidden
        // size separates the two sidecars in this direction.
        let err = draft_config()
            .check_against_target(5120, 64, DRAFT_TARGET_VOCAB)
            .unwrap_err();
        let text = err.to_string();
        assert!(text.contains("2048") && text.contains("5120"), "{text}");

        // And that serve actually RUNS it at startup, on the drafter it picked.
        // A path that is not a GGUF stands in for a drafter that does not fit:
        // both fail the same read, and what is pinned here is that the failure
        // happens during `validate_model` rather than inside the first job.
        let dir = std::env::temp_dir().join(format!("xwen_preflight_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bogus = dir.join("not-a-drafter.gguf");
        std::fs::write(&bogus, b"not a gguf at all").unwrap();
        let mut settings = crate::serve::testutil::settings();
        settings.draft = DraftMode::Custom(bogus.clone());
        let refused = validate_model(
            &settings,
            &draft_target_config(),
            Target::official(hub::Model::Qwen35BA3B),
            &ServeLogger::discarding(),
        );
        // `LagunaTokenizer` is not Debug, so the Ok side cannot be unwrapped by
        // `expect_err`; the error is what this test is about either way.
        let Err(err) = refused else {
            panic!("a drafter that cannot be read must be a startup error");
        };
        let text = format!("{err:#}");
        assert!(text.contains("validating the drafter"), "{text}");
        assert!(text.contains("not-a-drafter.gguf"), "{text}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// WHICH drafter startup can judge. The official sidecar of the checkpoint
    /// being served is judged too, not just a custom path: "official" does not
    /// mean "fits" — a custom GGUF served as its architecture's checkpoint gets
    /// that checkpoint's sidecar, and a geometry mismatch there would otherwise
    /// surface as every request failing at attach, forever, behind the job
    /// boundary. Any OTHER checkpoint's sidecar cannot be judged at startup: it
    /// may not be downloaded yet, and it is checked when it attaches.
    #[test]
    fn startup_judges_the_served_checkpoints_own_drafter() {
        let mut settings = crate::serve::testutil::settings();
        let served = Target::official(hub::Model::Qwen27B);

        settings.draft = DraftMode::Off;
        assert_eq!(startup_drafter(&settings, served), None);

        let custom = std::path::PathBuf::from("/drafters/mine.gguf");
        settings.draft = DraftMode::Custom(custom.clone());
        assert_eq!(startup_drafter(&settings, served), Some(custom));

        // The official mode preflights the SERVED checkpoint's own sidecar and
        // no other's — whatever else happens to be in the cache, which is what
        // keeps these assertions independent of the machine running them.
        settings.draft = DraftMode::Official;
        assert_eq!(
            startup_drafter(&settings, served),
            hub::cached_drafter(hub::Model::Qwen27B)
        );
        assert_eq!(
            startup_drafter(&settings, Target::official(hub::Model::Qwen3827B)),
            hub::cached_drafter(hub::Model::Qwen3827B)
        );
    }

    /// A server that configured nothing takes each checkpoint's own drafting
    /// default, and one that named the official drafter takes it on every
    /// checkpoint that ships one.
    ///
    /// The 35B-A3B is where the two part: its own default went off on
    /// 2026-09-06 after the drafted arm read below plain at every length, so
    /// `Default` finds nothing to preflight there while `Official` still
    /// preflights the sidecar the operator asked for. The other checkpoints
    /// answer the same under both, which is what keeps this a policy change and
    /// not a behaviour change everywhere.
    #[test]
    fn an_unconfigured_server_follows_each_checkpoints_drafting_default() {
        let mut settings = crate::serve::testutil::settings();
        settings.draft = DraftMode::Default;

        // Nothing to judge: this checkpoint will not attach a drafter unasked.
        assert_eq!(
            startup_drafter(&settings, Target::official(hub::Model::Qwen35BA3B)),
            None
        );
        // ...while an explicit request still preflights it.
        settings.draft = DraftMode::Official;
        assert_eq!(
            startup_drafter(&settings, Target::official(hub::Model::Qwen35BA3B)),
            hub::cached_drafter(hub::Model::Qwen35BA3B)
        );

        // The drafting-by-default checkpoints read the same either way.
        for model in [hub::Model::Qwen27B, hub::Model::Qwen3827B] {
            for mode in [DraftMode::Default, DraftMode::Official] {
                settings.draft = mode;
                assert_eq!(
                    startup_drafter(&settings, Target::official(model)),
                    hub::cached_drafter(model),
                    "{model:?}"
                );
            }
        }

        // And `--no-draft` still means nothing, on every checkpoint.
        settings.draft = DraftMode::Off;
        for model in [
            hub::Model::Qwen27B,
            hub::Model::Qwen35BA3B,
            hub::Model::Qwen3827B,
        ] {
            assert_eq!(startup_drafter(&settings, Target::official(model)), None);
        }
    }

    /// The per-request speculation report: made once, and only when speculation was
    /// the decode path. The wording it renders to is pinned in `log.rs`.
    #[test]
    fn the_spec_report_is_made_only_when_speculation_ran() {
        assert!(spec_report(None).is_none());
        // A decode that never ran a round has nothing to report.
        assert!(spec_report(Some(SpecStats::default())).is_none());

        let stats = SpecStats {
            rounds: 10,
            drafted: 40,
            accepted: 30,
            ..SpecStats::default()
        };
        assert_eq!(
            spec_report(Some(stats))
                .and_then(|log| log.render())
                .unwrap(),
            "xwen serve: spec: 10 rounds, drafted 40, accepted 30 (75%)"
        );
    }

    fn filter(sequences: &[&str]) -> StopFilter {
        StopFilter::new(&sequences.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    /// With no stop sequences configured, text passes straight through.
    #[test]
    fn no_stop_sequences_hold_nothing_back() {
        let mut stop = filter(&[]);
        assert_eq!(
            stop.push("anything at all"),
            ("anything at all".to_string(), None)
        );
        assert_eq!(stop.flush(), "");
    }

    /// Text that cannot be part of a match is delivered immediately; a partial match is
    /// held until the next delta decides it.
    #[test]
    fn a_partial_match_is_held_until_it_is_resolved() {
        let mut stop = filter(&["STOP"]);
        assert_eq!(stop.push("hello ST"), ("hello ".to_string(), None));
        // The tail turned out to be ordinary text, so it is released.
        assert_eq!(stop.push("EAM"), ("STEAM".to_string(), None));
        assert_eq!(stop.flush(), "");
    }

    /// A sequence split across several deltas still matches, and nothing from the match on
    /// is delivered.
    #[test]
    fn a_match_across_delta_boundaries_is_caught() {
        let mut stop = filter(&["STOP"]);
        assert_eq!(stop.push("answer S"), ("answer ".to_string(), None));
        assert_eq!(stop.push("T"), (String::new(), None));
        assert_eq!(
            stop.push("OP and more"),
            (String::new(), Some("STOP".to_string()))
        );
        assert_eq!(stop.flush(), "");
    }

    /// Text before the match is delivered; the match and its tail are not.
    #[test]
    fn text_before_a_match_is_delivered() {
        let mut stop = filter(&["\n\nHuman:"]);
        let (ready, matched) = stop.push("the answer.\n\nHuman: next");
        assert_eq!(ready, "the answer.");
        assert_eq!(matched, Some("\n\nHuman:".to_string()));
    }

    /// The earliest match wins, and among sequences starting together the longest one is
    /// reported.
    #[test]
    fn overlapping_candidates_report_the_longest_at_the_earliest_start() {
        let mut stop = filter(&["END", "ENDING"]);
        let (ready, matched) = stop.push("all ENDING now");
        assert_eq!(ready, "all ");
        assert_eq!(matched, Some("ENDING".to_string()));

        // A later, longer sequence does not preempt an earlier short one.
        let mut stop = filter(&["X", "abcd"]);
        let (ready, matched) = stop.push("__X__abcd");
        assert_eq!(ready, "__");
        assert_eq!(matched, Some("X".to_string()));
    }

    /// Several sequences are tracked at once: the holdback covers the longest live
    /// candidate, whichever sequence it belongs to.
    #[test]
    fn multiple_sequences_are_tracked_together() {
        let mut stop = filter(&["<|end|>", "\nUser:"]);
        assert_eq!(stop.push("text\nUse"), ("text".to_string(), None));
        assert_eq!(stop.push("ful <|en"), ("\nUseful ".to_string(), None));
        assert_eq!(
            stop.push("d|>"),
            (String::new(), Some("<|end|>".to_string()))
        );
    }

    /// Holdback splits on character boundaries, never inside a codepoint.
    #[test]
    fn unicode_is_never_split_mid_codepoint() {
        let mut stop = filter(&["…done"]);
        // "é" ends the delta but cannot start the sequence, so nothing is held.
        assert_eq!(stop.push("café"), ("café".to_string(), None));
        // The ellipsis can, so all three of its bytes stay behind together.
        assert_eq!(stop.push(" …"), (" ".to_string(), None));
        assert_eq!(
            stop.push("done"),
            (String::new(), Some("…done".to_string()))
        );

        // A multi-byte character mid-sequence is held and released intact.
        let mut stop = filter(&["→END"]);
        assert_eq!(stop.push("go →"), ("go ".to_string(), None));
        assert_eq!(stop.push(" back"), ("→ back".to_string(), None));
    }

    /// Unmatched holdback is delivered when generation ends on its own.
    #[test]
    fn the_holdback_is_flushed_at_the_end() {
        let mut stop = filter(&["STOP"]);
        assert_eq!(stop.push("done ST"), ("done ".to_string(), None));
        assert_eq!(stop.flush(), "ST");
        assert_eq!(stop.flush(), "");
    }

    /// A sequence the model writes as one token is caught just the same.
    #[test]
    fn a_whole_sequence_in_one_delta_matches() {
        let mut stop = filter(&["STOP"]);
        assert_eq!(stop.push("STOP"), (String::new(), Some("STOP".to_string())));
    }

    fn text(delta: &str) -> EngineEvent {
        EngineEvent::Text(delta.to_string())
    }

    /// A reader that has fallen behind is throttled, not dropped: the send waits
    /// for room and delivers.
    #[test]
    fn a_slow_reader_gets_backpressure_rather_than_a_hangup() {
        let (events, mut receiver) = tokio::sync::mpsc::channel(1);
        events
            .try_send(text("first"))
            .expect("the channel starts empty");

        let drain = std::thread::spawn(move || {
            std::thread::sleep(SEND_RETRY_INTERVAL * 4);
            receiver.blocking_recv();
            receiver
        });
        assert!(send_until_interrupted(
            &events,
            text("second"),
            Instant::now() + Duration::from_secs(5),
            &|| false,
            test_logger()
        ));
        drop(drain.join().expect("the reader thread finishes"));
    }

    /// A client that never reads again holds a full channel forever. The deadline
    /// is what keeps the one inference thread from waiting with it.
    #[test]
    fn a_client_that_stopped_reading_hits_the_deadline() {
        let (events, _receiver) = tokio::sync::mpsc::channel(1);
        events
            .try_send(text("first"))
            .expect("the channel starts empty");
        let started = Instant::now();
        assert!(!send_until_interrupted(
            &events,
            text("second"),
            started + Duration::from_millis(50),
            &|| false,
            test_logger()
        ));
        assert!(started.elapsed() >= Duration::from_millis(50));
    }

    /// A disconnect is noticed at once, without waiting out the deadline.
    #[test]
    fn a_hung_up_client_is_noticed_immediately() {
        let (events, receiver) = tokio::sync::mpsc::channel(1);
        drop(receiver);
        let started = Instant::now();
        assert!(!send_until_interrupted(
            &events,
            text("anything"),
            started + Duration::from_secs(60),
            &|| false,
            test_logger()
        ));
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    /// An abandonment context over `events` that nothing has cancelled: a live
    /// job with a deadline too far off to fire.
    fn live_abandon<'a>(
        shutdown: &'a Cancel,
        cancel: &'a Cancel,
        events: &'a Sender<EngineEvent>,
    ) -> Abandon<'a> {
        let started = Instant::now();
        Abandon::new(
            shutdown,
            cancel,
            events,
            test_logger(),
            started,
            Some(started + Duration::from_secs(3600)),
        )
    }

    /// The reply is opened exactly once, and every path that can end a turn opens it
    /// first: a client cannot be told how a reply ended before it has been told one
    /// started, and the SSE dialects render the opening event as the frame every later
    /// one belongs to.
    ///
    /// The number it carries is how much of the prompt came out of a cache, which is
    /// why it is sent from one place after the dispatch has settled: a stored image can
    /// be chosen and then turn out to be unusable, so the resume is a fact only once
    /// the dispatch is over.
    #[test]
    fn the_reply_is_opened_once_and_before_anything_that_ends_it() {
        let (events, mut receiver) = tokio::sync::mpsc::channel(8);
        let (shutdown, cancel) = (Cancel::default(), Cancel::default());
        let abandon = live_abandon(&shutdown, &cancel, &events);
        let opened = Cell::new(false);

        assert!(open_reply(&abandon, &opened, 4096, 2048));
        // A second call sends nothing: one opening event per reply.
        assert!(open_reply(&abandon, &opened, 4096, 9999));
        assert_eq!(
            receiver.try_recv().expect("the opening event"),
            EngineEvent::Start {
                input_tokens: 4096,
                cached_tokens: 2048,
            }
        );
        assert!(receiver.try_recv().is_err(), "and only one");

        // A job cut short before it opened the turn opens it on the way out, so the
        // terminal event is not the first thing its client sees.
        let (events, mut receiver) = tokio::sync::mpsc::channel(8);
        let abandon = live_abandon(&shutdown, &cancel, &events);
        let opened = Cell::new(false);
        let mut trace = JobTrace::new(
            RequestOrigin {
                id: 1,
                dialect: crate::serve::types::Dialect::Anthropic,
                streaming: true,
                client: None,
                session: None,
                agent: None,
            },
            "Qwen3.6-35B-A3B".to_string(),
            4096,
            false,
            Instant::now(),
        );
        finish_abandoned_before_decode(
            &abandon,
            &opened,
            &mut trace,
            CancelReason::Deadline,
            JobPhase::Prefill,
            2048,
            4096,
            2048,
        );
        assert_eq!(
            receiver.try_recv().expect("the opening event"),
            EngineEvent::Start {
                input_tokens: 4096,
                cached_tokens: 2048,
            }
        );
        assert!(matches!(
            receiver.try_recv().expect("then the terminal event"),
            EngineEvent::Done {
                stop: StopKind::MaxTokens,
                ..
            }
        ));

        // A client that is already gone is owed nothing at all, opening event
        // included.
        let (events, receiver) = tokio::sync::mpsc::channel(8);
        drop(receiver);
        let abandon = live_abandon(&shutdown, &cancel, &events);
        let opened = Cell::new(false);
        finish_abandoned_before_decode(
            &abandon,
            &opened,
            &mut trace,
            CancelReason::ClientGone,
            JobPhase::Prefill,
            0,
            4096,
            0,
        );
        assert!(!opened.get(), "nothing was sent to a client that left");
    }

    /// A logger with no sink behind it, `'static` so an [`Abandon`] can borrow it
    /// for whatever lifetime a test needs. What these sites would print is pinned
    /// in `log.rs`.
    fn test_logger() -> &'static ServeLogger {
        static LOGGER: std::sync::OnceLock<ServeLogger> = std::sync::OnceLock::new();
        LOGGER.get_or_init(ServeLogger::discarding)
    }

    /// A blocked send gives up the moment shutdown fires, instead of waiting out
    /// the reader's [`SEND_DEADLINE`] — which is what would otherwise pin the
    /// engine thread past the shutdown watchdog.
    #[test]
    fn shutdown_interrupts_a_blocked_send() {
        let (events, _receiver) = tokio::sync::mpsc::channel(1);
        events
            .try_send(text("first"))
            .expect("the channel starts empty");

        let shutdown = Cancel::default();
        let cancel = Cancel::default();
        let abandon = live_abandon(&shutdown, &cancel, &events);
        shutdown.cancel(CancelReason::Shutdown);
        let started = Instant::now();
        assert!(!abandon.send(text("second")));
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    /// The first send that fails to land is final: even after the channel gains
    /// room, nothing more is offered — a stream that already dropped an event
    /// must not deliver the ones after it.
    #[test]
    fn a_failed_send_suppresses_every_later_one() {
        let (events, mut receiver) = tokio::sync::mpsc::channel(1);
        events
            .try_send(text("first"))
            .expect("the channel starts empty");
        let shutdown = Cancel::default();
        let cancel = Cancel::default();
        let abandon = live_abandon(&shutdown, &cancel, &events);
        // The shutdown makes the failure immediate; any failure latches.
        shutdown.cancel(CancelReason::Shutdown);
        assert!(!abandon.send(text("dropped")));

        // The reader comes back and the channel has room again — but the send
        // after a dropped event stays suppressed rather than landing.
        assert_eq!(receiver.blocking_recv(), Some(text("first")));
        assert!(!abandon.send(text("after the drop")));
        assert!(
            matches!(
                receiver.try_recv(),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty)
            ),
            "a suppressed send offers nothing to the channel"
        );
    }

    /// While the job is live every event gets a fresh delivery deadline; once a
    /// cancellation has been observed, the remaining sends share one budget
    /// anchored at the observation, so the wind-down cannot park the inference
    /// thread for a full deadline per event.
    #[test]
    fn cancelled_sends_share_one_budget_instead_of_a_deadline_each() {
        let observed = Instant::now();
        let later = observed + Duration::from_secs(5);
        let even_later = observed + Duration::from_secs(50);
        assert_eq!(send_deadline(None, later), later + SEND_DEADLINE);
        assert_eq!(
            send_deadline(Some(observed), later),
            observed + SEND_DEADLINE
        );
        assert_eq!(
            send_deadline(Some(observed), even_later),
            send_deadline(Some(observed), later),
            "every send after the observation shares the same absolute deadline"
        );
    }

    /// Finalization judges by how the decode loop actually stopped: a reason
    /// stamped after the loop ended naturally — a deadline racing the final
    /// token — must not relabel a complete reply as truncated.
    #[test]
    fn a_late_cancel_stamp_never_relabels_a_natural_finish() {
        // Natural ends: EOG, the cap, or a matched stop sequence — whatever a
        // fresh poll of the token says.
        assert_eq!(
            decode_abandon_reason(false, false, false, Some(CancelReason::Deadline)),
            None
        );
        assert_eq!(
            decode_abandon_reason(true, true, false, Some(CancelReason::Deadline)),
            None,
            "a stop-sequence match ends the loop naturally even though it fires the predicate"
        );

        // The loop itself was cut: the settled reason stands, and a failed send
        // with no stamped reason is the client gone.
        assert_eq!(
            decode_abandon_reason(true, false, false, Some(CancelReason::Deadline)),
            Some(CancelReason::Deadline)
        );
        assert_eq!(
            decode_abandon_reason(true, false, true, None),
            Some(CancelReason::ClientGone)
        );
    }

    /// The fold reads every signal — the process-wide shutdown, the job's own
    /// token, the deadline, the closed channel — and stamps what it finds into
    /// the job token, where the first reason to land is the one that sticks.
    #[test]
    fn the_abandonment_fold_settles_on_one_reason() {
        let (events, receiver) = tokio::sync::mpsc::channel(1);
        let shutdown = Cancel::default();
        let cancel = Cancel::default();
        let abandon = live_abandon(&shutdown, &cancel, &events);
        assert_eq!(abandon.reason(), None);
        assert_eq!(
            abandon.cancelled_at.get(),
            None,
            "a live job has observed no cancellation"
        );

        // A closed channel is a gone client, stamped into the job token like
        // every other reason, so all consumers converge on the one settled value.
        drop(receiver);
        assert_eq!(abandon.reason(), Some(CancelReason::ClientGone));
        assert_eq!(cancel.reason(), Some(CancelReason::ClientGone));

        // The token is write-once: a shutdown arriving after the hangup was
        // stamped keeps reporting the hangup the finalization acted on.
        shutdown.cancel(CancelReason::Shutdown);
        assert_eq!(abandon.reason(), Some(CancelReason::ClientGone));
        assert_eq!(cancel.reason(), Some(CancelReason::ClientGone));

        // The first observation also pins the shared send budget's start.
        let observed = abandon
            .cancelled_at
            .get()
            .expect("observing a reason starts the send budget");
        assert_eq!(abandon.reason(), Some(CancelReason::ClientGone));
        assert_eq!(
            abandon.cancelled_at.get(),
            Some(observed),
            "later polls never move the budget's start"
        );
    }

    /// An expired deadline is stamped as `Deadline` — unless something else got
    /// there first, which the write-once token preserves.
    #[test]
    fn an_expired_deadline_is_stamped_unless_a_reason_already_landed() {
        let (events, _receiver) = tokio::sync::mpsc::channel(1);
        let shutdown = Cancel::default();
        let cancel = Cancel::default();
        let started = Instant::now();
        let abandon = Abandon::new(
            &shutdown,
            &cancel,
            &events,
            test_logger(),
            started,
            Some(started),
        );
        assert_eq!(abandon.reason(), Some(CancelReason::Deadline));
        assert_eq!(cancel.reason(), Some(CancelReason::Deadline));

        // The same expiry with the client already gone reports the client.
        let gone = Cancel::default();
        gone.cancel(CancelReason::ClientGone);
        let abandon = Abandon::new(
            &shutdown,
            &gone,
            &events,
            test_logger(),
            started,
            Some(started),
        );
        assert_eq!(abandon.reason(), Some(CancelReason::ClientGone));

        // A job with no ceiling never stamps `Deadline`, however long it runs.
        let unbounded = Cancel::default();
        let abandon = Abandon::new(&shutdown, &unbounded, &events, test_logger(), started, None);
        assert_eq!(abandon.reason(), None);
    }

    /// The deadline is stamped at pickup and the job starts after the lazy model
    /// load, so a load that outlasts the whole budget leaves a deadline behind
    /// the start. That is a ceiling of zero, not a panicked engine thread.
    #[test]
    fn a_ceiling_that_expired_during_the_model_load_reports_zero() {
        let (events, _receiver) = tokio::sync::mpsc::channel(1);
        let shutdown = Cancel::default();
        let cancel = Cancel::default();
        // Stamped at pickup, then thirty seconds of loading before the job ran.
        let stamped = Instant::now();
        let started = stamped + Duration::from_secs(30);
        let expired = Abandon::new(
            &shutdown,
            &cancel,
            &events,
            test_logger(),
            started,
            Some(stamped),
        );
        assert_eq!(expired.ceiling(), Duration::ZERO);

        let live = Abandon::new(
            &shutdown,
            &cancel,
            &events,
            test_logger(),
            started,
            Some(started + Duration::from_secs(30)),
        );
        assert_eq!(live.ceiling(), Duration::from_secs(30));

        let unbounded = Abandon::new(&shutdown, &cancel, &events, test_logger(), started, None);
        assert_eq!(unbounded.ceiling(), Duration::ZERO);
    }

    /// The drafting floor follows the checkpoint being loaded when the
    /// operator left it unset, and a pinned value applies to both — the arc's
    /// central config change, testable here because the resolution was
    /// extracted from the Metal-bound attach path.
    #[test]
    fn the_drafting_floor_follows_the_loaded_checkpoint_unless_pinned() {
        let mut settings = crate::serve::testutil::settings();
        settings.draft_p_min = None;
        settings.draft_max = None;
        assert_eq!(
            Some(resolved_p_min(&settings, hub::Model::Qwen27B)),
            hub::Model::Qwen27B.draft_p_min_default()
        );
        assert_eq!(
            Some(resolved_p_min(&settings, hub::Model::Qwen35BA3B)),
            hub::Model::Qwen35BA3B.draft_p_min_default()
        );
        assert_eq!(
            Some(resolved_p_min(&settings, hub::Model::Qwen3827B)),
            hub::Model::Qwen3827B.draft_p_min_default()
        );
        // The shared base behind the fitted floors is live again: a checkpoint
        // that ships no sidecar of its own has nothing to fit a floor with, so
        // an operator's custom `draft.path` on it drafts at the base.
        assert_eq!(hub::Model::Qwen38FlashNext.draft_p_min_default(), None);
        assert_eq!(
            resolved_p_min(&settings, hub::Model::Qwen38FlashNext),
            SpecParams::default().draft_p_min
        );
        assert_eq!(
            resolved_draft_max(&settings, hub::Model::Qwen38FlashNext),
            SpecParams::default().draft_max
        );
        settings.draft_p_min = Some(0.42);
        for model in hub::MODELS {
            assert_eq!(resolved_p_min(&settings, model), 0.42, "{model:?}");
        }
    }

    /// The ceiling grows with both the prompt and the reply budget, so spans
    /// three orders of magnitude apart each get a ceiling that fits them, and a
    /// tiny request still gets the full fixed slack.
    #[test]
    fn the_job_deadline_is_monotonic_in_both_inputs() {
        let s = crate::serve::testutil::settings();
        let ceiling = |prompt: usize, reply: usize| {
            let now = Instant::now();
            job_deadline(now, prompt, reply, &s).expect("the default rates bound jobs") - now
        };
        let now = Instant::now();
        assert_eq!(job_deadline(now, 0, 0, &s), Some(now + s.request_slack));
        assert!(job_deadline(now, 4, 4, &s) >= Some(now + s.request_slack));
        assert!(ceiling(2000, 100) > ceiling(1000, 100));
        assert!(ceiling(1000, 200) > ceiling(1000, 100));
        // The watchdog rates: a 4k prefill plus a 1k reply buys minutes, not seconds.
        assert!(ceiling(4007, 1024) > Duration::from_secs(120));
    }

    /// A watchdog rate of 0 makes its term unbounded, and the ceiling being one
    /// instant, an unbounded term means no deadline at all — for either rate.
    #[test]
    fn a_zero_watchdog_rate_disables_the_deadline() {
        let now = Instant::now();
        let with_rates = |prefill: u64, decode: u64| ServeSettings {
            request_prefill_rate: prefill,
            request_decode_rate: decode,
            ..crate::serve::testutil::settings()
        };
        assert_eq!(job_deadline(now, 1000, 1000, &with_rates(0, 10)), None);
        assert_eq!(job_deadline(now, 1000, 1000, &with_rates(150, 0)), None);
        assert_eq!(job_deadline(now, 1000, 1000, &with_rates(0, 0)), None);
        assert!(job_deadline(now, 1000, 1000, &with_rates(150, 10)).is_some());
    }

    /// A ceiling too far off for an `Instant` to hold — the config accepts slack
    /// up to u64::MAX seconds — is no ceiling at all, never a panic on the
    /// engine thread.
    #[test]
    fn an_unrepresentable_ceiling_disables_the_deadline() {
        let s = ServeSettings {
            request_slack: Duration::from_secs(u64::MAX),
            ..crate::serve::testutil::settings()
        };
        assert_eq!(job_deadline(Instant::now(), 1000, 1000, &s), None);
    }

    /// However the engine thread exits — the shutdown break or a panic escaping
    /// the per-job boundary — the guard closes the queue and clears the flag, so
    /// handlers get `EngineGone` instead of pushing jobs no thread will take.
    #[test]
    fn the_exit_guard_closes_the_queue_and_clears_the_flag() {
        let jobs = Arc::new(JobQueue::new(
            2,
            crate::serve::queue::SchedulePolicy {
                schedule: crate::serve::config::Schedule::Fifo,
                queue_timeout: Duration::from_secs(300),
                age_limit: Duration::from_secs(20),
            },
            ServeLogger::discarding(),
        ));
        let resident = Arc::new(ResidentModel::new());
        resident.store(Target::official(crate::hub::Model::Qwen35BA3B));
        let guard = EngineExitGuard {
            jobs: Arc::clone(&jobs),
            resident: Arc::clone(&resident),
        };
        assert!(!jobs.is_closed());

        // An unwind drops the guard exactly like a normal return does.
        let unwound = std::panic::catch_unwind(AssertUnwindSafe(move || {
            let _guard = guard;
            std::panic::panic_any("the engine loop blew up outside the job boundary")
        }));
        assert!(unwound.is_err());
        assert!(jobs.is_closed(), "the exit guard closes the queue");
        assert!(
            !resident.is_loaded(),
            "the exit guard stops /health claiming a model nobody serves"
        );
        assert_eq!(
            resident.get(),
            None,
            "and it leaves no checkpoint name behind either"
        );
    }

    /// A panic during the lazy load is caught, installs nothing, and a later job
    /// retries the load from scratch — the engine thread outlives the panic.
    #[test]
    fn a_load_panic_installs_nothing_and_the_next_job_retries() {
        let mut state: Option<u32> = None;
        let outcome = run_behind_boundary(
            &mut state,
            || std::panic::panic_any("the load blew up"),
            |_| Ok(()),
        );
        assert!(matches!(
            outcome,
            JobOutcome::Panicked { model_lost: false }
        ));
        assert!(state.is_none(), "a mid-load panic must install nothing");

        // The retry loads and serves normally.
        let outcome = run_behind_boundary(
            &mut state,
            || Ok(7),
            |loaded| {
                assert_eq!(*loaded, 7);
                Ok(())
            },
        );
        assert!(matches!(outcome, JobOutcome::Completed));
        assert_eq!(state, Some(7));
    }

    /// A panic mid-job reports the model lost: its caches can no longer be
    /// reasoned about, and the caller drops the state.
    #[test]
    fn a_job_panic_reports_the_model_lost() {
        let mut state = Some(7u32);
        let outcome = run_behind_boundary(
            &mut state,
            || unreachable!("the state is already loaded"),
            |_| std::panic::panic_any("the model layer blew up"),
        );
        assert!(matches!(outcome, JobOutcome::Panicked { model_lost: true }));
    }

    /// A load that fails without panicking reports the failure as the server's
    /// fault, installs nothing, and never runs the job.
    #[test]
    fn a_failed_load_is_reported_and_installs_nothing() {
        let mut state: Option<u32> = None;
        let outcome = run_behind_boundary(
            &mut state,
            || Err(JobFailure::from(anyhow!("no such checkpoint"))),
            |_| std::panic::panic_any("the job must not run without a load"),
        );
        match outcome {
            JobOutcome::Failed(failure) => {
                assert_eq!(
                    failure.into_event(),
                    EngineEvent::Error {
                        message: "no such checkpoint".into(),
                        request_fault: false,
                    }
                );
            }
            _ => panic!("a failed load must be reported to the requesting job"),
        }
        assert!(state.is_none());
    }

    /// An already loaded state skips the load entirely and hands the job the
    /// live value.
    #[test]
    fn a_loaded_state_is_reused_without_reloading() {
        let mut state = Some(7u32);
        let outcome = run_behind_boundary(
            &mut state,
            || unreachable!("the state is already loaded"),
            |loaded| {
                *loaded += 1;
                Ok(())
            },
        );
        assert!(matches!(outcome, JobOutcome::Completed));
        assert_eq!(state, Some(8));
    }

    /// Failures carry their own classification rather than being sniffed out of
    /// the message downstream: `?` propagation is a server failure, and a request
    /// fault has to be stated.
    #[test]
    fn only_a_stated_failure_is_the_requests_fault() {
        let propagated: JobFailure = anyhow!("Metal command buffer failed").into();
        match propagated.into_event() {
            EngineEvent::Error {
                message,
                request_fault,
            } => {
                assert!(!request_fault);
                assert_eq!(message, "Metal command buffer failed");
            }
            _ => panic!("a failure renders as an error event"),
        }

        let stated = JobFailure::request(anyhow!("the prompt is 900000 tokens"));
        match stated.into_event() {
            EngineEvent::Error { request_fault, .. } => assert!(request_fault),
            _ => panic!("a failure renders as an error event"),
        }
    }

    /// One tool declaration in the OpenAI shape the engine is handed.
    fn tool(name: &str, properties: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": properties},
            },
        })
    }

    /// The weather tool the span tests call, with one parameter of each shape
    /// the parser distinguishes.
    fn weather() -> Vec<serde_json::Value> {
        vec![tool(
            "get_weather",
            serde_json::json!({
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "filters": {"type": "object"},
                "tags": {"type": "array"},
            }),
        )]
    }

    /// What one scripted generation produced.
    struct Run {
        events: Vec<EngineEvent>,
        stop: StopKind,
        healed: usize,
        quoted: usize,
        degraded: usize,
    }

    impl Run {
        /// The answer text the client received.
        fn answer(&self) -> String {
            self.events
                .iter()
                .filter_map(|event| match event {
                    EngineEvent::Text(delta) => Some(delta.as_str()),
                    _ => None,
                })
                .collect()
        }

        /// The calls as `(name, arguments)`, with the protocol checked on the
        /// way: calls never interleave, every one is closed, and the deltas of
        /// a call concatenate into one JSON object. The arguments come back as
        /// the raw delta text, so a test can pin argument order too.
        fn calls(&self) -> Vec<(String, String)> {
            let mut calls = Vec::new();
            let mut open: Option<(String, String)> = None;
            for event in &self.events {
                match event {
                    EngineEvent::ToolCallStart { name } => {
                        assert!(open.is_none(), "a call started inside another");
                        open = Some((name.clone(), String::new()));
                    }
                    EngineEvent::ToolCallDelta(delta) => {
                        let (_, arguments) = open.as_mut().expect("a delta outside a call");
                        arguments.push_str(delta);
                    }
                    EngineEvent::ToolCallEnd => {
                        let (name, arguments) = open.take().expect("a call ended without starting");
                        let parsed: serde_json::Value = serde_json::from_str(&arguments)
                            .unwrap_or_else(|e| panic!("{arguments:?} is not valid JSON: {e}"));
                        assert!(parsed.is_object(), "{arguments:?} is not a JSON object");
                        calls.push((name, arguments));
                    }
                    _ => {}
                }
            }
            assert!(open.is_none(), "a call was left open");
            calls
        }

        /// The single call this generation made.
        fn call(&self) -> (String, String) {
            let calls = self.calls();
            assert_eq!(calls.len(), 1, "expected exactly one call: {calls:?}");
            calls.into_iter().next().expect("just counted")
        }
    }

    /// The embedded vocabulary's marker ids, which the scripted streams below
    /// spell in terms of. Resolved from the tokenizer rather than written out,
    /// so a script and the emitter under test always name the same tokens.
    fn embedded_specials() -> Specials {
        static SPECIALS: std::sync::OnceLock<Specials> = std::sync::OnceLock::new();
        *SPECIALS.get_or_init(|| *LagunaTokenizer::embedded().unwrap().specials())
    }

    /// `<tool_call>` and `</tool_call>` in the embedded vocabulary, the two ids
    /// every tool-parser script brackets a call with.
    fn tool_call_open() -> u32 {
        embedded_specials().tool_call_open
    }

    fn tool_call_close() -> u32 {
        embedded_specials().tool_call_close
    }

    /// Drive the emitter over a scripted `(token id, finalized text)` stream,
    /// then end the generation the way `run_job` does. `hit_eog` tells an
    /// end-of-generation token from the output cap.
    fn drive<S: AsRef<str>>(
        tools: Vec<serde_json::Value>,
        stops: &[&str],
        script: &[(u32, S)],
        hit_eog: bool,
    ) -> Run {
        let (sender, mut receiver) = tokio::sync::mpsc::channel(1024);
        let shutdown = Cancel::default();
        let cancel = Cancel::default();
        let abandon = live_abandon(&shutdown, &cancel, &sender);
        let stopped = Cell::new(false);
        let disconnected = Cell::new(false);
        let first_sent = Cell::new(None);
        let sequences = stops.iter().map(|s| s.to_string()).collect();
        let mut emitter = Emitter::new(
            &abandon,
            sequences,
            &tools,
            embedded_specials(),
            &stopped,
            &disconnected,
            &first_sent,
        );
        for (id, text) in script {
            // The decode loop polls the stop flag before drawing the next token.
            if stopped.get() {
                break;
            }
            emitter.accept(GenEvent::TextTok {
                id: *id,
                text: text.as_ref().to_string(),
            });
        }
        emitter.heal_open_call();
        if emitter.matched.is_none() {
            let tail = emitter.flush_tail();
            if !tail.is_empty() {
                emitter.send(EngineEvent::Text(tail));
            }
        }
        let stop = terminal_stop(
            emitter.matched.take(),
            emitter.internal_stop.as_deref(),
            emitter.called_tools,
            hit_eog,
        );
        assert!(!disconnected.get(), "the test channel never fills");
        let (healed, quoted, degraded) = (emitter.healed, emitter.quoted, emitter.degraded);
        drop(emitter);
        let mut events = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            events.push(event);
        }
        Run {
            events,
            stop,
            healed,
            quoted,
            degraded,
        }
    }

    /// The declared type decides how a value is rendered: a string is quoted and
    /// escaped, everything else is passed through as the JSON the model wrote.
    /// Argument order is the model's.
    #[test]
    fn declared_types_decide_how_a_value_is_rendered() {
        let run = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (0, "\n<function=get_weather>\n"),
                (0, "<parameter=city>\nSan Francisco\n</parameter>\n"),
                (0, "<parameter=days>\n3\n</parameter>\n"),
                (0, "<parameter=filters>\n{\"wind\":true}\n</parameter>\n"),
                (0, "<parameter=tags>\n[\"a\",\"b\"]\n</parameter>\n"),
                (0, "</function>\n"),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert_eq!(
            run.call(),
            (
                "get_weather".to_string(),
                r#"{"city":"San Francisco","days":3,"filters":{"wind":true},"tags":["a","b"]}"#
                    .to_string()
            )
        );
        assert_eq!(run.answer(), "");
        assert_eq!(run.stop, StopKind::ToolUse);
        assert_eq!((run.healed, run.quoted, run.degraded), (0, 0, 0));
    }

    /// A string value is the model's raw text, so it may contain anything that
    /// looks like a marker. Only a complete `</parameter>` ends it, and a prefix
    /// of one split across decode steps is held back until it is resolved —
    /// including the bare form the framing newline usually precedes.
    #[test]
    fn a_string_value_survives_marker_lookalikes() {
        let run = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=city>\nif (a < b) {",
                ),
                (0, "}</para"),
                (0, "meterX\n</parameter"),
                (0, ">\n</function>\n"),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert_eq!(
            run.call().1,
            r#"{"city":"if (a < b) {}</parameterX"}"#.to_string()
        );
    }

    /// A string value is escaped as it streams, fragment by fragment, and the
    /// holdback splits it only on character boundaries — so a multi-byte
    /// character never arrives cut in half, and the escaping still composes.
    /// Newlines inside the value are content; only the one framing the
    /// terminator is not.
    #[test]
    fn a_streamed_string_value_is_escaped_across_fragments() {
        let run = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (0, "\n<function=get_weather>\n<parameter=city>\nsay \"caf"),
                (0, "é\"\n\tor \\else"),
                (0, "\n</parameter>\n</function>\n"),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        let arguments = run.call().1;
        assert_eq!(
            arguments,
            r#"{"city":"say \"café\"\n\tor \\else"}"#.to_string()
        );
        let parsed: serde_json::Value = serde_json::from_str(&arguments).expect("valid JSON");
        assert_eq!(parsed["city"], "say \"café\"\n\tor \\else");
    }

    /// A name and a key each run to the first `>` and are trimmed, so whitespace
    /// the model padded a tag with is not part of what it named.
    #[test]
    fn whitespace_around_the_markers_is_tolerated() {
        let run = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (0, "\n<function= get_weather >\n"),
                (0, "<parameter= city >\nOslo\n</parameter>\n"),
                (0, "</function>\n"),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert_eq!(
            run.call(),
            ("get_weather".to_string(), r#"{"city":"Oslo"}"#.to_string())
        );
    }

    /// A call with no arguments still produces an object, so a consumer that
    /// parses the concatenated deltas never has to special-case an empty one.
    #[test]
    fn a_call_without_arguments_still_yields_an_object() {
        let run = drive(
            vec![tool("get_time", serde_json::json!({}))],
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (0, "\n<function=get_time>\n</function>\n"),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert_eq!(run.call(), ("get_time".to_string(), "{}".to_string()));
        assert_eq!((run.healed, run.quoted, run.degraded), (0, 0, 0));
    }

    /// Several calls in one turn arrive one after another, never interleaved —
    /// each in its own span, which is how the template writes them.
    #[test]
    fn sequential_calls_stay_separate() {
        let run = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n",
                ),
                (tool_call_close(), "</tool_call>"),
                (0, "\n"),
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=days>\n2\n</parameter>\n</function>\n",
                ),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert_eq!(
            run.calls(),
            vec![
                ("get_weather".to_string(), r#"{"city":"Oslo"}"#.to_string()),
                ("get_weather".to_string(), r#"{"days":2}"#.to_string()),
            ]
        );
        // The newline the template writes between two calls is answer text, and
        // the only answer text there is.
        assert_eq!(run.answer(), "\n");
        assert_eq!((run.healed, run.quoted, run.degraded), (0, 0, 0));
    }

    /// An argument the schema does not describe is classified by what the model
    /// wrote: JSON stays JSON, and anything else is the string it looks like.
    #[test]
    fn an_undeclared_argument_is_classified_by_its_value() {
        let run = drive(
            vec![tool("run", serde_json::json!({}))],
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=run>\n<parameter=payload>\n{\"a\":1}\n</parameter>\n",
                ),
                (
                    0,
                    "<parameter=note>\nhello world\n</parameter>\n</function>\n",
                ),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert_eq!(
            run.call().1,
            r#"{"payload":{"a":1},"note":"hello world"}"#.to_string()
        );
    }

    /// A name the generation cut short names nothing the client can call, so the
    /// span degrades to the text the model wrote — unless the request declared
    /// exactly that name, which is the only evidence available that the model had
    /// finished writing it.
    #[test]
    fn a_call_cut_off_in_its_name_degrades_unless_the_name_is_whole() {
        let truncated = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (0, "\n<function=get_wea"),
            ],
            true,
        );
        assert!(truncated.calls().is_empty());
        // Nothing is lost: what could not be read as a call is delivered as the
        // text it is.
        assert_eq!(truncated.answer(), "<tool_call>\n<function=get_wea");
        assert_eq!(truncated.degraded, 1);
        assert_eq!(truncated.stop, StopKind::EndTurn);

        let whole = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (0, "\n<function=get_weather"),
            ],
            true,
        );
        assert_eq!(whole.call(), ("get_weather".to_string(), "{}".to_string()));
        assert_eq!(whole.stop, StopKind::ToolUse);
        assert_eq!((whole.healed, whole.degraded), (1, 0));

        // A call the model opened and closed without naming anything invokes
        // nothing, and comes back verbatim.
        let nameless = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (0, "\n<function=>\n</function>\n"),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert!(nameless.calls().is_empty());
        assert_eq!(
            nameless.answer(),
            "<tool_call>\n<function=>\n</function>\n</tool_call>"
        );
        assert_eq!(nameless.degraded, 1);
        assert_eq!(nameless.stop, StopKind::EndTurn);
    }

    /// A span that goes straight to its arguments never names anything either.
    /// The arguments prove nothing about a name that is not there, so the whole
    /// span comes back as text — every byte of it, and without disturbing what
    /// follows.
    #[test]
    fn a_span_that_names_nothing_degrades_arguments_and_all() {
        let run = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (0, "\n<parameter=city>\nOslo\n</parameter>\n"),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert!(run.calls().is_empty());
        assert_eq!(
            run.answer(),
            "<tool_call>\n<parameter=city>\nOslo\n</parameter>\n</tool_call>"
        );
        assert_eq!((run.healed, run.degraded), (0, 1));
        assert_eq!(run.stop, StopKind::EndTurn);

        // The degraded span still ends where the model ended it, so the call
        // after it is parsed as its own.
        let followed = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (0, "\n<parameter=city>\nOslo\n</parameter>\n"),
                (tool_call_close(), "</tool_call>"),
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=days>\n2\n</parameter>\n</function>\n",
                ),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert_eq!(
            followed.calls(),
            vec![("get_weather".to_string(), r#"{"days":2}"#.to_string())]
        );
        assert_eq!(followed.stop, StopKind::ToolUse);

        // Healed rather than closed, the same span still gives its text back —
        // this is the case that used to vanish.
        let healed = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (0, "\n<parameter=city>\nOs"),
            ],
            true,
        );
        assert!(healed.calls().is_empty());
        assert_eq!(healed.answer(), "<tool_call>\n<parameter=city>\nOs");
        assert_eq!(healed.degraded, 1);
        assert_eq!(healed.stop, StopKind::EndTurn);
    }

    /// A trailing key whose value never arrived names an argument the model had
    /// not decided on, so it is dropped and the object closed without it. The
    /// value opening is `<parameter=key>` itself, so "cut short mid-key" and
    /// "cut short at the value's first character" are both this case.
    #[test]
    fn a_key_without_a_value_is_dropped_when_the_call_is_healed() {
        for tail in ["<parameter=ci", "<parameter=city>", "<parameter=city>\n"] {
            let run = drive(
                weather(),
                &[],
                &[
                    (tool_call_open(), "<tool_call>"),
                    (0, "\n<function=get_weather>\n"),
                    (0, tail),
                ],
                true,
            );
            assert_eq!(
                run.call(),
                ("get_weather".to_string(), "{}".to_string()),
                "tail {tail:?}"
            );
            assert_eq!(run.healed, 1, "tail {tail:?}");
        }

        // A value the model did write and close as empty is an empty value, not
        // an absent one.
        let empty = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=city>\n\n</parameter>\n</function>\n",
                ),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert_eq!(empty.call().1, r#"{"city":""}"#.to_string());
    }

    /// A value the generation cut short is closed with what did arrive: a string
    /// keeps its quotes, and a value that no longer parses as JSON becomes the
    /// string it literally is, so the deltas still form one object.
    #[test]
    fn a_value_cut_short_is_closed_rather_than_abandoned() {
        let text = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (0, "\n<function=get_weather>\n<parameter=city>\nSan Fran"),
            ],
            false,
        );
        assert_eq!(text.call().1, r#"{"city":"San Fran"}"#.to_string());
        // The cap is the truthful stop even for a turn that called a tool.
        assert_eq!(text.stop, StopKind::MaxTokens);

        // A half-written terminator is a terminator, not the tail of the value.
        let partial_marker = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=city>\nOslo\n</param",
                ),
            ],
            true,
        );
        assert_eq!(partial_marker.call().1, r#"{"city":"Oslo"}"#.to_string());

        // Half an object is not a string the client can use, and calling it one
        // would report an argument of a type the tool never declared, so the
        // pair goes the way a key with no value at all goes.
        let object = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n\
                     <parameter=filters>\n{\"wind\":tr",
                ),
            ],
            true,
        );
        assert_eq!(object.call().1, r#"{"city":"Oslo"}"#.to_string());
    }

    /// A value the model finished writing keeps the parse-or-quote rule: this is
    /// the model producing something the schema did not promise, which is a
    /// different thing from the engine cutting a value in half.
    #[test]
    fn a_completed_value_that_is_not_json_becomes_the_string_it_is() {
        let run = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=filters>\nwindy\n</parameter>\n\
                     </function>\n",
                ),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert_eq!(run.call().1, r#"{"filters":"windy"}"#.to_string());
    }

    /// The structural tags are ordinary BPE text, so a decode step may end in
    /// the middle of one. The parser reassembles them across steps rather than
    /// reading the fragments as a name, a key or a value.
    #[test]
    fn a_tag_split_across_decode_steps_is_reassembled() {
        let split = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (0, "\n<func"),
                (0, "tion=get_wea"),
                (0, "ther>\n<param"),
                (0, "eter=city>\nOslo\n</parameter>\n</function>\n"),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert_eq!(
            split.call(),
            ("get_weather".to_string(), r#"{"city":"Oslo"}"#.to_string())
        );
        assert_eq!((split.healed, split.degraded), (0, 0));
    }

    /// A call the model never terminated does not absorb the one that follows
    /// it: the opening token closes the first call and starts a second.
    #[test]
    fn an_unterminated_call_does_not_swallow_the_next_one() {
        let run = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n",
                ),
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=days>\n2\n</parameter>\n</function>\n",
                ),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert_eq!(
            run.calls(),
            vec![
                ("get_weather".to_string(), r#"{"city":"Oslo"}"#.to_string()),
                ("get_weather".to_string(), r#"{"days":2}"#.to_string()),
            ]
        );
        // The first call was complete but for its `</tool_call>`, which is a
        // repair even though nothing was lost.
        assert_eq!(run.healed, 1);
    }

    /// The `</tool_call>` token is structural wherever it lands, mid-value
    /// included. The template writes a literal `</tool_call>` inside an argument
    /// as ordinary content, so it never encodes to this token — the model
    /// emitting it means the call is over. Reading it as content instead would
    /// let one value swallow the rest of the reply.
    #[test]
    fn a_tool_call_token_inside_a_value_still_closes_the_call() {
        let run = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (0, "\n<function=get_weather>\n<parameter=city>\nsay "),
                (tool_call_close(), "</tool_call>"),
                (0, " loudly"),
            ],
            true,
        );
        assert_eq!(run.call().1, r#"{"city":"say "}"#.to_string());
        // The reply continues instead of being eaten by the open value.
        assert_eq!(run.answer(), " loudly");
        assert_eq!(run.healed, 1);
    }

    /// A union that includes `string` is the common optional-string shape. Only
    /// the union's other members are read as JSON, so a numeric-looking string
    /// stays a string while a literal null comes through as null.
    #[test]
    fn a_nullable_string_keeps_a_numeric_looking_value_a_string() {
        let tools = vec![tool(
            "note",
            serde_json::json!({
                "label": {"type": ["string", "null"]},
                "count": {"type": ["string", "number"]},
                "mode": {"enum": ["fast", "slow"]},
            }),
        )];
        let run = drive(
            tools,
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=note>\n<parameter=label>\n123\n</parameter>\n",
                ),
                (0, "<parameter=count>\n123\n</parameter>\n"),
                (0, "<parameter=mode>\nfast\n</parameter>\n</function>\n"),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        // label declares only null besides string, so 123 is the string it was
        // written as; count declares number, so the same text is the number.
        assert_eq!(
            run.call().1,
            r#"{"label":"123","count":123,"mode":"fast"}"#.to_string()
        );

        let null = drive(
            vec![tool(
                "note",
                serde_json::json!({"label": {"type": ["string", "null"]}}),
            )],
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=note>\n<parameter=label>\nnull\n</parameter>\n</function>\n",
                ),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert_eq!(null.call().1, r#"{"label":null}"#.to_string());
    }

    /// The holdback at the end of a generation is the client's text, except for
    /// a half-written `</assistant>` — the engine added that sequence, so the
    /// start of it is no more the client's than a whole match would be.
    #[test]
    fn a_half_written_internal_stop_is_not_delivered() {
        let run = drive(weather(), &[], &[(0, "bye</assist")], true);
        assert_eq!(run.answer(), "bye");

        // Held on a client sequence's account as well, it is text the client
        // would have received either way.
        let shared = drive(weather(), &["</assist-me"], &[(0, "bye</assist")], true);
        assert_eq!(shared.answer(), "bye</assist");

        // Without tools there is no internal sequence, so nothing is dropped.
        let toolless = drive(Vec::new(), &[], &[(0, "bye</assist")], true);
        assert_eq!(toolless.answer(), "bye</assist");
    }

    /// A stop sequence firing inside a call would deliver a call nothing can

    /// A stop sequence firing inside a call would deliver a call nothing can
    /// parse, so client stop sequences do not run until the call has closed.
    #[test]
    fn client_stop_sequences_are_suspended_inside_a_call() {
        let run = drive(
            weather(),
            &["STOP"],
            &[
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=city>\nSTOP HERE\n</parameter>\n\
                     </function>\n",
                ),
                (tool_call_close(), "</tool_call>"),
                (0, "now STOP tail"),
            ],
            true,
        );
        assert_eq!(run.call().1, r#"{"city":"STOP HERE"}"#.to_string());
        // Outside the call the sequence applies as usual: the match and
        // everything after it is withheld.
        assert_eq!(run.answer(), "now ");
        // The turn called a tool, which outranks the sequence that ended it.
        assert_eq!(run.stop, StopKind::ToolUse);
    }

    /// After a call the model often writes `</assistant>` as ordinary text
    /// rather than emitting token 24. For a job with tools that ends the turn —
    /// and it is the engine's own sequence, so it is never reported as one.
    #[test]
    fn a_spelled_out_assistant_close_ends_a_turn_that_has_tools() {
        let run = drive(
            weather(),
            &[],
            &[(0, "All done."), (0, "</assistant>"), (0, "never sent")],
            false,
        );
        assert_eq!(run.answer(), "All done.");
        assert_eq!(run.stop, StopKind::EndTurn);

        // With a call behind it, the same ending is a request for tool results.
        let after_call = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n",
                ),
                (tool_call_close(), "</tool_call>"),
                (0, "</assistant>"),
            ],
            false,
        );
        assert_eq!(after_call.calls().len(), 1);
        assert_eq!(after_call.stop, StopKind::ToolUse);

        // A client that asked for the sequence itself is told it matched.
        let asked_for_it = drive(
            weather(),
            &["</assistant>"],
            &[(0, "text"), (0, "</assistant>")],
            false,
        );
        assert_eq!(
            asked_for_it.stop,
            StopKind::StopSequence("</assistant>".to_string())
        );
    }

    /// Without tools in the request there is nothing to call, so the tool tokens
    /// and `</assistant>` are ordinary text and the answer keeps them.
    #[test]
    fn without_tools_the_markers_are_plain_text() {
        let run = drive(
            Vec::new(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (0, "\n<function=get_weather>\n</function>\n"),
                (tool_call_close(), "</tool_call>"),
                (0, "</assistant>"),
            ],
            true,
        );
        assert!(run.calls().is_empty());
        assert_eq!(
            run.answer(),
            "<tool_call>\n<function=get_weather>\n</function>\n</tool_call></assistant>"
        );
        assert_eq!(run.stop, StopKind::EndTurn);

        // With tools, a close marker that closes nothing is still just text.
        let stray = drive(
            weather(),
            &[],
            &[(0, "done"), (tool_call_close(), "</tool_call>")],
            true,
        );
        assert!(stray.calls().is_empty());
        assert_eq!(stray.answer(), "done</tool_call>");
    }

    /// Text the model wrote before opening a call is delivered as answer text,
    /// including a holdback that turned out not to be a stop sequence.
    #[test]
    fn text_before_a_call_is_settled_when_the_call_opens() {
        let run = drive(
            weather(),
            &["STOP"],
            &[
                (0, "Let me check. ST"),
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n",
                ),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert_eq!(run.answer(), "Let me check. ST");
        assert_eq!(run.calls().len(), 1);
    }

    /// A request whose calls all arrived as written reports nothing at all: the
    /// counters are the malformation evidence, so a clean run has to leave them
    /// at zero for a dirty one to mean anything.
    #[test]
    fn a_clean_run_has_nothing_to_report() {
        let run = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n\
                     <parameter=days>\n2\n</parameter>\n</function>\n",
                ),
                (tool_call_close(), "</tool_call>"),
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=filters>\n{\"wind\":true}\n\
                     </parameter>\n</function>\n",
                ),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert_eq!(run.calls().len(), 2);
        assert_eq!((run.healed, run.quoted, run.degraded), (0, 0, 0));
        assert!(malformation_report(run.healed, run.quoted, run.degraded).is_none());
    }

    /// A call the model never finished is one heal, whatever the repair had to
    /// throw away to close it.
    #[test]
    fn a_repaired_call_is_counted_as_evidence() {
        let truncated = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (0, "\n<function=get_weather>\n<parameter=city>\nOs"),
            ],
            false,
        );
        assert_eq!(
            (truncated.healed, truncated.quoted, truncated.degraded),
            (1, 0, 0)
        );

        // A heal whose repair dropped the pair outright still counts as one.
        let dropped_pair = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=filters>\n{\"wind\":tr",
                ),
            ],
            true,
        );
        assert_eq!(dropped_pair.call().1, "{}".to_string());
        assert_eq!(
            (
                dropped_pair.healed,
                dropped_pair.quoted,
                dropped_pair.degraded
            ),
            (1, 0, 0)
        );

        // So does a call the next call's opening token had to close.
        let unterminated = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n",
                ),
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=days>\n2\n</parameter>\n</function>\n",
                ),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert_eq!(unterminated.calls().len(), 2);
        assert_eq!(
            (
                unterminated.healed,
                unterminated.quoted,
                unterminated.degraded
            ),
            (1, 0, 0)
        );

        // A call the model closed without writing `</function>` is delivered,
        // and still counts: the parser had to decide where it ended.
        let no_function_close = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n",
                ),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert_eq!(no_function_close.call().1, r#"{"city":"Oslo"}"#.to_string());
        assert_eq!(no_function_close.healed, 1);
    }

    /// A finished value that is not the JSON its schema asked for is the other
    /// half of the evidence — but a union that also declares `string` resolving
    /// to the string is the schema being honored, and counts as nothing.
    #[test]
    fn only_a_missed_json_value_counts_as_parse_or_quote() {
        let missed = drive(
            weather(),
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=get_weather>\n<parameter=days>\nsoon\n</parameter>\n\
                     </function>\n",
                ),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert_eq!(missed.call().1, r#"{"days":"soon"}"#.to_string());
        assert_eq!((missed.healed, missed.quoted, missed.degraded), (0, 1, 0));
        assert_eq!(
            malformation_report(missed.healed, missed.quoted, missed.degraded)
                .and_then(|log| log.render())
                .as_deref(),
            Some("xwen serve: tool spans: 1 parse-or-quote")
        );

        let nullable = drive(
            vec![tool(
                "note",
                serde_json::json!({"label": {"type": ["string", "null"]}}),
            )],
            &[],
            &[
                (tool_call_open(), "<tool_call>"),
                (
                    0,
                    "\n<function=note>\n<parameter=label>\n123\n</parameter>\n</function>\n",
                ),
                (tool_call_close(), "</tool_call>"),
            ],
            true,
        );
        assert_eq!(nullable.call().1, r#"{"label":"123"}"#.to_string());
        assert_eq!(
            (nullable.healed, nullable.quoted, nullable.degraded),
            (0, 0, 0)
        );

        // Every kind at once is reported together.
        assert_eq!(
            malformation_report(2, 1, 1)
                .and_then(|log| log.render())
                .as_deref(),
            Some("xwen serve: tool spans: 2 healed, 1 parse-or-quote, 1 degraded to text")
        );
    }

    /// Tokenize an assistant turn the way the model would have produced it, one
    /// `(id, text)` pair per token, ready for [`drive`].
    ///
    /// `chat::build_prompt` renders the turn and `LagunaTokenizer::encode` maps
    /// the structural markers to their added-token ids, so what comes back is
    /// the real vocabulary's real ids — not a hand-written stand-in that can
    /// agree with a parser while both disagree with the model.
    fn tokenize_assistant_turn(calls: Vec<crate::chat::ToolCall>) -> Vec<(u32, String)> {
        let rendered = crate::chat::build_prompt(
            &[
                crate::chat::Message::User("what is the weather?".to_string()),
                crate::chat::Message::Assistant {
                    content: String::new(),
                    reasoning: None,
                    tool_calls: calls,
                },
            ],
            &crate::chat::ChatOptions {
                enable_thinking: false,
                preserve_thinking: false,
                tools: Vec::new(),
                ..Default::default()
            },
        )
        .expect("the template renders");
        // Just the calls: from the first `<tool_call>` to the last
        // `</tool_call>`, without the turn header or the `<|im_end|>`.
        let start = rendered
            .find(TOOL_CALL_OPEN_TEXT)
            .expect("a call was rendered");
        let end = rendered
            .rfind(TOOL_CALL_CLOSE_TEXT)
            .expect("a call was closed")
            + TOOL_CALL_CLOSE_TEXT.len();
        let tokenizer = LagunaTokenizer::embedded().expect("the embedded tokenizer");
        tokenizer
            .encode(&rendered[start..end])
            .expect("the rendered calls tokenize")
            .into_iter()
            .map(|id| {
                let text = tokenizer.decode(&[id]).expect("every id decodes");
                (id, text)
            })
            .collect()
    }

    fn chat_call(name: &str, arguments: Vec<(&str, serde_json::Value)>) -> crate::chat::ToolCall {
        crate::chat::ToolCall {
            name: name.to_string(),
            arguments: arguments
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
        }
    }

    /// What `chat.rs` writes, this parser reads back — over the real tokenizer's
    /// real ids, so the two halves of the wire format are pinned to each other
    /// rather than each to a hand-written idea of the other.
    #[test]
    fn a_rendered_call_round_trips_through_the_real_tokenizer() {
        let tools = vec![tool(
            "get_weather",
            serde_json::json!({
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "note": {"type": "string"},
                "filters": {"type": "object"},
            }),
        )];

        let one = drive(
            tools.clone(),
            &[],
            &tokenize_assistant_turn(vec![chat_call(
                "get_weather",
                vec![
                    ("city", serde_json::json!("San Francisco")),
                    ("days", serde_json::json!(3)),
                    // A multiline string, which the framing newlines must not
                    // eat into, and a nested object.
                    ("note", serde_json::json!("first line\nsecond line")),
                    (
                        "filters",
                        serde_json::json!({"wind": true, "depth": [1, 2]}),
                    ),
                ],
            )]),
            true,
        );
        let (name, arguments) = one.call();
        assert_eq!(name, "get_weather");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&arguments).expect("valid JSON"),
            serde_json::json!({
                "city": "San Francisco",
                "days": 3,
                "note": "first line\nsecond line",
                "filters": {"wind": true, "depth": [1, 2]},
            })
        );
        assert_eq!((one.healed, one.quoted, one.degraded), (0, 0, 0));

        // Two calls in one turn, as the template writes parallel tool use.
        let two = drive(
            tools,
            &[],
            &tokenize_assistant_turn(vec![
                chat_call("get_weather", vec![("city", serde_json::json!("Paris"))]),
                chat_call(
                    "get_weather",
                    vec![
                        ("city", serde_json::json!("Rome")),
                        ("days", serde_json::json!(2)),
                    ],
                ),
            ]),
            true,
        );
        assert_eq!(
            two.calls(),
            vec![
                ("get_weather".to_string(), r#"{"city":"Paris"}"#.to_string()),
                (
                    "get_weather".to_string(),
                    r#"{"city":"Rome","days":2}"#.to_string()
                ),
            ]
        );
        assert_eq!((two.healed, two.quoted, two.degraded), (0, 0, 0));
    }

    /// Prose is prose. A span is opened only by the `<tool_call>` token, so text
    /// describing the call format — `<function=`, `</parameter>` and all — is
    /// answer text, and reaches the client byte for byte.
    ///
    /// This is the regression test for the constants: `:` and `;` are ids 25 and
    /// 26, which the parser once used as its span markers. Under that bug this
    /// prose opened and closed phantom spans on its own punctuation, truncating
    /// the reply and inventing calls out of it.
    #[test]
    fn prose_that_looks_like_a_call_is_never_one() {
        let tokenizer = LagunaTokenizer::embedded().expect("the embedded tokenizer");
        let prose = "Here is the format: a call opens with <function=name>; inside it \
                     <parameter=key> holds a value, and </parameter> then </function> \
                     close it. Ratios: 3:1; 4:2; note the colons and semicolons.";
        let ids = tokenizer.encode(prose).expect("the prose tokenizes");
        // The punctuation this test is about really is in the stream, and the
        // span tokens really are not.
        assert!(
            ids.contains(&25) && ids.contains(&26),
            "no `:` or `;` in {ids:?}"
        );
        assert!(
            !ids.contains(&tool_call_open()) && !ids.contains(&tool_call_close()),
            "prose must not carry the span tokens"
        );
        let script: Vec<(u32, String)> = ids
            .into_iter()
            .map(|id| (id, tokenizer.decode(&[id]).expect("every id decodes")))
            .collect();

        let run = drive(weather(), &[], &script, true);
        assert!(
            run.calls().is_empty(),
            "prose is not a call: {:?}",
            run.calls()
        );
        assert_eq!(run.answer(), prose);
        assert_eq!((run.healed, run.quoted, run.degraded), (0, 0, 0));
    }

    /// A tool call outranks every ending but the output cap, which stays what it
    /// is: the reply was cut short, healed calls or not.
    #[test]
    fn a_turn_that_called_a_tool_stops_on_tool_use() {
        let sequence = || Some("STOP".to_string());
        assert_eq!(terminal_stop(None, None, true, true), StopKind::ToolUse);
        assert_eq!(terminal_stop(None, None, true, false), StopKind::MaxTokens);
        assert_eq!(terminal_stop(None, None, false, true), StopKind::EndTurn);
        assert_eq!(
            terminal_stop(sequence(), None, false, true),
            StopKind::StopSequence("STOP".to_string())
        );
        assert_eq!(
            terminal_stop(sequence(), None, true, true),
            StopKind::ToolUse
        );
        // The engine's own sequence is an end of turn, never a reported match.
        assert_eq!(
            terminal_stop(
                Some(ASSISTANT_CLOSE.to_string()),
                Some(ASSISTANT_CLOSE),
                false,
                false
            ),
            StopKind::EndTurn
        );
    }

    #[test]
    fn context_length_is_capped_at_the_checkpoints_own() {
        let (ctx, warning) = resolve_context_length(4096, 262144).unwrap();
        assert_eq!(ctx, 4096);
        assert!(warning.is_none(), "a context that fits is served as asked");
        let (ctx, warning) = resolve_context_length(1_000_000, 262144).unwrap();
        assert_eq!(ctx, 262144);
        assert!(
            warning
                .and_then(|w| w.render())
                .is_some_and(|w| w.contains("262144"))
        );
        assert!(resolve_context_length(0, 262144).is_err());
    }

    /// The shutdown flush waits for what it actually has queued. The fixed grace
    /// it replaced was sized on one ~4.2 GiB image; a long conversation images
    /// several times that, and a shutdown that splits a stored segment queues
    /// more than one image at once.
    #[test]
    fn the_disk_flush_budget_grows_with_what_is_queued() {
        // Nothing pending, or an image small enough to land inside the grace:
        // the grace is the floor and the answer.
        assert_eq!(disk_flush_budget(0), DISK_FLUSH_GRACE);
        assert_eq!(disk_flush_budget(4 * 1024 * 1024 * 1024), DISK_FLUSH_GRACE);
        // A 32 GiB queue is past it, and gets the time its own bytes need.
        let big = 32u64 * 1024 * 1024 * 1024;
        assert!(disk_flush_budget(big) > DISK_FLUSH_GRACE);
        assert_eq!(
            disk_flush_budget(big),
            Duration::from_secs(big / DISK_WRITE_FLOOR_BYTES_PER_SEC)
        );
        // Monotone in the queue size, which is the property the caller relies on.
        let mut last = Duration::ZERO;
        for gib in [0u64, 1, 4, 16, 64, 256] {
            let now = disk_flush_budget(gib * 1024 * 1024 * 1024);
            assert!(now >= last, "{gib} GiB went backwards");
            last = now;
        }
    }
}
