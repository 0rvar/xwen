//! Batch generation over one shared prompt prefix.
//!
//! A batch request holds N chat items that agree for most of their prompt — the
//! same system message and the same document, differing only in the question at
//! the end. Prefilling that prefix once per item is the whole cost of such a
//! run, so the batch runner prefills it ONCE, snapshots the KV cache there, and
//! replays the snapshot per item: restore, prefill the item's own tail, decode.
//!
//! The shared span is the literal longest common prefix of the items' TOKEN
//! vectors, held one token short of the shortest item so every item still has a
//! tail to prefill (a restore rewinds the cache to the snapshot's position, and
//! writing below that position would invalidate the snapshot for every later
//! item — `XwenModel::restore_cache_snapshot`). A prefix shorter than
//! [`MIN_SHARED_PREFIX`] is not worth a snapshot, and then every item runs from a
//! reset cache instead.
//!
//! The request may also DECLARE the shared text once — `shared_prefix`, which
//! the runner prepends to every item's first message before rendering. That
//! changes nothing about the paragraph above (the prefill dedup is token-level
//! and automatic); it exists so the request body does not carry a large shared
//! document once per item.
//!
//! Items run sequentially in request order. Failures are per item wherever they
//! can be: a prompt that will not render, a schema the grammar compiler rejects,
//! a decode that errors — each lands as an `error` on that item's response and
//! the batch carries on, because the whole point is that one bad item must not
//! cost the other N-1 their prefill.
//!
//! Two defaults deliberately differ from the chat surface. Sampling is GREEDY
//! (temperature 0) unless a request overrides it, and thinking is OFF unless a
//! request asks for it: batch items are structured-extraction jobs with tight
//! token budgets, where a reasoning block the caller did not ask for consumes
//! the entire budget before the answer starts.
//!
//! `XWEN_BATCH_NO_CACHE` (any value) disables the snapshot path, running every
//! item from a reset cache. It exists so replay can be A/B'd against scratch on
//! the same request.
//!
//! Moving cache state is therefore ordinary here, not an optimization a target
//! can opt out of — `XWEN_BATCH_NO_CACHE` skips only the shared prefix, while a
//! scored field snapshots per option regardless. So a checkpoint whose state no
//! cache image carries could not run a batch at all; every registry checkpoint
//! carries its state as of 2026-08-30 ([`Model::servable`], which gates this
//! surface and `xwen serve` together).
//!
//! A schema may instead ask for its fields to be SCORED: annotate an enum or
//! boolean property with `include_score` and that item stops free-decoding
//! altogether. The runner writes the JSON itself — the structural skeleton is
//! teacher-forced into the model's context and each field's value is chosen by
//! scoring every allowed option's full token sequence against the model, rather
//! than by drawing tokens under a grammar mask. The answer is the same shape
//! either way; what the scored path adds is the model's confidence in it, and
//! what it costs is one forward per option token instead of one per answer
//! token. See [`ScoredPlan`] and the v1 shape guard in [`scored_fields`].
//!
//! The two arms decode the same tokens but are NOT bit-identical, and reading a
//! difference as a snapshot bug would be wrong. Replay prefills an item's tail
//! as its own short span, and a span below `ops::mm_id_min_seq` takes the MoE's
//! per-token `mv_id` matmul where the cold arm's one long span takes `mm_id` —
//! two kernels of the same math at different precision, which flips a near-tie
//! now and then (measured 2026-08-09: one of four items chose `{"label"` over
//! `{\n  "label"`, the same value either way). Forcing both arms onto one
//! kernel with `XWEN_MM_ID_MIN_SEQ=1` makes them byte-identical, which is how
//! that was settled. It is the same precision class the serve prefix cache has
//! always had.

use std::cell::Cell;
use std::ops::Range;
use std::time::Instant;

use anyhow::{Result, anyhow, bail, ensure};
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::chat::{self, ChatDialect, ChatOptions, Continuation, Message, ReasoningEffort};
use crate::constrain::{self, ConstraintFactory, GrammarState};
use crate::generate::{GenEvent, Generator};
use crate::hub::Model;
use crate::kv_cache::CacheSnapshot;
use crate::sampler::SamplerOptions;
use crate::tokenizer::{LagunaTokenizer, Specials};

/// Shortest shared prefix worth a snapshot. Below this the snapshot/restore
/// bookkeeping costs more than re-prefilling the tokens it would save.
pub const MIN_SHARED_PREFIX: usize = 64;

/// Token budget for an item that names none, and for a request whose defaults
/// name none.
pub const DEFAULT_MAX_TOKENS: usize = 512;

/// Batch sampling before any request setting: greedy, so a batch is
/// reproducible and two runs of the same request can be compared token for
/// token. `top_k`/`top_p`/`seed` are inert at temperature 0 and carry the
/// values a request would land on if it raised the temperature alone.
///
/// `presence_penalty` is NOT one of these: it is not inert at temperature 0
/// (greedy takes the argmax of the penalized row), and unlike the three above
/// its default is the checkpoint's own card value for the item's thinking
/// mode. [`resolve_sampling`] fills it in from there — for an UNCONSTRAINED item;
/// a grammar-masked one defaults to 0, for the reason given there. A batch stays
/// reproducible either way because the penalty is a deterministic function of
/// what the reply has already emitted.
pub const BATCH_SAMPLING: SamplerOptions = SamplerOptions {
    temperature: 0.0,
    top_k: 20,
    top_p: 0.95,
    presence_penalty: 0.0,
    seed: 42,
};

// ---------------------------------------------------------------- request ---

/// One batch request, as it arrives on stdin.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BatchRequest {
    /// Which checkpoint to run — a full name ("Qwen3.6-35B-A3B"), or on the CLI
    /// also a short alias ("35b"). The model comes from the payload rather than
    /// a flag: one request is one model's work. Over HTTP the full name is the
    /// only spelling accepted; see `serve::batch`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Text prepended verbatim to the content of every item's FIRST message.
    ///
    /// Purely a wire-size measure. The runner already prefills the items'
    /// shared TOKEN prefix once however it arrived, but a batch whose items
    /// share a large document otherwise repeats it per item in the request
    /// body — the one place the repetition still costs something. The prompts
    /// this produces are byte-identical to spelling the document out in every
    /// item, so answers (and scores) are too. An item with no messages cannot
    /// take the prefix and fails as an item; an empty string means absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_prefix: Option<String>,
    /// Per-item settings every item inherits unless it names its own.
    #[serde(default)]
    pub defaults: ItemDefaults,
    pub items: Vec<BatchItem>,
}

/// The settings an item inherits. Every field is optional twice over: absent
/// here means the module default, absent on the item means whatever this says.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ItemDefaults {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
}

/// One item: a whole conversation, its output shape, and the knobs that differ
/// from the batch defaults.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BatchItem {
    /// Echoed on the response so a caller can match answers to questions
    /// without relying on order. Not required to be unique.
    pub id: String,
    pub messages: Vec<BatchMessage>,
    /// JSON schema the answer is constrained to. Absent decodes free text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingSpec>,
    /// The 3.8 template's `reasoning_effort` level, as its own spelling
    /// (`"low"` / `"medium"` / `"xhigh"`). Read by the template only with
    /// thinking on; refused on a 3.6 checkpoint, whose template has no such
    /// parameter. Absent, the template's default (`xhigh`) stands.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Assistant text the answer continues from, rendered into the generation
    /// header. It may not open with a newline (see [`Continuation::prefix`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingSpec>,
}

/// A conversation turn. `thinking` is an assistant turn's reasoning, replayed
/// verbatim into the prompt.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BatchMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
}

/// Sampling overrides. An absent field keeps whatever the layer below settled
/// on, so an item can raise the temperature without restating the rest.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    /// Subtracted from the logit of every token the item's reply has already
    /// produced, once per distinct token. Absent takes the checkpoint's card
    /// value for the item's resolved thinking mode, not the batch default
    /// above — see [`BATCH_SAMPLING`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// How an item reasons: off, on, or on with the reasoning supplied by the
/// caller — which is a string on the wire, so `false` / `true` / `"..."` are all
/// legal values of the one field.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ThinkingSpec {
    Enabled(bool),
    /// Reasoning to place inside the turn's `<think>` block, which is then
    /// closed so decoding starts in the answer.
    Injected(String),
}

impl BatchRequest {
    /// The checkpoint this request names, or the default one when it names none
    /// — [`Model::default_servable`], the same zero-flag default `xwen serve`
    /// resolves. Batch moves cache state exactly as the server does: the shared
    /// prefix is snapshotted here and every enum-scored field snapshots and
    /// restores around each option, so the two surfaces answer to the same rule
    /// and resolve their default through the same function. Every registry
    /// checkpoint clears it as of 2026-08-30, which is why that rule now agrees
    /// with the plain [`Model::default`].
    ///
    /// The server's batch route resolves against the checkpoint it is SERVING
    /// instead and rewrites the field before the runner sees it
    /// (`serve::batch`), so this default is never the answer there.
    pub fn model(&self) -> Result<Model> {
        match &self.model {
            Some(name) => name.parse().map_err(|e: String| anyhow!("batch: {e}")),
            None => Ok(Model::default_servable()),
        }
    }
}

// --------------------------------------------------------------- response ---

/// The whole answer, printed as JSON on stdout.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct BatchResponse {
    pub model: String,
    pub items: Vec<ItemResponse>,
    pub stats: BatchStats,
}

/// One item's answer. Present in request order, including for items that
/// failed — a failed item carries `error` and empty text, so the response is
/// always parallel to the request.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ItemResponse {
    pub id: String,
    /// Everything the model emitted, reasoning included. The `</think>` marker
    /// itself is not part of the stream (the decoder strips it), so this is the
    /// reasoning text followed by the answer text.
    pub content: String,
    /// The answer alone: `content` minus any reasoning. A `prefill` is prompt
    /// rather than output, so it is NOT part of this — only what the model
    /// wrote after it.
    pub text: String,
    /// The constrained value, for an item that carried a schema; null for an
    /// unconstrained one. This is the whole document, so an item that put the
    /// opening of its value in `prefill` finds it here even though `text` holds
    /// only the continuation.
    ///
    /// A scored item's value is assembled rather than parsed, and its annotated
    /// fields report as objects: `{"value": …, "score": …}`, plus
    /// `{"scores": {…}, "escape": …}` where the schema asked for `"all"`. Its
    /// unannotated fields stay bare values.
    pub json: Option<Value>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    /// Why this item produced nothing. Absent on success.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model stopped on its own: an EOG token, or a constrained value that
    /// completed.
    Stop,
    /// The item's `max_tokens` ran out first.
    Length,
    /// The item failed; see `error`.
    Error,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize, Serialize)]
pub struct Usage {
    /// Tokens in the rendered prompt, shared prefix included.
    pub prompt_tokens: usize,
    /// How many of those came out of the snapshot rather than a prefill.
    pub cached_prefix_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize, Serialize)]
pub struct BatchStats {
    /// Length of the shared prefix that was prefilled once, or 0 when every
    /// item ran from a reset cache.
    pub shared_prefix_tokens: usize,
    /// Wall time of the shared prefill plus taking the snapshot.
    pub snapshot_ms: f64,
    pub items: usize,
    /// Tokens forwarded as prefill across the whole batch, and the wall time
    /// they took: every item's tail, and — for scored items — every
    /// teacher-forced segment and option trial. This measures engine WORK, so
    /// on a scored batch it exceeds the sum of the items' logical prompt sizes.
    ///
    /// The shared prefix is NOT among them: it is prefilled before this
    /// accounting opens, and its tokens and time are `shared_prefix_tokens` and
    /// `snapshot_ms`. A caller reporting the whole prefill phase has to add both
    /// halves — see `batch_run_record` in the CLI and the engine's `BatchSummary`.
    pub prefill_tokens: usize,
    pub prefill_ms: f64,
    /// Tokens the items actually DECODED (free text, grammar-constrained
    /// answers, reasoning), and the wall time. A fully scored batch decodes
    /// only its items' reasoning — with thinking off, nothing at all — because
    /// assembled answer tokens are teacher-forced, not sampled: they count in
    /// `completion_tokens`, and their cost lives in the prefill figures.
    pub decode_tokens: usize,
    pub decode_ms: f64,
    /// Model + drafter load time, measured by the caller that loaded them.
    pub load_ms: f64,
    pub total_ms: f64,
}

// -------------------------------------------------------------- execution ---

/// An item rendered, encoded and validated: everything needed to find the
/// shared prefix, and everything [`run_item`] needs to run it.
struct Prepared {
    tokens: Vec<u32>,
    /// Trailing tokens of `tokens` that render the response prefix. The grammar
    /// has to consume them before the first draw, or it would open a second
    /// document instead of continuing this one.
    prefix_len: usize,
    /// The response prefix as it was rendered, empty when the item supplied
    /// none. Decoding continues it without re-emitting it, so it is the missing
    /// head of any constrained value the item produces.
    prefix_text: String,
    starts_in_thinking: bool,
    max_tokens: usize,
    sampling: SamplerOptions,
    /// The schema the grammar path compiles, `None` for an unconstrained item
    /// AND for a scored one — a scored schema carries `include_score`, which
    /// llguidance has never heard of, and the scored path constrains nothing
    /// anyway.
    schema: Option<Value>,
    /// The assembly plan, for an item whose schema asked for scoring.
    scored: Option<ScoredPlan>,
}

/// What one item's run produced.
struct ItemOutcome {
    content: String,
    text: String,
    finish_reason: FinishReason,
    completion_tokens: usize,
    /// Tokens this item DECODED and the wall time it took, from the decode
    /// loop's own outcome. Smaller than `completion_tokens` on a scored item
    /// (assembled tokens are teacher-forced, and only reasoning decodes) and
    /// zero for one that never reached a decode. Prefill has no per-item
    /// counterpart: the runner reads the generator's cumulative spend instead,
    /// which also covers the shared prefill no single item owns.
    decode_tokens: usize,
    decode_secs: f64,
    /// The value a SCORED item assembled. The grammar path leaves this `None`
    /// and its value is parsed back out of the decoded text instead.
    json: Option<Value>,
    /// Why this item has no usable answer even though the run itself did not
    /// fail. The scored path's budget refusal is the one case: it reports as
    /// `length`, like any other answer the token budget cut short.
    error: Option<String>,
}

/// True when the snapshot path is disabled and every item is to run from a
/// reset cache (`XWEN_BATCH_NO_CACHE`).
fn cache_disabled() -> bool {
    std::env::var_os("XWEN_BATCH_NO_CACHE").is_some()
}

/// Length of the longest prefix every sequence shares. Zero for an empty set —
/// there is no sequence to take a prefix of.
pub fn longest_common_prefix(seqs: &[&[u32]]) -> usize {
    let Some((first, rest)) = seqs.split_first() else {
        return 0;
    };
    let mut len = first.len();
    for seq in rest {
        len = len.min(seq.len());
        len = first[..len]
            .iter()
            .zip(seq.iter())
            .take_while(|(a, b)| a == b)
            .count();
        if len == 0 {
            break;
        }
    }
    len
}

/// The prefix worth prefilling once and snapshotting, or `None` to run every
/// item from a reset cache.
///
/// Two rules shape it. It is held one token short of the shortest sequence, so
/// every item has a tail to prefill: an item whose whole prompt is the shared
/// prefix would have to re-prefill a token BELOW the snapshot position, which
/// invalidates the snapshot for every item after it. And it has to clear
/// [`MIN_SHARED_PREFIX`], below which the snapshot costs more than it saves. A
/// lone item shares its prompt with nobody, so it never takes this path.
pub fn shared_prefix_len(seqs: &[&[u32]]) -> Option<usize> {
    if seqs.len() < 2 {
        return None;
    }
    let shortest = seqs.iter().map(|s| s.len()).min().unwrap_or(0);
    let len = longest_common_prefix(seqs).min(shortest.saturating_sub(1));
    (len >= MIN_SHARED_PREFIX).then_some(len)
}

/// Progress worth narrating while a batch runs. The runner reports it through
/// [`BatchHooks::progress`] rather than writing anywhere itself, so the CLI can
/// keep its stderr lines and the server can route the same facts into its log.
#[derive(Debug, Clone, PartialEq)]
pub enum BatchProgress {
    /// The shared prefix was prefilled once and snapshotted.
    SharedPrefix { tokens: usize, ms: f64 },
    /// One item ran to completion.
    Item {
        id: String,
        completion_tokens: usize,
        ms: f64,
    },
}

/// What the caller threads into a batch run: a progress sink, and a poll that
/// says the run has been abandoned. Cancellation is polled between items and
/// once per decoded token inside an item, so a cancelled batch stops within a
/// token of the signal; items never run after it, and report `error` instead.
/// The scored path checks only between items — it teacher-forces short forced
/// spans rather than decoding, so an item's worth of latency bounds it anyway.
pub struct BatchHooks<'a> {
    pub progress: &'a mut dyn FnMut(BatchProgress),
    pub cancelled: &'a mut dyn FnMut() -> bool,
}

/// What a cancelled batch reports on every item the cancellation reached: the
/// one it interrupted and every one it kept from running at all.
const CANCELLED: &str = "the batch was cancelled before this item completed";

/// Run a whole batch on `generator`, which must already be loaded (and, if the run
/// wants speculation, have its drafter attached). `load_ms` is what that load
/// cost, for the stats block.
///
/// `label` is the name the response document reports as its model. The caller
/// passes it rather than the runner re-deriving it from the request, because the
/// caller is the one that resolved which weights are actually loaded: over HTTP
/// that is the id the server answers under, which for a GGUF that is none of the
/// official checkpoints is its own file name and not a checkpoint's.
///
/// `model` is the loaded checkpoint, passed by the caller for the same reason
/// `label` is: the caller is the one that resolved which weights are actually
/// loaded, and a custom GGUF runs as its architecture's checkpoint. It decides
/// two things here — the chat dialect every item's conversation renders under,
/// and the card presence penalty an item that names none samples with.
pub fn run_batch(
    generator: &mut Generator,
    req: &BatchRequest,
    load_ms: f64,
    label: &str,
    model: Model,
    hooks: &mut BatchHooks<'_>,
) -> Result<BatchResponse> {
    let started = Instant::now();
    ensure!(!req.items.is_empty(), "batch: the request holds no items");
    let max_ctx = generator.max_ctx();

    // The generator may be long-lived (the server's engine) and arrive with
    // another job's thinking controls still armed. Batch items reason under
    // their own `thinking` spec and this surface has no budget knob, so both
    // controls are cleared rather than inherited: a stale ceiling would fail
    // every item whose budget it does not fit, and silently truncate the
    // reasoning of the ones it does. The CLI's fresh process has them at zero
    // already; this is for any caller that reuses a generator.
    generator.set_min_think(0);
    generator.set_max_think(0)?;

    // Every prompt is rendered and encoded before anything runs: the shared
    // prefix is a fact about the whole set, and an item that cannot be prepared
    // must not contribute a token vector to it.
    let shared_text = req.shared_prefix.as_deref().filter(|text| !text.is_empty());
    let prepared: Vec<std::result::Result<Prepared, String>> = req
        .items
        .iter()
        .map(|item| {
            prepare_item(
                generator.tokenizer(),
                max_ctx,
                item,
                &req.defaults,
                shared_text,
                label,
                model,
            )
            .map_err(|error| format!("{error:#}"))
        })
        .collect();

    let live: Vec<&[u32]> = prepared
        .iter()
        .filter_map(|p| p.as_ref().ok().map(|p| p.tokens.as_slice()))
        .collect();
    let shared = if cache_disabled() {
        None
    } else {
        shared_prefix_len(&live)
    };

    // The prefix is taken from the first live item, and by construction every
    // other live item agrees with it token for token.
    let mut snapshot_ms = 0.0;
    let snapshot = match shared {
        Some(len) => {
            let prefix: Vec<u32> = live[0][..len].to_vec();
            let prefill_started = Instant::now();
            generator.reset_cache()?;
            generator.reset_drafter()?;
            generator.prefill_tokens(&prefix, 0)?;
            let snapshot = generator.take_cache_snapshot()?;
            snapshot_ms = elapsed_ms(prefill_started);
            (hooks.progress)(BatchProgress::SharedPrefix {
                tokens: len,
                ms: snapshot_ms,
            });
            Some(snapshot)
        }
        None => None,
    };
    let shared_len = shared.unwrap_or(0);

    // Built once for the whole batch: the token trie costs ~150 ms, while
    // compiling one item's schema against it costs well under a millisecond.
    // Only paid when some item actually asks for a schema.
    let factory = if prepared
        .iter()
        .any(|p| p.as_ref().is_ok_and(|p| p.schema.is_some()))
    {
        Some(constrain::shared()?)
    } else {
        None
    };

    // Phase accounting: decode comes off each item's own outcome, prefill by
    // delta from the generator's cumulative spend — the shared prefill above
    // and the scored path's teacher-forced trials belong to no single item.
    let prefill_before = generator.prefill_spend();
    let mut decode_tokens = 0usize;
    let mut decode_secs = 0.0f64;

    let mut items = Vec::with_capacity(req.items.len());
    for (spec, prepared) in req.items.iter().zip(prepared.iter()) {
        // Polled at every item boundary, and again below after a run: a
        // cancelled batch stops running items and reports the rest untouched,
        // rather than being torn down mid-response.
        if (hooks.cancelled)() {
            items.push(failed_item(&spec.id, CANCELLED.to_string()));
            continue;
        }
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                items.push(failed_item(&spec.id, error.clone()));
                continue;
            }
        };
        let cached = snapshot.as_ref().map(|snapshot| (snapshot, shared_len));
        let item_started = Instant::now();
        match run_item(generator, factory, prepared, cached, hooks.cancelled) {
            Ok(outcome) => {
                (hooks.progress)(BatchProgress::Item {
                    id: spec.id.clone(),
                    completion_tokens: outcome.completion_tokens,
                    ms: elapsed_ms(item_started),
                });
                decode_tokens += outcome.decode_tokens;
                decode_secs += outcome.decode_secs;
                // A scored item brings its own value and its own refusal; only
                // the grammar path has a document left to parse.
                let (json, error) = match (&outcome.json, &outcome.error) {
                    (json, Some(error)) => (json.clone(), Some(error.clone())),
                    (Some(json), None) => (Some(json.clone()), None),
                    (None, None) => match constrained_value(prepared, &outcome) {
                        Some(Ok(value)) => (Some(value), None),
                        Some(Err(error)) => (None, Some(error)),
                        None => (None, None),
                    },
                };
                items.push(ItemResponse {
                    id: spec.id.clone(),
                    content: outcome.content,
                    text: outcome.text,
                    json,
                    finish_reason: outcome.finish_reason,
                    usage: Usage {
                        prompt_tokens: prepared.tokens.len(),
                        cached_prefix_tokens: shared_len,
                        completion_tokens: outcome.completion_tokens,
                    },
                    error,
                });
            }
            Err(error) => items.push(failed_item(&spec.id, format!("{error:#}"))),
        }
    }

    let prefill = generator.prefill_spend();
    Ok(BatchResponse {
        // What actually ran, under the name it answers to — the same spelling
        // `/v1/models` lists and the request field selects by.
        model: label.to_string(),
        stats: BatchStats {
            shared_prefix_tokens: shared_len,
            snapshot_ms,
            items: items.len(),
            prefill_tokens: prefill.tokens - prefill_before.tokens,
            prefill_ms: (prefill.secs - prefill_before.secs) * 1000.0,
            decode_tokens,
            decode_ms: decode_secs * 1000.0,
            load_ms,
            total_ms: elapsed_ms(started),
        },
        items,
    })
}

/// The value a constrained item produced, or the reason it has none. `None`
/// for an item that carried no schema.
///
/// The document is the rendered response prefix followed by what was decoded: a
/// prefill is prompt, so the model never re-emits it, and parsing the decoded
/// text alone would see a value with its head missing.
fn constrained_value(item: &Prepared, outcome: &ItemOutcome) -> Option<Result<Value, String>> {
    item.schema.as_ref()?;
    // A constrained value is complete only when the grammar said so. A run that
    // stopped at its cap stopped mid-document, and reporting the budget beats
    // reporting a parse error about a truncated value.
    if outcome.finish_reason != FinishReason::Stop {
        return Some(Err(format!(
            "the answer stopped at the {}-token budget before the value was complete",
            item.max_tokens
        )));
    }
    let document = format!("{}{}", item.prefix_text, outcome.text);
    Some(
        serde_json::from_str::<Value>(document.trim())
            .map_err(|error| format!("the constrained answer did not parse as JSON: {error}")),
    )
}

/// Run one prepared item to completion: rewind to the shared prefix (or to
/// nothing), prefill the item's own tail, and decode under its sampler and
/// grammar.
///
/// `cached` is the shared snapshot and the position it holds. A restore drops
/// the held prefill logits, so the prefill below is what every decode reads
/// from — there is no path here that decodes without prefilling first.
fn run_item(
    generator: &mut Generator,
    factory: Option<&ConstraintFactory>,
    item: &Prepared,
    cached: Option<(&CacheSnapshot, usize)>,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<ItemOutcome> {
    let resume = match cached {
        Some((snapshot, at)) => {
            generator.restore_cache_snapshot(snapshot)?;
            // The drafter holds the previous item's tail; truncating it back to
            // the shared position is what lets speculation resume immediately
            // instead of waiting for a prefill from zero.
            generator.sync_drafter_to(at)?;
            at
        }
        None => {
            generator.reset_cache()?;
            generator.reset_drafter()?;
            0
        }
    };
    // `shared_prefix_len` holds the prefix short of the shortest item, so the
    // tail is never empty.
    generator.prefill_tokens(&item.tokens[resume..], resume)?;

    generator.set_sampler(SamplerOptions {
        temperature: item.sampling.temperature,
        top_k: item.sampling.top_k,
        top_p: item.sampling.top_p,
        presence_penalty: item.sampling.presence_penalty,
        seed: item.sampling.seed,
    });
    if let Some(plan) = &item.scored {
        // Nothing on the scored path free-decodes against a mask — the runner
        // writes the structure itself — so the grammar is cleared rather than
        // compiled. `None` is also what clears the previous item's.
        generator.set_grammar(None);
        return assemble_scored(generator, item, plan);
    }
    // Set unconditionally: `None` is what clears the previous item's grammar.
    generator.set_grammar(item_grammar(
        factory,
        item,
        *generator.tokenizer().specials(),
    )?);

    let prompt_len = item.tokens.len();
    let mut content = String::new();
    let mut text = String::new();
    let mut on_event = |event: GenEvent| {
        content.push_str(event.text());
        if let GenEvent::TextTok { text: chunk, .. } = &event {
            text.push_str(chunk);
        }
    };
    // The cancel poll is latched so this item can tell "my decode was cut"
    // from "the batch was cancelled right after I finished": only the former
    // makes this item's answer not the one it was asked for.
    let mut cut = false;
    let mut should_stop = || {
        cut = cut || cancelled();
        cut
    };
    // The speculative loop draws the same tokens either way; this asks the
    // narrower question of whether it could actually speculate from here, so a
    // drafter that fell behind does not pay the round loop's overhead.
    generator.note_draft_horizon_at(prompt_len);
    let outcome = if generator.spec_ready_at(prompt_len) {
        generator.decode_loop_spec(
            prompt_len,
            item.starts_in_thinking,
            item.max_tokens,
            &mut on_event,
            &mut should_stop,
        )?
    } else {
        generator.decode_loop(
            prompt_len,
            item.starts_in_thinking,
            item.max_tokens,
            &mut on_event,
            &mut should_stop,
        )?
    };

    // A cut decode stopped mid-answer: whatever it produced is not the answer
    // the item asked for, so the text is dropped and the cancellation reported
    // in its place — a failed item carries `error` and empty text, which is
    // the response's documented contract. The usage still counts the tokens
    // the cut decode actually spent. `hit_eog` overrides the latch: a decode
    // that ENDED naturally (EOG, or a grammar that completed — a single-token
    // constrained value completes on its first draw) is a complete answer
    // however late the cancellation signal landed, and must never be relabeled.
    if cut && !outcome.hit_eog {
        return Ok(ItemOutcome {
            content: String::new(),
            text: String::new(),
            finish_reason: FinishReason::Error,
            completion_tokens: outcome.tokens_out,
            decode_tokens: outcome.tokens_out,
            decode_secs: outcome.decode_secs,
            json: None,
            error: Some(CANCELLED.to_string()),
        });
    }

    Ok(ItemOutcome {
        content,
        text,
        decode_tokens: outcome.tokens_out,
        decode_secs: outcome.decode_secs,
        finish_reason: if outcome.hit_eog {
            FinishReason::Stop
        } else {
            FinishReason::Length
        },
        completion_tokens: outcome.tokens_out,
        json: None,
        error: None,
    })
}

/// The grammar an item decodes under, `None` for an unconstrained one. A
/// compiled `Grammar` is consumed by `into_state`, so this compiles per item
/// rather than sharing one across the batch.
fn item_grammar(
    factory: Option<&ConstraintFactory>,
    item: &Prepared,
    specials: Specials,
) -> Result<Option<GrammarState>> {
    let Some(schema) = &item.schema else {
        return Ok(None);
    };
    let factory =
        factory.ok_or_else(|| anyhow!("batch: an item wants a schema but no factory was built"))?;
    let mut state = factory
        .compile(schema)?
        .into_state(item.starts_in_thinking, specials);
    if item.prefix_len > 0 {
        // A response prefix is already part of the answer document; feeding it
        // in is what makes the first mask continue it. A prefix can only be
        // rendered past a closed thinking span, so the state is live here.
        state.consume_prefix(&item.tokens[item.tokens.len() - item.prefix_len..])?;
    }
    Ok(Some(state))
}

// ----------------------------------------------------------- scored items ---

/// How much of a scored field's evidence the response reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScoreMode {
    /// The property carries no `include_score`. Its value is still CHOSEN by
    /// scoring — one item assembles one document — but it reports as a bare
    /// value, exactly as the grammar path would have written it.
    Bare,
    /// `include_score: true`.
    Value,
    /// `include_score: "all"`: the whole option table, plus the mass that fell
    /// outside it.
    All,
}

/// One value a scored field may take.
#[derive(Debug, Clone, PartialEq)]
struct FieldOption {
    /// What is written into the assembled document. For a string field this is
    /// the JSON-escaped body WITHOUT its quotes: the quotes belong to the
    /// skeleton, so that what gets scored is the tokens of the value itself.
    text: String,
    /// `text` encoded. Never empty — an option with no tokens would score as
    /// certainty for free.
    tokens: Vec<u32>,
    /// What the response reports as the chosen value: a JSON string, or a real
    /// JSON boolean.
    value: Value,
    /// This option's key in an `"all"` field's `scores` table. A boolean's
    /// options are keyed by the STRINGS `"true"`/`"false"`, a JSON object
    /// having no other kind of key.
    key: String,
}

/// One property of a scored schema.
#[derive(Debug, Clone, PartialEq)]
struct ScoredField {
    name: String,
    options: Vec<FieldOption>,
    /// Whether the value is written inside quotes — true for the string enums,
    /// false for the booleans.
    quoted: bool,
    mode: ScoreMode,
}

/// A run of literal JSON between two values.
#[derive(Debug, Clone, PartialEq)]
struct Segment {
    text: String,
    tokens: Vec<u32>,
    /// The client-content byte ranges of `text` — its field name. Kept so the
    /// seam checks can re-encode this segment against its neighbours under the
    /// same demotion rules it was first encoded with.
    ranges: Vec<Range<usize>>,
}

/// A scored item's whole output plan: the fields in schema order, the skeleton
/// that frames them, and what the two together cost.
#[derive(Debug, Clone, PartialEq)]
struct ScoredPlan {
    fields: Vec<ScoredField>,
    /// The literal JSON around the values, `fields.len() + 1` segments: the head
    /// (`{"first":"`), one per gap (`","second":`) and the tail (`"}`).
    segments: Vec<Segment>,
    /// Tokens the assembly writes at worst — every segment plus the LONGEST
    /// option of every field. The budget is checked against this before any
    /// forward runs, so an item that cannot fit fails without costing a prefill.
    ///
    /// It bounds the scoring branches too, though they reach a token deeper than
    /// the document does: a trial at field `i`'s choice point writes that whole
    /// option, where the document goes on to write the segment after it. The
    /// closing segment is counted here and never prefilled, which leaves the
    /// deepest branch strictly inside this.
    worst_case_tokens: usize,
}

/// The v1 shape `include_score` is defined against. A schema that carries the
/// annotation without fitting it is an error rather than a schema quietly
/// decoded the old way — the caller asked for scores and would otherwise get a
/// plausible answer with none.
const SCORED_SHAPE: &str = "a scored schema must be {\"type\": \"object\", \"properties\": {…}, \
     \"required\": [every property], \"additionalProperties\": false}, every property being \
     either {\"enum\": [strings]} or {\"type\": \"boolean\"}";

/// The assembly plan an item's schema asks for, or `None` when the schema
/// mentions no `include_score` and belongs on the grammar path untouched.
fn scored_plan(tokenizer: &LagunaTokenizer, schema: &Value) -> Result<Option<ScoredPlan>> {
    if !mentions_include_score(schema) {
        return Ok(None);
    }
    let fields = scored_fields(tokenizer, schema)?;
    let segments = skeleton(tokenizer, &fields)?;
    check_seams(tokenizer, &fields, &segments)?;
    let worst_case_tokens = segments.iter().map(|s| s.tokens.len()).sum::<usize>()
        + fields
            .iter()
            .map(|f| {
                f.options
                    .iter()
                    .map(|o| o.tokens.len())
                    .max()
                    .expect("a field always carries at least one option")
            })
            .sum::<usize>();
    Ok(Some(ScoredPlan {
        fields,
        segments,
        worst_case_tokens,
    }))
}

/// Whether `include_score` appears anywhere in the schema. The search is over
/// the whole document rather than the top level's properties, so an annotation
/// buried in a nested definition still routes the schema here — where the shape
/// guard refuses it by name instead of dropping it silently.
fn mentions_include_score(schema: &Value) -> bool {
    match schema {
        Value::Object(map) => {
            map.contains_key("include_score") || map.values().any(mentions_include_score)
        }
        Value::Array(items) => items.iter().any(mentions_include_score),
        _ => false,
    }
}

/// Validate a scored schema against [`SCORED_SHAPE`] and read its fields out in
/// schema order (`serde_json`'s `preserve_order` is what makes that order the
/// caller's rather than alphabetical).
fn scored_fields(tokenizer: &LagunaTokenizer, schema: &Value) -> Result<Vec<ScoredField>> {
    let object = schema
        .as_object()
        .ok_or_else(|| anyhow!("{SCORED_SHAPE}; this schema is not a JSON object"))?;
    for key in object.keys() {
        // `title`/`description` carry no semantics; anything else could change
        // what the schema means, and a scored item never sees a validator that
        // would enforce it.
        if !matches!(
            key.as_str(),
            "type" | "properties" | "required" | "additionalProperties" | "title" | "description"
        ) {
            bail!("{SCORED_SHAPE}; this schema also carries {key:?}");
        }
    }
    ensure!(
        object.get("type").and_then(Value::as_str) == Some("object"),
        "{SCORED_SHAPE}; this schema's \"type\" is not \"object\""
    );
    ensure!(
        object.get("additionalProperties") == Some(&Value::Bool(false)),
        "{SCORED_SHAPE}; this schema's \"additionalProperties\" is not false"
    );
    let properties = object
        .get("properties")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("{SCORED_SHAPE}; this schema has no \"properties\" object"))?;
    ensure!(
        !properties.is_empty(),
        "{SCORED_SHAPE}; this schema declares no properties"
    );
    let required = object
        .get("required")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("{SCORED_SHAPE}; this schema has no \"required\" array"))?;
    let mut listed = required
        .iter()
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| anyhow!("{SCORED_SHAPE}; its \"required\" holds a non-string entry"))
        })
        .collect::<Result<Vec<_>>>()?;
    listed.sort_unstable();
    listed.dedup();
    let mut declared: Vec<&str> = properties.keys().map(String::as_str).collect();
    declared.sort_unstable();
    ensure!(
        listed == declared,
        "{SCORED_SHAPE}; its \"required\" does not list exactly its properties"
    );
    properties
        .iter()
        .map(|(name, spec)| scored_field(tokenizer, name, spec))
        .collect()
}

/// One property of a scored schema, validated and encoded.
fn scored_field(tokenizer: &LagunaTokenizer, name: &str, spec: &Value) -> Result<ScoredField> {
    let object = spec
        .as_object()
        .ok_or_else(|| anyhow!("property {name:?} is not a JSON object; {SCORED_SHAPE}"))?;
    for key in object.keys() {
        if !matches!(
            key.as_str(),
            "type" | "enum" | "include_score" | "title" | "description"
        ) {
            bail!("property {name:?} carries {key:?}; {SCORED_SHAPE}");
        }
    }
    let mode = score_mode(name, object.get("include_score"))?;
    let declared = match object.get("type") {
        Some(value) => Some(
            value
                .as_str()
                .ok_or_else(|| anyhow!("property {name:?} has a non-string \"type\""))?,
        ),
        None => None,
    };
    let (options, quoted) = match object.get("enum") {
        Some(values) => {
            ensure!(
                matches!(declared, None | Some("string")),
                "property {name:?} is an enum of type {:?}; only string enums are scored",
                declared.unwrap_or_default(),
            );
            (enum_options(tokenizer, name, values)?, true)
        }
        None => {
            ensure!(
                declared == Some("boolean"),
                "property {name:?} is neither an enum nor a boolean; {SCORED_SHAPE}"
            );
            (boolean_options(tokenizer, name)?, false)
        }
    };
    Ok(ScoredField {
        name: name.to_string(),
        options,
        quoted,
        mode,
    })
}

/// Read one property's `include_score`. An unrecognized value is an error: the
/// annotation is the whole reason this item left the grammar path, so a typo in
/// it must not read as "no scores, then".
fn score_mode(name: &str, value: Option<&Value>) -> Result<ScoreMode> {
    match value {
        None | Some(Value::Bool(false)) => Ok(ScoreMode::Bare),
        Some(Value::Bool(true)) => Ok(ScoreMode::Value),
        Some(Value::String(mode)) if mode == "all" => Ok(ScoreMode::All),
        Some(other) => {
            bail!("property {name:?} has include_score {other}; it must be true, false or \"all\"")
        }
    }
}

/// The options of a string enum, in the order the schema lists them.
fn enum_options(
    tokenizer: &LagunaTokenizer,
    name: &str,
    values: &Value,
) -> Result<Vec<FieldOption>> {
    let values = values
        .as_array()
        .ok_or_else(|| anyhow!("property {name:?} has a non-array \"enum\""))?;
    ensure!(
        !values.is_empty(),
        "property {name:?} has an empty \"enum\""
    );
    let mut options: Vec<FieldOption> = Vec::with_capacity(values.len());
    for value in values {
        let text = value
            .as_str()
            .ok_or_else(|| anyhow!("property {name:?} has a non-string enum value {value}"))?;
        // An empty option encodes to nothing, and a zero-length sequence scores
        // as logprob 0 — certainty, for free, against every real option.
        ensure!(
            !text.is_empty(),
            "property {name:?} has an empty enum value, which has no tokens to score"
        );
        ensure!(
            !options.iter().any(|o| o.key == text),
            "property {name:?} lists the enum value {text:?} twice"
        );
        // A value needing JSON escapes is refused rather than escaped. The
        // escape sequence would be what gets scored and what the closing quote
        // is checked against, and a value containing a quote of its own would
        // no longer be delimited by one — v1 takes the values it can write
        // literally and says so about the rest.
        ensure!(
            json_body(text) == text,
            "property {name:?} has the enum value {text:?}, which needs JSON escaping; \
             unsupported"
        );
        options.push(FieldOption {
            tokens: encode_client_text(tokenizer, text)?,
            text: text.to_string(),
            value: Value::String(text.to_string()),
            key: text.to_string(),
        });
    }
    Ok(options)
}

/// The two options of a boolean field, written bare (a JSON `true` carries no
/// quotes) and keyed by their spellings.
fn boolean_options(tokenizer: &LagunaTokenizer, name: &str) -> Result<Vec<FieldOption>> {
    [true, false]
        .into_iter()
        .map(|flag| {
            let text = if flag { "true" } else { "false" };
            let tokens = encode_client_text(tokenizer, text)?;
            ensure!(
                !tokens.is_empty(),
                "property {name:?} could not encode the boolean literal {text:?}"
            );
            Ok(FieldOption {
                text: text.to_string(),
                tokens,
                value: Value::Bool(flag),
                key: text.to_string(),
            })
        })
        .collect()
}

/// The skeleton around the values: `fields.len() + 1` runs of literal JSON,
/// compact (no whitespace anywhere), so the only thing left unwritten at any
/// point is a value.
///
/// A string field's quotes live here rather than on its options. That is what
/// makes an option's score the logprob of its BODY: the model is already past
/// the opening quote when it is scored, so `"positive"` and `"positively awful"`
/// are compared on the tokens that actually distinguish them.
fn skeleton(tokenizer: &LagunaTokenizer, fields: &[ScoredField]) -> Result<Vec<Segment>> {
    let mut segments = Vec::with_capacity(fields.len() + 1);
    for index in 0..=fields.len() {
        let mut text = String::new();
        let mut ranges = Vec::new();
        if index > 0 && fields[index - 1].quoted {
            text.push('"');
        }
        text.push(match index {
            0 => '{',
            n if n == fields.len() => '}',
            _ => ',',
        });
        if let Some(field) = fields.get(index) {
            // The key is client-supplied, so its bytes are marked as content:
            // a field named after a control marker must encode as plain text
            // rather than mint the token.
            let key = json_string(&field.name);
            let start = text.len() + 1;
            text.push_str(&key);
            ranges.push(start..text.len() - 1);
            text.push(':');
            if field.quoted {
                text.push('"');
            }
        }
        let tokens = tokenizer.encode_prompt(&text, &ranges)?;
        // Every segment carries at least one structural character, and the
        // token that opens the segment AFTER a value is what terminates that
        // value's scored sequence.
        ensure!(
            !tokens.is_empty(),
            "the skeleton segment {text:?} encoded to zero tokens"
        );
        segments.push(Segment {
            text,
            tokens,
            ranges,
        });
    }
    Ok(segments)
}

/// Refuse a plan whose pieces do not tokenize the same apart as together.
///
/// The document is teacher-forced piece by piece — a skeleton segment, a value,
/// the next segment — but every text the model has ever seen was tokenized as a
/// whole. Where a value's last character would canonically MERGE with the
/// delimiter behind it (`yes!` followed by the closing quote encodes `!"` as one
/// token), the forced stream is a seam the model was never trained on, and both
/// the option's score and the answer's tokens would be measured across it.
///
/// Checking seams rather than whole documents is what makes this computable: a
/// document has one tokenization per combination of options, while a seam has
/// one per option.
fn check_seams(
    tokenizer: &LagunaTokenizer,
    fields: &[ScoredField],
    segments: &[Segment],
) -> Result<()> {
    for (index, field) in fields.iter().enumerate() {
        let (before, after) = (&segments[index], &segments[index + 1]);
        for option in &field.options {
            let value = (option.text.as_str(), [0..option.text.len()]);
            let opening: Vec<u32> = before
                .tokens
                .iter()
                .chain(&option.tokens)
                .copied()
                .collect();
            ensure!(
                encode_seam(
                    tokenizer,
                    (&before.text, &before.ranges),
                    (value.0, &value.1)
                )? == opening,
                "property {:?} takes the value {:?}, which tokenizes non-canonically against the \
                 {:?} before it; unsupported",
                field.name,
                option.key,
                before.text,
            );
            let closing: Vec<u32> = option.tokens.iter().chain(&after.tokens).copied().collect();
            ensure!(
                encode_seam(tokenizer, (value.0, &value.1), (&after.text, &after.ranges))?
                    == closing,
                "property {:?} takes the value {:?}, which tokenizes non-canonically against the \
                 {:?} after it; unsupported",
                field.name,
                option.key,
                after.text,
            );
        }
    }
    Ok(())
}

/// The canonical tokenization of two adjacent pieces of the document, encoded as
/// one string rather than as two — each piece keeping the content ranges it
/// carries on its own, the right one rebased onto the joined text.
fn encode_seam(
    tokenizer: &LagunaTokenizer,
    left: (&str, &[Range<usize>]),
    right: (&str, &[Range<usize>]),
) -> Result<Vec<u32>> {
    let text = format!("{}{}", left.0, right.0);
    let offset = left.0.len();
    let ranges: Vec<Range<usize>> = left
        .1
        .iter()
        .cloned()
        .chain(right.1.iter().map(|r| r.start + offset..r.end + offset))
        .collect();
    tokenizer.encode_prompt(&text, &ranges)
}

/// `text` as a quoted, escaped JSON string.
fn json_string(text: &str) -> String {
    Value::String(text.to_string()).to_string()
}

/// `text` escaped as JSON but WITHOUT its quotes — the bytes that WOULD go
/// between the skeleton's quotes. A value the assembler can write literally is
/// its own body; anything else differs here, which is how such a value is
/// spotted and refused.
fn json_body(text: &str) -> String {
    let quoted = json_string(text);
    quoted[1..quoted.len() - 1].to_string()
}

/// Encode caller-supplied text as content: an added-token string inside it
/// stays plain text rather than becoming the control token whose id it names.
fn encode_client_text(tokenizer: &LagunaTokenizer, text: &str) -> Result<Vec<u32>> {
    tokenizer.encode_prompt(text, &[0..text.len()])
}

/// One field's outcome: which option won, and the evidence behind it.
#[derive(Debug, Clone, PartialEq)]
struct Pick {
    index: usize,
    /// Per option, `softmax` over the options' full-sequence logprobs — value
    /// and closing delimiter both (see [`score_field`]) — which is the
    /// probability the response reports, renormalized over what the schema
    /// allows. Always at temperature 1 and never truncated, whatever sampling
    /// the item drew under.
    probs: Vec<f64>,
    /// Mass the choice point put on writing something that opens NO option —
    /// the vocabulary gap: everything the model would rather have SAID than any
    /// allowed value, with everything it would merely have FORMATTED
    /// differently factored out (see [`escape_mass`] for the classification).
    ///
    /// Measured on one row, while `probs` are measured over whole sequences.
    /// The two answer different questions and do not add up: a token this
    /// counts as inside may still lead somewhere no option goes. The reported
    /// probabilities are unaffected either way — every option is scored from
    /// this same row, so whatever the model spent elsewhere divides out of
    /// them.
    escape: f64,
}

/// Run one scored item: teacher-force the skeleton, score each field's options
/// where a value belongs, and report the document the two produced.
///
/// The item's prompt is already prefilled by the caller, so the logits in hand
/// describe the position the answer starts at.
fn assemble_scored(
    generator: &mut Generator,
    item: &Prepared,
    plan: &ScoredPlan,
) -> Result<ItemOutcome> {
    // Refused before any forward runs, and against the LONGEST option of each
    // field: which option wins is not known until it has been scored, and an
    // item that could only fit by picking short answers does not fit.
    let reasoning_floor = usize::from(item.starts_in_thinking);
    if plan.worst_case_tokens + reasoning_floor > item.max_tokens {
        return Ok(budget_refusal(item, plan, reasoning_floor));
    }

    let prompt_len = item.tokens.len();
    let mut reasoning = String::new();
    let mut written = Vec::new();
    let mut decode_tokens = 0;
    let mut decode_secs = 0.0;
    if item.starts_in_thinking {
        let run = decode_reasoning(
            generator,
            prompt_len,
            item.max_tokens - plan.worst_case_tokens,
        )?;
        ensure!(
            run.closed,
            "the reasoning block did not close within the {} tokens left after the assembled \
             answer's budget, so there is no answer position to score from",
            item.max_tokens - plan.worst_case_tokens,
        );
        reasoning = run.text;
        written = run.ids;
        decode_tokens = run.decode_tokens;
        decode_secs = run.decode_secs;
    }

    // Both decode loops may leave their last emitted token unforwarded, so the
    // cache — never the count of events — says how much of the reasoning is
    // context. Whatever it is short of gets written back with the first segment.
    let held = generator.cache_len().saturating_sub(prompt_len);
    ensure!(
        held <= written.len(),
        "the KV cache holds {held} tokens past the prompt but only {} were emitted",
        written.len(),
    );
    let mut pending = written[held..].to_vec();

    let mut rng = StdRng::seed_from_u64(item.sampling.seed);
    let mut document = String::new();
    let mut picks = Vec::with_capacity(plan.fields.len());
    for (index, field) in plan.fields.iter().enumerate() {
        pending.extend_from_slice(&plan.segments[index].tokens);
        prefill_for_scoring(generator, &pending)?;
        document.push_str(&plan.segments[index].text);

        // Every candidate is scored through to the token that opens the segment
        // after it, which is what makes the option set prefix-free.
        let terminator = plan.segments[index + 1].tokens[0];
        let pick = score_field(generator, field, terminator, &item.sampling, &mut rng)?;
        let chosen = &field.options[pick.index];
        document.push_str(&chosen.text);
        pending = chosen.tokens.clone();
        picks.push(pick);
    }
    // The closing segment is never prefilled: nothing is scored after it, and
    // the cache this item leaves behind is rewound by the next one anyway. It is
    // still part of the answer, so it is still part of the token count.
    document.push_str(&plan.segments[plan.fields.len()].text);

    let value: Map<String, Value> = plan
        .fields
        .iter()
        .zip(&picks)
        .map(|(field, pick)| (field.name.clone(), report(field, pick)))
        .collect();
    let assembled: usize = plan.segments.iter().map(|s| s.tokens.len()).sum::<usize>()
        + plan
            .fields
            .iter()
            .zip(&picks)
            .map(|(field, pick)| field.options[pick.index].tokens.len())
            .sum::<usize>();

    Ok(ItemOutcome {
        content: format!("{reasoning}{document}"),
        text: document,
        finish_reason: FinishReason::Stop,
        completion_tokens: written.len() + assembled,
        decode_tokens,
        decode_secs,
        json: Some(Value::Object(value)),
        error: None,
    })
}

/// The response for an item whose assembled answer cannot fit its token budget.
/// Reported as `length` rather than as an error: the budget is what refused it,
/// which is the same thing that truncates a decoded answer.
fn budget_refusal(item: &Prepared, plan: &ScoredPlan, reasoning_floor: usize) -> ItemOutcome {
    let reasoning = if reasoning_floor > 0 {
        ", and its reasoning block needs at least one token more"
    } else {
        ""
    };
    ItemOutcome {
        content: String::new(),
        text: String::new(),
        finish_reason: FinishReason::Length,
        completion_tokens: 0,
        decode_tokens: 0,
        decode_secs: 0.0,
        json: None,
        error: Some(format!(
            "the assembled answer needs up to {} tokens{reasoning}, past this item's \
             {}-token budget",
            plan.worst_case_tokens, item.max_tokens,
        )),
    }
}

/// What the reasoning phase of a scored item produced.
struct Reasoning {
    /// The reasoning as text, `</think>` stripped by the decoder as usual.
    text: String,
    /// Every id emitted, `</think>` included, in order.
    ids: Vec<u32>,
    closed: bool,
    /// The decode loop's own accounting, passed through for the item's totals.
    decode_tokens: usize,
    decode_secs: f64,
}

/// Decode a scored item's reasoning and stop the moment `</think>` commits.
///
/// Assembly needs the cache to hold the reasoning and nothing past it, and the
/// cancel poll — which both decode loops read once per emitted token, right
/// after the callback — is what hands the turn over on exactly that token. The
/// two loops differ in whether the last emitted token was forwarded, so the
/// caller reconciles against `cache_len()` rather than against the ids here.
fn decode_reasoning(
    generator: &mut Generator,
    prompt_len: usize,
    budget: usize,
) -> Result<Reasoning> {
    let closed = Cell::new(false);
    let mut text = String::new();
    let mut ids = Vec::new();
    let outcome;
    let think_close = generator.tokenizer().specials().think_close;
    {
        let mut on_event = |event: GenEvent| {
            ids.push(event.id());
            text.push_str(event.text());
            if event.id() == think_close {
                closed.set(true);
            }
        };
        let mut stop = || closed.get();
        generator.note_draft_horizon_at(prompt_len);
        outcome = if generator.spec_ready_at(prompt_len) {
            generator.decode_loop_spec(prompt_len, true, budget, &mut on_event, &mut stop)?
        } else {
            generator.decode_loop(prompt_len, true, budget, &mut on_event, &mut stop)?
        };
    }
    Ok(Reasoning {
        text,
        ids,
        closed: closed.get(),
        decode_tokens: outcome.tokens_out,
        decode_secs: outcome.decode_secs,
    })
}

/// Prefill `tokens` so the logits left behind are the ones a DECODE step would
/// have produced: everything but the last token as one span, the last token
/// alone.
///
/// The lm head runs on a span's final position only, and at one position it
/// takes the mat-vec bypass a longer span does not (`XwenModel::forward`). Every
/// row a score is read from therefore comes off the same path — the skeleton's
/// choice point and an option's continuations alike — instead of the choice
/// point's precision depending on how long the segment before it happened to be.
fn prefill_for_scoring(generator: &mut Generator, tokens: &[u32]) -> Result<()> {
    ensure!(
        !tokens.is_empty(),
        "prefill_for_scoring: nothing to prefill"
    );
    let (head, last) = tokens.split_at(tokens.len() - 1);
    let mut pos = generator.cache_len();
    if !head.is_empty() {
        generator.prefill_tokens(head, pos)?;
        pos += head.len();
    }
    generator.prefill_tokens(last, pos)
}

/// Score every option a field allows at the current choice point, and pick one.
///
/// An option's score is the logprob of its token sequence PLUS `terminator`, the
/// token that closes the value in the assembled document — the closing quote of
/// a string field, the `,` or `}` after a boolean. Scoring the body alone would
/// make the candidate set prefix-dependent: `low` is a strict prefix of
/// `low_priority`, so the longer one's score is the shorter one's plus a
/// negative, and greedy selection could never choose it however plainly the
/// model preferred it. Ending every candidate at a common delimiter makes the
/// set prefix-free, and turns the score into the probability of the event that
/// actually distinguishes them: this value, then the end of it.
///
/// The terminator is SCORED but never committed — it belongs to the segment
/// after the value, which the assembly writes in its own right.
///
/// The first token of every option is read from the one row the choice point
/// already holds, so that row is read FIRST, before anything touches the cache:
/// a restore drops it. Every option then costs one forward per body token, the
/// last of which is what the terminator is scored against.
fn score_field(
    generator: &mut Generator,
    field: &ScoredField,
    terminator: u32,
    sampling: &SamplerOptions,
    rng: &mut StdRng,
) -> Result<Pick> {
    let openers: Vec<u32> = field
        .options
        .iter()
        .map(|option| option.tokens[0])
        .collect();
    let mut scores = generator.last_logprobs_for(&openers)?;
    let escape = escape_mass(
        &generator.last_probs()?,
        generator.tokenizer().decoded_vocab(),
        field,
    )?;

    let choice_pos = generator.cache_len();
    // Taken once for every option: a restore copies by reference, so re-taking
    // it per option would pay for a whole cache copy each time.
    let snapshot = generator.take_cache_snapshot()?;
    for (option, score) in field.options.iter().zip(scores.iter_mut()) {
        for (step, token) in option.tokens.iter().enumerate() {
            generator.prefill_tokens(&[*token], choice_pos + step)?;
            // The next token of the option, or — once the body is spent — the
            // delimiter that ends it.
            let next = option.tokens.get(step + 1).copied().unwrap_or(terminator);
            *score += generator.last_logprobs_for(&[next])?[0];
        }
        generator.restore_cache_snapshot(&snapshot)?;
        // A prefill feeds the drafter as well as the target, so a branch that
        // rewinds one has to rewind the other with it.
        generator.sync_drafter_to(choice_pos)?;
    }

    Ok(Pick {
        // Reported untruncated and at temperature 1, whatever the draw below
        // does: the score is what the model thinks, not how the item sampled.
        probs: renormalize(&scores, 1.0),
        index: select_option(&scores, sampling, rng)?,
        escape,
    })
}

/// Draw one option from the posterior under the item's sampling settings.
///
/// Runs `Sampler`'s pipeline over the option set the way the sampler runs it over
/// the vocabulary — temperature, then top-k, then a nucleus cut measured on the
/// renormalized survivors — so that a scored field and a decoded answer answer to
/// the same knobs. A `top_k` of exactly one collapses to greedy there
/// (`Sampler::new`) and collapses to greedy here; a `top_k` of zero is "no
/// top-k cut" in both places, so it keeps every option for the nucleus cut.
///
/// `presence_penalty` is deliberately NOT among the knobs this shares. The
/// penalty is measured over the tokens a reply has emitted, and this path
/// emits none: it teacher-forces each candidate value and picks between whole
/// options by their scores. There is nothing here for a per-token penalty to
/// mean, so a scored field is drawn unpenalized however the item set it. Free
/// decoding inside the same batch item still is penalized — it runs through
/// the ordinary decode loops.
fn select_option(scores: &[f64], sampling: &SamplerOptions, rng: &mut StdRng) -> Result<usize> {
    if sampling.temperature <= 0.0 || sampling.top_k == 1 {
        return Ok(argmax_index(scores));
    }
    let probs = renormalize(scores, sampling.temperature);
    // Descending by probability, ties going to the option the schema listed
    // first — the direction `top_k_desc` breaks them in.
    let mut ranked: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if sampling.top_k > 0 {
        ranked.truncate(sampling.top_k);
    }
    truncate_nucleus(&mut ranked, sampling.top_p);
    let weights: Vec<f64> = ranked.iter().map(|(_, prob)| *prob).collect();
    let picked = WeightedIndex::new(&weights)
        .map_err(|error| anyhow!("no option carries any mass after top-k/top-p: {error}"))?
        .sample(rng);
    Ok(ranked[picked].0)
}

/// Keep the shortest prefix of `ranked` whose mass — renormalized over the
/// candidates themselves — reaches `top_p`, the token that crosses the threshold
/// included. That is llama.cpp's convention and the one `truncate_top_p` applies
/// to the vocabulary, early return at 1.0 included: renormalized weights sum to
/// one only to within rounding, and a cumulative walk could otherwise drop the
/// last candidate over an ulp.
fn truncate_nucleus(ranked: &mut Vec<(usize, f64)>, top_p: f64) {
    if top_p >= 1.0 {
        return;
    }
    let total: f64 = ranked.iter().map(|(_, prob)| prob).sum();
    if total <= 0.0 {
        return;
    }
    let mut cumulative = 0.0;
    let mut keep = ranked.len();
    for (at, (_, prob)) in ranked.iter().enumerate() {
        cumulative += prob / total;
        if cumulative >= top_p {
            keep = at + 1;
            break;
        }
    }
    ranked.truncate(keep);
}

/// The probability the choice point put on writing something that opens NO
/// option, with the mass it spent on FORMATTING factored out: `probs` is the
/// whole next-token distribution ([`Generator::last_probs`]) and `decoded` the
/// per-id raw bytes it is classified by ([`LagunaTokenizer::decoded_vocab`]).
///
/// The skeleton is compact JSON, and the model's preferred layout usually is
/// not: at an unquoted choice point (right after a boolean's colon) most of the
/// mass sits on whitespace-led spellings of the very values the schema allows —
/// ` true` is a single token — and on pure-whitespace tokens that say nothing
/// about the answer either way. Counting those as escape pins a document's
/// first field near 1.0 (later fields sit in an established compact document,
/// which conditions the formatting away) and buries the actual vocabulary-gap
/// signal. So every token is classified, by its BYTES — byte-level BPE is free
/// to cut a multi-byte character across tokens, and a text-level comparison
/// would misread the canonical opener of any non-ASCII option as escape:
///
/// * INSIDE — the bytes, read past any leading JSON whitespace at an unquoted
///   field (verbatim at a quoted one, where leading whitespace would be string
///   content), are a nonempty prefix of some option's bytes. This is mass on
///   paths that can still produce an allowed value, whatever spelling the
///   tokenizer gave them.
/// * FORMATTING (unquoted fields only) — nothing but JSON whitespace (space,
///   tab, CR, LF — the four bytes JSON allows between values, NOT Unicode
///   whitespace: an NBSP would make the document invalid and is honestly
///   outside): layout, not an answer. Excluded from both sides.
/// * OUTSIDE — everything else: the mass the model would rather spend on some
///   other continuation than any allowed value.
///
/// The escape is outside over inside-plus-outside — the gap conditioned on the
/// model saying anything at all. Prefix matching is deliberately one-way: a
/// token that BEGINS with an option and carries on (`true,` fused into one
/// token, were the vocabulary to hold one) counts outside, because so does
/// `yesterday` against the option `yes`, and the canonical tokenizations that
/// carry real mass never fuse across the value's edge (check_seams refuses the
/// plans where they would). Matching is also canonical-spelling-only for a
/// quoted field: JSON's alternate spellings of the same string (`\/` for `/`,
/// `\uXXXX` forms) count outside, the same stance check_seams takes on
/// non-canonical tokenizations — negligible mass, and an escape that read them
/// as inside would claim to understand a value the assembler would never write.
fn escape_mass(probs: &[f64], decoded: &[Vec<u8>], field: &ScoredField) -> Result<f64> {
    // The row is cut to the sampler's encodable bound and the byte table to the
    // tokenizer's vocab_size; they are the same number by construction
    // (`Generator::load` derives one from the other), and this is where that
    // coupling is enforced — a silent zip truncation would drop the surplus
    // side's mass from BOTH classes and read the escape low with no error.
    ensure!(
        probs.len() == decoded.len(),
        "escape: the probability row covers {} ids but the decoded vocabulary {}",
        probs.len(),
        decoded.len()
    );
    let json_ws = |b: &u8| matches!(b, b' ' | b'\t' | b'\r' | b'\n');
    let mut inside = 0.0f64;
    let mut outside = 0.0f64;
    for (bytes, &p) in decoded.iter().zip(probs) {
        let body: &[u8] = if field.quoted {
            bytes
        } else {
            &bytes[bytes.iter().take_while(|b| json_ws(b)).count()..]
        };
        if body.is_empty() {
            // Pure JSON whitespace (or an id with no bytes): formatting.
            continue;
        }
        if field
            .options
            .iter()
            .any(|option| option.text.as_bytes().starts_with(body))
        {
            inside += p;
        } else {
            outside += p;
        }
    }
    let content = inside + outside;
    if content <= 0.0 {
        // Every scrap of mass was formatting: there is no content distribution
        // to read a gap off, and 0 (rather than an arbitrary ratio) says so.
        return Ok(0.0);
    }
    Ok((outside / content).clamp(0.0, 1.0))
}

/// `softmax` over option scores at `temperature`, max-subtracted so a long
/// option's very negative logprob cannot underflow the whole row.
fn renormalize(scores: &[f64], temperature: f64) -> Vec<f64> {
    let scaled: Vec<f64> = scores.iter().map(|score| score / temperature).collect();
    let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentiated: Vec<f64> = scaled.iter().map(|score| (score - max).exp()).collect();
    let total: f64 = exponentiated.iter().sum();
    exponentiated.iter().map(|value| value / total).collect()
}

/// The highest-scoring option, ties going to the one the schema listed first.
fn argmax_index(scores: &[f64]) -> usize {
    let mut best = f64::NEG_INFINITY;
    let mut index = 0;
    for (at, &score) in scores.iter().enumerate() {
        if score > best {
            best = score;
            index = at;
        }
    }
    index
}

/// One field's entry in the response `json`.
fn report(field: &ScoredField, pick: &Pick) -> Value {
    let chosen = &field.options[pick.index];
    match field.mode {
        ScoreMode::Bare => chosen.value.clone(),
        ScoreMode::Value => json!({ "value": chosen.value, "score": pick.probs[pick.index] }),
        ScoreMode::All => {
            let scores: Map<String, Value> = field
                .options
                .iter()
                .zip(&pick.probs)
                .map(|(option, prob)| (option.key.clone(), json!(prob)))
                .collect();
            json!({
                "value": chosen.value,
                "score": pick.probs[pick.index],
                "scores": scores,
                "escape": pick.escape,
            })
        }
    }
}

/// Render, encode and validate one item. `shared_prefix` is the request-level
/// text prepended to the first message's content ([`BatchRequest::shared_prefix`]),
/// already normalized to `None` when empty.
fn prepare_item(
    tokenizer: &LagunaTokenizer,
    max_ctx: usize,
    item: &BatchItem,
    defaults: &ItemDefaults,
    shared_prefix: Option<&str>,
    label: &str,
    model: Model,
) -> Result<Prepared> {
    let dialect = model.chat_dialect();
    ensure!(
        shared_prefix.is_none() || !item.messages.is_empty(),
        "the request's shared_prefix has no first message to prepend to; this item has none"
    );
    let messages = item
        .messages
        .iter()
        .enumerate()
        .map(|(at, message)| chat_message(message, if at == 0 { shared_prefix } else { None }))
        .collect::<Result<Vec<_>>>()?;
    let (opts, continuation) = resolve_render(item, defaults, label, dialect)?;
    let (tokens, prefix_len, starts_in_thinking) =
        encode_item(tokenizer, &messages, &opts, continuation.as_ref())?;
    // The renderer writes the prefix verbatim, so what it was asked to render
    // is what the answer continues from.
    let prefix_text = continuation
        .and_then(|c| c.prefix)
        .unwrap_or_else(String::new);

    let scored = match &item.schema {
        Some(schema) => scored_plan(tokenizer, schema)?,
        None => None,
    };
    // The runner writes a scored item's whole document, so there is nowhere for
    // a caller's opening fragment to go: it would sit in front of a second
    // complete object rather than becoming its head.
    ensure!(
        scored.is_none() || prefix_len == 0,
        "scored output assembles the whole document, so the item cannot also supply a prefill"
    );

    let max_tokens = item
        .max_tokens
        .or(defaults.max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    ensure!(max_tokens > 0, "max_tokens must be at least 1");
    // Phrased as a subtraction rather than as `tokens.len() + max_tokens`: a
    // caller's `max_tokens` is arbitrary, and release builds wrap on overflow
    // rather than panic — a huge budget would come out small and pass.
    ensure!(
        max_tokens <= max_ctx.saturating_sub(tokens.len()),
        "the prompt ({} tokens) plus max_tokens ({max_tokens}) exceeds the context ({max_ctx}); \
         shorten the item or lower its max_tokens",
        tokens.len(),
    );

    // A scored item writes its structure rather than drawing it, so the only
    // decode that runs under a grammar mask is a schema the compiler took.
    let constrained = scored.is_none() && item.schema.is_some();

    Ok(Prepared {
        tokens,
        prefix_len,
        prefix_text,
        starts_in_thinking,
        max_tokens,
        // The item's own thinking state is what its penalty default is keyed
        // to, and `resolve_render` has just settled it.
        sampling: resolve_sampling(
            model,
            opts.enable_thinking,
            constrained,
            defaults.sampling.as_ref(),
            item.sampling.as_ref(),
        ),
        // A scored schema never reaches the grammar compiler: llguidance has
        // never heard of `include_score`, and the scored path masks nothing.
        schema: match scored {
            Some(_) => None,
            None => item.schema.clone(),
        },
        scored,
    })
}

/// Turn a wire message into a renderer message, with the request's shared
/// prefix (if this is the message it lands on) prepended to the content.
fn chat_message(message: &BatchMessage, shared_prefix: Option<&str>) -> Result<Message> {
    let content = match shared_prefix {
        Some(prefix) => format!("{prefix}{}", message.content),
        None => message.content.clone(),
    };
    match message.role.as_str() {
        "system" => Ok(Message::System(content)),
        "user" => Ok(Message::User(content)),
        "assistant" => Ok(Message::Assistant {
            content,
            reasoning: message.thinking.clone().filter(|text| !text.is_empty()),
            tool_calls: Vec::new(),
        }),
        "tool" => Ok(Message::ToolResponse(content)),
        other => bail!("unknown message role {other:?} (expected system, user, assistant or tool)"),
    }
}

/// Resolve an item's thinking and prefill settings into the renderer's
/// [`ChatOptions`] and [`Continuation`].
///
/// Three shapes, one per [`ThinkingSpec`] value. Off renders a closed empty
/// `<think>` block and starts the model in the answer. On leaves the block open
/// and the first decoded token is reasoning. Injected reasoning goes inside the
/// block and closes it, which is also what lets a prefill follow.
///
/// The one refused combination is a prefill with thinking merely on: the prefix
/// would be rendered inside the open thinking span and read back as reasoning
/// (`chat.rs` refuses it too — this is where it gets a message naming the two
/// fields).
///
/// `reasoning_effort` layers the same way `thinking` does — item over defaults
/// over the template's own level — and is a 3.8 template parameter: supplied
/// against a 3.6 checkpoint it is refused rather than silently dropped, the
/// same rule as every other surface (the CLI's startup error, serve's 400).
/// With thinking off it is accepted and inert (the template reads it only
/// inside `enable_thinking`'s guard), and it is independent of injected
/// reasoning. `label` is the name the refusal reports, [`run_batch`]'s own.
fn resolve_render(
    item: &BatchItem,
    defaults: &ItemDefaults,
    label: &str,
    dialect: ChatDialect,
) -> Result<(ChatOptions, Option<Continuation>)> {
    let effort = item.reasoning_effort.or(defaults.reasoning_effort);
    if effort.is_some() && !dialect.supports_reasoning_effort() {
        bail!(
            "reasoning_effort: {label} renders a chat template with no reasoning_effort \
             parameter (it is a Qwen 3.8 template feature)"
        );
    }
    let thinking = item
        .thinking
        .clone()
        .or_else(|| defaults.thinking.clone())
        .unwrap_or(ThinkingSpec::Enabled(false));
    let prefill = item.prefill.clone().filter(|text| !text.is_empty());

    let (enable_thinking, injected) = match &thinking {
        ThinkingSpec::Enabled(on) => (*on, None),
        // An empty string is how a caller spells "not supplied" without
        // dropping the key; it asks for thinking, not for injected reasoning.
        ThinkingSpec::Injected(text) if text.is_empty() => (true, None),
        ThinkingSpec::Injected(text) => (true, Some(text.clone())),
    };
    if enable_thinking && prefill.is_some() && injected.is_none() {
        bail!(
            "prefill needs the thinking span closed: set thinking to false, or to the reasoning \
             the turn should carry"
        );
    }

    let base = ChatOptions::for_dialect(dialect);
    let opts = ChatOptions {
        enable_thinking,
        // Absent at both levels, the template's own effort default stands.
        reasoning_effort: effort.unwrap_or(base.reasoning_effort),
        // The checkpoint's template decides everything else, its
        // preserve_thinking default included.
        ..base
    };
    let continuation = (injected.is_some() || prefill.is_some()).then(|| Continuation {
        close_thinking: injected.is_some(),
        thinking: injected,
        prefix: prefill,
    });
    Ok((opts, continuation))
}

/// Layer a request's sampling overrides onto the batch default.
///
/// The presence penalty starts from the card rather than from
/// [`BATCH_SAMPLING`]: it is the one sampling value keyed to the checkpoint and
/// the item's thinking mode, and batch resolves it the way every other surface
/// does (`SamplerOptions::recommended_for`).
///
/// EXCEPT under a grammar mask, where the default is 0. The card's 1.5 exists to
/// keep prose from circling, and it moves a greedy pick by moving logits: on a
/// constrained decode the tokens that repeat are the STRUCTURE — the `,` between
/// every pair of fields, the quotes around every key — so penalizing what the
/// answer has already emitted biases the choice between two structurally legal
/// continuations, `,` against `}`, for a reason that has nothing to do with the
/// document. The mask already guarantees the shape; the penalty can only
/// mis-rank inside it. An item or request that names a penalty still gets it,
/// constrained or not: the rule moves the DEFAULT, not the knob.
fn resolve_sampling(
    model: Model,
    thinking: bool,
    constrained: bool,
    defaults: Option<&SamplingSpec>,
    item: Option<&SamplingSpec>,
) -> SamplerOptions {
    let mut opts = SamplerOptions {
        presence_penalty: if constrained {
            0.0
        } else {
            SamplerOptions::recommended_for(model, thinking).presence_penalty
        },
        ..BATCH_SAMPLING
    };
    for spec in [defaults, item].into_iter().flatten() {
        if let Some(temperature) = spec.temperature {
            opts.temperature = temperature;
        }
        if let Some(top_p) = spec.top_p {
            opts.top_p = top_p;
        }
        if let Some(top_k) = spec.top_k {
            opts.top_k = top_k;
        }
        if let Some(presence_penalty) = spec.presence_penalty {
            opts.presence_penalty = presence_penalty;
        }
        if let Some(seed) = spec.seed {
            opts.seed = seed;
        }
    }
    opts
}

/// Render a conversation and encode it, returning the ids, the token count of
/// any rendered response prefix, and whether the prompt ends inside an open
/// thinking span.
///
/// The context and the generation header tokenize independently — the header
/// opens with the added token `<|im_start|>`, which BPE never merges into its
/// neighbours — and so does the response prefix inside the header, which the
/// added token `</think>` fences off. Encoding in spans is what counts the
/// prefix rather than guessing it.
///
/// Every span is encoded with its client-content byte ranges, so an added-token
/// string inside a message body — a document quoting `<|im_end|>` — stays plain
/// text instead of becoming a control token the decode loop would stop on.
fn encode_item(
    tokenizer: &LagunaTokenizer,
    messages: &[Message],
    opts: &ChatOptions,
    continuation: Option<&Continuation>,
) -> Result<(Vec<u32>, usize, bool)> {
    let chat::PromptParts {
        context,
        header,
        content_ranges,
        header_content_ranges,
        header_prefix_start,
        starts_in_thinking,
        ..
    } = chat::build_prompt_parts_with_spans_continued(messages, opts, continuation)?;

    let mut tokens = tokenizer.encode_prompt(&context, &content_ranges)?;
    let prefix_len = match header_prefix_start {
        Some(split) => {
            let (head, prefix) = split_content_ranges(&header_content_ranges, split);
            tokens.extend(tokenizer.encode_prompt(&header[..split], &head)?);
            let prefix = tokenizer.encode_prompt(&header[split..], &prefix)?;
            let prefix_len = prefix.len();
            tokens.extend(prefix);
            prefix_len
        }
        None => {
            tokens.extend(tokenizer.encode_prompt(&header, &header_content_ranges)?);
            0
        }
    };
    ensure!(!tokens.is_empty(), "the prompt encoded to zero tokens");
    Ok((tokens, prefix_len, starts_in_thinking))
}

/// Divide client-content byte ranges at `at` into the ranges before it and the
/// ranges after, the latter rebased onto that half. A range is clipped rather
/// than assigned to a side, so one spanning the split would keep both halves
/// marked as client content.
fn split_content_ranges(
    ranges: &[Range<usize>],
    at: usize,
) -> (Vec<Range<usize>>, Vec<Range<usize>>) {
    let head = ranges
        .iter()
        .map(|r| r.start.min(at)..r.end.min(at))
        .filter(|r| !r.is_empty())
        .collect();
    let tail = ranges
        .iter()
        .map(|r| r.start.max(at) - at..r.end.max(at) - at)
        .filter(|r| !r.is_empty())
        .collect();
    (head, tail)
}

fn failed_item(id: &str, error: String) -> ItemResponse {
    ItemResponse {
        id: id.to_string(),
        content: String::new(),
        text: String::new(),
        json: None,
        finish_reason: FinishReason::Error,
        usage: Usage::default(),
        error: Some(error),
    }
}

fn elapsed_ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str) -> BatchItem {
        BatchItem {
            id: id.to_string(),
            messages: vec![BatchMessage {
                role: "user".into(),
                content: "hello".into(),
                thinking: None,
            }],
            schema: None,
            thinking: None,
            reasoning_effort: None,
            prefill: None,
            max_tokens: None,
            sampling: None,
        }
    }

    // An empty set has no prefix to share, and a lone sequence shares its whole
    // length with itself.
    #[test]
    fn lcp_of_nothing_and_of_one() {
        assert_eq!(longest_common_prefix(&[]), 0);
        assert_eq!(longest_common_prefix(&[&[1, 2, 3]]), 3);
    }

    #[test]
    fn lcp_of_identical_sequences_is_their_length() {
        let a: &[u32] = &[7, 8, 9, 10];
        assert_eq!(longest_common_prefix(&[a, a, a]), 4);
    }

    // The shared span stops at the first disagreement, and one outlier pulls it
    // down for the whole set.
    #[test]
    fn lcp_stops_at_the_first_divergence() {
        let a: &[u32] = &[1, 2, 3, 4, 5];
        let b: &[u32] = &[1, 2, 3, 9, 9];
        let c: &[u32] = &[1, 2, 7, 4, 5];
        assert_eq!(longest_common_prefix(&[a, b]), 3);
        assert_eq!(longest_common_prefix(&[a, b, c]), 2);
        assert_eq!(longest_common_prefix(&[a, &[9, 9]]), 0);
    }

    // A sequence that is a strict prefix of another caps the shared span at its
    // own length.
    #[test]
    fn lcp_is_capped_by_the_shortest_sequence() {
        let long: &[u32] = &[1, 2, 3, 4, 5];
        let short: &[u32] = &[1, 2, 3];
        assert_eq!(longest_common_prefix(&[long, short]), 3);
        assert_eq!(longest_common_prefix(&[short, long]), 3);
    }

    // Below the floor the snapshot is not worth taking, and every item runs
    // from a reset cache instead.
    #[test]
    fn a_short_shared_prefix_falls_back_to_cold_items() {
        let a: Vec<u32> = (0..200).collect();
        let mut b = a.clone();
        b[MIN_SHARED_PREFIX - 2] = 9999;
        assert_eq!(shared_prefix_len(&[&a, &b]), None);
    }

    #[test]
    fn a_long_shared_prefix_is_snapshotted() {
        let a: Vec<u32> = (0..200).collect();
        let mut b = a.clone();
        b[MIN_SHARED_PREFIX + 10] = 9999;
        assert_eq!(shared_prefix_len(&[&a, &b]), Some(MIN_SHARED_PREFIX + 10));
    }

    // Identical items would share every token; the span is held one short so
    // each still has a tail to prefill, which is what keeps the snapshot valid
    // for the items after the first.
    #[test]
    fn the_shared_span_leaves_every_item_a_tail() {
        let a: Vec<u32> = (0..200).collect();
        assert_eq!(shared_prefix_len(&[&a, &a]), Some(199));
        // And equally when one item's prompt is a strict prefix of another's.
        let b: Vec<u32> = (0..120).collect();
        assert_eq!(shared_prefix_len(&[&a, &b]), Some(119));
    }

    // A single item shares its prompt with nobody: snapshotting it would cost a
    // restore and save nothing.
    #[test]
    fn a_lone_item_never_takes_the_snapshot_path() {
        let a: Vec<u32> = (0..200).collect();
        assert_eq!(shared_prefix_len(&[&a]), None);
        assert_eq!(shared_prefix_len(&[]), None);
    }

    // Batch sampling is greedy until something says otherwise, and the two
    // override layers apply in order without either restating the whole set.
    #[test]
    fn sampling_layers_default_then_item() {
        let plain = resolve_sampling(Model::Qwen35BA3B, true, false, None, None);
        assert_eq!(plain.temperature, 0.0);
        // The one value that does NOT come from BATCH_SAMPLING: the card's,
        // for this checkpoint in this mode.
        assert_eq!(plain.presence_penalty, 1.5);
        assert_eq!(
            resolve_sampling(Model::Qwen27B, true, false, None, None).presence_penalty,
            0.0
        );

        let defaults = SamplingSpec {
            temperature: Some(0.7),
            seed: Some(7),
            ..Default::default()
        };
        let layered = resolve_sampling(Model::Qwen35BA3B, true, false, Some(&defaults), None);
        assert_eq!(layered.temperature, 0.7);
        assert_eq!(layered.seed, 7);
        assert_eq!(layered.top_k, BATCH_SAMPLING.top_k);

        let item = SamplingSpec {
            temperature: Some(0.0),
            ..Default::default()
        };
        let both = resolve_sampling(Model::Qwen35BA3B, true, false, Some(&defaults), Some(&item));
        assert_eq!(both.temperature, 0.0);
        // Untouched by the item, so the default layer still shows through.
        assert_eq!(both.seed, 7);

        // An explicit penalty beats the card, including a zero.
        let off = SamplingSpec {
            presence_penalty: Some(0.0),
            ..Default::default()
        };
        assert_eq!(
            resolve_sampling(Model::Qwen35BA3B, true, false, None, Some(&off)).presence_penalty,
            0.0
        );
    }

    // A grammar-masked item takes no presence penalty by default. The card's
    // 1.5 is a prose knob: under a mask the repeated tokens are the document's
    // own punctuation, and penalizing them tilts a greedy pick between two
    // structurally legal continuations for a reason the caller never asked for.
    // Naming a penalty still gets it — the rule moves the default, not the knob.
    #[test]
    fn a_grammar_masked_item_defaults_to_no_presence_penalty() {
        // Same checkpoint, same thinking mode, the two sides of the rule.
        assert_eq!(
            resolve_sampling(Model::Qwen35BA3B, true, false, None, None).presence_penalty,
            1.5,
            "unconstrained free decode keeps the card default"
        );
        assert_eq!(
            resolve_sampling(Model::Qwen35BA3B, true, true, None, None).presence_penalty,
            0.0,
            "a grammar mask drops it"
        );

        // Pinned at either layer, it survives the mask.
        let pinned = SamplingSpec {
            presence_penalty: Some(0.8),
            ..Default::default()
        };
        assert_eq!(
            resolve_sampling(Model::Qwen35BA3B, true, true, Some(&pinned), None).presence_penalty,
            0.8,
            "a request default pins it through the mask"
        );
        assert_eq!(
            resolve_sampling(Model::Qwen35BA3B, true, true, None, Some(&pinned)).presence_penalty,
            0.8,
            "and so does an item"
        );

        // The rest of the sampling set is untouched by the rule.
        let masked = resolve_sampling(Model::Qwen35BA3B, true, true, None, None);
        assert_eq!(masked.temperature, BATCH_SAMPLING.temperature);
        assert_eq!(masked.top_k, BATCH_SAMPLING.top_k);
        assert_eq!(masked.top_p, BATCH_SAMPLING.top_p);
        assert_eq!(masked.seed, BATCH_SAMPLING.seed);
    }

    // Thinking is off unless a request asks for it, which is the divergence
    // from the chat surface batch makes on purpose.
    #[test]
    fn thinking_defaults_off() {
        let (opts, continuation) = resolve_render(
            &item("a"),
            &ItemDefaults::default(),
            "Qwen3.6-35B-A3B",
            ChatDialect::Qwen36,
        )
        .unwrap();
        assert!(!opts.enable_thinking);
        assert!(continuation.is_none());
    }

    #[test]
    fn thinking_true_leaves_the_span_open() {
        let mut spec = item("a");
        spec.thinking = Some(ThinkingSpec::Enabled(true));
        let (opts, continuation) = resolve_render(
            &spec,
            &ItemDefaults::default(),
            "Qwen3.6-35B-A3B",
            ChatDialect::Qwen36,
        )
        .unwrap();
        assert!(opts.enable_thinking);
        assert!(continuation.is_none());
    }

    // Injected reasoning goes inside the block and closes it, so decoding
    // starts in the answer.
    #[test]
    fn injected_thinking_closes_the_span() {
        let mut spec = item("a");
        spec.thinking = Some(ThinkingSpec::Injected("the label is positive".into()));
        let (opts, continuation) = resolve_render(
            &spec,
            &ItemDefaults::default(),
            "Qwen3.6-35B-A3B",
            ChatDialect::Qwen36,
        )
        .unwrap();
        assert!(opts.enable_thinking);
        let continuation = continuation.expect("injected reasoning renders a continuation");
        assert_eq!(
            continuation.thinking.as_deref(),
            Some("the label is positive")
        );
        assert!(continuation.close_thinking);
        assert!(continuation.prefix.is_none());
    }

    #[test]
    fn prefill_rides_the_continuation() {
        let mut spec = item("a");
        spec.prefill = Some("{\"label\":".into());
        let (opts, continuation) = resolve_render(
            &spec,
            &ItemDefaults::default(),
            "Qwen3.6-35B-A3B",
            ChatDialect::Qwen36,
        )
        .unwrap();
        assert!(!opts.enable_thinking);
        let continuation = continuation.expect("a prefill renders a continuation");
        assert_eq!(continuation.prefix.as_deref(), Some("{\"label\":"));
        assert!(!continuation.close_thinking);
    }

    // A prefill under an open thinking span would be read back as reasoning.
    #[test]
    fn prefill_with_thinking_merely_on_is_refused() {
        let mut spec = item("a");
        spec.thinking = Some(ThinkingSpec::Enabled(true));
        spec.prefill = Some("{".into());
        let error = resolve_render(
            &spec,
            &ItemDefaults::default(),
            "Qwen3.6-35B-A3B",
            ChatDialect::Qwen36,
        )
        .expect_err("a prefix inside the thinking span has no rendering");
        assert!(
            error.to_string().contains("thinking span closed"),
            "{error}"
        );
    }

    // An item's own setting wins over the batch default, in both directions.
    #[test]
    fn an_item_overrides_the_default_thinking() {
        let defaults = ItemDefaults {
            thinking: Some(ThinkingSpec::Enabled(true)),
            ..Default::default()
        };
        let (opts, _) = resolve_render(
            &item("a"),
            &defaults,
            "Qwen3.6-35B-A3B",
            ChatDialect::Qwen36,
        )
        .unwrap();
        assert!(opts.enable_thinking);

        let mut spec = item("a");
        spec.thinking = Some(ThinkingSpec::Enabled(false));
        let (opts, _) =
            resolve_render(&spec, &defaults, "Qwen3.6-35B-A3B", ChatDialect::Qwen36).unwrap();
        assert!(!opts.enable_thinking);
    }

    // Effort layers exactly like thinking: the item's level wins over the
    // defaults', and with neither supplied the template's own default (xhigh)
    // stands untouched.
    #[test]
    fn an_item_overrides_the_default_reasoning_effort() {
        let defaults = ItemDefaults {
            reasoning_effort: Some(ReasoningEffort::Low),
            ..Default::default()
        };
        let (opts, _) =
            resolve_render(&item("a"), &defaults, "Qwen3.8-27B", ChatDialect::Qwen38).unwrap();
        assert_eq!(opts.reasoning_effort, ReasoningEffort::Low);

        let mut spec = item("a");
        spec.reasoning_effort = Some(ReasoningEffort::Medium);
        let (opts, _) =
            resolve_render(&spec, &defaults, "Qwen3.8-27B", ChatDialect::Qwen38).unwrap();
        assert_eq!(opts.reasoning_effort, ReasoningEffort::Medium);

        let (opts, _) = resolve_render(
            &item("a"),
            &ItemDefaults::default(),
            "Qwen3.8-27B",
            ChatDialect::Qwen38,
        )
        .unwrap();
        assert_eq!(opts.reasoning_effort, ReasoningEffort::Xhigh);
    }

    // The resolved options drive the 3.8 renderer: low writes its preamble
    // sentence into the system block; medium writes no preamble at all, so with
    // no client system message the block itself disappears. An effort-absent
    // item is NOT the no-preamble render — it is the template's xhigh default,
    // sentence and all.
    #[test]
    fn the_effort_level_renders_the_38_preamble() {
        let msgs = [Message::User("hello".into())];
        let render = |spec: &BatchItem| {
            let (opts, _) = resolve_render(
                spec,
                &ItemDefaults::default(),
                "Qwen3.8-27B",
                ChatDialect::Qwen38,
            )
            .unwrap();
            chat::build_prompt(&msgs, &opts).unwrap()
        };
        let mut thinking = item("a");
        thinking.thinking = Some(ThinkingSpec::Enabled(true));
        let absent = render(&thinking);
        assert!(
            absent.starts_with("<|im_start|>system\nReasoning effort is set to xhigh."),
            "{absent}"
        );

        let mut low = thinking.clone();
        low.reasoning_effort = Some(ReasoningEffort::Low);
        assert_eq!(
            render(&low),
            "<|im_start|>system\nReasoning effort is set to low. Keep your thinking brief and \
             focused, moving directly to the conclusion without unnecessary elaboration.\
             <|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );

        let mut medium = thinking.clone();
        medium.reasoning_effort = Some(ReasoningEffort::Medium);
        assert_eq!(
            render(&medium),
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    // reasoning_effort is a 3.8 template parameter: supplied against a 3.6
    // checkpoint it is refused rather than silently dropped, matching every
    // other surface. The failure is the item's — a defaults-level effort
    // reaches the renderer through each item and fails each the same way — and
    // the message names the checkpoint that cannot honor it.
    #[test]
    fn a_reasoning_effort_on_a_36_checkpoint_fails_the_item() {
        let mut spec = item("a");
        spec.reasoning_effort = Some(ReasoningEffort::Low);
        let error = resolve_render(
            &spec,
            &ItemDefaults::default(),
            "Qwen3.6-27B",
            ChatDialect::Qwen36,
        )
        .expect_err("the 3.6 template has no reasoning_effort parameter")
        .to_string();
        assert!(error.contains("Qwen3.6-27B"), "{error}");
        assert!(error.contains("Qwen 3.8 template feature"), "{error}");

        let defaults = ItemDefaults {
            reasoning_effort: Some(ReasoningEffort::Xhigh),
            ..Default::default()
        };
        let error = resolve_render(
            &item("a"),
            &defaults,
            "Qwen3.6-35B-A3B",
            ChatDialect::Qwen36,
        )
        .expect_err("a defaults-level effort fails the same way an item's does")
        .to_string();
        assert!(error.contains("Qwen3.6-35B-A3B"), "{error}");
    }

    // The template reads the effort only inside `enable_thinking`'s guard, so
    // effort with thinking off is accepted and inert: the render is
    // byte-identical to the effort-less one.
    #[test]
    fn effort_with_thinking_off_is_accepted_and_inert() {
        let msgs = [Message::User("hello".into())];
        let render = |spec: &BatchItem| {
            let (opts, _) = resolve_render(
                spec,
                &ItemDefaults::default(),
                "Qwen3.8-27B",
                ChatDialect::Qwen38,
            )
            .unwrap();
            chat::build_prompt(&msgs, &opts).unwrap()
        };
        // Thinking defaults off on this surface; only the effort is supplied.
        let mut with_effort = item("a");
        with_effort.reasoning_effort = Some(ReasoningEffort::Low);
        assert_eq!(render(&with_effort), render(&item("a")));
    }

    // The preamble and injected reasoning are independent: the effort still
    // resolves and the caller's reasoning still lands in the closed block.
    #[test]
    fn effort_rides_alongside_injected_reasoning() {
        let mut spec = item("a");
        spec.thinking = Some(ThinkingSpec::Injected("the tone is upbeat".into()));
        spec.reasoning_effort = Some(ReasoningEffort::Low);
        let (opts, continuation) = resolve_render(
            &spec,
            &ItemDefaults::default(),
            "Qwen3.8-27B",
            ChatDialect::Qwen38,
        )
        .unwrap();
        assert_eq!(opts.reasoning_effort, ReasoningEffort::Low);
        assert!(opts.enable_thinking);
        let continuation = continuation.expect("injected reasoning renders a continuation");
        assert!(continuation.close_thinking);
        assert_eq!(continuation.thinking.as_deref(), Some("the tone is upbeat"));
    }

    // The wire shapes survive a round trip, `thinking` in all three of its
    // forms included — it is one field holding a bool or a string.
    #[test]
    fn a_request_round_trips() {
        let text = r#"{
            "model": "Qwen3.6-35B-A3B",
            "defaults": { "max_tokens": 512, "sampling": { "temperature": 0 }, "thinking": false,
                          "reasoning_effort": "low" },
            "items": [
                { "id": "sentiment",
                  "messages": [
                    { "role": "system", "content": "You label text." },
                    { "role": "user", "content": "great news" }
                  ],
                  "schema": { "type": "object" },
                  "thinking": "the tone is upbeat",
                  "reasoning_effort": "medium",
                  "prefill": "{",
                  "max_tokens": 64,
                  "sampling": { "top_k": 1 } },
                { "id": "topic",
                  "messages": [{ "role": "user", "content": "hi" }],
                  "thinking": true }
            ]
        }"#;
        let request: BatchRequest = serde_json::from_str(text).unwrap();
        assert_eq!(request.model.as_deref(), Some("Qwen3.6-35B-A3B"));
        assert_eq!(request.model().unwrap(), Model::Qwen35BA3B);
        assert_eq!(request.items.len(), 2);
        assert_eq!(
            request.items[0].thinking,
            Some(ThinkingSpec::Injected("the tone is upbeat".into()))
        );
        assert_eq!(request.items[1].thinking, Some(ThinkingSpec::Enabled(true)));
        assert_eq!(
            request.defaults.thinking,
            Some(ThinkingSpec::Enabled(false))
        );
        assert_eq!(
            request.defaults.reasoning_effort,
            Some(ReasoningEffort::Low)
        );
        assert_eq!(
            request.items[0].reasoning_effort,
            Some(ReasoningEffort::Medium)
        );
        assert_eq!(request.items[1].reasoning_effort, None);

        let again: BatchRequest =
            serde_json::from_str(&serde_json::to_string(&request).unwrap()).unwrap();
        assert_eq!(again.items[0].thinking, request.items[0].thinking);
        assert_eq!(again.items[0].prefill, request.items[0].prefill);
        assert_eq!(
            again.items[0].reasoning_effort,
            request.items[0].reasoning_effort
        );
        assert_eq!(again.defaults.reasoning_effort, Some(ReasoningEffort::Low));
        assert_eq!(again.items[1].id, "topic");
    }

    // The wire spelling is the template's own three; the OpenAI dialect's wider
    // effort scale ("none"/"minimal"/"high"/...) does not parse here.
    #[test]
    fn a_reasoning_effort_outside_the_templates_spellings_is_refused() {
        let text = r#"{ "items": [], "defaults": { "reasoning_effort": "high" } }"#;
        let error = serde_json::from_str::<BatchRequest>(text)
            .expect_err("only the template's own spellings parse")
            .to_string();
        assert!(error.contains("unknown reasoning effort"), "{error}");
    }

    // A typo'd field is a request error rather than a silently ignored setting.
    #[test]
    fn an_unknown_field_is_refused() {
        let text = r#"{ "items": [], "temperature": 0 }"#;
        assert!(serde_json::from_str::<BatchRequest>(text).is_err());
    }

    #[test]
    fn a_response_round_trips() {
        let response = BatchResponse {
            model: "Qwen3.6-35B-A3B".into(),
            items: vec![
                ItemResponse {
                    id: "sentiment".into(),
                    content: "{\"label\":\"a\"}".into(),
                    text: "{\"label\":\"a\"}".into(),
                    json: Some(serde_json::json!({ "label": "a" })),
                    finish_reason: FinishReason::Stop,
                    usage: Usage {
                        prompt_tokens: 210,
                        cached_prefix_tokens: 190,
                        completion_tokens: 9,
                    },
                    error: None,
                },
                failed_item("topic", "the prompt encoded to zero tokens".into()),
            ],
            stats: BatchStats {
                shared_prefix_tokens: 190,
                snapshot_ms: 41.5,
                items: 2,
                prefill_tokens: 230,
                prefill_ms: 260.0,
                decode_tokens: 9,
                decode_ms: 95.0,
                load_ms: 2900.0,
                total_ms: 3400.0,
            },
        };
        let text = serde_json::to_string(&response).unwrap();
        let again: BatchResponse = serde_json::from_str(&text).unwrap();
        assert_eq!(again.items[0].finish_reason, FinishReason::Stop);
        assert_eq!(again.items[0].json, response.items[0].json);
        assert_eq!(again.items[1].finish_reason, FinishReason::Error);
        assert!(again.items[1].error.is_some());
        assert_eq!(again.stats.shared_prefix_tokens, 190);
        // A successful item carries no `error` key at all, rather than a null.
        assert!(!text.contains("\"error\":null"));
    }

    /// The CLI reads the document's `model` in the CLI's own vocabulary: the
    /// short aliases and the full names alike, so a document answered by an
    /// OFFICIAL checkpoint can be resubmitted unedited — the server labels those
    /// with the full name, which parses here.
    ///
    /// That round trip does not hold for a server started on a GGUF that is none
    /// of the official checkpoints: it labels its answers with that file's own
    /// id, which names no checkpoint and so does not parse. Resubmitting such a
    /// document to the CLI needs `-m <that file>` and the field dropped, which
    /// is the honest outcome — the alternative would be a label claiming an
    /// official checkpoint ran.
    #[test]
    fn the_model_comes_from_the_payload() {
        let mut request: BatchRequest = serde_json::from_str(r#"{ "items": [] }"#).unwrap();
        // Absent means the default a cache-moving surface can actually run,
        // which is serve's too — not the plain CLI default, which batch snapshots
        // its way out of running.
        assert_eq!(request.model().unwrap(), Model::default_servable());
        request.model = Some("27b".into());
        assert_eq!(request.model().unwrap(), Model::Qwen27B);
        request.model = Some("Qwen3.8-27B".into());
        assert_eq!(request.model().unwrap(), Model::Qwen3827B);
        request.model = Some("13b".into());
        assert!(request.model().is_err());
        // A custom server's own label is not a checkpoint name, and says so
        // rather than resolving to something plausible.
        request.model = Some("my-finetune-Q4_K_M".into());
        assert!(request.model().is_err());
    }

    /// Every checkpoint resolves, in every spelling the field accepts.
    ///
    /// The qwen4exp checkpoint used to be refused here — batch snapshots the
    /// items' shared prefix and rescores fields off it, and until 2026-08-30 no
    /// cache image carried its QSA raw keys or its PLE state. Now that they do,
    /// naming it is an ordinary selection, and the field's only refusals are
    /// spellings that name no checkpoint at all.
    #[test]
    fn every_checkpoint_can_be_named_in_the_model_field() {
        let mut request: BatchRequest = serde_json::from_str(r#"{ "items": [] }"#).unwrap();
        for model in crate::hub::MODELS {
            request.model = Some(model.full_name().into());
            assert_eq!(request.model().unwrap(), model);
            request.model = Some(model.to_string());
            assert_eq!(request.model().unwrap(), model);
        }
        assert_eq!(
            {
                request.model = Some("Qwen3.8-Flash-Next".into());
                request.model().unwrap()
            },
            Model::Qwen38FlashNext
        );
    }

    fn prepared(schema: Option<Value>, prefix_text: &str) -> Prepared {
        Prepared {
            tokens: vec![1, 2, 3],
            prefix_len: 0,
            prefix_text: prefix_text.to_string(),
            starts_in_thinking: false,
            max_tokens: 32,
            sampling: BATCH_SAMPLING,
            schema,
            scored: None,
        }
    }

    fn outcome(text: &str, finish_reason: FinishReason) -> ItemOutcome {
        ItemOutcome {
            content: text.to_string(),
            text: text.to_string(),
            finish_reason,
            completion_tokens: 4,
            decode_tokens: 4,
            decode_secs: 0.1,
            json: None,
            error: None,
        }
    }

    #[test]
    fn an_unconstrained_item_has_no_parsed_value() {
        let item = prepared(None, "");
        assert!(constrained_value(&item, &outcome("hello", FinishReason::Stop)).is_none());
    }

    #[test]
    fn a_completed_value_parses() {
        let item = prepared(Some(serde_json::json!({ "type": "object" })), "");
        let value = constrained_value(&item, &outcome("{\"label\": \"a\"}", FinishReason::Stop))
            .unwrap()
            .unwrap();
        assert_eq!(value, serde_json::json!({ "label": "a" }));
    }

    // A prefill is prompt, so the model continues it rather than re-emitting
    // it: the parsed value is the prefix plus what was decoded.
    #[test]
    fn a_prefilled_value_parses_from_the_whole_document() {
        let item = prepared(Some(serde_json::json!({ "type": "object" })), "{\"label\":");
        let value = constrained_value(&item, &outcome(" \"a\"}", FinishReason::Stop))
            .unwrap()
            .unwrap();
        assert_eq!(value, serde_json::json!({ "label": "a" }));
    }

    // Stopping at the budget leaves the document half-written, and the item
    // reports the budget rather than a parse error about the truncation.
    #[test]
    fn a_truncated_value_reports_the_budget() {
        let item = prepared(Some(serde_json::json!({ "type": "object" })), "");
        let error = constrained_value(&item, &outcome("{\"label\":", FinishReason::Length))
            .unwrap()
            .expect_err("an incomplete value has nothing to parse");
        assert!(error.contains("32-token budget"), "{error}");
    }

    // ---- scored items

    /// A schema of the v1 shape, whose `label` property is annotated with
    /// `include_score` set to whatever the caller passes (`Value::Null` leaves
    /// the annotation off entirely).
    fn scored_schema(include: Value) -> Value {
        let mut label = serde_json::json!({ "enum": ["yes", "no"] });
        if !include.is_null() {
            label["include_score"] = include;
        }
        serde_json::json!({
            "type": "object",
            "properties": { "label": label },
            "required": ["label"],
            "additionalProperties": false,
        })
    }

    fn plan_of(schema: &Value) -> Result<Option<ScoredPlan>> {
        scored_plan(&LagunaTokenizer::embedded().unwrap(), schema)
    }

    // A schema that never mentions the annotation is not this path's business:
    // it keeps the grammar exactly as it was.
    #[test]
    fn an_unannotated_schema_stays_on_the_grammar_path() {
        assert!(plan_of(&scored_schema(Value::Null)).unwrap().is_none());
        assert!(
            plan_of(&serde_json::json!({ "type": "object" }))
                .unwrap()
                .is_none()
        );
    }

    // Both spellings of the annotation are accepted, and each says how much the
    // response reports.
    #[test]
    fn the_annotation_says_how_much_is_reported() {
        let plan = plan_of(&scored_schema(Value::Bool(true)))
            .unwrap()
            .expect("include_score routes the item to the scored path");
        assert_eq!(plan.fields[0].mode, ScoreMode::Value);

        let plan = plan_of(&scored_schema(Value::String("all".into())))
            .unwrap()
            .unwrap();
        assert_eq!(plan.fields[0].mode, ScoreMode::All);
    }

    // `include_score: false` is still an annotation: it routes the item here and
    // reports a bare value, rather than reading as "no annotation".
    #[test]
    fn a_false_annotation_still_scores_and_reports_bare() {
        let plan = plan_of(&scored_schema(Value::Bool(false)))
            .unwrap()
            .expect("the key is present, so the schema is a scored one");
        assert_eq!(plan.fields[0].mode, ScoreMode::Bare);
    }

    // A misspelled annotation value is refused rather than read as "off": the
    // caller asked for scores and would otherwise get an answer with none.
    #[test]
    fn an_unknown_annotation_value_is_refused() {
        let error = plan_of(&scored_schema(Value::String("some".into())))
            .expect_err("only true, false and \"all\" are defined");
        assert!(error.to_string().contains("include_score"), "{error}");
    }

    // A sibling property without the annotation is scored all the same — one
    // item assembles one document — and reports a bare value.
    #[test]
    fn an_unannotated_sibling_is_still_assembled() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "label": { "enum": ["yes", "no"], "include_score": true },
                "topic": { "type": "string", "enum": ["a", "b"] },
            },
            "required": ["label", "topic"],
            "additionalProperties": false,
        });
        let plan = plan_of(&schema).unwrap().unwrap();
        assert_eq!(plan.fields.len(), 2);
        assert_eq!(plan.fields[1].mode, ScoreMode::Bare);
        // Schema order, not alphabetical order, is what the document is written
        // in (`serde_json`'s preserve_order).
        assert_eq!(plan.fields[0].name, "label");
        assert_eq!(plan.fields[1].name, "topic");
    }

    // Every way the v1 shape can be missed is an error naming what is wrong, and
    // for a property, which property.
    #[test]
    fn the_scope_guard_names_what_it_refused() {
        let cases: Vec<(Value, &str)> = vec![
            (
                serde_json::json!({
                    "type": "array",
                    "properties": { "a": { "enum": ["x"], "include_score": true } },
                    "required": ["a"],
                    "additionalProperties": false,
                }),
                "\"type\"",
            ),
            (
                serde_json::json!({
                    "type": "object",
                    "properties": { "a": { "enum": ["x"], "include_score": true } },
                    "required": ["a"],
                }),
                "additionalProperties",
            ),
            (
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "a": { "enum": ["x"], "include_score": true },
                        "b": { "enum": ["y"] },
                    },
                    "required": ["a"],
                    "additionalProperties": false,
                }),
                "required",
            ),
            (
                serde_json::json!({
                    "type": "object",
                    "properties": { "a": { "type": "integer", "include_score": true } },
                    "required": ["a"],
                    "additionalProperties": false,
                }),
                "\"a\"",
            ),
            (
                serde_json::json!({
                    "type": "object",
                    "properties": { "a": { "enum": [1, 2], "include_score": true } },
                    "required": ["a"],
                    "additionalProperties": false,
                }),
                "\"a\"",
            ),
            (
                serde_json::json!({
                    "type": "object",
                    "properties": { "a": { "enum": ["x"], "include_score": true } },
                    "required": ["a"],
                    "additionalProperties": false,
                    "minProperties": 1,
                }),
                "minProperties",
            ),
            (
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "a": { "enum": ["x"], "include_score": true, "maxLength": 3 },
                    },
                    "required": ["a"],
                    "additionalProperties": false,
                }),
                "maxLength",
            ),
        ];
        for (schema, expected) in cases {
            let error = plan_of(&schema)
                .expect_err("this shape has no scoring definition")
                .to_string();
            assert!(error.contains(expected), "{expected} missing from: {error}");
        }
    }

    // An annotation buried where the shape guard cannot honour it is refused,
    // not dropped on the way to the grammar compiler.
    #[test]
    fn a_nested_annotation_is_refused_rather_than_ignored() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": { "a": { "enum": ["x"], "include_score": true } },
                },
            },
            "required": ["inner"],
            "additionalProperties": false,
        });
        assert!(plan_of(&schema).is_err());
    }

    // An empty option has no tokens, so it would score as free certainty; a
    // repeated one would take mass twice.
    #[test]
    fn degenerate_enum_values_are_refused() {
        let mut schema = scored_schema(Value::Bool(true));
        schema["properties"]["label"]["enum"] = serde_json::json!(["yes", ""]);
        assert!(plan_of(&schema).unwrap_err().to_string().contains("empty"));

        schema["properties"]["label"]["enum"] = serde_json::json!(["yes", "yes"]);
        assert!(plan_of(&schema).unwrap_err().to_string().contains("twice"));
    }

    // The skeleton is compact JSON with the values cut out, and a string field's
    // quotes belong to it — which is what makes an option's score the logprob of
    // its body rather than of an opening quote it shares with every sibling.
    #[test]
    fn the_skeleton_quotes_strings_and_leaves_booleans_bare() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "label": { "enum": ["yes"], "include_score": true },
                "urgent": { "type": "boolean" },
            },
            "required": ["label", "urgent"],
            "additionalProperties": false,
        });
        let plan = plan_of(&schema).unwrap().unwrap();
        let text: Vec<&str> = plan.segments.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(text, vec!["{\"label\":\"", "\",\"urgent\":", "}"]);
        assert_eq!(plan.fields[0].options[0].text, "yes");
        // A boolean's options are the bare literals, keyed by their spelling but
        // reported as real JSON booleans.
        let urgent = &plan.fields[1].options;
        assert_eq!(urgent[0].text, "true");
        assert_eq!(urgent[0].value, Value::Bool(true));
        assert_eq!(urgent[0].key, "true");
        assert_eq!(urgent[1].value, Value::Bool(false));

        // Reassembling the skeleton around one option per field gives the
        // compact document the model is teacher-forced through.
        let document = format!(
            "{}{}{}{}{}",
            text[0], plan.fields[0].options[0].text, text[1], urgent[1].text, text[2],
        );
        assert_eq!(document, "{\"label\":\"yes\",\"urgent\":false}");
        assert_eq!(
            serde_json::from_str::<Value>(&document).unwrap(),
            serde_json::json!({ "label": "yes", "urgent": false }),
        );
    }

    // A value that JSON cannot carry literally is refused rather than escaped:
    // the escape sequence, not the value, would be what gets scored, and a value
    // holding a quote of its own would no longer be delimited by one.
    #[test]
    fn an_option_needing_escapes_is_refused() {
        for value in ["say \"hi\"", "back\\slash", "line\nbreak"] {
            let mut schema = scored_schema(Value::Bool(true));
            schema["properties"]["label"]["enum"] = serde_json::json!([value]);
            let error = plan_of(&schema)
                .expect_err("v1 writes its values literally")
                .to_string();
            assert!(error.contains("JSON escaping"), "{error}");
        }
    }

    // The document is teacher-forced piece by piece, but the model has only seen
    // text tokenized whole. A value whose last character merges with the
    // delimiter behind it would be forced through a seam no training text
    // contains, so the plan refuses it instead of scoring across it.
    #[test]
    fn a_value_that_merges_with_its_delimiter_is_refused() {
        // `!` followed by the closing quote is one token in this vocabulary,
        // while `!` and `"` alone are two.
        for value in ["yes!", "done.", "why?", "trailing "] {
            let mut schema = scored_schema(Value::Bool(true));
            schema["properties"]["label"]["enum"] = serde_json::json!([value, "plain"]);
            let error = plan_of(&schema)
                .expect_err("this value does not tokenize canonically against its quote")
                .to_string();
            assert!(error.contains("non-canonically"), "{error}");
            assert!(error.contains(value), "{error}");
            assert!(error.contains("label"), "{error}");
        }
    }

    // The seam check is a guard, not a wall: ordinary word-shaped values — the
    // ones a classification schema is actually made of — pass it, in both the
    // quoted and the bare position.
    #[test]
    fn ordinary_values_tokenize_canonically() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "sentiment": {
                    "enum": ["positive", "negative", "neutral", "mixed"],
                    "include_score": "all",
                },
                "urgent": { "type": "boolean" },
                "route_to": { "enum": ["fulfilment", "product", "support"] },
            },
            "required": ["sentiment", "urgent", "route_to"],
            "additionalProperties": false,
        });
        assert!(plan_of(&schema).unwrap().is_some());
    }

    /// The token sequence a field's option is SCORED over: its body plus the
    /// token that opens the segment after it (see `score_field`).
    fn scored_sequences(plan: &ScoredPlan, field: usize) -> Vec<Vec<u32>> {
        let terminator = plan.segments[field + 1].tokens[0];
        plan.fields[field]
            .options
            .iter()
            .map(|option| {
                let mut sequence = option.tokens.clone();
                sequence.push(terminator);
                sequence
            })
            .collect()
    }

    // An option whose body is a strict prefix of another's could never win under
    // greedy scoring — the longer score is the shorter one plus a negative — so
    // every candidate is scored through the delimiter that ends it, which makes
    // the set prefix-free and the comparison a real one.
    #[test]
    fn a_prefix_pair_scores_as_a_prefix_free_set() {
        let mut schema = scored_schema(Value::String("all".into()));
        schema["properties"]["label"]["enum"] = serde_json::json!(["low", "low_priority"]);
        let plan = plan_of(&schema).unwrap().unwrap();

        // The bodies alone ARE a prefix pair: this is the case the terminator
        // exists for, not a hypothetical one.
        let bodies: Vec<Vec<u32>> = plan.fields[0]
            .options
            .iter()
            .map(|o| o.tokens.clone())
            .collect();
        assert!(bodies[1].starts_with(&bodies[0]), "{bodies:?}");

        let sequences = scored_sequences(&plan, 0);
        for (i, a) in sequences.iter().enumerate() {
            for (j, b) in sequences.iter().enumerate() {
                if i != j {
                    assert!(!b.starts_with(a), "{a:?} still opens {b:?}");
                }
            }
        }
    }

    // The terminator is the token that opens the NEXT segment: a string field's
    // closing quote, a boolean's separator. It is scored and never committed —
    // the segment writes it in its own right.
    #[test]
    fn the_terminator_is_the_next_segments_opening_token() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "label": { "enum": ["yes"], "include_score": true },
                "urgent": { "type": "boolean" },
            },
            "required": ["label", "urgent"],
            "additionalProperties": false,
        });
        let plan = plan_of(&schema).unwrap().unwrap();
        let tokenizer = LagunaTokenizer::embedded().unwrap();
        // The string field closes on the quote that opens `","urgent":`.
        assert!(plan.segments[1].text.starts_with('"'));
        assert_eq!(
            plan.segments[1].tokens[0],
            tokenizer
                .encode_prompt(&plan.segments[1].text, &[])
                .unwrap()[0],
        );
        // The boolean closes on the `}` that is the whole final segment.
        assert_eq!(plan.segments[2].text, "}");
        assert_eq!(plan.segments[2].tokens.len(), 1);
    }

    /// Sampling with everything but the named knobs left at the batch default.
    fn sampling(temperature: f64, top_k: usize, top_p: f64) -> SamplerOptions {
        SamplerOptions {
            temperature,
            top_k,
            top_p,
            presence_penalty: 0.0,
            seed: 7,
        }
    }

    // Scored selection answers to the same knobs ordinary generation does, so a
    // top-k or a nucleus cut narrows the option set rather than being ignored.
    #[test]
    fn selection_honors_top_k_and_top_p() {
        let scores = [-0.1, -1.0, -2.0, -8.0];
        let mut rng = StdRng::seed_from_u64(1);

        // Greedy on both of the sampler's greedy conditions.
        for options in [sampling(0.0, 20, 0.95), sampling(1.5, 1, 0.95)] {
            for _ in 0..8 {
                assert_eq!(select_option(&scores, &options, &mut rng).unwrap(), 0);
            }
        }
        // top_k = 2 can never reach the third option however hot the draw.
        let narrow = sampling(4.0, 2, 1.0);
        for _ in 0..200 {
            assert!(select_option(&scores, &narrow, &mut rng).unwrap() < 2);
        }
        // A tight nucleus keeps only the leader, even at a flattening
        // temperature that would otherwise spread the mass.
        let tight = sampling(4.0, 20, 0.1);
        for _ in 0..50 {
            assert_eq!(select_option(&scores, &tight, &mut rng).unwrap(), 0);
        }
        // Wide open, the tail is reachable — so the cuts above are what
        // excluded it, not the scores.
        let open = sampling(4.0, 20, 1.0);
        let reached = (0..400).any(|_| select_option(&scores, &open, &mut rng).unwrap() >= 2);
        assert!(reached, "a wide draw should reach past the top two");
    }

    // `top_k = 0` is "no top-k cut" here exactly as it is in the vocabulary
    // sampler, and `top_k = 1` is greedy. The two used to mean the same thing
    // (both collapsed to greedy), so this is the pair that pins them apart.
    #[test]
    fn a_zero_top_k_keeps_every_option_and_one_is_greedy() {
        let scores = [-0.1, -1.0, -2.0, -8.0];
        let mut rng = StdRng::seed_from_u64(1);

        let greedy = sampling(4.0, 1, 1.0);
        for _ in 0..50 {
            assert_eq!(select_option(&scores, &greedy, &mut rng).unwrap(), 0);
        }

        let uncut = sampling(4.0, 0, 1.0);
        let reached = (0..400).any(|_| select_option(&scores, &uncut, &mut rng).unwrap() >= 2);
        assert!(reached, "top_k 0 must not cut the option set");
    }

    // The nucleus cut measures mass renormalized over the candidates, keeps the
    // option that crosses the threshold, and returns early at 1.0 so rounding
    // cannot drop the last one.
    #[test]
    fn the_nucleus_cut_keeps_the_crossing_option() {
        let ranked = || vec![(0usize, 0.5), (1, 0.3), (2, 0.2)];
        let mut all = ranked();
        truncate_nucleus(&mut all, 1.0);
        assert_eq!(all.len(), 3);

        let mut half = ranked();
        truncate_nucleus(&mut half, 0.5);
        assert_eq!(half.len(), 1);

        // 0.5 alone does not reach 0.6; the option that crosses it is kept.
        let mut most = ranked();
        truncate_nucleus(&mut most, 0.6);
        assert_eq!(most.len(), 2);
    }

    // A budget past the context is refused however large it is. Release builds
    // wrap on overflow rather than panic, so a `max_tokens` near the top of the
    // range must not come out of the arithmetic looking small and legal.
    #[test]
    fn an_enormous_budget_is_refused_rather_than_wrapping() {
        let tokenizer = LagunaTokenizer::embedded().unwrap();
        let defaults = ItemDefaults::default();
        for max_tokens in [usize::MAX, usize::MAX - 1, 8193] {
            let mut spec = item("a");
            spec.max_tokens = Some(max_tokens);
            let error = prepare_item(
                &tokenizer,
                8192,
                &spec,
                &defaults,
                None,
                "Qwen3.6-35B-A3B",
                Model::Qwen35BA3B,
            )
            .err()
            .expect("a budget past the context has nowhere to decode")
            .to_string();
            assert!(error.contains("exceeds the context"), "{error}");
        }
        // And one that fits still prepares.
        let mut spec = item("a");
        spec.max_tokens = Some(64);
        assert!(
            prepare_item(
                &tokenizer,
                8192,
                &spec,
                &defaults,
                None,
                "Qwen3.6-35B-A3B",
                Model::Qwen35BA3B
            )
            .is_ok()
        );
    }

    // A request-level shared_prefix is spelled once on the wire but lands in
    // every item's prompt: the tokens must be identical to an item whose first
    // message carried the document inline — same prompts, same answers, same
    // scores — and only the first message takes it.
    #[test]
    fn a_shared_prefix_prepends_to_every_items_first_message() {
        let tokenizer = LagunaTokenizer::embedded().unwrap();
        let defaults = ItemDefaults::default();

        let mut declared = item("a");
        declared.messages = vec![
            BatchMessage {
                role: "user".into(),
                content: "Question one?".into(),
                thinking: None,
            },
            BatchMessage {
                role: "user".into(),
                content: "Really?".into(),
                thinking: None,
            },
        ];
        let mut inline = declared.clone();
        inline.messages[0].content = "A long shared story.\n\nQuestion one?".into();

        let with_prefix = prepare_item(
            &tokenizer,
            8192,
            &declared,
            &defaults,
            Some("A long shared story.\n\n"),
            "Qwen3.6-35B-A3B",
            Model::Qwen35BA3B,
        )
        .unwrap();
        let spelled_out = prepare_item(
            &tokenizer,
            8192,
            &inline,
            &defaults,
            None,
            "Qwen3.6-35B-A3B",
            Model::Qwen35BA3B,
        )
        .unwrap();
        assert_eq!(with_prefix.tokens, spelled_out.tokens);
    }

    // An item with no messages has nowhere to put the prefix; that is the
    // item's failure, not the batch's.
    #[test]
    fn a_shared_prefix_without_a_first_message_fails_the_item() {
        let tokenizer = LagunaTokenizer::embedded().unwrap();
        let mut spec = item("a");
        spec.messages.clear();
        let error = prepare_item(
            &tokenizer,
            8192,
            &spec,
            &ItemDefaults::default(),
            Some("story"),
            "Qwen3.6-35B-A3B",
            Model::Qwen35BA3B,
        )
        .err()
        .expect("no first message to prepend to")
        .to_string();
        assert!(error.contains("shared_prefix"), "{error}");
    }

    // The worst case is what the budget is checked against: every segment plus
    // the LONGEST option of every field, since which option wins is not known
    // until it has been scored.
    #[test]
    fn the_plan_budgets_for_the_longest_option() {
        let mut schema = scored_schema(Value::Bool(true));
        schema["properties"]["label"]["enum"] =
            serde_json::json!(["no", "a thoroughly unlikely multi-token label"]);
        let plan = plan_of(&schema).unwrap().unwrap();
        let segments: usize = plan.segments.iter().map(|s| s.tokens.len()).sum();
        let longest = plan.fields[0]
            .options
            .iter()
            .map(|o| o.tokens.len())
            .max()
            .unwrap();
        assert_eq!(plan.worst_case_tokens, segments + longest);
        assert!(longest > 1, "the long option should span several tokens");
    }

    /// Scores as a decode would leave them: one logprob per option.
    fn pick_from(scores: &[f64], escape: f64) -> Pick {
        Pick {
            index: argmax_index(scores),
            probs: renormalize(scores, 1.0),
            escape,
        }
    }

    // The reported probabilities are the options' sequence logprobs renormalized
    // over the options alone: what the model would have spent on anything else
    // is reported separately, as the escape, rather than shrinking them.
    #[test]
    fn probabilities_renormalize_over_the_options() {
        let probs = renormalize(&[-1.0, -2.0, -5.0], 1.0);
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "{total}");
        assert!(probs[0] > probs[1] && probs[1] > probs[2]);
        // Equal scores split the mass evenly however deep the row is.
        let even = renormalize(&[-3.5, -3.5, -3.5, -3.5], 1.0);
        assert!(even.iter().all(|p| (p - 0.25).abs() < 1e-12));
        // Very negative scores renormalize by their differences, not by their
        // magnitude — the max subtraction is what keeps them from underflowing.
        let far = renormalize(&[-900.0, -901.0], 1.0);
        assert!((far[0] / far[1] - std::f64::consts::E).abs() < 1e-9);
    }

    // Temperature reshapes the draw without touching what is reported: the
    // response always carries the renormalized probabilities themselves.
    #[test]
    fn temperature_flattens_the_draw_only() {
        let scores = [-1.0, -2.0];
        let cold = renormalize(&scores, 0.25);
        let warm = renormalize(&scores, 4.0);
        assert!(cold[0] > renormalize(&scores, 1.0)[0]);
        assert!(warm[0] < renormalize(&scores, 1.0)[0]);
        assert!(warm[0] > warm[1], "the ordering never flips");
    }

    // Greedy selection takes the best score, ties going to the option the schema
    // listed first.
    #[test]
    fn greedy_selection_breaks_ties_by_schema_order() {
        assert_eq!(argmax_index(&[-2.0, -0.5, -3.0]), 1);
        assert_eq!(argmax_index(&[-1.0, -1.0]), 0);
    }

    /// Token bytes from strings, for driving [`escape_mass`] directly.
    fn texts(list: &[&str]) -> Vec<Vec<u8>> {
        list.iter().map(|s| s.as_bytes().to_vec()).collect()
    }

    /// The one boolean field of a `{"urgent": bool}` plan — an UNQUOTED field.
    fn boolean_field() -> ScoredField {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "urgent": { "type": "boolean", "include_score": "all" } },
            "required": ["urgent"],
            "additionalProperties": false,
        });
        plan_of(&schema).unwrap().unwrap().fields.remove(0)
    }

    // At an unquoted choice point the model's mass sits mostly on
    // whitespace-led spellings of the allowed values (` true` is one token) and
    // on pure-whitespace layout tokens. The first are the answer and count
    // inside; the second are formatting and count on neither side; the escape
    // is what remains, renormalized over the content mass.
    #[test]
    fn escape_ignores_formatting_and_whitespace_led_spellings() {
        let field = boolean_field();
        let decoded = texts(&["true", " false", "\n  ", " \"", "fal"]);
        let probs = [0.3, 0.4, 0.2, 0.05, 0.05];
        // Inside 0.3 + 0.4 + 0.05, formatting 0.2, outside the quote's 0.05.
        let escape = escape_mass(&probs, &decoded, &field).unwrap();
        assert!((escape - 0.05 / 0.8).abs() < 1e-12, "{escape}");
    }

    // A quoted field reads token text verbatim: past the opening quote,
    // leading whitespace is string CONTENT, so a whitespace-led spelling of an
    // option is a different string and a pure-whitespace token is real mass
    // outside the options — neither is formatting there.
    #[test]
    fn escape_reads_a_quoted_field_verbatim() {
        let plan = one_field(Value::String("all".into())); // enum ["yes", "no"]
        let field = &plan.fields[0];
        assert!(field.quoted);
        let decoded = texts(&["yes", " yes", " ", "y"]);
        let probs = [0.4, 0.3, 0.2, 0.1];
        // Inside `yes` and the prefix `y`; ` yes` and ` ` are other strings.
        let escape = escape_mass(&probs, &decoded, field).unwrap();
        assert!((escape - 0.5).abs() < 1e-12, "{escape}");
    }

    // Prefix matching is one-way: a token that can still BECOME an option
    // counts inside, a token that starts with one and carries on past its edge
    // does not (`yesterday` is not headed for `yes`).
    #[test]
    fn escape_counts_prefixes_inside_and_extensions_outside() {
        let plan = one_field(Value::String("all".into())); // enum ["yes", "no"]
        let field = &plan.fields[0];
        let decoded = texts(&["ye", "yes", "yesterday", "maybe"]);
        let probs = [0.25, 0.25, 0.25, 0.25];
        let escape = escape_mass(&probs, &decoded, field).unwrap();
        assert!((escape - 0.5).abs() < 1e-12, "{escape}");
    }

    // A row that is nothing but formatting holds no content distribution to
    // read a gap off, and the escape says 0 rather than dividing by nothing.
    #[test]
    fn an_all_formatting_row_escapes_nothing() {
        let field = boolean_field();
        let decoded = texts(&[" ", "\n\n", "\t"]);
        let probs = [0.6, 0.3, 0.1];
        assert_eq!(escape_mass(&probs, &decoded, &field).unwrap(), 0.0);
    }

    // Classification is by BYTES: a token holding the leading bytes of a
    // multi-byte option character is that option's canonical opener and counts
    // inside, even though it decodes to no text on its own. And formatting is
    // JSON whitespace only — an NBSP would make the document invalid, so it is
    // real mass outside, not layout.
    #[test]
    fn escape_reads_bytes_not_lossy_text() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "mark": { "enum": ["\u{1F9FF}", "x"], "include_score": "all" }
            },
            "required": ["mark"],
            "additionalProperties": false,
        });
        let field = plan_of(&schema).unwrap().unwrap().fields.remove(0);
        // "🧿" is F0 9F A7 BF; the first two bytes alone open it.
        let decoded = vec![vec![0xF0, 0x9F], b"x".to_vec(), b"no".to_vec()];
        let probs = [0.5, 0.25, 0.25];
        let escape = escape_mass(&probs, &decoded, &field).unwrap();
        assert!((escape - 0.25).abs() < 1e-12, "{escape}");

        let boolean = boolean_field();
        let nbsp = vec![vec![0xC2, 0xA0], b"true".to_vec()];
        let probs = [0.5, 0.5];
        let escape = escape_mass(&probs, &nbsp, &boolean).unwrap();
        assert!(
            (escape - 0.5).abs() < 1e-12,
            "NBSP is outside, not formatting: {escape}"
        );
    }

    /// A one-field plan whose options are `yes`/`no`, annotated as `include`.
    fn one_field(include: Value) -> ScoredPlan {
        plan_of(&scored_schema(include)).unwrap().unwrap()
    }

    // An annotated field reports its value wrapped with the confidence behind
    // it; an unannotated one reports the value alone.
    #[test]
    fn reporting_wraps_only_the_annotated_fields() {
        let pick = pick_from(&[-0.1, -2.0], 0.05);
        let score = pick.probs[0];

        let bare = one_field(Value::Bool(false));
        assert_eq!(report(&bare.fields[0], &pick), serde_json::json!("yes"));

        let wrapped = one_field(Value::Bool(true));
        assert_eq!(
            report(&wrapped.fields[0], &pick),
            serde_json::json!({ "value": "yes", "score": score }),
        );
    }

    // `"all"` adds the whole option table and the escape, keyed by the enum
    // values themselves.
    #[test]
    fn reporting_all_carries_the_option_table() {
        let pick = pick_from(&[-0.1, -2.0], 0.05);
        let value = report(&one_field(Value::String("all".into())).fields[0], &pick);
        assert_eq!(value["value"], serde_json::json!("yes"));
        assert_eq!(value["score"], serde_json::json!(pick.probs[0]));
        assert_eq!(value["escape"], serde_json::json!(0.05));
        assert_eq!(value["scores"]["yes"], serde_json::json!(pick.probs[0]));
        assert_eq!(value["scores"]["no"], serde_json::json!(pick.probs[1]));
        let table: f64 = value["scores"]
            .as_object()
            .unwrap()
            .values()
            .map(|v| v.as_f64().unwrap())
            .sum();
        assert!((table - 1.0).abs() < 1e-12, "{table}");
    }

    // A boolean's table is keyed by the STRINGS "true"/"false" — a JSON object
    // has no other kind of key — while the chosen value stays a real boolean.
    #[test]
    fn a_boolean_field_reports_string_keys_and_a_real_boolean() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "urgent": { "type": "boolean", "include_score": "all" } },
            "required": ["urgent"],
            "additionalProperties": false,
        });
        let plan = plan_of(&schema).unwrap().unwrap();
        let value = report(&plan.fields[0], &pick_from(&[-3.0, -0.05], 0.01));
        assert_eq!(value["value"], Value::Bool(false));
        assert!(value["scores"]["true"].is_number());
        assert!(value["scores"]["false"].is_number());
        assert!(
            value["scores"]["false"].as_f64() > value["scores"]["true"].as_f64(),
            "{value}"
        );
    }

    // A budget that cannot hold the assembled answer refuses the item before any
    // forward runs, and reports the budget that refused it.
    #[test]
    fn a_budget_too_small_for_the_skeleton_refuses_the_item() {
        let plan = one_field(Value::Bool(true));
        let mut item = prepared(None, "");
        item.max_tokens = plan.worst_case_tokens - 1;
        let outcome = budget_refusal(&item, &plan, 0);
        assert_eq!(outcome.finish_reason, FinishReason::Length);
        assert_eq!(outcome.completion_tokens, 0);
        assert!(outcome.json.is_none());
        let error = outcome.error.expect("a refusal always says why");
        assert!(
            error.contains(&format!("{}-token budget", item.max_tokens)),
            "{error}"
        );
        assert!(
            error.contains(&plan.worst_case_tokens.to_string()),
            "{error}"
        );
    }

    // A reasoning block needs a token of its own on top of the assembly, so the
    // refusal says so rather than reporting a budget that would otherwise fit.
    #[test]
    fn a_reasoning_item_needs_room_beyond_the_assembly() {
        let plan = one_field(Value::Bool(true));
        let mut item = prepared(None, "");
        item.max_tokens = plan.worst_case_tokens;
        let error = budget_refusal(&item, &plan, 1)
            .error
            .expect("a refusal always says why");
        assert!(error.contains("reasoning"), "{error}");
    }

    // Role names map onto the renderer's message kinds, and an unknown one is
    // an item error rather than a silently dropped turn.
    #[test]
    fn roles_map_onto_renderer_messages() {
        let message = |role: &str| BatchMessage {
            role: role.into(),
            content: "x".into(),
            thinking: None,
        };
        assert!(matches!(
            chat_message(&message("system"), None).unwrap(),
            Message::System(_)
        ));
        assert!(matches!(
            chat_message(&message("user"), None).unwrap(),
            Message::User(_)
        ));
        assert!(matches!(
            chat_message(&message("assistant"), None).unwrap(),
            Message::Assistant { .. }
        ));
        assert!(matches!(
            chat_message(&message("tool"), None).unwrap(),
            Message::ToolResponse(_)
        ));
        assert!(chat_message(&message("developer"), None).is_err());
    }
}
