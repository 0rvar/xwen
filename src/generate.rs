use std::collections::VecDeque;
use std::ops::Range;
use std::path::Path;
use std::time::Instant;

use anyhow::{Result, anyhow, ensure};
use candle_core::{D, DType, Device, Tensor};

use crate::config::XwenConfig;
use crate::constrain::GrammarState;
use crate::dflash::{DflashDrafter, DrafterImage};
use crate::gguf;
use crate::kv_cache::{CacheSnapshot, HostFullKv, HostSnapshot, KvCheckpoint};
use crate::model::XwenModel;
use crate::ops::ExpertRunner;
use crate::sampler::{SampleControl, Sampler, SamplerOptions};
use crate::tokenizer::LagunaTokenizer;

/// Prompt tokens are fed to the model in chunks of this size. Keeps prefill
/// attention/MoE tensors bounded while still amortizing per-forward overhead.
const PREFILL_CHUNK: usize = 512;

/// Chunked prefill (512-token chunks) + single-token decode loop with a
/// streaming callback. One GPU->CPU sync per decoded token (the logits
/// readback that feeds the CPU sampler).
///
/// When a DFlash drafter is attached (`attach_drafter`), `generate` runs the
/// speculative decode loop instead; with the drafter absent it is byte-identical
/// to the pre-DFlash single-token loop.
pub struct Generator {
    model: XwenModel,
    tokenizer: LagunaTokenizer,
    sampler: Sampler,
    /// The speculative drafter, or `None` for the plain single-token loop.
    drafter: Option<DflashDrafter>,
    spec_params: SpecParams,
    /// Forced-reasoning floor: while fewer than this many tokens have been
    /// decoded in a call, `</think>` and the EOG ids are banned from sampling,
    /// holding the model inside the `<think>` block the chat template opened.
    /// 0 (the default) leaves the model free to close the block immediately —
    /// which the hybrid reasoner does on conversational prompts. Only
    /// meaningful when the prompt ends with `<assistant><think>`.
    min_think: usize,
    /// Reasoning ceiling (`--max-think`): the counterpart to `min_think`, idle
    /// while its budget is 0.
    think: ThinkBudget,
    /// The vocabulary scan every nonzero `think` budget needs, kept across
    /// `set_max_think` calls: a server sets the ceiling per request, and
    /// rescanning the whole vocabulary each time would cost more than the reply.
    /// Populated on the first nonzero ceiling and never invalidated — the
    /// tokenizer is fixed for the life of the generator.
    think_statics: Option<ThinkStatics>,
    /// Token ids banned from every draw for the life of the generator
    /// (`--ban-string`), sorted. Empty by default.
    blacklist: Vec<u32>,
    /// Per-request grammar constraint (`response_format` on the server): masks
    /// every answer-section draw to schema-legal tokens. `None` — the default
    /// and the state between constrained requests — leaves decoding untouched.
    /// Set per job by the server, like the sampler and thinking budget.
    grammar: Option<GrammarState>,
    /// Logits for the last position `prefill_tokens` fed, consumed by the next
    /// `decode_loop`. The token-level API splits prefill from decode across two
    /// calls (the server snapshots the cache between them); `generate` keeps its
    /// logits on the stack and never touches this.
    last_logits: Option<Tensor>,
}

/// One decoded token, tagged with the section of the reply it belongs to and
/// carrying the text it finalized. `text` is empty whenever the streaming
/// decoder withheld the token (mid-UTF-8-sequence, or a partial `<think>`-tag
/// prefix); the id is always real, so a caller tracking what the KV cache holds
/// can append every id it sees.
///
/// `</think>` arrives as the last `ThinkingTok` with the marker stripped from
/// its text — the reasoning/answer split is the variant, never a literal in the
/// stream.
#[derive(Debug, Clone)]
pub enum GenEvent {
    ThinkingTok { id: u32, text: String },
    TextTok { id: u32, text: String },
}

impl GenEvent {
    /// The sampled token id, whichever section it belongs to.
    pub fn id(&self) -> u32 {
        match self {
            GenEvent::ThinkingTok { id, .. } | GenEvent::TextTok { id, .. } => *id,
        }
    }

    /// The text this token finalized, empty when the decoder withheld it.
    pub fn text(&self) -> &str {
        match self {
            GenEvent::ThinkingTok { text, .. } | GenEvent::TextTok { text, .. } => text,
        }
    }
}

/// What a `decode_loop` run did. `tokens_out` counts emitted tokens (an EOG stop
/// token is not emitted, so it is not counted); exactly one of `hit_eog`,
/// `cancelled`, and `tokens_out == max_new` explains why the loop returned,
/// unless the caller passed `max_new == 0`.
#[derive(Debug, Default, Clone, Copy)]
pub struct DecodeOutcome {
    pub tokens_out: usize,
    /// Tokens emitted inside the `<think>` block, counting the closing
    /// `</think>`. Zero when the loop did not start in thinking.
    pub thinking_tokens: usize,
    pub hit_eog: bool,
    /// The loop returned because `should_stop` said so.
    pub cancelled: bool,
    pub decode_secs: f64,
    /// Speculative-decode accounting, present only for a `decode_loop_spec` run.
    pub spec: Option<SpecStats>,
}

/// Speculative-decode knobs (mirrors the fork's `common_params_speculative`).
#[derive(Debug, Clone)]
pub struct SpecParams {
    /// Max draft tokens per round (`--draft-max`); also clamped to `block_size-1`.
    pub draft_max: usize,
    /// Discard a round's whole draft if fewer than this many were collected
    /// (`--draft-min`).
    pub draft_min: usize,
    /// Stop collecting drafts at the first drafter token whose full-vocab softmax
    /// probability falls below this (`--draft-p-min`).
    pub draft_p_min: f32,
    /// Auto-pause margin (`--draft-pause-margin`). Speculation pauses when its
    /// measured wall-clock cost per committed token exceeds a plain decode
    /// step's cost times this factor. `0.0` disables auto-pause entirely
    /// (always draft — the pre-auto-pause behavior).
    pub pause_margin: f32,
}

impl Default for SpecParams {
    fn default() -> Self {
        // Adaptive draft length beats any fixed draft_max: a full fixed block
        // spends the whole round verifying a tail that gets rejected. p_min 0.3
        // measured best of {0.2, 0.3, 0.5, 0.7, 0.9} on the Qwen sidecars
        // (2026-07-29, 27B, interleaved arms, greedy): it is the only setting
        // that came out ahead of plain decode on BOTH a code prompt and a chat
        // prompt in every run, where 0.5 lost on chat twice. Lower p_min drafts
        // longer at lower acceptance, and that wins here because the verify
        // forward's cost is close to linear in its span — the DeltaNet layers
        // fall back to the per-token reference scan under an armed rollback
        // trail, so a long span amortizes the round's fixed cost without the
        // usual batching penalty. Revisit when the fused verify lands (TODO P9):
        // it changes the cost curve this was fitted to.
        Self {
            draft_max: 15,
            draft_min: 0,
            draft_p_min: 0.3,
            pause_margin: 1.0,
        }
    }
}

/// Per-run speculative-decode accounting. `GenStats::decode_tokens` counts only
/// EMITTED tokens (accepted drafts plus each round's final committed token); the
/// draft positions a round proposes and the target rejects cost wall time but
/// are never emitted, so `decode_tps` is the true end-to-end throughput. These
/// fields expose where that wall time goes.
#[derive(Debug, Default, Clone, Copy)]
pub struct SpecStats {
    /// Verify rounds executed (each commits >= 1 token).
    pub rounds: usize,
    /// Total draft tokens proposed across all rounds.
    pub drafted: usize,
    /// Total draft tokens accepted (matched the target's sample).
    pub accepted: usize,
    /// Rounds that ran a plain single-token step because auto-pause was active:
    /// paused-plain rounds, plus probe rounds whose draft came up empty (they
    /// too ran plain). Real drafting probes are excluded — those still
    /// speculate. Zero when auto-pause never engaged (good economics, or
    /// `pause_margin == 0`).
    pub paused_rounds: usize,
    /// Total target-forward positions across all rounds: the anchor row (the
    /// last committed token, re-forwarded) plus each round's drafts. A plain
    /// round contributes 1. Equals `rounds + drafted` when every round drafts.
    pub verify_positions: usize,
    /// Total wall ms spent in the draft phase (drafter forward + lm_head +
    /// argmax collection). A round that drafts nothing but still ran the drafter
    /// (all tokens below `draft_p_min`) charges its wasted draft time here.
    pub draft_ms: f64,
    /// Total wall ms spent in the commit phase (the target forward — batched
    /// verify or the plain single-token step — plus acceptance sampling).
    pub verify_ms: f64,
}

impl SpecStats {
    /// Fraction of proposed drafts the target accepted (0.0 if none proposed).
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted > 0 {
            self.accepted as f64 / self.drafted as f64
        } else {
            0.0
        }
    }

    /// Draft positions the target rejected (proposed but not accepted). Pure
    /// wall-time cost — these are never emitted.
    pub fn rejected(&self) -> usize {
        self.drafted - self.accepted
    }
}

#[derive(Debug, Default, Clone)]
pub struct GenStats {
    pub prefill_tokens: usize,
    pub prefill_secs: f64,
    pub decode_tokens: usize,
    pub decode_secs: f64,
    /// True when generation stopped because the cancel poll returned true
    /// (rather than hitting an EOG token or the max_tokens cap).
    pub cancelled: bool,
    /// Speculative-decode stats, present only when a drafter was attached.
    pub spec: Option<SpecStats>,
    /// Thinking-budget stats, present only when `--max-think` is set.
    pub think: Option<ThinkStats>,
}

/// What the thinking budget did over one generate call.
#[derive(Debug, Default, Clone, Copy)]
pub struct ThinkStats {
    /// Tokens emitted inside the `<think>` block, counting the closing
    /// `</think>`. When the block never closed this is every emitted token.
    pub tokens: usize,
    /// Whether `</think>` was committed at all.
    pub closed: bool,
    /// Whether the wrap-up sentence was injected.
    pub wrapup_fired: bool,
    /// Whether the hard force (transition sentence + `</think>`) fired, i.e. the
    /// model did not close the block on its own within the budget.
    pub forced: bool,
}

impl GenStats {
    pub fn prefill_tps(&self) -> f64 {
        if self.prefill_secs > 0.0 {
            self.prefill_tokens as f64 / self.prefill_secs
        } else {
            0.0
        }
    }

    pub fn decode_tps(&self) -> f64 {
        if self.decode_secs > 0.0 {
            self.decode_tokens as f64 / self.decode_secs
        } else {
            0.0
        }
    }
}

/// What kind of round the [`PauseController`] wants next. `Spec`/`Probe` draft
/// and verify; `ForcedPlain`/`PausedPlain` run a plain single-token step. The
/// distinction between the two plain kinds is only for accounting: a forced
/// plain is an Active round sacrificed to keep the plain-cost EMA fresh, while
/// a paused plain is a round where speculation is switched off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoundKind {
    /// Active speculative round.
    Spec,
    /// Active round forced to plain decode to refresh `ema_plain_ms`.
    ForcedPlain,
    /// Paused round run plain because speculation is not paying off.
    PausedPlain,
    /// Paused round run speculative to re-measure whether it pays off now.
    Probe,
}

/// Wall-clock auto-pause for speculative decoding: a pure, GPU-free state
/// machine that decides, per round, whether to speculate or fall back to plain
/// decode so `--draft` never loses to plain decode on low-acceptance text.
///
/// It compares the speculative cost per committed token against the cost of one
/// plain decode step (`ema_plain_ms`), all EMAs at `α = 0.25`. The plain
/// comparator is the empty-draft fallback path, which carries the same tap
/// encode/inject overhead a paused round pays, so the two are like-for-like.
///
/// The speculative cost is a RATE, not a mean of per-round ratios: two EMAs
/// (`ema_spec_round_ms`, `ema_spec_committed`) are tracked separately and the
/// decision uses their quotient. `EMA(ms) / EMA(tok)` estimates the
/// exponentially-windowed aggregate `Σms / Σtok`, which is the quantity that
/// actually equals throughput cost. A single EMA of `round_ms / committed`
/// would be a mean-of-ratios that over-weights low-commit rounds: on a measured
/// code-gen greedy run rounds are bimodal (~25 ms/tok for 11-commit rounds vs
/// ~350 ms/tok for 1-commit mismatch rounds), and the mean-of-ratios crossed
/// plain's ~58 ms even though the aggregate spec cost was ~42 ms/tok — pausing
/// 76 of 99 rounds on a workload where speculation wins by 1.4x. The quotient
/// form does not: a 1-commit round contributes 1 to the token EMA, so its cost
/// is amortized the same way throughput amortizes it.
///
/// Acceptance-count thresholds were rejected deliberately: the break-even
/// committed-tokens-per-round varies with draft span (~2.5 at span 2 to ~3.8 at
/// span 16), so a fixed count misfires. Round economics must be judged on the
/// clock, and the EMA absorbs single-round noise (one probe is not trusted on
/// its own — its cost is folded in and the decision made on the smoothed value).
///
/// Recovery from a stale pause is deliberately EAGER, because the cost
/// asymmetry is inverted: a wrongly-held pause loses 30%+ throughput for the
/// rest of the run, while a wrongly-lifted pause loses only a few rounds before
/// the (still-warm) pause logic re-pauses. A pause can go stale when the samples
/// that triggered it came from a transient world — GPU contention from a second
/// process, a hard text stretch — that has since passed. Three mechanisms fight
/// that: (1) a probe EVENT is a PAIR of consecutive speculative rounds, not one,
/// so the resume decision rests on up to two live samples (an empty-draft probe
/// round contributes none; a pair with one real sample decides on it; a pair
/// with zero decides nothing and retries one backoff later, since an immediate
/// retry would run the drafter on every paused round);
/// (2) probe samples fold at `α = 0.5` (vs 0.25) because while paused `ema_spec`
/// is stale by construction and fresh probe evidence must dominate memory;
/// (3) resume triggers on the pair's OWN aggregate rate (`Σround_ms / Σcommitted`
/// over its real rounds) crossing back under the threshold — regardless of the
/// smoothed EMA still being poisoned — and on such a resume the spec EMAs are
/// SNAPPED to the pair's values, dropping the poisoned history so they rebuild
/// from live active rounds. A pair that fails still doubles the backoff (a failed
/// pair is decent evidence spec still loses).
///
/// Warm-up itself is accelerated: since only plain rounds refresh `ema_plain`,
/// an all-drafting run would otherwise not collect its second plain sample until
/// the `FORCE_PLAIN_EVERY` (32) cadence fires twice — so a short net-negative
/// generation would never reach the `warm()` gate and never pause. Until the
/// plain warm-up is met the forced-plain cadence tightens to
/// `WARMUP_FORCE_PLAIN_EVERY` (every 4th Active round); a forced plain costs the
/// same as plain decode, so early sampling is free.
struct PauseController {
    /// Pause when the speculative cost per token exceeds `ema_plain_ms * margin`.
    /// `0.0` disables auto-pause: every round speculates.
    margin: f64,
    /// EMA of speculative round wall-ms. Paired with `ema_spec_committed`; the
    /// per-token cost is their quotient (see the type doc for why not a single
    /// per-round-ratio EMA).
    ema_spec_round_ms: Option<f64>,
    /// EMA of committed tokens per speculative round.
    ema_spec_committed: Option<f64>,
    /// EMA of a plain single-token round's wall-ms.
    ema_plain_ms: Option<f64>,
    /// Speculative samples folded in (warm-up gate).
    spec_samples: usize,
    /// Plain samples folded in (warm-up gate).
    plain_samples: usize,
    state: PauseState,
    /// Active-state rounds since the last plain sample; at the current forced-
    /// plain cadence (accelerated to `WARMUP_FORCE_PLAIN_EVERY` until the plain
    /// warm-up is met, then `FORCE_PLAIN_EVERY`) the next round is forced plain
    /// so `ema_plain_ms` never goes stale on a run where speculation always wins.
    rounds_since_plain: usize,
    /// Paused-state committed tokens since the last probe event; at `backoff` the
    /// next round starts a probe pair.
    tokens_since_probe: usize,
    /// The in-flight probe pair's accumulated real samples, or `None` between
    /// events. Drives the eager-resume decision once the pair completes.
    probe_event: Option<ProbeEvent>,
}

/// A probe pair in flight: up to [`PauseController::PROBE_PAIR`] consecutive
/// probe rounds, accumulating only the REAL (non-empty-draft) rounds' costs. An
/// empty-draft probe round advances `rounds` but adds no sample, so a pair that
/// never drafts decides nothing.
#[derive(Debug, Clone, Copy, Default)]
struct ProbeEvent {
    /// Probe rounds run in this event so far (real or empty).
    rounds: usize,
    /// Σ round_ms over the event's real rounds.
    real_ms: f64,
    /// Σ committed over the event's real rounds.
    real_committed: usize,
    /// Count of real (non-empty-draft) rounds.
    real: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PauseState {
    Active,
    /// Probe every `backoff` committed tokens; the backoff doubles on each
    /// failed probe up to [`PauseController::BACKOFF_CAP`].
    Paused {
        backoff: usize,
    },
}

impl PauseController {
    const ALPHA: f64 = 0.25;
    /// Fold weight for probe-pair samples: heavier than `ALPHA` so fresh probe
    /// evidence dominates the stale, paused `ema_spec` (see the type doc).
    const PROBE_ALPHA: f64 = 0.5;
    /// Minimum speculative + plain samples before the first pause decision.
    const WARMUP_SPEC: usize = 8;
    const WARMUP_PLAIN: usize = 2;
    /// Active rounds between forced plain rounds (keeps `ema_plain_ms` fresh).
    const FORCE_PLAIN_EVERY: usize = 32;
    /// Tighter forced-plain cadence until the plain warm-up is met, so a short
    /// all-drafting run reaches `warm()` fast enough to pause.
    const WARMUP_FORCE_PLAIN_EVERY: usize = 4;
    /// Probe rounds per probe event (a probe pair).
    const PROBE_PAIR: usize = 2;
    /// Committed tokens between probe events while paused; doubles per failed
    /// probe pair.
    const BACKOFF_START: usize = 32;
    const BACKOFF_CAP: usize = 256;

    fn new(margin: f64) -> Self {
        Self {
            margin,
            ema_spec_round_ms: None,
            ema_spec_committed: None,
            ema_plain_ms: None,
            spec_samples: 0,
            plain_samples: 0,
            state: PauseState::Active,
            rounds_since_plain: 0,
            tokens_since_probe: 0,
            probe_event: None,
        }
    }

    /// Forced-plain cadence for Active rounds: tighter until the plain warm-up
    /// is met so a short run collects plain samples in time to pause.
    fn force_plain_every(&self) -> usize {
        if self.plain_samples < Self::WARMUP_PLAIN {
            Self::WARMUP_FORCE_PLAIN_EVERY
        } else {
            Self::FORCE_PLAIN_EVERY
        }
    }

    /// The kind of round to run next. Pure read of the current state; the round
    /// is executed by the caller, which then feeds the outcome back via
    /// [`record`](Self::record).
    fn next_kind(&self) -> RoundKind {
        if self.margin == 0.0 {
            return RoundKind::Spec;
        }
        match self.state {
            PauseState::Active => {
                if self.rounds_since_plain >= self.force_plain_every() {
                    RoundKind::ForcedPlain
                } else {
                    RoundKind::Spec
                }
            }
            PauseState::Paused { backoff } => {
                // Mid-pair: the pair's remaining probe rounds always follow the
                // first. Otherwise probe once the backoff's worth of tokens have
                // elapsed.
                if self.probe_event.is_some() || self.tokens_since_probe >= backoff {
                    RoundKind::Probe
                } else {
                    RoundKind::PausedPlain
                }
            }
        }
    }

    /// Fold a completed round's cost into the EMAs and update the pause state.
    /// `round_ms` is the whole round's wall time; `draft_ms` is the drafter
    /// portion of it. `ran_spec` is whether the round actually speculated. An
    /// Active `Spec` round with an empty draft falls back to a plain step and
    /// feeds BOTH ledgers: the target-forward portion (`round_ms - draft_ms`)
    /// is a genuine plain-step sample, and the whole round is a speculative
    /// sample committing one token — the round chose to run the drafter, and
    /// that cost is speculative economics whether or not anything cleared
    /// `draft_p_min`. (Folding it only into the plain side, as this once did,
    /// left a persistently draftless stretch with zero spec samples: `warm()`
    /// was never met, the pause never tripped, and the drafter ran forever on
    /// exactly the text it was most useless for.) An empty-draft `Probe` folds
    /// only the plain portion: while paused, the spec ledger moves on real
    /// evidence alone. `committed` is the number of tokens the round committed
    /// (always `>= 1`).
    fn record(
        &mut self,
        kind: RoundKind,
        round_ms: f64,
        draft_ms: f64,
        committed: usize,
        ran_spec: bool,
    ) {
        if ran_spec {
            debug_assert!(committed >= 1, "a speculative round commits >= 1 token");
            // Track the ms and token EMAs separately; the per-token cost is
            // their quotient (a rate, not a mean of ratios — see the type doc).
            // Probe samples fold heavier so a stale pause's poisoned `ema_spec`
            // yields to fresh evidence.
            let alpha = if kind == RoundKind::Probe {
                Self::PROBE_ALPHA
            } else {
                Self::ALPHA
            };
            self.ema_spec_round_ms = Some(Self::ema(self.ema_spec_round_ms, round_ms, alpha));
            self.ema_spec_committed =
                Some(Self::ema(self.ema_spec_committed, committed as f64, alpha));
            self.spec_samples += 1;
        } else {
            // An empty-draft round still ran the drafter, but that cost is not
            // part of plain decode — fold only the target-forward portion
            // (`round_ms - draft_ms`) so the plain comparator is not inflated by
            // wasted drafting (which would delay pausing exactly on the
            // low-acceptance text auto-pause exists to catch). A genuine plain
            // round has `draft_ms == 0`, so this is a no-op there. Plain samples
            // always fold at the base rate; only `ema_spec` goes stale while
            // paused, so only probe SPEC samples get the heavier weight.
            self.ema_plain_ms = Some(Self::ema(
                self.ema_plain_ms,
                (round_ms - draft_ms).max(0.0),
                Self::ALPHA,
            ));
            self.plain_samples += 1;
            if kind == RoundKind::Spec {
                // The drafter ran and proposed nothing. Charge the whole round
                // to the speculative ledger as a one-token round (see the doc
                // comment): the wasted draft cost must be visible to the pause
                // decision, and the sample count must accrue or an all-draftless
                // run never warms. Empty-draft PROBES stay out of the spec
                // ledger — while paused, only real evidence decides (see the
                // probe-pair logic below).
                self.ema_spec_round_ms =
                    Some(Self::ema(self.ema_spec_round_ms, round_ms, Self::ALPHA));
                self.ema_spec_committed = Some(Self::ema(
                    self.ema_spec_committed,
                    committed.max(1) as f64,
                    Self::ALPHA,
                ));
                self.spec_samples += 1;
            }
        }

        match self.state {
            PauseState::Active => {
                // A plain round (forced or an empty-draft fallback) just
                // refreshed the plain EMA; otherwise count down to the next
                // forced plain.
                if ran_spec {
                    self.rounds_since_plain += 1;
                } else {
                    self.rounds_since_plain = 0;
                }
                if self.should_pause() {
                    self.state = PauseState::Paused {
                        backoff: Self::BACKOFF_START,
                    };
                    self.tokens_since_probe = 0;
                }
            }
            PauseState::Paused { backoff } => {
                self.tokens_since_probe += committed;
                if kind == RoundKind::Probe {
                    // Accumulate this probe round into the in-flight pair, starting
                    // one if needed. An empty-draft probe round (ran_spec == false)
                    // advances the round count but contributes NO spec sample and,
                    // on its own, decides nothing — it is accounting-wise a paused
                    // plain round that retries. Only real rounds move the decision.
                    let mut ev = self.probe_event.take().unwrap_or_default();
                    ev.rounds += 1;
                    if ran_spec {
                        ev.real_ms += round_ms;
                        ev.real_committed += committed;
                        ev.real += 1;
                    }
                    if ev.rounds < Self::PROBE_PAIR {
                        // Pair still in flight; wait for the remaining round.
                        self.probe_event = Some(ev);
                    } else if ev.real > 0 {
                        // Pair complete with live evidence: decide on the pair's
                        // OWN aggregate rate over its real rounds, not the smoothed
                        // EMA, which may still be poisoned.
                        self.tokens_since_probe = 0;
                        let pair_rate = ev.real_ms / ev.real_committed as f64;
                        let resume = self
                            .ema_plain_ms
                            .is_some_and(|plain| pair_rate <= plain * self.margin);
                        if resume {
                            // Drop the poisoned history: reseed the spec EMAs to
                            // the pair's per-real-round means (their quotient is
                            // the pair rate) so they rebuild from live rounds.
                            self.ema_spec_round_ms = Some(ev.real_ms / ev.real as f64);
                            self.ema_spec_committed =
                                Some(ev.real_committed as f64 / ev.real as f64);
                            self.state = PauseState::Active;
                            self.rounds_since_plain = 0;
                        } else {
                            self.state = PauseState::Paused {
                                backoff: (backoff * 2).min(Self::BACKOFF_CAP),
                            };
                        }
                    } else {
                        // Zero real samples — the drafter came up empty in both
                        // rounds — so there is nothing to decide on: the backoff
                        // stays undoubled and the spec EMAs untouched. The clock
                        // reset, though, is load-bearing: `tokens_since_probe` is
                        // already >= backoff when a pair starts, so without it
                        // every subsequent round would probe, running the drafter
                        // on every "paused" round — full draft cost on exactly
                        // the low-acceptance text auto-pause exists for. With it,
                        // the retry waits one backoff; a probe that does draft
                        // still resolves through the branch above.
                        self.tokens_since_probe = 0;
                    }
                }
            }
        }
    }

    fn ema(prev: Option<f64>, sample: f64, alpha: f64) -> f64 {
        match prev {
            Some(p) => alpha * sample + (1.0 - alpha) * p,
            None => sample,
        }
    }

    fn warm(&self) -> bool {
        self.spec_samples >= Self::WARMUP_SPEC && self.plain_samples >= Self::WARMUP_PLAIN
    }

    /// Speculative wall-ms per committed token: `EMA(round_ms) / EMA(committed)`.
    /// `None` until at least one speculative round has been recorded.
    fn spec_cost(&self) -> Option<f64> {
        match (self.ema_spec_round_ms, self.ema_spec_committed) {
            (Some(ms), Some(tok)) if tok > 0.0 => Some(ms / tok),
            _ => None,
        }
    }

    fn should_pause(&self) -> bool {
        match (self.spec_cost(), self.ema_plain_ms) {
            (Some(spec), Some(plain)) if self.margin > 0.0 && self.warm() => {
                spec > plain * self.margin
            }
            _ => false,
        }
    }
}

/// The sampler adjustments the thinking budget wants for one draw, borrowing the
/// controller's reusable bias buffer.
struct ThinkControl<'a> {
    bias: &'a [(u32, f32)],
    pull: Option<(u32, f32)>,
    force: Option<u32>,
}

impl ThinkControl<'_> {
    /// The inert control: what an unarmed (or budget-less) controller returns.
    fn inert() -> Self {
        Self {
            bias: &[],
            pull: None,
            force: None,
        }
    }
}

/// Where in the schedule a given thinking-token count falls. Derived from the
/// count and the controller's one-shot flags; carried as a type only so the
/// schedule can be asserted directly in tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkPhase {
    /// Below the first threshold: the controller has no effect at all.
    Free,
    /// Wait tokens are being biased down.
    Damping,
    /// Wait tokens biased down and `</think>` pulled up.
    Pulling,
    /// Pulling, plus the wrap-up injection is eligible at the next boundary.
    WrapUp,
    /// Past the budget, holding both levers at full while waiting for a
    /// sentence boundary to force at.
    Grace,
    /// The transition sentence and `</think>` are queued as forced tokens.
    Forcing,
    /// `</think>` was committed; the controller is done for this call.
    Done,
}

/// Thinking-budget controller (`--max-think N`): steers a reasoning model out of
/// its `<think>` block within roughly `N` decode tokens without cutting it off
/// mid-sentence.
///
/// The schedule escalates in stages, all expressed as fractions of the budget so
/// they scale with `N`:
///
/// 1. Below `BIAS_LO` the controller is inert — a reply that finishes thinking
///    on its own never sees it.
/// 2. From `BIAS_LO`, a growing negative bias on the wait-token set (the
///    "wait" / "but" / "alternatively" continuation cues). Suppressing those is
///    what ends a self-doubt loop, as opposed to merely interrupting it.
/// 3. From `PULL_LO`, a growing pull that lifts `</think>` toward the leading
///    logit so the model exits at its own next good boundary. The pull is
///    self-limiting (see `SampleControl::pull`) and so cannot, on its own,
///    override a decisively different prediction.
/// 4. Inside `[WRAPUP_LO, WRAPUP_HI]`, one injected wrap-up sentence at the first
///    sentence boundary — while there is still budget left to conclude with.
/// 5. Past the budget, a grace period looking for a sentence boundary; at that
///    boundary, or when the grace expires, a transition sentence plus `</think>`
///    are injected as forced tokens.
///
/// Injected tokens are drawn through the sampler with every other logit at -inf,
/// so they land in the KV cache as real context and consume the RNG exactly as a
/// sampled token would.
///
/// The thresholds and magnitudes are research-derived starting points; they have
/// not been swept against this checkpoint (see TODO.md).
struct ThinkBudget {
    /// Token budget `N`; 0 disables the controller entirely.
    budget: usize,
    /// False once `</think>` is committed (permanently, for the rest of the
    /// call) or whenever the budget is 0.
    armed: bool,
    /// Wait-token ids, sorted. Empty when the budget is 0.
    wait_ids: Vec<u32>,
    /// Encoded `WRAPUP_TEXT`.
    wrapup: Vec<u32>,
    /// Encoded `TRANSITION_TEXT` with `</think>` appended.
    transition: Vec<u32>,
    /// Encoded paragraph break, prefixed to an injection that would otherwise
    /// start in the middle of a line.
    paragraph: Vec<u32>,
    /// Forced tokens still to be drawn, in order.
    queue: VecDeque<u32>,
    /// The schedule's own clock: tokens the MODEL chose while armed. Tokens the
    /// controller forced do not advance it, so an injection cannot push the
    /// schedule into its own next stage — a wrap-up queued at 0.90N would
    /// otherwise land past the budget by the time it drained and drag the hard
    /// force in behind it, leaving the model no tokens to conclude with.
    clock: usize,
    /// The draw just handed out was a forced one, so the next commit pops the
    /// queue. Keeps the pop paired with an actual draw rather than with a
    /// caller's bookkeeping.
    forced_pending: bool,
    /// An injection is in flight or just finished, and no model-chosen token has
    /// been committed since. While this holds, `at_boundary` is not trusted:
    /// both injected strings end at a boundary of their own, and acting on it
    /// would chain the next injection straight onto the last.
    awaiting_natural: bool,
    wrapup_fired: bool,
    hard_forced: bool,
    /// The most recently emitted text chunk ended at a sentence or paragraph
    /// boundary. False while a token is mid-UTF-8-codepoint (the streaming
    /// decoder withheld it), which is also exactly when injecting would corrupt
    /// the output.
    at_boundary: bool,
    /// The most recently emitted chunk ended a line, so an injection can start
    /// without a paragraph break of its own.
    at_newline: bool,
    /// Emitted index of the committed `</think>`, once it lands.
    closed_at: Option<usize>,
    /// Reusable `(id, delta)` buffer for the wait-token bias.
    bias_buf: Vec<(u32, f32)>,
}

impl ThinkBudget {
    /// Fractions of the budget bounding the wait-token down-bias ramp.
    const BIAS_LO: f64 = 0.70;
    const BIAS_HI: f64 = 0.95;
    /// Logit delta applied to every wait token once the down-bias saturates.
    const BIAS_MAX: f32 = -4.0;
    /// Fraction at which the `</think>` pull starts. It saturates at the budget
    /// and holds through the grace period.
    const PULL_LO: f64 = 0.80;
    /// Peak pull strength: 0.9 leaves `</think>` just under the leading logit,
    /// so the model still has a say right up to the hard force.
    const PULL_MAX: f32 = 0.9;
    /// Wrap-up injection window, as fractions of the budget.
    const WRAPUP_LO: f64 = 0.90;
    const WRAPUP_HI: f64 = 0.95;
    /// Grace period past the budget: `clamp(N / 10, 16, 256)` tokens spent
    /// waiting for a sentence boundary before forcing.
    const GRACE_DIV: usize = 10;
    const GRACE_MIN: usize = 16;
    const GRACE_MAX: usize = 256;
    /// Tokens kept spare past `budget + grace` for the reply itself, on top of
    /// the injected sentences (which the schedule clock does not count, so they
    /// are added separately — see `reserve`). Firing the forced exit with no
    /// budget left to answer in is the failure mode this controller exists to
    /// avoid.
    const CONCLUSION_RESERVE: usize = 64;
    /// What `reserve()` comes to for the compiled-in injection strings under the
    /// Laguna vocabulary. `feasible_think_budget` answers before any tokenizer
    /// is in reach — an API layer clamps a request long before it takes the
    /// model — so the value is pinned here rather than encoded on demand.
    /// `pinned_reserve_matches_the_encoded_injections` fails the moment an
    /// injection string, `CONCLUSION_RESERVE` or the vocabulary moves it.
    const PINNED_RESERVE: usize = 109;
    /// Smallest budget the schedule can express. Below this the wrap-up window
    /// `[0.90N, 0.95N]` is narrower than one token and can contain no integer
    /// index at all (it needs `0.05N >= 1`), so a tiny budget would skip every
    /// graduated stage and jump straight to the hard force — the mid-sentence
    /// cutoff this design exists to avoid.
    const MIN_BUDGET: usize = 32;

    /// Injected once inside the wrap-up window: nudges the model to start
    /// concluding while it is still free to write its own ending.
    const WRAPUP_TEXT: &'static str =
        "I'm running low on time, so let me settle on the answer with what I have.\n\n";
    /// Prefixed to an injection that starts mid-line. Injections fire at
    /// sentence boundaries, which are often a bare `.` — without this the
    /// injected sentence runs straight onto the model's own ("…done.Considering
    /// the limited time…").
    const PARAGRAPH_TEXT: &'static str = "\n\n";
    /// The hard stop: Qwen3's published budget-forcing transition, injected
    /// verbatim ahead of `</think>` so the reply reads as a deliberate handoff
    /// rather than a truncation.
    const TRANSITION_TEXT: &'static str = "Considering the limited time by the user, I have to \
         give the solution based on the thinking directly now.\n";

    /// Continuation cues from the NoWait ablation: these are the words a model
    /// uses to restart reasoning it had almost finished, and suppressing them is
    /// what shortens the block. Matched EXACTLY against a token's decoded text
    /// (leading whitespace/punctuation stripped, lowercased) — a containment
    /// test would drag "any" into "company".
    const NO_WAIT: [&'static str; 17] = [
        "wait",
        "alternatively",
        "hmm",
        "but",
        "however",
        "alternative",
        "another",
        "check",
        "double-check",
        "oh",
        "maybe",
        "verify",
        "other",
        "again",
        "now",
        "ah",
        "any",
    ];

    /// The disabled controller: no vocab scan, no effect on any draw.
    fn off() -> Self {
        Self {
            budget: 0,
            armed: false,
            wait_ids: Vec::new(),
            wrapup: Vec::new(),
            transition: Vec::new(),
            paragraph: Vec::new(),
            queue: VecDeque::new(),
            clock: 0,
            forced_pending: false,
            awaiting_natural: false,
            wrapup_fired: false,
            hard_forced: false,
            at_boundary: false,
            at_newline: false,
            closed_at: None,
            bias_buf: Vec::new(),
        }
    }

    /// Build a controller for a nonzero budget, scanning the vocabulary and
    /// encoding the injection strings for it. A convenience for tests that want
    /// one controller from a tokenizer: the scan walks the whole ~100k
    /// vocabulary, so `Generator` holds a `ThinkStatics` and builds every
    /// controller through `from_statics` instead.
    #[cfg(test)]
    fn new(budget: usize, tokenizer: &LagunaTokenizer) -> Result<Self> {
        Ok(Self::from_statics(budget, ThinkStatics::new(tokenizer)?))
    }

    /// Build a controller for a nonzero budget from an already-scanned vocab.
    fn from_statics(budget: usize, statics: ThinkStatics) -> Self {
        debug_assert!(budget > 0, "a zero budget must use ThinkBudget::off");
        let ThinkStatics {
            wait_ids,
            wrapup,
            transition,
            paragraph,
        } = statics;
        Self {
            budget,
            armed: true,
            wait_ids,
            wrapup,
            transition,
            paragraph,
            ..Self::off()
        }
    }

    /// Clear the per-call state, keeping the scanned vocab sets.
    fn reset(&mut self) {
        self.armed = self.budget > 0;
        self.queue.clear();
        self.clock = 0;
        self.forced_pending = false;
        self.awaiting_natural = false;
        self.wrapup_fired = false;
        self.hard_forced = false;
        self.at_boundary = false;
        self.at_newline = false;
        self.closed_at = None;
    }

    fn enabled(&self) -> bool {
        self.budget > 0
    }

    /// Grace tokens allowed past the budget before the hard force fires.
    fn grace(&self) -> usize {
        think_grace(self.budget)
    }

    /// Decode tokens that must remain past `budget + grace`: every token both
    /// injections can emit (they are off the schedule clock, so they are pure
    /// extra) plus the room the reply needs.
    fn reserve(&self) -> usize {
        self.wrapup.len()
            + self.transition.len()
            + 2 * self.paragraph.len()
            + Self::CONCLUSION_RESERVE
    }

    /// Progress through the budget, as a fraction. Only called while armed, so
    /// the budget is nonzero.
    fn fraction(&self, t: usize) -> f64 {
        t as f64 / self.budget as f64
    }

    fn phase(&self, t: usize) -> ThinkPhase {
        if !self.armed {
            return ThinkPhase::Done;
        }
        if self.hard_forced {
            return ThinkPhase::Forcing;
        }
        let x = self.fraction(t);
        if x >= 1.0 {
            ThinkPhase::Grace
        } else if !self.wrapup_fired && (Self::WRAPUP_LO..=Self::WRAPUP_HI).contains(&x) {
            ThinkPhase::WrapUp
        } else if x >= Self::PULL_LO {
            ThinkPhase::Pulling
        } else if x >= Self::BIAS_LO {
            ThinkPhase::Damping
        } else {
            ThinkPhase::Free
        }
    }

    /// True while the schedule needs one-token-at-a-time rounds. From the
    /// wrap-up window on, both the boundary detector and the forced-token queue
    /// assume the next draw follows the last EMITTED token — which a batched
    /// verify round, whose rows are all drawn before any of them is emitted,
    /// breaks. Speculation resumes as soon as `</think>` is committed.
    ///
    /// `lookahead` is how many draws the caller may take before the next check;
    /// a round that starts below the threshold can still have rows above it.
    /// Counting each of those as a clock tick is the conservative direction —
    /// forced draws do not tick the clock at all.
    fn needs_serial_rounds(&self, lookahead: usize) -> bool {
        self.armed && self.fraction(self.clock + lookahead) >= Self::WRAPUP_LO
    }

    /// A boundary the schedule may act on: one the MODEL wrote. An injection's
    /// own trailing newline is a boundary too, and trusting it would fire the
    /// next stage immediately with nothing of the model's in between.
    fn at_safe_boundary(&self) -> bool {
        self.at_boundary && !self.awaiting_natural
    }

    /// Advance the schedule and return the sampler adjustments it wants for the
    /// next draw. Enqueues the wrap-up or the hard force when their conditions
    /// are met, so a forced draw is handed out on the same call that decides to
    /// inject.
    fn control_for(&mut self) -> ThinkControl<'_> {
        self.forced_pending = false;
        let t = self.clock;
        match self.phase(t) {
            ThinkPhase::Done => return ThinkControl::inert(),
            // Both injections wait for a sentence boundary; only the hard force
            // gives up waiting, once the grace period is spent. Neither arm can
            // fire twice — `phase` stops reporting them once their flag is set.
            ThinkPhase::WrapUp if self.at_safe_boundary() => {
                if !self.at_newline {
                    self.queue.extend(self.paragraph.iter().copied());
                }
                self.queue.extend(self.wrapup.iter().copied());
                self.wrapup_fired = true;
            }
            ThinkPhase::Grace if self.at_safe_boundary() || t >= self.budget + self.grace() => {
                if !self.at_newline {
                    self.queue.extend(self.paragraph.iter().copied());
                }
                self.queue.extend(self.transition.iter().copied());
                self.hard_forced = true;
            }
            _ => {}
        }
        let x = self.fraction(t);

        if let Some(&id) = self.queue.front() {
            self.forced_pending = true;
            return ThinkControl {
                bias: &[],
                pull: None,
                force: Some(id),
            };
        }

        self.bias_buf.clear();
        let bias = Self::BIAS_MAX * ramp(x, Self::BIAS_LO, Self::BIAS_HI);
        if bias != 0.0 {
            self.bias_buf
                .extend(self.wait_ids.iter().map(|&id| (id, bias)));
        }
        let alpha = Self::PULL_MAX * ramp(x, Self::PULL_LO, 1.0);
        let pull = (alpha > 0.0).then_some((LagunaTokenizer::THINK_CLOSE, alpha));
        ThinkControl {
            bias: &self.bias_buf,
            pull,
            force: None,
        }
    }

    /// Record a committed token drawn at thinking-token index `t`. Pops the
    /// injection queue when the draw was a forced one, and disarms permanently
    /// once `</think>` lands — after which the model stops seeing any bias and
    /// normal EOG stopping resumes.
    fn on_committed(&mut self, t: usize, token: u32) {
        if !self.armed {
            return;
        }
        if self.forced_pending {
            self.forced_pending = false;
            let popped = self.queue.pop_front();
            debug_assert_eq!(
                popped,
                Some(token),
                "a forced draw must return the forced id"
            );
            self.awaiting_natural = true;
        } else {
            self.clock += 1;
            self.awaiting_natural = false;
        }
        if token == LagunaTokenizer::THINK_CLOSE {
            self.armed = false;
            self.closed_at = Some(t);
        }
    }

    /// Record the text a committed token finalized, as the streaming decoder
    /// reported it: `None` while the token sits mid-codepoint.
    fn on_emitted(&mut self, chunk: Option<&str>) {
        self.at_boundary = chunk.is_some_and(ends_at_boundary);
        self.at_newline = chunk.is_some_and(ends_at_newline);
    }

    /// The call's report. `decoded` is the total emitted-token count, used when
    /// the block never closed.
    fn stats(&self, decoded: usize) -> ThinkStats {
        ThinkStats {
            tokens: self.closed_at.map_or(decoded, |t| t + 1),
            closed: self.closed_at.is_some(),
            wrapup_fired: self.wrapup_fired,
            forced: self.hard_forced,
        }
    }
}

/// The half of a `ThinkBudget` that depends only on the vocabulary: the
/// wait-token set (a decode of every id in the ~100k vocabulary) and the encoded
/// injection strings. Identical for every budget, so a generator that sets a
/// ceiling per request builds this once and reuses it.
#[derive(Clone)]
struct ThinkStatics {
    wait_ids: Vec<u32>,
    wrapup: Vec<u32>,
    transition: Vec<u32>,
    paragraph: Vec<u32>,
}

impl ThinkStatics {
    fn new(tokenizer: &LagunaTokenizer) -> Result<Self> {
        let mut transition = tokenizer.encode(ThinkBudget::TRANSITION_TEXT)?;
        transition.push(LagunaTokenizer::THINK_CLOSE);
        Ok(Self {
            wait_ids: scan_wait_tokens(tokenizer)?,
            wrapup: tokenizer.encode(ThinkBudget::WRAPUP_TEXT)?,
            transition,
            paragraph: tokenizer.encode(ThinkBudget::PARAGRAPH_TEXT)?,
        })
    }
}

/// Grace tokens a budget of `budget` is allowed past itself before the hard
/// force fires. Free-standing so the feasibility search can ask about budgets no
/// controller has been built for.
fn think_grace(budget: usize) -> usize {
    (budget / ThinkBudget::GRACE_DIV).clamp(ThinkBudget::GRACE_MIN, ThinkBudget::GRACE_MAX)
}

/// Total tokens a budget of `budget` claims: the budget itself, its grace window
/// and `reserve`. `None` when the sum overflows — a wrapping one would sail past
/// the guard and leave the whole schedule silently inert (release builds do not
/// trap).
///
/// The one place the "does this ceiling fit" arithmetic lives:
/// `check_think_budget` validates through it and `feasible_think_budget`
/// searches through it, so the two can never disagree about what fits.
fn think_span(budget: usize, reserve: usize) -> Option<usize> {
    budget
        .checked_add(think_grace(budget))?
        .checked_add(reserve)
}

/// The largest reasoning ceiling `<= requested` that `check_think_budget` accepts
/// for a `max_new`-token reply, or `None` when no ceiling at or above
/// `ThinkBudget::MIN_BUDGET` fits — in which case the caller has to reason
/// uncapped rather than fail, since every expressible ceiling would force the
/// exit with no tokens left to answer in.
///
/// This is the check an API layer runs while validating a request, so it takes
/// no tokenizer (the reserve is `ThinkBudget::PINNED_RESERVE`) and says nothing
/// about the forced-reasoning floor: a caller that also sets `min_think` still
/// has to keep it below the ceiling.
pub fn feasible_think_budget(requested: usize, max_new: usize) -> Option<usize> {
    let fits = |budget: usize| {
        think_span(budget, ThinkBudget::PINNED_RESERVE).is_some_and(|n| n < max_new)
    };

    if requested < ThinkBudget::MIN_BUDGET || !fits(ThinkBudget::MIN_BUDGET) {
        return None;
    }
    if fits(requested) {
        return Some(requested);
    }
    // `think_span` never decreases as the budget grows, so the feasible ceilings
    // are a prefix of the range and binary search lands on its last element.
    let (mut lo, mut hi) = (ThinkBudget::MIN_BUDGET, requested);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Some(lo)
}

/// Validate a reasoning ceiling against the forced-reasoning floor and the token
/// budget. The ceiling must leave room for the grace period AND the injected
/// sentences: firing the forced exit with no budget left to answer in is the
/// exact failure the graduated schedule exists to avoid.
fn check_think_budget(think: &ThinkBudget, min_think: usize, max_tokens: usize) -> Result<()> {
    if !think.enabled() {
        return Ok(());
    }
    let budget = think.budget;
    ensure!(
        budget >= ThinkBudget::MIN_BUDGET,
        "max_think ({budget}) is below the minimum of {}: a smaller budget has no room for the \
         graduated stages and would jump straight to forcing `</think>` mid-sentence",
        ThinkBudget::MIN_BUDGET,
    );
    ensure!(
        min_think == 0 || min_think < budget,
        "min_think ({min_think}) must be below max_think ({budget})",
    );
    let grace = think.grace();
    let reserve = think.reserve();
    ensure!(
        think_span(budget, reserve).is_some_and(|n| n < max_tokens),
        "max_think ({budget}) leaves no room to conclude: max_tokens ({max_tokens}) must exceed \
         max_think + {grace} grace + {reserve} reserved for the injected sentences and the answer",
    );
    Ok(())
}

/// Logistic ramp: exactly 0 at or below `lo`, exactly 1 at or above `hi`, and an
/// S-curve between, steep enough that it would read `EDGE` at `lo` and
/// `1 - EDGE` at `hi` were it not clamped. Clamping the ends matters: below `lo`
/// the schedule must be a true no-op, and above `hi` it must hold at full
/// strength however far past the budget generation runs.
fn ramp(x: f64, lo: f64, hi: f64) -> f32 {
    const EDGE: f64 = 0.02;
    if x <= lo {
        return 0.0;
    }
    if x >= hi {
        return 1.0;
    }
    let center = 0.5 * (lo + hi);
    let k = ((1.0 - EDGE) / EDGE).ln() / (0.5 * (hi - lo));
    (1.0 / (1.0 + (-k * (x - center)).exp())) as f32
}

/// True when `text` — a chunk the streaming decoder just finalized — ends at a
/// sentence or paragraph boundary. Injected text has to land at one of these or
/// it reads as an interruption mid-clause. Only horizontal whitespace is
/// trimmed: a trailing newline IS the boundary, not padding in front of one.
fn ends_at_boundary(text: &str) -> bool {
    ends_at_newline(text) || text.trim_end().ends_with(['.', '?', '!'])
}

/// True when `text` ends a line, so injected text can start on its own line
/// without a blank one being inserted first.
fn ends_at_newline(text: &str) -> bool {
    text.trim_end_matches([' ', '\t']).ends_with('\n')
}

/// Ids whose decoded text is exactly one of the `NO_WAIT` cues after leading
/// whitespace and punctuation are stripped and the text is lowercased.
fn scan_wait_tokens(tokenizer: &LagunaTokenizer) -> Result<Vec<u32>> {
    let mut ids = Vec::new();
    for id in 0..tokenizer.vocab_size() as u32 {
        let text = tokenizer.decode(&[id])?;
        let key = text
            .trim_start_matches(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .to_lowercase();
        if ThinkBudget::NO_WAIT.contains(&key.as_str()) {
            ids.push(id);
        }
    }
    Ok(ids)
}

/// Ids whose decoded text contains `pattern` — the `--ban-string` mask.
///
/// Structural tokens are never banned, only reported: every added-vocabulary
/// entry (`<|im_start|>`, `<|im_end|>`, `<think>`, `<tool_call>`, …) plus the
/// PAD id and `eog`. Those markers carry ordinary English inside their angle
/// brackets, so an innocuous pattern like "assistant" collides with them by
/// accident; banning `</tool_call>` would let the model open a tool call it can
/// never close, and banning an EOG would stop generation from ever ending.
/// Aborting the whole run over an incidental collision would be worse than
/// warning and carrying on with the rest of the matches.
///
/// `eog` must be the set the decode loop actually stops on — `Sampler::eog_ids`,
/// which `XwenConfig` derives from the checkpoint's own metadata — and not the
/// compile-time `LagunaTokenizer::EOG`. The two agree on both shipped
/// checkpoints, but they are independent sources: a checkpoint declaring some
/// other `eos_token_id` would leave its real stop token outside the protected
/// set, and `--ban-string` could then ban the only token that ends a reply.
/// Protecting what the loop stops on makes that impossible by construction.
///
/// KNOWN LIMITATION: byte-level BPE can spell a banned glyph out of raw
/// byte-fallback tokens (an em dash is also the byte triple 228/292/312), and a
/// static id mask cannot see that coming. This covers decoded-substring matches
/// only; a byte automaton over the emitted stream is the general fix (TODO.md).
fn scan_banned(tokenizer: &LagunaTokenizer, eog: &[u32], pattern: &str) -> Result<Vec<u32>> {
    ensure!(!pattern.is_empty(), "--ban-string cannot be empty");
    let mut protected = tokenizer.added_token_ids();
    protected.push(LagunaTokenizer::PAD);
    protected.extend_from_slice(eog);
    protected.sort_unstable();
    protected.dedup();

    let mut ids = Vec::new();
    let mut skipped = Vec::new();
    for id in 0..tokenizer.vocab_size() as u32 {
        let text = tokenizer.decode(&[id])?;
        if !text.contains(pattern) {
            continue;
        }
        if protected.binary_search(&id).is_ok() {
            skipped.push(format!("{id} ({text:?})"));
        } else {
            ids.push(id);
        }
    }
    if !skipped.is_empty() {
        eprintln!(
            "xwen: --ban-string {pattern:?} also matches the structural token(s) {} — \
             leaving them enabled, since generation and the chat template depend on them",
            skipped.join(", "),
        );
    }
    ensure!(
        !ids.is_empty(),
        "--ban-string {pattern:?} matches no bannable token in the vocabulary{}",
        if skipped.is_empty() {
            ""
        } else {
            " (only structural tokens matched)"
        },
    );
    Ok(ids)
}

impl Generator {
    pub fn new(model: XwenModel, tokenizer: LagunaTokenizer, sampler: Sampler) -> Self {
        Self {
            model,
            tokenizer,
            sampler,
            drafter: None,
            spec_params: SpecParams::default(),
            min_think: 0,
            think: ThinkBudget::off(),
            think_statics: None,
            blacklist: Vec::new(),
            grammar: None,
            last_logits: None,
        }
    }

    /// Open a checkpoint on `device` and assemble a generator around it: the
    /// GGUF, its config, the tokenizer and the sampler's EOG set all come
    /// from one place so the CLI and the server load a model identically.
    /// `tokenizer` is override-only — `None` uses the vocabulary embedded in
    /// the binary (`LagunaTokenizer::embedded`), which is what every default
    /// path wants. Attaching a DFlash drafter stays the caller's business —
    /// it needs the same `device`, which is why the device is passed in
    /// rather than made here.
    pub fn load(
        device: &Device,
        model: &Path,
        tokenizer: Option<&Path>,
        runner: ExpertRunner,
        max_ctx: usize,
        sampling: SamplerOptions,
    ) -> Result<Self> {
        let gguf = gguf::open(model, device)?;
        let cfg = XwenConfig::from_gguf(&gguf.content)?;
        let tok = match tokenizer {
            Some(path) => LagunaTokenizer::from_file(path)?,
            None => LagunaTokenizer::embedded()?,
        };
        let sampler = Sampler::new(sampling, cfg.eog_tokens.clone(), tok.vocab_size());
        let model = XwenModel::load(gguf, runner, max_ctx)?;
        Ok(Self::new(model, tok, sampler))
    }

    /// Rebuild the sampler from `opts`, keeping the loaded EOG set. Lets a
    /// long-lived generator serve requests with per-request sampling.
    pub fn set_sampler(&mut self, opts: SamplerOptions) {
        self.sampler = Sampler::new(
            opts,
            self.sampler.eog_ids().to_vec(),
            self.sampler.encodable(),
        );
    }

    /// Set the forced-reasoning floor (see the `min_think` field). Applies to
    /// every subsequent `generate` call, on both the plain and speculative
    /// decode paths.
    pub fn set_min_think(&mut self, n: usize) {
        self.min_think = n;
    }

    /// Set the reasoning ceiling (see the `think` field). Applies to every
    /// subsequent `generate` call, on both decode paths. A budget of 0 — the
    /// default — does no work at all; the first nonzero one scans the vocabulary
    /// for the wait-token set and every later call reuses that scan, so setting
    /// a ceiling per request is cheap.
    pub fn set_max_think(&mut self, n: usize) -> Result<()> {
        if n == 0 {
            self.think = ThinkBudget::off();
            return Ok(());
        }
        if self.think_statics.is_none() {
            self.think_statics = Some(ThinkStatics::new(&self.tokenizer)?);
        }
        let statics = self.think_statics.clone().expect("just populated");
        self.think = ThinkBudget::from_statics(n, statics);
        Ok(())
    }

    /// Install (or clear) the grammar constraint for subsequent decode calls.
    /// A grammar is single-use — it advances with the tokens it accepts — so
    /// the server sets one per job, unconditionally: `None` here is what clears
    /// the previous request's state.
    pub fn set_grammar(&mut self, grammar: Option<GrammarState>) {
        self.grammar = grammar;
    }

    /// Ban every token whose decoded text contains one of `patterns`, for the
    /// life of the generator (so a chat session configures it once and every
    /// turn inherits it). Errors when a pattern matches nothing or would ban a
    /// structural token.
    pub fn set_banned_strings(&mut self, patterns: &[String]) -> Result<()> {
        let mut ids = Vec::new();
        // The protected set comes from the sampler so that it is the same set
        // the decode loop stops on. See `scan_banned`.
        let eog = self.sampler.eog_ids().to_vec();
        for pattern in patterns {
            ids.extend(scan_banned(&self.tokenizer, &eog, pattern)?);
        }
        ids.sort_unstable();
        ids.dedup();
        self.blacklist = ids;
        Ok(())
    }

    /// The ids banned from a draw: the `--ban-string` blacklist always, plus —
    /// while the forced-reasoning floor is in effect — `</think>` and the EOG
    /// ids (banning EOG too keeps the forcing window from ending the reply
    /// outright). Empty in the default configuration, so the default path stays
    /// on the unmasked sampler.
    fn ban_ids(&self, with_think_exit: bool) -> Vec<u32> {
        let mut ban = self.blacklist.clone();
        if with_think_exit && self.min_think > 0 {
            ban.push(LagunaTokenizer::THINK_CLOSE);
            ban.extend_from_slice(self.sampler.eog_ids());
        }
        ban
    }

    /// Attach a DFlash drafter so subsequent `generate` calls run speculative
    /// decoding. The target's spec taps come from the drafter's config via
    /// `DflashConfig::spec_tap_layers` (the enforced `target_layers -> l_out`
    /// translation), kept in `target_layers` order — the order the drafter's
    /// encoder concatenates them.
    ///
    /// A drafter paired with the wrong target is rejected here, by the same
    /// `DflashConfig::check_against_target` serve runs at startup — the checks
    /// are shared so the two paths cannot drift apart. Without it the failure
    /// modes are a tap past the target's last layer (which `set_spec_taps`
    /// asserts, i.e. panics), a hidden size the encoder cannot consume (which
    /// would not surface until the first forward), and a mask token outside the
    /// target's vocabulary (which fails at the first speculative round).
    pub fn attach_drafter(&mut self, drafter: DflashDrafter, params: SpecParams) -> Result<()> {
        let target = self.model.config();
        drafter
            .config()
            .check_against_target(target.hidden, target.n_layer, target.vocab)?;
        let tap_layers = drafter.config().spec_tap_layers()?;
        self.model.set_spec_taps(Some(tap_layers));
        self.drafter = Some(drafter);
        self.spec_params = params;
        Ok(())
    }

    /// Longest prompt + generation budget the KV cache can hold. Callers can
    /// use this to report context headroom or validate input sizes up front.
    pub fn max_ctx(&self) -> usize {
        self.model.max_ctx()
    }

    /// The loaded tokenizer, for callers that encode prompts themselves and
    /// feed the ids to `prefill_tokens`.
    pub fn tokenizer(&self) -> &LagunaTokenizer {
        &self.tokenizer
    }

    /// Identity of the checkpoint behind this generator — the stamp a persisted
    /// cache image carries, and what it is refused against after a model change.
    pub fn checkpoint_id(&self) -> crate::gguf::CheckpointId {
        self.model.checkpoint_id()
    }

    /// Tokens the KV cache currently holds. This — not a caller's own count of
    /// what it fed in — is the authority on where the next `prefill_tokens`
    /// resumes: a `decode_loop` that stops at its `max_new` cap deliberately
    /// skips the final feed-back forward, so its last emitted token is NOT in
    /// the cache.
    pub fn cache_len(&self) -> usize {
        self.model.cache_len()
    }

    /// Drop the whole cached context. The next `prefill_tokens` starts at
    /// position 0.
    pub fn reset_cache(&mut self) -> Result<()> {
        self.model.reset_cache()?;
        self.last_logits = None;
        Ok(())
    }

    /// Capture the cache so a later `restore_cache_snapshot` can rewind to the
    /// current `cache_len()` — the shared prefix of a conversation's next turn.
    pub fn take_cache_snapshot(&self) -> Result<CacheSnapshot> {
        self.model.take_cache_snapshot()
    }

    /// Rewind the cache to `snapshot` (see `XwenModel::restore_cache_snapshot`
    /// for the precondition on the full-attention layers). Any logits held from
    /// an earlier prefill are dropped: they describe a cache tail that no longer
    /// exists, so the caller re-prefills from `cache_len()`.
    pub fn restore_cache_snapshot(&mut self, snapshot: &CacheSnapshot) -> Result<()> {
        self.model.restore_cache_snapshot(snapshot)?;
        self.last_logits = None;
        Ok(())
    }

    /// Copy the full-attention layers' committed rows to host RAM, so this
    /// conversation can be parked and the cache stack given to another one (see
    /// `XwenModel::export_full_kv`). Pair it with `take_cache_snapshot().
    /// to_host()`, which carries the SWA rings the row image cannot.
    pub fn export_full_kv(&self) -> Result<HostFullKv> {
        self.model.export_full_kv()
    }

    /// Upload `image`'s rows `[0, pos)` back into the full-attention layers (see
    /// `XwenModel::import_full_kv` — the SWA rings still need their snapshot
    /// restored). Held logits are dropped: they describe the cache tail of
    /// whichever conversation was resident before.
    pub fn import_full_kv(&mut self, image: &HostFullKv, pos: usize) -> Result<()> {
        self.model.import_full_kv(image, pos)?;
        self.last_logits = None;
        Ok(())
    }

    /// Whether the loaded model could take this pair of images at `pos`, without
    /// changing anything (see `XwenModel::check_importable`). The answer comes from
    /// the model that is loaded, not from re-reading the checkpoint that named it.
    pub fn check_importable(
        &self,
        image: &HostFullKv,
        rings: &HostSnapshot,
        pos: usize,
    ) -> Result<()> {
        self.model.check_importable(image, rings, pos)
    }

    /// Generate up to max_tokens from a rendered prompt, streaming text chunks
    /// to `on_text`. Stops on any EOG token, or early when `should_stop`
    /// returns true (polled once per prefill chunk and once per decoded token;
    /// an early stop is reported via `GenStats::cancelled`).
    ///
    /// The prompt string is encoded as-is: BOS ownership lives with the caller
    /// (the chat template writes BOS as literal text; the raw-prompt CLI path
    /// prepends it). This method never injects a BOS of its own, and it treats
    /// every byte as trusted — chat callers with client content in the prompt
    /// should use `generate_with_content_ranges` instead.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        on_text: &mut dyn FnMut(&str),
        should_stop: &mut dyn FnMut() -> bool,
    ) -> Result<GenStats> {
        self.generate_with_content_ranges(prompt, &[], max_tokens, on_text, should_stop)
    }

    /// `generate` for a chat-rendered prompt: `content_ranges` are the
    /// client-content byte ranges reported by `chat::build_prompt_with_spans`,
    /// and the prompt is encoded via `LagunaTokenizer::encode_prompt` so
    /// added-token strings inside client content stay plain text instead of
    /// becoming control tokens.
    pub fn generate_with_content_ranges(
        &mut self,
        prompt: &str,
        content_ranges: &[Range<usize>],
        max_tokens: usize,
        on_text: &mut dyn FnMut(&str),
        should_stop: &mut dyn FnMut() -> bool,
    ) -> Result<GenStats> {
        if self.drafter.is_some() {
            return self.generate_spec(prompt, content_ranges, max_tokens, on_text, should_stop);
        }
        self.model.reset_cache()?;
        // Any logits a previous `prefill_tokens` left behind describe a cache
        // that no longer exists.
        self.last_logits = None;
        let device = self.model.device().clone();

        let tokens = self.tokenizer.encode_prompt(prompt, content_ranges)?;
        ensure!(!tokens.is_empty(), "prompt encoded to zero tokens");

        // The Full KV cache errors if appended past max_ctx; validate the whole
        // budget (prompt + generation) up front with a clear message.
        let max_ctx = self.model.max_ctx();
        ensure!(
            tokens.len() + max_tokens <= max_ctx,
            "prompt ({} tokens) + max_tokens ({max_tokens}) exceeds max_ctx ({max_ctx}); \
             raise --max-ctx or lower -n",
            tokens.len()
        );
        // A floor at or past the token budget would ban `</think>`/EOG for the
        // whole generation: the reply is all reasoning, and the chat REPL
        // (which splits on the `</think>` literal) would show nothing and file
        // the raw reasoning into history as content. Reject it up front.
        ensure!(
            self.min_think == 0 || self.min_think < max_tokens,
            "min_think ({}) must be below max_tokens ({max_tokens}) or the think block \
             can never close",
            self.min_think
        );
        check_think_budget(&self.think, self.min_think, max_tokens)?;
        self.think.reset();

        // Benchmark mode (XWEN_BENCH): run a full prefill once to page the
        // touched weights into the GPU residency set and compile every pipeline,
        // then reset so the timed run below measures steady state rather than
        // one-time upload/compile costs. Off by default; never affects output.
        if std::env::var_os("XWEN_BENCH").is_some() {
            let mut warm_pos = 0usize;
            for chunk in tokens.chunks(PREFILL_CHUNK) {
                let input = Tensor::new(chunk, &device)?;
                let _ = self.model.forward(&input, warm_pos)?;
                warm_pos += chunk.len();
            }
            device.synchronize()?;
            self.model.reset_cache()?;
        }

        // ---- Prefill: feed the prompt in chunks, advancing the absolute pos.
        let prefill_start = Instant::now();
        let mut cancelled = false;
        let mut pos = 0usize;
        let mut logits = None;
        for chunk in tokens.chunks(PREFILL_CHUNK) {
            if should_stop() {
                cancelled = true;
                break;
            }
            let input = Tensor::new(chunk, &device)?;
            logits = Some(self.model.forward(&input, pos)?);
            pos += chunk.len();
        }
        // Force the queued prefill GPU work to complete before timing it (the
        // decode loop's per-token readback would otherwise absorb it).
        device.synchronize()?;
        let prefill_secs = prefill_start.elapsed().as_secs_f64();
        let logits = match logits {
            Some(logits) if !cancelled => logits,
            // Cancelled during (or before) prefill: no logits to decode from.
            _ => {
                return Ok(GenStats {
                    prefill_tokens: pos,
                    prefill_secs,
                    cancelled: true,
                    ..GenStats::default()
                });
            }
        };

        // ---- Decode: sample, stream, feed back, one token at a time. The
        // rendered prompt's `</think>` is part of the text stream this path
        // hands out (the REPL splits on the literal), so the loop runs entirely
        // in the answer section and forwards every chunk verbatim.
        let mut forward_text = |event: GenEvent| {
            if !event.text().is_empty() {
                on_text(event.text());
            }
        };
        let outcome = self.decode_tokens(
            logits,
            pos,
            false,
            max_tokens,
            &mut forward_text,
            should_stop,
        )?;
        let decoded = outcome.tokens_out;

        Ok(GenStats {
            prefill_tokens: tokens.len(),
            prefill_secs,
            decode_tokens: decoded,
            decode_secs: outcome.decode_secs,
            cancelled: cancelled || outcome.cancelled,
            spec: None,
            think: self.think.enabled().then(|| self.think.stats(decoded)),
        })
    }

    /// Feed `tokens` to the model in prefill chunks at absolute positions
    /// `start_pos..`, leaving the KV cache extended rather than reset, and keep
    /// the final position's logits for the next `decode_loop`. This is the
    /// server-side prefill: a caller that reuses a cached prefix prefills only
    /// the tokens past it, and may call this more than once (each call resumes
    /// where the caller says) before decoding.
    ///
    /// With a drafter attached this also feeds the drafter: each chunk's target
    /// taps are fused through the drafter's encoder and injected into its KV cache
    /// at the same absolute positions, so `decode_loop_spec` can speculate from
    /// the end of the prefill. How much of a chunk the drafter takes is
    /// `drafter_span_rows` — all of it, the prefix that fits its (smaller)
    /// context, or none when its committed length disagrees with the chunk's start
    /// position. The taps are drained either way, so a dropped chunk's taps never
    /// accumulate into the next one, and the target's tokens and logits are
    /// unaffected.
    pub fn prefill_tokens(&mut self, tokens: &[u32], start_pos: usize) -> Result<()> {
        ensure!(!tokens.is_empty(), "prefill_tokens: nothing to prefill");
        let max_ctx = self.model.max_ctx();
        ensure!(
            start_pos + tokens.len() <= max_ctx,
            "prefill of {} tokens at position {start_pos} exceeds max_ctx ({max_ctx})",
            tokens.len()
        );
        let Self {
            model,
            drafter,
            last_logits,
            ..
        } = self;
        let device = model.device().clone();
        let mut pos = start_pos;
        for chunk in tokens.chunks(PREFILL_CHUNK) {
            let input = Tensor::new(chunk, &device)?;
            *last_logits = Some(model.forward(&input, pos)?);
            // Drain unconditionally: taps left in the model would be read by a
            // later chunk as if they described its own positions.
            let taps = model.take_spec_taps();
            if let Some(drafter) = drafter.as_mut() {
                inject_capped(drafter, &taps, pos, chunk.len())?;
            }
            pos += chunk.len();
        }
        Ok(())
    }

    /// True when a DFlash drafter is attached, whether or not it can currently
    /// speculate (see `spec_ready_at`).
    ///
    /// A stored cache image can carry drafter planes into a process that has none: the
    /// same conversation, written by a server started with `--draft` and resumed by one
    /// started without it. Asking is how a page-in decides to leave those planes alone
    /// rather than discovering the absence as an error.
    pub fn has_drafter(&self) -> bool {
        self.drafter.is_some()
    }

    /// Positions the drafter's own KV cache can hold — smaller than the target's
    /// context by design, since its f32 planes cost 40 KiB/token on the 27B
    /// sidecar and 48 on the 35B-A3B's extra layer
    /// (`hub::Model::draft_kv_bytes_per_token`). `None` when no drafter is
    /// attached.
    pub fn drafter_max_ctx(&self) -> Option<usize> {
        self.drafter.as_ref().map(|d| d.max_ctx())
    }

    /// Tokens the drafter's KV cache holds, 0 with no drafter attached. Equal to
    /// `cache_len()` whenever speculation is live.
    pub fn drafter_committed_len(&self) -> usize {
        self.drafter.as_ref().map_or(0, |d| d.committed_len())
    }

    /// Whether `decode_loop_spec` can actually speculate when resumed at `pos`:
    /// a drafter is attached, its cache covers exactly the same tokens the
    /// target's does, and `pos` is inside its context. It is always SAFE to call
    /// `decode_loop_spec` when this is false — it falls back to plain rounds,
    /// which draw the same tokens — so this is a dispatch/telemetry question, not
    /// a correctness one.
    pub fn spec_ready_at(&self, pos: usize) -> bool {
        self.drafter
            .as_ref()
            .is_some_and(|d| d.committed_len() == pos && pos < d.max_ctx())
    }

    /// Align the drafter's cache with a target cache that was just rewound to
    /// `pos` within the same conversation. A drafter holding more than `pos`
    /// tokens is truncated, which is exact — its cache is position-indexed with no
    /// ring. A drafter that never reached `pos` is left ALONE: its rows
    /// `[0, committed)` are this conversation's own and stay valid across the
    /// rewind, injection refuses to resume anywhere but exactly `committed`
    /// anyway, and keeping them means a later rewind that lands on `committed`
    /// re-enables speculation instead of waiting for a prefill from zero. No-op
    /// with no drafter attached.
    ///
    /// This is the intra-slot rewind arm only. A conversation switch must go
    /// through `reset_drafter` or an `import_drafter_cache`, since the rows would
    /// then belong to somebody else.
    pub fn sync_drafter_to(&mut self, pos: usize) -> Result<()> {
        if let Some(drafter) = self.drafter.as_mut()
            && pos < drafter.committed_len()
        {
            drafter.truncate(pos)?;
        }
        Ok(())
    }

    /// Drop the drafter's cached context, so it resynchronizes on the next prefill
    /// from position 0. Pair it with `reset_cache` on the target. No-op with no
    /// drafter attached.
    pub fn reset_drafter(&mut self) {
        if let Some(drafter) = self.drafter.as_mut() {
            drafter.reset();
        }
    }

    /// Copy the drafter's committed rows to host RAM so this conversation can be
    /// parked and the drafter's cache given to another one — the drafter half of
    /// `export_full_kv`. `None` when no drafter is attached; a conversation parked
    /// without an image resumes with speculation off (see `import_drafter_cache`).
    pub fn export_drafter_cache(&self) -> Result<Option<DrafterImage>> {
        match &self.drafter {
            Some(drafter) => Ok(Some(drafter.export_cache()?)),
            None => Ok(None),
        }
    }

    /// Upload `image`'s rows `[0, pos)` back into the drafter and set its
    /// committed length to `pos`. Errors when no drafter is attached — a caller
    /// holding an image is a caller that had one.
    pub fn import_drafter_cache(&mut self, image: &DrafterImage, pos: usize) -> Result<()> {
        let drafter = self
            .drafter
            .as_mut()
            .ok_or_else(|| anyhow!("import_drafter_cache: no drafter is attached"))?;
        drafter.import_cache(image, pos)
    }

    /// Decode from the logits `prefill_tokens` left behind, emitting one
    /// [`GenEvent`] per token. `start_pos` is the absolute position the first
    /// decoded token occupies (the prefilled length). `starts_in_thinking` says
    /// the prompt ended inside a `<think>` block, so tokens are reported as
    /// reasoning until `</think>` is sampled — that token is the last
    /// `ThinkingTok`, and everything after it is answer text.
    ///
    /// Stops on any EOG token, at `max_new` emitted tokens, or when
    /// `should_stop` (polled once per token, before the draw) returns true. The
    /// thinking floor and budget (`set_min_think`/`set_max_think`) apply exactly
    /// as they do to `generate`.
    ///
    /// The stopping EOG token is neither emitted nor cached, and a run that ends
    /// at the `max_new` cap skips the feed-back forward for its last emitted
    /// token — so a caller tracking what the cache holds reads `cache_len()`
    /// rather than counting the events it received.
    pub fn decode_loop(
        &mut self,
        start_pos: usize,
        starts_in_thinking: bool,
        max_new: usize,
        on_event: &mut dyn FnMut(GenEvent),
        should_stop: &mut dyn FnMut() -> bool,
    ) -> Result<DecodeOutcome> {
        // Same guards `generate` applies: a floor at or past the budget would
        // ban `</think>` for the whole run and the block could never close.
        ensure!(
            self.min_think == 0 || self.min_think < max_new,
            "min_think ({}) must be below max_new ({max_new}) or the think block can never close",
            self.min_think
        );
        check_think_budget(&self.think, self.min_think, max_new)?;
        let logits = self
            .last_logits
            .take()
            .ok_or_else(|| anyhow!("decode_loop: no prefill logits; call prefill_tokens first"))?;
        self.think.reset();
        self.decode_tokens(
            logits,
            start_pos,
            starts_in_thinking,
            max_new,
            on_event,
            should_stop,
        )
    }

    /// Speculative counterpart to [`decode_loop`](Self::decode_loop): same
    /// arguments, same `GenEvent` surface, same stopping rules, but decode runs
    /// the DFlash round loop (draft a block, verify it in one batched target
    /// forward, accept the matching prefix) instead of one target forward per
    /// token. Requires an attached drafter.
    ///
    /// The sample stream is the invariant: exactly one target-sampler draw per
    /// committed token, in sequence order, on every path — so under the same seed
    /// this decodes what `decode_loop` would, except where the batched forward's
    /// numerics flip a near-tie. Rounds where speculation cannot help change only
    /// the round STRUCTURE, never the draws: the wall-clock auto-pause, a thinking
    /// budget that needs serial rounds, and a conversation that has run past the
    /// drafter's (smaller) context all fall back to plain single-token rounds.
    ///
    /// Cache bookkeeping matches `decode_loop`'s contract, which a caller reusing
    /// the prefix depends on: on return the KV cache holds every emitted token
    /// except, where the last one was never forwarded, that one — never a token
    /// the callback did not see. A cancel mid-span is what makes this non-trivial;
    /// the round rolls the verify span back to the positions backing its emitted
    /// tokens rather than the positions it verified.
    pub fn decode_loop_spec(
        &mut self,
        start_pos: usize,
        starts_in_thinking: bool,
        max_new: usize,
        on_event: &mut dyn FnMut(GenEvent),
        should_stop: &mut dyn FnMut() -> bool,
    ) -> Result<DecodeOutcome> {
        // The same guards `decode_loop` applies.
        ensure!(
            self.min_think == 0 || self.min_think < max_new,
            "min_think ({}) must be below max_new ({max_new}) or the think block can never close",
            self.min_think
        );
        check_think_budget(&self.think, self.min_think, max_new)?;
        // Every position this loop reasons about is absolute, and the rollback
        // arithmetic assumes the first decoded token lands right where the cache
        // ends. A disagreement means the caller's prefix bookkeeping and the model
        // have drifted apart, which would silently verify at the wrong offsets.
        ensure!(
            self.model.cache_len() == start_pos,
            "decode_loop_spec: start_pos {start_pos} disagrees with the {} tokens the KV cache \
             holds",
            self.model.cache_len()
        );
        // Checked before the logits are taken, so a caller that reaches here
        // without a drafter can still fall back to `decode_loop` — the held
        // prefill logits are the only thing that describes the cache's tail, and
        // an error path must not destroy them.
        ensure!(
            self.drafter.is_some(),
            "decode_loop_spec: no drafter is attached"
        );
        let ban_floor = self.ban_ids(true);
        let ban_plain = self.ban_ids(false);
        let min_think = self.min_think;
        let logits = self.last_logits.take().ok_or_else(|| {
            anyhow!("decode_loop_spec: no prefill logits; call prefill_tokens first")
        })?;
        // Destructure into per-field borrows so the drafter can stay borrowed
        // across target-model / sampler / tokenizer calls (distinct fields).
        let Self {
            model,
            tokenizer,
            sampler,
            drafter,
            spec_params,
            think,
            grammar,
            ..
        } = self;
        let drafter = drafter
            .as_mut()
            .expect("checked before the logits were taken");
        think.reset();
        let device = model.device().clone();
        let params = spec_params.clone();

        // Draft-block width is bounded by the trained block size (id_last plus
        // block_size-1 masks) and the drafter's mask token.
        let block_size = drafter.config().block_size;
        let mask_token_id = drafter.config().mask_token_id;
        let draft_max = params.draft_max.min(block_size.saturating_sub(1));
        let target_max_ctx = model.max_ctx();
        let draft_ctx = drafter.max_ctx();

        let decode_start = Instant::now();
        let mut stream = tokenizer.decode_stream();
        let mut stats = SpecStats::default();
        // Absolute position of the token that has been sampled but not yet
        // forwarded, and equally the number of tokens the KV cache holds.
        let mut n_past = start_pos;
        let mut decoded = 0usize;
        let mut thinking_tokens = 0usize;
        let mut in_thinking = starts_in_thinking;
        let mut hit_eog = false;
        let mut cancelled = false;
        // Wall-clock auto-pause, constructed per call: its EMAs describe THIS
        // request's text, and carrying them across requests would let one
        // conversation's economics pause another's.
        let mut pause = PauseController::new(params.pause_margin as f64);

        let finish = |decoded, thinking_tokens, hit_eog, cancelled, stats| DecodeOutcome {
            tokens_out: decoded,
            thinking_tokens,
            hit_eog,
            cancelled,
            decode_secs: decode_start.elapsed().as_secs_f64(),
            spec: Some(stats),
        };

        // A zero budget must not draw from the sampler at all.
        if max_new == 0 {
            device.synchronize()?;
            return Ok(finish(0, 0, false, false, stats));
        }
        if should_stop() {
            device.synchronize()?;
            return Ok(finish(0, 0, false, true, stats));
        }

        // First token: sampled from the prefill logits, exactly as plain decode
        // does. Every later token is committed by a verify round. Emitted index 0,
        // so the forced-reasoning ban applies whenever the floor is nonzero.
        let mut id_last = {
            let banned = if min_think > 0 {
                &ban_floor
            } else {
                &ban_plain
            };
            let allowed = match grammar.as_mut() {
                Some(g) => g.mask_words()?,
                None => None,
            };
            let tc = think.control_for();
            let ctl = SampleControl {
                allowed,
                banned,
                bias: tc.bias,
                pull: tc.pull,
                force: tc.force,
            };
            let token = sampler.sample_controlled(&logits, &ctl)?;
            think.on_committed(0, token);
            if let Some(g) = grammar.as_mut() {
                g.on_committed(token)?;
            }
            token
        };
        if sampler.is_eog(id_last) {
            device.synchronize()?;
            return Ok(finish(0, 0, true, false, stats));
        }
        {
            let chunk = stream.step(id_last)?;
            think.on_emitted(chunk.as_deref());
            on_event(section_event(
                id_last,
                chunk.unwrap_or_default(),
                &mut in_thinking,
                &mut thinking_tokens,
            ));
            decoded += 1;
            if should_stop() {
                cancelled = true;
            }
        }
        // A single-token value (`true`, a bare number) can complete the grammar
        // on the first draw; the round loop must not start.
        if grammar.as_ref().is_some_and(|g| g.is_done()) {
            device.synchronize()?;
            return Ok(finish(decoded, thinking_tokens, true, cancelled, stats));
        }

        while !cancelled && decoded < max_new {
            if should_stop() {
                cancelled = true;
                break;
            }

            // Speculation needs the drafter's cache to describe exactly the tokens
            // the target's does, and a drafter cache position to write into. Its
            // context is sized independently (and smaller), so a long conversation
            // eventually runs past it; from there every round is plain, and the
            // taps are drained and dropped rather than injected.
            let spec_live = drafter_span_rows(drafter.committed_len(), draft_ctx, n_past, 1) == 1;
            let round_kind = pause.next_kind();
            // Look a full draft span ahead: a round that STARTS below the
            // threshold can still have rows above it, and those rows would read a
            // boundary flag from before the round began.
            let serial = think.needs_serial_rounds(draft_max);
            let want_draft =
                spec_live && !serial && matches!(round_kind, RoundKind::Spec | RoundKind::Probe);
            // A round the thinking budget wants serial, or one the drafter cannot
            // cover, says nothing about whether speculation pays off — keep both
            // out of the controller's accounting.
            let accounted = spec_live && !serial;
            let round_start = Instant::now();

            // ---- Draft: run the drafter over [id_last, MASK * n_draft] and
            // greedily read one token per masked position.
            let n_draft = draft_max
                .min(target_max_ctx.saturating_sub(n_past + 1))
                .min(draft_ctx.saturating_sub(n_past + 1));
            let mut drafts: Vec<u32> = Vec::new();
            if want_draft && n_draft > 0 {
                let mut noise_ids = Vec::with_capacity(n_draft + 1);
                noise_ids.push(id_last);
                noise_ids.extend(std::iter::repeat_n(mask_token_id, n_draft));
                let noise_embd = model.embed_ids(&noise_ids)?; // [n_draft+1, n_embd]
                let hidden = drafter.draft_forward(&noise_embd, n_past)?;
                // Rows 1..=n_draft are the masked positions' drafts (row 0 is
                // id_last's own position, which the target verifies).
                let hidden_masked = hidden.narrow(0, 1, n_draft)?;
                let dlogits = model.lm_head(&hidden_masked)?; // [n_draft, vocab]
                let (ids, probs) = draft_argmax_probs(&dlogits)?;
                for (&tok, &p) in ids.iter().zip(&probs) {
                    if p < params.draft_p_min {
                        break;
                    }
                    drafts.push(tok);
                }
                if drafts.len() < params.draft_min {
                    drafts.clear();
                }
            }

            // Split the round timer at the draft/commit boundary. The draft phase
            // just ended at the `draft_argmax_probs` readback (a CPU sync); the
            // commit phase below ends at its own sampler readback, so neither split
            // adds a device sync. A round that ran the drafter but kept no drafts
            // still charges its draft time here.
            let draft_ms = round_start.elapsed().as_secs_f64() * 1000.0;

            // The tokens this round commits, and — when it verified a draft block —
            // the rollback point and taps the commit below needs.
            let committed: Vec<u32>;
            let mut verified: Option<(KvCheckpoint, Vec<Tensor>)> = None;
            // Drafts this round's target samples agreed with, before the retention
            // cap below.
            let mut matched = 0usize;
            if drafts.is_empty() {
                // No usable draft: a plain single-token decode step. This is the
                // same target forward + sampler draw plain decode runs, so it
                // stays byte-identical when drafting yields nothing.
                let input = Tensor::new(&[id_last], &device)?;
                let step_logits = model.forward(&input, n_past)?;
                let taps = model.take_spec_taps(); // each [1, n_embd]
                if spec_live {
                    let fused = drafter.encode(&taps)?;
                    drafter.inject(&fused, n_past)?;
                }
                let banned = if decoded < min_think {
                    &ban_floor
                } else {
                    &ban_plain
                };
                let allowed = match grammar.as_mut() {
                    Some(g) => g.mask_words()?,
                    None => None,
                };
                let tc = think.control_for();
                let ctl = SampleControl {
                    allowed,
                    banned,
                    bias: tc.bias,
                    pull: tc.pull,
                    force: tc.force,
                };
                let s = sampler.sample_controlled(&step_logits, &ctl)?;
                think.on_committed(decoded, s);
                if let Some(g) = grammar.as_mut() {
                    g.on_committed(s)?;
                }
                stats.rounds += 1;
                committed = vec![s];
            } else {
                // ---- Verify: forward [id_last, drafts...] in one batch,
                // checkpointing the KV first so a partial accept can roll the span
                // back to the positions it keeps.
                //
                // An error raised anywhere between the checkpoint and the rollback
                // below leaves the target cache holding the full span while the
                // drafter is still at the round's start — a state nothing repairs
                // locally, because the checkpoint is dropped with the stack. The
                // server does not need it repaired: a failed job leaves the slot
                // dirty, and the `cache_len() != resume` cross-check on the next
                // request replays the prompt from zero. The conversation loses its
                // cached prefix, not its correctness.
                let span = 1 + drafts.len();
                let ckpt = model.kv_checkpoint(span)?;
                let mut verify_ids = Vec::with_capacity(span);
                verify_ids.push(id_last);
                verify_ids.extend_from_slice(&drafts);
                let vinput = Tensor::new(verify_ids.as_slice(), &device)?;
                let logits_all = model.forward_all_logits(&vinput, n_past)?; // [span, vocab]
                let taps = model.take_spec_taps(); // each [span, n_embd]
                let logits_cpu = logits_all.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;

                // Sample the target token per verify row in order (the RNG advances
                // once per committed token, matching plain decode), accepting
                // matching drafts until the first mismatch, the bonus token, an
                // EOG, or the budget — never drawing past any of them. Row i
                // commits emitted index `decoded + i`, so the forced-reasoning ban
                // applies per row; a draft proposing a banned id simply mismatches
                // the masked sample and is rejected.
                let (m, accepted) = accept_drafts(&drafts, max_new - decoded, |i| {
                    let row = logits_cpu.narrow(0, i, 1)?;
                    let t = decoded + i;
                    let banned = if t < min_think {
                        &ban_floor
                    } else {
                        &ban_plain
                    };
                    // The grammar advances at SAMPLE time (`on_committed` below),
                    // not emit time, so row i+1's mask already reflects row i —
                    // including the `</think>` activation edge landing mid-span.
                    let allowed = match grammar.as_mut() {
                        Some(g) => g.mask_words()?,
                        None => None,
                    };
                    let tc = think.control_for();
                    let ctl = SampleControl {
                        allowed,
                        banned,
                        bias: tc.bias,
                        pull: tc.pull,
                        force: tc.force,
                    };
                    let s = sampler.sample_controlled(&row, &ctl)?;
                    think.on_committed(t, s);
                    if let Some(g) = grammar.as_mut() {
                        g.on_committed(s)?;
                    }
                    // A completed grammar ends the walk like an EOG: the token
                    // that completed the value is committed (and emitted below),
                    // and no row past it is drawn.
                    Ok((
                        s,
                        sampler.is_eog(s) || grammar.as_ref().is_some_and(|g| g.is_done()),
                    ))
                })?;
                stats.rounds += 1;
                stats.drafted += drafts.len();
                // Accepted drafts are a prefix of `committed` (the walk stops at
                // the first mismatch), so capping the count at what the round
                // retains below is enough to keep rolled-back drafts out of it.
                matched = m;
                verified = Some((ckpt, taps));
                committed = accepted;
            }
            id_last = *committed.last().expect("a round always commits >= 1 token");

            // ---- Emit. The KV span is committed AFTER this, because how much of
            // it to keep depends on how far the emit got: a cancel at row j must
            // not leave rows past j — tokens the callback never saw — in the cache
            // for the next request to reuse as if they were context.
            let emit_start = Instant::now();
            let mut emitted_here = 0usize;
            for &tok in &committed {
                if sampler.is_eog(tok) {
                    hit_eog = true;
                    break;
                }
                let chunk = stream.step(tok)?;
                think.on_emitted(chunk.as_deref());
                on_event(section_event(
                    tok,
                    chunk.unwrap_or_default(),
                    &mut in_thinking,
                    &mut thinking_tokens,
                ));
                decoded += 1;
                emitted_here += 1;
                if decoded >= max_new {
                    break;
                }
                if should_stop() {
                    cancelled = true;
                    break;
                }
            }
            // Streaming is the caller's cost, not the model's; keeping it out of
            // `round_ms` is what makes the speculative and plain round costs
            // comparable (a 12-token round pays 12 callbacks, a plain round one).
            let emit_ms = emit_start.elapsed().as_secs_f64() * 1000.0;

            // `hit_eog` can only have been set by THIS round's emit: a round that
            // sets it breaks out below.
            let retained = retained_commits(committed.len(), emitted_here, hit_eog);
            stats.accepted += matched.min(retained);

            // Keep exactly the verify positions backing emitted tokens: the anchor
            // row (id_last of the previous round, emitted then) plus every token
            // emitted here but the last, which is never forwarded — the same exit
            // state plain decode leaves.
            let keep = verify_keep(committed.len(), emitted_here);
            if let Some((ckpt, taps)) = verified {
                model.kv_rollback(&ckpt, keep)?;
                let prefix: Vec<Tensor> = taps
                    .iter()
                    .map(|t| t.narrow(0, 0, keep))
                    .collect::<candle_core::Result<_>>()?;
                let fused = drafter.encode(&prefix)?;
                drafter.inject(&fused, n_past)?;
                n_past += keep;
            } else {
                // The plain step's single position is already in the cache and the
                // drafter already holds its row.
                n_past += 1;
            }

            // Feed the round's wall-clock cost to the pause controller. A round's
            // kv_rollback + encode + inject are queued but not yet synced at this
            // point, so without a barrier ~5-15ms of GPU tail would bleed into the
            // NEXT round's window and skew whichever EMA that round feeds. One
            // synchronize per round trims that CPU run-ahead; the GPU queue is
            // in-order so total GPU time is unchanged, and the cost is negligible
            // at the ~200ms round cadence. A round that requested a draft but
            // produced none ran the plain fallback (`ran_spec == false`): its
            // target-forward time is a plain sample (drafter time excluded via
            // draft_ms) and its whole cost is a speculative sample, so wasted
            // drafting stays visible to the pause decision.
            let ran_spec = !drafts.is_empty();
            device.synchronize()?;
            let round_ms = round_start.elapsed().as_secs_f64() * 1000.0 - emit_ms;
            stats.draft_ms += draft_ms;
            stats.verify_ms += round_ms - draft_ms;
            stats.verify_positions += 1 + drafts.len();
            if accounted {
                pause.record(round_kind, round_ms, draft_ms, retained, ran_spec);
                // Count every round that ran plain because of the pause: the
                // paused-plain rounds, and any probe round whose draft came up
                // empty (it ran the plain fallback). Real drafting probes are
                // excluded — they speculated.
                if round_kind == RoundKind::PausedPlain
                    || (round_kind == RoundKind::Probe && !ran_spec)
                {
                    stats.paused_rounds += 1;
                }
            }

            // A completed grammar ends the reply as a natural stop. Checked after
            // the KV/pause bookkeeping above — the completing token was emitted
            // and retained like any other, unlike an EOG (which is never
            // emitted, so `hit_eog` must influence `retained_commits` and this
            // must not).
            if grammar.as_ref().is_some_and(|g| g.is_done()) {
                hit_eog = true;
            }
            if hit_eog {
                break;
            }
        }

        device.synchronize()?;
        Ok(finish(decoded, thinking_tokens, hit_eog, cancelled, stats))
    }

    /// The single-token decode loop both `generate` and `decode_loop` run:
    /// sample under the floor/budget controls, stop at EOG, stream the finalized
    /// text, feed the token back. `logits` are for the position before
    /// `start_pos`; the caller owns the think-budget reset and the up-front
    /// budget validation.
    fn decode_tokens(
        &mut self,
        mut logits: Tensor,
        start_pos: usize,
        starts_in_thinking: bool,
        max_new: usize,
        on_event: &mut dyn FnMut(GenEvent),
        should_stop: &mut dyn FnMut() -> bool,
    ) -> Result<DecodeOutcome> {
        let device = self.model.device().clone();
        let decode_start = Instant::now();
        let ban_floor = self.ban_ids(true);
        let ban_plain = self.ban_ids(false);
        let mut stream = self.tokenizer.decode_stream();
        let mut pos = start_pos;
        let mut decoded = 0usize;
        let mut thinking_tokens = 0usize;
        let mut in_thinking = starts_in_thinking;
        let mut hit_eog = false;
        let mut cancelled = false;
        while decoded < max_new {
            if should_stop() {
                cancelled = true;
                break;
            }
            let banned = if decoded < self.min_think {
                &ban_floor
            } else {
                &ban_plain
            };
            let allowed = match self.grammar.as_mut() {
                Some(g) => g.mask_words()?,
                None => None,
            };
            let tc = self.think.control_for();
            let ctl = SampleControl {
                allowed,
                banned,
                bias: tc.bias,
                pull: tc.pull,
                force: tc.force,
            };
            let token = self.sampler.sample_controlled(&logits, &ctl)?;
            self.think.on_committed(decoded, token);
            if self.sampler.is_eog(token) {
                hit_eog = true;
                break;
            }
            if let Some(g) = self.grammar.as_mut() {
                g.on_committed(token)?;
            }
            let chunk = stream.step(token)?;
            self.think.on_emitted(chunk.as_deref());
            let text = chunk.unwrap_or_default();
            on_event(section_event(
                token,
                text,
                &mut in_thinking,
                &mut thinking_tokens,
            ));
            decoded += 1;
            // A completed grammar ends the reply: every remaining legal token is
            // an EOG, and drawing one costs a forward this break skips. Reported
            // as a natural stop — the value is complete, not truncated.
            if self.grammar.as_ref().is_some_and(|g| g.is_done()) {
                hit_eog = true;
                break;
            }
            if decoded == max_new {
                break; // hit the cap: no need to run another forward
            }
            let input = Tensor::new(&[token], &device)?;
            logits = self.model.forward(&input, pos)?;
            pos += 1;
        }
        device.synchronize()?;
        Ok(DecodeOutcome {
            tokens_out: decoded,
            thinking_tokens,
            hit_eog,
            cancelled,
            decode_secs: decode_start.elapsed().as_secs_f64(),
            spec: None,
        })
    }

    /// Speculative decode with the attached DFlash drafter. A mirror of the
    /// fork's `common_speculative_impl_draft_dflash` (speculative.cpp:902-1219)
    /// wired into this engine's chunked prefill + streaming decode.
    ///
    /// Correctness property: with the same seed this consumes the main sampler's
    /// RNG exactly once per committed token, in the same sequence order plain
    /// decode would, so `--draft` output matches spec-off output token-for-token
    /// except where the batched-verify forward's numerics (QMatMul lm_head over a
    /// span, vs the decode mat-vec bypass at seq==1) flip a near-tie.
    fn generate_spec(
        &mut self,
        prompt: &str,
        content_ranges: &[Range<usize>],
        max_tokens: usize,
        on_text: &mut dyn FnMut(&str),
        should_stop: &mut dyn FnMut() -> bool,
    ) -> Result<GenStats> {
        // This loop's sample sites do not consult the grammar (the serve paths
        // — `decode_tokens` / `decode_loop_spec` — do). Refusing beats decoding
        // a constrained request unconstrained; wiring this loop is the ledger
        // item under "CLI --json-schema".
        ensure!(
            self.grammar.is_none(),
            "generate_spec does not apply a grammar constraint: detach the drafter or clear the \
             grammar"
        );
        check_think_budget(&self.think, self.min_think, max_tokens)?;
        let ban_floor = self.ban_ids(true);
        let ban_plain = self.ban_ids(false);
        let min_think = self.min_think;
        // Any logits a previous `prefill_tokens` left behind describe a cache
        // that no longer exists (this path resets it below).
        self.last_logits = None;
        // Destructure into per-field borrows so the drafter can stay borrowed
        // across target-model / sampler / tokenizer calls (distinct fields).
        let Self {
            model,
            tokenizer,
            sampler,
            drafter,
            spec_params,
            think,
            ..
        } = self;
        think.reset();
        let drafter = drafter.as_mut().expect("generate_spec requires a drafter");
        model.reset_cache()?;
        let device = model.device().clone();
        drafter.reset();
        let params = spec_params.clone();

        let tokens = tokenizer.encode_prompt(prompt, content_ranges)?;
        ensure!(!tokens.is_empty(), "prompt encoded to zero tokens");

        let max_ctx = model.max_ctx();
        ensure!(
            tokens.len() + max_tokens <= max_ctx,
            "prompt ({} tokens) + max_tokens ({max_tokens}) exceeds max_ctx ({max_ctx}); \
             raise --max-ctx or lower -n",
            tokens.len()
        );
        // Same floor-vs-budget guard as the plain path (see generate()).
        ensure!(
            min_think == 0 || min_think < max_tokens,
            "min_think ({min_think}) must be below max_tokens ({max_tokens}) or the think \
             block can never close",
        );

        // Draft-block width is bounded by the trained block size (id_last plus
        // block_size-1 masks) and the drafter's mask token.
        let block_size = drafter.config().block_size;
        let mask_token_id = drafter.config().mask_token_id;
        let draft_max = params.draft_max.min(block_size.saturating_sub(1));
        // The drafter's context is sized independently of the target's (and
        // smaller — it is the drafting depth limit): injection caps at it, and
        // rounds past it run plain.
        let draft_ctx = drafter.max_ctx();

        // Benchmark warm-up (XWEN_BENCH): page weights + compile pipelines on a
        // throwaway prefill (target and drafter both), then reset. Never affects
        // output. Mirrors the plain-decode path so spec timings are steady-state.
        if std::env::var_os("XWEN_BENCH").is_some() {
            let mut warm_pos = 0usize;
            for chunk in tokens.chunks(PREFILL_CHUNK) {
                let input = Tensor::new(chunk, &device)?;
                let _ = model.forward(&input, warm_pos)?;
                let taps = model.take_spec_taps();
                inject_capped(drafter, &taps, warm_pos, chunk.len())?;
                warm_pos += chunk.len();
            }
            device.synchronize()?;
            model.reset_cache()?;
            drafter.reset();
        }

        // ---- Prefill: feed the prompt in chunks. After each chunk, fuse the
        // target taps through the drafter's encoder and inject them into the
        // drafter's KV cache at the same absolute positions.
        let prefill_start = Instant::now();
        let mut cancelled = false;
        let mut pos = 0usize;
        let mut logits = None;
        for chunk in tokens.chunks(PREFILL_CHUNK) {
            if should_stop() {
                cancelled = true;
                break;
            }
            let input = Tensor::new(chunk, &device)?;
            logits = Some(model.forward(&input, pos)?);
            let taps = model.take_spec_taps();
            inject_capped(drafter, &taps, pos, chunk.len())?;
            pos += chunk.len();
        }
        device.synchronize()?;
        let prefill_secs = prefill_start.elapsed().as_secs_f64();
        let logits = match logits {
            Some(logits) if !cancelled => logits,
            _ => {
                return Ok(GenStats {
                    prefill_tokens: pos,
                    prefill_secs,
                    cancelled: true,
                    spec: Some(SpecStats::default()),
                    ..GenStats::default()
                });
            }
        };
        debug_assert_eq!(drafter.committed_len(), tokens.len().min(draft_ctx));

        // ---- Decode. `id_last` is the last sampled-but-not-yet-verified token,
        // living at absolute position `n_past`; the drafter's committed length
        // tracks `n_past` at every round boundary.
        let decode_start = Instant::now();
        let mut stream = tokenizer.decode_stream();
        let mut stats = SpecStats::default();
        let mut n_past = tokens.len();
        let mut decoded = 0usize;

        // Wall-clock auto-pause: never let speculation lose to plain decode on
        // low-acceptance text. Fed the measured cost of each round below; the
        // XWEN_BENCH warm-up prefill happened before this and is not a sample.
        let mut pause = PauseController::new(params.pause_margin as f64);

        // A zero budget must not draw from the sampler at all (plain decode's
        // first sample lives inside its `while decoded < max_tokens` loop).
        if max_tokens == 0 {
            return Ok(GenStats {
                prefill_tokens: pos,
                prefill_secs,
                spec: Some(stats),
                ..GenStats::default()
            });
        }

        // First token: sampled from the last prefill logits, exactly as plain
        // decode does. It is emitted here; every later token is emitted as a
        // committed token of a verify round. Emitted index 0, so the forced-
        // reasoning ban applies whenever the floor is nonzero.
        let mut id_last = {
            let banned = if min_think > 0 {
                &ban_floor
            } else {
                &ban_plain
            };
            let tc = think.control_for();
            let ctl = SampleControl {
                allowed: None,
                banned,
                bias: tc.bias,
                pull: tc.pull,
                force: tc.force,
            };
            let token = sampler.sample_controlled(&logits, &ctl)?;
            think.on_committed(0, token);
            token
        };
        let first_eog = sampler.is_eog(id_last);
        if !first_eog {
            let chunk = stream.step(id_last)?;
            think.on_emitted(chunk.as_deref());
            if let Some(text) = &chunk {
                on_text(text);
            }
            decoded = 1;
        }

        if !first_eog {
            'rounds: while decoded < max_tokens {
                if should_stop() {
                    cancelled = true;
                    break;
                }

                // The controller decides this round's kind before any GPU work;
                // paused rounds skip drafting and run the plain fallback below.
                // A round the thinking budget wants serial is drafted as plain
                // whatever the controller asked for, and is kept out of its
                // accounting entirely: its cost says nothing about whether
                // speculation is paying off.
                //
                // Speculation also needs the drafter's cache to describe exactly
                // the tokens the target's does, and a drafter cache position to
                // write into. Its context is sized independently (and smaller —
                // the drafting depth limit), so a long generation eventually runs
                // past it; from there every round is plain, and the taps are
                // drained and dropped rather than injected.
                let spec_live =
                    drafter_span_rows(drafter.committed_len(), draft_ctx, n_past, 1) == 1;
                let round_kind = pause.next_kind();
                // Look a full draft span ahead: a round that STARTS below the
                // threshold can still have rows above it, and those rows would
                // read a boundary flag from before the round began.
                let serial = think.needs_serial_rounds(draft_max);
                let want_draft = spec_live
                    && !serial
                    && matches!(round_kind, RoundKind::Spec | RoundKind::Probe);
                // A round the thinking budget wants serial, or one the drafter
                // cannot cover, says nothing about whether speculation pays off —
                // keep both out of the controller's accounting.
                let accounted = spec_live && !serial;
                let round_start = Instant::now();

                // ---- Draft: run the drafter over [id_last, MASK * n_draft] and
                // greedily read one token per masked position.
                let n_draft = draft_max
                    .min(max_ctx.saturating_sub(n_past + 1))
                    .min(draft_ctx.saturating_sub(n_past + 1));
                let mut drafts: Vec<u32> = Vec::new();
                if want_draft && n_draft > 0 {
                    let mut noise_ids = Vec::with_capacity(n_draft + 1);
                    noise_ids.push(id_last);
                    noise_ids.extend(std::iter::repeat(mask_token_id).take(n_draft));
                    let noise_embd = model.embed_ids(&noise_ids)?; // [n_draft+1, n_embd]
                    let hidden = drafter.draft_forward(&noise_embd, n_past)?; // [n_draft+1, n_embd]
                    // Rows 1..=n_draft are the masked positions' drafts (row 0 is
                    // id_last's own position, which the target verifies). Narrowing
                    // to those rows before lm_head is a row-OFFSET view; QLinear's
                    // forward now materializes such views before the quantized
                    // matmul (gguf.rs), so lm_head reads the intended rows rather
                    // than the base offset — the Metal QMatMul offset trap.
                    let hidden_masked = hidden.narrow(0, 1, n_draft)?;
                    let dlogits = model.lm_head(&hidden_masked)?; // [n_draft, vocab]
                    // Per-row argmax id + softmax prob of that argmax, reduced
                    // on-GPU so only the tiny [n_draft] id/prob arrays cross to
                    // the CPU — not the full [n_draft, vocab] logits. The p_min
                    // walk below is then a cheap CPU scan, identical in semantics
                    // to the old per-row full-vocab argmax_softmax.
                    let (ids, probs) = draft_argmax_probs(&dlogits)?;
                    for (&tok, &p) in ids.iter().zip(&probs) {
                        if p < params.draft_p_min {
                            break;
                        }
                        drafts.push(tok);
                    }
                    if drafts.len() < params.draft_min {
                        drafts.clear();
                    }
                }

                // Split the round timer at the draft/commit boundary. The draft
                // phase just ended at the `draft_argmax_probs` readback (a CPU
                // sync); the commit phase below ends at its own sampler readback,
                // so neither split adds a device sync. A round that ran the
                // drafter but kept no drafts still charges its draft time here.
                let draft_ms = round_start.elapsed().as_secs_f64() * 1000.0;

                // Committed tokens for this round and the last sampled token.
                let committed: Vec<u32>;
                if drafts.is_empty() {
                    // No usable draft: a plain single-token decode step. This is
                    // the same target forward + sampler draw plain decode runs, so
                    // it stays byte-identical when drafting yields nothing.
                    let input = Tensor::new(&[id_last], &device)?;
                    let step_logits = model.forward(&input, n_past)?;
                    let taps = model.take_spec_taps(); // each [1, n_embd]
                    if spec_live {
                        let fused = drafter.encode(&taps)?;
                        drafter.inject(&fused, n_past)?;
                    }
                    let banned = if decoded < min_think {
                        &ban_floor
                    } else {
                        &ban_plain
                    };
                    let tc = think.control_for();
                    let ctl = SampleControl {
                        allowed: None,
                        banned,
                        bias: tc.bias,
                        pull: tc.pull,
                        force: tc.force,
                    };
                    let s = sampler.sample_controlled(&step_logits, &ctl)?;
                    think.on_committed(decoded, s);
                    n_past += 1;
                    stats.rounds += 1;
                    committed = vec![s];
                } else {
                    // ---- Verify: forward [id_last, drafts...] in one batch, snapshot
                    // the KV first so a partial accept can roll the span back.
                    let span = 1 + drafts.len();
                    let ckpt = model.kv_checkpoint(span)?;
                    let mut verify_ids = Vec::with_capacity(span);
                    verify_ids.push(id_last);
                    verify_ids.extend_from_slice(&drafts);
                    let vinput = Tensor::new(verify_ids.as_slice(), &device)?;
                    let logits_all = model.forward_all_logits(&vinput, n_past)?; // [span, vocab]
                    let taps = model.take_spec_taps(); // each [span, n_embd]
                    let logits_cpu = logits_all.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;

                    // Sample the target token per verify row in order (RNG advances
                    // once per committed token, matching plain decode), accepting
                    // matching drafts until the first mismatch, the bonus token, an
                    // EOG, or the max_tokens budget — never drawing past any of them.
                    // Row i commits emitted index `decoded + i`, so the forced-
                    // reasoning ban applies per row; a draft proposing a banned id
                    // simply mismatches the masked sample and is rejected.
                    let (m, accepted) = accept_drafts(&drafts, max_tokens - decoded, |i| {
                        let row = logits_cpu.narrow(0, i, 1)?;
                        let t = decoded + i;
                        let banned = if t < min_think {
                            &ban_floor
                        } else {
                            &ban_plain
                        };
                        let tc = think.control_for();
                        let ctl = SampleControl {
                            allowed: None,
                            banned,
                            bias: tc.bias,
                            pull: tc.pull,
                            force: tc.force,
                        };
                        let s = sampler.sample_controlled(&row, &ctl)?;
                        think.on_committed(t, s);
                        Ok((s, sampler.is_eog(s)))
                    })?;

                    // Keep exactly the verify positions backing the committed
                    // tokens: every emitted token except the last is forwarded,
                    // the exit state plain decode leaves (an EOG or budget-capped
                    // final token is never retained in the caches).
                    let keep = accepted.len();
                    model.kv_rollback(&ckpt, keep)?;
                    let prefix: Vec<Tensor> = taps
                        .iter()
                        .map(|t| t.narrow(0, 0, keep))
                        .collect::<candle_core::Result<_>>()?;
                    let fused = drafter.encode(&prefix)?;
                    drafter.inject(&fused, n_past)?;

                    n_past += keep;
                    stats.rounds += 1;
                    stats.drafted += drafts.len();
                    stats.accepted += m;
                    committed = accepted;
                }

                // Feed the round's wall-clock cost to the pause controller. A
                // round's kv_rollback + encode + inject are queued but not yet
                // synced at this point, so without a barrier ~5-15ms of GPU tail
                // would bleed into the NEXT round's window and skew whichever EMA
                // that round feeds. One synchronize per round trims that CPU
                // run-ahead; the GPU queue is in-order so total GPU time is
                // unchanged, and the cost is negligible at the ~200ms round
                // cadence. A round that requested a draft but produced none ran
                // the plain fallback (`ran_spec == false`): its target-forward
                // time is a plain sample (drafter time excluded via draft_ms) and
                // its whole cost is a speculative sample, so wasted drafting
                // stays visible to the pause decision.
                let ran_spec = !drafts.is_empty();
                device.synchronize()?;
                let round_ms = round_start.elapsed().as_secs_f64() * 1000.0;
                stats.draft_ms += draft_ms;
                stats.verify_ms += round_ms - draft_ms;
                stats.verify_positions += 1 + drafts.len();
                if accounted {
                    pause.record(round_kind, round_ms, draft_ms, committed.len(), ran_spec);
                    // Count every round that ran plain because of the pause: the
                    // paused-plain rounds, and any probe round whose draft came up
                    // empty (it ran the plain fallback). Real drafting probes are
                    // excluded — they speculated.
                    if round_kind == RoundKind::PausedPlain
                        || (round_kind == RoundKind::Probe && !ran_spec)
                    {
                        stats.paused_rounds += 1;
                    }
                }

                id_last = *committed.last().expect("a round always commits >= 1 token");

                // Emit the committed tokens, stopping at the first EOG or the cap.
                for &tok in &committed {
                    if sampler.is_eog(tok) {
                        break 'rounds;
                    }
                    let chunk = stream.step(tok)?;
                    think.on_emitted(chunk.as_deref());
                    if let Some(text) = &chunk {
                        on_text(text);
                    }
                    decoded += 1;
                    if decoded >= max_tokens {
                        break 'rounds;
                    }
                }
            }
        }

        device.synchronize()?;
        let decode_secs = decode_start.elapsed().as_secs_f64();

        Ok(GenStats {
            prefill_tokens: tokens.len(),
            prefill_secs,
            decode_tokens: decoded,
            decode_secs,
            cancelled,
            spec: Some(stats),
            think: think.enabled().then(|| think.stats(decoded)),
        })
    }
}

/// How many rows of a `len`-token span starting at absolute position `pos` a
/// drafter whose cache holds `committed` tokens and has `draft_ctx` slots can be
/// fed: the whole span, a leading prefix of it when the span straddles the end of
/// the drafter's context, or none at all.
///
/// The exact-position condition carries weight: injection is positional and
/// appends only at the committed length, so a drafter that missed a span — a
/// rewind it could not follow, a conversation paged in without drafter rows —
/// must not be handed the next one, and the taps for that span are dropped
/// instead.
///
/// Taking a PREFIX rather than skipping a straddling span matters because the
/// drafter's context is sized independently of the target's (its f32 planes cost
/// 40-48 KiB/token depending on the sidecar) and a prefill chunk rarely lands on
/// that boundary. Filling the
/// drafter right up to its capacity is what keeps the widest set of later rewind
/// points able to re-enable speculation — a rewind can only do so if it lands
/// exactly on the committed length.
fn drafter_span_rows(committed: usize, draft_ctx: usize, pos: usize, len: usize) -> usize {
    if pos != committed {
        return 0;
    }
    len.min(draft_ctx.saturating_sub(pos))
}

/// Fuse and inject one prefill chunk's taps into the drafter, capped at its
/// capacity (the drafting depth limit): a chunk that straddles the boundary is
/// injected up to it, so the drafter reaches its full capacity rather than
/// stopping a chunk short of it; one wholly past it is dropped. The caller has
/// already drained the taps from the model either way.
fn inject_capped(
    drafter: &mut DflashDrafter,
    taps: &[Tensor],
    pos: usize,
    len: usize,
) -> Result<()> {
    let rows = drafter_span_rows(drafter.committed_len(), drafter.max_ctx(), pos, len);
    if rows == 0 {
        return Ok(());
    }
    let fused = if rows == len {
        drafter.encode(taps)?
    } else {
        let prefix: Vec<Tensor> = taps
            .iter()
            .map(|t| t.narrow(0, 0, rows))
            .collect::<candle_core::Result<_>>()?;
        drafter.encode(&prefix)?
    };
    drafter.inject(&fused, pos)
}

/// Verify-span positions a round keeps in the KV cache after emitting `emitted` of
/// its `committed` tokens: the anchor row (the previous round's last token,
/// emitted then) plus every token emitted here except the last, which plain decode
/// would not have forwarded either.
///
/// The clamp is what makes a mid-span exit safe. A round that emitted only part of
/// its span — a cancel, or the `max_new` cap — must not leave the rest in the
/// cache: the server reconciles its token history against `cache_len`, so cached
/// positions the callback never reported would be reused as context for tokens
/// nobody has.
fn verify_keep(committed: usize, emitted: usize) -> usize {
    committed.min(emitted + 1)
}

/// Committed tokens a round RETAINED, as opposed to merely drew: everything the
/// emit reported, plus a stopping EOG — plain decode commits that one too, as the
/// reason the reply ended.
///
/// Tokens past a cancel are excluded. The rollback discards their positions and
/// the callback never saw them, so counting them would credit speculation with
/// output nobody received: the acceptance rate and the pause controller's
/// tokens-per-round both describe delivered tokens. On every path but a mid-span
/// cancel this is the whole round.
fn retained_commits(committed: usize, emitted: usize, hit_eog: bool) -> usize {
    if hit_eog { committed } else { emitted }
}

/// Tag one committed token with the section of the reply it belongs to, advancing
/// the caller's reasoning bookkeeping. `text` is what the streaming decoder
/// finalized for this token (empty when it withheld it).
///
/// `in_thinking` starts true when the prompt ended inside a `<think>` block and
/// flips off on the `</think>` token, which is the LAST `ThinkingTok` and carries
/// the marker stripped from its text: a decode step finalizes text only through
/// the token just fed, so `</think>` is that chunk's tail — what precedes it is
/// the last of the reasoning and nothing follows it. `thinking_tokens` counts
/// every token inside the block, including the closing one.
///
/// Shared by the plain and speculative decode loops so the reasoning/answer split
/// has exactly one implementation.
fn section_event(
    token: u32,
    text: String,
    in_thinking: &mut bool,
    thinking_tokens: &mut usize,
) -> GenEvent {
    if !*in_thinking {
        return GenEvent::TextTok { id: token, text };
    }
    *thinking_tokens += 1;
    if token != LagunaTokenizer::THINK_CLOSE {
        return GenEvent::ThinkingTok { id: token, text };
    }
    *in_thinking = false;
    let reasoning = text
        .split_once("</think>")
        .map_or(text.as_str(), |(before, _)| before);
    GenEvent::ThinkingTok {
        id: token,
        text: reasoning.to_string(),
    }
}

/// Walk a verify block's rows in order, sampling the target's token for each and
/// accepting matching drafts until the first mismatch (or after the bonus token
/// when every draft matched). The walk also stops WITHOUT drawing when `budget`
/// committed tokens have been collected, and right after committing an EOG —
/// plain decode would never draw past either point, and the RNG must advance
/// exactly once per committed token in sequence order (the property that keeps
/// spec-on output equal to spec-off under the same seed).
///
/// Returns `(m, committed)` where `m` counts the committed tokens that were
/// accepted drafts (for stats) and `committed` is every sampled token, in
/// order. `committed.len()` is also the number of verify-span positions the
/// caller must keep in the KV caches: uniformly, every emitted token except
/// the last has been forwarded — the same exit state plain decode leaves.
///
/// `sample_row(i)` draws the target token for verify row `i` with the main
/// sampler and reports whether it is an EOG; it is called for `i = 0, 1, ...`
/// up to and INCLUDING the stopping row and never beyond. `budget` must
/// be >= 1.
fn accept_drafts(
    drafts: &[u32],
    budget: usize,
    mut sample_row: impl FnMut(usize) -> Result<(u32, bool)>,
) -> Result<(usize, Vec<u32>)> {
    debug_assert!(budget >= 1, "accept_drafts requires a nonzero budget");
    let span = drafts.len() + 1;
    let mut committed = Vec::with_capacity(span.min(budget));
    let mut m = 0usize;
    for i in 0..span {
        if committed.len() >= budget {
            break;
        }
        let (s, eog) = sample_row(i)?;
        let matched = i < drafts.len() && s == drafts[i];
        committed.push(s);
        if matched {
            m += 1;
        }
        // A matched non-EOG draft continues the walk; anything else (mismatch,
        // the bonus row past the last draft, or an EOG) commits and stops.
        if !matched || eog {
            break;
        }
    }
    Ok((m, committed))
}

/// For each row of `logits` `[n, vocab]`, the argmax token id and the full-vocab
/// softmax probability of that argmax (`1 / Σ_j exp(logit_j - max)`). The reduce
/// runs on whatever device `logits` is on (the GPU in production) and only the
/// tiny `[n]` id and prob vectors are read back — not the `[n, vocab]` logits.
/// On an exact logit tie candle's argmax returns the FIRST maximal index; the
/// probability is derived from the max VALUE, so it is unaffected by which index
/// wins.
fn draft_argmax_probs(logits: &Tensor) -> Result<(Vec<u32>, Vec<f32>)> {
    let logits = logits.to_dtype(DType::F32)?;
    let max = logits.max_keepdim(D::Minus1)?; // [n, 1]
    // prob(argmax) = exp(max - logsumexp) = 1 / Σ_j exp(logit_j - max).
    let probs = logits
        .broadcast_sub(&max)?
        .exp()?
        .sum_keepdim(D::Minus1)?
        .recip()?; // [n, 1]
    let ids = logits.argmax_keepdim(D::Minus1)?; // [n, 1]
    let ids = ids
        .flatten_all()?
        .to_dtype(DType::U32)?
        .to_device(&Device::Cpu)?
        .to_vec1::<u32>()?;
    let probs = probs
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    Ok((ids, probs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// EOG marker used by the tests: any token >= 900 "ends generation".
    fn test_eog(t: u32) -> bool {
        t >= 900
    }

    /// Drive `accept_drafts` with a fixed list of "target samples", tracking how
    /// many rows it read so we can assert it never samples past the stop point
    /// (the invariant that keeps RNG consumption aligned with plain decode).
    fn run_budget(drafts: &[u32], samples: &[u32], budget: usize) -> (usize, Vec<u32>, usize) {
        let calls = Cell::new(0usize);
        let (m, committed) = accept_drafts(drafts, budget, |i| {
            calls.set(calls.get() + 1);
            Ok((samples[i], test_eog(samples[i])))
        })
        .unwrap();
        (m, committed, calls.get())
    }

    fn run(drafts: &[u32], samples: &[u32]) -> (usize, Vec<u32>, usize) {
        run_budget(drafts, samples, usize::MAX)
    }

    // Every draft matches, so the target's bonus token past the last draft is a
    // free extra commit: m == drafts.len() and one more token than drafts.
    #[test]
    fn all_drafts_match_plus_bonus() {
        let (m, committed, calls) = run(&[10, 20, 30], &[10, 20, 30, 40]);
        assert_eq!(m, 3);
        assert_eq!(committed, vec![10, 20, 30, 40]);
        // Sampled row 0 (id_last's follow), rows 1..3 (drafts), row 3 (bonus).
        assert_eq!(calls, 4);
    }

    // The very first sampled token disagrees with draft 0: nothing is accepted,
    // and only that single correction is committed and sampled.
    #[test]
    fn first_token_mismatch() {
        let (m, committed, calls) = run(&[10, 20, 30], &[99, 20, 30, 40]);
        assert_eq!(m, 0);
        assert_eq!(committed, vec![99]);
        assert_eq!(calls, 1);
    }

    // A mismatch in the middle: drafts before it are accepted, the correction is
    // committed, and no rows past the mismatch are sampled.
    #[test]
    fn mid_block_mismatch() {
        let (m, committed, calls) = run(&[10, 20, 30], &[10, 20, 77, 40]);
        assert_eq!(m, 2);
        assert_eq!(committed, vec![10, 20, 77]);
        assert_eq!(calls, 3);
    }

    // Zero drafts (span 1): exactly one token is sampled and committed — this is
    // the plain single-token decode step expressed through the same walk.
    #[test]
    fn zero_drafts_single_commit() {
        let (m, committed, calls) = run(&[], &[42]);
        assert_eq!(m, 0);
        assert_eq!(committed, vec![42]);
        assert_eq!(calls, 1);
    }

    // A single draft that matches yields the draft plus its bonus token.
    #[test]
    fn single_draft_match() {
        let (m, committed, calls) = run(&[10], &[10, 55]);
        assert_eq!(m, 1);
        assert_eq!(committed, vec![10, 55]);
        assert_eq!(calls, 2);
    }

    // An accepted draft that is an EOG stops the walk immediately: no bonus row
    // is sampled (plain decode never draws past an EOG), and the EOG is the
    // last committed token so the caller's keep-count excludes it from the KV.
    #[test]
    fn accepted_eog_draft_stops_without_bonus() {
        let (m, committed, calls) = run(&[10, 900, 30], &[10, 900, 30, 40]);
        assert_eq!(m, 2);
        assert_eq!(committed, vec![10, 900]);
        assert_eq!(calls, 2);
    }

    // An EOG arriving as the correction (mismatch) also stops the walk — same
    // stopping row a mismatch alone would produce, no extra draws.
    #[test]
    fn eog_correction_stops() {
        let (m, committed, calls) = run(&[10, 20], &[10, 901, 30]);
        assert_eq!(m, 1);
        assert_eq!(committed, vec![10, 901]);
        assert_eq!(calls, 2);
    }

    // The budget caps commits BEFORE the draw: with 2 tokens of budget and every
    // draft matching, only rows 0 and 1 are ever sampled.
    #[test]
    fn budget_stops_before_sampling() {
        let (m, committed, calls) = run_budget(&[10, 20, 30], &[10, 20, 30, 40], 2);
        assert_eq!(m, 2);
        assert_eq!(committed, vec![10, 20]);
        assert_eq!(calls, 2);
    }

    // Budget of 1 commits exactly one token regardless of how many drafts match.
    #[test]
    fn budget_one_single_commit() {
        let (m, committed, calls) = run_budget(&[10, 20, 30], &[10, 20, 30, 40], 1);
        assert_eq!(m, 1);
        assert_eq!(committed, vec![10]);
        assert_eq!(calls, 1);
    }

    // ---- The speculative round loop's two pure decisions: which spans the
    // drafter can be fed, and how much of a verified span survives the emit.

    // A drafter is fed a span only when its cache ends exactly where the span
    // starts; how much of it it takes is bounded by its own context. A rewind it
    // could not follow and a conversation paged in without drafter rows both read
    // as zero rows, whatever the context has left.
    #[test]
    fn drafter_span_rows_fills_the_context_and_no_further() {
        // In sync and wholly inside the context.
        assert_eq!(drafter_span_rows(100, 512, 100, 8), 8);
        // Exactly filling the context.
        assert_eq!(drafter_span_rows(504, 512, 504, 8), 8);
        // Straddling its end: the prefix that fits is injected, so the drafter
        // reaches capacity instead of stopping a whole chunk short of it.
        assert_eq!(drafter_span_rows(508, 512, 508, 8), 4);
        // Full: nothing more fits.
        assert_eq!(drafter_span_rows(512, 512, 512, 8), 0);
        // Out of sync — behind the target (the drafter missed a span) or ahead of
        // it (the target was rewound without the drafter) — takes nothing even
        // with room to spare.
        assert_eq!(drafter_span_rows(0, 512, 100, 8), 0);
        assert_eq!(drafter_span_rows(120, 512, 100, 8), 0);
        // The single-position form the decode loop asks about.
        assert_eq!(drafter_span_rows(511, 512, 511, 1), 1);
        assert_eq!(drafter_span_rows(512, 512, 512, 1), 0);
    }

    // What a round keeps in the KV cache is decided by what it EMITTED, so no exit
    // leaves positions behind for tokens the caller never received.
    #[test]
    fn verify_keep_never_outruns_the_emitted_tokens() {
        // Every committed token emitted: keep the whole span, the last emitted
        // token deliberately unforwarded (plain decode's exit state).
        assert_eq!(verify_keep(5, 5), 5);
        // An EOG at the last row is committed but not emitted, and the span still
        // keeps every forwarded position.
        assert_eq!(verify_keep(5, 4), 5);
        // An EOG at row 0 leaves only the anchor row.
        assert_eq!(verify_keep(1, 0), 1);
        // A cancel or the max_new cap partway through a 12-token span trims the
        // rest: 3 emitted keeps the anchor plus the first two.
        assert_eq!(verify_keep(12, 3), 4);
        assert_eq!(verify_keep(12, 0), 1);
    }

    // The stats and the pause controller describe DELIVERED tokens, so a round
    // credits itself only with what it retained. Every path but a mid-span cancel
    // retains the whole round, which is what keeps the acceptance rate comparable
    // to the fork's.
    #[test]
    fn retained_commits_excludes_only_the_rolled_back_tail() {
        // Every committed token emitted.
        assert_eq!(retained_commits(5, 5, false), 5);
        // A stopping EOG is committed but not emitted, and still counts.
        assert_eq!(retained_commits(5, 4, true), 5);
        assert_eq!(retained_commits(1, 0, true), 1);
        // A cancel partway through a 12-token span drops the nine the rollback
        // discarded.
        assert_eq!(retained_commits(12, 3, false), 3);
    }

    // ---- The reasoning/answer split, shared by the plain and speculative loops.

    // `</think>` is the last ThinkingTok and its text stops at the marker; the
    // token itself counts toward the reasoning total, and everything after it is
    // answer text that does not.
    #[test]
    fn section_event_ends_reasoning_at_the_marker() {
        let mut in_thinking = true;
        let mut thinking = 0usize;

        let event = section_event(7, "step one".into(), &mut in_thinking, &mut thinking);
        assert!(matches!(event, GenEvent::ThinkingTok { id: 7, .. }));
        assert_eq!(event.text(), "step one");
        assert!(in_thinking);
        assert_eq!(thinking, 1);

        let event = section_event(
            LagunaTokenizer::THINK_CLOSE,
            "done.</think>".into(),
            &mut in_thinking,
            &mut thinking,
        );
        assert!(matches!(event, GenEvent::ThinkingTok { .. }));
        assert_eq!(event.text(), "done.");
        assert!(!in_thinking);
        assert_eq!(thinking, 2);

        let event = section_event(9, "the answer".into(), &mut in_thinking, &mut thinking);
        assert!(matches!(event, GenEvent::TextTok { id: 9, .. }));
        assert_eq!(event.text(), "the answer");
        assert_eq!(thinking, 2);
    }

    // A run that never started in a `<think>` block reports every token as answer
    // text, `</think>` included — there is no block for it to close.
    #[test]
    fn section_event_outside_a_think_block_counts_nothing() {
        let mut in_thinking = false;
        let mut thinking = 0usize;
        let event = section_event(
            LagunaTokenizer::THINK_CLOSE,
            "</think>".into(),
            &mut in_thinking,
            &mut thinking,
        );
        assert!(matches!(event, GenEvent::TextTok { .. }));
        assert_eq!(event.text(), "</think>");
        assert_eq!(thinking, 0);
    }

    // A token the streaming decoder withheld (mid-UTF-8, or a partial tag prefix)
    // still counts and still carries its real id — only its text is empty.
    #[test]
    fn section_event_counts_withheld_tokens() {
        let mut in_thinking = true;
        let mut thinking = 0usize;
        let event = section_event(42, String::new(), &mut in_thinking, &mut thinking);
        assert_eq!(event.id(), 42);
        assert!(event.text().is_empty());
        assert_eq!(thinking, 1);
    }

    // ---- PauseController: a pure state machine, tested without any GPU by
    // feeding synthetic per-round costs. Bad economics = a speculative round
    // that costs more per committed token than a plain round; good economics =
    // the reverse.

    /// Drive the controller for `rounds`, faithfully following `next_kind` and
    /// feeding each round a synthetic cost: `spec_ms` over `committed` tokens
    /// for speculative rounds, `plain_ms` for plain rounds.
    fn drive(
        margin: f64,
        rounds: usize,
        spec_ms: f64,
        committed: usize,
        plain_ms: f64,
    ) -> PauseController {
        let mut c = PauseController::new(margin);
        for _ in 0..rounds {
            let kind = c.next_kind();
            match kind {
                RoundKind::Spec | RoundKind::Probe => c.record(kind, spec_ms, 0.0, committed, true),
                RoundKind::ForcedPlain | RoundKind::PausedPlain => {
                    c.record(kind, plain_ms, 0.0, 1, false)
                }
            }
        }
        c
    }

    // Sustained bad economics (100 ms/spec-tok vs 40 ms/plain): the controller
    // pauses once warm-up is met (the forced-plain cadence supplies the plain
    // samples on an all-speculative run).
    #[test]
    fn pauses_under_sustained_bad_economics() {
        let c = drive(1.0, 100, 100.0, 1, 40.0);
        assert!(
            matches!(c.state, PauseState::Paused { .. }),
            "state {:?}",
            c.state
        );
    }

    // Sustained good economics (10 ms/spec-tok — 30 ms over 3 committed — vs
    // 50 ms/plain): speculation always pays, so the controller never pauses.
    #[test]
    fn stays_active_under_good_economics() {
        let c = drive(1.0, 100, 30.0, 3, 50.0);
        assert_eq!(c.state, PauseState::Active);
    }

    // Each failed probe PAIR doubles the backoff, capped at BACKOFF_CAP. A pair
    // is two consecutive probe rounds; the first stashes, the second decides.
    #[test]
    fn failed_probe_pairs_back_off_to_cap() {
        let mut c = PauseController::new(1.0);
        // Warm and paused with speculation clearly losing (100 > 40): 100 ms
        // per round over 1 committed token = 100 ms/tok.
        c.ema_spec_round_ms = Some(100.0);
        c.ema_spec_committed = Some(1.0);
        c.ema_plain_ms = Some(40.0);
        c.spec_samples = PauseController::WARMUP_SPEC;
        c.plain_samples = PauseController::WARMUP_PLAIN;
        c.state = PauseState::Paused { backoff: 32 };

        let mut backoff = 32;
        for expected in [64, 128, 256, 256, 256] {
            // The pair's aggregate (200 ms / 2 committed = 100 ms/tok) stays
            // above the margin: no resume, backoff doubles. The first probe only
            // stashes — the backoff must not change until the pair completes.
            c.record(RoundKind::Probe, 100.0, 0.0, 1, true);
            assert_eq!(
                c.state,
                PauseState::Paused { backoff },
                "first probe must not decide"
            );
            c.record(RoundKind::Probe, 100.0, 0.0, 1, true);
            assert_eq!(c.state, PauseState::Paused { backoff: expected });
            backoff = expected;
        }
    }

    // A probe PAIR whose own aggregate rate falls under the margin resumes to
    // Active and clears the backoff (the next pause starts at BACKOFF_START).
    #[test]
    fn resume_resets_backoff() {
        let mut c = PauseController::new(1.0);
        // Warm and paused with a stale, inflated spec cost (200 ms/tok) that a
        // live probe pair will contradict.
        c.ema_spec_round_ms = Some(200.0);
        c.ema_spec_committed = Some(1.0);
        c.ema_plain_ms = Some(40.0);
        c.spec_samples = PauseController::WARMUP_SPEC;
        c.plain_samples = PauseController::WARMUP_PLAIN;
        c.state = PauseState::Paused { backoff: 128 };

        // A cheap probe pair (2 x 30 ms / 1 = 30 ms/tok <= 40) resumes on its
        // own aggregate despite the stale EMA still reading 200.
        c.record(RoundKind::Probe, 30.0, 0.0, 1, true);
        assert_eq!(
            c.state,
            PauseState::Paused { backoff: 128 },
            "first probe must not resume"
        );
        c.record(RoundKind::Probe, 30.0, 0.0, 1, true);
        assert_eq!(c.state, PauseState::Active);

        // Economics sour again: the fresh pause starts from BACKOFF_START, not
        // the old 128 — the backoff was reset on resume.
        c.record(RoundKind::Spec, 400.0, 0.0, 1, true);
        assert_eq!(
            c.state,
            PauseState::Paused {
                backoff: PauseController::BACKOFF_START
            }
        );
    }

    // margin 0 disables auto-pause: every round speculates and the state never
    // leaves Active however bad the economics get.
    #[test]
    fn margin_zero_never_pauses() {
        let mut c = PauseController::new(0.0);
        for _ in 0..100 {
            assert_eq!(c.next_kind(), RoundKind::Spec);
            c.record(RoundKind::Spec, 1000.0, 0.0, 1, true);
        }
        assert_eq!(c.state, PauseState::Active);
    }

    // Warm-up: no pause until both sample counts are met, even under clearly
    // bad economics. The speculative count is the last to cross here.
    #[test]
    fn warmup_spec_threshold_respected() {
        let mut c = PauseController::new(1.0);
        for _ in 0..PauseController::WARMUP_SPEC - 1 {
            c.record(RoundKind::Spec, 100.0, 0.0, 1, true);
        }
        for _ in 0..PauseController::WARMUP_PLAIN {
            c.record(RoundKind::ForcedPlain, 40.0, 0.0, 1, false);
        }
        assert_eq!(
            c.state,
            PauseState::Active,
            "must not pause below the spec warm-up"
        );
        // The final speculative sample crosses the threshold and triggers pause.
        c.record(RoundKind::Spec, 100.0, 0.0, 1, true);
        assert!(matches!(c.state, PauseState::Paused { .. }));
    }

    // The plain warm-up count gates pausing symmetrically: enough spec samples
    // but too few plain samples stays Active until the last plain sample lands.
    #[test]
    fn warmup_plain_threshold_respected() {
        let mut c = PauseController::new(1.0);
        for _ in 0..PauseController::WARMUP_SPEC {
            c.record(RoundKind::Spec, 100.0, 0.0, 1, true);
        }
        for _ in 0..PauseController::WARMUP_PLAIN - 1 {
            c.record(RoundKind::ForcedPlain, 40.0, 0.0, 1, false);
        }
        assert_eq!(
            c.state,
            PauseState::Active,
            "must not pause below the plain warm-up"
        );
        c.record(RoundKind::ForcedPlain, 40.0, 0.0, 1, false);
        assert!(matches!(c.state, PauseState::Paused { .. }));
    }

    // Regression for the mean-of-ratios misfire. Bimodal spec rounds — cheap
    // high-commit (200 ms / 11 tokens) alternating with expensive 1-commit
    // mismatch rounds (360 ms / 1 token) — aggregate to 560 ms / 12 = 46.7
    // ms/tok, under the plain comparator, so speculation wins and the controller
    // must never pause. The old single EMA of per-round ratios folded the
    // 360/1 = 360 ms/tok spikes and paused this exact shape (measured: 76 of 99
    // rounds on a 1.4x-winning code-gen run). The rate form (EMA(ms)/EMA(tok))
    // amortizes a 1-commit round the way throughput does and tracks the true
    // aggregate.
    //
    // The comparator is 60 ms rather than the aggregate-hugging 55 ms on
    // purpose: the quotient is phase-dependent and peaks near 55 right after a
    // 1-commit sample (that round transiently dominates both EMAs), so 60 leaves
    // the assertion unambiguous headroom while still proving speculation wins.
    #[test]
    fn rate_ema_stays_active_when_aggregate_wins() {
        let mut c = PauseController::new(1.0);
        c.record(RoundKind::ForcedPlain, 60.0, 0.0, 1, false);
        c.record(RoundKind::ForcedPlain, 60.0, 0.0, 1, false);
        for i in 0..200 {
            let (ms, committed) = if i % 2 == 0 { (200.0, 11) } else { (360.0, 1) };
            c.record(RoundKind::Spec, ms, 0.0, committed, true);
            assert_eq!(
                c.state,
                PauseState::Active,
                "paused at round {i} on a winning workload"
            );
        }
    }

    // Mirror: when the aggregate genuinely loses (every round 360 ms / 3
    // committed = 120 ms/tok vs 55 ms plain), the controller must pause once
    // warm — the rate form does not mask a real regression.
    #[test]
    fn rate_ema_pauses_when_aggregate_loses() {
        let mut c = PauseController::new(1.0);
        c.record(RoundKind::ForcedPlain, 55.0, 0.0, 1, false);
        c.record(RoundKind::ForcedPlain, 55.0, 0.0, 1, false);
        for _ in 0..PauseController::WARMUP_SPEC {
            c.record(RoundKind::Spec, 360.0, 0.0, 3, true);
        }
        assert!(matches!(c.state, PauseState::Paused { .. }));
    }

    // An empty-draft round ran the drafter (draft_ms) then a plain step; only
    // the plain-step portion (round_ms - draft_ms) feeds the plain comparator,
    // so the wasted drafter time never inflates it.
    #[test]
    fn empty_draft_round_excludes_drafter_cost_from_plain() {
        let mut c = PauseController::new(1.0);
        c.record(RoundKind::Spec, 90.0, 40.0, 1, false);
        assert_eq!(c.ema_plain_ms, Some(50.0));
    }

    // Poisoned-pause recovery: a transient bad patch (e.g. GPU contention from a
    // second process) leaves the spec EMAs grossly inflated and the controller
    // paused. When the transient passes, the FIRST good probe pair must resume
    // immediately on its own aggregate — not crawl back through a 256-token
    // backoff and an α=0.25 EMA that would take dozens of probes to recover — and
    // the spec cost must be reseeded to the pair's rate, dropping the poison.
    #[test]
    fn poisoned_pause_recovers_on_first_good_probe_pair() {
        let mut c = PauseController::new(1.0);
        c.ema_spec_round_ms = Some(4000.0); // 4000 ms/tok — grossly poisoned
        c.ema_spec_committed = Some(1.0);
        c.ema_plain_ms = Some(55.0);
        c.spec_samples = PauseController::WARMUP_SPEC;
        c.plain_samples = PauseController::WARMUP_PLAIN;
        c.state = PauseState::Paused {
            backoff: PauseController::BACKOFF_START,
        };

        // Plain rounds run until the backoff's worth of tokens elapse.
        while c.next_kind() == RoundKind::PausedPlain {
            c.record(RoundKind::PausedPlain, 55.0, 0.0, 1, false);
        }
        assert_eq!(c.next_kind(), RoundKind::Probe);

        // A good probe pair (2 x 220 ms / 8 committed = 27.5 ms/tok). The first
        // round only stashes; the second resumes despite the smoothed EMA still
        // reading thousands, and reseeds the quotient to the pair's rate.
        c.record(RoundKind::Probe, 220.0, 0.0, 8, true);
        assert_eq!(c.next_kind(), RoundKind::Probe, "still mid-pair");
        assert!(
            matches!(c.state, PauseState::Paused { .. }),
            "first probe must not resume"
        );
        c.record(RoundKind::Probe, 220.0, 0.0, 8, true);
        assert_eq!(c.state, PauseState::Active);
        assert!(
            (c.spec_cost().unwrap() - 27.5).abs() < 1e-9,
            "spec cost {} must be reseeded to the pair rate 27.5",
            c.spec_cost().unwrap(),
        );
    }

    // Mirror: a bad probe pair (2 x 360 ms / 1 committed = 360 ms/tok vs 55 ms
    // plain) must NOT resume and must double the backoff — eager recovery does
    // not mean gullible recovery.
    #[test]
    fn bad_probe_pair_stays_paused_and_doubles_backoff() {
        let mut c = PauseController::new(1.0);
        c.ema_spec_round_ms = Some(4000.0);
        c.ema_spec_committed = Some(1.0);
        c.ema_plain_ms = Some(55.0);
        c.spec_samples = PauseController::WARMUP_SPEC;
        c.plain_samples = PauseController::WARMUP_PLAIN;
        c.state = PauseState::Paused {
            backoff: PauseController::BACKOFF_START,
        };

        while c.next_kind() == RoundKind::PausedPlain {
            c.record(RoundKind::PausedPlain, 55.0, 0.0, 1, false);
        }
        c.record(RoundKind::Probe, 360.0, 0.0, 1, true);
        c.record(RoundKind::Probe, 360.0, 0.0, 1, true);
        assert_eq!(
            c.state,
            PauseState::Paused {
                backoff: PauseController::BACKOFF_START * 2
            }
        );
    }

    // Accelerated warm-up: on an all-drafting run the normal 32-round forced-
    // plain cadence would delay the second plain sample (and thus the first
    // possible pause) to ~round 64, so a short net-negative generation would
    // never pause. The tightened warm-up cadence (a forced plain every 4th
    // Active round until the plain warm-up is met) collects the plain samples in
    // time, so a uniformly-losing workload (360 ms/tok spec vs 55 ms plain)
    // pauses within ~15 rounds.
    #[test]
    fn accelerated_warmup_pauses_short_generation() {
        let mut c = PauseController::new(1.0);
        let mut paused_round = None;
        for round in 1..=15 {
            let kind = c.next_kind();
            let ran_spec = matches!(kind, RoundKind::Spec | RoundKind::Probe);
            let (ms, committed) = if ran_spec { (360.0, 1) } else { (55.0, 1) };
            c.record(kind, ms, 0.0, committed, ran_spec);
            if matches!(c.state, PauseState::Paused { .. }) {
                paused_round = Some(round);
                break;
            }
        }
        assert!(
            paused_round.is_some(),
            "must pause within 15 rounds via accelerated warm-up"
        );
    }

    // The regression this guards: empty-draft Spec rounds used to fold ONLY into
    // the plain ledger, so a run whose drafter never cleared `draft_p_min` (deep
    // context is the live case: 0 drafted over 127 rounds at 30k tokens, 750 ms
    // of drafter per round) accrued zero spec samples, never met `warm()`, and
    // never paused — paying the full drafter cost on every round forever. An
    // empty-draft Spec round is a speculative sample: the round chose to run the
    // drafter, and its whole cost must reach the pause decision.
    #[test]
    fn persistently_draftless_active_run_pauses() {
        let mut c = PauseController::new(1.0);
        let mut paused_round = None;
        for round in 1..=15 {
            let kind = c.next_kind();
            // The drafter runs on every Spec round and proposes nothing (750 ms
            // wasted), then the plain fallback commits one token (43 ms). Forced
            // plains are genuine plain steps.
            let (ms, draft_ms) = if kind == RoundKind::Spec {
                (793.0, 750.0)
            } else {
                (43.0, 0.0)
            };
            c.record(kind, ms, draft_ms, 1, false);
            if matches!(c.state, PauseState::Paused { .. }) {
                paused_round = Some(round);
                break;
            }
        }
        assert!(
            paused_round.is_some(),
            "an all-draftless run must pause within 15 rounds"
        );
    }

    // An empty-draft probe pair (both rounds ran the drafter but kept no draft)
    // collects no real sample: it must not double the backoff, resume, or refresh
    // the still-poisoned spec EMA — only real evidence decides. It must, however,
    // restart the backoff clock so the retry lands one backoff later; probing
    // again immediately would run the drafter every round of the pause.
    #[test]
    fn empty_draft_probe_pair_decides_nothing() {
        let mut c = PauseController::new(1.0);
        c.ema_spec_round_ms = Some(4000.0);
        c.ema_spec_committed = Some(1.0);
        c.ema_plain_ms = Some(55.0);
        c.spec_samples = PauseController::WARMUP_SPEC;
        c.plain_samples = PauseController::WARMUP_PLAIN;
        c.state = PauseState::Paused {
            backoff: PauseController::BACKOFF_START,
        };
        while c.next_kind() == RoundKind::PausedPlain {
            c.record(RoundKind::PausedPlain, 55.0, 0.0, 1, false);
        }
        // Both probe rounds draft empty (ran_spec = false): the drafter ran
        // (30 ms) then a plain step.
        c.record(RoundKind::Probe, 85.0, 30.0, 1, false);
        c.record(RoundKind::Probe, 85.0, 30.0, 1, false);
        assert_eq!(
            c.state,
            PauseState::Paused {
                backoff: PauseController::BACKOFF_START
            },
            "backoff undoubled by a draftless probe pair",
        );
        assert_eq!(
            c.spec_cost(),
            Some(4000.0),
            "spec EMA not refreshed by empty probes"
        );

        // The retry is BACKED OFF, not immediate: the next round is a paused
        // plain step, and a full backoff's worth of committed tokens must elapse
        // before the controller probes again.
        assert_eq!(
            c.next_kind(),
            RoundKind::PausedPlain,
            "draftless pair must not re-probe at once"
        );
        let mut plain_rounds = 0usize;
        while c.next_kind() == RoundKind::PausedPlain {
            c.record(RoundKind::PausedPlain, 55.0, 0.0, 1, false);
            plain_rounds += 1;
        }
        assert_eq!(
            plain_rounds,
            PauseController::BACKOFF_START,
            "retry must wait a full backoff"
        );
        assert_eq!(c.next_kind(), RoundKind::Probe);
    }

    // The regression this guards: before the zero-real path reset the backoff
    // clock, `tokens_since_probe` stayed above `backoff` forever, so a drafter
    // that never cleared `draft_p_min` made EVERY paused round a probe — running
    // the drafter, and paying its wall time, on a run that had already decided
    // speculation was not worth it. Drive four backoff periods of consecutive
    // draftless rounds; the probes must stay proportional to the periods, not to
    // the rounds.
    #[test]
    fn persistently_draftless_pause_stops_probing_every_round() {
        let mut c = PauseController::new(1.0);
        c.ema_spec_round_ms = Some(4000.0);
        c.ema_spec_committed = Some(1.0);
        c.ema_plain_ms = Some(55.0);
        c.spec_samples = PauseController::WARMUP_SPEC;
        c.plain_samples = PauseController::WARMUP_PLAIN;
        c.state = PauseState::Paused {
            backoff: PauseController::BACKOFF_START,
        };

        let periods = 4;
        let rounds = periods * PauseController::BACKOFF_START;
        let mut probes = 0usize;
        for _ in 0..rounds {
            let kind = c.next_kind();
            if kind == RoundKind::Probe {
                probes += 1;
            }
            // Every round comes up draftless: the drafter ran (30 ms) then the
            // plain fallback, so ran_spec is false whatever the controller asked.
            c.record(kind, 85.0, 30.0, 1, false);
        }
        // At most one PROBE_PAIR per backoff period (plus a partial pair if the
        // window ends mid-pair) — bounded by the periods, not by `rounds`. The
        // pre-fix behavior was `rounds - BACKOFF_START` probes.
        let ceiling = PauseController::PROBE_PAIR * (periods + 1);
        assert!(
            probes <= ceiling,
            "draftless pause ran {probes} probe rounds over {rounds} rounds (ceiling {ceiling}); \
             the backoff clock is not resetting",
        );
        assert!(
            matches!(c.state, PauseState::Paused { .. }),
            "must stay paused without evidence"
        );
    }

    // A probe pair with one empty round and one real round decides on the real
    // sample alone — the empty round contributes nothing but does not block
    // recovery; the real evidence resumes and reseeds the spec cost to itself.
    #[test]
    fn probe_pair_decides_on_single_real_round() {
        let mut c = PauseController::new(1.0);
        c.ema_spec_round_ms = Some(4000.0);
        c.ema_spec_committed = Some(1.0);
        c.ema_plain_ms = Some(55.0);
        c.spec_samples = PauseController::WARMUP_SPEC;
        c.plain_samples = PauseController::WARMUP_PLAIN;
        c.state = PauseState::Paused {
            backoff: PauseController::BACKOFF_START,
        };
        while c.next_kind() == RoundKind::PausedPlain {
            c.record(RoundKind::PausedPlain, 55.0, 0.0, 1, false);
        }
        // First probe drafts empty (no sample); the second is a good real round
        // (220 ms / 8 committed = 27.5 ms/tok <= 55 -> resume).
        c.record(RoundKind::Probe, 85.0, 30.0, 1, false);
        assert!(
            matches!(c.state, PauseState::Paused { .. }),
            "one round in: no decision yet"
        );
        c.record(RoundKind::Probe, 220.0, 0.0, 8, true);
        assert_eq!(c.state, PauseState::Active);
        assert!(
            (c.spec_cost().unwrap() - 27.5).abs() < 1e-9,
            "spec cost {} must be reseeded to the single real sample",
            c.spec_cost().unwrap(),
        );
    }

    // ---- ThinkBudget: a pure CPU state machine, driven here without a model.

    /// Synthetic ids standing in for the encoded injection strings, so the
    /// schedule can be tested without a tokenizer. `transition` ends in the real
    /// `</think>` id because the controller disarms on it.
    const WRAPUP: [u32; 2] = [700, 701];
    const TRANSITION: [u32; 3] = [800, 801, LagunaTokenizer::THINK_CLOSE];
    const WAIT: [u32; 2] = [11, 12];

    fn budget(n: usize) -> ThinkBudget {
        ThinkBudget {
            budget: n,
            armed: true,
            wait_ids: WAIT.to_vec(),
            wrapup: WRAPUP.to_vec(),
            transition: TRANSITION.to_vec(),
            ..ThinkBudget::off()
        }
    }

    /// Place the schedule at `clock` — the count of MODEL-chosen tokens so far —
    /// and read the control it hands out for the next draw.
    fn at(b: &mut ThinkBudget, clock: usize) -> ThinkControl<'_> {
        b.clock = clock;
        b.control_for()
    }

    /// Commit one model-chosen token, asserting the controller was not forcing
    /// one instead. Advances the schedule clock by one.
    fn commit_natural(b: &mut ThinkBudget, token: u32) {
        assert!(
            b.control_for().force.is_none(),
            "the controller forced a token here"
        );
        let t = b.clock;
        b.on_committed(t, token);
    }

    /// Commit the token the controller is forcing. The schedule clock must not
    /// move: an injection is not the model spending its budget.
    fn commit_forced(b: &mut ThinkBudget) -> u32 {
        let id = b.control_for().force.expect("expected a forced token");
        b.on_committed(b.clock, id);
        id
    }

    // A chunk ends at a boundary when its last non-whitespace character closes a
    // sentence, or when a whitespace-only chunk carries a paragraph break.
    #[test]
    fn boundary_predicate_accepts_sentence_ends() {
        // A newline ends a line whether or not punctuation precedes it, so it
        // must survive whitespace trimming.
        for text in [
            " answer.",
            " really?",
            " no!",
            " done.\n",
            " step 3\n",
            "\n",
            "\n\n",
            " end. ",
        ] {
            assert!(ends_at_boundary(text), "{text:?} should be a boundary");
        }
        for text in [" and", " so,", " 3.5", "", " ", " (see"] {
            assert!(!ends_at_boundary(text), "{text:?} should not be a boundary");
        }
    }

    // An injection starts on its own line: mid-line it brings a paragraph break
    // with it, and after a newline it does not add a blank one.
    #[test]
    fn injections_open_a_paragraph_only_when_needed() {
        const PARA: [u32; 1] = [900];
        let mut mid = ThinkBudget {
            paragraph: PARA.to_vec(),
            ..budget(100)
        };
        mid.on_emitted(Some(" that settles it."));
        assert_eq!(at(&mut mid, 91).force, Some(PARA[0]));
        assert_eq!(commit_forced(&mut mid), PARA[0]);
        assert_eq!(commit_forced(&mut mid), WRAPUP[0]);

        let mut fresh = ThinkBudget {
            paragraph: PARA.to_vec(),
            ..budget(100)
        };
        fresh.on_emitted(Some(" that settles it.\n"));
        assert_eq!(
            at(&mut fresh, 91).force,
            Some(WRAPUP[0]),
            "added a redundant break"
        );
    }

    // A token the streaming decoder withheld (mid-UTF-8-codepoint) is never a
    // boundary: injecting there would split a character in half.
    #[test]
    fn withheld_chunk_is_not_a_boundary() {
        let mut b = budget(100);
        b.on_emitted(Some("done."));
        assert!(b.at_boundary);
        b.on_emitted(None);
        assert!(!b.at_boundary);
    }

    // The schedule's stages sit at fixed fractions of the budget, and everything
    // below the first one is inert.
    #[test]
    fn phases_follow_the_budget_fractions() {
        let b = budget(100);
        for (t, want) in [
            (0, ThinkPhase::Free),
            (69, ThinkPhase::Free),
            (70, ThinkPhase::Damping),
            (79, ThinkPhase::Damping),
            (80, ThinkPhase::Pulling),
            (89, ThinkPhase::Pulling),
            (90, ThinkPhase::WrapUp),
            (95, ThinkPhase::WrapUp),
            (96, ThinkPhase::Pulling),
            (99, ThinkPhase::Pulling),
            (100, ThinkPhase::Grace),
            (500, ThinkPhase::Grace),
        ] {
            assert_eq!(b.phase(t), want, "at t = {t}");
        }
    }

    // Below the first threshold the controller touches nothing at all, so a
    // reply that finishes thinking early is bit-identical to one generated
    // without a budget.
    #[test]
    fn controller_is_inert_below_the_ramp() {
        let mut b = budget(100);
        let c = at(&mut b, 69);
        assert!(c.bias.is_empty());
        assert!(c.pull.is_none());
        assert!(c.force.is_none());
    }

    // The wait-token bias grows from zero at 70% to its full magnitude at 95%,
    // and holds there past the budget.
    #[test]
    fn wait_bias_ramps_and_saturates() {
        let mut b = budget(100);
        let mid = at(&mut b, 82).bias.to_vec();
        assert_eq!(mid.len(), WAIT.len());
        assert!(
            mid.iter()
                .all(|&(_, d)| d < -1.0 && d > ThinkBudget::BIAS_MAX)
        );
        // Saturated at 95%, and held there for the whole grace period.
        for t in [95, 100, 110] {
            let full = at(&mut b, t).bias.to_vec();
            assert_eq!(full.len(), WAIT.len(), "bias dropped at t = {t}");
            assert!(
                full.iter().all(|&(_, d)| d == ThinkBudget::BIAS_MAX),
                "bias at t = {t} is {full:?}, expected the saturated {}",
                ThinkBudget::BIAS_MAX,
            );
        }
    }

    // The `</think>` pull grows over 80%..100% and holds at its peak through the
    // grace period. It always targets `</think>` and nothing else.
    #[test]
    fn think_close_pull_ramps_and_holds() {
        let mut b = budget(100);
        assert!(at(&mut b, 79).pull.is_none());
        let (id, mid) = at(&mut b, 90).pull.expect("pulling by 90%");
        assert_eq!(id, LagunaTokenizer::THINK_CLOSE);
        assert!(mid > 0.0 && mid < ThinkBudget::PULL_MAX);
        let (_, held) = at(&mut b, 110).pull.expect("pull holds through grace");
        assert_eq!(held, ThinkBudget::PULL_MAX);
    }

    // The wrap-up injection waits for a sentence boundary inside its window,
    // then forces its tokens one per draw, and never fires twice.
    #[test]
    fn wrapup_fires_once_at_a_boundary_in_window() {
        let mut b = budget(100);
        // In the window but mid-sentence: nothing is forced.
        b.on_emitted(Some(" and then"));
        assert!(at(&mut b, 90).force.is_none());
        b.on_emitted(Some(" so far so good."));
        for (i, &want) in WRAPUP.iter().enumerate() {
            assert_eq!(commit_forced(&mut b), want, "wrap-up token {i}");
        }
        assert!(b.wrapup_fired);
        assert_eq!(b.clock, 90, "the injection spent the model's budget");
        // A model token, then another boundary inside the window: the one-shot
        // does not repeat.
        commit_natural(&mut b, 5);
        b.on_emitted(Some(" more."));
        assert!(at(&mut b, 94).force.is_none());
    }

    // Outside the window the wrap-up never fires: before it the model has budget
    // left, and after it the grace-period force takes over.
    #[test]
    fn wrapup_stays_inside_its_window() {
        let mut b = budget(100);
        b.on_emitted(Some(" a sentence."));
        assert!(at(&mut b, 89).force.is_none(), "fired before the window");
        assert!(!b.wrapup_fired);
        let mut late = budget(100);
        late.on_emitted(Some(" a sentence."));
        assert!(at(&mut late, 96).force.is_none(), "fired after the window");
        assert!(!late.wrapup_fired);
    }

    // Past the budget the controller waits for a sentence boundary, then injects
    // the transition sentence and `</think>` — after which it is disarmed for
    // the rest of the call and stops touching the logits.
    #[test]
    fn grace_forces_at_the_first_boundary() {
        let mut b = budget(100);
        b.on_emitted(Some(" still going"));
        assert!(at(&mut b, 100).force.is_none(), "forced mid-sentence");
        assert_eq!(b.phase(100), ThinkPhase::Grace);
        b.on_emitted(Some(" that settles it."));
        for (i, &want) in TRANSITION.iter().enumerate() {
            assert_eq!(commit_forced(&mut b), want, "transition token {i}");
        }
        assert_eq!(b.phase(110), ThinkPhase::Done);
        let c = at(&mut b, 110);
        assert!(c.bias.is_empty() && c.pull.is_none() && c.force.is_none());
    }

    // With no boundary in sight the grace period expires and the force fires
    // anyway, at budget + grace exactly.
    #[test]
    fn grace_expiry_forces_without_a_boundary() {
        let mut b = budget(100);
        let grace = b.grace();
        assert_eq!(
            grace,
            ThinkBudget::GRACE_MIN,
            "a 100-token budget floors at the minimum grace"
        );
        b.on_emitted(Some(" no end in sight"));
        for t in 100..100 + grace {
            assert!(at(&mut b, t).force.is_none(), "forced early at t = {t}");
        }
        assert_eq!(at(&mut b, 100 + grace).force, Some(TRANSITION[0]));
    }

    // A model that closes the block on its own disarms the controller: no
    // injection ever happens and the remaining draws are untouched.
    #[test]
    fn natural_think_close_disarms_the_controller() {
        let mut b = budget(100);
        b.clock = 85;
        commit_natural(&mut b, LagunaTokenizer::THINK_CLOSE);
        assert_eq!(b.phase(85), ThinkPhase::Done);
        assert_eq!(
            b.stats(200).tokens,
            86,
            "the closing token counts as thinking"
        );
        assert!(b.stats(200).closed);
        assert!(!b.stats(200).forced);
        b.on_emitted(Some(" answer."));
        assert!(
            at(&mut b, 140).force.is_none(),
            "kept steering after the block closed"
        );
    }

    // Batched verify rounds draw every row before emitting any of them, which
    // would leave the boundary detector reading a stale position — so the
    // controller asks for serial rounds from the wrap-up window on, counting the
    // caller's look-ahead, and releases them the moment the block closes.
    #[test]
    fn serial_rounds_requested_only_when_boundaries_matter() {
        let mut b = budget(100);
        b.clock = 80;
        assert!(!b.needs_serial_rounds(0));
        assert!(!b.needs_serial_rounds(9));
        assert!(
            b.needs_serial_rounds(10),
            "a round reaching into the window must be serial"
        );
        b.clock = 90;
        assert!(b.needs_serial_rounds(0));
        commit_natural(&mut b, LagunaTokenizer::THINK_CLOSE);
        assert!(
            !b.needs_serial_rounds(0),
            "kept rounds serial after the block closed"
        );
    }

    // A disabled budget is inert everywhere and reports no stats, so the default
    // generation path is untouched.
    #[test]
    fn disabled_budget_does_nothing() {
        let mut b = ThinkBudget::off();
        assert!(!b.enabled());
        for t in [0, 100, 10_000] {
            let c = at(&mut b, t);
            assert!(c.bias.is_empty() && c.pull.is_none() && c.force.is_none());
        }
        assert!(!b.needs_serial_rounds(10_000));
    }

    // The wrap-up must leave the model room to conclude in its own words before
    // the hard force can fire. Two things used to collapse that gap, and both
    // need the REAL injection strings to show up — a synthetic two-token wrap-up
    // hides them: the injected tokens advanced the same counter the schedule
    // reads (a ~21-token wrap-up starting at 0.90N ended past the budget), and
    // the wrap-up's own trailing blank line satisfied the grace period's
    // boundary condition. The transition then followed the wrap-up back to back.
    #[test]
    fn wrapup_leaves_the_model_tokens_before_the_hard_force() {
        let tok = tokenizer();
        let mut b = ThinkBudget::new(120, &tok).unwrap();
        assert!(
            b.wrapup.len() >= 10,
            "the real wrap-up is only {} tokens",
            b.wrapup.len()
        );

        // The model reaches the wrap-up window and ends a sentence.
        b.clock = 108;
        b.on_emitted(Some(" so that works."));

        // Drain the injection, feeding each forced token's text back exactly as
        // the decode loop does.
        let mut forced = 0usize;
        loop {
            let Some(id) = b.control_for().force else {
                break;
            };
            b.on_committed(b.clock, id);
            let text = tok.decode(&[id]).unwrap();
            b.on_emitted(Some(&text));
            forced += 1;
            assert!(forced < 200, "the wrap-up injection never drained");
        }
        // The sentence ended without a newline, so the injection opened a
        // paragraph of its own ahead of the wrap-up.
        assert_eq!(
            forced,
            b.paragraph.len() + b.wrapup.len(),
            "drained something other than the wrap-up",
        );
        assert_eq!(b.clock, 108, "forced tokens advanced the schedule clock");
        assert!(
            !b.hard_forced,
            "the hard force followed the wrap-up with no model tokens between"
        );
        // The wrap-up ends on a blank line, so a boundary IS pending — but the
        // injection wrote it, not the model, and the schedule must not act on it.
        assert!(b.at_boundary);
        assert!(!b.at_safe_boundary());

        // The model now spends the rest of its budget; the grace force cannot
        // fire until the budget is actually gone.
        for _ in 0..(120 - 108) {
            assert!(
                b.control_for().force.is_none(),
                "forced before the budget was spent"
            );
            let t = b.clock;
            b.on_committed(t, 5);
            b.on_emitted(Some(" more."));
        }
        assert_eq!(b.clock, 120);
        assert!(
            b.control_for().force.is_some(),
            "the grace force never fired"
        );
    }

    // Budgets too small for the schedule are rejected rather than silently
    // degenerating: below the minimum the wrap-up window is narrower than one
    // token, so the run would go straight from unconstrained to a forced
    // mid-sentence exit.
    #[test]
    fn think_budget_guards_reject_degenerate_configurations() {
        let tok = tokenizer();
        for n in [1, 3, 9, ThinkBudget::MIN_BUDGET - 1] {
            let b = ThinkBudget::new(n, &tok).unwrap();
            assert!(
                check_think_budget(&b, 0, 4096).is_err(),
                "max_think {n} was accepted"
            );
        }
        let b = ThinkBudget::new(ThinkBudget::MIN_BUDGET, &tok).unwrap();
        assert!(check_think_budget(&b, 0, 4096).is_ok());
        // The floor must stay below the ceiling, or the two fight each other.
        assert!(check_think_budget(&b, ThinkBudget::MIN_BUDGET, 4096).is_err());
        // And the token budget must outlast the schedule plus its injections.
        assert!(check_think_budget(&b, 0, ThinkBudget::MIN_BUDGET + 8).is_err());
        // An absurd budget must not wrap around the guard's arithmetic.
        let huge = ThinkBudget {
            budget: usize::MAX,
            ..ThinkBudget::new(64, &tok).unwrap()
        };
        assert!(check_think_budget(&huge, 0, 4096).is_err());
    }

    // The reserve `feasible_think_budget` answers with, having no tokenizer to
    // encode the injections with, has to be exactly what a real controller keeps
    // spare — otherwise it clamps to ceilings the decode loop then rejects.
    #[test]
    fn pinned_reserve_matches_the_encoded_injections() {
        let b = ThinkBudget::new(64, &tokenizer()).unwrap();
        assert_eq!(
            b.reserve(),
            ThinkBudget::PINNED_RESERVE,
            "the injection strings now reserve {} tokens; update PINNED_RESERVE",
            b.reserve(),
        );
    }

    // A requested ceiling comes back untouched when it fits and lowered to the
    // largest one that does when it does not — never one the decode loop's guard
    // would reject once prefill has already been paid for.
    #[test]
    fn feasible_think_budget_agrees_with_the_guard() {
        let statics = ThinkStatics::new(&tokenizer()).unwrap();
        let accepted = |budget: usize, max_new: usize| {
            let b = ThinkBudget::from_statics(budget, statics.clone());
            check_think_budget(&b, 0, max_new).is_ok()
        };

        // Replies too short to conclude in have no expressible ceiling at all;
        // the caller reasons uncapped instead of failing after prefill.
        assert_eq!(feasible_think_budget(2048, 96), None);
        assert_eq!(feasible_think_budget(10_000, 100), None);
        // Nothing below the schedule's minimum can be honoured either.
        assert_eq!(feasible_think_budget(0, 4096), None);
        assert_eq!(
            feasible_think_budget(ThinkBudget::MIN_BUDGET - 1, 4096),
            None
        );

        // Room to spare: the request is honoured exactly.
        assert_eq!(feasible_think_budget(1024, 4096), Some(1024));
        assert_eq!(
            feasible_think_budget(ThinkBudget::MIN_BUDGET, 4096),
            Some(ThinkBudget::MIN_BUDGET),
        );

        // Not quite enough room: clamped to the largest ceiling that fits, with
        // one token more genuinely infeasible.
        let n = feasible_think_budget(1024, 1200).expect("1200 tokens leaves room for a ceiling");
        assert!(n < 1024, "1024 cannot fit in 1200 tokens");
        assert!(
            accepted(n, 1200),
            "the clamped ceiling {n} is rejected by the guard"
        );
        assert!(
            !accepted(n + 1, 1200),
            "{} fits and was left on the table",
            n + 1
        );

        // The same two properties over a spread of shapes, checked against the
        // guard the decode loop actually runs.
        for max_new in [64, 96, 100, 128, 200, 256, 512, 1200, 4096, 32_768] {
            for requested in [
                ThinkBudget::MIN_BUDGET,
                33,
                64,
                100,
                512,
                1024,
                8192,
                usize::MAX,
            ] {
                match feasible_think_budget(requested, max_new) {
                    Some(n) => {
                        assert!(
                            n <= requested,
                            "({requested}, {max_new}) raised the request to {n}"
                        );
                        assert!(
                            accepted(n, max_new),
                            "({requested}, {max_new}) -> {n} is rejected"
                        );
                        if n < requested {
                            assert!(
                                !accepted(n + 1, max_new),
                                "({requested}, {max_new}) stopped short at {n}",
                            );
                        }
                    }
                    // Nothing fits only if the smallest expressible ceiling does
                    // not: the span grows with the budget.
                    None => assert!(
                        requested < ThinkBudget::MIN_BUDGET
                            || !accepted(ThinkBudget::MIN_BUDGET, max_new),
                        "({requested}, {max_new}) gave up while a ceiling still fits",
                    ),
                }
            }
        }
    }

    // The ramp is a true no-op below its start and pinned at full above its end;
    // in between it rises monotonically through its midpoint at 0.5.
    #[test]
    fn ramp_is_clamped_and_monotonic() {
        assert_eq!(ramp(0.5, 0.7, 0.95), 0.0);
        assert_eq!(ramp(0.7, 0.7, 0.95), 0.0);
        assert_eq!(ramp(0.95, 0.7, 0.95), 1.0);
        assert_eq!(ramp(9.0, 0.7, 0.95), 1.0);
        assert!((ramp(0.825, 0.7, 0.95) - 0.5).abs() < 1e-6);
        let mut prev = 0.0;
        for i in 0..=100 {
            let x = 0.7 + 0.25 * i as f64 / 100.0;
            let v = ramp(x, 0.7, 0.95);
            assert!(v >= prev, "ramp dipped at x = {x}");
            prev = v;
        }
    }

    // ---- Vocabulary scans (real tokenizer).

    fn tokenizer() -> LagunaTokenizer {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/tokenizer.json");
        LagunaTokenizer::from_file(path).expect("load reference tokenizer")
    }

    /// Every id whose decoded text is exactly `text`.
    fn ids_decoding_to(t: &LagunaTokenizer, text: &str) -> Vec<u32> {
        (0..t.vocab_size() as u32)
            .filter(|&id| t.decode(&[id]).is_ok_and(|s| s == text))
            .collect()
    }

    // The wait-token set is matched on whole words: the continuation cues are
    // in, and words that merely contain one are not.
    #[test]
    fn wait_token_scan_matches_whole_words_only() {
        let t = tokenizer();
        let ids = scan_wait_tokens(&t).unwrap();
        for probe in [" wait", "Wait", " But", " however", " Hmm"] {
            let want = ids_decoding_to(&t, probe);
            assert!(!want.is_empty(), "no token decodes to {probe:?}");
            for id in want {
                assert!(
                    ids.contains(&id),
                    "{probe:?} (id {id}) missing from the wait set"
                );
            }
        }
        // Containment would drag these in through "any", "other" and "check".
        for probe in [" company", " brother", " checkpoint", " nowhere"] {
            for id in ids_decoding_to(&t, probe) {
                assert!(
                    !ids.contains(&id),
                    "{probe:?} (id {id}) wrongly in the wait set"
                );
            }
        }
    }

    // The em-dash mask: the glyph appears in dozens of tokens, and banning it
    // must catch all of them, not just the standalone punctuation token.
    #[test]
    fn ban_scan_catches_every_em_dash_token() {
        const EM_DASH: &str = "\u{2014}";
        let t = tokenizer();
        let ids = scan_banned(&t, &eog(), EM_DASH).unwrap();
        assert!(
            ids.len() >= 30,
            "only {} tokens contain an em dash",
            ids.len()
        );
        // Every token carrying the glyph is masked, and the mask reaches well
        // past the standalone punctuation token: the glyph is baked into dozens
        // of longer pieces, which is the whole reason the scan decodes the
        // vocabulary instead of banning one id.
        let banned: std::collections::HashSet<u32> = ids.iter().copied().collect();
        let mut composite = 0;
        for id in 0..t.vocab_size() as u32 {
            let text = t.decode(&[id]).unwrap();
            if !text.contains(EM_DASH) {
                continue;
            }
            assert!(
                banned.contains(&id),
                "id {id} ({text:?}) contains an em dash but was not banned"
            );
            if text.chars().count() > 1 {
                composite += 1;
            }
        }
        assert!(
            composite >= 20,
            "only {composite} tokens carry the glyph inside a longer piece"
        );
    }

    // A pattern that matches nothing is a typo, not a silent no-op.
    #[test]
    fn ban_scan_rejects_a_pattern_with_no_matches() {
        let t = tokenizer();
        assert!(scan_banned(&t, &eog(), "zzqzzqzzqzz").is_err());
        assert!(scan_banned(&t, &eog(), "").is_err());
    }

    // Structural tokens are off limits — but an ordinary pattern collides with
    // them by accident (the markers spell English inside their brackets), so the
    // collision is skipped and the rest of the matches still ban.
    #[test]
    fn ban_scan_skips_structural_tokens_and_keeps_the_rest() {
        let t = tokenizer();
        let protected = t.added_token_ids();
        let ids = scan_banned(&t, &eog(), "assistant").unwrap();
        assert!(
            !ids.is_empty(),
            "the ordinary word tokens should still be banned"
        );
        for id in &ids {
            assert!(!protected.contains(id), "banned structural token {id}");
        }
    }

    /// The runtime end-of-generation set, which `XwenConfig` derives from the
    /// checkpoint's own metadata. On both shipped files it comes out equal to
    /// `LagunaTokenizer::EOG`, which is what lets these tests stand in for it.
    fn eog() -> Vec<u32> {
        LagunaTokenizer::EOG.to_vec()
    }

    // The protected set is whatever the decode loop stops on, not a
    // compile-time list. A checkpoint whose stop id is an ordinary token must
    // still have that id protected — otherwise `--ban-string` could ban the one
    // token that ends a reply, and generation would run to the output cap every
    // time.
    #[test]
    fn ban_scan_protects_whatever_the_decode_loop_stops_on() {
        let t = tokenizer();
        let unguarded = scan_banned(&t, &eog(), "assistant").unwrap();
        let stop = *unguarded
            .first()
            .expect("an ordinary token matches the pattern");

        let guarded = scan_banned(&t, &[stop], "assistant").unwrap();
        assert!(
            !guarded.contains(&stop),
            "id {stop} is this checkpoint's stop token and must not be banned"
        );
        // Only that one is spared; the rest of the matches still ban.
        assert_eq!(guarded.len(), unguarded.len() - 1);
    }

    // A pattern that matches ONLY structural tokens bans nothing, and says so
    // rather than reporting success over an empty set.
    #[test]
    fn ban_scan_errors_when_only_structural_tokens_match() {
        let t = tokenizer();
        // `</tool_call>` is the one with teeth: banning it would let the model
        // open a tool call it could never close.
        for pattern in ["</think>", "<tool_call>", "</tool_call>", "|EOS|", "|PAD|"] {
            assert!(
                scan_banned(&t, &eog(), pattern).is_err(),
                "{pattern:?} banned a structural token without complaint",
            );
        }
    }

    // draft_argmax_probs reduces per-row argmax + softmax-of-argmax on-device;
    // it must match an independent scalar reference row-for-row. Runs on the GPU
    // when available (else CPU) over distinct random logits (no exact ties, so
    // the first-maximal argmax convention is unambiguous).
    #[test]
    fn draft_argmax_probs_matches_scalar() {
        let dev = crate::gguf::metal_device().unwrap_or(Device::Cpu);
        let (n, vocab) = (5usize, 257usize);
        // Distinct pseudo-random logits, deterministic across runs.
        let data: Vec<f32> = (0..n * vocab)
            .map(|i| (i as f32 * 1.2345).sin() * 7.0)
            .collect();
        let logits = Tensor::from_vec(data.clone(), (n, vocab), &dev).unwrap();

        let (ids, probs) = draft_argmax_probs(&logits).unwrap();
        assert_eq!(ids.len(), n);
        assert_eq!(probs.len(), n);

        for r in 0..n {
            let row = &data[r * vocab..(r + 1) * vocab];
            let mut max = f32::NEG_INFINITY;
            let mut arg = 0usize;
            for (i, &l) in row.iter().enumerate() {
                if l > max {
                    max = l;
                    arg = i;
                }
            }
            let sum: f64 = row.iter().map(|&l| ((l - max) as f64).exp()).sum();
            let want_p = (1.0 / sum) as f32;
            assert_eq!(ids[r] as usize, arg, "row {r} argmax");
            assert!(
                (probs[r] - want_p).abs() < 1e-5,
                "row {r} prob {} vs {want_p}",
                probs[r]
            );
        }
    }
}
