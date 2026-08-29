//! In-situ per-stage timing of the model's forward stack, gated on
//! `XWEN_STACK_PROFILE` (`ops::stack_profile`).
//!
//! Kernel-level benchmarks price one stage at a time on synthetic shapes. This
//! prices the SAME stages where they actually run — inside `run_stack`, on the
//! real weights, in the real dispatch order — and reports what the chunk's wall
//! clock holds that no stage claims. The gap between a chunk's wall time and the
//! sum of its stages is the number this module exists to produce; a stage budget
//! assembled from separate microbenchmarks cannot see it at all.
//!
//! Timing is by device sync: each stage is bracketed by `Device::synchronize`,
//! so a stage's total is completed GPU work, not enqueue time. Two adjacent
//! stages share the sync between them — the closing sync of one opens the next —
//! so the brackets add one sync per stage, not two, and the host time in the gap
//! is charged to [`Stage::InterStageHost`] instead of to whichever stage happens
//! to follow it. The host-only stages (the CPU mask fill) are timed with a plain
//! `Instant` and end by re-marking the clock, for the same reason.
//!
//! Every interval inside a chunk's bracket therefore belongs to some bucket, so
//! the reported `unaccounted` is a consistency check on the brackets themselves
//! — expect zero — and NOT the residual being hunted. The residual shows up as a
//! bucket: either a kernel stage costing more in situ than on the bench, or
//! `inter_stage_host` carrying real per-token cost.
//!
//! The stage set spans every architecture the binary runs, so a dump only ever
//! prints the stages that actually ran: `mixer_delta` is absent from a chunk
//! with no DeltaNet layer, and `ple`, `qsa_select` and `token_readback` — the
//! qwen4exp-only stages — are absent from a qwen35 dump entirely. Nothing on the
//! qwen35 path calls them.
//!
//! Which phase a chunk belongs to is DECLARED by the generation loop
//! (`XwenModel::set_phase`), not inferred: a prompt's last prefill chunk can hold
//! a single token, and a speculative verify forward feeds a whole span while
//! being decode, so token count identifies neither.
//!
//! Two limits worth knowing before reading a dump:
//!
//! - A stage that errors mid-chunk leaves the stage totals it already committed
//!   with no matching chunk wall or token count, so a dump taken after a failed
//!   forward overstates the stages and understates per-token cost. Nothing
//!   compensates for this, because a forward error aborts the generation that
//!   would have printed the dump.
//! - It is built for plain `--no-draft` generation. The speculative and server
//!   paths accumulate correctly (their phases are declared like anyone else's),
//!   but only plain `generate` prints dumps, so their numbers are reachable only
//!   by adding a dump call.
//!
//! Everything here is diagnosis: no arithmetic in the model changes, and with the
//! variable unset the cost at each instrumented site is one `Option` check.

use std::time::{Duration, Instant};

use anyhow::Result;
use candle_core::Device;

use crate::host_log::host_line;

/// The stack stages a chunk's wall clock is decomposed into. Ordered as the
/// forward runs them; the dump prints them in this order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    /// Embedding gather + f32 upcast. On qwen4exp it also covers the
    /// hyper-connection carrier seed (the `[x, x, x, x]` tile of that
    /// embedding), which is the same one-shot per-chunk device work and would
    /// otherwise be charged to whichever stage's sync happened to drain it.
    Embed,
    /// Reading this chunk's token ids back off the device, which the qwen4exp
    /// PLE hash needs on the host. Once per forward, and only on a checkpoint
    /// that carries a PLE layer.
    TokenReadback,
    /// The CPU-side `Vec<f32>` fill of the prefill attention mask (host work —
    /// timed without a device sync).
    MaskFillHost,
    /// Uploading that mask and materializing the broadcast f16 sdpa copy.
    MaskUploadAndBroadcast,
    /// A qwen4exp PLE layer's per-token embedding lookup and its add onto the
    /// carrier. Host/device hybrid: the n-gram hash runs on the CPU over the
    /// ids [`Stage::TokenReadback`] fetched.
    Ple,
    /// Per-layer pre-mixer RMSNorm.
    AttnNorm,
    /// A qwen4exp QSA layer's index selection: the indexer's own projections,
    /// the raw-key append, and the top-k that picks the attended set. Runs
    /// before the mixer it feeds, at the same position the K/V append uses.
    QsaSelect,
    /// A full-attention layer's whole mixer call (projections, rope, sdpa,
    /// output gate, o_proj).
    MixerFullAttn,
    /// A gated-DeltaNet layer's whole mixer call (conv, scan, gated norm,
    /// ssm_out).
    MixerDelta,
    /// The post-mixer residual add.
    ResidualAttn,
    /// Per-layer pre-FFN RMSNorm (`post_attention_norm`).
    FfnNorm,
    /// The FFN itself — dense SwiGLU or the MoE block.
    Ffn,
    /// The post-FFN residual add.
    ResidualFfn,
    /// The final `output_norm`.
    FinalNorm,
    /// The lm head, including the last-position narrow that feeds it.
    LmHead,
    /// Host time BETWEEN two stages, when the device is idle: loop bookkeeping,
    /// tap checks, tensor drops (which free device buffers), and whatever else
    /// candle does on the CPU between one stage's last dispatch and the next
    /// one's first. Not a position in the forward — it is every gap, summed.
    ///
    /// It has a bucket because the alternative is worse: the sync that closes a
    /// stage also opens the next one, so an unbucketed gap would be charged to
    /// the stage that follows it, and per-token cost that lives in the glue
    /// would show up as a kernel being mysteriously slower in situ than on the
    /// bench. Bounding it needs no extra sync — the device is already idle
    /// across the whole interval.
    InterStageHost,
}

const STAGES: [Stage; 16] = [
    Stage::Embed,
    Stage::TokenReadback,
    Stage::MaskFillHost,
    Stage::MaskUploadAndBroadcast,
    Stage::Ple,
    Stage::AttnNorm,
    Stage::QsaSelect,
    Stage::MixerFullAttn,
    Stage::MixerDelta,
    Stage::ResidualAttn,
    Stage::FfnNorm,
    Stage::Ffn,
    Stage::ResidualFfn,
    Stage::FinalNorm,
    Stage::LmHead,
    Stage::InterStageHost,
];

impl Stage {
    fn index(self) -> usize {
        match self {
            Stage::Embed => 0,
            Stage::TokenReadback => 1,
            Stage::MaskFillHost => 2,
            Stage::MaskUploadAndBroadcast => 3,
            Stage::Ple => 4,
            Stage::AttnNorm => 5,
            Stage::QsaSelect => 6,
            Stage::MixerFullAttn => 7,
            Stage::MixerDelta => 8,
            Stage::ResidualAttn => 9,
            Stage::FfnNorm => 10,
            Stage::Ffn => 11,
            Stage::ResidualFfn => 12,
            Stage::FinalNorm => 13,
            Stage::LmHead => 14,
            Stage::InterStageHost => 15,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Stage::Embed => "embed",
            Stage::TokenReadback => "token_readback",
            Stage::MaskFillHost => "mask_fill_host",
            Stage::MaskUploadAndBroadcast => "mask_upload",
            Stage::Ple => "ple",
            Stage::AttnNorm => "attn_norm",
            Stage::QsaSelect => "qsa_select",
            Stage::MixerFullAttn => "mixer_full_attn",
            Stage::MixerDelta => "mixer_delta",
            Stage::ResidualAttn => "residual_attn",
            Stage::FfnNorm => "ffn_norm",
            Stage::Ffn => "ffn",
            Stage::ResidualFfn => "residual_ffn",
            Stage::FinalNorm => "final_norm",
            Stage::LmHead => "lm_head",
            Stage::InterStageHost => "inter_stage_host",
        }
    }
}

/// Which accumulator a chunk's stages land in. DECLARED BY THE CALLER
/// (`XwenModel::set_phase`), never inferred from the forward: a chunk's token
/// count does not identify its phase in either direction — a prompt whose last
/// prefill chunk holds one token is still prefill, and a speculative verify
/// forward feeds a whole span of tokens while being decode. The generation loops
/// know which they are running; the profiler does not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    Prefill,
    Decode,
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Phase::Prefill => "prefill",
            Phase::Decode => "decode",
        }
    }
}

/// One stage's running total across every chunk of a phase.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
struct Bucket {
    total: Duration,
    calls: u64,
}

/// Everything one phase accumulated: per-stage totals plus the chunk-level wall
/// clock the stages have to add up to.
#[derive(Clone, Default)]
struct PhaseAcc {
    buckets: [Bucket; STAGES.len()],
    chunks: u64,
    /// Summed synced wall time of the phase's chunks.
    wall: Duration,
    /// Summed tokens fed by the phase's chunks.
    tokens: u64,
}

impl PhaseAcc {
    fn add(&mut self, stage: Stage, elapsed: Duration) {
        let b = &mut self.buckets[stage.index()];
        b.total += elapsed;
        b.calls += 1;
    }

    fn stage_total(&self) -> Duration {
        self.buckets.iter().map(|b| b.total).sum()
    }

    /// The part of the phase's wall clock no stage claimed. Every interval
    /// inside a chunk's bracket has a bucket — the glue between stages included
    /// — so this is expected to be zero, and anything else means a chunk closed
    /// on a path the brackets do not cover (an errored forward, or a chunk that
    /// ran no stages). Saturating: a negative residual is clock noise, not a
    /// finding.
    fn unaccounted(&self) -> Duration {
        self.wall.saturating_sub(self.stage_total())
    }

    fn is_empty(&self) -> bool {
        self.chunks == 0
    }
}

/// Per-stage timing accumulators for one model, split by phase.
pub struct StackProfiler {
    prefill: PhaseAcc,
    decode: PhaseAcc,
    /// The accumulator subsequent chunks land in, as last declared by
    /// [`StackProfiler::set_phase`]. Prefill until someone says otherwise, which
    /// is right for a caller that only ever prefills (offline scoring).
    current: Phase,
    /// Start of the open chunk's wall bracket, taken just after a device sync.
    chunk_start: Option<Instant>,
    /// Tokens the open chunk feeds.
    chunk_tokens: usize,
    /// The most recent instant at which the device was known idle, when it can
    /// still serve as the next stage's start without a fresh sync.
    mark: Option<Instant>,
}

impl StackProfiler {
    pub fn new() -> Self {
        Self {
            prefill: PhaseAcc::default(),
            decode: PhaseAcc::default(),
            current: Phase::Prefill,
            chunk_start: None,
            chunk_tokens: 0,
            mark: None,
        }
    }

    /// Declare which phase the forwards from here on belong to. Takes effect at
    /// the next chunk; the open chunk (if any) keeps the phase it opened with.
    pub fn set_phase(&mut self, phase: Phase) {
        self.current = phase;
    }

    /// Drop everything accumulated so far, keeping the declared phase. What a
    /// throwaway warm-up pass is erased with, so its page-in and pipeline-compile
    /// costs — the very costs the warm-up exists to keep out of the measurement —
    /// do not survive into the reported totals.
    pub fn reset(&mut self) {
        self.prefill = PhaseAcc::default();
        self.decode = PhaseAcc::default();
        self.chunk_start = None;
        self.chunk_tokens = 0;
        self.mark = None;
    }

    fn phase_mut(&mut self, phase: Phase) -> &mut PhaseAcc {
        match phase {
            Phase::Prefill => &mut self.prefill,
            Phase::Decode => &mut self.decode,
        }
    }

    fn phase(&self, phase: Phase) -> &PhaseAcc {
        match phase {
            Phase::Prefill => &self.prefill,
            Phase::Decode => &self.decode,
        }
    }

    /// Start accounting a chunk of `tokens` into the declared phase. The
    /// device-synced entry point is [`chunk_begin`]; this is the arithmetic
    /// underneath it.
    fn open_chunk(&mut self, tokens: usize) {
        self.chunk_tokens = tokens;
    }

    /// Close the open chunk with its measured wall time.
    fn close_chunk(&mut self, wall: Duration) {
        let (phase, tokens) = (self.current, self.chunk_tokens);
        let acc = self.phase_mut(phase);
        acc.chunks += 1;
        acc.wall += wall;
        acc.tokens += tokens as u64;
        self.chunk_tokens = 0;
    }

    fn add(&mut self, stage: Stage, elapsed: Duration) {
        let phase = self.current;
        self.phase_mut(phase).add(stage, elapsed);
    }

    /// Report both phases under `label` (which names the call site, e.g.
    /// `"after-prefill"`), one line per stage that ran. An empty phase prints
    /// nothing.
    pub fn dump(&self, label: &str) {
        for phase in [Phase::Prefill, Phase::Decode] {
            let acc = self.phase(phase);
            if acc.is_empty() {
                continue;
            }
            let tag = format!("stack-profile[{label}/{}]", phase.name());
            let tokens = acc.tokens.max(1) as f64;
            host_line(format!(
                "xwen: {tag} chunks={} tokens={} wall={} stages={} unaccounted={} ({:.1}us/tok)",
                acc.chunks,
                acc.tokens,
                ms(acc.wall),
                ms(acc.stage_total()),
                ms(acc.unaccounted()),
                acc.unaccounted().as_secs_f64() * 1e6 / tokens,
            ));
            for stage in STAGES {
                let b = acc.buckets[stage.index()];
                if b.calls == 0 {
                    continue;
                }
                host_line(format!(
                    "xwen: {tag} {:<16} calls={:<6} total={:>10} {:>9.1}us/tok",
                    stage.name(),
                    b.calls,
                    ms(b.total),
                    b.total.as_secs_f64() * 1e6 / tokens,
                ));
            }
        }
    }
}

impl Default for StackProfiler {
    fn default() -> Self {
        Self::new()
    }
}

fn ms(d: Duration) -> String {
    format!("{:.3}ms", d.as_secs_f64() * 1e3)
}

// ------------------------------------------------------------------ hooks
//
// Free functions over `&mut Option<StackProfiler>` rather than methods on the
// model: the instrumented sites sit inside `run_stack`'s per-layer loop, where
// the layer and its cache are already borrowed out of `self`, so a hook has to
// touch the profiler field alone.

/// Open a chunk of `tokens` tokens: sync so the bracket starts from an idle
/// device. The chunk lands in whichever phase was last declared.
pub fn chunk_begin(p: &mut Option<StackProfiler>, device: &Device, tokens: usize) -> Result<()> {
    let Some(p) = p.as_mut() else { return Ok(()) };
    device.synchronize()?;
    let now = Instant::now();
    p.open_chunk(tokens);
    p.chunk_start = Some(now);
    p.mark = Some(now);
    Ok(())
}

/// Close the open chunk, charging the full bracket to its phase. The last
/// stage's closing sync already left the device idle, so this adds no sync of
/// its own; the host tail since that sync is glue and is charged as such, which
/// keeps the chunk's wall clock equal to the sum of its buckets.
pub fn chunk_end(p: &mut Option<StackProfiler>, device: &Device) -> Result<()> {
    let Some(p) = p.as_mut() else { return Ok(()) };
    let Some(start) = p.chunk_start.take() else {
        return Ok(());
    };
    match p.mark.take() {
        Some(idle_since) => {
            let now = Instant::now();
            p.add(
                Stage::InterStageHost,
                now.saturating_duration_since(idle_since),
            );
            p.close_chunk(now.saturating_duration_since(start));
        }
        // No stage ran, or the last one left work queued: bound the chunk the
        // only way that is honest here.
        None => {
            device.synchronize()?;
            p.close_chunk(start.elapsed());
        }
    }
    Ok(())
}

/// Start a stage's clock. Syncs first unless the previous stage's closing sync
/// already left the device idle at a known instant — in which case the interval
/// since is host glue and is charged to [`Stage::InterStageHost`] rather than to
/// the stage about to run.
pub fn stage_begin(p: &mut Option<StackProfiler>, device: &Device) -> Result<()> {
    let Some(p) = p.as_mut() else { return Ok(()) };
    match p.mark {
        Some(idle_since) => {
            let now = Instant::now();
            p.add(
                Stage::InterStageHost,
                now.saturating_duration_since(idle_since),
            );
            p.mark = Some(now);
        }
        None => {
            device.synchronize()?;
            p.mark = Some(Instant::now());
        }
    }
    Ok(())
}

/// Close a stage: sync, charge the elapsed time to `stage`, and keep the sync
/// instant as the next stage's start.
pub fn stage_end(p: &mut Option<StackProfiler>, device: &Device, stage: Stage) -> Result<()> {
    let Some(p) = p.as_mut() else { return Ok(()) };
    device.synchronize()?;
    let now = Instant::now();
    let start = p.mark.unwrap_or(now);
    p.add(stage, now.saturating_duration_since(start));
    p.mark = Some(now);
    Ok(())
}

/// Start a host-only stage's clock, charging the interval since the previous
/// stage's closing sync to [`Stage::InterStageHost`] exactly as [`stage_begin`]
/// does. `None` when profiling is off, which is what [`host_end`] expects back.
pub fn host_begin(p: &mut Option<StackProfiler>) -> Option<Instant> {
    let p = p.as_mut()?;
    let now = Instant::now();
    if let Some(idle_since) = p.mark {
        p.add(
            Stage::InterStageHost,
            now.saturating_duration_since(idle_since),
        );
    }
    p.mark = Some(now);
    Some(now)
}

/// Close a host-only stage measured with a plain `Instant`: no sync is needed to
/// bound work that never touched the device, but the clock is re-marked so the
/// host time is not charged again to the stage that follows.
pub fn host_end(p: &mut Option<StackProfiler>, stage: Stage, start: Option<Instant>) {
    let (Some(p), Some(start)) = (p.as_mut(), start) else {
        return;
    };
    let now = Instant::now();
    p.add(stage, now.saturating_duration_since(start));
    p.mark = Some(now);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn us(n: u64) -> Duration {
        Duration::from_micros(n)
    }

    /// Stage totals and call counts accumulate across a phase's chunks, and the
    /// unaccounted residual is the phase's wall clock minus everything its
    /// stages claimed.
    #[test]
    fn stage_totals_and_residual_accumulate_per_phase() {
        let mut p = StackProfiler::new();
        p.set_phase(Phase::Prefill);

        p.open_chunk(4);
        p.add(Stage::Embed, us(100));
        p.add(Stage::Ffn, us(300));
        p.close_chunk(us(1000));

        p.open_chunk(4);
        p.add(Stage::Ffn, us(200));
        p.close_chunk(us(500));

        let pre = p.phase(Phase::Prefill);
        assert_eq!(pre.chunks, 2);
        assert_eq!(pre.tokens, 8);
        assert_eq!(pre.wall, us(1500));
        assert_eq!(pre.buckets[Stage::Ffn.index()].calls, 2);
        assert_eq!(pre.buckets[Stage::Ffn.index()].total, us(500));
        assert_eq!(pre.buckets[Stage::Embed.index()].calls, 1);
        assert_eq!(pre.stage_total(), us(600));
        assert_eq!(pre.unaccounted(), us(900));
    }

    /// Decode chunks accumulate into their own phase and leave the prefill one
    /// untouched.
    #[test]
    fn phases_do_not_mix() {
        let mut p = StackProfiler::new();

        p.set_phase(Phase::Prefill);
        p.open_chunk(16);
        p.add(Stage::Ffn, us(400));
        p.close_chunk(us(600));

        p.set_phase(Phase::Decode);
        for _ in 0..3 {
            p.open_chunk(1);
            p.add(Stage::Ffn, us(30));
            p.add(Stage::LmHead, us(20));
            p.close_chunk(us(100));
        }

        let dec = p.phase(Phase::Decode);
        assert_eq!(dec.chunks, 3);
        assert_eq!(dec.tokens, 3);
        assert_eq!(dec.wall, us(300));
        assert_eq!(dec.stage_total(), us(150));
        assert_eq!(dec.unaccounted(), us(150));

        let pre = p.phase(Phase::Prefill);
        assert_eq!(pre.tokens, 16);
        assert_eq!(pre.unaccounted(), us(200));
        assert_eq!(pre.buckets[Stage::LmHead.index()].calls, 0);
    }

    /// The declared phase decides where a chunk lands, whatever its token count:
    /// a one-token prefill tail is prefill, and a multi-token speculative verify
    /// span is decode.
    #[test]
    fn the_declared_phase_decides_not_the_token_count() {
        let mut p = StackProfiler::new();

        p.set_phase(Phase::Prefill);
        p.open_chunk(1);
        p.add(Stage::Ffn, us(70));
        p.close_chunk(us(100));

        p.set_phase(Phase::Decode);
        p.open_chunk(5);
        p.add(Stage::Ffn, us(40));
        p.close_chunk(us(60));

        let pre = p.phase(Phase::Prefill);
        assert_eq!(pre.chunks, 1);
        assert_eq!(pre.tokens, 1);
        assert_eq!(pre.buckets[Stage::Ffn.index()].total, us(70));

        let dec = p.phase(Phase::Decode);
        assert_eq!(dec.chunks, 1);
        assert_eq!(dec.tokens, 5);
        assert_eq!(dec.buckets[Stage::Ffn.index()].total, us(40));
    }

    /// A reset drops both phases whole — what a throwaway warm-up pass is erased
    /// with — and leaves the declared phase in place for the run that follows.
    #[test]
    fn reset_clears_both_phases_and_keeps_the_phase_declaration() {
        let mut p = StackProfiler::new();

        p.set_phase(Phase::Prefill);
        p.open_chunk(512);
        p.add(Stage::Ffn, us(900));
        p.close_chunk(us(1000));
        p.set_phase(Phase::Decode);
        p.open_chunk(1);
        p.add(Stage::LmHead, us(30));
        p.close_chunk(us(50));

        p.reset();

        for phase in [Phase::Prefill, Phase::Decode] {
            let acc = p.phase(phase);
            assert_eq!(acc.chunks, 0);
            assert_eq!(acc.tokens, 0);
            assert_eq!(acc.wall, Duration::ZERO);
            assert_eq!(acc.stage_total(), Duration::ZERO);
            assert!(acc.is_empty());
        }

        // Still decode, as declared before the reset.
        p.open_chunk(1);
        p.add(Stage::Ffn, us(10));
        p.close_chunk(us(20));
        assert_eq!(p.phase(Phase::Decode).chunks, 1);
        assert_eq!(p.phase(Phase::Prefill).chunks, 0);
    }

    /// A chunk whose stages measure slightly more than its own bracket reports
    /// no residual rather than wrapping around.
    #[test]
    fn residual_floors_at_zero() {
        let mut p = StackProfiler::new();
        p.set_phase(Phase::Decode);
        p.open_chunk(1);
        p.add(Stage::Embed, us(90));
        p.close_chunk(us(80));
        assert_eq!(p.phase(Phase::Decode).unaccounted(), Duration::ZERO);
    }

    /// The synced hooks leave nothing inside the chunk bracket unclaimed: host
    /// time between two stages — and between the last stage and the chunk's
    /// close — is charged to the glue bucket rather than to the stage that
    /// follows it, so the chunk's wall clock equals the sum of its buckets.
    /// Runs on the CPU device — the hooks are device-agnostic.
    #[test]
    fn synced_hooks_charge_every_gap_to_glue() {
        let device = Device::Cpu;
        let gap = Duration::from_millis(10);
        let mut p = Some(StackProfiler::new());

        chunk_begin(&mut p, &device, 4).unwrap();
        stage_begin(&mut p, &device).unwrap();
        std::thread::sleep(gap);
        stage_end(&mut p, &device, Stage::Embed).unwrap();
        std::thread::sleep(gap);
        stage_begin(&mut p, &device).unwrap();
        stage_end(&mut p, &device, Stage::Ffn).unwrap();
        std::thread::sleep(gap);
        chunk_end(&mut p, &device).unwrap();

        let p = p.unwrap();
        let acc = p.phase(Phase::Prefill);
        assert_eq!(acc.chunks, 1);
        assert_eq!(acc.tokens, 4);
        assert!(acc.buckets[Stage::Embed.index()].total >= gap);
        assert!(acc.buckets[Stage::Ffn.index()].total < gap);
        // The gap before the second stage and the one before the close: both glue.
        assert!(acc.buckets[Stage::InterStageHost.index()].total >= 2 * gap);
        assert_eq!(acc.unaccounted(), Duration::ZERO);
        assert_eq!(acc.stage_total(), acc.wall);
    }

    /// A host-only stage claims only its own work: the interval before it is
    /// glue, and the interval it covers is not charged again to the stage that
    /// follows.
    #[test]
    fn a_host_stage_does_not_swallow_the_gap_before_it() {
        let device = Device::Cpu;
        let gap = Duration::from_millis(10);
        let mut p = Some(StackProfiler::new());

        chunk_begin(&mut p, &device, 4).unwrap();
        std::thread::sleep(gap);
        let started = host_begin(&mut p);
        std::thread::sleep(gap);
        host_end(&mut p, Stage::MaskFillHost, started);
        stage_begin(&mut p, &device).unwrap();
        stage_end(&mut p, &device, Stage::MaskUploadAndBroadcast).unwrap();
        chunk_end(&mut p, &device).unwrap();

        let p = p.unwrap();
        let acc = p.phase(Phase::Prefill);
        let fill = acc.buckets[Stage::MaskFillHost.index()].total;
        assert!(fill >= gap);
        assert!(fill < 2 * gap);
        assert!(acc.buckets[Stage::InterStageHost.index()].total >= gap);
        assert!(acc.buckets[Stage::MaskUploadAndBroadcast.index()].total < gap);
        assert_eq!(acc.stage_total(), acc.wall);
    }

    /// Profiling off is inert: every hook returns without touching anything.
    #[test]
    fn the_hooks_are_no_ops_when_profiling_is_off() {
        let device = Device::Cpu;
        let mut p: Option<StackProfiler> = None;
        chunk_begin(&mut p, &device, 4).unwrap();
        stage_begin(&mut p, &device).unwrap();
        stage_end(&mut p, &device, Stage::Embed).unwrap();
        let started = host_begin(&mut p);
        host_end(&mut p, Stage::MaskFillHost, started);
        chunk_end(&mut p, &device).unwrap();
        assert!(p.is_none());
        assert!(started.is_none());
    }
}
