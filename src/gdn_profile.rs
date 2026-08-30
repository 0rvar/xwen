//! Sub-step timing INSIDE one gated-DeltaNet block, gated on
//! `XWEN_GDN_PROFILE` (`ops::gdn_profile`).
//!
//! `stack_profile` prices `Stage::MixerDelta` as one number per chunk — the
//! whole GDN mixer, summed over every DeltaNet layer. This splits that number
//! into the steps the block actually runs (the three big projections, the
//! beta/decay head, the fused conv, the recurrent scan, the gated output norm)
//! and says which of them the time is in.
//!
//! Same sync contract as `stack_profile`: a device step is closed by
//! `Device::synchronize`, so its figure is completed GPU work rather than
//! enqueue time, and the block opens with a sync so a backlog inherited from
//! the caller is charged to the caller's bracket. Adjacent steps share the sync
//! between them; the host time in a gap goes to `glue`, which is why the steps
//! sum to the block's bracket and nothing is silently reassigned to whichever
//! kernel happened to follow it.
//!
//! Two levels, because the interesting number spans layers rather than living
//! in one of them:
//!
//! - [`BlockProfile`] is per block call, held as an `Option` by the forward, so
//!   with the switch unset every hook is one `is_none` check and no clock is
//!   ever read. It buffers its own rows and never touches the shared
//!   accumulator until [`flush`].
//! - [`report`] prints ONE line for the whole forward — every GDN layer's rows
//!   folded together — and clears the accumulator, so a line is one forward's
//!   cost and consecutive lines are consecutive tokens.
//!
//! The brackets are not free, and at 36 layers x 9 steps they are not small
//! either: a token's brackets alone cost several times the mixer they measure.
//! So the block opens by measuring the bracket cost twice, and the line reports
//! every step both raw and CORRECTED:
//!
//! - [`Step::SyncFloor`] is an EMPTY bracket — a sync with nothing dispatched
//!   since the last one. On this machine it prices at ~1 us: committing an
//!   empty command buffer and waiting for it costs nothing worth naming.
//! - [`Step::DispatchFloor`] is that same bracket around ONE trivial kernel
//!   (an affine over a single element). It prices at ~0.17 ms at decode, and
//!   THAT is the number a step carries: the fixed round trip of committing a
//!   command buffer that has work in it and waiting for the GPU to schedule and
//!   finish it.
//!
//! The probe is not trusted on its own. No real step can cost less than the
//! floor, so the cheapest real step per call bounds it from above, and the
//! correction uses whichever of the two is smaller — the line reports both
//! (`floor=` and `probe=`) so a disagreement is visible. They agree at decode;
//! during prefill the probe reads several times high.
//!
//! So the correction subtracts the dispatch floor, once per synced close, and
//! the two floors together say why: it is the work-carrying commit that costs,
//! not the sync. The raw figures stay beside the corrected ones, because a step
//! that is ALL floor (`ba_head` at decode, whose kernel moves ~1 KB) is a step
//! whose corrected figure is noise, and only the raw pair shows that.
//!
//! A step whose GPU work is under that floor cannot be resolved by one
//! bracket, and at decode most of a DeltaNet block's steps are. `XWEN_GDN_REPS`
//! is the answer: each step runs its work N times inside its own bracket and
//! the line divides by N, so the floor is paid once and amortized N ways. The
//! repeated work is pure — the cache advance is in `glue` and is never repeated
//! — so the extra rounds recompute and discard.
//!
//! One thing the correction cannot give back: every step here runs alone. In a
//! real forward the block's independent dispatches — the `beta|alpha` and gate
//! projections against the qkv/conv chain — overlap on the GPU, so the
//! corrected sum is an upper bound on the block's true cost and the difference
//! from `stack_profile`'s `mixer_delta` is roughly what that overlap is worth.
//!
//! Byte counts are the caller's: each step declares the bytes it must move at
//! this geometry (weight planes at their stored width, state read and written,
//! activations where they are not noise), and the line divides them by the
//! measured time. A step's `GB/s` is therefore an ACHIEVED rate against a
//! declared floor — it says how close a step runs to being bandwidth-bound, not
//! what the hardware peak is (this machine's peak has never been measured; see
//! CLAUDE.md).
//!
//! Diagnosis only: no arithmetic changes, and a forward that errors prints
//! nothing, because the line it would print would be missing whichever step
//! failed.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::Result;
use candle_core::{Device, Tensor};

use crate::host_log::host_line;

/// One step of a DeltaNet block's forward. Ordered as the fused path runs them;
/// the report prints them in this order and omits the ones no block ran (the
/// fused path never runs `QkNorm`, whose work lives inside the scan kernel's
/// load stage; the reference path runs every one).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Step {
    /// `attn_qkv`: the fused q|k|v projection.
    QkvProj,
    /// The fused causal depthwise conv, its silu, and the next conv window —
    /// one kernel on the fused path, a cat + per-tap broadcast chain + silu on
    /// the reference one.
    Conv,
    /// The q/k L2 clamp-norm and the K-head tile-up, where they are their own
    /// dispatches. REFERENCE PATH ONLY: the fused scan kernel normalizes and
    /// tiles in its load stage, so a fused block leaves this at zero.
    QkNorm,
    /// The fused `beta|alpha` projection (one f32 gemv over `[hidden, 2 *
    /// v_heads]`), or the reference's two separate ones.
    BaProj,
    /// `beta = sigmoid(..)` and the log decay `g = ssm_a * softplus(..)` from
    /// that projection's output.
    BaHead,
    /// `attn_gate`: the z gate projection.
    GateProj,
    /// The delta-rule recurrence itself.
    Scan,
    /// The gated output RMSNorm, including the z activation.
    GnormZgate,
    /// `ssm_out`: back to the residual width.
    OutProj,
    /// Host time between two steps, when the device is idle: the cache's state
    /// handoff, the reshapes and views, the tensor drops that free device
    /// buffers. Not a position in the block — every gap, summed.
    Glue,
    /// One EMPTY bracket per block: a sync closed with no work dispatched since
    /// the previous one. Not a step of the block — the profiler measuring its
    /// own cost. Near zero, which is the point: it says the sync is not what
    /// the brackets cost.
    SyncFloor,
    /// One bracket per block around a single trivial dispatch. This is the
    /// floor every other device step's total carries — the round trip of a
    /// command buffer that has work in it — and what the corrected figures
    /// subtract. At 324 such closes a token, an uncorrected line says far more
    /// about command-buffer latency than about the mixer.
    DispatchFloor,
}

const STEPS: [Step; 12] = [
    Step::QkvProj,
    Step::Conv,
    Step::QkNorm,
    Step::BaProj,
    Step::BaHead,
    Step::GateProj,
    Step::Scan,
    Step::GnormZgate,
    Step::OutProj,
    Step::Glue,
    Step::SyncFloor,
    Step::DispatchFloor,
];

impl Step {
    fn index(self) -> usize {
        match self {
            Step::QkvProj => 0,
            Step::Conv => 1,
            Step::QkNorm => 2,
            Step::BaProj => 3,
            Step::BaHead => 4,
            Step::GateProj => 5,
            Step::Scan => 6,
            Step::GnormZgate => 7,
            Step::OutProj => 8,
            Step::Glue => 9,
            Step::SyncFloor => 10,
            Step::DispatchFloor => 11,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Step::QkvProj => "qkv_proj",
            Step::Conv => "conv",
            Step::QkNorm => "qk_norm",
            Step::BaProj => "ba_proj",
            Step::BaHead => "ba_head",
            Step::GateProj => "gate_proj",
            Step::Scan => "scan",
            Step::GnormZgate => "gnorm_zgate",
            Step::OutProj => "out_proj",
            Step::Glue => "glue",
            Step::SyncFloor => "sync_floor",
            Step::DispatchFloor => "dispatch_floor",
        }
    }
}

/// One step's running total across every GDN layer of the open forward.
#[derive(Clone, Copy, Default)]
struct Bucket {
    total: Duration,
    /// Times the step was closed. On the fused path each close is exactly one
    /// kernel dispatch, so this doubles as the step's dispatch count; on the
    /// reference path a close covers a whole candle chain and counts as one.
    calls: u64,
    /// How many of those closes synced, which is how many sync floors the
    /// step's total carries. Only `Glue` differs from `calls`: it is closed
    /// once on the host and once on the device per block.
    syncs: u64,
    /// Bytes the step declared it had to move, summed over those calls.
    bytes: u64,
}

/// What the open forward's GDN layers have accumulated so far.
#[derive(Clone, Default)]
struct Acc {
    buckets: [Bucket; STEPS.len()],
    /// GDN blocks folded in since the last report.
    blocks: u64,
    /// Tokens the blocks were fed. One forward feeds every layer the same
    /// count, so this is the forward's `n`; a disagreement would mean the
    /// accumulator spans two forwards and is recorded as the maximum rather
    /// than silently averaged.
    tokens: usize,
    /// Whether any folded block took the reference (`XWEN_DELTA_CLASSIC`) path,
    /// so a line cannot be mistaken for the fused one it does not describe.
    classic: bool,
    /// The repeat count the folded blocks ran at, which the report divides the
    /// corrected figures by.
    reps: usize,
}

static ACC: Mutex<Option<Acc>> = Mutex::new(None);

/// One closed step: what it was, how long it took, the bytes it declared, and
/// whether closing it cost a sync.
struct Row {
    step: Step,
    elapsed: Duration,
    bytes: u64,
    synced: bool,
}

/// One block call's steps, buffered until [`flush`].
///
/// Rows are held locally rather than written straight through to [`ACC`] so a
/// block's 9 steps take one lock instead of nine — the profiler's own cost
/// stays out of the `glue` bucket it is trying to measure.
pub struct BlockProfile {
    /// Start of the open step. Every interval since the previous step's close
    /// belongs to the next step recorded, so the steps sum to the bracket.
    mark: Instant,
    rows: Vec<Row>,
    tokens: usize,
    classic: bool,
    /// Times each step's work is repeated inside its bracket
    /// (`ops::gdn_reps`). The floor brackets are NOT repeated — one of them is
    /// what the repetition exists to divide away.
    reps: usize,
}

impl BlockProfile {
    /// Open the bracket from an idle device, or return `None` when the switch
    /// is unset. `tokens` is the block's `seq`, `classic` says which path is
    /// about to run, and `probe` is any small f32 device tensor the two floor
    /// brackets can dispatch a throwaway kernel over (the block hands over one
    /// of its own `[v_heads]` vectors — no allocation of its own, and nothing
    /// downstream reads the result).
    pub fn start(
        device: &Device,
        tokens: usize,
        classic: bool,
        probe: &Tensor,
    ) -> Result<Option<Self>> {
        if !crate::ops::gdn_profile() {
            return Ok(None);
        }
        device.synchronize()?;
        let mut p = Some(Self {
            mark: Instant::now(),
            rows: Vec::with_capacity(STEPS.len()),
            tokens,
            classic,
            // The reference path is never repeated: its scan mutates the state
            // as it walks the tokens, so a second round would not recompute the
            // same thing. It reads one-shot, floor and all.
            reps: if classic { 1 } else { crate::ops::gdn_reps() },
        });
        // The empty bracket, taken here because this is the one place in the
        // block where nothing can have been dispatched since the last sync: a
        // second `synchronize` with no work in between prices the call itself.
        step(&mut p, device, Step::SyncFloor, 0)?;
        // The same bracket around one trivial dispatch. The difference between
        // the two is what a work-carrying commit costs, and it is that
        // difference — not the sync — every other step's total carries.
        let _probe = probe.narrow(0, 0, 1)?.affine(1.0, 1.0)?;
        step(&mut p, device, Step::DispatchFloor, 0)?;
        Ok(p)
    }
}

/// Close a step that dispatched device work, waiting for it first. `bytes` is
/// what that step had to move at this geometry.
pub fn step(p: &mut Option<BlockProfile>, device: &Device, s: Step, bytes: u64) -> Result<()> {
    if p.is_some() {
        device.synchronize()?;
        push(p, s, bytes, true);
    }
    Ok(())
}

/// Close a step that dispatched nothing — no sync, because there is nothing to
/// wait for, and so no sync floor in its total either.
pub fn host_step(p: &mut Option<BlockProfile>, s: Step, bytes: u64) {
    push(p, s, bytes, false);
}

fn push(p: &mut Option<BlockProfile>, step: Step, bytes: u64, synced: bool) {
    if let Some(p) = p.as_mut() {
        let now = Instant::now();
        p.rows.push(Row {
            step,
            elapsed: now.duration_since(p.mark),
            bytes,
            synced,
        });
        p.mark = now;
    }
}

/// Run `f` the profile's repeat count times and return the LAST result, or
/// exactly once when the profile is off.
///
/// Only ever wrapped around work that is a pure function of tensors the block
/// already holds: the repeated rounds must be discardable, and anything that
/// advances the layer cache is not. With the switch unset this is one
/// `is_none` check and one call, which is what it always was.
pub fn rep<T>(p: &Option<BlockProfile>, mut f: impl FnMut() -> Result<T>) -> Result<T> {
    let reps = p.as_ref().map_or(1, |p| p.reps);
    let mut out = f()?;
    for _ in 1..reps {
        out = f()?;
    }
    Ok(out)
}

/// Fold one block's rows into the forward's accumulator.
pub fn flush(p: Option<BlockProfile>) {
    let Some(p) = p else { return };
    let Ok(mut guard) = ACC.lock() else { return };
    let acc = guard.get_or_insert_with(Acc::default);
    for r in &p.rows {
        let b = &mut acc.buckets[r.step.index()];
        b.total += r.elapsed;
        b.calls += 1;
        b.syncs += u64::from(r.synced);
        b.bytes += r.bytes;
    }
    acc.blocks += 1;
    acc.tokens = acc.tokens.max(p.tokens);
    acc.classic |= p.classic;
    acc.reps = p.reps;
}

/// Print the forward's line and clear the accumulator. A forward that ran no
/// GDN layer prints nothing; so does a run with the switch unset, which never
/// fills the accumulator in the first place.
pub fn report() {
    if !crate::ops::gdn_profile() {
        return;
    }
    let Ok(mut guard) = ACC.lock() else { return };
    let Some(acc) = guard.take() else { return };
    if acc.blocks == 0 {
        return;
    }
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let floor_bucket = acc.buckets[Step::DispatchFloor.index()];
    // Seconds one work-carrying commit-and-wait costs, from the single-dispatch
    // bracket every block closed. Zero when nothing measured it, in which case
    // the corrected figures are the raw ones and the line says so by reporting
    // floor=0.
    let probe = if floor_bucket.calls > 0 {
        floor_bucket.total.as_secs_f64() / floor_bucket.calls as f64
    } else {
        f64::INFINITY
    };
    // No real step can run BELOW the floor — every one of them is the floor
    // plus its own work — so the cheapest real step per call is an upper bound
    // on it, and the floor is the smaller of the two estimates. They agree at
    // decode; during prefill the probe reads several times high (the allocator
    // is walking a pool full of chunk-sized buffers by then), and without this
    // the correction would subtract more than a step's whole cost and clamp
    // half the line to zero.
    let cheapest = STEPS
        .iter()
        .filter(|s| **s != Step::SyncFloor && **s != Step::DispatchFloor && **s != Step::Glue)
        .map(|s| acc.buckets[s.index()])
        .filter(|b| b.syncs > 0)
        .map(|b| b.total.as_secs_f64() / b.syncs as f64)
        .fold(f64::INFINITY, f64::min);
    let floor = probe.min(cheapest);
    let floor = if floor.is_finite() { floor } else { 0.0 };
    // Net of the floor each step's own syncs contributed and divided by the
    // repeat count, never below zero: a step measured under its floor is noise,
    // not negative work. `glue` is never repeated, so it divides by one.
    let reps = acc.reps.max(1) as f64;
    let corrected = |s: Step, b: Bucket| {
        let per = if s == Step::Glue { 1.0 } else { reps };
        ((b.total.as_secs_f64() - floor * b.syncs as f64) / per).max(0.0)
    };
    // Both totals span the block's real steps only; the two floor brackets are
    // the profiler's own cost and are reported as `floor`, never as work.
    let steps = || {
        STEPS
            .iter()
            .filter(|s| **s != Step::SyncFloor && **s != Step::DispatchFloor)
    };
    let raw: Duration = steps().map(|s| acc.buckets[s.index()].total).sum();
    let corr_total: f64 = steps().map(|s| corrected(*s, acc.buckets[s.index()])).sum();
    let sync_floor = acc.buckets[Step::SyncFloor.index()];
    let sync_us = if sync_floor.calls > 0 {
        sync_floor.total.as_secs_f64() / sync_floor.calls as f64 * 1e6
    } else {
        0.0
    };
    let mut line = format!(
        "xwen: gdn-profile n={} blocks={} path={} corr={:.3}ms raw={:.3}ms floor={:.3}ms/dispatch reps={}",
        acc.tokens,
        acc.blocks,
        if acc.classic { "classic" } else { "fused" },
        corr_total * 1e3,
        ms(raw),
        floor * 1e3,
        acc.reps.max(1),
    );
    line.push_str(&format!(" sync={sync_us:.1}us probe={:.3}ms", probe * 1e3));
    for s in STEPS {
        let b = acc.buckets[s.index()];
        if b.calls == 0 || s == Step::SyncFloor || s == Step::DispatchFloor {
            continue;
        }
        let c = corrected(s, b);
        line.push_str(&format!(
            " {}={:.3}ms(raw {:.3})/d{}",
            s.name(),
            c * 1e3,
            ms(b.total),
            b.calls
        ));
        // A step whose corrected time is at or under a microsecond is a step
        // that was entirely floor; a rate computed from it is division noise,
        // not a bandwidth.
        if b.bytes > 0 && c > 1e-6 {
            line.push_str(&format!("/{:.0}GB/s", b.bytes as f64 / c / 1e9));
        }
    }
    host_line(line);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block's rows land in the accumulator under their own steps, and the
    /// report clears it — so the next forward's line is that forward's cost and
    /// not a running total.
    #[test]
    fn a_flushed_block_accumulates_and_a_report_clears_it() {
        // Not the env-gated entry points: those are process-wide and this test
        // must not depend on how the binary was launched.
        let mut p = Some(BlockProfile {
            mark: Instant::now(),
            rows: Vec::new(),
            tokens: 4,
            classic: false,
            reps: 1,
        });
        host_step(&mut p, Step::QkvProj, 100);
        host_step(&mut p, Step::Scan, 200);
        flush(p);

        let guard = ACC.lock().unwrap();
        let acc = guard
            .as_ref()
            .expect("a flushed block fills the accumulator");
        assert_eq!(acc.blocks, 1);
        assert_eq!(acc.tokens, 4);
        assert_eq!(acc.buckets[Step::QkvProj.index()].calls, 1);
        assert_eq!(acc.buckets[Step::QkvProj.index()].bytes, 100);
        assert_eq!(acc.buckets[Step::Scan.index()].bytes, 200);
        assert_eq!(acc.buckets[Step::Conv.index()].calls, 0);
        drop(guard);

        // `report` is env-gated, so clear the accumulator the way it does
        // rather than depending on the switch being set under `cargo test`.
        ACC.lock().unwrap().take();
        assert!(ACC.lock().unwrap().is_none());
    }

    /// With the switch unset `start` hands back no profile, and every hook on
    /// that `None` is inert: nothing is recorded, and `step` returns without
    /// ever reaching the device it was handed (a CPU device here, which cannot
    /// serve the sync a live profile would ask for). The accumulator is
    /// process-wide and shared with the sibling test, so this asserts on the
    /// hooks rather than on its contents.
    #[test]
    fn the_hooks_are_inert_without_a_profile() {
        let mut p: Option<BlockProfile> = None;
        host_step(&mut p, Step::QkvProj, 100);
        step(&mut p, &Device::Cpu, Step::Scan, 200).unwrap();
        flush(p);
    }
}
