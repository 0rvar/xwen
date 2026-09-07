//! Per-stage timing for one image run, behind [`PROFILE_ENV`].
//!
//! Metal work is queued asynchronously, so a stage's cost lands on whichever
//! later stage happens to wait for it. Every [`Profiler::mark`] therefore
//! synchronizes the device before it reads the clock, which is what makes a
//! row mean the stage it names, and what makes a profiled run slower than
//! the run it describes. Nothing holds a profiler unless [`PROFILE_ENV`]
//! asked for one, and without one a call site pays a single `Option` check.
//!
//! Buckets accumulate by label, so the 34 blocks of one transformer forward
//! sum into one row per stage rather than 34 tables. [`Profiler::set_phase`]
//! is how the refiner loops, which run the same block code as the main
//! layers, keep their own rows.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::{Device, Result};

/// The environment switch that asks for a profile, read when a
/// `ZImagePipeline` loads.
pub const PROFILE_ENV: &str = "XWEN_ZIMAGE_PROFILE";

/// Whether [`PROFILE_ENV`] asks for a profiler. Unset is off; a value that
/// names neither state is a load error rather than a silent unprofiled run,
/// the way the kernel-arm switches beside it are.
pub fn from_env() -> Result<bool> {
    match std::env::var(PROFILE_ENV) {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(_)) => {
            candle_core::bail!("{PROFILE_ENV} is not valid UTF-8")
        }
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "off" | "false" | "no" => Ok(false),
            "1" | "on" | "true" | "yes" => Ok(true),
            other => candle_core::bail!("{PROFILE_ENV}={other:?}: expected `1` or `0`"),
        },
    }
}

/// Accumulated device time per named stage of one run.
#[derive(Debug)]
pub struct Profiler {
    device: Device,
    state: Mutex<State>,
}

#[derive(Debug)]
struct State {
    /// When the stage now running started, device-synchronized.
    last: Instant,
    /// The prefix the next marks carry.
    phase: &'static str,
    /// `(phase, label, seconds)` in the order the stages were first seen,
    /// which is the order they run.
    buckets: Vec<(&'static str, &'static str, f64)>,
}

impl Profiler {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            state: Mutex::new(State {
                last: Instant::now(),
                phase: "",
                buckets: Vec::new(),
            }),
        }
    }

    /// Attribute the time since the previous mark to `label` under the
    /// current phase.
    pub fn mark(&self, label: &'static str) {
        // A failed synchronize leaves this mark meaningless, and the work it
        // was waiting on reports the failure itself on the next op that
        // touches the tensor. Timing is not the place to raise it.
        let _ = self.device.synchronize();
        let now = Instant::now();
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let elapsed = now.duration_since(state.last).as_secs_f64();
        state.last = now;
        let phase = state.phase;
        match state
            .buckets
            .iter_mut()
            .find(|(p, l, _)| *p == phase && *l == label)
        {
            Some(bucket) => bucket.2 += elapsed,
            None => state.buckets.push((phase, label, elapsed)),
        }
    }

    /// The prefix the marks that follow carry, so two loops over the same
    /// block code keep separate rows. `""` is the unprefixed default.
    pub fn set_phase(&self, phase: &'static str) {
        if let Ok(mut state) = self.state.lock() {
            state.phase = phase;
        }
    }

    /// Drop every bucket and re-anchor the clock, which is how one phase's
    /// table is kept out of the next one's.
    pub fn reset(&self) {
        let _ = self.device.synchronize();
        if let Ok(mut state) = self.state.lock() {
            state.buckets.clear();
            state.phase = "";
            state.last = Instant::now();
        }
    }

    /// `(label, seconds)` per stage, in the order the stages run.
    pub fn report(&self) -> Vec<(String, f64)> {
        let Ok(state) = self.state.lock() else {
            return Vec::new();
        };
        state
            .buckets
            .iter()
            .map(|(phase, label, secs)| (format!("{phase}{label}"), *secs))
            .collect()
    }
}

/// [`Profiler::mark`] through an `Option`: the shape every instrumented call
/// site takes, so an unprofiled run pays one branch.
pub fn mark(profiler: &Option<Arc<Profiler>>, label: &'static str) {
    if let Some(profiler) = profiler {
        profiler.mark(label);
    }
}

/// [`Profiler::set_phase`] through an `Option`.
pub fn set_phase(profiler: &Option<Arc<Profiler>>, phase: &'static str) {
    if let Some(profiler) = profiler {
        profiler.set_phase(phase);
    }
}

/// Print one profile table to stderr. `runs` is what the totals are divided
/// by, so a transformer table over several steps reads as ms per step; shares
/// are of the table's own total, which is the phase the table covers and not
/// the whole run.
pub fn print_table(title: &str, rows: &[(String, f64)], runs: usize) {
    if rows.is_empty() {
        return;
    }
    let total: f64 = rows.iter().map(|(_, secs)| *secs).sum();
    let width = rows.iter().map(|(label, _)| label.len()).max().unwrap_or(5);
    let runs = runs.max(1) as f64;
    eprintln!("xwen: profile, {title}");
    for (label, secs) in rows {
        let share = if total > 0.0 {
            secs / total * 100.0
        } else {
            0.0
        };
        eprintln!(
            "xwen:   {label:<width$}  {:9.2} ms  {share:5.1}%",
            secs * 1000.0 / runs
        );
    }
    eprintln!(
        "xwen:   {:<width$}  {:9.2} ms",
        "total",
        total * 1000.0 / runs
    );
}
