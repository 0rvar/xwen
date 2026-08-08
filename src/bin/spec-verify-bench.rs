//! Microbench for the speculative-verify forward: checkpoint → span forward
//! (`forward_all_logits`) → readback → rollback, at a sweep of span lengths,
//! over real fixture tokens. This is the exact per-round verify operation
//! generate_spec performs, so it isolates the cost the drafter must amortize —
//! and, run twice with `XWEN_MM_ID_MIN_SEQ` toggled, it measures the mv_id/mm_id
//! crossover for short spans (mm_id's per-expert compaction is the "expert
//! union" dedup; mv_id re-reads each routed expert per token).
//!
//! Two numbers per span, from two separate sets of reps:
//!
//! - The AGGREGATE (`ms/verify`): one `Instant` around a whole rep loop with no
//!   inner syncs. The shape the round-cost fits were made against, kept so a new
//!   run is comparable to an old one.
//! - The STAGES: the same four operations, each bracketed by device syncs and
//!   timed separately, reported as a median over the reps. Syncs serialize what
//!   would otherwise pipeline, so the stage sum runs above the aggregate; the
//!   split between stages is the point, not the total.
//!
//! Flags exist to attribute the fixed cost to its parts: `--no-checkpoint` runs
//! the forward UNARMED (no K-snapshot trail built inside the DeltaNet mixers),
//! `--keep` selects which rollback branch runs, and `--readback lastrow` moves
//! one row instead of the whole span.
//!
//! `XWEN_STACK_PROFILE` goes one level deeper: the forward stage above splits
//! into the stack profiler's own buckets, dumped per span over the timed reps
//! only (warm-up and the aggregate loop are reset away). Those dumps are
//! host lines on stderr, so they interleave with the stdout table rather than
//! corrupting it.
//!
//! Usage:
//!   spec-verify-bench --model <gguf> [--n-past 512] [--reps 20]
//!   XWEN_STACK_PROFILE=1 spec-verify-bench --model <gguf>     # per-span stage dump
//!   spec-verify-bench --model <gguf> --no-checkpoint          # unarmed forward
//!   spec-verify-bench --model <gguf> --keep 4                 # trail-entry rollback
//!   spec-verify-bench --model <gguf> --readback lastrow       # one row to the host
//!   XWEN_MM_ID_MIN_SEQ=2 spec-verify-bench --model <gguf>     # force mm_id

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use clap::{Parser, ValueEnum};

use xwen::XwenConfig;
use xwen::gguf;
use xwen::kv_cache::CacheSnapshot;
use xwen::model::XwenModel;
use xwen::ops::ExpertRunner;
use xwen::stack_profile::Phase;

/// Span lengths swept. Covers a real draft block (up to 16) and beyond, so the
/// fixed cost and the per-token cost separate.
const SPANS: [usize; 9] = [2, 4, 6, 8, 12, 16, 24, 32, 48];

/// How much of the verify forward's `[span, vocab]` logits the round copies to
/// the host.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Readback {
    /// The whole `[span, vocab]` block — what generate_spec does, because the
    /// accept walk samples every row.
    Full,
    /// The last row only. A round that accepted nothing still needs this one, so
    /// it is the floor; against `full` it separates the readback's byte cost from
    /// its fixed sync cost.
    Lastrow,
}

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    model: PathBuf,
    /// Tokens prefilled before the verify sweep (cache depth during the bench;
    /// the deeper it is, the closer the verify is to a real mid-generation one —
    /// both the KV cache it attends over and the DeltaNet state it carries).
    #[arg(long, default_value_t = 512)]
    n_past: usize,
    #[arg(long, default_value_t = 20)]
    reps: usize,
    /// Run the forward with no checkpoint and no rollback, leaving the DeltaNet
    /// layers UNARMED. An armed layer builds the K-snapshot trail inside the
    /// mixer (one state plane per token of the span), so the difference against
    /// a default run is what arming costs inside the forward itself, separate
    /// from what the checkpoint and rollback cost around it.
    #[arg(long)]
    no_checkpoint: bool,
    /// Tokens the rollback keeps, i.e. the accepted-draft count a real round
    /// would commit. 0 restores the checkpoint's own state (the cheap branch);
    /// anything higher restores from the trail entry that token wrote. Must be
    /// below the span, so spans at or under it are skipped.
    #[arg(long, default_value_t = 0)]
    keep: usize,
    /// How much of the logits block travels to the host.
    #[arg(long, value_enum, default_value_t = Readback::Full)]
    readback: Readback,
}

/// One rep's four stage times, in milliseconds. A stage that did not run is 0.0.
#[derive(Default, Clone, Copy)]
struct RepTimes {
    ckpt: f64,
    fwd: f64,
    read: f64,
    roll: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    // Checked before the model load: this bench holds a 20 GB checkpoint and the
    // GPU for the whole run, so a bad flag combination must not cost a load first.
    anyhow::ensure!(
        !(args.no_checkpoint && args.keep > 0),
        "--keep {} has nothing to act on under --no-checkpoint (no rollback runs)",
        args.keep
    );
    anyhow::ensure!(args.reps >= 1, "--reps must be at least 1");

    let device = gguf::metal_device()?;
    let file = gguf::open(&args.model, &device)?;
    let _cfg = XwenConfig::from_gguf(&file.content)?;
    let mut model = XwenModel::load(file, ExpertRunner::Fused, 4096)?;
    // Under XWEN_STACK_PROFILE the verify forwards decompose into the stack
    // profiler's buckets. They are decode-shaped — a verify span is tokens the
    // generation already committed to, not prompt — so the phase is declared
    // once here rather than left at the prefill default.
    if xwen::ops::stack_profile() {
        model.set_phase(Phase::Decode);
    }

    let fixture: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/parity-prompts.json"
    ))?)?;
    let tokens: Vec<u32> = fixture["prompts"]
        .as_array()
        .context("prompts array")?
        .iter()
        .find(|p| p["id"] == "long-mixed")
        .context("long-mixed fixture")?["tokens"]
        .as_array()
        .context("tokens array")?
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect();
    let max_span = SPANS[SPANS.len() - 1];
    anyhow::ensure!(tokens.len() >= args.n_past + max_span, "fixture too short");

    eprintln!(
        "effective mm_id_min_seq = {} (default {})",
        xwen::ops::mm_id_min_seq(),
        xwen::ops::MM_ID_MIN_SEQ
    );
    eprintln!(
        "checkpoint={} keep={} readback={} reps={} (staged reps first, then aggregate reps)",
        if args.no_checkpoint { "off" } else { "on" },
        args.keep,
        match args.readback {
            Readback::Full => "full",
            Readback::Lastrow => "lastrow",
        },
        args.reps,
    );

    // Prefill the context the sweep runs on top of (also serves as the
    // steady-state warm-up: weights resident, pipelines compiled).
    let ctx = candle_core::Tensor::from_vec(tokens[..args.n_past].to_vec(), args.n_past, &device)?;
    let _ = model.forward(&ctx, 0)?;
    device.synchronize()?;

    // Every rep has to start from the same cache state. A default rep ends there
    // on its own (rollback to commit 0 undoes the span), but an unarmed rep
    // leaves the span appended and a `--keep n` rep leaves n tokens committed, so
    // those are rewound from this snapshot — outside the timed brackets.
    let base = model.take_cache_snapshot()?;

    println!("span\tms/verify\ttok/s-equiv");
    for span in SPANS {
        if args.keep >= span {
            eprintln!(
                "skipping span {span}: --keep {} needs keep < span",
                args.keep
            );
            continue;
        }
        let ids = tokens[args.n_past..args.n_past + span].to_vec();
        let vinput = candle_core::Tensor::from_vec(ids, span, &device)?;

        // Warm-up reps outside the timed window, in the shape the timed ones run.
        for _ in 0..3 {
            staged_rep(&mut model, &vinput, span, &base, &args, &device)?;
        }
        device.synchronize()?;
        // The warm-up's forwards went through the profiler too; drop them so the
        // span's dump describes the timed reps alone.
        if xwen::ops::stack_profile() {
            model.reset_stack_profile();
        }

        let mut times = Vec::with_capacity(args.reps);
        for _ in 0..args.reps {
            times.push(staged_rep(
                &mut model, &vinput, span, &base, &args, &device,
            )?);
        }
        // Dumped before the aggregate loop runs: its reps go through the same
        // forwards and would fold into the same accumulators. The reset after the
        // dump discards them, so the next span starts clean either way.
        if xwen::ops::stack_profile() {
            model.dump_stack_profile(&format!("span-{span}/{}reps", args.reps));
            model.reset_stack_profile();
        }
        let ckpt = median(times.iter().map(|t| t.ckpt));
        let fwd = median(times.iter().map(|t| t.fwd));
        let read = median(times.iter().map(|t| t.read));
        let roll = median(times.iter().map(|t| t.roll));

        // The aggregate runs only in the shape the historical fits used: a
        // checkpoint/rollback rep loop that ends where it started, so nothing has
        // to rewind the cache inside the timed window. The other modes have no
        // comparable number to preserve, and faking one by timing a rewind along
        // with the round would report a cost no round pays.
        let aggregate = if !args.no_checkpoint && args.keep == 0 {
            device.synchronize()?;
            let t0 = Instant::now();
            for _ in 0..args.reps {
                let ckpt = model.kv_checkpoint(span)?;
                // Match generate_spec's real readback: the sampler walks CPU logits.
                let logits = model.forward_all_logits(&vinput, args.n_past)?;
                let _cpu = readback(&logits, span, args.readback)?;
                model.kv_rollback(&ckpt, 0)?;
            }
            device.synchronize()?;
            Some(t0.elapsed().as_secs_f64() * 1000.0 / args.reps as f64)
        } else {
            None
        };

        match aggregate {
            Some(ms) => println!("{span}\t{ms:.2}\t{:.1}", span as f64 / (ms / 1000.0)),
            None => println!("{span}\t-\t-"),
        }
        let cell = |ran: bool, ms: f64| {
            if ran {
                format!("{ms:.2}")
            } else {
                "-".to_string()
            }
        };
        let armed = !args.no_checkpoint;
        println!(
            "  stage\tspan={span}\tckpt={}\tfwd={fwd:.2}\tread={read:.2}\troll={}\tsum={:.2}\taggregate={}",
            cell(armed, ckpt),
            cell(armed, roll),
            ckpt + fwd + read + roll,
            match aggregate {
                Some(ms) => format!("{ms:.2}"),
                None => "-".to_string(),
            },
        );
    }
    Ok(())
}

/// One verify rep with each stage bracketed by device syncs, returning the four
/// stage times in milliseconds. Leaves the cache exactly where it found it.
fn staged_rep(
    model: &mut XwenModel,
    vinput: &Tensor,
    span: usize,
    base: &CacheSnapshot,
    args: &Args,
    device: &Device,
) -> Result<RepTimes> {
    let mut t = RepTimes::default();
    device.synchronize()?;

    let ckpt = if args.no_checkpoint {
        None
    } else {
        let start = Instant::now();
        let ckpt = model.kv_checkpoint(span)?;
        device.synchronize()?;
        t.ckpt = ms(start);
        Some(ckpt)
    };

    let start = Instant::now();
    let logits = model.forward_all_logits(vinput, args.n_past)?;
    device.synchronize()?;
    t.fwd = ms(start);

    let start = Instant::now();
    let cpu = readback(&logits, span, args.readback)?;
    device.synchronize()?;
    t.read = ms(start);
    drop(cpu);
    drop(logits);

    if let Some(ckpt) = &ckpt {
        let start = Instant::now();
        model.kv_rollback(ckpt, args.keep)?;
        device.synchronize()?;
        t.roll = ms(start);
    }

    // Untimed: put the cache back for the next rep. A commit-0 rollback already
    // did, so the default path never reaches the restore.
    if model.cache_len() != args.n_past {
        model.restore_cache_snapshot(base)?;
        device.synchronize()?;
    }
    Ok(t)
}

/// Copy the logits a round's sampler would read to the host.
fn readback(logits: &Tensor, span: usize, mode: Readback) -> Result<Tensor> {
    let src = match mode {
        Readback::Full => logits.clone(),
        Readback::Lastrow => {
            // A narrow is a view: it carries a start offset over storage that is
            // still the whole `[span, vocab]` block, and `to_device` copies the
            // storage entire and re-applies the layout. So the row has to own its
            // allocation before it crosses to the host, or "one row" would still
            // move every row's bytes.
            let row = logits.narrow(0, span - 1, 1)?;
            let owned = row.zeros_like()?;
            owned.slice_set(&row, 0, 0)?;
            owned
        }
    };
    Ok(src.to_dtype(DType::F32)?.to_device(&Device::Cpu)?)
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// Median of a rep's samples. Reps are few and the distribution is right-skewed
/// (a stray scheduler or thermal excursion lands on one rep), which is what the
/// median is here to ignore.
fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = values.collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("stage times are never NaN"));
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}
