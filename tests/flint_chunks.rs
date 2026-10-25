//! Flint's forward does not depend on how a prompt is chunked, up to the first
//! router.
//!
//! The same 2200-token prompt runs once in one chunk and again under every
//! chunking a forward can see: single rows, a small odd chunk, chunks one
//! under, at and one over the 513-slot sliding window, the shipped 2048, and
//! one-token decode from position 400 to the end (across the ring's wrap,
//! three times). Each schedule must reproduce the one-shot run's hidden states
//! at layers 0 to 2, row by row (relative L2 per position, worst position).
//!
//! Why only up to layer 2's router input (`ffn_norm-2`). Layers 0 and 1 are
//! dense and have no router; layer 2 is the first sliding layer, so its
//! attention reads the ring, the rope at the chunk's offset and the sliding
//! mask, which is what a chunking bug breaks. Past the first router the logits
//! are NOT chunk-invariant on this model even when every arm runs the f32
//! reference path (measured 2026-10-25: a 300-token chunk against one 700-token
//! chunk first moves a logit by 0.56 at position 45, and the median position by
//! 0.7). The matmuls of a different chunk shape round differently in the last
//! place, a sigmoid router sitting on a near tie flips an expert, and
//! attention carries the changed position into every later one.
//!
//! Two tolerances, because a single row and a chunk do not run the same
//! precision class. A chunk of two or more rows projects through a matmul that
//! stages the weight tile as f16, and Flint's dense lead MLP turns that into a
//! large error by cancellation: its output sits 3.7% (mean, per row) from the
//! exact f64 result of its own input, where the single-row gemv, which
//! dequantizes to f32, sits 2e-7 from it (2026-10-25). So every multi-row
//! schedule shares the one-shot run's class and is graded tightly on every tap
//! ([`TOL_CHUNKED`], measured worst 1.7e-3 at `attn_o_proj-2`), while a
//! schedule with single-row forwards is graded tightly only up to the first
//! MLP ([`TOL_SINGLE_ROW_ATTN`] through `ffn_norm-0`, measured worst 8.7e-4)
//! and only grossly after it ([`TOL_SINGLE_ROW`], measured worst 0.18). The
//! single-row ring path is held to the definition exactly by
//! `flint::attention`'s ring test, and end to end by the oracle gate's
//! decode phase.
//!
//! What this cannot see is a window that is wrong the same way in every
//! schedule: the ring test holds the window to its definition.
//!
//! The run pins the classic projections (`XWEN_DENSE_MM_CLASSIC`,
//! `XWEN_MV_EXT_CLASSIC`, `XWEN_ATTN_F32`) so a chunk width selects no kernel
//! of its own beyond the gemv/matmul split above; this file is its own test
//! binary so the switches are set before anything reads them.
//!
//! ```text
//! cargo test --release --test flint_chunks -- --ignored --nocapture
//! ```
//!
//! It fails rather than skips when the checkpoint is missing.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use candle_core::Tensor;
use serde::Deserialize;
use xwen::XwenModel;
use xwen::checkpoint::CheckpointSource;
use xwen::gguf;
use xwen::hub::Model;
use xwen::ops::ExpertRunner;

/// Prompt length: past the window four times over and past one 2048 chunk.
const N: usize = 2200;

/// The taps graded, in stack order: both dense full-attention layers and the
/// first sliding layer up to its router input.
const TAPS: [&str; 12] = [
    "attn_norm-0",
    "attn_o_proj-0",
    "ffn_inp-0",
    "ffn_norm-0",
    "ffn_out-0",
    "l_out-0",
    "attn_o_proj-1",
    "l_out-1",
    "attn_norm-2",
    "attn_o_proj-2",
    "ffn_inp-2",
    "ffn_norm-2",
];

/// Worst per-position relative L2 allowed on every tap for a schedule of
/// multi-row chunks.
const TOL_CHUNKED: f32 = 4e-3;
/// The same for a schedule with single-row forwards, on the taps up to
/// `ffn_norm-0` (layer 0's attention and residual, before any MLP).
const TOL_SINGLE_ROW_ATTN: f32 = 3e-3;
/// The same for a schedule with single-row forwards, on every later tap.
const TOL_SINGLE_ROW: f32 = 0.35;
/// The taps `TOL_SINGLE_ROW_ATTN` applies to.
const BEFORE_FIRST_MLP: usize = 4;

fn long_ids() -> Result<Vec<u32>> {
    #[derive(Deserialize)]
    struct Prompts {
        prompts: Vec<Ids>,
    }
    #[derive(Deserialize)]
    struct Ids {
        id: String,
        ids: Vec<u32>,
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/flint-prompts.json");
    let p: Prompts = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let ids = p
        .prompts
        .into_iter()
        .find(|p| p.id == "bench-prefill-4k")
        .context("flint-prompts.json has no bench-prefill-4k")?
        .ids;
    ensure!(ids.len() >= N, "bench-prefill-4k has {} ids", ids.len());
    Ok(ids[..N].to_vec())
}

/// Every graded tap over the whole prompt, `[N, hidden]` row-major, for one
/// chunk schedule.
fn run(
    model: &mut XwenModel,
    dev: &candle_core::Device,
    ids: &[u32],
    size: &dyn Fn(usize) -> usize,
) -> Result<HashMap<&'static str, Vec<f32>>> {
    model.reset_cache()?;
    let mut out: HashMap<&'static str, Vec<f32>> = TAPS.iter().map(|&t| (t, Vec::new())).collect();
    let mut pos = 0;
    while pos < ids.len() {
        let take = size(pos).min(ids.len() - pos);
        let t = Tensor::new(&ids[pos..pos + take], dev)?;
        // `forward` publishes the whole chunk's taps; `forward_all_logits`
        // publishes none.
        model.forward(&t, pos)?;
        for (name, tensor) in model.take_taps() {
            if let Some(&tap) = TAPS.iter().find(|&&t| t == name) {
                let rows = tensor.to_dtype(candle_core::DType::F32)?.flatten_all()?;
                out.get_mut(tap).unwrap().extend(rows.to_vec1::<f32>()?);
            }
        }
        pos += take;
    }
    Ok(out)
}

#[test]
#[ignore = "needs a Metal device and the Flint checkpoint"]
fn flint_hidden_states_do_not_depend_on_the_chunking() -> Result<()> {
    // SAFETY: this binary's only test sets these before any thread reads them.
    unsafe {
        std::env::set_var("XWEN_DENSE_MM_CLASSIC", "1");
        std::env::set_var("XWEN_MV_EXT_CLASSIC", "1");
        std::env::set_var("XWEN_ATTN_F32", "1");
    }
    let ids = long_ids()?;
    let path = xwen::hub::cached_model(Model::Flint)
        .context("Flint is not in the Hugging Face cache; run `xwen fetch --model flint`")?;
    let dev = gguf::metal_device()?;
    let source = CheckpointSource::open(&path, &dev, Some(Model::Flint))?;
    let mut model = XwenModel::load(source, ExpertRunner::Fused, N)?;
    model.set_tap_capture(true);

    let one = run(&mut model, &dev, &ids, &|_| usize::MAX)?;
    let hidden = model.config().hidden;
    for tap in TAPS {
        ensure!(
            one[tap].len() == N * hidden,
            "{tap}: {} values",
            one[tap].len()
        );
    }

    // (name, chunk size at a position, whether any forward is a single row)
    let schedules: [(&str, &dyn Fn(usize) -> usize, bool); 7] = [
        ("1 x40 then 512", &|p| if p < 40 { 1 } else { 512 }, true),
        ("7", &|_| 7, false),
        ("512", &|_| 512, false),
        ("513", &|_| 513, false),
        ("514", &|_| 514, false),
        ("2048", &|_| 2048, false),
        ("400 then decode", &|p| if p < 400 { 400 } else { 1 }, true),
    ];
    let mut report = format!(
        "FLINT CHUNK INVARIANCE ({N} tokens; worst per-position relative L2; chunked {TOL_CHUNKED:e}, \
         single-row {TOL_SINGLE_ROW_ATTN:e} to ffn_norm-0 then {TOL_SINGLE_ROW})\n"
    );
    let mut failures = Vec::new();
    for (name, size, single_row) in schedules {
        let got = run(&mut model, &dev, &ids, size)?;
        let mut worst = (0f32, "", 0usize);
        let mut per_tap = String::new();
        for (ti, tap) in TAPS.into_iter().enumerate() {
            let mut tap_worst = (0f32, 0usize);
            for (p, (a, b)) in got[tap]
                .chunks(hidden)
                .zip(one[tap].chunks(hidden))
                .enumerate()
            {
                let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
                let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
                let rel = (num / den.max(1e-30)).sqrt() as f32;
                if !(rel <= tap_worst.0) {
                    tap_worst = (rel, p);
                }
            }
            per_tap.push_str(&format!(" {tap} {:.1e}@{}", tap_worst.0, tap_worst.1));
            let tol = match (single_row, ti < BEFORE_FIRST_MLP) {
                (false, _) => TOL_CHUNKED,
                (true, true) => TOL_SINGLE_ROW_ATTN,
                (true, false) => TOL_SINGLE_ROW,
            };
            if !(tap_worst.0 <= tol) {
                failures.push(format!(
                    "{name}: {tap} at position {} off by {:.3e} (tolerance {tol:e})",
                    tap_worst.1, tap_worst.0
                ));
            }
            if !(tap_worst.0 <= worst.0) {
                worst = (tap_worst.0, tap, tap_worst.1);
            }
        }
        report.push_str(&format!("   {per_tap}\n"));
        report.push_str(&format!(
            "  {name:<16} worst {:.3e} ({} at position {})\n",
            worst.0, worst.1, worst.2
        ));
    }
    println!("{report}");
    ensure!(failures.is_empty(), "{report}{}", failures.join("\n"));
    Ok(())
}
