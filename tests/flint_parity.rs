//! The Flint parity gate: xwen's per-position logits against llama.cpp's, over
//! the committed oracle fixture `tests/fixtures/flint-parity.json`.
//!
//! The fixture holds, per prompt and position, the oracle's top-10 next-token
//! ids and logits (`scripts/flint-ref-logits.ts`, llama.cpp at a37cc4610 on its
//! Metal backend, which the gate checks). The full rows are 600 MB per thousand
//! tokens and stay out of the repo; ten entries per row are enough to grade what
//! the gate grades:
//!
//! - **argmax**: the share of positions where xwen's top-1 is the oracle's, or
//!   the oracle decided its own top-1 by less than [`NEAR_TIE`].
//! - **top-5**: overlap of the two top-5 sets.
//! - **logit error**: the mean absolute difference over the oracle's top-10
//!   entries; the max is reported.
//!
//! Each is graded twice. POOLED over every position, against two bars: the
//! ACCEPTANCE bar ([`ACCEPT`]) and the tighter REGRESSION bar ([`REGRESS`]). And
//! per WINDOW of [`WINDOW`] consecutive positions of one prompt, worst window
//! against [`WORST_WINDOW`], so a fault confined to a stretch of rows (a ring
//! wrap, a chunk boundary) is not diluted by the clean rows around it.
//!
//! Why the acceptance bar is where it is. Flint amplifies rounding: its 47
//! sigmoid routers sit on near ties, a flipped sixth expert moves the residual
//! stream, and a few layers later a different expert with a massive activation
//! gets picked. llama.cpp's own CPU and Metal backends, both correct, disagree
//! by mean |d| 0.613, top-5 0.863 and argmax 0.907 over these prompts
//! (2026-10-25). That spread is the oracle's resolution, so the acceptance bar
//! admits anything at least as close to llama.cpp Metal as llama.cpp CPU is,
//! and a CPU-made fixture would sit on it, which is why the gate refuses one.
//!
//! The regression bar sits just outside what xwen measures today, so a change
//! that costs accuracy without breaking the model still fails here and has to
//! say why. The brackets (below) are what show both bars can tell a broken
//! graph from this one.
//!
//! Phases, one model load each:
//!
//! 1. Every prompt, prefilled in chunks of 512 and again at the model's own
//!    default chunk (2048, what every request over 2048 tokens runs). `long-1400`
//!    crosses the 513 window twice; at 512 it also crosses chunk boundaries.
//! 2. `parity-long-mixed` prefilled to 300 tokens and DECODED one token at a
//!    time to its end (602), walking the sliding layers' ring past its 513
//!    slots on the single-row path.
//! 3. Three wrong graphs (`XwenModel::load_flint_wrong_graph`) over every
//!    prompt: sinks dropped, rope on the NoPE layers, and a sliding window one
//!    position short (512). The first two must FAIL, or the bars cannot tell
//!    this model from a broken one. The third is reported and not asserted:
//!    it reads 0.969 / 0.945 / 0.214 against the real graph's 0.972 / 0.947 /
//!    0.203 (2026-10-25), inside the oracle's own noise, so no logits bar can
//!    see it; `flint::attention`'s ring test holds the window exactly.
//!
//! ```text
//! cargo test --release --test flint_parity -- --ignored --nocapture
//! ```
//!
//! It fails rather than skips when the fixture or the checkpoint is missing.
//!
//! `XWEN_FLINT_PARITY_QUICK=<n>` scores the first `n` prompts at prefill and
//! nothing else, for bisecting; a quick run never passes the gate.

use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use candle_core::Tensor;
use serde::Deserialize;
use xwen::XwenModel;
use xwen::checkpoint::CheckpointSource;
use xwen::flint::attention::WrongGraph;
use xwen::gguf;
use xwen::hub::Model;
use xwen::ops::ExpertRunner;

/// An argmax disagreement is a near tie when the oracle's own top-1 beat its
/// top-2 by less than this.
const NEAR_TIE: f32 = 0.05;

/// Pooled bars: argmax floor, top-5 floor, mean |Δlogit| ceiling.
#[derive(Clone, Copy)]
struct Bars {
    argmax: f64,
    top5: f64,
    mean_abs: f64,
}

/// The acceptance bar: at least as close to llama.cpp Metal as llama.cpp CPU.
const ACCEPT: Bars = Bars {
    argmax: 0.88,
    top5: 0.85,
    mean_abs: 0.65,
};

/// The regression bar at prefill: just outside xwen's own figures as of
/// 2026-10-25 (0.970 / 0.949 / 0.210 at chunk 512, 0.972 / 0.947 / 0.203 at
/// 2048).
const REGRESS: Bars = Bars {
    argmax: 0.955,
    top5: 0.93,
    mean_abs: 0.26,
};

/// The regression bar for the decode phase, which grades one prompt (602
/// positions) and so sits further from its measured 0.940 / 0.921 / 0.238. Its
/// first 300 rows come from one 300-token prefill, a chunk shape the prefill
/// phases do not run, which on this model moves the logits as much as a
/// different backend does (tests/flint_chunks.rs has the measurement).
const REGRESS_DECODE: Bars = Bars {
    argmax: 0.92,
    top5: 0.90,
    mean_abs: 0.30,
};

/// Positions per window for the worst-window grade.
const WINDOW: usize = 64;

/// Worst-window bars over every window of every prompt. xwen's worst windows
/// read 0.875 / 0.875 / 0.638 at prefill and 0.88 / 0.866 / 0.445 at decode
/// (2026-10-25); the broken graphs' read at most 0.17 / 0.27 / 11.8.
const WORST_WINDOW: Bars = Bars {
    argmax: 0.82,
    top5: 0.82,
    mean_abs: 0.75,
};

#[derive(Deserialize)]
struct Fixture {
    vocab: usize,
    backend: String,
    prompts: Vec<Prompt>,
}

#[derive(Deserialize)]
struct Prompt {
    id: String,
    ids: Vec<u32>,
    /// Per position, `(id, logit)` best first.
    top: Vec<Vec<(u32, f32)>>,
}

fn fixture() -> Result<Fixture> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/flint-parity.json");
    let text = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "reading {} (regenerate with scripts/flint-ref-logits.ts)",
            path.display()
        )
    })?;
    let f: Fixture = serde_json::from_str(&text)?;
    ensure!(
        f.backend.starts_with("metal"),
        "the fixture was produced on llama.cpp's {:?} backend; the bars are calibrated \
         against its Metal backend (regenerate with scripts/flint-ref-logits.ts, whose \
         default is Metal)",
        f.backend
    );
    for p in &f.prompts {
        ensure!(
            p.top.len() == p.ids.len(),
            "{}: {} rows for {} ids",
            p.id,
            p.top.len(),
            p.ids.len()
        );
    }
    Ok(f)
}

/// One graded position.
#[derive(Clone, Copy)]
struct Row {
    agree: bool,
    near_tie: bool,
    top5_hits: usize,
    abs_sum: f64,
    abs_n: usize,
    abs_max: f32,
}

/// The worst window per metric.
struct Worst {
    argmax: f64,
    top5: f64,
    mean_abs: f64,
    at: String,
}

impl Worst {
    fn passes(&self, bars: Bars) -> bool {
        self.argmax >= bars.argmax && self.top5 >= bars.top5 && self.mean_abs <= bars.mean_abs
    }
}

#[derive(Default)]
struct Pool {
    /// Per prompt, its rows in position order.
    prompts: Vec<(String, Vec<Row>)>,
    hard_flips: Vec<String>,
}

#[derive(Default, Clone, Copy)]
struct Tally {
    positions: usize,
    agree: usize,
    near_tie: usize,
    top5_hits: usize,
    abs_sum: f64,
    abs_n: usize,
    abs_max: f32,
}

impl Tally {
    fn add(&mut self, r: &Row) {
        self.positions += 1;
        self.agree += r.agree as usize;
        self.near_tie += r.near_tie as usize;
        self.top5_hits += r.top5_hits;
        self.abs_sum += r.abs_sum;
        self.abs_n += r.abs_n;
        self.abs_max = self.abs_max.max(r.abs_max);
    }

    fn argmax(&self) -> f64 {
        (self.agree + self.near_tie) as f64 / self.positions.max(1) as f64
    }

    fn top5(&self) -> f64 {
        self.top5_hits as f64 / (5 * self.positions).max(1) as f64
    }

    fn mean_abs(&self) -> f64 {
        self.abs_sum / self.abs_n.max(1) as f64
    }

    fn passes(&self, bars: Bars) -> bool {
        self.argmax() >= bars.argmax && self.top5() >= bars.top5 && self.mean_abs() <= bars.mean_abs
    }
}

impl Pool {
    fn start(&mut self, prompt: &str) {
        self.prompts.push((prompt.to_string(), Vec::new()));
    }

    fn observe(&mut self, pos: usize, ours: &[f32], reference: &[(u32, f32)]) {
        let mut order: Vec<u32> = (0..ours.len() as u32).collect();
        let k = 5.min(order.len());
        order.select_nth_unstable_by(k - 1, |&a, &b| {
            ours[b as usize]
                .total_cmp(&ours[a as usize])
                .then(a.cmp(&b))
        });
        let mut top5: Vec<u32> = order[..k].to_vec();
        top5.sort_by(|&a, &b| {
            ours[b as usize]
                .total_cmp(&ours[a as usize])
                .then(a.cmp(&b))
        });
        let (ref_top1, ref_l1) = reference[0];
        let margin = ref_l1 - reference[1].1;
        let agree = top5[0] == ref_top1;
        let near_tie = !agree && margin < NEAR_TIE;
        let (prompt, rows) = self.prompts.last_mut().expect("observe before start");
        if !agree && !near_tie {
            self.hard_flips.push(format!(
                "{prompt} pos {pos}: oracle {ref_top1} by {margin:.3}, xwen {} ({:.3} vs {:.3})",
                top5[0], ours[top5[0] as usize], ours[ref_top1 as usize]
            ));
        }
        let mut abs_sum = 0.0;
        let mut abs_max = 0f32;
        for &(id, l) in reference {
            let d = (ours[id as usize] - l).abs();
            abs_sum += d as f64;
            abs_max = abs_max.max(d);
        }
        rows.push(Row {
            agree,
            near_tie,
            top5_hits: reference[..5]
                .iter()
                .filter(|(id, _)| top5.contains(id))
                .count(),
            abs_sum,
            abs_n: reference.len(),
            abs_max,
        });
    }

    fn absorb(&mut self, other: Pool) {
        self.prompts.extend(other.prompts);
        self.hard_flips.extend(other.hard_flips);
    }

    fn total(&self) -> Tally {
        let mut t = Tally::default();
        for (_, rows) in &self.prompts {
            rows.iter().for_each(|r| t.add(r));
        }
        t
    }

    /// The worst [`WINDOW`]-position window of any prompt, per metric (each
    /// metric's worst may be a different window). A prompt shorter than one
    /// window is one window, and a trailing remainder shorter than a window
    /// joins the window before it, so no window grades a handful of rows.
    fn worst_window(&self) -> Worst {
        let mut w = Worst {
            argmax: f64::INFINITY,
            top5: f64::INFINITY,
            mean_abs: 0.0,
            at: String::new(),
        };
        for (id, rows) in &self.prompts {
            let n_windows = (rows.len() / WINDOW).max(1);
            for i in 0..n_windows {
                let end = if i + 1 == n_windows {
                    rows.len()
                } else {
                    (i + 1) * WINDOW
                };
                let chunk = &rows[i * WINDOW..end];
                let mut t = Tally::default();
                chunk.iter().for_each(|r| t.add(r));
                if t.argmax() < w.argmax {
                    w.argmax = t.argmax();
                    w.at = format!("{id} rows {}..", i * WINDOW);
                }
                w.top5 = w.top5.min(t.top5());
                w.mean_abs = w.mean_abs.max(t.mean_abs());
            }
        }
        w
    }

    fn passes(&self, regress: Bars) -> bool {
        self.total().passes(ACCEPT)
            && self.total().passes(regress)
            && self.worst_window().passes(WORST_WINDOW)
    }

    fn line(&self, label: &str, regress: Bars) -> String {
        let t = self.total();
        let w = self.worst_window();
        format!(
            "  {label:<28} {:>5} pos  argmax {:.4} ({:>5} agree {:>3} near-tie {:>3} hard)  \
             top-5 {:.4}  mean|d| {:.4}  max|d| {:.3}\n  {:<28} worst {WINDOW}-window: argmax \
             {:.4} ({})  top-5 {:.4}  mean|d| {:.4}   accept {} regress {} window {}",
            t.positions,
            t.argmax(),
            t.agree,
            t.near_tie,
            self.hard_flips.len(),
            t.top5(),
            t.mean_abs(),
            t.abs_max,
            "",
            w.argmax,
            w.at,
            w.top5,
            w.mean_abs,
            pass(t.passes(ACCEPT)),
            pass(t.passes(regress)),
            pass(w.passes(WORST_WINDOW)),
        )
    }
}

fn pass(ok: bool) -> &'static str {
    if ok { "pass" } else { "FAIL" }
}

fn source(max_ctx: usize) -> Result<(CheckpointSource, candle_core::Device, usize)> {
    let path = xwen::hub::cached_model(Model::Flint)
        .context("Flint is not in the Hugging Face cache; run `xwen fetch --model flint`")?;
    let device = gguf::metal_device()?;
    let source = CheckpointSource::open(&path, &device, Some(Model::Flint))?;
    Ok((source, device, max_ctx))
}

fn load(max_ctx: usize) -> Result<(XwenModel, candle_core::Device)> {
    let (source, device, max_ctx) = source(max_ctx)?;
    Ok((
        XwenModel::load(source, ExpertRunner::Fused, max_ctx)?,
        device,
    ))
}

/// Prefill a prompt in chunks of `chunk`, scoring every position.
fn prefill_all(
    model: &mut XwenModel,
    dev: &candle_core::Device,
    p: &Prompt,
    chunk: usize,
    pool: &mut Pool,
) -> Result<()> {
    pool.start(&p.id);
    model.reset_cache()?;
    let mut pos = 0;
    while pos < p.ids.len() {
        let take = chunk.min(p.ids.len() - pos);
        let tokens = Tensor::new(&p.ids[pos..pos + take], dev)?;
        let rows = model.forward_all_logits(&tokens, pos)?.to_vec2::<f32>()?;
        for (r, row) in rows.iter().enumerate() {
            pool.observe(pos + r, row, &p.top[pos + r]);
        }
        pos += take;
    }
    Ok(())
}

#[test]
#[ignore = "needs a Metal device and the Flint checkpoint"]
fn flint_logits_match_the_llamacpp_oracle() -> Result<()> {
    let fx = fixture()?;
    let longest = fx.prompts.iter().map(|p| p.ids.len()).max().unwrap_or(0);
    let (mut model, dev) = load(longest)?;
    ensure!(
        model.config().vocab == fx.vocab,
        "vocab {} vs fixture {}",
        model.config().vocab,
        fx.vocab
    );
    let shipped_chunk = model.prefill_chunk();

    let mut report = format!(
        "FLINT LOGITS PARITY (oracle {}; accept: argmax >= {}, top-5 >= {}, mean|d| <= {}; \
         regress: {} / {} / {}; worst {WINDOW}-window: {} / {} / {})\n",
        fx.backend,
        ACCEPT.argmax,
        ACCEPT.top5,
        ACCEPT.mean_abs,
        REGRESS.argmax,
        REGRESS.top5,
        REGRESS.mean_abs,
        WORST_WINDOW.argmax,
        WORST_WINDOW.top5,
        WORST_WINDOW.mean_abs,
    );
    let quick: Option<usize> = std::env::var("XWEN_FLINT_PARITY_QUICK")
        .ok()
        .map(|v| v.parse())
        .transpose()?;
    let mut prefills = Vec::new();
    for chunk in [512, shipped_chunk] {
        let mut total = Pool::default();
        for p in fx.prompts.iter().take(quick.unwrap_or(usize::MAX)) {
            let mut pool = Pool::default();
            prefill_all(&mut model, &dev, p, chunk, &mut pool)?;
            if quick.is_some() || chunk == 512 {
                report.push_str(&pool.line(&p.id, REGRESS));
                report.push('\n');
            }
            total.absorb(pool);
        }
        report.push_str(&total.line(&format!("ALL (prefill chunk {chunk})"), REGRESS));
        report.push('\n');
        if quick.is_some() {
            println!("{report}");
            anyhow::bail!("XWEN_FLINT_PARITY_QUICK is set: a quick run reports and never passes");
        }
        prefills.push((chunk, total));
    }

    // Decode across the ring boundary.
    let p = fx
        .prompts
        .iter()
        .find(|p| p.id == "parity-long-mixed")
        .context("the fixture has no parity-long-mixed")?;
    let split = 300;
    ensure!(
        p.ids.len() > 513 + 16,
        "parity-long-mixed must run past the window"
    );
    let mut decode = Pool::default();
    decode.start(&p.id);
    model.reset_cache()?;
    let rows = model
        .forward_all_logits(&Tensor::new(&p.ids[..split], &dev)?, 0)?
        .to_vec2::<f32>()?;
    for (r, row) in rows.iter().enumerate() {
        decode.observe(r, row, &p.top[r]);
    }
    for pos in split..p.ids.len() {
        let row = model
            .forward(&Tensor::new(&p.ids[pos..pos + 1], &dev)?, pos)?
            .to_vec1::<f32>()?;
        decode.observe(pos, &row, &p.top[pos]);
    }
    report.push_str(&decode.line("parity-long-mixed decode", REGRESS_DECODE));
    report.push('\n');
    drop(model);

    // The wrong graphs, over every prompt at the shipped chunk.
    let mut brackets = Vec::new();
    for arm in [
        WrongGraph::NoSinks,
        WrongGraph::RopeEverywhere,
        WrongGraph::Window512,
    ] {
        let (source, dev, max_ctx) = source(longest)?;
        let mut wrong =
            XwenModel::load_flint_wrong_graph(source, ExpertRunner::Fused, max_ctx, arm)?;
        let mut pool = Pool::default();
        for p in &fx.prompts {
            prefill_all(&mut wrong, &dev, p, shipped_chunk, &mut pool)?;
        }
        report.push_str(&pool.line(&format!("BRACKET {arm:?}"), REGRESS));
        report.push('\n');
        brackets.push((arm, pool.passes(REGRESS)));
    }

    println!("{report}");
    let mut problems = Vec::new();
    for (chunk, total) in &prefills {
        if !total.passes(REGRESS) {
            problems.push(format!(
                "prefill at chunk {chunk} fails; first hard flips: {:?}",
                &total.hard_flips[..total.hard_flips.len().min(20)]
            ));
        }
    }
    if !decode.passes(REGRESS_DECODE) {
        problems.push(format!(
            "decode fails; first hard flips: {:?}",
            &decode.hard_flips[..decode.hard_flips.len().min(20)]
        ));
    }
    for (arm, passed) in brackets {
        // A window one short is invisible at this resolution (it reads within
        // the noise of the real graph, 2026-10-25): the arm is run to keep
        // that measured, and the window is held to its definition by
        // `flint::attention`'s ring test instead.
        if passed && arm != WrongGraph::Window512 {
            problems.push(format!(
                "the wrong graph {arm:?} PASSES every bar, so they grade nothing"
            ));
        }
    }
    ensure!(problems.is_empty(), "{report}\n{}", problems.join("\n"));
    Ok(())
}
