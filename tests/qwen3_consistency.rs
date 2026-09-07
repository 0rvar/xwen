//! Decode-path consistency of the Qwen3 dense stack, with no oracle: the same
//! ids teacher-forced as one prefill, one token at a time, and in chunks of
//! 7 / 8 / 9 / 16 must give the same logits at every position. This is where
//! the three splits the stack straddles would show a seam — the flash kernel
//! (multi-token) against the f16 vector sdpa (one token), `matmul_bf16`'s gemv
//! (t <= 8) against its tensor gemm (t > 8), and the shipped attention arm
//! against the f32 sdpa bisect arm (`XWEN_QWEN3_ATTN=sdpa`). Bar: max-abs
//! logit difference <= 2e-2 (`XWEN_QWEN3_CONSISTENCY_MAX_ABS` overrides it)
//! and an identical argmax at every position (docs/parity.md, the qwen3
//! section).
//!
//! Every comparison runs before anything is asserted: the whole table is
//! printed (prompt, arm, chunk size, positions, the worst |Δ| and where, argmax
//! agreements, positions over the bar), and the test then fails listing every
//! row that missed. A first failure that stopped the run would leave the other
//! chunkings and the flash-vs-sdpa arm unmeasured, which is exactly the number
//! the bar's owner needs to decide where it belongs.
//!
//! Ignored by default (needs the 8 GB checkpoint and a Metal device). ONE test
//! body, deliberately: the second half switches the attention arm through the
//! process environment before a load, and two tests in this binary would run
//! in parallel under the default harness and race on it.
//!
//!   cargo test --release --test qwen3_consistency -- --ignored --nocapture
//!
//! `XWEN_QWEN3_DIR` names the safetensors directory; unset, the base
//! checkpoint's cached snapshot (`Qwen/Qwen3-4B`) is used and the test skips
//! with a message when it is not in the Hugging Face cache.

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use serde_json::Value;

use xwen::checkpoint::CheckpointSource;
use xwen::hub::Model;
use xwen::model::XwenModel;
use xwen::ops::ExpertRunner;
use xwen::qwen3::stack::{ATTN_ENV, AttnImpl};

/// The bar unless `XWEN_QWEN3_CONSISTENCY_MAX_ABS` moves it.
const DEFAULT_MAX_ABS: f32 = 2e-2;
const CHUNKS: [usize; 5] = [1, 7, 8, 9, 16];

fn max_abs_bar() -> Result<f32> {
    match std::env::var("XWEN_QWEN3_CONSISTENCY_MAX_ABS") {
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_MAX_ABS),
        Err(e) => anyhow::bail!("XWEN_QWEN3_CONSISTENCY_MAX_ABS: {e}"),
        Ok(v) => v
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|b| b.is_finite() && *b >= 0.0)
            .with_context(|| {
                format!("XWEN_QWEN3_CONSISTENCY_MAX_ABS={v:?} is not a non-negative number")
            }),
    }
}
const MAX_CTX: usize = 4096;

/// The three fixture prompts the brief names: short, medium and the 610-token one.
const PROMPTS: [&str; 3] = ["parity-code-short", "corpus-middle", "parity-long-mixed"];

fn checkpoint_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("XWEN_QWEN3_DIR") {
        return Some(PathBuf::from(dir));
    }
    // `cached_model` hands back the entry's first file, `config.json`.
    xwen::hub::cached_model(Model::Qwen34B)
        .map(|config| config.parent().map(|p| p.to_path_buf()).unwrap_or(config))
}

fn fixture_ids() -> Result<Vec<(String, Vec<u32>)>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3-prompts.json");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let json: Value = serde_json::from_str(&text)?;
    let prompts = json["prompts"]
        .as_array()
        .context("fixture has no `prompts` array")?;
    let mut out = Vec::new();
    for want in PROMPTS {
        let p = prompts
            .iter()
            .find(|p| p["id"] == want)
            .with_context(|| format!("fixture has no prompt {want:?}"))?;
        let ids: Vec<u32> = p["ids"]
            .as_array()
            .context("prompt has no `ids`")?
            .iter()
            .map(|v| v.as_u64().map(|x| x as u32).context("id is not an integer"))
            .collect::<Result<_>>()?;
        out.push((want.to_string(), ids));
    }
    Ok(out)
}

fn load(dir: &PathBuf, device: &Device) -> Result<XwenModel> {
    // The entry supplies nothing a base checkpoint needs (no allowlist, the
    // tokenizer beside the shards), but a custom `XWEN_QWEN3_DIR` may be the
    // Instruct release, whose theta the cross-check would refuse under the
    // wrong entry — so no entry is named and the directory identifies itself.
    let source = CheckpointSource::open(dir, device, None)?;
    XwenModel::load(source, ExpertRunner::Fused, MAX_CTX)
}

fn rows(t: &Tensor) -> Result<Vec<Vec<f32>>> {
    Ok(t.to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec2::<f32>()?)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

/// All-position logits over `ids` fed in `chunk`-token steps from an empty
/// cache, positions continuous.
fn logits_in_chunks(
    model: &mut XwenModel,
    ids: &[u32],
    chunk: usize,
    device: &Device,
) -> Result<Vec<Vec<f32>>> {
    model.reset_cache()?;
    let mut out = Vec::with_capacity(ids.len());
    let mut pos = 0;
    for c in ids.chunks(chunk) {
        let t = Tensor::new(c, device)?;
        out.extend(rows(&model.forward_all_logits(&t, pos)?)?);
        pos += c.len();
    }
    model.reset_cache()?;
    Ok(out)
}

/// One comparison of two all-position logit (or hidden-state) sets.
struct Row {
    prompt: String,
    arm: &'static str,
    /// What was compared against the arm's one-prefill run: a chunk size, or
    /// a description for the cross-arm and encode rows.
    against: String,
    positions: usize,
    /// The worst |Δ| over every position and vocabulary entry, and its position.
    max_abs: f32,
    max_at: usize,
    argmax_agree: usize,
    over_bar: usize,
    /// Values that were inf or NaN on either side. A comparison with any is a
    /// failure whatever the fold says: `f32::max` returns its non-NaN operand,
    /// so a NaN-filled output would otherwise read as a match.
    nonfinite: usize,
}

impl Row {
    fn failed(&self, bar: f32) -> bool {
        self.nonfinite > 0
            || self.over_bar > 0
            || self.argmax_agree != self.positions
            || self.max_abs > bar
    }
}

/// Every row of the run, printed as one table at the end and asserted after.
struct Table {
    bar: f32,
    rows: Vec<Row>,
}

impl Table {
    /// Compare `a` (the reference side) with `b` position by position and
    /// record the row. Shape disagreements are the one thing that stops the
    /// run: they are a harness bug, not a number to tabulate.
    fn compare(
        &mut self,
        prompt: &str,
        arm: &'static str,
        against: impl Into<String>,
        a: &[Vec<f32>],
        b: &[Vec<f32>],
    ) {
        let label = format!("{prompt} {arm} {}", against.into());
        assert_eq!(a.len(), b.len(), "{label}: position counts differ");
        let mut row = Row {
            prompt: prompt.to_string(),
            arm,
            against: label[prompt.len() + arm.len() + 2..].to_string(),
            positions: a.len(),
            max_abs: 0.0,
            max_at: 0,
            argmax_agree: 0,
            over_bar: 0,
            nonfinite: 0,
        };
        for (p, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.len(), y.len(), "{label}: widths differ at position {p}");
            let mut d = 0f32;
            for (u, v) in x.iter().zip(y) {
                if !(u.is_finite() && v.is_finite()) {
                    row.nonfinite += 1;
                    continue;
                }
                d = d.max((u - v).abs());
            }
            if d > row.max_abs {
                row.max_abs = d;
                row.max_at = p;
            }
            if d > self.bar {
                row.over_bar += 1;
            }
            if argmax(x) == argmax(y) {
                row.argmax_agree += 1;
            }
        }
        self.rows.push(row);
    }

    fn print(&self) {
        println!(
            "\nqwen3 consistency, bar max |Δ| <= {:.3e} (XWEN_QWEN3_CONSISTENCY_MAX_ABS), identical argmax",
            self.bar
        );
        println!(
            "{:<20} {:<6} {:<24} {:>5} {:>11} {:>6} {:>9} {:>8} {:>9}  {}",
            "prompt",
            "arm",
            "against",
            "pos",
            "max|Δ|",
            "at",
            "argmax=",
            "over",
            "nonfinite",
            "verdict"
        );
        for r in &self.rows {
            println!(
                "{:<20} {:<6} {:<24} {:>5} {:>11.4e} {:>6} {:>4}/{:<4} {:>8} {:>9}  {}",
                r.prompt,
                r.arm,
                r.against,
                r.positions,
                r.max_abs,
                r.max_at,
                r.argmax_agree,
                r.positions,
                r.over_bar,
                r.nonfinite,
                if r.failed(self.bar) { "FAIL" } else { "ok" }
            );
        }
    }

    /// Print, then fail once with every row that missed the bar.
    fn finish(self) -> Result<()> {
        self.print();
        let failures: Vec<String> = self
            .rows
            .iter()
            .filter(|r| r.failed(self.bar))
            .map(|r| {
                format!(
                    "{} {} {}: max |Δ| {:.4e} at position {}, {} of {} positions over the bar, \
                     argmax agrees at {} of {}, {} non-finite values",
                    r.prompt,
                    r.arm,
                    r.against,
                    r.max_abs,
                    r.max_at,
                    r.over_bar,
                    r.positions,
                    r.argmax_agree,
                    r.positions,
                    r.nonfinite
                )
            })
            .collect();
        anyhow::ensure!(
            failures.is_empty(),
            "{} of {} comparisons missed the bar {:.3e}:\n  {}",
            failures.len(),
            self.rows.len(),
            self.bar,
            failures.join("\n  ")
        );
        Ok(())
    }
}

#[test]
#[ignore]
fn the_qwen3_stack_is_consistent_across_chunkings_arms_and_encode_indices() -> Result<()> {
    let Some(dir) = checkpoint_dir() else {
        eprintln!(
            "skipping: Qwen3-4B is not in the Hugging Face cache and XWEN_QWEN3_DIR is unset"
        );
        return Ok(());
    };
    let device = xwen::gguf::metal_device()?;
    let mut table = Table {
        bar: max_abs_bar()?,
        rows: Vec::new(),
    };
    // The encode-index half first: it loads under the default arm and must
    // not observe the environment switch the second half sets.
    encode_indices_match_the_forward_taps(&dir, &device, &mut table)?;
    chunked_teacher_forcing_matches_one_prefill_at_every_position(&dir, &device, &mut table)?;
    table.finish()
}

/// The same ids as one prefill, one token at a time and in uneven chunks, on
/// the fused arm; then the sdpa arm against the fused arm and its own chunkings.
fn chunked_teacher_forcing_matches_one_prefill_at_every_position(
    dir: &PathBuf,
    device: &Device,
    table: &mut Table,
) -> Result<()> {
    let device = device.clone();
    let dir = dir.clone();
    let prompts = fixture_ids()?;

    // The shipped arm first: one prefill against every chunking.
    let mut flash = load(&dir, &device)?;
    assert_eq!(
        flash.qwen3_parts().map(|p| p.attn_impl()),
        Some(AttnImpl::Fused),
        "the default load must run the fused attention arm"
    );
    let mut single_flash: Vec<Vec<Vec<f32>>> = Vec::new();
    for (name, ids) in &prompts {
        let single = logits_in_chunks(&mut flash, ids, ids.len(), &device)?;
        for &chunk in &CHUNKS {
            let chunked = logits_in_chunks(&mut flash, ids, chunk, &device)?;
            table.compare(name, "flash", format!("chunk {chunk}"), &single, &chunked);
        }
        single_flash.push(single);
    }
    drop(flash);

    // The bisect arm, loaded in-process with the switch set before the load
    // (the arm is resolved per load, not per process), against the fused
    // arm's one-prefill logits and its own chunkings.
    // SAFETY: this binary has exactly one test and it runs both halves
    // sequentially on this thread; nothing else in the process reads or writes
    // the environment while the variable is set.
    unsafe { std::env::set_var(ATTN_ENV, "sdpa") };
    let mut sdpa = load(&dir, &device)?;
    assert_eq!(
        sdpa.qwen3_parts().map(|p| p.attn_impl()),
        Some(AttnImpl::Sdpa),
        "{ATTN_ENV}=sdpa must select the sdpa arm at load"
    );
    for ((name, ids), single) in prompts.iter().zip(&single_flash) {
        let sdpa_single = logits_in_chunks(&mut sdpa, ids, ids.len(), &device)?;
        table.compare(name, "sdpa", "one prefill vs flash", single, &sdpa_single);
        for &chunk in &CHUNKS {
            let chunked = logits_in_chunks(&mut sdpa, ids, chunk, &device)?;
            table.compare(
                name,
                "sdpa",
                format!("chunk {chunk}"),
                &sdpa_single,
                &chunked,
            );
        }
    }
    unsafe { std::env::remove_var(ATTN_ENV) };
    Ok(())
}

/// `encode` follows transformers' `hidden_states` numbering on the real
/// checkpoint: index 0 is the embedding rows, index 36 is the normed residual
/// after layer 35 (the full forward's `l_out-35` tap through `output_norm`),
/// and index 35 is that residual raw (the `l_out-34` tap).
fn encode_indices_match_the_forward_taps(
    dir: &PathBuf,
    device: &Device,
    table: &mut Table,
) -> Result<()> {
    let mut model = load(dir, device)?;
    let device = device.clone();
    let n_layer = model.config().n_layer;
    assert_eq!(n_layer, 36);
    let (_, ids) = fixture_ids()?.swap_remove(1); // corpus-middle, 199 tokens

    let (h0, t) = model.encode(&ids, 0)?;
    assert_eq!(
        (h0.dims(), h0.dtype(), t),
        (&[ids.len(), 2560][..], DType::BF16, ids.len())
    );
    let embed = model.embed_ids(&ids)?.to_dtype(DType::BF16)?;
    table.compare(
        "corpus-middle",
        "encode",
        "index 0 vs embeddings",
        &rows(&h0)?,
        &rows(&embed)?,
    );

    model.set_tap_capture(true);
    model.reset_cache()?;
    model.forward(&Tensor::new(ids.as_slice(), &device)?, 0)?;
    let taps = model.take_taps();
    model.set_tap_capture(false);
    model.reset_cache()?;
    let tap = |name: &str| -> Tensor {
        taps.iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t.clone())
            .unwrap_or_else(|| panic!("no tap {name}"))
    };

    let (h35, _) = model.encode(&ids, 35)?;
    let l_out_34 = tap("l_out-34").to_dtype(DType::BF16)?;
    table.compare(
        "corpus-middle",
        "encode",
        "index 35 vs l_out-34",
        &rows(&h35)?,
        &rows(&l_out_34)?,
    );

    let (h36, _) = model.encode(&ids, 36)?;
    let normed = model.final_norm(&tap("l_out-35"))?.to_dtype(DType::BF16)?;
    table.compare(
        "corpus-middle",
        "encode",
        "index 36 vs norm(l_out-35)",
        &rows(&h36)?,
        &rows(&normed)?,
    );

    assert!(
        model.encode(&ids, 37).is_err(),
        "index 37 must be refused on a 36-layer model"
    );
    assert_eq!(tap("kqv_out-0").dims(), &[ids.len(), 4096]);
    Ok(())
}
