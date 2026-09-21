//! Stage 1 of the Qwen-Image 2.1 verification plan: the text-encoder hidden
//! state.
//!
//! The graded claim is that `XwenModel::encode_tap` over the prompt
//! `qwen_image::conditioning` renders reproduces the conditioning tensor the
//! Qwen-Image 2.1 pipeline feeds its diffusion transformer: the residual after
//! ALL 36 layers of the Qwen3-VL-8B language model, BEFORE its final norm, with
//! the rows of the fixed system turn dropped. The reference is a CPU fp32 torch
//! run (`scripts/qwen-image-ref-dump.py`); its rendered strings and ids are
//! committed in `tests/fixtures/qwen-image-encoder/tokens.json`, its per-file
//! sha256 and its own bf16 spread in `reference.json`, and its arrays are not
//! committed and live in `$XWEN_QWEN_IMAGE_REF_DIR`.
//!
//! Four checks in order, because a later one is meaningless if an earlier one
//! fails: the rendered prompt is byte-equal to the fixture, the ids and the
//! drop index are equal, the hidden state is inside the bar — and the WRONG
//! hidden state is outside it. That last one is the bracket: the reference's
//! NORMED output has the same shape and is what a port reading
//! `hidden_states[-1]` without the pipeline's hook produces, so a bar that
//! passed it would not be grading the thing this gate exists for.
//!
//! ```text
//! XWEN_QWEN_IMAGE_REF_DIR=/tmp/qwen-image-ref \
//!   cargo test --release --test qwen_image_encoder -- --ignored --nocapture
//! ```
//!
//! Environment:
//!
//! * `XWEN_QWEN_IMAGE_REF_DIR` — required; the dump directory, holding
//!   `00/`..`11/`.
//! * `XWEN_QWEN_IMAGE_DIR` — the checkpoint's `text_encoder/` directory.
//!   Defaults to the registry entry's cached snapshot; absent from the cache,
//!   the test says to run `xwen fetch --model qwen-image-2.1-encoder`.
//! * `XWEN_QWEN_IMAGE_ONLY` — comma-separated prompt indices, to run a subset.
//!
//! * `XWEN_QWEN_IMAGE_DUMP_DIR` — also write xwen's kept rows there, one
//!   `<idx>.safetensors` per prompt, for a per-row analysis against the dump.
//!
//! # The bars, and where they come from
//!
//! Two metrics per kept row, as in the Z-Image gate: cosine, and the largest
//! absolute difference over the reference row's largest magnitude (`rel`).
//! Every bar is bracketed from both sides by numbers `reference.json` carries
//! (`spread`, written by the dump's `finalize`), and
//! `the_bars_sit_between_the_reference_spread_and_the_wrong_graph` asserts the
//! bracket against the file, so a bar cannot drift outside it:
//!
//! * a CORRECT graph at lower precision — the reference's own bf16 arm, whose
//!   rows reach min cosine 0.9929 and rel p50 0.034 / p99 0.157 / max 0.233;
//! * the WRONG graph — the final norm applied — whose best row is cosine 0.959
//!   and rel 0.40.
//!
//! xwen (bf16 weights, f32 activations) measured rel p50 0.0023, p99 0.0065,
//! max 0.0187 and min cosine 0.99997 over 1805 rows, an order of magnitude
//! inside the reference's own bf16 arm. So: every row at cosine >= 0.9999 and
//! rel <= 0.03, which is under the MEDIAN row of torch's bf16 arm and 13x under
//! the wrong graph's best; and, over a full run, at least 99% of rows at the
//! rel <= 0.01 the Z-Image gate holds every row to. The handful past 0.01 are
//! the rows where torch's bf16 arm is at its own worst (0.03-0.16 on the same
//! rows): ill-conditioned rows, not a divergent graph.
//!
//! Kept row 0 has its own stated bar, the way position 0 does in the Z-Image
//! gate, and for a sharper version of the same reason. It is the user turn's
//! `<|im_start|>`, under the same context in every prompt, so its reference row
//! is identical across prompts. The model parks a massive activation on it:
//! measured in torch, its residual norm is ~9400 through layers 24-34 against
//! ~200-700 for every other token, and the last layers cancel it back to an
//! ordinary ~850. Its final value is therefore a small difference of large
//! numbers, and every arithmetic is at its worst there: torch's own bf16 arm
//! reads cosine 0.954 and rel 1.58 on it, xwen cosine 0.99998 and rel 0.0254
//! (an absolute error of 2.6 on values computed at magnitude 9344, 2.7e-4 of
//! the scale the row was computed at). Its bar is cosine >= 0.9999 and
//! rel <= 0.05: 30x inside torch's bf16 arm and 20x inside the wrong graph,
//! which reads cosine 0.905 and rel 1.0 on that row.
//!
//! The fixture tests below the gate are not `#[ignore]`d and need no model and
//! no GPU.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use xwen::hub::Model;
use xwen::qwen_image::conditioning;

/// Every kept row. See the module doc for where each number comes from.
const COS_MIN: f64 = 0.9999;
/// Every kept row but row 0.
const REL_MAX: f64 = 0.03;
/// What all but [`TYPICAL_SHARE`] of the rows past row 0 hold, over a full run:
/// the Z-Image encoder gate's per-row bar.
const REL_TYPICAL: f64 = 1e-2;
const TYPICAL_SHARE: f64 = 0.99;
/// Kept row 0 alone: the massive-activation row.
const ROW0_REL_MAX: f64 = 0.05;
/// Floor under the relative-error denominator, for a reference row of zeros.
const REL_DENOM_FLOOR: f64 = 1e-6;

const ENTRY: Model = Model::QwenImage21Encoder;

/// One row's distance from its reference row.
#[derive(Debug, Clone, Copy)]
struct TokenMetrics {
    cosine: f64,
    /// Largest absolute difference over the reference row's largest magnitude.
    rel: f64,
}

impl TokenMetrics {
    /// Whether kept row `row` is inside its bar.
    fn passes(self, row: usize) -> bool {
        let rel_max = if row == 0 { ROW0_REL_MAX } else { REL_MAX };
        self.cosine >= COS_MIN && self.rel <= rel_max
    }
}

fn token_metrics(x: &[f32], r: &[f32]) -> TokenMetrics {
    let (mut dot, mut nx, mut nr, mut diff, mut peak) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in x.iter().zip(r) {
        let (a, b) = (a as f64, b as f64);
        dot += a * b;
        nx += a * a;
        nr += b * b;
        diff = diff.max((a - b).abs());
        peak = peak.max(b.abs());
    }
    let denom = nx.sqrt() * nr.sqrt();
    let cosine = if denom == 0.0 {
        if nx == 0.0 && nr == 0.0 { 1.0 } else { 0.0 }
    } else {
        dot / denom
    };
    TokenMetrics {
        cosine,
        rel: diff / peak.max(REL_DENOM_FLOOR),
    }
}

/// Every row of a `[tokens, hidden]` pair: row 0 on its own, the worst figures
/// of the rest, and the failing rows.
#[derive(Debug)]
struct Comparison {
    row0: TokenMetrics,
    /// Over the rows past row 0; a one-row sequence leaves them at their
    /// identities.
    min_cosine: f64,
    max_rel: f64,
    /// Rows past row 0 whose `rel` is over [`REL_TYPICAL`].
    over_typical: usize,
    failures: Vec<(usize, TokenMetrics)>,
}

fn compare(candidate: &[f32], reference: &[f32], hidden: usize) -> Result<Comparison> {
    ensure!(
        hidden > 0 && !reference.is_empty() && reference.len() % hidden == 0,
        "the reference holds {} values, not rows of {hidden}",
        reference.len()
    );
    ensure!(
        candidate.len() == reference.len(),
        "candidate holds {} values, the reference {}",
        candidate.len(),
        reference.len()
    );
    let mut out = Comparison {
        row0: TokenMetrics {
            cosine: 1.0,
            rel: 0.0,
        },
        min_cosine: 1.0,
        max_rel: 0.0,
        over_typical: 0,
        failures: Vec::new(),
    };
    for (t, (x, r)) in candidate
        .chunks(hidden)
        .zip(reference.chunks(hidden))
        .enumerate()
    {
        let m = token_metrics(x, r);
        if !m.passes(t) {
            out.failures.push((t, m));
        }
        if t == 0 {
            out.row0 = m;
            continue;
        }
        out.min_cosine = out.min_cosine.min(m.cosine);
        out.max_rel = out.max_rel.max(m.rel);
        out.over_typical += usize::from(m.rel > REL_TYPICAL);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Deserialize)]
struct PromptFile {
    prompts: Vec<PromptText>,
}

#[derive(Debug, serde::Deserialize)]
struct PromptText {
    idx: usize,
    text: String,
    /// The text spells an added token. The reference tokenizer reads it as the
    /// token; xwen's renderer keeps a user's text as text.
    #[serde(default)]
    literal_specials: bool,
}

#[derive(Debug, serde::Deserialize)]
struct Tokens {
    drop: usize,
    prompts: Vec<TokenPrompt>,
}

#[derive(Debug, serde::Deserialize)]
struct TokenPrompt {
    idx: usize,
    label: String,
    #[serde(rename = "T")]
    tokens: usize,
    #[serde(rename = "T_kept")]
    kept: usize,
    ids: Vec<u32>,
    rendered: String,
}

#[derive(Debug, serde::Deserialize)]
struct Reference {
    hidden_index: usize,
    final_norm: bool,
    drop: usize,
    hidden_size: usize,
    check_prompt_idx: usize,
    prompts: Vec<RefPrompt>,
}

#[derive(Debug, serde::Deserialize)]
struct RefPrompt {
    idx: usize,
    dir: String,
    hidden_fp32: String,
    /// Present on the check prompt alone: the reference's normed state.
    hidden_normed_fp32: Option<String>,
    bf16_vs_fp32: Spread,
}

#[derive(Debug, serde::Deserialize)]
struct Spread {
    min_cosine: f64,
    max_rel_error: f64,
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen-image-encoder")
}

/// `reference.json`, which exists only once the weight stages of the dump have
/// run: it records their per-file hashes and the reference's own bf16 spread.
/// Its absence FAILS the gate, naming what produces it.
fn load_reference() -> Result<Reference> {
    let path = fixture_dir().join("reference.json");
    ensure!(
        path.is_file(),
        "{} does not exist: the encoder reference has not been dumped. With the checkpoint \
         cached (`xwen fetch --model qwen-image-2.1-encoder`) and the venv the script's \
         header describes, run\n  /tmp/qwen-image-venv/bin/python \
         scripts/qwen-image-ref-dump.py --stage fp32\n  /tmp/qwen-image-venv/bin/python \
         scripts/qwen-image-ref-dump.py --stage bf16\n  /tmp/qwen-image-venv/bin/python \
         scripts/qwen-image-ref-dump.py --stage finalize\nwhich writes the arrays to \
         /tmp/qwen-image-ref (point $XWEN_QWEN_IMAGE_REF_DIR there) and this file",
        path.display()
    );
    load_json(&path)
}

fn load_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

fn ref_dir() -> Result<PathBuf> {
    let dir = std::env::var_os("XWEN_QWEN_IMAGE_REF_DIR").context(
        "set $XWEN_QWEN_IMAGE_REF_DIR to the directory scripts/qwen-image-ref-dump.py wrote \
         (it holds 00/../11/ and the manifests); the arrays are not committed",
    )?;
    let dir = PathBuf::from(dir);
    ensure!(
        dir.is_dir(),
        "$XWEN_QWEN_IMAGE_REF_DIR {} is not a directory",
        dir.display()
    );
    Ok(dir)
}

/// The `text_encoder/` directory of the Qwen-Image 2.1 checkpoint.
fn encoder_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("XWEN_QWEN_IMAGE_DIR") {
        let dir = PathBuf::from(dir);
        ensure!(
            dir.is_dir(),
            "$XWEN_QWEN_IMAGE_DIR {} is not a directory",
            dir.display()
        );
        return Ok(dir);
    }
    let config = xwen::hub::cached_model(ENTRY).context(
        "the Qwen-Image 2.1 text encoder is not in the Hugging Face cache: run \
         `xwen fetch --model qwen-image-2.1-encoder`, or point $XWEN_QWEN_IMAGE_DIR at an \
         existing text_encoder/ directory",
    )?;
    Ok(config
        .parent()
        .context("the cached config.json has no parent directory")?
        .to_path_buf())
}

fn selected_indices() -> Result<Option<BTreeSet<usize>>> {
    let Ok(raw) = std::env::var("XWEN_QWEN_IMAGE_ONLY") else {
        return Ok(None);
    };
    raw.split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .with_context(|| format!("$XWEN_QWEN_IMAGE_ONLY: {part:?} is not an index"))
        })
        .collect::<Result<BTreeSet<_>>>()
        .map(Some)
}

fn load_hidden(path: &Path, rows: usize, hidden: usize) -> Result<Vec<f32>> {
    let tensor = candle_core::safetensors::load(path, &candle_core::Device::Cpu)
        .with_context(|| format!("reading {}", path.display()))?
        .remove("hidden")
        .with_context(|| format!("{} has no `hidden` tensor", path.display()))?
        .to_dtype(candle_core::DType::F32)?;
    ensure!(
        tensor.dims2()? == (rows, hidden),
        "{} holds {:?}, the fixture says [{rows}, {hidden}]",
        path.display(),
        tensor.dims()
    );
    Ok(tensor.flatten_all()?.to_vec1()?)
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

#[test]
#[ignore = "needs the Qwen-Image 2.1 checkpoint, a Metal device and a reference dump"]
fn qwen_image_encoder_matches_the_fp32_reference() -> Result<()> {
    let reference = load_reference()?;
    let tokens: Tokens = load_json(&fixture_dir().join("tokens.json"))?;
    let prompt_file: PromptFile = load_json(&fixture_dir().join("prompts.json"))?;
    let ref_dir = ref_dir()?;
    let selected = selected_indices()?;

    let spec = ENTRY
        .encoder_spec()
        .context("the Qwen-Image entry carries no encoder spec")?;
    ensure!(
        (spec.layer, spec.final_norm) == (reference.hidden_index, reference.final_norm),
        "the registry reads hidden state {} with final_norm {}, the reference dumped {} with {}",
        spec.layer,
        spec.final_norm,
        reference.hidden_index,
        reference.final_norm
    );
    ensure!(
        reference.drop == tokens.drop,
        "reference.json drops {} rows and tokens.json {}; rerun the dump",
        reference.drop,
        tokens.drop
    );

    let dir = encoder_dir()?;
    let device = xwen::gguf::metal_device().context("this test needs the Metal device")?;
    let source = xwen::CheckpointSource::open(&dir, &device, Some(ENTRY)).with_context(|| {
        format!(
            "opening the Qwen-Image 2.1 text encoder at {}",
            dir.display()
        )
    })?;
    let tokenizer_path = source
        .safetensors()
        .context("the text encoder did not open as a safetensors set")?
        .tokenizer_path()
        .to_path_buf();
    // An untied set: only the encoder load accepts it.
    let mut model = xwen::XwenModel::load_encoder(source, spec.max_tokens)
        .context("loading the Qwen-Image 2.1 text encoder")?;

    let mut failures: Vec<String> = Vec::new();
    let mut ran = 0;
    let mut bracketed = false;
    let (mut rows_past_zero, mut rows_over_typical) = (0usize, 0usize);
    println!(
        "{:>3} {:22} {:>5} | {:>10} {:>8} | {:>10} {:>8} {:>5} | {:>10} {:>8}",
        "idx",
        "label",
        "kept",
        "row0 cos",
        "row0 rel",
        "min cos",
        "max rel",
        ">1e-2",
        "torch bf16",
        "max rel"
    );
    for prompt in &tokens.prompts {
        if selected.as_ref().is_some_and(|s| !s.contains(&prompt.idx)) {
            continue;
        }
        let source_text = prompt_file
            .prompts
            .iter()
            .find(|p| p.idx == prompt.idx)
            .with_context(|| format!("prompts.json has no prompt {}", prompt.idx))?;
        let refp = reference
            .prompts
            .iter()
            .find(|p| p.idx == prompt.idx)
            .with_context(|| format!("reference.json has no prompt {}", prompt.idx))?;

        // 1 and 2. The rendered string, then the ids and the drop. A prompt
        // that spells an added token is the one place the renderer and the
        // reference tokenizer disagree on purpose, and the encoder is graded
        // on the reference's ids either way.
        let rendered = conditioning::prompt_ids(ENTRY, &tokenizer_path, &source_text.text, &[])?;
        ensure!(
            rendered.text == prompt.rendered,
            "prompt {} ({}): the rendered string differs from the fixture",
            prompt.idx,
            prompt.label
        );
        ensure!(
            rendered.drop == tokens.drop,
            "prompt {}: xwen drops {} rows, the fixture {}",
            prompt.idx,
            rendered.drop,
            tokens.drop
        );
        ensure!(
            (rendered.ids == prompt.ids) != source_text.literal_specials,
            "prompt {} ({}): ids {} the fixture's, and literal_specials is {}",
            prompt.idx,
            prompt.label,
            if rendered.ids == prompt.ids {
                "equal"
            } else {
                "differ from"
            },
            source_text.literal_specials
        );

        // 3. The hidden state.
        let (hidden, n) = model
            .encode_spec(&prompt.ids, &spec)
            .with_context(|| format!("encoding prompt {}", prompt.idx))?;
        ensure!(n == prompt.tokens, "prompt {}: T disagrees", prompt.idx);
        let hidden = hidden
            .narrow(0, tokens.drop, prompt.kept)?
            .to_device(&candle_core::Device::Cpu)?
            .to_dtype(candle_core::DType::F32)?;
        let candidate: Vec<f32> = hidden.flatten_all()?.to_vec1()?;
        let dump = ref_dir.join(&refp.dir);
        let fp32 = load_hidden(
            &dump.join(&refp.hidden_fp32),
            prompt.kept,
            reference.hidden_size,
        )?;
        if let Some(out) = std::env::var_os("XWEN_QWEN_IMAGE_DUMP_DIR") {
            let path = PathBuf::from(out).join(format!("{:02}.safetensors", prompt.idx));
            candle_core::safetensors::save(
                &std::collections::HashMap::from([("hidden", hidden.clone())]),
                &path,
            )?;
        }
        let against = compare(&candidate, &fp32, reference.hidden_size)?;
        println!(
            "{:>3} {:22} {:>5} | {:>10.8} {:>8.6} | {:>10.8} {:>8.6} {:>5} | {:>10.8} {:>8.6}",
            prompt.idx,
            prompt.label,
            prompt.kept,
            against.row0.cosine,
            against.row0.rel,
            against.min_cosine,
            against.max_rel,
            against.over_typical,
            refp.bf16_vs_fp32.min_cosine,
            refp.bf16_vs_fp32.max_rel_error
        );
        rows_past_zero += prompt.kept - 1;
        rows_over_typical += against.over_typical;
        for (row, m) in &against.failures {
            failures.push(format!(
                "  prompt {} ({}): kept row {row} (token {}) cosine {:.8}, rel {:.6}",
                prompt.idx,
                prompt.label,
                prompt.ids[tokens.drop + row],
                m.cosine,
                m.rel
            ));
        }

        // 4. The wrong graph, which must be outside the bar: the reference's
        // normed state against xwen's output, and xwen's own normed reading
        // against the reference.
        if let Some(normed) = &refp.hidden_normed_fp32 {
            ensure!(
                prompt.idx == reference.check_prompt_idx,
                "prompt {} carries a normed dump but is not the check prompt",
                prompt.idx
            );
            let normed = load_hidden(&dump.join(normed), prompt.kept, reference.hidden_size)?;
            let wrong_reference = compare(&candidate, &normed, reference.hidden_size)?;
            ensure!(
                wrong_reference.failures.len() == prompt.kept,
                "prompt {}: {} of {} rows of the pre-norm state PASS against the reference's \
                 normed state; the bar does not tell the two apart",
                prompt.idx,
                prompt.kept - wrong_reference.failures.len(),
                prompt.kept
            );
            let (tied, _) = model.encode(&prompt.ids, spec.layer)?;
            let tied: Vec<f32> = tied
                .narrow(0, tokens.drop, prompt.kept)?
                .to_device(&candle_core::Device::Cpu)?
                .to_dtype(candle_core::DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            let wrong_graph = compare(&tied, &fp32, reference.hidden_size)?;
            ensure!(
                wrong_graph.failures.len() == prompt.kept,
                "prompt {}: xwen's NORMED state passes the pre-norm reference on {} rows",
                prompt.idx,
                prompt.kept - wrong_graph.failures.len()
            );
            // And it is the reference's normed state, so the bracket fails for
            // the reason it is meant to.
            let tied_vs_normed = compare(&tied, &normed, reference.hidden_size)?;
            ensure!(
                tied_vs_normed.failures.is_empty(),
                "prompt {}: xwen's normed state misses the reference's normed state (min \
                 cosine {:.8}, max rel {:.6})",
                prompt.idx,
                tied_vs_normed.min_cosine,
                tied_vs_normed.max_rel
            );
            println!(
                "    bracket: normed reference min cosine {:.6}, max rel {:.4}, all {} rows \
                 outside the bar",
                wrong_reference.min_cosine, wrong_reference.max_rel, prompt.kept
            );
            bracketed = true;
        }
        ran += 1;
    }

    ensure!(ran > 0, "no prompt was run; check $XWEN_QWEN_IMAGE_ONLY");
    ensure!(
        bracketed || selected.is_some(),
        "the normed bracket did not run: reference.json names no normed dump"
    );
    ensure!(
        failures.is_empty(),
        "{} rows miss their bar (cosine >= {COS_MIN}; rel <= {ROW0_REL_MAX} on kept row 0, \
         <= {REL_MAX} elsewhere):\n{}",
        failures.len(),
        failures.join("\n")
    );
    // A share of rows, so it is a claim about the whole prompt set: one short
    // prompt holds too few rows for one percent of them to be a row.
    let share = 1.0 - rows_over_typical as f64 / rows_past_zero.max(1) as f64;
    println!(
        "{rows_over_typical} of {rows_past_zero} rows past row 0 are over rel {REL_TYPICAL} \
         ({:.2}% inside)",
        share * 100.0
    );
    ensure!(
        selected.is_some() || share >= TYPICAL_SHARE,
        "only {:.2}% of the rows past row 0 hold rel <= {REL_TYPICAL}; the bar is {:.0}%",
        share * 100.0,
        TYPICAL_SHARE * 100.0
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Fixture and metric tests: no model, no GPU
// ---------------------------------------------------------------------------

/// The committed fixture is self-consistent, which is checkable with no model,
/// no GPU and no dump directory: every tokenized prompt has its text in
/// `prompts.json`, its counts add up, it opens with the dropped system turn and
/// ends on the open assistant turn, and the one prompt that spells an added
/// token is marked. `reference.json` joins once the weight stages have written
/// it: it must describe the same prompts, the same drop and the registry's own
/// reading of the hidden state, with the normed bracket on its check prompt.
#[test]
fn the_committed_fixture_is_self_consistent() -> Result<()> {
    let tokens: Tokens = load_json(&fixture_dir().join("tokens.json"))?;
    let prompt_file: PromptFile = load_json(&fixture_dir().join("prompts.json"))?;
    let spec = ENTRY.encoder_spec().context("no encoder spec")?;
    ensure!(!tokens.prompts.is_empty() && tokens.drop > 0);
    ensure!(tokens.prompts.len() == prompt_file.prompts.len());
    let system = &tokens.prompts[0].ids[..tokens.drop];
    for prompt in &tokens.prompts {
        let source = prompt_file
            .prompts
            .iter()
            .find(|p| p.idx == prompt.idx)
            .with_context(|| format!("prompt {} has no text in prompts.json", prompt.idx))?;
        ensure!(prompt.ids.len() == prompt.tokens, "prompt {}", prompt.idx);
        ensure!(
            prompt.kept + tokens.drop == prompt.tokens,
            "prompt {}",
            prompt.idx
        );
        ensure!(prompt.tokens <= spec.max_tokens, "prompt {}", prompt.idx);
        ensure!(
            &prompt.ids[..tokens.drop] == system,
            "prompt {} does not open with the shared system turn",
            prompt.idx
        );
        ensure!(
            prompt.rendered.ends_with("<|im_start|>assistant\n"),
            "prompt {} does not end on the open assistant turn",
            prompt.idx
        );
        ensure!(
            source.literal_specials == source.text.contains("<think>"),
            "prompt {}: literal_specials is wrong",
            prompt.idx
        );
    }
    ensure!(
        prompt_file.prompts.iter().any(|p| p.literal_specials),
        "no prompt exercises a special spelled in the text"
    );

    let reference_path = fixture_dir().join("reference.json");
    if !reference_path.is_file() {
        return Ok(());
    }
    let reference: Reference = load_json(&reference_path)?;
    ensure!(reference.drop == tokens.drop);
    ensure!((reference.hidden_index, reference.final_norm) == (spec.layer, spec.final_norm));
    ensure!(reference.prompts.len() == tokens.prompts.len());
    for prompt in &reference.prompts {
        ensure!(
            tokens.prompts.iter().any(|p| p.idx == prompt.idx),
            "reference prompt {} is not in tokens.json",
            prompt.idx
        );
        ensure!(!prompt.dir.is_empty() && !prompt.hidden_fp32.is_empty());
        ensure!(
            prompt.hidden_normed_fp32.is_some() == (prompt.idx == reference.check_prompt_idx),
            "prompt {}: the normed bracket belongs to the check prompt alone",
            prompt.idx
        );
    }
    Ok(())
}

#[test]
fn the_renderer_reproduces_every_fixture_string() -> Result<()> {
    let tokens: Tokens = load_json(&fixture_dir().join("tokens.json"))?;
    let prompt_file: PromptFile = load_json(&fixture_dir().join("prompts.json"))?;
    ensure!(tokens.prompts.len() == prompt_file.prompts.len());
    for prompt in &tokens.prompts {
        let source = prompt_file
            .prompts
            .iter()
            .find(|p| p.idx == prompt.idx)
            .with_context(|| format!("prompts.json has no prompt {}", prompt.idx))?;
        let (text, content) = conditioning::render(&source.text, 0);
        ensure!(
            text == prompt.rendered,
            "prompt {} ({}): rendered string differs",
            prompt.idx,
            prompt.label
        );
        ensure!(text[content] == source.text);
        ensure!(prompt.tokens == prompt.ids.len());
        ensure!(prompt.kept + tokens.drop == prompt.tokens);
    }
    Ok(())
}

#[test]
fn the_metrics_tell_a_scaled_row_from_an_equal_one() -> Result<()> {
    let r = [1.0f32, -2.0, 4.0, 0.5];
    let same = compare(&r, &r, 4)?;
    assert!(same.failures.is_empty());
    assert_eq!(same.max_rel, 0.0);
    // A third of the magnitude at cosine 1: what a norm in the wrong place does
    // to a row, and what cosine alone would wave through.
    let scaled: Vec<f32> = r.iter().map(|v| v / 3.0).collect();
    let off = compare(&scaled, &r, 4)?;
    assert!(off.row0.cosine > 0.999_999);
    assert_eq!(off.failures.len(), 1);
    assert!(compare(&r[..3], &r, 4).is_err());
    Ok(())
}

/// Row 0 is graded on its own bar and kept out of the pooled figures, and a row
/// past it is held to the tighter one.
#[test]
fn row_zero_has_its_own_bar_and_stays_out_of_the_pool() -> Result<()> {
    let reference = [100.0f32, 1.0, 100.0, 1.0];
    let between = 100.0 * (1.0 - (REL_MAX + ROW0_REL_MAX) as f32 / 2.0);
    let got = compare(&[between, 1.0, between, 1.0], &reference, 2)?;
    assert!(got.row0.rel > REL_MAX && got.row0.rel < ROW0_REL_MAX);
    assert_eq!(got.failures.len(), 1, "inside on row 0, outside on row 1");
    assert_eq!(got.failures[0].0, 1);
    assert_eq!(got.over_typical, 1);
    assert!((got.max_rel - got.row0.rel).abs() < 1e-6);
    Ok(())
}

/// Each bar sits where the module doc says it does, against the numbers the
/// dump recorded: tighter than the reference's own bf16 arm (a correct graph at
/// lower precision), and with the wrong graph (the final norm applied) outside.
/// Runs once `reference.json` exists.
#[test]
fn the_bars_sit_between_the_reference_spread_and_the_wrong_graph() -> Result<()> {
    let path = fixture_dir().join("reference.json");
    if !path.is_file() {
        return Ok(());
    }
    let reference: serde_json::Value = load_json(&path)?;
    let at = |pointer: &str| -> Result<f64> {
        reference
            .pointer(pointer)
            .and_then(serde_json::Value::as_f64)
            .with_context(|| format!("reference.json has no number at {pointer}"))
    };
    // Past row 0: under the MEDIAN row of torch's bf16 arm, so under its p99 and
    // max too, and the wrong graph's best row is outside on both metrics.
    ensure!(REL_TYPICAL < REL_MAX && REL_MAX < at("/spread/rest/bf16/p50_rel_error")?);
    ensure!(REL_MAX < at("/spread/rest/normed/min_rel_error")?);
    ensure!(COS_MIN > at("/spread/rest/normed/max_cosine")?);
    ensure!(COS_MIN > at("/spread/rest/bf16/min_cosine")?);
    // Row 0: torch's bf16 arm is outside this bar itself, and so is the wrong
    // graph, on both metrics.
    ensure!(ROW0_REL_MAX < at("/spread/row0/bf16/max_rel_error")?);
    ensure!(ROW0_REL_MAX < at("/spread/row0/normed/min_rel_error")?);
    ensure!(COS_MIN > at("/spread/row0/bf16/min_cosine")?);
    ensure!(COS_MIN > at("/spread/row0/normed/max_cosine")?);
    Ok(())
}
