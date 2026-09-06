//! Stage 2 of the Z-Image verification plan: the text-encoder hidden state.
//!
//! The graded claim is that `XwenModel::encode` reproduces the conditioning
//! tensor the Z-Image-Turbo pipeline feeds its diffusion transformer — the
//! output of `layers[34]`, pre-norm, which HF numbers `hidden_states[35]`. The
//! reference is a CPU fp32 torch run (`scripts/zimage-ref-dump.py`); its
//! rendered strings, ids and per-file sha256 are committed under
//! `tests/fixtures/zimage-encoder/`, its arrays are not (50 MB) and live in
//! `$XWEN_ZIMAGE_REF_DIR`.
//!
//! Three checks in order, because a later one is meaningless if an earlier one
//! fails: the rendered prompt is byte-equal to the fixture, the ids are equal
//! (after the encoder's own 512-token truncation), and only then the hidden
//! state is compared.
//!
//! ```text
//! XWEN_ZIMAGE_REF_DIR=/tmp/zimage-ref \
//!   cargo test --release --test qwen3_encoder -- --ignored --nocapture
//! ```
//!
//! Environment:
//!
//! * `XWEN_ZIMAGE_REF_DIR` — required; the dump directory, holding `00/`..`11/`.
//! * `XWEN_ZIMAGE_DIR` — the checkpoint's `text_encoder/` directory. Defaults to
//!   the registry entry's cached snapshot; absent from the cache, the test says
//!   to run `xwen fetch --model-size zimage-turbo`.
//! * `XWEN_ZIMAGE_ONLY` — comma-separated prompt indices, to run a subset.
//! * `XWEN_ZIMAGE_VERIFY_SHA=1` — also verify each dump file against the sha256
//!   the fixture records, which catches a stale or half-written dump directory.
//!
//! The metric module and the `.npy` reader carry their own CPU unit tests; those
//! are not `#[ignore]`d and run in a plain `cargo test --test qwen3_encoder`
//! with no model and no GPU.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, ensure};

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// The two per-token distances the plan grades on, and the split that keeps
/// position 0 from hiding everything else.
///
/// Position 0 is `<|im_start|>` under every prompt, it attends to nothing, and
/// its fp32 row is therefore bitwise identical across all twelve prompts. It is
/// also a massive activation — max magnitude ~13753 against ~150-380 for every
/// other token — so it dominates a relative-error metric whose denominator is
/// per token. Reporting it on its own line is what tells "the encoder is wrong"
/// apart from "one known-hard row is a few ulps off".
mod metric {
    use anyhow::{Result, ensure};

    /// Per-token cosine floor against the fp32 reference.
    pub const COS_MIN: f64 = 0.9999;

    /// Per-token max-abs relative error ceiling against the fp32 reference.
    pub const REL_MAX: f64 = 1e-2;

    /// Floor on the relative-error denominator, so an all-but-zero reference row
    /// reports an absolute error rather than dividing by nothing.
    pub const REL_DENOM_FLOOR: f64 = 1e-6;

    /// One token's distance from its reference row.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct TokenMetrics {
        pub cosine: f64,
        /// `max_i |x_i - r_i| / max(max_i |r_i|, REL_DENOM_FLOOR)`.
        pub rel: f64,
        /// `max_i |r_i|` — the magnitude of the reference row, reported for
        /// position 0 because that is what explains its relative error.
        pub ref_max_abs: f64,
    }

    impl TokenMetrics {
        /// Whether this token clears both bars.
        pub fn passes(&self) -> bool {
            self.cosine >= COS_MIN && self.rel <= REL_MAX
        }
    }

    /// Cosine similarity, accumulated in f64. Two zero rows are identical, so
    /// they score 1.0; one zero row against a nonzero one scores 0.0.
    pub fn cosine(x: &[f32], r: &[f32]) -> f64 {
        let mut dot = 0.0f64;
        let mut nx = 0.0f64;
        let mut nr = 0.0f64;
        for (&a, &b) in x.iter().zip(r) {
            let (a, b) = (a as f64, b as f64);
            dot += a * b;
            nx += a * a;
            nr += b * b;
        }
        let denom = nx.sqrt() * nr.sqrt();
        if denom == 0.0 {
            return if nx == 0.0 && nr == 0.0 { 1.0 } else { 0.0 };
        }
        dot / denom
    }

    /// `max_i |x_i - r_i|` over the token, divided by the reference row's own
    /// max magnitude (floored). The denominator is per token, per the plan.
    pub fn max_abs_rel_error(x: &[f32], r: &[f32]) -> f64 {
        let mut max_diff = 0.0f64;
        let mut max_ref = 0.0f64;
        for (&a, &b) in x.iter().zip(r) {
            max_diff = max_diff.max((a as f64 - b as f64).abs());
            max_ref = max_ref.max((b as f64).abs());
        }
        max_diff / max_ref.max(REL_DENOM_FLOOR)
    }

    /// `max_i |r_i|`.
    pub fn max_abs(r: &[f32]) -> f64 {
        r.iter().fold(0.0f64, |m, &v| m.max((v as f64).abs()))
    }

    /// A whole prompt's comparison: position 0 alone, positions >= 1 pooled, and
    /// the list of every position that misses a bar.
    #[derive(Debug, Clone)]
    pub struct Comparison {
        pub tokens: usize,
        pub pos0: TokenMetrics,
        /// Min cosine over positions >= 1, `None` for a single-token prompt.
        pub pooled_min_cosine: Option<f64>,
        /// Max relative error over positions >= 1, `None` for a single-token
        /// prompt.
        pub pooled_max_rel: Option<f64>,
        /// The position holding `pooled_min_cosine`.
        pub worst_cosine_pos: Option<usize>,
        /// The position holding `pooled_max_rel`.
        pub worst_rel_pos: Option<usize>,
        /// Every position that misses a bar, position 0 included.
        pub failures: Vec<usize>,
    }

    impl Comparison {
        /// True when something failed and all of it is position 0 — the case the
        /// failure message has to call out, because it is the known-hard row.
        pub fn only_position_zero_failed(&self) -> bool {
            self.failures == [0]
        }
    }

    /// Compare two `[tokens, hidden]` row-major buffers.
    pub fn compare(
        candidate: &[f32],
        reference: &[f32],
        tokens: usize,
        hidden: usize,
    ) -> Result<Comparison> {
        ensure!(tokens > 0, "cannot compare an empty sequence");
        ensure!(hidden > 0, "cannot compare a zero-width hidden state");
        ensure!(
            candidate.len() == tokens * hidden,
            "candidate holds {} values, expected {tokens} x {hidden}",
            candidate.len()
        );
        ensure!(
            reference.len() == tokens * hidden,
            "reference holds {} values, expected {tokens} x {hidden}",
            reference.len()
        );

        let row =
            |buf: &[f32], t: usize| -> Vec<f32> { buf[t * hidden..(t + 1) * hidden].to_vec() };

        let mut failures = Vec::new();
        let mut pos0 = TokenMetrics {
            cosine: 1.0,
            rel: 0.0,
            ref_max_abs: 0.0,
        };
        let mut pooled_min_cosine: Option<f64> = None;
        let mut pooled_max_rel: Option<f64> = None;
        let mut worst_cosine_pos = None;
        let mut worst_rel_pos = None;

        for t in 0..tokens {
            let (x, r) = (row(candidate, t), row(reference, t));
            let m = TokenMetrics {
                cosine: cosine(&x, &r),
                rel: max_abs_rel_error(&x, &r),
                ref_max_abs: max_abs(&r),
            };
            if !m.passes() {
                failures.push(t);
            }
            if t == 0 {
                pos0 = m;
                continue;
            }
            if pooled_min_cosine.is_none_or(|c| m.cosine < c) {
                pooled_min_cosine = Some(m.cosine);
                worst_cosine_pos = Some(t);
            }
            if pooled_max_rel.is_none_or(|e| m.rel > e) {
                pooled_max_rel = Some(m.rel);
                worst_rel_pos = Some(t);
            }
        }

        Ok(Comparison {
            tokens,
            pos0,
            pooled_min_cosine,
            pooled_max_rel,
            worst_cosine_pos,
            worst_rel_pos,
            failures,
        })
    }
}

// ---------------------------------------------------------------------------
// .npy
// ---------------------------------------------------------------------------

/// A reader for exactly the arrays `scripts/zimage-ref-dump.py` writes: 2-D,
/// C-order, little-endian float32. Anything else is an error rather than a
/// reinterpretation — a Fortran-order or f64 dump read as if it were this one
/// would produce plausible numbers and a wrong verdict.
///
/// The fp32 reference is read through candle's safetensors loader instead; this
/// exists for the bf16 diagnostic, which is dumped only as `.npy`.
mod npy {
    use anyhow::{Context, Result, bail, ensure};
    use std::path::Path;

    /// `(values, rows, cols)`, row-major.
    pub fn read_f32_2d(path: &Path) -> Result<(Vec<f32>, usize, usize)> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        parse_f32_2d(&bytes).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn parse_f32_2d(bytes: &[u8]) -> Result<(Vec<f32>, usize, usize)> {
        ensure!(bytes.len() >= 10, "too short to be a .npy file");
        ensure!(&bytes[..6] == b"\x93NUMPY", "not a .npy file (bad magic)");
        let (header_len, header_at) = match bytes[6] {
            1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
            2 => {
                ensure!(bytes.len() >= 12, "truncated v2 header length");
                (
                    u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                    12,
                )
            }
            v => bail!("unsupported .npy major version {v}"),
        };
        let end = header_at + header_len;
        ensure!(bytes.len() >= end, "truncated .npy header");
        let header = std::str::from_utf8(&bytes[header_at..end]).context("header is not UTF-8")?;

        let descr = field(header, "'descr'")?;
        ensure!(
            descr == "'<f4'",
            "expected a little-endian float32 array, got descr {descr}"
        );
        let order = field(header, "'fortran_order'")?;
        ensure!(
            order == "False",
            "expected C order, got fortran_order {order}"
        );

        let shape = field(header, "'shape'")?;
        let dims: Vec<usize> = shape
            .trim_start_matches('(')
            .trim_end_matches(')')
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<usize>()
                    .with_context(|| format!("shape entry {s:?}"))
            })
            .collect::<Result<_>>()?;
        ensure!(dims.len() == 2, "expected a 2-D array, got shape {shape}");
        let (rows, cols) = (dims[0], dims[1]);

        let data = &bytes[end..];
        ensure!(
            data.len() == rows * cols * 4,
            "{} data bytes for a {rows}x{cols} float32 array (expected {})",
            data.len(),
            rows * cols * 4
        );
        let values = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok((values, rows, cols))
    }

    /// The value of one key in the header's dict literal: everything between the
    /// key's colon and the next top-level comma, trimmed. `'shape'` is the one
    /// value holding a comma of its own, so its tuple is closed on `)` first.
    fn field<'a>(header: &'a str, key: &str) -> Result<&'a str> {
        let start = header
            .find(key)
            .with_context(|| format!("no {key} in the .npy header"))?
            + key.len();
        let rest = header[start..]
            .trim_start()
            .strip_prefix(':')
            .with_context(|| format!("{key} is not followed by a colon"))?
            .trim_start();
        let end = if rest.starts_with('(') {
            rest.find(')')
                .with_context(|| format!("unterminated tuple for {key}"))?
                + 1
        } else {
            rest.find(',').unwrap_or(rest.len())
        };
        Ok(rest[..end].trim())
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct Reference {
    max_length: usize,
    hidden_index: usize,
    hidden_size: usize,
    prompts: Vec<RefPrompt>,
}

#[derive(serde::Deserialize)]
struct RefPrompt {
    idx: usize,
    label: String,
    dir: String,
    #[serde(rename = "T")]
    tokens: usize,
    truncated: bool,
    untruncated_len: usize,
    ids: Vec<u32>,
    rendered: String,
    rendered_sha256: String,
    hidden_fp32_safetensors: String,
    hidden_fp32_safetensors_sha256: String,
    hidden_bf16_npy: String,
    hidden_bf16_npy_sha256: String,
}

#[derive(serde::Deserialize)]
struct PromptFile {
    prompts: Vec<PromptText>,
}

#[derive(serde::Deserialize)]
struct PromptText {
    idx: usize,
    text: String,
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/zimage-encoder")
}

fn load_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

fn ref_dir() -> Result<PathBuf> {
    let dir = std::env::var_os("XWEN_ZIMAGE_REF_DIR").context(
        "set $XWEN_ZIMAGE_REF_DIR to the directory scripts/zimage-ref-dump.py wrote \
         (it holds 00/../11/ and the manifests); the arrays are 50 MB and are not committed",
    )?;
    let dir = PathBuf::from(dir);
    ensure!(
        dir.is_dir(),
        "$XWEN_ZIMAGE_REF_DIR {} is not a directory",
        dir.display()
    );
    Ok(dir)
}

/// The `text_encoder/` directory of the Z-Image checkpoint.
///
/// `cached_model` hands back the entry's first listed file, which for this
/// checkpoint is `text_encoder/config.json`; the checkpoint is its directory.
fn encoder_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("XWEN_ZIMAGE_DIR") {
        let dir = PathBuf::from(dir);
        ensure!(
            dir.is_dir(),
            "$XWEN_ZIMAGE_DIR {} is not a directory",
            dir.display()
        );
        return Ok(dir);
    }
    let config = xwen::hub::cached_model(xwen::hub::Model::ZImageTurboEncoder).context(
        "the Z-Image text encoder is not in the Hugging Face cache: run \
         `xwen fetch --model-size zimage-turbo`, or point $XWEN_ZIMAGE_DIR at an \
         existing text_encoder/ directory",
    )?;
    Ok(config
        .parent()
        .context("the cached config.json has no parent directory")?
        .to_path_buf())
}

/// The checkpoint's own `tokenizer.json`, resolved the way
/// `checkpoint::registry_tokenizer` resolves it: the registry names it relative
/// to the REPO root, and this checkpoint's weights sit one level down.
fn tokenizer_path(encoder_dir: &Path) -> Result<PathBuf> {
    let entry = xwen::hub::Model::ZImageTurboEncoder;
    let relative = entry
        .safetensors_tokenizer()
        .context("the Z-Image entry names no tokenizer")?;
    let roots = [
        Some(encoder_dir.to_path_buf()),
        encoder_dir.parent().map(Path::to_path_buf),
    ];
    for root in roots.into_iter().flatten() {
        let candidate = root.join(relative);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    xwen::hub::cached_file(entry.repo(), relative).with_context(|| {
        format!(
            "cannot find {relative} under {} or its parent, and it is not in the \
             Hugging Face cache",
            encoder_dir.display()
        )
    })
}

/// The prompt indices to run, or all of them.
fn selected_indices() -> Result<Option<BTreeSet<usize>>> {
    match std::env::var_os("XWEN_ZIMAGE_ONLY") {
        Some(raw) => parse_indices(&raw.to_string_lossy()).map(Some),
        None => Ok(None),
    }
}

/// `XWEN_ZIMAGE_ONLY`'s grammar, split out so it is testable without touching
/// the process environment (which a parallel test binary cannot do safely).
fn parse_indices(raw: &str) -> Result<BTreeSet<usize>> {
    let mut set = BTreeSet::new();
    for part in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        set.insert(
            part.parse::<usize>()
                .with_context(|| format!("$XWEN_ZIMAGE_ONLY entry {part:?} is not an index"))?,
        );
    }
    ensure!(
        !set.is_empty(),
        "$XWEN_ZIMAGE_ONLY is set but names no index"
    );
    Ok(set)
}

/// sha256 of a file, via `shasum(1)` — the repo has no hashing crate and this
/// runs at most twice per prompt, behind an opt-in flag.
fn sha256_file(path: &Path) -> Result<String> {
    let out = std::process::Command::new("shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()
        .with_context(|| format!("running shasum on {}", path.display()))?;
    ensure!(
        out.status.success(),
        "shasum failed on {}: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr).trim()
    );
    let stdout = String::from_utf8(out.stdout).context("shasum wrote non-UTF-8")?;
    Ok(stdout
        .split_whitespace()
        .next()
        .context("shasum wrote no digest")?
        .to_string())
}

fn verify_sha(path: &Path, expected: &str, what: &str) -> Result<()> {
    let got = sha256_file(path)?;
    ensure!(
        got == expected,
        "{what} {} has sha256 {got}, the fixture records {expected}: the dump \
         directory is stale or was written by a different reference run",
        path.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// A byte-level diff for the render assertion. The rendered prompt is the input
/// to everything downstream, so a mismatch has to name the offset and show both
/// sides rather than dumping two 2 kB strings.
fn render_mismatch(idx: usize, label: &str, got: &str, want: &str) -> String {
    const TAIL: usize = 160;
    let at = got
        .as_bytes()
        .iter()
        .zip(want.as_bytes())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| got.len().min(want.len()));
    let tail = |s: &str| {
        let bytes = &s.as_bytes()[at.min(s.len())..];
        let shown = &bytes[..bytes.len().min(TAIL)];
        let more = if bytes.len() > TAIL { " …" } else { "" };
        format!("{:?}{more}", String::from_utf8_lossy(shown))
    };
    format!(
        "prompt {idx} ({label}): the rendered prompt differs from the fixture at byte {at} \
         (xwen {} bytes, fixture {} bytes)\n     xwen: {}\n  fixture: {}",
        got.len(),
        want.len(),
        tail(got),
        tail(want)
    )
}

/// Render one prompt the way `encode-text` does: a single user turn, thinking
/// on, under the checkpoint's own dialect.
fn render(text: &str) -> Result<String> {
    let opts = xwen::chat::ChatOptions::for_dialect(xwen::chat::ChatDialect::Qwen3);
    ensure!(
        opts.enable_thinking,
        "the Qwen3 dialect's defaults no longer enable thinking; the reference was \
         dumped with enable_thinking=True"
    );
    xwen::chat::build_prompt(&[xwen::chat::Message::User(text.to_string())], &opts)
        .map_err(|e| anyhow!("rendering the prompt: {e}"))
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

/// One prompt's outcome, for the table and for the failure summary.
struct Row {
    idx: usize,
    label: String,
    tokens: usize,
    fp32: metric::Comparison,
    bf16: metric::Comparison,
}

#[test]
#[ignore = "needs the Z-Image checkpoint, a Metal device and a reference dump"]
fn zimage_encoder_matches_the_fp32_reference() -> Result<()> {
    let reference: Reference = load_json(&fixture_dir().join("reference.json"))?;
    let prompt_file: PromptFile = load_json(&fixture_dir().join("prompts.json"))?;
    let ref_dir = ref_dir()?;
    let selected = selected_indices()?;
    let verify_sha_flag = std::env::var("XWEN_ZIMAGE_VERIFY_SHA").as_deref() == Ok("1");

    let entry = xwen::hub::Model::ZImageTurboEncoder;
    let spec = entry
        .encoder_spec()
        .context("the Z-Image entry carries no encoder spec")?;
    ensure!(
        spec.max_tokens == reference.max_length,
        "the registry truncates at {} tokens, the reference was dumped at {}",
        spec.max_tokens,
        reference.max_length
    );
    ensure!(
        spec.layer == reference.hidden_index,
        "the registry reads hidden_states[{}], the reference dumped [{}]",
        spec.layer,
        reference.hidden_index
    );

    let dir = encoder_dir()?;
    let tokenizer = xwen::tokenizer::LagunaTokenizer::from_file(tokenizer_path(&dir)?)?;

    let device = xwen::gguf::metal_device().context("this test needs the Metal device")?;
    let source = xwen::CheckpointSource::open(&dir, &device, Some(entry))
        .with_context(|| format!("opening the Z-Image text encoder at {}", dir.display()))?;
    let mut model = xwen::XwenModel::load(source, xwen::ops::ExpertRunner::Fused, spec.max_tokens)
        .context("loading the Z-Image text encoder")?;

    let mut rows = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for prompt in &reference.prompts {
        if selected.as_ref().is_some_and(|s| !s.contains(&prompt.idx)) {
            continue;
        }
        let text = &prompt_file
            .prompts
            .iter()
            .find(|p| p.idx == prompt.idx)
            .with_context(|| format!("prompts.json has no prompt {}", prompt.idx))?
            .text;

        // 1. The rendered string, before anything else looks at it.
        let rendered = render(text)?;
        ensure!(
            rendered == prompt.rendered,
            "{}",
            render_mismatch(prompt.idx, &prompt.label, &rendered, &prompt.rendered)
        );

        // 2. The ids, and the encoder's own truncation.
        let full = tokenizer.encode(&rendered)?;
        let truncated = full.len() > spec.max_tokens;
        let ids = &full[..full.len().min(spec.max_tokens)];
        ensure!(
            full.len() == prompt.untruncated_len,
            "prompt {} ({}): xwen tokenizes the rendered prompt to {} ids, the fixture \
             records {}",
            prompt.idx,
            prompt.label,
            full.len(),
            prompt.untruncated_len
        );
        ensure!(
            truncated == prompt.truncated,
            "prompt {} ({}): xwen {} at {} tokens, the fixture records truncated={}",
            prompt.idx,
            prompt.label,
            if truncated {
                "truncates"
            } else {
                "does not truncate"
            },
            spec.max_tokens,
            prompt.truncated
        );
        ensure!(
            ids == prompt.ids.as_slice(),
            "prompt {} ({}): ids differ from the fixture at position {:?}",
            prompt.idx,
            prompt.label,
            ids.iter()
                .zip(&prompt.ids)
                .position(|(a, b)| a != b)
                .map(|at| format!("{at} (xwen {}, fixture {})", ids[at], prompt.ids[at]))
                .unwrap_or_else(|| format!(
                    "<none; lengths {} vs {}>",
                    ids.len(),
                    prompt.ids.len()
                ))
        );
        ensure!(
            ids.len() == prompt.tokens,
            "prompt {}: T disagrees",
            prompt.idx
        );

        // 3. The hidden state.
        let (hidden, tokens) = model
            .encode(ids, spec.layer)
            .with_context(|| format!("encoding prompt {}", prompt.idx))?;
        ensure!(
            tokens == prompt.tokens,
            "prompt {}: encode reports {tokens} tokens, expected {}",
            prompt.idx,
            prompt.tokens
        );
        let hidden = hidden
            .to_device(&candle_core::Device::Cpu)?
            .to_dtype(candle_core::DType::F32)?;
        let (got_rows, got_cols) = hidden.dims2().context("encode must return [T, hidden]")?;
        ensure!(
            (got_rows, got_cols) == (prompt.tokens, reference.hidden_size),
            "prompt {}: encode returned [{got_rows}, {got_cols}], expected [{}, {}]",
            prompt.idx,
            prompt.tokens,
            reference.hidden_size
        );
        let candidate: Vec<f32> = hidden.flatten_all()?.to_vec1()?;

        let dump = ref_dir.join(&prompt.dir);
        let fp32_path = dump.join(&prompt.hidden_fp32_safetensors);
        let bf16_path = dump.join(&prompt.hidden_bf16_npy);
        if verify_sha_flag {
            verify_sha(
                &fp32_path,
                &prompt.hidden_fp32_safetensors_sha256,
                "the fp32 reference",
            )?;
            verify_sha(
                &bf16_path,
                &prompt.hidden_bf16_npy_sha256,
                "the bf16 reference",
            )?;
        }

        let fp32 = candle_core::safetensors::load(&fp32_path, &candle_core::Device::Cpu)
            .with_context(|| format!("reading {}", fp32_path.display()))?
            .remove("hidden")
            .with_context(|| format!("{} has no `hidden` tensor", fp32_path.display()))?
            .to_dtype(candle_core::DType::F32)?;
        let (fp32_rows, fp32_cols) = fp32.dims2()?;
        ensure!(
            (fp32_rows, fp32_cols) == (prompt.tokens, reference.hidden_size),
            "{} holds [{fp32_rows}, {fp32_cols}], the fixture says [{}, {}]",
            fp32_path.display(),
            prompt.tokens,
            reference.hidden_size
        );
        let fp32: Vec<f32> = fp32.flatten_all()?.to_vec1()?;

        let (bf16, bf16_rows, bf16_cols) = npy::read_f32_2d(&bf16_path)?;
        ensure!(
            (bf16_rows, bf16_cols) == (prompt.tokens, reference.hidden_size),
            "{} holds [{bf16_rows}, {bf16_cols}], the fixture says [{}, {}]",
            bf16_path.display(),
            prompt.tokens,
            reference.hidden_size
        );

        let against_fp32 =
            metric::compare(&candidate, &fp32, prompt.tokens, reference.hidden_size)?;
        // Diagnostic only: how far the pipeline's own bf16 arithmetic sits from
        // the same fp32 reference. It is the context for a marginal xwen result,
        // never a bar (docs/parity.md; the bf16 run does not itself clear 0.9999).
        let bf16_vs_fp32 = metric::compare(&bf16, &fp32, prompt.tokens, reference.hidden_size)?;

        if !against_fp32.failures.is_empty() {
            failures.push(describe_failure(prompt, &against_fp32, &bf16_vs_fp32));
        }

        rows.push(Row {
            idx: prompt.idx,
            label: prompt.label.clone(),
            tokens: prompt.tokens,
            fp32: against_fp32,
            bf16: bf16_vs_fp32,
        });
    }

    ensure!(
        !rows.is_empty(),
        "no prompt was run; check $XWEN_ZIMAGE_ONLY"
    );
    print_table(&rows);

    ensure!(
        failures.is_empty(),
        "{} of {} prompts miss the bar (cosine >= {}, rel <= {}):\n{}",
        failures.len(),
        rows.len(),
        metric::COS_MIN,
        metric::REL_MAX,
        failures.join("\n")
    );
    Ok(())
}

fn describe_failure(
    prompt: &RefPrompt,
    fp32: &metric::Comparison,
    bf16: &metric::Comparison,
) -> String {
    let scope = if fp32.only_position_zero_failed() {
        format!(
            "ONLY position 0 failed (the <|im_start|> massive activation, reference \
             magnitude {:.1}); every other position is inside both bars",
            fp32.pos0.ref_max_abs
        )
    } else if fp32.failures.contains(&0) {
        format!(
            "{} positions failed, position 0 among them",
            fp32.failures.len()
        )
    } else {
        format!(
            "{} positions failed, position 0 is not one of them",
            fp32.failures.len()
        )
    };
    let head: Vec<String> = fp32.failures.iter().take(8).map(usize::to_string).collect();
    let more = if fp32.failures.len() > head.len() {
        format!(", … ({} more)", fp32.failures.len() - head.len())
    } else {
        String::new()
    };
    format!(
        "  prompt {} ({}, T={}): {scope}\n    failing positions: {}{}\n    \
         pos 0: cos {:.8} rel {:.5} |ref|max {:.1}\n    pos>=1: min cos {} \
         max rel {}\n    bf16 reference on the same rows: min cos {} max rel {}",
        prompt.idx,
        prompt.label,
        prompt.tokens,
        head.join(", "),
        more,
        fp32.pos0.cosine,
        fp32.pos0.rel,
        fp32.pos0.ref_max_abs,
        fmt_opt(fp32.pooled_min_cosine, fp32.worst_cosine_pos, 8),
        fmt_opt(fp32.pooled_max_rel, fp32.worst_rel_pos, 5),
        fmt_opt(bf16.pooled_min_cosine, bf16.worst_cosine_pos, 8),
        fmt_opt(bf16.pooled_max_rel, bf16.worst_rel_pos, 5),
    )
}

fn fmt_opt(value: Option<f64>, at: Option<usize>, digits: usize) -> String {
    match (value, at) {
        (Some(v), Some(at)) => format!("{v:.digits$} (pos {at})"),
        _ => "n/a".to_string(),
    }
}

fn print_table(rows: &[Row]) {
    println!(
        "\n  Z-Image text encoder vs the fp32 reference — bars: cosine >= {}, rel <= {}",
        metric::COS_MIN,
        metric::REL_MAX
    );
    println!(
        "  {:>3}  {:<22} {:>5}  {:>12} {:>10}  {:>12} {:>10} {:>10}  {:>12} {:>10}",
        "idx",
        "label",
        "T",
        "min cos>=1",
        "max rel>=1",
        "pos0 cos",
        "pos0 rel",
        "pos0 |r|",
        "bf16 min cos",
        "bf16 max rel"
    );
    for row in rows {
        println!(
            "  {:>3}  {:<22} {:>5}  {:>12} {:>10}  {:>12.8} {:>10.5} {:>10.1}  {:>12} {:>10}",
            row.idx,
            row.label,
            row.tokens,
            row.fp32
                .pooled_min_cosine
                .map_or_else(|| "n/a".into(), |v| format!("{v:.8}")),
            row.fp32
                .pooled_max_rel
                .map_or_else(|| "n/a".into(), |v| format!("{v:.5}")),
            row.fp32.pos0.cosine,
            row.fp32.pos0.rel,
            row.fp32.pos0.ref_max_abs,
            row.bf16
                .pooled_min_cosine
                .map_or_else(|| "n/a".into(), |v| format!("{v:.8}")),
            row.bf16
                .pooled_max_rel
                .map_or_else(|| "n/a".into(), |v| format!("{v:.5}")),
        );
    }
    println!();
}

// ---------------------------------------------------------------------------
// CPU unit tests: the metric math and the .npy reader, on synthetic data.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod metric_tests {
    use super::metric::*;

    /// Cosine on rows whose answer is known by hand, including the two
    /// degenerate ones.
    #[test]
    fn cosine_matches_hand_computed_values() {
        assert!((cosine(&[3.0, 4.0], &[3.0, 4.0]) - 1.0).abs() < 1e-12);
        assert!((cosine(&[1.0, 0.0], &[0.0, 1.0]) - 0.0).abs() < 1e-12);
        assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-12);
        // 45 degrees: [1,1] against [1,0].
        assert!((cosine(&[1.0, 1.0], &[1.0, 0.0]) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        // Scale-invariant, which is why the relative-error metric exists too.
        assert!((cosine(&[100.0, 200.0], &[1.0, 2.0]) - 1.0).abs() < 1e-12);
        // Two zero rows are the same row; one zero row is orthogonal to anything.
        assert_eq!(cosine(&[0.0, 0.0], &[0.0, 0.0]), 1.0);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    /// The relative error divides the largest absolute deviation by the
    /// REFERENCE row's largest magnitude, not by the candidate's and not
    /// elementwise.
    #[test]
    fn relative_error_uses_the_reference_rows_own_magnitude() {
        // Every value here is exact in binary, so the expectations are equalities
        // rather than tolerances. max |diff| = 0.5 at element 0; max |ref| = 4.0
        // at element 1.
        assert_eq!(max_abs_rel_error(&[3.5, -4.0], &[3.0, -4.0]), 0.125);
        // A deviation that is total on a small element still measures against
        // the row: 2^-10 missed entirely, over a row whose magnitude is 128.
        assert_eq!(
            max_abs_rel_error(&[0.0, 128.0], &[0.0009765625, 128.0]),
            0.0009765625 / 128.0
        );
        assert_eq!(max_abs_rel_error(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    }

    /// A reference row with no magnitude divides by the floor instead, so the
    /// metric reports an absolute error rather than infinity.
    #[test]
    fn the_denominator_floor_catches_an_all_zero_reference_row() {
        // A power of two, so `tiny as f64` is exact and the expectation is the
        // metric's own arithmetic rather than a decimal literal it can only
        // approximate.
        let tiny = 2.0f32.powi(-30);
        let err = max_abs_rel_error(&[tiny, 0.0], &[0.0, 0.0]);
        assert_eq!(err, f64::from(tiny) / REL_DENOM_FLOOR);
        assert!(err.is_finite());

        // Above the floor the row's own magnitude takes over again: the
        // deviation here is exactly the reference element, so the ratio is 1.
        let err = max_abs_rel_error(&[2e-3, 0.0], &[1e-3, 0.0]);
        assert_eq!(err, 1.0, "{err}");
    }

    #[test]
    fn max_abs_is_the_rows_largest_magnitude() {
        assert_eq!(max_abs(&[1.0, -7.5, 3.0]), 7.5);
        assert_eq!(max_abs(&[0.0, 0.0]), 0.0);
    }

    /// Position 0 is reported on its own and excluded from the pooled figures,
    /// while the pass/fail verdict still covers it.
    #[test]
    fn position_zero_is_split_out_of_the_pooled_figures() {
        let hidden = 2;
        // Row 0 is off by 10% of its own magnitude; rows 1 and 2 are exact.
        let reference = vec![10.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let candidate = vec![11.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let cmp = compare(&candidate, &reference, 3, hidden).unwrap();

        assert_eq!(cmp.tokens, 3);
        assert!((cmp.pos0.rel - 0.1).abs() < 1e-9, "{:?}", cmp.pos0);
        assert_eq!(cmp.pos0.ref_max_abs, 10.0);
        assert!((cmp.pooled_min_cosine.unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(cmp.pooled_max_rel.unwrap(), 0.0);
        assert_eq!(cmp.failures, vec![0]);
        assert!(cmp.only_position_zero_failed());
    }

    /// The mirror image: a bad row past position 0 shows up in the pooled
    /// figures, is located, and is not mistaken for the position-0 case.
    #[test]
    fn a_bad_row_past_position_zero_is_pooled_and_located() {
        let reference = vec![1.0, 0.0, 10.0, 0.0, 5.0, 0.0];
        let candidate = vec![1.0, 0.0, 10.0, 9.0, 5.0, 0.0];
        let cmp = compare(&candidate, &reference, 3, 2).unwrap();

        assert!(cmp.pos0.passes());
        assert_eq!(cmp.failures, vec![1]);
        assert!(!cmp.only_position_zero_failed());
        assert_eq!(cmp.worst_cosine_pos, Some(1));
        assert_eq!(cmp.worst_rel_pos, Some(1));
        assert!((cmp.pooled_max_rel.unwrap() - 0.9).abs() < 1e-9);
    }

    /// A one-token sequence has nothing to pool, and says so rather than
    /// reporting position 0 twice.
    #[test]
    fn a_single_token_sequence_pools_nothing() {
        let cmp = compare(&[1.0, 2.0], &[1.0, 2.0], 1, 2).unwrap();
        assert!(cmp.pooled_min_cosine.is_none());
        assert!(cmp.pooled_max_rel.is_none());
        assert!(cmp.failures.is_empty());
    }

    /// The bars are the plan's, and a row exactly on them passes.
    #[test]
    fn the_bars_are_inclusive() {
        let on_the_bar = TokenMetrics {
            cosine: COS_MIN,
            rel: REL_MAX,
            ref_max_abs: 1.0,
        };
        assert!(on_the_bar.passes());
        assert!(
            !TokenMetrics {
                cosine: COS_MIN - 1e-9,
                ..on_the_bar
            }
            .passes()
        );
        assert!(
            !TokenMetrics {
                rel: REL_MAX + 1e-9,
                ..on_the_bar
            }
            .passes()
        );
    }

    /// A shape the caller got wrong is an error, not a silently short compare.
    #[test]
    fn mismatched_lengths_are_an_error() {
        assert!(compare(&[1.0, 2.0, 3.0], &[1.0, 2.0], 1, 2).is_err());
        assert!(compare(&[1.0, 2.0], &[1.0, 2.0], 0, 2).is_err());
    }
}

#[cfg(test)]
mod npy_tests {
    use super::npy::parse_f32_2d;

    /// Build the same bytes `numpy.save` writes: the v1 magic, a padded dict
    /// literal, then C-order little-endian data.
    fn npy_v1(descr: &str, fortran: &str, shape: &str, data: &[f32]) -> Vec<u8> {
        let mut header =
            format!("{{'descr': '{descr}', 'fortran_order': {fortran}, 'shape': {shape}, }}");
        // numpy pads the header with spaces to a 64-byte boundary and ends it
        // with a newline; the reader must not depend on the padding length.
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        for v in data {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn reads_a_hand_built_c_order_float32_array() {
        let data = [1.0f32, -2.5, 3.25, 4.0, 5.5, -6.0];
        let bytes = npy_v1("<f4", "False", "(2, 3)", &data);
        let (values, rows, cols) = parse_f32_2d(&bytes).unwrap();
        assert_eq!((rows, cols), (2, 3));
        assert_eq!(values, data);
    }

    /// The v2 header (4-byte length) is the same array with a wider prefix.
    #[test]
    fn reads_a_version_2_header() {
        let header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (1, 2), }\n";
        let mut bytes = b"\x93NUMPY\x02\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        let (values, rows, cols) = parse_f32_2d(&bytes).unwrap();
        assert_eq!((rows, cols), (1, 2));
        assert_eq!(values, vec![1.0, 2.0]);
    }

    /// Everything this reader does not handle is refused. Reading any of these
    /// as if it were the supported layout would produce plausible numbers and a
    /// wrong verdict, which is the whole reason the reader is strict.
    #[test]
    fn refuses_layouts_it_cannot_read() {
        let data = [1.0f32, 2.0];
        assert!(
            parse_f32_2d(&npy_v1("<f8", "False", "(1, 2)", &data)).is_err(),
            "f64 descr"
        );
        assert!(
            parse_f32_2d(&npy_v1(">f4", "False", "(1, 2)", &data)).is_err(),
            "big endian"
        );
        assert!(
            parse_f32_2d(&npy_v1("<f4", "True", "(1, 2)", &data)).is_err(),
            "fortran order"
        );
        assert!(
            parse_f32_2d(&npy_v1("<f4", "False", "(2,)", &data)).is_err(),
            "1-D"
        );
        assert!(
            parse_f32_2d(&npy_v1("<f4", "False", "(1, 2, 1)", &data)).is_err(),
            "3-D"
        );
        // A shape that does not match the payload.
        assert!(
            parse_f32_2d(&npy_v1("<f4", "False", "(3, 2)", &data)).is_err(),
            "short data"
        );
        // Not a .npy file at all.
        assert!(parse_f32_2d(b"not a numpy file at all").is_err());
        assert!(parse_f32_2d(b"").is_err());
    }

    /// A 1-row array still parses: the shape tuple's trailing comma habit and a
    /// single row are both ordinary here.
    #[test]
    fn reads_a_single_row() {
        let bytes = npy_v1("<f4", "False", "(1, 1)", &[42.0]);
        let (values, rows, cols) = parse_f32_2d(&bytes).unwrap();
        assert_eq!((rows, cols), (1, 1));
        assert_eq!(values, vec![42.0]);
    }
}

#[cfg(test)]
mod fixture_tests {
    use super::*;

    /// The committed fixture is self-consistent, which is checkable with no
    /// model, no GPU and no dump directory: every prompt in `reference.json`
    /// has its text in `prompts.json`, its recorded `T` is its id count, and
    /// `truncated` agrees with the 512-token cap the registry entry carries.
    #[test]
    fn the_committed_fixture_is_self_consistent() {
        let reference: Reference = load_json(&fixture_dir().join("reference.json")).unwrap();
        let prompts: PromptFile = load_json(&fixture_dir().join("prompts.json")).unwrap();
        let spec = xwen::hub::Model::ZImageTurboEncoder.encoder_spec().unwrap();

        assert_eq!(spec.max_tokens, reference.max_length);
        assert_eq!(spec.layer, reference.hidden_index);
        assert!(!reference.prompts.is_empty());

        for prompt in &reference.prompts {
            assert!(
                prompts.prompts.iter().any(|p| p.idx == prompt.idx),
                "prompt {} has no text in prompts.json",
                prompt.idx
            );
            assert_eq!(prompt.ids.len(), prompt.tokens, "prompt {}", prompt.idx);
            assert!(prompt.tokens <= spec.max_tokens, "prompt {}", prompt.idx);
            assert_eq!(
                prompt.truncated,
                prompt.untruncated_len > spec.max_tokens,
                "prompt {}",
                prompt.idx
            );
            assert!(
                prompt.rendered.ends_with("<|im_start|>assistant\n"),
                "prompt {} does not end on the generation header",
                prompt.idx
            );
            assert!(!prompt.dir.is_empty() && !prompt.rendered_sha256.is_empty());
        }
    }

    /// xwen's own renderer reproduces the fixture's rendered string. This is
    /// step 1 of the gate, and it needs neither the checkpoint nor the dump —
    /// so it runs here too, where every `cargo test` sees it.
    #[test]
    fn the_renderer_reproduces_every_fixture_prompt() {
        let reference: Reference = load_json(&fixture_dir().join("reference.json")).unwrap();
        let prompts: PromptFile = load_json(&fixture_dir().join("prompts.json")).unwrap();

        for prompt in &reference.prompts {
            let text = &prompts
                .prompts
                .iter()
                .find(|p| p.idx == prompt.idx)
                .unwrap()
                .text;
            let rendered = render(text).unwrap();
            assert!(
                rendered == prompt.rendered,
                "{}",
                render_mismatch(prompt.idx, &prompt.label, &rendered, &prompt.rendered)
            );
        }
    }

    /// `XWEN_ZIMAGE_ONLY` parsing, which decides which prompts run. Whitespace
    /// and trailing separators are tolerated; anything that is not an index is
    /// an error rather than a silently narrower run.
    #[test]
    fn the_subset_filter_parses_indices() {
        let set = parse_indices(" 0, 11 ,4").unwrap();
        assert_eq!(set.into_iter().collect::<Vec<_>>(), vec![0, 4, 11]);
        assert_eq!(
            parse_indices("7,").unwrap().into_iter().collect::<Vec<_>>(),
            vec![7]
        );
        assert!(parse_indices("0,nope").is_err());
        assert!(parse_indices(" , ").is_err());
        assert!(parse_indices("").is_err());
        assert!(parse_indices("-1").is_err());
    }

    /// The diff message names the first differing byte and shows both tails,
    /// which is what makes a template regression readable.
    #[test]
    fn the_render_diff_names_the_first_differing_byte() {
        let message = render_mismatch(3, "landscape-storm", "abcXef", "abcdef");
        assert!(message.contains("byte 3"), "{message}");
        assert!(message.contains("\"Xef\""), "{message}");
        assert!(message.contains("\"def\""), "{message}");
        assert!(message.contains("prompt 3 (landscape-storm)"), "{message}");
    }
}
