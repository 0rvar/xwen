//! Stage 1 of the dense Qwen3-4B parity gate: xwen's per-position logits against
//! llama.cpp's, over the committed prompt fixtures.
//!
//! Two things live in this file and they have different requirements.
//!
//! The comparison math is [`metrics`]: pure functions over one row of logits,
//! unit tested on synthetic rows. `cargo test --test qwen3_parity` runs those on
//! any machine, with no model, no GPU and no oracle directory — they are what
//! keeps "argmax agrees", "top-5 overlap" and "near tie" from quietly meaning
//! something else after an edit.
//!
//! The gate itself is [`qwen3_4b_logits_match_the_llamacpp_oracle`], which is
//! `#[ignore]`d because it needs a Metal device, the Qwen3-4B checkpoint and a
//! directory of reference logits written by `scripts/qwen3-ref-logits.ts`:
//!
//! ```text
//! XWEN_QWEN3_PARITY_DIR=/tmp/qwen3-ref-logits-cpu \
//!   cargo test --release --test qwen3_parity -- --ignored --nocapture
//! ```
//!
//! `--nocapture` is not decoration: the per-prompt table is printed on a PASS as
//! well as a failure, and cargo swallows a passing test's stdout without it. On a
//! failure the same table is repeated inside the panic message, so a captured run
//! still says what went wrong.
//!
//! Environment:
//!
//! - `XWEN_QWEN3_PARITY_DIR` (required) — the oracle directory holding
//!   `manifest.json` and the per-prompt `.f32` / `.json` files.
//! - `XWEN_QWEN3_DIR` — a Qwen3-4B safetensors directory (or the `config.json`
//!   inside one). Unset means the registry entry's own cached Hugging Face
//!   snapshot, and a cache miss is an error naming `xwen fetch`; this test never
//!   downloads 8 GB behind the runner's back.
//! - `XWEN_QWEN3_ENTRY` — the registry checkpoint the directory is, as a
//!   `--model-size` alias. Defaults to `qwen3-4b`; the entry decides where the
//!   loader looks for the tokenizer and which planes may be zero-filled.
//! - `XWEN_QWEN3_PARITY_ONLY` — comma-separated manifest indices, to run a
//!   subset (`0,8,11` is the three cheapest prompts).
//! - `XWEN_QWEN3_PARITY_VERIFY_SHA=1` — re-hash every `.f32` and check it
//!   against the manifest before comparing. Off by default: it reads several GB.
//! - `XWEN_QWEN3_PARITY_CHUNK` — prefill chunk size, defaulting to the model's
//!   own. Only the peak host allocation changes with it, not the arithmetic.
//! - `XWEN_QWEN3_PARITY_LIST` — how many failing positions the panic message
//!   spells out before summarising the rest. Default 50.
//!
//! The bars (`metrics::MAX_ABS_BAR`, argmax 100%, `metrics::TOP5_BAR`) and the
//! near-tie band come from docs/parity.md. Near ties are REPORTED, never
//! excused: an argmax flip the reference itself decided by less than the band is
//! counted and printed, and it still fails the gate.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

// ---------------------------------------------------------------------------
// The comparison math.
// ---------------------------------------------------------------------------

/// Per-position and pooled agreement between two rows of logits.
///
/// Nothing here knows about models, files or devices, which is why it can be
/// tested on ten-wide synthetic rows.
mod metrics {
    /// The largest per-position, per-vocabulary-entry absolute logit difference
    /// the gate allows. docs/parity.md owns the number; it is tighter than the
    /// repo's older cosine bar and was committed to only after the oracle's
    /// arithmetic was pinned.
    pub const MAX_ABS_BAR: f32 = 2e-2;

    /// Pooled top-5 agreement floor. Per position the overlap can only move in
    /// 20% steps, so a fraction like 99.9% is meaningful only over the pool.
    pub const TOP5_BAR: f64 = 0.999;

    /// An argmax flip is called a near tie when the REFERENCE's own top-1
    /// beat its top-2 by less than this. It describes the flip; it does not
    /// forgive it.
    pub const NEAR_TIE_MARGIN: f32 = 2e-2;

    /// The k of the pooled top-k agreement.
    pub const TOP_K: usize = 5;

    /// The `k` highest-scoring ids, best first.
    ///
    /// Ties break to the LOWER id, on both sides of the comparison, so a tie is
    /// never itself a disagreement. That falls out of scanning ascending and
    /// inserting only on a strict improvement: an equal value arriving later
    /// carries a higher index and loses.
    ///
    /// A non-finite entry ranks as negative infinity, below every real logit.
    /// It is not enough to let NaN lose its comparisons: NaN loses `>` in both
    /// directions, so it slips into a not-yet-full list unchallenged and then,
    /// sitting in the worst slot, makes the full-list guard `v > worst` false
    /// for every later value — silently truncating the ranking to whatever
    /// preceded it. The count of such entries is what fails the gate; the
    /// ranking stays the real one so the failure report still reads.
    pub fn top_k(row: &[f32], k: usize) -> Vec<u32> {
        let k = k.min(row.len());
        if k == 0 {
            return Vec::new();
        }
        let mut best: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
        for (i, &raw) in row.iter().enumerate() {
            let v = if raw.is_finite() {
                raw
            } else {
                f32::NEG_INFINITY
            };
            if best.len() == k && !(v > best[k - 1].0) {
                continue;
            }
            // `best` is sorted descending; land after every value >= ours so the
            // earlier of two equal values keeps the better slot.
            let at = best.partition_point(|&(bv, _)| bv >= v);
            best.insert(at, (v, i as u32));
            if best.len() > k {
                best.pop();
            }
        }
        best.into_iter().map(|(_, i)| i).collect()
    }

    /// How one position compares. Both margins are `top1 - top2` on that side's
    /// own row: the reference's is what decides a near tie, and the candidate's
    /// is printed beside it so a flip's shape is readable without a second run.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct RowMetrics {
        pub max_abs: f32,
        pub reference_top1: u32,
        pub candidate_top1: u32,
        pub reference_margin: f32,
        pub candidate_margin: f32,
        pub top5_overlap: usize,
        /// Vocabulary entries at this position where EITHER side was not
        /// finite. A non-finite difference compares false against every bound,
        /// so without this counter a NaN row would read as a perfect one.
        pub nonfinite: usize,
    }

    impl RowMetrics {
        pub fn argmax_agrees(&self) -> bool {
            self.reference_top1 == self.candidate_top1
        }

        /// Whether the reference itself barely made this call.
        pub fn near_tie(&self) -> bool {
            self.reference_margin < NEAR_TIE_MARGIN
        }
    }

    /// Compare one position. Both rows are the full vocabulary and must be the
    /// same width; a width mismatch is a harness bug, not a parity result.
    pub fn row_metrics(candidate: &[f32], reference: &[f32]) -> RowMetrics {
        assert_eq!(
            candidate.len(),
            reference.len(),
            "row width mismatch: candidate {} vs reference {}",
            candidate.len(),
            reference.len()
        );
        assert!(
            reference.len() >= 2,
            "a vocabulary of {} cannot form the top-2 a margin needs",
            reference.len()
        );

        let mut max_abs = 0.0f32;
        let mut nonfinite = 0usize;
        for (&c, &r) in candidate.iter().zip(reference.iter()) {
            if !c.is_finite() || !r.is_finite() {
                nonfinite += 1;
                continue;
            }
            let d = (c - r).abs();
            if d > max_abs {
                max_abs = d;
            }
        }

        let want = TOP_K.max(2);
        let reference_top = top_k(reference, want);
        let candidate_top = top_k(candidate, want);

        let margin = |row: &[f32], top: &[u32]| -> f32 {
            match (top.first(), top.get(1)) {
                (Some(&a), Some(&b)) => row[a as usize] - row[b as usize],
                _ => f32::NAN,
            }
        };

        let head = TOP_K.min(reference_top.len());
        let reference_set: std::collections::BTreeSet<u32> =
            reference_top[..head].iter().copied().collect();
        let top5_overlap = candidate_top[..TOP_K.min(candidate_top.len())]
            .iter()
            .filter(|id| reference_set.contains(id))
            .count();

        RowMetrics {
            max_abs,
            reference_top1: reference_top[0],
            candidate_top1: candidate_top[0],
            reference_margin: margin(reference, &reference_top),
            candidate_margin: margin(candidate, &candidate_top),
            top5_overlap,
            nonfinite,
        }
    }

    /// Everything the gate judges, accumulated over positions. Pooling is the
    /// point: a per-position top-5 overlap is one of six values, and the floor
    /// the gate sets lives between two of them.
    #[derive(Debug, Default, Clone, Copy, PartialEq)]
    pub struct Pool {
        pub positions: usize,
        pub max_abs: f32,
        pub argmax_agree: usize,
        pub top5_overlap: usize,
        pub near_tie_flips: usize,
        pub nonfinite: usize,
    }

    impl Pool {
        pub fn observe(&mut self, m: &RowMetrics) {
            self.positions += 1;
            if m.max_abs > self.max_abs {
                self.max_abs = m.max_abs;
            }
            if m.argmax_agrees() {
                self.argmax_agree += 1;
            } else if m.near_tie() {
                self.near_tie_flips += 1;
            }
            self.top5_overlap += m.top5_overlap;
            self.nonfinite += m.nonfinite;
        }

        pub fn merge(&mut self, other: &Pool) {
            self.positions += other.positions;
            if other.max_abs > self.max_abs {
                self.max_abs = other.max_abs;
            }
            self.argmax_agree += other.argmax_agree;
            self.top5_overlap += other.top5_overlap;
            self.near_tie_flips += other.near_tie_flips;
            self.nonfinite += other.nonfinite;
        }

        /// Sum of per-position overlaps over `TOP_K x positions`. NaN over an
        /// empty pool, so an empty run reads as "no answer" rather than as a
        /// perfect one.
        pub fn top5_agreement(&self) -> f64 {
            if self.positions == 0 {
                return f64::NAN;
            }
            self.top5_overlap as f64 / (TOP_K * self.positions) as f64
        }

        pub fn argmax_agreement(&self) -> f64 {
            if self.positions == 0 {
                return f64::NAN;
            }
            self.argmax_agree as f64 / self.positions as f64
        }

        /// The whole bar in one place: every position scored, none non-finite,
        /// max-abs under the band, every argmax agreeing, pooled top-5 at the
        /// floor.
        pub fn passes(&self) -> bool {
            self.positions > 0
                && self.nonfinite == 0
                && self.max_abs <= MAX_ABS_BAR
                && self.argmax_agree == self.positions
                && self.top5_agreement() >= TOP5_BAR
        }
    }
}

mod metrics_tests {
    use super::metrics::*;

    /// A row builder: index i gets value `vals[i]`, everything else a floor.
    fn row(vals: &[f32]) -> Vec<f32> {
        vals.to_vec()
    }

    #[test]
    fn top_k_takes_the_largest_best_first() {
        let r = row(&[1.0, 9.0, 3.0, 7.0, 0.0, 8.0, 2.0, 6.0, 4.0, 5.0]);
        assert_eq!(top_k(&r, 5), vec![1, 5, 3, 7, 9]);
        assert_eq!(top_k(&r, 1), vec![1]);
    }

    #[test]
    fn top_k_breaks_ties_towards_the_lower_id() {
        // Three entries share the top value; the first three ids must win, in
        // ascending order, or the two sides of a comparison could disagree on a
        // tie and report it as a divergence.
        let r = row(&[5.0, 5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(top_k(&r, 3), vec![0, 1, 2]);
        assert_eq!(top_k(&r, 5), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn identical_rows_agree_everywhere() {
        let r = row(&[0.5, 3.0, 2.0, 1.0, 0.25, 0.1, 0.0, -1.0, -2.0, -3.0]);
        let m = row_metrics(&r, &r);
        assert_eq!(m.max_abs, 0.0);
        assert_eq!(m.reference_top1, 1);
        assert!(m.argmax_agrees());
        assert_eq!(m.top5_overlap, 5);
        assert_eq!(m.nonfinite, 0);

        let mut pool = Pool::default();
        pool.observe(&m);
        assert_eq!(pool.top5_agreement(), 1.0);
        assert_eq!(pool.argmax_agreement(), 1.0);
        assert!(pool.passes());
    }

    #[test]
    fn an_argmax_flip_inside_the_band_is_reported_as_a_near_tie() {
        // The reference separates its top two by 0.005, under the 2e-2 band.
        let reference = row(&[0.0, 3.000, 2.995, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]);
        let candidate = row(&[0.0, 2.995, 3.000, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]);
        let m = row_metrics(&candidate, &reference);
        assert_eq!(m.reference_top1, 1);
        assert_eq!(m.candidate_top1, 2);
        assert!(!m.argmax_agrees());
        assert!(
            m.near_tie(),
            "reference margin {} should be inside the band",
            m.reference_margin
        );
        // The set is unchanged by the swap, so top-5 agreement is blind to it.
        assert_eq!(m.top5_overlap, 5);

        let mut pool = Pool::default();
        pool.observe(&m);
        assert_eq!(pool.near_tie_flips, 1);
        assert_eq!(pool.argmax_agree, 0);
        // A near tie is described, not excused.
        assert!(!pool.passes());
    }

    #[test]
    fn an_argmax_flip_outside_the_band_is_not_a_near_tie() {
        let reference = row(&[0.0, 9.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0]);
        let candidate = row(&[0.0, 1.0, 9.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0]);
        let m = row_metrics(&candidate, &reference);
        assert!(!m.argmax_agrees());
        assert!(!m.near_tie());
        assert!((m.reference_margin - 8.0).abs() < 1e-6);
        assert!((m.candidate_margin - 8.0).abs() < 1e-6);

        let mut pool = Pool::default();
        pool.observe(&m);
        assert_eq!(pool.near_tie_flips, 0);
    }

    #[test]
    fn a_partial_top5_overlap_pools_towards_the_floor() {
        // Reference top-5 is {1,2,3,4,5}; the candidate keeps 1,2,3 and swaps
        // 4,5 for 8,9 — three of five.
        let reference = row(&[0.0, 9.0, 8.0, 7.0, 6.0, 5.0, 0.1, 0.2, 0.3, 0.4]);
        let candidate = row(&[0.0, 9.0, 8.0, 7.0, 0.5, 0.6, 0.1, 0.2, 6.0, 5.0]);
        let m = row_metrics(&candidate, &reference);
        assert!(m.argmax_agrees());
        assert_eq!(m.top5_overlap, 3);

        let mut pool = Pool::default();
        pool.observe(&m);
        // One perfect row alongside it: (5 + 3) / (5 * 2).
        pool.observe(&row_metrics(&reference, &reference));
        assert_eq!(pool.positions, 2);
        assert!((pool.top5_agreement() - 0.8).abs() < 1e-12);
        assert!(!pool.passes());
    }

    #[test]
    fn max_abs_is_the_worst_entry_and_pools_as_a_maximum() {
        let reference = row(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let mut candidate = reference.clone();
        candidate[6] += 0.003;
        let small = row_metrics(&candidate, &reference);
        assert!((small.max_abs - 0.003).abs() < 1e-6);

        let mut candidate = reference.clone();
        candidate[0] -= 0.5;
        let large = row_metrics(&candidate, &reference);
        assert!((large.max_abs - 0.5).abs() < 1e-6);

        let mut pool = Pool::default();
        pool.observe(&small);
        assert!(pool.passes(), "0.003 is inside the {MAX_ABS_BAR} band");
        pool.observe(&large);
        assert!((pool.max_abs - 0.5).abs() < 1e-6);
        assert!(!pool.passes());
    }

    #[test]
    fn a_nonfinite_entry_is_counted_and_fails_on_either_side() {
        let reference = row(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let mut candidate = reference.clone();
        candidate[3] = f32::NAN;
        let m = row_metrics(&candidate, &reference);
        assert_eq!(m.nonfinite, 1);
        // A NaN ranks below every real logit, so the ranking is still the real
        // one — and, crucially, the entries AFTER it are still ranked at all.
        assert_eq!(m.candidate_top1, 9);
        assert_eq!(top_k(&candidate, 5), vec![9, 8, 7, 6, 5]);
        assert_eq!(m.max_abs, 0.0);

        let mut pool = Pool::default();
        pool.observe(&m);
        assert!(!pool.passes());

        // A non-finite ORACLE entry is a broken oracle, and is counted too:
        // otherwise the row it sits in would read as agreement.
        let mut broken = reference.clone();
        broken[3] = f32::INFINITY;
        let m = row_metrics(&reference, &broken);
        assert_eq!(m.nonfinite, 1);
        let mut pool = Pool::default();
        pool.observe(&m);
        assert!(!pool.passes());
    }

    #[test]
    fn a_nan_in_the_ranking_does_not_truncate_it() {
        // The regression this file was written to hold: with a plain `v > worst`
        // guard, a NaN that reaches the worst slot blocks every later insertion,
        // because both `>` comparisons against it are false. The five largest
        // values here all come after the NaN.
        let mut r = vec![0.0f32; 10];
        r[0] = f32::NAN;
        for (i, slot) in r.iter_mut().enumerate().skip(1) {
            *slot = i as f32;
        }
        assert_eq!(top_k(&r, 5), vec![9, 8, 7, 6, 5]);
        assert_eq!(top_k(&r, 1), vec![9]);
        // It sorts below every real value, so it surfaces only once k covers
        // the whole row, and then it is last.
        assert_eq!(*top_k(&r, 10).last().unwrap(), 0);
    }

    #[test]
    fn an_empty_pool_answers_nan_rather_than_perfect() {
        let pool = Pool::default();
        assert!(pool.top5_agreement().is_nan());
        assert!(pool.argmax_agreement().is_nan());
        assert!(!pool.passes());
    }

    #[test]
    fn merge_is_the_same_as_observing_into_one_pool() {
        let reference = row(&[0.0, 9.0, 8.0, 7.0, 6.0, 5.0, 0.1, 0.2, 0.3, 0.4]);
        let candidate = row(&[0.0, 9.0, 8.0, 7.0, 0.5, 0.6, 0.1, 0.2, 6.0, 5.0]);
        let a = row_metrics(&candidate, &reference);
        let b = row_metrics(&reference, &reference);

        let mut one = Pool::default();
        one.observe(&a);
        one.observe(&b);

        let mut left = Pool::default();
        left.observe(&a);
        let mut right = Pool::default();
        right.observe(&b);
        left.merge(&right);

        assert_eq!(one, left);
    }
}

// ---------------------------------------------------------------------------
// The committed fixtures, which need neither a model nor an oracle.
// ---------------------------------------------------------------------------

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture_path() -> PathBuf {
    manifest_dir().join("tests/fixtures/qwen3-prompts.json")
}

/// One entry of `tests/fixtures/qwen3-prompts.json`.
struct Fixture {
    id: String,
    ids: Vec<u32>,
}

fn load_fixtures() -> Result<Vec<Fixture>> {
    let path = fixture_path();
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let doc: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing {} as JSON", path.display()))?;
    let prompts = doc["prompts"]
        .as_array()
        .with_context(|| format!("{} has no `prompts` array", path.display()))?;
    let mut out = Vec::with_capacity(prompts.len());
    for (i, p) in prompts.iter().enumerate() {
        let id = p["id"]
            .as_str()
            .with_context(|| format!("prompt {i} has no string `id`"))?
            .to_string();
        let ids = p["ids"]
            .as_array()
            .with_context(|| format!("prompt {i} ({id}) has no `ids` array"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|n| n as u32)
                    .with_context(|| format!("prompt {i} ({id}) has a non-integer id"))
            })
            .collect::<Result<Vec<u32>>>()?;
        out.push(Fixture { id, ids });
    }
    Ok(out)
}

#[test]
fn committed_qwen3_prompt_fixtures_stay_valid() {
    let fixtures = load_fixtures().expect("tests/fixtures/qwen3-prompts.json must parse");
    assert!(
        fixtures.len() >= 20,
        "the Stage 1 prompt set is 20 prompts; found {}",
        fixtures.len()
    );
    let mut seen = BTreeSet::new();
    for (i, f) in fixtures.iter().enumerate() {
        assert!(
            !f.ids.is_empty(),
            "prompt {i} ({}) tokenized to nothing",
            f.id
        );
        assert!(seen.insert(f.id.clone()), "duplicate prompt id {}", f.id);
    }
}

// ---------------------------------------------------------------------------
// The oracle directory.
// ---------------------------------------------------------------------------

/// One prompt's row in the oracle's `manifest.json`.
struct OracleEntry {
    idx: usize,
    id: String,
    n_tokens: usize,
    logits: PathBuf,
    sidecar: PathBuf,
    f32_sha256: Option<String>,
    f32_bytes: Option<u64>,
}

fn load_manifest(dir: &Path) -> Result<Vec<OracleEntry>> {
    let path = dir.join("manifest.json");
    let bytes = std::fs::read(&path).with_context(|| {
        format!(
            "reading {} — XWEN_QWEN3_PARITY_DIR must name a directory written by \
             scripts/qwen3-ref-logits.ts",
            path.display()
        )
    })?;
    let doc: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing {} as JSON", path.display()))?;
    let prompts = doc["prompts"]
        .as_array()
        .with_context(|| format!("{} has no `prompts` array", path.display()))?;
    let mut out = Vec::with_capacity(prompts.len());
    for p in prompts {
        let idx = p["idx"].as_u64().context("a manifest row has no `idx`")? as usize;
        let id = p["id"]
            .as_str()
            .with_context(|| format!("manifest row {idx} has no `id`"))?
            .to_string();
        let n_tokens = p["n_tokens"]
            .as_u64()
            .with_context(|| format!("manifest row {idx} has no `n_tokens`"))?
            as usize;
        let logits = p["files"]["f32"]
            .as_str()
            .with_context(|| format!("manifest row {idx} names no `files.f32`"))?;
        let sidecar = p["files"]["json"]
            .as_str()
            .with_context(|| format!("manifest row {idx} names no `files.json`"))?;
        out.push(OracleEntry {
            idx,
            id,
            n_tokens,
            logits: dir.join(logits),
            sidecar: dir.join(sidecar),
            f32_sha256: p["f32_sha256"].as_str().map(str::to_string),
            f32_bytes: p["f32_bytes"].as_u64(),
        });
    }
    ensure!(!out.is_empty(), "{} lists no prompts", path.display());
    Ok(out)
}

/// The two numbers the sidecar exists to pin: how wide a row is and how many
/// there are.
fn read_sidecar(path: &Path) -> Result<(usize, usize)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let doc: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing {} as JSON", path.display()))?;
    let n_tokens = doc["n_tokens"]
        .as_u64()
        .with_context(|| format!("{} has no `n_tokens`", path.display()))?
        as usize;
    let n_vocab = doc["n_vocab"]
        .as_u64()
        .with_context(|| format!("{} has no `n_vocab`", path.display()))?
        as usize;
    Ok((n_tokens, n_vocab))
}

/// `shasum -a 256`, the same tool the oracle driver hashed with. Shelling out
/// keeps the dependency list where it is for a check that is off by default.
fn sha256_of(path: &Path) -> Result<String> {
    let out = std::process::Command::new("shasum")
        .arg("-a")
        .arg("256")
        .arg(path)
        .output()
        .with_context(|| format!("running shasum over {}", path.display()))?;
    ensure!(
        out.status.success(),
        "shasum over {} exited {}",
        path.display(),
        out.status
    );
    let text = String::from_utf8(out.stdout).context("shasum wrote non-UTF-8")?;
    let hex = text
        .split_whitespace()
        .next()
        .context("shasum wrote no digest")?;
    Ok(hex.to_string())
}

/// Streams `[n_tokens, n_vocab]` little-endian f32 rows, in order.
struct OracleRows {
    reader: BufReader<File>,
    bytes: Vec<u8>,
    row: Vec<f32>,
}

impl OracleRows {
    fn open(path: &Path, n_tokens: usize, n_vocab: usize) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let want = (n_tokens as u64)
            .checked_mul(n_vocab as u64)
            .and_then(|n| n.checked_mul(4))
            .context("oracle geometry overflows a u64")?;
        let have = file
            .metadata()
            .with_context(|| format!("stat {}", path.display()))?
            .len();
        ensure!(
            have == want,
            "{} is {have} bytes but [{n_tokens}, {n_vocab}] f32 is {want}",
            path.display()
        );
        Ok(Self {
            // 4 MiB: a row of a 151936 vocabulary is 594 KiB, so this reads
            // several rows per syscall on the long prompts.
            reader: BufReader::with_capacity(4 << 20, file),
            bytes: vec![0u8; n_vocab * 4],
            row: vec![0f32; n_vocab],
        })
    }

    fn next_row(&mut self) -> Result<&[f32]> {
        self.reader
            .read_exact(&mut self.bytes)
            .context("reading an oracle logits row")?;
        for (dst, src) in self.row.iter_mut().zip(self.bytes.chunks_exact(4)) {
            *dst = f32::from_le_bytes([src[0], src[1], src[2], src[3]]);
        }
        Ok(&self.row)
    }
}

// ---------------------------------------------------------------------------
// The gate.
// ---------------------------------------------------------------------------

fn env_string(key: &str) -> Option<String> {
    match std::env::var(key) {
        Ok(v) if !v.trim().is_empty() => Some(v),
        _ => None,
    }
}

fn env_usize(key: &str, default: usize) -> Result<usize> {
    match env_string(key) {
        None => Ok(default),
        Some(v) => v
            .trim()
            .parse::<usize>()
            .with_context(|| format!("{key} must be a positive integer, got {v:?}")),
    }
}

/// The subset of manifest indices to run, or `None` for all of them.
fn selected_indices() -> Result<Option<BTreeSet<usize>>> {
    let Some(raw) = env_string("XWEN_QWEN3_PARITY_ONLY") else {
        return Ok(None);
    };
    let mut out = BTreeSet::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        out.insert(
            part.parse::<usize>()
                .with_context(|| format!("XWEN_QWEN3_PARITY_ONLY: {part:?} is not an index"))?,
        );
    }
    ensure!(!out.is_empty(), "XWEN_QWEN3_PARITY_ONLY selected nothing");
    Ok(Some(out))
}

/// Where the checkpoint is, without ever starting a download.
fn resolve_checkpoint(entry: xwen::hub::Model) -> Result<PathBuf> {
    if let Some(dir) = env_string("XWEN_QWEN3_DIR") {
        let path = PathBuf::from(dir);
        ensure!(
            path.exists(),
            "XWEN_QWEN3_DIR points at {} which does not exist",
            path.display()
        );
        return Ok(path);
    }
    xwen::hub::cached_model(entry).with_context(|| {
        format!(
            "{entry} is not in the Hugging Face cache. Run `xwen fetch --model-size {entry}`, \
             or point XWEN_QWEN3_DIR at a safetensors directory."
        )
    })
}

/// One position that failed, kept for the panic message.
struct Failure {
    prompt: usize,
    prompt_id: String,
    position: usize,
    metrics: metrics::RowMetrics,
}

impl Failure {
    fn line(&self) -> String {
        format!(
            "  prompt {:>2} ({:<24}) pos {:>5}: ref top1 {:>6} (margin {:+.4e}), \
             xwen top1 {:>6} (margin {:+.4e}), max-abs {:.4e}, top5 {}/5{}",
            self.prompt,
            self.prompt_id,
            self.position,
            self.metrics.reference_top1,
            self.metrics.reference_margin,
            self.metrics.candidate_top1,
            self.metrics.candidate_margin,
            self.metrics.max_abs,
            self.metrics.top5_overlap,
            if !self.metrics.argmax_agrees() && self.metrics.near_tie() {
                "  [near tie]"
            } else {
                ""
            }
        )
    }
}

/// Everything the gate does EXCEPT run the model: manifest and sidecar parsing,
/// the fixture cross-check, the byte-length contract, the row stream and the
/// pool, over real 151936-wide rows.
///
/// The oracle compared against itself must be a perfect pass, so a failure here
/// is a harness bug and never a parity result — which is what makes it safe to
/// run on any machine, with no GPU and no checkpoint. Self-skips when
/// `XWEN_QWEN3_PARITY_DIR` is unset; reads the three shortest prompts only,
/// because the point is the plumbing and the longest file is 2.4 GB.
#[test]
fn the_oracle_reader_agrees_with_the_oracle_about_itself() -> Result<()> {
    let Some(dir) = env_string("XWEN_QWEN3_PARITY_DIR") else {
        eprintln!("skipped: XWEN_QWEN3_PARITY_DIR is unset");
        return Ok(());
    };
    let dir = PathBuf::from(dir);
    let mut manifest = load_manifest(&dir)?;
    let fixtures = load_fixtures()?;
    manifest.sort_by_key(|e| e.n_tokens);

    let mut pooled = metrics::Pool::default();
    for e in manifest.iter().take(3) {
        let fixture = fixtures
            .get(e.idx)
            .with_context(|| format!("no fixture at index {}", e.idx))?;
        ensure!(
            fixture.id == e.id,
            "index {}: {} vs {}",
            e.idx,
            e.id,
            fixture.id
        );
        ensure!(
            fixture.ids.len() == e.n_tokens,
            "index {}: token count",
            e.idx
        );

        let (n_tokens, n_vocab) = read_sidecar(&e.sidecar)?;
        ensure!(
            n_tokens == e.n_tokens,
            "index {}: sidecar token count",
            e.idx
        );
        let mut rows = OracleRows::open(&e.logits, n_tokens, n_vocab)?;
        for _ in 0..n_tokens {
            let row = rows.next_row()?.to_vec();
            pooled.observe(&metrics::row_metrics(&row, &row));
        }
    }

    ensure!(pooled.positions > 0, "the manifest yielded no positions");
    assert!(
        pooled.passes(),
        "the oracle disagrees with itself, which can only be a reader bug: {pooled:?}"
    );
    assert_eq!(pooled.max_abs, 0.0);
    assert_eq!(pooled.argmax_agree, pooled.positions);
    assert_eq!(pooled.top5_agreement(), 1.0);
    eprintln!(
        "oracle reader: {} positions over 3 prompts, self-comparison perfect",
        pooled.positions
    );
    Ok(())
}

#[test]
#[ignore = "needs a Metal device, the Qwen3-4B checkpoint and XWEN_QWEN3_PARITY_DIR"]
fn qwen3_4b_logits_match_the_llamacpp_oracle() -> Result<()> {
    use candle_core::Tensor;
    use xwen::checkpoint::CheckpointSource;
    use xwen::ops::ExpertRunner;
    use xwen::{XwenModel, gguf};

    let dir = PathBuf::from(env_string("XWEN_QWEN3_PARITY_DIR").context(
        "XWEN_QWEN3_PARITY_DIR is unset: it must name the oracle directory \
         (manifest.json + prompt-N.f32), written by scripts/qwen3-ref-logits.ts",
    )?);
    let manifest = load_manifest(&dir)?;
    let fixtures = load_fixtures()?;
    let only = selected_indices()?;
    let list_cap = env_usize("XWEN_QWEN3_PARITY_LIST", 50)?;
    let verify_sha = env_string("XWEN_QWEN3_PARITY_VERIFY_SHA").as_deref() == Some("1");

    let entry: xwen::hub::Model = match env_string("XWEN_QWEN3_ENTRY") {
        Some(alias) => alias.parse().map_err(anyhow::Error::msg)?,
        None => xwen::hub::Model::Qwen34B,
    };

    let run: Vec<&OracleEntry> = manifest
        .iter()
        .filter(|e| only.as_ref().is_none_or(|set| set.contains(&e.idx)))
        .collect();
    ensure!(
        !run.is_empty(),
        "XWEN_QWEN3_PARITY_ONLY selected no manifest row"
    );

    // Cross-check the two files describe the same prompts before spending eight
    // gigabytes of load on them.
    for e in &run {
        let f = fixtures.get(e.idx).with_context(|| {
            format!(
                "the oracle has a row at index {} but the fixture file has only {} prompts",
                e.idx,
                fixtures.len()
            )
        })?;
        ensure!(
            f.id == e.id,
            "index {}: the oracle calls this prompt {:?} and the fixture calls it {:?}",
            e.idx,
            e.id,
            f.id
        );
        ensure!(
            f.ids.len() == e.n_tokens,
            "prompt {} ({}): the fixture is {} ids and the oracle scored {} positions",
            e.idx,
            e.id,
            f.ids.len(),
            e.n_tokens
        );
    }

    let longest = run.iter().map(|e| e.n_tokens).max().unwrap_or(0);
    let max_ctx = env_usize("XWEN_QWEN3_PARITY_MAX_CTX", longest)?;
    ensure!(
        max_ctx >= longest,
        "max_ctx {max_ctx} is shorter than the longest selected prompt ({longest} tokens); \
         this gate never truncates a prompt"
    );

    let checkpoint = resolve_checkpoint(entry)?;
    let device = gguf::metal_device()?;
    let source = CheckpointSource::open(&checkpoint, &device, Some(entry))?;
    let cfg = source.config()?;
    let vocab = cfg.vocab;
    // A dense checkpoint runs no expert kernels at all, so the runner is a
    // formality here; Fused is what every other surface defaults to.
    let mut model = XwenModel::load(source, ExpertRunner::Fused, max_ctx)?;
    let chunk = env_usize("XWEN_QWEN3_PARITY_CHUNK", model.prefill_chunk())?;
    ensure!(chunk > 0, "XWEN_QWEN3_PARITY_CHUNK must be at least 1");

    let mut report = String::new();
    report.push_str(&format!(
        "QWEN3-4B STAGE 1 LOGITS PARITY\n  oracle:     {}\n  checkpoint: {} ({entry})\n  \
         vocab: {vocab}   max_ctx: {max_ctx}   prefill chunk: {chunk}\n  bars: max-abs <= {:.1e}, \
         argmax 100%, pooled top-5 >= {:.4}   near-tie band: {:.1e}\n\n",
        dir.display(),
        checkpoint.display(),
        metrics::MAX_ABS_BAR,
        metrics::TOP5_BAR,
        metrics::NEAR_TIE_MARGIN,
    ));
    report.push_str(&format!(
        "  {:>3}  {:<26} {:>7} {:>11} {:>13} {:>9} {:>9}\n",
        "idx", "prompt", "tokens", "max-abs", "argmax", "top-5", "near-tie"
    ));

    let mut pooled = metrics::Pool::default();
    let mut failures: Vec<Failure> = Vec::new();

    for e in &run {
        let fixture = &fixtures[e.idx];
        let (side_tokens, side_vocab) = read_sidecar(&e.sidecar)?;
        ensure!(
            side_vocab == vocab,
            "prompt {} ({}): the oracle rows are {side_vocab} wide but the model's vocabulary is \
             {vocab}",
            e.idx,
            e.id
        );
        ensure!(
            side_tokens == fixture.ids.len(),
            "prompt {} ({}): the sidecar says {side_tokens} positions and the fixture is {} ids",
            e.idx,
            e.id,
            fixture.ids.len()
        );
        if let Some(bytes) = e.f32_bytes {
            let have = std::fs::metadata(&e.logits)
                .with_context(|| format!("stat {}", e.logits.display()))?
                .len();
            ensure!(
                have == bytes,
                "{} is {have} bytes; the manifest recorded {bytes}",
                e.logits.display()
            );
        }
        if verify_sha {
            let want = e.f32_sha256.as_deref().with_context(|| {
                format!(
                    "XWEN_QWEN3_PARITY_VERIFY_SHA=1 but the manifest records no digest for {}",
                    e.logits.display()
                )
            })?;
            let have = sha256_of(&e.logits)?;
            ensure!(
                have == want,
                "{} hashes to {have}; the manifest recorded {want}",
                e.logits.display()
            );
        }

        let mut rows = OracleRows::open(&e.logits, side_tokens, side_vocab)?;
        let mut prompt_pool = metrics::Pool::default();

        model.reset_cache()?;
        let mut pos = 0usize;
        while pos < fixture.ids.len() {
            let take = chunk.min(fixture.ids.len() - pos);
            let tokens = Tensor::new(&fixture.ids[pos..pos + take], &device)?;
            let logits = model.forward_all_logits(&tokens, pos)?;
            let dims = logits.dims().to_vec();
            ensure!(
                dims == vec![take, vocab],
                "forward_all_logits returned {dims:?} for a {take}-token chunk at position {pos}; \
                 expected [{take}, {vocab}]"
            );
            let ours = logits.to_vec2::<f32>()?;
            for (r, candidate) in ours.iter().enumerate() {
                let reference = rows.next_row()?;
                let m = metrics::row_metrics(candidate, reference);
                prompt_pool.observe(&m);
                let bad = m.max_abs > metrics::MAX_ABS_BAR
                    || !m.argmax_agrees()
                    || m.nonfinite > 0
                    || m.top5_overlap < metrics::TOP_K;
                if bad {
                    failures.push(Failure {
                        prompt: e.idx,
                        prompt_id: e.id.clone(),
                        position: pos + r,
                        metrics: m,
                    });
                }
            }
            pos += take;
        }

        ensure!(
            prompt_pool.positions == fixture.ids.len(),
            "prompt {} ({}): scored {} positions of {}",
            e.idx,
            e.id,
            prompt_pool.positions,
            fixture.ids.len()
        );
        report.push_str(&format!(
            "  {:>3}  {:<26} {:>7} {:>11.3e} {:>7}/{:<5} {:>8.4}% {:>9}\n",
            e.idx,
            e.id,
            prompt_pool.positions,
            prompt_pool.max_abs,
            prompt_pool.argmax_agree,
            prompt_pool.positions,
            100.0 * prompt_pool.top5_agreement(),
            prompt_pool.near_tie_flips,
        ));
        pooled.merge(&prompt_pool);
    }

    report.push_str(&format!(
        "\n  POOLED over {} prompts, {} positions\n    max-abs error:     {:.4e}  [{}]\n    \
         argmax agreement:  {}/{} ({:.4}%)  [{}]\n    top-5 agreement:   {:.6}%  [{}]\n    \
         argmax flips inside the near-tie band: {} of {}\n    non-finite logit entries \
         (either side): {}\n",
        run.len(),
        pooled.positions,
        pooled.max_abs,
        if pooled.max_abs <= metrics::MAX_ABS_BAR {
            "PASS"
        } else {
            "FAIL"
        },
        pooled.argmax_agree,
        pooled.positions,
        100.0 * pooled.argmax_agreement(),
        if pooled.argmax_agree == pooled.positions {
            "PASS"
        } else {
            "FAIL"
        },
        100.0 * pooled.top5_agreement(),
        if pooled.top5_agreement() >= metrics::TOP5_BAR {
            "PASS"
        } else {
            "FAIL"
        },
        pooled.near_tie_flips,
        pooled.positions - pooled.argmax_agree,
        pooled.nonfinite,
    ));
    report.push_str(&format!(
        "\n  VERDICT: {}\n",
        if pooled.passes() { "PASS" } else { "FAIL" }
    ));

    // Printed on a pass too — the numbers are the point of the run, not just
    // the verdict. Needs `-- --nocapture`.
    println!("{report}");

    if pooled.passes() {
        return Ok(());
    }

    let mut message = String::from("Qwen3-4B Stage 1 logits parity FAILED\n\n");
    message.push_str(&report);
    message.push_str(&format!("\n  {} failing positions", failures.len()));
    if failures.len() > list_cap {
        message.push_str(&format!(
            " (listing the first {list_cap}; raise XWEN_QWEN3_PARITY_LIST for more)"
        ));
    }
    message.push('\n');
    for f in failures.iter().take(list_cap) {
        message.push_str(&f.line());
        message.push('\n');
    }
    bail!(message);
}
