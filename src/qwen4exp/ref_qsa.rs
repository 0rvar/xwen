//! Frozen CPU f32 reference for the QSA (Qwen sparse attention) indexer.
//!
//! The indexer is the part of a QSA layer that decides WHICH cached tokens the
//! ordinary attention is allowed to see. It keeps its own tiny MQA projection
//! (`n_q_heads` query heads, exactly one key head) beside the real attention
//! weights, scores four-token blocks of the visible prefix, and turns the
//! top-scoring blocks — plus the incomplete tail — into the token set the
//! causal mask is intersected with.
//!
//! Pipeline per query, all of it here:
//!
//! 1. Keys are cached RAW: `k_proj(x)` only, un-normed and un-roped, because a
//!    block's key is normed and roped AFTER pooling, at a position that depends
//!    on the block rather than on the token.
//! 2. Visible raw keys are cut into consecutive runs of `ratio`; each run is
//!    mean-pooled in f32, RMS-normed, and roped at the position of the run's
//!    FIRST token.
//! 3. The query is projected, per-head RMS-normed, and roped at its own
//!    position. Its score against a block is `Σ_heads relu(q_h · k_b) / √d`.
//! 4. The top `min(budget / ratio, n_complete_blocks)` blocks by score are taken
//!    WHOLE, and the raw tail (`visible.len() % ratio` tokens that never formed
//!    a complete block) is always appended.
//!
//! Step 4 is the one place a plausible-looking implementation goes wrong. The
//! selection is whole blocks plus the tail, so the count is
//! `keep * ratio + tail_len`, which is USUALLY less than the `budget + ratio - 1`
//! that HF's fixed-width buffer can hold: a short tail admits nothing from the
//! next-ranked block. llama.cpp's merged QSA always fills `top_k + ratio - 1`
//! token slots and therefore selects a different set whenever
//! `visible % ratio != ratio - 1` above budget. HF is ground truth here; the
//! `select_matches_hf_not_the_fill_to_width_rule` test pins the difference.
//!
//! The query's own token gets no special treatment. When the tail is empty and
//! the query's block loses the top-k, the query cannot see itself — that is HF
//! behavior, not a bug (fixture `case_above_budget` query 15).
//!
//! # Pooled keys stay f32 (D13)
//!
//! The pooled block key is NOT rounded back to the key cache's dtype before the
//! norm and the rope. HF does round it (`key_groups.float().mean(dim=1)
//! .to(raw_keys.dtype)`, modular:437), which at the real checkpoint's BF16
//! indexer cache costs every pooled key its low mantissa bits BEFORE scoring;
//! llama.cpp pools through `ggml_get_rows` into f32 and never rounds back.
//! xwen follows llama.cpp, because llama.cpp is this port's parity oracle.
//!
//! The consequence, stated so it is never rediscovered as a bug: exact
//! index-set parity against an HF tap at real geometry is NOT a goal. The
//! bf16 round-back perturbs scores by ~1.2e-2 where the top-k cut margin at
//! 1k-4k blocks is ~2e-3, so a handful of the 512 selected blocks per query
//! differ at every context length above budget. Which dtype the cache itself
//! holds is a separate, still-open P2/P3 choice; this decision is only about
//! not re-rounding after the f32 pool.
//!
//! # Block boundaries anchor to the sequence, not to a chunk
//!
//! Raw keys are cached, and the selection for a query at position `q` runs over
//! ALL visible raw keys from the start of the sequence. Blocks are therefore
//! cut from the sequence-start-anchored run of visible keys, never from the
//! prefill chunk that happened to produce them: a chunked prefill and a
//! single-shot one must select identical token sets, which
//! `chunked_raw_keys_select_like_a_single_shot_run` pins.
//!
//! Everything is f32 row-major `Vec`s with no candle and no device: this is a
//! correctness oracle the Metal path is graded against, never optimized.

use std::cmp::Ordering;

/// Indexer weights and geometry. Norm weights are multiply-ready (the GGUF
/// convention: the converter has already baked HF's `1 +` into them).
pub struct QsaIndexerRef {
    /// Model hidden size — the width both projections read.
    pub hidden: usize,
    /// Indexer query heads. The key side is MQA: always exactly one head.
    pub n_q_heads: usize,
    /// Indexer key/value heads. MQA is a hard requirement of this reference:
    /// it must be 1, and the functions that touch `k_w` assert it. A file
    /// declaring more would otherwise use head 0 and silently drop the rest.
    pub n_kv_heads: usize,
    /// Indexer head dim (the norm and rope width; the score scale is `1/√head_dim`).
    pub head_dim: usize,
    /// Rotary width: dims `0..n_rot` rotate, dims `n_rot..head_dim` pass through.
    pub n_rot: usize,
    pub rope_theta: f32,
    /// Token budget. Whole blocks are kept, so `budget / ratio` blocks.
    pub budget: usize,
    /// Tokens per block.
    pub ratio: usize,
    /// `[n_q_heads * head_dim, hidden]` row-major.
    pub q_w: Vec<f32>,
    /// `[head_dim, hidden]` row-major (one key head).
    pub k_w: Vec<f32>,
    /// `[head_dim]`, multiply-ready.
    pub q_norm_w: Vec<f32>,
    /// `[head_dim]`, multiply-ready.
    pub k_norm_w: Vec<f32>,
    pub eps: f32,
}

impl QsaIndexerRef {
    /// The indexer key side is single-head by construction — `k_w` is
    /// `[head_dim, hidden]` and every pooling and scoring path here assumes it.
    fn assert_mqa(&self) {
        assert_eq!(
            self.n_kv_heads, 1,
            "the indexer reference is MQA-only: n_kv_heads must be 1"
        );
        assert_eq!(
            self.k_w.len(),
            self.head_dim * self.hidden,
            "k_w is not [head_dim][hidden]"
        );
    }

    /// Cached indexer keys for `x` (`[n_tok, hidden]`): the k projection and
    /// nothing else, `[n_tok, head_dim]`. Norm and rope are deliberately absent
    /// — they are applied to the pooled block key, at the block's position.
    pub fn raw_keys(&self, x: &[f32]) -> Vec<f32> {
        self.assert_mqa();
        let n_tok = x.len() / self.hidden;
        let mut out = vec![0f32; n_tok * self.head_dim];
        for t in 0..n_tok {
            let row = &x[t * self.hidden..(t + 1) * self.hidden];
            matvec_into(
                &self.k_w,
                self.head_dim,
                self.hidden,
                row,
                &mut out[t * self.head_dim..(t + 1) * self.head_dim],
            );
        }
        out
    }

    /// Indexer queries for `x` (`[n_tok, hidden]`) at `positions`:
    /// q projection, per-head RMS norm, then partial NEoX rope at each token's
    /// own position. Returns `[n_tok, n_q_heads, head_dim]`.
    pub fn queries(&self, x: &[f32], positions: &[usize]) -> Vec<f32> {
        let n_tok = x.len() / self.hidden;
        assert_eq!(positions.len(), n_tok, "one position per token");
        let per_tok = self.n_q_heads * self.head_dim;
        let mut out = vec![0f32; n_tok * per_tok];
        for t in 0..n_tok {
            let row = &x[t * self.hidden..(t + 1) * self.hidden];
            let dst = &mut out[t * per_tok..(t + 1) * per_tok];
            matvec_into(&self.q_w, per_tok, self.hidden, row, dst);
            for h in 0..self.n_q_heads {
                let head = &mut dst[h * self.head_dim..(h + 1) * self.head_dim];
                rms_norm_in_place(head, &self.q_norm_w, self.eps);
                rope_neox_in_place(head, positions[t], self.n_rot, self.rope_theta);
            }
        }
        out
    }

    /// Block keys for the visible raw keys (`[n_visible, head_dim]`), one per
    /// COMPLETE run of `ratio` consecutive visible keys: f32 mean over the run,
    /// RMS norm, rope at `first_positions[b]` (the position of the run's first
    /// token, which is not `b * ratio` once the visible set has holes).
    /// The trailing `n_visible % ratio` keys form no block; they are the tail
    /// `select` appends unconditionally. Returns `[n_blocks, head_dim]`.
    pub fn block_keys(&self, raw_keys_visible: &[f32], first_positions: &[usize]) -> Vec<f32> {
        self.assert_mqa();
        let d = self.head_dim;
        let n_visible = raw_keys_visible.len() / d;
        let n_blocks = n_visible / self.ratio;
        assert_eq!(
            first_positions.len(),
            n_blocks,
            "one first-token position per complete block"
        );
        let mut out = vec![0f32; n_blocks * d];
        for (b, &pos) in first_positions.iter().enumerate() {
            let dst = &mut out[b * d..(b + 1) * d];
            for t in b * self.ratio..(b + 1) * self.ratio {
                let src = &raw_keys_visible[t * d..(t + 1) * d];
                for (o, &v) in dst.iter_mut().zip(src) {
                    *o += v;
                }
            }
            let inv = 1.0 / self.ratio as f32;
            for o in dst.iter_mut() {
                *o *= inv;
            }
            rms_norm_in_place(dst, &self.k_norm_w, self.eps);
            rope_neox_in_place(dst, pos, self.n_rot, self.rope_theta);
        }
        out
    }

    /// Block scores for ONE query's heads (`[n_q_heads, head_dim]`) against
    /// `block_keys` (`[n_blocks, head_dim]`): `Σ_heads relu(q_h · k_b) / √d`.
    /// The relu is per head, before the sum, so a block every head dislikes
    /// scores exactly 0 rather than going negative.
    pub fn scores(&self, q: &[f32], block_keys: &[f32]) -> Vec<f32> {
        let d = self.head_dim;
        let scale = 1.0 / (d as f32).sqrt();
        block_keys
            .chunks_exact(d)
            .map(|kb| {
                let mut sum = 0f32;
                for h in 0..self.n_q_heads {
                    let qh = &q[h * d..(h + 1) * d];
                    let dot: f32 = qh.iter().zip(kb).map(|(a, b)| a * b).sum();
                    sum += dot.max(0.0);
                }
                sum * scale
            })
            .collect()
    }

    /// The full indexer over a causal prefix: for every token `q` in `x`
    /// (`[n_tok, hidden]`), the token indices query `q` may attend to, given
    /// that everything up to and including `q` is visible. Entry `q` is sorted
    /// ascending. `positions[t]` is token `t`'s rope position.
    pub fn select_all(&self, x: &[f32], positions: &[usize]) -> Vec<Vec<usize>> {
        self.assert_mqa();
        let n_tok = x.len() / self.hidden;
        let raw = self.raw_keys(x);
        let q_all = self.queries(x, positions);
        let per_tok = self.n_q_heads * self.head_dim;

        (0..n_tok)
            .map(|q| {
                let visible: Vec<usize> = (0..=q).collect();
                let n_blocks = visible.len() / self.ratio;
                let firsts: Vec<usize> = (0..n_blocks)
                    .map(|b| positions[visible[b * self.ratio]])
                    .collect();
                let keys = self.block_keys(&raw[..visible.len() * self.head_dim], &firsts);
                let scores = self.scores(&q_all[q * per_tok..(q + 1) * per_tok], &keys);
                select(&scores, &visible, self.budget, self.ratio)
            })
            .collect()
    }
}

/// Turn block scores into the selected token indices, sorted ascending.
///
/// `visible` lists the token indices in block order (`scores[b]` scores
/// `visible[b*ratio .. (b+1)*ratio]`), so the caller decides what "visible"
/// means; `scores.len()` must be `visible.len() / ratio`. The top
/// `min(budget / ratio, n_blocks)` blocks are expanded WHOLE, and the tail
/// `visible[n_blocks * ratio ..]` (0 to `ratio - 1` tokens) is always included.
/// When every block fits, the result is exactly `visible` — sparse selection
/// degenerates to dense attention below budget.
///
/// Ties: blocks with equal scores are kept in ascending block order (a stable
/// sort of ascending indices by descending score), so the lower block index
/// wins the last slot. This is xwen's own deterministic choice, not a rule read
/// off a reference: neither `torch.topk` nor ggml's `ggml_top_k` specifies an
/// order for equal values. Downstream kernels are therefore graded on set
/// equality ONLY where the top-k margin is strictly positive — a kernel that
/// breaks a tie the other way is not wrong. The fixture's smallest top-k margin
/// is 0.47, far outside f32 noise, so the rule is inert there; exact-zero ties
/// are the normal case at long context, where every head dislikes a block and
/// the relu floors its score at 0.
pub fn select(scores: &[f32], visible: &[usize], budget: usize, ratio: usize) -> Vec<usize> {
    let n_blocks = visible.len() / ratio;
    assert_eq!(scores.len(), n_blocks, "one score per complete block");
    let keep = (budget / ratio).min(n_blocks);

    let mut order: Vec<usize> = (0..n_blocks).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal));

    let tail = &visible[n_blocks * ratio..];
    let mut out = Vec::with_capacity(keep * ratio + tail.len());
    for &b in &order[..keep] {
        out.extend_from_slice(&visible[b * ratio..(b + 1) * ratio]);
    }
    out.extend_from_slice(tail);
    out.sort_unstable();
    out
}

/// `out = w · x` for row-major `w` `[rows, cols]` and `x` `[cols]`.
///
/// The shape assert is load-bearing: an over-tall `w` would otherwise be
/// silently truncated to its first `rows` rows and produce a plausible wrong
/// answer instead of an error.
fn matvec_into(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    assert_eq!(w.len(), rows * cols, "matrix is not [rows][cols]");
    assert_eq!(
        x.len(),
        cols,
        "vector width does not match matrix input width"
    );
    assert_eq!(out.len(), rows, "output width does not match matrix rows");
    for (r, o) in out.iter_mut().enumerate() {
        let row = &w[r * cols..(r + 1) * cols];
        *o = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
}

/// `x = x * rsqrt(mean(x²) + eps) * w`, with multiply-ready `w`.
///
/// Delegates to the hyper-connection reference's grouped norm with one group
/// spanning the whole row: one implementation of the math for the whole port,
/// with its `x.len() == w.len()` assert and its f64 accumulation.
fn rms_norm_in_place(x: &mut [f32], w: &[f32], eps: f32) {
    let normed = super::ref_hc::grouped_rms_norm(x, w, x.len(), eps);
    x.copy_from_slice(&normed);
}

/// Partial NEoX rope: dim `i` pairs with `i + n_rot/2` over the first `n_rot`
/// dims, dims `n_rot..` pass through untouched. Angles are accumulated in f64
/// and the cos/sin rounded once to f32, matching `crate::rope`'s tables.
fn rope_neox_in_place(x: &mut [f32], pos: usize, n_rot: usize, theta: f32) {
    let half = n_rot / 2;
    for i in 0..half {
        let freq = (theta as f64).powf(-(2.0 * i as f64) / n_rot as f64);
        let angle = pos as f64 * freq;
        let (c, s) = (angle.cos() as f32, angle.sin() as f32);
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen4exp/qsa_indexer.json"
    );

    /// Fixture floats are shortest-f64 reprs of exact f32 values: parse as f64,
    /// cast back, get the original bits.
    fn f32s(v: &Value) -> Vec<f32> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|e| e.as_f64().expect("number") as f32)
            .collect()
    }

    fn f32s_2d(v: &Value) -> Vec<f32> {
        v.as_array()
            .expect("array of rows")
            .iter()
            .flat_map(f32s)
            .collect()
    }

    fn usizes_2d(v: &Value) -> Vec<Vec<usize>> {
        v.as_array()
            .expect("array of rows")
            .iter()
            .map(|row| {
                row.as_array()
                    .expect("row")
                    .iter()
                    .map(|e| e.as_u64().expect("index") as usize)
                    .collect()
            })
            .collect()
    }

    fn fixture() -> Value {
        serde_json::from_str(&std::fs::read_to_string(FIXTURE).expect("fixture readable"))
            .expect("fixture parses")
    }

    fn indexer(f: &Value) -> QsaIndexerRef {
        let cfg = &f["config"];
        let n_q_heads = cfg["indexer_n_heads"].as_u64().unwrap() as usize;
        let head_dim = cfg["indexer_head_dim"].as_u64().unwrap() as usize;
        let hidden = cfg["hidden_size"].as_u64().unwrap() as usize;
        // The fixture ships one fused `[(n_heads + 1) * head_dim, hidden]`
        // projection: the q heads first, the single MQA k head last.
        let proj = f32s_2d(&f["weights"]["index_qk_proj"]);
        let split = n_q_heads * head_dim * hidden;
        QsaIndexerRef {
            hidden,
            n_q_heads,
            n_kv_heads: cfg["indexer_kv_heads"].as_u64().unwrap() as usize,
            head_dim,
            n_rot: cfg["rotary_dim"].as_u64().unwrap() as usize,
            rope_theta: cfg["rope_theta"].as_f64().unwrap() as f32,
            budget: cfg["indexer_budget"].as_u64().unwrap() as usize,
            ratio: cfg["indexer_compress_ratio"].as_u64().unwrap() as usize,
            q_w: proj[..split].to_vec(),
            k_w: proj[split..split + head_dim * hidden].to_vec(),
            // The GGUF path multiplies by the pre-baked `1 + w` weights.
            q_norm_w: f32s(&f["weights"]["q_layernorm_weight_mult"]),
            k_norm_w: f32s(&f["weights"]["k_layernorm_weight_mult"]),
            eps: cfg["rms_norm_eps"].as_f64().unwrap() as f32,
        }
    }

    /// The fixture's own tolerance: `max(1e-6, 1e-5 relative)`.
    fn close(got: f32, want: f32) -> bool {
        (got - want).abs() <= (1e-6f32).max(1e-5 * want.abs())
    }

    #[test]
    fn raw_keys_match_fixture() {
        let f = fixture();
        let ix = indexer(&f);
        for case in ["case_below_budget", "case_above_budget"] {
            let x = f32s_2d(&f[case]["hidden_states"]);
            let want = f32s_2d(&f[case]["raw_keys"]);
            let got = ix.raw_keys(&x);
            assert_eq!(got.len(), want.len(), "{case}: key count");
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!(close(g, w), "{case}: raw key {i}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn block_scores_match_fixture() {
        let f = fixture();
        let ix = indexer(&f);
        let case = &f["case_above_budget"];
        let x = f32s_2d(&case["hidden_states"]);
        let n_tok = x.len() / ix.hidden;
        let positions: Vec<usize> = (0..n_tok).collect();
        let raw = ix.raw_keys(&x);
        let q_all = ix.queries(&x, &positions);
        let per_tok = ix.n_q_heads * ix.head_dim;

        // Smallest gap between the last selected and best rejected block, over
        // all queries: the fixture records it to show the seeded data sits far
        // from a tie, so the tie rule cannot change the selection here.
        let mut min_margin = f32::INFINITY;

        for q in 0..n_tok {
            let visible: Vec<usize> = (0..=q).collect();
            let n_blocks = visible.len() / ix.ratio;
            // The same derivation `select_all` uses: a block's rope position is
            // the POSITION of its first visible token, not the block index
            // times the ratio. They coincide only for a hole-free 0-based
            // prefix, which is exactly what this fixture is.
            let firsts: Vec<usize> = (0..n_blocks)
                .map(|b| positions[visible[b * ix.ratio]])
                .collect();
            let keys = ix.block_keys(&raw[..visible.len() * ix.head_dim], &firsts);
            let got = ix.scores(&q_all[q * per_tok..(q + 1) * per_tok], &keys);
            let want = f32s(&case["block_scores_per_query"][q]);
            assert_eq!(got.len(), want.len(), "query {q}: block count");
            for (b, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!(close(g, w), "query {q} block {b}: {g} vs {w}");
            }

            let keep = (ix.budget / ix.ratio).min(n_blocks);
            if keep < n_blocks {
                let mut sorted = got.clone();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
                min_margin = min_margin.min(sorted[keep - 1] - sorted[keep]);
            }
        }

        let want_margin = case["min_topk_margin"].as_f64().unwrap() as f32;
        assert!(
            (min_margin - want_margin).abs() < 1e-5,
            "min top-k margin {min_margin} vs fixture {want_margin}"
        );
    }

    #[test]
    fn below_budget_selection_is_dense_causal() {
        let f = fixture();
        let ix = indexer(&f);
        let case = &f["case_below_budget"];
        assert!(case["selected_equals_causal"].as_bool().unwrap());
        let x = f32s_2d(&case["hidden_states"]);
        let n_tok = x.len() / ix.hidden;
        let positions: Vec<usize> = (0..n_tok).collect();
        let got = ix.select_all(&x, &positions);
        let want = usizes_2d(&case["selected_token_indices"]);

        for (q, sel) in got.iter().enumerate() {
            let dense: Vec<usize> = (0..=q).collect();
            assert_eq!(*sel, dense, "query {q}: below budget must be dense causal");
            assert_eq!(*sel, want[q], "query {q}: fixture selection");
        }
    }

    #[test]
    fn above_budget_selection_matches_fixture() {
        let f = fixture();
        let ix = indexer(&f);
        let case = &f["case_above_budget"];
        let x = f32s_2d(&case["hidden_states"]);
        let n_tok = x.len() / ix.hidden;
        let positions: Vec<usize> = (0..n_tok).collect();
        let got = ix.select_all(&x, &positions);
        let want = usizes_2d(&case["selected_token_indices"]);
        let counts = f32s(&case["selected_counts"]);

        for (q, sel) in got.iter().enumerate() {
            assert!(sel.windows(2).all(|w| w[0] < w[1]), "query {q}: sorted");
            let mut want_sorted = want[q].clone();
            want_sorted.sort_unstable();
            assert_eq!(*sel, want_sorted, "query {q}: selected set");
            assert_eq!(sel.len(), counts[q] as usize, "query {q}: count");
        }
    }

    /// 13 visible tokens: 3 complete blocks (0-3, 4-7, 8-11) plus a 1-token
    /// tail. Two blocks fit the budget of 8, so 2*4 + 1 = 9 tokens — not the 11
    /// that a fill-to-`budget + ratio - 1` rule would produce.
    #[test]
    fn query_12_selects_two_blocks_plus_a_one_token_tail() {
        let sel = above_budget_selection(12);
        assert_eq!(sel.len(), 9);
        assert_eq!(sel, vec![4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }

    /// 15 visible tokens: the tail is the full `ratio - 1`, the one case where
    /// whole-blocks-plus-tail and fill-to-width agree: 2*4 + 3 = 11.
    #[test]
    fn query_14_selects_two_blocks_plus_a_full_tail() {
        let sel = above_budget_selection(14);
        assert_eq!(sel.len(), 11);
        assert_eq!(sel, vec![0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14]);
    }

    /// 16 visible tokens: 4 complete blocks and NO tail, so exactly the budget
    /// of 8 tokens. Block 3 (tokens 12-15) loses the top-k, so the query cannot
    /// see its own token — HF masks it, and nothing re-adds it.
    #[test]
    fn query_15_selects_exactly_budget_and_masks_its_own_token() {
        let sel = above_budget_selection(15);
        assert_eq!(sel.len(), 8);
        assert_eq!(sel, vec![4, 5, 6, 7, 8, 9, 10, 11]);
        assert!(!sel.contains(&15));
    }

    fn above_budget_selection(query: usize) -> Vec<usize> {
        let f = fixture();
        let ix = indexer(&f);
        let x = f32s_2d(&f["case_above_budget"]["hidden_states"]);
        let n_tok = x.len() / ix.hidden;
        let positions: Vec<usize> = (0..n_tok).collect();
        ix.select_all(&x, &positions).swap_remove(query)
    }

    /// Divergence guard, independent of the fixture: HF takes whole blocks plus
    /// the raw tail, llama.cpp's merged QSA fills `top_k + ratio - 1` token
    /// slots. With 13 visible tokens, budget 8, ratio 4 the two disagree — HF
    /// stops at 9 tokens where the fill rule would reach 11 by admitting the
    /// third-ranked block's tokens.
    #[test]
    fn select_matches_hf_not_the_fill_to_width_rule() {
        let visible: Vec<usize> = (0..13).collect();
        let scores = [0.1f32, 0.9, 0.5]; // ranked: block 1, block 2, block 0
        let sel = select(&scores, &visible, 8, 4);

        assert_eq!(sel, vec![4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(sel.len(), 2 * 4 + 1);
        // The fill-to-width rule would have taken block 0's tokens as well.
        assert_ne!(sel.len(), 8 + 4 - 1);
        assert!(!sel.contains(&0));
    }

    /// Equal scores keep ascending block order, so the lower block index takes
    /// the last slot.
    #[test]
    fn select_breaks_ties_toward_the_lower_block_index() {
        let visible: Vec<usize> = (0..12).collect();
        let sel = select(&[0.5f32, 0.2, 0.5], &visible, 8, 4);
        assert_eq!(sel, vec![0, 1, 2, 3, 8, 9, 10, 11]);

        let sel = select(&[0.5f32, 0.5, 0.9], &visible, 8, 4);
        assert_eq!(sel, vec![0, 1, 2, 3, 8, 9, 10, 11]);
    }

    /// `select` reads token identity from `visible`, so a gap in the cache
    /// shifts which tokens a block covers without changing the block math.
    #[test]
    fn select_expands_blocks_through_the_visible_index_list() {
        let visible = vec![0usize, 3, 4, 7, 9, 11, 12, 15, 20];
        let sel = select(&[0.2f32, 0.8], &visible, 4, 4);
        // One block fits: the higher-scoring second one (9, 11, 12, 15), plus
        // the single-token tail (20).
        assert_eq!(sel, vec![9, 11, 12, 15, 20]);
    }

    // ---- independent scalar path, written differently on purpose ----
    //
    // These re-derive the norm and the rope from the formulas rather than
    // calling the reference's helpers, so a test built on them fails when the
    // reference changes shape — which is the point of an oracle's test.

    /// `x * rsqrt(mean(x²) + eps) * w`, straight f32.
    fn scalar_rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let mean: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv = 1.0 / (mean + eps).sqrt();
        x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
    }

    /// Partial NEoX rope written as a division by a positive power (the
    /// reference raises the base to a negative one) and applied out of place.
    fn scalar_rope(x: &[f32], pos: usize, n_rot: usize, theta: f32) -> Vec<f32> {
        let half = n_rot / 2;
        let mut out = x.to_vec();
        for i in 0..half {
            let angle = pos as f64 / (theta as f64).powf(2.0 * i as f64 / n_rot as f64);
            let (c, sn) = (angle.cos() as f32, angle.sin() as f32);
            out[i] = x[i] * c - x[i + half] * sn;
            out[i + half] = x[i] * sn + x[i + half] * c;
        }
        out
    }

    /// One query's block scores, built end to end from the formulas.
    fn scalar_scores(
        ix: &QsaIndexerRef,
        x: &[f32],
        positions: &[usize],
        q: usize,
        n_blocks: usize,
    ) -> Vec<f32> {
        let d = ix.head_dim;
        let hidden = ix.hidden;
        let row = |t: usize| &x[t * hidden..(t + 1) * hidden];
        let dot = |w: &[f32], r: usize, v: &[f32]| -> f32 {
            w[r * hidden..(r + 1) * hidden]
                .iter()
                .zip(v)
                .map(|(a, b)| a * b)
                .sum()
        };

        let heads: Vec<Vec<f32>> = (0..ix.n_q_heads)
            .map(|h| {
                let raw: Vec<f32> = (0..d).map(|i| dot(&ix.q_w, h * d + i, row(q))).collect();
                scalar_rope(
                    &scalar_rms(&raw, &ix.q_norm_w, ix.eps),
                    positions[q],
                    ix.n_rot,
                    ix.rope_theta,
                )
            })
            .collect();

        (0..n_blocks)
            .map(|b| {
                let mut pooled = vec![0f32; d];
                for t in b * ix.ratio..(b + 1) * ix.ratio {
                    for (o, p) in pooled.iter_mut().enumerate() {
                        *p += dot(&ix.k_w, o, row(t));
                    }
                }
                for p in pooled.iter_mut() {
                    *p /= ix.ratio as f32;
                }
                let key = scalar_rope(
                    &scalar_rms(&pooled, &ix.k_norm_w, ix.eps),
                    positions[b * ix.ratio],
                    ix.n_rot,
                    ix.rope_theta,
                );
                let sum: f32 = heads
                    .iter()
                    .map(|qh| {
                        qh.iter()
                            .zip(&key)
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                            .max(0.0)
                    })
                    .sum();
                sum / (d as f32).sqrt()
            })
            .collect()
    }

    /// Positions are consumed, and they are consumed correctly.
    ///
    /// Three claims, in the order they are asserted:
    ///
    /// 1. At positions `1000..`, every score equals what an independently
    ///    written scalar norm+rope path says it should be. This is the real
    ///    proof that `positions` reaches the rope with the right value — the
    ///    scalar path derives the angle from the position it is handed.
    /// 2. Shifting query AND block-first positions by the same constant is a
    ///    no-op to within f32 noise. That is RoPE's defining property (a score
    ///    depends only on the difference of the two positions), and it is worth
    ///    pinning: a reference that roped the query at its position but the
    ///    block key at the block INDEX would fail it.
    /// 3. Shifting the query's position alone DOES move the scores, which is
    ///    what rules out a reference that ignores `positions` altogether —
    ///    claim 2 alone would be satisfied by roping at nothing.
    #[test]
    fn positions_reach_the_rope_and_only_their_differences_matter() {
        let f = fixture();
        let ix = indexer(&f);
        let case = &f["case_above_budget"];
        let x = f32s_2d(&case["hidden_states"]);
        let n_tok = x.len() / ix.hidden;
        let base: Vec<usize> = (0..n_tok).collect();
        let shifted: Vec<usize> = (0..n_tok).map(|t| 1000 + t).collect();

        let raw = ix.raw_keys(&x);
        let per_tok = ix.n_q_heads * ix.head_dim;
        let q_base = ix.queries(&x, &base);
        let q_shift = ix.queries(&x, &shifted);

        let q_plus_one = ix.queries(&x, &(0..n_tok).map(|t| t + 1).collect::<Vec<_>>());

        let mut worst_uniform = 0f32;
        let mut any_moved = false;
        for q in 0..n_tok {
            let n_blocks = (q + 1) / ix.ratio;
            if n_blocks == 0 {
                continue;
            }
            let keys_base = ix.block_keys(
                &raw[..(q + 1) * ix.head_dim],
                &(0..n_blocks)
                    .map(|b| base[b * ix.ratio])
                    .collect::<Vec<_>>(),
            );
            let keys_shift = ix.block_keys(
                &raw[..(q + 1) * ix.head_dim],
                &(0..n_blocks)
                    .map(|b| shifted[b * ix.ratio])
                    .collect::<Vec<_>>(),
            );
            let s_base = ix.scores(&q_base[q * per_tok..(q + 1) * per_tok], &keys_base);
            let s_shift = ix.scores(&q_shift[q * per_tok..(q + 1) * per_tok], &keys_shift);

            let want = scalar_scores(&ix, &x, &shifted, q, n_blocks);
            for (b, (&g, &w)) in s_shift.iter().zip(&want).enumerate() {
                assert!(close(g, w), "query {q} block {b}: {g} vs scalar {w}");
            }
            for (a, b) in s_base.iter().zip(&s_shift) {
                worst_uniform = worst_uniform.max((a - b).abs());
            }

            let s_q_only = ix.scores(&q_plus_one[q * per_tok..(q + 1) * per_tok], &keys_base);
            any_moved |= s_base
                .iter()
                .zip(&s_q_only)
                .any(|(a, b)| (a - b).abs() > 1e-3);
        }
        // Measured at 1.2e-7 — a single f32 ulp on scores of order 1, which is
        // the rounding of cos/sin at the shifted angle and nothing else.
        assert!(
            worst_uniform < 1e-5,
            "a uniform position shift moved the scores by {worst_uniform:e}: \
             the query and the block key are not roped on the same scale"
        );
        assert!(
            any_moved,
            "moving the query one position further from every block left every \
             score unchanged — `positions` is not reaching the rope"
        );
    }

    /// `block_keys` ropes block `b` at `first_positions[b]` and at nothing
    /// else. With scrambled, non-monotonic first positions each row must equal
    /// the row a single-block call at that one position produces, and the whole
    /// output must differ from the `b * ratio` default.
    #[test]
    fn block_keys_follow_the_given_first_positions() {
        let f = fixture();
        let ix = indexer(&f);
        let x = f32s_2d(&f["case_above_budget"]["hidden_states"]);
        let raw = ix.raw_keys(&x);
        let d = ix.head_dim;
        let n_blocks = (x.len() / ix.hidden) / ix.ratio;
        assert!(n_blocks >= 3, "the scramble needs at least three blocks");

        let scrambled: Vec<usize> = (0..n_blocks).map(|b| [37usize, 2, 91][b % 3] + b).collect();
        let default: Vec<usize> = (0..n_blocks).map(|b| b * ix.ratio).collect();
        let got = ix.block_keys(&raw, &scrambled);
        let plain = ix.block_keys(&raw, &default);

        for b in 0..n_blocks {
            let one = ix.block_keys(
                &raw[b * ix.ratio * d..(b + 1) * ix.ratio * d],
                &scrambled[b..b + 1],
            );
            assert_eq!(one, got[b * d..(b + 1) * d], "block {b} row");
        }
        assert!(
            got.iter().zip(&plain).any(|(a, b)| (a - b).abs() > 1e-5),
            "scrambling the first positions changed nothing — `block_keys` is \
             roping at `b * ratio` regardless of its argument"
        );
    }

    /// Chunked prefill equivalence. Raw keys are cached, so a run that projects
    /// them in three chunks and concatenates must select exactly what a
    /// single-shot run selects: blocks are cut from the sequence-start-anchored
    /// visible run, never from the chunk that produced the keys.
    #[test]
    fn chunked_raw_keys_select_like_a_single_shot_run() {
        let f = fixture();
        let ix = indexer(&f);
        let x = f32s_2d(&f["case_above_budget"]["hidden_states"]);
        let n_tok = x.len() / ix.hidden;
        let positions: Vec<usize> = (0..n_tok).collect();
        let want = ix.select_all(&x, &positions);

        let bounds = [0, 3, 7, n_tok];
        assert!(bounds[3] > bounds[2], "the fixture is long enough to cut");
        let mut raw = Vec::new();
        let mut q_all = Vec::new();
        for w in bounds.windows(2) {
            let (lo, hi) = (w[0], w[1]);
            let chunk = &x[lo * ix.hidden..hi * ix.hidden];
            raw.extend(ix.raw_keys(chunk));
            q_all.extend(ix.queries(chunk, &positions[lo..hi]));
        }
        assert_eq!(raw.len(), n_tok * ix.head_dim);

        let per_tok = ix.n_q_heads * ix.head_dim;
        for q in 0..n_tok {
            let visible: Vec<usize> = (0..=q).collect();
            let n_blocks = visible.len() / ix.ratio;
            let firsts: Vec<usize> = (0..n_blocks)
                .map(|b| positions[visible[b * ix.ratio]])
                .collect();
            let keys = ix.block_keys(&raw[..visible.len() * ix.head_dim], &firsts);
            let scores = ix.scores(&q_all[q * per_tok..(q + 1) * per_tok], &keys);
            let got = select(&scores, &visible, ix.budget, ix.ratio);
            assert_eq!(got, want[q], "query {q}: chunked selection");
        }
    }
}
