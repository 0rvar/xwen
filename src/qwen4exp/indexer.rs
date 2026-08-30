//! The QSA indexer: the device path of `ref_qsa`.
//!
//! One of these sits beside every full-attention block of a `qwen4exp` layer
//! stack. It owns a tiny MQA projection pair (`indexer.q_proj` /
//! `indexer.k_proj`), its own two norms, and a cache of RAW key vectors — one
//! per token, un-normed and un-roped, because a block's key is normed and roped
//! AFTER pooling, at a position that belongs to the block rather than to any of
//! its tokens. Its whole output is [`QsaSelection`]: which cached tokens the
//! ordinary attention of that layer is allowed to see.
//!
//! `ref_qsa` states the math and owns the decisions (pooled keys stay f32;
//! whole blocks plus the raw tail, NOT llama.cpp's fill-to-width; ties broken
//! toward the lower block index). This module is graded against it and repeats
//! none of that reasoning — read `ref_qsa`'s header for the why.
//!
//! # Where the work runs
//!
//! Projections, both norms, both ropes, the block mean-pool and the scores run
//! on device as ordinary candle ops. **The top-k and the expansion of blocks
//! into a token set run on the HOST**, off a `[n_q, n_blocks]` score readback.
//! Two reasons, both P2-shaped:
//!
//! 1. The tie rule is a set-identity property, and a host `select_nth_unstable`
//!    under an explicit (score descending, block index ascending) total order
//!    reproduces it exactly. `arg_sort` would make it a property of candle's
//!    sort stability instead — and at long context the relu floors a large
//!    fraction of block scores at exactly 0, so ties are the normal case, not
//!    an edge case.
//! 2. The prefill overlay is materialized as a `[n_q, n_kv]` additive mask
//!    anyway, so the round trip buys the assembly for free.
//!
//! The cost is real and known: at 2200 tokens the readback is ~5 MB and the
//! mask upload ~19 MB (the f32 `[n_q, n_kv]` plane), per QSA layer per forward,
//! plus the ~9.7 MB f16 copy `PrefillMask::from_raw` makes of it for sdpa. That
//! f16 copy is ONE plane broadcast across the heads, not one per head — see
//! `from_raw`, where materializing it would be 232 MB at this length. A partial
//! top-k kernel plus a device-side block→token expansion is the P3 replacement
//! (TODO.md); the selection sets it must reproduce are pinned by the tests
//! below.
//!
//! # Blocks are query-independent
//!
//! Every query's visible set is the contiguous prefix `0..=q_pos`, so block `b`
//! covers tokens `b*ratio .. (b+1)*ratio` for EVERY query and its rope position
//! is `b*ratio` for every query. The block keys are therefore pooled, normed and
//! roped ONCE per forward over the longest prefix any query in the chunk sees,
//! and query `q` simply scores against the first `(q_pos+1)/ratio` of them. That
//! is also why a chunked prefill and a single-shot one select identically: the
//! blocks are cut from the sequence, never from the chunk.

use std::cmp::Ordering;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Module, Tensor};

use crate::config::XwenConfig;
use crate::gguf::{QLinear, Weights};
use crate::rope::Rope;

/// What the indexer tells [`crate::attention::AttnBlock`] to attend over.
///
/// The three arms are the three shapes the overlay takes, not three policies:
/// `Dense` is "every visible token wins", which is what selection degenerates
/// to below budget, and it must leave the attention path byte-identical to a
/// run with no indexer at all.
pub enum QsaSelection {
    /// Every query sees its whole causal prefix — the caller's own mask (or the
    /// maskless decode route) is already correct, so nothing is overlaid.
    Dense,
    /// Prefill overlay: `[n_q, n_kv]` f32, additive, `0.0` visible and `-inf`
    /// hidden. Causality is ALREADY in it (a selected set is a subset of the
    /// prefix), so it replaces the causal mask rather than composing with it.
    Mask(Tensor),
    /// Decode overlay: `u32 [n_sel]`, the selected token indices ascending. The
    /// attention gathers exactly these K/V rows and runs maskless — candle's
    /// vector sdpa silently ignores a mask, so a mask is not an option at
    /// `seq == 1` (D11).
    Rows(Tensor),
}

/// The indexer's `[out_dim, in_dim]` projection, held at whatever the file
/// stores.
///
/// BF16 is the shipped case and the reason this enum exists: upstream's
/// converter puts `indexer.{q,k}_proj` on the quantizer's SKIP list, so they
/// arrive at the source precision in every quant mix (verified on
/// `UD-Q4_K_XL`: `[2560,512]` and `[2560,128]`, both BF16, while the two norms
/// are F32) — and candle's `QMatMul` has no bf16 route. The other arms cover a
/// self-converted file that let the quantizer touch them, and the f32 weights a
/// test constructs directly.
enum IdxProj {
    /// Dense bf16 `[out, in]`, mmap-aliased; `ops::matmul_bf16`.
    Bf16(Tensor),
    /// Dense f16 `[out, in]`, mmap-aliased; `ops::matmul_f16`.
    F16(Tensor),
    /// Dense f32 `[out, in]`; plain candle matmul. The `from_tensors` path.
    F32(Tensor),
    /// Anything quantized, behind candle's `QMatMul`.
    Quant(QLinear),
}

impl IdxProj {
    fn load(w: &Weights, name: &str) -> Result<Self> {
        Ok(match w.stored_dtype(name)? {
            GgmlDType::BF16 => IdxProj::Bf16(w.dense_bf16(name)?),
            GgmlDType::F16 => IdxProj::F16(w.dense_f16(name)?),
            GgmlDType::F32 => IdxProj::F32(w.dense_f32(name)?),
            _ => IdxProj::Quant(w.qlinear(name)?),
        })
    }

    /// `[n, in] f32` → `[n, out] f32`. The vendored mixed-dtype kernels want a
    /// contiguous f32 activation; every caller here passes one.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            IdxProj::Bf16(w) => crate::ops::matmul_bf16(w, x),
            IdxProj::F16(w) => crate::ops::matmul_f16(w, x),
            IdxProj::F32(w) => Ok(x.matmul(&w.t()?)?),
            IdxProj::Quant(q) => Ok(q.forward(x)?),
        }
    }

    /// `[out_dim, in_dim]`, for the load-time shape check.
    fn dims(&self) -> Result<(usize, usize)> {
        Ok(match self {
            IdxProj::Bf16(w) | IdxProj::F16(w) | IdxProj::F32(w) => w.dims2()?,
            IdxProj::Quant(q) => (q.out_dim, q.in_dim),
        })
    }
}

/// One QSA layer's indexer. Geometry on the shipped checkpoint: 4 query heads,
/// one key head, head dim 128, budget 2048 tokens, ratio 4 — top-512 blocks.
pub struct QsaIndexer {
    q_proj: IdxProj,
    /// MQA: exactly one key head, so `k_proj` is `[head_dim, hidden]`.
    k_proj: IdxProj,
    q_norm: candle_nn::RmsNorm,
    k_norm: candle_nn::RmsNorm,
    /// The TRUNK's rope tables, shared: same theta and same `n_rot` (64), and
    /// `Rope::rotate` reads the head dim off its input, so the indexer's 128
    /// and the attention's 256 use one table pair.
    rope: Arc<Rope>,
    n_heads: usize,
    head_dim: usize,
    /// Token budget. Whole blocks are kept, so `budget / ratio` of them.
    budget: usize,
    /// Tokens per block.
    ratio: usize,
}

impl QsaIndexer {
    /// `w` is positioned at the block prefix (e.g. `blk.3`); the tensors read
    /// are `indexer.{q_proj,k_proj,q_norm,k_norm}.weight`.
    pub fn load(w: &Weights, cfg: &XwenConfig, rope: Arc<Rope>) -> Result<Self> {
        let q4 = cfg
            .qwen4exp
            .as_ref()
            .context("QsaIndexer::load: the config carries no qwen4exp section")?;
        let (n_heads, head_dim) = (q4.indexer_heads, q4.indexer_head_dim);
        let me = Self {
            q_proj: IdxProj::load(w, "indexer.q_proj")?,
            k_proj: IdxProj::load(w, "indexer.k_proj")?,
            q_norm: candle_nn::RmsNorm::new(w.dense_f32("indexer.q_norm")?, cfg.rms_eps),
            k_norm: candle_nn::RmsNorm::new(w.dense_f32("indexer.k_norm")?, cfg.rms_eps),
            rope,
            n_heads,
            head_dim,
            budget: q4.indexer_top_k,
            ratio: q4.indexer_compress_ratio,
        };
        me.check_shapes()?;
        Ok(me)
    }

    /// An indexer over weights the caller chose, instead of a GGUF: `q_w`
    /// `[n_heads * head_dim, hidden]`, `k_w` `[head_dim, hidden]`, both f32 and
    /// both `[out, in]` (the GGUF layout); the two norm weights `[head_dim]`
    /// f32 and multiply-ready.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_tensors(
        q_w: Tensor,
        k_w: Tensor,
        q_norm_w: Tensor,
        k_norm_w: Tensor,
        rope: Arc<Rope>,
        n_heads: usize,
        head_dim: usize,
        budget: usize,
        ratio: usize,
        eps: f64,
    ) -> Result<Self> {
        let me = Self {
            q_proj: IdxProj::F32(q_w),
            k_proj: IdxProj::F32(k_w),
            q_norm: candle_nn::RmsNorm::new(q_norm_w, eps),
            k_norm: candle_nn::RmsNorm::new(k_norm_w, eps),
            rope,
            n_heads,
            head_dim,
            budget,
            ratio,
        };
        me.check_shapes()?;
        Ok(me)
    }

    /// The MQA assumption is load-bearing everywhere below (`k_proj`'s output
    /// IS the key vector, with no head axis to split), so it is checked once
    /// here rather than discovered as a wrong answer: a file with a wider
    /// `k_proj` would otherwise be silently truncated to its first head.
    fn check_shapes(&self) -> Result<()> {
        ensure!(self.ratio > 0, "indexer compress ratio is 0");
        ensure!(self.n_heads > 0, "indexer head count is 0");
        // Selection keeps WHOLE blocks, so the budget is spent in units of
        // `ratio` tokens: `keep_max = budget / ratio`. A budget under one block
        // truncates to keeping nothing, and one that is not a whole number of
        // blocks quietly grants fewer tokens than the file asked for — 2049 at
        // ratio 4 buys the same 512 blocks as 2048. Neither shows up as an
        // error later; the selection is simply smaller than the checkpoint was
        // trained with.
        ensure!(
            self.budget >= self.ratio && self.budget.is_multiple_of(self.ratio),
            "indexer top_k {} is not a whole number of {}-token blocks: selection keeps whole \
             blocks, so this budget would silently round down",
            self.budget,
            self.ratio
        );
        let (q_out, q_in) = self.q_proj.dims()?;
        let (k_out, k_in) = self.k_proj.dims()?;
        ensure!(
            q_out == self.n_heads * self.head_dim,
            "indexer.q_proj is [{q_out}, {q_in}], expected [{} , {q_in}] for {} heads of {}",
            self.n_heads * self.head_dim,
            self.n_heads,
            self.head_dim
        );
        ensure!(
            k_out == self.head_dim,
            "indexer.k_proj is [{k_out}, {k_in}]: the indexer key side is MQA, so it must be \
             exactly one head of {}",
            self.head_dim
        );
        ensure!(
            q_in == k_in,
            "indexer q/k projections read different widths ({q_in} vs {k_in})"
        );
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// A raw-key cache sized for this indexer.
    pub fn new_cache(&self, max_ctx: usize, device: &Device) -> Result<IndexerCache> {
        IndexerCache::new(max_ctx, self.head_dim, device)
    }

    /// Append this chunk's raw keys and decide what its queries may attend to.
    ///
    /// `x_normed` is `[n, hidden]` f32 — the SAME pre-attention normed stream
    /// the layer's q/k/v projections read. `pos` is the chunk's absolute start
    /// position and must equal the cache's current length: the indexer is
    /// driven strictly sequentially, exactly like the K/V cache beside it.
    pub fn select(
        &self,
        x_normed: &Tensor,
        cache: &mut IndexerCache,
        pos: usize,
    ) -> Result<QsaSelection> {
        let (n, _hidden) = x_normed.dims2()?;
        ensure!(
            cache.len == pos,
            "QsaIndexer::select: the raw-key cache holds {} tokens but the chunk starts at {pos}",
            cache.len
        );
        ensure!(cache.head_dim() == self.head_dim, "cache head dim mismatch");

        // Keys are cached RAW: the projection and nothing else. Norm and rope
        // are applied to the POOLED block key, at the block's position.
        let raw = self.k_proj.forward(x_normed)?;
        cache.append(&raw)?;

        let n_kv = pos + n;
        // Below budget every complete block fits, the tail is always visible,
        // and selection is exactly the causal prefix. Returning `Dense` here is
        // not an optimization of a different answer — it IS the answer, and it
        // keeps the shipped short-context case bit-identical to a run with no
        // indexer (which is also the cheap dev-time equivalence check).
        if n_kv <= self.budget {
            return Ok(QsaSelection::Dense);
        }

        let device = x_normed.device();
        // Queries: project, per-head RMS norm over head_dim, rope at each
        // token's own position — which IS a consecutive run from `pos`.
        let q = self
            .q_proj
            .forward(x_normed)?
            .reshape((n, self.n_heads, self.head_dim))?;
        let q = self.q_norm.forward(&q)?;
        let q = self.rope.rotate(
            &q.transpose(0, 1)?.contiguous()?, // [n_heads, n, head_dim]
            pos,
            DType::F32,
        )?;

        // Block keys, once for the whole chunk (see the module header): mean in
        // f32 over each complete run of `ratio` raw keys, RMS norm, rope at the
        // run's first token. The pooled key is deliberately NOT rounded back to
        // the cache dtype first (ref_qsa "Pooled keys stay f32").
        let n_blocks = n_kv / self.ratio;
        let pooled = cache
            .visible(n_blocks * self.ratio)?
            .reshape((n_blocks, self.ratio, self.head_dim))?
            .mean(1)?;
        let block_pos: Vec<u32> = (0..n_blocks).map(|b| (b * self.ratio) as u32).collect();
        let block_pos = Tensor::from_vec(block_pos, n_blocks, device)?;
        let keys = self
            .rope
            .rotate_at(
                &self
                    .k_norm
                    .forward(&pooled)?
                    .reshape((1, n_blocks, self.head_dim))?,
                &block_pos,
                DType::F32,
            )?
            .reshape((n_blocks, self.head_dim))?;

        // score[q, b] = Σ_heads relu(q_h · k_b) / √head_dim. The relu is per
        // head, BEFORE the sum, so a block every head dislikes scores exactly 0
        // rather than going negative — which is why exact ties are ordinary at
        // long context and the tie rule below is load-bearing.
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let per_head = q
            .reshape((self.n_heads * n, self.head_dim))?
            .matmul(&keys.t()?)?
            .relu()?
            .reshape((self.n_heads, n, n_blocks))?;
        let scores = (per_head.sum(0)? * scale)?; // [n, n_blocks]
        let scores = scores.flatten_all()?.to_vec1::<f32>()?;

        let sets = self.top_blocks(&scores, n, n_blocks, pos);

        if n == 1 {
            let mut rows: Vec<u32> = Vec::with_capacity(sets[0].len() * self.ratio + self.ratio);
            self.expand_into(&sets[0], pos, &mut rows);
            let n_sel = rows.len();
            return Ok(QsaSelection::Rows(Tensor::from_vec(rows, n_sel, device)?));
        }

        let mut mask = vec![f32::NEG_INFINITY; n * n_kv];
        let mut tokens: Vec<u32> = Vec::new();
        for (i, blocks) in sets.iter().enumerate() {
            tokens.clear();
            self.expand_into(blocks, pos + i, &mut tokens);
            let row = &mut mask[i * n_kv..(i + 1) * n_kv];
            for &t in &tokens {
                row[t as usize] = 0.0;
            }
        }
        Ok(QsaSelection::Mask(Tensor::from_vec(
            mask,
            (n, n_kv),
            device,
        )?))
    }

    /// The top blocks for each query of a chunk, ascending by block index.
    ///
    /// `scores` is the flattened `[n, n_blocks]` readback. Query `i` sits at
    /// absolute position `pos + i` and sees the first `(pos + i + 1) / ratio`
    /// blocks; it keeps `min(budget / ratio, that)` of them.
    ///
    /// Ties keep the LOWER block index, matching `ref_qsa::select`. The
    /// comparator says so outright — score descending, then index ascending —
    /// rather than leaning on a sort's stability, because the relu makes exact
    /// ties the common case rather than a curiosity.
    fn top_blocks(&self, scores: &[f32], n: usize, n_blocks: usize, pos: usize) -> Vec<Vec<usize>> {
        let keep_max = self.budget / self.ratio;
        let mut order: Vec<usize> = Vec::with_capacity(n_blocks);
        (0..n)
            .map(|i| {
                let visible = pos + i + 1;
                let nb = (visible / self.ratio).min(n_blocks);
                let keep = keep_max.min(nb);
                if keep == 0 {
                    return Vec::new();
                }
                let row = &scores[i * n_blocks..i * n_blocks + nb];
                order.clear();
                order.extend(0..nb);
                if keep < nb {
                    // A total order (the index breaks every tie), so a partial
                    // selection names exactly the same set a full sort would.
                    order.select_nth_unstable_by(keep - 1, |&a, &b| {
                        row[b]
                            .partial_cmp(&row[a])
                            .unwrap_or(Ordering::Equal)
                            .then(a.cmp(&b))
                    });
                }
                let mut kept = order[..keep].to_vec();
                kept.sort_unstable();
                kept
            })
            .collect()
    }

    /// Expand a query's selected blocks into its ascending token set: each
    /// block WHOLE, plus the raw tail — the `visible % ratio` most recent
    /// tokens that never formed a complete block, which are always visible.
    ///
    /// `blocks` arrives ascending and the tail sits above every block token by
    /// construction, so the appended result is ascending — which `Rows`
    /// promises its consumer and the mask path does not care about.
    fn expand_into(&self, blocks: &[usize], q_pos: usize, out: &mut Vec<u32>) {
        for &b in blocks {
            for t in b * self.ratio..(b + 1) * self.ratio {
                out.push(t as u32);
            }
        }
        let visible = q_pos + 1;
        for t in (visible - visible % self.ratio)..visible {
            out.push(t as u32);
        }
    }
}

/// One QSA layer's raw indexer keys, `[max_ctx, head_dim]` f32, appended in
/// position order.
///
/// This is the `LayerCache::Full` story with one head and no values: every
/// token writes its own absolute slot, so a rollback is a pure length
/// truncation and a checkpoint carries no data. It lives here rather than in
/// `kv_cache.rs` deliberately (D15) — it is a qwen4exp overlay, not a fifth
/// layer kind. What `kv_cache.rs` does carry is its IMAGE: a page-out copies
/// these rows into the `HostFullKv` beside the trunk's K/V, because a
/// position-indexed plane is exactly the half of a conversation's state a
/// fixed-size snapshot cannot reconstruct.
///
/// f32, not the BF16 the HF implementation caches: the pooled key is used at
/// f32 anyway (`ref_qsa` D13), and 4 MB per layer at 8k context is not worth a
/// precision argument.
pub struct IndexerCache {
    raw: Tensor,
    len: usize,
}

/// Bytes one more position of context costs in ONE QSA layer's indexer key
/// plane.
///
/// Two things about that plane make the obvious arithmetic wrong, which is why
/// this is a function every caller shares rather than a formula each restates.
/// It is MQA — exactly ONE key head, whatever the query head count is (4 on the
/// shipped checkpoint) — and [`IndexerCache::new`] allocates it f32, not the
/// f16 the trunk's KV rows use. 512 bytes per token per QSA layer here, where
/// reasoning from the query heads and the trunk's dtype gives 1024.
///
/// Both the size an operator is told to budget for
/// (`Model::kv_bytes_per_token`) and the size actually allocated
/// (`super::stack::extra_state_bytes`) come from here, so the two cannot drift.
pub const fn indexer_bytes_per_token(head_dim: usize) -> usize {
    // One key head x head_dim x size_of::<f32>().
    head_dim * 4
}

/// [`IndexerCache::checkpoint`]'s output. Carries nothing, for the same reason
/// [`crate::kv_cache::LayerCheckpoint::Full`] carries nothing: rows below the
/// rollback point still hold what they held, and rows above it are reclaimed by
/// the next append.
pub struct IndexerCheckpoint;

impl IndexerCache {
    pub fn new(max_ctx: usize, head_dim: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            raw: Tensor::zeros((max_ctx, head_dim), DType::F32, device)?,
            len: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn head_dim(&self) -> usize {
        self.raw.dim(1).expect("raw keys are rank-2")
    }

    pub fn capacity(&self) -> usize {
        self.raw.dim(0).expect("raw keys are rank-2")
    }

    /// The device this plane lives on, for an import that has to build its
    /// upload somewhere. Read off the allocation itself rather than passed in, so
    /// a caller cannot hand rows to a cache on another device.
    pub(crate) fn device(&self) -> Result<Device> {
        Ok(self.raw.device().clone())
    }

    /// The first `upto` cached keys, `[upto, head_dim]`.
    fn visible(&self, upto: usize) -> Result<Tensor> {
        ensure!(
            upto <= self.len,
            "indexer cache holds {} keys, asked for {upto}",
            self.len
        );
        Ok(self.raw.narrow(0, 0, upto)?)
    }

    /// Append `[n, head_dim]` f32 raw keys at the current length.
    fn append(&mut self, keys: &Tensor) -> Result<()> {
        let (n, d) = keys.dims2()?;
        ensure!(
            d == self.head_dim(),
            "indexer cache is {} wide, got {d}",
            self.head_dim()
        );
        ensure!(
            self.len + n <= self.capacity(),
            "indexer cache holds {} of {} slots; {n} more overruns it",
            self.len,
            self.capacity()
        );
        self.raw.slice_set(keys, 0, self.len)?;
        self.len += n;
        Ok(())
    }

    /// The committed rows `[0, len)` as little-endian f32 bytes, row-major
    /// `(len, head_dim)`.
    ///
    /// The trunk's K/V planes are `(head, position, dim)` and a prefix of the
    /// positions is therefore a gather; this plane has ONE key head, so a prefix
    /// of the positions IS a prefix of the buffer. Every range operation on the
    /// host image leans on that.
    pub(crate) fn export_rows(&self) -> Result<Vec<u8>> {
        crate::kv_cache::f32_bytes(&self.raw.narrow(0, 0, self.len)?)
    }

    /// Upload `bytes` — `(pos, head_dim)` f32, as produced by
    /// [`Self::export_rows`] — into rows `[0, pos)` and set the length to `pos`.
    ///
    /// Rows above `pos` keep whatever they held; nothing reads them, exactly as
    /// nothing reads a truncated `Full` layer's stale slots.
    pub(crate) fn import_rows(&mut self, bytes: &[u8], pos: usize, device: &Device) -> Result<()> {
        let head_dim = self.head_dim();
        ensure!(
            pos <= self.capacity(),
            "indexer cache holds {} slots; an import of {pos} rows overruns it",
            self.capacity()
        );
        // Nothing to upload, and a zero-row tensor is not worth asking candle
        // for: an empty import is the same state a reset leaves behind.
        if pos > 0 {
            let rows = crate::kv_cache::f32_tensor(bytes, &[pos, head_dim], device)?;
            self.raw.slice_set(&rows, 0, 0)?;
        }
        self.len = pos;
        Ok(())
    }

    /// Drop everything at or after `len`. Rows above it keep their bytes and
    /// are overwritten by the next append, exactly as a truncated `Full` layer
    /// does.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        ensure!(
            len <= self.len,
            "indexer cache holds {} keys; cannot truncate up to {len}",
            self.len
        );
        self.len = len;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Arm a rollback over the next `span` tokens. Nothing to record — see
    /// [`IndexerCheckpoint`].
    pub fn checkpoint(&mut self, _span: usize) -> Result<IndexerCheckpoint> {
        Ok(IndexerCheckpoint)
    }

    /// Roll a verify forward back to `len0 + commit`, discarding the rejected
    /// tail. Mirrors `LayerCache::rollback`'s `Full` arm, argument for argument,
    /// so the model can call the two from one loop.
    pub fn rollback(
        &mut self,
        _ckpt: &IndexerCheckpoint,
        len0: usize,
        span: usize,
        commit: usize,
    ) -> Result<()> {
        ensure!(
            commit <= span,
            "indexer rollback: commit {commit} exceeds the {span}-token span"
        );
        ensure!(
            len0 + span <= self.len,
            "indexer rollback: the verify forward stepped {} tokens, not the {span} the \
             checkpoint reserved from {len0}",
            self.len.saturating_sub(len0)
        );
        self.len = len0 + commit;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RopeKind;
    use crate::gguf::metal_device;
    use crate::qwen4exp::ref_qsa::QsaIndexerRef;
    use serde_json::Value;

    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen4exp/qsa_indexer.json"
    );

    /// Deterministic pseudo-random f32s in [lo, hi] (xorshift, no deps).
    fn rand(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = (s >> 11) as f64 / (1u64 << 53) as f64;
                lo + (hi - lo) * u as f32
            })
            .collect()
    }

    fn fixture() -> Value {
        serde_json::from_str(&std::fs::read_to_string(FIXTURE).expect("fixture readable"))
            .expect("fixture parses")
    }

    fn f32s(v: &Value) -> Vec<f32> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|e| e.as_f64().expect("number") as f32)
            .collect()
    }

    fn f32s_2d(v: &Value) -> Vec<f32> {
        v.as_array().expect("rows").iter().flat_map(f32s).collect()
    }

    /// The fixture's geometry, its weights, and the hidden states of `case`.
    fn fixture_indexer(f: &Value, device: &Device) -> (QsaIndexer, usize, usize) {
        let cfg = &f["config"];
        let n_heads = cfg["indexer_n_heads"].as_u64().unwrap() as usize;
        let head_dim = cfg["indexer_head_dim"].as_u64().unwrap() as usize;
        let hidden = cfg["hidden_size"].as_u64().unwrap() as usize;
        let n_rot = cfg["rotary_dim"].as_u64().unwrap() as usize;
        let theta = cfg["rope_theta"].as_f64().unwrap() as f32;
        let budget = cfg["indexer_budget"].as_u64().unwrap() as usize;
        let ratio = cfg["indexer_compress_ratio"].as_u64().unwrap() as usize;
        let eps = cfg["rms_norm_eps"].as_f64().unwrap();

        // One fused `[(n_heads + 1) * head_dim, hidden]` projection: q heads
        // first, the single MQA k head last — the split the converter makes.
        let proj = f32s_2d(&f["weights"]["index_qk_proj"]);
        let split = n_heads * head_dim * hidden;
        let q_w =
            Tensor::from_vec(proj[..split].to_vec(), (n_heads * head_dim, hidden), device).unwrap();
        let k_w = Tensor::from_vec(
            proj[split..split + head_dim * hidden].to_vec(),
            (head_dim, hidden),
            device,
        )
        .unwrap();
        let q_norm = Tensor::from_vec(
            f32s(&f["weights"]["q_layernorm_weight_mult"]),
            head_dim,
            device,
        )
        .unwrap();
        let k_norm = Tensor::from_vec(
            f32s(&f["weights"]["k_layernorm_weight_mult"]),
            head_dim,
            device,
        )
        .unwrap();

        let rope = Arc::new(
            Rope::new(
                &RopeKind::Plain {
                    freq_base: theta,
                    n_rot,
                },
                4096,
                device,
            )
            .unwrap(),
        );
        let ix = QsaIndexer::from_tensors(
            q_w, k_w, q_norm, k_norm, rope, n_heads, head_dim, budget, ratio, eps,
        )
        .unwrap();
        (ix, hidden, ratio)
    }

    /// Read a `[n_q, n_kv]` additive mask back as one ascending token set per
    /// query: finite entries are visible, `-inf` entries are not.
    fn mask_sets(mask: &Tensor) -> Vec<Vec<usize>> {
        let (n_q, n_kv) = mask.dims2().unwrap();
        let flat: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        (0..n_q)
            .map(|i| {
                (0..n_kv)
                    .filter(|&t| flat[i * n_kv + t].is_finite())
                    .collect()
            })
            .collect()
    }

    fn rows_set(rows: &Tensor) -> Vec<usize> {
        rows.to_vec1::<u32>()
            .unwrap()
            .into_iter()
            .map(|t| t as usize)
            .collect()
    }

    fn usizes_2d(v: &Value) -> Vec<Vec<usize>> {
        v.as_array()
            .expect("rows")
            .iter()
            .map(|row| {
                let mut r: Vec<usize> = row
                    .as_array()
                    .expect("row")
                    .iter()
                    .map(|e| e.as_u64().expect("index") as usize)
                    .collect();
                r.sort_unstable();
                r
            })
            .collect()
    }

    /// The device selection over the whole above-budget fixture case, in one
    /// prefill chunk, equals the fixture's per-query token sets exactly.
    #[test]
    fn fixture_above_budget_selection_matches() {
        let device = metal_device().unwrap();
        let f = fixture();
        let (ix, hidden, _) = fixture_indexer(&f, &device);
        let case = &f["case_above_budget"];
        let x = f32s_2d(&case["hidden_states"]);
        let n = x.len() / hidden;
        let x = Tensor::from_vec(x, (n, hidden), &device).unwrap();

        let mut cache = ix.new_cache(64, &device).unwrap();
        let sel = ix.select(&x, &mut cache, 0).unwrap();
        let QsaSelection::Mask(mask) = sel else {
            panic!("an above-budget prefill must produce a Mask");
        };
        assert_eq!(mask.dims2().unwrap(), (n, n));

        let want = usizes_2d(&case["selected_token_indices"]);
        let counts = f32s(&case["selected_counts"]);
        for (q, got) in mask_sets(&mask).iter().enumerate() {
            assert_eq!(*got, want[q], "query {q}: selected set");
            assert_eq!(got.len(), counts[q] as usize, "query {q}: count");
        }
    }

    /// A single-token decode step above budget produces `Rows` naming exactly
    /// the fixture's set for that query — including the case where the query
    /// cannot see its own token (fixture query 15: no tail, and its own block
    /// loses the top-k).
    #[test]
    fn fixture_decode_rows_match() {
        let device = metal_device().unwrap();
        let f = fixture();
        let (ix, hidden, _) = fixture_indexer(&f, &device);
        let case = &f["case_above_budget"];
        let x = f32s_2d(&case["hidden_states"]);
        let n = x.len() / hidden;
        let want = usizes_2d(&case["selected_token_indices"]);

        let x = Tensor::from_vec(x, (n, hidden), &device).unwrap();
        let mut cache = ix.new_cache(64, &device).unwrap();
        // Prefill everything but the last token, then step the last one alone.
        let head = x.narrow(0, 0, n - 1).unwrap().contiguous().unwrap();
        ix.select(&head, &mut cache, 0).unwrap();
        let tail = x.narrow(0, n - 1, 1).unwrap().contiguous().unwrap();
        let sel = ix.select(&tail, &mut cache, n - 1).unwrap();
        let QsaSelection::Rows(rows) = sel else {
            panic!("an above-budget decode step must produce Rows");
        };
        let got = rows_set(&rows);
        assert!(
            got.windows(2).all(|w| w[0] < w[1]),
            "rows must be ascending"
        );
        assert_eq!(got, want[n - 1], "query {}: selected set", n - 1);
        assert!(
            !got.contains(&(n - 1)),
            "fixture query 15 cannot see its own token"
        );
    }

    /// Below budget, selection is the whole causal prefix — reported as
    /// `Dense`, so the attention path is left exactly as it would be with no
    /// indexer at all.
    #[test]
    fn fixture_below_budget_is_dense() {
        let device = metal_device().unwrap();
        let f = fixture();
        let (ix, hidden, _) = fixture_indexer(&f, &device);
        let case = &f["case_below_budget"];
        assert!(case["selected_equals_causal"].as_bool().unwrap());
        let x = f32s_2d(&case["hidden_states"]);
        let n = x.len() / hidden;
        let x = Tensor::from_vec(x, (n, hidden), &device).unwrap();

        let mut cache = ix.new_cache(64, &device).unwrap();
        assert!(matches!(
            ix.select(&x, &mut cache, 0).unwrap(),
            QsaSelection::Dense
        ));
        assert_eq!(cache.len(), n, "the raw keys are cached either way");
    }

    // ---- real geometry ----

    const HIDDEN: usize = 2560;
    const N_HEADS: usize = 4;
    const HEAD_DIM: usize = 128;
    const BUDGET: usize = 2048;
    const RATIO: usize = 4;
    const N_ROT: usize = 64;
    const THETA: f32 = 1e7;
    const EPS: f64 = 1e-6;
    const SEQ: usize = 2200;

    /// The shipped geometry with random weights, plus the `ref_qsa` oracle over
    /// the same numbers.
    fn real_geometry(device: &Device) -> (QsaIndexer, QsaIndexerRef, Tensor, Vec<f32>) {
        let q_w = rand(0x51A, N_HEADS * HEAD_DIM * HIDDEN, -0.04, 0.04);
        let k_w = rand(0x51B, HEAD_DIM * HIDDEN, -0.04, 0.04);
        let q_norm = rand(0x51C, HEAD_DIM, 0.5, 1.5);
        let k_norm = rand(0x51D, HEAD_DIM, 0.5, 1.5);
        let x = rand(0x51E, SEQ * HIDDEN, -1.0, 1.0);

        let rope = Arc::new(
            Rope::new(
                &RopeKind::Plain {
                    freq_base: THETA,
                    n_rot: N_ROT,
                },
                8192,
                device,
            )
            .unwrap(),
        );
        let ix = QsaIndexer::from_tensors(
            Tensor::from_vec(q_w.clone(), (N_HEADS * HEAD_DIM, HIDDEN), device).unwrap(),
            Tensor::from_vec(k_w.clone(), (HEAD_DIM, HIDDEN), device).unwrap(),
            Tensor::from_vec(q_norm.clone(), HEAD_DIM, device).unwrap(),
            Tensor::from_vec(k_norm.clone(), HEAD_DIM, device).unwrap(),
            rope,
            N_HEADS,
            HEAD_DIM,
            BUDGET,
            RATIO,
            EPS,
        )
        .unwrap();
        let reference = QsaIndexerRef {
            hidden: HIDDEN,
            n_q_heads: N_HEADS,
            n_kv_heads: 1,
            head_dim: HEAD_DIM,
            n_rot: N_ROT,
            rope_theta: THETA,
            budget: BUDGET,
            ratio: RATIO,
            q_w,
            k_w,
            q_norm_w: q_norm,
            k_norm_w: k_norm,
            eps: EPS as f32,
        };
        let xt = Tensor::from_vec(x.clone(), (SEQ, HIDDEN), device).unwrap();
        (ix, reference, xt, x)
    }

    /// Every query's selected token set, exactly, against the reference.
    ///
    /// Exact set equality is the right assertion here even though the top-k cut
    /// at this geometry lands among near-ties. Only the last 149 of the 2200
    /// queries reject any block at all (a query needs 513 complete blocks
    /// before `min(512, n_blocks)` bites, so position 2051 onward), and 128 of
    /// those 149 have a gap between the last selected and best rejected block
    /// of less than 1e-3 — the cut runs through blocks the per-head relu
    /// floored at exactly 0. What makes the two sides agree anyway is not luck
    /// but the shared tie rule — score descending, then block index ascending —
    /// which `top_blocks` states outright rather than inheriting from a sort's
    /// stability. Measured with that rule in place: 0 of 2200 queries differ.
    fn assert_sets_match(got: &[Vec<usize>], want: &[Vec<usize>], first_query: usize, what: &str) {
        for (i, g) in got.iter().enumerate() {
            let q = first_query + i;
            assert_eq!(*g, want[q], "{what}: query {q}");
        }
    }

    /// The causal prefix of every query in `0..n` — what `Dense` means as a set.
    fn causal_sets(n: usize) -> Vec<Vec<usize>> {
        (0..n).map(|q| (0..=q).collect()).collect()
    }

    /// Real geometry, one 2200-token prefill: every query's selected token set
    /// equals `ref_qsa::select_all`'s.
    #[test]
    fn real_geometry_prefill_matches_reference() {
        let device = metal_device().unwrap();
        let (ix, reference, xt, x) = real_geometry(&device);
        let positions: Vec<usize> = (0..SEQ).collect();
        let want = reference.select_all(&x, &positions);

        let mut cache = ix.new_cache(8192, &device).unwrap();
        let QsaSelection::Mask(mask) = ix.select(&xt, &mut cache, 0).unwrap() else {
            panic!("2200 tokens is above the 2048 budget: expected a Mask");
        };
        assert_sets_match(&mask_sets(&mask), &want, 0, "prefill");
    }

    /// A decode step at position 2200 gathers the same rows the reference
    /// selects for that query.
    #[test]
    fn real_geometry_decode_matches_reference() {
        let device = metal_device().unwrap();
        let (ix, reference, xt, x) = real_geometry(&device);
        // One more token than the prefill, so the decode query is position 2200.
        let extra = rand(0x51F, HIDDEN, -1.0, 1.0);
        let mut all = x.clone();
        all.extend_from_slice(&extra);
        let positions: Vec<usize> = (0..=SEQ).collect();
        let want = reference.select_all(&all, &positions);

        let mut cache = ix.new_cache(8192, &device).unwrap();
        ix.select(&xt, &mut cache, 0).unwrap();
        let step = Tensor::from_vec(extra, (1, HIDDEN), &device).unwrap();
        let QsaSelection::Rows(rows) = ix.select(&step, &mut cache, SEQ).unwrap() else {
            panic!("a single token above budget must produce Rows");
        };
        let got = rows_set(&rows);
        assert!(
            got.windows(2).all(|w| w[0] < w[1]),
            "rows must be ascending"
        );
        assert_sets_match(&[got], &want, SEQ, "decode");
    }

    /// A chunked prefill selects what a single-shot one selects: blocks are cut
    /// from the sequence-start-anchored run of cached raw keys, never from the
    /// chunk that produced them.
    ///
    /// The 1500-token first chunk is entirely below the 2048 budget and so
    /// reports `Dense`, which is the same SET as the single-shot run's first
    /// 1500 mask rows — the single-shot run is above budget only because of the
    /// tokens that come later, and a query at position 1499 still sees every
    /// one of its 375 blocks.
    #[test]
    fn chunked_prefill_selects_like_one_shot() {
        let device = metal_device().unwrap();
        let (ix, _, xt, _) = real_geometry(&device);

        let mut one = ix.new_cache(8192, &device).unwrap();
        let QsaSelection::Mask(full) = ix.select(&xt, &mut one, 0).unwrap() else {
            panic!("expected a Mask");
        };
        let want = mask_sets(&full);

        const CUT: usize = 1500;
        let mut cache = ix.new_cache(8192, &device).unwrap();
        let a = xt.narrow(0, 0, CUT).unwrap().contiguous().unwrap();
        let b = xt.narrow(0, CUT, SEQ - CUT).unwrap().contiguous().unwrap();
        assert!(
            matches!(ix.select(&a, &mut cache, 0).unwrap(), QsaSelection::Dense),
            "1500 tokens is below the 2048 budget"
        );
        let QsaSelection::Mask(m_b) = ix.select(&b, &mut cache, CUT).unwrap() else {
            panic!("expected a Mask");
        };
        assert_eq!(m_b.dims2().unwrap(), (SEQ - CUT, SEQ));

        let mut got = causal_sets(CUT);
        got.extend(mask_sets(&m_b));
        assert_sets_match(&got, &want, 0, "chunked");
    }

    /// A rollback puts the raw-key cache back to the committed length, and the
    /// selection it then produces is the one an unbroken run would have.
    #[test]
    fn rollback_restores_the_selection() {
        let device = metal_device().unwrap();
        let f = fixture();
        let (ix, hidden, _) = fixture_indexer(&f, &device);
        let x = f32s_2d(&f["case_above_budget"]["hidden_states"]);
        let n = x.len() / hidden;
        let x = Tensor::from_vec(x, (n, hidden), &device).unwrap();

        let plain = {
            let mut cache = ix.new_cache(64, &device).unwrap();
            let QsaSelection::Mask(m) = ix.select(&x, &mut cache, 0).unwrap() else {
                panic!("expected a Mask");
            };
            mask_sets(&m)
        };

        let mut cache = ix.new_cache(64, &device).unwrap();
        let head = x.narrow(0, 0, 4).unwrap().contiguous().unwrap();
        ix.select(&head, &mut cache, 0).unwrap();
        // Speculate four tokens, then accept one of them.
        let ckpt = cache.checkpoint(4).unwrap();
        let spec = x.narrow(0, 4, 4).unwrap().contiguous().unwrap();
        ix.select(&spec, &mut cache, 4).unwrap();
        cache.rollback(&ckpt, 4, 4, 1).unwrap();
        assert_eq!(cache.len(), 5);

        let rest = x.narrow(0, 5, n - 5).unwrap().contiguous().unwrap();
        let QsaSelection::Mask(m) = ix.select(&rest, &mut cache, 5).unwrap() else {
            panic!("expected a Mask");
        };
        for (i, g) in mask_sets(&m).iter().enumerate() {
            assert_eq!(*g, plain[5 + i], "query {}: after a rollback", 5 + i);
        }
    }

    /// `select` is strictly sequential: a chunk that does not start where the
    /// cache ends is a caller bug, not a resync.
    #[test]
    fn a_position_gap_is_refused() {
        let device = metal_device().unwrap();
        let f = fixture();
        let (ix, hidden, _) = fixture_indexer(&f, &device);
        let x = f32s_2d(&f["case_above_budget"]["hidden_states"]);
        let n = x.len() / hidden;
        let x = Tensor::from_vec(x, (n, hidden), &device).unwrap();
        let mut cache = ix.new_cache(64, &device).unwrap();
        assert!(ix.select(&x, &mut cache, 3).is_err());
    }

    /// The token budget is spent in whole blocks (`keep_max = budget / ratio`),
    /// so a budget that is not a whole number of blocks is refused at load
    /// rather than rounded down in silence. A file asking for 2049 tokens at
    /// ratio 4 gets the same 512 blocks as one asking for 2048, and a file
    /// asking for fewer than `ratio` gets nothing at all — neither reads as an
    /// error anywhere downstream, they just make the selection smaller than the
    /// checkpoint was trained with.
    #[test]
    fn a_budget_that_is_not_whole_blocks_is_refused() {
        let device = metal_device().unwrap();
        let (heads, hd, hidden) = (2usize, 8usize, 16usize);
        let rope = Arc::new(
            Rope::new(
                &RopeKind::Plain {
                    freq_base: THETA,
                    n_rot: 4,
                },
                64,
                &device,
            )
            .unwrap(),
        );
        let build = |budget: usize, ratio: usize| {
            QsaIndexer::from_tensors(
                Tensor::zeros((heads * hd, hidden), DType::F32, &device).unwrap(),
                Tensor::zeros((hd, hidden), DType::F32, &device).unwrap(),
                Tensor::ones(hd, DType::F32, &device).unwrap(),
                Tensor::ones(hd, DType::F32, &device).unwrap(),
                rope.clone(),
                heads,
                hd,
                budget,
                ratio,
                EPS,
            )
        };

        assert!(build(2048, 4).is_ok(), "the shipped geometry");
        assert!(build(4, 4).is_ok(), "exactly one block is a legal budget");

        let err = build(2049, 4).err().unwrap().to_string();
        assert!(err.contains("whole number"), "{err}");
        assert!(build(2, 4).is_err(), "a budget under one block");
        assert!(build(0, 4).is_err(), "a zero budget");
        assert!(build(2048, 0).is_err(), "a zero ratio");
    }
}
