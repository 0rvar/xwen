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
//! on device as ordinary candle ops. What happens next depends on the chunk:
//!
//! - **A decode step (one query) selects ON DEVICE**: `kernel_qsa_select`
//!   (`ops::qsa_select`) takes the `[n_blocks]` score row, finds the top-k by
//!   radix select over the score bits and writes the expanded row list, so the
//!   step never reads anything back. Before it, each of the 12 QSA layers
//!   drained the pipeline once per token for a readback of a few KB, and the
//!   GPU idled while the CPU encoded the next layer. `XWEN_QSA_HOST_TOPK`
//!   restores the readback.
//! - **A prefill chunk selects on the HOST**, off a `[n_q, n_blocks]` score
//!   readback, because its overlay is materialized as a `[n_q, n_kv]` additive
//!   mask anyway, so the round trip buys the assembly for free.
//!
//! Both paths implement one tie rule, and it is a set-identity property: a
//! total order (score descending, block index ascending) that the host states
//! outright in `select_nth_unstable`'s comparator and the kernel reproduces
//! over the score bits. `arg_sort` would have made it a property of candle's
//! sort stability instead — and at long context the relu floors a large
//! fraction of block scores at exactly 0, so ties are the normal case, not an
//! edge case. The kernel is held to the host's rows bit for bit
//! (`device_select_matches_host_top_blocks_bitwise`).
//!
//! The prefill cost is real and known: at 2200 tokens the readback is ~5 MB
//! and the mask upload ~19 MB (the f32 `[n_q, n_kv]` plane), per QSA layer per
//! forward, plus the ~9.7 MB f16 copy `PrefillMask::from_raw` makes of it for
//! sdpa. That f16 copy is ONE plane broadcast across the heads, not one per
//! head — see `from_raw`, where materializing it would be 232 MB at this
//! length. A device-side mask assembly is the remaining replacement (TODO.md);
//! the selection sets it must reproduce are pinned by the tests below.
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
        IndexerCache::new(max_ctx, self.head_dim, self.ratio, device)
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
        let classic = crate::ops::qsa_classic();
        self.select_with(
            x_normed,
            cache,
            pos,
            classic,
            classic || crate::ops::qsa_host_topk(),
        )
    }

    /// [`Self::select`] with the two decode paths chosen by the caller instead
    /// of by `XWEN_QSA_CLASSIC` / `XWEN_QSA_HOST_TOPK`: `classic` recomputes
    /// every block key from the raw rows each call
    /// ([`Self::block_keys_classic`]), the default reads them off the cache's
    /// block plane ([`Self::block_keys_cached`]); `host_topk` reads a decode
    /// step's scores back and selects on the host ([`Self::top_blocks`]), the
    /// default selects on device (`ops::qsa_select`). Each pair is
    /// bit-identical (`cached_block_keys_match_the_classic_recompute`,
    /// `device_select_matches_host_top_blocks_bitwise`); the split exists so
    /// one test can run the arms over one scripted sequence.
    fn select_with(
        &self,
        x_normed: &Tensor,
        cache: &mut IndexerCache,
        pos: usize,
        classic: bool,
        host_topk: bool,
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

        // Block keys, `[n_blocks, head_dim]` f32 (see the module header and
        // `block_keys`).
        let n_blocks = n_kv / self.ratio;
        let keys = if classic {
            self.block_keys_classic(cache, n_blocks)?
        } else {
            self.block_keys_cached(cache, n_blocks)?
        };

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
        let scores = (strided_sum(&per_head, 0)? * scale)?; // [n, n_blocks]

        // A decode step selects on device: no readback, so the CPU keeps
        // encoding the next layer while the GPU is still on this one. The
        // prefill overlay is a host-assembled mask and keeps the readback.
        if n == 1 && !host_topk && matches!(device, Device::Metal(_)) {
            let keep = (self.budget / self.ratio).min(n_blocks);
            // At a single-token step `n_kv == pos + 1` (the cache length was
            // checked to equal `pos` above), so this tail is exactly the
            // `(pos + 1) % ratio` positions `expand_into` would append.
            let tail = n_kv - n_blocks * self.ratio;
            let rows = crate::ops::qsa_select::select_rows(
                &scores.flatten_all()?.contiguous()?,
                keep,
                self.ratio,
                tail,
            )?;
            return Ok(QsaSelection::Rows(rows));
        }

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

    /// The keys of blocks `[start, start + m)` from their raw rows: mean in f32
    /// over each run of `ratio` raw keys, RMS norm, rope at the run's first
    /// token. The pooled key is deliberately NOT rounded back to the cache dtype
    /// first (ref_qsa "Pooled keys stay f32"). `[m, head_dim]` f32.
    ///
    /// Every step is per-block — the pool reads one block's rows, the norm is
    /// per row, the rope is per element at that block's own position — so the
    /// keys of a block range are the same bits whether they are computed alone
    /// or inside a wider range. That independence is what lets the cached path
    /// build only the blocks that are new and still agree with the classic
    /// path's whole-prefix recompute, bit for bit.
    fn block_keys(&self, cache: &IndexerCache, start: usize, m: usize) -> Result<Tensor> {
        let device = cache.raw.device();
        let runs = cache
            .visible((start + m) * self.ratio)?
            .narrow(0, start * self.ratio, m * self.ratio)?
            .reshape((m, self.ratio, self.head_dim))?;
        let pooled = (strided_sum(&runs, 1)? * (1.0 / self.ratio as f64))?;
        let block_pos: Vec<u32> = (start..start + m)
            .map(|b| (b * self.ratio) as u32)
            .collect();
        let block_pos = Tensor::from_vec(block_pos, m, device)?;
        Ok(self
            .rope
            .rotate_at(
                &self
                    .k_norm
                    .forward(&pooled)?
                    .reshape((1, m, self.head_dim))?,
                &block_pos,
                DType::F32,
            )?
            .reshape((m, self.head_dim))?)
    }

    /// The `XWEN_QSA_CLASSIC` arm: every complete block's key, recomputed from
    /// the raw rows on every call.
    fn block_keys_classic(&self, cache: &IndexerCache, n_blocks: usize) -> Result<Tensor> {
        self.block_keys(cache, 0, n_blocks)
    }

    /// The default arm: the first `n_blocks` rows of the cache's block plane,
    /// after building the ones it does not hold yet — `[blocks_ready, n_blocks)`
    /// — in ONE batch. A complete block's key depends only on its own `ratio`
    /// raw rows, which never change while they sit below the cache length, so a
    /// row of the plane stays valid until a truncation drops below its block
    /// ([`IndexerCache::set_len`]). At decode that is no key work on three
    /// steps of four and one block on the fourth; at prefill it is the chunk's
    /// blocks; after an import it is a one-time rebuild.
    ///
    /// The returned tensor is a LIVE narrow over the block plane, not a copy:
    /// a later append can rewrite rows at or above `blocks_ready`, so it must
    /// not outlive the `select_with` call that asked for it.
    fn block_keys_cached(&self, cache: &mut IndexerCache, n_blocks: usize) -> Result<Tensor> {
        ensure!(
            cache.ratio == self.ratio,
            "indexer cache was sized for ratio {}, the indexer's is {}",
            cache.ratio,
            self.ratio
        );
        if cache.blocks_ready < n_blocks {
            let start = cache.blocks_ready;
            let fresh = self.block_keys(cache, start, n_blocks - start)?;
            cache.blocks.slice_set(&fresh, 0, start)?;
            cache.blocks_ready = n_blocks;
        }
        Ok(cache.blocks.narrow(0, 0, n_blocks)?)
    }

    /// The top blocks for each query of a chunk, ascending by block index.
    ///
    /// `scores` is the flattened `[n, n_blocks]` readback. Query `i` sits at
    /// absolute position `pos + i` and sees the first `(pos + i + 1) / ratio`
    /// blocks; it keeps `min(budget / ratio, that)` of them.
    ///
    /// Ties keep the LOWER block index, matching `ref_qsa::select`. The
    /// comparator says so outright — score descending, then index ascending,
    /// both through [`score_key`] — rather than leaning on a sort's stability,
    /// because the relu makes exact ties the common case rather than a
    /// curiosity.
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
                    // selection names exactly the same set a full sort would
                    // — and the same set the device kernel names, since both
                    // rank by `score_key`.
                    order.select_nth_unstable_by(keep - 1, |&a, &b| {
                        score_key(row[b]).cmp(&score_key(row[a])).then(a.cmp(&b))
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

/// The ordering key of one block score, shared by the host selection
/// (`QsaIndexer::top_blocks`) and the device one (`kernel_qsa_select`'s
/// `score_key`, which must stay the same function).
///
/// A non-negative finite float orders by its bit pattern, denormals included,
/// so the key IS the score's order for everything the relu'd score can be. A
/// set sign bit — `-0.0`, or a negative that the relu makes impossible — keys
/// as 0, and so does a NaN (host and device alike, so a NaN is a tie with
/// every zero-scored block and nothing else; it is outside the contract
/// either way). Because every input maps to an integer, the comparator built
/// on it is a TOTAL order (key descending, block index ascending) — where a
/// `partial_cmp` fallback would have made a NaN "equal" to everything and
/// the selection depend on the walk order.
pub(crate) fn score_key(s: f32) -> u32 {
    let u = s.to_bits();
    if u & 0x8000_0000 != 0 || s.is_nan() {
        0
    } else {
        u
    }
}

/// Sum `t` over `dim` as narrows added in index order — `((t_0 + t_1) + t_2)
/// + ...` — instead of `Tensor::sum`.
///
/// candle routes a sum over a non-trailing axis of a contiguous tensor to
/// `fast_sum_f32_strided`, which launches one threadgroup of TWO threads per
/// output element: over the `[n_blocks, ratio, head_dim]` pool that is
/// `n_blocks * head_dim` tiny threadgroups per call, the single largest item
/// on the above-budget decode step. The extents summed here are 4 (the block
/// ratio, the query head count), so a handful of elementwise adds over
/// contiguous rows is both cheaper and — with the reduce's own
/// per-thread-then-tree order reproduced for an extent of 4, `(t_0 + t_2) +
/// (t_1 + t_3)` — bit-identical to it, which is what keeps the classic and
/// default arms of `select_with` on the same selections.
///
/// Two limits of that identity, one enforced and one noted. Enforced: extents
/// above 5 are refused — from 4 threads up candle's reducer folds the partials
/// through a 4-lane `simd_sum`, whose order this tree does not reproduce (a
/// last-ulp difference, measured at extent 6). Noted: candle seeds each
/// thread's accumulator from +0.0 where this seeds from the first element, so
/// an all-`-0.0` slice sums to `-0.0` here and `+0.0` there — a sign of zero
/// only, and neither caller can produce it (relu'd scores and pooled
/// projections are never all negative zero).
fn strided_sum(t: &Tensor, dim: usize) -> Result<Tensor> {
    let n = t.dim(dim)?;
    ensure!(n > 0, "strided_sum over an empty axis");
    ensure!(
        n <= 5,
        "strided_sum over an extent of {n}: only extents up to 5 (one or two reduce threads) \
         reproduce candle's strided reduce bit for bit; above that its 4-lane simd_sum fold \
         orders the partials differently"
    );
    let part = |i: usize| -> Result<Tensor> { Ok(t.narrow(dim, i, 1)?.squeeze(dim)?) };
    // candle's strided reduce runs `(n / 2).next_power_of_two()` threads
    // (integer division: 1 thread up to n = 3, 2 at n = 4 and 5, then 4), each
    // accumulating the elements `i ≡ tid (mod width)` in order, then combines
    // the per-thread partials. With one or two partials there is exactly one
    // way to add them, so extents up to 5 reproduce the kernel's bits
    // (`strided_sum_matches_candle_reduce_bitwise`); from 4 threads up the
    // kernel folds through `simd_sum`, whose order this halving tree does not
    // promise to match — a last-ulp difference, and no caller here has an
    // extent above 4.
    let width = (n / 2).next_power_of_two().max(1);
    let mut partial: Vec<Option<Tensor>> = vec![None; width];
    for i in 0..n {
        let slot = &mut partial[i % width];
        *slot = Some(match slot.take() {
            None => part(i)?,
            Some(acc) => (acc + part(i)?)?,
        });
    }
    let mut stride = width / 2;
    while stride > 0 {
        for tid in 0..stride {
            let lo = partial[tid].take();
            let hi = partial[tid + stride].take();
            partial[tid] = match (lo, hi) {
                (Some(a), Some(b)) => Some((a + b)?),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            };
        }
        stride /= 2;
    }
    partial[0].take().context("strided_sum: no partials")
}

/// One QSA layer's raw indexer keys, `[max_ctx, head_dim]` f32, appended in
/// position order, plus the derived plane of complete-block keys.
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
///
/// The block plane (`blocks`, `[max_ctx / ratio, head_dim]` f32) is DERIVED
/// state — the pooled, normed, roped key of each complete block, filled in
/// batches by `QsaIndexer::block_keys_cached` and valid for the first
/// `blocks_ready` rows only. It is never exported: an image carries the raw
/// rows, and the import leaves `blocks_ready` at 0 so the next `select`
/// rebuilds the plane in one batch. Every write to `len` goes through
/// [`Self::set_len`], which is where the plane is invalidated.
pub struct IndexerCache {
    raw: Tensor,
    len: usize,
    /// Tokens per block, fixing the block plane's row count and the
    /// `len -> blocks_ready` clamp.
    ratio: usize,
    blocks: Tensor,
    /// Leading rows of `blocks` that hold a valid key. Rows at or above it are
    /// stale and never read.
    blocks_ready: usize,
}

/// Bytes one more position of context costs in ONE QSA layer's indexer planes:
/// the raw key row plus its share of the derived block-key plane.
///
/// Two things about the raw plane make the obvious arithmetic wrong, which is
/// why this is a function every caller shares rather than a formula each
/// restates. It is MQA — exactly ONE key head, whatever the query head count is
/// (4 on the shipped checkpoint) — and [`IndexerCache::new`] allocates it f32,
/// not the f16 the trunk's KV rows use. 512 bytes per token per QSA layer
/// there, where reasoning from the query heads and the trunk's dtype gives
/// 1024. The block plane adds one f32 key per `ratio` tokens: 128 more at the
/// shipped ratio of 4, 640 in all.
///
/// The block-plane term is AMORTIZED — one row per `ratio` tokens, spread
/// evenly — and the plane itself is device-only derived state: it is not part
/// of the host image a page-out or snapshot carries, so this figure sizes
/// device memory per position, not the bytes a snapshot stores. (The serve
/// `--init` template's `ctx_gb` estimate is built from it and so now counts
/// ~4% of device-only bytes; that is the intended reading of a context
/// budget.) The exact allocation of a whole plane pair is
/// [`indexer_plane_bytes`].
///
/// Both the size an operator is told to budget for
/// (`Model::kv_bytes_per_token`) and the size actually allocated
/// (`super::stack::extra_state_bytes`, via [`indexer_plane_bytes`]) rest on the
/// same row size, so the two cannot drift.
pub const fn indexer_bytes_per_token(head_dim: usize, ratio: usize) -> usize {
    // One key head x head_dim x size_of::<f32>(), plus the block plane's row
    // spread over the ratio tokens it summarizes.
    head_dim * 4 + head_dim * 4 / ratio
}

/// Bytes [`IndexerCache::new`] allocates for ONE QSA layer at `max_ctx`: the
/// raw plane (`max_ctx` rows) plus the block plane (`max_ctx / ratio` rows, at
/// least one), each row `head_dim` f32. Exact, where
/// [`indexer_bytes_per_token`] amortizes the block row.
pub const fn indexer_plane_bytes(head_dim: usize, ratio: usize, max_ctx: usize) -> usize {
    let row = head_dim * 4;
    let block_rows = if max_ctx / ratio > 0 {
        max_ctx / ratio
    } else {
        1
    };
    max_ctx * row + block_rows * row
}

/// [`IndexerCache::checkpoint`]'s output. Carries nothing, for the same reason
/// [`crate::kv_cache::LayerCheckpoint::Full`] carries nothing: rows below the
/// rollback point still hold what they held, and rows above it are reclaimed by
/// the next append.
pub struct IndexerCheckpoint;

impl IndexerCache {
    /// `ratio` is the indexer's block size; the block plane holds `max_ctx /
    /// ratio` keys (at least one row, so a plane can always be allocated).
    pub fn new(max_ctx: usize, head_dim: usize, ratio: usize, device: &Device) -> Result<Self> {
        ensure!(ratio > 0, "indexer cache: block ratio is 0");
        Ok(Self {
            raw: Tensor::zeros((max_ctx, head_dim), DType::F32, device)?,
            len: 0,
            ratio,
            blocks: Tensor::zeros(((max_ctx / ratio).max(1), head_dim), DType::F32, device)?,
            blocks_ready: 0,
        })
    }

    /// The ONE place `len` changes. A block's cached key is valid only while
    /// all `ratio` of its raw rows sit below `len` — the next append rewrites
    /// the rows above it — so every shrink clamps `blocks_ready` to the
    /// complete blocks that remain. A growth never invalidates anything.
    fn set_len(&mut self, len: usize) {
        self.len = len;
        self.blocks_ready = self.blocks_ready.min(len / self.ratio);
    }

    /// Complete blocks whose cached key is currently valid (test/diagnostic
    /// visibility into the derived plane).
    #[cfg(test)]
    pub(crate) fn blocks_ready(&self) -> usize {
        self.blocks_ready
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
        self.set_len(self.len + n);
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
        // The rows below `pos` were just REPLACED, not kept, so no block key
        // computed from the previous contents survives: a clamp would keep
        // keys of the conversation this image is overwriting.
        self.blocks_ready = 0;
        self.set_len(pos);
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
        self.set_len(len);
        Ok(())
    }

    pub fn reset(&mut self) {
        self.set_len(0);
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
        self.set_len(len0 + commit);
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

    fn bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// The selection a `select_with` call produced, as comparable bits: the
    /// mask plane for a chunk, the row list for a step, nothing for `Dense`.
    fn selection_bits(sel: &QsaSelection) -> Option<Vec<u32>> {
        match sel {
            QsaSelection::Dense => None,
            QsaSelection::Mask(m) => Some(bits(m)),
            QsaSelection::Rows(r) => Some(r.to_vec1::<u32>().unwrap()),
        }
    }

    /// The block plane's valid rows against a from-scratch recompute of the
    /// same blocks, bit for bit. A call that selected (anything but `Dense`)
    /// must have left every complete block cached.
    fn assert_block_plane_matches(
        ix: &QsaIndexer,
        cache: &IndexerCache,
        selected: bool,
        what: &str,
    ) {
        let n_blocks = cache.blocks_ready();
        assert!(
            n_blocks <= cache.len() / RATIO,
            "{what}: ready blocks exceed the length"
        );
        if selected {
            assert_eq!(
                n_blocks,
                cache.len() / RATIO,
                "{what}: every complete block is cached"
            );
        }
        if n_blocks == 0 {
            return;
        }
        let cached = cache.blocks.narrow(0, 0, n_blocks).unwrap();
        let scratch = ix.block_keys_classic(cache, n_blocks).unwrap();
        let (c, s) = (bits(&cached), bits(&scratch));
        let diff = c.iter().zip(&s).filter(|(a, b)| a != b).count();
        assert_eq!(
            diff,
            0,
            "{what}: {diff} of {} block-key elements differ",
            c.len()
        );
    }

    /// The cached block-key path and the classic full recompute make the same
    /// selections over one scripted sequence — prefill chunks that cross the
    /// budget, single-token steps, a rollback, a truncation below a block
    /// boundary, more steps — and the plane the cached path holds after each
    /// call equals a from-scratch recompute of those blocks bit for bit.
    #[test]
    fn cached_block_keys_match_the_classic_recompute() {
        let device = metal_device().unwrap();
        let (ix, _, xt, _) = real_geometry(&device);
        let extra = rand(0x520, 16 * HIDDEN, -1.0, 1.0);
        let extra = Tensor::from_vec(extra, (16, HIDDEN), &device).unwrap();
        let step = |i: usize| extra.narrow(0, i, 1).unwrap().contiguous().unwrap();

        let mut classic = ix.new_cache(8192, &device).unwrap();
        let mut cached = ix.new_cache(8192, &device).unwrap();
        let mut host = ix.new_cache(8192, &device).unwrap();
        // Three arms: classic keys + host top-k (the pre-fast-path answer),
        // cached keys + device top-k (what ships), and cached keys + host
        // top-k in between so a difference names the path that caused it.
        fn run(
            ix: &QsaIndexer,
            classic: &mut IndexerCache,
            cached: &mut IndexerCache,
            host: &mut IndexerCache,
            x: &Tensor,
            pos: usize,
            what: &str,
        ) -> QsaSelection {
            let a = ix.select_with(x, classic, pos, true, true).unwrap();
            let h = ix.select_with(x, host, pos, false, true).unwrap();
            let b = ix.select_with(x, cached, pos, false, false).unwrap();
            assert_eq!(
                selection_bits(&a),
                selection_bits(&h),
                "{what}: selection, classic vs cached keys"
            );
            assert_eq!(
                selection_bits(&h),
                selection_bits(&b),
                "{what}: selection, host vs device top-k"
            );
            assert_eq!(classic.len(), cached.len(), "{what}: length");
            assert_eq!(host.len(), cached.len(), "{what}: length");
            assert_block_plane_matches(ix, cached, !matches!(b, QsaSelection::Dense), what);
            b
        }

        // Two prefill chunks: the first below budget (Dense, but the plane
        // still fills), the second crossing it.
        let a = xt.narrow(0, 0, 1500).unwrap().contiguous().unwrap();
        assert!(matches!(
            run(
                &ix,
                &mut classic,
                &mut cached,
                &mut host,
                &a,
                0,
                "chunk 0..1500"
            ),
            QsaSelection::Dense
        ));
        assert_eq!(cached.blocks_ready(), 0, "a Dense chunk builds no keys");
        let b = xt
            .narrow(0, 1500, SEQ - 1500)
            .unwrap()
            .contiguous()
            .unwrap();
        assert!(matches!(
            run(
                &ix,
                &mut classic,
                &mut cached,
                &mut host,
                &b,
                1500,
                "chunk 1500..2200"
            ),
            QsaSelection::Mask(_)
        ));
        assert_eq!(cached.blocks_ready(), SEQ / RATIO);

        // Five decode steps: a new block completes on one of every four.
        for i in 0..5 {
            let pos = SEQ + i;
            let before = cached.blocks_ready();
            assert!(matches!(
                run(
                    &ix,
                    &mut classic,
                    &mut cached,
                    &mut host,
                    &step(i),
                    pos,
                    "step"
                ),
                QsaSelection::Rows(_)
            ));
            assert_eq!(cached.blocks_ready(), (pos + 1) / RATIO);
            assert!(cached.blocks_ready() - before <= 1);
        }
        // 2205 tokens: 551 complete blocks. Speculate 4 more, keep 2 — the
        // rollback drops the block that closed at 2208.
        let ck_c = classic.checkpoint(4).unwrap();
        let ck_f = cached.checkpoint(4).unwrap();
        let ck_h = host.checkpoint(4).unwrap();
        let spec = extra.narrow(0, 5, 4).unwrap().contiguous().unwrap();
        run(
            &ix,
            &mut classic,
            &mut cached,
            &mut host,
            &spec,
            2205,
            "speculated 4",
        );
        assert_eq!(cached.blocks_ready(), 2209 / RATIO);
        classic.rollback(&ck_c, 2205, 4, 2).unwrap();
        cached.rollback(&ck_f, 2205, 4, 2).unwrap();
        host.rollback(&ck_h, 2205, 4, 2).unwrap();
        assert_eq!(cached.len(), 2207);
        assert_eq!(
            cached.blocks_ready(),
            2207 / RATIO,
            "rollback clamps the plane"
        );
        assert_block_plane_matches(&ix, &cached, true, "after rollback");

        // Truncate below a block boundary, then refill those rows with
        // DIFFERENT tokens: the block that spanned the cut is rebuilt.
        classic.truncate(2202).unwrap();
        cached.truncate(2202).unwrap();
        host.truncate(2202).unwrap();
        assert_eq!(cached.blocks_ready(), 550, "truncate clamps the plane");
        for i in 9..14 {
            let pos = 2202 + (i - 9);
            run(
                &ix,
                &mut classic,
                &mut cached,
                &mut host,
                &step(i),
                pos,
                "post-truncate step",
            );
        }
        assert_eq!(cached.len(), 2207);

        // A reset empties the plane; an import rebuilds it from the rows.
        let image = cached.export_rows().unwrap();
        cached.reset();
        host.reset();
        assert_eq!(cached.blocks_ready(), 0);
        cached.import_rows(&image, 2207, &device).unwrap();
        host.import_rows(&image, 2207, &device).unwrap();
        assert_eq!(cached.blocks_ready(), 0, "an import trusts no cached key");
        run(
            &ix,
            &mut classic,
            &mut cached,
            &mut host,
            &step(14),
            2207,
            "post-import step",
        );
        assert_eq!(cached.blocks_ready(), 2208 / RATIO);
    }

    /// A minimal indexer whose only job is to own `budget` and `ratio` for
    /// `top_blocks` / `expand_into` — the host side of the selection.
    fn selector(budget: usize, ratio: usize, device: &Device) -> QsaIndexer {
        let (heads, hd, hidden) = (1usize, 8usize, 8usize);
        let rope = Arc::new(
            Rope::new(
                &RopeKind::Plain {
                    freq_base: THETA,
                    n_rot: 4,
                },
                64,
                device,
            )
            .unwrap(),
        );
        QsaIndexer::from_tensors(
            Tensor::zeros((heads * hd, hidden), DType::F32, device).unwrap(),
            Tensor::zeros((hd, hidden), DType::F32, device).unwrap(),
            Tensor::ones(hd, DType::F32, device).unwrap(),
            Tensor::ones(hd, DType::F32, device).unwrap(),
            rope,
            heads,
            hd,
            budget,
            ratio,
            EPS,
        )
        .unwrap()
    }

    /// The host selection for one query at position `pos` over `scores`:
    /// `top_blocks` then `expand_into`, exactly as `select_with`'s host arm
    /// runs them.
    fn host_rows(ix: &QsaIndexer, scores: &[f32], nb: usize, pos: usize) -> Vec<u32> {
        let sets = ix.top_blocks(scores, 1, nb, pos);
        let mut rows = Vec::new();
        ix.expand_into(&sets[0], pos, &mut rows);
        rows
    }

    /// The device selection for the same query.
    fn device_rows(
        scores: &[f32],
        keep: usize,
        ratio: usize,
        tail: usize,
        device: &Device,
    ) -> Vec<u32> {
        let t = Tensor::from_vec(scores.to_vec(), scores.len(), device).unwrap();
        crate::ops::qsa_select::select_rows(&t, keep, ratio, tail)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()
    }

    /// Scores the way long context produces them: a few distinct values with
    /// exact 0.0 the most common by far, plus the two things the key
    /// canonicalization has to get right — `-0.0` (equal to 0.0, and the
    /// host's `partial_cmp` says so) and denormals (above 0.0, below
    /// everything else, and monotone in their bit pattern).
    fn tied_scores(seed: u64, nb: usize) -> Vec<f32> {
        rand(seed, nb, 0.0, 1.0)
            .into_iter()
            .map(|u| match (u * 16.0) as u32 {
                0..=6 => 0.0,
                7 => -0.0,
                8 => 1e-40,
                9 => 1e-41,
                10 => 0.5,
                11 => 0.5,
                12 => 1.0,
                13 => 2.0,
                14 => 7.25,
                _ => 3.0e5,
            })
            .collect()
    }

    /// `kernel_qsa_select` produces exactly the rows `top_blocks` +
    /// `expand_into` produce, over a sweep of block counts (one block, a
    /// partial stripe, exactly and either side of the 512 the shipped budget
    /// keeps, and the 65536 blocks of a full-context stripe), keep counts
    /// (one, half, the shipped 512, everything), tail lengths, and scores with
    /// many exact ties — the tie rule is the load-bearing part.
    #[test]
    fn device_select_matches_host_top_blocks_bitwise() {
        let device = metal_device().unwrap();
        for &nb in &[1usize, 5, 100, 511, 512, 513, 2000, 65536] {
            let mut keeps = vec![1, nb / 2, 512, nb];
            keeps.retain(|&k| k >= 1 && k <= nb);
            keeps.dedup();
            for &keep in &keeps {
                let ix = selector(keep * RATIO, RATIO, &device);
                for tail in 0..RATIO {
                    let pos = nb * RATIO + tail - 1;
                    for (kind, scores) in [
                        ("tied", tied_scores(0x600 + nb as u64, nb)),
                        ("continuous", rand(0x700 + nb as u64, nb, 0.0, 4.0)),
                    ] {
                        let want = host_rows(&ix, &scores, nb, pos);
                        let got = device_rows(&scores, keep, RATIO, tail, &device);
                        assert_eq!(want.len(), keep * RATIO + tail);
                        assert_eq!(got, want, "nb {nb} keep {keep} tail {tail} {kind}");
                    }
                }
            }
        }
    }

    /// Scores outside the contract — NaN, negative — still select the same
    /// rows on both arms, because both rank through one `score_key` (a NaN
    /// or a negative keys as 0, a tie with every zero-scored block).
    #[test]
    fn device_select_matches_host_on_nan_and_negative_scores() {
        let device = metal_device().unwrap();
        let nb = 3000;
        let mut scores = rand(0x900, nb, -2.0, 2.0);
        for (i, s) in scores.iter_mut().enumerate() {
            match i % 7 {
                0 => *s = f32::NAN,
                1 => *s = -f32::NAN,
                2 => *s = 0.0,
                3 => *s = -0.0,
                _ => {}
            }
        }
        for &keep in &[1usize, 700, 1500, 2999] {
            let ix = selector(keep * RATIO, RATIO, &device);
            let want = host_rows(&ix, &scores, nb, nb * RATIO + 1);
            let got = device_rows(&scores, keep, RATIO, 2, &device);
            assert_eq!(got, want, "keep {keep}");
        }
    }

    /// The equal-to-threshold quota spans many threads' stripes: every score
    /// equal, so the threshold is that value and `need_eq` is the whole keep,
    /// which the ranks scanned across the threadgroup must hand to the LOWEST
    /// block indices; and a two-value case where the quota is filled from a
    /// tie scattered across the stripes above a band that is kept outright.
    #[test]
    fn device_select_tie_quota_spans_stripes() {
        let device = metal_device().unwrap();
        // 4096 blocks over 1024 threads: four per stripe, the 2000 kept
        // blocks come out of 500 stripes.
        let nb = 4096;
        let flat = vec![1.0f32; nb];
        let ix = selector(2000 * RATIO, RATIO, &device);
        let got = device_rows(&flat, 2000, RATIO, 2, &device);
        assert_eq!(got, host_rows(&ix, &flat, nb, nb * RATIO + 1));
        let want: Vec<u32> = (0..2000 * RATIO as u32).chain([16384, 16385]).collect();
        assert_eq!(
            got, want,
            "all-equal: the lowest 2000 blocks, then the tail"
        );

        // All zero at the widest stripe: 65536 blocks, keep 512 = 8 stripes.
        let nb = 65536;
        let zeros = vec![0.0f32; nb];
        let ix = selector(512 * RATIO, RATIO, &device);
        let got = device_rows(&zeros, 512, RATIO, 0, &device);
        assert_eq!(got, host_rows(&ix, &zeros, nb, nb * RATIO - 1));
        assert_eq!(got, (0..512 * RATIO as u32).collect::<Vec<u32>>());

        // 3000 blocks at 2.0 scattered among 1.0s, keep 3500: every 2.0 and
        // the first 500 of the 1.0s, in index order.
        let nb = 8000;
        let u = rand(0x800, nb, 0.0, 1.0);
        let mut two = 0;
        let mixed: Vec<f32> = u
            .iter()
            .map(|&x| {
                if x < 0.375 && two < 3000 {
                    two += 1;
                    2.0
                } else {
                    1.0
                }
            })
            .collect();
        assert_eq!(two, 3000);
        let ix = selector(3500 * RATIO, RATIO, &device);
        let got = device_rows(&mixed, 3500, RATIO, 3, &device);
        assert_eq!(got, host_rows(&ix, &mixed, nb, nb * RATIO + 2));
        let mut ones_kept = 0;
        let blocks: Vec<usize> = got[..3500 * RATIO]
            .chunks(RATIO)
            .map(|c| c[0] as usize / RATIO)
            .collect();
        for &b in &blocks {
            if mixed[b] == 1.0 {
                ones_kept += 1;
            }
        }
        assert_eq!(ones_kept, 500);
        let first_one_rejected = (0..nb).filter(|&b| mixed[b] == 1.0).nth(500).unwrap();
        assert!(
            blocks
                .iter()
                .all(|&b| mixed[b] == 2.0 || b < first_one_rejected),
            "the kept 1.0 blocks are the 500 lowest-indexed ones"
        );
    }

    /// `strided_sum` reproduces candle's strided reduce bit for bit at the two
    /// extents the indexer sums over — the block ratio and the head count —
    /// and so does the pool it feeds against the `mean(1)` it replaced.
    #[test]
    fn strided_sum_matches_candle_reduce_bitwise() {
        let device = metal_device().unwrap();
        let runs = Tensor::from_vec(
            rand(0x530, 300 * RATIO * HEAD_DIM, -3.0, 3.0),
            (300, RATIO, HEAD_DIM),
            &device,
        )
        .unwrap();
        let want = runs.mean(1).unwrap();
        let got = (strided_sum(&runs, 1).unwrap() * (1.0 / RATIO as f64)).unwrap();
        let (w, g) = (bits(&want), bits(&got));
        let diff = w.iter().zip(&g).filter(|(a, b)| a != b).count();
        assert_eq!(
            diff,
            0,
            "pool: {diff} of {} elements differ from mean(1)",
            w.len()
        );

        let per_head = Tensor::from_vec(
            rand(0x531, N_HEADS * 7 * 600, 0.0, 5.0),
            (N_HEADS, 7, 600),
            &device,
        )
        .unwrap();
        let want = per_head.sum(0).unwrap();
        let got = strided_sum(&per_head, 0).unwrap();
        let (w, g) = (bits(&want), bits(&got));
        let diff = w.iter().zip(&g).filter(|(a, b)| a != b).count();
        assert_eq!(
            diff,
            0,
            "scores: {diff} of {} elements differ from sum(0)",
            w.len()
        );

        // The other extents a one- or two-thread reduce covers.
        for n in [1usize, 2, 3, 5] {
            let t = Tensor::from_vec(rand(0x540 + n as u64, n * 33, -1.0, 1.0), (n, 33), &device)
                .unwrap();
            assert_eq!(
                bits(&t.sum(0).unwrap()),
                bits(&strided_sum(&t, 0).unwrap()),
                "extent {n}"
            );
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
