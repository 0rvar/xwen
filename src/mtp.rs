//! The Qwen 3.8 MTP (multi-token prediction) draft head: one extra
//! trunk-flavour full-attention layer that predicts the token after next.
//!
//! Ground truth is llama.cpp master's `src/models/qwen35.cpp` `graph_mtp` (the
//! pinned clone under reference/llama.cpp carries the byte-identical graph), and
//! the chain semantics are `common/speculative.cpp`'s MTP path. The head is
//! shipped as a sidecar GGUF of 18 tensors — `blk.64.*` plus `output_norm` and
//! BF16 duplicates of the target's `token_embd`/`output`, which this module
//! deliberately ignores: the design reuses the TARGET's quantized embedding and
//! lm_head, both because they are the weights the trunk actually trained with
//! and because the duplicates would cost 3 GB of the sidecar's 3.16 for nothing.
//!
//! Two things about the graph are easy to get backwards, so both are pinned by
//! tests in this module rather than by comment alone:
//!
//! 1. **Concat order is EMBEDDING FIRST.** `eh_proj` consumes
//!    `[norm(embed) ⊕ norm(hidden)]` with the embedding in the low half. Swapping
//!    the halves produces a graph that runs, trains nothing, and drafts noise.
//! 2. **Both residuals anchor on `eh_proj`'s OUTPUT.** There is no outer residual
//!    that re-adds the embedding or the incoming hidden. `inpSA = eh_proj(cat)`,
//!    and attention and FFN both add back to that.
//!
//! A third detail worth stating because it is invisible in the tensor names: the
//! `h` input is the target's hidden AFTER `output_norm` — the post-final-norm
//! tensor that the trunk otherwise feeds straight to its lm_head (upstream
//! commit 166fe294 made that choice deliberately). The existing DFlash spec taps
//! are PRE-norm layer outputs and are the wrong source; `XwenModel` grows a
//! separate accessor for this one.
//!
//! The attention block itself is the trunk's [`AttnBlock`] over the trunk's
//! [`Rope`] and a [`LayerCache`], not a re-derivation: the sidecar's attention
//! tensors are trunk-shaped (`attn_q` double-width per-head `[q, gate]`,
//! QK-RMSNorm over `[256]`, 24 Q heads over 4 KV heads), so reusing the blessed
//! block is both less code and the only way to be sure the partial NEoX rope,
//! the QK-norm and the sigmoid output gate match what the target does.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use candle_core::{Device, Module, Tensor};
use candle_nn::RmsNorm;

use crate::attention::{AttnBlock, AttnWeights};
use crate::config::{LayerKind, XwenConfig};
use crate::dflash::{DrafterImage, DrafterImageKind, f32_bytes, f32_tensor};
use crate::gguf::{GgufFile, QLinear, Weights};
use crate::kv_cache::LayerCache;
use crate::rope::Rope;

/// What the sidecar declares about itself, and what the target must agree with.
///
/// Deliberately thin: the head IS a trunk layer, so every shape it needs already
/// lives in the target's [`XwenConfig`]. What is read here is only what could
/// disagree — which is what [`MtpConfig::check_against_target`] exists to catch.
#[derive(Debug, Clone)]
pub struct MtpConfig {
    /// `<arch>.block_count` in the sidecar: 65, i.e. the trunk's 64 plus this one.
    pub block_count: usize,
    /// `<arch>.nextn_predict_layers`: how many MTP layers the file carries. One,
    /// on every published Qwen 3.8 sidecar; more would be a different design.
    pub nextn_predict_layers: usize,
    pub hidden: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub rms_eps: f64,
}

impl MtpConfig {
    pub fn from_gguf(content: &candle_core::quantized::gguf_file::Content) -> Result<Self> {
        let arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .context("the drafter GGUF declares no general.architecture")?;
        ensure!(
            arch == "qwen35",
            "expected an MTP sidecar for the dense arch (\"qwen35\"), got {arch:?}"
        );
        let usize_at = |key: &str| -> Result<usize> {
            let v = content
                .metadata
                .get(key)
                .with_context(|| format!("the MTP sidecar is missing {key}"))?;
            let n = v
                .to_u32()
                .map(|n| n as usize)
                .or_else(|_| v.to_u64().map(|n| n as usize))
                .with_context(|| format!("{key} is not an integer"))?;
            Ok(n)
        };
        let f32_at = |key: &str, default: f32| -> f32 {
            content
                .metadata
                .get(key)
                .and_then(|v| v.to_f32().ok())
                .unwrap_or(default)
        };
        Ok(Self {
            block_count: usize_at("qwen35.block_count")?,
            nextn_predict_layers: usize_at("qwen35.nextn_predict_layers")?,
            hidden: usize_at("qwen35.embedding_length")?,
            n_head: usize_at("qwen35.attention.head_count")?,
            n_kv_head: usize_at("qwen35.attention.head_count_kv")?,
            head_dim: usize_at("qwen35.attention.key_length")?,
            n_ff: usize_at("qwen35.feed_forward_length")?,
            rms_eps: f32_at("qwen35.attention.layer_norm_rms_epsilon", 1e-6) as f64,
        })
    }

    /// Whether this sidecar can serve `target`. The head reuses the target's
    /// embedding and lm_head and appends a layer to its stack, so every shape it
    /// touches has to match exactly — a mismatch here would otherwise surface as
    /// a shape error inside the first draft round, behind the job boundary.
    pub fn check_against_target(&self, target: &XwenConfig) -> Result<()> {
        ensure!(
            self.nextn_predict_layers == 1,
            "this MTP sidecar declares {} predict layers; xwen implements the 1-layer head",
            self.nextn_predict_layers
        );
        ensure!(
            self.block_count == target.n_layer + 1,
            "the MTP sidecar declares {} blocks for a {}-layer target; the head is the trunk's \
             layers plus one",
            self.block_count,
            target.n_layer
        );
        // The head's own norms use the sidecar's eps while the attention block
        // it borrows uses the target's, so a disagreement would silently run the
        // one graph at two epsilons. Compared exactly: both come from a GGUF key
        // that either matches or was built for another checkpoint, and a
        // tolerance here would only decide how wrong is acceptable.
        ensure!(
            self.rms_eps == target.rms_eps,
            "the MTP sidecar's RMSNorm epsilon is {} but the target's is {}: this sidecar was \
             built for a different checkpoint",
            self.rms_eps,
            target.rms_eps
        );
        for (what, got, want) in [
            ("hidden size", self.hidden, target.hidden),
            ("head count", self.n_head, target.n_head(0)),
            ("KV head count", self.n_kv_head, target.n_kv_head),
            ("head dim", self.head_dim, target.head_dim),
            ("FFN width", self.n_ff, target.dense_ff),
        ] {
            ensure!(
                got == want,
                "the MTP sidecar's {what} is {got} but the target's is {want}: this sidecar was \
                 built for a different checkpoint"
            );
        }
        Ok(())
    }
}

/// The loaded head, with its own KV cache for its single layer.
pub struct MtpDrafter {
    cfg: MtpConfig,
    /// `nextn.enorm` / `nextn.hnorm`: the two input norms, applied BEFORE the
    /// concat, one to the embedding and one to the target hidden.
    enorm: RmsNorm,
    hnorm: RmsNorm,
    /// `nextn.eh_proj` `[10240, 5120]`: the fused projection whose output is the
    /// residual anchor for the whole block.
    eh_proj: QLinear,
    attn_norm: RmsNorm,
    attn: AttnBlock,
    post_attention_norm: RmsNorm,
    ffn_gate: QLinear,
    ffn_up: QLinear,
    ffn_down: QLinear,
    /// `nextn.shared_head_norm`: the final norm before the lm_head. Present on
    /// every published sidecar and it WINS over `output_norm`, which the file
    /// also carries and which nothing here reads.
    shared_head_norm: RmsNorm,
    /// The head's own KV, one full-attention layer's worth. Rows `0..committed`
    /// are the synced context; a draft chain appends past them and is rolled
    /// back by lowering the length, which is why rollback costs nothing.
    cache: LayerCache,
    committed: usize,
    max_ctx: usize,
    /// The shift-right-by-one carry: the target's post-final-norm hidden at
    /// position `committed - 1`, which is what the row at `committed` will be
    /// built from. Zeros at `committed == 0` — llama.cpp seeds its `pending_h`
    /// the same way, and the head's first row genuinely has no previous
    /// position to read.
    ///
    /// Held here rather than at the call sites because it is the one piece of
    /// state the shift rule can be got wrong with, and there are three places
    /// that sync (prefill, the decode round's plain step, the decode round's
    /// verify).
    carry: Tensor,
    device: Device,
}

impl MtpDrafter {
    /// Load the sidecar's single block. `target` is the checkpoint this head will
    /// draft for: its config supplies every geometry the head shares with the
    /// trunk, and `check_against_target` has already refused a sidecar that
    /// disagrees.
    pub fn load(
        gguf: &Arc<GgufFile>,
        target: &XwenConfig,
        device: &Device,
        max_ctx: usize,
    ) -> Result<Self> {
        ensure!(
            max_ctx >= 1,
            "the MTP drafter needs at least one cache slot"
        );
        let cfg = MtpConfig::from_gguf(&gguf.content)?;
        cfg.check_against_target(target)?;

        // The head is stored under the block index AFTER the trunk's last: a
        // 64-layer trunk puts it at `blk.64`. Derived from the target rather
        // than pinned to a constant so a sidecar built for a different-depth
        // trunk is refused by `check_against_target` above — which names both
        // depths — instead of loading and then failing to find its tensors.
        let w = Weights::from_gguf(gguf.clone());
        let blk = w.pp(format!("blk.{}", target.n_layer));
        let norm =
            |name: &str| -> Result<RmsNorm> { Ok(RmsNorm::new(blk.dense_f32(name)?, cfg.rms_eps)) };

        // The head is a full-attention layer, so it borrows the trunk's rope and
        // the trunk's attention block wholesale rather than restating either.
        // `il` only selects head counts out of the target config, so it must name
        // a full-attention layer; the first one is 3 on both dense checkpoints.
        let il = full_attention_layer(target)?;
        let rope = Arc::new(Rope::new(target.rope(), max_ctx, device)?);
        let attn = AttnBlock::new(&blk, target, il, rope, AttnWeights::F16)?;

        Ok(Self {
            enorm: norm("nextn.enorm")?,
            hnorm: norm("nextn.hnorm")?,
            eh_proj: blk.qlinear("nextn.eh_proj")?,
            attn_norm: norm("attn_norm")?,
            attn,
            post_attention_norm: norm("post_attention_norm")?,
            ffn_gate: blk.qlinear("ffn_gate")?,
            ffn_up: blk.qlinear("ffn_up")?,
            ffn_down: blk.qlinear("ffn_down")?,
            shared_head_norm: norm("nextn.shared_head_norm")?,
            cache: LayerCache::new(target, il, max_ctx.min(1024), device)?,
            committed: 0,
            max_ctx,
            carry: Tensor::zeros((1, cfg.hidden), candle_core::DType::F32, device)?,
            device: device.clone(),
            cfg,
        })
    }

    pub fn config(&self) -> &MtpConfig {
        &self.cfg
    }

    /// Positions the head's cache holds. Never advances during a draft chain —
    /// see [`MtpDrafter::draft_rollback`].
    pub fn committed_len(&self) -> usize {
        self.committed
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// The longest chain this head could run in one round.
    ///
    /// Unlike a block drafter this is not a property of the weights: there is no
    /// structural bound at all, since every step is just another forward of the
    /// same layer. So this is a sanity ceiling and nothing more — what a round
    /// actually asks for is `--draft-max`, whose default is fitted per
    /// checkpoint on [`crate::hub::Model::draft_max_default`] precisely because
    /// the two kinds have opposite economics and read the same flag.
    ///
    /// Kept as a ceiling anyway because a chain that deep is drafting into noise
    /// whatever the flag says: step k compounds k-1 of its own guesses, so
    /// acceptance falls off a cliff long before this, and a mistyped
    /// `--draft-max 500` should cost a bad round rather than five hundred
    /// forwards.
    pub fn max_chain_len(&self) -> usize {
        16
    }

    /// The hidden the next synced row will be built from: the target's
    /// post-final-norm hidden at position `committed - 1`, `[1, hidden]` f32.
    ///
    /// A draft chain's FIRST step reads exactly this — the chain starts at
    /// position `committed`, whose row pairs the token the target just sampled
    /// with the hidden of the position before it.
    pub(crate) fn carry(&self) -> &Tensor {
        &self.carry
    }

    /// Give context back after a rewind — a conversation resumed from a shorter
    /// prefix, not a partial accept (that is [`MtpDrafter::draft_rollback`],
    /// which never moves `committed`).
    ///
    /// Truncating to anything SHORTER than the head holds drops the head
    /// entirely, which is a real difference from the DFlash drafter's cheap
    /// length cut. The row at position `len` is built from the target's hidden
    /// at `len - 1`, and the head keeps exactly one such hidden — the one for
    /// the position it currently ends at. A rewind discards it, and nothing
    /// short of re-running the target at `len - 1` can produce it again, so the
    /// head cannot resume at `len`: it resets and rebuilds on the next prefill
    /// from zero. Syncing at `len` with the carry the rewind left behind would
    /// poison that row with another position's hidden instead, which is a wrong
    /// draft context that nothing later repairs.
    ///
    /// Speculation is off in the meantime, not incorrect: every path gates on
    /// the head's committed length matching the target's, and 0 matches only a
    /// prefill from the start. See the ledger item in TODO.md.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        ensure!(
            len <= self.committed,
            "truncate: cannot grow the MTP context from {} to {len}",
            self.committed
        );
        if len == self.committed {
            return Ok(());
        }
        self.reset()
    }

    /// Drop the head's context entirely, so it resynchronizes from position 0.
    ///
    /// Allocation FIRST, mutation after, and the order is load-bearing: the
    /// zero carry is a device allocation and can fail. Clearing the cache before
    /// it would leave a head that reports zero committed positions while still
    /// holding the previous conversation's carry, and the next row 0 would then
    /// be built from a hidden belonging to somebody else's text — a silently
    /// poisoned draft context rather than a visible failure. Failing with
    /// everything untouched is the only state a caller can reason about.
    pub fn reset(&mut self) -> Result<()> {
        let carry = Tensor::zeros((1, self.cfg.hidden), candle_core::DType::F32, &self.device)?;
        self.cache.reset()?;
        self.committed = 0;
        self.carry = carry;
        Ok(())
    }

    /// One MTP step, the pinned graph exactly.
    ///
    /// `embed` is the TARGET's embedding of the token at these positions and `h`
    /// the TARGET's post-final-norm hidden of the position BEFORE each of them
    /// (the shift-right-by-one the sync rule describes); both `[seq, hidden]`
    /// f32. `pos` is the absolute position of the first row, which is what the
    /// trunk's rope is given.
    ///
    /// Returns the post-`shared_head_norm` hidden `[seq, hidden]` — what the
    /// lm_head consumes, and equally what the next chain step carries forward.
    ///
    /// Writes `seq` rows into the head's KV cache and does NOT advance
    /// `committed`: callers either commit them ([`MtpDrafter::sync`]) or roll
    /// them back ([`MtpDrafter::draft_rollback`]).
    pub fn step(&mut self, embed: &Tensor, h: &Tensor, pos: usize) -> Result<Tensor> {
        let (seq, hidden) = embed.dims2()?;
        let (h_seq, h_hidden) = h.dims2()?;
        ensure!(
            seq == h_seq && hidden == h_hidden,
            "step: embedding is [{seq}, {hidden}] but the hidden is [{h_seq}, {h_hidden}]"
        );
        ensure!(
            hidden == self.cfg.hidden,
            "step: hidden size {hidden} is not the head's {}",
            self.cfg.hidden
        );
        ensure!(
            pos + seq <= self.max_ctx,
            "step: positions {pos}..{} exceed the head's context {}",
            pos + seq,
            self.max_ctx
        );

        // The cache is allocated small and grown on demand, exactly as the
        // trunk's is (`XwenModel::grow_kv_capacity`): `max_ctx` is a ceiling, and
        // a head sized eagerly at it would cost its whole cache on every load
        // for conversations that never reach a tenth of it. Growth is O(log) per
        // lifetime and idempotent, so this is a length check on all but a
        // handful of steps.
        self.cache.ensure_full_capacity(pos + seq, self.max_ctx)?;

        // (1+2) the input fusion, whose output is the residual anchor for BOTH
        // residuals below; nothing re-adds `embed` or `h`.
        let inp_sa = self.fuse(embed, h)?;

        // (3) the trunk's own attention block, over the head's own cache.
        let mask = self.attn.prefill_mask(&self.cache, seq, pos)?;
        let attn_out = self.attn.forward(
            &self.attn_norm.forward(&inp_sa)?,
            &mut self.cache,
            pos,
            mask.as_ref(),
        )?;
        let x = (&inp_sa + attn_out)?;

        // (4) SwiGLU FFN, residual back onto the same anchor lineage.
        let normed = self.post_attention_norm.forward(&x)?;
        let gate = self.ffn_gate.forward(&normed)?;
        let up = self.ffn_up.forward(&normed)?;
        let ffn = self
            .ffn_down
            .forward(&(candle_nn::ops::silu(&gate)? * up)?)?;
        let x = (x + ffn)?;

        // (5) the shared head norm — NOT `output_norm`, which the sidecar also
        // ships and which this head never reads.
        Ok(self.shared_head_norm.forward(&x)?)
    }

    /// The head's input fusion: norm each input, concatenate them with the
    /// EMBEDDING IN THE LOW HALF, and project `[seq, 2*hidden] -> [seq, hidden]`.
    ///
    /// Split out of [`MtpDrafter::step`] because the concat order is one of this
    /// graph's two silent traps: reversed, every shape still checks out and the
    /// head drafts noise. `the_concat_puts_the_embedding_in_the_low_half` pins it
    /// against `eh_proj`'s own column blocks.
    fn fuse(&self, embed: &Tensor, h: &Tensor) -> Result<Tensor> {
        let e = self.enorm.forward(embed)?;
        let hn = self.hnorm.forward(h)?;
        let cat = Tensor::cat(&[&e, &hn], 1)?.contiguous()?;
        Ok(self.eh_proj.forward(&cat)?)
    }

    /// Commit `seq` synced rows: the catch-up that keeps the head's KV aligned
    /// with the target's accepted context.
    ///
    /// This is the shift-right-by-one rule, and it is the whole reason the rule
    /// is not spelled out at any call site. `embed` is the target's embedding of
    /// the tokens AT positions `[pos, pos+seq)` and `hiddens` the target's
    /// post-final-norm hiddens at those SAME positions — both straight out of
    /// the forward that just ran, neither shifted by the caller. The head's row
    /// for position `p` is built from `(token_p, hidden_{p-1})`, so the shift
    /// happens here: row `pos` takes the carry (the hidden at `pos-1`, zeros at
    /// position 0), rows past it take `hiddens` one row back, and the carry
    /// becomes this batch's last hidden.
    ///
    /// Returns the post-`shared_head_norm` hidden of the synced rows, which the
    /// decode loops discard — a chain reads its own steps' outputs, not these.
    pub fn sync(&mut self, embed: &Tensor, hiddens: &Tensor, pos: usize) -> Result<Tensor> {
        ensure!(
            pos == self.committed,
            "sync: the head holds {} positions but the batch starts at {pos}",
            self.committed
        );
        let (seq, _) = embed.dims2()?;
        let (h_seq, h_hidden) = hiddens.dims2()?;
        ensure!(
            seq == h_seq,
            "sync: {seq} embedding rows but {h_seq} hidden rows"
        );
        ensure!(
            h_hidden == self.cfg.hidden,
            "sync: hidden size {h_hidden} is not the head's {}",
            self.cfg.hidden
        );
        ensure!(seq >= 1, "sync: nothing to sync");

        // [h_{pos-1}, h_pos, ..., h_{pos+seq-2}] — the carry in front, the
        // batch's own last hidden dropped off the end (it belongs to the row
        // AFTER this batch, which is what the carry becomes).
        let shifted = if seq == 1 {
            self.carry.clone()
        } else {
            Tensor::cat(&[&self.carry, &hiddens.narrow(0, 0, seq - 1)?], 0)?.contiguous()?
        };
        let out = self.step(embed, &shifted, pos)?;
        self.committed += seq;
        self.carry = hiddens.narrow(0, seq - 1, 1)?.contiguous()?;
        Ok(out)
    }

    /// Discard whatever a draft chain appended, back to the committed context.
    pub fn draft_rollback(&mut self) -> Result<()> {
        self.set_len(self.committed)
    }

    /// Copy the committed rows `[0, committed)` out to host RAM so a
    /// conversation can be parked and the head's cache handed to another one.
    ///
    /// The carry travels with the rows, and it has to: the head resumes by
    /// syncing the row at `committed`, which is built from the hidden at
    /// `committed - 1`, so an image without it would restore a head that holds
    /// the right KV and still cannot take another token. That is the same
    /// coupling [`MtpDrafter::truncate`] documents, seen from the other side.
    pub fn export_cache(&self) -> Result<DrafterImage> {
        let (k, v) = self.cache.export_full_rows()?;
        let carry = f32_bytes(&self.carry)?;
        DrafterImage::new_mtp(
            self.committed,
            self.cfg.n_kv_head,
            self.cfg.head_dim,
            self.cfg.hidden,
            vec![(k, v)],
            carry,
        )
    }

    /// Upload `image`'s rows back into the head's cache and restore its carry,
    /// so the next sync continues at `pos`.
    ///
    /// Unlike the DFlash drafter's import this takes the image WHOLE or not at
    /// all: a shorter prefix would need the hidden at `pos - 1`, and the image
    /// carries exactly one hidden — the one for the position it ends at. A
    /// caller resuming somewhere earlier is in the same position as one that
    /// rewound a live head, and gets the same answer.
    pub fn import_cache(&mut self, image: &DrafterImage, pos: usize) -> Result<()> {
        ensure!(
            image.kind() == DrafterImageKind::Mtp,
            "this image was written by a {} drafter; the MTP head cannot read it",
            image.kind().name()
        );
        ensure!(
            pos == image.pos,
            "the MTP head resumes only at the position its image ends at ({}), not {pos}: the \
             row at {pos} needs the target hidden at {}, which the image does not carry",
            image.pos,
            pos.saturating_sub(1)
        );
        ensure!(
            image.layers() == 1,
            "the MTP image covers {} layers; the head has one",
            image.layers()
        );
        // The image's own declared geometry, checked before its bytes are
        // believed. Byte length alone does not settle this: (8, 128) and
        // (4, 256) describe the same number of values, so a DFlash-shaped image
        // that reached here would load as a plausible-looking head instead of
        // being refused. The kind check above already stops the shipped pair,
        // and this stops the next one.
        let (n_kv, hd) = (self.cfg.n_kv_head, self.cfg.head_dim);
        ensure!(
            image.n_kv_head() == n_kv && image.head_dim() == hd,
            "the MTP image is ({}, _, {}) but the head's cache is ({n_kv}, _, {hd})",
            image.n_kv_head(),
            image.head_dim()
        );
        ensure!(
            pos <= self.max_ctx,
            "the MTP image's {pos} positions exceed the head's {} cache slots",
            self.max_ctx
        );
        // The cache is allocated lazily and grown on demand (see `step`), so a
        // resume deeper than the current allocation has to grow it first —
        // `import_full_rows` refuses a `pos` past the slots it finds, and the
        // whole point of the disk tier is resuming conversations longer than the
        // initial allocation.
        self.cache.ensure_full_capacity(pos, self.max_ctx)?;
        // The carry is built BEFORE the rows are written, for the reason
        // `reset` documents: a failure between the two would leave the head
        // holding this image's KV under the previous conversation's carry.
        let carry = f32_tensor(image.carry(), &[1, self.cfg.hidden], &self.device)?;
        let (k, v) = image.plane(0)?;
        self.cache.import_full_rows(k, v, pos)?;
        self.carry = carry;
        self.committed = pos;
        Ok(())
    }

    /// Lower the cache's length. `LayerCache::Full` truncation is exactly this —
    /// the rejected rows are overwritten by the next append.
    fn set_len(&mut self, len: usize) -> Result<()> {
        let held = self.cache.len();
        ensure!(len <= held, "set_len: cannot grow {held} to {len}");
        if len == held {
            return Ok(());
        }
        let span = held - len;
        let ckpt = self.cache.checkpoint(0)?;
        self.cache.rollback(&ckpt, len, span, 0)
    }
}

/// The index of a full-attention layer in `cfg`, which is what the head's
/// geometry is read from (its own layer is not in the target's stack).
fn full_attention_layer(cfg: &XwenConfig) -> Result<usize> {
    (0..cfg.n_layer)
        .find(|il| cfg.layer_kind(*il) == LayerKind::Full)
        .context("the target has no full-attention layer for the MTP head to mirror")
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::{GgmlDType, QTensor};

    /// The fusion half of the head — the two input norms and `eh_proj` — built
    /// from weights a test chooses, with no GGUF and no Metal device needed.
    /// Enough to interrogate the concat order, which is the whole point.
    struct Fusion {
        enorm: RmsNorm,
        hnorm: RmsNorm,
        eh_proj: QLinear,
    }

    impl Fusion {
        /// `eh` is the `[hidden, 2*hidden]` projection weight, in GGUF layout
        /// (`[out, in]`), so its COLUMNS are the concat's components.
        fn new(hidden: usize, eh: Tensor, device: &Device) -> Result<Self> {
            let ones = Tensor::ones(hidden, candle_core::DType::F32, device)?;
            let qt = QTensor::quantize(&eh, GgmlDType::F32)?;
            Ok(Self {
                // Identity norms: the trap under test is the concat order, and a
                // scaling norm would only make the assertions noisier.
                enorm: RmsNorm::new(ones.clone(), 1e-6),
                hnorm: RmsNorm::new(ones, 1e-6),
                eh_proj: QLinear::from_qtensor(Arc::new(qt))?,
            })
        }

        /// Byte-for-byte the fusion in `MtpDrafter::fuse`.
        fn fuse(&self, embed: &Tensor, h: &Tensor) -> Result<Tensor> {
            let e = self.enorm.forward(embed)?;
            let hn = self.hnorm.forward(h)?;
            let cat = Tensor::cat(&[&e, &hn], 1)?.contiguous()?;
            Ok(self.eh_proj.forward(&cat)?)
        }
    }

    fn device() -> Device {
        Device::Cpu
    }

    /// The dense 27B's shape, as much of it as the head's checks read. Both
    /// dense checkpoints share it — 3.8's config is byte-identical to 3.6's —
    /// which is exactly why the sidecar's own declarations have to be checked.
    fn dense_27b_config() -> XwenConfig {
        use crate::config::RopeKind;
        let n_layer = 64usize;
        XwenConfig {
            arch: crate::config::Arch::Dense,
            general_name: Some("Qwen3.8-27B".to_string()),
            n_layer,
            hidden: 5120,
            vocab: 248_320,
            n_head: vec![24; n_layer],
            n_kv_head: 4,
            head_dim: 256,
            layer_kind: (0..n_layer)
                .map(|il| {
                    if (il + 1).is_multiple_of(4) {
                        LayerKind::Full
                    } else {
                        LayerKind::Linear
                    }
                })
                .collect(),
            linear_k_heads: 16,
            linear_v_heads: 48,
            linear_head_dim: 128,
            conv_kernel: 4,
            dense_ff: 17408,
            n_expert: 0,
            n_expert_used: 0,
            expert_ff: 0,
            shared_expert_ff: 0,
            rms_eps: 1e-6,
            n_ctx_train: 262_144,
            rope: RopeKind::Plain {
                freq_base: 1e7,
                n_rot: 64,
            },
            eog_tokens: vec![248_046, 248_044],
            qwen4exp: None,
        }
    }

    /// **Trap 1.** `eh_proj` consumes `[norm(embed) ⊕ norm(hidden)]` with the
    /// EMBEDDING in the low half. Reversed, every shape still checks out and the
    /// head drafts noise, so this is pinned against the weight's own column
    /// blocks rather than by reading the concat call.
    ///
    /// The instrument: a projection that is zero on its high columns can only
    /// see the concat's LOW half. If the low half is the embedding (correct),
    /// its output moves with `embed` and ignores `h`. If the halves were
    /// swapped, the same weight would ignore `embed` instead — so the two
    /// assertions below cannot both hold under a reversed concat.
    #[test]
    fn the_concat_puts_the_embedding_in_the_low_half() -> Result<()> {
        let dev = device();
        let hidden = 4usize;
        let seq = 2usize;

        // Low columns pass the first component through; high columns are dead.
        let mut w = vec![0f32; hidden * 2 * hidden];
        for i in 0..hidden {
            w[i * (2 * hidden) + i] = 1.0;
        }
        let low_only = Fusion::new(
            hidden,
            Tensor::from_vec(w, (hidden, 2 * hidden), &dev)?,
            &dev,
        )?;

        let embed_a =
            Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6., 7., 8.], (seq, hidden), &dev)?;
        let embed_b =
            Tensor::from_vec(vec![9f32, 8., 7., 6., 5., 4., 3., 2.], (seq, hidden), &dev)?;
        let h_a = Tensor::from_vec(vec![0.5f32; seq * hidden], (seq, hidden), &dev)?;
        let h_b = Tensor::from_vec(vec![-3f32; seq * hidden], (seq, hidden), &dev)?;

        let base = low_only
            .fuse(&embed_a, &h_a)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let other_h = low_only
            .fuse(&embed_a, &h_b)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let other_e = low_only
            .fuse(&embed_b, &h_a)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        assert_eq!(
            base, other_h,
            "a projection blind to the concat's HIGH half still changed when only the hidden \
             changed — the hidden is in the low half, i.e. the concat is reversed"
        );
        assert_ne!(
            base, other_e,
            "a projection reading the concat's LOW half did not change when the embedding \
             changed — the embedding is not in the low half"
        );

        // And the mirror image, so neither assertion can pass by accident: a
        // projection blind to the LOW half must track `h` and ignore `embed`.
        let mut w = vec![0f32; hidden * 2 * hidden];
        for i in 0..hidden {
            w[i * (2 * hidden) + hidden + i] = 1.0;
        }
        let high_only = Fusion::new(
            hidden,
            Tensor::from_vec(w, (hidden, 2 * hidden), &dev)?,
            &dev,
        )?;
        let base = high_only
            .fuse(&embed_a, &h_a)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let other_e = high_only
            .fuse(&embed_b, &h_a)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let other_h = high_only
            .fuse(&embed_a, &h_b)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(base, other_e, "the high half must not carry the embedding");
        assert_ne!(base, other_h, "the high half must carry the hidden");
        Ok(())
    }

    /// A tiny target config with a full-attention layer at index 0, so a
    /// synthetic head can be built at shapes a test can reason about.
    fn tiny_config(hidden: usize, n_head: usize, n_kv_head: usize, head_dim: usize) -> XwenConfig {
        use crate::config::RopeKind;
        XwenConfig {
            arch: crate::config::Arch::Dense,
            general_name: None,
            n_layer: 1,
            hidden,
            vocab: 32,
            n_head: vec![n_head],
            n_kv_head,
            head_dim,
            layer_kind: vec![LayerKind::Full],
            linear_k_heads: 1,
            linear_v_heads: 1,
            linear_head_dim: 2,
            conv_kernel: 4,
            dense_ff: 2 * hidden,
            n_expert: 0,
            n_expert_used: 0,
            expert_ff: 0,
            shared_expert_ff: 0,
            rms_eps: 1e-6,
            n_ctx_train: 64,
            rope: RopeKind::Plain {
                freq_base: 1e7,
                n_rot: head_dim.min(2),
            },
            eog_tokens: vec![0],
            qwen4exp: None,
        }
    }

    /// The synthetic head both device tests are built on: real graph, real
    /// cache, real rope, and an attention block and FFN whose output
    /// projections are ZERO, so neither contributes anything whatever it
    /// computed internally.
    ///
    /// What that leaves is the graph's skeleton — the input fusion, the two
    /// residuals onto it, and the final norm — which is exactly the part these
    /// tests are about and the part that has no oracle other than the graph
    /// itself. `eh_proj` is filled with distinct per-column values so its output
    /// is not symmetric and no assertion below can pass by coincidence.
    ///
    /// Needs Metal: the block's projections are the vendored f16 kernels, like
    /// every other ops test in this crate.
    fn inert_head(dev: &Device) -> Result<MtpDrafter> {
        inert_head_with_ctx(dev, 16)
    }

    /// [`inert_head`] at a chosen context ceiling, for the tests that need the
    /// head to grow past its initial allocation.
    fn inert_head_with_ctx(dev: &Device, max_ctx: usize) -> Result<MtpDrafter> {
        // Multiples of 32: the vendored f16 projection kernel requires k % 32 == 0,
        // so a toy shape has to respect the same constraint the real one does.
        let (hidden, n_head, n_kv_head, head_dim) = (64usize, 2usize, 1usize, 32usize);
        let cfg = tiny_config(hidden, n_head, n_kv_head, head_dim);
        let ones =
            |n: usize| -> Result<Tensor> { Ok(Tensor::ones(n, candle_core::DType::F32, &dev)?) };
        let f16 = |rows: usize, cols: usize, fill: f32| -> Result<Tensor> {
            Ok(Tensor::full(fill, (rows, cols), &dev)?.to_dtype(candle_core::DType::F16)?)
        };
        let qlin = |t: Tensor| -> Result<QLinear> {
            let qt = candle_core::quantized::QTensor::quantize(
                &t,
                candle_core::quantized::GgmlDType::F32,
            )?;
            QLinear::from_qtensor(Arc::new(qt))
        };

        // eh_proj: distinct per-column values so its output is not symmetric and
        // an accidental match cannot be a coincidence.
        let eh: Vec<f32> = (0..hidden * 2 * hidden)
            .map(|i| ((i % 7) as f32 - 3.0) / 11.0)
            .collect();
        let eh_w = Tensor::from_vec(eh, (hidden, 2 * hidden), &dev)?;

        let attn = AttnBlock::from_raw_weights(
            f16(n_head * 2 * head_dim, hidden, 0.05)?,
            f16(n_kv_head * head_dim, hidden, 0.03)?,
            f16(n_kv_head * head_dim, hidden, 0.02)?,
            // The whole point: attention contributes nothing.
            f16(hidden, n_head * head_dim, 0.0)?,
            RmsNorm::new(ones(head_dim)?, 1e-6),
            RmsNorm::new(ones(head_dim)?, 1e-6),
            Arc::new(Rope::new(cfg.rope(), max_ctx, dev)?),
            n_head,
            n_kv_head,
            head_dim,
        );

        Ok(MtpDrafter {
            cfg: MtpConfig {
                block_count: 2,
                nextn_predict_layers: 1,
                hidden,
                n_head,
                n_kv_head,
                head_dim,
                n_ff: 2 * hidden,
                rms_eps: 1e-6,
            },
            enorm: RmsNorm::new(ones(hidden)?, 1e-6),
            hnorm: RmsNorm::new(ones(hidden)?, 1e-6),
            eh_proj: qlin(eh_w.clone())?,
            attn_norm: RmsNorm::new(ones(hidden)?, 1e-6),
            attn,
            post_attention_norm: RmsNorm::new(ones(hidden)?, 1e-6),
            ffn_gate: qlin(Tensor::full(0.1f32, (2 * hidden, hidden), &dev)?)?,
            ffn_up: qlin(Tensor::full(0.1f32, (2 * hidden, hidden), &dev)?)?,
            // And the FFN contributes nothing either.
            ffn_down: qlin(Tensor::zeros(
                (hidden, 2 * hidden),
                candle_core::DType::F32,
                &dev,
            )?)?,
            shared_head_norm: RmsNorm::new(ones(hidden)?, 1e-6),
            // Deliberately the SAME lazy initial allocation the real head takes,
            // so a test that runs past it exercises the growth rather than a
            // buffer that happened to be big enough.
            cache: LayerCache::new(&cfg, 0, max_ctx.min(1024), dev)?,
            committed: 0,
            max_ctx,
            carry: Tensor::zeros((1, hidden), candle_core::DType::F32, dev)?,
            device: dev.clone(),
        })
    }

    /// **Trap 2.** Both residuals anchor on `eh_proj`'s OUTPUT — there is no
    /// outer residual that re-adds the embedding or the incoming hidden.
    ///
    /// The instrument is [`inert_head`]: with both blocks contributing nothing,
    /// what is left of a step is its residual skeleton, and the anchor is the
    /// only thing that can still be in it. If the graph were `x + attn` anchored
    /// anywhere else — on `embed`, on `h`, on a sum of them — the output would
    /// carry that term and the equality below would fail.
    #[test]
    fn both_residuals_anchor_on_the_projection_output() -> Result<()> {
        let Ok(dev) = crate::gguf::metal_device() else {
            eprintln!("skipping: no Metal device");
            return Ok(());
        };
        let mut head = inert_head(&dev)?;
        let hidden = head.cfg.hidden;

        let embed = Tensor::from_vec(
            (0..hidden)
                .map(|i| 0.1 + i as f32 * 0.05)
                .collect::<Vec<f32>>(),
            (1, hidden),
            &dev,
        )?;
        let h = Tensor::from_vec(
            (0..hidden)
                .map(|i| -0.7 + i as f32 * 0.09)
                .collect::<Vec<f32>>(),
            (1, hidden),
            &dev,
        )?;

        let got = head.step(&embed, &h, 0)?.flatten_all()?.to_vec1::<f32>()?;

        // What the anchor predicts: the fusion, normed, and nothing else.
        let anchor = head.fuse(&embed, &h)?;
        let want = head
            .shared_head_norm
            .forward(&anchor)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (g, w) in got.iter().zip(&want) {
            assert!(
                (g - w).abs() < 1e-4,
                "the step did not collapse to the normed fusion: {got:?} vs {want:?}"
            );
        }

        // And the alternatives an outer residual would have produced are all
        // measurably different, so the equality above is not vacuous.
        for (label, extra) in [("embedding", &embed), ("hidden", &h)] {
            let wrong = head
                .shared_head_norm
                .forward(&(&anchor + extra)?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let differs = got.iter().zip(&wrong).any(|(g, w)| (g - w).abs() > 1e-3);
            assert!(
                differs,
                "the step is indistinguishable from one that re-adds the {label} — the residual \
                 anchor is not pinned by this fixture"
            );
        }
        Ok(())
    }

    /// **The sync rule.** The head's row for position `p` pairs the token at `p`
    /// with the target's post-final-norm hidden at `p - 1`, and position 0 takes
    /// a zero hidden. `sync` owns that shift, so this is where it is pinned —
    /// every call site hands it unshifted inputs straight out of a forward, and
    /// none of them could be caught getting it wrong.
    ///
    /// The instrument is [`inert_head`] again: with attention and the FFN
    /// contributing nothing, a synced row's output collapses to the normed
    /// fusion of that row's own two inputs, which makes each row's hidden
    /// directly readable off the output. An unshifted sync would pair row 0 with
    /// `h0` instead of zeros, and every row after it would be off by one.
    #[test]
    fn the_sync_pairs_each_token_with_the_previous_positions_hidden() -> Result<()> {
        let Ok(dev) = crate::gguf::metal_device() else {
            eprintln!("skipping: no Metal device");
            return Ok(());
        };
        let mut head = inert_head(&dev)?;
        let hidden = head.cfg.hidden;

        // Distinct, non-proportional rows: a shift that lands on the wrong row
        // cannot produce the right numbers by symmetry.
        let row = |seed: f32| -> Result<Tensor> {
            Ok(Tensor::from_vec(
                (0..hidden)
                    .map(|i| seed + i as f32 * (0.01 + seed / 50.0))
                    .collect::<Vec<f32>>(),
                (1, hidden),
                &dev,
            )?)
        };
        let zeros = Tensor::zeros((1, hidden), candle_core::DType::F32, &dev)?;
        let (e0, e1, e2) = (row(0.3)?, row(-0.8)?, row(1.7)?);
        let (h0, h1, h2) = (row(0.5)?, row(-1.1)?, row(2.3)?);

        // What the rule predicts for a batch of three rows synced at position 0.
        let expect = |hd: &MtpDrafter, embed: &Tensor, h: &Tensor| -> Result<Vec<f32>> {
            Ok(hd
                .shared_head_norm
                .forward(&hd.fuse(embed, h)?)?
                .flatten_all()?
                .to_vec1::<f32>()?)
        };
        let want: Vec<Vec<f32>> = vec![
            expect(&head, &e0, &zeros)?,
            expect(&head, &e1, &h0)?,
            expect(&head, &e2, &h1)?,
        ];

        let embeds = Tensor::cat(&[&e0, &e1, &e2], 0)?;
        let hiddens = Tensor::cat(&[&h0, &h1, &h2], 0)?;
        let got = head.sync(&embeds, &hiddens, 0)?;
        assert_eq!(got.dims2()?, (3, hidden));
        assert_eq!(head.committed_len(), 3);

        let got = got.to_vec2::<f32>()?;
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            for (a, b) in g.iter().zip(w) {
                assert!(
                    (a - b).abs() < 1e-4,
                    "row {i} was not built from the hidden at position {}: the sync is not \
                     shifted right by one",
                    i as i64 - 1
                );
            }
        }

        // The row that would have been produced WITHOUT the shift is measurably
        // different, so the agreement above is not something any pairing would
        // have produced.
        let unshifted = expect(&head, &e0, &h0)?;
        assert!(
            got[0]
                .iter()
                .zip(&unshifted)
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "position 0 must take a ZERO hidden, not its own"
        );

        // And the carry advanced to the batch's LAST hidden, so the next sync
        // continues the same shift across the batch boundary rather than
        // restarting it.
        let e3 = row(-0.4)?;
        let h3 = row(0.9)?;
        let want_next = expect(&head, &e3, &h2)?;
        let got_next = head.sync(&e3, &h3, 3)?.flatten_all()?.to_vec1::<f32>()?;
        for (a, b) in got_next.iter().zip(&want_next) {
            assert!(
                (a - b).abs() < 1e-4,
                "the row after a batch must read the batch's last hidden: the carry did not \
                 advance"
            );
        }
        assert_eq!(head.committed_len(), 4);

        // A rewind cannot be followed: the row at the rewind point needs a
        // hidden the head no longer has, so it drops everything rather than
        // resume on the wrong one.
        head.truncate(2)?;
        assert_eq!(
            head.committed_len(),
            0,
            "a truncate to anything shorter than the head holds must reset it"
        );
        // Truncating to exactly what it holds is the no-op the decode loop's
        // rewind check performs every round, and must not reset anything.
        head.sync(&e3, &h3, 0)?;
        head.truncate(1)?;
        assert_eq!(head.committed_len(), 1);
        Ok(())
    }

    /// **The lazy allocation.** The head's cache starts at 1024 slots and grows
    /// on demand like the trunk's, and both the stepping path and the IMPORT
    /// path have to grow it. The import path is the one that hid a bug: a
    /// conversation paged back in past 1024 positions is the disk tier's whole
    /// reason to exist, and `import_full_rows` refuses a position past the slots
    /// it finds — so an image that fit the ceiling and not the allocation failed,
    /// and the conversation silently decoded plain from then on.
    ///
    /// Exported and re-imported at a position past the initial allocation, which
    /// is the shape of the case that failed. The round trip is checked too: the
    /// head is only resumable if its carry came back with its rows, and a carry
    /// that did not survive would leave the next row built from zeros.
    #[test]
    fn an_image_past_the_initial_allocation_imports() -> Result<()> {
        let Ok(dev) = crate::gguf::metal_device() else {
            eprintln!("skipping: no Metal device");
            return Ok(());
        };
        // Past the 1024-slot initial allocation, so the import has to grow.
        let pos = 1100usize;
        let mut head = inert_head_with_ctx(&dev, 2048)?;
        let hidden = head.cfg.hidden;

        let embeds = Tensor::rand(-1f32, 1f32, (pos, hidden), &dev)?;
        let hiddens = Tensor::rand(-1f32, 1f32, (pos, hidden), &dev)?;
        head.sync(&embeds, &hiddens, 0)?;
        assert_eq!(head.committed_len(), pos);
        let carry = head.carry().flatten_all()?.to_vec1::<f32>()?;

        let image = head.export_cache()?;
        assert_eq!(image.pos, pos);
        assert_eq!(image.kind(), DrafterImageKind::Mtp);

        let mut fresh = inert_head_with_ctx(&dev, 2048)?;
        fresh.import_cache(&image, pos)?;
        assert_eq!(fresh.committed_len(), pos);
        assert_eq!(
            fresh.carry().flatten_all()?.to_vec1::<f32>()?,
            carry,
            "the carry must survive the round trip, or the resumed head builds its next row \
             from zeros"
        );

        // And the resumed head takes another row, which is the thing an image
        // that imported but did not grow could not do.
        let more_e = Tensor::rand(-1f32, 1f32, (1, hidden), &dev)?;
        let more_h = Tensor::rand(-1f32, 1f32, (1, hidden), &dev)?;
        fresh.sync(&more_e, &more_h, pos)?;
        assert_eq!(fresh.committed_len(), pos + 1);

        // A resume anywhere but the image's own end is refused by name: the
        // carry it holds belongs to position `pos - 1` and to nowhere else.
        let mut short = inert_head_with_ctx(&dev, 2048)?;
        let err = short
            .import_cache(&image, pos - 1)
            .expect_err("a short resume must be refused")
            .to_string();
        assert!(err.contains("resumes only at"), "{err}");
        assert_eq!(short.committed_len(), 0, "a refused import writes nothing");
        Ok(())
    }

    /// The real sidecar loads and steps: tensor names, shapes, the `AttnBlock`
    /// reuse and the cache wiring, against the published weights rather than
    /// against a fixture.
    ///
    /// `#[ignore]`d because it needs the 3.16 GB sidecar and a Metal device —
    /// run it explicitly with
    /// `cargo test --release --lib mtp:: -- --ignored --nocapture`, pointing
    /// `XWEN_MTP_SIDECAR` at the file (Stage B gives it a hub entry and this
    /// test its cache lookup).
    #[test]
    #[ignore = "needs the MTP sidecar and a Metal device"]
    fn the_published_sidecar_loads_and_steps() -> Result<()> {
        let Ok(path) = std::env::var("XWEN_MTP_SIDECAR") else {
            panic!("set XWEN_MTP_SIDECAR to the mtp-Qwen3.8-27B-*.gguf to run this");
        };
        let target_path = std::env::var("XWEN_MTP_TARGET")
            .expect("set XWEN_MTP_TARGET to the Qwen3.8-27B target GGUF");
        let dev = crate::gguf::metal_device()?;

        // The target's config only — metadata, no weights.
        let target_gguf = crate::gguf::open(&target_path, &Device::Cpu)?;
        let target = XwenConfig::from_gguf(&target_gguf.content)?;

        let gguf = crate::gguf::open(&path, &dev)?;
        let mut head = MtpDrafter::load(&gguf, &target, &dev, 64)?;
        assert_eq!(head.config().block_count, target.n_layer + 1);
        assert_eq!(head.config().nextn_predict_layers, 1);
        assert_eq!(head.committed_len(), 0);

        let hidden = target.hidden;
        let row = |v: f32| Tensor::full(v, (1, hidden), &dev).map_err(anyhow::Error::from);

        // Sync two rows — the hiddens are the target's at those same positions,
        // unshifted, exactly as a forward hands them over — then draft one step
        // past them: the shapes and the committed/scratch discipline in one pass.
        let embeds = Tensor::cat(&[&row(0.02)?, &row(-0.01)?], 0)?;
        let hiddens = Tensor::cat(&[&row(0.03)?, &row(0.04)?], 0)?;
        let synced = head.sync(&embeds, &hiddens, 0)?;
        assert_eq!(synced.dims2()?, (2, hidden));
        assert_eq!(head.committed_len(), 2);

        let out = head.step(&row(0.01)?, &row(0.02)?, 2)?;
        assert_eq!(out.dims2()?, (1, hidden));
        let finite = out.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            finite.iter().all(|v| v.is_finite()),
            "the head produced non-finite values"
        );
        // Drafting must not move the committed context, and the rollback must
        // put the cache back where the sync left it.
        assert_eq!(head.committed_len(), 2);
        head.draft_rollback()?;
        assert_eq!(head.committed_len(), 2);

        // A rewind the head cannot follow drops its context rather than resume
        // on a hidden that belongs to another position.
        head.truncate(1)?;
        assert_eq!(head.committed_len(), 0);
        head.reset()?;
        assert_eq!(head.committed_len(), 0);
        Ok(())
    }

    /// The sidecar's declared geometry has to match the checkpoint it will draft
    /// for, and every mismatch names both sides — these fire at load, not inside
    /// the first draft round.
    #[test]
    fn a_sidecar_for_another_checkpoint_is_refused() {
        let target = dense_27b_config();
        let ok = MtpConfig {
            block_count: target.n_layer + 1,
            nextn_predict_layers: 1,
            hidden: target.hidden,
            n_head: target.n_head(0),
            n_kv_head: target.n_kv_head,
            head_dim: target.head_dim,
            n_ff: target.dense_ff,
            rms_eps: target.rms_eps,
        };
        ok.check_against_target(&target)
            .expect("the matching sidecar loads");

        let cases: [(&str, MtpConfig); 5] = [
            (
                "hidden size",
                MtpConfig {
                    hidden: 2048,
                    ..ok.clone()
                },
            ),
            (
                "head count",
                MtpConfig {
                    n_head: 16,
                    ..ok.clone()
                },
            ),
            (
                "KV head count",
                MtpConfig {
                    n_kv_head: 2,
                    ..ok.clone()
                },
            ),
            (
                "head dim",
                MtpConfig {
                    head_dim: 128,
                    ..ok.clone()
                },
            ),
            (
                "FFN width",
                MtpConfig {
                    n_ff: 4096,
                    ..ok.clone()
                },
            ),
        ];
        for (what, cfg) in cases {
            let err = cfg
                .check_against_target(&target)
                .expect_err("a mismatched sidecar is refused")
                .to_string();
            assert!(err.contains(what), "{err}");
            assert!(err.contains("different checkpoint"), "{err}");
        }

        // A multi-layer MTP head is a different design, not a tunable.
        let err = MtpConfig {
            nextn_predict_layers: 2,
            ..ok.clone()
        }
        .check_against_target(&target)
        .expect_err("a 2-layer head is refused")
        .to_string();
        assert!(err.contains("1-layer head"), "{err}");

        // The block count must be the trunk's layers plus this one.
        let err = MtpConfig {
            block_count: target.n_layer,
            ..ok
        }
        .check_against_target(&target)
        .expect_err("a sidecar that is not the trunk + 1 is refused")
        .to_string();
        assert!(err.contains("plus one"), "{err}");
    }
}
