//! The qwen4exp forward graph: the hyper-connection carrier, the QSA overlay
//! and the PLE injection, assembled around the SAME attention, DeltaNet and MoE
//! blocks the qwen35 stack runs (D14).
//!
//! What makes this a second stack rather than a branch inside
//! [`XwenModel::run_stack`] is the residual: qwen4exp carries `hc_count`
//! parallel streams concatenated into one `hc_count * hidden` row and has no
//! `attn_norm` / `post_attention_norm` / `output_norm` tensors at all. Every
//! block reads a `hidden`-wide vector out of that carrier through its own gate
//! and writes its output back into all streams; the tail collapses the carrier
//! with one more read. Everything else — the blocks, the KV caches, the taps,
//! the profiler stages, the lm head — is the trunk's, unchanged.
//!
//! Layer order, mirroring `reference/llama.cpp/src/models/qwen4exp.cpp`
//! (`llama_model_qwen4exp::graph::build_graph`, lines 322-392):
//!
//! ```text
//!   res_hc = repeat(embed, hc_count)                          :322-327
//!   for il in 0..n_layer:
//!       if is_ple(il): res_hc += ple(tokens, res_hc)          :332-334
//!       (cur, inject) = hc_mix(res_hc, hc_attn_*)             :337-343
//!       cur           = attn(cur) | delta_net(cur)            :347-350
//!       res_hc        = hc_combine(res_hc, cur, inject)       :362
//!       (cur, inject) = hc_mix(res_hc, hc_ffn_*)              :364-370
//!       cur           = ffn(cur)                              :371
//!       res_hc        = hc_combine(res_hc, cur, inject)       :374
//!   h = hc_mix(res_hc, output_hc_*)   // no inject head       :381-386
//!   logits = output(h)                                        :388
//! ```
//!
//! The PLE addend lands on the carrier BEFORE the attention gate reads it, and
//! both write-backs land on the RAW carrier — the normed copy feeds the
//! bottleneck and the injection head and nothing else (port-doc trap #6).
//!
//! Correctness first, per the P2 plan: the PLE layer is a host hybrid (D17) and
//! the QSA selection is candle ops rather than a top-k kernel (D16). Neither
//! shape is a perf claim.

use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Tensor};

use crate::config::{LayerKind, XwenConfig};
use crate::gguf::{GgufFile, Weights};
use crate::model::{Ffn, Mixer, XwenModel};
use crate::rope::Rope;
use crate::stack_profile::Stage;

use super::hc::{HcRead, hc_write, seed_stream};
use super::indexer::{IndexerCache, IndexerCheckpoint, QsaIndexer};
use super::ple::{PleLayer, PleSnapshot, PleState};

/// Everything one qwen4exp layer has that a qwen35 layer does not. The block
/// itself (attention / DeltaNet / MoE) lives in `XwenModel::layers` beside its
/// KV cache, exactly as on the other checkpoints; this is the parallel vector.
pub struct Qwen4ExpLayer {
    /// The gate the mixer reads through, and writes its output back through.
    pub hc_attn: HcRead,
    /// The gate the FFN reads through.
    pub hc_ffn: HcRead,
    /// `Some` on a full-attention (QSA) layer, `None` on a DeltaNet one.
    pub indexer: Option<QsaIndexer>,
    /// The indexer's raw-key cache, present exactly when `indexer` is.
    pub indexer_cache: Option<IndexerCache>,
    /// `Some` on the one layer `ple.layers` names.
    pub ple: Option<PleLayer>,
    /// The PLE conv window and n-gram token history, present exactly when
    /// `ple` is.
    pub ple_state: Option<PleState>,
}

/// The qwen4exp additions to `XwenModel`: the per-layer gates and extra
/// recurrent state, plus the tail mixer that replaces `output_norm`.
///
/// The new recurrent state deliberately does NOT live in `LayerCache` (D15):
/// the indexer's raw keys and the PLE's two states have their own
/// checkpoint/rollback/reset here, driven from the existing `XwenModel` cache
/// methods, so `kv_cache.rs`'s five enums stay untouched through P2.
pub struct Qwen4ExpParts {
    pub layers: Vec<Qwen4ExpLayer>,
    /// The tail mixer (`output_hc_*`): the same read path with no injection
    /// head, collapsing the carrier to `hidden` for the lm head. There is no
    /// `output_norm` tensor on this architecture — this is it.
    pub output_hc: HcRead,
    pub hc_count: usize,
    pub hidden: usize,
    /// The state a `kv_checkpoint` armed, consumed by `kv_rollback`. Held here
    /// rather than in `KvCheckpoint` for the same reason the rest of this
    /// struct exists: `kv_cache.rs` stays out of P2.
    pending: Option<Qwen4ExpCheckpoint>,
}

/// What a checkpoint of the qwen4exp-only state costs: nothing per QSA layer
/// (its raw-key cache rolls back by truncation, like a `Full` KV layer) and one
/// `PleState` clone (~360 KB at the shipped geometry) for the PLE layer.
struct Qwen4ExpCheckpoint {
    /// One entry per model layer; `Some` on the QSA layers.
    indexer: Vec<Option<IndexerCheckpoint>>,
    /// One entry per model layer; `Some` on the PLE layer.
    ple: Vec<Option<PleSnapshot>>,
    /// The `(len0, span)` the matching [`crate::kv_cache::KvCheckpoint`] was
    /// taken at, so a rollback can prove the two are the same checkpoint.
    ///
    /// These parts are armed and rolled back BESIDE the `KvCheckpoint` rather
    /// than inside it (D15 keeps `kv_cache.rs` out of P2), which means nothing
    /// in the type system ties one to the other: a caller holding two
    /// checkpoints could roll the KV back against one and these against the
    /// other, and every state would be self-consistently wrong. Stamping the
    /// pair and checking it turns that into a loud error.
    len0: usize,
    span: usize,
}

impl Qwen4ExpParts {
    /// Load the gates, indexers and PLE layer for every layer of a qwen4exp
    /// file. `w` is the root weights handle; `gguf` is needed for the PLE
    /// table, which is read from the mapping rather than uploaded.
    pub fn load(
        w: &Weights,
        gguf: &GgufFile,
        cfg: &XwenConfig,
        rope: &Arc<Rope>,
        max_ctx: usize,
        device: &Device,
    ) -> Result<Self> {
        let q4 = cfg
            .qwen4exp
            .as_ref()
            .context("qwen4exp parts on a config with no qwen4exp section")?;
        let ple_layers = q4
            .ple
            .as_ref()
            .map(|p| p.layers.clone())
            .unwrap_or_default();

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let lw = w.pp(format!("blk.{il}"));
            let (indexer, indexer_cache) = match cfg.layer_kind(il) {
                LayerKind::Full => {
                    let idx = QsaIndexer::load(&lw, cfg, rope.clone())?;
                    let cache = idx.new_cache(max_ctx, device)?;
                    (Some(idx), Some(cache))
                }
                LayerKind::Linear => (None, None),
            };
            // The PLE layer list is 0-based in the GGUF (the converter shifts
            // config.json's one-indexed ids) and is treated as data: this
            // checkpoint carries one, a later one may carry another.
            let (ple, ple_state) = if ple_layers.contains(&il) {
                let layer = PleLayer::load(&lw, gguf, cfg)?;
                let state = layer.new_state();
                (Some(layer), Some(state))
            } else {
                (None, None)
            };
            layers.push(Qwen4ExpLayer {
                hc_attn: HcRead::load(&lw, "hc_attn", cfg, true)?,
                hc_ffn: HcRead::load(&lw, "hc_ffn", cfg, true)?,
                indexer,
                indexer_cache,
                ple,
                ple_state,
            });
        }

        Ok(Self {
            layers,
            output_hc: HcRead::load(w, "output_hc", cfg, false)?,
            hc_count: q4.hc_count,
            hidden: cfg.hidden,
            pending: None,
        })
    }

    /// Width of the residual carrier.
    pub fn width(&self) -> usize {
        self.hc_count * self.hidden
    }

    /// Arm a rollback over the next `span` tokens starting at cache length
    /// `len0`, alongside `LayerCache::checkpoint`.
    pub fn checkpoint(&mut self, len0: usize, span: usize) -> Result<()> {
        let mut indexer = Vec::with_capacity(self.layers.len());
        let mut ple = Vec::with_capacity(self.layers.len());
        for layer in &mut self.layers {
            indexer.push(match &mut layer.indexer_cache {
                Some(c) => Some(c.checkpoint(span)?),
                None => None,
            });
            ple.push(layer.ple_state.as_mut().map(|s| s.checkpoint(span)));
        }
        self.pending = Some(Qwen4ExpCheckpoint {
            indexer,
            ple,
            len0,
            span,
        });
        Ok(())
    }

    /// Roll back to `len0 + commit`, alongside `LayerCache::rollback`.
    ///
    /// The checkpoint is TAKEN, not kept, because one is all it is good for: a
    /// PLE state and a DeltaNet layer both answer a partial accept from a trail
    /// their checkpoint armed, and both clear that trail as they roll back. A
    /// second rollback against the same `KvCheckpoint` already fails in
    /// `kv_cache.rs` for exactly that reason, so keeping `pending` alive would
    /// only mean these parts silently accepted a call the KV path refuses.
    ///
    /// `len0` and `span` are checked against the ones the checkpoint was armed
    /// with. Nothing else enforces that pairing: these parts are armed beside
    /// the `KvCheckpoint` rather than inside it (D15), so a caller holding two
    /// checkpoints could roll the KV back against one and these against the
    /// other, and every state would be self-consistently wrong.
    pub fn rollback(&mut self, len0: usize, span: usize, commit: usize) -> Result<()> {
        // Validated before the checkpoint is consumed, so a caller that named
        // the wrong one has changed nothing and can still name the right one.
        match self.pending.as_ref() {
            None => bail!(
                "qwen4exp rollback without a checkpoint: the indexer caches and the PLE \
                 state are only recoverable through kv_checkpoint, and one checkpoint \
                 answers one rollback"
            ),
            Some(ckpt) => ensure!(
                ckpt.len0 == len0 && ckpt.span == span,
                "qwen4exp rollback against the wrong checkpoint: these parts were armed at \
                 len0 {} span {}, the KV checkpoint being rolled back is len0 {len0} \
                 span {span}",
                ckpt.len0,
                ckpt.span
            ),
        }
        let ckpt = self.pending.take().expect("checked just above");
        for (il, layer) in self.layers.iter_mut().enumerate() {
            if let (Some(cache), Some(c)) =
                (layer.indexer_cache.as_mut(), ckpt.indexer[il].as_ref())
            {
                cache.rollback(c, len0, span, commit)?;
            }
            if let (Some(state), Some(snap)) = (layer.ple_state.as_mut(), ckpt.ple[il].as_ref()) {
                state.rollback(snap, span, commit)?;
            }
        }
        Ok(())
    }

    /// Drop every sequence-scoped state, alongside `LayerCache::reset`.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            if let Some(cache) = layer.indexer_cache.as_mut() {
                cache.reset();
            }
            if let Some(state) = layer.ple_state.as_mut() {
                state.reset();
            }
        }
        self.pending = None;
    }

    /// Force every QSA layer to attend densely, by dropping its indexer.
    ///
    /// The equivalence this exists to test is the one D16 rests on: below the
    /// token budget the indexer's answer IS dense attention, so a model with no
    /// indexer at all must produce the same logits bit for bit.
    #[cfg(test)]
    pub(crate) fn force_dense_qsa(&mut self) {
        for layer in &mut self.layers {
            layer.indexer = None;
            layer.indexer_cache = None;
        }
    }
}

/// Bytes of qwen4exp-only recurrent state a load will hold: the indexers' raw
/// key planes plus the PLE conv window. Read by `warn_if_over_budget`.
///
/// The indexer planes are allocated at `max_ctx` up front rather than grown
/// with the KV cache (`IndexerCache` has no growth path in P2) — 4 MB per QSA
/// layer at 8k positions, and the reason this term is reported rather than
/// assumed small.
pub fn extra_state_bytes(cfg: &XwenConfig, max_ctx: usize) -> u64 {
    let Some(q4) = cfg.qwen4exp.as_ref() else {
        return 0;
    };
    let n_full = (0..cfg.n_layer).filter(|&il| cfg.is_full_attn(il)).count() as u64;
    let indexer = n_full
        * max_ctx as u64
        * super::indexer::indexer_bytes_per_token(q4.indexer_head_dim) as u64;
    let ple = match q4.ple.as_ref() {
        // conv window: `[hc_count * hidden, (kernel - 1) * ngram_size]` f32.
        Some(p) => {
            p.layers.len() as u64
                * (q4.hc_count * cfg.hidden) as u64
                * ((p.conv_kernel.saturating_sub(1)) * p.ngram_size) as u64
                * 4
        }
        None => 0,
    };
    indexer + ple
}

/// Run the qwen4exp stack over `tokens` at absolute position `pos` and return
/// `(h, taps, spec_taps)` — the same triple [`XwenModel::run_stack`] returns,
/// so `forward` and `forward_all_logits` are shared unchanged.
///
/// `h` is `[seq, hidden]`: the tail mixer's output, which is what feeds the lm
/// head and therefore what `post_norm_hidden` means on this architecture (there
/// is no `output_norm`). Spec taps are empty — qwen4exp ships no drafter (D6).
pub fn run_stack_hc(
    model: &mut XwenModel,
    tokens: &Tensor,
    pos: usize,
) -> Result<crate::model::StackOutput> {
    let seq = tokens.elem_count();
    ensure!(
        pos + seq <= model.max_ctx,
        "context overflow: position {pos} + {seq} tokens exceeds max_ctx {} \
         (raise --max-ctx or shorten the prompt)",
        model.max_ctx
    );
    model.grow_kv_capacity(pos + seq)?;

    // The profiler hooks the trunk uses, spelled against `model` instead of
    // `self`. Off — the normal case — each is one `Option` check.
    macro_rules! stage {
        ($stage:expr, $e:expr) => {{
            crate::stack_profile::stage_begin(&mut model.profile, &model.device)?;
            let out = $e;
            crate::stack_profile::stage_end(&mut model.profile, &model.device, $stage)?;
            out
        }};
    }
    crate::stack_profile::chunk_begin(&mut model.profile, &model.device, seq)?;

    let hc_count = model
        .qwen4exp
        .as_ref()
        .context("run_stack_hc on a model with no qwen4exp parts")?
        .hc_count;

    // The carrier is seeded by TILING the embedding, `[x, x, x, x]` — not
    // interleaving it (port-doc trap #15).
    let embedded = stage!(Stage::Embed, model.embed_tokens(tokens)?); // [seq, hidden] f32
    let mut stream = seed_stream(&embedded, hc_count)?; // [seq, hc_count * hidden]

    // The PLE hash runs on raw token ids on the host, so this chunk's ids have
    // to come back off the device — once per forward, and only when the
    // checkpoint carries a PLE layer at all.
    let wants_ids = model
        .qwen4exp
        .as_ref()
        .is_some_and(|p| p.layers.iter().any(|l| l.ple.is_some()));
    let token_ids: Vec<u32> = if wants_ids {
        tokens.to_dtype(DType::U32)?.flatten_all()?.to_vec1()?
    } else {
        Vec::new()
    };

    let mut taps: Vec<(String, Tensor)> = Vec::new();
    macro_rules! tap {
        ($name:expr, $il:expr, $t:expr) => {
            if model.tap_enabled {
                taps.push((format!("{}-{}", $name, $il), $t.clone()));
            }
        };
    }

    // The hoisted causal prefill mask, as on the trunk: a pure function of
    // (pos, seq) shared by every attention layer. A QSA layer that selects
    // above budget REPLACES it with the indexer's own mask inside `AttnBlock`
    // (the selected set is already causal); one that is below budget uses it.
    let full_mask = if seq > 1 {
        let n_head = (0..model.cfg.n_layer)
            .find(|&il| model.cfg.is_full_attn(il))
            .map(|il| model.cfg.n_head(il));
        match n_head {
            Some(n) => model.build_prefill_mask(n, seq, pos)?,
            None => None,
        }
    } else {
        None
    };

    for il in 0..model.layers.len() {
        // Disjoint field borrows: the blocks are read, the caches and the
        // qwen4exp state are written, and the profiler is a third field again.
        let layer = &model.layers[il];
        let cache = &mut model.caches[il];
        let parts = model
            .qwen4exp
            .as_mut()
            .context("run_stack_hc on a model with no qwen4exp parts")?;
        let p = &mut parts.layers[il];

        // PLE injection, onto the carrier and BEFORE the attention gate reads
        // it (qwen4exp.cpp:332-334). Not bracketed as its own profiler stage:
        // it is a host hybrid in P2 and would need a Stage variant of its own
        // to be reported honestly, so its cost shows up in `InterStageHost`.
        if let (Some(ple), Some(state)) = (p.ple.as_ref(), p.ple_state.as_mut()) {
            let addend = ple.forward(&token_ids, &stream, state)?;
            stream = (&stream + &addend)?;
        }

        // --- attention half.
        let (x, inject) = stage!(Stage::AttnNorm, p.hc_attn.read(&stream)?);
        let inject = inject.context("the hc_attn gate has no injection head")?;
        tap!("hc_mixed_attn", il, x);

        let out = match &layer.mixer {
            Mixer::Full(block) => {
                // Selection first: it appends this chunk's raw indexer keys, so
                // it must run at the same `pos` the K/V append below does.
                let qsa = match (p.indexer.as_ref(), p.indexer_cache.as_mut()) {
                    (Some(idx), Some(cache)) => Some(idx.select(&x, cache, pos)?),
                    _ => None,
                };
                stage!(
                    Stage::MixerFullAttn,
                    block.forward(&x, cache, pos, full_mask.as_ref(), qsa.as_ref())?
                )
            }
            Mixer::Linear(block) => stage!(Stage::MixerDelta, block.forward(&x, cache)?),
        };
        tap!("attn_o_proj", il, out);
        stream = stage!(Stage::ResidualAttn, hc_write(&stream, &out, &inject)?);

        // --- FFN half.
        let (x, inject) = stage!(Stage::FfnNorm, p.hc_ffn.read(&stream)?);
        let inject = inject.context("the hc_ffn gate has no injection head")?;
        tap!("hc_mixed_ffn", il, x);

        let out = match &layer.ffn {
            Ffn::Moe(moe) => stage!(Stage::Ffn, moe.forward(&x)?),
            Ffn::Dense(_) => bail!(
                "layer {il} carries a dense FFN, but every qwen4exp layer is MoE; this model \
                 was assembled from the wrong architecture arm"
            ),
        };
        tap!("ffn_out", il, out);
        stream = stage!(Stage::ResidualFfn, hc_write(&stream, &out, &inject)?);
        // Deliberately NOT named `l_out`: the trunk's `l_out` is a
        // `hidden`-wide residual and this is the `hc_count * hidden` carrier, so
        // sharing the name would hand the parity harness two tensors of
        // different width under one key. `attn_o_proj` and `ffn_out` above DO
        // keep their trunk names — those are the block outputs, and they mean
        // exactly what they mean on qwen35.
        tap!("hc_carrier", il, stream);
    }

    // The pre-tail carrier, the analogue of the trunk's pre-final-norm residual.
    if model.tap_enabled {
        taps.push(("h_nextn".to_string(), stream.clone()));
    }

    // The tail mixer replaces `output_norm`: same read path, no injection head.
    crate::stack_profile::stage_begin(&mut model.profile, &model.device)?;
    let h = model
        .qwen4exp
        .as_ref()
        .context("run_stack_hc on a model with no qwen4exp parts")?
        .output_hc
        .mix(&stream)?; // [seq, hidden]
    crate::stack_profile::stage_end(&mut model.profile, &model.device, Stage::FinalNorm)?;

    // The lm head's input, and therefore what `post_norm_hidden` means here:
    // there is no `output_norm` on this architecture, the tail mixer is it.
    if model.keep_post_norm {
        model.post_norm_hidden = Some(h.clone());
    }
    Ok((h, taps, Vec::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::ops::ExpertRunner;
    use crate::qwen4exp::tiny_gguf::{TinyGeometry, write_tiny_qwen4exp};
    use std::path::PathBuf;

    /// The tiny fixture needs a Metal device: the PLE table is read from the
    /// GGUF MAPPING rather than uploaded (D17), and `gguf::open` only builds a
    /// mapping on Metal. A machine without one skips rather than fails — the
    /// design target is this machine.
    fn device_or_skip(what: &str) -> Option<Device> {
        match metal_device() {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("skipping {what}: no Metal device ({e})");
                None
            }
        }
    }

    /// A fresh directory per test, so one test's fixture never shadows another's.
    fn fixture_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xwen_stack_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Which synthetic file a test runs against.
    ///
    /// `Exact` stores every tensor F32, so the fixture introduces no rounding of
    /// its own and a failure is the graph's. `Mixed` stores the REAL file's
    /// dtype spread — BF16 indexer projections, Q8_0 planes, Q4_K/Q5_K experts,
    /// Q5_1 and Q8_0 `down_exps`, and the IQ4_NL PLE table that candle cannot
    /// name at all. The graph tests below run on both, because every one of
    /// those is a separate load-time dispatch arm and the exact fixture reaches
    /// none of them.
    #[derive(Clone, Copy, Debug)]
    enum Fixture {
        Exact,
        Mixed,
    }

    impl Fixture {
        fn label(self) -> &'static str {
            match self {
                Fixture::Exact => "f32",
                Fixture::Mixed => "mixed",
            }
        }
    }

    /// Build the tiny qwen4exp model, at `max_ctx` positions.
    fn tiny_model(tag: &str, device: &Device, max_ctx: usize) -> (XwenModel, PathBuf) {
        tiny_model_of(Fixture::Exact, tag, device, max_ctx)
    }

    fn tiny_model_of(
        fixture: Fixture,
        tag: &str,
        device: &Device,
        max_ctx: usize,
    ) -> (XwenModel, PathBuf) {
        let dir = fixture_dir(&format!("{tag}_{}", fixture.label()));
        let path = dir.join("tiny-qwen4exp.gguf");
        match fixture {
            Fixture::Exact => write_tiny_qwen4exp(&path, &TinyGeometry::default()).unwrap(),
            // The quantizable geometry, not the default: ggml quantizes along
            // the row, so every quantized plane's row has to be a whole number
            // of blocks (see `TinyGeometry::quantizable`).
            Fixture::Mixed => {
                super::super::tiny_gguf::write_tiny_qwen4exp_mixed(
                    &path,
                    &TinyGeometry::quantizable(),
                )
                .unwrap();
            }
        }
        let gguf = crate::gguf::open(&path, device).unwrap();
        let model = XwenModel::load(gguf, ExpertRunner::Reference, max_ctx).unwrap();
        (model, dir)
    }

    fn ids(device: &Device, tokens: &[u32]) -> Tensor {
        Tensor::new(tokens, device).unwrap()
    }

    fn host(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// Relative L2 distance, the measure the parity harness grades with.
    fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
        assert_eq!(got.len(), want.len());
        let (mut num, mut den) = (0f64, 0f64);
        for (g, w) in got.iter().zip(want) {
            num += (*g as f64 - *w as f64).powi(2);
            den += (*w as f64).powi(2);
        }
        (num / den.max(1e-30)).sqrt()
    }

    /// The 15 ids both chunking tests run, including the PLE segment separator
    /// (id 3, the fixture's `ple.eos_token_id`) in the middle: the n-gram window
    /// must not reach across it, and a chunk boundary must not change where it
    /// stops.
    const SEQ: [u32; 15] = [7, 91, 12, 200, 3, 44, 44, 8, 130, 17, 3, 61, 5, 249, 33];

    /// A 12-token prefill plus three decode steps against a single 15-token
    /// prefill, measured with the QSA indexers live and again with them removed.
    ///
    /// Every piece of per-sequence state has to be right for the two to agree:
    /// the PLE conv window and its two-token n-gram history carry across the
    /// boundary, the DeltaNet recurrent state advances token by token, and the
    /// QSA indexer cuts its key blocks from the SEQUENCE rather than the chunk
    /// (15 tokens over a budget of 8 means selection is really running, not
    /// degenerating to dense).
    ///
    /// Agreement here is NUMERIC, not bitwise, and the floor is the trunk's,
    /// not this graph's: prefill and decode take different attention projection
    /// routes (a batched f16 mm against a decode gemv), so the K/V rows they
    /// round into the f16 cache differ in low bits — the same cross-partition
    /// caveat [`XwenModel::kv_rollback`] documents. That floor is what the
    /// forced-dense arm measures, which is why the tolerance is compared
    /// against a control rather than picked: under `XWEN_ATTN_F32`, which pins
    /// one weight representation across both routes, both arms drop to ~3e-5.
    #[test]
    fn chunked_prefill_and_decode_match_a_single_prefill() {
        let Some(device) = device_or_skip("chunked_prefill_and_decode_match_a_single_prefill")
        else {
            return;
        };

        // `dense` removes the indexers, so the arm exercises the identical
        // graph with plain causal attention.
        let split = |fixture: Fixture, tag: &str, dense: bool| -> f64 {
            let (mut one_shot, dir_a) = tiny_model_of(fixture, tag, &device, 128);
            let (mut chunked, dir_b) = tiny_model_of(fixture, &format!("{tag}_b"), &device, 128);
            if dense {
                one_shot.qwen4exp.as_mut().unwrap().force_dense_qsa();
                chunked.qwen4exp.as_mut().unwrap().force_dense_qsa();
            }
            let want = host(&one_shot.forward(&ids(&device, &SEQ), 0).unwrap());

            chunked.forward(&ids(&device, &SEQ[..12]), 0).unwrap();
            let mut got = Vec::new();
            for (i, &tok) in SEQ[12..].iter().enumerate() {
                got = host(&chunked.forward(&ids(&device, &[tok]), 12 + i).unwrap());
            }
            let _ = std::fs::remove_dir_all(&dir_a);
            let _ = std::fs::remove_dir_all(&dir_b);
            rel_l2(&got, &want)
        };

        // Run on both fixtures: the mixed one is the only thing in the suite
        // that puts a BF16 indexer projection, a Q4_K expert or an IQ4_NL PLE
        // table through a whole forward.
        for fixture in [Fixture::Exact, Fixture::Mixed] {
            let qsa = split(fixture, "chunk_qsa", false);
            let dense = split(fixture, "chunk_dense", true);
            assert_chunking_agrees(fixture, qsa, dense);
        }
    }

    /// The two bounds `chunked_prefill_and_decode_match_a_single_prefill`
    /// applies, factored out so both fixtures are graded by exactly the same
    /// rule rather than by two copies of it that could drift.
    fn assert_chunking_agrees(fixture: Fixture, qsa: f64, dense: f64) {
        let what = fixture.label();
        // The absolute bound: an order of magnitude above the measured floor,
        // and far below anything a wrong carried state produces (a dropped PLE
        // history or a chunk-local block cut moves this into the 1e-1 range).
        assert!(
            qsa < 2e-3,
            "{what}: chunked decode diverged from one-shot prefill: rel_l2 {qsa}"
        );
        // The claim that is actually about this graph: running the sparse
        // overlay across a chunk boundary costs nothing over running dense
        // attention across the same boundary — i.e. the two partitions select
        // the same blocks.
        assert!(
            qsa <= dense * 1.5,
            "{what}: the QSA overlay chunked worse than dense attention did \
             (qsa {qsa} vs dense {dense}): the two partitions are selecting \
             different key blocks"
        );
    }

    /// A rejected speculative span leaves no trace: checkpoint, decode two
    /// tokens, roll back to zero committed, decode the same two again, and the
    /// logits are the ones the first attempt produced.
    ///
    /// The qwen4exp-only state is what this pins — the QSA raw-key caches roll
    /// back by truncation and the PLE conv window and token history by an exact
    /// clone, both driven from `kv_checkpoint`/`kv_rollback` alongside the KV
    /// caches (D15).
    #[test]
    fn checkpoint_and_rollback_reproduce_the_rolled_back_decode() {
        let Some(device) =
            device_or_skip("checkpoint_and_rollback_reproduce_the_rolled_back_decode")
        else {
            return;
        };
        for fixture in [Fixture::Exact, Fixture::Mixed] {
            let what = fixture.label();
            let (mut model, dir) = tiny_model_of(fixture, "rollback", &device, 128);

            model.forward(&ids(&device, &SEQ[..10]), 0).unwrap();

            let mut first = Vec::new();
            let ckpt = model.kv_checkpoint(2).unwrap();
            for (i, &tok) in SEQ[10..12].iter().enumerate() {
                first = host(&model.forward(&ids(&device, &[tok]), 10 + i).unwrap());
            }
            model.kv_rollback(&ckpt, 0).unwrap();
            assert_eq!(
                model.cache_len(),
                10,
                "{what}: rollback did not restore the length"
            );

            let mut second = Vec::new();
            for (i, &tok) in SEQ[10..12].iter().enumerate() {
                second = host(&model.forward(&ids(&device, &[tok]), 10 + i).unwrap());
            }
            assert_eq!(
                first, second,
                "{what}: a rolled-back span replayed to different logits"
            );

            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// A PARTIAL accept is the ordinary case, not the edge one: every
    /// speculative round keeps at least one token, so `commit > 0` is what a
    /// generation actually runs. Every piece of qwen4exp state has to land on
    /// the accepted prefix — the QSA raw-key caches, and the PLE conv window and
    /// n-gram history, which rewind through a per-token trail rather than to the
    /// checkpoint.
    ///
    /// Graded against the straight line: decode `span` tokens under a
    /// checkpoint, roll back to `commit`, and the next token's logits must be
    /// the ones a run that only ever decoded those `commit` tokens produces.
    /// A state that rewound too far does not error anywhere — it just keeps
    /// conditioning on a history that never happened, for the rest of the
    /// generation — so this comparison is the only thing that sees it.
    #[test]
    fn a_partial_commit_leaves_every_state_on_the_accepted_prefix() {
        let Some(device) =
            device_or_skip("a_partial_commit_leaves_every_state_on_the_accepted_prefix")
        else {
            return;
        };

        let base = 10usize;
        let span = 3usize;
        // The token decoded AFTER the rollback. Its logits depend on every
        // carried state, which is what makes them the probe.
        let probe = SEQ[14];

        for commit in 0..=span {
            // Straight line: prefill, decode only the accepted tokens, probe.
            let (mut want_model, dir_a) = tiny_model("commit_want", &device, 128);
            want_model.forward(&ids(&device, &SEQ[..base]), 0).unwrap();
            for (i, &tok) in SEQ[base..base + commit].iter().enumerate() {
                want_model.forward(&ids(&device, &[tok]), base + i).unwrap();
            }
            let want = host(
                &want_model
                    .forward(&ids(&device, &[probe]), base + commit)
                    .unwrap(),
            );

            // Speculative: prefill, checkpoint, decode the whole span, roll back
            // to `commit`, probe.
            let (mut got_model, dir_b) = tiny_model("commit_got", &device, 128);
            got_model.forward(&ids(&device, &SEQ[..base]), 0).unwrap();
            let ckpt = got_model.kv_checkpoint(span).unwrap();
            for (i, &tok) in SEQ[base..base + span].iter().enumerate() {
                got_model.forward(&ids(&device, &[tok]), base + i).unwrap();
            }
            got_model.kv_rollback(&ckpt, commit).unwrap();
            assert_eq!(
                got_model.cache_len(),
                base + commit,
                "commit {commit}: rollback did not restore the length"
            );
            let got = host(
                &got_model
                    .forward(&ids(&device, &[probe]), base + commit)
                    .unwrap(),
            );

            // Bitwise: both sides reach the probe by the same one-token decode
            // route over states that were written by the same one-token decodes,
            // so there is no cross-partition rounding to allow for here.
            assert_eq!(got, want, "commit {commit}: the probe token's logits");

            let _ = std::fs::remove_dir_all(&dir_a);
            let _ = std::fs::remove_dir_all(&dir_b);
        }
    }

    /// One checkpoint answers one rollback, and only the rollback it was armed
    /// for. The qwen4exp parts are armed BESIDE the `KvCheckpoint` rather than
    /// inside it (D15), so nothing in the type system pairs them — a caller
    /// holding two checkpoints could roll the KV back against one and these
    /// against the other, and every state would be self-consistently wrong.
    #[test]
    fn a_rollback_is_refused_against_a_checkpoint_it_was_not_armed_for() {
        let Some(device) =
            device_or_skip("a_rollback_is_refused_against_a_checkpoint_it_was_not_armed_for")
        else {
            return;
        };
        let (mut model, dir) = tiny_model("ckpt_pairing", &device, 128);
        model.forward(&ids(&device, &SEQ[..8]), 0).unwrap();

        // Driven directly rather than through `XwenModel::kv_rollback`, because
        // that rolls the KV caches FIRST: by the time these parts saw a
        // mismatched checkpoint the caches would already have acted on it, and
        // the test would be measuring the order of two loops rather than the
        // guard. Both refusals below happen before anything is touched.
        model.qwen4exp.as_mut().unwrap().checkpoint(8, 2).unwrap();
        // Step the reserved span, so the trails can answer a rollback at all and
        // the only thing left to refuse is the checkpoint's identity.
        for (i, &tok) in SEQ[8..10].iter().enumerate() {
            model.forward(&ids(&device, &[tok]), 8 + i).unwrap();
        }
        let parts = model.qwen4exp.as_mut().unwrap();

        // A checkpoint armed at a different length answers nothing.
        let err = parts.rollback(9, 2, 0).unwrap_err().to_string();
        assert!(err.contains("wrong checkpoint"), "{err}");
        let err = parts.rollback(8, 3, 0).unwrap_err().to_string();
        assert!(err.contains("wrong checkpoint"), "{err}");

        // Refusing left it armed, so the right call still works — once.
        parts.rollback(8, 2, 0).unwrap();
        let err = parts.rollback(8, 2, 0).unwrap_err().to_string();
        assert!(err.contains("without a checkpoint"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Below the indexer's token budget, sparse attention IS dense attention:
    /// every complete block fits and the incomplete tail is always visible, so
    /// `QsaSelection::Dense` is the answer rather than an approximation of one.
    /// A model with the indexers removed entirely must therefore agree bit for
    /// bit (D16 — and the cheap dev-time equivalence check the port doc names).
    #[test]
    fn below_budget_qsa_is_bit_identical_to_dense_attention() {
        let Some(device) = device_or_skip("below_budget_qsa_is_bit_identical_to_dense_attention")
        else {
            return;
        };

        // The fixture's budget is 8 tokens; 8 is the last length that fits.
        let below = &SEQ[..8];

        for fixture in [Fixture::Exact, Fixture::Mixed] {
            let (mut sparse, dir_a) = tiny_model_of(fixture, "qsa", &device, 128);
            let with_indexer = host(&sparse.forward(&ids(&device, below), 0).unwrap());

            let (mut dense, dir_b) = tiny_model_of(fixture, "dense", &device, 128);
            dense.qwen4exp.as_mut().unwrap().force_dense_qsa();
            let without_indexer = host(&dense.forward(&ids(&device, below), 0).unwrap());

            assert_eq!(
                with_indexer,
                without_indexer,
                "{}: a below-budget QSA prefill differs from the same model with no indexer",
                fixture.label()
            );

            let _ = std::fs::remove_dir_all(&dir_a);
            let _ = std::fs::remove_dir_all(&dir_b);
        }
    }

    /// The real Qwen3.8-Flash-Next file: it loads, reports the geometry the port
    /// doc pins, and prefills to finite logits.
    ///
    /// Opt-in twice over — the checkpoint has to be in the HF cache AND
    /// `XWEN_QWEN4EXP_SMOKE` set — because this machine runs one large model
    /// process at a time (CLAUDE.md, "Operational hazards") and a plain
    /// `cargo test` must not page a 20 GB checkpoint onto the GPU behind
    /// whoever else is using it.
    #[test]
    fn real_flash_next_file_loads_and_prefills() {
        if std::env::var_os("XWEN_QWEN4EXP_SMOKE").is_none() {
            eprintln!(
                "skipping real_flash_next_file_loads_and_prefills: set XWEN_QWEN4EXP_SMOKE to run it"
            );
            return;
        }
        let Some(path) = crate::hub::cached_model(crate::hub::Model::Qwen38FlashNext) else {
            eprintln!(
                "skipping real_flash_next_file_loads_and_prefills: checkpoint not in the HF cache"
            );
            return;
        };
        let Some(device) = device_or_skip("real_flash_next_file_loads_and_prefills") else {
            return;
        };

        let gguf = crate::gguf::open(&path, &device).unwrap();
        let mut model = XwenModel::load(gguf, ExpertRunner::Fused, 8192).unwrap();

        let cfg = model.config();
        assert_eq!(cfg.n_layer, 48);
        assert_eq!(
            (0..cfg.n_layer).filter(|&il| cfg.is_full_attn(il)).count(),
            12,
            "12 of the 48 layers are QSA attention, the rest gated DeltaNet"
        );
        let ple = cfg
            .qwen4exp
            .as_ref()
            .expect("the real file carries a qwen4exp config section")
            .ple
            .as_ref()
            .expect("the real file ships the PLE table");
        assert_eq!(ple.layers.len(), 1, "one PLE layer, 0-based in the GGUF");

        let vocab = cfg.vocab;
        let prompt: Vec<u32> = vec![9707, 11, 1879, 0, 358, 1079, 264, 1273];
        let logits = model.forward(&ids(&device, &prompt), 0).unwrap();
        let logits = host(&logits);
        assert_eq!(logits.len(), vocab);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "the prefill produced non-finite logits"
        );
        let top = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        assert!(top < vocab, "argmax {top} is outside the vocabulary");
    }
}
