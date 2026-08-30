use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::RmsNorm;

use crate::attention::{AttnBlock, AttnWeights, PrefillMask};
use crate::config::{Arch, LayerKind, XwenConfig};
use crate::gguf::{GgufFile, QLinear, Weights};
use crate::kv_cache::{
    CacheSnapshot, HostFullKv, HostSnapshot, KvCheckpoint, LayerCache, LayerCheckpoint,
    LayerSnapshot, MaskKind,
};
use crate::linear_attn::LinearAttnBlock;
use crate::moe::{DenseMlp, MoeBlock};
use crate::ops::ExpertRunner;
use crate::rope::Rope;
use crate::stack_profile::Stage;

/// Warn at load if the worst-case resident footprint (weights mmap-uploaded to
/// the device plus the KV cache grown to its `max_ctx` ceiling) exceeds this.
/// The Q4_K_M checkpoints are 19-20GB and the largest blessed file (27B Q8_0)
/// 28.6GB, with a 17GB KV ceiling at the 27B's full 256k window — so this
/// fires only on a `--max-ctx` far past anything the checkpoints support.
const MEMORY_WARN_BYTES: u64 = 90 * 1024 * 1024 * 1024;

/// Initial full-attention KV allocation, in positions. `max_ctx` is a CEILING,
/// not an allocation: each full-attention layer starts at this many slots
/// (or `max_ctx`, if smaller) and doubles on demand as a sequence grows into
/// it (`grow_kv_capacity`), so a 128k+ context budget costs memory only when a
/// conversation actually reaches it — and costs it only until the model is
/// dropped, which is what shrinks a grown cache back down (the serve engine's
/// idle unload rides this). 8192 positions are 0.5 GiB across the 27B's 16
/// full layers (64 KiB/token, `Model::kv_bytes_per_token`) and 0.16 GiB
/// across the 35B's 10 (20 KiB/token).
const KV_INITIAL_CTX: usize = 8192;

/// What a stack run hands back: the pre-lm_head hidden states `[seq, hidden]`
/// for every position, the named parity taps collected when capture is on, and
/// the `(layer, l_out)` spec-decode taps the drafter reads.
pub(crate) type StackOutput = (Tensor, Vec<(String, Tensor)>, Vec<(usize, Tensor)>);

/// The per-layer FFN: a plain SwiGLU MLP on the dense checkpoint, the
/// softmax-routed MoE block (routed experts + gated shared expert) on both MoE
/// ones. Whole-model, not per-layer — no checkpoint mixes the two.
///
/// `pub(crate)` because `qwen4exp::stack` runs the same blocks through a
/// different residual (D14) and therefore has to match on this.
pub(crate) enum Ffn {
    Dense(DenseMlp),
    Moe(MoeBlock),
}

/// The per-layer sequence mixer: softmax attention over a KV cache on the
/// `(il + 1) % 4 == 0` layers, gated DeltaNet over a recurrent state on the
/// other three in four.
pub(crate) enum Mixer {
    Full(AttnBlock),
    Linear(LinearAttnBlock),
}

/// One transformer layer: pre-attention norm + mixer, pre-FFN norm + FFN. Both
/// residual adds are owned by `XwenModel::forward`, not here.
///
/// `ffn_norm` is loaded from `post_attention_norm` — there is no `ffn_norm`
/// tensor, and despite the name it is the PRE-MLP norm (HF semantics), not a
/// Gemma-style post-norm.
///
/// Both norms are `None` on qwen4exp, whose file carries neither: there the
/// normalization lives inside the hyper-connection gate that reads the residual
/// carrier, and `qwen4exp::stack::run_stack_hc` — the only code that runs those
/// layers — never looks at these fields.
pub(crate) struct Layer {
    attn_norm: Option<RmsNorm>,
    pub(crate) mixer: Mixer,
    ffn_norm: Option<RmsNorm>,
    pub(crate) ffn: Ffn,
}

/// The assembled model: embeddings, the layer stack (attention/DeltaNet +
/// dense/MoE FFN), final norm, lm_head. Holds the per-layer KV and recurrent
/// caches; batch=1 by design. Exposes per-layer residual taps for the parity
/// harness and the DFlash drafter.
pub struct XwenModel {
    pub(crate) cfg: XwenConfig,
    pub(crate) device: Device,
    /// Dequantized token embeddings `[vocab, hidden]`, f16 on Metal (halves the
    /// 1.2GB f32 footprint) or f32 elsewhere. Rows are gathered per forward.
    embed: Tensor,
    pub(crate) layers: Vec<Layer>,
    pub(crate) caches: Vec<LayerCache>,
    /// The qwen4exp subsystems: per-layer hyper-connection gates, QSA indexers
    /// with their raw-key caches, the PLE layer with its own recurrent state,
    /// and the tail mixer. `Some` exactly on `Arch::Qwen4Exp`, and its presence
    /// is what routes `run_stack` to the second graph (D14).
    pub(crate) qwen4exp: Option<crate::qwen4exp::stack::Qwen4ExpParts>,
    /// `None` on qwen4exp, which has no `output_norm` tensor — its tail mixer
    /// (`Qwen4ExpParts::output_hc`) is the final normalization.
    output_norm: Option<RmsNorm>,
    lm_head: QLinear,
    /// Retained handle to the lm_head weight's Metal buffer, shared with
    /// `lm_head`'s QTensor (zero-copy). Present only on Metal for the vendored
    /// plain mat-vec (`mv_vendored_supported` — q8_0 on the current official
    /// file, q6_K on the retired original); `None` off Metal or for an
    /// unsupported dtype, in which
    /// case the decode path stays on `lm_head.forward`.
    lm_head_buffer: Option<Arc<candle_metal_kernels::metal::Buffer>>,
    lm_head_dtype: candle_core::quantized::GgmlDType,
    /// Attention weight dtype resolved once at load (F16 default, F32 under
    /// XWEN_ATTN_F32); activations are f32 either way. Surfaced so dump
    /// provenance can record which path ran.
    attn_dtype: DType,
    /// Attention PREFILL gemm path resolved once at load, for dump provenance:
    /// "f32-bypass" when XWEN_ATTN_F32 sends attention through the legacy
    /// dequant-f32 QMatMul (the f16 library, and thus its mm branch, never runs),
    /// else the f16 library's mm-branch kernel — "tensor" (the shipped Metal-4
    /// cooperative-tensor default) or "classic" (the simdgroup kernel, under the
    /// XWEN_ATTN_MM_CLASSIC kill-switch).
    attn_mm: &'static str,
    /// Attention DECODE-projection path resolved once at load, for dump
    /// provenance: "f32-bypass" under XWEN_ATTN_F32 (the whole block is the
    /// legacy dequant-f32 QMatMul), "q8" when the checkpoint stores its attention
    /// weights q8_0 and the vendored decode gemv is active (the current official
    /// file — the production default), else "f16" (the dense f16 gemv — an
    /// f16-attention checkpoint like the retired original, or a q8_0 file under
    /// XWEN_ATTN_DEQUANT).
    attn_decode: &'static str,
    pub(crate) max_ctx: usize,
    /// Positions every full-attention layer currently has allocated — the lazy
    /// KV allocation's high-water mark, `KV_INITIAL_CTX` at load and grown in
    /// lockstep by `grow_kv_capacity` up to the `max_ctx` ceiling.
    kv_slots: usize,
    pub(crate) tap_enabled: bool,
    taps: Vec<(String, Tensor)>,
    /// DFlash spec-decode residual-stream taps: the `l_out` layer indices the
    /// drafter reads (e.g. `[1,10,19,29,38,47]`), or `None` (default). Separate
    /// from the heavyweight parity `taps` above — a single `Option` check per
    /// layer when unset, and only the configured layers' handles are cloned when
    /// set (a cheap Arc bump each, no dtype conversion).
    spec_tap_layers: Option<Vec<usize>>,
    /// The most recent forward's spec taps, in configured order, drained by
    /// `take_spec_taps`.
    spec_taps: Vec<Tensor>,
    /// Whether to retain the post-final-norm hidden for the MTP draft head.
    /// Off by default: a forward that nobody is drafting from should not hold a
    /// `[seq, hidden]` handle alive past its own scope.
    pub(crate) keep_post_norm: bool,
    /// The most recent forward's post-final-norm hidden `[seq, hidden]` f32,
    /// drained by `take_post_norm_hidden`. This is what the MTP head's `h` input
    /// is — NOT the pre-norm residual `spec_taps` carries.
    pub(crate) post_norm_hidden: Option<Tensor>,
    /// Per-stage forward timing, present only under `XWEN_STACK_PROFILE`
    /// (`ops::stack_profile`). `None` — the normal case — costs one `Option`
    /// check per instrumented site; `Some` brackets every stage with device
    /// syncs, which serializes the pipeline and makes the run's throughput
    /// meaningless. Diagnosis only: no arithmetic depends on it.
    pub(crate) profile: Option<crate::stack_profile::StackProfiler>,
    /// Keeps the GGUF mapping (and its Metal view buffers' residency set) alive
    /// for the model's lifetime on the mmap alias load: the aliased attention
    /// f16 planes are plain `Tensor`s that cannot carry the mapping themselves,
    /// so dropping it while the model lives would leave their view buffers (and
    /// the GPU) reading unmapped pages (`gguf::MmapSource`'s lifetime
    /// invariant; expert stacks additionally hold their own clones). One
    /// mapping per shard of a split GGUF — a tensor aliases whichever shard
    /// holds it, so all must stay alive. Empty on the classic copying load.
    _weights_mmap: Vec<Arc<crate::gguf::MmapSource>>,
    /// Identity of the checkpoint this model was loaded from, carried so that
    /// anything persisted from a running model (cache images) can be stamped with
    /// it and refused against a different one. The `GgufFile` it came from is
    /// dropped once loading is done, so the id is copied in here.
    checkpoint: crate::gguf::CheckpointId,
}

impl XwenModel {
    /// Preconditions this stack cannot recover from, checked on the parsed
    /// config before the rope table and the ~1.2 GB token-embedding dequant
    /// materialize anything.
    ///
    /// Only one such precondition exists: a qwen4exp file whose metadata carries
    /// no `qwen4exp` config section. Every hyper-connection gate, indexer and
    /// PLE layer is sized from that section, so its absence is a broken
    /// conversion rather than a checkpoint variant — and failing here rather
    /// than at the first `HcRead::load` keeps the ~1.2 GB dequant out of a run
    /// that cannot finish.
    fn check_arch(cfg: &XwenConfig) -> Result<()> {
        match cfg.arch {
            Arch::Dense | Arch::Moe => Ok(()),
            Arch::Qwen4Exp => {
                ensure!(
                    cfg.qwen4exp.is_some(),
                    "this file declares architecture qwen4exp but carries no \
                     qwen4exp.hyper_connection.* metadata: the residual carrier, the QSA \
                     indexers and the PLE table are all sized from it, so the graph cannot \
                     be built"
                );
                Ok(())
            }
        }
    }

    pub fn load(gguf: Arc<GgufFile>, runner: ExpertRunner, max_ctx: usize) -> Result<Self> {
        let cfg = XwenConfig::from_gguf(&gguf.content)?;
        Self::check_arch(&cfg)?;
        let device = gguf.device.clone();
        let w = Weights::from_gguf(gguf.clone());

        // Attention WEIGHT dtype, read ONCE at load: f16 by default — the GGUF
        // stores the attention weights as F16, so the default keeps them dense
        // f16 and runs each projection through the vendored mixed-dtype kernels
        // (ops::matmul_f16: f16 weights x f32 activations, f32 accumulate and
        // output — the fork's exact mul_mat precision structure, with the
        // stored weights as the only f16 rounding). XWEN_ATTN_F32
        // (presence-based, like the other XWEN_* switches) dequantizes them
        // to dense f32 instead — the fully legacy path, which the strict
        // parity tier gates.
        let attn_dtype = if std::env::var_os("XWEN_ATTN_F32").is_some() {
            DType::F32
        } else {
            DType::F16
        };
        let attn_weights = match attn_dtype {
            DType::F32 => AttnWeights::DequantF32,
            _ => AttnWeights::F16,
        };

        // The attention prefill gemm path, for dump provenance. XWEN_ATTN_F32
        // routes the whole attention block through the legacy dequant-f32 QMatMul,
        // so the f16 library's mm branch never runs ("f32-bypass"). Otherwise the
        // mm branch runs, defaulting to the Metal-4 cooperative-tensor kernel, or
        // the classic simdgroup kernel under XWEN_ATTN_MM_CLASSIC. Resolved
        // once here (single source of truth is the cached `attn_mm_classic` switch
        // the dispatch reads).
        let attn_mm = if attn_dtype == DType::F32 {
            "f32-bypass"
        } else if crate::ops::attn_mm_classic() {
            "classic"
        } else {
            "tensor"
        };

        // One RoPE table, Arc-shared into every attention layer: the DeltaNet
        // layers have no positional encoding at all. Built to the runtime context
        // budget, not n_ctx_train, and only over the rotated dims (n_rot 64 of
        // head_dim 256), so it is small at any context.
        let rope = Arc::new(Rope::new(cfg.rope(), max_ctx, &device)?);

        // Token embeddings: dequantize once. On Metal the f32 table is 1.2GB, so
        // keep it as f16 (halved) and upcast the gathered rows to f32 per forward.
        let embed = w.qtensor("token_embd")?.dequantize(&device)?;
        let embed = if matches!(device, Device::Metal(_)) {
            embed.to_dtype(DType::F16)?
        } else {
            embed.to_dtype(DType::F32)?
        };

        // qwen4exp shares every block with qwen35moe and differs only in what
        // sits BETWEEN them: a 4-stream residual carrier instead of a single
        // one, which is why its file has no `attn_norm`, no
        // `post_attention_norm` and no `output_norm` (D14). So the block
        // construction below is not branched — only the norms are, and the
        // gates/indexers/PLE are loaded alongside afterwards.
        let hc = cfg.arch == Arch::Qwen4Exp;

        let kv_slots = max_ctx.min(KV_INITIAL_CTX);
        let mut layers = Vec::with_capacity(cfg.n_layer);
        let mut caches = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let lw = w.pp(format!("blk.{il}"));
            let mixer = match cfg.layer_kind(il) {
                LayerKind::Full => {
                    Mixer::Full(AttnBlock::new(&lw, &cfg, il, rope.clone(), attn_weights)?)
                }
                LayerKind::Linear => Mixer::Linear(LinearAttnBlock::new(&lw, &cfg, attn_weights)?),
            };
            let ffn = match cfg.arch {
                Arch::Dense => Ffn::Dense(DenseMlp::new(&lw)?),
                Arch::Moe | Arch::Qwen4Exp => Ffn::Moe(MoeBlock::new(&lw, &cfg, runner)?),
            };
            layers.push(Layer {
                attn_norm: if hc {
                    None
                } else {
                    Some(lw.rms_norm("attn_norm", cfg.rms_eps)?)
                },
                mixer,
                // There is no `ffn_norm` tensor; `post_attention_norm` is the
                // pre-MLP norm.
                ffn_norm: if hc {
                    None
                } else {
                    Some(lw.rms_norm("post_attention_norm", cfg.rms_eps)?)
                },
                ffn,
            });
            caches.push(LayerCache::new(&cfg, il, kv_slots, &device)?);
        }

        // The hyper-connection gates, the per-attention-layer QSA indexers with
        // their raw-key caches, and the one PLE layer with its conv window and
        // n-gram history. Their presence is what routes `run_stack` to the
        // second graph.
        let qwen4exp = if hc {
            Some(crate::qwen4exp::stack::Qwen4ExpParts::load(
                &w, &gguf, &cfg, &rope, max_ctx, &device,
            )?)
        } else {
            None
        };

        // Attention decode-projection path, for dump provenance. XWEN_ATTN_F32
        // routes the whole block through the dequant-f32 QMatMul ("f32-bypass",
        // like attn_mm). Otherwise a q8_0-attention checkpoint runs the vendored
        // decode gemv ("q8") unless XWEN_ATTN_DEQUANT forces the f16 plane, and
        // an f16-attention checkpoint always decodes through the f16 gemv ("f16").
        let attn_decode = if attn_dtype == DType::F32 {
            "f32-bypass"
        } else if !crate::ops::attn_dequant()
            && layers.iter().any(|l| match &l.mixer {
                Mixer::Full(attn) => attn.uses_q8_decode(),
                Mixer::Linear(_) => false,
            })
        {
            "q8"
        } else {
            "f16"
        };

        // No `output_norm` on qwen4exp: `Qwen4ExpParts::output_hc` is its final
        // normalization, and reading a tensor that does not exist would fail
        // the load of a perfectly good file.
        let output_norm = if hc {
            None
        } else {
            Some(w.rms_norm("output_norm", cfg.rms_eps)?)
        };
        let (lm_head, lm_head_buffer, lm_head_dtype) = w.qlinear_with_buffer("output")?;

        // Batch-register every mmap weight view in candle's queue-attached
        // residency set (one commit per shard mapping); MmapSource::drop
        // unregisters them, so load→drop cycles are leak-free.
        for src in gguf.mmap_sources() {
            src.register_views();
        }
        warn_if_over_budget(&gguf, &cfg, kv_slots, max_ctx);

        Ok(Self {
            cfg,
            device,
            embed,
            layers,
            caches,
            qwen4exp,
            output_norm,
            lm_head,
            lm_head_buffer,
            lm_head_dtype,
            attn_dtype,
            attn_mm,
            attn_decode,
            max_ctx,
            kv_slots,
            tap_enabled: false,
            taps: Vec::new(),
            spec_tap_layers: None,
            spec_taps: Vec::new(),
            keep_post_norm: false,
            post_norm_hidden: None,
            profile: crate::ops::stack_profile().then(crate::stack_profile::StackProfiler::new),
            _weights_mmap: gguf.mmap_sources(),
            checkpoint: gguf.checkpoint_id(),
        })
    }

    pub fn config(&self) -> &XwenConfig {
        &self.cfg
    }

    /// Identity of the checkpoint this model was loaded from — what a persisted
    /// cache image is stamped with and validated against.
    pub fn checkpoint_id(&self) -> crate::gguf::CheckpointId {
        self.checkpoint
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// The attention WEIGHT dtype `load` resolved: `F16` (dense f16 weight
    /// planes — GGUF-stored f16 or dequantized from q8_0 storage — through
    /// the vendored mixed-dtype kernels; the shipped default)
    /// or `F32` (dequantized dense f32, the legacy path selected by
    /// `XWEN_ATTN_F32`). Activations are f32 in both modes.
    pub fn attn_dtype(&self) -> DType {
        self.attn_dtype
    }

    /// The attention PREFILL gemm path `load` resolved, for dump provenance:
    /// "tensor" (the shipped Metal-4 cooperative-tensor default), "classic" (the
    /// simdgroup kernel, under the `XWEN_ATTN_MM_CLASSIC` kill-switch), or
    /// "f32-bypass" (`XWEN_ATTN_F32` — the f16 library, and thus its mm
    /// branch, is bypassed entirely).
    pub fn attn_mm(&self) -> &'static str {
        self.attn_mm
    }

    /// The attention DECODE-projection path `load` resolved, for dump provenance:
    /// "f32-bypass" (XWEN_ATTN_F32), "q8" (a q8_0-attention checkpoint's
    /// vendored decode gemv), or "f16" (the dense f16 gemv — the official
    /// checkpoint, or a q8_0 file under XWEN_ATTN_DEQUANT).
    pub fn attn_decode(&self) -> &'static str {
        self.attn_decode
    }

    /// Ask every PLE layer to fault in the n-gram table rows a LATER forward
    /// over `tokens` will gather, given the state each layer is in right now.
    ///
    /// A no-op on every architecture but qwen4exp, and advisory even there: the
    /// hint moves no state and is dropped outright if the prefetch thread is
    /// behind, so a caller can issue it for a token that an EOG or a rejected
    /// speculative block then discards. The gather it is running ahead of is
    /// 16 unrelated 90-byte reads over a 28.8 GB mapping, i.e. page faults
    /// rather than arithmetic (`qwen4exp::ple::PlePrefetcher`). The rows are
    /// chosen from the token history alone and NEVER gated on the PLE gate
    /// value — the gate is a mid-forward quantity, so waiting for it would put
    /// the prefetch behind the gather it exists to run ahead of
    /// (`qwen4exp::ple::PleLayer::prefetch`).
    ///
    /// Callers: the decode loop, the moment a token is sampled and before the
    /// forward that consumes it starts, and the qwen4exp prefill chunk.
    pub fn ple_prefetch(&self, tokens: &[u32]) {
        let Some(parts) = self.qwen4exp.as_ref() else {
            return;
        };
        for layer in &parts.layers {
            if let (Some(ple), Some(state)) = (layer.ple.as_ref(), layer.ple_state.as_ref()) {
                ple.prefetch(state.history(), tokens);
            }
        }
    }

    /// Run the transformer stack (embedding → 48 layers → final norm) and return
    /// the post-final-norm hidden states `[seq, hidden]` for EVERY position,
    /// together with the per-layer taps collected when capture is enabled.
    /// Shared by `forward` (which narrows to the last position for the lm head)
    /// and `forward_all_logits` (which keeps every position). Advances the KV
    /// caches, so callers feeding chunks must pass a monotonically increasing
    /// `pos`.
    fn run_stack(&mut self, tokens: &Tensor, pos: usize) -> Result<StackOutput> {
        // qwen4exp is a second graph over the same blocks (D14): a 4-stream
        // residual carrier, a QSA overlay on the attention layers and a PLE
        // injection, none of which fit as a branch inside the loop below.
        if self.qwen4exp.is_some() {
            return crate::qwen4exp::stack::run_stack_hc(self, tokens, pos);
        }

        let seq = tokens.elem_count();
        ensure!(
            pos + seq <= self.max_ctx,
            "context overflow: position {pos} + {seq} tokens exceeds max_ctx {} \
             (raise --max-ctx or shorten the prompt)",
            self.max_ctx
        );
        self.grow_kv_capacity(pos + seq)?;

        // Per-stage timing hooks (`XWEN_STACK_PROFILE`). Off — the normal case —
        // each is one `Option` check; on, each brackets its stage with device
        // syncs so the stage totals and the chunk's own wall clock are measured
        // the same way and their difference is meaningful.
        macro_rules! stage {
            ($stage:expr, $e:expr) => {{
                crate::stack_profile::stage_begin(&mut self.profile, &self.device)?;
                let out = $e;
                crate::stack_profile::stage_end(&mut self.profile, &self.device, $stage)?;
                out
            }};
        }
        crate::stack_profile::chunk_begin(&mut self.profile, &self.device, seq)?;

        // Embedding lookup, upcast to the f32 residual stream (shared with the
        // drafter via `embed_ids`).
        let mut x = stage!(Stage::Embed, self.embed_tokens(tokens)?); // [seq, hidden] f32

        // Taps are collected into a local vec (no self-borrow tangle with the
        // per-layer cache mutation) and published by the caller when enabled.
        let mut taps: Vec<(String, Tensor)> = Vec::new();
        // Lightweight spec-decode taps: cloned per configured layer only. Read
        // the config into a local so the per-layer cache mutation below does not
        // tangle with the immutable borrow of `self.spec_tap_layers`.
        let spec_layers = self.spec_tap_layers.clone();
        let mut spec_captured: Vec<(usize, Tensor)> = Vec::new();
        macro_rules! tap {
            ($name:expr, $il:expr, $t:expr) => {
                if self.tap_enabled {
                    taps.push((format!("{}-{}", $name, $il), $t.clone()));
                }
            };
        }

        // Hoist the prefill mask out of the per-layer loop: it is a pure function
        // of (kind, pos, seq_len), every attention layer here is plain causal at
        // the same head count, and the DeltaNet layers take no mask at all — so
        // one build serves the whole stack instead of one per attention layer.
        // Only at prefill (seq > 1); decode builds no mask and the layers see
        // None. The vendored flash kernel, which would compute the mask
        // in-kernel and skip this allocation entirely, is compiled at head dim
        // 128 and cannot serve this architecture's 256.
        let full_mask = if seq > 1 {
            let n_head = (0..self.cfg.n_layer)
                .find(|&il| self.cfg.is_full_attn(il))
                .map(|il| self.cfg.n_head(il));
            match n_head {
                Some(n) => self.build_prefill_mask(n, seq, pos)?,
                None => None,
            }
        } else {
            None
        };

        for il in 0..self.layers.len() {
            let layer = &self.layers[il];
            let cache = &mut self.caches[il];

            let normed = stage!(
                Stage::AttnNorm,
                norm_of(&layer.attn_norm, "attn_norm")?.forward(&x)?
            );
            tap!("attn_norm", il, normed);

            // x += mixer(attn_norm(x)) — post-o_proj for an attention layer,
            // post-ssm_out for a DeltaNet one. Both advance their layer's cache.
            let attn = match &layer.mixer {
                Mixer::Full(block) => stage!(
                    Stage::MixerFullAttn,
                    block.forward(&normed, cache, pos, full_mask.as_ref(), None)?
                ),
                Mixer::Linear(block) => {
                    stage!(Stage::MixerDelta, block.forward(&normed, cache)?)
                }
            };
            tap!("attn_o_proj", il, attn);
            let ffn_inp = stage!(Stage::ResidualAttn, (&x + &attn)?);
            tap!("ffn_inp", il, ffn_inp);

            let ffn_normed = stage!(
                Stage::FfnNorm,
                norm_of(&layer.ffn_norm, "post_attention_norm")?.forward(&ffn_inp)?
            );
            tap!("ffn_norm", il, ffn_normed);

            // x += ffn(ffn_norm(x)).
            let ffn_out = match &layer.ffn {
                Ffn::Dense(mlp) => stage!(Stage::Ffn, mlp.forward(&ffn_normed)?),
                Ffn::Moe(moe) => stage!(Stage::Ffn, moe.forward(&ffn_normed)?),
            };
            tap!("ffn_out", il, ffn_out);

            x = stage!(Stage::ResidualFfn, (&ffn_inp + &ffn_out)?);
            tap!("l_out", il, x);

            // Spec-decode capture: the same post-FFN-residual `l_out` value, held
            // as a cheap Arc clone (f32 already — the residual stream never leaves
            // f32 on the fused path). Ordered into configured order by the caller.
            if let Some(layers) = &spec_layers {
                if layers.contains(&il) {
                    spec_captured.push((il, x.clone()));
                }
            }
        }

        // Pre-final-norm residual stream (DFlash drafter's last capture point).
        if self.tap_enabled {
            taps.push(("h_nextn".to_string(), x.clone()));
        }

        let normed = stage!(
            Stage::FinalNorm,
            norm_of(&self.output_norm, "output_norm")?.forward(&x)?
        ); // [seq, hidden]
        // The MTP draft head consumes the POST-final-norm hidden, per position —
        // deliberately not the pre-norm residual the DFlash taps above capture.
        // A handle clone when armed, an `Option` check when not; the tensor is
        // returned unchanged either way, so no forward computes anything
        // differently for being observed.
        if self.keep_post_norm {
            self.post_norm_hidden = Some(normed.clone());
        }
        Ok((normed, taps, spec_captured))
    }

    /// Build the hoisted prefill mask, splitting the CPU fill from the upload so
    /// the profiler can charge them to separate stages. The tensors are what
    /// `PrefillMask::build` produces either way.
    pub(crate) fn build_prefill_mask(
        &mut self,
        n_head: usize,
        seq: usize,
        pos: usize,
    ) -> Result<Option<PrefillMask>> {
        if self.profile.is_none() {
            return PrefillMask::build(MaskKind::Full, n_head, seq, pos, &self.device);
        }
        let started = crate::stack_profile::host_begin(&mut self.profile);
        let host = crate::kv_cache::attn_mask_data(MaskKind::Full, seq, pos);
        crate::stack_profile::host_end(&mut self.profile, Stage::MaskFillHost, started);
        let Some(host) = host else { return Ok(None) };

        crate::stack_profile::stage_begin(&mut self.profile, &self.device)?;
        let mask = PrefillMask::from_host(host, n_head, &self.device)?;
        crate::stack_profile::stage_end(
            &mut self.profile,
            &self.device,
            Stage::MaskUploadAndBroadcast,
        )?;
        Ok(Some(mask))
    }

    /// tokens: [seq] u32 at absolute position pos. Returns last-position
    /// logits [vocab] f32.
    pub fn forward(&mut self, tokens: &Tensor, pos: usize) -> Result<Tensor> {
        let seq = tokens.elem_count();
        let (normed, mut taps, spec_captured) = self.run_stack(tokens, pos)?;
        self.publish_spec_taps(spec_captured);

        // Final norm over the full sequence, then the lm head on the LAST
        // position only — never run the vocab matmul over the whole prefill
        // chunk. `result_norm` matches the fork, which captures it after the
        // last-position gather, so it is last-position-only too.
        crate::stack_profile::stage_begin(&mut self.profile, &self.device)?;
        let last = normed.narrow(0, seq - 1, 1)?.contiguous()?; // [1, hidden]
        // Decode bypass at one query position; a prefill chunk (seq > 1) keeps
        // the QMatMul path, which is the numerics the whole prefill shares.
        let logits = if seq == 1 {
            self.lm_head_row(&last)?
        } else {
            self.lm_head(&last)? // [1, vocab] — shared raw projection
        };
        let logits = logits.flatten_all()?; // [vocab]
        crate::stack_profile::stage_end(&mut self.profile, &self.device, Stage::LmHead)?;
        crate::stack_profile::chunk_end(&mut self.profile, &self.device)?;
        // One `XWEN_GDN_PROFILE` line per forward, holding every DeltaNet
        // layer's sub-steps folded together. Here rather than in the block
        // because the block does not know where a forward ends, and a line per
        // layer would be 36 lines a token.
        crate::gdn_profile::report();
        if self.tap_enabled {
            taps.push(("result_norm".to_string(), last));
            taps.push(("result_output".to_string(), logits.clone()));
            self.taps = taps;
        }

        Ok(logits)
    }

    /// All-position logits `[seq, vocab]` f32 for offline scoring (perplexity
    /// parity). Runs the identical transformer stack as `forward` but keeps the
    /// lm head over EVERY position instead of narrowing to the last, so the
    /// caller can gather a next-token log-probability at every prefill position.
    ///
    /// The lm head runs through the plain QMatMul path (`lm_head.forward`) over
    /// the full `[seq, hidden]` — the same path `forward` uses for a seq > 1
    /// prefill chunk — so this shares the default prefill numerics and never
    /// touches the decode-only vendored mat-vec bypass. Offline tooling only;
    /// `forward`/`generate` are unaffected. Advances the KV caches like
    /// `forward`, so a chunked continuous pass must feed a monotonic `pos`.
    pub fn forward_all_logits(&mut self, tokens: &Tensor, pos: usize) -> Result<Tensor> {
        let (normed, _taps, spec_captured) = self.run_stack(tokens, pos)?;
        self.publish_spec_taps(spec_captured);
        // This method publishes NO parity taps: the set `run_stack` produced is
        // incomplete here (`result_norm`/`result_output` are last-position-only and
        // belong to `forward`), so handing it to `take_taps` would misreport. Drop
        // the previous forward's taps rather than leave them readable — otherwise a
        // `take_taps()` after this call silently returns a stale, wrong-position
        // capture instead of nothing. Spec taps ARE published above: those are
        // whole-sequence `l_out` residuals, valid at every position, and the
        // speculative verify path depends on them.
        self.taps.clear();
        crate::stack_profile::stage_begin(&mut self.profile, &self.device)?;
        let logits = self.lm_head(&normed)?; // [seq, vocab] — QMatMul path (seq > 1)
        crate::stack_profile::stage_end(&mut self.profile, &self.device, Stage::LmHead)?;
        crate::stack_profile::chunk_end(&mut self.profile, &self.device)?;
        crate::gdn_profile::report();
        Ok(logits)
    }

    /// Report the per-stage forward timing collected under `XWEN_STACK_PROFILE`,
    /// tagging every line with `label` (the call site — a generation reports once
    /// after prefill and once at the end). No-op when profiling is off. The
    /// accumulators are cumulative and per-phase, so a later dump repeats the
    /// earlier one's phases unchanged and adds whatever has run since.
    pub fn dump_stack_profile(&self, label: &str) {
        if let Some(p) = &self.profile {
            p.dump(label);
        }
    }

    /// Declare which phase the forwards from here on belong to, for the
    /// `XWEN_STACK_PROFILE` accumulators. No-op when profiling is off. Only the
    /// generation loop knows this: neither the token count nor the entry point
    /// distinguishes a one-token tail of a prompt from a decode step, or a
    /// speculative verify span from a prefill chunk. Unset means prefill, which
    /// is what a pure scoring pass wants.
    pub fn set_phase(&mut self, phase: crate::stack_profile::Phase) {
        if let Some(p) = self.profile.as_mut() {
            p.set_phase(phase);
        }
    }

    /// Drop everything the `XWEN_STACK_PROFILE` accumulators hold. No-op when
    /// profiling is off. A throwaway warm-up pass calls this so the page-in and
    /// pipeline-compile costs it exists to absorb stay out of the report.
    pub fn reset_stack_profile(&mut self) {
        if let Some(p) = self.profile.as_mut() {
            p.reset();
        }
    }

    /// Enable capture of named intermediate tensors (parity bisection).
    pub fn set_tap_capture(&mut self, enabled: bool) {
        self.tap_enabled = enabled;
        if !enabled {
            self.taps.clear();
        }
    }

    pub fn take_taps(&mut self) -> Vec<(String, Tensor)> {
        std::mem::take(&mut self.taps)
    }

    /// Configure the lightweight spec-decode taps: `Some(layers)` captures each
    /// listed layer's `l_out` residual on subsequent forwards (in the given
    /// order); `None` disables capture and drops any pending taps. Unset is the
    /// default and costs a single `Option` check per layer.
    ///
    /// Indices are `l_out` (post-FFN residual) layer indices and must be in
    /// range — note the drafter's `dflash.target_layers` names layer INPUTS,
    /// so use `DflashConfig::spec_tap_layers()` for the `t - 1` translation
    /// rather than wiring `target_layers` through raw.
    pub fn set_spec_taps(&mut self, layers: Option<Vec<usize>>) {
        if let Some(layers) = &layers {
            for &il in layers {
                assert!(
                    il < self.layers.len(),
                    "spec tap layer {il} out of range (model has {} layers); \
                     dflash target_layers must be translated via DflashConfig::spec_tap_layers()",
                    self.layers.len()
                );
            }
        }
        self.spec_tap_layers = layers;
        // Unconditional: a Some -> Some reconfigure must not leave the old
        // config's tensors readable until the next forward.
        self.spec_taps.clear();
    }

    /// Drain the most recent forward's spec taps, one `f32 [seq, hidden]` tensor
    /// per configured layer, in the order `set_spec_taps` was given. Empty if no
    /// forward has run since taps were configured (or if taps are unset).
    pub fn take_spec_taps(&mut self) -> Vec<Tensor> {
        std::mem::take(&mut self.spec_taps)
    }

    /// Retain the post-final-norm hidden on every forward, for the MTP draft
    /// head. Its `h` input is the tensor AFTER `output_norm` — the one the trunk
    /// otherwise feeds straight to its lm_head — which is a different tensor from
    /// the pre-norm residual [`XwenModel::set_spec_taps`] captures, and using the
    /// wrong one drafts plausible noise rather than failing.
    pub fn set_keep_post_norm(&mut self, on: bool) {
        self.keep_post_norm = on;
        // Unconditional, like the spec-tap reconfigure: turning capture off must
        // not leave the last forward's tensor readable.
        self.post_norm_hidden = None;
    }

    /// Drain the most recent forward's post-final-norm hidden `[seq, hidden]`
    /// f32, or `None` if capture is off or no forward has run since it was armed.
    pub fn take_post_norm_hidden(&mut self) -> Option<Tensor> {
        self.post_norm_hidden.take()
    }

    /// Reorder a forward's loop-captured spec taps into the configured order and
    /// stash them for `take_spec_taps`. No-op when taps are unset.
    fn publish_spec_taps(&mut self, captured: Vec<(usize, Tensor)>) {
        let ordered = match &self.spec_tap_layers {
            Some(cfg) => order_spec_taps(cfg, captured),
            None => return,
        };
        self.spec_taps = ordered;
    }

    /// Gather token embeddings for `ids`, upcast to the f32 residual stream —
    /// `[ids.len(), hidden]`. The drafter shares the target's embeddings through
    /// this; identical to the lookup `run_stack` runs at the top of a forward.
    pub fn embed_ids(&self, ids: &[u32]) -> Result<Tensor> {
        let tokens = Tensor::new(ids, &self.device)?;
        self.embed_tokens(&tokens)
    }

    /// [`XwenModel::embed_ids`] for ids that are already a device tensor `[n]`.
    ///
    /// What this buys over the slice form is that the ids never have to be
    /// known on the host. A draft chain picks each step's token with a device
    /// argmax and immediately needs that token's embedding row for the next
    /// step; going through `embed_ids` would mean reading the id back, and a
    /// per-step readback is a per-step device sync — the cost the whole chain
    /// is shaped to avoid.
    pub fn embed_rows(&self, ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens(ids)
    }

    /// The embedding lookup `run_stack` uses: gather rows and upcast to f32.
    pub(crate) fn embed_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
        let tokens = tokens.to_dtype(DType::U32)?;
        Ok(self.embed.index_select(&tokens, 0)?.to_dtype(DType::F32)?)
    }

    /// The RAW output projection: `[seq, hidden] -> [seq, vocab]`, no final norm
    /// (the caller applies `output_norm` itself). This is the QMatMul/dequant
    /// path `forward` uses for a prefill chunk and `forward_all_logits` uses for
    /// every position; the drafter shares the target's lm_head through it. The
    /// decode-only vendored mat-vec bypass stays inline in `forward` (a perf
    /// path, numerically equivalent).
    pub fn lm_head(&self, h: &Tensor) -> Result<Tensor> {
        // Offset/non-contiguous views are materialized inside QLinear::forward
        // (the Metal quantized matmul silently drops the input start_offset).
        Ok(self.lm_head.forward(h)?)
    }

    /// The output projection at ONE query position: `[1, hidden] -> [1, vocab]`,
    /// no final norm.
    ///
    /// This is the vendored ggml-geometry plain mat-vec over the shared lm_head
    /// buffer — candle's baked quantized mat-vec kernels run ~15x under
    /// bandwidth here, measured on the retired q6_K lm_head — falling back to
    /// [`XwenModel::lm_head`] off Metal, on an unsupported dtype, or under
    /// `XWEN_MV_CLASSIC`. Numerically equivalent to that fallback, so which one
    /// runs is a perf question and not an output one.
    ///
    /// Decode reads it through `forward`; the MTP draft chain reads it directly,
    /// because a chain step is exactly one query position and going through the
    /// quantized matmul instead would spend the chain's whole budget on the
    /// vocab projection.
    pub fn lm_head_row(&self, h: &Tensor) -> Result<Tensor> {
        match &self.lm_head_buffer {
            Some(buf)
                if !crate::ops::mv_classic()
                    && crate::ops::mv_vendored_supported(self.lm_head_dtype) =>
            {
                Ok(crate::ops::mul_mv(
                    buf,
                    self.lm_head_dtype,
                    self.lm_head.out_dim,
                    self.lm_head.in_dim,
                    h,
                )?)
            }
            _ => self.lm_head(h),
        }
    }

    /// Snapshot the KV caches BEFORE a `span`-token verify forward, so a partial
    /// accept can roll back to the committed prefix. `span` must be >= 1 and fit
    /// the remaining context. See `kv_rollback`.
    /// Takes `&mut self` because a DeltaNet layer cannot be checkpointed by
    /// copying anything: its state is overwritten every step, so the checkpoint
    /// ARMS the layer and the verify forward records the state after each token
    /// as it runs. See `LayerCache::checkpoint`.
    pub fn kv_checkpoint(&mut self, span: usize) -> Result<KvCheckpoint> {
        ensure!(span >= 1, "kv_checkpoint: span must be >= 1");
        let len0 = self.caches.first().map(LayerCache::len).unwrap_or(0);
        ensure!(
            len0 + span <= self.max_ctx,
            "kv_checkpoint: len {len0} + span {span} exceeds max_ctx {}",
            self.max_ctx
        );
        let layers: Vec<LayerCheckpoint> = self
            .caches
            .iter_mut()
            .map(|c| c.checkpoint(span))
            .collect::<Result<_>>()?;
        // The qwen4exp indexer caches and PLE state arm alongside, and are
        // rolled back from `kv_rollback` — they are not carried in
        // `KvCheckpoint` because `kv_cache.rs` stays out of P2 (D15).
        if let Some(parts) = self.qwen4exp.as_mut() {
            parts.checkpoint(len0, span)?;
        }
        Ok(KvCheckpoint::new(len0, span, layers))
    }

    /// Roll the KV caches back to `len0 + commit` (0 <= commit <= the
    /// checkpoint's span), where `len0` is the length captured by `kv_checkpoint`.
    /// The restore is bit-exact: every layer holds exactly the bytes it recorded
    /// when the committed tokens were written in THIS run. That is deliberately
    /// weaker than bitwise identity with a differently-partitioned run over the
    /// same tokens: projections switch weight representation across the
    /// `Q8_DECODE_MAX_SEQ` batch boundary (dual storage, see gguf.rs), so a
    /// verify batch of 9+ tokens writes state that differs in low bits from
    /// one-token decode. Cross-partition agreement is numeric (parity-gated),
    /// not bitwise; `XWEN_ATTN_DEQUANT` pins the single f16 representation when
    /// bitwise partition-independence is required.
    pub fn kv_rollback(&mut self, ckpt: &KvCheckpoint, commit: usize) -> Result<()> {
        ensure!(
            commit <= ckpt.span(),
            "kv_rollback: commit {commit} exceeds span {}",
            ckpt.span()
        );
        ensure!(
            ckpt.layers.len() == self.caches.len(),
            "kv_rollback: checkpoint covers {} layers, model has {}",
            ckpt.layers.len(),
            self.caches.len()
        );
        for (cache, lc) in self.caches.iter_mut().zip(&ckpt.layers) {
            cache.rollback(lc, ckpt.len0, ckpt.span, commit)?;
        }
        if let Some(parts) = self.qwen4exp.as_mut() {
            parts.rollback(ckpt.len0, ckpt.span, commit)?;
        }
        Ok(())
    }

    /// Tokens currently committed to the KV cache. Every layer is driven in
    /// lockstep, so any layer's length is the model's.
    pub fn cache_len(&self) -> usize {
        self.caches.first().map(LayerCache::len).unwrap_or(0)
    }

    /// Capture the cache state so a later `restore_cache_snapshot` returns to
    /// exactly the `cache_len()` tokens committed right now. Deep-copies the SWA
    /// rings (~72 MiB on the shipped checkpoint, independent of context length);
    /// full-attention layers need no data, their positions keep their slots.
    pub fn take_cache_snapshot(&self) -> Result<CacheSnapshot> {
        let mut layers: Vec<LayerSnapshot> = self
            .caches
            .iter()
            .map(LayerCache::snapshot)
            .collect::<Result<_>>()?;
        // On qwen4exp the PLE injection layer carries a second recurrent state
        // that lives in `Qwen4ExpParts` rather than in a `LayerCache` — the
        // dilated conv window and the n-gram token history, neither of which has
        // an inverse. It rides on that layer's own snapshot entry, so the vector
        // still has one entry per layer and the pairing needs no layer ids. The
        // QSA indexers carry nothing here: their raw keys are position-indexed
        // like a full-attention layer's K/V, so a restore truncates them.
        if let Some(parts) = self.qwen4exp.as_ref() {
            let images = parts.ple_images();
            ensure!(
                images.len() == layers.len(),
                "take_cache_snapshot: the qwen4exp parts cover {} layers, the cache stack has {}",
                images.len(),
                layers.len()
            );
            for (layer, image) in layers.iter_mut().zip(images) {
                if let Some(image) = image {
                    let taken = std::mem::replace(layer, LayerSnapshot::Full);
                    *layer = taken.with_ple(image)?;
                }
            }
        }
        Ok(CacheSnapshot::new(self.cache_len(), layers))
    }

    /// Rewind the cache to `snapshot`. SWA rings are overwritten with the
    /// captured contents; full-attention layers are shortened to the snapshot's
    /// position, which restores them only if their slots below that position
    /// still hold the same keys — i.e. the caller must not have rewritten
    /// positions `[0, pos)` with a different token sequence since. Callers that
    /// diverge inside the snapshot's prefix must reset the cache instead.
    pub fn restore_cache_snapshot(&mut self, snapshot: &CacheSnapshot) -> Result<()> {
        ensure!(
            snapshot.layers().len() == self.caches.len(),
            "restore_cache_snapshot: snapshot covers {} layers, model has {}",
            snapshot.layers().len(),
            self.caches.len()
        );
        ensure!(
            snapshot.pos() <= self.max_ctx,
            "restore_cache_snapshot: pos {} exceeds max_ctx {}",
            snapshot.pos(),
            self.max_ctx
        );
        // Asked of every half before ANY of them is written, and that has to
        // include the trunk layers themselves. `LayerCache::restore` re-checks
        // each layer inline, which refuses the bad layer but not the ones before
        // it: a kind mismatch at layer five leaves four already overwritten and
        // the rest holding the previous conversation, at lengths that still
        // agree. Nothing downstream can see that, so the preflight is a separate
        // immutable pass over the same rule rather than a comment claiming the
        // inline checks add up to one.
        self.check_ple_images(&snapshot.ple_shapes(), snapshot.pos())?;
        for (cache, layer) in self.caches.iter().zip(snapshot.layers()) {
            cache.check_restorable(layer, snapshot.pos())?;
        }
        if let Some(parts) = self.qwen4exp.as_ref() {
            parts.check_restore_at(snapshot.pos())?;
        }
        for (cache, layer) in self.caches.iter_mut().zip(snapshot.layers()) {
            cache.restore(layer, snapshot.pos())?;
        }
        if let Some(parts) = self.qwen4exp.as_mut() {
            parts.restore(&snapshot.ple_images(), snapshot.pos())?;
        }
        Ok(())
    }

    /// Copy the full-attention layers' committed rows `[0, cache_len())` to host
    /// RAM. Together with a `take_cache_snapshot().to_host()` this is the whole
    /// cache state of the current conversation, so the GPU stack can be handed to
    /// a different one and this conversation resumed later without re-prefilling.
    ///
    /// Cost is 48 KiB per token on the shipped checkpoint (see
    /// `HostFullKv::byte_len`) plus the device-to-host copy; unlike a snapshot it
    /// scales with context length.
    pub fn export_full_kv(&self) -> Result<HostFullKv> {
        let qsa = self
            .qwen4exp
            .as_ref()
            .map(|parts| parts.indexer_caches())
            .unwrap_or_default();
        crate::kv_cache::export_full_kv_from(&self.caches, &qsa)
    }

    /// Whether this model could take `image` and `rings` at `pos`, decided without
    /// changing anything.
    ///
    /// Every rule the two imports below apply, asked in advance. A caller that has to
    /// give the cache to somebody else before it can attempt an import — the server
    /// pages the resident conversation out first — would otherwise learn that the state
    /// does not fit only once the conversation it displaced is gone, and a restore that
    /// fails at layer five has already overwritten four. The checks come from the import
    /// paths themselves rather than a second copy of the geometry.
    pub fn check_importable(
        &self,
        image: &HostFullKv,
        rings: &HostSnapshot,
        pos: usize,
    ) -> Result<()> {
        ensure!(
            pos <= self.max_ctx,
            "check_importable: pos {pos} exceeds max_ctx {}",
            self.max_ctx
        );
        let qsa = self
            .qwen4exp
            .as_ref()
            .map(|parts| parts.indexer_caches())
            .unwrap_or_default();
        crate::kv_cache::check_full_kv_importable(&self.caches, &qsa, image, pos)?;
        rings.check_restorable(&self.caches, pos)?;
        self.check_ple_images(&rings.ple_shapes(), pos)?;
        Ok(())
    }

    /// Upload `image`'s rows `[0, pos)` into every full-attention layer and set
    /// their length to `pos`. `pos` may be shorter than the image (resuming a
    /// conversation at an earlier turn boundary keeps the longer image intact).
    ///
    /// The SWA rings are NOT touched — they cannot be reconstructed from a row
    /// image, so the caller restores them from the `HostSnapshot` taken at the
    /// same `pos`, and until it does the cache stack is inconsistent.
    ///
    /// This is the write side of the invariant `restore_cache_snapshot` documents:
    /// full-attention layers hold each position in its own slot, so a cache is
    /// only correct at length `pos` if slots `[0, pos)` hold that conversation's
    /// keys. Importing establishes exactly that, which is why a snapshot restore
    /// is sound after another conversation has run over those slots.
    pub fn import_full_kv(&mut self, image: &HostFullKv, pos: usize) -> Result<()> {
        // Only the positions actually being imported have to fit: an image longer
        // than this model's context is still a valid source for a shorter resume
        // (an on-disk record written by a larger-context server, say).
        ensure!(
            pos <= self.max_ctx,
            "import_full_kv: pos {pos} exceeds max_ctx {}",
            self.max_ctx
        );
        // A paged-in conversation can be longer than anything this instance has
        // run yet, so the import grows the buffers exactly as a prefill would.
        self.grow_kv_capacity(pos)?;
        // The QSA indexer planes travel with the K/V rows and are imported with
        // them: their rows are position-indexed, so they are the half of the
        // qwen4exp state that grows with the conversation and cannot be carried
        // by a fixed-size snapshot.
        let mut qsa = self
            .qwen4exp
            .as_mut()
            .map(|parts| parts.indexer_caches_mut())
            .unwrap_or_default();
        crate::kv_cache::import_full_kv_into(&mut self.caches, &mut qsa, image, pos)
    }

    /// Whether the PLE images a snapshot carries — one entry per layer, `None`
    /// where a layer carries none — describe states this model could hold.
    ///
    /// The `None` arm is the one worth spelling out: a model with no PLE layers
    /// at all must REFUSE an image that carries one rather than quietly ignore
    /// it. A dropped PLE window is not a missing optimization, it is a
    /// conversation resuming mid-n-gram against a window of somebody else's
    /// activations — which runs, answers, and attends to the wrong tokens. The
    /// checkpoint binding makes this unreachable through the disk tier, and no
    /// in-process path can cross two architectures either; it is checked anyway
    /// because the cost of being wrong here is silent.
    fn check_ple_images(
        &self,
        shapes: &[Option<crate::qwen4exp::ple::PleShape>],
        pos: usize,
    ) -> Result<()> {
        match self.qwen4exp.as_ref() {
            Some(parts) => parts.check_ple_restorable(shapes, pos),
            None => {
                ensure!(
                    shapes.iter().all(Option::is_none),
                    "cache image: the snapshot carries a PLE state, this model has no PLE layer"
                );
                Ok(())
            }
        }
    }

    pub fn reset_cache(&mut self) -> Result<()> {
        for cache in &mut self.caches {
            cache.reset()?;
        }
        if let Some(parts) = self.qwen4exp.as_mut() {
            parts.reset();
        }
        Ok(())
    }

    /// Grow every full-attention layer to hold at least `needed` positions,
    /// `max_ctx` staying the hard ceiling (the callers' own overflow checks run
    /// first, so hitting the ceiling here is a bug, not a user error). Growth is
    /// lockstep across layers and monotonic for the model's lifetime — nothing
    /// shrinks a cache but dropping the model — and each step is logged because
    /// it is a real memory event the operator sized the machine around.
    ///
    /// An allocation failure partway (a device OOM growing toward a large
    /// ceiling) leaves some layers grown and others not, and that state is
    /// SAFE: `kv_slots` is only advanced after every layer succeeded, each
    /// layer's own `ensure_full_capacity` re-checks its real allocation and is
    /// idempotent, so a retried forward re-runs the growth and converges —
    /// already-grown layers no-op, the failed one retries.
    pub(crate) fn grow_kv_capacity(&mut self, needed: usize) -> Result<()> {
        if needed <= self.kv_slots {
            return Ok(());
        }
        let mut grown = self.kv_slots;
        for cache in &mut self.caches {
            grown = grown.max(cache.ensure_full_capacity(needed, self.max_ctx)?);
        }
        // Candle's Metal pool frees dropped buffers only at a device sync, so
        // without one every layer's OLD allocation stays resident beside its
        // replacement until whenever the next sync happens to run — ~1.5x the
        // new KV size at the top doubling step. Growth is rare (O(log) per
        // lifetime), so one sync here is cheap and bounds the peak; it also
        // waits for the copy blits above, which it must anyway.
        if let Device::Metal(mdev) = &self.device {
            mdev.wait_until_completed()?;
        }
        self.kv_slots = grown.max(needed);
        crate::host_log::host_line(format!(
            "xwen: KV cache grew to {} of {} positions ({:.1}GB)",
            self.kv_slots,
            self.max_ctx,
            gb(kv_bytes(&self.cfg, self.kv_slots)),
        ));
        Ok(())
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// The norm a qwen35/qwen35moe layer must have.
///
/// These three tensors exist on every architecture this stack runs and on none
/// that `run_stack_hc` runs, so the `Option` never fails here — it is what lets
/// one `Layer` type carry both graphs' layers. An error rather than an unwrap so
/// a future architecture that reaches the wrong loop says which tensor it was
/// missing instead of panicking mid-forward.
fn norm_of<'a>(norm: &'a Option<RmsNorm>, name: &str) -> Result<&'a RmsNorm> {
    norm.as_ref()
        .with_context(|| format!("this layer carries no {name}: it belongs to the qwen4exp graph"))
}

/// Reorder the forward's loop-captured `(layer, l_out)` spec taps (captured in
/// ascending layer order) into the caller's configured order. A configured layer
/// with no capture is skipped (it never happens — every configured layer runs
/// every forward); a repeated layer yields the same handle again.
fn order_spec_taps(config: &[usize], captured: Vec<(usize, Tensor)>) -> Vec<Tensor> {
    config
        .iter()
        .filter_map(|&il| {
            captured
                .iter()
                .find(|(c, _)| *c == il)
                .map(|(_, t)| t.clone())
        })
        .collect()
}

/// KV bytes at `slots` allocated positions: k and v, f16 (2 bytes),
/// `[n_kv_head, slots, head_dim]` per full-attention layer. A DeltaNet layer
/// has no K/V.
fn kv_bytes(cfg: &XwenConfig, slots: usize) -> u64 {
    let n_full = (0..cfg.n_layer).filter(|&il| cfg.is_full_attn(il)).count() as u64;
    n_full * 2 * cfg.n_kv_head as u64 * slots as u64 * cfg.head_dim as u64 * 2
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Sum the resident bytes (weights + KV cache + recurrent state), say what the
/// KV can grow to, and warn if the worst case clears the budget. Only
/// full-attention layers hold context-scaled state, and they allocate lazily
/// (`grow_kv_capacity`), so what is resident at load is the initial allocation;
/// `max_ctx` is the ceiling a long conversation can grow it to.
fn warn_if_over_budget(gguf: &GgufFile, cfg: &XwenConfig, kv_slots: usize, max_ctx: usize) {
    let tensor_bytes = |info: &candle_core::quantized::gguf_file::TensorInfo| -> u64 {
        let elems = info.shape.elem_count() as u64;
        let dt = info.ggml_dtype;
        elems / dt.block_size() as u64 * dt.type_size() as u64
    };
    let weight_bytes: u64 = gguf.content.tensor_infos.values().map(tensor_bytes).sum();

    // The PLE n-gram table is the one weight nothing uploads: its rows are
    // hashed and gathered from the host mapping, sixteen per token (D17). At
    // 28.8 GB on the shipped file it would dominate — and misreport — a
    // "resident" figure, so it is reported as what it is.
    let ple_table_bytes = gguf
        .content
        .tensor_infos
        .get("per_layer_token_embd.weight")
        .map(tensor_bytes)
        .unwrap_or(0);
    let weight_bytes = weight_bytes - ple_table_bytes;

    // Conv window + delta state, f32, per DeltaNet layer — context-independent.
    let n_full = (0..cfg.n_layer).filter(|&il| cfg.is_full_attn(il)).count() as u64;
    let n_linear = cfg.n_layer as u64 - n_full;
    let hd = cfg.linear_head_dim as u64;
    let state_bytes = n_linear
        * 4
        * ((cfg.conv_kernel as u64 - 1) * cfg.conv_dim() as u64
            + cfg.linear_v_heads as u64 * hd * hd)
        // qwen4exp only: the QSA raw-key planes (allocated at max_ctx, not
        // grown) and the PLE conv window. Zero on the other checkpoints.
        + crate::qwen4exp::stack::extra_state_bytes(cfg, max_ctx);

    let total = weight_bytes + kv_bytes(cfg, kv_slots) + state_bytes;
    let ceiling = weight_bytes + kv_bytes(cfg, max_ctx) + state_bytes;
    crate::host_log::host_line(format!(
        "xwen: weights {:.1}GB + KV {:.1}GB + state {:.1}GB = {:.1}GB resident \
         (KV grows to {:.1}GB at max_ctx {max_ctx})",
        gb(weight_bytes),
        gb(kv_bytes(cfg, kv_slots)),
        gb(state_bytes),
        gb(total),
        gb(kv_bytes(cfg, max_ctx)),
    ));
    if ple_table_bytes > 0 {
        crate::host_log::host_line(format!(
            "xwen: PLE n-gram table {:.1}GB mapped, read from the host per token — not \
             uploaded, and not counted above",
            gb(ple_table_bytes)
        ));
    }
    if ceiling > MEMORY_WARN_BYTES {
        crate::host_log::host_line(format!(
            "xwen: WARNING footprint can reach {:.1}GB at full context, over the {:.0}GB budget",
            gb(ceiling),
            gb(MEMORY_WARN_BYTES)
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RopeKind;

    /// A qwen4exp file with no `qwen4exp.*` metadata is refused up front:
    /// `load` runs `check_arch` on the parsed config before the rope table and
    /// the token-embedding dequant, so the refusal materializes no tensors. A
    /// file that carries the section passes, as do both qwen35 arms.
    #[test]
    fn qwen4exp_without_its_config_section_is_refused_before_any_tensor_work() {
        let cfg = |arch: Arch| XwenConfig {
            arch,
            general_name: None,
            n_layer: 48,
            hidden: 2560,
            vocab: 248320,
            n_head: vec![24; 48],
            n_kv_head: 2,
            head_dim: 256,
            layer_kind: vec![LayerKind::Linear; 48],
            linear_k_heads: 16,
            linear_v_heads: 48,
            linear_head_dim: 128,
            conv_kernel: 4,
            dense_ff: 0,
            n_expert: 512,
            n_expert_used: 10,
            expert_ff: 640,
            shared_expert_ff: 640,
            rms_eps: 1e-6,
            n_ctx_train: 262144,
            rope: RopeKind::Plain {
                freq_base: 1e7,
                n_rot: 64,
            },
            eog_tokens: vec![248046, 248044],
            qwen4exp: None,
        };
        let err = XwenModel::check_arch(&cfg(Arch::Qwen4Exp))
            .unwrap_err()
            .to_string();
        // The message names what the file is missing, so the operator can tell
        // a broken conversion from an unsupported checkpoint.
        assert!(err.contains("qwen4exp.hyper_connection"), "{err}");
        assert!(XwenModel::check_arch(&cfg(Arch::Dense)).is_ok());
        assert!(XwenModel::check_arch(&cfg(Arch::Moe)).is_ok());

        // With the section present the arch is built, not refused.
        let mut ok = cfg(Arch::Qwen4Exp);
        ok.qwen4exp = Some(crate::config::Qwen4ExpConfig {
            hc_count: 4,
            hc_low_rank: 320,
            indexer_heads: 4,
            indexer_head_dim: 128,
            indexer_top_k: 2048,
            indexer_compress_ratio: 4,
            ple: None,
        });
        assert!(XwenModel::check_arch(&ok).is_ok());
    }

    fn probe(id: usize) -> Tensor {
        // Distinct 1x3 f32 tensor per layer id, so ordering is observable.
        Tensor::from_vec(
            vec![id as f32, id as f32 + 0.5, id as f32 + 0.25],
            (1, 3),
            &Device::Cpu,
        )
        .unwrap()
    }

    fn first(t: &Tensor) -> f32 {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]
    }

    /// The captured `(layer, tensor)` pairs arrive in ascending layer order (loop
    /// order); `order_spec_taps` re-emits them in the caller's configured order,
    /// including a non-ascending config and a subset of the run layers.
    #[test]
    fn spec_taps_follow_configured_order() {
        // Forward captured layers 1, 10, 19, 38, 47 (ascending, as the loop runs).
        let captured: Vec<(usize, Tensor)> = [1, 10, 19, 38, 47]
            .iter()
            .map(|&il| (il, probe(il)))
            .collect();

        // Configured out of order and as a subset.
        let cfg = vec![47usize, 1, 38];
        let ordered = order_spec_taps(&cfg, captured);
        let got: Vec<usize> = ordered.iter().map(|t| first(t) as usize).collect();
        assert_eq!(got, vec![47, 1, 38]);
    }

    /// A configured layer with no matching capture is skipped (defensive — every
    /// configured layer runs in practice), and a repeated layer repeats its tap.
    #[test]
    fn spec_taps_skip_missing_and_repeat() {
        let captured: Vec<(usize, Tensor)> =
            [1usize, 2].iter().map(|&il| (il, probe(il))).collect();
        let cfg = vec![2usize, 99, 2];
        let ordered = order_spec_taps(&cfg, captured);
        let got: Vec<usize> = ordered.iter().map(|t| first(t) as usize).collect();
        assert_eq!(got, vec![2, 2]);
    }
}
