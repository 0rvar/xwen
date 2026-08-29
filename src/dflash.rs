//! DFlash drafter model — the speculative-decoding drafter for the Qwen 3.6
//! targets. This is a self-contained module: the drafter is a pure function of
//! its OWN weights and its OWN KV cache and never references `XwenModel`. Token
//! embeddings and the lm_head live in the target and are applied by the driver,
//! not here (the drafter returns pre-lm_head hidden states).
//!
//! Ground truth: `reference/llama.cpp/src/models/dflash.cpp` — the encoder graph
//! at :109-123, the KV-injection graph at :238-300, the noise-block decoder at
//! :325-384. The architecture is "dflash": 5 dense layers at hidden 5120 (the
//! 27B sidecar) or 6 at hidden 2048 (the 35B-A3B sidecar), 32 Q / 8 KV heads,
//! head_dim 128, QK-norm before rope, SwiGLU FFN, plain NEOX rope over all 128
//! dims (theta from the GGUF, 1e7 on both sidecars, no YaRN), and NON-CAUSAL
//! attention windowed to `dflash.attention.sliding_window` on every layer the
//! `sliding_window_pattern` marks — the last layer is full on both sidecars.
//! There is no attention output gate and no per-tap encoder norm.
//!
//! The drafter is a block-diffusion model: it denoises `[id_last, MASK × k]` in
//! one forward, so its block attends to itself in both directions
//! (`llama_set_causal_attn(ctx_dft, false)`, common/speculative.cpp:1004) and
//! only the sliding window ever removes a key.
//!
//! Matmul weights: by DEFAULT the GGUF's BF16 planes are mmap-aliased no-copy
//! (the target's `MmapSource` machinery — the `GgufFile`'s mapping is shared,
//! the views join the residency set, and the drafter keeps the `Arc` alive)
//! and multiplied against f32 activations by the vendored bf16 mixed-dtype
//! kernels (`crate::ops::matmul_bf16`: f32 products/accumulation, f32 output —
//! the same convention as the target's attention projections; bf16 -> f32
//! widening is exact, so the stored weights remain the only sub-f32 rounding).
//! Each aliased plane is range-scanned at load and any value outside f16's
//! range is a hard error (`ensure_bf16_fits_f16` — the tensor-path gemm stages
//! bf16 -> half, and the old materialize-to-f16 load enforced the same bound
//! via its finiteness check). Without a mapping (CPU device,
//! `XWEN_LOAD_CLASSIC`) weights are materialized DENSE f16
//! (BF16 -> f32 -> f16, `crate::ops::matmul_f16`) — the pre-alias behavior.
//! Norm weights (attn/ffn/output norms, q/k-norm) stay f32 on every
//! path. `XWEN_DRAFT_F32` restores full-f32 weights with plain candle matmul
//! (the path the scalar-reference tests pin). The drafter is tiny (1.7B / 386M,
//! <= block_size query rows per forward).

use std::borrow::Cow;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result, bail, ensure};
use candle_core::quantized::GgmlDType;
use candle_core::quantized::gguf_file::{Content, Value};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::RmsNorm;

use crate::gguf::{GgufFile, MmapSource, dense_alias_tensor};
use crate::kv_cache::{
    MAX_STORED_HEAD_DIM, MAX_STORED_KV_HEADS, MAX_STORED_LAYERS, MAX_STORED_POS, RecordReader,
    plane_bytes, plausible, write_count, write_plane,
};

/// Default positions the drafter's KV cache is sized for, shared by `generate`,
/// `chat` and serve. This doubles as the DRAFTING DEPTH LIMIT: past it decode
/// continues plain (the round loops go plain once `n_past` runs off the
/// drafter's cache). The cap keeps the drafter's f32 cache planes and their
/// exported images small: 40 KiB/token on the 27B sidecar (5 layers x K/V x
/// 8 heads x 128 dims x f32) and 48 KiB/token on the 35B-A3B's 6 layers.
///
/// The value is inherited from laguna and has not been re-derived against the
/// Qwen sidecars (TODO ledger). Laguna's argument was that the drafter forward
/// is O(depth) and its proposal quality collapses with depth; the O(depth) half
/// no longer holds here, because every layer but the last is windowed to
/// 2048/4096 positions and `attention` narrows the cache to that window, so only
/// one layer of five or six grows with the context.
pub const DEFAULT_DRAFT_CTX: usize = 8192;

/// Static architecture facts parsed from the `dflash.*` GGUF metadata. Errors
/// unless the file is a DFlash drafter (`general.architecture == "dflash"`).
/// The shipped sidecars carry no key naming their target architecture, so the
/// pairing of a drafter to a target is the caller's business — `hub.rs` resolves
/// both from one `Model`, and `spec_tap_layers` is range-checked against the
/// target at attach time.
#[derive(Debug, Clone)]
pub struct DflashConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub rms_eps: f64,
    pub rope_theta: f32,
    /// Sliding-window width in positions (`dflash.attention.sliding_window`), or
    /// `None` when the file declares no window. A windowed layer's query at
    /// position `p` may attend to keys at `[p - window + 1, p]` on the past side
    /// (llama-hparams.h `is_masked_swa`: masked iff `p1 - p0 >= n_swa`); the
    /// future side is unbounded because the drafter is non-causal.
    pub sliding_window: Option<usize>,
    /// Per-layer window flags (`dflash.attention.sliding_window_pattern`), one
    /// per layer, in layer order. All-`false` when the file declares no window.
    /// Both sidecars window every layer but the last.
    pub swa_layers: Vec<bool>,
    /// Noise-block width the drafter drafts per step (`dflash.block_size`).
    pub block_size: usize,
    /// Target-model layer indices whose residual taps feed the encoder, in the
    /// order the encoder concatenates them (`dflash.target_layers`).
    pub target_layers: Vec<usize>,
    /// Token id used to fill the masked positions of a noise block
    /// (`tokenizer.ggml.mask_token_id`).
    pub mask_token_id: u32,
    pub context_length: usize,
}

impl DflashConfig {
    pub fn from_gguf(content: &Content) -> Result<Self> {
        let get = |k: &str| {
            content
                .metadata
                .get(k)
                .with_context(|| format!("missing GGUF key {k}"))
        };

        let arch = get("general.architecture")?.to_string()?.as_str();
        ensure!(
            arch == "dflash",
            "expected a dflash GGUF, got architecture {arch:?}"
        );

        let target_layers = get("dflash.target_layers")?
            .to_vec()
            .context("dflash.target_layers is not an array")?
            .iter()
            .map(value_usize)
            .collect::<Result<Vec<_>>>()
            .context("dflash.target_layers has non-integer entries")?;
        ensure!(!target_layers.is_empty(), "dflash.target_layers is empty");

        let n_layer = value_usize(get("dflash.block_count")?)?;

        // The window key is optional; once present it makes the pattern key
        // mandatory and the pattern must cover every layer, exactly as
        // dflash.cpp:23-30 reads them. A declared width of 0 is a file that
        // means "no window" the long way round, and is normalized to None so
        // nothing downstream has to special-case it.
        let sliding_window = match content.metadata.get("dflash.attention.sliding_window") {
            Some(v) => {
                let w =
                    value_usize(v).context("dflash.attention.sliding_window is not an integer")?;
                (w > 0).then_some(w)
            }
            None => None,
        };
        let swa_layers = match sliding_window {
            None => vec![false; n_layer],
            Some(_) => {
                // llama.cpp reads this through `get_key_or_arr`, which takes
                // EITHER an array of exactly n_layer entries OR a single value
                // repeated across every layer (llama-model-loader.cpp:444-482).
                // Both shipped sidecars write the array form; the scalar form is
                // what a converter emitting a uniformly-windowed drafter would
                // produce, and rejecting it would refuse a file llama.cpp loads.
                let key = get("dflash.attention.sliding_window_pattern")?;
                match key.to_vec() {
                    Ok(entries) => {
                        let pattern = entries
                            .iter()
                            .map(value_bool)
                            .collect::<Result<Vec<_>>>()
                            .context(
                                "dflash.attention.sliding_window_pattern has non-boolean entries",
                            )?;
                        ensure!(
                            pattern.len() == n_layer,
                            "dflash.attention.sliding_window_pattern covers {} layers, the model has {n_layer}",
                            pattern.len()
                        );
                        pattern
                    }
                    Err(_) => {
                        let all = value_bool(key).context(
                            "dflash.attention.sliding_window_pattern is neither an array nor a \
                             single boolean",
                        )?;
                        vec![all; n_layer]
                    }
                }
            }
        };

        Ok(Self {
            n_layer,
            n_embd: value_usize(get("dflash.embedding_length")?)?,
            n_head: value_usize(get("dflash.attention.head_count")?)?,
            n_head_kv: value_usize(get("dflash.attention.head_count_kv")?)?,
            head_dim: value_usize(get("dflash.attention.key_length")?)?,
            n_ff: value_usize(get("dflash.feed_forward_length")?)?,
            rms_eps: get("dflash.attention.layer_norm_rms_epsilon")?.to_f32()? as f64,
            rope_theta: get("dflash.rope.freq_base")?.to_f32()?,
            sliding_window,
            swa_layers,
            block_size: value_usize(get("dflash.block_size")?)?,
            target_layers,
            mask_token_id: get("tokenizer.ggml.mask_token_id")?.to_u32()?,
            context_length: value_usize(get("dflash.context_length")?)?,
        })
    }

    /// The window layer `il` attends under, or `None` for full attention.
    fn layer_window(&self, il: usize) -> Option<usize> {
        match self.swa_layers.get(il) {
            Some(true) => self.sliding_window,
            _ => None,
        }
    }

    /// The encoder's concatenated feature width: `target_layers.len() * n_embd`.
    fn n_embd_inp(&self) -> usize {
        self.target_layers.len() * self.n_embd
    }

    /// Everything about this drafter that can be judged before a weight is read:
    /// its own geometry has to be arithmetically buildable, and the parts of it
    /// that reach across to the target have to match the target.
    ///
    /// This is the ONE place these checks live, and it has to stay that way.
    /// Both entry points that attach a drafter — `Generator::attach_drafter` on
    /// the CLI path and serve's startup preflight — call it, so a check added
    /// for one is a check the other gets. Split across two call sites they drift,
    /// and a drafter built for the other checkpoint reaches a running server.
    ///
    /// The cross-target checks are the whole of what a sidecar's metadata can be
    /// held to: nothing in a DFlash file names the checkpoint it belongs to, so
    /// pairing one with a target is only ever the caller's assertion. The
    /// drafter's own depth, FFN width and head counts describe the drafter and
    /// cannot be compared against anything — only their internal consistency is
    /// checkable, which is what the first group does.
    pub fn check_against_target(
        &self,
        target_hidden: usize,
        target_n_layer: usize,
        target_vocab: usize,
    ) -> Result<()> {
        // Internal consistency. `n_head_kv > 0` comes first and is the one that
        // matters most: `DflashDrafter::build` guards the head ratio with an
        // `ensure!`, but the `%` inside it is evaluated first and a zero divisor
        // panics there before anything can report why.
        ensure!(
            self.n_head_kv > 0,
            "the drafter declares 0 KV heads (dflash.attention.head_count_kv), which no \
             attention geometry can be built from"
        );
        ensure!(
            self.n_head > 0,
            "the drafter declares 0 attention heads (dflash.attention.head_count)"
        );
        ensure!(
            self.n_head.is_multiple_of(self.n_head_kv),
            "the drafter declares {} attention heads over {} KV heads, which do not divide: \
             grouped-query attention needs a whole number of query heads per KV head",
            self.n_head,
            self.n_head_kv
        );
        ensure!(
            self.head_dim > 0 && self.head_dim.is_multiple_of(2),
            "the drafter declares a head dim of {}, which rope cannot split into halves",
            self.head_dim
        );
        ensure!(self.n_layer > 0, "the drafter declares 0 layers");
        ensure!(
            self.n_ff > 0,
            "the drafter declares a feed-forward width of 0"
        );
        // A block of 1 is the anchor token alone: `block_size - 1` draft tokens
        // is zero, so such a drafter could never propose anything.
        ensure!(
            self.block_size >= 2,
            "the drafter declares a noise block of {} tokens, which leaves no room for a \
             draft (the first block position is the anchor)",
            self.block_size
        );

        // Cross-checks against the model being served.
        ensure!(
            self.n_embd == target_hidden,
            "the drafter has a hidden size of {} but the target has {target_hidden}: \
             this drafter was built for a different checkpoint",
            self.n_embd
        );
        for il in self.spec_tap_layers()? {
            ensure!(
                il < target_n_layer,
                "the drafter reads a residual tap from layer {il} of the target, which has \
                 {target_n_layer} layers: this drafter was built for a deeper model"
            );
        }
        // The mask token is fed straight into the target's embedding table every
        // speculative round, so an out-of-range id is an indexing failure at the
        // first round and an in-range wrong one degrades acceptance silently.
        // Only the range is checkable here — the two shipped sidecars disagree
        // on the id (248070 vs 248077), so there is no single right value.
        ensure!(
            (self.mask_token_id as usize) < target_vocab,
            "the drafter's mask token id {} is outside the target's {target_vocab}-token \
             vocabulary",
            self.mask_token_id
        );
        Ok(())
    }

    /// The target-model `l_out` indices whose residuals feed the encoder, in
    /// tap order — i.e. what to pass to `XwenModel::set_spec_taps`.
    ///
    /// `dflash.target_layers` entries name the residual ENTERING target layer
    /// `t` (the fork's `t_layer_inp[t]`), which is the residual LEAVING layer
    /// `t - 1`; the sentinel `t == n_layer_tgt` names the pre-final-norm
    /// state, which is likewise `l_out` of the last layer. So the uniform
    /// translation to our post-FFN `l_out` capture points is `t - 1`.
    pub fn spec_tap_layers(&self) -> Result<Vec<usize>> {
        self.target_layers
            .iter()
            .map(|&t| {
                t.checked_sub(1).with_context(|| {
                    format!("target_layers entry {t} taps the raw embedding, which has no l_out capture point")
                })
            })
            .collect()
    }
}

/// How the drafter's matmul weights are stored/multiplied when they have to be
/// MATERIALIZED from f32 (`matmul_weight`): `F16` narrows to the vendored
/// mixed-dtype kernels' dense storage (`crate::ops::matmul_f16`, Metal only);
/// `F32` keeps them dense f32 and uses plain candle matmul (portable, the path
/// the scalar-reference tests pin); `Bf16` narrows to BF16 for the vendored
/// bf16 kernels — in production BF16 storage arrives pre-built as an mmap
/// alias (which `matmul_weight` passes through under ANY requested dtype), so
/// this variant exists for the synthetic tests. Threaded explicitly through
/// `build` so tests can construct each path without racing a process-global
/// env switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WeightDtype {
    F32,
    F16,
    #[cfg_attr(not(test), allow(dead_code))]
    Bf16,
}

/// The drafter matmul-weight dtype selected by the environment, read once.
/// `XWEN_DRAFT_F32` (presence-based, like the other `XWEN_*` switches — any
/// value selects it) reverts to full f32; otherwise f16 is the default.
fn draft_weight_dtype() -> WeightDtype {
    static V: OnceLock<WeightDtype> = OnceLock::new();
    *V.get_or_init(|| {
        if std::env::var_os("XWEN_DRAFT_F32").is_some() {
            WeightDtype::F32
        } else {
            WeightDtype::F16
        }
    })
}

/// Enforces the vendored f16/bf16 kernel constraints on a matmul weight:
/// rank-2 `[n_out, k]`, `k % 32 == 0`, `n_out % 4 == 0`.
fn ensure_kernel_dims(t: &Tensor, name: &str) -> Result<()> {
    let (n_out, k) = t
        .dims2()
        .with_context(|| format!("matmul weight {name} is not rank-2 [n_out, k]"))?;
    ensure!(
        k % 32 == 0,
        "matmul weight {name} k={k} must be a multiple of 32 for the f16/bf16 kernels"
    );
    ensure!(
        n_out % 4 == 0,
        "matmul weight {name} n_out={n_out} must be a multiple of 4 for the f16/bf16 kernels"
    );
    Ok(())
}

/// Cast a dense matmul weight to the requested storage dtype. An already-BF16
/// tensor is an mmap-aliased plane whose storage is final (range-scanned at
/// alias time — see `bf16_alias_tensor`): it is validated and passed through
/// under ANY requested dtype. Otherwise the provider yielded dense f32: `F32`
/// passes it through unchanged; `F16` narrows it (f32 -> f16) and enforces
/// finiteness — BF16's exponent range exceeds f16's, so a weight that
/// overflows f16 must be a hard load error rather than a silent inf; `Bf16`
/// narrows to BF16 (the synthetic-test constructor for the bf16 kernel path —
/// production BF16 always arrives pre-aliased).
fn matmul_weight(t: Tensor, dtype: WeightDtype, name: &str) -> Result<Tensor> {
    if t.dtype() == DType::BF16 {
        ensure_kernel_dims(&t, name)?;
        return Ok(t);
    }
    match dtype {
        WeightDtype::F32 => Ok(t),
        WeightDtype::F16 => {
            ensure_kernel_dims(&t, name)?;
            let f16 = t.to_dtype(DType::F16)?;
            // Widen back and reduce: any inf/nan produced by the narrowing makes
            // the sum non-finite (finite f16 magnitudes stay well within f32).
            let mag = f16
                .to_dtype(DType::F32)?
                .sqr()?
                .sum_all()?
                .to_scalar::<f32>()?;
            ensure!(
                mag.is_finite(),
                "matmul weight {name} has non-finite values after f16 narrowing (weight overflows f16 range)"
            );
            Ok(f16)
        }
        WeightDtype::Bf16 => {
            ensure_kernel_dims(&t, name)?;
            Ok(t.to_dtype(DType::BF16)?)
        }
    }
}

/// Accept a GGUF bool or any integer width as a flag. The shipped sidecars
/// store `sliding_window_pattern` as bool[], but llama.cpp's array reader
/// accepts integer arrays for the same key (llama-model-loader.cpp:383-387), so
/// a converter that writes 0/1 stays readable.
fn value_bool(v: &Value) -> Result<bool> {
    if let Ok(b) = v.to_bool() {
        return Ok(b);
    }
    Ok(value_usize(v)? != 0)
}

/// Accept any GGUF integer width/signedness (target_layers is i32[]).
fn value_usize(v: &Value) -> Result<usize> {
    if let Ok(u) = v.to_u64() {
        return Ok(u as usize);
    }
    if let Ok(u) = v.to_u32() {
        return Ok(u as usize);
    }
    let i = v.to_i64().or_else(|_| v.to_i32().map(i64::from))?;
    ensure!(i >= 0, "negative integer {i}");
    Ok(i as usize)
}

/// One decoder layer's weights. Matmul weights (`wq/wk/wv/wo` and the FFN
/// projections) are `[out, in]` (GGUF layout, `y = x @ wᵀ`) and dense f16 by
/// default / f32 under `XWEN_DRAFT_F32`; norm weights are always f32. `linear`
/// dispatches on each weight's dtype.
struct LayerWeights {
    attn_norm: RmsNorm,
    wq: Tensor,      // [n_head*head_dim, n_embd]
    wk: Tensor,      // [n_head_kv*head_dim, n_embd]
    wv: Tensor,      // [n_head_kv*head_dim, n_embd]
    wo: Tensor,      // [n_embd, n_head*head_dim]
    q_norm: RmsNorm, // [head_dim]
    k_norm: RmsNorm, // [head_dim]
    ffn_norm: RmsNorm,
    ffn_gate: Tensor, // [n_ff, n_embd]
    ffn_up: Tensor,   // [n_ff, n_embd]
    ffn_down: Tensor, // [n_embd, n_ff]
}

/// One layer's full-attention KV cache: `[n_head_kv, max_ctx, head_dim]` f32.
/// Chosen f32 (not the target's f16) for exactness — the drafter is tiny, so the
/// extra bytes are negligible and it keeps the naive-reference tests bit-tight.
/// Rows `0..committed` are the injected context; rows `committed..max_ctx` are
/// scratch — `draft_forward` writes each round's noise K/V there in place (no
/// growing per-round concat) and `inject` overwrites them when it commits.
struct LayerCache {
    k: Tensor,
    v: Tensor,
}

/// The DFlash drafter: encoder (feature fusion of target taps), per-layer KV
/// injection of the fused context, and noise-block draft forwards. Batch=1.
pub struct DflashDrafter {
    cfg: DflashConfig,
    device: Device,

    // Encoder.
    enc_output_norm: RmsNorm,
    fc: Tensor, // [n_embd, n_aux*n_embd]

    // Decoder.
    output_norm: RmsNorm, // final norm (decoder), NOT the encoder's
    layers: Vec<LayerWeights>,

    caches: Vec<LayerCache>,
    /// Committed (injected) context length, shared across layers — every layer
    /// advances together on `inject`.
    committed: usize,
    max_ctx: usize,

    /// Keeps the GGUF mapping alive while the matmul planes are mmap aliases
    /// (their `Tensor`s cannot carry the Arc — `MmapSource`'s lifetime
    /// invariant, same as `XwenModel._weights_mmap`). `None` on the
    /// materialized paths. Dropping the drafter drops the mapping, which
    /// quiesces the GPU and unregisters the views (serve's idle-unload path).
    _weights_mmap: Option<Arc<MmapSource>>,

    // Plain NEOX rope tables (theta from cfg, all head_dim dims rotary, no YaRN
    // scaling): cos/sin `[max_ctx, head_dim/2]` f32.
    rope_cos: Tensor,
    rope_sin: Tensor,
}

impl DflashDrafter {
    /// Load the drafter from an opened GGUF. Default (Metal open, no
    /// `XWEN_DRAFT_F32`): every rank-2 BF16 tensor — the matmul planes — is
    /// mmap-aliased no-copy off the `GgufFile`'s mapping after a range scan
    /// (`bf16_alias_tensor`), the views join the residency set in one batch,
    /// and the drafter keeps the mapping's Arc alive; only the small F32 norm
    /// tensors are read + uploaded. Without a mapping (CPU device,
    /// `XWEN_LOAD_CLASSIC`) every tensor takes the copying reader and
    /// `matmul_weight` materializes f16 — the pre-alias behavior. `max_ctx`
    /// sizes the per-layer KV caches and the rope tables.
    pub fn load(gguf: &GgufFile, device: &Device, max_ctx: usize) -> Result<Self> {
        let cfg = DflashConfig::from_gguf(&gguf.content)?;
        let mut file = File::open(&gguf.path)
            .with_context(|| format!("opening {} for drafter load", gguf.path.display()))?;
        let content = &gguf.content;
        let dtype = draft_weight_dtype();
        // The alias path serves the kernel-storage default; the F32 debug path
        // keeps its full-copy semantics (the scalar-reference contract).
        let mmap = if dtype == WeightDtype::F32 {
            None
        } else {
            gguf.mmap_source().cloned()
        };
        let mut drafter = Self::build(cfg, device, max_ctx, dtype, |name| {
            if let Some(src) = &mmap {
                if let Some(t) = bf16_alias_tensor(src, content, name, device)? {
                    return Ok(t);
                }
            }
            read_tensor_f32(&mut file, content, name, device)
        })?;
        if let Some(src) = mmap {
            // One batch residency registration for every aliased plane (the
            // same single-commit pattern as XwenModel::load; per-view
            // registration costs a synchronous commit each).
            src.register_views();
            drafter._weights_mmap = Some(src);
        }
        Ok(drafter)
    }

    /// Assemble the drafter from a tensor provider yielding dense f32 tensors by
    /// GGUF name (with the `.weight` suffix). Shared by `load` (reads the GGUF)
    /// and the tests (an in-memory synthetic map), so both exercise identical
    /// assembly. `dtype` selects the matmul-weight storage (f16 default / f32);
    /// norm weights are always kept f32.
    fn build(
        cfg: DflashConfig,
        device: &Device,
        max_ctx: usize,
        dtype: WeightDtype,
        mut get: impl FnMut(&str) -> Result<Tensor>,
    ) -> Result<Self> {
        ensure!(max_ctx >= 1, "max_ctx must be >= 1");
        ensure!(
            cfg.head_dim % 2 == 0,
            "head_dim {} must be even for rope",
            cfg.head_dim
        );
        // Ordered before the ratio: `%` is evaluated before `ensure!` tests its
        // condition, so a zero divisor panics with a remainder-by-zero instead
        // of reporting the malformed metadata. `check_against_target` catches
        // this earlier on both the CLI and serve paths, but `build` is `pub`
        // enough to reach directly and must not panic on a bad file.
        ensure!(
            cfg.n_head_kv > 0,
            "n_head_kv is 0, which no attention geometry can be built from"
        );
        ensure!(
            cfg.n_head % cfg.n_head_kv == 0,
            "n_head {} must be a multiple of n_head_kv {}",
            cfg.n_head,
            cfg.n_head_kv
        );

        let eps = cfg.rms_eps;
        let norm = |t: Tensor| RmsNorm::new(t, eps);

        let enc_output_norm = norm(get("enc.output_norm")?);
        let fc = get("fc")?;
        ensure!(
            fc.dims() == [cfg.n_embd, cfg.n_embd_inp()],
            "fc shape {:?} != [{}, {}]",
            fc.dims(),
            cfg.n_embd,
            cfg.n_embd_inp()
        );
        let fc = matmul_weight(fc, dtype, "fc")?;
        let output_norm = norm(get("output_norm")?);

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let p = |n: &str| format!("blk.{il}.{n}");
            layers.push(LayerWeights {
                attn_norm: norm(get(&p("attn_norm"))?),
                wq: matmul_weight(get(&p("attn_q"))?, dtype, "attn_q")?,
                wk: matmul_weight(get(&p("attn_k"))?, dtype, "attn_k")?,
                wv: matmul_weight(get(&p("attn_v"))?, dtype, "attn_v")?,
                wo: matmul_weight(get(&p("attn_output"))?, dtype, "attn_output")?,
                q_norm: norm(get(&p("attn_q_norm"))?),
                k_norm: norm(get(&p("attn_k_norm"))?),
                ffn_norm: norm(get(&p("ffn_norm"))?),
                ffn_gate: matmul_weight(get(&p("ffn_gate"))?, dtype, "ffn_gate")?,
                ffn_up: matmul_weight(get(&p("ffn_up"))?, dtype, "ffn_up")?,
                ffn_down: matmul_weight(get(&p("ffn_down"))?, dtype, "ffn_down")?,
            });
        }

        let (n_kv, hd) = (cfg.n_head_kv, cfg.head_dim);
        let caches = (0..cfg.n_layer)
            .map(|_| {
                Ok(LayerCache {
                    k: Tensor::zeros((n_kv, max_ctx, hd), DType::F32, device)?,
                    v: Tensor::zeros((n_kv, max_ctx, hd), DType::F32, device)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let (rope_cos, rope_sin) = rope_tables(cfg.rope_theta, hd, max_ctx, device)?;

        Ok(Self {
            cfg,
            device: device.clone(),
            enc_output_norm,
            fc,
            output_norm,
            layers,
            caches,
            committed: 0,
            max_ctx,
            _weights_mmap: None,
            rope_cos,
            rope_sin,
        })
    }

    pub fn config(&self) -> &DflashConfig {
        &self.cfg
    }

    pub fn committed_len(&self) -> usize {
        self.committed
    }

    /// Roll the committed cache back to `len` (<= current committed length). The
    /// bytes beyond `len` stay in the cache tensors but are ignored; the next
    /// `inject` overwrites them.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        ensure!(
            len <= self.committed,
            "truncate({len}) beyond committed length {}",
            self.committed
        );
        self.committed = len;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.committed = 0;
    }

    /// Positions the drafter's own KV cache can hold. Sized independently of the
    /// target's context (the drafter's f32 planes cost 40-48 KiB/token), so a driver
    /// has to stop injecting — and stop speculating — once the conversation runs
    /// past this.
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// Copy the committed rows `[0, committed)` of every layer out to host RAM,
    /// so a conversation can be parked and the drafter's cache handed to another
    /// one. The image owns its bytes: the drafter goes on writing the scratch rows
    /// behind it (see `f32_rows` for why a short prefix is compacted on the device
    /// before the readback rather than after).
    pub fn export_cache(&self) -> Result<DrafterImage> {
        let (n_kv, hd) = (self.cfg.n_head_kv, self.cfg.head_dim);
        let pos = self.committed;
        let planes = self
            .caches
            .iter()
            .map(|c| {
                if pos == 0 {
                    return Ok((Vec::new(), Vec::new()));
                }
                Ok((f32_rows(&c.k, pos)?, f32_rows(&c.v, pos)?))
            })
            .collect::<Result<Vec<_>>>()?;
        DrafterImage::new_dflash(pos, n_kv, hd, planes)
    }

    /// Upload `image`'s rows `[0, pos)` back into every layer and set the
    /// committed length to `pos`, so the next `inject`/`draft_forward` continues
    /// from there. Rows past `pos` keep whatever they held — they are scratch,
    /// and every forward overwrites the scratch rows it reads.
    pub fn import_cache(&mut self, image: &DrafterImage, pos: usize) -> Result<()> {
        ensure!(
            image.kind() == DrafterImageKind::Dflash,
            "this image was written by a {} drafter; the DFlash drafter cannot read it",
            image.kind().name()
        );
        ensure!(
            pos <= image.pos,
            "drafter_image: requested pos {pos} exceeds the image's {} positions",
            image.pos
        );
        ensure!(
            image.layers() == self.caches.len(),
            "drafter_image: image covers {} layers, the drafter has {}",
            image.layers(),
            self.caches.len()
        );
        let (n_kv, hd) = (self.cfg.n_head_kv, self.cfg.head_dim);
        ensure!(
            image.n_kv_head == n_kv && image.head_dim == hd,
            "drafter_image: image is ({}, _, {}) but the drafter's cache is ({n_kv}, _, {hd})",
            image.n_kv_head,
            image.head_dim
        );
        ensure!(
            pos <= self.max_ctx,
            "drafter_image: pos {pos} exceeds the drafter's {} cache slots",
            self.max_ctx
        );
        for (idx, cache) in self.caches.iter_mut().enumerate() {
            let (k, v) = image.prefix(idx, pos)?;
            if pos > 0 {
                let shape = [n_kv, pos, hd];
                cache
                    .k
                    .slice_set(&f32_tensor(&k, &shape, &self.device)?, 1, 0)?;
                cache
                    .v
                    .slice_set(&f32_tensor(&v, &shape, &self.device)?, 1, 0)?;
            }
        }
        self.committed = pos;
        Ok(())
    }

    /// Encoder: fuse the target-model residual taps into the drafter's hidden
    /// space. `taps` are `n_aux` (= `target_layers.len()`) f32 tensors, each
    /// `[seq, n_embd]`, ordered as `target_layers`.
    ///
    /// Three ops, per dflash.cpp:109-123: concatenate the taps tap-major within
    /// each row to `[seq, n_aux*n_embd]`, apply `fc`, then RMS-norm with
    /// `enc.output_norm`. The taps enter raw — the sidecars ship no per-tap norm
    /// or scale, and the taps are the target's residual stream as it enters the
    /// tapped block, unnormalized. Returns fused `[seq, n_embd]` f32.
    pub fn encode(&self, taps: &[Tensor]) -> Result<Tensor> {
        let n_aux = self.cfg.target_layers.len();
        ensure!(
            taps.len() == n_aux,
            "encode expects {n_aux} taps (one per target layer), got {}",
            taps.len()
        );
        let (seq, n_embd) = taps[0]
            .dims2()
            .context("tap 0 is not rank-2 [seq, n_embd]")?;
        ensure!(
            n_embd == self.cfg.n_embd,
            "tap 0 hidden {n_embd} != n_embd {}",
            self.cfg.n_embd
        );
        for (i, t) in taps.iter().enumerate() {
            let (s, e) = t
                .dims2()
                .with_context(|| format!("tap {i} is not rank-2"))?;
            ensure!(
                s == seq && e == n_embd,
                "tap {i} shape [{s}, {e}] != [{seq}, {n_embd}]"
            );
        }

        // [seq, n_aux * n_embd], tap-major within each row.
        let refs: Vec<&Tensor> = taps.iter().collect();
        let flat = Tensor::cat(&refs, 1)?.to_dtype(DType::F32)?.contiguous()?;
        debug_assert_eq!(flat.dims(), [seq, n_aux * n_embd]);

        let fused = linear(&flat, &self.fc)?; // [seq, n_embd]
        Ok(self.enc_output_norm.forward(&fused)?)
    }

    /// KV-injection: project the fused context into every layer's K/V and append
    /// to the committed cache. `pos0` must equal the current committed length
    /// (positions are sequential). K is QK-normed then roped; V is neither.
    ///
    /// The fused input goes into `wk`/`wv` RAW, with no `attn_norm` — the
    /// encoder's `enc.output_norm` is the only norm on this path
    /// (dflash.cpp:252-253, where the injection graph projects `inp_g`
    /// directly). The query path in `draft_forward` does apply `attn_norm`,
    /// so the two paths deliberately disagree.
    pub fn inject(&mut self, fused: &Tensor, pos0: usize) -> Result<()> {
        ensure!(
            pos0 == self.committed,
            "inject pos0 {pos0} must equal committed length {}",
            self.committed
        );
        let (seq, n_embd) = fused.dims2().context("fused is not rank-2 [seq, n_embd]")?;
        ensure!(
            n_embd == self.cfg.n_embd,
            "fused hidden {n_embd} != n_embd {}",
            self.cfg.n_embd
        );
        ensure!(
            self.committed + seq <= self.max_ctx,
            "inject overflows cache: committed {} + {seq} > max_ctx {}",
            self.committed,
            self.max_ctx
        );
        let fused = fused.to_dtype(DType::F32)?;
        let (n_kv, hd) = (self.cfg.n_head_kv, self.cfg.head_dim);

        for il in 0..self.layers.len() {
            let layer = &self.layers[il];

            let k = linear(&fused, &layer.wk)?.reshape((seq, n_kv, hd))?;
            let v = linear(&fused, &layer.wv)?.reshape((seq, n_kv, hd))?;

            let k = layer.k_norm.forward(&k)?; // QK-norm over head_dim, pre-rope
            let k = k.transpose(0, 1)?.contiguous()?; // [n_kv, seq, hd]
            let k = self.rope(&k, pos0)?;
            let v = v.transpose(0, 1)?.contiguous()?; // [n_kv, seq, hd]

            self.caches[il].k.slice_set(&k, 1, self.committed)?;
            self.caches[il].v.slice_set(&v, 1, self.committed)?;
        }
        self.committed += seq;
        Ok(())
    }

    /// Noise-block draft forward. `noise_embd` is `[n_block, n_embd]` f32 —
    /// target-side token embeddings of `[id_last, MASK * (n_block-1)]` at
    /// positions `pos0..pos0+n_block` (pos0 must equal the committed length). Runs
    /// the decoder attending over `[committed cache | noise block]`, non-causally
    /// and windowed per layer (see `attention`).
    /// Each layer's noise K/V is written into the cache scratch region (rows
    /// `committed..committed+n_block`) and attention reads a narrowed cache view,
    /// so the committed prefix is never re-copied. `committed` is unchanged on
    /// return: those scratch rows stay uncommitted and a later `inject` overwrites
    /// them. The invariant is that `draft_forward` never dirties rows `< committed`
    /// — rows at and beyond `committed` are scratch. Returns final-normed hidden
    /// states `[n_block, n_embd]` f32 (pre-lm_head; the caller applies the target's
    /// shared lm_head).
    pub fn draft_forward(&mut self, noise_embd: &Tensor, pos0: usize) -> Result<Tensor> {
        ensure!(
            pos0 == self.committed,
            "draft_forward pos0 {pos0} must equal committed length {}",
            self.committed
        );
        let (n_block, n_embd) = noise_embd.dims2().context("noise_embd is not rank-2")?;
        ensure!(
            n_embd == self.cfg.n_embd,
            "noise hidden {n_embd} != n_embd {}",
            self.cfg.n_embd
        );
        ensure!(n_block >= 1, "noise block must have >= 1 row");
        ensure!(
            pos0 + n_block <= self.max_ctx,
            "noise block [{pos0}, {}) exceeds max_ctx {}",
            pos0 + n_block,
            self.max_ctx
        );

        let committed = self.committed;
        let (n_kv, hd, n_head) = (self.cfg.n_head_kv, self.cfg.head_dim, self.cfg.n_head);
        let scale = 1.0f32 / (hd as f32).sqrt();

        let mut inp = noise_embd.to_dtype(DType::F32)?;

        for il in 0..self.layers.len() {
            let layer = &self.layers[il];
            let noise_norm = layer.attn_norm.forward(&inp)?; // [nb, n_embd]

            let q = linear(&noise_norm, &layer.wq)?.reshape((n_block, n_head, hd))?;
            let k = linear(&noise_norm, &layer.wk)?.reshape((n_block, n_kv, hd))?;
            let v = linear(&noise_norm, &layer.wv)?.reshape((n_block, n_kv, hd))?;

            let q = layer.q_norm.forward(&q)?;
            let k = layer.k_norm.forward(&k)?;

            let q = self.rope(&q.transpose(0, 1)?.contiguous()?, pos0)?; // [n_head, nb, hd]
            let k = self.rope(&k.transpose(0, 1)?.contiguous()?, pos0)?; // [n_kv, nb, hd]
            let v = v.transpose(0, 1)?.contiguous()?; // [n_kv, nb, hd]

            // Effective K/V = committed context ++ this block's noise K/V. Write
            // the noise K/V into the cache scratch rows (committed..committed+nb)
            // and attend over a narrowed cache view, so the committed prefix is
            // read in place rather than re-copied into a fresh concat each round.
            // These rows are scratch (committed does not advance); a later inject
            // overwrites them, and this forward overwrites every scratch row it
            // reads before reading it, so the reuse never leaks stale K/V.
            self.caches[il].k.slice_set(&k, 1, committed)?;
            self.caches[il].v.slice_set(&v, 1, committed)?;

            let window = self.cfg.layer_window(il);
            let attn = self.attention(
                &self.caches[il].k,
                &self.caches[il].v,
                &q,
                committed,
                n_block,
                scale,
                window,
            )?; // [n_head, nb, hd]

            let attn = attn
                .transpose(0, 1)?
                .contiguous()?
                .reshape((n_block, n_head * hd))?;
            let cur = linear(&attn, &layer.wo)?; // [nb, n_embd]

            let ffn_inp = (&cur + &inp)?;
            let ffn_normed = layer.ffn_norm.forward(&ffn_inp)?;
            let ffn_out = self.swiglu(layer, &ffn_normed)?;
            inp = (&ffn_out + &ffn_inp)?;
        }

        Ok(self.output_norm.forward(&inp)?)
    }

    /// Non-causal GQA attention of the `n_block` noise queries over the cache
    /// rows `[0, committed + n_block)`, windowed to `window` positions on the
    /// past side.
    ///
    /// The drafter denoises its whole block in one forward, so every query sees
    /// every other block position in both directions — the only key a query
    /// loses is one that fell out of its window. Query `i` sits at absolute
    /// position `committed + i` and attends to absolute positions
    /// `> committed + i - window` (llama-hparams.h:393-398, `p1 - p0 >= n_swa`
    /// masks).
    ///
    /// Rather than mask a full-length score row, the cache is narrowed to the
    /// UNION of the block's windows, `[lo, committed + n_block)`, where `lo` is
    /// query 0's floor. A windowed layer therefore costs O(window) per round
    /// instead of O(context), which is the whole point of the window; only the
    /// at most `n_block - 1` columns between the queries' individual floors need
    /// an additive mask, and when the context still fits inside the window there
    /// is no mask at all.
    ///
    /// `q [n_head, nb, hd]` contiguous; `k_cache`/`v_cache` are the full
    /// `[n_kv, max_ctx, hd]` planes. The GQA head axis is added with `unsqueeze`
    /// (a metadata-only view) rather than `reshape` — the following
    /// `broadcast_matmul` materializes its broadcast operand anyway (candle
    /// contiguises the GQA expansion) and reads the strided view directly, so no
    /// separate prefix copy is introduced. All f32.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        k_cache: &Tensor,
        v_cache: &Tensor,
        q: &Tensor,
        committed: usize,
        n_block: usize,
        scale: f32,
        window: Option<usize>,
    ) -> Result<Tensor> {
        let (n_kv, hd, n_head) = (self.cfg.n_head_kv, self.cfg.head_dim, self.cfg.n_head);
        let g = n_head / n_kv;
        let k_seq = committed + n_block;
        // Query 0's floor: it may attend back to `committed - window + 1`.
        let lo = match window {
            Some(w) => committed.saturating_sub(w - 1),
            None => 0,
        };
        let span = k_seq - lo;

        let q4 = q.reshape((n_kv, g, n_block, hd))?;
        let k4 = k_cache.narrow(1, lo, span)?.unsqueeze(1)?; // [n_kv, 1, span, hd]
        let v4 = v_cache.narrow(1, lo, span)?.unsqueeze(1)?; // [n_kv, 1, span, hd]

        let scores = q4
            .broadcast_matmul(&k4.transpose(2, 3)?)?
            .affine(scale as f64, 0.0)?;
        let scores = match self.window_mask(committed, n_block, lo, span, window)? {
            Some(mask) => scores.broadcast_add(&mask)?,
            None => scores,
        };
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.broadcast_matmul(&v4)?; // [n_kv, g, nb, hd]
        Ok(out.reshape((n_head, n_block, hd))?)
    }

    /// Additive mask `[1, 1, n_block, span]` (0 attend / -inf block) over the
    /// narrowed key view starting at absolute position `lo`: query `i` blocks
    /// absolute key positions below `committed + i + 1 - window`. `None` when
    /// nothing in the view is out of window for any query — which is every round
    /// until the context outgrows the window, and every round on a full layer.
    fn window_mask(
        &self,
        committed: usize,
        n_block: usize,
        lo: usize,
        span: usize,
        window: Option<usize>,
    ) -> Result<Option<Tensor>> {
        let Some(w) = window else {
            return Ok(None);
        };
        // The deepest floor is the last query's; if even that is at or below the
        // view's start, no key in the view is masked for any query.
        if committed + n_block <= w {
            return Ok(None);
        }
        let mut data = vec![0f32; n_block * span];
        for i in 0..n_block {
            let floor = (committed + i + 1).saturating_sub(w); // first visible absolute position
            for c in lo..floor.min(lo + span) {
                data[i * span + (c - lo)] = f32::NEG_INFINITY;
            }
        }
        Ok(Some(Tensor::from_vec(
            data,
            (1, 1, n_block, span),
            &self.device,
        )?))
    }

    /// SwiGLU: `down(silu(gate(x)) * up(x))`, x `[nb, n_embd]` -> `[nb, n_embd]`.
    fn swiglu(&self, layer: &LayerWeights, x: &Tensor) -> Result<Tensor> {
        let gate = linear(x, &layer.ffn_gate)?;
        let up = linear(x, &layer.ffn_up)?;
        let hidden = (candle_nn::ops::silu(&gate)? * up)?;
        linear(&hidden, &layer.ffn_down)
    }

    /// Plain NEOX rope over the full head_dim (rotate-half: dim i pairs with
    /// i + head_dim/2), for `x [n_head_or_kv, seq, head_dim]` at absolute
    /// positions `pos..pos+seq`. Bit-mirrors the engine's NEOX convention
    /// (src/rope.rs), computed with plain candle ops.
    fn rope(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let (_, seq, hd) = x.dims3()?;
        let half = hd / 2;
        let cos = self.rope_cos.narrow(0, pos, seq)?.reshape((1, seq, half))?;
        let sin = self.rope_sin.narrow(0, pos, seq)?.reshape((1, seq, half))?;
        let x1 = x.narrow(2, 0, half)?;
        let x2 = x.narrow(2, half, half)?;
        let o1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let o2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        Ok(Tensor::cat(&[&o1, &o2], 2)?.contiguous()?)
    }
}

/// One layer's K/V byte images as [`DrafterImage::prefix`] hands them out:
/// borrowed from the stored image when the whole thing is wanted, owned when a
/// shorter prefix had to be gathered.
type DrafterBytes<'a> = (Cow<'a, [u8]>, Cow<'a, [u8]>);

/// Which drafter wrote an image, which is also what its bytes mean.
///
/// The two kinds' caches are not the same record wearing different numbers. A
/// DFlash image is f32 (its cache is stored exact — the drafter is tiny and it
/// keeps the naive-reference tests bit-tight) across the sidecar's several
/// layers; an MTP image is f16 (the head reuses the trunk's full-attention
/// `LayerCache`, so it inherits the trunk's dtype) across exactly one, and it
/// carries a hidden state besides its KV. Two records that differ in element
/// size, plane count AND field list must not be distinguishable only by
/// arithmetic that happens not to add up, so the kind is stored and checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrafterImageKind {
    Dflash,
    Mtp,
}

impl DrafterImageKind {
    /// The name an error uses when an image meets the wrong drafter.
    pub fn name(self) -> &'static str {
        match self {
            DrafterImageKind::Dflash => "DFlash",
            DrafterImageKind::Mtp => "MTP",
        }
    }

    /// Bytes per stored cache element. See the type's own doc for why they
    /// differ.
    fn elem_size(self) -> usize {
        match self {
            DrafterImageKind::Dflash => size_of::<f32>(),
            DrafterImageKind::Mtp => size_of::<half::f16>(),
        }
    }

    /// The on-disk discriminant. Written first so a reader learns how to size
    /// everything after it.
    fn tag(self) -> usize {
        match self {
            DrafterImageKind::Dflash => 1,
            DrafterImageKind::Mtp => 2,
        }
    }

    fn from_tag(tag: usize) -> Result<Self> {
        match tag {
            1 => Ok(DrafterImageKind::Dflash),
            2 => Ok(DrafterImageKind::Mtp),
            other => anyhow::bail!("drafter_image: unknown drafter kind tag {other}"),
        }
    }
}

/// Host-RAM image of a drafter's committed KV rows `[0, pos)`. A drafter's
/// cache is plain position-indexed storage with no ring, so these rows ARE the
/// whole of its KV state: a conversation whose rows are copied out here resumes
/// bit-exactly through the matching drafter's `import_cache`.
///
/// Like `HostFullKv` and `HostSnapshot` this is an on-disk record, and its layout
/// is part of the contract: the kind, then one `(K, V)` pair per drafter layer in
/// layer order, little-endian, row-major `(n_kv_head, pos, head_dim)` at the
/// kind's element size, then the carry. Elements are stored at the dtype the
/// cache holds them in — narrowing would put rounding into a round trip.
/// `write_to`/`read_from` frame exactly that.
pub struct DrafterImage {
    kind: DrafterImageKind,
    /// Positions the image covers: rows `[0, pos)` of every drafter layer.
    pub pos: usize,
    /// KV heads and head dim the planes are laid out over. Kept in the image so
    /// it describes itself — an on-disk record cannot lean on a live config.
    n_kv_head: usize,
    head_dim: usize,
    /// Width of the carry hidden, 0 on a DFlash image. Stored for the same
    /// reason the KV shape is: the record has to size its own fields without a
    /// live config to ask.
    hidden: usize,
    planes: Vec<(Vec<u8>, Vec<u8>)>,
    /// The MTP head's shift-right-by-one carry: the target's post-final-norm
    /// hidden at position `pos - 1`, little-endian f32, empty on a DFlash image.
    ///
    /// It rides along because the KV alone is not a resumable MTP head: the row
    /// at `pos` is built from the hidden at `pos - 1`, so an image without this
    /// restores a head that holds the right context and still cannot take
    /// another token. A block drafter has no such coupling — every one of its
    /// rows is a function of that position's taps alone — which is why the field
    /// is empty for one kind and load-bearing for the other.
    carry: Vec<u8>,
}

impl DrafterImage {
    /// Assemble a DFlash image, checking every plane against the shape it
    /// claims. The record is only self-describing if those agree, and every
    /// later read (`prefix`, `byte_len`, the on-disk write) trusts them.
    pub(crate) fn new_dflash(
        pos: usize,
        n_kv_head: usize,
        head_dim: usize,
        planes: Vec<(Vec<u8>, Vec<u8>)>,
    ) -> Result<Self> {
        Self::new(
            DrafterImageKind::Dflash,
            pos,
            n_kv_head,
            head_dim,
            0,
            planes,
            Vec::new(),
        )
    }

    /// Assemble an MTP image: the head's single layer plus the carry hidden,
    /// which is as much a part of its state as the rows are.
    pub(crate) fn new_mtp(
        pos: usize,
        n_kv_head: usize,
        head_dim: usize,
        hidden: usize,
        planes: Vec<(Vec<u8>, Vec<u8>)>,
        carry: Vec<u8>,
    ) -> Result<Self> {
        Self::new(
            DrafterImageKind::Mtp,
            pos,
            n_kv_head,
            head_dim,
            hidden,
            planes,
            carry,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        kind: DrafterImageKind,
        pos: usize,
        n_kv_head: usize,
        head_dim: usize,
        hidden: usize,
        planes: Vec<(Vec<u8>, Vec<u8>)>,
        carry: Vec<u8>,
    ) -> Result<Self> {
        let want = elem_bytes_for(kind, &[n_kv_head, pos, head_dim])?;
        for (idx, (k, v)) in planes.iter().enumerate() {
            ensure!(
                k.len() == want && v.len() == want,
                "drafter_image: layer {idx} holds {} K and {} V bytes, expected {want} for \
                 ({n_kv_head}, {pos}, {head_dim}) at {} bytes per element",
                k.len(),
                v.len(),
                kind.elem_size()
            );
        }
        if kind == DrafterImageKind::Dflash {
            ensure!(
                hidden == 0,
                "drafter_image: a DFlash image carries no hidden state, but declares a width of \
                 {hidden}"
            );
        } else {
            ensure!(
                hidden > 0,
                "drafter_image: an MTP image without a carry width cannot describe its own carry"
            );
        }
        let want_carry = hidden
            .checked_mul(size_of::<f32>())
            .context("drafter_image: the carry's byte count overflows usize")?;
        ensure!(
            carry.len() == want_carry,
            "drafter_image: the carry is {} bytes, expected {want_carry} for a {hidden}-wide f32 \
             hidden row",
            carry.len()
        );
        Ok(Self {
            kind,
            pos,
            n_kv_head,
            head_dim,
            hidden,
            planes,
            carry,
        })
    }

    /// Which drafter wrote this, and therefore which can read it back.
    pub fn kind(&self) -> DrafterImageKind {
        self.kind
    }

    /// The MTP carry hidden, little-endian f32; empty on a DFlash image.
    pub(crate) fn carry(&self) -> &[u8] {
        &self.carry
    }

    /// The shape the planes are laid out over, for an importer checking that
    /// this image describes ITS cache. Byte length does not settle that on its
    /// own — (8, 128) and (4, 256) hold the same number of values.
    pub(crate) fn n_kv_head(&self) -> usize {
        self.n_kv_head
    }

    pub(crate) fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Layer `idx`'s whole K/V planes, for an importer that takes the image
    /// entire rather than a prefix of it.
    pub(crate) fn plane(&self, idx: usize) -> Result<(&[u8], &[u8])> {
        let (k, v) = self.planes.get(idx).with_context(|| {
            format!(
                "drafter_image: layer {idx} is past the image's {} layers",
                self.planes.len()
            )
        })?;
        Ok((k, v))
    }

    /// Host bytes this image occupies — 40 KiB per token on the 27B DFlash
    /// sidecar (5 layers x K/V x 8 heads x 128 dims x f32), 48 KiB on the
    /// 35B-A3B's six layers, 4 KiB on the MTP head (1 layer x K/V x 4 heads x
    /// 256 dims x f16).
    pub fn byte_len(&self) -> usize {
        self.planes
            .iter()
            .map(|(k, v)| k.len() + v.len())
            .sum::<usize>()
            + self.carry.len()
    }

    /// Number of drafter layers covered.
    pub fn layers(&self) -> usize {
        self.planes.len()
    }

    /// K/V bytes for rows `[0, pos)` of layer `idx`. The whole plane is borrowed
    /// when `pos` is its full length; a shorter prefix is gathered because the
    /// stored layout is `(n_kv_head, self.pos, head_dim)` — each head's rows are
    /// one contiguous run, so a prefix of the positions is `n_kv_head` runs, not
    /// a prefix of the buffer. The stored image is left intact either way.
    fn prefix(&self, idx: usize, pos: usize) -> Result<DrafterBytes<'_>> {
        ensure!(
            pos <= self.pos,
            "drafter_image: requested pos {pos} exceeds the image's {} positions",
            self.pos
        );
        let (k, v) = self.planes.get(idx).with_context(|| {
            format!(
                "drafter_image: layer {idx} is past the image's {} layers",
                self.planes.len()
            )
        })?;
        if pos == self.pos {
            return Ok((Cow::Borrowed(k), Cow::Borrowed(v)));
        }
        let stride = self.pos * self.head_dim * self.kind.elem_size();
        let take = pos * self.head_dim * self.kind.elem_size();
        let gather = |src: &[u8]| {
            let mut out = Vec::with_capacity(self.n_kv_head * take);
            for head in 0..self.n_kv_head {
                out.extend_from_slice(&src[head * stride..head * stride + take]);
            }
            out
        };
        Ok((Cow::Owned(gather(k)), Cow::Owned(gather(v))))
    }

    /// Frame this image into `w`: the kind, the position, the shape its planes
    /// are laid out over, the plane count, then K and V per layer in layer
    /// order, then the carry.
    ///
    /// The kind leads because it decides how many bytes an element of everything
    /// after it takes.
    pub(crate) fn write_to(&self, w: &mut impl Write) -> std::io::Result<()> {
        write_count(w, self.kind.tag())?;
        write_count(w, self.pos)?;
        write_count(w, self.n_kv_head)?;
        write_count(w, self.head_dim)?;
        write_count(w, self.hidden)?;
        write_count(w, self.planes.len())?;
        for (k, v) in &self.planes {
            write_plane(w, k)?;
            write_plane(w, v)?;
        }
        write_plane(w, &self.carry)
    }

    /// Bytes `write_to` produces, for the container directory that has to declare
    /// this record's length before the planes are streamed out.
    pub(crate) fn serialized_len(&self) -> usize {
        6 * size_of::<u64>()
            + self
                .planes
                .iter()
                .map(|(k, v)| plane_bytes(k.len()) + plane_bytes(v.len()))
                .sum::<usize>()
            + plane_bytes(self.carry.len())
    }

    /// Read back a `write_to` body of `len` bytes. The shape and plane lengths are
    /// untrusted input, bounded by the record's framed length while reading and
    /// then reconciled by `new`, which is where a shape that does not describe the
    /// bytes is rejected.
    pub(crate) fn read_from(src: impl Read, len: u64) -> Result<Self> {
        let mut r = RecordReader::new(src, len);
        let kind = DrafterImageKind::from_tag(r.count()?)?;
        let pos = plausible("position count", r.count()?, MAX_STORED_POS)?;
        let n_kv_head = plausible("KV head count", r.count()?, MAX_STORED_KV_HEADS)?;
        let head_dim = plausible("head dim", r.count()?, MAX_STORED_HEAD_DIM)?;
        let hidden = plausible("hidden size", r.count()?, MAX_STORED_HIDDEN)?;
        let plane_count = plausible("layer count", r.count()?, MAX_STORED_LAYERS)?;
        let plane = elem_bytes_for(kind, &[n_kv_head, pos, head_dim])?;
        // Grown rather than reserved, for the reason `plausible` documents.
        let mut planes = Vec::new();
        for _ in 0..plane_count {
            let k = r.plane_of(plane)?;
            let v = r.plane_of(plane)?;
            planes.push((k, v));
        }
        // Sized from the declared width like every other plane, so a carry that
        // does not describe a hidden row is refused before it is allocated.
        let carry = r.plane_of(hidden * size_of::<f32>())?;
        r.finish()?;
        Self::new(kind, pos, n_kv_head, head_dim, hidden, planes, carry)
    }
}

/// The widest residual stream this build will read a stored carry for. The
/// dense checkpoints are 5120 and the MoE 2048; this is well clear of both and
/// still small enough that a corrupted field cannot ask for a large allocation.
const MAX_STORED_HIDDEN: usize = 1 << 16;

/// Bytes a `kind`-dtype tensor of `dims` occupies, erroring rather than wrapping
/// on overflow.
///
/// A [`DrafterImage`] computes this from its OWN declared shape. In process that
/// comes from a real allocation and cannot overflow, but the same record is the
/// on-disk format, where the shape fields are untrusted input — a wrapped product
/// would size a buffer that a later index walks straight past.
fn elem_bytes_for(kind: DrafterImageKind, dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(kind.elem_size(), |acc, &d| {
        acc.checked_mul(d)
            .with_context(|| format!("drafter_image: byte count for {dims:?} overflows usize"))
    })
}

/// Raw little-endian f32 bytes of rows `[0, len)` of a `(n_head_kv, slots,
/// head_dim)` drafter cache plane.
///
/// A short prefix is compacted on the device first, and that is not optional:
/// reading a tensor out to host copies its WHOLE storage and only then applies
/// the layout, so a `narrow` view of a max_ctx-slot cache would drag the entire
/// allocation across to fetch a few thousand rows. Rows that fill the plane have
/// nothing to trim, so exactly one device-side copy happens either way, never
/// two.
fn f32_rows(plane: &Tensor, len: usize) -> Result<Vec<u8>> {
    let slots = plane.dim(1)?;
    let view = plane.narrow(1, 0, len)?;
    if len == slots {
        f32_bytes(&view)
    } else {
        // A real strided copy into fresh storage, unlike `contiguous()`, which is
        // a shallow Arc clone on Metal when the input already passes the
        // contiguity check (CLAUDE.md trap).
        f32_bytes(&view.force_contiguous()?)
    }
}

/// Raw little-endian f32 bytes of `t` in row-major order — a [`DrafterImage`]
/// plane's storage format.
///
/// `t` must own its storage rather than view part of a larger allocation: the
/// readback copies the whole storage and only then applies the layout. Callers
/// holding a slice of a cache plane go through `f32_rows`.
pub(crate) fn f32_bytes(t: &Tensor) -> Result<Vec<u8>> {
    ensure!(
        t.dtype() == DType::F32,
        "drafter_image: drafter caches are f32, got {:?}",
        t.dtype()
    );
    let values = t.flatten_all()?.to_vec1::<f32>()?;
    // SAFETY: `f32` has a fixed 4-byte layout with no padding and no niche, so a
    // slice of them is a valid slice of 4x the bytes. On a little-endian target —
    // the only kind this engine builds for — those bytes ARE the little-endian
    // encoding the image format calls for, so this is byte-for-byte what a
    // per-element `to_le_bytes` loop produces and an order of magnitude cheaper
    // (one memcpy instead of hundreds of millions of four-byte appends).
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            values.len() * size_of::<f32>(),
        )
    };
    Ok(bytes.to_vec())
}

/// A device tensor of `shape` read from little-endian f32 `bytes`, the inverse of
/// [`f32_bytes`].
pub(crate) fn f32_tensor(bytes: &[u8], shape: &[usize], device: &Device) -> Result<Tensor> {
    let elems: usize = shape.iter().product();
    ensure!(
        bytes.len() == elems * size_of::<f32>(),
        "drafter_image: {} bytes is not {elems} f32 values for shape {shape:?}",
        bytes.len()
    );
    Ok(Tensor::from_raw_buffer(bytes, DType::F32, shape, device)?)
}

/// `y = x @ wᵀ` for a dense `[out, in]` weight and rank-2 `x [t, in]` f32,
/// dispatched on the weight's dtype: an f16 or bf16 weight goes through the
/// vendored mixed-dtype kernels (`crate::ops::matmul_f16` / `matmul_bf16`, f32
/// accumulate, Metal only — bf16 is the mmap-alias default), an f32 weight
/// through plain candle matmul. The vendored kernels need a contiguous f32
/// `x`; every call site already passes a freshly produced contiguous tensor,
/// but a strided input is made contiguous before the kernel dispatch.
fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    match w.dtype() {
        DType::F16 | DType::BF16 => {
            let x = if x.is_contiguous() {
                x.clone()
            } else {
                x.contiguous()?
            };
            if w.dtype() == DType::F16 {
                Ok(crate::ops::matmul_f16(w, &x)?)
            } else {
                Ok(crate::ops::matmul_bf16(w, &x)?)
            }
        }
        _ => Ok(x.matmul(&w.t()?)?),
    }
}

/// Plain NEOX rope cos/sin tables `[max_ctx, head_dim/2]` f32, mscale 1 (no
/// YaRN): `inv_freq[j] = theta^(-2j/head_dim)`, angle `p * inv_freq[j]`.
fn rope_tables(
    theta: f32,
    head_dim: usize,
    max_ctx: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let base = theta as f64;
    let inv_freq: Vec<f64> = (0..half)
        .map(|j| base.powf(-(2.0 * j as f64) / head_dim as f64))
        .collect();
    let mut cos = vec![0f32; max_ctx * half];
    let mut sin = vec![0f32; max_ctx * half];
    for p in 0..max_ctx {
        for j in 0..half {
            let angle = p as f64 * inv_freq[j];
            cos[p * half + j] = angle.cos() as f32;
            sin[p * half + j] = angle.sin() as f32;
        }
    }
    Ok((
        Tensor::from_vec(cos, (max_ctx, half), device)?,
        Tensor::from_vec(sin, (max_ctx, half), device)?,
    ))
}

/// Read a GGUF tensor by fully-qualified name (with `.weight` suffix) into a
/// dense f32 tensor. BF16 widens exactly to f32; F32 passes through. The GGUF
/// stores matmul weights `[out, in]` and norms `[dim]`, which this preserves.
/// Aliases drafter tensor `name` no-copy off the mapping IF it is stored BF16
/// and rank-2 — the matmul planes; anything else (the F32 norms) returns
/// `None` for the copying reader. The plane's bytes are scanned BEFORE
/// aliasing (`ensure_bf16_fits_f16`): the bf16 kernels' tensor-path gemm
/// stages bf16 -> half, so a value outside f16's range would become inf
/// mid-kernel — the same bound the materialize path enforces as a hard load
/// error via its finiteness check. The scan doubles as the page-cache warm-up
/// the upload pass used to provide.
fn bf16_alias_tensor(
    src: &Arc<MmapSource>,
    content: &Content,
    name: &str,
    device: &Device,
) -> Result<Option<Tensor>> {
    let full = format!("{name}.weight");
    let info = content
        .tensor_infos
        .get(&full)
        .with_context(|| format!("drafter tensor {full} not found"))?;
    if info.ggml_dtype != GgmlDType::BF16 {
        return Ok(None);
    }
    let dims = info.shape.dims().to_vec();
    let [out_dim, in_dim] = dims[..] else {
        // A non-rank-2 BF16 tensor would be a novel file layout; let the
        // copying reader handle (and shape-check) it rather than alias blindly.
        return Ok(None);
    };
    let abs_off = (content.tensor_data_offset + info.offset) as usize;
    let len = out_dim * in_dim * DType::BF16.size_in_bytes();
    ensure_bf16_fits_f16(src.bytes(abs_off, len)?, &full)?;
    dense_alias_tensor(src, device, abs_off, out_dim, in_dim, DType::BF16)
        .map(Some)
        .with_context(|| format!("mmap-aliasing {full}"))
}

/// Hard-errors if any bf16 value in `bytes` (little-endian u16 pairs) is
/// non-finite or would overflow f16: masked magnitude bits >= 0x4780 ⇔
/// |value| >= 65536, the smallest bf16 magnitude that rounds to f16 inf (the
/// next value down, 65280, is representable; inf/nan sort above the bound).
/// The scan is an OVERFLOW guard, not an exactness proof: an admitted value
/// converts to f16 exactly in f16's NORMAL range (7 stored mantissa bits ⊂
/// f16's 10), but below 2^-14 the f16 grid is the fixed-spacing subnormal one
/// (2^-24), so values under ~2^-17 can round and values under 2^-24 flush.
/// The tensor-path gemm's bf16 -> half staging therefore treats such weights
/// EXACTLY as the old materialize-to-f16 path did, while the widening gemv
/// and the float-tile classic gemm keep the full bf16 value — a seq-boundary
/// asymmetry pinned by ops/bf16.rs
/// `bf16_subnormal_range_weights_follow_the_documented_split`. Measured on
/// the official drafter: 24,125 of 1.11e9 weights (2.2e-5, all |w| ≲ 1e-5)
/// sit in that range — far below spec-verify's noise floor. Parallel scan,
/// memory-bandwidth bound (~2.2GB total for the official drafter).
pub(crate) fn ensure_bf16_fits_f16(bytes: &[u8], name: &str) -> Result<()> {
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(16);
    // Even chunk sizes keep every chunk element-aligned.
    let chunk = bytes.len().div_ceil(n_threads).next_multiple_of(2).max(2);
    let bad = std::thread::scope(|s| {
        let handles: Vec<_> = bytes
            .chunks(chunk)
            .enumerate()
            .map(|(i, c)| {
                s.spawn(move || {
                    c.chunks_exact(2)
                        .position(|p| (u16::from_le_bytes([p[0], p[1]]) & 0x7FFF) >= 0x4780)
                        .map(|off| i * (chunk / 2) + off)
                })
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|h| h.join().expect("bf16 range-scan thread panicked"))
            .min()
    });
    if let Some(i) = bad {
        let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
        let v = half::bf16::from_bits(bits).to_f32();
        bail!(
            "drafter tensor {name} holds a bf16 value outside f16 range at element {i} \
             ({v}, bits {bits:#06x}); the bf16 kernels stage weights to f16, so this \
             checkpoint needs the full-f32 path (XWEN_DRAFT_F32)"
        );
    }
    Ok(())
}

fn read_tensor_f32(
    file: &mut File,
    content: &Content,
    name: &str,
    device: &Device,
) -> Result<Tensor> {
    let full = format!("{name}.weight");
    let info = content
        .tensor_infos
        .get(&full)
        .with_context(|| format!("drafter tensor {full} not found"))?;
    let dims = info.shape.dims().to_vec();
    let elems: usize = dims.iter().product();
    let start = content.tensor_data_offset + info.offset;
    file.seek(SeekFrom::Start(start))
        .with_context(|| format!("seeking to {full}"))?;

    let values: Vec<f32> = match info.ggml_dtype {
        GgmlDType::F32 => {
            let mut raw = vec![0u8; elems * 4];
            file.read_exact(&mut raw)
                .with_context(|| format!("reading {full}"))?;
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        GgmlDType::BF16 => {
            let mut raw = vec![0u8; elems * 2];
            file.read_exact(&mut raw)
                .with_context(|| format!("reading {full}"))?;
            raw.chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect()
        }
        other => {
            bail!("drafter tensor {full} has unsupported dtype {other:?} (expected F32 or BF16)")
        }
    };
    Ok(Tensor::from_vec(values, dims, device)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // --- synthetic weight plumbing -------------------------------------------

    /// Deterministic pseudo-random f32s in roughly [-0.5, 0.5] (LCG, no deps).
    fn seeded(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / u32::MAX as f32) - 0.5
            })
            .collect()
    }

    /// A synthetic config with full attention on every layer. `with_window`
    /// turns it into the sidecars' shape (every layer windowed but the last).
    fn tiny_cfg(
        n_layer: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        n_embd: usize,
        n_ff: usize,
    ) -> DflashConfig {
        DflashConfig {
            n_layer,
            n_embd,
            n_head,
            n_head_kv: n_kv,
            head_dim: hd,
            n_ff,
            rms_eps: 1e-6,
            rope_theta: 10_000_000.0,
            sliding_window: None,
            swa_layers: vec![false; n_layer],
            block_size: 16,
            target_layers: vec![2, 11],
            mask_token_id: 248077,
            context_length: 1024,
        }
    }

    /// The shipped per-layer pattern — windowed everywhere but the final layer.
    fn with_window(mut cfg: DflashConfig, window: usize) -> DflashConfig {
        cfg.sliding_window = Some(window);
        cfg.swa_layers = (0..cfg.n_layer).map(|il| il + 1 < cfg.n_layer).collect();
        cfg
    }

    /// A synthetic weight map for `cfg`: every tensor filled with seeded pseudo-
    /// random f32 (norm weights near 1.0 so the RMSNorms stay well-conditioned).
    fn synth_weights(cfg: &DflashConfig, dev: &Device) -> HashMap<String, Tensor> {
        let mut m = HashMap::new();
        let seed = std::cell::Cell::new(1u64);
        let next = || {
            seed.set(seed.get() + 1);
            seed.get()
        };
        let mat = |rows: usize, cols: usize| {
            Tensor::from_vec(seeded(rows * cols, next()), (rows, cols), dev).unwrap()
        };
        let (n_embd, hd, n_head, n_kv, n_ff) = (
            cfg.n_embd,
            cfg.head_dim,
            cfg.n_head,
            cfg.n_head_kv,
            cfg.n_ff,
        );
        let n_aux = cfg.target_layers.len();

        m.insert("fc".to_string(), mat(n_embd, n_aux * n_embd));
        // Norm-like weights: 1.0 + small perturbation.
        let norm_w = |dim: usize| {
            let v: Vec<f32> = seeded(dim, next()).iter().map(|x| 1.0 + 0.1 * x).collect();
            Tensor::from_vec(v, dim, dev).unwrap()
        };
        m.insert("enc.output_norm".to_string(), norm_w(n_embd));
        m.insert("output_norm".to_string(), norm_w(n_embd));

        for il in 0..cfg.n_layer {
            let p = |n: &str| format!("blk.{il}.{n}");
            m.insert(p("attn_norm"), norm_w(n_embd));
            m.insert(p("attn_q"), mat(n_head * hd, n_embd));
            m.insert(p("attn_k"), mat(n_kv * hd, n_embd));
            m.insert(p("attn_v"), mat(n_kv * hd, n_embd));
            m.insert(p("attn_output"), mat(n_embd, n_head * hd));
            m.insert(p("attn_q_norm"), norm_w(hd));
            m.insert(p("attn_k_norm"), norm_w(hd));
            m.insert(p("ffn_norm"), norm_w(n_embd));
            m.insert(p("ffn_gate"), mat(n_ff, n_embd));
            m.insert(p("ffn_up"), mat(n_ff, n_embd));
            m.insert(p("ffn_down"), mat(n_embd, n_ff));
        }
        m
    }

    fn drafter_from_map(
        cfg: DflashConfig,
        m: &HashMap<String, Tensor>,
        dev: &Device,
        max_ctx: usize,
        dtype: WeightDtype,
    ) -> DflashDrafter {
        DflashDrafter::build(cfg, dev, max_ctx, dtype, |name| {
            m.get(name)
                .cloned()
                .with_context(|| format!("synthetic tensor {name} missing"))
        })
        .unwrap()
    }

    fn to_vec(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len());
        let num: f64 = got
            .iter()
            .zip(want)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
        (num / den.max(1e-30)) as f32
    }

    // --- scalar f64 reference (fully independent of the tensor impl) ----------

    /// Host-side view of one layer's weights, row-major f64.
    struct RefLayer {
        attn_norm: Vec<f64>,
        wq: Vec<f64>,
        wk: Vec<f64>,
        wv: Vec<f64>,
        wo: Vec<f64>,
        q_norm: Vec<f64>,
        k_norm: Vec<f64>,
        ffn_norm: Vec<f64>,
        ffn_gate: Vec<f64>,
        ffn_up: Vec<f64>,
        ffn_down: Vec<f64>,
    }

    fn host(m: &HashMap<String, Tensor>, name: &str) -> Vec<f64> {
        to_vec(m.get(name).unwrap())
            .iter()
            .map(|x| *x as f64)
            .collect()
    }

    /// `y[o] = Σ_i w[o*in + i] * x[i]`, w row-major `[out, in]`.
    fn matvec(w: &[f64], x: &[f64], out: usize, inn: usize) -> Vec<f64> {
        (0..out)
            .map(|o| (0..inn).map(|i| w[o * inn + i] * x[i]).sum())
            .collect()
    }

    fn rmsnorm(x: &[f64], weight: &[f64], eps: f64) -> Vec<f64> {
        let ms: f64 = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let s = 1.0 / (ms + eps).sqrt();
        x.iter().zip(weight).map(|(v, w)| v * s * w).collect()
    }

    /// Rotate-half NEOX rope on a single head vector `[hd]` at absolute `pos`.
    fn rope_vec(v: &[f64], pos: usize, hd: usize, theta: f64) -> Vec<f64> {
        let half = hd / 2;
        let mut out = vec![0.0; hd];
        for j in 0..half {
            let inv = theta.powf(-(2.0 * j as f64) / hd as f64);
            let (c, s) = ((pos as f64 * inv).cos(), (pos as f64 * inv).sin());
            let (x1, x2) = (v[j], v[j + half]);
            out[j] = x1 * c - x2 * s;
            out[j + half] = x1 * s + x2 * c;
        }
        out
    }

    fn silu_s(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    /// A from-scratch f64 draft forward: inject `ctx` context tokens, then run
    /// the noise block through the decoder. Returns `[n_block][n_embd]`.
    #[allow(clippy::too_many_arguments)]
    fn naive_draft(
        cfg: &DflashConfig,
        m: &HashMap<String, Tensor>,
        ctx: &[Vec<f64>],   // [committed][n_embd]
        noise: &[Vec<f64>], // [n_block][n_embd]
    ) -> Vec<Vec<f64>> {
        let (n_embd, hd, n_head, n_kv, n_ff) = (
            cfg.n_embd,
            cfg.head_dim,
            cfg.n_head,
            cfg.n_head_kv,
            cfg.n_ff,
        );
        let (eps, theta) = (cfg.rms_eps, cfg.rope_theta as f64);
        let g = n_head / n_kv;
        let scale = 1.0 / (hd as f64).sqrt();
        let committed = ctx.len();
        let n_block = noise.len();

        let layers: Vec<RefLayer> = (0..cfg.n_layer)
            .map(|il| {
                let p = |n: &str| host(m, &format!("blk.{il}.{n}"));
                RefLayer {
                    attn_norm: p("attn_norm"),
                    wq: p("attn_q"),
                    wk: p("attn_k"),
                    wv: p("attn_v"),
                    wo: p("attn_output"),
                    q_norm: p("attn_q_norm"),
                    k_norm: p("attn_k_norm"),
                    ffn_norm: p("ffn_norm"),
                    ffn_gate: p("ffn_gate"),
                    ffn_up: p("ffn_up"),
                    ffn_down: p("ffn_down"),
                }
            })
            .collect();

        // Per-layer context K/V from the fused ctx: the injection path projects
        // the encoder output raw (no attn_norm) and QK-norms + ropes K only.
        // ctxk[il][kv] = Vec of [hd] per context token.
        let mut ctxk = vec![vec![Vec::<Vec<f64>>::new(); n_kv]; cfg.n_layer];
        let mut ctxv = vec![vec![Vec::<Vec<f64>>::new(); n_kv]; cfg.n_layer];
        for (il, layer) in layers.iter().enumerate() {
            for (t, tok) in ctx.iter().enumerate() {
                let nn = tok.clone();
                let kf = matvec(&layer.wk, &nn, n_kv * hd, n_embd);
                let vf = matvec(&layer.wv, &nn, n_kv * hd, n_embd);
                for kv in 0..n_kv {
                    let kh = &kf[kv * hd..(kv + 1) * hd];
                    let kn = rmsnorm(kh, &layer.k_norm, eps);
                    ctxk[il][kv].push(rope_vec(&kn, t, hd, theta));
                    ctxv[il][kv].push(vf[kv * hd..(kv + 1) * hd].to_vec());
                }
            }
        }

        let mut hs: Vec<Vec<f64>> = noise.to_vec();
        for (il, layer) in layers.iter().enumerate() {
            // Per-token projections for the block.
            let mut qh = vec![vec![Vec::<f64>::new(); n_head]; n_block]; // [b][head] -> [hd]
            let mut nk = vec![vec![Vec::<f64>::new(); n_kv]; n_block];
            let mut nv = vec![vec![Vec::<f64>::new(); n_kv]; n_block];
            let window = cfg.layer_window(il);
            for b in 0..n_block {
                let pos = committed + b;
                let nn = rmsnorm(&hs[b], &layer.attn_norm, eps);
                let qf = matvec(&layer.wq, &nn, n_head * hd, n_embd);
                let kf = matvec(&layer.wk, &nn, n_kv * hd, n_embd);
                let vf = matvec(&layer.wv, &nn, n_kv * hd, n_embd);
                for h in 0..n_head {
                    let qn = rmsnorm(&qf[h * hd..(h + 1) * hd], &layer.q_norm, eps);
                    qh[b][h] = rope_vec(&qn, pos, hd, theta);
                }
                for kv in 0..n_kv {
                    let kn = rmsnorm(&kf[kv * hd..(kv + 1) * hd], &layer.k_norm, eps);
                    nk[b][kv] = rope_vec(&kn, pos, hd, theta);
                    nv[b][kv] = vf[kv * hd..(kv + 1) * hd].to_vec();
                }
            }

            // Attention + o_proj + residual + FFN, per block token.
            for b in 0..n_block {
                let p1 = committed + b;
                let mut attn_cat = vec![0.0f64; n_head * hd];
                for h in 0..n_head {
                    let kv = h / g;
                    // Every key position the window admits, in position order:
                    // non-causal, so the whole noise block is visible to every
                    // query and only the past side is cut.
                    let visible: Vec<usize> = (0..committed + n_block)
                        .filter(|&p0| match window {
                            Some(w) => (p1 as i64 - p0 as i64) < (w as i64),
                            None => true,
                        })
                        .collect();
                    let key = |p0: usize| -> &Vec<f64> {
                        if p0 < committed {
                            &ctxk[il][kv][p0]
                        } else {
                            &nk[p0 - committed][kv]
                        }
                    };
                    let val = |p0: usize| -> &Vec<f64> {
                        if p0 < committed {
                            &ctxv[il][kv][p0]
                        } else {
                            &nv[p0 - committed][kv]
                        }
                    };
                    let scores: Vec<f64> = visible
                        .iter()
                        .map(|&p0| {
                            let s: f64 = qh[b][h].iter().zip(key(p0)).map(|(a, c)| a * c).sum();
                            s * scale
                        })
                        .collect();
                    // Softmax.
                    let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exps: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
                    let sum: f64 = exps.iter().sum();
                    let mut ov = vec![0.0f64; hd];
                    for (idx, w) in exps.iter().map(|e| e / sum).enumerate() {
                        let vv = val(visible[idx]);
                        for d in 0..hd {
                            ov[d] += w * vv[d];
                        }
                    }
                    attn_cat[h * hd..(h + 1) * hd].copy_from_slice(&ov);
                }
                let cur = matvec(&layer.wo, &attn_cat, n_embd, n_head * hd);
                let ffn_inp: Vec<f64> = cur.iter().zip(&hs[b]).map(|(a, b)| a + b).collect();
                let fn_ = rmsnorm(&ffn_inp, &layer.ffn_norm, eps);
                let gate = matvec(&layer.ffn_gate, &fn_, n_ff, n_embd);
                let up = matvec(&layer.ffn_up, &fn_, n_ff, n_embd);
                let hidden: Vec<f64> = gate
                    .iter()
                    .zip(&up)
                    .map(|(gg, uu)| silu_s(*gg) * uu)
                    .collect();
                let down = matvec(&layer.ffn_down, &hidden, n_embd, n_ff);
                hs[b] = down.iter().zip(&ffn_inp).map(|(a, b)| a + b).collect();
            }
        }

        hs.iter()
            .map(|h| rmsnorm(h, &host(m, "output_norm"), eps))
            .collect()
    }

    // --- tests ----------------------------------------------------------------

    /// The geometry one shipped sidecar must parse to. Both are checked by
    /// `real_file_load_and_shapes`, which is the gate on the whole adaptation:
    /// these numbers come from the GGUF headers of the ggml-org files.
    struct SidecarSpec {
        model: crate::hub::Model,
        n_layer: usize,
        n_embd: usize,
        n_ff: usize,
        target_layers: Vec<usize>,
        tap_layers: Vec<usize>,
        mask_token_id: u32,
        sliding_window: usize,
        /// The checkpoint this sidecar belongs to: layers and hidden size, for
        /// `check_against_target`. Vocab is 248320 on both.
        target_layers_count: usize,
        target_hidden: usize,
    }

    fn sidecar_specs() -> Vec<SidecarSpec> {
        vec![
            SidecarSpec {
                model: crate::hub::Model::Qwen27B,
                n_layer: 5,
                n_embd: 5120,
                n_ff: 17408,
                target_layers: vec![2, 17, 32, 47, 62],
                tap_layers: vec![1, 16, 31, 46, 61],
                mask_token_id: 248070,
                sliding_window: 2048,
                target_layers_count: 64,
                target_hidden: 5120,
            },
            SidecarSpec {
                model: crate::hub::Model::Qwen35BA3B,
                n_layer: 6,
                n_embd: 2048,
                n_ff: 6144,
                target_layers: vec![2, 7, 12, 17, 23, 28, 33, 38],
                tap_layers: vec![1, 6, 11, 16, 22, 27, 32, 37],
                mask_token_id: 248077,
                sliding_window: 4096,
                target_layers_count: 40,
                target_hidden: 2048,
            },
        ]
    }

    /// A minimal in-memory `dflash` metadata map, with the window keys left to
    /// the caller so each parsing case can vary them.
    fn synth_metadata(n_layer: usize) -> HashMap<String, Value> {
        let mut m = HashMap::new();
        let mut put = |k: &str, v: Value| {
            m.insert(k.to_string(), v);
        };
        put("general.architecture", Value::String("dflash".into()));
        put("dflash.block_count", Value::U32(n_layer as u32));
        put("dflash.embedding_length", Value::U32(2048));
        put("dflash.feed_forward_length", Value::U32(6144));
        put("dflash.attention.head_count", Value::U32(32));
        put("dflash.attention.head_count_kv", Value::U32(8));
        put("dflash.attention.key_length", Value::U32(128));
        put("dflash.attention.layer_norm_rms_epsilon", Value::F32(1e-6));
        put("dflash.rope.freq_base", Value::F32(1e7));
        put("dflash.block_size", Value::U32(16));
        put("dflash.context_length", Value::U32(262144));
        put(
            "dflash.target_layers",
            Value::Array(vec![Value::I32(2), Value::I32(7)]),
        );
        put("tokenizer.ggml.mask_token_id", Value::U32(248077));
        m
    }

    fn synth_content(metadata: HashMap<String, Value>) -> Content {
        Content {
            magic: candle_core::quantized::gguf_file::VersionedMagic::GgufV3,
            metadata,
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
        }
    }

    /// Window parsing: absent keys mean full attention everywhere, a declared
    /// window makes the pattern mandatory, and a pattern that does not cover
    /// every layer is refused rather than silently padded.
    #[test]
    fn the_sliding_window_pattern_is_read_per_layer_and_checked() {
        // No window keys at all: every layer is full.
        let cfg = DflashConfig::from_gguf(&synth_content(synth_metadata(4))).unwrap();
        assert_eq!(cfg.sliding_window, None);
        assert_eq!(cfg.swa_layers, vec![false; 4]);
        assert_eq!(cfg.layer_window(0), None);

        // Declaring a window without the pattern is a hard error.
        let mut meta = synth_metadata(4);
        meta.insert(
            "dflash.attention.sliding_window".to_string(),
            Value::U32(2048),
        );
        let err = DflashConfig::from_gguf(&synth_content(meta.clone())).unwrap_err();
        assert!(err.to_string().contains("sliding_window_pattern"), "{err}");

        // A pattern shorter than the layer count names the mismatch.
        meta.insert(
            "dflash.attention.sliding_window_pattern".to_string(),
            Value::Array(vec![Value::Bool(true), Value::Bool(false)]),
        );
        let err = DflashConfig::from_gguf(&synth_content(meta.clone())).unwrap_err();
        let text = err.to_string();
        assert!(text.contains('2') && text.contains('4'), "{text}");

        // The shipped shape: windowed everywhere but the last layer.
        meta.insert(
            "dflash.attention.sliding_window_pattern".to_string(),
            Value::Array(vec![
                Value::Bool(true),
                Value::Bool(true),
                Value::Bool(true),
                Value::Bool(false),
            ]),
        );
        let cfg = DflashConfig::from_gguf(&synth_content(meta.clone())).unwrap();
        assert_eq!(cfg.sliding_window, Some(2048));
        assert_eq!(cfg.swa_layers, vec![true, true, true, false]);
        assert_eq!(cfg.layer_window(2), Some(2048));
        assert_eq!(cfg.layer_window(3), None);

        // A SCALAR pattern repeats across every layer, the second form
        // llama.cpp's `get_key_or_arr` takes (llama-model-loader.cpp:444-482).
        // Neither shipped sidecar uses it, but a uniformly-windowed drafter is
        // what a converter would emit it for, and llama.cpp would load that file.
        meta.insert(
            "dflash.attention.sliding_window_pattern".to_string(),
            Value::Bool(true),
        );
        let cfg = DflashConfig::from_gguf(&synth_content(meta.clone())).unwrap();
        assert_eq!(cfg.swa_layers, vec![true; 4]);

        meta.insert(
            "dflash.attention.sliding_window_pattern".to_string(),
            Value::Bool(false),
        );
        let cfg = DflashConfig::from_gguf(&synth_content(meta.clone())).unwrap();
        assert_eq!(cfg.swa_layers, vec![false; 4]);
        // The width is still declared, but no layer uses it.
        assert_eq!(cfg.sliding_window, Some(2048));
        assert_eq!(cfg.layer_window(0), None);

        // A declared width of 0 means "no window" whatever the pattern says, so
        // nothing downstream has to special-case a zero-width window.
        meta.insert("dflash.attention.sliding_window".to_string(), Value::U32(0));
        let cfg = DflashConfig::from_gguf(&synth_content(meta)).unwrap();
        assert_eq!(cfg.sliding_window, None);
        assert_eq!(cfg.swa_layers, vec![false; 4]);
    }

    /// The geometry of the checkpoint `sidecar_35b_config` belongs to.
    const T35_HIDDEN: usize = 2048;
    const T35_LAYERS: usize = 40;
    const T35_VOCAB: usize = 248320;

    /// The 35B-A3B sidecar's metadata, for the checks to vary one field of.
    fn sidecar_35b_config() -> DflashConfig {
        DflashConfig {
            n_layer: 6,
            n_embd: T35_HIDDEN,
            n_head: 32,
            n_head_kv: 8,
            head_dim: 128,
            n_ff: 6144,
            rms_eps: 1e-6,
            rope_theta: 10_000_000.0,
            sliding_window: Some(4096),
            swa_layers: vec![true, true, true, true, true, false],
            block_size: 16,
            target_layers: vec![2, 7, 12, 17, 23, 28, 33, 38],
            mask_token_id: 248077,
            context_length: 262144,
        }
    }

    /// Everything a drafter can be rejected for before a weight is read, in one
    /// place — the shared check both the CLI's `attach_drafter` and serve's
    /// startup preflight run.
    #[test]
    fn a_drafter_is_checked_against_the_target_it_will_serve() {
        let check = |cfg: &DflashConfig, hidden, layers, vocab| {
            cfg.check_against_target(hidden, layers, vocab)
        };
        let ok = |cfg: &DflashConfig| check(cfg, T35_HIDDEN, T35_LAYERS, T35_VOCAB);
        assert!(ok(&sidecar_35b_config()).is_ok());

        // --- internal consistency ------------------------------------------
        // Zero KV heads is the one that would otherwise panic: `build` computes
        // the head ratio with `%` before its `ensure!` can report anything.
        let err = ok(&DflashConfig {
            n_head_kv: 0,
            ..sidecar_35b_config()
        })
        .unwrap_err();
        assert!(err.to_string().contains("head_count_kv"), "{err}");

        let err = ok(&DflashConfig {
            n_head: 0,
            ..sidecar_35b_config()
        })
        .unwrap_err();
        assert!(err.to_string().contains("head_count"), "{err}");

        // Heads that do not divide into the KV heads leave no whole number of
        // query heads per KV head.
        let err = ok(&DflashConfig {
            n_head: 30,
            ..sidecar_35b_config()
        })
        .unwrap_err();
        let text = err.to_string();
        assert!(text.contains("30") && text.contains('8'), "{text}");

        // Rope splits the head dim in half, so an odd one has no pairing.
        assert!(
            ok(&DflashConfig {
                head_dim: 127,
                ..sidecar_35b_config()
            })
            .is_err()
        );

        // A noise block of one position is the anchor alone — it can never carry
        // a draft token, so such a drafter would run at a permanent loss.
        let err = ok(&DflashConfig {
            block_size: 1,
            ..sidecar_35b_config()
        })
        .unwrap_err();
        assert!(err.to_string().contains("anchor"), "{err}");
        assert!(
            ok(&DflashConfig {
                block_size: 2,
                ..sidecar_35b_config()
            })
            .is_ok()
        );

        // --- cross-checks against the target -------------------------------
        // The pairing nothing else catches: this drafter's taps top out at 37,
        // inside the 27B's 64 layers, so only the hidden size tells them apart.
        assert_eq!(
            sidecar_35b_config().spec_tap_layers().unwrap(),
            vec![1, 6, 11, 16, 22, 27, 32, 37]
        );
        let err = check(&sidecar_35b_config(), 5120, 64, T35_VOCAB).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("2048") && text.contains("5120"), "{text}");

        // A tap past the target's last layer, reported by the first offender.
        // `set_spec_taps` asserts the same bound, i.e. panics, so this check is
        // what keeps a configuration mistake from reaching it.
        let err = check(&sidecar_35b_config(), T35_HIDDEN, 20, T35_VOCAB).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("22") && text.contains("20"), "{text}");
        // A tap exactly at the layer count is one past the last index.
        assert!(check(&sidecar_35b_config(), T35_HIDDEN, 38, T35_VOCAB).is_ok());
        assert!(check(&sidecar_35b_config(), T35_HIDDEN, 37, T35_VOCAB).is_err());

        // The mask token is embedded through the target's table every round, so
        // an id past its vocabulary is an indexing failure at the first one.
        let err = check(&sidecar_35b_config(), T35_HIDDEN, T35_LAYERS, 248_077).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("248077"), "{text}");
        assert!(check(&sidecar_35b_config(), T35_HIDDEN, T35_LAYERS, 248_078).is_ok());
    }

    /// Test 1: load each real drafter GGUF and check the parsed config plus the
    /// on-disk tensor shapes/dtypes (BF16 matmuls, F32 norms). Reads the sidecar
    /// files only — no target model process, safe to run.
    #[test]
    fn real_file_load_and_shapes() {
        let dev = Device::Cpu;
        for spec in sidecar_specs() {
            let Some(path) = crate::hub::cached_drafter(spec.model) else {
                eprintln!(
                    "skipping real_file_load_and_shapes for {:?}: drafter not in the HF cache",
                    spec.model
                );
                continue;
            };
            let gguf = crate::gguf::open(&path, &dev).unwrap();

            let cfg = DflashConfig::from_gguf(&gguf.content).unwrap();
            assert_eq!(cfg.n_layer, spec.n_layer);
            assert_eq!(cfg.n_embd, spec.n_embd);
            assert_eq!(cfg.n_head, 32);
            assert_eq!(cfg.n_head_kv, 8);
            assert_eq!(cfg.head_dim, 128);
            assert_eq!(cfg.n_ff, spec.n_ff);
            assert_eq!(cfg.block_size, 16);
            assert_eq!(cfg.context_length, 262144);
            assert_eq!(cfg.target_layers, spec.target_layers);
            // Layer-input ids translate to post-FFN l_out capture indices (t - 1).
            assert_eq!(cfg.spec_tap_layers().unwrap(), spec.tap_layers);
            assert_eq!(cfg.mask_token_id, spec.mask_token_id);
            assert!((cfg.rms_eps - 1e-6).abs() < 1e-12);
            assert_eq!(cfg.rope_theta, 10_000_000.0);
            // Every layer is windowed but the last, which runs full attention.
            assert_eq!(cfg.sliding_window, Some(spec.sliding_window));
            let want_pattern: Vec<bool> =
                (0..spec.n_layer).map(|il| il + 1 < spec.n_layer).collect();
            assert_eq!(cfg.swa_layers, want_pattern);
            assert_eq!(cfg.layer_window(0), Some(spec.sliding_window));
            assert_eq!(cfg.layer_window(spec.n_layer - 1), None);

            // On-disk tensor shapes and dtypes.
            let info = |n: &str| {
                gguf.content
                    .tensor_infos
                    .get(n)
                    .unwrap_or_else(|| panic!("missing {n}"))
            };
            let shape = |n: &str| info(n).shape.dims().to_vec();
            let dtype = |n: &str| info(n).ggml_dtype;
            let (n_embd, n_ff) = (spec.n_embd, spec.n_ff);

            assert_eq!(
                shape("fc.weight"),
                vec![n_embd, spec.target_layers.len() * n_embd]
            );
            assert_eq!(dtype("fc.weight"), GgmlDType::BF16);
            assert_eq!(dtype("enc.output_norm.weight"), GgmlDType::F32);
            assert_eq!(dtype("output_norm.weight"), GgmlDType::F32);
            // The encoder is fc + one norm: no per-tap norm ships.
            assert!(
                !gguf
                    .content
                    .tensor_infos
                    .contains_key("enc.aux_norm.weight")
            );
            for il in 0..cfg.n_layer {
                let p = |n: &str| format!("blk.{il}.{n}.weight");
                assert_eq!(shape(&p("attn_q")), vec![4096, n_embd]);
                assert_eq!(shape(&p("attn_k")), vec![1024, n_embd]);
                assert_eq!(shape(&p("attn_v")), vec![1024, n_embd]);
                assert_eq!(shape(&p("attn_output")), vec![n_embd, 4096]);
                assert_eq!(shape(&p("attn_q_norm")), vec![128]);
                assert_eq!(shape(&p("attn_k_norm")), vec![128]);
                assert_eq!(shape(&p("ffn_gate")), vec![n_ff, n_embd]);
                assert_eq!(shape(&p("ffn_up")), vec![n_ff, n_embd]);
                assert_eq!(shape(&p("ffn_down")), vec![n_embd, n_ff]);
                assert_eq!(dtype(&p("attn_q")), GgmlDType::BF16);
                assert_eq!(dtype(&p("attn_q_norm")), GgmlDType::F32);
                assert_eq!(dtype(&p("attn_norm")), GgmlDType::F32);
                // No attention output gate on this arch.
                assert!(!gguf.content.tensor_infos.contains_key(&p("attn_gate")));
            }

            // The shipped pairing passes the check both attach paths run, and
            // the vocabulary is the padded 248320 of both checkpoints.
            cfg.check_against_target(spec.target_hidden, spec.target_layers_count, 248_320)
                .unwrap();

            // Full load materializes every weight dense f32 and builds the caches.
            let drafter = DflashDrafter::load(&gguf, &dev, 64).unwrap();
            assert_eq!(drafter.config().n_layer, spec.n_layer);
            assert_eq!(drafter.committed_len(), 0);
        }
    }

    /// Test 2: the encoder matches a hand-rolled scalar re-implementation.
    #[test]
    fn encoder_matches_scalar() {
        let dev = Device::Cpu;
        let (n_embd, seq) = (8usize, 3usize);
        let cfg = tiny_cfg(1, 2, 1, 4, n_embd, 16); // 2 target layers -> n_aux 2
        let n_aux = cfg.target_layers.len();
        let m = synth_weights(&cfg, &dev);

        let taps: Vec<Tensor> = (0..n_aux)
            .map(|a| {
                Tensor::from_vec(seeded(seq * n_embd, 100 + a as u64), (seq, n_embd), &dev).unwrap()
            })
            .collect();
        let drafter = drafter_from_map(cfg.clone(), &m, &dev, 32, WeightDtype::F32);
        let got = to_vec(&drafter.encode(&taps).unwrap());

        // Scalar reference: the taps enter raw, concatenated tap-major.
        let enc_out = host(&m, "enc.output_norm");
        let fc = host(&m, "fc");
        let taps_h: Vec<Vec<f64>> = taps
            .iter()
            .map(|t| to_vec(t).iter().map(|x| *x as f64).collect())
            .collect();
        let eps = cfg.rms_eps;
        let mut want = Vec::new();
        for s in 0..seq {
            let mut concat = vec![0.0f64; n_aux * n_embd];
            for a in 0..n_aux {
                for e in 0..n_embd {
                    concat[a * n_embd + e] = taps_h[a][s * n_embd + e];
                }
            }
            let fused = matvec(&fc, &concat, n_embd, n_aux * n_embd);
            let normed = rmsnorm(&fused, &enc_out, eps);
            want.extend(normed.iter().map(|x| *x as f32));
        }
        let rel = rel_l2(&got, &want);
        assert!(rel < 1e-5, "encoder rel_l2 {rel} too high");
    }

    /// Test 3: draft_forward on the f32 weight path matches the independent
    /// scalar reference — verifies cache-vs-block equivalence, GQA, rope
    /// positions, causal masking, gate. The f16 path is graded against the same
    /// reference (looser tolerance) in `draft_forward_f16_matches_scalar`.
    #[test]
    fn draft_forward_matches_scalar() {
        let dev = Device::Cpu;
        let (n_embd, hd) = (8usize, 4usize);
        let cfg = tiny_cfg(2, 2, 1, hd, n_embd, 16);
        let m = synth_weights(&cfg, &dev);
        let mut drafter = drafter_from_map(cfg.clone(), &m, &dev, 32, WeightDtype::F32);

        let committed = 4usize;
        let n_block = 3usize;
        // Fused context (post-encoder residuals) and the noise block, at residual
        // scale so the block stays well-conditioned.
        let scale_in = 0.25f32;
        let fused = Tensor::from_vec(seeded(committed * n_embd, 7), (committed, n_embd), &dev)
            .unwrap()
            .affine(scale_in as f64, 0.0)
            .unwrap();
        let noise = Tensor::from_vec(seeded(n_block * n_embd, 8), (n_block, n_embd), &dev)
            .unwrap()
            .affine(scale_in as f64, 0.0)
            .unwrap();

        drafter.inject(&fused, 0).unwrap();
        assert_eq!(drafter.committed_len(), committed);
        let got = to_vec(&drafter.draft_forward(&noise, committed).unwrap());
        // draft_forward must not advance the committed cache.
        assert_eq!(drafter.committed_len(), committed);

        let to_rows = |t: &Tensor, rows: usize| -> Vec<Vec<f64>> {
            let flat = to_vec(t);
            (0..rows)
                .map(|r| (0..n_embd).map(|c| flat[r * n_embd + c] as f64).collect())
                .collect()
        };
        let ctx = to_rows(&fused, committed);
        let nz = to_rows(&noise, n_block);
        let want_rows = naive_draft(&cfg, &m, &ctx, &nz);
        let want: Vec<f32> = want_rows.iter().flatten().map(|x| *x as f32).collect();

        let rel = rel_l2(&got, &want);
        assert!(rel < 1e-4, "draft_forward rel_l2 {rel} too high");
    }

    /// The windowed layers drop keys older than `window` positions while the
    /// final full layer keeps all of them, matching the scalar reference that
    /// filters key positions by `p1 - p0 < window`. The context here is deeper
    /// than the window, so the narrowed key view starts past position 0 and the
    /// per-query mask has work to do — the two things a full-attention drafter
    /// would get wrong.
    #[test]
    fn sliding_window_matches_scalar() {
        let dev = Device::Cpu;
        let (n_embd, hd) = (8usize, 4usize);
        let cfg = with_window(tiny_cfg(2, 2, 1, hd, n_embd, 16), 5);
        assert_eq!(cfg.layer_window(0), Some(5));
        assert_eq!(cfg.layer_window(1), None);
        let m = synth_weights(&cfg, &dev);
        let mut drafter = drafter_from_map(cfg.clone(), &m, &dev, 32, WeightDtype::F32);

        let committed = 12usize;
        let n_block = 3usize;
        let scale_in = 0.25f32;
        let fused = Tensor::from_vec(seeded(committed * n_embd, 31), (committed, n_embd), &dev)
            .unwrap()
            .affine(scale_in as f64, 0.0)
            .unwrap();
        let noise = Tensor::from_vec(seeded(n_block * n_embd, 32), (n_block, n_embd), &dev)
            .unwrap()
            .affine(scale_in as f64, 0.0)
            .unwrap();

        drafter.inject(&fused, 0).unwrap();
        let got = to_vec(&drafter.draft_forward(&noise, committed).unwrap());

        let to_rows = |t: &Tensor, rows: usize| -> Vec<Vec<f64>> {
            let flat = to_vec(t);
            (0..rows)
                .map(|r| (0..n_embd).map(|c| flat[r * n_embd + c] as f64).collect())
                .collect()
        };
        let want_rows = naive_draft(
            &cfg,
            &m,
            &to_rows(&fused, committed),
            &to_rows(&noise, n_block),
        );
        let want: Vec<f32> = want_rows.iter().flatten().map(|x| *x as f32).collect();
        let rel = rel_l2(&got, &want);
        assert!(rel < 1e-4, "windowed draft_forward rel_l2 {rel} too high");

        // And the window is not a no-op: the same weights with full attention
        // everywhere produce a different block.
        let full_cfg = tiny_cfg(2, 2, 1, hd, n_embd, 16);
        let mut full = drafter_from_map(full_cfg, &m, &dev, 32, WeightDtype::F32);
        full.inject(&fused, 0).unwrap();
        let full_out = to_vec(&full.draft_forward(&noise, committed).unwrap());
        assert!(
            rel_l2(&full_out, &want) > 1e-3,
            "full attention produced the windowed result — the window was ignored"
        );
    }

    /// The windowed path ON METAL, against the same scalar reference — the
    /// scalar test above runs on the CPU device and so never exercises what the
    /// window actually does to the tensors: `narrow` at a NONZERO offset into the
    /// cache planes, producing strided views that candle's Metal
    /// `broadcast_matmul` has to read (and contiguise for the GQA broadcast)
    /// rather than the offset-0 views every other path hands it.
    ///
    /// Three positions of the block relative to the window are covered, because
    /// they take different branches of `attention`/`window_mask`: a context
    /// shorter than the window (no narrow, no mask), one straddling it (offset 0
    /// but a live per-query mask), and one well past it (nonzero offset AND a
    /// mask). The f16 weight path is the shipped one, so it is what runs.
    #[test]
    fn windowed_attention_on_metal_matches_scalar() {
        let Ok(dev) = crate::gguf::metal_device() else {
            eprintln!("skipping windowed_attention_on_metal_matches_scalar: no Metal device");
            return;
        };
        // Shapes satisfy the f16 kernels' k % 32 == 0 / n_out % 4 == 0.
        let (n_embd, hd, n_head, n_kv, n_ff) = (64usize, 32usize, 4usize, 2usize, 128usize);
        let window = 12usize;
        let cfg = with_window(tiny_cfg(2, n_head, n_kv, hd, n_embd, n_ff), window);
        let m = synth_weights(&cfg, &dev);
        let n_block = 5usize;

        for committed in [6usize, 10, 30] {
            let mut drafter = drafter_from_map(cfg.clone(), &m, &dev, 64, WeightDtype::F16);
            let fused = Tensor::from_vec(
                seeded(committed * n_embd, 51 + committed as u64),
                (committed, n_embd),
                &dev,
            )
            .unwrap()
            .affine(0.25, 0.0)
            .unwrap();
            let noise = Tensor::from_vec(seeded(n_block * n_embd, 52), (n_block, n_embd), &dev)
                .unwrap()
                .affine(0.25, 0.0)
                .unwrap();

            // The three regimes this loop is here to cover.
            let lo = committed.saturating_sub(window - 1);
            let masked = committed + n_block > window;
            match committed {
                6 => assert!(lo == 0 && !masked, "expected the wholly-inside-window case"),
                10 => assert!(lo == 0 && masked, "expected the straddling case"),
                _ => assert!(lo > 0 && masked, "expected the offset-view case"),
            }

            drafter.inject(&fused, 0).unwrap();
            let got = to_vec(&drafter.draft_forward(&noise, committed).unwrap());
            assert!(
                got.iter().all(|v| v.is_finite()),
                "committed {committed}: non-finite output"
            );

            let to_rows = |t: &Tensor, rows: usize| -> Vec<Vec<f64>> {
                let flat = to_vec(t);
                (0..rows)
                    .map(|r| (0..n_embd).map(|c| flat[r * n_embd + c] as f64).collect())
                    .collect()
            };
            let want_rows = naive_draft(
                &cfg,
                &m,
                &to_rows(&fused, committed),
                &to_rows(&noise, n_block),
            );
            let want: Vec<f32> = want_rows.iter().flatten().map(|x| *x as f32).collect();
            // f16 weight rounding is the only departure from the full-precision
            // reference, bounded at the same per-matmul precision class as the
            // other f16 tests (docs/parity.md §3b).
            let rel = rel_l2(&got, &want);
            assert!(
                rel < 5e-4,
                "committed {committed}: windowed Metal rel_l2 {rel} too high"
            );
        }
    }

    /// The noise block is denoised in one non-causal forward, so a later block
    /// position informs an earlier one. Perturbing only the LAST noise row must
    /// therefore move the FIRST output row; under causal masking within the
    /// block it could not.
    #[test]
    fn the_noise_block_attends_to_itself_in_both_directions() {
        let dev = Device::Cpu;
        let (n_embd, hd) = (8usize, 4usize);
        let cfg = tiny_cfg(2, 2, 1, hd, n_embd, 16);
        let m = synth_weights(&cfg, &dev);

        let (committed, n_block) = (4usize, 3usize);
        let fused = Tensor::from_vec(seeded(committed * n_embd, 41), (committed, n_embd), &dev)
            .unwrap()
            .affine(0.25, 0.0)
            .unwrap();
        let mut noise = seeded(n_block * n_embd, 42)
            .iter()
            .map(|x| x * 0.25)
            .collect::<Vec<f32>>();

        let run = |noise: &[f32]| {
            let mut d = drafter_from_map(cfg.clone(), &m, &dev, 32, WeightDtype::F32);
            let t = Tensor::from_vec(noise.to_vec(), (n_block, n_embd), &dev).unwrap();
            d.inject(&fused, 0).unwrap();
            to_vec(&d.draft_forward(&t, committed).unwrap())
        };

        let base = run(&noise);
        for v in noise.iter_mut().skip((n_block - 1) * n_embd) {
            *v += 0.5;
        }
        let perturbed = run(&noise);

        let first_row_delta: f32 = base
            .iter()
            .zip(&perturbed)
            .take(n_embd)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            first_row_delta > 1e-3,
            "the first block row ignored a change to the last one ({first_row_delta}), so the \
             block was masked causally"
        );
    }

    /// Companion to Test 3 on the SHIPPED f16 weight path (Metal only): the same
    /// noise-block forward graded against the SAME full-precision scalar
    /// reference. The only difference from Test 3 is the f16 weight rounding, so
    /// the residual is larger but still small. Shapes are sized to satisfy the
    /// f16 kernel constraints (k % 32 == 0, n_out % 4 == 0) and to exercise BOTH
    /// the gemv branch (inject, t <= 8) and the tiled gemm branch (the t = 10
    /// noise block, the production default). Skips cleanly with no Metal device.
    #[test]
    fn draft_forward_f16_matches_scalar() {
        let Ok(dev) = crate::gguf::metal_device() else {
            eprintln!("skipping draft_forward_f16_matches_scalar: no Metal device");
            return;
        };
        let (n_embd, hd, n_head, n_kv, n_ff) = (64usize, 8usize, 8usize, 4usize, 128usize);
        let cfg = tiny_cfg(2, n_head, n_kv, hd, n_embd, n_ff);
        let m = synth_weights(&cfg, &dev);
        let mut drafter = drafter_from_map(cfg.clone(), &m, &dev, 32, WeightDtype::F16);

        let committed = 4usize;
        let n_block = 10usize; // > 8: exercises the tiled gemm (mm) branch
        let scale_in = 0.25f32;
        let fused = Tensor::from_vec(seeded(committed * n_embd, 7), (committed, n_embd), &dev)
            .unwrap()
            .affine(scale_in as f64, 0.0)
            .unwrap();
        let noise = Tensor::from_vec(seeded(n_block * n_embd, 8), (n_block, n_embd), &dev)
            .unwrap()
            .affine(scale_in as f64, 0.0)
            .unwrap();

        drafter.inject(&fused, 0).unwrap();
        let got = to_vec(&drafter.draft_forward(&noise, committed).unwrap());
        assert_eq!(drafter.committed_len(), committed);

        let to_rows = |t: &Tensor, rows: usize| -> Vec<Vec<f64>> {
            let flat = to_vec(t);
            (0..rows)
                .map(|r| (0..n_embd).map(|c| flat[r * n_embd + c] as f64).collect())
                .collect()
        };
        let ctx = to_rows(&fused, committed);
        let nz = to_rows(&noise, n_block);
        let want_rows = naive_draft(&cfg, &m, &ctx, &nz);
        let want: Vec<f32> = want_rows.iter().flatten().map(|x| *x as f32).collect();

        // f16 weight rounding (plus the tensor gemm's reduced-precision
        // cooperative-tensor math on the mm branch) is the only departure from
        // the full-precision reference, and it is heavily dampened at the output
        // by the exact-f32 residual stream: observed rel_l2 ~1.3e-5 (~10x the
        // f32 path's pure-accumulation floor, so f16 is genuinely exercised).
        // The bound sits at the mm branch's per-matmul precision class (5e-4,
        // docs/parity.md §3b) rather than hugging the dampened observation —
        // the dampening factor is a property of this synthetic net, not a
        // contract; gross kernel breakage still lands orders of magnitude out.
        let rel = rel_l2(&got, &want);
        assert!(rel < 5e-4, "draft_forward f16 rel_l2 {rel} too high");
    }

    /// The bf16 twin of `draft_forward_f16_matches_scalar`: the same
    /// full-precision scalar reference, weights narrowed to BF16
    /// (`WeightDtype::Bf16` — the synthetic constructor for the storage the
    /// mmap alias path serves) and multiplied by the vendored bf16 kernels.
    /// bf16 keeps f16's exponent range but only 8 mantissa bits, so the
    /// weight-rounding residual is ~8x the f16 twin's; bound one precision
    /// class up (4e-3) by the same not-hugging-the-observation reasoning.
    /// Skips cleanly with no Metal device.
    #[test]
    fn draft_forward_bf16_matches_scalar() {
        let Ok(dev) = crate::gguf::metal_device() else {
            eprintln!("skipping draft_forward_bf16_matches_scalar: no Metal device");
            return;
        };
        let (n_embd, hd, n_head, n_kv, n_ff) = (64usize, 8usize, 8usize, 4usize, 128usize);
        let cfg = tiny_cfg(2, n_head, n_kv, hd, n_embd, n_ff);
        let m = synth_weights(&cfg, &dev);
        let mut drafter = drafter_from_map(cfg.clone(), &m, &dev, 32, WeightDtype::Bf16);

        let committed = 4usize;
        let n_block = 10usize; // > 8: exercises the tiled gemm (mm) branch
        let scale_in = 0.25f32;
        let fused = Tensor::from_vec(seeded(committed * n_embd, 7), (committed, n_embd), &dev)
            .unwrap()
            .affine(scale_in as f64, 0.0)
            .unwrap();
        let noise = Tensor::from_vec(seeded(n_block * n_embd, 8), (n_block, n_embd), &dev)
            .unwrap()
            .affine(scale_in as f64, 0.0)
            .unwrap();

        drafter.inject(&fused, 0).unwrap();
        let got = to_vec(&drafter.draft_forward(&noise, committed).unwrap());
        assert_eq!(drafter.committed_len(), committed);

        let to_rows = |t: &Tensor, rows: usize| -> Vec<Vec<f64>> {
            let flat = to_vec(t);
            (0..rows)
                .map(|r| (0..n_embd).map(|c| flat[r * n_embd + c] as f64).collect())
                .collect()
        };
        let ctx = to_rows(&fused, committed);
        let nz = to_rows(&noise, n_block);
        let want_rows = naive_draft(&cfg, &m, &ctx, &nz);
        let want: Vec<f32> = want_rows.iter().flatten().map(|x| *x as f32).collect();

        let rel = rel_l2(&got, &want);
        assert!(rel < 4e-3, "draft_forward bf16 rel_l2 {rel} too high");
    }

    /// Loads each REAL drafter GGUF on Metal and checks the mmap-alias default:
    /// every matmul plane comes out a BF16 alias (no materialization), the
    /// mapping Arc is retained, and an inject + draft_forward round produces
    /// finite hidden states through the bf16 kernels at production shapes. The
    /// injected context runs past each sidecar's own window, so the windowed
    /// layers take the narrow-plus-mask path and a broken window shows up as a
    /// NaN row rather than only in the synthetic tests. Skips cleanly without
    /// the files or a Metal device.
    #[test]
    fn real_file_bf16_alias_load_and_forward() {
        let Ok(dev) = crate::gguf::metal_device() else {
            eprintln!("skipping real_file_bf16_alias_load_and_forward: no Metal device");
            return;
        };
        for spec in sidecar_specs() {
            let Some(path) = crate::hub::cached_drafter(spec.model) else {
                eprintln!(
                    "skipping real_file_bf16_alias_load_and_forward for {:?}: drafter not in the \
                     HF cache",
                    spec.model
                );
                continue;
            };
            let gguf = crate::gguf::open(&path, &dev).unwrap();
            let drafter_load = std::time::Instant::now();
            // Past this sidecar's own window, so query 0's floor is above 0 and
            // the later block queries need the per-query mask.
            let ctx = spec.sliding_window + 52;
            let mut drafter = DflashDrafter::load(&gguf, &dev, ctx + 16).unwrap();
            eprintln!(
                "{:?} drafter alias load: {:.3}s",
                spec.model,
                drafter_load.elapsed().as_secs_f64()
            );

            assert!(drafter._weights_mmap.is_some(), "expected the alias path");
            assert_eq!(drafter.fc.dtype(), DType::BF16);
            for l in &drafter.layers {
                for w in [
                    &l.wq,
                    &l.wk,
                    &l.wv,
                    &l.wo,
                    &l.ffn_gate,
                    &l.ffn_up,
                    &l.ffn_down,
                ] {
                    assert_eq!(w.dtype(), DType::BF16);
                }
            }

            let n_embd = drafter.cfg.n_embd;
            let fused = Tensor::from_vec(seeded(ctx * n_embd, 21), (ctx, n_embd), &dev)
                .unwrap()
                .affine(0.02, 0.0)
                .unwrap();
            let noise = Tensor::from_vec(seeded(16 * n_embd, 22), (16, n_embd), &dev)
                .unwrap()
                .affine(0.02, 0.0)
                .unwrap();
            drafter.inject(&fused, 0).unwrap();
            let out = to_vec(&drafter.draft_forward(&noise, ctx).unwrap());
            assert!(
                out.iter().all(|v| v.is_finite()),
                "draft_forward produced non-finite values on the alias path for {:?}",
                spec.model
            );
        }
    }

    /// Test 4: cache hygiene — inject/draft/truncate roundtrips leave the cache
    /// in a clean state. A drafter that injects 5, drafts, injects 2 more must
    /// produce identical draft logits to a fresh drafter given the SAME final
    /// injections, proving draft_forward never mutates the committed cache and
    /// truncate rolls back cleanly.
    #[test]
    fn truncate_inject_roundtrip() {
        let dev = Device::Cpu;
        let (n_embd, hd) = (8usize, 4usize);
        let cfg = tiny_cfg(2, 2, 1, hd, n_embd, 16);
        let m = synth_weights(&cfg, &dev);

        let mk = |n: usize, seed: u64| {
            Tensor::from_vec(seeded(n * n_embd, seed), (n, n_embd), &dev)
                .unwrap()
                .affine(0.25, 0.0)
                .unwrap()
        };
        let inj_a = mk(5, 1); // first injection (5 tokens)
        let inj_b = mk(2, 2); // second injection (2 tokens)
        let noise = mk(3, 3);

        // Path 1: inject 5, draft 3 (cache back to 5), truncate is implicit
        // (draft never advances), inject 2 more -> committed 7, then draft.
        let mut d1 = drafter_from_map(cfg.clone(), &m, &dev, 32, WeightDtype::F32);
        d1.inject(&inj_a, 0).unwrap();
        // Two draft_forwards at the same committed state must be bit-identical:
        // each round overwrites the scratch rows it reads, so their reuse never
        // leaks state from the prior draft.
        let da = to_vec(&d1.draft_forward(&noise, 5).unwrap());
        let db = to_vec(&d1.draft_forward(&noise, 5).unwrap());
        assert_eq!(
            da, db,
            "repeated draft_forward at the same committed state diverged"
        );
        assert_eq!(d1.committed_len(), 5);
        d1.inject(&inj_b, 5).unwrap();
        assert_eq!(d1.committed_len(), 7);
        let out1 = to_vec(&d1.draft_forward(&noise, 7).unwrap());

        // Path 2: a fresh drafter given the same two injections back to back.
        let mut d2 = drafter_from_map(cfg.clone(), &m, &dev, 32, WeightDtype::F32);
        d2.inject(&inj_a, 0).unwrap();
        d2.inject(&inj_b, 5).unwrap();
        let out2 = to_vec(&d2.draft_forward(&noise, 7).unwrap());

        assert_eq!(out1.len(), out2.len());
        for (i, (a, b)) in out1.iter().zip(&out2).enumerate() {
            assert!((a - b).abs() < 1e-6, "logit {i} differs: {a} vs {b}");
        }

        // Explicit truncate: roll back to 5 and re-inject 2 -> identical to d2.
        let mut d3 = drafter_from_map(cfg.clone(), &m, &dev, 32, WeightDtype::F32);
        d3.inject(&inj_a, 0).unwrap();
        d3.inject(&inj_b, 5).unwrap();
        d3.truncate(5).unwrap();
        assert_eq!(d3.committed_len(), 5);
        d3.inject(&inj_b, 5).unwrap();
        let out3 = to_vec(&d3.draft_forward(&noise, 7).unwrap());
        for (a, b) in out3.iter().zip(&out2) {
            assert!(
                (a - b).abs() < 1e-6,
                "post-truncate logit differs: {a} vs {b}"
            );
        }
    }

    /// Test 5: the drafter's committed rows are its whole state, so a host image
    /// of them plus an import reproduces a drafter exactly — the property that
    /// lets a conversation be parked out of the drafter's cache and resumed with
    /// speculation still live. Importing a PREFIX of an image must land the same
    /// rows a drafter that only ever injected that prefix holds, and must leave
    /// the drafter injectable from there.
    #[test]
    fn cache_image_round_trips_including_prefixes() {
        let dev = Device::Cpu;
        let (n_embd, hd) = (8usize, 4usize);
        let cfg = tiny_cfg(2, 2, 1, hd, n_embd, 16);
        let m = synth_weights(&cfg, &dev);

        let mk = |n: usize, seed: u64| {
            Tensor::from_vec(seeded(n * n_embd, seed), (n, n_embd), &dev)
                .unwrap()
                .affine(0.25, 0.0)
                .unwrap()
        };
        let inj_a = mk(5, 1);
        let inj_b = mk(2, 2);
        let noise = mk(3, 3);
        let build = || drafter_from_map(cfg.clone(), &m, &dev, 32, WeightDtype::F32);

        let mut src = build();
        src.inject(&inj_a, 0).unwrap();
        src.inject(&inj_b, 5).unwrap();
        let want_7 = to_vec(&src.draft_forward(&noise, 7).unwrap());

        let image = src.export_cache().unwrap();
        assert_eq!(image.pos, 7);
        assert_eq!(image.layers(), cfg.n_layer);
        assert_eq!(
            image.byte_len(),
            cfg.n_layer * 2 * cfg.n_head_kv * 7 * hd * size_of::<f32>()
        );
        // The image is a copy, not a view: the drafter keeps writing the scratch
        // rows behind it, and on Metal a contiguous() of an already-contiguous
        // view would have shared their storage.
        src.draft_forward(&noise, 7).unwrap();

        let mut restored = build();
        restored.import_cache(&image, 7).unwrap();
        assert_eq!(restored.committed_len(), 7);
        assert_eq!(to_vec(&restored.draft_forward(&noise, 7).unwrap()), want_7);

        // A prefix import: rows [0, 5) only.
        let mut only_a = build();
        only_a.inject(&inj_a, 0).unwrap();
        let want_5 = to_vec(&only_a.draft_forward(&noise, 5).unwrap());
        let mut short = build();
        short.import_cache(&image, 5).unwrap();
        assert_eq!(short.committed_len(), 5);
        assert_eq!(to_vec(&short.draft_forward(&noise, 5).unwrap()), want_5);
        // ...and injection continues from the imported length.
        short.inject(&inj_b, 5).unwrap();
        assert_eq!(to_vec(&short.draft_forward(&noise, 7).unwrap()), want_7);

        // An empty drafter images as zero positions and imports back as one.
        let empty = build().export_cache().unwrap();
        assert_eq!(empty.pos, 0);
        assert_eq!(empty.byte_len(), 0);
        let mut blank = build();
        blank.import_cache(&empty, 0).unwrap();
        assert_eq!(blank.committed_len(), 0);

        // Bounds and shape are hard errors, not silently short or transposed
        // restores.
        assert!(build().import_cache(&image, 8).is_err());
        let wide = tiny_cfg(2, 4, 2, hd, n_embd, 16);
        let wide_weights = synth_weights(&wide, &dev);
        let mut other = drafter_from_map(wide, &wide_weights, &dev, 32, WeightDtype::F32);
        assert!(other.import_cache(&image, 7).is_err());
        // A drafter whose cache is shorter than the image cannot hold it.
        let mut tiny_ctx = drafter_from_map(cfg.clone(), &m, &dev, 4, WeightDtype::F32);
        assert!(tiny_ctx.import_cache(&image, 7).is_err());
    }

    /// A drafter image survives the framed round trip byte-for-byte. The shape is
    /// deliberately asymmetric — heads, positions and head dim all differ — so a
    /// transposed or swapped field cannot pass on byte counts alone. A declared
    /// shape that wants more bytes than the planes hold is rejected by the same
    /// validating constructor the in-process export goes through, and an
    /// implausible plane length is refused by the record's byte budget before it
    /// sizes a buffer.
    #[test]
    fn drafter_image_survives_a_framed_round_trip() {
        let (pos, n_kv, hd, layers) = (7usize, 3usize, 5usize, 2usize);
        let plane = n_kv * pos * hd * size_of::<f32>();
        let pattern =
            |seed: u8| -> Vec<u8> { (0..plane).map(|i| (i as u8) ^ seed ^ 0xa5).collect() };
        let planes: Vec<(Vec<u8>, Vec<u8>)> = (0..layers)
            .map(|il| (pattern(il as u8), pattern(0x40 + il as u8)))
            .collect();
        let image = DrafterImage::new_dflash(pos, n_kv, hd, planes.clone()).unwrap();

        let mut bytes = Vec::new();
        image.write_to(&mut bytes).unwrap();
        assert_eq!(
            bytes.len(),
            image.serialized_len(),
            "the declared record length must be what the writer emits"
        );

        let back = DrafterImage::read_from(&bytes[..], bytes.len() as u64).unwrap();
        assert_eq!(back.kind(), DrafterImageKind::Dflash);
        assert_eq!(back.pos, pos);
        assert_eq!(back.n_kv_head, n_kv);
        assert_eq!(back.head_dim, hd);
        assert_eq!(back.layers(), layers);
        for (il, (k, v)) in planes.iter().enumerate() {
            let (bk, bv) = back.prefix(il, pos).unwrap();
            assert_eq!(&*bk, &k[..], "layer {il}: K");
            assert_eq!(&*bv, &v[..], "layer {il}: V");
        }

        assert!(
            DrafterImage::read_from(&bytes[..], bytes.len() as u64 - 8).is_err(),
            "a record framed shorter than its body must not read past the frame"
        );
        assert!(
            DrafterImage::read_from(&bytes[..], bytes.len() as u64 + 8).is_err(),
            "a record that leaves framed bytes unread disagrees with the directory"
        );

        // A self-consistent body whose declared shape wants more bytes than its
        // planes carry. Refused where the plane's declared length is compared
        // against the shape, which is before a buffer is sized from either.
        let mut lie = Vec::new();
        write_count(&mut lie, DrafterImageKind::Dflash.tag()).unwrap();
        write_count(&mut lie, pos).unwrap();
        write_count(&mut lie, n_kv).unwrap();
        write_count(&mut lie, hd).unwrap();
        write_count(&mut lie, 0usize).unwrap();
        write_count(&mut lie, 1usize).unwrap();
        write_plane(&mut lie, &pattern(1)[..plane - 4]).unwrap();
        write_plane(&mut lie, &pattern(2)[..plane - 4]).unwrap();
        let Err(err) = DrafterImage::read_from(&lie[..], lie.len() as u64) else {
            panic!("a shape lie must not parse");
        };
        assert!(
            format!("{err:#}").contains("its shape needs"),
            "a shape lie must be refused before it sizes anything: {err:#}"
        );

        // A shape whose f32 byte count would wrap a usize sizes nothing.
        assert!(
            DrafterImage::new_dflash(usize::MAX / 2, 3, 5, Vec::new()).is_err(),
            "an overflowing byte count must error rather than wrap"
        );
    }

    /// The MTP image is a different record wearing the same tag: half the
    /// element size, one layer, and a carry the DFlash image does not have. It
    /// has to survive its own round trip, and — the point of storing the kind at
    /// all — it must not be readable AS a DFlash image, because the two shapes
    /// can agree on byte counts by coincidence and would then differ only in what
    /// the numbers mean.
    #[test]
    fn an_mtp_image_round_trips_and_cannot_be_read_as_a_dflash_one() {
        let (pos, n_kv, hd, hidden) = (7usize, 4usize, 5usize, 6usize);
        let plane = n_kv * pos * hd * size_of::<half::f16>();
        let pattern =
            |seed: u8, n: usize| -> Vec<u8> { (0..n).map(|i| (i as u8) ^ seed ^ 0x5a).collect() };
        let k = pattern(1, plane);
        let v = pattern(2, plane);
        let carry = pattern(3, hidden * size_of::<f32>());
        let image = DrafterImage::new_mtp(
            pos,
            n_kv,
            hd,
            hidden,
            vec![(k.clone(), v.clone())],
            carry.clone(),
        )
        .unwrap();
        assert_eq!(
            image.byte_len(),
            2 * plane + carry.len(),
            "the carry is part of what the image costs to hold"
        );

        let mut bytes = Vec::new();
        image.write_to(&mut bytes).unwrap();
        assert_eq!(bytes.len(), image.serialized_len());
        let back = DrafterImage::read_from(&bytes[..], bytes.len() as u64).unwrap();
        assert_eq!(back.kind(), DrafterImageKind::Mtp);
        assert_eq!(back.pos, pos);
        assert_eq!(back.layers(), 1);
        assert_eq!(back.carry(), &carry[..]);
        let (bk, bv) = back.plane(0).unwrap();
        assert_eq!(bk, &k[..]);
        assert_eq!(bv, &v[..]);

        // The same planes declared as a DFlash image describe twice the bytes,
        // so the kind is not decoration: without it these records could only be
        // told apart by an arithmetic coincidence.
        assert!(
            DrafterImage::new_dflash(pos, n_kv, hd, vec![(k, v)]).is_err(),
            "f16 planes must not pass as f32 ones"
        );

        // And a DFlash image that reaches the MTP head is refused by name rather
        // than by whatever its bytes happen to mean.
        let dflash_plane = n_kv * pos * hd * size_of::<f32>();
        let dflash = DrafterImage::new_dflash(
            pos,
            n_kv,
            hd,
            vec![(pattern(4, dflash_plane), pattern(5, dflash_plane))],
        )
        .unwrap();
        assert_eq!(
            dflash.carry(),
            b"",
            "a DFlash image carries no hidden state"
        );
        assert_eq!(dflash.kind().name(), "DFlash");
        assert_eq!(image.kind().name(), "MTP");
    }
}
