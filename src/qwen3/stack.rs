//! The Qwen3 dense forward graph, driven through [`XwenModel`] like the
//! qwen4exp one is: the model owns the embedding, the per-layer KV caches, the
//! final norm and the lm head, and this module owns the per-layer weights and
//! the layer math. `XwenModel::run_stack` short-circuits here when the model
//! carries [`Qwen3Parts`], so every cache operation — reset, checkpoint and
//! rollback, snapshots, export and import, the disk tier — works unchanged,
//! and so does every surface built on `forward` / `forward_all_logits`.
//!
//! Ground truth is llama.cpp `src/models/qwen3.cpp` and HF `modeling_qwen3.py`.
//! Per layer, activations f32 `[t, hidden]`:
//!
//! ```text
//!   n  = rms_norm(x, input_layernorm)
//!   q  = q_proj(n) -> [t, n_head, 128] -> rms_norm(q_norm) -> rope
//!   k  = k_proj(n) -> [t, n_kv,   128] -> rms_norm(k_norm) -> rope
//!   v  = v_proj(n) -> [t, n_kv,   128]
//!   a  = softmax(q kᵀ / √128, causal, GQA) v      -> [t, n_head * 128]   (kqv_out)
//!   h  = x + o_proj(a)                                                    (ffn_inp)
//!   m  = rms_norm(h, post_attention_layernorm)                            (ffn_norm)
//!   x' = h + down_proj(silu(gate_proj(m)) * up_proj(m))                   (l_out)
//! ```
//!
//! The norm is applied BEFORE rope on both q and k (both sources), the rope is
//! NEoX over the whole 128-wide head, there is no bias anywhere and no output
//! gate — the `[q | gate]` split and the sigmoid gate of the Qwen 3.6 block do
//! not exist here, which is why this is its own stack rather than a
//! configuration of `AttnBlock`.
//!
//! Kernels, per step: every projection is `ops::matmul_bf16` over the loaded
//! BF16 planes as stored (`[n_out, k]`), the FFN activation is `ops::silu_mul`,
//! and attention is `ops::flash_attn` for a multi-token step (its first
//! production caller: causal mask and GQA in-kernel, K/V read straight out of
//! the f16 cache views by stride) and candle's f16 vector sdpa for a one-token
//! decode step — the split the shipped attention block would use at this head
//! size. `XWEN_QWEN3_ATTN=sdpa` swaps both for candle's f32 sdpa over widened
//! K/V and a materialized causal mask, which is the reference chain the flash
//! kernel's tests grade it against; it exists for bisecting, not for speed.
//! K and V land in the cache as f16 (rope stores K f16 directly); everything
//! else stays f32 until the encode output, which is the one place a bf16 cast
//! happens.

use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Module, Tensor};

use crate::kv_cache::{LayerCache, MaskKind};
use crate::model::{StackOutput, XwenModel};
use crate::rope::Rope;
use crate::stack_profile::Stage;

use super::{NormVariant, Qwen3Config, Qwen3LayerWeights};

/// The environment switch that picks the attention implementation at LOAD
/// time. Read once per `XwenModel::load`, not once per process, so a test can
/// load one model per arm inside one process and compare them.
pub const ATTN_ENV: &str = "XWEN_QWEN3_ATTN";

/// Which attention chain a loaded stack runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnImpl {
    /// The shipped path: `ops::flash_attn` for `t > 1`, candle's f16 vector
    /// sdpa over the f16 cache views for `t == 1`.
    Fused,
    /// candle's f32 sdpa over widened K/V with a materialized causal mask, for
    /// every `t`. Bit-for-bit the composed reference `flash_attn`'s own tests
    /// compare against, which is what makes it the bisect arm.
    Sdpa,
}

impl AttnImpl {
    /// Resolve from [`ATTN_ENV`]: unset means the shipped path, anything else
    /// must name an arm.
    pub fn from_env() -> Result<Self> {
        match std::env::var(ATTN_ENV) {
            Err(std::env::VarError::NotPresent) => Ok(Self::Fused),
            Err(std::env::VarError::NotUnicode(_)) => bail!("{ATTN_ENV} is not valid UTF-8"),
            Ok(value) => Self::parse(&value),
        }
    }

    /// `flash` / `fused` (or empty) select the shipped path, `sdpa` the
    /// reference chain; anything else is refused rather than defaulted, so a
    /// typo in a bisect run cannot silently measure the wrong arm.
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "flash" | "fused" => Ok(Self::Fused),
            "sdpa" => Ok(Self::Sdpa),
            other => bail!("{ATTN_ENV}={other:?}: expected `flash` (the default) or `sdpa`"),
        }
    }

    /// The name a dump records for provenance.
    pub fn label(self) -> &'static str {
        match self {
            Self::Fused => "flash",
            Self::Sdpa => "sdpa",
        }
    }
}

/// The per-layer weights of a loaded Qwen3 stack plus what every layer
/// shares: the rope table and the attention choice. The embedding, the final
/// norm, the lm head and the KV caches live on the [`XwenModel`] that holds
/// this, exactly as on the other architectures.
pub struct Qwen3Parts {
    cfg: Qwen3Config,
    /// One entry per layer, as the loader handed them over: BF16 projections
    /// `[n_out, k]`, F32 norm weights.
    layers: Vec<Qwen3LayerWeights>,
    /// One rope table over the full 128-wide head (`n_rot == head_dim`),
    /// shared by every layer.
    rope: Arc<Rope>,
    attn: AttnImpl,
    /// `1 / sqrt(head_dim)`.
    scale: f32,
    /// `rms_norm_eps`, narrowed once for the norm kernel.
    eps: f32,
}

impl Qwen3Parts {
    /// Assemble the stack over loaded weights, checking every plane against
    /// the kernel contracts ONCE here rather than on every forward: each
    /// projection must be a contiguous BF16 `[n_out, k]` plane with `k % 32 ==
    /// 0` and `n_out % 4 == 0` starting at a 16-byte-aligned offset
    /// (`ops::matmul_bf16`), and each norm an F32 vector of the right length
    /// (candle's Metal rms_norm needs the weight in the activations' dtype).
    /// `head_dim` is pinned to 128 by [`Qwen3Config`] already, which is what
    /// `ops::flash_attn` is compiled for.
    pub fn new(
        cfg: Qwen3Config,
        layers: Vec<Qwen3LayerWeights>,
        rope: Arc<Rope>,
        attn: AttnImpl,
    ) -> Result<Self> {
        // Matched exhaustively on purpose: a future variant that has to be
        // implemented (a zero-centred `(1 + w)` norm, say) must fail to compile
        // here rather than run the standard form over its weights.
        match cfg.norm {
            NormVariant::Standard => {}
        }
        ensure!(
            layers.len() == cfg.n_layer,
            "qwen3 stack: {} layers of weights for a {}-layer config",
            layers.len(),
            cfg.n_layer
        );
        ensure!(
            cfg.rope.rotary_dim == cfg.head_dim(),
            "qwen3 stack: rotary_dim {} must cover the whole {}-wide head (llama.cpp asserts \
             n_embd_head == n_rot for this architecture)",
            cfg.rope.rotary_dim,
            cfg.head_dim()
        );
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let inter = cfg.intermediate_size;
        let hd = cfg.head_dim();
        for (il, w) in layers.iter().enumerate() {
            let name = |plane: &str| format!("model.layers.{il}.{plane}");
            check_projection(&name("self_attn.q_proj"), &w.q_proj, q_dim, hidden)?;
            check_projection(&name("self_attn.k_proj"), &w.k_proj, kv_dim, hidden)?;
            check_projection(&name("self_attn.v_proj"), &w.v_proj, kv_dim, hidden)?;
            check_projection(&name("self_attn.o_proj"), &w.o_proj, hidden, q_dim)?;
            check_projection(&name("mlp.gate_proj"), &w.gate_proj, inter, hidden)?;
            check_projection(&name("mlp.up_proj"), &w.up_proj, inter, hidden)?;
            check_projection(&name("mlp.down_proj"), &w.down_proj, hidden, inter)?;
            check_norm(&name("input_layernorm"), &w.input_layernorm, hidden)?;
            check_norm(
                &name("post_attention_layernorm"),
                &w.post_attention_layernorm,
                hidden,
            )?;
            check_norm(&name("self_attn.q_norm"), &w.q_norm, hd)?;
            check_norm(&name("self_attn.k_norm"), &w.k_norm, hd)?;
        }
        Ok(Self {
            scale: 1.0 / (hd as f32).sqrt(),
            eps: cfg.rms_norm_eps as f32,
            cfg,
            layers,
            rope,
            attn,
        })
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.cfg
    }

    pub fn n_layer(&self) -> usize {
        self.layers.len()
    }

    /// Which attention chain this stack was loaded with, for dump provenance.
    pub fn attn_impl(&self) -> AttnImpl {
        self.attn
    }

    /// Bytes of layer weight resident on the device: the seven BF16
    /// projections and four F32 norms of every layer. The embedding is the
    /// model's and is added by the caller.
    pub fn weight_bytes(&self) -> u64 {
        self.layers
            .iter()
            .flat_map(|w| {
                [
                    &w.q_proj,
                    &w.k_proj,
                    &w.v_proj,
                    &w.o_proj,
                    &w.gate_proj,
                    &w.up_proj,
                    &w.down_proj,
                    &w.input_layernorm,
                    &w.post_attention_layernorm,
                    &w.q_norm,
                    &w.k_norm,
                ]
            })
            .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
            .sum()
    }
}

/// `ops::matmul_bf16`'s weight contract, asked of one plane.
fn check_projection(name: &str, t: &Tensor, n_out: usize, k: usize) -> Result<()> {
    let (rows, cols) = t.dims2().with_context(|| {
        format!(
            "{name}: expected a rank-2 [n_out, k] plane, got {:?}",
            t.dims()
        )
    })?;
    ensure!(
        (rows, cols) == (n_out, k),
        "{name}: shape [{rows}, {cols}], expected [{n_out}, {k}]"
    );
    ensure!(
        t.dtype() == DType::BF16,
        "{name}: dtype {:?}, the qwen3 projections run through the bf16 kernels",
        t.dtype()
    );
    ensure!(t.is_contiguous(), "{name}: not contiguous");
    ensure!(
        k.is_multiple_of(32) && n_out.is_multiple_of(4),
        "{name}: [{n_out}, {k}] needs k % 32 == 0 and n_out % 4 == 0 for matmul_bf16"
    );
    let (_guard, layout) = t.storage_and_layout();
    let byte_offset = layout.start_offset() * t.dtype().size_in_bytes();
    ensure!(
        byte_offset.is_multiple_of(16),
        "{name}: view starts at byte offset {byte_offset}, matmul_bf16 needs 16-byte alignment"
    );
    Ok(())
}

/// The norm-weight contract: an F32 vector of `len`, which is what candle's
/// Metal rms_norm accepts against f32 activations.
fn check_norm(name: &str, t: &Tensor, len: usize) -> Result<()> {
    let n = t
        .dims1()
        .with_context(|| format!("{name}: expected a rank-1 norm weight, got {:?}", t.dims()))?;
    ensure!(n == len, "{name}: length {n}, expected {len}");
    ensure!(
        t.dtype() == DType::F32,
        "{name}: dtype {:?}, norm weights must be widened to f32 at load",
        t.dtype()
    );
    ensure!(t.is_contiguous(), "{name}: not contiguous");
    Ok(())
}

/// How many layers a forward runs and whether the final norm applies, from
/// the HF `hidden_states` index the caller asked for: `None` is a full forward
/// (every layer, then the norm — what `forward` and `forward_all_logits`
/// want); `Some(n)` runs layers `[0, n)` and applies the norm only when
/// `n == n_layer`, which mirrors transformers' `tie_last_hidden_states`, so
/// `Some(0)` is the embedding output and `Some(n_layer)` equals `None`.
pub(crate) fn plan(stop_after: Option<usize>, n_layer: usize) -> Result<(usize, bool)> {
    match stop_after {
        None => Ok((n_layer, true)),
        Some(n) => {
            ensure!(
                n <= n_layer,
                "hidden-state index {n} is past the last one: this model has {n_layer} layers, \
                 so its indices run 0 (the embeddings) through {n_layer} (after the final norm)"
            );
            Ok((n, n == n_layer))
        }
    }
}

/// Run the Qwen3 stack over `tokens` at absolute position `pos`, advancing
/// every layer's KV cache. Returns the same triple the trunk's `run_stack`
/// does: the hidden states for every position (post-final-norm on a full
/// forward, the raw residual when `stop_after` stops short of the norm), the
/// named parity taps when capture is on, and the spec taps.
///
/// Tap names match the trunk's (`attn_norm`, `attn_o_proj`, `ffn_inp`,
/// `ffn_norm`, `ffn_out`, `l_out`, each `-{il}`, plus `h_nextn` before the
/// final norm), all identity-mapped to llama.cpp `qwen3.cpp`'s `cb()` names,
/// plus one this architecture has and the trunk does not: `kqv_out-{il}`, the
/// `[t, n_head * 128]` attention output BEFORE `o_proj`, which llama.cpp names
/// the same way. `result_norm` and `result_output` are the caller's
/// (`forward`), as on the trunk.
pub fn run_stack(
    model: &mut XwenModel,
    tokens: &Tensor,
    pos: usize,
    stop_after: Option<usize>,
) -> Result<StackOutput> {
    let seq = tokens.elem_count();
    ensure!(seq > 0, "qwen3 stack: a forward needs at least one token");
    ensure!(
        pos + seq <= model.max_ctx,
        "context overflow: position {pos} + {seq} tokens exceeds max_ctx {} \
         (raise --max-ctx or shorten the prompt)",
        model.max_ctx
    );
    let n_layer = model
        .qwen3
        .as_ref()
        .context("qwen3 stack on a model with no qwen3 parts")?
        .n_layer();
    let (n_run, apply_norm) = plan(stop_after, n_layer)?;
    model.grow_kv_capacity(pos + seq)?;

    // The profiler hooks the trunk uses, spelled against `model`. Off — the
    // normal case — each is one `Option` check.
    macro_rules! stage {
        ($stage:expr, $e:expr) => {{
            crate::stack_profile::stage_begin(&mut model.profile, &model.device)?;
            let out = $e;
            crate::stack_profile::stage_end(&mut model.profile, &model.device, $stage)?;
            out
        }};
    }
    crate::stack_profile::chunk_begin(&mut model.profile, &model.device, seq)?;

    let mut x = stage!(Stage::Embed, model.embed_tokens(tokens)?); // [seq, hidden] f32

    let mut taps: Vec<(String, Tensor)> = Vec::new();
    let spec_layers = model.spec_tap_layers.clone();
    let mut spec_captured: Vec<(usize, Tensor)> = Vec::new();
    macro_rules! tap {
        ($name:expr, $il:expr, $t:expr) => {
            if model.tap_enabled {
                taps.push((format!("{}-{}", $name, $il), $t.clone()));
            }
        };
    }

    // The fused permute/cast glue is Metal-only and shares the attention
    // block's kill-switch; every path below is bit-identical to the candle
    // chain it replaces (ops::attn_glue's tests), so the choice is invisible.
    let fused_glue = matches!(model.device, Device::Metal(_)) && !crate::ops::attn_glue_classic();

    for il in 0..n_run {
        // Disjoint field borrows: the parts are read, the caches are written,
        // and the profiler is a third field.
        let parts = model
            .qwen3
            .as_ref()
            .context("qwen3 stack on a model with no qwen3 parts")?;
        let cache = &mut model.caches[il];
        let w = &parts.layers[il];

        let normed = stage!(
            Stage::AttnNorm,
            rms_norm(&x, &w.input_layernorm, parts.eps)?
        );
        tap!("attn_norm", il, normed);

        let (attn, kqv_out) = stage!(
            Stage::MixerFullAttn,
            attention(parts, w, &normed, cache, pos, fused_glue, model.tap_enabled)?
        );
        if let Some(kqv) = kqv_out {
            tap!("kqv_out", il, kqv);
        }
        tap!("attn_o_proj", il, attn);
        let ffn_inp = stage!(Stage::ResidualAttn, (&x + &attn)?);
        tap!("ffn_inp", il, ffn_inp);

        let ffn_normed = stage!(
            Stage::FfnNorm,
            rms_norm(&ffn_inp, &w.post_attention_layernorm, parts.eps)?
        );
        tap!("ffn_norm", il, ffn_normed);

        let ffn_out = stage!(Stage::Ffn, ffn(w, &ffn_normed)?);
        tap!("ffn_out", il, ffn_out);

        x = stage!(Stage::ResidualFfn, (&ffn_inp + &ffn_out)?);
        tap!("l_out", il, x);

        if let Some(layers) = &spec_layers {
            if layers.contains(&il) {
                spec_captured.push((il, x.clone()));
            }
        }
    }

    if !apply_norm {
        // A hidden-state read below the norm: the raw residual after
        // `layers[n_run - 1]` (or the embeddings at `n_run == 0`). Nothing
        // downstream of the norm runs, and no tap names it.
        crate::stack_profile::chunk_end(&mut model.profile, &model.device)?;
        return Ok((x, taps, spec_captured));
    }

    // Pre-final-norm residual stream, named as on the trunk.
    if model.tap_enabled {
        taps.push(("h_nextn".to_string(), x.clone()));
    }
    let normed = stage!(Stage::FinalNorm, model.final_norm(&x)?);
    if model.keep_post_norm {
        model.post_norm_hidden = Some(normed.clone());
    }
    Ok((normed, taps, spec_captured))
}

/// `x / sqrt(mean(x²) + eps) * w` over the last dim — the standard form, no
/// `(1 + w)`. `w` must be f32 like `x` (candle's Metal kernel checks it).
fn rms_norm(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    Ok(candle_nn::ops::rms_norm(x, w, eps)?)
}

/// One layer's attention half over the normed input `[t, hidden]`: the three
/// projections, per-head QK-norm, rope, the cache append, the attention
/// itself and `o_proj`. Returns `(o_proj output [t, hidden], kqv_out)`, the
/// second being the pre-`o_proj` `[t, n_head * 128]` tensor when `want_kqv`
/// asks for it (tap capture) and `None` otherwise, so the untapped path holds
/// no extra handle.
fn attention(
    parts: &Qwen3Parts,
    w: &Qwen3LayerWeights,
    normed: &Tensor,
    cache: &mut LayerCache,
    pos: usize,
    fused_glue: bool,
    want_kqv: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    let cfg = &parts.cfg;
    let t = normed.dim(0)?;
    let (n_head, n_kv, hd) = (cfg.n_head, cfg.n_kv_head, cfg.head_dim());
    let q_dim = cfg.q_dim();

    // Projections, then per-head RMSNorm in `[t, head, dim]` layout where the
    // head dim is contiguous last. Norm BEFORE rope, as both sources order it.
    let q = crate::ops::matmul_bf16(&w.q_proj, normed)?.reshape((t, n_head, hd))?;
    let q = rms_norm(&q, &w.q_norm, parts.eps)?;
    let k = crate::ops::matmul_bf16(&w.k_proj, normed)?.reshape((t, n_kv, hd))?;
    let k = rms_norm(&k, &w.k_norm, parts.eps)?;
    let v = crate::ops::matmul_bf16(&w.v_proj, normed)?.reshape((t, n_kv, hd))?;

    // To `[head, t, dim]` for rope and attention. At t == 1 the two layouts
    // share byte order, so a reshape is the whole permutation; a multi-token
    // step is a real permute, fused with V's f16 cast on the Metal glue path.
    let (q, k, v) = if t == 1 {
        (
            q.reshape((n_head, 1, hd))?,
            k.reshape((n_kv, 1, hd))?,
            v.reshape((n_kv, 1, hd))?,
        )
    } else if fused_glue {
        (
            crate::ops::permute_01(&q)?,
            crate::ops::permute_01(&k)?,
            crate::ops::permute_01_f16(&v)?, // [n_kv, t, hd] f16, cache-ready
        )
    } else {
        (
            q.transpose(0, 1)?.contiguous()?,
            k.transpose(0, 1)?.contiguous()?,
            v.transpose(0, 1)?.contiguous()?,
        )
    };

    // Rope over the whole head. K is stored f16 straight into the cache. Q
    // stays f32 wherever its consumer wants f32 — the flash kernel and the f32
    // sdpa arm — and is stored f16 only where the f16 decode sdpa consumes it.
    let q_dt = if t == 1 && parts.attn == AttnImpl::Fused {
        DType::F16
    } else {
        DType::F32
    };
    let q = parts.rope.rotate(&q, pos, q_dt)?;
    let k16 = parts.rope.rotate(&k, pos, DType::F16)?;
    let v16 = if v.dtype() == DType::F16 {
        v
    } else if fused_glue {
        crate::ops::cast_f16(&v)?
    } else {
        v.to_dtype(DType::F16)?
    };
    let (k_all, v_all) = cache.append(&k16, &v16)?; // [n_kv, K, hd] f16, head-strided

    let out = match (parts.attn, t) {
        // Decode: candle's f16 vector sdpa over the cache views (the shipped
        // attention block's decode path). The kernel ignores any mask, and at
        // one query row none is needed.
        (AttnImpl::Fused, 1) => {
            let q = q.unsqueeze(0)?; // [1, n_head, 1, hd] f16
            let k = k_all.unsqueeze(0)?;
            let v = v_all.unsqueeze(0)?;
            let out = candle_nn::ops::sdpa(&q, &k, &v, None, false, parts.scale, 1.0)?;
            let out = out.squeeze(0)?; // [n_head, 1, hd] f16
            if fused_glue {
                crate::ops::cast_f32(&out)?
            } else {
                out.to_dtype(DType::F32)?
            }
        }
        // Prefill and any multi-token step: the vendored flash kernel. Every
        // layer here is full attention, so the key range starts at absolute 0
        // (`k_off` 0) and there is no window.
        (AttnImpl::Fused, _) => {
            crate::ops::flash_attn(&q, &k_all, &v_all, pos, 0, None, parts.scale)?
        }
        (AttnImpl::Sdpa, _) => sdpa_f32(&q, &k_all, &v_all, pos, n_head, parts.scale)?,
    }; // [n_head, t, hd] f32

    // Back to token-major `[t, n_head * hd]` for o_proj.
    let kqv = if t == 1 {
        out.reshape((1, q_dim))?
    } else if fused_glue {
        crate::ops::permute_01(&out)?.reshape((t, q_dim))?
    } else {
        out.transpose(0, 1)?.contiguous()?.reshape((t, q_dim))?
    };
    let o = crate::ops::matmul_bf16(&w.o_proj, &kqv)?; // [t, hidden]
    Ok((o, want_kqv.then_some(kqv)))
}

/// The bisect arm: candle's f32 sdpa with the cache's f16 K/V widened (exact)
/// and the causal mask materialized as `kv_cache::attn_mask_for` builds it —
/// the composed reference `ops::flash_attn`'s tests grade the kernel against.
/// `q` is `[n_head, t, hd]` f32; returns the same shape.
fn sdpa_f32(
    q: &Tensor,
    k_all: &Tensor,
    v_all: &Tensor,
    pos: usize,
    n_head: usize,
    scale: f32,
) -> Result<Tensor> {
    let t = q.dim(1)?;
    let k_len = k_all.dim(1)?;
    let q = q.unsqueeze(0)?; // [1, n_head, t, hd]
    let k = k_all.to_dtype(DType::F32)?.unsqueeze(0)?;
    let v = v_all.to_dtype(DType::F32)?.unsqueeze(0)?;
    // `None` at t == 1: a single query row sees every cached key.
    let mask = crate::kv_cache::attn_mask_for(MaskKind::Full, t, pos, q.device())?
        .map(|raw| -> Result<Tensor> {
            ensure!(
                raw.dims() == [t, k_len],
                "qwen3 sdpa: mask {:?} for {t} queries over {k_len} keys",
                raw.dims()
            );
            Ok(raw
                .reshape((1, 1, t, k_len))?
                .broadcast_as((1, n_head, t, k_len))?
                .contiguous()?)
        })
        .transpose()?;
    let out = candle_nn::ops::sdpa(&q, &k, &v, mask.as_ref(), false, scale, 1.0)?;
    Ok(out.squeeze(0)?)
}

/// The SwiGLU FFN over the normed input `[t, hidden]`:
/// `down_proj(silu(gate_proj(m)) * up_proj(m))`.
fn ffn(w: &Qwen3LayerWeights, m: &Tensor) -> Result<Tensor> {
    let gate = crate::ops::matmul_bf16(&w.gate_proj, m)?; // [t, inter]
    let up = crate::ops::matmul_bf16(&w.up_proj, m)?;
    let act = if matches!(m.device(), Device::Metal(_)) && !crate::ops::act_classic() {
        crate::ops::silu_mul(&gate, &up)?
    } else {
        (candle_nn::ops::silu(&gate)? * up)?
    };
    Ok(crate::ops::matmul_bf16(&w.down_proj, &act)?) // [t, hidden]
}

/// The final `model.norm` the model applies after its last layer; here so the
/// stack and [`XwenModel::final_norm`] share one definition.
pub(crate) fn final_rms_norm(x: &Tensor, norm: &candle_nn::RmsNorm) -> Result<Tensor> {
    Ok(norm.forward(x)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::qwen3::RopeSpec;

    /// The HF `hidden_states` index semantics, without a device: 0 is the
    /// embeddings (no layer, no norm), `n_layer` is a full forward (every
    /// layer and the norm), anything between is a raw residual, and past the
    /// end is refused.
    #[test]
    fn the_hidden_state_index_selects_layers_and_the_norm() {
        assert_eq!(plan(None, 36).unwrap(), (36, true));
        assert_eq!(plan(Some(36), 36).unwrap(), (36, true));
        assert_eq!(plan(Some(35), 36).unwrap(), (35, false));
        assert_eq!(plan(Some(1), 36).unwrap(), (1, false));
        assert_eq!(plan(Some(0), 36).unwrap(), (0, false));
        let err = plan(Some(37), 36).unwrap_err().to_string();
        assert!(err.contains("37") && err.contains("36"), "{err}");
    }

    /// The load-time switch names its two arms and refuses anything else.
    #[test]
    fn the_attention_switch_parses_its_two_arms() {
        assert_eq!(AttnImpl::parse("").unwrap(), AttnImpl::Fused);
        assert_eq!(AttnImpl::parse("flash").unwrap(), AttnImpl::Fused);
        assert_eq!(AttnImpl::parse(" Fused ").unwrap(), AttnImpl::Fused);
        assert_eq!(AttnImpl::parse("sdpa").unwrap(), AttnImpl::Sdpa);
        assert_eq!(AttnImpl::parse("SDPA").unwrap(), AttnImpl::Sdpa);
        assert!(AttnImpl::parse("flash-attn").is_err());
        assert_eq!(AttnImpl::Fused.label(), "flash");
        assert_eq!(AttnImpl::Sdpa.label(), "sdpa");
    }

    /// A tiny config at the real head size (128 is what the flash kernel and
    /// the loader both pin) with every projection inside the kernels' shape
    /// rules: hidden 64, q 256, kv 128, intermediate 96, vocab 32.
    fn tiny_config(n_layer: usize) -> Qwen3Config {
        Qwen3Config {
            hidden_size: 64,
            intermediate_size: 96,
            n_layer,
            n_head: 2,
            n_kv_head: 1,
            rms_norm_eps: 1e-6,
            vocab_size: 32,
            max_position_embeddings: 4096,
            tie_word_embeddings: true,
            norm: NormVariant::Standard,
            rope: RopeSpec {
                head_dim: 128,
                rotary_dim: 128,
                theta: 1e6,
            },
            eog: crate::qwen3::QWEN3_EOG,
        }
    }

    /// Deterministic pseudo-random f32s in [lo, hi] (xorshift, no deps).
    fn rand(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
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

    fn bf16_plane(seed: u64, n_out: usize, k: usize, dev: &Device) -> Tensor {
        // Small magnitudes, so a 36-deep stack of random planes stays finite;
        // the tiny stack is two layers, but the scale keeps it honest anyway.
        let scale = 1.0 / (k as f32).sqrt();
        Tensor::from_vec(
            rand(seed, n_out * k, -scale, scale),
            (n_out, k),
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(dev)
        .unwrap()
    }

    fn norm_vec(seed: u64, n: usize, dev: &Device) -> Tensor {
        Tensor::from_vec(rand(seed, n, 0.8, 1.2), n, &Device::Cpu)
            .unwrap()
            .to_device(dev)
            .unwrap()
    }

    /// Random weights at the tiny geometry, in the loader's own shapes.
    fn tiny_layers(cfg: &Qwen3Config, seed: u64, dev: &Device) -> Vec<Qwen3LayerWeights> {
        let (h, q, kv, ff, hd) = (
            cfg.hidden_size,
            cfg.q_dim(),
            cfg.kv_dim(),
            cfg.intermediate_size,
            cfg.head_dim(),
        );
        (0..cfg.n_layer)
            .map(|il| {
                let s = seed + 100 * il as u64;
                Qwen3LayerWeights {
                    input_layernorm: norm_vec(s + 1, h, dev),
                    q_proj: bf16_plane(s + 2, q, h, dev),
                    k_proj: bf16_plane(s + 3, kv, h, dev),
                    v_proj: bf16_plane(s + 4, kv, h, dev),
                    o_proj: bf16_plane(s + 5, h, q, dev),
                    q_norm: norm_vec(s + 6, hd, dev),
                    k_norm: norm_vec(s + 7, hd, dev),
                    post_attention_layernorm: norm_vec(s + 8, h, dev),
                    gate_proj: bf16_plane(s + 9, ff, h, dev),
                    up_proj: bf16_plane(s + 10, ff, h, dev),
                    down_proj: bf16_plane(s + 11, h, ff, dev),
                }
            })
            .collect()
    }

    /// A whole tiny model on random weights, assembled through the same
    /// constructor the real load uses, with the attention arm chosen.
    fn tiny_model(attn: AttnImpl, dev: &Device) -> XwenModel {
        let cfg = tiny_config(2);
        let rope = Arc::new(
            Rope::new(
                &crate::config::RopeKind::Plain {
                    freq_base: cfg.rope.theta as f32,
                    n_rot: cfg.rope.rotary_dim,
                },
                256,
                dev,
            )
            .unwrap(),
        );
        let layers = tiny_layers(&cfg, 7, dev);
        let parts = Qwen3Parts::new(cfg.clone(), layers, rope, attn).unwrap();
        let embed = bf16_plane(999, cfg.vocab_size, cfg.hidden_size, dev);
        let norm = norm_vec(998, cfg.hidden_size, dev);
        XwenModel::assemble_qwen3(
            parts,
            embed,
            norm,
            crate::gguf::CheckpointId::from_parts(0xdead, 1),
            256,
            dev.clone(),
        )
        .unwrap()
    }

    fn device_or_skip(what: &str) -> Option<Device> {
        match metal_device() {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("skipping {what}: no Metal device ({e})");
                None
            }
        }
    }

    fn rows(t: &Tensor) -> Vec<Vec<f32>> {
        t.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap()
    }

    fn max_abs_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .flat_map(|(x, y)| {
                assert_eq!(x.len(), y.len());
                x.iter().zip(y).map(|(p, q)| (p - q).abs())
            })
            .fold(0.0, f32::max)
    }

    /// The shapes every entry point returns at the tiny geometry: all-position
    /// logits `[t, vocab]`, last-position logits `[vocab]`, and the encode
    /// output `[t, hidden]` bf16 at every hidden-state index.
    #[test]
    fn the_stack_returns_the_trunk_shapes() {
        let Some(dev) = device_or_skip("the_stack_returns_the_trunk_shapes") else {
            return;
        };
        let mut model = tiny_model(AttnImpl::Fused, &dev);
        let ids: Vec<u32> = (0..9).map(|i| (i * 7 + 3) % 32).collect();
        let tokens = Tensor::new(ids.as_slice(), &dev).unwrap();
        let all = model.forward_all_logits(&tokens, 0).unwrap();
        assert_eq!(all.dims(), &[9, 32]);
        assert_eq!(all.dtype(), DType::F32);
        assert_eq!(model.cache_len(), 9);
        let next = Tensor::new(&[5u32], &dev).unwrap();
        let last = model.forward(&next, 9).unwrap();
        assert_eq!(last.dims(), &[32]);
        assert_eq!(model.cache_len(), 10);

        for n in 0..=2 {
            let (hidden, t) = model.encode(&ids, n).unwrap();
            assert_eq!(
                (hidden.dims(), hidden.dtype(), t),
                (&[9, 64][..], DType::BF16, 9)
            );
            // encode leaves the cache empty behind it.
            assert_eq!(model.cache_len(), 0);
        }
        assert!(model.encode(&ids, 3).is_err());
        assert!(model.encode(&[], 1).is_err());
    }

    /// Feeding the same ids as one prefill, then token by token, then in
    /// uneven chunks, gives the same logits at every position: the flash
    /// kernel (multi-token) and the f16 vector sdpa (one token) agree over the
    /// same f16 cache, and the gemv / gemm split of `matmul_bf16` (t <= 8 vs
    /// above) does not move the result past the parity bar.
    #[test]
    fn chunked_prefill_matches_single_pass() {
        let Some(dev) = device_or_skip("chunked_prefill_matches_single_pass") else {
            return;
        };
        let mut model = tiny_model(AttnImpl::Fused, &dev);
        let ids: Vec<u32> = (0..23).map(|i| (i * 11 + 5) % 32).collect();
        let tokens = Tensor::new(ids.as_slice(), &dev).unwrap();
        let single = rows(&model.forward_all_logits(&tokens, 0).unwrap());

        for chunk in [1usize, 7, 8, 9, 16] {
            model.reset_cache().unwrap();
            let mut got: Vec<Vec<f32>> = Vec::new();
            let mut pos = 0;
            for c in ids.chunks(chunk) {
                let t = Tensor::new(c, &dev).unwrap();
                got.extend(rows(&model.forward_all_logits(&t, pos).unwrap()));
                pos += c.len();
            }
            let diff = max_abs_diff(&single, &got);
            eprintln!("chunk {chunk}: max |Δlogit| {diff:.3e}");
            assert!(diff <= 2e-2, "chunk {chunk}: max abs diff {diff}");
            for (p, (a, b)) in single.iter().zip(&got).enumerate() {
                let am = argmax(a);
                let bm = argmax(b);
                assert_eq!(am, bm, "chunk {chunk}: argmax differs at position {p}");
            }
        }
    }

    /// The sdpa bisect arm and the shipped fused arm agree on the same
    /// weights and ids, as one prefill and as one-token steps.
    #[test]
    fn the_sdpa_arm_matches_the_fused_arm() {
        let Some(dev) = device_or_skip("the_sdpa_arm_matches_the_fused_arm") else {
            return;
        };
        let ids: Vec<u32> = (0..19).map(|i| (i * 5 + 1) % 32).collect();
        let tokens = Tensor::new(ids.as_slice(), &dev).unwrap();
        let mut fused = tiny_model(AttnImpl::Fused, &dev);
        let mut sdpa = tiny_model(AttnImpl::Sdpa, &dev);
        let a = rows(&fused.forward_all_logits(&tokens, 0).unwrap());
        let b = rows(&sdpa.forward_all_logits(&tokens, 0).unwrap());
        let diff = max_abs_diff(&a, &b);
        eprintln!("prefill flash vs sdpa: max |Δlogit| {diff:.3e}");
        assert!(diff <= 2e-2, "prefill: max abs diff {diff}");

        // Decode steps: the f16 vector sdpa against the f32 chain.
        for (i, &id) in ids.iter().enumerate().skip(10) {
            let t = Tensor::new(&[id], &dev).unwrap();
            let pos = 19 + (i - 10);
            let x = fused.forward(&t, pos).unwrap().to_vec1::<f32>().unwrap();
            let y = sdpa.forward(&t, pos).unwrap().to_vec1::<f32>().unwrap();
            let d = x
                .iter()
                .zip(&y)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0, f32::max);
            assert!(d <= 2e-2, "decode step {i}: max abs diff {d}");
            assert_eq!(argmax(&x), argmax(&y), "decode step {i}: argmax");
        }
    }

    /// `encode` follows the HF `hidden_states` numbering: index 0 is the
    /// embedding rows, `n_layer` is the normed output, and `n_layer - 1` is
    /// the raw residual the full forward's `l_out` tap shows.
    #[test]
    fn encode_indexes_hidden_states_like_transformers() {
        let Some(dev) = device_or_skip("encode_indexes_hidden_states_like_transformers") else {
            return;
        };
        let mut model = tiny_model(AttnImpl::Fused, &dev);
        let ids: Vec<u32> = (0..12).map(|i| (i * 3 + 2) % 32).collect();

        let (h0, _) = model.encode(&ids, 0).unwrap();
        let embed = model
            .embed_ids(&ids)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        assert_eq!(max_abs_diff(&rows(&h0), &rows(&embed)), 0.0);

        model.set_tap_capture(true);
        let tokens = Tensor::new(ids.as_slice(), &dev).unwrap();
        model.reset_cache().unwrap();
        model.forward(&tokens, 0).unwrap();
        let taps = model.take_taps();
        model.set_tap_capture(false);
        let tap = |name: &str| {
            taps.iter()
                .find(|(n, _)| n == name)
                .map(|(_, t)| t.clone())
                .unwrap_or_else(|| panic!("no tap {name}"))
        };
        let l_out_0 = tap("l_out-0").to_dtype(DType::BF16).unwrap();
        let (h1, _) = model.encode(&ids, 1).unwrap();
        assert_eq!(max_abs_diff(&rows(&h1), &rows(&l_out_0)), 0.0);

        let h_nextn = tap("h_nextn");
        let normed = model
            .final_norm(&h_nextn)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let (h2, _) = model.encode(&ids, 2).unwrap();
        assert_eq!(max_abs_diff(&rows(&h2), &rows(&normed)), 0.0);
        // And the pre-o_proj tap exists at the attention width.
        assert_eq!(tap("kqv_out-1").dims(), &[12, 256]);
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap()
    }
}
