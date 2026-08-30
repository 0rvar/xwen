use std::sync::Arc;

use anyhow::{Result, ensure};
use candle_core::{DType, Device, Module, Tensor};

use crate::config::XwenConfig;
use crate::gguf::{AttnQ8, QLinear, Weights};
use crate::kv_cache::{LayerCache, MaskKind};
use crate::qwen4exp::indexer::QsaSelection;
use crate::rope::Rope;

/// Token count up to and including which a q8_0-stored attention projection
/// reads its raw q8_0 bytes (the single-token gemv `ops::matmul_q8`, or the
/// small-batch mat-vec `ops::matmul_mv_ext` over the same plane where
/// `ops::mv_ext_window` admits the count); above it the dense f16 plane's
/// `matmul_f16` runs the tiled gemm. Set to the f16 path's own mv/mm break-even
/// so the q8 bytes cover the ENTIRE gemv range (the f16 plane would otherwise
/// run its own gemv over double the bytes for seq 1..=8) and the f16 gemm takes
/// over exactly where tiling wins — decode and short verify spans take the q8
/// path, prefill takes the f16 tensor gemm.
const Q8_DECODE_MAX_SEQ: usize = 8;

/// The prefill attention mask, built once per forward and shared across every
/// full-attention layer. It is a pure function of (kind, pos, seq_len) — and on
/// this architecture every attention layer is plain causal at the same head
/// count, so one build serves the whole stack. `raw` is the additive
/// `[seq, k_seq]` f32 mask the
/// manual/CPU fallback broadcast-adds; `sdpa` is that mask reshaped to
/// `[1, n_head, seq, k_seq]`, cast f16 and made contiguous for the Metal sdpa
/// kernel. Both are byte-identical to what the pre-hoist per-layer path built.
pub struct PrefillMask {
    raw: Tensor,
    sdpa: Tensor,
}

impl PrefillMask {
    /// Build the raw + sdpa-materialized masks for `n_head` query heads of
    /// `kind`, for a `seq`-token chunk at absolute position `pos`. `None` for a
    /// single decode token (seq==1), matching the pre-hoist path (which built no
    /// mask there).
    pub fn build(
        kind: MaskKind,
        n_head: usize,
        seq: usize,
        pos: usize,
        device: &Device,
    ) -> Result<Option<Self>> {
        let host = match crate::kv_cache::attn_mask_data(kind, seq, pos) {
            Some(h) => h,
            None => return Ok(None),
        };
        Ok(Some(Self::from_host(host, n_head, device)?))
    }

    /// The device half of `build`: upload an already-filled host mask and
    /// materialize the sdpa copy. Split from the CPU fill so a caller can time
    /// the two separately; the tensors are identical either way.
    pub fn from_host(
        host: crate::kv_cache::MaskHost,
        n_head: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::from_raw(crate::kv_cache::mask_tensor(host, device)?, n_head)
    }

    /// The same materialization over an additive `[seq, k_seq]` f32 mask a
    /// caller built itself. The QSA indexer is that caller: its per-query block
    /// selection already carries causality, so its mask REPLACES the causal one
    /// rather than composing with it, and it reaches sdpa through exactly this
    /// broadcast-to-f16 path.
    /// The f16 copy is ONE `[1, 1, s, kk]` plane, broadcast to `[1, n_head, s,
    /// kk]` as a view — the head axis carries stride 0 and no bytes.
    ///
    /// Every mask that reaches here is head-uniform (each is built from a
    /// `[seq, k_seq]` host mask), and candle's Metal sdpa takes the mask's
    /// strides rather than assuming it contiguous: `call_sdpa_full` forwards
    /// `M_strides[0..3]` (batch, head, query — the key stride is fixed at 1) and
    /// `scaled_dot_product_attention.metal:1940` advances the mask pointer by
    /// `head * M_strides[1]`, so a zero there hands every head the same row.
    /// Only the shape check is strict, and a broadcast view has the shape.
    ///
    /// Materializing the copy instead costs `n_head` times as much for nothing:
    /// 232 MB per QSA layer at 2200 tokens on the qwen4exp geometry, and
    /// 800 MB per layer at a 4k dense prefill on the 27B.
    pub fn from_raw(raw: Tensor, n_head: usize) -> Result<Self> {
        let (s, kk) = raw.dims2()?;
        let sdpa = raw
            .reshape((1, 1, s, kk))?
            .to_dtype(DType::F16)?
            .contiguous()?
            .broadcast_as((1, n_head, s, kk))?;
        Ok(Self { raw, sdpa })
    }
}

/// How the attention projection weights are held, decided at load (model.rs,
/// `XWEN_ATTN_F32`). Activations are f32 either way — the `F16` mode streams
/// dense f16 weight planes (GGUF-stored f16 on an f16-attention checkpoint, or
/// dequantized from a q8_0-stored one, whose decode gemv reads the raw q8_0
/// bytes instead) through the vendored mixed-dtype kernels (ops::matmul_f16),
/// so the stored weights are the only non-f32 values in the projection math
/// (the fork's exact precision structure).
#[derive(Clone, Copy)]
pub enum AttnWeights {
    /// Dense f16 weight planes — GGUF-stored f16, or dequantized from q8_0
    /// storage plus a raw-q8_0 decode alias (shipped default; Metal only).
    F16,
    /// Weights dequantized to dense f32 behind `QMatMul` (fully legacy).
    DequantF32,
}

/// One attention projection. `DenseF16` holds the weight as a dense f16 tensor
/// consumed by the vendored ggml-geometry f16-weight kernels: the f32
/// activation is never cast and the output is written f32 — f16 weight
/// streaming (the bandwidth win) with zero non-weight rounding. `Quant` keeps
/// the GGUF tensor behind candle's `QMatMul` (an F16-stored weight is
/// dequantized to dense f32 at load; a quantized-stored one runs candle's
/// quantized matmul).
pub(crate) enum Proj {
    Quant(QLinear),
    /// `[out_dim, in_dim]` f16.
    DenseF16(Tensor),
    /// A q8_0-stored attention weight (the current official checkpoint — the
    /// production default; introduced for the since-deleted unsloth UD file): the
    /// dequantized f16 dense plane consumed by the prefill/mm path (byte-identical
    /// to `DenseF16`) PLUS the raw q8_0 bytes consumed at seq <= 8 by the decode
    /// gemv (`ops::matmul_q8`, one weight pass per token) and, over the same
    /// plane, by the small-batch mat-vec (`ops::matmul_mv_ext`, one weight pass
    /// for the whole batch) wherever `ops::mv_ext_window` admits the token count.
    /// The stored q8_0 weights are the only quantized values in the decode chain
    /// (no activation/output rounding), at ~half the decode bandwidth of
    /// streaming the f16 plane.
    DenseF16Q8 {
        f16: Tensor,
        q8: AttnQ8,
    },
}

impl Proj {
    /// Load `<prefix>.<name>.weight` in the storage mode `weights` selects. The
    /// DeltaNet layers' projections ship under attention tensor names and take
    /// the same route, so this is shared with `linear_attn.rs`.
    pub(crate) fn load(w: &Weights, name: &str, weights: AttnWeights) -> Result<Self> {
        Ok(match weights {
            // q8_0-stored weights (the shipped Q4_K_M mix keeps attention and ssm
            // planes at 8 bits) build the dual-storage Proj; an f16-stored weight
            // returns no q8 alias and stays on the plain f16 plane.
            AttnWeights::F16 => match w.attn_proj(name)? {
                (f16, Some(q8)) => Proj::DenseF16Q8 { f16, q8 },
                (f16, None) => Proj::DenseF16(f16),
            },
            AttnWeights::DequantF32 => Proj::Quant(w.qlinear(name)?),
        })
    }

    /// Weight bytes this projection streams for a `seq`-token call — the byte
    /// floor its matmul cannot go under, for the `XWEN_GDN_PROFILE` line.
    ///
    /// It follows `forward`'s routing rather than the storage alone, because
    /// the two disagree exactly where it matters: a q8_0-stored weight streams
    /// its q8_0 bytes at decode and its (twice as wide) dequantized f16 plane
    /// above `Q8_DECODE_MAX_SEQ`. Activation and output bytes are excluded —
    /// at every shape here they are three orders of magnitude below the weight
    /// pass, and including them would blur the one quantity the line exists to
    /// compare against.
    pub(crate) fn weight_bytes(&self, seq: usize) -> u64 {
        match self {
            Proj::Quant(q) => q.weight_bytes(),
            Proj::DenseF16(w) => (w.elem_count() * DType::F16.size_in_bytes()) as u64,
            Proj::DenseF16Q8 { f16, q8 } => {
                if seq <= Q8_DECODE_MAX_SEQ && !crate::ops::attn_dequant() {
                    let p = &q8.plane;
                    let blocks = (p.out_dim * p.in_dim) / p.dtype.block_size();
                    (blocks * p.dtype.type_size()) as u64
                } else {
                    (f16.elem_count() * DType::F16.size_in_bytes()) as u64
                }
            }
        }
    }

    /// f32 in, f32 out on every variant; the stored quantized/f16 weights are the
    /// only non-f32 values the matmul ever sees.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Proj::Quant(q) => Ok(q.forward(x)?),
            Proj::DenseF16(w) => crate::ops::matmul_f16(w, x),
            Proj::DenseF16Q8 { f16, q8 } => {
                let seq = x.dim(0)?;
                if seq <= Q8_DECODE_MAX_SEQ && !crate::ops::attn_dequant() {
                    let p = &q8.plane;
                    // Small-batch window: one weight pass serves the whole token
                    // batch over the SAME q8_0 bytes the single-token gemv reads,
                    // where the gemv re-reads the entire weight once per token.
                    // The plan comes from `mv_ext_window` and is threaded through
                    // verbatim, so `XWEN_MV_EXT_CLASSIC` reverts this site exactly
                    // as it reverts the `QLinear` ones. `XWEN_MV_EXT_MAX_SEQ` is
                    // narrower here: the outer `Q8_DECODE_MAX_SEQ` arm has already
                    // routed seq > 8 to the dense f16 plane, so raising the window
                    // past 8 widens the `QLinear` sites but not this one — at this
                    // site the knob's effective range is capped at 8.
                    // The small-batch kernel reads the activation as
                    // float4/float4x4, which Metal requires 16-byte aligned,
                    // while the gemv reads scalars and takes any offset — so a
                    // view the kernel could not read keeps the gemv. Every
                    // activation reaching this window today is offset-0, and the
                    // `mv_ext` provenance field is env-derived and cannot see a
                    // per-call fallback: if a strided view ever starts landing
                    // here, this guard would quietly stamp "fused" provenance on
                    // gemv rounds — record what ran (as the `delta` field does)
                    // before letting that happen.
                    if let Some(r1ptg) = crate::ops::mv_ext_window(seq)
                        && crate::ops::mv_ext_supported(p.dtype, p.in_dim)
                        && x.layout()
                            .start_offset()
                            .is_multiple_of(16 / DType::F32.size_in_bytes())
                    {
                        crate::ops::matmul_mv_ext(p, x, r1ptg)
                    } else {
                        crate::ops::matmul_q8(&p.buffer, p.base_off, p.out_dim, p.in_dim, x)
                    }
                } else {
                    // Prefill (and XWEN_ATTN_DEQUANT): the dense f16 plane, the
                    // path an f16-attention checkpoint always takes.
                    crate::ops::matmul_f16(f16, x)
                }
            }
        }
    }
}

/// One Qwen 3.6 full-attention block (layer indices where `(il + 1) % 4 == 0`;
/// every other layer is gated DeltaNet, in `linear_attn.rs`).
///
/// `attn_q` is DOUBLE width and per-head interleaved — head h occupies
/// `[q_h(head_dim) | gate_h(head_dim)]` — so q and the output gate come out of
/// one projection and are separated by a strided view. Both q and k take an
/// RMSNorm over the head dim before rope; rope is partial NEoX over the first
/// `n_rot` dims only; sdpa runs at scale `1/sqrt(head_dim)`; and `sigmoid(gate)`
/// multiplies the attention output ELEMENTWISE (the gate is head_dim wide, not
/// one scalar per head) before `attn_output`. Masking is plain causal
/// everywhere — there is no sliding window in this architecture.
///
/// Activations are f32 end-to-end; with `AttnWeights::F16` each projection runs
/// the vendored mixed-dtype kernels (f16 weights x f32 activations, f32
/// accumulate/output), so the dense f16 weight planes stream at f16 width with
/// no other rounding (a q8_0-stored checkpoint's decode gemv streams the raw
/// q8_0 bytes instead). f16 otherwise appears only where both weight modes share
/// it: the KV cache and the sdpa kernel.
pub struct AttnBlock {
    /// `attn_q`: q and the output gate, interleaved per head.
    qg_proj: Proj,
    k_proj: Proj,
    v_proj: Proj,
    o_proj: Proj,
    /// QK-norm weights, f32 (candle's rms_norm requires weight dtype == x dtype).
    q_norm: candle_nn::RmsNorm,
    k_norm: candle_nn::RmsNorm,
    rope: Arc<Rope>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
}

impl AttnBlock {
    /// `w` is positioned at the block prefix (e.g. `blk.7`).
    pub fn new(
        w: &Weights,
        cfg: &XwenConfig,
        il: usize,
        rope: Arc<Rope>,
        weights: AttnWeights,
    ) -> Result<Self> {
        let proj = |name: &str| -> Result<Proj> { Proj::load(w, name, weights) };
        let norm = |name: &str| -> Result<candle_nn::RmsNorm> {
            Ok(candle_nn::RmsNorm::new(w.dense_f32(name)?, cfg.rms_eps))
        };
        Ok(Self {
            qg_proj: proj("attn_q")?,
            k_proj: proj("attn_k")?,
            v_proj: proj("attn_v")?,
            o_proj: proj("attn_output")?,
            q_norm: norm("attn_q_norm")?,
            k_norm: norm("attn_k_norm")?,
            rope,
            n_head: cfg.n_head(il),
            n_kv_head: cfg.n_kv_head,
            head_dim: cfg.head_dim,
        })
    }

    /// TEST ONLY: a block over weights the caller chose, instead of a GGUF.
    ///
    /// It exists so a test can make one contribution of the block vanish — a
    /// zeroed `attn_output` — and assert what the rest of a graph does without
    /// it. `MtpDrafter`'s residual-anchor test is the reason it exists: with real
    /// weights that property has no falsifiable knob. Additive and unreachable
    /// from any shipped path; the loading constructor above is untouched.
    ///
    /// Every weight is `[out_dim, in_dim]` f16, the layout `Proj::DenseF16`
    /// consumes, so this needs a Metal device like the rest of the ops tests.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_raw_weights(
        qg: Tensor,
        k: Tensor,
        v: Tensor,
        o: Tensor,
        q_norm: candle_nn::RmsNorm,
        k_norm: candle_nn::RmsNorm,
        rope: Arc<Rope>,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            qg_proj: Proj::DenseF16(qg),
            k_proj: Proj::DenseF16(k),
            v_proj: Proj::DenseF16(v),
            o_proj: Proj::DenseF16(o),
            q_norm,
            k_norm,
            rope,
            n_head,
            n_kv_head,
            head_dim,
        }
    }

    /// Whether this block's projections hold the q8_0 decode alias (a
    /// q8_0-quantized checkpoint under `AttnWeights::F16`). Uniform across the
    /// block's projections, so `q_proj` is representative. Surfaced so the model
    /// can record the attention decode path in dump provenance.
    pub fn uses_q8_decode(&self) -> bool {
        matches!(self.qg_proj, Proj::DenseF16Q8 { .. })
    }

    /// Build this block's prefill mask for `cache` at (seq, pos), or None for a
    /// single decode token. Production prefill hoists this out of the layer loop
    /// — one build per kind, shared across every layer of that kind — while
    /// single-block callers (tests, benches) build it per call.
    pub fn prefill_mask(
        &self,
        cache: &LayerCache,
        seq: usize,
        pos: usize,
    ) -> Result<Option<PrefillMask>> {
        PrefillMask::build(cache.mask_kind(), self.n_head, seq, pos, &cache.device())
    }

    /// x_normed: [seq, hidden] f32 (already attn_norm'ed by the caller).
    /// `mask` is the pre-built, hoisted mask for this layer's kind (None for a
    /// single decode token). Returns [seq, hidden] f32.
    ///
    /// `qsa` is the sparse-attention overlay for a `qwen4exp` QSA layer (D16).
    /// `None` and `QsaSelection::Dense` are the same path, byte for byte: every
    /// other architecture passes `None`, and a below-budget QSA layer passes
    /// `Dense`, which is not an approximation of dense attention but literally
    /// is it.
    pub fn forward(
        &self,
        x_normed: &Tensor,
        cache: &mut LayerCache,
        pos: usize,
        mask: Option<&PrefillMask>,
        qsa: Option<&QsaSelection>,
    ) -> Result<Tensor> {
        let (seq, _hidden) = x_normed.dims2()?;

        // One projection, two outputs: `attn_q` is [n_head, 2, head_dim] per
        // token with q first and the gate second WITHIN each head, so the split
        // is a stride away — not a halving of the whole row.
        let qg = self
            .qg_proj
            .forward(x_normed)?
            .reshape((seq, self.n_head, 2, self.head_dim))?;
        let q = qg
            .narrow(2, 0, 1)?
            .contiguous()?
            .reshape((seq, self.n_head, self.head_dim))?;
        let gate = qg
            .narrow(2, 1, 1)?
            .contiguous()?
            .reshape((seq, self.n_head, self.head_dim))?;
        let k = self
            .k_proj
            .forward(x_normed)?
            .reshape((seq, self.n_kv_head, self.head_dim))?;
        let v = self
            .v_proj
            .forward(x_normed)?
            .reshape((seq, self.n_kv_head, self.head_dim))?;

        // QK-norm: RMSNorm over head_dim before rope, in [seq, head, dim]
        // layout where head_dim is contiguous last.
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Fused attention-glue kernels (Metal): each is bit-identical to the
        // candle chain it replaces (ops::attn_glue bitwise tests), so this is a
        // pure dispatch/traffic optimization. XWEN_ATTN_GLUE_CLASSIC reverts
        // every glue site (here, sdpa_attention, and Rope::rotate) to the
        // candle chains; non-Metal devices always run them.
        let fused_glue =
            matches!(x_normed.device(), Device::Metal(_)) && !crate::ops::attn_glue_classic();

        // To [head, seq, head_dim] for rope + attention. At seq==1 (decode) the
        // [1, head, dim] and [head, 1, dim] layouts share byte order, so a reshape
        // (metadata only) is bit-identical to transpose+contiguous and drops three
        // copy dispatches per layer on the hot decode path. seq>1 (prefill) is a
        // real permutation: one fused permute-copy pass each for q/k (which must
        // stay f32 for rope), and for v a single permute+f16-cast pass that also
        // absorbs the cache-append cast (v is not roped).
        let (q, k, v) = if seq == 1 {
            (
                q.reshape((self.n_head, 1, self.head_dim))?,
                k.reshape((self.n_kv_head, 1, self.head_dim))?,
                v.reshape((self.n_kv_head, 1, self.head_dim))?,
            )
        } else if fused_glue {
            (
                crate::ops::permute_01(&q)?,
                crate::ops::permute_01(&k)?,
                crate::ops::permute_01_f16(&v)?, // [n_kv, seq, hd] f16, cache-ready
            )
        } else {
            (
                q.transpose(0, 1)?.contiguous()?,
                k.transpose(0, 1)?.contiguous()?,
                v.transpose(0, 1)?.contiguous()?,
            )
        };

        // On the fused-glue path rope stores its OUTPUT dtype directly (the
        // kernel computes in f32 and rounds only the final store — bit-
        // identical to f32 rope + cast_f16), folding the standalone post-rope
        // casts away: k is stored f16 always (it flows straight to the f16
        // cache), and q is stored f16 exactly where its consumer is the f16
        // decode sdpa. q stays f32 for prefill (the flash kernel REQUIRES f32
        // q, and the flash-classic route keeps its op sequence unchanged: rope
        // f32 + the in-sdpa cast) and under the XWEN_SDPA_F32 experiment
        // (whose sdpa consumes f32 q).
        let (q, k) = if fused_glue {
            let q_dt = if seq == 1 && !crate::ops::sdpa_f32() {
                DType::F16
            } else {
                DType::F32
            };
            self.rope.apply_dt(&q, &k, pos, q_dt, DType::F16)?
        } else {
            self.rope.apply(&q, &k, pos)?
        };

        // Cache in f16; sdpa runs in f16. The additive mask is pre-built and
        // hoisted by the caller (one per kind per forward; None at decode).
        // The fused paths delivered k (rope f16 store) and prefill v (cast
        // folded into its permute above) already f16.
        let k16 = if k.dtype() == DType::F16 {
            k
        } else {
            k.to_dtype(DType::F16)?
        };
        let v16 = if v.dtype() == DType::F16 {
            v
        } else if fused_glue {
            crate::ops::cast_f16(&v)?
        } else {
            v.to_dtype(DType::F16)?
        };
        let (k_all, v_all) = cache.append(&k16, &v16)?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();

        // The QSA overlay (D16), in its own two branches so the four seq==1
        // shortcuts above and below stay exactly as they were.
        //
        //  - `Rows` is the decode shape: gather the selected K/V rows into
        //    packed contiguous planes and attend over those with NO mask.
        //    candle's vector sdpa (seq == 1) silently IGNORES a mask, so
        //    masking is not available on this route — the gather is what makes
        //    the selection real (D11).
        //  - `Mask` is the prefill shape: an additive per-query mask that
        //    already includes causality, so it replaces the hoisted causal one
        //    instead of being added to it.
        //
        // Each arm is checked against the shape it is FOR, because getting it
        // wrong is silent rather than loud: a `Mask` at `seq == 1` reaches
        // candle's vector sdpa, which ignores the mask argument entirely and
        // returns dense attention over the whole prefix — the right shape, the
        // wrong answer, no error anywhere. `Rows` at `seq > 1` is the mirror
        // image: the gather is not per-query, so every query would attend over
        // one query's selection, and dropping the caller's causal mask along
        // with it lets a token see its own future.
        let (k_all, v_all) = match qsa {
            Some(QsaSelection::Rows(rows)) => {
                ensure!(
                    seq == 1,
                    "QsaSelection::Rows is the decode overlay and selects one query's rows; \
                     got seq {seq}"
                );
                ensure!(
                    mask.is_none(),
                    "QsaSelection::Rows attends over gathered rows with NO mask (candle's \
                     vector sdpa ignores one), so a caller-supplied mask would be silently \
                     dropped"
                );
                (gather_rows(&k_all, rows)?, gather_rows(&v_all, rows)?)
            }
            _ => (k_all, v_all),
        };
        let qsa_mask = match qsa {
            Some(QsaSelection::Mask(m)) => {
                ensure!(
                    seq > 1,
                    "QsaSelection::Mask is the prefill overlay; at seq 1 candle's vector sdpa \
                     ignores the mask and attends densely instead — use Rows"
                );
                Some(PrefillMask::from_raw(m.clone(), self.n_head)?)
            }
            _ => None,
        };
        let mask = qsa_mask.as_ref().or(mask);

        // Attention proper. The vendored flash kernel is NOT a route here: it is
        // compiled at head dim 128 (flash.metal's `BD == 128`) and this
        // architecture is 256, so prefill goes through candle's sdpa with the
        // materialized mask that `model.rs` builds. Two routes remain:
        //  - Metal: candle's sdpa (including XWEN_SDPA_F32), consuming the
        //    materialized f16 mask. On the fused-glue decode route the entry
        //    casts are folded away (q arrives f16 from rope) — bit-identical by
        //    construction.
        //  - non-Metal: the explicit f32 fallback (CPU tests, reference oracle),
        //    consuming the raw `[seq, k_seq]` additive mask.
        let attn = if matches!(x_normed.device(), Device::Metal(_)) {
            self.sdpa_attention(&q, &k_all, &v_all, mask.map(|m| &m.sdpa), scale)?
        } else {
            self.manual_attention(&q, &k_all, &v_all, mask.map(|m| &m.raw), scale, seq)?
        }; // [n_head, seq, head_dim] f32 — except the fused-glue decode route,
        // which hands over the raw f16 sdpa output (see sdpa_attention).

        // The output gate is head_dim wide, so it multiplies elementwise rather
        // than broadcasting one scalar per head. It comes from the SAME
        // projection as q (interleaved per head) and applies after attention,
        // before o_proj.
        let gate = if seq == 1 {
            gate.reshape((self.n_head, 1, self.head_dim))?
        } else if fused_glue {
            crate::ops::permute_01(&gate)?
        } else {
            gate.transpose(0, 1)?.contiguous()?
        };
        let attn = (attn.to_dtype(DType::F32)? * candle_nn::ops::sigmoid(&gate)?)?;

        // Back to [seq, n_head*head_dim] then o_proj. Same seq==1 shortcut: the
        // [head, 1, dim] -> [1, head*dim] regroup is byte-identical to
        // transpose+contiguous+reshape, so decode skips the copy.
        let out = if seq == 1 {
            attn.reshape((seq, self.n_head * self.head_dim))?
        } else if fused_glue {
            crate::ops::permute_01(&attn)?.reshape((seq, self.n_head * self.head_dim))?
        } else {
            attn.transpose(0, 1)?
                .contiguous()?
                .reshape((seq, self.n_head * self.head_dim))?
        };
        self.o_proj.forward(&out)
    }

    /// Metal MLX fused attention. q [n_head, seq, hd] — f32, or already f16 on
    /// the fused-glue decode path (rope stored it f16; same bits as the cast
    /// this method would otherwise run). k/v [n_kv_head, K, hd] f16. GQA
    /// (n_head multiple of n_kv_head) is handled by the kernel; k/v are not
    /// pre-tiled. The kernel runs in f16: q is cast below where still f32, and
    /// `mask` arrives pre-materialized (f16, `[1, n_head, seq, k]`, contiguous)
    /// from the hoisted `PrefillMask`. Returns [n_head, seq, hd] — f32, except
    /// on the fused-glue decode path (seq == 1), where the raw f16 sdpa output
    /// is returned and the caller widens it (exact — widening never rounds).
    fn sdpa_attention(
        &self,
        q: &Tensor,
        k_all: &Tensor,
        v_all: &Tensor,
        mask: Option<&Tensor>,
        scale: f32,
    ) -> Result<Tensor> {
        // Experiment hook (`XWEN_SDPA_F32`): run the whole sdpa in f32. The
        // default path below is untouched when the env is absent.
        if crate::ops::sdpa_f32() {
            return self.sdpa_attention_f32(q, k_all, v_all, mask, scale);
        }
        // Metal-only path, so the glue switch alone picks the cast kernels. q
        // arrives contiguous (rope output / decode reshape), so the fused cast
        // needs no trailing contiguous; both casts are bit-identical to
        // to_dtype (RTNE narrowing, exact widening).
        let fused_glue = !crate::ops::attn_glue_classic();
        let seq = q.dim(1)?;
        let q = if q.dtype() == DType::F16 {
            q.unsqueeze(0)? // fused decode: rope stored q f16, metadata only
        } else if fused_glue {
            crate::ops::cast_f16(q)?.unsqueeze(0)? // [1, n_head, seq, hd]
        } else {
            q.to_dtype(DType::F16)?.unsqueeze(0)?.contiguous()?
        };
        // k/v stay as the cache's narrowed views: rows within a head are
        // contiguous and only the head dimension carries the max_ctx gap, which
        // sdpa handles via the per-head k/v stride it is passed. Materializing a
        // packed copy here would grow with context for no benefit.
        let k = k_all.unsqueeze(0)?; // [1, n_kv_head, K, hd], head-strided
        let v = v_all.unsqueeze(0)?;

        let out = candle_nn::ops::sdpa(&q, &k, &v, mask, false, scale, 1.0)?;
        let out = out.squeeze(0)?;
        Ok(if fused_glue && seq == 1 {
            out // f16, consumed directly by the f16-input gate kernel
        } else if fused_glue {
            crate::ops::cast_f32(&out)?
        } else {
            out.to_dtype(DType::F32)?
        })
    }

    /// f32 sdpa experiment path (`XWEN_SDPA_F32`, non-default): the same
    /// candle Metal sdpa kernel family as the default path, dispatched in f32
    /// — q stays f32 (no f16 cast), the cached f16 k/v are widened to f32
    /// (exact: widening never rounds), and the pre-materialized f16 mask is
    /// widened too (also exact — it holds only 0 and -inf). The pinned candle
    /// rev supports SdpaDType::F32 for both the full (seq > 1,
    /// `steel_attention_float32_*`) and vector (seq == 1,
    /// `sdpa_vector_float_*`) kernels at head_dim 128 with GQA; f32 is only
    /// rejected at head_dim 512. Output matches the default path's shape and
    /// dtype ([n_head, seq, hd] f32) so the downstream flow is unchanged.
    fn sdpa_attention_f32(
        &self,
        q: &Tensor,
        k_all: &Tensor,
        v_all: &Tensor,
        mask: Option<&Tensor>,
        scale: f32,
    ) -> Result<Tensor> {
        // q arrives contiguous (rope output / decode reshape). Widening the
        // cache's head-strided k/v views also packs them contiguous.
        let q = q.unsqueeze(0)?; // [1, n_head, seq, hd] f32
        let k = k_all.to_dtype(DType::F32)?.unsqueeze(0)?;
        let v = v_all.to_dtype(DType::F32)?.unsqueeze(0)?;
        // candle requires the mask dtype to match q's.
        let mask32 = mask.map(|m| m.to_dtype(DType::F32)).transpose()?;
        let out = candle_nn::ops::sdpa(&q, &k, &v, mask32.as_ref(), false, scale, 1.0)?;
        Ok(out.squeeze(0)?)
    }

    /// Explicit softmax(q·kᵀ·scale + mask)·v in q's dtype (f32), GQA via a
    /// broadcast over the query group dim. q [n_head, seq, hd] f32, k/v
    /// [n_kv_head, K, hd] f16. Non-Metal fallback (CPU tests, Reference oracle).
    fn manual_attention(
        &self,
        q: &Tensor,
        k_all: &Tensor,
        v_all: &Tensor,
        mask: Option<&Tensor>,
        scale: f32,
        seq: usize,
    ) -> Result<Tensor> {
        let g = self.n_head / self.n_kv_head;
        let k_seq = k_all.dim(1)?;
        let k = k_all.to_dtype(q.dtype())?;
        let v = v_all.to_dtype(q.dtype())?;

        let q4 = q.reshape((self.n_kv_head, g, seq, self.head_dim))?;
        let k4 = k.reshape((self.n_kv_head, 1, k_seq, self.head_dim))?;
        let v4 = v.reshape((self.n_kv_head, 1, k_seq, self.head_dim))?;

        let scores = q4
            .broadcast_matmul(&k4.transpose(2, 3)?)?
            .affine(scale as f64, 0.0)?;
        let scores = match mask {
            // The additive mask is built f32, matching the scores' dtype.
            Some(m) => {
                scores.broadcast_add(&m.to_dtype(scores.dtype())?.reshape((1, 1, seq, k_seq))?)?
            }
            None => scores,
        };
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.broadcast_matmul(&v4)?; // [n_kv_head, g, seq, hd]
        Ok(out.reshape((self.n_head, seq, self.head_dim))?)
    }
}

/// Pack the cache rows `rows` names out of a `[heads, len, head_dim]` cache
/// view into a contiguous `[heads, n_sel, head_dim]` plane — the decode half of
/// the QSA overlay.
///
/// One `index_select` per head, deliberately, rather than one call over the
/// whole rank-3 view. A cache view is a `narrow` of a `max_ctx`-slot buffer, so
/// it is strided across the head axis, and candle's Metal `index_select`
/// MIS-HANDLES a strided source at the pinned rev: `call_index_select` passes
/// the indexed dimension's SIZE where the kernel's `get_strided_index` expects
/// the tensor's RANK (candle-metal-kernels indexing.metal), so every gathered
/// element is read from a garbage offset — silently, with the right shape. A
/// single head's slice IS contiguous by candle's own rule (a leading axis of
/// extent 1 is skipped when checking strides), which puts each of these
/// dispatches on the kernel's correct contiguous path. Head counts here are 2-4,
/// so this is a handful of dispatches, not a loop that matters.
fn gather_rows(t: &Tensor, rows: &Tensor) -> Result<Tensor> {
    let (heads, len, head_dim) = t.dims3()?;
    let mut packed = Vec::with_capacity(heads);
    for h in 0..heads {
        packed.push(
            t.narrow(0, h, 1)?
                .reshape((len, head_dim))?
                .index_select(rows, 0)?,
        );
    }
    Ok(Tensor::stack(&packed, 0)?)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Arch, LayerKind, RopeKind};
    use candle_core::Device;
    use candle_core::quantized::{GgmlDType, QTensor};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

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

    fn dense(rows: usize, cols: usize, seed: u64, dev: &Device) -> Tensor {
        Tensor::from_vec(seeded(rows * cols, seed), (rows, cols), dev).unwrap()
    }

    struct RawWeights {
        /// [n_head * 2 * hd, hidden] — q and gate interleaved per head.
        wqg: Tensor,
        wk: Tensor,
        wv: Tensor,
        wo: Tensor,
        qn: Tensor,
        kn: Tensor,
    }

    fn raw_weights(
        n_head: usize,
        n_kv: usize,
        hd: usize,
        hidden: usize,
        dev: &Device,
    ) -> RawWeights {
        let near_one = |dim: usize, seed: u64| {
            Tensor::from_vec(
                seeded(dim, seed)
                    .iter()
                    .map(|x| 1.0 + 0.1 * x)
                    .collect::<Vec<f32>>(),
                dim,
                dev,
            )
            .unwrap()
        };
        RawWeights {
            wqg: dense(n_head * 2 * hd, hidden, 1, dev),
            wk: dense(n_kv * hd, hidden, 2, dev),
            wv: dense(n_kv * hd, hidden, 3, dev),
            wo: dense(hidden, n_head * hd, 5, dev),
            qn: near_one(hd, 6),
            kn: near_one(hd, 7),
        }
    }

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    /// Write the weights to a throwaway GGUF (F32 quant) and load an AttnBlock,
    /// exercising the real gguf.rs loading seam rather than a test-only shortcut.
    fn build_block(
        w: &RawWeights,
        cfg: &XwenConfig,
        il: usize,
        rope: Arc<Rope>,
        dev: &Device,
        weights: AttnWeights,
    ) -> AttnBlock {
        let q = |t: &Tensor| {
            QTensor::quantize(&t.to_device(&Device::Cpu).unwrap(), GgmlDType::F32).unwrap()
        };
        let (wqg, wk, wv, wo, qn, kn) =
            (q(&w.wqg), q(&w.wk), q(&w.wv), q(&w.wo), q(&w.qn), q(&w.kn));
        let tensors: Vec<(&str, &QTensor)> = vec![
            ("blk.0.attn_q.weight", &wqg),
            ("blk.0.attn_k.weight", &wk),
            ("blk.0.attn_v.weight", &wv),
            ("blk.0.attn_output.weight", &wo),
            ("blk.0.attn_q_norm.weight", &qn),
            ("blk.0.attn_k_norm.weight", &kn),
        ];
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path: PathBuf =
            std::env::temp_dir().join(format!("xwen_attn_test_{}_{id}.gguf", std::process::id()));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            candle_core::quantized::gguf_file::write(&mut f, &[], &tensors).unwrap();
        }
        let src = crate::gguf::open(&path, dev).unwrap();
        let loaded = Weights::from_gguf(src).pp("blk.0");
        let block = AttnBlock::new(&loaded, cfg, il, rope, weights).unwrap();
        let _ = std::fs::remove_file(&path);
        block
    }

    fn test_cfg(n_head: usize, n_kv: usize, hd: usize, hidden: usize, n_rot: usize) -> XwenConfig {
        XwenConfig {
            arch: Arch::Moe,
            general_name: None,
            n_layer: 4,
            hidden,
            vocab: 32,
            n_head: vec![n_head; 4],
            n_kv_head: n_kv,
            head_dim: hd,
            // Every layer full-attention: this module's tests never build a
            // DeltaNet cache.
            layer_kind: vec![LayerKind::Full; 4],
            linear_k_heads: 1,
            linear_v_heads: 1,
            linear_head_dim: 2,
            conv_kernel: 4,
            dense_ff: 0,
            n_expert: 0,
            n_expert_used: 0,
            expert_ff: 0,
            shared_expert_ff: 0,
            rms_eps: 1e-6,
            n_ctx_train: 4096,
            rope: RopeKind::Plain {
                freq_base: 1e7,
                n_rot,
            },
            eog_tokens: vec![2],
            qwen4exp: None,
        }
    }

    /// A from-scratch f32 attention over the full token sequence: the strided
    /// q/gate split, QK-norm, partial rope, the f16 KV round-trip and the
    /// elementwise sigmoid gate, written out independently of the block.
    /// Returns [total, hidden].
    ///
    /// `last_row_keys`, when given, restricts the FINAL query row to exactly the
    /// named key positions instead of the whole causal prefix — the reference
    /// for a decode step under a QSA `Rows` selection. Every other row keeps its
    /// ordinary causal visibility. Masking the dropped columns with -inf is
    /// exactly slicing them out: softmax gives them weight 0 and the surviving
    /// weights renormalize over the same sum a sliced K/V would produce.
    fn naive_forward(
        w: &RawWeights,
        rope: &Rope,
        x: &Tensor,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        last_row_keys: Option<&[u32]>,
    ) -> Tensor {
        let dev = x.device();
        let (total, _hidden) = x.dims2().unwrap();
        let eps = 1e-6f64;

        let lin = |x: &Tensor, wt: &Tensor| x.matmul(&wt.t().unwrap()).unwrap();
        let rms = |t: &Tensor, weight: &Tensor| {
            let ms = (t.sqr().unwrap().sum_keepdim(2).unwrap() / hd as f64).unwrap();
            t.broadcast_div(&(ms + eps).unwrap().sqrt().unwrap())
                .unwrap()
                .broadcast_mul(&weight.reshape((1, 1, hd)).unwrap())
                .unwrap()
        };

        // The double-width projection, split per head: [.., h, 0, :] is q and
        // [.., h, 1, :] is the gate.
        let qg = lin(x, &w.wqg).reshape((total, n_head, 2, hd)).unwrap();
        let q = qg
            .narrow(2, 0, 1)
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((total, n_head, hd))
            .unwrap();
        let gate = qg
            .narrow(2, 1, 1)
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((total, n_head, hd))
            .unwrap();
        let k = lin(x, &w.wk).reshape((total, n_kv, hd)).unwrap();
        let v = lin(x, &w.wv).reshape((total, n_kv, hd)).unwrap();

        let q = rms(&q.transpose(0, 1).unwrap().contiguous().unwrap(), &w.qn);
        let k = rms(&k.transpose(0, 1).unwrap().contiguous().unwrap(), &w.kn);
        let v = v.transpose(0, 1).unwrap().contiguous().unwrap();
        let (q, k) = rope.apply(&q, &k, 0).unwrap();
        // Round-trip k/v through f16 exactly as the cache does.
        let k = k
            .to_dtype(DType::F16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let v = v
            .to_dtype(DType::F16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        let g = n_head / n_kv;
        let scale = 1.0f64 / (hd as f64).sqrt();
        let q4 = q.reshape((n_kv, g, total, hd)).unwrap();
        let k4 = k.reshape((n_kv, 1, total, hd)).unwrap();
        let v4 = v.reshape((n_kv, 1, total, hd)).unwrap();
        let scores = q4
            .broadcast_matmul(&k4.transpose(2, 3).unwrap())
            .unwrap()
            .affine(scale, 0.0)
            .unwrap();

        let mut mask = vec![0f32; total * total];
        for qi in 0..total {
            for kj in 0..total {
                if kj > qi {
                    mask[qi * total + kj] = f32::NEG_INFINITY;
                }
            }
        }
        if let Some(keys) = last_row_keys {
            let last = (total - 1) * total;
            mask[last..].fill(f32::NEG_INFINITY);
            for &k in keys {
                assert!(
                    (k as usize) < total,
                    "key position {k} is beyond the {total}-token sequence"
                );
                mask[last + k as usize] = 0.0;
            }
        }
        let mask = Tensor::from_vec(mask, (1, 1, total, total), dev).unwrap();
        let scores = scores.broadcast_add(&mask).unwrap();
        let probs = candle_nn::ops::softmax_last_dim(&scores).unwrap();
        let out = probs
            .broadcast_matmul(&v4)
            .unwrap()
            .reshape((n_head, total, hd))
            .unwrap();

        // Elementwise sigmoid gate over the full head width, then o_proj.
        let gate = gate.transpose(0, 1).unwrap().contiguous().unwrap();
        let gated = (out * candle_nn::ops::sigmoid(&gate).unwrap()).unwrap();
        let flat = gated
            .transpose(0, 1)
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((total, n_head * hd))
            .unwrap();
        lin(&flat, &w.wo)
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    /// The block's prefill-then-decode walk equals a single naive pass over the
    /// whole sequence: the strided split, the QK-norm, the partial rope, the
    /// cache and the gate all have to line up for this to hold.
    #[test]
    fn forward_matches_naive_prefill_and_decode() {
        let dev = Device::Cpu;
        let (n_head, n_kv, hd, hidden) = (4usize, 2usize, 8usize, 12usize);
        let cfg = test_cfg(n_head, n_kv, hd, hidden, 4);
        let rope = Arc::new(Rope::new(cfg.rope(), 32, &dev).unwrap());
        let w = raw_weights(n_head, n_kv, hd, hidden, &dev);
        let block = build_block(&w, &cfg, 0, rope.clone(), &dev, AttnWeights::DequantF32);

        let total = 6usize;
        let x = dense(total, hidden, 42, &dev);
        let want = naive_forward(&w, &rope, &x, n_head, n_kv, hd, None);

        // Prefill the first 4 tokens, then decode the last 2 one at a time.
        let mut cache = LayerCache::new(&cfg, 0, 32, &dev).unwrap();
        let prefill = x.narrow(0, 0, 4).unwrap().contiguous().unwrap();
        let mask = block.prefill_mask(&cache, 4, 0).unwrap();
        let mut got = vec![
            block
                .forward(&prefill, &mut cache, 0, mask.as_ref(), None)
                .unwrap(),
        ];
        for t in 4..total {
            let row = x.narrow(0, t, 1).unwrap().contiguous().unwrap();
            got.push(block.forward(&row, &mut cache, t, None, None).unwrap());
        }
        let got = Tensor::cat(&got, 0).unwrap();

        assert!(
            max_abs_diff(&got, &want) < 2e-3,
            "block diverged from the naive reference by {}",
            max_abs_diff(&got, &want)
        );
    }

    /// q and the gate are interleaved WITHIN each head, not stacked as two
    /// halves of the row. A block whose weight puts a known constant in one
    /// head's gate slot and zero in every other must gate exactly that head.
    #[test]
    fn gate_is_read_from_the_interleaved_half_of_each_head() {
        let dev = Device::Cpu;
        let (n_head, n_kv, hd, hidden) = (3usize, 1usize, 4usize, 5usize);
        let cfg = test_cfg(n_head, n_kv, hd, hidden, 2);
        let rope = Arc::new(Rope::new(cfg.rope(), 8, &dev).unwrap());
        let mut w = raw_weights(n_head, n_kv, hd, hidden, &dev);

        // Rebuild attn_q so that every q slot is zero and only head 1's gate
        // slot is non-zero. Rows are ordered [h0.q, h0.gate, h1.q, h1.gate, ...],
        // each `hd` rows wide.
        let mut wqg = vec![0f32; n_head * 2 * hd * hidden];
        let gate_row0 = (1 * 2 + 1) * hd; // head 1, gate half
        for r in 0..hd {
            wqg[(gate_row0 + r) * hidden] = 8.0; // large -> sigmoid ~ 1
        }
        w.wqg = Tensor::from_vec(wqg, (n_head * 2 * hd, hidden), &dev).unwrap();
        // The output projection reads head 1's slice straight through so the
        // gating is visible in the result.
        let mut wo = vec![0f32; hidden * n_head * hd];
        for r in 0..hd {
            wo[r * (n_head * hd) + hd + r] = 1.0; // hidden row r <- head 1, dim r
        }
        w.wo = Tensor::from_vec(wo, (hidden, n_head * hd), &dev).unwrap();

        let block = build_block(&w, &cfg, 0, rope.clone(), &dev, AttnWeights::DequantF32);
        let mut cache = LayerCache::new(&cfg, 0, 8, &dev).unwrap();

        // One token whose first input component is 1, so head 1's gate logit is
        // 8 and every other head's is 0.
        let mut row = vec![0f32; hidden];
        row[0] = 1.0;
        let x = Tensor::from_vec(row, (1, hidden), &dev).unwrap();
        let out = block.forward(&x, &mut cache, 0, None, None).unwrap();

        let want = naive_forward(&w, &rope, &x, n_head, n_kv, hd, None);
        assert!(
            max_abs_diff(&out, &want) < 1e-4,
            "interleaved gate split disagrees with the reference"
        );
        // And the gate really is near-saturated rather than near-0.5: a
        // half-and-half split of the row would have read a zero weight here.
        let got: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        let attn_only: Vec<f32> = {
            let v = naive_forward(&w, &rope, &x, n_head, n_kv, hd, None);
            v.flatten_all().unwrap().to_vec1().unwrap()
        };
        assert_eq!(got.len(), attn_only.len());
        assert!(
            got.iter().any(|v| v.abs() > 1e-6),
            "gated output collapsed to zero, so nothing was tested"
        );
    }

    /// Metal and the CPU fallback compute the same attention. The Metal sdpa
    /// route is the only one production takes (the vendored flash kernel is
    /// compiled at head dim 128 and cannot serve this architecture's 256).
    #[test]
    fn metal_sdpa_matches_cpu_reference() {
        let Ok(metal) = crate::gguf::metal_device() else {
            return;
        };
        let cpu = Device::Cpu;
        let (n_head, n_kv, hd, hidden) = (4usize, 2usize, 128usize, 16usize);
        let cfg = test_cfg(n_head, n_kv, hd, hidden, 64);
        let w_cpu = raw_weights(n_head, n_kv, hd, hidden, &cpu);

        let rope_cpu = Arc::new(Rope::new(cfg.rope(), 32, &cpu).unwrap());
        let rope_metal = Arc::new(Rope::new(cfg.rope(), 32, &metal).unwrap());
        let b_cpu = build_block(&w_cpu, &cfg, 0, rope_cpu, &cpu, AttnWeights::DequantF32);
        let b_metal = build_block(&w_cpu, &cfg, 0, rope_metal, &metal, AttnWeights::DequantF32);

        let x_cpu = dense(5, hidden, 77, &cpu);
        let x_metal = x_cpu.to_device(&metal).unwrap();

        let mut c_cpu = LayerCache::new(&cfg, 0, 32, &cpu).unwrap();
        let mut c_metal = LayerCache::new(&cfg, 0, 32, &metal).unwrap();
        let m_cpu = b_cpu.prefill_mask(&c_cpu, 5, 0).unwrap();
        let m_metal = b_metal.prefill_mask(&c_metal, 5, 0).unwrap();

        let out_cpu = b_cpu
            .forward(&x_cpu, &mut c_cpu, 0, m_cpu.as_ref(), None)
            .unwrap();
        let out_metal = b_metal
            .forward(&x_metal, &mut c_metal, 0, m_metal.as_ref(), None)
            .unwrap()
            .to_device(&cpu)
            .unwrap();

        assert!(
            max_abs_diff(&out_cpu, &out_metal) < 5e-3,
            "Metal sdpa diverged from the CPU reference by {}",
            max_abs_diff(&out_cpu, &out_metal)
        );
    }

    /// A q8_0-stored projection sends each token count to the kernel that owns
    /// it: the single-token gemv at seq 1, the vendored small-batch mat-vec
    /// across the whole `ops::mv_ext_window`, and the dense f16 plane above
    /// `Q8_DECODE_MAX_SEQ`. Asserted BITWISE against a direct call to the kernel
    /// each range should reach, which is what makes it a routing test rather
    /// than another accuracy one — the three paths differ in the last ulps, so
    /// only the one that actually ran reproduces the output exactly.
    ///
    /// Loaded through `Weights::attn_proj` off a real GGUF, so the plane the
    /// small-batch kernel is handed is the production one: on Metal that is a
    /// page-floored mmap alias bound at a sub-page `base_off`, not an
    /// offset-0 upload.
    #[test]
    fn q8_projection_routes_each_token_count_to_its_kernel() {
        let Ok(dev) = crate::gguf::metal_device() else {
            return;
        };
        // The window's env knobs are read once per process; a run with the
        // kill-switch set has nothing to route and the accuracy tests still
        // cover the kernel itself. `XWEN_LOAD_CLASSIC` skips too: the doc
        // comment's sub-page-offset claim (asserted below) only holds for the
        // mmap alias load this test exists to exercise.
        if crate::ops::attn_dequant()
            || crate::ops::mv_ext_window(2).is_none()
            || crate::gguf::load_classic()
        {
            return;
        }

        // k = 512 satisfies the kernel's pass-width requirement (a multiple of
        // 128) and is a whole number of q8_0 blocks, like every production
        // in_dim; out_dim 64 covers eight full threadgroup row groups.
        let (out_dim, in_dim) = (64usize, 512usize);
        let w_cpu = dense(out_dim, in_dim, 0x5B, &Device::Cpu);
        let qt = QTensor::quantize(&w_cpu, GgmlDType::Q8_0).unwrap();
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path: PathBuf =
            std::env::temp_dir().join(format!("xwen_proj_route_{}_{id}.gguf", std::process::id()));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            candle_core::quantized::gguf_file::write(&mut f, &[], &[("blk.0.attn_k.weight", &qt)])
                .unwrap();
        }
        let loaded = Weights::from_gguf(crate::gguf::open(&path, &dev).unwrap()).pp("blk.0");
        let proj = Proj::load(&loaded, "attn_k", AttnWeights::F16).unwrap();
        let _ = std::fs::remove_file(&path);

        let Proj::DenseF16Q8 { f16, q8 } = &proj else {
            panic!("a q8_0-stored weight must load as the dual-storage projection")
        };
        let plane = &q8.plane;
        // The production shape this test claims to exercise: a page-floored
        // mmap alias whose tensor starts at a sub-page byte offset. An
        // offset-0 upload here would mean the alias load silently fell back
        // to a copy and the offset arithmetic under test never ran.
        assert_ne!(
            plane.base_off, 0,
            "mmap-aliased plane must sit at a sub-page base_off"
        );

        let bits = |t: &Tensor| -> Vec<u32> {
            t.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };

        for t in 1..=Q8_DECODE_MAX_SEQ + 1 {
            let x = dense(t, in_dim, 0x5C + t as u64, &dev);
            let got = bits(&proj.forward(&x).unwrap());
            let gemv = bits(
                &crate::ops::matmul_q8(
                    &plane.buffer,
                    plane.base_off,
                    plane.out_dim,
                    plane.in_dim,
                    &x,
                )
                .unwrap(),
            );
            let want = match crate::ops::mv_ext_window(t) {
                _ if t > Q8_DECODE_MAX_SEQ => bits(&crate::ops::matmul_f16(f16, &x).unwrap()),
                Some(r1ptg) => {
                    let ext = bits(&crate::ops::matmul_mv_ext(plane, &x, r1ptg).unwrap());
                    // The whole assertion rests on the two q8_0 paths being
                    // distinguishable at this shape: they reduce K in different
                    // orders, so they agree only to the last ulps. If they ever
                    // matched bitwise, the check below would hold no matter
                    // which one ran.
                    assert_ne!(
                        ext, gemv,
                        "seq {t}: the small-batch kernel and the gemv agree bitwise, \
                         so this test cannot tell which one ran"
                    );
                    ext
                }
                None => gemv,
            };
            assert_eq!(
                got, want,
                "seq {t} did not reach the kernel that owns its token count"
            );
        }
    }

    /// Per-stage isolation timing for ONE 27B full-attention layer, walked over
    /// the chunk sequence a real prefill issued when this bench was written
    /// (512 tokens at a time, which is still the dense checkpoints' chunk —
    /// `Arch::prefill_chunk_default`; the MoE ones run 2048), with the KV operands laid out the way the cache hands
    /// them to sdpa: a `[1, n_kv, max_ctx, head_dim]` f16 allocation narrowed on
    /// the token axis, so each head carries the full max_ctx stride.
    ///
    /// The projections are flat in position and the mask/sdpa pair is not — the
    /// mask is a materialized `[1, n_head, seq, k_seq]` f16 tensor and sdpa's
    /// work is `seq * k_seq`, so both grow as the prompt advances. Multiply a
    /// per-chunk total by 16 (the 27B's full-attention layer count) and sum over
    /// the chunk list to get attention's share of one prefill.
    ///
    /// `#[ignore]`d — run on a `pgrep`-verified free GPU with:
    ///   cargo test --release -p xwen attn_prefill_stage_timing -- --ignored --nocapture
    /// `XWEN_BENCH_WARMUP` / `XWEN_BENCH_ITERS` override the loop counts.
    #[test]
    #[ignore = "perf bench"]
    fn attn_prefill_stage_timing() {
        use std::time::Instant;

        const HIDDEN: usize = 5120;
        const N_HEAD: usize = 24;
        const N_KV: usize = 4;
        const HD: usize = 256;
        const MAX_CTX: usize = 8192;
        /// The dense 27B's production prefill chunk
        /// (`Arch::prefill_chunk_default`).
        const CHUNK: usize = 512;

        let device = crate::gguf::metal_device().unwrap();
        let Device::Metal(mdev) = &device else {
            unreachable!("metal_device() returned a non-Metal device")
        };
        let get = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let (warm, iters) = (get("XWEN_BENCH_WARMUP", 3), get("XWEN_BENCH_ITERS", 10));

        // Waiting on the device rather than reading back: these stages produce
        // hundreds of megabytes and a readback would time the memcpy.
        let bench = |f: &mut dyn FnMut()| -> f64 {
            for _ in 0..warm {
                f();
                mdev.wait_until_completed().unwrap();
            }
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = Instant::now();
                f();
                mdev.wait_until_completed().unwrap();
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            times[iters / 2..].iter().sum::<f64>() / (iters - iters / 2) as f64
        };

        let f16w = |rows: usize, cols: usize, seed: u64| {
            dense(rows, cols, seed, &device)
                .to_dtype(DType::F16)
                .unwrap()
        };
        let w_qg = f16w(N_HEAD * 2 * HD, HIDDEN, 11);
        let w_k = f16w(N_KV * HD, HIDDEN, 12);
        let w_v = f16w(N_KV * HD, HIDDEN, 13);
        let w_o = f16w(HIDDEN, N_HEAD * HD, 14);
        let qn = candle_nn::RmsNorm::new(
            Tensor::from_vec(vec![1.0f32; HD], HD, &device).unwrap(),
            1e-6,
        );

        // The cache's own allocation: one contiguous [n_kv, max_ctx, hd] f16
        // block per tensor, which sdpa reads through a narrowed token axis.
        let kv_alloc = |seed: u64| {
            Tensor::from_vec(
                seeded(N_KV * MAX_CTX * HD, seed),
                (1, N_KV, MAX_CTX, HD),
                &device,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
        };
        let k_alloc = kv_alloc(21);
        let v_alloc = kv_alloc(22);

        let scale = 1.0f32 / (HD as f32).sqrt();

        // The chunk sequences the two bench fixtures produce.
        let chunks_for = |total: usize| -> Vec<(usize, usize)> {
            let mut v = Vec::new();
            let mut pos = 0;
            while pos < total {
                let seq = CHUNK.min(total - pos);
                v.push((seq, pos));
                pos += seq;
            }
            v
        };

        for &total in &[880usize, 3851] {
            let mut sums = [0f64; 6];
            let mut amort = 0f64;
            eprintln!("--- 27B attention, prefill of {total} tokens, per chunk (ms) ---");
            for (seq, pos) in chunks_for(total) {
                let k_seq = pos + seq;
                let x = dense(seq, HIDDEN, 0x51 + seq as u64, &device);
                let k_view = k_alloc.narrow(2, 0, k_seq).unwrap();
                let v_view = v_alloc.narrow(2, 0, k_seq).unwrap();
                let q16 = dense(N_HEAD * seq, HD, 0x52, &device)
                    .reshape((1, N_HEAD, seq, HD))
                    .unwrap()
                    .to_dtype(DType::F16)
                    .unwrap();
                let mask = PrefillMask::build(MaskKind::Full, N_HEAD, seq, pos, &device)
                    .unwrap()
                    .unwrap();

                let t_mask = bench(&mut || {
                    PrefillMask::build(MaskKind::Full, N_HEAD, seq, pos, &device).unwrap();
                });
                let t_sdpa = bench(&mut || {
                    candle_nn::ops::sdpa(
                        &q16,
                        &k_view,
                        &v_view,
                        Some(&mask.sdpa),
                        false,
                        scale,
                        1.0,
                    )
                    .unwrap();
                });
                let t_qg = bench(&mut || {
                    crate::ops::matmul_f16(&w_qg, &x).unwrap();
                });
                let t_kv = bench(&mut || {
                    crate::ops::matmul_f16(&w_k, &x).unwrap();
                    crate::ops::matmul_f16(&w_v, &x).unwrap();
                });
                let o_in = dense(seq, N_HEAD * HD, 0x53, &device);
                let t_o = bench(&mut || {
                    crate::ops::matmul_f16(&w_o, &o_in).unwrap();
                });

                // Everything between the projections and sdpa: the strided
                // q/gate split, QK-norm, the three permute-copies, the q cast,
                // the sdpa output widening, and the eager sigmoid gate.
                let t_glue = bench(&mut || {
                    let qg = crate::ops::matmul_f16(&w_qg, &x)
                        .unwrap()
                        .reshape((seq, N_HEAD, 2, HD))
                        .unwrap();
                    let q = qg
                        .narrow(2, 0, 1)
                        .unwrap()
                        .contiguous()
                        .unwrap()
                        .reshape((seq, N_HEAD, HD))
                        .unwrap();
                    let gate = qg
                        .narrow(2, 1, 1)
                        .unwrap()
                        .contiguous()
                        .unwrap()
                        .reshape((seq, N_HEAD, HD))
                        .unwrap();
                    let q = qn.forward(&q).unwrap();
                    let q = crate::ops::permute_01(&q).unwrap();
                    let q = crate::ops::cast_f16(&q).unwrap();
                    let attn = crate::ops::cast_f32(&q).unwrap();
                    let gate = crate::ops::permute_01(&gate).unwrap();
                    (&attn * &candle_nn::ops::sigmoid(&gate).unwrap()).unwrap();
                });
                // The glue timing re-ran the qg projection to get a real input;
                // charge only the remainder.
                let t_glue = t_glue - t_qg;

                // A real forward issues all 16 attention layers back to back, so
                // the per-call commit-and-wait above is overhead the model never
                // pays. Batch the whole per-layer sequence and sync once; this is
                // the rate to multiply by the layer count.
                const BATCH: usize = 8;
                let t_amort = bench(&mut || {
                    let mut keep = Vec::with_capacity(BATCH);
                    for _ in 0..BATCH {
                        let qg = crate::ops::matmul_f16(&w_qg, &x)
                            .unwrap()
                            .reshape((seq, N_HEAD, 2, HD))
                            .unwrap();
                        let q = qg
                            .narrow(2, 0, 1)
                            .unwrap()
                            .contiguous()
                            .unwrap()
                            .reshape((seq, N_HEAD, HD))
                            .unwrap();
                        let gate = qg
                            .narrow(2, 1, 1)
                            .unwrap()
                            .contiguous()
                            .unwrap()
                            .reshape((seq, N_HEAD, HD))
                            .unwrap();
                        crate::ops::matmul_f16(&w_k, &x).unwrap();
                        crate::ops::matmul_f16(&w_v, &x).unwrap();
                        let q = qn.forward(&q).unwrap();
                        let q = crate::ops::permute_01(&q).unwrap();
                        let q16 = crate::ops::cast_f16(&q).unwrap();
                        let a = candle_nn::ops::sdpa(
                            &q16.reshape((1, N_HEAD, seq, HD)).unwrap(),
                            &k_view,
                            &v_view,
                            Some(&mask.sdpa),
                            false,
                            scale,
                            1.0,
                        )
                        .unwrap();
                        let a = crate::ops::cast_f32(&a.squeeze(0).unwrap()).unwrap();
                        let gate = crate::ops::permute_01(&gate).unwrap();
                        let a = (&a * &candle_nn::ops::sigmoid(&gate).unwrap()).unwrap();
                        let a = crate::ops::permute_01(&a)
                            .unwrap()
                            .reshape((seq, N_HEAD * HD))
                            .unwrap();
                        keep.push(crate::ops::matmul_f16(&w_o, &a).unwrap());
                    }
                }) / BATCH as f64;
                amort += t_amort;

                let row = [t_qg, t_kv, t_o, t_mask, t_sdpa, t_glue];
                for (s, r) in sums.iter_mut().zip(row) {
                    *s += r;
                }
                eprintln!(
                    "  seq={seq:<4} pos={pos:<5} qg {t_qg:6.3} | kv {t_kv:6.3} | o {t_o:6.3} \
                     | mask {t_mask:6.3} | sdpa {t_sdpa:6.3} | glue {t_glue:6.3} \
                     || whole-layer amortized {t_amort:6.3}"
                );
            }
            // The mask is built once per chunk in model.rs `run_stack` and shared
            // by every attention layer, so its column is a whole-prefill total
            // already; the other five are per layer and scale by 16.
            let per_layer: f64 = sums[0] + sums[1] + sums[2] + sums[4] + sums[5];
            eprintln!(
                "  TOTAL  qg {:.2} | kv {:.2} | o {:.2} | sdpa {:.2} | glue {:.2} \
                 => {per_layer:.2} ms/layer x16 = {:.1} ms; mask {:.2} ms (hoisted, once per chunk)",
                sums[0],
                sums[1],
                sums[2],
                sums[4],
                sums[5],
                per_layer * 16.0,
                sums[3],
            );
            eprintln!(
                "  AMORTIZED whole layer {amort:.2} ms/layer x16 = {:.1} ms (+ mask {:.2} ms)",
                amort * 16.0,
                sums[3]
            );
        }
    }

    // ---- the QSA overlay (D16) ----

    /// A `qwen4exp`-shaped attention block: head dim 256 (so the vendored flash
    /// kernel, compiled at 128, is not a route) and GQA, on Metal.
    use crate::gguf::metal_device;

    fn qsa_block(dev: &Device) -> (AttnBlock, XwenConfig, RawWeights) {
        let (n_head, n_kv, hd, hidden) = (4usize, 2usize, 256usize, 512usize);
        let cfg = test_cfg(n_head, n_kv, hd, hidden, 64);
        let rope = Arc::new(Rope::new(cfg.rope(), 256, dev).unwrap());
        let w = raw_weights(n_head, n_kv, hd, hidden, dev);
        let block = build_block(&w, &cfg, 0, rope, dev, AttnWeights::DequantF32);
        (block, cfg, w)
    }

    /// `QsaSelection::Dense` is not an approximation of the unoverlaid path —
    /// it IS that path, which is what makes a below-budget QSA layer a free
    /// correctness check against a run with no indexer at all.
    #[test]
    fn dense_selection_is_bit_identical_to_no_overlay() {
        let dev = metal_device().unwrap();
        let (block, cfg, _) = qsa_block(&dev);
        let x = dense(64, cfg.hidden, 91, &dev);

        let run = |qsa: Option<&QsaSelection>| {
            let mut cache = LayerCache::new(&cfg, 0, 128, &dev).unwrap();
            let mask = block.prefill_mask(&cache, 64, 0).unwrap();
            block
                .forward(&x, &mut cache, 0, mask.as_ref(), qsa)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        assert_eq!(run(None), run(Some(&QsaSelection::Dense)));
    }

    /// A decode step whose `Rows` names EVERY cached position must equal the
    /// unoverlaid decode: the gather packs the cache's strided per-head views
    /// into contiguous planes, and packing must not change the answer.
    ///
    /// The second half is the subset half, and it pins WHICH rows were used:
    /// the output has to equal the naive reference restricted to exactly those
    /// key positions, so an off-by-one in the gather, a reversed index order or
    /// a selection that quietly kept the whole prefix all fail. The "moved"
    /// assertion is kept beside it as the falsifiability guard — without it, an
    /// equality that happened to hold because the subset barely differs from the
    /// full prefix would read as a pass.
    #[test]
    fn decode_rows_gather_selects_the_named_positions() {
        let dev = metal_device().unwrap();
        let (block, cfg, w) = qsa_block(&dev);
        // The same rope `qsa_block` handed the block, rebuilt for the reference.
        let rope = Rope::new(cfg.rope(), 256, &dev).unwrap();
        let prefill = dense(64, cfg.hidden, 92, &dev);
        let step = dense(1, cfg.hidden, 93, &dev);

        let run = |rows: Option<Vec<u32>>| {
            let mut cache = LayerCache::new(&cfg, 0, 128, &dev).unwrap();
            let mask = block.prefill_mask(&cache, 64, 0).unwrap();
            block
                .forward(&prefill, &mut cache, 0, mask.as_ref(), None)
                .unwrap();
            let sel = rows
                .map(|r| QsaSelection::Rows(Tensor::from_vec(r.clone(), r.len(), &dev).unwrap()));
            block
                .forward(&step, &mut cache, 64, None, sel.as_ref())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };

        let plain = run(None);
        let all = run(Some((0..65).collect()));
        let diff = plain
            .iter()
            .zip(&all)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            diff < 1e-5,
            "gathering every row changed the output by {diff}"
        );

        // Half the prefix plus the query's own token: a different attention
        // distribution, so a different answer.
        let rows: Vec<u32> = (0..32).chain(std::iter::once(64)).collect();
        let subset = run(Some(rows.clone()));
        let moved = plain
            .iter()
            .zip(&subset)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            moved > 1e-4,
            "restricting the rows to half the prefix left the output unchanged ({moved}): \
             the gather is not reaching sdpa"
        );

        // And it moved to the RIGHT place: the naive f32 reference over the
        // whole 65-token sequence, with the final query row's visibility cut
        // down to exactly `rows`.
        let x = Tensor::cat(&[&prefill, &step], 0).unwrap();
        let want = naive_forward(
            &w,
            &rope,
            &x,
            cfg.n_head[0],
            cfg.n_kv_head,
            cfg.head_dim,
            Some(&rows),
        );
        let want: Vec<f32> = want
            .narrow(0, 64, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let err = subset
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // RELATIVE, unlike the 1e-5 above. That comparison differences two runs
        // of the same kernel chain, which agree to well under one ulp of these
        // values; this one differences the Metal sdpa kernel against a
        // broadcast_matmul/softmax chain reading the same f16 KV rows, and this
        // block's random weights put the o_proj output around 1e4, where f32's
        // own relative step is already ~1e-3 in absolute terms. 1e-4 relative
        // sits 4x above the ~2.5e-5 the two paths actually differ by, and 5x
        // below the ~4.9e-4 that the smallest interesting defect produces —
        // measured by sliding the reference's key window one position, i.e. an
        // off-by-one in the gather. Coarser failures (the selection quietly
        // keeping the whole prefix) are order-1 relative and nowhere near this.
        let mag = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(
            err < 1e-4 * mag,
            "the row-restricted decode diverged from the naive reference by {err} \
             on values of magnitude {mag}"
        );
    }

    /// A prefill `Mask` replaces the causal mask rather than composing with it,
    /// so an all-visible-below-the-diagonal QSA mask reproduces the ordinary
    /// causal prefill.
    #[test]
    fn prefill_mask_overlay_reproduces_causal_when_nothing_is_masked() {
        let dev = metal_device().unwrap();
        let (block, cfg, _) = qsa_block(&dev);
        let seq = 64usize;
        let x = dense(seq, cfg.hidden, 94, &dev);

        let mut causal = vec![f32::NEG_INFINITY; seq * seq];
        for (q, row) in causal.chunks_mut(seq).enumerate() {
            for slot in row.iter_mut().take(q + 1) {
                *slot = 0.0;
            }
        }
        let overlay = QsaSelection::Mask(Tensor::from_vec(causal, (seq, seq), &dev).unwrap());

        let run = |qsa: Option<&QsaSelection>| {
            let mut cache = LayerCache::new(&cfg, 0, 128, &dev).unwrap();
            let mask = block.prefill_mask(&cache, seq, 0).unwrap();
            block
                .forward(&x, &mut cache, 0, mask.as_ref(), qsa)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let plain = run(None);
        let overlaid = run(Some(&overlay));
        let diff = plain
            .iter()
            .zip(&overlaid)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            diff < 1e-5,
            "a fully causal overlay moved the output by {diff}"
        );
    }

    /// Each overlay arm is refused outside the shape it is for. Both mistakes
    /// are otherwise SILENT: a `Mask` at seq 1 reaches candle's vector sdpa,
    /// which ignores the mask argument and returns dense attention over the
    /// whole prefix; a `Rows` at seq > 1 gathers one selection for every query
    /// and drops the causal mask with it. Neither produces a wrong shape, so
    /// nothing downstream would notice.
    #[test]
    fn an_overlay_used_at_the_wrong_shape_is_refused() {
        let dev = metal_device().unwrap();
        let (block, cfg, _) = qsa_block(&dev);
        let seq = 8usize;
        let prefill = dense(seq, cfg.hidden, 95, &dev);
        let step = dense(1, cfg.hidden, 96, &dev);

        // A prefill mask handed to a decode step.
        let mut cache = LayerCache::new(&cfg, 0, 128, &dev).unwrap();
        let mask = block.prefill_mask(&cache, seq, 0).unwrap();
        block
            .forward(&prefill, &mut cache, 0, mask.as_ref(), None)
            .unwrap();
        let one_row = QsaSelection::Mask(Tensor::zeros((1, seq + 1), DType::F32, &dev).unwrap());
        let err = block
            .forward(&step, &mut cache, seq, None, Some(&one_row))
            .unwrap_err()
            .to_string();
        assert!(err.contains("prefill overlay"), "{err}");

        // A decode gather handed a whole prefill chunk.
        let mut cache = LayerCache::new(&cfg, 0, 128, &dev).unwrap();
        let mask = block.prefill_mask(&cache, seq, 0).unwrap();
        let rows = QsaSelection::Rows(Tensor::from_vec(vec![0u32, 1, 2], 3, &dev).unwrap());
        let err = block
            .forward(&prefill, &mut cache, 0, mask.as_ref(), Some(&rows))
            .unwrap_err()
            .to_string();
        assert!(err.contains("decode overlay"), "{err}");

        // A decode gather WITH a mask: the gather route runs maskless, so the
        // mask would be dropped without a word.
        let mut cache = LayerCache::new(&cfg, 0, 128, &dev).unwrap();
        block
            .forward(&prefill, &mut cache, 0, mask.as_ref(), None)
            .unwrap();
        let decode_mask = PrefillMask::from_raw(
            Tensor::zeros((1, seq + 1), DType::F32, &dev).unwrap(),
            cfg.n_head(0),
        )
        .unwrap();
        let err = block
            .forward(&step, &mut cache, seq, Some(&decode_mask), Some(&rows))
            .unwrap_err()
            .to_string();
        assert!(err.contains("silently dropped"), "{err}");
    }

    /// The sdpa mask is ONE head-uniform plane broadcast across the head axis,
    /// not `n_head` copies of it. Pinned at the qwen4exp geometry (24 query
    /// heads, 2 KV, head dim 256) and a QSA-sized 2200-token prefill, where the
    /// difference is 9.7 MB against 232 MB per layer.
    ///
    /// The property under test is candle's, not ours: its Metal sdpa forwards
    /// the mask's strides to the kernel rather than assuming the mask
    /// contiguous, so a stride-0 head axis feeds every head the same row. If a
    /// future candle bump stops honoring that, this test says so — either by
    /// failing the shape check outright or by disagreeing with the materialized
    /// mask it is compared against.
    #[test]
    fn the_sdpa_mask_is_one_plane_broadcast_across_the_heads() {
        let dev = metal_device().unwrap();
        let (n_head, n_kv, hd, seq) = (24usize, 2usize, 256usize, 2200usize);

        let mut causal = vec![f32::NEG_INFINITY; seq * seq];
        for (q, row) in causal.chunks_mut(seq).enumerate() {
            // A QSA-shaped selection: the query's own token plus a strided
            // sample of its prefix, so most of the plane really is masked.
            for (k, slot) in row.iter_mut().enumerate().take(q + 1) {
                if k == q || k % 4 == 0 {
                    *slot = 0.0;
                }
            }
        }
        let raw = Tensor::from_vec(causal, (seq, seq), &dev).unwrap();
        let mask = PrefillMask::from_raw(raw, n_head).unwrap();

        // One plane's worth of storage, viewed as n_head.
        assert_eq!(mask.sdpa.dims(), [1, n_head, seq, seq]);
        assert_eq!(
            mask.sdpa.stride()[1],
            0,
            "the head axis must carry no bytes"
        );

        let q = dense(n_head * seq, hd, 97, &dev)
            .reshape((1, n_head, seq, hd))
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let k = dense(n_kv * seq, hd, 98, &dev)
            .reshape((1, n_kv, seq, hd))
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let v = dense(n_kv * seq, hd, 99, &dev)
            .reshape((1, n_kv, seq, hd))
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let scale = 1.0f32 / (hd as f32).sqrt();

        let run = |m: &Tensor| {
            candle_nn::ops::sdpa(&q, &k, &v, Some(m), false, scale, 1.0)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let materialized = mask.sdpa.contiguous().unwrap();
        let diff = run(&mask.sdpa)
            .iter()
            .zip(&run(&materialized))
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            diff < 1e-5,
            "the broadcast mask disagreed with the materialized one by {diff}"
        );
    }
}
