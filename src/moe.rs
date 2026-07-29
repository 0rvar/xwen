use std::sync::Arc;

use anyhow::Result;
use candle_core::quantized::QTensor;
use candle_core::{DType, Tensor};
use candle_nn::ops::{sigmoid, silu};

use crate::config::XwenConfig;
use crate::gguf::{ExpertStack, QLinear, QuantPlane, Weights};
use crate::ops::{ExpertRunner, mm_id, mv_id};

/// F16's smallest positive normal, the denominator floor llama.cpp clamps the
/// routing-weight sum to before normalizing (`ffn_moe_weights_sum_clamped`).
pub(crate) const WEIGHTS_SUM_FLOOR: f64 = 6.103_515_625e-5;

/// Runs the selected experts' SwiGLU FFNs.
/// x: [seq, hidden] f32; ids: [seq, top_k] u32 on-device; weights: [seq, top_k] f32.
/// Returns [seq, hidden] f32 (already weight-combined).
pub trait ExpertFfn: Send {
    fn forward(&self, x: &Tensor, ids: &Tensor, weights: &Tensor) -> Result<Tensor>;

    /// The per-slot down-projection output `[seq, top_k, hidden]` BEFORE the
    /// routing-weight combine, for callers that fold the combine into a larger
    /// fused pass. `None` means this runner cannot hand one over for these
    /// operands and the caller must use `forward` — either because it never
    /// materializes an uncombined result (the reference oracle scatters each
    /// expert's rows as it goes) or because the projection carries a rescale the
    /// plain combine does not undo. Implementations returning `None` must do so
    /// before spending any work.
    fn project(&self, _x: &Tensor, _ids: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }
}

/// One MoE FFN block — every layer of `qwen35moe`. Routing math: a SOFTMAX over
/// all 256 experts in f32, then top-k by that probability, then a sum
/// renormalization of the selected weights with the f16 floor. There is no
/// selection bias and no expert weight scale; a router that carried either
/// would rank and weight the experts differently. The shared expert runs in
/// parallel and is added in, scaled by its own scalar sigmoid gate.
pub struct MoeBlock {
    /// Router weight, pre-transposed to `[hidden, n_expert]` and made contiguous
    /// once at load so the per-forward routing matmul needs no transpose/copy.
    router_t: Tensor,
    experts: Box<dyn ExpertFfn>,
    shared: SharedExpert,
    n_expert: usize,
    n_expert_used: usize,
}

impl MoeBlock {
    /// `w` positioned at the block prefix. `runner` picks Fused vs Reference experts.
    pub fn new(w: &Weights, cfg: &XwenConfig, runner: ExpertRunner) -> Result<Self> {
        let router_t = w.dense_f32("ffn_gate_inp")?.t()?.contiguous()?;
        let n_expert = router_t.dim(1)?;
        let experts: Box<dyn ExpertFfn> = match runner {
            ExpertRunner::Reference => Box::new(ReferenceExperts::new(w)?),
            ExpertRunner::Fused => Box::new(FusedExperts::new(w)?),
        };
        let shared = SharedExpert::new(w)?;
        Ok(Self {
            router_t,
            experts,
            shared,
            n_expert,
            n_expert_used: cfg.n_expert_used,
        })
    }

    /// Whether this block runs the fused glue kernels (`ops::moe_router`,
    /// `ops::moe_epilogue`, and the shared expert's `ops::silu_mul`) rather than
    /// the candle chains. Needs a Metal device, the kill-switch unset, and a
    /// routing geometry the router kernel's threadgroup arrays cover — the
    /// bounds are the kernel's, so a checkpoint outside them falls back rather
    /// than failing.
    fn glue_fused(&self, x: &Tensor) -> bool {
        !crate::ops::moe_glue_classic()
            && x.device().is_metal()
            && crate::ops::moe_router_supported(self.n_expert, self.n_expert_used)
    }

    /// Returns (ids [seq, top_k] u32 on-device, weights [seq, top_k] f32).
    pub fn route(&self, x_normed: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = route_logits(&self.router_t, x_normed)?;
        if self.glue_fused(x_normed) {
            // One kernel for softmax + arg-sort + gather + sum + clamp + divide,
            // bit-identical to the candle chain below (ops::moe_glue).
            return crate::ops::moe_router(&logits, self.n_expert_used, WEIGHTS_SUM_FLOOR as f32);
        }
        route_from_logits(&logits, self.n_expert_used)
    }

    pub fn forward(&self, x_normed: &Tensor) -> Result<Tensor> {
        let (ids, weights) = self.route(x_normed)?;

        // Fused tail: the routed weighted combine, the shared expert's sigmoid
        // gate, its multiply and the routed+shared add in one pass over the
        // uncombined down projection. Only when the expert runner can hand that
        // projection over — the reference oracle never materializes one, and the
        // f16-tile prefill variants carry an L2 rescale this epilogue does not
        // undo — otherwise the classic chain below runs unchanged.
        if self.glue_fused(x_normed)
            && let Some(down) = self.experts.project(x_normed, &ids)?
        {
            let shexp = self.shared.swiglu_out(x_normed)?;
            let gate = self.shared.gate_logits(x_normed)?;
            return crate::ops::moe_epilogue(&down, &weights, &shexp, &gate);
        }

        let routed = self.experts.forward(x_normed, &ids, &weights)?;
        let shared = self.shared.forward(x_normed)?;
        Ok((routed + shared)?)
    }
}

/// Router projection: `logits[t, e] = <x[t], router[e]>`, computed in f32.
/// `router_t` is the `[hidden, n_expert]` transpose of the ggml `ffn_gate_inp`
/// weight, precomputed once at load; result is `[seq, n_expert]`.
fn route_logits(router_t: &Tensor, x: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    Ok(x.matmul(router_t)?)
}

/// The routing decision from router logits, kept separate from the router matmul
/// so it can be exercised with hand-built logits.
///
/// The softmax runs over ALL experts BEFORE the top-k, so a selected expert's
/// weight depends on the whole distribution and not only on its k peers; the
/// gathered weights are then renormalized to sum to 1, with the denominator
/// floored at f16's smallest normal.
fn route_from_logits(logits: &Tensor, n_expert_used: usize) -> Result<(Tensor, Tensor)> {
    let probs = candle_nn::ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?; // [seq, n_expert]

    // Descending arg-sort is stable on CPU (std slice sort), so ties resolve to
    // the lower expert index first — matching ggml_argsort_top_k.
    let order = probs.contiguous()?.arg_sort_last_dim(false)?;
    let ids = order.narrow(1, 0, n_expert_used)?.contiguous()?; // [seq, top_k] u32

    let weights = probs.gather(&ids, 1)?;
    let sum = weights
        .sum_keepdim(1)?
        .clamp(WEIGHTS_SUM_FLOOR as f32, f32::INFINITY)?;
    Ok((ids, weights.broadcast_div(&sum)?))
}

/// Correctness oracle: keeps each routed expert's gate/up/down as an individual
/// quantized weight and evaluates only the experts a token actually selected.
/// Reads the selection ids back to the CPU to bucket tokens per expert — slow,
/// but exercises the same dequantized weights the fused kernels consume.
struct ReferenceExperts {
    gate: Vec<Arc<QTensor>>, // each [expert_ff, hidden]
    up: Vec<Arc<QTensor>>,   // each [expert_ff, hidden]
    down: Vec<Arc<QTensor>>, // each [hidden, expert_ff]
}

impl ReferenceExperts {
    fn new(w: &Weights) -> Result<Self> {
        Ok(Self {
            gate: w.expert_qtensors("ffn_gate_exps")?,
            up: w.expert_qtensors("ffn_up_exps")?,
            down: w.expert_qtensors("ffn_down_exps")?,
        })
    }
}

impl ExpertFfn for ReferenceExperts {
    fn forward(&self, x: &Tensor, ids: &Tensor, weights: &Tensor) -> Result<Tensor> {
        let (seq, hidden) = x.dims2()?;
        let top_k = ids.dim(1)?;
        let device = x.device();
        let x = x.to_dtype(DType::F32)?;

        let ids_v: Vec<u32> = ids.flatten_all()?.to_vec1()?;
        let w_v: Vec<f32> = weights.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;

        // Bucket (token, weight) pairs per expert so each selected expert runs
        // one batched matmul over its assigned tokens.
        let n_expert = self.gate.len();
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); n_expert];
        let mut wts: Vec<Vec<f32>> = vec![Vec::new(); n_expert];
        for t in 0..seq {
            for k in 0..top_k {
                let e = ids_v[t * top_k + k] as usize;
                rows[e].push(t as u32);
                wts[e].push(w_v[t * top_k + k]);
            }
        }

        let mut out = Tensor::zeros((seq, hidden), DType::F32, device)?;
        for e in 0..n_expert {
            if rows[e].is_empty() {
                continue;
            }
            let m = rows[e].len();
            let idx = Tensor::from_vec(rows[e].clone(), m, device)?;
            let xe = x.index_select(&idx, 0)?; // [m, hidden]

            let gate_w = self.gate[e].dequantize(device)?.to_dtype(DType::F32)?;
            let up_w = self.up[e].dequantize(device)?.to_dtype(DType::F32)?;
            let down_w = self.down[e].dequantize(device)?.to_dtype(DType::F32)?;

            let g = silu(&xe.matmul(&gate_w.t()?)?)?; // [m, expert_ff]
            let u = xe.matmul(&up_w.t()?)?; // [m, expert_ff]
            let h = (&g * &u)?;
            let d = h.matmul(&down_w.t()?)?; // [m, hidden]

            let we = Tensor::from_vec(wts[e].clone(), (m, 1), device)?;
            let d = d.broadcast_mul(&we)?;
            out = out.index_add(&idx, &d, 0)?;
        }
        Ok(out)
    }
}

/// Fused path: the routed experts stay stacked in their quantized layout and the
/// `ops::mv_id` gather-matvec kernel indexes them by id on-device (no readback).
/// Mirrors the fork's build_moe_ffn expert evaluation, including the per-column
/// L2 rescale that keeps the down-projection input inside f16 range.
struct FusedExperts {
    gate: ExpertStack,
    up: ExpertStack,
    down: ExpertStack,
}

impl FusedExperts {
    fn new(w: &Weights) -> Result<Self> {
        let gate = w.expert_stack("ffn_gate_exps")?;
        let up = w.expert_stack("ffn_up_exps")?;
        let down = w.expert_stack("ffn_down_exps")?;
        // The shared map0 pass is built once from `gate.n_expert` and reused for
        // all three projections; a stack disagreeing on n_expert would read the
        // ids-map at the wrong offset. (run_mm_shared re-validates per dispatch,
        // but a mismatch here is a malformed checkpoint — fail at load.)
        anyhow::ensure!(
            gate.n_expert == up.n_expert && gate.n_expert == down.n_expert,
            "MoE expert stacks disagree on n_expert: gate={}, up={}, down={}",
            gate.n_expert,
            up.n_expert,
            down.n_expert
        );
        Ok(Self { gate, up, down })
    }
}

impl FusedExperts {
    /// Whether these operands take the two-pass token-grouped prefill matmul
    /// rather than the per-token matvec, and whether that choice stages the
    /// down-projection activation as f16 (so the L2 rescale guard is required).
    /// Split out of the projection so a caller can learn the answer without
    /// spending any work.
    fn use_mm(&self, seq: usize, top_k: usize) -> bool {
        let variant = crate::ops::mm_id_variant();
        let mm_supported = mm_id::supported(self.gate.dtype, top_k, variant)
            && mm_id::supported(self.up.dtype, top_k, variant)
            && mm_id::supported(self.down.dtype, top_k, variant);
        seq >= crate::ops::mm_id_min_seq() && !crate::ops::no_mm_id() && mm_supported
    }

    fn needs_rescale(&self, seq: usize, top_k: usize) -> bool {
        self.use_mm(seq, top_k) && crate::ops::mm_id_variant().casts_activation_f16()
    }

    /// The routed experts' down projection, `[seq, top_k, hidden]`, before any
    /// weighted combine — plus the per-column L2 rescale factor a combine must
    /// undo when the prefill kernels staged the activation as f16.
    fn project_inner(&self, x: &Tensor, ids: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let (seq, hidden) = x.dims2()?;
        let top_k = ids.dim(1)?;
        let x = x.to_dtype(DType::F32)?;

        // Prefill (many tokens) uses the two-pass token-grouped matmul so each
        // expert's rows are dequantized once for all the tokens routed to it;
        // decode (batch=1, and short chunks) stays on the per-token matvec, which
        // wins at low token counts. MM_ID_MIN_SEQ is ggml's mm_id break-even point.
        // `XWEN_NO_MM_ID` forces mv_id everywhere as a fallback; mm_id is also
        // skipped (mv_id fallback) for any dtype/top_k/variant the vendored kernels
        // are not instantiated for, so other checkpoints still run. Note mm_id's
        // tiled f32 accumulation drifts a little further from the per-row f32
        // reference oracle than mv_id does (fork-equivalent tiled behavior; see
        // docs/parity.md §3b), so mv_id is the reference for the strict gate.
        let use_mm = self.use_mm(seq, top_k);

        // One map0 pass (per-expert token-slot map) shared by gate/up/down: the
        // three projections route by the SAME ids, so their maps are byte-identical.
        // Built once here and held for the whole forward — the down projection reads
        // it after silu/rescale — replacing two of the three per-projection passes.
        let map0 = if use_mm {
            Some(mm_id::prepare_map0(self.gate.n_expert, ids)?)
        } else {
            None
        };
        let matmul = |stack: &ExpertStack, x: &Tensor, ids: &Tensor| -> Result<Tensor> {
            match &map0 {
                Some(scratch) => mm_id::mul_mm_id_shared(stack, x, ids, scratch),
                None => mv_id::mul_mv_id(stack, x, ids),
            }
        };

        // gate/up share one activation per token: x_per_row = 1.
        let x_g = x.reshape((seq, 1, hidden))?;

        // OPT-IN (XWEN_MOE_DUAL): one dual-weight gather computes both
        // projections and their SwiGLU activation, replacing three dispatches
        // with one and never writing the two intermediates. Bit-identical to the
        // split chain below, but MEASURED SLOWER on this device — the gather is
        // bandwidth-bound and the extra live accumulators cost more occupancy
        // than the two saved dispatches buy back (docs/decisions.md "The
        // dual-weight expert gather"). Only reachable on the matvec path (the
        // prefill mm_id kernels have no dual form) for a q4_K gate/up pair.
        let dual = crate::ops::moe_dual()
            && !use_mm
            && !crate::ops::moe_glue_classic()
            && !crate::ops::act_classic()
            && !crate::ops::mv_classic()
            && mv_id::mul_mv_id_dual_supported(&self.gate, &self.up);
        if dual {
            let act = mv_id::mul_mv_id_dual(&self.gate, &self.up, &x_g, ids)?;
            return Ok((matmul(&self.down, &act, ids)?, None));
        }

        let gate = matmul(&self.gate, &x_g, ids)?; // [seq, top_k, expert_ff]
        let up = matmul(&self.up, &x_g, ids)?; // [seq, top_k, expert_ff]

        // The down projection consumes the SwiGLU activation. Some mm_id variants
        // stage that activation as f16, where a large value would overflow, and
        // need the L2 rescale guard; others read it as f32 and do not:
        //   - decode / mv_id: candle's kernel_mul_mv_q{4,6}_K_f32_impl reads src1
        //     as f32 and accumulates in f32 (quantized.metal:4889/4930, 5188/5225);
        //   - prefill / mm_id-hp (classic f32 tiles): stages src1 as float — no cast;
        //   - prefill / mm_id tensor default + classic-f16: stage src1 as half —
        //     f16 cast, so the rescale is required.
        let needs_rescale = self.needs_rescale(seq, top_k);

        // silu(gate)*up fused into one vendored pass (`ops::silu_mul`), which is
        // BIT-IDENTICAL to candle's `silu(gate) * up` chain by construction — the
        // silu expression and the multiply keep candle's exact per-op rounding
        // boundaries (see silu_mul.metal). Bit-identity is what makes this safe on
        // every parity tier: an earlier NON-identical fused activation was abandoned
        // because ~1e-6 differences cascaded through the MoE router (near-tie expert
        // flips) to ~1e-3 final-logit divergence, but a bitwise match cannot cascade.
        // `XWEN_ACT_CLASSIC` reverts to the candle two-op chain (the reference
        // oracle's ReferenceExperts also runs that chain). See docs/parity.md §3b.
        let act = if crate::ops::act_classic() {
            (silu(&gate)? * up)?
        } else {
            crate::ops::silu_mul(&gate, &up)?
        }; // [seq, top_k, expert_ff]

        if needs_rescale {
            // Per-column L2 rescale keeps the f16-tile down cast in range; the
            // factor divides back out in the combine (a per-column identity). 32768
            // (f16's safe headroom) is only meaningful on this f16-tile branch.
            let f16_safe = 32768.0_f64;
            let col_l2 = act
                .sqr()?
                .sum_keepdim(2)?
                .sqrt()?
                .clamp(1e-8_f32, 1e30_f32)?; // [seq, top_k, 1]
            let act_s = (&act * f16_safe)?.broadcast_div(&col_l2)?;
            let down = matmul(&self.down, &act_s, ids)?; // [seq, top_k, hidden]
            return Ok((down, Some(col_l2)));
        }

        // Default: f32 down projection (mv_id or mm_id-hp) — no f16 cast, so the
        // activation feeds the down matmul directly, no rescale needed.
        Ok((matmul(&self.down, &act, ids)?, None))
    }
}

impl ExpertFfn for FusedExperts {
    fn forward(&self, x: &Tensor, ids: &Tensor, weights: &Tensor) -> Result<Tensor> {
        let (seq, top_k) = (x.dim(0)?, ids.dim(1)?);
        let (down, col_l2) = self.project_inner(x, ids)?;

        // Routing weights, f32 [seq, top_k]; the fused combine takes this shape,
        // the classic candle chain reshapes to [seq, top_k, 1] for broadcasting.
        let w = weights.to_dtype(DType::F32)?;

        // Apply the routing weights and sum over top_k, undoing the L2 scale on
        // the rescale branch. The fused kernel does all of it in one pass over
        // `down`, bit-identically to the candle chain; XWEN_COMBINE_CLASSIC keeps
        // the candle chain.
        if crate::ops::combine_classic() {
            let f16_safe = 32768.0_f64;
            let down = match &col_l2 {
                Some(l2) => (down.broadcast_mul(l2)? * (1.0 / f16_safe))?,
                None => down,
            };
            let w = w.reshape((seq, top_k, 1))?;
            return Ok(down.broadcast_mul(&w)?.sum(1)?);
        }
        crate::ops::combine(&down, col_l2.as_ref(), &w)
    }

    /// The uncombined down projection, for the fused block epilogue. Declines
    /// (before doing any work) on the rescale branch: that projection carries an
    /// L2 scale only the rescale combine undoes, and the epilogue's reduction
    /// has no term for it.
    fn project(&self, x: &Tensor, ids: &Tensor) -> Result<Option<Tensor>> {
        if self.needs_rescale(x.dim(0)?, ids.dim(1)?) {
            return Ok(None);
        }
        let (down, _) = self.project_inner(x, ids)?;
        Ok(Some(down))
    }
}

/// The always-on shared expert: a plain SwiGLU with intermediate
/// `shared_expert_ff`, scaled by ONE scalar per token —
/// `sigmoid(ffn_gate_inp_shexp . x)`, a `[hidden]` vector, not a per-channel
/// gate. Its output is added to the routed output unscaled otherwise.
pub struct SharedExpert {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
    /// `[hidden, 1]` f32: the shared-expert router, pre-transposed so the
    /// per-token scalar is one matmul.
    router_t: Tensor,
}

impl SharedExpert {
    pub fn new(w: &Weights) -> Result<Self> {
        let router = w.dense_f32("ffn_gate_inp_shexp")?;
        // Stored as a bare `[hidden]` vector on both checkpoints; a `[1, hidden]`
        // conversion means the same thing.
        let hidden = router.elem_count();
        Ok(Self {
            gate: w.qlinear("ffn_gate_shexp")?,
            up: w.qlinear("ffn_up_shexp")?,
            down: w.qlinear("ffn_down_shexp")?,
            router_t: router.reshape((hidden, 1))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.swiglu_out(x)?;
        let gate = sigmoid(&self.gate_logits(x)?)?; // [seq, 1]
        Ok(out.broadcast_mul(&gate)?)
    }

    /// The SwiGLU output BEFORE the scalar gate — what the fused block epilogue
    /// scales by the gate itself.
    ///
    /// The `silu(gate) * up` step runs as one vendored pass (`ops::silu_mul`),
    /// bit-identical to candle's two dispatches. The dense checkpoint's MLP
    /// keeps the candle chain: it is not part of an MoE block, and its
    /// dispatch count is not what bounds decode.
    pub fn swiglu_out(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate.forward(x)?;
        let u = self.up.forward(x)?;
        let h = if !crate::ops::moe_glue_classic() && x.device().is_metal() {
            crate::ops::silu_mul(&g, &u)?
        } else {
            (silu(&g)? * u)?
        };
        Ok(self.down.forward(&h)?)
    }

    /// The RAW pre-sigmoid shared-expert gate logit, `[seq, 1]` f32. The sigmoid
    /// itself lives with whoever applies the gate — the fused epilogue folds it
    /// into its own pass.
    pub fn gate_logits(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.to_dtype(DType::F32)?.matmul(&self.router_t)?)
    }
}

/// The dense checkpoint's SwiGLU MLP, on every layer. Carries the majority of
/// the 27B's prefill FLOPs: 64 layers x 3 x 2 x T x 5120 x 17408.
///
/// Each projection is held twice over ONE device allocation — as a `QLinear` for
/// decode and as a raw `QuantPlane` for the vendored dense cooperative-tensor
/// prefill gemm (`gguf::Weights::qlinear_with_plane`). The planes are `None` off
/// Metal, for a checkpoint whose FFN dtype has no vendored kernel, and under the
/// `XWEN_DENSE_MM_CLASSIC` kill-switch; every such case runs `QLinear` at every
/// seq, exactly as before this path existed.
pub struct DenseMlp {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
    planes: Option<DensePlanes>,
}

/// The three FFN projections' quantized bytes, for the prefill gemm.
struct DensePlanes {
    gate: QuantPlane,
    up: QuantPlane,
    down: QuantPlane,
}

impl DenseMlp {
    pub fn new(w: &Weights) -> Result<Self> {
        // The plane load shares its upload with the QLinear, so asking for both
        // costs no extra device memory over asking for the QLinear alone — which
        // is the only reason this is affordable at 14 GB of FFN weights.
        let (gate, gate_p) = w.qlinear_with_plane("ffn_gate")?;
        let (up, up_p) = w.qlinear_with_plane("ffn_up")?;
        let (down, down_p) = w.qlinear_with_plane("ffn_down")?;
        let planes = match (gate_p, up_p, down_p) {
            (Some(gate), Some(up), Some(down)) if !crate::ops::dense_mm_classic() => {
                Some(DensePlanes { gate, up, down })
            }
            _ => None,
        };
        Ok(Self {
            gate,
            up,
            down,
            planes,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Prefill takes the vendored gemm, which beats candle's kernel_mul_mm_q4_K
        // by ~2.4x at these shapes; decode and short spans keep QLinear, whose
        // gemv wins below the crossover (docs/decisions.md, "The dense-FFN
        // prefill gemm").
        match &self.planes {
            Some(planes) if x.dim(0)? > crate::ops::dense_mm_min_seq() => swiglu_dense_q(planes, x),
            _ => swiglu(&self.gate, &self.up, &self.down, x),
        }
    }
}

/// SwiGLU FFN: `down(silu(gate(x)) * up(x))`.
fn swiglu(gate: &QLinear, up: &QLinear, down: &QLinear, x: &Tensor) -> Result<Tensor> {
    let g = silu(&gate.forward(x)?)?;
    let u = up.forward(x)?;
    let h = (&g * &u)?;
    Ok(down.forward(&h)?)
}

/// The same SwiGLU with the three matmuls on the vendored dense prefill gemm.
/// The activation chain between them is untouched, so this differs from
/// `swiglu` only in which kernel computes each projection.
fn swiglu_dense_q(planes: &DensePlanes, x: &Tensor) -> Result<Tensor> {
    // The kernel reads x as f32 from device memory with no staging, so a
    // non-contiguous or offset view has to be materialized first. Prefill
    // hidden states arrive contiguous; this is a guard, not a hot path.
    let x = if x.is_contiguous() {
        x.clone()
    } else {
        x.contiguous()?
    };
    let g = silu(&crate::ops::matmul_dense_q(&planes.gate, &x)?)?;
    let u = crate::ops::matmul_dense_q(&planes.up, &x)?;
    let h = (&g * &u)?;
    crate::ops::matmul_dense_q(&planes.down, &h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device, Tensor};

    fn logits_row() -> Tensor {
        Tensor::from_vec(
            vec![2.0_f32, -2.0, 0.0, 1.0, -1.0, 3.0, 0.5, -0.5],
            (1, 8),
            &Device::Cpu,
        )
        .unwrap()
    }

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "expected {b}, got {a} (tol {tol})");
    }

    /// Softmax over the FULL expert set, then top-k, then renormalize. The
    /// distinction that matters: the softmax denominator includes every expert,
    /// including the ones that lose the selection, so the renormalized weights
    /// are the selected probabilities' shares of their own sum — which is the
    /// same as a softmax restricted to the selected logits, and NOT the same as
    /// any per-expert sigmoid.
    #[test]
    fn routing_softmaxes_before_the_top_k() {
        let (ids, weights) = route_from_logits(&logits_row(), 3).unwrap();

        let ids_v: Vec<u32> = ids.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(ids_v, vec![5, 0, 3], "the three largest logits, in order");

        let all = [2.0_f64, -2.0, 0.0, 1.0, -1.0, 3.0, 0.5, -0.5];
        let denom: f64 = all.iter().map(|l| l.exp()).sum();
        let sel: Vec<f64> = [3.0_f64, 2.0, 1.0]
            .iter()
            .map(|l| l.exp() / denom)
            .collect();
        let sum: f64 = sel.iter().sum();
        let expected: Vec<f64> = sel.iter().map(|p| p / sum).collect();

        let got: Vec<f32> = weights.flatten_all().unwrap().to_vec1().unwrap();
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_close(*g as f64, *e, 1e-5);
        }
        // Renormalized, so the row sums to one. Nothing rescales it afterwards:
        // this architecture has no expert weight scale.
        assert_close(got.iter().map(|g| *g as f64).sum(), 1.0, 1e-5);
    }

    /// Equal logits resolve to the lowest expert indices first.
    #[test]
    fn routing_ties_pick_lowest_indices() {
        let logits = Tensor::zeros((1, 8), DType::F32, &Device::Cpu).unwrap();
        let (ids, _) = route_from_logits(&logits, 3).unwrap();
        let ids_v: Vec<u32> = ids.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(ids_v, vec![0, 1, 2]);
    }

    /// The renormalization denominator is floored at f16's smallest normal. A
    /// softmax row cannot reach that on its own, so the clamp is exercised
    /// against a hand-built weight sum: what it protects is the division, and
    /// the floor's value is part of the contract with llama.cpp.
    #[test]
    fn routing_clamps_tiny_denominator() {
        let tiny = 1e-30_f32;
        let sum = Tensor::from_vec(vec![tiny], (1, 1), &Device::Cpu)
            .unwrap()
            .clamp(WEIGHTS_SUM_FLOOR as f32, f32::INFINITY)
            .unwrap();
        let got: Vec<f32> = sum.flatten_all().unwrap().to_vec1().unwrap();
        assert_close(got[0] as f64, WEIGHTS_SUM_FLOOR, 0.0);
    }

    /// The router matmul followed by the routing decision agrees with feeding
    /// the same logits in directly.
    #[test]
    fn router_matmul_matches_direct_logits() {
        // hidden == n_expert with an identity router, so logits == x.
        let router = Tensor::eye(8, DType::F32, &Device::Cpu).unwrap();
        let x = logits_row();

        let logits = route_logits(&router, &x).unwrap();
        let direct: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        let viamm: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in viamm.iter().zip(direct.iter()) {
            assert_close(*a as f64, *b as f64, 1e-6);
        }
    }

    /// Deterministic pseudo-random f32 tensor in roughly `[-scale, scale]`, so
    /// the quantization-vs-dequantization comparison is reproducible run to run.
    fn det_tensor(dims: &[usize], seed: u64, scale: f32) -> Tensor {
        let n: usize = dims.iter().product();
        let mut s = seed;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 40) as f32) / ((1u64 << 24) as f32); // [0, 1)
            v.push((u - 0.5) * 2.0 * scale);
        }
        Tensor::from_vec(v, dims, &Device::Cpu).unwrap()
    }

    /// A quantized f32 stack, sliced per-expert, must reproduce a
    /// dequantize-everything-then-matmul oracle. Both sides dequantize the same
    /// bytes, so the only gap is f32 accumulation order: the implementation
    /// batches each expert's assigned tokens into one gemm while the oracle runs
    /// a per-token gemv, and cancellation in the 256-wide dot products lifts that
    /// gap to a few 1e-4 — well below any wiring/weighting bug.
    fn check_reference_experts(dtype: GgmlDType) {
        let dev = Device::Cpu;
        let n_expert = 4usize;
        let n_ff = 256usize;
        let hidden = 256usize; // multiple of the 256-element K-quant block
        let seq = 3usize;
        let top_k = 2usize;

        let gate_stack = Arc::new(
            QTensor::quantize(&det_tensor(&[n_expert, n_ff, hidden], 1, 0.5), dtype).unwrap(),
        );
        let up_stack = Arc::new(
            QTensor::quantize(&det_tensor(&[n_expert, n_ff, hidden], 2, 0.5), dtype).unwrap(),
        );
        let down_stack = Arc::new(
            QTensor::quantize(&det_tensor(&[n_expert, hidden, n_ff], 3, 0.5), dtype).unwrap(),
        );

        let re = ReferenceExperts {
            gate: crate::gguf::split_expert_stack(&gate_stack, n_expert, n_ff, hidden, &dev)
                .unwrap(),
            up: crate::gguf::split_expert_stack(&up_stack, n_expert, n_ff, hidden, &dev).unwrap(),
            down: crate::gguf::split_expert_stack(&down_stack, n_expert, hidden, n_ff, &dev)
                .unwrap(),
        };

        let x = det_tensor(&[seq, hidden], 4, 1.0);
        let ids_v: Vec<u32> = vec![0, 3, 1, 2, 3, 0]; // [seq, top_k]
        let w_v: Vec<f32> = vec![0.5, 0.25, 1.5, 0.75, 0.1, 0.9];
        let ids = Tensor::from_vec(ids_v.clone(), (seq, top_k), &dev).unwrap();
        let weights = Tensor::from_vec(w_v.clone(), (seq, top_k), &dev).unwrap();

        let got = re.forward(&x, &ids, &weights).unwrap();

        // Oracle: dequantize the whole stack, evaluate each token's selected
        // experts with plain f32 matmuls.
        let gate_d = gate_stack.dequantize(&dev).unwrap();
        let up_d = up_stack.dequantize(&dev).unwrap();
        let down_d = down_stack.dequantize(&dev).unwrap();
        let mut expected_rows: Vec<Tensor> = Vec::with_capacity(seq);
        for t in 0..seq {
            let xt = x.narrow(0, t, 1).unwrap(); // [1, hidden]
            let mut acc = Tensor::zeros((1, hidden), DType::F32, &dev).unwrap();
            for k in 0..top_k {
                let e = ids_v[t * top_k + k] as usize;
                let gw = gate_d.narrow(0, e, 1).unwrap().squeeze(0).unwrap();
                let uw = up_d.narrow(0, e, 1).unwrap().squeeze(0).unwrap();
                let dw = down_d.narrow(0, e, 1).unwrap().squeeze(0).unwrap();
                let g = silu(&xt.matmul(&gw.t().unwrap()).unwrap()).unwrap();
                let u = xt.matmul(&uw.t().unwrap()).unwrap();
                let h = (&g * &u).unwrap();
                let d = h.matmul(&dw.t().unwrap()).unwrap();
                let contrib = (d * w_v[t * top_k + k] as f64).unwrap();
                acc = (acc + contrib).unwrap();
            }
            expected_rows.push(acc);
        }
        let expected = Tensor::cat(&expected_rows, 0).unwrap();

        let got_v: Vec<Vec<f32>> = got.to_vec2().unwrap();
        let exp_v: Vec<Vec<f32>> = expected.to_vec2().unwrap();
        for (gr, er) in got_v.iter().zip(exp_v.iter()) {
            for (g, e) in gr.iter().zip(er.iter()) {
                let tol = 1e-3 * (e.abs() as f64).max(1.0);
                assert_close(*g as f64, *e as f64, tol);
            }
        }
    }

    #[test]
    fn reference_experts_q8_0() {
        check_reference_experts(GgmlDType::Q8_0);
    }

    #[test]
    fn reference_experts_q4k() {
        check_reference_experts(GgmlDType::Q4K);
    }

    #[test]
    fn reference_experts_q5k() {
        check_reference_experts(GgmlDType::Q5K);
    }

    #[test]
    fn reference_experts_q6k() {
        check_reference_experts(GgmlDType::Q6K);
    }

    /// Dense-weight parts for a SwiGLU FFN, plus the hand-rolled
    /// `down(silu(gate(x)) * up(x))` its output must match. F32 QTensors
    /// round-trip losslessly, so agreement is tight.
    fn swiglu_parts(
        intermediate: usize,
        hidden: usize,
        seq: usize,
    ) -> (QLinear, QLinear, QLinear, Tensor, Tensor) {
        let gate_w = det_tensor(&[intermediate, hidden], 11, 0.5);
        let up_w = det_tensor(&[intermediate, hidden], 12, 0.5);
        let down_w = det_tensor(&[hidden, intermediate], 13, 0.5);
        let ql = |t: &Tensor| {
            QLinear::from_qtensor(Arc::new(QTensor::quantize(t, GgmlDType::F32).unwrap())).unwrap()
        };

        let x = det_tensor(&[seq, hidden], 14, 1.0);
        let g = silu(&x.matmul(&gate_w.t().unwrap()).unwrap()).unwrap();
        let u = x.matmul(&up_w.t().unwrap()).unwrap();
        let h = (&g * &u).unwrap();
        let expected = h.matmul(&down_w.t().unwrap()).unwrap();

        (ql(&gate_w), ql(&up_w), ql(&down_w), x, expected)
    }

    fn assert_rows_close(got: &Tensor, expected: &Tensor) {
        let got_v: Vec<Vec<f32>> = got.to_vec2().unwrap();
        let exp_v: Vec<Vec<f32>> = expected.to_vec2().unwrap();
        for (gr, er) in got_v.iter().zip(exp_v.iter()) {
            for (a, b) in gr.iter().zip(er.iter()) {
                assert_close(*a as f64, *b as f64, 1e-4 * (b.abs() as f64).max(1.0));
            }
        }
    }

    /// The shared expert is a SwiGLU whose output is scaled by ONE sigmoid
    /// scalar per token, from its own `[hidden]` router. A per-channel gate — or
    /// no gate — would give a different row.
    #[test]
    fn shared_expert_applies_one_sigmoid_scalar_per_token() {
        let (hidden, seq) = (16usize, 4usize);
        let (gate, up, down, x, expected) = swiglu_parts(24, hidden, seq);
        let router = det_tensor(&[hidden], 15, 0.5);
        let ffn = SharedExpert {
            gate,
            up,
            down,
            router_t: router.reshape((hidden, 1)).unwrap(),
        };

        let got = ffn.forward(&x).unwrap();
        let scale = sigmoid(&x.matmul(&router.reshape((hidden, 1)).unwrap()).unwrap()).unwrap();
        assert_rows_close(&got, &expected.broadcast_mul(&scale).unwrap());

        // The scale really is one number per token, applied to the whole row.
        let scale_v: Vec<Vec<f32>> = scale.to_vec2().unwrap();
        assert_eq!(scale_v.len(), seq);
        assert!(scale_v.iter().all(|r| r.len() == 1));
    }

    /// The whole fused MoE block — the routing kernel, the fused shared-expert
    /// activation and the fused epilogue — must reproduce the candle
    /// composition it replaces BIT-FOR-BIT at decode shape. The ops-level tests
    /// pin each kernel against its own chain; this one pins the WIRING: an
    /// operand handed to the wrong kernel argument, a gate logit that picked up
    /// a stray sigmoid, or a shared-expert term added in the wrong order would
    /// all survive the per-kernel tests and die here.
    ///
    /// The reference side is spelled out rather than taken from
    /// `MoeBlock::forward`'s classic branch, because that branch itself runs
    /// bit-identical fused kernels (`ops::combine`, `ops::silu_mul`) unless
    /// their own switches are set — comparing against it would compare two
    /// fused paths. Here every reference step is a candle op.
    #[test]
    fn fused_block_matches_candle_bitwise() {
        use candle_core::quantized::QStorage;
        use std::borrow::Cow;

        let device = crate::gguf::metal_device().unwrap();
        // q4_K quantizes in 256-element blocks, so every contracted dimension
        // is a multiple of 256.
        let (hidden, expert_ff, n_expert, top_k, seq) =
            (256usize, 256usize, 32usize, 4usize, 1usize);

        let stack = |n_out: usize, k: usize, seed: u64| -> ExpertStack {
            let one = det_tensor(&[n_out, k], seed, 0.5);
            let qt = QTensor::quantize(&one, GgmlDType::Q4K).unwrap();
            let bytes = qt.data().unwrap();
            let mut all = Vec::with_capacity(bytes.len() * n_expert);
            for e in 0..n_expert {
                // Perturb each expert's first byte so the experts are not
                // interchangeable and a mis-indexed gather cannot pass.
                let mut copy = bytes.to_vec();
                copy[0] = copy[0].wrapping_add(e as u8);
                all.extend_from_slice(&copy);
            }
            let storage = QStorage::from_data(Cow::Owned(all), &device, GgmlDType::Q4K).unwrap();
            let buffer = match &storage {
                QStorage::Metal(qms) => Some(Arc::new(qms.buffer().clone())),
                _ => panic!("expert stacks require Metal storage"),
            };
            let qtensor = Arc::new(QTensor::new(storage, (n_expert, n_out, k)).unwrap());
            ExpertStack {
                qtensor: Some(qtensor),
                buffer,
                base_off: 0,
                mmap: None,
                dtype: GgmlDType::Q4K,
                n_expert,
                n_out,
                k,
            }
        };
        let ql = |t: &Tensor| {
            let qt = QTensor::quantize_onto(t, GgmlDType::Q4K, &device).unwrap();
            QLinear::from_qtensor(Arc::new(qt)).unwrap()
        };

        let shared = SharedExpert {
            gate: ql(&det_tensor(&[expert_ff, hidden], 0x20, 0.5)),
            up: ql(&det_tensor(&[expert_ff, hidden], 0x21, 0.5)),
            down: ql(&det_tensor(&[hidden, expert_ff], 0x22, 0.5)),
            router_t: det_tensor(&[hidden, 1], 0x23, 0.4)
                .to_device(&device)
                .unwrap(),
        };
        let block = MoeBlock {
            router_t: det_tensor(&[n_expert, hidden], 0x10, 0.4)
                .to_device(&device)
                .unwrap()
                .t()
                .unwrap()
                .contiguous()
                .unwrap(),
            experts: Box::new(FusedExperts {
                gate: stack(expert_ff, hidden, 0x01),
                up: stack(expert_ff, hidden, 0x02),
                down: stack(hidden, expert_ff, 0x03),
            }),
            shared,
            n_expert,
            n_expert_used: top_k,
        };
        assert!(
            crate::ops::moe_router_supported(n_expert, top_k),
            "this geometry must reach the fused router, or the test proves nothing"
        );

        let x = det_tensor(&[seq, hidden], 0x42, 1.7)
            .to_device(&device)
            .unwrap();
        let got = block.forward(&x).unwrap();

        // Reference: the candle routing chain, the candle broadcast/sum combine,
        // the candle silu+mul shared expert, then routed + shared.
        let logits = route_logits(&block.router_t, &x).unwrap();
        let (ids, weights) = route_from_logits(&logits, top_k).unwrap();
        let down = block.experts.project(&x, &ids).unwrap().unwrap();
        let routed = down
            .broadcast_mul(&weights.reshape((seq, top_k, 1)).unwrap())
            .unwrap()
            .sum(1)
            .unwrap();
        let g = block.shared.gate.forward(&x).unwrap();
        let u = block.shared.up.forward(&x).unwrap();
        let h = (silu(&g).unwrap() * u).unwrap();
        let sh = block.shared.down.forward(&h).unwrap();
        let gate = sigmoid(&x.matmul(&block.shared.router_t).unwrap()).unwrap();
        let want = (routed + sh.broadcast_mul(&gate).unwrap()).unwrap();

        let gv: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let wv: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(gv.len(), wv.len());
        for (i, (a, b)) in gv.iter().zip(wv.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "fused MoE block element {i} differs (fused {a:?}, candle {b:?})"
            );
        }
    }

    /// The dense checkpoint's FFN is a plain ungated SwiGLU.
    #[test]
    fn dense_mlp_swiglu() {
        let (hidden, seq) = (16usize, 4usize);
        let (gate, up, down, x, expected) = swiglu_parts(48, hidden, seq);
        // No planes: a CPU-device fixture, so the prefill gemm is unreachable
        // and every seq takes the QLinear chain.
        let ffn = DenseMlp {
            gate,
            up,
            down,
            planes: None,
        };
        assert_rows_close(&ffn.forward(&x).unwrap(), &expected);
    }

    /// A `DenseMlp` on a Metal device holding BOTH views of the same quantized
    /// weights — the `QLinear`s the classic path uses and the `QuantPlane`s the
    /// prefill gemm uses — built over one upload per projection, exactly as
    /// `Weights::qlinear_with_plane` does in production. `with_planes = false`
    /// yields the block the `XWEN_DENSE_MM_CLASSIC` kill-switch (and any
    /// non-Metal or unsupported-dtype load) produces.
    fn dense_mlp_q4k(device: &Device, hidden: usize, ff: usize, with_planes: bool) -> DenseMlp {
        use candle_core::quantized::QStorage;

        let build = |rows: usize, cols: usize, seed: u64| -> (QLinear, QuantPlane) {
            let dense = det_tensor(&[rows, cols], seed, 0.5);
            let qcpu = QTensor::quantize(&dense, GgmlDType::Q4K).unwrap();
            let storage =
                QStorage::from_data(qcpu.data().unwrap(), device, GgmlDType::Q4K).unwrap();
            let QStorage::Metal(qms) = &storage else {
                panic!("expected Metal quantized storage")
            };
            let buffer = Arc::new(qms.buffer().clone());
            let qtensor = Arc::new(QTensor::new(storage, (rows, cols)).unwrap());
            (
                QLinear::from_qtensor(qtensor).unwrap(),
                QuantPlane {
                    buffer,
                    base_off: 0,
                    dtype: GgmlDType::Q4K,
                    out_dim: rows,
                    in_dim: cols,
                },
            )
        };

        let (gate, gate_p) = build(ff, hidden, 0x501);
        let (up, up_p) = build(ff, hidden, 0x502);
        let (down, down_p) = build(hidden, ff, 0x503);
        DenseMlp {
            gate,
            up,
            down,
            planes: with_planes.then(|| DensePlanes {
                gate: gate_p,
                up: up_p,
                down: down_p,
            }),
        }
    }

    /// The whole dense FFN block on the vendored prefill gemm must reproduce the
    /// `QLinear` chain it replaces, over the SAME quantized bytes, at a real
    /// prefill chunk. Both sides stage the weight tile as half and accumulate in
    /// f32; the vendored one additionally runs matmul2d's reduced-precision
    /// tensor-core path, so it lands in the fork's ~2e-4 prefill precision class
    /// (docs/parity.md §3b). Bound at 5e-4 — the same bound the attention tensor
    /// gemm is held to against its classic twin, and the bound the kernel-level
    /// test in ops::dense_mm uses. Errors compound across three chained matmuls
    /// and a silu, which is exactly what this block-level check is for.
    #[test]
    fn dense_mlp_prefill_gemm_matches_qlinear() {
        let device = crate::gguf::metal_device().unwrap();
        // K must be a whole multiple of the 256-element Q4_K super-block, as
        // every production FFN dim is; the shapes are scaled down from the 27B's
        // 5120/17408 to keep the test fast.
        let (hidden, ff, seq) = (512usize, 1536usize, 96usize);
        let fused = dense_mlp_q4k(&device, hidden, ff, true);
        let classic = dense_mlp_q4k(&device, hidden, ff, false);

        let x = det_tensor(&[seq, hidden], 0x504, 1.0)
            .to_device(&device)
            .unwrap();
        let flat = |t: Tensor| -> Vec<f32> { t.flatten_all().unwrap().to_vec1().unwrap() };
        let rel_l2 = |a: &[f32], b: &[f32]| -> f32 {
            let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
            let den: f32 = b.iter().map(|y| y * y).sum();
            (num / den.max(f32::MIN_POSITIVE)).sqrt()
        };

        // Stage by stage, so the block-level number is attributable rather than
        // asserted blind: each projection lands in the fork's ~2e-4 prefill
        // precision class, and the block's larger residual is that error
        // compounding through the silu product and the second projection's
        // K-wide dot, not a wiring difference.
        let planes = fused.planes.as_ref().unwrap();
        let g_new = flat(crate::ops::matmul_dense_q(&planes.gate, &x).unwrap());
        let g_old = flat(classic.gate.forward(&x).unwrap());
        let u_new = flat(crate::ops::matmul_dense_q(&planes.up, &x).unwrap());
        let u_old = flat(classic.up.forward(&x).unwrap());
        let rel_gate = rel_l2(&g_new, &g_old);
        let rel_up = rel_l2(&u_new, &u_old);

        let a = flat(fused.forward(&x).unwrap());
        let b = flat(classic.forward(&x).unwrap());
        let rel = rel_l2(&a, &b);
        eprintln!(
            "dense FFN, prefill gemm vs QLinear: gate {rel_gate:.3e}  up {rel_up:.3e}  \
             block {rel:.3e}"
        );

        // Each projection alone: the kernel-level class (ops::dense_mm's TOL).
        assert!(
            rel_gate < 5e-4,
            "gate projection rel_l2 {rel_gate} too high"
        );
        assert!(rel_up < 5e-4, "up projection rel_l2 {rel_up} too high");
        // The block: two more error-amplifying steps past those projections (the
        // silu product doubles the relative perturbation, the down projection
        // then contracts it over ff with cancellation), so the bound is the
        // per-projection class scaled by the chain rather than reused flat.
        // Measured ~1.1e-3 at these shapes; bound at 3e-3 for the same ~3x
        // headroom the per-matmul bounds carry over their measured values.
        assert!(rel < 3e-3, "dense FFN block rel_l2 {rel} too high");
    }

    /// At or below `DENSE_MM_MIN_SEQ` the block must take the `QLinear` chain
    /// even when planes are loaded: decode is explicitly untouched by this path,
    /// and the threshold is exclusive. A short span therefore has to come out
    /// BITWISE identical to the same block with no planes at all, while a span
    /// one token past the threshold does not (it runs a different kernel).
    #[test]
    fn dense_mlp_below_threshold_takes_the_classic_path() {
        let device = crate::gguf::metal_device().unwrap();
        let (hidden, ff) = (512usize, 1536usize);
        let fused = dense_mlp_q4k(&device, hidden, ff, true);
        let classic = dense_mlp_q4k(&device, hidden, ff, false);

        let min_seq = crate::ops::dense_mm_min_seq();
        let bits = |mlp: &DenseMlp, x: &Tensor| -> Vec<u32> {
            mlp.forward(x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };

        for seq in [1usize, min_seq - 1, min_seq] {
            let x = det_tensor(&[seq, hidden], 0x505, 1.0)
                .to_device(&device)
                .unwrap();
            assert_eq!(
                bits(&fused, &x),
                bits(&classic, &x),
                "seq {seq} does not exceed the threshold {min_seq} and must be \
                 bit-identical to the plane-free block"
            );
        }

        // One token past the threshold the vendored gemm runs, so the outputs
        // are close but not bit-identical — the guard that the assertions above
        // are actually observing the gate and not a no-op.
        let x = det_tensor(&[min_seq + 1, hidden], 0x506, 1.0)
            .to_device(&device)
            .unwrap();
        assert_ne!(
            bits(&fused, &x),
            bits(&classic, &x),
            "seq {} must take the vendored prefill gemm",
            min_seq + 1
        );
    }

    /// Decode MoE-FFN and per-token-overhead perf benches (phase 0 of the
    /// decode budget work; the attention half lives in attention.rs
    /// tests::decode_bench, whose conventions these copy). Synthetic weights at
    /// production geometry — never loads a model file. All are `#[ignore]`d;
    /// run one at a time with e.g.
    /// `cargo test --release moe_decode_ffn_bench -- --ignored --nocapture`.
    /// Iteration counts: XWEN_BENCH_WARMUP (default 10) / XWEN_BENCH_ITERS
    /// (default 50). Each iter ends in one small CPU readback so it measures
    /// end-to-end latency including the command-buffer flush.
    mod decode_bench {
        use super::*;
        use candle_core::Module;
        use candle_core::quantized::{QMatMul, QStorage};
        use candle_nn::RmsNorm;
        use std::borrow::Cow;
        use std::time::Instant;

        use crate::sampler::{Sampler, SamplerOptions};

        const HIDDEN: usize = 2048;
        /// GGUF `qwen35moe.expert_feed_forward_length` (and
        /// `expert_shared_feed_forward_length` — both 512 in the Q4_K_M file).
        const EXPERT_FF: usize = 512;
        const SHARED_FF: usize = 512;
        const N_EXPERT: usize = 256;
        const TOP_K: usize = 8;
        const VOCAB: usize = 248320;
        /// Every layer of the 35B-A3B is MoE; results are scaled to this.
        const MOE_LAYERS: usize = 40;
        /// Memory constraint: 40 distinct synthetic layers would be ~18GB of
        /// expert stacks, so build 4 distinct layers (~1.8GB) and loop them 10
        /// times per iter — exactly 40 layer-evals.
        const N_DISTINCT: usize = 4;
        const PASSES: usize = 10;
        /// Row-block height for the tiled synthetic `[VOCAB, HIDDEN]` tables;
        /// must divide VOCAB (248320 = 485 x 512).
        const VOCAB_TILE: usize = 512;

        fn metal() -> Device {
            Device::new_metal(0).expect("decode benches require the Metal device")
        }

        fn iter_counts() -> (usize, usize) {
            let get = |k: &str, d: usize| {
                std::env::var(k)
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(d)
            };
            (get("XWEN_BENCH_WARMUP", 10), get("XWEN_BENCH_ITERS", 50))
        }

        /// Small readback forcing command-buffer completion (the per-iter sync).
        fn read_scalar(t: &Tensor) -> f32 {
            let t = if t.dtype() == DType::F32 {
                t.clone()
            } else {
                t.to_dtype(DType::F32).unwrap()
            };
            t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]
        }

        /// Warm-up + timed loop; returns the mean ms/iter.
        fn bench(name: &str, mut f: impl FnMut() -> f32) -> f64 {
            let (warm, iters) = iter_counts();
            let mut sink = 0f32;
            for _ in 0..warm {
                sink += f();
            }
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = Instant::now();
                sink += f();
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
            eprintln!(
                "{name}: mean {mean:.3} ms/iter, min {min:.3} ms/iter ({iters} iters, sink {sink:.1})"
            );
            mean
        }

        /// A quantized `[n_expert, n_out, k]` ExpertStack whose bytes are 256
        /// tiled copies of ONE quantized expert. Weight VALUES are
        /// timing-irrelevant (the mv_id gather reads each selected expert's
        /// slice by address, so per-pass DRAM traffic is unchanged), and tiling
        /// cuts the synthetic quantization work 256x. The buffer/QTensor
        /// construction mirrors `ops::dispatch::testutil::build_stack` (and
        /// production `gguf::expert_stack`): one `QStorage::from_data` upload,
        /// buffer retained BEFORE the storage moves into the QTensor, so the
        /// fused kernels and the QTensor share a single resident allocation.
        fn tiled_stack(dev: &Device, n_out: usize, k: usize, seed: u64) -> ExpertStack {
            let one = det_tensor(&[n_out, k], seed, 0.5);
            let qt = QTensor::quantize(&one, GgmlDType::Q4K).unwrap();
            let bytes = qt.data().unwrap();
            let mut all = Vec::with_capacity(bytes.len() * N_EXPERT);
            for _ in 0..N_EXPERT {
                all.extend_from_slice(&bytes);
            }
            let storage = QStorage::from_data(Cow::Owned(all), dev, GgmlDType::Q4K).unwrap();
            let buffer = match &storage {
                QStorage::Metal(qms) => Some(Arc::new(qms.buffer().clone())),
                _ => None,
            };
            let qtensor = Arc::new(QTensor::new(storage, (N_EXPERT, n_out, k)).unwrap());
            ExpertStack {
                qtensor: Some(qtensor),
                buffer,
                base_off: 0,
                mmap: None,
                dtype: GgmlDType::Q4K,
                n_expert: N_EXPERT,
                n_out,
                k,
            }
        }

        /// RMSNorm weights near 1.0, as in the attention decode benches.
        fn norm(dim: usize, seed: u64, dev: &Device) -> RmsNorm {
            let w = det_tensor(&[dim], seed, 0.1).affine(1.0, 1.0).unwrap();
            RmsNorm::new(w.to_device(dev).unwrap(), 1e-6)
        }

        /// A residual-stream-shaped input: uniform with RMS ~1, like the output
        /// of a well-conditioned ffn_norm.
        fn normed_input(seed: u64, dev: &Device) -> Tensor {
            det_tensor(&[1, HIDDEN], seed, 1.7).to_device(dev).unwrap()
        }

        /// One production MoeBlock (Fused experts) plus its ffn_norm, at real
        /// geometry. Router weights are scaled small so the sigmoid probs stay
        /// spread out and top-10 selection varies with the input.
        fn build_moe_layer(il: usize, dev: &Device) -> (RmsNorm, MoeBlock) {
            let s = 0x5000 + il as u64 * 64;
            let router_t = det_tensor(&[N_EXPERT, HIDDEN], s + 10, 0.02)
                .to_device(dev)
                .unwrap()
                .t()
                .unwrap()
                .contiguous()
                .unwrap();
            let experts = FusedExperts {
                gate: tiled_stack(dev, EXPERT_FF, HIDDEN, s + 1),
                up: tiled_stack(dev, EXPERT_FF, HIDDEN, s + 2),
                down: tiled_stack(dev, HIDDEN, EXPERT_FF, s + 3),
            };
            let ql = |t: &Tensor| {
                let qt = QTensor::quantize_onto(t, GgmlDType::Q4K, dev).unwrap();
                QLinear::from_qtensor(Arc::new(qt)).unwrap()
            };
            let shared = SharedExpert {
                gate: ql(&det_tensor(&[SHARED_FF, HIDDEN], s + 20, 0.5)),
                up: ql(&det_tensor(&[SHARED_FF, HIDDEN], s + 21, 0.5)),
                down: ql(&det_tensor(&[HIDDEN, SHARED_FF], s + 22, 0.5)),
                router_t: det_tensor(&[HIDDEN, 1], s + 23, 0.02)
                    .to_device(dev)
                    .unwrap(),
            };
            let block = MoeBlock {
                router_t,
                experts: Box::new(experts),
                shared,
                n_expert: N_EXPERT,
                n_expert_used: TOP_K,
            };
            (norm(HIDDEN, s + 30, dev), block)
        }

        /// The f16 token-embedding table `[VOCAB, HIDDEN]` production keeps on
        /// Metal, as tiled row-blocks (lookup timing is value-independent).
        fn build_embed(dev: &Device) -> Tensor {
            let block = det_tensor(&[VOCAB_TILE, HIDDEN], 0x111, 1.0)
                .to_device(dev)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            let tiles: Vec<Tensor> = vec![block; VOCAB / VOCAB_TILE];
            let embed = Tensor::cat(&tiles.iter().collect::<Vec<_>>(), 0).unwrap();
            assert_eq!(embed.dims(), &[VOCAB, HIDDEN]);
            embed
        }

        /// A synthetic q6_K lm_head `[VOCAB, HIDDEN]` from tiled quantized
        /// row-blocks (mv timing is value-independent). Same zero-copy
        /// construction as `gguf::qlinear_with_buffer`: the buffer is retained
        /// before the storage moves into the QTensor, and the QTensor must be
        /// kept alive so the shared allocation stays resident.
        fn build_lm_head(dev: &Device) -> (Arc<candle_metal_kernels::metal::Buffer>, Arc<QTensor>) {
            let one = det_tensor(&[VOCAB_TILE, HIDDEN], 0x333, 0.5);
            let qt = QTensor::quantize(&one, GgmlDType::Q6K).unwrap();
            let bytes = qt.data().unwrap();
            let mut all = Vec::with_capacity(bytes.len() * (VOCAB / VOCAB_TILE));
            for _ in 0..VOCAB / VOCAB_TILE {
                all.extend_from_slice(&bytes);
            }
            let storage = QStorage::from_data(Cow::Owned(all), dev, GgmlDType::Q6K).unwrap();
            let buffer = match &storage {
                QStorage::Metal(qms) => Arc::new(qms.buffer().clone()),
                _ => panic!("lm_head bench weights require Metal storage"),
            };
            let qtensor = Arc::new(QTensor::new(storage, (VOCAB, HIDDEN)).unwrap());
            (buffer, qtensor)
        }

        /// One token through 48 MoE layer-evals (4 distinct layers x 12
        /// passes), mirroring the model.rs per-layer FFN half exactly:
        /// ffn_norm -> MoeBlock::forward (route + mv_id experts + shared) ->
        /// residual add. The residual evolves across layer-evals, so the
        /// routing input (and thus the selected experts) differs every pass.
        fn moe_chain(layers: &[(RmsNorm, MoeBlock)], x0: &Tensor) -> Tensor {
            let mut x = x0.clone();
            for _ in 0..PASSES {
                for (ffn_norm, moe) in layers {
                    let normed = ffn_norm.forward(&x).unwrap();
                    let out = moe.forward(&normed).unwrap();
                    x = (&x + &out).unwrap();
                }
            }
            x
        }

        /// Headline: the MoE-FFN half of decode at production geometry through
        /// the PRODUCTION `MoeBlock::forward` path (on-GPU routing + vendored
        /// mv_id gather + silu*mul + weighted combine + shared expert), each
        /// layer wrapped with its ffn_norm and residual add as in model.rs.
        /// Also times routing-only, shared-expert-only and norm+residual-only
        /// variants to split the budget.
        #[test]
        #[ignore = "perf bench"]
        fn moe_decode_ffn_bench() {
            let dev = metal();
            eprintln!(
                "building {N_DISTINCT} synthetic q4_K MoE layers (~5.5GB device memory; \
                 expert stacks are {N_EXPERT} tiled copies of one quantized expert)..."
            );
            let t0 = Instant::now();
            let layers: Vec<(RmsNorm, MoeBlock)> = (0..N_DISTINCT)
                .map(|il| build_moe_layer(il, &dev))
                .collect();
            eprintln!(
                "build+quantize+upload time: {:.1}s",
                t0.elapsed().as_secs_f64()
            );

            let x0 = normed_input(0x4242, &dev);
            let evals = N_DISTINCT * PASSES;

            let full = bench(
                &format!(
                    "moe ffn chain x{evals} (ffn_norm + route + mv_id experts + shared + residual)"
                ),
                || read_scalar(&moe_chain(&layers, &x0)),
            );

            // Isolation variants run over pre-built normed inputs (one per
            // layer-eval) so their inputs vary the same way the chain's do.
            let inputs: Vec<Tensor> = (0..evals)
                .map(|i| normed_input(0x9000 + i as u64, &dev))
                .collect();

            let routing = bench(
                &format!(
                    "routing only x{evals} (router matmul + softmax + top-k + gather/renormalize)"
                ),
                || {
                    let mut last = None;
                    for (i, x) in inputs.iter().enumerate() {
                        let (_ids, w) = layers[i % N_DISTINCT].1.route(x).unwrap();
                        last = Some(w);
                    }
                    read_scalar(&last.unwrap())
                },
            );
            let shared = bench(&format!("shared expert only x{evals}"), || {
                let mut last = None;
                for (i, x) in inputs.iter().enumerate() {
                    last = Some(layers[i % N_DISTINCT].1.shared.forward(x).unwrap());
                }
                read_scalar(&last.unwrap())
            });
            let normres = bench(&format!("ffn_norm + residual add only x{evals}"), || {
                let mut x = x0.clone();
                for i in 0..evals {
                    let normed = layers[i % N_DISTINCT].0.forward(&x).unwrap();
                    x = (&x + &normed).unwrap();
                }
                read_scalar(&x)
            });

            let scaled = |ms: f64| ms * MOE_LAYERS as f64 / evals as f64;
            eprintln!("{MOE_LAYERS}-layer-equivalent per token:");
            eprintln!("  full MoE FFN half:   {:.3} ms", scaled(full));
            eprintln!("  routing:             {:.3} ms", scaled(routing));
            eprintln!("  shared expert:       {:.3} ms", scaled(shared));
            eprintln!("  ffn_norm + residual: {:.3} ms", scaled(normres));
            eprintln!(
                "  derived mv_id gather + silu*mul + weighted combine: {:.3} ms",
                scaled(full - routing - shared - normres)
            );
            eprintln!(
                "caveat: {N_DISTINCT} distinct layers looped {PASSES}x — repeated expert-stack \
                 reads may be SLC-cached more than {MOE_LAYERS} distinct layers would be; treat \
                 expert-read numbers as a lower bound"
            );
        }

        /// Per-token sampler + token-feedback overhead, mirroring generate.rs's
        /// decode loop: Sampler::sample (the per-token GPU->CPU logits readback
        /// + CPU top-k/top-p sampling at temp 1.0 / top_k 20) and the follow-on
        /// Tensor::new of the sampled id + f16 embedding row gather + f32
        /// upcast that model.rs's next forward starts with.
        #[test]
        #[ignore = "perf bench"]
        fn sampler_decode_bench() {
            let dev = metal();
            eprintln!(
                "building synthetic f16 embedding [{VOCAB}, {HIDDEN}] (~1.0GB; tiled \
                 row-blocks — lookup timing is value-independent) + logits pool..."
            );
            let embed = build_embed(&dev);

            const POOL: usize = 8;
            let pool: Vec<Tensor> = (0..POOL)
                .map(|i| {
                    det_tensor(&[VOCAB], 0x222 + i as u64, 4.0)
                        .to_device(&dev)
                        .unwrap()
                })
                .collect();
            // Drawing over the full synthetic row measures the widest the
            // sampler ever works; the real encodable bound is a couple of
            // hundred ids narrower, which is below this bench's resolution.
            let mut sampler = Sampler::new(SamplerOptions::default(), vec![], VOCAB);

            // Attribution floor: the GPU->CPU copy of the [VOCAB] f32 row every
            // sampling strategy has to pay, with no sampling on top.
            let mut r = 0usize;
            let readback = bench("logits readback only (GPU->CPU f32 copy)", || {
                let v = pool[r % POOL]
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                r += 1;
                v[0]
            });

            let mut i = 0usize;
            let sample_only = bench("sampler only (logits readback + CPU top-k sample)", || {
                let t = sampler.sample(&pool[i % POOL]).unwrap();
                i += 1;
                t as f32
            });
            eprintln!(
                "derived CPU sampling work (sampler minus readback): {:.3} ms",
                sample_only - readback
            );
            let mut j = 0usize;
            let full = bench(
                "sampler + token upload + f16 embed gather + f32 upcast",
                || {
                    let t = sampler.sample(&pool[j % POOL]).unwrap();
                    j += 1;
                    let input = Tensor::new(&[t], &dev).unwrap();
                    let x = embed
                        .index_select(&input.to_dtype(DType::U32).unwrap(), 0)
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap();
                    read_scalar(&x)
                },
            );
            eprintln!(
                "derived token-feedback (Tensor::new + index_select + upcast): {:.3} ms",
                full - sample_only
            );
            eprintln!(
                "caveat: the derived value includes one extra GPU sync absent in production \
                 (the embed gather is normally absorbed by the next token's layer chain)"
            );
        }

        /// The per-token tail after the layer stack, mirroring model.rs's
        /// decode path: final RMSNorm on [1, hidden] + the q6_K lm_head mat-vec
        /// (vendored `ops::mul_mv` over a retained buffer, QMatMul under
        /// XWEN_MV_CLASSIC) + full-logits readback. Cross-checks the earlier
        /// isolated lm_head number (ops::mv_id plain_mv_lmhead_bench, ~0.7 ms).
        #[test]
        #[ignore = "perf bench"]
        fn token_tail_bench() {
            let dev = metal();
            eprintln!(
                "building synthetic q6_K lm_head [{VOCAB}, {HIDDEN}] (~0.4GB; tiled \
                 quantized row-blocks — mv timing is value-independent)..."
            );
            let (buffer, qtensor) = build_lm_head(&dev);
            let qmm = QMatMul::from_arc(qtensor.clone()).unwrap();
            let use_vendored =
                !crate::ops::mv_classic() && crate::ops::mv_vendored_supported(GgmlDType::Q6K);
            eprintln!(
                "lm_head path: {}",
                if use_vendored {
                    "vendored mul_mv"
                } else {
                    "QMatMul (classic)"
                }
            );

            let output_norm = norm(HIDDEN, 0x444, &dev);
            const POOL: usize = 8;
            let pool: Vec<Tensor> = (0..POOL)
                .map(|i| normed_input(0x555 + i as u64, &dev))
                .collect();

            let mut i = 0usize;
            bench(
                "token tail (final RMSNorm + q6_K lm_head mv + full logits readback)",
                || {
                    let x = &pool[i % POOL];
                    i += 1;
                    let normed = output_norm.forward(x).unwrap();
                    let last = normed.narrow(0, 0, 1).unwrap().contiguous().unwrap();
                    let logits = if use_vendored {
                        crate::ops::mul_mv(&buffer, GgmlDType::Q6K, VOCAB, HIDDEN, &last).unwrap()
                    } else {
                        qmm.forward(&last).unwrap()
                    };
                    logits.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]
                },
            );
            eprintln!(
                "caveat: the full-logits readback here is the same per-token readback the \
                 sampler bench times; count it once when summing the budget"
            );
        }

        // --- decode expert-gather attribution benches (WP-G1) --------------
        //
        // The decode MoE budget's byte floor is the routed expert gather: three
        // `mul_mv_id` launches per layer (gate/up/down) over the top_k selected
        // experts' q4_K planes. At 35B-A3B geometry (hidden 2048, expert_ff 512,
        // top_k 8) that is ~14 MB of weight per layer and ~570 MB per token
        // across the 40 MoE layers. Comparing each launch's effective GB/s
        // against the ~365 GB/s the q6_K lm_head matvec sustains at the same
        // seq==1 says WHERE a shortfall lives: kernel-internal streaming at
        // expert geometry, inter-dispatch boundaries, or a wrong byte floor.
        // Synthetic weights at production geometry; never load a model file.

        /// Effective bandwidth: `bytes` moved in `ms` milliseconds, as GB/s
        /// (decimal GB, matching how the lm_head/prefill numbers are quoted).
        fn eff_gbps(bytes: usize, ms: f64) -> f64 {
            bytes as f64 / (ms * 1e-3) / 1e9
        }

        /// q4_K/q6_K weight bytes of ONE `[n_out, k]` expert row-plane: the ggml
        /// packing this bench measures against — `k/block_size` super-blocks per
        /// output row, `type_size` bytes each (q4_K 144/256 = 0.5625 B/weight,
        /// q6_K 210/256 = 0.8203 B/weight). Mirrors `dispatch::run`'s
        /// `bytes_per_row = k / block_size * type_size`.
        fn expert_bytes(dt: GgmlDType, n_out: usize, k: usize) -> usize {
            n_out * (k / dt.block_size()) * dt.type_size()
        }

        /// A `[n_out, k]` stack of `N_EXPERT` tiled quantized experts at dtype
        /// `dt` — `tiled_stack` generalized past q4_K for the q6_K gather
        /// variants. Same zero-copy construction (buffer retained before the
        /// storage moves into the QTensor); weight VALUES are timing-irrelevant.
        fn tiled_stack_dt(
            dev: &Device,
            n_out: usize,
            k: usize,
            seed: u64,
            dt: GgmlDType,
        ) -> ExpertStack {
            let one = det_tensor(&[n_out, k], seed, 0.5);
            let qt = QTensor::quantize(&one, dt).unwrap();
            let bytes = qt.data().unwrap();
            let mut all = Vec::with_capacity(bytes.len() * N_EXPERT);
            for _ in 0..N_EXPERT {
                all.extend_from_slice(&bytes);
            }
            let storage = QStorage::from_data(Cow::Owned(all), dev, dt).unwrap();
            let buffer = match &storage {
                QStorage::Metal(qms) => Some(Arc::new(qms.buffer().clone())),
                _ => None,
            };
            let qtensor = Arc::new(QTensor::new(storage, (N_EXPERT, n_out, k)).unwrap());
            ExpertStack {
                qtensor: Some(qtensor),
                buffer,
                base_off: 0,
                mmap: None,
                dtype: dt,
                n_expert: N_EXPERT,
                n_out,
                k,
            }
        }

        /// A rank-2 `[n_out, k]` quantized weight sharing one allocation between a
        /// retained Metal buffer and a QTensor (the `qlinear_with_buffer` / lm_head
        /// zero-copy shape), built from `n_out/1024` tiled quantized row-blocks so
        /// large weights quantize fast. `n_out` must be a multiple of 1024.
        fn tiled_plain(
            dev: &Device,
            n_out: usize,
            k: usize,
            seed: u64,
            dt: GgmlDType,
        ) -> (Arc<candle_metal_kernels::metal::Buffer>, Arc<QTensor>) {
            const TILE: usize = 1024;
            assert!(
                n_out.is_multiple_of(TILE),
                "n_out {n_out} must be a multiple of {TILE}"
            );
            let one = det_tensor(&[TILE, k], seed, 0.5);
            let qt = QTensor::quantize(&one, dt).unwrap();
            let bytes = qt.data().unwrap();
            let mut all = Vec::with_capacity(bytes.len() * (n_out / TILE));
            for _ in 0..n_out / TILE {
                all.extend_from_slice(&bytes);
            }
            let storage = QStorage::from_data(Cow::Owned(all), dev, dt).unwrap();
            let buffer = match &storage {
                QStorage::Metal(qms) => Arc::new(qms.buffer().clone()),
                _ => panic!("tiled_plain weights require Metal storage"),
            };
            let qtensor = Arc::new(QTensor::new(storage, (n_out, k)).unwrap());
            (buffer, qtensor)
        }

        /// A pool of `n` distinct random `[1, top_k]` u32 id vectors (values in
        /// `0..n_expert`), on device. Rotating through the pool between launches
        /// makes each launch gather a DIFFERENT ~50 MiB slice of the 256-expert
        /// stack (>> any SLC), so back-to-back isolated launches stay DRAM-bound
        /// instead of re-reading one cache-resident set — the reason these benches
        /// aren't fooled into reporting cache bandwidth.
        fn ids_pool(
            dev: &Device,
            n: usize,
            top_k: usize,
            n_expert: usize,
            seed: u64,
        ) -> Vec<Tensor> {
            (0..n)
                .map(|p| {
                    let mut s = seed
                        .wrapping_add(p as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(1);
                    let ids: Vec<u32> = (0..top_k)
                        .map(|_| {
                            s ^= s << 13;
                            s ^= s >> 7;
                            s ^= s << 17;
                            (s % n_expert as u64) as u32
                        })
                        .collect();
                    Tensor::from_vec(ids, (1, top_k), dev).unwrap()
                })
                .collect()
        }

        /// Amortized GPU timing: each timed batch issues `inner` calls to
        /// `launch` (each pushing its output tensor(s) into `outs`, kept alive for
        /// the whole batch so candle can't recycle a buffer a still-pending
        /// dispatch reads), then drains the batch with ONE readback of the last
        /// output. Dividing by `inner` amortizes the command-buffer flush, so the
        /// result is the kernel+encode cost of one unit, NOT the per-launch
        /// GPU->CPU sync latency (measure that separately with `inner == 1`). A
        /// unit may issue more than one kernel (the concurrency pair does two).
        fn bench_amortized(
            name: &str,
            inner: usize,
            mut launch: impl FnMut(usize, &mut Vec<Tensor>),
        ) -> f64 {
            let (warm, iters) = iter_counts();
            let mut sink = 0f32;
            for _ in 0..warm {
                let mut outs = Vec::new();
                for i in 0..inner {
                    launch(i, &mut outs);
                }
                sink += read_scalar(outs.last().unwrap());
            }
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let mut outs = Vec::new();
                let t = Instant::now();
                for i in 0..inner {
                    launch(i, &mut outs);
                }
                sink += read_scalar(outs.last().unwrap());
                times.push(t.elapsed().as_secs_f64() * 1e3 / inner as f64);
            }
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
            eprintln!(
                "{name}: mean {mean:.4} ms/unit, min {min:.4} ms/unit (inner {inner}, {iters} iters, sink {sink:.1})"
            );
            mean
        }

        /// Bench 1: ONE isolated `mul_mv_id` launch at the exact decode geometry,
        /// looped back-to-back with nothing between (ids rotate so the gather stays
        /// DRAM-bound). Reports ms/launch and effective GB/s counting only the
        /// `top_k` gathered experts' bytes, for gate/up ([1024,3072], x_per_row=1)
        /// and down ([3072,1024], x_per_row=top_k), q4_K and q6_K. The decisive
        /// read: ~350 GB/s says the kernel streams fine and the decode gap lives at
        /// the pipeline boundaries; ~180 GB/s says the kernel itself under-streams
        /// at expert geometry (the lm_head conclusion does not transfer). The
        /// `inner == 1` gate row exposes the per-launch sync latency for contrast.
        #[test]
        #[ignore = "perf bench"]
        fn decode_expert_gather_mv_bench() {
            let dev = metal();
            const INNER: usize = 128;
            eprintln!("building isolated gate/down expert stacks (q4_K + q6_K)...");
            let ids = ids_pool(&dev, 64, TOP_K, N_EXPERT, 0xABCDE);
            let x_gate = det_tensor(&[1, 1, HIDDEN], 0x4242, 1.7)
                .to_device(&dev)
                .unwrap();
            let x_down = det_tensor(&[1, TOP_K, EXPERT_FF], 0x4243, 1.7)
                .to_device(&dev)
                .unwrap();

            let run_gather = |label: &str, stack: &ExpertStack, x: &Tensor, inner: usize| {
                let mut c = 0usize;
                let ms = bench_amortized(label, inner, |_, outs| {
                    let id = &ids[c % ids.len()];
                    c += 1;
                    outs.push(mv_id::mul_mv_id(stack, x, id).unwrap());
                });
                let gathered = TOP_K * expert_bytes(stack.dtype, stack.n_out, stack.k);
                eprintln!(
                    "  {label}: {ms:.4} ms/launch, {:.1} GB/s (gathered {TOP_K} experts = {:.2} MiB)",
                    eff_gbps(gathered, ms),
                    gathered as f64 / (1usize << 20) as f64,
                );
            };

            let gate_q4 = tiled_stack_dt(&dev, EXPERT_FF, HIDDEN, 0xA001, GgmlDType::Q4K);
            let down_q4 = tiled_stack_dt(&dev, HIDDEN, EXPERT_FF, 0xA002, GgmlDType::Q4K);
            let gate_q6 = tiled_stack_dt(&dev, EXPERT_FF, HIDDEN, 0xA003, GgmlDType::Q6K);
            let down_q6 = tiled_stack_dt(&dev, HIDDEN, EXPERT_FF, 0xA004, GgmlDType::Q6K);

            run_gather("gate q4_K [1024,3072] amortized", &gate_q4, &x_gate, INNER);
            run_gather(
                "gate q4_K [1024,3072] sync-per-launch",
                &gate_q4,
                &x_gate,
                1,
            );
            run_gather("down q4_K [3072,1024] amortized", &down_q4, &x_down, INNER);
            run_gather("gate q6_K [1024,3072] amortized", &gate_q6, &x_gate, INNER);
            run_gather("down q6_K [3072,1024] amortized", &down_q6, &x_down, INNER);
            eprintln!(
                "read: ~350 GB/s => kernel streams fine, gap is pipeline boundaries; \
                 ~180 GB/s => the mv_id kernel under-streams at expert geometry. The \
                 amortized-vs-sync-per-launch gate delta is the per-dispatch sync/boundary cost."
            );
        }

        /// Bench 2: plain `mul_mv` (q6_K, seq==1) over a GROWING output-row count
        /// at fixed k=3072 — effective GB/s vs weight size. Locates where the
        /// matvec kernel climbs off cache onto the DRAM floor; the expert row
        /// plane (n_out 1024/3072) and the lm_head (100352) sit on the same curve,
        /// so it says whether an expert-sized matvec is inherently slower than the
        /// lm_head or just cache-cold. Same weight reused each launch, so small
        /// sizes (< SLC) report CACHE bandwidth by design; the large end is the
        /// true DRAM number.
        #[test]
        #[ignore = "perf bench"]
        fn decode_plain_mv_rowsweep_bench() {
            let dev = metal();
            const INNER: usize = 64;
            const K: usize = HIDDEN; // 3072, the model's hidden dim / lm_head k.
            let dt = GgmlDType::Q6K;
            let x = det_tensor(&[1, K], 0x1234, 1.0).to_device(&dev).unwrap();
            eprintln!("plain q6_K mv, seq==1, k={K}, growing n_out:");
            for &n_out in &[1024usize, 3072, 4096, 16384, 100352] {
                let (buffer, _qt) = tiled_plain(&dev, n_out, K, 0x900 + n_out as u64, dt);
                let ms =
                    bench_amortized(&format!("plain mv q6_K [{n_out}x{K}]"), INNER, |_, outs| {
                        outs.push(crate::ops::mul_mv(&buffer, dt, n_out, K, &x).unwrap());
                    });
                let bytes = expert_bytes(dt, n_out, K);
                eprintln!(
                    "  n_out {n_out:>6}: {ms:.4} ms/launch, {:.1} GB/s ({:.2} MiB weight{})",
                    eff_gbps(bytes, ms),
                    bytes as f64 / (1usize << 20) as f64,
                    if bytes < (32 << 20) {
                        ", may be SLC-resident"
                    } else {
                        ""
                    },
                );
            }
        }

        /// Bench 3: cumulative stage ablation of the decode routed block, built
        /// from the SAME ops `FusedExperts::forward` runs at seq==1 (mv_id gate,
        /// mv_id up, candle silu*mul, mv_id down, `ops::combine`) — production has
        /// `use_mm=false` and `needs_rescale=false` at seq==1, so this IS that
        /// path minus routing. Each stage adds one operation; the deltas expose
        /// per-boundary drain/launch cost. Chains run `N_DISTINCT*PASSES` (48)
        /// layer-evals and scale to 47; stage (d) should reconcile with
        /// `moe_decode_ffn_bench`'s derived "mv_id gather + silu*mul + combine".
        #[test]
        #[ignore = "perf bench"]
        fn decode_moe_stage_ablation_bench() {
            let dev = metal();
            let evals = N_DISTINCT * PASSES;
            eprintln!("building {N_DISTINCT} q4_K expert triples (gate/up/down)...");
            struct Triple {
                gate: ExpertStack,
                up: ExpertStack,
                down: ExpertStack,
            }
            let triples: Vec<Triple> = (0..N_DISTINCT)
                .map(|il| {
                    let s = 0x7000 + il as u64 * 64;
                    Triple {
                        gate: tiled_stack_dt(&dev, EXPERT_FF, HIDDEN, s + 1, GgmlDType::Q4K),
                        up: tiled_stack_dt(&dev, EXPERT_FF, HIDDEN, s + 2, GgmlDType::Q4K),
                        down: tiled_stack_dt(&dev, HIDDEN, EXPERT_FF, s + 3, GgmlDType::Q4K),
                    }
                })
                .collect();
            let xs: Vec<Tensor> = (0..evals)
                .map(|i| {
                    det_tensor(&[1, 1, HIDDEN], 0x8000 + i as u64, 1.7)
                        .to_device(&dev)
                        .unwrap()
                })
                .collect();
            let ids = ids_pool(&dev, evals, TOP_K, N_EXPERT, 0x8888);
            let ws: Vec<Tensor> = (0..evals)
                .map(|i| {
                    det_tensor(&[1, TOP_K], 0x9000 + i as u64, 0.3)
                        .to_device(&dev)
                        .unwrap()
                })
                .collect();

            let gate_up = bench(&format!("(a) gate+up x{evals}"), || {
                let mut last = None;
                for i in 0..evals {
                    let t = &triples[i % N_DISTINCT];
                    let _g = mv_id::mul_mv_id(&t.gate, &xs[i], &ids[i]).unwrap();
                    let u = mv_id::mul_mv_id(&t.up, &xs[i], &ids[i]).unwrap();
                    last = Some((_g, u));
                }
                read_scalar(&last.unwrap().1)
            });
            let plus_act = bench(&format!("(b) gate+up+silu*mul x{evals}"), || {
                let mut last = None;
                for i in 0..evals {
                    let t = &triples[i % N_DISTINCT];
                    let g = mv_id::mul_mv_id(&t.gate, &xs[i], &ids[i]).unwrap();
                    let u = mv_id::mul_mv_id(&t.up, &xs[i], &ids[i]).unwrap();
                    last = Some((silu(&g).unwrap() * u).unwrap());
                }
                read_scalar(&last.unwrap())
            });
            let plus_down = bench(&format!("(c) +down x{evals}"), || {
                let mut last = None;
                for i in 0..evals {
                    let t = &triples[i % N_DISTINCT];
                    let g = mv_id::mul_mv_id(&t.gate, &xs[i], &ids[i]).unwrap();
                    let u = mv_id::mul_mv_id(&t.up, &xs[i], &ids[i]).unwrap();
                    let act = (silu(&g).unwrap() * u).unwrap();
                    last = Some(mv_id::mul_mv_id(&t.down, &act, &ids[i]).unwrap());
                }
                read_scalar(&last.unwrap())
            });
            let plus_combine = bench(&format!("(d) +combine x{evals}"), || {
                let mut last = None;
                for i in 0..evals {
                    let t = &triples[i % N_DISTINCT];
                    let g = mv_id::mul_mv_id(&t.gate, &xs[i], &ids[i]).unwrap();
                    let u = mv_id::mul_mv_id(&t.up, &xs[i], &ids[i]).unwrap();
                    let act = (silu(&g).unwrap() * u).unwrap();
                    let down = mv_id::mul_mv_id(&t.down, &act, &ids[i]).unwrap();
                    last = Some(crate::ops::combine(&down, None, &ws[i]).unwrap());
                }
                read_scalar(&last.unwrap())
            });

            let scale47 = |ms: f64| ms * MOE_LAYERS as f64 / evals as f64;
            eprintln!("{MOE_LAYERS}-layer-equivalent per token (cumulative):");
            eprintln!("  (a) gate+up:            {:.3} ms", scale47(gate_up));
            eprintln!(
                "  (b) +silu*mul:          {:.3} ms  (Δ {:.3})",
                scale47(plus_act),
                scale47(plus_act - gate_up)
            );
            eprintln!(
                "  (c) +down:              {:.3} ms  (Δ {:.3})",
                scale47(plus_down),
                scale47(plus_down - plus_act)
            );
            eprintln!(
                "  (d) +combine:           {:.3} ms  (Δ {:.3})",
                scale47(plus_combine),
                scale47(plus_combine - plus_down)
            );
            eprintln!(
                "  (d) reconciles with moe_decode_ffn_bench's derived \
                 'mv_id gather + silu*mul + combine' (~13.7 ms LPM)."
            );
        }

        /// Bench 4: concurrency probe. gate and up are independent (distinct dsts,
        /// shared read-only x/ids) and candle uses a concurrent encoder, so they
        /// COULD overlap. Time one gate launch vs a gate+up pair (identical shapes,
        /// so T(up)==T(gate)); ratio pair/(2*single) < 1 means real overlap, ~1
        /// means serialized OR already bandwidth-saturated on a single launch (no
        /// spare bandwidth to overlap into — read alongside bench 1's GB/s).
        #[test]
        #[ignore = "perf bench"]
        fn decode_gather_concurrency_bench() {
            let dev = metal();
            const INNER: usize = 128;
            let gate = tiled_stack_dt(&dev, EXPERT_FF, HIDDEN, 0xC001, GgmlDType::Q4K);
            let up = tiled_stack_dt(&dev, EXPERT_FF, HIDDEN, 0xC002, GgmlDType::Q4K);
            let ids = ids_pool(&dev, 64, TOP_K, N_EXPERT, 0xC0DE);
            let x = det_tensor(&[1, 1, HIDDEN], 0x4242, 1.7)
                .to_device(&dev)
                .unwrap();

            let mut c = 0usize;
            let single = bench_amortized("gate alone", INNER, |_, outs| {
                let id = &ids[c % ids.len()];
                c += 1;
                outs.push(mv_id::mul_mv_id(&gate, &x, id).unwrap());
            });
            let mut c = 0usize;
            let pair = bench_amortized("gate;up pair", INNER, |_, outs| {
                let id = &ids[c % ids.len()];
                c += 1;
                outs.push(mv_id::mul_mv_id(&gate, &x, id).unwrap());
                outs.push(mv_id::mul_mv_id(&up, &x, id).unwrap());
            });
            eprintln!(
                "single {single:.4} ms, pair {pair:.4} ms; pair/(2*single) = {:.3} \
                 (< 1 overlap, ~1 serialized or bandwidth-saturated)",
                pair / (2.0 * single)
            );
        }

        // --- prefill-isolation benches (seq=512) ---------------------------
        //
        // Each isolates one dispatch group of the production MoE block at
        // prefill width so the per-layer prefill budget can be attributed to a
        // category (routing glue / mm_id expert matmuls / silu+rescale glue /
        // weighted combine / shared expert / norm+residual). Synthetic weights
        // at production geometry — never load a model file.
        // `prefill_moe_block_bench` runs the whole production `MoeBlock::forward`
        // (mm_id prefill path) and is the sum-check for the isolated parts.
        // Run one at a time, e.g.
        // `cargo test --release prefill_route_bench -- --ignored --nocapture`.

        /// Prefill chunk length the isolation benches share, so their numbers
        /// are directly comparable (same seq everywhere, top_k 10, hidden 3072,
        /// expert_ff 1024, 256 experts).
        const PREFILL_SEQ: usize = 512;

        /// A [seq, hidden] prefill activation with residual-stream conditioning
        /// (RMS ~1, like a well-behaved ffn_norm output).
        fn prefill_input(seed: u64, dev: &Device) -> Tensor {
            det_tensor(&[PREFILL_SEQ, HIDDEN], seed, 1.7)
                .to_device(dev)
                .unwrap()
        }

        /// Routing glue alone (`MoeBlock::route`): the router matmul plus the
        /// decision chain (softmax, top-k argsort, narrow/contiguous, gather,
        /// sum, clamp, div) over the prefill activation.
        #[test]
        #[ignore = "perf bench"]
        fn prefill_route_bench() {
            let dev = metal();
            let (_ffn_norm, moe) = build_moe_layer(0, &dev);
            let x = prefill_input(0x4242, &dev);
            bench(
                &format!(
                    "prefill routing glue, seq={PREFILL_SEQ} \
                     (router matmul + softmax + top-k argsort + gather/renormalize)"
                ),
                || {
                    let (_ids, w) = moe.route(&x).unwrap();
                    read_scalar(&w)
                },
            );
        }

        /// The three expert mm_id dispatches (gate/up/down) at prefill width,
        /// each timed on its own. gate/up read one shared activation per token
        /// (x=[512,1,3072] -> [512,10,1024]); down reads a per-slot activation
        /// (act=[512,10,1024] -> [512,10,3072]). Real top-10 selections from the
        /// production router drive the per-expert row grouping, so the
        /// token-to-expert distribution matches a real chunk. Default `_t`
        /// tensor variant (seq >= MM_ID_MIN_SEQ).
        #[test]
        #[ignore = "perf bench"]
        fn prefill_mm_id_bench() {
            let dev = metal();
            let (_ffn_norm, moe) = build_moe_layer(0, &dev);
            let x = prefill_input(0x4242, &dev);
            let (ids, _w) = moe.route(&x).unwrap();

            let gate = tiled_stack(&dev, EXPERT_FF, HIDDEN, 0x9001);
            let up = tiled_stack(&dev, EXPERT_FF, HIDDEN, 0x9002);
            let down = tiled_stack(&dev, HIDDEN, EXPERT_FF, 0x9003);
            let x_g = x.reshape((PREFILL_SEQ, 1, HIDDEN)).unwrap();
            let act = det_tensor(&[PREFILL_SEQ, TOP_K, EXPERT_FF], 0x9004, 0.5)
                .to_device(&dev)
                .unwrap();

            // gate/up/down are identical-cost matmuls, but timing them in three
            // SEPARATE bench() calls let the first (gate) catch the DVFS boost clock
            // and read low while the later ones ran on the clamped sustained clock.
            // One shared warmup then a single interleaved loop with per-projection
            // timers keeps all three on the same (sustained) clock state; each
            // read_scalar still flushes so the per-projection numbers stay isolated.
            let (warm, iters) = iter_counts();
            let mut sink = 0f32;
            for _ in 0..warm {
                sink += read_scalar(&crate::ops::mul_mm_id(&gate, &x_g, &ids).unwrap());
                sink += read_scalar(&crate::ops::mul_mm_id(&up, &x_g, &ids).unwrap());
                sink += read_scalar(&crate::ops::mul_mm_id(&down, &act, &ids).unwrap());
            }
            let mut gate_t = Vec::with_capacity(iters);
            let mut up_t = Vec::with_capacity(iters);
            let mut down_t = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = Instant::now();
                sink += read_scalar(&crate::ops::mul_mm_id(&gate, &x_g, &ids).unwrap());
                gate_t.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                sink += read_scalar(&crate::ops::mul_mm_id(&up, &x_g, &ids).unwrap());
                up_t.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                sink += read_scalar(&crate::ops::mul_mm_id(&down, &act, &ids).unwrap());
                down_t.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let report = |name: &str, times: &[f64]| {
                let mean = times.iter().sum::<f64>() / times.len() as f64;
                let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
                eprintln!(
                    "{name}: mean {mean:.3} ms/iter, min {min:.3} ms/iter ({iters} iters, sink {sink:.1})"
                );
            };
            report(
                &format!(
                    "prefill mm_id gate matmul, seq={PREFILL_SEQ} (x[512,1,3072] -> [512,10,1024])"
                ),
                &gate_t,
            );
            report(
                &format!(
                    "prefill mm_id up matmul, seq={PREFILL_SEQ} (x[512,1,3072] -> [512,10,1024])"
                ),
                &up_t,
            );
            report(
                &format!(
                    "prefill mm_id down matmul, seq={PREFILL_SEQ} (act[512,10,1024] -> [512,10,3072])"
                ),
                &down_t,
            );
        }

        /// Amortized variant of `prefill_mm_id_bench`: a batch of back-to-back
        /// mm_id calls per timed round with one flush+readback at the round
        /// boundary (outputs held alive until the readback so the buffer pool
        /// cannot recycle one mid-round and inject a false WAW barrier). This
        /// mirrors how a real prefill forward issues the expert gemm — no
        /// per-op sync — and how ggml's `test-backend-ops perf` measures, so
        /// these are the numbers to put next to the fork's mul_mat_id figures.
        #[test]
        #[ignore = "perf bench"]
        fn prefill_mm_id_amortized_bench() {
            let dev = metal();
            let (_ffn_norm, moe) = build_moe_layer(0, &dev);
            let x = prefill_input(0x4242, &dev);
            let (ids, _w) = moe.route(&x).unwrap();

            let gate = tiled_stack(&dev, EXPERT_FF, HIDDEN, 0x9001);
            let up = tiled_stack(&dev, EXPERT_FF, HIDDEN, 0x9002);
            let down = tiled_stack(&dev, HIDDEN, EXPERT_FF, 0x9003);
            let x_g = x.reshape((PREFILL_SEQ, 1, HIDDEN)).unwrap();
            let act = det_tensor(&[PREFILL_SEQ, TOP_K, EXPERT_FF], 0x9004, 0.5)
                .to_device(&dev)
                .unwrap();

            const BATCH: usize = 16;
            let get = |k: &str, d: usize| {
                std::env::var(k)
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(d)
            };
            let (warm, rounds) = (get("XWEN_BENCH_WARMUP", 3), get("XWEN_BENCH_ITERS", 20));

            let projections: [(&str, &crate::gguf::ExpertStack, &Tensor); 3] = [
                ("gate", &gate, &x_g),
                ("up", &up, &x_g),
                ("down", &down, &act),
            ];
            for (name, stack, input) in projections {
                let mut sink = 0f32;
                let mut round = || {
                    let outs: Vec<Tensor> = (0..BATCH)
                        .map(|_| crate::ops::mul_mm_id(stack, input, &ids).unwrap())
                        .collect();
                    read_scalar(outs.last().unwrap())
                };
                for _ in 0..warm {
                    sink += round();
                }
                let mut times = Vec::with_capacity(rounds);
                for _ in 0..rounds {
                    let t = Instant::now();
                    sink += round();
                    times.push(t.elapsed().as_secs_f64() * 1e3 / BATCH as f64);
                }
                let mean = times.iter().sum::<f64>() / times.len() as f64;
                let plateau: f64 =
                    times[rounds / 2..].iter().sum::<f64>() / (rounds - rounds / 2) as f64;
                eprintln!(
                    "prefill mm_id {name} amortized ({BATCH}/round): mean {mean:.3} ms | \
                     plateau {plateau:.3} ms per dispatch (sink {sink:.1})"
                );
            }
        }

        /// The silu*mul + per-column L2 rescale glue over [512,10,1024], the
        /// default `_t` tensor variant's activation path: it stages the down
        /// input as f16, so it runs the L2 guard (moe.rs FusedExperts::forward).
        /// Mirrors that exact chain — silu(gate)*up, then column L2, scale up by
        /// f16's safe headroom and divide out — with synthetic gate/up inputs.
        #[test]
        #[ignore = "perf bench"]
        fn prefill_silu_rescale_bench() {
            let dev = metal();
            let gate = det_tensor(&[PREFILL_SEQ, TOP_K, EXPERT_FF], 0xA001, 0.5)
                .to_device(&dev)
                .unwrap();
            let up = det_tensor(&[PREFILL_SEQ, TOP_K, EXPERT_FF], 0xA002, 0.5)
                .to_device(&dev)
                .unwrap();
            let f16_safe = 32768.0_f64;
            bench(
                &format!(
                    "prefill silu*mul + L2 rescale glue, seq={PREFILL_SEQ} over [512,10,1024]"
                ),
                || {
                    let act = (silu(&gate).unwrap() * up.clone()).unwrap();
                    let col_l2 = act
                        .sqr()
                        .unwrap()
                        .sum_keepdim(2)
                        .unwrap()
                        .sqrt()
                        .unwrap()
                        .clamp(1e-8_f32, 1e30_f32)
                        .unwrap();
                    let act_s = (&act * f16_safe).unwrap().broadcast_div(&col_l2).unwrap();
                    read_scalar(&act_s)
                },
            );
        }

        /// The weighted-combine tail of the rescale branch over [512,10,3072]:
        /// undo the per-column L2 scale, apply the routing weights, sum over the
        /// top-k experts to [512,3072] — the production `ops::combine` fused kernel
        /// (rescale variant), which reads `down` once. Synthetic down / col_l2 /
        /// weights at production shapes.
        #[test]
        #[ignore = "perf bench"]
        fn prefill_combine_bench() {
            let dev = metal();
            let down = det_tensor(&[PREFILL_SEQ, TOP_K, HIDDEN], 0xB001, 0.5)
                .to_device(&dev)
                .unwrap();
            // col_l2 is a positive per-column norm; keep it strictly > 0.
            let col_l2 = det_tensor(&[PREFILL_SEQ, TOP_K, 1], 0xB002, 0.5)
                .abs()
                .unwrap()
                .affine(1.0, 0.1)
                .unwrap()
                .to_device(&dev)
                .unwrap();
            let weights = det_tensor(&[PREFILL_SEQ, TOP_K], 0xB003, 0.25)
                .to_device(&dev)
                .unwrap();
            bench(
                &format!(
                    "prefill weighted combine (fused), seq={PREFILL_SEQ} over [512,10,3072] -> [512,3072]"
                ),
                || read_scalar(&crate::ops::combine(&down, Some(&col_l2), &weights).unwrap()),
            );
        }

        /// The always-on shared-expert SwiGLU (q4_K QLinear gate/up/down) at
        /// prefill width, through the production `SharedExpert::forward`.
        #[test]
        #[ignore = "perf bench"]
        fn prefill_shared_expert_bench() {
            let dev = metal();
            let (_ffn_norm, moe) = build_moe_layer(0, &dev);
            let x = prefill_input(0x4242, &dev);
            bench(
                &format!("prefill shared expert SwiGLU (q4_K), seq={PREFILL_SEQ}"),
                || read_scalar(&moe.shared.forward(&x).unwrap()),
            );
        }

        /// The per-layer norm/residual glue model.rs wraps each block with:
        /// attn_norm -> +attn, ffn_norm -> +moe — 2 rms_norm + 2 adds at
        /// [512,3072]. Synthetic block outputs stand in for the attention and
        /// MoE contributions; only the norm+add dispatch cost is being priced.
        #[test]
        #[ignore = "perf bench"]
        fn prefill_norm_residual_bench() {
            let dev = metal();
            let attn_norm = norm(HIDDEN, 0xC001, &dev);
            let ffn_norm = norm(HIDDEN, 0xC002, &dev);
            let x0 = prefill_input(0x4242, &dev);
            let attn_out = prefill_input(0xC003, &dev);
            let moe_out = prefill_input(0xC004, &dev);
            bench(
                &format!(
                    "prefill norm+residual glue, seq={PREFILL_SEQ} (2 rms_norm + 2 adds at [512,3072])"
                ),
                || {
                    let _attn_normed = attn_norm.forward(&x0).unwrap();
                    let x = (&x0 + &attn_out).unwrap();
                    let _ffn_normed = ffn_norm.forward(&x).unwrap();
                    let x = (&x + &moe_out).unwrap();
                    read_scalar(&x)
                },
            );
        }

        /// Whole synthetic MoE block at prefill width through the production
        /// `MoeBlock::forward`: ffn_norm -> route -> mm_id experts (default
        /// tensor variant, seq >= MM_ID_MIN_SEQ) -> silu*mul + rescale ->
        /// weighted combine -> +shared expert. The sum-check for the isolated
        /// parts above; prints a 10-iter-group time series so the boost ->
        /// sustained plateau is visible.
        #[test]
        #[ignore = "perf bench"]
        fn prefill_moe_block_bench() {
            let dev = metal();
            let (ffn_norm, moe) = build_moe_layer(0, &dev);
            let x0 = prefill_input(0x4242, &dev);
            let (warm, iters) = iter_counts();
            let step = || -> f32 {
                let normed = ffn_norm.forward(&x0).unwrap();
                read_scalar(&moe.forward(&normed).unwrap())
            };
            let mut sink = 0f32;
            for _ in 0..warm {
                sink += step();
            }
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = Instant::now();
                sink += step();
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
            eprintln!(
                "prefill MoE block (ffn_norm + route + mm_id experts + silu/rescale + combine + \
                 shared), seq={PREFILL_SEQ}: mean {mean:.3} ms/iter, min {min:.3} ms/iter \
                 ({iters} iters, sink {sink:.1})"
            );
            let chunk = (iters / 10).max(1);
            let series: Vec<String> = times
                .chunks(chunk)
                .map(|c| format!("{:.2}", c.iter().sum::<f64>() / c.len() as f64))
                .collect();
            eprintln!(
                "time series (means of {chunk}-iter groups, ms): [{}]",
                series.join(", ")
            );
        }

        /// Isolation timing for the 27B's dense SwiGLU FFN, which every one of
        /// its 64 layers runs and which carries the majority of the model's
        /// prefill FLOPs (3 x 2 x T x 5120 x 17408 per layer).
        ///
        /// The three matmuls go through `QLinear` -> candle's
        /// `kernel_mul_mm_q4_K_f32`; unlike the attention and DeltaNet
        /// projections, the FFN weights have no dequantized f16 plane, so the
        /// `f16 plane` rows below price the counterfactual against the same
        /// shapes on the vendored cooperative-tensor gemm. `silu*up` is the pair
        /// of eager candle ops between them.
        ///
        /// Achieved TFLOP/s is printed next to every matmul: `2 * T * K * N`
        /// over the plateau time. Compare against the ~26 TFLOP/s llama.cpp
        /// sustains over a whole 27B prefill on this machine.
        ///
        /// `#[ignore]`d — run on a `pgrep`-verified free GPU with:
        ///   cargo test --release -p xwen dense_ffn_prefill_timing -- --ignored --nocapture
        #[test]
        #[ignore = "perf bench"]
        fn dense_ffn_prefill_timing() {
            const HIDDEN: usize = 5120;
            const FF: usize = 17408;
            const LAYERS: usize = 64;

            let device = metal();
            let Device::Metal(mdev) = &device else {
                unreachable!("metal() returned a non-Metal device")
            };
            let (warm, iters) = iter_counts();
            let (warm, iters) = (warm.min(3), iters.min(10));

            let sync_bench = |f: &mut dyn FnMut()| -> f64 {
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

            let q4k = |rows: usize, cols: usize, seed: u64| {
                let t = det_tensor(&[rows, cols], seed, 0.5);
                QLinear::from_qtensor(Arc::new(
                    QTensor::quantize_onto(&t, GgmlDType::Q4K, &device).unwrap(),
                ))
                .unwrap()
            };
            let plane = |rows: usize, cols: usize, seed: u64| {
                det_tensor(&[rows, cols], seed, 0.5)
                    .to_device(&device)
                    .unwrap()
                    .to_dtype(DType::F16)
                    .unwrap()
            };

            let (gate, up, down) = (
                q4k(FF, HIDDEN, 0x401),
                q4k(FF, HIDDEN, 0x402),
                q4k(HIDDEN, FF, 0x403),
            );
            let (gate16, down16) = (plane(FF, HIDDEN, 0x401), plane(HIDDEN, FF, 0x403));

            // 512 is the production chunk; the other two are the whole-prompt
            // lengths the bench fixtures reach, kept to show the stage is linear
            // in T (it has no position-dependent term).
            for &t in &[512usize, 880, 3851] {
                let x = det_tensor(&[t, HIDDEN], 0x404, 0.5)
                    .to_device(&device)
                    .unwrap();
                let h = det_tensor(&[t, FF], 0x405, 0.5).to_device(&device).unwrap();
                let tf =
                    |ms: f64, k: usize, n: usize| 2.0 * t as f64 * k as f64 * n as f64 / (ms * 1e9);

                let t_gate = sync_bench(&mut || {
                    gate.forward(&x).unwrap();
                });
                let t_down = sync_bench(&mut || {
                    down.forward(&h).unwrap();
                });
                let t_gate16 = sync_bench(&mut || {
                    crate::ops::matmul_f16(&gate16, &x).unwrap();
                });
                let t_down16 = sync_bench(&mut || {
                    crate::ops::matmul_f16(&down16, &h).unwrap();
                });
                let t_act = sync_bench(&mut || {
                    let g = silu(&h).unwrap();
                    (&g * &h).unwrap();
                });
                let t_block = sync_bench(&mut || {
                    swiglu(&gate, &up, &down, &x).unwrap();
                });
                // A real forward issues all 64 layers' FFN dispatches without
                // waiting in between, so the per-call commit-and-wait the
                // measurements above each pay is overhead the model never sees.
                // Batch BATCH blocks per sync and hold the outputs alive so the
                // allocator pool cannot inject a false write-after-write barrier
                // between them; this is the rate to multiply by the layer count.
                const BATCH: usize = 8;
                let t_amort = sync_bench(&mut || {
                    let mut keep = Vec::with_capacity(BATCH);
                    for _ in 0..BATCH {
                        keep.push(swiglu(&gate, &up, &down, &x).unwrap());
                    }
                }) / BATCH as f64;

                eprintln!("--- dense FFN, 27B shapes, T={t} ---");
                eprintln!(
                    "  q4k gate/up  [{t},{HIDDEN}]x[{HIDDEN},{FF}]  {t_gate:7.3} ms  {:6.2} TFLOP/s",
                    tf(t_gate, HIDDEN, FF)
                );
                eprintln!(
                    "  q4k down     [{t},{FF}]x[{FF},{HIDDEN}]  {t_down:7.3} ms  {:6.2} TFLOP/s",
                    tf(t_down, FF, HIDDEN)
                );
                eprintln!(
                    "  f16 plane gate                          {t_gate16:7.3} ms  {:6.2} TFLOP/s",
                    tf(t_gate16, HIDDEN, FF)
                );
                eprintln!(
                    "  f16 plane down                          {t_down16:7.3} ms  {:6.2} TFLOP/s",
                    tf(t_down16, FF, HIDDEN)
                );
                eprintln!("  silu*up (eager)                         {t_act:7.3} ms");
                eprintln!(
                    "  whole swiglu                            {t_block:7.3} ms  {:6.2} TFLOP/s \
                     => {:.1} ms for {LAYERS} layers",
                    3.0 * 2.0 * t as f64 * HIDDEN as f64 * FF as f64 / (t_block * 1e9),
                    t_block * LAYERS as f64
                );
                eprintln!(
                    "  swiglu amortized                        {t_amort:7.3} ms  {:6.2} TFLOP/s \
                     => {:.1} ms for {LAYERS} layers",
                    3.0 * 2.0 * t as f64 * HIDDEN as f64 * FF as f64 / (t_amort * 1e9),
                    t_amort * LAYERS as f64
                );
            }
        }
    }
}
