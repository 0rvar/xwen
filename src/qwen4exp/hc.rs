//! Device-side hyper-connections: the residual carrier that replaces the
//! `attn_norm` / `post_attention_norm` / `output_norm` chain in qwen4exp.
//!
//! qwen4exp carries `hc_count` parallel residual streams concatenated into one
//! `hc_count * hidden` row instead of a single `hidden`-wide residual. Every
//! attention and every MLP block sits behind one of these gates: it READS a
//! single `hidden`-wide vector out of the carrier ([`HcRead::read`]) and WRITES
//! its output back into all streams with a per-stream strength
//! ([`hc_write`]). The model tail runs the same read path with no injection
//! head to collapse the carrier down to `hidden` for the lm_head.
//!
//! [`crate::qwen4exp::ref_hc`] is the frozen CPU f32 oracle for all of this and
//! the tests below grade against it; llama.cpp's `build_hc_mix` /
//! `build_hc_combine` (`reference/llama.cpp/src/models/qwen4exp.cpp`) is the
//! executable ground truth both follow.
//!
//! Two implementations live here. The CLASSIC one is a plain candle op per
//! step — three dispatched matmuls plus a dozen elementwise passes per read,
//! three more per write — and is kept verbatim as the `XWEN_HC_CLASSIC`
//! kill-switch and provenance anchor. The DEFAULT one keeps the two Q8_0
//! bottleneck matmuls on `QLinear` and replaces everything around them with the
//! four vendored kernels in `src/ops/hc.metal`: the grouped norm and the
//! injection head in one threadgroup per token, the bottleneck activation, the
//! mix-and-collapse, and the write-back. Four gates per layer over 48 layers is
//! why: the candle chain is 34% of prefill wall.
//!
//! The fused path is taken only for a geometry the kernels cover
//! (`ops::hc_norm_supported`); anything else falls back rather than failing.

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Tensor};
use candle_nn::ops::{sigmoid, silu};

use crate::config::XwenConfig;
use crate::gguf::{QLinear, Weights};

/// One hyper-connection read gate: the grouped norm, the low-rank mix-weight
/// bottleneck, and — for a block gate, but not for the tail mixer — the
/// injection head that scores how strongly each stream takes the block output.
///
/// GGUF tensors, per `reference/llama.cpp/src/llama-arch.cpp`:
/// `blk.N.hc_attn_{norm,down,up,inject}.weight`,
/// `blk.N.hc_ffn_{norm,down,up,inject}.weight` and the tail's
/// `output_hc_{norm,down,up}.weight` (no inject — there is no
/// `output_norm` on this architecture; the tail mixer carries it).
pub struct HcRead {
    /// Number of parallel residual streams.
    pub hc_count: usize,
    /// Width of one stream — the model's hidden size.
    pub hidden: usize,
    /// Width of the mix-weight bottleneck.
    pub low_rank: usize,
    /// Grouped-norm weight, `[hc_count * hidden]` f32, multiply-ready (the GGUF
    /// converter already baked the `1 +` in — never add it here).
    norm_w: Tensor,
    /// `[low_rank, hc_count * hidden]`.
    down: QLinear,
    /// `[hc_count * hidden, low_rank]`.
    up: QLinear,
    /// `[hc_count, hc_count * hidden]`; absent on the tail mixer.
    inject: Option<QLinear>,
    /// The same injection head as a dense f32 `[hc_count, hc_count * hidden]`
    /// tensor, which is what the fused norm kernel contracts against (it reads
    /// raw f32 rows, not a `QMatMul`). Present exactly when `inject` is.
    ///
    /// A deliberate duplicate rather than a replacement: the classic path must
    /// keep the exact `QLinear` chain it was blessed with, whatever the file
    /// stored the head at. The head is the smallest weight in the gate —
    /// `hc_count * hc_count * hidden` f32, 160 KiB at the real geometry, ~16 MiB
    /// across the whole stack.
    inject_dense: Option<Tensor>,
    /// RMS epsilon, the model-wide `rms_norm_eps`.
    eps: f64,
}

impl HcRead {
    /// Width of the residual carrier, `hc_count * hidden`.
    pub fn width(&self) -> usize {
        self.hc_count * self.hidden
    }

    /// Whether this gate has an injection head — true for a block gate, false
    /// for the tail mixer.
    pub fn has_inject(&self) -> bool {
        self.inject.is_some()
    }

    /// Loads `{prefix}_{norm,down,up[,inject]}.weight` from `w`, which the
    /// caller has already prefixed (`w.pp("blk.7")` for a block gate, the root
    /// handle for the tail).
    ///
    /// `prefix` is `"hc_attn"`, `"hc_ffn"` or `"output_hc"`. `with_inject` is
    /// false only for the tail mixer.
    pub fn load(w: &Weights, prefix: &str, cfg: &XwenConfig, with_inject: bool) -> Result<Self> {
        let hc = cfg
            .qwen4exp
            .as_ref()
            .context("hyper-connections need the qwen4exp config block")?;
        let hc_count = hc.hc_count;
        let low_rank = hc.hc_low_rank;
        let hidden = cfg.hidden;

        let norm_w = w.dense_f32(&format!("{prefix}_norm"))?;
        // Loaded through the buffer-retaining path so each carries a plane for
        // the fused read's prefill gemm (`forward_gemm`); the plane views the
        // QMatMul allocation, so it costs no device memory (each of the two
        // q8_0 tensors is ~3.3 MB). `without_mv_ext` keeps `forward` off the
        // small-batch window the plane would otherwise open at 2..8 tokens, so
        // every hc path — `XWEN_HC_CLASSIC` included — stays on QMatMul there.
        let down = w
            .qlinear_with_buffer(&format!("{prefix}_down"))?
            .0
            .without_mv_ext();
        let up = w
            .qlinear_with_buffer(&format!("{prefix}_up"))?
            .0
            .without_mv_ext();
        let inject = if with_inject {
            Some(w.qlinear(&format!("{prefix}_inject"))?)
        } else {
            None
        };
        // Same tensor name the `QLinear` above resolved, dequantized once for
        // the fused kernel. `dense_f32` yields f32 whatever the file stored.
        let inject_dense = if with_inject {
            Some(w.dense_f32(&format!("{prefix}_inject"))?)
        } else {
            None
        };

        Self::assemble(
            hc_count,
            hidden,
            low_rank,
            cfg.rms_eps,
            norm_w,
            down,
            up,
            inject,
            inject_dense,
        )
    }

    /// A gate over weights the caller already holds as dense f32 device tensors
    /// (`[out, in]` row-major, the layout GGUF yields and `ref_hc` documents),
    /// for tests and fixtures rather than a GGUF.
    ///
    /// The projections go through the same [`QLinear`] the GGUF path uses: an
    /// F32-typed `QTensor` is a memcpy of the source and `QMatMul` dequantizes
    /// it straight back to a dense f32 matmul, so the only difference from a
    /// loaded gate is the stored precision of the weights.
    // Four weights and four dimensions is what the gate is; bundling them into a
    // struct would only move the same list one level out.
    #[allow(clippy::too_many_arguments)]
    pub fn from_tensors(
        hc_count: usize,
        hidden: usize,
        low_rank: usize,
        eps: f64,
        norm_w: Tensor,
        down_w: &Tensor,
        up_w: &Tensor,
        inject_w: Option<&Tensor>,
    ) -> Result<Self> {
        let lin = |t: &Tensor| -> Result<QLinear> {
            let qt = candle_core::quantized::QTensor::quantize(
                &t.to_dtype(DType::F32)?,
                candle_core::quantized::GgmlDType::F32,
            )?;
            QLinear::from_qtensor(std::sync::Arc::new(qt))
        };
        let inject = inject_w.map(lin).transpose()?;
        let inject_dense = inject_w
            .map(|t| t.to_dtype(DType::F32)?.contiguous())
            .transpose()?;
        Self::assemble(
            hc_count,
            hidden,
            low_rank,
            eps,
            norm_w,
            lin(down_w)?,
            lin(up_w)?,
            inject,
            inject_dense,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(
        hc_count: usize,
        hidden: usize,
        low_rank: usize,
        eps: f64,
        norm_w: Tensor,
        down: QLinear,
        up: QLinear,
        inject: Option<QLinear>,
        inject_dense: Option<Tensor>,
    ) -> Result<Self> {
        ensure!(
            hc_count > 0 && hidden > 0 && low_rank > 0,
            "hyper-connection geometry must be positive, got hc_count {hc_count}, \
             hidden {hidden}, low_rank {low_rank}"
        );
        let width = hc_count * hidden;
        let norm_w = norm_w.to_dtype(DType::F32)?.flatten_all()?;
        ensure!(
            norm_w.elem_count() == width,
            "hyper-connection norm weight is {} wide, expected hc_count * hidden = {width}",
            norm_w.elem_count()
        );
        // Orientation guards. A transposed read of `up` would be undetectable by
        // element count alone unless low_rank == width, and the down/inject pair
        // share an input width, so check every axis explicitly.
        ensure!(
            down.in_dim == width && down.out_dim == low_rank,
            "hyper-connection down projection is [{}, {}], expected [{low_rank}, {width}]",
            down.out_dim,
            down.in_dim
        );
        ensure!(
            up.in_dim == low_rank && up.out_dim == width,
            "hyper-connection up projection is [{}, {}], expected [{width}, {low_rank}]",
            up.out_dim,
            up.in_dim
        );
        if let Some(inject) = &inject {
            ensure!(
                inject.in_dim == width && inject.out_dim == hc_count,
                "hyper-connection injection head is [{}, {}], expected [{hc_count}, {width}]",
                inject.out_dim,
                inject.in_dim
            );
        }
        ensure!(
            inject.is_some() == inject_dense.is_some(),
            "the injection head and its dense copy must be present together"
        );
        if let Some(dense) = &inject_dense {
            ensure!(
                dense.dims() == [hc_count, width],
                "the dense injection head is {:?}, expected [{hc_count}, {width}]",
                dense.dims()
            );
        }
        Ok(Self {
            hc_count,
            hidden,
            low_rank,
            norm_w,
            down,
            up,
            inject,
            inject_dense,
            eps,
        })
    }

    /// RMS norm whose statistics are taken per stream, with the weight applied
    /// over the FULL carrier width.
    ///
    /// Grouping is load-bearing: the streams can sit decades apart in scale and
    /// one set of statistics over the whole row would flatten that (the
    /// `grouped_rmsnorm` fixture pins the divergence). llama.cpp gets the same
    /// effect by keeping the carrier as `[n_embd, hc, nt]` and letting
    /// `ggml_rms_norm` reduce over `ne[0]`; the reshape here is that view.
    ///
    /// Statistics accumulate in f32 on device where `ref_hc` uses f64. The two
    /// agree to well inside the fixture tolerances at these widths; the f64 is
    /// the oracle's insurance, not a requirement of the math.
    fn grouped_norm(&self, stream: &Tensor) -> Result<Tensor> {
        let (n, width) = stream.dims2()?;
        ensure!(
            width == self.width(),
            "carrier is {width} wide, expected hc_count * hidden = {}",
            self.width()
        );
        let grouped = stream.reshape((n, self.hc_count, self.hidden))?;
        let inv_rms = grouped
            .sqr()?
            .mean_keepdim(2)?
            .affine(1.0, self.eps)?
            .sqrt()?
            .recip()?;
        Ok(grouped
            .broadcast_mul(&inv_rms)?
            .reshape((n, width))?
            .broadcast_mul(&self.norm_w)?)
    }

    /// The read path: grouped-norm the carrier, build the per-element mix
    /// weights through the low-rank bottleneck, and take the MEAN over streams.
    ///
    /// `stream` is `[n, hc_count * hidden]` f32. Returns `(mixed [n, hidden],
    /// inject [n, hc_count])`, where `mixed` is the vector the block runs on and
    /// `inject` — `Some` exactly when this gate has an injection head — is the
    /// per-stream write strength [`hc_write`] scales the block output by. It is
    /// `2·sigmoid(·)`, so it spans `(0, 2)` and is centered on 1: an untrained
    /// gate leaves the carrier's scale alone.
    ///
    /// The normed carrier is deliberately not returned. It feeds the bottleneck
    /// and the injection head and nothing else — the write-back lands on the RAW
    /// stream, because normalization feeds the block, it does not replace the
    /// residual.
    pub fn read(&self, stream: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        ensure!(
            stream.dtype() == DType::F32,
            "the hyper-connection carrier is f32, got {:?}",
            stream.dtype()
        );
        let (n, _) = stream.dims2()?;
        if self.fused(stream) {
            return self.read_fused(stream);
        }
        let normed = self.grouped_norm(stream)?;

        // Low-rank bottleneck. The 1/hc_count scale is on the bottleneck
        // ACTIVATION, before the silu — the carrier is a sum over hc_count
        // streams, so the projection's output grows with the stream count.
        let low = silu(
            &self
                .down
                .forward(&normed)?
                .affine(1.0 / self.hc_count as f64, 0.0)?,
        )?;
        let mix = sigmoid(&self.up.forward(&low)?)?;

        let mixed = (mix * &normed)?
            .reshape((n, self.hc_count, self.hidden))?
            .sum(1)?
            .affine(1.0 / self.hc_count as f64, 0.0)?;

        let inject = match &self.inject {
            Some(head) => Some(
                sigmoid(
                    &head
                        .forward(&normed)?
                        .affine(1.0 / self.hc_count as f64, 0.0)?,
                )?
                .affine(2.0, 0.0)?,
            ),
            None => None,
        };
        Ok((mixed, inject))
    }

    /// Whether this gate takes the vendored kernels rather than the candle
    /// chain: a Metal device, the kill-switch unset, a geometry the norm
    /// kernel's launch covers, and WEIGHTS the kernel can bind. The bounds are
    /// the kernel's, so a gate outside them falls back rather than failing —
    /// which means every condition `hc_norm` hard-errors on has to be asked
    /// here, the operands included. The carrier alone is not enough: the norm
    /// weight and the dense injection head are bound the same way and are the
    /// two operands the constructor validates by shape but not by dtype,
    /// contiguity, or device.
    fn fused(&self, stream: &Tensor) -> bool {
        !crate::ops::hc_classic()
            && stream.device().is_metal()
            && stream.is_contiguous()
            && crate::ops::hc_norm_supported(self.hc_count, self.hidden)
            && Self::bindable(&self.norm_w, stream)
            && self
                .inject_dense
                .as_ref()
                .is_none_or(|head| Self::bindable(head, stream))
    }

    /// An operand `hc_norm` can bind: contiguous f32 on the carrier's own
    /// device. Shape is not re-checked — the constructor pins it, and the
    /// dispatch would catch a mismatch as an error rather than as garbage.
    fn bindable(t: &Tensor, stream: &Tensor) -> bool {
        t.dtype() == DType::F32 && t.is_contiguous() && stream.device().same_device(t.device())
    }

    /// [`read`](Self::read) through the vendored kernels: one threadgroup per
    /// token does the grouped norm and the injection head together, the
    /// bottleneck keeps its two `QLinear` matmuls, and the mix-and-collapse is
    /// one pass over the up projection's RAW logits (the sigmoid is folded into
    /// it, so no full-width sigmoid pass is materialized).
    fn read_fused(&self, stream: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let rows = stream.dim(0)?;
        let (normed, inject) = crate::ops::dup(crate::ops::DupStage::Hc, rows, || {
            crate::ops::hc_norm(
                stream,
                &self.norm_w,
                self.inject_dense.as_ref(),
                self.hc_count,
                self.hidden,
                self.eps as f32,
            )
        })?;
        // The bottleneck's two matmuls take the vendored dense gemm at prefill
        // (`QLinear::forward_gemm`, above `dense_mm_min_seq()`), each arm
        // separately revocable by `XWEN_HC_GEMM_QMATMUL` because their shapes
        // differ (down k = hc_count*hidden, up k = low_rank). Decode is the same
        // `QMatMul` gemv either way.
        let arms = crate::ops::hc_gemm_qmatmul();
        let low_in = crate::ops::dup(crate::ops::DupStage::Hc, rows, || {
            crate::ops::dup(crate::ops::DupStage::HcGemm, rows, || {
                if arms.down_on_qmatmul() {
                    Ok(self.down.forward(&normed)?)
                } else {
                    Ok(self.down.forward_gemm(&normed)?)
                }
            })
        })?;
        let low = crate::ops::dup(crate::ops::DupStage::Hc, rows, || {
            crate::ops::hc_silu_quarter(&low_in, self.hc_count)
        })?;
        let up = crate::ops::dup(crate::ops::DupStage::Hc, rows, || {
            crate::ops::dup(crate::ops::DupStage::HcGemm, rows, || {
                if arms.up_on_qmatmul() {
                    Ok(self.up.forward(&low)?)
                } else {
                    Ok(self.up.forward_gemm(&low)?)
                }
            })
        })?;
        let mixed = crate::ops::dup(crate::ops::DupStage::Hc, rows, || {
            crate::ops::hc_mix(&up, &normed, self.hc_count, self.hidden)
        })?;
        Ok((mixed, inject))
    }

    /// [`read`](Self::read) for the tail mixer, where there is no injection head
    /// and the caller wants the `[n, hidden]` vector alone.
    pub fn mix(&self, stream: &Tensor) -> Result<Tensor> {
        let (mixed, inject) = self.read(stream)?;
        if inject.is_some() {
            bail!("mix() called on a gate that has an injection head; use read()");
        }
        Ok(mixed)
    }
}

/// Writes one block's output back into the RAW carrier:
/// `stream[s] += block_out * inject[s]` for every stream `s`.
///
/// `stream` is `[n, hc_count * hidden]`, `block_out` is `[n, hidden]` and
/// `inject` is `[n, hc_count]` — the `2·sigmoid` strengths [`HcRead::read`]
/// returned. The write is purely additive, so a zero block output leaves the
/// carrier bit-identical whatever the strengths are.
///
/// A free function rather than a method: the write consumes the injection
/// weights the read produced, and takes nothing else from the gate.
pub fn hc_write(stream: &Tensor, block_out: &Tensor, inject: &Tensor) -> Result<Tensor> {
    let (n, width) = stream.dims2()?;
    let (n_out, hidden) = block_out.dims2()?;
    let (n_inj, hc_count) = inject.dims2()?;
    ensure!(
        n == n_out && n == n_inj,
        "row counts differ: carrier {n}, block output {n_out}, injection {n_inj}"
    );
    ensure!(
        hc_count * hidden == width,
        "carrier is {width} wide but the block output ({hidden}) times the stream \
         count ({hc_count}) is {}",
        hc_count * hidden
    );
    // One pass, out of place, bit-identical to the chain below. The kernel
    // takes contiguous f32 operands only; anything else keeps the chain rather
    // than failing.
    if !crate::ops::hc_classic()
        && stream.device().is_metal()
        && [stream, block_out, inject]
            .iter()
            .all(|t| t.dtype() == DType::F32 && t.is_contiguous())
    {
        return crate::ops::dup(crate::ops::DupStage::Hc, n, || {
            crate::ops::hc_write(stream, block_out, inject)
        });
    }
    let scaled = block_out
        .reshape((n, 1, hidden))?
        .broadcast_mul(&inject.reshape((n, hc_count, 1))?)?;
    Ok((stream.reshape((n, hc_count, hidden))? + scaled)?.reshape((n, width))?)
}

/// Seeds the residual carrier from the token embeddings: `[n, hidden]` becomes
/// `[n, hc_count * hidden]` by TILING the embedding, `[x, x, x, x]`.
///
/// Tiling, not interleaving. Every consumer downstream reads stream `s` as the
/// contiguous slice `s * hidden .. (s + 1) * hidden`, so an interleaved seed
/// would put one element of every stream where the first stream belongs and
/// still typecheck all the way to the logits.
pub fn seed_stream(embed: &Tensor, hc_count: usize) -> Result<Tensor> {
    ensure!(hc_count > 0, "the carrier needs at least one stream");
    let (_n, _hidden) = embed.dims2()?;
    let copies = vec![embed; hc_count];
    Ok(Tensor::cat(&copies, 1)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen4exp::ref_hc::{GatedResidualRef, HcMixerRef};
    use candle_core::Device;
    use serde_json::Value;

    const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/qwen4exp/");

    /// The ops here are device ops; Metal is the target and the only place the
    /// production path runs, so a machine without it cannot grade this.
    fn dev() -> Device {
        Device::new_metal(0).expect("qwen4exp hc tests need the Metal device")
    }

    fn fixture(name: &str) -> Value {
        let path = format!("{FIXTURE_DIR}{name}");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path}: {e}"))
    }

    /// Fixture floats are the shortest f64 repr of an exact f32, so parsing as
    /// f64 and casting recovers the original bits.
    fn vec1(v: &Value) -> Vec<f32> {
        v.as_array()
            .expect("expected an array")
            .iter()
            .map(|x| x.as_f64().expect("expected a number") as f32)
            .collect()
    }

    /// A `[rows][cols]` fixture array flattened row-major.
    fn vec2(v: &Value) -> Vec<f32> {
        v.as_array()
            .expect("expected an array of rows")
            .iter()
            .flat_map(vec1)
            .collect()
    }

    fn f32_field(v: &Value, key: &str) -> f32 {
        v[key]
            .as_f64()
            .unwrap_or_else(|| panic!("missing float field {key}")) as f32
    }

    fn tensor2(data: &[f32], rows: usize, cols: usize, dev: &Device) -> Tensor {
        assert_eq!(data.len(), rows * cols, "tensor2 shape mismatch");
        Tensor::from_slice(data, (rows, cols), dev).unwrap()
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        let mut worst = 0.0f32;
        let mut worst_at = 0usize;
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let d = (g - w).abs();
            if d > worst {
                worst = d;
                worst_at = i;
            }
        }
        assert!(
            worst <= tol,
            "{what}: max abs deviation {worst} > {tol} at index {worst_at} \
             (got {}, want {})",
            got[worst_at],
            want[worst_at]
        );
    }

    /// Relative L2 error, the measure the ops tests grade device kernels by.
    fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len(), "rel_l2: length mismatch");
        let num: f64 = got
            .iter()
            .zip(want)
            .map(|(g, w)| ((g - w) as f64).powi(2))
            .sum();
        let den: f64 = want.iter().map(|w| (*w as f64).powi(2)).sum();
        (num.sqrt() / den.sqrt().max(f64::MIN_POSITIVE)) as f32
    }

    /// Deterministic pseudo-random f32s (LCG, no deps), in roughly [-0.5, 0.5].
    fn seeded(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    /// The elementwise floor the fixture reports, widened as its README
    /// suggests.
    fn suggested_tol(floor: f32) -> f32 {
        (10.0 * floor).max(1e-6)
    }

    fn block_gate_from_fixture(f: &Value, dev: &Device) -> HcRead {
        let cfg = &f["config"];
        let hc_count = cfg["hc_count"].as_u64().unwrap() as usize;
        let hidden = cfg["hidden_size"].as_u64().unwrap() as usize;
        let low_rank = cfg["hc_lowrank"].as_u64().unwrap() as usize;
        let eps = cfg["rms_norm_eps"].as_f64().unwrap();
        let width = hc_count * hidden;
        let w = &f["weights"];
        HcRead::from_tensors(
            hc_count,
            hidden,
            low_rank,
            eps,
            Tensor::from_slice(&vec1(&w["hc_norm_weight_mult"]), width, dev).unwrap(),
            &tensor2(&vec2(&w["input_mix_weight_down"]), low_rank, width, dev),
            &tensor2(&vec2(&w["input_mix_weight_up"]), width, low_rank, dev),
            Some(&tensor2(
                &vec2(&w["block_inject_weight"]),
                hc_count,
                width,
                dev,
            )),
        )
        .unwrap()
    }

    /// The whole gate against the transformers-generated fixture: read (mixed
    /// vector and injection weights), write-back with an identity block, and the
    /// injection-head-less tail mixer.
    #[test]
    fn hc_read_write_matches_fixture() {
        let dev = dev();
        let f = fixture("gated_residual.json");
        let gate = block_gate_from_fixture(&f, &dev);
        let n = f["input_stream"].as_array().unwrap().len();
        let stream = tensor2(&vec2(&f["input_stream"]), n, gate.width(), &dev);

        let (mixed, inject) = gate.read(&stream).unwrap();
        let inject = inject.expect("a block gate has an injection head");
        assert_eq!(mixed.dims(), &[n, gate.hidden]);
        assert_eq!(inject.dims(), &[n, gate.hc_count]);
        assert_close(
            &flat(&mixed),
            &vec2(&f["mixed_output"]),
            suggested_tol(f32_field(&f, "f64_delta_mixed")),
            "hc read mixed_output",
        );
        assert_close(
            &flat(&inject),
            &vec2(&f["injection_weights"]),
            suggested_tol(f32_field(&f, "f64_delta_injection")),
            "hc read injection_weights",
        );

        // The fixture's block is the identity, so the block output is the mixed
        // vector itself, and the write lands on the RAW carrier.
        let written = hc_write(&stream, &mixed, &inject).unwrap();
        assert_close(
            &flat(&written),
            &vec2(&f["stream_out_identity_block"]),
            suggested_tol(f32_field(&f, "f64_delta_stream_out")),
            "hc_write stream_out_identity_block",
        );

        let tm = &f["tail_mixer"];
        let cfg = &f["config"];
        let hc_count = cfg["hc_count"].as_u64().unwrap() as usize;
        let hidden = cfg["hidden_size"].as_u64().unwrap() as usize;
        let low_rank = cfg["hc_lowrank"].as_u64().unwrap() as usize;
        let width = hc_count * hidden;
        let tw = &tm["weights"];
        let mixer = HcRead::from_tensors(
            hc_count,
            hidden,
            low_rank,
            cfg["rms_norm_eps"].as_f64().unwrap(),
            Tensor::from_slice(&vec1(&tw["hc_norm_weight_mult"]), width, &dev).unwrap(),
            &tensor2(&vec2(&tw["input_mix_weight_down"]), low_rank, width, &dev),
            &tensor2(&vec2(&tw["input_mix_weight_up"]), width, low_rank, &dev),
            None,
        )
        .unwrap();
        assert!(!mixer.has_inject());
        let tn = tm["input_stream"].as_array().unwrap().len();
        let tail_in = tensor2(&vec2(&tm["input_stream"]), tn, width, &dev);
        assert_close(
            &flat(&mixer.mix(&tail_in).unwrap()),
            &vec2(&tm["mixed_output"]),
            suggested_tol(f32_field(&f, "f64_delta_tail_mixed")),
            "tail mixer mixed_output",
        );
    }

    /// A zero block output leaves the carrier bit-identical no matter what the
    /// injection weights say — the write is purely additive.
    #[test]
    fn hc_write_of_zero_is_identity() {
        let dev = dev();
        let f = fixture("gated_residual.json");
        let gate = block_gate_from_fixture(&f, &dev);
        let n = f["input_stream"].as_array().unwrap().len();
        let stream = tensor2(&vec2(&f["input_stream"]), n, gate.width(), &dev);
        let (_, inject) = gate.read(&stream).unwrap();
        let zeros = Tensor::zeros((n, gate.hidden), DType::F32, &dev).unwrap();
        let out = hc_write(&stream, &zeros, &inject.unwrap()).unwrap();
        assert_eq!(flat(&out), flat(&stream));
    }

    /// The device path against the frozen CPU oracle at the checkpoint's real
    /// geometry (hidden 2560, hc_count 4, low_rank 320), where the bottleneck
    /// reduces over 10240 elements and any accumulation difference would show.
    #[test]
    fn hc_matches_reference_at_real_geometry() {
        let dev = dev();
        let (hc_count, hidden, low_rank, n) = (4usize, 2560usize, 320usize, 5usize);
        let width = hc_count * hidden;
        let eps = 1e-6f64;

        // Norm weights sit near 1 (the converter's baked `1 + w`), everything
        // else is zero-centered.
        let norm_w: Vec<f32> = seeded(width, 11).iter().map(|v| 1.0 + 0.1 * v).collect();
        let down_w = seeded(low_rank * width, 22);
        let up_w = seeded(width * low_rank, 33);
        let inject_w = seeded(hc_count * width, 44);
        let stream_v = seeded(n * width, 55);
        let block_out_v = seeded(n * hidden, 66);

        let reference = GatedResidualRef {
            hc_count,
            hidden,
            low_rank,
            norm_w: norm_w.clone(),
            down_w: down_w.clone(),
            up_w: up_w.clone(),
            inject_w: inject_w.clone(),
        };
        let (want_mixed, want_inject) = reference.read_batch(&stream_v, eps as f32);
        let mut want_stream = stream_v.clone();
        reference.write_batch(&mut want_stream, &block_out_v, &want_inject);

        let gate = HcRead::from_tensors(
            hc_count,
            hidden,
            low_rank,
            eps,
            Tensor::from_slice(&norm_w, width, &dev).unwrap(),
            &tensor2(&down_w, low_rank, width, &dev),
            &tensor2(&up_w, width, low_rank, &dev),
            Some(&tensor2(&inject_w, hc_count, width, &dev)),
        )
        .unwrap();

        let stream = tensor2(&stream_v, n, width, &dev);
        let (mixed, inject) = gate.read(&stream).unwrap();
        let inject = inject.unwrap();
        let got_mixed = flat(&mixed);
        let got_inject = flat(&inject);
        assert!(
            rel_l2(&got_mixed, &want_mixed) <= 1e-5,
            "mixed rel_l2 {} > 1e-5",
            rel_l2(&got_mixed, &want_mixed)
        );
        assert!(
            rel_l2(&got_inject, &want_inject) <= 1e-5,
            "inject rel_l2 {} > 1e-5",
            rel_l2(&got_inject, &want_inject)
        );

        let block_out = tensor2(&block_out_v, n, hidden, &dev);
        let got_stream = flat(&hc_write(&stream, &block_out, &inject).unwrap());
        assert!(
            rel_l2(&got_stream, &want_stream) <= 1e-5,
            "written carrier rel_l2 {} > 1e-5",
            rel_l2(&got_stream, &want_stream)
        );

        // The same weights with the injection head dropped are the tail mixer,
        // and its mixed vector is the block gate's — the inject head is the only
        // difference between the two.
        let mixer_ref = HcMixerRef {
            hc_count,
            hidden,
            low_rank,
            norm_w,
            down_w: down_w.clone(),
            up_w: up_w.clone(),
        };
        let mixer = HcRead::from_tensors(
            hc_count,
            hidden,
            low_rank,
            eps,
            Tensor::from_slice(&mixer_ref.norm_w, width, &dev).unwrap(),
            &tensor2(&down_w, low_rank, width, &dev),
            &tensor2(&up_w, width, low_rank, &dev),
            None,
        )
        .unwrap();
        let want_tail = mixer_ref.mix_batch(&stream_v, eps as f32);
        let got_tail = flat(&mixer.mix(&stream).unwrap());
        assert!(
            rel_l2(&got_tail, &want_tail) <= 1e-5,
            "tail mixer rel_l2 {} > 1e-5",
            rel_l2(&got_tail, &want_tail)
        );
    }

    /// The live read path against the candle chain it replaces, at the real
    /// geometry and over the same gate weights.
    ///
    /// The chain here is spelled out rather than called, so that an edit to
    /// either implementation shows up as a failure instead of silently moving
    /// the target. Under `XWEN_HC_CLASSIC` the two sides are the same code and
    /// this only re-proves the transcription; under the default it is what pins
    /// the vendored kernels to the chain. The norm and the mix partition
    /// reductions the chain runs in one order, so it grades at 1e-6 rather than
    /// bitwise.
    #[test]
    fn read_matches_the_candle_chain() {
        let dev = dev();
        let (hc_count, hidden, low_rank, n) = (4usize, 2560usize, 320usize, 5usize);
        let width = hc_count * hidden;
        let eps = 1e-6f64;

        let norm_w: Vec<f32> = seeded(width, 111).iter().map(|v| 1.0 + 0.1 * v).collect();
        let gate = HcRead::from_tensors(
            hc_count,
            hidden,
            low_rank,
            eps,
            Tensor::from_slice(&norm_w, width, &dev).unwrap(),
            &tensor2(&seeded(low_rank * width, 222), low_rank, width, &dev),
            &tensor2(&seeded(width * low_rank, 333), width, low_rank, &dev),
            Some(&tensor2(
                &seeded(hc_count * width, 444),
                hc_count,
                width,
                &dev,
            )),
        )
        .unwrap();
        let stream = tensor2(&seeded(n * width, 555), n, width, &dev);

        let (mixed, inject) = gate.read(&stream).unwrap();
        let inject = inject.expect("a block gate has an injection head");

        let inv_hc = 1.0 / hc_count as f64;
        let normed = gate.grouped_norm(&stream).unwrap();
        let low = silu(
            &gate
                .down
                .forward(&normed)
                .unwrap()
                .affine(inv_hc, 0.0)
                .unwrap(),
        )
        .unwrap();
        let mix = sigmoid(&gate.up.forward(&low).unwrap()).unwrap();
        let want_mixed = (mix * &normed)
            .unwrap()
            .reshape((n, hc_count, hidden))
            .unwrap()
            .sum(1)
            .unwrap()
            .affine(inv_hc, 0.0)
            .unwrap();
        let want_inject = sigmoid(
            &gate
                .inject
                .as_ref()
                .unwrap()
                .forward(&normed)
                .unwrap()
                .affine(inv_hc, 0.0)
                .unwrap(),
        )
        .unwrap()
        .affine(2.0, 0.0)
        .unwrap();

        let e = rel_l2(&flat(&mixed), &flat(&want_mixed));
        assert!(e <= 1e-6, "mixed vs the candle chain: rel_l2 {e} > 1e-6");
        let e = rel_l2(&flat(&inject), &flat(&want_inject));
        assert!(e <= 1e-6, "inject vs the candle chain: rel_l2 {e} > 1e-6");

        // The write-back is a multiply and an add in a fixed order on both
        // paths, so it is held to bits, not a tolerance.
        let block_out = tensor2(&seeded(n * hidden, 666), n, hidden, &dev);
        let got = hc_write(&stream, &block_out, &inject).unwrap();
        let scaled = block_out
            .reshape((n, 1, hidden))
            .unwrap()
            .broadcast_mul(&inject.reshape((n, hc_count, 1)).unwrap())
            .unwrap();
        let want = (stream.reshape((n, hc_count, hidden)).unwrap() + scaled)
            .unwrap()
            .reshape((n, width))
            .unwrap();
        for (i, (g, w)) in flat(&got).iter().zip(flat(&want).iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "hc_write element {i} differs (got {g:?}, want {w:?})"
            );
        }
    }

    /// The carrier is seeded by TILING the embedding — `[x, x, x, x]` — not by
    /// interleaving it. Distinct per-column values make the two orders
    /// distinguishable.
    #[test]
    fn seed_stream_tiles_rather_than_interleaves() {
        let dev = dev();
        let (n, hidden, hc_count) = (2usize, 3usize, 4usize);
        let embed_v: Vec<f32> = (0..n * hidden).map(|i| i as f32 + 1.0).collect();
        let embed = tensor2(&embed_v, n, hidden, &dev);

        let mut want = Vec::with_capacity(n * hc_count * hidden);
        for t in 0..n {
            for _ in 0..hc_count {
                want.extend_from_slice(&embed_v[t * hidden..(t + 1) * hidden]);
            }
        }
        let got = seed_stream(&embed, hc_count).unwrap();
        assert_eq!(got.dims(), &[n, hc_count * hidden]);
        assert_eq!(flat(&got), want);

        // What interleaving would have produced, so the assertion above is
        // pinning a real distinction and not a shape coincidence.
        let mut interleaved = Vec::with_capacity(n * hc_count * hidden);
        for t in 0..n {
            for j in 0..hidden {
                for _ in 0..hc_count {
                    interleaved.push(embed_v[t * hidden + j]);
                }
            }
        }
        assert_ne!(want, interleaved);
    }
}
