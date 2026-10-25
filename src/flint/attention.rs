//! Flint's attention block: separate q/k/v projections, QK-RMSNorm before the
//! rope, NEoX rope over the whole 128-wide head on the sliding layers and none
//! on the full ones, GQA by contiguous KV-head groups, and a learned sink logit
//! per query head in the softmax denominator.
//!
//! The attention itself runs on one of two arms, picked once per load by
//! [`ATTN_ENV`] ([`AttnImpl`]): [`reference_attention`], an explicit f32 candle
//! chain that is the frozen correctness oracle and is never optimised, and
//! `production_attention`, the shipped arm, which is the one a faster kernel
//! replaces. The two compute the same math today, in separate functions, so the
//! kernel work edits one and the oracle it is graded against stays put.

use std::sync::Arc;

use anyhow::{Result, ensure};
use candle_core::{D, DType, Module, Tensor};

use crate::attention::{AttnWeights, Proj};
use crate::config::{LayerKind, XwenConfig};
use crate::gguf::Weights;
use crate::kv_cache::LayerCache;
use crate::rope::Rope;

/// The environment switch naming the attention arm: `production` (the default)
/// or `reference`, the frozen f32 oracle. Read once per load.
pub const ATTN_ENV: &str = "XWEN_FLINT_ATTN";

/// Which function computes Flint's attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnImpl {
    /// The shipped arm, `production_attention`: the one a fused kernel
    /// replaces.
    Production,
    /// [`reference_attention`], the frozen f32 oracle the production arm is
    /// graded against. Never optimised.
    Reference,
}

impl AttnImpl {
    /// The arm an unset [`ATTN_ENV`] selects. The loader logs the arm only when
    /// it differs from this one.
    pub const SHIPPED: Self = Self::Production;

    /// Resolve from [`ATTN_ENV`]: unset means [`Self::SHIPPED`], anything else
    /// must name an arm, so a typo is a load error rather than a silent run of
    /// the wrong one.
    pub fn from_env() -> Result<Self> {
        match std::env::var(ATTN_ENV) {
            Err(std::env::VarError::NotPresent) => Ok(Self::SHIPPED),
            Err(std::env::VarError::NotUnicode(_)) => {
                anyhow::bail!("{ATTN_ENV} is not valid UTF-8")
            }
            Ok(value) => Self::parse(&value),
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "production" => Ok(Self::Production),
            "reference" => Ok(Self::Reference),
            other => anyhow::bail!(
                "{ATTN_ENV}={other:?}: expected `production` (the default) or `reference`"
            ),
        }
    }
}

/// A deliberately WRONG Flint graph, for the parity gate's brackets
/// (tests/flint_parity.rs). It is a load option that only
/// [`crate::XwenModel::load_flint_wrong_graph`] takes, so no surface and no
/// environment can build one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrongGraph {
    /// The sink logits leave the softmax.
    NoSinks,
    /// The NoPE full-attention layers are roped too.
    RopeEverywhere,
    /// The sliding window holds 512 positions, the current one included, one
    /// fewer than the file says.
    Window512,
}

/// Query rows the reference attention scores per pass. The score tile is
/// `n_head x rows x keys` f32, so an unblocked 2048-token chunk at 32k keys
/// would be 8.6 GB; 256 rows hold it to 1.1 GB there.
const QUERY_BLOCK: usize = 256;

/// The production arm's score-tile ceiling. A tile is `n_head x rows x keys`
/// f32 and the arm lowers `rows` below [`QUERY_BLOCK`] to stay under this, so
/// a forward's attention transients stop growing with the context: at 256
/// rows the tile is 32 KiB per key, and the pool candle keeps of past tiles
/// measured 41 GB of footprint at 57k tokens (2026-10-25).
const TILE_BYTES: usize = 256 << 20;

/// One Flint attention layer. Projections run on the trunk's [`Proj`] (Q8_0 on
/// the shipped file: the dense f16 plane at prefill, the q8_0 gemv at decode);
/// norms and sinks are F32.
pub struct FlintAttn {
    q_proj: Proj,
    k_proj: Proj,
    v_proj: Proj,
    o_proj: Proj,
    q_norm: candle_nn::RmsNorm,
    k_norm: candle_nn::RmsNorm,
    /// `attn_sinks.weight`, `[n_head]` f32: one logit per query head that
    /// joins the softmax over that head's keys and attends to nothing. Raw
    /// logits, NOT scaled by `1/√head_dim` (llama.cpp `ggml_soft_max_add_sinks`
    /// adds them after the scale).
    sinks: Tensor,
    /// `None` on a NoPE layer (every full-attention layer of the shipped file).
    rope: Option<Arc<Rope>>,
    attn_impl: AttnImpl,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
}

impl FlintAttn {
    /// `w` is positioned at the block prefix (`blk.N`).
    pub fn new(
        w: &Weights,
        cfg: &XwenConfig,
        il: usize,
        rope: Arc<Rope>,
        weights: AttnWeights,
        attn_impl: AttnImpl,
        wrong: Option<WrongGraph>,
    ) -> Result<Self> {
        let flint = cfg
            .flint
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("FlintAttn on a config with no flint section"))?;
        let n_head = cfg.n_head(il);
        ensure!(
            n_head.is_multiple_of(cfg.n_kv_head),
            "layer {il}: {n_head} query heads over {} KV heads do not group",
            cfg.n_kv_head
        );
        let proj = |name: &str| -> Result<Proj> { Proj::load(w, name, weights) };
        let norm = |name: &str| -> Result<candle_nn::RmsNorm> {
            Ok(candle_nn::RmsNorm::new(w.dense_f32(name)?, cfg.rms_eps))
        };
        let sinks = w.dense_f32("attn_sinks")?;
        ensure!(
            sinks.dims() == [n_head],
            "layer {il}: attn_sinks is {:?}, expected [{n_head}]",
            sinks.dims()
        );
        let (sinks, roped) = match wrong {
            None | Some(WrongGraph::Window512) => (sinks, flint.roped[il]),
            // exp(-1e30) underflows to an exact zero in the denominator.
            Some(WrongGraph::NoSinks) => ((sinks.zeros_like()? - 1e30)?, flint.roped[il]),
            Some(WrongGraph::RopeEverywhere) => (sinks, true),
        };
        // llama.cpp reads the window pattern and the rope pattern as separate
        // keys; this build does too, and lets the file pair them however it
        // likes.
        debug_assert!(matches!(
            cfg.layer_kind(il),
            LayerKind::Full | LayerKind::Swa { .. }
        ));
        Ok(Self {
            q_proj: proj("attn_q")?,
            k_proj: proj("attn_k")?,
            v_proj: proj("attn_v")?,
            o_proj: proj("attn_output")?,
            q_norm: norm("attn_q_norm")?,
            k_norm: norm("attn_k_norm")?,
            sinks,
            rope: roped.then_some(rope),
            attn_impl,
            n_head,
            n_kv_head: cfg.n_kv_head,
            head_dim: cfg.head_dim,
        })
    }

    /// Whether the projections hold the q8_0 decode alias, for dump
    /// provenance (the trunk's `AttnBlock::uses_q8_decode`).
    pub fn uses_q8_decode(&self) -> bool {
        matches!(self.q_proj, Proj::DenseF16Q8 { .. })
    }

    /// x_normed: `[seq, hidden]` f32, already `attn_norm`ed. Appends this
    /// step's K/V to `cache` (a full layer's rows or a sliding layer's ring)
    /// and returns `attn_output`'s `[seq, hidden]` f32.
    ///
    /// `mask` is `attn_mask_for(cache.mask_kind(), seq, pos)`, which the stack
    /// builds once per forward per kind and shares across the layers of that
    /// kind; `None` for a single decode row.
    pub fn forward(
        &self,
        x_normed: &Tensor,
        cache: &mut LayerCache,
        pos: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (seq, _) = x_normed.dims2()?;
        let (nh, nkv, hd) = (self.n_head, self.n_kv_head, self.head_dim);
        let q = self.q_proj.forward(x_normed)?.reshape((seq, nh, hd))?;
        let k = self.k_proj.forward(x_normed)?.reshape((seq, nkv, hd))?;
        let v = self.v_proj.forward(x_normed)?.reshape((seq, nkv, hd))?;
        let q = self.q_norm.forward(&q)?.transpose(0, 1)?.contiguous()?; // [nh, seq, hd]
        let k = self.k_norm.forward(&k)?.transpose(0, 1)?.contiguous()?; // [nkv, seq, hd]
        let v = v.transpose(0, 1)?.contiguous()?;
        let (q, k) = match &self.rope {
            Some(rope) => rope.apply(&q, &k, pos)?,
            None => (q, k),
        };
        // The cache holds f16, as llama.cpp's does; the reference reads it back
        // as f32 and does everything else at f32.
        let (k_all, v_all) = cache.append(&k.to_dtype(DType::F16)?, &v.to_dtype(DType::F16)?)?;
        let out = match self.attn_impl {
            AttnImpl::Production => {
                production_attention(&q, &k_all, &v_all, mask, &self.sinks, self.head_dim)?
            }
            AttnImpl::Reference => {
                reference_attention(&q, &k_all, &v_all, mask, &self.sinks, self.head_dim)?
            }
        }; // [nh, seq, hd]
        let out = out.transpose(0, 1)?.contiguous()?.reshape((seq, nh * hd))?;
        self.o_proj.forward(&out)
    }

    /// The arm this layer runs.
    pub fn attn_impl(&self) -> AttnImpl {
        self.attn_impl
    }
}

/// `softmax([q kᵀ·s + mask | sink]) [v | 0]` per query head, in f32: the
/// frozen oracle. Never optimise it; `production_attention` is the arm that
/// gets faster.
///
/// `q` is `[n_head, seq, hd]`; `k`/`v` are the cache's view
/// `[n_kv, keys, hd]` in the column order `mask` (`[seq, keys]`, additive,
/// `None` for one decode row) was built for. Query head `h` reads KV head
/// `h / (n_head / n_kv)`: ggml's broadcast in `ggml_mul_mat`, i.e. HF's
/// `repeat_kv` (repeat_interleave). The heads of one group are folded into
/// the row axis so K and V are never widened.
pub fn reference_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sinks: &Tensor,
    head_dim: usize,
) -> Result<Tensor> {
    let (nh, seq, hd) = q.dims3()?;
    let (nkv, keys, _) = k.dims3()?;
    let group = nh / nkv;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let kt = k.transpose(1, 2)?.contiguous()?; // [nkv, hd, keys]
    let sinks = sinks.to_dtype(DType::F32)?.reshape((nkv, group, 1, 1))?;
    let mut blocks = Vec::new();
    let mut r0 = 0;
    while r0 < seq {
        let rows = QUERY_BLOCK.min(seq - r0);
        // [nh, rows, hd] -> [nkv, group * rows, hd]: head h = j * group + i.
        let qb = q
            .narrow(1, r0, rows)?
            .contiguous()?
            .reshape((nkv, group * rows, hd))?;
        let scores = (qb.matmul(&kt)? * scale)?.reshape((nkv, group, rows, keys))?;
        let scores = match mask {
            Some(m) => {
                scores.broadcast_add(&m.narrow(0, r0, rows)?.reshape((1, 1, rows, keys))?)?
            }
            None => scores,
        };
        let sink_col = sinks.broadcast_as((nkv, group, rows, 1))?.contiguous()?;
        let logits = Tensor::cat(&[&scores, &sink_col], D::Minus1)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?
            .narrow(D::Minus1, 0, keys)?
            .contiguous()?
            .reshape((nkv, group * rows, keys))?;
        let out = probs.matmul(&v)?.reshape((nh, rows, hd))?;
        blocks.push(out);
        r0 += rows;
    }
    Ok(Tensor::cat(&blocks, 1)?)
}

/// The shipped attention arm, same operands and contract as
/// [`reference_attention`]. It computes the same math today; it is the function
/// a fused kernel replaces, with the reference left as the oracle.
pub(crate) fn production_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sinks: &Tensor,
    head_dim: usize,
) -> Result<Tensor> {
    let (nh, seq, hd) = q.dims3()?;
    let (nkv, keys, _) = k.dims3()?;
    let group = nh / nkv;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let kt = k.transpose(1, 2)?.contiguous()?; // [nkv, hd, keys]
    let sinks = sinks.to_dtype(DType::F32)?.reshape((nkv, group, 1, 1))?;
    // Rows are independent, so the block height changes no value: it bounds
    // the score tile, `n_head x rows x keys` f32, to [`TILE_BYTES`] whatever
    // the context, where a fixed height makes it grow with every key.
    let block =
        (TILE_BYTES / (nh * keys.max(1) * DType::F32.size_in_bytes())).clamp(1, QUERY_BLOCK);
    let mut blocks = Vec::new();
    let mut r0 = 0;
    while r0 < seq {
        let rows = block.min(seq - r0);
        // [nh, rows, hd] -> [nkv, group * rows, hd]: head h = j * group + i.
        let qb = q
            .narrow(1, r0, rows)?
            .contiguous()?
            .reshape((nkv, group * rows, hd))?;
        let scores = (qb.matmul(&kt)? * scale)?.reshape((nkv, group, rows, keys))?;
        let scores = match mask {
            Some(m) => {
                scores.broadcast_add(&m.narrow(0, r0, rows)?.reshape((1, 1, rows, keys))?)?
            }
            None => scores,
        };
        let sink_col = sinks.broadcast_as((nkv, group, rows, 1))?.contiguous()?;
        let logits = Tensor::cat(&[&scores, &sink_col], D::Minus1)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?
            .narrow(D::Minus1, 0, keys)?
            .contiguous()?
            .reshape((nkv, group * rows, keys))?;
        let out = probs.matmul(&v)?.reshape((nh, rows, hd))?;
        blocks.push(out);
        r0 += rows;
    }
    Ok(Tensor::cat(&blocks, 1)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

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

    /// Scalar attention with a sink, one query row at a time, written from the
    /// llama.cpp definition: `out = Σ e^{s_j} v_j / (e^{sink} + Σ e^{s_j})`.
    #[allow(clippy::too_many_arguments)]
    fn scalar(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        sinks: &[f32],
        nh: usize,
        nkv: usize,
        seq: usize,
        keys: usize,
        hd: usize,
        visible: impl Fn(usize, usize) -> bool,
    ) -> Vec<f32> {
        let mut out = vec![0f32; nh * seq * hd];
        let scale = 1.0 / (hd as f64).sqrt();
        for h in 0..nh {
            let j = h / (nh / nkv);
            for r in 0..seq {
                let mut s = vec![f64::NEG_INFINITY; keys];
                for c in 0..keys {
                    if visible(r, c) {
                        s[c] = (0..hd)
                            .map(|d| {
                                q[(h * seq + r) * hd + d] as f64 * k[(j * keys + c) * hd + d] as f64
                            })
                            .sum::<f64>()
                            * scale;
                    }
                }
                let m = s.iter().copied().fold(sinks[h] as f64, f64::max);
                let denom =
                    (sinks[h] as f64 - m).exp() + s.iter().map(|x| (x - m).exp()).sum::<f64>();
                for d in 0..hd {
                    let num: f64 = (0..keys)
                        .map(|c| (s[c] - m).exp() * v[(j * keys + c) * hd + d] as f64)
                        .sum();
                    out[(h * seq + r) * hd + d] = (num / denom) as f32;
                }
            }
        }
        out
    }

    /// The reference matches the definition with a sink, GQA grouping and a
    /// causal mask, including a query block boundary and sinks large enough to
    /// switch a head nearly off (the shipped file's run to ~10).
    #[test]
    fn the_reference_attention_is_the_sink_softmax() {
        let dev = Device::Cpu;
        let (nh, nkv, hd) = (8, 2, 16);
        let seq = QUERY_BLOCK + 9;
        let keys = seq;
        let q = seeded(nh * seq * hd, 1);
        let k = seeded(nkv * keys * hd, 2);
        let v = seeded(nkv * keys * hd, 3);
        let sinks: Vec<f32> = (0..nh).map(|h| h as f32 * 1.5 - 3.0).collect();
        let mut mask = vec![0f32; seq * keys];
        for r in 0..seq {
            for c in r + 1..keys {
                mask[r * keys + c] = f32::NEG_INFINITY;
            }
        }
        let t =
            |d: &[f32], s: (usize, usize, usize)| Tensor::from_vec(d.to_vec(), s, &dev).unwrap();
        let got = reference_attention(
            &t(&q, (nh, seq, hd)),
            &t(&k, (nkv, keys, hd)),
            &t(&v, (nkv, keys, hd)),
            Some(&Tensor::from_vec(mask, (seq, keys), &dev).unwrap()),
            &Tensor::from_vec(sinks.clone(), nh, &dev).unwrap(),
            hd,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        let want = scalar(&q, &k, &v, &sinks, nh, nkv, seq, keys, hd, |r, c| c <= r);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 1e-5, "max abs error {worst}");

        // The sinks must actually move the output: the same operands without
        // them (a vanishing sink logit) are plain softmax attention.
        let without = scalar(&q, &k, &v, &[-1e30; 8], nh, nkv, seq, keys, hd, |r, c| {
            c <= r
        });
        let moved = want
            .iter()
            .zip(&without)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(moved > 1e-2, "sinks moved the output by only {moved}");
    }

    /// The sliding layers' ring and mask against the definition, through the
    /// path a forward takes (`LayerCache::append`, `attn_mask_for`, both
    /// attention arms), at every chunking a forward can see: single rows, a
    /// small odd chunk, chunks one under, at and one over the window, and one
    /// wider than it. Query position `p` sees key position `j` exactly when
    /// `j <= p` and `p - j < 513`, the current position included; a window
    /// one short (512) is far outside the tolerance here, which is the
    /// off-by-one the logits gate cannot see (tests/flint_parity.rs).
    #[test]
    fn the_sliding_ring_attends_exactly_the_window() {
        use crate::kv_cache::{LayerCache, attn_mask_for};
        let dev = Device::Cpu;
        let (nh, nkv, hd, window) = (4usize, 2usize, 16usize, 513usize);
        let total = 1400usize;
        const TOL: f32 = 1e-5;
        let round = |v: Vec<f32>| -> Vec<f32> {
            v.into_iter()
                .map(|x| half::f16::from_f32(x).to_f32())
                .collect()
        };
        let q = seeded(nh * total * hd, 11);
        let k = round(seeded(nkv * total * hd, 12));
        let v = round(seeded(nkv * total * hd, 13));
        let sinks: Vec<f32> = (0..nh).map(|h| h as f32 - 1.5).collect();
        // [heads, total, hd] row-major -> one position's slice per head.
        let head_rows = |data: &[f32], heads: usize, from: usize, n: usize| -> Tensor {
            let mut out = Vec::with_capacity(heads * n * hd);
            for h in 0..heads {
                out.extend_from_slice(&data[(h * total + from) * hd..(h * total + from + n) * hd]);
            }
            Tensor::from_vec(out, (heads, n, hd), &dev).unwrap()
        };
        let definition = |p: usize, w: usize| -> Vec<f32> {
            // One query row against keys 0..=p, visible inside the window.
            let mut row = vec![0f32; nh * hd];
            let scale = 1.0 / (hd as f64).sqrt();
            for h in 0..nh {
                let j = h / (nh / nkv);
                let visible: Vec<usize> = (0..=p).filter(|&c| p - c < w).collect();
                let s: Vec<f64> = visible
                    .iter()
                    .map(|&c| {
                        (0..hd)
                            .map(|d| {
                                q[(h * total + p) * hd + d] as f64
                                    * k[(j * total + c) * hd + d] as f64
                            })
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let m = s.iter().copied().fold(sinks[h] as f64, f64::max);
                let denom =
                    (sinks[h] as f64 - m).exp() + s.iter().map(|x| (x - m).exp()).sum::<f64>();
                for d in 0..hd {
                    let num: f64 = visible
                        .iter()
                        .zip(&s)
                        .map(|(&c, &x)| (x - m).exp() * v[(j * total + c) * hd + d] as f64)
                        .sum();
                    row[h * hd + d] = (num / denom) as f32;
                }
            }
            row
        };
        let sink_t = Tensor::from_vec(sinks.clone(), nh, &dev).unwrap();
        let schedules: [(&str, &dyn Fn(usize) -> usize); 6] = [
            ("1", &|_| 1),
            ("7", &|_| 7),
            ("512", &|_| 512),
            ("513", &|_| 513),
            ("514", &|_| 514),
            ("1x40 then 700", &|p| if p < 40 { 1 } else { 700 }),
        ];
        let mut short_window_gap = 0f32;
        for (name, size) in schedules {
            let mut cache = LayerCache::Swa {
                k: Tensor::zeros((nkv, window, hd), DType::F16, &dev).unwrap(),
                v: Tensor::zeros((nkv, window, hd), DType::F16, &dev).unwrap(),
                len: 0,
                window,
            };
            let mut pos = 0;
            while pos < total {
                let seq = size(pos).min(total - pos);
                let kc = head_rows(&k, nkv, pos, seq).to_dtype(DType::F16).unwrap();
                let vc = head_rows(&v, nkv, pos, seq).to_dtype(DType::F16).unwrap();
                let (k_all, v_all) = cache.append(&kc, &vc).unwrap();
                let mask = attn_mask_for(cache.mask_kind(), seq, pos, &dev).unwrap();
                let qc = head_rows(&q, nh, pos, seq);
                type Arm = fn(
                    &Tensor,
                    &Tensor,
                    &Tensor,
                    Option<&Tensor>,
                    &Tensor,
                    usize,
                ) -> Result<Tensor>;
                let arms: [(&str, Arm); 2] = [
                    ("reference", reference_attention),
                    ("production", production_attention),
                ];
                for (arm, f) in arms {
                    let out = f(&qc, &k_all, &v_all, mask.as_ref(), &sink_t, hd)
                        .unwrap()
                        .to_vec3::<f32>()
                        .unwrap(); // [nh, seq, hd]
                    for r in 0..seq {
                        let want = definition(pos + r, window);
                        let mut worst = 0f32;
                        for h in 0..nh {
                            for d in 0..hd {
                                worst = worst.max((out[h][r][d] - want[h * hd + d]).abs());
                            }
                        }
                        assert!(
                            worst < TOL,
                            "schedule {name}, {arm}: position {} off the definition by {worst}",
                            pos + r
                        );
                        if arm == "reference" && pos + r >= window {
                            let short = definition(pos + r, window - 1);
                            let gap = (0..nh * hd)
                                .map(|i| (want[i] - short[i]).abs())
                                .fold(0f32, f32::max);
                            short_window_gap = short_window_gap.max(gap);
                        }
                    }
                }
                pos += seq;
            }
        }
        // Dropping the oldest of 513 keys moves a row by about one key's share
        // of the softmax, ~6e-4 on these operands: tens of tolerances.
        assert!(
            short_window_gap > 10.0 * TOL,
            "a 512 window lands within {short_window_gap} of the 513 one, so this test could not see it"
        );
    }
}
