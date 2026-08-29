//! A synthetic single-file `qwen4exp` GGUF at toy geometry, for tests.
//!
//! The shipped checkpoint is 100+ GB and cannot be a test dependency, but every
//! `qwen4exp` loader — the hyper-connection gates, the QSA indexer, the PLE
//! layer and its n-gram table, the attention and DeltaNet blocks, the MoE block
//! — is a shape contract against a GGUF tensor table. This module writes a file
//! that satisfies all of those contracts with random weights, so a test can
//! exercise the real `crate::gguf::open` → `XwenConfig::from_gguf` → loader path
//! end to end.
//!
//! Two things make the fixture exact rather than approximate:
//!
//! * Every tensor is stored as `GgmlDType::F32`. An F32 `QTensor` is a memcpy of
//!   the source and `QMatMul` dequantizes it straight back, so nothing here
//!   depends on a quantizer's rounding and the file is readable by every loader
//!   arm that dispatches on stored dtype.
//! * The shapes are the ones `reference/llama.cpp/src/models/qwen4exp.cpp`
//!   declares, written in candle order. candle's GGUF writer emits `ne`
//!   reversed, so a `QTensor` built as `(out, in)` lands in the file as
//!   `{in, out}` — llama.cpp's own listing — and reads back as `(out, in)`.
//!   The per-tensor comments below give the llama.cpp `ne` beside the candle
//!   shape for exactly that reason.
//!
//! The one place the tiny geometry deviates from llama.cpp's listing on purpose:
//! `ple_key` / `ple_value` are declared there over `n_embd` because the shipped
//! file has `ple_head_dim * ple_n_heads == n_embd` by coincidence (port-doc trap
//! #13). Our loader asserts the DERIVED PLE embedding width, so this fixture
//! picks a geometry where the two differ and writes the derived one.

use std::path::Path;

use anyhow::Result;
use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};

/// The geometry of a synthetic `qwen4exp` file.
///
/// [`TinyGeometry::default`] is the geometry the loaders are tested at; every
/// field is public so a test can perturb one number and assert that the loader
/// which depends on it refuses the file.
#[derive(Debug, Clone)]
pub struct TinyGeometry {
    /// `general.name` — what identifies a checkpoint (never the architecture).
    pub general_name: String,
    pub n_layer: usize,
    /// Full attention at every layer where `(il + 1) % interval == 0`.
    pub full_attention_interval: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub context_length: usize,

    // Full-attention geometry.
    pub n_head: usize,
    pub n_head_kv: usize,
    /// `attention.key_length`, also written as `value_length`.
    pub head_dim: usize,
    pub rope_dim_count: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,

    // DeltaNet geometry. The `ssm.*` keys mean what `config.rs` reads them as:
    // `state_size` is the per-head dim, `time_step_rank` the V-head count,
    // `group_count` the K-head count.
    pub ssm_state_size: usize,
    pub ssm_v_heads: usize,
    pub ssm_k_heads: usize,
    pub conv_kernel: usize,

    // Hyper-connections: the carrier is `hc_count * hidden` wide.
    pub hc_count: usize,
    pub hc_low_rank: usize,

    // QSA indexer.
    pub indexer_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_top_k: usize,
    /// Written into `attention.compress_ratios` on the full-attention layers and
    /// zero on the DeltaNet ones, the per-layer form the converter emits.
    pub indexer_compress_ratio: usize,

    // MoE. Every layer is MoE; there is no dense FFN.
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub expert_ff: usize,
    pub shared_expert_ff: usize,

    // PLE.
    /// Layer indices that carry a PLE injection, 0-based.
    pub ple_layers: Vec<usize>,
    pub ple_ngram_size: usize,
    pub ple_heads_per_ngram: usize,
    pub ple_conv_kernel: usize,
    pub ple_eos_token_id: u32,
    /// `embedding_length_per_layer_input` — the table's row width.
    pub ple_row_dim: usize,
    /// One modulus per hash head; the flat table is their sum of rows.
    pub ple_head_vocab_sizes: Vec<u64>,
    /// The running sum of `ple_head_vocab_sizes`, which the config validates.
    pub ple_head_offsets: Vec<u64>,
    /// One hash multiplier per token position of the longest n-gram.
    pub ple_layer_multipliers: Vec<u64>,

    /// `tokenizer.ggml.eos_token_id`.
    pub eos_token_id: u32,
}

impl Default for TinyGeometry {
    fn default() -> Self {
        Self {
            general_name: "Tiny Qwen4Exp".to_string(),
            n_layer: 4,
            full_attention_interval: 4,
            hidden: 64,
            vocab: 256,
            context_length: 4096,

            n_head: 2,
            n_head_kv: 1,
            head_dim: 32,
            rope_dim_count: 16,
            rope_freq_base: 1e7,
            rms_eps: 1e-6,

            ssm_state_size: 16,
            ssm_v_heads: 4,
            ssm_k_heads: 2,
            conv_kernel: 4,

            hc_count: 4,
            hc_low_rank: 8,

            indexer_heads: 2,
            indexer_head_dim: 16,
            indexer_top_k: 8,
            indexer_compress_ratio: 2,

            n_expert: 4,
            n_expert_used: 2,
            expert_ff: 32,
            shared_expert_ff: 32,

            ple_layers: vec![1],
            ple_ngram_size: 3,
            ple_heads_per_ngram: 2,
            ple_conv_kernel: 4,
            ple_eos_token_id: 3,
            ple_row_dim: 8,
            ple_head_vocab_sizes: vec![101, 103, 107, 109],
            ple_head_offsets: vec![0, 101, 204, 311],
            // Distinct large odd constants, the shape the reference hash mixes.
            ple_layer_multipliers: vec![
                0x9E37_79B9_7F4A_7C15,
                0xC2B2_AE3D_27D4_EB4F,
                0x1656_67B1_9E37_79F9,
            ],

            eos_token_id: 3,
        }
    }
}

impl TinyGeometry {
    /// `ssm.inner_size`: the DeltaNet value width, V-heads by head dim.
    pub fn ssm_inner(&self) -> usize {
        self.ssm_v_heads * self.ssm_state_size
    }

    /// The fused `attn_qkv` width: q and k at K-head count, v at V-head count.
    pub fn conv_dim(&self) -> usize {
        (2 * self.ssm_k_heads + self.ssm_v_heads) * self.ssm_state_size
    }

    /// The residual carrier width, `hc_count * hidden`.
    pub fn carrier(&self) -> usize {
        self.hc_count * self.hidden
    }

    /// Hash heads: one per head of each n-gram order `2..=ngram_size`.
    pub fn ple_n_heads(&self) -> usize {
        (self.ple_ngram_size - 1) * self.ple_heads_per_ngram
    }

    /// The PLE embedding width the projections read: all heads' rows
    /// concatenated.
    pub fn ple_emb_dim(&self) -> usize {
        self.ple_n_heads() * self.ple_row_dim
    }

    /// Rows in the flat n-gram table — every head's slice, end to end.
    pub fn ple_rows(&self) -> usize {
        self.ple_head_vocab_sizes.iter().sum::<u64>() as usize
    }

    /// Whether layer `il` runs full attention (QSA) rather than DeltaNet.
    pub fn is_full(&self, il: usize) -> bool {
        (il + 1).is_multiple_of(self.full_attention_interval)
    }

    /// Whether layer `il` carries the PLE injection.
    pub fn is_ple(&self, il: usize) -> bool {
        self.ple_layers.contains(&il)
    }
}

/// Deterministic pseudo-random f32s in `[lo, hi]` (xorshift64, no deps).
fn rand(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    // xorshift is a fixed point at zero; any seed landing there would emit a
    // constant tensor instead of noise.
    if s == 0 {
        s = 0x2545_F491_4F6C_DD1D;
    }
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

/// FNV-1a over the tensor name, so each tensor's values depend only on its name
/// and the file is byte-reproducible whatever order the tensors are built in.
fn seed_of(name: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The value range a tensor's weights are drawn from. Magnitudes are kept small
/// so a forward over the whole stack stays finite.
#[derive(Clone, Copy)]
enum Fill {
    /// A projection weight.
    Weight,
    /// A norm weight, near 1. Every GGUF norm arrives multiply-ready — the
    /// converter bakes the `1 +` in — so these are the final multipliers.
    Norm,
    /// `ssm_a`, which ships pre-baked as `-exp(A_log)` and is therefore strictly
    /// negative: `g = ssm_a * softplus(...)` must decay the state, not grow it.
    SsmA,
    /// The `ssm_dt` offset vector.
    Dt,
}

impl Fill {
    fn range(self) -> (f32, f32) {
        match self {
            Fill::Weight => (-0.1, 0.1),
            Fill::Norm => (0.9, 1.1),
            Fill::SsmA => (-1.0, -0.05),
            Fill::Dt => (-0.5, 0.5),
        }
    }
}

/// One named F32 tensor of random values, at a candle shape.
fn tensor(name: &str, dims: &[usize], fill: Fill) -> Result<(String, QTensor)> {
    let (lo, hi) = fill.range();
    let n: usize = dims.iter().product();
    let values = rand(seed_of(name), n, lo, hi);
    let t = Tensor::from_vec(values, dims.to_vec(), &Device::Cpu)?;
    Ok((name.to_string(), QTensor::quantize(&t, GgmlDType::F32)?))
}

/// Writes a single-file `qwen4exp` GGUF of random-but-deterministic weights to
/// `path`, overwriting whatever is there.
///
/// The file carries every metadata key `XwenConfig::from_gguf` reads for this
/// architecture and every tensor the block loaders open, at `geo`'s geometry.
/// Values depend only on the tensor name, so two writes of the same geometry
/// produce identical bytes.
pub fn write_tiny_qwen4exp(path: &Path, geo: &TinyGeometry) -> Result<()> {
    let a = "qwen4exp";
    let u32v = |v: usize| Value::U32(v as u32);
    let u32_array = |vs: Vec<usize>| Value::Array(vs.into_iter().map(u32v).collect());
    let u64_array = |vs: &[u64]| Value::Array(vs.iter().copied().map(Value::U64).collect());

    let mut kv: Vec<(String, Value)> = vec![
        ("general.architecture".into(), Value::String(a.into())),
        (
            "general.name".into(),
            Value::String(geo.general_name.clone()),
        ),
        (format!("{a}.block_count"), u32v(geo.n_layer)),
        (format!("{a}.context_length"), u32v(geo.context_length)),
        (format!("{a}.embedding_length"), u32v(geo.hidden)),
        (format!("{a}.vocab_size"), u32v(geo.vocab)),
        (
            format!("{a}.full_attention_interval"),
            u32v(geo.full_attention_interval),
        ),
        // Attention. `head_count` goes in as a SCALAR: `Meta::usize_per_layer`
        // expands a scalar to a uniform per-layer vec, which is what the shipped
        // files carry.
        (format!("{a}.attention.head_count"), u32v(geo.n_head)),
        (format!("{a}.attention.head_count_kv"), u32v(geo.n_head_kv)),
        (format!("{a}.attention.key_length"), u32v(geo.head_dim)),
        (format!("{a}.attention.value_length"), u32v(geo.head_dim)),
        (
            format!("{a}.attention.layer_norm_rms_epsilon"),
            Value::F32(geo.rms_eps),
        ),
        // Rope. `dimension_sections` is deliberately absent: `Meta::check_sections`
        // only validates a section list that is present, and an absent key is a
        // non-IMROPE conversion, which is what this fixture is.
        (
            format!("{a}.rope.dimension_count"),
            u32v(geo.rope_dim_count),
        ),
        (
            format!("{a}.rope.freq_base"),
            Value::F32(geo.rope_freq_base),
        ),
        // DeltaNet, under the misleading Mamba key names config.rs decodes.
        (format!("{a}.ssm.state_size"), u32v(geo.ssm_state_size)),
        (format!("{a}.ssm.time_step_rank"), u32v(geo.ssm_v_heads)),
        (format!("{a}.ssm.group_count"), u32v(geo.ssm_k_heads)),
        (format!("{a}.ssm.inner_size"), u32v(geo.ssm_inner())),
        (format!("{a}.ssm.conv_kernel"), u32v(geo.conv_kernel)),
        // Hyper-connections.
        (format!("{a}.hyper_connection.count"), u32v(geo.hc_count)),
        (
            format!("{a}.hyper_connection.low_rank"),
            u32v(geo.hc_low_rank),
        ),
        // QSA indexer.
        (
            format!("{a}.attention.indexer.head_count"),
            u32v(geo.indexer_heads),
        ),
        (
            format!("{a}.attention.indexer.key_length"),
            u32v(geo.indexer_head_dim),
        ),
        (
            format!("{a}.attention.indexer.top_k"),
            u32v(geo.indexer_top_k),
        ),
        (
            format!("{a}.attention.compress_ratios"),
            u32_array(
                (0..geo.n_layer)
                    .map(|il| {
                        if geo.is_full(il) {
                            geo.indexer_compress_ratio
                        } else {
                            0
                        }
                    })
                    .collect(),
            ),
        ),
        // MoE.
        (format!("{a}.expert_count"), u32v(geo.n_expert)),
        (format!("{a}.expert_used_count"), u32v(geo.n_expert_used)),
        (
            format!("{a}.expert_feed_forward_length"),
            u32v(geo.expert_ff),
        ),
        (
            format!("{a}.expert_shared_feed_forward_length"),
            u32v(geo.shared_expert_ff),
        ),
        // PLE.
        (format!("{a}.ple.layers"), u32_array(geo.ple_layers.clone())),
        (format!("{a}.ple.ngram_size"), u32v(geo.ple_ngram_size)),
        (
            format!("{a}.ple.heads_per_ngram"),
            u32v(geo.ple_heads_per_ngram),
        ),
        (format!("{a}.ple.conv_kernel"), u32v(geo.ple_conv_kernel)),
        (
            format!("{a}.ple.eos_token_id"),
            Value::U32(geo.ple_eos_token_id),
        ),
        (
            format!("{a}.embedding_length_per_layer_input"),
            u32v(geo.ple_row_dim),
        ),
        (
            format!("{a}.ple.head_vocab_sizes"),
            u64_array(&geo.ple_head_vocab_sizes),
        ),
        (
            format!("{a}.ple.head_offsets"),
            u64_array(&geo.ple_head_offsets),
        ),
        (
            format!("{a}.ple.layer_multipliers"),
            u64_array(&geo.ple_layer_multipliers),
        ),
        (
            "tokenizer.ggml.eos_token_id".into(),
            Value::U32(geo.eos_token_id),
        ),
    ];
    kv.sort_by(|x, y| x.0.cmp(&y.0));

    let carrier = geo.carrier();
    // ---- root ----
    let mut ts: Vec<(String, QTensor)> = vec![
        // llama.cpp ne {n_embd, n_vocab} -> candle [vocab, hidden].
        tensor("token_embd.weight", &[geo.vocab, geo.hidden], Fill::Weight)?,
        tensor("output.weight", &[geo.vocab, geo.hidden], Fill::Weight)?,
        // The tail mixer, which carries what other architectures call
        // output_norm.
        tensor("output_hc_norm.weight", &[carrier], Fill::Norm)?,
        tensor(
            "output_hc_down.weight",
            &[geo.hc_low_rank, carrier],
            Fill::Weight,
        )?,
        tensor(
            "output_hc_up.weight",
            &[carrier, geo.hc_low_rank],
            Fill::Weight,
        )?,
        // The flat n-gram table: llama.cpp ne {ple_head_dim, ple_rows} -> candle
        // [rows, row_dim], which is the orientation `PleTable::open` reads.
        tensor(
            "per_layer_token_embd.weight",
            &[geo.ple_rows(), geo.ple_row_dim],
            Fill::Weight,
        )?,
    ];

    // ---- blocks ----
    for il in 0..geo.n_layer {
        let p = format!("blk.{il}");

        // Two hyper-connection gates per layer: one before the token mixer, one
        // before the MoE. There is no attn_norm and no post_attention_norm on
        // this architecture — these gates replace both.
        for gate in ["hc_attn", "hc_ffn"] {
            ts.push(tensor(
                &format!("{p}.{gate}_norm.weight"),
                &[carrier],
                Fill::Norm,
            )?);
            ts.push(tensor(
                &format!("{p}.{gate}_down.weight"),
                &[geo.hc_low_rank, carrier],
                Fill::Weight,
            )?);
            ts.push(tensor(
                &format!("{p}.{gate}_up.weight"),
                &[carrier, geo.hc_low_rank],
                Fill::Weight,
            )?);
            ts.push(tensor(
                &format!("{p}.{gate}_inject.weight"),
                &[geo.hc_count, carrier],
                Fill::Weight,
            )?);
        }

        if geo.is_full(il) {
            // `attn_q` is DOUBLE width: per-head interleaved [q_head, gate_head].
            ts.push(tensor(
                &format!("{p}.attn_q.weight"),
                &[2 * geo.n_head * geo.head_dim, geo.hidden],
                Fill::Weight,
            )?);
            for name in ["attn_k", "attn_v"] {
                ts.push(tensor(
                    &format!("{p}.{name}.weight"),
                    &[geo.n_head_kv * geo.head_dim, geo.hidden],
                    Fill::Weight,
                )?);
            }
            ts.push(tensor(
                &format!("{p}.attn_output.weight"),
                &[geo.hidden, geo.n_head * geo.head_dim],
                Fill::Weight,
            )?);
            for name in ["attn_q_norm", "attn_k_norm"] {
                ts.push(tensor(
                    &format!("{p}.{name}.weight"),
                    &[geo.head_dim],
                    Fill::Norm,
                )?);
            }
            // The indexer key side is MQA: `k_proj` is exactly one head.
            ts.push(tensor(
                &format!("{p}.indexer.q_proj.weight"),
                &[geo.indexer_heads * geo.indexer_head_dim, geo.hidden],
                Fill::Weight,
            )?);
            ts.push(tensor(
                &format!("{p}.indexer.k_proj.weight"),
                &[geo.indexer_head_dim, geo.hidden],
                Fill::Weight,
            )?);
            for name in ["indexer.q_norm", "indexer.k_norm"] {
                ts.push(tensor(
                    &format!("{p}.{name}.weight"),
                    &[geo.indexer_head_dim],
                    Fill::Norm,
                )?);
            }
        } else {
            // DeltaNet. The projections ship under attention tensor names.
            ts.push(tensor(
                &format!("{p}.attn_qkv.weight"),
                &[geo.conv_dim(), geo.hidden],
                Fill::Weight,
            )?);
            ts.push(tensor(
                &format!("{p}.attn_gate.weight"),
                &[geo.ssm_inner(), geo.hidden],
                Fill::Weight,
            )?);
            ts.push(tensor(
                &format!("{p}.ssm_out.weight"),
                &[geo.hidden, geo.ssm_inner()],
                Fill::Weight,
            )?);
            // Kernel-major in the file — ne {conv_kernel, conv_dim} — which
            // candle reads as [conv_dim, conv_kernel] and the block transposes.
            ts.push(tensor(
                &format!("{p}.ssm_conv1d.weight"),
                &[geo.conv_dim(), geo.conv_kernel],
                Fill::Weight,
            )?);
            for name in ["ssm_beta", "ssm_alpha"] {
                ts.push(tensor(
                    &format!("{p}.{name}.weight"),
                    &[geo.ssm_v_heads, geo.hidden],
                    Fill::Weight,
                )?);
            }
            // `ssm_a` has no suffix at all and `ssm_dt` is bias-suffixed (it is
            // the dt offset vector, a bias in name only); `dense_f32_any` finds
            // both.
            ts.push(tensor(
                &format!("{p}.ssm_a"),
                &[geo.ssm_v_heads],
                Fill::SsmA,
            )?);
            ts.push(tensor(
                &format!("{p}.ssm_dt.bias"),
                &[geo.ssm_v_heads],
                Fill::Dt,
            )?);
            // The gated RMS norm is per V-head, so it spans one head dim.
            ts.push(tensor(
                &format!("{p}.ssm_norm.weight"),
                &[geo.ssm_state_size],
                Fill::Norm,
            )?);
        }

        if geo.is_ple(il) {
            let emb = geo.ple_emb_dim();
            // Both projections read the PLE embedding width, NOT the hidden size
            // (see the module header).
            ts.push(tensor(
                &format!("{p}.ple_key.weight"),
                &[carrier, emb],
                Fill::Weight,
            )?);
            ts.push(tensor(
                &format!("{p}.ple_value.weight"),
                &[geo.hidden, emb],
                Fill::Weight,
            )?);
            // All three norms span the whole carrier.
            for name in ["ple_norm_key", "ple_norm_query", "ple_norm_conv"] {
                ts.push(tensor(
                    &format!("{p}.{name}.weight"),
                    &[carrier],
                    Fill::Norm,
                )?);
            }
            ts.push(tensor(
                &format!("{p}.ple_conv1d.weight"),
                &[carrier, geo.ple_conv_kernel],
                Fill::Weight,
            )?);
        }

        // Every layer is MoE.
        ts.push(tensor(
            &format!("{p}.ffn_gate_inp.weight"),
            &[geo.n_expert, geo.hidden],
            Fill::Weight,
        )?);
        for name in ["ffn_gate_exps", "ffn_up_exps"] {
            ts.push(tensor(
                &format!("{p}.{name}.weight"),
                &[geo.n_expert, geo.expert_ff, geo.hidden],
                Fill::Weight,
            )?);
        }
        ts.push(tensor(
            &format!("{p}.ffn_down_exps.weight"),
            &[geo.n_expert, geo.hidden, geo.expert_ff],
            Fill::Weight,
        )?);
        // The shared expert's router is one [hidden] vector: a scalar gate per
        // token, not a per-expert distribution.
        ts.push(tensor(
            &format!("{p}.ffn_gate_inp_shexp.weight"),
            &[geo.hidden],
            Fill::Weight,
        )?);
        for name in ["ffn_gate_shexp", "ffn_up_shexp"] {
            ts.push(tensor(
                &format!("{p}.{name}.weight"),
                &[geo.shared_expert_ff, geo.hidden],
                Fill::Weight,
            )?);
        }
        ts.push(tensor(
            &format!("{p}.ffn_down_shexp.weight"),
            &[geo.hidden, geo.shared_expert_ff],
            Fill::Weight,
        )?);
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let kv_refs: Vec<(&str, &Value)> = kv.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let t_refs: Vec<(&str, &QTensor)> = ts.iter().map(|(k, t)| (k.as_str(), t)).collect();
    let mut f = std::fs::File::create(path)?;
    gguf_file::write(&mut f, &kv_refs, &t_refs)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::attention::{AttnBlock, AttnWeights};
    use crate::config::{LayerKind, XwenConfig};
    use crate::gguf::{Weights, metal_device};
    use crate::linear_attn::LinearAttnBlock;
    use crate::moe::MoeBlock;
    use crate::ops::ExpertRunner;
    use crate::qwen4exp::hc::HcRead;
    use crate::qwen4exp::indexer::QsaIndexer;
    use crate::qwen4exp::ple::{PleLayer, PleTable};
    use crate::rope::Rope;

    /// Removes the fixture directory on the way in and on the way out, so a
    /// panicking run does not leave a stale file for the next one.
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!("xwen_tiny_q4e_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).expect("creating the fixture directory");
            Self(p)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The whole point of the fixture: a file this module writes opens through
    /// the production GGUF reader, parses into a `XwenConfig` with the geometry
    /// it was written at, and satisfies every `qwen4exp` block loader's shape
    /// contract.
    #[test]
    fn tiny_qwen4exp_file_loads_every_block() {
        let dir = TempDir::new();
        let path = dir.0.join("tiny-qwen4exp.gguf");
        let geo = TinyGeometry::default();
        write_tiny_qwen4exp(&path, &geo).expect("writing the tiny GGUF");

        // Metal is the production device; CPU keeps the shape contracts gradable
        // on a machine without it.
        let device = metal_device().unwrap_or(candle_core::Device::Cpu);
        let gguf = crate::gguf::open(&path, &device).expect("opening the tiny GGUF");

        let cfg = XwenConfig::from_gguf(&gguf.content).expect("parsing the tiny config");
        assert_eq!(cfg.n_layer, 4);
        assert_eq!(
            cfg.layer_kind,
            vec![
                LayerKind::Linear,
                LayerKind::Linear,
                LayerKind::Linear,
                LayerKind::Full
            ]
        );
        assert_eq!(cfg.hidden, 64);
        let q4 = cfg.qwen4exp.as_ref().expect("a qwen4exp config block");
        assert_eq!(q4.hc_count, 4);
        assert_eq!(q4.hc_low_rank, 8);
        assert_eq!(q4.indexer_heads, 2);
        assert_eq!(q4.indexer_head_dim, 16);
        assert_eq!(q4.indexer_top_k, 8);
        assert_eq!(q4.indexer_compress_ratio, 2);
        let ple = q4.ple.as_ref().expect("a PLE config block");
        assert_eq!(ple.layers, vec![1]);
        assert_eq!(ple.ngram_size, 3);
        assert_eq!(ple.row_dim, 8);

        let w = Weights::from_gguf(gguf.clone());
        let rope = Arc::new(Rope::new(cfg.rope(), 128, &device).expect("rope tables"));

        // Hyper-connection gates: both block gates and the injection-head-less
        // tail mixer.
        assert!(HcRead::load(&w.pp("blk.0"), "hc_attn", &cfg, true).is_ok());
        assert!(HcRead::load(&w.pp("blk.0"), "hc_ffn", &cfg, true).is_ok());
        assert!(HcRead::load(&w, "output_hc", &cfg, false).is_ok());

        // The QSA layer's indexer and its attention block.
        assert!(QsaIndexer::load(&w.pp("blk.3"), &cfg, rope.clone()).is_ok());
        assert!(AttnBlock::new(&w.pp("blk.3"), &cfg, 3, rope, AttnWeights::F16).is_ok());

        // A DeltaNet layer and a MoE block.
        assert!(LinearAttnBlock::new(&w.pp("blk.0"), &cfg, AttnWeights::F16).is_ok());
        assert!(MoeBlock::new(&w.pp("blk.0"), &cfg, ExpertRunner::Reference).is_ok());

        // The embedding table and the lm head.
        assert!(w.qtensor("token_embd").is_ok());
        assert!(w.qlinear_with_buffer("output").is_ok());

        // The PLE table is read straight out of the file mapping, which exists
        // only on the Metal mmap path; off it there is nothing to demand-page.
        if gguf.mmap_source().is_some() {
            let table =
                PleTable::open(&gguf, "per_layer_token_embd.weight").expect("opening the table");
            assert_eq!(table.row_dim(), 8);
            assert_eq!(table.rows(), 420);
            let mut row = [0f32; 8];
            assert!(table.row(419, &mut row).is_ok());

            assert!(PleLayer::load(&w.pp("blk.1"), &gguf, &cfg).is_ok());
        } else {
            eprintln!(
                "skipping the PLE table assertions: no file mapping (non-Metal device, or \
                 XWEN_LOAD_CLASSIC)"
            );
        }
    }
}
