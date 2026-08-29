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
//! There are TWO variants of the same file, sharing one tensor list and one
//! metadata block and differing only in the dtype each tensor is stored at.
//!
//! # The all-F32 variant ([`write_tiny_qwen4exp`], the default)
//!
//! Two things make it exact rather than approximate:
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
//! # The mixed-dtype variant ([`write_tiny_qwen4exp_mixed`])
//!
//! An all-F32 file never reaches the loaders' dtype-dispatch arms: the BF16
//! indexer projections, the Q8_0 planes, the Q4_K/Q5_K experts, the Q5_1
//! `ffn_down_exps` and — the one that is not merely a different arm but a
//! different HALF of the tensor table — the IQ4_NL PLE n-gram table, which
//! candle's parser cannot name at all and which therefore only exists on the
//! xwen-owned raw path. [`mixed_stored_dtype`] assigns each plane the dtype
//! `docs/qwen4exp-tensors.md` read out of the shipped Unsloth file, so this
//! variant is the fixture for anything that depends on how a tensor is STORED.
//!
//! Quantized tensors constrain the geometry — ggml quantizes along the fastest
//! dimension, so a Q4_K/Q5_K plane needs a row width that is a multiple of 256
//! — which is why the mixed variant has its own geometry constructor,
//! [`TinyGeometry::quantizable`]. Passing [`TinyGeometry::default`] to the mixed
//! writer errors on the first plane too narrow for its dtype (`output_hc_up`,
//! whose row is the low rank) rather than writing a bad file.
//!
//! The one place the tiny geometry deviates from llama.cpp's listing on purpose:
//! `ple_key` / `ple_value` are declared there over `n_embd` because the shipped
//! file has `ple_head_dim * ple_n_heads == n_embd` by coincidence (port-doc trap
//! #13). Our loader asserts the DERIVED PLE embedding width, so this fixture
//! picks a geometry where the two differ and writes the derived one.

use std::path::Path;

use anyhow::{Result, bail, ensure};
use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use half::f16;

use super::iq4nl::{BLOCK_BYTES, KVALUES_IQ4NL, QK4_NL};
use crate::gguf::{RawDtype, StoredDtype};

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
    /// The geometry [`write_tiny_qwen4exp_mixed`] is written at: the default,
    /// widened everywhere a quantized plane's fastest dimension has to be a
    /// whole number of blocks.
    ///
    /// ggml quantizes along the fastest dimension — the ROW, `ne[0]` in the
    /// file, the last dim in candle order — so every quantized tensor's row
    /// width must be a multiple of its dtype's block size. Three fields set
    /// those widths and none of the default's values clear the bar:
    ///
    /// * `hidden` is the row width of the Q4_K/Q5_K expert planes (block 256).
    /// * `hc_low_rank` is the row width of `hc_*_up` and `output_hc_up`, Q8_0
    ///   (block 32).
    /// * `ple_row_dim` is the row width of the IQ4_NL n-gram table (block 32),
    ///   and is also the gather granularity — one table row is read at a time,
    ///   so it must be a whole number of blocks for the row reader as well.
    ///
    /// Everything else is inherited, including the four-layer skeleton: layers
    /// 0-2 are DeltaNet, layer 3 is QSA, layer 1 carries the PLE injection, and
    /// layer 2 is the one the shipped file gives the odd expert dtypes to.
    pub fn quantizable() -> Self {
        Self {
            hidden: 256,
            hc_low_rank: 32,
            ple_row_dim: 32,
            ..Self::default()
        }
    }

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

/// One tensor of the fixture, independent of how it will be stored: a name, a
/// candle shape, and the range its contents are drawn from. Both writers build
/// the same list from [`tensor_specs`] and differ only in the dtype they give
/// each entry, which is what keeps the two variants the same file.
struct Spec {
    name: String,
    dims: Vec<usize>,
    fill: Fill,
}

impl Spec {
    fn new(name: &str, dims: &[usize], fill: Fill) -> Self {
        Self {
            name: name.to_string(),
            dims: dims.to_vec(),
            fill,
        }
    }

    /// The tensor's contents. They depend only on its name, so the same tensor
    /// carries the same floats in both variants and a value read back from one
    /// file can be compared against the other.
    fn values(&self) -> Vec<f32> {
        let (lo, hi) = self.fill.range();
        rand(seed_of(&self.name), self.dims.iter().product(), lo, hi)
    }

    fn tensor(&self) -> Result<Tensor> {
        Ok(Tensor::from_vec(
            self.values(),
            self.dims.clone(),
            &Device::Cpu,
        )?)
    }
}

/// The metadata block, in the key order GGUF wants (sorted, as the converter
/// emits): every key `XwenConfig::from_gguf` reads for this architecture.
fn metadata(geo: &TinyGeometry) -> Vec<(String, Value)> {
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
    kv
}

/// Every tensor the `qwen4exp` block loaders open, at `geo`'s geometry and in
/// the order the converter writes them (root first, then blocks).
fn tensor_specs(geo: &TinyGeometry) -> Vec<Spec> {
    let carrier = geo.carrier();
    // ---- root ----
    let mut ts: Vec<Spec> = vec![
        // llama.cpp ne {n_embd, n_vocab} -> candle [vocab, hidden].
        Spec::new("token_embd.weight", &[geo.vocab, geo.hidden], Fill::Weight),
        Spec::new("output.weight", &[geo.vocab, geo.hidden], Fill::Weight),
        // The tail mixer, which carries what other architectures call
        // output_norm.
        Spec::new("output_hc_norm.weight", &[carrier], Fill::Norm),
        Spec::new(
            "output_hc_down.weight",
            &[geo.hc_low_rank, carrier],
            Fill::Weight,
        ),
        Spec::new(
            "output_hc_up.weight",
            &[carrier, geo.hc_low_rank],
            Fill::Weight,
        ),
        // The flat n-gram table: llama.cpp ne {ple_head_dim, ple_rows} -> candle
        // [rows, row_dim], which is the orientation `PleTable::open` reads.
        Spec::new(
            "per_layer_token_embd.weight",
            &[geo.ple_rows(), geo.ple_row_dim],
            Fill::Weight,
        ),
    ];

    // ---- blocks ----
    for il in 0..geo.n_layer {
        let p = format!("blk.{il}");

        // Two hyper-connection gates per layer: one before the token mixer, one
        // before the MoE. There is no attn_norm and no post_attention_norm on
        // this architecture — these gates replace both.
        for gate in ["hc_attn", "hc_ffn"] {
            ts.push(Spec::new(
                &format!("{p}.{gate}_norm.weight"),
                &[carrier],
                Fill::Norm,
            ));
            ts.push(Spec::new(
                &format!("{p}.{gate}_down.weight"),
                &[geo.hc_low_rank, carrier],
                Fill::Weight,
            ));
            ts.push(Spec::new(
                &format!("{p}.{gate}_up.weight"),
                &[carrier, geo.hc_low_rank],
                Fill::Weight,
            ));
            ts.push(Spec::new(
                &format!("{p}.{gate}_inject.weight"),
                &[geo.hc_count, carrier],
                Fill::Weight,
            ));
        }

        if geo.is_full(il) {
            // `attn_q` is DOUBLE width: per-head interleaved [q_head, gate_head].
            ts.push(Spec::new(
                &format!("{p}.attn_q.weight"),
                &[2 * geo.n_head * geo.head_dim, geo.hidden],
                Fill::Weight,
            ));
            for name in ["attn_k", "attn_v"] {
                ts.push(Spec::new(
                    &format!("{p}.{name}.weight"),
                    &[geo.n_head_kv * geo.head_dim, geo.hidden],
                    Fill::Weight,
                ));
            }
            ts.push(Spec::new(
                &format!("{p}.attn_output.weight"),
                &[geo.hidden, geo.n_head * geo.head_dim],
                Fill::Weight,
            ));
            for name in ["attn_q_norm", "attn_k_norm"] {
                ts.push(Spec::new(
                    &format!("{p}.{name}.weight"),
                    &[geo.head_dim],
                    Fill::Norm,
                ));
            }
            // The indexer key side is MQA: `k_proj` is exactly one head.
            ts.push(Spec::new(
                &format!("{p}.indexer.q_proj.weight"),
                &[geo.indexer_heads * geo.indexer_head_dim, geo.hidden],
                Fill::Weight,
            ));
            ts.push(Spec::new(
                &format!("{p}.indexer.k_proj.weight"),
                &[geo.indexer_head_dim, geo.hidden],
                Fill::Weight,
            ));
            for name in ["indexer.q_norm", "indexer.k_norm"] {
                ts.push(Spec::new(
                    &format!("{p}.{name}.weight"),
                    &[geo.indexer_head_dim],
                    Fill::Norm,
                ));
            }
        } else {
            // DeltaNet. The projections ship under attention tensor names.
            ts.push(Spec::new(
                &format!("{p}.attn_qkv.weight"),
                &[geo.conv_dim(), geo.hidden],
                Fill::Weight,
            ));
            ts.push(Spec::new(
                &format!("{p}.attn_gate.weight"),
                &[geo.ssm_inner(), geo.hidden],
                Fill::Weight,
            ));
            ts.push(Spec::new(
                &format!("{p}.ssm_out.weight"),
                &[geo.hidden, geo.ssm_inner()],
                Fill::Weight,
            ));
            // Kernel-major in the file — ne {conv_kernel, conv_dim} — which
            // candle reads as [conv_dim, conv_kernel] and the block transposes.
            ts.push(Spec::new(
                &format!("{p}.ssm_conv1d.weight"),
                &[geo.conv_dim(), geo.conv_kernel],
                Fill::Weight,
            ));
            for name in ["ssm_beta", "ssm_alpha"] {
                ts.push(Spec::new(
                    &format!("{p}.{name}.weight"),
                    &[geo.ssm_v_heads, geo.hidden],
                    Fill::Weight,
                ));
            }
            // `ssm_a` has no suffix at all and `ssm_dt` is bias-suffixed (it is
            // the dt offset vector, a bias in name only); `dense_f32_any` finds
            // both.
            ts.push(Spec::new(
                &format!("{p}.ssm_a"),
                &[geo.ssm_v_heads],
                Fill::SsmA,
            ));
            ts.push(Spec::new(
                &format!("{p}.ssm_dt.bias"),
                &[geo.ssm_v_heads],
                Fill::Dt,
            ));
            // The gated RMS norm is per V-head, so it spans one head dim.
            ts.push(Spec::new(
                &format!("{p}.ssm_norm.weight"),
                &[geo.ssm_state_size],
                Fill::Norm,
            ));
        }

        if geo.is_ple(il) {
            let emb = geo.ple_emb_dim();
            // Both projections read the PLE embedding width, NOT the hidden size
            // (see the module header).
            ts.push(Spec::new(
                &format!("{p}.ple_key.weight"),
                &[carrier, emb],
                Fill::Weight,
            ));
            ts.push(Spec::new(
                &format!("{p}.ple_value.weight"),
                &[geo.hidden, emb],
                Fill::Weight,
            ));
            // All three norms span the whole carrier.
            for name in ["ple_norm_key", "ple_norm_query", "ple_norm_conv"] {
                ts.push(Spec::new(
                    &format!("{p}.{name}.weight"),
                    &[carrier],
                    Fill::Norm,
                ));
            }
            ts.push(Spec::new(
                &format!("{p}.ple_conv1d.weight"),
                &[carrier, geo.ple_conv_kernel],
                Fill::Weight,
            ));
        }

        // Every layer is MoE.
        ts.push(Spec::new(
            &format!("{p}.ffn_gate_inp.weight"),
            &[geo.n_expert, geo.hidden],
            Fill::Weight,
        ));
        for name in ["ffn_gate_exps", "ffn_up_exps"] {
            ts.push(Spec::new(
                &format!("{p}.{name}.weight"),
                &[geo.n_expert, geo.expert_ff, geo.hidden],
                Fill::Weight,
            ));
        }
        ts.push(Spec::new(
            &format!("{p}.ffn_down_exps.weight"),
            &[geo.n_expert, geo.hidden, geo.expert_ff],
            Fill::Weight,
        ));
        // The shared expert's router is one [hidden] vector: a scalar gate per
        // token, not a per-expert distribution.
        ts.push(Spec::new(
            &format!("{p}.ffn_gate_inp_shexp.weight"),
            &[geo.hidden],
            Fill::Weight,
        ));
        for name in ["ffn_gate_shexp", "ffn_up_shexp"] {
            ts.push(Spec::new(
                &format!("{p}.{name}.weight"),
                &[geo.shared_expert_ff, geo.hidden],
                Fill::Weight,
            ));
        }
        ts.push(Spec::new(
            &format!("{p}.ffn_down_shexp.weight"),
            &[geo.hidden, geo.shared_expert_ff],
            Fill::Weight,
        ));
    }

    ts
}

/// Writes a single-file `qwen4exp` GGUF of random-but-deterministic weights to
/// `path`, overwriting whatever is there. Every tensor is stored F32.
///
/// The file carries every metadata key `XwenConfig::from_gguf` reads for this
/// architecture and every tensor the block loaders open, at `geo`'s geometry.
/// Values depend only on the tensor name, so two writes of the same geometry
/// produce identical bytes.
pub fn write_tiny_qwen4exp(path: &Path, geo: &TinyGeometry) -> Result<()> {
    let kv = metadata(geo);
    let ts: Vec<(String, QTensor)> = tensor_specs(geo)
        .into_iter()
        .map(|s| {
            Ok((
                s.name.clone(),
                QTensor::quantize(&s.tensor()?, GgmlDType::F32)?,
            ))
        })
        .collect::<Result<_>>()?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let kv_refs: Vec<(&str, &Value)> = kv.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let t_refs: Vec<(&str, &QTensor)> = ts.iter().map(|(k, t)| (k.as_str(), t)).collect();
    let mut f = std::fs::File::create(path)?;
    gguf_file::write(&mut f, &kv_refs, &t_refs)?;
    Ok(())
}

/// The dtype the shipped Unsloth `UD-Q4_K_XL` file stores `name` at, per
/// `docs/qwen4exp-tensors.md` §2 (read out of the real headers).
///
/// This is the whole content of the mixed variant: the tensor list, the shapes
/// and the values are the F32 fixture's, and only this mapping differs. Layer
/// index matters for exactly one subsystem — the experts — where the real file
/// is NOT uniform, and the tiny file reproduces that on its own layer 2 (which
/// is layer 2 upstream too: `ffn_{gate,up}_exps` Q5_K there and Q4_K
/// everywhere else, `ffn_down_exps` Q8_0 there and Q5_1 everywhere else).
///
/// Panics on a name it has no entry for, so adding a tensor to
/// [`tensor_specs`] forces a dtype decision here rather than silently
/// defaulting to F32 and quietly un-covering a dispatch arm.
pub(crate) fn mixed_stored_dtype(name: &str) -> StoredDtype {
    let q8 = StoredDtype::Ggml(GgmlDType::Q8_0);
    let f32 = StoredDtype::Ggml(GgmlDType::F32);

    // `blk.N.` prefix off, so the table below is one row per PLANE the way the
    // tensor doc lists them.
    let (layer, plane) = match name.strip_prefix("blk.") {
        Some(rest) => match rest.split_once('.') {
            Some((il, plane)) => (il.parse::<usize>().ok(), plane),
            None => (None, rest),
        },
        None => (None, name),
    };

    match plane {
        // Embeddings / head, both untied and both Q8_0.
        "token_embd.weight" | "output.weight" => q8,
        // The PLE n-gram table: the one plane candle's parser cannot name.
        "per_layer_token_embd.weight" => StoredDtype::Raw(RawDtype::Iq4Nl),
        // Hyper-connections, per layer and on the tail. The low-rank pair is
        // quantized; the norm and the injection head are not.
        "output_hc_norm.weight"
        | "hc_attn_norm.weight"
        | "hc_ffn_norm.weight"
        | "hc_attn_inject.weight"
        | "hc_ffn_inject.weight" => f32,
        "output_hc_down.weight"
        | "output_hc_up.weight"
        | "hc_attn_down.weight"
        | "hc_attn_up.weight"
        | "hc_ffn_down.weight"
        | "hc_ffn_up.weight" => q8,
        // QSA attention.
        "attn_q.weight" | "attn_k.weight" | "attn_v.weight" | "attn_output.weight" => q8,
        "attn_q_norm.weight" | "attn_k_norm.weight" => f32,
        // The indexer projections are on the converter's quantize skip list and
        // arrive at the source precision in every mix.
        "indexer.q_proj.weight" | "indexer.k_proj.weight" => StoredDtype::Ggml(GgmlDType::BF16),
        "indexer.q_norm.weight" | "indexer.k_norm.weight" => f32,
        // Gated DeltaNet: the three projections quantized, the small per-head
        // vectors and the conv left alone.
        "attn_qkv.weight" | "attn_gate.weight" | "ssm_out.weight" => q8,
        "ssm_conv1d.weight" | "ssm_alpha.weight" | "ssm_beta.weight" | "ssm_a" | "ssm_dt.bias"
        | "ssm_norm.weight" => f32,
        // PLE injection layer.
        "ple_key.weight" | "ple_value.weight" => q8,
        "ple_norm_key.weight"
        | "ple_norm_query.weight"
        | "ple_norm_conv.weight"
        | "ple_conv1d.weight" => f32,
        // MoE. Routers F32, shared expert Q8_0, routed experts per-layer.
        "ffn_gate_inp.weight" | "ffn_gate_inp_shexp.weight" => f32,
        "ffn_gate_shexp.weight" | "ffn_up_shexp.weight" | "ffn_down_shexp.weight" => q8,
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" => {
            if layer == Some(2) {
                StoredDtype::Ggml(GgmlDType::Q5K)
            } else {
                StoredDtype::Ggml(GgmlDType::Q4K)
            }
        }
        "ffn_down_exps.weight" => {
            if layer == Some(2) {
                q8
            } else {
                StoredDtype::Ggml(GgmlDType::Q5_1)
            }
        }
        _ => panic!(
            "tensor {name} has no entry in the shipped file's dtype table — add one from \
             docs/qwen4exp-tensors.md"
        ),
    }
}

/// The ggml type id a stored dtype is written under in the tensor table.
///
/// candle's own `GgmlDType::to_u32` is `pub(crate)`, so this is a second
/// transcription of the same ggml enum — the same reason `gguf.rs` carries
/// `candle_dtype`, and the round trip through `GgufFile::stored_dtype_of` in
/// the tests below is what keeps the two agreeing.
fn ggml_type_id(dtype: StoredDtype) -> u32 {
    match dtype {
        StoredDtype::Ggml(d) => match d {
            GgmlDType::F32 => 0,
            GgmlDType::F16 => 1,
            GgmlDType::Q4_0 => 2,
            GgmlDType::Q4_1 => 3,
            GgmlDType::Q5_0 => 6,
            GgmlDType::Q5_1 => 7,
            GgmlDType::Q8_0 => 8,
            GgmlDType::Q8_1 => 9,
            GgmlDType::Q2K => 10,
            GgmlDType::Q3K => 11,
            GgmlDType::Q4K => 12,
            GgmlDType::Q5K => 13,
            GgmlDType::Q6K => 14,
            GgmlDType::Q8K => 15,
            GgmlDType::BF16 => 30,
        },
        StoredDtype::Raw(RawDtype::Iq4Nl) => 20,
        StoredDtype::Raw(other) => {
            panic!("the fixture writer cannot emit {other:?} — no packer for it")
        }
    }
}

/// Quantizes `values` into IQ4_NL blocks in ggml's on-disk layout: an f16 scale
/// followed by 16 bytes whose low nibble is element `j` and whose high nibble
/// is element `j + 16`.
///
/// This mirrors `quantize_row_iq4_nl` in shape, not bit-for-bit. ggml runs a
/// fixed candidate ladder over the scale and re-fits; this scans a small grid
/// around both ends of the codebook and keeps the lowest squared error. Both
/// end at the same thing — one f16 scale plus 32 nearest-codebook indices — and
/// the fixture only has to be dequantizable and close to its source, so
/// reproducing ggml's search would buy nothing the tests could observe.
///
/// Both ends are tried because the codebook is asymmetric (-127..113): pinning
/// the block's extremum to `kvalues[0]` wastes range when the extremum is on
/// the positive side, and vice versa. The candidate scale is rounded to f16
/// BEFORE the indices are chosen, so the indices are fitted against the scale
/// the file will actually carry.
fn quantize_iq4nl(values: &[f32]) -> Vec<u8> {
    assert!(
        values.len().is_multiple_of(QK4_NL),
        "IQ4_NL input length {} is not a whole number of {QK4_NL}-element blocks",
        values.len()
    );

    let fit = |d: f32, block: &[f32]| -> (f32, [u8; 32]) {
        let mut idx = [0u8; 32];
        let mut sse = 0f32;
        for (i, &x) in block.iter().enumerate() {
            let mut best = (f32::INFINITY, 0u8);
            for (k, &level) in KVALUES_IQ4NL.iter().enumerate() {
                let err = (x - d * f32::from(level)).powi(2);
                if err < best.0 {
                    best = (err, k as u8);
                }
            }
            sse += best.0;
            idx[i] = best.1;
        }
        (sse, idx)
    };

    let mut out = Vec::with_capacity(values.len() / QK4_NL * BLOCK_BYTES);
    for block in values.chunks_exact(QK4_NL) {
        // The signed extremum, which is what sets the scale: `max` keeps the
        // sign so a block that is mostly negative gets a negative scale, the
        // same way ggml's quantizer does.
        let max = block
            .iter()
            .copied()
            .fold(0f32, |m, v| if v.abs() > m.abs() { v } else { m });

        let mut best: Option<(f32, f16, [u8; 32])> = None;
        if max != 0.0 {
            for end in [KVALUES_IQ4NL[0], KVALUES_IQ4NL[15]] {
                let base = max / f32::from(end);
                for step in 0..=20 {
                    let d = f16::from_f32(base * (0.9 + 0.01 * step as f32));
                    if d == f16::ZERO {
                        continue;
                    }
                    let (sse, idx) = fit(f32::from(d), block);
                    if best.as_ref().is_none_or(|(b, _, _)| sse < *b) {
                        best = Some((sse, d, idx));
                    }
                }
            }
        }
        // An all-zero block dequantizes to zero for any indices, so the scale
        // carries the whole answer and the nibbles stay zero.
        let (d, idx) = best.map_or((f16::ZERO, [0u8; 32]), |(_, d, idx)| (d, idx));

        out.extend_from_slice(&d.to_le_bytes());
        for j in 0..QK4_NL / 2 {
            out.push((idx[j] & 0xf) | (idx[j + QK4_NL / 2] << 4));
        }
    }
    out
}

/// One tensor as it goes into the file: raw bytes plus the ggml type id naming
/// their layout. The mixed writer works at this level because candle's GGUF
/// writer takes `QTensor`s, and no `QTensor` can hold IQ4_NL.
struct RawEntry {
    name: String,
    /// candle order (`[out, in]`); the writer reverses them into GGUF's `ne`.
    dims: Vec<usize>,
    type_id: u32,
    data: Vec<u8>,
}

/// GGUF's tensor-data alignment. The `general.alignment` key is absent from
/// this fixture, which means the default, which is this.
const ALIGN: usize = 32;

fn gguf_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// The GGUF value-type id of a metadata value. candle's `ValueType::to_u32` is
/// private, so this is the same enum transcribed from the GGUF spec.
fn value_type_id(v: &Value) -> u32 {
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array(_) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

fn write_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(x) => out.push(*x),
        Value::I8(x) => out.push(*x as u8),
        Value::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::Bool(x) => out.push(u8::from(*x)),
        Value::String(s) => gguf_string(out, s),
        Value::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::Array(items) => {
            // An empty array's element type is unobservable; U32 is what
            // candle's own writer picks for one.
            let elem = items.first().map_or(4, value_type_id);
            out.extend_from_slice(&elem.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                write_value(out, item);
            }
        }
    }
}

/// Serializes a GGUF v3 file by hand and writes it to `path`.
///
/// candle's `gguf_file::write` cannot produce this file: it takes `QTensor`s,
/// and candle has no dtype — and therefore no `QTensor` — for IQ4_NL. Layout is
/// the spec's: magic, version, counts, the metadata block, the tensor table
/// (`ne` fastest-first, so candle dims reversed), then the tensor data padded
/// to [`ALIGN`] both at its start and after every tensor.
fn write_raw_gguf(path: &Path, kv: &[(String, Value)], tensors: &[RawEntry]) -> Result<()> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(kv.len() as u64).to_le_bytes());
    for (k, v) in kv {
        gguf_string(&mut buf, k);
        buf.extend_from_slice(&value_type_id(v).to_le_bytes());
        write_value(&mut buf, v);
    }

    let mut offset = 0u64;
    for t in tensors {
        gguf_string(&mut buf, &t.name);
        buf.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in t.dims.iter().rev() {
            buf.extend_from_slice(&(*d as u64).to_le_bytes());
        }
        buf.extend_from_slice(&t.type_id.to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
        offset += (t.data.len() as u64).div_ceil(ALIGN as u64) * ALIGN as u64;
    }

    buf.resize(buf.len().div_ceil(ALIGN) * ALIGN, 0);
    for t in tensors {
        buf.extend_from_slice(&t.data);
        buf.resize(buf.len().div_ceil(ALIGN) * ALIGN, 0);
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, &buf)?;
    Ok(())
}

/// Writes the same fixture [`write_tiny_qwen4exp`] writes — same metadata, same
/// tensors, same shapes, same values — with every tensor stored at the dtype the
/// shipped Unsloth file stores it at ([`mixed_stored_dtype`]).
///
/// This is the fixture for anything downstream of a dtype decision: the BF16
/// indexer projections, the Q8_0 attention/DeltaNet/shared-expert planes, the
/// Q4_K and Q5_K routed experts, the Q5_1 `ffn_down_exps`, and the IQ4_NL PLE
/// table that lands in the raw half of the tensor table because candle cannot
/// name type id 20.
///
/// `geo` must be block-friendly: pass [`TinyGeometry::quantizable`]. A geometry
/// whose quantized planes are not a whole number of blocks wide is refused here
/// rather than written out.
pub(crate) fn write_tiny_qwen4exp_mixed(path: &Path, geo: &TinyGeometry) -> Result<()> {
    let kv = metadata(geo);
    let mut entries = Vec::new();
    for spec in tensor_specs(geo) {
        let dtype = mixed_stored_dtype(&spec.name);
        let row = *spec.dims.last().expect("every fixture tensor has a shape");
        let data = match dtype {
            StoredDtype::Raw(RawDtype::Iq4Nl) => {
                ensure!(
                    row.is_multiple_of(QK4_NL),
                    "{}: a {row}-wide row is not a whole number of {QK4_NL}-element IQ4_NL blocks",
                    spec.name
                );
                quantize_iq4nl(&spec.values())
            }
            StoredDtype::Ggml(d) => {
                ensure!(
                    row.is_multiple_of(d.block_size()),
                    "{}: a {row}-wide row is not a whole number of {:?} blocks ({})",
                    spec.name,
                    d,
                    d.block_size()
                );
                QTensor::quantize(&spec.tensor()?, d)?.data()?.into_owned()
            }
            StoredDtype::Raw(other) => bail!("{}: no packer for {other:?}", spec.name),
        };
        entries.push(RawEntry {
            name: spec.name,
            dims: spec.dims,
            type_id: ggml_type_id(dtype),
            data,
        });
    }
    write_raw_gguf(path, &kv, &entries)
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
    use crate::qwen4exp::iq4nl;
    use crate::qwen4exp::ple::{PleLayer, PleTable};
    use crate::rope::Rope;

    /// Removes the fixture directory on the way in and on the way out, so a
    /// panicking run does not leave a stale file for the next one. `tag` keeps
    /// concurrently running tests in separate directories — they share a
    /// process, and the wipe-on-entry would otherwise delete a sibling's file.
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let p =
                std::env::temp_dir().join(format!("xwen_tiny_q4e_{tag}_{}", std::process::id()));
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

    /// The values one named fixture tensor carries, for a test that wants to
    /// compare what came back out of the file against what went in.
    fn source_values(geo: &TinyGeometry, name: &str) -> Vec<f32> {
        tensor_specs(geo)
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("no fixture tensor named {name}"))
            .values()
    }

    /// The whole point of the fixture: a file this module writes opens through
    /// the production GGUF reader, parses into a `XwenConfig` with the geometry
    /// it was written at, and satisfies every `qwen4exp` block loader's shape
    /// contract.
    #[test]
    fn tiny_qwen4exp_file_loads_every_block() {
        let dir = TempDir::new("f32");
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

    /// The mixed variant is a real file: it parses, it configures, and every
    /// block loader takes it — including the ones that branch on stored dtype
    /// (the BF16 indexer projections, the Q8_0 attention planes, the
    /// Q4_K/Q5_K/Q5_1 experts), which the all-F32 fixture cannot reach.
    ///
    /// The PLE table is the load-bearing half: candle's parser refuses type id
    /// 20 outright, so a table that did not land in xwen's raw half would take
    /// the whole file down with it. If that split ever regressed, an all-F32
    /// fixture would stay green.
    #[test]
    fn mixed_dtype_fixture_loads_every_block() {
        let dir = TempDir::new("mixed");
        let path = dir.0.join("tiny-qwen4exp-mixed.gguf");
        let geo = TinyGeometry::quantizable();
        write_tiny_qwen4exp_mixed(&path, &geo).expect("writing the mixed-dtype GGUF");

        let device = metal_device().unwrap_or(candle_core::Device::Cpu);
        let gguf = crate::gguf::open(&path, &device).expect("opening the mixed-dtype GGUF");

        let cfg = XwenConfig::from_gguf(&gguf.content).expect("parsing the mixed-dtype config");
        assert_eq!(cfg.n_layer, 4);
        assert_eq!(cfg.hidden, 256);
        let q4 = cfg.qwen4exp.as_ref().expect("a qwen4exp config block");
        assert_eq!(q4.hc_low_rank, 32);
        assert_eq!(q4.ple.as_ref().expect("a PLE config block").row_dim, 32);

        // The tensor table is split: everything but the n-gram table is candle's,
        // the n-gram table is xwen's.
        assert_eq!(gguf.tensor_count(), tensor_specs(&geo).len());
        assert_eq!(
            gguf.raw_tensor_names(),
            vec![("per_layer_token_embd.weight", crate::gguf::RawDtype::Iq4Nl)]
        );

        let w = Weights::from_gguf(gguf.clone());
        let rope = Arc::new(Rope::new(cfg.rope(), 128, &device).expect("rope tables"));

        HcRead::load(&w.pp("blk.0"), "hc_attn", &cfg, true).expect("Q8_0 hyper-connection gate");
        HcRead::load(&w, "output_hc", &cfg, false).expect("Q8_0 tail mixer");
        QsaIndexer::load(&w.pp("blk.3"), &cfg, rope.clone()).expect("BF16 indexer projections");
        AttnBlock::new(&w.pp("blk.3"), &cfg, 3, rope, AttnWeights::F16).expect("Q8_0 attention");
        LinearAttnBlock::new(&w.pp("blk.0"), &cfg, AttnWeights::F16).expect("Q8_0 DeltaNet");
        // Layer 0 has Q4_K gate/up and Q5_1 down; layer 2 has Q5_K gate/up and
        // Q8_0 down — the shipped file's two expert mixes, both loaded.
        MoeBlock::new(&w.pp("blk.0"), &cfg, ExpertRunner::Reference).expect("Q4_K/Q5_1 experts");
        MoeBlock::new(&w.pp("blk.2"), &cfg, ExpertRunner::Reference).expect("Q5_K/Q8_0 experts");
        w.qtensor("token_embd").expect("Q8_0 embedding table");
        w.qlinear_with_buffer("output").expect("Q8_0 lm head");

        if gguf.mmap_source().is_some() {
            // `raw_tensor` is the only way to the table's bytes, and the PLE
            // reader is the only thing that can decode them.
            let raw = gguf
                .raw_tensor("per_layer_token_embd.weight")
                .expect("the raw half serves the IQ4_NL table");
            assert_eq!(raw.shape, vec![geo.ple_rows(), geo.ple_row_dim]);
            assert_eq!(raw.len, geo.ple_rows() * BLOCK_BYTES);

            let table =
                PleTable::open(&gguf, "per_layer_token_embd.weight").expect("opening the table");
            assert_eq!(table.rows(), geo.ple_rows() as u64);
            assert_eq!(table.row_dim(), geo.ple_row_dim);
            PleLayer::load(&w.pp("blk.1"), &gguf, &cfg).expect("Q8_0 PLE projections");
        } else {
            eprintln!(
                "skipping the raw-table assertions: no file mapping (non-Metal device, or \
                 XWEN_LOAD_CLASSIC)"
            );
        }
    }

    /// Every tensor lands in the file at the dtype the shipped Unsloth file
    /// stores it at, read back through the production accessor rather than
    /// trusted from the writer's own table.
    ///
    /// This is what makes the fixture worth having: a writer bug that quietly
    /// stored a plane F32 — or a `ggml_type_id` transcription that disagreed
    /// with candle's private one — would leave the file loading fine and the
    /// dtype-dispatch arms untested, which is the exact hole this variant
    /// exists to close. The counts below pin that the mix is actually mixed.
    #[test]
    fn mixed_dtype_fixture_stores_every_tensor_at_the_real_files_dtype() {
        let dir = TempDir::new("dtypes");
        let path = dir.0.join("tiny-qwen4exp-mixed.gguf");
        let geo = TinyGeometry::quantizable();
        write_tiny_qwen4exp_mixed(&path, &geo).expect("writing the mixed-dtype GGUF");

        let device = metal_device().unwrap_or(candle_core::Device::Cpu);
        let gguf = crate::gguf::open(&path, &device).expect("opening the mixed-dtype GGUF");

        let mut seen: std::collections::BTreeMap<String, usize> = Default::default();
        for spec in tensor_specs(&geo) {
            let want = mixed_stored_dtype(&spec.name);
            let got = gguf
                .stored_dtype_of(&spec.name)
                .unwrap_or_else(|e| panic!("{}: {e}", spec.name));
            assert_eq!(got, want, "{} stored dtype", spec.name);
            *seen.entry(format!("{want:?}")).or_default() += 1;
        }

        // Named planes, not just a count: these are the dispatch arms the
        // all-F32 fixture cannot reach.
        assert_eq!(
            gguf.stored_dtype_of("blk.3.indexer.q_proj.weight").unwrap(),
            StoredDtype::Ggml(GgmlDType::BF16)
        );
        assert_eq!(
            gguf.stored_dtype_of("blk.0.ffn_gate_exps.weight").unwrap(),
            StoredDtype::Ggml(GgmlDType::Q4K)
        );
        assert_eq!(
            gguf.stored_dtype_of("blk.2.ffn_gate_exps.weight").unwrap(),
            StoredDtype::Ggml(GgmlDType::Q5K),
            "layer 2 is the one the shipped file gives Q5_K experts"
        );
        assert_eq!(
            gguf.stored_dtype_of("blk.0.ffn_down_exps.weight").unwrap(),
            StoredDtype::Ggml(GgmlDType::Q5_1)
        );
        assert_eq!(
            gguf.stored_dtype_of("blk.2.ffn_down_exps.weight").unwrap(),
            StoredDtype::Ggml(GgmlDType::Q8_0)
        );
        assert_eq!(
            gguf.stored_dtype_of("per_layer_token_embd.weight").unwrap(),
            StoredDtype::Raw(RawDtype::Iq4Nl)
        );

        // Every dtype the shipped file uses is present, so no arm is covered by
        // an empty set.
        for want in [
            "Ggml(F32)",
            "Ggml(BF16)",
            "Ggml(Q8_0)",
            "Ggml(Q4K)",
            "Ggml(Q5K)",
            "Ggml(Q5_1)",
            "Raw(Iq4Nl)",
        ] {
            assert!(
                seen.get(want).copied().unwrap_or(0) > 0,
                "no tensor stored as {want}: {seen:?}"
            );
        }
    }

    /// The n-gram table survives the quantizer: a row read back through the
    /// production PLE reader reconstructs the floats it was written from, to
    /// within one IQ4_NL codebook step.
    ///
    /// Both halves are pinned. A quantizer that packed the nibbles in the wrong
    /// order would still produce a dequantizable block — the row would keep its
    /// norm and every shape — so the comparison is elementwise against the
    /// source, which is the only thing a permutation fails.
    #[test]
    fn the_iq4nl_quantizer_round_trips_the_ple_table_rows() {
        let geo = TinyGeometry::quantizable();
        let src = source_values(&geo, "per_layer_token_embd.weight");
        let packed = quantize_iq4nl(&src);
        assert_eq!(packed.len(), geo.ple_rows() * BLOCK_BYTES);

        // Straight through the row dequantizer first: no file, no mapping, so a
        // failure here is the quantizer's and nothing else's.
        let mut got = vec![0f32; src.len()];
        iq4nl::dequant_row(&packed, &mut got);
        let mut max_err = 0f32;
        let mut sum_err = 0f32;
        for (a, b) in src.iter().zip(got.iter()) {
            let e = (a - b).abs();
            max_err = max_err.max(e);
            sum_err += e;
        }
        let mean_err = sum_err / src.len() as f32;
        // The fill is uniform on [-0.1, 0.1] and a block's scale is set by its
        // extremum, so the reconstruction error is bounded by half a codebook
        // step: the widest gap is 24 of the 240 codebook units, over a scale of
        // about 0.1/113, which is ~1.1e-2 at the worst and ~3e-3 on average.
        // The fill is seeded, so these are exact rather than probabilistic:
        // measured max 1.25e-2, mean 2.97e-3.
        assert!(
            max_err < 1.4e-2,
            "IQ4_NL round-trip max error {max_err} exceeds one codebook step"
        );
        assert!(mean_err < 3.2e-3, "IQ4_NL round-trip mean error {mean_err}");

        // And through the real path: the file, the raw tensor table, the mmap,
        // the PLE reader's row gather.
        let dir = TempDir::new("iq4nl");
        let path = dir.0.join("tiny-qwen4exp-mixed.gguf");
        write_tiny_qwen4exp_mixed(&path, &geo).expect("writing the mixed-dtype GGUF");
        let device = metal_device().unwrap_or(candle_core::Device::Cpu);
        let gguf = crate::gguf::open(&path, &device).expect("opening the mixed-dtype GGUF");
        if gguf.mmap_source().is_none() {
            eprintln!(
                "skipping the PleTable half: no file mapping (non-Metal device, or \
                 XWEN_LOAD_CLASSIC)"
            );
            return;
        }
        let table =
            PleTable::open(&gguf, "per_layer_token_embd.weight").expect("opening the table");
        let mut row = vec![0f32; geo.ple_row_dim];
        for r in [0usize, 1, geo.ple_rows() / 2, geo.ple_rows() - 1] {
            table.row(r as u64, &mut row).expect("reading a table row");
            let want = &got[r * geo.ple_row_dim..(r + 1) * geo.ple_row_dim];
            assert_eq!(row, want, "row {r} through PleTable");
        }
    }
}
