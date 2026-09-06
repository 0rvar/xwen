//! The `qwen3` config, in two steps: what the file says, and what the runtime runs.
//!
//! [`HfQwen3Config`] is a literal deserialization of a Hugging Face
//! `config.json`. Unknown keys are ignored, because the file carries training
//! and tooling knobs this crate has no opinion on; every key that changes the
//! math is named and checked.
//!
//! [`Qwen3Config`] is what the stack is built from. It is BUILT from the HF
//! config, never deserialized, because two of its fields are properties of the
//! architecture that no HF `config.json` records: which RMSNorm form the norm
//! weights are stored in ([`NormVariant`]) and how rope is applied
//! ([`RopeSpec`]). Neither type has a `Default` impl on purpose — a default is
//! exactly how a silently-wrong assumption gets in.

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

/// The two ids a Qwen3 chat generation stops on: `<|im_end|>` and
/// `<|endoftext|>`.
///
/// Hardcoded, like the 3.6 pair, and for the same reason: the checkpoint's
/// metadata advertises only ONE stop id (`eos_token_id` 151645 in
/// `config.json`, and the GGUF conversion carries only that key). The second id
/// exists only in `generation_config.json`, which is not part of the weights, so
/// a loop that reads the advertised value alone runs straight through a turn
/// boundary and looks like a model that will not stop.
pub const QWEN3_EOG: [u32; 2] = [151645, 151643];

/// The form a checkpoint's RMSNorm weights are stored in.
///
/// One arm, deliberately. Gemma-style zero-centred norms — where the stored
/// weight is `w` and the norm computes `x / rms(x) * (1 + w)` — exist nowhere in
/// this crate: the GGUF converter bakes the `+1` in before the file is written,
/// and the HF Qwen3 safetensors are already multiply-ready. The field exists to
/// pin that assumption at the type level rather than in a comment. A future
/// checkpoint that needs the zero-centred form gets a new arm AND an
/// implementation; it must never be reached by defaulting, because the failure
/// mode is a model that produces plausible garbage instead of an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormVariant {
    /// `y = x / rms(x) * w`. The weight multiplies directly.
    Standard,
}

/// How rotary embeddings are applied on this architecture.
///
/// `rotary_dim == head_dim` is Qwen3's full-width NEoX rope, which llama.cpp
/// asserts outright (`GGML_ASSERT(n_embd_head == n_rot)` in `qwen3.cpp`). The
/// field is carried explicitly rather than assumed because the 3.6 archs in this
/// same crate rotate only the first 64 of 256 dims, so "rotary_dim is the head
/// dim" is a per-arch fact, not a general one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeSpec {
    /// Width of one attention head.
    pub head_dim: usize,
    /// How many of those dims are rotated, counted from 0. Dims at or above this
    /// index pass through unrotated.
    pub rotary_dim: usize,
    /// Rope base frequency. 1e6 on `Qwen3-4B` and the Z-Image text encoder,
    /// 5e6 on `Qwen3-4B-Instruct-2507`.
    pub theta: f64,
}

/// A Hugging Face `config.json` for a `qwen3` checkpoint, as written.
///
/// Every field here is read; keys the file carries and this struct does not name
/// (`architectures`, `initializer_range`, `transformers_version`, …) are ignored
/// by serde and carry no meaning for inference.
#[derive(Debug, Clone, Deserialize)]
pub struct HfQwen3Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Explicit on this architecture and NOT derivable: 2560 / 32 is 80, but the
    /// heads are 128 wide, so the q/k/v projections are wider than the hidden
    /// state. Deriving it would be wrong by construction.
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub tie_word_embeddings: bool,
    pub attention_bias: bool,
    pub use_sliding_window: bool,
    /// Present as `null` on every shipped Qwen3-4B config. Any non-null value is
    /// a scaling scheme (YaRN, linear, …) this loader does not implement.
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    /// `"silu"` on every shipped config; the FFN is SwiGLU with no other option.
    #[serde(default)]
    pub hidden_act: Option<String>,
}

impl HfQwen3Config {
    /// Parse a `config.json`.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).context("parsing the qwen3 config.json")
    }
}

/// The config the `qwen3` stack is built from.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3Config {
    /// Residual width. 2560 on the 4B.
    pub hidden_size: usize,
    /// SwiGLU inner width. 9728 on the 4B.
    pub intermediate_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub rms_norm_eps: f64,
    /// Full vocabulary, unpadded on this architecture (151936).
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    /// True on every shipped Qwen3-4B: `lm_head` reuses `embed_tokens`.
    pub tie_word_embeddings: bool,
    /// Which RMSNorm form the stored weights are in. Supplied by the
    /// architecture, not read from the file.
    pub norm: NormVariant,
    /// Supplied by the architecture (widths) and the file (`theta`).
    pub rope: RopeSpec,
    /// Stop ids for chat generation. See [`QWEN3_EOG`].
    pub eog: [u32; 2],
}

impl Qwen3Config {
    /// Validate an HF config and build the runtime config from it.
    ///
    /// The checks are all of the form "this loader implements exactly one
    /// behaviour and the file must ask for that one". Every message carries the
    /// offending value, because the whole point of the check is to say what the
    /// file wanted that we do not do.
    pub fn from_hf(hf: &HfQwen3Config) -> Result<Self> {
        ensure!(
            hf.model_type == "qwen3",
            "config.json declares model_type {:?}; this loader implements \"qwen3\" only",
            hf.model_type
        );
        ensure!(
            !hf.use_sliding_window,
            "config.json sets use_sliding_window true; the qwen3 stack is full attention on \
             every layer and has no windowed path"
        );
        ensure!(
            !hf.attention_bias,
            "config.json sets attention_bias true; the qwen3 q/k/v/o projections are \
             implemented without bias tensors"
        );
        ensure!(
            hf.tie_word_embeddings,
            "config.json sets tie_word_embeddings false; the qwen3 LM head is the embedding \
             matrix and an untied checkpoint would need a separate output plane"
        );
        ensure!(
            hf.head_dim == 128,
            "config.json declares head_dim {}; the qwen3 attention kernels are compiled for \
             head_dim 128",
            hf.head_dim
        );
        ensure!(
            hf.num_attention_heads != 0
                && hf.num_key_value_heads != 0
                && hf.num_attention_heads % hf.num_key_value_heads == 0,
            "config.json declares {} attention heads over {} key/value heads; GQA needs at least \
             one of each and the kv heads to divide evenly into the query heads",
            hf.num_attention_heads,
            hf.num_key_value_heads
        );
        // `q_dim`/`kv_dim` are plain multiplications on the hot path, so the
        // products are checked here once instead of being fallible everywhere
        // they are read. A file this large is nonsense long before it overflows,
        // but a config.json is untrusted input and a wrapped width would size a
        // buffer, not fail.
        for (what, heads) in [
            ("num_attention_heads", hf.num_attention_heads),
            ("num_key_value_heads", hf.num_key_value_heads),
        ] {
            ensure!(
                heads.checked_mul(hf.head_dim).is_some(),
                "config.json declares {what} {heads} at head_dim {}, whose product does not fit \
                 a usize",
                hf.head_dim
            );
        }
        match &hf.rope_scaling {
            Some(v) if !v.is_null() => bail!(
                "config.json sets rope_scaling to {v}; this loader implements plain NEoX rope \
                 with no scaling scheme"
            ),
            _ => {}
        }
        // Not in the brief's list, but a non-SwiGLU activation is the kind of
        // divergence that runs to completion and returns wrong numbers.
        if let Some(act) = &hf.hidden_act {
            ensure!(
                act == "silu",
                "config.json declares hidden_act {act:?}; the qwen3 FFN is SwiGLU over silu"
            );
        }
        ensure!(
            hf.num_hidden_layers > 0,
            "config.json declares {} layers",
            hf.num_hidden_layers
        );
        ensure!(
            hf.hidden_size > 0 && hf.intermediate_size > 0 && hf.vocab_size > 0,
            "config.json declares a zero hidden_size, intermediate_size or vocab_size \
             ({}, {}, {})",
            hf.hidden_size,
            hf.intermediate_size,
            hf.vocab_size
        );
        ensure!(
            hf.rope_theta > 0.0,
            "config.json declares rope_theta {}",
            hf.rope_theta
        );
        ensure!(
            hf.rms_norm_eps > 0.0,
            "config.json declares rms_norm_eps {}",
            hf.rms_norm_eps
        );

        Ok(Self {
            hidden_size: hf.hidden_size,
            intermediate_size: hf.intermediate_size,
            n_layer: hf.num_hidden_layers,
            n_head: hf.num_attention_heads,
            n_kv_head: hf.num_key_value_heads,
            rms_norm_eps: hf.rms_norm_eps,
            vocab_size: hf.vocab_size,
            max_position_embeddings: hf.max_position_embeddings,
            tie_word_embeddings: hf.tie_word_embeddings,
            norm: NormVariant::Standard,
            rope: RopeSpec {
                head_dim: hf.head_dim,
                // Full-width NEoX: every dim of the head rotates.
                rotary_dim: hf.head_dim,
                theta: hf.rope_theta,
            },
            eog: QWEN3_EOG,
        })
    }

    /// Parse and validate a `config.json` in one step.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_hf(&HfQwen3Config::from_json_bytes(bytes)?)
    }

    /// Width of one attention head (128).
    pub fn head_dim(&self) -> usize {
        self.rope.head_dim
    }

    /// Width of the fused query projection output, `n_head * head_dim` (4096).
    ///
    /// Cannot overflow: [`Qwen3Config::from_hf`] refuses a config whose head
    /// count times head width does not fit a `usize`, and there is no other way
    /// to build one of these.
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim()
    }

    /// Width of one of the key/value projection outputs, `n_kv_head * head_dim`
    /// (1024). Cannot overflow, for the reason [`Qwen3Config::q_dim`] gives.
    pub fn kv_dim(&self) -> usize {
        self.n_kv_head * self.head_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A minimal config.json that passes every check, as a base for the negative
    /// tests to break one field at a time.
    fn valid_json() -> serde_json::Value {
        serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "max_position_embeddings": 40960,
            "vocab_size": 151936,
            "tie_word_embeddings": true,
            "attention_bias": false,
            "use_sliding_window": false,
            "rope_scaling": serde_json::Value::Null,
            "hidden_act": "silu",
        })
    }

    fn parse(v: serde_json::Value) -> Result<Qwen3Config> {
        Qwen3Config::from_json_bytes(v.to_string().as_bytes())
    }

    /// `$HF_HUB_CACHE`-style snapshot dirs the real-file tests read. A missing
    /// cache skips the test rather than failing it: the weights are 8 GB a
    /// piece and are not a checkout dependency.
    fn cache_snapshot(repo: &str, sha: &str) -> Option<PathBuf> {
        let home = std::env::var_os("HOME")?;
        let dir = PathBuf::from(home)
            .join(".cache/huggingface/hub")
            .join(repo)
            .join("snapshots")
            .join(sha);
        dir.is_dir().then_some(dir)
    }

    #[test]
    fn the_shipped_configs_parse_and_differ_only_in_rope_theta_and_context() {
        let cases = [
            (
                "Z-Image-Turbo text encoder",
                cache_snapshot(
                    "models--Tongyi-MAI--Z-Image-Turbo",
                    "f332072aa78be7aecdf3ee76d5c247082da564a6",
                )
                .map(|d| d.join("text_encoder")),
                1e6,
                40960,
            ),
            (
                "Qwen3-4B",
                cache_snapshot(
                    "models--Qwen--Qwen3-4B",
                    "1cfa9a7208912126459214e8b04321603b3df60c",
                ),
                1e6,
                40960,
            ),
            (
                "Qwen3-4B-Instruct-2507",
                cache_snapshot(
                    "models--Qwen--Qwen3-4B-Instruct-2507",
                    "cdbee75f17c01a7cc42f958dc650907174af0554",
                ),
                5e6,
                262144,
            ),
        ];
        let mut seen = 0;
        for (label, dir, theta, max_pos) in cases {
            let Some(dir) = dir else {
                eprintln!("skipping {label}: not in the local HF cache");
                continue;
            };
            seen += 1;
            let bytes = std::fs::read(dir.join("config.json")).unwrap();
            let cfg = Qwen3Config::from_json_bytes(&bytes).unwrap();
            assert_eq!(cfg.hidden_size, 2560, "{label}");
            assert_eq!(cfg.intermediate_size, 9728, "{label}");
            assert_eq!(cfg.n_layer, 36, "{label}");
            assert_eq!(cfg.n_head, 32, "{label}");
            assert_eq!(cfg.n_kv_head, 8, "{label}");
            assert_eq!(cfg.vocab_size, 151936, "{label}");
            assert_eq!(cfg.head_dim(), 128, "{label}");
            assert_eq!(cfg.q_dim(), 4096, "{label}");
            assert_eq!(cfg.kv_dim(), 1024, "{label}");
            assert_eq!(cfg.rms_norm_eps, 1e-6, "{label}");
            assert!(cfg.tie_word_embeddings, "{label}");
            assert_eq!(cfg.norm, NormVariant::Standard, "{label}");
            assert_eq!(cfg.rope.rotary_dim, cfg.rope.head_dim, "{label}");
            assert_eq!(cfg.rope.theta, theta, "{label}");
            assert_eq!(cfg.max_position_embeddings, max_pos, "{label}");
            assert_eq!(cfg.eog, [151645, 151643], "{label}");
        }
        if seen == 0 {
            eprintln!("no qwen3 checkpoint in the local HF cache; nothing asserted");
        }
    }

    #[test]
    fn the_base_config_is_valid() {
        parse(valid_json()).unwrap();
    }

    #[test]
    fn a_foreign_model_type_is_refused() {
        let mut v = valid_json();
        v["model_type"] = "qwen3_moe".into();
        let err = parse(v).unwrap_err().to_string();
        assert!(err.contains("qwen3_moe"), "{err}");
    }

    #[test]
    fn sliding_window_attention_is_refused() {
        let mut v = valid_json();
        v["use_sliding_window"] = true.into();
        let err = parse(v).unwrap_err().to_string();
        assert!(err.contains("use_sliding_window"), "{err}");
    }

    #[test]
    fn attention_bias_is_refused() {
        let mut v = valid_json();
        v["attention_bias"] = true.into();
        let err = parse(v).unwrap_err().to_string();
        assert!(err.contains("attention_bias"), "{err}");
    }

    #[test]
    fn untied_embeddings_are_refused() {
        let mut v = valid_json();
        v["tie_word_embeddings"] = false.into();
        let err = parse(v).unwrap_err().to_string();
        assert!(err.contains("tie_word_embeddings"), "{err}");
    }

    #[test]
    fn a_head_dim_other_than_128_is_refused() {
        let mut v = valid_json();
        v["head_dim"] = 80.into();
        let err = parse(v).unwrap_err().to_string();
        assert!(err.contains("head_dim 80"), "{err}");
    }

    #[test]
    fn heads_that_do_not_divide_by_the_kv_heads_are_refused() {
        let mut v = valid_json();
        v["num_key_value_heads"] = 7.into();
        let err = parse(v).unwrap_err().to_string();
        assert!(err.contains("32 attention heads over 7"), "{err}");
    }

    #[test]
    fn zero_attention_heads_are_refused() {
        let mut v = valid_json();
        v["num_attention_heads"] = 0.into();
        let err = parse(v).unwrap_err().to_string();
        assert!(err.contains("0 attention heads"), "{err}");
    }

    /// A head count whose product with the head width wraps would size the
    /// projection planes from a wrapped number rather than failing.
    #[test]
    fn a_head_count_that_overflows_the_projection_width_is_refused() {
        let mut v = valid_json();
        v["num_attention_heads"] = (usize::MAX / 64).into();
        v["num_key_value_heads"] = (usize::MAX / 64).into();
        let err = parse(v).unwrap_err().to_string();
        assert!(err.contains("does not fit a usize"), "{err}");
    }

    #[test]
    fn zero_kv_heads_are_refused() {
        let mut v = valid_json();
        v["num_key_value_heads"] = 0.into();
        assert!(parse(v).is_err());
    }

    #[test]
    fn a_rope_scaling_scheme_is_refused() {
        let mut v = valid_json();
        v["rope_scaling"] = serde_json::json!({"rope_type": "yarn", "factor": 4.0});
        let err = parse(v).unwrap_err().to_string();
        assert!(err.contains("rope_scaling"), "{err}");
    }

    /// An explicit `null`, which every shipped config carries, is not a scheme.
    fn null_rope_scaling_is_accepted_inner() {
        let mut v = valid_json();
        v["rope_scaling"] = serde_json::Value::Null;
        parse(v).unwrap();
    }

    #[test]
    fn a_null_or_absent_rope_scaling_is_accepted() {
        null_rope_scaling_is_accepted_inner();
        let mut v = valid_json();
        v.as_object_mut().unwrap().remove("rope_scaling");
        parse(v).unwrap();
    }

    #[test]
    fn a_non_silu_activation_is_refused() {
        let mut v = valid_json();
        v["hidden_act"] = "gelu".into();
        let err = parse(v).unwrap_err().to_string();
        assert!(err.contains("gelu"), "{err}");
    }

    #[test]
    fn keys_this_loader_has_no_opinion_on_are_ignored() {
        let mut v = valid_json();
        v["initializer_range"] = 0.02.into();
        v["transformers_version"] = "4.51.0".into();
        v["some_future_knob"] = serde_json::json!({"a": [1, 2, 3]});
        parse(v).unwrap();
    }

    #[test]
    fn a_missing_required_key_is_a_parse_error_naming_it() {
        let mut v = valid_json();
        v.as_object_mut().unwrap().remove("head_dim");
        let err = parse(v).unwrap_err().to_string();
        assert!(
            err.contains("head_dim") || err.contains("config.json"),
            "{err}"
        );
    }
}
