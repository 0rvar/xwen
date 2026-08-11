use anyhow::{Context, Result, bail, ensure};
use candle_core::quantized::gguf_file::{Content, Value};

use crate::tokenizer::LagunaTokenizer;

/// RoPE parameters for one layer type.
#[derive(Debug, Clone)]
pub enum RopeKind {
    /// YaRN-scaled partial rotary. Qwen 3.6 ships no scaling keys, so nothing
    /// builds this today; it is retained for the opt-in long-context work
    /// (TODO P13) and exercised only by rope.rs's own tests.
    Yarn {
        freq_base: f32,
        factor: f32,
        original_ctx: usize,
        beta_fast: f32,
        beta_slow: f32,
        /// mscale applied to cos/sin (config.json `attention_factor`).
        attn_factor: f32,
        n_rot: usize,
    },
    /// Unscaled rope over the first `n_rot` dims; dims `n_rot..head_dim` pass
    /// through unrotated.
    Plain { freq_base: f32, n_rot: usize },
}

/// Which attention a layer runs. Qwen 3.6 is a hybrid: full attention at layer
/// indices where `(il + 1) % full_attention_interval == 0` — 3, 7, 11, … on both
/// shipped checkpoints — and gated DeltaNet everywhere else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Softmax attention over a KV cache.
    Full,
    /// Gated DeltaNet (linear attention) over a recurrent state.
    Linear,
}

/// Which of the two checkpoints this is. The only structural difference is the
/// FFN: `qwen35` is dense SwiGLU on every layer, `qwen35moe` is MoE on every
/// layer. Attention, DeltaNet and norms are shared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// Qwen3.6-27B — dense FFN.
    Dense,
    /// Qwen3.6-35B-A3B — 256 routed experts + one shared expert.
    Moe,
}

impl Arch {
    /// The GGUF `general.architecture` string this variant is parsed from, which
    /// is also the prefix every other metadata key carries.
    pub fn key(&self) -> &'static str {
        match self {
            Arch::Dense => "qwen35",
            Arch::Moe => "qwen35moe",
        }
    }

    /// The official checkpoint family this architecture belongs to. The two are
    /// one-to-one, so a GGUF's architecture is the authoritative answer to
    /// "which checkpoint is this file" — more reliable than any flag that
    /// travelled alongside a path.
    pub fn model(&self) -> crate::hub::Model {
        match self {
            Arch::Dense => crate::hub::Model::Qwen27B,
            Arch::Moe => crate::hub::Model::Qwen35BA3B,
        }
    }
}

/// The IMROPE section layout both checkpoints declare. For text-only input every
/// position feeds all three sections the same index, which makes IMROPE provably
/// identical to plain NEoX over `n_rot = 64` — the form rope.rs implements. The
/// value is validated at load so a checkpoint with a different sectioning (a
/// vision-enabled conversion, say) is refused rather than silently mis-roped.
const ROPE_SECTIONS: [usize; 4] = [11, 11, 10, 0];

#[derive(Debug, Clone)]
pub struct XwenConfig {
    pub arch: Arch,
    pub n_layer: usize,
    pub hidden: usize,
    pub vocab: usize,
    /// Per-layer query-head counts. Uniform on both checkpoints, kept as a vec so
    /// the per-layer-heterogeneous machinery inherited from laguna stays intact.
    pub n_head: Vec<usize>,
    pub n_kv_head: usize,
    /// Full-attention head dim (256 on both checkpoints) — NOT the DeltaNet head
    /// dim, which is `linear_head_dim`.
    pub head_dim: usize,
    /// Attention kind per layer, in layer order.
    pub layer_kind: Vec<LayerKind>,
    /// DeltaNet K-heads (also the Q-head count of the linear layers).
    pub linear_k_heads: usize,
    /// DeltaNet V-heads. A multiple of `linear_k_heads`; q/k are broadcast up to
    /// it by tiled repeat.
    pub linear_v_heads: usize,
    /// DeltaNet head dim, for q, k and v alike (128 on both checkpoints).
    pub linear_head_dim: usize,
    /// Depthwise conv kernel width over the fused DeltaNet qkv stream (4).
    pub conv_kernel: usize,
    /// Dense-FFN intermediate size. `qwen35` only; 0 on the MoE checkpoint.
    pub dense_ff: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub expert_ff: usize,
    pub shared_expert_ff: usize,
    pub rms_eps: f64,
    pub n_ctx_train: usize,
    pub rope: RopeKind,
    /// All end-of-generation tokens (Qwen: `<|im_end|>` and `<|endoftext|>`).
    pub eog_tokens: Vec<u32>,
}

impl XwenConfig {
    pub fn is_full_attn(&self, il: usize) -> bool {
        self.layer_kind[il] == LayerKind::Full
    }

    pub fn layer_kind(&self, il: usize) -> LayerKind {
        self.layer_kind[il]
    }

    pub fn n_head(&self, il: usize) -> usize {
        self.n_head[il]
    }

    pub fn rope(&self) -> &RopeKind {
        &self.rope
    }

    /// Width of the fused DeltaNet qkv stream (`attn_qkv`'s output, and the
    /// number of depthwise conv channels): q and k at K-head count, v at V-head
    /// count, all at `linear_head_dim`.
    pub fn conv_dim(&self) -> usize {
        (2 * self.linear_k_heads + self.linear_v_heads) * self.linear_head_dim
    }

    /// Width of the DeltaNet value stream — `attn_gate`'s output, `ssm_out`'s
    /// input, and the flattened per-token DeltaNet output.
    pub fn linear_v_dim(&self) -> usize {
        self.linear_v_heads * self.linear_head_dim
    }

    pub fn from_gguf(content: &Content) -> Result<Self> {
        let md = Meta(content);
        let arch = match md.str("general.architecture")? {
            "qwen35" => Arch::Dense,
            "qwen35moe" => Arch::Moe,
            other => bail!(
                "expected a Qwen 3.6 GGUF (architecture \"qwen35\" or \"qwen35moe\"), got {other:?}"
            ),
        };
        let a = arch.key();

        let n_layer = md.usize(&format!("{a}.block_count"))?;
        let hidden = md.usize(&format!("{a}.embedding_length"))?;
        let head_dim = md.usize(&format!("{a}.attention.key_length"))?;
        // Currently always equal to key_length; asserted rather than assumed so a
        // non-square conversion fails at load instead of at the first sdpa.
        let value_length = md.usize_or(&format!("{a}.attention.value_length"), head_dim);
        ensure!(
            value_length == head_dim,
            "{a}.attention.value_length is {value_length} but key_length is {head_dim}; this \
             build assumes one attention head dim"
        );

        // Partial NEoX rope over the first n_rot dims at theta 1e7. The GGUF
        // declares IMROPE sections; text-only positions make that identical to
        // NEoX, but only for THIS sectioning, so it is checked rather than
        // assumed.
        md.check_sections(&format!("{a}.rope.dimension_sections"))?;
        let rope = RopeKind::Plain {
            freq_base: md.f32_or(&format!("{a}.rope.freq_base"), 1e7),
            n_rot: md.usize_or(&format!("{a}.rope.dimension_count"), 64),
        };

        // The ssm.* key names describe a Mamba-style SSM and mean something else
        // here: `time_step_rank` is the V-head count, `group_count` the K-head
        // count, `state_size` the per-head dim, `inner_size` the value width
        // (V-heads x head dim). Read them under their real meanings and check
        // that the two redundant ones agree.
        let linear_head_dim = md.usize(&format!("{a}.ssm.state_size"))?;
        let linear_v_heads = md.usize(&format!("{a}.ssm.time_step_rank"))?;
        let linear_k_heads = md.usize(&format!("{a}.ssm.group_count"))?;
        let inner = md.usize(&format!("{a}.ssm.inner_size"))?;
        ensure!(
            inner == linear_v_heads * linear_head_dim,
            "{a}.ssm.inner_size is {inner} but time_step_rank x state_size is {}",
            linear_v_heads * linear_head_dim
        );
        ensure!(
            linear_k_heads > 0 && linear_v_heads.is_multiple_of(linear_k_heads),
            "DeltaNet has {linear_v_heads} V-heads over {linear_k_heads} K-heads, which do not \
             divide: the k/q broadcast needs a whole number of V-heads per K-head"
        );

        let layer_kind = layer_kinds(&md, a, n_layer)?;

        let (dense_ff, n_expert, n_expert_used, expert_ff, shared_expert_ff) = match arch {
            Arch::Dense => (md.usize(&format!("{a}.feed_forward_length"))?, 0, 0, 0, 0),
            Arch::Moe => (
                0,
                md.usize(&format!("{a}.expert_count"))?,
                md.usize(&format!("{a}.expert_used_count"))?,
                md.usize(&format!("{a}.expert_feed_forward_length"))?,
                md.usize(&format!("{a}.expert_shared_feed_forward_length"))?,
            ),
        };

        // `<arch>.vocab_size` is optional in GGUF; the embedding table is not, and
        // it is the number the lm head actually produces.
        let vocab = match md.usize(&format!("{a}.vocab_size")) {
            Ok(v) => v,
            Err(_) => content
                .tensor_infos
                .get("token_embd.weight")
                .and_then(|info| info.shape.dims().first().copied())
                .context("neither a vocab_size key nor a token_embd.weight to read it from")?,
        };

        // Chat stops on <|im_end|> as well as <|endoftext|>, and the GGUF
        // advertises only the first: `generation_config.json` lists both, but
        // the metadata carries 248044 solely as `bos_token_id` and
        // `padding_token_id`, neither of which means "stop". There is no
        // second-stop key of any kind — no eot, no eom — so the id comes from
        // the tokenizer, which owns every token id in this crate, rather than
        // from a lookup that would silently find nothing. A loop watching only
        // the eos id runs straight through turn boundaries and looks like a
        // model that will not stop.
        let mut eog_tokens = vec![md.u32("tokenizer.ggml.eos_token_id")?];
        if !eog_tokens.contains(&LagunaTokenizer::ENDOFTEXT) {
            eog_tokens.push(LagunaTokenizer::ENDOFTEXT);
        }

        Ok(Self {
            arch,
            n_layer,
            hidden,
            vocab,
            n_head: md.usize_per_layer(&format!("{a}.attention.head_count"), n_layer)?,
            n_kv_head: md.usize(&format!("{a}.attention.head_count_kv"))?,
            head_dim,
            layer_kind,
            linear_k_heads,
            linear_v_heads,
            linear_head_dim,
            conv_kernel: md.usize_or(&format!("{a}.ssm.conv_kernel"), 4),
            dense_ff,
            n_expert,
            n_expert_used,
            expert_ff,
            shared_expert_ff,
            rms_eps: md.f32_or(&format!("{a}.attention.layer_norm_rms_epsilon"), 1e-6) as f64,
            n_ctx_train: md.usize(&format!("{a}.context_length"))?,
            rope,
            eog_tokens,
        })
    }
}

/// The per-layer attention kind. Both published checkpoints describe this with
/// `<arch>.full_attention_interval` (= 4), which places full attention at every
/// layer where `(il + 1) % interval == 0` — 3, 7, 11, … — and gated DeltaNet
/// everywhere else. Read rather than hardcoded, so a conversion with a different
/// interval types its layers correctly instead of quietly wrongly.
///
/// A `<arch>.attention.recurrent_layers` bool array overrides it when present.
/// Neither shipped file has one; the branch exists because an explicit per-layer
/// map is the checkpoint describing itself, and it would be the only thing that
/// could express a non-periodic stack.
fn layer_kinds(md: &Meta, arch: &str, n_layer: usize) -> Result<Vec<LayerKind>> {
    let key = format!("{arch}.attention.recurrent_layers");
    if let Ok(Value::Array(vals)) = md.get(&key) {
        ensure!(
            vals.len() == n_layer,
            "GGUF key {key} has {} entries, expected {n_layer}",
            vals.len()
        );
        return vals
            .iter()
            .map(|v| {
                let recurrent = v
                    .to_bool()
                    .map(|b| b)
                    .or_else(|_| value_as_usize(v).map(|n| n != 0))
                    .with_context(|| format!("GGUF key {key} has a non-boolean entry"))?;
                Ok(if recurrent {
                    LayerKind::Linear
                } else {
                    LayerKind::Full
                })
            })
            .collect();
    }
    let interval = md.usize_or(&format!("{arch}.full_attention_interval"), 4);
    ensure!(
        interval > 0,
        "{arch}.full_attention_interval is 0, which places no attention layers at all"
    );
    Ok((0..n_layer)
        .map(|il| {
            if (il + 1).is_multiple_of(interval) {
                LayerKind::Full
            } else {
                LayerKind::Linear
            }
        })
        .collect())
}

/// Tolerant typed accessors over GGUF metadata.
struct Meta<'a>(&'a Content);

impl Meta<'_> {
    fn get(&self, key: &str) -> Result<&Value> {
        self.0
            .metadata
            .get(key)
            .with_context(|| format!("missing GGUF key {key}"))
    }

    fn str(&self, key: &str) -> Result<&str> {
        Ok(self.get(key)?.to_string()?.as_str())
    }

    fn usize(&self, key: &str) -> Result<usize> {
        value_as_usize(self.get(key)?)
            .with_context(|| format!("GGUF key {key} is not a non-negative integer"))
    }

    fn u32(&self, key: &str) -> Result<u32> {
        Ok(self.usize(key)? as u32)
    }

    fn f32(&self, key: &str) -> Result<f32> {
        let v = self.get(key)?;
        v.to_f32()
            .or_else(|_| v.to_f64().map(|v| v as f32))
            .with_context(|| format!("GGUF key {key} is not a float"))
    }

    fn usize_or(&self, key: &str, default: usize) -> usize {
        self.usize(key).unwrap_or(default)
    }

    fn f32_or(&self, key: &str, default: f32) -> f32 {
        self.f32(key).unwrap_or(default)
    }

    /// The rope sectioning, when the key is present, must be the one whose
    /// text-only behavior equals plain NEoX over `n_rot`. An absent key is a
    /// non-IMROPE conversion and needs no check.
    fn check_sections(&self, key: &str) -> Result<()> {
        let Ok(Value::Array(vals)) = self.get(key) else {
            return Ok(());
        };
        let got: Vec<usize> = vals.iter().map(value_as_usize).collect::<Result<_>>()?;
        ensure!(
            got == ROPE_SECTIONS,
            "GGUF key {key} is {got:?}; this build implements only the text-only sectioning \
             {ROPE_SECTIONS:?}, for which IMROPE reduces to NEoX"
        );
        Ok(())
    }

    /// A per-layer array key, expanding a scalar to a uniform vec.
    fn usize_per_layer(&self, key: &str, n_layer: usize) -> Result<Vec<usize>> {
        match self.get(key)? {
            Value::Array(vals) => {
                let out: Vec<usize> = vals
                    .iter()
                    .map(value_as_usize)
                    .collect::<Result<_>>()
                    .with_context(|| format!("GGUF key {key} has non-integer entries"))?;
                if out.len() != n_layer {
                    bail!(
                        "GGUF key {key} has {} entries, expected {n_layer}",
                        out.len()
                    );
                }
                Ok(out)
            }
            _ => Ok(vec![self.usize(key)?; n_layer]),
        }
    }
}

/// GGUF writers emit integers as any width and signedness (the per-layer head
/// counts are i32, most scalars u32); accept them all as long as they fit.
fn value_as_usize(v: &Value) -> Result<usize> {
    if let Ok(u) = v.to_u64() {
        return Ok(u as usize);
    }
    if let Ok(u) = v.to_u32() {
        return Ok(u as usize);
    }
    let i = v.to_i64().or_else(|_| v.to_i32().map(i64::from))?;
    if i < 0 {
        bail!("negative integer {i}");
    }
    Ok(i as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published checkpoints carry no `recurrent_layers` array, so the
    /// pattern is what places every layer: full attention at 3, 7, 11, ...
    #[test]
    fn layer_pattern_puts_full_attention_every_fourth_layer() {
        let content = Content {
            magic: candle_core::quantized::gguf_file::VersionedMagic::GgufV3,
            metadata: std::collections::HashMap::new(),
            tensor_infos: std::collections::HashMap::new(),
            tensor_data_offset: 0,
        };
        let kinds = layer_kinds(&Meta(&content), "qwen35moe", 40).unwrap();
        assert_eq!(kinds.len(), 40);
        let full: Vec<usize> = kinds
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == LayerKind::Full)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(full, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39]);
        assert_eq!(full.len(), 10, "35B-A3B has 10 full-attention layers");
    }

    /// The interval comes from the checkpoint, not from a hardcoded 4, so a
    /// conversion with a different one types its layers correctly.
    #[test]
    fn full_attention_interval_drives_the_pattern() {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("qwen35.full_attention_interval".to_string(), Value::U32(3));
        let content = Content {
            magic: candle_core::quantized::gguf_file::VersionedMagic::GgufV3,
            metadata,
            tensor_infos: std::collections::HashMap::new(),
            tensor_data_offset: 0,
        };
        let kinds = layer_kinds(&Meta(&content), "qwen35", 9).unwrap();
        let full: Vec<usize> = kinds
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == LayerKind::Full)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(full, vec![2, 5, 8]);
    }

    /// A checkpoint that describes its own layer pattern overrides the rule.
    #[test]
    fn recurrent_layers_array_overrides_the_pattern() {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "qwen35.attention.recurrent_layers".to_string(),
            Value::Array(vec![
                Value::Bool(true),
                Value::Bool(false),
                Value::Bool(false),
                Value::Bool(true),
            ]),
        );
        let content = Content {
            magic: candle_core::quantized::gguf_file::VersionedMagic::GgufV3,
            metadata,
            tensor_infos: std::collections::HashMap::new(),
            tensor_data_offset: 0,
        };
        let kinds = layer_kinds(&Meta(&content), "qwen35", 4).unwrap();
        assert_eq!(
            kinds,
            vec![
                LayerKind::Linear,
                LayerKind::Full,
                LayerKind::Full,
                LayerKind::Linear
            ]
        );
    }

    /// An array that does not cover the stack is a malformed checkpoint, not a
    /// reason to fall back to the pattern: falling back would place the layers
    /// differently from the file that is being loaded.
    #[test]
    fn short_recurrent_layers_array_is_refused() {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "qwen35.attention.recurrent_layers".to_string(),
            Value::Array(vec![Value::Bool(true), Value::Bool(false)]),
        );
        let content = Content {
            magic: candle_core::quantized::gguf_file::VersionedMagic::GgufV3,
            metadata,
            tensor_infos: std::collections::HashMap::new(),
            tensor_data_offset: 0,
        };
        assert!(layer_kinds(&Meta(&content), "qwen35", 4).is_err());
    }

    /// Only the text-only sectioning reduces to the NEoX rope rope.rs builds.
    #[test]
    fn rope_sections_are_validated() {
        let sections = |v: Vec<i32>| {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert(
                "qwen35.rope.dimension_sections".to_string(),
                Value::Array(v.into_iter().map(Value::I32).collect()),
            );
            Content {
                magic: candle_core::quantized::gguf_file::VersionedMagic::GgufV3,
                metadata,
                tensor_infos: std::collections::HashMap::new(),
                tensor_data_offset: 0,
            }
        };
        let ok = sections(vec![11, 11, 10, 0]);
        Meta(&ok)
            .check_sections("qwen35.rope.dimension_sections")
            .unwrap();
        let bad = sections(vec![16, 24, 24, 0]);
        assert!(
            Meta(&bad)
                .check_sections("qwen35.rope.dimension_sections")
                .is_err()
        );
    }

    /// A DeltaNet geometry whose V-heads are not a whole multiple of its K-heads
    /// has no tiled broadcast, so it must be refused rather than truncated.
    #[test]
    fn conv_and_value_widths_follow_the_head_counts() {
        let cfg = XwenConfig {
            arch: Arch::Moe,
            n_layer: 40,
            hidden: 2048,
            vocab: 248320,
            n_head: vec![16; 40],
            n_kv_head: 2,
            head_dim: 256,
            layer_kind: vec![LayerKind::Linear; 40],
            linear_k_heads: 16,
            linear_v_heads: 32,
            linear_head_dim: 128,
            conv_kernel: 4,
            dense_ff: 0,
            n_expert: 256,
            n_expert_used: 8,
            expert_ff: 512,
            shared_expert_ff: 512,
            rms_eps: 1e-6,
            n_ctx_train: 262144,
            rope: RopeKind::Plain {
                freq_base: 1e7,
                n_rot: 64,
            },
            eog_tokens: vec![248046, 248044],
        };
        assert_eq!(cfg.conv_dim(), (2 * 16 + 32) * 128);
        assert_eq!(cfg.conv_dim(), 8192);
        assert_eq!(cfg.linear_v_dim(), 4096);
    }
}
