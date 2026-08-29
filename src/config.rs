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

/// Which graph a checkpoint holds. `qwen35` and `qwen35moe` differ only in the
/// FFN (dense SwiGLU vs MoE on every layer) and share attention, DeltaNet and
/// norms. `qwen4exp` composes the same attention/DeltaNet/MoE blocks under a
/// hyper-connection residual carrier, with sparse (QSA) full-attention layers
/// and an optional per-layer n-gram embedding table (PLE) —
/// see docs/qwen4exp-port.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// Dense FFN — Qwen3.6-27B and Qwen3.8-27B both.
    Dense,
    /// Qwen3.6-35B-A3B — 256 routed experts + one shared expert.
    Moe,
    /// Qwen3.8-Flash-Next — 512 routed experts + one shared expert on every
    /// layer, hyper-connection residuals, QSA attention, PLE.
    Qwen4Exp,
}

/// Activation applied to the DeltaNet z-gate (`attn_gate`'s output) before the
/// gated RMSNorm's final multiply. Resolved at construction — the block never
/// branches per token — because swapping it yields a graph that runs and
/// generates garbage rather than one that fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZGate {
    Silu,
    Sigmoid,
}

/// The floor the renormalized top-k router weight sum is clamped to (2^-14),
/// llama.cpp's clamp and the parity ground truth for the qwen35moe checkpoint.
/// moe.rs's candle-chain fallback applies the same value.
pub const MOE_SUM_FLOOR: f32 = 6.103515625e-5;

impl Arch {
    /// The GGUF `general.architecture` string this variant is parsed from, which
    /// is also the prefix every other metadata key carries.
    pub fn key(&self) -> &'static str {
        match self {
            Arch::Dense => "qwen35",
            Arch::Moe => "qwen35moe",
            Arch::Qwen4Exp => "qwen4exp",
        }
    }

    /// The checkpoint to assume for this architecture when nothing else
    /// identifies the file. Unambiguous for `qwen35moe`, which only one official
    /// checkpoint ships; a coin-flip for `qwen35`, which the Qwen3.6-27B and
    /// Qwen3.8-27B releases share graph-for-graph; and unambiguous again for
    /// `qwen4exp`, whose one registry checkpoint is Qwen3.8-Flash-Next. Ask
    /// [`XwenConfig::checkpoint`] first — it reads what the file says about
    /// itself — and reach for this only when that comes back `None`, which is a
    /// conversion that names no checkpoint, and say so when you do.
    ///
    /// Still an `Option`: the answer is a fact about the registry, and an
    /// architecture with no checkpoint in it is a state this has been in and
    /// will be again (`qwen4exp` was exactly that until the Unsloth file was
    /// entered).
    pub fn model(&self) -> Option<crate::hub::Model> {
        match self {
            Arch::Dense => Some(crate::hub::Model::Qwen27B),
            Arch::Moe => Some(crate::hub::Model::Qwen35BA3B),
            Arch::Qwen4Exp => Some(crate::hub::Model::Qwen38FlashNext),
        }
    }

    /// The DeltaNet z-gate activation: `silu(z)` on the Qwen 3.6/3.8 graphs,
    /// `sigmoid(z)` on qwen4exp (HF `output_gate_type: "sigmoid"`).
    pub fn z_gate(&self) -> ZGate {
        match self {
            Arch::Dense | Arch::Moe => ZGate::Silu,
            Arch::Qwen4Exp => ZGate::Sigmoid,
        }
    }

    /// The clamp floor for the renormalized routed-expert weight sum. The
    /// qwen4exp HF math has no clamp (llama.cpp applies [`MOE_SUM_FLOOR`]
    /// unconditionally there too — a recorded divergence, practically a no-op
    /// since top-10 softmax sums are far larger).
    pub fn moe_sum_floor(&self) -> f32 {
        match self {
            Arch::Dense | Arch::Moe => MOE_SUM_FLOOR,
            Arch::Qwen4Exp => 0.0,
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
    /// `general.name` — what the file calls the model it holds
    /// ("Qwen3.6-27B", "Qwen3.8-27B", …). Optional in GGUF, and the only thing
    /// in a dense file that says which release it is; see
    /// [`XwenConfig::checkpoint`].
    pub general_name: Option<String>,
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
    /// The `qwen4exp` subsystems (hyper-connections, QSA indexer, PLE);
    /// `None` on the qwen35/qwen35moe arms.
    pub qwen4exp: Option<Qwen4ExpConfig>,
}

/// Config the `qwen4exp` architecture carries beyond the shared geometry.
#[derive(Debug, Clone)]
pub struct Qwen4ExpConfig {
    /// Hyper-connection stream count: the residual carrier is
    /// `hc_count × hidden` wide, seeded by repeating the token embedding.
    pub hc_count: usize,
    /// Bottleneck width of the hyper-connection gate MLP
    /// (down: `hc_count × hidden` → this, up: this → `hc_count × hidden`).
    pub hc_low_rank: usize,
    /// QSA indexer query-head count (its K side is MQA: one head).
    pub indexer_heads: usize,
    /// Indexer head dim — the width its q/k projections, norms and partial
    /// rope all run at.
    pub indexer_head_dim: usize,
    /// Sparse-attention token budget: each query attends to the
    /// top-(`top_k / compress_ratio`) scored key blocks.
    pub indexer_top_k: usize,
    /// Tokens per indexer key block, uniform across the full-attention layers
    /// (reduced from the per-layer `attention.compress_ratios` array).
    pub indexer_compress_ratio: usize,
    /// The per-layer n-gram embedding table; `None` when the checkpoint ships
    /// none.
    pub ple: Option<PleConfig>,
}

/// The PLE n-gram table: hash constants and injection geometry. Everything
/// here is read from the file, never recomputed — the multipliers exceed f32
/// precision and a recomputation that rounds them corrupts every lookup.
#[derive(Debug, Clone)]
pub struct PleConfig {
    /// Decoder layer indices carrying a PLE injection. 0-based in the GGUF —
    /// the converter shifts config.json's one-indexed `ple_layer_ids`.
    pub layers: Vec<usize>,
    /// Maximum n-gram order (3: bigram and trigram heads, no unigrams).
    pub ngram_size: usize,
    /// Hash heads per n-gram order; total heads = `(ngram_size - 1) *
    /// heads_per_ngram`.
    pub heads_per_ngram: usize,
    /// Depthwise conv kernel width of the PLE output conv (run at dilation 3).
    pub conv_kernel: usize,
    /// The boundary id n-gram windows never cross: `<|endoftext|>` 248044,
    /// NOT the chat stop 248046 — the wrong id silently corrupts lookups at
    /// every turn boundary.
    pub eos_token_id: u32,
    /// Vision placeholder id; absent on a text-only conversion.
    pub image_token_id: Option<u32>,
    /// Table row width; one row per head, concatenated into the injection
    /// input.
    pub row_dim: usize,
    /// Per-position hash multipliers (`mixed = t₀·m₀ ⊕ t₁·m₁ ⊕ …`), ~45-bit
    /// values shipped as an I64/U64 buffer.
    pub layer_multipliers: Vec<u64>,
    /// Per-head prime table sizes; row = `mixed % vocab + offset`.
    pub head_vocab_sizes: Vec<u64>,
    /// Per-head base offsets into the flat table, paired 1:1 with
    /// `head_vocab_sizes`.
    pub head_offsets: Vec<u64>,
}

impl XwenConfig {
    /// Which official checkpoint this file is, or `None` when nothing in it
    /// says. `path` is where the file was read from, consulted only when
    /// `general.name` does not answer.
    ///
    /// Not the same question as [`Arch::model`]: two releases ship the dense
    /// architecture, so the graph alone no longer names a checkpoint. A caller
    /// that must have an answer falls back to `arch.model()` and says so.
    pub fn checkpoint(&self, path: &std::path::Path) -> Option<crate::hub::Model> {
        crate::hub::Model::identify(self.arch, self.general_name.as_deref(), Some(path))
    }

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
            "qwen4exp" => Arch::Qwen4Exp,
            other => bail!(
                "expected a Qwen GGUF (architecture \"qwen35\", \"qwen35moe\" or \"qwen4exp\"), \
                 got {other:?}"
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
            Arch::Moe | Arch::Qwen4Exp => (
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

        // Chat stops on <|im_end|> as well as <|endoftext|>, and a GGUF
        // advertises only one of them: `generation_config.json` lists both, but
        // the metadata carries a single eos id (the 3.6/3.8 files say 248046;
        // 248044 appears solely as `bos_token_id` and `padding_token_id`,
        // neither of which means "stop"). There is no second-stop key of any
        // kind — no eot, no eom — so the missing ids come from the tokenizer,
        // which owns every token id in this crate, rather than from a lookup
        // that would silently find nothing. Both known stops are DELIBERATELY
        // guaranteed regardless of which one the file names — even a file
        // advertising some other eos entirely keeps both, because the two ids
        // are properties of the Qwen tokenizer family, not of a checkpoint: a
        // loop watching only the advertised eos runs straight through turn
        // boundaries and looks like a model that will not stop.
        let mut eog_tokens = vec![md.u32("tokenizer.ggml.eos_token_id")?];
        for id in LagunaTokenizer::EOG {
            if !eog_tokens.contains(&id) {
                eog_tokens.push(id);
            }
        }

        let qwen4exp = match arch {
            Arch::Qwen4Exp => Some(Qwen4ExpConfig::from_gguf(&md, a, &layer_kind)?),
            Arch::Dense | Arch::Moe => None,
        };

        Ok(Self {
            arch,
            general_name: md.str("general.name").ok().map(str::to_string),
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
            qwen4exp,
        })
    }
}

impl Qwen4ExpConfig {
    fn from_gguf(md: &Meta, a: &str, layer_kind: &[LayerKind]) -> Result<Self> {
        let hc_count = md.usize(&format!("{a}.hyper_connection.count"))?;
        ensure!(
            hc_count > 0,
            "{a}.hyper_connection.count is 0; the residual carrier needs at least one stream"
        );
        Ok(Self {
            hc_count,
            hc_low_rank: md.usize(&format!("{a}.hyper_connection.low_rank"))?,
            indexer_heads: md.usize(&format!("{a}.attention.indexer.head_count"))?,
            indexer_head_dim: md.usize(&format!("{a}.attention.indexer.key_length"))?,
            indexer_top_k: md.usize(&format!("{a}.attention.indexer.top_k"))?,
            indexer_compress_ratio: indexer_compress_ratio(md, a, layer_kind)?,
            ple: PleConfig::from_gguf(md, a, layer_kind.len())?,
        })
    }
}

impl PleConfig {
    /// `None` when `<arch>.ple.layers` is absent or empty — a checkpoint
    /// without the table. A present list makes every other PLE key required.
    fn from_gguf(md: &Meta, a: &str, n_layer: usize) -> Result<Option<Self>> {
        let layers_key = format!("{a}.ple.layers");
        if md.get(&layers_key).is_err() {
            return Ok(None);
        }
        let layers = md.usize_array(&layers_key)?;
        if layers.is_empty() {
            return Ok(None);
        }
        for &il in &layers {
            ensure!(
                il < n_layer,
                "GGUF key {layers_key} names layer {il}, but the model has {n_layer} layers \
                 (the GGUF list is 0-based; a value at or past the layer count suggests a \
                 conversion that kept config.json's one-indexed ids)"
            );
        }
        let ngram_size = md.usize(&format!("{a}.ple.ngram_size"))?;
        ensure!(
            ngram_size >= 2,
            "{a}.ple.ngram_size is {ngram_size}; the table holds heads for n-gram orders \
             2..=ngram_size, so anything below 2 describes no heads at all"
        );
        let heads_per_ngram = md.usize(&format!("{a}.ple.heads_per_ngram"))?;
        // The hash addresses head `(order - 2) * heads_per_ngram + g`, so the
        // flat per-head tables must cover exactly this many heads.
        let n_heads = (ngram_size - 1) * heads_per_ngram;
        let head_vocab_sizes = md.u64_array(&format!("{a}.ple.head_vocab_sizes"))?;
        ensure!(
            head_vocab_sizes.len() == n_heads,
            "{a}.ple.head_vocab_sizes has {} entries, expected {n_heads} ((ngram_size \
             {ngram_size} - 1) x heads_per_ngram {heads_per_ngram}): one table slice per hash \
             head of each order 2..=ngram_size",
            head_vocab_sizes.len(),
        );
        let head_offsets = md.u64_array(&format!("{a}.ple.head_offsets"))?;
        ensure!(
            head_offsets.len() == n_heads,
            "{a}.ple.head_offsets has {} entries, expected {n_heads}, pairing one flat-table \
             offset with each of {a}.ple.head_vocab_sizes' entries",
            head_offsets.len(),
        );
        // Rows are addressed `hash % vocab + offset` into one flat table: every
        // head needs a nonzero modulus, and each head's slice must start where
        // the previous one ends or two heads would read each other's rows.
        let mut running_sum = 0u64;
        for (h, (&size, &offset)) in head_vocab_sizes.iter().zip(&head_offsets).enumerate() {
            ensure!(
                size > 0,
                "{a}.ple.head_vocab_sizes is 0 at head {h}; the row lookup is `hash % vocab`, \
                 so a zero-sized head addresses nothing"
            );
            ensure!(
                offset == running_sum,
                "{a}.ple.head_offsets is {offset} at head {h}, expected {running_sum} (the \
                 running sum of head_vocab_sizes): the heads tile one flat table"
            );
            running_sum = running_sum.checked_add(size).with_context(|| {
                format!("{a}.ple.head_vocab_sizes overflows u64 summed through head {h}")
            })?;
        }
        let layer_multipliers = md.u64_array(&format!("{a}.ple.layer_multipliers"))?;
        ensure!(
            layer_multipliers.len() == ngram_size,
            "{a}.ple.layer_multipliers has {} entries, expected ngram_size ({ngram_size}): the \
             hash mixes one multiplier per token position of the longest n-gram \
             (`mixed = t0*m0 ^ t1*m1 ^ ...` in the reference)",
            layer_multipliers.len()
        );
        // Absent means a text-only conversion; a present-but-malformed value is
        // a broken file, not an absent key.
        let image_key = format!("{a}.ple.image_token_id");
        let image_token_id = match md.get(&image_key) {
            Err(_) => None,
            Ok(_) => Some(md.u32_checked(&image_key)?),
        };
        Ok(Some(Self {
            layers,
            ngram_size,
            heads_per_ngram,
            conv_kernel: md.usize_or(&format!("{a}.ple.conv_kernel"), 4),
            eos_token_id: md.u32_checked(&format!("{a}.ple.eos_token_id"))?,
            image_token_id,
            row_dim: md.usize(&format!("{a}.embedding_length_per_layer_input"))?,
            layer_multipliers,
            head_vocab_sizes,
            head_offsets,
        }))
    }
}

/// The QSA block width, one uniform value reduced from the per-layer
/// `<arch>.attention.compress_ratios` array the converter writes: the ratio on
/// every full-attention layer, zero on every DeltaNet layer. Deliberately
/// validated against the FILE's declared cadence (`layer_kinds`, from
/// `full_attention_interval` or a `recurrent_layers` array), never a hardcoded
/// interval-4, so a file whose sparse layers sit elsewhere — or that mixes
/// ratios — is refused rather than masked wrongly.
fn indexer_compress_ratio(md: &Meta, arch: &str, layer_kind: &[LayerKind]) -> Result<usize> {
    let key = format!("{arch}.attention.compress_ratios");
    let ratios = md.usize_array(&key)?;
    ensure!(
        ratios.len() == layer_kind.len(),
        "GGUF key {key} has {} entries, expected {}",
        ratios.len(),
        layer_kind.len()
    );
    let mut ratio = None;
    for (il, (&r, kind)) in ratios.iter().zip(layer_kind).enumerate() {
        match kind {
            LayerKind::Full => {
                ensure!(
                    r > 0,
                    "GGUF key {key} is 0 at layer {il}, a full-attention layer that needs a \
                     compression ratio"
                );
                match ratio {
                    None => ratio = Some(r),
                    Some(prev) => ensure!(
                        prev == r,
                        "GGUF key {key} mixes ratios {prev} and {r}; this build assumes one \
                         indexer geometry across the attention layers"
                    ),
                }
            }
            LayerKind::Linear => ensure!(
                r == 0,
                "GGUF key {key} is {r} at layer {il}, a DeltaNet layer that runs no indexer"
            ),
        }
    }
    ratio.with_context(|| format!("GGUF key {key} covers no full-attention layer"))
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

    /// Like [`Self::u32`], but refusing a value that does not fit instead of
    /// truncating it — for token ids, where a wrapped value is a different,
    /// valid-looking id rather than an obvious error.
    fn u32_checked(&self, key: &str) -> Result<u32> {
        let v = value_as_u64(self.get(key)?)
            .with_context(|| format!("GGUF key {key} is not a non-negative integer"))?;
        u32::try_from(v).with_context(|| format!("GGUF key {key} is {v}, which exceeds u32"))
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

    /// An array of non-negative integers, whatever width and signedness each
    /// entry was written with.
    fn usize_array(&self, key: &str) -> Result<Vec<usize>> {
        match self.get(key)? {
            Value::Array(vals) => vals
                .iter()
                .map(value_as_usize)
                .collect::<Result<_>>()
                .with_context(|| format!("GGUF key {key} has non-integer entries")),
            _ => bail!("GGUF key {key} is not an array"),
        }
    }

    /// An array of non-negative integers read at full u64 width — for values
    /// (the PLE hash multipliers) too wide for f32 round-trips and written as
    /// I64 or U64 buffers interchangeably. Negative entries are refused.
    fn u64_array(&self, key: &str) -> Result<Vec<u64>> {
        match self.get(key)? {
            Value::Array(vals) => vals
                .iter()
                .map(value_as_u64)
                .collect::<Result<_>>()
                .with_context(|| format!("GGUF key {key} has non-integer entries")),
            _ => bail!("GGUF key {key} is not an array"),
        }
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
    Ok(value_as_u64(v)? as usize)
}

/// Like [`value_as_usize`], preserving the full u64 range. A Bool is refused
/// up front: candle's `to_u64` upcasts it to 0/1, which would read a boolean
/// key silently as an integer.
fn value_as_u64(v: &Value) -> Result<u64> {
    if matches!(v, Value::Bool(_)) {
        bail!("boolean where an integer is expected");
    }
    // `to_u64` also upcasts U8/U16/U32; the signed widths each need their own
    // accessor.
    if let Ok(u) = v.to_u64() {
        return Ok(u);
    }
    let i = v
        .to_i64()
        .or_else(|_| v.to_i32().map(i64::from))
        .or_else(|_| v.to_i16().map(i64::from))
        .or_else(|_| v.to_i8().map(i64::from))?;
    if i < 0 {
        bail!("negative integer {i}");
    }
    Ok(i as u64)
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
            general_name: None,
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
            qwen4exp: None,
        };
        assert_eq!(cfg.conv_dim(), (2 * 16 + 32) * 128);
        assert_eq!(cfg.conv_dim(), 8192);
        assert_eq!(cfg.linear_v_dim(), 4096);
    }

    fn content(metadata: std::collections::HashMap<String, Value>) -> Content {
        Content {
            magic: candle_core::quantized::gguf_file::VersionedMagic::GgufV3,
            metadata,
            tensor_infos: std::collections::HashMap::new(),
            tensor_data_offset: 0,
        }
    }

    /// Metadata mirroring the real Qwen3.8-Flash-Next GGUF (values read from
    /// the file 2026-08-26), with the integer typing its converter uses:
    /// I32 for the per-layer ratio array, I64 for the PLE layer list, U64 for
    /// the hash-constant buffers.
    fn qwen4exp_metadata() -> std::collections::HashMap<String, Value> {
        let mut m = std::collections::HashMap::new();
        let compress: Vec<Value> = (0..48)
            .map(|il| Value::I32(if (il + 1) % 4 == 0 { 4 } else { 0 }))
            .collect();
        let scalars: &[(&str, u32)] = &[
            ("qwen4exp.block_count", 48),
            ("qwen4exp.embedding_length", 2560),
            ("qwen4exp.attention.head_count", 24),
            ("qwen4exp.attention.head_count_kv", 2),
            ("qwen4exp.attention.key_length", 256),
            ("qwen4exp.rope.dimension_count", 64),
            ("qwen4exp.context_length", 262144),
            ("qwen4exp.vocab_size", 248320),
            ("qwen4exp.full_attention_interval", 4),
            ("qwen4exp.ssm.state_size", 128),
            ("qwen4exp.ssm.time_step_rank", 48),
            ("qwen4exp.ssm.group_count", 16),
            ("qwen4exp.ssm.inner_size", 6144),
            ("qwen4exp.ssm.conv_kernel", 4),
            ("qwen4exp.expert_count", 512),
            ("qwen4exp.expert_used_count", 10),
            ("qwen4exp.expert_feed_forward_length", 640),
            ("qwen4exp.expert_shared_feed_forward_length", 640),
            ("tokenizer.ggml.eos_token_id", 248044),
            ("qwen4exp.hyper_connection.count", 4),
            ("qwen4exp.hyper_connection.low_rank", 320),
            ("qwen4exp.attention.indexer.head_count", 4),
            ("qwen4exp.attention.indexer.key_length", 128),
            ("qwen4exp.attention.indexer.top_k", 2048),
            ("qwen4exp.embedding_length_per_layer_input", 160),
            ("qwen4exp.ple.ngram_size", 3),
            ("qwen4exp.ple.heads_per_ngram", 8),
            ("qwen4exp.ple.conv_kernel", 4),
            ("qwen4exp.ple.eos_token_id", 248044),
        ];
        for &(k, v) in scalars {
            m.insert(k.to_string(), Value::U32(v));
        }
        m.insert(
            "general.architecture".to_string(),
            Value::String("qwen4exp".to_string()),
        );
        m.insert(
            "general.name".to_string(),
            Value::String("Qwen3.8 Flash Next".to_string()),
        );
        m.insert(
            "qwen4exp.attention.layer_norm_rms_epsilon".to_string(),
            Value::F32(1e-6),
        );
        m.insert("qwen4exp.rope.freq_base".to_string(), Value::F32(1e7));
        m.insert(
            "qwen4exp.rope.dimension_sections".to_string(),
            Value::Array(vec![11, 11, 10, 0].into_iter().map(Value::I32).collect()),
        );
        m.insert(
            "qwen4exp.attention.compress_ratios".to_string(),
            Value::Array(compress),
        );
        m.insert(
            "qwen4exp.ple.layers".to_string(),
            Value::Array(vec![Value::I64(1)]),
        );
        // Hash multipliers are ~45-bit values; anything that survives only as
        // f32 would round them, so the fixture keeps them above u32 range.
        m.insert(
            "qwen4exp.ple.layer_multipliers".to_string(),
            Value::Array(
                vec![25_214_903_917_u64, 22_695_477_037, 30_268_512_953]
                    .into_iter()
                    .map(Value::U64)
                    .collect(),
            ),
        );
        // Offsets are the running sum of the vocab sizes — the flat-table
        // tiling the real file ships and the parser requires.
        let sizes: Vec<u64> = (0..16).map(|i| 20_000_003 + 6 * i).collect();
        let offsets: Vec<u64> = sizes
            .iter()
            .scan(0u64, |acc, &v| {
                let offset = *acc;
                *acc += v;
                Some(offset)
            })
            .collect();
        m.insert(
            "qwen4exp.ple.head_vocab_sizes".to_string(),
            Value::Array(sizes.into_iter().map(Value::U64).collect()),
        );
        m.insert(
            "qwen4exp.ple.head_offsets".to_string(),
            Value::Array(offsets.into_iter().map(Value::U64).collect()),
        );
        m
    }

    /// The full qwen4exp parse against the real file's values: shared geometry
    /// through the existing paths, plus the hyper-connection, indexer and PLE
    /// groups in the arch-specific sub-config.
    #[test]
    fn qwen4exp_parses_the_real_files_values() {
        let cfg = XwenConfig::from_gguf(&content(qwen4exp_metadata())).unwrap();
        assert_eq!(cfg.arch, Arch::Qwen4Exp);
        assert_eq!(cfg.general_name.as_deref(), Some("Qwen3.8 Flash Next"));
        assert_eq!(cfg.n_layer, 48);
        assert_eq!(cfg.hidden, 2560);
        assert_eq!(cfg.n_head(0), 24);
        assert_eq!(cfg.n_kv_head, 2);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.dense_ff, 0);
        assert_eq!(cfg.n_expert, 512);
        assert_eq!(cfg.n_expert_used, 10);
        assert_eq!(cfg.expert_ff, 640);
        assert_eq!(cfg.shared_expert_ff, 640);
        assert_eq!(cfg.linear_k_heads, 16);
        assert_eq!(cfg.linear_v_heads, 48);
        assert_eq!(cfg.linear_head_dim, 128);
        assert_eq!(cfg.n_ctx_train, 262144);
        assert!(cfg.is_full_attn(3) && cfg.is_full_attn(47));
        assert!(!cfg.is_full_attn(0) && !cfg.is_full_attn(46));
        // Both chat stops, whichever one the file advertised as eos.
        assert!(cfg.eog_tokens.contains(&248046) && cfg.eog_tokens.contains(&248044));

        let q4 = cfg.qwen4exp.as_ref().unwrap();
        assert_eq!(q4.hc_count, 4);
        assert_eq!(q4.hc_low_rank, 320);
        assert_eq!(q4.indexer_heads, 4);
        assert_eq!(q4.indexer_head_dim, 128);
        assert_eq!(q4.indexer_top_k, 2048);
        assert_eq!(q4.indexer_compress_ratio, 4);

        let ple = q4.ple.as_ref().unwrap();
        assert_eq!(ple.layers, vec![1]);
        assert_eq!(ple.ngram_size, 3);
        assert_eq!(ple.heads_per_ngram, 8);
        assert_eq!(ple.conv_kernel, 4);
        assert_eq!(ple.eos_token_id, 248044);
        assert_eq!(ple.image_token_id, None);
        assert_eq!(ple.row_dim, 160);
        // Read back at full width: these exceed u32 and must not round.
        assert_eq!(
            ple.layer_multipliers,
            vec![25_214_903_917, 22_695_477_037, 30_268_512_953]
        );
        assert_eq!(ple.head_vocab_sizes.len(), 16);
        assert_eq!(ple.head_offsets.len(), 16);
        assert_eq!(ple.head_vocab_sizes[0], 20_000_003);
        assert_eq!(ple.head_offsets[0], 0);
        assert_eq!(ple.head_offsets[1], 20_000_003);
    }

    /// The per-layer ratio array must agree with the layer cadence: nonzero
    /// exactly on the full-attention layers, one uniform value across them.
    #[test]
    fn qwen4exp_compress_ratios_must_match_the_cadence() {
        let key = "qwen4exp.attention.compress_ratios";
        let with_ratios = |f: &dyn Fn(usize) -> i32| {
            let mut m = qwen4exp_metadata();
            m.insert(
                key.to_string(),
                Value::Array((0..48).map(|il| Value::I32(f(il))).collect()),
            );
            content(m)
        };
        // A ratio on a DeltaNet layer.
        let bad = with_ratios(&|il| if (il + 1) % 4 == 0 || il == 0 { 4 } else { 0 });
        assert!(XwenConfig::from_gguf(&bad).is_err());
        // A full-attention layer without one.
        let bad = with_ratios(&|il| if (il + 1) % 4 == 0 && il != 3 { 4 } else { 0 });
        assert!(XwenConfig::from_gguf(&bad).is_err());
        // Mixed nonzero values have no single indexer geometry.
        let bad = with_ratios(&|il| match il {
            il if (il + 1) % 4 != 0 => 0,
            7 => 8,
            _ => 4,
        });
        assert!(XwenConfig::from_gguf(&bad).is_err());
        // An array that does not cover the stack.
        let mut m = qwen4exp_metadata();
        m.insert(key.to_string(), Value::Array(vec![Value::I32(4)]));
        assert!(XwenConfig::from_gguf(&content(m)).is_err());
    }

    /// Hyper-connections are structural on this arch — a file without the keys
    /// cannot be sized, so it is refused rather than defaulted.
    #[test]
    fn qwen4exp_requires_the_hyper_connection_keys() {
        for key in [
            "qwen4exp.hyper_connection.count",
            "qwen4exp.hyper_connection.low_rank",
        ] {
            let mut m = qwen4exp_metadata();
            m.remove(key);
            assert!(
                XwenConfig::from_gguf(&content(m)).is_err(),
                "parse succeeded without {key}"
            );
        }
    }

    /// A checkpoint without `ple.layers` — or with an empty list — ships no
    /// n-gram table; the rest of the config still parses.
    #[test]
    fn absent_or_empty_ple_layers_means_no_ple() {
        let mut m = qwen4exp_metadata();
        m.remove("qwen4exp.ple.layers");
        let cfg = XwenConfig::from_gguf(&content(m)).unwrap();
        assert!(cfg.qwen4exp.as_ref().unwrap().ple.is_none());

        let mut m = qwen4exp_metadata();
        m.insert("qwen4exp.ple.layers".to_string(), Value::Array(vec![]));
        let cfg = XwenConfig::from_gguf(&content(m)).unwrap();
        assert!(cfg.qwen4exp.as_ref().unwrap().ple.is_none());
    }

    /// A present PLE group must describe a coherent flat table: head counts
    /// tied to the n-gram geometry, one hash multiplier per token position,
    /// a nonzero modulus per head, and offsets tiling the table.
    #[test]
    fn ple_geometry_is_validated() {
        let with = |key: &str, v: Value| {
            let mut m = qwen4exp_metadata();
            m.insert(key.to_string(), v);
            content(m)
        };
        let u64s = |vals: Vec<u64>| Value::Array(vals.into_iter().map(Value::U64).collect());

        // 15 vocab sizes for (ngram_size 3 - 1) x heads_per_ngram 8 = 16 heads.
        let bad = with(
            "qwen4exp.ple.head_vocab_sizes",
            u64s((0..15).map(|i| 20_000_003 + 6 * i).collect()),
        );
        assert!(XwenConfig::from_gguf(&bad).is_err());
        // Offsets not covering every head.
        let bad = with("qwen4exp.ple.head_offsets", u64s(vec![0]));
        assert!(XwenConfig::from_gguf(&bad).is_err());
        // Two multipliers cannot hash the trigram head's third position.
        let bad = with(
            "qwen4exp.ple.layer_multipliers",
            u64s(vec![25_214_903_917, 22_695_477_037]),
        );
        assert!(XwenConfig::from_gguf(&bad).is_err());
        // A zero vocab size is a future modulo-by-zero. Zeroing the LAST head
        // keeps the fixture's offsets a valid running sum, so this trips the
        // zero check specifically.
        let bad = with(
            "qwen4exp.ple.head_vocab_sizes",
            u64s(
                (0..16)
                    .map(|i| if i == 15 { 0 } else { 20_000_003 + 6 * i })
                    .collect(),
            ),
        );
        assert!(XwenConfig::from_gguf(&bad).is_err());
        // Offsets that are not the running sum of the vocab sizes (the
        // fixture's former uniform-stride values) make heads overlap.
        let bad = with(
            "qwen4exp.ple.head_offsets",
            u64s((0..16).map(|i| i * 20_000_096).collect()),
        );
        assert!(XwenConfig::from_gguf(&bad).is_err());
        // An order below 2 describes no n-gram heads at all.
        let bad = with("qwen4exp.ple.ngram_size", Value::U32(1));
        assert!(XwenConfig::from_gguf(&bad).is_err());
    }

    /// The PLE token ids are exact u32s: a value past u32::MAX is refused, not
    /// truncated to a different valid-looking id. And an absent image id (a
    /// text-only conversion) is `None`, while a present-but-malformed one is
    /// an error — never collapsed into absence.
    #[test]
    fn ple_token_ids_are_checked_not_truncated() {
        let with = |key: &str, v: Value| {
            let mut m = qwen4exp_metadata();
            m.insert(key.to_string(), v);
            content(m)
        };
        let bad = with("qwen4exp.ple.eos_token_id", Value::U64(1 << 33));
        assert!(XwenConfig::from_gguf(&bad).is_err());

        let bad = with("qwen4exp.ple.image_token_id", Value::U64(u64::MAX));
        assert!(XwenConfig::from_gguf(&bad).is_err());
        let bad = with(
            "qwen4exp.ple.image_token_id",
            Value::String("nope".to_string()),
        );
        assert!(XwenConfig::from_gguf(&bad).is_err());
        let ok = with("qwen4exp.ple.image_token_id", Value::U32(248047));
        let cfg = XwenConfig::from_gguf(&ok).unwrap();
        let ple = cfg.qwen4exp.unwrap().ple.unwrap();
        assert_eq!(ple.image_token_id, Some(248047));
        // The absent-key => None case is pinned by
        // `qwen4exp_parses_the_real_files_values`.
    }

    /// GGUF writers emit integer arrays at any width and signedness; the u64
    /// reader accepts every non-negative integer entry — the narrow signed
    /// widths included — and refuses negatives, booleans (candle's `to_u64`
    /// would upcast them to 0/1) and non-arrays.
    #[test]
    fn u64_arrays_accept_every_integer_width_and_refuse_non_integers() {
        let with = |v: Value| {
            let mut m = std::collections::HashMap::new();
            m.insert("k".to_string(), v);
            content(m)
        };
        let big = 30_268_512_953_u64;
        let ok = with(Value::Array(vec![Value::I64(big as i64), Value::U32(7)]));
        assert_eq!(Meta(&ok).u64_array("k").unwrap(), vec![big, 7]);
        let ok = with(Value::Array(vec![Value::U64(u64::MAX)]));
        assert_eq!(Meta(&ok).u64_array("k").unwrap(), vec![u64::MAX]);
        let ok = with(Value::Array(vec![
            Value::I8(7),
            Value::I16(300),
            Value::U8(9),
            Value::U16(70),
        ]));
        assert_eq!(Meta(&ok).u64_array("k").unwrap(), vec![7, 300, 9, 70]);
        let negative = with(Value::Array(vec![Value::I64(-1)]));
        assert!(Meta(&negative).u64_array("k").is_err());
        let negative = with(Value::Array(vec![Value::I8(-1)]));
        assert!(Meta(&negative).u64_array("k").is_err());
        let boolean = with(Value::Array(vec![Value::Bool(true)]));
        assert!(Meta(&boolean).u64_array("k").is_err());
        let scalar = with(Value::U64(3));
        assert!(Meta(&scalar).u64_array("k").is_err());
        let missing = with(Value::U64(3));
        assert!(Meta(&missing).u64_array("absent").is_err());
    }

    /// The per-arch construction-time knobs: z-gate activation and the MoE
    /// renorm floor. Divergence lives here, never in a per-token branch.
    #[test]
    fn per_arch_gates_and_floors() {
        assert_eq!(Arch::Dense.z_gate(), ZGate::Silu);
        assert_eq!(Arch::Moe.z_gate(), ZGate::Silu);
        assert_eq!(Arch::Qwen4Exp.z_gate(), ZGate::Sigmoid);
        assert_eq!(Arch::Dense.moe_sum_floor(), MOE_SUM_FLOOR);
        assert_eq!(Arch::Moe.moe_sum_floor(), 6.103515625e-5);
        assert_eq!(Arch::Qwen4Exp.moe_sum_floor(), 0.0);
        assert_eq!(Arch::Qwen4Exp.key(), "qwen4exp");
        assert_eq!(
            Arch::Qwen4Exp.model(),
            Some(crate::hub::Model::Qwen38FlashNext)
        );
        assert!(Arch::Dense.model().is_some() && Arch::Moe.model().is_some());
    }
}
