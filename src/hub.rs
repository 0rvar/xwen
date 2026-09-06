//! Resolution of the official checkpoints via the standard Hugging Face hub
//! cache (`~/.cache/huggingface/hub` layout: content-addressed `blobs/`,
//! `snapshots/<commit>/<file>` symlinks into them, `refs/main` naming the
//! current commit).
//!
//! Two tiers, deliberately separate: the `cached_*`/`official_*` lookups are
//! offline (never touch the network — safe anywhere, including tests), and
//! the `ensure_*` calls are idempotent downloads via hf-hub's cache-first
//! `get`: the cached path comes back without a request, only a missing file
//! is downloaded. `xwen fetch` prefetches; every default-model code path
//! ensures lazily.
//!
//! hf-hub reads `refs/main` verbatim (no trim), so anything else that writes
//! the cache must store the bare commit hash — `hf download` and hf-hub both
//! do.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hf_hub::Cache;
use hf_hub::api::sync::ApiBuilder;

use crate::config::Arch;
use crate::drafter::DrafterKind;

/// Which official checkpoint to run. All are ggml-org GGUF conversions; the
/// GGUF filenames keep the model's name — only the engine is called xwen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Model {
    /// Qwen3.6-27B, dense (`qwen35`).
    Qwen27B,
    /// Qwen3.6-35B-A3B, MoE (`qwen35moe`) — the bring-up model, and the
    /// checkpoint [`Model::default_servable`] names as its fallback for a plain
    /// default a cache-moving surface cannot run. Nothing takes that branch
    /// today.
    Qwen35BA3B,
    /// Qwen3.8-27B, dense (`qwen35`). The 3.8 release's config is byte-identical
    /// to [`Model::Qwen27B`]'s, so the two run the same graph at the same
    /// geometry and differ only in weights, repo — and drafter KIND: 3.8 ships an
    /// MTP head where the 3.6 checkpoints ship DFlash sidecars.
    Qwen3827B,
    /// Qwen3.8-Flash-Next (`qwen4exp`) — 512 routed experts, hyper-connection
    /// residuals, QSA attention and a PLE n-gram table. The first checkpoint
    /// here whose GGUF is a gguf-split SET rather than one file, the first
    /// that is not a ggml-org conversion (Qwen published no GGUF; Unsloth's
    /// UD-Q4_K_XL is the de-facto default), and the first that ships no drafter
    /// sidecar at all.
    ///
    /// The default checkpoint, on every surface: it is the best model here and a
    /// zero-flag run of any mode runs it. `xwen serve` and `xwen batch` joined
    /// the rest in P4 (2026-08-30), when the cache images learned to carry the
    /// QSA indexer rows and the PLE state. What it still does NOT do is download
    /// itself inside a request ([`Model::auto_fetch`]) or draft
    /// ([`Model::supports_drafting`]).
    #[default]
    Qwen38FlashNext,
    /// Qwen3-4B, dense (`qwen3`) — the base release, hybrid thinking. The first
    /// checkpoint here whose weights are Hugging Face BF16 safetensors rather
    /// than a GGUF, and the first on a second vocabulary (151936 against the
    /// 3.6 family's 248320).
    ///
    /// Not listed and not fetchable on its own yet ([`Model::servable`],
    /// [`Model::auto_fetch`]): the `qwen3` layer stack is not implemented, so
    /// nothing can run it. `generate`/`chat` flip both in the stack arc, serve
    /// after that.
    Qwen34B,
    /// Qwen3-4B-Instruct-2507, dense (`qwen3`). The same graph and geometry as
    /// [`Model::Qwen34B`]; its `config.json` differs in exactly two values,
    /// `rope_theta` 5e6 (against 1e6) and `max_position_embeddings` 262144
    /// (against 40960), and its chat template has no thinking mode at all.
    Qwen34BInstruct2507,
    /// The Z-Image-Turbo text encoder — `Tongyi-MAI/Z-Image-Turbo`'s
    /// `text_encoder/` directory, which is Qwen3-4B's config and (for layers
    /// 0-34) its weights, shipped inside a diffusion pipeline.
    ///
    /// ENCODE-ONLY, and the one entry here that is not a language model anyone
    /// should generate from: its third shard is corrupt. Two planes of layer 35
    /// — `mlp.up_proj` and `mlp.down_proj` — arrive as long runs of exact zeros
    /// that the base model's identical shard does not have. Harmless for what
    /// this entry is for, because the pipeline reads the hidden state after
    /// layer 34 ([`EncoderSpec::layer`]) and never evaluates layer 35's MLP;
    /// fatal for generation, which runs the whole stack. The loader knows the
    /// two planes by name ([`Model::safetensors_allowed_zero_runs`]) and
    /// refuses any OTHER zero-filled plane in any set.
    ZImageTurboEncoder,
}

/// Which tokenizer vocabulary a checkpoint speaks — see
/// [`Model::vocab_family`]. Two families ship here, and nothing about one
/// transfers to the other: not the ids, not the width, not the specials.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocabFamily {
    /// Qwen 3.6/3.8 — 248320 ids padded, real tokens to 248076.
    Qwen36,
    /// Qwen3 — 151936 ids padded, real tokens to 151668.
    Qwen3,
}

/// Every checkpoint this build knows, in the order surfaces that enumerate them
/// (`/v1/models`, an unknown-model error's list of valid names) print them.
pub const MODELS: [Model; 7] = [
    Model::Qwen27B,
    Model::Qwen35BA3B,
    Model::Qwen3827B,
    Model::Qwen38FlashNext,
    Model::Qwen34B,
    Model::Qwen34BInstruct2507,
    Model::ZImageTurboEncoder,
];

/// The hub coordinates of one checkpoint: the repo, the Q4_K_M target (the
/// quant this machine is built around), and its drafter sidecar — of whichever
/// kind the release ships — when it ships one. The sizes are the hub's own byte counts, rounded, and
/// exist only for the "this is about to download" notice.
struct Checkpoint {
    repo: &'static str,
    /// The target GGUF: one file, or — for a gguf-split checkpoint — every
    /// shard in shard order, whose FIRST entry is the path the loader is
    /// handed (`gguf::open` reads the sibling set off that name). Every entry
    /// is a path within the repo, so a quant that lives in a subfolder (the
    /// Unsloth layout) resolves into the same snapshot directory and the
    /// siblings sit next to each other, which is what the split open needs.
    files: &'static [&'static str],
    /// The model's own name, as the repo and the GGUF's `general.name` spell it.
    full_name: &'static str,
    /// Which graph the GGUF holds. Dense is no longer one-to-one with a
    /// checkpoint — two releases share it — so this identifies the arch alone.
    arch: Arch,
    model_size: &'static str,
    /// The drafter sidecar, of whichever kind the release ships, or `None` for
    /// one that ships none. Everything speculation needs is in here together: a
    /// checkpoint either drafts or it does not.
    drafter: Option<Drafter>,
    geometry: CacheGeometry,
    /// What the weights ARE on disk, which decides which loader opens them —
    /// and, for a safetensors set, everything the loader needs beyond the file
    /// list.
    format: Format,
}

/// How a checkpoint's weights are stored, and what opening them needs.
///
/// The distinction the registry has to carry rather than infer: `files()` is
/// just a list of names either way, but a GGUF is opened as a FILE and a
/// safetensors checkpoint as a DIRECTORY, with a separate tokenizer, a shard
/// index, and per-checkpoint knowledge of which planes are allowed to be
/// zero-filled. `crate::checkpoint::CheckpointSource` is what reads this.
///
/// Every path here is relative to the REPO root, like [`Checkpoint::files`],
/// which is not always the checkpoint directory: the Z-Image encoder lives in a
/// `text_encoder/` subdirectory with its tokenizer in a sibling one.
pub enum Format {
    Gguf,
    SafeTensors {
        /// `config.json` — the file `files()` lists first, so `cached_model`
        /// and `ensure_model` hand back a path inside the checkpoint directory.
        config: &'static str,
        index: &'static str,
        tokenizer: &'static str,
        /// Tensors whose long runs of exact zeros are known and tolerated. Any
        /// other plane with such a run fails the load: a run that long is a
        /// truncated or mis-uploaded shard, and it produces a model that runs
        /// and returns wrong numbers rather than one that fails.
        allow_zero_runs: &'static [&'static str],
        /// Present when this checkpoint is (also) a conditioning encoder, and
        /// says which hidden state its pipeline reads. `None` on a plain LM,
        /// which encodes from its last layer and caps nothing.
        encoder: Option<EncoderSpec>,
    },
}

/// Where a conditioning encoder's pipeline takes its hidden state, and how much
/// text it feeds in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderSpec {
    /// The `hidden_states` index the pipeline reads, in HF's numbering: 0 is the
    /// embedding output, N is the residual after `layers[N-1]`, before the final
    /// norm. Z-Image takes `hidden_states[-2]` of 37 entries, which is 35.
    pub layer: usize,
    /// The pipeline's own `max_length`, which it pads and truncates every prompt
    /// to. Z-Image's is 512; longer prompts are truncated, exactly as the
    /// pipeline truncates them.
    pub max_tokens: usize,
    /// The deepest layer this checkpoint's weights can be trusted to compute —
    /// the ceiling on `--layer`. Equal to [`EncoderSpec::layer`] on Z-Image,
    /// because layer 35's MLP is the corrupt plane and asking for a deeper
    /// hidden state would evaluate it.
    pub max_layer: usize,
}

/// One checkpoint's drafter sidecar.
struct Drafter {
    kind: DrafterKind,
    file: &'static str,
    size: &'static str,
    /// Layers in the sidecar the cache is sized over: `dflash.block_count` for a
    /// DFlash sidecar, and one for an MTP head, which is a single block however
    /// many the file declares. Within a kind it is the only dimension the
    /// shipped sidecars differ in, so it alone decides what a drafter cache
    /// costs.
    layers: usize,
    /// The drafting confidence floor this checkpoint decodes fastest at; see
    /// [`Model::draft_p_min_default`].
    p_min: f32,
}

/// The parts of a checkpoint's shape that decide what its caches cost, so a
/// sizing estimate can be quoted before the GGUF is open.
///
/// This duplicates what [`crate::XwenConfig`] reads from the file, and is only
/// for the estimates written before a model is loaded — `xwen serve --init`
/// writes its template with no checkpoint in hand. Anything holding a real
/// `XwenConfig` should measure from that instead of quoting these.
struct CacheGeometry {
    /// Layers whose KV cache grows with context. The rest are DeltaNet, whose
    /// state is a fixed size no matter how long the context is.
    full_attn_layers: usize,
    /// DeltaNet layers, which carry the recurrent state a snapshot copies.
    linear_layers: usize,
    n_kv_head: usize,
    /// Full-attention head dim (256 on both checkpoints).
    head_dim: usize,
    /// DeltaNet K-heads and V-heads, and the head dim shared by q, k and v.
    linear_k_heads: usize,
    linear_v_heads: usize,
    linear_head_dim: usize,
    /// Depthwise conv kernel over the fused DeltaNet qkv stream.
    conv_kernel: usize,
    /// The caches only the qwen4exp trunk carries, `None` on the qwen35 pair.
    extras: Option<Qwen4ExpCaches>,
}

/// The two extra caches a qwen4exp checkpoint carries beyond the KV rows and
/// the DeltaNet state every checkpoint has: the QSA indexer's own key plane
/// (per-token, so it grows with context alongside the KV rows) and the PLE
/// layer's dilated conv window (fixed-size recurrent state, so it rides in a
/// snapshot alongside the DeltaNet state).
///
/// Shapes only. Neither figure's DTYPE is decided here — the indexer plane is
/// sized by [`crate::qwen4exp::indexer::indexer_bytes_per_token`] and the conv
/// window by the f32 the PLE block allocates — because the graph units own that
/// and this table got it wrong once by assuming (f16 keys, matching the trunk's
/// KV rows; they are f32, and MQA besides).
struct Qwen4ExpCaches {
    /// Lightning-indexer key dim per full-attention layer
    /// (`attention.indexer.key_length` = 128). Keys only, and one head only:
    /// the indexer is MQA and it scores positions, it does not attend over
    /// values — so its 4 query heads (`head_count`) cost nothing per token and
    /// are not recorded here.
    indexer_head_dim: usize,
    /// Tokens per indexer block (`attention.indexer.compress_ratio` = 4): the
    /// derived block-key plane holds one key per this many tokens.
    indexer_compress_ratio: usize,
    /// Layers carrying a PLE block (`ple.layers` — a single layer today).
    ple_layers: usize,
    /// Columns of history the PLE conv keeps: kernel 4 at dilation 3 spans 10
    /// positions, so 9 previous columns are state.
    ple_conv_cols: usize,
    /// Width of that conv — the hyper-connection carrier, `hyper_connection.count`
    /// x `embedding_length` (4 x 2560), NOT the DeltaNet `conv_dim` it happens
    /// to equal on this checkpoint.
    ple_conv_width: usize,
}

/// 64 layers, full attention every fourth. Shared by both dense checkpoints:
/// Qwen3.8-27B's config is byte-identical to Qwen3.6-27B's.
const DENSE_27B_GEOMETRY: CacheGeometry = CacheGeometry {
    full_attn_layers: 16,
    linear_layers: 48,
    n_kv_head: 4,
    head_dim: 256,
    linear_k_heads: 16,
    linear_v_heads: 48,
    linear_head_dim: 128,
    conv_kernel: 4,
    extras: None,
};

const QWEN_27B: Checkpoint = Checkpoint {
    repo: "ggml-org/Qwen3.6-27B-GGUF",
    files: &["Qwen3.6-27B-Q4_K_M.gguf"],
    full_name: "Qwen3.6-27B",
    arch: Arch::Dense,
    model_size: "19.1 GB",
    drafter: Some(Drafter {
        kind: DrafterKind::Dflash,
        file: "dflash-Qwen3.6-27B-BF16.gguf",
        size: "3.5 GB",
        layers: 5,
        p_min: 0.5,
    }),
    geometry: DENSE_27B_GEOMETRY,
    format: Format::Gguf,
};

const QWEN_35B_A3B: Checkpoint = Checkpoint {
    repo: "ggml-org/Qwen3.6-35B-A3B-GGUF",
    files: &["Qwen3.6-35B-A3B-Q4_K_M.gguf"],
    full_name: "Qwen3.6-35B-A3B",
    arch: Arch::Moe,
    model_size: "20.4 GB",
    drafter: Some(Drafter {
        kind: DrafterKind::Dflash,
        file: "dflash-Qwen3.6-35B-A3B-BF16.gguf",
        size: "0.8 GB",
        layers: 6,
        p_min: 0.3,
    }),
    // 40 layers, full attention every fourth.
    geometry: CacheGeometry {
        full_attn_layers: 10,
        linear_layers: 30,
        n_kv_head: 2,
        head_dim: 256,
        linear_k_heads: 16,
        linear_v_heads: 32,
        linear_head_dim: 128,
        conv_kernel: 4,
        extras: None,
    },
    format: Format::Gguf,
};

/// The 3.8 release ships no DFlash sidecar. It ships an MTP head instead — one
/// extra trunk-flavour layer, chained rather than blocked — which is why this is
/// the one checkpoint whose speculation is a different shape from the other two.
const QWEN_38_27B: Checkpoint = Checkpoint {
    repo: "ggml-org/Qwen3.8-27B-GGUF",
    files: &["Qwen3.8-27B-Q4_K_M.gguf"],
    full_name: "Qwen3.8-27B",
    arch: Arch::Dense,
    model_size: "19.0 GB",
    drafter: Some(Drafter {
        kind: DrafterKind::Mtp,
        file: "mtp-Qwen3.8-27B-Q8_0.gguf",
        size: "3.2 GB",
        // The sidecar declares 65 blocks — the trunk's 64 plus the head — but
        // only the head's one is ever cached.
        layers: 1,
        p_min: 0.7,
    }),
    geometry: DENSE_27B_GEOMETRY,
    format: Format::Gguf,
};

/// Qwen3.8-Flash-Next, the `qwen4exp` checkpoint — and the one entry in this
/// table that is nobody's official conversion. Qwen published no GGUF at all;
/// `unsloth/Qwen3.8-Flash-Next-GGUF` is the only full quant ladder, and its
/// UD-Q4_K_XL mix is the de-facto default (`general.file_type` 15, the same
/// Q4_K_M family value the ggml-org files carry — the mix is an imatrix "XL"
/// override, so nothing should assert on which plane got which quant). The
/// ggml-org repo has Q8_0 only, which does not fit this machine.
///
/// Four shards, and the loader is handed the first: `gguf::open` reads
/// `split.count` and locates the siblings by the `-000NN-of-000MM` convention,
/// so all four must be in the cache — [`ensure_model`] fetches every one and
/// [`cached_model`] counts a partial set as a miss.
///
/// No drafter: the release ships an MTP head, which is not ported yet (the
/// sidecar-less shape every `Option` accessor here was written for, and which
/// no shipped checkpoint had exercised until now).
const QWEN_38_FLASH_NEXT: Checkpoint = Checkpoint {
    repo: "unsloth/Qwen3.8-Flash-Next-GGUF",
    files: &[
        "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
        "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf",
        "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf",
        "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf",
    ],
    full_name: "Qwen3.8-Flash-Next",
    arch: Arch::Qwen4Exp,
    model_size: "111.33 GB",
    drafter: None,
    // 48 layers, full attention every fourth (`full_attention_interval` 4, and
    // the `attention.compress_ratios` array marks the same twelve). The ssm
    // keys mislead exactly as they do on the qwen35 pair: `group_count` 16 is
    // K-heads, `time_step_rank` 48 is V-heads, `state_size` 128 is the head dim.
    geometry: CacheGeometry {
        full_attn_layers: 12,
        linear_layers: 36,
        n_kv_head: 2,
        head_dim: 256,
        linear_k_heads: 16,
        linear_v_heads: 48,
        linear_head_dim: 128,
        conv_kernel: 4,
        extras: Some(Qwen4ExpCaches {
            indexer_head_dim: 128,
            indexer_compress_ratio: 4,
            ple_layers: 1,
            ple_conv_cols: 9,
            ple_conv_width: 4 * 2560,
        }),
    },
    format: Format::Gguf,
};

/// 36 layers, ALL full attention, 8 KV heads at head_dim 128. Shared by every
/// `qwen3` checkpoint: the three releases differ only in `rope_theta` and
/// `max_position_embeddings`, neither of which a cache is sized by.
///
/// The `linear_*` and `conv_kernel` fields are zero because this architecture
/// has no recurrent layers at all — which is also why its snapshots cost
/// nothing: a `qwen3` conversation's whole state is its KV rows.
const QWEN3_4B_GEOMETRY: CacheGeometry = CacheGeometry {
    full_attn_layers: 36,
    linear_layers: 0,
    n_kv_head: 8,
    head_dim: 128,
    linear_k_heads: 0,
    linear_v_heads: 0,
    linear_head_dim: 0,
    conv_kernel: 0,
    extras: None,
};

/// The three shards and the two JSON files of a Qwen3-4B safetensors set, as
/// the Hugging Face repos lay them out at the repo root. The Z-Image encoder
/// ships the same five under `text_encoder/`.
const QWEN3_4B_SHARDS: [&str; 3] = [
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
];

const QWEN3_4B: Checkpoint = Checkpoint {
    repo: "Qwen/Qwen3-4B",
    files: &[
        "config.json",
        "model.safetensors.index.json",
        QWEN3_4B_SHARDS[0],
        QWEN3_4B_SHARDS[1],
        QWEN3_4B_SHARDS[2],
        "tokenizer.json",
    ],
    full_name: "Qwen3-4B",
    arch: Arch::Qwen3,
    // Measured with `stat -L` over the six files above: 8,056,438,199 bytes,
    // of which 8,044,981,999 is the three shards.
    model_size: "8.06 GB",
    drafter: None,
    geometry: QWEN3_4B_GEOMETRY,
    format: Format::SafeTensors {
        config: "config.json",
        index: "model.safetensors.index.json",
        tokenizer: "tokenizer.json",
        allow_zero_runs: &[],
        encoder: None,
    },
};

const QWEN3_4B_INSTRUCT_2507: Checkpoint = Checkpoint {
    repo: "Qwen/Qwen3-4B-Instruct-2507",
    files: &[
        "config.json",
        "model.safetensors.index.json",
        QWEN3_4B_SHARDS[0],
        QWEN3_4B_SHARDS[1],
        QWEN3_4B_SHARDS[2],
        "tokenizer.json",
    ],
    full_name: "Qwen3-4B-Instruct-2507",
    arch: Arch::Qwen3,
    model_size: "8.06 GB",
    drafter: None,
    geometry: QWEN3_4B_GEOMETRY,
    format: Format::SafeTensors {
        config: "config.json",
        index: "model.safetensors.index.json",
        tokenizer: "tokenizer.json",
        allow_zero_runs: &[],
        encoder: None,
    },
};

/// The Z-Image-Turbo text encoder: one subdirectory of a diffusion repo, with
/// its tokenizer in a sibling subdirectory — the reason [`Checkpoint::files`]
/// and [`Format::SafeTensors`] name paths relative to the REPO rather than to
/// the checkpoint directory.
///
/// The two allowlisted planes are the corrupt ones (see
/// [`Model::ZImageTurboEncoder`]), and `max_layer` is the encoder's own layer
/// for the same reason: 35 is as deep as these weights can be trusted.
const Z_IMAGE_TURBO_ENCODER: Checkpoint = Checkpoint {
    repo: "Tongyi-MAI/Z-Image-Turbo",
    files: &[
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/model-00001-of-00003.safetensors",
        "text_encoder/model-00002-of-00003.safetensors",
        "text_encoder/model-00003-of-00003.safetensors",
        "tokenizer/tokenizer.json",
    ],
    full_name: "Z-Image-Turbo-text-encoder",
    arch: Arch::Qwen3,
    model_size: "8.06 GB",
    drafter: None,
    geometry: QWEN3_4B_GEOMETRY,
    format: Format::SafeTensors {
        config: "text_encoder/config.json",
        index: "text_encoder/model.safetensors.index.json",
        tokenizer: "tokenizer/tokenizer.json",
        allow_zero_runs: &[
            "model.layers.35.mlp.up_proj.weight",
            "model.layers.35.mlp.down_proj.weight",
        ],
        encoder: Some(EncoderSpec {
            layer: 35,
            max_tokens: 512,
            max_layer: 35,
        }),
    },
};

impl Model {
    const fn checkpoint(self) -> &'static Checkpoint {
        match self {
            Model::Qwen27B => &QWEN_27B,
            Model::Qwen35BA3B => &QWEN_35B_A3B,
            Model::Qwen3827B => &QWEN_38_27B,
            Model::Qwen38FlashNext => &QWEN_38_FLASH_NEXT,
            Model::Qwen34B => &QWEN3_4B,
            Model::Qwen34BInstruct2507 => &QWEN3_4B_INSTRUCT_2507,
            Model::ZImageTurboEncoder => &Z_IMAGE_TURBO_ENCODER,
        }
    }

    /// Whether this checkpoint's weights are a Hugging Face safetensors
    /// DIRECTORY rather than a GGUF file — which decides which loader opens it
    /// and, on the surfaces that name a file, what `--model` may point at.
    pub const fn is_safetensors(self) -> bool {
        matches!(self.checkpoint().format, Format::SafeTensors { .. })
    }

    /// The repo-relative path of this checkpoint's `tokenizer.json`, or `None`
    /// for a GGUF, whose tokenizer is embedded in the file.
    pub const fn safetensors_tokenizer(self) -> Option<&'static str> {
        match &self.checkpoint().format {
            Format::SafeTensors { tokenizer, .. } => Some(tokenizer),
            Format::Gguf => None,
        }
    }

    /// The tensors this checkpoint is allowed to ship zero-filled. Empty for
    /// every checkpoint but the Z-Image encoder, and empty for a GGUF, whose
    /// loader does not scan.
    pub const fn safetensors_allowed_zero_runs(self) -> &'static [&'static str] {
        match &self.checkpoint().format {
            Format::SafeTensors {
                allow_zero_runs, ..
            } => allow_zero_runs,
            Format::Gguf => &[],
        }
    }

    /// Where this checkpoint's conditioning-encoder pipeline reads its hidden
    /// state, or `None` for a checkpoint that is only a language model.
    pub const fn encoder_spec(self) -> Option<EncoderSpec> {
        match &self.checkpoint().format {
            Format::SafeTensors {
                encoder: Some(spec),
                ..
            } => Some(*spec),
            Format::SafeTensors { .. } | Format::Gguf => None,
        }
    }

    /// The `rope_theta` this checkpoint's `config.json` carries, or `None` for a
    /// GGUF checkpoint, whose release is named in the file instead.
    ///
    /// This is the cross-check a safetensors set gets in place of the name one:
    /// a directory says nothing about which release it is, but the two `qwen3`
    /// releases in this registry disagree here, so `--model-size` naming one of
    /// them against a directory carrying the other's theta is refused
    /// ([`crate::XwenConfig::identify`]).
    pub const fn safetensors_rope_theta(self) -> Option<f32> {
        match self {
            Model::Qwen34B | Model::ZImageTurboEncoder => Some(1e6),
            Model::Qwen34BInstruct2507 => Some(5e6),
            Model::Qwen27B | Model::Qwen35BA3B | Model::Qwen3827B | Model::Qwen38FlashNext => None,
        }
    }

    /// Which `qwen3` release a `rope_theta` names, for a directory that
    /// identifies as nothing else.
    ///
    /// 5e6 is Instruct-2507's alone. 1e6 is shared by the base model and the
    /// Z-Image text encoder, whose configs are byte-identical, and the tie goes
    /// to the base model: it is the language model, it is what an unpacked copy
    /// of these weights most likely is, and the encoder entry is reachable by
    /// naming it (`--model-size zimage-turbo`) where the LM is what silence
    /// gets. A theta neither release uses answers `None`.
    pub fn by_safetensors_rope_theta(theta: f32) -> Option<Model> {
        [Model::Qwen34B, Model::Qwen34BInstruct2507]
            .into_iter()
            .find(|model| model.safetensors_rope_theta() == Some(theta))
    }

    /// Which tokenizer vocabulary this checkpoint speaks.
    ///
    /// A checkpoint's ids only mean anything inside its own family: the Qwen 3.6
    /// vocabulary is 248320 wide with `<|im_end|>` at 248046, the Qwen3 one is
    /// 151936 wide with it at 151645. Anything holding a tokenizer, a grammar
    /// trie or a hardcoded special id per checkpoint keys on this rather than on
    /// the checkpoint, because every member of a family shares all of it.
    pub const fn vocab_family(self) -> VocabFamily {
        match self {
            Model::Qwen27B | Model::Qwen35BA3B | Model::Qwen3827B | Model::Qwen38FlashNext => {
                VocabFamily::Qwen36
            }
            Model::Qwen34B | Model::Qwen34BInstruct2507 | Model::ZImageTurboEncoder => {
                VocabFamily::Qwen3
            }
        }
    }

    /// The Hugging Face repo holding both the target and its drafter.
    pub const fn repo(self) -> &'static str {
        self.checkpoint().repo
    }

    /// The target GGUF the loader is handed: the file itself for a single-file
    /// checkpoint, and the FIRST SHARD for a split one — which is what
    /// `gguf::open` wants, since it reads `split.count` there and locates the
    /// siblings by name. Callers that need every byte on disk (a cache check, a
    /// fetch) want [`Model::files`] instead.
    pub const fn file(self) -> &'static str {
        self.checkpoint().files[0]
    }

    /// Every file of the target GGUF, in shard order — one entry for a
    /// single-file checkpoint, four for Qwen3.8-Flash-Next. A split checkpoint
    /// is only usable with all of them present, so anything that decides
    /// "is it here?" or "fetch it" iterates this rather than [`Model::file`].
    pub const fn files(self) -> &'static [&'static str] {
        self.checkpoint().files
    }

    /// The checkpoint's own name — what ggml-org calls the repo, what the GGUF
    /// carries as `general.name`, and the only spelling the HTTP APIs accept or
    /// echo. Quant-independent: one name covers every quantization of the
    /// checkpoint. The short aliases ([`Model::to_string`]) are the CLI's.
    pub const fn full_name(self) -> &'static str {
        self.checkpoint().full_name
    }

    /// Which graph this checkpoint's GGUF holds.
    pub const fn arch(self) -> Arch {
        self.checkpoint().arch
    }

    /// Which release's chat template renders this checkpoint's conversations.
    /// The 3.8 release ships its own template (a `reasoning_effort` system
    /// preamble, a flipped `preserve_thinking` default); the 3.6 pair share
    /// the original. Everything that builds a prompt for a checkpoint reaches
    /// its [`crate::chat::ChatOptions`] through this, usually via
    /// [`crate::chat::ChatOptions::for_dialect`].
    ///
    /// Qwen3.8-Flash-Next renders as Qwen38 too: its embedded template is the
    /// 3.8 one with Unsloth's own packaging fixes on top (a `developer` role,
    /// leading system/developer messages merged instead of refused, `high`
    /// aliased onto `xhigh`, tool-call arguments validated). None of those
    /// touch what this dialect decides — the reasoning_effort preamble and its
    /// levels, the preserve_thinking default, the open `<think>` generation
    /// prompt and the no-inline-split rule are character-for-character the 3.8
    /// template's.
    pub const fn chat_dialect(self) -> crate::chat::ChatDialect {
        match self {
            Model::Qwen27B | Model::Qwen35BA3B => crate::chat::ChatDialect::Qwen36,
            Model::Qwen3827B | Model::Qwen38FlashNext => crate::chat::ChatDialect::Qwen38,
            // The encoder renders the base template: Z-Image's own
            // `tokenizer_config.json` carries Qwen3-4B's chat template
            // character for character, and the pipeline renders one user turn
            // through it with `add_generation_prompt`.
            Model::Qwen34B | Model::ZImageTurboEncoder => crate::chat::ChatDialect::Qwen3,
            Model::Qwen34BInstruct2507 => crate::chat::ChatDialect::Qwen3Instruct,
        }
    }

    /// The presence penalty this checkpoint's model card asks for in the given
    /// chat mode. On the four Qwen 3.6/3.8 checkpoints non-thinking is 1.5 and
    /// thinking is 1.5 on Qwen3.6-35B-A3B alone, 0.0 on the other three. Zero
    /// on every `qwen3` checkpoint: those cards list `presence_penalty` as an
    /// optional 0-2 knob for reducing repetition, not as a recommended default,
    /// so applying one would be this runtime's opinion rather than the card's.
    ///
    /// This is the one card value that is NOT shared across the checkpoints,
    /// which is why it sits here on `Model` rather than beside temperature and
    /// top-p on `SamplerOptions::recommended`. Everything that resolves a
    /// request's sampling reads it through
    /// [`crate::sampler::SamplerOptions::recommended_for`], which is the single
    /// place the checkpoint and the mode meet.
    ///
    /// Hardcoded here for the same reason the second stop id is: the GGUF
    /// converters only know `repetition_penalty`, so no converted file carries
    /// a presence-penalty key at any spelling, and a value read from the file
    /// would be a value that is never there.
    /// Exhaustive on purpose, with no `_` arm: a wildcard here handed every new
    /// checkpoint the 3.6 family's 1.5 without anyone deciding it.
    pub const fn recommended_presence_penalty(self, thinking: bool) -> f64 {
        match self {
            Model::Qwen35BA3B => 1.5,
            Model::Qwen27B | Model::Qwen3827B | Model::Qwen38FlashNext => {
                if thinking {
                    0.0
                } else {
                    1.5
                }
            }
            Model::Qwen34B | Model::Qwen34BInstruct2507 | Model::ZImageTurboEncoder => 0.0,
        }
    }

    /// The drafter sidecar for speculative decoding, or `None` for a checkpoint
    /// that ships none — which decodes plain. Which KIND of drafter the file
    /// holds is [`Model::drafter_kind`], and the file itself is the authority
    /// once opened (`drafter::classify`).
    pub const fn drafter_file(self) -> Option<&'static str> {
        match &self.checkpoint().drafter {
            Some(drafter) => Some(drafter.file),
            None => None,
        }
    }

    /// Human-readable download size of the target, for the fetch notice.
    pub const fn size(self) -> &'static str {
        self.checkpoint().model_size
    }

    /// Whether this checkpoint may be downloaded as a side effect of something
    /// else — a serve request that names it, say — or only when an operator
    /// asked for it in so many words.
    ///
    /// True for every checkpoint but Qwen3.8-Flash-Next. The rule is a size one:
    /// a fetch over 100 GB is explicit-only. The three ~20 GB files are a
    /// download someone can absorb inside a request that stalls for a few
    /// minutes; 111 GB across four shards is not — it is most of a disk and the
    /// better part of an hour, started by a client that only misspelled which
    /// model it wanted. `xwen fetch` and the CLI one-shots still fetch it,
    /// because there the operator named it.
    ///
    /// Not a per-checkpoint policy knob and not read from the table's size
    /// string: it is spelled out here so that adding a big checkpoint is a
    /// decision someone makes rather than one they inherit.
    /// False on the three `qwen3` checkpoints for a different reason than the
    /// size one: nothing can RUN them yet, so a fetch triggered on their behalf
    /// would download 8 GB for a load that fails. Flipped to true by the arc
    /// that lands the `qwen3` layer stack.
    pub const fn auto_fetch(self) -> bool {
        match self {
            Model::Qwen27B | Model::Qwen35BA3B | Model::Qwen3827B => true,
            Model::Qwen38FlashNext => false,
            Model::Qwen34B | Model::Qwen34BInstruct2507 | Model::ZImageTurboEncoder => false,
        }
    }

    /// Whether the surfaces that MOVE cache state can run this checkpoint at
    /// all: `xwen serve` and `xwen batch`.
    ///
    /// False on the three `qwen3` checkpoints, which is the first time this
    /// seam has actually said no since it was kept for the purpose: that
    /// architecture is half-ported — its config and its weights load, its layer
    /// stack does not exist — so serve and batch refuse it rather than
    /// accepting a job they cannot finish. The two language models flip to true
    /// in the serve arc; `ZImageTurboEncoder` stays false permanently, because
    /// it is an encoder and generating from it would run its corrupt layer.
    ///
    /// True for every GGUF checkpoint, and has been since the
    /// qwen4exp cache images landed (2026-08-30): the QSA indexers' raw keys
    /// travel with the full-attention rows in a `HostFullKv` and the PLE conv
    /// window and n-gram history ride on their layer's snapshot entry, so a
    /// snapshot, a rewind, a page-out and a stored image all carry the whole
    /// conversation on qwen4exp exactly as they do on the DeltaNet trunk.
    ///
    /// Kept as a method rather than deleted because it is the question those
    /// surfaces actually ask, and the next architecture that arrives half-ported
    /// needs somewhere to say no. It is not a policy knob: a checkpoint answers
    /// false here only while some part of its state has no image.
    pub const fn servable(self) -> bool {
        match self {
            Model::Qwen27B | Model::Qwen35BA3B | Model::Qwen3827B | Model::Qwen38FlashNext => true,
            Model::Qwen34B | Model::Qwen34BInstruct2507 | Model::ZImageTurboEncoder => false,
        }
    }

    /// The checkpoint the cache-moving surfaces run when nothing named one:
    /// [`Model::default`] when they can run it, and otherwise the best one they
    /// can. Both `xwen serve` (no `--model`/`--model-size`, no config `model`)
    /// and `xwen batch` (no `"model"` in the payload) resolve their zero-flag
    /// default through here, so the two cannot drift into answering with
    /// different checkpoints.
    ///
    /// Separate from `default` because the two answer different questions.
    /// `default` is "the best checkpoint here", which `xwen generate` and
    /// `xwen chat` run with no flags; this is "the best one a surface that moves
    /// cache state can run". Every registry checkpoint is servable as of
    /// 2026-08-30, so the two answer the same today and the fallback below is
    /// never taken — it is kept for the next architecture that arrives before
    /// its cache images do.
    ///
    /// The fallback is NAMED rather than derived from [`MODELS`], because that
    /// array's order is the one `/v1/models` prints in — a display order, not a
    /// preference order. Taking its first servable entry would hand the server
    /// the 27B, which decodes at a quarter of the 35B-A3B's rate: a silent
    /// regression for everyone already serving, dressed up as a derivation.
    pub fn default_servable() -> Model {
        let default = Model::default();
        if default.servable() {
            return default;
        }
        Model::Qwen35BA3B
    }

    /// Whether this checkpoint's graph can be drafted for at all — by its own
    /// sidecar or by anyone's.
    ///
    /// Distinct from [`Model::drafter_kind`] being `None`, which only says the
    /// release ships no sidecar; a checkpoint like that could still be handed a
    /// drafter GGUF of someone's own, and `--draft <path>` is the flag for it.
    /// This is about the TARGET side: qwen4exp has no verify seam wired yet
    /// (D6), so no sidecar of any kind could be attached to it, and a run that
    /// asked for one is told so instead of being handed a file that will not
    /// load.
    pub const fn supports_drafting(self) -> bool {
        match self {
            Model::Qwen27B | Model::Qwen35BA3B | Model::Qwen3827B => true,
            Model::Qwen38FlashNext => false,
            // No drafter of any kind exists for the qwen3 graph, and no release
            // ships one: there is nothing to attach and nothing to verify with.
            Model::Qwen34B | Model::Qwen34BInstruct2507 | Model::ZImageTurboEncoder => false,
        }
    }

    /// Whether a run that said nothing about drafting speculates on this
    /// checkpoint. Not whether it CAN (that is
    /// [`Model::supports_drafting`]) and not whether a sidecar exists (that is
    /// [`Model::drafter_kind`]) — only what silence means. An explicit
    /// `--draft official` (or the serve config's `draft.path`/`draft.enabled`)
    /// still attaches the sidecar where this is false, and `--no-draft` still
    /// declines it where this is true.
    ///
    /// True on the 27B and the 3.8-27B, whose drafted arms were measured well
    /// above plain at their fitted defaults — +46 to +52% and +44 to +45% on
    /// code — and stay there.
    ///
    /// False on the 35B-A3B since 2026-09-06. Two independent measurements that
    /// day read its drafted arm BELOW plain at every length: -8% at 1k tokens
    /// deepening to -37% at 16k, and -4% on a 256-token code prompt. The cause
    /// is not the drafter: the router gemv lifted PLAIN decode on this
    /// checkpoint by 10.3%, and the drafting defaults were fitted against the
    /// old plain level, so what used to be a +26 to +28% win is now a loss. The
    /// fitted floor and depth below are deliberately untouched — refitting them
    /// is the experiment that decides whether this arm goes back to true.
    /// See docs/decisions/speculative-decoding.md and docs/log.md, 2026-09-06.
    ///
    /// False for a checkpoint that cannot be drafted for at all, which has
    /// nothing for silence to turn on.
    pub const fn draft_default_on(self) -> bool {
        match self {
            Model::Qwen27B | Model::Qwen3827B => true,
            Model::Qwen35BA3B => false,
            Model::Qwen38FlashNext => false,
            Model::Qwen34B | Model::Qwen34BInstruct2507 | Model::ZImageTurboEncoder => false,
        }
    }

    /// The one sentence every surface says when a run that asked for nothing
    /// decodes plain only because this checkpoint's default is off. It ships a
    /// drafter, so saying nothing would read as a missing sidecar.
    ///
    /// `opt_in` is the surface's own spelling of the flag that turns it on,
    /// because the CLI and a config file spell it differently and a line that
    /// named the wrong one would be worse than no line.
    pub fn draft_default_off_message(self, opt_in: &str) -> String {
        format!(
            "drafting is off by default on {} (it read below plain on 2026-09-06); \
             {opt_in} to enable",
            self.full_name()
        )
    }

    /// The one sentence every surface says when a run asks a checkpoint that
    /// cannot be drafted for to draft.
    pub fn no_drafting_message(self) -> String {
        format!(
            "no drafter kind is supported for {} yet: its graph has no speculative verify \
             seam, so neither an official sidecar nor a drafter GGUF of your own can be \
             attached. Decode plain — drop the flag, or pass --no-draft",
            self.full_name()
        )
    }

    /// Human-readable download size of the drafter, for the fetch notice.
    pub const fn drafter_size(self) -> Option<&'static str> {
        match &self.checkpoint().drafter {
            Some(drafter) => Some(drafter.size),
            None => None,
        }
    }

    /// The checkpoint a full name selects, case-insensitively — the API's whole
    /// model vocabulary. Deliberately narrower than [`std::str::FromStr`]: the
    /// short aliases are a CLI convenience, and honoring them on the wire made
    /// one checkpoint answer to several ids.
    pub fn from_api_name(name: &str) -> Option<Self> {
        MODELS
            .into_iter()
            .find(|model| model.full_name().eq_ignore_ascii_case(name.trim()))
    }

    /// Which official checkpoint a GGUF is, from what the file says about
    /// itself: `general.name` first, then the file name, and `None` when
    /// neither identifies one.
    ///
    /// A name, not an architecture, because the architecture answers neither
    /// question it is asked here. It cannot tell the 3.6 and 3.8 dense releases
    /// apart — they share the graph and every hyperparameter — and it cannot
    /// tell an official checkpoint from someone's conversion of something else
    /// onto the same graph, which is the difference between reporting a served
    /// file under a checkpoint's name and reporting it under its own. The
    /// architecture only narrows the candidates.
    ///
    /// One rule for both sources: an exact full-name match, then a full name
    /// found INSIDE the name (case-insensitive). The substring form is what
    /// accepts the shapes real files take — `Qwen3.8-27B-Q8_0.gguf` with its
    /// quant suffix, `Qwen3.6-27B-Instruct` from a re-quantized conversion — and
    /// the requirement that a WHOLE checkpoint name appear is what keeps it
    /// honest. A looser rule was tried and refused: matching a bare release
    /// series ("3.6"/"3.8") identifies `My-Qwen3.6-14B-finetune.gguf` as the
    /// official 27B, and `MyMoE-3.6` as Qwen3.6-35B-A3B — and since the MoE
    /// architecture has exactly one candidate, that second one needs no
    /// ambiguity to go wrong. Both would answer an official name with weights
    /// nobody checked, which is the one thing this function exists to prevent.
    ///
    /// The two passes compare differently, and deliberately. The EXACT pass
    /// folds spaces to hyphens (`hyphenate`) so that the space spelling a GGUF
    /// carries — "Qwen3.8 Flash Next" — still names the checkpoint the repo and
    /// the API hyphenate. The SUBSTRING pass does not fold: it asks whether the
    /// name literally spells a checkpoint. Folding it too would make
    /// "Qwen3.6 27B MyFinetune" — a finetune that says whose 27B it started from
    /// and is not that checkpoint — identify as the official Qwen3.6-27B, which
    /// is the failure the whole-name requirement exists to prevent. A file whose
    /// name really is the checkpoint plus a suffix spells it with hyphens,
    /// because that is how the release spells it.
    ///
    /// A name that matches more than one checkpoint identifies as none of them
    /// rather than as whichever the table lists first — an ambiguous name is
    /// exactly the case where guessing is worst.
    ///
    /// `general.name` is tried before the file name because it is what the
    /// converter wrote INTO the file about the model it holds, where a file name
    /// is only what somebody called the file. Both blessed checkpoints carry
    /// their exact full name there, so the substring pass is a courtesy to
    /// re-quantizers rather than something the shipped files need.
    ///
    /// Callers that need an answer regardless fall back to [`Arch::model`] and
    /// say so.
    pub fn identify(arch: Arch, general_name: Option<&str>, file: Option<&Path>) -> Option<Self> {
        // A safetensors checkpoint answers a different question, because it has
        // no name to read: see `identify_cached_dir`.
        if arch == Arch::Qwen3 {
            return file.and_then(Self::identify_cached_dir);
        }
        let candidates = || MODELS.into_iter().filter(|model| model.arch() == arch);
        // Only one candidate can match, or none does.
        let sole = |mut hits: Vec<Model>| -> Option<Model> {
            hits.dedup();
            match hits.as_slice() {
                [only] => Some(*only),
                _ => None,
            }
        };
        let by_name = |name: &str| -> Option<Model> {
            let name = name.trim();
            if name.is_empty() {
                return None;
            }
            let folded = hyphenate(name);
            sole(
                candidates()
                    .filter(|model| hyphenate(model.full_name()) == folded)
                    .collect(),
            )
            .or_else(|| {
                sole(
                    candidates()
                        .filter(|model| contains_ignore_ascii_case(name, model.full_name()))
                        .collect(),
                )
            })
        };
        if let Some(model) = general_name.and_then(by_name) {
            return Some(model);
        }
        let stem = file.and_then(Path::file_stem)?.to_string_lossy();
        by_name(&stem)
    }

    /// Which official checkpoint a safetensors DIRECTORY is: the one whose own
    /// cached snapshot it IS, and otherwise none.
    ///
    /// The GGUF rule cannot be reused, because it reads a name and there is no
    /// name to read. A `config.json` says what the ARCHITECTURE is and (through
    /// `rope_theta`) which release, but it says nothing about whose weights sit
    /// beside it — and two of the three checkpoints on this architecture,
    /// `Qwen/Qwen3-4B` and the Z-Image text encoder, have byte-identical
    /// configs while shipping different weights (one of the encoder's shards is
    /// corrupt). A directory name is worse still: in the Hugging Face cache it
    /// is a commit hash, and anywhere else it is whatever somebody called the
    /// folder.
    ///
    /// So the only thing that can honestly say "these are the official
    /// weights" is provenance: this directory is the one the hub cache holds
    /// for that entry. Anything else identifies as nothing, runs under its own
    /// directory name as [`crate::Identity::Assumed`], and can still be pinned
    /// with `--model-size` — which is exactly the shape of the GGUF rule, for
    /// the same reason.
    ///
    /// `path` may be the directory or a file inside it (`ensure_model` hands
    /// back `config.json`), and only DIRECTORIES are canonicalized: the cache's
    /// files are symlinks into a shared `blobs/` store, so resolving one would
    /// compare the blob directory against itself for every entry.
    fn identify_cached_dir(path: &Path) -> Option<Model> {
        let dir = canonical_checkpoint_dir(path)?;
        MODELS
            .into_iter()
            .filter(|model| model.is_safetensors())
            .find(|model| {
                cached_model(*model)
                    .as_deref()
                    .and_then(canonical_checkpoint_dir)
                    .is_some_and(|cached| cached == dir)
            })
    }

    /// The deepest draft this checkpoint's drafter is asked for when the run did
    /// not pin `--draft-max`, or `None` for one that ships no sidecar.
    ///
    /// Per-checkpoint because the two kinds have opposite economics and read the
    /// same flag. A DFlash sidecar proposes its whole block in ONE forward, so
    /// widening it is nearly free and its structural ceiling (`block_size - 1`,
    /// 15 on both shipped sidecars) is the sensible ask. An MTP head pays a
    /// forward per step AND compounds k-1 of its own guesses into step k, so
    /// depth costs linearly and pays off geometrically less.
    ///
    /// The MTP arm is 4, FITTED 2026-08-15 (Stage C) rather than inherited:
    /// llama.cpp's default for this head is 3, and a 3x3 p_min-by-depth sweep on
    /// this machine had every depth-4 arm ahead of its depth-3 sibling, almost
    /// entirely on the chat fixture (+36.7 to +39.2% over plain against +27.5 to
    /// +32.9%) while code was a wash. The optimum is bracketed, not merely the
    /// grid edge: a follow-up probe at p_min 0.7 read 34.9 / 34.0 / 32.6 / 25.4
    /// mean-of-medians at depths 4 / 5 / 6 / 8, so it falls away on both sides.
    /// Depth 8 is where the auto-pause controller starts firing (34-80 rounds
    /// paused) and drafting stops paying at all — the ceiling is real and the
    /// controller finds it.
    ///
    /// A default only. `--draft-max` and the serve config's `draft.max` override
    /// it, which is what lets a sweep explore depth without a rebuild — the
    /// reason this stopped being a constant inside the head.
    pub const fn draft_max_default(self) -> Option<usize> {
        match &self.checkpoint().drafter {
            Some(drafter) => Some(match drafter.kind {
                DrafterKind::Dflash => 15,
                DrafterKind::Mtp => 4,
            }),
            None => None,
        }
    }

    /// The drafting confidence floor (`--draft-p-min`) this checkpoint decodes
    /// fastest at: a round stops proposing at the first drafter token whose
    /// full-vocab probability falls below it, so a higher floor drafts shorter
    /// at higher acceptance.
    ///
    /// The drafting checkpoints sit at different points on that trade, which is
    /// why this is per-model rather than one shipped constant. Fitted 2026-08-08
    /// by two independent 120-run sweeps (`scripts/retune-draft.ts`) that picked
    /// the same winner each time: 0.5 on the 27B — 37.2-37.3 tok/s
    /// mean-of-medians against 33.0-33.5 at 0.3, where the shorter drafts run
    /// 78-86% acceptance and the chat prompt stops auto-pausing altogether —
    /// and 0.3 on the 35B-A3B, whose cheaper target forward still profits from
    /// drafting deeper at lower acceptance. `pause_margin` was swept alongside
    /// and stayed a shared 1.0.
    ///
    /// 0.7 on the 3.8-27B, fitted 2026-08-15 (Stage C) crossed with depth. This
    /// is the WEAKEST-held of the three arms and should be read as such: across
    /// the 3x3 grid the floor moved the mean-of-medians by at most 1.8% at a
    /// fixed depth (33.5 / 33.2 / 33.8 at depth 4 for 0.3 / 0.5 / 0.7), where
    /// depth moved it by 12%. What the floor clearly does change is wasted work
    /// — acceptance at depth 4 runs 65.5% at 0.3 against 80.0% at 0.7 — which
    /// costs nothing measurable at batch 1 on this machine because the target
    /// forward dominates, but would matter anywhere the drafter competes for the
    /// same silicon. 0.7 won every comparison it was in; it did not win by much.
    ///
    /// `None` for a checkpoint that ships no sidecar: there was nothing to fit a
    /// floor with, and nothing that reads one.
    ///
    /// This is the default only: `--draft-p-min` and the serve config's
    /// `draft.p_min` override it as before.
    pub const fn draft_p_min_default(self) -> Option<f32> {
        match &self.checkpoint().drafter {
            Some(drafter) => Some(drafter.p_min),
            None => None,
        }
    }

    /// Bytes of KV cache one more token of context costs.
    ///
    /// Only the full-attention layers grow with context; the DeltaNet layers
    /// carry a fixed recurrent state instead, which is what
    /// [`Model::snapshot_bytes`] accounts for. K and V are stored f16, matching
    /// what `LayerCache::new` allocates.
    ///
    /// A qwen4exp checkpoint's full-attention layers carry a SECOND per-token
    /// plane: the QSA lightning indexer's raw keys, which have to be kept for
    /// every position because that is what the indexer scores. It is one MQA
    /// key head at 128, held f32 — 512 B/token/layer against the trunk's 2048,
    /// plus 128 for the derived block-key plane (one f32 key per 4 tokens) —
    /// so leaving it out would under-size an operator's context budget by
    /// nearly a quarter. The per-token figure comes from
    /// [`crate::qwen4exp::indexer::indexer_bytes_per_token`], the same function
    /// the allocation itself is sized with, rather than being restated here:
    /// the head count and the dtype are both easy to get wrong from the outside
    /// (4 query heads and the trunk's f16 would give 1024).
    pub const fn kv_bytes_per_token(self) -> usize {
        let g = &self.checkpoint().geometry;
        let trunk = g.n_kv_head * g.head_dim * 2 * 2;
        let indexer = match &g.extras {
            Some(extras) => crate::qwen4exp::indexer::indexer_bytes_per_token(
                extras.indexer_head_dim,
                extras.indexer_compress_ratio,
            ),
            None => 0,
        };
        g.full_attn_layers * (trunk + indexer)
    }

    /// Bytes of drafter KV cache one more token of context costs, when a
    /// drafter is attached, or `None` for a checkpoint with no sidecar to size.
    ///
    /// The two kinds differ in heads, head dim AND dtype, so neither figure is
    /// the other's with a different layer count. A DFlash sidecar has a geometry
    /// of its OWN — 8 KV heads at head_dim 128 — and stores its cache f32,
    /// because the drafter is tiny and the exactness keeps its export/import
    /// round trip free of rounding (see `DrafterImage`). An MTP head has no
    /// geometry of its own at all: it reuses the trunk's full-attention
    /// `LayerCache`, so it inherits the trunk's heads, head dim and f16 dtype,
    /// and is read off the same [`CacheGeometry`] the target's own
    /// [`Model::kv_bytes_per_token`] is read from rather than restated here.
    ///
    /// 40 KiB/token on the 27B's five DFlash layers, 48 on the 35B-A3B's six,
    /// and 4 KiB on the 3.8's single-layer MTP head — an order of magnitude
    /// cheaper to give context to than either block drafter.
    pub const fn draft_kv_bytes_per_token(self) -> Option<usize> {
        let checkpoint = self.checkpoint();
        let g = &checkpoint.geometry;
        match &checkpoint.drafter {
            Some(drafter) => Some(match drafter.kind {
                DrafterKind::Dflash => drafter.layers * 2 * 8 * 128 * 4,
                DrafterKind::Mtp => drafter.layers * g.n_kv_head * g.head_dim * 2 * 2,
            }),
            None => None,
        }
    }

    /// Which shape of drafter this checkpoint ships, or `None` for one that
    /// ships none. Asked before a sidecar is opened — the two kinds are loaded
    /// and driven by different code, and the file itself is the authority once
    /// it is open (an operator's own `--draft <path>` need not be this one).
    pub const fn drafter_kind(self) -> Option<DrafterKind> {
        match &self.checkpoint().drafter {
            Some(drafter) => Some(drafter.kind),
            None => None,
        }
    }

    /// Bytes one prefix-cache snapshot costs, whatever position it covers.
    ///
    /// A snapshot deep-copies every DeltaNet layer's recurrent state — the conv
    /// window over the fused qkv stream and the delta state — both f32. Unlike
    /// the KV rows this is a fixed cost per snapshot, not a per-token one: the
    /// recurrent state is the same size at position 10 as at position 100000.
    ///
    /// A qwen4exp checkpoint adds its PLE layer's dilated conv window to that —
    /// small beside the DeltaNet state, but state all the same, and a snapshot
    /// that skipped it would resume mid-n-gram. (The PLE block's two-id token
    /// history is state too, and at 8 bytes is not worth a term here.)
    pub const fn snapshot_bytes(self) -> usize {
        let g = &self.checkpoint().geometry;
        let conv_dim = (2 * g.linear_k_heads + g.linear_v_heads) * g.linear_head_dim;
        // Saturating because an architecture with no recurrent layers declares
        // no conv either (`conv_kernel` 0, as `qwen3` does), and `0 - 1` on a
        // usize is a panic rather than the zero the whole term multiplies out
        // to.
        let conv = g.conv_kernel.saturating_sub(1) * conv_dim * 4;
        let delta = g.linear_v_heads * g.linear_head_dim * g.linear_head_dim * 4;
        let ple = match &g.extras {
            Some(extras) => extras.ple_layers * extras.ple_conv_cols * extras.ple_conv_width * 4,
            None => 0,
        };
        g.linear_layers * (conv + delta) + ple
    }
}

/// The CLI's short alias for the checkpoint — what `--model-size` takes and
/// what a log line names it by. The APIs speak [`Model::full_name`] instead.
impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Model::Qwen27B => "27b",
            Model::Qwen35BA3B => "35b",
            Model::Qwen3827B => "3.8-27b",
            Model::Qwen38FlashNext => "flash-next",
            Model::Qwen34B => "qwen3-4b",
            Model::Qwen34BInstruct2507 => "qwen3-4b-instruct-2507",
            Model::ZImageTurboEncoder => "zimage-turbo",
        })
    }
}

/// The CLI spelling: the short aliases above, plus each checkpoint's full name,
/// so anything a `/v1/models` listing shows also works as a `--model-size`. The
/// bare `27b`/`35b` keep meaning the 3.6 checkpoints they always did.
impl std::str::FromStr for Model {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(model) = Model::from_api_name(s) {
            return Ok(model);
        }
        match s.trim().to_ascii_lowercase().as_str() {
            "27" | "27b" => Ok(Model::Qwen27B),
            "35" | "35b" | "35b-a3b" => Ok(Model::Qwen35BA3B),
            "38" | "3.8" | "3.8-27b" => Ok(Model::Qwen3827B),
            "flash-next" | "3.8-flash-next" => Ok(Model::Qwen38FlashNext),
            "qwen3-4b" | "4b" => Ok(Model::Qwen34B),
            "qwen3-4b-instruct-2507" | "4b-instruct" => Ok(Model::Qwen34BInstruct2507),
            "zimage-turbo" | "z-image-turbo" => Ok(Model::ZImageTurboEncoder),
            other => Err(format!(
                "unknown model {other:?} (expected 27b, 35b, 3.8-27b, flash-next, qwen3-4b, \
                 qwen3-4b-instruct-2507 or zimage-turbo)"
            )),
        }
    }
}

/// Case-insensitive substring, ASCII — every checkpoint name is ASCII, and a
/// file name that spells one in another case is still spelling it.
fn contains_ignore_ascii_case(haystack: &str, needle: &str) -> bool {
    let haystack = haystack.to_ascii_lowercase();
    haystack.contains(&needle.to_ascii_lowercase())
}

/// The form the EXACT name comparison in [`Model::identify`] runs in:
/// ASCII-lowercased, with spaces folded onto the hyphens the canonical spellings
/// use. Every checkpoint name is ASCII, and a file that spells one in another
/// case is still spelling it.
///
/// The space fold is what lets one entry answer to both of its spellings, which
/// Qwen3.8-Flash-Next needs: the repo and the API name hyphenate it, but the
/// GGUF's own `general.name` is "Qwen3.8 Flash Next". Only the whole-name
/// comparison folds — the substring pass stays literal, so a name that merely
/// mentions a checkpoint in prose ("Qwen3.6 27B MyFinetune") is not that
/// checkpoint. See [`Model::identify`] for why the two passes differ.
fn hyphenate(s: &str) -> String {
    s.trim()
        .chars()
        .map(|c| {
            if c == ' ' {
                '-'
            } else {
                c.to_ascii_lowercase()
            }
        })
        .collect()
}

/// The checkpoint directory `path` names, resolved through symlinks: `path`
/// itself when it is a directory, otherwise its parent.
///
/// The directory is what gets canonicalized, never the file. A Hugging Face
/// cache snapshot is a tree of real directories holding symlinks into a shared
/// `blobs/` store, so canonicalizing `snapshots/<commit>/config.json` yields a
/// path in `blobs/` whose parent is the same for every checkpoint in the cache.
pub fn canonical_checkpoint_dir(path: &Path) -> Option<PathBuf> {
    let dir = if path.is_dir() { path } else { path.parent()? };
    std::fs::canonicalize(dir).ok()
}

/// The hub cache root, following the python tooling's precedence:
/// `$HF_HUB_CACHE` > `$HF_HOME/hub` > `~/.cache/huggingface/hub`.
/// (A superset of hf-hub's own `Cache::from_env`, which skips `HF_HUB_CACHE`.)
pub fn hub_cache_root() -> Option<PathBuf> {
    if let Some(cache) = std::env::var_os("HF_HUB_CACHE") {
        return Some(PathBuf::from(cache));
    }
    if let Some(home) = std::env::var_os("HF_HOME") {
        return Some(PathBuf::from(home).join("hub"));
    }
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".cache/huggingface/hub"))
}

fn cache() -> Option<Cache> {
    hub_cache_root().map(Cache::new)
}

/// The cached path of `file` in `repo` at the `main` ref, or `None` when it
/// is not present. Offline — never downloads.
pub fn cached_file(repo: &str, file: &str) -> Option<PathBuf> {
    cache()?.model(repo.to_string()).get(file)
}

/// The cached target GGUF for `model`, or `None`. Offline.
///
/// A split checkpoint is cached only when EVERY shard is: the path handed back
/// is the first shard's, and opening it would fail on the first missing sibling
/// — so a half-downloaded set has to read as a miss here, or the fetch that a
/// miss triggers would never run.
pub fn cached_model(model: Model) -> Option<PathBuf> {
    let mut shards = model
        .files()
        .iter()
        .map(|file| cached_file(model.repo(), file));
    let first = shards.next().flatten()?;
    shards.all(|shard| shard.is_some()).then_some(first)
}

/// The cached drafter sidecar for `model`, or `None` — which is also the answer
/// for a checkpoint that ships none at all. Offline.
pub fn cached_drafter(model: Model) -> Option<PathBuf> {
    cached_file(model.repo(), model.drafter_file()?)
}

/// Idempotent ensure: the cached path when present (no network), otherwise a
/// progress-bar download into the hub cache. Honors `$HF_ENDPOINT` and a
/// `huggingface-cli login` token, like the python tooling.
///
/// Concurrent-safe: hf-hub guards each blob with a non-blocking flock and
/// gives up after ~5s, so a second process starting mid-download would error.
/// Instead of surfacing that, wait and re-check the cache — when the other
/// process finishes, its snapshot is our answer.
pub fn ensure_file(repo: &str, file: &str) -> Result<PathBuf> {
    let cache = cache().context("cannot locate the Hugging Face cache (HOME is unset)")?;
    let mut builder = ApiBuilder::from_cache(cache).with_progress(true);
    if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
        builder = builder.with_endpoint(endpoint);
    }
    let api = builder
        .build()
        .context("building the Hugging Face client")?;
    let mut waiting = false;
    loop {
        match api.model(repo.to_string()).get(file) {
            Ok(path) => return Ok(path),
            Err(hf_hub::api::sync::ApiError::LockAcquisition(_)) => {
                if !waiting {
                    eprintln!("xwen: another process is downloading {repo}/{file}; waiting for it");
                    waiting = true;
                }
                std::thread::sleep(std::time::Duration::from_secs(3));
                if let Some(path) = cached_file(repo, file) {
                    return Ok(path);
                }
            }
            Err(err) => {
                return Err(err).with_context(|| {
                    format!("fetching {repo}/{file} into the Hugging Face cache")
                });
            }
        }
    }
}

/// The target GGUF for `model`, downloaded on first use — every shard of it for
/// a split checkpoint, since the loader needs the whole set beside the first
/// shard it is handed. The path returned is that first shard's.
pub fn ensure_model(model: Model) -> Result<PathBuf> {
    let mut first = None;
    for file in model.files() {
        let path = ensure_file(model.repo(), file)?;
        first.get_or_insert(path);
    }
    first.context("checkpoint table lists no file for this model")
}

/// The drafter sidecar for `model`, downloaded on first use, or `None` for a
/// checkpoint that ships none — which is not an error anywhere: it decodes
/// plain.
pub fn ensure_drafter(model: Model) -> Result<Option<PathBuf>> {
    match model.drafter_file() {
        Some(file) => ensure_file(model.repo(), file).map(Some),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Offline lookup against an explicit cache root — the same call
    /// `cached_file` makes, minus the env-derived root.
    fn cached_file_in(root: &Path, repo: &str, file: &str) -> Option<PathBuf> {
        Cache::new(root.to_path_buf())
            .model(repo.to_string())
            .get(file)
    }

    /// The cache figures, spelled out so a typo in the geometry table is a test
    /// failure rather than a wrong number in an operator's config file.
    ///
    /// Both are derived per the architecture cheat sheet in CLAUDE.md: KV rows
    /// are f16 over the full-attention layers only, and a snapshot is f32 conv
    /// window plus f32 delta state over the DeltaNet layers.
    #[test]
    fn the_cache_figures_match_each_checkpoints_geometry() {
        // 10 full-attn layers x 2 KV heads x 256 head_dim x (K and V) x 2 bytes.
        assert_eq!(Model::Qwen35BA3B.kv_bytes_per_token(), 20 * 1024);
        // 16 x 4 x 256 x 2 x 2.
        assert_eq!(Model::Qwen27B.kv_bytes_per_token(), 64 * 1024);

        // 30 DeltaNet layers x (3 x 8192 conv + 32 x 128 x 128 delta) x 4 bytes.
        assert_eq!(Model::Qwen35BA3B.snapshot_bytes(), 65_863_680);
        // 48 x (3 x 10240 conv + 48 x 128 x 128 delta) x 4 bytes.
        assert_eq!(Model::Qwen27B.snapshot_bytes(), 156_893_184);

        // The 27B is the expensive one on both axes among the qwen35 pair,
        // which is what the sizing prose that quotes a worst case has to be
        // read against.
        assert!(Model::Qwen27B.kv_bytes_per_token() > Model::Qwen35BA3B.kv_bytes_per_token());
        assert!(Model::Qwen27B.snapshot_bytes() > Model::Qwen35BA3B.snapshot_bytes());
    }

    /// The qwen4exp checkpoint's figures, which carry two terms no other
    /// checkpoint has.
    ///
    /// The indexer term is asserted against the function the ALLOCATION is
    /// sized with, not against a formula restated here. A restated formula is
    /// how this figure was wrong in the first place — it read the 4 query heads
    /// and the trunk's f16 off the table and got 1024 B/token/layer for a plane
    /// that is one MQA head at f32, four times over.
    #[test]
    fn the_qwen4exp_figures_count_its_indexer_and_ple_state() {
        use crate::qwen4exp::indexer::indexer_bytes_per_token;

        // 12 full-attn layers x (2 KV heads x 256 head_dim x K-and-V x 2 bytes
        // f16 = 2048, plus the indexer's one 128-wide f32 key row = 512, plus
        // that row's share of the block-key plane, 512 / ratio 4 = 128).
        assert_eq!(Model::Qwen38FlashNext.kv_bytes_per_token(), 12 * 2688);
        let trunk_only = 12 * 2 * 256 * 2 * 2;
        assert_eq!(
            Model::Qwen38FlashNext.kv_bytes_per_token(),
            trunk_only + 12 * indexer_bytes_per_token(128, 4)
        );
        // Over a quarter more than the trunk's rows, not a rounding error:
        // dropping it would under-size a context budget by almost a quarter.
        assert_eq!(indexer_bytes_per_token(128, 4), 640);
        assert_eq!(
            trunk_only * 21 / 16,
            Model::Qwen38FlashNext.kv_bytes_per_token()
        );

        // 36 DeltaNet layers x (3 x 10240 conv + 48 x 128 x 128 delta) x 4
        // bytes, plus one PLE layer's 9 x 10240 conv window at f32.
        let delta_net = 36 * (3 * 10240 + 48 * 128 * 128) * 4;
        assert_eq!(
            Model::Qwen38FlashNext.snapshot_bytes(),
            delta_net + 9 * 10240 * 4
        );
        assert_eq!(Model::Qwen38FlashNext.snapshot_bytes(), 118_038_528);
    }

    /// The two remaining capability gates on the qwen4exp checkpoint, and the
    /// fact that they are gates rather than the table's shape: every other
    /// checkpoint answers yes to both, so a predicate that stopped consulting
    /// `self` would still pass the rest of the suite.
    ///
    /// Each is a different limitation with a different lifetime, which is why
    /// they are two predicates and not one "experimental" flag: `auto_fetch` is
    /// about a 111 GB download and would stay false even after everything else
    /// is finished, while `supports_drafting` is D6's missing verify seam.
    /// `servable` used to be a third and no longer is — the cache images carry
    /// the QSA raw keys and the PLE state as of 2026-08-30, so `xwen serve` and
    /// `xwen batch` run this checkpoint like any other.
    #[test]
    fn the_qwen4exp_checkpoint_is_gated_out_of_fetching_and_drafting() {
        assert!(!Model::Qwen38FlashNext.auto_fetch());
        assert!(!Model::Qwen38FlashNext.supports_drafting());
        assert!(Model::Qwen38FlashNext.servable());

        for model in [Model::Qwen27B, Model::Qwen35BA3B, Model::Qwen3827B] {
            assert!(model.auto_fetch(), "{model}");
            assert!(model.supports_drafting(), "{model}");
        }
        // Every GGUF checkpoint's state has a cache image, so serve and batch
        // run all four. The qwen3 entries are the ones this seam now says no
        // to, and they are covered in
        // `the_qwen3_checkpoints_are_registered_but_not_runnable_yet`.
        for model in MODELS.into_iter().filter(|m| !m.is_safetensors()) {
            assert!(model.servable(), "{model}");
        }

        // The message names the checkpoint and points somewhere the operator can
        // actually go — a message that only says "no" is a dead end.
        let undraftable = Model::Qwen38FlashNext.no_drafting_message();
        assert!(undraftable.contains("Qwen3.8-Flash-Next"), "{undraftable}");
        assert!(undraftable.contains("--no-draft"), "{undraftable}");
    }

    /// The drafter cache figure, pinned for the same reason as the target's: it is
    /// quoted in the `serve --init` template and in `--draft-ctx`'s help, where a
    /// wrong number becomes an operator's wrong memory budget.
    ///
    /// Within the DFlash kind the two sidecars differ only in layer count — both
    /// are 32 Q / 8 KV heads at head_dim 128 — so that is the only factor that
    /// varies. K and V are stored f32, not f16 like the target's: the drafter is
    /// tiny and the exactness keeps its export/import round trip free of
    /// rounding. The MTP head is the other kind entirely and is pinned in
    /// `the_mtp_head_costs_an_order_of_magnitude_less_to_cache`.
    #[test]
    fn the_drafter_cache_figure_follows_each_sidecars_layer_count() {
        // 5 drafter layers x (K and V) x 8 KV heads x 128 head_dim x 4 bytes.
        assert_eq!(Model::Qwen27B.draft_kv_bytes_per_token(), Some(40 * 1024));
        // 6 x 2 x 8 x 128 x 4.
        assert_eq!(
            Model::Qwen35BA3B.draft_kv_bytes_per_token(),
            Some(48 * 1024)
        );

        // The drafter ordering is the OPPOSITE of the target's: the 27B has the
        // bigger KV cache and the smaller drafter cache, so neither model is the
        // worst case on both and a sizing estimate has to name which it means.
        assert!(
            Model::Qwen27B.draft_kv_bytes_per_token()
                < Model::Qwen35BA3B.draft_kv_bytes_per_token()
        );
    }

    /// The fitted drafting floors, pinned so changing either one is a deliberate
    /// act with a sweep behind it — these are the values `--draft-p-min` and the
    /// serve config fall back to, and they came out of the 2026-08-08 retune.
    #[test]
    fn the_draft_depth_default_is_per_kind() {
        // A block drafter proposes its whole block in one forward, so the ask is
        // the structural ceiling both shipped sidecars have.
        assert_eq!(Model::Qwen27B.draft_max_default(), Some(15));
        assert_eq!(Model::Qwen35BA3B.draft_max_default(), Some(15));
        // A chain pays a forward per step, so it is asked for far less. Fitted
        // by the Stage C sweep (2026-08-15), which moved it off llama.cpp's 3
        // and bracketed the optimum on both sides.
        assert_eq!(Model::Qwen3827B.draft_max_default(), Some(4));
    }

    /// What a zero-flag run does per checkpoint, pinned because it is a policy
    /// and not a fitted value: the 35B-A3B arm went false on 2026-09-06 after
    /// its drafted arm read below plain at every length, and it goes back to
    /// true only on a retune that reads above plain again. The other two arms
    /// are the measured wins they always were.
    ///
    /// Deliberately independent of `drafter_kind`: the 35B-A3B still SHIPS a
    /// sidecar and still attaches it on `--draft official`. What changed is
    /// what silence means.
    #[test]
    fn drafting_defaults_on_per_checkpoint() {
        assert!(Model::Qwen27B.draft_default_on());
        assert!(Model::Qwen3827B.draft_default_on());
        assert!(!Model::Qwen35BA3B.draft_default_on());
        // Nothing to default on: no verify seam, and no sidecar.
        assert!(!Model::Qwen38FlashNext.draft_default_on());

        // The 35B-A3B is the one checkpoint where the two questions part ways,
        // which is the whole reason this accessor exists rather than being read
        // off the registry.
        assert!(Model::Qwen35BA3B.drafter_kind().is_some());
        assert!(Model::Qwen35BA3B.supports_drafting());

        // A checkpoint nothing can draft for cannot default to drafting.
        for model in MODELS {
            if !model.supports_drafting() {
                assert!(!model.draft_default_on(), "{model:?}");
            }
        }
    }

    #[test]
    fn the_drafting_floor_is_per_checkpoint() {
        assert_eq!(Model::Qwen27B.draft_p_min_default(), Some(0.5));
        assert_eq!(Model::Qwen35BA3B.draft_p_min_default(), Some(0.3));
        // Fitted by the Stage C sweep (2026-08-15), crossed with depth, which
        // moved it off llama.cpp's 0.5 — by a small margin on this axis; see
        // `draft_p_min_default` for how weakly this arm is held.
        assert_eq!(Model::Qwen3827B.draft_p_min_default(), Some(0.7));
    }

    /// The MTP head caches an order of magnitude cheaper than either block
    /// drafter, which is what makes `--draft-ctx` a different decision on this
    /// checkpoint: the same context budget buys ten times the drafting depth.
    ///
    /// The figure is quoted wherever an operator sizes memory, so it is pinned
    /// with its arithmetic rather than as a number somebody would have to
    /// re-derive to check.
    #[test]
    fn the_mtp_head_costs_an_order_of_magnitude_less_to_cache() {
        // 1 head layer x (K and V) x 4 KV heads x 256 head_dim x 2 bytes (f16,
        // the trunk's own full-attention cache dtype, which the head reuses).
        assert_eq!(Model::Qwen3827B.draft_kv_bytes_per_token(), Some(4 * 1024));
        assert!(
            Model::Qwen3827B.draft_kv_bytes_per_token() < Model::Qwen27B.draft_kv_bytes_per_token()
        );
    }

    /// Everything speculation needs travels together, so a checkpoint either
    /// offers all of it or none — and both answers are live: the three Qwen 3.6
    /// and 3.8 dense/MoE checkpoints each name their own kind, file and fitted
    /// floor, while Qwen3.8-Flash-Next answers `None` to every one of them.
    /// A checkpoint that answered some and not others would have a drafter
    /// nothing could size or fetch, which is what this pins against.
    #[test]
    fn every_checkpoint_answers_the_whole_drafter_question() {
        for model in MODELS {
            let Some(kind) = model.drafter_kind() else {
                assert!(model.drafter_file().is_none(), "{model:?} names a file");
                assert!(model.drafter_size().is_none(), "{model:?} names a size");
                assert!(
                    model.draft_kv_bytes_per_token().is_none(),
                    "{model:?} sizes a drafter cache"
                );
                assert!(
                    model.draft_p_min_default().is_none(),
                    "{model:?} has a fitted floor"
                );
                assert!(
                    model.draft_max_default().is_none(),
                    "{model:?} has a draft depth"
                );
                continue;
            };
            assert!(model.drafter_file().is_some(), "{model:?} names no file");
            assert!(model.drafter_size().is_some(), "{model:?} names no size");
            assert!(
                model.draft_kv_bytes_per_token().is_some(),
                "{model:?} sizes no drafter cache"
            );
            assert!(
                model.draft_p_min_default().is_some(),
                "{model:?} has no fitted floor"
            );
            // The file name says which kind it is, and it had better be the kind
            // the checkpoint claims: everything that loads one branches on this.
            let file = model.drafter_file().unwrap();
            let expected = match kind {
                DrafterKind::Dflash => "dflash-",
                DrafterKind::Mtp => "mtp-",
            };
            assert!(
                file.starts_with(expected),
                "{model:?} claims {kind:?} but its sidecar is named {file}"
            );
        }
        assert_eq!(Model::Qwen3827B.drafter_kind(), Some(DrafterKind::Mtp));
        assert_eq!(Model::Qwen27B.drafter_kind(), Some(DrafterKind::Dflash));
        // Qwen3.8-Flash-Next decodes plain: the release's MTP head is not
        // ported, and every drafter accessor's `Option` shape — written for a
        // checkpoint that ships none, and unexercised until now — is why that
        // costs nothing downstream.
        assert_eq!(Model::Qwen38FlashNext.drafter_kind(), None);
    }

    /// The one card value that is not shared across the checkpoints, and the
    /// only sampling default that has to be resolved with a checkpoint in hand.
    #[test]
    fn the_presence_penalty_is_per_checkpoint_and_per_mode() {
        let qwen36 = || MODELS.into_iter().filter(|m| !m.is_safetensors());
        // Every Qwen 3.6/3.8 card asks for 1.5 in non-thinking mode.
        for model in qwen36() {
            assert_eq!(model.recommended_presence_penalty(false), 1.5, "{model:?}");
        }
        // Thinking mode is where they part: only the 35B-A3B card carries one.
        assert_eq!(Model::Qwen35BA3B.recommended_presence_penalty(true), 1.5);
        for model in qwen36() {
            if model != Model::Qwen35BA3B {
                assert_eq!(model.recommended_presence_penalty(true), 0.0, "{model:?}");
            }
        }
        // The Qwen3 cards recommend none in either mode: they list
        // `presence_penalty` as an optional 0-2 knob against repetition, and
        // applying 1.5 unasked would be this runtime's opinion, not the card's.
        for model in MODELS.into_iter().filter(|m| m.is_safetensors()) {
            for thinking in [true, false] {
                assert_eq!(
                    model.recommended_presence_penalty(thinking),
                    0.0,
                    "{model:?} thinking={thinking}"
                );
            }
        }
    }

    /// The split checkpoint's hub coordinates: four shards named the way
    /// `gguf::open` expects to find their siblings, with `file()` handing back
    /// the first — the only one the loader is ever opened on.
    #[test]
    fn the_split_checkpoint_names_every_shard_in_order() {
        let files = Model::Qwen38FlashNext.files();
        assert_eq!(files.len(), 4);
        assert_eq!(Model::Qwen38FlashNext.file(), files[0]);
        for (i, file) in files.iter().enumerate() {
            assert!(
                file.ends_with(&format!("-{:05}-of-00004.gguf", i + 1)),
                "shard {i} is named {file}"
            );
        }
        // Every other GGUF checkpoint is one file, and `file()` is that file.
        // A safetensors set is many files by nature and is covered by
        // `the_qwen3_entries_name_every_file_of_their_set`.
        for model in MODELS {
            if model != Model::Qwen38FlashNext && !model.is_safetensors() {
                assert_eq!(model.files(), &[model.file()], "{model:?}");
            }
        }
    }

    /// A safetensors checkpoint is a DIRECTORY, and what the registry lists is
    /// every file that has to be in it: the config the loader opens, the shard
    /// index, the three shards and the tokenizer. `file()` is the config,
    /// deliberately — it is the entry `cached_model` and `ensure_model` hand
    /// back, and `CheckpointSource::open` resolves the directory from it.
    #[test]
    fn the_qwen3_entries_name_every_file_of_their_set() {
        for model in MODELS.into_iter().filter(|m| m.is_safetensors()) {
            let files = model.files();
            assert_eq!(files.len(), 6, "{model:?}");
            assert!(files[0].ends_with("config.json"), "{model:?}: {}", files[0]);
            assert_eq!(model.file(), files[0], "{model:?}");
            assert_eq!(
                files.iter().filter(|f| f.ends_with(".safetensors")).count(),
                3,
                "{model:?}"
            );
            assert!(
                files.contains(&model.safetensors_tokenizer().unwrap()),
                "{model:?} does not list the tokenizer it names"
            );
            // Every path is repo-relative, never absolute and never a parent
            // escape: `cached_file` joins them onto a snapshot directory.
            for file in files {
                assert!(
                    !file.starts_with('/') && !file.contains(".."),
                    "{model:?}: {file}"
                );
            }
            // The two JSON files the format names are files the entry lists.
            let Format::SafeTensors { config, index, .. } = &model.checkpoint().format else {
                panic!("{model:?} is not a safetensors entry");
            };
            assert_eq!(*config, files[0], "{model:?}");
            assert!(files.contains(index), "{model:?}");
        }

        // The encoder is the one whose files live in subdirectories, which is
        // why the paths are relative to the REPO rather than to the checkpoint.
        assert_eq!(Model::ZImageTurboEncoder.file(), "text_encoder/config.json");
        assert_eq!(
            Model::ZImageTurboEncoder.safetensors_tokenizer(),
            Some("tokenizer/tokenizer.json")
        );
        // And a GGUF checkpoint names no tokenizer at all: its vocabulary is in
        // the file.
        for model in MODELS.into_iter().filter(|m| !m.is_safetensors()) {
            assert_eq!(model.safetensors_tokenizer(), None, "{model:?}");
            assert!(
                model.safetensors_allowed_zero_runs().is_empty(),
                "{model:?}"
            );
            assert_eq!(model.encoder_spec(), None, "{model:?}");
            assert_eq!(model.safetensors_rope_theta(), None, "{model:?}");
        }
    }

    /// The three qwen3 entries are registered so that names, sizes and cache
    /// figures exist for them — and gated off every surface that would try to
    /// run one, because the layer stack that would run them does not exist.
    ///
    /// Two of the three gates lift when it does. The encoder's `servable` does
    /// not: it is an encode-only entry over weights with a corrupt layer, and
    /// generating from it would evaluate exactly that layer.
    #[test]
    fn the_qwen3_checkpoints_are_registered_but_not_runnable_yet() {
        for model in [
            Model::Qwen34B,
            Model::Qwen34BInstruct2507,
            Model::ZImageTurboEncoder,
        ] {
            assert!(model.is_safetensors(), "{model:?}");
            assert_eq!(model.arch(), Arch::Qwen3, "{model:?}");
            assert!(!model.servable(), "{model:?}");
            assert!(!model.auto_fetch(), "{model:?}");
            assert!(!model.supports_drafting(), "{model:?}");
            assert!(!model.draft_default_on(), "{model:?}");
            assert_eq!(model.drafter_kind(), None, "{model:?}");
            assert_eq!(model.vocab_family(), VocabFamily::Qwen3, "{model:?}");
            // Not selectable on the wire while they cannot run, whatever the
            // cache holds.
            assert!(!crate::serve::checkpoint_selectable(model), "{model:?}");
        }
        // Nothing else moved family: the four GGUF checkpoints still speak the
        // 3.6 vocabulary, which is what makes this a checkpoint property worth
        // keying a tokenizer on rather than a constant.
        for model in MODELS.into_iter().filter(|m| !m.is_safetensors()) {
            assert_eq!(model.vocab_family(), VocabFamily::Qwen36, "{model:?}");
        }

        // The encoder is the entry with an encode spec, and its ceiling is its
        // own layer: 35's MLP is the corrupt plane, so nothing deeper can be
        // asked for.
        let spec = Model::ZImageTurboEncoder.encoder_spec().unwrap();
        assert_eq!(spec.layer, 35);
        assert_eq!(spec.max_layer, 35);
        assert_eq!(spec.max_tokens, 512);
        assert_eq!(Model::Qwen34B.encoder_spec(), None);
        assert_eq!(Model::Qwen34BInstruct2507.encoder_spec(), None);

        // And the entry that allowlists the two zero-filled planes is the
        // encoder alone: the same tensors in `Qwen/Qwen3-4B` have no zero runs,
        // so an allowlist there would hide a real corruption.
        assert_eq!(
            Model::ZImageTurboEncoder.safetensors_allowed_zero_runs(),
            [
                "model.layers.35.mlp.up_proj.weight",
                "model.layers.35.mlp.down_proj.weight"
            ]
        );
        assert!(Model::Qwen34B.safetensors_allowed_zero_runs().is_empty());
        assert!(
            Model::Qwen34BInstruct2507
                .safetensors_allowed_zero_runs()
                .is_empty()
        );
    }

    /// The qwen3 cache figures. All three checkpoints share one geometry — the
    /// releases differ only in `rope_theta` and context length — and its
    /// snapshot cost is genuinely zero: every layer is full attention, so the
    /// whole conversation is in the KV rows and there is no recurrent state for
    /// a snapshot to copy.
    #[test]
    fn the_qwen3_geometry_is_all_attention_and_no_recurrent_state() {
        // 36 layers x 8 KV heads x 128 head_dim x (K and V) x 2 bytes f16.
        assert_eq!(Model::Qwen34B.kv_bytes_per_token(), 144 * 1024);
        for model in MODELS.into_iter().filter(|m| m.is_safetensors()) {
            assert_eq!(
                model.kv_bytes_per_token(),
                Model::Qwen34B.kv_bytes_per_token(),
                "{model:?}"
            );
            // No DeltaNet layers, so nothing to snapshot — and the formula has
            // to reach that answer rather than underflow on a conv kernel of 0.
            assert_eq!(model.snapshot_bytes(), 0, "{model:?}");
        }
        // And the counter-intuitive half of it, worth pinning because a sizing
        // estimate that assumed "small model, small cache" would be wrong by
        // more than a factor of two: the 4B model's KV rows cost MORE per token
        // than the 27B's. Being a hybrid is what makes the 27B cheap — only 16
        // of its 64 layers hold KV at all — while every one of these 36 does.
        assert_eq!(Model::Qwen27B.kv_bytes_per_token(), 64 * 1024);
        assert!(Model::Qwen34B.kv_bytes_per_token() > Model::Qwen27B.kv_bytes_per_token());
        // Snapshots go the other way, and to the limit: a 27B prefix snapshot
        // is 150 MB of recurrent state and a qwen3 one is nothing at all.
        assert!(Model::Qwen27B.snapshot_bytes() > 0);
    }

    /// A safetensors directory has no name to read, so identity is provenance:
    /// the directory the hub cache holds for an entry IS that entry, and every
    /// other directory identifies as nothing and runs under its own name.
    ///
    /// The `Assumed` fallback is then the one config value that separates the
    /// releases, `rope_theta` — which is also the cross-check `--model-size`
    /// gets in place of the name one.
    #[test]
    fn a_safetensors_directory_identifies_by_provenance_then_by_rope_theta() {
        use crate::config::{Identity, RopeKind};

        // A directory nobody's cache knows is nobody's checkpoint.
        let scratch = scratch("identify_dir");
        assert_eq!(
            Model::identify(Arch::Qwen3, None, Some(&scratch)),
            None,
            "an arbitrary directory must not claim an official checkpoint"
        );
        // Not even one whose NAME spells a checkpoint: a name is exactly what a
        // directory cannot be trusted about.
        let named = scratch.join("Qwen3-4B");
        std::fs::create_dir_all(&named).unwrap();
        assert_eq!(Model::identify(Arch::Qwen3, None, Some(&named)), None);
        // A general.name is not consulted on this architecture either — there
        // is nowhere in a safetensors set for one to come from.
        assert_eq!(
            Model::identify(Arch::Qwen3, Some("Qwen3-4B"), Some(&named)),
            None
        );

        // The release cross-check. `--model-size` naming a release whose theta
        // the directory does not carry is refused, both ways round.
        let base = qwen3_config_at(1e6);
        let instruct = qwen3_config_at(5e6);
        assert!(
            instruct
                .identify(&scratch, Some(Model::Qwen34B), "--model-size")
                .is_err(),
            "a 5e6 directory is not the base release"
        );
        assert!(
            base.identify(&scratch, Some(Model::Qwen34BInstruct2507), "--model-size")
                .is_err(),
            "a 1e6 directory is not Instruct-2507"
        );
        // The encoder shares the base config exactly, so naming it against an
        // unidentified 1e6 directory is allowed — that is the case where the
        // flag settles what the files cannot.
        assert_eq!(
            base.identify(&scratch, Some(Model::ZImageTurboEncoder), "--model-size")
                .unwrap(),
            Identity::Official(Model::ZImageTurboEncoder)
        );
        assert_eq!(
            base.identify(&scratch, Some(Model::Qwen34B), "--model-size")
                .unwrap(),
            Identity::Official(Model::Qwen34B)
        );

        // With no flag, theta alone decides, and the base model wins the 1e6
        // tie it shares with the encoder.
        assert_eq!(
            base.identify(&scratch, None, "--model-size").unwrap(),
            Identity::Assumed(Model::Qwen34B)
        );
        assert_eq!(
            instruct.identify(&scratch, None, "--model-size").unwrap(),
            Identity::Assumed(Model::Qwen34BInstruct2507)
        );
        assert_eq!(Model::by_safetensors_rope_theta(1e6), Some(Model::Qwen34B));
        assert_eq!(
            Model::by_safetensors_rope_theta(5e6),
            Some(Model::Qwen34BInstruct2507)
        );
        // A theta no release uses names none of them, and the architecture's
        // registry entry is what such a file falls back to.
        assert_eq!(Model::by_safetensors_rope_theta(1e4), None);
        assert_eq!(
            qwen3_config_at(1e4)
                .identify(&scratch, None, "--model-size")
                .unwrap(),
            Identity::Assumed(Model::Qwen34B)
        );

        // The architecture still narrows first: a qwen3 selection against a
        // GGUF architecture is the same error it has always been.
        assert!(
            base.identify(&scratch, Some(Model::Qwen27B), "--model-size")
                .is_err()
        );
        assert!(matches!(base.rope(), RopeKind::Plain { n_rot: 128, .. }));

        std::fs::remove_dir_all(&scratch).unwrap();
    }

    /// The provenance half of the rule, against the real cache. Self-skips when
    /// the checkpoint is not downloaded — there is nothing to have provenance
    /// from.
    #[test]
    fn the_cached_snapshot_of_an_entry_identifies_as_that_entry() {
        for model in MODELS.into_iter().filter(|m| m.is_safetensors()) {
            let Some(config) = cached_model(model) else {
                eprintln!("skipping {model}: not in the Hugging Face cache");
                continue;
            };
            let dir = config.parent().unwrap();
            // Both spellings the callers have: the directory, and the
            // `config.json` inside it that `ensure_model` hands back.
            assert_eq!(
                Model::identify(Arch::Qwen3, None, Some(dir)),
                Some(model),
                "{model:?} at {}",
                dir.display()
            );
            assert_eq!(
                Model::identify(Arch::Qwen3, None, Some(&config)),
                Some(model),
                "{model:?} via its config.json"
            );
        }
    }

    /// A synthetic `XwenConfig` on the qwen3 architecture at a given rope base.
    /// Built through `from_qwen3` from a hand-written `config.json`, so it is
    /// the real constructor under test and not a struct literal that could
    /// drift from it.
    fn qwen3_config_at(theta: f32) -> crate::XwenConfig {
        let json = format!(
            r#"{{"model_type":"qwen3","hidden_size":2560,"intermediate_size":9728,
               "num_hidden_layers":36,"num_attention_heads":32,"num_key_value_heads":8,
               "head_dim":128,"rms_norm_eps":1e-6,"rope_theta":{theta},
               "max_position_embeddings":40960,"vocab_size":151936,
               "tie_word_embeddings":true,"attention_bias":false,
               "use_sliding_window":false,"hidden_act":"silu","rope_scaling":null}}"#
        );
        let cfg = crate::qwen3::Qwen3Config::from_json_bytes(json.as_bytes()).unwrap();
        crate::XwenConfig::from_qwen3(&cfg, None)
    }

    /// The two dense checkpoints share a graph and a geometry — 3.8's config is
    /// byte-identical to 3.6's — so every cache figure is shared too, and only
    /// the hub coordinates and the drafter tell them apart.
    #[test]
    fn the_two_dense_checkpoints_differ_only_off_the_graph() {
        assert_eq!(
            Model::Qwen3827B.kv_bytes_per_token(),
            Model::Qwen27B.kv_bytes_per_token()
        );
        assert_eq!(
            Model::Qwen3827B.snapshot_bytes(),
            Model::Qwen27B.snapshot_bytes()
        );
        assert_eq!(Model::Qwen3827B.arch(), Model::Qwen27B.arch());
        assert_ne!(Model::Qwen3827B.repo(), Model::Qwen27B.repo());
    }

    /// The API's model vocabulary is the full names and nothing else: the CLI's
    /// short aliases are refused on the wire, where one checkpoint answering to
    /// several ids is what listing them all duplicated in the first place.
    #[test]
    fn the_api_speaks_full_names_only() {
        assert_eq!(Model::from_api_name("Qwen3.6-27B"), Some(Model::Qwen27B));
        assert_eq!(
            Model::from_api_name("qwen3.6-35b-a3b"),
            Some(Model::Qwen35BA3B)
        );
        assert_eq!(Model::from_api_name("QWEN3.8-27B"), Some(Model::Qwen3827B));
        assert_eq!(
            Model::from_api_name("qwen3.8-flash-next"),
            Some(Model::Qwen38FlashNext)
        );
        assert_eq!(Model::from_api_name("35b"), None);
        assert_eq!(Model::from_api_name("3.8"), None);
        assert_eq!(Model::from_api_name("flash-next"), None);
        // The API vocabulary is the hyphenated names exactly: the space
        // spelling the qwen4exp GGUF carries identifies a FILE (`identify`),
        // but it is not a second id this server answers to.
        assert_eq!(Model::from_api_name("Qwen3.8 Flash Next"), None);
        assert_eq!(Model::from_api_name(""), None);

        // Every full name is distinct and round-trips.
        for model in MODELS {
            assert_eq!(Model::from_api_name(model.full_name()), Some(model));
        }
    }

    /// What a GGUF says about itself decides which checkpoint it is: a file that
    /// names none identifies as nothing and the caller falls back with a warning
    /// rather than reporting someone's conversion under a checkpoint's name.
    #[test]
    fn a_gguf_identifies_itself_by_name_then_by_file_name() {
        use std::path::Path;

        // `general.name` outranks the file name, which the blessed files make
        // moot and a renamed copy does not.
        let moe = Path::new("some-other-name.gguf");
        assert_eq!(
            Model::identify(Arch::Moe, Some("Qwen3.6-35B-A3B"), Some(moe)),
            Some(Model::Qwen35BA3B)
        );
        // A conversion onto the MoE graph that names no checkpoint is not one:
        // the arch is shared by anything converted to it, so it identifies
        // nothing on its own and the file keeps reporting under its own name.
        assert_eq!(Model::identify(Arch::Moe, None, None), None);
        assert_eq!(
            Model::identify(Arch::Moe, Some("my-finetune"), Some(moe)),
            None
        );
        // The arch narrows the candidates before any name is read, so a dense
        // file claiming the MoE checkpoint's name matches nothing: a name cannot
        // make a file the other graph, and the dense names are not in it.
        assert_eq!(
            Model::identify(Arch::Dense, Some("Qwen3.6-35B-A3B"), None),
            None
        );

        // A whole checkpoint name is required — a bare release series is not one.
        // The MoE case is the sharp one: that architecture has a single
        // candidate, so a stray "3.6" needs no ambiguity to hand someone's
        // finetune an official checkpoint's identity.
        for stray in ["MyMoE-3.6", "3.6", "my-3.6-merge", "Qwen3.6"] {
            assert_eq!(
                Model::identify(Arch::Moe, Some(stray), None),
                None,
                "{stray}"
            );
        }
        for stray in ["Qwen3.6-14B", "some-3.8-thing", "3.8-27b"] {
            assert_eq!(
                Model::identify(Arch::Dense, Some(stray), None),
                None,
                "{stray}"
            );
        }
        // The real MoE name still identifies, in any case.
        assert_eq!(
            Model::identify(Arch::Moe, Some("qwen3.6-35b-a3b (Q8_0)"), None),
            Some(Model::Qwen35BA3B)
        );

        // general.name as both blessed dense files carry it.
        assert_eq!(
            Model::identify(Arch::Dense, Some("Qwen3.6-27B"), None),
            Some(Model::Qwen27B)
        );
        assert_eq!(
            Model::identify(Arch::Dense, Some("Qwen3.8-27B"), None),
            Some(Model::Qwen3827B)
        );
        // A re-quantized conversion that kept the whole name inside its own.
        assert_eq!(
            Model::identify(Arch::Dense, Some("Qwen3.6-27B-Instruct"), None),
            Some(Model::Qwen27B)
        );
        assert_eq!(
            Model::identify(Arch::Dense, Some("Qwen3.8-27B-Instruct (imatrix)"), None),
            Some(Model::Qwen3827B)
        );
        // No general.name: the file name answers instead — but only when it
        // spells a whole checkpoint name, quant suffix and directories aside.
        for (path, expected) in [
            ("/models/Qwen3.8-27B-Q8_0.gguf", Some(Model::Qwen3827B)),
            ("/models/Qwen3.6-27B-Q4_K_M.gguf", Some(Model::Qwen27B)),
            ("qwen3.8-27b.gguf", Some(Model::Qwen3827B)),
            // A release digit is NOT a checkpoint name: this is somebody's 14B
            // finetune, and calling it the official 27B would serve unchecked
            // weights under an official name.
            ("My-Qwen3.6-14B-finetune.gguf", None),
            ("qwen-3.8-something.gguf", None),
        ] {
            assert_eq!(
                Model::identify(Arch::Dense, None, Some(Path::new(path))),
                expected,
                "{path}"
            );
        }
        // The MoE file name identifies its own checkpoint, and never a dense one.
        assert_eq!(
            Model::identify(
                Arch::Moe,
                None,
                Some(Path::new("Qwen3.6-35B-A3B-Q4_K_M.gguf"))
            ),
            Some(Model::Qwen35BA3B)
        );

        // A name that spells two checkpoints identifies as neither: an ambiguous
        // name is where guessing by table order is worst.
        assert_eq!(
            Model::identify(
                Arch::Dense,
                None,
                Some(Path::new("Qwen3.6-27B-vs-Qwen3.8-27B.gguf"))
            ),
            None
        );
        assert_eq!(
            Model::identify(Arch::Dense, Some("Qwen3.6 and Qwen3.8 merge"), None),
            None
        );
        // A dense file that names no release is not guessed at.
        assert_eq!(
            Model::identify(
                Arch::Dense,
                Some("mymodel"),
                Some(Path::new("mymodel.gguf"))
            ),
            None
        );
    }

    /// Qwen3.8-Flash-Next is the one checkpoint whose file spells its name
    /// differently from its repo and its API id: the GGUF's `general.name` is
    /// "Qwen3.8 Flash Next" with spaces. Both spellings identify it — names are
    /// compared with spaces folded onto hyphens — and the fold buys no
    /// looseness anywhere else: a whole checkpoint name must still appear.
    #[test]
    fn the_qwen4exp_checkpoint_identifies_under_both_of_its_spellings() {
        use std::path::Path;

        for name in [
            "Qwen3.8 Flash Next",
            "Qwen3.8-Flash-Next",
            "qwen3.8 flash next",
            // Unsloth's own file names, as the quant folder spells them.
            "Qwen3.8-Flash-Next-UD-Q4_K_XL",
        ] {
            assert_eq!(
                Model::identify(Arch::Qwen4Exp, Some(name), None),
                Some(Model::Qwen38FlashNext),
                "{name}"
            );
        }
        // The shipped first shard, by file name alone.
        assert_eq!(
            Model::identify(
                Arch::Qwen4Exp,
                None,
                Some(Path::new(
                    "/hub/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
                ))
            ),
            Some(Model::Qwen38FlashNext)
        );

        // Somebody else's qwen4exp conversion is not this checkpoint, however
        // much of the release series or the word "Flash" its name borrows.
        // qwen4exp has a single candidate, so — as with the MoE graph — a
        // stray fragment needs no ambiguity to hand a finetune an official
        // identity.
        for stray in [
            "3.8",
            "Qwen3.8",
            "Flash Next",
            "flash next",
            "My-Qwen3.8-Flash-14B",
            "Qwen4-Next",
        ] {
            assert_eq!(
                Model::identify(Arch::Qwen4Exp, Some(stray), None),
                None,
                "{stray}"
            );
        }
        // A re-quantizer that kept the whole name inside its own still
        // identifies, which is what the substring pass exists for — spelled the
        // way the release spells it, with hyphens.
        assert_eq!(
            Model::identify(Arch::Qwen4Exp, Some("Qwen3.8-Flash-Next (imatrix)"), None),
            Some(Model::Qwen38FlashNext)
        );
        // The SPACE spelling plus a suffix is not the same thing: only the
        // whole-name comparison folds spaces, so this reads as prose mentioning
        // the checkpoint rather than as the checkpoint. Deliberate — the
        // alternative makes "Qwen3.6 27B MyFinetune" the official 27B.
        assert_eq!(
            Model::identify(Arch::Qwen4Exp, Some("Qwen3.8 Flash Next (imatrix)"), None),
            None
        );

        // The arch narrows first, so the qwen4exp name never lands on a dense
        // or MoE file, and the dense names never land on this one.
        assert_eq!(
            Model::identify(Arch::Dense, Some("Qwen3.8 Flash Next"), None),
            None
        );
        assert_eq!(
            Model::identify(Arch::Qwen4Exp, Some("Qwen3.8-27B"), None),
            None
        );
        // A qwen4exp file that names nothing identifies as nothing; the caller
        // falls back to `Arch::model` with a warning.
        assert_eq!(Model::identify(Arch::Qwen4Exp, Some("mymodel"), None), None);
    }

    /// Only the WHOLE-name comparison folds spaces onto hyphens. The substring
    /// pass stays literal, so a name that merely mentions the checkpoint it was
    /// derived from — spelled the way prose spells it, with spaces — is not that
    /// checkpoint. Folding both passes would hand every "<official> <suffix>"
    /// finetune an official identity, which is the failure the whole-name
    /// requirement exists to prevent.
    #[test]
    fn the_space_fold_applies_to_the_whole_name_comparison_only() {
        use std::path::Path;

        // Space-spelled prose around a checkpoint name: a mention, not a claim.
        for stray in [
            "Qwen3.6 27B MyFinetune",
            "Qwen3.8 27B Coder",
            "distilled from Qwen3.6 27B",
        ] {
            assert_eq!(
                Model::identify(Arch::Dense, Some(stray), None),
                None,
                "{stray}"
            );
            let file = format!("{stray}.gguf");
            assert_eq!(
                Model::identify(Arch::Dense, None, Some(Path::new(&file))),
                None,
                "{stray}"
            );
        }

        // The fold still does its one job: the qwen4exp GGUF's space-spelled
        // `general.name` is the checkpoint's name exactly, so it identifies.
        assert_eq!(
            Model::identify(Arch::Qwen4Exp, Some("Qwen3.8 Flash Next"), None),
            Some(Model::Qwen38FlashNext)
        );
        // And a real file name — which spells the release the way the release
        // spells it, with hyphens — identifies through the literal pass.
        assert_eq!(
            Model::identify(
                Arch::Qwen4Exp,
                None,
                Some(Path::new(
                    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
                ))
            ),
            Some(Model::Qwen38FlashNext)
        );
    }

    fn scratch(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xwen_hub_{}_{label}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The hub's on-disk folder for a repo: `/` becomes `--` under a
    /// `models--` prefix.
    fn repo_dir(root: &Path, repo: &str) -> PathBuf {
        root.join(format!("models--{}", repo.replace('/', "--")))
    }

    fn install(root: &Path, commit: &str, repo_id: &str, file: &str) {
        let repo = repo_dir(root, repo_id);
        let blobs = repo.join("blobs");
        let snap = repo.join("snapshots").join(commit);
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(blobs.join("aa00"), b"gguf bytes").unwrap();
        std::os::unix::fs::symlink(Path::new("../../blobs/aa00"), snap.join(file)).unwrap();
        // Bare hash, no trailing newline: hf-hub reads this verbatim.
        std::fs::write(repo.join("refs/main"), commit).unwrap();
    }

    #[test]
    fn resolves_through_refs_main() {
        let root = scratch("refs");
        let model = Model::Qwen35BA3B;
        install(&root, "cafe01", model.repo(), model.file());
        let path = cached_file_in(&root, model.repo(), model.file()).unwrap();
        assert!(
            path.ends_with("snapshots/cafe01/Qwen3.6-35B-A3B-Q4_K_M.gguf"),
            "{path:?}"
        );
        assert!(cached_file_in(&root, model.repo(), model.drafter_file().unwrap()).is_none());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_dangling_blob_symlink_is_a_miss() {
        let root = scratch("dangling");
        let model = Model::Qwen35BA3B;
        install(&root, "cafe03", model.repo(), model.file());
        std::fs::remove_file(repo_dir(&root, model.repo()).join("blobs/aa00")).unwrap();
        assert!(cached_file_in(&root, model.repo(), model.file()).is_none());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_missing_repo_is_a_miss() {
        let root = scratch("missing");
        let model = Model::Qwen35BA3B;
        assert!(cached_file_in(&root, model.repo(), model.file()).is_none());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The two checkpoints resolve to different repos, so a cached 35B never
    /// satisfies a 27B request.
    #[test]
    fn the_two_models_live_in_separate_repos() {
        let root = scratch("separate");
        install(
            &root,
            "cafe05",
            Model::Qwen35BA3B.repo(),
            Model::Qwen35BA3B.file(),
        );
        assert!(
            cached_file_in(&root, Model::Qwen35BA3B.repo(), Model::Qwen35BA3B.file()).is_some()
        );
        assert!(cached_file_in(&root, Model::Qwen27B.repo(), Model::Qwen27B.file()).is_none());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A zero-flag run gets the best checkpoint here, not the one bring-up
    /// started on. Pinned because the default is a product decision that lives
    /// in one derive attribute, where nothing else would notice it moving.
    #[test]
    fn the_default_is_flash_next() {
        assert_eq!(Model::default(), Model::Qwen38FlashNext);
        assert_eq!(Model::default().full_name(), "Qwen3.8-Flash-Next");
    }

    /// Serve answers with a checkpoint it can actually run — which, now that
    /// every checkpoint's state has a cache image, is the plain default. The
    /// fallback rule stays because the question it answers stays; what changed
    /// is that nothing takes the fallback branch.
    #[test]
    fn serve_default_is_the_first_servable_checkpoint() {
        let fallback = Model::default_servable();
        assert!(fallback.servable(), "{fallback}");
        // A server must not keep serving an older model for a reason nobody
        // wrote down: while the default is servable the two converge.
        assert_eq!(fallback, Model::default());
    }

    /// `xwen batch` answers with the same zero-flag checkpoint the server does:
    /// both move cache state on their ordinary path — batch snapshots the items'
    /// shared prefix and restores per item, and snapshots again around every
    /// enum-scored option — so both resolve through the same rule.
    ///
    /// Pinned across the two modules rather than inside either: the two defaults
    /// are resolved by different code (`BatchRequest::model`, `run_serve`), and
    /// a batch document written for a server — or resubmitted from one — must
    /// not quietly run on a different checkpoint than the server would have used.
    #[test]
    fn the_batch_default_is_the_serve_default() {
        let absent: crate::batch::BatchRequest =
            serde_json::from_str(r#"{ "items": [] }"#).unwrap();
        assert_eq!(absent.model().unwrap(), Model::default_servable());
        assert_eq!(absent.model().unwrap(), Model::default());
        assert!(absent.model().unwrap().servable());
    }

    #[test]
    fn model_names_round_trip_through_the_cli_spelling() {
        for model in MODELS {
            assert_eq!(model.to_string().parse::<Model>().unwrap(), model);
            // A name a `/v1/models` listing shows is also a `--model-size`.
            assert_eq!(model.full_name().parse::<Model>().unwrap(), model);
        }
        assert_eq!("35B-A3B".parse::<Model>().unwrap(), Model::Qwen35BA3B);
        assert_eq!("38".parse::<Model>().unwrap(), Model::Qwen3827B);
        assert_eq!("3.8".parse::<Model>().unwrap(), Model::Qwen3827B);
        assert!("70b".parse::<Model>().is_err());
    }
}
