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
    /// Qwen3.6-35B-A3B, MoE (`qwen35moe`) — the bring-up model.
    #[default]
    Qwen35BA3B,
    /// Qwen3.8-27B, dense (`qwen35`). The 3.8 release's config is byte-identical
    /// to [`Model::Qwen27B`]'s, so the two run the same graph at the same
    /// geometry and differ only in weights, repo — and drafter KIND: 3.8 ships an
    /// MTP head where the 3.6 checkpoints ship DFlash sidecars.
    Qwen3827B,
}

/// Every checkpoint this build knows, in the order surfaces that enumerate them
/// (`/v1/models`, an unknown-model error's list of valid names) print them.
pub const MODELS: [Model; 3] = [Model::Qwen27B, Model::Qwen35BA3B, Model::Qwen3827B];

/// The hub coordinates of one checkpoint: the repo, the Q4_K_M target (the
/// quant this machine is built around), and its drafter sidecar — of whichever
/// kind the release ships — when it ships one. The sizes are the hub's own byte counts, rounded, and
/// exist only for the "this is about to download" notice.
struct Checkpoint {
    repo: &'static str,
    model: &'static str,
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
};

const QWEN_27B: Checkpoint = Checkpoint {
    repo: "ggml-org/Qwen3.6-27B-GGUF",
    model: "Qwen3.6-27B-Q4_K_M.gguf",
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
};

const QWEN_35B_A3B: Checkpoint = Checkpoint {
    repo: "ggml-org/Qwen3.6-35B-A3B-GGUF",
    model: "Qwen3.6-35B-A3B-Q4_K_M.gguf",
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
    },
};

/// The 3.8 release ships no DFlash sidecar. It ships an MTP head instead — one
/// extra trunk-flavour layer, chained rather than blocked — which is why this is
/// the one checkpoint whose speculation is a different shape from the other two.
const QWEN_38_27B: Checkpoint = Checkpoint {
    repo: "ggml-org/Qwen3.8-27B-GGUF",
    model: "Qwen3.8-27B-Q4_K_M.gguf",
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
};

impl Model {
    const fn checkpoint(self) -> &'static Checkpoint {
        match self {
            Model::Qwen27B => &QWEN_27B,
            Model::Qwen35BA3B => &QWEN_35B_A3B,
            Model::Qwen3827B => &QWEN_38_27B,
        }
    }

    /// The Hugging Face repo holding both the target and its drafter.
    pub const fn repo(self) -> &'static str {
        self.checkpoint().repo
    }

    /// The Q4_K_M target GGUF.
    pub const fn file(self) -> &'static str {
        self.checkpoint().model
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
    pub const fn chat_dialect(self) -> crate::chat::ChatDialect {
        match self {
            Model::Qwen27B | Model::Qwen35BA3B => crate::chat::ChatDialect::Qwen36,
            Model::Qwen3827B => crate::chat::ChatDialect::Qwen38,
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
            sole(
                candidates()
                    .filter(|model| model.full_name().eq_ignore_ascii_case(name))
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
    pub const fn kv_bytes_per_token(self) -> usize {
        let g = &self.checkpoint().geometry;
        g.full_attn_layers * g.n_kv_head * g.head_dim * 2 * 2
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
    pub const fn snapshot_bytes(self) -> usize {
        let g = &self.checkpoint().geometry;
        let conv_dim = (2 * g.linear_k_heads + g.linear_v_heads) * g.linear_head_dim;
        let conv = (g.conv_kernel - 1) * conv_dim * 4;
        let delta = g.linear_v_heads * g.linear_head_dim * g.linear_head_dim * 4;
        g.linear_layers * (conv + delta)
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
            other => Err(format!(
                "unknown model {other:?} (expected 27b, 35b or 3.8-27b)"
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
pub fn cached_model(model: Model) -> Option<PathBuf> {
    cached_file(model.repo(), model.file())
}

/// The cached drafter sidecar for `model`, or `None` — which is also the answer
/// for a checkpoint that ships none at all. Offline.
pub fn cached_drafter(model: Model) -> Option<PathBuf> {
    cached_file(model.repo(), model.drafter_file()?)
}

/// The cached drafter of the default model, or `None`. Offline. The zero-arg
/// entry point for callers that have no model selection to pass.
pub fn official_drafter() -> Option<PathBuf> {
    cached_drafter(Model::default())
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

/// The target GGUF for `model`, downloaded on first use.
pub fn ensure_model(model: Model) -> Result<PathBuf> {
    ensure_file(model.repo(), model.file())
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

        // The 27B is the expensive one on both axes, which is what the sizing
        // prose that quotes a worst case has to be read against.
        assert!(Model::Qwen27B.kv_bytes_per_token() > Model::Qwen35BA3B.kv_bytes_per_token());
        assert!(Model::Qwen27B.snapshot_bytes() > Model::Qwen35BA3B.snapshot_bytes());
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
    /// offers all of it or none. Every shipped checkpoint now offers all of it,
    /// each naming its own kind, file and fitted floor — so what this pins is
    /// that the answers agree with each other, rather than a `None` that no
    /// shipped checkpoint produces any more.
    #[test]
    fn every_checkpoint_answers_the_whole_drafter_question() {
        for model in MODELS {
            let kind = model
                .drafter_kind()
                .unwrap_or_else(|| panic!("{model:?} names no drafter kind"));
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
        assert_eq!(Model::from_api_name("35b"), None);
        assert_eq!(Model::from_api_name("3.8"), None);
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
        let model = Model::default();
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
        let model = Model::default();
        install(&root, "cafe03", model.repo(), model.file());
        std::fs::remove_file(repo_dir(&root, model.repo()).join("blobs/aa00")).unwrap();
        assert!(cached_file_in(&root, model.repo(), model.file()).is_none());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_missing_repo_is_a_miss() {
        let root = scratch("missing");
        let model = Model::default();
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
