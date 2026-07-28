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

use std::path::PathBuf;

use anyhow::{Context, Result};
use hf_hub::Cache;
use hf_hub::api::sync::ApiBuilder;

/// Which official checkpoint to run. Both are ggml-org's Qwen 3.6 GGUF
/// conversions; the GGUF filenames keep the model's name — only the engine is
/// called xwen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Model {
    /// Dense 27B (`qwen35`).
    Qwen27B,
    /// MoE 35B-A3B (`qwen35moe`) — the bring-up model.
    #[default]
    Qwen35BA3B,
}

/// The hub coordinates of one checkpoint: the repo, the Q4_K_M target (the
/// quant this machine is built around), and its DFlash drafter sidecar. The
/// sizes are the hub's own byte counts, rounded, and exist only for the
/// "this is about to download" notice.
struct Checkpoint {
    repo: &'static str,
    model: &'static str,
    drafter: &'static str,
    model_size: &'static str,
    drafter_size: &'static str,
    geometry: CacheGeometry,
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

const QWEN_27B: Checkpoint = Checkpoint {
    repo: "ggml-org/Qwen3.6-27B-GGUF",
    model: "Qwen3.6-27B-Q4_K_M.gguf",
    drafter: "dflash-Qwen3.6-27B-BF16.gguf",
    model_size: "19.1 GB",
    drafter_size: "3.5 GB",
    // 64 layers, full attention every fourth.
    geometry: CacheGeometry {
        full_attn_layers: 16,
        linear_layers: 48,
        n_kv_head: 4,
        head_dim: 256,
        linear_k_heads: 16,
        linear_v_heads: 48,
        linear_head_dim: 128,
        conv_kernel: 4,
    },
};

const QWEN_35B_A3B: Checkpoint = Checkpoint {
    repo: "ggml-org/Qwen3.6-35B-A3B-GGUF",
    model: "Qwen3.6-35B-A3B-Q4_K_M.gguf",
    drafter: "dflash-Qwen3.6-35B-A3B-BF16.gguf",
    model_size: "20.4 GB",
    drafter_size: "0.8 GB",
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

impl Model {
    const fn checkpoint(self) -> &'static Checkpoint {
        match self {
            Model::Qwen27B => &QWEN_27B,
            Model::Qwen35BA3B => &QWEN_35B_A3B,
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

    /// The DFlash block-diffusion drafter for speculative decoding.
    pub const fn drafter_file(self) -> &'static str {
        self.checkpoint().drafter
    }

    /// Human-readable download size of the target, for the fetch notice.
    pub const fn size(self) -> &'static str {
        self.checkpoint().model_size
    }

    /// Human-readable download size of the drafter, for the fetch notice.
    pub const fn drafter_size(self) -> &'static str {
        self.checkpoint().drafter_size
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

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Model::Qwen27B => "27b",
            Model::Qwen35BA3B => "35b",
        })
    }
}

impl std::str::FromStr for Model {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "27" | "27b" => Ok(Model::Qwen27B),
            "35" | "35b" | "35b-a3b" => Ok(Model::Qwen35BA3B),
            other => Err(format!("unknown model {other:?} (expected 27b or 35b)")),
        }
    }
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

/// The cached DFlash drafter for `model`, or `None`. Offline.
pub fn cached_drafter(model: Model) -> Option<PathBuf> {
    cached_file(model.repo(), model.drafter_file())
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

/// The DFlash drafter for `model`, downloaded on first use.
pub fn ensure_drafter(model: Model) -> Result<PathBuf> {
    ensure_file(model.repo(), model.drafter_file())
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
        assert!(cached_file_in(&root, model.repo(), model.drafter_file()).is_none());
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
        for model in [Model::Qwen27B, Model::Qwen35BA3B] {
            assert_eq!(model.to_string().parse::<Model>().unwrap(), model);
        }
        assert_eq!("35B-A3B".parse::<Model>().unwrap(), Model::Qwen35BA3B);
        assert!("70b".parse::<Model>().is_err());
    }
}
