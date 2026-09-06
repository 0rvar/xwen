//! One resolved checkpoint, whatever it is stored as.
//!
//! Two things now count as a checkpoint here: a GGUF file (one, or a
//! gguf-split set presented as one) and a Hugging Face BF16 safetensors
//! DIRECTORY. They are opened by different loaders, identified by different
//! rules and read by different weight layers — but every surface asks them the
//! same five questions, and asked those questions in five different places
//! before this module existed.
//!
//! [`CheckpointSource`] is where opening happens, and the only place that
//! decides what a path IS. Everything downstream — `Generator::load`,
//! `XwenModel::load`, the serve engine's startup read, the disk tier's
//! checkpoint id, the CLI's one-shot identity check — takes a resolved source
//! rather than a path, so a directory cannot be a checkpoint on one surface and
//! a parse error on another.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::Device;

use crate::config::{Identity, XwenConfig};
use crate::gguf::{CheckpointId, GgufFile};
use crate::hub::Model;
use crate::qwen3::Qwen3Set;

/// An opened checkpoint: the weights, the metadata, and enough about where it
/// came from to say which checkpoint it is.
///
/// Cheap to clone (both arms are an `Arc`) and safe to hold across threads,
/// which the serve engine does.
#[derive(Clone)]
pub enum CheckpointSource {
    Gguf(Arc<GgufFile>),
    SafeTensors(Arc<Qwen3Set>),
}

impl std::fmt::Debug for CheckpointSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gguf(gguf) => f
                .debug_tuple("Gguf")
                .field(&gguf.checkpoint_path())
                .finish(),
            Self::SafeTensors(set) => f.debug_tuple("SafeTensors").field(&set.dir()).finish(),
        }
    }
}

impl CheckpointSource {
    /// Open whatever `path` names.
    ///
    /// The path decides, not the flag: a `.gguf` is a GGUF, and a directory —
    /// or a `config.json` or `.safetensors` file inside one, which is what
    /// [`crate::hub::ensure_model`] hands back for a safetensors entry — is a
    /// safetensors set opened at that directory. Anything else is tried as a
    /// GGUF, which is what it has always been.
    ///
    /// `entry` is the registry checkpoint the caller believes this is, when it
    /// knows one. It supplies the two things a safetensors set cannot work out
    /// for itself: where the tokenizer is (the Z-Image encoder's sits in a
    /// sibling directory, not its own) and which planes are allowed to be
    /// zero-filled. Passing `None` is safe — the loader searches for the
    /// tokenizer and allows no zero runs — but it means a Z-Image directory
    /// opened without naming it is refused for the corruption it really has.
    pub fn open(path: &Path, device: &Device, entry: Option<Model>) -> Result<Self> {
        if let Some(dir) = safetensors_dir(path) {
            let tokenizer = registry_tokenizer(entry, &dir);
            let allow = entry
                .map(Model::safetensors_allowed_zero_runs)
                .unwrap_or(&[]);
            let set = Qwen3Set::open(&dir, tokenizer, allow)
                .with_context(|| format!("reading the checkpoint directory {}", dir.display()))?;
            return Ok(Self::SafeTensors(Arc::new(set)));
        }
        Ok(Self::Gguf(crate::gguf::open(path, device)?))
    }

    /// The runtime config, parsed from whichever metadata this checkpoint
    /// carries.
    pub fn config(&self) -> Result<XwenConfig> {
        match self {
            Self::Gguf(gguf) => XwenConfig::from_gguf(&gguf.content),
            Self::SafeTensors(set) => Ok(XwenConfig::from_qwen3(set.config(), None)),
        }
    }

    /// What persisted state (a cache image, a disk-tier segment) is stamped
    /// with and refused against. Covers every file's metadata section and every
    /// file's length, and never the tensor payload — on either arm.
    pub fn checkpoint_id(&self) -> CheckpointId {
        match self {
            Self::Gguf(gguf) => gguf.checkpoint_id(),
            Self::SafeTensors(set) => set.checkpoint_id(),
        }
    }

    /// The path this checkpoint was opened at — the GGUF (shard 0 of a split
    /// set), or the safetensors directory. What identity and labelling are
    /// computed from; not a file to read whole.
    pub fn path(&self) -> &Path {
        match self {
            Self::Gguf(gguf) => gguf.checkpoint_path(),
            Self::SafeTensors(set) => set.dir(),
        }
    }

    /// Which checkpoint to RUN this as, cross-checked against an explicit
    /// selection — [`XwenConfig::identify`] against this source's own path, so
    /// no caller has to remember which path it opened.
    pub fn identify(
        &self,
        cfg: &XwenConfig,
        selected: Option<Model>,
        selector: &str,
    ) -> Result<Identity> {
        cfg.identify(self.path(), selected, selector)
    }

    /// What to call a checkpoint that identifies as none of the official ones:
    /// a GGUF's file stem, a set's directory name. Both are what the operator
    /// typed, minus the path.
    pub fn label(&self) -> String {
        let path = self.path();
        let name = match self {
            Self::Gguf(_) => path.file_stem(),
            Self::SafeTensors(_) => path.file_name(),
        };
        name.map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "xwen".to_string())
    }

    /// Bytes of weight this checkpoint holds on disk, for the memory line a
    /// load prints before it allocates anything.
    ///
    /// The GGUF arm sums the tensor table, which is the weights and nothing
    /// else. The safetensors arm sums the shard FILES, which is the same thing
    /// to within their JSON headers (tens of kilobytes against eight
    /// gigabytes): a safetensors shard is a header and then payload, with
    /// nothing else in it.
    pub fn footprint_bytes(&self) -> u64 {
        match self {
            Self::Gguf(gguf) => gguf
                .content
                .tensor_infos
                .values()
                .map(|info| {
                    let dt = info.ggml_dtype;
                    info.shape.elem_count() as u64 / dt.block_size() as u64 * dt.type_size() as u64
                })
                .sum(),
            Self::SafeTensors(set) => set
                .shard_paths()
                .into_iter()
                .filter_map(|path| std::fs::metadata(path).ok())
                .map(|meta| meta.len())
                .sum(),
        }
    }

    /// The GGUF behind this source, or `None` for a safetensors set. For the
    /// consumers that are still GGUF-only and say so — the drafter loaders,
    /// `xwen inspect`'s tensor table.
    pub fn gguf(&self) -> Option<&Arc<GgufFile>> {
        match self {
            Self::Gguf(gguf) => Some(gguf),
            Self::SafeTensors(_) => None,
        }
    }

    /// The error every surface gives for a checkpoint whose weights load but
    /// whose graph does not exist yet. One sentence, one place, so the three
    /// surfaces that can reach it say the same thing.
    pub fn unimplemented_stack(&self) -> anyhow::Error {
        anyhow::anyhow!(
            "the qwen3 stack is not implemented yet: {} is a Qwen3 dense safetensors \
             checkpoint, whose config, tokenizer and weights this build reads but whose \
             layer stack it cannot run",
            self.path().display()
        )
    }
}

/// The safetensors checkpoint directory `path` names, or `None` when `path` is
/// a GGUF (or anything else a GGUF open should be tried on).
///
/// Three shapes resolve to a directory, and they are the three real callers:
/// the directory itself (`--model <dir>`), its `config.json` (what
/// `hub::ensure_model` returns for a safetensors entry, since `files()[0]` is
/// the config) and one of its shards (someone who tab-completed).
fn safetensors_dir(path: &Path) -> Option<PathBuf> {
    if path.extension().is_some_and(|ext| ext == "gguf") {
        return None;
    }
    if path.is_dir() {
        return Some(path.to_path_buf());
    }
    let is_set_member = path.file_name().is_some_and(|name| name == "config.json")
        || path.extension().is_some_and(|ext| ext == "safetensors");
    is_set_member.then(|| path.parent().unwrap_or(Path::new(".")).to_path_buf())
}

/// The tokenizer the registry names for `entry`, resolved against `dir`.
///
/// The registry's paths are relative to the REPO root, which is not always the
/// checkpoint directory: the Z-Image encoder is `text_encoder/` inside a repo
/// whose tokenizer is `tokenizer/tokenizer.json`, one level up. Both roots are
/// tried, and `None` — no registry entry, or neither path on disk — leaves the
/// loader to its own search, which knows the same two layouts.
fn registry_tokenizer(entry: Option<Model>, dir: &Path) -> Option<PathBuf> {
    let relative = entry?.safetensors_tokenizer()?;
    let mut roots = vec![dir.to_path_buf()];
    if let Some(parent) = dir.parent() {
        roots.push(parent.to_path_buf());
    }
    roots
        .into_iter()
        .map(|root| root.join(relative))
        .find(|path| path.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The serve engine holds a resolved source across its worker threads, so
    /// both arms have to cross a thread boundary. The GGUF arm always could;
    /// this is the assertion that the safetensors one does too, and that a
    /// future field in `Qwen3Set` cannot quietly take it away.
    #[test]
    fn a_source_can_cross_a_thread_boundary() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CheckpointSource>();
        assert_send_sync::<Qwen3Set>();
        assert_send_sync::<Arc<GgufFile>>();
    }

    /// What a path IS decides which loader opens it, and the three shapes a
    /// safetensors checkpoint is named by all resolve to its directory.
    #[test]
    fn a_path_resolves_to_the_loader_that_can_read_it() {
        let dir = std::env::temp_dir().join(format!("xwen_ckpt_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // A directory is a set.
        assert_eq!(safetensors_dir(&dir), Some(dir.clone()));
        // So are the files `ensure_model` and a shell completion hand back.
        assert_eq!(safetensors_dir(&dir.join("config.json")), Some(dir.clone()));
        assert_eq!(
            safetensors_dir(&dir.join("model-00001-of-00003.safetensors")),
            Some(dir.clone())
        );
        // A GGUF is not, even one sitting next to a config.json.
        assert_eq!(safetensors_dir(&dir.join("Qwen3.6-27B-Q4_K_M.gguf")), None);
        assert_eq!(safetensors_dir(Path::new("/models/x.gguf")), None);
        // Neither is anything else: an unrecognized file keeps being tried as a
        // GGUF, which is what it was before this module existed.
        assert_eq!(safetensors_dir(Path::new("/models/weights.bin")), None);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The registry names the tokenizer relative to the repo, and the encoder's
    /// checkpoint directory is one level below that — so both roots are tried,
    /// and a directory that is neither layout falls through to the loader's own
    /// search rather than to a path that does not exist.
    #[test]
    fn the_registry_tokenizer_resolves_against_both_roots() {
        let root = std::env::temp_dir().join(format!("xwen_tok_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let encoder = root.join("text_encoder");
        std::fs::create_dir_all(&encoder).unwrap();
        std::fs::create_dir_all(root.join("tokenizer")).unwrap();

        // Nothing on disk yet: the loader is left to search.
        assert_eq!(
            registry_tokenizer(Some(Model::ZImageTurboEncoder), &encoder),
            None
        );

        // The sibling layout the Z-Image repo really has.
        let sibling = root.join("tokenizer/tokenizer.json");
        std::fs::write(&sibling, b"{}").unwrap();
        assert_eq!(
            registry_tokenizer(Some(Model::ZImageTurboEncoder), &encoder),
            Some(sibling)
        );

        // The flat layout the two language models have: the tokenizer sits in
        // the checkpoint directory itself.
        let flat = root.join("tokenizer.json");
        std::fs::write(&flat, b"{}").unwrap();
        assert_eq!(registry_tokenizer(Some(Model::Qwen34B), &root), Some(flat));

        // A GGUF checkpoint names no tokenizer path at all: its vocabulary is
        // inside the file.
        assert_eq!(registry_tokenizer(Some(Model::Qwen27B), &root), None);
        assert_eq!(registry_tokenizer(None, &root), None);

        std::fs::remove_dir_all(&root).unwrap();
    }
}
