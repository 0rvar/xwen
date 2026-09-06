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
///
/// The safetensors arm carries the device the caller opened it for: the set
/// itself is CPU-only (it maps and validates, and allocates nothing), but the
/// stack that runs it has to land on the SAME device as the caller's other
/// tensors — a second `Device::new_metal(0)` is a different candle device, and
/// an op between the two is a device mismatch, not a slow path. A GGUF carries
/// its device inside `GgufFile`.
#[derive(Clone)]
pub enum CheckpointSource {
    Gguf(Arc<GgufFile>),
    SafeTensors(Arc<Qwen3Set>, Device),
}

impl std::fmt::Debug for CheckpointSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gguf(gguf) => f
                .debug_tuple("Gguf")
                .field(&gguf.checkpoint_path())
                .finish(),
            Self::SafeTensors(set, _) => f.debug_tuple("SafeTensors").field(&set.dir()).finish(),
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
        if let Some(dir) = safetensors_dir(path)? {
            let tokenizer = registry_tokenizer(entry, &dir);
            let allow = entry
                .map(Model::safetensors_allowed_zero_runs)
                .unwrap_or(&[]);
            let set = Qwen3Set::open(&dir, tokenizer, allow)
                .with_context(|| format!("reading the checkpoint directory {}", dir.display()))?;
            return Ok(Self::SafeTensors(Arc::new(set), device.clone()));
        }
        Ok(Self::Gguf(crate::gguf::open(path, device)?))
    }

    /// The runtime config, parsed from whichever metadata this checkpoint
    /// carries.
    pub fn config(&self) -> Result<XwenConfig> {
        match self {
            Self::Gguf(gguf) => XwenConfig::from_gguf(&gguf.content),
            Self::SafeTensors(set, _) => Ok(XwenConfig::from_qwen3(set.config(), None)),
        }
    }

    /// What persisted state (a cache image, a disk-tier segment) is stamped
    /// with and refused against. Covers every file's metadata section and every
    /// file's length, and never the tensor payload — on either arm.
    pub fn checkpoint_id(&self) -> CheckpointId {
        match self {
            Self::Gguf(gguf) => gguf.checkpoint_id(),
            Self::SafeTensors(set, _) => set.checkpoint_id(),
        }
    }

    /// The path this checkpoint was opened at — the GGUF (shard 0 of a split
    /// set), or the safetensors directory. What identity and labelling are
    /// computed from; not a file to read whole.
    pub fn path(&self) -> &Path {
        match self {
            Self::Gguf(gguf) => gguf.checkpoint_path(),
            Self::SafeTensors(set, _) => set.dir(),
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
            Self::SafeTensors(set, _) => set
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
            Self::SafeTensors(..) => None,
        }
    }

    /// The safetensors set behind this source, or `None` for a GGUF. The mirror
    /// of [`CheckpointSource::gguf`], for the consumers that want what only a
    /// set has — its zero-run report, its shard list.
    pub fn safetensors(&self) -> Option<&Arc<Qwen3Set>> {
        match self {
            Self::SafeTensors(set, _) => Some(set),
            Self::Gguf(_) => None,
        }
    }

    /// The device this source was opened for — the GGUF's own, or the one
    /// the caller handed `open` for a safetensors set.
    pub fn device(&self) -> &Device {
        match self {
            Self::Gguf(gguf) => &gguf.device,
            Self::SafeTensors(_, device) => device,
        }
    }

    /// The `tokenizer.json` this checkpoint brings with it, when it brings one.
    ///
    /// `Some` for a safetensors set, whose tokenizer is a file beside it (or in
    /// a sibling directory, for the encoder), and which is the ONLY tokenizer
    /// that can read it: the Qwen3 vocabulary is 151936 ids wide with
    /// `<|im_end|>` at 151645, against the 3.6 family's 248320 and 248046. A
    /// run that loaded the wrong one would tokenize a prompt into the wrong ids,
    /// never see its stop token, and look like a model that will not stop.
    ///
    /// `None` for a GGUF, whose vocabulary is inside the file and whose callers
    /// use the copy compiled into the binary — the same bytes, and no second
    /// 12 MB parse per load.
    pub fn tokenizer_path(&self) -> Option<&Path> {
        match self {
            Self::Gguf(_) => None,
            Self::SafeTensors(set, _) => Some(set.tokenizer_path()),
        }
    }
}

/// What to call a checkpoint that identifies as none of the official ones: the
/// final path component the OPERATOR named, without an extension.
///
/// The path is the caller's, never one read back off an opened checkpoint, and
/// that distinction is load-bearing for a gguf-split set. `GgufFile` knows only
/// shard 0's path, so a server started on shard 2 of four would report shard 0's
/// name from here and shard 2's from `serve::model_id` — two names for one
/// running model, in the history and on the wire. Both call this instead.
///
/// `file_stem` on a directory too, which costs the tail of a dotted directory
/// name (`Qwen3-4B-v1.5` labels as `Qwen3-4B-v1`) and buys the two surfaces
/// agreeing without either of them stating the rule twice or touching the disk
/// to ask what kind of path it has.
pub fn label_for(opened: &Path) -> String {
    opened
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "xwen".to_string())
}

/// The safetensors checkpoint directory `path` names, `None` when `path` is a
/// GGUF (or anything else a GGUF open should be tried on), and an error when it
/// is neither.
///
/// Three shapes resolve to a directory, and they are the three real callers:
/// the directory itself (`--model <dir>`), its `config.json` (what
/// `hub::ensure_model` returns for a safetensors entry, since `files()[0]` is
/// the config) and one of its shards (someone who tab-completed).
///
/// Order matters in two places. The path must EXIST before anything is decided
/// from it, so a typo fails as a typo rather than as whatever the extension
/// suggests. And `is_dir` is asked before the extension, because a directory
/// called `weights.gguf` is a directory: the extension is a hint about a file
/// and says nothing about a folder someone named after one.
///
/// A file's DIRECTORY is canonicalized; the file itself never is, and the two
/// are not interchangeable. `Path::new("config.json").parent()` is `Some("")`,
/// not `None`, so a bare relative path would otherwise resolve to the empty
/// directory and fail on a config that was right there — while canonicalizing
/// the file instead would be worse: in the Hugging Face cache every file is a
/// symlink into a shared `blobs/` store, so resolving `config.json` yields a
/// content-hashed name in `blobs/`, which is neither called `config.json` nor
/// sitting anywhere near the checkpoint.
fn safetensors_dir(path: &Path) -> Result<Option<PathBuf>> {
    anyhow::ensure!(
        path.exists(),
        "{} does not exist: a checkpoint is a GGUF file or a safetensors directory",
        path.display()
    );
    if path.is_dir() {
        return Ok(Some(path.to_path_buf()));
    }
    if path.extension().is_some_and(|ext| ext == "gguf") {
        return Ok(None);
    }
    let Some(name) = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
    else {
        return Ok(None);
    };
    let is_shard = path.extension().is_some_and(|ext| ext == "safetensors");
    if name != "config.json" && !is_shard {
        // Not a shape this module recognizes as part of a set. A GGUF need not
        // be named `.gguf`, so the open is still tried — and fails there with
        // the parser's own message, which is what it did before this existed.
        return Ok(None);
    }
    let parent = path
        .parent()
        .with_context(|| format!("{} has no parent directory", path.display()))?;
    // `Some("")` is the whole reason this is not just `parent.to_path_buf()`.
    let parent = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    let dir = std::fs::canonicalize(parent)
        .with_context(|| format!("resolving the directory holding {}", path.display()))?;
    if is_shard {
        // A shard names its directory only if that directory's index claims it.
        // Without this, pointing at a stray `.safetensors` in an unrelated
        // folder opens that folder as a checkpoint and fails somewhere further
        // in, naming the directory rather than the file the operator typed.
        ensure_index_lists(&dir, &name)
            .with_context(|| format!("{} is not part of a checkpoint", path.display()))?;
    }
    Ok(Some(dir))
}

/// That `dir`'s shard index names `shard`.
///
/// Deliberately shallow: this is a membership question asked before the loader
/// runs, not a second validation of the index. `Qwen3Set::open` is what checks
/// that every tensor is in the shard the index assigns it, that no shard is
/// stray and that no name appears twice — and it will run on this directory a
/// moment later.
fn ensure_index_lists(dir: &Path, shard: &str) -> Result<()> {
    let index = dir.join("model.safetensors.index.json");
    let bytes = std::fs::read(&index)
        .with_context(|| format!("reading {} to check which shards it names", index.display()))?;
    let parsed: serde_json::Value =
        serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", index.display()))?;
    let listed = parsed
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .map(|map| map.values().any(|value| value.as_str() == Some(shard)))
        .unwrap_or(false);
    anyhow::ensure!(
        listed,
        "{} does not name {shard}, so that file is not a shard of this checkpoint",
        index.display()
    );
    Ok(())
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
        let dir = scratch("ckpt");
        let write = |name: &str| std::fs::write(dir.join(name), b"x").unwrap();
        write("config.json");
        write("model-00001-of-00003.safetensors");
        write("Qwen3.6-27B-Q4_K_M.gguf");
        write("weights.bin");
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            br#"{"weight_map":{"a":"model-00001-of-00003.safetensors"}}"#,
        )
        .unwrap();

        // A directory is a set, and is kept as the operator wrote it: an error
        // about it should quote the path they typed.
        assert_eq!(safetensors_dir(&dir).unwrap(), Some(dir.clone()));
        // So are the files `ensure_model` and a shell completion hand back —
        // resolved to the canonical directory, which is what makes a bare
        // relative path work (see `a_file_resolves_to_its_canonical_directory`).
        let canonical = std::fs::canonicalize(&dir).unwrap();
        assert_eq!(
            safetensors_dir(&dir.join("config.json")).unwrap(),
            Some(canonical.clone())
        );
        assert_eq!(
            safetensors_dir(&dir.join("model-00001-of-00003.safetensors")).unwrap(),
            Some(canonical)
        );
        // A GGUF is not, even one sitting next to a config.json.
        assert_eq!(
            safetensors_dir(&dir.join("Qwen3.6-27B-Q4_K_M.gguf")).unwrap(),
            None
        );
        // Neither is anything else: an unrecognized file keeps being tried as a
        // GGUF, because a GGUF need not be named one.
        assert_eq!(safetensors_dir(&dir.join("weights.bin")).unwrap(), None);

        // A DIRECTORY named like a GGUF is a directory. The extension describes
        // a file; it says nothing about a folder somebody named after one.
        let masquerading = dir.join("mymodel.gguf");
        std::fs::create_dir(&masquerading).unwrap();
        assert_eq!(
            safetensors_dir(&masquerading).unwrap(),
            Some(masquerading.clone())
        );

        // A path that is not there fails as a path that is not there, before
        // any of the above is consulted.
        let err = safetensors_dir(&dir.join("nope.gguf"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not exist"), "{err}");

        // A stray shard in a folder that is not a checkpoint names the FILE the
        // operator typed, rather than opening its folder and failing later on
        // something they did not mention.
        let stray = scratch("stray");
        std::fs::write(stray.join("model-00001-of-00003.safetensors"), b"x").unwrap();
        let err = safetensors_dir(&stray.join("model-00001-of-00003.safetensors"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("is not part of a checkpoint"), "{err}");
        // And one the index does not claim is refused even where an index
        // exists.
        std::fs::write(
            stray.join("model.safetensors.index.json"),
            br#"{"weight_map":{"a":"model-00002-of-00003.safetensors"}}"#,
        )
        .unwrap();
        let err = safetensors_dir(&stray.join("model-00001-of-00003.safetensors"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("is not part of a checkpoint"), "{err}");

        std::fs::remove_dir_all(&dir).unwrap();
        std::fs::remove_dir_all(&stray).unwrap();
    }

    /// A file resolves to its RESOLVED directory, never to the parent of the
    /// string the operator typed.
    ///
    /// That is what makes a bare relative `config.json` work:
    /// `Path::new("config.json").parent()` is `Some("")`, not `None`, so taking
    /// the parent of the operator's own path would have named the empty
    /// directory and failed on a config sitting right there. Asserted as
    /// "equals the canonical directory" rather than by changing the process's
    /// working directory, which no test should do to a suite that runs in
    /// threads — and it is a real assertion, because a temp directory reaches
    /// its canonical form through a symlink on this platform.
    #[test]
    fn a_file_resolves_to_its_canonical_directory() {
        let dir = scratch("relative");
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        let canonical = std::fs::canonicalize(&dir).unwrap();

        assert_eq!(
            safetensors_dir(&dir.join("config.json")).unwrap(),
            Some(canonical.clone())
        );
        // The premise of the assertion above: the typed path and the canonical
        // one really are different here, so equality with the canonical one is
        // evidence and not a tautology.
        assert_ne!(dir, canonical, "temp_dir is expected to be a symlink here");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A hub-cache checkpoint is reached through symlinks, and the FILE must not
    /// be resolved through them.
    ///
    /// Every file in a Hugging Face snapshot is a symlink into a shared
    /// `blobs/` store under a content-hashed name. Canonicalizing
    /// `snapshots/<commit>/config.json` therefore yields something called
    /// `abc123` sitting in `blobs/` — not named `config.json`, and nowhere near
    /// the shards. An earlier version of this did exactly that and sent every
    /// cached safetensors checkpoint to the GGUF parser, which reported an
    /// unknown magic number for a JSON file.
    #[test]
    fn a_cache_symlink_resolves_to_the_snapshot_and_not_to_the_blob_store() {
        let root = scratch("blobs");
        let blobs = root.join("blobs");
        let snapshot = root.join("snapshots/cafe01");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(blobs.join("aa00"), b"{}").unwrap();
        std::os::unix::fs::symlink(Path::new("../../blobs/aa00"), snapshot.join("config.json"))
            .unwrap();

        assert_eq!(
            safetensors_dir(&snapshot.join("config.json")).unwrap(),
            Some(std::fs::canonicalize(&snapshot).unwrap()),
            "a cached config.json must name its snapshot, not the blob store"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The label is the operator's own path, minus directories and extension —
    /// and it is the CALLER's path, which for a gguf-split set is the shard they
    /// named rather than shard 0. `serve::model_id` resolves through here for
    /// exactly that reason.
    #[test]
    fn the_label_is_the_operators_own_path() {
        assert_eq!(
            label_for(Path::new("/models/laguna-s-2.1-Q4_K_M.gguf")),
            "laguna-s-2.1-Q4_K_M"
        );
        // The shard the operator named, not the one the loader walked to.
        assert_eq!(
            label_for(Path::new(
                "/m/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf"
            )),
            "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004"
        );
        assert_eq!(label_for(Path::new("/models/my-qwen3-4b")), "my-qwen3-4b");
        assert_eq!(label_for(Path::new("/")), "xwen");
    }

    fn scratch(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "xwen_ckpt_{}_{label}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
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
