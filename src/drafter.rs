//! The drafter seam: what the decode loops know about a speculative drafter,
//! independent of which kind is attached.
//!
//! Two kinds ship, and they are genuinely different shapes rather than two
//! implementations of one idea. DFlash is a block-diffusion drafter: it takes a
//! noise block of `block_size` masked positions and denoises them in one
//! non-causal forward, so its whole draft arrives at once and its context is fed
//! by the target's PRE-norm residual taps. MTP is a chain drafter: one extra
//! trunk-flavour layer stepped repeatedly, each step consuming its own previous
//! token and hidden, fed by the target's POST-final-norm hidden.
//!
//! So this trait deliberately carries only what a decode loop needs to know
//! WITHOUT knowing which is attached — context length, rollback, capacity — and
//! the round itself branches on the enum. A trait wide enough to cover both
//! rounds would have to be a union of their inputs, and every caller would then
//! have to know which half applied, which is the knowledge it was supposed to
//! hide.

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file::Content;

use crate::dflash::{DflashDrafter, DrafterImage, DrafterImageKind};
use crate::mtp::MtpDrafter;

/// Which shape of drafter a sidecar holds.
///
/// Named separately from the drafters themselves because it is asked before
/// either is loaded — by the hub, which sizes a cache from it, and by both
/// attach paths, which decide from it which graph to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrafterKind {
    /// A DFlash block-diffusion sidecar: a small model of its own that proposes
    /// a whole block per forward.
    Dflash,
    /// A Qwen MTP head: one extra trunk-flavour layer, chained a few steps at a
    /// time.
    Mtp,
}

impl DrafterKind {
    /// The name an error or a log line uses.
    pub fn name(self) -> &'static str {
        match self {
            DrafterKind::Dflash => "DFlash",
            DrafterKind::Mtp => "MTP",
        }
    }

    /// The tag a cache image written by this kind carries, for a caller asking
    /// whether a stored record belongs to the drafter it has.
    pub fn image_kind(self) -> DrafterImageKind {
        match self {
            DrafterKind::Dflash => DrafterImageKind::Dflash,
            DrafterKind::Mtp => DrafterImageKind::Mtp,
        }
    }
}

/// Which kind of drafter an opened sidecar GGUF holds, or why it is not a
/// drafter at all.
///
/// The kind is decided by the FILE, never by the checkpoint being served: an
/// operator's `--draft <path>` may be either kind against any checkpoint, and a
/// classifier that consulted `hub::Model::drafter_kind` would build the wrong
/// graph for a file that says plainly what it is.
///
/// One function rather than one per entry point. The CLI and the server both
/// reach this, and two classifiers that disagreed would mean a sidecar accepted
/// at startup and refused at attach, or worse, loaded as the wrong kind by one
/// of them.
///
/// The two archs identify themselves differently, which is the whole reason this
/// needs more than one line. A DFlash sidecar is its own architecture. An MTP
/// sidecar declares the arch of the model it EXTENDS, so `qwen35` alone does not
/// identify one — a plain dense checkpoint says exactly the same thing — and
/// what separates a sidecar from a target somebody pointed `--draft` at by
/// mistake is the head's own presence. Both the declared count and the head
/// tensor must be there: a file declaring `nextn_predict_layers` without the
/// tensors would pass a metadata-only check at startup and then fail at every
/// attach, which is the failure this refuses up front.
pub fn classify(content: &Content) -> Result<DrafterKind> {
    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .cloned()
        .context("the drafter GGUF declares no general.architecture")?;
    match arch.as_str() {
        "dflash" => Ok(DrafterKind::Dflash),
        "qwen35" => {
            let declared = content
                .metadata
                .get("qwen35.nextn_predict_layers")
                .and_then(|v| v.to_u32().ok())
                .is_some_and(|n| n >= 1);
            if !declared {
                anyhow::bail!(
                    "this GGUF is a dense qwen35 model, not a drafter: it declares no MTP head \
                     (qwen35.nextn_predict_layers)"
                );
            }
            anyhow::ensure!(
                carries_mtp_head(content),
                "this GGUF declares an MTP head (qwen35.nextn_predict_layers) but carries none of \
                 its tensors: no blk.N.nextn.eh_proj.weight is present"
            );
            Ok(DrafterKind::Mtp)
        }
        other => anyhow::bail!(
            "a drafter must be a DFlash sidecar (\"dflash\") or an MTP sidecar (\"qwen35\" with \
             an MTP head); this GGUF declares {other:?}"
        ),
    }
}

/// Whether the head's own tensors are in the file.
///
/// The suffix is `.nextn.eh_proj.weight`, not `.nextn.eh_proj`: GGUF names a
/// projection's weight plane with the `.weight` on the end, and a probe without
/// it matches nothing at all — which is a check that silently always says no.
fn carries_mtp_head(content: &Content) -> bool {
    content
        .tensor_infos
        .keys()
        .any(|name| name.starts_with("blk.") && name.ends_with(".nextn.eh_proj.weight"))
}

/// The operations a decode loop performs on a drafter without caring which kind
/// it is: how much context it holds, how to give some back, and how far it can
/// draft.
pub(crate) trait DrafterOps {
    /// Positions the drafter's own cache holds — never advanced by drafting
    /// itself, only by the sync that commits accepted tokens.
    fn committed_len(&self) -> usize;
    /// Give back context after a partial accept. Cheap on both kinds: the
    /// rejected rows sit past the length and the next write overwrites them.
    fn truncate(&mut self, len: usize) -> Result<()>;
    /// Drop all context (a new conversation, or a cache the engine reset).
    fn reset(&mut self) -> Result<()>;
    /// Positions the drafter's cache is sized for, which is also the depth past
    /// which speculation stops.
    fn max_ctx(&self) -> usize;
}

/// The drafter a generator has attached, if any.
pub enum AttachedDrafter {
    /// The DFlash block-diffusion sidecar (both Qwen 3.6 checkpoints).
    Dflash(DflashDrafter),
    /// The Qwen 3.8 MTP head.
    Mtp(MtpDrafter),
}

impl AttachedDrafter {
    /// Which kind this is: for log lines and stats that name it, and for a
    /// caller deciding whether a stored cache image belongs to this drafter.
    pub fn kind(&self) -> DrafterKind {
        match self {
            AttachedDrafter::Dflash(_) => DrafterKind::Dflash,
            AttachedDrafter::Mtp(_) => DrafterKind::Mtp,
        }
    }

    /// The DFlash drafter, for the round code that speaks its block-diffusion
    /// shape (noise blocks, tap injection).
    pub(crate) fn as_dflash_mut(&mut self) -> Option<&mut DflashDrafter> {
        match self {
            AttachedDrafter::Dflash(d) => Some(d),
            AttachedDrafter::Mtp(_) => None,
        }
    }

    /// The MTP head, for the round code that speaks its chain shape.
    pub(crate) fn as_mtp_mut(&mut self) -> Option<&mut MtpDrafter> {
        match self {
            AttachedDrafter::Mtp(m) => Some(m),
            AttachedDrafter::Dflash(_) => None,
        }
    }

    /// Export the drafter's cache for the prefix-cache tier. Both kinds write a
    /// [`DrafterImage`], which carries the kind that wrote it — the two records
    /// share a tag (`serve::disk_cache`'s `TAG_DRAFTER`) and not a layout, so an
    /// image can only ever be imported back into its own kind.
    pub fn export_cache(&self) -> Result<Option<DrafterImage>> {
        match self {
            AttachedDrafter::Dflash(d) => d.export_cache().map(Some),
            AttachedDrafter::Mtp(m) => m.export_cache().map(Some),
        }
    }

    /// Import a stored cache prefix.
    ///
    /// The two kinds differ in what a PARTIAL import means. A block drafter's
    /// rows are each a function of their own position's taps, so any prefix of
    /// an image restores a usable drafter. The MTP head's next row needs the
    /// target hidden at the position before it, and an image carries exactly one
    /// such hidden — the one at its own end — so it resumes at that position or
    /// not at all, and says so.
    pub fn import_cache(&mut self, image: &DrafterImage, pos: usize) -> Result<()> {
        match self {
            AttachedDrafter::Dflash(d) => d.import_cache(image, pos),
            AttachedDrafter::Mtp(m) => m.import_cache(image, pos),
        }
    }
}

impl DrafterOps for AttachedDrafter {
    fn committed_len(&self) -> usize {
        match self {
            AttachedDrafter::Dflash(d) => d.committed_len(),
            AttachedDrafter::Mtp(m) => m.committed_len(),
        }
    }

    fn truncate(&mut self, len: usize) -> Result<()> {
        match self {
            AttachedDrafter::Dflash(d) => d.truncate(len),
            AttachedDrafter::Mtp(m) => m.truncate(len),
        }
    }

    fn reset(&mut self) -> Result<()> {
        match self {
            AttachedDrafter::Dflash(d) => {
                d.reset();
                Ok(())
            }
            AttachedDrafter::Mtp(m) => m.reset(),
        }
    }

    fn max_ctx(&self) -> usize {
        match self {
            AttachedDrafter::Dflash(d) => d.max_ctx(),
            AttachedDrafter::Mtp(m) => m.max_ctx(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;
    use candle_core::quantized::gguf_file::{TensorInfo, Value, VersionedMagic};
    use std::collections::HashMap;

    /// A GGUF header with chosen metadata and tensor names, which is all the
    /// classifier reads. Cheap enough that every arm gets its own — no file, no
    /// device, no weights.
    fn header(
        arch: &str,
        nextn: Option<u32>,
        tensors: &[&str],
    ) -> candle_core::quantized::gguf_file::Content {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            Value::String(arch.to_string()),
        );
        if let Some(n) = nextn {
            metadata.insert("qwen35.nextn_predict_layers".to_string(), Value::U32(n));
        }
        let tensor_infos = tensors
            .iter()
            .map(|name| {
                (
                    name.to_string(),
                    TensorInfo {
                        ggml_dtype: GgmlDType::F32,
                        shape: candle_core::Shape::from(1),
                        offset: 0,
                    },
                )
            })
            .collect();
        candle_core::quantized::gguf_file::Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos,
            tensor_data_offset: 0,
        }
    }

    /// A DFlash sidecar is its own architecture and says so.
    #[test]
    fn a_dflash_sidecar_is_classified_by_its_architecture() {
        assert_eq!(
            classify(&header("dflash", None, &[])).unwrap(),
            DrafterKind::Dflash
        );
    }

    /// An MTP sidecar declares the arch of the model it EXTENDS, so it takes both
    /// the declared head count AND the head's own tensors to tell one from the
    /// dense target that declares exactly the same architecture.
    ///
    /// The tensor name is the part that has already been wrong once: GGUF names a
    /// projection's weight plane with `.weight` on the end, so a probe for a bare
    /// `.nextn.eh_proj` matches nothing at all — a check that silently always says
    /// no, leaving the metadata key deciding alone.
    #[test]
    fn an_mtp_sidecar_needs_both_the_declaration_and_the_tensors() {
        let sidecar = header(
            "qwen35",
            Some(1),
            &["blk.64.nextn.eh_proj.weight", "blk.64.attn_norm.weight"],
        );
        assert_eq!(classify(&sidecar).unwrap(), DrafterKind::Mtp);

        // Declared, but the tensors are not there: this file cannot draft
        // whatever it says about itself, and it fails at startup rather than at
        // every attach for the rest of the process's life.
        let err = classify(&header("qwen35", Some(1), &["blk.0.attn_norm.weight"]))
            .expect_err("a declaration without tensors is not a drafter")
            .to_string();
        assert!(err.contains("carries none of its tensors"), "{err}");

        // The tensors without the declaration is the same answer, reported as
        // what it most likely is: somebody pointed --draft at a target GGUF.
        let err = classify(&header("qwen35", None, &["blk.64.nextn.eh_proj.weight"]))
            .expect_err("an undeclared head is not a drafter")
            .to_string();
        assert!(err.contains("dense qwen35 model, not a drafter"), "{err}");

        // And a plain dense checkpoint, which declares the identical
        // architecture and is the file this check exists to separate out.
        let err = classify(&header("qwen35", None, &["blk.0.attn_norm.weight"]))
            .expect_err("a dense target is not a drafter")
            .to_string();
        assert!(err.contains("dense qwen35 model, not a drafter"), "{err}");
    }

    /// Anything else names what it found and what it would have accepted, since
    /// the operator who reaches this passed a path by hand.
    #[test]
    fn any_other_architecture_is_refused_by_name() {
        let err = classify(&header("llama", None, &[]))
            .expect_err("an unrelated arch is not a drafter")
            .to_string();
        assert!(err.contains("llama"), "{err}");
        assert!(err.contains("dflash"), "{err}");

        let mut headless = header("dflash", None, &[]);
        headless.metadata.remove("general.architecture");
        let err = classify(&headless)
            .expect_err("a GGUF with no architecture at all is not a drafter")
            .to_string();
        assert!(err.contains("no general.architecture"), "{err}");
    }
}
