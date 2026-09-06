//! The BF16 safetensors loader for `qwen3` checkpoints.
//!
//! A checkpoint here is a DIRECTORY, not a file: `config.json`,
//! `model.safetensors.index.json` and the shards it names. [`Qwen3Set::open`]
//! does every check that does not need a device — the index agrees with the
//! shards, the shards hold exactly the tensors this config implies, at the
//! shapes it implies, stored as BF16 — and only then does [`Qwen3Set::load_all`]
//! allocate anything. That split is why the whole loader is testable without a
//! Metal device.
//!
//! # Copy, not alias
//!
//! `gguf::dense_alias_tensor` builds no-copy Metal views over an mmap, which the
//! matmul kernels can use because a GGUF's tensor payload starts 16-byte
//! aligned. Safetensors gives no such guarantee: the payload starts at
//! `8 + header_len`, and the shipped Qwen3-4B files put it at `data_start % 16
//! == 8` for shards 1 and 2 (headers 20000 and 25232 bytes) and `== 0` for shard
//! 3 (header 552). Two of three shards are therefore unaliasable, so this loader
//! copies through candle's `MmapedSafetensors::load`, which reads the mmap into a
//! fresh device buffer. Eight gigabytes of copy, once, at load.
//!
//! # Integrity
//!
//! [`Qwen3Set::open`] scans every projection plane's raw bytes for two things
//! neither the index nor the shapes can catch. A long run of exact zeros means
//! a truncated or zero-filled upload — the Z-Image-Turbo `text_encoder/` copy of
//! this model has two such planes in layer 35 — and is an error unless the
//! caller allowlists that tensor by name. Values outside f16's range are
//! counted, not refused: the tensor-path gemm stages BF16 weights to half, so
//! values below f16's subnormal floor flush to zero there while the gemv path
//! keeps them, and the counts are what say whether that split can matter for
//! this checkpoint.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor};

use crate::gguf::CheckpointId;
use serde::Deserialize;

use super::config::{HfQwen3Config, Qwen3Config};

/// A run of exact zeros longer than this many elements is treated as corruption.
///
/// Well above anything a trained plane produces by accident: 4096 consecutive
/// zeros is a whole row and a half of a 2560-wide projection.
pub const ZERO_RUN_LIMIT: usize = 4096;

/// The longest run of exact zeros found in one tensor, in elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ZeroRun {
    /// Index of the first zero element of the run, counting from the start of
    /// the tensor in row-major order.
    pub start: usize,
    /// Length of the run in elements.
    pub len: usize,
}

/// How much of a checkpoint's weight lives outside the range f16 can hold.
///
/// Both counts are over BF16 projection weights only; norm planes are excluded
/// because they never reach a matmul kernel as weights.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RangeScan {
    /// Nonzero values with `|x| < 2^-24`, f16's smallest subnormal. These flush
    /// to zero when a weight is staged to half for the tensor-core gemm and
    /// survive on the widening gemv path.
    pub below_f16_subnormal: u64,
    /// Values with `|x| > 65504`, f16's largest finite value, plus infinities
    /// and NaNs, which sort above that bound in the bit pattern. A finite value
    /// over the bound becomes inf when a weight is staged to half; an inf stays
    /// inf and a NaN stays NaN.
    pub above_f16_max: u64,
    /// How many BF16 projection elements were scanned.
    pub elements: u64,
}

/// One shard of a set: the file, its mmap, and the size and header the
/// checkpoint id is built from.
struct Shard {
    /// File name as the index names it, e.g. `model-00001-of-00003.safetensors`.
    name: String,
    path: PathBuf,
    /// Where the JSON header stops and the tensor payload begins: the 8-byte
    /// length prefix plus the header itself. This is the shard's whole metadata
    /// section, and the only part of it the checkpoint id reads.
    metadata_len: u64,
    st: MmapedSafetensors,
}

/// `model.safetensors.index.json`.
#[derive(Debug, Deserialize)]
struct ShardIndex {
    weight_map: WeightMap,
}

/// The index's tensor-to-shard table.
///
/// Deserialized through a visitor rather than straight into a `BTreeMap`,
/// because a map deserializer takes the LAST value for a repeated JSON key and
/// says nothing. An index that names one tensor twice is either two shards
/// claiming it or a generator that lost track, and neither should resolve to
/// whichever entry happened to come second.
#[derive(Debug)]
struct WeightMap(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for WeightMap {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = WeightMap;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a map of tensor name to shard file name")
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<WeightMap, A::Error> {
                let mut out: BTreeMap<String, String> = BTreeMap::new();
                while let Some((tensor, shard)) = map.next_entry::<String, String>()? {
                    if let Some(first) = out.get(&tensor) {
                        return Err(serde::de::Error::custom(format!(
                            "weight_map lists {tensor} twice, as {first} and as {shard}"
                        )));
                    }
                    out.insert(tensor, shard);
                }
                Ok(WeightMap(out))
            }
        }
        d.deserialize_map(Visitor)
    }
}

/// Which kind of plane a tensor is, which decides how it is loaded and whether
/// it is scanned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Plane {
    /// An RMSNorm weight vector. Widened to F32 at load, because candle's Metal
    /// `rms_norm` needs the weight and the activations at the same dtype and the
    /// activations are F32. Never scanned: a 2560-element vector cannot hold a
    /// run over the limit.
    Norm,
    /// A matmul weight. Stays BF16 and is scanned.
    Projection,
}

/// One expected entry: what it is called, what shape it must have, what kind of
/// plane it is.
struct Expected {
    name: String,
    shape: Vec<usize>,
    plane: Plane,
}

/// Every tensor a `qwen3` checkpoint of this config must contain, in a stable
/// order. `lm_head.weight` is NOT here: it is optional, because the embedding is
/// tied.
fn expected_tensors(cfg: &Qwen3Config) -> Vec<Expected> {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let head_dim = cfg.head_dim();
    let q = cfg.q_dim();
    let kv = cfg.kv_dim();
    let mut out = vec![Expected {
        name: "model.embed_tokens.weight".to_string(),
        shape: vec![cfg.vocab_size, hidden],
        plane: Plane::Projection,
    }];
    for i in 0..cfg.n_layer {
        let p = format!("model.layers.{i}");
        for (suffix, shape, plane) in [
            ("input_layernorm.weight", vec![hidden], Plane::Norm),
            (
                "self_attn.q_proj.weight",
                vec![q, hidden],
                Plane::Projection,
            ),
            (
                "self_attn.k_proj.weight",
                vec![kv, hidden],
                Plane::Projection,
            ),
            (
                "self_attn.v_proj.weight",
                vec![kv, hidden],
                Plane::Projection,
            ),
            (
                "self_attn.o_proj.weight",
                vec![hidden, q],
                Plane::Projection,
            ),
            ("self_attn.q_norm.weight", vec![head_dim], Plane::Norm),
            ("self_attn.k_norm.weight", vec![head_dim], Plane::Norm),
            ("post_attention_layernorm.weight", vec![hidden], Plane::Norm),
            (
                "mlp.gate_proj.weight",
                vec![inter, hidden],
                Plane::Projection,
            ),
            ("mlp.up_proj.weight", vec![inter, hidden], Plane::Projection),
            (
                "mlp.down_proj.weight",
                vec![hidden, inter],
                Plane::Projection,
            ),
        ] {
            out.push(Expected {
                name: format!("{p}.{suffix}"),
                shape,
                plane,
            });
        }
    }
    out.push(Expected {
        name: "model.norm.weight".to_string(),
        shape: vec![hidden],
        plane: Plane::Norm,
    });
    out
}

/// The name a present-but-optional LM head goes under.
const LM_HEAD: &str = "lm_head.weight";

/// A validated, opened `qwen3` safetensors checkpoint.
///
/// Holding one keeps every shard mmapped. Nothing is on a device yet; call
/// [`Qwen3Set::load_all`] for that.
pub struct Qwen3Set {
    dir: PathBuf,
    tokenizer: PathBuf,
    config: Qwen3Config,
    hf: HfQwen3Config,
    shards: Vec<Shard>,
    /// Tensor name to its index into `shards`, from the validated index file.
    routing: BTreeMap<String, usize>,
    /// True when the set ships an explicit `lm_head.weight`. It is byte-equal to
    /// the embedding (checked at open), so the loader still reports it as tied.
    has_lm_head: bool,
    zero_runs: Vec<(String, ZeroRun)>,
    range: RangeScan,
    checkpoint: CheckpointId,
}

/// Everything about a set except the mmaps, which have no useful rendering.
impl std::fmt::Debug for Qwen3Set {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3Set")
            .field("dir", &self.dir)
            .field("tokenizer", &self.tokenizer)
            .field(
                "shards",
                &self.shards.iter().map(|s| &s.name).collect::<Vec<_>>(),
            )
            .field("tensors", &self.routing.len())
            .field("has_lm_head", &self.has_lm_head)
            .field("zero_runs", &self.zero_runs)
            .field("range", &self.range)
            .field("checkpoint", &self.checkpoint.dir_name())
            .finish()
    }
}

impl Qwen3Set {
    /// Open and fully validate a checkpoint directory.
    ///
    /// `tokenizer` overrides where `tokenizer.json` is looked for. With `None`
    /// the directory's own `tokenizer.json` is used, falling back to a sibling
    /// `tokenizer/tokenizer.json` — the Z-Image layout, where the text encoder
    /// and its tokenizer are separate subdirectories of one repo.
    ///
    /// `allow_zero_runs` names tensors whose long zero runs are known and
    /// tolerated. Anything not named there is an error.
    pub fn open(dir: &Path, tokenizer: Option<PathBuf>, allow_zero_runs: &[&str]) -> Result<Self> {
        let config_path = dir.join("config.json");
        let config_bytes = std::fs::read(&config_path)
            .with_context(|| format!("reading {}", config_path.display()))?;
        let hf = HfQwen3Config::from_json_bytes(&config_bytes)
            .with_context(|| format!("in {}", config_path.display()))?;
        let config =
            Qwen3Config::from_hf(&hf).with_context(|| format!("in {}", config_path.display()))?;

        let index_path = dir.join("model.safetensors.index.json");
        let index_bytes = std::fs::read(&index_path)
            .with_context(|| format!("reading {}", index_path.display()))?;
        let index: ShardIndex = serde_json::from_slice(&index_bytes)
            .with_context(|| format!("parsing {}", index_path.display()))?;
        let weight_map = &index.weight_map.0;
        ensure!(
            !weight_map.is_empty(),
            "{} lists no tensors",
            index_path.display()
        );

        // Shards in the order the index first mentions them, deduplicated and
        // sorted by name so the checkpoint id does not depend on map order.
        let shard_names: BTreeSet<&String> = weight_map.values().collect();
        let mut shards = Vec::with_capacity(shard_names.len());
        for name in &shard_names {
            let path = dir.join(name.as_str());
            let metadata_len = read_metadata_len(&path)?;
            // SAFETY: same contract as everywhere else in this crate — the
            // checkpoint files are not modified while mapped.
            let st = unsafe { MmapedSafetensors::new(&path) }
                .with_context(|| format!("mapping {}", path.display()))?;
            shards.push(Shard {
                name: (*name).clone(),
                path,
                metadata_len,
                st,
            });
        }
        // A `.safetensors` file the index does not name is not part of the set
        // the index describes, and running past it would mean loading some other
        // checkpoint's shard by accident or silently ignoring the one that
        // matters.
        for entry in std::fs::read_dir(dir)
            .with_context(|| format!("listing {}", dir.display()))?
            .flatten()
        {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.ends_with(".safetensors") || shard_names.iter().any(|s| **s == name) {
                continue;
            }
            bail!(
                "{} holds {name}, which {} does not reference; a set with a shard the index \
                 does not name is not the set the index describes",
                dir.display(),
                index_path.display()
            );
        }

        let shard_slot: BTreeMap<&str, usize> = shards
            .iter()
            .enumerate()
            .map(|(i, s)| (s.name.as_str(), i))
            .collect();

        // What each shard actually holds.
        let held: Vec<BTreeSet<String>> = shards
            .iter()
            .map(|s| s.st.tensors().into_iter().map(|(n, _)| n).collect())
            .collect();

        let mut routing = BTreeMap::new();
        for (tensor, shard) in weight_map {
            let slot = *shard_slot.get(shard.as_str()).ok_or_else(|| {
                anyhow!(
                    "{} routes {tensor} to {shard}, which is not one of the shards it names",
                    index_path.display()
                )
            })?;
            ensure!(
                held[slot].contains(tensor),
                "{} routes {tensor} to {shard}, which does not contain it",
                index_path.display()
            );
            for (other, names) in held.iter().enumerate() {
                if other != slot && names.contains(tensor) {
                    bail!(
                        "{tensor} is stored in both {} and {}; a tensor must live in exactly the \
                         shard the index names",
                        shards[slot].name,
                        shards[other].name
                    );
                }
            }
            routing.insert(tensor.clone(), slot);
        }
        for (slot, names) in held.iter().enumerate() {
            for name in names {
                ensure!(
                    routing.contains_key(name),
                    "{} holds {name}, which {} does not list",
                    shards[slot].name,
                    index_path.display()
                );
            }
        }

        // Shapes, dtypes, and the exact name set the config implies.
        let expected = expected_tensors(&config);
        let mut allowed: BTreeSet<&str> = expected.iter().map(|e| e.name.as_str()).collect();
        allowed.insert(LM_HEAD);
        for name in routing.keys() {
            ensure!(
                allowed.contains(name.as_str()),
                "{} holds {name}, which is not part of a qwen3 checkpoint with {} layers",
                dir.display(),
                config.n_layer
            );
        }
        let has_lm_head = routing.contains_key(LM_HEAD);
        let lm_head_expected = Expected {
            name: LM_HEAD.to_string(),
            shape: vec![config.vocab_size, config.hidden_size],
            plane: Plane::Projection,
        };
        for want in expected
            .iter()
            .chain(has_lm_head.then_some(&lm_head_expected))
        {
            let slot = *routing.get(&want.name).ok_or_else(|| {
                anyhow!(
                    "{} is missing {}; a qwen3 checkpoint with {} layers needs it",
                    dir.display(),
                    want.name,
                    config.n_layer
                )
            })?;
            let view = shards[slot]
                .st
                .get(&want.name)
                .map_err(|e| anyhow!("reading {} from {}: {e}", want.name, shards[slot].name))?;
            ensure!(
                view.shape() == want.shape.as_slice(),
                "{} has shape {:?} in {}, expected {:?}",
                want.name,
                view.shape(),
                shards[slot].name,
                want.shape
            );
            ensure!(
                DType::try_from(view.dtype()).ok() == Some(DType::BF16),
                "{} is stored as {:?} in {}; this loader reads BF16 checkpoints only",
                want.name,
                view.dtype(),
                shards[slot].name
            );
        }

        // A shipped lm_head must be the embedding it is tied to, byte for byte.
        // Anything else means the file is not the tied checkpoint its config
        // claims, and silently preferring one of the two would be a coin flip.
        if has_lm_head {
            let embed = "model.embed_tokens.weight";
            let a = shards[routing[embed]]
                .st
                .get(embed)
                .map_err(|e| anyhow!("{e}"))?;
            let b = shards[routing[LM_HEAD]]
                .st
                .get(LM_HEAD)
                .map_err(|e| anyhow!("{e}"))?;
            let (a, b) = (a.data(), b.data());
            if let Some(byte) = a.iter().zip(b).position(|(x, y)| x != y) {
                bail!(
                    "{} sets tie_word_embeddings true, but {LM_HEAD} in {} differs from {embed} \
                     in {}: first difference at element {} (byte {byte}), {:?} against {:?}. The \
                     checkpoint contradicts itself and preferring either plane would be a coin \
                     flip",
                    dir.display(),
                    shards[routing[LM_HEAD]].name,
                    shards[routing[embed]].name,
                    byte / 2,
                    bf16_at(a, byte / 2 * 2),
                    bf16_at(b, byte / 2 * 2),
                );
            }
        }

        // Integrity, over the projection planes only, on one thread pool.
        let allow: BTreeSet<&str> = allow_zero_runs.iter().copied().collect();
        let mut planes: Vec<(&str, &[u8])> = Vec::new();
        for want in expected
            .iter()
            .chain(has_lm_head.then_some(&lm_head_expected))
        {
            if want.plane != Plane::Projection {
                continue;
            }
            let bytes = shards[routing[&want.name]]
                .st
                .get(&want.name)
                .map_err(|e| anyhow!("reading {} for the integrity scan: {e}", want.name))?
                .data();
            planes.push((want.name.as_str(), bytes));
        }
        let scans = scan_planes(&planes);

        let mut zero_runs = Vec::new();
        let mut range = RangeScan::default();
        let mut worst_below: Option<(&str, u64)> = None;
        let mut worst_above: Option<(&str, u64)> = None;
        for ((name, _), scan) in planes.iter().zip(&scans) {
            range.below_f16_subnormal += scan.below;
            range.above_f16_max += scan.above;
            range.elements += scan.elements;
            if scan.below > worst_below.map_or(0, |(_, n)| n) {
                worst_below = Some((name, scan.below));
            }
            if scan.above > worst_above.map_or(0, |(_, n)| n) {
                worst_above = Some((name, scan.above));
            }
            if let Some(run) = scan.run
                && run.len > ZERO_RUN_LIMIT
            {
                ensure!(
                    allow.contains(name),
                    "{name} in {} holds {} consecutive zero elements starting at element {}; that \
                     plane is corrupt or was never written",
                    shards[routing[*name]].name,
                    run.len,
                    run.start
                );
                zero_runs.push(((*name).to_string(), run));
            }
        }
        if range.below_f16_subnormal > 0 || range.above_f16_max > 0 {
            let worst = |what: &str, w: Option<(&str, u64)>| match w {
                Some((name, n)) => format!(", worst {what} {name} with {n}"),
                None => String::new(),
            };
            crate::host_log::host_line(format!(
                "xwen: qwen3 weights: {} of {} BF16 projection values below f16's subnormal floor \
                 and {} above f16 max{}{}",
                range.below_f16_subnormal,
                range.elements,
                range.above_f16_max,
                worst("below", worst_below.filter(|(_, n)| *n > 0)),
                worst("above", worst_above.filter(|(_, n)| *n > 0)),
            ));
        }
        if !zero_runs.is_empty() {
            crate::host_log::host_line(format!(
                "xwen: qwen3 weights: {} allowlisted zero-filled plane(s): {}",
                zero_runs.len(),
                zero_runs
                    .iter()
                    .map(|(n, r)| format!("{n} ({} zeros at {})", r.len, r.start))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        let tokenizer = resolve_tokenizer(dir, tokenizer)?;
        let checkpoint = checkpoint_id(&config_path, &index_path, &shards)?;

        Ok(Self {
            dir: dir.to_path_buf(),
            tokenizer,
            config,
            hf,
            shards,
            routing,
            has_lm_head,
            zero_runs,
            range,
            checkpoint,
        })
    }

    /// The directory the set was opened from.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The resolved `tokenizer.json`, ready for `LagunaTokenizer::from_file`.
    pub fn tokenizer_path(&self) -> &Path {
        &self.tokenizer
    }

    /// The validated runtime config.
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    /// The raw HF config, for anything that needs a key the runtime config does
    /// not carry.
    pub fn hf_config(&self) -> &HfQwen3Config {
        &self.hf
    }

    /// Paths of the shards, in the order the checkpoint id folds them.
    pub fn shard_paths(&self) -> Vec<&Path> {
        self.shards.iter().map(|s| s.path.as_path()).collect()
    }

    /// Whether the set ships an explicit `lm_head.weight`. When it does, it is
    /// byte-equal to the embedding, so nothing downstream needs to care.
    pub fn has_lm_head(&self) -> bool {
        self.has_lm_head
    }

    /// The allowlisted zero-filled planes this set was opened with, with the
    /// run each one actually holds. Empty for an intact checkpoint.
    pub fn zero_runs(&self) -> &[(String, ZeroRun)] {
        &self.zero_runs
    }

    /// How much of the weight sits outside f16's range. See [`RangeScan`].
    pub fn range_scan(&self) -> RangeScan {
        self.range
    }

    /// Identity of this checkpoint: the same [`CheckpointId`] a GGUF open
    /// produces, so a safetensors set keys cache images and disk-tier segments
    /// through the machinery that already exists.
    ///
    /// See [`checkpoint_id`] for which bytes it pins.
    pub fn checkpoint_id(&self) -> CheckpointId {
        self.checkpoint
    }

    /// A consume-exactly-once ledger over this set's tensors.
    ///
    /// Every expected name starts in it; [`TensorSet::take`] removes one and
    /// [`TensorSet::finish`] errors if anything is left. A loader that forgets a
    /// plane therefore fails loudly instead of running a layer with whatever was
    /// in the struct.
    pub fn tensor_set<'a>(&'a self, device: &Device) -> TensorSet<'a> {
        TensorSet {
            set: self,
            device: device.clone(),
            remaining: self.routing.keys().cloned().collect(),
        }
    }

    /// Load one tensor onto `device`, as stored (BF16).
    ///
    /// candle's `MmapedSafetensors::load` copies: it reads the mapped bytes into
    /// a fresh storage buffer, keeping the stored dtype. That copy is what makes
    /// the shards' 8-mod-16 payload alignment a non-issue.
    fn load_raw(&self, name: &str, device: &Device) -> Result<Tensor> {
        let slot = *self
            .routing
            .get(name)
            .ok_or_else(|| anyhow!("{name} is not in this checkpoint"))?;
        self.shards[slot]
            .st
            .load(name, device)
            .map_err(|e| anyhow!("loading {name} from {}: {e}", self.shards[slot].name))
    }

    /// Load every weight onto `device`.
    pub fn load_all(&self, device: &Device) -> Result<Qwen3Weights> {
        let cfg = &self.config;
        let mut set = self.tensor_set(device);
        let embed_tokens = set.take("model.embed_tokens.weight")?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("model.layers.{i}");
            layers.push(Qwen3LayerWeights {
                input_layernorm: set.take_f32(&format!("{p}.input_layernorm.weight"))?,
                q_proj: set.take(&format!("{p}.self_attn.q_proj.weight"))?,
                k_proj: set.take(&format!("{p}.self_attn.k_proj.weight"))?,
                v_proj: set.take(&format!("{p}.self_attn.v_proj.weight"))?,
                o_proj: set.take(&format!("{p}.self_attn.o_proj.weight"))?,
                q_norm: set.take_f32(&format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: set.take_f32(&format!("{p}.self_attn.k_norm.weight"))?,
                post_attention_layernorm: set
                    .take_f32(&format!("{p}.post_attention_layernorm.weight"))?,
                gate_proj: set.take(&format!("{p}.mlp.gate_proj.weight"))?,
                up_proj: set.take(&format!("{p}.mlp.up_proj.weight"))?,
                down_proj: set.take(&format!("{p}.mlp.down_proj.weight"))?,
            });
        }
        let norm = set.take_f32("model.norm.weight")?;
        if self.has_lm_head {
            // `open` has already proven this plane byte-equal to the embedding,
            // so it is struck off the ledger without being read: loading it
            // would put a second copy of the vocabulary plane on the device
            // (742 MiB on the 4B) only to drop it.
            set.consume_alias(LM_HEAD)?;
        }
        // Always tied on this architecture: the config check refuses
        // `tie_word_embeddings: false`, so an untied head cannot reach here. The
        // field is `Option` for the checkpoint that one day is untied, and that
        // one will have to lift the config check first.
        let lm_head = None;
        set.finish()?;
        Ok(Qwen3Weights {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }
}

/// Every weight of a `qwen3` checkpoint, on one device.
///
/// Projections are BF16, as stored. Norm weights are F32: candle's Metal
/// `rms_norm` requires the weight dtype to match the activations, which are F32
/// through the whole stack.
pub struct Qwen3Weights {
    /// `[vocab, hidden]` BF16. Doubles as the LM head when `lm_head` is `None`.
    pub embed_tokens: Tensor,
    pub layers: Vec<Qwen3LayerWeights>,
    /// `[hidden]` F32, the final pre-head norm.
    pub norm: Tensor,
    /// `[vocab, hidden]` BF16, or `None` when the head is the embedding. `None`
    /// on every shipped Qwen3-4B.
    pub lm_head: Option<Tensor>,
}

/// One transformer layer's weights.
pub struct Qwen3LayerWeights {
    /// `[hidden]` F32, pre-attention.
    pub input_layernorm: Tensor,
    /// `[n_head * head_dim, hidden]` BF16.
    pub q_proj: Tensor,
    /// `[n_kv_head * head_dim, hidden]` BF16.
    pub k_proj: Tensor,
    /// `[n_kv_head * head_dim, hidden]` BF16.
    pub v_proj: Tensor,
    /// `[hidden, n_head * head_dim]` BF16.
    pub o_proj: Tensor,
    /// `[head_dim]` F32, applied per head before rope.
    pub q_norm: Tensor,
    /// `[head_dim]` F32, applied per head before rope.
    pub k_norm: Tensor,
    /// `[hidden]` F32, pre-MLP.
    pub post_attention_layernorm: Tensor,
    /// `[intermediate, hidden]` BF16.
    pub gate_proj: Tensor,
    /// `[intermediate, hidden]` BF16.
    pub up_proj: Tensor,
    /// `[hidden, intermediate]` BF16.
    pub down_proj: Tensor,
}

/// A ledger that hands out each tensor exactly once. See
/// [`Qwen3Set::tensor_set`].
pub struct TensorSet<'a> {
    set: &'a Qwen3Set,
    device: Device,
    remaining: BTreeSet<String>,
}

impl TensorSet<'_> {
    /// Take one tensor as stored, BF16.
    ///
    /// The ledger entry is cleared only once the load has succeeded, so a plane
    /// that failed to load is still outstanding at [`TensorSet::finish`] rather
    /// than looking consumed.
    pub fn take(&mut self, name: &str) -> Result<Tensor> {
        ensure!(
            self.remaining.contains(name),
            "{name} was already taken from this checkpoint, or is not part of it"
        );
        let tensor = self.set.load_raw(name, &self.device)?;
        self.remaining.remove(name);
        Ok(tensor)
    }

    /// Clear a ledger entry for a plane that is a byte-for-byte alias of one
    /// that IS loaded, without reading it.
    ///
    /// The one caller is the tied `lm_head.weight`: `open` has already proven it
    /// identical to the embedding, so loading it would allocate a second copy of
    /// a 742 MiB plane on the device to throw it away.
    pub(crate) fn consume_alias(&mut self, name: &str) -> Result<()> {
        ensure!(
            self.remaining.remove(name),
            "{name} was already taken from this checkpoint, or is not part of it"
        );
        Ok(())
    }

    /// Take one tensor and widen it to F32. For norm weights only.
    pub fn take_f32(&mut self, name: &str) -> Result<Tensor> {
        let t = self.take(name)?;
        t.to_dtype(DType::F32)
            .map_err(|e| anyhow!("widening {name} to f32: {e}"))
    }

    /// How many tensors have not been taken yet.
    pub fn remaining(&self) -> usize {
        self.remaining.len()
    }

    /// Assert that the checkpoint was consumed completely.
    pub fn finish(self) -> Result<()> {
        if self.remaining.is_empty() {
            return Ok(());
        }
        let mut names: Vec<&str> = self.remaining.iter().map(String::as_str).collect();
        names.truncate(8);
        bail!(
            "{} tensor(s) of the checkpoint were never loaded: {}{}",
            self.remaining.len(),
            names.join(", "),
            if self.remaining.len() > names.len() {
                ", …"
            } else {
                ""
            }
        )
    }
}

/// Where a set's `tokenizer.json` is.
fn resolve_tokenizer(dir: &Path, override_path: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = override_path {
        ensure!(p.is_file(), "no tokenizer.json at {}", p.display());
        return Ok(p);
    }
    let own = dir.join("tokenizer.json");
    if own.is_file() {
        return Ok(own);
    }
    // The Z-Image layout: `text_encoder/` holds the weights and a sibling
    // `tokenizer/` holds the tokenizer.
    let sibling = dir
        .parent()
        .map(|p| p.join("tokenizer").join("tokenizer.json"));
    if let Some(p) = &sibling
        && p.is_file()
    {
        return Ok(p.clone());
    }
    bail!(
        "no tokenizer.json for {}: tried {} and {}, and none was supplied",
        dir.display(),
        own.display(),
        sibling
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<no parent directory>".to_string())
    )
}

/// The BF16 value at byte offset `at` of a raw plane, for an error message that
/// has to say what actually differed. `f32` because that is what a reader wants
/// to compare, and the two values are already known to differ in their bits.
fn bf16_at(bytes: &[u8], at: usize) -> f32 {
    match bytes.get(at..at + 2) {
        Some(w) => half::bf16::from_bits(u16::from_le_bytes([w[0], w[1]])).to_f32(),
        None => f32::NAN,
    }
}

/// Where a safetensors file's JSON header stops: `8 + header_len`, which is the
/// whole of its metadata section.
fn read_metadata_len(path: &Path) -> Result<u64> {
    let mut file = File::open(path).with_context(|| format!("opening shard {}", path.display()))?;
    let file_len = file
        .metadata()
        .with_context(|| format!("stat {}", path.display()))?
        .len();
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .with_context(|| format!("reading the header length of {}", path.display()))?;
    let header_len = u64::from_le_bytes(prefix);
    ensure!(
        header_len <= file_len.saturating_sub(8) && header_len <= 1 << 30,
        "{} declares a {header_len}-byte header in a {file_len}-byte file",
        path.display()
    );
    Ok(8 + header_len)
}

/// The identity of a safetensors checkpoint, as the [`CheckpointId`] every
/// persisted artifact in this crate is already keyed on.
///
/// The bytes folded, in this order: the whole of `config.json`, the whole of
/// `model.safetensors.index.json`, and each shard's metadata section — its
/// 8-byte header length prefix plus the JSON header — in the shard order the set
/// opened them in. The total length is the sum of all those files in full.
///
/// The eight gigabytes of tensor payload are never read, for the same reason a
/// GGUF's are not: the three metadata pieces already pin every architectural
/// parameter, every tensor name, shape, dtype and offset, and which shard holds
/// what, while the summed file lengths catch a shard whose SIZE changed. What it
/// does not catch is a payload overwritten in place at the same length under an
/// unchanged header — two checkpoints that agree on every byte of metadata and
/// differ only in weight values share an id. That is the accepted limit:
/// the id exists to stop one checkpoint's cache image being fed to another among
/// the owner's own files, not to detect tampering, and reading eight gigabytes
/// at every load to close it would cost seconds for a case no shipped workflow
/// produces.
fn checkpoint_id(config: &Path, index: &Path, shards: &[Shard]) -> Result<CheckpointId> {
    let whole = |p: &Path| -> Result<u64> {
        Ok(std::fs::metadata(p)
            .with_context(|| format!("stat {}", p.display()))?
            .len())
    };
    let mut files: Vec<(&Path, u64)> = vec![(config, whole(config)?), (index, whole(index)?)];
    files.extend(shards.iter().map(|s| (s.path.as_path(), s.metadata_len)));
    CheckpointId::chain(&files).context("computing the qwen3 checkpoint id")
}
/// What one pass over a plane's raw BF16 bytes found.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct PlaneScan {
    below: u64,
    above: u64,
    /// Elements in the plane.
    elements: u64,
    /// The longest run of exact zeros, if there is one at all.
    run: Option<ZeroRun>,
}

/// One chunk's contribution, before the runs are stitched across boundaries.
#[derive(Debug, Clone, Copy, Default)]
struct ChunkScan {
    below: u64,
    above: u64,
    /// Elements in this chunk.
    len: usize,
    /// Length of the zero run this chunk opens with.
    lead: usize,
    /// Length of the zero run this chunk ends with.
    trail: usize,
    /// The longest run wholly inside this chunk, with its offset in the chunk.
    best: usize,
    best_start: usize,
    all_zero: bool,
}

/// How much of a plane one worker takes at a time. Even, so every chunk starts
/// on a BF16 element boundary, and large enough that a set's ~250 planes come
/// to about a thousand work items rather than one per plane.
const SCAN_CHUNK_BYTES: usize = 8 << 20;

/// f16's smallest subnormal, `2^-24`, as a sign-masked BF16 magnitude. Anything
/// strictly below it and nonzero flushes to zero when a weight is staged to
/// half.
const F16_SUBNORMAL_FLOOR: u16 = 0x3380;

/// `2^16`, the smallest BF16 magnitude that exceeds f16's largest finite value.
///
/// 65504 is not itself a BF16 value — with 7 mantissa bits the grid near there
/// is 65280 (`0x477F`) then 65536 (`0x4780`) — so "greater than f16 max" and
/// "at least `0x4780`" are the same set. Infinities and NaNs sort above the
/// bound, which is why they count here. The same bound
/// `dflash::ensure_bf16_fits_f16` uses.
const F16_OVERFLOW: u16 = 0x4780;

/// Scan one contiguous chunk of a plane's raw bytes as little-endian BF16 words.
///
/// Two things in one pass, because the pass is memory-bandwidth bound and a
/// checkpoint is eight gigabytes: how many values fall outside f16's range, and
/// where the zero runs are. A BF16 negative zero (`0x8000`) is a zero here: the
/// sign is masked off, and a plane filled with either bit pattern is equally
/// unwritten.
fn scan_chunk(bytes: &[u8]) -> ChunkScan {
    let mut out = ChunkScan {
        len: bytes.len() / 2,
        ..Default::default()
    };
    let mut cur = 0usize;
    let mut cur_start = 0usize;
    let mut leading = true;
    for (i, w) in bytes.chunks_exact(2).enumerate() {
        let mag = u16::from_le_bytes([w[0], w[1]]) & 0x7FFF;
        if mag == 0 {
            if cur == 0 {
                cur_start = i;
            }
            cur += 1;
            if cur > out.best {
                out.best = cur;
                out.best_start = cur_start;
            }
            continue;
        }
        if leading {
            out.lead = cur;
            leading = false;
        }
        cur = 0;
        if mag < F16_SUBNORMAL_FLOOR {
            out.below += 1;
        } else if mag >= F16_OVERFLOW {
            out.above += 1;
        }
    }
    if leading {
        // Never saw a nonzero element, so the whole chunk is one run.
        out.lead = cur;
        out.all_zero = out.len > 0;
    }
    out.trail = cur;
    out
}

/// Join a plane's chunk results back into one answer.
///
/// A run that spans a chunk boundary, or spans whole chunks, is one run: a
/// chunk carries the run it opens with and the run it ends with separately from
/// the longest run wholly inside it, and this walk joins them in order. The
/// result is therefore independent of how the plane was split, which is what
/// `the_scan_does_not_depend_on_the_chunk_size` pins.
fn stitch(chunks: &[ChunkScan]) -> PlaneScan {
    let mut out = PlaneScan::default();
    let mut best = ZeroRun { start: 0, len: 0 };
    // A run still open at the chunk boundary, and where it began.
    let mut carry = 0usize;
    let mut carry_start = 0usize;
    let mut base = 0usize;
    let mut record = |run: ZeroRun| {
        if run.len > best.len {
            best = run;
        }
    };
    for c in chunks {
        out.below += c.below;
        out.above += c.above;
        out.elements += c.len as u64;
        if c.all_zero {
            if carry == 0 {
                carry_start = base;
            }
            carry += c.len;
        } else {
            let start = if carry > 0 { carry_start } else { base };
            record(ZeroRun {
                start,
                len: carry + c.lead,
            });
            record(ZeroRun {
                start: base + c.best_start,
                len: c.best,
            });
            carry = c.trail;
            carry_start = base + c.len - c.trail;
        }
        base += c.len;
    }
    record(ZeroRun {
        start: carry_start,
        len: carry,
    });
    out.run = (best.len > 0).then_some(best);
    out
}

/// Scan one plane sequentially at a chosen chunk size. The scan's answer does
/// not depend on that size, which is the whole point of testing it at several.
fn scan_one(bytes: &[u8], chunk_bytes: usize) -> PlaneScan {
    let chunks: Vec<ChunkScan> = bytes.chunks(chunk_bytes.max(2)).map(scan_chunk).collect();
    stitch(&chunks)
}

/// Scan every projection plane of a set on ONE thread pool.
///
/// A checkpoint has around 250 projection planes. Giving each its own set of
/// worker threads would be thousands of thread starts per load for work that is
/// memory-bandwidth bound anyway, so every plane's chunks go into one work list
/// and a fixed number of workers take from it by index. Each worker returns the
/// results it produced tagged with their position, so the scatter back into
/// per-plane order needs no shared mutable state.
///
/// Returns one [`PlaneScan`] per input plane, in the order given.
fn scan_planes(planes: &[(&str, &[u8])]) -> Vec<PlaneScan> {
    // (plane index, chunk index within the plane, the bytes).
    let mut work: Vec<(usize, usize, &[u8])> = Vec::new();
    for (plane, (_, bytes)) in planes.iter().enumerate() {
        for (chunk, slice) in bytes.chunks(SCAN_CHUNK_BYTES).enumerate() {
            work.push((plane, chunk, slice));
        }
    }
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(16)
        .min(work.len().max(1));

    let next = std::sync::atomic::AtomicUsize::new(0);
    let work = &work;
    let taken: Vec<(usize, ChunkScan)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let next = &next;
                s.spawn(move || {
                    let mut mine = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&(_, _, bytes)) = work.get(i) else {
                            break;
                        };
                        mine.push((i, scan_chunk(bytes)));
                    }
                    mine
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("bf16 plane-scan thread panicked"))
            .collect()
    });

    let mut per_plane: Vec<Vec<ChunkScan>> = planes
        .iter()
        .map(|(_, bytes)| vec![ChunkScan::default(); bytes.len().div_ceil(SCAN_CHUNK_BYTES)])
        .collect();
    for (i, scan) in taken {
        let (plane, chunk, _) = work[i];
        per_plane[plane][chunk] = scan;
    }
    per_plane.iter().map(|c| stitch(c)).collect()
}

/// A tiny two-shard `qwen3` checkpoint, written by hand, for the loader tests.
///
/// The shipped checkpoints are eight gigabytes and cannot be a test dependency,
/// and every check in this module is a contract over a safetensors HEADER —
/// names, shapes, dtypes, which shard holds what. Hand-writing the file is what
/// makes the negative cases reachable at all: no serializer will produce a set
/// whose index points at the wrong shard for you.
///
/// The geometry is as small as the architecture allows. `head_dim` is pinned at
/// 128 by the config check, so it is 128 here too and everything else shrinks
/// around it — deliberately to four DIFFERENT widths: hidden 64, kv_dim 128,
/// q_dim 256, intermediate 96. A fixture where any two of those coincide would
/// let a transposed or swapped projection through the shape table unnoticed,
/// which is most of what the shape table is for.
#[cfg(test)]
pub(crate) mod fixture {
    use super::*;

    pub(crate) const HIDDEN: usize = 64;
    pub(crate) const INTERMEDIATE: usize = 96;
    pub(crate) const LAYERS: usize = 2;
    pub(crate) const HEADS: usize = 2;
    pub(crate) const KV_HEADS: usize = 1;
    pub(crate) const HEAD_DIM: usize = 128;
    pub(crate) const VOCAB: usize = 32;

    /// Fixed, so a fixture written twice is byte-identical and the checkpoint
    /// id is stable across runs.
    const SEED: u64 = 0x5eed_0f_11_22_33_44;

    pub(crate) const SHARD_A: &str = "model-00001-of-00002.safetensors";
    pub(crate) const SHARD_B: &str = "model-00002-of-00002.safetensors";

    /// Whether the fixture ships an explicit LM head, and whether it agrees with
    /// the embedding it claims to be tied to.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum LmHead {
        /// Byte-equal to `model.embed_tokens.weight`, as a tied checkpoint's is.
        Tied,
        /// A different plane under the same name, which contradicts the config.
        Divergent,
    }

    /// One deviation from an intact set. Everything is `None` by default, which
    /// writes a checkpoint that opens cleanly.
    #[derive(Default)]
    pub(crate) struct Tweaks {
        /// Remove a tensor from the index and from its shard.
        pub drop: Option<&'static str>,
        /// Keep the index entry, write the bytes into the other shard.
        pub wrong_shard: Option<&'static str>,
        /// Write the bytes into both shards.
        pub duplicate: Option<&'static str>,
        /// Store a tensor at some dtype other than BF16. The replacement must be
        /// two bytes wide so the byte count still matches the shape.
        pub dtype: Option<(&'static str, &'static str)>,
        /// Declare, and write, a different shape for a tensor.
        pub shape: Option<(&'static str, Vec<usize>)>,
        /// Zero `len` elements of a tensor starting at element `start`.
        pub zero_run: Option<(&'static str, usize, usize)>,
        /// Ship an `lm_head.weight`.
        pub lm_head: Option<LmHead>,
        /// Write an extra tensor into shard A and leave it out of the index.
        pub unlisted: Option<&'static str>,
        /// Write an extra tensor and list it, under a name the config does not
        /// imply.
        pub extra: Option<&'static str>,
        /// Fill a zero run with BF16 NEGATIVE zero (`0x8000`) instead of
        /// `0x0000`. Both are zero to the scan.
        pub negative_zeros: bool,
        /// Splice a second entry for this tensor into the index's `weight_map`
        /// JSON text, which no serializer will do for you.
        pub duplicate_index_key: Option<&'static str>,
        /// Leave a `.safetensors` file in the directory that the index does not
        /// name.
        pub stray_shard: bool,
        /// Write a `tokenizer.json` beside the weights.
        pub tokenizer: bool,
    }

    impl Tweaks {
        /// An intact set.
        pub(crate) fn intact() -> Self {
            Self {
                tokenizer: true,
                ..Default::default()
            }
        }
    }

    /// The fixture's `config.json`, as bytes.
    pub(crate) fn config_json() -> String {
        serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "max_position_embeddings": 4096,
            "vocab_size": VOCAB,
            "tie_word_embeddings": true,
            "attention_bias": false,
            "use_sliding_window": false,
            "rope_scaling": serde_json::Value::Null,
            "hidden_act": "silu",
        })
        .to_string()
    }

    /// Deterministic BF16 weights with magnitudes in `[0.5, 1)`: never zero,
    /// never near f16's subnormal floor, never past f16's max. An intact fixture
    /// therefore scans clean on all three counts.
    fn weights(seed: &mut u64, elements: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(elements * 2);
        for _ in 0..elements {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((*seed >> 33) as f32) / ((1u64 << 31) as f32);
            let magnitude = 0.5 + unit * 0.5;
            let signed = if (*seed >> 32) & 1 == 0 {
                magnitude
            } else {
                -magnitude
            };
            out.extend_from_slice(&half::bf16::from_f32(signed).to_bits().to_le_bytes());
        }
        out
    }

    /// One tensor as it will be written.
    struct Entry {
        name: String,
        shape: Vec<usize>,
        dtype: String,
        data: Vec<u8>,
        /// 0 for shard A, 1 for shard B.
        shard: usize,
        /// Whether the index lists it.
        listed: bool,
    }

    /// Write a two-shard set into `dir`, with `tweaks` applied.
    pub(crate) fn write_set(dir: &Path, tweaks: &Tweaks) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let config = config_json();
        let cfg = Qwen3Config::from_json_bytes(config.as_bytes())?;
        let expected = expected_tensors(&cfg);
        // Shard A takes the embedding and layer 0; shard B takes layer 1 and the
        // final norm, so both shards hold a mix of norm and projection planes.
        let split = 1 + (expected.len() - 2) / 2;
        let mut seed = SEED;
        let mut entries: Vec<Entry> = expected
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let n: usize = e.shape.iter().product();
                Entry {
                    name: e.name.clone(),
                    shape: e.shape.clone(),
                    dtype: "BF16".to_string(),
                    data: weights(&mut seed, n),
                    shard: usize::from(i >= split),
                    listed: true,
                }
            })
            .collect();

        if let Some(mode) = tweaks.lm_head {
            let embed = entries
                .iter()
                .find(|e| e.name == "model.embed_tokens.weight")
                .expect("the embedding is always written");
            let data = match mode {
                LmHead::Tied => embed.data.clone(),
                LmHead::Divergent => weights(&mut seed, VOCAB * HIDDEN),
            };
            entries.push(Entry {
                name: LM_HEAD.to_string(),
                shape: vec![VOCAB, HIDDEN],
                dtype: "BF16".to_string(),
                data,
                shard: 1,
                listed: true,
            });
        }
        if let Some(name) = tweaks.unlisted {
            entries.push(Entry {
                name: name.to_string(),
                shape: vec![HIDDEN],
                dtype: "BF16".to_string(),
                data: weights(&mut seed, HIDDEN),
                shard: 0,
                listed: false,
            });
        }
        if let Some(name) = tweaks.extra {
            entries.push(Entry {
                name: name.to_string(),
                shape: vec![HIDDEN],
                dtype: "BF16".to_string(),
                data: weights(&mut seed, HIDDEN),
                shard: 0,
                listed: true,
            });
        }
        if let Some(name) = tweaks.drop {
            let before = entries.len();
            entries.retain(|e| e.name != name);
            ensure!(
                entries.len() < before,
                "fixture: nothing named {name} to drop"
            );
        }
        if let Some(name) = tweaks.wrong_shard {
            let e = find(&mut entries, name)?;
            e.shard = 1 - e.shard;
        }
        if let Some((name, dtype)) = tweaks.dtype {
            find(&mut entries, name)?.dtype = dtype.to_string();
        }
        if let Some((name, shape)) = &tweaks.shape {
            let mut fresh = weights(&mut seed, shape.iter().product());
            let e = find(&mut entries, name)?;
            e.shape = shape.clone();
            std::mem::swap(&mut e.data, &mut fresh);
        }
        if let Some((name, start, len)) = tweaks.zero_run {
            let e = find(&mut entries, name)?;
            ensure!(
                (start + len) * 2 <= e.data.len(),
                "fixture: a {len}-element zero run at {start} does not fit {name}"
            );
            let zero = if tweaks.negative_zeros {
                half::bf16::NEG_ZERO.to_bits()
            } else {
                0
            }
            .to_le_bytes();
            for word in e.data[start * 2..(start + len) * 2].chunks_exact_mut(2) {
                word.copy_from_slice(&zero);
            }
        }
        // A duplicate is written into both shards; the index keeps naming one.
        let mut duplicate = None;
        if let Some(name) = tweaks.duplicate {
            let e = find(&mut entries, name)?;
            duplicate = Some(Entry {
                name: e.name.clone(),
                shape: e.shape.clone(),
                dtype: e.dtype.clone(),
                data: e.data.clone(),
                shard: 1 - e.shard,
                listed: false,
            });
        }

        let mut weight_map = serde_json::Map::new();
        for e in &entries {
            if e.listed {
                weight_map.insert(
                    e.name.clone(),
                    serde_json::Value::String(
                        if e.shard == 0 { SHARD_A } else { SHARD_B }.to_string(),
                    ),
                );
            }
        }
        // A tensor written into the wrong shard is still routed to its own.
        if let Some(name) = tweaks.wrong_shard {
            let e = find(&mut entries, name)?;
            let home = if e.shard == 0 { SHARD_B } else { SHARD_A };
            weight_map.insert(
                name.to_string(),
                serde_json::Value::String(home.to_string()),
            );
        }
        let index = serde_json::json!({
            "metadata": {"total_size": entries.iter().map(|e| e.data.len()).sum::<usize>()},
            "weight_map": weight_map,
        });

        if let Some(d) = duplicate {
            entries.push(d);
        }
        let mut index_text = index.to_string();
        if let Some(name) = tweaks.duplicate_index_key {
            // A repeated JSON key: legal JSON, and exactly what a map
            // deserializer would silently resolve last-wins.
            let at = index_text
                .find("\"weight_map\":{")
                .ok_or_else(|| anyhow!("fixture: no weight_map to duplicate a key in"))?
                + "\"weight_map\":{".len();
            index_text.insert_str(at, &format!("{:?}:{SHARD_B:?},", name));
        }
        std::fs::write(dir.join("config.json"), &config)?;
        std::fs::write(dir.join("model.safetensors.index.json"), index_text)?;
        if tweaks.stray_shard {
            std::fs::write(
                dir.join("model-00003-of-00002.safetensors"),
                b"not a shard this index names",
            )?;
        }
        for (slot, file) in [(0usize, SHARD_A), (1usize, SHARD_B)] {
            let of: Vec<&Entry> = entries.iter().filter(|e| e.shard == slot).collect();
            write_shard(&dir.join(file), &of)?;
        }
        if tweaks.tokenizer {
            std::fs::write(dir.join("tokenizer.json"), b"{}")?;
        }
        Ok(())
    }

    fn find<'a>(entries: &'a mut [Entry], name: &str) -> Result<&'a mut Entry> {
        entries
            .iter_mut()
            .find(|e| e.name == name)
            .ok_or_else(|| anyhow!("fixture: nothing named {name}"))
    }

    /// Serialize one shard: the 8-byte header length, the JSON header, then the
    /// payloads back to back. Offsets must be contiguous and the file must end
    /// exactly where the last payload does — safetensors validates both.
    fn write_shard(path: &Path, entries: &[&Entry]) -> Result<()> {
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        for e in entries {
            let end = offset + e.data.len();
            header.insert(
                e.name.clone(),
                serde_json::json!({
                    "dtype": e.dtype,
                    "shape": e.shape,
                    "data_offsets": [offset, end],
                }),
            );
            offset = end;
        }
        let json = serde_json::Value::Object(header).to_string();
        let mut out = Vec::with_capacity(8 + json.len() + offset);
        out.extend_from_slice(&(json.len() as u64).to_le_bytes());
        out.extend_from_slice(json.as_bytes());
        for e in entries {
            out.extend_from_slice(&e.data);
        }
        std::fs::write(path, out)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::{LmHead, SHARD_A, SHARD_B, Tweaks, write_set};
    use super::*;

    /// A fresh directory per test, named after the case so a failure leaves
    /// something inspectable behind.
    fn scratch(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xwen_qwen3_{}_{label}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn open_intact(label: &str) -> (PathBuf, Result<Qwen3Set>) {
        open_with(label, &Tweaks::intact(), &[])
    }

    fn open_with(label: &str, tweaks: &Tweaks, allow: &[&str]) -> (PathBuf, Result<Qwen3Set>) {
        let dir = scratch(label);
        write_set(&dir, tweaks).unwrap();
        let set = Qwen3Set::open(&dir, None, allow);
        (dir, set)
    }

    #[test]
    fn an_intact_set_opens_and_reports_a_clean_scan() {
        let (dir, set) = open_intact("intact");
        let set = set.unwrap();
        assert_eq!(set.config().n_layer, fixture::LAYERS);
        assert_eq!(set.config().hidden_size, fixture::HIDDEN);
        assert_eq!(set.config().head_dim(), 128);
        assert!(!set.has_lm_head());
        assert!(set.zero_runs().is_empty());
        let scan = set.range_scan();
        assert_eq!(scan.below_f16_subnormal, 0);
        assert_eq!(scan.above_f16_max, 0);
        assert!(scan.elements > 0);
        assert_eq!(set.tokenizer_path(), dir.join("tokenizer.json"));
        assert_eq!(set.shard_paths().len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_all_returns_bf16_projections_and_f32_norms() {
        let (dir, set) = open_intact("load_all");
        let set = set.unwrap();
        let w = set.load_all(&Device::Cpu).unwrap();
        assert_eq!(w.embed_tokens.dtype(), DType::BF16);
        assert_eq!(w.embed_tokens.dims(), &[fixture::VOCAB, fixture::HIDDEN]);
        assert_eq!(w.norm.dtype(), DType::F32);
        assert_eq!(w.norm.dims(), &[fixture::HIDDEN]);
        // Tied on every qwen3 checkpoint, so the head is the embedding.
        assert!(w.lm_head.is_none());
        assert_eq!(w.layers.len(), fixture::LAYERS);
        let q = fixture::HEADS * fixture::HEAD_DIM;
        let kv = fixture::KV_HEADS * fixture::HEAD_DIM;
        for layer in &w.layers {
            assert_eq!(layer.q_proj.dims(), &[q, fixture::HIDDEN]);
            assert_eq!(layer.k_proj.dims(), &[kv, fixture::HIDDEN]);
            assert_eq!(layer.v_proj.dims(), &[kv, fixture::HIDDEN]);
            assert_eq!(layer.o_proj.dims(), &[fixture::HIDDEN, q]);
            assert_eq!(
                layer.gate_proj.dims(),
                &[fixture::INTERMEDIATE, fixture::HIDDEN]
            );
            assert_eq!(
                layer.up_proj.dims(),
                &[fixture::INTERMEDIATE, fixture::HIDDEN]
            );
            assert_eq!(
                layer.down_proj.dims(),
                &[fixture::HIDDEN, fixture::INTERMEDIATE]
            );
            for t in [
                &layer.q_proj,
                &layer.k_proj,
                &layer.v_proj,
                &layer.o_proj,
                &layer.gate_proj,
                &layer.up_proj,
                &layer.down_proj,
            ] {
                assert_eq!(t.dtype(), DType::BF16);
            }
            for t in [
                &layer.input_layernorm,
                &layer.post_attention_layernorm,
                &layer.q_norm,
                &layer.k_norm,
            ] {
                assert_eq!(t.dtype(), DType::F32);
            }
            assert_eq!(layer.q_norm.dims(), &[fixture::HEAD_DIM]);
            assert_eq!(layer.k_norm.dims(), &[fixture::HEAD_DIM]);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The loaded values are the stored ones: candle's `load` copies the mapped
    /// bytes and keeps the dtype, it does not reinterpret or requantize them.
    #[test]
    fn loaded_bytes_are_the_stored_bytes() {
        let (dir, set) = open_intact("bytes");
        let set = set.unwrap();
        let name = "model.layers.0.self_attn.q_proj.weight";
        let stored: Vec<u8> = {
            let bytes = std::fs::read(dir.join(SHARD_A)).unwrap();
            let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
            let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + hlen]).unwrap();
            let off = &header[name]["data_offsets"];
            let (a, b) = (
                off[0].as_u64().unwrap() as usize,
                off[1].as_u64().unwrap() as usize,
            );
            bytes[8 + hlen + a..8 + hlen + b].to_vec()
        };
        let t = set.tensor_set(&Device::Cpu).take(name).unwrap();
        let loaded: Vec<half::bf16> = t.flatten_all().unwrap().to_vec1().unwrap();
        let expected: Vec<half::bf16> = stored
            .chunks_exact(2)
            .map(|w| half::bf16::from_bits(u16::from_le_bytes([w[0], w[1]])))
            .collect();
        assert_eq!(loaded, expected);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_tensor_is_named() {
        let name = "model.layers.1.mlp.up_proj.weight";
        let (dir, set) = open_with(
            "missing",
            &Tweaks {
                drop: Some(name),
                ..Tweaks::intact()
            },
            &[],
        );
        let err = set.unwrap_err().to_string();
        assert!(err.contains(name), "{err}");
        assert!(err.contains("missing"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_tensor_in_the_wrong_shard_is_refused() {
        let name = "model.layers.0.self_attn.k_proj.weight";
        let (dir, set) = open_with(
            "wrong_shard",
            &Tweaks {
                wrong_shard: Some(name),
                ..Tweaks::intact()
            },
            &[],
        );
        let err = set.unwrap_err().to_string();
        assert!(err.contains(name), "{err}");
        assert!(err.contains("does not contain it"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_tensor_in_two_shards_is_refused() {
        let name = "model.layers.0.mlp.gate_proj.weight";
        let (dir, set) = open_with(
            "duplicate",
            &Tweaks {
                duplicate: Some(name),
                ..Tweaks::intact()
            },
            &[],
        );
        let err = set.unwrap_err().to_string();
        assert!(err.contains(name), "{err}");
        assert!(err.contains(SHARD_A) && err.contains(SHARD_B), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_tensor_the_index_does_not_list_is_refused() {
        let (dir, set) = open_with(
            "unlisted",
            &Tweaks {
                unlisted: Some("model.layers.0.self_attn.rotary_emb.inv_freq"),
                ..Tweaks::intact()
            },
            &[],
        );
        let err = set.unwrap_err().to_string();
        assert!(err.contains("inv_freq"), "{err}");
        assert!(err.contains("does not list"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A tensor the config does not imply — here a layer beyond the declared
    /// count — is refused even when the index and the shard agree about it.
    #[test]
    fn a_tensor_outside_the_config_is_refused() {
        let (dir, set) = open_with(
            "extra",
            &Tweaks {
                extra: Some("model.layers.9.input_layernorm.weight"),
                ..Tweaks::intact()
            },
            &[],
        );
        let err = set.unwrap_err().to_string();
        assert!(
            err.contains("model.layers.9.input_layernorm.weight"),
            "{err}"
        );
        assert!(err.contains("not part of a qwen3 checkpoint"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_non_bf16_tensor_is_refused_by_name_and_dtype() {
        let name = "model.layers.1.self_attn.v_proj.weight";
        let (dir, set) = open_with(
            "dtype",
            &Tweaks {
                dtype: Some((name, "F16")),
                ..Tweaks::intact()
            },
            &[],
        );
        let err = set.unwrap_err().to_string();
        assert!(err.contains(name), "{err}");
        assert!(err.contains("F16"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_wrong_shape_is_refused_with_both_shapes() {
        let name = "model.layers.0.mlp.down_proj.weight";
        let (dir, set) = open_with(
            "shape",
            &Tweaks {
                shape: Some((name, vec![fixture::HIDDEN, fixture::INTERMEDIATE + 8])),
                ..Tweaks::intact()
            },
            &[],
        );
        let err = set.unwrap_err().to_string();
        assert!(err.contains(name), "{err}");
        assert!(err.contains("104"), "{err}");
        assert!(err.contains("96"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_long_zero_run_is_refused_unless_allowlisted() {
        let name = "model.layers.0.self_attn.q_proj.weight";
        let planted = (name, 1234usize, 5000usize);
        let (dir, set) = open_with(
            "zero_run",
            &Tweaks {
                zero_run: Some(planted),
                ..Tweaks::intact()
            },
            &[],
        );
        let err = set.unwrap_err().to_string();
        assert!(err.contains(name), "{err}");
        assert!(err.contains("5000 consecutive zero"), "{err}");
        assert!(err.contains("element 1234"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);

        let (dir, set) = open_with(
            "zero_run_allowed",
            &Tweaks {
                zero_run: Some(planted),
                ..Tweaks::intact()
            },
            &[name],
        );
        let set = set.unwrap();
        assert_eq!(
            set.zero_runs(),
            &[(
                name.to_string(),
                ZeroRun {
                    start: 1234,
                    len: 5000
                }
            )]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A run at the limit is not a hit; one element more is.
    #[test]
    fn the_zero_run_limit_is_exclusive() {
        let name = "model.layers.0.self_attn.q_proj.weight";
        let (dir, set) = open_with(
            "zero_run_at_limit",
            &Tweaks {
                zero_run: Some((name, 7, ZERO_RUN_LIMIT)),
                ..Tweaks::intact()
            },
            &[],
        );
        assert!(set.is_ok());
        let _ = std::fs::remove_dir_all(&dir);

        let (dir, set) = open_with(
            "zero_run_over_limit",
            &Tweaks {
                zero_run: Some((name, 7, ZERO_RUN_LIMIT + 1)),
                ..Tweaks::intact()
            },
            &[],
        );
        assert!(set.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_tied_lm_head_that_matches_the_embedding_is_accepted() {
        let (dir, set) = open_with(
            "lm_head_tied",
            &Tweaks {
                lm_head: Some(LmHead::Tied),
                ..Tweaks::intact()
            },
            &[],
        );
        let set = set.unwrap();
        assert!(set.has_lm_head());
        // Still tied: the head is the embedding, so nothing extra is loaded.
        let w = set.load_all(&Device::Cpu).unwrap();
        assert!(w.lm_head.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_lm_head_that_differs_from_the_embedding_is_refused() {
        let (dir, set) = open_with(
            "lm_head_divergent",
            &Tweaks {
                lm_head: Some(LmHead::Divergent),
                ..Tweaks::intact()
            },
            &[],
        );
        let err = set.unwrap_err().to_string();
        assert!(err.contains("lm_head.weight"), "{err}");
        assert!(err.contains("contradicts itself"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn taking_a_tensor_twice_is_refused_and_a_leftover_fails_finish() {
        let (dir, set) = open_intact("ledger");
        let set = set.unwrap();
        let mut ts = set.tensor_set(&Device::Cpu);
        let total = ts.remaining();
        ts.take("model.norm.weight").unwrap();
        assert_eq!(ts.remaining(), total - 1);
        let err = ts.take("model.norm.weight").unwrap_err().to_string();
        assert!(err.contains("already taken"), "{err}");
        let err = ts.finish().unwrap_err().to_string();
        assert!(err.contains("never loaded"), "{err}");
        assert!(err.contains("model.embed_tokens.weight"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_tokenizer_is_an_error_naming_the_paths_tried() {
        let dir = scratch("no_tokenizer");
        write_set(
            &dir,
            &Tweaks {
                tokenizer: false,
                ..Tweaks::intact()
            },
        )
        .unwrap();
        let err = Qwen3Set::open(&dir, None, &[]).unwrap_err().to_string();
        assert!(err.contains("tokenizer.json"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The Z-Image layout: weights in `text_encoder/`, tokenizer in a sibling
    /// `tokenizer/`.
    #[test]
    fn a_sibling_tokenizer_directory_is_found() {
        let root = scratch("sibling_tokenizer");
        let weights = root.join("text_encoder");
        write_set(
            &weights,
            &Tweaks {
                tokenizer: false,
                ..Tweaks::intact()
            },
        )
        .unwrap();
        std::fs::create_dir_all(root.join("tokenizer")).unwrap();
        std::fs::write(root.join("tokenizer/tokenizer.json"), b"{}").unwrap();
        let set = Qwen3Set::open(&weights, None, &[]).unwrap();
        assert_eq!(
            set.tokenizer_path(),
            root.join("tokenizer").join("tokenizer.json")
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_explicit_tokenizer_path_wins() {
        let (dir, set) = open_intact("tokenizer_override");
        drop(set);
        let elsewhere = dir.join("other.json");
        std::fs::write(&elsewhere, b"{}").unwrap();
        let set = Qwen3Set::open(&dir, Some(elsewhere.clone()), &[]).unwrap();
        assert_eq!(set.tokenizer_path(), elsewhere);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The id is a function of the metadata, so the same set twice gives the
    /// same id and a changed config or a changed shard gives a different one.
    #[test]
    fn the_checkpoint_id_is_stable_and_discriminating() {
        let (a, set_a) = open_intact("ckpt_a");
        let (b, set_b) = open_intact("ckpt_b");
        let (set_a, set_b) = (set_a.unwrap(), set_b.unwrap());
        assert_eq!(set_a.checkpoint_id(), set_b.checkpoint_id());
        assert_eq!(set_a.checkpoint_id().dir_name().len(), 16);
        // The length is every file in full, not just the metadata that is hashed.
        let on_disk: u64 = std::fs::read_dir(&a)
            .unwrap()
            .filter_map(|e| {
                let e = e.ok()?;
                let name = e.file_name().into_string().ok()?;
                (name != "tokenizer.json").then(|| e.metadata().ok())?
            })
            .map(|m| m.len())
            .sum();
        assert_eq!(set_a.checkpoint_id().file_len(), on_disk);

        let (c, set_c) = open_with(
            "ckpt_c",
            &Tweaks {
                lm_head: Some(LmHead::Tied),
                ..Tweaks::intact()
            },
            &[],
        );
        assert_ne!(set_a.checkpoint_id(), set_c.unwrap().checkpoint_id());

        // Each of the two JSON files is in the hash on its own. Both edits keep
        // the byte count identical, so it is the hash and not the length term
        // that moves.
        let edit = |dir: &Path, file: &str, from: &str, to: &str| {
            assert_eq!(from.len(), to.len(), "the edit must not change the length");
            let path = dir.join(file);
            let before = std::fs::read_to_string(&path).unwrap();
            assert!(before.contains(from), "{file} has no {from} to change");
            let after = before.replace(from, to);
            assert_eq!(before.len(), after.len());
            std::fs::write(&path, after).unwrap();
        };

        let (d, _) = open_intact("ckpt_config_edit");
        edit(
            &d,
            "config.json",
            "\"max_position_embeddings\":4096",
            "\"max_position_embeddings\":8192",
        );
        let edited = Qwen3Set::open(&d, None, &[]).unwrap();
        assert_ne!(set_a.checkpoint_id().hash(), edited.checkpoint_id().hash());
        assert_eq!(
            set_a.checkpoint_id().file_len(),
            edited.checkpoint_id().file_len()
        );

        let (e, _) = open_intact("ckpt_index_edit");
        edit(
            &e,
            "model.safetensors.index.json",
            "\"metadata\":{\"total_size\"",
            "\"metadata\":{\"TOTAL_size\"",
        );
        let edited = Qwen3Set::open(&e, None, &[]).unwrap();
        assert_ne!(set_a.checkpoint_id().hash(), edited.checkpoint_id().hash());
        assert_eq!(
            set_a.checkpoint_id().file_len(),
            edited.checkpoint_id().file_len()
        );

        for dir in [a, b, c, d, e] {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    /// The f16-range counters see values on both sides of the range, and a
    /// zero is neither.
    #[test]
    fn the_range_scan_counts_both_tails() {
        let dir = scratch("range");
        write_set(&dir, &Tweaks::intact()).unwrap();
        // Plant three values in one projection: below f16's subnormal floor,
        // above f16's max, and an exact zero that must count as neither.
        let name = "model.layers.1.mlp.up_proj.weight";
        let shard = dir.join(SHARD_B);
        let mut bytes = std::fs::read(&shard).unwrap();
        let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + hlen]).unwrap();
        let start = 8 + hlen + header[name]["data_offsets"][0].as_u64().unwrap() as usize;
        // 2^-30, well under f16's 2^-24 floor.
        let tiny = half::bf16::from_f32(2f32.powi(-30)).to_bits();
        // 2^17, over f16's 65504 ceiling.
        let huge = half::bf16::from_f32(2f32.powi(17)).to_bits();
        bytes[start..start + 2].copy_from_slice(&tiny.to_le_bytes());
        bytes[start + 2..start + 4].copy_from_slice(&huge.to_le_bytes());
        bytes[start + 4..start + 6].copy_from_slice(&0u16.to_le_bytes());
        std::fs::write(&shard, bytes).unwrap();

        let set = Qwen3Set::open(&dir, None, &[]).unwrap();
        let scan = set.range_scan();
        assert_eq!(scan.below_f16_subnormal, 1);
        assert_eq!(scan.above_f16_max, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The scan's answer must not depend on how the plane was split: a run that
    /// spans a chunk boundary, or spans whole chunks, is one run. Driven at
    /// several chunk sizes because the production split is not observable from
    /// the outside.
    #[test]
    fn the_scan_does_not_depend_on_the_chunk_size() {
        let n = 1 << 16;
        let one = half::bf16::from_f32(1.0).to_bits().to_le_bytes();
        let plane = |runs: &[(usize, usize)]| {
            let mut bytes: Vec<u8> = (0..n).flat_map(|_| one).collect();
            for &(start, len) in runs {
                bytes[start * 2..(start + len) * 2].fill(0);
            }
            bytes
        };
        let cases: [(&[(usize, usize)], Option<ZeroRun>); 6] = [
            // Spans every chunk at every size below the plane.
            (
                &[(1, n - 2)],
                Some(ZeroRun {
                    start: 1,
                    len: n - 2,
                }),
            ),
            // The whole plane.
            (&[(0, n)], Some(ZeroRun { start: 0, len: n })),
            // Two runs: the longest wins wherever it sits.
            (
                &[(10, 100), (n / 2 - 50, 4000)],
                Some(ZeroRun {
                    start: n / 2 - 50,
                    len: 4000,
                }),
            ),
            // One run ending just before element 2048 (a 4096-byte boundary at
            // several of the chunk sizes below) and a longer one starting on it,
            // with a nonzero element between them.
            (
                &[(2048 - 31, 30), (2048, 700)],
                Some(ZeroRun {
                    start: 2048,
                    len: 700,
                }),
            ),
            // Runs with nothing between them are one run, whichever side of a
            // boundary each half falls on.
            (
                &[(2048 - 30, 30), (2048, 700)],
                Some(ZeroRun {
                    start: 2048 - 30,
                    len: 730,
                }),
            ),
            (&[], None),
        ];
        for (runs, want) in cases {
            let bytes = plane(runs);
            for chunk in [2usize, 4096, 8192, 30000, bytes.len(), bytes.len() * 2] {
                let got = scan_one(&bytes, chunk);
                assert_eq!(got.run, want, "runs {runs:?} at chunk size {chunk}");
                assert_eq!(got.elements, n as u64, "chunk size {chunk}");
            }
        }
        assert_eq!(scan_one(&[], 4096).run, None);
    }

    /// BF16 negative zero is a zero: a plane filled with `0x8000` is as unwritten
    /// as one filled with `0x0000`, and a run made of both is one run.
    #[test]
    fn signed_zero_counts_as_zero_in_the_run_scan() {
        let n = 4000usize;
        let mut bytes: Vec<u8> = Vec::with_capacity(n * 2);
        for i in 0..n {
            let bits = match i {
                0 => half::bf16::from_f32(1.0).to_bits(),
                // Alternating positive and negative zero, one run.
                i if i % 2 == 0 => 0x0000,
                _ => half::bf16::NEG_ZERO.to_bits(),
            };
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        let scan = scan_one(&bytes, 512);
        assert_eq!(
            scan.run,
            Some(ZeroRun {
                start: 1,
                len: n - 1
            })
        );
        // Neither zero is on either f16 tail.
        assert_eq!(scan.below, 0);
        assert_eq!(scan.above, 0);
    }

    /// The two f16-range bounds, at the BF16 values that sit either side of them.
    ///
    /// 65504 is f16's largest finite value but is NOT a BF16 value: with seven
    /// mantissa bits the grid near there is 65280 then 65536, so the test uses
    /// those two. Infinity and NaN sort above the bound and count with the
    /// overflows; a NaN narrowed to f16 stays a NaN rather than becoming an
    /// infinity, which is why the counter is named for the range and not for
    /// what the value becomes.
    #[test]
    fn the_range_scan_bounds_are_exact() {
        let cases: [(&str, u16, bool, bool); 9] = [
            // (label, bits, counts below, counts above)
            ("largest bf16 under 2^-24", 0x337F, true, false),
            ("exactly 2^-24", 0x3380, false, false),
            ("just over 2^-24", 0x3381, false, false),
            ("1.0", half::bf16::from_f32(1.0).to_bits(), false, false),
            (
                "65280, the largest bf16 under f16 max",
                0x477F,
                false,
                false,
            ),
            ("65536, the first bf16 over f16 max", 0x4780, false, true),
            ("+inf", half::bf16::INFINITY.to_bits(), false, true),
            ("-inf", half::bf16::NEG_INFINITY.to_bits(), false, true),
            ("NaN", half::bf16::NAN.to_bits(), false, true),
        ];
        for (label, bits, below, above) in cases {
            for sign in [0u16, 0x8000] {
                // NaN and infinity already carry their own sign in the table;
                // flipping it must not change which tail they land on.
                let word = (bits ^ sign).to_le_bytes();
                let scan = scan_one(&word, 2);
                assert_eq!(
                    scan.below,
                    u64::from(below),
                    "{label} (sign bit {sign:#06x})"
                );
                assert_eq!(
                    scan.above,
                    u64::from(above),
                    "{label} (sign bit {sign:#06x})"
                );
                assert_eq!(scan.elements, 1, "{label}");
            }
        }
        // 65280 is representable in f16 and 65536 is not, which is what the
        // bound is asserting about.
        assert_eq!(half::bf16::from_bits(0x477F).to_f32(), 65280.0);
        assert_eq!(half::bf16::from_bits(0x4780).to_f32(), 65536.0);
    }

    /// The fixture's four widths are all different, so a projection stored
    /// transposed, or a q plane where a kv plane belongs, cannot pass the shape
    /// table by coincidence.
    #[test]
    fn the_fixture_geometry_distinguishes_every_projection_width() {
        let cfg = Qwen3Config::from_json_bytes(fixture::config_json().as_bytes()).unwrap();
        let widths = [
            cfg.hidden_size,
            cfg.q_dim(),
            cfg.kv_dim(),
            cfg.intermediate_size,
        ];
        for (i, a) in widths.iter().enumerate() {
            for b in &widths[i + 1..] {
                assert_ne!(a, b, "fixture widths must all differ: {widths:?}");
            }
        }
    }

    #[test]
    fn a_transposed_projection_is_refused() {
        let name = "model.layers.0.self_attn.q_proj.weight";
        let q = fixture::HEADS * fixture::HEAD_DIM;
        let (dir, set) = open_with(
            "transposed",
            &Tweaks {
                // q_proj stored as o_proj's shape.
                shape: Some((name, vec![fixture::HIDDEN, q])),
                ..Tweaks::intact()
            },
            &[],
        );
        let err = set.unwrap_err().to_string();
        assert!(err.contains(name), "{err}");
        assert!(err.contains("expected [256, 64]"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_index_that_lists_a_tensor_twice_is_refused() {
        let name = "model.layers.0.mlp.up_proj.weight";
        let (dir, set) = open_with(
            "dup_index_key",
            &Tweaks {
                duplicate_index_key: Some(name),
                ..Tweaks::intact()
            },
            &[],
        );
        let err = format!("{:#}", set.unwrap_err());
        assert!(err.contains(name), "{err}");
        assert!(err.contains("twice"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_safetensors_file_the_index_does_not_name_is_refused() {
        let (dir, set) = open_with(
            "stray_shard",
            &Tweaks {
                stray_shard: true,
                ..Tweaks::intact()
            },
            &[],
        );
        let err = set.unwrap_err().to_string();
        assert!(err.contains("model-00003-of-00002.safetensors"), "{err}");
        assert!(err.contains("does not reference"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A failed load leaves its plane outstanding, so `finish` still catches it.
    #[test]
    fn a_load_that_fails_does_not_clear_the_ledger() {
        let (dir, set) = open_intact("failed_take");
        let set = set.unwrap();
        let mut ts = set.tensor_set(&Device::Cpu);
        let before = ts.remaining();
        assert!(ts.take("model.layers.0.mlp.no_such_plane.weight").is_err());
        assert_eq!(ts.remaining(), before);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- Real checkpoints -------------------------------------------------
    //
    // These read the whole 8 GB set from the mmap once, so they are one pass
    // per file and nothing more. They skip themselves when the HF cache does
    // not have the checkpoint: the weights are not a checkout dependency.

    fn snapshot(repo: &str, sha: &str) -> Option<PathBuf> {
        let home = std::env::var_os("HOME")?;
        let dir = PathBuf::from(home)
            .join(".cache/huggingface/hub")
            .join(repo)
            .join("snapshots")
            .join(sha);
        dir.is_dir().then_some(dir)
    }

    /// The Z-Image-Turbo `text_encoder/` copy of Qwen3-4B has two zero-filled
    /// planes in layer 35. Harmless for the encoder, which reads
    /// `hidden_states[-2]` and never evaluates that layer's MLP, but the copy is
    /// not a faithful full LM and the loader has to say so rather than run it.
    #[test]
    fn the_zimage_text_encoder_reports_its_two_corrupt_planes() {
        let Some(root) = snapshot(
            "models--Tongyi-MAI--Z-Image-Turbo",
            "f332072aa78be7aecdf3ee76d5c247082da564a6",
        ) else {
            eprintln!("skipping: Z-Image-Turbo is not in the local HF cache");
            return;
        };
        let dir = root.join("text_encoder");
        let corrupt = [
            "model.layers.35.mlp.up_proj.weight",
            "model.layers.35.mlp.down_proj.weight",
        ];

        // Without the allowlist the set is refused, naming one of them.
        let err = Qwen3Set::open(&dir, None, &[]).unwrap_err().to_string();
        assert!(
            corrupt.iter().any(|n| err.contains(n)),
            "expected a zero-run error naming a layer-35 mlp plane, got: {err}"
        );

        let set = Qwen3Set::open(&dir, None, &corrupt).unwrap();
        let mut runs: Vec<(&str, ZeroRun)> = set
            .zero_runs()
            .iter()
            .map(|(n, r)| (n.as_str(), *r))
            .collect();
        runs.sort();
        assert_eq!(
            runs,
            [
                (
                    "model.layers.35.mlp.down_proj.weight",
                    ZeroRun {
                        start: 20_930_265,
                        len: 3_938_425
                    }
                ),
                (
                    "model.layers.35.mlp.up_proj.weight",
                    ZeroRun {
                        start: 27_003,
                        len: 14_772_816
                    }
                ),
            ]
        );
        assert_eq!(set.config().n_layer, 36);
        assert_eq!(set.config().rope.theta, 1e6);
        assert!(!set.has_lm_head());
        // The tokenizer lives in the repo's sibling `tokenizer/` directory.
        assert_eq!(
            set.tokenizer_path(),
            root.join("tokenizer").join("tokenizer.json")
        );
        eprintln!(
            "Z-Image text encoder f16 range scan: {:?} over {} elements",
            set.range_scan(),
            set.range_scan().elements
        );
    }

    /// The checkpoint the Z-Image copy was made from is intact: an empty
    /// allowlist opens it.
    #[test]
    fn the_base_checkpoint_has_no_zero_runs() {
        let Some(dir) = snapshot(
            "models--Qwen--Qwen3-4B",
            "1cfa9a7208912126459214e8b04321603b3df60c",
        ) else {
            eprintln!("skipping: Qwen/Qwen3-4B is not in the local HF cache");
            return;
        };
        let set = Qwen3Set::open(&dir, None, &[]).unwrap();
        assert!(set.zero_runs().is_empty());
        assert_eq!(set.config().n_layer, 36);
        assert_eq!(set.config().vocab_size, 151936);
        assert_eq!(set.config().rope.theta, 1e6);
        assert!(!set.has_lm_head());
        assert_eq!(set.tokenizer_path(), dir.join("tokenizer.json"));
        assert_eq!(set.shard_paths().len(), 3);
        eprintln!(
            "Qwen3-4B f16 range scan: {:?} over {} elements",
            set.range_scan(),
            set.range_scan().elements
        );
    }
}
