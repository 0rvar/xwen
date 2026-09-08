//! Load-time transformer LoRA merging. Base weights stay immutable between loads.
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::{VarBuilder, var_builder::SimpleBackend};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::UNIX_EPOCH,
};

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LoraSpec {
    pub name: String,
    pub weight: f64,
}

#[derive(Clone, Debug)]
pub struct AdapterError(pub String);
impl std::fmt::Display for AdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for AdapterError {}

/// File identity used together with weight when deciding whether a pipeline can be reused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fingerprint {
    pub bytes: u64,
    pub modified_ns: u128,
    pub inode: u64,
    pub changed_s: i64,
    pub changed_ns: i64,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedLora {
    pub path: PathBuf,
    pub weight: f64,
    pub fingerprint: Fingerprint,
}

impl LoraSpec {
    pub fn resolve(&self) -> Result<ResolvedLora> {
        ensure!(!self.name.is_empty(), "LoRA name must not be empty");
        ensure!(self.weight.is_finite(), "LoRA weight must be finite");
        let direct = Path::new(&self.name);
        let path = if direct.is_file() {
            direct.to_path_buf()
        } else {
            ensure!(
                direct.components().count() == 1,
                "LoRA file does not exist: {}",
                self.name
            );
            let dir = match std::env::var_os("XWEN_LORA_DIR") {
                Some(p) => PathBuf::from(p),
                None => PathBuf::from(
                    std::env::var_os("HOME").context("HOME is unset; set XWEN_LORA_DIR")?,
                )
                .join(".local/share/xwen/loras"),
            };
            let exact = dir.join(&self.name);
            if exact.is_file() {
                exact
            } else {
                dir.join(format!("{}.safetensors", self.name))
            }
        };
        let path = path
            .canonicalize()
            .with_context(|| format!("resolving LoRA {}", path.display()))?;
        let metadata = std::fs::metadata(&path)?;
        ensure!(
            metadata.is_file(),
            "LoRA must be a file: {}",
            path.display()
        );
        use std::os::unix::fs::MetadataExt;
        validate_header(&path, metadata.len())?;
        let fingerprint = Fingerprint {
            bytes: metadata.len(),
            modified_ns: metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos(),
            inode: metadata.ino(),
            changed_s: metadata.ctime(),
            changed_ns: metadata.ctime_nsec(),
        };
        Ok(ResolvedLora {
            path,
            weight: self.weight,
            fingerprint,
        })
    }
}

#[derive(Clone)]
struct Delta {
    a: Tensor,
    b: Tensor,
    scale: f64,
    label: String,
}
#[derive(Default)]
struct Pair {
    a: Option<Tensor>,
    b: Option<Tensor>,
    alpha: Option<Tensor>,
}

fn canonical_module(raw: &str) -> Result<String> {
    let mut name = raw;
    for prefix in ["model.diffusion_model.", "diffusion_model.", "transformer."] {
        if let Some(rest) = name.strip_prefix(prefix) {
            name = rest;
            break;
        }
    }
    ensure!(
        !name.starts_with("lora_unet_"),
        "unsupported underscored LoRA key {raw}; export diffusers keys"
    );
    let name = if let Some(base) = name.strip_suffix(".attention.out") {
        format!("{base}.attention.to_out.0")
    } else {
        name.to_owned()
    };
    ensure!(!name.is_empty(), "empty LoRA module");
    Ok(name)
}

fn finite(t: &Tensor, label: &str) -> Result<()> {
    ensure!(
        matches!(
            t.dtype(),
            DType::F32 | DType::F16 | DType::BF16 | DType::F64
        ),
        "LoRA {label} must be floating point"
    );
    ensure!(
        t.to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| v.is_finite()),
        "LoRA {label} contains nonfinite values"
    );
    Ok(())
}

fn parse(
    tensors: HashMap<String, Tensor>,
    weight: f64,
    source: &str,
) -> Result<BTreeMap<String, Vec<Delta>>> {
    ensure!(weight.is_finite(), "LoRA weight must be finite");
    let mut pairs: BTreeMap<String, Pair> = BTreeMap::new();
    for (key, tensor) in tensors {
        let suffixes = [
            (".lora_A.default.weight", 0),
            (".lora_B.default.weight", 1),
            (".lora_A.weight", 0),
            (".lora_B.weight", 1),
            (".lora_down.weight", 0),
            (".lora_up.weight", 1),
            (".alpha", 2),
        ];
        let (module, kind) = suffixes
            .iter()
            .find_map(|(suffix, k)| key.strip_suffix(suffix).map(|m| (m, *k)))
            .with_context(|| format!("unsupported LoRA tensor {key} in {source}"))?;
        finite(&tensor, &key)?;
        let pair = pairs.entry(canonical_module(module)?).or_default();
        let slot = match kind {
            0 => &mut pair.a,
            1 => &mut pair.b,
            _ => &mut pair.alpha,
        };
        ensure!(
            slot.is_none(),
            "duplicate LoRA tensor for {module} in {source}"
        );
        *slot = Some(tensor.to_dtype(DType::F32)?);
    }
    ensure!(!pairs.is_empty(), "LoRA {source} contains no adapter pairs");
    let mut out: BTreeMap<String, Vec<Delta>> = BTreeMap::new();
    for (module, pair) in pairs {
        let a = pair
            .a
            .with_context(|| format!("missing LoRA A/down for {module} in {source}"))?;
        let b = pair
            .b
            .with_context(|| format!("missing LoRA B/up for {module} in {source}"))?;
        let (rank, input) = a
            .dims2()
            .with_context(|| format!("LoRA A {module} must be rank-2"))?;
        let (output, b_rank) = b
            .dims2()
            .with_context(|| format!("LoRA B {module} must be rank-2"))?;
        ensure!(
            rank > 0 && input > 0 && output > 0 && rank == b_rank,
            "invalid LoRA ranks/shapes for {module}"
        );
        let alpha = match pair.alpha {
            Some(alpha) => {
                ensure!(
                    alpha.elem_count() == 1,
                    "LoRA alpha for {module} must be scalar"
                );
                alpha.flatten_all()?.to_vec1::<f32>()?[0] as f64 / rank as f64
            }
            None => 1.0,
        };
        let scale = weight * alpha;
        ensure!(
            scale.is_finite() && (scale as f32).is_finite(),
            "LoRA scale overflows for {module}"
        );
        if let Some(prefix) = module.strip_suffix(".attention.qkv") {
            ensure!(
                output % 3 == 0,
                "fused QKV LoRA rows must divide by three: {module}"
            );
            for (index, projection) in ["to_q", "to_k", "to_v"].iter().enumerate() {
                let key = format!("{prefix}.attention.{projection}.weight");
                ensure!(
                    !out.contains_key(&key),
                    "overlapping fused and split LoRA targets: {key}"
                );
                out.entry(key).or_default().push(Delta {
                    a: a.clone(),
                    b: b.narrow(0, index * output / 3, output / 3)?.contiguous()?,
                    scale,
                    label: format!("{source}:{module}/{projection}"),
                });
            }
        } else {
            ensure!(
                !out.contains_key(&format!("{module}.weight")),
                "overlapping fused and split LoRA targets: {module}"
            );
            out.entry(format!("{module}.weight"))
                .or_default()
                .push(Delta {
                    a,
                    b,
                    scale,
                    label: format!("{source}:{module}"),
                });
        }
    }
    Ok(out)
}

/// Validated CPU adapters, prepared before loading the base transformer.
pub struct PreparedLoras {
    resolved: Vec<ResolvedLora>,
    deltas: BTreeMap<String, Vec<Delta>>,
}
impl PreparedLoras {
    pub fn prepare(specs: &[LoraSpec]) -> Result<Self> {
        let resolved = specs
            .iter()
            .map(LoraSpec::resolve)
            .collect::<Result<Vec<_>>>()?;
        Self::load(&resolved)
    }
    pub fn load(resolved: &[ResolvedLora]) -> Result<Self> {
        let mut deltas: BTreeMap<String, Vec<Delta>> = BTreeMap::new();
        for spec in resolved {
            let tensors = candle_core::safetensors::load(&spec.path, &Device::Cpu)
                .with_context(|| format!("reading LoRA {}", spec.path.display()))?;
            let parsed = parse(tensors, spec.weight, &spec.path.display().to_string())?;
            // Ensure the identity still describes the file that was validated.
            let after = LoraSpec {
                name: spec.path.to_string_lossy().into_owned(),
                weight: spec.weight,
            }
            .resolve()?;
            ensure!(
                after == *spec,
                "LoRA file changed while loading: {}",
                spec.path.display()
            );
            for (name, updates) in parsed {
                deltas.entry(name).or_default().extend(updates);
            }
        }
        Ok(Self {
            resolved: resolved.to_vec(),
            deltas,
        })
    }
    pub fn resolved(&self) -> &[ResolvedLora] {
        &self.resolved
    }
    pub fn is_empty(&self) -> bool {
        self.deltas.is_empty()
    }
    pub fn wrap<'a>(&self, base: VarBuilder<'a>) -> (VarBuilder<'a>, MergeCheck) {
        let check = MergeCheck {
            expected: self.deltas.keys().cloned().collect(),
            consumed: Arc::new(Mutex::new(BTreeSet::new())),
            request_error: Arc::new(Mutex::new(None)),
        };
        let dtype = base.dtype();
        let device = base.device().clone();
        let overlay = Overlay {
            base,
            deltas: self.deltas.clone(),
            consumed: check.consumed.clone(),
            request_error: check.request_error.clone(),
        };
        (
            VarBuilder::from_backend(Box::new(overlay), dtype, device),
            check,
        )
    }
}

pub struct MergeCheck {
    expected: BTreeSet<String>,
    consumed: Arc<Mutex<BTreeSet<String>>>,
    request_error: Arc<Mutex<Option<AdapterError>>>,
}
impl MergeCheck {
    pub fn request_error(&self) -> Option<AdapterError> {
        self.request_error
            .lock()
            .ok()
            .and_then(|error| error.clone())
    }
    pub fn finish(&self) -> Result<()> {
        let consumed = self
            .consumed
            .lock()
            .map_err(|_| anyhow::anyhow!("LoRA consumed lock poisoned"))?;
        let unused = self
            .expected
            .difference(&consumed)
            .cloned()
            .collect::<Vec<_>>();
        if !unused.is_empty() {
            return Err(AdapterError(format!(
                "LoRA targets were not loaded by the transformer: {}",
                unused.join(", ")
            ))
            .into());
        }
        Ok(())
    }
}
struct Overlay<'a> {
    base: VarBuilder<'a>,
    deltas: BTreeMap<String, Vec<Delta>>,
    consumed: Arc<Mutex<BTreeSet<String>>>,
    request_error: Arc<Mutex<Option<AdapterError>>>,
}
impl Overlay<'_> {
    fn invalid_adapter(&self, message: String) -> candle_core::Error {
        if let Ok(mut error) = self.request_error.lock() {
            *error = Some(AdapterError(message.clone()));
        }
        candle_core::Error::Msg(message)
    }
    fn merge(
        &self,
        mut tensor: Tensor,
        name: &str,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        if let Some(updates) = self.deltas.get(name) {
            let (out, input) = tensor.dims2()?;
            for delta in updates {
                if delta.a.dim(1)? != input || delta.b.dim(0)? != out {
                    return Err(self.invalid_adapter(format!(
                        "LoRA {} does not match base {name} [{out}, {input}]",
                        delta.label
                    )));
                }
                if delta.scale != 0.0 {
                    let a = delta.a.to_device(dev)?;
                    let b = delta.b.to_device(dev)?;
                    tensor = (tensor + (b.matmul(&a)? * delta.scale)?)?;
                }
            }
            // CPU validation also catches overflow caused by otherwise finite factors.
            let values = tensor
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            if values
                .iter()
                .any(|v| !v.is_finite() || v.abs() > super::linear::F16_MAX)
            {
                return Err(self.invalid_adapter(format!(
                    "merged LoRA weight {name} is nonfinite or outside f16 range"
                )));
            }
            self.consumed
                .lock()
                .map_err(|_| candle_core::Error::Msg("LoRA consumed lock poisoned".into()))?
                .insert(name.to_owned());
        }
        tensor.to_dtype(dtype)
    }
}
impl SimpleBackend for Overlay<'_> {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        hints: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        if !self.deltas.contains_key(name) {
            return self
                .base
                .get_with_hints_dtype(shape, name, hints, dtype)?
                .to_device(dev);
        }
        let tensor = self
            .base
            .get_with_hints_dtype(shape, name, hints, DType::F32)?
            .to_device(dev)?;
        self.merge(tensor, name, dtype, dev)
    }
    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        if !self.deltas.contains_key(name) {
            return self.base.get_unchecked_dtype(name, dtype)?.to_device(dev);
        }
        self.merge(
            self.base
                .get_unchecked_dtype(name, DType::F32)?
                .to_device(dev)?,
            name,
            dtype,
            dev,
        )
    }
    fn contains_tensor(&self, name: &str) -> bool {
        self.base.contains_tensor(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn adapter(module: &str, alpha: Option<f32>) -> HashMap<String, Tensor> {
        let dev = &Device::Cpu;
        let mut tensors = HashMap::from([
            (
                format!("diffusion_model.{module}.lora_A.weight"),
                Tensor::new(&[[1f32, 2.]], dev).unwrap(),
            ),
            (
                format!("diffusion_model.{module}.lora_B.weight"),
                Tensor::new(&[[3f32], [4.]], dev).unwrap(),
            ),
        ]);
        if let Some(alpha) = alpha {
            tensors.insert(
                format!("diffusion_model.{module}.alpha"),
                Tensor::new(alpha, dev).unwrap(),
            );
        }
        tensors
    }
    fn base() -> VarBuilder<'static> {
        VarBuilder::from_tensors(
            HashMap::from([(
                "layers.0.attention.to_q.weight".into(),
                Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap(),
            )]),
            DType::F32,
            &Device::Cpu,
        )
    }
    fn prepared(tensors: HashMap<String, Tensor>, weight: f64) -> PreparedLoras {
        PreparedLoras {
            resolved: vec![],
            deltas: parse(tensors, weight, "fixture").unwrap(),
        }
    }
    #[test]
    fn split_attention_merge_scales_and_does_not_mutate_base() {
        let base = base();
        let lora = prepared(adapter("layers.0.attention.to_q", Some(2.)), 0.5);
        let (vb, check) = lora.wrap(base.clone());
        let w = vb.get((2, 2), "layers.0.attention.to_q.weight").unwrap();
        assert_eq!(
            w.to_vec2::<f32>().unwrap(),
            vec![vec![3., 6.], vec![4., 8.]]
        );
        check.finish().unwrap();
        assert_eq!(
            base.get((2, 2), "layers.0.attention.to_q.weight")
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![0., 0.], vec![0., 0.]]
        );
        let (zero, check) = prepared(adapter("layers.0.attention.to_q", None), 0.).wrap(base);
        assert_eq!(
            zero.get((2, 2), "layers.0.attention.to_q.weight")
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap(),
            0.
        );
        check.finish().unwrap();
    }
    #[test]
    fn finite_factors_that_overflow_the_merge_retain_request_classification() {
        let (vb, check) = prepared(adapter("layers.0.attention.to_q", None), 1e10).wrap(base());
        assert!(vb.get((2, 2), "layers.0.attention.to_q.weight").is_err());
        let error = check
            .request_error()
            .expect("adapter overflow is a request fault");
        assert!(error.0.contains("outside f16 range"));
    }
    #[test]
    fn missing_alpha_means_unit_scale_even_at_rank_two() {
        let mut t = HashMap::new();
        t.insert(
            "x.lora_A.weight".into(),
            Tensor::new(&[[1f32, 0.], [0., 1.]], &Device::Cpu).unwrap(),
        );
        t.insert(
            "x.lora_B.weight".into(),
            Tensor::new(&[[1f32, 0.], [0., 1.]], &Device::Cpu).unwrap(),
        );
        let parsed = parse(t, 1., "fixture").unwrap();
        assert_eq!(parsed["x.weight"][0].scale, 1.);
    }
    #[test]
    fn fused_qkv_splits_output_rows_without_rank_rescaling() {
        let dev = &Device::Cpu;
        let tensors = HashMap::from([
            (
                "transformer.layers.0.attention.qkv.lora_down.weight".into(),
                Tensor::new(&[[2f32, 3.]], dev).unwrap(),
            ),
            (
                "transformer.layers.0.attention.qkv.lora_up.weight".into(),
                Tensor::new(&[[1f32], [2.], [3.], [4.], [5.], [6.]], dev).unwrap(),
            ),
        ]);
        let p = parse(tensors, 1., "fixture").unwrap();
        for (name, expected) in [
            ("to_q", vec![vec![2., 3.], vec![4., 6.]]),
            ("to_k", vec![vec![6., 9.], vec![8., 12.]]),
            ("to_v", vec![vec![10., 15.], vec![12., 18.]]),
        ] {
            let d = &p[&format!("layers.0.attention.{name}.weight")][0];
            assert_eq!(
                d.b.matmul(&d.a).unwrap().to_vec2::<f32>().unwrap(),
                expected
            );
        }
    }
    #[test]
    fn invalid_and_unused_adapters_are_refused() {
        let mut missing = adapter("x", None);
        missing.remove("diffusion_model.x.lora_B.weight");
        assert!(parse(missing, 1., "fixture").is_err());
        assert!(parse(adapter("x", Some(f32::NAN)), 1., "fixture").is_err());
        assert!(parse(adapter("x", None), f64::INFINITY, "fixture").is_err());
        let (vb, check) = prepared(adapter("layers.99.attention.to_q", None), 1.).wrap(base());
        assert!(check.finish().is_err());
        drop(vb);
    }
}

/// Validate all header entries without materializing adapter planes on a request thread.
fn validate_header(path: &Path, file_len: u64) -> Result<()> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)?;
    let size = u64::from_le_bytes(prefix);
    ensure!(
        size <= 16 * 1024 * 1024 && size.checked_add(8).is_some_and(|v| v <= file_len),
        "invalid LoRA safetensors header size"
    );
    let mut bytes = vec![0u8; size as usize];
    file.read_exact(&mut bytes)?;
    let header: serde_json::Value = serde_json::from_slice(&bytes)?;
    let header = header
        .as_object()
        .context("LoRA safetensors header must be an object")?;
    let mut regions = Vec::new();
    let mut pairs: BTreeMap<String, [Option<Vec<u64>>; 3]> = BTreeMap::new();
    for (key, value) in header {
        if key == "__metadata__" {
            continue;
        }
        let suffixes = [
            (".lora_A.default.weight", 0),
            (".lora_B.default.weight", 1),
            (".lora_A.weight", 0),
            (".lora_B.weight", 1),
            (".lora_down.weight", 0),
            (".lora_up.weight", 1),
            (".alpha", 2),
        ];
        let (module, kind) = suffixes
            .iter()
            .find_map(|(s, k)| key.strip_suffix(s).map(|m| (m, *k)))
            .with_context(|| format!("unsupported LoRA tensor {key}"))?;
        let module = canonical_module(module)?;
        let shape = value["shape"]
            .as_array()
            .context("LoRA tensor shape is missing")?
            .iter()
            .map(|v| v.as_u64().context("invalid LoRA dimension"))
            .collect::<Result<Vec<_>>>()?;
        let element_bytes: u64 = match value["dtype"].as_str() {
            Some("F32") => 4,
            Some("BF16" | "F16") => 2,
            Some("F64") => 8,
            _ => anyhow::bail!("LoRA {key} must be floating point"),
        };
        let expected = shape
            .iter()
            .try_fold(element_bytes, |n, x| n.checked_mul(*x))
            .context("LoRA tensor shape overflows")?;
        let offsets = value["data_offsets"]
            .as_array()
            .context("missing LoRA tensor data offsets")?;
        ensure!(offsets.len() == 2, "invalid LoRA data offsets");
        let begin = offsets[0].as_u64().context("invalid LoRA data offset")?;
        let end = offsets[1].as_u64().context("invalid LoRA data offset")?;
        ensure!(
            end.checked_sub(begin) == Some(expected) && end <= file_len - size - 8,
            "LoRA tensor {key} lies outside file or has incorrect size"
        );
        regions.push((begin, end));
        let slots = pairs.entry(module).or_default();
        ensure!(slots[kind].is_none(), "duplicate LoRA pair entry {key}");
        slots[kind] = Some(shape);
    }
    ensure!(!pairs.is_empty(), "LoRA contains no adapter pairs");
    for (name, pair) in pairs {
        let a = pair[0]
            .as_ref()
            .with_context(|| format!("missing LoRA A/down for {name}"))?;
        let b = pair[1]
            .as_ref()
            .with_context(|| format!("missing LoRA B/up for {name}"))?;
        ensure!(
            a.len() == 2 && b.len() == 2 && a[0] > 0 && a[1] > 0 && b[0] > 0 && a[0] == b[1],
            "invalid LoRA ranks/shapes for {name}"
        );
        if let Some(alpha) = &pair[2] {
            ensure!(
                alpha.iter().product::<u64>() == 1,
                "LoRA alpha for {name} must be scalar"
            );
        }
        if name.ends_with(".attention.qkv") {
            ensure!(b[0] % 3 == 0, "fused QKV LoRA rows must divide by three");
        }
    }
    regions.sort_unstable();
    let mut cursor = 0;
    for (begin, end) in regions {
        ensure!(
            begin == cursor,
            "LoRA tensor data has overlapping regions or gaps"
        );
        cursor = end;
    }
    ensure!(
        cursor == file_len - size - 8,
        "LoRA has trailing or missing tensor data"
    );
    Ok(())
}

impl PreparedLoras {
    /// Check target names and shapes against base headers without loading base planes.
    pub fn validate_base(&self, root: &Path) -> Result<()> {
        use std::io::Read;
        if self.is_empty() {
            return Ok(());
        }
        let mut shapes = BTreeMap::new();
        for entry in std::fs::read_dir(root.join("transformer"))? {
            let path = entry?.path();
            if path.extension().is_none_or(|x| x != "safetensors") {
                continue;
            }
            let mut file = std::fs::File::open(&path)?;
            let mut prefix = [0u8; 8];
            file.read_exact(&mut prefix)?;
            let length = u64::from_le_bytes(prefix);
            ensure!(
                length <= 16 * 1024 * 1024,
                "base safetensors header is too large"
            );
            let mut bytes = vec![0u8; length as usize];
            file.read_exact(&mut bytes)?;
            let header: serde_json::Value = serde_json::from_slice(&bytes)?;
            for (name, value) in header
                .as_object()
                .context("invalid base safetensors header")?
            {
                if let Some(shape) = value.get("shape").and_then(|v| v.as_array()) {
                    let shape = shape
                        .iter()
                        .map(|v| v.as_u64().map(|n| n as usize).context("invalid base shape"))
                        .collect::<Result<Vec<_>>>()?;
                    ensure!(
                        shapes.insert(name.clone(), shape).is_none(),
                        "duplicate base tensor {name}"
                    );
                }
            }
        }
        for (name, updates) in &self.deltas {
            let shape = shapes
                .get(name)
                .with_context(|| format!("LoRA target is absent from transformer: {name}"))?;
            ensure!(shape.len() == 2, "LoRA target {name} is not a matrix");
            for delta in updates {
                ensure!(
                    delta.a.dim(1)? == shape[1] && delta.b.dim(0)? == shape[0],
                    "LoRA {} shape does not match base {name} {shape:?}",
                    delta.label
                );
            }
        }
        Ok(())
    }
}
