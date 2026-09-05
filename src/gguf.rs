use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result, bail, ensure};
use candle_core::quantized::gguf_file::{
    Content, DEFAULT_ALIGNMENT, TensorInfo, Value, VersionedMagic,
};
use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, MetalDevice, MetalStorage, Module, Shape, Storage, Tensor};
use candle_metal_kernels::metal::Buffer;
use candle_nn::RmsNorm;

/// `XWEN_LOAD_CLASSIC` reverts the model load from the default mmap aliasing
/// (the big weights — expert stacks, attention weights' stored bytes (f16
/// planes or raw q8_0 blocks) — alias the GGUF's page cache through no-copy
/// Metal buffer views, so a warm load takes seconds)
/// to the legacy full-copy load (read every tensor into a Vec, upload via
/// `QStorage::from_data`). Consulted once at `open`: a classic-opened `GgufFile`
/// carries no mapping, so every downstream loader copies.
///
/// PRESENCE-BASED and cached (read once), like the sibling `ops::*` switches
/// (`flash_classic`, `attn_glue_classic`): any value enables it — only leaving
/// it unset keeps the mmap path.
pub fn load_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_LOAD_CLASSIC").is_some())
}

unsafe extern "C" {
    /// Mach's exported page-size global (libSystem, always linked): 16384 on
    /// Apple silicon, but read at runtime rather than hardcoded.
    /// `newBufferWithBytesNoCopy` requires page-aligned pointers and
    /// page-multiple lengths, so `MmapSource::view` floors/ceils with this.
    safe static vm_page_size: usize;
}

/// The GGUF file mapped read-only, plus the raw Metal device and residency set
/// its no-copy buffer views hang off. Built once per `open` on Metal (unless
/// `XWEN_LOAD_CLASSIC`) and shared by every aliased tensor.
///
/// LIFETIME INVARIANT: the mapping must outlive every Metal view buffer created
/// over it. Views are created with `deallocator: None`, so dropping a `Buffer`
/// never unmaps — but dropping this struct DOES (memmap2 unmaps on drop), which
/// would leave a live view (and the GPU) reading unmapped pages. Everything
/// holding an aliased view therefore keeps a clone of the owning
/// `Arc<MmapSource>`: `ExpertStack.mmap` per stack, and `XwenModel` holds one
/// for the aliased attention tensors (whose `Tensor`s cannot carry it).
pub struct MmapSource {
    map: memmap2::Mmap,
    /// The candle Metal device views are created on. View buffers are batch-
    /// registered in ITS queue-attached residency set (the vendored candle
    /// patch `register_buffers`), so aliased weights stay permanently
    /// GPU-resident exactly like candle's own pool buffers — without this the
    /// weight working set pays per-command-buffer residency bookkeeping,
    /// measured at ~10% of sustained 4k prefill on laguna's ~70GB checkpoint
    /// (this loader is inherited from there, measurement and all).
    /// Residency is perf-only: setBuffer-bound buffers are made resident per
    /// command buffer regardless, so unregistered views compute correctly.
    mdev: MetalDevice,
    /// Views created but not yet registered resident; drained into
    /// `registered` by `register_views` (XwenModel::load calls it once after
    /// all weights are built — one batch, one residency-set commit).
    pending: Mutex<Vec<Arc<Buffer>>>,
    /// Views currently registered in the device's residency set. Drop
    /// unregisters them — the set RETAINS its allocations, so without this a
    /// dumped model would leave every view's MTLBuffer (and its GPU mapping)
    /// alive forever. Load→drop cycles (e.g. a serve-then-unload server) are
    /// leak-free: drop order guarantees this struct outlives all view holders,
    /// so by the time Drop runs the set holds the only remaining retains.
    registered: Mutex<Vec<Arc<Buffer>>>,
}

impl Drop for MmapSource {
    fn drop(&mut self) {
        // Quiesce before unmapping: an in-flight command buffer retains the
        // view MTLBuffers it has bound, but a buffer being alive does NOT keep
        // the underlying pages mapped (`deallocator: None`) — the munmap when
        // `self.map` drops below would yank pages the GPU may still be
        // reading. Errors are ignored: teardown must not panic.
        let _ = self.mdev.wait_until_completed();
        let registered = self.registered.get_mut().unwrap_or_else(|e| e.into_inner());
        self.mdev
            .unregister_buffers(registered.iter().map(|b| b.as_ref()));
    }
}

impl MmapSource {
    /// Maps `path` read-only for aliasing on `device` (must be Metal).
    pub fn open(path: &Path, device: &Device) -> Result<Arc<Self>> {
        let Device::Metal(mdev) = device else {
            bail!("mmap aliasing requires a Metal device");
        };
        let file =
            File::open(path).with_context(|| format!("opening {} for mmap", path.display()))?;
        // SAFETY: the mapping is read-only, and the GGUF file being truncated or
        // rewritten under a running process is out of contract (the same
        // assumption llama.cpp's mmap loader makes).
        let map = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("mmapping {}", path.display()))?;
        // Cheap prefetch hint; harmless if the kernel ignores it.
        let _ = map.advise(memmap2::Advice::WillNeed);
        Ok(Arc::new(Self {
            map,
            mdev: mdev.clone(),
            pending: Mutex::new(Vec::new()),
            registered: Mutex::new(Vec::new()),
        }))
    }

    /// One no-copy Metal buffer aliasing bytes `[abs_off, abs_off + len)` of the
    /// mapping: page-floored start, page-ceiled length (Metal requires both),
    /// candle's `RESOURCE_OPTIONS` (Shared + hazard-untracked, same as every
    /// candle allocation so the encoder fence discipline stays uniform).
    /// Returns the buffer plus `base_off`, the tensor's byte offset inside the
    /// view — always < page size, and 32-byte aligned because GGUF aligns
    /// tensor data to 32. Overlapping views over one mapping are legal (the
    /// fork's giant per-file views rely on the same property).
    fn view(&self, abs_off: usize, len: usize) -> Result<(Arc<Buffer>, usize)> {
        ensure!(
            abs_off
                .checked_add(len)
                .is_some_and(|end| end <= self.map.len()),
            "mmap view [{abs_off}, +{len}) exceeds the {}-byte mapping",
            self.map.len()
        );
        let page = vm_page_size;
        let start = abs_off / page * page;
        let base_off = abs_off - start;
        // Page-ceiled: the kernel maps whole pages, so a tail past EOF inside
        // the last page is mapped (zero-filled) and safe to cover.
        let view_len = (base_off + len).div_ceil(page) * page;
        let ptr = std::ptr::NonNull::new(
            unsafe { self.map.as_ptr().add(start) } as *mut std::ffi::c_void
        )
        .context("mmap base pointer is null")?;
        // SAFETY: `ptr` is page-aligned inside the mapping and `view_len` is a
        // page multiple; the bytes stay valid as long as `self.map` lives (the
        // Arc<MmapSource> lifetime invariant above). `deallocator: None` means
        // Metal never frees the pages — unmapping stays the Mmap drop's job.
        let raw = unsafe {
            use objc2_metal::MTLDevice as _;
            self.mdev
                .device()
                .as_ref()
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr,
                    view_len,
                    candle_metal_kernels::RESOURCE_OPTIONS,
                    None,
                )
        }
        .with_context(|| {
            format!("newBufferWithBytesNoCopy failed for {view_len} bytes at {start}")
        })?;
        let buffer = Arc::new(Buffer::new(raw));
        // Collected for batch residency registration — see `pending`/
        // `register_views`. Per-view registration (a synchronous residency-set
        // commit each) measured ~7s of load across the 381 views.
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(buffer.clone());
        Ok((buffer, base_off))
    }

    /// The mapping's raw bytes for `[abs_off, abs_off + len)` — the CPU-side
    /// twin of `view`, for load-time validation scans over tensors about to be
    /// aliased (the drafter's bf16 f16-range scan). Bounds-checked; the slice
    /// borrows the mapping, so it cannot outlive the `Arc<MmapSource>`.
    pub(crate) fn bytes(&self, abs_off: usize, len: usize) -> Result<&[u8]> {
        ensure!(
            abs_off
                .checked_add(len)
                .is_some_and(|end| end <= self.map.len()),
            "mmap byte range [{abs_off}, +{len}) exceeds the {}-byte mapping",
            self.map.len()
        );
        Ok(&self.map[abs_off..abs_off + len])
    }

    /// `madvise(MADV_RANDOM)` over `[abs_off, abs_off + len)` of the mapping.
    ///
    /// A HINT, so failure is silently ignored — the caller has nothing to do
    /// with the error and the mapping reads correctly either way. The one
    /// caller is the PLE n-gram table (`qwen4exp::ple`), whose access pattern is
    /// 16 unrelated 90-byte rows per token over 28.8 GB: the default readahead
    /// turns each of those into a large sequential window that nothing else in
    /// the row will ever be read from. The whole-file `WillNeed` in `open`
    /// deliberately stays — it is for the weights, which ARE read sequentially.
    pub(crate) fn advise_random(&self, abs_off: usize, len: usize) {
        if abs_off
            .checked_add(len)
            .is_some_and(|end| end <= self.map.len())
        {
            let _ = self.map.advise_range(memmap2::Advice::Random, abs_off, len);
        }
    }

    /// Registers every not-yet-registered view in the device's queue-attached
    /// residency set, one batch + one commit. XwenModel::load calls this
    /// once after all weights are built; Drop unregisters everything this
    /// registered.
    pub fn register_views(&self) {
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        let mut registered = self.registered.lock().unwrap_or_else(|e| e.into_inner());
        self.mdev
            .register_buffers(pending.iter().map(|b| b.as_ref()));
        registered.append(&mut pending);
    }
}

/// Mach's page size (16384 on Apple silicon), read at runtime rather than
/// hardcoded — the same global `MmapSource::view` aligns against. Used by the
/// PLE prefetcher to collapse a row list to the distinct pages behind it.
pub(crate) fn host_page_size() -> usize {
    vm_page_size
}

/// Identity of the checkpoint a persisted artifact (a cache image) was produced
/// from: an FNV-1a 64 hash over the GGUF's metadata section — the header, the KV
/// metadata and the tensor-info table, i.e. everything before
/// `tensor_data_offset` — plus the file's total length. The tensor payload is
/// never hashed: it is tens of gigabytes, and the metadata alone already pins the
/// architecture, the rope parameters, the quant mix and the tensor layout, while
/// the length catches a payload that was replaced under an unchanged header.
///
/// Not cryptographic and does not need to be. It exists to stop a cache image
/// from one checkpoint being fed to another, which is a mismatch to detect among
/// the owner's own files, not an attack to resist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointId {
    hash: u64,
    file_len: u64,
}

impl CheckpointId {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x100_0000_01b3;

    /// FNV-1a 64 over `file`'s first `metadata_len` bytes, plus the file length.
    /// Reads in chunks rather than mapping, so the cost is one pass over a few
    /// megabytes at load time.
    fn compute(file: &mut File, metadata_len: u64) -> Result<Self> {
        let (hash, file_len) = Self::fold(file, metadata_len, Self::OFFSET_BASIS)?;
        Ok(Self { hash, file_len })
    }

    /// One file's contribution to the id: continues `hash` over the file's
    /// first `metadata_len` bytes and returns it with the file's total length.
    /// A split checkpoint chains this across its shards in shard order and sums
    /// the lengths, so the id pins every shard's tensor table (each shard's
    /// metadata section includes its own), not only the first shard's.
    fn fold(file: &mut File, metadata_len: u64, mut hash: u64) -> Result<(u64, u64)> {
        let file_len = file.metadata().context("stat for the checkpoint id")?.len();
        ensure!(
            metadata_len <= file_len,
            "checkpoint id: metadata section ends at {metadata_len} but the file is {file_len} \
             bytes"
        );
        file.seek(SeekFrom::Start(0))
            .context("rewinding for the checkpoint id")?;
        let mut left = metadata_len;
        let mut buf = vec![0u8; 1 << 20];
        while left > 0 {
            let want = left.min(buf.len() as u64) as usize;
            file.read_exact(&mut buf[..want])
                .context("reading the metadata section for the checkpoint id")?;
            for &byte in &buf[..want] {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(Self::PRIME);
            }
            left -= want as u64;
        }
        Ok((hash, file_len))
    }

    /// A specific id, for tests that have to bind and mis-bind a persisted
    /// artifact without a checkpoint on disk to derive one from.
    #[cfg(test)]
    pub(crate) fn from_parts(hash: u64, file_len: u64) -> Self {
        Self { hash, file_len }
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }

    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    /// The 16 hex chars persisted artifacts group themselves under, so images for
    /// a checkpoint that is no longer loaded stay identifiable on disk.
    pub fn dir_name(&self) -> String {
        format!("{:016x}", self.hash)
    }
}

/// The backing file's path, for consumers that go back to disk and read the
/// WHOLE file by path (the drafter loaders do exactly that). A split GGUF has
/// no such single file — shard 0's path would silently stand in for bytes that
/// live in other shards — so any access through this handle panics for a
/// multi-shard open. Split-aware consumers route reads per tensor
/// (`shard_for`) and hold `mmap_sources` instead. Field syntax (`gguf.path`)
/// is deliberately preserved: existing single-file consumers compile and
/// behave unchanged.
pub struct SingleFilePath {
    path: PathBuf,
    shard_count: usize,
}

impl SingleFilePath {
    fn whole(&self) -> &Path {
        assert!(
            self.shard_count == 1,
            "GgufFile::path on a split GGUF ({} shards): {} is only shard 0, not the whole \
             checkpoint — read per shard via mmap_sources()/shard-aware accessors instead",
            self.shard_count,
            self.path.display()
        );
        &self.path
    }
}

impl std::ops::Deref for SingleFilePath {
    type Target = Path;
    fn deref(&self) -> &Path {
        self.whole()
    }
}

impl AsRef<Path> for SingleFilePath {
    fn as_ref(&self) -> &Path {
        self.whole()
    }
}

/// One shard of an opened (possibly split) GGUF: the file handle its tensors
/// are read from, and (on Metal, unless `XWEN_LOAD_CLASSIC`) that file's
/// mapping for the no-copy alias load path. A single-file open is exactly one
/// shard.
struct Shard {
    path: PathBuf,
    file: Mutex<File>,
    mmap: Option<Arc<MmapSource>>,
}

/// An opened GGUF — one file, or every shard of a gguf-split set presented as
/// one logical file.
pub struct GgufFile {
    /// The unified header. Single-file: exactly as parsed. Split: shard 0's
    /// metadata block, the union tensor table with each tensor's offset
    /// rebased to its absolute position in its own shard's file, and
    /// `tensor_data_offset` 0 — so `tensor_data_offset + info.offset` is the
    /// correct read position in the backing shard on both paths, provided the
    /// read goes through that shard's file/mapping (`shard_for`).
    pub content: Content,
    pub device: Device,
    /// THE file's path for a single-file GGUF; panics on access for a split
    /// one (see `SingleFilePath` — there is no single path whole-file reads
    /// would be correct against).
    pub path: SingleFilePath,
    checkpoint: CheckpointId,
    /// The files backing `content`'s tensors, in shard order.
    shards: Vec<Shard>,
    /// Tensor name → index into `shards`, covering BOTH halves of the table
    /// (`content.tensor_infos` and `raw_of`). Populated only for a split GGUF;
    /// single-file lookups never consult it.
    shard_of: HashMap<String, usize>,
    /// The other half of the tensor table: the tensors whose ggml type candle
    /// has no `GgmlDType` for, and which are therefore absent from `content`
    /// (see the tensor-table section above). Offsets are rebased exactly as
    /// `content`'s are, so `content.tensor_data_offset + info.offset` is the
    /// read position in the backing shard on both paths.
    ///
    /// Nothing here is ever uploaded to the device — candle could not, and the
    /// one production member (the 28.8 GB PLE n-gram table) is deliberately
    /// demand-paged on the host instead (docs/qwen4exp-port.md D2). That is why
    /// these tensors stay OUT of `content.tensor_infos` and so out of
    /// `model.rs`'s resident-weight budget: counting 28.8 GB of page cache as
    /// resident VRAM would make every footprint line wrong. `raw_tensor_bytes`
    /// reports them separately for anyone who does want the file-size total.
    raw_of: HashMap<String, RawTensorInfo>,
}

/// One tensor's bytes as a location in a mapped shard rather than a loaded
/// tensor — see [`GgufFile::raw_tensor`].
pub(crate) struct RawTensor {
    /// The mapping the bytes live in. Held as an `Arc` because the reader
    /// outlives the call (`MmapSource`'s lifetime invariant).
    pub src: Arc<MmapSource>,
    /// Absolute byte offset of the tensor's first block inside `src`.
    pub offset: usize,
    /// The tensor's byte length in its stored dtype.
    pub len: usize,
    pub dtype: StoredDtype,
    pub shape: Vec<usize>,
}

impl GgufFile {
    /// THE alias-load mapping of a single-file GGUF, present on Metal unless
    /// `XWEN_LOAD_CLASSIC`. A holder of aliased weights whose tensors cannot
    /// carry the Arc themselves (the attention planes) must keep the mapping
    /// alive — see `MmapSource`'s lifetime invariant — and for a
    /// possibly-split file that means holding `mmap_sources`, all of them.
    /// Panics on a split GGUF: shard 0's mapping alone is not the checkpoint,
    /// and a whole-file consumer reading it as such would silently get wrong
    /// bytes.
    pub fn mmap_source(&self) -> Option<&Arc<MmapSource>> {
        assert!(
            self.shards.len() == 1,
            "mmap_source() on a split GGUF ({} shards): no single mapping covers it — use \
             mmap_sources()",
            self.shards.len()
        );
        self.shards[0].mmap.as_ref()
    }

    /// Every shard's alias-load mapping (one entry for a single-file GGUF,
    /// empty off Metal or under `XWEN_LOAD_CLASSIC`). Keep-alive and view
    /// registration must cover all of them: a tensor aliases whichever shard's
    /// mapping holds it.
    pub fn mmap_sources(&self) -> Vec<Arc<MmapSource>> {
        self.shards.iter().filter_map(|s| s.mmap.clone()).collect()
    }

    /// The shard whose file (and mapping) hold `name`'s data. Single-file
    /// opens have an empty `shard_of`, so every lookup lands on the one shard;
    /// so does an unknown name, whose caller then fails its own tensor-info
    /// lookup.
    fn shard_for(&self, name: &str) -> &Shard {
        match self.shard_of.get(name) {
            Some(&i) => &self.shards[i],
            None => &self.shards[0],
        }
    }

    /// Where one tensor's bytes physically live: the shard mapping holding
    /// them, the absolute offset into it, the byte length, the stored dtype and
    /// the shape.
    ///
    /// For the one consumer that reads a tensor on the CPU instead of uploading
    /// it: the PLE n-gram table, 28.8 GB that never becomes a device tensor and
    /// is gathered one 160-float row at a time (docs/qwen4exp-port.md D2). It
    /// therefore returns a location, not a `Tensor` — and needs the mmap, so a
    /// classic (non-aliasing) open is refused here rather than silently
    /// reading through the file handle a row at a time.
    ///
    /// It serves BOTH halves of the tensor table: a tensor candle named
    /// resolves through `content.tensor_infos` exactly as before, and one it
    /// could not (the IQ4_NL PLE table of every Unsloth Q3/Q4 mix) resolves
    /// through `raw_of`. Either way the caller gets bytes plus a `StoredDtype`
    /// and does its own dequant — `qwen4exp::ple::PleTable` is the reader
    /// (docs/qwen4exp-port.md D8 class 1).
    pub(crate) fn raw_tensor(&self, name: &str) -> Result<RawTensor> {
        let (dtype, shape, offset, len) = match self.content.tensor_infos.get(name) {
            Some(info) => {
                let dtype = info.ggml_dtype;
                let block = dtype.block_size();
                let elems = info.shape.elem_count();
                ensure!(
                    elems.is_multiple_of(block),
                    "{name}: {elems} elements is not a multiple of {dtype:?} block size {block}"
                );
                (
                    StoredDtype::Ggml(dtype),
                    info.shape.dims().to_vec(),
                    info.offset,
                    elems / block * dtype.type_size(),
                )
            }
            None => {
                let info = self
                    .raw_of
                    .get(name)
                    .with_context(|| format!("tensor {name} not found"))?;
                (
                    StoredDtype::Raw(info.dtype),
                    info.shape.clone(),
                    info.offset,
                    usize::try_from(info.byte_len).with_context(|| {
                        format!("{name} is {} bytes, past usize", info.byte_len)
                    })?,
                )
            }
        };
        let src = self
            .shard_for(name)
            .mmap
            .as_ref()
            .with_context(|| {
                format!(
                    "tensor {name} has no file mapping to read rows from (non-Metal device, or \
                     XWEN_LOAD_CLASSIC)"
                )
            })?
            .clone();
        Ok(RawTensor {
            src,
            offset: (self.content.tensor_data_offset + offset) as usize,
            len,
            dtype,
            shape,
        })
    }

    /// The dtype a tensor is STORED at, across both halves of the table — the
    /// whole-file twin of `Weights::stored_dtype`, which can only answer for
    /// the tensors candle named.
    pub fn stored_dtype_of(&self, name: &str) -> Result<StoredDtype> {
        if let Some(info) = self.content.tensor_infos.get(name) {
            return Ok(StoredDtype::Ggml(info.ggml_dtype));
        }
        self.raw_of
            .get(name)
            .map(|info| StoredDtype::Raw(info.dtype))
            .with_context(|| format!("tensor {name} not found"))
    }

    /// Whether `name` is in the tensor table at all, either half.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name) || self.raw_of.contains_key(name)
    }

    /// Total tensor count across both halves — what the file's header declared.
    pub fn tensor_count(&self) -> usize {
        self.content.tensor_infos.len() + self.raw_of.len()
    }

    /// Names and stored dtypes of the tensors candle could not name, for
    /// diagnostics (`describe_file`) and for tests that assert which planes a
    /// mix keeps outside candle's reach.
    pub fn raw_tensor_names(&self) -> Vec<(&str, RawDtype)> {
        self.raw_of
            .iter()
            .map(|(name, info)| (name.as_str(), info.dtype))
            .collect()
    }

    /// Stored bytes of the raw half of the table. NOT part of the resident
    /// weight footprint — see `raw_of` — but the number to reach for when
    /// accounting for the file rather than for VRAM.
    pub fn raw_tensor_bytes(&self) -> u64 {
        self.raw_of.values().map(|i| i.byte_len).sum()
    }

    /// Identity of this checkpoint, computed once at `open`. Persisted artifacts
    /// derived from a loaded model are stamped with it and refused when it does
    /// not match.
    pub fn checkpoint_id(&self) -> CheckpointId {
        self.checkpoint
    }
}

// ---------------------------------------------------------------------------
// The tensor table, parsed by xwen rather than by candle.
//
// candle's `Content::read` maps every tensor's ggml type id through
// `GgmlDType::from_u32` and FAILS the whole file when one id has no `GgmlDType`
// variant — and candle names none of the IQ* types. The Unsloth Q3/Q4 mixes of
// Qwen3.8-Flash-Next hold `per_layer_token_embd.weight` (28.8 GB of n-gram
// table) at IQ4_NL, so `Content::read` refuses those files outright with
// "unknown dtype for tensor 20" even though every tensor the GPU ever sees is a
// type candle knows.
//
// So xwen parses the header itself (docs/qwen4exp-port.md D8) and splits the
// tensor table in two: everything candle can name goes into a `Content` built
// exactly as `Content::read` would have built it — so `Weights`, `config.rs`
// and every existing loader path are untouched — and everything else goes into
// `GgufFile::raw_of`, reachable only through `raw_tensor`, which hands out a
// byte location for an xwen-side reader (the PLE table's CPU row gather). The
// pinned candle stays unpatched.
// ---------------------------------------------------------------------------

/// Block geometry of a ggml tensor type id: `(name, block_size, type_size)`,
/// mirroring the `type_traits` table of `reference/llama.cpp/ggml/src/ggml.c`
/// and the ids of `ggml/include/ggml.h`. A tensor's stored byte length is
/// `elem_count / block_size * type_size`, so this table is what lets the loader
/// size a tensor whose type candle has no variant for.
///
/// `None` is a type id ggml itself does not define: the removed Q4_3 (5) and
/// Q4_0_4_4/4_8/8_8 (36-38) slots, or anything at or past `GGML_TYPE_COUNT`. A
/// file carrying one is refused at open — the same outcome candle's parser gave
/// for every unknown id, kept deliberately so a tensor is never silently
/// dropped from the table.
fn ggml_type_geometry(id: u32) -> Option<(&'static str, usize, usize)> {
    Some(match id {
        0 => ("F32", 1, 4),
        1 => ("F16", 1, 2),
        2 => ("Q4_0", 32, 18),
        3 => ("Q4_1", 32, 20),
        6 => ("Q5_0", 32, 22),
        7 => ("Q5_1", 32, 24),
        8 => ("Q8_0", 32, 34),
        9 => ("Q8_1", 32, 36),
        10 => ("Q2_K", 256, 84),
        11 => ("Q3_K", 256, 110),
        12 => ("Q4_K", 256, 144),
        13 => ("Q5_K", 256, 176),
        14 => ("Q6_K", 256, 210),
        15 => ("Q8_K", 256, 292),
        16 => ("IQ2_XXS", 256, 66),
        17 => ("IQ2_XS", 256, 74),
        18 => ("IQ3_XXS", 256, 98),
        19 => ("IQ1_S", 256, 50),
        20 => ("IQ4_NL", 32, 18),
        21 => ("IQ3_S", 256, 110),
        22 => ("IQ2_S", 256, 82),
        23 => ("IQ4_XS", 256, 136),
        24 => ("I8", 1, 1),
        25 => ("I16", 1, 2),
        26 => ("I32", 1, 4),
        27 => ("I64", 1, 8),
        28 => ("F64", 1, 8),
        29 => ("IQ1_M", 256, 56),
        30 => ("BF16", 1, 2),
        34 => ("TQ1_0", 256, 54),
        35 => ("TQ2_0", 256, 66),
        39 => ("MXFP4", 32, 17),
        40 => ("NVFP4", 64, 36),
        41 => ("Q1_0", 128, 18),
        42 => ("Q2_0", 64, 18),
        _ => return None,
    })
}

/// The ggml type ids candle's `GgmlDType` names — i.e. the ids whose tensors
/// can be loaded, uploaded and dequantized through candle. Transcribed from
/// `GgmlDType::from_u32`, which is `pub(crate)` and so not callable from here;
/// `ggml_type_ids_agree_with_candle` pins the transcription against candle's
/// own `block_size`/`type_size` so a candle bump cannot silently desync it.
fn candle_dtype(id: u32) -> Option<GgmlDType> {
    Some(match id {
        0 => GgmlDType::F32,
        1 => GgmlDType::F16,
        2 => GgmlDType::Q4_0,
        3 => GgmlDType::Q4_1,
        6 => GgmlDType::Q5_0,
        7 => GgmlDType::Q5_1,
        8 => GgmlDType::Q8_0,
        9 => GgmlDType::Q8_1,
        10 => GgmlDType::Q2K,
        11 => GgmlDType::Q3K,
        12 => GgmlDType::Q4K,
        13 => GgmlDType::Q5K,
        14 => GgmlDType::Q6K,
        15 => GgmlDType::Q8K,
        30 => GgmlDType::BF16,
        _ => return None,
    })
}

/// A ggml tensor type that candle's `GgmlDType` has no variant for. Such a
/// tensor is never uploaded, dequantized or matmul'd — nothing downstream of
/// candle can touch it — so the only thing xwen does with one is hand out its
/// bytes (`GgufFile::raw_tensor`) for a reader that knows the format itself.
///
/// IQ4_NL is the variant that exists for a reason: it is where every Unsloth
/// Q3/Q4 mix keeps the Qwen3.8-Flash-Next PLE n-gram table. IQ4_XS is named
/// because it is the other IQ type those mixes reach for. Everything else stays
/// `Other`, sized from [`ggml_type_geometry`] like the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawDtype {
    Iq4Nl,
    Iq4Xs,
    Other(u32),
}

impl RawDtype {
    fn from_id(id: u32) -> Self {
        match id {
            20 => Self::Iq4Nl,
            23 => Self::Iq4Xs,
            other => Self::Other(other),
        }
    }

    pub fn type_id(self) -> u32 {
        match self {
            Self::Iq4Nl => 20,
            Self::Iq4Xs => 23,
            Self::Other(id) => id,
        }
    }

    /// The geometry, from the table. Infallible: a `RawDtype` only ever comes
    /// from an id the table already accepted at open.
    fn geometry(self) -> (&'static str, usize, usize) {
        ggml_type_geometry(self.type_id())
            .expect("RawDtype is only built from a known ggml type id")
    }

    pub fn name(self) -> &'static str {
        self.geometry().0
    }

    pub fn block_size(self) -> usize {
        self.geometry().1
    }

    pub fn type_size(self) -> usize {
        self.geometry().2
    }
}

/// The dtype a GGUF tensor is STORED at, across both halves of the split tensor
/// table: the types candle names and can load, and the types only xwen's own
/// parse names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoredDtype {
    Ggml(GgmlDType),
    Raw(RawDtype),
}

impl StoredDtype {
    pub fn block_size(self) -> usize {
        match self {
            Self::Ggml(d) => d.block_size(),
            Self::Raw(d) => d.block_size(),
        }
    }

    pub fn type_size(self) -> usize {
        match self {
            Self::Ggml(d) => d.type_size(),
            Self::Raw(d) => d.type_size(),
        }
    }

    /// The candle dtype, for a caller that needs to hand the tensor to candle;
    /// `None` for a raw type, which candle cannot represent at all.
    pub fn ggml(self) -> Option<GgmlDType> {
        match self {
            Self::Ggml(d) => Some(d),
            Self::Raw(_) => None,
        }
    }
}

/// One tensor of the raw half of the table — the half candle's `Content` cannot
/// hold. Mirrors candle's `TensorInfo` field for field, plus the byte length
/// (which `TensorInfo` recomputes from the dtype and candle cannot here).
///
/// `offset` follows candle's convention exactly: it is relative to the owning
/// `Content`'s `tensor_data_offset`, so a split open rebases it to
/// shard-absolute alongside the candle-known infos and `tensor_data_offset`
/// becomes 0 (see [`GgufFile::content`]).
#[derive(Debug, Clone)]
struct RawTensorInfo {
    dtype: RawDtype,
    shape: Vec<usize>,
    offset: u64,
    byte_len: u64,
}

/// A GGUF header as xwen parses it: candle's `Content` for the tensors candle
/// can name, and the raw table for the rest.
struct ParsedGguf {
    content: Content,
    raw: HashMap<String, RawTensorInfo>,
}

// Caps transcribed from candle's parser, which took them from
// ggml-org/llama.cpp#19856 (GGUF_MAX_STRING_LENGTH, GGUF_MAX_ARRAY_ELEMENTS)
// and GGML_MAX_DIMS. They bound what a corrupt or hostile header can make this
// loader allocate, and the depth cap bounds recursion through nested arrays.
const GGUF_MAX_STRING_LENGTH: u64 = 1 << 30;
const GGUF_MAX_ARRAY_ELEMENTS: u64 = 1 << 30;
const GGUF_MAX_TENSOR_DIMS: u32 = 4;
const GGUF_MAX_VALUE_DEPTH: usize = 64;

macro_rules! read_le {
    ($name:ident, $t:ty, $n:literal) => {
        fn $name<R: Read>(r: &mut R) -> Result<$t> {
            let mut b = [0u8; $n];
            r.read_exact(&mut b)?;
            Ok(<$t>::from_le_bytes(b))
        }
    };
}

read_le!(read_u8, u8, 1);
read_le!(read_i8, i8, 1);
read_le!(read_u16, u16, 2);
read_le!(read_i16, i16, 2);
read_le!(read_u32, u32, 4);
read_le!(read_i32, i32, 4);
read_le!(read_u64, u64, 8);
read_le!(read_i64, i64, 8);
read_le!(read_f32, f32, 4);
read_le!(read_f64, f64, 8);

/// GGUF v1 length-prefixes strings and counts with u32, v2/v3 with u64.
fn read_length<R: Read>(r: &mut R, magic: VersionedMagic) -> Result<u64> {
    match magic {
        VersionedMagic::GgufV1 => Ok(u64::from(read_u32(r)?)),
        VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => read_u64(r),
    }
}

fn length_prefix_size(magic: VersionedMagic) -> u64 {
    match magic {
        VersionedMagic::GgufV1 => 4,
        VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => 8,
    }
}

fn remaining_bytes<R: Seek>(r: &mut R, file_size: u64) -> Result<u64> {
    Ok(file_size.saturating_sub(r.stream_position()?))
}

/// A length-prefixed GGUF string. Trailing NULs are stripped and invalid UTF-8
/// is replaced rather than rejected — both are real in the wild, and candle's
/// parser did the same, so keys and values reach `config.rs` unchanged.
fn read_gguf_string<R: Read + Seek>(
    r: &mut R,
    magic: VersionedMagic,
    file_size: u64,
) -> Result<String> {
    let len = read_length(r, magic)?;
    ensure!(
        len <= GGUF_MAX_STRING_LENGTH,
        "gguf: string length {len} exceeds max {GGUF_MAX_STRING_LENGTH}"
    );
    let remaining = remaining_bytes(r, file_size)?;
    ensure!(
        len <= remaining,
        "gguf: string length {len} exceeds remaining file bytes {remaining}"
    );
    let mut v = vec![0u8; len as usize];
    r.read_exact(&mut v)?;
    while let Some(0) = v.last() {
        v.pop();
    }
    Ok(String::from_utf8_lossy(&v).into_owned())
}

/// Minimum on-disk size of one value of a given GGUF value type, used to reject
/// an array length that cannot fit in the file before allocating for it.
fn value_min_disk_size(value_type: u32, magic: VersionedMagic) -> u64 {
    match value_type {
        0 | 1 | 7 => 1,                     // U8, I8, Bool
        2 | 3 => 2,                         // U16, I16
        4..=6 => 4,                         // U32, I32, F32
        10..=12 => 8,                       // U64, I64, F64
        8 => length_prefix_size(magic),     // String
        9 => 4 + length_prefix_size(magic), // Array
        _ => 1,
    }
}

/// One GGUF metadata value. Value-type ids are the GGUF spec's
/// (0..=12, 9 = array), and the variants are candle's `Value` so `config.rs`
/// and every other metadata consumer keeps the exact type it already reads.
fn read_gguf_value<R: Read + Seek>(
    r: &mut R,
    value_type: u32,
    magic: VersionedMagic,
    depth: usize,
    file_size: u64,
) -> Result<Value> {
    ensure!(
        depth <= GGUF_MAX_VALUE_DEPTH,
        "gguf: value nesting depth exceeds max {GGUF_MAX_VALUE_DEPTH}"
    );
    let v = match value_type {
        0 => Value::U8(read_u8(r)?),
        1 => Value::I8(read_i8(r)?),
        2 => Value::U16(read_u16(r)?),
        3 => Value::I16(read_i16(r)?),
        4 => Value::U32(read_u32(r)?),
        5 => Value::I32(read_i32(r)?),
        6 => Value::F32(read_f32(r)?),
        7 => match read_u8(r)? {
            0 => Value::Bool(false),
            1 => Value::Bool(true),
            b => bail!("gguf: unexpected bool value {b}"),
        },
        8 => Value::String(read_gguf_string(r, magic, file_size)?),
        9 => {
            let elem_type = read_u32(r)?;
            ensure!(
                elem_type <= 12,
                "gguf: unrecognized value-type {elem_type:#08x}"
            );
            let len = read_length(r, magic)?;
            ensure!(
                len <= GGUF_MAX_ARRAY_ELEMENTS,
                "gguf: array length {len} exceeds max {GGUF_MAX_ARRAY_ELEMENTS}"
            );
            let needed = len.saturating_mul(value_min_disk_size(elem_type, magic));
            let remaining = remaining_bytes(r, file_size)?;
            ensure!(
                needed <= remaining,
                "gguf: array of {len} elements needs at least {needed} bytes, only {remaining} \
                 remaining"
            );
            let mut vs = Vec::new();
            for _ in 0..len {
                vs.push(read_gguf_value(r, elem_type, magic, depth + 1, file_size)?);
            }
            Value::Array(vs)
        }
        10 => Value::U64(read_u64(r)?),
        11 => Value::I64(read_i64(r)?),
        12 => Value::F64(read_f64(r)?),
        other => bail!("gguf: unrecognized value-type {other:#08x}"),
    };
    Ok(v)
}

/// Parses one GGUF file's header — magic, KV metadata, tensor table and the
/// aligned tensor-data offset — leaving the reader positioned wherever the
/// table ended.
///
/// Deliberately byte-for-byte equivalent to candle's `Content::read` for every
/// file candle could read: same caps, same string handling, same
/// `dimensions.reverse()` (so shapes stay row-major `[out_dim, in_dim]`), same
/// `general.alignment` rule down to which integer widths it honors. That
/// equivalence is load-bearing twice over — `CheckpointId` hashes the bytes up
/// to `tensor_data_offset`, so a different alignment rule would invalidate
/// every persisted cache image on disk, and `config.rs` reads the metadata map
/// this produces.
///
/// The one difference is the point of the exercise: a tensor whose ggml type id
/// candle cannot name lands in the raw table instead of failing the file.
fn read_gguf_header(file: &mut File) -> Result<ParsedGguf> {
    let mut r = BufReader::new(file);
    let start = r.stream_position()?;
    let file_size = r.seek(SeekFrom::End(0))?;
    r.seek(SeekFrom::Start(start))?;

    let magic_word = read_u32(&mut r)?;
    ensure!(
        matches!(magic_word, 0x4655_4747 | 0x4747_5546),
        "gguf: unknown magic 0x{magic_word:08x}"
    );
    let version = read_u32(&mut r)?;
    let magic = match version {
        1 => VersionedMagic::GgufV1,
        2 => VersionedMagic::GgufV2,
        3 => VersionedMagic::GgufV3,
        v => bail!("gguf: unsupported version {v}"),
    };

    let tensor_count = read_length(&mut r, magic)?;
    let metadata_kv_count = read_length(&mut r, magic)?;
    ensure!(
        tensor_count <= GGUF_MAX_ARRAY_ELEMENTS,
        "gguf: tensor_count {tensor_count} exceeds max {GGUF_MAX_ARRAY_ELEMENTS}"
    );
    ensure!(
        metadata_kv_count <= GGUF_MAX_ARRAY_ELEMENTS,
        "gguf: metadata_kv_count {metadata_kv_count} exceeds max {GGUF_MAX_ARRAY_ELEMENTS}"
    );

    // Reject header-declared counts that cannot fit in the file even at the
    // minimum per-entry size, before any of them sizes an allocation.
    let prefix = length_prefix_size(magic);
    let needed = metadata_kv_count
        .saturating_mul(prefix + 4 + 1)
        .saturating_add(tensor_count.saturating_mul(prefix + 4 + 4 + 8));
    let remaining = remaining_bytes(&mut r, file_size)?;
    ensure!(
        needed <= remaining,
        "gguf: header declares {tensor_count} tensors and {metadata_kv_count} metadata entries, \
         needs at least {needed} bytes, only {remaining} remaining"
    );

    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = read_gguf_string(&mut r, magic, file_size)?;
        let value_type = read_u32(&mut r)?;
        let value = read_gguf_value(&mut r, value_type, magic, 0, file_size)?;
        metadata.insert(key, value);
    }

    let mut tensor_infos: HashMap<String, TensorInfo> = HashMap::new();
    let mut raw: HashMap<String, RawTensorInfo> = HashMap::new();
    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut r, magic, file_size)?;
        let n_dims = read_u32(&mut r)?;
        ensure!(
            n_dims <= GGUF_MAX_TENSOR_DIMS,
            "gguf: tensor '{name}' has {n_dims} dimensions, max is {GGUF_MAX_TENSOR_DIMS}"
        );
        let mut dims: Vec<usize> = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let d = match magic {
                VersionedMagic::GgufV1 => u64::from(read_u32(&mut r)?),
                VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => read_u64(&mut r)?,
            };
            dims.push(usize::try_from(d).with_context(|| {
                format!("gguf: tensor '{name}' declares a dimension of {d}, past usize")
            })?);
        }
        // GGUF writes dimensions fastest-varying first; candle reverses them so
        // a `{in_dim, out_dim}` weight reads as the row-major `[out_dim,
        // in_dim]` every loader here indexes.
        dims.reverse();
        let type_id = read_u32(&mut r)?;
        let offset = read_u64(&mut r)?;
        let Some((type_name, block_size, type_size)) = ggml_type_geometry(type_id) else {
            bail!(
                "gguf: tensor '{name}' has ggml type id {type_id}, which is not a type ggml \
                 defines"
            );
        };
        let elems = dims
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .with_context(|| {
                format!("gguf: tensor '{name}' element count overflows usize: {dims:?}")
            })?;
        ensure!(
            elems.is_multiple_of(block_size),
            "gguf: tensor '{name}': {elems} elements is not a multiple of the {type_name} block \
             size {block_size}"
        );
        let duplicate = match candle_dtype(type_id) {
            Some(ggml_dtype) => tensor_infos
                .insert(
                    name.clone(),
                    TensorInfo {
                        ggml_dtype,
                        shape: Shape::from(dims),
                        offset,
                    },
                )
                .is_some(),
            None => raw
                .insert(
                    name.clone(),
                    RawTensorInfo {
                        dtype: RawDtype::from_id(type_id),
                        shape: dims,
                        offset,
                        byte_len: (elems / block_size) as u64 * type_size as u64,
                    },
                )
                .is_some(),
        };
        // candle's HashMap insert silently kept the last of a repeated name and
        // left the file's declared tensor count disagreeing with the table.
        ensure!(
            !duplicate,
            "gguf: tensor '{name}' appears twice in the tensor table"
        );
    }

    let position = r.stream_position()?;
    // candle's alignment rule, honored width for width: only the unsigned and
    // non-negative signed 8/16/32-bit forms count, anything else (including a
    // u64) falls back to 32. Reproduced rather than improved — see the doc
    // comment on why the exact offset matters.
    let alignment = match metadata.get("general.alignment") {
        Some(Value::U8(v)) => u64::from(*v),
        Some(Value::U16(v)) => u64::from(*v),
        Some(Value::U32(v)) => u64::from(*v),
        Some(Value::I8(v)) if *v >= 0 => *v as u64,
        Some(Value::I16(v)) if *v >= 0 => *v as u64,
        Some(Value::I32(v)) if *v >= 0 => *v as u64,
        _ => DEFAULT_ALIGNMENT,
    };
    ensure!(alignment > 0, "gguf: general.alignment is 0");
    let tensor_data_offset = position.div_ceil(alignment) * alignment;

    Ok(ParsedGguf {
        content: Content {
            magic,
            metadata,
            tensor_infos,
            tensor_data_offset,
        },
        raw,
    })
}

pub fn metal_device() -> Result<Device> {
    Ok(Device::new_metal(0)?)
}

pub fn open(path: impl AsRef<Path>, device: &Device) -> Result<Arc<GgufFile>> {
    let path = path.as_ref();
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let ParsedGguf { content, raw } =
        read_gguf_header(&mut file).with_context(|| format!("parsing GGUF {}", path.display()))?;
    // A shard of a split GGUF (the gguf-split layout) hands the whole sibling
    // set to `open_split`. A self-contained split.count = 1 export (split.no
    // 0) is a complete single-file GGUF; any other partial or self-
    // contradicting combination of the split.* trio is refused rather than
    // read as a single file that would be missing most of its tensors.
    let split_count = metadata_uint(&content, SPLIT_COUNT_KEY)
        .with_context(|| format!("reading {SPLIT_COUNT_KEY} of {}", path.display()))?;
    let split_no = metadata_uint(&content, SPLIT_NO_KEY)
        .with_context(|| format!("reading {SPLIT_NO_KEY} of {}", path.display()))?;
    let split_tensors = metadata_uint(&content, SPLIT_TENSORS_COUNT_KEY)
        .with_context(|| format!("reading {SPLIT_TENSORS_COUNT_KEY} of {}", path.display()))?;
    match split_count {
        Some(count) if count > 1 => return open_split(path, device, count),
        Some(1) => ensure!(
            matches!(split_no, None | Some(0)),
            "{}: {SPLIT_COUNT_KEY} is 1 but {SPLIT_NO_KEY} is {} — a shard of a larger set, not \
             a one-shard export",
            path.display(),
            split_no.unwrap_or(0)
        ),
        Some(count) => bail!(
            "{}: {SPLIT_COUNT_KEY} is {count}, which describes no shard set",
            path.display()
        ),
        None => ensure!(
            split_no.is_none() && split_tensors.is_none(),
            "{}: carries split shard keys ({SPLIT_NO_KEY}/{SPLIT_TENSORS_COUNT_KEY}) but no \
             {SPLIT_COUNT_KEY} — a shard with half-stripped split metadata, not a single-file \
             GGUF",
            path.display()
        ),
    }
    // Hashed here rather than lazily: `Content::read` has just told us where the
    // metadata section ends, and every consumer of the id wants it before the
    // first request.
    let checkpoint = CheckpointId::compute(&mut file, content.tensor_data_offset)
        .with_context(|| format!("identifying GGUF {}", path.display()))?;
    let mmap = if matches!(device, Device::Metal(_)) && !load_classic() {
        Some(MmapSource::open(path, device)?)
    } else {
        None
    };
    Ok(Arc::new(GgufFile {
        content,
        device: device.clone(),
        path: SingleFilePath {
            path: path.to_path_buf(),
            shard_count: 1,
        },
        checkpoint,
        shards: vec![Shard {
            path: path.to_path_buf(),
            file: Mutex::new(file),
            mmap,
        }],
        shard_of: HashMap::new(),
        raw_of: raw,
    }))
}

/// gguf-split KV keys. Every shard of a split GGUF carries all three:
/// `split.no` (0-INDEXED, unlike the 1-indexed file names), `split.count`, and
/// `split.tensors.count` (the TOTAL tensor count across all shards).
const SPLIT_NO_KEY: &str = "split.no";
const SPLIT_COUNT_KEY: &str = "split.count";
const SPLIT_TENSORS_COUNT_KEY: &str = "split.tensors.count";

/// A non-negative integer metadata value, whatever width the writer chose
/// (gguf-split writes `split.no`/`split.count` as u16 but `split.tensors.count`
/// as i32). `Ok(None)` when the key is absent; a present key that is not a
/// non-negative integer is an error.
fn metadata_uint(content: &Content, key: &str) -> Result<Option<u64>> {
    let Some(v) = content.metadata.get(key) else {
        return Ok(None);
    };
    let n = match v {
        Value::U8(v) => u64::from(*v),
        Value::U16(v) => u64::from(*v),
        Value::U32(v) => u64::from(*v),
        Value::U64(v) => *v,
        Value::I8(v) if *v >= 0 => *v as u64,
        Value::I16(v) if *v >= 0 => *v as u64,
        Value::I32(v) if *v >= 0 => *v as u64,
        Value::I64(v) if *v >= 0 => *v as u64,
        other => bail!("metadata key {key} is not a non-negative integer: {other:?}"),
    };
    Ok(Some(n))
}

/// Parses the gguf-split naming convention `<base>-000NN-of-000MM.gguf`
/// (shard numbers 1-indexed and zero-padded to five digits). Returns
/// `(base, shard_number, shard_count)`; `None` for any other name shape.
fn split_name_parts(path: &Path) -> Option<(String, usize, usize)> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf")?;
    let (rest, count) = stem.rsplit_once("-of-")?;
    let (base, no) = rest.rsplit_once('-')?;
    if no.len() != 5 || count.len() != 5 {
        return None;
    }
    if !no.bytes().all(|b| b.is_ascii_digit()) || !count.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((base.to_string(), no.parse().ok()?, count.parse().ok()?))
}

/// Opens every shard of a split GGUF (discovered as siblings of `handed`, any
/// one shard of the set) and assembles the one logical file the rest of the
/// loader sees: shard 0's complete metadata block, the union tensor table with
/// offsets rebased to shard-absolute positions (see `GgufFile::content`), and
/// the tensor → shard map that routes each read to its backing file.
fn open_split(handed: &Path, device: &Device, split_count: u64) -> Result<Arc<GgufFile>> {
    let Some((base, handed_no, name_count)) = split_name_parts(handed) else {
        bail!(
            "{} declares {SPLIT_COUNT_KEY} = {split_count} but its file name does not follow the \
             <base>-000NN-of-000MM.gguf convention, so the sibling shards cannot be located",
            handed.display()
        );
    };
    ensure!(
        name_count as u64 == split_count,
        "{}: the file name says {name_count} shards but {SPLIT_COUNT_KEY} says {split_count}",
        handed.display()
    );
    // The handed file must itself be a member of the set its name describes
    // (shard numbers are 1-indexed): a stray `-00000-` or `-99999-` file would
    // otherwise silently open the canonical siblings while its own bytes are
    // never validated against them.
    ensure!(
        (1..=name_count).contains(&handed_no),
        "{}: shard number {handed_no} is outside 1..={name_count} — not a member of the set its \
         name describes",
        handed.display()
    );
    let dir = handed.parent().map(Path::to_path_buf).unwrap_or_default();
    let count = name_count;

    let mut parts: Vec<(PathBuf, File, ParsedGguf)> = Vec::with_capacity(count);
    let mut expected_tensors = 0u64;
    for i in 1..=count {
        let path = dir.join(format!("{base}-{i:05}-of-{count:05}.gguf"));
        let mut file = File::open(&path).with_context(|| {
            format!(
                "opening shard {i} of {count} ({}) — a split GGUF needs every sibling shard \
                 present",
                path.display()
            )
        })?;
        let parsed = read_gguf_header(&mut file)
            .with_context(|| format!("parsing GGUF shard {}", path.display()))?;
        let content = &parsed.content;
        let key = |k: &'static str| {
            metadata_uint(content, k)?
                .with_context(|| format!("{}: shard carries no {k} key", path.display()))
        };
        let no = key(SPLIT_NO_KEY)?;
        ensure!(
            no == (i - 1) as u64,
            "{}: {SPLIT_NO_KEY} is {no}, expected {} (shard file names are 1-indexed, \
             {SPLIT_NO_KEY} 0-indexed)",
            path.display(),
            i - 1
        );
        let sc = key(SPLIT_COUNT_KEY)?;
        ensure!(
            sc == split_count,
            "{}: {SPLIT_COUNT_KEY} is {sc}, but the shard set was opened as {split_count}",
            path.display()
        );
        let tc = key(SPLIT_TENSORS_COUNT_KEY)?;
        if i == 1 {
            expected_tensors = tc;
        } else {
            ensure!(
                tc == expected_tensors,
                "{}: {SPLIT_TENSORS_COUNT_KEY} is {tc}, but shard 1 of the set says \
                 {expected_tensors}",
                path.display()
            );
            // Real gguf-split shards carry ONLY the split.* trio beyond shard
            // 0 (verified against the Unsloth set), so absent keys are the
            // normal case — but a key present in BOTH shards with different
            // values means the files are not shards of one export (e.g. a mix
            // of two quants under one base name) and must be refused.
            // candle's `Value` has no `PartialEq`; the Debug rendering carries
            // the variant name and full contents, so equal renderings mean
            // equal values.
            for (k, v) in &content.metadata {
                if matches!(
                    k.as_str(),
                    SPLIT_NO_KEY | SPLIT_COUNT_KEY | SPLIT_TENSORS_COUNT_KEY
                ) {
                    continue;
                }
                if let Some(first) = parts[0].2.content.metadata.get(k) {
                    ensure!(
                        format!("{first:?}") == format!("{v:?}"),
                        "{}: metadata key {k} is {v:?} but shard 1 ({}) says {first:?} — these \
                         files are not shards of one split",
                        path.display(),
                        parts[0].0.display()
                    );
                }
            }
        }
        parts.push((path, file, parsed));
    }

    // The id chains the FNV fold over every shard's metadata section in shard
    // order and sums the file lengths, so it pins each shard's tensor table
    // and catches a replaced payload in any shard — and is identical whichever
    // shard the set was opened through.
    let mut hash = CheckpointId::OFFSET_BASIS;
    let mut total_len = 0u64;
    for (path, file, parsed) in &mut parts {
        let file_len = file
            .metadata()
            .with_context(|| format!("stat of shard {}", path.display()))?
            .len();
        // A metadata-only shard may end exactly where its KV block does, short
        // of the aligned tensor-data offset no tensor data exists to occupy;
        // its metadata section is then the whole file.
        let metadata_len = parsed.content.tensor_data_offset.min(file_len);
        (hash, _) = CheckpointId::fold(file, metadata_len, hash)
            .with_context(|| format!("identifying GGUF shard {}", path.display()))?;
        total_len += file_len;
    }
    let checkpoint = CheckpointId {
        hash,
        file_len: total_len,
    };

    // Union tensor table, offsets rebased to shard-absolute so the unified
    // content's tensor_data_offset of 0 keeps every existing offset
    // computation correct. Capacity comes from the tensor tables actually
    // parsed, never from the declared split.tensors.count — two tiny shards
    // declaring u64::MAX must fail the count check below, not size an
    // allocation here.
    let actual_tensors = parts
        .iter()
        .try_fold(0usize, |acc, p| {
            acc.checked_add(p.2.content.tensor_infos.len())?
                .checked_add(p.2.raw.len())
        })
        .context("tensor count across shards overflows usize")?;
    let mut tensor_infos = HashMap::with_capacity(actual_tensors);
    let mut raw_of: HashMap<String, RawTensorInfo> = HashMap::new();
    let mut shard_of = HashMap::with_capacity(actual_tensors);
    for idx in 0..parts.len() {
        let shard_path = parts[idx].0.clone();
        let data_offset = parts[idx].2.content.tensor_data_offset;
        let known = std::mem::take(&mut parts[idx].2.content.tensor_infos);
        let unknown = std::mem::take(&mut parts[idx].2.raw);
        // `shard_of` is the duplicate check for BOTH halves at once: a name
        // cannot repeat within one shard's table (`read_gguf_header` refuses
        // that) and cannot land in both halves of one shard, so a collision
        // here always means two shards claim the same tensor.
        let mut rebase = |name: &str, offset: u64| -> Result<u64> {
            let rebased = offset.checked_add(data_offset).with_context(|| {
                format!(
                    "shard {}: tensor {name}: offset {offset} + tensor data offset {data_offset} \
                     overflows u64",
                    shard_path.display(),
                )
            })?;
            if let Some(prev) = shard_of.insert(name.to_string(), idx) {
                bail!(
                    "duplicate tensor {name}: appears in both {} and {}",
                    parts[prev].0.display(),
                    shard_path.display()
                );
            }
            Ok(rebased)
        };
        for (name, mut info) in known {
            info.offset = rebase(&name, info.offset)?;
            tensor_infos.insert(name, info);
        }
        for (name, mut info) in unknown {
            info.offset = rebase(&name, info.offset)?;
            raw_of.insert(name, info);
        }
    }
    let total_tensors = tensor_infos.len() + raw_of.len();
    ensure!(
        total_tensors as u64 == expected_tensors,
        "split GGUF {}: its shards carry {total_tensors} tensors but {SPLIT_TENSORS_COUNT_KEY} \
         says {expected_tensors}",
        handed.display(),
    );

    let content = Content {
        magic: parts[0].2.content.magic,
        metadata: std::mem::take(&mut parts[0].2.content.metadata),
        tensor_infos,
        tensor_data_offset: 0,
    };
    let mmap_wanted = matches!(device, Device::Metal(_)) && !load_classic();
    let mut shards = Vec::with_capacity(parts.len());
    for (path, file, _) in parts {
        let mmap = if mmap_wanted {
            Some(MmapSource::open(&path, device)?)
        } else {
            None
        };
        shards.push(Shard {
            path,
            file: Mutex::new(file),
            mmap,
        });
    }
    Ok(Arc::new(GgufFile {
        content,
        device: device.clone(),
        path: SingleFilePath {
            path: shards[0].path.clone(),
            shard_count: shards.len(),
        },
        checkpoint,
        shards,
        shard_of,
        raw_of,
    }))
}

/// A quantized (or dense — QMatMul dequantizes F32/F16 sources) linear layer.
pub struct QLinear {
    inner: QMatMul,
    pub in_dim: usize,
    pub out_dim: usize,
    /// A raw-bytes view of the SAME allocation `inner` matmuls against, when the
    /// loader had one to hand out (`qlinear_with_buffer`) and a vendored kernel
    /// is instantiated for the dtype. Present only so `forward` can route the
    /// small-batch token window to `ops::matmul_mv_ext`; `None` leaves every
    /// token count on `QMatMul`, which is what the plain `qlinear` loader gives
    /// (it never retains a buffer) and therefore what the XWEN_ATTN_F32 parity
    /// path keeps.
    plane: Option<QuantPlane>,
    /// Whether `forward` may take the small-batch (2..=8 token) mv_ext window
    /// when a plane is present. True everywhere except the hyper-connection
    /// bottleneck: its planes exist ONLY for the prefill gemm (`forward_gemm`),
    /// and letting them open the window would change the 2..8-token numerics of
    /// every hc path — `XWEN_HC_CLASSIC` included, which promises the pre-plane
    /// candle chain (`without_mv_ext`).
    mv_ext_ok: bool,
}

impl QLinear {
    /// Weight bytes one call streams, for the `XWEN_GDN_PROFILE` byte floor.
    ///
    /// Exact when the loader handed out a raw plane (the bytes both the
    /// quantized matmul and the small-batch kernel read). Without one this is
    /// the dense-f32 UPPER BOUND, which is what the `XWEN_ATTN_F32` parity path
    /// actually holds — a quantized-stored weight kept behind a bare `QMatMul`
    /// would be narrower, and that combination is off every shipped route.
    pub fn weight_bytes(&self) -> u64 {
        match &self.plane {
            Some(p) => ((p.out_dim * p.in_dim) / p.dtype.block_size() * p.dtype.type_size()) as u64,
            None => (self.in_dim * self.out_dim * 4) as u64,
        }
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // candle's Metal quantized matmul rebuilds the input layout from its
        // SHAPE (quantized/metal.rs, call_quantized_matmul_mm_t) and so silently
        // drops any storage start_offset — a dim-0 narrowed view reads the wrong
        // rows. `.contiguous()` cannot repair that (an offset-only view still
        // passes the contiguity check and no-ops), so genuinely materialize such
        // inputs via the zeros_like + slice_set blit (Tensor::copy is a shallow
        // Arc clone on Metal — see CLAUDE.md).
        let x = if !x.is_contiguous() {
            x.contiguous()?
        } else {
            x.clone()
        };
        let x = if x.layout().start_offset() != 0 {
            let out = x.zeros_like()?;
            out.slice_set(&x, 0, 0)?;
            out
        } else {
            x
        };
        // Small-batch window: one weight pass serves 2..8 token rows, where
        // QMatMul would run candle's `mul_mm` at a fraction of the achievable
        // bandwidth. Asked after the normalization above, so the kernel always
        // sees a contiguous, offset-0 rank-2 activation. Any other rank or a
        // token count outside the window falls through to QMatMul unchanged.
        if let Some(plane) = &self.plane
            && self.mv_ext_ok
            && let Ok((t, _)) = x.dims2()
            && let Some(r1ptg) = crate::ops::mv_ext_window(t)
            && crate::ops::mv_ext_supported(plane.dtype, plane.in_dim)
        {
            return crate::ops::matmul_mv_ext(plane, &x, r1ptg)
                .map_err(|e| candle_core::Error::Msg(format!("{e:?}")));
        }
        self.inner.forward(&x)
    }

    /// [`forward`](Self::forward), plus the vendored dense cooperative-tensor
    /// gemm (`ops::matmul_dense_q`) for a rank-2 f32 activation of more than
    /// `ops::dense_mm_min_seq()` rows — the route the 27B's dense FFN takes at
    /// prefill, offered here to any planed projection whose caller opts in (the
    /// MoE shared expert, the hyper-connection bottleneck). Everything else —
    /// fewer rows, no plane, a dtype the gemm is not instantiated for, other
    /// ranks, or `XWEN_DENSE_MM_CLASSIC` — is exactly `forward`.
    ///
    /// The gemm is the reduced-precision class (~4e-4 rel_l2 from the f32
    /// oracle, docs/parity.md §3b), not `QMatMul`'s ~2e-4, which is why it is
    /// opt-in per caller and off under the same switch the strict tier pins.
    pub fn forward_gemm(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        if let Some(plane) = &self.plane
            && !crate::ops::dense_mm_classic()
            && x.dtype() == DType::F32
            && let Ok((t, _)) = x.dims2()
            && t > crate::ops::dense_mm_min_seq()
            && crate::ops::dense_mm_supported(plane.dtype, plane.in_dim)
        {
            // The kernel reads x straight from device memory, so a strided view
            // has to be materialized; an offset-only view is fine (the dispatch
            // binds the start offset).
            let x = if x.is_contiguous() {
                x.clone()
            } else {
                x.contiguous()?
            };
            return crate::ops::matmul_dense_q(plane, &x)
                .map_err(|e| candle_core::Error::Msg(format!("{e:?}")));
        }
        self.forward(x)
    }

    /// Wraps an already-loaded rank-2 weight `[out_dim, in_dim]` as a linear layer,
    /// for callers holding a QTensor directly rather than a GGUF tensor name.
    pub fn from_qtensor(qt: Arc<QTensor>) -> Result<Self> {
        let dims = qt.shape().dims().to_vec();
        let [out_dim, in_dim] = dims[..] else {
            bail!("QLinear source is not a rank-2 weight: {dims:?}");
        };
        Ok(QLinear {
            inner: QMatMul::from_arc(qt)?,
            in_dim,
            out_dim,
            plane: None,
            mv_ext_ok: true,
        })
    }

    /// Wraps an already-loaded rank-2 weight together with a raw view of ITS
    /// OWN device allocation, exactly as `Weights::qlinear_with_buffer` pairs
    /// them for file-loaded tensors — for tests and benches that build
    /// synthetic weights and need the planed routes (`forward_gemm`, the
    /// mv_ext window) reachable. Keeps the loader's support predicate, so an
    /// unsupported dtype/width yields a plane-less layer, not an error.
    pub fn from_qtensor_with_buffer(qt: Arc<QTensor>, buffer: Arc<Buffer>) -> Result<Self> {
        let dtype = qt.dtype();
        let mut lin = Self::from_qtensor(qt)?;
        lin.plane = (crate::ops::dense_mm_supported(dtype, lin.in_dim)
            || crate::ops::mv_ext_supported(dtype, lin.in_dim))
        .then(|| QuantPlane {
            buffer,
            base_off: 0,
            dtype,
            out_dim: lin.out_dim,
            in_dim: lin.in_dim,
        });
        Ok(lin)
    }

    /// The weight's raw quantized bytes, when the loader retained the
    /// allocation (`qlinear_with_buffer`) and a vendored kernel is instantiated
    /// for the dtype — `None` otherwise, and the caller then keeps whatever path
    /// it had. For the kernels that read the GGUF block layout themselves rather
    /// than going through `forward`/`forward_gemm`: the fused hyper-connection
    /// decode gate (`ops::hc_gate_down`) is the one that does.
    pub fn plane(&self) -> Option<&QuantPlane> {
        self.plane.as_ref()
    }

    /// The same layer with the small-batch mv_ext window disabled: `forward`
    /// is candle's `QMatMul` at every token count (bitwise the plane-less
    /// loader's behavior), while `forward_gemm` keeps the plane. For
    /// projections whose plane exists only for the prefill gemm — the
    /// hyper-connection bottleneck.
    pub fn without_mv_ext(mut self) -> Self {
        self.mv_ext_ok = false;
        self
    }
}

/// A rank-2 quantized weight `[out_dim, in_dim]` as raw device bytes, for the
/// vendored kernels that read the GGUF layout directly: the dense
/// cooperative-tensor prefill gemm (`ops::matmul_dense_q`) and the small-batch
/// mat-vec (`ops::matmul_mv_ext`). The buffer is the SAME allocation the weight
/// already lives in — one `QStorage::from_data` upload shared with the companion
/// `QLinear` (`qlinear_with_plane`), or the attention alias's own view
/// (`attn_proj`) — so a plane costs no extra device memory. That matters here
/// more than anywhere else in the loader: the 27B's FFN weights are 14 of its
/// 19 GB, and a materialized f16 copy of them would not fit at all.
#[derive(Clone)]
pub struct QuantPlane {
    /// The weight's quantized bytes as a raw device buffer.
    pub buffer: Arc<Buffer>,
    /// Byte offset of the weight's first block inside `buffer` — 0 for the
    /// dedicated allocation `qlinear_with_plane` builds. Every dispatch
    /// consuming `buffer` must add it.
    pub base_off: usize,
    pub dtype: GgmlDType,
    pub out_dim: usize,
    pub in_dim: usize,
}

/// Stacked per-expert weights kept in their quantized GGUF layout,
/// `[n_expert, n_out, k]`, whose device buffer the `ops::{mv_id,mm_id}`
/// kernels index directly by expert id.
pub struct ExpertStack {
    /// The stack as a QTensor, on the classic copying load. `None` on the mmap
    /// alias load — the fused Metal kernels read only `buffer` plus the
    /// shape/dtype fields, and every consumer that needs real QTensors
    /// (`ReferenceExperts`, `split_expert_stack`) loads its own via
    /// `expert_qtensors`, which always copies.
    pub qtensor: Option<Arc<QTensor>>,
    /// The stack's quantized bytes as a raw device buffer, for the fused
    /// `ops::{mv_id,mm_id}` kernels. Classic load: a retained handle to the
    /// SAME `MTLBuffer` that backs `qtensor` (both were cloned from one
    /// `QStorage`), so the fused path indexes the resident weights with no
    /// second upload. Mmap load: a page-floored no-copy view over the GGUF
    /// mapping (see `MmapSource::view`). `None` off Metal — the Reference
    /// runner uses `expert_qtensors`.
    pub buffer: Option<Arc<Buffer>>,
    /// Byte offset of the stack's first block inside `buffer`: 0 on the classic
    /// path (dedicated allocation), the sub-page remainder of the tensor's file
    /// offset on the mmap path (< page size, 32-byte aligned per GGUF data
    /// alignment). Every kernel dispatch consuming `buffer` must add it
    /// (dispatch.rs `IdDispatch.w_off`).
    pub base_off: usize,
    /// Keeps the file mapping (and its residency set) alive while `buffer`
    /// aliases it — `MmapSource`'s lifetime invariant. `None` on the classic
    /// path.
    pub mmap: Option<Arc<MmapSource>>,
    pub dtype: GgmlDType,
    pub n_expert: usize,
    pub n_out: usize,
    pub k: usize,
}

/// The vendored q8_0 attention decode gemv covers output rows in `N_R0`-row
/// groups (`kernel_mul_mv_q8_0_f32_attn`, N_R0_Q8_0 = 2 in ggml-metal-impl.h;
/// mirrored by `ops`'s `MV_Q8_NR0`). ggml guards only the STORE against a ragged
/// final group — the COMPUTE reads the whole `N_R0`-row group — so the classic
/// private weight copy is padded up to a whole multiple of this many rows.
///
/// The ragged over-read is ggml's own design, not a defect — see the canonical
/// statement in `mv.metal`'s K-quant tail comment ("benign by design — do not
/// 'fix' by guarding the compute loop"). The store guard
/// (`helper_mv_reduce_and_write`'s `r0 + row < ne01`) keeps the RESULT correct
/// whatever the extra row contains, so this is only ever a question of whether
/// those bytes are readable. The two loaders differ, and only one is covered:
///
///   * classic copy: padded explicitly to a whole row group (below), so the
///     extra row is always inside the buffer.
///   * mmap alias: NOT covered, and not claimed to be. `MmapSource::view`
///     page-ceils to the TENSOR's own extent, so the slack after the last row is
///     a uniform 0..`page - 1` bytes, while one extra row is `in_dim / 32 * 34` =
///     3264 / 6528 / 9792 bytes at the production `in_dim`s (3072 / 6144 / 9216).
///     It therefore leaves the MTLBuffer a large fraction of the time, and past a
///     buffer's length the access is Metal-UNDEFINED. The whole-file CPU mapping
///     is no defense: each of the ~380 views is its own `newBufferWithBytesNoCopy`
///     buffer with its own GPU VA range, so adjacency in the file says nothing
///     about what the GPU dereferences past this buffer's end. (The K-quant stack
///     over-read that `mv.metal` blesses is a different case — it stays inside
///     ONE buffer.)
///
/// This is safe only because it is UNREACHABLE, not because it is guarded: every
/// Laguna attention `out_dim` is even — 6144/9216 (q), 1024 (k/v), 3072 (o),
/// n_head ∈ {48, 72} (gate) — so the final group is never ragged on a real
/// checkpoint. `ops`'s ragged-row tests do exercise the odd case, but on pool
/// allocations, where the phantom row lands in same-heap slack — the same
/// unguaranteed mechanism, with kinder odds. A checkpoint with an odd attention
/// `out_dim` would need a load-time refusal here.
const Q8_DECODE_NR0: usize = 2;

/// The raw q8_0 bytes of one attention projection weight `[out_dim, in_dim]`,
/// as a device buffer for the vendored decode kernels — the single-token gemv
/// (`ops::matmul_q8`) and the small-batch mat-vec (`ops::matmul_mv_ext`). Loaded
/// ONLY for a q8_0-attention checkpoint (the current official file — so this is
/// the production decode path; the unsloth UD-Q4_K_XL file that motivated it is
/// deleted); the dense f16 plane that carries the prefill/mm path lives
/// alongside it in `Proj`. Mmap load: a page-floored no-copy view over the GGUF
/// mapping, with `base_off` the tensor's byte offset inside the view. Classic
/// load: a dedicated private buffer (`base_off` 0).
pub struct AttnQ8 {
    /// The q8_0 bytes as a rank-2 plane over a raw device buffer (aliased view
    /// or private copy). Both decode kernels read it through this one
    /// description — a `QuantPlane` here is a VIEW of the bytes the gemv already
    /// used, not a second upload, so the small-batch route costs no device
    /// memory.
    pub plane: QuantPlane,
    /// Keeps the file mapping (and its residency set) alive while the plane's
    /// buffer aliases it — `MmapSource`'s lifetime invariant. `None` on the
    /// classic path.
    pub mmap: Option<Arc<MmapSource>>,
}

/// VarBuilder-shaped accessor: `w.pp("blk.0").qlinear("attn_q")` reads `blk.0.attn_q.weight`.
#[derive(Clone)]
pub struct Weights {
    src: Arc<GgufFile>,
    prefix: String,
}

impl Weights {
    pub fn from_gguf(src: Arc<GgufFile>) -> Self {
        Self {
            src,
            prefix: String::new(),
        }
    }

    pub fn pp(&self, p: impl AsRef<str>) -> Weights {
        let prefix = if self.prefix.is_empty() {
            p.as_ref().to_string()
        } else {
            format!("{}.{}", self.prefix, p.as_ref())
        };
        Self {
            src: self.src.clone(),
            prefix,
        }
    }

    pub fn device(&self) -> &Device {
        &self.src.device
    }

    /// Whether `<prefix>.<name>.weight` is in the tensor table — either half of
    /// it, so a presence probe for an optional plane answers the same whether
    /// or not candle can name the dtype it happens to be stored at.
    pub fn has(&self, name: &str) -> bool {
        self.src.has_tensor(&self.name(name))
    }

    fn name(&self, n: &str) -> String {
        if self.prefix.is_empty() {
            format!("{n}.weight")
        } else {
            format!("{}.{n}.weight", self.prefix)
        }
    }

    pub fn qtensor(&self, name: &str) -> Result<Arc<QTensor>> {
        let full = self.name(name);
        let mut file = self.src.shard_for(&full).file.lock().unwrap();
        let qt = self
            .src
            .content
            .tensor(&mut *file, &full, &self.src.device)
            .with_context(|| format!("loading tensor {full}"))?;
        Ok(Arc::new(qt))
    }

    pub fn qlinear(&self, name: &str) -> Result<QLinear> {
        let qt = self.qtensor(name)?;
        let dims = qt.shape().dims().to_vec();
        let [out_dim, in_dim] = dims[..] else {
            bail!("{} is not a rank-2 weight: {dims:?}", self.name(name));
        };
        Ok(QLinear {
            inner: QMatMul::from_arc(qt)?,
            in_dim,
            out_dim,
            mv_ext_ok: true,
            // This loader never retains the device buffer, so there is nothing
            // to hand a vendored kernel; `qlinear_with_buffer` is the one that
            // can. Deliberate for the XWEN_ATTN_F32 attention projections, the
            // only production user: that path IS the parity oracle and must
            // keep the exact QMatMul chain the references were blessed with.
            plane: None,
        })
    }

    /// Loads a rank-2 weight `[out_dim, in_dim]` as a `QLinear`, additionally
    /// returning a retained handle to its Metal buffer (or `None` off Metal) so a
    /// caller can dispatch the vendored plain mat-vec kernel over the SAME
    /// allocation the `QLinear` uses — no second upload. Same zero-copy
    /// construction as `expert_stack`: read the quantized bytes ourselves, upload
    /// once via `QStorage::from_data`, retain the buffer BEFORE the storage moves
    /// into the `QTensor`. Used for the lm_head decode bypass (q8_0 on the
    /// current official file, q6_K on the retired original); a non-Metal
    /// device yields `buffer = None` and the caller stays on `QLinear::forward`.
    pub fn qlinear_with_buffer(
        &self,
        name: &str,
    ) -> Result<(QLinear, Option<Arc<Buffer>>, GgmlDType)> {
        let full = self.name(name);
        let info = self
            .src
            .content
            .tensor_infos
            .get(&full)
            .with_context(|| format!("tensor {full} not found"))?;
        let dims = info.shape.dims().to_vec();
        let [out_dim, in_dim] = dims[..] else {
            bail!("{full} is not a rank-2 weight: {dims:?}");
        };
        let dtype = info.ggml_dtype;
        let block = dtype.block_size();
        let elems = out_dim * in_dim;
        if !elems.is_multiple_of(block) {
            bail!("{full}: {elems} elements not a multiple of {dtype:?} block size {block}");
        }
        let size_in_bytes = elems / block * dtype.type_size();
        let tensor_start = self.src.content.tensor_data_offset + info.offset;

        let mut raw = vec![0u8; size_in_bytes];
        {
            let mut file = self.src.shard_for(&full).file.lock().unwrap();
            file.seek(SeekFrom::Start(tensor_start))
                .with_context(|| format!("seeking to {full}"))?;
            file.read_exact(&mut raw)
                .with_context(|| format!("reading {full} ({size_in_bytes} bytes)"))?;
        }

        let storage = QStorage::from_data(std::borrow::Cow::Owned(raw), &self.src.device, dtype)?;
        let buffer = match &storage {
            QStorage::Metal(qms) => Some(Arc::new(qms.buffer().clone())),
            _ => None,
        };
        let qtensor = Arc::new(QTensor::new(storage, (out_dim, in_dim))?);
        // The plane views the same allocation the QMatMul reads, so it costs no
        // device memory. Built whenever a vendored kernel could want it; each
        // consumer still checks its own support predicate before dispatching.
        let plane = buffer
            .clone()
            .filter(|_| {
                crate::ops::dense_mm_supported(dtype, in_dim)
                    || crate::ops::mv_ext_supported(dtype, in_dim)
            })
            .map(|buffer| QuantPlane {
                buffer,
                base_off: 0,
                dtype,
                out_dim,
                in_dim,
            });
        let qlinear = QLinear {
            inner: QMatMul::from_arc(qtensor)?,
            in_dim,
            out_dim,
            plane,
            mv_ext_ok: true,
        };
        Ok((qlinear, buffer, dtype))
    }

    /// A rank-2 weight `[out_dim, in_dim]` as a `QLinear` PLUS a `QuantPlane`
    /// view of the same device allocation, so prefill can dispatch the vendored
    /// dense cooperative-tensor gemm over the identical bytes `QLinear::forward`
    /// reads at decode. Built on `qlinear_with_buffer` — one upload, two views,
    /// no extra device memory.
    ///
    /// The plane is `None` off Metal (no buffer to hand out) and for any dtype
    /// or `in_dim` the vendored kernel is not instantiated for; the caller then
    /// stays on `QLinear::forward` everywhere.
    pub fn qlinear_with_plane(&self, name: &str) -> Result<(QLinear, Option<QuantPlane>)> {
        let (qlinear, _buffer, dtype) = self.qlinear_with_buffer(name)?;
        let in_dim = qlinear.in_dim;
        // The QLinear already holds a plane whenever any vendored kernel is
        // instantiated for the weight; the one handed out here is the DENSE
        // gemm's, so it keeps that kernel's narrower predicate.
        let plane = qlinear
            .plane
            .clone()
            .filter(|_| crate::ops::dense_mm_supported(dtype, in_dim));
        Ok((qlinear, plane))
    }

    pub fn rms_norm(&self, name: &str, eps: f64) -> Result<RmsNorm> {
        let w = self.dense_f32(name)?;
        Ok(RmsNorm::new(w, eps))
    }

    /// A small tensor needed densely on-device (norm weights, router, exp_probs_b),
    /// dequantized to f32 whatever its stored dtype.
    pub fn dense_f32(&self, name: &str) -> Result<Tensor> {
        let qt = self.qtensor(name)?;
        Ok(qt
            .dequantize(&self.src.device)?
            .to_dtype(candle_core::DType::F32)?)
    }

    /// A rank-2 weight `[out_dim, in_dim]` as a dense f16 tensor, for layers
    /// whose matmuls run natively in f16 (an f16-attention checkpoint's weights
    /// stay at their stored precision; a quantized-stored one — q8_0 on the
    /// current official file — dequantizes to f16 here. `QMatMul`'s dense f32
    /// would double the streamed bytes).
    ///
    /// On the mmap load an F16-stored weight ALIASES the file's page cache
    /// (`f16_alias_tensor`) instead of being read + re-uploaded: the GGUF F16
    /// bytes ARE the dense plane, and the classic `dequantize_f16` round-trip
    /// (f16→f32→f16) is exact, so the two paths are bit-identical. Any other
    /// stored dtype (or a classic open) takes the copying path, whose f32
    /// intermediate is dropped before this returns; no f32 copy stays alive.
    pub fn dense_f16(&self, name: &str) -> Result<Tensor> {
        let full = self.name(name);
        if let Some(src) = self.src.shard_for(&full).mmap.as_ref() {
            let info = self
                .src
                .content
                .tensor_infos
                .get(&full)
                .with_context(|| format!("tensor {full} not found"))?;
            if info.ggml_dtype == GgmlDType::F16 {
                let dims = info.shape.dims().to_vec();
                let [out_dim, in_dim] = dims[..] else {
                    bail!("{full} is not a rank-2 weight: {dims:?}");
                };
                let tensor_start = (self.src.content.tensor_data_offset + info.offset) as usize;
                return f16_alias_tensor(src, &self.src.device, tensor_start, out_dim, in_dim)
                    .with_context(|| format!("mmap-aliasing {full}"));
            }
        }
        let qt = self.qtensor(name)?;
        let t = qt.dequantize_f16(&self.src.device)?;
        let dims = t.dims().to_vec();
        let [_out_dim, _in_dim] = dims[..] else {
            bail!("{} is not a rank-2 weight: {dims:?}", self.name(name));
        };
        Ok(t)
    }

    /// The dtype `<prefix>.<name>.weight` is STORED at, before any dequantize.
    /// A caller whose kernel choice depends on the storage (the QSA indexer,
    /// whose projections upstream keeps off the quantizer's list) asks this
    /// rather than guessing from the file's `general.file_type`, which
    /// describes the mix and not any one tensor.
    ///
    /// Answers only for the dtypes candle names, because every caller's next
    /// move is to load the tensor through candle. A tensor stored at a type
    /// candle has no variant for (the IQ4_NL PLE table) is reported as such
    /// rather than as missing — it is in the file, it is just not loadable this
    /// way; `GgufFile::stored_dtype_of` is the accessor that spans both halves.
    pub fn stored_dtype(&self, name: &str) -> Result<GgmlDType> {
        let full = self.name(name);
        match self.src.stored_dtype_of(&full)? {
            StoredDtype::Ggml(d) => Ok(d),
            StoredDtype::Raw(d) => bail!(
                "tensor {full} is stored as {}, which candle cannot represent — it is reachable \
                 only through GgufFile::raw_tensor",
                d.name()
            ),
        }
    }

    /// A rank-2 weight `[out_dim, in_dim]` stored BF16, as a dense bf16 tensor
    /// for `ops::matmul_bf16`. The `qwen4exp` QSA indexer's `indexer.q_proj` /
    /// `indexer.k_proj` are the callers: upstream's converter puts both on the
    /// quantizer's skip list, so they arrive at the source precision (BF16) in
    /// every quant mix, and candle's `QMatMul` has no bf16 route.
    ///
    /// On an mmap load the GGUF bytes ARE the dense plane, so this aliases the
    /// page cache exactly as `dense_f16` does. `ensure_bf16_fits_f16` runs
    /// first for the same reason it runs on the drafter planes: the bf16
    /// tensor-path gemm stages bf16 → half, and a value outside f16's range
    /// would become inf mid-kernel.
    ///
    /// A file that ships these at some other dtype (a self-conversion that let
    /// the quantizer touch them) takes the copying path — dequantize, then
    /// round to bf16 — rather than failing to load.
    pub fn dense_bf16(&self, name: &str) -> Result<Tensor> {
        let full = self.name(name);
        let info = self
            .src
            .content
            .tensor_infos
            .get(&full)
            .with_context(|| format!("tensor {full} not found"))?;
        let dims = info.shape.dims().to_vec();
        let [out_dim, in_dim] = dims[..] else {
            bail!("{full} is not a rank-2 weight: {dims:?}");
        };
        if info.ggml_dtype == GgmlDType::BF16
            && let Some(src) = self.src.shard_for(&full).mmap.as_ref()
        {
            let abs_off = (self.src.content.tensor_data_offset + info.offset) as usize;
            let len = out_dim * in_dim * DType::BF16.size_in_bytes();
            crate::dflash::ensure_bf16_fits_f16(src.bytes(abs_off, len)?, &full)?;
            return dense_alias_tensor(
                src,
                &self.src.device,
                abs_off,
                out_dim,
                in_dim,
                DType::BF16,
            )
            .with_context(|| format!("mmap-aliasing {full}"));
        }
        let qt = self.qtensor(name)?;
        Ok(qt.dequantize(&self.src.device)?.to_dtype(DType::BF16)?)
    }

    /// One attention projection weight `[out_dim, in_dim]`: the dense f16 plane
    /// (`dense_f16` — the prefill/mm path) plus, ONLY when the GGUF stores the
    /// weight as Q8_0, the raw q8_0 bytes as an `AttnQ8` for the vendored decode
    /// kernels — the single-token gemv (`ops::matmul_q8`) and the small-batch
    /// mat-vec (`ops::matmul_mv_ext`), which read the same plane. An F16-stored
    /// weight (an f16-attention
    /// checkpoint, like the retired original) returns `None` for the q8 handle
    /// and stays on the f16 path everywhere — no extra load work.
    ///
    /// The f16 plane and the q8_0 bytes are two views of the SAME weight: the plane
    /// is `dequantize_f16` of the q8_0 tensor, the alias is the untouched q8_0
    /// bytes. The two decode paths across the `Q8_DECODE_MAX_SEQ` seq boundary are
    /// NOT bit-identical: the f16 plane rounds each dequantized value `d·q_i` to
    /// f16 (one extra rounding per weight element) and the seq > boundary path
    /// multiplies those f16 weights by the f32 activation, whereas the seq <=
    /// boundary q8 gemv multiplies the raw int8 quants by `d` and accumulates in
    /// f32 with no such rounding. Both inherit q8_0's quantization error, so they
    /// agree to that plus the f16 plane's per-element rounding — the same
    /// numerical class as the f16 path's OWN gemv/gemm split (already ulp-different
    /// across its seq boundary). The discontinuity is accepted design: the decode
    /// parity tier grades greedy/perplexity statistically, not bitwise.
    pub fn attn_proj(&self, name: &str) -> Result<(Tensor, Option<AttnQ8>)> {
        let f16 = self.dense_f16(name)?;
        let full = self.name(name);
        let info = self
            .src
            .content
            .tensor_infos
            .get(&full)
            .with_context(|| format!("tensor {full} not found"))?;
        if info.ggml_dtype != GgmlDType::Q8_0 {
            return Ok((f16, None));
        }
        let dims = info.shape.dims().to_vec();
        let [out_dim, in_dim] = dims[..] else {
            bail!("{full} is not a rank-2 weight: {dims:?}");
        };
        let dtype = info.ggml_dtype;
        let block = dtype.block_size();
        let elems = out_dim * in_dim;
        if !elems.is_multiple_of(block) {
            bail!("{full}: {elems} elements not a multiple of {dtype:?} block size {block}");
        }
        let size_in_bytes = elems / block * dtype.type_size();
        let tensor_start = (self.src.content.tensor_data_offset + info.offset) as usize;

        let shard = self.src.shard_for(&full);
        let (buffer, base_off, mmap) = match shard.mmap.as_ref() {
            Some(src) => {
                let (buffer, base_off) = src
                    .view(tensor_start, size_in_bytes)
                    .with_context(|| format!("mmap-aliasing {full}"))?;
                // The gemv walks whole q8_0 blocks (half delta + int8 quants), so
                // the bound offset needs 2-byte alignment; every real GGUF is
                // 32-aligned (this guards hand-crafted files).
                ensure!(
                    base_off.is_multiple_of(2),
                    "{full}: mmap base_off {base_off} is not 2-byte aligned"
                );
                (buffer, base_off, Some(src.clone()))
            }
            None => {
                // Classic copy: read the raw q8_0 bytes into a fresh private buffer
                // (base_off 0). The dense f16 plane above is an independent
                // allocation on this path, so no allocation is shared.
                //
                // The decode gemv reads output rows in Q8_DECODE_NR0-row groups and
                // (ggml convention) reads the whole group even when the final one is
                // ragged — only the STORE is row-guarded. `new_buffer_with_data`
                // allocates exactly the data length (the mmap view is page-padded,
                // this is not), so an odd out_dim would put the last group's second
                // row past the buffer. Pad up to a whole Q8_DECODE_NR0-row multiple
                // with zeros; the padding rows are read and discarded by the guard.
                let bytes_per_row = in_dim / block * dtype.type_size();
                let padded_rows = out_dim.div_ceil(Q8_DECODE_NR0) * Q8_DECODE_NR0;
                let mut raw = vec![0u8; padded_rows * bytes_per_row];
                {
                    let mut file = shard.file.lock().unwrap();
                    file.seek(SeekFrom::Start(tensor_start as u64))
                        .with_context(|| format!("seeking to {full}"))?;
                    file.read_exact(&mut raw[..size_in_bytes])
                        .with_context(|| format!("reading {full} ({size_in_bytes} bytes)"))?;
                }
                let Device::Metal(mdev) = &self.src.device else {
                    bail!("q8_0 attention weights require a Metal device");
                };
                let buffer = mdev.new_buffer_with_data(&raw)?;
                (buffer, 0usize, None)
            }
        };
        Ok((
            f16,
            Some(AttnQ8 {
                plane: QuantPlane {
                    buffer,
                    base_off,
                    dtype,
                    out_dim,
                    in_dim,
                },
                mmap,
            }),
        ))
    }

    /// Loads a stacked expert tensor `[n_expert, n_out, k]` for the fused MoE
    /// kernels. Default (mmap open): the quantized bytes stay in the file's
    /// page cache and the kernels read them through a no-copy view. Classic
    /// open: read + upload once, sharing the allocation with a wrapping
    /// QTensor.
    pub fn expert_stack(&self, name: &str) -> Result<ExpertStack> {
        let full = self.name(name);
        match self.src.shard_for(&full).mmap.clone() {
            Some(src) => self.expert_stack_mmap(name, &src),
            None => self.expert_stack_classic(name),
        }
    }

    /// The classic copying load: the fused MoE kernels and the wrapping
    /// `QTensor` share ONE device allocation. We read the quantized bytes from
    /// the file ourselves, upload them once via `QStorage::from_data`, retain a
    /// handle to that buffer, and only then wrap the same storage in a
    /// `QTensor` — so no second copy of the (large) expert weights ever lands
    /// in VRAM.
    fn expert_stack_classic(&self, name: &str) -> Result<ExpertStack> {
        let full = self.name(name);
        let info = self
            .src
            .content
            .tensor_infos
            .get(&full)
            .with_context(|| format!("expert stack tensor {full} not found"))?;
        let dims = info.shape.dims().to_vec();
        let [n_expert, n_out, k] = dims[..] else {
            bail!("{full} is not a rank-3 expert stack: {dims:?}");
        };
        let dtype = info.ggml_dtype;
        let block = dtype.block_size();
        let elems = n_expert * n_out * k;
        if !elems.is_multiple_of(block) {
            bail!("{full}: {elems} elements not a multiple of {dtype:?} block size {block}");
        }
        let size_in_bytes = elems / block * dtype.type_size();
        let tensor_start = self.src.content.tensor_data_offset + info.offset;

        let mut raw = vec![0u8; size_in_bytes];
        {
            let mut file = self.src.shard_for(&full).file.lock().unwrap();
            file.seek(SeekFrom::Start(tensor_start))
                .with_context(|| format!("seeking to {full}"))?;
            file.read_exact(&mut raw)
                .with_context(|| format!("reading {full} ({size_in_bytes} bytes)"))?;
        }

        let storage = QStorage::from_data(std::borrow::Cow::Owned(raw), &self.src.device, dtype)?;
        // Retain the storage's buffer before it moves into the QTensor: cloning a
        // candle Buffer retains the underlying MTLBuffer (no data copy), so this
        // handle and the QTensor point at the same allocation.
        let buffer = match &storage {
            QStorage::Metal(qms) => Some(Arc::new(qms.buffer().clone())),
            _ => None,
        };
        let qtensor = Arc::new(QTensor::new(storage, (n_expert, n_out, k))?);
        Ok(ExpertStack {
            qtensor: Some(qtensor),
            buffer,
            base_off: 0,
            mmap: None,
            dtype,
            n_expert,
            n_out,
            k,
        })
    }

    /// The mmap alias load: the stack's quantized bytes are never read into
    /// host memory — the fused kernels index a page-floored no-copy view of the
    /// GGUF mapping, with `base_off` marking the stack's first block inside the
    /// view. The GGUF stores expert stacks expert-major-contiguous (the classic
    /// path does no re-layout either), so the view IS the stack. No QTensor is
    /// built: nothing on the fused Metal path reads one, and the consumers that
    /// need real QTensors (`ReferenceExperts`, `split_expert_stack`) load
    /// theirs via `expert_qtensors`.
    fn expert_stack_mmap(&self, name: &str, src: &Arc<MmapSource>) -> Result<ExpertStack> {
        let full = self.name(name);
        let info = self
            .src
            .content
            .tensor_infos
            .get(&full)
            .with_context(|| format!("expert stack tensor {full} not found"))?;
        let dims = info.shape.dims().to_vec();
        let [n_expert, n_out, k] = dims[..] else {
            bail!("{full} is not a rank-3 expert stack: {dims:?}");
        };
        let dtype = info.ggml_dtype;
        let block = dtype.block_size();
        let elems = n_expert * n_out * k;
        if !elems.is_multiple_of(block) {
            bail!("{full}: {elems} elements not a multiple of {dtype:?} block size {block}");
        }
        let size_in_bytes = elems / block * dtype.type_size();
        let tensor_start = (self.src.content.tensor_data_offset + info.offset) as usize;

        let (buffer, base_off) = src
            .view(tensor_start, size_in_bytes)
            .with_context(|| format!("mmap-aliasing {full}"))?;
        // Same contract the f16 alias path enforces: the expert kernels do
        // vector loads from the bound offset, so it must stay 16-byte aligned
        // (every real GGUF is 32-aligned; this guards hand-crafted files).
        ensure!(
            base_off.is_multiple_of(16),
            "{full}: mmap base_off {base_off} is not 16-byte aligned"
        );
        Ok(ExpertStack {
            qtensor: None,
            buffer: Some(buffer),
            base_off,
            mmap: Some(src.clone()),
            dtype,
            n_expert,
            n_out,
            k,
        })
    }

    /// Loads a tensor by its fully-qualified GGUF name (dtype suffix included),
    /// bypassing the implicit `.weight` suffix that `qtensor` appends.
    fn qtensor_named(&self, full: &str) -> Result<Arc<QTensor>> {
        let mut file = self.src.shard_for(full).file.lock().unwrap();
        let qt = self
            .src
            .content
            .tensor(&mut *file, full, &self.src.device)
            .with_context(|| format!("loading tensor {full}"))?;
        Ok(Arc::new(qt))
    }

    /// Dense f32 for a small tensor whose GGUF name may not end in `.weight`.
    /// The DeltaNet layers carry both exceptions: `blk.N.ssm_dt.bias` is
    /// bias-suffixed (it is the dt offset vector, a bias in name only) and
    /// `blk.N.ssm_a` has no suffix at all. Tries `.weight`, then `.bias`, then
    /// the bare name.
    pub fn dense_f32_any(&self, name: &str) -> Result<Tensor> {
        let qualified = |suffix: &str| {
            if self.prefix.is_empty() {
                format!("{name}{suffix}")
            } else {
                format!("{}.{name}{suffix}", self.prefix)
            }
        };
        let full = [".weight", ".bias", ""]
            .into_iter()
            .map(qualified)
            .find(|n| self.src.content.tensor_infos.contains_key(n))
            .with_context(|| {
                format!(
                    "no tensor {} with a .weight, .bias or bare name",
                    qualified("")
                )
            })?;
        let qt = self.qtensor_named(&full)?;
        Ok(qt
            .dequantize(&self.src.device)?
            .to_dtype(candle_core::DType::F32)?)
    }

    /// Slices a stacked expert tensor `[n_expert, n_out, k]` into `n_expert`
    /// per-expert rank-2 QTensors `[n_out, k]`, keeping the quantized bytes
    /// (no dequantization). The stack is contiguous in expert-major order, so
    /// each expert's footprint is a fixed byte stride: its `n_out * k` elements
    /// form a whole number of quantization blocks.
    pub fn expert_qtensors(&self, name: &str) -> Result<Vec<Arc<QTensor>>> {
        let qt = self.qtensor(name)?;
        let dims = qt.shape().dims().to_vec();
        let [n_expert, n_out, k] = dims[..] else {
            bail!("{} is not a rank-3 expert stack: {dims:?}", self.name(name));
        };
        split_expert_stack(&qt, n_expert, n_out, k, &self.src.device)
    }
}

/// Byte-slices a stacked expert QTensor `[n_expert, n_out, k]` into per-expert
/// `[n_out, k]` QTensors, preserving the quantized layout. Kept free-standing so
/// callers holding a stack directly (not via GGUF) can reuse it.
pub fn split_expert_stack(
    stack: &QTensor,
    n_expert: usize,
    n_out: usize,
    k: usize,
    device: &Device,
) -> Result<Vec<Arc<QTensor>>> {
    let dtype = stack.dtype();
    let block = dtype.block_size();
    let type_size = dtype.type_size();
    let per_expert_elems = n_out * k;
    if per_expert_elems % block != 0 {
        bail!("expert size {per_expert_elems} is not a multiple of block size {block}");
    }
    let stride = per_expert_elems / block * type_size;
    let data = stack.data()?;
    if data.len() != stride * n_expert {
        bail!(
            "stacked expert data is {} bytes, expected {n_expert} x {stride}",
            data.len()
        );
    }
    let mut out = Vec::with_capacity(n_expert);
    for e in 0..n_expert {
        // to_vec() gives a fresh, over-aligned allocation, satisfying the block
        // struct's alignment requirement on the Metal/CUDA load paths.
        let bytes = data[e * stride..(e + 1) * stride].to_vec();
        let storage = QStorage::from_data(std::borrow::Cow::Owned(bytes), device, dtype)?;
        out.push(Arc::new(QTensor::new(storage, (n_out, k))?));
    }
    Ok(out)
}

/// Wraps `[out_dim, in_dim]` f16 bytes at absolute file offset `abs_off` of
/// `src`'s mapping as a dense f16 tensor WITHOUT copying — `dense_alias_tensor`
/// at `DType::F16` (the attention planes' path).
fn f16_alias_tensor(
    src: &MmapSource,
    device: &Device,
    abs_off: usize,
    out_dim: usize,
    in_dim: usize,
) -> Result<Tensor> {
    dense_alias_tensor(src, device, abs_off, out_dim, in_dim, DType::F16)
}

/// Wraps `[out_dim, in_dim]` dense 2-byte-element bytes (f16 or bf16) at
/// absolute file offset `abs_off` of `src`'s mapping as a tensor WITHOUT
/// copying: page-floored no-copy view → whole-view 1-D `MetalStorage` →
/// `narrow` off the sub-page `base_off` → `reshape`. The result is a
/// contiguous VIEW whose layout start_offset is `base_off / 2` elements; the
/// vendored `matmul_f16`/`matmul_bf16` dispatches honor
/// `layout.start_offset()` and require the byte offset 16-aligned, which the
/// GGUF's 32-byte tensor-data alignment guarantees. F16 callers: the attention
/// planes (`dense_f16`); BF16: the drafter's matmul planes (dflash.rs).
pub(crate) fn dense_alias_tensor(
    src: &MmapSource,
    device: &Device,
    abs_off: usize,
    out_dim: usize,
    in_dim: usize,
    dtype: DType,
) -> Result<Tensor> {
    let Device::Metal(mdev) = device else {
        bail!("mmap aliasing requires a Metal device");
    };
    ensure!(
        matches!(dtype, DType::F16 | DType::BF16),
        "dense_alias_tensor covers the 2-byte dense dtypes (F16/BF16), got {dtype:?}"
    );
    let elems = out_dim * in_dim;
    let (buffer, base_off) = src.view(abs_off, elems * dtype.size_in_bytes())?;
    // base_off is 32-byte aligned (GGUF data alignment), so it is a whole
    // number of elements and satisfies the dispatches' 16-byte view check.
    ensure!(
        base_off.is_multiple_of(16),
        "{dtype:?} alias at file offset {abs_off} is not 16-byte aligned inside its page (base_off {base_off})"
    );
    let count = buffer.length() / dtype.size_in_bytes(); // page-multiple, exact
    let storage = MetalStorage::new(buffer, mdev.clone(), count, dtype);
    let whole = Tensor::from_storage(
        Storage::Metal(storage),
        count,
        candle_core::op::BackpropOp::none(),
        false,
    );
    // The returned tensor's storage IS the no-copy view for the model's
    // lifetime: with the view registered in the queue-attached residency set,
    // Shared page-cache-backed planes stream at full rate (measured even with
    // the classic loader's driver-allocated buffers; docs/log.md mmap entry).
    // Do NOT "fix" this with Tensor::copy() — Metal's try_clone at the pinned
    // rev is a shallow Arc clone, not a data copy.
    Ok(whole
        .narrow(0, base_off / dtype.size_in_bytes(), elems)?
        .reshape((out_dim, in_dim))?)
}

/// Human-readable metadata + tensor listing for `xwen inspect`.
pub fn describe(content: &Content) -> String {
    let mut out = String::new();
    let mut keys: Vec<_> = content.metadata.keys().collect();
    keys.sort();
    for k in keys {
        let v = &content.metadata[k];
        let mut s = format!("{v:?}");
        if s.len() > 120 {
            let cut = (0..=117)
                .rev()
                .find(|&i| s.is_char_boundary(i))
                .unwrap_or(0);
            s.truncate(cut);
            s.push_str("...");
        }
        let _ = writeln!(out, "{k} = {s}");
    }
    let mut infos: Vec<_> = content.tensor_infos.iter().collect();
    infos.sort_by(|a, b| a.0.cmp(b.0));
    let _ = writeln!(out, "\n{} tensors:", infos.len());
    for (name, info) in infos {
        let _ = writeln!(
            out,
            "{name}  {:?}  {:?}",
            info.shape.dims(),
            info.ggml_dtype
        );
    }
    out
}

/// `describe` plus the half of the tensor table candle cannot name, which lives
/// on the `GgufFile` rather than on the `Content`. The whole-file listing an
/// `xwen inspect` wants: on an Unsloth Q3/Q4 Flash-Next mix, `describe` alone
/// silently omits the 28.8 GB IQ4_NL PLE table.
pub fn describe_file(gguf: &GgufFile) -> String {
    let mut out = describe(&gguf.content);
    let mut raw = gguf.raw_tensor_names();
    if raw.is_empty() {
        return out;
    }
    raw.sort_by(|a, b| a.0.cmp(b.0));
    let _ = writeln!(
        out,
        "\n{} tensors candle cannot name ({:.1} GB, never uploaded):",
        raw.len(),
        gguf.raw_tensor_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    for (name, dtype) in raw {
        let info = &gguf.raw_of[name];
        let _ = writeln!(out, "{name}  {:?}  {}", info.shape, dtype.name());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::gguf_file;

    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// The checkpoint id hashes the metadata section and nothing beyond it. The
    /// expected value is the published FNV-1a 64 of "foobar", so this pins the hash
    /// FUNCTION and not merely its self-consistency: a checkpoint id that changed
    /// meaning between builds would silently reject every image on disk.
    /// Everything past `metadata_len` — the tensor payload, tens of gigabytes on a
    /// real file — must leave the hash alone while still moving the file length,
    /// which is the other half of the id.
    #[test]
    fn checkpoint_id_hashes_the_metadata_section_only() {
        const FNV1A_FOOBAR: u64 = 0x8594_4171_f739_67e8;
        let path = std::env::temp_dir().join(format!("xwen_ckpt_id_{}.bin", std::process::id()));

        std::fs::write(&path, b"foobar").unwrap();
        let bare = CheckpointId::compute(&mut File::open(&path).unwrap(), 6).unwrap();
        assert_eq!(bare.hash(), FNV1A_FOOBAR, "FNV-1a 64 of the metadata bytes");
        assert_eq!(bare.file_len(), 6);
        assert_eq!(bare.dir_name(), format!("{FNV1A_FOOBAR:016x}"));

        std::fs::write(&path, b"foobar-and-a-tensor-payload").unwrap();
        let with_payload = CheckpointId::compute(&mut File::open(&path).unwrap(), 6).unwrap();
        assert_eq!(
            with_payload.hash(),
            bare.hash(),
            "the payload is outside the hashed section"
        );
        assert_eq!(
            with_payload.file_len(),
            27,
            "but the length covers the whole file"
        );
        assert_ne!(bare, with_payload, "so the two files are told apart");

        // A metadata section that claims to run past the file's end describes no
        // GGUF this loader could have parsed.
        assert!(
            CheckpointId::compute(&mut File::open(&path).unwrap(), 1 << 20).is_err(),
            "metadata section past the end of the file"
        );

        // Longer than the read chunk, so the streaming loop is exercised rather
        // than only its first pass.
        let long = vec![0x5au8; (1 << 20) + 3];
        std::fs::write(&path, &long).unwrap();
        let a = CheckpointId::compute(&mut File::open(&path).unwrap(), long.len() as u64).unwrap();
        let mut expect = 0xcbf2_9ce4_8422_2325u64;
        for &b in &long {
            expect ^= u64::from(b);
            expect = expect.wrapping_mul(0x100_0000_01b3);
        }
        assert_eq!(a.hash(), expect, "chunked read matches a single-pass hash");

        std::fs::remove_file(&path).unwrap();
    }

    /// QLinear::forward must read the RIGHT rows from an offset view: candle's
    /// Metal quantized matmul drops the input's storage start_offset (it rebuilds
    /// the layout from the shape), so a dim-0 `narrow` fed straight through would
    /// silently multiply the wrong rows. The guard materializes such views; this
    /// asserts narrow(0, 1, n) == the same rows copied into fresh storage.
    #[test]
    fn qlinear_forward_honors_row_offset_views() {
        let device = metal_device().unwrap();
        let (out_dim, in_dim, rows) = (16usize, 256usize, 4usize);

        let w = Tensor::from_vec(
            fill(out_dim * in_dim, 0xA7),
            (out_dim, in_dim),
            &Device::Cpu,
        )
        .unwrap();
        let qt = QTensor::quantize(&w.to_device(&device).unwrap(), GgmlDType::Q8_0).unwrap();
        let lin = QLinear::from_qtensor(Arc::new(qt)).unwrap();

        let x = Tensor::from_vec(fill(rows * in_dim, 0xB3), (rows, in_dim), &device).unwrap();
        let tail_view = x.narrow(0, 1, rows - 1).unwrap();
        // Genuinely materialized copy of the same rows (offset 0 storage).
        let tail_rows = Tensor::from_vec(
            x.to_device(&Device::Cpu).unwrap().to_vec2::<f32>().unwrap()[1..].concat(),
            (rows - 1, in_dim),
            &device,
        )
        .unwrap();

        let via_view = lin
            .forward(&tail_view)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let via_copy = lin
            .forward(&tail_rows)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(
            via_view, via_copy,
            "offset view multiplied different rows than its materialized copy"
        );
    }

    /// The fused expert stack must load as ONE device allocation shared with its
    /// QTensor — no second upload of the (large) weights. Writes a synthetic GGUF,
    /// loads it through the real `expert_stack` path, and checks the buffer is a
    /// single stack in size and holds the correct weights (the fused matvec through
    /// it matches dequantizing the stack's QTensor).
    #[test]
    fn expert_stack_loads_single_shared_buffer() {
        let device = metal_device().unwrap();
        let (n_expert, n_out, k) = (4usize, 8usize, 256usize);
        let dt = GgmlDType::Q4K;

        let w = Tensor::from_vec(
            fill(n_expert * n_out * k, 0xE1),
            (n_expert, n_out, k),
            &Device::Cpu,
        )
        .unwrap();
        let qt_cpu = QTensor::quantize(&w, dt).unwrap();
        let path =
            std::env::temp_dir().join(format!("xwen_expert_stack_{}.gguf", std::process::id()));
        {
            let mut f = File::create(&path).unwrap();
            gguf_file::write(&mut f, &[], &[("ffn_gate_exps.weight", &qt_cpu)]).unwrap();
        }

        let gguf = open(&path, &device).unwrap();
        let weights = Weights::from_gguf(gguf);
        // The CLASSIC path explicitly: this test pins the shared-allocation
        // invariant of the copying load (the default mmap path is covered by
        // `expert_stack_mmap_matches_classic`).
        let stack = weights.expert_stack_classic("ffn_gate_exps").unwrap();

        // One buffer, sized to exactly one stack (a double upload would not change
        // this length, but the size check catches a wrong-tensor / wrong-dtype load
        // and pairs with the by-construction single `from_data` in `expert_stack`).
        let expected = n_expert * n_out * k / dt.block_size() * dt.type_size();
        let buf = stack
            .buffer
            .as_ref()
            .expect("expert stack has a Metal buffer");
        assert_eq!(buf.length(), expected, "fused buffer must be one stack");
        let qtensor = stack
            .qtensor
            .as_ref()
            .expect("classic load carries a QTensor");
        assert_eq!(qtensor.storage_size_in_bytes(), expected);
        assert_eq!(stack.dtype, dt);
        assert_eq!(stack.base_off, 0, "classic load starts at the buffer head");

        // The shared buffer carries the right weights: a fused gather-matvec through
        // stack.buffer matches a CPU reference over the dequantized QTensor.
        let deq = qtensor
            .dequantize(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let (t, top_k) = (2usize, 2usize);
        let x_vec = fill(t * k, 0xC0FFEE);
        let x = Tensor::from_vec(x_vec.clone(), (t, 1, k), &device).unwrap();
        let ids_v: Vec<u32> = vec![0, 3, 1, 2];
        let ids = Tensor::from_vec(ids_v.clone(), (t, top_k), &device).unwrap();
        let got = crate::ops::mul_mv_id(&stack, &x, &ids)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let mut want = vec![0f32; t * top_k * n_out];
        for token in 0..t {
            for slot in 0..top_k {
                let e = ids_v[token * top_k + slot] as usize;
                for o in 0..n_out {
                    let mut acc = 0f32;
                    for i in 0..k {
                        acc += deq[(e * n_out + o) * k + i] * x_vec[token * k + i];
                    }
                    want[(token * top_k + slot) * n_out + o] = acc;
                }
            }
        }
        let (mut num, mut den) = (0f64, 0f64);
        for (g, wv) in got.iter().zip(&want) {
            num += (*g as f64 - *wv as f64).powi(2);
            den += (*wv as f64).powi(2);
        }
        let rel = (num / den.max(1e-30)).sqrt();
        assert!(
            rel < 1e-3,
            "fused-through-shared-buffer rel_l2 {rel} too high"
        );

        let _ = std::fs::remove_file(&path);
    }

    /// A fresh directory for a split-GGUF shard set, so sibling discovery only
    /// ever sees the shards the test wrote.
    fn split_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xwen_split_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The three KV keys every shard of a split GGUF carries, at the widths
    /// gguf-split writes them (`split.no`/`split.count` u16,
    /// `split.tensors.count` i32).
    fn split_kvs(no: u16, count: u16, total_tensors: i32) -> Vec<(&'static str, Value)> {
        vec![
            (SPLIT_NO_KEY, Value::U16(no)),
            (SPLIT_COUNT_KEY, Value::U16(count)),
            (SPLIT_TENSORS_COUNT_KEY, Value::I32(total_tensors)),
        ]
    }

    fn write_gguf(path: &Path, kvs: &[(&'static str, Value)], tensors: &[(&str, &QTensor)]) {
        let kv_refs: Vec<(&str, &Value)> = kvs.iter().map(|(k, v)| (*k, v)).collect();
        let mut f = File::create(path).unwrap();
        gguf_file::write(&mut f, &kv_refs, tensors).unwrap();
    }

    /// The full context chain of an `open` expected to fail.
    fn open_err(path: PathBuf) -> String {
        match open(&path, &Device::Cpu) {
            Ok(_) => panic!("open unexpectedly succeeded for {}", path.display()),
            Err(e) => format!("{e:#}"),
        }
    }

    fn q8_tensor(rows: usize, cols: usize, seed: u64) -> QTensor {
        let w = Tensor::from_vec(fill(rows * cols, seed), (rows, cols), &Device::Cpu).unwrap();
        QTensor::quantize(&w, GgmlDType::Q8_0).unwrap()
    }

    fn dequant(qt: &QTensor) -> Vec<f32> {
        qt.dequantize(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    /// A split GGUF opens from ANY shard's path into one logical file: the
    /// unified metadata is shard 0's complete KV block (shard 0 itself holding
    /// no tensors, the gguf-split layout), the tensor table is the union
    /// across shards, and every tensor reads back its own shard's bytes. The
    /// checkpoint id covers all shards, so it is identical whichever shard the
    /// set was opened through.
    #[test]
    fn split_gguf_presents_one_logical_file() {
        let dir = split_dir("logical");
        let (a, b, c) = (
            q8_tensor(4, 64, 0x5EED_A),
            q8_tensor(2, 96, 0x5EED_B),
            q8_tensor(3, 32, 0x5EED_C),
        );
        let mut meta = split_kvs(0, 3, 3);
        meta.push(("general.name", Value::String("Synthetic Split".into())));
        write_gguf(&dir.join("m-00001-of-00003.gguf"), &meta, &[]);
        write_gguf(
            &dir.join("m-00002-of-00003.gguf"),
            &split_kvs(1, 3, 3),
            &[("a.weight", &a)],
        );
        write_gguf(
            &dir.join("m-00003-of-00003.gguf"),
            &split_kvs(2, 3, 3),
            &[("b.weight", &b), ("c.weight", &c)],
        );

        let gguf = open(dir.join("m-00001-of-00003.gguf"), &Device::Cpu).unwrap();
        assert_eq!(
            gguf.content
                .metadata
                .get("general.name")
                .map(|v| v.to_string().unwrap().as_str()),
            Some("Synthetic Split"),
            "unified metadata is shard 0's KV block"
        );
        assert_eq!(gguf.content.tensor_infos.len(), 3, "union tensor table");
        let w = Weights::from_gguf(gguf.clone());
        for (name, src) in [("a", &a), ("b", &b), ("c", &c)] {
            let loaded = w.qtensor(name).unwrap();
            assert_eq!(loaded.shape(), src.shape());
            assert_eq!(
                dequant(&loaded),
                dequant(src),
                "tensor {name} read back the wrong bytes"
            );
        }

        // Opened through a different shard: same logical file, same identity.
        let via_last = open(dir.join("m-00003-of-00003.gguf"), &Device::Cpu).unwrap();
        assert_eq!(via_last.checkpoint_id(), gguf.checkpoint_id());
        let loaded = Weights::from_gguf(via_last).qtensor("a").unwrap();
        assert_eq!(dequant(&loaded), dequant(&a));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A complete single-file GGUF whose NAME happens to be split-shaped (no
    /// split.* keys) must take the single-file path — no sibling probing, no
    /// new failure mode.
    #[test]
    fn split_shaped_name_without_split_keys_stays_single_file() {
        let dir = split_dir("solo");
        let a = q8_tensor(4, 64, 0x5010);
        let path = dir.join("solo-00001-of-00002.gguf");
        write_gguf(&path, &[], &[("a.weight", &a)]);

        let gguf = open(&path, &Device::Cpu).unwrap();
        assert_eq!(gguf.content.tensor_infos.len(), 1);
        let loaded = Weights::from_gguf(gguf).qtensor("a").unwrap();
        assert_eq!(dequant(&loaded), dequant(&a));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn split_gguf_missing_shard_is_an_error() {
        let dir = split_dir("missing");
        let a = q8_tensor(4, 64, 0x0111);
        write_gguf(&dir.join("m-00001-of-00003.gguf"), &split_kvs(0, 3, 1), &[]);
        // Shard 2 is never written.
        write_gguf(
            &dir.join("m-00003-of-00003.gguf"),
            &split_kvs(2, 3, 1),
            &[("a.weight", &a)],
        );

        let err = open_err(dir.join("m-00001-of-00003.gguf"));
        assert!(err.contains("shard 2 of 3"), "unexpected error: {err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Shard self-description must agree with the shard's position in the set:
    /// a wrong split.no (a renamed or shuffled file) and a wrong split.count (a
    /// shard from a different split of the same base name) are both refused.
    #[test]
    fn split_gguf_inconsistent_split_keys_are_an_error() {
        let dir = split_dir("badno");
        write_gguf(&dir.join("m-00001-of-00002.gguf"), &split_kvs(0, 2, 0), &[]);
        // split.no says 0 where the file name says shard 2 (split.no 1).
        write_gguf(&dir.join("m-00002-of-00002.gguf"), &split_kvs(0, 2, 0), &[]);
        let err = open_err(dir.join("m-00001-of-00002.gguf"));
        assert!(err.contains(SPLIT_NO_KEY), "unexpected error: {err}");

        // split.count disagrees with the set the file names describe.
        write_gguf(&dir.join("m-00002-of-00002.gguf"), &split_kvs(1, 4, 0), &[]);
        let err = open_err(dir.join("m-00001-of-00002.gguf"));
        assert!(err.contains(SPLIT_COUNT_KEY), "unexpected error: {err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn split_gguf_duplicate_tensor_name_is_an_error() {
        let dir = split_dir("dup");
        let a = q8_tensor(4, 64, 0xD0);
        write_gguf(&dir.join("m-00001-of-00003.gguf"), &split_kvs(0, 3, 2), &[]);
        write_gguf(
            &dir.join("m-00002-of-00003.gguf"),
            &split_kvs(1, 3, 2),
            &[("a.weight", &a)],
        );
        write_gguf(
            &dir.join("m-00003-of-00003.gguf"),
            &split_kvs(2, 3, 2),
            &[("a.weight", &a)],
        );

        let err = open_err(dir.join("m-00001-of-00003.gguf"));
        assert!(
            err.contains("duplicate tensor a.weight"),
            "unexpected error: {err}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn split_gguf_tensor_count_mismatch_is_an_error() {
        let dir = split_dir("count");
        let a = q8_tensor(4, 64, 0xC0);
        // The set claims 5 tensors but carries 1.
        write_gguf(&dir.join("m-00001-of-00002.gguf"), &split_kvs(0, 2, 5), &[]);
        write_gguf(
            &dir.join("m-00002-of-00002.gguf"),
            &split_kvs(1, 2, 5),
            &[("a.weight", &a)],
        );
        let err = open_err(dir.join("m-00001-of-00002.gguf"));
        assert!(
            err.contains(SPLIT_TENSORS_COUNT_KEY),
            "unexpected error: {err}"
        );

        // Shards that disagree with each other about the total are refused too.
        write_gguf(
            &dir.join("m-00002-of-00002.gguf"),
            &split_kvs(1, 2, 1),
            &[("a.weight", &a)],
        );
        let err = open_err(dir.join("m-00001-of-00002.gguf"));
        assert!(
            err.contains("shard 1 of the set"),
            "unexpected error: {err}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A corrupt tensor offset large enough that rebasing it to its shard-
    /// absolute position would wrap u64 must fail with an error naming the
    /// shard and tensor, not wrap into a bogus (and possibly in-bounds) read
    /// position.
    #[test]
    fn split_gguf_overflowing_tensor_offset_is_an_error() {
        let dir = split_dir("overflow");
        let a = q8_tensor(4, 64, 0x0F);
        write_gguf(&dir.join("m-00001-of-00002.gguf"), &split_kvs(0, 2, 1), &[]);
        let shard2 = dir.join("m-00002-of-00002.gguf");
        write_gguf(&shard2, &split_kvs(1, 2, 1), &[("a.weight", &a)]);
        // Patch the written tensor's offset field to u64::MAX. GGUF v2 tensor
        // info layout: name bytes, then u32 n_dims, n_dims × u64 dims (2 here),
        // u32 dtype, u64 offset.
        let mut bytes = std::fs::read(&shard2).unwrap();
        let name = b"a.weight";
        let at = bytes
            .windows(name.len())
            .position(|w| w == name)
            .expect("tensor name appears in the shard's info table");
        let off = at + name.len() + 4 + 2 * 8 + 4;
        bytes[off..off + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&shard2, &bytes).unwrap();

        let err = open_err(dir.join("m-00001-of-00002.gguf"));
        assert!(
            err.contains("overflows u64") && err.contains("a.weight"),
            "unexpected error: {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The whole-file accessors (`mmap_source`, the `path` handle) must refuse
    /// a split GGUF loudly: shard 0 alone is not the checkpoint, and a caller
    /// re-reading it by path (the drafter loaders' pattern) would silently get
    /// wrong bytes. Single-file behavior stays identical.
    #[test]
    fn split_gguf_whole_file_accessors_panic() {
        let dir = split_dir("wholefile");
        let a = q8_tensor(4, 64, 0x77);
        write_gguf(&dir.join("m-00001-of-00002.gguf"), &split_kvs(0, 2, 1), &[]);
        write_gguf(
            &dir.join("m-00002-of-00002.gguf"),
            &split_kvs(1, 2, 1),
            &[("a.weight", &a)],
        );
        let gguf = open(dir.join("m-00001-of-00002.gguf"), &Device::Cpu).unwrap();

        let panic_msg = |r: std::thread::Result<()>| -> String {
            let err = r.expect_err("whole-file access must panic on a split GGUF");
            err.downcast_ref::<String>().cloned().unwrap_or_default()
        };
        let msg = panic_msg(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || {
                let _ = gguf.mmap_source();
            },
        )));
        assert!(msg.contains("mmap_sources"), "unexpected panic: {msg}");
        let msg = panic_msg(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || {
                let _ = gguf.path.to_path_buf();
            },
        )));
        assert!(msg.contains("only shard 0"), "unexpected panic: {msg}");

        // A single-file open still hands out its path and (one) mapping.
        let solo = dir.join("solo.gguf");
        write_gguf(&solo, &[], &[("a.weight", &a)]);
        let gguf = open(&solo, &Device::Cpu).unwrap();
        assert_eq!(&*gguf.path, solo.as_path());
        assert!(gguf.mmap_source().is_none(), "no mapping on Cpu");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A declared split.tensors.count is untrusted input until the shard
    /// tables are read: two tiny shards claiming u64::MAX must produce the
    /// count-mismatch error, not size an allocation.
    #[test]
    fn split_gguf_huge_declared_tensor_count_is_an_error() {
        let dir = split_dir("huge");
        let a = q8_tensor(4, 64, 0x81);
        let kvs = |no: u16| {
            vec![
                (SPLIT_NO_KEY, Value::U16(no)),
                (SPLIT_COUNT_KEY, Value::U16(2)),
                (SPLIT_TENSORS_COUNT_KEY, Value::U64(u64::MAX)),
            ]
        };
        write_gguf(&dir.join("m-00001-of-00002.gguf"), &kvs(0), &[]);
        write_gguf(
            &dir.join("m-00002-of-00002.gguf"),
            &kvs(1),
            &[("a.weight", &a)],
        );
        let err = open_err(dir.join("m-00001-of-00002.gguf"));
        assert!(
            err.contains(SPLIT_TENSORS_COUNT_KEY),
            "unexpected error: {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Shard numbers in file names are 1-indexed members of the set the name
    /// describes: a handed `-00000-` or `-99999-` file must be refused, not
    /// silently swapped for the canonical siblings.
    #[test]
    fn split_gguf_out_of_range_name_number_is_an_error() {
        let dir = split_dir("range");
        // The canonical set exists and is valid on its own.
        write_gguf(&dir.join("m-00001-of-00002.gguf"), &split_kvs(0, 2, 0), &[]);
        write_gguf(&dir.join("m-00002-of-00002.gguf"), &split_kvs(1, 2, 0), &[]);
        for stray in ["m-00000-of-00002.gguf", "m-99999-of-00002.gguf"] {
            write_gguf(&dir.join(stray), &split_kvs(0, 2, 0), &[]);
            let err = open_err(dir.join(stray));
            assert!(err.contains("outside 1..=2"), "unexpected error: {err}");
        }
        // The canonical set itself still opens.
        assert!(open(dir.join("m-00001-of-00002.gguf"), &Device::Cpu).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Partial split metadata is a hard error, never a silent single-file
    /// open: a shard key without split.count, or a split.count = 1 claiming a
    /// nonzero shard number, describes no file this loader should read whole.
    /// A genuine one-shard export (split.count 1, split.no 0) stays a
    /// complete single-file GGUF.
    #[test]
    fn partial_split_metadata_is_an_error() {
        let dir = split_dir("partial");
        let a = q8_tensor(4, 64, 0xAB);

        let path = dir.join("no_count.gguf");
        write_gguf(&path, &[(SPLIT_NO_KEY, Value::U16(1))], &[("a.weight", &a)]);
        let err = open_err(path);
        assert!(err.contains(SPLIT_COUNT_KEY), "unexpected error: {err}");

        let path = dir.join("tensors_only.gguf");
        write_gguf(
            &path,
            &[(SPLIT_TENSORS_COUNT_KEY, Value::I32(1))],
            &[("a.weight", &a)],
        );
        let err = open_err(path);
        assert!(err.contains(SPLIT_COUNT_KEY), "unexpected error: {err}");

        let path = dir.join("count_one_no_one.gguf");
        write_gguf(&path, &split_kvs(1, 1, 1), &[("a.weight", &a)]);
        let err = open_err(path);
        assert!(err.contains("one-shard export"), "unexpected error: {err}");

        let path = dir.join("count_zero.gguf");
        write_gguf(&path, &split_kvs(0, 0, 1), &[("a.weight", &a)]);
        let err = open_err(path);
        assert!(
            err.contains("describes no shard set"),
            "unexpected error: {err}"
        );

        let path = dir.join("one_shard_export.gguf");
        write_gguf(&path, &split_kvs(0, 1, 1), &[("a.weight", &a)]);
        let gguf = open(&path, &Device::Cpu).unwrap();
        assert_eq!(gguf.content.tensor_infos.len(), 1);
        assert_eq!(
            dequant(&Weights::from_gguf(gguf).qtensor("a").unwrap()),
            dequant(&a)
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Non-first shards of real split files carry only the split.* trio, so
    /// keys absent from later shards must stay legal (every other split test
    /// and the real-shard smoke are that shape) — but a key present in TWO
    /// shards with different values means the files are not shards of one
    /// export and must be refused.
    #[test]
    fn split_gguf_conflicting_shard_metadata_is_an_error() {
        let dir = split_dir("meta");
        let a = q8_tensor(4, 64, 0x99);
        let mut m1 = split_kvs(0, 2, 1);
        m1.push(("general.name", Value::String("A".into())));
        write_gguf(&dir.join("m-00001-of-00002.gguf"), &m1, &[]);
        let mut m2 = split_kvs(1, 2, 1);
        m2.push(("general.name", Value::String("B".into())));
        write_gguf(&dir.join("m-00002-of-00002.gguf"), &m2, &[("a.weight", &a)]);
        let err = open_err(dir.join("m-00001-of-00002.gguf"));
        assert!(
            err.contains("general.name") && err.contains("not shards of one split"),
            "unexpected error: {err}"
        );

        // The same value in both shards is consistent, not a conflict.
        m2.pop();
        m2.push(("general.name", Value::String("A".into())));
        write_gguf(&dir.join("m-00002-of-00002.gguf"), &m2, &[("a.weight", &a)]);
        assert!(open(dir.join("m-00001-of-00002.gguf"), &Device::Cpu).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// On the mmap alias load, a tensor from a non-first shard must alias its
    /// OWN shard's mapping: the aliased f16 plane reads back bitwise identical
    /// to the source weight even though the unified tensor table (offsets
    /// rebased, tensor_data_offset 0) no longer matches any single file's
    /// layout.
    #[test]
    fn split_gguf_mmap_alias_reads_the_right_shard() {
        let device = metal_device().unwrap();
        let dir = split_dir("mmap");
        let (out_dim, in_dim) = (8usize, 64usize);
        let plane_f32 = Tensor::from_vec(
            fill(out_dim * in_dim, 0xF00D),
            (out_dim, in_dim),
            &Device::Cpu,
        )
        .unwrap();
        let plane = QTensor::quantize(&plane_f32, GgmlDType::F16).unwrap();
        let w3 = q8_tensor(4, 64, 0xF33D);
        write_gguf(&dir.join("m-00001-of-00003.gguf"), &split_kvs(0, 3, 2), &[]);
        write_gguf(
            &dir.join("m-00002-of-00003.gguf"),
            &split_kvs(1, 3, 2),
            &[("plane.weight", &plane)],
        );
        write_gguf(
            &dir.join("m-00003-of-00003.gguf"),
            &split_kvs(2, 3, 2),
            &[("w.weight", &w3)],
        );

        let gguf = open(dir.join("m-00001-of-00003.gguf"), &device).unwrap();
        let w = Weights::from_gguf(gguf.clone());
        let aliased = w.dense_f16("plane").unwrap();
        for src in gguf.mmap_sources() {
            src.register_views();
        }
        assert_eq!(aliased.dims(), &[out_dim, in_dim]);
        let got: Vec<half::f16> = aliased
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let want: Vec<half::f16> = plane
            .dequantize_f16(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, (g, e)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "aliased plane differs from its shard's bytes at element {i}"
            );
        }
        // A quantized read from the third shard through the same logical file.
        let loaded = w.qtensor("w").unwrap();
        assert_eq!(
            loaded
                .dequantize(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            dequant(&w3)
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Locates a real gguf-split first shard to smoke against, when one is
    /// reachable: `XWEN_SPLIT_SMOKE_SHARD` names a `-00001-of-*.gguf` file
    /// directly; otherwise the HF cache is scanned for the Unsloth
    /// Qwen3.8-Flash-Next split. `None` skips the smoke.
    fn real_first_shard() -> Option<PathBuf> {
        if let Some(p) = std::env::var_os("XWEN_SPLIT_SMOKE_SHARD") {
            let p = PathBuf::from(p);
            return p.exists().then_some(p);
        }
        let hub = std::env::var_os("HF_HUB_CACHE")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
            .or_else(|| {
                std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub"))
            })?;
        fn find(dir: &Path) -> Option<PathBuf> {
            for entry in std::fs::read_dir(dir).ok()? {
                let p = entry.ok()?.path();
                if p.is_dir() {
                    if let Some(hit) = find(&p) {
                        return Some(hit);
                    }
                } else if p.extension().is_some_and(|e| e == "gguf")
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.contains("-00001-of-"))
                {
                    return Some(p);
                }
            }
            None
        }
        find(&hub.join("models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots"))
    }

    /// Smoke against a real metadata-only first shard: its split keys parse at
    /// the widths gguf-split writes, and opening it either assembles the full
    /// set (siblings on disk) or names the first missing sibling. Passes
    /// vacuously when no real shard is reachable (`real_first_shard`).
    #[test]
    fn real_split_first_shard_parses_and_demands_its_siblings() {
        let Some(shard) = real_first_shard() else {
            return;
        };
        let mut file = File::open(&shard).unwrap();
        let ParsedGguf { content, raw } = read_gguf_header(&mut file).unwrap();
        assert!(raw.is_empty(), "the first shard is metadata-only");
        assert_eq!(metadata_uint(&content, SPLIT_NO_KEY).unwrap(), Some(0));
        let count = metadata_uint(&content, SPLIT_COUNT_KEY)
            .unwrap()
            .expect("a real first shard carries split.count");
        assert!(count >= 2, "split.count {count}");
        let total = metadata_uint(&content, SPLIT_TENSORS_COUNT_KEY)
            .unwrap()
            .expect("a real first shard carries split.tensors.count");
        assert!(total > 0, "split.tensors.count {total}");
        assert!(
            content.tensor_infos.is_empty(),
            "the Unsloth layout's first shard is metadata-only"
        );

        let (base, _, name_count) = split_name_parts(&shard).expect("split-shaped name");
        assert_eq!(name_count as u64, count);
        let dir = shard.parent().unwrap();
        let siblings_present = (2..=name_count).all(|i| {
            dir.join(format!("{base}-{i:05}-of-{name_count:05}.gguf"))
                .exists()
        });

        if siblings_present {
            let gguf = open(&shard, &Device::Cpu).unwrap();
            // Both halves of the table: the Unsloth Q4_K_XL mix keeps
            // `per_layer_token_embd.weight` at IQ4_NL, which candle's `Content`
            // cannot hold, so `content.tensor_infos` alone is one short.
            assert_eq!(gguf.tensor_count() as u64, total);
        } else {
            let err = open_err(shard);
            assert!(err.contains("shard 2 of"), "unexpected error: {err}");
        }
    }

    /// The alias-load mapping for `path`: reuse the GgufFile's own (default
    /// open), or map explicitly when the test env forced `XWEN_LOAD_CLASSIC`.
    fn mmap_source_for(gguf: &GgufFile, path: &Path, device: &Device) -> Arc<MmapSource> {
        match gguf.mmap_source() {
            Some(s) => s.clone(),
            None => MmapSource::open(path, device).unwrap(),
        }
    }

    /// An mmap-aliased f16 plane must be BITWISE identical to uploading the
    /// same bytes, both at rest and through the vendored matmul_f16 kernels
    /// (whose weight extraction must honor the view's nonzero start_offset).
    /// The plane sits at a non-page-aligned (but 32-aligned, as in a GGUF)
    /// offset, so the view is page-floored and base_off is nonzero.
    #[test]
    fn mmap_alias_f16_matches_upload() {
        let device = metal_device().unwrap();
        let (out_dim, in_dim) = (24usize, 64usize);
        let elems = out_dim * in_dim;
        let header = 96usize; // 32-aligned like GGUF tensor data, not page-aligned

        let vals: Vec<half::f16> = fill(elems, 0xF16)
            .iter()
            .map(|v| half::f16::from_f32(*v))
            .collect();
        let path = std::env::temp_dir().join(format!("xwen_mmap_f16_{}.bin", std::process::id()));
        {
            let mut bytes = vec![0u8; header];
            for v in &vals {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(&path, &bytes).unwrap();
        }

        let src = MmapSource::open(&path, &device).unwrap();
        // View plumbing: page-floored start, sub-page base_off, page-multiple length.
        let (buf, base_off) = src.view(header, elems * 2).unwrap();
        assert_eq!(base_off, header, "view must page-floor to offset 0 here");
        assert_eq!(
            buf.length() % vm_page_size,
            0,
            "no-copy view length must be page-multiple"
        );

        let aliased = f16_alias_tensor(&src, &device, header, out_dim, in_dim).unwrap();
        // Exercise the full residency lifecycle: batch-register the views (as
        // XwenModel::load does) so the end-of-test drop runs the unregister
        // path against a set that actually holds them.
        src.register_views();
        assert!(aliased.is_contiguous());
        assert_eq!(aliased.dims(), &[out_dim, in_dim]);
        let uploaded = Tensor::from_vec(vals.clone(), (out_dim, in_dim), &device).unwrap();

        let a: Vec<half::f16> = aliased.flatten_all().unwrap().to_vec1().unwrap();
        let u: Vec<half::f16> = uploaded.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (x, y)) in a.iter().zip(&u).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "aliased f16 bytes differ at element {i}"
            );
        }

        // Through the vendored kernels, both the decode gemv (t=1) and the
        // prefill mm (t=16 > F16_MM_MIN_SEQ) branches: identical kernels over
        // identical bytes, only the weight buffer offset differs — bitwise.
        for t in [1usize, 16] {
            let x =
                Tensor::from_vec(fill(t * in_dim, 0xAB + t as u64), (t, in_dim), &device).unwrap();
            let got: Vec<f32> = crate::ops::matmul_f16(&aliased, &x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let want: Vec<f32> = crate::ops::matmul_f16(&uploaded, &x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "matmul_f16 t={t} differs at element {i}: aliased {g} vs uploaded {w}"
                );
            }
        }

        let _ = std::fs::remove_file(&path);
    }

    /// The mmap-aliased expert stack must be BITWISE identical to the classic
    /// upload through BOTH fused kernel families (mv_id decode gather, mm_id
    /// two-pass prefill): same kernels, same bytes, only the weight buffer
    /// binding offset (`base_off`) differs. The synthetic GGUF's small header
    /// puts the tensor at a non-page-aligned offset, so base_off is exercised
    /// nonzero.
    #[test]
    fn expert_stack_mmap_matches_classic() {
        let device = metal_device().unwrap();
        let (n_expert, n_out, k) = (4usize, 8usize, 256usize);
        let dt = GgmlDType::Q4K;

        let w = Tensor::from_vec(
            fill(n_expert * n_out * k, 0xA11A5),
            (n_expert, n_out, k),
            &Device::Cpu,
        )
        .unwrap();
        let qt_cpu = QTensor::quantize(&w, dt).unwrap();
        let path =
            std::env::temp_dir().join(format!("xwen_mmap_stack_{}.gguf", std::process::id()));
        {
            let mut f = File::create(&path).unwrap();
            gguf_file::write(&mut f, &[], &[("ffn_up_exps.weight", &qt_cpu)]).unwrap();
        }

        let gguf = open(&path, &device).unwrap();
        let weights = Weights::from_gguf(gguf.clone());
        let classic = weights.expert_stack_classic("ffn_up_exps").unwrap();
        let src = mmap_source_for(&gguf, &path, &device);
        let aliased = weights.expert_stack_mmap("ffn_up_exps", &src).unwrap();

        assert!(
            aliased.qtensor.is_none(),
            "mmap stack must not build a QTensor"
        );
        assert!(
            aliased.mmap.is_some(),
            "mmap stack must keep the mapping alive"
        );
        assert_ne!(
            aliased.base_off, 0,
            "the synthetic GGUF's tensor offset must exercise a nonzero base_off"
        );
        assert_eq!(
            aliased.base_off % 32,
            0,
            "GGUF data alignment makes base_off 32-aligned"
        );

        let (t, top_k) = (4usize, 2usize);
        let x = Tensor::from_vec(fill(t * k, 0xBEEF), (t, 1, k), &device).unwrap();
        let ids = Tensor::from_vec(vec![0u32, 3, 1, 2, 2, 0, 3, 1], (t, top_k), &device).unwrap();

        let read = |t: Tensor| -> Vec<f32> { t.flatten_all().unwrap().to_vec1().unwrap() };
        let assert_bitwise = |got: &[f32], want: &[f32], path_name: &str| {
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{path_name} differs at element {i}: aliased {g} vs classic {w}"
                );
            }
        };

        // Decode gather (mv_id; vendored or classic per env — same for both stacks).
        let got = read(crate::ops::mul_mv_id(&aliased, &x, &ids).unwrap());
        let want = read(crate::ops::mul_mv_id(&classic, &x, &ids).unwrap());
        assert_bitwise(&got, &want, "mv_id");

        // Prefill two-pass matmul (mm_id, active variant — same for both stacks).
        let got = read(crate::ops::mul_mm_id(&aliased, &x, &ids).unwrap());
        let want = read(crate::ops::mul_mm_id(&classic, &x, &ids).unwrap());
        assert_bitwise(&got, &want, "mm_id");

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // The xwen-owned tensor table (docs/qwen4exp-port.md D8)
    // -----------------------------------------------------------------------

    /// `ggml_type_geometry` and `candle_dtype` are hand-transcribed from ggml's
    /// `type_traits` and candle's `pub(crate)` `GgmlDType::from_u32`, neither of
    /// which xwen can call. Every id candle DOES name must therefore agree with
    /// candle on both block size and type size, or a tensor would be sized with
    /// one table and read with the other — a silent, byte-shifted load rather
    /// than an error. This is the test that catches a candle bump changing one.
    #[test]
    fn ggml_type_ids_agree_with_candle() {
        let mut named = 0;
        for id in 0..64u32 {
            let Some(dtype) = candle_dtype(id) else {
                continue;
            };
            named += 1;
            let (name, block, size) = ggml_type_geometry(id)
                .unwrap_or_else(|| panic!("type id {id} ({dtype:?}) is missing from the table"));
            assert_eq!(block, dtype.block_size(), "{name} (id {id}) block size");
            assert_eq!(size, dtype.type_size(), "{name} (id {id}) type size");
        }
        assert_eq!(named, 15, "every GgmlDType variant is reachable by id");

        // The IQ geometry the raw half exists for, from ggml-common.h's
        // static_asserts at QK_K = 256 and QK4_NL = 32.
        for (id, name, block, size) in [
            (16u32, "IQ2_XXS", 256usize, 66usize),
            (17, "IQ2_XS", 256, 74),
            (18, "IQ3_XXS", 256, 98),
            (19, "IQ1_S", 256, 50),
            (20, "IQ4_NL", 32, 18),
            (21, "IQ3_S", 256, 110),
            (22, "IQ2_S", 256, 82),
            (23, "IQ4_XS", 256, 136),
            (29, "IQ1_M", 256, 56),
        ] {
            assert_eq!(ggml_type_geometry(id), Some((name, block, size)), "id {id}");
            assert!(candle_dtype(id).is_none(), "{name} is not a candle dtype");
        }

        // A slot ggml removed, and one past GGML_TYPE_COUNT: both refused, so a
        // tensor never disappears from the table without an error.
        assert_eq!(ggml_type_geometry(5), None, "the removed Q4_3 slot");
        assert_eq!(ggml_type_geometry(43), None, "past GGML_TYPE_COUNT");
    }

    /// Appends a GGUF length-prefixed string (v2/v3 widths).
    fn gguf_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    /// One tensor-table entry, GGUF order: name, n_dims, dims (fastest-varying
    /// first), ggml type id, offset into the tensor-data section.
    fn gguf_tensor_entry(out: &mut Vec<u8>, name: &str, dims: &[u64], type_id: u32, offset: u64) {
        gguf_str(out, name);
        out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&type_id.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
    }

    /// One Q8_0 block: an f16 scale of 1.0 followed by 32 signed quants, so a
    /// dequantized element is exactly its quant value.
    fn q8_0_block(quants: impl Fn(usize) -> i8) -> Vec<u8> {
        let mut b = half::f16::from_f32(1.0).to_le_bytes().to_vec();
        b.extend((0..32).map(|i| quants(i) as u8));
        b
    }

    /// One IQ4_NL block: an f16 scale, then 16 bytes whose low nibble is
    /// element `j` and whose high nibble is element `j + 16` (ggml's split-half
    /// interleave — see `qwen4exp::iq4nl`).
    fn iq4_nl_block(scale: f32, nibble: impl Fn(usize) -> u8) -> Vec<u8> {
        let mut b = half::f16::from_f32(scale).to_le_bytes().to_vec();
        b.extend((0..16).map(|j| (nibble(j) & 0xf) | ((nibble(j + 16) & 0xf) << 4)));
        b
    }

    /// A hand-built GGUF holding one Q8_0 tensor and one IQ4_NL tensor, which
    /// is the shape of every Unsloth Q3/Q4 Flash-Next mix in miniature. candle's
    /// `Content::read` refuses such a file outright ("unknown dtype for tensor
    /// 20"), so this is written by hand rather than through candle's writer,
    /// which cannot emit an IQ type either.
    ///
    /// Returns the file's path plus the byte offset of the tensor-data section,
    /// so the test can pin absolute offsets and not merely self-consistency.
    fn write_mixed_dtype_gguf(path: &Path) -> u64 {
        let mut head = Vec::new();
        head.extend_from_slice(b"GGUF");
        head.extend_from_slice(&3u32.to_le_bytes()); // version
        head.extend_from_slice(&2u64.to_le_bytes()); // tensor count
        head.extend_from_slice(&1u64.to_le_bytes()); // metadata kv count
        gguf_str(&mut head, "general.architecture");
        head.extend_from_slice(&8u32.to_le_bytes()); // value type: string
        gguf_str(&mut head, "qwen4exp");
        // GGUF dims are fastest-varying first, so {64, 2} is candle's [2, 64].
        gguf_tensor_entry(&mut head, "known.weight", &[64, 2], 8, 0);
        // 4 rows of 160 IQ4_NL elements — the PLE table's row width, 5 blocks
        // (90 bytes) per row. Placed after the Q8_0 tensor's 136 bytes, padded
        // to the 32-byte tensor alignment.
        gguf_tensor_entry(&mut head, "per_layer_token_embd.weight", &[160, 4], 20, 160);
        let data_offset = (head.len() as u64).div_ceil(32) * 32;
        head.resize(data_offset as usize, 0);

        let mut data = Vec::new();
        // 2 rows x 64 elements = 4 Q8_0 blocks; element i of the flat tensor
        // dequantizes to (i % 127) - 63.
        for blk in 0..4 {
            data.extend(q8_0_block(|i| ((blk * 32 + i) % 127) as i8 - 63));
        }
        assert_eq!(data.len(), 136);
        data.resize(160, 0); // pad to the declared IQ4_NL offset
        // 4 rows x 5 blocks; block b of row r has scale (r + 1) and nibble
        // pattern (j + b) % 16.
        for r in 0..4u32 {
            for b in 0..5usize {
                data.extend(iq4_nl_block((r + 1) as f32, |j| ((j + b) % 16) as u8));
            }
        }
        assert_eq!(data.len(), 160 + 4 * 5 * 18);

        head.extend_from_slice(&data);
        std::fs::write(path, &head).unwrap();
        data_offset
    }

    /// The loader parses a tensor table candle cannot: the Q8_0 tensor lands in
    /// `content` and loads through candle exactly as before, and the IQ4_NL one
    /// lands in the raw half with the right dtype, shape, byte length and
    /// absolute offset — and is reachable as a PLE table.
    #[test]
    fn mixed_dtype_gguf_splits_the_tensor_table() {
        let device = metal_device().unwrap();
        let path =
            std::env::temp_dir().join(format!("xwen_mixed_dtype_{}.gguf", std::process::id()));
        let data_offset = write_mixed_dtype_gguf(&path);

        // The premise: candle's own parser cannot read this file at all.
        let err = Content::read(&mut File::open(&path).unwrap())
            .expect_err("candle must refuse the IQ4_NL tensor")
            .to_string();
        assert!(err.contains("unknown dtype"), "unexpected error: {err}");

        let gguf = open(&path, &device).unwrap();
        assert_eq!(gguf.content.tensor_data_offset, data_offset);
        assert_eq!(gguf.tensor_count(), 2);
        assert_eq!(
            gguf.content.tensor_infos.len(),
            1,
            "one candle-known tensor"
        );
        assert_eq!(
            gguf.raw_tensor_names(),
            vec![("per_layer_token_embd.weight", RawDtype::Iq4Nl)]
        );
        assert_eq!(gguf.raw_tensor_bytes(), 360);
        assert!(gguf.has_tensor("per_layer_token_embd.weight"));
        assert_eq!(
            gguf.stored_dtype_of("known.weight").unwrap(),
            StoredDtype::Ggml(GgmlDType::Q8_0)
        );
        assert_eq!(
            gguf.stored_dtype_of("per_layer_token_embd.weight").unwrap(),
            StoredDtype::Raw(RawDtype::Iq4Nl)
        );

        // The candle-known tensor still loads through candle, values and all.
        let w = Weights::from_gguf(gguf.clone());
        assert!(w.has("known") && w.has("per_layer_token_embd"));
        let qt = w.qtensor("known").unwrap();
        assert_eq!(qt.shape().dims(), &[2, 64]);
        let got = qt
            .dequantize(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let want: Vec<f32> = (0..128).map(|i| ((i % 127) - 63) as f32).collect();
        assert_eq!(
            got, want,
            "Q8_0 tensor loads unchanged alongside an IQ type"
        );

        // ...while `stored_dtype` still answers in candle's vocabulary, and says
        // so plainly rather than "not found" for the tensor it cannot name.
        assert_eq!(w.stored_dtype("known").unwrap(), GgmlDType::Q8_0);
        let err = w
            .stored_dtype("per_layer_token_embd")
            .unwrap_err()
            .to_string();
        assert!(err.contains("IQ4_NL"), "unexpected error: {err}");

        if gguf.mmap_source().is_some() {
            let raw = gguf.raw_tensor("per_layer_token_embd.weight").unwrap();
            assert_eq!(raw.dtype, StoredDtype::Raw(RawDtype::Iq4Nl));
            assert_eq!(raw.shape, vec![4, 160]);
            assert_eq!(raw.len, 360);
            assert_eq!(raw.offset as u64, data_offset + 160);

            let table =
                crate::qwen4exp::ple::PleTable::open(&gguf, "per_layer_token_embd.weight").unwrap();
            assert_eq!(table.rows(), 4);
            assert_eq!(table.row_dim(), 160);
            let mut row = vec![0f32; 160];
            table.row(3, &mut row).unwrap();
            // Row 3's blocks carry scale 4.0 and nibble (j + b) % 16, so element
            // 0 of block 0 is 4 * kvalues[0].
            assert_eq!(
                row[0],
                4.0 * f32::from(crate::qwen4exp::iq4nl::KVALUES_IQ4NL[0])
            );
            assert!(row.iter().all(|v| v.is_finite()));
            assert!(table.row(4, &mut row).is_err(), "row 4 is past the table");
        }

        let listing = describe_file(&gguf);
        assert!(
            listing.contains("per_layer_token_embd.weight") && listing.contains("IQ4_NL"),
            "the raw half must show up in an inspect listing:\n{listing}"
        );

        let _ = std::fs::remove_file(&path);
    }

    /// A tensor whose ggml type id ggml itself does not define fails the open
    /// rather than vanishing from the table — the behaviour candle's parser had
    /// for every id it did not know, kept for the ids nobody knows.
    #[test]
    fn undefined_ggml_type_id_is_an_error() {
        let path = std::env::temp_dir().join(format!("xwen_bad_type_{}.gguf", std::process::id()));
        let mut head = Vec::new();
        head.extend_from_slice(b"GGUF");
        head.extend_from_slice(&3u32.to_le_bytes());
        head.extend_from_slice(&1u64.to_le_bytes());
        head.extend_from_slice(&0u64.to_le_bytes());
        gguf_tensor_entry(&mut head, "mystery.weight", &[32, 1], 99, 0);
        head.resize(head.len().div_ceil(32) * 32 + 64, 0);
        std::fs::write(&path, &head).unwrap();

        let err = open_err(path.clone());
        assert!(err.contains("ggml type id 99"), "unexpected error: {err}");
        let _ = std::fs::remove_file(&path);
    }

    /// The real Unsloth Qwen3.8-Flash-Next split: 1224 tensors across its
    /// shards, exactly one of which candle cannot name — the 28.8 GB IQ4_NL PLE
    /// table — and that one reads as a table, at both ends of its 320-million
    /// rows. Skips when the checkpoint is not in the cache.
    ///
    /// The tail differences the two POPULATIONS the split loader rebases
    /// separately (`open_split`'s `known` and `unknown` loops): a candle-known
    /// Q8_0 tensor that physically lives in shard 2 must yield the same stored
    /// bytes through the raw/mmap route (`raw_tensor` → `MmapSource::bytes`,
    /// which resolves `shard_for(name).mmap` and reads at the rebased absolute
    /// offset) as through candle's own route (`Weights::qtensor` →
    /// `Content::tensor`, which seeks `shard_for(name).file` at
    /// `content.tensor_data_offset + info.offset`). Those are the only two ways
    /// a tensor's bytes are ever reached, they share no code below the tensor
    /// table, and both depend on the rebase being right — so if a future change
    /// to `open_split` rebased one half and not the other, or paired a tensor
    /// with the wrong shard, this fails where nothing else would: a wrong
    /// offset inside a 50 GB shard reads plausible quantized garbage, and the
    /// only tensor that is currently read from a real split at all is the
    /// IQ4_NL one, which exercises the `unknown` loop alone.
    ///
    /// `raw_tensor` serves both halves, so it is the raw side for either
    /// population; the candle side has no equivalent for IQ4_NL (that is why
    /// `raw_of` exists), which is why the pairing has to be made on a
    /// candle-known tensor. `output_hc_down.weight` is the pick: Q8_0, in shard
    /// 2, 3.5 MB — small enough that `QTensor::data()`'s device readback is
    /// free, while sitting ~675 MB into the shard, far past anything shard 1
    /// (10 MB, metadata-only) could satisfy.
    #[test]
    fn real_flash_next_iq4_nl_table_opens_and_reads() {
        let Some(shard) = real_first_shard() else {
            eprintln!("skipping: no Qwen3.8-Flash-Next shard in the HF cache");
            return;
        };
        let (base, _, count) = split_name_parts(&shard).expect("split-shaped name");
        let dir = shard.parent().unwrap();
        if !(2..=count).all(|i| {
            dir.join(format!("{base}-{i:05}-of-{count:05}.gguf"))
                .exists()
        }) {
            eprintln!("skipping: {} is missing sibling shards", shard.display());
            return;
        }
        let Ok(device) = metal_device() else {
            eprintln!("skipping: no Metal device, so no file mapping to read rows from");
            return;
        };

        let gguf = open(&shard, &device).unwrap();
        assert_eq!(gguf.tensor_count(), 1224);
        assert_eq!(
            gguf.raw_tensor_names(),
            vec![("per_layer_token_embd.weight", RawDtype::Iq4Nl)],
            "exactly one tensor candle cannot name"
        );
        assert_eq!(gguf.raw_tensor_bytes(), 28_800_138_240);

        let Some(_) = gguf.mmap_sources().first().cloned() else {
            eprintln!("skipping the row reads: XWEN_LOAD_CLASSIC leaves the file unmapped");
            return;
        };
        let raw = gguf.raw_tensor("per_layer_token_embd.weight").unwrap();
        assert_eq!(raw.shape, vec![320_001_536, 160]);
        assert_eq!(raw.len, 28_800_138_240);
        assert_eq!(
            raw.shape.iter().product::<usize>(),
            51_200_245_760,
            "elements"
        );

        let table =
            crate::qwen4exp::ple::PleTable::open(&gguf, "per_layer_token_embd.weight").unwrap();
        assert_eq!(table.rows(), 320_001_536);
        assert_eq!(table.row_dim(), 160);
        let mut row = vec![0f32; 160];
        for r in [0u64, 320_001_535] {
            table.row(r, &mut row).unwrap();
            assert!(
                row.iter().all(|v| v.is_finite()),
                "row {r} dequantized to a non-finite value"
            );
        }
        assert!(
            table.row(320_001_536, &mut row).is_err(),
            "one past the last row is out of bounds"
        );

        // The candle-known half of the same shard, differenced against the raw
        // half (see the doc comment).
        let known = "output_hc_down.weight";
        assert_eq!(
            gguf.stored_dtype_of(known).unwrap(),
            StoredDtype::Ggml(GgmlDType::Q8_0),
            "{known} is the shard-2 pick because it is a small Q8_0 plane"
        );
        let known_raw = gguf.raw_tensor(known).unwrap();
        assert!(
            Arc::ptr_eq(&known_raw.src, &raw.src),
            "{known} and the PLE table both live in shard 2, so they must alias \
             the same mapping"
        );
        let mapped = known_raw.src.bytes(known_raw.offset, 64).unwrap().to_vec();

        let loaded = Weights::from_gguf(gguf.clone())
            .qtensor("output_hc_down")
            .unwrap();
        assert_eq!(loaded.dtype(), GgmlDType::Q8_0);
        assert_eq!(
            loaded.storage_size_in_bytes(),
            known_raw.len,
            "the two routes must agree on how many bytes {known} occupies"
        );
        let via_candle = loaded.data().unwrap();
        assert_eq!(
            &via_candle[..64],
            &mapped[..],
            "the mmap route and candle's file route disagree on {known}'s first \
             block: one of the two rebases is off"
        );
        // Not a slab of padding zeros that would match anywhere in the file: a
        // Q8_0 block is a f16 scale plus 32 signed bytes, and real weights fill
        // both.
        assert!(
            mapped.iter().any(|&b| b != 0),
            "{known}'s first 64 bytes are all zero, so the comparison is vacuous"
        );
    }

    /// The header parser is a REPLACEMENT for candle's `Content::read`, not a
    /// variant of it: `CheckpointId` hashes the bytes up to
    /// `tensor_data_offset`, so a drift in the alignment rule silently
    /// invalidates every persisted cache image on disk, and `config.rs` reads
    /// the metadata map this produces. The only intended difference is the one
    /// the rewrite exists for — a tensor whose ggml type id candle cannot name
    /// goes to the raw table instead of failing the file.
    ///
    /// So this asserts equivalence against candle itself, on the real blessed
    /// files rather than on a fixture the same author wrote both sides of. Any
    /// official checkpoint absent from the hub cache is skipped with a line
    /// saying so; nothing is downloaded. Qwen3.8-Flash-Next is deliberately not
    /// in the list — it is the file candle CANNOT read (IQ4_NL
    /// `per_layer_token_embd`), and the test above covers it instead.
    #[test]
    fn the_header_parser_agrees_with_candle_on_every_blessed_file() {
        use crate::hub::Model;

        let mut checked = 0;
        for model in [Model::Qwen35BA3B, Model::Qwen27B, Model::Qwen3827B] {
            let Some(path) = crate::hub::cached_model(model) else {
                eprintln!("skipping {model}: not in the Hugging Face cache");
                continue;
            };
            checked += 1;

            let mut file = File::open(&path).unwrap();
            let ours = read_gguf_header(&mut file).unwrap();
            assert!(
                ours.raw.is_empty(),
                "{model}: candle can name every tensor in this file, so nothing belongs \
                 in the raw table"
            );
            let mut file = File::open(&path).unwrap();
            let theirs = gguf_file::Content::read(&mut file).unwrap();

            // The offset every persisted cache image is keyed against.
            assert_eq!(
                ours.content.tensor_data_offset, theirs.tensor_data_offset,
                "{model}: tensor_data_offset"
            );
            assert_eq!(
                format!("{:?}", ours.content.magic),
                format!("{:?}", theirs.magic),
                "{model}: magic"
            );

            // Metadata: the same keys, each carrying the same typed value.
            // Compared per key rather than as whole maps because `HashMap`'s
            // Debug order is not stable, and by Debug rather than by value
            // because candle's `Value` has no `PartialEq` — Debug distinguishes
            // both the variant and the payload, which is what is at stake.
            let ours_keys: std::collections::BTreeSet<&String> =
                ours.content.metadata.keys().collect();
            let theirs_keys: std::collections::BTreeSet<&String> = theirs.metadata.keys().collect();
            assert_eq!(ours_keys, theirs_keys, "{model}: metadata keys");
            for key in ours_keys {
                assert_eq!(
                    format!("{:?}", ours.content.metadata[key]),
                    format!("{:?}", theirs.metadata[key]),
                    "{model}: metadata value for {key}"
                );
            }

            // The tensor table: same names, and for each the same dtype, shape
            // and offset. The shape check is what pins `dimensions.reverse()` —
            // a parser that dropped it would produce transposed weights that
            // load and multiply and are simply wrong.
            let ours_names: std::collections::BTreeSet<&String> =
                ours.content.tensor_infos.keys().collect();
            let theirs_names: std::collections::BTreeSet<&String> =
                theirs.tensor_infos.keys().collect();
            assert_eq!(ours_names, theirs_names, "{model}: tensor names");
            for name in ours_names {
                let a = &ours.content.tensor_infos[name];
                let b = &theirs.tensor_infos[name];
                assert_eq!(a.ggml_dtype, b.ggml_dtype, "{model}: dtype of {name}");
                assert_eq!(a.shape.dims(), b.shape.dims(), "{model}: shape of {name}");
                assert_eq!(a.offset, b.offset, "{model}: offset of {name}");
            }
        }
        if checked == 0 {
            eprintln!("no official checkpoint is cached; the parser was not differenced");
        }
    }

    /// Relative L2 distance, `||got - want|| / ||want||`.
    fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
        let (mut d, mut n) = (0.0f64, 0.0f64);
        for (g, w) in got.iter().zip(want) {
            d += ((*g - *w) as f64).powi(2);
            n += (*w as f64).powi(2);
        }
        (d / n.max(1e-30)).sqrt() as f32
    }

    /// A planed `QLinear` built exactly as `qlinear_with_buffer` builds one —
    /// the `QMatMul` and the plane sharing ONE device allocation — so the two
    /// routes below read identical bytes and any difference is the kernel's.
    fn planed_qlinear(
        device: &Device,
        dt: GgmlDType,
        out_dim: usize,
        in_dim: usize,
        seed: u64,
    ) -> QLinear {
        let dense = Tensor::from_vec(
            fill(out_dim * in_dim, seed),
            (out_dim, in_dim),
            &Device::Cpu,
        )
        .unwrap();
        let qcpu = QTensor::quantize(&dense, dt).unwrap();
        let raw = qcpu.data().unwrap();
        let storage = QStorage::from_data(raw, device, dt).unwrap();
        let QStorage::Metal(qms) = &storage else {
            panic!("expected Metal storage")
        };
        let buffer = Arc::new(qms.buffer().clone());
        let qtensor = Arc::new(QTensor::new(storage, (out_dim, in_dim)).unwrap());
        QLinear {
            inner: QMatMul::from_arc(qtensor).unwrap(),
            in_dim,
            out_dim,
            plane: Some(QuantPlane {
                buffer,
                base_off: 0,
                dtype: dt,
                out_dim,
                in_dim,
            }),
            mv_ext_ok: true,
        }
    }

    /// `forward_gemm` routes the vendored dense gemm above `dense_mm_min_seq()`
    /// and is exactly `forward` at or below it. Graded against the `QMatMul`
    /// route over the same bytes at the shapes the two opt-in callers bring:
    /// the shared expert (q8_0, hidden 2560 <-> 640 on Flash-Next, 2048 <-> 512
    /// on the 35B) and the hyper-connection bottleneck (q8_0, 10240 -> 320 and
    /// 320 -> 10240, the latter the ten-NK-step shape). Bound: the dense gemm's
    /// precision class (~4e-4 rel_l2 from the f32 oracle at the 27B FFN shapes,
    /// dense_mm.rs `TOL`), with the two routes' separate rounding summed — the
    /// same 1e-3 the 27B FFN is graded at, never tighter than the kernel's
    /// reduced-precision descriptor allows.
    #[test]
    fn forward_gemm_matches_qmatmul_at_prefill_and_is_forward_below() {
        let device = metal_device().unwrap();
        let t_gemm = crate::ops::dense_mm_min_seq() + 31; // above the floor, with a ragged tile
        let shapes: [(usize, usize); 6] = [
            (640, 2560),
            (2560, 640), // Flash-Next shexp gate/up, down
            (512, 2048),
            (2048, 512), // 35B shexp
            (320, 10240),
            (10240, 320), // hc down, up
        ];
        for (i, &(out_dim, in_dim)) in shapes.iter().enumerate() {
            let lin = planed_qlinear(&device, GgmlDType::Q8_0, out_dim, in_dim, 0x51 + i as u64);
            let x = Tensor::from_vec(
                fill(t_gemm * in_dim, 0x77 + i as u64),
                (t_gemm, in_dim),
                &Device::Cpu,
            )
            .unwrap()
            .to_device(&device)
            .unwrap();
            let got: Vec<f32> = lin
                .forward_gemm(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let want: Vec<f32> = lin
                .forward(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let rel = rel_l2(&got, &want);
            eprintln!("forward_gemm [{out_dim},{in_dim}] t={t_gemm}: rel_l2 vs QMatMul {rel:.3e}");
            assert!(
                rel < 1e-3,
                "[{out_dim},{in_dim}]: rel_l2 {rel:.3e} vs QMatMul"
            );

            // At one token the two are the same call: bitwise.
            let x1 = x.narrow(0, 0, 1).unwrap().contiguous().unwrap();
            let g1: Vec<f32> = lin
                .forward_gemm(&x1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let w1: Vec<f32> = lin
                .forward(&x1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert!(
                g1.iter().zip(&w1).all(|(a, b)| a.to_bits() == b.to_bits()),
                "[{out_dim},{in_dim}] t=1 differs"
            );

            // The gate boundary, once (the shapes share one dispatcher): AT the
            // floor forward_gemm is exactly forward (same QMatMul call, bitwise);
            // one past it the gemm routes and lands in its precision class.
            if i == 0 {
                for (t, must_route) in [
                    (crate::ops::dense_mm_min_seq(), false),
                    (crate::ops::dense_mm_min_seq() + 1, true),
                ] {
                    let xb = Tensor::from_vec(
                        fill(t * in_dim, 0xB0 + t as u64),
                        (t, in_dim),
                        &Device::Cpu,
                    )
                    .unwrap()
                    .to_device(&device)
                    .unwrap();
                    let gb: Vec<f32> = lin
                        .forward_gemm(&xb)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1()
                        .unwrap();
                    let wb: Vec<f32> = lin
                        .forward(&xb)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1()
                        .unwrap();
                    if must_route {
                        let rel = rel_l2(&gb, &wb);
                        assert!(rel < 1e-3, "t={t}: rel_l2 {rel:.3e} vs QMatMul");
                    } else {
                        assert!(
                            gb.iter().zip(&wb).all(|(a, b)| a.to_bits() == b.to_bits()),
                            "t={t} (the floor) must not route the gemm"
                        );
                    }
                }
            }
        }
    }

    /// The hyper-connection projections carry planes ONLY for the prefill gemm:
    /// `without_mv_ext` keeps `forward` off the small-batch (2..=8 token)
    /// window, so a planed-but-gemm-only layer is bitwise the plane-less
    /// `from_qtensor` one there — which is what makes the hc paths
    /// (`XWEN_HC_CLASSIC` included) numerically unchanged by the planes.
    #[test]
    fn without_mv_ext_keeps_small_batch_on_qmatmul() {
        let device = metal_device().unwrap();
        let (out_dim, in_dim) = (320usize, 10240usize); // hc down, the mv_ext-eligible shape
        let gemm_only =
            planed_qlinear(&device, GgmlDType::Q8_0, out_dim, in_dim, 0xC1).without_mv_ext();
        let mut plain = planed_qlinear(&device, GgmlDType::Q8_0, out_dim, in_dim, 0xC1);
        plain.plane = None; // the pre-plane loader's shape, over identical bytes
        let x = Tensor::from_vec(fill(4 * in_dim, 0xC2), (4, in_dim), &Device::Cpu)
            .unwrap()
            .to_device(&device)
            .unwrap();
        let got: Vec<f32> = gemm_only
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let want: Vec<f32> = plain
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            got.iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "a gemm-only plane must leave 2..=8-token forwards bitwise on QMatMul"
        );
    }
}
