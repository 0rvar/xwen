//! PLE (per-layer embeddings): the runtime n-gram table and its injection layer.
//!
//! One layer of the qwen4exp trunk (layer 1 on the shipped checkpoint) reads 16
//! rows of a 320-million-row table per token, projects them to a key and a
//! value, gates the value per hyper-connection stream against the incoming
//! carrier, runs a dilated depthwise conv over the gated result, and hands the
//! sum back as an addend to the 10240-wide carrier.
//!
//! Ground truth is llama.cpp `reference/llama.cpp/src/models/qwen4exp.cpp`
//! (`set_input` for the hash, `build_ple` for the layer); the numbers are pinned
//! by `tests/fixtures/qwen4exp/ple.json` through the frozen oracle in
//! [`super::ref_ple`], which this module is graded against.
//!
//! **P2 is host-hybrid (docs/qwen4exp-port.md D17).** The table is 28.8 GB of
//! IQ4_NL that never becomes a device tensor: the hash, the row gather and the
//! row dequant run on the CPU straight out of the file mapping (D2). Only the
//! two projections and the key norm run on the GPU; the gate, the conv and the
//! silu run on the host in f32 over a downloaded copy of the carrier, which
//! costs one device→host sync per forward. That is a known P3 cost, taken so
//! the first correct version is a short walk from the oracle. `XWEN_PLE_PROFILE`
//! ([`crate::ops::ple_profile`]) splits one forward into its sub-steps and says
//! which half of that hybrid the layer's `stack_profile` figure actually is.
//!
//! The gather's cost is not arithmetic, it is page faults — 16 unrelated
//! 90-byte rows per token over a table far larger than memory — so
//! [`PlePrefetcher`] runs those faults on a background thread ahead of the
//! forward that needs them, off addresses that are known one position early
//! (`XWEN_PLE_NO_PREFETCH` / `XWEN_PLE_NO_RANDOM` turn the two halves of that
//! off for an A/B).
//!
//! The reuse here is deliberate and not an accident of convenience: the hash
//! ([`super::ref_ple::PleHashRef`]) and the grouped RMS norm
//! ([`super::ref_hc::grouped_rms_norm`]) are called, not reimplemented, so the
//! host half of the hybrid IS the oracle rather than a second transcription of
//! it. The references are frozen, so nothing can drift underneath.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use candle_core::quantized::GgmlDType;
use candle_core::{D, DType, Device, Tensor};

use super::iq4nl;
use super::ref_hc::grouped_rms_norm;
use super::ref_ple::{PleHashRef, gate_function_probe};
use crate::config::XwenConfig;
use crate::gguf::{GgufFile, MmapSource, QLinear, RawDtype, StoredDtype, Weights, host_page_size};
use crate::host_log::host_line;

/// The stored dtypes the n-gram table is read at.
///
/// IQ4_NL is the one that matters — every Unsloth mix from Q3 to Q4 holds the
/// table there because its access pattern is random and 4 bits is the floor
/// they will quantize it to. Q8_0 covers the Q5/Q6/Q8 mixes; the float arms
/// cover BF16 and a self-converted file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableDtype {
    Iq4Nl,
    Q8_0,
    F32,
    F16,
    Bf16,
}

impl TableDtype {
    /// Bytes one `row_dim`-wide row occupies. IQ4_NL at the shipped 160-wide
    /// row is 5 blocks = 90 bytes; a row is always a whole number of blocks
    /// because the table's row width is the gather granularity.
    fn row_bytes(self, row_dim: usize) -> Result<usize> {
        let per_block = |elems: usize, bytes: usize| -> Result<usize> {
            ensure!(
                row_dim.is_multiple_of(elems),
                "a {row_dim}-wide table row is not a whole number of {elems}-element blocks"
            );
            Ok(row_dim / elems * bytes)
        };
        match self {
            Self::Iq4Nl => per_block(iq4nl::QK4_NL, iq4nl::BLOCK_BYTES),
            Self::Q8_0 => per_block(iq4nl::QK8_0, iq4nl::BLOCK_BYTES_Q8_0),
            Self::F32 => Ok(row_dim * 4),
            Self::F16 | Self::Bf16 => Ok(row_dim * 2),
        }
    }

    /// The dtype the loader read out of the GGUF tensor table, mapped onto the
    /// arms this reader dequantizes.
    ///
    /// IQ4_NL arrives as [`StoredDtype::Raw`]: candle's `GgmlDType` has no
    /// variant for it, so the xwen-owned tensor-table parse is what names it
    /// (docs/qwen4exp-port.md D8 class 1) and this reader is the only thing
    /// that can read the bytes.
    fn from_stored(dtype: StoredDtype) -> Result<Self> {
        let unsupported = || {
            anyhow::anyhow!(
                "the PLE table is stored as {dtype:?}, which this reader does not dequantize \
                 (supported: IQ4_NL, Q8_0, F32, F16, BF16)"
            )
        };
        Ok(match dtype {
            StoredDtype::Raw(RawDtype::Iq4Nl) => Self::Iq4Nl,
            StoredDtype::Raw(_) => return Err(unsupported()),
            StoredDtype::Ggml(GgmlDType::F32) => Self::F32,
            StoredDtype::Ggml(GgmlDType::F16) => Self::F16,
            StoredDtype::Ggml(GgmlDType::BF16) => Self::Bf16,
            StoredDtype::Ggml(GgmlDType::Q8_0) => Self::Q8_0,
            StoredDtype::Ggml(_) => return Err(unsupported()),
        })
    }
}

/// Where the table's bytes come from. The production arm is the file mapping;
/// the owned arm exists so a table can be built in memory at test geometry and
/// still travel every line of the real dequant dispatch.
enum TableBytes {
    Mapped(Arc<MmapSource>),
    Owned(Vec<u8>),
}

impl TableBytes {
    fn slice(&self, off: usize, len: usize) -> Result<&[u8]> {
        match self {
            Self::Mapped(src) => src.bytes(off, len),
            Self::Owned(v) => {
                ensure!(
                    off.checked_add(len).is_some_and(|end| end <= v.len()),
                    "table byte range [{off}, +{len}) exceeds the {}-byte table",
                    v.len()
                );
                Ok(&v[off..off + len])
            }
        }
    }
}

/// The flat n-gram embedding table, read one row at a time on the CPU.
///
/// `per_layer_token_embd.weight` is `[rows, row_dim]` — 320,001,536 × 160 on the
/// shipped checkpoint, 28.8 GB at IQ4_NL. It is never uploaded and never
/// dequantized whole; it is demand-paged by the kernel and touched 16 rows per
/// token (docs/qwen4exp-port.md D2).
pub struct PleTable {
    bytes: Arc<TableBytes>,
    /// Byte offset of row 0 within `bytes`.
    base: usize,
    dtype: TableDtype,
    row_dim: usize,
    rows: u64,
    row_bytes: usize,
    /// The background page-toucher, built on the first hint and never rebuilt.
    /// `None` inside the `OnceLock` means prefetching is switched off
    /// (`XWEN_PLE_NO_PREFETCH`); an untouched `OnceLock` means no caller has
    /// asked for a prefetch yet, which is every table a test builds and never
    /// hints — those spawn no thread at all.
    prefetch: OnceLock<Option<PlePrefetcher>>,
}

impl PleTable {
    /// The general constructor: a byte source, the offset of row 0 inside it,
    /// and the geometry. `from_source` is what a loader that parsed the tensor
    /// table itself calls — including for IQ4_NL, which candle cannot name.
    pub fn from_source(
        src: Arc<MmapSource>,
        base: usize,
        dtype: TableDtype,
        row_dim: usize,
        rows: u64,
    ) -> Result<Self> {
        let table = Self::build(TableBytes::Mapped(src.clone()), base, dtype, row_dim, rows)?;
        // `MADV_RANDOM` over THIS TENSOR's bytes and nothing else: the gather
        // reads 16 unrelated rows per token out of a table far larger than
        // memory, so readahead around one row is bandwidth spent on bytes
        // nothing will ask for. The mapping's whole-file `WillNeed` stays as it
        // is — the weights around this tensor are read sequentially and want it.
        // Switchable (`XWEN_PLE_NO_RANDOM`) because "the pattern is random" is
        // an argument, and the `gather` figure under `XWEN_PLE_PROFILE` is the
        // measurement.
        if !crate::ops::ple_no_random()
            && let Some(len) = table.row_bytes.checked_mul(rows as usize)
        {
            src.advise_random(base, len);
        }
        Ok(table)
    }

    fn build(
        bytes: TableBytes,
        base: usize,
        dtype: TableDtype,
        row_dim: usize,
        rows: u64,
    ) -> Result<Self> {
        ensure!(row_dim > 0, "the PLE table row width must be positive");
        ensure!(rows > 0, "the PLE table has no rows");
        let row_bytes = dtype.row_bytes(row_dim)?;
        Ok(Self {
            bytes: Arc::new(bytes),
            base,
            dtype,
            row_dim,
            rows,
            row_bytes,
            prefetch: OnceLock::new(),
        })
    }

    /// Locates `name` in an opened GGUF (across shards) and wraps its bytes.
    ///
    /// The tensor is `[rows, row_dim]` in candle's row-major reading of the
    /// GGUF's `{row_dim, rows}` — the same orientation `qlinear` sees, so a
    /// transposed reading would be caught by the row width, not silently
    /// gather 160 rows of 320-million-wide nonsense.
    pub fn open(gguf: &GgufFile, name: &str) -> Result<Self> {
        let raw = gguf
            .raw_tensor(name)
            .with_context(|| format!("locating the PLE table {name}"))?;
        let [rows, row_dim] = raw.shape[..] else {
            bail!("{name} is not a rank-2 table: {:?}", raw.shape);
        };
        let table = Self::from_source(
            raw.src,
            raw.offset,
            TableDtype::from_stored(raw.dtype)?,
            row_dim,
            rows as u64,
        )?;
        // Checked, because this is the one table in the port whose row count is
        // in the hundreds of millions: 320,001,536 rows at 90 bytes is 28.8 GB,
        // three quarters of the way to a u32 overflow and comfortably inside a
        // usize — but "comfortably inside" is a fact about today's file, not
        // about the arithmetic, and the wrong answer here is a length check that
        // passes on a wrapped product.
        let total = table.row_bytes.checked_mul(rows).with_context(|| {
            format!(
                "{name}: {rows} rows of {} bytes overflows a usize",
                table.row_bytes
            )
        })?;
        ensure!(
            total == raw.len,
            "{name}: {rows} rows of {} bytes do not fill the tensor's {} bytes",
            table.row_bytes,
            raw.len
        );
        Ok(table)
    }

    /// An in-memory table from already-dequantized rows, `[rows, row_dim]` flat.
    /// The bytes are stored and re-read as F32 rather than kept as floats, so
    /// this constructor exercises the same `row` dispatch the file path does.
    pub fn from_f32(table: &[f32], row_dim: usize) -> Result<Self> {
        ensure!(
            row_dim > 0 && table.len().is_multiple_of(row_dim),
            "a {}-element table is not a whole number of {row_dim}-wide rows",
            table.len()
        );
        let mut bytes = Vec::with_capacity(table.len() * 4);
        for v in table {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let rows = (table.len() / row_dim) as u64;
        Self::build(TableBytes::Owned(bytes), 0, TableDtype::F32, row_dim, rows)
    }

    /// An in-memory table from raw quantized bytes in the file's own layout.
    pub fn from_bytes(
        bytes: Vec<u8>,
        dtype: TableDtype,
        row_dim: usize,
        rows: u64,
    ) -> Result<Self> {
        Self::build(TableBytes::Owned(bytes), 0, dtype, row_dim, rows)
    }

    pub fn row_dim(&self) -> usize {
        self.row_dim
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }

    pub fn dtype(&self) -> TableDtype {
        self.dtype
    }

    /// Byte offset of row `r` inside the table's byte source.
    ///
    /// The ONE place a row index becomes an address. Both readers go through
    /// it — [`row`](Self::row), which dequantizes, and
    /// [`prefetch`](Self::prefetch), which only faults the page in — so the
    /// prefetcher can never name a different byte than the gather will read.
    ///
    /// The bounds check is not defensive decoration: the row index is
    /// `hash mod head_vocab_size + head_offset` over attacker-visible token
    /// ids, and the table's row count is padded well past the reachable range,
    /// so a metadata misread (an offset array truncated, a vocab size read at
    /// the wrong width) produces an index that is plausible, out of range, and
    /// would otherwise read whatever the mapping holds next.
    fn row_offset(&self, r: u64) -> Result<usize> {
        ensure!(
            r < self.rows,
            "PLE row {r} is past the table's {} rows",
            self.rows
        );
        // Checked for the same reason `open`'s length check is: `r` is bounded
        // by `self.rows` just above, so this cannot overflow on a table that
        // passed `open` — and an offset that wrapped would name a valid-looking
        // in-bounds slice of somebody else's tensor, which `slice` would hand
        // back without complaint.
        (r as usize)
            .checked_mul(self.row_bytes)
            .and_then(|o| o.checked_add(self.base))
            .with_context(|| {
                format!(
                    "PLE row {r} at {} bytes per row overflows the mapping offset",
                    self.row_bytes
                )
            })
    }

    /// The prefetch thread, spawned on the first hint. `None` when the switch
    /// is set, and the `OnceLock` stays empty until somebody actually hints, so
    /// a table nothing prefetches from costs no thread.
    fn prefetcher(&self) -> Option<&PlePrefetcher> {
        self.prefetch
            .get_or_init(|| {
                (!crate::ops::ple_no_prefetch())
                    .then(|| PlePrefetcher::spawn(self.bytes.clone(), self.row_bytes))
            })
            .as_ref()
    }

    /// Ask the background thread to fault in the pages behind `rows`.
    ///
    /// PURELY ADVISORY: it moves no state, returns nothing, and every failure
    /// mode — a full queue, an out-of-range index, a dead thread — degrades to
    /// the gather taking the fault itself, which is what it did before. That is
    /// what makes it safe across a speculative rollback: a prefetch for a
    /// position that never happens costs one wasted page touch.
    pub(crate) fn prefetch(&self, rows: &[u64]) {
        let Some(pf) = self.prefetcher() else { return };
        let mut offs = Vec::with_capacity(rows.len());
        for &r in rows {
            // An out-of-range row is dropped rather than reported: `row` will
            // raise on the same index moments later, with the context this
            // call site does not have.
            if let Ok(off) = self.row_offset(r) {
                offs.push(off);
            }
        }
        pf.hint(offs);
    }

    /// `(pages touched, hints dropped)` since load, or `None` if no prefetch
    /// thread was ever started. Non-initializing on purpose — reading the
    /// counters must not be what spawns the thread.
    fn prefetch_stats(&self) -> Option<(u64, u64)> {
        let pf = self.prefetch.get()?.as_ref()?;
        Some((
            pf.pages.load(Ordering::Relaxed),
            pf.dropped.load(Ordering::Relaxed),
        ))
    }

    /// Dequantizes row `r` into `out`, which must be exactly `row_dim` wide.
    /// The index is bounds-checked by [`row_offset`](Self::row_offset).
    pub fn row(&self, r: u64, out: &mut [f32]) -> Result<()> {
        ensure!(
            out.len() == self.row_dim,
            "PLE row buffer is {} wide, not {}",
            out.len(),
            self.row_dim
        );
        let off = self.row_offset(r)?;
        let src = self.bytes.slice(off, self.row_bytes)?;
        match self.dtype {
            TableDtype::Iq4Nl => iq4nl::dequant_row(src, out),
            TableDtype::Q8_0 => iq4nl::dequant_row_q8_0(src, out),
            TableDtype::F32 => {
                for (o, c) in out.iter_mut().zip(src.chunks_exact(4)) {
                    *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
            }
            TableDtype::F16 => {
                for (o, c) in out.iter_mut().zip(src.chunks_exact(2)) {
                    *o = f32::from(half::f16::from_le_bytes([c[0], c[1]]));
                }
            }
            TableDtype::Bf16 => {
                for (o, c) in out.iter_mut().zip(src.chunks_exact(2)) {
                    *o = f32::from(half::bf16::from_le_bytes([c[0], c[1]]));
                }
            }
        }
        Ok(())
    }
}

/// A background thread that faults in the table pages a future gather will read.
///
/// The gather is not arithmetic, it is page faults: 16 unrelated 90-byte rows
/// per token scattered over a 28.8 GB mapping, measured at 1.0-1.2 ms median
/// with 6.5 ms spikes at decode and FLAT over 128 tokens (4.7% of rows hit the
/// page cache — the table is far larger than memory, so it never warms up).
/// Every one of those addresses is knowable BEFORE the forward that needs them:
/// at decode, the moment token `t` is sampled, position `t + 1`'s 16 rows are
/// determined; at prefill the whole chunk's are determined before layer 0 runs.
/// This thread takes those faults in parallel with the trunk instead of in
/// series with it.
///
/// It is a HINT and nothing else. It returns no value, touches no state, and
/// every failure path — a full queue, a bad index, a hint for a position a
/// speculative rollback then discards — costs at most one wasted page touch.
/// That is why it needs no interaction with checkpoint or rollback, and why the
/// fetch must NEVER be gated on the PLE gate value (TODO.md P3 (6)): the gate
/// is computed mid-forward, so consulting it would serialize the lookup behind
/// the very forward the prefetch exists to run ahead of.
struct PlePrefetcher {
    /// Row byte offsets to touch. Bounded and `try_send`: a hint that cannot be
    /// delivered immediately is DROPPED, because a queued hint is a hint that
    /// arrives after the gather it was meant to precede.
    tx: Option<SyncSender<Vec<usize>>>,
    /// Asks the worker to stop between rows, so dropping the table does not
    /// wait out a whole prefill chunk's worth of faults.
    stop: Arc<AtomicBool>,
    /// Distinct pages touched, for the `XWEN_PLE_PROFILE` line.
    pages: Arc<AtomicU64>,
    /// Hints refused because the queue was full — the signal that the prefetch
    /// is running behind the forward rather than ahead of it.
    dropped: Arc<AtomicU64>,
    join: Option<JoinHandle<()>>,
}

impl PlePrefetcher {
    /// Queue depth. Two: one batch in flight plus one waiting is already a hint
    /// arriving a whole forward late, and anything deeper only delays the batch
    /// that actually matters.
    const QUEUE: usize = 2;

    /// Rows per stop-flag check inside a batch. A decode hint is 16 rows and
    /// never reaches it; a prefill chunk's is thousands, and without the check
    /// a dropped table's thread would run to the end of it.
    const STOP_EVERY: usize = 64;

    fn spawn(bytes: Arc<TableBytes>, row_bytes: usize) -> Self {
        let (tx, rx) = sync_channel::<Vec<usize>>(Self::QUEUE);
        let stop = Arc::new(AtomicBool::new(false));
        let pages = Arc::new(AtomicU64::new(0));
        let (w_stop, w_pages) = (stop.clone(), pages.clone());
        let join = std::thread::Builder::new()
            .name("xwen-ple-prefetch".into())
            .spawn(move || worker(&bytes, row_bytes, &rx, &w_stop, &w_pages))
            .ok();
        Self {
            // A thread that failed to spawn drops its sender, so every later
            // hint is a no-op: prefetching is an optimization, and refusing to
            // load a model over it would be the wrong trade.
            tx: join.is_some().then_some(tx),
            stop,
            pages,
            dropped: Arc::new(AtomicU64::new(0)),
            join,
        }
    }

    fn hint(&self, offsets: Vec<usize>) {
        if offsets.is_empty() {
            return;
        }
        let Some(tx) = self.tx.as_ref() else { return };
        if tx.try_send(offsets).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Block until the worker has drained everything hinted so far, or until
    /// `timeout` passes. Only the tests need it — a forward never waits on a
    /// hint, which is the entire point — but they need it to assert on `pages`
    /// at all.
    #[cfg(test)]
    fn quiesce(&self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        let Some(tx) = self.tx.as_ref() else { return };
        // A round trip through the same bounded queue: once the worker has
        // taken QUEUE + 1 further batches, every batch sent before them has
        // been processed. The empty batches are no-ops on the worker side.
        for _ in 0..=Self::QUEUE {
            while Instant::now() < deadline && tx.try_send(Vec::new()).is_err() {
                std::thread::yield_now();
            }
        }
    }
}

impl Drop for PlePrefetcher {
    fn drop(&mut self) {
        // Order matters: raise the flag first so a worker mid-batch bails, THEN
        // drop the sender so its `recv` returns. Joining is what keeps the
        // thread from outliving the `Arc<TableBytes>` it reads — which on the
        // mapped arm is the file mapping itself.
        self.stop.store(true, Ordering::Relaxed);
        self.tx = None;
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// The prefetch thread body: touch one byte of every distinct page behind the
/// hinted rows.
///
/// A row is 90 bytes at the shipped geometry and a page is 16 KB, so a row can
/// straddle a boundary — both ends are probed. Pages are deduplicated within a
/// batch (a prefill chunk names thousands of rows, and n-gram heads collide
/// often enough to matter) in FIRST-TOUCH order rather than sorted, so the
/// thread stays ahead of a gather walking the same list.
fn worker(
    bytes: &TableBytes,
    row_bytes: usize,
    rx: &Receiver<Vec<usize>>,
    stop: &AtomicBool,
    pages: &AtomicU64,
) {
    let page = host_page_size().max(1);
    let mut seen: HashSet<usize> = HashSet::new();
    let mut sink = 0u8;
    while let Ok(batch) = rx.recv() {
        seen.clear();
        for (i, off) in batch.iter().enumerate() {
            if i.is_multiple_of(PlePrefetcher::STOP_EVERY) && stop.load(Ordering::Relaxed) {
                break;
            }
            for probe in [*off, off + row_bytes.saturating_sub(1)] {
                if !seen.insert(probe / page) {
                    continue;
                }
                let Ok(b) = bytes.slice(probe, 1) else {
                    continue;
                };
                // Volatile so the read cannot be optimized away: the VALUE is
                // worthless and the page fault it takes is the entire point.
                // SAFETY: `slice` bounds-checked the byte against the mapping.
                sink ^= unsafe { std::ptr::read_volatile(b.as_ptr()) };
                pages.fetch_add(1, Ordering::Relaxed);
            }
        }
        if stop.load(Ordering::Relaxed) {
            break;
        }
    }
    std::hint::black_box(sink);
}

/// The PLE layer's recurrent state: everything a forward carries across a chunk
/// boundary.
///
/// A PLE layer is also a DeltaNet layer, so the sequence carries THREE
/// recurrent states in total — the GDN conv state lives in the usual layer
/// cache; the two here are the PLE's own.
#[derive(Clone)]
pub struct PleState {
    /// Rolling raw-token history, at most `ngram_size - 1` ids, oldest first.
    /// Raw ids: no NFKC, no lowercasing, and the segment separator is
    /// `<|endoftext|>` (248044), not the chat stop.
    history: Vec<u32>,
    /// Depthwise conv history, `[width, (k - 1) * dilation]` channel-major with
    /// the oldest column first — the same layout the oracle's
    /// `zero_conv_state` describes, so the two can be compared directly.
    conv: Vec<f32>,
    /// `(k - 1) * dilation`, kept so the state can validate itself against a
    /// layer without carrying a reference to one.
    state_len: usize,
    /// The state after EACH token stepped since [`checkpoint`](Self::checkpoint)
    /// armed this state, in token order — `(history, conv window)` per token.
    ///
    /// Same shape and same reason as a DeltaNet layer's trail
    /// (`kv_cache::LayerCache::Linear`): both halves of this state are
    /// overwritten by every step, so no image of a single moment reconstructs an
    /// intermediate one. A partial accept — which is EVERY speculative round,
    /// since a verify keeps at least one token — needs the state after the
    /// accepted prefix, and that is only recoverable if the forward recorded it
    /// on the way past.
    ///
    /// Empty and unrecorded when `armed` is `None`, which is every ordinary
    /// prefill: a 2200-token chunk would otherwise cost 2200 × 360 KB for
    /// something nothing will read.
    trail: Vec<(Vec<u32>, Vec<f32>)>,
    /// Tokens the live checkpoint reserved room for, or `None` when unarmed.
    armed: Option<usize>,
}

/// The PLE state at the moment [`PleState::checkpoint`] was taken — the commit-0
/// answer, and only that. Every other commit comes from the trail the armed
/// state records as the verify forward runs, exactly as a DeltaNet layer's
/// `LayerCheckpoint::Linear` carries only its base.
///
/// The two halves are cloned rather than referenced: 10240 × 9 f32 (360 KB) plus
/// two token ids is small next to a KV checkpoint.
#[derive(Clone)]
pub struct PleSnapshot {
    history: Vec<u32>,
    conv: Vec<f32>,
}

impl PleState {
    /// Whether the tail of a sequence can be dropped without replaying it.
    ///
    /// It cannot. Both halves of this state are recurrent summaries: the conv
    /// history is a window of *transformed* activations and the token history
    /// is consumed by the hash, so neither can be run backwards. A caller that
    /// needs to rewind must reset and re-prefill; a caller that only needs to
    /// undo a speculative append uses [`checkpoint`](Self::checkpoint) instead,
    /// which is exact. Ledgered rather than hidden — the same shape as the MTP
    /// drafter's carry hidden.
    pub const SUPPORTS_TRUNCATE: bool = false;

    /// A zeroed state, the correct start for a fresh sequence: no history (so
    /// every predecessor slot reads as eos) and a zero conv window.
    pub fn new(width: usize, state_len: usize) -> Self {
        Self {
            history: Vec::new(),
            conv: vec![0.0; width * state_len],
            state_len,
            trail: Vec::new(),
            armed: None,
        }
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.conv.fill(0.0);
        self.trail.clear();
        self.armed = None;
    }

    /// Arm a rollback over the next `span` tokens and return the commit-0 state.
    ///
    /// Takes `&mut self` for the same reason `LayerCache::checkpoint` does: a
    /// recurrent state cannot be checkpointed by copying one moment, so the
    /// checkpoint arms the state and the verify forward records each token's as
    /// it goes.
    pub fn checkpoint(&mut self, span: usize) -> PleSnapshot {
        self.trail.clear();
        self.armed = Some(span);
        PleSnapshot {
            history: self.history.clone(),
            conv: self.conv.clone(),
        }
    }

    /// Whether a live checkpoint needs this state's per-token trail. The PLE
    /// forward asks before recording, so an unarmed prefill keeps only the final
    /// state and lets every intermediate one drop.
    pub fn trail_armed(&self) -> bool {
        self.armed.is_some()
    }

    /// Record the state after one stepped token. Called by [`PleLayer::forward`]
    /// once per token, in token order, and only while armed.
    pub fn record(&mut self, history: Vec<u32>, conv: Vec<f32>) -> Result<()> {
        let Some(span) = self.armed else {
            bail!("PleState::record on an unarmed state");
        };
        ensure!(
            self.trail.len() < span,
            "PleState::record: one more token overruns the {span}-token checkpoint span \
             ({} already recorded)",
            self.trail.len()
        );
        ensure!(
            conv.len() == self.conv.len(),
            "PleState::record: conv window is {} wide, not {}",
            conv.len(),
            self.conv.len()
        );
        self.trail.push((history, conv));
        Ok(())
    }

    /// Roll a verify forward back to the state after `commit` accepted tokens.
    ///
    /// `commit == 0` is the checkpoint's own state; `commit == n` is the state
    /// after the nth accepted token, which the trail holds at `n - 1`. This is
    /// the whole point of the trail: restoring `snap` unconditionally would
    /// rewind the conv window and the n-gram history FURTHER than the KV cache,
    /// the indexer and `n_past`, and every subsequent token would be conditioned
    /// on a history that never happened.
    ///
    /// `span` is the checkpoint's, and the trail must cover it: a state that was
    /// not stepped through every reserved token cannot answer for an arbitrary
    /// commit within it. Same refusal as the DeltaNet arm.
    pub fn rollback(&mut self, snap: &PleSnapshot, span: usize, commit: usize) -> Result<()> {
        ensure!(
            commit <= span,
            "PleState::rollback: commit {commit} exceeds span {span}"
        );
        ensure!(
            self.trail.len() == span,
            "PleState::rollback: the verify forward recorded {} of {span} states; the PLE state \
             cannot be rolled back to a token it never stepped through",
            self.trail.len()
        );
        match commit {
            0 => {
                self.history.clone_from(&snap.history);
                self.conv.clone_from(&snap.conv);
            }
            n => {
                let (history, conv) = &self.trail[n - 1];
                self.history.clone_from(history);
                self.conv.clone_from(conv);
            }
        }
        self.trail.clear();
        self.armed = None;
        Ok(())
    }

    /// Always an error — see [`SUPPORTS_TRUNCATE`](Self::SUPPORTS_TRUNCATE).
    /// Returned rather than panicked so the stack can turn it into a refused
    /// request or a forced re-prefill.
    pub fn truncate(&mut self, keep: usize) -> Result<()> {
        bail!(
            "PLE state cannot be rewound to {keep} tokens: the conv window and the n-gram token \
             history are recurrent summaries with no inverse. Reset and re-prefill, or use \
             checkpoint/rollback for a speculative append."
        )
    }

    /// The raw-token history, oldest first, for a caller that has to persist it.
    pub fn history(&self) -> &[u32] {
        &self.history
    }

    /// The conv window, `[width, state_len]` channel-major, oldest column first.
    pub fn conv_window(&self) -> &[f32] {
        &self.conv
    }
}

/// The weights and geometry of one PLE injection layer, for
/// [`PleLayer::from_parts`] — a struct rather than fourteen positional
/// arguments so a test cannot silently swap two same-typed norms.
pub struct PleParts {
    pub hash: PleHashRef,
    pub table: PleTable,
    /// `[hc_count * hidden, n_heads * head_dim]`.
    pub key_proj: QLinear,
    /// `[hidden, n_heads * head_dim]` — one value shared by every stream.
    pub value_proj: QLinear,
    /// All three norm weights are FULL width (`hc_count * hidden`); only the
    /// statistics are per `hidden`-wide group (port-doc trap #16).
    pub key_norm_w: Vec<f32>,
    pub query_norm_w: Vec<f32>,
    pub conv_norm_w: Vec<f32>,
    /// `[hc_count * hidden, k]`, depthwise: one contiguous kernel per channel.
    pub conv_w: Vec<f32>,
    pub hidden: usize,
    pub hc_count: usize,
    /// Table row width; the concatenated embedding is `n_heads * head_dim`.
    pub head_dim: usize,
    pub conv_kernel: usize,
    pub eps: f32,
}

/// One PLE injection layer.
pub struct PleLayer {
    hash: PleHashRef,
    table: PleTable,
    key_proj: QLinear,
    value_proj: QLinear,
    /// On device: the key norm is the one norm applied before the download.
    key_norm_w: Tensor,
    /// On the host: applied to the downloaded carrier and to the gated value.
    query_norm_w: Vec<f32>,
    conv_norm_w: Vec<f32>,
    conv_w: Vec<f32>,
    hidden: usize,
    hc_count: usize,
    n_heads: usize,
    head_dim: usize,
    conv_kernel: usize,
    eps: f32,
    device: Device,
}

impl PleLayer {
    /// The carrier width this layer reads and writes.
    pub fn width(&self) -> usize {
        self.hc_count * self.hidden
    }

    /// The conv dilation, which is the n-gram size (3 on the shipped
    /// checkpoint), giving a receptive field of 10 over a kernel of 4.
    ///
    /// DERIVED, never loaded: HF computes `conv_dilation = config.ngram_size`
    /// and llama.cpp reads `hparams.ple_ngram_size` for it. There is no GGUF
    /// dilation key, so a loader that invents one and defaults it to 1 would
    /// allocate a conv state it never fills and quietly shrink the receptive
    /// field.
    pub fn dilation(&self) -> usize {
        self.hash.ngram_size
    }

    /// Conv state columns per channel, `(k - 1) * dilation` — 9 on the shipped
    /// checkpoint.
    pub fn conv_state_len(&self) -> usize {
        (self.conv_kernel - 1) * self.dilation()
    }

    /// A zeroed state sized for this layer: the correct start for a sequence.
    pub fn new_state(&self) -> PleState {
        PleState::new(self.width(), self.conv_state_len())
    }

    /// Loads the layer from its `blk.N` weights plus the root-level table.
    ///
    /// GGUF names, per `reference/llama.cpp/src/llama-arch.cpp`: `ple_key`,
    /// `ple_value`, `ple_norm_key`, `ple_norm_query`, `ple_norm_conv`,
    /// `ple_conv1d` under the block prefix, and `per_layer_token_embd` at the
    /// root. `Weights` appends `.weight` to each.
    pub fn load(blk: &Weights, gguf: &GgufFile, cfg: &XwenConfig) -> Result<Self> {
        let q4 = cfg
            .qwen4exp
            .as_ref()
            .context("PLE layer on a checkpoint without qwen4exp config")?;
        let ple = q4
            .ple
            .as_ref()
            .context("PLE layer on a checkpoint whose config carries no PLE table")?;

        let host = |name: &str| -> Result<Vec<f32>> {
            blk.dense_f32(name)?
                .flatten_all()?
                .to_vec1::<f32>()
                .with_context(|| format!("reading {name} to the host"))
        };

        Self::from_parts(
            PleParts {
                hash: PleHashRef {
                    ngram_size: ple.ngram_size,
                    heads_per_ngram: ple.heads_per_ngram,
                    multipliers: ple.layer_multipliers.clone(),
                    head_vocab_sizes: ple.head_vocab_sizes.clone(),
                    head_offsets: ple.head_offsets.clone(),
                    eos: ple.eos_token_id,
                },
                table: PleTable::open(gguf, "per_layer_token_embd.weight")?,
                key_proj: blk.qlinear("ple_key")?,
                value_proj: blk.qlinear("ple_value")?,
                key_norm_w: host("ple_norm_key")?,
                query_norm_w: host("ple_norm_query")?,
                conv_norm_w: host("ple_norm_conv")?,
                conv_w: host("ple_conv1d")?,
                hidden: cfg.hidden,
                hc_count: q4.hc_count,
                head_dim: ple.row_dim,
                conv_kernel: ple.conv_kernel,
                eps: cfg.rms_eps as f32,
            },
            blk.device(),
        )
    }

    /// Assembles a layer from already-loaded parts, validating every shape that
    /// would otherwise fail silently.
    pub fn from_parts(p: PleParts, device: &Device) -> Result<Self> {
        let n_heads = p.hash.n_heads();
        let width = p.hc_count * p.hidden;
        let emb_dim = n_heads * p.head_dim;

        ensure!(p.hidden > 0 && p.hc_count > 0, "degenerate PLE geometry");
        ensure!(
            p.conv_kernel >= 2,
            "a PLE conv kernel of {} leaves no history to carry",
            p.conv_kernel
        );
        ensure!(
            p.table.row_dim() == p.head_dim,
            "the table's {}-wide rows do not match the {}-wide PLE head",
            p.table.row_dim(),
            p.head_dim
        );
        // Port-doc trap #13: `ple_head_dim * ple_n_heads == n_embd` holds by
        // coincidence on the shipped file (16 × 160 = 2560), so sizing these
        // projections from the hidden size instead of the derived PLE dim
        // works here and breaks on the next checkpoint. Assert the derived one.
        ensure!(
            p.key_proj.in_dim == emb_dim && p.key_proj.out_dim == width,
            "ple_key is [{}, {}], not [{width}, {emb_dim}]",
            p.key_proj.out_dim,
            p.key_proj.in_dim
        );
        ensure!(
            p.value_proj.in_dim == emb_dim && p.value_proj.out_dim == p.hidden,
            "ple_value is [{}, {}], not [{}, {emb_dim}]",
            p.value_proj.out_dim,
            p.value_proj.in_dim,
            p.hidden
        );
        // Trap #16: all three norm weights span the whole carrier. A
        // `[hidden]`-wide load would leave streams 1.. multiplied by nothing.
        for (name, w) in [
            ("ple_norm_key", &p.key_norm_w),
            ("ple_norm_query", &p.query_norm_w),
            ("ple_norm_conv", &p.conv_norm_w),
        ] {
            ensure!(
                w.len() == width,
                "{name} is {} wide, not the full carrier width {width}",
                w.len()
            );
        }
        ensure!(
            p.conv_w.len() == width * p.conv_kernel,
            "ple_conv1d holds {} weights, not {width} channels × {} taps",
            p.conv_w.len(),
            p.conv_kernel
        );

        let key_norm_w = Tensor::from_vec(p.key_norm_w, (1, width), device)?;
        Ok(Self {
            hash: p.hash,
            table: p.table,
            key_proj: p.key_proj,
            value_proj: p.value_proj,
            key_norm_w,
            query_norm_w: p.query_norm_w,
            conv_norm_w: p.conv_norm_w,
            conv_w: p.conv_w,
            hidden: p.hidden,
            hc_count: p.hc_count,
            n_heads,
            head_dim: p.head_dim,
            conv_kernel: p.conv_kernel,
            eps: p.eps,
            device: device.clone(),
        })
    }

    /// Grouped RMS norm on device: normalize over each `hidden`-wide stream,
    /// then multiply by the full-carrier-width weight.
    ///
    /// The host twin is [`super::ref_hc::grouped_rms_norm`], which accumulates
    /// the sum of squares in f64; this one is candle's f32 reduction, so the
    /// two agree to roughly f32 rounding rather than exactly. That is the whole
    /// numerical gap between this layer and the oracle — everything else in the
    /// forward runs the oracle's own code.
    fn grouped_norm_device(&self, x: &Tensor) -> Result<Tensor> {
        let (n, width) = x.dims2()?;
        let x3 = x.reshape((n, self.hc_count, self.hidden))?;
        let inv = (x3.sqr()?.mean_keepdim(D::Minus1)? + f64::from(self.eps))?
            .sqrt()?
            .recip()?;
        Ok(x3
            .broadcast_mul(&inv)?
            .reshape((n, width))?
            .broadcast_mul(&self.key_norm_w)?)
    }

    /// The table rows this layer gathers for `tokens` continuing `history`,
    /// flat with an `n_heads` stride.
    ///
    /// The ONE place the layer asks the hash for row indices. Both callers go
    /// through it — [`forward`](Self::forward), which gathers them, and
    /// [`prefetch`](Self::prefetch), which faults them in ahead of a later
    /// forward — so the prefetch cannot drift from the gather it is meant to
    /// precede. The hash itself is the frozen oracle's
    /// ([`PleHashRef::rows`]), called rather than reimplemented.
    fn gather_rows(&self, history: &[u32], tokens: &[u32]) -> Vec<u64> {
        self.hash.rows(history, tokens)
    }

    /// Hand the table's prefetch thread the rows a LATER forward over `tokens`
    /// will gather, given the n-gram `history` that will be in force then.
    ///
    /// `history` is the live [`PleState::history`] — at decode, after token `t`
    /// has been forwarded and `t + 1` sampled, that state's history is exactly
    /// what the next forward hashes `t + 1` against, so a caller passes it
    /// straight through. At prefill the same call names the whole chunk.
    ///
    /// Advisory in every direction: no state moves, nothing is returned, and a
    /// hint for a position that a rollback or an EOG then discards costs one
    /// wasted page touch. Never gate this on the PLE gate value (TODO.md P3
    /// (6)) — the gate is a mid-forward quantity, and waiting for it would put
    /// the lookup back on the critical path it is being lifted off.
    pub fn prefetch(&self, history: &[u32], tokens: &[u32]) {
        if tokens.is_empty() {
            return;
        }
        self.table.prefetch(&self.gather_rows(history, tokens));
    }

    /// One forward over `tokens`, whose carrier rows are `stream` `[n, width]`.
    ///
    /// Returns the ADDEND, `[n, width]` — what the caller adds to the carrier
    /// before the attention hyper-connection read. llama.cpp folds that add
    /// into `build_ple`; keeping it out here leaves the carrier's arithmetic in
    /// one place (the stack) and makes the oracle comparison direct.
    ///
    /// `state` is read for the pre-chunk history and left holding the new tail,
    /// so consecutive chunks of one sequence reproduce a single-shot run.
    pub fn forward(&self, tokens: &[u32], stream: &Tensor, state: &mut PleState) -> Result<Tensor> {
        let n = tokens.len();
        ensure!(n > 0, "PLE forward over an empty token chunk");
        let width = self.width();
        let state_len = self.conv_state_len();
        ensure!(
            stream.dims() == [n, width],
            "PLE carrier is {:?}, not [{n}, {width}]",
            stream.dims()
        );
        ensure!(
            state.conv.len() == width * state_len && state.state_len == state_len,
            "PLE conv state is {} wide over {} columns, not {width} × {state_len}",
            state.conv.len(),
            state.state_len
        );

        // Sub-step timing, off unless XWEN_PLE_PROFILE is set (ops::ple_profile).
        let mut prof = FwdProfile::start(&self.device)?;

        // --- host: hash, then gather and dequantize 16 table rows per token.
        let rows = self.gather_rows(&state.history, tokens);
        ple_step(&mut prof, "hash");
        let emb_dim = self.n_heads * self.head_dim;
        let mut emb = vec![0.0f32; n * emb_dim];
        for (t, row_set) in rows.chunks_exact(self.n_heads).enumerate() {
            for (h, &r) in row_set.iter().enumerate() {
                let off = t * emb_dim + h * self.head_dim;
                self.table
                    .row(r, &mut emb[off..off + self.head_dim])
                    .with_context(|| format!("gathering PLE row for token {t}, head {h}"))?;
            }
        }
        ple_step(&mut prof, "gather");

        // --- device: the two projections and the key norm.
        let emb_t = Tensor::from_vec(emb, (n, emb_dim), &self.device)?;
        ple_step_device(&mut prof, &self.device, "upload")?;
        let key = self.grouped_norm_device(&self.key_proj.forward(&emb_t)?)?;
        let value = self.value_proj.forward(&emb_t)?;
        ple_step_device(&mut prof, &self.device, "proj")?;

        // --- the one sync per forward (D17). 40 KB per token at the shipped
        // geometry, and the reason this layer is on P3's list.
        let key_h = key.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let value_h = value
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let stream_h = stream
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // No sync of its own: each `to_vec1` commits and waits on the way past,
        // so the device is already idle here.
        ple_step(&mut prof, "readback");

        // --- host: the per-stream gate, and the conv input's own norm.
        let scale = 1.0 / (self.hidden as f32).sqrt();
        let mut gated = vec![0.0f32; n * width];
        let mut gated_normed = vec![0.0f32; n * width];
        for t in 0..n {
            let row = t * width..(t + 1) * width;
            let query = grouped_rms_norm(
                &stream_h[row.clone()],
                &self.query_norm_w,
                self.hidden,
                self.eps,
            );
            let key_row = &key_h[row.clone()];
            let value_row = &value_h[t * self.hidden..(t + 1) * self.hidden];
            for s in 0..self.hc_count {
                let span = s * self.hidden..(s + 1) * self.hidden;
                let dot: f32 = key_row[span.clone()]
                    .iter()
                    .zip(&query[span.clone()])
                    .map(|(a, b)| a * b)
                    .sum();
                let g = gate_function_probe(dot * scale);
                let out = &mut gated[row.start + span.start..row.start + span.end];
                for (o, v) in out.iter_mut().zip(value_row) {
                    *o = g * v;
                }
            }
            let normed = grouped_rms_norm(
                &gated[row.clone()],
                &self.conv_norm_w,
                self.hidden,
                self.eps,
            );
            gated_normed[row].copy_from_slice(&normed);
        }
        ple_step(&mut prof, "gate");

        // --- host: depthwise causal conv, channel-major over state ++ chunk:
        //   out[c, t] = Σ_j w[c, j] · x[c, t - (k - 1 - j) · dilation]
        // so tap j = k-1 is the current position and tap 0 reaches the oldest
        // state column. Prepending the carried state is exactly what makes a
        // chunked prefill agree with a single-shot one.
        let line_len = state_len + n;
        let mut padded = vec![0.0f32; width * line_len];
        for c in 0..width {
            let dst = &mut padded[c * line_len..(c + 1) * line_len];
            dst[..state_len].copy_from_slice(&state.conv[c * state_len..(c + 1) * state_len]);
            for (t, d) in dst[state_len..].iter_mut().enumerate() {
                *d = gated_normed[t * width + c];
            }
        }

        let dilation = self.dilation();
        let mut out = vec![0.0f32; n * width];
        for c in 0..width {
            let line = &padded[c * line_len..(c + 1) * line_len];
            let kern = &self.conv_w[c * self.conv_kernel..(c + 1) * self.conv_kernel];
            for t in 0..n {
                let acc: f32 = kern
                    .iter()
                    .enumerate()
                    .map(|(j, w)| w * line[state_len + t - (self.conv_kernel - 1 - j) * dilation])
                    .sum();
                out[t * width + c] = gated[t * width + c] + silu(acc);
            }
            state.conv[c * state_len..(c + 1) * state_len]
                .copy_from_slice(&line[line_len - state_len..]);
        }
        ple_step(&mut prof, "conv");

        // While a checkpoint is armed, record the state after each token so a
        // partial accept can land between them. `padded` still holds the whole
        // line — the carried state followed by this chunk — so token `t`'s
        // window is the `state_len` columns ending at `state_len + t`, and the
        // last one is the same slice the loop above just committed.
        if state.trail_armed() {
            let history_before = state.history.clone();
            for t in 0..n {
                let mut window = vec![0.0f32; width * state_len];
                for c in 0..width {
                    let line = &padded[c * line_len..(c + 1) * line_len];
                    window[c * state_len..(c + 1) * state_len]
                        .copy_from_slice(&line[t + 1..t + 1 + state_len]);
                }
                state.record(
                    self.hash.next_history(&history_before, &tokens[..=t]),
                    window,
                )?;
            }
        }
        // Zero on every ordinary forward — an unarmed state records nothing —
        // and non-zero only inside a speculative verify.
        ple_step(&mut prof, "trail");

        state.history = self.hash.next_history(&state.history, tokens);
        let addend = Tensor::from_vec(out, (n, width), &self.device)?;
        ple_step_device(&mut prof, &self.device, "out_upload")?;
        ple_report(&prof, n, &rows, self.table.prefetch_stats());
        Ok(addend)
    }
}

/// One forward's sub-step wall clock, gated on [`crate::ops::ple_profile`].
///
/// Held as an `Option` by the forward, so with the switch unset every hook is
/// one `is_none` check and no clock is ever read. The steps are recorded in the
/// order they run and printed once, at the end of the forward — a forward that
/// returns an error prints nothing, because the numbers it would print would be
/// missing whichever step failed.
struct FwdProfile {
    /// Start of the open sub-step. Every interval since the previous step's
    /// close belongs to the next step recorded, so the printed steps sum to the
    /// bracket and nothing between them goes unaccounted.
    mark: Instant,
    steps: Vec<(&'static str, Duration)>,
}

impl FwdProfile {
    /// Opens the bracket from an idle device, or returns `None` when the switch
    /// is unset. The entry sync is what makes a device sub-step's figure mean
    /// "this step's GPU work" rather than "this step's GPU work plus whatever
    /// the caller left queued".
    fn start(device: &Device) -> Result<Option<Self>> {
        if !crate::ops::ple_profile() {
            return Ok(None);
        }
        device.synchronize()?;
        Ok(Some(Self {
            mark: Instant::now(),
            steps: Vec::new(),
        }))
    }
}

/// Close a host sub-step: no sync, because nothing was dispatched.
fn ple_step(p: &mut Option<FwdProfile>, name: &'static str) {
    if let Some(p) = p.as_mut() {
        let now = Instant::now();
        p.steps.push((name, now.duration_since(p.mark)));
        p.mark = now;
    }
}

/// Close a sub-step that dispatched device work, waiting for it first — the
/// same contract `stack_profile` holds its stages to.
fn ple_step_device(p: &mut Option<FwdProfile>, device: &Device, name: &'static str) -> Result<()> {
    if p.is_some() {
        device.synchronize()?;
        ple_step(p, name);
    }
    Ok(())
}

/// Print the bracket. `rows` is the table row set this forward gathered; its
/// distinct count is computed here and only here, so an unprofiled forward
/// never builds the set.
fn ple_report(p: &Option<FwdProfile>, n: usize, rows: &[u64], prefetch: Option<(u64, u64)>) {
    let Some(p) = p.as_ref() else { return };
    let distinct: std::collections::HashSet<u64> = rows.iter().copied().collect();
    let total: Duration = p.steps.iter().map(|(_, d)| *d).sum();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let mut line = format!(
        "xwen: ple-profile n={n} rows={} distinct={} total={:.3}ms",
        rows.len(),
        distinct.len(),
        ms(total),
    );
    for (name, d) in &p.steps {
        line.push_str(&format!(" {name}={:.3}", ms(*d)));
    }
    // Cumulative since load, not per forward: `pf_pages` is what the prefetch
    // thread has faulted in and `pf_dropped` the hints it was too far behind to
    // accept. A `gather` that is still slow while `pf_dropped` climbs means the
    // prefetch is running behind the forward, not that prefetching does not
    // help. Absent entirely when no hint was ever issued.
    if let Some((pages, dropped)) = prefetch {
        line.push_str(&format!(" pf_pages={pages} pf_dropped={dropped}"));
    }
    host_line(line);
}

/// `x · sigmoid(x)`, written as the oracle writes it
/// (`super::ref_ple`'s private `silu`) so the host half of the hybrid is bit-
/// identical to it rather than merely mathematically equal — `x / (1 + e^-x)`
/// is the same function and a different float.
fn silu(x: f32) -> f32 {
    x * (1.0 / (1.0 + (-x).exp()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::QTensor;

    use crate::qwen4exp::ref_ple::PleLayerRef;
    use serde_json::Value;

    /// The advertised flag and the actual behavior have to agree: a stack that
    /// reads `SUPPORTS_TRUNCATE` and skips the call must get the same answer
    /// the call would have given. Compile-time so it cannot drift.
    const _: () = assert!(!PleState::SUPPORTS_TRUNCATE);

    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen4exp/ple.json"
    );

    fn fixture() -> Value {
        serde_json::from_str(&std::fs::read_to_string(FIXTURE).unwrap()).unwrap()
    }

    fn vec_f32(v: &Value) -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    fn flat_f32(v: &Value) -> Vec<f32> {
        v.as_array().unwrap().iter().flat_map(vec_f32).collect()
    }

    fn vec_u32(v: &Value) -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    }

    fn hasher(j: &Value) -> PleHashRef {
        let c = &j["config"];
        PleHashRef {
            ngram_size: c["ngram_size"].as_u64().unwrap() as usize,
            heads_per_ngram: c["heads_per_ngram"].as_u64().unwrap() as usize,
            multipliers: c["layer_multipliers_i64_str"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s.as_str().unwrap().parse::<u64>().unwrap())
                .collect(),
            head_vocab_sizes: c["head_vocab_sizes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect(),
            head_offsets: c["head_offsets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect(),
            eos: c["eos_token_id"].as_u64().unwrap() as u32,
        }
    }

    fn reference(j: &Value) -> PleLayerRef {
        let c = &j["config"];
        let w = &j["weights"];
        PleLayerRef {
            hidden: c["hidden_size"].as_u64().unwrap() as usize,
            hc_count: c["hc_count"].as_u64().unwrap() as usize,
            n_heads: c["ngram_heads"].as_u64().unwrap() as usize,
            head_dim: c["head_dim_per_ngram"].as_u64().unwrap() as usize,
            table: flat_f32(&w["ngram_embedding_table"]),
            key_w: flat_f32(&w["key_proj"]),
            key_norm_w: vec_f32(&w["norm_key_weight_mult"]),
            value_w: flat_f32(&w["value_proj"]),
            query_norm_w: vec_f32(&w["norm_query_weight_mult"]),
            conv_norm_w: vec_f32(&w["norm_conv_weight_mult"]),
            conv_w: flat_f32(&w["conv1d_weight"]),
            k: c["ple_conv_kernel_size"].as_u64().unwrap() as usize,
            ngram_size: c["ngram_size"].as_u64().unwrap() as usize,
            eps: c["rms_norm_eps"].as_f64().unwrap() as f32,
        }
    }

    /// An exact-f32 `QLinear` over a `[out, inp]` row-major host matrix, so a
    /// test layer's projections introduce no quantization of their own and any
    /// difference from the oracle is the device norm's alone.
    fn exact_linear(w: &[f32], out_dim: usize, in_dim: usize, device: &Device) -> QLinear {
        assert_eq!(w.len(), out_dim * in_dim);
        let t = Tensor::from_slice(w, (out_dim, in_dim), device).unwrap();
        QLinear::from_qtensor(Arc::new(QTensor::quantize(&t, GgmlDType::F32).unwrap())).unwrap()
    }

    /// The device layer built from the fixture, sharing the oracle's weights.
    fn layer(j: &Value, device: &Device) -> PleLayer {
        let r = reference(j);
        let emb_dim = r.n_heads * r.head_dim;
        let width = r.width();
        PleLayer::from_parts(
            PleParts {
                hash: hasher(j),
                table: PleTable::from_f32(&r.table, r.head_dim).unwrap(),
                key_proj: exact_linear(&r.key_w, width, emb_dim, device),
                value_proj: exact_linear(&r.value_w, r.hidden, emb_dim, device),
                key_norm_w: r.key_norm_w.clone(),
                query_norm_w: r.query_norm_w.clone(),
                conv_norm_w: r.conv_norm_w.clone(),
                conv_w: r.conv_w.clone(),
                hidden: r.hidden,
                hc_count: r.hc_count,
                head_dim: r.head_dim,
                conv_kernel: r.k,
                eps: r.eps,
            },
            device,
        )
        .unwrap()
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                assert!(!x.is_nan() && !y.is_nan(), "NaN in a compared tensor");
                (x - y).abs()
            })
            .fold(0.0f32, f32::max)
    }

    fn host(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// The device layer's addend against the fixture's own `output`.
    ///
    /// Tolerance 3e-5 rather than the oracle's 4.8e-6: the key norm runs on
    /// device in f32 where the oracle sums squares in f64, and that difference
    /// rides through the gate's sigmoid into every element. Everything else in
    /// the forward is the oracle's code.
    #[test]
    fn forward_matches_the_fixture() {
        let Ok(device) = crate::gguf::metal_device() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let j = fixture();
        let l = layer(&j, &device);
        let c = &j["layer_case"];
        let toks = vec_u32(&c["input_ids"]);
        let width = l.width();

        let stream_h = flat_f32(&c["hidden_stream_in"]);
        let stream = Tensor::from_slice(&stream_h, (toks.len(), width), &device).unwrap();
        let mut state = l.new_state();
        let got = host(&l.forward(&toks, &stream, &mut state).unwrap());

        let want = flat_f32(&c["output"]);
        let d = max_abs(&got, &want);
        assert!(d <= 3e-5, "PLE addend: max abs {d:e} exceeds 3e-5");
    }

    /// `ple_conv1d` is `[width, conv_kernel]` — channel-major, so channel `c`'s
    /// taps are `conv_w[c * k .. (c + 1) * k]`. The transposed reading is the
    /// silent one: `[conv_kernel, width]` has the SAME element count, so the
    /// length check in `from_parts` passes, every shape downstream is right,
    /// and each channel simply convolves with four weights belonging to four
    /// other channels. This asserts the fixture rejects that reading, the way
    /// `value_proj`'s row width rejects its own transpose.
    #[test]
    fn a_transposed_conv_weight_does_not_match_the_fixture() {
        let Ok(device) = crate::gguf::metal_device() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let j = fixture();
        let r = reference(&j);
        let (width, k) = (r.width(), r.k);
        let emb_dim = r.n_heads * r.head_dim;

        // Read the same bytes as `[conv_kernel, width]` instead.
        let conv_w = &r.conv_w;
        let transposed: Vec<f32> = (0..width)
            .flat_map(|c| (0..k).map(move |t| conv_w[t * width + c]))
            .collect();
        assert_eq!(transposed.len(), r.conv_w.len());
        assert_ne!(
            transposed, r.conv_w,
            "the fixture's conv weights are transpose-symmetric, so this test proves nothing"
        );

        let l = PleLayer::from_parts(
            PleParts {
                hash: hasher(&j),
                table: PleTable::from_f32(&r.table, r.head_dim).unwrap(),
                key_proj: exact_linear(&r.key_w, width, emb_dim, &device),
                value_proj: exact_linear(&r.value_w, r.hidden, emb_dim, &device),
                key_norm_w: r.key_norm_w.clone(),
                query_norm_w: r.query_norm_w.clone(),
                conv_norm_w: r.conv_norm_w.clone(),
                conv_w: transposed,
                hidden: r.hidden,
                hc_count: r.hc_count,
                head_dim: r.head_dim,
                conv_kernel: r.k,
                eps: r.eps,
            },
            &device,
        )
        .unwrap();

        let c = &j["layer_case"];
        let toks = vec_u32(&c["input_ids"]);
        let stream_h = flat_f32(&c["hidden_stream_in"]);
        let stream = Tensor::from_slice(&stream_h, (toks.len(), width), &device).unwrap();
        let mut state = l.new_state();
        let got = host(&l.forward(&toks, &stream, &mut state).unwrap());

        let d = max_abs(&got, &flat_f32(&c["output"]));
        assert!(
            d > 3e-5,
            "a transposed ple_conv1d reproduced the fixture to {d:e}: the orientation is not \
             pinned by this fixture"
        );
    }

    /// The same run against the frozen oracle rather than the fixture, which
    /// isolates the device path's own deviation from the transcription's.
    #[test]
    fn forward_tracks_the_oracle() {
        let Ok(device) = crate::gguf::metal_device() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let j = fixture();
        let l = layer(&j, &device);
        let r = reference(&j);
        let h = hasher(&j);
        let c = &j["layer_case"];
        let toks = vec_u32(&c["input_ids"]);
        let width = l.width();
        let stream_h = flat_f32(&c["hidden_stream_in"]);

        let mut ref_state = r.zero_conv_state();
        let want = r
            .forward(&h.rows(&[], &toks), &stream_h, &mut ref_state)
            .output;

        let stream = Tensor::from_slice(&stream_h, (toks.len(), width), &device).unwrap();
        let mut state = l.new_state();
        let got = host(&l.forward(&toks, &stream, &mut state).unwrap());

        let d = max_abs(&got, &want);
        eprintln!("fixture geometry: device vs oracle max abs {d:e}");
        assert!(d <= 2e-5, "device vs oracle: max abs {d:e} exceeds 2e-5");

        // The conv window left behind must match the oracle's column for
        // column, or the next chunk starts from a different history.
        let d = max_abs(state.conv_window(), &ref_state);
        assert!(d <= 2e-5, "conv state: max abs {d:e} exceeds 2e-5");
    }

    /// A deterministic pseudo-random stream in `[-0.5, 0.5)`. A named LCG
    /// rather than a crate so the real-geometry case below reproduces exactly,
    /// including on a machine that resolves dependencies differently.
    fn lcg(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32) / 16_777_216.0 - 0.5
        }
    }

    /// The fixture's groups are 32 wide; the shipped checkpoint's are 2560.
    /// The one place this module diverges from the oracle is the key norm's
    /// f32 device reduction against the oracle's f64 host one, and that gap
    /// grows with the group width — so the fixture alone cannot say whether
    /// the tolerance holds at the geometry that ships. This runs the real one:
    /// hidden 2560, 4 streams, 16 heads of 160, kernel 4 at dilation 3.
    ///
    /// The table is small (one row per reachable index rather than 320
    /// million) because the row count only bounds the hash, and the hash is
    /// the oracle's own code either way.
    #[test]
    fn real_geometry_tracks_the_oracle() {
        let Ok(device) = crate::gguf::metal_device() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let (hidden, hc_count, head_dim, heads_per_ngram, ngram_size, k) = (2560, 4, 160, 8, 3, 4);
        let n_heads = (ngram_size - 1) * heads_per_ngram;
        let width = hc_count * hidden;
        let emb_dim = n_heads * head_dim;
        assert_eq!(emb_dim, hidden, "trap #13's coincidence, as shipped");

        // 16 head slices, each its own small prime, laid end to end.
        let head_vocab_sizes: Vec<u64> = vec![
            17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
        ];
        let mut head_offsets = Vec::with_capacity(n_heads);
        let mut acc = 0u64;
        for v in &head_vocab_sizes {
            head_offsets.push(acc);
            acc += v;
        }
        let table_rows = acc as usize;

        let mut rng = lcg(0xC0FFEE);
        let mut fill = |n: usize| -> Vec<f32> { (0..n).map(|_| rng()).collect() };
        let table = fill(table_rows * head_dim);
        let key_w = fill(width * emb_dim);
        let value_w = fill(hidden * emb_dim);
        let key_norm_w = fill(width);
        let query_norm_w = fill(width);
        let conv_norm_w = fill(width);
        let conv_w = fill(width * k);

        let hash = || PleHashRef {
            ngram_size,
            heads_per_ngram,
            multipliers: vec![
                112_381_549_946_653_559,
                107_512_210_695_146_523,
                55_314_113_489_879_299,
            ],
            head_vocab_sizes: head_vocab_sizes.clone(),
            head_offsets: head_offsets.clone(),
            eos: 248_044,
        };

        let r = PleLayerRef {
            hidden,
            hc_count,
            n_heads,
            head_dim,
            table: table.clone(),
            key_w: key_w.clone(),
            key_norm_w: key_norm_w.clone(),
            value_w: value_w.clone(),
            query_norm_w: query_norm_w.clone(),
            conv_norm_w: conv_norm_w.clone(),
            conv_w: conv_w.clone(),
            k,
            ngram_size,
            eps: 1e-6,
        };

        let l = PleLayer::from_parts(
            PleParts {
                hash: hash(),
                table: PleTable::from_f32(&table, head_dim).unwrap(),
                key_proj: exact_linear(&key_w, width, emb_dim, &device),
                value_proj: exact_linear(&value_w, hidden, emb_dim, &device),
                key_norm_w,
                query_norm_w,
                conv_norm_w,
                conv_w,
                hidden,
                hc_count,
                head_dim,
                conv_kernel: k,
                eps: 1e-6,
            },
            &device,
        )
        .unwrap();

        // Six tokens including an eos, so the segment cut is exercised too.
        let toks: Vec<u32> = vec![11, 4096, 248_044, 7, 1234, 99];
        let stream_h: Vec<f32> = (0..toks.len() * width).map(|_| rng()).collect();

        let mut ref_state = r.zero_conv_state();
        let want = r
            .forward(&hash().rows(&[], &toks), &stream_h, &mut ref_state)
            .output;

        let stream = Tensor::from_slice(&stream_h, (toks.len(), width), &device).unwrap();
        let mut state = l.new_state();
        let got = host(&l.forward(&toks, &stream, &mut state).unwrap());

        let d = max_abs(&got, &want);
        eprintln!("real geometry: device vs oracle max abs {d:e}");
        assert!(
            d <= 1e-4,
            "device vs oracle at real geometry: max abs {d:e}"
        );
    }

    /// Three chunks, not two: a two-chunk split only ever reads the state the
    /// first chunk wrote, so an implementation that writes the state correctly
    /// once and then corrupts it still passes. The third chunk reads a state
    /// written by a forward that itself started from a non-zero state — and
    /// the token history has to survive the same two boundaries.
    #[test]
    fn chunked_forward_matches_single_shot() {
        let Ok(device) = crate::gguf::metal_device() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let j = fixture();
        let l = layer(&j, &device);
        let c = &j["layer_case"];
        let toks = vec_u32(&c["input_ids"]);
        let width = l.width();
        let n_tok = toks.len();
        let stream_h = flat_f32(&c["hidden_stream_in"]);
        let stream = Tensor::from_slice(&stream_h, (n_tok, width), &device).unwrap();

        let mut state = l.new_state();
        let want = host(&l.forward(&toks, &stream, &mut state).unwrap());

        for a in 1..n_tok - 1 {
            for b in a + 1..n_tok {
                let mut chunk_state = l.new_state();
                let mut got: Vec<f32> = Vec::new();
                for w in [0, a, b, n_tok].windows(2) {
                    let (lo, hi) = (w[0], w[1]);
                    let piece = stream.narrow(0, lo, hi - lo).unwrap().contiguous().unwrap();
                    got.extend(host(
                        &l.forward(&toks[lo..hi], &piece, &mut chunk_state).unwrap(),
                    ));
                }
                // Close, not bit-identical: the projections and the key norm
                // run on device, where the kernel a matmul picks (and the
                // order it reduces in) depends on the token count, so a
                // one-token chunk and a three-token chunk differ in the last
                // ulps before the host half ever sees them. The host half —
                // gate, conv, state carry — is exact, which is what this test
                // is actually guarding: a dropped or mis-shifted conv column
                // moves the output by a whole tap, not by 1e-6.
                let d = max_abs(&got, &want);
                assert!(d <= 1e-5, "splits at {a} and {b}: max abs {d:e}");
            }
        }
    }

    /// The negative control for the tolerance above: dropping the carry — a
    /// fresh state at every chunk boundary, which is what a stack that forgets
    /// to thread `PleState` through would do — moves the output by orders of
    /// magnitude more than 1e-5. Without this, a 1e-5 bound on a quantity that
    /// happened to be insensitive would prove nothing.
    #[test]
    fn dropping_the_carry_is_visible_far_above_the_tolerance() {
        let Ok(device) = crate::gguf::metal_device() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let j = fixture();
        let l = layer(&j, &device);
        let c = &j["layer_case"];
        let toks = vec_u32(&c["input_ids"]);
        let width = l.width();
        let n_tok = toks.len();
        let stream_h = flat_f32(&c["hidden_stream_in"]);
        let stream = Tensor::from_slice(&stream_h, (n_tok, width), &device).unwrap();

        let mut state = l.new_state();
        let want = host(&l.forward(&toks, &stream, &mut state).unwrap());

        let split = n_tok / 2;
        let mut got: Vec<f32> = Vec::new();
        for (lo, hi) in [(0, split), (split, n_tok)] {
            let piece = stream.narrow(0, lo, hi - lo).unwrap().contiguous().unwrap();
            let mut forgotten = l.new_state();
            got.extend(host(
                &l.forward(&toks[lo..hi], &piece, &mut forgotten).unwrap(),
            ));
        }
        let d = max_abs(&got, &want);
        assert!(
            d > 1e-3,
            "a dropped conv/history carry only moved the addend by {d:e}, so the chunking \
             tolerance is not discriminating"
        );
    }

    /// Rollback restores a state exactly, which is what a rejected speculative
    /// block needs; truncation is refused, which is what the stack has to see.
    #[test]
    fn checkpoint_rolls_back_and_truncate_is_refused() {
        let Ok(device) = crate::gguf::metal_device() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let j = fixture();
        let l = layer(&j, &device);
        let c = &j["layer_case"];
        let toks = vec_u32(&c["input_ids"]);
        let width = l.width();
        let stream_h = flat_f32(&c["hidden_stream_in"]);
        let stream = Tensor::from_slice(&stream_h, (toks.len(), width), &device).unwrap();

        let mut state = l.new_state();
        let first = stream.narrow(0, 0, 2).unwrap().contiguous().unwrap();
        l.forward(&toks[..2], &first, &mut state).unwrap();

        let span = toks.len() - 2;
        let snap = state.checkpoint(span);
        let want_hist = state.history().to_vec();
        let want_conv = state.conv_window().to_vec();

        let rest = stream.narrow(0, 2, span).unwrap().contiguous().unwrap();
        l.forward(&toks[2..], &rest, &mut state).unwrap();
        assert_ne!(state.conv_window(), want_conv.as_slice());

        state.rollback(&snap, span, 0).unwrap();
        assert_eq!(state.history(), want_hist.as_slice());
        assert_eq!(state.conv_window(), want_conv.as_slice());

        assert!(state.truncate(1).is_err());

        state.reset();
        assert!(state.history().is_empty());
        assert!(state.conv_window().iter().all(|v| *v == 0.0));
    }

    /// A PARTIAL accept lands between two tokens, and the state has to land
    /// there with it.
    ///
    /// This is the whole reason the trail exists. Restoring the checkpoint
    /// unconditionally — which is what a snapshot-only rollback does — rewinds
    /// the conv window and the n-gram history all the way to the pre-verify
    /// moment while the KV cache, the indexer and `n_past` keep the accepted
    /// tokens. Nothing errors; the PLE layer simply conditions every later token
    /// on a history that never happened, for the rest of the generation.
    ///
    /// Graded against the straight-line run: after `rollback(commit = k)` the
    /// state must match a run that only ever stepped the first `k` tokens of the
    /// span. The n-gram history is compared exactly (it is token ids); the conv
    /// window to 1e-5, the same bound and for the same reason as
    /// `chunked_forward_matches_single_shot` — the two runs batch the device
    /// projections differently, so they differ in the last ulps before the host
    /// half sees them.
    ///
    /// The final assertion is the negative control that gives that bound
    /// meaning: rewinding to the CHECKPOINT instead of to the accepted prefix —
    /// the bug this trail exists to fix — moves the window by orders of
    /// magnitude more than 1e-5.
    #[test]
    fn a_partial_commit_rolls_back_to_the_state_after_the_accepted_tokens() {
        let Ok(device) = crate::gguf::metal_device() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let j = fixture();
        let l = layer(&j, &device);
        let c = &j["layer_case"];
        let toks = vec_u32(&c["input_ids"]);
        let width = l.width();
        let stream_h = flat_f32(&c["hidden_stream_in"]);
        let stream = Tensor::from_slice(&stream_h, (toks.len(), width), &device).unwrap();

        // Two committed tokens, then a `span`-token speculative block.
        let base = 2usize;
        let span = 3usize;
        assert!(toks.len() >= base + span);
        let prefix = stream.narrow(0, 0, base).unwrap().contiguous().unwrap();
        let block = stream.narrow(0, base, span).unwrap().contiguous().unwrap();

        for commit in 0..=span {
            // The straight-line reference: step the prefix, then only the
            // accepted tokens, with no checkpoint anywhere.
            let mut want = l.new_state();
            l.forward(&toks[..base], &prefix, &mut want).unwrap();
            if commit > 0 {
                let accepted = block.narrow(0, 0, commit).unwrap().contiguous().unwrap();
                l.forward(&toks[base..base + commit], &accepted, &mut want)
                    .unwrap();
            }

            // The speculative run: step the whole block, then roll back.
            let mut got = l.new_state();
            l.forward(&toks[..base], &prefix, &mut got).unwrap();
            let snap = got.checkpoint(span);
            l.forward(&toks[base..base + span], &block, &mut got)
                .unwrap();
            got.rollback(&snap, span, commit).unwrap();

            assert_eq!(
                got.history(),
                want.history(),
                "commit {commit}: n-gram history"
            );
            let d = max_abs(got.conv_window(), want.conv_window());
            assert!(d <= 1e-5, "commit {commit}: conv window max abs {d:e}");
        }

        // The negative control. A rollback that restores the checkpoint whatever
        // the commit is — the snapshot-only behavior this trail replaced — puts
        // the state a whole accepted token behind, which is nothing like 1e-5.
        let mut pre = l.new_state();
        l.forward(&toks[..base], &prefix, &mut pre).unwrap();
        let checkpoint_window = pre.conv_window().to_vec();

        let mut after_one = l.new_state();
        l.forward(&toks[..base], &prefix, &mut after_one).unwrap();
        let one = block.narrow(0, 0, 1).unwrap().contiguous().unwrap();
        l.forward(&toks[base..base + 1], &one, &mut after_one)
            .unwrap();

        let d = max_abs(&checkpoint_window, after_one.conv_window());
        assert!(
            d > 1e-3,
            "the checkpoint's window and the after-one-token window differ by only {d:e}: \
             this fixture cannot tell a correct partial rollback from a full one"
        );
    }

    /// The trail is recorded only while a checkpoint is armed, and a rollback
    /// that cannot be answered from it is refused rather than guessed at.
    #[test]
    fn the_trail_is_armed_by_a_checkpoint_and_must_cover_the_span() {
        let Ok(device) = crate::gguf::metal_device() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let j = fixture();
        let l = layer(&j, &device);
        let c = &j["layer_case"];
        let toks = vec_u32(&c["input_ids"]);
        let width = l.width();
        let stream_h = flat_f32(&c["hidden_stream_in"]);
        let stream = Tensor::from_slice(&stream_h, (toks.len(), width), &device).unwrap();

        let mut state = l.new_state();
        assert!(!state.trail_armed(), "a fresh state records nothing");
        let snap = state.checkpoint(3);
        assert!(state.trail_armed());

        // Stepped two of the three reserved tokens: commit 3 names a token this
        // state never took, and commit 0 is no safer — the trail is short, so
        // nothing here can be trusted to answer any commit.
        let two = stream.narrow(0, 0, 2).unwrap().contiguous().unwrap();
        l.forward(&toks[..2], &two, &mut state).unwrap();
        assert!(state.rollback(&snap, 3, 2).is_err());
        assert!(state.rollback(&snap, 3, 0).is_err());

        // Overrunning the reserved span is refused as it happens, not later.
        let rest = stream.narrow(0, 2, 4).unwrap().contiguous().unwrap();
        assert!(l.forward(&toks[2..6], &rest, &mut state).is_err());

        // A rollback disarms, so the next forward records nothing again.
        let mut state = l.new_state();
        let snap = state.checkpoint(2);
        let two = stream.narrow(0, 0, 2).unwrap().contiguous().unwrap();
        l.forward(&toks[..2], &two, &mut state).unwrap();
        state.rollback(&snap, 2, 1).unwrap();
        assert!(!state.trail_armed());
        assert!(
            state.rollback(&snap, 2, 1).is_err(),
            "one checkpoint answers one rollback"
        );
    }

    /// The rows the prefetcher names for the next position are exactly the
    /// rows that position's forward goes on to gather.
    ///
    /// Structurally the two share [`PleLayer::gather_rows`], so what is
    /// actually under test is the STATE the decode call site feeds it: after
    /// forwarding a prefix, the live `PleState`'s n-gram history has to produce
    /// the same rows a single-shot run over the whole sequence produces at
    /// those positions. Get that wrong — hand it the history from before the
    /// prefix, say — and the prefetch silently warms 16 pages the gather never
    /// reads, which costs nothing and helps nothing, and no other test would
    /// notice.
    #[test]
    fn prefetch_names_the_rows_the_next_forward_gathers() {
        let Ok(device) = crate::gguf::metal_device() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let j = fixture();
        let l = layer(&j, &device);
        let c = &j["layer_case"];
        let toks = vec_u32(&c["input_ids"]);
        let width = l.width();
        let stream_h = flat_f32(&c["hidden_stream_in"]);
        let stream = Tensor::from_slice(&stream_h, (toks.len(), width), &device).unwrap();

        // The shared accessor is the frozen oracle's hash and nothing else.
        let single = hasher(&j).rows(&[], &toks);
        assert_eq!(l.gather_rows(&[], &toks), single);

        let split = 3usize;
        assert!(toks.len() > split);
        let head = stream.narrow(0, 0, split).unwrap().contiguous().unwrap();
        let mut state = l.new_state();
        l.forward(&toks[..split], &head, &mut state).unwrap();

        // Exactly what the decode call site passes: the live history, and the
        // tokens the next forward will consume.
        let predicted = l.gather_rows(state.history(), &toks[split..]);
        assert_eq!(
            predicted.as_slice(),
            &single[split * l.n_heads..],
            "the prefetch and the next gather disagree about the rows"
        );

        // And the hint itself is inert: no state moves, nothing is returned,
        // and the following forward still reproduces the fixture.
        l.prefetch(state.history(), &toks[split..]);
        l.prefetch(state.history(), &[]);
        let tail = stream
            .narrow(0, split, toks.len() - split)
            .unwrap()
            .contiguous()
            .unwrap();
        let got = host(&l.forward(&toks[split..], &tail, &mut state).unwrap());
        let want = &flat_f32(&c["output"])[split * width..];
        assert!(max_abs(&got, want) <= 3e-5);
    }

    /// The prefetch thread touches one byte of every distinct page behind the
    /// rows it was handed — no more (pages repeat across heads and tokens) and
    /// no fewer (a 90-byte row can straddle a page boundary, so both ends are
    /// probed).
    ///
    /// The table is 32 pages wide so "distinct pages" is a real quantity here;
    /// on the fixture's own table every row shares one page and the assertion
    /// would hold for a worker that touched nothing but the first address.
    #[test]
    fn the_prefetch_thread_touches_every_distinct_page_once() {
        let page = crate::gguf::host_page_size();
        let (row_dim, n_rows) = (32usize, 4096usize);
        let table = PleTable::from_f32(&vec![0.5f32; row_dim * n_rows], row_dim).unwrap();
        assert!(
            table.row_bytes * n_rows > 8 * page,
            "the test table has to span many pages or it proves nothing"
        );

        let Some(pf) = table.prefetcher() else {
            eprintln!("XWEN_PLE_NO_PREFETCH is set; skipping");
            return;
        };
        // Deliberately repetitive: a duplicated row, and two rows that share a
        // page, so a worker that skipped its dedup would over-count.
        let rows: Vec<u64> = vec![0, 1, 200, 200, (n_rows - 1) as u64, 0];
        let mut want: HashSet<usize> = HashSet::new();
        for &r in &rows {
            let off = table.row_offset(r).unwrap();
            want.insert(off / page);
            want.insert((off + table.row_bytes - 1) / page);
        }

        table.prefetch(&rows);
        pf.quiesce(Duration::from_secs(5));
        assert_eq!(
            table.prefetch_stats(),
            Some((want.len() as u64, 0)),
            "pages touched (expected the {} distinct pages behind {} rows)",
            want.len(),
            rows.len()
        );

        // An index past the end is dropped by the hint rather than raised, and
        // costs the worker nothing.
        table.prefetch(&[n_rows as u64, (n_rows + 99) as u64]);
        pf.quiesce(Duration::from_secs(5));
        assert_eq!(table.prefetch_stats(), Some((want.len() as u64, 0)));
    }

    /// A table read through the IQ4_NL path returns what the quantizer put in.
    /// The blocks are hand-built rather than produced by a re-derived
    /// quantizer, so this pins the reader against the format and not against a
    /// second implementation of the same possible mistake.
    #[test]
    fn iq4nl_table_rows_dequantize() {
        // Two 32-wide rows, one block each, with different scales.
        let scales = [half::f16::from_f32(0.125), half::f16::from_f32(-3.0)];
        let idx: [[u8; 32]; 2] = [
            std::array::from_fn(|i| (i % 16) as u8),
            std::array::from_fn(|i| (15 - i % 16) as u8),
        ];
        let mut bytes = Vec::new();
        for (d, ix) in scales.iter().zip(&idx) {
            bytes.extend_from_slice(&d.to_le_bytes());
            for j in 0..16 {
                bytes.push((ix[j] & 0xf) | (ix[j + 16] << 4));
            }
        }
        assert_eq!(bytes.len(), 2 * iq4nl::BLOCK_BYTES);

        let table = PleTable::from_bytes(bytes, TableDtype::Iq4Nl, 32, 2).unwrap();
        assert_eq!(table.dtype(), TableDtype::Iq4Nl);
        let mut out = [0.0f32; 32];
        for r in 0..2u64 {
            table.row(r, &mut out).unwrap();
            for (i, v) in out.iter().enumerate() {
                let want = f32::from(scales[r as usize])
                    * f32::from(iq4nl::KVALUES_IQ4NL[idx[r as usize][i] as usize]);
                assert_eq!(*v, want, "row {r} element {i}");
            }
        }

        // Row 2 does not exist. The index comes from a hash over token ids, so
        // this bound is the difference between an error and reading whatever
        // the mapping holds next.
        assert!(table.row(2, &mut out).is_err());
    }

    /// Row striding is 90 bytes at the shipped 160-wide row — five IQ4_NL
    /// blocks. Written out because the stride is the one number that turns a
    /// correct dequantizer into a table that gathers the wrong rows.
    #[test]
    fn shipped_row_stride_is_ninety_bytes() {
        assert_eq!(TableDtype::Iq4Nl.row_bytes(160).unwrap(), 90);
        assert_eq!(TableDtype::Q8_0.row_bytes(160).unwrap(), 170);
        assert_eq!(TableDtype::F32.row_bytes(160).unwrap(), 640);
        assert_eq!(TableDtype::Bf16.row_bytes(160).unwrap(), 320);
        // A row that is not a whole number of blocks has no stride at all.
        assert!(TableDtype::Iq4Nl.row_bytes(20).is_err());
    }

    /// Trap #16: a `[hidden]`-wide norm weight must be refused at construction.
    /// It would otherwise normalize stream 0 and leave streams 1.. multiplied
    /// by whatever fell off the end — a model that runs and is quietly wrong.
    #[test]
    fn a_narrow_norm_weight_is_refused() {
        let Ok(device) = crate::gguf::metal_device() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let j = fixture();
        let r = reference(&j);
        let emb_dim = r.n_heads * r.head_dim;
        let width = r.width();
        let parts = |query_norm_w: Vec<f32>| PleParts {
            hash: hasher(&j),
            table: PleTable::from_f32(&r.table, r.head_dim).unwrap(),
            key_proj: exact_linear(&r.key_w, width, emb_dim, &device),
            value_proj: exact_linear(&r.value_w, r.hidden, emb_dim, &device),
            key_norm_w: r.key_norm_w.clone(),
            query_norm_w,
            conv_norm_w: r.conv_norm_w.clone(),
            conv_w: r.conv_w.clone(),
            hidden: r.hidden,
            hc_count: r.hc_count,
            head_dim: r.head_dim,
            conv_kernel: r.k,
            eps: r.eps,
        };
        assert!(PleLayer::from_parts(parts(r.query_norm_w.clone()), &device).is_ok());
        assert!(PleLayer::from_parts(parts(vec![1.0; r.hidden]), &device).is_err());
    }

    /// The conv dilation is the n-gram size and the state is nine columns —
    /// both derived, neither loaded, and the fixture carries them
    /// independently so this is an assertion rather than a tautology.
    #[test]
    fn conv_geometry_is_derived_from_the_ngram_size() {
        let Ok(device) = crate::gguf::metal_device() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let j = fixture();
        let l = layer(&j, &device);
        assert_eq!(
            l.dilation(),
            j["config"]["conv_dilation"].as_u64().unwrap() as usize
        );
        assert_eq!(
            l.conv_state_len(),
            j["config"]["conv_state_len"].as_u64().unwrap() as usize
        );
    }
}
