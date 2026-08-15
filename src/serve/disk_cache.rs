//! The `.lkv` container: one file per persisted prefix SEGMENT.
//!
//! A conversation is not stored as one image but as a chain of segments, each
//! covering a token span `[start, end)` and naming the segment that covers the span
//! before it. Shared prefixes are stored once — two conversations behind the same
//! 20k-token system prompt share the base segment and keep only their own tails —
//! and a conversation that grew by a turn writes a tail instead of rewriting a
//! gigabyte. A segment holds the full-attention rows for its own span, the SWA ring
//! snapshots taken inside it, and the drafter's rows when speculation is on, wrapped
//! in a header that says which checkpoint and which tokenization rules produced it.
//! The records themselves are `kv_cache::{HostFullKv, HostSnapshot}` and
//! `dflash::DrafterImage`, which frame their own bodies; this module owns only the
//! header, the record directory, the chain identity and the atomic write.
//!
//! Persistence is PERF-ONLY. Every way a file can be wrong ends in a cache miss,
//! never in one conversation being served another's KV, which is why nothing here
//! has a trusted fast path: a record is rebuilt through its validating
//! constructor or not at all.
//!
//! Layout, little-endian throughout:
//!
//! ```text
//! magic "LAGKVIMG"                    8 bytes
//! container version                   u32
//! checkpoint hash                     u64
//! checkpoint file length              u64
//! tokenization rules version          u32
//! container digest                    u64 (hashed as zeros)
//! start position                      u64
//! parent name                         u64  (only when start > 0)
//! parent chain hash                   u64  (only when start > 0)
//! span token count                    u32
//! span token ids                      count x u32
//! record count                        u32
//! record directory                    count x { tag u32, pos u64, byte length u64 }
//! record bodies                       in directory order
//! ```
//!
//! The directory carries each record's position and byte length so a reader can
//! learn what a file covers — which is all the startup scan needs — without
//! touching a byte of the gigabyte-scale planes behind it. Records appear in a
//! fixed order: exactly one `HostFullKv` covering the span's rows, then any number
//! of `HostSnapshot`s at ascending positions inside the span, then at most one
//! `DrafterImage`. A segment with no snapshot at all is ordinary: a tail written
//! after a fork can legitimately hold none, and the positions its chain can resume
//! at then all live in its ancestors.
//!
//! Positions in the directory are ABSOLUTE for snapshots and for the drafter — they
//! name a token index in the conversation — while the full-attention record's is a
//! ROW COUNT, the length of the span it covers. That asymmetry follows the records
//! themselves: a snapshot restores to a position, while `HostFullKv` is a block of
//! rows that knows nothing about where in a conversation it sits.
//!
//! The digest is FNV-1a 64 over the WHOLE container — magic through the last plane
//! byte, with the digest field itself hashed as eight zeros so it can cover the file
//! it lives in. The writer lays those zeros down, hashes them, and patches the real
//! value in afterwards; the reader mixes zeros in their place. Checked by the full
//! read and skipped by the header-only one.
//!
//! It covers everything because everything is load-bearing. Shape validation cannot
//! see a flipped bit inside a plane — the lengths still add up and the shapes still
//! describe their bytes — and APFS checksums metadata but not file data. The token
//! ids matter just as much: a hydration is only safe because the stored history is
//! compared against the arriving prompt, and a bit flip in a token id can make a
//! corrupted history match a DIFFERENT conversation past the flip, which is the one
//! way this tier could serve one conversation another's keys. It is a corruption
//! check and nothing more: anything that can rewrite a plane can recompute the
//! digest.
//!
//! What the digest deliberately does NOT cover is the parent's bytes. A segment's
//! parent reference binds the chain's IDENTITY — which tokens precede this span —
//! and not the parent file's contents, because a parent that is split later keeps
//! its identity while its bytes change, and binding to bytes would orphan every
//! child of every split. Nothing is left unprotected: a parent's planes are covered
//! by the parent's own digest, verified on every read, and the final authority is
//! neither hash but the token-id comparison against the live prompt.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::dflash::DrafterImage;
use crate::gguf::CheckpointId;
use crate::kv_cache::{
    HostFullKv, HostSnapshot, MAX_STORED_POS, MAX_STORED_SNAPSHOTS, MAX_STORED_TOKENS, write_count,
    write_u32, write_u64,
};

/// First bytes of every `.lkv` file.
const MAGIC: &[u8; 8] = b"LAGKVIMG";

/// Container version. Bumped when the FRAMING changes — the header fields, the
/// directory layout, the record tags — and nothing else. A reader refuses
/// anything it does not know, and there is no migration path: a rejected file
/// costs a re-prefill, so the cheap answer is always to discard and rebuild.
/// Version 1 was the flat one-file-per-conversation image, and a v1 file read by
/// this build is an ordinary `Binding` rejection, which is to say it is deleted
/// at scan.
///
/// INVARIANT: this version does NOT discriminate what a snapshot record
/// contains. Two things already do, and between them leave nothing for a version
/// bump to catch. The checkpoint binding — hash plus file length, checked in
/// `read_header` — means an image can only ever be read back beside the exact
/// file that wrote it, so a Laguna-era image cannot reach a Qwen build at all.
/// Within a checkpoint, `kv_cache`'s per-layer kind tags (`LAYER_FULL` /
/// `LAYER_SWA` / `LAYER_LINEAR`) give each layer kind its own field layout and
/// dtype, and `check_restorable` rejects a layer whose kind or shape does not
/// match the live cache. That is why the DeltaNet recurrent-state snapshot
/// landed here without a bump: it is a new per-layer tag inside an unchanged
/// framing, and the tag is what tells it apart.
///
/// Version 3 is where the drafter record (`TAG_DRAFTER`) grew its own kind tag
/// and a carry field, because the two drafter kinds a checkpoint can ship differ
/// in element size AND field list. The bump is what makes a v2 record — which
/// begins with the position where a v3 one begins with the kind — unreadable
/// rather than misread: the checkpoint binding cannot help here, since the same
/// target file can be served with either drafter attached.
const CONTAINER_VERSION: u32 = 3;

/// Extension every stored segment carries. The name in front of it is the segment's
/// identity (see [`chain_id`]).
pub const EXTENSION: &str = "lkv";

/// Record type tags, in the order the directory must list them.
const TAG_FULL_KV: u32 = 1;
const TAG_SNAPSHOT: u32 = 2;
const TAG_DRAFTER: u32 = 3;

/// Bytes from the magic through the start position: magic, container version,
/// checkpoint id, tokenization version, container digest, start.
const HEAD_FIXED: u64 = 8 + 4 + 8 + 8 + 4 + 8 + 8;

/// Bytes a parent reference occupies, present only when `start > 0`.
const PARENT_REF: u64 = 8 + 8;

/// The shortest header that could be complete: a root segment's fixed fields plus
/// the two counts, with no tokens and no records behind them.
const HEADER_MIN: u64 = HEAD_FIXED + 4 + 4;

/// Byte offset of the digest, which is the one field the writer cannot know when it
/// reaches it: the file is streamed straight through (gigabytes, never buffered to be
/// measured), so eight zeros go down in its place and the real value is patched in
/// once the last plane has gone past. The seek is over a `.tmp` nothing can see until
/// the rename, and the zeros are what both sides hash, which is how a digest over the
/// whole container covers the field it sits in.
const DIGEST_OFFSET: u64 = 8 + 4 + 8 + 8 + 4;

/// Bytes one record-directory entry occupies.
const DIRECTORY_ENTRY: u64 = 4 + 8 + 8;

/// Why a `.lkv` file cannot be used. The two rejection classes are kept apart
/// because they say different things about the machine: a `Binding` file is
/// intact and simply belongs to a model, a tokenizer or a container version that is
/// no longer in use, while a `Corrupt` one means bytes on disk do not hold together.
/// Both end in the file being dropped; only one is worth raising an eyebrow at.
#[derive(Debug)]
pub enum DiskImageError {
    /// The file was produced by a different container version, checkpoint or set
    /// of tokenization rules. Nothing is wrong with it — it just cannot be used
    /// here.
    Binding(String),
    /// The file does not hold together: a bad magic, a truncated body, a declared
    /// length that disagrees with the file's size, or a record whose bytes do not
    /// match the shape it claims.
    Corrupt(String),
    /// The file could not be read at all — a missing file, a permission problem,
    /// a failing disk. Says nothing about the image's contents.
    Io(std::io::Error),
}

impl std::fmt::Display for DiskImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Binding(why) => write!(f, "segment does not bind to this server: {why}"),
            Self::Corrupt(why) => write!(f, "segment is corrupt: {why}"),
            Self::Io(e) => write!(f, "segment could not be read: {e}"),
        }
    }
}

impl std::error::Error for DiskImageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DiskImageError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl DiskImageError {
    /// The class a rejection falls in, for a log line that has to name it.
    pub fn class(&self) -> &'static str {
        match self {
            Self::Binding(_) => "binding",
            Self::Corrupt(_) => "corrupt",
            Self::Io(_) => "io",
        }
    }
}

type Result<T> = std::result::Result<T, DiskImageError>;

fn corrupt(why: impl std::fmt::Display) -> DiskImageError {
    DiskImageError::Corrupt(why.to_string())
}

/// Classify a record reader's failure, which arrives as an `anyhow::Error` that may
/// have an `io::Error` under it.
///
/// A body that ran out of bytes IS a verdict about the file — it is shorter than it
/// says it is — and the file is deleted for it. Any other I/O fault is not: a disk
/// that returned EIO on one read, a file yanked out from under the reader, a
/// permission change mid-run. Treating those as corruption would delete a
/// multi-gigabyte segment that is almost certainly intact, so the distinction is
/// carried rather than flattened.
fn record_error(e: anyhow::Error) -> DiskImageError {
    let transient = e
        .chain()
        .filter_map(|cause| cause.downcast_ref::<std::io::Error>())
        .find(|io| io.kind() != std::io::ErrorKind::UnexpectedEof);
    match transient {
        Some(io) => DiskImageError::Io(std::io::Error::new(io.kind(), format!("{e:#}"))),
        None => corrupt(format!("{e:#}")),
    }
}

/// FNV-1a 64, the same construction the checkpoint id uses: a few lines inline, no
/// dependency, and no cryptographic claim. It answers "are these the same tokens",
/// and the token-id comparison at hydration is what makes that answer safe to be
/// wrong about.
const FNV_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x100_0000_01b3;

/// What the second chain hash's basis is offset by, so the two hashes over the same
/// token stream are independent rather than equal. Any constant would do; this is
/// the golden-ratio word the usual mixers use.
const CHAIN_SALT: u64 = 0x9e37_79b9_7f4a_7c15;

fn fnv_mix(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

/// A segment's identity: two independent FNV-1a 64 hashes over the CUMULATIVE token
/// history `[0, end)` the segment ends at — not over its own span.
///
/// Hashing the cumulative history is what makes the tree work:
///
/// - **Splitting is stable.** Cutting a segment `[p, n)` at k leaves a tail `[k, n)`
///   whose cumulative history is unchanged, so its name is unchanged, so it replaces
///   the old file by rename and every child that pointed at that name still does.
/// - **Dedup is by construction.** Two conversations that share a prefix compute the
///   same name for the segment covering it, so the second one finds the first's file
///   instead of writing its own copy.
/// - **Crashes leave valid states.** A split writes the base under a new name before
///   the tail is rewritten under the old one; a crash in between leaves the un-split
///   segment plus a base nothing points at, which is imperfect dedup and never
///   corruption.
///
/// `name` is the file name. `chain` is what a child stores next to it in its parent
/// reference, so a child binds to WHICH TOKENS precede it with a second 64-bit
/// witness — cheap, and independent of the parent's bytes, which is what lets a
/// parent be re-partitioned without touching its children.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainId {
    pub name: u64,
    pub chain: u64,
}

impl ChainId {
    /// The reference a child stores to name this segment as its parent.
    pub fn as_parent(&self) -> ParentRef {
        ParentRef {
            name: self.name,
            chain: self.chain,
        }
    }

    /// The file this identity lives in, under the checkpoint's directory.
    pub fn file_name(&self) -> String {
        segment_file_name(self.name)
    }
}

/// The identity of the segment ending at `history.len()`, over the whole history
/// that leads there. The count is mixed in first, so a history and the same history
/// extended by a zero token cannot hash alike.
pub fn chain_id(history: &[u32]) -> ChainId {
    ChainId {
        name: fnv_history(FNV_BASIS, history),
        chain: fnv_history(FNV_BASIS ^ CHAIN_SALT, history),
    }
}

fn fnv_history(basis: u64, history: &[u32]) -> u64 {
    let mut hash = fnv_mix(basis, &(history.len() as u64).to_le_bytes());
    for id in history {
        hash = fnv_mix(hash, &id.to_le_bytes());
    }
    hash
}

/// The file name a segment name resolves to. The mapping is the whole parent
/// lookup: a child names its parent, and the parent is at that name.
pub fn segment_file_name(name: u64) -> String {
    format!("{name:016x}.{EXTENSION}")
}

/// The segment name a path claims, or `None` when the file is not named by this
/// rule and so is not one of ours to read or to parent anything.
///
/// Canonical form only — the name is rendered back and compared — so exactly one
/// spelling of a name can exist in the directory.
pub fn segment_name_of(path: &Path) -> Option<u64> {
    let stem = path.file_stem()?.to_str()?;
    let name = u64::from_str_radix(stem, 16).ok()?;
    (segment_file_name(name) == path.file_name()?.to_str()?).then_some(name)
}

/// A segment's reference to the one covering the span before it. Absent exactly when
/// the segment starts at position 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParentRef {
    /// The parent's name, which is also its file name.
    pub name: u64,
    /// The parent's chain hash, the second witness that the parent covers the tokens
    /// this segment expects to sit behind.
    pub chain: u64,
}

/// What a segment covers, read without loading any plane bytes: enough to place it
/// in the tree and to decide whether hydrating its chain is worth doing.
#[derive(Debug, Clone)]
pub struct SegmentHeader {
    /// First position the segment covers.
    pub start: usize,
    /// The segment before it, absent iff `start == 0`.
    pub parent: Option<ParentRef>,
    /// The token ids of the span `[start, end)`, in order. Matching is a
    /// longest-common-prefix over the cumulative histories these concatenate into.
    pub tokens: Vec<u32>,
    /// Absolute positions the stored ring snapshots restore to, ascending, all
    /// inside `(start, end]`. A resume can only land on one of these, or on one in
    /// an ancestor.
    pub snapshot_positions: Vec<usize>,
    /// The position a drafter image rides along at, when one does.
    pub drafter_pos: Option<usize>,
    /// The file's total size, for the disk budget.
    pub bytes: u64,
}

impl SegmentHeader {
    /// One past the last position the segment covers.
    pub fn end(&self) -> usize {
        self.start + self.tokens.len()
    }
}

/// A fully read and validated segment: the records, rebuilt through their own
/// constructors, ready to be composed into a chain.
pub struct Segment {
    pub start: usize,
    pub parent: Option<ParentRef>,
    /// Token ids of the span `[start, end)`.
    pub tokens: Vec<u32>,
    /// The span's full-attention rows, `pos` == the span's length.
    pub full_kv: HostFullKv,
    /// Ring snapshots with the absolute positions they restore to, ascending.
    pub snapshots: Vec<(usize, HostSnapshot)>,
    pub drafter: Option<DrafterImage>,
}

impl Segment {
    pub fn end(&self) -> usize {
        self.start + self.tokens.len()
    }
}

/// A whole chain, composed and ready to install into a cache slot: exactly what a
/// slot paged out in this process holds, so the install path cannot tell a hydration
/// from an ordinary swap.
pub struct DiskImage {
    /// The cumulative history `[0, end)` of the chain's last segment.
    pub tokens: Vec<u32>,
    /// The full-attention rows for all of it, the spans re-strided into one image.
    pub full_kv: HostFullKv,
    /// Every ring snapshot in the chain, with the positions they restore to,
    /// ascending.
    pub snapshots: Vec<(usize, HostSnapshot)>,
    pub drafter: Option<DrafterImage>,
}

/// One record as the directory describes it.
#[derive(Debug, Clone, Copy)]
struct DirectoryEntry {
    tag: u32,
    pos: usize,
    len: u64,
}

/// Write a segment to `path`, atomically: the bytes go to a `.tmp` sibling that is
/// flushed and synced before being renamed over `path`, so a reader (or a crash)
/// never sees a half-written file under the real name. The sibling is removed on
/// any failure.
///
/// `tokens` is the span's own ids and `full_kv` its own rows, so `full_kv.pos` must
/// be the span's length. `snapshots` pairs each ring snapshot with the ABSOLUTE
/// position it restores to, as a slot holds them, and every one must fall inside
/// `(start, end]` — a snapshot at `start` belongs to the parent, which is the
/// segment whose span ends there. A segment with no snapshot is allowed: the
/// positions its chain resumes at may all live in its ancestors.
///
/// Returns the file's size in bytes.
#[allow(clippy::too_many_arguments)]
pub fn write_segment(
    path: &Path,
    checkpoint: &CheckpointId,
    tokenization_rules: u32,
    start: usize,
    parent: Option<ParentRef>,
    tokens: &[u32],
    full_kv: &HostFullKv,
    snapshots: &[(usize, &HostSnapshot)],
    drafter: Option<&DrafterImage>,
) -> anyhow::Result<u64> {
    anyhow::ensure!(
        !tokens.is_empty(),
        "disk segment: a segment covering no token covers nothing"
    );
    anyhow::ensure!(
        parent.is_some() == (start > 0),
        "disk segment: a segment starting at {start} {} a parent",
        if start > 0 { "needs" } else { "cannot have" }
    );
    let end = start
        .checked_add(tokens.len())
        .ok_or_else(|| anyhow::anyhow!("disk segment: span end overflows"))?;
    anyhow::ensure!(
        full_kv.pos == tokens.len(),
        "disk segment: the rows cover {} positions, the span is {} tokens",
        full_kv.pos,
        tokens.len()
    );
    let mut previous = None;
    for (pos, snapshot) in snapshots {
        anyhow::ensure!(
            *pos == snapshot.pos,
            "disk segment: snapshot listed at position {pos} restores to {}",
            snapshot.pos
        );
        // Both properties the reader insists on. A snapshot outside the span names a
        // position this file's rows cannot back, and the ascending order is what lets
        // a scan pick a resume depth off the directory alone. A slot maintains both;
        // writing a file that violates them would only ever be written to be
        // rejected.
        anyhow::ensure!(
            *pos > start && *pos <= end,
            "disk segment: snapshot at {pos} is outside the span ({start}, {end}]"
        );
        anyhow::ensure!(
            previous.is_none_or(|prev| prev < *pos),
            "disk segment: snapshot positions are not ascending ({previous:?} then {pos})"
        );
        previous = Some(*pos);
    }
    if let Some(drafter) = drafter {
        anyhow::ensure!(
            drafter.pos > start && drafter.pos <= end,
            "disk segment: drafter rows reach {} , outside the span ({start}, {end}]",
            drafter.pos
        );
    }
    anyhow::ensure!(
        u32::try_from(tokens.len()).is_ok(),
        "disk segment: {} tokens do not fit the format's u32 count",
        tokens.len()
    );

    let tmp = tmp_sibling(path);
    let written = write_to_tmp(
        &tmp,
        checkpoint,
        tokenization_rules,
        start,
        parent,
        tokens,
        full_kv,
        snapshots,
        drafter,
    );
    match written {
        Ok(bytes) => {
            if let Err(e) = std::fs::rename(&tmp, path) {
                let _ = std::fs::remove_file(&tmp);
                return Err(anyhow::Error::new(e).context(format!(
                    "renaming {} into place at {}",
                    tmp.display(),
                    path.display()
                )));
            }
            Ok(bytes)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// The sibling an in-progress write goes to. A sibling rather than a temp
/// directory so the rename stays within one filesystem, which is what makes it
/// atomic; stamped with the process id so two servers sharing a cache directory
/// cannot write the same partial file, in which case the rename simply resolves to
/// last-writer-wins over two whole segments. A crash can therefore leave a `.tmp`
/// behind that no later write to the same path reclaims — the directory scan is
/// what sweeps those.
fn tmp_sibling(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".{}.tmp", std::process::id()));
    path.with_file_name(name)
}

#[allow(clippy::too_many_arguments)]
fn write_to_tmp(
    tmp: &Path,
    checkpoint: &CheckpointId,
    tokenization_rules: u32,
    start: usize,
    parent: Option<ParentRef>,
    tokens: &[u32],
    full_kv: &HostFullKv,
    snapshots: &[(usize, &HostSnapshot)],
    drafter: Option<&DrafterImage>,
) -> anyhow::Result<u64> {
    let file =
        File::create(tmp).map_err(|e| anyhow::Error::new(e).context(tmp.display().to_string()))?;
    let w = BufWriter::new(file);

    // Every record's length is known before a byte of it is written, so the
    // directory can be laid down first and the gigabyte-scale planes streamed
    // straight through afterwards — nothing seeks back to patch a length, and no
    // segment is buffered in RAM to be measured. (The digest is patched, once, and it
    // is the one field that cannot be known in advance.)
    let mut directory = vec![DirectoryEntry {
        tag: TAG_FULL_KV,
        pos: full_kv.pos,
        len: full_kv.serialized_len() as u64,
    }];
    for (pos, snapshot) in snapshots {
        directory.push(DirectoryEntry {
            tag: TAG_SNAPSHOT,
            pos: *pos,
            len: snapshot.serialized_len() as u64,
        });
    }
    if let Some(drafter) = drafter {
        directory.push(DirectoryEntry {
            tag: TAG_DRAFTER,
            pos: drafter.pos,
            len: drafter.serialized_len() as u64,
        });
    }

    // Digesting from the first byte: the digest covers the whole container, and the
    // eight zeros written for its own field below are what it hashes there.
    let mut w = Digesting::new(w);
    w.write_all(MAGIC)?;
    write_u32(&mut w, CONTAINER_VERSION)?;
    write_u64(&mut w, checkpoint.hash())?;
    write_u64(&mut w, checkpoint.file_len())?;
    write_u32(&mut w, tokenization_rules)?;
    // Hashed as zeros, patched with the real value once the file is complete.
    write_u64(&mut w, 0)?;
    write_count(&mut w, start)?;
    if let Some(parent) = parent {
        write_u64(&mut w, parent.name)?;
        write_u64(&mut w, parent.chain)?;
    }
    write_u32(&mut w, tokens.len() as u32)?;
    for &id in tokens {
        write_u32(&mut w, id)?;
    }
    write_u32(&mut w, directory.len() as u32)?;
    for entry in &directory {
        write_u32(&mut w, entry.tag)?;
        write_count(&mut w, entry.pos)?;
        write_u64(&mut w, entry.len)?;
    }

    full_kv.write_to(&mut w)?;
    for (_, snapshot) in snapshots {
        snapshot.write_to(&mut w)?;
    }
    if let Some(drafter) = drafter {
        drafter.write_to(&mut w)?;
    }
    let (w, digest) = w.finish();

    let mut file = w
        .into_inner()
        .map_err(|e| anyhow::anyhow!("flushing {}: {e}", tmp.display()))?;
    file.seek(SeekFrom::Start(DIGEST_OFFSET))?;
    file.write_all(&digest.to_le_bytes())?;
    // Durability is not the point — a lost segment costs a re-prefill. The sync is
    // here so the rename cannot publish a name whose contents are still in flight.
    file.sync_all()?;
    Ok(file.metadata()?.len())
}

/// A writer that digests everything passing through it, so the record bodies are
/// hashed as they stream to disk rather than in a second pass over gigabytes.
struct Digesting<W> {
    inner: W,
    hash: u64,
}

impl<W: Write> Digesting<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hash: FNV_BASIS,
        }
    }

    fn finish(self) -> (W, u64) {
        (self.inner, self.hash)
    }
}

impl<W: Write> Write for Digesting<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(buf)?;
        self.hash = fnv_mix(self.hash, &buf[..written]);
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// The reading counterpart: the file is digested as it is consumed, so nothing is
/// buffered to be hashed and a record still reads straight into its own planes.
struct Digested<R> {
    inner: R,
    hash: u64,
}

impl<R: Read> Digested<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            hash: FNV_BASIS,
        }
    }
}

impl<R: Read> Read for Digested<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let read = self.inner.read(buf)?;
        self.hash = fnv_mix(self.hash, &buf[..read]);
        Ok(read)
    }
}

/// A reader over a container, which may or may not be digesting what it reads.
///
/// The one field that needs a say in this is the digest itself: it cannot cover its
/// own bytes, so both sides hash eight zeros in its place. A header-only parse
/// digests nothing and simply reads it.
trait ContainerRead: Read {
    fn read_digest(&mut self) -> std::io::Result<u64>;
}

impl<R: Read> ContainerRead for Digested<R> {
    fn read_digest(&mut self) -> std::io::Result<u64> {
        let mut buf = [0u8; 8];
        self.inner.read_exact(&mut buf)?;
        self.hash = fnv_mix(self.hash, &[0u8; 8]);
        Ok(u64::from_le_bytes(buf))
    }
}

impl ContainerRead for BufReader<File> {
    fn read_digest(&mut self) -> std::io::Result<u64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }
}

/// Read a segment's header and record directory, stopping before the first plane
/// byte. This is what a startup scan runs over every file in the cache directory,
/// so it must stay cheap: one sequential read of tens of kilobytes regardless of
/// how many gigabytes sit behind it.
pub fn read_segment_header(
    path: &Path,
    expected: &CheckpointId,
    tokenization_rules: u32,
) -> Result<SegmentHeader> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut r = BufReader::new(file);
    let parsed = read_header_into(&mut r, file_len, expected, tokenization_rules)?;
    Ok(header_of(parsed, file_len))
}

/// Read and validate a whole segment, rebuilding every record through its own
/// validating constructor. A shape that does not describe its bytes fails inside
/// the record, which is the point: no code path in this module can produce a
/// record the in-process paths would not also accept.
pub fn read_segment(
    path: &Path,
    expected: &CheckpointId,
    tokenization_rules: u32,
) -> Result<Segment> {
    read_segment_from(File::open(path)?, expected, tokenization_rules)
}

/// The same read over a file that is already open.
///
/// The caller that has to know how big the read will be before paying for it opens the
/// file itself, sizes it from THAT descriptor, and hands it over — so the length it
/// budgeted for and the bytes that arrive are the same file. Going back to the path for
/// a second look would be a different question: a rename can land between the two, and
/// the answer would describe a file this read never touches.
pub fn read_segment_from(
    file: File,
    expected: &CheckpointId,
    tokenization_rules: u32,
) -> Result<Segment> {
    let file_len = file.metadata()?.len();
    // Digesting from the first byte: the header and the token ids are covered as well
    // as the planes, because a hydration's safety rests on the stored token ids being
    // the conversation they claim to be.
    let mut r = Digested::new(BufReader::new(file));
    let Parsed {
        start,
        parent,
        tokens,
        directory,
        digest,
    } = read_header_into(&mut r, file_len, expected, tokenization_rules)?;

    let mut full_kv = None;
    let mut snapshots = Vec::new();
    let mut drafter = None;
    for entry in &directory {
        // Reading exactly `entry.len` bytes per record keeps a body that overruns
        // its frame from eating into the next one: the record's own reader refuses
        // the read, and a body that stops short is caught by its `finish`.
        let body = (&mut r).take(entry.len);
        match entry.tag {
            TAG_FULL_KV => {
                let image = HostFullKv::read_from(body, entry.len).map_err(record_error)?;
                if image.pos != entry.pos {
                    return Err(corrupt(format!(
                        "full-attention record covers {} positions, the directory says {}",
                        image.pos, entry.pos
                    )));
                }
                full_kv = Some(image);
            }
            TAG_SNAPSHOT => {
                let snapshot = HostSnapshot::read_from(body, entry.len).map_err(record_error)?;
                if snapshot.pos != entry.pos {
                    return Err(corrupt(format!(
                        "snapshot record restores to {}, the directory says {}",
                        snapshot.pos, entry.pos
                    )));
                }
                snapshots.push((snapshot.pos, snapshot));
            }
            TAG_DRAFTER => {
                let image = DrafterImage::read_from(body, entry.len).map_err(record_error)?;
                if image.pos != entry.pos {
                    return Err(corrupt(format!(
                        "drafter record covers {} positions, the directory says {}",
                        image.pos, entry.pos
                    )));
                }
                drafter = Some(image);
            }
            tag => return Err(corrupt(format!("unknown record tag {tag}"))),
        }
    }

    // Every byte of the file has now passed through the digest. A mismatch is
    // corruption under an intact frame — a rotted plane, or a flipped token id — which
    // is exactly what the structural checks are blind to, since the lengths still add
    // up and the shapes still describe their bytes.
    if r.hash != digest {
        return Err(corrupt(format!(
            "container digest {:016x}, the header says {digest:016x}",
            r.hash
        )));
    }

    let full_kv = full_kv.ok_or_else(|| corrupt("no full-attention record"))?;
    Ok(Segment {
        start,
        parent,
        tokens,
        full_kv,
        snapshots,
        drafter,
    })
}

/// The header as parsed, leaving the reader at the first record body.
struct Parsed {
    start: usize,
    parent: Option<ParentRef>,
    tokens: Vec<u32>,
    directory: Vec<DirectoryEntry>,
    /// The digest the bodies have to hash to, which only a full read checks.
    digest: u64,
}

/// Parse the header and directory off `r`, leaving it positioned at the first
/// record body. `file_len` bounds every count the file declares, and the model
/// plausibility caps bound them again — a file's length is not a limit on what
/// reading it costs, since a sparse one can declare gigabytes it never stored.
fn read_header_into(
    r: &mut impl ContainerRead,
    file_len: u64,
    expected: &CheckpointId,
    tokenization_rules: u32,
) -> Result<Parsed> {
    if file_len < HEADER_MIN {
        return Err(corrupt(format!(
            "{file_len} bytes is shorter than the {HEADER_MIN}-byte header"
        )));
    }
    let mut magic = [0u8; 8];
    read_exact(r, &mut magic)?;
    if &magic != MAGIC {
        return Err(DiskImageError::Binding(format!(
            "magic {magic:?} is not a cache segment"
        )));
    }
    let version = read_u32(r)?;
    if version != CONTAINER_VERSION {
        return Err(DiskImageError::Binding(format!(
            "container version {version}, this build writes {CONTAINER_VERSION}"
        )));
    }
    let hash = read_u64(r)?;
    if hash != expected.hash() {
        return Err(DiskImageError::Binding(format!(
            "checkpoint hash {hash:016x}, the loaded model is {:016x}",
            expected.hash()
        )));
    }
    let stored_len = read_u64(r)?;
    if stored_len != expected.file_len() {
        return Err(DiskImageError::Binding(format!(
            "checkpoint file length {stored_len}, the loaded model is {}",
            expected.file_len()
        )));
    }
    let rules = read_u32(r)?;
    if rules != tokenization_rules {
        return Err(DiskImageError::Binding(format!(
            "tokenization rules version {rules}, this build produces {tokenization_rules}"
        )));
    }

    let digest = r.read_digest().map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            corrupt("file ends inside the digest field")
        } else {
            DiskImageError::Io(e)
        }
    })?;

    let start = read_count(r, "start position")?;
    plausible_pos(start, "start position")?;
    let mut consumed = HEAD_FIXED;

    // The parent reference is present exactly when there is a span before this one.
    // Its absence is not an optional field the reader guesses at: `start` decides it,
    // and `start` is inside the digest.
    let parent = if start > 0 {
        if remaining(file_len, consumed)? < PARENT_REF + 4 + 4 {
            return Err(corrupt("no room for the parent reference"));
        }
        let name = read_u64(r)?;
        let chain = read_u64(r)?;
        consumed += PARENT_REF;
        Some(ParentRef { name, chain })
    } else {
        None
    };

    let token_count = u64::from(read_u32(r)?);
    consumed += 4;
    // Two bounds, both before the ids are allocated: what is left of the file, and
    // what a conversation could plausibly be. The first alone is not enough — a
    // sparse file can claim bytes it does not hold — and a corrupt count must cost
    // an error rather than gigabytes of RAM.
    if token_count > MAX_STORED_TOKENS as u64 {
        return Err(corrupt(format!(
            "{token_count} token ids, past the {MAX_STORED_TOKENS} this build will read"
        )));
    }
    if token_count == 0 {
        return Err(corrupt("a segment covering no token covers nothing"));
    }
    let token_bytes = token_count * 4;
    if token_bytes > remaining(file_len, consumed)? {
        return Err(corrupt(format!(
            "{token_count} token ids do not fit the {} bytes after the header",
            remaining(file_len, consumed)?
        )));
    }
    let mut tokens = Vec::with_capacity(token_count as usize);
    for _ in 0..token_count {
        tokens.push(read_u32(r)?);
    }
    consumed += token_bytes;
    let end = start
        .checked_add(tokens.len())
        .ok_or_else(|| corrupt("the span's end overflows"))?;
    plausible_pos(end, "span end")?;

    if remaining(file_len, consumed)? < 4 {
        return Err(corrupt("no room for the record directory"));
    }
    let record_count = u64::from(read_u32(r)?);
    consumed += 4;
    // One set of rows, one drafter record, and the snapshots between them.
    let max_records = (MAX_STORED_SNAPSHOTS + 2) as u64;
    if record_count > max_records {
        return Err(corrupt(format!(
            "{record_count} records, past the {max_records} this build will read"
        )));
    }
    let directory_bytes = record_count * DIRECTORY_ENTRY;
    if directory_bytes > remaining(file_len, consumed)? {
        return Err(corrupt(format!(
            "{record_count} directory entries do not fit the {} bytes after the token ids",
            remaining(file_len, consumed)?
        )));
    }
    let mut directory = Vec::with_capacity(record_count as usize);
    for _ in 0..record_count {
        let tag = read_u32(r)?;
        let pos = read_count(r, "record position")?;
        let len = read_u64(r)?;
        directory.push(DirectoryEntry { tag, pos, len });
    }
    consumed += directory_bytes;

    // The bodies must account for the rest of the file exactly. A file the writer
    // never finished, or one truncated afterwards, fails here rather than partway
    // through a plane.
    let mut bodies = 0u64;
    for entry in &directory {
        bodies = bodies
            .checked_add(entry.len)
            .ok_or_else(|| corrupt("record lengths overflow"))?;
    }
    let want = remaining(file_len, consumed)?;
    if bodies != want {
        return Err(corrupt(format!(
            "record bodies declare {bodies} bytes, the file has {want} left"
        )));
    }
    check_records(&directory, start, end, tokens.len())?;
    Ok(Parsed {
        start,
        parent,
        tokens,
        directory,
        digest,
    })
}

/// What the record directory must say, given the span the header declared.
///
/// Exactly one full-attention record first, covering the span's rows and no others;
/// then any number of snapshots at ascending positions inside the span; then at most
/// one drafter record, also inside it. Anything else is a file this build did not
/// write, and the reader is not the place to invent semantics for it.
fn check_records(
    directory: &[DirectoryEntry],
    start: usize,
    end: usize,
    span: usize,
) -> Result<()> {
    let Some((first, rest)) = directory.split_first() else {
        return Err(corrupt("no records"));
    };
    if first.tag != TAG_FULL_KV {
        return Err(corrupt("the first record is not the full-attention rows"));
    }
    if first.pos != span {
        return Err(corrupt(format!(
            "the full-attention rows cover {} positions, the span is {span} tokens",
            first.pos
        )));
    }
    let snapshots = rest.iter().take_while(|e| e.tag == TAG_SNAPSHOT).count();
    let trailing = &rest[snapshots..];
    match trailing {
        [] => {}
        [drafter] if drafter.tag == TAG_DRAFTER => {
            if drafter.pos <= start || drafter.pos > end {
                return Err(corrupt(format!(
                    "the drafter rows reach {}, outside the span ({start}, {end}]",
                    drafter.pos
                )));
            }
        }
        tail => {
            let tags: Vec<u32> = tail.iter().map(|e| e.tag).collect();
            return Err(corrupt(format!("unexpected trailing records {tags:?}")));
        }
    }
    let positions: Vec<usize> = rest[..snapshots].iter().map(|e| e.pos).collect();
    if positions.windows(2).any(|w| w[0] >= w[1]) {
        return Err(corrupt(format!(
            "snapshot positions {positions:?} are not ascending"
        )));
    }
    // A snapshot at `start` restores to the state the PARENT's span ends in and is
    // that segment's to hold; one past `end` names a position these rows cannot back.
    if let Some(pos) = positions
        .iter()
        .find(|pos| **pos <= start || **pos > end)
        .copied()
    {
        return Err(corrupt(format!(
            "a snapshot restores to {pos}, outside the span ({start}, {end}]"
        )));
    }
    Ok(())
}

fn header_of(parsed: Parsed, bytes: u64) -> SegmentHeader {
    let Parsed {
        start,
        parent,
        tokens,
        directory,
        ..
    } = parsed;
    SegmentHeader {
        start,
        parent,
        tokens,
        snapshot_positions: directory
            .iter()
            .filter(|e| e.tag == TAG_SNAPSHOT)
            .map(|e| e.pos)
            .collect(),
        drafter_pos: directory
            .iter()
            .find(|e| e.tag == TAG_DRAFTER)
            .map(|e| e.pos),
        bytes,
    }
}

/// Bytes left after `consumed`, refusing a header whose own accounting has already
/// run past the file it is in.
fn remaining(file_len: u64, consumed: u64) -> Result<u64> {
    file_len.checked_sub(consumed).ok_or_else(|| {
        corrupt(format!(
            "the header claims more than the file's {file_len} bytes"
        ))
    })
}

/// A position no conversation could reach is refused before it is used to size or
/// index anything.
fn plausible_pos(pos: usize, what: &str) -> Result<()> {
    if pos > MAX_STORED_POS {
        return Err(corrupt(format!(
            "{what} {pos}, past the {MAX_STORED_POS} this build will read"
        )));
    }
    Ok(())
}

/// Fill `buf` from `r`, reporting a short read as corruption rather than as an
/// I/O fault: the header's own length accounting is checked against the file size
/// first, so running out of bytes here means what is on disk disagrees with what
/// the file claims.
fn read_exact(r: &mut impl Read, buf: &mut [u8]) -> Result<()> {
    r.read_exact(buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            corrupt(format!("file ends {} bytes short in the header", buf.len()))
        } else {
            DiskImageError::Io(e)
        }
    })
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    read_exact(r, &mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    read_exact(r, &mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_count(r: &mut impl Read, what: &str) -> Result<usize> {
    let v = read_u64(r)?;
    usize::try_from(v).map_err(|_| corrupt(format!("{what} {v} does not fit a usize")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::HostLayerSnapshot;

    /// The shapes every fixture is built at: heads, positions, head dim and window
    /// all differ, so a transposed field or a plane framed in the wrong slot cannot
    /// pass on byte counts alone.
    const N_KV: usize = 3;
    const HEAD_DIM: usize = 5;
    const WINDOW: usize = 7;
    /// A span's length, and the position a root segment ends at.
    const SPAN: usize = 23;

    /// Full attention iff `il % 4 == 0`, as the shipped model interleaves them.
    const KINDS: [bool; 6] = [true, false, false, false, true, false];

    fn checkpoint() -> CheckpointId {
        CheckpointId::from_parts(0x0123_4567_89ab_cdef, 68_719_476_736)
    }

    const RULES: u32 = 1;

    /// Bytes keyed to their own offset and a seed, so a plane read back from the
    /// wrong place fails on content rather than length.
    fn pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len).map(|i| (i as u8) ^ seed ^ 0xa5).collect()
    }

    fn full_kv(rows: usize) -> HostFullKv {
        let plane = N_KV * rows * HEAD_DIM * size_of::<half::f16>();
        let planes = KINDS
            .iter()
            .filter(|full| **full)
            .enumerate()
            .map(|(il, _)| (pattern(plane, il as u8), pattern(plane, 0x40 + il as u8)))
            .collect();
        HostFullKv::new(rows, N_KV, HEAD_DIM, planes).unwrap()
    }

    fn snapshot(pos: usize) -> HostSnapshot {
        let ring = N_KV * WINDOW * HEAD_DIM * size_of::<half::f16>();
        let layers = KINDS
            .iter()
            .enumerate()
            .map(|(il, full)| {
                if *full {
                    HostLayerSnapshot::Full
                } else {
                    HostLayerSnapshot::Swa {
                        k: pattern(ring, (pos + il) as u8),
                        v: pattern(ring, 0x40u8.wrapping_add((pos + il) as u8)),
                        shape: (N_KV, WINDOW, HEAD_DIM),
                        window: WINDOW,
                    }
                }
            })
            .collect();
        HostSnapshot::new(pos, layers).unwrap()
    }

    fn drafter(pos: usize) -> DrafterImage {
        let plane = N_KV * pos * HEAD_DIM * size_of::<f32>();
        let planes = (0..2u8)
            .map(|il| (pattern(plane, 0x80 + il), pattern(plane, 0xc0 + il)))
            .collect();
        DrafterImage::new_dflash(pos, N_KV, HEAD_DIM, planes).unwrap()
    }

    fn tokens(n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| i * 7 + 3).collect()
    }

    /// A directory of this test's own, removed when the guard drops so a failing
    /// assertion does not leave files behind for the next run to trip over.
    struct Dir(PathBuf);

    impl Dir {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("xwen_disk_cache_{}_{label}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn file(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn assert_snapshots_equal(got: &HostSnapshot, want: &HostSnapshot, label: &str) {
        assert_eq!(got.pos, want.pos, "{label}: position");
        let mut bytes = (Vec::new(), Vec::new());
        got.write_to(&mut bytes.0).unwrap();
        want.write_to(&mut bytes.1).unwrap();
        assert_eq!(bytes.0, bytes.1, "{label}: planes");
    }

    fn bytes_of(rows: &HostFullKv) -> Vec<u8> {
        let mut out = Vec::new();
        rows.write_to(&mut out).unwrap();
        out
    }

    /// A root segment survives the round trip: every record comes back
    /// byte-identical, and the header-only parse — the one a startup scan runs over
    /// every file — reports exactly what the full parse found.
    #[test]
    fn a_root_segment_round_trips_and_the_header_agrees() {
        let dir = Dir::new("round_trip");
        let path = dir.file("root.lkv");
        let ids = tokens(SPAN);
        let rows = full_kv(SPAN);
        let early = snapshot(9);
        let tip = snapshot(SPAN);
        let draft = drafter(SPAN);

        let bytes = write_segment(
            &path,
            &checkpoint(),
            RULES,
            0,
            None,
            &ids,
            &rows,
            &[(9, &early), (SPAN, &tip)],
            Some(&draft),
        )
        .unwrap();
        assert_eq!(bytes, std::fs::metadata(&path).unwrap().len());

        let header = read_segment_header(&path, &checkpoint(), RULES).unwrap();
        assert_eq!(header.start, 0);
        assert_eq!(header.end(), SPAN);
        assert_eq!(header.parent, None);
        assert_eq!(header.tokens, ids);
        assert_eq!(header.snapshot_positions, vec![9, SPAN]);
        assert_eq!(header.drafter_pos, Some(SPAN));
        assert_eq!(header.bytes, bytes);

        let segment = read_segment(&path, &checkpoint(), RULES).unwrap();
        assert_eq!(segment.start, 0);
        assert_eq!(segment.parent, None);
        assert_eq!(segment.tokens, header.tokens);
        assert_eq!(segment.full_kv.pos, SPAN);
        assert_eq!(
            segment
                .snapshots
                .iter()
                .map(|(p, _)| *p)
                .collect::<Vec<_>>(),
            header.snapshot_positions
        );
        assert_eq!(bytes_of(&segment.full_kv), bytes_of(&rows));
        assert_snapshots_equal(&segment.snapshots[0].1, &early, "early snapshot");
        assert_snapshots_equal(&segment.snapshots[1].1, &tip, "tip snapshot");

        let mut want_draft = Vec::new();
        draft.write_to(&mut want_draft).unwrap();
        let mut got_draft = Vec::new();
        segment.drafter.unwrap().write_to(&mut got_draft).unwrap();
        assert_eq!(got_draft, want_draft, "drafter planes");
    }

    /// A tail segment round trips with its parent reference and its absolute
    /// positions — and with no snapshot at all, which is ordinary: a tail written
    /// after a fork resumes from a position its ancestors hold.
    #[test]
    fn a_tail_segment_round_trips_with_its_parent_and_without_snapshots() {
        let dir = Dir::new("tail");
        let history = tokens(SPAN + 8);
        let parent = chain_id(&history[..SPAN]).as_parent();
        let path = dir.file(&chain_id(&history).file_name());

        write_segment(
            &path,
            &checkpoint(),
            RULES,
            SPAN,
            Some(parent),
            &history[SPAN..],
            &full_kv(8),
            &[],
            None,
        )
        .unwrap();

        let header = read_segment_header(&path, &checkpoint(), RULES).unwrap();
        assert_eq!(header.start, SPAN);
        assert_eq!(header.end(), SPAN + 8);
        assert_eq!(header.parent, Some(parent));
        assert_eq!(header.tokens, history[SPAN..]);
        assert!(header.snapshot_positions.is_empty());
        assert_eq!(header.drafter_pos, None);

        let segment = read_segment(&path, &checkpoint(), RULES).unwrap();
        assert_eq!(segment.start, SPAN);
        assert_eq!(segment.parent, Some(parent));
        assert!(segment.snapshots.is_empty());
        assert!(segment.drafter.is_none());
        assert_eq!(segment.full_kv.pos, 8);

        // The name a child of this segment would look its parent up by is this
        // segment's own, computed from the cumulative history rather than from
        // anything the file says.
        assert_eq!(
            path.file_name().unwrap().to_str().unwrap(),
            segment_file_name(chain_id(&history).name)
        );
        assert_eq!(segment_name_of(&path), Some(chain_id(&history).name));
    }

    /// A segment's identity is over the CUMULATIVE history, which is what makes a
    /// split stable: cutting a span leaves a tail whose history — and so whose name
    /// — is unchanged.
    #[test]
    fn identities_are_over_the_cumulative_history_and_the_two_hashes_are_independent() {
        let history = tokens(64);
        assert_eq!(chain_id(&history), chain_id(&tokens(64)));
        assert_ne!(chain_id(&history), chain_id(&tokens(65)));
        // The name and the chain witness are two hashes of the same tokens, and a
        // child stores both: they must not be the same number.
        assert_ne!(chain_id(&history).name, chain_id(&history).chain);
        // The count is part of both, so a history and its extension by a zero token
        // cannot collide.
        assert_ne!(chain_id(&[0]).name, chain_id(&[0, 0]).name);
        assert_ne!(chain_id(&[0]).chain, chain_id(&[0, 0]).chain);

        // Splitting the history at k names the base by the prefix and leaves the
        // tail's identity — the whole history — untouched.
        let split = chain_id(&history[..40]);
        assert_eq!(split.name, chain_id(&tokens(64)[..40]).name);
        assert_eq!(chain_id(&history).name, chain_id(&tokens(64)).name);

        // A name round trips through the file name it becomes, and only in the
        // canonical spelling.
        let name = chain_id(&history).name;
        assert_eq!(
            segment_name_of(Path::new(&segment_file_name(name))),
            Some(name)
        );
        assert_eq!(segment_name_of(Path::new("0x12.lkv")), None);
        assert_eq!(segment_name_of(Path::new("12.lkv")), None);
        assert_eq!(segment_name_of(Path::new("image.lkv")), None);
        assert_eq!(
            segment_name_of(Path::new("00000000000000AB.lkv")),
            None,
            "one spelling of a name, so one file can hold it"
        );
    }

    /// A single flipped bit inside a plane is caught, and it is the container digest
    /// that catches it: every length still adds up, every shape still describes its
    /// bytes, and the record's own constructor is perfectly happy. APFS checksums
    /// metadata but not file data, so without the digest a rotted plane would be
    /// uploaded as a conversation's keys.
    ///
    /// The header-only parse still accepts the file — it reads no bodies, so it has
    /// nothing to check them against — which is why a hydration re-reads and
    /// re-validates rather than trusting the scan.
    #[test]
    fn a_flipped_plane_byte_is_caught_by_the_container_digest() {
        let dir = Dir::new("digest");
        let path = dir.file("root.lkv");
        let tip = snapshot(SPAN);
        write_segment(
            &path,
            &checkpoint(),
            RULES,
            0,
            None,
            &tokens(SPAN),
            &full_kv(SPAN),
            &[(SPAN, &tip)],
            Some(&drafter(SPAN)),
        )
        .unwrap();

        let good = std::fs::read(&path).unwrap();
        // Well inside the last record's planes, so nothing structural moves.
        let mut rotted = good.clone();
        let at = rotted.len() - 32;
        rotted[at] ^= 0x01;
        std::fs::write(&path, &rotted).unwrap();

        assert!(
            read_segment_header(&path, &checkpoint(), RULES).is_ok(),
            "the header parse reads no bodies, so it cannot see this"
        );
        assert_corrupt(
            read_segment(&path, &checkpoint(), RULES),
            "flipped plane byte",
        );

        // A digest field that disagrees with intact bodies fails the same way: the
        // check is an equality, not a heuristic.
        let mut lying = good.clone();
        let digest = DIGEST_OFFSET as usize;
        lying[digest] ^= 0xff;
        std::fs::write(&path, &lying).unwrap();
        assert_corrupt(read_segment(&path, &checkpoint(), RULES), "wrong digest");

        // And the untouched file still reads.
        std::fs::write(&path, &good).unwrap();
        assert!(read_segment(&path, &checkpoint(), RULES).is_ok());
    }

    /// The digest covers the token ids and the parent reference too, and that is not
    /// a nicety: the hydration guard compares the STORED history against the arriving
    /// prompt, so a bit flip in a token id can leave a history that still matches a
    /// prompt — a different conversation's, past the flip — and pass a guard that has
    /// nothing else to go on. A flipped parent name would chain a span onto the wrong
    /// history, which the link check catches only because the ref is intact enough to
    /// be compared at all.
    #[test]
    fn a_flipped_token_id_or_parent_reference_is_caught_by_the_digest() {
        let dir = Dir::new("digest_tokens");
        let history = tokens(SPAN + 8);
        let path = dir.file("tail.lkv");
        let tip = snapshot(SPAN + 8);
        write_segment(
            &path,
            &checkpoint(),
            RULES,
            SPAN,
            Some(chain_id(&history[..SPAN]).as_parent()),
            &history[SPAN..],
            &full_kv(8),
            &[(SPAN + 8, &tip)],
            None,
        )
        .unwrap();

        let good = std::fs::read(&path).unwrap();
        let tokens_at = (HEAD_FIXED + PARENT_REF + 4) as usize;

        // The first span token id, one bit different.
        let mut flipped = good.clone();
        flipped[tokens_at] ^= 0x01;
        std::fs::write(&path, &flipped).unwrap();
        assert_corrupt(
            read_segment(&path, &checkpoint(), RULES),
            "flipped token id",
        );

        // The parent reference, likewise.
        let mut reparented = good.clone();
        reparented[HEAD_FIXED as usize] ^= 0x01;
        std::fs::write(&path, &reparented).unwrap();
        assert_corrupt(
            read_segment(&path, &checkpoint(), RULES),
            "flipped parent name",
        );

        // The record directory is covered as well, and every claim it makes is checked
        // against the record it describes: a position it lies about is caught whether
        // the lie is inside the span or outside it.
        let mut moved = good.clone();
        let directory_at = tokens_at + 4 * 8 + 4;
        moved[directory_at + 4] ^= 0x02;
        std::fs::write(&path, &moved).unwrap();
        assert_corrupt(
            read_segment(&path, &checkpoint(), RULES),
            "patched directory",
        );

        // And the digest field itself is hashed as zeros, so the file it lives in can
        // cover it: the untouched segment still reads.
        std::fs::write(&path, &good).unwrap();
        assert!(read_segment(&path, &checkpoint(), RULES).is_ok());
    }

    /// A read that fails for a reason other than running out of bytes is an I/O fault,
    /// not a verdict about the segment — and the difference decides whether a
    /// multi-gigabyte file is deleted. A body that ends early IS a verdict: the file is
    /// shorter than it says it is.
    #[test]
    fn a_transient_read_fault_is_not_corruption() {
        /// A reader that hands over `ok` bytes and then fails the way a disk does.
        struct Flaky {
            bytes: Vec<u8>,
            at: usize,
            kind: std::io::ErrorKind,
        }

        impl Read for Flaky {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if self.at >= self.bytes.len() {
                    return Err(std::io::Error::new(self.kind, "the disk said no"));
                }
                let take = buf.len().min(self.bytes.len() - self.at);
                buf[..take].copy_from_slice(&self.bytes[self.at..self.at + take]);
                self.at += take;
                Ok(take)
            }
        }

        // A well-formed body, cut off at the point its first plane begins.
        let rows = full_kv(SPAN);
        let body = bytes_of(&rows);
        let head = 4 * size_of::<u64>() + size_of::<u64>();

        let failure = |kind: std::io::ErrorKind| {
            let flaky = Flaky {
                bytes: body[..head].to_vec(),
                at: 0,
                kind,
            };
            let Err(e) = HostFullKv::read_from(flaky, body.len() as u64) else {
                panic!("a body that stops has to fail");
            };
            record_error(e)
        };

        // The disk failing mid-plane says nothing about the bytes: the segment is kept.
        let transient = failure(std::io::ErrorKind::PermissionDenied);
        assert_eq!(
            transient.class(),
            "io",
            "a transient fault must not condemn the file: {transient}"
        );
        // Running out of bytes is the file disagreeing with itself.
        let short = failure(std::io::ErrorKind::UnexpectedEof);
        assert_eq!(short.class(), "corrupt", "{short}");
    }

    /// A count no checkpoint could have produced is refused before it sizes
    /// anything, whatever the file's length claims — a sparse file can declare
    /// gigabytes it never stored, and an allocation that fails aborts the process
    /// where a rejected file only costs a re-prefill.
    #[test]
    fn implausible_counts_are_refused_before_they_allocate() {
        let dir = Dir::new("caps");

        let head = |start: u64| {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(MAGIC);
            bytes.extend_from_slice(&CONTAINER_VERSION.to_le_bytes());
            bytes.extend_from_slice(&checkpoint().hash().to_le_bytes());
            bytes.extend_from_slice(&checkpoint().file_len().to_le_bytes());
            bytes.extend_from_slice(&RULES.to_le_bytes());
            bytes.extend_from_slice(&0u64.to_le_bytes());
            bytes.extend_from_slice(&start.to_le_bytes());
            bytes
        };

        // A header claiming more token ids than any conversation could hold, framed
        // so the file's own length agrees with it.
        let path = dir.file("tokens.lkv");
        let mut bytes = head(0);
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        // Sparse: the length says the ids are all there, the disk holds none of them.
        let sparse = File::options().write(true).open(&path).unwrap();
        sparse
            .set_len(HEAD_FIXED + 4 + 4 * u64::from(u32::MAX))
            .unwrap();
        drop(sparse);
        assert_corrupt(
            read_segment_header(&path, &checkpoint(), RULES),
            "implausible token count",
        );

        // The same for the record directory.
        let path = dir.file("records.lkv");
        let mut bytes = head(0);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&7u32.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let sparse = File::options().write(true).open(&path).unwrap();
        sparse
            .set_len(HEAD_FIXED + 4 + 4 + 4 + DIRECTORY_ENTRY * u64::from(u32::MAX))
            .unwrap();
        drop(sparse);
        assert_corrupt(
            read_segment_header(&path, &checkpoint(), RULES),
            "record flood",
        );

        // And a start position no conversation could reach, before it is added to
        // anything.
        let path = dir.file("start.lkv");
        let mut bytes = head(u64::MAX);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&7u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        assert_corrupt(
            read_segment_header(&path, &checkpoint(), RULES),
            "implausible start",
        );
    }

    /// Every way a file can fail to bind to this server, and every way its own
    /// bytes can fail to hold together, lands in the right class: a `Binding`
    /// rejection is an intact file for a model, a tokenizer or a container version
    /// that is no longer in use, a `Corrupt` one means the bytes disagree with
    /// themselves. The tier deletes on both and logs them differently, so the
    /// distinction has to survive.
    #[test]
    fn rejections_are_classified() {
        let dir = Dir::new("rejections");
        let path = dir.file("root.lkv");
        let tip = snapshot(SPAN);
        let early = snapshot(9);
        write_segment(
            &path,
            &checkpoint(),
            RULES,
            0,
            None,
            &tokens(SPAN),
            &full_kv(SPAN),
            &[(9, &early), (SPAN, &tip)],
            Some(&drafter(SPAN)),
        )
        .unwrap();
        let good = std::fs::read(&path).unwrap();

        // Written with one binding, read with another.
        let wrong_hash = CheckpointId::from_parts(0xdead_beef_dead_beef, checkpoint().file_len());
        assert_binding(read_segment_header(&path, &wrong_hash, RULES), "hash");
        assert_binding(read_segment(&path, &wrong_hash, RULES), "hash");

        let wrong_len = CheckpointId::from_parts(checkpoint().hash(), 1234);
        assert_binding(read_segment_header(&path, &wrong_len, RULES), "file length");
        assert_binding(read_segment(&path, &wrong_len, RULES), "file length");

        assert_binding(
            read_segment_header(&path, &checkpoint(), RULES + 1),
            "rules",
        );
        assert_binding(read_segment(&path, &checkpoint(), RULES + 1), "rules");

        // Patched files. Each is written under its own name so a failure names the
        // case that produced it.
        let patched = |label: &str, edit: &dyn Fn(&mut Vec<u8>)| -> PathBuf {
            let mut bytes = good.clone();
            edit(&mut bytes);
            let p = dir.file(&format!("{label}.lkv"));
            std::fs::write(&p, &bytes).unwrap();
            p
        };

        let flipped = patched("magic", &|b| b[0] ^= 0xff);
        assert_binding(read_segment_header(&flipped, &checkpoint(), RULES), "magic");
        assert_binding(read_segment(&flipped, &checkpoint(), RULES), "magic");

        // The previous arc's flat per-conversation container, which this build reads
        // as a binding rejection and the tier therefore deletes: there is no
        // migration, and a v1 file costs a re-prefill to replace.
        let v1 = patched("v1", &|b| b[8..12].copy_from_slice(&1u32.to_le_bytes()));
        assert_binding(read_segment_header(&v1, &checkpoint(), RULES), "version 1");
        assert_binding(read_segment(&v1, &checkpoint(), RULES), "version 1");

        let bumped = patched("version", &|b| {
            b[8..12].copy_from_slice(&99u32.to_le_bytes())
        });
        assert_binding(
            read_segment_header(&bumped, &checkpoint(), RULES),
            "container version",
        );

        // Truncation, at three depths: inside the fixed header, inside the record
        // directory, and inside the planes — the last is the case a header-only
        // scan has to catch from the file size alone, since it reads none of them.
        for cut in [4usize, HEAD_FIXED as usize + 8, good.len() - 16] {
            let short = patched(&format!("cut{cut}"), &|b| b.truncate(cut));
            assert_corrupt(
                read_segment_header(&short, &checkpoint(), RULES),
                "truncated",
            );
            assert_corrupt(read_segment(&short, &checkpoint(), RULES), "truncated");
        }

        // An empty file has neither a binding nor a body.
        let empty = patched("empty", &|b| b.clear());
        assert_corrupt(read_segment_header(&empty, &checkpoint(), RULES), "empty");

        let tokens_at = (HEAD_FIXED + 4) as usize;
        let directory_at = tokens_at + 4 * SPAN + 4;

        // A record length that no longer adds up to the file's size: the directory
        // and the file disagree, which the header check catches before a plane is
        // touched.
        let lied = patched("length_lie", &|b| {
            let at = directory_at + 4 + 8;
            b[at..at + 8].copy_from_slice(&99u64.to_le_bytes());
        });
        assert_corrupt(
            read_segment_header(&lied, &checkpoint(), RULES),
            "length lie",
        );

        // A plane-length lie INSIDE a record body, framed so the container's own
        // accounting still balances: the first record's declared position is raised,
        // so its planes are short for the shape it claims. Nothing but the record's
        // validating constructor can catch this, which is why deserialization has no
        // trusted path around it.
        // Four records: the rows, two snapshots, the drafter.
        let body_at = directory_at + 4 * DIRECTORY_ENTRY as usize;
        let shape_lie = patched("shape_lie", &|b| {
            b[body_at..body_at + 8].copy_from_slice(&(SPAN as u64 + 1).to_le_bytes());
        });
        assert_corrupt(read_segment(&shape_lie, &checkpoint(), RULES), "shape lie");
        // The header does not read bodies, so it still parses — the full read is
        // where this file dies.
        assert!(read_segment_header(&shape_lie, &checkpoint(), RULES).is_ok());

        // The rows must cover the span exactly, and the directory says how many they
        // cover: a file whose rows and span disagree is refused from the header
        // alone, since composing a chain out of it would leave a hole.
        let short_rows = patched("short_rows", &|b| {
            let at = directory_at + 4;
            b[at..at + 8].copy_from_slice(&(SPAN as u64 - 2).to_le_bytes());
        });
        assert_corrupt(
            read_segment_header(&short_rows, &checkpoint(), RULES),
            "rows shorter than the span",
        );

        // A tag this build does not write, and a directory whose records are out of
        // order: both are files another version produced, and guessing at them is
        // how one conversation ends up restoring another's rings.
        let bad_tag = patched("bad_tag", &|b| {
            b[directory_at..directory_at + 4].copy_from_slice(&77u32.to_le_bytes());
        });
        assert_corrupt(
            read_segment_header(&bad_tag, &checkpoint(), RULES),
            "unknown tag",
        );

        let reordered = patched("reordered", &|b| {
            let a = directory_at;
            let c = directory_at + DIRECTORY_ENTRY as usize;
            let (mut first, mut second) = ([0u8; 20], [0u8; 20]);
            first.copy_from_slice(&b[a..a + 20]);
            second.copy_from_slice(&b[c..c + 20]);
            b[a..a + 20].copy_from_slice(&second);
            b[c..c + 20].copy_from_slice(&first);
        });
        assert_corrupt(
            read_segment_header(&reordered, &checkpoint(), RULES),
            "out of order",
        );

        // A file that is not there at all is an I/O fact, not a verdict on any
        // segment's contents.
        let missing = dir.file("nothing.lkv");
        assert!(matches!(
            read_segment_header(&missing, &checkpoint(), RULES),
            Err(DiskImageError::Io(_))
        ));
    }

    fn assert_binding<T>(got: Result<T>, label: &str) {
        match got {
            Err(DiskImageError::Binding(_)) => {}
            Err(other) => panic!("{label}: expected a binding rejection, got {other}"),
            Ok(_) => panic!("{label}: expected a binding rejection, the file was accepted"),
        }
    }

    fn assert_corrupt<T>(got: Result<T>, label: &str) {
        match got {
            Err(DiskImageError::Corrupt(_)) => {}
            Err(other) => panic!("{label}: expected a corruption rejection, got {other}"),
            Ok(_) => panic!("{label}: expected a corruption rejection, the file was accepted"),
        }
    }

    /// A failed write leaves nothing under the real name, and never a half-written
    /// file: the bytes land in a sibling that is renamed only once it is complete.
    /// A reader that finds the name at all must find a whole segment — which is what
    /// makes a split's rewrite of a tail under its own name an atomic replacement.
    #[test]
    fn a_failed_write_leaves_the_target_alone() {
        let dir = Dir::new("atomic");
        let path = dir.file("root.lkv");
        let tip = snapshot(SPAN);
        write_segment(
            &path,
            &checkpoint(),
            RULES,
            0,
            None,
            &tokens(SPAN),
            &full_kv(SPAN),
            &[(SPAN, &tip)],
            None,
        )
        .unwrap();
        let before = std::fs::read(&path).unwrap();

        // A write into a directory that does not exist fails at file creation.
        let nested = dir.file("missing").join("root.lkv");
        assert!(
            write_segment(
                &nested,
                &checkpoint(),
                RULES,
                0,
                None,
                &tokens(SPAN),
                &full_kv(SPAN),
                &[(SPAN, &tip)],
                None,
            )
            .is_err()
        );
        assert!(!nested.exists());

        // Rewriting the same name with a different partition of the same history —
        // which is exactly what a split does to the segment it cuts — replaces it
        // wholesale.
        let shorter = snapshot(SPAN);
        write_segment(
            &path,
            &checkpoint(),
            RULES,
            SPAN - 4,
            Some(chain_id(&tokens(SPAN)[..SPAN - 4]).as_parent()),
            &tokens(SPAN)[SPAN - 4..],
            &full_kv(4),
            &[(SPAN, &shorter)],
            None,
        )
        .unwrap();
        let after = std::fs::read(&path).unwrap();
        assert_ne!(before, after);
        assert_eq!(
            read_segment_header(&path, &checkpoint(), RULES)
                .unwrap()
                .start,
            SPAN - 4
        );
        assert!(
            !tmp_sibling(&path).exists(),
            "the sibling must not survive a successful write"
        );
    }

    /// Both ends refuse a segment whose stored positions fall outside the span it
    /// covers — the writer because a slot that produced one is a bug, the reader
    /// because bytes on disk are untrusted whatever wrote them.
    ///
    /// A snapshot at `start` is the interesting case: it restores to the state the
    /// PARENT's span ends in, so it belongs in the parent. Storing it here would
    /// duplicate it in every child of a branch point and make the boundary's
    /// resumability depend on which child happened to be read.
    #[test]
    fn positions_outside_the_span_are_refused() {
        let dir = Dir::new("span_bounds");
        let path = dir.file("tail.lkv");
        let history = tokens(SPAN + 8);
        let parent = chain_id(&history[..SPAN]).as_parent();
        let write = |snapshots: &[(usize, &HostSnapshot)], drafter: Option<&DrafterImage>| {
            write_segment(
                &path,
                &checkpoint(),
                RULES,
                SPAN,
                Some(parent),
                &history[SPAN..],
                &full_kv(8),
                snapshots,
                drafter,
            )
        };

        let at_start = snapshot(SPAN);
        assert!(
            write(&[(SPAN, &at_start)], None).is_err(),
            "a snapshot at the span's start is the parent's"
        );
        let past_end = snapshot(SPAN + 9);
        assert!(write(&[(SPAN + 9, &past_end)], None).is_err());
        let inside = snapshot(SPAN + 4);
        let tip = snapshot(SPAN + 8);
        assert!(
            write(&[(SPAN + 8, &tip), (SPAN + 4, &inside)], None).is_err(),
            "and snapshots out of order"
        );
        assert!(
            write(&[(SPAN + 4, &inside)], Some(&drafter(SPAN))).is_err(),
            "a drafter at the span's start covers rows this segment does not hold"
        );
        // Inside the span, ascending: the file this build writes.
        write(&[(SPAN + 4, &inside), (SPAN + 8, &tip)], None).unwrap();

        // A root cannot claim a parent, and a tail cannot go without one: the
        // presence of the reference is decided by `start`, not chosen.
        assert!(
            write_segment(
                &path,
                &checkpoint(),
                RULES,
                0,
                Some(parent),
                &history[..SPAN],
                &full_kv(SPAN),
                &[],
                None,
            )
            .is_err()
        );
        assert!(
            write_segment(
                &path,
                &checkpoint(),
                RULES,
                SPAN,
                None,
                &history[SPAN..],
                &full_kv(8),
                &[],
                None,
            )
            .is_err()
        );

        // Rows that do not cover the span exactly would leave a hole in the composed
        // chain, so neither end accepts them.
        assert!(
            write_segment(
                &path,
                &checkpoint(),
                RULES,
                SPAN,
                Some(parent),
                &history[SPAN..],
                &full_kv(7),
                &[],
                None,
            )
            .is_err()
        );

        // The same file as it would arrive from another build: written coherently,
        // then patched to move the snapshot past the span's end in both the directory
        // and the record body, so the container's own accounting still balances.
        write_segment(
            &path,
            &checkpoint(),
            RULES,
            SPAN,
            Some(parent),
            &history[SPAN..],
            &full_kv(8),
            &[(SPAN + 4, &inside)],
            None,
        )
        .unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let directory_at = (HEAD_FIXED + PARENT_REF + 4) as usize + 4 * 8 + 4;
        let rows_len = u64::from_le_bytes(
            bytes[directory_at + 12..directory_at + 20]
                .try_into()
                .unwrap(),
        );
        let entry = directory_at + DIRECTORY_ENTRY as usize;
        let body = directory_at + 2 * DIRECTORY_ENTRY as usize + rows_len as usize;
        let past = (SPAN + 9) as u64;
        bytes[entry + 4..entry + 12].copy_from_slice(&past.to_le_bytes());
        bytes[body..body + 8].copy_from_slice(&past.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        assert_corrupt(
            read_segment_header(&path, &checkpoint(), RULES),
            "snapshot past the span",
        );
    }
}
