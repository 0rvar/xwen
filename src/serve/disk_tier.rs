//! The disk tier under the prefix cache: what is stored, who writes it, and what
//! gets deleted.
//!
//! Three tiers serve a conversation, strictly layered. The GPU cache holds one.
//! `SlotManager` holds the rest as host-RAM images. This module is the third: the
//! same state, written to `<cache_dir>/kv/<checkpoint>/*.lkv` so a restart resumes
//! a warm conversation instead of paying its prefill again — a minute of GPU time
//! for a 20k-token system prompt against about a second of NVMe.
//!
//! What is stored is a TREE of prefix segments, not a file per conversation. A
//! segment covers a token span and names the segment covering the span before it, so
//! two conversations behind the same system prompt share one copy of it, a
//! conversation that grew by a turn writes a tail instead of rewriting a gigabyte,
//! and a conversation that forks from another splits the segment they diverge inside
//! rather than duplicating everything up to that point. A hydration reads a chain
//! root-first and composes it into exactly the image a slot paged out in this process
//! would hold.
//!
//! Two properties shape everything here. The tier is PERF-ONLY: every way a file
//! can be absent, stale or damaged ends in an ordinary cache miss, which is why
//! nothing is read without going through the validating constructors in
//! [`super::disk_cache`] and why no failure is ever propagated into a job. And the
//! engine thread does no disk I/O for it, with exactly one decided exception: the
//! hydration read, which happens inside the job it serves because it *is* that
//! job's prefill, ~1 s of reading in place of ~60 s of compute. Writing, splitting,
//! deleting and scanning all belong to the `disk-cache` thread.

use std::cell::Cell;
use std::collections::HashMap;
use std::fs::{File, FileTimes};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, SyncSender, channel, sync_channel};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant, SystemTime};

use candle_core::Device;

use crate::chat::TOKENIZATION_RULES_VERSION;
use crate::dflash::DrafterImage;
use crate::gguf::{self, CheckpointId};
use crate::kv_cache::{HostFullKv, HostSnapshot, MAX_STORED_SNAPSHOTS};

use super::config::ServeSettings;
use super::disk_cache::{self, ChainId, DiskImage, ParentRef, Segment, chain_id};
use super::log::{DiskEvictReason, DiskSegmentRole, ServeLog, ServeLogger};

/// The writer thread's name, for a stack trace that has to say whose it is.
const WRITER_THREAD: &str = "disk-cache";

/// Directory under the cache dir that holds cache segments, one subdirectory per
/// checkpoint. Named so the cache dir can hold other things later without a
/// migration.
const KV_SUBDIR: &str = "kv";

/// How much longer a conversation has to have grown before another segment of it is
/// written.
///
/// Every page-out is a candidate write, and writing a segment to extend a stored
/// conversation by one turn is a poor trade: what is already on disk still resumes
/// that conversation, and the only cost of skipping is that a restart resumes at the
/// stored chain's deepest snapshot instead of a newer one — tokens of prefill, not a
/// miss. Two clients talking in turn would otherwise write on every request. The
/// floor is measured against what the store could ACTUALLY resume the conversation
/// at, not against the deepest boundary: a boundary with no snapshot behind it
/// resumes nothing.
const DISK_MIN_GROWTH: usize = 256;

/// What one chain's files may add up to before a hydration refuses to read them.
///
/// A bound on ALLOCATION, not on taste: every segment's planes are read into host RAM,
/// and an allocation that fails aborts the process where a refused chain costs a
/// re-prefill. The peak is above this figure, not at it — the composed image is
/// allocated while the chain is still held, so a chain of C bytes peaks near 2C before
/// `HostFullKv::concat` releases the spans it has copied. This cap is therefore set so
/// that TWICE it still fits beside seventy gigabytes of resident weights on the machine
/// this serves.
///
/// The figures, per `hub::Model::kv_bytes_per_token` / `snapshot_bytes`: at the trained
/// context of 262144 positions the full-attention rows are 5 GiB on the 35B-A3B
/// (20 KiB/token) and 16 GiB on the 27B (64 KiB/token), and each retained snapshot is a
/// fixed 62.8 MiB or 149.6 MiB of DeltaNet recurrent state. At the default four
/// snapshots the worst case is a 27B conversation filling the trained context, about
/// 16.6 GiB — under the cap, and far under what the server actually sees.
///
/// A 27B conversation that both fills the trained context AND retains dozens of
/// snapshots can exceed it and be refused. That is the cap working as designed rather
/// than a figure to raise: refusing costs that one conversation a re-prefill, while an
/// allocation failure at twice the chain size takes the process down with it.
const MAX_CHAIN_BYTES: u64 = 24 << 30;

/// How many segments one chain may hold.
///
/// A bound on the scan's tree walk and on a hydration's read, and the reason a
/// growing conversation cannot append tails forever: at the cap the writer rewrites
/// the chain's last segment as a longer one instead of chaining another onto it, so
/// the depth stays put and the cost stays proportional to that segment's span.
/// Segments found deeper than this at scan are treated as unreachable and deleted —
/// a chain this build cannot compose is a chain nothing can hydrate.
const MAX_CHAIN_DEPTH: usize = 64;

/// Whether a rejection is a verdict about a file's CONTENTS, and so a reason to
/// delete it.
///
/// An allowlist rather than "anything but I/O", so that it fails closed: a rejection
/// class added later protects the file until someone decides otherwise, which is the
/// safe default for the one operation here that cannot be undone. The three that
/// delete are a file bound to another checkpoint, tokenizer or container version,
/// bytes that do not hold together, and a segment whose stored history is not the
/// conversation that found it. A file that could not be READ says nothing about what
/// is in it — a permission change, a disk that hiccuped, a rename racing the read —
/// and deleting on that would throw away good segments over a transient fault.
fn deletable(class: &str) -> bool {
    matches!(class, "binding" | "corrupt" | "collided")
}

/// One segment of the tree, as its header describes it. Token spans are kept
/// resident on purpose: matching is a longest-common-prefix over the cumulative
/// histories they concatenate into, and at 4 bytes a token a 20k-token conversation
/// costs 80 KB to keep in mind. Dozens of them are nothing next to the planes they
/// describe.
struct IndexedSegment {
    /// The segment's identity, which is also its file name: FNV over the whole
    /// history `[0, end)`.
    name: u64,
    /// First position the span covers.
    start: usize,
    /// The segment covering the span before it, absent iff `start == 0`.
    parent: Option<ParentRef>,
    /// Token ids of `[start, end)`.
    span: Vec<u32>,
    /// Absolute positions this segment's snapshots restore to, ascending.
    snapshots: Vec<usize>,
}

impl IndexedSegment {
    fn end(&self) -> usize {
        self.start + self.span.len()
    }
}

/// What may still be done with a stored file this run.
///
/// The three are kept apart because they answer different questions, and collapsing
/// any two of them deletes something that should have been kept. Nothing here is about
/// a file's PLACE in the tree: a segment is its parent's child in every standing, so
/// the parent stays protected from eviction whatever happened to the child.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Standing {
    /// Matched, hydrated, extended — the ordinary state.
    Ready,
    /// Kept and counted, but nothing is built on it: a read of it, or of something in
    /// front of it, went badly this run. The size budget may still reclaim it, as it
    /// may reclaim any file.
    Demoted,
    /// Waiting on a file whose header could not be READ. Built on by nothing and
    /// deleted by nothing, including the size budget: an I/O fault on one file is not a
    /// verdict about another, and this one's only fault is where it sits.
    Held,
}

/// What one stored file is, from its header alone.
struct Entry {
    /// The segment it holds, absent for a file this server cannot read: another
    /// checkpoint's, or one in our own directory that is not named by our rule. Those
    /// are counted by the size budget and never read, since nothing can reference them
    /// and their bytes were not produced by the tree this server walks.
    segment: Option<IndexedSegment>,
    standing: Standing,
    bytes: u64,
    /// Last use, which is the LRU clock: stamped by the write and bumped by every
    /// hydration, on every file in the chain. Kept on the file itself rather than in
    /// an index of our own, so it survives a crash and reads with `ls -lt`.
    mtime: SystemTime,
}

impl Entry {
    fn segment(&self) -> Option<&IndexedSegment> {
        self.segment.as_ref()
    }

    /// The segment, if anything may still be built on it.
    fn buildable(&self) -> Option<&IndexedSegment> {
        self.segment
            .as_ref()
            .filter(|_| self.standing == Standing::Ready)
    }
}

/// Everything both threads have to agree about. The lock is held for pointer
/// arithmetic and never across I/O: the engine thread reads this to decide whether
/// a stored chain beats its warm slots, and must never wait behind a gigabyte
/// being written.
#[derive(Default)]
struct Store {
    /// What is on disk, by path.
    index: HashMap<PathBuf, Entry>,
    /// Which segment each cache slot's conversation ends in — its tail. The size
    /// budget never deletes one of these, nor anything the tail chains onto: they are
    /// what the warm conversations would come back from, and evicting the image of a
    /// conversation the server still holds is the one deletion that buys nothing.
    links: HashMap<usize, PathBuf>,
    /// How many times each slot has been repurposed, bumped by every `unlink`.
    ///
    /// A write is claimed by the writer thread and finishes seconds later, and the slot
    /// it came from can be emptied or handed to another conversation in between. The
    /// segment is still a good image of the conversation it holds, so it is kept — but
    /// the LINK must not be re-established, or the size budget would protect a file on
    /// behalf of a slot that no longer holds it, forever. Comparing the epoch the
    /// request was enqueued at against the current one is how the completion knows.
    epochs: HashMap<usize, u64>,
    /// The write waiting for each slot, at most one, newest wins.
    ///
    /// The payload lives here rather than in the channel deliberately. A queued
    /// write holds `Arc`s to a gigabyte of host images, and a conversation paged out
    /// twice before the writer catches up would otherwise leave the superseded
    /// payload pinned in an unbounded channel — a slow disk turning into unbounded
    /// host memory. Replacing the map entry drops those `Arc`s at enqueue, and the
    /// channel carries only a wake.
    pending: HashMap<usize, WriteRequest>,
}

impl Store {
    /// The segments whose parent reference names `parent` — the roots when it is
    /// `None` — that anything may still be built on. A demoted segment is skipped
    /// here, which is what stops a walk from reaching it or anything below it.
    fn children_of(&self, parent: Option<u64>) -> Vec<(&PathBuf, &IndexedSegment)> {
        self.index
            .iter()
            .filter_map(|(path, entry)| entry.buildable().map(|seg| (path, seg)))
            .filter(|(_, seg)| seg.parent.map(|p| p.name) == parent)
            .collect()
    }

    /// Whether anything chains onto `name`. A segment with children is never
    /// evictable: dropping it would strand every conversation behind it.
    ///
    /// Demoted children count. Their own contents are in doubt, not their place in the
    /// tree — and a parent deleted while one of them still points at it is a chain
    /// that can never be repaired, where a demoted child costs only this run's reads.
    fn has_children(&self, name: u64) -> bool {
        self.index
            .values()
            .filter_map(Entry::segment)
            .any(|seg| seg.parent.is_some_and(|p| p.name == name))
    }

    /// Stop building on `name` and on everything chained behind it.
    ///
    /// What a failed read of one file says about the rest of its subtree: nothing can
    /// be composed through a segment that could not be read, so every descendant would
    /// win `candidate()` only to fail the same way, once per request until the next
    /// scan. Index-only — no file is touched, and the next scan re-links whatever is
    /// still good.
    fn demote(&mut self, name: u64) {
        let mut doomed = vec![name];
        let mut depth = 0;
        while let Some(name) = doomed.pop() {
            let children: Vec<u64> = self
                .index
                .values()
                .filter_map(Entry::segment)
                .filter(|seg| seg.parent.is_some_and(|p| p.name == name))
                .map(|seg| seg.name)
                .collect();
            for entry in self.index.values_mut() {
                if entry.segment().is_some_and(|seg| seg.name == name) {
                    entry.standing = Standing::Demoted;
                }
            }
            depth += 1;
            if depth > MAX_CHAIN_DEPTH {
                return;
            }
            doomed.extend(children);
        }
    }

    /// Stop building on the segment stored at `path`, and on everything behind it.
    fn demote_at(&mut self, path: &Path) {
        let Some(name) = self.segment_at(path).map(|seg| seg.name) else {
            return;
        };
        self.demote(name);
    }

    /// The whole token history a stored segment ends at, assembled from its chain, or
    /// `None` when the chain cannot be walked to a root.
    ///
    /// This is the authority a WRITE needs. A segment's name is a hash of exactly this
    /// history, and everything the writer does by name — replacing a file, deleting a
    /// superseded one — would act on the wrong conversation if two histories ever
    /// hashed alike. The load path has always compared tokens for that reason; this is
    /// the same check on the other side.
    fn cumulative(&self, path: &Path) -> Option<Vec<u32>> {
        let mut spans: Vec<&[u32]> = Vec::new();
        let mut at = self.segment_at(path);
        while let Some(segment) = at {
            spans.push(&segment.span);
            if spans.len() > MAX_CHAIN_DEPTH {
                return None;
            }
            let Some(parent) = segment.parent else {
                let mut history = Vec::new();
                for span in spans.iter().rev() {
                    history.extend_from_slice(span);
                }
                return Some(history);
            };
            at = self.path_of(parent.name).cloned().and_then(|parent| {
                // The borrow has to end before the next lookup, so the path is owned.
                self.index.get(&parent).and_then(Entry::segment)
            });
        }
        None
    }

    /// How many segments deep the subtree rooted at `name` reaches, counting that
    /// segment as one. What a split has to know: cutting a segment pushes everything
    /// behind it one level further from the root.
    fn subtree_height(&self, name: u64) -> usize {
        let mut height = 0;
        let mut level = vec![name];
        while !level.is_empty() && height < MAX_CHAIN_DEPTH {
            height += 1;
            level = self
                .index
                .values()
                .filter_map(Entry::segment)
                .filter(|seg| seg.parent.is_some_and(|p| level.contains(&p.name)))
                .map(|seg| seg.name)
                .collect();
        }
        height
    }

    /// Whether a warm slot would come back from this file.
    fn is_linked(&self, path: &Path) -> bool {
        self.links.values().any(|linked| linked == path)
    }

    fn segment_at(&self, path: &Path) -> Option<&IndexedSegment> {
        self.index.get(path).and_then(Entry::segment)
    }

    fn path_of(&self, name: u64) -> Option<&PathBuf> {
        self.index
            .iter()
            .find(|(_, entry)| entry.segment().is_some_and(|seg| seg.name == name))
            .map(|(path, _)| path)
    }

    /// The files a chain ending at `leaf` occupies, root first. Best-effort: a chain
    /// whose parent is missing from the index stops there, since the only caller
    /// stamps mtimes.
    fn chain_paths(&self, leaf: &Path) -> Vec<PathBuf> {
        let mut chain = vec![leaf.to_path_buf()];
        let mut at = self.segment_at(leaf);
        while let Some(parent) = at.and_then(|seg| seg.parent) {
            if chain.len() >= MAX_CHAIN_DEPTH {
                break;
            }
            let Some(path) = self.path_of(parent.name).cloned() else {
                break;
            };
            at = self.segment_at(&path);
            chain.push(path);
        }
        chain.reverse();
        chain
    }
}

/// The half of the tier both threads share.
struct Shared {
    store: Mutex<Store>,
    /// `<cache_dir>/kv`, the root the size budget spans.
    root: PathBuf,
    /// `<cache_dir>/kv/<checkpoint>`, where this server's segments go.
    dir: PathBuf,
    checkpoint: CheckpointId,
    /// Ceiling in bytes over everything under `root`.
    budget: u64,
    logger: ServeLogger,
}

impl Shared {
    /// The store, recovering from a poisoned lock rather than propagating it: a
    /// panic in the writer must not take the engine thread down with it on the next
    /// lookup, and the worst a torn index costs is a cache miss.
    fn store(&self) -> MutexGuard<'_, Store> {
        self.store.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn failed(&self, action: &'static str, error: impl std::fmt::Display) {
        self.logger.log(ServeLog::DiskCacheFailed {
            action,
            error: error.to_string(),
        });
    }

    /// The file a segment identity lives in. The mapping is the whole parent lookup:
    /// a child names its parent, and the parent is at that name.
    fn path_for(&self, id: &ChainId) -> PathBuf {
        self.dir.join(id.file_name())
    }
}

/// What the writer thread is sent. The payloads live in `Store::pending`, so this
/// only ever says "there may be work" — a wake that finds the map empty is
/// ordinary, since a replaced request leaves its wake behind.
enum Message {
    Wake,
    /// A rendezvous, acknowledged once everything pending has been written. The
    /// shutdown path waits on one with a bounded budget.
    Flush(SyncSender<()>),
}

/// One conversation, ready to serialize without touching the slot it came from.
///
/// The records arrive as `Arc` clones, which is the whole reason slot images are
/// `Arc`-wrapped: they are immutable once built and slots replace rather than
/// mutate them, so the writer can spend a second on a gigabyte while the engine
/// pages the next conversation in.
struct WriteRequest {
    slot: usize,
    /// Enqueue order, so the writer serves the slots in the order they were paged
    /// out rather than in whatever order the map iterates.
    stamp: u64,
    /// The slot's repurpose count when this was enqueued (see [`Store::epochs`]).
    epoch: u64,
    /// The conversation's history, never longer than the rows behind it: positions
    /// the image cannot back are positions no segment could store.
    tokens: Vec<u32>,
    full_kv: Arc<HostFullKv>,
    /// Ascending by position, every one inside the history.
    snapshots: Vec<(usize, Arc<HostSnapshot>)>,
    drafter: Option<Arc<DrafterImage>>,
}

impl WriteRequest {
    /// Host bytes this write carries. The three images each know their own size;
    /// the token history is small enough beside them to leave out.
    fn byte_len(&self) -> u64 {
        let snapshots: usize = self.snapshots.iter().map(|(_, s)| s.byte_len()).sum();
        let drafter = self.drafter.as_ref().map_or(0, |d| d.byte_len());
        (self.full_kv.byte_len() + snapshots + drafter) as u64
    }
}

/// The disk tier as the engine holds it. Absent (`DiskCache::open` returning
/// `None`) means the tier is off, which every call site treats as "there is
/// nothing on disk" rather than as an error.
pub(super) struct DiskCache {
    tx: Sender<Message>,
    shared: Arc<Shared>,
    /// Conversations shorter than this are not written.
    min_tokens: usize,
    /// Enqueue order for the pending map. A `Cell` because only the engine thread
    /// enqueues.
    stamp: Cell<u64>,
    /// Whether the store still binds to the model that is loaded. Cleared by
    /// [`DiskCache::verify`] when the checkpoint the scan bound to is not the one the
    /// weights came from, and never set again: nothing on disk describes those
    /// weights, and the operator is told once rather than per request. A `Cell`
    /// because only the engine thread reads or writes it.
    trusted: Cell<bool>,
}

/// A stored chain that could serve an arriving prompt, and how deep.
pub(super) struct DiskCandidate {
    /// The chain's last segment. The rest of the chain is resolved from the files'
    /// own parent references at load, never from the index: the writer may
    /// re-partition a span between the choice and the read, and a segment's name
    /// survives that while its place in a chain does not.
    path: PathBuf,
    /// Positions the chain covers.
    pub(super) tokens: usize,
    /// Where a slot hydrated from this chain would resume, which is the number the
    /// choice between disk and RAM is made on.
    pub(super) resume: usize,
    /// How many segments the chain holds, and their total size: what the read costs.
    segments: usize,
    bytes: u64,
}

/// Which FILE a path resolved to, as opposed to which name.
///
/// The writer publishes every segment by renaming a finished file over its name, so a
/// name is not a file: between a reader deciding a file is unusable and unlinking it,
/// the writer can have put a fresh, valid segment there. Deleting by name would then
/// throw away the new one. The inode says which bytes were actually condemned; the
/// size and mtime come along because an inode number can be reused once the original
/// is gone.
///
/// It narrows the window rather than closing it — there is no unlink-if-inode-matches
/// — and what remains is one lost cache file, which costs a re-prefill and never
/// serves a wrong answer.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FileIdentity {
    ino: u64,
    len: u64,
    mtime: Option<SystemTime>,
}

impl FileIdentity {
    /// Taken from an open file, so it describes the bytes the reader has hold of
    /// rather than whatever the name resolves to at the next syscall.
    fn of(file: &File) -> Option<Self> {
        Self::from(std::fs::File::metadata(file).ok()?)
    }

    fn at(path: &Path) -> Option<Self> {
        Self::from(std::fs::metadata(path).ok()?)
    }

    fn from(meta: std::fs::Metadata) -> Option<Self> {
        use std::os::unix::fs::MetadataExt;
        Some(Self {
            ino: meta.ino(),
            len: meta.len(),
            mtime: meta.modified().ok(),
        })
    }

    /// Whether the file at `path` is still the one this identity was taken of.
    fn still_at(&self, path: &Path) -> bool {
        Self::at(path).is_some_and(|now| now == *self)
    }
}

/// One segment of a chain as the load path holds it: where it came from, which file
/// that was at the moment it was opened, and what it held.
struct Read {
    path: PathBuf,
    seen: Option<FileIdentity>,
    segment: Segment,
}

impl DiskCache {
    /// Open the tier for the checkpoint `settings` names: create the directory,
    /// sweep whatever a crashed process left behind, index the tree, bring the
    /// store inside its budget, and start the writer.
    ///
    /// `None` when the tier is off, when no cache directory could be resolved, or
    /// when the checkpoint could not be identified — all three mean the same thing
    /// to every caller. The scan blocks the engine thread's startup, which is what
    /// it costs to have the first request find a warm store: headers only, tens of
    /// kilobytes per file.
    pub(super) fn open(settings: &ServeSettings, logger: &ServeLogger) -> Option<Self> {
        if !settings.disk_cache {
            return None;
        }
        let cache_dir = settings.cache_dir.as_ref()?;
        let checkpoint = match checkpoint_id(&settings.model) {
            Ok(id) => id,
            Err(e) => {
                logger.log(ServeLog::DiskCacheFailed {
                    action: "identifying the checkpoint",
                    error: format!("{e:#}"),
                });
                return None;
            }
        };
        let root = cache_dir.join(KV_SUBDIR);
        let dir = root.join(checkpoint.dir_name());
        if let Err(e) = std::fs::create_dir_all(&dir) {
            logger.log(ServeLog::DiskCacheFailed {
                action: "creating the segment directory",
                error: format!("{}: {e}", dir.display()),
            });
            return None;
        }
        let shared = Arc::new(Shared {
            store: Mutex::new(Store::default()),
            root,
            dir,
            checkpoint,
            budget: settings.disk_max_gib.saturating_mul(1 << 30),
            logger: logger.clone(),
        });
        scan(&shared);
        enforce_budget(&shared, shared.budget, &[]);

        let (tx, rx) = channel();
        let writer = Arc::clone(&shared);
        if let Err(e) = std::thread::Builder::new()
            .name(WRITER_THREAD.to_string())
            .spawn(move || writer_loop(rx, writer))
        {
            // Without the writer the tier can still hydrate what is already on
            // disk, and reporting that beats refusing to serve.
            shared.failed("starting the writer", e);
        }
        Some(Self {
            tx,
            shared,
            min_tokens: settings.disk_min_tokens,
            stamp: Cell::new(0),
            trusted: Cell::new(true),
        })
    }

    /// Check the store's binding against the checkpoint the weights were actually
    /// loaded from, and disable the tier for the rest of the process if they differ.
    ///
    /// The scan binds to the file the settings name, read before the model is loaded;
    /// nothing stops that file from being replaced in between — a re-quantized GGUF
    /// under the same path is exactly the case — and then everything stored is bound
    /// to weights nobody is serving. Hydration would upload another model's keys.
    /// Called on every load, the lazy first one and each reload after an idle unload.
    pub(super) fn verify(&self, loaded: CheckpointId) {
        if !self.trusted.get() || loaded == self.shared.checkpoint {
            return;
        }
        self.trusted.set(false);
        self.shared.failed(
            "binding the store to the loaded checkpoint",
            format!(
                "the segments were scanned against {} but the weights loaded are {}; the disk \
                 cache is off for the rest of this run",
                self.shared.checkpoint.dir_name(),
                loaded.dir_name()
            ),
        );
    }

    /// The stored chain that would serve `prompt` better than `beat` — the position
    /// the warm slots offered — or `None` when none does.
    ///
    /// The rule is a cold slot's: longest common prefix over token histories, then
    /// the deepest stored snapshot at or before it, capped so the prefill that
    /// follows still has a token to run. A tie goes to `beat`, since reading a
    /// gigabyte to resume exactly where RAM already could is pure cost.
    ///
    /// EVERY segment is a candidate, not only the tails: a new conversation that
    /// shares nothing but the system prompt hydrates the base chain and resumes at
    /// its boundary. The snapshot it resumes at may live anywhere in that chain, which
    /// is why the walk carries the chain's snapshots down with it.
    pub(super) fn candidate(&self, prompt: &[u32], beat: usize) -> Option<DiskCandidate> {
        if !self.trusted.get() {
            return None;
        }
        let reusable = prompt.len().checked_sub(1)?;
        let store = self.shared.store();
        let mut best: Option<DiskCandidate> = None;
        let mut best_mtime = SystemTime::UNIX_EPOCH;
        walk_chains(&store, prompt, |node| {
            // A segment the prompt never reaches into offers nothing its ancestors do
            // not already offer, and every one of those is its own candidate.
            if node.matched <= node.segment.start {
                return;
            }
            let cap = node.matched.min(reusable);
            let Some(resume) = deepest_at_or_before(node.snapshots, cap).filter(|pos| *pos > beat)
            else {
                return;
            };
            // Deeper wins; between two chains that reach equally deep the more
            // recently used one is the one to keep alive.
            let better = best
                .as_ref()
                .is_none_or(|held| (resume, node.mtime) > (held.resume, best_mtime));
            if better {
                best_mtime = node.mtime;
                best = Some(DiskCandidate {
                    path: node.path.clone(),
                    tokens: node.segment.end(),
                    resume,
                    segments: node.depth,
                    bytes: node.bytes,
                });
            }
        });
        best
    }

    /// How far the deepest stored chain agrees with `prompt`, whatever any of them
    /// could be resumed at.
    ///
    /// A different question from [`DiskCache::candidate`], which answers where a chain
    /// could be PICKED UP — capped by the snapshots it happens to hold. A chain that
    /// shares twenty thousand tokens but holds no snapshot below them offers no
    /// candidate at all, and that is exactly the case worth knowing about: the
    /// divergence is real, the writer is about to split a segment there, and only the
    /// job that prefills through the position can leave a snapshot at it. Without one,
    /// every later conversation sharing that prefix prefills it again from zero.
    ///
    /// Index only — one walk of the segment tree, no file touched. `0` when the tier
    /// is off, untrusted, or empty.
    pub(super) fn deepest_shared(&self, prompt: &[u32]) -> usize {
        if !self.trusted.get() {
            return 0;
        }
        let store = self.shared.store();
        let mut deepest = 0;
        walk_chains(&store, prompt, |node| deepest = deepest.max(node.matched));
        deepest
    }

    /// Read and validate a candidate's whole chain, on the engine thread as part of
    /// the job it serves, and compose it into one image. `None` means the chain could
    /// not be used after all, and the caller falls back to what the warm slots
    /// offered.
    ///
    /// The chain is resolved from the files' own parent references rather than from
    /// the index, and every link is checked (see [`DiskCache::verify_chain`]). Then
    /// `prompt` is checked against the history the chain actually holds, over exactly
    /// the positions about to be resumed. Nothing before this point has compared
    /// them — a candidate is chosen from the headers the scan read, and a segment's
    /// name is a hash of its token ids, so a hash collision, or a file another build
    /// named by a different rule, would otherwise resume this conversation on another
    /// one's keys. Cheap, and it turns a probabilistic argument into a structural one.
    pub(super) fn load(&self, candidate: &DiskCandidate, prompt: &[u32]) -> Option<DiskImage> {
        let chain = self.read_chain(candidate)?;
        let tokens = self.verify_chain(&chain, candidate)?;

        let restore = candidate.resume;
        let agrees = tokens.len() >= restore
            && prompt.len() >= restore
            && tokens[..restore] == prompt[..restore];
        if !agrees {
            self.shared.failed(
                "matching a stored chain to the prompt",
                format!(
                    "the chain holds {} tokens that diverge from this conversation inside the \
                     {restore} being resumed",
                    tokens.len()
                ),
            );
            // The segment holding the first position they disagree at is the one whose
            // stored history is not what its name claims. Blaming the whole chain would
            // delete a base other conversations are correctly resuming from.
            let diverged = common_prefix_len(&tokens, prompt);
            let liar = chain
                .iter()
                .find(|read| read.segment.end() > diverged)
                .or_else(|| chain.last());
            if let Some(read) = liar {
                self.reject(&read.path, read.seen.as_ref(), "collided");
                if read.path != candidate.path {
                    self.forget_stale(candidate);
                }
            }
            return None;
        }
        // The resume position came from the header the scan read, and the chain may
        // have been re-partitioned since — by a perfectly good set of segments over the
        // same conversation with different snapshot boundaries. So the position is
        // checked against what was just read, and a chain that cannot serve it is a
        // MISS, not damage: the index is brought up to date instead, and the next
        // request chooses again knowing what is really there.
        if !chain.iter().any(|read| {
            read.segment
                .snapshots
                .iter()
                .any(|(pos, _)| *pos == restore)
        }) {
            self.shared.failed(
                "resuming a stored chain",
                format!("the chain has no snapshot at {restore} any more"),
            );
            self.refresh(&chain);
            return None;
        }
        match compose(chain) {
            Ok(image) => Some(image),
            Err(e) => {
                self.shared
                    .failed("composing a stored chain", format!("{e:#}"));
                None
            }
        }
    }

    /// Read the chain ending at `candidate`, leaf first, resolving each segment's
    /// parent by the name it stores. Every file is validated in full, which is what
    /// makes a rotted plane a miss instead of an upload.
    ///
    /// Three things bound the read, all of them BEFORE a body is touched, because the
    /// bodies are gigabyte-scale and every one already read is still held while the
    /// next is parsed. A name may appear once — a segment that names itself, or two
    /// that name each other, would otherwise be read up to the depth cap, each read a
    /// full copy in RAM. The files' declared sizes are charged against
    /// [`MAX_CHAIN_BYTES`] as they go. And the depth cap ends it regardless. An
    /// allocation that fails aborts the process, which no amount of error handling
    /// downstream can turn back into a cache miss.
    fn read_chain(&self, candidate: &DiskCandidate) -> Option<Vec<Read>> {
        let mut chain: Vec<Read> = Vec::new();
        let mut seen: Vec<u64> = Vec::new();
        let mut charged = 0u64;
        let mut next = Some(candidate.path.clone());
        while let Some(path) = next {
            let refuse = |why: String| {
                self.shared.failed("reading a stored chain", why);
                self.forget_stale(candidate);
            };
            if chain.len() >= MAX_CHAIN_DEPTH {
                refuse(format!(
                    "the chain is longer than the {MAX_CHAIN_DEPTH} segments this build will \
                     compose"
                ));
                return None;
            }
            match disk_cache::segment_name_of(&path) {
                Some(name) if !seen.contains(&name) => seen.push(name),
                Some(name) => {
                    refuse(format!("segment {name:016x} chains onto itself"));
                    return None;
                }
                None => {
                    refuse(format!("{} is not named by this build", path.display()));
                    return None;
                }
            }
            // ONE descriptor answers all three questions: which file this is, how much
            // reading it will cost, and what it holds. Going back to the path between
            // them would be asking about a name, not a file — the writer publishes
            // segments by renaming, so the length charged against the budget and the
            // bytes that arrive could come from different files, and a rejection could
            // condemn one the reader never saw.
            let file = match File::open(&path) {
                Ok(file) => file,
                Err(e) => {
                    self.shared.failed("opening a stored segment", e);
                    if path != candidate.path {
                        self.forget_stale(candidate);
                    }
                    return None;
                }
            };
            let seen_as = FileIdentity::of(&file);
            charged = charged.saturating_add(seen_as.as_ref().map_or(0, |id| id.len));
            if charged > MAX_CHAIN_BYTES {
                refuse(format!(
                    "the chain declares more than the {MAX_CHAIN_BYTES} bytes this build will read \
                     into memory at once"
                ));
                return None;
            }
            let segment = match disk_cache::read_segment_from(
                file,
                &self.shared.checkpoint,
                TOKENIZATION_RULES_VERSION,
            ) {
                Ok(segment) => segment,
                Err(e) => {
                    self.shared.failed("reading a stored segment", &e);
                    self.reject(&path, seen_as.as_ref(), e.class());
                    // A parent that could not be read leaves the whole chain unusable,
                    // and the index does not know that: demoting the subtree stops this
                    // run from choosing any of it again, once per request.
                    if path != candidate.path {
                        self.forget_stale(candidate);
                    }
                    return None;
                }
            };
            next = segment.parent.map(|parent| {
                self.shared
                    .dir
                    .join(disk_cache::segment_file_name(parent.name))
            });
            chain.push(Read {
                path,
                seen: seen_as,
                segment,
            });
        }
        chain.reverse();
        Some(chain)
    }

    /// Verify the chain root-first and return the cumulative history it holds.
    ///
    /// Three claims per segment, all of them cheap and all of them load-bearing: it
    /// starts where the chain in front of it reaches, it names that segment's identity
    /// with both hashes, and the history leading to it hashes to its own file name. A
    /// segment that fails any of them is the one deleted — the descendant is always the
    /// one making the claim, so a shared base is never condemned for a child's lie.
    fn verify_chain(&self, chain: &[Read], candidate: &DiskCandidate) -> Option<Vec<u32>> {
        let mut tokens: Vec<u32> = Vec::new();
        let mut previous: Option<ChainId> = None;
        for read in chain {
            let (path, segment) = (&read.path, &read.segment);
            let blame = |why: String| {
                self.shared.failed("verifying a stored chain", why);
                self.reject(path, read.seen.as_ref(), "corrupt");
                if *path != candidate.path {
                    self.forget_stale(candidate);
                }
            };
            if segment.start != tokens.len() {
                blame(format!(
                    "{} starts at {}, the chain in front of it reaches {}",
                    path.display(),
                    segment.start,
                    tokens.len()
                ));
                return None;
            }
            if segment.parent != previous.map(|id| id.as_parent()) {
                blame(format!(
                    "{} names a parent that is not the chain in front of it",
                    path.display()
                ));
                return None;
            }
            tokens.extend_from_slice(&segment.tokens);
            let id = chain_id(&tokens);
            if disk_cache::segment_name_of(path) != Some(id.name) {
                blame(format!(
                    "{} holds a history that names {:016x}",
                    path.display(),
                    id.name
                ));
                return None;
            }
            previous = Some(id);
        }
        Some(tokens)
    }

    /// Bring the index entries back in step with the files behind them, from segments
    /// that were just read in full. Cheaper than a re-scan and more accurate: this is
    /// what the chain holds now, rather than what it held when the server started.
    fn refresh(&self, chain: &[Read]) {
        let mut store = self.shared.store();
        for read in chain {
            let bytes = std::fs::metadata(&read.path).map(|meta| meta.len()).ok();
            let Some(name) = disk_cache::segment_name_of(&read.path) else {
                continue;
            };
            if let Some(entry) = store.index.get_mut(&read.path) {
                entry.segment = Some(IndexedSegment {
                    name,
                    start: read.segment.start,
                    parent: read.segment.parent,
                    span: read.segment.tokens.clone(),
                    snapshots: read.segment.snapshots.iter().map(|(pos, _)| *pos).collect(),
                });
                // The size too: a replacement is a different file, and the budget is
                // counting it.
                if let Some(bytes) = bytes {
                    entry.bytes = bytes;
                }
            }
        }
    }

    /// Stop building on a candidate whose chain turned out to be unusable, without
    /// touching any file: what is wrong is in front of it, and nothing has been learned
    /// about its own bytes. It stays counted by the budget and stays its parent's
    /// child, so nothing is evicted out from under it; it simply stops being offered
    /// until the next scan, which is what keeps a broken chain from costing a doomed
    /// read once per request.
    fn forget_stale(&self, candidate: &DiskCandidate) {
        let mut store = self.shared.store();
        store.demote_at(&candidate.path);
        store.links.retain(|_, linked| linked != &candidate.path);
    }

    /// Drop a file that cannot be used. Only a verdict about the file's CONTENTS
    /// deletes it: an I/O fault says nothing about the bytes, and deleting on one would
    /// throw away a good segment because a disk hiccuped.
    ///
    /// `seen` is the file as it was when this rejection's evidence was gathered. The
    /// writer publishes segments by renaming them into place, so the name may hold a
    /// different — and perfectly good — file by now; that one is not what was
    /// condemned, and it is left alone.
    ///
    /// The file goes before the index entry does, as everywhere else here: an entry
    /// dropped for a file still on disk makes the budget claim room the disk does not
    /// have. Whatever chained onto it stops being built on either way — those segments
    /// cannot be composed without it, and would otherwise win `candidate()` and fail
    /// the same way once per request.
    fn reject(&self, path: &Path, seen: Option<&FileIdentity>, class: &'static str) {
        if !deletable(class) {
            return;
        }
        if seen.is_some_and(|seen| !seen.still_at(path)) {
            self.shared.failed(
                "deleting a rejected segment",
                format!(
                    "{} is not the file this rejection was decided on any more; it stays",
                    path.display()
                ),
            );
            self.shared.store().demote_at(path);
            return;
        }
        condemn(
            &self.shared,
            path,
            match class {
                "collided" => DiskEvictReason::Collided,
                class => DiskEvictReason::Invalid { class },
            },
        );
    }

    /// Stop building on a stored chain that the ENGINE turned down after reading it —
    /// its image did not fit the slot it was meant for, or the job that asked for it
    /// went away.
    ///
    /// Nothing is wrong with the files, so nothing is deleted; but re-reading and
    /// re-composing a multi-gigabyte chain on every later request that shares this
    /// prefix, only to turn it down again, is the whole cost this tier exists to avoid.
    /// The next scan puts it back in circulation.
    pub(super) fn set_unusable(&self, candidate: &DiskCandidate) {
        self.forget_stale(candidate);
    }

    /// Forget which segment a slot's conversation ends in, because the slot no
    /// longer holds that conversation: it was emptied, or reused for another one.
    ///
    /// The link exists to keep the size budget from evicting a chain a warm
    /// conversation would come back from. A link left behind after the slot moved on
    /// protects files nothing will ever read — and, since the budget refuses to
    /// touch them, ones no amount of pressure can reclaim.
    pub(super) fn unlink(&self, slot: usize) {
        let mut store = self.shared.store();
        store.links.remove(&slot);
        // Anything already claimed by the writer for this slot finishes without
        // re-linking it: the segments stay, the protection does not.
        *store.epochs.entry(slot).or_default() += 1;
    }

    /// Record that `slot` now holds the conversation `candidate`'s chain was read
    /// from: bump the mtime of every file in the chain, which is the LRU clock, and
    /// link the slot to its tail so a later write from that slot continues this chain
    /// rather than starting a second copy of it.
    ///
    /// Every file, not just the tail: a shared base is read by every conversation
    /// behind it and would otherwise age out while the tails that need it stay fresh.
    pub(super) fn note_hydrated(&self, slot: usize, candidate: &DiskCandidate, ms: u64) {
        let now = SystemTime::now();
        let chain = self.shared.store().chain_paths(&candidate.path);
        for path in &chain {
            if let Err(e) = touch(path, now) {
                // Only the LRU order suffers: the conversation is in the slot either
                // way.
                self.shared.failed("stamping a segment", e);
            }
        }
        {
            let mut store = self.shared.store();
            for path in &chain {
                if let Some(entry) = store.index.get_mut(path) {
                    entry.mtime = now;
                }
            }
            store.links.insert(slot, candidate.path.clone());
        }
        self.shared.logger.log(ServeLog::DiskChainHydrated {
            segments: candidate.segments.max(chain.len()),
            tokens: candidate.tokens,
            resume: candidate.resume,
            bytes: candidate.bytes,
            ms,
        });
    }

    /// Queue `slot`'s conversation to be written, unless it is not worth writing.
    ///
    /// Four ways it is not: the tier no longer binds to the loaded weights, the
    /// conversation is shorter than `disk_min_tokens`, it has no snapshot to resume
    /// at, or what is already stored covers it well enough (see [`DISK_MIN_GROWTH`]).
    /// Everything else is coalesced into the pending map — at most one payload per
    /// slot, newest wins — and the wake that follows cannot fail or block: this runs
    /// on the engine thread, right after a page-out, and a slow or dead writer must
    /// never be something a request waits on.
    pub(super) fn queue_write(
        &self,
        slot: usize,
        tokens: &[u32],
        full_kv: &Arc<HostFullKv>,
        snapshots: impl Iterator<Item = (usize, Arc<HostSnapshot>)>,
        drafter: Option<&Arc<DrafterImage>>,
    ) {
        if !self.trusted.get() {
            return;
        }
        // Only positions the rows can back are storable: a slot's history can outrun
        // its image (a cancelled job truncates its tokens to what the cache held), and
        // a segment whose span is longer than its rows is a file the container refuses.
        let history = &tokens[..tokens.len().min(full_kv.pos)];
        if history.len() < self.min_tokens || history.is_empty() {
            return;
        }
        // Ascending, distinct, inside the history and never at zero: the properties
        // the container refuses a file for. A slot maintains all of them (the anchor is
        // the shallowest position it holds, and every snapshot names a position the
        // image covers), so the filtering here is what keeps one impossible snapshot
        // from costing the whole write.
        let mut snapshots: Vec<(usize, Arc<HostSnapshot>)> = snapshots
            .filter(|(pos, rings)| *pos > 0 && *pos <= history.len() && rings.pos == *pos)
            .collect();
        snapshots.sort_by_key(|(pos, _)| *pos);
        snapshots.dedup_by_key(|(pos, _)| *pos);
        if snapshots.is_empty() {
            return;
        }
        {
            let mut store = self.shared.store();
            if !worth_writing(&locate(&store, history), history.len()) {
                return;
            }
            let stamp = self.stamp.get().wrapping_add(1);
            self.stamp.set(stamp);
            let epoch = store.epochs.get(&slot).copied().unwrap_or_default();
            // Replacing the entry drops the superseded request's `Arc`s here, on the
            // engine thread, instead of leaving a gigabyte pinned in the channel.
            store.pending.insert(
                slot,
                WriteRequest {
                    slot,
                    stamp,
                    epoch,
                    tokens: history.to_vec(),
                    full_kv: Arc::clone(full_kv),
                    snapshots,
                    drafter: drafter.map(Arc::clone),
                },
            );
        }
        let _ = self.tx.send(Message::Wake);
    }

    /// Host bytes the writer still has queued. The caller turns this into the
    /// wait it will allow (`disk_flush_budget`), because how long a flush should
    /// be given is a property of how much is in it: the fixed grace this
    /// replaced was sized on a single ~4.2 GiB image and a long conversation is
    /// several of those.
    pub(super) fn pending_bytes(&self) -> u64 {
        self.shared
            .store()
            .pending
            .values()
            .map(WriteRequest::byte_len)
            .sum()
    }

    /// Wait up to `budget` for the queued writes to land. Called on the way down —
    /// a graceful shutdown, an idle unload — where losing warmth is acceptable and
    /// hanging is not, so the wait is bounded and its expiry is reported rather than
    /// retried.
    pub(super) fn flush(&self, budget: Duration) {
        let (ack, written) = sync_channel(0);
        if self.tx.send(Message::Flush(ack)).is_err() {
            return;
        }
        if written.recv_timeout(budget).is_err() {
            self.shared.failed(
                "flushing the store",
                format!(
                    "the writer did not finish within {:.0}s; what it had left costs a \
                     re-prefill, nothing more",
                    budget.as_secs_f64()
                ),
            );
        }
    }
}

/// Identify the checkpoint the segments will be bound to. Metadata only, on the CPU
/// device: this runs before the model is loaded, and the hash covers the header,
/// the metadata and the tensor index — never the tens of gigabytes behind them.
fn checkpoint_id(model: &Path) -> anyhow::Result<CheckpointId> {
    Ok(gguf::open(model, &Device::Cpu)?.checkpoint_id())
}

/// One segment as a chain walk reaches it.
struct Node<'a> {
    path: &'a PathBuf,
    segment: &'a IndexedSegment,
    /// How far the prompt and this segment's cumulative history agree.
    matched: usize,
    /// Every snapshot position in the chain from the root to here, ascending.
    snapshots: &'a [usize],
    /// Segments in that chain, and their total size.
    depth: usize,
    bytes: u64,
    mtime: SystemTime,
}

/// Walk every stored chain from its roots, visiting each segment once with the
/// longest common prefix its cumulative history shares with `prompt` and the
/// snapshots its whole chain offers.
///
/// The walk is what makes cumulative histories free: a segment's agreement with the
/// prompt is its parent's, extended by its own span only when the parent's reached
/// the boundary, so nothing has to store or re-hash the tokens in front of it.
/// Depth-first, so the chain state each node needs is the stack's own — every node
/// truncates the shared snapshot list back to its parent's length before extending
/// it, which is sound only because a depth-first pop order visits everything pushed
/// after a node before that node itself.
fn walk_chains(store: &Store, prompt: &[u32], mut visit: impl FnMut(Node<'_>)) {
    let mut snapshots: Vec<usize> = Vec::new();
    // The segment, its parent's agreement, the length its parent left the snapshot
    // list at, its parent's depth, and the bytes its chain has cost so far.
    let mut stack: Vec<(&PathBuf, &IndexedSegment, usize, usize, usize, u64)> = store
        .children_of(None)
        .into_iter()
        .map(|(path, segment)| (path, segment, 0usize, 0usize, 0usize, 0u64))
        .collect();
    while let Some((path, segment, reached, base, depth, spent)) = stack.pop() {
        snapshots.truncate(base);
        // A span that does not begin where the one in front of it ended is not part of
        // this chain at all. Such a segment is dropped at scan, and a walk that met one
        // anyway must not credit it with its parent's agreement.
        if segment.start != reached {
            continue;
        }
        let matched = segment.start + common_prefix_len(&segment.span, &prompt[segment.start..]);
        snapshots.extend_from_slice(&segment.snapshots);
        let entry = &store.index[path];
        let bytes = spent.saturating_add(entry.bytes);
        visit(Node {
            path,
            segment,
            matched,
            snapshots: &snapshots,
            depth: depth + 1,
            bytes,
            mtime: entry.mtime,
        });
        if depth + 1 >= MAX_CHAIN_DEPTH {
            continue;
        }
        // A child can only extend an agreement that reached this span's end; below
        // that the prompt has already diverged and nothing deeper can match.
        if matched < segment.end() {
            continue;
        }
        let below = snapshots.len();
        for (child, seg) in store.children_of(Some(segment.name)) {
            stack.push((child, seg, matched, below, depth + 1, bytes));
        }
    }
}

/// What the store already holds for one conversation: how far it covers it, how deep
/// it could resume it, and where it would have to be cut for the two to share what
/// they have in common.
struct Located {
    /// The deepest position a chain of stored segments covers that is a prefix of the
    /// history — where a new tail would start.
    boundary: usize,
    /// How many segments that chain holds.
    depth: usize,
    /// The deepest position anything could resume this conversation at today: the
    /// deepest snapshot at or before the boundary anywhere in that chain. What the
    /// growth floor is measured from, because it is what a restart would pay.
    resumable: usize,
    /// The stored segment the history diverges INSIDE: its file, the position it
    /// diverges at, and how deep the subtree hanging off that segment reaches
    /// (1 when it is a leaf). The height is what tells a split whether the chains
    /// behind the cut can afford the level it adds to every one of them.
    diverged: Option<Divergence>,
    /// The chain's last segment and the position its span starts at, for the
    /// re-partition a chain at its depth cap forces.
    tail: Option<(PathBuf, usize)>,
}

/// Where a conversation leaves the segment that was holding its prefix.
struct Divergence {
    path: PathBuf,
    at: usize,
    /// Segments in the deepest chain hanging off the segment being cut, itself
    /// included.
    height: usize,
}

/// Follow `history` down the tree as far as stored segments go.
///
/// At every level the child sharing the most tokens with the history wins: two
/// children of one branch point can each share a few tokens with it, and cutting the
/// one that matches deepest is what keeps the shared prefix maximal. Nothing is
/// hashed here — the spans themselves are compared, which is a stronger check than
/// their names would be.
fn locate(store: &Store, history: &[u32]) -> Located {
    let mut located = Located {
        boundary: 0,
        depth: 0,
        resumable: 0,
        diverged: None,
        tail: None,
    };
    let mut parent = None;
    loop {
        let mut best: Option<(&PathBuf, &IndexedSegment, usize)> = None;
        for (path, segment) in store.children_of(parent) {
            if segment.start != located.boundary {
                continue;
            }
            let shared = common_prefix_len(&segment.span, &history[located.boundary..]);
            if shared == 0 {
                continue;
            }
            // The name breaks a tie, so two children matching equally far do not make
            // the choice depend on hash-map order.
            let better = best.is_none_or(|(_, held, held_shared)| {
                (shared, segment.name) > (held_shared, held.name)
            });
            if better {
                best = Some((path, segment, shared));
            }
        }
        let Some((path, segment, shared)) = best else {
            return located;
        };
        let reached = located.boundary + shared;
        if let Some(deepest) = deepest_at_or_before(&segment.snapshots, reached) {
            located.resumable = located.resumable.max(deepest);
        }
        if shared < segment.span.len() {
            located.diverged = Some(Divergence {
                path: path.clone(),
                at: reached,
                height: store.subtree_height(segment.name),
            });
            return located;
        }
        located.boundary = segment.end();
        located.depth += 1;
        located.tail = Some((path.clone(), segment.start));
        parent = Some(segment.name);
        if located.boundary == history.len() || located.depth >= MAX_CHAIN_DEPTH {
            return located;
        }
    }
}

/// Whether writing this conversation is worth the bytes: the store must not already
/// cover all of it, and it must reach at least [`DISK_MIN_GROWTH`] tokens past the
/// deepest position the store could resume it at. A conversation nothing on disk
/// resumes yet is always worth its first segment.
fn worth_writing(located: &Located, len: usize) -> bool {
    located.boundary < len && (located.resumable == 0 || len - located.resumable >= DISK_MIN_GROWTH)
}

fn deepest_at_or_before(positions: &[usize], at: usize) -> Option<usize> {
    positions.iter().rev().find(|pos| **pos <= at).copied()
}

/// Index the tree on disk and clean up what is not part of it.
///
/// Four kinds of file are found. This checkpoint's segments are validated by their
/// headers and linked up into chains, or deleted; a `.tmp` sibling is a write a
/// process did not finish and is swept (the name carries the writer's pid, so nothing
/// else ever reclaims it); another checkpoint's segments are counted without being
/// read, because the budget spans the whole store while their bytes describe a model
/// this server has not loaded; and a `.lkv` file in our own directory that is not
/// named by our rule gets that same treatment — it can be nobody's parent and will
/// not be read, but its bytes are real.
///
/// A file is deleted only for a verdict about its CONTENTS: a rejected header — which
/// is where the previous arc's flat v1 images go, since they fail the version check —
/// or a segment that chains onto nothing this build can walk. A header that cannot be
/// read at all — an I/O fault, a permission problem — is reported and left where it
/// is, unindexed for this run: nothing has been learned about it, and a disk that
/// hiccuped once must not cost the segments on it.
fn scan(shared: &Arc<Shared>) {
    let mut store = Store::default();
    let mut rejected = 0usize;
    let mut doomed: Vec<(PathBuf, u64, DiskEvictReason)> = Vec::new();
    let mut found: Vec<Found> = Vec::new();
    // Segments whose headers could not be READ. Their names are known from their file
    // names, which is enough to recognise what chains onto them — and everything that
    // does has to survive, since an I/O fault is not evidence about any file's
    // contents, least of all a child's.
    let mut unreadable: Vec<u64> = Vec::new();

    match std::fs::read_dir(&shared.dir) {
        Ok(entries) => {
            for path in files(entries, shared) {
                let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                match path.extension().and_then(|e| e.to_str()) {
                    Some("tmp") => sweep(shared, &path),
                    Some(disk_cache::EXTENSION) => {
                        let Some(name) = disk_cache::segment_name_of(&path) else {
                            // Not named by this build's rule, so nothing can reference
                            // it and nothing will read it — but the bytes count.
                            let mtime = modified(&path);
                            store.index.insert(
                                path,
                                Entry {
                                    segment: None,
                                    standing: Standing::Demoted,
                                    bytes,
                                    mtime,
                                },
                            );
                            continue;
                        };
                        match disk_cache::read_segment_header(
                            &path,
                            &shared.checkpoint,
                            TOKENIZATION_RULES_VERSION,
                        ) {
                            Ok(header) => found.push(Found {
                                mtime: modified(&path),
                                path,
                                name,
                                header,
                            }),
                            Err(e) => {
                                shared.failed("reading a segment header", &e);
                                if deletable(e.class()) {
                                    rejected += 1;
                                    doomed.push((
                                        path,
                                        bytes,
                                        DiskEvictReason::Invalid { class: e.class() },
                                    ));
                                } else {
                                    unreadable.push(name);
                                }
                            }
                        }
                    }
                    // Anything else in the directory belongs to somebody else.
                    _ => {}
                }
            }
        }
        Err(e) => shared.failed("reading the segment directory", e),
    }

    // Only what links up into a chain from a root is indexed and built on; what chains
    // onto a file that could not be read is kept but not built on; the rest is
    // unreachable, and rows with no spans in front of them are worth nothing however
    // intact they are.
    let Linked {
        linked,
        held,
        orphans,
    } = link_up(found, &unreadable);
    for orphan in orphans {
        rejected += 1;
        doomed.push((orphan.path, orphan.header.bytes, DiskEvictReason::Orphaned));
    }
    for (entry, standing) in linked
        .into_iter()
        .map(|entry| (entry, Standing::Ready))
        .chain(held.into_iter().map(|entry| (entry, Standing::Held)))
    {
        store.index.insert(
            entry.path,
            Entry {
                segment: Some(IndexedSegment {
                    name: entry.name,
                    start: entry.header.start,
                    parent: entry.header.parent,
                    span: entry.header.tokens,
                    snapshots: entry.header.snapshot_positions,
                }),
                standing,
                bytes: entry.header.bytes,
                mtime: entry.mtime,
            },
        );
    }

    // The other checkpoints' segments: their sizes and ages, and nothing else.
    match std::fs::read_dir(&shared.root) {
        Err(e) => shared.failed("reading the cache root", e),
        Ok(entries) => {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() || path == shared.dir {
                    continue;
                }
                let Ok(inner) = std::fs::read_dir(&path) else {
                    continue;
                };
                for image in files(inner, shared) {
                    // Another checkpoint's unfinished writes are swept too. Nothing
                    // will ever reclaim them — the pid in the name belongs to a
                    // process that is gone — and left alone they are a pile no
                    // eviction can reach, since only `.lkv` files are indexed.
                    match image.extension().and_then(|e| e.to_str()) {
                        Some("tmp") => {
                            sweep(shared, &image);
                            continue;
                        }
                        Some(disk_cache::EXTENSION) => {}
                        _ => continue,
                    }
                    let bytes = std::fs::metadata(&image).map(|m| m.len()).unwrap_or(0);
                    let mtime = modified(&image);
                    store.index.insert(
                        image,
                        Entry {
                            segment: None,
                            standing: Standing::Demoted,
                            bytes,
                            mtime,
                        },
                    );
                }
            }
        }
    }

    let segments = store.index.len();
    let bytes = store.index.values().map(|entry| entry.bytes).sum();
    *shared.store() = store;
    for (path, bytes, reason) in doomed {
        delete(shared, &path, bytes, reason);
    }
    shared.logger.log(ServeLog::DiskCacheScanned {
        segments,
        bytes,
        rejected,
    });
}

/// One `.lkv` file the scan read a header off.
struct Found {
    path: PathBuf,
    name: u64,
    header: disk_cache::SegmentHeader,
    mtime: SystemTime,
}

/// What the scan made of the files it read: the chains, what is waiting on a file
/// that could not be read, and what belongs to no chain at all.
struct Linked {
    /// Reachable from a root by verified links. Indexed and built on.
    linked: Vec<Found>,
    /// Chained onto a file whose header could not be READ. Indexed for its bytes and
    /// its place in the tree, never built on, and never deleted: the only thing wrong
    /// with it is a fault somewhere in front of it.
    held: Vec<Found>,
    /// Reachable from nothing. Deleted.
    orphans: Vec<Found>,
}

/// Sort what the scan found into those three.
///
/// A chain is walked down from its root so that every link can be checked against the
/// cumulative history the walk has assembled: a segment must start where its parent
/// ended, must name that parent's identity with both hashes, and must itself be named
/// by the history that leads to it. Anything a walk never reaches — a missing parent,
/// a mismatched hash, a span that does not continue its parent's, a chain deeper than
/// [`MAX_CHAIN_DEPTH`] — is an orphan. A cycle is unreachable by construction: the
/// walk starts only at roots, and a root has no parent to close a loop with.
///
/// `unreadable` names the files whose headers raised an I/O fault. Everything behind
/// one of them looks exactly like an orphan — its parent is not among the segments the
/// scan could read — and deleting it for that would let one unreadable file take a
/// whole subtree of intact ones with it. So those are held instead: nothing is learned
/// about a file from a fault on another, and the next scan links them up if the fault
/// was transient.
fn link_up(found: Vec<Found>, unreadable: &[u64]) -> Linked {
    let mut children: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut accepted = vec![false; found.len()];
    let mut waiting = vec![false; found.len()];
    let mut stack: Vec<(usize, Option<ChainId>, usize, usize)> = Vec::new();
    for (idx, entry) in found.iter().enumerate() {
        match entry.header.parent {
            Some(parent) => children.entry(parent.name).or_default().push(idx),
            None => stack.push((idx, None, 0, 1)),
        }
    }
    // The cumulative history of the chain being walked, extended on the way down and
    // truncated back on the way across (see `walk_chains` for why a depth-first pop
    // order makes that sound).
    let mut history: Vec<u32> = Vec::new();
    while let Some((idx, parent, base, depth)) = stack.pop() {
        history.truncate(base);
        let entry = &found[idx];
        if depth > MAX_CHAIN_DEPTH
            || entry.header.start != history.len()
            || entry.header.parent != parent.map(|id| id.as_parent())
        {
            continue;
        }
        history.extend_from_slice(&entry.header.tokens);
        let id = chain_id(&history);
        if id.name != entry.name {
            continue;
        }
        accepted[idx] = true;
        let below = history.len();
        for child in children.get(&entry.name).into_iter().flatten() {
            stack.push((*child, Some(id), below, depth + 1));
        }
    }

    // Everything behind a file that could not be read, however deep. The verified
    // chains are already settled above; this only decides which of the REST are held
    // rather than deleted, so no link is being trusted here — just the shape of what
    // is waiting on what.
    let mut blocked: Vec<u64> = unreadable.to_vec();
    let mut depth = 0;
    while let Some(name) = blocked.pop() {
        depth += 1;
        if depth > found.len() + unreadable.len() {
            break;
        }
        for idx in children.get(&name).into_iter().flatten() {
            if accepted[*idx] || waiting[*idx] {
                continue;
            }
            waiting[*idx] = true;
            blocked.push(found[*idx].name);
        }
    }

    let mut linked = Vec::new();
    let mut held = Vec::new();
    let mut orphans = Vec::new();
    for (idx, entry) in found.into_iter().enumerate() {
        match (accepted[idx], waiting[idx]) {
            (true, _) => linked.push(entry),
            (false, true) => held.push(entry),
            (false, false) => orphans.push(entry),
        }
    }
    Linked {
        linked,
        held,
        orphans,
    }
}

/// Remove a write nobody finished. Reported rather than silent — a `.tmp` that will
/// not go away is bytes the budget cannot account for — and never fatal.
fn sweep(shared: &Arc<Shared>, path: &Path) {
    if let Err(e) = std::fs::remove_file(path) {
        shared.failed(
            "sweeping an unfinished write",
            format!("{}: {e}", path.display()),
        );
    }
}

/// The plain files in one directory, reporting what could not be read rather than
/// silently indexing a short listing.
fn files(entries: std::fs::ReadDir, shared: &Arc<Shared>) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for entry in entries {
        match entry {
            Ok(entry) => {
                let path = entry.path();
                if path.is_file() {
                    found.push(path);
                }
            }
            Err(e) => shared.failed("listing the segment directory", e),
        }
    }
    found
}

fn modified(path: &Path) -> SystemTime {
    std::fs::metadata(path)
        .and_then(|meta| meta.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

/// Stamp `path`'s modification time, which is the tier's LRU clock.
fn touch(path: &Path, at: SystemTime) -> std::io::Result<()> {
    File::options()
        .write(true)
        .open(path)?
        .set_times(FileTimes::new().set_modified(at))
}

fn writer_loop(rx: Receiver<Message>, shared: Arc<Shared>) {
    while let Ok(message) = rx.recv() {
        match message {
            // One wake, one write. A wake left over from a request that was replaced
            // before it was claimed finds nothing, which is how coalescing shows up
            // here.
            Message::Wake => {
                if let Some(request) = claim(&shared) {
                    write_one(&shared, &request);
                }
            }
            // The barrier the shutdown path waits on, so everything pending has to be
            // written before it is acknowledged rather than everything sent so far.
            Message::Flush(ack) => {
                while let Some(request) = claim(&shared) {
                    write_one(&shared, &request);
                }
                let _ = ack.send(());
            }
        }
    }
}

/// Take the oldest pending write, so slots are served in the order they were paged
/// out. Claiming removes it: the map holds work nobody has started, which is what
/// makes replacing an entry a coalesce rather than a race with the writer.
fn claim(shared: &Arc<Shared>) -> Option<WriteRequest> {
    let mut store = shared.store();
    let slot = store
        .pending
        .iter()
        .min_by_key(|(_, request)| request.stamp)
        .map(|(slot, _)| *slot)?;
    store.pending.remove(&slot)
}

/// What one conversation's write does to the tree.
enum Plan {
    /// Add a tail `[at, len)` under the chain that already covers `[0, at)`.
    Tail { at: usize },
    /// Re-partition a stored segment at `at` first, so this conversation and the one
    /// already stored share everything before it.
    Split { path: PathBuf, at: usize },
    /// Rewrite the chain's last segment as a longer one starting at `at`, because
    /// chaining another onto it would push the chain past its depth cap.
    Rebase { at: usize },
}

fn plan_write(located: &Located) -> Plan {
    match &located.diverged {
        // A split does not just add the base and this conversation's tail: it pushes
        // the segment it cuts, and EVERYTHING already chained behind that segment, one
        // level further from the root. So the chain that has to fit afterwards is the
        // longest one through the cut — `depth` segments in front of it plus `height`
        // behind it — with room for the level the split adds. A subtree that cannot
        // afford it is left uncut, and this conversation writes a tail overlapping it:
        // worse dedup and nothing else, where splitting anyway would leave existing
        // segments deeper than the next scan will walk, and it would delete them.
        Some(Divergence { path, at, height }) if located.depth + height < MAX_CHAIN_DEPTH => {
            Plan::Split {
                path: path.clone(),
                at: *at,
            }
        }
        _ => match located.tail {
            Some((_, start)) if located.depth + 1 > MAX_CHAIN_DEPTH => Plan::Rebase { at: start },
            _ => Plan::Tail {
                at: located.boundary,
            },
        },
    }
}

/// Write one conversation's tail, splitting a stored segment first when it forks
/// inside one, then collect what the write made redundant and whatever the budget no
/// longer has room for.
fn write_one(shared: &Arc<Shared>, request: &WriteRequest) {
    let history = request.tokens.as_slice();
    let located = {
        let store = shared.store();
        locate(&store, history)
    };
    // Re-decided here and not only at enqueue: another slot's write may have landed in
    // between and already stored this prefix.
    if !worth_writing(&located, history.len()) {
        return;
    }
    let mut fresh: Vec<PathBuf> = Vec::new();
    let at = match plan_write(&located) {
        Plan::Tail { at } | Plan::Rebase { at } => at,
        // A split that cannot be completed is not a failure: the tail below still goes
        // under the deepest boundary that IS stored, which leaves two segments whose
        // spans overlap — imperfect dedup, never a wrong resume.
        Plan::Split { path, at } => match split_segment(shared, &path, at, request, &mut fresh) {
            Some(()) => {
                shared.logger.log(ServeLog::DiskSegmentSplit { at });
                at
            }
            None => located.boundary,
        },
    };
    let parent = (at > 0).then(|| chain_id(&history[..at]).as_parent());
    // The resume point at the boundary belongs to the segment ENDING there, which the
    // tail below is about to be chained onto without touching. Before that, so that
    // every intermediate state stays a cache somebody can use.
    backfill_boundary(shared, at, request, &mut fresh);

    let rows = match request.full_kv.range(at, history.len()) {
        Ok(rows) => rows,
        Err(e) => {
            shared.failed("cutting a conversation's tail rows", format!("{e:#}"));
            return;
        }
    };
    let snapshots: Vec<(usize, &HostSnapshot)> = request
        .snapshots
        .iter()
        .filter(|(pos, _)| *pos > at && *pos <= history.len())
        .map(|(pos, rings)| (*pos, rings.as_ref()))
        .collect();
    let drafter = request
        .drafter
        .as_deref()
        .filter(|planes| planes.pos > at && planes.pos <= history.len());
    let path = shared.path_for(&chain_id(history));
    if !put_segment(
        shared,
        &path,
        at,
        parent,
        history,
        &rows,
        &snapshots,
        drafter,
        DiskSegmentRole::Tail,
        &mut fresh,
    ) {
        return;
    }

    {
        let mut store = shared.store();
        // Only while the slot still holds the conversation this covers. A slot
        // repurposed since the write was claimed gets no link back, or the budget would
        // guard these files on behalf of a conversation nothing is serving.
        let current = store.epochs.get(&request.slot).copied().unwrap_or_default();
        if current == request.epoch {
            store.links.insert(request.slot, path.clone());
        }
    }
    for stored in superseded(shared, history, &fresh) {
        condemn(shared, &stored, DiskEvictReason::Superseded);
    }
    enforce_budget(shared, shared.budget, &fresh);
}

/// Put the conversation's snapshot at `at` into the stored segment that ENDS there,
/// when that segment has none.
///
/// A tail starting at `at` cannot hold it — a snapshot at a span's start restores to
/// the state the span before it ends in, so it belongs to that one — and a tail write
/// otherwise never touches its parent. Which is fine as long as the parent has a
/// resume point of its own there, and two populations of stored segments do not: ones
/// written before this position was ever a boundary anybody stopped at, and ones whose
/// write was cut short. For those, every conversation forking at that boundary drops
/// its snapshot and the boundary stays unresumable forever, however many times it is
/// forked. Backfilling is what heals them, once per boundary.
///
/// The parent is rewritten under its OWN name — the same-name rewrite a split already
/// does, and safe for the same reason: a name binds a token history, not a file's
/// bytes, so every child still resolves to it, and a segment that gains a snapshot
/// covers strictly more than it did. A crash between this and the tail below leaves a
/// parent with one more resume point and no tail, which is an ordinary cache.
fn backfill_boundary(
    shared: &Arc<Shared>,
    at: usize,
    request: &WriteRequest,
    fresh: &mut Vec<PathBuf>,
) {
    let history = request.tokens.as_slice();
    let Some((_, rings)) = request.snapshots.iter().find(|(pos, _)| *pos == at) else {
        return;
    };
    if at == 0 {
        return;
    }
    let id = chain_id(&history[..at]);
    let path = shared.path_for(&id);
    {
        // Only when the segment is one this store knows and it really lacks the
        // position: the rewrite moves the whole span's rows, which is gigabytes for a
        // system prompt, and is worth paying exactly once.
        let store = shared.store();
        let Some(parent) = store.segment_at(&path) else {
            return;
        };
        if parent.snapshots.contains(&at) || parent.snapshots.len() >= MAX_STORED_SNAPSHOTS {
            return;
        }
    }
    let segment =
        match disk_cache::read_segment(&path, &shared.checkpoint, TOKENIZATION_RULES_VERSION) {
            Ok(segment) => segment,
            Err(e) => {
                shared.failed("reading a segment to store a resume point in it", &e);
                if deletable(e.class()) {
                    condemn(shared, &path, DiskEvictReason::Invalid { class: e.class() });
                }
                return;
            }
        };
    // The file is the authority, and it may have been re-partitioned since the index
    // read it. Anything but the span this conversation's history says ends here is left
    // alone.
    if segment.end() != at || segment.tokens != history[segment.start..at] {
        return;
    }
    if segment.snapshots.iter().any(|(pos, _)| *pos == at) {
        return;
    }
    let mut snapshots: Vec<(usize, &HostSnapshot)> = segment
        .snapshots
        .iter()
        .map(|(pos, stored)| (*pos, stored))
        .collect();
    snapshots.push((at, rings.as_ref()));
    if put_segment(
        shared,
        &path,
        segment.start,
        segment.parent,
        &history[..at],
        &segment.full_kv,
        &snapshots,
        segment.drafter.as_ref(),
        DiskSegmentRole::Base,
        fresh,
    ) {
        shared
            .logger
            .log(ServeLog::DiskBoundarySnapshotStored { at });
    }
}

/// Cut the stored segment at `path` in two at `at`, so the conversation stored in it
/// and the one being written share everything before that position.
///
/// Two writes make a split, and their order is the crash-safety argument: the base
/// `[p, at)` under a name nothing points at yet, then the segment's own span rewritten
/// as `[at, n)` under its EXISTING name — which is unchanged, because a name is a hash
/// of the cumulative history and the tail still ends where it did. Every prefix of that
/// order is a consistent cache: after the base alone, the un-split segment is still
/// there and the base is an unreferenced duplicate of its head; after the rewrite,
/// every child of the old segment still resolves to it by name, which is why an
/// interior split rewrites no child.
///
/// `None` means nothing was cut and the caller falls back to the deepest boundary that
/// is stored. `Some` means the base is on disk and is the parent to build on, whether
/// or not the rewrite that follows it succeeded.
fn split_segment(
    shared: &Arc<Shared>,
    path: &Path,
    at: usize,
    request: &WriteRequest,
    fresh: &mut Vec<PathBuf>,
) -> Option<()> {
    let history = request.tokens.as_slice();
    let name = disk_cache::segment_name_of(path)?;
    let segment =
        match disk_cache::read_segment(path, &shared.checkpoint, TOKENIZATION_RULES_VERSION) {
            Ok(segment) => segment,
            Err(e) => {
                shared.failed("reading a segment to split it", &e);
                if deletable(e.class()) {
                    condemn(shared, path, DiskEvictReason::Invalid { class: e.class() });
                }
                return None;
            }
        };
    // The file is the authority on what it holds, and it may have been replaced since
    // the index read its header — by a split of its own. A cut that no longer applies
    // is abandoned rather than forced.
    let cut = at.checked_sub(segment.start)?;
    let inside = segment.start < at && at < segment.end() && at < history.len();
    if !inside || segment.tokens[..cut] != history[segment.start..at] {
        return None;
    }
    // The tail keeps this file's name, which is the property the whole tree rests on.
    // Checked before a byte is written, so a naming rule that ever stopped agreeing
    // with itself costs a skipped split rather than a file nothing can reach.
    let base = chain_id(&history[..at]);
    let mut tail_history = history[..at].to_vec();
    tail_history.extend_from_slice(&segment.tokens[cut..]);
    if chain_id(&tail_history).name != name {
        shared.failed(
            "splitting a stored segment",
            format!("the tail of {} would not keep its own name", path.display()),
        );
        return None;
    }

    let (base_rows, tail_rows) = match (
        segment.full_kv.range(0, cut),
        segment.full_kv.range(cut, segment.tokens.len()),
    ) {
        (Ok(base_rows), Ok(tail_rows)) => (base_rows, tail_rows),
        _ => {
            shared.failed(
                "splitting a stored segment",
                format!("the rows of {} could not be cut at {at}", path.display()),
            );
            return None;
        }
    };
    let mut base_snapshots: Vec<(usize, &HostSnapshot)> = segment
        .snapshots
        .iter()
        .filter(|(pos, _)| *pos <= at)
        .map(|(pos, rings)| (*pos, rings))
        .collect();
    // The branch-point snapshot the arriving conversation took, when the stored
    // segment has none there: it is what makes the boundary cheap to resume for
    // everything that forks here later.
    let branch = request
        .snapshots
        .iter()
        .find(|(pos, _)| *pos == at)
        .filter(|_| !base_snapshots.iter().any(|(pos, _)| *pos == at));
    if let Some((pos, rings)) = branch {
        base_snapshots.push((*pos, rings.as_ref()));
    }
    let tail_snapshots: Vec<(usize, &HostSnapshot)> = segment
        .snapshots
        .iter()
        .filter(|(pos, _)| *pos > at)
        .map(|(pos, rings)| (*pos, rings))
        .collect();

    let base_path = shared.path_for(&base);
    if !put_segment(
        shared,
        &base_path,
        segment.start,
        segment.parent,
        &history[..at],
        &base_rows,
        &base_snapshots,
        segment.drafter.as_ref().filter(|planes| planes.pos <= at),
        DiskSegmentRole::Base,
        fresh,
    ) {
        return None;
    }
    // The rewrite can fail without costing anything: the base is a valid segment and
    // the un-split file is still a valid child of the same parent.
    put_segment(
        shared,
        path,
        at,
        Some(base.as_parent()),
        &tail_history,
        &tail_rows,
        &tail_snapshots,
        segment.drafter.as_ref().filter(|planes| planes.pos > at),
        DiskSegmentRole::Tail,
        fresh,
    );
    Some(())
}

/// Write one segment and index it. `history` is the CUMULATIVE history the segment
/// ends at — its span is what follows `start` in it, and its name is that history's
/// hash, which is what `path` must already be.
#[allow(clippy::too_many_arguments)]
fn put_segment(
    shared: &Arc<Shared>,
    path: &Path,
    start: usize,
    parent: Option<ParentRef>,
    history: &[u32],
    rows: &HostFullKv,
    snapshots: &[(usize, &HostSnapshot)],
    drafter: Option<&DrafterImage>,
    role: DiskSegmentRole,
    fresh: &mut Vec<PathBuf>,
) -> bool {
    let span = &history[start..];
    // Renaming a segment into place destroys whatever stood under that name, and the
    // name is a 64-bit hash: two histories that ever collided would have one quietly
    // overwrite the other, orphaning every chain behind it. So the file being replaced
    // has to be the same conversation, checked on the tokens themselves — the same
    // authority the load path uses, applied on the side that does the destroying.
    // Every legitimate rewrite passes: a conversation storing its tail again, a split
    // laying its tail back down under its own name, a base rewritten over an
    // interrupted split's.
    {
        let store = shared.store();
        if store.index.contains_key(path) {
            let stored = store.cumulative(path);
            if stored.as_deref() != Some(history) {
                shared.failed(
                    "writing a segment",
                    format!(
                        "{} already holds a different conversation; it is kept and this segment \
                         is not written",
                        path.display()
                    ),
                );
                return false;
            }
        }
    }
    let started = Instant::now();
    let bytes = match disk_cache::write_segment(
        path,
        &shared.checkpoint,
        TOKENIZATION_RULES_VERSION,
        start,
        parent,
        span,
        rows,
        snapshots,
        drafter,
    ) {
        Ok(bytes) => bytes,
        Err(e) => {
            shared.failed("writing a segment", format!("{e:#}"));
            return false;
        }
    };
    let ms = started.elapsed().as_millis() as u64;
    let replaced = {
        let mut store = shared.store();
        store.index.insert(
            path.to_path_buf(),
            Entry {
                segment: Some(IndexedSegment {
                    name: chain_id(history).name,
                    start,
                    parent,
                    span: span.to_vec(),
                    snapshots: snapshots.iter().map(|(pos, _)| *pos).collect(),
                }),
                // Freshly written, so whatever was demoted under this name is settled:
                // these are bytes this process just laid down.
                standing: Standing::Ready,
                bytes,
                mtime: SystemTime::now(),
            },
        )
    };
    fresh.push(path.to_path_buf());
    shared.logger.log(ServeLog::DiskSegmentWritten {
        role,
        tokens: span.len(),
        bytes,
        ms,
    });
    if let Some(old) = replaced {
        // The name was reused, so the file it replaced is already gone: the rename put
        // this segment over it. Reported anyway, so a consumer counting what is on disk
        // stays in step.
        shared.logger.log(ServeLog::DiskSegmentEvicted {
            reason: DiskEvictReason::Superseded,
            bytes: old.bytes,
        });
    }
    true
}

/// The segments this write made redundant: an earlier, shorter tail of the SAME
/// conversation that nothing chains onto any more.
///
/// A segment qualifies when the history that names it is a strict prefix of the one
/// just written — which is exactly what its name being that prefix's hash says — and
/// it is childless and unlinked. The shape that produces one is a chain at its depth
/// cap, whose last segment was rewritten as a longer span; a chain that simply grew
/// leaves its old tail as the new one's PARENT, and a parent is never redundant.
///
/// A file another warm slot is linked to is left alone even when this history extends
/// it: that slot's conversation is still being served and this is what it would come
/// back from after a restart — the same reason the budget pass never takes a linked
/// file.
///
/// Only NAMES them. The deleting is the caller's, through `condemn`, so that a file
/// which refuses to be deleted stays counted rather than being forgotten while it is
/// still occupying the disk.
fn superseded(shared: &Arc<Shared>, history: &[u32], fresh: &[PathBuf]) -> Vec<PathBuf> {
    let store = shared.store();
    let doomed: Vec<PathBuf> = store
        .index
        .iter()
        .filter(|(path, entry)| {
            let Some(segment) = entry.segment() else {
                return false;
            };
            // The name narrows the field for free; the tokens then decide. A name is a
            // 64-bit hash, and this is a DELETION — the one operation here that cannot
            // be undone — so it is not allowed to rest on two histories never colliding.
            !fresh.iter().any(|written| written == *path)
                && segment.end() < history.len()
                && chain_id(&history[..segment.end()]).name == segment.name
                && store.cumulative(path).as_deref() == Some(&history[..segment.end()])
                && !store.has_children(segment.name)
                && !store.is_linked(path)
        })
        .map(|(path, _)| path.clone())
        .collect();
    doomed
}

/// Delete segments, oldest use first, until the store fits `budget` bytes.
///
/// Only CHILDLESS segments are evictable, and that structural rule is what pins a
/// shared prefix for free: a base with living children survives any budget pressure,
/// while deleting a tail may leave its parent childless and therefore eligible in the
/// same pass — the cascade that eventually reclaims a whole abandoned chain, leaf
/// first.
///
/// `keep` is what the write just laid down, which is never the answer to being over
/// budget — nor is any file a warm slot is linked to. With nothing else left to
/// delete the store simply stays over: a budget smaller than the conversations
/// currently being served cannot be honored by throwing away the segments of the
/// conversations being served.
///
/// The file goes before the index entry does. An entry dropped for a file that is
/// still there would make the accounting claim room the disk does not have, and
/// every later pass would measure against a store it no longer describes; a file
/// that refuses to be deleted stays indexed, is reported, and is passed over for the
/// rest of this pass so a stubborn one cannot spin the loop.
fn enforce_budget(shared: &Arc<Shared>, budget: u64, keep: &[PathBuf]) {
    let mut stuck: Vec<PathBuf> = Vec::new();
    loop {
        let victim = {
            let store = shared.store();
            let total: u64 = store
                .index
                .values()
                .fold(0u64, |sum, entry| sum.saturating_add(entry.bytes));
            if total <= budget {
                return;
            }
            store
                .index
                .iter()
                .filter(|(path, _)| !keep.iter().any(|kept| kept == *path))
                .filter(|(path, _)| !stuck.iter().any(|failed| failed == *path))
                .filter(|(path, _)| !store.is_linked(path))
                // A segment waiting on a file that could not be read is not the
                // budget's to take either. Its own bytes are fine and nothing has been
                // learned against it; it is only unreachable because of a fault
                // somewhere in front of it, and a fault is never a reason to delete.
                .filter(|(_, entry)| entry.standing != Standing::Held)
                .filter(|(_, entry)| {
                    entry
                        .segment()
                        .is_none_or(|segment| !store.has_children(segment.name))
                })
                // The path breaks a tie: two files written in the same clock tick
                // must not make the choice depend on hash order.
                .min_by(|a, b| (a.1.mtime, a.0).cmp(&(b.1.mtime, b.0)))
                .map(|(path, entry)| (path.clone(), entry.bytes))
        };
        let Some((path, bytes)) = victim else {
            return;
        };
        if delete(shared, &path, bytes, DiskEvictReason::Budget) {
            shared.store().index.remove(&path);
        } else {
            stuck.push(path);
        }
    }
}

/// Delete a file for a verdict about its CONTENTS, and only then forget it.
///
/// That order is the invariant every deletion here keeps: an index entry dropped for a
/// file still on disk makes the budget claim room the disk does not have, and every
/// later pass measures against a store it no longer describes. So the file goes first,
/// and the entry follows only if it went. A file that refuses to be deleted stays
/// counted — but nothing is built on it or on anything behind it, because whatever was
/// wrong with it is still wrong.
///
/// `true` when the file is gone (including "was already gone", since the point was for
/// it not to be there).
fn condemn(shared: &Arc<Shared>, path: &Path, reason: DiskEvictReason) -> bool {
    let bytes = shared
        .store()
        .index
        .get(path)
        .map(|entry| entry.bytes)
        .unwrap_or_else(|| std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0));
    let gone = delete(shared, path, bytes, reason);
    let mut store = shared.store();
    store.demote_at(path);
    if gone {
        store.index.remove(path);
        store.links.retain(|_, linked| linked != path);
    }
    gone
}

/// Delete one segment and say so. `false` means the file is still on disk, so a caller
/// keeping accounts should keep counting it; a file that was already gone is a
/// success, since the point was for it not to be there.
fn delete(shared: &Arc<Shared>, path: &Path, bytes: u64, reason: DiskEvictReason) -> bool {
    match std::fs::remove_file(path) {
        Ok(()) => {
            shared
                .logger
                .log(ServeLog::DiskSegmentEvicted { reason, bytes });
            true
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
        Err(e) => {
            shared.failed("deleting a segment", format!("{}: {e}", path.display()));
            false
        }
    }
}

/// Fold a verified chain into the one image a cache slot installs: the spans' rows
/// re-strided into a single block, every snapshot in chain order, and the deepest
/// drafter image any segment carried.
fn compose(chain: Vec<Read>) -> anyhow::Result<DiskImage> {
    let mut tokens: Vec<u32> = Vec::new();
    let mut spans: Vec<HostFullKv> = Vec::new();
    let mut snapshots: Vec<(usize, HostSnapshot)> = Vec::new();
    let mut drafter: Option<DrafterImage> = None;
    for Read { segment, .. } in chain {
        tokens.extend_from_slice(&segment.tokens);
        spans.push(segment.full_kv);
        snapshots.extend(segment.snapshots);
        // Only one set of drafter planes can be installed, so the deepest wins: it is
        // the only one a resume at the chain's tip could use, and the tip is where a
        // resume normally lands.
        //
        // Deliberately kind-blind, because the restore point is not known here — it is
        // chosen against the request's own prefix match, long after assembly — so there
        // is nothing better to select on. For a DFlash image the choice is free: the
        // deepest backs every resume at or below it. For an MTP image, which backs only
        // its own exact position (`drafter_planes_usable`), a resume short of the tip
        // finds these planes unusable and decodes plain, where a shallower set might
        // coincidentally have matched. That is the same limitation the head's single
        // carry hidden imposes everywhere else, and it is on the ledger with the rest.
        let deeper = segment
            .drafter
            .filter(|planes| drafter.as_ref().is_none_or(|held| held.pos < planes.pos));
        if deeper.is_some() {
            drafter = deeper;
        }
    }
    Ok(DiskImage {
        tokens,
        full_kv: HostFullKv::concat(spans)?,
        snapshots,
        drafter,
    })
}

fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::HostLayerSnapshot;

    /// Shapes small enough to write hundreds of, and all different from each other
    /// so a transposed field cannot pass on byte counts alone.
    const N_KV: usize = 2;
    const HEAD_DIM: usize = 4;
    const WINDOW: usize = 3;
    const KINDS: [bool; 3] = [true, false, true];

    fn pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len).map(|i| (i as u8) ^ seed ^ 0x5a).collect()
    }

    /// A conversation's rows, with every byte keyed to the POSITION it covers, so
    /// rows cut or composed at the wrong offset fail on content rather than length.
    fn full_kv(pos: usize) -> Arc<HostFullKv> {
        let plane = |seed: u8| {
            let mut bytes = Vec::new();
            for head in 0..N_KV {
                for p in 0..pos {
                    for i in 0..HEAD_DIM * size_of::<half::f16>() {
                        bytes.push(
                            (p as u8).wrapping_mul(7) ^ ((head as u8) << 5) ^ seed ^ (i as u8),
                        );
                    }
                }
            }
            bytes
        };
        let planes = KINDS
            .iter()
            .filter(|full| **full)
            .enumerate()
            .map(|(il, _)| (plane(il as u8), plane(0x40 + il as u8)))
            .collect();
        Arc::new(HostFullKv::new(pos, N_KV, HEAD_DIM, planes, Vec::new(), 0).unwrap())
    }

    /// Drafter rows covering `[0, pos)`, in the shape the drafter's own cache has.
    fn drafter(pos: usize) -> DrafterImage {
        let plane = N_KV * pos * HEAD_DIM * size_of::<f32>();
        let planes = (0..2u8)
            .map(|il| (pattern(plane, 0x80 + il), pattern(plane, 0xc0 + il)))
            .collect();
        DrafterImage::new_dflash(pos, N_KV, HEAD_DIM, planes).unwrap()
    }

    fn snapshot(pos: usize) -> Arc<HostSnapshot> {
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
        Arc::new(HostSnapshot::new(pos, layers).unwrap())
    }

    /// A conversation's token ids: `mark` tells two conversations apart, and every
    /// prefix of one history is a valid earlier turn of it.
    fn tokens(mark: u32, len: usize) -> Vec<u32> {
        (0..len as u32).map(|i| mark * 100_000 + i).collect()
    }

    /// A conversation that shares `at` tokens with `mark`'s and then goes its own way.
    fn forked(mark: u32, at: usize, len: usize) -> Vec<u32> {
        let mut ids = tokens(mark, at);
        ids.extend((0..(len - at) as u32).map(|i| 900_000 + i));
        ids
    }

    fn bytes_of(rows: &HostFullKv) -> Vec<u8> {
        let mut out = Vec::new();
        rows.write_to(&mut out).unwrap();
        out
    }

    /// A tier rooted in a directory of this test's own, torn down with the guard so
    /// a failing assertion leaves nothing for the next run to index.
    struct Tier {
        shared: Arc<Shared>,
        dir: PathBuf,
    }

    impl Tier {
        fn new(label: &str, budget_bytes: u64) -> Self {
            Self::with_logger(label, budget_bytes, ServeLogger::discarding())
        }

        /// A tier whose reports are kept, for the tests that assert WHY something was
        /// refused and not only that it was.
        fn watched(label: &str, budget_bytes: u64) -> (Self, super::super::log::EventLog) {
            let (logger, events) = super::super::log::collecting();
            (Self::with_logger(label, budget_bytes, logger), events)
        }

        fn with_logger(label: &str, budget_bytes: u64, logger: ServeLogger) -> Self {
            let dir =
                std::env::temp_dir().join(format!("xwen_disk_tier_{}_{label}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let checkpoint = CheckpointId::from_parts(0x1234_5678_9abc_def0, 4096);
            let root = dir.join(KV_SUBDIR);
            let segments = root.join(checkpoint.dir_name());
            std::fs::create_dir_all(&segments).unwrap();
            Self {
                shared: Arc::new(Shared {
                    store: Mutex::new(Store::default()),
                    root,
                    dir: segments,
                    checkpoint,
                    budget: budget_bytes,
                    logger,
                }),
                dir,
            }
        }

        /// Write one conversation the way the writer thread does, and return the file
        /// its tail landed in.
        fn write(&self, slot: usize, tokens: &[u32], snapshots: &[usize]) -> PathBuf {
            write_one(&self.shared, &self.request(slot, tokens, snapshots));
            self.path(tokens)
        }

        /// The file the segment ending a history is named by.
        fn path(&self, history: &[u32]) -> PathBuf {
            self.shared.path_for(&chain_id(history))
        }

        fn request(&self, slot: usize, tokens: &[u32], snapshots: &[usize]) -> WriteRequest {
            WriteRequest {
                slot,
                stamp: 1,
                epoch: self
                    .shared
                    .store()
                    .epochs
                    .get(&slot)
                    .copied()
                    .unwrap_or_default(),
                tokens: tokens.to_vec(),
                full_kv: full_kv(tokens.len()),
                snapshots: snapshots.iter().map(|pos| (*pos, snapshot(*pos))).collect(),
                drafter: None,
            }
        }

        /// A tier the engine holds, over this test's store. The channel's receiver is
        /// returned so a test can see what was enqueued without a writer thread
        /// draining it.
        fn engine_side(&self, min_tokens: usize) -> (DiskCache, Receiver<Message>) {
            let (tx, rx) = channel();
            (
                DiskCache {
                    tx,
                    shared: Arc::clone(&self.shared),
                    min_tokens,
                    stamp: Cell::new(0),
                    trusted: Cell::new(true),
                },
                rx,
            )
        }

        fn files(&self) -> Vec<PathBuf> {
            let mut found: Vec<PathBuf> = std::fs::read_dir(&self.shared.dir)
                .unwrap()
                .flatten()
                .map(|entry| entry.path())
                .collect();
            found.sort();
            found
        }

        fn indexed(&self) -> usize {
            self.shared.store().index.len()
        }

        /// What the index says a stored segment covers: its span and its snapshots.
        fn span(&self, path: &Path) -> (usize, usize, Vec<usize>) {
            let store = self.shared.store();
            let segment = store.segment_at(path).expect("an indexed segment");
            (segment.start, segment.end(), segment.snapshots.clone())
        }

        fn read(&self, path: &Path) -> Segment {
            disk_cache::read_segment(path, &self.shared.checkpoint, TOKENIZATION_RULES_VERSION)
                .expect("a readable segment")
        }

        /// A second tier over the same directory, as a restart sees it.
        fn restarted(&self) -> Arc<Shared> {
            let fresh = Arc::new(Shared {
                store: Mutex::new(Store::default()),
                root: self.shared.root.clone(),
                dir: self.shared.dir.clone(),
                checkpoint: self.shared.checkpoint,
                budget: 1 << 40,
                logger: ServeLogger::discarding(),
            });
            scan(&fresh);
            fresh
        }
    }

    impl Drop for Tier {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// A first write is one root segment covering the whole conversation, indexed and
    /// readable back through the same binding it was written with.
    #[test]
    fn a_written_conversation_is_one_root_segment() {
        let tier = Tier::new("write", 1 << 30);
        let ids = tokens(1, 400);
        let path = tier.write(0, &ids, &[200, 400]);
        assert_eq!(tier.files(), vec![path.clone()]);
        assert_eq!(tier.indexed(), 1);
        assert_eq!(tier.span(&path), (0, 400, vec![200, 400]));

        let segment = tier.read(&path);
        assert_eq!(segment.start, 0);
        assert_eq!(segment.parent, None);
        assert_eq!(segment.tokens, ids);
        assert_eq!(segment.full_kv.pos, 400);
        assert_eq!(
            segment
                .snapshots
                .iter()
                .map(|(pos, _)| *pos)
                .collect::<Vec<_>>(),
            vec![200, 400]
        );
        // The slot is linked to its tail, which is what makes the next write from it
        // continue this chain rather than start a second copy of it.
        assert_eq!(tier.shared.store().links.get(&0), Some(&path));
    }

    /// A conversation that grew writes only its TAIL: what is already on disk covers
    /// the turns before it and becomes the new segment's parent, untouched.
    #[test]
    fn a_grown_conversation_writes_only_its_tail() {
        let tier = Tier::new("grow", 1 << 30);
        let first = tokens(1, 400);
        let base = tier.write(0, &first, &[400]);
        let base_bytes = std::fs::read(&base).unwrap();

        let second = tokens(1, 800);
        let tail = tier.write(0, &second, &[400, 800]);
        assert_ne!(tail, base);
        assert!(base.exists(), "the earlier turns are not rewritten");
        assert!(
            std::fs::read(&base).unwrap() == base_bytes,
            "and not one byte of them is touched"
        );
        assert_eq!(tier.span(&tail), (400, 800, vec![800]));
        let segment = tier.read(&tail);
        assert_eq!(segment.tokens, second[400..]);
        assert_eq!(segment.full_kv.pos, 400);
        assert_eq!(
            segment.parent,
            Some(chain_id(&first).as_parent()),
            "the tail names the segment in front of it"
        );
        assert_eq!(tier.shared.store().links.get(&0), Some(&tail));
    }

    /// Two conversations behind one system prompt store that prompt ONCE: the second
    /// splits the segment they diverge inside, so what they share becomes a base with
    /// two tails hanging off it — and the split leaves the first conversation's tail
    /// under its own name, which is what keeps every reference to a split segment
    /// valid without rewriting anything.
    #[test]
    fn a_fork_splits_the_shared_prefix_into_a_base_with_two_tails() {
        let tier = Tier::new("fork", 1 << 40);
        let mine = tokens(1, 800);
        let stored = tier.write(0, &mine, &[400, 800]);
        let before = tier.read(&stored);

        // Another conversation with the same 400-token opening.
        let theirs = forked(1, 400, 700);
        let their_tail = tier.write(1, &theirs, &[400, 700]);

        let base = tier.path(&mine[..400]);
        assert!(base.exists(), "the shared prefix is its own segment now");
        assert_eq!(tier.span(&base), (0, 400, vec![400]));
        assert_eq!(
            stored,
            tier.path(&mine),
            "and the first conversation's tail kept its name"
        );
        assert_eq!(tier.span(&stored), (400, 800, vec![800]));
        assert_eq!(tier.span(&their_tail), (400, 700, vec![700]));
        assert_eq!(tier.files().len(), 3, "one base, two tails");

        // Both tails point at the base, and the base at nothing.
        let base_segment = tier.read(&base);
        assert_eq!(base_segment.parent, None);
        assert_eq!(base_segment.tokens, mine[..400]);
        let parent = Some(chain_id(&mine[..400]).as_parent());
        assert_eq!(tier.read(&stored).parent, parent);
        assert_eq!(tier.read(&their_tail).parent, parent);

        // The rows of the conversation that was split are exactly the rows it had, cut
        // in two: base then tail composes back into what was stored before the split.
        let after =
            HostFullKv::concat(vec![base_segment.full_kv, tier.read(&stored).full_kv]).unwrap();
        assert_eq!(bytes_of(&after), bytes_of(&before.full_kv));
        // The snapshot at the branch point went into the base, where everything forking
        // there can resume from it, and the deeper one stayed with the span holding its
        // rows.
        assert_eq!(
            tier.read(&stored)
                .snapshots
                .iter()
                .map(|(pos, _)| *pos)
                .collect::<Vec<_>>(),
            vec![800]
        );
        // A restart makes the same tree out of the files alone.
        let fresh = tier.restarted();
        assert_eq!(fresh.store().index.len(), 3);
        assert_eq!(fresh.store().chain_paths(&stored), vec![base, stored]);
    }

    /// A conversation forking at exactly a stored boundary puts its resume point INTO
    /// the segment ending there, when that segment has none.
    ///
    /// The tail it writes starts at the boundary and cannot hold a snapshot there — that
    /// position restores to the state the span in front of it ends in — so without this
    /// the snapshot is dropped and the boundary stays unresumable no matter how many
    /// conversations fork at it. Which is the state segments written before anybody
    /// stopped at that position are in, and the state a write cut short by a shutdown
    /// leaves behind.
    ///
    /// The parent is rewritten under its own name, so its children are untouched, and
    /// the pair of events a same-name rewrite produces keeps a consumer counting files
    /// in step.
    #[test]
    fn a_fork_at_a_boundary_stores_its_resume_point_in_the_segment_ending_there() {
        let (tier, events) = Tier::watched("backfill", 1 << 40);
        let (disk, _wakes) = tier.engine_side(0);
        // A base with no resume point at its own end, and a child hanging off it.
        let base_history = tokens(1, 400);
        let base = tier.write(0, &base_history, &[200]);
        let child = tier.write(0, &tokens(1, 800), &[800]);
        assert_eq!(tier.span(&base), (0, 400, vec![200]));
        let child_before = std::fs::read(&child).unwrap();
        let base_size = std::fs::metadata(&base).unwrap().len();
        let _ = events.drain();

        // Another conversation that leaves the stored one at exactly the boundary, and
        // that stopped there on the way past.
        let theirs = forked(1, 400, 800);
        let their_tail = tier.write(1, &theirs, &[400, 800]);

        assert_eq!(
            tier.span(&base),
            (0, 400, vec![200, 400]),
            "the boundary is resumable now"
        );
        assert_eq!(
            tier.read(&base).start,
            0,
            "and the segment is otherwise what it was"
        );
        assert!(
            std::fs::read(&child).unwrap() == child_before,
            "a same-name rewrite leaves every child alone"
        );
        assert_eq!(tier.span(&their_tail), (400, 800, vec![800]));

        // One write, one replacement: a consumer folding these events keeps counting
        // the same number of files.
        let (written, evicted) =
            events
                .drain()
                .into_iter()
                .fold((0, 0), |(w, e), event| match event {
                    ServeLog::DiskSegmentWritten { .. } => (w + 1, e),
                    ServeLog::DiskSegmentEvicted {
                        reason: DiskEvictReason::Superseded,
                        bytes,
                    } => {
                        assert_eq!(bytes, base_size, "the file the rewrite replaced");
                        (w, e + 1)
                    }
                    _ => (w, e),
                });
        assert_eq!((written, evicted), (2, 1), "the base rewrite and the tail");

        // And a conversation forking there now resumes at the boundary instead of
        // prefilling up to it.
        let mut another = forked(1, 400, 600);
        another.push(9001);
        assert_eq!(
            disk.candidate(&another, 0).expect("a candidate").resume,
            400
        );
    }

    /// The rewrite happens once. A boundary that already has a resume point costs
    /// nothing to fork at again — which matters, because the rewrite moves the whole
    /// span's rows, and a shared system prompt is gigabytes of them.
    #[test]
    fn a_boundary_that_can_already_be_resumed_is_not_rewritten() {
        let tier = Tier::new("backfill_once", 1 << 40);
        let base = tier.write(0, &tokens(1, 400), &[200, 400]);
        tier.write(0, &tokens(1, 800), &[800]);
        let before = std::fs::read(&base).unwrap();

        tier.write(1, &forked(1, 400, 800), &[400, 800]);
        assert!(
            std::fs::read(&base).unwrap() == before,
            "the segment already holds that position"
        );

        // Nor is anything rewritten for a conversation that never stopped at the
        // boundary: there is no resume point to store.
        let tier = Tier::new("backfill_none", 1 << 40);
        let base = tier.write(0, &tokens(1, 400), &[200]);
        tier.write(0, &tokens(1, 800), &[800]);
        let before = std::fs::read(&base).unwrap();
        tier.write(1, &forked(1, 400, 800), &[800]);
        assert!(std::fs::read(&base).unwrap() == before);
        assert_eq!(tier.span(&base), (0, 400, vec![200]));
    }

    /// An interior split — one where the segment being cut already has children —
    /// leaves those children's files untouched, byte for byte. That is the whole
    /// reason names are hashes of the cumulative history: the cut segment's tail ends
    /// where it always did, so it keeps its name and every reference to it stays
    /// valid.
    #[test]
    fn an_interior_split_rewrites_no_child() {
        let tier = Tier::new("interior", 1 << 40);
        // One conversation, three turns: [0,400), [400,800), [800,1200).
        tier.write(0, &tokens(1, 400), &[400]);
        let middle = tier.write(0, &tokens(1, 800), &[800]);
        let deepest = tier.write(0, &tokens(1, 1200), &[1200]);
        let child_before = std::fs::read(&deepest).unwrap();
        assert_eq!(tier.files().len(), 3);

        // A conversation that forks in the MIDDLE of the second segment.
        let theirs = forked(1, 600, 900);
        let their_tail = tier.write(1, &theirs, &[900]);

        assert!(
            std::fs::read(&deepest).unwrap() == child_before,
            "the child of the split segment is not rewritten"
        );
        assert_eq!(tier.span(&middle), (600, 800, vec![800]));
        let base = tier.path(&tokens(1, 600));
        assert_eq!(tier.span(&base), (400, 600, vec![]));
        assert_eq!(tier.span(&their_tail), (600, 900, vec![900]));
        // The chain the deepest segment sits in is still whole: root, base, the split
        // tail, and itself.
        let chain = tier.shared.store().chain_paths(&deepest);
        assert_eq!(chain.len(), 4);
        assert_eq!(chain[1], base);
        assert_eq!(chain[2], middle);
        assert_eq!(chain[3], deepest);
        // And a restart agrees, from the files alone: the root, the base, the two spans
        // hanging off it, and the child of the one that was cut.
        assert_eq!(tier.restarted().store().index.len(), 5);
    }

    /// A hydration composes the chain into exactly the image the conversation was
    /// written from — the rows re-strided back into one block, every snapshot in the
    /// chain, and the whole history all of it covers.
    #[test]
    fn a_chain_composes_into_the_image_it_was_written_from() {
        let tier = Tier::new("compose", 1 << 40);
        let (disk, _wakes) = tier.engine_side(0);
        let ids = tokens(1, 1200);
        // Three segments, from three turns.
        tier.write(0, &tokens(1, 400), &[200, 400]);
        tier.write(0, &tokens(1, 800), &[800]);
        tier.write(0, &ids, &[1200]);
        assert_eq!(tier.files().len(), 3);

        let mut prompt = ids.clone();
        prompt.push(9001);
        let candidate = disk.candidate(&prompt, 0).expect("a chain serves it");
        assert_eq!(candidate.resume, 1200);
        assert_eq!(candidate.tokens, 1200);
        assert_eq!(candidate.segments, 3);
        let image = disk.load(&candidate, &prompt).expect("it reads back");

        assert_eq!(image.tokens, ids, "the cumulative history, not one span");
        assert_eq!(image.full_kv.pos, 1200);
        assert_eq!(
            bytes_of(&image.full_kv),
            bytes_of(&full_kv(1200)),
            "the composed rows are the conversation's own, per head"
        );
        assert_eq!(
            image
                .snapshots
                .iter()
                .map(|(pos, _)| *pos)
                .collect::<Vec<_>>(),
            vec![200, 400, 800, 1200],
            "every snapshot in the chain, ascending"
        );

        // A hydration stamps every file in the chain, not just the tail: a shared base
        // is read by every conversation behind it and must not age out while they stay
        // fresh.
        let old = SystemTime::now() - Duration::from_secs(3600);
        let chain = tier.shared.store().chain_paths(&candidate.path);
        assert_eq!(chain.len(), 3);
        for path in &chain {
            touch(path, old).unwrap();
            tier.shared.store().index.get_mut(path).unwrap().mtime = old;
        }
        disk.note_hydrated(2, &candidate, 12);
        for path in &chain {
            assert!(
                std::fs::metadata(path).unwrap().modified().unwrap() > old,
                "{} was not stamped",
                path.display()
            );
            assert!(tier.shared.store().index[path].mtime > old);
        }
        assert_eq!(tier.shared.store().links.get(&2), Some(&candidate.path));
    }

    /// Every segment is a candidate, not only a conversation's tail: a prompt that
    /// shares nothing but the opening resumes at the deepest snapshot the shared part
    /// of the chain offers, wherever in the chain that snapshot lives.
    #[test]
    fn an_interior_segment_serves_a_prompt_that_only_reaches_it() {
        let tier = Tier::new("interior_candidate", 1 << 40);
        let (disk, _wakes) = tier.engine_side(0);
        tier.write(0, &tokens(1, 400), &[200, 400]);
        tier.write(0, &tokens(1, 800), &[600, 800]);
        let tail = tier.write(0, &tokens(1, 1200), &[1200]);

        // A conversation that diverges inside the second segment: the deepest snapshot
        // at or before 700 is the one at 600, in that same segment.
        let forked_prompt = forked(1, 700, 900);
        let candidate = disk.candidate(&forked_prompt, 0).expect("a candidate");
        assert_eq!(candidate.resume, 600);
        assert_eq!(
            candidate.segments, 2,
            "root and the segment it forks inside"
        );
        let image = disk
            .load(&candidate, &forked_prompt)
            .expect("it reads back");
        assert_eq!(image.tokens, tokens(1, 800));
        assert_eq!(image.full_kv.pos, 800);

        // One that diverges inside the FIRST segment reaches only its snapshots.
        let early = forked(1, 300, 500);
        let candidate = disk.candidate(&early, 0).expect("a candidate");
        assert_eq!(candidate.resume, 200);
        assert_eq!(candidate.segments, 1);
        assert!(disk.load(&candidate, &early).is_some());

        // A prompt that continues the whole conversation reaches the tail.
        let mut whole = tokens(1, 1200);
        whole.push(9001);
        let candidate = disk.candidate(&whole, 0).expect("a candidate");
        assert_eq!(candidate.resume, 1200);
        assert_eq!(candidate.path, tail);

        // A warm slot that already reaches as deep wins the tie: reading a gigabyte to
        // resume where the cache already is buys nothing.
        assert!(disk.candidate(&whole, 1200).is_none());
        // Every prompt keeps a token to prefill — that token produces the logits the
        // decode starts from — so a snapshot at exactly the prompt's length is capped
        // away and the one below it serves instead.
        assert_eq!(disk.candidate(&tokens(1, 1200), 0).unwrap().resume, 800);
        // And a conversation sharing nothing is not a candidate at all.
        assert!(disk.candidate(&tokens(9, 1200), 0).is_none());
    }

    /// A conversation that is already covered is not written again, and the floor is
    /// measured against what the store could actually RESUME it at — a boundary with
    /// no snapshot behind it resumes nothing.
    #[test]
    fn a_write_is_skipped_when_it_would_buy_nothing() {
        let tier = Tier::new("skip", 1 << 40);
        let (disk, _wakes) = tier.engine_side(32);
        let pending = || tier.shared.store().pending.len();
        let clear = || {
            tier.shared.store().pending.clear();
        };
        let queue = |history: &[u32], snapshots: &[usize]| {
            let rings: Vec<(usize, Arc<HostSnapshot>)> =
                snapshots.iter().map(|pos| (*pos, snapshot(*pos))).collect();
            disk.queue_write(0, history, &full_kv(history.len()), rings.into_iter(), None);
        };

        // Too short to pay for its snapshots.
        queue(&tokens(1, 16), &[16]);
        assert_eq!(pending(), 0);

        // No snapshot, so no position anything could resume at.
        let ids = tokens(1, 400);
        disk.queue_write(0, &ids, &full_kv(400), std::iter::empty(), None);
        assert_eq!(pending(), 0);

        // Long enough, with a resumable position: queued.
        queue(&ids, &[400]);
        assert_eq!(pending(), 1);
        clear();

        // Already stored: the same history again is nothing new.
        tier.write(0, &ids, &[400]);
        queue(&ids, &[400]);
        assert_eq!(pending(), 0);

        // A turn's worth longer is still covered by what is stored: the chain resumes
        // this conversation a little shallower, which is tokens of prefill against a
        // gigabyte of writing.
        let turn = tokens(1, 400 + DISK_MIN_GROWTH - 1);
        queue(&turn, &[turn.len()]);
        assert_eq!(pending(), 0);

        // Grown past the floor, it is worth the bytes.
        let grown = tokens(1, 400 + DISK_MIN_GROWTH);
        queue(&grown, &[grown.len()]);
        assert_eq!(pending(), 1);
        clear();

        // A conversation that DIVERGED inside what is stored is only covered up to the
        // fork: past that the stored segment holds another conversation's keys.
        queue(&forked(1, 390, 400 + DISK_MIN_GROWTH), &[400]);
        assert_eq!(pending(), 1);
        clear();

        // Another conversation sharing the whole stored prefix IS covered by it, which
        // is the dedup: it resumes at 400 and only adds a few tokens.
        queue(&forked(1, 400, 420), &[420]);
        assert_eq!(pending(), 0);
    }

    /// A conversation whose chain reached its depth cap is not extended by another
    /// segment: the last one is rewritten as a longer span under the same parent, and
    /// the shorter one it replaces — childless and unlinked now — goes.
    #[test]
    fn a_chain_at_its_depth_cap_rewrites_its_last_segment_instead_of_growing() {
        let tier = Tier::new("depth", 1 << 40);
        let step = DISK_MIN_GROWTH;
        let mut last = PathBuf::new();
        for turn in 1..=MAX_CHAIN_DEPTH {
            let history = tokens(1, turn * step);
            last = tier.write(0, &history, &[history.len()]);
        }
        assert_eq!(tier.files().len(), MAX_CHAIN_DEPTH);
        assert_eq!(
            tier.shared.store().chain_paths(&last).len(),
            MAX_CHAIN_DEPTH
        );
        assert_eq!(
            tier.span(&last),
            (
                (MAX_CHAIN_DEPTH - 1) * step,
                MAX_CHAIN_DEPTH * step,
                vec![MAX_CHAIN_DEPTH * step]
            )
        );

        // One more turn: the chain stays at the cap and its last segment covers more.
        let history = tokens(1, (MAX_CHAIN_DEPTH + 1) * step);
        let rebased = tier.write(0, &history, &[history.len()]);
        assert!(!last.exists(), "the shorter tail it replaces is deleted");
        assert_eq!(tier.files().len(), MAX_CHAIN_DEPTH);
        assert_eq!(
            tier.span(&rebased),
            (
                (MAX_CHAIN_DEPTH - 1) * step,
                (MAX_CHAIN_DEPTH + 1) * step,
                vec![(MAX_CHAIN_DEPTH + 1) * step]
            )
        );
        assert_eq!(
            tier.shared.store().chain_paths(&rebased).len(),
            MAX_CHAIN_DEPTH,
            "the chain a hydration has to compose stays bounded"
        );
    }

    /// A split pushes the segment it cuts, AND everything already chained behind that
    /// segment, one level further from the root. When the deepest chain back there
    /// cannot afford that level, the split is skipped: the arriving conversation writes
    /// a tail that overlaps what is stored, which is worse dedup and nothing else.
    /// Splitting anyway would leave existing segments deeper than the next scan is
    /// willing to walk, and it would delete them.
    #[test]
    fn a_split_that_would_push_a_chain_past_the_depth_cap_is_skipped() {
        let tier = Tier::new("split_depth", 1 << 40);
        let step = DISK_MIN_GROWTH;
        for turn in 1..=MAX_CHAIN_DEPTH {
            let history = tokens(1, turn * step);
            tier.write(0, &history, &[history.len()]);
        }
        let root = tier.path(&tokens(1, step));
        let deepest = tier.path(&tokens(1, MAX_CHAIN_DEPTH * step));
        let root_before = std::fs::read(&root).unwrap();
        assert_eq!(tier.files().len(), MAX_CHAIN_DEPTH);

        // A conversation forking inside the ROOT: cutting there would deepen all sixty
        // four segments behind it.
        let theirs = forked(1, step / 2, 2 * step);
        let their_tail = tier.write(1, &theirs, &[theirs.len()]);

        assert!(
            std::fs::read(&root).unwrap() == root_before,
            "the root is not cut"
        );
        assert_eq!(
            tier.span(&their_tail),
            (0, 2 * step, vec![2 * step]),
            "the arriving conversation stores its own root instead"
        );
        assert_eq!(
            tier.shared.store().chain_paths(&deepest).len(),
            MAX_CHAIN_DEPTH,
            "and the chain that was already stored is exactly as deep as it was"
        );
        // Which is what keeps a restart from deleting any of it.
        assert_eq!(
            tier.restarted().store().index.len(),
            MAX_CHAIN_DEPTH + 1,
            "every segment survives the scan"
        );
    }

    /// The write path never destroys a file on the strength of a name alone.
    ///
    /// A name is a 64-bit hash of a history, and both destructive things the writer
    /// does — renaming a segment over an existing one, deleting a superseded one —
    /// would act on the wrong conversation if two histories ever collided. The load
    /// path has always compared tokens for exactly this reason; these are the same
    /// check on the side that does the destroying.
    #[test]
    fn the_write_path_checks_tokens_before_it_destroys_anything() {
        let tier = Tier::new("write_collision", 1 << 40);
        let mine = tokens(1, 800);

        // Another conversation's segment, indexed under the name THIS conversation's
        // history hashes to: the shape a collision takes, as the writer would meet it.
        let squatter = tier.path(&mine);
        let theirs = tokens(7, 800);
        let rings = snapshot(800);
        disk_cache::write_segment(
            &squatter,
            &tier.shared.checkpoint,
            TOKENIZATION_RULES_VERSION,
            0,
            None,
            &theirs,
            &full_kv(800),
            &[(800, &rings)],
            None,
        )
        .unwrap();
        let before = std::fs::read(&squatter).unwrap();
        tier.shared.store().index.insert(
            squatter.clone(),
            Entry {
                segment: Some(IndexedSegment {
                    name: chain_id(&mine).name,
                    start: 0,
                    parent: None,
                    span: theirs.clone(),
                    snapshots: vec![800],
                }),
                standing: Standing::Ready,
                bytes: std::fs::metadata(&squatter).unwrap().len(),
                mtime: SystemTime::now(),
            },
        );

        write_one(&tier.shared, &tier.request(0, &mine, &[800]));
        assert!(
            std::fs::read(&squatter).unwrap() == before,
            "the other conversation's segment is left exactly as it was"
        );

        // And the supersede rule refuses the same way: a stored segment whose name
        // matches a prefix of the history being written, but whose tokens do not, is
        // not this conversation's earlier turn to delete.
        let tier = Tier::new("supersede_collision", 1 << 40);
        let grown = tokens(1, 1200);
        let impostor = tier.path(&tokens(2, 400));
        let rings = snapshot(400);
        disk_cache::write_segment(
            &impostor,
            &tier.shared.checkpoint,
            TOKENIZATION_RULES_VERSION,
            0,
            None,
            &tokens(2, 400),
            &full_kv(400),
            &[(400, &rings)],
            None,
        )
        .unwrap();
        tier.shared.store().index.insert(
            impostor.clone(),
            Entry {
                segment: Some(IndexedSegment {
                    name: chain_id(&grown[..400]).name,
                    start: 0,
                    parent: None,
                    span: tokens(2, 400),
                    snapshots: vec![400],
                }),
                standing: Standing::Ready,
                bytes: std::fs::metadata(&impostor).unwrap().len(),
                mtime: SystemTime::now(),
            },
        );
        write_one(&tier.shared, &tier.request(0, &grown, &[1200]));
        assert!(
            impostor.exists(),
            "a segment that only shares a name is not superseded"
        );
    }

    /// The size budget evicts the least recently used CHILDLESS segment, cascading
    /// into a parent the moment its last child is gone — and a base with living
    /// children survives any pressure, which is what pins a shared system prompt for
    /// free.
    #[test]
    fn the_budget_evicts_childless_segments_and_cascades_into_their_parents() {
        let tier = Tier::new("budget", 1 << 40);
        // A shared base with two tails, and one unrelated conversation.
        let mine = tokens(1, 800);
        let my_tail = tier.write(0, &mine, &[400, 800]);
        let theirs = forked(1, 400, 700);
        let their_tail = tier.write(1, &theirs, &[700]);
        let base = tier.path(&mine[..400]);
        let other = tier.write(2, &tokens(2, 400), &[400]);
        assert_eq!(tier.files().len(), 4);

        // No slot holds any of them any more, so nothing is protected by a link.
        for slot in 0..3 {
            tier.shared.store().links.remove(&slot);
        }
        // Ages the store cannot have produced on its own inside one test run.
        for (path, seconds) in [
            (&base, 4000u64),
            (&my_tail, 3000),
            (&their_tail, 2000),
            (&other, 1000),
        ] {
            let at = SystemTime::now() - Duration::from_secs(seconds);
            touch(path, at).unwrap();
            tier.shared.store().index.get_mut(path).unwrap().mtime = at;
        }

        // Room for everything but one file: the base is the oldest, but it has children,
        // so the oldest CHILDLESS one goes instead.
        let room: u64 = [&base, &their_tail, &other]
            .iter()
            .map(|p| std::fs::metadata(p).unwrap().len())
            .sum();
        enforce_budget(&tier.shared, room, &[]);
        assert!(!my_tail.exists(), "the oldest childless segment goes first");
        assert!(base.exists(), "the base has a child left and survives");
        assert!(their_tail.exists() && other.exists());

        // Under real pressure the cascade reaches the base: once its last child is
        // gone it is childless itself, in the same pass.
        enforce_budget(&tier.shared, 0, &[]);
        assert!(!their_tail.exists());
        assert!(!base.exists(), "the cascade reclaims the whole chain");
        assert!(!other.exists());
        assert_eq!(tier.indexed(), 0);
    }

    /// A chain a warm slot would come back from is never the answer to being over
    /// budget: the link protects the tail, and everything the tail chains onto is
    /// protected by having a child.
    #[test]
    fn a_linked_chain_outlives_any_budget() {
        let tier = Tier::new("linked", 1 << 40);
        tier.write(0, &tokens(1, 400), &[400]);
        let tail = tier.write(0, &tokens(1, 800), &[800]);
        let base = tier.path(&tokens(1, 400));

        enforce_budget(&tier.shared, 0, &[]);
        assert!(tail.exists(), "the linked tail stays");
        assert!(base.exists(), "and so does everything it chains onto");

        // The slot is reused for another conversation, so the chain is nobody's.
        let (disk, _wakes) = tier.engine_side(0);
        disk.unlink(0);
        enforce_budget(&tier.shared, 0, &[]);
        assert!(!tail.exists() && !base.exists());
        assert_eq!(tier.indexed(), 0);
    }

    /// One pending write per slot, newest wins, and the coalescing is of the
    /// PAYLOADS: a superseded request must not sit in the channel holding a gigabyte
    /// of host images alive, or a slow disk turns into unbounded memory.
    #[test]
    fn only_the_newest_queued_write_per_slot_is_kept_or_written() {
        let tier = Tier::new("coalesce", 1 << 30);
        let (disk, rx) = tier.engine_side(0);
        let first = tokens(1, 300);
        let second = tokens(1, 600);
        let third = tokens(1, 900);

        // Three page-outs of one slot before the writer gets a turn.
        for history in [&first, &second, &third] {
            disk.queue_write(
                0,
                history,
                &full_kv(history.len()),
                [(history.len(), snapshot(history.len()))].into_iter(),
                None,
            );
        }
        assert_eq!(
            tier.shared.store().pending.len(),
            1,
            "one payload per slot, not one per page-out"
        );
        assert_eq!(
            tier.shared.store().pending[&0].tokens,
            third,
            "and it is the newest"
        );

        // The writer claims it once; the wakes left over by the replacements find
        // nothing, which is where the coalescing shows up on that side.
        let claimed = claim(&tier.shared).expect("a pending write");
        write_one(&tier.shared, &claimed);
        assert!(claim(&tier.shared).is_none());
        assert_eq!(
            tier.files(),
            vec![tier.path(&third)],
            "exactly one segment, the newest history"
        );
        assert_eq!(
            rx.try_iter().count(),
            3,
            "one wake per enqueue, all harmless"
        );

        // Two slots pending at once are both written, oldest enqueue first.
        let other = tokens(2, 300);
        disk.queue_write(
            1,
            &other,
            &full_kv(other.len()),
            [(other.len(), snapshot(other.len()))].into_iter(),
            None,
        );
        let fourth = tokens(1, 1200);
        disk.queue_write(
            0,
            &fourth,
            &full_kv(fourth.len()),
            [(fourth.len(), snapshot(fourth.len()))].into_iter(),
            None,
        );
        assert_eq!(claim(&tier.shared).expect("the older enqueue").slot, 1);
        assert_eq!(claim(&tier.shared).expect("then the newer").slot, 0);
    }

    /// What the startup scan does with each kind of file it can find: a chain is
    /// linked up and indexed, an unfinished write is swept, a damaged file is deleted,
    /// the previous arc's flat v1 container is deleted (there is no migration), a
    /// segment whose parent is missing is deleted as an orphan, and another
    /// checkpoint's files are counted without being read or touched.
    #[test]
    fn the_scan_links_up_chains_and_deletes_what_cannot_be_reached() {
        let tier = Tier::new("scan", 1 << 40);
        tier.write(0, &tokens(1, 400), &[400]);
        let tail = tier.write(0, &tokens(1, 800), &[800]);
        let base = tier.path(&tokens(1, 400));
        let good_bytes: u64 = [&base, &tail]
            .iter()
            .map(|p| std::fs::metadata(p).unwrap().len())
            .sum();

        // A crashed writer's sibling, and a file whose bytes do not hold together.
        let partial = tier.shared.dir.join("dead.lkv.4242.tmp");
        std::fs::write(&partial, b"half a segment").unwrap();
        let damaged = tier
            .shared
            .dir
            .join(disk_cache::segment_file_name(0xdead_beef_dead_beef));
        let mut bytes = std::fs::read(&tail).unwrap();
        bytes.truncate(bytes.len() - 8);
        std::fs::write(&damaged, &bytes).unwrap();

        // The previous arc's container: version 1, otherwise intact.
        let v1 = tier
            .shared
            .dir
            .join(disk_cache::segment_file_name(0x0102_0304_0506_0708));
        let mut bytes = std::fs::read(&tail).unwrap();
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        std::fs::write(&v1, &bytes).unwrap();

        // An orphan: a perfectly good segment whose parent is not on disk. Written by
        // hand, since the writer cannot produce one.
        let orphan_history = tokens(3, 800);
        let orphan = tier.path(&orphan_history);
        let rings = snapshot(800);
        disk_cache::write_segment(
            &orphan,
            &tier.shared.checkpoint,
            TOKENIZATION_RULES_VERSION,
            400,
            Some(chain_id(&orphan_history[..400]).as_parent()),
            &orphan_history[400..],
            &full_kv(400),
            &[(800, &rings)],
            None,
        )
        .unwrap();

        // A file in our directory that is not named by our rule: nobody's parent, never
        // read, but its bytes are real.
        let stranger = tier.shared.dir.join("not-a-segment-name.lkv");
        std::fs::write(&stranger, vec![3u8; 2048]).unwrap();

        // Another checkpoint's directory, which this server must not read — and one
        // more crashed write, in there.
        let foreign_dir = tier.shared.root.join("ffffffffffffffff");
        std::fs::create_dir_all(&foreign_dir).unwrap();
        let foreign = foreign_dir.join("someone-elses.lkv");
        std::fs::write(&foreign, vec![7u8; 4096]).unwrap();
        let foreign_partial = foreign_dir.join("someone-elses.lkv.9999.tmp");
        std::fs::write(&foreign_partial, b"half a segment").unwrap();

        let fresh = tier.restarted();

        assert!(
            base.exists() && tail.exists(),
            "a chain survives its own scan"
        );
        assert!(!partial.exists(), "an unfinished write is swept");
        assert!(!damaged.exists(), "a damaged segment is deleted");
        assert!(!v1.exists(), "and so is the previous arc's container");
        assert!(!orphan.exists(), "a segment with no parent is deleted");
        assert!(stranger.exists(), "a file we did not name is left alone");
        assert!(
            foreign.exists(),
            "another checkpoint's segments are left alone"
        );
        assert!(
            !foreign_partial.exists(),
            "but its unfinished writes are swept too"
        );

        let store = fresh.store();
        assert_eq!(store.index.len(), 4, "the chain, the stranger, the foreign");
        let indexed = store.segment_at(&base).expect("the base is indexed");
        assert_eq!(indexed.span, tokens(1, 400));
        assert_eq!(indexed.snapshots, vec![400]);
        let indexed = store.segment_at(&tail).expect("the tail is indexed");
        assert_eq!(indexed.start, 400);
        assert_eq!(indexed.parent, Some(chain_id(&tokens(1, 400)).as_parent()));
        assert!(
            store.segment_at(&stranger).is_none() && store.segment_at(&foreign).is_none(),
            "neither is readable, and neither can be anybody's parent"
        );
        // The budget spans the whole store, everything counted included.
        assert_eq!(
            store.index.values().map(|entry| entry.bytes).sum::<u64>(),
            good_bytes + 2048 + 4096
        );
    }

    /// A parent whose header cannot be READ does not take its children with it.
    ///
    /// From the scan's side a segment behind an unreadable file is indistinguishable
    /// from an orphan — its parent is not among the segments that could be read — and
    /// deleting it for that would let one I/O fault destroy a whole subtree of intact
    /// files. So it is kept and counted, simply not built on, and the next scan links
    /// it up if the fault was transient. Which it usually is: a permission change, a
    /// disk that hiccuped, a file locked by something else for a moment.
    #[test]
    fn a_child_of_an_unreadable_parent_is_kept_rather_than_deleted() {
        use std::os::unix::fs::PermissionsExt;

        let tier = Tier::new("io_parent_scan", 1 << 40);
        tier.write(0, &tokens(1, 400), &[400]);
        let ids = tokens(1, 800);
        let tail = tier.write(0, &ids, &[800]);
        let base = tier.path(&tokens(1, 400));
        let tail_bytes = std::fs::metadata(&tail).unwrap().len();

        // Unreadable, not damaged: the header read fails without saying anything about
        // what the file holds.
        let readable = std::fs::metadata(&base).unwrap().permissions();
        std::fs::set_permissions(&base, std::fs::Permissions::from_mode(0o000)).unwrap();
        let fresh = tier.restarted();

        assert!(base.exists(), "an I/O fault is never a reason to delete");
        assert!(
            tail.exists(),
            "and neither is a fault on something in front"
        );
        {
            let store = fresh.store();
            // The file that could not be read is left out of the index entirely — the
            // scan learned nothing about it, not even its shape — while the segment
            // behind it is kept, counted, and simply not built on.
            assert_eq!(store.index.len(), 1);
            assert_eq!(
                store.index.values().map(|entry| entry.bytes).sum::<u64>(),
                tail_bytes
            );
            assert!(
                store.segment_at(&tail).is_some(),
                "the tail's place in the tree is known"
            );
            assert!(
                store.children_of(None).is_empty(),
                "but nothing is built on a chain whose front could not be read"
            );
        }

        // Nor is it the size budget's to take. It is childless and unlinked, so every
        // other rule would offer it up first — and deleting it would finish what the
        // I/O fault started, on a file nothing has been learned against.
        enforce_budget(&fresh, 0, &[]);
        assert!(tail.exists(), "held is held against the budget too");

        // Readable again, the next scan links the chain up as if nothing had happened.
        std::fs::set_permissions(&base, readable).unwrap();
        let fresh = tier.restarted();
        assert_eq!(fresh.store().chain_paths(&tail), vec![base, tail]);
    }

    /// A segment that is no longer built on is still its parent's child, and the budget
    /// has to keep seeing it that way.
    ///
    /// A read fault demotes a chain — nothing is composed through it for the rest of the
    /// run — but the demotion is about what can be USED, not about what exists. If it
    /// dropped the segment's place in the tree, the budget would find the parent
    /// childless and evict it, and the demoted tail, whose own bytes may be perfect,
    /// would be orphaned for good.
    #[test]
    fn a_demoted_segment_still_protects_its_parent_from_eviction() {
        let tier = Tier::new("demote_edge", 1 << 40);
        let (disk, _wakes) = tier.engine_side(0);
        tier.write(0, &tokens(1, 400), &[400]);
        let ids = tokens(1, 800);
        let tail = tier.write(0, &ids, &[800]);
        let base = tier.path(&tokens(1, 400));

        // The tail is what a warm slot would come back from, so the budget may not take
        // it; the base is protected only by having a child.
        let mut prompt = ids.clone();
        prompt.push(9001);
        let candidate = disk.candidate(&prompt, 0).expect("a candidate");
        disk.forget_stale(&candidate);
        tier.shared.store().links.insert(0, tail.clone());

        enforce_budget(&tier.shared, 0, &[]);
        assert!(tail.exists(), "the linked tail stays");
        assert!(
            base.exists(),
            "and so does the segment it chains onto, demoted child or not"
        );
    }

    /// A crash between a split's two writes leaves a cache the scan can use: the base
    /// is there, the segment it was cut from is untouched, and the two simply overlap.
    /// Imperfect dedup, never a wrong resume — which is the whole reason the base goes
    /// first.
    #[test]
    fn a_split_interrupted_after_its_base_leaves_a_consistent_store() {
        let tier = Tier::new("crash", 1 << 40);
        let mine = tokens(1, 800);
        let stored = tier.write(0, &mine, &[400, 800]);

        // The base of a split at 400, written on its own — the state a crash between
        // the two writes leaves behind.
        let rows = full_kv(800).range(0, 400).unwrap();
        let rings = snapshot(400);
        let base = tier.path(&mine[..400]);
        disk_cache::write_segment(
            &base,
            &tier.shared.checkpoint,
            TOKENIZATION_RULES_VERSION,
            0,
            None,
            &mine[..400],
            &rows,
            &[(400, &rings)],
            None,
        )
        .unwrap();

        let fresh = tier.restarted();
        assert!(base.exists() && stored.exists(), "both are valid roots");
        assert_eq!(fresh.store().index.len(), 2);

        // And the store still serves the conversation, from either.
        let (tx, _rx) = channel();
        let disk = DiskCache {
            tx,
            shared: Arc::clone(&fresh),
            min_tokens: 0,
            stamp: Cell::new(0),
            trusted: Cell::new(true),
        };
        let mut prompt = mine.clone();
        prompt.push(9001);
        let candidate = disk.candidate(&prompt, 0).expect("a candidate");
        assert_eq!(candidate.resume, 800);
        assert!(disk.load(&candidate, &prompt).is_some());
    }

    /// A segment whose bytes turn out not to hold together is deleted and forgotten,
    /// and the caller is told to fall back rather than being handed a half-parsed
    /// chain.
    #[test]
    fn a_damaged_segment_is_deleted_at_load() {
        let tier = Tier::new("load_damage", 1 << 30);
        let (disk, _wakes) = tier.engine_side(0);
        let ids = tokens(1, 400);
        let path = tier.write(0, &ids, &[200, 400]);
        let mut prompt = ids.clone();
        prompt.push(9001);
        let candidate = disk.candidate(&prompt, 0).expect("a candidate");

        // Truncated after it was indexed, which is exactly what the header-only scan
        // cannot catch.
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 16);
        std::fs::write(&path, &bytes).unwrap();

        assert!(disk.load(&candidate, &prompt).is_none());
        assert!(!path.exists(), "an unusable segment is deleted");
        assert_eq!(tier.indexed(), 0);
        assert!(disk.candidate(&prompt, 0).is_none());
    }

    /// A damaged PARENT is the file that gets deleted, and the chain behind it stops
    /// being offered — its own bytes may be perfect, but they mean nothing without the
    /// spans in front of them.
    #[test]
    fn a_damaged_parent_costs_the_parent_and_the_chains_candidacy() {
        let tier = Tier::new("load_parent", 1 << 40);
        let (disk, _wakes) = tier.engine_side(0);
        tier.write(0, &tokens(1, 400), &[400]);
        let ids = tokens(1, 800);
        let tail = tier.write(0, &ids, &[800]);
        let base = tier.path(&tokens(1, 400));

        let mut prompt = ids.clone();
        prompt.push(9001);
        let candidate = disk.candidate(&prompt, 0).expect("a candidate");
        assert_eq!(candidate.resume, 800);

        let mut bytes = std::fs::read(&base).unwrap();
        let at = bytes.len() - 32;
        bytes[at] ^= 0x01;
        std::fs::write(&base, &bytes).unwrap();

        assert!(disk.load(&candidate, &prompt).is_none());
        assert!(!base.exists(), "the file that failed is the one deleted");
        assert!(tail.exists(), "the tail's own bytes are not in question");
        assert!(
            disk.candidate(&prompt, 0).is_none(),
            "but nothing can be composed from it, so it stops being offered"
        );
    }

    /// Deleting a segment takes the WHOLE subtree behind it out of circulation, not
    /// just the chain that happened to be read.
    ///
    /// Nothing can be composed through a segment that is gone, and a descendant that
    /// could still be offered would fail exactly the same way once per request until
    /// the next scan. What makes that structural rather than bookkeeping is that
    /// matching walks DOWN from the roots: a segment whose parent is not in the index
    /// is not reachable, so it cannot be chosen. The descendants' own files are
    /// untouched, and the next scan links up whatever is still whole.
    #[test]
    fn deleting_a_segment_stops_everything_behind_it_from_being_offered() {
        let tier = Tier::new("subtree_demote", 1 << 40);
        let (disk, _wakes) = tier.engine_side(0);
        tier.write(0, &tokens(1, 400), &[400]);
        let middle = tier.write(0, &tokens(1, 800), &[800]);
        let ids = tokens(1, 1200);
        let deepest = tier.write(0, &ids, &[1200]);
        let base = tier.path(&tokens(1, 400));

        let mut prompt = ids.clone();
        prompt.push(9001);
        let candidate = disk.candidate(&prompt, 0).expect("a candidate");

        let mut bytes = std::fs::read(&base).unwrap();
        let at = bytes.len() - 32;
        bytes[at] ^= 0x01;
        std::fs::write(&base, &bytes).unwrap();
        assert!(disk.load(&candidate, &prompt).is_none());
        assert!(!base.exists());

        // Neither descendant is offered any more — including the middle one, which no
        // candidate in this run ever pointed at.
        assert!(middle.exists() && deepest.exists(), "their files stay");
        assert!(disk.candidate(&prompt, 0).is_none());
        let mut shorter = tokens(1, 800);
        shorter.push(9001);
        assert!(
            disk.candidate(&shorter, 0).is_none(),
            "a prompt reaching only the middle segment is not offered it either"
        );
    }

    /// A segment that names itself as its own parent is refused for THAT reason,
    /// before it is read a second time.
    ///
    /// The chain is followed by reading files, and every body already read is still in
    /// memory while the next is parsed, so a cycle bounded only by the depth cap would
    /// pull the same gigabyte-scale segment in sixty-four times over — an allocation
    /// failure, which aborts the process where a refused chain costs a re-prefill.
    /// Refusing a name the walk has already seen is what bounds it, which is why the
    /// reason it reports is the assertion: the depth cap would refuse this file too,
    /// only after reading it sixty-three more times.
    ///
    /// Nothing is deleted for it: a file that cannot be composed has said nothing about
    /// whether its own bytes are sound.
    #[test]
    fn a_segment_that_chains_onto_itself_is_refused_without_being_read_twice() {
        let (tier, events) = Tier::watched("cycle", 1 << 40);
        let (disk, _wakes) = tier.engine_side(0);
        let history = tokens(1, 800);
        let path = tier.path(&history);
        let name = disk_cache::segment_name_of(&path).expect("a canonical name");
        let rings = snapshot(800);
        // A tail whose parent reference is its own name. The writer cannot produce one;
        // a corrupted reference or another build's rule can.
        disk_cache::write_segment(
            &path,
            &tier.shared.checkpoint,
            TOKENIZATION_RULES_VERSION,
            400,
            Some(ParentRef { name, chain: 7 }),
            &history[400..],
            &full_kv(400),
            &[(800, &rings)],
            None,
        )
        .unwrap();
        tier.shared.store().index.insert(
            path.clone(),
            Entry {
                segment: Some(IndexedSegment {
                    name,
                    start: 400,
                    parent: Some(ParentRef { name, chain: 7 }),
                    span: history[400..].to_vec(),
                    snapshots: vec![800],
                }),
                standing: Standing::Ready,
                bytes: std::fs::metadata(&path).unwrap().len(),
                mtime: SystemTime::now(),
            },
        );

        let mut prompt = history.clone();
        prompt.push(9001);
        let candidate = DiskCandidate {
            path: path.clone(),
            tokens: 800,
            resume: 800,
            segments: 1,
            bytes: std::fs::metadata(&path).unwrap().len(),
        };
        assert!(disk.load(&candidate, &prompt).is_none());
        assert!(path.exists(), "a chain that cannot be walked is not damage");
        assert!(
            disk.candidate(&prompt, 0).is_none(),
            "and it stops being offered rather than being retried every request"
        );
        let why: Vec<String> = events
            .drain()
            .into_iter()
            .filter_map(|event| match event {
                ServeLog::DiskCacheFailed { error, .. } => Some(error),
                _ => None,
            })
            .collect();
        assert!(
            why.iter().any(|line| line.contains("chains onto itself")),
            "the cycle is what refused it, not the depth cap sixty-three reads later: {why:?}"
        );
    }

    /// A rejection deletes the FILE it was decided on, not whatever holds that name by
    /// the time the unlink happens.
    ///
    /// The writer publishes segments by renaming them into place, so between a reader
    /// finding a file unusable and deleting it, a fresh and perfectly good segment can
    /// have taken the name. Deleting by name alone throws that one away.
    #[test]
    fn a_rejection_does_not_delete_a_file_that_was_replaced_since_it_was_read() {
        let tier = Tier::new("reject_race", 1 << 40);
        let (disk, _wakes) = tier.engine_side(0);
        let ids = tokens(1, 400);
        let path = tier.write(0, &ids, &[400]);

        // The file as some earlier read saw it, and then the writer replacing it.
        let stale = FileIdentity::at(&path).expect("a file to identify");
        let rings = snapshot(400);
        disk_cache::write_segment(
            &path,
            &tier.shared.checkpoint,
            TOKENIZATION_RULES_VERSION,
            0,
            None,
            &ids,
            &full_kv(400),
            &[(400, &rings)],
            None,
        )
        .unwrap();
        assert_ne!(
            FileIdentity::at(&path),
            Some(stale.clone()),
            "the rename put a different file at this name"
        );

        disk.reject(&path, Some(&stale), "corrupt");
        assert!(path.exists(), "the file that replaced it is not condemned");

        // Decided on the file that is actually there, the same rejection deletes it.
        let current = FileIdentity::at(&path);
        disk.reject(&path, current.as_ref(), "corrupt");
        assert!(!path.exists());
        assert_eq!(tier.indexed(), 0);
    }

    /// Every deletion for a verdict about a file's contents goes through one function,
    /// and it deletes first and forgets second: an entry dropped for a file still on
    /// disk would make the budget claim room the disk does not have. A file that
    /// refuses to be deleted therefore stays counted — and stays out of circulation,
    /// because whatever was wrong with it still is.
    ///
    /// The callers are a rejected hydration, a superseded earlier turn, and a segment
    /// that could not be read to be split; they share this path so the invariant cannot
    /// hold in one of them and not the others.
    #[test]
    fn a_deletion_that_fails_keeps_the_file_counted_whoever_asked() {
        let tier = Tier::new("condemn_stuck", 1 << 40);
        let ids = tokens(1, 400);
        let path = tier.write(0, &ids, &[400]);

        let mut perms = std::fs::metadata(&tier.shared.dir).unwrap().permissions();
        let writable = perms.clone();
        perms.set_readonly(true);
        std::fs::set_permissions(&tier.shared.dir, perms).unwrap();
        let gone = condemn(&tier.shared, &path, DiskEvictReason::Superseded);
        std::fs::set_permissions(&tier.shared.dir, writable).unwrap();

        assert!(!gone, "and it says so, rather than reporting a deletion");
        assert!(path.exists());
        assert_eq!(tier.indexed(), 1, "the bytes are still on the disk");
        assert!(
            tier.shared.store().children_of(None).is_empty(),
            "and nothing is built on it"
        );

        // Writable again, the same call clears it and the entry goes with it.
        assert!(condemn(&tier.shared, &path, DiskEvictReason::Superseded));
        assert!(!path.exists());
        assert_eq!(tier.indexed(), 0);
    }

    /// A chain the ENGINE turned down after reading it stops being offered, without
    /// anything being deleted for it.
    ///
    /// Nothing is wrong with those files — the job that asked for them went away, or
    /// the image did not suit the slot it was meant for — but re-reading and
    /// re-composing a multi-gigabyte chain on every later request that shares the
    /// prefix, only to turn it down again, is exactly the cost this tier exists to
    /// avoid.
    #[test]
    fn a_chain_the_engine_turned_down_stops_being_offered_and_is_not_deleted() {
        let tier = Tier::new("set_unusable", 1 << 40);
        let (disk, _wakes) = tier.engine_side(0);
        let ids = tokens(1, 400);
        let path = tier.write(0, &ids, &[200, 400]);
        let mut prompt = ids.clone();
        prompt.push(9001);

        let candidate = disk.candidate(&prompt, 0).expect("a candidate");
        disk.set_unusable(&candidate);
        assert!(path.exists(), "nothing was wrong with the file");
        assert_eq!(tier.indexed(), 1, "and it is still counted");
        assert!(
            disk.candidate(&prompt, 0).is_none(),
            "but this run will not read it again"
        );

        // A restart puts it back in circulation: the demotion was about this run.
        let fresh = tier.restarted();
        assert!(fresh.store().segment_at(&path).is_some());
        assert_eq!(fresh.store().children_of(None).len(), 1);
    }

    /// A rejection that cannot delete its file keeps the file COUNTED — the order is
    /// delete first, forget second, everywhere in this module. An entry dropped for a
    /// file still on disk would make the budget claim room the disk does not have.
    #[test]
    fn a_rejection_that_cannot_delete_keeps_the_file_counted() {
        let tier = Tier::new("reject_stuck", 1 << 40);
        let (disk, _wakes) = tier.engine_side(0);
        let ids = tokens(1, 400);
        let path = tier.write(0, &ids, &[400]);
        let seen = FileIdentity::at(&path);

        let mut perms = std::fs::metadata(&tier.shared.dir).unwrap().permissions();
        let writable = perms.clone();
        perms.set_readonly(true);
        std::fs::set_permissions(&tier.shared.dir, perms).unwrap();
        disk.reject(&path, seen.as_ref(), "corrupt");
        std::fs::set_permissions(&tier.shared.dir, writable).unwrap();

        assert!(path.exists(), "nothing could be deleted");
        assert_eq!(
            tier.indexed(),
            1,
            "so the bytes are still counted against the budget"
        );
        let mut prompt = ids.clone();
        prompt.push(9001);
        assert!(
            disk.candidate(&prompt, 0).is_none(),
            "but nothing is built on it either"
        );
    }

    /// A chain whose links do not hold up is refused, and it is the segment that made
    /// the false claim that goes — never the shared base it points at, which other
    /// conversations may be resuming from correctly.
    #[test]
    fn a_tail_that_lies_about_its_parent_is_the_one_deleted() {
        let tier = Tier::new("bad_link", 1 << 40);
        let (disk, _wakes) = tier.engine_side(0);
        let base_history = tokens(1, 400);
        let base = tier.write(0, &base_history, &[400]);

        // A tail that chains onto the base by name but carries the wrong second hash —
        // the shape a file from another naming rule takes.
        let history = tokens(1, 800);
        let path = tier.path(&history);
        let mut parent = chain_id(&base_history).as_parent();
        parent.chain ^= 0xff;
        let rings = snapshot(800);
        let write = || {
            disk_cache::write_segment(
                &path,
                &tier.shared.checkpoint,
                TOKENIZATION_RULES_VERSION,
                400,
                Some(parent),
                &history[400..],
                &full_kv(400),
                &[(800, &rings)],
                None,
            )
            .unwrap();
        };
        write();
        scan(&tier.shared);
        assert!(
            !path.exists(),
            "the scan already refuses it: it links onto nothing it can verify"
        );
        assert!(base.exists());

        // The same lie, reached at load: the index is made to hold it so the load path
        // is the one that has to catch it.
        write();
        tier.shared.store().index.insert(
            path.clone(),
            Entry {
                segment: Some(IndexedSegment {
                    name: chain_id(&history).name,
                    start: 400,
                    parent: Some(parent),
                    span: history[400..].to_vec(),
                    snapshots: vec![800],
                }),
                standing: Standing::Ready,
                bytes: std::fs::metadata(&path).unwrap().len(),
                mtime: SystemTime::now(),
            },
        );
        let mut prompt = history.clone();
        prompt.push(9001);
        let candidate = disk.candidate(&prompt, 0).expect("a candidate");
        assert_eq!(candidate.resume, 800);
        assert!(disk.load(&candidate, &prompt).is_none());
        assert!(!path.exists(), "the segment that lied is deleted");
        assert!(base.exists(), "the base it pointed at is untouched");
    }

    /// A file whose stored history is not the conversation that found it is refused
    /// and deleted, whatever its header says.
    ///
    /// Segment names are a hash of the token ids, so a collision — or a file another
    /// build named by a different rule — can put one conversation's prompt in front of
    /// another's keys. The header the scan read cannot see it: it is checked against
    /// the CHECKPOINT, not against the prompt. This is the check that makes the
    /// argument structural rather than a statement about how unlikely FNV collisions
    /// are.
    #[test]
    fn a_segment_holding_another_conversation_is_refused_at_load() {
        let tier = Tier::new("collision", 1 << 30);
        let mine = tokens(1, 400);
        let theirs = tokens(7, 400);
        let (disk, _wakes) = tier.engine_side(0);

        // Another conversation's rows, written under the name THIS conversation hashes
        // to — the shape a collision takes.
        let path = tier.path(&mine);
        let tip = snapshot(400);
        let early = snapshot(200);
        disk_cache::write_segment(
            &path,
            &tier.shared.checkpoint,
            TOKENIZATION_RULES_VERSION,
            0,
            None,
            &theirs,
            &full_kv(400),
            &[(200, &early), (400, &tip)],
            None,
        )
        .unwrap();

        // Matching is over the histories the scan read, so the file is not even a
        // candidate for a prompt it shares nothing with. The collision case is the one
        // where the NAME was reached by this conversation, which is what the load check
        // has to catch — so the candidate is built by hand, as a colliding lookup would
        // produce it.
        let mut prompt = mine.clone();
        prompt.push(9001);
        let candidate = DiskCandidate {
            path: path.clone(),
            tokens: 400,
            resume: 400,
            segments: 1,
            bytes: std::fs::metadata(&path).unwrap().len(),
        };
        assert!(disk.load(&candidate, &prompt).is_none());
        assert!(!path.exists(), "a file holding someone else is deleted");
    }

    /// A file that cannot be read at all is not evidence about the segment inside it,
    /// so it is never deleted for an I/O fault — and the index keeps it, because the
    /// next attempt may well succeed.
    #[test]
    fn an_io_fault_never_deletes_a_segment() {
        let tier = Tier::new("io", 1 << 30);
        let (disk, _wakes) = tier.engine_side(0);
        let ids = tokens(1, 400);
        let path = tier.write(0, &ids, &[200, 400]);
        let mut prompt = ids.clone();
        prompt.push(9001);
        let candidate = disk.candidate(&prompt, 0).expect("a candidate");

        // Moved out from under the reader, which is the cheapest way to produce an
        // `Io` verdict rather than one about the bytes.
        let hidden = tier.shared.dir.join("moved.aside");
        std::fs::rename(&path, &hidden).unwrap();
        assert!(disk.load(&candidate, &prompt).is_none());
        assert_eq!(
            tier.indexed(),
            1,
            "an I/O fault leaves the entry alone: nothing was learned about the file"
        );

        // Back in place, it reads as it always did.
        std::fs::rename(&hidden, &path).unwrap();
        assert!(disk.load(&candidate, &prompt).is_some());
    }

    /// The same rule at scan time: a header that could not be read leaves the file on
    /// disk, while one that read and did not bind is deleted.
    ///
    /// The predicate is an allowlist so that it fails closed — a rejection class added
    /// later protects the file until somebody decides otherwise, which is the safe
    /// default for the one operation here that cannot be undone.
    #[test]
    fn only_a_verdict_about_the_bytes_deletes_at_scan() {
        assert!(!deletable("io"), "an I/O fault is not a verdict");
        assert!(deletable("binding"));
        assert!(deletable("corrupt"));
        assert!(deletable("collided"));
        assert!(
            !deletable("something-a-later-build-invented"),
            "an unknown class must protect the file, not delete it"
        );
    }

    /// A resume position the chain cannot serve any more is a MISS, not damage.
    ///
    /// The position was chosen from the header the scan read, and a segment can be
    /// replaced between the scan and the read by a perfectly good one over the same
    /// span with different snapshot boundaries. Deleting it there would throw away the
    /// newer file and fail nothing but the next request; instead the index catches up
    /// from what was actually read, and the job takes the cold path.
    #[test]
    fn a_resume_the_chain_can_no_longer_serve_is_a_miss() {
        let tier = Tier::new("stale_resume", 1 << 30);
        let (disk, _wakes) = tier.engine_side(0);
        let ids = tokens(1, 400);
        let path = tier.write(0, &ids, &[200, 400]);
        let mut prompt = ids.clone();
        prompt.push(9001);
        let candidate = disk.candidate(&prompt, 0).expect("a candidate");
        assert_eq!(candidate.resume, 400);

        // The same span, rewritten with shallower boundaries — same history, so the
        // same content-addressed name.
        let shallow = snapshot(200);
        disk_cache::write_segment(
            &path,
            &tier.shared.checkpoint,
            TOKENIZATION_RULES_VERSION,
            0,
            None,
            &ids,
            &full_kv(400),
            &[(200, &shallow)],
            None,
        )
        .unwrap();

        assert!(
            disk.load(&candidate, &prompt).is_none(),
            "the chain cannot resume at 400 any more"
        );
        assert!(path.exists(), "and it is not damaged, so it stays");
        assert_eq!(tier.indexed(), 1);

        // The index learned what is really there, so the next choice is the honest one
        // and it hydrates.
        let next = disk.candidate(&prompt, 0).expect("a shallower candidate");
        assert_eq!(next.resume, 200);
        assert!(disk.load(&next, &prompt).is_some());
    }

    /// A slot repurposed while its write was in flight gets no link back when that
    /// write lands. The segment is still worth keeping — it is a real conversation —
    /// but the link is what stops the budget from ever reclaiming it, and it must not
    /// be held on behalf of a slot that moved on.
    #[test]
    fn a_write_that_lands_after_its_slot_moved_on_does_not_relink_it() {
        let tier = Tier::new("epoch", 1 << 40);
        let (disk, _wakes) = tier.engine_side(0);
        let ids = tokens(1, 400);
        disk.queue_write(
            0,
            &ids,
            &full_kv(400),
            [(400, snapshot(400))].into_iter(),
            None,
        );
        let claimed = claim(&tier.shared).expect("a pending write");

        // The engine hands slot 0 to another conversation while the writer works.
        disk.unlink(0);
        write_one(&tier.shared, &claimed);

        let path = tier.path(&ids);
        assert!(path.exists(), "the segment is still worth keeping");
        assert!(
            tier.shared.store().links.is_empty(),
            "but nothing is linked to it"
        );
        // Which is what makes it reclaimable under pressure.
        enforce_budget(&tier.shared, 0, &[]);
        assert!(!path.exists());
    }

    /// A store bound to a checkpoint the loaded weights did not come from is not a
    /// store at all: nothing on disk describes those weights, so the tier stops
    /// hydrating AND stops writing for the rest of the process.
    #[test]
    fn a_checkpoint_that_changed_under_us_disables_the_tier() {
        let tier = Tier::new("verify", 1 << 30);
        let (disk, _wakes) = tier.engine_side(0);
        let ids = tokens(1, 400);
        tier.write(0, &ids, &[200, 400]);
        let mut prompt = ids.clone();
        prompt.push(9001);
        assert!(
            disk.candidate(&prompt, 0).is_some(),
            "the store serves this conversation while the binding holds"
        );

        // The same path, different bytes: a re-quantized GGUF between the scan and
        // the load.
        disk.verify(CheckpointId::from_parts(0xdead_beef, 4096));
        assert!(disk.candidate(&prompt, 0).is_none(), "no hydration");
        disk.queue_write(
            1,
            &tokens(2, 400),
            &full_kv(400),
            [(400, snapshot(400))].into_iter(),
            None,
        );
        assert_eq!(tier.shared.store().pending.len(), 0, "and no writes");

        // The binding it was scanned against is still accepted, and the refusal is
        // permanent whatever is verified afterwards.
        disk.verify(tier.shared.checkpoint);
        assert!(disk.candidate(&prompt, 0).is_none());
    }

    /// A file that refuses to be deleted stays in the index, so the accounting keeps
    /// counting the bytes that are still on disk — and the pass moves on instead of
    /// trying the same stubborn file forever.
    #[test]
    fn a_failed_deletion_keeps_the_file_counted() {
        let tier = Tier::new("stuck", 1 << 40);
        let stuck = tier.write(0, &tokens(1, 400), &[400]);
        let other = tier.write(0, &tokens(2, 400), &[400]);
        // Only the last write is linked, so both would otherwise be evictable.
        tier.shared.store().links.remove(&0);

        // A read-only directory refuses the unlink without touching the files.
        let mut perms = std::fs::metadata(&tier.shared.dir).unwrap().permissions();
        let writable = perms.clone();
        perms.set_readonly(true);
        std::fs::set_permissions(&tier.shared.dir, perms).unwrap();
        enforce_budget(&tier.shared, 0, &[]);
        std::fs::set_permissions(&tier.shared.dir, writable).unwrap();

        assert!(stuck.exists() && other.exists(), "nothing could be deleted");
        assert_eq!(
            tier.indexed(),
            2,
            "and both are still counted, since the bytes are still there"
        );

        // Writable again, the same pass clears them.
        enforce_budget(&tier.shared, 0, &[]);
        assert!(!stuck.exists() && !other.exists());
        assert_eq!(tier.indexed(), 0);
    }

    /// Another checkpoint's file counts toward the budget and can be evicted by it:
    /// switching models must not strand an unbounded pile.
    #[test]
    fn a_foreign_image_is_evictable_by_the_budget() {
        let tier = Tier::new("foreign_budget", 0);
        let foreign_dir = tier.shared.root.join("ffffffffffffffff");
        std::fs::create_dir_all(&foreign_dir).unwrap();
        let foreign = foreign_dir.join("someone-elses.lkv");
        std::fs::write(&foreign, vec![7u8; 4096]).unwrap();
        scan(&tier.shared);
        enforce_budget(&tier.shared, 0, &[]);
        assert!(!foreign.exists());
        assert_eq!(tier.indexed(), 0);
    }

    /// A snapshot the container would refuse costs that snapshot, never the write:
    /// the write is what the conversation's warmth depends on.
    #[test]
    fn an_unusable_snapshot_does_not_cost_the_write() {
        let tier = Tier::new("filter", 1 << 30);
        let (disk, _wakes) = tier.engine_side(0);
        let ids = tokens(1, 400);
        // Out of order, one past the rows behind it, one mislabelled, one repeated, one
        // at position zero — where nothing can resume.
        disk.queue_write(
            0,
            &ids,
            &full_kv(400),
            [
                (400, snapshot(400)),
                (600, snapshot(600)),
                (200, snapshot(200)),
                (200, snapshot(200)),
                (0, snapshot(0)),
                (300, snapshot(350)),
            ]
            .into_iter(),
            None,
        );
        let request = claim(&tier.shared).expect("queued");
        assert_eq!(
            request
                .snapshots
                .iter()
                .map(|(pos, _)| *pos)
                .collect::<Vec<_>>(),
            vec![200, 400]
        );
        // And what survives is exactly what the container accepts.
        write_one(&tier.shared, &request);
        assert_eq!(tier.files().len(), 1);
    }

    /// The writer thread runs the queue in order, and a flush is the barrier the
    /// shutdown path waits on.
    #[test]
    fn the_writer_thread_drains_what_it_is_sent() {
        let tier = Tier::new("thread", 1 << 30);
        let (disk, rx) = tier.engine_side(0);
        let shared = Arc::clone(&tier.shared);
        let thread = std::thread::spawn(move || writer_loop(rx, shared));
        let ids = tokens(1, 400);
        disk.queue_write(
            0,
            &ids,
            &full_kv(400),
            [(200, snapshot(200)), (400, snapshot(400))].into_iter(),
            None,
        );
        disk.flush(Duration::from_secs(10));
        assert_eq!(
            tier.files(),
            vec![tier.path(&ids)],
            "the queued segment landed before the flush returned"
        );
        drop(disk);
        thread.join().unwrap();
    }

    /// The drafter's rows live in the segment whose span holds the position they
    /// reach, and a re-partition moves them to whichever side of the cut still covers
    /// that position. A chain composes the deepest set it finds, since only one can be
    /// installed and only at its exact position.
    #[test]
    fn drafter_planes_ride_along_in_the_segment_that_covers_them() {
        let tier = Tier::new("drafter", 1 << 40);
        let (disk, _wakes) = tier.engine_side(0);
        let mine = tokens(1, 800);
        let mut request = tier.request(0, &mine, &[400, 800]);
        request.drafter = Some(Arc::new(drafter(600)));
        write_one(&tier.shared, &request);
        let stored = tier.path(&mine);
        assert_eq!(
            tier.read(&stored).drafter.map(|planes| planes.pos),
            Some(600)
        );

        // A fork at 400 cuts the segment: the planes reach past the cut, so they stay
        // with the tail.
        let theirs = forked(1, 400, 700);
        tier.write(1, &theirs, &[700]);
        let base = tier.path(&mine[..400]);
        assert!(tier.read(&base).drafter.is_none());
        assert_eq!(
            tier.read(&stored).drafter.map(|planes| planes.pos),
            Some(600),
            "the side of the cut that still covers position 600 keeps them"
        );

        // And a hydration of that chain hands them back.
        let mut prompt = mine.clone();
        prompt.push(9001);
        let candidate = disk.candidate(&prompt, 0).expect("a candidate");
        let image = disk.load(&candidate, &prompt).expect("it reads back");
        assert_eq!(image.drafter.map(|planes| planes.pos), Some(600));

        // A fork PAST the planes leaves them in the base, where the position they reach
        // still is.
        let later = forked(1, 700, 900);
        tier.write(2, &later, &[900]);
        let base = tier.path(&mine[..700]);
        assert_eq!(tier.read(&base).drafter.map(|planes| planes.pos), Some(600));
        assert!(tier.read(&stored).drafter.is_none());
    }

    /// A conversation's history can outrun the rows behind it — a cancelled job
    /// truncates its tokens to what the cache held — and only positions the rows can
    /// back are storable.
    #[test]
    fn a_history_longer_than_its_rows_is_stored_up_to_the_rows() {
        let tier = Tier::new("short_rows", 1 << 30);
        let (disk, _wakes) = tier.engine_side(0);
        let ids = tokens(1, 500);
        disk.queue_write(
            0,
            &ids,
            &full_kv(400),
            [(400, snapshot(400))].into_iter(),
            None,
        );
        let request = claim(&tier.shared).expect("queued");
        assert_eq!(request.tokens, ids[..400]);
        write_one(&tier.shared, &request);
        assert_eq!(tier.files(), vec![tier.path(&ids[..400])]);
        assert_eq!(tier.span(&tier.path(&ids[..400])), (0, 400, vec![400]));
    }
}
