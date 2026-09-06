# 2026-08-30 (P4, later the same day) — Flash-Next serves: the QSA indexer rows ride in `HostFullKv`, the PLE state rides on its own layer's snapshot entry, and the container goes to v4

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


P4 was the item that kept this checkpoint off two surfaces, and it was always one
question: what does a cache image have to carry. The answer turned out to be two
answers, because qwen4exp carries two pieces of state that no image carried before and
they are not the same kind of thing.

**The QSA indexers' raw keys are position-indexed, so a snapshot stores nothing for
them.** Every token writes its own row into `IndexerCache` — one MQA key head,
`[max_ctx, indexer_head_dim]` f32 per full-attention layer — exactly the way a
full-attention layer writes its K/V. Nothing about a row depends on what came before it,
so a rewind is `IndexerCache::truncate(pos)` and it is exact, not approximate, and a
`CacheSnapshot` needs no indexer data at all. Only the page-out path has to move bytes,
and there the rows belong beside the K/V planes they mirror: `HostFullKv` gained a `qsa`
plane set and a `qsa_head_dim`, carried through `range`, `concat` and a new `qsa_prefix`
accessor. One head is what makes that cheap — a position range of a QSA plane is a slice
of the buffer, where the K/V planes need the per-head gather they have always needed.
`export_full_kv_from`, `import_full_kv_into` and `check_full_kv_importable` all take the
indexer caches now, and the import sets `IndexerCache::len` to the number of rows it
imported, which is what keeps the `cache.len == pos` invariant that QSA selection reads
before it scores anything.

**The PLE state has no inverse, so it travels as data.** The dilated conv window
(`[hc_count*hidden, (k-1)*ngram_size]` f32) and the rolling n-gram token history (at most
`ngram_size - 1` raw ids) are summaries of everything that came before; no position
determines them and nothing reconstructs them short of re-running the prefix. So they get
an image: `PleImage { history, conv, state_len }` and `PleShape { conv_len, state_len,
history_len }` in `src/qwen4exp/ple.rs`, with `PleState::image()`, `shape()`,
`accepts(PleShape)` and `restore(&PleImage)`. `restore` clears the per-token trail and
disarms the checkpoint, the same thing `LayerCache::restore` does to a DeltaNet layer —
an image is a new history, and a rollback trail from the old one is worse than no trail.

**The layer alignment is the decision worth keeping.** The snapshot's `layers` vector
stays ONE ENTRY PER TRUNK LAYER. The PLE image rides on its layer's own entry through a
wrapper variant — `LayerSnapshot::Ple { inner: Box<LayerSnapshot>, ple: PleImage }`, host
mirror `HostLayerSnapshot::Ple`, disk tag `LAYER_PLE = 3` — rather than becoming a fourth
kind beside `Full`/`Swa`/`Linear`. The reason is that the PLE layer is ALSO a DeltaNet
layer: a flat `Ple` variant standing where a `Linear` one stands would have quietly
dropped that layer's conv and delta state, and the failure mode of dropping recurrent
state is not an error, it is a conversation that restores, runs, and answers differently.
The wrapper keeps the inner snapshot exactly what it was. Nesting is one deep and a `Ple`
inside a `Ple` is refused on both the assembly path and the read path, because a wrapper
that can wrap itself is a framing that can describe a state the model does not have.
`LayerCache::restore` and `check_restorable` unwrap to `inner` and never learn what a PLE
is — the state does not live in a `LayerCache` at all, it lives in `Qwen4ExpParts`, and
`XwenModel` is the one place that pairs the two. That half of D15 stands; only its
refusal was P2 scope.

**The container went 3 → 4, and only one of the two halves forced it.** `disk_cache.rs`
carries an invariant at the constant: the version discriminates FRAMING, never content,
because the checkpoint binding and the per-layer kind tags already discriminate content
between them. The two halves landed on opposite sides of it. The PLE state is a new
per-layer tag inside unchanged framing, so an old reader refuses that layer by its tag
and no bump is needed — the same way the DeltaNet recurrent state landed at v2. The QSA
planes sit INSIDE the existing full-attention record, after its K/V planes, where nothing
tags them: a v3 reader would parse the K/V planes, stop, and then fail on framed bytes it
never consumed, which is a corruption error over a file that is not corrupt. The bump
turns that into what it actually is — a v4 file on an older build is a clean `Binding`
rejection, the scan deletes it, and the conversation costs a re-prefill.

**What that let us delete.** `XwenModel::refuse_state_transfer` is gone, and all five of
its call sites (`take_cache_snapshot`, `restore_cache_snapshot`, `export_full_kv`,
`check_importable`, `import_full_kv`) do the real work. `Model::unservable_reason`,
`unservable_message` and `unbatchable_message` are gone with it, along with serve's
startup refusal and its fallback notice in `src/bin/xwen/main.rs` and batch's refusal in
`src/batch.rs` and `main.rs`. `Model::default_servable()` now returns `Model::default()`;
its fallback branch is dead and deliberately kept, as is `Model::servable()` itself —
that predicate is the question the cache-moving surfaces actually ask, and the next
architecture that arrives half-ported needs somewhere to say no. Its doc now says what it
answers today: true for every registry checkpoint. `xwen serve` and `xwen batch` both run
Flash-Next with no flags, `/v1/models` lists it and a request may select it by name.

Two gates did NOT move, and the ledger keeps them. `auto_fetch()` is still false, which
is now the ONLY thing gating this checkpoint on the wire: it is listed and selectable
exactly when its shards are really in the HF cache, and a request for an uncached one is
a 400 pointing at `xwen fetch` rather than a 111 GB download started by a stranger.
`supports_drafting()` is still false — D6's verify seam, no MTP or other drafter wired —
so `DraftMode` resolution logs "no drafter available" for this checkpoint and leaves
every other checkpoint's drafting alone. That path already existed and needed no change.
`Model::snapshot_bytes()` already counted the PLE conv window and
`Model::kv_bytes_per_token()` already counted the indexer's 512 B/token/layer; both were
checked rather than assumed, both are unchanged, and both are now load-bearing for
page-out sizing instead of forward-looking.

Two things went differently from what TODO.md predicted. The n-gram history was ledgered
as sequence-level state to be stored beside `CacheSnapshot::pos`, needing its own plane
type and validator because it is u32 in an all-f32 plane world. It is neither: it rides
on its layer's entry with the conv window, as raw ids inside `PleImage`, and never
becomes a framed plane. And the same bullet expected new disk `LAYER_*` tags in the
plural — there is exactly one, because the QSA rows turned out to belong to the
full-attention record rather than to a layer kind of their own.

**What the tests pin.** Six new unit tests in `src/kv_cache.rs`: a PLE image through the
whole snapshot chain, the nested-PLE refusal, PLE shapes and lengths that lie, QSA planes
through `range`/`concat`/framing, QSA shape rejections, and a qwen35 image carrying no
QSA planes whose record is byte-for-byte the old one plus two zero counts. Three in
`src/qwen4exp/stack.rs` against the tiny Mixed and Exact fixtures: snapshot and rewind
reproduce the continuation; export/import pages a conversation back in across a
displacing conversation; and an image from another geometry is refused by whichever half
disagrees — QSA head dim on the rows path, PLE conv width on the snapshot path. One in
`src/serve/disk_cache.rs` round-trips a qwen4exp segment with its indexer planes and PLE
state, plus a v3-container rejection case.

Worth being exact about what that does NOT cover: there is no serve-engine harness that
runs a real model. `page_out_live` and `page_in` are private free functions over a
private `EngineState`, and the engine's own tests use stand-in payloads. So the
equivalence is pinned one level down, at `XwenModel::export_full_kv` +
`take_cache_snapshot().to_host()` → `check_importable` → `import_full_kv` →
`restore_cache_snapshot`, which is exactly the sequence those two functions perform. The
real file through a real server — load, converse, page out, evict, page back in, continue
— has not been run and is ledgered.

Still open for this checkpoint (TODO.md, "Deferred from the qwen4exp cache-image arc"):
`IndexerCache` still allocates at `max_ctx` up front with no growth path where the
trunk's KV grows lazily, 4 MB per QSA layer at 8k, and page-in now gives that a second
edge — a conversation longer than the live allocation is refused rather than grown, since
the import can only set `len` within the plane it was handed. No drafter. The disk tier's
stored-image path is unit-tested only. And no qwen4exp serve run has ever been
benchmarked: every perf number for this checkpoint comes from `generate`, and the
one-shot figures must not be quoted as serve figures.

Two review passes over the finished diff, one of them a different model family, and
between them they moved the same check twice — which is worth recording, because the
second correction was to the first one.

The check is `PleState::accepts`, and the state it guards is the one with no inverse.
It originally bounded the restored history with `<=` the n-gram order. But the reachable
lengths are not a range: a state that has stepped no token holds NOTHING, and every state
that has stepped at least one holds EXACTLY `ngram_size - 1` ids, because `next_history`
left-pads with eos and then takes that many. An image declaring one id on a 3-gram layer
would have installed and made the hash pad an extra eos in front of the real predecessor.
So the bound became the set `{0, ngram_size - 1}`. The second pass then pointed out that
this is still wrong in the other direction: `0` is only legitimate AT POSITION ZERO, and
an empty history on a segment restoring to position 900 makes the hash pad two eos
instead of one — the same bug wearing a "fresh state" costume, and one the set-membership
form waves straight through. The rule is position-dependent and cannot be stated without
`pos`, so `pos` is now threaded from `restore_cache_snapshot` and `check_importable` down
through `check_ple_restorable` into `accepts`, and the check is a single equality against
what that position implies. The test pins both directions and was confirmed to fail
against the set-membership form.

The trigger for all of these is a crafted or rotted v4 segment rather than an in-process
snapshot — a live state at a nonzero position always holds a full history — so this is
untrusted-input hardening of a path whose own comments say it is written to be loud.

Two smaller ones from the same passes. `export_full_kv_from` now checks every indexer's
head dim instead of taking the first layer's and trusting it: at `pos == 0` every plane
is zero bytes, so `HostFullKv::new`'s per-plane length check says nothing, and a
heterogeneous stack would have described itself with one layer's geometry. And
`restore_cache_snapshot` now runs a real immutable preflight over the trunk layers. Its
comment claimed "asked of both halves before either is written", but only the PLE half
had a preflight; `LayerCache::restore` re-checks each layer inline, which refuses the bad
layer without unwinding the ones already overwritten — a kind mismatch at layer five
leaves four holding the previous conversation at lengths that still agree. Narrowly
reachable (the disk path preflights through `HostSnapshot::check_restorable`, and an
in-process snapshot comes from the same model), but a comment that claims an invariant
should be backed by code that enforces it.

1009 lib tests and 5 bin tests pass.
