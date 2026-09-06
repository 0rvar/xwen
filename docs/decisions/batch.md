# Batch

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic; moved
here verbatim on 2026-09-06. Dated paragraphs, newest additions appended within their topic.


**`xwen batch` snapshots at the EXACT longest common token, not at a chunk-64
boundary.** The chunk-boundary plan was inherited from Marconi, whose constraint is
real for it: a chunked prefill kernel cannot materialize the recurrent state mid-chunk,
so the only cheap snapshot positions are chunk multiples, and the prefix is truncated
down to one. xwen has no such constraint — the generation layer materializes full state
at whatever position a prefill stops, which is what serve's `PrefixCache` has always
relied on to snapshot at arbitrary stops. Rounding down would therefore have meant
building alignment logic to give up reuse that costs nothing to keep. Notable because
the argument that first recommended the boundary — keep it simple — is the argument
that refuted it (2026-08-09).

**Grouping is single-level, the cache is RAM-only, and it dies with the request.** One
batch computes one longest common prefix across all items and that is the whole cache
hierarchy: no prefix tree, no disk tier, no TTL, nothing that outlives the process. The
run is one command with all its items in hand, so a request-scoped snapshot needs no
eviction policy, no cross-request invalidation and no checkpoint binding — the three
places the serve-side prefix cache spends its complexity. Multi-level trees and a
pinned cross-batch system-prompt snapshot are ledgered, not built (2026-08-09).

**Three floors keep the snapshot honest, and one of them is a correctness precondition
rather than a heuristic.** `MIN_SHARED_PREFIX` = 64: below that the snapshot/restore
bookkeeping costs more than re-prefilling the tokens it saves, so every item runs cold
instead. A single-item batch never snapshots — there is nothing to share it with. And
the shared span is capped at `min_len − 1`, so the longest item's prefix can never be
the whole of the shortest item: an item with no tail of its own would decode from the
snapshot position itself and write into positions the snapshot covers, which
`XwenModel::restore_cache_snapshot` explicitly forbids (model.rs:716 — full-attention
layers are restored by truncation, valid only while the slots below the position still
hold the same keys). The cap is not a margin; it is the restore precondition expressed
as arithmetic (2026-08-09).

**The drafter rides along by truncation, not by a `DrafterImage` export.** Every item
shares every token below the snapshot position, so restoring the target's cache and
calling `sync_drafter_to(pos)` leaves the drafter holding exactly the rows it would
have built by prefilling that prefix itself. Exporting and re-importing a drafter image
per item would reconstruct the same rows at the price of a copy per item. Truncation is
correct here only because of the single-level grouping — a prefix tree with divergent
branches would need the image (2026-08-09).

**Batch sampling defaults to greedy and thinking defaults to OFF, and both are
deliberate divergences from the chat surface.** A batch request is a
classification/extraction surface: two runs of the same payload should agree token for
token, and a caller comparing scores across items needs the sampler out of the way, so
`BATCH_SAMPLING` is temperature 0 with the chat defaults parked on the inert
`top_k`/`top_p`/`seed` fields for a request that raises the temperature alone. Thinking
is off for a blunter reason, observed directly during bring-up: items carry tight token
budgets, and an unrequested reasoning block eats a 64-token budget before the answer
starts, so the item finishes on `length` with no JSON at all. Either default is one
field away for a request that wants it (2026-08-09).

**A scored field's value is ASSEMBLED by the engine, not decoded under a grammar mask
with the choice points detected after the fact.** The alternative was to keep the
llguidance path and read confidence off the mask at the positions where the grammar
offered a choice, which needs the grammar to say where those positions are. llguidance
1.7.6 exposes no structural-position API: captures do not exist for grammars compiled
from JSON schema, and capture semantics are on-completion anyway — the wrong shape for
a per-step reading. Assembly inverts the problem instead of working around it. The
runner teacher-forces the JSON skeleton it wrote itself, so it knows exactly where each
value begins, and selection is exact rather than inferred: every allowed option is
scored in full and the answer is chosen from the scores. It needs no llguidance change,
and the response is the same document the grammar path would have produced. What it
costs is one forward per option token instead of one per answer token. v1 scope is a
flat all-required object of enum/boolean fields; `include_score` PRESENCE routes the
item, so `false` is an annotation like any other and differs from absent — a schema
cannot silently switch engines by toggling a boolean (2026-08-09).

**An option's score includes the one delimiter token that follows it, which is what
makes the option set prefix-free.** Without it a strict-prefix option can never lose:
score is a sum of log-probabilities, every added token contributes something negative,
so `score("low_priority") = score("low") + (negative)` and `"low"` wins every time
regardless of what the model believes. Scoring the terminator — the closing quote for a
string enum, the delimiter after a bare literal — makes the two option sequences
diverge at a token they cannot share, and the comparison becomes a real one. Found by
the outside-model review pass (codex, `gpt-5.6-sol`) and confirmed live: on the
adversarial pair `["low", "low_priority"]` the long option now takes 0.9988 when the
prompt asks for it (2026-08-09).

**Seam canonicality is CHECKED at plan time, not assumed.** The assembled document is
scored as `encode(segment) ++ encode(option) ++ encode(segment)`, which is only the
document's own tokenization when the tokenizer agrees — BPE is free to merge across a
seam and produce a different sequence for the same bytes. `check_seams` re-encodes every
(segment, option) and (option, segment) pair and refuses the schema when the merged
encoding differs from the concatenation. In practice this rejects values that fuse with
their delimiter (trailing punctuation or whitespace) and values needing JSON escapes,
where the escape sequence rather than the label would be what gets scored. Refusal was
chosen over scoring-the-escape-anyway: an option whose score is not the score of the
value the caller named is worse than an error message. Lifting the limits is ledgered
(2026-08-09).

**`escape` was opener-level mass, confounded by formatting for bare literals —
SUPERSEDED 2026-08-11 by the whole-row classification below.** As shipped 2026-08-09 it
reported the probability on tokens that open no option at the first choice-point token.
For a quoted enum the forced opening quote filtered formatting out; for a boolean, whose
choice point sits after `:`, whitespace tokens a pretty-printer would emit competed with
`true`/`false` and the escape read near 1 beside a near-certain answer score. Kept then
with the refinement ledgered; the first external client hit it (every multi-field
item's first field at 0.999-1.000, pinning mean escape at 1/fieldCount) and the
refinement shipped.

**`escape` is a whole-row classification by token TEXT, with formatting factored out —
because the mass the opener set missed was mostly the ANSWER in the model's preferred
spelling.** The client's one-token-early hypothesis was checked and refuted first: a
row dump at the exact read shows the first boolean slot holding 54.9% ` true` / 44.9%
` false` — single space-led tokens, the model wanting `{"k": true` against the compact
skeleton — with bare `true`/`false` at ~5e-5, and the style pinning to compact (bare
`false` 99.8%) from the second field on, which is why only first fields read ≈1.
`escape_mass` now classifies every encodable id by its raw BYTES: stripped of leading
JSON whitespace at an unquoted field (verbatim at a quoted one, where leading
whitespace is string content), a nonempty prefix of some option's bytes is INSIDE;
nothing but JSON whitespace at an unquoted field is FORMATTING, excluded from both
sides; escape = outside / (inside + outside). Bytes, not decoded text, because
byte-level BPE cuts multi-byte characters across tokens and `decode` of such an id is
lossy (U+FFFD) — a text-level match would misread the canonical opener of any
non-ASCII option as escape (found by the second-model review, codex `gpt-5.6-sol`;
`LagunaTokenizer::decoded_vocab` reverses the byte-level alphabet instead, pinned
against `encode` by test). JSON whitespace means the four bytes JSON allows — an NBSP
would invalidate the document and honestly counts outside (same review). Prefix
matching is one-way — `yesterday` is not headed for `yes`, so a token that begins with
an option and carries on counts outside; the canonical tokenizations carrying real
mass never fuse across the value's edge (check_seams refuses the plans where they
would) — and canonical-spelling-only for quoted fields (`\/` and `\uXXXX` alternates
count outside, the check_seams stance; negligible mass). Measured: first-field escape
0.9999 → 0.00197, and the residue is genuine vocabulary-gap signal (mostly ` "`, a
string where a boolean belongs); scores are untouched by construction (they never read
the classification) and verified bit-identical. Costs one ~248k-entry vocab walk
cached per tokenizer plus one full-row softmax readback per field
(`Generator::last_probs`, normalized by the same code path as the scores) (2026-08-11).
The residual first-field elevation (absolute 0.010-0.035 without thinking, 0.042-0.109
with, against later fields at 4.3e-7 to 1.7e-4 — 35B measurements; the client's 27B
data tails higher, max 0.109 without thinking) was re-reported by the client post-fix and
confirmed by a position-vs-identity controlled row dump: it follows document position,
not field identity, and its mass is concentrated on plausible alternative openers —
mostly ` "` and the space-led capitalized booleans. It is conditioning signal, kept
as-is; consumers comparing escape across categories should aggregate with and without
first fields (2026-08-12, log.md same date).

**`shared_prefix` is a wire-size field, deliberately NOT a prefill feature.** The
runner has prefilled the items' shared TOKEN prefix once since batch shipped; what
repeated was the request body — a 377 KB story per item forced a real client into 14
POSTs under the old 2 MB cap. The field is prepended verbatim to every item's first
message before rendering, so the resulting prompts (and answers, and scores) are
byte-identical to spelling the document per item — pinned by test and verified live
over both transports. Alternatives refused: request-level shared MESSAGES would change
prompt structure (separate turns) and so change answers against the inline spelling;
placeholder interpolation buys nothing over prepending. An item with no messages
fails as an item, an empty string means absent (2026-08-11).

**Scores are not bit-stable between the cached and cold arms, and consumers must
compare with tolerance.** Replaying an item from the snapshot prefills its tail as a
short span, which routes the MoE through `mv_id` where the cold arm's single long
prefill takes `mm_id` — the same math at different precision. Measured across the demo
batch: chosen values identical on every item but one, whose two candidates sat at 0.502
and 0.493 and swapped, and scores differing in the third to fourth decimal throughout.
This is the partition-dependence class already recorded above ("Persistent state is
partition-dependent in its low bits"), reached through a new door, not a new phenomenon;
`XWEN_MM_ID_MIN_SEQ=1` forces both arms onto one kernel and makes them byte-identical,
which is how it was diagnosed. `XWEN_BATCH_NO_CACHE=1` is the A/B lever (2026-08-09).
