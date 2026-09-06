# 2026-08-11 — First client feedback lands: the scored-field escape stops lying about first fields, `shared_prefix` collapses 14 POSTs into one, the body cap goes to 100 MB, and max_ctx becomes a ceiling (lazy KV, 128k CLI default)

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


**Where it came from.** The first external consumer of `/xwen/v1/batch`'s scored path
reported two things and asked for a third. The bug report: on `include_score: "all"`
items with several fields, the FIRST field's `escape` reads 0.999-1.000 (18/18 of their
multi-field items) while all 142 non-first fields sit at 1e-7 to 4e-4, pinning every
category's mean escape at exactly 1/fieldCount and killing the signal — their
hypothesis was escape sampled one token early, at the skeleton's opening punctuation.
The feature ask: a `shared_prefix` field, because each item repeats the story text and
a 377 KB story against the ~2 MB body cap forced one batch into 14 POSTs, each
re-prefilling the same prefix. And the operational ask: raise that cap, raise the
context default, and make context memory demand-driven in serve.

**The hypothesis was wrong, and the bug was ours anyway.** The escape and the opener
scores are read off the same row, so a stale row would have bent the scores too. A
temporary row dump at the exact `score_field` read settled it: at the first field's
choice point (after `{"urgent":`) the model puts **54.9% on ` true` and 44.9% on
` false` — single tokens carrying a leading space** — and ~5e-5 on the bare `true`/
`false` the opener set actually contains. The model wants `{"urgent": true`; the
teacher-forced skeleton is compact; the opener-level escape read the model's preferred
spelling of the ANSWER as "none of the above". By the second field the compact document
has pinned the style (bare `false` at 99.8%) — which is why only first fields read ≈1,
single-field items included (verified: 0.9998 on a one-field item). This was the
already-ledgered formatting confound, now measured at its mechanism.

**The fix reclassifies the whole row instead of looking up openers.** `escape_mass` now
takes the full next-token distribution (`Generator::last_probs`, the full-row companion
to `last_logprobs_for`, normalized by the same code) and classifies every encodable id
by its raw BYTES (`LagunaTokenizer::decoded_vocab` reverses the byte-level alphabet,
one ~248k-entry walk cached per tokenizer): bytes that — past any leading JSON
whitespace at an unquoted field, verbatim at a quoted one — are a nonempty prefix of
some option count INSIDE whatever spelling BPE gave it; pure-JSON-whitespace tokens at
an unquoted field are FORMATTING and excluded from both sides; the escape is outside
over inside-plus-outside. (Bytes and JSON-only whitespace are both second-model review
catches, same day: per-id `decode` is lossy on a token holding part of a multi-byte
character, so a text-level match misread the canonical opener of any non-ASCII option
as escape; and `trim_start` would have discarded an NBSP as layout when it would make
the document invalid.) Measured on the repro:
first-field escape **0.9999 → 0.00197** — and the 0.00197 is real signal, mostly ` "`,
the model considering a quoted string where a boolean belongs — second field 0.00052 →
0.000051, scores bit-identical throughout (they never read the classification). The
client's mean-escape workaround can be reverted. (For the next field-0 anomaly report:
the one genuinely field-0-specific shape left in the assembly loop is the THINKING
path — a reasoning-enabled scored item prepends the decode loop's unforwarded
reasoning tail to field 0's prefill (batch.rs, the `held` reconciliation), where every
later field prepends its chosen option. Correct today and `ensure!`-guarded; look
there before suspecting the escape measure.)

**A finding the fix does not touch, ledgered.** The same row dump shows the two
spellings disagree on the near-tie: the space-channel says true at 0.55, the bare
channel (which the teacher-forced skeleton scores through) says true at 0.44. Scores
are conditional on the compact format by construction — documented, defensible, but on
near-ties the channel choice can flip the chosen value. New TODO entry with the
numbers.

**`shared_prefix` is a wire-size field, not a prefill feature.** The runner has always
prefilled the items' shared TOKEN prefix once (the LCP snapshot); what repeated was the
request body. `BatchRequest.shared_prefix` is prepended verbatim to every item's first
message before rendering, so the prompts — and therefore answers and scores — are
byte-identical to spelling the document per item (pinned by test and verified live:
identical scored output CLI-vs-CLI and over HTTP). An item with no messages fails as an
item; the serve estimate counts the prefix bytes once.

**The body cap was axum's implicit 2 MB; it is now an explicit 100 MB.**
`DefaultBodyLimit::max(MAX_BODY_BYTES)` on the API router. Verified: a 5 MB body
parses (400 on an unknown field — the parser saw all of it), 101 MB answers 413. Real
cost stays judged in tokens by the queue estimates and max_ctx, not in body bytes.

**max_ctx is now a ceiling, not an allocation.** Full-attention `LayerCache`s start at
8192 positions (`KV_INITIAL_CTX`) and double on demand up to max_ctx
(`ensure_full_capacity`: the whole old buffer copied bit-exact — not just the committed
rows, so even rows above a rewound `len` survive a growth and forward-restores keep
whatever validity the rewind discipline gave them — with one host log line per growth
step). `LayerSnapshot::Full` carries no data, so snapshots taken before a growth
restore after it — pinned by bit-exactness tests against a flat allocation, the
forward-restore case included — and the page-in path grows via `import_full_kv` while
the host-image pre-flight no longer bounds pos by allocated slots (max_ctx, checked at
model level, is the real bound). The CLI default rises 8192 → 131072 at no up-front memory cost (the 131072 ceiling is
8 GiB of KV on the 27B and 2.5 GiB on the 35B, paid only if a prompt gets there; load
allocates 0.5 / 0.16 GiB); serve keeps its 262144-clamped-to-trained default, now
meaning "up to". A growth pass ends in one device sync, since candle's pool frees the
replaced buffers only there — without it the top doubling step transiently holds old
and new allocations together (~1.5x). Idle unload is
the reset: dropping the model drops whatever the cache grew to, and the reload starts
at 8192 again. Verified live: an 11215-token prompt on the new default logged one
growth step (8192 → 16384, 0.3 GB) and answered correctly.

**The TUI stops lying about batch jobs (same day, operator feedback).** Watching the
client's workload live exposed two dashboard failures. The NOW pane said "deadline in
96m15s" — the watchdog kill-ceiling, computed from the queue's deliberately-loose
bytes/3 estimate (731,183 "tokens" of a ~2.2 MB body ÷ the 150 tok/s floor rate + the
summed budgets ÷ 10 + slack), rendered as if it were an ETA; it now reads
"watchdog in …" and the prompt figure carries "~…(est)" when it is the estimate
(`JobPicked.estimated`, true for a batch, whose text is untokenized at pickup). And
HISTORY rows for batches read `321,930+0→2,252  —  0.0s  —  —`: unlabeled columns,
`prefill_tokens`/`*_secs`/`stop` never filled by the batch path — so the flow read as
"322k from cache, nothing prefilled", the duration as instant, and outcome/rate as
nothing. The record now carries a `BatchSummary`, `run_batch_job` fills the token arithmetic in
the record's own `prompt = cache_read + prefill` terms (the shared prefix is prefilled
once and READ N−1 times, so the summed per-item `cached_prefix_tokens` overcount by
exactly one span), and the table grew a header row. The first cut blended everything
into one `output ÷ whole-run` rate; that was called out as misleading (correctly — it
described neither machine) and replaced the same day by MEASURED per-phase columns:
the batch runner now reports what it actually forwarded and what it actually decoded
(`BatchStats.{prefill,decode}_{tokens,ms}`, on the wire for clients too — prefill by
delta off a new cumulative `Generator::prefill_spend` counter, since the shared prefix
and the scored trials belong to no single item; decode straight off each item's
`DecodeOutcome`), and HISTORY shows `prefill` and `decode` as separate columns, each
with its own time and its own t/s, with TTFT/spec%/whole-run total in a `detail`
column. A fully scored batch decodes nothing and its decode cell is honestly a dash —
its work shows up as prefill (teacher-forced trials), which is what it is. The LOG pane's follow latch also stopped being one-way:
scrolling used to disengage follow permanently (only End re-armed it), so a pane once
touched sat pinned to a stale line forever; scrolling back down ONTO the newest line
now re-engages follow — arriving at the live edge means wanting it live.

**Also swept while in there.** Three comments still carried laguna's ~70GB checkpoint
into a repo whose files are 19-20 GB (`MEMORY_WARN_BYTES` doc, the config template's
`idle_unload` note, gguf.rs's residency measurement — the last now attributes its ~10%
figure to laguna's working set explicitly). The parity-gate script's refusal message
still says it; that sits under the standing "scripts/ maxuna-isms" ledger item.

**Result.** 791 + 69 lib/integration tests pass, 0 fail; new tests cover the escape
classification (6, byte-level and NBSP cases included), the byte table's alphabet
reversal (1), shared_prefix (2 + the estimate), and KV growth (3, forward-restore
across growth included). Three-perspective review, each catching something the others
did not: the in-family reviewer found the scheduling estimate assuming the prefill
dedup always fires (now per-item, erring loose for the watchdog); the serve explorer
found three wrong figures in fresh comments, the growth-pass pool peak (candle frees
replaced buffers only at a sync — `grow_kv_capacity` now ends in one) and the load-
bearing import-before-restore order at the page-in site (now commented); the
second-model pass (codex `gpt-5.6-sol`) found the byte-level lossiness and NBSP
classes above plus the buffered-body concurrency exposure (accepted for loopback,
ledgered). The parity gate was attempted per the standing rule (the growth path
touches the forward, if not the math) and refused to start: a foreign `xwen serve`
process holds the GPU. Pending on a free GPU; everything else in this entry is done.
