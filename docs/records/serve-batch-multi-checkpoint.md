# 2026-08-11 — `/xwen/v1/batch` ships and the server stops being pinned to one checkpoint: every request names its model, the engine swaps lazily

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


**Where it came from.** The batch runner had been CLI-only since it shipped
(2026-08-09), with the HTTP endpoint parked behind P10 in the ledger. The ask that
unparked it also reshaped it: the endpoint should not be pinned to the served model —
`--model` should only decide which checkpoint is the DEFAULT for the compat dialects,
and any checkpoint should lazy-load and idle-unload like the one the server started
around. That turns "add a route" into "make the engine checkpoint-aware", and the route
falls out of it.

**What shipped.** Two things, one mechanism:

- **The engine is checkpoint-aware.** `EngineState` now records which official
  checkpoint it holds (`size`), every queued job names the checkpoint it needs (a new
  `Job` enum over generations and batches), and the pickup compares the two: on a
  mismatch the live conversation is imaged out through the same path an idle unload
  takes, the state drops, and the lazy load brings in the named checkpoint — one model
  resident at a time, which is this machine's memory invariant. Which checkpoint the
  default GGUF is comes from its `general.architecture` via the new `Arch::model()`,
  never from `--model-size` (the flag and a `-m` path can disagree; the file cannot).
  A non-default checkpoint hub-resolves lazily (download on miss, logged), speculates
  with its own official sidecar at its own fitted `p_min` — `draft.p_min` left unset
  now stays `None` through the config merge and resolves per loaded checkpoint at
  attach time — and runs without the disk tier, which stays bound to the default
  checkpoint (verifying it against foreign weights would permanently disable it, and
  feeding it foreign images would poison the store; decisions.md "Serving").
- **`POST /xwen/v1/batch`** takes exactly the JSON document `xwen batch` reads on
  stdin and returns exactly the document it prints. Its `model` field is honored per
  request (absent = the server's default; unknown = 400, the CLI's strictness). The
  job rides the ordinary `JobQueue` as the second `Job` variant, scored and
  deadline-bounded by a bytes/3 overestimate so small chat requests keep jumping ahead
  of it, and answers with a single terminal `EngineEvent::BatchDone` carrying the whole
  response. The compat dialects gained the lighter half of the same feature: a `model`
  that parses as a known checkpoint ("27b"/"35b") selects it, anything else falls back
  to the default and is echoed as before, and `/v1/models` now lists the default id
  plus both selectable names.

**The runner grew hooks instead of opinions.** `run_batch`'s two stderr `eprintln!`s
became a `BatchProgress` callback (the CLI prints byte-identical lines; the server logs
`ServeLog::BatchProgress`), and a `cancelled` poll threads client-gone/deadline/
shutdown from the job's cancel token between items and per decoded token — with a latch
in `run_item` so an item whose decode was actually cut reports the cancellation in its
own `error` field while an item that finished just before the signal keeps its true
answer. A cancelled batch therefore still returns a truthful partial document to a
client that is owed one (deadline), and costs nothing to one that is not (gone).

**Cache discipline around a batch.** The batch runner owns the whole KV cache while it
runs (its shared-prefix snapshot is its own machinery, not the slot manager's), so
`run_batch_job` pages the live conversation out first — it survives warm in its slot's
host image, and on disk when the tier is on — and marks the engine dirty before the
first runner call; the existing post-job reset is what puts the cache back. A batch on
the same checkpoint therefore costs the next chat turn a page-in, not a re-prefill,
and a batch on the other checkpoint costs what the swap costs (~3 s load plus the
outgoing checkpoint's warm slots).

**Verification.** `cargo test --release` green: 780 + 69 passed, 0 failed (25 + 3
ignored as usual), including new tests for the endpoint's request shape (the CLI's,
`deny_unknown_fields`), per-request model resolution (named wins / absent means the
server default / unknown 400s), the queue-side size estimates, the config merge
leaving `p_min` unresolved unless pinned, and the scheduler's per-checkpoint discount.
No model math changed, so the parity gate was not re-run. The clippy warnings the arc
introduced were fixed (both `Job` variants are boxed for the queue's sake); the
pre-existing ones were left alone.

**Review (same day).** A two-family pass — a Claude reviewer with two corroborating
explorers, plus a Codex second-model review — confirmed the swap/dirty/disk-tier
architecture sound (disk gating airtight, no unlink-of-fresh-image path, no
sampler/grammar leakage, `state.size` cannot disagree with the loaded weights) and
surfaced six fixes, all landed within the arc. The one real correctness bug was NEW to
the arc, opened by exactly what the arc did: (1) **a chat job's thinking budget leaked
into the next batch job** — `run_job` arms `set_max_think` per generation but the
batch runner never touched it, and before this arc a batch always ran in a fresh CLI
process where the leak could not exist; a stale ceiling would fail every batch item it
did not fit and silently truncate the reasoning of the ones it did. `run_batch` now
clears both thinking controls up front. The rest: (2) an absent `model` on the HTTP
batch would RUN on the server's default but be LABELED with the CLI's compile-time
default — the handler now writes the resolved name back into the request before
queueing, so runner, job and response label cannot disagree; (3) a batch item cut
mid-decode kept its partial text beside `error`, contradicting `ItemResponse`'s
"failed items carry error and empty text" contract — the text is now dropped, usage
still counts the spent tokens; (4) the spec loop polled `should_stop` right after
committing its first token with no budget guard, so a `max_tokens: 1` item — the
single-token classification shape — whose deadline fired at exactly that instant
reported a complete, correct answer as cancelled; the poll is now guarded by
`decoded < max_new` like its sibling, AND (Codex caught the residual half) `run_item`'s
cut latch now defers to `hit_eog`: a decode that ended naturally — EOG, or a
single-token grammar value completing on its first draw — is a complete answer
however late the cancellation landed, on both decode paths; (5) `run_serve` resolved the official drafter
sidecar from `--model-size` before ever opening the GGUF, the last holdout against
this arc's file-is-authoritative rule — it now reads the served file's architecture
first (the one-shot CLI commands keep the flag's double duty deliberately); (6) the
scheduler's cache discount was blind to checkpoints — both models share a tokenizer,
so a long conversation re-sent for the non-resident checkpoint could ride a
token-level match against the wrong model's warm slots to the front of the queue,
then pay a swap plus a cold prefill; the cost closure now sees the whole job and gives
no discount across checkpoints. The per-checkpoint `p_min` resolution was extracted
into a testable helper (`resolved_p_min`) with a unit test, per the review's coverage
note. The Codex pass (gpt-5.6-sol) independently re-found the think-budget leak and
the cross-checkpoint discount, verified five of the six fixes complete, caught the
grammar-completion residual folded into (4) above, and added three smaller closes:
the compat dialects now parse the CLIENT's own model string rather than the
default-substituted echo (a GGUF someone names `35b.gguf` must not route model-less
requests by its basename), the batch size estimates saturate instead of overflowing
on hostile budgets, and the checkpoint-swap log line no longer claims imaging that
only happens when the disk tier serves the outgoing checkpoint. Nits documented
rather than coded around: the `--init` template's commented `p_min` line warns it is
the 35B's value; hf-hub's raw-stderr download bar under `--tui`, the route's inherited
~2 MB body limit, batch prefills not polling cancellation, and the zero-cost estimate
for schema-only items all joined the TODO ledger. Final count after fixes: 781 + 69
passed, 0 failed.

**First live exercise (same day).** `scripts/classify-demo.ts` was switched from
spawning `xwen batch` to POSTing `/xwen/v1/batch` — against a server already running
on the default port when one answers `/health` (a hazard guard as much as a
convenience: spawning a second server beside a running one would resident two ~20 GB
models), `--url` for anywhere else, else a server it spawns on a free port and tears
down. Its first run went against a live `xwen serve` and exercised the whole arc in
one process: the 35B lazy-loaded cold (6.1 s) for the first batch, the second batch
named the 27B and the engine swapped (14.5 s — 27B weights plus its 3.5 GB drafter
sidecar, cold), and both reports came back with the demo's documented accuracy shape
(35B 8/12 with the known near-tie misses, 27B 10/12), shared prefix 317 tokens cached
across all items on both. `--no-draft` now applies only to a server the script spawns;
against a running server, that server's config decides and the script says so.
