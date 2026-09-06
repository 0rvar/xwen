# 2026-09-05 — Per-run metrics on disk and `xwen stats`: every surface appends a JSONL record, aggregated by day/model/surface/session

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


**Context.** Nothing in xwen remembered a run after it finished. The `--stats` line and
the API usage objects report a run to whoever asked for it and then the numbers are
gone, so questions the machine should be able to answer — how much did this week cost,
which checkpoint did the work, is serve's prefix reuse actually hitting, which Claude
Code session spent the afternoon — had no source but scrollback. This arc gives every
surface a place to write one line and gives that file a reader.

**What ships.** `src/metrics.rs` plus a `stats` subcommand. Every finished run appends
one JSON object to `$HOME/.local/state/xwen/metrics.jsonl`: schema version, completion
timestamp, surface, checkpoint name, prompt/cached/prefill/decode tokens with the
seconds each phase took, `ok`, and the optionals a surface may know (thinking tokens,
drafted and accepted positions, batch items, client and session ids). An absent optional
means not measured and never 0, which matters most for `thinking_tokens`: serve always
writes it, 0 included, while `generate` and `chat` count thinking only under a think
budget. Seven
surfaces name themselves: `generate`, `chat`, `batch`, and `serve:{anthropic,openai,
native,batch}` — a served batch is submitted on the native dialect but costs nothing
like a native generation, so it gets its own label rather than being folded in.
Recording is on by default; `XWEN_METRICS_FILE` points it elsewhere or turns it off with
`off` in any casing, an empty value counting as unset. A failing write prints one
warning for the life of the process, which matters for a server that would otherwise
repeat itself once per request, and never fails the run. The checkpoint name is the
official full name when the GGUF identifies as one and the file's own stem when it
identifies as nothing, spelled identically on the one-shot surfaces and on the wire, so
a custom GGUF cannot appear under two names depending on which surface ran it.

**The three choices with a why.** The file is under the XDG state directory rather than
`~/.cache/xwen`, because a lost cache costs a re-download and a lost history is gone —
and `rm -rf ~/.cache/xwen` is a thing this project does routinely. A whole record goes
out in one `write_all` to an `O_APPEND` handle, so a server and a `generate` running at
the same moment interleave records and never fragments of one; the natural spelling, a
`BufWriter` or a `writeln!` pair, is the wrong one. And no dependency was added: dates
are Hinnant's `civil_from_days` and its inverse (forty lines, exact over the range), and
the local UTC offset comes from the `date +%z` shell-out the serve TUI was already
doing, moved into `src/metrics.rs` rather than written twice. Bucketing by UTC would
have removed the shell-out and cut the evening off the day the operator spent it in.

**`xwen stats` groups by `day|week|month|model|surface|client|session|all`**, filtered by
`--since 24h|7d|4w|YYYY-MM-DD` (a date means local midnight), exact `--model` and
`--surface`, substring `--client` and `--session`, with `--json` for the rows and
`--file` to read another history without ever recording into it. Every rate is
sum(tokens)/sum(seconds) over the bucket, never a mean of per-run rates: averaging rates
weights a two-token reply like an hour of decoding, and summing both sides first is also
the only version where the total row and the rows above it are computed the same way.
Where a bucket measured no seconds the cell is `-`, because zero tok/s is a claim and no
measurement is not. Malformed lines are skipped and counted in the stderr footer
alongside the path and the run count, so a piped table is nothing but its own rows. The
footer also says when `date +%z` could not be read and the days were bucketed in UTC,
that report being otherwise indistinguishable from one on a machine that really is UTC.

**`cached` needed a per-surface decision and batch needed two independent counts.** Serve
is the only surface with prefix reuse, so its cached count is the job record's
`cache_read`. `generate` runs from a reset cache and `chat` re-prefills the whole
conversation every turn, so both record 0 — that is the chat surface working as
designed, not a gap. Batch is the one that cannot be summed straight off the items: one
record covers the whole run, and every item's `cached_prefix_tokens` counts the shared
prefix, so adding that column multiplies the prefix by the item count. Cached is that
sum minus the shared prefix, prefilled once and read back for every item after. Prefill
is measured on its own as what the engine forwarded, the runner's figure plus the shared
prefix, whose tokens and time sit outside the runner's accounting in
`shared_prefix_tokens`/`snapshot_ms` — hence the recorded seconds being `prefill_ms +
snapshot_ms`. Neither count is derived from the other, and `prompt = cached + prefill`
is an ordinary-run identity rather than an invariant: a scored batch forwards more than
its prompt, an abandoned or failed serve job less. The `/xwen/v1/batch` route's
`BatchSummary` folds both prefill halves the same way.

**A failed run is a record, not a gap, and `ok` is stricter than "no error".** A history
that only holds successes reads a bad afternoon as an idle one, so every error path
writes a record. `ok` means the run reached an end-of-generation token or its token cap
with nothing thrown: a served job the client hung up on, one a deadline killed, and a
chat turn cancelled with Ctrl-C are all `ok` false, because their counts stop wherever
they were interrupted and averaging them in drags every rate down. Those records still
carry the tokens they reached. Zero counts mean zero measurements — a `generate` whose
generation errored, a turn that died in prefill, a CLI `batch` that failed as a whole
request, a served batch holding only the queue's bytes estimate. Every surface records
its own failures, so the only run that leaves nothing behind is one whose process was
killed. A served batch takes its surface from the job's kind at pickup
rather than from a summary it may never have written, so one that dies before the runner
returns is still a `serve:batch` row and not a `serve:native` one. On the reading side
`ok` separates populations rather than filtering them: a bucket sums every record the
query matched, unfinished ones included, and reports how many of them were unfinished in
the `--json` rows, the table keeping its columns. A batch counts as failed only when
every item failed. A CLI `batch` failure records the model as `-`, since most
whole-request failures happen before the payload names a checkpoint. The
acknowledged hole is process-level Ctrl-C: `generate` installs no signal handler and an
interrupted one leaves nothing behind, where the REPL already owns Ctrl-C as a cancel
and records the turn.

**The identity research, and what it settled.** Both the session and the client id were
accepted-and-dropped before this arc, and the question was which one a report should be
keyed on. `x-claude-code-session-id` is Anthropic's documented per-session header for
gateways and Claude Code sends it on every request, so it is a single value with a
single meaning and needs no parsing. `metadata.user_id` (and OpenAI's `user`) is the
opposite: undocumented, and observed in two different shapes across Claude Code
releases — an underscore-joined `user_<hex>_account_<uuid>_session_<uuid>` in one
published capture, a JSON blob carrying `"session_id"` in another. Parsing it at write
time would bake this session's reading of an unstable format into a file meant to
outlive it, so the raw value is stored and `session_key` does the interpreting: header
first, then whatever follows the last `session_` marker, then `-`. Both ids are cut to
128 characters, the file being the one place a client's own bytes land unfiltered.
`x-claude-code-agent-id` is deliberately not recorded — it would split one session into
a row per subagent. **Not confirmed:** whether the header's id is the same id
`claude --resume` lists as the transcript id. Nothing here depends on it, but a report
that claimed it would.

**The review round.** Twenty findings came back before this shipped and all twenty are
fixed. Five were material. Batch accounting was wrong in both directions: cached was
taken as prompt minus prefill, which goes negative and saturates to 0 on a scored batch,
and prefill excluded the shared prefix whose time was nevertheless being counted, so the
run's own prefill rate was inflated. A served batch that failed before its runner
returned was filed as `serve:native` with a native generation's numbers, because the
surface was read off the presence of a summary rather than the job's kind. Reading the
client and session ids had narrowed the two request fields enough that a wrongly-typed
`user` or `metadata` became a 400, which is this server's accept-and-drop policy broken
by a feature that only wanted a label. A custom GGUF recorded under the official
checkpoint whose graph it runs, so one file could appear under two names depending on
the surface. And a test set `XWEN_METRICS_FILE` in the process environment, which is
undefined behaviour beside every other thread in the runner; the path rule is now a
function over values (`metrics_path_from`) and the test passes it arguments. The rest
were smaller: the `off` casing, the empty-value case, the reader losing a whole file to
one torn UTF-8 line, `--since 2026-02-31` being accepted, the writer's directory
creation, error paths on the CLI surfaces recording nothing at all.

The outside-model pass that followed ran on Qwen3.8-Flash-Next through `qwen-review`,
xwen reviewing xwen. Codex could not be reached for it — the login on that wrapper is
revoked — so this round had one outside family rather than the usual two. Qwen
contributed two findings, both the same thing seen from two surfaces: a serve job
abandoned when its client disconnected or its deadline expired was recorded `ok` true,
and so was a chat turn the reader cancelled with Ctrl-C. Both stop counting wherever
they were interrupted, so both would have sat in the table as completed runs quietly
pulling every rate down. `ok` now means "reached its own end", which is what the two
cases have in common and what neither of them did. The stderr note when `date +%z`
cannot be read, without which a UTC-bucketed report is indistinguishable from a report
on a UTC machine, came from the same pass.

A second round over those fixes found four more. The table measured its label column in
`char`s, so a CJK or emoji label broke the alignment of every row under it, and a raw
client id could run the table off any terminal; the column is measured in display
columns now (`unicode-width` was already a dependency, so this cost nothing new) and cut
to 48 with an ellipsis. `generate` was the last surface whose failure path recorded
nothing, which made "every surface records its runs" not quite true and made a prompt
that outgrew `--max-ctx` invisible. The REPL's record warning was going out through a
path that does not return the carriage in raw mode. And `--since` was parsed after the
empty-history early return, so a typo on a fresh machine answered "no metrics recorded
yet" instead of saying what was wrong with the argument.

**Verification, and what it is not.** 26 tests in `src/metrics.rs` (record shape,
session-key derivation across both observed client shapes, `--since` parsing including a
day its month does not have, the civil date helpers across leap boundaries, bucketing,
the sum/sum rates, malformed and torn-line skipping, the environment rule, the label
column's width arithmetic) and serve at
442 against 428 before the arc — the surface split including `serve:batch` decided from
the job kind, the failed-batch cases, and the header and body ids captured per dialect,
malformed ones included. 11 in the binary and 82 in batch. The report path was exercised
against a hand-written fixture rather than a real history. **No model was run end to end
for the smoke** — another model process held the GPU for the session, and the
operational rule here is one large model process at a time. So the assertion "a real
`generate` lands a line in the real file" is tested at the unit level and unproven at
the surface, and running one is the first item on the ledger.

**Deferred (TODO.md, "Deferred from the metrics arc").** The `[metrics]` serve.toml
table; the bench and parity scripts recording into the same file and skewing usage
stats; `x-claude-code-agent-id`; the unconfirmed header-vs-transcript id; the file
growing without rotation, `--since` being the only bound; the single UTC offset a report
applies to all of its records, which buckets a run on the wrong local day across a
daylight-saving change; the end-to-end smoke above; and the serve TUI still not showing
a job's model though `JobRecord` now carries it.
