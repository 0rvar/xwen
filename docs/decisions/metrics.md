# Metrics

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic; moved
here verbatim on 2026-09-06. Dated paragraphs, newest additions appended within their topic.


**The history is an append-only JSONL file, not SQLite and not a binary log.** What a
run records is ten-odd scalars written once, at the end, by a process that is about to
exit; what a reader does with them is a full scan and a group-by over a file that grows
by a line per run. A database buys indexed lookup and concurrent writers, and this
workload wants neither: `xwen stats` reads the whole file in milliseconds at any size a
year of this machine's runs produces, and the writers are already serialized by
`O_APPEND`. It also costs a dependency and a schema migration path, against a format
`jq`, `grep` and `wc -l` already read. The one property JSONL had to earn rather than
inherit is atomicity, which the single-write rule below gives it (2026-09-05).

**It lives in the XDG state directory, not the cache directory.** `~/.cache/xwen` holds
things xwen can rebuild: a lost cache costs a re-download. The metrics file is the
opposite — nothing regenerates last month's runs — so it goes to
`$HOME/.local/state/xwen/metrics.jsonl`, which is exactly what the XDG basedir spec
means by state: data that should persist between restarts but is neither configuration
nor cache. Putting it beside the weights would have made `rm -rf ~/.cache/xwen`, a thing
this project does routinely, silently destructive (2026-09-05).

**One `write_all` of the whole line, terminator included.** Two xwen processes recording
at the same moment is the normal case, not the edge one: a server runs for days while
`generate` and `batch` runs come and go against the same file. An `O_APPEND` write below
the pipe buffer is atomic, so the rule that keeps records whole is simply never to split
one across two writes. Formatting the line into a `String` first and writing it once is
the whole implementation. A `BufWriter` or a `writeln!` pair would have been the natural
spelling and the wrong one (2026-09-05).

**`XWEN_METRICS_FILE` is an environment variable, with no `serve.toml` key.** The
setting has to reach four surfaces, three of which have no config file at all, and the
thing operators actually want is to turn recording off for one command
(`XWEN_METRICS_FILE=off xwen generate …`) or point a script at a scratch file. A serve
config key would cover the one surface where a flag is least needed and leave the other
three on the variable anyway. The `[metrics]` table is ledgered rather than refused, for
the day the server wants a path it does not share (2026-09-05).

**Dates are hand-rolled, not a `chrono`/`time` dependency.** What the report needs is a
civil date from a Unix timestamp, the Monday of a week, and the local midnight of a
named day. That is Howard Hinnant's `civil_from_days` and its inverse, forty lines and
exact over the whole representable range, against a dependency tree pulled in for
formatting a `YYYY-MM-DD`. The local offset comes from `date +%z`, which the serve TUI
was already shelling out for; the helper moved into `src/metrics.rs` rather than being
written twice. Bucketing by UTC instead would have removed the shell-out and cut the
evening off the day the operator spent it in, which is the wrong trade for a table
someone reads to ask what they did today (2026-09-05). A shell-out can fail, and when it
does the report buckets in UTC and the stderr footer says `dates in UTC (local offset
unavailable)`: a report bucketed in UTC because the offset could not be read is
otherwise indistinguishable from one on a machine that really is UTC, and silently
handing someone the wrong days is worse than handing them the right ones with a
caveat. Known cost of reading the offset once rather than per record: a single value is
applied to every record in a report, so a report spanning a daylight-saving change
buckets the far side of it an hour off, which near midnight puts a run on the wrong
local day (ledgered in TODO.md).

**A bucket's rate is sum(tokens) / sum(seconds), never the mean of its runs' rates.**
Averaging per-run rates weights a two-token reply exactly like an hour of decoding, so a
day with a hundred one-liners and one long generation reports the one-liners' number.
Summing both sides first gives the rate the machine actually ran at over the bucket,
which is the only figure that composes: a row's rate and the total row's rate are
computed the same way and the total is not a mean of the rows. Where a bucket measured
no seconds at all the cell is `-` rather than 0, because zero tok/s is a claim and no
measurement is not (2026-09-05).

**`cached_tokens` is per-surface by construction, and on a batch neither token count is
derived from the other.** Serve is the only surface with prefix reuse, so its cached
count is the job record's `cache_read` and everything else follows from it. `generate`
runs from a reset cache and `chat` re-prefills the whole conversation every turn — that
is what the chat surface does, not a limitation being papered over — so both record 0
and their prompt is entirely prefill. Batch is the one that cannot be read off the items
as they stand: a batch is ONE record covering the run, and every item's
`cached_prefix_tokens` counts the shared prefix, so summing that column multiplies the
prefix by the item count. Cached is therefore that sum minus `shared_prefix_tokens`, the
prefix having been prefilled once and read back for each item after. Prefill is measured
separately as what the engine forwarded, `BatchStats::prefill_tokens` plus the shared
prefix, because the runner opens its own prefill accounting only after the prefix is
resident and reports that span as `shared_prefix_tokens`/`snapshot_ms` — which is why
the recorded seconds are `prefill_ms + snapshot_ms`, so the pair still describes a rate
that was observed. Taking cached as prompt minus prefill instead, which is how the arc
first drafted it, is wrong on exactly the run that matters: a scored batch forwards
teacher-forced trials against no prompt token at all, so the difference goes negative
and saturates to 0. The consequence to state rather than hide is that
`prompt = cached + prefill` is an ordinary-run identity and not an invariant — a scored
batch exceeds it, an abandoned or failed serve job falls short. `/xwen/v1/batch` reaches
the same two numbers by its own route: the fold already subtracts the shared prefix into
`cache_read`, and its `BatchSummary` folds both prefill halves together, so the served
batch and the CLI one report the same thing about the same work (2026-09-05).

**`ok` is "the run reached its own end", not "the machinery did not throw".** The narrow
reading — `ok` false only on an error — was what shipped into review, and it counts a
job the client hung up on as a completed run. That is the reading that quietly corrupts
the table it feeds: an abandoned job's counts stop wherever the interruption landed, so
averaging them in drags every rate down with runs that were never going to finish. So
`ok` means an end-of-generation token or the token cap with no error, and a deadline
kill, a client disconnect, a shutdown and a chat turn cancelled with Ctrl-C are each
`ok` false beside the outright failures. The test the definition has to pass is
cross-surface: a reader who gives up on a generation does the same thing whether they
are in the REPL or closing a client against the server, and the history has no business
reporting that one action three different ways depending on which surface caught it.
"Reached its own end" is the phrasing every surface can answer identically; "no error"
is not, because only some surfaces route an abandonment through an error at all.
Recording them at all is the other half of the choice: a history that only holds
successes reads a bad afternoon as an idle one, and the error rate of a server is then
the one thing its own history cannot answer.

`ok` is a discriminator and never a filter. `xwen stats` sums every record its query
matches, unfinished ones included, and reports how many of them were unfinished per
bucket in the `--json` rows. Dropping them from the aggregate would have been the other
way to keep interrupted runs from dragging the rates down, and it hides work that really
happened and really cost GPU time; carrying the count instead lets a reader separate the
two populations without the file having decided for them.

**A failed or interrupted record carries what was measured, and zero only when nothing
was.** These are not the same claim as the one above and it is worth keeping them apart.
An abandoned serve job and a cancelled chat turn both spent real tokens, and those
counts are real, so they are written; `ok` is what says not to average them in. A record
whose counts are all zero is one where nothing was ever measured: a `generate` whose
generation errored, a chat turn that died during prefill and never produced stats, a CLI
`batch` that failed as a whole request, a served batch that never reached the runner and
holds only the queue's bytes-based estimate, which is not a measurement and is recorded
as nothing rather than as a number. What this settles into is worth stating as a rule,
because otherwise the exceptions are what a reader has to carry: every surface records
its own failures, and the only run that leaves nothing behind is one whose process was
killed.
Two details around it are load-bearing rather than incidental. A served batch takes its
surface from the job's own kind at pickup, not from the presence of a summary, or a
batch that died before the runner returned would file itself under `serve:native` with a
native generation's numbers. And "failed" for a batch means every item failed; a batch
that lost some items still did the rest, and the same rule holds on both batch surfaces.
A CLI `batch` failure records the model as `-` rather than a guess, most whole-request
failures happening before the payload has been read far enough to name a checkpoint. Not
covered, deliberately: Ctrl-C on the `generate` process leaves no record, there being no
signal handler and no obviously right thing for one to write — the REPL is different
because it already owns Ctrl-C as a cancel and the turn is still there to record
(2026-09-05).

**An absent optional means not measured; it never means zero.** The fields a surface may
not know are `Option`s and are omitted from the line rather than written as 0, so a
reader can tell "this surface does not measure that" from "it measured 0". The case that
forces the rule is `thinking_tokens`: serve counts thinking on every job and writes it
including 0, while `generate` and `chat` only count it when a think budget is in effect,
so a thinking run without one records no thinking count at all. Writing 0 there would
make a whole class of run look like it never thought, which is a false statement in a
file meant to be read a year later, where the alternative is only a gap. The cost to
know: summing the field sums the runs that measured it, not the runs that thought, so a
thinking-token total across mixed surfaces is a serve figure with the one-shot runs
missing rather than a wrong number (2026-09-05).

**The metrics label for a one-shot run is the checkpoint's full name, or the GGUF's file
stem when the file identifies as nothing.** `generate`, `chat` and `batch` resolve a
`Model` to run as, and for a custom GGUF that answer is which graph to execute, not
which checkpoint this is. Recording it as the model would file someone's finetune under
`Qwen3.6-27B` permanently, in a file nothing rewrites. Serve already drew this
distinction for the wire (`serve::model_id`), so the CLI surfaces reuse the same string
for the same file, which is what keeps `--by model` from splitting one file across two
names depending on which surface ran it. The `BatchResponse`'s own `model` field is a
different question — the checkpoint the run answers as — and was left alone
(2026-09-05).

**The session id is read from the header and the client id is stored raw.** Both were
accepted-and-dropped before this arc. `x-claude-code-session-id` is Anthropic's
documented per-session header for gateways and Claude Code sends it on every request, so
it is the identifier a "which session was that" question should be answered from: one
value, one meaning, no parsing. The body id (`metadata.user_id` on Anthropic, `user` on
OpenAI) is the opposite — undocumented, and observed in two different shapes across
Claude Code releases, an underscore-joined string in one capture and a JSON blob
carrying `session_id` in another. Parsing it at write time would bake this session's
understanding of an unstable format into a file meant to outlive it, so the raw value is
what gets stored and the reader does the interpreting: `session_key` prefers the header,
falls back to reading past the last `session_` marker, and can learn a third shape later
without the old records being wrong. `x-claude-code-agent-id` is deliberately not
recorded — it would split one session into a row per subagent — and is ledgered
(2026-09-05).

**Reading the ids did not make either of them validated.** Both dialects accept and drop
what they do not understand, and a field nothing in the reply depends on must never be
the thing that fails a request, so `metadata` and `user` are held as raw JSON values
exactly like `tool_choice`. A `metadata` that is not an object, an absent or null
`user_id`, a `user` that is a number or an array: each yields no client id and a 200,
and on the Anthropic side a `user_id` that is present but not a string is kept as its
own compact JSON rather than dropped, the shape having moved before and a reader being
able to find a session marker in either spelling. The arc's first shape typed the two
fields narrowly enough that a wrongly-typed one became a 400, which review caught: a
feature that only reads an identifier for a report had no business changing which
requests a server answers. Empty strings normalize to absent in one place,
`ClientId::new`, so
the two dialects cannot drift into disagreeing about what an empty id means. The one
strictly-validated request field on this server remains `chat_template_kwargs`, for the
reason recorded under Serving (2026-09-05).

**Client-supplied ids are cut to 128 characters.** Nothing legitimate comes close: a
session uuid is 36. The bound is not about disk, it is about the file being the one
place a client's own bytes land unfiltered, and an unbounded id is a request-at-a-time
lever on a file nothing prunes (2026-09-05).

**Recording is on by default on every surface, benchmark runs included.** The obvious
refinement is to exclude sweeps and parity runs so the history reflects real use, and it
is the wrong default: a metrics file that silently omits some runs is worse than one
that over-counts, because the omission is invisible at read time and the over-count is
visible the moment someone groups by day and sees the sweep. `--surface` and `--since`
already carve a bench session out after the fact. Whether the scripts should set
`XWEN_METRICS_FILE=off` or tag their records is ledgered rather than decided here
(2026-09-05).

**Records carry a `v` and readers ignore fields they do not know.** The schema will
gain fields — per-item batch rows, a request id, whatever the next surface measures —
and the file is append-only, so old records and new ones sit in the same file forever
and every reader has to handle both. Serde's default behaviour does the ignoring;
`v` exists so a future change that is not additive has something to branch on rather
than having to guess from which keys are present (2026-09-05).

**The agent id is recorded after all, as its own field with its own grouping.** The
paragraph above retired `x-claude-code-agent-id` for a good reason, which still holds:
folding it into the session key would split one session into a row per subagent, and
"which session was that" would stop having an answer. What that reasoning missed is that
the two questions are different questions. A session-keyed report answers where a
conversation's tokens went; an agent-keyed one answers which subagent spent them, and on
a machine where most of the decoding happens inside fanned-out agents that is the more
actionable of the two. So the header is read beside the session one, bounded the same
128 characters, and stored in `agent` — a field of its own, never a fallback inside
`session_key`, which is untouched. `--by agent` groups on it and labels every run
without one `-`, the same marker `--by client` and `--by session` use, so the subagent
traffic and the rest of a session read as separate rows without either grouping
disturbing the other. Recorded on the owner's decision (2026-09-06).
