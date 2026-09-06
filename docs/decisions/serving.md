# Serving

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic; moved
here verbatim on 2026-09-06. Dated paragraphs, newest additions appended within their topic.


**The serve/ tree (Anthropic + OpenAI + native dialects, TUI, queue, prefix cache, disk
tier) is inherited as-is**; it is architecture-agnostic. The KV export/import and disk
tier must additionally carry the recurrent state for the 3-of-4 linear layers — KV cache
alone no longer reconstructs a prefix. Native endpoint moved `/maxuna/v1/*` →

**A mid-conversation system turn demotes to a user turn on the wire dialects
(2026-08-31).** Harnesses inject system-role messages past the head of the conversation
— Claude Code's token-budget reminders are the live case — and the Anthropic messages
spec does not even define a `system` role inside `messages`. The choice was demote
or relax: the official templates for every checkpoint hard-raise
`System message must be at the beginning.` in their message loop, so rendering a
mid-stream system block is formatting the checkpoint's template forbids (llama.cpp,
applying the template verbatim, 500s on the same request) — relaxing was refused on
that evidence alone. `push_turn` in both dialects demotes the turn to user in place,
merging into adjacent user text; position is preserved deliberately (the reminder's
meaning is positional, and this matches how Claude Code embeds `<system-reminder>`
blocks in user turns itself). The renderer stays template-faithful and chat.rs's
refusal remains as the backstop for the direct chat surface.

**Two cache slots and an opt-in disk tier (2026-08-30; were four and on-by-default).**
Both defaults were set when the default checkpoint's image was a few MB of DeltaNet
state plus 4 KiB/token. The default is now Flash-Next: 30 KiB/token (2 KV heads at 256
in f16 plus the QSA indexer's f32 key row, over 12 layers) and a 113 MiB DeltaNet floor,
so a conversation at the checkpoint's 262144 context is an ~8 GB image. Four slots is
~33 GB of host RAM beside the model; the disk tier writes that same image per
conversation, and SSD wear is the operator's cost, not the server's. Two slots keep one
host image beside the live conversation (the two-agents case); `--disk-cache` /
`disk_cache = true` turns the tier on for anyone who wants restarts to resume. The
`--cache-slots` / `[cache] slots` and `--no-disk-cache` surfaces are unchanged.
`/xwen/v1/*` (2026-07-28).

**One server serves both checkpoints; `--model` is only the default (2026-08-11).**
Every job (generation or batch) names the checkpoint it needs, and the engine's pickup
compares that against what is resident: a mismatch images the live conversation out
through the same path an idle unload takes, drops the state, and lazy-loads the named
checkpoint — one model resident at a time, by construction, which is this machine's
memory invariant, not a scheduling choice. The alternatives were refused deliberately:
keeping both models loaded risks GPU OOM (CLAUDE.md operational hazards), and a
temp-generator-per-foreign-batch would reload on every consecutive same-model batch
where the swap design keeps the second one warm. Which checkpoint the default GGUF is
comes from its `general.architecture` (`Arch::model()`), never from `--model-size` —
the flag and a `-m` path can disagree, the file cannot. The same rule now covers the
startup drafter resolution (review fix, same day): `run_serve` reads the served GGUF's
architecture before resolving the official sidecar, so a config-file `model` (or `-m`)
that disagrees with `--model-size` gets the sidecar for the model actually served
rather than a geometry error blaming the drafter. The one-shot CLI commands keep the
flag's old double duty deliberately — there the flag and the payload are the intent.
Selection strictness followed
each surface's existing character: the batch route 400s an unknown model (the field
exists to select), the compat dialects fall back to the default (SDKs send their own
model ids and must keep working), the native generate endpoint stays modelless.
SUPERSEDED in two places on 2026-08-14 — the architecture stopped identifying a
checkpoint (see "The GGUF names itself" below), and the compat dialects stopped falling
back (see "On the wire a checkpoint has exactly one name").

**On the wire a checkpoint has exactly one name — its full name — and an unknown one
is a 400 on every surface (2026-08-14).** Two bugs shared a root: the model vocabulary
was the CLI's. `/v1/models` listed the served file's basename AND every checkpoint's
short alias, so one model appeared under two ids (a live server listed
`Qwen3.6-35B-A3B-Q4_K_M`, `27b` and `35b` for two checkpoints), which is not a listing
a client can pick from; and the compat dialects' fall-back-to-default meant an SDK's
own id (`gpt-4o`) was answered by whatever checkpoint the server happened to default
to, indistinguishably from a correct request. Both are fixed by naming: the APIs accept
and echo `Model::full_name()` only — the string ggml-org names the repo with and the
GGUF carries as `general.name`, quant-independent — while `--model-size` keeps the
short aliases (and now also accepts full names, so a `/v1/models` id pastes into the
CLI). Absent or empty `model` still means the served checkpoint; anything unrecognized
is a 400 in the dialect's own error format, listing the valid names. The one real cost
is deliberate: a client that used to get an answer for `"model": "35b"` now gets a 400
telling it what to send, which is the SDK-default surprise turned into a message
instead of a silently wrong model. Quant is not part of the name — one server serves
one file per checkpoint, and the response's job is to say which MODEL answered.

**The GGUF names itself; the architecture is only a fallback (2026-08-14).** Adding
Qwen3.8-27B broke `Arch::model()`'s one-to-one claim: two releases now ship the dense
`qwen35` graph with byte-identical configs, so the architecture can no longer say which
checkpoint a file is. The identification chain is now explicit `--model-size` (the
operator naming a custom file), then `general.name` (both blessed files carry their
exact full name; a substring pass catches a re-quantized conversion), then the file
name, then `Arch::model()` as the last resort with a logged warning. `Arch::model()`
survives as exactly that fallback and its doc says so. It matters because the identity
picks the hub repo and the sidecar: guessing 3.6 for a 3.8 file would attach the wrong
drafter to a graph that accepts it, costing acceptance rather than failing.
SUPERSEDED in its ordering the same day (review round): `--model-size` is NOT the first
link and does not override the file — see "`--model-size` is a tie-break, not an
override" below. The rest of this paragraph stands.

`Model::identify` uses the architecture only to NARROW the candidates, never to answer —
including for the MoE graph, which one official checkpoint ships. Answering from the
arch there would have been safe for the engine and wrong for the API: any conversion
onto that graph would then be reported by `/v1/models` and every response under
`Qwen3.6-35B-A3B`, which is a claim about weights nobody checked. Unidentified files
therefore keep reporting under their own file name (unchanged behavior) while the engine
still runs them as the architecture's checkpoint — the identity and the id are two
questions, and only the first has a safe default.

Both name sources are read by ONE rule (review round, same day): an exact full name, or a
whole full name found inside the name, case-insensitively — which accepts the shapes real
files take (`Qwen3.8-27B-Q8_0`, `Qwen3.6-27B-Instruct`) while requiring that a complete
checkpoint name actually appear. `general.name` is consulted before the file name because
it is what the converter wrote INTO the file, but it earns no looser matching for it.
Matching a bare release series was tried and refused twice over: it identifies
`My-Qwen3.6-14B-finetune.gguf` as the official 27B, and — since `Arch::Moe` has exactly
one candidate, so no ambiguity check can save it — `MyMoE-3.6` as Qwen3.6-35B-A3B. Either
one answers an official name with weights nobody checked, which is the single thing this
function exists to prevent. A name matching more than one checkpoint identifies as none
rather than as whichever `MODELS` lists first. The blessed files are unaffected: their
`general.name` is the exact full name, verified on all three.

**`--model-size` is a tie-break, not an override (2026-08-14, review round).** It names
the checkpoint a file that says nothing about itself holds. Against a file that DOES say,
a contradicting flag is a startup error naming both sides, and it must agree with the
architecture too. It had silently won, which meant the server started clean and then
failed `EngineState::load`'s own arch/identity checks on every request — a 500 per
request for a mistake that was fully knowable at startup. Those load-time checks remain
as a backstop for the case they can still catch: a file replaced under a running server.

**A job names a FILE, not just a checkpoint (2026-08-14, review round).** `Target` is a
checkpoint plus "is this the served file". The distinction only exists on a server whose
GGUF identifies as none of the official checkpoints, and there it is the whole ballgame:
the official checkpoint of the same architecture is a DIFFERENT FILE that happens to size
its caches identically, so a bare `Model` could not say which was meant. With it: the
served file answers for its own advertised id (which the resolver now accepts — it was
400ing the one id `/v1/models` published), an official name resolves that checkpoint's
real hub file and swaps to it like any other checkpoint, and the batch document is
labeled with the id that answered rather than with the arch fallback's full name. Two
things fall out for free: `Target` equality is the engine's swap check (two files, two
targets), and the disk tier's binding is a `Target` comparison rather than a checkpoint
one — the tier is bound to `settings.model`, which is a file.

**The disk tier stays bound to the default checkpoint.** `DiskCache::open` scans and
binds at startup against `settings.model`, and `verify()` deliberately disables the
tier for the rest of the process if weights with a different checkpoint id load. Under
a non-default checkpoint every engine call site is therefore handed `None` instead of
the tier: feeding it foreign images would poison the store with bytes that claim the
default binding, and verifying it against foreign weights would permanently disable it.
The segment layout is already per-checkpoint directories, so a tier-per-checkpoint is a
straightforward lift when a workload wants it (TODO.md, 2026-08-11 arc).

**Speculation is decided per checkpoint, not per process (2026-08-14, review round).**
`ServeSettings.draft` was one resolved `Option<PathBuf>`, which was correct while
"official sidecar" named one file and became silently wrong when a checkpoint shipping
none arrived: a server whose DEFAULT checkpoint was Qwen3.8-27B ran every OTHER
checkpoint plain too, costing the 27B its measured +46 to +52% with nothing in the log
about it. It is now a `DraftMode` — `Off`, `Official`, or `Custom(path)` — and
`checkpoint_paths` resolves `Official` when a checkpoint loads, so each drafts with its
own sidecar and a sidecar-less one decodes plain with its own line. A `Custom` path still
belongs to the checkpoint it was validated against and never transfers (2026-08-11,
unchanged); any other checkpoint falls back to its official sidecar rather than borrowing
it. Startup validation follows the same split: a custom drafter is judged at startup as
before, an official sidecar when the checkpoint that owns it attaches it, and the served
checkpoint's sidecar is still PREFETCHED at startup so a first request does not stall
behind a 3.5 GB download. The dashboard's drafting cell tracks the loaded checkpoint
(`ModelLoaded` clears it, `DrafterLoaded` sets it) instead of the setting, for the same
reason the setting stopped being the answer.

**`draft.p_min` resolves at drafter attach, not in the config merge (2026-08-11).**
The merge used to bake `--model-size`'s per-checkpoint default into the settings, which
was correct while one process meant one checkpoint and silently wrong the moment it
did not — the fitted floor is a property of the loaded checkpoint. Unset now stays
`None` through the merge and each load applies `Model::draft_p_min_default()`; an
explicit value pins one floor for every checkpoint served, which is exactly what the
CLI flag means. A non-default checkpoint always speculates with its official sidecar —
a custom `draft.path` belongs to the default checkpoint alone, since sidecars never
transfer between checkpoints.

**`/xwen/v1/batch` is the CLI's document over HTTP, run as one queue job
(2026-08-11).** Same request JSON as `xwen batch` stdin, same response document as its
stdout — one surface, two transports; the core was written transport-agnostic for
exactly this. The job rides the ordinary `JobQueue` (so it serializes with chat
requests, honors shutdown, and can be swept when its client leaves) as a second `Job`
variant whose single terminal event carries the whole response. Scheduling and the
watchdog deadline use a bytes/3 token overestimate, which errs the right way twice:
small chat requests keep jumping ahead of a batch still in the queue (once picked, a
batch runs to completion — the batch=1 engine preempts nothing), and the deadline errs
loose. The
runner's stderr `eprintln!`s became a `BatchHooks` progress callback — the CLI prints
the same lines as before, the server routes them into its log — and the same hooks
carry cancellation: client-gone/deadline/shutdown fold into the job's cancel token,
polled between items and per decoded token, and items the cancellation reached report
it in their own `error` field so a deadline still yields a truthful partial document.
The batch marks the engine dirty up front and the live conversation is paged out before
it runs: the runner owns the whole cache, and the existing post-job reset machinery is
what puts the cache back.

**The request-body cap is an explicit 100 MB, replacing axum's implicit 2 MB
(2026-08-11).** The implicit cap was never a decision — nobody chose 2 MB, axum's
default arrived with the framework — and it bound first in practice: a real client
split one batch over a 377 KB story into 14 POSTs to fit under it, re-prefilling the
shared prefix each time. The wire is the wrong layer to police cost: the queue's
bytes/3 token estimates and max_ctx judge what a request actually costs, and both keep
doing so at any body size. 100 MB is far past any request the engine can serve while
still bounding a hostile stream; it covers every dialect on the API router (`/health`
carries no body). Two accepted edges, both recorded in the ledger: `Router::layer`
wraps only the routes registered before it, so a POST route added after that line
silently gets axum's 2 MB default back — the layer call carries the warning; and
bodies are buffered and parsed BEFORE the queue can answer 429, with no concurrency
bound on connections, which is acceptable exactly because the default bind is loopback
on a single-user machine (2026-08-11).

**One `reasoning_effort` field drives both the think budget and the 3.8 template
preamble, with off-scale levels nearest-mapped instead of raised (2026-08-19).** The
OpenAI dialect's field kept its budget mapping (none=off, minimal=1024, low=4096,
medium=16384, high/xhigh/max=uncapped — the budget scale is this server's own) and now
also selects the template level a 3.8 prompt renders: minimal→low, high/max→xhigh. One
knob rather than two because a client saying "low effort" means low effort, not "cap
the tokens but instruct the model to think hard" — the split-knob reading was the
"conflicting field" the 2026-08-14 ledger item worried about, and it dissolves once the
field drives both. The nearest-mapping is a deliberate divergence from llama.cpp, which
passes the raw string into the jinja and lets the template raise: `minimal`, `high` and
`max` are real levels of this API's vocabulary that the template happens not to define,
and answering them with the nearest defined level serves the request where upstream
turns it into a template error. The level is dropped whenever thinking resolves off, because the template
renders it only inside the thinking guard. Clients that want the raw template parameter
have it: `chat_template_kwargs.reasoning_effort` takes exactly the template's three
levels, and the top-level field wins over the kwarg — llama.cpp's precedence, kept so a
client speaking both shapes gets upstream's answer. The server-wide default
(`[thinking] effort` / serve `--reasoning-effort`) fills in when a request names
nothing, and `count_tokens` renders under the same default so a count matches the
generation it predicts. The native dialect exposes the raw parameter directly
(`reasoning_effort`, three levels only) plus `preserve_thinking`; the Anthropic dialect
exposes no per-request effort knob — its API has no natural field for it, so the
server-wide default applies (TODO.md).

**`chat_template_kwargs` is validated strictly — the one exception to the compat
dialects' accept-and-drop permissiveness (2026-08-19).** The OpenAI dialect accepts and
drops sampling parameters this engine has no equivalent for (penalties, logprobs,
`min_p`) because a client sending them still gets a correct completion, merely sampled
without them. Template kwargs are a different category: they steer the rendered prompt
itself, so a kwarg silently ignored means the client got a DIFFERENT PROMPT than it
believes it asked for, with nothing anywhere saying so. An unknown key, a wrong type,
or a `reasoning_effort` outside the template's three levels is therefore a 400 naming
the offender (the error for an off-scale level points at the top-level field, where the
wider none/minimal/high/max vocabulary belongs). The accepted keys are the three
parameters the vendored templates actually take — `enable_thinking`,
`preserve_thinking`, `reasoning_effort` — the official Qwen card's (and vLLM's) request
shape.

Extended 2026-08-19 (the arc's review pass): a request-level TEMPLATE effort on a 3.6
target is a 400 by the same argument. `chat_template_kwargs.reasoning_effort` (and the
native dialect's `reasoning_effort` field — the same raw parameter) name a parameter
only the 3.8 template defines; on a 3.6 checkpoint they would render nothing, which is
precisely the silently-different-prompt outcome strict validation exists to prevent,
and it contradicted the CLI, where `--reasoning-effort` on a 3.6 checkpoint had been a
startup error since the arc landed. llama.cpp would silently ignore an unused kwarg —
a deliberate divergence, consistent with this repo's cross-check-instead-of-shrug flag
policy. Both `prepare`s take the resolved `Target` and the error names the model. The
boundaries, each deliberate: the OpenAI TOP-LEVEL `reasoning_effort` field stays
accepted on 3.6 (it carries budget semantics on every checkpoint, and the error points
clients at it); kwargs `enable_thinking`/`preserve_thinking` stay accepted on 3.6
(real parameters of that template); and the server-wide `[thinking] effort` default
stays inert-but-legal on 3.6 — it is an operator setting covering whatever checkpoints
a server serves, not a request asking this model for a level.

Extended again 2026-08-19 (later still): the batch surface applies the same refusal
with the module's own failure shape. A `reasoning_effort` on a batch item or on the
request's `defaults` against a 3.6 checkpoint fails PER ITEM (message naming the
checkpoint and the 3.8-template provenance), not as a request-level error — batch
validation failures land on the item so the other N-1 keep their prefill, and a
defaults-level effort fails every item identically because the defaults reach the
renderer only through the items. Effort with thinking off (batch's default) stays
accepted and inert on 3.8, as everywhere else.

**Normalization passes assistant reasoning through in native tools mode; retention is
decided once, in the renderer, per dialect (2026-08-19, the arc's review pass).** Both
compat normalizers used to strip `reasoning` from every assistant turn before the
trailing assistant/tool run. That rule predates the dialect arc, when the renderer
dropped exactly those turns anyway (`preserve_thinking || index > last_query`, with
preserve always false), so the early strip was invisible. The 3.8 template made it a
bug: its `preserve_thinking` default is TRUE — the 3.8 card recommends preserved
thinking for agent workloads — so the dialect asked the renderer to keep reasoning the
normalizers had already destroyed, the OpenAI kwarg `preserve_thinking: true` was
defeated on every checkpoint, and the three dialects disagreed (native replayed
everything, the compat APIs didn't). Now native tools mode passes every turn's
reasoning through and the renderer's dialect rule is the single owner; the
`trailing_run_start` predicates are gone. The debug tools modes keep dropping
reasoning everywhere — they render the history as if tools had never existed, which is
their documented point. One nuance recorded so nobody "fixes" it back: Anthropic's
real Messages API strips non-trailing thinking blocks, and this dialect deliberately
does NOT emulate that — it is a wire-compatibility layer over Qwen checkpoints, and
what renders must follow the checkpoint's template, not the API vendor's serving
policy. A 3.6 request still renders without superseded reasoning, but because the 3.6
template says so, not because the API layer pre-judged it.

**Prefix reuse is quantized to snapshots, so a conversation resumes almost free and an
edited prompt does not (2026-08-30, from the first serve benchmark).** `PrefixCache::plan`
(src/serve/engine.rs:3727) takes the longest common prefix between the incoming prompt and
what the slot holds. At the cache's own end it extends; anywhere short of it, it must
rewind, and `rewind_to` (engine.rs:3701) quantizes DOWN to the nearest snapshot — the
anchor over the leading system block, a turn boundary, a fork point, or the tail a
page-out took (`plan_snapshot_stops`, engine.rs:2009) — returning `Cold` below the
shallowest one. The constraint is the state, not the bookkeeping: DeltaNet and PLE state
is recurrent, restorable only where it was captured, and no snapshot is ever taken inside
a message. The two consequences are both real and both measured on Flash-Next (log.md
2026-08-30 "serve on Flash-Next"): growing a conversation by one exchange resumes off the
turn boundary — 32k tokens took 67.5 s cold and 489 ms for the next turn, 7.6k took
10.5 s and 348 ms — while rewriting the last user message of a single-message prompt
lands under every snapshot and reports `cached_tokens: 0` with a full cold prefill.
Clients that edit prompts in place should expect that, and the fix, if it is ever worth
its cost, is mid-message snapshots (TODO.md), not a smarter matcher.
