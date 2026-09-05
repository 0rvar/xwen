# xwen

Pure-Rust, Metal-only inference engine for **Qwen3.8-Flash-Next** (the default),
**Qwen3.6-27B**, **Qwen3.6-35B-A3B** and **Qwen3.8-27B** (GGUF), optimized for a single
Apple Silicon machine (M5 Max). Manual fork of the
laguna/maxuna engine: candle-based, mmap no-copy weight loading, vendored Metal
kernels, speculative decoding, and an HTTP server speaking Anthropic Messages and
OpenAI Chat Completions.

**Status: runs, and matches upstream.** All four checkpoints generate correctly end to
end (greedy, chat template, thinking split, clean stops), and the 3.6 pair passes the
parity gate against upstream llama.cpp on identical GGUF weights (`docs/parity.md`) —
Qwen3.8-27B runs that same dense graph. Flash-Next has no harness of its own yet and was
verified by forced replay against llama.cpp (`docs/qwen4exp-port.md`). Decode rates are
in Speculative decoding below and in CLAUDE.md's Perf state. See `TODO.md` for the
priority ledger and `docs/log.md` for the narrative.

## Docs

- `docs/decisions.md` — every deliberate choice and refuted direction, with evidence
- `docs/log.md` — dated engineering narrative
- `docs/parity.md` — verification runbook (vs upstream llama.cpp)
- `CLAUDE.md` — agent context: ground truth, architecture cheat sheet, hazards
- `TODO.md` — deferred-work ledger

## Build

System Rust toolchain, system Apple CLT SDK (the nix shell deliberately provides no
SDK — see flake.nix). `cargo build --release`. Ops tests need a Metal device.

## Models

Four checkpoints — Q4_K_M, except Flash-Next's UD-Q4_K_XL — all resolved through the HF
cache and downloaded on first use:

| Full name | Repo | `--model-size` | Drafter |
| --- | --- | --- | --- |
| `Qwen3.8-Flash-Next` **(experimental)** | `unsloth/Qwen3.8-Flash-Next-GGUF`, UD-Q4_K_XL, 4 shards | `flash-next` / `3.8-flash-next` (default) | none |
| `Qwen3.6-27B` | `ggml-org/Qwen3.6-27B-GGUF` | `27b` | DFlash block drafter, 3.5 GB |
| `Qwen3.6-35B-A3B` | `ggml-org/Qwen3.6-35B-A3B-GGUF` | `35b` | DFlash block drafter, 0.8 GB |
| `Qwen3.8-27B` | `ggml-org/Qwen3.8-27B-GGUF` | `3.8-27b` | MTP head, 3.2 GB |

**Flash-Next is the default (2026-08-30), so a zero-flag first run downloads 111 GB**
across four shards — the notice naming the size prints before the fetch starts, and it
resumes in place. `xwen fetch` prefetches it; `bun scripts/hf-fetch.ts
unsloth/Qwen3.8-Flash-Next-GGUF <shard>... --jobs 2` does the same with parallel,
verified, resumable downloads. Pass `--model-size 35b` (or `27b`, `3.8-27b`) for a
~20 GB checkpoint instead.

**Every surface defaults to Flash-Next as of 2026-08-30**, `xwen serve` and `xwen batch`
included. Both move a whole cache state around on their ordinary path — the server
snapshots, rewinds and pages conversations out, and a batch prefills the items' shared
prefix once and replays that snapshot per item — and as of P4 a cache image carries
everything this checkpoint needs, so neither refuses it and neither falls back. What did
not change is the fetch rule: the server never downloads 111 GB inside a request, so
Flash-Next is listed by `/v1/models` and selectable by name only while its shards are
already in the HF cache; an uncached one is a 400 pointing at `xwen fetch`.

**Qwen3.8-Flash-Next is EXPERIMENTAL (P3, 2026-08-29; servable since P4, 2026-08-30)** —
unlike Qwen3.8-27B this one is a whole second architecture rather than a registry entry
over an existing graph: sparse attention, hyper-connections and a 51B n-gram embedding table on
top of the familiar gated DeltaNet and MoE. It loads, generates and stops correctly, and
its graph agrees with upstream llama.cpp (185/192 forced-replay steps at 0261e17,
186/192 after the P3 kernel pass, 189/192 before it, zero hard mismatches at any of them
— every divergence a rank-2-or-3 near-tie, margins down to 0.0002 logit;
`docs/qwen4exp-parity-2026-08-29.md`). It now runs on every surface:
snapshots, rewind, page-out and the on-disk tier carry its QSA indexer rows and its PLE
conv window and n-gram history, so `serve` and `batch` treat it like any other
checkpoint. It is still not finished: there is no drafter for its graph (so it decodes
plain, and `--draft` is refused rather than ignored), it is not auto-fetched, and it has
no parity harness or perplexity floor of its own. It is, however, fast: after the P3 kernel
pass it runs **prefill 795.7 tok/s, and decode 46.5-46.7 tok/s since the beta|alpha fold
of 2026-08-30 (44.4-45.8 before it, 43.1 before the PLE row prefetch)** where llama.cpp on
the same file in the same hour as the prefill arm runs 789 and 41.4 — plain,
no drafter, 530-token prompt, interleaved rounds,
medians, `powermode 0` with no high-power claim. Take those from unprofiled runs: the
per-step profilers (`XWEN_STACK_PROFILE`, `XWEN_GDN_PROFILE`) sync-bracket each step, so
they rank steps and do not price them. See `docs/qwen4exp-port.md`.

**Two vocabularies, deliberately.** The CLI takes the short aliases above (and the full
names); the HTTP APIs take the **full names only**. Qwen3.8-27B (added 2026-08-14) runs
the same graph as Qwen3.6-27B — its config is byte-identical — so it needs no model math
of its own. It ships no DFlash sidecar, but it does ship a first-party MTP head, which is
a different drafter shape and became a second drafter implementation (2026-08-15); all
three qwen35 checkpoints speculate. Flash-Next does not (below).

## Thinking, effort and sampling

Thinking is on by default everywhere except batch: the prompt ends inside an open
`<think>` block and the reply's reasoning is split from its answer. `xwen generate` and
`xwen chat` take `--no-think` (the prompt closes an empty think block; the reply is all
answer) and `--reasoning-effort <low|medium|xhigh>` — a Qwen 3.8 chat-template parameter
that renders a system-preamble instruction (`xhigh` is the template's own default;
`medium` renders nothing). On a 3.6 checkpoint `--reasoning-effort` is a startup error:
that template has no such parameter, and inert flags are refused rather than ignored.
Both flags are also rejected with `--raw`, which never renders a template, and
`--no-think` rejects a nonzero `--min-think`/`--max-think` — both budgets govern the
`<think>` block a no-think prompt closes itself.

Each checkpoint renders under its own template dialect (`Model::chat_dialect`,
2026-08-19): besides the effort preamble, the 3.8 template defaults `preserve_thinking`
to true where 3.6 defaults it false, and no longer parses inline `<think>` blocks out of
assistant content. Details and evidence in decisions.md "Tokenization, chat, tool
calls".

**Sampling defaults are keyed to thinking mode**, per the official model cards and
identical across all three checkpoints: thinking temp 1.0 / top_p 0.95 / top_k 20,
non-thinking 0.7 / 0.80 / 20. Explicit flags, config keys and request fields always win;
a server-configured sampling value pins one number for both modes, unset lets each
request use its mode's own. The cards also recommend `presence_penalty` 1.5 for
non-thinking mode — not implemented: the sampler has no penalty machinery and penalties
entangle the speculative verify path (TODO.md).

On the serve side, requests pick thinking per dialect: Anthropic `thinking`, native
`thinking`, OpenAI `reasoning_effort` — which drives both the think-token budget
(none/minimal/low/medium/high/xhigh/max) and, on 3.8, the template preamble
(nearest-mapping the levels the template lacks). The OpenAI dialect also accepts
`chat_template_kwargs` with `enable_thinking`, `preserve_thinking` and
`reasoning_effort` (the official Qwen card's shape; strictly validated — an unknown key,
wrong type, or off-scale level is a 400, unlike the sampling params this dialect accepts
and drops). The native dialect takes `reasoning_effort` and `preserve_thinking`
directly. A request-level template effort — the kwarg or the native field — on a 3.6
target is a 400 naming the model, the same rule as the CLI flag; the top-level OpenAI
field stays accepted there (it carries budget semantics on every checkpoint).
`[thinking] effort` in a serve config, or `serve --reasoning-effort`, sets a
server-wide template-effort default (inert-but-legal on the 3.6 checkpoints); the
Anthropic dialect has no per-request effort field, so that default is what its
requests get. Replayed reasoning is passed through to the renderer on every assistant
turn; each checkpoint's `preserve_thinking` rule decides what renders (3.6 drops
superseded reasoning, 3.8 keeps it).

## Speculative decoding

The three qwen35 checkpoints ship a drafter and speculate, in one of **two kinds**;
Flash-Next ships none and decodes plain, saying so in a startup line. The Qwen 3.6
pair carry DFlash sidecars: block drafters that propose a whole block out of one forward,
so depth is nearly free (`--draft-max` defaults to 15, a structural ceiling rather than a
fitted number). Qwen3.8-27B carries a first-party MTP head, which is a different shape —
one extra transformer layer that chains a forward per step and feeds itself, so depth
costs linearly and is fitted rather than capped (4).

Drafting is **opt-out** as of 2026-07-29: a zero-flag run speculates with the
checkpoint's official sidecar where one exists (which the default checkpoint's does
not), `--no-draft` decodes plain, and `--draft <gguf>` swaps in
a custom drafter. Speculation is decided **per checkpoint, not per process**: a server
drafts for each checkpoint it loads with that checkpoint's own sidecar, and the
dashboard's `draft` cell reports what the LOADED checkpoint is doing rather than what was
configured. The default costs a sidecar load per run (3.5 GB on the 27B, 0.8 GB on the
35B-A3B, 3.2 GB on the 3.8-27B).

Measured over plain decode on the same machine state (greedy, 128 tokens, warm, medians
of 3 reps, arms interleaved, `lowpowermode 0` on AC):

| Checkpoint | code | chat | acceptance | fitted |
| --- | --- | --- | --- | --- |
| `Qwen3.6-27B` | +46 to +52% | +46 to +52% | 78-86% | 2026-08-08 |
| `Qwen3.6-35B-A3B` | +26 to +28% | +15 to +17% | 68-74% | 2026-08-08 |
| `Qwen3.8-27B` | +44 to +45% | +37 to +38% | 78-80% | 2026-08-15 |

The 35B-A3B's PLAIN decode level moved on 2026-08-30 (105.1 → 114.4 tok/s, the
beta|alpha fold at 0261e17); the gains in this table were fitted against the older level
and have not been re-swept, so read them as gains over their own sweep's plain arm.

Acceptance trades against draft length, so raising `--draft-p-min` buys acceptance and
loses tok/s. Ranges span the medians each shipped configuration was measured at; compare
a drafted number only against the plain arm of its OWN sweep, never against another
session's.

**Both defaults are per-checkpoint.** `--draft-p-min` is 0.5 on the 27B, 0.3 on the
35B-A3B and 0.7 on the 3.8-27B (`Model::draft_p_min_default`); `--draft-max` is 15 on the
two block drafters and 4 on the MTP head (`Model::draft_max_default`, keyed by drafter
kind). The 27B's target forward is expensive, so it wants short confident drafts; the
35B-A3B's is cheap enough to profit from drafting deeper at lower acceptance. On the
3.8-27B depth is the knob that matters and the floor barely does — across a 3x3 sweep the
floor moved throughput by at most 1.8% at fixed depth where depth moved it 12%. Passing
either flag, or `draft.p_min` / `draft.max` in a serve config, overrides it — and since
one server loads whichever checkpoint a request names, an explicit value pins one setting
for every checkpoint it serves; leave them unset to give each its own.
`--draft-pause-margin` stays a single shared 1.0. `bun scripts/retune-draft.ts` re-fits
these and prints recommendations; it never edits a default. See docs/decisions.md
"Speculative decoding".

`--draft` should reproduce `--no-draft`; `bun scripts/spec-equivalence.ts` checks that on
all three models in two modes — greedy, and sampled at a fixed seed. **Greedy is the
gate.** Sampled mode is a diagnostic and currently fails on healthy builds: the batched
verify forward reassociates its f32 sums differently from the single-token forward, so at
temperature a near tie can resolve to a different token, and the shipped 27B diverges on
the chat fixture at every seed tried. Read a sampled divergence only against a control
run of another checkpoint, and note the script's "a first-line fork means the sampler
stream" rule is a heuristic that mis-grades (TODO.md). What actually separates a near tie
from a sampler-stream bug is seed-dependence: a stream bug diverges at every seed.

## Batch

`xwen batch` answers N chat items that share a prompt prefix: one JSON request on
stdin, one JSON response on stdout, progress lines on stderr. The shared prefix is
prefilled once and the KV cache snapshotted there; every item restores that snapshot
and prefills only its own tail, so nine questions about the same document cost one
prefill of it rather than nine. The checkpoint comes from the payload, not a flag — or
from `-m <gguf>`'s own identity when one is given, with the payload's name as the
cross-check. A payload naming nothing gets the default checkpoint, Flash-Next included:
those snapshots carry the qwen4exp recurrent state as of 2026-08-30, so this surface no
longer has a checkpoint it refuses.

```bash
xwen batch < request.json > response.json
```

```json
{
  "model": "Qwen3.6-35B-A3B",
  "defaults": { "max_tokens": 32 },
  "items": [
    {
      "id": "sentiment",
      "messages": [
        { "role": "system", "content": "You classify support email." },
        { "role": "user", "content": "<the email>\n\nOverall sentiment?" }
      ],
      "schema": {
        "type": "object",
        "properties": {
          "label": { "enum": ["positive", "mixed", "negative"], "include_score": true }
        },
        "required": ["label"],
        "additionalProperties": false
      }
    }
  ]
}
```

```json
{
  "model": "Qwen3.6-35B-A3B",
  "items": [
    {
      "id": "sentiment",
      "content": "{\"label\":\"mixed\"}",
      "text": "{\"label\":\"mixed\"}",
      "json": { "label": { "value": "mixed", "score": 0.5601 } },
      "finish_reason": "stop",
      "usage": { "prompt_tokens": 349, "cached_prefix_tokens": 317, "completion_tokens": 9 }
    }
  ],
  "stats": {
    "shared_prefix_tokens": 317,
    "snapshot_ms": 148.3,
    "items": 1,
    "prefill_tokens": 391,
    "prefill_ms": 361.0,
    "decode_tokens": 9,
    "decode_ms": 92.0,
    "load_ms": 3470.0,
    "total_ms": 1828.0
  }
}
```

An item takes `messages`, an optional JSON `schema` (constrained decode), `thinking`
(`false` / `true` / a string to inject), `reasoning_effort` (`"low"` / `"medium"` /
`"xhigh"` — the 3.8 template's levels, defaulting to the template's own `xhigh`;
supplying one on a 3.6 checkpoint fails the item, since that template has no such
parameter), an assistant `prefill`, `max_tokens` and `sampling`; `defaults` sets any of
them batch-wide. Batch sampling is greedy and thinking is off unless a request says
otherwise — both differ from the chat surface on purpose (docs/decisions.md "Batch").
Per-item failures land as `error` on that item and the rest of the batch still runs.

**`shared_prefix`.** A request-level string prepended verbatim to every item's first
message content, so a large shared document is spelled once on the wire instead of once
per item. Purely a body-size measure: the prompts (and answers, and scores) are
byte-identical to writing the document into every item, and the one-prefill snapshot
above works the same either way.

**`include_score`.** Annotate an enum or boolean property with `include_score` and the
item leaves the grammar path: xwen writes the JSON skeleton itself and picks each
field's value by scoring every allowed option against the model, reporting
`{"value": …, "score": …}` instead of a bare value (`"all"` adds the full `scores`
table and the `escape`). `escape` is the probability the model would rather have
written something NO option opens, measured over the whole vocabulary at the field's
choice point with formatting factored out — whitespace-led spellings of an allowed
value count as that value, pure-whitespace layout tokens count as neither side
(2026-08-11; before that it was raw opener mass and read ≈1 on a document's first
bare-literal field). v1 accepts a flat all-required object of enum/boolean fields and
refuses anything else by name.

`XWEN_BATCH_NO_CACHE=1` runs every item from a reset cache — the A/B lever for what the
snapshot saves (35B demo: 1203 ms cached vs 1989 ms cold). The two arms decode the same
answers but not always the same bytes; see docs/decisions.md "Batch".

`bun scripts/classify-demo.ts` is the worked example: one support email classified along
nine taxonomies as nine batch items, with ground truth embedded, run on both checkpoints
through `POST /xwen/v1/batch` — against a server already running on the default port (or
`--url`), else one the script spawns and tears down itself. Running both checkpoints
through one server is a deliberate exercise of the checkpoint swap.

## Serve

`xwen serve` runs an HTTP server over the engine. Routes: `POST /v1/messages` (+
`/v1/messages/count_tokens`) in the Anthropic dialect, `POST /v1/chat/completions` in
the OpenAI dialect, `POST /xwen/v1/generate` and `POST /xwen/v1/batch` (the native
surface), `GET /v1/models`, `GET /health`. `xwen serve --init` writes a commented
config template; every setting is also a flag.

**One server serves every checkpoint (2026-08-11).** `--model`/`--model-size` picks the
DEFAULT checkpoint; any request may name another one and the engine lazy-loads it,
imaging the live conversation out first (the same path an idle unload takes) — one
model resident at a time, always, and `idle_unload` applies to whichever is loaded.

**On the wire a checkpoint has exactly one name: its full name** (`Qwen3.6-27B`,
`Qwen3.6-35B-A3B`, `Qwen3.8-27B`, `Qwen3.8-Flash-Next`) — 2026-08-14. The CLI's short
aliases are a CLI spelling and are refused by every API. Selection by surface:

- `GET /v1/models` lists each checkpoint once, the served one first, under exactly the
  string a `model` field selects it by — every listed id is selectable, which is the
  point of a listing. Flash-Next is listed only while its shards are in the HF cache,
  for the same reason: it is the one checkpoint a request may not download, so listing
  it uncached would list an id that is a 400. A served GGUF that is none of the
  official checkpoints (a custom `--model` path) leads the list under its file name,
  which is then its only id.
- The compat dialects, `/v1/messages/count_tokens` and `/xwen/v1/batch` all resolve
  `model` the same way: absent or empty means the served file, this server's own id
  means the served file, another checkpoint's full name selects that checkpoint out of
  its own hub file, and anything else is a 400 in that dialect's error format naming the
  valid ones. The response echoes the canonical name of what answered, not the client's
  string.
- `/xwen/v1/generate` carries no model field and always runs on the served file.

An SDK's own model id (`gpt-4o`, `claude-…`) is therefore a 400 rather than a silent
answer from whichever checkpoint happened to be the default; point the client at a full
name, or omit the field.

**A custom GGUF answers under its own id and no other.** If the served file is none of
the official checkpoints, it is served as its architecture's checkpoint (a startup line
says which) but reported and selected by its file name — and a request naming an
official checkpoint gets that checkpoint's real hub file, downloading it if need be,
rather than these weights under an official name. `--model-size` names the checkpoint a
file that says nothing about itself holds; a flag that contradicts a file that DOES say
is a startup error rather than a server that 500s every request.

Swapping costs a full model load (~3 s warm) plus losing the outgoing checkpoint's warm
KV slots, so interleaving checkpoints request-by-request is legal but slow. The on-disk
prefix cache stays bound to the default checkpoint; the other checkpoint runs without
it. A requested checkpoint (or its drafter) missing from the HF cache is downloaded
inside the request — a one-line notice in the server log; hf-hub's byte-level progress
bar goes to raw stderr, so under `--tui` it draws over the dashboard (TODO.md).
Flash-Next is the exception: 111 GB is not something a stranger's request gets to start,
so an uncached one is a 400 naming `xwen fetch` instead.

Request bodies are capped at 100 MB (real cost is judged in tokens by the queue and
`context_length`, not in bytes). `context_length` — default: the checkpoint's trained
256k window — is a ceiling, not an allocation: the KV cache starts at 8k positions and
grows on demand as a conversation lengthens, and an idle unload drops whatever it grew
to, so the next load starts small again. The one-shot CLI commands' `--max-ctx` works
the same way and defaults to 131072.

The default bind is loopback because with no `api_key` the server accepts every request;
set `host = "0.0.0.0"` (or `--host`) together with an `api_key` to serve the LAN.

## Metrics

Every run appends one JSON line to `$HOME/.local/state/xwen/metrics.jsonl`: what it was,
what it cost, and who asked for it. `generate`, `chat`, `batch` and every served request
record themselves, on by default. `XWEN_METRICS_FILE=<path>` records somewhere else and
`XWEN_METRICS_FILE=off`, in any casing, records nothing; setting the variable to an
empty string is not setting it and the default path applies. A write that fails prints
one warning for the life of the process and never fails the run that produced it, and a
whole record goes out in a single append, so a server and a `generate` recording at the
same moment interleave records rather than fragments.

A record carries the schema version, the completion timestamp, the surface (`generate`,
`chat`, `batch`, `serve:anthropic`, `serve:openai`, `serve:native`, `serve:batch`), the
checkpoint name, prompt / cached / prefill / decode token counts with the seconds each
phase took, `ok`, and whatever else the surface knows: thinking tokens, drafted and
accepted positions, batch item count, the client and session ids. Readers ignore fields
they do not recognize, so an older xwen still reads a newer one's history.

An absent optional means not measured, which is not the same as zero. `thinking_tokens`
is the one to know: serve always measures it and writes it, 0 included, while `generate`
and `chat` count thinking only when a think budget is in effect, so a thinking run with
no budget set records no thinking count at all. Summing the field therefore sums the
runs that measured it, not the runs that thought.

The checkpoint name is the official full name when the GGUF identifies as one and the
GGUF's own file stem when it identifies as nothing. Every surface spells it the same
way, `generate`, `chat` and `batch` included, so `--by model` never splits one file
across two names. A batch response's own `model` field answers a different question,
naming the checkpoint the run replies as, and is unchanged.

`xwen stats` reports over the file. `--by day|week|month|model|surface|client|session|all`
(default `day`), `--since 24h|7d|4w|YYYY-MM-DD` (a date means local midnight), `--model`
and `--surface` filter exactly, `--client` and `--session` by substring, `--json` prints
the rows instead of the table, and `--file` reads another history without ever recording
to it. A bad `--since` is an error before the file is opened, so a typo says so rather
than reporting an empty history. The label column is measured in display columns, so a
CJK or emoji label still lines up, and is cut to 48 with a trailing ellipsis; a raw
client id is the one thing long enough to need it.

```bash
xwen stats                                    # today and the days before it
xwen stats --by session --since 7d            # where the week went
xwen stats --by model --surface serve:openai --json
```

```
surface          runs  prompt  cached  hit%  prefill  pf tok/s  decode  dec tok/s  accept%
serve:batch         1  48,000  31,000  64.6   17,000      2636   1,200      120.0        -
chat                1   3,120       0   0.0    3,120       612     840       37.5        -
serve:anthropic     2   9,630   7,800  81.0    1,830       806     836       46.5        -
generate            1     925       0   0.0      925       701     512       37.6     77.1
serve:openai        1   7,606   6,144  80.8    1,462       860     301       46.3        -
---------------  ----  ------  ------  ----  -------  --------  ------  ---------  -------
total               6  69,281  44,944  64.9   24,337      1445   3,689       52.4     77.1
```

A rate is the bucket's tokens over the bucket's seconds, never a mean of per-run rates:
a hundred two-token replies would otherwise outweigh one long generation. `-` marks a
column nothing in the bucket measured. The table is the whole of stdout; the file, the
run count, any line that did not parse, and a note if the local offset could not be read
and the dates fell back to UTC all go to stderr. The history is read as bytes and
decoded one line at a time, so a torn or non-UTF-8 line is counted as unreadable and
costs only itself. `--since` on a date refuses a day its month does not have, leap years
included. A history that does not exist yet prints `no metrics recorded yet (<path>)`
and exits 0.

**`cached` means something slightly different on each surface.** Serve is the only one
with prefix reuse, so it is the only one whose cached count comes from a real cache
read. `generate` prefills from a reset cache and `chat` re-prefills the whole
conversation every turn, so both record zero by construction. A batch run is ONE record
covering all its items, and its counts are measured independently rather than derived
from each other. Cached is the sum of the items' own `cached_prefix_tokens` less the
shared prefix, which is prefilled once and read back for every item after the first, so
summing that column alone counts the prefix one time too many. Prefill is what the
engine really forwarded, shared prefix included: the runner opens its own prefill
accounting once the prefix is already resident, which is why the recorded seconds are
its prefill plus its snapshot time. The `/xwen/v1/batch` route folds both halves into
its summary the same way.

So `prompt = cached + prefill` holds for an ordinary run and is not an invariant. A
scored batch forwards more than its prompt, every teacher-forced trial being real work
against no prompt token, and a served job that was abandoned or failed forwards less.

**`ok` means the run reached its own end: an end-of-generation token or its token cap,
with no error.** Anything else is `ok` false, and the record is still written, because a
history that dropped its bad runs would show a quiet hour where there was a struggling
one. So a served job whose client disconnected or whose deadline killed it is `ok`
false, as is a chat turn cancelled with Ctrl-C mid-generation, alongside the outright
errors. What those records carry is whatever was really measured, which for an
interrupted run is the tokens it reached before it stopped. Zero counts mean zero
measurements, not a convention: a `generate` whose generation errored, say because the
prompt outgrew `--max-ctx`, a chat turn that died during prefill, a `batch` whose whole
payload failed, a served batch that never reached the runner.

A served batch that fails before producing a summary is still filed under `serve:batch`,
the surface coming from the job's own kind at pickup rather than from a summary it never
wrote. A batch where every item failed is a failed run; one that lost only some of its
items is not. A CLI `batch` failure records the model as `-`, most whole-request
failures happening before the payload has named a checkpoint. Every surface records its
own failures; the one thing that leaves nothing behind is a killed process. `generate`
installs no signal handler, so Ctrl-C on it writes no record, where a `chat` turn
cancelled inside the REPL does write one.

`ok` separates populations; it does not filter. Every record a query matches is summed
into its bucket, unfinished ones included, and the count of them comes back as
`unfinished` in the `--json` rows. The table has no column for it.

**`session` comes from a header, `client` from the body verbatim.** `session` is the
`x-claude-code-session-id` request header, Anthropic's documented per-session identifier
and one Claude Code sends on every request; the Anthropic and OpenAI routes read it, and
so does `/xwen/v1/batch`, whose payload carries no client id of its own. `client` is the
body's own id (Anthropic `metadata.user_id`, OpenAI `user`), stored unparsed because its
format is undocumented and has changed between Claude Code releases: one capture spells
it `user_…_session_…`, another embeds a JSON `session_id`. Both fields stay
accept-and-drop, held as raw JSON like `tool_choice`, so a wrongly-typed `metadata` or
`user` costs at most the client id and is never a 400. `--by session` takes the header
when there is one, otherwise reads past the last `session_` marker in the client string,
and labels whatever is left `-`. Both ids are cut to 128 characters. Whether the
header's id is the same one `claude --resume` shows is not confirmed.

## Verifying a change

Any change to model math re-runs the parity gate. It compares our forward pass against
upstream llama.cpp on the identical GGUF, so it needs the oracle built once:

```bash
just init                                     # fetch the llama.cpp submodule (pinned)
bash scripts/build-llamacpp.sh
bun scripts/parity-gate.ts                    # 35B-A3B, all tiers
bun scripts/parity-gate.ts --model-size 27b   # 27b dense
```

`docs/parity.md` is the runbook: tiers, floors, tap mapping, and the pinned oracle
commit.
