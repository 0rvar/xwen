# xwen

Pure-Rust, Metal-only inference engine for **Qwen3.6-27B**, **Qwen3.6-35B-A3B** and
**Qwen3.8-27B** (GGUF), with **Qwen3.8-Flash-Next** in progress, optimized for a single
Apple Silicon machine (M5 Max). Manual fork of the
laguna/maxuna engine: candle-based, mmap no-copy weight loading, vendored Metal
kernels, speculative decoding, and an HTTP server speaking Anthropic Messages and
OpenAI Chat Completions.

**Status: runs, and matches upstream.** The 35B-A3B generates correctly end to end
(greedy, chat template, thinking split, clean stops) at ~59 tok/s decode, and both
checkpoints now pass the parity gate against upstream llama.cpp on identical GGUF
weights (`docs/parity.md`). No DeltaNet Metal kernels yet. See `TODO.md` for the
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

Three shipped checkpoints, all Q4_K_M, plus one in progress — all resolved through the
HF cache and downloaded on first use:

| Full name | Repo | `--model-size` | Drafter |
| --- | --- | --- | --- |
| `Qwen3.6-27B` | `ggml-org/Qwen3.6-27B-GGUF` | `27b` | DFlash block drafter, 3.5 GB |
| `Qwen3.6-35B-A3B` | `ggml-org/Qwen3.6-35B-A3B-GGUF` | `35b` (default) | DFlash block drafter, 0.8 GB |
| `Qwen3.8-27B` | `ggml-org/Qwen3.8-27B-GGUF` | `3.8-27b` | MTP head, 3.2 GB |
| `Qwen3.8-Flash-Next` **(experimental)** | `unsloth/Qwen3.8-Flash-Next-GGUF`, UD-Q4_K_XL, 4 shards | `flash-next` / `3.8-flash-next` | none |

**Qwen3.8-Flash-Next is EXPERIMENTAL and CLI-ONLY (P2, 2026-08-29)** — unlike
Qwen3.8-27B this one is a whole second architecture rather than a registry entry over an
existing graph: sparse attention, hyper-connections and a 51B n-gram embedding table on
top of the familiar gated DeltaNet and MoE. It loads, generates and stops correctly, and
its graph agrees with upstream llama.cpp (189/192 forced-replay steps, zero hard
mismatches — `docs/qwen4exp-parity-2026-08-29.md`). It is not finished: **`xwen serve`
REFUSES this checkpoint until P4**, because snapshots and the prefix cache cannot carry
its recurrent state; there is no drafter; and it has no parity harness or perplexity
floor of its own. Decode is 37.5-38.1 tok/s and prefill is 3.5x behind llama.cpp, both
first measurements with the usual caveats. See `docs/qwen4exp-port.md`.

**Two vocabularies, deliberately.** The CLI takes the short aliases above (and the full
names); the HTTP APIs take the **full names only**. Qwen3.8-27B (added 2026-08-14) runs
the same graph as Qwen3.6-27B — its config is byte-identical — so it needs no model math
of its own. It ships no DFlash sidecar, but it does ship a first-party MTP head, which is
a different drafter shape and became a second drafter implementation (2026-08-15); all
three checkpoints now speculate.

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

Every checkpoint ships a drafter and speculates, in one of **two kinds**. The Qwen 3.6
pair carry DFlash sidecars: block drafters that propose a whole block out of one forward,
so depth is nearly free (`--draft-max` defaults to 15, a structural ceiling rather than a
fitted number). Qwen3.8-27B carries a first-party MTP head, which is a different shape —
one extra transformer layer that chains a forward per step and feeds itself, so depth
costs linearly and is fitted rather than capped (4).

Drafting is **opt-out** as of 2026-07-29: a zero-flag run speculates with the
checkpoint's official sidecar, `--no-draft` decodes plain, and `--draft <gguf>` swaps in
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
prefill of it rather than nine. The checkpoint comes from the payload, not a flag.

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
`Qwen3.6-35B-A3B`, `Qwen3.8-27B`) — 2026-08-14. The CLI's short aliases are a CLI
spelling and are refused by every API. Selection by surface:

- `GET /v1/models` lists each checkpoint once, the served one first, under exactly the
  string a `model` field selects it by — every listed id is selectable, which is the
  point of a listing. A served GGUF that is none of the official checkpoints (a custom
  `--model` path) leads the list under its file name, which is then its only id.
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

Request bodies are capped at 100 MB (real cost is judged in tokens by the queue and
`context_length`, not in bytes). `context_length` — default: the checkpoint's trained
256k window — is a ceiling, not an allocation: the KV cache starts at 8k positions and
grows on demand as a conversation lengthens, and an idle unload drops whatever it grew
to, so the next load starts small again. The one-shot CLI commands' `--max-ctx` works
the same way and defaults to 131072.

The default bind is loopback because with no `api_key` the server accepts every request;
set `host = "0.0.0.0"` (or `--host`) together with an `api_key` to serve the LAN.

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
