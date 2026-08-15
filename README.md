# xwen

Pure-Rust, Metal-only inference engine for **Qwen3.6-27B**, **Qwen3.6-35B-A3B** and
**Qwen3.8-27B** (GGUF), optimized for a single Apple Silicon machine (M5 Max). Manual fork of the
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

Three checkpoints, all Q4_K_M, resolved through the HF cache and downloaded on first
use:

| Full name | Repo | `--model-size` | Drafter |
| --- | --- | --- | --- |
| `Qwen3.6-27B` | `ggml-org/Qwen3.6-27B-GGUF` | `27b` | DFlash, 3.5 GB |
| `Qwen3.6-35B-A3B` | `ggml-org/Qwen3.6-35B-A3B-GGUF` | `35b` (default) | DFlash, 0.8 GB |
| `Qwen3.8-27B` | `ggml-org/Qwen3.8-27B-GGUF` | `3.8-27b` | none — decodes plain |

**Two vocabularies, deliberately.** The CLI takes the short aliases above (and the full
names); the HTTP APIs take the **full names only**. Qwen3.8-27B (added 2026-08-14) runs
the same graph as Qwen3.6-27B — its config is byte-identical — so it needs no model math
of its own; what it lacks is a DFlash sidecar, so it decodes plain wherever the other two
speculate, which costs it the whole speculative win (TODO.md tracks an MTP drafter).

## Speculative decoding

DFlash drafter sidecars ship alongside both Qwen 3.6 checkpoints and are adapted. It is
**opt-out** as of 2026-07-29: a zero-flag run speculates with the checkpoint's official
sidecar, `--no-draft` decodes plain, and `--draft <gguf>` swaps in a custom drafter.
Qwen3.8-27B ships no sidecar, so it decodes plain with one line saying so (23.8 tok/s
measured, one greedy run, 2026-08-14); `--draft <gguf>` still attaches a drafter you
supply, and an explicit `--draft official` is an error rather than a silent downgrade.
Speculation is decided **per checkpoint, not per process**: a server whose default
checkpoint ships no sidecar still drafts for every other checkpoint it loads, each with
its own sidecar, and the dashboard's `draft` cell reports what the LOADED checkpoint is
doing rather than what was configured.
At the defaults fitted 2026-08-08, drafting measured +46 to +52% on the 27B (both prompt
kinds) and +26 to +28% on code / +15 to +17% on chat on the 35B-A3B, over plain decode
on the same machine state (greedy, 128 tokens, warm, medians of 3 reps, two independent
runs). Acceptance at those defaults is 78-86% on the 27B and 68-74% on the 35B-A3B; it
trades against draft length, so raising `--draft-p-min` buys acceptance and loses tok/s.
The default costs a sidecar load per run (3.5 GB on the 27B, 0.8 GB on the 35B-A3B).

**`--draft-p-min` has a per-checkpoint default: 0.5 on the 27B, 0.3 on the 35B-A3B**
(`Model::draft_p_min_default`). The 27B's target forward is expensive, so it wants short
confident drafts; the 35B-A3B's is cheap enough to profit from drafting deeper at lower
acceptance, and 0.5 costs it ~2.5%. Passing the flag, or `draft.p_min` in a serve
config, overrides it — and since one server now loads whichever checkpoint a request
names, an explicit value pins one floor for every checkpoint it serves; leave it unset
to give each checkpoint its own. `--draft-pause-margin` stays a single 1.0 for both.
`bun scripts/retune-draft.ts` re-fits both knobs and prints recommendations; it never
edits a default. See docs/decisions.md "Speculative decoding".

`--draft` should reproduce `--no-draft`; `bun scripts/spec-equivalence.ts` checks that on
both models in two modes — greedy, and sampled at a fixed seed (the only one that can
catch the spec loop drawing from the RNG a different number of times than plain decoding).
It prints the fork point when they differ; a near-tie landing differently is expected, a
first-line fork in sampled mode is not. See the script's header.

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
(`false` / `true` / a string to inject), an assistant `prefill`, `max_tokens` and
`sampling`; `defaults` sets any of them batch-wide. Batch sampling is greedy and
thinking is off unless a request says otherwise — both differ from the chat surface on
purpose (docs/decisions.md "Batch"). Per-item failures land as `error` on that item and
the rest of the batch still runs.

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
