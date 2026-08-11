# xwen

Pure-Rust, Metal-only inference engine for **Qwen3.6-27B** and **Qwen3.6-35B-A3B**
(GGUF), optimized for a single Apple Silicon machine (M5 Max). Manual fork of the
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

Default checkpoints: `ggml-org/Qwen3.6-27B-GGUF` and `ggml-org/Qwen3.6-35B-A3B-GGUF`
(Q4_K_M), resolved through the HF cache, downloaded on first use.

## Speculative decoding

DFlash drafter sidecars ship alongside both checkpoints and are adapted. It is
**opt-out** as of 2026-07-29: a zero-flag run speculates with the checkpoint's official
sidecar, `--no-draft` decodes plain, and `--draft <gguf>` swaps in a custom drafter.
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
  "model": "35b",
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
  "model": "35b",
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

**`include_score`.** Annotate an enum or boolean property with `include_score` and the
item leaves the grammar path: xwen writes the JSON skeleton itself and picks each
field's value by scoring every allowed option against the model, reporting
`{"value": …, "score": …}` instead of a bare value (`"all"` adds the full `scores`
table and the `escape` mass that fell outside the option set). v1 accepts a flat
all-required object of enum/boolean fields and refuses anything else by name.

`XWEN_BATCH_NO_CACHE=1` runs every item from a reset cache — the A/B lever for what the
snapshot saves (35B demo: 1203 ms cached vs 1989 ms cold). The two arms decode the same
answers but not always the same bytes; see docs/decisions.md "Batch".

`bun scripts/classify-demo.ts` is the worked example: one support email classified along
nine taxonomies as nine batch items, with ground truth embedded, run on both checkpoints.

## Serve

`xwen serve` runs an HTTP server over the engine. Routes: `POST /v1/messages` (+
`/v1/messages/count_tokens`) in the Anthropic dialect, `POST /v1/chat/completions` in
the OpenAI dialect, `POST /xwen/v1/generate` and `POST /xwen/v1/batch` (the native
surface), `GET /v1/models`, `GET /health`. `xwen serve --init` writes a commented
config template; every setting is also a flag.

**One server serves both checkpoints (2026-08-11).** `--model`/`--model-size` picks the
DEFAULT checkpoint; any request may name the other one and the engine lazy-loads it,
imaging the live conversation out first (the same path an idle unload takes) — one
model resident at a time, always, and `idle_unload` applies to whichever is loaded.
Selection by surface:

- `/xwen/v1/batch` takes the same JSON document `xwen batch` reads on stdin and returns
  the same document it prints. Its `model` field ("27b"/"35b") is honored per request;
  absent means the server's default, unknown is a 400.
- The compat dialects honor a `model` that names a known checkpoint ("27b"/"35b") and
  fall back to the default for anything else — SDKs sending their own model ids keep
  working, and the response echoes whatever the client sent, as before.
- `/xwen/v1/generate` carries no model field and always runs on the default.

Swapping costs a full model load (~3 s warm) plus losing the outgoing checkpoint's warm
KV slots, so interleaving checkpoints request-by-request is legal but slow. The on-disk
prefix cache stays bound to the default checkpoint; the other checkpoint runs without
it. A requested checkpoint (or its drafter) missing from the HF cache is downloaded
inside the request — a one-line notice in the server log; hf-hub's byte-level progress
bar goes to raw stderr, so under `--tui` it draws over the dashboard (TODO.md).

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
