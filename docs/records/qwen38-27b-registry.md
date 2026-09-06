# 2026-08-14 — Qwen3.8-27B added as a registry entry (same graph, no drafter), and the APIs go to full model names only: `/v1/models` stops listing one checkpoint three ways and an unknown `model` is a 400 instead of a silent default

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


**Two changes, one root.** Qwen3.8-27B released today; its `config.json` is
byte-identical to Qwen3.6-27B's, so it is the same forward pass over different weights
and needed no model math (the parity gate was deliberately not run: nothing it grades
changed). Adding it broke two claims the code made. `Arch::model()`'s doc said the GGUF
architecture identifies the checkpoint one-to-one — true until two releases shipped the
same dense `qwen35` graph. And the model-name vocabulary, which had been the CLI's short
aliases, could no longer be stretched: `27b` already meant 3.6 and a third checkpoint
made the aliases a naming scheme rather than a shorthand.

**What the old `/v1/models` actually returned**, from the server running on this machine
while the work was underway: `Qwen3.6-35B-A3B-Q4_K_M` (the file stem), `27b`, `35b` —
three ids for two checkpoints, one of them listed twice under two spellings. A client
picking a `model` string from that listing is picking between two names for the same
model. Alongside it, the compat dialects resolved an unrecognized `model` by falling
back to the default, so an SDK's own id was answered indistinguishably from a correct
request. Both are now one rule: the APIs speak `Model::full_name()` and nothing else —
`Qwen3.6-27B`, `Qwen3.6-35B-A3B`, `Qwen3.8-27B`, the strings ggml-org names the repos
with and the GGUFs carry as `general.name` — absent or empty still means the served
checkpoint, and anything else is a 400 listing the valid names. `--model-size` keeps the
aliases and now also takes full names, so an id from a listing pastes into the CLI.

**The identification chain that replaced `arch.model()`**: explicit `--model-size`
(threaded into `serve::run` as an `Option<Model>` so an explicit flag is distinguishable
from the default), then `general.name`, then the file name, then `arch.model()` as the
last resort with a logged warning naming what it assumed. Verified against the real
files rather than assumed: the cached 3.6 GGUFs carry `general.name = "Qwen3.6-27B"` and
`"Qwen3.6-35B-A3B"`, exactly the full names, which is why the first pass is an exact
match with a substring pass behind it.

**The drafter became optional, which was most of the diff.** Qwen3.8-27B ships no DFlash
sidecar (the repo's MTP sidecar is unread — ledger item), so `Checkpoint.drafter` is now
an `Option<Drafter>` holding file, size, layer count and fitted `p_min` together: a
checkpoint either drafts or it does not, and no caller can ask half the question. Every
consumer handles `None` by running plain with one line saying so — `ensure_drafter`,
`resolve_draft`, `checkpoint_paths` (a new `ServeLog::NoDrafterAvailable`), `xwen fetch`
(prints `drafter none`) — and `resolved_p_min` falls back to the shared base for the one
path that can still attach a drafter to a sidecar-less checkpoint, a custom `--draft`.

**Facts checked rather than trusted.** The tokenizer is NOT byte-identical between the
releases: 3.8 adds seven audio/TTS specials at 248070-248076 over an identical base
vocab (248044 entries) and merge table (247587), verified structurally after the blob
hashes differed. Text tokenizes identically, so the embedded 3.6 tokenizer still ships
and nothing was wired; whether a text-only checkpoint can emit those ids is a ledger
item, not something to improvise a second 12.8 MB embed over. 3.8's chat template DOES
differ (reasoning_effort preamble, `preserve_thinking` defaulting true, no inline
`<think>` parsing) and is vendored as `reference/chat_template-qwen38.jinja`; its
generation prompt is byte-identical to 3.6's, which is why the hand-written renderer is
unchanged, and chat.rs now cross-checks both vendored templates so a future divergence
fails a test instead of a reply.

**Verification.** `cargo build --release` clean; `cargo test --release` 869 passed, 0
failed (the two `unused_mut` warnings predate this work). The parity gate was not run and
did not need to be: no model math changed.

Qwen3.8-27B end to end, from an empty cache: `xwen fetch --model-size 3.8-27b` downloaded
18,973,870,432 bytes (the published size) and printed `drafter none (Qwen3.8-27B ships no
sidecar)`; a second run resolved from cache without a request. `xwen inspect` confirms
what the registry entry claims — `general.name = "Qwen3.8-27B"` (the exact full name, so
identification is an exact match rather than the substring fallback), `qwen35`, file_type
15, rope sections [11,11,10,0], ssm 48/16/128/6144, conv_kernel 4, eos 248046 with
`add_bos_token = false`, and all 16 `attn_output.weight` at Q6_K, which the loader parses
and the Metal load accepts without a murmur. `xwen generate --model-size 3.8-27b`: 18.3 GB
resident, 9.3 s cold load, the line `no drafter available for Qwen3.8-27B; decoding
without speculation`, coherent answer, thinking block closed, stopped on an EOG id well
inside the token budget. Plain decode measured **23.8 tok/s** (21-token prompt, 145
tokens, single greedy run, cold-ish, machine shared with other work — one run, not a
sweep; the 27B's plain figure is 24.8-25.3 under the 2026-08-08 protocol, which is the
number to compare against, and both are plain).

Serve, on a 35B-A3B default: `/v1/models` returns exactly
`["Qwen3.6-35B-A3B", "Qwen3.6-27B", "Qwen3.8-27B"]` — the served one first, each once,
against the three-ids-for-two-checkpoints listing the old binary was still returning on
this machine. `"model": "35b"` → 400 in the OpenAI envelope, `"27b"` → 400 in the
Anthropic envelope, `"gpt-4o"` → 400, `/xwen/v1/batch` `"35b"` → 400, every message
listing the three valid names. `"Qwen3.6-35B-A3B"` → 200; `"qwen3.6-35b-a3b"` → 200
(canonical echo); no `model` field → 200 echoing `Qwen3.6-35B-A3B`; batch with no field
labels its response document with the full name too. On a 3.8-27B default the same
listing leads with `Qwen3.8-27B`, startup logs `no drafter available for Qwen3.8-27B;
serving without speculative decoding`, and a request naming `Qwen3.6-35B-A3B` swapped
checkpoints and answered — the path where `EngineState::load`'s new architecture and
identity checks run against a non-default checkpoint.
