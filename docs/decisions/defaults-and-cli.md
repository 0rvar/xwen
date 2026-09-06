# Defaults and CLI surface

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic; moved
here verbatim on 2026-09-06. Dated paragraphs, newest additions appended within their topic.


**Default checkpoints are ggml-org's Q4_K_M files.** `ggml-org/Qwen3.6-27B-GGUF` and
`ggml-org/Qwen3.6-35B-A3B-GGUF` are HF's own llama.cpp org — the closest thing to
official GGUFs (Qwen published safetensors/FP8 only). Q4_K_M over Q8_0 because decode is
bandwidth-bound and the Q4_K_M mix (attention/ssm/shared-expert Q8_0, expert stacks
Q4_K, lm_head Q6_K) keeps the quality-critical planes at 8-bit anyway. Single files, no
sharding (2026-07-28).

**Qwen3.8-27B is a registry entry, not a port (2026-08-14).** Its `config.json` is
byte-identical to Qwen3.6-27B's — same graph, hparams, tokenizer ids, generation config
— and its GGUF declares the same `qwen35` architecture, rope sectioning and ssm keys, so
it needs no model math, no new geometry and no parity run: it is the same forward pass
over different weights. `ggml-org/Qwen3.8-27B-GGUF` for the same reason the 3.6 files
were chosen. Three things about it are genuinely new. It ships **no DFlash sidecar**, so
speculation is absent rather than configurable — every drafter accessor is `Option` and
a zero-flag run logs one line and decodes plain (~25 tok/s, against the 27B's drafted
37-38); the repo's MTP sidecar is unread (TODO.md). [SUPERSEDED 2026-08-15 by the MTP
arc: the sidecar is read, the accessors resolve, and a zero-flag run drafts. The
`Option` shape stayed — it is about a checkpoint that ships no sidecar, not about this
one.] Its Q4_K_M mix puts the 16
`attn_output.weight` tensors at **Q6_K** where 3.6 had Q8_0 — upstream's
`output.weight=q6_k` rule substring-catches `attn_output`; nothing asserts on that
plane's quant and lm_head already exercises Q6_K. And its tokenizer.json is NOT
byte-identical to 3.6's: it adds seven audio/TTS specials at 248070-248076 over an
identical base vocab and merge table, which the embedded 3.6 tokenizer therefore
tokenizes text identically to (see "Tokenization" for what was and was not decided).

**Sampling defaults follow generation_config.json: temp 1.0, top_p 0.95, top_k 20.**
[SUPERSEDED 2026-08-19 for the sampling half: defaults are now keyed to thinking mode —
see the mode-keyed entry below. generation_config.json's values ARE the thinking set,
which stays the default for thinking runs and for every mode-less path.] Stop tokens
are the generation_config list `[248046 <|im_end|>, 248044 <|endoftext|>]` —
config.json's single `eos_token_id: 248044` is wrong for chat and runs straight past
turn boundaries (2026-07-28).

**Sampling defaults are mode-keyed: thinking temp 1.0 / top_p 0.95, non-thinking 0.7 /
0.80, top_k 20 both, identical across all three checkpoints (2026-08-19).** The
evidence is the official model cards, which key their recommendation to thinking on/off
and nothing else: the HF READMEs of Qwen/Qwen3.6-27B, Qwen/Qwen3.6-35B-A3B and
Qwen/Qwen3.8-27B all give thinking 1.0/0.95/20 and instruct 0.7/0.80/20 — three cards,
two sets, no per-checkpoint variation, which is why
`SamplerOptions::recommended(thinking)` takes a bool and not a `Model`.
generation_config.json carries only the
thinking set, which is why the old fixed defaults were that set and why `Default` stays
it (raw prompts and benches have no chat mode and keep sampling as they always did).
The resolution order everywhere is explicit value → mode recommendation: CLI sampling
flags became Option-valued (a mode-dependent default cannot live on a clap
`default_value_t` — the DraftArgs precedent), and serve's fixed
`DEFAULT_TEMPERATURE`/`TOP_K`/`TOP_P` constants are gone because a server-wide constant
cannot know a request's mode; `ServeSettings` sampling keys are Options resolved per
request AFTER thinking is known, request over config over recommendation. A pinned
config value deliberately pins one number for both modes — that is what an operator
writing a number means — and unset gives each request its mode's own. The cards'
remaining recommendations did not ship: the penalties (see "Thinking budget and
sampling controls") and the 3.6 pair's third "thinking, precise coding" set
(0.6/0.95/20), which is not auto-selectable — nothing in a request says "coding" — and
is reachable as an explicit `--temp 0.6`.

**`--reasoning-effort` on a 3.6 checkpoint is a startup error, not a no-op
(2026-08-19).** The flag names a parameter of the 3.8 chat template; the 3.6 template
has none, so a supplied level would change nothing. Ignoring it would train the
operator to believe it did something — the `--model-size` rule again: flags
cross-check instead of shrugging, and the check runs before the 20 GB load. Unset is
allowed everywhere because the default level (xhigh) renders nothing on 3.6 anyway.
The serve-side `[thinking] effort` / `--reasoning-effort` default is the deliberate
exception: it is a server-wide setting on a server that may load any checkpoint per
request, so it is documented as inert on 3.6 rather than refused (refusing would make
a 3.8-tuned config invalid the day a 3.6 request arrives). Both `--no-think` and
`--reasoning-effort` are rejected with `--raw`, the same class as the existing guarded
combos: they describe the chat template, which a raw prompt never renders.

**`max_ctx` is a ceiling, not an allocation; the CLI defaults to 131072 and serve to
the trained window (2026-08-11).** Full-attention KV buffers start at 8192 positions
(`model::KV_INITIAL_CTX`) and double on demand up to `max_ctx`
(`LayerCache::ensure_full_capacity`; growth is monotonic for a model's life and logged
per step). Growth copies the WHOLE old buffer, deliberately not just the committed
rows: the grown buffer is the old one plus a zeroed extension, so every property that
held of the fixed allocation — `LayerSnapshot::Full`/`LayerCheckpoint::Full` carrying
no data because restore and rollback are length truncations, rows above a rewound
`len` still holding what they held — carries over with nothing to argue about which
engine flow depends on rows past `len`. A committed-rows-only copy was the first cut
and was replaced for exactly that argument's sake; the cost difference is a bounded
device blit, O(log) times per lifetime. Page-in grows through `import_full_kv` exactly
as a prefill would, and the host-image pre-flight no longer bounds a restore position
by allocated slots (max_ctx, checked at model level, is the real bound). What "reset" means is
dropping the model: the serve idle unload therefore shrinks a grown cache for free, and
no in-life shrink mechanism exists or is wanted (capacity is a high-water mark of real
usage). The alternative — preallocating at max_ctx, the pre-2026-08-11 behavior — made
big defaults expensive (16 GiB of idle KV on a 27B serve at 256k) and kept the CLI
default pinned at a timid 8192; lazy growth makes the 131072 CLI default cost 0.5 GiB
(27B) / 0.16 GiB (35B) until a prompt actually grows past 8k. A growth pass ends in
one `wait_until_completed`: candle's Metal pool frees the replaced buffers only at a
sync, and without one the whole pass holds old and new allocations side by side. Rope
tables still build to max_ctx at load — 64 MB at 262144, not worth the machinery
(2026-08-11).
