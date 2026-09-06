# 2026-08-19 — The chat template becomes a per-checkpoint DIALECT: 3.8 gets its reasoning_effort preamble and preserve_thinking default, gen/chat gain --no-think and --reasoning-effort, and sampling defaults go mode-keyed (1.0/0.95/20 thinking, 0.7/0.80/20 instruct)

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


Two commits, one arc (a2e02d0, 205d9ba). The first makes the renderer
dialect-aware; the second surfaces the knobs on every entry point and re-keys the
sampling defaults. Together they close the divergence recorded on 2026-08-14: every
default 3.8 conversation was rendering without the system instruction its template
injects, equal to the official `reasoning_effort="medium"` rendering rather than the
official default.

**`ChatDialect { Qwen36, Qwen38 }`, from `Model::chat_dialect()`.** One renderer
still serves every checkpoint — the generation prompt and every turn render
byte-identically between the templates, which is what made a parameterized renderer
honest — but `ChatOptions` now carries the dialect plus a `ReasoningEffort { Low,
Medium, Xhigh }`, and `ChatOptions::for_dialect` carries each template's own
defaults. What the 3.8 dialect renders that 3.6 does not, each pinned by a test:

- The `reasoning_effort` system preamble, verbatim template prose (the xhigh and low
  sentences are constants asserted to appear character-for-character in the 3.8
  template and NOT in the 3.6 one, lengths included). `xhigh` is the template's own
  default; `medium` injects nothing (so a medium render is byte-equal to a 3.6 render
  of the same conversation); rendered only while thinking is on. With no system
  message the dialect synthesizes a system block to carry the sentence — and that
  block anchors the prefix cache like a client's own, with the preamble kept out of
  the client-content spans. With tools, the sentence opens the block AHEAD of the
  `# Tools` header.
- `preserve_thinking` defaults TRUE (3.6 stays false): a replayed 3.8 turn keeps its
  reasoning where 3.6 drops it once superseded.
- An empty system message emits NO block, where 3.6 emits the empty block its
  template unconditionally writes.
- No inline `<think>`-in-content splitting: `split_reasoning` now runs under the 3.6
  dialect only, so a 3.8 turn that carries a `<think>` block inside content renders
  it verbatim, as its template does. (The 2026-08-14 record claimed xwen "never
  implemented that fallback" — wrong: it existed and ran unconditionally; it is now
  gated, not added.)

**`TOKENIZATION_RULES_VERSION` 2 → 3**, because the same 3.8 conversation now
encodes to a different token stream and a stale disk image must not
longest-common-prefix-match a stream the current rules would never produce.

**CLI: `--no-think` and `--reasoning-effort <low|medium|xhigh>` on gen and chat.**
`--reasoning-effort` on a 3.6 checkpoint is a startup error naming the checkpoint —
the flag would change nothing there, and this repo's flags cross-check instead of
shrugging (the `--model-size` rule); unset costs nothing to allow, since the default
level renders nothing on 3.6. Both are rejected with `--raw` (the same distortion
class as the existing guarded combos: they describe a template a raw prompt never
renders), and both validate before the 20 GB load. A repl side-fix rode along:
`stream_reply`'s `in_think` was initialized unconditionally true, which under
`--no-think` would have filed an entire reply as hidden reasoning; it now follows
`enable_thinking`, and the cancel-retry rule stops waiting for a `</think>` that a
no-think turn will never emit.

**Sampling defaults are now keyed to thinking mode, per the official cards.** All
three checkpoints' HF cards recommend the same two sets: thinking temp 1.0 / top_p
0.95 / top_k 20 (what generation_config.json carries and what everything always
used), non-thinking ("instruct") temp 0.7 / top_p 0.80 / top_k 20.
`SamplerOptions::recommended(thinking)` is the single source; `Default` stays the
thinking set, so every mode-less path (raw prompts, benches) samples as it always
has. The CLI's `SamplingArgs` became Option-valued (the DraftArgs pattern — a
mode-dependent default cannot live on the flag), and serve's
`DEFAULT_TEMPERATURE`/`TOP_K`/`TOP_P` constants are gone: `ServeSettings` sampling
keys are Options, resolved per request AFTER the request's thinking mode is known,
as request value → server-configured value → mode recommendation. A pinned server
value pins one number for both modes; unset gives each request its mode's own. The
cards also recommend penalties (presence_penalty 1.5 on some arms) and a third
"precise coding" set — neither shipped, both ledgered with the reasoning (TODO.md;
the penalties one is a real entanglement with the speculative verify path, not an
oversight).

**Serve: one `reasoning_effort` field drives BOTH the think budget and the 3.8
template preamble.** The OpenAI dialect's field keeps its budget mapping unchanged
(none=off, minimal=1024, low=4096, medium=16384, high/xhigh/max=uncapped) and now
also selects the template level, nearest-mapping the levels the template does not
define: minimal→low, high/max→xhigh. That is a deliberate divergence from llama.cpp,
which passes the raw string into the template and lets the jinja raise — serving the
nearest level beats a template error, and the nearest level is what the client
meant. New `chat_template_kwargs` request field (the official card's / vLLM's
shape) accepting `enable_thinking`, `preserve_thinking`, `reasoning_effort` — the
LAST of these takes the template's three levels only, since it is the raw template
parameter. Kwargs are validated STRICTLY: unknown key, wrong type, or an off-scale
level is a 400 naming the offender, the one exception to the dialect's
accept-and-drop permissiveness (a dropped sampling param degrades a completion; a
dropped template kwarg changes the prompt the client believes it asked for). The
top-level field beats the kwarg, llama.cpp's precedence. `[thinking] effort` in the
config and serve `--reasoning-effort` set a server-wide template-effort default
(inert on 3.6, whose template has no such parameter — which is why serving one is
not a config error). The native dialect gained `reasoning_effort` and
`preserve_thinking` fields; the Anthropic dialect's shape is unchanged (no natural
field — server-wide default applies, ledgered), but its sampling went mode-keyed
and `count_tokens` now resolves the target checkpoint and renders the preamble, so
a count matches the generation it predicts. Batch renders each item under the
loaded checkpoint's dialect (`run_batch` takes it like it takes the label); batch
thinking still defaults off, so the preamble never renders there unless asked.

**Verification.** `cargo test --release`: 838 lib + 69 binary passed, 0 failed. The
dialect behaviors are each pinned — the preamble prose against both vendored
templates, the synthesized block's cache anchor and token-boundary invariant, the
medium/thinking-off byte-equalities with 3.6, the empty-system divergence, the
inline-`<think>` divergence, the kwargs 400s, the effort-level FromStr refusing the
wider OpenAI scale, and the mode-keyed resolution order on all three dialects. No
model math changed; the parity gate was not run and did not need to be.
