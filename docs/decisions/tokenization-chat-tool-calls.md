# Tokenization, chat, tool calls

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic; moved
here verbatim on 2026-09-06. Dated paragraphs, newest additions appended within their topic.


**The Qwen tokenizer.json (12,807,982 bytes, byte-identical between the two model
repos, sha256 5f9e4d49…) is vendored at reference/tokenizer.json and embedded via
include_bytes!, following laguna's embedded-tokenizer decision.** Qwen2 byte-level BPE,
NFC normalizer, no BOS ever prepended (`add_bos_token: false`, no post-processor). The
split regex differs from Qwen3 by `\p{M}` handling — do not reuse a Qwen3 regex
(2026-07-28).

**Qwen3.8's tokenizer.json differs from 3.6's by seven added tokens and nothing else,
and the embedded 3.6 file is what still ships (2026-08-14).** Compared structurally, not
by hash alone: `model.vocab` (248044 entries), `model.merges` (247587), the normalizer,
pre-tokenizer, post-processor and decoder are byte-identical; 3.8 adds
`<|audio_start|>`, `<|audio_end|>`, `<tts_pad>`, `<tts_text_bos>`, `<tts_text_eod>`,
`<tts_text_bos_single>`, `<|audio_pad|>` at ids 248070-248076, above every id the chat
path uses. Text therefore tokenizes identically under the embedded file, and client text
spelling one of those markers encodes as plain BPE — which is the safer behavior for
client content anyway. What is NOT decided here: whether a text-only checkpoint can emit
one of those ids at all, and what the embedded tokenizer would decode it to. Left as a
ledger item rather than improvised into a per-checkpoint tokenizer, since a second
12.8 MB embed for seven ids nothing renders is the kind of thing to decide deliberately.

**Qwen3.8 ships a different chat template; it is vendored beside 3.6's and the renderer
is unchanged — which means every default 3.8 conversation renders differently from the
official template, by one sentence (2026-08-14).** [SUPERSEDED 2026-08-19: the renderer
is now dialect-parameterized and the divergence is closed — see the next entry. The
template facts below all stand.] `reference/chat_template-qwen38.jinja`
(8952 bytes, verbatim from Qwen/Qwen3.8-27B). Diffed hunk by hunk against 3.6's: a
`reasoning_effort` system preamble, `preserve_thinking` defaulting to true instead of
false, the inline `<think>`-in-content parsing fallback removed, and an empty-arguments
guard on tool calls. The generation prompt — the block that decides what the model is
handed to continue — is byte-identical, and so is the `# Tools` prose, which is why one
hand-written renderer still serves every checkpoint.

The divergence is not hypothetical and is worth stating in full, because the defaults
make it universal rather than opt-in: with thinking ON (the default) and no
`reasoning_effort` given, 3.8's template resolves the effort to `xhigh` and prepends
"Reasoning effort is set to xhigh. Please think carefully through the task, validate key
assumptions, consider plausible alternatives, and prioritize correctness, consistency,
and clarity in the final answer." to the system block — creating one if the request has
no system message. xwen rendered neither that sentence nor the `low` variant, so every
default 3.8 conversation this server rendered was missing a system instruction the model
was trained to see. `medium` is the one effort level that injects nothing, so what xwen
rendered then was exactly the official `reasoning_effort="medium"` rendering. Accepted
knowingly for the arc that added the checkpoint (it is prompt semantics, not model math,
and the serve layer already had a conflicting `reasoning_effort` field of its own to
reconcile), but nobody should read "the generation prompt is byte-identical"
as "the prompts are the same".

Both vendored templates are cross-checked by chat.rs's tests (the fixed prose must
appear in each, and the generation-prompt block must match between them), so a future
release that moves either one fails a test rather than a reply.

**The renderer is parameterized by `ChatDialect`, and the 3.8 divergences above are
implemented behavior, not an accepted gap (2026-08-19).** `Model::chat_dialect()` maps
the 3.6 pair to `Qwen36` and the 3.8 to `Qwen38`; `ChatOptions::for_dialect` carries
each template's own defaults, and every prompt-building path (CLI gen/chat, all three
serve dialects, count_tokens, batch) reaches its options through it. The dialect was
kept a two-value enum on the options rather than a second renderer because the
templates' turn rendering and generation prompt are byte-identical — the differences
are confined to the system block and two defaults, and each is pinned by a test rather
than asserted in prose:

- The `reasoning_effort` preamble renders under `Qwen38` with thinking on: `xhigh`
  (the template's default) and `low` prepend their sentences — held as constants
  asserted verbatim, length included, against the vendored 3.8 template and asserted
  ABSENT from the 3.6 one — while `medium` injects nothing, making a medium render
  byte-equal to a 3.6 render of the same conversation. With no system message the
  dialect synthesizes a system block to carry the sentence (the template's own
  behavior); the block anchors the prefix cache like a client's and the preamble stays
  out of the client-content spans, since it is template prose, not client content.
  With tools it opens the block ahead of the `# Tools` header.
- `preserve_thinking` defaults true under `Qwen38` (template line 116's `is undefined
  or is true`), false under `Qwen36`.
- An empty system message emits no block under `Qwen38` where `Qwen36` emits the empty
  block its template unconditionally writes.
- `split_reasoning` — the inline `<think>`-in-content fallback — runs under `Qwen36`
  only. The 2026-08-14 record (and the ledger item it fed) claimed xwen "never
  implemented that fallback"; that was WRONG — chat.rs had it and ran it
  unconditionally, so a 3.8 turn replaying reasoning inside content was getting the
  3.6 reading. It is now dialect-gated, and a 3.8 turn renders such content verbatim,
  as its template does.

`TOKENIZATION_RULES_VERSION` went 2 → 3 with this: the same 3.8 conversation encodes
differently under the current rules, and a stale disk image must fail the stamp check
rather than longest-common-prefix-match a stream these rules would never produce.

**chat.rs is a hand-written Rust port of the official chat_template.jinja (7764 bytes,
byte-identical across both Qwen 3.6 repos), keeping laguna's content/structure separation** so
pasted text discussing control tokens can never become control tokens. The subtle rules,
verified by rendering the real template: string tool-arguments render RAW (non-strings
JSON-encode); OpenAI-style JSON-string `arguments` must be parsed into a map first
(template raises on strings); thinking blocks are kept only for turns strictly after the
last user turn that is not wholly a `<tool_response>` wrapper (or all, under
`preserve_thinking`); generation prompt opens an unclosed `<think>\n` (thinking on) or
emits a closed empty block (thinking off); consecutive tool results collapse into one
user turn. Rendered test vectors from the bootstrap research are the fixture set
(2026-07-28).

**One deliberate divergence from template byte-parity: a tool result as the FIRST
message is refused** (`ChatError::ToolResultOpensConversation`). The reference template
hits undefined `loop.previtem` there and emits a turn that closes without ever opening
(`<|im_end|>` with no `<|im_start|>user`); byte-parity would mean handing the model a
malformed boundary. Refusal ordering otherwise follows the template exactly:
NoMessages → NoUserQuery → SystemNotFirst (2026-07-28).

**Bodies are stripped with Python's str.strip() whitespace set (29 codepoints,
including U+001C–U+001F), not Rust's `trim`.** Jinja's `|trim` is str.strip(), and the
difference decides real behavior: whether a body reads as a bare `<tool_response>`
wrapper (which moves last_query_index and therefore which turns keep their reasoning)
and whether an assistant turn counts as empty when its first tool call picks its
separator. Verified against an exhaustive Unicode sweep (2026-07-28).

**Constrained decoding's control-token safety is a compile-time property enforced by a
test, not a runtime force-mask.** toktrie marks every `<…>`-shaped added token special
regardless of the tokenizer's special flag, so no grammar byte can ever match a control
marker; a per-draw 250-id mask sweep would duplicate that guarantee on the hot path and
need an EOG carve-out. The guarantee rests on toktrie's bracket HEURISTIC, not on
special:true — a future marker spelled without angle brackets would be
grammar-reachable, which is why `no_control_token_is_ever_offered` sweeps the full
control range at every step and asserts the mask stayed wide. The constrain trie is
sized to the model's logit width 248320 (padded tail unreachable by construction), and
`new()` refuses a checkpoint with a different width — this sizing fixed a latent bug
where every constrained serve request died on a short mask (2026-07-28).

**tokenizer.rs is the single owner of every token id in the crate**, including the
hardcoded second stop id 248044 that no GGUF key advertises; config.rs imports it. Two
vocab sizes are exposed deliberately: `vocab_size()` = 248070 encodable id space,
`PADDED_VOCAB` = 248320 logit width — callers pick by which side of the sampler they
are on (2026-07-28). The serve engine's tool-call span parser is now the same rule
rather than an exception — see the entry below for what the exception cost.

**The serve engine parses Qwen's real call format, and a span it cannot read degrades to
text.** The inherited parser was laguna's twice over. Its span markers were literal ids
`25`/`26`, which in Qwen's vocabulary are `:` and `;` (the real `<tool_call>` pair is
248058/248059), so every colon in ordinary prose opened a phantom span and every
semicolon closed one — truncating replies into a discarded span or reporting a
fabricated call, while genuine `<tool_call>` tokens passed through as text. Its interior
grammar was laguna's `<arg_key>`/`<arg_value>`, strings absent from Qwen's vocabulary
and never emitted by chat.rs. The parser now sources both ids from `LagunaTokenizer` and
reads the format chat.rs renders: `<function=NAME>` then per-argument
`<parameter=KEY>\nVALUE\n</parameter>` then `</function>`, one function per span, with
the newlines around a value treated as framing rather than content. Two rules follow
from the template rather than from taste. The `</tool_call>` token is structural
wherever it lands, mid-value included — chat.rs writes a literal `</tool_call>` inside
an argument as ordinary content, so it never encodes to the added token, and reading the
token as content would let one malformed value swallow the rest of the reply. And a span
that never names a callable tool degrades: its raw text, markers included, goes to the
client as answer text with a logged warning, instead of being silently dropped as the
old parser dropped it. Never discard, never fabricate. The class of bug is closed by
construction in the tests: they drive the emitter over ids from the real embedded
tokenizer, round-tripping conversations that chat.rs rendered, and one hostile case
feeds prose full of `:`, `;` and `<function=` text and asserts zero calls with
byte-identical output (2026-07-28).

**`--ban-string` protects the stop ids the decode loop actually uses.** `scan_banned`
guarded the compile-time `LagunaTokenizer::EOG` while the loop stops on
`Sampler::eog_ids()`, which `XwenConfig` derives from the checkpoint's metadata. The two
agree on both shipped files, so nothing was broken — but they are independent sources,
and a checkpoint declaring a different `eos_token_id` would leave its real stop token
bannable, letting `--ban-string` remove the only token that can end a reply. The
protected set is now passed in from the sampler, so the guarantee holds by construction
rather than by the two sources happening to match (2026-07-28).
