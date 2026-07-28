# Engineering log

Reverse-chronological. Heading convention: `## YYYY-MM-DD — headline stating what
shipped, ideally with the number`. Same-day entries disambiguate in the heading text.
Superseded entries are marked in the headline, never deleted.

## 2026-07-28 (later still) — Second-model review: one finding, and it was our own docstring overclaiming

**Context.** The external-model reviewer (GPT-5.6 via codex, ~35 min, xhigh) ran over
the full retarget alongside the two Claude reviewers. It cleared every model-math trap
independently (interleaved gate split, partial rope, rollback indexing, router order,
chunk carry) and missed the tool-call parser bug the integration reviewer caught — the
two reviewer families found disjoint issues, which is the argument for running both.

**Change.** Its one finding (MEDIUM): `kv_rollback`'s docstring promised caches
"byte-identical to ones that only ever advanced over the committed tokens" — but the
q8/f16 dual-storage split makes cache-mutating projections partition-dependent
(a verify batch of 9+ tokens runs the f16 plane where one-token decode runs q8 GEMV),
so the counterfactual-identity half of that promise is false in exactly the
speculative case it describes; the parity gate cannot see it because it runs one fixed
partitioning. Resolution: the mechanism is kept (the bytes-halving is why dual storage
exists), the CONTRACT was corrected — restore is a bit-exact replay of recorded bytes,
cross-partition agreement is numeric and parity-gated, `XWEN_ATTN_DEQUANT` pins one
representation when partition-independence matters. decisions.md records it under
Kernel policy; drift magnitude at the 8↔9 boundary is a ledger item, unmeasured.

**Verdict.** No code behavior changed; a claimed identity was demoted to the observed
one, per this project's own first rule. Review pass complete: three reviewers, one
critical fix (tool-call parser), one contract correction, model math clean.

## 2026-07-28 (late night) — Serve integration fixes: the tool-call parser was reading `:` and `;` as span markers

**Context.** An adversarial review of the inherited serve/ tree against the Qwen
retarget turned up four integration defects. One is severe and had no chance of being
caught by the suite that covered it.

**The headline: `serve/engine.rs` opened tool-call spans on ordinary punctuation.** The
span parser carried laguna's token ids as literals, `TOOL_CALL_OPEN: u32 = 25` and
`TOOL_CALL_CLOSE: u32 = 26`. In Qwen's vocabulary 25 is `:` and 26 is `;`; the real
`<tool_call>` pair is 248058/248059, and `tokenizer.rs` has held those constants since
the fork. So for any request carrying tools, every colon the model wrote in prose opened
a span and every semicolon closed one. What followed a colon stopped being answer text
and started being parsed as a call — delivered as a fabricated tool call if it happened
to parse, silently discarded by the heal path if it did not, which is to say the reply
truncated at the first colon. Genuine `<tool_call>` tokens meanwhile fell through to the
`_` arm and reached the client as literal text. The interior grammar was laguna's too:
`<arg_key>`/`<arg_value>`, strings that are not in Qwen's vocabulary and that chat.rs
has never emitted — so even a correctly-framed span parsed to nothing.

**Why the suite missed it.** There were seventeen tests over this parser and they all
passed. Every one of them scripted the token stream by hand as `(TOOL_CALL_OPEN,
"<tool_call>")` — the constant paired with the text a correct constant would decode to.
That pairing is the bug, asserted as a fixture. The tests agreed with the parser about
what id 25 meant, and neither had ever been asked what the tokenizer thought. A test
that builds its input from the same wrong constant as the code cannot fail when the
constant is wrong; it can only fail when the code stops being self-consistent. The fix
is not more assertions, it is a different input source, so the tests now drive the
emitter over ids produced by the real embedded tokenizer — round-tripping conversations
that `chat.rs` rendered, and one hostile case that feeds prose full of `:`, `;` and
`<function=` text and asserts zero calls with byte-identical output. Under the old
constants that case produces a tool call named `name`.

**Change.** The ids come from `LagunaTokenizer` now; the interior parser reads what
chat.rs writes (`<function=NAME>`, `<parameter=KEY>\nVALUE\n</parameter>`,
`</function>`, one function per span, framing newlines stripped from values). Two
behaviors changed on purpose rather than by translation. `</tool_call>` is structural
wherever it lands, mid-value included — the template writes a literal one inside an
argument as content so it never encodes to the added token, and treating the token as
content is what let a malformed value swallow the rest of a reply. And a span that never
names a callable tool now degrades: raw text, markers included, to the client as answer
text with a logged warning (`ServeLog::ToolSpanDegraded`, counted separately from
`healed` in the per-request report). Never discard, never fabricate.

**Three smaller items.** Drafting was on by default, inherited from laguna, which made
every zero-flag `xwen generate` and `xwen serve` abort at startup — `xwen serve` before
the listener bound. The error turns out to be `missing GGUF key dflash.decoder_arch`
rather than the `decoder_arch == "laguna"` mismatch the review predicted: the shipped
sidecars have no such key, so the failure precedes the check adaptation was expected to
repoint, with `enc.aux_norm` and `blk.N.attn_gate` absent behind it. Default flipped to
off; asking for a drafter still fails loudly. `serve/config.rs` was sizing caches with
laguna's geometry — 48 KiB/token from 12 full layers × 8 KV heads × 128 head_dim, and a
72 MiB snapshot described as copies of 36 SWA rings, in a model with no SWA layer at
all. Real figures are 20 KiB/token (35B-A3B) and 64 KiB (27B), with snapshots of
DeltaNet recurrent state at a fixed 62.8/149.6 MiB; they are derived on `hub::Model` now
rather than carried as constants. And `scan_banned` was protecting the compile-time EOG
set while the decode loop stopped on the GGUF-derived one — harmless today, since the
two match on both checkpoints, but only by coincidence.

**Verdict.** 645 lib tests pass, up from 642. Two pre-existing failures remain, both
`dflash::tests::real_file_*`, both the unadapted-drafter story above; they cannot pass
until TODO.md P9 lands and were failing before this arc. The tool-call parser is the
first inherited subsystem found to be not merely unadapted but actively wrong on Qwen
input, and the way it survived — tests built from the same constant as the code — is
worth carrying into the review of every other inherited dialect layer.

## 2026-07-28 (late night) — Parity harness live (P7): both checkpoints match upstream llama.cpp, floors an order of magnitude tighter than laguna's

**Context.** Everything before this rested on cited source lines, hand-computed unit
tests, and one greedy eyeball. The engine had never been compared against another
implementation on the same weights, and the 27B had never been run at all. P7 was the
item standing between "it produces fluent text" and "it computes the right thing".

**Change.** Upstream `ggml-org/llama.cpp` shallow-cloned into `reference/llama.cpp`
and PINNED at `e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0` (recorded in docs/parity.md).
This session materialized it as a plain clone and gitignored the path, reasoning that
a submodule puts the reference tree in the index and the index is the human's review
surface; a `.gitmodules` entry declaring the same path as a submodule was staged
concurrently from elsewhere, and the owner settled it that night in favour of the
submodule — the gitlink makes the oracle sha reviewable in the diff, so moving the pin
becomes a staged change someone approves, which is the property the pin exists to
have. `scripts/build-llamacpp.sh` retargeted at the path.
`tests/fixtures/parity-prompts.json` regenerated with Qwen ids from the oracle's own
`llama-tokenize --no-bos`; the SWA-specific `long-swa` fixture replaced by
`long-mixed`, 612 tokens of prose that stresses the DeltaNet recurrence instead of a
sliding window that does not exist here. `hf.ts` repointed at the two ggml-org repos
with a size selector; `parity-gate.ts` gained `--model-size 27b|35b` and namespaces
every artifact by checkpoint basename (two official files, two architectures, two
sets of floors — they must never share a parity dir or a frozen ppl fixture).
`ref-dump.sh` retargeted. `scripts/parity.ts` learned the tap-name mapping
(`refTapNames`) rather than renaming engine taps. Laguna's `reference-ppl.json`
deleted; the committed-fixture test now validates every per-checkpoint fixture.

**Two latent parser bugs in the inherited `parity.ts`, both silent corrupters.**
First, the node-header regex captured names as `(\S+)`, so headers with spaces
(`cache_r_l0 (reshaped)`, `(view)`) were skipped and their value rows were attributed
to the previous node — which kept that node's `sum` but replaced its sampled row with
an unrelated tensor's. Symptom: `attn_norm-0` reporting `rowRelL2 = 2.29e+6` while its
values were in fact digit-identical to the oracle's. Second, `FLOAT_RE.test(line)` on a
shared `/g` regex advanced `lastIndex`, dropping every other value row. Neither would
ever produce a false PASS, but both produced convincing false divergences — the first
one cost a real detour before it was pinned down.

**Result.** Both checkpoints agree with upstream on identical GGUF weights. The final
four-tier run with the calibrated constants in place is `ALL PASS (6 graded)` on each —
42 s warm for the 35B, 2.0 min for the 27B, since the Reference dumps are cached and
only the candidates regenerate.

Track A (first-divergence bisection, 35B, code-short, 242 taps compared): no cliff.
The sampled-row rel-L2 profile is smooth and *flattens* rather than compounding —
`l_out` runs 1.8e-3 at layer 0, 1.2e-3 at 7, 2.4e-2 at 23, 1.4e-2 at 39 — and the
final-logits sampled cosine is 0.999995. Individual `sumRelErr` spikes up to 1.9e-1
are near-cancelling residual sums, not divergences; their own `rowRelL2` stays in the
neighbourhood's band.

Track B, `bun scripts/parity-gate.ts`, 35B-A3B Q4_K_M:

| tier | fixture | result |
|---|---|---|
| strict | code-short | PASS, cosine 0.999999861, top-1 = ref, top5 5/5 |
| mm | code-short | PASS, cosine 0.999539782, top-1 = ref, top5 5/5 |
| decode | code-short | PASS, 63/64 agree, 1 excused near-tie (0.0040 logit) |
| decode | text-mixed | PASS, 62/64 agree, 2 excused (0.5567, 0.2606) |
| decode | long-mixed | PASS, **64/64 agree**, 0 excused |
| ppl | — | PASS, Δmean_nll 0.000511 (fused 1.694170 vs reference 1.693659 over 4218 tokens) |

**The 27B's first forward was correct.** No bisection was needed: it loaded (18.2 GB
resident), prefilled 58 tokens in 9.6 s cold, and produced a top-5 that tracks the
35B's on the same prompt. Its parity numbers are *better* than the 35B's across the
board — strict bitwise 1.000000000, mm ≥ 0.999993294, decode **64/64 with zero excused
near-ties on all three fixtures**, ppl Δ 0.000221 (fused 1.748093 vs reference
1.747872) — with the caveat that on a dense model the
strict tier is near-vacuous, since `--moe-impl reference` and `fused` run the same
`DenseMlp` and the strict env pins everything else classic on both sides. The 27B's
real signal is the mm/decode tiers, which exercise the f16 attention path and the
fused glue.

**Floors, calibrated across both checkpoints and all three fixtures** (the constants
are global, so they are set under the WORST observed value): `COS_MIN_STRICT = 0.9998`
(worst achieved 0.999894, 35B long-mixed) and `COS_MIN_MM = 0.999` (worst achieved
0.999540, 35B code-short — ~1.7x the observed prompt-to-prompt spread). Both are an
order of magnitude tighter than laguna's 0.9955 / 0.985: these kernels track the
oracle much more closely on this architecture, largely because the Qwen Q4_K_M mix
keeps attention, ssm and shared-expert weights at q8_0.

**Verdict.** The engine is parity-validated. This is the checkpoint that makes every
later kernel change checkable instead of hopeful, and it is the gate P8's DeltaNet
kernels will be graded against. Four things are explicitly NOT proven and are in the
ledger: Track A cannot localize inside a layer (the tap set is still laguna's six —
no DeltaNet core, router logits, or shared-expert gate taps, which would need plumbing
in the model-math files); `provenance.flash` says `"fused"` while `flash.metal` is
compiled at head dim 128 and cannot serve Qwen's 256, so prefill is really candle sdpa
with a materialized mask; the dense strict tier as noted above; and the `_Q8` widened
bands were inherited, not recalibrated — measured worst per-step l2 deviation is
1.0211 against a 1.5 band (far too loose), while the near-tie window genuinely needed
its widened 1.0 (text-mixed step 15 excused at 0.5567).

## 2026-07-28 (night) — First real-weights decode: correct output, clean stop, 59 tok/s on the 35B-A3B

**Context.** Everything to this point rested on cited source lines and hand-computed
tests; nothing had touched real weights. The 20.4 GB ggml-org Q4_K_M finished
downloading and the release binary was fresh.

**Change.** None — this is the first execution of the P2-P6 stack:
`xwen generate -p "What is 2+2? Answer with just the number." --temp 0 --no-draft`.

**Result.** Correct end to end on the first attempt: a coherent five-step thinking
block, `</think>`, the answer `4`, and a natural stop on `<|im_end|>` (143 tokens
emitted with 369 of budget left — the stop list works). Load 2.8-3.0s via the mmap
path, 19.2 GB resident as predicted (weights 19.0 + KV 0.2 + recurrent state 0.1 at
max_ctx 8192). Warm numbers, power mode unmeasured: prefill 23 tokens at 167.6 tok/s,
decode 58.6-59.0 tok/s. The 8.76s cold-run prefill was Metal pipeline compilation, not
the DeltaNet scan — the "will look like a hang" warning in the P2-P4 entry overstated
short-prompt cost; long-prompt prefill through the per-token scan remains unmeasured
and P8 still owns it. Coherence at temp 0 is strong indirect evidence for the whole
trap set (tiled V-heads, norm baking, gate split, router order): any of those wrong
produces fluent garbage, not correct arithmetic under a formatting constraint.

**Verdict.** The engine runs Qwen 3.6 correctly by the greedy-eyeball half of the
fallback gate. Not yet parity-validated — P7 is unchanged as the next honest
checkpoint. Decode at ~59 tok/s pre-kernel-work is 2.7x laguna's decode on 2.7x
fewer active params, i.e. exactly the bandwidth story, and the DeltaNet reference's
~240 extra dispatches per token are not yet visibly the bottleneck at this length.

## 2026-07-28 (evening) — ChatML port, tokenizer swap, and a latent constrain-width bug (P5+P6): suite fully green, 642/0

**Context.** The fork still rendered Laguna's angle-bracket template over Qwen's vocab,
tokenizer.rs carried Laguna ids, and two tests failed on vocab assumptions. hub/CLI
still resolved poolside checkpoints.

**Change.** chat.rs rewritten against the vendored chat_template.jinja (ChatML, tools
preamble, XML-ish call format with string-args-raw, collapsed tool-result turns, the
last_query_index thinking-retention rule, open-`<think>` generation seeding), keeping
laguna's content/structure separation; typed `ChatError` mirroring the template's raise
cases plus one deliberate divergence (tool-result-first refused — the template emits a
malformed boundary there). tokenizer.rs repointed at Qwen ids and made the single owner
of every token id, exposing both vocab sizes (248070 encodable / 248320 logit width).
hub.rs got a per-model checkpoint table (`--model-size 27b|35b`, 35b default,
API-verified filenames); sampling defaults aligned to 1.0/20/0.95. The constrain trie
is now sized to the logit width — which surfaced and fixed a latent bug where every
constrained serve request would have died on a 248,096-bit mask against 248,320 logits.
Control-token safety under grammars turned out to rest on toktrie's angle-bracket
heuristic rather than the special flag; documented and pinned by a full-range sweep
test. Design calls recorded in decisions.md "Tokenization, chat, tool calls".

**Result.** 662 lib tests: 642 pass, 0 fail, 20 ignored (perf benches). The five
template vectors are byte-exact; a differential harness ran 20,000 fuzzed conversations
plus exhaustive role-shape and Unicode-whitespace sweeps against the reference jinja
with zero divergences. Adversarial review by a second model family found two real bugs
pre-merge (silent second-system drop; trim whitespace set), both fixed.

**Verdict.** The prompt/token surface is done and independently verified. What remains
before trusting the whole stack is a forward pass against real weights — the 35B
Q4_K_M was downloading as this entry was written.

## 2026-07-28 — Model core retargeted to Qwen 3.6 (P2-P4): config, loader, DeltaNet reference, attention and MoE

**Context.** The mechanical fork built green but still computed Laguna: 48 uniform
attention layers, a sliding-window ring every fourth layer, a softplus per-head output
gate, and a sigmoid/bias/scale MoE router. P2-P4 replaced that core wholesale with the
Qwen 3.6 hybrid, on the critical path for everything downstream. Two research agents ran
alongside the implementation — one extracting llama.cpp master's `qwen35.cpp` /
`qwen35moe.cpp` / `delta-net-base.cpp` graphs, one range-parsing the shipped ggml-org
GGUF headers — so every load-bearing form was written first from the CLAUDE.md cheat
sheet and then confirmed or corrected against a cited source line.

**Change.** `LagunaConfig`/`LagunaModel` → `XwenConfig`/`XwenModel`. config.rs parses
both archs, rejects anything else, and derives per-layer `LayerKind::{Full, Linear}`
from the `full_attention_interval` key rather than a hardcoded 4. New `linear_attn.rs`
holds the gated-DeltaNet layer as a frozen oracle in the `ReferenceExperts` sense:
composed candle ops, recurrent form only, fp32 state, one sequential scan step per
token. attention.rs gained the double-width `attn_q` with its per-head interleaved gate
(strided split, not a halving of the row), QK-RMSNorm over 256 dims, partial NEoX rope
at n_rot 64 / theta 1e7, sdpa scale 1/√256, and an ELEMENTWISE `sigmoid(gate)` — 4096
independent values per token, not one scalar per head. moe.rs swapped the router for
softmax-over-all-256 → top-8 → renormalize with the f16 floor, no selection bias and no
weight scale, and gave the shared expert its scalar sigmoid gate. kv_cache.rs grew a
third `LayerCache::Linear` variant carrying conv window and delta state in f32 through
checkpoint, rollback, snapshot and the on-disk framing; the SWA ring machinery is left
in place but nothing on the model path constructs it. model.rs dispatches per layer and
loads the pre-MLP norm from `post_attention_norm`. The mmap/no-copy loader, ExpertStack,
QLinear and the dual-storage attention planes were not touched — only the name table.

**Result.** `cargo build` and `cargo test --no-run` green. 659 lib tests: 637 pass, 20
ignored, 2 fail — `generate::ban_scan_catches_every_em_dash_token` and
`constrain::walks_a_valid_document_and_completes`, both in tokenizer/constrain code this
arc never touched, and the second is the already-logged `<think>`-is-not-special trap.
23 tests are new, including the three the recurrence needed: the delta rule walked
against a scalar f64 reimplementation of the update equations at head dim 2 over three
tokens, the gated-norm ordering pinned with a non-uniform weight and opposite-sign gate
factors, and conv-state continuity checked by feeding seven tokens one at a time versus
one batch. Six corrections came out of the ground-truth pass: no fused `ssm_ba` (two
separate tensors), `ssm_conv1d.weight` is 2-D not 3-D, `full_attention_interval` exists
and should be read, `value_length` exists and should be asserted square, neither
`eot_token_id` nor `eom_token_id` exists so the second stop id is a named constant, and
the L2 norm is ggml's clamp form rather than HF's rsqrt form. The tiled-versus-
interleaved k-head broadcast — the one assumption that could have silently corrupted
every DeltaNet layer — came back tiled, as the cheat sheet said, traced to
`ggml_compute_forward_repeat_f32`'s destination index.

One finding is a real regression rather than a correction: **the vendored flash kernel
is unreachable on this architecture.** `flash.metal` is compiled at `BD == 128` and
Qwen 3.6 is head dim 256, so prefill falls back to candle's sdpa with a materialized
`[1, n_head, seq, k_seq]` f16 mask — precisely the allocation laguna's flash path was
written to avoid. Correct but slower, and now a ledger item alongside the deleted
attention benches and the DeltaNet rollback trail's ~1 GB verify-walk cost.

**Verdict.** The engine computes Qwen 3.6. Nothing has been run against real weights
yet, and it should not be mistaken for validated: the fallback gate (reference unit
tests plus a greedy eyeball) has only had its first half satisfied, and prefill through
a per-token sequential scan will make that first real load slow enough to be
mistaken for a hang. The next honest checkpoint is P7's parity harness; until it lands,
every claim here rests on cited source lines and hand-computed tests, not on a number
measured against llama.cpp.

## 2026-07-28 — Fork bootstrap: laguna mapped, Qwen 3.6 architecture pinned down, mechanical fork started

**Context.** xwen is a manual fork of ../laguna (crate `maxuna`, ~72k lines: candle+Metal
GGUF inference engine for poolside Laguna S 2.1) retargeted at Qwen3.6-27B and
Qwen3.6-35B-A3B. Bootstrap ran as five parallel agent workstreams: laguna codebase map,
Qwen 3.6 architecture research, GGUF header survey (range-request parsing of ggml-org
files, no downloads), llama.cpp reference extraction, and the cp-based mechanical fork.

**Findings that set the design.** Qwen 3.6 is not Qwen3 — it is the Qwen3-Next-derived
hybrid: 3 gated-DeltaNet linear-attention layers per full-attention layer (full at
indices 3,7,11,…), sigmoid-gated attention output fused into a double-width q_proj
(per-head interleaved [q_h, gate_h]), QK-RMSNorm over head_dim 256, partial RoPE (n_rot
64 of 256, theta 1e7; IMROPE in llama.cpp, but provably identical to NEoX rope for
text-only), MoE with 256 experts / top-8 / softmax-then-renorm plus a sigmoid-gated
shared expert (35B), MTP head shipped as sidecar GGUFs. candle has zero support for any
of this — the DeltaNet recurrence is new code. Full config/tensor tables live in
CLAUDE.md's cheat sheet; conversion traps (norm +1 baking, tiled V-heads, ssm_a =
-exp(A_log), no ffn_norm, no ssm_in) are recorded in decisions.md "Weights and loading".

**The dflash reversal.** The plan dropped dflash.rs as Laguna-specific (a diffusion
drafter bound to a poolside checkpoint). The GGUF survey then found ggml-org publishes
DFlash sidecar drafters for both Qwen 3.6 models under the same `dflash` architecture —
the subsystem is portable after all. The removal was cancelled mid-flight; the fork
keeps dflash.rs and all drafter wiring, with Qwen adaptation tracked as its own TODO
item. Lesson, same one laguna's decisions.md preaches: check the artifact before
deleting the code that consumes it.

**Fork state.** Mechanical fork (cp-based copy, maxuna→xwen rename, MAXUNA_*→XWEN_* env
prefix, Qwen tokenizer/template vendored into reference/) running as this entry is
written; build gate is `cargo build` on the unmodified-logic tree. Docs
(this file, decisions.md, parity.md, CLAUDE.md, TODO.md) written fresh, mirroring
laguna's documentation system.

**Verdict.** Research phase complete with high confidence on every load-bearing fact
(all numbers read from shipped GGUF headers, llama.cpp master source, and live HF
repos, not from memory or blogs). Implementation fan-out next: config/loader, DeltaNet
reference, attention/MoE adaptation, chat.rs, parity harness — in that order.
