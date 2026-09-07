# Dense Qwen3-4B, arc by arc

The multi-arc write-up for dense Qwen3-4B (`model_type: qwen3`, HF BF16 safetensors) as
a full LM checkpoint and as the text-conditioning encoder for Z-Image-Turbo and the
diffusion transformers that follow. One section per arc, appended as each lands. The
architecture reference is [docs/qwen3-dense.md](../qwen3-dense.md), the encoder role is
[docs/zimage.md](../zimage.md), and the decisions are in
[docs/decisions.md](../decisions.md) under the topics they belong to.

## Arc 0, 2026-09-06: the CPU-only foundation

Four commits (ed713ba, 760647c, b36f816, f0e0cc2). Nothing runs the graph, on purpose:
the arc plan registers the three checkpoints with `servable()` and `auto_fetch()` false
and flips each gate in the arc whose surface makes it true, so nothing is listed or
fetchable before it works. Everything below is CPU-only and `cargo test` green without a
GPU.

**Why this checkpoint at all.** Two reasons at once, and they pull in the same
direction. The user wants full inference on the dense 4B (generate, chat, serve, batch),
and the diffusion image models to come need a text encoder, which for Z-Image is exactly
this model called in-process for a hidden state. The design target is unchanged:
Qwen3-4B is not a tok/s target and nothing in this work is allowed to cost the shipped
checkpoints anything. The amendment is recorded in decisions.md "Dense Qwen3-4B is a
full checkpoint AND the conditioning encoder".

**The config is explicit in two steps.** `HfQwen3Config` deserializes `config.json` and
validates it; `Qwen3Config` is built from that plus two fields the file does not carry,
`NormVariant::Standard` and `RopeSpec`, neither of which has a `Default` impl. Beyond
the planned checks the loader also refuses a non-silu activation, zero dimensions and
non-positive `rope_theta`/`rms_norm_eps`: a different activation runs to completion and
returns wrong numbers, which is the class of failure this whole two-step exists to
prevent.

**The loader validates before it allocates.** `Qwen3Set::open` is CPU-only and takes no
device. It checks shard membership, duplicate names, duplicate index keys, stray
`*.safetensors` files, the shape table and BF16-only dtype, then runs two scans over the
raw mapping. `TensorSet` is consume-once. Candle's `MmapedSafetensors::multi` turned out
to be unusable for the membership checks, since it collapses a duplicated name to the
last shard and `tensors()` loses shard provenance, so the loader opens one instance per
shard and owns its own routing map. Same mmap cost, same `load` path.

**The integrity scan is the piece with teeth.** It refuses any projection carrying a
zero run longer than 4096 elements unless the registry entry allowlists that tensor.
It was written against a real defect: Z-Image's shard 3 ships `model.layers.35.mlp
.up_proj.weight` with 14,772,816 contiguous zeros from element 27,003 and `down_proj`
with 3,938,425 from element 20,930,265, where base Qwen3-4B has none. Full numbers and
the torn-write argument are in [docs/zimage.md](../zimage.md). The f16-range scan is the
other half: 10,917 of 4,022,272,000 BF16 projection values below f16's subnormal floor
on the base set and 10,876 on Z-Image's, none above 65504, both reproduced
independently with a bun script reading the same planes. So nothing overflows the tensor
gemm's half staging, and 1.1e-5 of the weight flushes there and does not in the gemv.

**Tokenizer specials became per-instance data.** Nine ids resolved by token text at
`from_inner`, with a load error naming any that is missing, and eleven call sites in
`generate.rs`, `batch.rs`, `constrain.rs` and the serve tree now read the instance
rather than a constant. `ConstraintFactory::new` takes tokenizer bytes, the EOG ids and
the logit width, so a second vocabulary can build its own trie. `TOKENIZATION_RULES
_VERSION` stays at 3, deliberately: the shipped dialects render every existing
conversation identically, and a bump would invalidate every disk-tier image for nothing.

**Two chat dialects, byte-verified.** `Qwen3` (hybrid thinking) and `Qwen3Instruct` (no
thinking at all), both templates vendored byte-exact under `reference/`. All 16 fixture
literals are byte-equal to `llama-server --jinja /apply-template`, the Z-Image
single-turn prompt among them. The base Qwen3 template differs from 3.6's in more places
than the thinking tail: it trims no message body, it writes a reasoning block only where
there is reasoning to write, and it renders three conversations that the 3.6 lineage
refuses outright.

**Three registry entries and one resolved checkpoint source.** `Qwen34B`,
`Qwen34BInstruct2507` and `ZImageTurboEncoder`, all with `servable()` and `auto_fetch()`
false, no drafter, `VocabFamily::Qwen3`, and sampling read off each model card rather
than inherited. `Checkpoint` gained a `Format`, so `files`/`size`/`ensure_model` stay
generic over GGUF and safetensors. `CheckpointSource` is now the one place a checkpoint
gets opened, and `Generator::load`, `XwenModel::load`, serve's `read_config`, the disk
tier's checkpoint id, serve startup, `one_shot_checkpoint` and `inspect` all go through
it. The drafter open stays GGUF.

Two things that fell out of touching every MODELS-iterating path and are worth keeping.
`recommended_presence_penalty` had a `_` arm that would have silently handed the three
new entries the 3.6 family's 1.5; it is exhaustive per checkpoint now. And
`snapshot_bytes` computed `conv_kernel - 1` on a usize, which is a debug panic the
moment a checkpoint has no conv at all.

**A sizing assumption caught by its own test.** "Small model, small cache" is false
here: Qwen3-4B holds 144 KiB of KV per token against the 27B's 64 KiB, because being a
hybrid is what makes the 27B cheap, only 16 of its 64 layers holding KV where all 36 of
these do. An estimate made the other way would have been wrong by more than 2x.

**The Stage 1 oracle exists and is qualified only on CPU.**
`scripts/llama-logits-all.cpp` decodes an id file with logits requested at every
position and streams raw f32 `[n_tokens, n_vocab]` to disk with a JSON sidecar recording
backend, GPU layers, KV types, batch geometry, flash-attn, threads, the GGUF sha256 and
the llama.cpp commit. Neither existing oracle gives this: eval-callback dumps
activations and computes logits for the last position only, and `llama-perplexity
--kl-divergence-base` is uint16-compressed with a 16-logit floor. Chunked decode is
ubatch-invariant, checked bitwise at `--batch 4` against the default. The 20 fixture
prompts dumped on CPU in 55.6 s.

The finding that keeps the bar open: llama.cpp's CPU path narrows F32 activations to
BF16 before every BF16 matmul (`ggml-cpu.c` type traits, and the llamafile fast path
declines on ARM so the narrowing always runs), while the Metal path keeps F32
activations, which is xwen's arithmetic. A CPU reference is therefore not the same
arithmetic as the candidate, and the 2e-2 max-abs bar is not committed to until the same
binary has produced the Metal arm. The tool takes `--n-gpu-layers` for exactly that; the
GPU was busy this arc.

**The Stage 2 encoder reference landed early**, the arc plan having put it in Arc 2. It
needs no GPU and no xwen stack, so it was produced alongside the rest:
`scripts/zimage-ref-dump.py` under `uv`, torch 2.14.0 and transformers 5.16.1 on the CPU,
twelve prompts dumped at fp32 (the acceptance reference, `eager` attention) and at bf16
(what the pipeline executes, `sdpa`), with fixtures committed under
`tests/fixtures/zimage-encoder/` and the 50 MB of arrays deliberately not committed.

It settled three things by measurement. The `hidden_states` index convention is proven
with forward hooks rather than assumed, on every prompt in both dtypes, and the proof the
plan proposed for it is WRONG: `hidden_states[36] == norm(hidden_states[35])` is false by
12.15 max absolute, because index 36 is the norm of layer 35's output and not the norm of
index 35. The right assertions hold bitwise. Padded-to-512-with-mask against unpadded
batch-1 is bitwise equal, max absolute difference 0.0, so xwen may run unpadded. And
fp32 `sdpa` against fp32 `eager` differs by 5.49e-4 on magnitudes up to 1.4e4, about 4e-8
relative, so the attention kernel is irrelevant at fp32 and the reference does not
constrain how xwen computes it. The two findings that are not bookkeeping are below.

### Verified this arc

CPU-only, no GPU, no model run beyond the oracle's own CPU decode:

- `cargo test --lib` over the touched modules: 46 qwen3 tests, 132 across
  hub/config/checkpoint/sampler/qwen3, 598 across serve/batch/chat/tokenizer/constrain/
  drafter, 13 in the `xwen` binary. `cargo test --no-run` compiles every target;
  `cargo fmt --check` clean.
- `bun scripts/verify_chat_template_qwen3.ts`: 16/16 byte-exact against llama-server.
- `bun scripts/qwen3-fixtures.ts`: 20 prompts, ids from `llama-tokenize --ids --no-bos`
  on `Qwen3-4B-BF16.gguf`, longest 3890 tokens, and a round-trip test asserting xwen's
  ids equal the oracle's.
- `xwen inspect`, `xwen fetch` and the unknown-model message on all three entries, plus
  the refusal of the Z-Image directory when no entry names it.

### Findings recorded rather than fixed

**NFC normalization diverges from llama.cpp.** Every Qwen `tokenizer.json`, 3.6's and
Qwen3's alike, declares an NFC normalizer. The HF runtime applies it and llama.cpp
implements no normalizers at all, so on text that is not already NFC the two disagree:
`e` + U+0301 is one id under xwen and two under the oracle. This is pre-existing and
affects the shipped checkpoints identically, and xwen is on the side the model was
trained on. The consequence for callers is that `encode` is not injective over
canonically equivalent spellings and `decode(encode(text))` returns the NFC form. Pinned
by a test on the embedded vocabulary, and the fixture generator refuses a non-NFC
prompt. Recorded in decisions.md "The HF tokenizer normalizes to NFC".

**The bf16 reference sits outside the planned Stage 2 bars.** The dump reports per-token
minimum cosine 0.99960 and maximum relative error 0.03236 between the bf16 arm, which is
what the pipeline executes, and the fp32 arm, which the plan grades xwen against at
0.9999 and 1e-2. So the acceptance bar is tighter than diffusers' own arithmetic. That is
not automatically a reason to loosen it, since xwen keeps F32 activations against BF16
weights and may clear 0.9999 outright; the decision is needed only if it lands between
the two, and these are the numbers for it.

**Position 0 is a massive activation and it leads the relative-error metric.** Token 0 is
`<|im_start|>` in every prompt and depends on nothing else under causal attention, so
that row is bitwise identical across all twelve prompts, with a maximum magnitude of
13,753.5 against 150 to 380 elsewhere. bf16's ulp there is 64, which is why seven prompts
report exactly 0.01814 as their worst token and why it is the same token every time. The
Rust test reports position 0 separately for that reason. Detail in
[docs/zimage.md](../zimage.md).

**`serve::unknown_model_message` now lists names a request cannot select.** It
enumerates every entry in `MODELS`, so a client given a 400 is told `Qwen3-4B` is valid
and then gets a 400 for that too. It was already true of an uncached Flash-Next; the
three unservable entries make it unconditional. The fix is one line filtering by
`checkpoint_selectable`, and it belongs with whichever arc next touches that file.

### Not taken now

Each of these was in scope, was considered, and is deliberately not planned. None of
them carries a number or a waiting user, so none is a ledger item; the reopen condition
is what makes them findable.

- **No-copy aliasing of the safetensors payload.** Shards 1 and 2 of every set start
  their data at `% 16 == 8`, and both `gguf::dense_alias_tensor` and `ops::matmul_bf16`
  require 16-byte alignment, so the loader copies into device buffers instead. Reopen if
  load time ever matters: the copy is roughly seconds for 8 GB, and the natural fix
  would be an alignment-aware alias rather than a repack.
- **A verified-once bypass for the 8 GB scan.** `Qwen3Set::open` scans every projection
  on every open, about 1.4 s in the dev profile, and every routed metadata-only caller
  now pays it, including `serve::read_config` and `disk_tier::checkpoint_id`, which used
  to read a header and stop. Harmless while nothing serves these entries. Reopen when
  `servable()` flips: the shape is a per-`checkpoint_id` "already verified" cache.
- **An HTTP route for `encode`.** The library API and `xwen encode-text` cover the
  in-process callers, which is what the diffusion pipelines will be. Reopen when a
  pipeline needs the encoder over the wire.
- **Tool calling on the Qwen3 dialects.** Refused with a named error rather than
  half-rendered. Three things differ from 3.6 at once: the header prose, the placement
  of the client's system content, and decisively the call format, which is JSON inside
  `<tool_call>` where 3.6 writes `<function=NAME><parameter=KEY>`. The serve parser
  reads only the latter, so rendering the Qwen3 header would produce calls the engine
  cannot read back. Reopen when someone wants tools on a 4B: the work is a renderer arm
  plus a second parser dialect in `serve::engine`.
- **The GGUF form of `qwen3`.** Safetensors is the form this architecture ships in here,
  and a `qwen3` GGUF is refused with a message naming the safetensors directory. Reopen
  if someone needs a quantized 4B; note that the substring rule in `Model::identify` is
  dead code for this architecture today and the `Qwen3-4B` inside `Qwen3-4B-Instruct
  -2507` ambiguity would have to be handled at the same time.
- **Assistant prefill on the new dialects.** `section_event` recognizes only the closing
  `</think>` marker, and a thinking-on Qwen3 prompt opens no block, so the model's own
  reasoning would be classified as ordinary text and the serve reasoning channel would
  stay empty. Making the scanner flip on `think_open` would also change 3.6 behaviour,
  so it was not done blind. Reopen with the Arc 1 generate and serve integration, which
  is where the behaviour first becomes observable.
- **`preserve_thinking` is inert rather than a 400 on the Qwen3 dialects.** The template
  has no such parameter, so the renderer ignores it. The `reasoning_effort` treatment,
  refusing a field the template would ignore, is the alternative. Reopen if a client is
  confused by a field that is accepted and does nothing.
- **The HF-cache tests self-skip.** Several tests read the real checkpoints and print a
  reason when the cache lacks them, rather than failing. Reopen if a green run ever
  needs to mean those paths were exercised; the fix is a marker file or an environment
  variable that turns a skip into a failure.

### Next

Arc 1 is the layer stack, `load_qwen3`, the `LmHead` split, `encode`, `encode-text` and
generate/chat, then the oracle qualification run on Metal and the Stage 1 and
decode-consistency tests. Its entry points are `src/qwen3/stack.rs` (new) and
`XwenModel::load`'s SafeTensors arm, which today returns `unimplemented_stack()`. Two
prerequisites are not negotiable and neither is a docs step: the shared code that arc
moves (`LmHead` and the stack dispatch) is on the path of every GGUF checkpoint, so
`cargo test --release` and the parity gate on the 27B and the 35B plus the Flash-Next
replay run before it ships, not after. The unpriced risks it carries are
`ops::flash_attn`'s first production use, f16 KV against an fp32 reference on long
prompts, and the gemv-versus-gemm asymmetry the f16-range scan measured but did not
price.

## Arc 1, 2026-09-07: the stack runs, and the Stage 1 bar turns out to be the open question

Five commits (1761dab, 5bbe15a, 70bdb58, feba8c9, 5589f3c). The graph runs on the GPU:
`generate` and `chat` work on the two LM entries, `encode-text` on all three, `serve` and
`batch` still refuse. Both parity stages have been executed, and Stage 2 passes outright. Stage 1 passes on top-5 and fails a max-abs
bar that, as it turns out, llama.cpp cannot meet against itself. That last measurement is
the substance of this arc and it ends in a decision the owner has to make.

Machine state for every figure below: `pmset -g` reads `lowpowermode 2`. No high-power
claim is made.

**The stack.** `src/qwen3/stack.rs` runs the dense graph through `XwenModel` the way the
Flash-Next stack does, with `run_stack` short-circuiting into it. Per layer: rms_norm,
`matmul_bf16` on the BF16 projections as stored, per-head QK-norm then rope (full 128,
NEoX, K stored f16), `LayerCache::Full` append, `flash_attn` above one token and the f16
vector sdpa at one, o_proj, then gate/up, `silu_mul`, down. `XWEN_QWEN3_ATTN=sdpa` is a
bit-exact materialized-mask fallback for bisecting, resolved at load so one process can
hold both arms. Kernel contracts are asserted once at load rather than per call. This is
`ops::flash_attn`'s first production caller.

`LmHead` became an enum, `Quant(QLinear)` for the GGUF checkpoints and `Bf16(Tensor)` for
the tied embedding here, with a bf16 gemv arm in `lm_head_row`. That is the change with
blast radius on every shipped checkpoint, which is why the GGUF parity gate is a
precondition for this arc and not a formality.

**`encode` and `encode-text`.** `XwenModel::encode(ids, n_layers)` returns `[T, hidden]`
bf16 under the HF `hidden_states` index semantics and runs only the layers it needs. The
index semantics are not merely tested, they are exact: on the real checkpoint,
`encode(35)` against the `l_out-34` tap reads max absolute difference **0**, and
`encode(36)` against `final_norm(l_out-35)` also **0**. `xwen encode-text` renders with
the checkpoint's own dialect and tokenizer, truncates to the entry's 512, caps `--layer`
at the corrupt plane, and writes `hidden` plus `input_ids`.

**Thinking became model-opened.** The Arc 0 record left the gap open: a thinking-on Qwen3
prompt opens no `<think>`, so the model writes its own and the old scanner, which knew
only the closing marker, would have filed the whole reply as answer text. A prompt's
reasoning state is now `chat::ThinkingEntry` rather than a bool, the decode loop enters
thinking on a think opener that is the reply's first tagged token, the think budget holds
until that opener, and `--min-think` is gated on being inside a block. A review round
found the first version had quietly changed the raw-text loops for the shipped
checkpoints; marker retention is now an explicit per-call policy with two tests pinning
both halves, and the 16/16 llama-server renders still hold.

### Stage 2 passes, comfortably

`tests/qwen3_encoder.rs` against the fp32 torch reference, 12 prompts, rendered templates
and ids byte-equal on all twelve before any number is compared:

| | xwen | bar | the pipeline's own bf16 |
| --- | --- | --- | --- |
| min cosine, positions >= 1 | 0.99999449 | >= 0.9999 | 0.99960 |
| max relative error, positions >= 1 | 0.00388 | <= 1e-2 | 0.03236 |
| position 0 cosine | 0.99999955 | >= 0.9999 | |
| position 0 relative error | 0.00089 | <= 1e-2 | |

So xwen sits about an order of magnitude closer to the fp32 reference than the arithmetic
diffusers actually ships. Read the relative-error column against one more measurement:
rounding the fp32 reference itself to bf16 and back scores 0.003784, so the bf16 return
type alone spends 38% of that budget and the bar has 2.6x headroom over pure output
quantization, not 100x. A future result between 0.004 and 0.01 should account for the
output cast first, and comparing against an f32 encode output is how to separate the graph
from the cast.

### Stage 1 passes on top-5 and fails max-abs, and the failure is not xwen's

The gate ran against both oracle arms over all 20 prompts, 6307 positions:

| oracle arm | pooled max-abs | argmax | pooled top-5 |
| --- | --- | --- | --- |
| llama.cpp CPU | 0.379 | 6300/6307 | 99.9176% |
| llama.cpp Metal | 0.222 | 6304/6307 | 99.9239% |
| llama.cpp CPU vs its own Metal | 0.358 | 6303/6307 | 99.9239% |

Every argmax flip on every arm falls inside the 2e-2 near-tie band. The third row is the
decisive one: **xwen is closer to the Metal oracle than the two oracle backends are to
each other.** A 2e-2 max-abs bar with 100% argmax agreement is therefore not a bar
llama.cpp meets against itself on these prompts, and holding xwen to it would be holding
it to a standard the reference does not have.

Two ablations say the error is not where one would first look. `XWEN_QWEN3_ATTN=sdpa`
produces numbers identical to the flash arm on six prompts, so the flash kernel, this
being its first production use, is not the source. And `XWEN_ATTN_MM_CLASSIC=1` is
**worse**, pooled 0.337 against 0.222 and 0.217 against 0.073 on the corpus-middle
prompt, so the tensor gemm is more accurate here than the classic chain it replaces,
which is the same direction the dense-FFN gemm went and the opposite of the `dense_mm`
case.

Error does not grow with position, which is the shape a cache or rope bug would have. Per
prompt the max-abs runs 1.8e-5 at 1 token, 6.3e-3 at 8, 9.6e-3 at 16, 2.5e-2 at 53,
7.3e-2 at 199, 5.3e-2 at 610 and 0.222 at 3890; within the 3890-token prompt the failing
fraction is about 7% in every position bucket with a median of 2.3e-2, and the outliers
(0.22 at position 853, 0.21 at 1084) are isolated positions rather than a tail.

**The decode-consistency test is in the same position.** `tests/qwen3_consistency.rs`
compares chunk-1 decode against a single-pass prefill and reaches max absolute logit
difference 2.59e-2 at one position of the 53-token prompt, over the same 2e-2. The other
chunkings are not tabulated because the test stops at the first failure. This one has no
oracle in it at all, so it is a statement about xwen's own partition-dependence, and the
repo already has a decision on that shape of fact: persistent state is partition-dependent
in its low bits and that is accepted rather than denied (decisions.md "Persistent state is
partition-dependent"). Whether 2e-2 is the right number for a full-vocabulary logit row is
part of the same open question.

### The open decision, stated as open

**Nobody has decided what the Stage 1 bars should be, and this record does not decide it.**
What is measured is above. The recommendation to react to, not a settled policy:

- gate on pooled top-5 >= 99.9%, which both arms clear;
- gate argmax as "no flip outside the near-tie band" rather than 100% agreement, since
  every flip observed on every arm, llama.cpp's own two backends included, is inside it;
- report max-abs against the oracle's own CPU-versus-Metal spread rather than a fixed
  2e-2, because that spread is the resolution the reference itself has.

The counter-argument deserves stating too: a bar derived from the reference's internal
disagreement can only ever certify "as close as llama.cpp is to itself", which is weaker
than the fixed bar the plan wanted and would not catch a systematic error smaller than
0.358. Ledgered as an owner decision under Parity, provenance and tooling.

### Verified this arc

- Stack unit tests on a tiny 4-Q-head / 2-KV-head model, release: shapes, the attention
  switch, chunked-versus-single-pass, the sdpa arm against the fused arm, and the encode
  index semantics. All pass.
- Stage 1 against both oracle arms and Stage 2 against the torch reference, as tabulated.
- `encode-text` on the encoder entry: 23 tokens in, `hidden [23, 2560]` bf16 out, 3.7 s
  including load (0.5 s load, 7.5 GB of weights on the device, 8.6 GB resident at
  `max_ctx` 40960).
- `generate` on both LMs: qwen3-4b with thinking on opens its own `<think>` and reasons
  through a haiku prompt; Instruct-2507 answers cleanly. Load 0.4 s and 0.8 s to first
  token.
- Track A's tap table is implemented and source-verified rather than predicted:
  `scripts/parity.ts --arch qwen3` maps our taps onto `qwen3.cpp`'s as the identity plus
  `kqv_out-{il}`, and skips `attn_o_proj`. The deciding fact is that llama.cpp's
  `cb(cur, "kqv_out", il)` fires on the output of `build_attn_mha`, BEFORE `wo`, and the
  only cb after `wo` is commented out, so there is no post-o_proj node to compare against.
- **The GGUF parity gate and the Flash-Next replay both PASS**, from a pinned checkout of
  5589f3c under /tmp. This was the precondition, not a formality: `LmHead` became an enum
  and `run_stack` gained a dispatch, both on the path of every shipped checkpoint, so a
  regression here would have been in the 35B and the 27B rather than in anything qwen3.

| gate | strict | mm | decode | perplexity |
| --- | --- | --- | --- | --- |
| 35B-A3B, 6 tiers graded | cos 1.000000, top-5 5/5 | cos 0.999618 | 63/64, 62/64, 61/64 agree, 1/2/3 excused, 0 mismatch | Δnll 0.001179 |
| 27B | cos 1.000000 | cos 1.000000 | 64/64 on all three fixtures, 0 excused | Δnll 0.000243 |

  Flash-Next has no harness, so it takes the forced replay instead
  (`--control XWEN_PLE_TAIL_CLASSIC=1`, oracle reused at pin `6fe7498`): code-short 62/64
  with 2 excused, text-mixed 64/64, long-mixed 59/64 with 5 excused, **zero hard
  mismatches** on all three, all passed. The 27B's 64/64 with nothing excused is the
  cleanest of the set and the most direct statement that the dense path is untouched.

### Not taken now, added this arc

- **An f32 encode output alongside the bf16 one.** The bf16 cast spends 38% of the Stage 2
  relative-error budget. Reopen if a Stage 2 result ever lands between 0.004 and 0.01, at
  which point separating the graph from the cast is the first diagnostic.
- **Tabulating every chunking in the consistency test.** It stops at the first failure, so
  chunk 1 is the only arm measured. Reopen when the bar is decided, since the bar is what
  decides whether the other chunkings are failures at all.

### Next

Arc 2 is the Instruct-2507 smoke recorded properly and whatever the bar decision implies
for the two gates. Arc 3 is `serve` and `batch`: the per-target tokenizer and grammar
factory (D6c), the Target mapping, the disk tier on a safetensors `checkpoint_id`, and the
`servable()` flip. The 8 GB scan on every `Qwen3Set::open` becomes a real cost there and
is the first thing to fix in that arc; `encode-text` and `logits-dump` already pay it.
