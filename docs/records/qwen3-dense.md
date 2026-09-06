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

**The bf16 reference sits outside the planned Stage 2 bars.** The 2026-09-06 dump
reports per-token minimum cosine 0.9996 to 0.9998 and maximum relative error 1.8e-2 to
3.2e-2 between the bf16 arm, which is what the pipeline executes, and the fp32 arm,
which the plan grades xwen against at 0.9999 and 1e-2. Whoever lands Stage 2 decides
which reference the gate uses; it is a question about what the bar means, not a
regression. Detail in [docs/zimage.md](../zimage.md).

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
