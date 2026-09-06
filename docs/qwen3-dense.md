# Dense Qwen3-4B (`qwen3`)

What the dense Qwen3 architecture is in this repo, what of it runs today, and how it
gets verified. The WHY lives in [decisions.md](decisions.md) under the topics it
touches; the arc narrative is in [records/qwen3-dense.md](records/qwen3-dense.md); the
Z-Image encoder role has its own file, [zimage.md](zimage.md).

This is a second weight format and a second vocabulary, not a second variant of the
Qwen 3.6 graph. It arrives for two reasons at once: full inference on a small dense
checkpoint, and the text-conditioning encoder that the diffusion image transformers
will call in-process. It is not a tok/s target.

## Status, 2026-09-06

**Registered, not runnable.** Arc 0 landed the CPU-only foundation: the config, the
BF16 safetensors loader with its integrity scan, per-instance tokenizer specials, two
chat dialects, three registry entries and one resolved checkpoint source. No layer
stack exists yet, so `XwenModel::load` refuses the safetensors arm with a "stack not
implemented" error, and `servable()` and `auto_fetch()` are false on all three entries.
What works today is `xwen fetch` and `xwen inspect` on each entry, plus everything CPU
side: identity, config parsing, the loader's validation and scans, the tokenizer and
the two dialects.

The arcs that follow flip the gates as each surface starts working: the layer stack,
`encode` and `generate`/`chat` in Arc 1, the torch reference and the Instruct-2507
smoke in Arc 2, `serve` and `batch` in Arc 3.

## Config

Three releases, one shape. `Qwen/Qwen3-4B` (base), `Qwen/Qwen3-4B-Instruct-2507`, and
the `text_encoder/` subdirectory of `Tongyi-MAI/Z-Image-Turbo`, whose `config.json` is a
byte-for-byte copy of the base model's.

| field | base and Z-Image | Instruct-2507 |
| --- | --- | --- |
| `model_type` | `qwen3` | `qwen3` |
| `hidden_size` | 2560 | 2560 |
| `intermediate_size` | 9728 | 9728 |
| `num_hidden_layers` | 36 | 36 |
| `num_attention_heads` | 32 | 32 |
| `num_key_value_heads` | 8 | 8 |
| `head_dim` | 128 | 128 |
| `rms_norm_eps` | 1e-6 | 1e-6 |
| `vocab_size` | 151936 | 151936 |
| `tie_word_embeddings` | true | true |
| `attention_bias` | false | false |
| `use_sliding_window` | false | false |
| `hidden_act` | silu | silu |
| `torch_dtype` | bfloat16 | bfloat16 |
| `rope_theta` | 1000000 | **5000000** |
| `max_position_embeddings` | 40960 | **262144** |

Those last two rows are the whole config diff between the releases, which is why
`rope_theta` is what tells them apart when a directory identifies as nothing else.

Derived: `q_dim` 4096, `kv_dim` 1024, all 36 layers full attention, KV 147,456 bytes per
token (144 KiB). That last figure is larger than the 27B's 64 KiB, and the reason is
worth keeping in mind when sizing anything: the 27B is a hybrid and only 16 of its 64
layers hold KV, where all 36 of these do.

Every weight set is 8,056,438,199 bytes over three shards plus config, index and
tokenizer (Instruct-2507 is one byte larger, its `config.json` being one byte longer).
The registry records `8.06 GB`.

Layer skeleton, from HF `modeling_qwen3.py` and llama.cpp `src/models/qwen3.cpp`, which
agree: `x + attn(input_layernorm(x))`, then `h + mlp(post_attention_layernorm(h))`.
Attention is `q_proj` then per-head RMSNorm over [128] then rope, the same for k, v
untouched by both; scale `1/sqrt(128)`; GQA 32/8 by `repeat_interleave`, so KV head `j`
serves Q heads `4j..4j+3`. Rope is full-width NEoX over all 128 head dims, unlike Qwen
3.6's partial 64-of-256. MLP is SwiGLU. There are no biases and no output gate.

## The two fields the file does not carry

`Qwen3Config` is built from the validated `HfQwen3Config` plus two fields that the
architecture definition supplies and `config.json` does not name. Neither has a
`Default` impl, so neither can be assumed into existence:

- `norm: NormVariant::Standard` - the plain `x/rms*w` form, which is what
  `candle_nn::RmsNorm` computes. The Gemma-style zero-centred `(1+w)` variant exists
  nowhere in this crate, because the GGUF path receives its norms pre-baked. Pinning
  the assumption is the point: a future `ZeroCentred` arm would have to be implemented,
  never defaulted into.
- `rope: RopeSpec { head_dim: 128, rotary_dim: 128, theta }` - `rotary_dim == head_dim`
  says full rope in a repo whose other architecture rotates 64 of 256, and the theta
  comes from the file.

## The layer index convention for `encode`

`XwenModel::encode(ids, n_layers)` will take `n_layers` as the HF `hidden_states` index,
so that a number read off a diffusers pipeline transfers without arithmetic:

| `n_layers` | what comes back |
| --- | --- |
| 0 | the embedding lookup, no norm |
| N, 1..35 | the residual stream after `layers[N-1]`, no norm |
| 36 (`n_layer`) | after `layers[35]` and then `output_norm` |

That mirrors transformers' own tuple, which has `num_hidden_layers + 1` entries and
applies the final norm to the last one only. Z-Image's default is 35, derived in
[zimage.md](zimage.md). The stack takes a `stop_after` so layers at or past the index
never run.

## Weights, and why they are copied

BF16 safetensors, read through the pinned candle's own `MmapedSafetensors`, one instance
per shard rather than `::multi`. The copy into device buffers is deliberate and the
shard alignment is the reason; the reasoning is in decisions.md "Dense Qwen3 arrives as
BF16 safetensors".

`Qwen3Set::open` validates on the CPU before anything is allocated: every index entry
present in exactly its named shard, no name in two shards, no duplicate index key, no
stray `*.safetensors` the index does not reference, shapes matching the expected table
for the config, BF16 and nothing else. `TensorSet` is consume-once, and `finish()`
errors listing whatever is left over. A `lm_head.weight` that ships anyway must be
byte-equal to the embedding and is then struck off the ledger without being read, so no
second copy of the 742 MiB vocabulary plane is allocated. Norm planes widen to F32 at
load because candle's Metal `rms_norm` needs the weight dtype to match the activations;
projections stay BF16.

Two scans run over the raw mapping during that pass. The integrity scan refuses any
projection carrying a zero run longer than 4096 elements unless the registry entry
allowlists that tensor, which is what catches the Z-Image shard-3 corruption
([zimage.md](zimage.md)). The f16-range scan counts values outside f16's representable
range, because the tensor-core gemm stages BF16 weights to half:

| set | below 2^-24 | above 65504 | elements |
| --- | --- | --- | --- |
| `Qwen/Qwen3-4B` | 10,917 | 0 | 4,022,272,000 |
| Z-Image `text_encoder/` | 10,876 | 0 | 4,022,272,000 |

Nothing overflows, and 1.1e-5 of the weight sits under the subnormal floor where the
gemm's half staging flushes it and the gemv keeps it. That is a small gemv-versus-gemm
asymmetry to be aware of when reading a parity number, not an overflow hazard. The
Z-Image count is 41 lower because its two zero-filled planes contribute to neither tail.

`checkpoint_id` is a `gguf::CheckpointId` chained over `config.json`, the index, and
each shard's header, so snapshots and the disk tier can key a safetensors set the way
they key a GGUF. It hashes metadata only and sums whole-file lengths, which catches a
shard whose size changed and does not catch a same-length in-place payload overwrite.

## Identity

A safetensors directory identifies by provenance first: `Official` only when the
directory is that registry entry's own cached HF snapshot, compared after canonicalizing
the directory (never a file inside it, since hub cache files are symlinks into a shared
blob store). Otherwise it is `Assumed` under its own directory name, with `rope_theta`
choosing the release: 5e6 is Instruct-2507, 1e6 is the base model, which wins the tie it
shares with the byte-identical Z-Image config. `--model-size` stays a cross-check, so
naming the wrong release for a directory is a startup error rather than an override.
There is no name inside a safetensors set to read, so the GGUF `general.name` passes are
never reached on this architecture.

A GGUF whose architecture string is `qwen3` is refused, pointing at the safetensors
directory and the three aliases. The GGUF form of this architecture is not implemented;
see the record for the reopen condition.

## Verification

The runbook lives in [parity.md](parity.md) under "The Qwen3 dense track". The bars and
the measurements as they land:

| stage | reference | bar | measured |
| --- | --- | --- | --- |
| tokenizer round trip | `llama-tokenize --ids --no-bos` on the BF16 GGUF | ids equal, `decode(ids) == text` | 20 prompts, 2026-09-06 (self-skips without the cached tokenizer) |
| chat template | `llama-server --jinja /apply-template` | byte-equal rendering | 16/16 cases, 2026-09-06 |
| Stage 1, full-vocab logits | `llama-logits-all` per-position dump | max-abs <= 2e-2, argmax 100%, pooled top-5 >= 99.9% | pending |
| decode consistency | the engine against itself | same 2e-2 bar across chunk sizes | pending |
| Stage 2, encoder hidden states | torch fp32 dump | cosine >= 0.9999, relative error <= 1e-2 | pending |

Two things about those bars are not yet settled and should not be quoted as if they
were. The Stage 1 bar of 2e-2 is provisional: llama.cpp's CPU path rounds F32
activations to BF16 before every BF16 matmul, so a CPU reference is not the arithmetic
xwen performs, and the bar is committed to only after the same oracle is run with
`--n-gpu-layers` on Metal. And pooled top-5 is pooled on purpose: per-position overlap
of five items moves in 20% steps, so 99.9% only means something summed over positions,
as `sum |top5_xwen ∩ top5_ref| / (5 × positions)`.

The Stage 2 relative error is defined per token as
`max_i |x_i − r_i| / max(max_i |r_i|, 1e-6)`, the denominator being that token's own
largest magnitude in the reference. The reference is dumped at fp32 and graded there;
the same script also dumps bf16, which is what the real pipeline executes, and that
distance is reported alongside rather than gated.
