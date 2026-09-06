# Z-Image and the text-conditioning encoder

Z-Image-Turbo is a flow-matching image model whose text conditioning is a dense
Qwen3-4B. This file is the reading of the diffusers pipeline that xwen has to match, the
checkpoint facts that came out of that reading, and the reference dump that grades it.
The architecture itself is in [qwen3-dense.md](qwen3-dense.md).

Everything below was fetched from primary sources on 2026-09-06 and, where it is a claim
about the shipped files, verified against the downloaded bytes. The diffusers source is
`src/diffusers/pipelines/z_image/pipeline_z_image.py` on main
(https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/z_image/pipeline_z_image.py);
the transformers sources are `models/qwen3/modeling_qwen3.py` and
`utils/output_capturing.py` on main.

## What the pipeline does

`ZImagePipeline._encode_prompt`, in full, is a chat render, a padded tokenize, one
forward, and a slice:

```python
        for i, prompt_item in enumerate(prompt):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt[i] = prompt_item

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        embeddings_list = []

        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list
```

**The render.** One user turn, no system message, nothing around it. Z-Image's chat
template is byte-identical to `Qwen/Qwen3-4B`'s, and with `tools` unset,
`add_generation_prompt=True` and `enable_thinking=True` it produces exactly

```
<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n
```

and nothing more. In particular no `<think>` token is emitted at all: the template's
`{%- if enable_thinking is defined and enable_thinking is false %}` branch, which would
append `<think>\n\n</think>\n\n`, does not fire when thinking is on. The render ends
inside an open assistant turn. `add_bos_token` is false and `bos_token` is null, so
there is no BOS.

**The tokenize.** `max_length` defaults to 512 on `__call__`, `encode_prompt` and
`_encode_prompt` alike. Padding is to `max_length` with `<|endoftext|>` 151643, and
`tokenizer_config.json` sets no `padding_side`, so transformers' default right padding
applies.

**The slice.** The padded batch goes through the encoder with its mask and each row is
then cut back to its real tokens, so the return value is a list of variable-length
`[n_tokens, 2560]` tensors, not a padded batch.

**Why xwen can ignore the padding.** Padding is on the right and attention is causal, so
no padded position can influence a retained one. An unpadded batch-1 forward of length
`n` is mathematically identical to padding to 512 and slicing. The reference dump checks
that claim rather than trusting it: on the 73-token prompt the padded and unpadded
hidden states came back **bitwise equal**, max absolute difference 0. What 512 still
controls is truncation, and that xwen must reproduce.

**No negative prompt on Turbo.** `encode_prompt` encodes an empty-string negative prompt
only when `do_classifier_free_guidance` holds, which is `guidance_scale > 0`; the
pipeline docstring's own Turbo usage passes `guidance_scale=0.0`. So Turbo as documented
runs one text-encoder pass per prompt and no second one.

**Dtype.** The pipeline never casts the encoder. `text_encoder/config.json` says
`bfloat16`, the shard tensors are BF16, and the docstring loads the pipeline
`torch_dtype=torch.bfloat16`. The executed reference numerics are bf16 weights with bf16
activations, with HF's `Qwen3RMSNorm` upcasting to fp32 internally and casting back
before the weight multiply.

## `hidden_states[-2]` is index 35, and why

Current transformers collects hidden states through forward hooks, not an in-loop
append. `Qwen3PreTrainedModel` declares `_can_record_outputs = {"hidden_states":
Qwen3DecoderLayer, ...}` and `Qwen3Model.forward` is decorated `@capture_outputs`. The
hook prepends the input of layer 0, one entry is appended per decoder layer, and
`capture_outputs` with its default `tie_last_hidden_states=True` then replaces the last
entry with `outputs.last_hidden_state`, which is post-`model.norm`. For 36 layers the
tuple therefore has 37 entries:

| index | contents |
| --- | --- |
| 0 | `inputs_embeds`, the embedding lookup, no norm |
| 1..35 | output of `model.layers[i-1]`, post-residual, no norm |
| 36 = `[-1]` | `model.norm(output of model.layers[35])` |

So `[-2]` is index 35, the output of `model.layers[34]`: run layers 0 through 34
inclusive, take the residual stream, apply no final norm. `model.layers[35]`,
`model.norm` and the LM head are never used by Z-Image.

Two independent confirmations. The reference dump proves the index with forward hooks
instead of assuming it. And the shipped weights corroborate it by accident, below.

## The shard-3 corruption

Z-Image's `text_encoder/` is a copy of `Qwen/Qwen3-4B` base, not Instruct-2507:
`config.json` and `generation_config.json` are byte-identical to base's, the tokenizer
files match, and weight shards 1 and 2 have the same LFS sha256 as base's. Shard 3 does
not, and comparing the two files tensor by tensor shows why. The 552-byte headers are
identical. Three of the five tensors are identical. Two are not:

| tensor | zero elements | runs | first zero | last zero | total elements |
| --- | --- | --- | --- | --- | --- |
| `model.layers.35.mlp.up_proj.weight` | 14,772,816 | 1 | 27,003 | 14,799,818 | 24,903,680 |
| `model.layers.35.mlp.down_proj.weight` | 3,938,425 | 1 | 20,930,265 | 24,868,689 | 24,903,680 |

In every differing position the base model has a normal weight and Z-Image has an exact
zero. `Qwen/Qwen3-4B` has no zero run in either tensor. One contiguous zero-filled byte
range in each is the signature of a torn or truncated write, not of pruning: the zeroed
set is not magnitude-selected (the largest zeroed weight in `up_proj` is 0.855, which is
that tensor's global maximum) and it is not row- or column-structured (5770 of 9728
`up_proj` rows are fully zero, 3956 fully nonzero, with one partial row at each end of
the run).

It is harmless for the encoder role, because index 35 never evaluates layer 35's MLP,
and that is the accidental confirmation of the index reading: Z-Image produces correct
images with those planes zeroed. It does mean the Z-Image copy is **not a faithful full
LM**, which is why its registry entry is encode-only.

xwen's response is the loader's integrity scan. Any projection with a zero run longer
than 4096 elements is a load error unless the registry entry allowlists that exact
tensor; `ZImageTurboEncoder` allowlists those two and refuses the layer index that would
evaluate them. Pointed at the same directory without naming the entry, `xwen inspect`
refuses it:

```
Error: reading the checkpoint directory .../text_encoder
Caused by: model.layers.35.mlp.up_proj.weight in model-00003-of-00003.safetensors
           holds 14772816 consecutive zero elements starting at element 27003;
           that plane is corrupt or was never written
```

The corruption is tolerated only when someone names the entry that documents it. If a
faithful full 4B LM is what is wanted, use `Qwen/Qwen3-4B` and not this copy.

## The reference dump

`scripts/zimage-ref-dump.py` is the only Python in the repo. It exists because no ONNX
export of this encoder was found and there is no bun path to torch; it runs once, by
hand, under `uv` in a throwaway venv, and never in CI. It reproduces the pipeline's
rendering exactly, runs the encoder on the CPU, proves the 37-entry index table with
hooks, and dumps two references per prompt: **fp32**, which is the acceptance reference,
and **bf16**, which is what the pipeline actually executes and is reported as a
diagnostic rather than gated.

Committed under `tests/fixtures/zimage-encoder/`: the prompts, the rendered strings and
their sha256, the token ids, and the sha256 of every dump. The arrays themselves are not
committed; `tests/qwen3_encoder.rs` reads them from a directory named by an environment
variable. Twelve prompts, 11 to 512 tokens, including one that is exactly 512 and one
that truncates from 643.

Two numbers from the 2026-09-06 dump matter beyond bookkeeping. Padded and unpadded
agree bitwise, quoted above. And **the bf16 arm sits further from fp32 than the planned
acceptance bars**: per-token minimum cosine 0.9996 to 0.9998 and maximum relative error
1.8e-2 to 3.2e-2 across the twelve prompts, against a plan that grades xwen at cosine
0.9999 and relative error 1e-2 versus fp32. A bar tighter than the distance between the
real pipeline and its own fp32 idealization is a bar about arithmetic, not about
conditioning quality. Whoever sets the Stage 2 gate in the arc that lands it should
decide deliberately which reference it grades against; the record carries this as an
open question rather than a settled bar.
