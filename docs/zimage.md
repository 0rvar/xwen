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
that claim rather than trusting it: on the 73-token prompt the padded-with-mask and
unpadded hidden states came back **bitwise equal**, max absolute difference 0.0, which is
exact agreement and not fp32 noise. What 512 still controls is truncation, and that xwen
must reproduce.

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

**Proven, and one obvious proof of it is wrong.** The reference dump does not assume the
table: it hangs forward hooks on `layers[34]` and `layers[35]` and checks every prompt in
both dtypes. `hidden_states[35]` is bitwise the output of `layers[34]`, max absolute
difference 0.0, and `hidden_states[36]` is bitwise `norm(output of layers[35])`, also 0.0.
The tempting one-liner, `hidden_states[36] == norm(hidden_states[35])`, is **false** and
must not be written down as a check: index 36 is the norm of layer 35's output, not the
norm of index 35, and the two differ by 12.15 max absolute on the first prompt. The
conclusion the plan drew is right, the shortest proof of it is not.

The shipped weights corroborate the same reading by accident, below.

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

Committed under `tests/fixtures/zimage-encoder/`: `prompts.json` and `reference.json`,
holding the prompts, the rendered strings and their sha256, the token ids, and the file
names and sha256 of every dump, with no absolute paths. The arrays themselves are not
committed (50 MB, written to `/tmp/zimage-ref` by default); `tests/qwen3_encoder.rs`
finds them under a directory named by an environment variable, and asserts the rendered
strings and the ids with no dump present at all. Twelve prompts, 11 to 512 tokens,
including one tuned to land on exactly 512 templated tokens and one that truncates from
643. Template overhead is 8 tokens.

Run on 2026-09-06 with torch 2.14.0, transformers 5.16.1, CPU, 8 threads. That
transformers is well past the 4.51 these configs were written against, which is exactly
why the index convention was proven empirically rather than read off the version's
source. The fp32 arm runs `eager` attention for a transparent softmax and the bf16 arm
runs `sdpa`, which is what the pipeline executes; fp32 sdpa against fp32 eager differs by
5.49e-4 on a per-token magnitude up to 1.4e4, about 4e-8 relative, so the kernel choice
is irrelevant at fp32 and xwen may be graded against this reference however it computes
attention.

Two findings from that run matter beyond bookkeeping.

**The bf16 arm sits further from fp32 than the planned acceptance bars.** Per-token
minimum cosine 0.99960 and maximum relative error 0.03236 across the twelve prompts,
against a plan that grades xwen at cosine 0.9999 and relative error 1e-2 versus fp32. So
the bar is tighter than the distance between the real pipeline and its own fp32
idealization. That is not by itself a reason to loosen it: xwen keeps F32 activations
against BF16 weights, which is a different and probably closer arithmetic than torch's
all-bf16 path, and it may clear 0.9999 outright. The decision is only needed if it lands
between the two, and these numbers are the context for it.

**Position 0 is a massive activation and it dominates the relative-error metric.** Token
0 is `<|im_start|>` in every prompt and, under causal attention, its hidden state depends
on nothing else, so that row is bitwise identical across all twelve prompts. Its maximum
magnitude is 13,753.5 against 150 to 380 for every other token in the same prompt. bf16's
ulp at that magnitude is 64, so a few ulps of accumulated error reads as 1.8% relative,
which is why seven of the twelve prompts report exactly 0.01814 as their worst token and
why it is the same token every time. Any relative-error metric with a per-token
denominator will be led by this row, so `tests/qwen3_encoder.rs` reports position 0
separately; a per-prompt denominator is the other option. If xwen ever lands between the
bars, this row is the first term to look at, because one fix there moves eleven prompts.
