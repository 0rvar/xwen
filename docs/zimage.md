# Z-Image and the text-conditioning encoder

Z-Image-Turbo is a flow-matching image model whose text conditioning is a dense
Qwen3-4B. This file is the per-architecture reference for the whole pipeline: first the
reading of the diffusers text-conditioning path that xwen has to match, the checkpoint
facts that came out of that reading and the reference dump that grades it, then, from
"The transformer" onward, the transformer, the VAE and the scheduler as they were read
off the shipped configs and the two reference implementations, and the traps in them.
The encoder architecture itself is in [qwen3-dense.md](qwen3-dense.md), the decisions
are in [decisions/zimage.md](decisions/zimage.md), and the arc that landed the pipeline
is [records/zimage-pipeline.md](records/zimage-pipeline.md).

The encoder half was fetched from primary sources on 2026-09-06 and the pipeline half on
2026-09-07; where either is a claim about the shipped files, it is verified against the
downloaded bytes. The diffusers source is
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

## Serving

Since 2026-09-07 the pipeline is also behind `xwen serve`, as `POST
/v1/images/generations` in the OpenAI images shape, with the same handler on
`/images/generations` and `/proxy/openai/images/generations`. The route is present
whenever the OpenAI dialect is on; the encoder, the transformer and the VAE load on the
first request and leave after `--idle-unload`, on an `image-engine` thread of their own
beside the language engine. `src/serve/images.rs` is the whole route, and
[records/zimage-pipeline.md](records/zimage-pipeline.md) "Arc C" has the field-by-field
contract, the statuses and the measured times; the choices behind it are in
[decisions/zimage.md](decisions/zimage.md) "CLI first, then serve as OpenAI".

ComfyUI needs nothing installed: start ComfyUI with `--comfy-api-base
http://<this mac>:<port>` and its stock OpenAI image node POSTs to the proxy path here.
Run the server without an API key for that node, which sends none and would get a 403.
The node's `model` dropdown is accepted whatever it says, its `size` dropdown offers
1024x1024, 1024x1536, 1536x1024 and auto, and it carries no seed or step count on the
wire; the `seed` and `steps` extension fields exist for clients that do.

`serve --image-steps <N>`, or `steps` under the config file's `[image]` table, is the
step count the route renders at when a request names none. It takes 1 to 50, the same
range a request may ask for, and a value outside it is a startup error; unset, the
pipeline's own eight steps apply. A request's `steps` or `num_inference_steps` still
wins, so the setting is what serves the clients that cannot send one, the ComfyUI node
above among them.

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
idealization. That is not by itself a reason to loosen it, and as of 2026-09-07 it is
settled by measurement rather than by argument: xwen keeps F32 activations against BF16
weights, which is a different and closer arithmetic than torch's all-bf16 path, and it
clears the bar outright at minimum cosine 0.99999449 and maximum relative error 0.00388
over positions 1 and up, with position 0 at 0.99999955 and 0.00089. It is about ten times
closer to the fp32 reference than the pipeline's own bf16 execution. The question the
numbers above would have posed therefore does not arise.

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


**One more term to hold on to, for whoever reads a future Stage 2 number.** Rounding the
fp32 reference itself to bf16 and back, with no graph involved at all, scores 0.003784
relative error and 0.999996 cosine. `encode` returns bf16, so a perfect graph already
spends 38% of the 1e-2 relative budget on the output cast alone. The bar has 2.6x
headroom over pure output quantization, not 100x. Today's 0.00388 is barely above that
floor, which is the strongest available statement that the graph itself contributes
almost nothing; a future result between 0.004 and 0.01 should be read as the cast plus
something, and comparing against an f32 encode output is how to tell the two apart.

## The transformer reference dump, Stages 3 and 4

Live since 2026-09-07. `scripts/zimage-ref-dump.py --stage transformer --dtype fp32`
then `--dtype bf16` (flags `--width`, `--height`, `--prompt-idx`, `--seed`) reproduces
diffusers' `ZImagePipeline.__call__` from `prepare_latents` onwards on mps, with the two
inputs fixed by files so xwen can start from the same tensors: `latents0`, a CPU
`torch.randn` draw labelled by its seed, and `cap_feats`, the encoder's `hidden_states[-2]`
for one prompt of `prompts.json` computed fp32 on cpu as Stage 2 does and rounded once to
bf16. Both sides read those two files, so the encoder is out of the picture here and
Stage 2 stays its only gate. The fp32 arm is the reference and writes the inputs; the
bf16 arm reads them back, so the two arms differ in arithmetic alone, and it copies the
fixture set into `tests/fixtures/zimage-transformer/<WxH>-p<idx>-s<seed>/` with a
`meta.json` holding its own gap to fp32 and every file's sha256. diffusers 0.40's
`get_default_z_image_sigmas` is `linspace(1, 1/N, N)`, then the static shift, then a
terminal zero, which is the grid in "The scheduler and the Euler loop" exactly, and the
script asserts it. The whole dump, encoder included, ran in under two minutes.

`tests/zimage_parity.rs` grades xwen against it (`cargo test --release --test
zimage_parity -- --ignored --nocapture`, 13-14 s). xwen's side is `ZImagePipeline::
velocity` for the single forward and `generate` for the full run, and on the CLI the same
two inputs go in through `xwen image --latents <file> --cap-feats <file>`, with `--dump
<dir>` writing the step-0 velocity and the final latent for grading by hand.

| 512x512, prompt 1 (73 caption tokens), seed 0 | cosine | mean rel | max rel |
| --- | --- | --- | --- |
| step-0 velocity, xwen vs fp32 reference | 0.999999 | 0.0008 | 0.0045 |
| step-0 velocity, reference bf16 vs fp32 (its own spread) | 0.999560 | 0.0175 | 0.0890 |
| bracket: timestep one grid point off | 0.610825 | | |
| bracket: caption tokens reversed | 0.870767 | | |
| final latent after 8 steps, xwen vs fp32 | 0.999708 | 0.0064 | |
| final latent after 8 steps, reference bf16 vs fp32 | 0.994750 | 0.0492 | |

Image PSNR against the reference PNG: xwen 47.03 dB, the reference's own bf16 arm
32.40 dB. The reference's final latent decoded through xwen's VAE against the reference
PNG: 92.62 dB.

Those xwen rows are as of 2026-09-07 evening, with an f32 activation stream over bf16
weights. **The bf16-stream numbers they replace are history worth keeping**, because they
are what the graph reads at when its activations are bf16: step-0 velocity 0.999302 /
0.0205 / 0.1006, final latent 0.990845 / 0.0631, image PSNR 29.71 dB, brackets 0.6045 and
0.8633. Against the reference's own bf16 arm directly the bf16 stream read cosine 0.99971,
closer than to fp32, which was the shared bf16 rounding showing.

**Stage 3 is the gate**, at cosine >= 0.998 and mean relative error <= 0.04 on the step-0
velocity; **Stage 4 is reported**, the final latent and the PSNR, because eight Euler steps
compound the bf16 differences on both sides; and **the VAE alone is gated** at 60 dB,
because it decodes in f32 on both sides and 92.6 dB is a handful of pixels one level off.
The bars and their bracketing are decisions.md "Verification is a torch dump with an
injected latent". One reading mattered for a day and is now closed. On the bf16 stream
xwen's loss (1 - cosine, 7.0e-4) was about 1.6x the reference's own bf16 loss (4.4e-4),
which said the graph was right and its arithmetic a little noisier than torch's, and the
suspects were candle's Metal sdpa accumulation and the bf16 elementwise chains inside the
block. It was the second one: the f32 activation stream took the loss to 1e-6, an order
of magnitude inside torch's own bf16 spread, and no sdpa change was needed
([records/zimage-perf.md](records/zimage-perf.md)).

## The transformer

An S3-DiT, single stream, 6,154,908,736 parameters. `transformer/config.json`, read off
the downloaded file:

```json
{
  "_class_name": "ZImageTransformer2DModel",
  "all_f_patch_size": [1], "all_patch_size": [2],
  "axes_dims": [32, 48, 48], "axes_lens": [1536, 512, 512],
  "cap_feat_dim": 2560, "dim": 3840, "in_channels": 16,
  "n_heads": 30, "n_kv_heads": 30, "n_layers": 30, "n_refiner_layers": 2,
  "norm_eps": 1e-05, "qk_norm": true, "rope_theta": 256.0, "t_scale": 1000.0
}
```

Everything else derives from it. `head_dim` is 3840 / 30 = 128, which is also
`sum(axes_dims)` = 32 + 48 + 48, and the code asserts that. `n_kv_heads == n_heads`, so
there is no GQA: plain MHA, 30 heads of 128. The FFN hidden is `int(dim / 3 * 8)` =
10240, a SwiGLU `w2(silu(w1(x)) * w3(x))` with no bias. `cap_feat_dim` 2560 is the
encoder's hidden size, which is the join between the two halves of this file.

The parameter count is the strongest single check available on that reading: computed
analytically from the module tree it comes to 6,154,908,736, which matches the F32
parameter total HF reports for the `transformer/` folder exactly, to the parameter. Every
shape and every bias below is therefore confirmed rather than inferred. **The tech report
contradicts the config and the report is wrong**: arXiv 2511.22699 Table 2 says 32
attention heads, but 3840 / 32 = 120 is not 128, `norm_q.weight` is [128], and the
`head_dim == sum(axes_dims)` assertion only holds at 30. Use 30.

The weights ship **F32, every tensor**, three shards totalling 24.62 GB, and both
reference implementations load them as bf16. xwen casts at load, so the resident
transformer is 12.3 GB (decisions.md "Weights load through candle's `VarBuilder`").
**The shards are bf16 values in an F32 container**, verified 2026-09-07 by scanning
117 M words and finding the low mantissa bits zero throughout, so that cast loses
nothing whatever. The largest weight in the whole set is 14.0, in
`layers.6.feed_forward.w2.weight`, and 0.0347% of projection values sit below f16's
normal floor of 6.1e-5. Both of those numbers are load-time gates now, because the tensor
gemm stages its weight tiles to f16 (see "The linear layers run on the Metal-4 tensor
gemm" below).

**Activations are f32 and only the weights are bf16**, as of 2026-09-07: the tensor gemm
takes an f32 activation against a bf16 weight and returns f32, so keeping the whole block
f32 avoids a cast at every kernel boundary, and it is also what took the step-0 parity
from cosine 0.999302 to 0.999999. Resident memory is unchanged. Anything that reads "the
transformer is bf16 end to end" predates that evening
(decisions.md "The transformer's linears run on xwen's Metal-4 tensor gemm").

**34 blocks execute and 32 of them are modulated.** Two `context_refiner` blocks see
only the caption and are UNMODULATED, with no adaLN tensor at all; two `noise_refiner`
blocks see only the image and are modulated; then 30 `layers` see the concatenation. The
refiners are modality-specific preprocessors, not extra depth on the joint stream.
Counting `all_final_layer`'s, the checkpoint holds 33 adaLN projections; only
`t_embedder`'s 256-wide output is shared between them.

Tensor names, with the shapes that pin them:

```
all_x_embedder.2-1.{weight [3840,64], bias [3840]}   # ModuleDict key is "{patch}-{f_patch}"
cap_embedder.0.weight [2560]                          # RMSNorm(cap_feat_dim)
cap_embedder.1.{weight [3840,2560], bias [3840]}
t_embedder.mlp.0.{weight [1024,256], bias [1024]}
t_embedder.mlp.2.{weight [256,1024], bias [256]}
x_pad_token [1,3840]      cap_pad_token [1,3840]
noise_refiner.{0,1}.*     context_refiner.{0,1}.*     layers.{0..29}.*
all_final_layer.2-1.linear.{weight [64,3840], bias [64]}
all_final_layer.2-1.adaLN_modulation.1.{weight [3840,256], bias [3840]}
```

and per block:

```
attention.to_q/to_k/to_v/to_out.0.weight  [3840,3840]  (no bias)
attention.norm_q.weight / norm_k.weight   [128]        RMSNorm over head_dim
attention_norm1 / attention_norm2         [3840]       pre- and POST-attention
ffn_norm1 / ffn_norm2                     [3840]       pre- and POST-FFN
feed_forward.w1 [10240,3840]  w3 [10240,3840]  w2 [3840,10240]   (no bias)
adaLN_modulation.0.{weight [15360,256], bias [15360]}  # modulated blocks only
```

There are **no biases anywhere in attention or the FFN**. Biases exist only on
`all_x_embedder`, `cap_embedder.1`, `t_embedder.mlp.{0,2}`, every `adaLN_modulation`, and
`all_final_layer.linear`. RMSNorm is the HF `x * rsqrt(mean(x²) + eps) * w` form at eps
1e-5, not llama.cpp's clamp form, which matters if a norm is ever reused from the GGUF
side of this repo.

The block, which is the whole model:

```
adaln_input = t_embedder(t * 1000.0)                       # [B, 256]

# modulated
scale_msa, gate_msa, scale_mlp, gate_mlp = adaLN_modulation(adaln_input).chunk(4)
gate_msa, gate_mlp   = tanh(gate_msa), tanh(gate_mlp)
scale_msa, scale_mlp = 1 + scale_msa, 1 + scale_mlp
x = x + gate_msa * attention_norm2(attention(attention_norm1(x) * scale_msa))
x = x + gate_mlp * ffn_norm2(feed_forward(ffn_norm1(x) * scale_mlp))

# unmodulated (context_refiner only)
x = x + attention_norm2(attention(attention_norm1(x)))
x = x + ffn_norm2(feed_forward(ffn_norm1(x)))
```

Four things in that are not the usual DiT. There is **no shift term**: four modulation
vectors, not adaLN-Zero's six, hence `4 * dim` = 15360. The **gates are `tanh`**, bounded
in (-1, 1), and the network was trained that way. The `*_norm2` norms are a **sandwich**,
applied to the sublayer output before the gate rather than to the residual stream. And
the block's `adaLN_modulation` has **no SiLU** in front of it, consuming `t_embedder`'s
raw output, where `all_final_layer`'s adaLN does have one; the asymmetry is real and both
reference implementations agree on it. The modulation broadcasts per token from a
`[B, 3840]` vector, one modulation for the whole sequence, image and caption alike.

Attention: `to_q/to_k/to_v`, unflatten to [B, S, 30, 128], RMSNorm q and k over the 128
axis (**before rope**, v untouched), rope q and k, then non-causal SDPA at scale
1/sqrt(128), flatten, `to_out`. Bidirectional over the whole joint sequence.

Rope is **3-axis and the axes are concatenated, not interleaved across axes**. Per axis
`i` of dim `d` and length `e`, `freqs = 1 / theta^(arange(0,d,2)/d)` accumulated in
float64 and then cast to float32, `angles = outer(arange(e), freqs)`, a complex table of
[e, d/2]. A token at integer position `(p0, p1, p2)` takes
`cat([table_0[p0], table_1[p1], table_2[p2]])`: 64 complex values, 128 real dims, exactly
`head_dim`. Complex pairs 0..15 come from axis 0, 16..39 from axis 1, 40..63 from axis 2.
Position `(5, 3, 7)` gives first pairs `(0.283662, -0.958924)`, `(-0.923403, -0.383831)`,
`(-0.801144, 0.598472)`, with pair 16 at `(-0.989992, 0.141120)` and pair 40 at
`(0.753902, 0.656987)`; those five values pin the whole scheme and are what
`src/zimage/transformer.rs`'s unit test asserts.

Position ids on the txt2img path. Let `cap_len` be the caption count padded up to a
multiple of 32 and `(H_t, W_t)` the image token grid:

| tokens | axis 0 | axis 1 | axis 2 |
| --- | --- | --- | --- |
| caption, real and inner pad | `1 .. cap_len`, sequential | 0 | 0 |
| image, real | `cap_len + 1`, constant | `0 .. H_t-1` | `0 .. W_t-1` |
| image, inner pad | 0 | 0 | 0 |

So axis 0 is a text-position axis on which the whole image occupies one slot past the
caption, and axes 1 and 2 are the image's height and width. `axes_lens` bound the tables
at 1536 on axis 0 and 512 tokens on axes 1 and 2, which is 8192 px per side.

**Those bounds are checked, and the check is not a formality.** candle's Metal
`index_select` CLAMPS an out-of-range id to the table's last row rather than failing
(`indexing.metal`, "Force prevent out of bounds indexing"), while the CPU backend errors
— so an over-long caption or an over-large image would come back on this machine as a
plausible picture built from the wrong rotations, and a CPU unit test would catch what the
shipped path would not. `check_size` refuses a side past 8192 px before anything loads,
and `ZImageTransformer2DModel::forward` re-checks `cap_len + f_tokens` against
`axes_lens[0]` and the token grid against axes 1 and 2 from the LOADED config, which is
the authority for what runs. Neither is reachable through `xwen image` today (the encoder
truncates to 512 tokens and no admitted size exceeds the grid), but `generate` and
`forward` are both `pub` and take arbitrary `cap_feats`.

Sequence construction, and the order is load-bearing:

```
cap = cap_embedder(cap_feats)             # RMSNorm(2560) then Linear(2560, 3840)
cap[pad positions] = cap_pad_token        # learned [1,3840], AFTER the embedder
cap = context_refiner(cap)                # 2 unmodulated blocks

x = all_x_embedder["2-1"](patchify(latent))
x[pad positions] = x_pad_token
x = noise_refiner(x, adaln_input)         # 2 modulated blocks

unified = cat([x, cap], dim=1)            # IMAGE FIRST
unified = layers(unified, adaln_input)    # 30 modulated blocks
out     = unpatchify(final_layer(unified)[:image_len])
```

`patchify` maps `(C=16, F=1, H, W)` to `[F_t*H_t*W_t, 64]` with the channel axis LAST in
the patch vector, so the 64-vector is ordered `(pf, ph, pw, c)` with `c` fastest.
`unpatchify` is the exact inverse over the first `image_len` rows, which is why the
image-first order matters: recovery is a prefix narrow.

The timestep embedding is sinusoidal at half 128, `max_period` 10000, **cos first**, in
f32 with autocast off, then `Linear(256,1024)` / SiLU / `Linear(1024,256)`. The `t` handed
in is `1 - sigma`, so the argument runs 0 at pure noise up to 1000 as sigma reaches 0,
which is inverted relative to the Flux and SD3 convention.

The final layer is the one LayerNorm in the model: `scale = 1 + adaLN(silu(adaln_input))`,
then `layer_norm(x, eps=1e-6, elementwise_affine=False) * scale`, then
`Linear(3840, 64)`. Everything else is RMSNorm at 1e-5.

At 1024x1024 the latent is 128x128, patch 2 gives 64x64 = **4096 image tokens**, and
4096 mod 32 = 0, so there is no image padding at all at the default size. Caption padding
is the only padding in practice, and at batch 1 the attention mask is `None`: plain dense
bidirectional attention over `4096 + cap_len` tokens.

## The VAE is the Flux VAE

`vae/config.json` says `AutoencoderKL` with `_name_or_path: "flux-dev"`, and it means it
literally: 16 latent channels, 8x spatial, `scaling_factor` 0.3611 and `shift_factor`
0.1159, which are the Flux values. `block_out_channels` [128,256,512,512],
`layers_per_block` 2, `norm_num_groups` 32, `force_upcast` true, `use_quant_conv` and
`use_post_quant_conv` both false, so there are no 1x1 quant convs at all.

It ships **BF16**, one file, 167.7 MB, 83,819,683 parameters, of which the decoder is
49,545,475 across 138 tensors. Only the decoder is needed for txt2img. Decoder topology,
derived from the tensor shapes: `conv_in` 16 to 512, a mid block of resnet / single-head
spatial self-attention over HW / resnet, then four up blocks of three `ResnetBlock2D`
each (`layers_per_block + 1`, the diffusers decoder asymmetry) at 512, 512, 256, 128 with
a nearest-2x-then-conv3x3 `Upsample2D` on the first three, then `GroupNorm(32, eps=1e-6)`
and `conv_out` to 3. `ResnetBlock2D` is GroupNorm / silu / conv3x3 / GroupNorm / silu /
conv3x3 with an unscaled residual and a 1x1 shortcut where the channel count changes.

The decode path is `latents / 0.3611 + 0.1159`, decode, then `image / 2 + 0.5` clamped to
[0, 1] and rounded to bytes. **`force_upcast: true` means the VAE runs in f32 even though
the transformer is bf16**, which xwen matches: the Flux VAE overflows in fp16 and is
marginal in bf16. The pipeline's divisibility constraint is 16 per side rather than the
VAE's 8, because the transformer's patch size is 2 on top of it. Since 2026-09-08 the
decoder's convs run on xwen's direct conv kernel rather than candle's im2col `conv2d`, still
in f32 (the section "The fused kernels" below, and decisions.md "The VAE decodes on a direct
implicit-gemm conv over NCHW"); the decode is 1.36-1.44 s at 1024x1024.

## The scheduler and the Euler loop

`scheduler/scheduler_config.json`, complete:

```json
{"_class_name":"FlowMatchEulerDiscreteScheduler","num_train_timesteps":1000,
 "use_dynamic_shifting":false,"shift":3.0}
```

The raw grid is `sigma_k = 1 - k/n` for `k = 0..n-1`; the official repo spells it
`linspace(1000, 0, n+1)[:-1] / 1000`, which is algebraically the same for every n. The
lower endpoint is 0 there only because the pipeline assigns `scheduler.sigma_min = 0.0`
on the line before it asks for the timesteps; left at the constructor's own 0.0029940 the
same interpolation gives a grid up to 5.0e-3 away, which is candle upstream's and which
nothing ships (decisions.md "The sigma grid is diffusers', and the official pipeline
computes the same one"). Each sigma then takes the static shift
`sigma' = 3*sigma / (1 + 2*sigma)`, and a terminal 0 is appended to the sigma array. At
the shipped defaults, n = 8 and shift 3.0:

| k | raw σ | shifted σ | t = 1 − σ | dt = σ<sub>k+1</sub> − σ<sub>k</sub> |
| --- | --- | --- | --- | --- |
| 0 | 1.000000 | 1.000000 | 0.000000 | −0.045455 |
| 1 | 0.875000 | 0.954545 | 0.045455 | −0.054545 |
| 2 | 0.750000 | 0.900000 | 0.100000 | −0.066667 |
| 3 | 0.625000 | 0.833333 | 0.166667 | −0.083333 |
| 4 | 0.500000 | 0.750000 | 0.250000 | −0.107143 |
| 5 | 0.375000 | 0.642857 | 0.357143 | −0.142857 |
| 6 | 0.250000 | 0.500000 | 0.500000 | −0.200000 |
| 7 | 0.125000 | 0.300000 | 0.700000 | −0.300000 |
| — | — | 0.0 terminal | — | — |

That table is what `src/zimage/scheduler.rs`'s unit test pins. The step, all in f32:

```
noise_pred = -transformer(latent, 1 - sigma_k, cap_feats)
latents    = latents + (sigma_{k+1} - sigma_k) * noise_pred
```

`dt` is negative and the model output is negated, so the two signs compose into ordinary
flow-matching descent. Latents stay f32 for the whole loop; both reference
implementations assert that.

**There is no resolution-dependent shift in effect.** Both pipelines compute
`mu = calculate_shift(image_seq_len, 256, 4096, 0.5, 1.15)` and hand it to
`set_timesteps`, and `use_dynamic_shifting: false` throws it away. The
`calculate_shift` call is dead code inherited from Flux; xwen does not implement it and
refuses a config that sets `use_dynamic_shifting: true` rather than pretending to support
it. `shift` itself is a legitimate user-facing knob upstream (the official Space exposes
1.0 to 10.0), and non-Turbo `Z-Image` ships 6.0 against Turbo's 3.0.

**Eight steps, not nine.** The HF model card's sample says `num_inference_steps=9, # This
actually results in 8 DiT forwards`, and that comment is false against current diffusers:
n = 9 gives nine sigmas and nine forwards. diffusers commit 32ecbe383, 2026-05-29, "Fix
redundant Z-Image terminal timestep (#13730)", changed every Z-Image default from 9 to 8
and replaced `scheduler.sigma_min = 0.0` with an explicit
`get_default_z_image_sigmas(num_inference_steps)`; the official repo's
`DEFAULT_INFERENCE_STEPS` is 8 and the paper says 8 NFE. Anything that says 9 predates
that commit. `DEFAULT_STEPS` here is 8.

**Turbo does no CFG.** `guidance_scale` is 0.0 in the official repo's defaults, the model
card's Model Zoo marks Turbo's CFG column as unsupported, and diffusers' own `__call__`
default of 5.0 is a footgun to pass 0.0 past. One encoder pass, one transformer forward
per step, no negative prompt. Non-Turbo `Z-Image` does use CFG, at 50 steps.

## Traps in the pipeline, each of which runs and produces plausible garbage

None of these fails loudly. They are ordered by how easy they are to get wrong.

- **Rope is INTERLEAVED-pair, not NEoX.** `reshape(..., -1, 2)` pairs adjacent dims
  `(x0,x1), (x2,x3), ...`. Every Qwen graph in this repo pairs `i` with `i + d/2`. Reusing
  a NEoX rope here runs and denoises noise. The rotation is also done in f32 regardless of
  the activation dtype (`x_in.float()` then `.type_as`), and the tables are accumulated in
  float64 before the f32 cast.
- **The joint sequence is IMAGE FIRST**, `cat([x, cap])`, and the output is recovered as a
  prefix narrow. Reversing it is silent because the rope positions travel with the tokens.
- **Modulation is `1 + scale` and `tanh(gate)`, with no shift term.** Four vectors, in the
  order `scale_msa, gate_msa, scale_mlp, gate_mlp`.
- **`t` fed to the transformer is `1 - sigma`, and the model output is NEGATED** before
  the Euler step. Getting exactly one of the two right produces a diverging trajectory
  that still looks like an image early on.
- **The pad tokens are learned and UNMASKED.** Both modalities pad up to a multiple of 32
  with `cap_pad_token` / `x_pad_token`, applied AFTER their embedder, and those rows
  participate in attention as ordinary keys and queries. Zero-padding before the embedder
  and masking the result, which is the obvious implementation, is a different model. Do
  not optimize them away.
- **Eight steps, not nine**, per the commit above.
- **The static shift is 3.0 and the dynamic-shift code is dead.** Implementing
  `calculate_shift` because both pipelines call it reproduces neither reference.
- **fp16 is disqualified**, not merely inadvisable: activations exceed 65504 and the
  result is NaN latents and a black image
  (decisions.md "The transformer runs bf16 end to end"). The transformer's weights are
  bf16 and its activations f32, with f32 accumulation throughout (2026-09-07).
- **The VAE is bf16 on disk and runs in f32** under `force_upcast`, with the Flux
  `shift_factor` 0.1159 and `scaling_factor` 0.3611 applied as
  `latents / scale + shift` on the way in.
- **`cap_embedder.0` is an RMSNorm over 2560 and `cap_embedder.1` is the projection**, so
  the caption is normed before it is projected; `all_final_layer`'s adaLN has a SiLU and
  the blocks' do not.
- **`all_patch_size` / `all_f_patch_size` are a ModuleDict with exactly one key, `"2-1"`.**
  Hardcode it. `siglip_feat_dim` is null on both public checkpoints, so there is no SigLIP
  tower and no Omni path. The control hooks now carry the Fun Union residuals described
  below.

## Image edits, adapters and control

Added 2026-09-08. `src/zimage/inputs.rs` owns RGB decoding, grid sizing, masks and the
final source-pixel composite. `ImageEdit` selects a tail of the existing schedule with
`floor(N - N * strength)`. Source VAE sampling uses a separate reproducible noise draw,
clamped log variance `[-30,20]`, and `(z - 0.1159) * 0.3611` once. Tier-one inpainting
restores the source latent at the next sigma after each step. White means repaint;
black-mask output pixels are copied exactly from the source. Strength zero returns
the prepared source without denoising.

`src/zimage/lora.rs` overlays the transformer VarBuilder at load. It merges deltas in
f32 before the weight cast, supports split and fused QKV exports, and rejects unknown
or unused targets. A changed adapter set reloads a fresh base. No per-step adapter
operations are added.

`src/zimage/controlnet.rs` implements the author's full and lite Fun Union 8-step
graphs. Both reuse the generator's embedders and refiners, including merged adapters.
Two control-refiner outputs reach the generator; main residuals follow layers
0,2,...,28 for full and 0,10,20 for lite. Main control attention is joint image/caption.
The input is `[control mode:16, keep-mask:1, masked-source mode:16]`; mask the normalized
source pixels to grey before VAE mode encoding. Missing source conditioning is zero
latent. This trained inpaint path omits tier-one latent restoration but retains the
final pixel composite. Control is active in a half-open fraction of the full schedule,
default `[0,0.8)` at scale 0.75. Zero scale skips its forward.

`src/zimage/preprocess.rs` owns native Canny and lazy CPU depth/pose models. Depth uses
the trained 518-square grid; DWPose uses the author's detector and whole-body ONNX
weights. Models are cache-only and unload with the image engine. Native render and
preprocess routes share that engine with OpenAI generation, edits and variations.

Reference results and remaining limits: [edits](records/zimage-img2img.md),
[adapters](records/zimage-lora.md), [control](records/zimage-controlnet.md),
[preprocessors](records/zimage-preprocessors.md), [HTTP and CLI](records/zimage-control-api.md).

## What the vendored candle module got wrong

`src/zimage/` is candle's `z_image` at rev 21cca0b (PR #3261), vendored and corrected
(decisions.md "candle's `z_image` module is vendored into `src/zimage/`"). It was
right on every trap in the list above except the ones below, which is a better record
than two commits of upstream history would suggest, and it was wrong in four places that
matter. Each correction moves it TOWARD the reference; none is an optimization.

1. **Caption padding.** Upstream zero-padded the caption before `cap_embedder` and masked
   the pad. The reference pads after the embedder with the learned `cap_pad_token` and
   does not mask. `forward` now takes `(x, t, cap_feats)` with no mask at all and bails on
   batch != 1.
2. **The sigma grid.** Upstream `set_timesteps` interpolated between the already-shifted
   training extremes and then shifted again, which is not the reference grid at any n. It
   is now the raw `1 - k/n` ladder, the static shift, and an appended terminal zero, with
   the 8-step table pinned by a test. `mu` and dynamic shifting are gone, and a config
   asking for them is refused at load.
3. **Rope dtype.** Upstream built the tables in f32 in the model dtype, so bf16. They are
   now built in f64 and kept in f32, and the rotation is done in f32 and cast back.
4. **QK-norm eps** came from a hardcoded 1e-5 and now comes from `cfg.norm_eps`, which is
   the same value on this checkpoint and is not guaranteed to be.

Four smaller ones. The timestep stays f32 into the model rather than being cast to bf16
first. The VAE and the Euler loop run f32 with the model output negated before the step,
which the reference does and candle's library did not (its example did). The postprocess
rounds to nearest byte where upstream truncated. And the CUDA flash-attn arm, the CFG
helpers, `calculate_shift` and `preprocess.rs` are removed; what is kept is an attention
switch, which since 2026-09-08 is `AttnImpl` with three arms rather than the original
`use_accelerated_attn` boolean.

**`XWEN_ZIMAGE_ATTN` names three arms**, read when the `Config` is built, which is where
the shipped `transformer/config.json` — carrying no such key — takes its `serde` default.
A value that names none of them is a load error rather than a silent default.

- **`flash`, or `xwen`, or unset** is the shipped path: xwen's own
  `ops::flash_attn_bidirectional`, below.
- **`fused`, or `sdpa`**, is candle's Metal SDPA. It was the default before 2026-09-08 and
  it is what a masked or non-Metal or batched call still takes.
- **`basic`** is the explicit matmul, scale, softmax, matmul chain, the reference arm.

The pipeline logs the attention arm only when it is NOT the default, which is the rule
`XWEN_ZIMAGE_LINEAR` already followed: a silent startup means `flash`, and any line naming
an arm means someone set the variable.

The arms are a real A/B: no two of them share an attention kernel, and the unit tests
assert the outputs differ by a NONZERO amount under the bar, because a bit-identical
result would mean the switch selected one kernel twice. The switch was unreachable and
untested in the first arc, which is the shape the `XWEN_QWEN3_ATTN=sdpa` ablation was
vacuous in for a whole arc (AGENTS.md "Verification workflow").

## The linear layers run on the Metal-4 tensor gemm

Since 2026-09-07, `src/zimage/linear.rs` is the seam every projection in the transformer
goes through, and it is xwen's own code rather than anything vendored. `Projection` holds
a bf16 `[out, in]` weight plane and an optional f32 bias, reshapes a rank-2 or rank-3
input to `[t, k]`, and calls `crate::ops::matmul_bf16` on Metal, which is the
cooperative-tensor kernel the language models prefill on. It measures 36.6-38.6 TFLOPS at
this model's shapes where candle's steel gemm measures 14-15.6 whatever dtype it is
handed, and the linears are 57 of a step's 62 TFLOP. Off Metal the kernel does not exist
and the candle chain runs instead.

Three things about it will not survive being changed casually.

- **The stream is f32 because the kernel's contract is.** It takes an f32 activation
  against the bf16 weight and returns f32, so a bf16 stream would pay a widen and a
  narrow at every one of the seven token-width linears per block. Norm weights, pad
  tokens and biases load f32; the projection weights alone are fetched bf16, so resident
  bytes are unchanged. The pipeline's `dtype` field means the activation dtype now.
- **A weight past f16's finite range would be silent garbage**, the kernel staging each
  weight tile to f16 on its way into the tensor unit. `ensure_weights_fit_f16` refuses
  such a checkpoint at load, naming the tensor, for 0.2 s of load time. Do not widen that
  to make a load succeed: the shipped checkpoint's largest weight is 14.0, so a refusal
  means the weights are not the ones this graph was read against.
- **`in % 32 == 0 && out % 4 == 0`** is the kernel's shape requirement, asserted in
  `Projection::new` on Metal only, the CPU tests using a `cap_feat_dim` of 16 that the
  kernel never sees.

A third switch profiles a run rather than changing it. `XWEN_ZIMAGE_PROFILE=1 xwen image
...` prints per-stage milliseconds for the transformer, as a mean per step over steps 2
through 8 and split by phase, and for the one VAE decode; `src/zimage/profile.rs` is the
whole instrument and off it costs one `Option` check per site. Its numbers are NOT
figures: every mark syncs and also evicts candle's buffer pool, so a table runs 1.39x
high at 1024x1024 and its small rows about 1.9x, which
[docs/benching.md](benching.md) spells out and
[records/zimage-perf.md](records/zimage-perf.md) deflates.

`XWEN_ZIMAGE_LINEAR=candle` is the bisect arm, beside `XWEN_ZIMAGE_ATTN` and read the
same way, when the `Config` is built: candle's own bf16 gemm over bf16-rounded
activations, sharing no matmul code with the shipped path, with a unit test asserting the
two agree to a nonzero 2.34e-3 under a 5e-3 bar. A value naming neither arm is a load
error. `tests/zimage_microbench.rs` is the ignored bench that priced the kernel choice,
and [records/zimage-perf.md](records/zimage-perf.md) is where its tables live.

## The fused kernels: rope, the norm scale, the gated residual, bidirectional attention

Four seams added 2026-09-08 (bee11da and the arc after it), all in `src/ops` except the
norm, and none of them a math change. The record is
[records/zimage-perf.md](records/zimage-perf.md) and the decisions are
[decisions/zimage.md](decisions/zimage.md).

- **`ops::rope_pair(x, cos, sin)`** is the interleaved-pair rotation in one kernel over
  `[batch, seq, heads, head_dim]` f32 with `[seq, head_dim/2]` f32 tables, one thread per
  pair. It is **bitwise identical** to the candle chain it replaced, tested at the
  production shape `[1, 4128, 30, 128]`, and it stays that way only because FP contraction
  and reassociation are pinned off in the kernel: turn either back on and the f32
  rotation, which is one of the four corrections above, quietly stops being the correction.
  `ops::rope_neox` is by-halves and does not apply to this graph.
- **`BlockNorm`** replaces `with_tracing::RmsNorm` for the block's four modulated norms
  (`QkNorm` and `cap_embedder_norm` still use candle's). `forward` is the same fused
  `candle_nn::ops::rms_norm`; `forward_scaled(x, scale)` multiplies `1 + scale` into the
  `[dim]` weight and runs ONE norm, which removes a full-tensor `broadcast_mul` per site.
  It is the one part of the arc that is not bitwise: two f32 multiplies round in a
  different order, worth 0.9 dB of image PSNR over eight steps and nothing at step 0, and
  that cost is accepted (decisions.md "The adaLN scale folds into the norm weight").
- **`ops::gated_residual(h, y, gate)`** is `h + gate[c] * y` in one pass, bitwise
  identical to candle's broadcast-multiply-then-add, which ran at 48 GB/s against 530 for
  the contiguous kernel.
- **`ops::flash_attn_bidirectional(q, k, v, scale)`** reuses the CAUSAL flash kernel with
  no kernel edit, by placing the queries at absolute position K while the keys sit at 0
  through K-1 with an unbounded window, which makes both of the kernel's mask tests vacuous
  and its block-skip bound evaluate to the full key count. It is bitwise identical to
  candle's unmasked f32 sdpa at every shape tested. k and v go in through
  `ops::permute_01_f16`, which does the permute and the f16 cast in one pass. It is the
  `flash` arm (alias `steel`) and was the default for part of 2026-09-08.

**The trap on that one: the flash kernel is not a faster arithmetic path.** It is a
vendored copy of candle's MLX steel attention, simdgroup matmul with f32 accumulate, so it
runs at candle's rate, 13.1 TFLOP/s isolated at the production shape, and the switch moved
`attn.sdpa` only 740 to 687 ms profiled. What it bought was the f16 k/v traffic and the
single-pass permutes, not arithmetic (decisions.md "Bidirectional attention is a
query-position trick on the causal flash kernel").

Two more seams landed later on 2026-09-08 (a763c61 and e5d9775), and one of them is the
kernel that trap said did not exist.

- **`ops::flash_attn_tensor(q, k, v, scale)`**, `src/ops/flash_t.metal`, is the shipped
  attention, the `tensor` arm (alias `xwen`) that `AttnImpl::SHIPPED` names. QK^T and PV both
  run through `mpp::tensor_ops::matmul2d`, the cooperative-tensor primitive the gemms use,
  with the online softmax over cooperative-tensor elements. One threadgroup per 64-query
  block and head, four simdgroups, each owning 16 rows and walking the keys in blocks of 32
  with no barrier: Q is staged once as half in threadgroup memory, K and V are read from
  device as f16 tensors, S to P happens in registers and O stays in its cooperative
  destination tensor for the whole key loop, rescaled per row through a slot mask. Isolated
  at 30 x 4128 x 128 it reads 5.80 ms against the steel copy's 20.05, 45.1 against 13.1
  TFLOP/s; profiled `attn.sdpa` fell 698 to 231 ms per step. Against candle's f32 sdpa it is
  rel L2 1.6e-4 to 4.5e-4, not bitwise: q, k and v are rounded to f16 on the way in, and the
  test asserts the tensor and steel arms differ. Parity on this arm is step-0 cosine 0.999999
  at mean rel 0.0011 and image PSNR 46.54 dB, 0.45 dB over the steel arm (45.60 as first
  landed, before 7456e5c masked the padded columns after the scaling). Two things about it
  are forced by the SDK rather than chosen: the per-simdgroup structure, because input
  cooperative tensors, `reduce_rows` and `map_iterator` are all `static_assert`ed to
  simdgroup scope, and the half Q, because an f32 operand under `relaxed_precision` is
  consumed at less than f16 precision (decisions.md "The shipped attention arm is a Metal-4
  tensor-op kernel").
- **`ops::conv2d_direct` and `ops::group_norm`**, `src/ops/conv2d_direct.metal` and
  `src/ops/group_norm.metal`, are the VAE decoder's conv path, the `xwen` arm of
  `XWEN_ZIMAGE_VAE` (`candle` is the vendored chain). An implicit-gemm f32 3x3 and 1x1
  convolution on NCHW as candle stores it, simdgroup 8x8 matrix ops, the input tile and its
  halo staged in threadgroup memory and the im2col operand formed as a transposed
  `simdgroup_load` at each tap's offset so it is never written; 10-11 TFLOP/s at every decoder
  shape against candle's 1.2-4.4. The bias is in the store, and the GroupNorm affine, the
  silu, the 2x nearest upsample (a read at half coordinates) and the residual add are folded
  into the conv's read and store, so GroupNorm is one statistics read plus a per-channel fold
  and the normalized tensor exists only for the mid-block attention. The decode went 5.2 s to
  1.36-1.44 s at 1024x1024 and 1.06 to 0.26 s at 512x512, VAE-alone PSNR 92.62 to 92.32 dB.
  Two traps pinned in the code: the simdgroup accumulator array spills unless the tile loops
  are force-unrolled (1 TFLOP/s otherwise), and a glue kernel bounds on an explicit `n`,
  never on `threads_per_grid` (decisions.md "The VAE decodes on a direct implicit-gemm conv
  over NCHW").

The record for both is [records/zimage-perf.md](records/zimage-perf.md) "The VAE on a direct
conv kernel, attention on the tensor units, and the SwiGLU bf16 store refuted"; the third
part of that arc, a bf16 store for the SwiGLU intermediate, is refuted there and not shipped.
