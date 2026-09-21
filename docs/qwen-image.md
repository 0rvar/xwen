# Qwen-Image 2.1, the pipeline and its text encoder

The architecture reference for the second diffusion pipeline in the repo, the way
[zimage.md](zimage.md) is Z-Image-Turbo's: what the model is, how `src/qwen_image/` runs
it, the traps that run and produce plausible garbage, and how the gates are run. The
decisions are in [decisions/qwen-image.md](decisions/qwen-image.md), the arc with its
tables and review rounds in [records/qwen-image-t2i.md](records/qwen-image-t2i.md), and
the research that preceded the code in [qwen-image-2.1-plan.md](qwen-image-2.1-plan.md).
Where that plan and this file disagree, this file was written after the shipped files
were open and the plan was not.

State as of 2026-09-21 (5ea0d43): text-to-image renders from the CLI,
`xwen image --model qwen-image-2.1`, and `xwen encode-text --model qwen-image-2.1-encoder`
runs the encoder alone. The serve images route refuses the model. Editing with reference
images is designed for and not wired: the layout, the rope walk and the prompt renderer
take reference blocks, and nothing feeds them. The module is written from diffusers at
`6256aa7666cedd47443adc8f82da9a10e110b09c` (`transformer_qwenimage21.py`,
`pipeline_qwenimage21.py`, `autoencoder_kl_qwenimage21.py`), candle having no module for
this model, so unlike `src/zimage/` nothing here is vendored.

## The checkpoint

`Qwen/Qwen-Image-2.1`, snapshot `790c92633540aa0cb11d9abf19eb46d861714758`, 33,134,949,212
bytes, read off the files themselves on 2026-09-21. Qwen Research License: non-commercial,
and it binds the weights, not the images made with them.

| part | files | on disk |
| --- | --- | --- |
| `transformer/` | two shards, bf16, 297 tensors | 14,230,249,472 bytes (9.97 + 4.26 GB) |
| `text_encoder/` | four shards, bf16, the FULL Qwen3-VL-8B (8,767,123,696 parameters) | 17,534,247,392 bytes |
| `vae/` | one file, **F32**, 238 tensors | 1,350,989,512 bytes |
| `processor/` | `tokenizer.json`, byte-identical to `Qwen/Qwen3-4B`'s, plus the vision preprocessor config | MB |
| `scheduler/`, `model_index.json` | configs | KB |

Two registry entries (`src/hub.rs`). `qwen-image-2.1`, full name `Qwen-Image-2.1`, is
`Format::Diffusion` with `model_index.json` listed first so the resolved path's parent is
the snapshot root, and it repeats the encoder's files so one fetch leaves nothing to
download. `qwen-image-2.1-encoder`, full name `Qwen-Image-2.1-text-encoder`, is a
safetensors entry at `text_encoder/` with the tokenizer at `processor/tokenizer.json`, no
zero-run allowlist, and an `EncoderSpec` of layer 36, `final_norm: false`, `max_tokens`
4096. Both are `VocabFamily::Qwen3`, neither is servable, neither is listed on
`/v1/models`, and `auto_fetch` is false for both, which governs serve: the CLI announces
the download and fetches, as it does for Z-Image.

## The architecture in numbers

Verified against `transformer/config.json`, `scheduler/scheduler_config.json`,
`vae/config.json`, `text_encoder/config.json`, the safetensors headers and the two
reference gates below.

| | value |
| --- | --- |
| transformer | 7,115,112,448 projection values, 32 single-stream blocks, all modulated |
| width | 4096 = 32 heads of 128, no GQA, no biases anywhere |
| FFN | SwiGLU 4096 to 12288 to 4096 (`mlp_ratio` 3); `gate_layer` is the SiLU'd plane, `proj` the ungated one |
| block norms | LayerNorm without affine, eps 1e-6 |
| QK-norm | RMSNorm over 128, eps 1e-6, before rope, on text rows too |
| modulation | ONE `SiLU -> Linear(4096, 16384)` for the whole model, chunked `[scale1, gate1, scale2, gate2]`, `1 + scale`, `tanh(gate)`, no shift |
| timestep | `t = sigma`, times 1000 inside the model, sinusoid 256 with the COSINES in the first half, freqs `exp(-ln(1e4) * i / 128)`, then `linear_1`, SiLU, `linear_2`, no biases |
| rope | interleaved-pair, axes (16, 56, 56), theta 1e4, angles built in f32 |
| `txt_in` | zero-centred RMSNorm storing `w - 1`, Linear, tanh-GELU, Linear; the only `(1 + w)` norm in the model |
| final layer | LayerNorm without affine times `1 + scale`, then `proj_out` 4096 to 64 |
| latent | 64 channels at 16x, patch 1, packing is a raster flatten |
| image tokens | 1024 at 512x512, 4096 at 1024x1024, 16384 at the native 2048x2048 |
| largest weight | 3.4531, in `txt_in.out_layer.weight`, so every plane fits the f16 tile staging |
| encoder | Qwen3-VL-8B text tower: 36 layers, hidden 4096, 32 Q / 8 KV heads of 128, SwiGLU 12288, vocab 151936, theta 5e6, untied head |
| VAE | Wan-2.2-style residual VAE specialised to one frame, RGBA, 64-channel latent, f32 |
| scheduler | flow-match Euler, dynamic exponential shift, `shift_terminal` 0.02, 40 steps, no guidance |

## The text encoder

The text tower of Qwen3-VL-8B is the dense Qwen3 graph this repo already grades
([qwen3-dense.md](qwen3-dense.md)), wider, so it opens through `src/qwen3/` and there is
no second encoder implementation. What the loader learned for it:

- `model_type: qwen3_vl` is accepted by flattening `text_config`, whose own `model_type`
  must be `qwen3_vl_text`. `tie_word_embeddings` sits at the top level of that file.
- The tensor prefix is `model.language_model.`. The 351 `model.visual.*` tensors and the
  untied `lm_head.weight` are dropped from routing after the index and shard cross-check:
  not validated, not scanned, not loadable. Every other strictness is unchanged, so a
  missing or renamed language tensor still fails.
- An untied set parses, `XwenModel::load` refuses it through `ensure_tied_head()`, and
  `load_encoder` is the one path that accepts it. No language surface can run this set.
- `rope_scaling` must be exactly `{rope_type: "default", mrope_interleaved: true,
  mrope_section: [a, b, c]}` with the sections summing to `head_dim / 2`. It lands in
  `RopeSpec::mrope`.

**MRoPE collapses to plain NEoX rope for text, and that is now measured.** With the same
position on all three axes the interleaved recomposition is a no-op. The reference dump
asserts `get_rope_index` returns equal, `arange` ids on all three axes, and the encoder
gate reads cosine 0.99997 or better on the plain table. `rope::mrope_interleaved_tables`
builds the real three-axis rows for the editing phase (slots 1, 4, ..., 58 take H, slots
2, 5, ..., 59 take W, the rest T, for sections [24, 20, 20]); equal ids reproduce the
plain table bit for bit, and the text path does not call it.

**The hidden state is the residual after all 36 layers, BEFORE the final norm.** The
repo's index convention norms at index `n_layer`, so `EncoderSpec { layer: 36 }` alone
would return exactly the tensor the reference hooks around. `EncoderSpec::final_norm` and
`qwen3::HiddenTap { depth, final_norm }` express it, `XwenModel::encode_spec` is the one
helper every spec-driven caller uses, and bare `encode(ids, n)` keeps its old meaning.
The dump confirms that transformers 5.17 returns the NORMED state from
`hidden_states[-1]` without the pipeline's hook, and that the normed state sits at mean
cosine 0.81 and minimum 0.62 against the right one.

**The prompt is a raw template string**, not `apply_chat_template` output
(`src/qwen_image/conditioning.rs`):

```
<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
```

An empty prompt becomes one space. The rows of the system turn are dropped from the
hidden state, their count computed by tokenizing the system block with the same
tokenizer: 14 on the shipped file, equal to the pipeline's `_drop_idx`. Everything from
the second `<|im_start|>` on is kept, the open assistant turn included. The reference has
no length cap, so `max_tokens` 4096 is a refusal and not a truncation, cutting the tail
being what would remove the assistant turn. One deliberate difference from the reference:
literal special-token text in the user's prompt (`<|image_pad|>`, `<think>`) stays plain
text, which is what keeps a prompt from changing the image-slot count once editing lands.
`render()` already writes the `<imageN><|vision_start|><|image_pad|><|vision_end|>`
markup for reference slots; `prompt_ids` refuses a non-empty list until Phase 5.

## The transformer

`src/qwen_image/transformer.rs`. Every projection is `zimage::linear::Projection`: a bf16
weight plane through `ops::matmul_bf16` on an f32 activation stream, which is the
kernel's contract. The shards are bf16 on disk and are read through an f32 `VarBuilder`
with each projection asking for its own plane in bf16, so the model is never held in f32.
`ensure_weights_fit_f16` runs at load.

**The sequence is an ordered list of segments, and the target image is last.**
`Layout` holds `Segment::Text { len }` and `Segment::Image { height, width }`. For
text-to-image it is `[text ; target]`. With reference images it is NOT
`[text ; references ; target]`: the reference expands each `<|image_pad|>` slot four-fold
INSIDE the text stream and writes the clean latents into those rows, so text comes before
and after each reference. `Layout::from_slots` builds that form from the slot mask, which
covers the encoder sequence only; the reference's own mask also carries the target's
slots, and a mask that ends with them is refused by name.

**Attention is block-causal and runs as segments, with no dense mask on the shipped
arm.** The rule is `allowed = (q >= kv) OR same_image_block`. One call per prefix segment
over keys `[0, end)`, a causal triangle for a text segment only, and one call for the
target over all keys. On Metal an image segment is `ops::flash_attn_tensor` with
`q = [32, N, 128]` f32 and `k, v = [32, K, 128]` f16, and a text segment is an explicit
f32 chain, being tens of tokens. `XWEN_QWEN_IMAGE_ATTN=basic` is one dense f32 attention
under the explicit mask, which is the off-Metal path and the bisect arm, and the CPU
tests hold the two equal to 1e-4 for text-to-image and for a layout with two adjacent
reference images.

**The prefix K/V is kept across steps**, which is diffusers' `use_kv_cache=True` and the
mode the reference sample is produced under. Text and reference rows are modulated from a
fixed `t = 0` row in every block and in the final norm, and they attend only to
themselves, so they are identical at every step. `forward_prefill` runs the whole
sequence at step 0 and returns a `PrefixCache` of each block's post-QK-norm, post-rope K
and V (f16 head-major on the tensor arm, copied so the cache does not pin the prefill's
tensors) plus the target's rope rows. `forward_cached` then runs ONLY the target rows: no
`txt_in`, no prefix attention rows, no prefix FFN. Nothing in the final layer is
step-independent. `XWEN_QWEN_IMAGE_CACHE=off` runs `forward_full` every step, the same
math, so it is a bisect arm and not a wrong-graph bracket.

**The rope walk.** One running `position`: a text token takes `(p, p, p)` and advances it
by one; an image block takes `frame = position` for all its rows, `h` in
`[-(H - H/2), H/2)` and `w` likewise, centred so its spatial ids do not depend on where
it sits, then `position += max(H, W)`. The angle is a float32 inverse frequency times the
index, as the reference computes it, because an f64 table drifts from the reference at
large positions; the cos and sin of that f32 angle are rounded once from f64. The table
covers `[0, 8192)` and `[-1024, 0)`. `positions()` bounds the positions actually WRITTEN
(a frame or text position at 8192 is refused, an image side past 2048 latent tokens is
refused) before it allocates, and `MAX_IMAGE_SIDE` is what `check_size` reads.

`GraphVariant::{FullyBidirectional, RealTimestepForText}` are the two wrong graphs the
parity gate asserts fall outside its bar.

## The scheduler

`src/qwen_image/scheduler.rs`, its own file and not an arm in the vendored
`src/zimage/scheduler.rs`. It requires `use_dynamic_shifting` and the exponential time
shift and refuses the karras, beta, exponential-sigma, inverted and stochastic variants.
`mu = 0.5 + (0.9 - 0.5) / (8192 - 256) * (N - 256)` from the TARGET token count, then
`exp(mu) / (exp(mu) + (1/t - 1))`, then the `shift_terminal` stretch, over
`linspace(1, 1/steps, steps)`, then a trailing zero. `mu` is 0.538710 at 1024 tokens,
0.693548 at 4096 and 1.312903 at 16384 (past `max_image_seq_len`, the line extrapolates).
Float widths follow numpy: `mu` f64, the grid f32. The first sigma is exactly 1.0 and the
last non-zero one is 0.019999980926513672, the f32 stretch not landing on 0.02. The model
is fed `sigma` and its output is used as is: `x += (sigma_next - sigma) * v`, no negation.

## The VAE

`src/qwen_image/vae.rs`, f32 end to end, candle convs. `decode` takes the NORMALISED
latent `[B, 64, h, w]`, applies `z * std + mean` per channel itself (the reference does
that in the pipeline), and returns `[B, 4, 16h, 16w]` clamped to [-1, 1]. `encode` is the
mirror: the posterior MEAN, the first 64 of 128 channels after `quant_conv`, normalised.
Text-to-image never runs it; it exists because editing encodes each reference once.

- Encoder channels `96 x [1, 1, 2, 4, 8, 8]` with two residual blocks a stage, decoder
  `144 x [8, 8, 8, 4, 2, 1]` with three. The residual block is norm, SiLU, conv, norm,
  SiLU, conv over a 1x1 shortcut when widths differ. The mid block is one attention head
  over all positions between two residual blocks.
- The norm is an L2 norm over the CHANNEL axis per pixel, `x / max(||x||, 1e-12) *
  sqrt(C) * gamma`, not an RMS over `mean(x^2) + eps`. `sqrt(C)` folds into gamma at
  load. The attention norm stores gamma as `[C, 1, 1]`, every other one as `[C, 1, 1, 1]`.
- The file holds 2-D convs, and the 12 `time_conv` tensors (weight and bias on encoder
  down blocks 1 to 3 and decoder up blocks 0 to 2) are never evaluated for one frame.
  They are excluded by name, and a test holds the file header equal to live plus dead.
- Downsampling pads right and bottom only, then a stride-2 conv with no padding.

**The single-frame shortcut stages are the trap of this module.** There are nine, five
down and four up, and five of them are not what their docstrings say:

| stage | channels | closed form at T = 1 |
| --- | --- | --- |
| encoder 0 | 96 to 96 | 2x2 average pool |
| encoder 1, 2, 3 | 96 to 192, 192 to 384, 384 to 768 | `out[2c] = 0`, `out[2c+1] = avgpool2(in[c])` (the zero frame padded in front) |
| encoder 4 | 768 to 768 | identity |
| decoder 0, 1 | 1152 to 1152 | nearest 2x |
| decoder 2 | 1152 to 576 | `out[o] = nearest2x(in[2o+1])`, the odd input channels |
| decoder 3 | 576 to 288 | `out[o][2y+i][2x+j] = in[2o+i][y][x]`, even rows from `in[2o]`, odd from `in[2o+1]` |
| decoder 4 | 288 to 144 | no shortcut |

Each closed form is tested bit for bit against a literal transcription of the reference's
`view/permute/view` on a 5-D tensor. A port from the docstrings reconstructs a plausible
image and is wrong on five stages.

The decode is where the memory goes. With no sync inside it candle's buffer pool keeps
every conv intermediate alive, 71 GiB at 1024x1024. `device.synchronize()`, which on
Metal is what evicts the pool, now runs between convolutions, which takes the peak to
55 to 56 GiB with byte-identical PNGs. What is left is one layer's im2col column buffer,
10.9 GB for the 288-channel conv at 1024x1024. `XWEN_QWEN_IMAGE_VAE` accepts `candle`
only and refuses `xwen` and `direct` by name until the direct-conv arm exists.

## The pipeline

`src/qwen_image/pipeline.rs`. Load order: the four bisect switches are read first so a
typo fails before any load, the scheduler config is validated before anything large is
resident, then the transformer, then the VAE through a second f32 `VarBuilder`.

- Sizes: both sides positive multiples of 32, refused and not floored, and a side at most
  32768 px by the rope rule, checked in closed form before anything allocates.
  `check_layout` applies the same rule to a caption's length. There is no token-count
  multiple, 2.1 having no pad tokens.
- The noise is `[1, 64, h/16, w/16]` from `zimage::sampling::seeded_noise`, unscaled,
  packed by a raster flatten. It is unrelated to torch's `randn`, so reference
  comparisons inject `--latents`.
- 40 steps by default. Step 0 is the prefill, the rest are cached.
- The latent the loop steps is in the normalised space; `latents-final` in the fixture is
  that latent before denormalisation and `decode_latents` takes it as is.

**The image comes back RGBA only when at least 10 pixels have alpha at most 8** (`u8`,
the same quantisation as the colour planes), and RGB with the alpha plane dropped
otherwise. `keeps_alpha` owns the rule, `CLEAR_ALPHA_MAX` and `CLEAR_PIXELS_MIN` are its
constants, `Rendered` carries `alpha_min` and `clear_pixels`, and `encode_png` picks RGB8
or RGBA8 by channel count. "Any pixel that is not opaque" would fire on every image: an
ordinary prompt's alpha is 252 to 255 with a sixth of the pixels at 254. A transparent
background sits at alpha 1, not 0, on the reference and on xwen alike, which is why the
cutoff is 8. The histograms are in [the record](records/qwen-image-t2i.md).

**Memory admission is fitted, not guessed.** `peak_bytes = 17 GiB + 48 GiB per
megapixel`, the line through the two measured decode peaks (25 GiB at 512x512, 56 GiB at
1024x1024) with 15% on top: 29 GiB and 65 GiB. The step phase peaks at 18 to 19 and
27 GiB. `memory::qwen_image_peak` wraps it under a 1,048,576-pixel cap whose message says
the cap is about measurement and not a model limit. The encoder is released before the
transformer loads, and the run prints the device at 0.0 GB after the drop, 15.7 before.

## The CLI

`xwen image --model qwen-image-2.1 --prompt <text>`. `run_image` dispatches on the
pipeline entry, by alias or by `_class_name` in a snapshot root's `model_index.json`
(`pipeline_at`). `--steps` defaults per model, 8 for Z-Image and 40 here. Everything that
can refuse does so before any weight is resolved: the flags this model lacks (`--init`,
`--strength`, `--mask`, `--mask-blur`, `--lora`, `--control*`), the four bisect switches
and `XWEN_QWEN3_ATTN`, the size, the step count, admission, an overlong prompt (the
tokenizer alone is resolved first), and the shape of an injected file. `--cap-feats`
skips the encoder and fetches none of its files. `--latents` and `--dump` work as they do
for Z-Image.

## Traps, each of which runs and produces a plausible result

- The hidden state is pre-norm. transformers 5.17 hands back the normed one from
  `hidden_states[-1]`, and the repo's own index 36 norms too.
- The prompt is the raw template, with the first 14 rows dropped AFTER encoding. The chat
  template tokenizes differently.
- The sequence is TEXT FIRST and the output is the target SUFFIX. Z-Image is image first
  and a prefix narrow.
- `t` is `sigma` and the velocity is used as is. Z-Image feeds `1 - sigma` and negates.
- Text and reference rows read the `t = 0` modulation row, in every block and in the
  final norm. Modulating them from the real `t` reads cosine 0.9874 at step 0.
- Attention is block-causal. Fully bidirectional attention reads cosine 0.9825.
- `txt_in`'s norm weight is stored as `w - 1`. It is the only such norm in the model.
- Rope is interleaved-pair in the transformer and NEoX in the encoder, the h and w ids
  are centred and NEGATIVE, and the angles are f32.
- The modulation chunk order is `[scale1, gate1, scale2, gate2]`, and `gate_layer` is the
  SiLU'd half of the FFN.
- The VAE's norm is an L2 norm over channels, five of its nine shortcut stages are not
  pools or upsamples, and it denormalises the latent inside `decode`, exactly once.
- The VAE file is F32 and 1.35 GB. fp16 is disqualified for the transformer as it is for
  Z-Image, the reference clipping at 65504.
- The noise is not scaled, and `mu` comes from the target's token count alone.
- A reference image's latent rows sit INSIDE the text stream.

## The gates

Both are `#[ignore]`d and both FAIL, not skip, when a fixture or the weights are missing,
naming the command that supplies them. `scripts/qwen-image-ref-dump.py` writes the
references: the second Python file in the repo, on the same terms as
`scripts/zimage-ref-dump.py`, run by hand in a throwaway venv and never in CI.

```
uv venv /tmp/qwen-image-venv --python 3.12
uv pip install --python /tmp/qwen-image-venv/bin/python torch 'transformers>=5.17' \
  safetensors numpy accelerate pillow torchvision \
  'diffusers @ git+https://github.com/huggingface/diffusers@6256aa7666cedd47443adc8f82da9a10e110b09c'
P=/tmp/qwen-image-venv/bin/python
$P scripts/qwen-image-ref-dump.py --stage tokens
$P scripts/qwen-image-ref-dump.py --stage fp32 && $P scripts/qwen-image-ref-dump.py --stage bf16 \
  && $P scripts/qwen-image-ref-dump.py --stage finalize
$P scripts/qwen-image-ref-dump.py --stage transformer --dtype fp32 \
  && $P scripts/qwen-image-ref-dump.py --stage transformer --dtype bf16
```

Measured with torch 2.14.0, transformers 5.17.0 and diffusers 0.41.0.dev0. `tokens` needs
no weights. The encoder stages run on the CPU (fp32 eager about a minute, bf16 about four
and a half), the transformer stages on mps (165 s fp32, 48 s bf16 at 512x512), the
encoder freed before the transformer loads. Every weight stage asserts three things: its
own path equals the pipeline's method bit for bit on one prompt, the text position ids
are equal on all three MRoPE axes, and the normed state differs from the pre-norm one.

**The encoder gate**, `tests/qwen_image_encoder.rs`:

```
XWEN_QWEN_IMAGE_REF_DIR=/tmp/qwen-image-ref \
  cargo test --release --test qwen_image_encoder -- --ignored --nocapture
```

`tests/fixtures/qwen-image-encoder/` commits `prompts.json` (12 prompts), `tokens.json`
(rendered strings, ids, the drop index) and `reference.json` (versions, checks, and the
reference's own spread); the arrays, about 50 MB, stay in the out dir. String, ids and
drop are compared first. Every kept row must hold cosine 0.9999. Rows past kept row 0
must hold max relative error 0.03 and 99% of them 0.01; kept row 0 has its own bar, 0.05,
because it carries a massive activation inside the stack. A non-ignored test holds every
constant between the reference's bf16 spread and the normed wrong graph, read from
`reference.json`. xwen reads minimum cosine 0.99997 and 99.67% of rows inside 0.01.
`XWEN_QWEN_IMAGE_ONLY` selects prompts and `XWEN_QWEN_IMAGE_DUMP_DIR` writes xwen's rows.

**The parity gate**, `tests/qwen_image_parity.rs`, reading the snapshot root from
`XWEN_QWEN_IMAGE_DIR` when it is not in the hub cache:

```
cargo test --release --test qwen_image_parity -- --ignored --nocapture
```

`tests/fixtures/qwen-image-transformer/512x512-p1-s0/`, 1.8 MB: `latents0` (the noise,
unpacked), `cap_feats` (the kept pre-norm rows the pipeline fed the transformer, rounded
once to bf16), `velocity0-fp32`, `latents-final-fp32`, `image-fp32.png` (RGBA) and
`meta.json` with the sigma grid, `mu`, and the reference's bf16-against-fp32 spread. The
gate holds the sigma grid and `mu` to 1e-6, the step-0 velocity to cosine 0.998 and mean
relative error 0.04 with both wrong graphs asserted OUTSIDE, the run's first velocity
equal to the standalone forward, and the VAE alone to 60 dB, RGB against RGB. It reports
the final latent and the image PSNR after 40 steps. As of 5ea0d43: velocity cosine
1.000000 and mean relative error 0.0006 (the reference's bf16 arm 0.999948 and 0.0075),
brackets at 0.9825 and 0.9874, VAE alone 91.07 dB, final latent 0.999999, image 57.30 dB
where the reference's bf16 arm gets 35.58.

`tests/qwen_image_image.rs` is the ignored end-to-end render, 512x512 at 8 steps.

## The bisect arms

| switch | arms |
| --- | --- |
| `XWEN_QWEN_IMAGE_LINEAR` | `xwen` (default, the tensor gemm) / `candle` |
| `XWEN_QWEN_IMAGE_ATTN` | `tensor` (default, segments on the Metal-4 kernel) / `basic` (dense f32 under the explicit mask) |
| `XWEN_QWEN_IMAGE_VAE` | `candle` (the only arm) |
| `XWEN_QWEN_IMAGE_CACHE` | `on` (default, the prefix K/V kept) / `off` (the full sequence every step) |

A typo in any of them is a load error. The candle linear arm runs bf16 activations and
differs from the tensor arm by about 2.8% of scale on random tiny weights, which is the
linear arm's rounding and not the attention's.
